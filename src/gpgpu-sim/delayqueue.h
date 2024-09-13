// Copyright (c) 2009-2011, Wilson W.L. Fung, Tor M. Aamodt
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The University of British Columbia nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef DELAYQUEUE_H
#define DELAYQUEUE_H

#include "../statwrapper.h"
#include "gpu-misc.h"

template <class T>
struct fifo_data {
  T* m_data;
  fifo_data* m_next;
};

/*
从内存分区微架构模型部分描述的子组件来看，data_cache类型的成员对象为二级缓存建模，dram_t类型为片外
DRAM Channel，各种队列是使用fifo_pipeline类来建模的。
*/
template <class T>
class fifo_pipeline {
 public:
  //构造函数。三个参数分别为：
  //    nm：队列的name，字符串。
  //    minlen：队列的最小长度。
  //    maxlen：队列的最大长度。
  fifo_pipeline(const char* nm, unsigned int minlen, unsigned int maxlen) {
    assert(maxlen);
    m_name = nm;
    m_min_len = minlen;
    m_max_len = maxlen;
    //队列的当前长度。
    m_length = 0;
    //队列中的元素个数。
    m_n_element = 0;
    m_head = NULL;
    m_tail = NULL;
    for (unsigned i = 0; i < m_min_len; i++) push(NULL);
  }
  //析构函数。
  ~fifo_pipeline() {
    while (m_head) {
      m_tail = m_head;
      m_head = m_head->m_next;
      delete m_tail;
    }
  }
  //向队列中压栈，压入数据data。data存在m_tail->next指向的位置。
  void push(T* data) {
    assert(m_length < m_max_len);
    //如果m_head不为空，则说明当前状态下，队列中已经有元素。
    if (m_head) {
      if (m_tail->m_data || m_length < m_min_len) {
        m_tail->m_next = new fifo_data<T>();
        m_tail = m_tail->m_next;
        m_length++;
        m_n_element++;
      }
    } else {
      m_head = m_tail = new fifo_data<T>();
      m_length++;
      m_n_element++;
    }
    m_tail->m_next = NULL;
    m_tail->m_data = data;
  }
  //弹出m_head指向的元素。
  T* pop() {
    fifo_data<T>* next;
    T* data;
    if (m_head) {
      next = m_head->m_next;
      data = m_head->m_data;
      if (m_head == m_tail) {
        assert(next == NULL);
        m_tail = NULL;
      }
      delete m_head;
      m_head = next;
      m_length--;
      if (m_length == 0) {
        assert(m_head == NULL);
        m_tail = m_head;
      }
      m_n_element--;
      if (m_min_len && m_length < m_min_len) {
        push(NULL);
        m_n_element--;  // uncount NULL elements inserted to create delays
      }
    } else {
      data = NULL;
    }
    return data;
  }
  //仅返回（不弹出）m_head指向的元素。
  T* top() const {
    if (m_head) {
      return m_head->m_data;
    } else {
      return NULL;
    }
  }

  void set_min_length(unsigned int new_min_len) {
    if (new_min_len == m_min_len) return;

    if (new_min_len > m_min_len) {
      m_min_len = new_min_len;
      while (m_length < m_min_len) {
        push(NULL);
        //push里每压入一个元素就会m_n_element++，因此后面应该把压入NULL增加的减去。
        m_n_element--;  // uncount NULL elements inserted to create delays
      }
    } else {
      // in this branch imply that the original min_len is larger then 0
      // ie. head != 0
      assert(m_head);
      m_min_len = new_min_len;
      while ((m_length > m_min_len) && (m_tail->m_data == 0)) {
        fifo_data<T>* iter;
        iter = m_head;
        while (iter && (iter->m_next != m_tail)) iter = iter->m_next;
        if (!iter) {
          // there is only one node, and that node is empty
          assert(m_head->m_data == 0);
          pop();
        } else {
          // there are more than one node, and tail node is empty
          assert(iter->m_next == m_tail);
          delete m_tail;
          m_tail = iter;
          m_tail->m_next = 0;
          m_length--;
        }
      }
    }
  }
  //返回是否队列已满。
  bool full() const { return (m_max_len && m_length >= m_max_len); }
  //size数量的数据放入队列中，可以放下返回False，放不下返回True。
  bool is_avilable_size(unsigned size) const {
    return (m_max_len && m_length + size - 1 >= m_max_len);
  }
  //返回是否队列为空。
  bool empty() const { return m_head == NULL; }
  //返回队列中的元素个数。
  unsigned get_n_element() const { return m_n_element; }
  //返回队列的长度。
  unsigned get_length() const { return m_length; }
  //返回队列的长度。
  unsigned get_max_len() const { return m_max_len; }

  void print() const {
    fifo_data<T>* ddp = m_head;
    printf("%s(%d): ", m_name, m_length);
    while (ddp) {
      printf("%p ", ddp->m_data);
      ddp = ddp->m_next;
    }
    printf("\n");
  }

 private:
  const char* m_name;

  unsigned int m_min_len;
  unsigned int m_max_len;
  unsigned int m_length;
  unsigned int m_n_element;

  fifo_data<T>* m_head;
  fifo_data<T>* m_tail;
};

#endif
