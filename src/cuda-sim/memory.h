// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung
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

#ifndef memory_h_INCLUDED
#define memory_h_INCLUDED

#include "../abstract_hardware_model.h"

#include "../tr1_hash_map.h"
//"../tr1_hash_map.h"�������¶��壺
//    #include <unordered_map>                  ����ӳ��
//    #define tr1_hash_map std::unordered_map   std::unordered_map ������Ϊ tr1_hash_map
//    #define tr1_hash_map_ismap 0              ���� tr1_hash_map_ismap = 0
#define mem_map tr1_hash_map                  //tr1_hash_map ������Ϊ mem_map
#if tr1_hash_map_ismap == 1
#define MEM_MAP_RESIZE(hash_size)
#else
#define MEM_MAP_RESIZE(hash_size) (m_data.rehash(hash_size))
#endif

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <map>
#include <string>

typedef address_type mem_addr_t;

#define MEM_BLOCK_SIZE (4 * 1024)

/*
�ڴ�ҳ������ģ�� mem_storage ʵ�֣�����ʹ��STL����Map���������Map�����ã���ָ�ΪSTL Map������ҳ
��������Ӧ�ĵ�ַ��ϵ������ÿ�� mem_storage ������һ�����ж�д���ܵ��ֽ����顣
*/
template <unsigned BSIZE>
class mem_storage {
 public:
  //mem_storage�Ĺ��캯��������һ��mem_storage����another�����ڴ�ҳ��m_data�ǹ�����ڴ�ҳ�е�ȫ��
  //���ݣ�����calloc�ĺ���ԭ�ͺ͹����ǣ�
  //    void *calloc (size_t __nmemb, size_t __size)
  //    ����洢�ĵ���Ԫ�ش�СΪSIZE�ֽڣ��ܹ�����NMEMB������Ԫ�أ���ȫ����ʼ��Ϊ0��
  //memcpy�ĺ���ԭ�ͺ͹����ǣ�
  //    void *memcpy (void *__restrict __dest, const void *__restrict __src, size_t __n)
  //    ��__src��ַ������__dst��ַ����__n�ֽڵ����ݡ�
  mem_storage(const mem_storage &another) {
    //����1����СΪBSIZE��С���ڴ�ҳ����ȫ����ʼ��Ϊ0��
    m_data = (unsigned char *)calloc(1, BSIZE);
    //����һ��mem_storage����another�����ݣ����Ƶ���ǰ�����m_data�У����ƴ�СΪBSIZE�ֽ�����
    memcpy(m_data, another.m_data, BSIZE);
  }
  //mem_storage�Ĺ��캯����ֱ��Ϊ��ǰ�����m_data����BSIZE�ֽڵĴ洢��
  mem_storage() { m_data = (unsigned char *)calloc(1, BSIZE); }
  //�����������ͷ�ǰ�����m_data��
  ~mem_storage() { free(m_data); }

  //д�洢�������ֱ�Ϊ��
  //    1. unsigned offset��д��ַ��Χ����ʼ��ַ���m_data��ƫ����
  //    2. size_t length��д�����ݵĳ��ȣ����ֽ�Ϊ��λ
  //    3. const unsigned char *data��д����������
  void write(unsigned offset, size_t length, const unsigned char *data) {
    //���ڵ�ǰ�����m_data�ܹ�BSIZE�ֽڣ�д��ַ��Χ����Խ�硣
    assert(offset + length <= BSIZE);
    //д���ݡ�
    memcpy(m_data + offset, data, length);
  }

  //���洢�������ֱ�Ϊ��
  //    1. unsigned offset������ַ��Χ����ʼ��ַ���m_data��ƫ����
  //    2. size_t length���������ݵĳ��ȣ����ֽ�Ϊ��λ
  //    3. const unsigned char *data��������������
  void read(unsigned offset, size_t length, unsigned char *data) const {
    //���ڵ�ǰ�����m_data�ܹ�BSIZE�ֽڣ�����ַ��Χ����Խ�硣
    assert(offset + length <= BSIZE);
    //�����ݡ�
    memcpy(data, m_data + offset, length);
  }

  //��ӡ�洢�е����ݡ�
  void print(const char *format, FILE *fout) const {
    unsigned int *i_data = (unsigned int *)m_data;
    for (int d = 0; d < (BSIZE / sizeof(unsigned int)); d++) {
      if (d % 1 == 0) {
        fprintf(fout, "\n");
      }
      fprintf(fout, format, i_data[d]);
      fprintf(fout, " ");
    }
    fprintf(fout, "\n");
    fflush(fout);
  }

 private:
  //��Ч������û�õ���
  unsigned m_nbytes;
  //��ǰmem_storage��Ķ�������ݵ�ָ�룬ָ���һ�ֽڵ����ݡ�
  unsigned char *m_data;
};

class ptx_thread_info;
class ptx_instruction;

/*
memory_space������ʵ�ֹ���ģ��״̬���ڴ�洢�ĳ�����ࡣ�ں���������ʹ�õĶ�̬����ֵ�Ĵ洢ʹ���˲�ͬ
�ļĴ������ڴ�ռ��ࡣ�Ĵ�����ֵ������ ptx_thread_info::m_regs �У�����һ���ӷ���ָ�뵽C�������� 
ptx_reg_t ��ӳ�䡣�Ĵ����ķ���ʹ�÷��� ptx_thread_info::get_operand_value()����ʹ�� operand_info 
��Ϊ���롣�����ڴ���������÷��������ڴ����������Ч��ַ�����ģ���е�ÿ���ڴ�ռ䶼������һ������Ϊ 
memory_space �Ķ����С�GPU�������߳̿ɼ����ڴ�ռ䶼������ gpgpu_t �У���ͨ�� ptx_thread_info �е�
�ӿڽ��з��ʣ����磬ptx_thread_info::get_global_memory����

memory_space ��Ϊ���࣬������ memory_space_impl�����溯���Ĺ������ memory_space_impl ���ע�͡� 
*/
class memory_space {
 public:
  virtual ~memory_space() {}
  virtual void write(mem_addr_t addr, size_t length, const void *data,
                     ptx_thread_info *thd, const ptx_instruction *pI) = 0;
  virtual void write_only(mem_addr_t index, mem_addr_t offset, size_t length,
                          const void *data) = 0;
  virtual void read(mem_addr_t addr, size_t length, void *data) const = 0;
  virtual void print(const char *format, FILE *fout) const = 0;
  virtual void set_watch(addr_t addr, unsigned watchpoint) = 0;
};

/*
memory_space Ϊ���࣬memory_space_impl Ϊ���������࣬�����Թ��еķ����̳�ǰ�ߡ�memory_space_impl 
��ʵ�����ɳ����� memory_space ����Ķ�д�ӿڡ�
*/
template <unsigned BSIZE>
class memory_space_impl : public memory_space {
 public:
  //���캯����
  memory_space_impl(std::string name, unsigned hash_size);
  //�ڴ洢��д�����ݣ��漰��ҳ�洢�����⡣
  virtual void write(mem_addr_t addr, size_t length, const void *data,
                     ptx_thread_info *thd, const ptx_instruction *pI);
  //�򵥵��ڴ洢��д�����ݣ����漰��ҳ�洢�����⡣
  virtual void write_only(mem_addr_t index, mem_addr_t offset, size_t length,
                          const void *data);
  //�����ܿ��ڴ�ҳ�����ݡ�
  virtual void read(mem_addr_t addr, size_t length, void *data) const;
  //��ӡ�洢�е����ݡ�һ��DEBUG�ã��õ��ٲ��䡣
  virtual void print(const char *format, FILE *fout) const;
  //DEBUG�ã��õ��ٲ��䡣
  virtual void set_watch(addr_t addr, unsigned watchpoint);

 private:
  //�������ڴ�ҳ�����ݣ���� memory.cc��
  void read_single_block(mem_addr_t blk_idx, mem_addr_t addr, size_t length,
                         void *data) const;
  //m_nameΪ���洢���ַ������֣��ڹ��캯���и�ֵ��
  std::string m_name;
  //m_log2_block_size=Log2(BSIZE)��������� Log2(BSIZE)���������BSIZEһ��ӦΪ 2 �ı������ڹ��캯
  //���и�ֵ��
  unsigned m_log2_block_size;
  typedef mem_map<mem_addr_t, mem_storage<BSIZE> > map_t;
  //�� memory_space_impl �����е� m_data �� mem_storage ����ͬ��ǰ������Ϊһ�� std::unordered_map��
  //�� key-value �Էֱ�Ϊ��
  //    key: mem_addr_t ���͵� blk_idx���ڴ�ҳ��ţ���
  //    value: mem_storage<BSIZE> �ڴ�ҳ��
  map_t m_data;
  //�۲�㣬DEBUG�ã������õ��ٲ��䡣
  std::map<unsigned, mem_addr_t> m_watchpoints;
};

#endif
