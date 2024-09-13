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
//"../tr1_hash_map.h"中有如下定义：
//    #include <unordered_map>                  无序映射
//    #define tr1_hash_map std::unordered_map   std::unordered_map 重命名为 tr1_hash_map
//    #define tr1_hash_map_ismap 0              设置 tr1_hash_map_ismap = 0
#define mem_map tr1_hash_map                  //tr1_hash_map 重命名为 mem_map
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
内存页（由类模板 mem_storage 实现）。它使用STL无序Map（如果无序Map不可用，则恢复为STL Map）来将页
与它们相应的地址联系起来。每个 mem_storage 对象是一个具有读写功能的字节数组。
*/
template <unsigned BSIZE>
class mem_storage {
 public:
  //mem_storage的构造函数，从另一个mem_storage对象another复制内存页。m_data是构造的内存页中的全部
  //数据，这里calloc的函数原型和功能是：
  //    void *calloc (size_t __nmemb, size_t __size)
  //    分配存储的单个元素大小为SIZE字节，总共分配NMEMB个这种元素，并全部初始化为0。
  //memcpy的函数原型和功能是：
  //    void *memcpy (void *__restrict __dest, const void *__restrict __src, size_t __n)
  //    从__src地址拷贝到__dst地址，共__n字节的数据。
  mem_storage(const mem_storage &another) {
    //分配1个大小为BSIZE大小的内存页，并全部初始化为0。
    m_data = (unsigned char *)calloc(1, BSIZE);
    //把另一个mem_storage对象another的数据，复制到当前对象的m_data中，复制大小为BSIZE字节数。
    memcpy(m_data, another.m_data, BSIZE);
  }
  //mem_storage的构造函数，直接为当前对象的m_data分配BSIZE字节的存储。
  mem_storage() { m_data = (unsigned char *)calloc(1, BSIZE); }
  //析构函数，释放前对象的m_data。
  ~mem_storage() { free(m_data); }

  //写存储，参数分别为：
  //    1. unsigned offset：写地址范围的起始地址相对m_data的偏移量
  //    2. size_t length：写的内容的长度，以字节为单位
  //    3. const unsigned char *data：写的数据内容
  void write(unsigned offset, size_t length, const unsigned char *data) {
    //由于当前对象的m_data总共BSIZE字节，写地址范围不能越界。
    assert(offset + length <= BSIZE);
    //写内容。
    memcpy(m_data + offset, data, length);
  }

  //读存储，参数分别为：
  //    1. unsigned offset：读地址范围的起始地址相对m_data的偏移量
  //    2. size_t length：读的内容的长度，以字节为单位
  //    3. const unsigned char *data：读的数据内容
  void read(unsigned offset, size_t length, unsigned char *data) const {
    //由于当前对象的m_data总共BSIZE字节，读地址范围不能越界。
    assert(offset + length <= BSIZE);
    //读内容。
    memcpy(data, m_data + offset, length);
  }

  //打印存储中的内容。
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
  //无效变量，没用到。
  unsigned m_nbytes;
  //当前mem_storage类的对象的数据的指针，指向第一字节的数据。
  unsigned char *m_data;
};

class ptx_thread_info;
class ptx_instruction;

/*
memory_space是用于实现功能模拟状态的内存存储的抽象基类。在函数仿真中使用的动态数据值的存储使用了不同
的寄存器和内存空间类。寄存器的值包含在 ptx_thread_info::m_regs 中，这是一个从符号指针到C联合类型 
ptx_reg_t 的映射。寄存器的访问使用方法 ptx_thread_info::get_operand_value()，它使用 operand_info 
作为输入。对于内存操作数，该方法返回内存操作数的有效地址。编程模型中的每个内存空间都包含在一个类型为 
memory_space 的对象中。GPU中所有线程可见的内存空间都包含在 gpgpu_t 中，并通过 ptx_thread_info 中的
接口进行访问（例如，ptx_thread_info::get_global_memory）。

memory_space 作为基类，派生出 memory_space_impl，下面函数的功能详见 memory_space_impl 类的注释。 
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
memory_space 为基类，memory_space_impl 为派生出的类，后者以公有的方法继承前者。memory_space_impl 
类实现了由抽象类 memory_space 定义的读写接口。
*/
template <unsigned BSIZE>
class memory_space_impl : public memory_space {
 public:
  //构造函数。
  memory_space_impl(std::string name, unsigned hash_size);
  //在存储里写入内容，涉及多页存储等问题。
  virtual void write(mem_addr_t addr, size_t length, const void *data,
                     ptx_thread_info *thd, const ptx_instruction *pI);
  //简单地在存储里写入内容，不涉及多页存储等问题。
  virtual void write_only(mem_addr_t index, mem_addr_t offset, size_t length,
                          const void *data);
  //读可能跨内存页的数据。
  virtual void read(mem_addr_t addr, size_t length, void *data) const;
  //打印存储中的数据。一般DEBUG用，用到再补充。
  virtual void print(const char *format, FILE *fout) const;
  //DEBUG用，用到再补充。
  virtual void set_watch(addr_t addr, unsigned watchpoint);

 private:
  //读单个内存页的数据，详见 memory.cc。
  void read_single_block(mem_addr_t blk_idx, mem_addr_t addr, size_t length,
                         void *data) const;
  //m_name为这块存储的字符串名字，在构造函数中赋值。
  std::string m_name;
  //m_log2_block_size=Log2(BSIZE)，这里计算 Log2(BSIZE)，且这里的BSIZE一般应为 2 的倍数。在构造函
  //数中赋值。
  unsigned m_log2_block_size;
  typedef mem_map<mem_addr_t, mem_storage<BSIZE> > map_t;
  //在 memory_space_impl 对象中的 m_data 与 mem_storage 对象不同，前者是作为一个 std::unordered_map，
  //其 key-value 对分别为：
  //    key: mem_addr_t 类型的 blk_idx（内存页编号）；
  //    value: mem_storage<BSIZE> 内存页。
  map_t m_data;
  //观察点，DEBUG用，后面用到再补充。
  std::map<unsigned, mem_addr_t> m_watchpoints;
};

#endif
