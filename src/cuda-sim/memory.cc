// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung,
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

#include "memory.h"
#include <stdlib.h>
#include "../../libcuda/gpgpu_context.h"
#include "../debug.h"

/*
为了优化功能模拟的性能，内存是用哈希表实现的。哈希表的块大小是模板类 memory_space_impl 的模板参数。
memory_space_impl 类实现了由抽象类 memory_space 定义的读写接口。在内部，每个 memory_space_impl 
对象包含一组内存页（由类模板 mem_storage 实现）。它使用STL无序Map（如果无序Map不可用，则恢复为STL 
Map）来将页与它们相应的地址联系起来。每个 mem_storage 对象是一个具有读写功能的字节数组。最初，每个 
memory_space 对象是空的，当访问内存空间中单个页面对应的地址时（通过 LD/ST 指令或 cudaMemcpy()），
页面被按需分配。
*/
template <unsigned BSIZE>
memory_space_impl<BSIZE>::memory_space_impl(std::string name,
                                            unsigned hash_size) {
  //m_name为这块存储的字符串名字，例如cuda-sim.cc中创建shared memory的时候就给出了该块共享存储的
  //名字：
  //    char buf[512];
  //    snprintf(buf, 512, "shared_%u", sid); <================== m_name
  //    shared_mem = new memory_space_impl<16 * 1024>(buf, 4);
  m_name = name;
  //MEM_MAP_RESIZE()在memory.h中定义：
  //    #define MEM_MAP_RESIZE(hash_size) (m_data.rehash(hash_size))
  MEM_MAP_RESIZE(hash_size);

  //m_log2_block_size=Log2(BSIZE)，这里计算 Log2(BSIZE)，且这里的BSIZE一般应为 2 的倍数。
  m_log2_block_size = -1;
  for (unsigned n = 0, mask = 1; mask != 0; mask <<= 1, n++) {
    if (BSIZE & mask) {
      assert(m_log2_block_size == (unsigned)-1);
      m_log2_block_size = n;
    }
  }
  assert(m_log2_block_size != (unsigned)-1);
}

/*
简单地在存储里写入内容，不涉及多页存储等问题。四个参数分别为：
1. mem_addr_t offset：写地址范围的起始地址相对m_data的偏移量。
2. mem_addr_t index：mem_storage<BSIZE> 对象（内存页）的索引。
3. size_t length：写的内容的长度，以字节为单位。
4. const unsigned char *data：写的数据内容。
*/
template <unsigned BSIZE>
void memory_space_impl<BSIZE>::write_only(mem_addr_t offset, mem_addr_t index,
                                          size_t length, const void *data) {
  //第 index 号内存页写入数据。m_data[index] 是一个mem_storage<BSIZE> 对象（内存页），
  //调用它的成员函数来实现对其内存页的写入数据。
  m_data[index].write(offset, length, (const unsigned char *)data);
}

/*
在存储里写入内容，涉及多页存储等问题。四个参数分别为：
1. mem_addr_t addr：写地址范围的起始地址。
2. size_t length：写的内容的长度，以字节为单位。
3. const unsigned char *data：写的数据内容。
4. class ptx_thread_info *thd：DEBUG用，后面用到再补充。
5. const ptx_instruction *pI：DEBUG用，后面用到再补充。
*/
template <unsigned BSIZE>
void memory_space_impl<BSIZE>::write(mem_addr_t addr, size_t length,
                                     const void *data,
                                     class ptx_thread_info *thd,
                                     const ptx_instruction *pI) {
  //第 index 号内存页写入数据。m_data[index] 是一个mem_storage<BSIZE> 对象（内存页），调用它的
  //成员函数来实现对其内存页的写入数据。addr的编址是从第 0 个内存页开始的。这里来看是：
  //    当 addr∈[0, 2^m_log2_block_size)时，index=0；
  //    当 addr∈[2^m_log2_block_size, 2^(m_log2_block_sizes+1))时，index=1；
  //    ......以此类推。
  //即，每个内存页只存储 BSIZE 字节大小的数据，超过就要换内存页。
  mem_addr_t index = addr >> m_log2_block_size;
  //printf("addr:%x, m_log2_block_size=%d, index=%x, BSIZE=%d\n", 
  //        addr, m_log2_block_size, index,BSIZE);

  //判断写数据的长度是否超过当前内存页。
  if ((addr + length) <= (index + 1) * BSIZE) {
    // fast route for intra-block access
    //如果写数据的长度没有超过当前内存页，就可以执行块内的快速访问。
    //offset指的是写地址范围的起始地址相对当前内存页的起始地址的偏移量。例如：
    //    有一个存储器，它的每个内存页的大小为 BSIZE=16字节，则：
    //        addr为  0~15 时，处于第 0 号内存页；
    //        addr为 16~31 时，处于第 1 号内存页；
    //    则，如果 addr=17，
    //        addr & BSIZE = 'b10001 & ('b10000-'b1) = 'b10001 & 'b01111 = 'b1
    //    即addr=17相对当前内存页的起始地址的偏移量为 1。
    unsigned offset = addr & (BSIZE - 1);
    //写数据的长度。
    unsigned nbytes = length;
    //第 index 号内存页写入数据。m_data[index] 是一个mem_storage<BSIZE> 对象（内存页），
    //调用它的成员函数来实现对其内存页的写入数据。
    m_data[index].write(offset, nbytes, (const unsigned char *)data);
  } else {
    // slow route for inter-block access
    //如果写数据的长度超过了当前内存页，就可以执行块间的（跨内存页）的慢速访问。
    //临时变量保存住[写数据的长度]/[相对当前内存页的起始地址的偏移量]/[写数据的全局地址]，后
    //面根据在什么位置跨内存页再调整。nbytes_remain 即为还剩余的需要写的数据长度，初始时设置
    //为完整的写数据长度 length。current_addr 为当前写入的全局起始地址，在换页后，需要变为换
    //页后的写入的全局起始地址。src_offset 是指当前页需要写入的源端数据的偏移地址，例如，第一
    //页写入时，该偏移量为0，假设写入长度为 _length_；换页后的需要写入的源端数据的偏移地址变为
    //0+_length_=_length_。
    unsigned nbytes_remain = length;
    unsigned src_offset = 0;
    mem_addr_t current_addr = addr;

    //由于写入过程存在换页，且不知道究竟会换多少页才能把数据写完，因此这里对[还剩余的需要写的数
    //据长度]循环，直到nbytes_remain变为0，才说明已经把所有数据全写完成了。
    while (nbytes_remain > 0) {
      //计算current_addr相对当前内存页的起始地址的偏移量。
      unsigned offset = current_addr & (BSIZE - 1);
      //计算当前起始地址所在的被写入数据的内存页的 page 号。
      mem_addr_t page = current_addr >> m_log2_block_size;
      //access_limit = current_addr相对当前内存页的起始地址的偏移量 + 写入长度。
      mem_addr_t access_limit = offset + nbytes_remain;
      //如果access_limit超过页大小 BSIZE，则需要换页。
      if (access_limit > BSIZE) {
        access_limit = BSIZE;
      }
      //换页前的写入长度为：BSIZE - offset。
      size_t tx_bytes = access_limit - offset;
      //第 page 号内存页写入数据。m_data[page] 是一个mem_storage<BSIZE> 对象（内存页），
      //调用它的成员函数来实现对其内存页的写入数据。
      m_data[page].write(offset, tx_bytes,
                         &((const unsigned char *)data)[src_offset]);

      // advance pointers
      //前进指针。用于指导换页后的数据写入。
      //换页后的需要写入的源端数据的偏移地址变为 src_offset+tx_bytes。
      src_offset += tx_bytes;
      //换页后的写入的全局起始地址为 current_addr+tx_bytes。
      current_addr += tx_bytes;
      //当前页已经写入 tx_bytes 字节长度数据，剩余需写入数据长度为 nbytes_remain-tx_bytes。
      nbytes_remain -= tx_bytes;
    }
    assert(nbytes_remain == 0);
  }
  
  //DEBUG用，后面用到再补充。
  if (!m_watchpoints.empty()) {
    std::map<unsigned, mem_addr_t>::iterator i;
    for (i = m_watchpoints.begin(); i != m_watchpoints.end(); i++) {
      mem_addr_t wa = i->second;
      if (((addr <= wa) && ((addr + length) > wa)) ||
          ((addr > wa) && (addr < (wa + 4))))
        thd->get_gpu()->gpgpu_ctx->the_gpgpusim->g_the_gpu->hit_watchpoint(
            i->first, thd, pI);
    }
  }
}

/*
读单个内存页的数据。四个参数分别为：
1. mem_addr_t blk_idx：读的数据所在的内存页编号。
2. mem_addr_t addr：读地址范围的起始地址。
3. size_t length：读的内容的长度，以字节为单位。
4. void *data：读到的数据放到 data 中。
*/
template <unsigned BSIZE>
void memory_space_impl<BSIZE>::read_single_block(mem_addr_t blk_idx,
                                                 mem_addr_t addr, size_t length,
                                                 void *data) const {
  //看一个例子：
  //    有一个存储器，它的每个内存页的大小为 BSIZE=16字节，则：
  //        addr为  0~15 时，处于第 0 号内存页；
  //        addr为 16~31 时，处于第 1 号内存页；
  //        addr为 32~47 时，处于第 3 号内存页；则：
  //    1. 如果 读地址 addr=17，读长度为 10，读的内存页号为 1：
  //       读范围的终止地址=(addr + length)=27
  //       读的内存页的最末尾地址=(blk_idx + 1) * BSIZE=32
  //       27 <= 32，未跨页，合法。
  //    2. 如果 读地址 addr=28，读长度为 10，读的内存页号为 1：
  //       读范围的终止地址=(addr + length)=38
  //       读的内存页的最末尾地址=(blk_idx + 1) * BSIZE=32
  //       38 > 32，跨页，非法。
  //下面的if判断即为判断读内存页是否在[读单个内存页数据]函数中合法。
  if ((addr + length) > (blk_idx + 1) * BSIZE) {
    printf(
        "GPGPU-Sim PTX: ERROR * access to memory \'%s\' is unaligned : "
        "addr=0x%llx, length=%zu\n",
        m_name.c_str(), addr, length);
    printf(
        "GPGPU-Sim PTX: (addr+length)=0x%llx > 0x%llx=(index+1)*BSIZE, "
        "index=0x%llx, BSIZE=0x%x\n",
        (addr + length), (blk_idx + 1) * BSIZE, blk_idx, BSIZE);
    throw 1;
  }
  //在 memory_space_impl 对象中的 m_data 与 mem_storage 对象不同，memory_space_impl 对象中
  //的 m_data 是作为一个 unordered_map，其 key-value 对分别为：
  //    key: mem_addr_t 类型的 blk_idx（内存页编号）；
  //    value: mem_storage<BSIZE> 内存页。
  //函数 unordered_map.find(key) 的功能：
  //    参数：它以键（key）作为参数。
  //    返回值：如果给定的键存在于unordered_map中，则它向该元素返回一个迭代器，否则返回映射迭
  //           代器的末尾。
  typename map_t::const_iterator i = m_data.find(blk_idx);
  //如果 i == m_data.end()，说明 m_data 不存在 blk_idx 标识的内存页。
  if (i == m_data.end()) {
    //m_data 不存在 blk_idx 标识的内存页，将 data 全部置零。
    for (size_t n = 0; n < length; n++)
      ((unsigned char *)data)[n] = (unsigned char)0;
    // printf("GPGPU-Sim PTX:  WARNING reading %zu bytes from unititialized
    // memory at address 0x%x in space %s\n", length, addr, m_name.c_str() );
  } else {
    //如果 i != m_data.end()，m_data 存在 blk_idx 标识的内存页，i是指向该内存页的迭代器。
    //计算 addr 相对当前内存页的起始地址的偏移量。
    unsigned offset = addr & (BSIZE - 1);
    unsigned nbytes = length;
    //读数据，读到的数据放入 data。
    i->second.read(offset, nbytes, (unsigned char *)data);
  }
}

/*
读可能跨内存页的数据。四个参数分别为：
1. mem_addr_t addr：读地址范围的起始地址。
2. size_t length：读的内容的长度，以字节为单位。
3. void *data：读到的数据放到 data 中。
*/
template <unsigned BSIZE>
void memory_space_impl<BSIZE>::read(mem_addr_t addr, size_t length,
                                    void *data) const {
  //计算当前起始地址所在的被读出数据的内存页的 index 号。
  mem_addr_t index = addr >> m_log2_block_size;
  //看一个例子：
  //    有一个存储器，它的每个内存页的大小为 BSIZE=16字节，则：
  //        addr为  0~15 时，处于第 0 号内存页；
  //        addr为 16~31 时，处于第 1 号内存页；
  //        addr为 32~47 时，处于第 3 号内存页；则：
  //    1. 如果 读地址 addr=17，读长度为 10，读的内存页号为 1：
  //       读范围的终止地址=(addr + length)=27
  //       读的内存页的最末尾地址=(blk_idx + 1) * BSIZE=32
  //       27 <= 32，未跨页，合法。
  //    2. 如果 读地址 addr=28，读长度为 10，读的内存页号为 1：
  //       读范围的终止地址=(addr + length)=38
  //       读的内存页的最末尾地址=(blk_idx + 1) * BSIZE=32
  //       38 > 32，跨页，非法。
  //下面的if判断即为判断读内存页是否跨页读。
  if ((addr + length) <= (index + 1) * BSIZE) {
    // fast route for intra-block access
    //不跨页读的话，就简单地执行单页内读数据即可，执行块内的快速访问。
    read_single_block(index, addr, length, data);
  } else {
    // slow route for inter-block access
    //跨页读的话，就需要多次读不同页的数据，执行块间的（跨内存页）的慢速访问。
    //nbytes_remain 即为还剩余的需要读的数据长度，初始时设置为完整的读数据长度 length。
    //dst_offset 是指当前页需要读出数据存入的目的端数据的偏移地址，例如，第一页读出保存到 data 
    //时，该偏移量为0，假设读出长度保存到 data 中的长度为 _length_；换页后的需要再次读出，保存
    //到目的端数据的偏移地址变为0+_length_=_length_。current_addr 为当前读出的全局起始地址，
    //在换页后，需要变为换页后读出的全局起始地址。
    unsigned nbytes_remain = length;
    unsigned dst_offset = 0;
    mem_addr_t current_addr = addr;

    //由于读出过程存在换页，且不知道究竟会换多少页才能把数据读完，因此这里对[还剩余的需要读的数
    //据长度]循环，直到nbytes_remain变为0，才说明已经把所有数据全读出完成。
    while (nbytes_remain > 0) {
      //计算current_addr相对当前内存页的起始地址的偏移量。
      unsigned offset = current_addr & (BSIZE - 1);
      //计算当前起始地址所在的被读数据的内存页的 page 号。
      mem_addr_t page = current_addr >> m_log2_block_size;
      //access_limit = current_addr相对当前内存页的起始地址的偏移量 + 读长度。
      mem_addr_t access_limit = offset + nbytes_remain;
      //如果access_limit超过页大小 BSIZE，则需要换页。
      if (access_limit > BSIZE) {
        access_limit = BSIZE;
      }
      //换页前的读出长度为：BSIZE - offset。
      size_t tx_bytes = access_limit - offset;
      //第 page 号内存页读出数据。从起始地址 current_addr 开始，连续读 tx_bytes 个字节的数据，
      //将读出的数据放入 data 的 dst_offset 偏移位置。
      read_single_block(page, current_addr, tx_bytes,
                        &((unsigned char *)data)[dst_offset]);

      // advance pointers
      //前进指针。用于指导换页后的数据读。
      //换页后的需要读出保存的目的端数据的偏移地址变为 src_offset+tx_bytes。
      dst_offset += tx_bytes;
      //换页后的读出的全局起始地址为 current_addr+tx_bytes。
      current_addr += tx_bytes;
      //当前页已经读出 tx_bytes 字节长度数据，剩余需读出数据长度为 nbytes_remain-tx_bytes。
      nbytes_remain -= tx_bytes;
    }
    assert(nbytes_remain == 0);
  }
}

/*
打印存储中的数据。一般DEBUG用，用到再补充。
*/
template <unsigned BSIZE>
void memory_space_impl<BSIZE>::print(const char *format, FILE *fout) const {
  typename map_t::const_iterator i_page;

  for (i_page = m_data.begin(); i_page != m_data.end(); ++i_page) {
    fprintf(fout, "%s %08llx:", m_name.c_str(), i_page->first);
    i_page->second.print(format, fout);
  }
}

/*
DEBUG用，用到再补充。
*/
template <unsigned BSIZE>
void memory_space_impl<BSIZE>::set_watch(addr_t addr, unsigned watchpoint) {
  m_watchpoints[watchpoint] = addr;
}

template class memory_space_impl<32>;
template class memory_space_impl<64>;
template class memory_space_impl<8192>;
template class memory_space_impl<16 * 1024>;

void g_print_memory_space(memory_space *mem, const char *format = "%08x",
                          FILE *fout = stdout) {
  mem->print(format, fout);
}

#ifdef UNIT_TEST

int main(int argc, char *argv[]) {
  int errors_found = 0;
  memory_space *mem = new memory_space_impl<32>("test", 4);
  // write address to [address]
  for (mem_addr_t addr = 0; addr < 16 * 1024; addr += 4)
    mem->write(addr, 4, &addr, NULL, NULL);

  for (mem_addr_t addr = 0; addr < 16 * 1024; addr += 4) {
    unsigned tmp = 0;
    mem->read(addr, 4, &tmp);
    if (tmp != addr) {
      errors_found = 1;
      printf("ERROR ** mem[0x%x] = 0x%x, expected 0x%x\n", addr, tmp, addr);
    }
  }

  for (mem_addr_t addr = 0; addr < 16 * 1024; addr += 1) {
    unsigned char val = (addr + 128) % 256;
    mem->write(addr, 1, &val, NULL, NULL);
  }

  for (mem_addr_t addr = 0; addr < 16 * 1024; addr += 1) {
    unsigned tmp = 0;
    mem->read(addr, 1, &tmp);
    unsigned char val = (addr + 128) % 256;
    if (tmp != val) {
      errors_found = 1;
      printf("ERROR ** mem[0x%x] = 0x%x, expected 0x%x\n", addr, tmp,
             (unsigned)val);
    }
  }

  if (errors_found) {
    printf("SUMMARY:  ERRORS FOUND\n");
  } else {
    printf("SUMMARY: UNIT TEST PASSED\n");
  }
}

#endif
