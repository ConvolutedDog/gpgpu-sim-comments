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
Ϊ���Ż�����ģ������ܣ��ڴ����ù�ϣ��ʵ�ֵġ���ϣ��Ŀ��С��ģ���� memory_space_impl ��ģ�������
memory_space_impl ��ʵ�����ɳ����� memory_space ����Ķ�д�ӿڡ����ڲ���ÿ�� memory_space_impl 
�������һ���ڴ�ҳ������ģ�� mem_storage ʵ�֣�����ʹ��STL����Map���������Map�����ã���ָ�ΪSTL 
Map������ҳ��������Ӧ�ĵ�ַ��ϵ������ÿ�� mem_storage ������һ�����ж�д���ܵ��ֽ����顣�����ÿ�� 
memory_space �����ǿյģ��������ڴ�ռ��е���ҳ���Ӧ�ĵ�ַʱ��ͨ�� LD/ST ָ��� cudaMemcpy()����
ҳ�汻������䡣
*/
template <unsigned BSIZE>
memory_space_impl<BSIZE>::memory_space_impl(std::string name,
                                            unsigned hash_size) {
  //m_nameΪ���洢���ַ������֣�����cuda-sim.cc�д���shared memory��ʱ��͸����˸ÿ鹲��洢��
  //���֣�
  //    char buf[512];
  //    snprintf(buf, 512, "shared_%u", sid); <================== m_name
  //    shared_mem = new memory_space_impl<16 * 1024>(buf, 4);
  m_name = name;
  //MEM_MAP_RESIZE()��memory.h�ж��壺
  //    #define MEM_MAP_RESIZE(hash_size) (m_data.rehash(hash_size))
  MEM_MAP_RESIZE(hash_size);

  //m_log2_block_size=Log2(BSIZE)��������� Log2(BSIZE)���������BSIZEһ��ӦΪ 2 �ı�����
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
�򵥵��ڴ洢��д�����ݣ����漰��ҳ�洢�����⡣�ĸ������ֱ�Ϊ��
1. mem_addr_t offset��д��ַ��Χ����ʼ��ַ���m_data��ƫ������
2. mem_addr_t index��mem_storage<BSIZE> �����ڴ�ҳ����������
3. size_t length��д�����ݵĳ��ȣ����ֽ�Ϊ��λ��
4. const unsigned char *data��д���������ݡ�
*/
template <unsigned BSIZE>
void memory_space_impl<BSIZE>::write_only(mem_addr_t offset, mem_addr_t index,
                                          size_t length, const void *data) {
  //�� index ���ڴ�ҳд�����ݡ�m_data[index] ��һ��mem_storage<BSIZE> �����ڴ�ҳ����
  //�������ĳ�Ա������ʵ�ֶ����ڴ�ҳ��д�����ݡ�
  m_data[index].write(offset, length, (const unsigned char *)data);
}

/*
�ڴ洢��д�����ݣ��漰��ҳ�洢�����⡣�ĸ������ֱ�Ϊ��
1. mem_addr_t addr��д��ַ��Χ����ʼ��ַ��
2. size_t length��д�����ݵĳ��ȣ����ֽ�Ϊ��λ��
3. const unsigned char *data��д���������ݡ�
4. class ptx_thread_info *thd��DEBUG�ã������õ��ٲ��䡣
5. const ptx_instruction *pI��DEBUG�ã������õ��ٲ��䡣
*/
template <unsigned BSIZE>
void memory_space_impl<BSIZE>::write(mem_addr_t addr, size_t length,
                                     const void *data,
                                     class ptx_thread_info *thd,
                                     const ptx_instruction *pI) {
  //�� index ���ڴ�ҳд�����ݡ�m_data[index] ��һ��mem_storage<BSIZE> �����ڴ�ҳ������������
  //��Ա������ʵ�ֶ����ڴ�ҳ��д�����ݡ�addr�ı�ַ�Ǵӵ� 0 ���ڴ�ҳ��ʼ�ġ����������ǣ�
  //    �� addr��[0, 2^m_log2_block_size)ʱ��index=0��
  //    �� addr��[2^m_log2_block_size, 2^(m_log2_block_sizes+1))ʱ��index=1��
  //    ......�Դ����ơ�
  //����ÿ���ڴ�ҳֻ�洢 BSIZE �ֽڴ�С�����ݣ�������Ҫ���ڴ�ҳ��
  mem_addr_t index = addr >> m_log2_block_size;
  //printf("addr:%x, m_log2_block_size=%d, index=%x, BSIZE=%d\n", 
  //        addr, m_log2_block_size, index,BSIZE);

  //�ж�д���ݵĳ����Ƿ񳬹���ǰ�ڴ�ҳ��
  if ((addr + length) <= (index + 1) * BSIZE) {
    // fast route for intra-block access
    //���д���ݵĳ���û�г�����ǰ�ڴ�ҳ���Ϳ���ִ�п��ڵĿ��ٷ��ʡ�
    //offsetָ����д��ַ��Χ����ʼ��ַ��Ե�ǰ�ڴ�ҳ����ʼ��ַ��ƫ���������磺
    //    ��һ���洢��������ÿ���ڴ�ҳ�Ĵ�СΪ BSIZE=16�ֽڣ���
    //        addrΪ  0~15 ʱ�����ڵ� 0 ���ڴ�ҳ��
    //        addrΪ 16~31 ʱ�����ڵ� 1 ���ڴ�ҳ��
    //    ����� addr=17��
    //        addr & BSIZE = 'b10001 & ('b10000-'b1) = 'b10001 & 'b01111 = 'b1
    //    ��addr=17��Ե�ǰ�ڴ�ҳ����ʼ��ַ��ƫ����Ϊ 1��
    unsigned offset = addr & (BSIZE - 1);
    //д���ݵĳ��ȡ�
    unsigned nbytes = length;
    //�� index ���ڴ�ҳд�����ݡ�m_data[index] ��һ��mem_storage<BSIZE> �����ڴ�ҳ����
    //�������ĳ�Ա������ʵ�ֶ����ڴ�ҳ��д�����ݡ�
    m_data[index].write(offset, nbytes, (const unsigned char *)data);
  } else {
    // slow route for inter-block access
    //���д���ݵĳ��ȳ����˵�ǰ�ڴ�ҳ���Ϳ���ִ�п��ģ����ڴ�ҳ�������ٷ��ʡ�
    //��ʱ��������ס[д���ݵĳ���]/[��Ե�ǰ�ڴ�ҳ����ʼ��ַ��ƫ����]/[д���ݵ�ȫ�ֵ�ַ]����
    //�������ʲôλ�ÿ��ڴ�ҳ�ٵ�����nbytes_remain ��Ϊ��ʣ�����Ҫд�����ݳ��ȣ���ʼʱ����
    //Ϊ������д���ݳ��� length��current_addr Ϊ��ǰд���ȫ����ʼ��ַ���ڻ�ҳ����Ҫ��Ϊ��
    //ҳ���д���ȫ����ʼ��ַ��src_offset ��ָ��ǰҳ��Ҫд���Դ�����ݵ�ƫ�Ƶ�ַ�����磬��һ
    //ҳд��ʱ����ƫ����Ϊ0������д�볤��Ϊ _length_����ҳ�����Ҫд���Դ�����ݵ�ƫ�Ƶ�ַ��Ϊ
    //0+_length_=_length_��
    unsigned nbytes_remain = length;
    unsigned src_offset = 0;
    mem_addr_t current_addr = addr;

    //����д����̴��ڻ�ҳ���Ҳ�֪�������ỻ����ҳ���ܰ�����д�꣬��������[��ʣ�����Ҫд����
    //�ݳ���]ѭ����ֱ��nbytes_remain��Ϊ0����˵���Ѿ�����������ȫд����ˡ�
    while (nbytes_remain > 0) {
      //����current_addr��Ե�ǰ�ڴ�ҳ����ʼ��ַ��ƫ������
      unsigned offset = current_addr & (BSIZE - 1);
      //���㵱ǰ��ʼ��ַ���ڵı�д�����ݵ��ڴ�ҳ�� page �š�
      mem_addr_t page = current_addr >> m_log2_block_size;
      //access_limit = current_addr��Ե�ǰ�ڴ�ҳ����ʼ��ַ��ƫ���� + д�볤�ȡ�
      mem_addr_t access_limit = offset + nbytes_remain;
      //���access_limit����ҳ��С BSIZE������Ҫ��ҳ��
      if (access_limit > BSIZE) {
        access_limit = BSIZE;
      }
      //��ҳǰ��д�볤��Ϊ��BSIZE - offset��
      size_t tx_bytes = access_limit - offset;
      //�� page ���ڴ�ҳд�����ݡ�m_data[page] ��һ��mem_storage<BSIZE> �����ڴ�ҳ����
      //�������ĳ�Ա������ʵ�ֶ����ڴ�ҳ��д�����ݡ�
      m_data[page].write(offset, tx_bytes,
                         &((const unsigned char *)data)[src_offset]);

      // advance pointers
      //ǰ��ָ�롣����ָ����ҳ�������д�롣
      //��ҳ�����Ҫд���Դ�����ݵ�ƫ�Ƶ�ַ��Ϊ src_offset+tx_bytes��
      src_offset += tx_bytes;
      //��ҳ���д���ȫ����ʼ��ַΪ current_addr+tx_bytes��
      current_addr += tx_bytes;
      //��ǰҳ�Ѿ�д�� tx_bytes �ֽڳ������ݣ�ʣ����д�����ݳ���Ϊ nbytes_remain-tx_bytes��
      nbytes_remain -= tx_bytes;
    }
    assert(nbytes_remain == 0);
  }
  
  //DEBUG�ã������õ��ٲ��䡣
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
�������ڴ�ҳ�����ݡ��ĸ������ֱ�Ϊ��
1. mem_addr_t blk_idx�������������ڵ��ڴ�ҳ��š�
2. mem_addr_t addr������ַ��Χ����ʼ��ַ��
3. size_t length���������ݵĳ��ȣ����ֽ�Ϊ��λ��
4. void *data�����������ݷŵ� data �С�
*/
template <unsigned BSIZE>
void memory_space_impl<BSIZE>::read_single_block(mem_addr_t blk_idx,
                                                 mem_addr_t addr, size_t length,
                                                 void *data) const {
  //��һ�����ӣ�
  //    ��һ���洢��������ÿ���ڴ�ҳ�Ĵ�СΪ BSIZE=16�ֽڣ���
  //        addrΪ  0~15 ʱ�����ڵ� 0 ���ڴ�ҳ��
  //        addrΪ 16~31 ʱ�����ڵ� 1 ���ڴ�ҳ��
  //        addrΪ 32~47 ʱ�����ڵ� 3 ���ڴ�ҳ����
  //    1. ��� ����ַ addr=17��������Ϊ 10�������ڴ�ҳ��Ϊ 1��
  //       ����Χ����ֹ��ַ=(addr + length)=27
  //       �����ڴ�ҳ����ĩβ��ַ=(blk_idx + 1) * BSIZE=32
  //       27 <= 32��δ��ҳ���Ϸ���
  //    2. ��� ����ַ addr=28��������Ϊ 10�������ڴ�ҳ��Ϊ 1��
  //       ����Χ����ֹ��ַ=(addr + length)=38
  //       �����ڴ�ҳ����ĩβ��ַ=(blk_idx + 1) * BSIZE=32
  //       38 > 32����ҳ���Ƿ���
  //�����if�жϼ�Ϊ�ж϶��ڴ�ҳ�Ƿ���[�������ڴ�ҳ����]�����кϷ���
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
  //�� memory_space_impl �����е� m_data �� mem_storage ����ͬ��memory_space_impl ������
  //�� m_data ����Ϊһ�� unordered_map���� key-value �Էֱ�Ϊ��
  //    key: mem_addr_t ���͵� blk_idx���ڴ�ҳ��ţ���
  //    value: mem_storage<BSIZE> �ڴ�ҳ��
  //���� unordered_map.find(key) �Ĺ��ܣ�
  //    ���������Լ���key����Ϊ������
  //    ����ֵ����������ļ�������unordered_map�У��������Ԫ�ط���һ�������������򷵻�ӳ���
  //           ������ĩβ��
  typename map_t::const_iterator i = m_data.find(blk_idx);
  //��� i == m_data.end()��˵�� m_data ������ blk_idx ��ʶ���ڴ�ҳ��
  if (i == m_data.end()) {
    //m_data ������ blk_idx ��ʶ���ڴ�ҳ���� data ȫ�����㡣
    for (size_t n = 0; n < length; n++)
      ((unsigned char *)data)[n] = (unsigned char)0;
    // printf("GPGPU-Sim PTX:  WARNING reading %zu bytes from unititialized
    // memory at address 0x%x in space %s\n", length, addr, m_name.c_str() );
  } else {
    //��� i != m_data.end()��m_data ���� blk_idx ��ʶ���ڴ�ҳ��i��ָ����ڴ�ҳ�ĵ�������
    //���� addr ��Ե�ǰ�ڴ�ҳ����ʼ��ַ��ƫ������
    unsigned offset = addr & (BSIZE - 1);
    unsigned nbytes = length;
    //�����ݣ����������ݷ��� data��
    i->second.read(offset, nbytes, (unsigned char *)data);
  }
}

/*
�����ܿ��ڴ�ҳ�����ݡ��ĸ������ֱ�Ϊ��
1. mem_addr_t addr������ַ��Χ����ʼ��ַ��
2. size_t length���������ݵĳ��ȣ����ֽ�Ϊ��λ��
3. void *data�����������ݷŵ� data �С�
*/
template <unsigned BSIZE>
void memory_space_impl<BSIZE>::read(mem_addr_t addr, size_t length,
                                    void *data) const {
  //���㵱ǰ��ʼ��ַ���ڵı��������ݵ��ڴ�ҳ�� index �š�
  mem_addr_t index = addr >> m_log2_block_size;
  //��һ�����ӣ�
  //    ��һ���洢��������ÿ���ڴ�ҳ�Ĵ�СΪ BSIZE=16�ֽڣ���
  //        addrΪ  0~15 ʱ�����ڵ� 0 ���ڴ�ҳ��
  //        addrΪ 16~31 ʱ�����ڵ� 1 ���ڴ�ҳ��
  //        addrΪ 32~47 ʱ�����ڵ� 3 ���ڴ�ҳ����
  //    1. ��� ����ַ addr=17��������Ϊ 10�������ڴ�ҳ��Ϊ 1��
  //       ����Χ����ֹ��ַ=(addr + length)=27
  //       �����ڴ�ҳ����ĩβ��ַ=(blk_idx + 1) * BSIZE=32
  //       27 <= 32��δ��ҳ���Ϸ���
  //    2. ��� ����ַ addr=28��������Ϊ 10�������ڴ�ҳ��Ϊ 1��
  //       ����Χ����ֹ��ַ=(addr + length)=38
  //       �����ڴ�ҳ����ĩβ��ַ=(blk_idx + 1) * BSIZE=32
  //       38 > 32����ҳ���Ƿ���
  //�����if�жϼ�Ϊ�ж϶��ڴ�ҳ�Ƿ��ҳ����
  if ((addr + length) <= (index + 1) * BSIZE) {
    // fast route for intra-block access
    //����ҳ���Ļ����ͼ򵥵�ִ�е�ҳ�ڶ����ݼ��ɣ�ִ�п��ڵĿ��ٷ��ʡ�
    read_single_block(index, addr, length, data);
  } else {
    // slow route for inter-block access
    //��ҳ���Ļ�������Ҫ��ζ���ͬҳ�����ݣ�ִ�п��ģ����ڴ�ҳ�������ٷ��ʡ�
    //nbytes_remain ��Ϊ��ʣ�����Ҫ�������ݳ��ȣ���ʼʱ����Ϊ�����Ķ����ݳ��� length��
    //dst_offset ��ָ��ǰҳ��Ҫ�������ݴ����Ŀ�Ķ����ݵ�ƫ�Ƶ�ַ�����磬��һҳ�������浽 data 
    //ʱ����ƫ����Ϊ0������������ȱ��浽 data �еĳ���Ϊ _length_����ҳ�����Ҫ�ٴζ���������
    //��Ŀ�Ķ����ݵ�ƫ�Ƶ�ַ��Ϊ0+_length_=_length_��current_addr Ϊ��ǰ������ȫ����ʼ��ַ��
    //�ڻ�ҳ����Ҫ��Ϊ��ҳ�������ȫ����ʼ��ַ��
    unsigned nbytes_remain = length;
    unsigned dst_offset = 0;
    mem_addr_t current_addr = addr;

    //���ڶ������̴��ڻ�ҳ���Ҳ�֪�������ỻ����ҳ���ܰ����ݶ��꣬��������[��ʣ�����Ҫ������
    //�ݳ���]ѭ����ֱ��nbytes_remain��Ϊ0����˵���Ѿ�����������ȫ������ɡ�
    while (nbytes_remain > 0) {
      //����current_addr��Ե�ǰ�ڴ�ҳ����ʼ��ַ��ƫ������
      unsigned offset = current_addr & (BSIZE - 1);
      //���㵱ǰ��ʼ��ַ���ڵı������ݵ��ڴ�ҳ�� page �š�
      mem_addr_t page = current_addr >> m_log2_block_size;
      //access_limit = current_addr��Ե�ǰ�ڴ�ҳ����ʼ��ַ��ƫ���� + �����ȡ�
      mem_addr_t access_limit = offset + nbytes_remain;
      //���access_limit����ҳ��С BSIZE������Ҫ��ҳ��
      if (access_limit > BSIZE) {
        access_limit = BSIZE;
      }
      //��ҳǰ�Ķ�������Ϊ��BSIZE - offset��
      size_t tx_bytes = access_limit - offset;
      //�� page ���ڴ�ҳ�������ݡ�����ʼ��ַ current_addr ��ʼ�������� tx_bytes ���ֽڵ����ݣ�
      //�����������ݷ��� data �� dst_offset ƫ��λ�á�
      read_single_block(page, current_addr, tx_bytes,
                        &((unsigned char *)data)[dst_offset]);

      // advance pointers
      //ǰ��ָ�롣����ָ����ҳ������ݶ���
      //��ҳ�����Ҫ���������Ŀ�Ķ����ݵ�ƫ�Ƶ�ַ��Ϊ src_offset+tx_bytes��
      dst_offset += tx_bytes;
      //��ҳ��Ķ�����ȫ����ʼ��ַΪ current_addr+tx_bytes��
      current_addr += tx_bytes;
      //��ǰҳ�Ѿ����� tx_bytes �ֽڳ������ݣ�ʣ����������ݳ���Ϊ nbytes_remain-tx_bytes��
      nbytes_remain -= tx_bytes;
    }
    assert(nbytes_remain == 0);
  }
}

/*
��ӡ�洢�е����ݡ�һ��DEBUG�ã��õ��ٲ��䡣
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
DEBUG�ã��õ��ٲ��䡣
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
