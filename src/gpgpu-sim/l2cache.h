// Copyright (c) 2009-2021, Tor M. Aamodt, Vijay Kandiah, Nikos Hardavellas,
// Mahmoud Khairy, Junrui Pan, Timothy G. Rogers
// The University of British Columbia, Northwestern University, Purdue
// University All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this
//    list of conditions and the following disclaimer;
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution;
// 3. Neither the names of The University of British Columbia, Northwestern
//    University nor the names of their contributors may be used to
//    endorse or promote products derived from this software without specific
//    prior written permission.
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

#ifndef MC_PARTITION_INCLUDED
#define MC_PARTITION_INCLUDED

#include "../abstract_hardware_model.h"
#include "dram.h"

#include <list>
#include <queue>

class mem_fetch;

/*
�������ڴ������L2 Cache����mem_fetch�����ڴ����󣩡�
*/
class partition_mf_allocator : public mem_fetch_allocator {
 public:
  partition_mf_allocator(const memory_config *config) {
    m_memory_config = config;
  }
  virtual mem_fetch *alloc(const class warp_inst_t &inst,
                           const mem_access_t &access,
                           unsigned long long cycle) const {
    abort();
    return NULL;
  }
  virtual mem_fetch *alloc(new_addr_type addr, mem_access_type type,
                           unsigned size, bool wr, unsigned long long cycle,
                           unsigned long long streamID) const;
  virtual mem_fetch *alloc(new_addr_type addr, mem_access_type type,
                           const active_mask_t &active_mask,
                           const mem_access_byte_mask_t &byte_mask,
                           const mem_access_sector_mask_t &sector_mask,
                           unsigned size, bool wr, unsigned long long cycle,
                           unsigned wid, unsigned sid, unsigned tpc,
                           mem_fetch *original_mf,
                           unsigned long long streamID) const;

 private:
  const memory_config *m_memory_config;
};

// Memory partition unit contains all the units assolcated with a single DRAM
// channel.
// - It arbitrates the DRAM channel among multiple sub partitions.
// - It does not connect directly with the interconnection network.
class memory_partition_unit {
 public:
  // һ��memory partition unitʵ������һ���洢��������DRAM Channel������V100��m_n_mem����Ϊ32��ÿ���ڴ�
  // �������ְ�������ӷ�����V100��m_n_sub_partition_per_memory_channel����Ϊ2��m_n_mem_sub_partition
  // ʵ����������GPU�ϵ��ӷ�������=m_n_mem*m_n_sub_partition_per_memory_channel=64�������GPU����ʱ����
  // �ڴ�������ڴ��ӷ����У�
  //   m_memory_partition_unit =
  //       new memory_partition_unit *[m_memory_config->m_n_mem];
  //   m_memory_sub_partition =
  //       new memory_sub_partition *[m_memory_config->m_n_mem_sub_partition];
  //   for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
  //     m_memory_partition_unit[i] = // �����i��partition_id
  //         new memory_partition_unit(i, m_memory_config, m_memory_stats, this);
  //     for (unsigned p = 0;
  //          p < m_memory_config->m_n_sub_partition_per_memory_channel; p++) {
  //       unsigned submpid =
  //           i * m_memory_config->m_n_sub_partition_per_memory_channel + p;
  //       m_memory_sub_partition[submpid] = // �����submpid��sub_partition_id
  //           m_memory_partition_unit[i]->get_sub_partition(p);
  //     }
  //   }
  memory_partition_unit(unsigned partition_id, const memory_config *config,
                        class memory_stats_t *stats, class gpgpu_sim *gpu);
  ~memory_partition_unit();

  bool busy() const;

  void cache_cycle(unsigned cycle);
  void dram_cycle();
  void simple_dram_model_cycle();

  void set_done(mem_fetch *mf);

  void visualizer_print(gzFile visualizer_file) const;
  void print_stat(FILE *fp) { m_dram->print_stat(fp); }
  void visualize() const { m_dram->visualize(); }
  void print(FILE *fp) const;
  void handle_memcpy_to_gpu(size_t dst_start_addr, unsigned subpart_id,
                            mem_access_sector_mask_t mask);

  class memory_sub_partition *get_sub_partition(int sub_partition_id) {
    return m_sub_partition[sub_partition_id];
  }

  // Power model
  void set_dram_power_stats(unsigned &n_cmd, unsigned &n_activity,
                            unsigned &n_nop, unsigned &n_act, unsigned &n_pre,
                            unsigned &n_rd, unsigned &n_wr, unsigned &n_wr_WB,
                            unsigned &n_req) const;

  int global_sub_partition_id_to_local_id(int global_sub_partition_id) const;

  unsigned get_mpid() const { return m_id; }

  class gpgpu_sim *get_mgpu() const {
    return m_gpu;
  }

 private:
  unsigned m_id;
  const memory_config *m_config;
  class memory_stats_t *m_stats;
  class memory_sub_partition **m_sub_partition;
  class dram_t *m_dram;

  class arbitration_metadata {
   public:
    arbitration_metadata(const memory_config *config);

    // check if a subpartition still has credit
    bool has_credits(int inner_sub_partition_id) const;
    // borrow a credit for a subpartition
    void borrow_credit(int inner_sub_partition_id);
    // return a credit from a subpartition
    void return_credit(int inner_sub_partition_id);

    // return the last subpartition that borrowed credit
    int last_borrower() const { return m_last_borrower; }

    void print(FILE *fp) const;

   private:
    // id of the last subpartition that borrowed credit
    int m_last_borrower;

    int m_shared_credit_limit;
    int m_private_credit_limit;

    // credits borrowed by the subpartitions
    std::vector<int> m_private_credit;
    int m_shared_credit;
  };
  arbitration_metadata m_arbitration_metadata;

  // determine wheither a given subpartition can issue to DRAM
  bool can_issue_to_dram(int inner_sub_partition_id);

  // model DRAM access scheduler latency (fixed latency between L2 and DRAM)
  struct dram_delay_t {
    unsigned long long ready_cycle;
    class mem_fetch *req;
  };
  std::list<dram_delay_t> m_dram_latency_queue;

  class gpgpu_sim *m_gpu;
};


/*
L1���ݸ��ٻ����ڸø��ٻ�����ά��ȫ�ִ洢����ַ�ռ���Ӽ�����һЩ�ܹ��У�L1���ٻ�������������ں��޸ĵ�λ�ã�
�������ڱ�������GPU��ȱ�����ٻ���һ���Զ����µĸ����ԡ��ӳ���Ա�ĽǶ�����������ȫ�ִ洢��ʱ�Ĺؼ��������ɸ�
���߳����Ĳ�ͬ�̷߳��ʵĴ洢��λ������ڱ˴˵Ĺ�ϵ������߳����е������̷߳������ڵ���L1���ݸ��ٻ�����ڵ�λ
�ò��Ҹÿ鲻���ڸø��ٻ����У������Ҫ��ϵͼ���ĸ��ٻ��淢�͵������������ķ��ʱ���Ϊ���ϲ��ġ�������߳���
�ڵ��̷߳��ʲ�ͬ�ĸ��ٻ���飬����Ҫ���ɶ���洢�����ʡ������ķ��ʱ���Ϊδ�ϲ��ġ�����Ա��ͼ����洢���ͻ��
δ�ϲ��ķ��ʣ���Ϊ�˼򻯱�̣�Ӳ���������ߡ�

���ȿ�����δ�����洢�����ʣ�Ȼ���Ǻϲ��ĸ��ٻ������У�����Ǹ��ٻ���δ���к�δ�ϲ��ķ��ʡ������������
�洢�������������ȴ�ָ����ˮ���ڵļ���/�洢��Ԫ���͵�L1���ٻ��档�洢������������һ��洢����ַ��ɣ�һ����
�����е�ÿ���̶߳�Ӧһ���洢����ַ�Լ��������͡�

���ڹ���洢����ȡ���ٲ���ȷ���߳����ڵ������ַ�Ƿ�����洢���ͻ�����������ĵ�ַ������һ�������洢��
��ͻ�����ٲ���������ֳ��������֡���һ���ְ����߳����в����д洢���ͻ���߳��Ӽ��ĵ�ַ���ٲ�������ԭʼ�����
��һ���֣��Թ��ø��ٻ����һ�������ڶ����ְ������һ�����еĵ�ַ���´洢���ͻ����Щ��ַ��ԭʼ�������һ��
�ֱ����ص�ָ����ˮ�ߣ����ұ����ٴ�ִ�У��ú���ִ�б���Ϊ"reply"���ڴ洢ԭʼ����洢�������reply����ʱ������
�ԡ���Ȼ����ͨ��reply����ָ������Ĵ洢������ָ������ʡ����������ڷ��ʴ�Ĵ����ļ�ʱ��������������Ч�ʵ�
������������������ṩ���޵Ļ���������reply����/�洢��Ԫ�еĴ洢������ָ����ұ����ڸû������еĿ��пռ��
��ʱ��������ָ������Ĵ洢�����ʲ������ڿ���reply����ᷢ��ʲô֮ǰ�������ǿ�����δ���洢������Ľ��ܲ��֡�

����洢������Ľ��ܲ����ƹ���ǩ��Ԫ�ڵı�ǩ���ң���Ҫ�ǲ�ѯ�����Ƿ���L1 Cache�У�����Ϊ����洢����ֱ��ӳ�䡣
�����ܹ���洢����������ʱ���ٲ�����д���¼����ȵ�ָ����ˮ���ڵļĴ����ļ�����Ϊ��û�д洢���ͻ�������ֱ��
ӳ��洢�����ҵĵȴ�ʱ���Ǻ㶨�ġ���ǩ��Ԫȷ��ÿ���̵߳�����ӳ�䵽�ĸ�Bank���Ա���Ƶ�ַ���濪�أ���ַ���濪
�ؽ���ַ���䵽���������ڵĸ����洢�塣���������ڵ�ÿ���洢����32λ��ģ����Ҿ������Լ��Ľ������������������
ÿ���洢���еĲ�ͬ�С����ݾ������ݽ��濪�ط��ص��ʵ��̵߳�ͨ���Դ洢�ڼĴ������С�ֻ�ж�Ӧ���߳����еĻ�Ծ��
�̵�ͨ����ֵд��Ĵ����ļ���

����洢�������reply���ֿ�������ǰ���ܵĲ���֮������ڷ���L1���ٻ����ٲ����������reply�����ٴ������洢���
ͻ�������һ��ϸ��Ϊ���ܺ�reply���֡� <== bank conflict consumes a lot of cycles

�������������ǿ�����δ���ȫ���ڴ�ռ�ļ��ء�����ֻ��ȫ�ִ洢���ռ���Ӽ���������L1���ٻ����У���ǩ��Ԫ����
Ҫ��������Ƿ�����ڸø��ٻ����С���Ȼ�������б��߶ȴ洢��ʹ���ܹ��ɸ����߳������ط��ʹ���洢��������ȫ��
�洢���ķ��ʱ�����Ϊÿ�����ڵ������ٻ���顣�����������ڼ�������ڸ��ٻ����������ı�ǩ�洢����������Ҳ�Ǳ�׼
DRAMоƬ�ı�׼�ӿڵĽ�����ڷ��׺Ϳ������У�L1������СΪ128�ֽڣ������˹Τ��Pascal�У���һ����Ϊ�ĸ�32��
��������32�ֽ�������С��Ӧ�ڿ��ڵ��δ�ȡ�д��½�ͼ��DRAMоƬ��ȡ�����ݵ���С��С�����磬GDDR5����ÿ��128��
�ڸ��ٻ������32���洢���е�ÿһ������ͬ�е�32λ��Ŀ��ɡ�

����/�洢��Ԫ����洢����ַ��Ӧ�úϲ������Խ��߳����Ĵ洢�����ʷֽ�ɵ����ĺϲ��ķ��ʣ���Щ�ϲ��ķ���Ȼ����
�͵��ٲ����С����û���㹻����Դ���ã����ٲ������Ծܾ��������磬�������ӳ�䵽�ø��ٻ������е�����ways��æ
µ��������pending request table��û�п�����Ŀ���⽫�������������������㹻����Դ�����ڴ���δ���У��ٲ�������
ָ����ˮ���ڶ�Ӧ�ڸ��ٻ������е�δ���̶������������е��Ȼ�д���Ĵ����ļ������еأ��ٲ����������ǩ��Ԫ����
��ʵ�����Ƿ��¸��ٻ������л�δ���С��ڸ��ٻ������е�����£������д洢���з����������е��ʵ��У��������ݱ�
���ص�ָ����ˮ���еļĴ����ѡ����ڹ���洢����ȡ������£������¶�Ӧ���������̵߳ļĴ���ͨ����

�����ʱ�ǩ��Ԫʱ�����ȷ�����󴥷����ٻ���δ���У����ٲ���֪ͨ����/�洢��Ԫ������reply�����󣬲��Ҳ��е�����
������Ϣ���͵�pending request table��PRT����pending request table�ṩ�Ĺ�����CPU���ٻ���洢��ϵͳ�еĴ�
ͳȱʧ״̬���ּĴ�����֧�ֵĹ���û��ʲô��ͬ����NVIDIAר����ͼ4.1��ʾ��L1������ϵ�ṹ��صİ汾�������е���
���ڴ�ͳ��MSHR�����ݻ���Ĵ�ͳMSHR��������δ���еĿ��ַ�Լ���ƫ��������ؼĴ�������Ϣ�����鱻���ø��ٻ���
��ʱ����Ҫд�롣ͨ����¼�����ƫ�ƺͼĴ�����֧�ֶ�ͬһ��Ķ��δ���С�ͼ4.1�е�PRT֧�ֽ���������ϲ���ͬһ�飬
����¼��Ϣ����ָ֪ͨ����ˮ���ӳٴ洢��������reply��

![ͼ4.1](../../comments-figs/UnifiedL1DataCacheAndSharedMemory.png)

ͼ4.1��ʾ��L1���ݻ��������������������ǵġ������������������/�����ǵ�L1���ݸ��ٻ�����ִ�CPU΢�ܹ���
��ʱ������������˾��ȵġ�CPUʹ��������֯���������������л���ˢ��L1���ݻ���Ŀ�������ȻGPU���߳���������ÿ��
������Ч��ִ���������л������߳�����ͬһӦ�õ�һ���֡�����ҳ�������洢����GPU����Ȼ�������ģ���ʹ��������Ϊ
һ�����е���OSӦ�ó�����Ϊ�������ڼ򻯴洢�����䲢���ٴ洢����Ƭ����PRT�з�����Ŀ֮�󣬴洢������ת������
��������Ԫ��MMU�����������⵽�����ַת�������Ҵ�����ͨ�����滥��ת�����ʵ��Ĵ洢��������Ԫ�����罫��4.3��
����չ���������ڴ������Ԫ����һ��L2���ٻ��������һ���ڴ���ʵ����������Ź���Ҫ�����ĸ�����洢����ַ��Ҫ��
ȡ�����ֽڵ���Ϣ���洢���������"subid"�����洢�����󷵻ص���ʱ����"subid"�������ڲ���PRT�а��������������
Ϣ����Ŀ��

һ����Լ��صĴ洢��������Ӧ�����ص��ˣ����ͱ�MMU���ݵ���䵥Ԫ����䵥Ԫ����ʹ�ô洢�������е�subid�ֶ�����
PRT�в��ҹ����������Ϣ���������������䵥Ԫ�����ٲ������ݵ�����/�洢��Ԫ�����µ��ȼ��ص���Ϣ��Ȼ��ͨ���ڸ�
�ٻ����е����Ѿ������õ�����������֮���������ٻ����е�������֤�������и��ٻ��档

ͼ4.1�е�L1���ݻ������֧��ֱд�ͻ�д���ԡ���ˣ���ȫ�ִ洢���Ĵ洢ָ�д�룩���������ɷ�ʽ����д����ض�
�ڴ�ռ�ȷ��д���Ǳ���Ϊֱд���ǻ�д�������GPGPUӦ�ó����ж�ȫ�ִ洢���ķ��ʿ���Ԥ�ھ��зǳ����ʱ��ֲ��ԣ�
��Ϊͨ���ں����߳����˳�֮ǰ������д���������еķ�ʽ��д�����������ķ��ʣ���д�����ֱд���Կ��������塣���֮
�£���������Ĵ�������ջ�ı��ش洢��д�������ʾ�����õ�ʱ��ֲ��ԣ��������ļ���֤������д�������ԵĻ�д��
����ġ�

Ҫд�빲��洢����ȫ�ִ洢�����������ȱ�������д�����ݻ�������WDB���С����ڷǺϲ����ʻ�ĳЩ�̱߳�����ʱ����д
�뻺����һ���֡��������ڸø��ٻ����У�����Ծ������ݽ��濪�ؽ�����д���������С�������ݲ����ڸø��ٻ����У�
��������ȴ�L2���ٻ����DRAM�洢����ȡ�顣�����ȫ�����ٻ����ĺϲ�д��ʹ���ٻ����е��κγ¾����ݵı�ǩ��Ч��
��ϲ�д������ƹ��ø��ٻ��档

ע��ͼ4.1�������ø��ٻ�����֯��֧�ֻ���һ���ԡ����磬������SM1��ִ�е��̶߳�ȡ�洢��λ��A���Ҹ�ֵ���洢��SM1��
L1���ݸ��ٻ����У�Ȼ����SM2��ִ�е���һ�߳�д��洢��λ��A�����SM1�ϵ��κ��߳�����ڴ洢��λ��A����SM1��L1��
�ݸ��ٻ������֮ǰ��ȡ�洢��λ��A�����佫��þ�ֵ��������ֵ��Ϊ�˱���������⣬��Kepler��ʼ��NVIDIA GPUֻ����
�Ĵ�������Ͷ�ջ���ݻ�ֻ��ȫ���ڴ����ݵı����ڴ���ʱ�������L1���ݻ����С�������о��Ѿ�̽���������GPU��������
�ɵ�L1���ݻ����Լ�����ȷ�����GPU�洢��һ����ģ�͵���Ҫ��

# 32 sets, each 128 bytes 24-way for each memory sub partition (96 KB per memory sub partition). This gives us 6MB L2 cache.
# With previous defined configs: -gpgpu_n_mem 32 and -gpgpu_n_sub_partition_per_mchannel 2, we can calculate that the num of
# sub partition of memories is gpgpu_n_mem*gpgpu_n_sub_partition_per_mchannel = 64. So, with 96 KB per memory sub partition,
# we get 6MB (64*96KB) L2 cache. To match the DRAM atom size of 32 bytes in GDDR5, each cache line inside the slice has four
# 32-byte sectors.
*/
class memory_sub_partition {
 public:
  // һ��memory partition unitʵ������һ���洢��������DRAM Channel������V100��m_n_mem����Ϊ32��ÿ���ڴ�
  // �������ְ�������ӷ�����V100��m_n_sub_partition_per_memory_channel����Ϊ2��m_n_mem_sub_partition
  // ʵ����������GPU�ϵ��ӷ�������=m_n_mem*m_n_sub_partition_per_memory_channel=64�������GPU����ʱ����
  // �ڴ�������ڴ��ӷ����У�
  //   m_memory_partition_unit =
  //       new memory_partition_unit *[m_memory_config->m_n_mem];
  //   m_memory_sub_partition =
  //       new memory_sub_partition *[m_memory_config->m_n_mem_sub_partition];
  //   for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
  //     m_memory_partition_unit[i] = // �����i��partition_id
  //         new memory_partition_unit(i, m_memory_config, m_memory_stats, this);
  //     for (unsigned p = 0;
  //          p < m_memory_config->m_n_sub_partition_per_memory_channel; p++) {
  //       unsigned submpid =
  //           i * m_memory_config->m_n_sub_partition_per_memory_channel + p;
  //       m_memory_sub_partition[submpid] = // �����submpid��sub_partition_id
  //           m_memory_partition_unit[i]->get_sub_partition(p);
  //     }
  //   }
  memory_sub_partition(unsigned sub_partition_id, const memory_config *config,
                       class memory_stats_t *stats, class gpgpu_sim *gpu);
  ~memory_sub_partition();
  //m_sub_partition[dest_spid]->get_id()�����ڴ��ӷ�����ȫ��ID��
  unsigned get_id() const { return m_id; }

  bool busy() const;

  void cache_cycle(unsigned cycle);

  bool full() const;
  bool full(unsigned size) const;
  void push(class mem_fetch *mf, unsigned long long clock_cycle);
  class mem_fetch *pop();
  class mem_fetch *top();
  void set_done(mem_fetch *mf);

  unsigned flushL2();
  unsigned invalidateL2();

  // interface to L2_dram_queue
  bool L2_dram_queue_empty() const;
  class mem_fetch *L2_dram_queue_top() const;
  void L2_dram_queue_pop();

  // interface to dram_L2_queue
  bool dram_L2_queue_full() const;
  void dram_L2_queue_push(class mem_fetch *mf);

  void visualizer_print(gzFile visualizer_file);
  void print_cache_stat(unsigned &accesses, unsigned &misses) const;
  void print(FILE *fp) const;

  void accumulate_L2cache_stats(class cache_stats &l2_stats) const;
  void get_L2cache_sub_stats(struct cache_sub_stats &css) const;

  // Support for getting per-window L2 stats for AerialVision
  void get_L2cache_sub_stats_pw(struct cache_sub_stats_pw &css) const;
  void clear_L2cache_stats_pw();

  void force_l2_tag_update(new_addr_type addr, unsigned time,
                           mem_access_sector_mask_t mask) {
    m_L2cache->force_tag_access(addr, m_memcpy_cycle_offset + time, mask);
    m_memcpy_cycle_offset += 1;
  }

 private:
  // data
  //�ڴ��ӷ�����ȫ��ID��
  unsigned m_id;  //< the global sub partition ID
  const memory_config *m_config;
  class l2_cache *m_L2cache;
  class L2interface *m_L2interface;
  class gpgpu_sim *m_gpu;
  partition_mf_allocator *m_mf_allocator;

  // model delay of ROP units with a fixed latency
  struct rop_delay_t {
    unsigned long long ready_cycle;
    class mem_fetch *req;
  };
  std::queue<rop_delay_t> m_rop;

  // these are various FIFOs between units within a memory partition
  fifo_pipeline<mem_fetch> *m_icnt_L2_queue;
  fifo_pipeline<mem_fetch> *m_L2_dram_queue;
  fifo_pipeline<mem_fetch> *m_dram_L2_queue;
  fifo_pipeline<mem_fetch> *m_L2_icnt_queue;  // L2 cache hit response queue

  class mem_fetch *L2dramout;
  unsigned long long int wb_addr;

  class memory_stats_t *m_stats;

  std::set<mem_fetch *> m_request_tracker;

  friend class L2interface;

  std::vector<mem_fetch *> breakdown_request_to_sector_requests(mem_fetch *mf);

  // This is a cycle offset that has to be applied to the l2 accesses to account
  // for the cudamemcpy read/writes. We want GPGPU-Sim to only count cycles for
  // kernel execution but we want cudamemcpy to go through the L2. Everytime an
  // access is made from cudamemcpy this counter is incremented, and when the l2
  // is accessed (in both cudamemcpyies and otherwise) this value is added to
  // the gpgpu-sim cycle counters.
  unsigned m_memcpy_cycle_offset;
};

/*
��L2�ô�Ľӿڡ�mem_fetch_interface�Ƕ�mem�ô�Ľӿڡ�
*/
class L2interface : public mem_fetch_interface {
 public:
  L2interface(memory_sub_partition *unit) { m_unit = unit; }
  virtual ~L2interface() {}
  //����L2�ô���������Ƿ����ˡ�
  virtual bool full(unsigned size, bool write) const {
    // assume read and write packets all same size
    return m_unit->m_L2_dram_queue->full();
  }
  //���·ô��������L2�ô�������С�
  virtual void push(mem_fetch *mf) {
    mf->set_status(IN_PARTITION_L2_TO_DRAM_QUEUE, 0 /*FIXME*/);
    m_unit->m_L2_dram_queue->push(mf);
  }

 private:
  memory_sub_partition *m_unit;
};

#endif
