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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <list>
#include <set>

#include "../abstract_hardware_model.h"
#include "../option_parser.h"
#include "../statwrapper.h"
#include "dram.h"
#include "gpu-cache.h"
#include "gpu-sim.h"
#include "histogram.h"
#include "l2cache.h"
#include "l2cache_trace.h"
#include "mem_fetch.h"
#include "mem_latency_stat.h"
#include "shader.h"

mem_fetch *partition_mf_allocator::alloc(new_addr_type addr,
                                         mem_access_type type, unsigned size,
                                         bool wr, unsigned long long cycle,
                                         unsigned long long streamID) const {
  assert(wr);
  mem_access_t access(type, addr, size, wr, m_memory_config->gpgpu_ctx);
  mem_fetch *mf = new mem_fetch(access, NULL, streamID, WRITE_PACKET_SIZE, -1,
                                -1, -1, m_memory_config, cycle);
  return mf;
}

mem_fetch *partition_mf_allocator::alloc(
    new_addr_type addr, mem_access_type type, const active_mask_t &active_mask,
    const mem_access_byte_mask_t &byte_mask,
    const mem_access_sector_mask_t &sector_mask, unsigned size, bool wr,
    unsigned long long cycle, unsigned wid, unsigned sid, unsigned tpc,
    mem_fetch *original_mf, unsigned long long streamID) const {
  mem_access_t access(type, addr, size, wr, active_mask, byte_mask, sector_mask,
                      m_memory_config->gpgpu_ctx);
  mem_fetch *mf = new mem_fetch(access, NULL, streamID,
                                wr ? WRITE_PACKET_SIZE : READ_PACKET_SIZE, wid,
                                sid, tpc, m_memory_config, cycle, original_mf);
  return mf;
}
memory_partition_unit::memory_partition_unit(unsigned partition_id,
                                             const memory_config *config,
                                             class memory_stats_t *stats,
                                             class gpgpu_sim *gpu)
    : m_id(partition_id),
      m_config(config),
      m_stats(stats),
      m_arbitration_metadata(config),
      m_gpu(gpu) {
  m_dram = new dram_t(m_id, m_config, m_stats, this, gpu);

  m_sub_partition = new memory_sub_partition
      *[m_config->m_n_sub_partition_per_memory_channel];
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    unsigned sub_partition_id =
        m_id * m_config->m_n_sub_partition_per_memory_channel + p;
    m_sub_partition[p] =
        new memory_sub_partition(sub_partition_id, m_config, stats, gpu);
  }
}

void memory_partition_unit::handle_memcpy_to_gpu(
    size_t addr, unsigned global_subpart_id, mem_access_sector_mask_t mask) {
  unsigned p = global_sub_partition_id_to_local_id(global_subpart_id);
  std::string mystring = mask.to_string<char, std::string::traits_type,
                                        std::string::allocator_type>();
  MEMPART_DPRINTF(
      "Copy Engine Request Received For Address=%zx, local_subpart=%u, "
      "global_subpart=%u, sector_mask=%s \n",
      addr, p, global_subpart_id, mystring.c_str());
  m_sub_partition[p]->force_l2_tag_update(
      addr, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle, mask);
}

memory_partition_unit::~memory_partition_unit() {
  delete m_dram;
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    delete m_sub_partition[p];
  }
  delete[] m_sub_partition;
}

memory_partition_unit::arbitration_metadata::arbitration_metadata(
    const memory_config *config)
    : m_last_borrower(config->m_n_sub_partition_per_memory_channel - 1),
      m_private_credit(config->m_n_sub_partition_per_memory_channel, 0),
      m_shared_credit(0) {
  // each sub partition get at least 1 credit for forward progress
  // the rest is shared among with other partitions
  m_private_credit_limit = 1;
  m_shared_credit_limit = config->gpgpu_frfcfs_dram_sched_queue_size +
                          config->gpgpu_dram_return_queue_size -
                          (config->m_n_sub_partition_per_memory_channel - 1);
  if (config->seperate_write_queue_enabled)
    m_shared_credit_limit += config->gpgpu_frfcfs_dram_write_queue_size;
  if (config->gpgpu_frfcfs_dram_sched_queue_size == 0 or
      config->gpgpu_dram_return_queue_size == 0) {
    m_shared_credit_limit =
        0;  // no limit if either of the queue has no limit in size
  }
  assert(m_shared_credit_limit >= 0);
}

bool memory_partition_unit::arbitration_metadata::has_credits(
    int inner_sub_partition_id) const {
  int spid = inner_sub_partition_id;
  if (m_private_credit[spid] < m_private_credit_limit) {
    return true;
  } else if (m_shared_credit_limit == 0 ||
             m_shared_credit < m_shared_credit_limit) {
    return true;
  } else {
    return false;
  }
}

void memory_partition_unit::arbitration_metadata::borrow_credit(
    int inner_sub_partition_id) {
  int spid = inner_sub_partition_id;
  if (m_private_credit[spid] < m_private_credit_limit) {
    m_private_credit[spid] += 1;
  } else if (m_shared_credit_limit == 0 ||
             m_shared_credit < m_shared_credit_limit) {
    m_shared_credit += 1;
  } else {
    assert(0 && "DRAM arbitration error: Borrowing from depleted credit!");
  }
  m_last_borrower = spid;
}

void memory_partition_unit::arbitration_metadata::return_credit(
    int inner_sub_partition_id) {
  int spid = inner_sub_partition_id;
  if (m_private_credit[spid] > 0) {
    m_private_credit[spid] -= 1;
  } else {
    m_shared_credit -= 1;
  }
  assert((m_shared_credit >= 0) &&
         "DRAM arbitration error: Returning more than available credits!");
}

void memory_partition_unit::arbitration_metadata::print(FILE *fp) const {
  fprintf(fp, "private_credit = ");
  for (unsigned p = 0; p < m_private_credit.size(); p++) {
    fprintf(fp, "%d ", m_private_credit[p]);
  }
  fprintf(fp, "(limit = %d)\n", m_private_credit_limit);
  fprintf(fp, "shared_credit = %d (limit = %d)\n", m_shared_credit,
          m_shared_credit_limit);
}

bool memory_partition_unit::busy() const {
  bool busy = false;
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    if (m_sub_partition[p]->busy()) {
      busy = true;
    }
  }
  return busy;
}

void memory_partition_unit::cache_cycle(unsigned cycle) {
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    m_sub_partition[p]->cache_cycle(cycle);
  }
}

void memory_partition_unit::visualizer_print(gzFile visualizer_file) const {
  m_dram->visualizer_print(visualizer_file);
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    m_sub_partition[p]->visualizer_print(visualizer_file);
  }
}

// determine whether a given subpartition can issue to DRAM
bool memory_partition_unit::can_issue_to_dram(int inner_sub_partition_id) {
  int spid = inner_sub_partition_id;
  bool sub_partition_contention = m_sub_partition[spid]->dram_L2_queue_full();
  bool has_dram_resource = m_arbitration_metadata.has_credits(spid);

  MEMPART_DPRINTF(
      "sub partition %d sub_partition_contention=%c has_dram_resource=%c\n",
      spid, (sub_partition_contention) ? 'T' : 'F',
      (has_dram_resource) ? 'T' : 'F');

  return (has_dram_resource && !sub_partition_contention);
}

/*
m_id���ڴ������Ԫ���ڴ�ͨ������ID��m_n_sub_partition_per_memory_channel��ÿ���ڴ�ͨ�����ӷ�������
global_sub_partition_id���ڴ��ӷ�����ȫ��ID�������Ǽ��㵱ǰ�ڴ��ӷ����ı���ID������ǰ�ڴ��ӷ�����
��ǰ�ڴ�ͨ���еı���ID��
*/
int memory_partition_unit::global_sub_partition_id_to_local_id(
    int global_sub_partition_id) const {
  //m_id���ڴ������Ԫ���ڴ�ͨ������ID��m_n_sub_partition_per_memory_channel��ÿ���ڴ�ͨ�����ӷ�
  //������global_sub_partition_id���ڴ��ӷ�����ȫ��ID�������Ǽ��㵱ǰ�ڴ��ӷ����ı���ID������ǰ��
  //���ӷ����ڵ�ǰ�ڴ�ͨ���еı���ID��
  return (global_sub_partition_id -
          m_id * m_config->m_n_sub_partition_per_memory_channel);
}

void memory_partition_unit::simple_dram_model_cycle() {
  // pop completed memory request from dram and push it to dram-to-L2 queue
  // of the original sub partition
  if (!m_dram_latency_queue.empty() &&
      ((m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle) >=
       m_dram_latency_queue.front().ready_cycle)) {
    mem_fetch *mf_return = m_dram_latency_queue.front().req;
    if (mf_return->get_access_type() != L1_WRBK_ACC &&
        mf_return->get_access_type() != L2_WRBK_ACC) {
      mf_return->set_reply();

      unsigned dest_global_spid = mf_return->get_sub_partition_id();
      int dest_spid = global_sub_partition_id_to_local_id(dest_global_spid);
      assert(m_sub_partition[dest_spid]->get_id() == dest_global_spid);
      if (!m_sub_partition[dest_spid]->dram_L2_queue_full()) {
        if (mf_return->get_access_type() == L1_WRBK_ACC) {
          m_sub_partition[dest_spid]->set_done(mf_return);
          delete mf_return;
        } else {
          m_sub_partition[dest_spid]->dram_L2_queue_push(mf_return);
          mf_return->set_status(
              IN_PARTITION_DRAM_TO_L2_QUEUE,
              m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
          m_arbitration_metadata.return_credit(dest_spid);
          MEMPART_DPRINTF(
              "mem_fetch request %p return from dram to sub partition %d\n",
              mf_return, dest_spid);
        }
        m_dram_latency_queue.pop_front();
      }

    } else {
      this->set_done(mf_return);
      delete mf_return;
      m_dram_latency_queue.pop_front();
    }
  }

  // mem_fetch *mf = m_sub_partition[spid]->L2_dram_queue_top();
  // if( !m_dram->full(mf->is_write()) ) {
  // L2->DRAM queue to DRAM latency queue
  // Arbitrate among multiple L2 subpartitions
  int last_issued_partition = m_arbitration_metadata.last_borrower();
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    int spid = (p + last_issued_partition + 1) %
               m_config->m_n_sub_partition_per_memory_channel;
    if (!m_sub_partition[spid]->L2_dram_queue_empty() &&
        can_issue_to_dram(spid)) {
      mem_fetch *mf = m_sub_partition[spid]->L2_dram_queue_top();
      if (m_dram->full(mf->is_write())) break;

      m_sub_partition[spid]->L2_dram_queue_pop();
      MEMPART_DPRINTF(
          "Issue mem_fetch request %p from sub partition %d to dram\n", mf,
          spid);
      dram_delay_t d;
      d.req = mf;
      d.ready_cycle = m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle +
                      m_config->dram_latency;
      m_dram_latency_queue.push_back(d);
      mf->set_status(IN_PARTITION_DRAM_LATENCY_QUEUE,
                     m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
      m_arbitration_metadata.borrow_credit(spid);
      break;  // the DRAM should only accept one request per cycle
    }
  }
  //}
}

/*
���ڴ������L2->dram�����ƶ���DRAM Channel������DRAM�ķ�����������е����ݰ���DRAM Channel��dram->
L2���У�����Ƭ��GDDR3 DRAM�ڴ���ǰ�ƽ�һ�ġ�
*/
void memory_partition_unit::dram_cycle() {
  // pop completed memory request from dram and push it to dram-to-L2 queue
  // of the original sub partition
  //��DRAM��������ɵ��ڴ����󲢽������͵�ԭʼ�ӷ�����DRAM��L2���С�m_dram->return_queue_top()����
  //dram->returnq����������еĶ���Ԫ��mf_return���������Ϊ�գ��򷵻�NULL��
  mem_fetch *mf_return = m_dram->return_queue_top();
  if (mf_return) {
    //���mf_return��Ч�Ļ���˵��DRAM�Ѿ�����˶�mf_return�Ĵ������Խ����DRAM���ض����е�����
    unsigned dest_global_spid = mf_return->get_sub_partition_id();
    //���㵱ǰ�ڴ��ӷ����ı���ID������ǰ�ڴ��ӷ����ڵ�ǰ�ڴ�ͨ���еı���ID��
    int dest_spid = global_sub_partition_id_to_local_id(dest_global_spid);
    //m_sub_partition[dest_spid]->get_id()�����ڴ��ӷ�����ȫ��ID��
    assert(m_sub_partition[dest_spid]->get_id() == dest_global_spid);
    //���dest_spid����ʶ���ڴ��ӷ�����DRAM_to_L2����δ����
    if (!m_sub_partition[dest_spid]->dram_L2_queue_full()) {
      if (mf_return->get_access_type() == L1_WRBK_ACC) {
        //mf_return�ڴ�������L1д�صĻ���ֻ����������ɼ��ɡ�
        m_sub_partition[dest_spid]->set_done(mf_return);
        delete mf_return;
      } else {
        //��mf_return���͵�DRAM_to_L2�����С�
        m_sub_partition[dest_spid]->dram_L2_queue_push(mf_return);
        mf_return->set_status(IN_PARTITION_DRAM_TO_L2_QUEUE,
                              m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
        m_arbitration_metadata.return_credit(dest_spid);
        MEMPART_DPRINTF(
            "mem_fetch request %p return from dram to sub partition %d\n",
            mf_return, dest_spid);
      }
      //��������ɵ��ڴ�����
      m_dram->return_queue_pop();
    }
  } else {
    //���mf_return��Ч�Ļ���˵��������ݰ���Чֱ�ӵ������ϼ��ɡ�
    m_dram->return_queue_pop();
  }

  //DRAM��ǰ�ƽ�һ�ġ�
  m_dram->cycle();
  m_dram->dram_log(SAMPLELOG);

  // mem_fetch *mf = m_sub_partition[spid]->L2_dram_queue_top();
  // if( !m_dram->full(mf->is_write()) ) {
  // L2->DRAM queue to DRAM latency queue
  // Arbitrate among multiple L2 subpartitions
  int last_issued_partition = m_arbitration_metadata.last_borrower();
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    int spid = (p + last_issued_partition + 1) %
               m_config->m_n_sub_partition_per_memory_channel;
    if (!m_sub_partition[spid]->L2_dram_queue_empty() &&
        can_issue_to_dram(spid)) {
      mem_fetch *mf = m_sub_partition[spid]->L2_dram_queue_top();
      if (m_dram->full(mf->is_write())) break;

      m_sub_partition[spid]->L2_dram_queue_pop();
      MEMPART_DPRINTF(
          "Issue mem_fetch request %p from sub partition %d to dram\n", mf,
          spid);
      dram_delay_t d;
      d.req = mf;
      d.ready_cycle = m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle +
                      m_config->dram_latency;
      m_dram_latency_queue.push_back(d);
      mf->set_status(IN_PARTITION_DRAM_LATENCY_QUEUE,
                     m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
      m_arbitration_metadata.borrow_credit(spid);
      break;  // the DRAM should only accept one request per cycle
    }
  }
  //}

  // DRAM latency queue
  if (!m_dram_latency_queue.empty() &&
      ((m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle) >=
       m_dram_latency_queue.front().ready_cycle) &&
      !m_dram->full(m_dram_latency_queue.front().req->is_write())) {
    mem_fetch *mf = m_dram_latency_queue.front().req;
    m_dram_latency_queue.pop_front();
    m_dram->push(mf);
  }
}

void memory_partition_unit::set_done(mem_fetch *mf) {
  unsigned global_spid = mf->get_sub_partition_id();
  int spid = global_sub_partition_id_to_local_id(global_spid);
  assert(m_sub_partition[spid]->get_id() == global_spid);
  //��V100�У���L2 cacheд������ʱ����ȡlazy_fetch_on_read���ԣ����ҵ�һ��cache block
  //���ʱ��������cache block�Ǳ�MODIFIED������Ҫ�����cache blockд�ص���һ���洢��
  //��˻����L2_WRBK_ACC���ʣ�������ʾ���Ϊ��д�ر������MODIFIED cache block��
  if (mf->get_access_type() == L1_WRBK_ACC ||
      mf->get_access_type() == L2_WRBK_ACC) {
    m_arbitration_metadata.return_credit(spid);
    MEMPART_DPRINTF(
        "mem_fetch request %p return from dram to sub partition %d\n", mf,
        spid);
  }
  m_sub_partition[spid]->set_done(mf);
}

void memory_partition_unit::set_dram_power_stats(
    unsigned &n_cmd, unsigned &n_activity, unsigned &n_nop, unsigned &n_act,
    unsigned &n_pre, unsigned &n_rd, unsigned &n_wr, unsigned &n_wr_WB,
    unsigned &n_req) const {
  m_dram->set_dram_power_stats(n_cmd, n_activity, n_nop, n_act, n_pre, n_rd,
                               n_wr, n_wr_WB, n_req);
}

void memory_partition_unit::print(FILE *fp) const {
  fprintf(fp, "Memory Partition %u: \n", m_id);
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    m_sub_partition[p]->print(fp);
  }
  fprintf(fp, "In Dram Latency Queue (total = %zd): \n",
          m_dram_latency_queue.size());
  for (std::list<dram_delay_t>::const_iterator mf_dlq =
           m_dram_latency_queue.begin();
       mf_dlq != m_dram_latency_queue.end(); ++mf_dlq) {
    mem_fetch *mf = mf_dlq->req;
    fprintf(fp, "Ready @ %llu - ", mf_dlq->ready_cycle);
    if (mf)
      mf->print(fp);
    else
      fprintf(fp, " <NULL mem_fetch?>\n");
  }
  m_dram->print(fp);
}

/*
memory_sub_partition���캯����
*/
memory_sub_partition::memory_sub_partition(unsigned sub_partition_id, // �ڴ��ӷ�����ID
                                           const memory_config *config,
                                           class memory_stats_t *stats,
                                           class gpgpu_sim *gpu) {
  m_id = sub_partition_id;
  m_config = config;
  m_stats = stats;
  m_gpu = gpu;
  m_memcpy_cycle_offset = 0;

  //gpgpu_n_memΪ�����е��ڴ��������DRAM Channel������������Ϊ��
  //  option_parser_register(
  //      opp, "-gpgpu_n_mem", OPT_UINT32, &m_n_mem,
  //      "number of memory modules (e.g. memory controllers) in gpu", "8");
  //��V100�����У���32���ڴ��������DRAM Channel����m_n_mem_sub_partition�Ķ���Ϊ��
  //    m_n_mem_sub_partition = m_n_mem * m_n_sub_partition_per_memory_channel;
  assert(m_id < m_config->m_n_mem_sub_partition);

  char L2c_name[32];
  snprintf(L2c_name, 32, "L2_bank_%03d", m_id);
  m_L2interface = new L2interface(this);
  m_mf_allocator = new partition_mf_allocator(config);

  if (!m_config->m_L2_config.disabled())
    m_L2cache = new l2_cache(L2c_name, m_config->m_L2_config, -1, -1,
                             m_L2interface, m_mf_allocator,
                             IN_PARTITION_L2_MISS_QUEUE, gpu, L2_GPU_CACHE);

  unsigned int icnt_L2;
  unsigned int L2_dram;
  unsigned int dram_L2;
  unsigned int L2_icnt;
  //��ЩDelay Queue������ģ������������Ĭ��Ϊ8��8��8��8��
  sscanf(m_config->gpgpu_L2_queue_config, "%u:%u:%u:%u", &icnt_L2, &L2_dram,
         &dram_L2, &L2_icnt);
  //��Щqueue��GPGPU-Sim v3.0�ֲ��е�#�ڴ���������н��ܡ�
  //�ڴ��������ݰ�ͨ��ICNT->L2 queue�ӻ�����������ڴ������L2 Cache Bank��ÿ��L2ʱ�����ڴ�ICNT->L2 
  //queue����һ��������з���L2���ɵ�оƬ��DRAM���κ��ڴ����󶼱�����L2->DRAM queue�����L2 Cache
  //�����ã����ݰ�����ICNT->L2 queue��������ֱ������L2->DRAM queue����Ȼ��L2ʱ��Ƶ�ʡ���Ƭ��DRAM����
  //����������DRAM->L2 queue����������L2 Cache Bank���ġ���L2��SIMT Core�Ķ���Ӧͨ��L2->ICNT que-
  //ue���͡�
  //Delay Queue. See http://gpgpu-sim.org/manual/images/0/0e/Mempart-arch.png.
  //Param of fifo_pipeline:
  //  nm�����е�name���ַ�����
  //  minlen�����е���С���ȡ�
  //  maxlen�����е���󳤶ȡ�
  m_icnt_L2_queue = new fifo_pipeline<mem_fetch>("icnt-to-L2", 0, icnt_L2);
  m_L2_dram_queue = new fifo_pipeline<mem_fetch>("L2-to-dram", 0, L2_dram);
  m_dram_L2_queue = new fifo_pipeline<mem_fetch>("dram-to-L2", 0, dram_L2);
  m_L2_icnt_queue = new fifo_pipeline<mem_fetch>("L2-to-icnt", 0, L2_icnt);
  wb_addr = -1;
}

memory_sub_partition::~memory_sub_partition() {
  delete m_icnt_L2_queue;
  delete m_L2_dram_queue;
  delete m_dram_L2_queue;
  delete m_L2_icnt_queue;
  delete m_L2cache;
  delete m_L2interface;
}

/*
�Զ�������Bank���м�ʱ����������������Ƴ��������档���潫����memory_partition_unit::cache_cycle()��
�ڲ��ṹ��
*/
void memory_sub_partition::cache_cycle(unsigned cycle) {
  // L2 fill responses
  //��V100�����ļ��У�L2 Cache��δ���á�
  if (!m_config->m_L2_config.disabled()) {
    //L2 Cache�ڲ���MSHRά����һ�������ڴ���ʵ��б�m_current_response��m_L2cache->access_ready()��
    //�ص���m_current_response�ǿգ������m_current_response�ǿգ�˵��L2 Cache��MSHR���о������ڴ��
    //�ʡ�m_L2_icnt_queue������Ҫ���ֲ��еĵ��������ڴ��������ϸϸ��ͼ��memory_sub_partition������
    //���Ƴ����ݰ��Ľӿھ���L2_icnt_queue->ICNT������������ж��ڴ��ӷ����е�m_L2_icnt_queue�����Ƿ��
    //�������������˵���������������Ƴ����ݰ���m_current_response���洢�˾����ڴ���ʵĵ�ַ��
    //δ����״̬���ּĴ�����the miss status holding register��MSHR��MSHR��ģ������mshr_table����ģ��
    //һ���������������ĺϲ��������ȫ����������ͨ��next_access()������MSHR���ͷš�MSHR����й̶�����
    //��MSHR��Ŀ��ÿ��MSHR��Ŀ����Ϊ���������У�Cache Line���ṩ�̶�������δ��������MSHR��Ŀ��������
    //ÿ����Ŀ������������ǿ����õġ�
    //����δ����״̬���ּĴ������������к󣬽�������Ĵ����ļ��������ݣ������������ڻ���δ����ʱ��δ����
    //�����߼������ȼ��δ����״̬���ּĴ�����MSHR�����Բ鿴��ǰ�Ƿ���������ǰ�������ͬ�����������ǣ�
    //������󽫺ϲ���ͬһ��Ŀ�У����Ҳ���Ҫ�����µ��������󡣷��򣬽�Ϊ������������һ���µ�MSHR��Ŀ�ͻ�
    //���С�����״̬���������ܻ�����Դ������ʱʧ�ܣ�����û�п��õ�MSHR��Ŀ���ü��е����л���鶼�ѱ�����
    //��δ��䡢δ���ж��������ȡ�
    //����m_mshrs.access_ready()���ص��Ǿ����ڴ���ʵ��б�m_current_response�Ƿ�ǿգ������ڴ���ʵ���
    //����洢�˾����ڴ���ʵĵ�ַ����������Ѿ�������MSHR��Ŀ�ķ��ʣ��򷵻�true��
    //��cache����fill����ʱ�����ݰ�fill��cache��ʱ���ã���һ�����ݰ����˺󣬸������ݰ����ص������Ѿ�������
    //��˽�����뵽MSHR��m_current_response�С�
    if (m_L2cache->access_ready() && !m_L2_icnt_queue->full()) {
      //m_L2cache->next_access()����MSHR��next_access()����һ���������ڴ���ʣ���m_current_response
      //�еĶ�����ַ��־�����ݰ���m_current_response���洢�˾����ڴ���ʵĵ�ַ����
      mem_fetch *mf = m_L2cache->next_access();
      //mem_access_type��������ʱ��ģ�����жԲ�ͬ���͵Ĵ洢�����в�ͬ�ķô����ͣ�
      //    MA_TUP(GLOBAL_ACC_R),        ��global memory��
      //    MA_TUP(LOCAL_ACC_R),         ��local memory��
      //    MA_TUP(CONST_ACC_R),         �ӳ��������
      //    MA_TUP(TEXTURE_ACC_R),       ���������
      //    MA_TUP(GLOBAL_ACC_W),        ��global memoryд
      //    MA_TUP(LOCAL_ACC_W),         ��local memoryд
      //��V100�У�L1 cache��m_write_policyΪWRITE_THROUGH��ʵ����L1_WRBK_ACCҲ�����õ���
      //    MA_TUP(L1_WRBK_ACC),         L1����write back
      //��V100�У���L2 cacheд������ʱ����ȡlazy_fetch_on_read���ԣ����ҵ�һ��cache block
      //���ʱ��������cache block�Ǳ�MODIFIED������Ҫ�����cache blockд�ص���һ���洢��
      //��˻����L2_WRBK_ACC���ʣ�������ʾ���Ϊ��д�ر������MODIFIED cache block��
      //    MA_TUP(L2_WRBK_ACC),         L2����write back
      //    MA_TUP(INST_ACC_R),          ��ָ����
      //L1_WR_ALLOC_R/L2_WR_ALLOC_R��V100��������ʱ�ò�����
      //    MA_TUP(L1_WR_ALLOC_R),       L1����write-allocate��cacheд�����У��������п����cache��
      //                                 д���cache�飩
      //L1_WR_ALLOC_R/L2_WR_ALLOC_R��V100��������ʱ�ò�����
      //    MA_TUP(L2_WR_ALLOC_R),       L2����write-allocate��cacheд�����У��������п����cache��
      //                                 д���cache�飩
      //    MA_TUP(NUM_MEM_ACCESS_TYPE), �洢�����ʵ���������
      if (mf->get_access_type() !=
          L2_WR_ALLOC_R) {  // Don't pass write allocate read request back to
                            // upper level cache
        // �ο���https://blog.csdn.net/qq_41587740/article/details/109104962
        //��ǰ��������L2���棬���mf��������L2_WR_ALLOC_R��˵��L2���淢����д�����У���Ҫ�������п��
        //��L2������д��ÿ飬���mf��������L2_WR_ALLOC_Rʱ�����ܽ�mf����ICNT���ͣ�������������L2���ɡ�
        //set_reply()���������ڴ����������Ӧ�����ͣ��ڴ���������а����������ͣ�������д���󡢶���Ӧ��
        //дȷ�ϡ����������ö���Ӧ������дȷ�ϡ�
        // void set_reply() {
        //       assert(m_access.get_type() != L1_WRBK_ACC &&
        //             m_access.get_type() != L2_WRBK_ACC);
        //       //����ڴ��������������Ƕ����󣬽�������Ϊ����Ӧ��
        //       if (m_type == READ_REQUEST) {
        //         assert(!get_is_write());
        //         m_type = READ_REPLY;
        //       //����ڴ���������������д���󣬽�������Ϊдȷ�ϡ�
        //       } else if (m_type == WRITE_REQUEST) {
        //         assert(get_is_write());
        //         m_type = WRITE_ACK;
        //       }
        //     }
        mf->set_reply();
        mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,
                       m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
        //������L2 Cache�����˾����ڴ���ʣ��Ϳ����������þ����ڴ���ʣ���L2_WR_ALLOC_R���󣩷��͵�
        //ICNT����mf����L2���浽ICNT�Ķ���m_L2_icnt_queue��
        m_L2_icnt_queue->push(mf);
      } else {
        //��ǰ��������L2���棬���mf��������L2_WR_ALLOC_R��˵��L2���淢����д�����У���Ҫ�������п��
        //��L2������д��ÿ飬���mf��������L2_WR_ALLOC_Rʱ�����ܽ�mf����ICNT���͡�
        //FETCH_ON_WRITE ��һ��д���䣨write allocate�������е�һ��ѡ�д������ָ��д���������ʱ����
        //��Ŀ���ַ���ڻ����У��Ὣ�õ�ַ�����ݴ��ڴ��ж�ȡ�������У�Ȼ���ٽ���д�������FETCH_ON_WRITE 
        //��ָ��д���������ʱ��ִ�ж�ȡ������Ҳ�����ڽ���д��֮ǰ�ȴ��ڴ��л�ȡ���ݡ�������Ե��ŵ����ܹ�
        //����д��������Ҫ���ڴ���ʴ������Ӷ������ӳ١���д�����Ƶ��ʱ��ʹ�� FETCH_ON_WRITE ���Կ�����
        //Ч����߻�������ܡ�V100�����У�m_L2_config.m_write_alloc_policy������ΪLAZY_FETCH_ON_READ��
        //�����if�鲻��Ч��
        if (m_config->m_L2_config.m_write_alloc_policy == FETCH_ON_WRITE) {
          //��ʹ��fetch-on-write����ʱ��mf->get_original_wr_mf()ָ��ָ��ԭʼд����
          mem_fetch *original_wr_mf = mf->get_original_wr_mf();
          assert(original_wr_mf);
          //set_reply()���������ڴ����������Ӧ�����ͣ��ڴ���������а����������ͣ�������д���󡢶���Ӧ��
          //дȷ�ϡ����������ö���Ӧ������дȷ�ϡ�
          // void set_reply() {
          //       assert(m_access.get_type() != L1_WRBK_ACC &&
          //             m_access.get_type() != L2_WRBK_ACC);
          //       //����ڴ��������������Ƕ����󣬽�������Ϊ����Ӧ��
          //       if (m_type == READ_REQUEST) {
          //         assert(!get_is_write());
          //         m_type = READ_REPLY;
          //       //����ڴ���������������д���󣬽�������Ϊдȷ�ϡ�
          //       } else if (m_type == WRITE_REQUEST) {
          //         assert(get_is_write());
          //         m_type = WRITE_ACK;
          //       }
          //     }
          original_wr_mf->set_reply();
          original_wr_mf->set_status(
              IN_PARTITION_L2_TO_ICNT_QUEUE,
              m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
          m_L2_icnt_queue->push(original_wr_mf);
        }
        //V100�����У�m_L2_config.m_write_alloc_policy������ΪLAZY_FETCH_ON_READ�������if�鲻��Ч��
        m_request_tracker.erase(mf);
        delete mf;
      }
    }
  }

  // DRAM to L2 (texture) and icnt (not texture)
  //���m_dram_L2_queue�ǿգ��ͻ�ȡm_dram_L2_queue�Ķ������ݰ�����������L2 Cache��
  if (!m_dram_L2_queue->empty()) {
    //��ȡm_dram_L2_queue�Ķ������ݰ���
    mem_fetch *mf = m_dram_L2_queue->top();
    // m_L2cache->waiting_for_fill(mf) checks if mf is waiting to be filled by lower memory level.
    //����Ƿ�mf���ڵȴ����͵Ĵ洢�����䡣waiting_for_fill(mem_fetch *mf)�Ķ���Ϊ��
    //     bool baseline_cache::waiting_for_fill(mem_fetch *mf) {
    //       //extra_mf_fields_lookup�Ķ��壺
    //       //  typedef std::map<mem_fetch *, extra_mf_fields> extra_mf_fields_lookup;
    //       //��cache������������mfʱ�����δ���У�����MSHR��Ҳδ���У�û��mf��Ŀ����������뵽MSHR�У�
    //       //ͬʱ������m_extra_mf_fields[mf]����ζ�����mf��m_extra_mf_fields�д��ڣ���mf�ȴ���DRAM
    //       //�����ݻص�L2������䣺
    //       //m_extra_mf_fields[mf] = extra_mf_fields(
    //       //      mshr_addr, mf->get_addr(), cache_index, mf->get_data_size(), m_config);
    //       extra_mf_fields_lookup::iterator e = m_extra_mf_fields.find(mf);
    //       return e != m_extra_mf_fields.end();
    //     }
    //��m_L2cache->waiting_for_fill(mf)Ϊ��˵���˴�L2�����MSHR�д���mf��Ŀ�����ڵȴ�DRAM���ص�������䡣
    if (!m_config->m_L2_config.disabled() && m_L2cache->waiting_for_fill(mf)) {
      if (m_L2cache->fill_port_free()) {
        mf->set_status(IN_PARTITION_L2_FILL_QUEUE,
                       m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
        //��m_dram_L2_queue�Ķ������ݰ�����L2 Cache��
        m_L2cache->fill(mf, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle +
                                m_memcpy_cycle_offset);
        //��m_dram_L2_queue�Ķ������ݰ�������
        m_dram_L2_queue->pop();
      }
    } else if (!m_L2_icnt_queue->full()) {
      //���m_L2cache->waiting_for_fill(mf)��Ϊ�棬��˵��L2�����MSHR�в�����mf��Ŀ�����ڵȴ�DRAM����
      //��������䣬��ô�Ϳ���ֱ�ӽ�mf���ͻ�ICNT��
      if (mf->is_write() && mf->get_type() == WRITE_ACK)
        mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,
                       m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
      //�����ݰ�mf����m_L2_icnt_queue���С�
      m_L2_icnt_queue->push(mf);
      m_dram_L2_queue->pop();
    }
  }

  // prior L2 misses inserted into m_L2_dram_queue here
  //L2������ǰ�ƽ�һ�ġ�
  if (!m_config->m_L2_config.disabled()) m_L2cache->cycle();

  // new L2 texture accesses and/or non-texture accesses
  //���L2��DRAM�Ķ��в�������ICNT��L2�Ķ��в��գ��ͽ�ICNT��L2�Ķ��еĶ������ݰ�����������L2 Cache��
  if (!m_L2_dram_queue->full() && !m_icnt_L2_queue->empty()) {
    //��ICNT��L2�Ķ��еĶ������ݰ�������
    mem_fetch *mf = m_icnt_L2_queue->top();
    //��V100�����У�-gpgpu_cache:dl2_texture_only������Ϊ0��
    if (!m_config->m_L2_config.disabled() &&
        ((m_config->m_L2_texure_only && mf->istexture()) ||
         (!m_config->m_L2_texure_only))) {
      // L2 is enabled and access is for L2
      //L2���汻���ã����ҷ��������L2�ġ�
      //m_L2_icnt_queue->full()�ж�L2������ICNT�Ķ����Ƿ�����
      bool output_full = m_L2_icnt_queue->full();
      //m_L2cache->data_port_free()�ж�L2��������ݶ˿��Ƿ���С�
      bool port_free = m_L2cache->data_port_free();
      //���L2������ICNT�Ķ��в�������L2��������ݶ˿ڿ��У�ICNT��L2�Ķ��еĶ������ݰ�����L2���ݷ��ʡ�
      if (!output_full && port_free) {
        std::list<cache_event> events;
        //��ICNT��L2�Ķ��еĶ������ݰ�����L2���ݷ��ʣ���ȡ���ʵ�״̬��
        enum cache_request_status status =
            m_L2cache->access(mf->get_addr(), mf,
                              m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle +
                                  m_memcpy_cycle_offset,
                              events);
        //�ж�һϵ�еķ���cache�¼�events�Ƿ����WRITE_REQUEST_SENT��
        //�����¼����Ͱ�����
        // enum cache_event_type {
        //       //д������
        //       WRITE_BACK_REQUEST_SENT,
        //       //������
        //       READ_REQUEST_SENT,
        //       //д����
        //       WRITE_REQUEST_SENT,
        //       //д��������
        //       WRITE_ALLOCATE_SENT
        //     };
        bool write_sent = was_write_sent(events);
        //�ж�һϵ�еķ���cache�¼�events�Ƿ����READ_REQUEST_SENT��
        bool read_sent = was_read_sent(events);
        MEM_SUBPART_DPRINTF("Probing L2 cache Address=%llx, status=%u\n",
                            mf->get_addr(), status);

        if (status == HIT) {
          //�������L2�������С�
          if (!write_sent) {
            //�������д����������L2 Cache������Ҫ�ж��Ƿ���L1_WRBK_ACC��
            //��V100�У�L1 cache��m_write_policyΪWRITE_THROUGH��ʵ����L1_WRBK_ACCҲ�����õ���
            // L2 cache replies
            assert(!read_sent);
            //!write_sent��!read_sent�����͵���WRITE_BACK_REQUEST_SENT/WRITE_ALLOCATE_SENT��
            if (mf->get_access_type() == L1_WRBK_ACC) {
              m_request_tracker.erase(mf);
              delete mf;
            } else {
              //�������L1_WRBK_ACC����˵�������ݶ�������Ҫ��reply���ݰ����ظ�ICNT��
              //��V100�У�L1 cache��m_write_policyΪWRITE_THROUGH��ʵ����L1_WRBK_ACCҲ�����õ���
              mf->set_reply();
              mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,
                             m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
              m_L2_icnt_queue->push(mf);
            }
            //��ICNT��L2�Ķ����е�����
            m_icnt_L2_queue->pop();
          } else {
            assert(write_sent);
            //�����д����������L2 Cache����ֱ�Ӵ�ICNT��L2�Ķ����е���������ݰ����ɡ�
            m_icnt_L2_queue->pop();
          }
        } else if (status != RESERVATION_FAIL) {
          //�������L2���治�������Ҳ��Ǳ���ʧ�ܣ�����HIT_RESERVED/MISS/SECTOR_MISS/MSHR_HIT��
          if (mf->is_write() &&
              //V100�����У�m_L2_config.m_write_alloc_policy������ΪLAZY_FETCH_ON_READ��
              (m_config->m_L2_config.m_write_alloc_policy == FETCH_ON_WRITE ||
               m_config->m_L2_config.m_write_alloc_policy ==
                   LAZY_FETCH_ON_READ) &&
              !was_writeallocate_sent(events)) {
            //��V100�У�L1 cache��m_write_policyΪWRITE_THROUGH��ʵ����L1_WRBK_ACCҲ�����õ���
            if (mf->get_access_type() == L1_WRBK_ACC) {
              m_request_tracker.erase(mf);
              delete mf;
            } else if (m_config->m_L2_config.get_write_policy() == WRITE_BACK) {
              mf->set_reply();
              mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,
                             m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
              m_L2_icnt_queue->push(mf);
            }
          }
          // L2 cache accepted request
          m_icnt_L2_queue->pop();
        } else {
          assert(!write_sent);
          assert(!read_sent);
          // L2 cache lock-up: will try again next cycle
        }
      }
    } else {
      // L2 is disabled or non-texture access to texture-only L2
      //L2���汻���û��߷��������texture-only L2�����ǲ����������������Ϊ��V100�����У�ѡ��
      //-gpgpu_cache:dl2_texture_only������Ϊ0��
      mf->set_status(IN_PARTITION_L2_TO_DRAM_QUEUE,
                     m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
      m_L2_dram_queue->push(mf);
      m_icnt_L2_queue->pop();
    }
  }

  // ROP delay queue
  //��դ������ˮ�ߣ�Raster Operations Pipeline��ROP���ӳٶ��С�
  if (!m_rop.empty() && (cycle >= m_rop.front().ready_cycle) &&
      !m_icnt_L2_queue->full()) {
    mem_fetch *mf = m_rop.front().req;
    m_rop.pop();
    m_icnt_L2_queue->push(mf);
    mf->set_status(IN_PARTITION_ICNT_TO_L2_QUEUE,
                   m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
  }
}

bool memory_sub_partition::full() const { return m_icnt_L2_queue->full(); }

/*
������Ҫ���ֲ��еĵ��������ڴ��������ϸϸ��ͼ������������memory_sub_partition�Ƴ����ݰ��Ľ�
�ھ���ICNT->icnt_L2_queue������������ж��ڴ��ӷ����е�m_icnt_L2_queue�����Ƿ���Է���size
��С�����ݣ����Է��·���False���Ų��·���True��
*/
bool memory_sub_partition::full(unsigned size) const {
  return m_icnt_L2_queue->is_avilable_size(size);
}

bool memory_sub_partition::L2_dram_queue_empty() const {
  return m_L2_dram_queue->empty();
}

class mem_fetch *memory_sub_partition::L2_dram_queue_top() const {
  return m_L2_dram_queue->top();
}

void memory_sub_partition::L2_dram_queue_pop() { m_L2_dram_queue->pop(); }

bool memory_sub_partition::dram_L2_queue_full() const {
  return m_dram_L2_queue->full();
}

void memory_sub_partition::dram_L2_queue_push(class mem_fetch *mf) {
  m_dram_L2_queue->push(mf);
}

void memory_sub_partition::print_cache_stat(unsigned &accesses,
                                            unsigned &misses) const {
  FILE *fp = stdout;
  if (!m_config->m_L2_config.disabled()) m_L2cache->print(fp, accesses, misses);
}

void memory_sub_partition::print(FILE *fp) const {
  if (!m_request_tracker.empty()) {
    fprintf(fp, "Memory Sub Parition %u: pending memory requests:\n", m_id);
    for (std::set<mem_fetch *>::const_iterator r = m_request_tracker.begin();
         r != m_request_tracker.end(); ++r) {
      mem_fetch *mf = *r;
      if (mf)
        mf->print(fp);
      else
        fprintf(fp, " <NULL mem_fetch?>\n");
    }
  }
  if (!m_config->m_L2_config.disabled()) m_L2cache->display_state(fp);
}

void memory_stats_t::visualizer_print(gzFile visualizer_file) {
  gzprintf(visualizer_file, "Ltwowritemiss: %d\n", L2_write_miss);
  gzprintf(visualizer_file, "Ltwowritehit: %d\n", L2_write_hit);
  gzprintf(visualizer_file, "Ltworeadmiss: %d\n", L2_read_miss);
  gzprintf(visualizer_file, "Ltworeadhit: %d\n", L2_read_hit);
  clear_L2_stats_pw();

  if (num_mfs)
    gzprintf(visualizer_file, "averagemflatency: %lld\n",
             mf_total_lat / num_mfs);
}

void memory_stats_t::clear_L2_stats_pw() {
  L2_write_miss = 0;
  L2_write_hit = 0;
  L2_read_miss = 0;
  L2_read_hit = 0;
}

void gpgpu_sim::print_dram_stats(FILE *fout) const {
  unsigned cmd = 0;
  unsigned activity = 0;
  unsigned nop = 0;
  unsigned act = 0;
  unsigned pre = 0;
  unsigned rd = 0;
  unsigned wr = 0;
  unsigned wr_WB = 0;
  unsigned req = 0;
  unsigned tot_cmd = 0;
  unsigned tot_nop = 0;
  unsigned tot_act = 0;
  unsigned tot_pre = 0;
  unsigned tot_rd = 0;
  unsigned tot_wr = 0;
  unsigned tot_req = 0;

  for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
    m_memory_partition_unit[i]->set_dram_power_stats(cmd, activity, nop, act,
                                                     pre, rd, wr, wr_WB, req);
    tot_cmd += cmd;
    tot_nop += nop;
    tot_act += act;
    tot_pre += pre;
    tot_rd += rd;
    tot_wr += wr + wr_WB;
    tot_req += req;
  }
  fprintf(fout, "gpgpu_n_dram_reads = %d\n", tot_rd);
  fprintf(fout, "gpgpu_n_dram_writes = %d\n", tot_wr);
  fprintf(fout, "gpgpu_n_dram_activate = %d\n", tot_act);
  fprintf(fout, "gpgpu_n_dram_commands = %d\n", tot_cmd);
  fprintf(fout, "gpgpu_n_dram_noops = %d\n", tot_nop);
  fprintf(fout, "gpgpu_n_dram_precharges = %d\n", tot_pre);
  fprintf(fout, "gpgpu_n_dram_requests = %d\n", tot_req);
}

unsigned memory_sub_partition::flushL2() {
  if (!m_config->m_L2_config.disabled()) {
    m_L2cache->flush();
  }
  return 0;  // TODO: write the flushed data to the main memory
}

unsigned memory_sub_partition::invalidateL2() {
  if (!m_config->m_L2_config.disabled()) {
    m_L2cache->invalidate();
  }
  return 0;
}

bool memory_sub_partition::busy() const { return !m_request_tracker.empty(); }

/*
cache_config�ĵ�һ����ĸ����cache����������λ�������"N"�����Normal�������"S"�����Sector��
Normalģʽ��ʵ������Ƕ��������Set-Associative��ɽṹ����Sectorģʽ�������cache�������һ��
Sector Buffer��ɽṹ����V100�������ļ��У�
    -gpgpu_cache:dl1  S:4:128:64,L:T:m:L:L,A:512:8,16:0,32
    -gpgpu_cache:dl2  S:32:128:24,L:B:m:L:P,A:192:4,32:0,32
    -gpgpu_cache:il1  N:64:128:16,L:R:f:N:L,S:2:48,4
���L1 Data Cache��L2 Data Cache����Sectorģʽ����L1 Instruction Cache��Normalģʽ���򵥽���
Sector Buffer��ɽṹ���ٶ���һ��΢�ܹ��У�Cache��СΪ16KB��ʹ��Sector Buffer��ʽʱ�����16KB
���ֽ�Ϊ16��1KB��С��Sector��CPU����ͬʱ������16��Sector�������ʵ����ݲ�����16��Sector������ʱ��
�����Ƚ���Sector��̭�������ڻ��һ���µ�Sector�󣬽�������Ҫ���ʵ�64B�����������Sector�������
�ʵ�����������ĳ��Sector���������ݲ���������Sectorʱ������Ӧ�����ݼ����������Sector�С���������
Sector Buffer����ʱ��Cache�Ļ������Ƚ�Ϊ���ԣ��Գ���ľֲ��Ե�Ҫ����ߡ�Cache�����������ʲ����
��Set-Associative����ɷ�ʽ��

�������L2 Cache��Sector Bufferģʽ������������m_req���Ϊ���Sector����
*/
std::vector<mem_fetch *>
memory_sub_partition::breakdown_request_to_sector_requests(mem_fetch *mf) {
  std::vector<mem_fetch *> result;
  //��ȡ��������mf��byte mask�������byte mask�����ڱ��һ�ηô�����е��������룬4��������ÿ��
  //����32���ֽ����ݣ����sector_mask��һ�����128 byte���ݵ����룬��128λbitset��
  //    typedef std::bitset<SECTOR_CHUNCK_SIZE> mem_access_sector_mask_t;
  //    const unsigned SECTOR_CHUNCK_SIZE = 4;  // four sectors
  //    const unsigned SECTOR_SIZE = 32;        // sector is 32 bytes width
  mem_access_sector_mask_t sector_mask = mf->get_access_sector_mask();
  if (mf->get_data_size() == SECTOR_SIZE &&
      mf->get_access_sector_mask().count() == 1) {
    //�����������Ĵ�С������һ��SECTOR����ֻ��һ��SECTOR�����ʣ�����Ҫ��֣�ֱ�ӽ�������ѹ��
    //result��mf->get_access_sector_mask().count()=1ʱ��˵��ֻ������һλΪ1����ֻ��һ���ֽڱ�
    //���ʣ������һ���ֽ�һ��λ�ڵ��������С�
    result.push_back(mf);
  } else if (mf->get_data_size() == MAX_MEMORY_ACCESS_SIZE) {
    // break down every sector
    //MAX_MEMORY_ACCESS_SIZE����Ϊ��const unsigned MAX_MEMORY_ACCESS_SIZE = 128�������mask
    //Ҳ�����ڱ��һ�ηô�����е������ֽ����룬MAX_MEMORY_ACCESS_SIZE����Ϊ128����ÿ�ηô����
    //����128�ֽڣ���128λbitset��mem_access_byte_mask_t�Ķ���Ϊ��
    //    typedef std::bitset<MAX_MEMORY_ACCESS_SIZE> mem_access_byte_mask_t;
    mem_access_byte_mask_t mask;
    //����һ��MAX_MEMORY_ACCESS_SIZE=128�ֽڴ�С��������˵���ܹ���Ϊ4��SECTOR_CHUNCK���һ��ֵ�
    //SECTOR�Ĵ�СΪSECTOR_SIZE=32����������4��SECTOR_CHUNCKѭ����Ŀ���ǽ�һ��128�ֽڵĴ�����
    //����ΪSECTOR_CHUNCK_SIZE = 4������������
    for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; i++) {
      //k������ǵ�i��SECTOR_CHUNCK�е��ֽڱ�ŷ�Χ����0��CHUNCK��k�ķ�Χ��[0,32)����1��CHUNCK
      //��k�ķ�Χ��[32,64)����2��CHUNCK��k�ķ�Χ��[64,96)����3��CHUNCK��k�ķ�Χ��[96,128)��
      for (unsigned k = i * SECTOR_SIZE; k < (i + 1) * SECTOR_SIZE; k++) {
        //��i��SECTOR_CHUNCK�е��ֽ�mask����Ϊk�ķ�Χ��Ӧλ��
        mask.set(k);
      }
      //����i��SECTOR_CHUNCK�е��ֽ�mask��mf���ֽ�mask������������õ���i��SECTOR_CHUNCK������
      mem_fetch *n_mf = m_mf_allocator->alloc(
          mf->get_addr() + SECTOR_SIZE * i, mf->get_access_type(),
          mf->get_access_warp_mask(), mf->get_access_byte_mask() & mask,
          std::bitset<SECTOR_CHUNCK_SIZE>().set(i), SECTOR_SIZE, mf->is_write(),
          m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, mf->get_wid(),
          mf->get_sid(), mf->get_tpc(), mf, mf->get_streamID());
      //����i��SECTOR_CHUNCK������ѹ��result��
      result.push_back(n_mf);
    }
    // This is for constant cache
  } else if (mf->get_data_size() == 64 &&
             (mf->get_access_sector_mask().all() ||
              mf->get_access_sector_mask().none())) {
    unsigned start;
    if (mf->get_addr() % MAX_MEMORY_ACCESS_SIZE == 0)
      start = 0;
    else
      start = 2;
    mem_access_byte_mask_t mask;
    for (unsigned i = start; i < start + 2; i++) {
      for (unsigned k = i * SECTOR_SIZE; k < (i + 1) * SECTOR_SIZE; k++) {
        mask.set(k);
      }
      mem_fetch *n_mf = m_mf_allocator->alloc(
          mf->get_addr(), mf->get_access_type(), mf->get_access_warp_mask(),
          mf->get_access_byte_mask() & mask,
          std::bitset<SECTOR_CHUNCK_SIZE>().set(i), SECTOR_SIZE, mf->is_write(),
          m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, mf->get_wid(),
          mf->get_sid(), mf->get_tpc(), mf, mf->get_streamID());

      result.push_back(n_mf);
    }
  } else {
    for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; i++) {
      if (sector_mask.test(i)) {
        mem_access_byte_mask_t mask;
        for (unsigned k = i * SECTOR_SIZE; k < (i + 1) * SECTOR_SIZE; k++) {
          mask.set(k);
        }
        mem_fetch *n_mf = m_mf_allocator->alloc(
            mf->get_addr() + SECTOR_SIZE * i, mf->get_access_type(),
            mf->get_access_warp_mask(), mf->get_access_byte_mask() & mask,
            std::bitset<SECTOR_CHUNCK_SIZE>().set(i), SECTOR_SIZE,
            mf->is_write(), m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle,
            mf->get_wid(), mf->get_sid(), mf->get_tpc(), mf,
            mf->get_streamID());

        result.push_back(n_mf);
      }
    }
  }
  if (result.size() == 0) assert(0 && "no mf sent");
  return result;
}

/*
�����ݶ�����m_req�ӻ����������뵽�ڴ��ӷ��������к���ȡ���ݴ���
*/
void memory_sub_partition::push(mem_fetch *m_req, unsigned long long cycle) {
  if (m_req) {
    m_stats->memlatstat_icnt2mem_pop(m_req);
    std::vector<mem_fetch *> reqs;
    //cache_config�ĵ�һ����ĸ����cache����������λ�������"N"�����Normal�������"S"��
    //����Sector��Normalģʽ��ʵ������Ƕ��������Set-Associative��ɽṹ����Sectorģʽ��
    //�����cache����һ��Sector Buffer��ɽṹ����V100�������ļ��У�
    //    -gpgpu_cache:dl1  S:4:128:64,L:T:m:L:L,A:512:8,16:0,32
    //    -gpgpu_cache:dl2  S:32:128:24,L:B:m:L:P,A:192:4,32:0,32
    //    -gpgpu_cache:il1  N:64:128:16,L:R:f:N:L,S:2:48,4
    //���L1 Data Cache��L2 Data Cache����Sectorģʽ����L1 Instruction Cache��Normalģʽ��
    //�򵥽���Sector Buffer��ɽṹ���ٶ���һ��΢�ܹ��У�Cache��СΪ16KB��ʹ��Sector Buffer
    //��ʽʱ�����16KB���ֽ�Ϊ16��1KB��С��Sector��CPU����ͬʱ������16��Sector�������ʵ�����
    //������16��Sector������ʱ�������Ƚ���Sector��̭�������ڻ��һ���µ�Sector�󣬽�������Ҫ
    //���ʵ�64B�����������Sector��������ʵ�����������ĳ��Sector���������ݲ���������Sectorʱ��
    //����Ӧ�����ݼ����������Sector�С��������ַ���ʱ��Cache�Ļ������Ƚ�Ϊ���ԣ��Գ���ľֲ�
    //�Ե�Ҫ����ߡ�Cache�����������ʲ������Set-Associative����ɷ�ʽ��

    //�������L2 Cache��Sector Bufferģʽ������������m_req���Ϊ���Sector����
    if (m_config->m_L2_config.m_cache_type == SECTOR)
      reqs = breakdown_request_to_sector_requests(m_req);
    else
      reqs.push_back(m_req);

    //����ÿ�����󣬽���ѹ��m_icnt_L2_queue���У����������ʣ��� m_rop��Raster Operations 
    //Pipeline��ROP���У���Է�������������ڴ��������ݰ�ͨ��ICNT->L2 queue�ӻ�����������ڴ�
    //��������GT200΢��׼�����о����۲쵽�ģ����������ͨ����դ������ˮ�ߣ�Raster Operations 
    //Pipeline��ROP�����н��У���ģ��460 L2ʱ�����ڵ���С��ˮ���ӳ١�L2 Cache Bank��ÿ��L2ʱ
    //�����ڴ�ICNT->L2 queue����һ��������з���L2���ɵ�оƬ��DRAM���κ��ڴ����󶼱�����L2->
    //DRAM queue�����L2 Cache�����ã����ݰ�����ICNT->L2 queue��������ֱ������L2->DRAM queue��
    //��Ȼ��L2ʱ��Ƶ�ʡ���Ƭ��DRAM���ص���������DRAM->L2 queue����������L2 Cache Bank���ġ�
    //��L2��SIMT Core�Ķ���Ӧͨ��L2->ICNT queue���͡�
    for (unsigned i = 0; i < reqs.size(); ++i) {
      mem_fetch *req = reqs[i];
      m_request_tracker.insert(req);
      if (req->istexture()) {
        m_icnt_L2_queue->push(req);
        req->set_status(IN_PARTITION_ICNT_TO_L2_QUEUE,
                        m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
      } else {
        rop_delay_t r;
        r.req = req;
        r.ready_cycle = cycle + m_config->rop_latency;
        m_rop.push(r);
        req->set_status(IN_PARTITION_ROP_DELAY,
                        m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
      }
    }
  }
}

mem_fetch *memory_sub_partition::pop() {
  mem_fetch *mf = m_L2_icnt_queue->pop();
  m_request_tracker.erase(mf);
  if (mf && mf->isatomic()) mf->do_atomic();
  //��V100�У�L1 cache��m_write_policyΪWRITE_THROUGH��ʵ����L1_WRBK_ACCҲ�����õ���
  //��V100�У���L2 cacheд������ʱ����ȡlazy_fetch_on_read���ԣ����ҵ�һ��cache block
  //���ʱ��������cache block�Ǳ�MODIFIED������Ҫ�����cache blockд�ص���һ���洢��
  //��˻����L2_WRBK_ACC���ʣ�������ʾ���Ϊ��д�ر������MODIFIED cache block��
  if (mf && (mf->get_access_type() == L2_WRBK_ACC ||
             mf->get_access_type() == L1_WRBK_ACC)) {
    delete mf;
    mf = NULL;
  }
  return mf;
}

/*
�����д洢�ӷ����������絯���������ݰ�mf��gpgpu_n_memΪ�����е��ڴ��������DRAM Channel��
����������Ϊ��
 option_parser_register(opp, "-gpgpu_n_mem", OPT_UINT32, &m_n_mem,
                        "number of memory modules (e.g. memory controllers) in gpu",
                        "8");
��V100�����У���32���ڴ��������DRAM Channel����ͬʱÿ���ڴ��������Ϊ�������ӷ�������ˣ�
m_n_sub_partition_per_memory_channelΪ2������Ϊ��
 option_parser_register(opp, "-gpgpu_n_sub_partition_per_mchannel", OPT_UINT32,
                        &m_n_sub_partition_per_memory_channel,
                        "number of memory subpartition in each memory module",
                        "1");
��m_n_mem_sub_partition = m_n_mem * m_n_sub_partition_per_memory_channel������ȫ���ڴ���
������������������Ҫ���ֲ��е��ڴ����ͼ��memory_sub_partition���������Ƴ����ݰ��Ľӿھ���
L2_icnt_queue->ICNT����������ǽ��ڴ��ӷ����е�m_L2_icnt_queue���ж��������ݰ����������ء�
*/
mem_fetch *memory_sub_partition::top() {
  mem_fetch *mf = m_L2_icnt_queue->top();
  //��V100�У�L1 cache��m_write_policyΪWRITE_THROUGH��ʵ����L1_WRBK_ACCҲ�����õ���
  //��V100�У���L2 cacheд������ʱ����ȡlazy_fetch_on_read���ԣ����ҵ�һ��cache block
  //���ʱ��������cache block�Ǳ�MODIFIED������Ҫ�����cache blockд�ص���һ���洢��
  //��˻����L2_WRBK_ACC���ʣ�������ʾ���Ϊ��д�ر������MODIFIED cache block��
  if (mf && (mf->get_access_type() == L2_WRBK_ACC ||
             mf->get_access_type() == L1_WRBK_ACC)) {
    m_L2_icnt_queue->pop();
    m_request_tracker.erase(mf);
    delete mf;
    mf = NULL;
  }
  return mf;
}

void memory_sub_partition::set_done(mem_fetch *mf) {
  m_request_tracker.erase(mf);
}

void memory_sub_partition::accumulate_L2cache_stats(
    class cache_stats &l2_stats) const {
  if (!m_config->m_L2_config.disabled()) {
    l2_stats += m_L2cache->get_stats();
  }
}

void memory_sub_partition::get_L2cache_sub_stats(
    struct cache_sub_stats &css) const {
  if (!m_config->m_L2_config.disabled()) {
    m_L2cache->get_sub_stats(css);
  }
}

void memory_sub_partition::get_L2cache_sub_stats_pw(
    struct cache_sub_stats_pw &css) const {
  if (!m_config->m_L2_config.disabled()) {
    m_L2cache->get_sub_stats_pw(css);
  }
}

void memory_sub_partition::clear_L2cache_stats_pw() {
  if (!m_config->m_L2_config.disabled()) {
    m_L2cache->clear_pw();
  }
}

void memory_sub_partition::visualizer_print(gzFile visualizer_file) {
  // Support for L2 AerialVision stats
  // Per-sub-partition stats would be trivial to extend from this
  cache_sub_stats_pw temp_sub_stats;
  get_L2cache_sub_stats_pw(temp_sub_stats);

  m_stats->L2_read_miss += temp_sub_stats.read_misses;
  m_stats->L2_write_miss += temp_sub_stats.write_misses;
  m_stats->L2_read_hit += temp_sub_stats.read_hits;
  m_stats->L2_write_hit += temp_sub_stats.write_hits;

  clear_L2cache_stats_pw();
}
