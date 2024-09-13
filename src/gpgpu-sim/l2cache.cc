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
m_id是内存分区单元（内存通道）的ID，m_n_sub_partition_per_memory_channel是每个内存通道的子分区数，
global_sub_partition_id是内存子分区的全局ID，这里是计算当前内存子分区的本地ID，即当前内存子分区在
当前内存通道中的本地ID。
*/
int memory_partition_unit::global_sub_partition_id_to_local_id(
    int global_sub_partition_id) const {
  //m_id是内存分区单元（内存通道）的ID，m_n_sub_partition_per_memory_channel是每个内存通道的子分
  //区数，global_sub_partition_id是内存子分区的全局ID，这里是计算当前内存子分区的本地ID，即当前内
  //存子分区在当前内存通道中的本地ID。
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
将内存请求从L2->dram队列移动到DRAM Channel，并将DRAM的返回请求队列中的数据包从DRAM Channel到dram->
L2队列，并对片外GDDR3 DRAM内存向前推进一拍。
*/
void memory_partition_unit::dram_cycle() {
  // pop completed memory request from dram and push it to dram-to-L2 queue
  // of the original sub partition
  //从DRAM弹出已完成的内存请求并将其推送到原始子分区的DRAM到L2队列。m_dram->return_queue_top()返回
  //dram->returnq返回请求队列的顶部元素mf_return，如果队列为空，则返回NULL。
  mem_fetch *mf_return = m_dram->return_queue_top();
  if (mf_return) {
    //如果mf_return有效的话，说明DRAM已经完成了对mf_return的处理，可以将其从DRAM返回队列中弹出。
    unsigned dest_global_spid = mf_return->get_sub_partition_id();
    //计算当前内存子分区的本地ID，即当前内存子分区在当前内存通道中的本地ID。
    int dest_spid = global_sub_partition_id_to_local_id(dest_global_spid);
    //m_sub_partition[dest_spid]->get_id()返回内存子分区的全局ID。
    assert(m_sub_partition[dest_spid]->get_id() == dest_global_spid);
    //如果dest_spid所标识的内存子分区的DRAM_to_L2队列未满。
    if (!m_sub_partition[dest_spid]->dram_L2_queue_full()) {
      if (mf_return->get_access_type() == L1_WRBK_ACC) {
        //mf_return内存请求是L1写回的话，只需设置其完成即可。
        m_sub_partition[dest_spid]->set_done(mf_return);
        delete mf_return;
      } else {
        //将mf_return推送到DRAM_to_L2队列中。
        m_sub_partition[dest_spid]->dram_L2_queue_push(mf_return);
        mf_return->set_status(IN_PARTITION_DRAM_TO_L2_QUEUE,
                              m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
        m_arbitration_metadata.return_credit(dest_spid);
        MEMPART_DPRINTF(
            "mem_fetch request %p return from dram to sub partition %d\n",
            mf_return, dest_spid);
      }
      //弹出已完成的内存请求。
      m_dram->return_queue_pop();
    }
  } else {
    //如果mf_return无效的话，说明这个数据包无效直接弹出作废即可。
    m_dram->return_queue_pop();
  }

  //DRAM向前推进一拍。
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
  //在V100中，当L2 cache写不命中时，采取lazy_fetch_on_read策略，当找到一个cache block
  //逐出时，如果这个cache block是被MODIFIED，则需要将这个cache block写回到下一级存储，
  //因此会产生L2_WRBK_ACC访问，这个访问就是为了写回被逐出的MODIFIED cache block。
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
memory_sub_partition构造函数。
*/
memory_sub_partition::memory_sub_partition(unsigned sub_partition_id, // 内存子分区的ID
                                           const memory_config *config,
                                           class memory_stats_t *stats,
                                           class gpgpu_sim *gpu) {
  m_id = sub_partition_id;
  m_config = config;
  m_stats = stats;
  m_gpu = gpu;
  m_memcpy_cycle_offset = 0;

  //gpgpu_n_mem为配置中的内存控制器（DRAM Channel）数量，定义为：
  //  option_parser_register(
  //      opp, "-gpgpu_n_mem", OPT_UINT32, &m_n_mem,
  //      "number of memory modules (e.g. memory controllers) in gpu", "8");
  //在V100配置中，有32个内存控制器（DRAM Channel）。m_n_mem_sub_partition的定义为：
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
  //这些Delay Queue在整个模拟所有配置中默认为8，8，8，8。
  sscanf(m_config->gpgpu_L2_queue_config, "%u:%u:%u:%u", &icnt_L2, &L2_dram,
         &dram_L2, &L2_icnt);
  //这些queue在GPGPU-Sim v3.0手册中的#内存分区部分有介绍。
  //内存请求数据包通过ICNT->L2 queue从互连网络进入内存分区。L2 Cache Bank在每个L2时钟周期从ICNT->L2 
  //queue弹出一个请求进行服务。L2生成的芯片外DRAM的任何内存请求都被推入L2->DRAM queue。如果L2 Cache
  //被禁用，数据包将从ICNT->L2 queue弹出，并直接推入L2->DRAM queue，仍然以L2时钟频率。从片外DRAM返回
  //的填充请求从DRAM->L2 queue弹出，并由L2 Cache Bank消耗。从L2到SIMT Core的读响应通过L2->ICNT que-
  //ue推送。
  //Delay Queue. See http://gpgpu-sim.org/manual/images/0/0e/Mempart-arch.png.
  //Param of fifo_pipeline:
  //  nm：队列的name，字符串。
  //  minlen：队列的最小长度。
  //  maxlen：队列的最大长度。
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
对二级缓存Bank进行计时，并将请求移入或移出二级缓存。下面将描述memory_partition_unit::cache_cycle()的
内部结构。
*/
void memory_sub_partition::cache_cycle(unsigned cycle) {
  // L2 fill responses
  //在V100配置文件中，L2 Cache并未禁用。
  if (!m_config->m_L2_config.disabled()) {
    //L2 Cache内部的MSHR维护了一个就绪内存访问的列表m_current_response，m_L2cache->access_ready()返
    //回的是m_current_response非空，即如果m_current_response非空，说明L2 Cache的MSHR中有就绪的内存访
    //问。m_L2_icnt_queue这里需要看手册中的第五章中内存分区的详细细节图，memory_sub_partition向互连网
    //络推出数据包的接口就是L2_icnt_queue->ICNT，因此这里是判断内存子分区中的m_L2_icnt_queue队列是否非
    //满，如果非满，说明可以向互连网络推出数据包。m_current_response仅存储了就绪内存访问的地址。
    //未命中状态保持寄存器，the miss status holding register，MSHR。MSHR的模型是用mshr_table类来模拟
    //一个具有有限数量的合并请求的完全关联表。请求通过next_access()函数从MSHR中释放。MSHR表具有固定数量
    //的MSHR条目。每个MSHR条目可以为单个缓存行（Cache Line）提供固定数量的未命中请求。MSHR条目的数量和
    //每个条目的最大请求数是可配置的。
    //缓存未命中状态保持寄存器。缓存命中后，将立即向寄存器文件发送数据，以满足请求。在缓存未命中时，未命中
    //处理逻辑将首先检查未命中状态保持寄存器（MSHR），以查看当前是否有来自先前请求的相同请求挂起。如果是，
    //则此请求将合并到同一条目中，并且不需要发出新的数据请求。否则，将为该数据请求保留一个新的MSHR条目和缓
    //存行。缓存状态处理程序可能会在资源不可用时失败，例如没有可用的MSHR条目、该集中的所有缓存块都已保留但
    //尚未填充、未命中队列已满等。
    //这里m_mshrs.access_ready()返回的是就绪内存访问的列表m_current_response是否非空，就绪内存访问的列
    //表仅存储了就绪内存访问的地址。如果存在已经被填入MSHR条目的访问，则返回true。
    //在cache调用fill函数时，数据包fill到cache中时调用，当一个数据包到了后，跟随数据包返回的数据已经就绪，
    //因此将其加入到MSHR的m_current_response中。
    if (m_L2cache->access_ready() && !m_L2_icnt_queue->full()) {
      //m_L2cache->next_access()调用MSHR的next_access()返回一个就绪的内存访问，即m_current_response
      //中的顶部地址标志的数据包（m_current_response仅存储了就绪内存访问的地址）。
      mem_fetch *mf = m_L2cache->next_access();
      //mem_access_type定义了在时序模拟器中对不同类型的存储器进行不同的访存类型：
      //    MA_TUP(GLOBAL_ACC_R),        从global memory读
      //    MA_TUP(LOCAL_ACC_R),         从local memory读
      //    MA_TUP(CONST_ACC_R),         从常量缓存读
      //    MA_TUP(TEXTURE_ACC_R),       从纹理缓存读
      //    MA_TUP(GLOBAL_ACC_W),        向global memory写
      //    MA_TUP(LOCAL_ACC_W),         向local memory写
      //在V100中，L1 cache的m_write_policy为WRITE_THROUGH，实际上L1_WRBK_ACC也不会用到：
      //    MA_TUP(L1_WRBK_ACC),         L1缓存write back
      //在V100中，当L2 cache写不命中时，采取lazy_fetch_on_read策略，当找到一个cache block
      //逐出时，如果这个cache block是被MODIFIED，则需要将这个cache block写回到下一级存储，
      //因此会产生L2_WRBK_ACC访问，这个访问就是为了写回被逐出的MODIFIED cache block。
      //    MA_TUP(L2_WRBK_ACC),         L2缓存write back
      //    MA_TUP(INST_ACC_R),          从指令缓存读
      //L1_WR_ALLOC_R/L2_WR_ALLOC_R在V100配置中暂时用不到：
      //    MA_TUP(L1_WR_ALLOC_R),       L1缓存write-allocate（cache写不命中，将主存中块调入cache，
      //                                 写入该cache块）
      //L1_WR_ALLOC_R/L2_WR_ALLOC_R在V100配置中暂时用不到：
      //    MA_TUP(L2_WR_ALLOC_R),       L2缓存write-allocate（cache写不命中，将主存中块调入cache，
      //                                 写入该cache块）
      //    MA_TUP(NUM_MEM_ACCESS_TYPE), 存储器访问的类型总数
      if (mf->get_access_type() !=
          L2_WR_ALLOC_R) {  // Don't pass write allocate read request back to
                            // upper level cache
        // 参考：https://blog.csdn.net/qq_41587740/article/details/109104962
        //当前缓存层次是L2缓存，如果mf的类型是L2_WR_ALLOC_R，说明L2缓存发生了写不命中，需要将主存中块调
        //入L2缓存再写入该块，因此mf的类型是L2_WR_ALLOC_R时，不能将mf再向ICNT发送，仅将主存块放入L2即可。
        //set_reply()用来设置内存访问请求响应的类型，内存访问请求中包含四种类型：读请求、写请求、读响应、
        //写确认。这里是设置读响应或者是写确认。
        // void set_reply() {
        //       assert(m_access.get_type() != L1_WRBK_ACC &&
        //             m_access.get_type() != L2_WRBK_ACC);
        //       //如果内存访问请求的类型是读请求，将其设置为读响应。
        //       if (m_type == READ_REQUEST) {
        //         assert(!get_is_write());
        //         m_type = READ_REPLY;
        //       //如果内存访问请求的类型是写请求，将其设置为写确认。
        //       } else if (m_type == WRITE_REQUEST) {
        //         assert(get_is_write());
        //         m_type = WRITE_ACK;
        //       }
        //     }
        mf->set_reply();
        mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,
                       m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
        //这里，如果L2 Cache中有了就绪内存访问，就可以立即将该就绪内存访问（非L2_WR_ALLOC_R请求）发送到
        //ICNT，将mf填充进L2缓存到ICNT的队列m_L2_icnt_queue。
        m_L2_icnt_queue->push(mf);
      } else {
        //当前缓存层次是L2缓存，如果mf的类型是L2_WR_ALLOC_R，说明L2缓存发生了写不命中，需要将主存中块调
        //入L2缓存再写入该块，因此mf的类型是L2_WR_ALLOC_R时，不能将mf再向ICNT发送。
        //FETCH_ON_WRITE 是一种写分配（write allocate）策略中的一个选项。写分配是指在写入操作发生时，如
        //果目标地址不在缓存中，会将该地址的数据从内存中读取到缓存中，然后再进行写入操作。FETCH_ON_WRITE 
        //是指在写入操作发生时才执行读取操作，也就是在进行写入之前先从内存中获取数据。这个策略的优点是能够
        //减少写操作所需要的内存访问次数，从而降低延迟。当写入操作频繁时，使用 FETCH_ON_WRITE 策略可以有
        //效地提高缓存的性能。V100配置中，m_L2_config.m_write_alloc_policy被配置为LAZY_FETCH_ON_READ，
        //下面的if块不生效。
        if (m_config->m_L2_config.m_write_alloc_policy == FETCH_ON_WRITE) {
          //当使用fetch-on-write策略时，mf->get_original_wr_mf()指针指向原始写请求。
          mem_fetch *original_wr_mf = mf->get_original_wr_mf();
          assert(original_wr_mf);
          //set_reply()用来设置内存访问请求响应的类型，内存访问请求中包含四种类型：读请求、写请求、读响应、
          //写确认。这里是设置读响应或者是写确认。
          // void set_reply() {
          //       assert(m_access.get_type() != L1_WRBK_ACC &&
          //             m_access.get_type() != L2_WRBK_ACC);
          //       //如果内存访问请求的类型是读请求，将其设置为读响应。
          //       if (m_type == READ_REQUEST) {
          //         assert(!get_is_write());
          //         m_type = READ_REPLY;
          //       //如果内存访问请求的类型是写请求，将其设置为写确认。
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
        //V100配置中，m_L2_config.m_write_alloc_policy被配置为LAZY_FETCH_ON_READ，上面的if块不生效。
        m_request_tracker.erase(mf);
        delete mf;
      }
    }
  }

  // DRAM to L2 (texture) and icnt (not texture)
  //如果m_dram_L2_queue非空，就获取m_dram_L2_queue的顶部数据包，后续填入L2 Cache。
  if (!m_dram_L2_queue->empty()) {
    //获取m_dram_L2_queue的顶部数据包。
    mem_fetch *mf = m_dram_L2_queue->top();
    // m_L2cache->waiting_for_fill(mf) checks if mf is waiting to be filled by lower memory level.
    //检查是否mf正在等待更低的存储层次填充。waiting_for_fill(mem_fetch *mf)的定义为：
    //     bool baseline_cache::waiting_for_fill(mem_fetch *mf) {
    //       //extra_mf_fields_lookup的定义：
    //       //  typedef std::map<mem_fetch *, extra_mf_fields> extra_mf_fields_lookup;
    //       //向cache发出数据请求mf时，如果未命中，且在MSHR中也未命中（没有mf条目），则将其加入到MSHR中，
    //       //同时，设置m_extra_mf_fields[mf]，意味着如果mf在m_extra_mf_fields中存在，即mf等待着DRAM
    //       //的数据回到L2缓存填充：
    //       //m_extra_mf_fields[mf] = extra_mf_fields(
    //       //      mshr_addr, mf->get_addr(), cache_index, mf->get_data_size(), m_config);
    //       extra_mf_fields_lookup::iterator e = m_extra_mf_fields.find(mf);
    //       return e != m_extra_mf_fields.end();
    //     }
    //若m_L2cache->waiting_for_fill(mf)为真说明此处L2缓存的MSHR中存在mf条目，正在等待DRAM返回的数据填充。
    if (!m_config->m_L2_config.disabled() && m_L2cache->waiting_for_fill(mf)) {
      if (m_L2cache->fill_port_free()) {
        mf->set_status(IN_PARTITION_L2_FILL_QUEUE,
                       m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
        //将m_dram_L2_queue的顶部数据包填充进L2 Cache。
        m_L2cache->fill(mf, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle +
                                m_memcpy_cycle_offset);
        //将m_dram_L2_queue的顶部数据包弹出。
        m_dram_L2_queue->pop();
      }
    } else if (!m_L2_icnt_queue->full()) {
      //如果m_L2cache->waiting_for_fill(mf)不为真，则说明L2缓存的MSHR中不存在mf条目，不在等待DRAM返回
      //的数据填充，那么就可以直接将mf发送回ICNT。
      if (mf->is_write() && mf->get_type() == WRITE_ACK)
        mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,
                       m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
      //将数据包mf推入m_L2_icnt_queue队列。
      m_L2_icnt_queue->push(mf);
      m_dram_L2_queue->pop();
    }
  }

  // prior L2 misses inserted into m_L2_dram_queue here
  //L2缓存向前推进一拍。
  if (!m_config->m_L2_config.disabled()) m_L2cache->cycle();

  // new L2 texture accesses and/or non-texture accesses
  //如果L2向DRAM的队列不满，且ICNT向L2的队列不空，就将ICNT向L2的队列的顶部数据包弹出，填入L2 Cache。
  if (!m_L2_dram_queue->full() && !m_icnt_L2_queue->empty()) {
    //将ICNT向L2的队列的顶部数据包弹出。
    mem_fetch *mf = m_icnt_L2_queue->top();
    //在V100配置中，-gpgpu_cache:dl2_texture_only被设置为0。
    if (!m_config->m_L2_config.disabled() &&
        ((m_config->m_L2_texure_only && mf->istexture()) ||
         (!m_config->m_L2_texure_only))) {
      // L2 is enabled and access is for L2
      //L2缓存被启用，并且访问是针对L2的。
      //m_L2_icnt_queue->full()判断L2缓存向ICNT的队列是否满。
      bool output_full = m_L2_icnt_queue->full();
      //m_L2cache->data_port_free()判断L2缓存的数据端口是否空闲。
      bool port_free = m_L2cache->data_port_free();
      //如果L2缓存向ICNT的队列不满，且L2缓存的数据端口空闲，ICNT向L2的队列的顶部数据包进行L2数据访问。
      if (!output_full && port_free) {
        std::list<cache_event> events;
        //对ICNT向L2的队列的顶部数据包进行L2数据访问，获取访问的状态。
        enum cache_request_status status =
            m_L2cache->access(mf->get_addr(), mf,
                              m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle +
                                  m_memcpy_cycle_offset,
                              events);
        //判断一系列的访问cache事件events是否存在WRITE_REQUEST_SENT。
        //缓存事件类型包括：
        // enum cache_event_type {
        //       //写回请求。
        //       WRITE_BACK_REQUEST_SENT,
        //       //读请求。
        //       READ_REQUEST_SENT,
        //       //写请求。
        //       WRITE_REQUEST_SENT,
        //       //写分配请求。
        //       WRITE_ALLOCATE_SENT
        //     };
        bool write_sent = was_write_sent(events);
        //判断一系列的访问cache事件events是否存在READ_REQUEST_SENT。
        bool read_sent = was_read_sent(events);
        MEM_SUBPART_DPRINTF("Probing L2 cache Address=%llx, status=%u\n",
                            mf->get_addr(), status);

        if (status == HIT) {
          //如果访问L2缓存命中。
          if (!write_sent) {
            //如果不是写操作且命中L2 Cache，则需要判断是否是L1_WRBK_ACC。
            //在V100中，L1 cache的m_write_policy为WRITE_THROUGH，实际上L1_WRBK_ACC也不会用到。
            // L2 cache replies
            assert(!read_sent);
            //!write_sent且!read_sent，发送的是WRITE_BACK_REQUEST_SENT/WRITE_ALLOCATE_SENT。
            if (mf->get_access_type() == L1_WRBK_ACC) {
              m_request_tracker.erase(mf);
              delete mf;
            } else {
              //如果不是L1_WRBK_ACC，则说明是数据读，就需要将reply数据包返回给ICNT。
              //在V100中，L1 cache的m_write_policy为WRITE_THROUGH，实际上L1_WRBK_ACC也不会用到。
              mf->set_reply();
              mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,
                             m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
              m_L2_icnt_queue->push(mf);
            }
            //从ICNT向L2的队列中弹出。
            m_icnt_L2_queue->pop();
          } else {
            assert(write_sent);
            //如果是写操作且命中L2 Cache，则直接从ICNT向L2的队列中弹出这个数据包即可。
            m_icnt_L2_queue->pop();
          }
        } else if (status != RESERVATION_FAIL) {
          //如果访问L2缓存不是命中且不是保留失败，包括HIT_RESERVED/MISS/SECTOR_MISS/MSHR_HIT。
          if (mf->is_write() &&
              //V100配置中，m_L2_config.m_write_alloc_policy被配置为LAZY_FETCH_ON_READ。
              (m_config->m_L2_config.m_write_alloc_policy == FETCH_ON_WRITE ||
               m_config->m_L2_config.m_write_alloc_policy ==
                   LAZY_FETCH_ON_READ) &&
              !was_writeallocate_sent(events)) {
            //在V100中，L1 cache的m_write_policy为WRITE_THROUGH，实际上L1_WRBK_ACC也不会用到。
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
      //L2缓存被禁用或者非纹理访问texture-only L2，但是不存在这种情况，因为在V100配置中，选项
      //-gpgpu_cache:dl2_texture_only被设置为0。
      mf->set_status(IN_PARTITION_L2_TO_DRAM_QUEUE,
                     m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
      m_L2_dram_queue->push(mf);
      m_icnt_L2_queue->pop();
    }
  }

  // ROP delay queue
  //光栅操作流水线（Raster Operations Pipeline，ROP）延迟队列。
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
这里需要看手册中的第五章中内存分区的详细细节图，互连网络向memory_sub_partition推出数据包的接
口就是ICNT->icnt_L2_queue，因此这里是判断内存子分区中的m_icnt_L2_queue队列是否可以放下size
大小的数据，可以放下返回False，放不下返回True。
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
cache_config的第一个字母代表cache的数据请求单位，如果是"N"则代表Normal，如果是"S"则代表Sector。
Normal模式其实代表的是耳熟能详的Set-Associative组成结构，而Sector模式代表的是cache的另外的一种
Sector Buffer组成结构。在V100的配置文件中：
    -gpgpu_cache:dl1  S:4:128:64,L:T:m:L:L,A:512:8,16:0,32
    -gpgpu_cache:dl2  S:32:128:24,L:B:m:L:P,A:192:4,32:0,32
    -gpgpu_cache:il1  N:64:128:16,L:R:f:N:L,S:2:48,4
因此L1 Data Cache和L2 Data Cache都是Sector模式，而L1 Instruction Cache是Normal模式。简单介绍
Sector Buffer组成结构：假定在一个微架构中，Cache大小为16KB，使用Sector Buffer方式时，这个16KB
被分解为16个1KB大小的Sector，CPU可以同时查找这16个Sector。当访问的数据不在这16个Sector中命中时，
将首先进行Sector淘汰操作，在获得一个新的Sector后，将即将需要访问的64B数据填入这个Sector。如果访
问的数据命中了某个Sector，但是数据并不包含在Sector时，将相应的数据继续读到这个Sector中。采用这种
Sector Buffer方法时，Cache的划分粒度较为粗略，对程序的局部性的要求过高。Cache的整体命中率不如采
用Set-Associative的组成方式。

这里如果L2 Cache是Sector Buffer模式，则将数据请求m_req拆分为多个Sector请求。
*/
std::vector<mem_fetch *>
memory_sub_partition::breakdown_request_to_sector_requests(mem_fetch *mf) {
  std::vector<mem_fetch *> result;
  //获取数据请求mf的byte mask。这里的byte mask是用于标记一次访存操作中的扇区掩码，4个扇区，每个
  //扇区32个字节数据，因此sector_mask是一个标记128 byte数据的掩码，共128位bitset：
  //    typedef std::bitset<SECTOR_CHUNCK_SIZE> mem_access_sector_mask_t;
  //    const unsigned SECTOR_CHUNCK_SIZE = 4;  // four sectors
  //    const unsigned SECTOR_SIZE = 32;        // sector is 32 bytes width
  mem_access_sector_mask_t sector_mask = mf->get_access_sector_mask();
  if (mf->get_data_size() == SECTOR_SIZE &&
      mf->get_access_sector_mask().count() == 1) {
    //如果数据请求的大小正好是一个SECTOR，且只有一个SECTOR被访问，则不需要拆分，直接将该请求压入
    //result。mf->get_access_sector_mask().count()=1时，说明只有其中一位为1，即只有一个字节被
    //访问，因此这一个字节一定位于单个扇区中。
    result.push_back(mf);
  } else if (mf->get_data_size() == MAX_MEMORY_ACCESS_SIZE) {
    // break down every sector
    //MAX_MEMORY_ACCESS_SIZE定义为：const unsigned MAX_MEMORY_ACCESS_SIZE = 128。这里的mask
    //也是用于标记一次访存操作中的数据字节掩码，MAX_MEMORY_ACCESS_SIZE设置为128，即每次访存最大
    //数据128字节，共128位bitset。mem_access_byte_mask_t的定义为：
    //    typedef std::bitset<MAX_MEMORY_ACCESS_SIZE> mem_access_byte_mask_t;
    mem_access_byte_mask_t mask;
    //对于一个MAX_MEMORY_ACCESS_SIZE=128字节大小的请求来说，总共分为4个SECTOR_CHUNCK，且划分的
    //SECTOR的大小为SECTOR_SIZE=32。因此这里对4个SECTOR_CHUNCK循环。目的是将一个128字节的大请求
    //划分为SECTOR_CHUNCK_SIZE = 4个独立的请求。
    for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; i++) {
      //k代表的是第i个SECTOR_CHUNCK中的字节编号范围，第0个CHUNCK中k的范围是[0,32)，第1个CHUNCK
      //中k的范围是[32,64)，第2个CHUNCK中k的范围是[64,96)，第3个CHUNCK中k的范围是[96,128)。
      for (unsigned k = i * SECTOR_SIZE; k < (i + 1) * SECTOR_SIZE; k++) {
        //第i个SECTOR_CHUNCK中的字节mask设置为k的范围对应位。
        mask.set(k);
      }
      //将第i个SECTOR_CHUNCK中的字节mask与mf的字节mask进行与操作，得到第i个SECTOR_CHUNCK的请求。
      mem_fetch *n_mf = m_mf_allocator->alloc(
          mf->get_addr() + SECTOR_SIZE * i, mf->get_access_type(),
          mf->get_access_warp_mask(), mf->get_access_byte_mask() & mask,
          std::bitset<SECTOR_CHUNCK_SIZE>().set(i), SECTOR_SIZE, mf->is_write(),
          m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, mf->get_wid(),
          mf->get_sid(), mf->get_tpc(), mf, mf->get_streamID());
      //将第i个SECTOR_CHUNCK的请求压入result。
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
将数据读请求m_req从互连网络推入到内存子分区来进行后续取数据处理。
*/
void memory_sub_partition::push(mem_fetch *m_req, unsigned long long cycle) {
  if (m_req) {
    m_stats->memlatstat_icnt2mem_pop(m_req);
    std::vector<mem_fetch *> reqs;
    //cache_config的第一个字母代表cache的数据请求单位，如果是"N"则代表Normal，如果是"S"则
    //代表Sector。Normal模式其实代表的是耳熟能详的Set-Associative组成结构，而Sector模式代
    //表的是cache的另一种Sector Buffer组成结构。在V100的配置文件中：
    //    -gpgpu_cache:dl1  S:4:128:64,L:T:m:L:L,A:512:8,16:0,32
    //    -gpgpu_cache:dl2  S:32:128:24,L:B:m:L:P,A:192:4,32:0,32
    //    -gpgpu_cache:il1  N:64:128:16,L:R:f:N:L,S:2:48,4
    //因此L1 Data Cache和L2 Data Cache都是Sector模式，而L1 Instruction Cache是Normal模式。
    //简单介绍Sector Buffer组成结构：假定在一个微架构中，Cache大小为16KB，使用Sector Buffer
    //方式时，这个16KB被分解为16个1KB大小的Sector，CPU可以同时查找这16个Sector。当访问的数据
    //不在这16个Sector中命中时，将首先进行Sector淘汰操作，在获得一个新的Sector后，将即将需要
    //访问的64B数据填入这个Sector。如果访问的数据命中了某个Sector，但是数据并不包含在Sector时，
    //将相应的数据继续读到这个Sector中。采用这种方法时，Cache的划分粒度较为粗略，对程序的局部
    //性的要求过高。Cache的整体命中率不如采用Set-Associative的组成方式。

    //这里如果L2 Cache是Sector Buffer模式，则将数据请求m_req拆分为多个Sector请求。
    if (m_config->m_L2_config.m_cache_type == SECTOR)
      reqs = breakdown_request_to_sector_requests(m_req);
    else
      reqs.push_back(m_req);

    //对于每个请求，将其压入m_icnt_L2_queue队列（针对纹理访问）或 m_rop（Raster Operations 
    //Pipeline，ROP队列，针对非纹理操作）。内存请求数据包通过ICNT->L2 queue从互连网络进入内存
    //分区。如GT200微基准测试研究所观察到的，非纹理访问通过光栅操作流水线（Raster Operations 
    //Pipeline，ROP）队列进行，以模拟460 L2时钟周期的最小流水线延迟。L2 Cache Bank在每个L2时
    //钟周期从ICNT->L2 queue弹出一个请求进行服务。L2生成的芯片外DRAM的任何内存请求都被推入L2->
    //DRAM queue。如果L2 Cache被禁用，数据包将从ICNT->L2 queue弹出，并直接推入L2->DRAM queue，
    //仍然以L2时钟频率。从片外DRAM返回的填充请求从DRAM->L2 queue弹出，并由L2 Cache Bank消耗。
    //从L2到SIMT Core的读响应通过L2->ICNT queue推送。
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
  //在V100中，L1 cache的m_write_policy为WRITE_THROUGH，实际上L1_WRBK_ACC也不会用到。
  //在V100中，当L2 cache写不命中时，采取lazy_fetch_on_read策略，当找到一个cache block
  //逐出时，如果这个cache block是被MODIFIED，则需要将这个cache block写回到下一级存储，
  //因此会产生L2_WRBK_ACC访问，这个访问就是为了写回被逐出的MODIFIED cache block。
  if (mf && (mf->get_access_type() == L2_WRBK_ACC ||
             mf->get_access_type() == L1_WRBK_ACC)) {
    delete mf;
    mf = NULL;
  }
  return mf;
}

/*
从所有存储子分区向互连网络弹出顶部数据包mf。gpgpu_n_mem为配置中的内存控制器（DRAM Channel）
数量，定义为：
 option_parser_register(opp, "-gpgpu_n_mem", OPT_UINT32, &m_n_mem,
                        "number of memory modules (e.g. memory controllers) in gpu",
                        "8");
在V100配置中，有32个内存控制器（DRAM Channel），同时每个内存控制器分为了两个子分区，因此，
m_n_sub_partition_per_memory_channel为2，定义为：
 option_parser_register(opp, "-gpgpu_n_sub_partition_per_mchannel", OPT_UINT32,
                        &m_n_sub_partition_per_memory_channel,
                        "number of memory subpartition in each memory module",
                        "1");
而m_n_mem_sub_partition = m_n_mem * m_n_sub_partition_per_memory_channel，代表全部内存子
分区的总数。这里需要看手册中的内存分区图，memory_sub_partition向互连网络推出数据包的接口就是
L2_icnt_queue->ICNT，因此这里是将内存子分区中的m_L2_icnt_queue队列顶部的数据包弹出并返回。
*/
mem_fetch *memory_sub_partition::top() {
  mem_fetch *mf = m_L2_icnt_queue->top();
  //在V100中，L1 cache的m_write_policy为WRITE_THROUGH，实际上L1_WRBK_ACC也不会用到。
  //在V100中，当L2 cache写不命中时，采取lazy_fetch_on_read策略，当找到一个cache block
  //逐出时，如果这个cache block是被MODIFIED，则需要将这个cache block写回到下一级存储，
  //因此会产生L2_WRBK_ACC访问，这个访问就是为了写回被逐出的MODIFIED cache block。
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
