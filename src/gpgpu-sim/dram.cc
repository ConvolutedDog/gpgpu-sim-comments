// Copyright (c) 2009-2021, Tor M. Aamodt, Wilson W.L. Fung, Ali Bakhoda,
// Ivan Sham, George L. Yuan, Vijay Kandiah, Nikos Hardavellas,
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

#include "dram.h"
#include "dram_sched.h"
#include "gpu-misc.h"
#include "gpu-sim.h"
#include "hashing.h"
#include "l2cache.h"
#include "mem_fetch.h"
#include "mem_latency_stat.h"

#ifdef DRAM_VERIFY
int PRINT_CYCLE = 0;
#endif

template class fifo_pipeline<mem_fetch>;
template class fifo_pipeline<dram_req_t>;

/*
dram_t的构造函数。每个memory_partition_unit有一个dram_t模型，dram_t是隶属于单个
memory_partition_unit的DRAM模型。
参考: https://zhuanlan.zhihu.com/p/561501585
*/
dram_t::dram_t(unsigned int partition_id, const memory_config *config,
               memory_stats_t *stats, memory_partition_unit *mp,
               gpgpu_sim *gpu) {
  //memory_partition_unit的ID。
  id = partition_id;
  //隶属于的memory_partition_unit。
  m_memory_partition_unit = mp;
  m_stats = stats;
  //memory config。
  m_config = config;
  m_gpu = gpu;

  // rowblp
  //访问总数。
  access_num = 0;
  //命中总数。在允许 row buffer 长时间保持行数据的情况下，如果读写请求发生在 row 
  //buffer 保存的单元行中（这种情况称为“行命中”），那么 DRAM 的读写速度会很快，因
  //为 DRAM 可以直接操作 row buffer ，而不需要读取新的单元行；而如果读写请求发生在 
  //row buffer 之外的单元行中（这种情况称为“行缺失”），那 DRAM 就要把 row buffer 
  //写回并读取新的单元行，这样做速度会很慢。
  hits_num = 0;
  //读总数。
  read_num = 0;
  //写总数。
  write_num = 0;
  //命中的读总数。
  hits_read_num = 0;
  //命中的写总数。
  hits_write_num = 0;
  //对所有bank的读请求总次数。每运行dram_t::cycle()增加1。
  banks_1time = 0;
  //对任意bank有读请求的总拍数。
  banks_acess_total = 0;
  banks_acess_total_after = 0;
  banks_time_ready = 0;
  banks_access_ready_total = 0;
  //同时发送行命令和列命令的次数：issued_col_cmd && issued_row_cmd。
  issued_two = 0;
  //发送行命令和列命令的总次数。
  issued_total = 0;
  //发送行命令的总次数。
  issued_total_row = 0;
  //发送列命令的总次数。
  issued_total_col = 0;

  //tCCD - CAS to CAS Delay. 
  //tCCD（CAS to CAS Delay）：CAS命令到CAS命令之间的时间间隔。
  CCDc = 0;
  //tRRD - Row to Row Delay.
  //tRRD（Row to Row Delay）：行单元到行单元的延时。
  //该值还表示向相同bank中的同一个行单元两次发送激活指令(即：REF)之间的时间间隔。
  //tRRD的值越小越好，延迟越低，表示下一个bank能更快地被激活，进行读写操作。然而如果
  //延迟太短，会引起连续数据膨胀。
  //tRRD_S表示不同bank，tRRD_L表示相同bank。
  RRDc = 0;
  //read to write penalty applies across banks, time to switch from read to write.
  //tWTR - Write to Read Delay.
  //tWTR（Write to Read Delay）：写到读延时（即：最后的数据进入读指令）。
  //该值表示的是，DDR内存模块中的同一个单元中，在最后一次有效的写操作和下一次读操作之
  //间必须等待的时钟周期。
  //增加tWTR的值，可以让内容模块运行比其默认速度更快的速度下；减小tWTR的值，可以提高
  //读性能，但会降低系统稳定性。
  //tWTR_S表示不同bank，tWTR_L表示相同bank。
  RTWc = 0;
  //write to read penalty applies across banks, time to switch from write to read.
  WTRc = 0;

  wasted_bw_row = 0;
  wasted_bw_col = 0;
  util_bw = 0;
  idle_bw = 0;
  RCDc_limit = 0;
  CCDLc_limit = 0;
  CCDLc_limit_alone = 0;
  CCDc_limit = 0;
  WTRc_limit = 0;
  WTRc_limit_alone = 0;
  RCDWRc_limit = 0;
  RTWc_limit = 0;
  RTWc_limit_alone = 0;
  rwq_limit = 0;
  write_to_read_ratio_blp_rw_average = 0;
  bkgrp_parallsim_rw = 0;

  rw = READ;  // read mode is default

  bkgrp = (bankgrp_t **)calloc(sizeof(bankgrp_t *), m_config->nbkgrp);
  bkgrp[0] = (bankgrp_t *)calloc(sizeof(bank_t), m_config->nbkgrp);
  for (unsigned i = 1; i < m_config->nbkgrp; i++) {
    bkgrp[i] = bkgrp[0] + i;
  }
  for (unsigned i = 0; i < m_config->nbkgrp; i++) {
    bkgrp[i]->CCDLc = 0;
    bkgrp[i]->RTPLc = 0;
  }

  bk = (bank_t **)calloc(sizeof(bank_t *), m_config->nbk);
  bk[0] = (bank_t *)calloc(sizeof(bank_t), m_config->nbk);
  for (unsigned i = 1; i < m_config->nbk; i++) bk[i] = bk[0] + i;
  for (unsigned i = 0; i < m_config->nbk; i++) {
    bk[i]->state = BANK_IDLE;
    bk[i]->bkgrpindex = i / (m_config->nbk / m_config->nbkgrp);
  }
  prio = 0;
  //CAS Latency Control(也被描述为tCL、CL、CAS Latency Time、CAS Timing Delay)，CAS latency是
  //“内存读写操作前列地址控制器的潜伏时间”。 CAS控制从接受一个指令到执行指令之间的时间。
  rwq = new fifo_pipeline<dram_req_t>("rwq", m_config->CL, m_config->CL + 1);
  //内存请求队列。
  mrqq = new fifo_pipeline<dram_req_t>("mrqq", 0, 2);
  returnq = new fifo_pipeline<mem_fetch>(
      "dramreturnq", 0,
      m_config->gpgpu_dram_return_queue_size == 0
          ? 1024
          : m_config->gpgpu_dram_return_queue_size);
  m_frfcfs_scheduler = NULL;
  if (m_config->scheduler_type == DRAM_FRFCFS)
    m_frfcfs_scheduler = new frfcfs_scheduler(m_config, this, stats);
  n_cmd = 0;
  n_activity = 0;
  n_nop = 0;
  n_act = 0;
  n_pre = 0;
  n_rd = 0;
  n_wr = 0;
  n_wr_WB = 0;
  n_rd_L2_A = 0;
  n_req = 0;
  max_mrqs_temp = 0;
  bwutil = 0;
  max_mrqs = 0;
  ave_mrqs = 0;

  for (unsigned i = 0; i < 10; i++) {
    dram_util_bins[i] = 0;
    dram_eff_bins[i] = 0;
  }
  last_n_cmd = last_n_activity = last_bwutil = 0;

  n_cmd_partial = 0;
  n_activity_partial = 0;
  n_nop_partial = 0;
  n_act_partial = 0;
  n_pre_partial = 0;
  n_req_partial = 0;
  ave_mrqs_partial = 0;
  bwutil_partial = 0;

  if (queue_limit())
    mrqq_Dist = StatCreate("mrqq_length", 1, queue_limit());
  else                                             // queue length is unlimited;
    mrqq_Dist = StatCreate("mrqq_length", 1, 64);  // track up to 64 entries
}

bool dram_t::full(bool is_write) const {
  //scheduler_type在V100中配置为FR-FCFS，即DRAM_FRFCFS。
  //First-Row First-Come-First-Served scheduler gives higher priority to requests to a 
  //currently open row in any of the DRAM banks. The scheduler will schedule all requests 
  //in the queue to open rows first. If no such request exists it will open a new row for 
  //the oldest request. The code for this scheduler is located in dram_sched.h/.cc.
  if (m_config->scheduler_type == DRAM_FRFCFS) {
    //gpgpu_frfcfs_dram_sched_queue_size在V100中配置为64。
    if (m_config->gpgpu_frfcfs_dram_sched_queue_size == 0) return false;
    //dram_seperate_write_queue_enable在V100中配置为1。
    if (m_config->seperate_write_queue_enabled) {
      if (is_write)
        return m_frfcfs_scheduler->num_write_pending() >=
               m_config->gpgpu_frfcfs_dram_write_queue_size;
      else
        return m_frfcfs_scheduler->num_pending() >=
               m_config->gpgpu_frfcfs_dram_sched_queue_size;
    } else
      return m_frfcfs_scheduler->num_pending() >=
             m_config->gpgpu_frfcfs_dram_sched_queue_size;
  } else
    return mrqq->full();
}

unsigned dram_t::que_length() const {
  unsigned nreqs = 0;
  if (m_config->scheduler_type == DRAM_FRFCFS) {
    nreqs = m_frfcfs_scheduler->num_pending();
  } else {
    nreqs = mrqq->get_length();
  }
  return nreqs;
}

bool dram_t::returnq_full() const { return returnq->full(); }

unsigned int dram_t::queue_limit() const {
  return m_config->gpgpu_frfcfs_dram_sched_queue_size;
}

dram_req_t::dram_req_t(class mem_fetch *mf, unsigned banks,
                       unsigned dram_bnk_indexing_policy,
                       class gpgpu_sim *gpu) {
  txbytes = 0;
  dqbytes = 0;
  data = mf;
  m_gpu = gpu;

  const addrdec_t &tlx = mf->get_tlx_addr();

  switch (dram_bnk_indexing_policy) {
    case LINEAR_BK_INDEX: {
      bk = tlx.bk;
      break;
    }
    case BITWISE_XORING_BK_INDEX: {
      // xoring bank bits with lower bits of the page
      bk = bitwise_hash_function(tlx.row, tlx.bk, banks);
      assert(bk < banks);
      break;
    }
    case IPOLY_BK_INDEX: {
      /*IPOLY for bank indexing function from "Pseudo-randomly interleaved
       * memory." Rau, B. R et al. ISCA 1991
       * http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=348DEA37A3E440473B3C075EAABC63B6?doi=10.1.1.12.7149&rep=rep1&type=pdf
       */
      // xoring bank bits with lower bits of the page
      bk = ipoly_hash_function(tlx.row, tlx.bk, banks);
      assert(bk < banks);
      break;
    }
    case CUSTOM_BK_INDEX:
      /* No custom set function implemented */
      // Do you custom index here
      break;
    default:
      assert("\nUndefined bank index function.\n" && 0);
      break;
  }

  row = tlx.row;
  col = tlx.col;
  nbytes = mf->get_data_size();

  timestamp = m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle;
  addr = mf->get_addr();
  insertion_time = (unsigned)m_gpu->gpu_sim_cycle;
  rw = data->get_is_write() ? WRITE : READ;
}

void dram_t::push(class mem_fetch *data) {
  assert(id == data->get_tlx_addr()
                   .chip);  // Ensure request is in correct memory partition

  dram_req_t *mrq =
      new dram_req_t(data, m_config->nbk, m_config->dram_bnk_indexing_policy,
                     m_memory_partition_unit->get_mgpu());

  data->set_status(IN_PARTITION_MC_INTERFACE_QUEUE,
                   m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
  mrqq->push(mrq);

  // stats...
  n_req += 1;
  n_req_partial += 1;
  if (m_config->scheduler_type == DRAM_FRFCFS) {
    unsigned nreqs = m_frfcfs_scheduler->num_pending();
    if (nreqs > max_mrqs_temp) max_mrqs_temp = nreqs;
  } else {
    max_mrqs_temp = (max_mrqs_temp > mrqq->get_length()) ? max_mrqs_temp
                                                         : mrqq->get_length();
  }
  m_stats->memlatstat_dram_access(data);
}

void dram_t::scheduler_fifo() {
  if (!mrqq->empty()) {
    unsigned int bkn;
    dram_req_t *head_mrqq = mrqq->top();
    head_mrqq->data->set_status(
        IN_PARTITION_MC_BANK_ARB_QUEUE,
        m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
    bkn = head_mrqq->bk;
    if (!bk[bkn]->mrq) bk[bkn]->mrq = mrqq->pop();
  }
}

#define DEC2ZERO(x) x = (x) ? (x - 1) : 0;
#define SWAP(a, b) \
  a ^= b;          \
  b ^= a;          \
  a ^= b;

/*
dram_t向前推进一拍。
*/
void dram_t::cycle() {
  //returnq非满的话，可以将新的就绪的数据放入returnq。
  if (!returnq->full()) {
    dram_req_t *cmd = rwq->pop();
    if (cmd) {
#ifdef DRAM_VIEWCMD
      printf("\tDQ: BK%d Row:%03x Col:%03x", cmd->bk, cmd->row,
             cmd->col + cmd->dqbytes);
#endif
      cmd->dqbytes += m_config->dram_atom_size;

      if (cmd->dqbytes >= cmd->nbytes) {
        mem_fetch *data = cmd->data;
        data->set_status(IN_PARTITION_MC_RETURNQ,
                         m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
        //在V100中，L1 cache的m_write_policy为WRITE_THROUGH，实际上L1_WRBK_ACC不会用到。
        //在V100中，当L2 cache写不命中时，采取lazy_fetch_on_read策略，当找到一个cache block
        //逐出时，如果这个cache block是被MODIFIED，则需要将这个cache block写回到下一级存储，
        //因此会产生L2_WRBK_ACC访问，这个访问就是为了写回被逐出的MODIFIED cache block。
        if (data->get_access_type() != L1_WRBK_ACC &&
            data->get_access_type() != L2_WRBK_ACC) {
          data->set_reply();
          returnq->push(data);
        } else {
          m_memory_partition_unit->set_done(data);
          delete data;
        }
        delete cmd;
      }
#ifdef DRAM_VIEWCMD
      printf("\n");
#endif
    }
  }

  /* check if the upcoming request is on an idle bank */
  /* Should we modify this so that multiple requests are checked? */

  switch (m_config->scheduler_type) {
    case DRAM_FIFO:
      scheduler_fifo();
      break;
    //V100中采用FR-FCFS调度策略，这里是在dram_t::cycle()中进行调度。
    case DRAM_FRFCFS:
      //FR-FCFS调度器进行调度。主要工作是，将memory request queue中的请求加入到调度器中，
      //然后从调度器中取出一个请求，并依据请求是读数据还是写数据，分别将其加入到它对应Bank
      //的读取队列或写入队列中。
      scheduler_frfcfs();
      break;
    default:
      printf("Error: Unknown DRAM scheduler type\n");
      assert(0);
  }
  if (m_config->scheduler_type == DRAM_FRFCFS) {
    unsigned nreqs = m_frfcfs_scheduler->num_pending();
    if (nreqs > max_mrqs) {
      max_mrqs = nreqs;
    }
    ave_mrqs += nreqs;
    ave_mrqs_partial += nreqs;
  } else {
    if (mrqq->get_length() > max_mrqs) {
      max_mrqs = mrqq->get_length();
    }
    ave_mrqs += mrqq->get_length();
    ave_mrqs_partial += mrqq->get_length();
  }

  unsigned k = m_config->nbk;
  bool issued = false;

  // collect row buffer locality, BLP and other statistics
  /////////////////////////////////////////////////////////////////////////
  unsigned int memory_pending = 0;
  for (unsigned i = 0; i < m_config->nbk; i++) {
    if (bk[i]->mrq) memory_pending++;
  }
  //对所有bank的读请求总次数。
  banks_1time += memory_pending;
  //对任意bank有读请求的总拍数。
  if (memory_pending > 0) banks_acess_total++;

  unsigned int memory_pending_rw = 0;
  unsigned read_blp_rw = 0;
  unsigned write_blp_rw = 0;
  std::bitset<8> bnkgrp_rw_found;  // assume max we have 8 bank groups

  for (unsigned j = 0; j < m_config->nbk; j++) {
    unsigned grp = get_bankgrp_number(j);
    if (bk[j]->mrq &&
        (((bk[j]->curr_row == bk[j]->mrq->row) && (bk[j]->mrq->rw == READ) &&
          (bk[j]->state == BANK_ACTIVE)))) {
      memory_pending_rw++;
      read_blp_rw++;
      bnkgrp_rw_found.set(grp);
    } else if (bk[j]->mrq &&
               (((bk[j]->curr_row == bk[j]->mrq->row) &&
                 (bk[j]->mrq->rw == WRITE) && (bk[j]->state == BANK_ACTIVE)))) {
      memory_pending_rw++;
      write_blp_rw++;
      bnkgrp_rw_found.set(grp);
    }
  }
  banks_time_rw += memory_pending_rw;
  bkgrp_parallsim_rw += bnkgrp_rw_found.count();
  if (memory_pending_rw > 0) {
    write_to_read_ratio_blp_rw_average +=
        (double)write_blp_rw / (write_blp_rw + read_blp_rw);
    banks_access_rw_total++;
  }

  unsigned int memory_Pending_ready = 0;
  for (unsigned j = 0; j < m_config->nbk; j++) {
    unsigned grp = get_bankgrp_number(j);
    if (bk[j]->mrq &&
        ((!CCDc && !bk[j]->RCDc && !(bkgrp[grp]->CCDLc) &&
          (bk[j]->curr_row == bk[j]->mrq->row) && (bk[j]->mrq->rw == READ) &&
          (WTRc == 0) && (bk[j]->state == BANK_ACTIVE) && !rwq->full()) ||
         (!CCDc && !bk[j]->RCDWRc && !(bkgrp[grp]->CCDLc) &&
          (bk[j]->curr_row == bk[j]->mrq->row) && (bk[j]->mrq->rw == WRITE) &&
          (RTWc == 0) && (bk[j]->state == BANK_ACTIVE) && !rwq->full()))) {
      memory_Pending_ready++;
    }
  }
  banks_time_ready += memory_Pending_ready;
  if (memory_Pending_ready > 0) banks_access_ready_total++;
  ///////////////////////////////////////////////////////////////////////////////////

  bool issued_col_cmd = false;
  bool issued_row_cmd = false;

  //在V100配置中为1。
  if (m_config->dual_bus_interface) {
    // dual bus interface
    // issue one row command and one column command
    for (unsigned i = 0; i < m_config->nbk; i++) {
      unsigned j = (i + prio) % m_config->nbk;
      issued_col_cmd = issue_col_command(j);
      if (issued_col_cmd) break;
    }
    for (unsigned i = 0; i < m_config->nbk; i++) {
      unsigned j = (i + prio) % m_config->nbk;
      issued_row_cmd = issue_row_command(j);
      if (issued_row_cmd) break;
    }
    for (unsigned i = 0; i < m_config->nbk; i++) {
      unsigned j = (i + prio) % m_config->nbk;
      if (!bk[j]->mrq) {
        if (!CCDc && !RRDc && !RTWc && !WTRc && !bk[j]->RCDc && !bk[j]->RASc &&
            !bk[j]->RCc && !bk[j]->RPc && !bk[j]->RCDWRc)
          k--;
        bk[j]->n_idle++;
      }
    }
  } else {
    // single bus interface
    // issue only one row/column command
    for (unsigned i = 0; i < m_config->nbk; i++) {
      unsigned j = (i + prio) % m_config->nbk;
      if (!issued_col_cmd) issued_col_cmd = issue_col_command(j);

      if (!issued_col_cmd && !issued_row_cmd)
        issued_row_cmd = issue_row_command(j);

      if (!bk[j]->mrq) {
        if (!CCDc && !RRDc && !RTWc && !WTRc && !bk[j]->RCDc && !bk[j]->RASc &&
            !bk[j]->RCc && !bk[j]->RPc && !bk[j]->RCDWRc)
          k--;
        bk[j]->n_idle++;
      }
    }
  }

  issued = issued_row_cmd || issued_col_cmd;
  if (!issued) {
    n_nop++;
    n_nop_partial++;
#ifdef DRAM_VIEWCMD
    printf("\tNOP                        ");
#endif
  }
  if (k) {
    n_activity++;
    n_activity_partial++;
  }
  n_cmd++;
  n_cmd_partial++;
  if (issued) {
    issued_total++;
    if (issued_col_cmd && issued_row_cmd) issued_two++;
  }
  if (issued_col_cmd) issued_total_col++;
  if (issued_row_cmd) issued_total_row++;

  // Collect some statistics
  // check the limitation, see where BW is wasted?
  /////////////////////////////////////////////////////////
  unsigned int memory_pending_found = 0;
  for (unsigned i = 0; i < m_config->nbk; i++) {
    if (bk[i]->mrq) memory_pending_found++;
  }
  if (memory_pending_found > 0) banks_acess_total_after++;

  bool memory_pending_rw_found = false;
  for (unsigned j = 0; j < m_config->nbk; j++) {
    if (bk[j]->mrq &&
        (((bk[j]->curr_row == bk[j]->mrq->row) && (bk[j]->mrq->rw == READ) &&
          (bk[j]->state == BANK_ACTIVE)) ||
         ((bk[j]->curr_row == bk[j]->mrq->row) && (bk[j]->mrq->rw == WRITE) &&
          (bk[j]->state == BANK_ACTIVE))))
      memory_pending_rw_found = true;
  }

  if (issued_col_cmd || CCDc)
    util_bw++;
  else if (memory_pending_rw_found) {
    wasted_bw_col++;
    for (unsigned j = 0; j < m_config->nbk; j++) {
      unsigned grp = get_bankgrp_number(j);
      // read
      if (bk[j]->mrq &&
          (((bk[j]->curr_row == bk[j]->mrq->row) && (bk[j]->mrq->rw == READ) &&
            (bk[j]->state == BANK_ACTIVE)))) {
        if (bk[j]->RCDc) RCDc_limit++;
        if (bkgrp[grp]->CCDLc) CCDLc_limit++;
        if (WTRc) WTRc_limit++;
        if (CCDc) CCDc_limit++;
        if (rwq->full()) rwq_limit++;
        if (bkgrp[grp]->CCDLc && !WTRc) CCDLc_limit_alone++;
        if (!bkgrp[grp]->CCDLc && WTRc) WTRc_limit_alone++;
      }
      // write
      else if (bk[j]->mrq &&
               ((bk[j]->curr_row == bk[j]->mrq->row) &&
                (bk[j]->mrq->rw == WRITE) && (bk[j]->state == BANK_ACTIVE))) {
        if (bk[j]->RCDWRc) RCDWRc_limit++;
        if (bkgrp[grp]->CCDLc) CCDLc_limit++;
        if (RTWc) RTWc_limit++;
        if (CCDc) CCDc_limit++;
        if (rwq->full()) rwq_limit++;
        if (bkgrp[grp]->CCDLc && !RTWc) CCDLc_limit_alone++;
        if (!bkgrp[grp]->CCDLc && RTWc) RTWc_limit_alone++;
      }
    }
  } else if (memory_pending_found)
    wasted_bw_row++;
  else if (!memory_pending_found)
    idle_bw++;
  else
    assert(1);

  /////////////////////////////////////////////////////////

  // decrements counters once for each time dram_issueCMD is called
  DEC2ZERO(RRDc);
  DEC2ZERO(CCDc);
  DEC2ZERO(RTWc);
  DEC2ZERO(WTRc);
  for (unsigned j = 0; j < m_config->nbk; j++) {
    DEC2ZERO(bk[j]->RCDc);
    DEC2ZERO(bk[j]->RASc);
    DEC2ZERO(bk[j]->RCc);
    DEC2ZERO(bk[j]->RPc);
    DEC2ZERO(bk[j]->RCDWRc);
    DEC2ZERO(bk[j]->WTPc);
    DEC2ZERO(bk[j]->RTPc);
  }
  for (unsigned j = 0; j < m_config->nbkgrp; j++) {
    DEC2ZERO(bkgrp[j]->CCDLc);
    DEC2ZERO(bkgrp[j]->RTPLc);
  }

#ifdef DRAM_VISUALIZE
  visualize();
#endif
}

bool dram_t::issue_col_command(int j) {
  bool issued = false;
  unsigned grp = get_bankgrp_number(j);
  if (bk[j]->mrq) {  // if currently servicing a memory request
    bk[j]->mrq->data->set_status(
        IN_PARTITION_DRAM, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
    // correct row activated for a READ
    if (!issued && !CCDc && !bk[j]->RCDc && !(bkgrp[grp]->CCDLc) &&
        (bk[j]->curr_row == bk[j]->mrq->row) && (bk[j]->mrq->rw == READ) &&
        (WTRc == 0) && (bk[j]->state == BANK_ACTIVE) && !rwq->full()) {
      if (rw == WRITE) {
        rw = READ;
        rwq->set_min_length(m_config->CL);
      }
      rwq->push(bk[j]->mrq);
      bk[j]->mrq->txbytes += m_config->dram_atom_size;
      CCDc = m_config->tCCD;
      bkgrp[grp]->CCDLc = m_config->tCCDL;
      RTWc = m_config->tRTW;
      bk[j]->RTPc = m_config->BL / m_config->data_command_freq_ratio;
      bkgrp[grp]->RTPLc = m_config->tRTPL;
      issued = true;
      if (bk[j]->mrq->data->get_access_type() == L2_WR_ALLOC_R)
        n_rd_L2_A++;
      else
        n_rd++;

      bwutil += m_config->BL / m_config->data_command_freq_ratio;
      bwutil_partial += m_config->BL / m_config->data_command_freq_ratio;
      bk[j]->n_access++;

#ifdef DRAM_VERIFY
      PRINT_CYCLE = 1;
      printf("\tRD  Bk:%d Row:%03x Col:%03x \n", j, bk[j]->curr_row,
             bk[j]->mrq->col + bk[j]->mrq->txbytes - m_config->dram_atom_size);
#endif
      // transfer done
      if (!(bk[j]->mrq->txbytes < bk[j]->mrq->nbytes)) {
        bk[j]->mrq = NULL;
      }
    } else
      // correct row activated for a WRITE
      if (!issued && !CCDc && !bk[j]->RCDWRc && !(bkgrp[grp]->CCDLc) &&
          (bk[j]->curr_row == bk[j]->mrq->row) && (bk[j]->mrq->rw == WRITE) &&
          (RTWc == 0) && (bk[j]->state == BANK_ACTIVE) && !rwq->full()) {
        if (rw == READ) {
          rw = WRITE;
          rwq->set_min_length(m_config->WL);
        }
        rwq->push(bk[j]->mrq);

        bk[j]->mrq->txbytes += m_config->dram_atom_size;
        CCDc = m_config->tCCD;
        bkgrp[grp]->CCDLc = m_config->tCCDL;
        WTRc = m_config->tWTR;
        bk[j]->WTPc = m_config->tWTP;
        issued = true;

      //在V100中，当L2 cache写不命中时，采取lazy_fetch_on_read策略，当找到一个cache block
      //逐出时，如果这个cache block是被MODIFIED，则需要将这个cache block写回到下一级存储，
      //因此会产生L2_WRBK_ACC访问，这个访问就是为了写回被逐出的MODIFIED cache block。
        if (bk[j]->mrq->data->get_access_type() == L2_WRBK_ACC)
          n_wr_WB++;
        else
          n_wr++;
        bwutil += m_config->BL / m_config->data_command_freq_ratio;
        bwutil_partial += m_config->BL / m_config->data_command_freq_ratio;
#ifdef DRAM_VERIFY
        PRINT_CYCLE = 1;
        printf(
            "\tWR  Bk:%d Row:%03x Col:%03x \n", j, bk[j]->curr_row,
            bk[j]->mrq->col + bk[j]->mrq->txbytes - m_config->dram_atom_size);
#endif
        // transfer done
        if (!(bk[j]->mrq->txbytes < bk[j]->mrq->nbytes)) {
          bk[j]->mrq = NULL;
        }
      }
  }

  return issued;
}

bool dram_t::issue_row_command(int j) {
  bool issued = false;
  //传入参数bank号j，计算第j个bank属于哪个bank group。
  unsigned grp = get_bankgrp_number(j);
  if (bk[j]->mrq) {  // if currently servicing a memory request
    //如果第j个bank的memory request有效，即第j个bank正在服务bk[j]->mrq（memory request）。
    bk[j]->mrq->data->set_status(
        IN_PARTITION_DRAM, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
    //     bank is idle
    // else

    //若!RRDc为true，表示RRDc为0，即RRDc计数器已经计数完毕，Row to Row Delay计数已完成。
    //若第j个bank的状态时BANK_IDLE，表示第j个bank处于空闲状态。
    //若!bk[j]->RPc为true，代表位线都 precharge 预充电成功。
    //    RPc（Row Precharge Time，又可以称为：Precharge to Active）：内存行地址控制器预
    //    充电时间，一般单位为单位时间周期。
    //    在读取 cell 行（之后也称作单元行）前，需要把每根位线都 precharge（预充电）到电容
    //    电压/供电电压最大值的一半，如果供电电压是 3 V，那么就预充电到 1.5 V。预充电完毕后
    //    打开字线，单元行中每个 cell 电容或是向位线放电，或是由位线充电。放电者位线电压上升
    //    一点，充电者位线电压下降一点。放大器可以捕捉位线上的电压波动，继而在本地还原、暂存
    //    对应 cell 电压。
    //若!bk[j]->RCc为true，同一bank两次行激活命令所间隔的最小时间间隔已经完成。
    //    row cycle time ie. precharge current, then activate different row.
    //    RCc（Row Cycle Time）：定义了同一bank两次行激活命令所间隔的最小时间，或者说是一个
    //    bank中完成一次行操作周期（Row Cycle）的时间。
    if (!issued && !RRDc && (bk[j]->state == BANK_IDLE) && !bk[j]->RPc &&
        !bk[j]->RCc) {  //
#ifdef DRAM_VERIFY
      PRINT_CYCLE = 1;
      printf("\tACT BK:%d NewRow:%03x From:%03x \n", j, bk[j]->mrq->row,
             bk[j]->curr_row);
#endif
      // activate the row with current memory request
      //依据当前memory request的行号，激活第j个bank的对应的row。
      bk[j]->curr_row = bk[j]->mrq->row;
      //设置第j个bank的状态为BANK_ACTIVE。
      bk[j]->state = BANK_ACTIVE;
      //重置Row to Row Delay.
      RRDc = m_config->tRRD;
      //重置Row to Column delay - time required to activate a row before a read.
      //RCD为Row Command Delay的缩写，是一个Row被Active之后，数据从DRAM cell到Sense Amp的
      //拍数。
      bk[j]->RCDc = m_config->tRCD;
      //重置Row to Column delay for a write command - time required to activate a row 
      //before a write.
      bk[j]->RCDWRc = m_config->tRCDWR;
      //重置Active to Prechage Delay. 
      //RAS（RAS Active Time，又可以称为：Active to Prechage Delay）：预充电至内存行激活的
      //最短间隔周期。
      bk[j]->RASc = m_config->tRAS;
      //重置row cycle time ie. precharge current, then activate different row.
      //RCc（Row Cycle Time）：定义了同一bank两次行激活命令所间隔的最小时间，或者说是一个bank
      //中完成一次行操作周期（Row Cycle）的时间。
      bk[j]->RCc = m_config->tRC;
      prio = (j + 1) % m_config->nbk;
      //issue_row_command成功标志。
      issued = true;
      n_act_partial++;
      n_act++;
    }

    else
      // different row activated
      //bk[j]->curr_row != bk[j]->mrq->row 说明当前第j个bank已经打开的行和当前mrq的所在行不
      //一致。
      //若!bk[j]->RASc为真，则说明从激活状态转至预充电时的延时已经完成。
      //    RAS（RAS Active Time，又可以称为：Active to Prechage Delay）：预充电至内存行激
      //    活的最短周期。
      //若!bk[j]->WTPc为真，则说明同一bank从写入状态到预充电的最小时间间隔已经完成。
      //    WTPc: time to switch from write to precharge in the same bank.
      //若!bk[j]->RTPc为真，则说明同一bank从读取状态到预充电的最小时间间隔已经完成。
      //    WTPc: time to switch from read to precharge in the same bank.
      //若!bkgrp[grp]->RTPLc为真，则说明同一bank group从读取到预充电的最小时间间隔已经完成。
      //    RTPLc: read to precharge delay between accesses to different bank groups.
      if ((!issued) && (bk[j]->curr_row != bk[j]->mrq->row) &&
          (bk[j]->state == BANK_ACTIVE) &&
          (!bk[j]->RASc && !bk[j]->WTPc && !bk[j]->RTPc &&
           !bkgrp[grp]->RTPLc)) {
        // make the bank idle again
        //设置第j个bank的状态为BANK_IDLE。
        bk[j]->state = BANK_IDLE;
        //开启预充电延迟的计数器倒计时。
        //    RPc（Row Precharge Time，又可以称为：Precharge to Active）：内存行地址控制器预
        //    充电时间，一般单位为单位时间周期。
        bk[j]->RPc = m_config->tRP;
        prio = (j + 1) % m_config->nbk;
        issued = true;
        n_pre++;
        n_pre_partial++;
#ifdef DRAM_VERIFY
        PRINT_CYCLE = 1;
        printf("\tPRE BK:%d Row:%03x \n", j, bk[j]->curr_row);
#endif
      }
  }
  return issued;
}

// if mrq is being serviced by dram, gets popped after CL latency fulfilled
class mem_fetch *dram_t::return_queue_pop() {
  return returnq->pop();
}

class mem_fetch *dram_t::return_queue_top() {
  return returnq->top();
}

void dram_t::print(FILE *simFile) const {
  unsigned i;
  fprintf(simFile, "DRAM[%d]: %d bks, busW=%d BL=%d CL=%d, ", id, m_config->nbk,
          m_config->busW, m_config->BL, m_config->CL);
  fprintf(simFile, "tRRD=%d tCCD=%d, tRCD=%d tRAS=%d tRP=%d tRC=%d\n",
          m_config->tRRD, m_config->tCCD, m_config->tRCD, m_config->tRAS,
          m_config->tRP, m_config->tRC);
  fprintf(
      simFile,
      "n_cmd=%llu n_nop=%llu n_act=%llu n_pre=%llu n_ref_event=%llu n_req=%llu "
      "n_rd=%llu n_rd_L2_A=%llu n_write=%llu n_wr_bk=%llu bw_util=%.4g\n",
      n_cmd, n_nop, n_act, n_pre, n_ref, n_req, n_rd, n_rd_L2_A, n_wr, n_wr_WB,
      (float)bwutil / n_cmd);
  fprintf(simFile, "n_activity=%llu dram_eff=%.4g\n", n_activity,
          (float)bwutil / n_activity);
  for (i = 0; i < m_config->nbk; i++) {
    fprintf(simFile, "bk%d: %da %di ", i, bk[i]->n_access, bk[i]->n_idle);
  }
  fprintf(simFile, "\n");
  fprintf(simFile,
          "\n------------------------------------------------------------------"
          "------\n");

  printf("\nRow_Buffer_Locality = %.6f", (float)hits_num / access_num);
  printf("\nRow_Buffer_Locality_read = %.6f", (float)hits_read_num / read_num);
  printf("\nRow_Buffer_Locality_write = %.6f",
         (float)hits_write_num / write_num);
  printf("\nBank_Level_Parallism = %.6f",
         (float)banks_1time / banks_acess_total);
  printf("\nBank_Level_Parallism_Col = %.6f",
         (float)banks_time_rw / banks_access_rw_total);
  printf("\nBank_Level_Parallism_Ready = %.6f",
         (float)banks_time_ready / banks_access_ready_total);
  printf("\nwrite_to_read_ratio_blp_rw_average = %.6f",
         write_to_read_ratio_blp_rw_average / banks_access_rw_total);
  printf("\nGrpLevelPara = %.6f \n",
         (float)bkgrp_parallsim_rw / banks_access_rw_total);

  printf("\nBW Util details:\n");
  printf("bwutil = %.6f \n", (float)bwutil / n_cmd);
  printf("total_CMD = %llu \n", n_cmd);
  printf("util_bw = %llu \n", util_bw);
  printf("Wasted_Col = %llu \n", wasted_bw_col);
  printf("Wasted_Row = %llu \n", wasted_bw_row);
  printf("Idle = %llu \n", idle_bw);

  printf("\nBW Util Bottlenecks: \n");
  printf("RCDc_limit = %llu \n", RCDc_limit);
  printf("RCDWRc_limit = %llu \n", RCDWRc_limit);
  printf("WTRc_limit = %llu \n", WTRc_limit);
  printf("RTWc_limit = %llu \n", RTWc_limit);
  printf("CCDLc_limit = %llu \n", CCDLc_limit);
  printf("rwq = %llu \n", rwq_limit);
  printf("CCDLc_limit_alone = %llu \n", CCDLc_limit_alone);
  printf("WTRc_limit_alone = %llu \n", WTRc_limit_alone);
  printf("RTWc_limit_alone = %llu \n", RTWc_limit_alone);

  printf("\nCommands details: \n");
  printf("total_CMD = %llu \n", n_cmd);
  printf("n_nop = %llu \n", n_nop);
  printf("Read = %llu \n", n_rd);
  printf("Write = %llu \n", n_wr);
  printf("L2_Alloc = %llu \n", n_rd_L2_A);
  printf("L2_WB = %llu \n", n_wr_WB);
  printf("n_act = %llu \n", n_act);
  printf("n_pre = %llu \n", n_pre);
  printf("n_ref = %llu \n", n_ref);
  printf("n_req = %llu \n", n_req);
  printf("total_req = %llu \n", n_rd + n_wr + n_rd_L2_A + n_wr_WB);

  printf("\nDual Bus Interface Util: \n");
  printf("issued_total_row = %llu \n", issued_total_row);
  printf("issued_total_col = %llu \n", issued_total_col);
  printf("Row_Bus_Util =  %.6f \n", (float)issued_total_row / n_cmd);
  printf("CoL_Bus_Util = %.6f \n", (float)issued_total_col / n_cmd);
  printf("Either_Row_CoL_Bus_Util = %.6f \n", (float)issued_total / n_cmd);
  printf("Issued_on_Two_Bus_Simul_Util = %.6f \n", (float)issued_two / n_cmd);
  printf("issued_two_Eff = %.6f \n", (float)issued_two / issued_total);
  printf("queue_avg = %.6f \n\n", (float)ave_mrqs / n_cmd);

  fprintf(simFile, "\n");
  fprintf(simFile, "dram_util_bins:");
  for (i = 0; i < 10; i++) fprintf(simFile, " %d", dram_util_bins[i]);
  fprintf(simFile, "\ndram_eff_bins:");
  for (i = 0; i < 10; i++) fprintf(simFile, " %d", dram_eff_bins[i]);
  fprintf(simFile, "\n");
  if (m_config->scheduler_type == DRAM_FRFCFS)
    fprintf(simFile, "mrqq: max=%d avg=%g\n", max_mrqs,
            (float)ave_mrqs / n_cmd);
}

void dram_t::visualize() const {
  printf("RRDc=%d CCDc=%d mrqq.Length=%d rwq.Length=%d\n", RRDc, CCDc,
         mrqq->get_length(), rwq->get_length());
  for (unsigned i = 0; i < m_config->nbk; i++) {
    printf("BK%d: state=%c curr_row=%03x, %2d %2d %2d %2d %p ", i, bk[i]->state,
           bk[i]->curr_row, bk[i]->RCDc, bk[i]->RASc, bk[i]->RPc, bk[i]->RCc,
           bk[i]->mrq);
    if (bk[i]->mrq)
      printf("txf: %d %d", bk[i]->mrq->nbytes, bk[i]->mrq->txbytes);
    printf("\n");
  }
  if (m_frfcfs_scheduler) m_frfcfs_scheduler->print(stdout);
}

void dram_t::print_stat(FILE *simFile) {
  fprintf(simFile,
          "DRAM (%u): n_cmd=%llu n_nop=%llu n_act=%llu n_pre=%llu n_ref=%llu "
          "n_req=%llu n_rd=%llu n_write=%llu bw_util=%.4g ",
          id, n_cmd, n_nop, n_act, n_pre, n_ref, n_req, n_rd, n_wr,
          (float)bwutil / n_cmd);
  fprintf(simFile, "mrqq: %d %.4g mrqsmax=%llu ", max_mrqs,
          (float)ave_mrqs / n_cmd, max_mrqs_temp);
  fprintf(simFile, "\n");
  fprintf(simFile, "dram_util_bins:");
  for (unsigned i = 0; i < 10; i++) fprintf(simFile, " %d", dram_util_bins[i]);
  fprintf(simFile, "\ndram_eff_bins:");
  for (unsigned i = 0; i < 10; i++) fprintf(simFile, " %d", dram_eff_bins[i]);
  fprintf(simFile, "\n");
  max_mrqs_temp = 0;
}

void dram_t::visualizer_print(gzFile visualizer_file) {
  // dram specific statistics
  gzprintf(visualizer_file, "dramncmd: %u %u\n", id, n_cmd_partial);
  gzprintf(visualizer_file, "dramnop: %u %u\n", id, n_nop_partial);
  gzprintf(visualizer_file, "dramnact: %u %u\n", id, n_act_partial);
  gzprintf(visualizer_file, "dramnpre: %u %u\n", id, n_pre_partial);
  gzprintf(visualizer_file, "dramnreq: %u %u\n", id, n_req_partial);
  gzprintf(visualizer_file, "dramavemrqs: %u %u\n", id,
           n_cmd_partial ? (ave_mrqs_partial / n_cmd_partial) : 0);

  // utilization and efficiency
  gzprintf(visualizer_file, "dramutil: %u %u\n", id,
           n_cmd_partial ? 100 * bwutil_partial / n_cmd_partial : 0);
  gzprintf(visualizer_file, "drameff: %u %u\n", id,
           n_activity_partial ? 100 * bwutil_partial / n_activity_partial : 0);

  // reset for next interval
  bwutil_partial = 0;
  n_activity_partial = 0;
  ave_mrqs_partial = 0;
  n_cmd_partial = 0;
  n_nop_partial = 0;
  n_act_partial = 0;
  n_pre_partial = 0;
  n_req_partial = 0;

  // dram access type classification
  for (unsigned j = 0; j < m_config->nbk; j++) {
    gzprintf(visualizer_file, "dramglobal_acc_r: %u %u %u\n", id, j,
             m_stats->mem_access_type_stats[GLOBAL_ACC_R][id][j]);
    gzprintf(visualizer_file, "dramglobal_acc_w: %u %u %u\n", id, j,
             m_stats->mem_access_type_stats[GLOBAL_ACC_W][id][j]);
    gzprintf(visualizer_file, "dramlocal_acc_r: %u %u %u\n", id, j,
             m_stats->mem_access_type_stats[LOCAL_ACC_R][id][j]);
    gzprintf(visualizer_file, "dramlocal_acc_w: %u %u %u\n", id, j,
             m_stats->mem_access_type_stats[LOCAL_ACC_W][id][j]);
    gzprintf(visualizer_file, "dramconst_acc_r: %u %u %u\n", id, j,
             m_stats->mem_access_type_stats[CONST_ACC_R][id][j]);
    gzprintf(visualizer_file, "dramtexture_acc_r: %u %u %u\n", id, j,
             m_stats->mem_access_type_stats[TEXTURE_ACC_R][id][j]);
  }
}

void dram_t::set_dram_power_stats(unsigned &cmd, unsigned &activity,
                                  unsigned &nop, unsigned &act, unsigned &pre,
                                  unsigned &rd, unsigned &wr, unsigned &wr_WB,
                                  unsigned &req) const {
  // Point power performance counters to low-level DRAM counters
  cmd = n_cmd;
  activity = n_activity;
  nop = n_nop;
  act = n_act;
  pre = n_pre;
  rd = n_rd;
  wr = n_wr;
  wr_WB = n_wr_WB;
  req = n_req;
}

/*
传入参数bank号，计算属于哪个bank group。
*/
unsigned dram_t::get_bankgrp_number(unsigned i) {
  if (m_config->dram_bnkgrp_indexing_policy == HIGHER_BITS) {  // higher bits
    return i >> m_config->bk_tag_length;
  } else if (m_config->dram_bnkgrp_indexing_policy ==
             LOWER_BITS) {  // lower bits
    //低位与运算。
    return i & ((m_config->nbkgrp - 1));
  } else {
    assert(1);
  }
  return 0;  // we should never get here
}
