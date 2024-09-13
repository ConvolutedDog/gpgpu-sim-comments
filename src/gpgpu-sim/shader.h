// Copyright (c) 2009-2021, Tor M. Aamodt, Wilson W.L. Fung, Andrew Turner,
// Ali Bakhoda, Vijay Kandiah, Nikos Hardavellas,
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

#ifndef SHADER_H
#define SHADER_H

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <bitset>
#include <deque>
#include <list>
#include <map>
#include <set>
#include <utility>
#include <vector>

//#include "../cuda-sim/ptx.tab.h"

#include "../abstract_hardware_model.h"
#include "delayqueue.h"
#include "dram.h"
#include "gpu-cache.h"
#include "mem_fetch.h"
#include "scoreboard.h"
#include "stack.h"
#include "stats.h"
#include "traffic_breakdown.h"

#define NO_OP_FLAG 0xFF

/* READ_PACKET_SIZE:
   bytes: 6 address (flit can specify chanel so this gives up to ~2GB/channel,
   so good for now), 2 bytes   [shaderid + mshrid](14 bits) + req_size(0-2 bits
   if req_size variable) - so up to 2^14 = 16384 mshr total
 */

#define READ_PACKET_SIZE 8

// WRITE_PACKET_SIZE: bytes: 6 address, 2 miscelaneous.
#define WRITE_PACKET_SIZE 8

#define WRITE_MASK_SIZE 8

class gpgpu_context;

enum exec_unit_type_t {
  NONE = 0,
  SP = 1,
  SFU = 2,
  MEM = 3,
  DP = 4,
  INT = 5,
  TENSOR = 6,
  SPECIALIZED = 7
};

/*
线程的状态上下文。
*/
class thread_ctx_t {
 public:
  //该线程所属的CTA的ID。
  unsigned m_cta_id;  // hardware CTA this thread belongs

  // per thread stats (ac stands for accumulative).
  //该线程内处理的指令总条数。
  unsigned n_insn;
  unsigned n_insn_ac;
  unsigned n_l1_mis_ac;
  unsigned n_l1_mrghit_ac;
  unsigned n_l1_access_ac;
  //标识线程是否处于活跃状态。
  bool m_active;
};

/*
模拟了内核中warp的模拟状态，是一个性能模拟过程中的warp的对象。SIMT Core就是一个shd_warp_t对象的集合，
它模拟了内核中每个warp的模拟状态。手册中<<#Simt-core图>>所示的I-Buffer被实现在shader_core_ctx内部的
shd_warp_t对象中。每个shd_warp_t都有一组m_ibuffer的I-Buffer条目(ibuffer_entry)，持有可配置的指令数
量（一个周期内允许获取的最大指令）。另外，shd_warp_t有一些标志，这些标志被调度器用来确定warp的发射资格。
存储在ibuffer_entry中的解码指令是一个指向warp_inst_t对象的指针。warp_inst_t持有关于这条指令的操作类
型和所用操作数的信息。
*/
class shd_warp_t {
 public:
  //构造函数。参数分别为：
  //    shader_core_ctx *shader：SIMT Core的对象；
  //    unsigned warp_size：单个warp内的线程数量，warp的大小。
  shd_warp_t(class shader_core_ctx *shader, unsigned warp_size)
      : m_shader(shader), m_warp_size(warp_size) {
    //初始化已发送但尚未确认的存储请求数为零。
    m_stores_outstanding = 0;
    //初始化在流水线中执行的指令数为零。
    m_inst_in_pipeline = 0;
    reset();
  }
  //初始化。
  void reset() {
    assert(m_stores_outstanding == 0);
    assert(m_inst_in_pipeline == 0);
    //初始化设置warp因指令缓冲未命中而挂起的状态为false。
    m_imiss_pending = false;
    //warp ID初始化为-1。
    m_warp_id = (unsigned)-1;
    //动态warp ID初始化为-1。
    m_dynamic_warp_id = (unsigned)-1;
    //设置已经完成的线程的数量为warp大小。后面还需要将活跃的线程数减掉。
    n_completed = m_warp_size;
    //设置未完成的原子操作数为零。名字里的n代表not。
    m_n_atomic = 0;
    //设置warp处于memory barrier状态的标识为false。
    m_membar = false;
    //设置线程退出的标识为true。
    m_done_exit = true;
    //设置上次取指的时钟周期，时刻值为零时刻。
    m_last_fetch = 0;
    //设置指令缓冲中下一条待取的指令的编号为零。
    m_next = 0;
    m_streamID = (unsigned long long)-1;

    // Jin: cdp support
    m_cdp_latency = 0;
    m_cdp_dummy = false;

    // Ni: Initialize ldgdepbar_id
    m_ldgdepbar_id = 0;
    m_depbar_start_id = 0;
    m_depbar_group = 0;

    // Ni: Set waiting to false
    m_waiting_ldgsts = false;

    // Ni: Clear m_ldgdepbar_buf
    for (unsigned i = 0; i < m_ldgdepbar_buf.size(); i++) {
      m_ldgdepbar_buf[i].clear();
    }
    m_ldgdepbar_buf.clear();
  }
  void init(address_type start_pc, unsigned cta_id, unsigned wid,
            const std::bitset<MAX_WARP_SIZE> &active, unsigned dynamic_warp_id,
            unsigned long long streamID) {
    m_streamID = streamID;
    m_cta_id = cta_id;
    m_warp_id = wid;
    m_dynamic_warp_id = dynamic_warp_id;
    m_next_pc = start_pc;
    assert(n_completed >= active.count());
    assert(n_completed <= m_warp_size);
    //将已经完成的线程的数量初始值，减去活跃线程的数量，因为活跃线程代表它们尚未完成执行。
    n_completed -= active.count();  // active threads are not yet completed
    //设置活跃线程的位图，为参数active。
    m_active_threads = active;
    //设置线程退出的标识为false。
    m_done_exit = false;

    // Jin: cdp support
    m_cdp_latency = 0;
    m_cdp_dummy = false;

    // Ni: Initialize ldgdepbar_id
    m_ldgdepbar_id = 0;
    m_depbar_start_id = 0;
    m_depbar_group = 0;

    // Ni: Set waiting to false
    m_waiting_ldgsts = false;

    // Ni: Clear m_ldgdepbar_buf
    for (unsigned i = 0; i < m_ldgdepbar_buf.size(); i++) {
      m_ldgdepbar_buf[i].clear();
    }
    m_ldgdepbar_buf.clear();
  }
  //返回warp已经执行完毕的标志，已经完成的线程数量=warp的大小时，就代表该warp已经完成。
  bool functional_done() const;
  //返回warp是否由于（warp已经执行完毕且在等待新内核初始化、CTA处于barrier、memory barrier、还有未
  //完成的原子操作）四个条件处于等待状态。
  bool waiting();  // not const due to membar
  //hardware_done()检查这个warp是否已经完成执行并且可以回收。
  bool hardware_done() const;
  //返回线程退出的标识。
  bool done_exit() const { return m_done_exit; }
  //设置线程退出的标识。
  void set_done_exit() { m_done_exit = true; }

  void print(FILE *fout) const;
  void print_ibuffer(FILE *fout) const;
  //返回单个warp中已经执行完毕的线程数量。
  unsigned get_n_completed() const { return n_completed; }
  //增加单个warp中已经执行完毕的线程数量，m_active_threads是活跃线程的位图，为1代表一个线程处于活跃
  //状态，这里将其reset为零，并增加n_completed。
  void set_completed(unsigned lane) {
    assert(m_active_threads.test(lane));
    m_active_threads.reset(lane);
    n_completed++;
  }
  //设置上次取指的时钟周期，时刻值为 sim_cycle。
  void set_last_fetch(unsigned long long sim_cycle) {
    m_last_fetch = sim_cycle;
  }
  //返回未完成的原子操作数。
  unsigned get_n_atomic() const { return m_n_atomic; }
  //增加未完成的原子操作数。
  void inc_n_atomic() { m_n_atomic++; }
  //减掉未完成的原子操作数。
  void dec_n_atomic(unsigned n) { m_n_atomic -= n; }
  //设置内存屏障状态的标识为true，warp正在memory barrier处等待。
  void set_membar() { m_membar = true; }
  //清除内存屏障状态的标识，重置为false，即此刻warp没有在memory barrier处等待。
  void clear_membar() { m_membar = false; }
  //返回内存屏障状态的标识。
  bool get_membar() const { return m_membar; }
  //返回warp内下一个要执行的指令的PC值。
  virtual address_type get_pc() const { return m_next_pc; }
  //返回绑定在当前Shader Core的kernel的内核函数信息，kernel_info_t对象。
  virtual kernel_info_t *get_kernel_info() const;
  //设置warp内下一个要执行的指令的PC值。
  void set_next_pc(address_type pc) { m_next_pc = pc; }
  //保存上一条处于屏障指令处的指令。
  void store_info_of_last_inst_at_barrier(const warp_inst_t *pI) {
    m_inst_at_barrier = *pI;
  }
  //返回上一条处于屏障指令处的指令。
  warp_inst_t *restore_info_of_last_inst_at_barrier() {
    return &m_inst_at_barrier;
  }
  //将一条新指令存入I-Bufer。传入的参数：
  //    unsigned slot：存入I-Bufer的槽编号；
  //    warp_inst_t *pI：存入的指令。
  void ibuffer_fill(unsigned slot, const warp_inst_t *pI) {
    assert(slot < IBUFFER_SIZE);
    m_ibuffer[slot].m_inst = pI;
    m_ibuffer[slot].m_valid = true;
    //指令缓冲中下一条待取的指令的编号。
    m_next = 0;
  }
  //返回I-Bufer是否为空。
  bool ibuffer_empty() const {
    //遍历I-Bufer所有槽，有一个有效的话就返回false。
    for (unsigned i = 0; i < IBUFFER_SIZE; i++)
      if (m_ibuffer[i].m_valid) return false;
    return true;
  }
  //清除I-Buffer中的所有槽。
  void ibuffer_flush() {
    for (unsigned i = 0; i < IBUFFER_SIZE; i++) {
      if (m_ibuffer[i].m_valid) dec_inst_in_pipeline();
      m_ibuffer[i].m_inst = NULL;
      m_ibuffer[i].m_valid = false;
    }
  }
  //返回I-Buffer中的下一条待取的指令。
  const warp_inst_t *ibuffer_next_inst() { return m_ibuffer[m_next].m_inst; }
  //返回I-Buffer中的下一条待取的指令是否有效。
  bool ibuffer_next_valid() { return m_ibuffer[m_next].m_valid; }
  //释放I-Buffer中的下一条待取的指令槽。
  void ibuffer_free() {
    m_ibuffer[m_next].m_inst = NULL;
    m_ibuffer[m_next].m_valid = false;
  }
  //刷新m_next的值，I-Buffer中下一条待取的指令槽。
  void ibuffer_step() { m_next = (m_next + 1) % IBUFFER_SIZE; }
  //返回warp是否因指令缓冲未命中而挂起的状态标识。
  bool imiss_pending() const { return m_imiss_pending; }
  //设置warp因指令缓冲未命中而挂起的状态。
  void set_imiss_pending() { m_imiss_pending = true; }
  //清除warp因指令缓冲未命中而挂起的状态。
  void clear_imiss_pending() { m_imiss_pending = false; }
  //返回所有store访存请求是否已经全部执行完，已发送但尚未确认的存储请求数m_stores_outstanding=0时，
  //代表所有store访存请求已经全部执行完。
  bool stores_done() const { return m_stores_outstanding == 0; }
  //增加已发送但尚未确认的存储请求数。
  void inc_store_req() { m_stores_outstanding++; }
  //减少已发送但尚未收到写确认的存储请求数。
  void dec_store_req() {
    assert(m_stores_outstanding > 0);
    m_stores_outstanding--;
  }
  //返回I-Buffer中的有效指令的总数。
  unsigned num_inst_in_buffer() const {
    unsigned count = 0;
    for (unsigned i = 0; i < IBUFFER_SIZE; i++) {
      if (m_ibuffer[i].m_valid) count++;
    }
    return count;
  }
  //返回在流水线中执行的指令数。
  unsigned num_inst_in_pipeline() const { return m_inst_in_pipeline; }
  //返回已经发射到流水线中的指令数，但这个函数貌似计算不对，也没有被用到，暂时先不管。
  unsigned num_issued_inst_in_pipeline() const {
    return (num_inst_in_pipeline() - num_inst_in_buffer());
  }
  //返回是否在流水线中有正在执行的指令。
  bool inst_in_pipeline() const { return m_inst_in_pipeline > 0; }
  //增加在流水线中执行的指令数。
  void inc_inst_in_pipeline() { m_inst_in_pipeline++; }
  //减少在流水线中执行的指令数。
  void dec_inst_in_pipeline() {
    assert(m_inst_in_pipeline > 0);
    m_inst_in_pipeline--;
  }
  unsigned long long get_streamID() const { return m_streamID; }
  //返回warp所在的CTA的ID。
  unsigned get_cta_id() const { return m_cta_id; }
  //返回动态warp的ID。
  unsigned get_dynamic_warp_id() const { return m_dynamic_warp_id; }
  //返回warp的ID。
  unsigned get_warp_id() const { return m_warp_id; }
  //返回warp所在的SIMT Core对象。
  class shader_core_ctx *get_shader() {
    return m_shader;
  }

 private:
  //设置指令缓冲的大小为2。
  static const unsigned IBUFFER_SIZE = 2;
  //SIMT Core的对象。
  class shader_core_ctx *m_shader;
  unsigned long long m_streamID;
  //warp所在的CTA的ID。
  unsigned m_cta_id;
  unsigned m_warp_id;
  //单个warp内的线程数量，warp的大小。
  unsigned m_warp_size;
  //动态warp的ID。
  unsigned m_dynamic_warp_id;
  //warp内下一个要执行的指令的PC值，在shd_warp_t对象初建时，被设置为start_pc。
  address_type m_next_pc;
  //单个warp中已经执行完毕的线程数量，当此线程数量达到32时，代表一个warp执行完毕。
  unsigned n_completed;  // number of threads in warp completed
  //活跃线程的位图，为1代表一个线程处于活跃状态。
  std::bitset<MAX_WARP_SIZE> m_active_threads;
  //标识是否因指令缓冲未命中而挂起的状态。
  bool m_imiss_pending;

  //指令缓冲的条目结构。
  struct ibuffer_entry {
    ibuffer_entry() {
      //初始化条目的有效位。
      m_valid = false;
      //初始化条目内储存的指令。
      m_inst = NULL;
    }
    //条目内储存的指令。
    const warp_inst_t *m_inst;
    //条目的有效位。
    bool m_valid;
  };

  warp_inst_t m_inst_at_barrier;
  //IBUFFER_SIZE大小的指令缓冲。
  ibuffer_entry m_ibuffer[IBUFFER_SIZE];
  //I-Buffer中下一条待取的指令槽。
  unsigned m_next;
  //未完成的原子操作数。
  unsigned m_n_atomic;  // number of outstanding atomic operations
  //内存屏障状态的标识，如果为true，则warp正在memory barrier处等待。
  bool m_membar;        // if true, warp is waiting at memory barrier

  //线程退出的标识，一旦为该warp中的线程注册了线程退出，则为true。
  bool m_done_exit;  // true once thread exit has been registered for threads in
                     // this warp

  //上次取指的时钟周期，时刻值.
  unsigned long long m_last_fetch;
  //已发送但尚未确认的存储请求数。
  unsigned m_stores_outstanding;  // number of store requests sent but not yet
                                  // acknowledged
  //在流水线中执行的指令数。
  unsigned m_inst_in_pipeline;

  // Jin: cdp support
 public:
  unsigned int m_cdp_latency;
  bool m_cdp_dummy;

  // Ni: LDGDEPBAR barrier support
 public:
  unsigned int m_ldgdepbar_id;  // LDGDEPBAR barrier ID
  std::vector<std::vector<warp_inst_t>>
      m_ldgdepbar_buf;  // LDGDEPBAR barrier buffer
  unsigned int m_depbar_start_id;
  unsigned int m_depbar_group;
  bool m_waiting_ldgsts;  // Ni: Whether the warp is waiting for the LDGSTS
                          // instrs to finish
};

inline unsigned hw_tid_from_wid(unsigned wid, unsigned warp_size, unsigned i) {
  return wid * warp_size + i;
};
inline unsigned wid_from_hw_tid(unsigned tid, unsigned warp_size) {
  return tid / warp_size;
};

const unsigned WARP_PER_CTA_MAX = 64;
//单个CTA内的所有warp数量大小的位图，后面可用于多种功能。
typedef std::bitset<WARP_PER_CTA_MAX> warp_set_t;

unsigned register_bank(int regnum, int wid, unsigned num_banks,
                       bool sub_core_model, unsigned banks_per_sched,
                       unsigned sched_id);

class shader_core_ctx;
class shader_core_config;
class shader_core_stats;

enum scheduler_prioritization_type {
  SCHEDULER_PRIORITIZATION_LRR = 0,   // Loose Round Robin
  SCHEDULER_PRIORITIZATION_SRR,       // Strict Round Robin
  SCHEDULER_PRIORITIZATION_GTO,       // Greedy Then Oldest
  SCHEDULER_PRIORITIZATION_GTLRR,     // Greedy Then Loose Round Robin
  SCHEDULER_PRIORITIZATION_GTY,       // Greedy Then Youngest
  SCHEDULER_PRIORITIZATION_OLDEST,    // Oldest First
  SCHEDULER_PRIORITIZATION_YOUNGEST,  // Youngest First
};

// Each of these corresponds to a string value in the gpgpsim.config file
// For example - to specify the LRR scheudler the config must contain lrr
enum concrete_scheduler {
  CONCRETE_SCHEDULER_LRR = 0,
  CONCRETE_SCHEDULER_GTO,
  CONCRETE_SCHEDULER_TWO_LEVEL_ACTIVE,
  CONCRETE_SCHEDULER_RRR,
  CONCRETE_SCHEDULER_WARP_LIMITING,
  CONCRETE_SCHEDULER_OLDEST_FIRST,
  NUM_CONCRETE_SCHEDULERS
};

/*
调度单元类。每个对象负责从其warp集中选择一条或多条指令，并发射这些指令进行执行。单个Shader Core里有
可配置数量的调度器单元。从下面的代码可以看到，调度器单元内含有一个scoreboard，一个SIMT栈，一个可供本
调度器单元仲裁的warp子集合m_supervised_warps，还有一堆用于指令发射的sp_out等发射出口。调度器单元的
核心方法是cycle()，它会被派生类覆盖，以实现不同的调度策略。调度器单元的构造函数中，参数分别为：
    shader_core_stats *stats：SIMT Core的统计信息对象；
    shader_core_ctx *shader：SIMT Core对象；
    Scoreboard *scoreboard：SIMT Core的记分牌对象；
    simt_stack **simt：SIMT栈；
    std::vector<shd_warp_t *> *warp：SIMT Core内的所有warp；
    register_set *sp_out：SP单元的发射出口；
    register_set *dp_out：DP单元的发射出口；
    register_set *sfu_out：SFU单元的发射出口；
    register_set *int_out：INT单元的发射出口；
    register_set *tensor_core_out：Tensor Core单元的发射出口；
    std::vector<register_set *> &spec_cores_out：特殊功能单元的发射出口；
    register_set *mem_out：存储器单元的发射出口；
    int id：调度器单元的ID。
*/
class scheduler_unit {  // this can be copied freely, so can be used in std
                        // containers.
 public:
  //构造函数。
  scheduler_unit(shader_core_stats *stats, shader_core_ctx *shader,
                 Scoreboard *scoreboard, simt_stack **simt,
                 std::vector<shd_warp_t *> *warp, register_set *sp_out,
                 register_set *dp_out, register_set *sfu_out,
                 register_set *int_out, register_set *tensor_core_out,
                 std::vector<register_set *> &spec_cores_out,
                 register_set *mem_out, int id)
      : m_supervised_warps(),
        m_stats(stats),
        m_shader(shader),
        m_scoreboard(scoreboard),
        m_simt_stack(simt),
        /*m_pipeline_reg(pipe_regs),*/ m_warp(warp),
        m_sp_out(sp_out),
        m_dp_out(dp_out),
        m_sfu_out(sfu_out),
        m_int_out(int_out),
        m_tensor_core_out(tensor_core_out),
        m_mem_out(mem_out),
        m_spec_cores_out(spec_cores_out),
        m_id(id) {}
  virtual ~scheduler_unit() {}
  virtual void add_supervised_warp_id(int i) {
    m_supervised_warps.push_back(&warp(i));
  }
  virtual void done_adding_supervised_warps() {
    m_last_supervised_issued = m_supervised_warps.end();
  }

  // The core scheduler cycle method is meant to be common between
  // all the derived schedulers.  The scheduler's behaviour can be
  // modified by changing the contents of the m_next_cycle_prioritized_warps
  // list.
  //核心调度器cycle()方法是指在所有派生调度器之间通用。可以通过更改m_next_cycle_prioritized_warps
  //列表的内容来修改调度程序的行为。
  void cycle();

  // These are some common ordering fucntions that the
  // higher order schedulers can take advantage of
  //LRR调度策略的调度器单元的order_warps()函数，为当前调度单元内所划分到的warp进行排序。order_lrr
  //的定义为：
  //     void scheduler_unit::order_lrr(
  //         std::vector<T> &result_list, const typename std::vector<T> &input_list,
  //         const typename std::vector<T>::const_iterator &last_issued_from_input,
  //         unsigned num_warps_to_add)
  //参数列表：
  //result_list：m_next_cycle_prioritized_warps是一个vector，里面存储当前调度单元当前拍经过warp
  //             排序后，在下一拍具有优先级调度的warp。
  //input_list：m_supervised_warps，是一个vector，里面存储当前调度单元所需要仲裁的warp。
  //last_issued_from_input：则存储了当前调度单元上一拍调度过的warp。
  //num_warps_to_add：m_supervised_warps.size()，则是当前调度单元在下一拍需要调度的warp数目，在这
  //                  里这个warp数目就是当前调度器所划分到的warp子集合m_supervised_warps的大小。
  template <typename T>
  void order_lrr(
      typename std::vector<T> &result_list,
      const typename std::vector<T> &input_list,
      const typename std::vector<T>::const_iterator &last_issued_from_input,
      unsigned num_warps_to_add);
  template <typename T>
  void order_rrr(
      typename std::vector<T> &result_list,
      const typename std::vector<T> &input_list,
      const typename std::vector<T>::const_iterator &last_issued_from_input,
      unsigned num_warps_to_add);

  enum OrderingType {
    // The item that issued last is prioritized first then the sorted result
    // of the priority_function
    ORDERING_GREEDY_THEN_PRIORITY_FUNC = 0,
    // No greedy scheduling based on last to issue. Only the priority function
    // determines priority
    ORDERED_PRIORITY_FUNC_ONLY,
    NUM_ORDERING,
  };
  template <typename U>
  void order_by_priority(
      std::vector<U> &result_list, const typename std::vector<U> &input_list,
      const typename std::vector<U>::const_iterator &last_issued_from_input,
      unsigned num_warps_to_add, OrderingType age_ordering,
      bool (*priority_func)(U lhs, U rhs));
  static bool sort_warps_by_oldest_dynamic_id(shd_warp_t *lhs, shd_warp_t *rhs);

  // Derived classes can override this function to populate
  // m_supervised_warps with their scheduling policies
  //派生类可以覆盖此函数，以使用其调度策略填充m_supervisored_warps。
  virtual void order_warps() = 0;

  int get_schd_id() const { return m_id; }

 protected:
  virtual void do_on_warp_issued(
      unsigned warp_id, unsigned num_issued,
      const std::vector<shd_warp_t *>::const_iterator &prioritized_iter);
  inline int get_sid() const;

 protected:
  shd_warp_t &warp(int i);

  // This is the prioritized warp list that is looped over each cycle to
  // determine which warp gets to issue.
  std::vector<shd_warp_t *> m_next_cycle_prioritized_warps;
  // The m_supervised_warps list is all the warps this scheduler is supposed to
  // arbitrate between.  This is useful in systems where there is more than
  // one warp scheduler. In a single scheduler system, this is simply all
  // the warps assigned to this core.
  //m_supervisored_twarps列表是此调度程序应该在其间进行仲裁的所有warps。这在存在多个warp调度器的
  //系统中非常有用。在单个调度器系统中，这只是分配给该核心的所有warp。
  std::vector<shd_warp_t *> m_supervised_warps;
  // This is the iterator pointer to the last supervised warp you issued
  std::vector<shd_warp_t *>::const_iterator m_last_supervised_issued;
  shader_core_stats *m_stats;
  shader_core_ctx *m_shader;
  // these things should become accessors: but would need a bigger rearchitect
  // of how shader_core_ctx interacts with its parts.
  //每个SIMT Core都有一个记分牌。
  Scoreboard *m_scoreboard;
  //对于每个调度器单元，有一个SIMT堆栈阵列。每个SIMT堆栈对应一个warp。
  simt_stack **m_simt_stack;
  // warp_inst_t** m_pipeline_reg;
  std::vector<shd_warp_t *> *m_warp;
  //m_sp_out, m_sfu_out, m_mem_out是指向SP、SFU和Mem流水线接收的发射阶段和执行阶段之间的第一个
  //流水线寄存器。
  //SP单元的发射出口。
  register_set *m_sp_out;
  //DP单元的发射出口。
  register_set *m_dp_out;
  //SFU单元的发射出口。
  register_set *m_sfu_out;
  //INT单元的发射出口。
  register_set *m_int_out;
  //Tensor Core单元的发射出口。
  register_set *m_tensor_core_out;
  //Mem单元的发射出口。
  register_set *m_mem_out;
  std::vector<register_set *> &m_spec_cores_out;
  //记录上一拍发射的指令数。
  unsigned m_num_issued_last_cycle;
  //在RRR调度策略中用到，Volta架构中采用的LRR调度策略，暂时不管。
  unsigned m_current_turn_warp;
  //调度器单元的唯一标识ID。
  int m_id;
};

class lrr_scheduler : public scheduler_unit {
 public:
  lrr_scheduler(shader_core_stats *stats, shader_core_ctx *shader,
                Scoreboard *scoreboard, simt_stack **simt,
                std::vector<shd_warp_t *> *warp, register_set *sp_out,
                register_set *dp_out, register_set *sfu_out,
                register_set *int_out, register_set *tensor_core_out,
                std::vector<register_set *> &spec_cores_out,
                register_set *mem_out, int id)
      : scheduler_unit(stats, shader, scoreboard, simt, warp, sp_out, dp_out,
                       sfu_out, int_out, tensor_core_out, spec_cores_out,
                       mem_out, id) {}
  virtual ~lrr_scheduler() {}
  virtual void order_warps();
  virtual void done_adding_supervised_warps() {
    m_last_supervised_issued = m_supervised_warps.end();
  }
};

class rrr_scheduler : public scheduler_unit {
 public:
  rrr_scheduler(shader_core_stats *stats, shader_core_ctx *shader,
                Scoreboard *scoreboard, simt_stack **simt,
                std::vector<shd_warp_t *> *warp, register_set *sp_out,
                register_set *dp_out, register_set *sfu_out,
                register_set *int_out, register_set *tensor_core_out,
                std::vector<register_set *> &spec_cores_out,
                register_set *mem_out, int id)
      : scheduler_unit(stats, shader, scoreboard, simt, warp, sp_out, dp_out,
                       sfu_out, int_out, tensor_core_out, spec_cores_out,
                       mem_out, id) {}
  virtual ~rrr_scheduler() {}
  virtual void order_warps();
  virtual void done_adding_supervised_warps() {
    m_last_supervised_issued = m_supervised_warps.end();
  }
};

class gto_scheduler : public scheduler_unit {
 public:
  gto_scheduler(shader_core_stats *stats, shader_core_ctx *shader,
                Scoreboard *scoreboard, simt_stack **simt,
                std::vector<shd_warp_t *> *warp, register_set *sp_out,
                register_set *dp_out, register_set *sfu_out,
                register_set *int_out, register_set *tensor_core_out,
                std::vector<register_set *> &spec_cores_out,
                register_set *mem_out, int id)
      : scheduler_unit(stats, shader, scoreboard, simt, warp, sp_out, dp_out,
                       sfu_out, int_out, tensor_core_out, spec_cores_out,
                       mem_out, id) {}
  virtual ~gto_scheduler() {}
  virtual void order_warps();
  virtual void done_adding_supervised_warps() {
    m_last_supervised_issued = m_supervised_warps.begin();
  }
};

class oldest_scheduler : public scheduler_unit {
 public:
  oldest_scheduler(shader_core_stats *stats, shader_core_ctx *shader,
                   Scoreboard *scoreboard, simt_stack **simt,
                   std::vector<shd_warp_t *> *warp, register_set *sp_out,
                   register_set *dp_out, register_set *sfu_out,
                   register_set *int_out, register_set *tensor_core_out,
                   std::vector<register_set *> &spec_cores_out,
                   register_set *mem_out, int id)
      : scheduler_unit(stats, shader, scoreboard, simt, warp, sp_out, dp_out,
                       sfu_out, int_out, tensor_core_out, spec_cores_out,
                       mem_out, id) {}
  virtual ~oldest_scheduler() {}
  virtual void order_warps();
  virtual void done_adding_supervised_warps() {
    m_last_supervised_issued = m_supervised_warps.begin();
  }
};

class two_level_active_scheduler : public scheduler_unit {
 public:
  two_level_active_scheduler(shader_core_stats *stats, shader_core_ctx *shader,
                             Scoreboard *scoreboard, simt_stack **simt,
                             std::vector<shd_warp_t *> *warp,
                             register_set *sp_out, register_set *dp_out,
                             register_set *sfu_out, register_set *int_out,
                             register_set *tensor_core_out,
                             std::vector<register_set *> &spec_cores_out,
                             register_set *mem_out, int id, char *config_str)
      : scheduler_unit(stats, shader, scoreboard, simt, warp, sp_out, dp_out,
                       sfu_out, int_out, tensor_core_out, spec_cores_out,
                       mem_out, id),
        m_pending_warps() {
    unsigned inner_level_readin;
    unsigned outer_level_readin;
    int ret =
        sscanf(config_str, "two_level_active:%d:%d:%d", &m_max_active_warps,
               &inner_level_readin, &outer_level_readin);
    assert(3 == ret);
    m_inner_level_prioritization =
        (scheduler_prioritization_type)inner_level_readin;
    m_outer_level_prioritization =
        (scheduler_prioritization_type)outer_level_readin;
  }
  virtual ~two_level_active_scheduler() {}
  virtual void order_warps();
  void add_supervised_warp_id(int i) {
    if (m_next_cycle_prioritized_warps.size() < m_max_active_warps) {
      m_next_cycle_prioritized_warps.push_back(&warp(i));
    } else {
      m_pending_warps.push_back(&warp(i));
    }
  }
  virtual void done_adding_supervised_warps() {
    m_last_supervised_issued = m_supervised_warps.begin();
  }

 protected:
  virtual void do_on_warp_issued(
      unsigned warp_id, unsigned num_issued,
      const std::vector<shd_warp_t *>::const_iterator &prioritized_iter);

 private:
  std::deque<shd_warp_t *> m_pending_warps;
  scheduler_prioritization_type m_inner_level_prioritization;
  scheduler_prioritization_type m_outer_level_prioritization;
  unsigned m_max_active_warps;
};

// Static Warp Limiting Scheduler
class swl_scheduler : public scheduler_unit {
 public:
  swl_scheduler(shader_core_stats *stats, shader_core_ctx *shader,
                Scoreboard *scoreboard, simt_stack **simt,
                std::vector<shd_warp_t *> *warp, register_set *sp_out,
                register_set *dp_out, register_set *sfu_out,
                register_set *int_out, register_set *tensor_core_out,
                std::vector<register_set *> &spec_cores_out,
                register_set *mem_out, int id, char *config_string);
  virtual ~swl_scheduler() {}
  virtual void order_warps();
  virtual void done_adding_supervised_warps() {
    m_last_supervised_issued = m_supervised_warps.begin();
  }

 protected:
  scheduler_prioritization_type m_prioritization;
  unsigned m_num_warps_to_limit;
};

/*
操作数收集器类。Operand Collector Based Register File Unit。每个SM里有一个单独的操作数收集器。
英伟达的多项专利描述了一种名为"操作数收集器"的结构。操作数收集器是一组缓冲器和仲裁逻辑，用于提供一
个实际上使用多bank单端口RAM而能够表现出多端口寄存器文件的外观。整个安排节省了能源和面积，这对提高
吞吐量很重要。AMD公司也使用bank式寄存器文件，但编译器负责确保这些文件的访问不会发生bank冲突。

在指令被解码后，收集器单元被分配来缓冲指令的源操作数。收集器单元不是用来通过寄存器重命名来消除名称
的依赖性，而是作为一种方法来安排寄存器操作数访问的时间，以便在一个周期内对一个bank的访问不超过一次。
在其组织中，四个收集器单元中的每一个都包含三个操作数条目。每个操作数条目有四个域：一个有效位、一个
寄存器标识符、一个就绪位和操作数数据。每个操作数数据字段可以容纳一个由32个四字节元素组成的128字节源
操作数（warp中每个标量线程有一个四字节值）。此外，收集器单元包含一个标识符，表明该指令属于哪个warp。
仲裁器包含一个每个bank的读请求队列，以保持访问请求，直到它们被批准。

当一个指令从解码阶段收到，并且有一个收集器单元可用时，它被分配给该指令，并且操作数、warp ID、寄存器
标识符和有效位被设置。此外，源操作数的读取请求在仲裁器中被排队。为了简化设计，被执行单元写回的数据总
是优先于读请求。仲裁器选择一组最多四个不冲突的访问来发送至寄存器文件。为了减少Crossbar和收集器单元的
面积，选择时每个收集器单元每周期只接收一个操作数。

当每个操作数从寄存器文件中读出并放入相应的收集器单元时，一个"就绪位"被设置。最后，当所有的操作数都准
备好了，指令就被发射到SIMD执行单元。

在GPGPU-Sim模型中，每个后端流水线（SP、SFU、MEM）都有一组专用的收集器单元，它们共享一个通用收集器单
元池。每个流水线可用的单元数量和一般单元池的容量是可配置的。

该单元包括：
  1. 端口（m_in_Ports）：包含输入流水线寄存器集（ID_OC）和输出寄存器集（OC_EX）。ID_OC端口中的
     warp_inst_t将被发布到收集器单元。此外，当收集器单元获得所有所需的源寄存器时，它将由调度单元
     调度到输出管道寄存器集（OC_EX）。
  2. 收集器单元（m_cu）：每个收集器单元一次可以容纳一条指令。它将向仲裁员发送对源寄存器的请求。一
     旦所有源寄存器都准备好了，调度单元就可以将其调度到输出流水线寄存器集（OC_EX）。
  3. 仲裁器（m_arbiter）：仲裁器从收集器单元接收对源操作数的请求，然后放入请求队列。仲裁器将在每个
     周期向寄存器文件发出无Bank冲突请求。值得注意的是，仲裁器还用于处理对寄存器堆的写回，并且写回具
     有比读取更高的优先级。
  4. 调度单元（m_Dispatch_units）：一旦收集器单元准备就绪，调度单元将把收集器单元中的warp_inst_t
     调度到OC_EX寄存器集。

操作数收集器被建模为主流水线中的一个阶段，由函数shader_core_ctx::cycle()执行。关于操作数收集器的接
口，请参考#ALU流水线的更多细节。

opndcoll_rfu_t类是基于操作数收集器的寄存器文件单元的模型。它包含了对收集器单元集、仲裁器和调度单元
进行抽象的类。

opndcoll_rfu_t::allocate_cu(...)负责将warp_inst_t分配给其指定的操作数收集器组中的空闲操作数收集
器单元。同时，它在仲裁器的相应Bank队中为所有的源操作数增加一个读取请求。

然而，opndcoll_rfu_t::allocate_reads(...)处理没有冲突的读请求，换句话说，在不同寄存器Bank中的读
请求和不去同一个操作数收集器的读请求会从仲裁器队列中弹出。这说明写请求的优先级高于读请求。

函数opndcoll_rfu_t::dispatch_ready_cu()将准备好的操作数收集器的操作数寄存器（所有操作数都已收集）
分配到执行阶段。

函数opndcoll_rfu_t::writeback(const warp_inst_t &inst)在内存流水线的写回阶段被调用。它负责写的
分配。

在前面的warp调度器代码里单个Sahder Core内的warp调度器的个数由gpgpu_num_sched_per_core配置参数决
定，Volta架构每核心有4个warp调度器。每个调度器的创建代码：
     schedulers.push_back(new lrr_scheduler(
             m_stats, this, m_scoreboard, m_simt_stack, &m_warp,
             &m_pipeline_reg[ID_OC_SP], &m_pipeline_reg[ID_OC_DP],
             &m_pipeline_reg[ID_OC_SFU], &m_pipeline_reg[ID_OC_INT],
             &m_pipeline_reg[ID_OC_TENSOR_CORE], m_specilized_dispatch_reg,
             &m_pipeline_reg[ID_OC_MEM], i));
在发射过程中，warp调度器将可发射的指令按照其指令类型分发给不同的单元，这些单元包括SP/DP/SFU/INT/
TENSOR_CORE/MEM，在发射过程完成后，需要针对指令通过操作数收集器将指令所需的操作数全部收集齐。对于一
个SM，对应于一个操作数收集器，调度器的发射过程将指令放入：
    m_pipeline_reg[ID_OC_SP]、m_pipeline_reg[ID_OC_DP]、m_pipeline_reg[ID_OC_SFU]、
    m_pipeline_reg[ID_OC_INT]、m_pipeline_reg[ID_OC_TENSOR_CORE]、
    m_pipeline_reg[ID_OC_MEM]
等寄存器集合中，用以操作数收集器来收集操作数。
*/
class opndcoll_rfu_t {  // operand collector based register file unit
 public:
  // constructors
  opndcoll_rfu_t() {
    //寄存器文件的bank数。详见操作数收集器示意图。
    m_num_banks = 0;
    //该操作数收集器隶属于哪个SM。
    m_shader = NULL;
    //该操作数收集器的初始化状态。
    m_initialized = false;
  }
  //增加collector unit的数量。这里的cu_set定义为：
  //    enum { SP_CUS, DP_CUS, SFU_CUS, TENSOR_CORE_CUS, INT_CUS, MEM_CUS, GEN_CUS };
  //这里是collector unit有多个，对应于SP单元一个，对应于DP单元一个，......。
  void add_cu_set(unsigned cu_set, unsigned num_cu, unsigned num_dispatch);
  
  //port_vector_t的类型定义为存储寄存器集合register_set的向量：
  //    typedef std::vector<register_set *> port_vector_t;
  typedef std::vector<register_set *> port_vector_t;
  
  //uint_vector_t的类型定义为存储收集器单元set_id的向量：
  //    typedef std::vector<unsigned int> uint_vector_t;
  typedef std::vector<unsigned int> uint_vector_t;
  
  //add_port是将发射阶段的几个流水线寄存器集合ID_OC_SP等，以及后续操作数收集器发出的寄存器集合
  //OC_EX_SP等，对应于其所属的收集器单元set_id，添加进操作数收集器类。
  void add_port(port_vector_t &input, port_vector_t &ouput,
                uint_vector_t cu_sets);
  void init(unsigned num_banks, shader_core_ctx *shader);

  // modifiers
  bool writeback(warp_inst_t &warp);

  //操作数收集器向前执行一步。
  void step() {
    //遍历所有调度单元。每个单元找到一个准备好的收集器单元并进行调度。如果能够分别从各个调度器
    //找到一个空闲准备好可以接收的收集器单元的话，就执行它的分发函数dispatch()。该函数执行的主
    //要过程是，经过收集器单元收集完源操作数后，将原先暂存在收集器单元指令槽m_warp中的指令推出
    //到m_output_register中。
    dispatch_ready_cu();
    //仲裁器检查请求，并返回不同寄存器Bank中的op_t列表，并且这些寄存器Bank不处于Write状态。在
    //该函数中，仲裁器检查请求并返回op_t的列表，这些op_t位于不同的寄存器Bank中，并且这些寄存器
    //Bank不处于Write状态。
    allocate_reads();
    //端口（m_in_Ports）：包含输入流水线寄存器集合（ID_OC）和输出寄存器集合（OC_EX）。ID_OC端
    //口中的warp_inst_t将被发布到收集器单元。此外，当收集器单元获得所有所需的源寄存器时，它将由
    //调度单元调度到输出管道寄存器集（OC_EX）。m_in_ports中有多个input_port_t对象，每个对象分
    //别对应于SP/DP/SFU/INT/MEM/TC单元（但是一个单元可能会有多个input_port_t对象，不是一一对
    //应的），例如添加SP单元的input_port_t对象时：
    //   for (unsigned i = 0; i < m_config->gpgpu_operand_collector_num_in_ports_sp;
    //     i++) {
    //     in_ports.push_back(&m_pipeline_reg[ID_OC_SP]);
    //     out_ports.push_back(&m_pipeline_reg[OC_EX_SP]);
    //     cu_sets.push_back((unsigned)SP_CUS);
    //     cu_sets.push_back((unsigned)GEN_CUS);
    //     m_operand_collector.add_port(in_ports, out_ports, cu_sets);
    //     in_ports.clear(), out_ports.clear(), cu_sets.clear();
    //   }
    //   void opndcoll_rfu_t::add_port(port_vector_t &input, port_vector_t &output,
    //                                 uint_vector_t cu_sets) {
    //     m_in_ports.push_back(input_port_t(input, output, cu_sets));
    //   }
    //因此，m_in_ports对象：
    // 0-7 -> {{m_pipeline_reg[ID_OC_SP], m_pipeline_reg[ID_OC_SFU], m_pipeline_reg[ID_OC_MEM],
    //          m_pipeline_reg[ID_OC_TENSOR_CORE], m_pipeline_reg[ID_OC_DP], m_pipeline_reg[ID_OC_INT],
    //          m_config->m_specialized_unit[0].ID_OC_SPEC_ID, m_config->m_specialized_unit[1].ID_OC_SPEC_ID, 
    //          m_config->m_specialized_unit[2].ID_OC_SPEC_ID, m_config->m_specialized_unit[3].ID_OC_SPEC_ID,
    //          m_config->m_specialized_unit[4].ID_OC_SPEC_ID, m_config->m_specialized_unit[5].ID_OC_SPEC_ID,
    //          m_config->m_specialized_unit[6].ID_OC_SPEC_ID, m_config->m_specialized_unit[7].ID_OC_SPEC_ID},
    //         {m_pipeline_reg[OC_EX_SP], m_pipeline_reg[OC_EX_SFU], m_pipeline_reg[OC_EX_MEM],
    //          m_pipeline_reg[OC_EX_TENSOR_CORE], m_pipeline_reg[OC_EX_DP], m_pipeline_reg[OC_EX_INT],
    //          m_config->m_specialized_unit[0].OC_EX_SPEC_ID, m_config->m_specialized_unit[1].OC_EX_SPEC_ID, 
    //          m_config->m_specialized_unit[2].OC_EX_SPEC_ID, m_config->m_specialized_unit[3].OC_EX_SPEC_ID,
    //          m_config->m_specialized_unit[4].OC_EX_SPEC_ID, m_config->m_specialized_unit[5].OC_EX_SPEC_ID,
    //          m_config->m_specialized_unit[6].OC_EX_SPEC_ID, m_config->m_specialized_unit[7].OC_EX_SPEC_ID},
    //         GEN_CUS}
    //   8 -> {m_pipeline_reg[ID_OC_SP], m_pipeline_reg[OC_EX_SP], {SP_CUS, GEN_CUS}}
    //   9 -> {m_pipeline_reg[ID_OC_SFU], m_pipeline_reg[OC_EX_SFU], {SFU_CUS, GEN_CUS}}
    //  10 -> {m_pipeline_reg[ID_OC_TENSOR_CORE], m_pipeline_reg[OC_EX_TENSOR_CORE]
    //  11 -> {m_pipeline_reg[ID_OC_MEM], m_pipeline_reg[OC_EX_MEM], {MEM_CUS, GEN_CUS}}
    //所以这里的m_in_ports[p]是第p个input_port_t对象。
    for (unsigned p = 0; p < m_in_ports.size(); p++) allocate_cu(p);
    //process_banks()会重置所有Bank的状态为NO_ALLOC，空闲状态。
    process_banks();
  }

  void dump(FILE *fp) const {
    fprintf(fp, "\n");
    fprintf(fp, "Operand Collector State:\n");
    for (unsigned n = 0; n < m_cu.size(); n++) {
      fprintf(fp, "   CU-%2u: ", n);
      m_cu[n]->dump(fp, m_shader);
    }
    m_arbiter.dump(fp);
  }

  //返回当前操作数收集器隶属于的SM。
  shader_core_ctx *shader_core() { return m_shader; }

 private:
  void process_banks() { m_arbiter.reset_alloction(); }

  //遍历所有调度单元。每个单元找到一个准备好的收集器单元并进行调度。如果能够分别从各个调度器
  //找到一个空闲准备好可以接收的收集器单元的话，就执行它的分发函数dispatch()。该函数执行的
  //主要过程是，经过收集器单元收集完源操作数后，将原先暂存在收集器单元指令槽m_warp中的指令推
  //出到m_output_register中。
  void dispatch_ready_cu();
  void allocate_cu(unsigned port);
  //仲裁器检查请求，并返回不同寄存器Bank中的op_t列表，并且这些寄存器Bank不处于Write状态。在
  //该函数中，仲裁器检查请求并返回op_t的列表，这些op_t位于不同的寄存器Bank中，并且这些寄存器
  //Bank不处于Write状态。
  void allocate_reads();

  // types

  class collector_unit_t;

  //保留源操作数的类。op_t用来存储一条指令的单个源操作数。如果需要保存所有源操作数，需要使用
  //op_t*向量。
  class op_t {
   public:
    //源操作数的有效状态。
    op_t() { m_valid = false; }
    //初始化当前操作数。重要的参数为：
    //    collector_unit_t *cu：对应于哪个收集器单元；
    //    unsigned op：源操作数在其指令所有的源操作数中的排序；
    //    unsigned reg：源操作数对应的寄存器编号。
    //register_bank函数就是用来计算regnum所在的bank数。
    op_t(collector_unit_t *cu, unsigned op, unsigned reg, unsigned num_banks,
         bool sub_core_model, unsigned banks_per_sched, unsigned sched_id) {
      m_valid = true;
      m_warp = NULL;
      m_cu = cu;
      m_operand = op;
      m_register = reg;
      m_shced_id = sched_id;
      m_bank = register_bank(reg, cu->get_warp_id(), num_banks, sub_core_model,
                             banks_per_sched, sched_id);
    }
    op_t(const warp_inst_t *warp, unsigned reg, unsigned num_banks,
         bool sub_core_model, unsigned banks_per_sched, unsigned sched_id) {
      m_valid = true;
      m_warp = warp;
      m_register = reg;
      m_cu = NULL;
      m_operand = -1;
      m_shced_id = sched_id;
      m_bank = register_bank(reg, warp->warp_id(), num_banks, sub_core_model,
                             banks_per_sched, sched_id);
    }

    // accessors
    bool valid() const { return m_valid; }
    unsigned get_reg() const {
      assert(m_valid);
      return m_register;
    }
    unsigned get_wid() const {
      if (m_warp)
        return m_warp->warp_id();
      else if (m_cu)
        return m_cu->get_warp_id();
      else
        abort();
    }
    unsigned get_sid() const { return m_shced_id; }
    unsigned get_active_count() const {
      if (m_warp)
        return m_warp->active_count();
      else if (m_cu)
        return m_cu->get_active_count();
      else
        abort();
    }
    const active_mask_t &get_active_mask() {
      if (m_warp)
        return m_warp->get_active_mask();
      else if (m_cu)
        return m_cu->get_active_mask();
      else
        abort();
    }
    unsigned get_sp_op() const {
      if (m_warp)
        return m_warp->sp_op;
      else if (m_cu)
        return m_cu->get_sp_op();
      else
        abort();
    }
    //返回当前操作数所属的收集器单元的ID。
    unsigned get_oc_id() const { return m_cu->get_id(); }
    //返回当前操作数所属的Bank。
    unsigned get_bank() const { return m_bank; }
    //返回当前操作数在其指令所有的源操作数中的排序。
    unsigned get_operand() const { return m_operand; }
    void dump(FILE *fp) const {
      if (m_cu)
        fprintf(fp, " <R%u, CU:%u, w:%02u> ", m_register, m_cu->get_id(),
                m_cu->get_warp_id());
      else if (!m_warp->empty())
        fprintf(fp, " <R%u, wid:%02u> ", m_register, m_warp->warp_id());
    }
    //返回当前操作数的寄存器字符串。
    std::string get_reg_string() const {
      char buffer[64];
      snprintf(buffer, 64, "R%u", m_register);
      return std::string(buffer);
    }

    // modifiers
    //重置当前操作数的状态为无效。
    void reset() { m_valid = false; }

   private:
    //当前操作数是否有效。
    bool m_valid;
    //当前操作数所属的收集器单元。
    collector_unit_t *m_cu;
    //当前操作数所属的指令。
    const warp_inst_t *m_warp;
    //当前操作数在其指令所有的源操作数中的排序。
    unsigned m_operand;  // operand offset in instruction. e.g., add r1,r2,r3;
                         // r2 is oprd 0, r3 is 1 (r1 is dst)
    //当前操作数对应的寄存器编号。
    unsigned m_register;
    //当前操作数隶属于哪个Bank。
    unsigned m_bank;
    //当前操作数所属指令是哪个调度器发射的。
    unsigned m_shced_id;  // scheduler id that has issued this inst
  };

  enum alloc_t {
    NO_ALLOC,
    READ_ALLOC,
    WRITE_ALLOC,
  };

  //跟踪一个Bank的状态，一个allocation_t对象是一个Bank的状态。
  class allocation_t {
   public:
    //初始化时，设置Bank状态为NO_ALLOC，空闲状态。
    allocation_t() { m_allocation = NO_ALLOC; }
    //返回当前Bank是否是读状态，读状态时，m_allocation为READ_ALLOC。
    bool is_read() const { return m_allocation == READ_ALLOC; }
    //返回当前Bank是否是写状态，写状态时，m_allocation为WRITE_ALLOC。
    bool is_write() const { return m_allocation == WRITE_ALLOC; }
    //返回当前Bank是否是空闲状态，空闲状态时，m_allocation为NO_ALLOC。
    bool is_free() const { return m_allocation == NO_ALLOC; }
    void dump(FILE *fp) const {
      if (m_allocation == NO_ALLOC) {
        fprintf(fp, "<free>");
      } else if (m_allocation == READ_ALLOC) {
        fprintf(fp, "rd: ");
        m_op.dump(fp);
      } else if (m_allocation == WRITE_ALLOC) {
        fprintf(fp, "wr: ");
        m_op.dump(fp);
      }
      fprintf(fp, "\n");
    }
    //当前Bank是空闲状态时，才可以将其分配为读状态，读的操作数为op。
    void alloc_read(const op_t &op) {
      assert(is_free());
      m_allocation = READ_ALLOC;
      m_op = op;
    }
    //当前Bank是空闲状态时，才可以将其分配为写状态，写的操作数为op。
    void alloc_write(const op_t &op) {
      assert(is_free());
      m_allocation = WRITE_ALLOC;
      m_op = op;
    }
    //重置当前Bank的状态为NO_ALLOC，空闲状态。
    void reset() { m_allocation = NO_ALLOC; }

   private:
    //m_allocation的定义为：enum alloc_t {NO_ALLOC, READ_ALLOC, WRITE_ALLOC,}; 它存储了当
    //前Bank的状态，或是空闲状态，或是读状态，或是写状态。
    enum alloc_t m_allocation;
    //读或写当前Bank的操作数。
    op_t m_op;
  };

  //仲裁器。仲裁器（m_arbiter）：仲裁器从收集器单元接收对源操作数的请求，然后放入请求队列。器将
  //在每个周期向寄存器文件发出无Bank冲突请求。值得注意的是，仲裁器还用于处理对寄存器堆的写回，并
  //且写回具有比读取更高的优先级。
  class arbiter_t {
   public:
    // constructors
    arbiter_t() {
      m_queue = NULL;
      m_allocated_bank = NULL;
      m_allocator_rr_head = NULL;
      _inmatch = NULL;
      _outmatch = NULL;
      _request = NULL;
      m_last_cu = 0;
    }
    void init(unsigned num_cu, unsigned num_banks) {
      assert(num_cu > 0);
      assert(num_banks > 0);
      //当前操作数收集器的收集器单元的数目。
      m_num_collectors = num_cu;
      //当前操作数收集器的寄存器文件Bank数。
      m_num_banks = num_banks;
      _inmatch = new int[m_num_banks];
      _outmatch = new int[m_num_collectors];
      _request = new int *[m_num_banks];
      //每次访问当前操作数收集器的寄存器文件的访问数是收集器单元的个数，即为每个收集器单元收集一
      //个操作数。
      for (unsigned i = 0; i < m_num_banks; i++)
        _request[i] = new int[m_num_collectors];
      //add_read_requests函数会从收集器单元获取所有的源操作数，并将它们放入m_queue[bank]队列。
      //add_read_requests函数的定义：
      //     void add_read_requests(collector_unit_t *cu) {
      //       //获取操作数单元的所有操作数，src[i]是第i个操作数。
      //       const op_t *src = cu->get_operands();
      //       for (unsigned i = 0; i < MAX_REG_OPERANDS * 2; i++) {
      //         //对所有操作数循环。
      //         const op_t &op = src[i];
      //         if (op.valid()) {
      //           //如果操作数有效，则获取它们的Bank编号，并将其放入m_queue[bank]队列。
      //           unsigned bank = op.get_bank();
      //           m_queue[bank].push_back(op);
      //         }
      //       }
      //     }
      //可以看出，m_queue是一个以bank来索引的操作数队列，m_queue[i]是第i个bank获取的操作数。
      m_queue = new std::list<op_t>[num_banks];
      //用于存储每个Bank的状态，包括NO_ALLOC, READ_ALLOC, WRITE_ALLOC。
      m_allocated_bank = new allocation_t[num_banks];
      //m_allocator_rr_head是一个以收集器单元的ID为索引的数组，m_allocator_rr_head[i]是第i
      //个cu下一个需要检查的Bank的Bank ID。cu # -> next bank to check for request (rr-arb)
      m_allocator_rr_head = new unsigned[num_cu];
      for (unsigned n = 0; n < num_cu; n++)
        m_allocator_rr_head[n] = n % num_banks;
      reset_alloction();
    }

    // accessors
    void dump(FILE *fp) const {
      fprintf(fp, "\n");
      fprintf(fp, "  Arbiter State:\n");
      fprintf(fp, "  requests:\n");
      for (unsigned b = 0; b < m_num_banks; b++) {
        fprintf(fp, "    bank %u : ", b);
        std::list<op_t>::const_iterator o = m_queue[b].begin();
        for (; o != m_queue[b].end(); o++) {
          o->dump(fp);
        }
        fprintf(fp, "\n");
      }
      fprintf(fp, "  grants:\n");
      for (unsigned b = 0; b < m_num_banks; b++) {
        fprintf(fp, "    bank %u : ", b);
        m_allocated_bank[b].dump(fp);
      }
      fprintf(fp, "\n");
    }

    // modifiers
    std::list<op_t> allocate_reads();

    //从收集器单元获取所有的源操作数，并将它们放入m_queue[bank]队列。
    void add_read_requests(collector_unit_t *cu) {
      //获取操作数单元的所有操作数，src[i]是第i个操作数。
      const op_t *src = cu->get_operands();
      for (unsigned i = 0; i < MAX_REG_OPERANDS * 2; i++) {
        //对所有操作数循环。
        const op_t &op = src[i];
        if (op.valid()) {
          //如果操作数有效，则获取它们的Bank编号，并将其放入m_queue[bank]队列。
          unsigned bank = op.get_bank();
          //m_queue是一个以bank来索引的操作数队列，m_queue[i]是第i个bank获取的操作数。
          m_queue[bank].push_back(op);
        }
      }
    }
    //m_allocated_bank是一组状态机，用于跟踪每个register bank的状态。它具有以下三个状态：
    //    READ_ALLOC、WRITE_ALLOC、NO_ALLOC
    //m_queue是一个FIFO队列，用于缓冲对register bank的所有读取请求。基本上，m_allocated_bank
    //和m_queue中的条目数等于SM核心中的寄存器组数（V100为8）。
    bool bank_idle(unsigned bank) const {
      return m_allocated_bank[bank].is_free();
    }
    //分配给第bank号Bank的写状态，写的操作数为op，设置m_allocated_bank。
    void allocate_bank_for_write(unsigned bank, const op_t &op) {
      assert(bank < m_num_banks);
      m_allocated_bank[bank].alloc_write(op);
    }
    //分配给第bank号Bank的读状态，读的操作数为op，设置m_allocated_bank。
    void allocate_for_read(unsigned bank, const op_t &op) {
      assert(bank < m_num_banks);
      m_allocated_bank[bank].alloc_read(op);
    }
    //重置所有Bank的状态为NO_ALLOC，空闲状态。
    void reset_alloction() {
      for (unsigned b = 0; b < m_num_banks; b++) m_allocated_bank[b].reset();
    }

   private:
    //当前操作数收集器的寄存器单元的Bank数目。
    unsigned m_num_banks;
    //当前操作数收集器的收集器单元的数目。
    unsigned m_num_collectors;

    //m_allocated_bank是一组状态机，用于跟踪每个register bank的状态。它具有以下三个状态：
    //    READ_ALLOC、WRITE_ALLOC、NO_ALLOC
    allocation_t *m_allocated_bank;  // bank # -> register that wins
    //m_queue是一个FIFO队列，用于缓冲对register bank的所有读取请求。m_queue是一个以bank
    //来索引的操作数队列，m_queue[i]是第i个bank获取的操作数。
    std::list<op_t> *m_queue;

    //m_allocator_rr_head是一个以收集器单元的ID为索引的数组，m_allocator_rr_head[i]是第i
    //个cu下一个需要检查的Bank的Bank ID。cu # -> next bank to check for request (rr-arb)
    unsigned *
        m_allocator_rr_head;  // cu # -> next bank to check for request (rr-arb)
    unsigned m_last_cu;       // first cu to check while arb-ing banks (rr)

    int *_inmatch;
    int *_outmatch;
    int **_request;
  };

  //输入端口类。input_port_t的定义：
  //port_vector_t的类型定义为存储寄存器集合register_set的向量：
  //    typedef std::vector<register_set *> port_vector_t;
  //uint_vector_t的类型定义为存储收集器单元set_id的向量：
  //    typedef std::vector<unsigned int> uint_vector_t;
  //后续add_port是将发射阶段的几个流水线寄存器集合ID_OC_SP等，以及后续操作数收集器发出的
  //寄存器集合OC_EX_SP等，对应于其所属的收集器单元set_id，添加进操作数收集器类，这三者的
  //组合便是input_port_t对象。
  class input_port_t {
   public:
    input_port_t(port_vector_t &input, port_vector_t &output,
                 uint_vector_t cu_sets)
        : m_in(input), m_out(output), m_cu_sets(cu_sets) {
      assert(input.size() == output.size());
      assert(not m_cu_sets.empty());
    }
    // private:
    port_vector_t m_in, m_out;
    uint_vector_t m_cu_sets;
  };

  //收集器单元类。
  class collector_unit_t {
   public:
    // constructors
    //构造函数。
    collector_unit_t() {
      m_free = true;
      m_warp = NULL;
      //经过收集器单元收集完源操作数后，将指令推出到m_output_register中。
      m_output_register = NULL;
      m_src_op = new op_t[MAX_REG_OPERANDS * 2];
      m_not_ready.reset();
      m_warp_id = -1;
      m_num_banks = 0;
    }
    // accessors
    //返回当前收集器单元是否所有源操作数都准备好了。
    bool ready() const;
    const op_t *get_operands() const { return m_src_op; }
    void dump(FILE *fp, const shader_core_ctx *shader) const;

    unsigned get_warp_id() const { return m_warp_id; }
    unsigned get_active_count() const { return m_warp->active_count(); }
    const active_mask_t &get_active_mask() const {
      return m_warp->get_active_mask();
    }
    unsigned get_sp_op() const { return m_warp->sp_op; }
    unsigned get_id() const { return m_cuid; }  // returns CU hw id
    unsigned get_reg_id() const { return m_reg_id; }

    // modifiers
    void init(unsigned n, unsigned num_banks, const core_config *config,
              opndcoll_rfu_t *rfu, bool m_sub_core_model, unsigned reg_id,
              unsigned num_banks_per_sched);
    bool allocate(register_set *pipeline_reg, register_set *output_reg);

    //m_not_ready的定义为：
    //    std::bitset<MAX_REG_OPERANDS * 2> m_not_ready;
    //m_not_ready是一个位向量，用来存储一条指令的所有源操作数是否处于非就绪状态。这里设置
    //第op个源操作数为就绪状态。
    void collect_operand(unsigned op) { m_not_ready.reset(op); }
    unsigned get_num_operands() const { return m_warp->get_num_operands(); }
    unsigned get_num_regs() const { return m_warp->get_num_regs(); }
    void dispatch();
    bool is_free() { return m_free; }

   private:
    bool m_free;
    unsigned m_cuid;  // collector unit hw id
    unsigned m_warp_id;
    //将一条指令分配给一条指令后，m_warp存储这条指令。
    warp_inst_t *m_warp;
    //经过收集器单元收集完源操作数后，将指令推出到m_output_register中。
    register_set
        *m_output_register;  // pipeline register to issue to when ready
    op_t *m_src_op;
    //m_not_ready的定义为：
    //    std::bitset<MAX_REG_OPERANDS * 2> m_not_ready;
    //m_not_ready是一个位向量，用来存储m_warp指令的所有源操作数是否处于非就绪状态。
    std::bitset<MAX_REG_OPERANDS * 2> m_not_ready;
    unsigned m_num_banks;
    opndcoll_rfu_t *m_rfu;

    unsigned m_num_banks_per_sched;
    bool m_sub_core_model;
    //这里reg_id其实是对应的调度器的ID。
    unsigned m_reg_id;  // if sub_core_model enabled, limit regs this cu can r/w
  };

  //一个输出端口对应一个调度器。
  class dispatch_unit_t {
   public:
    // for now each collector set gets dedicated dispatch units.
    //目前，每个收集器set都有专用的调度单元，由gpgpu_operand_collector_num_out_ports_sp
    //等确定。在V100配置中：
    //    gpgpu_operand_collector_num_out_ports_sp = 1
    //    gpgpu_operand_collector_num_out_ports_dp = 0
    //    gpgpu_operand_collector_num_out_ports_sfu = 1
    //    gpgpu_operand_collector_num_out_ports_int = 0
    //    gpgpu_operand_collector_num_out_ports_tensor_core = 1
    //    gpgpu_operand_collector_num_out_ports_mem = 1
    //    gpgpu_operand_collector_num_out_ports_gen = 8
    //这里调度单元的数目与输出端口的数目一致，即：
    //    对应于m_cus[SP_CUS         ]有1个调度器；
    //    对应于m_cus[DP_CUS         ]有0个调度器；
    //    对应于m_cus[SFU_CUS        ]有1个调度器；
    //    对应于m_cus[INT_CUS        ]有0个调度器；
    //    对应于m_cus[TENSOR_CORE_CUS]有1个调度器；
    //    对应于m_cus[MEM_CUS        ]有1个调度器；
    //    对应于m_cus[GEN_CUS        ]有8个调度器。
    //这里是调度器的初始化，调用时：
    //    for (unsigned i = 0; i < num_dispatch; i++)
    //      m_dispatch_units.push_back(dispatch_unit_t(&m_cus[set_id]));
    //传入的参数cus是m_cus[set_id]，对应于set_id的收集器单元：
    //    m_cus[SP_CUS         ]是一个vector，存储了SP         单元的4个收集器单元；
    //    m_cus[DP_CUS         ]是一个vector，存储了DP         单元的0个收集器单元；
    //    m_cus[SFU_CUS        ]是一个vector，存储了SFU        单元的4个收集器单元；
    //    m_cus[INT_CUS        ]是一个vector，存储了INT        单元的0个收集器单元；
    //    m_cus[TENSOR_CORE_CUS]是一个vector，存储了TENSOR_CORE单元的4个收集器单元；
    //    m_cus[MEM_CUS        ]是一个vector，存储了MEM        单元的2个收集器单元；
    //    m_cus[GEN_CUS        ]是一个vector，存储了GEN        单元的8个收集器单元。
    dispatch_unit_t(std::vector<collector_unit_t> *cus) {
      m_last_cu = 0;
      //对应于set_id的收集器单元向量。
      m_collector_units = cus;
      //对应于set_id的收集器单元的个数。
      m_num_collectors = (*cus).size();
      m_next_cu = 0;
    }

    //初始化。
    void init(bool sub_core_model, unsigned num_warp_scheds) {
      //sub_core_model模式。
      m_sub_core_model = sub_core_model;
      //warp调度器个数。
      m_num_warp_scheds = num_warp_scheds;
    }

    //找到一个空闲准备好可以接收的收集器单元。
    collector_unit_t *find_ready() {
      // With sub-core enabled round robin starts with the next cu assigned to a
      // different sub-core than the one that dispatched last

      //每个warp调度器可以分到的收集器单元的个数。例如，在创建m_cus[TENSOR_CORE_CUS]时，
      //m_cus[TENSOR_CORE_CUS]的大小即为m_num_collectors，m_cus[TENSOR_CORE_CUS]是一
      //个vector，存储了TENSOR_CORE单元的4个收集器单元；那么这里m_num_warp_scheds=4，则
      //cusPerSched=4/4=1。因此0号调度器可使用第0个收集器单元，1号调度器可使用第1个收集器
      //单元，2号调度器可使用第2个收集器单元，3号调度器可使用第3个收集器单元。
      unsigned cusPerSched = m_num_collectors / m_num_warp_scheds;
      //rr_increment是在保证下一个选定的cu与上一个选定的cu不同属于同一个warp调度器的范
      //围。例如，如果m_num_collectors=16，m_num_warp_scheds=4，m_last_cu=0，那么就有
      //cusPerSched=4，rr_increment=4，那下一个选定的cu就是4，5，6，7，8，9，10，...。
      //正好掠过了0，1，2，3，这4个cu，这4个cu属于同一个warp调度器。
      unsigned rr_increment =
          m_sub_core_model ? cusPerSched - (m_last_cu % cusPerSched) : 1;
      for (unsigned n = 0; n < m_num_collectors; n++) {
        unsigned c = (m_last_cu + n + rr_increment) % m_num_collectors;
        //如果收集器单元准备好了，那么就返回该收集器单元。注意这里，调度单元的数目与输入端
        //口的数目一致，即：
        //    对应于m_cus[SP_CUS         ]有1个调度器；
        //    对应于m_cus[DP_CUS         ]有0个调度器；
        //    对应于m_cus[SFU_CUS        ]有1个调度器；
        //    对应于m_cus[INT_CUS        ]有0个调度器；
        //    对应于m_cus[TENSOR_CORE_CUS]有1个调度器；
        //    对应于m_cus[MEM_CUS        ]有1个调度器；
        //    对应于m_cus[GEN_CUS        ]有8个调度器。
        //这里的m_collector_units是一个指针，指向m_cus[set_id]，对应于set_id的收集器单
        //元，且它只调度对应于set_id的收集器单元。(*m_collector_units)[c]是对应于set_id
        //的第c个收集器单元，如果该收集器单元准备好了，那么就返回该收集器单元。
        if ((*m_collector_units)[c].ready()) {
          m_last_cu = c;
          return &((*m_collector_units)[c]);
        }
      }
      return NULL;
    }

   private:
    //对应于set_id的收集器单元的个数。
    unsigned m_num_collectors;
    //对应于set_id的收集器单元向量。
    std::vector<collector_unit_t> *m_collector_units;
    //上一个调度的收集器单元。
    unsigned m_last_cu;  // dispatch ready cu's rr
    //没有用到这个变量。
    unsigned m_next_cu;  // for initialization
    //sub_core_model模式。
    bool m_sub_core_model;
    //一个SM内warp调度器的个数。
    unsigned m_num_warp_scheds;
  };

  // opndcoll_rfu_t data members
  //是否操作数收集器已经被初始化。
  bool m_initialized;
  //没用到这个变量。
  unsigned m_num_collector_sets;
  // unsigned m_num_collectors;
  //寄存器文件的bank数，在V100配置中，m_num_banks被初始化为16。
  unsigned m_num_banks;
  unsigned m_warp_size;
  //收集器单元列表。收集器单元（m_cu）：每个收集器单元一次可以容纳一条指令。它将向器发送对源寄存
  //器的请求。一旦所有源寄存器都准备好了，调度单元就可以将其调度到输出流水线寄存器集（OC_EX）。
  std::vector<collector_unit_t *> m_cu;
  //仲裁器。仲裁器（m_arbiter）：仲裁器从收集器单元接收对源操作数的请求，然后放入请求队列。器将
  //在每个周期向寄存器文件发出无Bank冲突请求。值得注意的是，仲裁器还用于处理对寄存器堆的写回，并
  //且写回具有比读取更高的优先级。
  // arbiter_t m_arbiter;                                // yangjianchao16 del
  arbiter_t m_arbiter;
  //每个warp调度器可用的bank。在sub_core_model模式中，每个warp调度器可用的bank数量是
  //有限的。在V100配置中，共有4个warp调度器，0号warp调度器可用的bank为0-3，1号warp调
  //度器可用的bank为4-7，2号warp调度器可用的bank为8-11，3号warp调度器可用的bank为12-
  //15：m_num_banks_per_sched = num_banks / shader->get_config()->gpgpu_num_sched_per_core;
  unsigned m_num_banks_per_sched;
  //每个SM内warp调度器的个数。
  unsigned m_num_warp_scheds;
  //sub_core_model模式。
  bool sub_core_model;

  // unsigned m_num_ports;
  // std::vector<warp_inst_t**> m_input;
  // std::vector<warp_inst_t**> m_output;
  // std::vector<unsigned> m_num_collector_units;
  // warp_inst_t **m_alu_port;

  //端口（m_in_Ports）：包含输入流水线寄存器集合（ID_OC）和输出寄存器集合（OC_EX）。ID_OC端口中的
  //warp_inst_t将被发布到收集器单元。此外，当收集器单元获得所有所需的源寄存器时，它将由调度单元调度
  //到输出管道寄存器集（OC_EX）。
  std::vector<input_port_t> m_in_ports;
  //id对应收集器单元的的字典。
  typedef std::map<unsigned /* collector set */,
                   std::vector<collector_unit_t> /*collector sets*/>
      cu_sets_t;
  //操作数收集器的集合。
  cu_sets_t m_cus;
  //调度单元。调度单元（m_Dispatch_units）：一旦收集器单元准备就绪，调度单元将把收集器单元中的warp
  //_inst_t调度到OC_EX寄存器集。
  std::vector<dispatch_unit_t> m_dispatch_units;

  // typedef std::map<warp_inst_t**/*port*/,dispatch_unit_t> port_to_du_t;
  // port_to_du_t                     m_dispatch_units;
  // std::map<warp_inst_t**,std::list<collector_unit_t*> > m_free_cu;
  shader_core_ctx *m_shader;
};

/*
barrier的集合。
*/
class barrier_set_t {
 public:
  barrier_set_t(shader_core_ctx *shader, unsigned max_warps_per_core,
                unsigned max_cta_per_core, unsigned max_barriers_per_cta,
                unsigned warp_size);

  // during cta allocation
  void allocate_barrier(unsigned cta_id, warp_set_t warps);

  // during cta deallocation
  void deallocate_barrier(unsigned cta_id);
  //Map<CTA ID，单个CTA内的所有warp数量大小的位图>。
  typedef std::map<unsigned, warp_set_t> cta_to_warp_t;
  //Map<barrier ID，单个CTA内的所有warp数量大小的位图>。
  typedef std::map<unsigned, warp_set_t>
      bar_id_to_warp_t; /*set of warps reached a specific barrier id*/

  // individual warp hits barrier
  //单个warp到达barrier。
  void warp_reaches_barrier(unsigned cta_id, unsigned warp_id,
                            warp_inst_t *inst);

  // warp reaches exit
  void warp_exit(unsigned warp_id);

  // assertions
  bool warp_waiting_at_barrier(unsigned warp_id) const;

  // debug
  void dump();

 private:
  unsigned m_max_cta_per_core;
  unsigned m_max_warps_per_core;
  unsigned m_max_barriers_per_cta;
  unsigned m_warp_size;
  //Map<CTA ID，单个CTA内的所有warp数量大小的位图>。
  cta_to_warp_t m_cta_to_warps;
  bar_id_to_warp_t m_bar_id_to_warps;
  warp_set_t m_warp_active;
  warp_set_t m_warp_at_barrier;
  shader_core_ctx *m_shader;
};

struct insn_latency_info {
  unsigned pc;
  unsigned long latency;
};

/*
指令获取缓冲区。指令获取缓冲区（ifetch_Buffer_t）对指令缓存（I-cache）和SM Core之间的接口进行建模。
它有一个成员m_valid，用于指示缓冲区是否有有效的指令。它还将指令的warp id记录在m_warp_id中。
*/
struct ifetch_buffer_t {
  ifetch_buffer_t() { m_valid = false; }

  ifetch_buffer_t(address_type pc, unsigned nbytes, unsigned warp_id) {
    m_valid = true;
    m_pc = pc;
    m_nbytes = nbytes;
    m_warp_id = warp_id;
  }

  bool m_valid;
  //获取的指令的PC值。
  address_type m_pc;
  unsigned m_nbytes;
  unsigned m_warp_id;
};

class shader_core_config;

/*
simd_function_unit对象实现了SP单元和SFU单元（ALU流水线）。
*/
class simd_function_unit {
 public:
  //构造函数。
  simd_function_unit(const shader_core_config *config);
  ~simd_function_unit() { delete m_dispatch_reg; }

  // modifiers
  //issue(warp_inst_t*&)成员函数将给定的流水线寄存器的内容移入m_dispatch_reg。
  virtual void issue(register_set &source_reg);
  virtual void cycle() = 0;
  //lane的意思为一个warp中有32个线程，而在流水线寄存器中可能暂存了很多条指令，这些指令的每对应的线程掩
  //码的每一位都是一个lane。即遍历流水线寄存器中的非空指令，返回所有指令的整体线程掩码（所有指令线程掩
  //码的或值）。
  virtual void active_lanes_in_pipeline() = 0;

  // accessors
  virtual unsigned clock_multiplier() const { return 1; }
  //判断一条指令能否发射，即判断m_dispatch_reg是否为空，其在occupied对应的标识位是否为空。
  virtual bool can_issue(const warp_inst_t &inst) const {
    return m_dispatch_reg->empty() && !occupied.test(inst.latency);
  }
  virtual bool is_issue_partitioned() = 0;
  //获取发射寄存器的ID。
  virtual unsigned get_issue_reg_id() = 0;
  virtual bool stallable() const = 0;
  //打印SIMD单元的dispatch寄存器。
  virtual void print(FILE *fp) const {
    fprintf(fp, "%s dispatch= ", m_name.c_str());
    m_dispatch_reg->print(fp);
  }
  //获取SIMD单元的名称。
  const char *get_name() { return m_name.c_str(); }

 protected:
  //SIMD单元的名称。
  std::string m_name;
  const shader_core_config *m_config;
  //SIMD单元的dispatch寄存器。
  warp_inst_t *m_dispatch_reg;
  //最长的ALU指令的延迟，即流水线寄存器至多有512个槽。
  static const unsigned MAX_ALU_LATENCY = 512;
  //流水线寄存器至多512个槽的位图，标识每个槽是否被占用。
  std::bitset<MAX_ALU_LATENCY> occupied;
};

/*
SP单元和SFU单元的时序模型主要在 shader.h 中定义的 pipelined_simd_unit 类中实现。模拟单元的具体类（
sp_unit类和sfu类）是从这个类派生出来的，由可重载的 can_issue() 成员函数来指定单元可执行的指令类型。

SP单元通过OC_EX_SP流水线寄存器连接到操作收集器单元；SFU单元通过OC_EX_SFU流水线寄存器连接到操作数收集
器单元。两个单元通过WB_EX流水线寄存器共享一个共同的写回阶段。为了防止两个单元因写回阶段的冲突而停滞，
每条进入任何一个单元的指令都必须在发出到目标单元之前在结果总线（m_result_bus）上分配一个槽（见shader
_core_ctx::execute()）。

手册[ALU流水线软件模型]中的图提供了一个概览，介绍了pipelined_simd_unit如何为不同类型的指令建立吞吐量
和延迟。

在每个pipelined_simd_unit中，issue(warp_inst_t*&)成员函数将给定的流水线寄存器的内容移入m_dispatch_
reg。然后指令在m_dispatch_reg等待initiation_interval个周期。在此期间，没有其他的指令可以发到这个单
元，所以这个等待是指令的吞吐量的模型。等待之后，指令被派发到内部流水线寄存器m_pipeline_reg进行延迟建
模。派发的位置是确定的，所以在m_dispatch_reg中花费的时间也被计入延迟中。每个周期，指令将通过流水线寄
存器前进，最终进入m_result_port，这是共享的流水线寄存器，通向SP和SFU单元的共同写回阶段。

各类指令的吞吐量和延迟在cuda-sim.cc的ptx_instruction::set_opcode_and_latency()中指定。这个函数在预
解码时被调用。
*/
class pipelined_simd_unit : public simd_function_unit {
 public:
  pipelined_simd_unit(register_set *result_port,
                      const shader_core_config *config, unsigned max_latency,
                      shader_core_ctx *core, unsigned issue_reg_id);

  // modifiers
  virtual void cycle();
  //issue(warp_inst_t*&)成员函数将给定的流水线寄存器的内容移入m_dispatch_reg。
  virtual void issue(register_set &source_reg);
  //lane的意思为一个warp中有32个线程，而在流水线寄存器中可能暂存了很多条指令，这些指令的每对应的线程掩
  //码的每一位都是一个lane。即遍历流水线寄存器中的非空指令，返回所有指令的整体线程掩码（所有指令线程掩
  //码的或值）。
  virtual unsigned get_active_lanes_in_pipeline();

  virtual void active_lanes_in_pipeline() = 0;
  /*
      virtual void issue( register_set& source_reg )
      {
          //move_warp(m_dispatch_reg,source_reg);
          //source_reg.move_out_to(m_dispatch_reg);
          simd_function_unit::issue(source_reg);
      }
  */
  // accessors
  virtual bool stallable() const { return false; }
  //判断一条指令能否发射，即判断m_dispatch_reg是否为空，其在occupied对应的标识位是否为空。
  virtual bool can_issue(const warp_inst_t &inst) const {
    return simd_function_unit::can_issue(inst);
  }
  virtual bool is_issue_partitioned() = 0;
  //获取发射寄存器的ID。
  unsigned get_issue_reg_id() { return m_issue_reg_id; }
  virtual void print(FILE *fp) const {
    simd_function_unit::print(fp);
    for (int s = m_pipeline_depth - 1; s >= 0; s--) {
      if (!m_pipeline_reg[s]->empty()) {
        fprintf(fp, "      %s[%2d] ", m_name.c_str(), s);
        m_pipeline_reg[s]->print(fp);
      }
    }
  }

 protected:
  //流水线的深度。
  unsigned m_pipeline_depth;
  //流水线寄存器。
  warp_inst_t **m_pipeline_reg;
  //结果端口。
  register_set *m_result_port;
  class shader_core_ctx *m_core;
  //发射寄存器的ID。
  unsigned m_issue_reg_id;  // if sub_core_model is enabled we can only issue
                            // from a subset of operand collectors

  unsigned active_insts_in_pipeline;
};

/*
特殊功能单元的定义。
*/
class sfu : public pipelined_simd_unit {
 public:
  //SFU特殊功能单元的构造函数。仅m_name不同。
  sfu(register_set *result_port, const shader_core_config *config,
      shader_core_ctx *core, unsigned issue_reg_id);
  //仅操作码为SFU_OP/ALU_SFU_OP以及对计算能力小于29的DP_OP(compute <= 29)才会发射到SFU。
  virtual bool can_issue(const warp_inst_t &inst) const {
    switch (inst.op) {
      case SFU_OP:
        break;
      case ALU_SFU_OP:
        break;
      case DP_OP:
        break;  // for compute <= 29 (i..e Fermi and GT200)
      default:
        return false;
    }
    return pipelined_simd_unit::can_issue(inst);
  }
  virtual void active_lanes_in_pipeline();
  virtual void issue(register_set &source_reg);
  bool is_issue_partitioned() { return true; }
};

/*
DP单元的定义。
*/
class dp_unit : public pipelined_simd_unit {
 public:
  dp_unit(register_set *result_port, const shader_core_config *config,
          shader_core_ctx *core, unsigned issue_reg_id);
  //仅操作码为DP_OP才会发射到SFU。
  virtual bool can_issue(const warp_inst_t &inst) const {
    switch (inst.op) {
      case DP_OP:
        break;
      default:
        return false;
    }
    return pipelined_simd_unit::can_issue(inst);
  }
  virtual void active_lanes_in_pipeline();
  virtual void issue(register_set &source_reg);
  bool is_issue_partitioned() { return true; }
};

class tensor_core : public pipelined_simd_unit {
 public:
  tensor_core(register_set *result_port, const shader_core_config *config,
              shader_core_ctx *core, unsigned issue_reg_id);
  virtual bool can_issue(const warp_inst_t &inst) const {
    switch (inst.op) {
      case TENSOR_CORE_OP:
        break;
      default:
        return false;
    }
    return pipelined_simd_unit::can_issue(inst);
  }
  virtual void active_lanes_in_pipeline();
  virtual void issue(register_set &source_reg);
  bool is_issue_partitioned() { return true; }
};

class int_unit : public pipelined_simd_unit {
 public:
  int_unit(register_set *result_port, const shader_core_config *config,
           shader_core_ctx *core, unsigned issue_reg_id);
  virtual bool can_issue(const warp_inst_t &inst) const {
    switch (inst.op) {
      case SFU_OP:
        return false;
      case LOAD_OP:
        return false;
      case TENSOR_CORE_LOAD_OP:
        return false;
      case STORE_OP:
        return false;
      case TENSOR_CORE_STORE_OP:
        return false;
      case MEMORY_BARRIER_OP:
        return false;
      case SP_OP:
        return false;
      case DP_OP:
        return false;
      default:
        break;
    }
    return pipelined_simd_unit::can_issue(inst);
  }
  virtual void active_lanes_in_pipeline();
  virtual void issue(register_set &source_reg);
  bool is_issue_partitioned() { return true; }
};

class sp_unit : public pipelined_simd_unit {
 public:
  sp_unit(register_set *result_port, const shader_core_config *config,
          shader_core_ctx *core, unsigned issue_reg_id);
  virtual bool can_issue(const warp_inst_t &inst) const {
    switch (inst.op) {
      case SFU_OP:
        return false;
      case LOAD_OP:
        return false;
      case TENSOR_CORE_LOAD_OP:
        return false;
      case STORE_OP:
        return false;
      case TENSOR_CORE_STORE_OP:
        return false;
      case MEMORY_BARRIER_OP:
        return false;
      case DP_OP:
        return false;
      default:
        break;
    }
    return pipelined_simd_unit::can_issue(inst);
  }
  virtual void active_lanes_in_pipeline();
  virtual void issue(register_set &source_reg);
  bool is_issue_partitioned() { return true; }
};

class specialized_unit : public pipelined_simd_unit {
 public:
  specialized_unit(register_set *result_port, const shader_core_config *config,
                   shader_core_ctx *core, int supported_op, char *unit_name,
                   unsigned latency, unsigned issue_reg_id);
  virtual bool can_issue(const warp_inst_t &inst) const {
    if (inst.op != m_supported_op) {
      return false;
    }
    return pipelined_simd_unit::can_issue(inst);
  }
  virtual void active_lanes_in_pipeline();
  virtual void issue(register_set &source_reg);
  bool is_issue_partitioned() { return true; }

 private:
  int m_supported_op;
};

class simt_core_cluster;
class shader_memory_interface;
class shader_core_mem_fetch_allocator;
class cache_t;

/*
LDST单元类。ldst_unit类实现了Shader流水线的内存阶段，实例化并操作所有的Shader内存：纹理（m_L1T）、
常量（m_L1C）和数据（m_L1D）。ldst_unit::cycle()实现了该单元操作的拍数向前推进，并在每周期被泵入
m_config->mem_warp_parts次数。

ldst_unit::cycle()处理来自互连网络的内存响应（存储在m_response_fifo中），填充缓存并标记存储为完成。
该函数还使得缓存拍数向前推进，以便它们可以向互连网络发送它们Miss的数据的请求。对每种类型的L1存储的缓
存访问分别在shared_cycle()、constant_cycle()、texture_cycle()和memory_cycle()中完成。 

memory_cycle用于访问L1 data cache。这些函数中的每一个都会调用process_memory_access_queue()，这是
一个通用函数，从指令的内部访问队列中抽取一个访问，并将这个请求发送到缓存中。如果这个访问在这个周期内不
能被处理（也就是说，它既没有错过也没有命中缓存，这可能发生在各种系统队列已经满了的情况下，或者是所有
lines in a particular way都被reserved，还没有被filled），那么这个访问将在下一个周期再次尝试。

值得注意的是，并不是所有的指令都能到达该单元的写回阶段。所有的存储指令和加载指令在所有请求的缓存块被命
中的情况下都会在cycle()函数中退出流水线。这是因为它们不需要等待互连网络的响应，可以绕过写回逻辑，将指
令所请求的cache lines和已经返回的cache lines记录下来。
*/
class ldst_unit : public pipelined_simd_unit {
 public:
  ldst_unit(mem_fetch_interface *icnt,
            shader_core_mem_fetch_allocator *mf_allocator,
            shader_core_ctx *core, opndcoll_rfu_t *operand_collector,
            Scoreboard *scoreboard, const shader_core_config *config,
            const memory_config *mem_config, class shader_core_stats *stats,
            unsigned sid, unsigned tpc, gpgpu_sim *gpu);

  // Add a structure to record the LDGSTS instructions,
  // similar to m_pending_writes, but since LDGSTS does not have a output
  // register to write to, so a new structure needs to be added
  /* A multi-level map: unsigned (warp_id) -> unsigned (pc) -> unsigned (addr)
   * -> unsigned (count)
   */
  std::map<unsigned /*warp_id*/,
           std::map<unsigned /*pc*/,
                    std::map<unsigned /*addr*/, unsigned /*count*/>>>
      m_pending_ldgsts;
  // modifiers
  //LDST单元issue函数。
  virtual void issue(register_set &inst);
  bool is_issue_partitioned() { return false; }
  virtual void cycle();

  void fill(mem_fetch *mf);
  void flush();
  void invalidate();
  void writeback();

  // accessors
  //时钟倍增器：一些单元可能在更高的循环速率下运行。
  virtual unsigned clock_multiplier() const;

  virtual bool can_issue(const warp_inst_t &inst) const {
    switch (inst.op) {
      case LOAD_OP:
        break;
      case TENSOR_CORE_LOAD_OP:
        break;
      case STORE_OP:
        break;
      case TENSOR_CORE_STORE_OP:
        break;
      case MEMORY_BARRIER_OP:
        break;
      default:
        return false;
    }
    return m_dispatch_reg->empty();
  }

  virtual void active_lanes_in_pipeline();
  virtual bool stallable() const { return true; }
  bool response_buffer_full() const;
  void print(FILE *fout) const;
  void print_cache_stats(FILE *fp, unsigned &dl1_accesses,
                         unsigned &dl1_misses);
  void get_cache_stats(unsigned &read_accesses, unsigned &write_accesses,
                       unsigned &read_misses, unsigned &write_misses,
                       unsigned cache_type);
  void get_cache_stats(cache_stats &cs);

  void get_L1D_sub_stats(struct cache_sub_stats &css) const;
  void get_L1C_sub_stats(struct cache_sub_stats &css) const;
  void get_L1T_sub_stats(struct cache_sub_stats &css) const;

 protected:
  ldst_unit(mem_fetch_interface *icnt,
            shader_core_mem_fetch_allocator *mf_allocator,
            shader_core_ctx *core, opndcoll_rfu_t *operand_collector,
            Scoreboard *scoreboard, const shader_core_config *config,
            const memory_config *mem_config, shader_core_stats *stats,
            unsigned sid, unsigned tpc, l1_cache *new_l1d_cache);
  void init(mem_fetch_interface *icnt,
            shader_core_mem_fetch_allocator *mf_allocator,
            shader_core_ctx *core, opndcoll_rfu_t *operand_collector,
            Scoreboard *scoreboard, const shader_core_config *config,
            const memory_config *mem_config, shader_core_stats *stats,
            unsigned sid, unsigned tpc);

 protected:
  bool shared_cycle(warp_inst_t &inst, mem_stage_stall_type &rc_fail,
                    mem_stage_access_type &fail_type);
  bool constant_cycle(warp_inst_t &inst, mem_stage_stall_type &rc_fail,
                      mem_stage_access_type &fail_type);
  bool texture_cycle(warp_inst_t &inst, mem_stage_stall_type &rc_fail,
                     mem_stage_access_type &fail_type);
  bool memory_cycle(warp_inst_t &inst, mem_stage_stall_type &rc_fail,
                    mem_stage_access_type &fail_type);

  virtual mem_stage_stall_type process_cache_access(
      cache_t *cache, new_addr_type address, warp_inst_t &inst,
      std::list<cache_event> &events, mem_fetch *mf,
      enum cache_request_status status);
  mem_stage_stall_type process_memory_access_queue(cache_t *cache,
                                                   warp_inst_t &inst);
  mem_stage_stall_type process_memory_access_queue_l1cache(l1_cache *cache,
                                                           warp_inst_t &inst);
  gpgpu_sim *m_gpu;

  const memory_config *m_memory_config;
  class mem_fetch_interface *m_icnt;
  shader_core_mem_fetch_allocator *m_mf_allocator;
  class shader_core_ctx *m_core;
  unsigned m_sid;
  unsigned m_tpc;

  tex_cache *m_L1T;        // texture cache
  read_only_cache *m_L1C;  // constant cache
  l1_cache *m_L1D;         // data cache
  std::map<unsigned /*warp_id*/,
           std::map<unsigned /*regnum*/, unsigned /*count*/>>
      m_pending_writes;
  std::list<mem_fetch *> m_response_fifo;
  opndcoll_rfu_t *m_operand_collector;
  Scoreboard *m_scoreboard;

  mem_fetch *m_next_global;
  //下一条需要写回的指令。
  warp_inst_t m_next_wb;
  unsigned m_writeback_arb;  // round-robin arbiter for writeback contention
                             // between L1T, L1C, shared
  unsigned m_num_writeback_clients;

  enum mem_stage_stall_type m_mem_rc;

  shader_core_stats *m_stats;

  // for debugging
  unsigned long long m_last_inst_gpu_sim_cycle;
  unsigned long long m_last_inst_gpu_tot_sim_cycle;

  std::vector<std::deque<mem_fetch *>> l1_latency_queue;
  void L1_latency_queue_cycle();
};

/*
流水线阶段名。N_PIPELINE_STAGES正好是阶段的总个数。
*/
enum pipeline_stage_name_t {
  ID_OC_SP = 0,
  ID_OC_DP,
  ID_OC_INT,
  ID_OC_SFU,
  ID_OC_MEM,
  OC_EX_SP,
  OC_EX_DP,
  OC_EX_INT,
  OC_EX_SFU,
  OC_EX_MEM,
  EX_WB,
  ID_OC_TENSOR_CORE,
  OC_EX_TENSOR_CORE,
  N_PIPELINE_STAGES
};

/*
流水线阶段名。N_PIPELINE_STAGES正好是阶段的总个数。
*/
const char *const pipeline_stage_name_decode[] = {
    "ID_OC_SP",          "ID_OC_DP",         "ID_OC_INT", "ID_OC_SFU",
    "ID_OC_MEM",         "OC_EX_SP",         "OC_EX_DP",  "OC_EX_INT",
    "OC_EX_SFU",         "OC_EX_MEM",        "EX_WB",     "ID_OC_TENSOR_CORE",
    "OC_EX_TENSOR_CORE", "N_PIPELINE_STAGES"};

/*
除SP/DP/INT/TC/MEM/SFU等单元外的奇特具体工作单元的信息在这里指定。其总数量由SPECIALIZED_UNIT_NUM
指定。
*/
struct specialized_unit_params {
  unsigned latency;
  unsigned num_units;
  unsigned id_oc_spec_reg_width;
  unsigned oc_ex_spec_reg_width;
  char name[20];
  unsigned ID_OC_SPEC_ID;
  unsigned OC_EX_SPEC_ID;
};

/*
Shader Core的配置类。
*/
class shader_core_config : public core_config {
 public:
  shader_core_config(gpgpu_context *ctx) : core_config(ctx) {
    pipeline_widths_string = NULL;
    gpgpu_ctx = ctx;
  }

  void init() {
    int ntok = sscanf(gpgpu_shader_core_pipeline_opt, "%d:%d",
                      &n_thread_per_shader, &warp_size);
    if (ntok != 2) {
      printf(
          "GPGPU-Sim uArch: error while parsing configuration string "
          "gpgpu_shader_core_pipeline_opt\n");
      abort();
    }

    char *toks = new char[100];
    char *tokd = toks;
    strcpy(toks, pipeline_widths_string);

    toks = strtok(toks, ",");

    /*	Removing the tensorcore pipeline while reading the config files if the
       tensor core is not available. If we won't remove it, old regression will
       be broken. So to support the legacy config files it's best to handle in
       this way.
     */
    int num_config_to_read = N_PIPELINE_STAGES - 2 * (!gpgpu_tensor_core_avail);

    for (int i = 0; i < num_config_to_read; i++) {
      assert(toks);
      //这里读取的是以下流水线阶段的宽度，该配置在-gpgpu_pipeline_widths中设置：
      // const char *const pipeline_stage_name_decode[] = {
      //   "ID_OC_SP",          "ID_OC_DP",         "ID_OC_INT", "ID_OC_SFU",
      //   "ID_OC_MEM",         "OC_EX_SP",         "OC_EX_DP",  "OC_EX_INT",
      //   "OC_EX_SFU",         "OC_EX_MEM",        "EX_WB",     "ID_OC_TENSOR_CORE",
      //   "OC_EX_TENSOR_CORE", "N_PIPELINE_STAGES"};
      // option_parser_register(
      //   opp, "-gpgpu_pipeline_widths", OPT_CSTR, &pipeline_widths_string,
      //   "Pipeline widths "
      //   "ID_OC_SP,ID_OC_DP,ID_OC_INT,ID_OC_SFU,ID_OC_MEM,OC_EX_SP,OC_EX_DP,OC_EX_"
      //   "INT,OC_EX_SFU,OC_EX_MEM,EX_WB,ID_OC_TENSOR_CORE,OC_EX_TENSOR_CORE",
      //   "1,1,1,1,1,1,1,1,1,1,1,1,1");
      ntok = sscanf(toks, "%d", &pipe_widths[i]);
      assert(ntok == 1);
      toks = strtok(NULL, ",");
    }

    delete[] tokd;

    if (n_thread_per_shader > MAX_THREAD_PER_SM) {
      printf(
          "GPGPU-Sim uArch: Error ** increase MAX_THREAD_PER_SM in "
          "abstract_hardware_model.h from %u to %u\n",
          MAX_THREAD_PER_SM, n_thread_per_shader);
      abort();
    }
    max_warps_per_shader = n_thread_per_shader / warp_size;
    assert(!(n_thread_per_shader % warp_size));

    set_pipeline_latency();

    m_L1I_config.init(m_L1I_config.m_config_string, FuncCachePreferNone);
    m_L1T_config.init(m_L1T_config.m_config_string, FuncCachePreferNone);
    m_L1C_config.init(m_L1C_config.m_config_string, FuncCachePreferNone);
    m_L1D_config.init(m_L1D_config.m_config_string, FuncCachePreferNone);
    gpgpu_cache_texl1_linesize = m_L1T_config.get_line_sz();
    gpgpu_cache_constl1_linesize = m_L1C_config.get_line_sz();
    m_valid = true;

    m_specialized_unit_num = 0;
    // parse the specialized units
    for (unsigned i = 0; i < SPECIALIZED_UNIT_NUM; ++i) {
      unsigned enabled;
      specialized_unit_params sparam;
      sscanf(specialized_unit_string[i], "%u,%u,%u,%u,%u,%s", &enabled,
             &sparam.num_units, &sparam.latency, &sparam.id_oc_spec_reg_width,
             &sparam.oc_ex_spec_reg_width, sparam.name);

      if (enabled) {
        m_specialized_unit.push_back(sparam);
        strncpy(m_specialized_unit.back().name, sparam.name,
                sizeof(m_specialized_unit.back().name));
        m_specialized_unit_num += sparam.num_units;
      } else
        break;  // we only accept continuous specialized_units, i.e., 1,2,3,4
    }

    // parse gpgpu_shmem_option for adpative cache config
    if (adaptive_cache_config) {
      std::stringstream ss(gpgpu_shmem_option);
      while (ss.good()) {
        std::string option;
        std::getline(ss, option, ',');
        shmem_opt_list.push_back((unsigned)std::stoi(option) * 1024);
      }
      std::sort(shmem_opt_list.begin(), shmem_opt_list.end());
    }
  }
  void reg_options(class OptionParser *opp);
  unsigned max_cta(const kernel_info_t &k) const;
  //返回硬件所有的SM（又称Shader Core）的总数。
  unsigned num_shader() const {
    return n_simt_clusters * n_simt_cores_per_cluster;
  }
  //依据SM的ID，获取SIMT Core集群的ID。这里SM的ID，即sid是所有集群的所有SM一起编号的。
  unsigned sid_to_cluster(unsigned sid) const {
    return sid / n_simt_cores_per_cluster;
  }
  //依据SM的ID，获取SIMT Core集群的ID。这里SM的ID，即sid是所有集群的所有SM一起编号的。
  unsigned sid_to_cid(unsigned sid) const {
    return sid % n_simt_cores_per_cluster;
  }
  unsigned cid_to_sid(unsigned cid, unsigned cluster_id) const {
    return cluster_id * n_simt_cores_per_cluster + cid;
  }
  void set_pipeline_latency();

  // backward pointer
  class gpgpu_context *gpgpu_ctx;
  // data
  char *gpgpu_shader_core_pipeline_opt;
  bool gpgpu_perfect_mem;
  bool gpgpu_clock_gated_reg_file;
  bool gpgpu_clock_gated_lanes;
  enum divergence_support_t model;
  //每个Shader Core的线程数。
  unsigned n_thread_per_shader;
  unsigned n_regfile_gating_group;
  unsigned max_warps_per_shader;
  unsigned
      max_cta_per_core;  // Limit on number of concurrent CTAs in shader core
  unsigned max_barriers_per_cta;
  char *gpgpu_scheduler_string;
  //每个线程块或CTA的共享内存大小（默认48KB）。由GPGPU-Sim的-gpgpu_shmem_per_block选项配置。
  unsigned gpgpu_shmem_per_block;
  //每个CTA的最大寄存器数。由GPGPU-Sim的-gpgpu_registers_per_block选项配置。
  unsigned gpgpu_registers_per_block;
  char *pipeline_widths_string;
  int pipe_widths[N_PIPELINE_STAGES];

  mutable cache_config m_L1I_config;
  mutable cache_config m_L1T_config;
  mutable cache_config m_L1C_config;
  mutable l1d_cache_config m_L1D_config;

  bool gpgpu_dwf_reg_bankconflict;

  unsigned gpgpu_num_sched_per_core;
  int gpgpu_max_insn_issue_per_warp;
  bool gpgpu_dual_issue_diff_exec_units;

  // op collector
  bool enable_specialized_operand_collector;
  int gpgpu_operand_collector_num_units_sp;
  int gpgpu_operand_collector_num_units_dp;
  int gpgpu_operand_collector_num_units_sfu;
  int gpgpu_operand_collector_num_units_tensor_core;
  int gpgpu_operand_collector_num_units_mem;
  int gpgpu_operand_collector_num_units_gen;
  int gpgpu_operand_collector_num_units_int;

  unsigned int gpgpu_operand_collector_num_in_ports_sp;
  unsigned int gpgpu_operand_collector_num_in_ports_dp;
  unsigned int gpgpu_operand_collector_num_in_ports_sfu;
  unsigned int gpgpu_operand_collector_num_in_ports_tensor_core;
  unsigned int gpgpu_operand_collector_num_in_ports_mem;
  unsigned int gpgpu_operand_collector_num_in_ports_gen;
  unsigned int gpgpu_operand_collector_num_in_ports_int;

  unsigned int gpgpu_operand_collector_num_out_ports_sp;
  unsigned int gpgpu_operand_collector_num_out_ports_dp;
  unsigned int gpgpu_operand_collector_num_out_ports_sfu;
  unsigned int gpgpu_operand_collector_num_out_ports_tensor_core;
  unsigned int gpgpu_operand_collector_num_out_ports_mem;
  unsigned int gpgpu_operand_collector_num_out_ports_gen;
  unsigned int gpgpu_operand_collector_num_out_ports_int;

  unsigned int gpgpu_num_sp_units;
  unsigned int gpgpu_tensor_core_avail;
  unsigned int gpgpu_num_dp_units;
  unsigned int gpgpu_num_sfu_units;
  unsigned int gpgpu_num_tensor_core_units;
  unsigned int gpgpu_num_mem_units;
  unsigned int gpgpu_num_int_units;

  // Shader core resources
  //每个Shader Core的寄存器数。并发CTA的限制因素之一。由GPGPU-Sim的-gpgpu_shader_registers选项
  //配置。
  unsigned gpgpu_shader_registers;
  int gpgpu_warpdistro_shader;
  int gpgpu_warp_issue_shader;
  unsigned gpgpu_num_reg_banks;
  bool gpgpu_reg_bank_use_warp_id;
  bool gpgpu_local_mem_map;
  bool gpgpu_ignore_resources_limitation;
  bool sub_core_model;

  unsigned max_sp_latency;
  unsigned max_int_latency;
  unsigned max_sfu_latency;
  unsigned max_dp_latency;
  unsigned max_tensor_core_latency;

  //GPU配置的单个SIMT Core集群中SIMT Core的个数。
  unsigned n_simt_cores_per_cluster;
  //GPU配置的SIMT Core集群的个数。
  unsigned n_simt_clusters;
  //GPU配置的SIMT Core集群的弹出缓冲区中的数据包数。弹出缓冲区指的是，[互连网络->弹出缓冲区->SIMT 
  //Core集群]的中间节点。
  unsigned n_simt_ejection_buffer_size;
  unsigned ldst_unit_response_queue_size;

  int simt_core_sim_order;

  unsigned smem_latency;

  unsigned mem2device(unsigned memid) const { return memid + n_simt_clusters; }

  // Jin: concurrent kernel on sm
  //支持SM上的并发内核（默认为禁用）。
  bool gpgpu_concurrent_kernel_sm;

  bool perfect_inst_const_cache;
  unsigned inst_fetch_throughput;
  unsigned reg_file_port_throughput;

  // specialized unit config strings
  char *specialized_unit_string[SPECIALIZED_UNIT_NUM];
  mutable std::vector<specialized_unit_params> m_specialized_unit;
  unsigned m_specialized_unit_num;
};

/*
struct shader_core_stats_pod 是一个用于记录GPU核心统计信息的结构体，在GPGPU-Sim模拟器中扮演重要
角色。在GPGPU-Sim中，struct shader_core_stats_pod 记录了每个GPU核心的处理器时间、指令执行数、存
储器访问数，以及其他有关GPU核心性能和资源使用情况的信息。这些信息可以用于评估GPU核心的性能、检测潜
在的瓶颈问题和优化GPU核心的配置。
*/
struct shader_core_stats_pod {
  void *
      shader_core_stats_pod_start[0];  // DO NOT MOVE FROM THE TOP - spaceless
                                       // pointer to the start of this structure
  unsigned long long *shader_cycles;
  unsigned *m_num_sim_insn;   // number of scalar thread instructions committed
                              // by this shader core
  unsigned *m_num_sim_winsn;  // number of warp instructions committed by this
                              // shader core
  unsigned *m_last_num_sim_insn;
  unsigned *m_last_num_sim_winsn;
  unsigned *
      m_num_decoded_insn;  // number of instructions decoded by this shader core
  float *m_pipeline_duty_cycle;
  unsigned *m_num_FPdecoded_insn;
  unsigned *m_num_INTdecoded_insn;
  unsigned *m_num_storequeued_insn;
  unsigned *m_num_loadqueued_insn;
  unsigned *m_num_tex_inst;
  double *m_num_ialu_acesses;
  double *m_num_fp_acesses;
  double *m_num_imul_acesses;
  double *m_num_fpmul_acesses;
  double *m_num_idiv_acesses;
  double *m_num_fpdiv_acesses;
  double *m_num_sp_acesses;
  double *m_num_sfu_acesses;
  double *m_num_tensor_core_acesses;
  double *m_num_tex_acesses;
  double *m_num_const_acesses;
  double *m_num_dp_acesses;
  double *m_num_dpmul_acesses;
  double *m_num_dpdiv_acesses;
  double *m_num_sqrt_acesses;
  double *m_num_log_acesses;
  double *m_num_sin_acesses;
  double *m_num_exp_acesses;
  double *m_num_mem_acesses;
  unsigned *m_num_sp_committed;
  unsigned *m_num_tlb_hits;
  unsigned *m_num_tlb_accesses;
  unsigned *m_num_sfu_committed;
  unsigned *m_num_tensor_core_committed;
  unsigned *m_num_mem_committed;
  unsigned *m_read_regfile_acesses;
  unsigned *m_write_regfile_acesses;
  unsigned *m_non_rf_operands;
  double *m_num_imul24_acesses;
  double *m_num_imul32_acesses;
  unsigned *m_active_sp_lanes;
  unsigned *m_active_sfu_lanes;
  unsigned *m_active_tensor_core_lanes;
  unsigned *m_active_fu_lanes;
  unsigned *m_active_fu_mem_lanes;
  double *m_active_exu_threads;  // For power model
  double *m_active_exu_warps;    // For power model
  unsigned *m_n_diverge;  // number of divergence occurring in this shader
  unsigned gpgpu_n_load_insn;
  unsigned gpgpu_n_store_insn;
  unsigned gpgpu_n_shmem_insn;
  unsigned gpgpu_n_sstarr_insn;
  unsigned gpgpu_n_tex_insn;
  unsigned gpgpu_n_const_insn;
  unsigned gpgpu_n_param_insn;
  unsigned gpgpu_n_shmem_bkconflict;
  unsigned gpgpu_n_l1cache_bkconflict;
  int gpgpu_n_intrawarp_mshr_merge;
  unsigned gpgpu_n_cmem_portconflict;
  unsigned gpu_stall_shd_mem_breakdown[N_MEM_STAGE_ACCESS_TYPE]
                                      [N_MEM_STAGE_STALL_TYPE];
  unsigned gpu_reg_bank_conflict_stalls;
  unsigned *shader_cycle_distro;
  unsigned *last_shader_cycle_distro;
  unsigned *num_warps_issuable;
  unsigned gpgpu_n_stall_shd_mem;
  unsigned *single_issue_nums;
  unsigned *dual_issue_nums;
  //已经完成的CTA数量。
  unsigned ctas_completed;
  // memory access classification
  int gpgpu_n_mem_read_local;
  int gpgpu_n_mem_write_local;
  int gpgpu_n_mem_texture;
  int gpgpu_n_mem_const;
  int gpgpu_n_mem_read_global;
  int gpgpu_n_mem_write_global;
  int gpgpu_n_mem_read_inst;

  int gpgpu_n_mem_l2_writeback;
  int gpgpu_n_mem_l1_write_allocate;
  int gpgpu_n_mem_l2_write_allocate;

  unsigned made_write_mfs;
  unsigned made_read_mfs;

  unsigned *gpgpu_n_shmem_bank_access;
  long *n_simt_to_mem;  // Interconnect power stats
  long *n_mem_to_simt;
};

class shader_core_stats : public shader_core_stats_pod {
 public:
  shader_core_stats(const shader_core_config *config) {
    m_config = config;
    shader_core_stats_pod *pod = reinterpret_cast<shader_core_stats_pod *>(
        this->shader_core_stats_pod_start);
    memset(pod, 0, sizeof(shader_core_stats_pod));
    shader_cycles = (unsigned long long *)calloc(config->num_shader(),
                                                 sizeof(unsigned long long));
    m_num_sim_insn = (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_num_sim_winsn =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_last_num_sim_winsn =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_last_num_sim_insn =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_pipeline_duty_cycle =
        (float *)calloc(config->num_shader(), sizeof(float));
    m_num_decoded_insn =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_num_FPdecoded_insn =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_num_storequeued_insn =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_num_loadqueued_insn =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_num_tex_inst = (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_num_INTdecoded_insn =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_num_ialu_acesses = (double *)calloc(config->num_shader(), sizeof(double));
    m_num_fp_acesses = (double *)calloc(config->num_shader(), sizeof(double));
    m_num_imul_acesses = (double *)calloc(config->num_shader(), sizeof(double));
    m_num_imul24_acesses =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_imul32_acesses =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_fpmul_acesses =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_idiv_acesses = (double *)calloc(config->num_shader(), sizeof(double));
    m_num_fpdiv_acesses =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_dp_acesses = (double *)calloc(config->num_shader(), sizeof(double));
    m_num_dpmul_acesses =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_dpdiv_acesses =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_sp_acesses = (double *)calloc(config->num_shader(), sizeof(double));
    m_num_sfu_acesses = (double *)calloc(config->num_shader(), sizeof(double));
    m_num_tensor_core_acesses =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_const_acesses =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_tex_acesses = (double *)calloc(config->num_shader(), sizeof(double));
    m_num_sqrt_acesses = (double *)calloc(config->num_shader(), sizeof(double));
    m_num_log_acesses = (double *)calloc(config->num_shader(), sizeof(double));
    m_num_sin_acesses = (double *)calloc(config->num_shader(), sizeof(double));
    m_num_exp_acesses = (double *)calloc(config->num_shader(), sizeof(double));
    m_num_mem_acesses = (double *)calloc(config->num_shader(), sizeof(double));
    m_num_sp_committed =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_num_tlb_hits = (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_num_tlb_accesses =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_active_sp_lanes =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_active_sfu_lanes =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_active_tensor_core_lanes =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_active_fu_lanes =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_active_exu_threads =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_active_exu_warps = (double *)calloc(config->num_shader(), sizeof(double));
    m_active_fu_mem_lanes =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_num_sfu_committed =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_num_tensor_core_committed =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_num_mem_committed =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_read_regfile_acesses =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_write_regfile_acesses =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_non_rf_operands =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_n_diverge = (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    shader_cycle_distro =
        (unsigned *)calloc(config->warp_size + 3, sizeof(unsigned));
    last_shader_cycle_distro =
        (unsigned *)calloc(m_config->warp_size + 3, sizeof(unsigned));
    single_issue_nums =
        (unsigned *)calloc(config->gpgpu_num_sched_per_core, sizeof(unsigned));
    dual_issue_nums =
        (unsigned *)calloc(config->gpgpu_num_sched_per_core, sizeof(unsigned));

    ctas_completed = 0;
    n_simt_to_mem = (long *)calloc(config->num_shader(), sizeof(long));
    n_mem_to_simt = (long *)calloc(config->num_shader(), sizeof(long));

    m_outgoing_traffic_stats = new traffic_breakdown("coretomem");
    m_incoming_traffic_stats = new traffic_breakdown("memtocore");

    gpgpu_n_shmem_bank_access =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));

    m_shader_dynamic_warp_issue_distro.resize(config->num_shader());
    m_shader_warp_slot_issue_distro.resize(config->num_shader());
  }

  ~shader_core_stats() {
    delete m_outgoing_traffic_stats;
    delete m_incoming_traffic_stats;
    free(m_num_sim_insn);
    free(m_num_sim_winsn);
    free(m_num_FPdecoded_insn);
    free(m_num_INTdecoded_insn);
    free(m_num_storequeued_insn);
    free(m_num_loadqueued_insn);
    free(m_num_ialu_acesses);
    free(m_num_fp_acesses);
    free(m_num_imul_acesses);
    free(m_num_tex_inst);
    free(m_num_fpmul_acesses);
    free(m_num_idiv_acesses);
    free(m_num_fpdiv_acesses);
    free(m_num_sp_acesses);
    free(m_num_sfu_acesses);
    free(m_num_tensor_core_acesses);
    free(m_num_tex_acesses);
    free(m_num_const_acesses);
    free(m_num_dp_acesses);
    free(m_num_dpmul_acesses);
    free(m_num_dpdiv_acesses);
    free(m_num_sqrt_acesses);
    free(m_num_log_acesses);
    free(m_num_sin_acesses);
    free(m_num_exp_acesses);
    free(m_num_mem_acesses);
    free(m_num_sp_committed);
    free(m_num_tlb_hits);
    free(m_num_tlb_accesses);
    free(m_num_sfu_committed);
    free(m_num_tensor_core_committed);
    free(m_num_mem_committed);
    free(m_read_regfile_acesses);
    free(m_write_regfile_acesses);
    free(m_non_rf_operands);
    free(m_num_imul24_acesses);
    free(m_num_imul32_acesses);
    free(m_active_sp_lanes);
    free(m_active_sfu_lanes);
    free(m_active_tensor_core_lanes);
    free(m_active_fu_lanes);
    free(m_active_exu_threads);
    free(m_active_exu_warps);
    free(m_active_fu_mem_lanes);
    free(m_n_diverge);
    free(shader_cycle_distro);
    free(last_shader_cycle_distro);
  }

  void new_grid() {}

  void event_warp_issued(unsigned s_id, unsigned warp_id, unsigned num_issued,
                         unsigned dynamic_warp_id);

  void visualizer_print(gzFile visualizer_file);

  void print(FILE *fout) const;

  const std::vector<std::vector<unsigned>> &get_dynamic_warp_issue() const {
    return m_shader_dynamic_warp_issue_distro;
  }

  const std::vector<std::vector<unsigned>> &get_warp_slot_issue() const {
    return m_shader_warp_slot_issue_distro;
  }

 private:
  const shader_core_config *m_config;

  traffic_breakdown *m_outgoing_traffic_stats;  // core to memory partitions
  traffic_breakdown *m_incoming_traffic_stats;  // memory partition to core

  // Counts the instructions issued for each dynamic warp.
  std::vector<std::vector<unsigned>> m_shader_dynamic_warp_issue_distro;
  std::vector<unsigned> m_last_shader_dynamic_warp_issue_distro;
  std::vector<std::vector<unsigned>> m_shader_warp_slot_issue_distro;
  std::vector<unsigned> m_last_shader_warp_slot_issue_distro;

  friend class power_stat_t;
  friend class shader_core_ctx;
  friend class ldst_unit;
  friend class simt_core_cluster;
  friend class scheduler_unit;
  friend class TwoLevelScheduler;
  friend class LooseRoundRobbinScheduler;
};

class memory_config;
class shader_core_mem_fetch_allocator : public mem_fetch_allocator {
 public:
  shader_core_mem_fetch_allocator(unsigned core_id, unsigned cluster_id,
                                  const memory_config *config) {
    m_core_id = core_id;
    m_cluster_id = cluster_id;
    m_memory_config = config;
  }
  mem_fetch *alloc(new_addr_type addr, mem_access_type type, unsigned size,
                   bool wr, unsigned long long cycle,
                   unsigned long long streamID) const;
  mem_fetch *alloc(new_addr_type addr, mem_access_type type,
                   const active_mask_t &active_mask,
                   const mem_access_byte_mask_t &byte_mask,
                   const mem_access_sector_mask_t &sector_mask, unsigned size,
                   bool wr, unsigned long long cycle, unsigned wid,
                   unsigned sid, unsigned tpc, mem_fetch *original_mf,
                   unsigned long long streamID) const;
  mem_fetch *alloc(const warp_inst_t &inst, const mem_access_t &access,
                   unsigned long long cycle) const {
    warp_inst_t inst_copy = inst;
    mem_fetch *mf = new mem_fetch(
        access, &inst_copy, inst.get_streamID(),
        access.is_write() ? WRITE_PACKET_SIZE : READ_PACKET_SIZE,
        inst.warp_id(), m_core_id, m_cluster_id, m_memory_config, cycle);
    return mf;
  }

 private:
  unsigned m_core_id;
  unsigned m_cluster_id;
  const memory_config *m_memory_config;
};

/*
Shader Core。
*/
class shader_core_ctx : public core_t {
 public:
  // creator:
  shader_core_ctx(class gpgpu_sim *gpu, class simt_core_cluster *cluster,
                  unsigned shader_id, unsigned tpc_id,
                  const shader_core_config *config,
                  const memory_config *mem_config, shader_core_stats *stats);

  // used by simt_core_cluster:
  // modifiers
  void cycle();
  void reinit(unsigned start_thread, unsigned end_thread,
              bool reset_not_completed);
  void issue_block2core(class kernel_info_t &kernel);

  void cache_flush();
  void cache_invalidate();
  void accept_fetch_response(mem_fetch *mf);
  void accept_ldst_unit_response(class mem_fetch *mf);
  void broadcast_barrier_reduction(unsigned cta_id, unsigned bar_id,
                                   warp_set_t warps);
  void set_kernel(kernel_info_t *k) {
    assert(k);
    m_kernel = k;
    //        k->inc_running();
    printf("GPGPU-Sim uArch: Shader %d bind to kernel %u \'%s\'\n", m_sid,
           m_kernel->get_uid(), m_kernel->name().c_str());
  }
  PowerscalingCoefficients *scaling_coeffs;
  // accessors
  bool fetch_unit_response_buffer_full() const;
  bool ldst_unit_response_buffer_full() const;
  //返回当前SM未完成的线程数。
  unsigned get_not_completed() const { return m_not_completed; }
  //返回当前SM上的活跃线程块的数量。
  unsigned get_n_active_cta() const { return m_n_active_cta; }

  //m_n_active_cta指当前在此Shader Core上运行的CTA的数量。如果该数量大于0，则代表当前Core是活跃的；反之，
  //则代表当前Core是非活跃的。
  unsigned isactive() const {
    if (m_n_active_cta > 0)
      return 1;
    else
      return 0;
  }
  kernel_info_t *get_kernel() { return m_kernel; }
  unsigned get_sid() const { return m_sid; }

  // used by functional simulation:
  // modifiers
  virtual void warp_exit(unsigned warp_id);

  // Ni: Unset ldgdepbar
  void unset_depbar(const warp_inst_t &inst);

  // accessors
  virtual bool warp_waiting_at_barrier(unsigned warp_id) const;
  void get_pdom_stack_top_info(unsigned tid, unsigned *pc, unsigned *rpc) const;
  float get_current_occupancy(unsigned long long &active,
                              unsigned long long &total) const;

  // used by pipeline timing model components:
  // modifiers
  void mem_instruction_stats(const warp_inst_t &inst);
  void decrement_atomic_count(unsigned wid, unsigned n);
  void inc_store_req(unsigned warp_id) { m_warp[warp_id]->inc_store_req(); }
  void dec_inst_in_pipeline(unsigned warp_id) {
    m_warp[warp_id]->dec_inst_in_pipeline();
  }  // also used in writeback()
  void store_ack(class mem_fetch *mf);
  bool warp_waiting_at_mem_barrier(unsigned warp_id);
  void set_max_cta(const kernel_info_t &kernel);
  void warp_inst_complete(const warp_inst_t &inst);

  // accessors
  std::list<unsigned> get_regs_written(const inst_t &fvt) const;
  const shader_core_config *get_config() const { return m_config; }
  void print_cache_stats(FILE *fp, unsigned &dl1_accesses,
                         unsigned &dl1_misses);

  void get_cache_stats(cache_stats &cs);
  void get_L1I_sub_stats(struct cache_sub_stats &css) const;
  void get_L1D_sub_stats(struct cache_sub_stats &css) const;
  void get_L1C_sub_stats(struct cache_sub_stats &css) const;
  void get_L1T_sub_stats(struct cache_sub_stats &css) const;

  void get_icnt_power_stats(long &n_simt_to_mem, long &n_mem_to_simt) const;

  // debug:
  void display_simt_state(FILE *fout, int mask) const;
  void display_pipeline(FILE *fout, int print_mem, int mask3bit) const;

  void incload_stat() { m_stats->m_num_loadqueued_insn[m_sid]++; }
  void incstore_stat() { m_stats->m_num_storequeued_insn[m_sid]++; }
  void incialu_stat(unsigned active_count, double latency) {
    if (m_config->gpgpu_clock_gated_lanes == false) {
      m_stats->m_num_ialu_acesses[m_sid] =
          m_stats->m_num_ialu_acesses[m_sid] + (double)active_count * latency +
          inactive_lanes_accesses_nonsfu(active_count, latency);
    } else {
      m_stats->m_num_ialu_acesses[m_sid] =
          m_stats->m_num_ialu_acesses[m_sid] + (double)active_count * latency;
    }
    m_stats->m_active_exu_threads[m_sid] += active_count;
    m_stats->m_active_exu_warps[m_sid]++;
  }
  void incimul_stat(unsigned active_count, double latency) {
    if (m_config->gpgpu_clock_gated_lanes == false) {
      m_stats->m_num_imul_acesses[m_sid] =
          m_stats->m_num_imul_acesses[m_sid] + (double)active_count * latency +
          inactive_lanes_accesses_nonsfu(active_count, latency);
    } else {
      m_stats->m_num_imul_acesses[m_sid] =
          m_stats->m_num_imul_acesses[m_sid] + (double)active_count * latency;
    }
    m_stats->m_active_exu_threads[m_sid] += active_count;
    m_stats->m_active_exu_warps[m_sid]++;
  }
  void incimul24_stat(unsigned active_count, double latency) {
    if (m_config->gpgpu_clock_gated_lanes == false) {
      m_stats->m_num_imul24_acesses[m_sid] =
          m_stats->m_num_imul24_acesses[m_sid] +
          (double)active_count * latency +
          inactive_lanes_accesses_nonsfu(active_count, latency);
    } else {
      m_stats->m_num_imul24_acesses[m_sid] =
          m_stats->m_num_imul24_acesses[m_sid] + (double)active_count * latency;
    }
    m_stats->m_active_exu_threads[m_sid] += active_count;
    m_stats->m_active_exu_warps[m_sid]++;
  }
  void incimul32_stat(unsigned active_count, double latency) {
    if (m_config->gpgpu_clock_gated_lanes == false) {
      m_stats->m_num_imul32_acesses[m_sid] =
          m_stats->m_num_imul32_acesses[m_sid] +
          (double)active_count * latency +
          inactive_lanes_accesses_sfu(active_count, latency);
    } else {
      m_stats->m_num_imul32_acesses[m_sid] =
          m_stats->m_num_imul32_acesses[m_sid] + (double)active_count * latency;
    }
    m_stats->m_active_exu_threads[m_sid] += active_count;
    m_stats->m_active_exu_warps[m_sid]++;
  }
  void incidiv_stat(unsigned active_count, double latency) {
    if (m_config->gpgpu_clock_gated_lanes == false) {
      m_stats->m_num_idiv_acesses[m_sid] =
          m_stats->m_num_idiv_acesses[m_sid] + (double)active_count * latency +
          inactive_lanes_accesses_sfu(active_count, latency);
    } else {
      m_stats->m_num_idiv_acesses[m_sid] =
          m_stats->m_num_idiv_acesses[m_sid] + (double)active_count * latency;
    }
    m_stats->m_active_exu_threads[m_sid] += active_count;
    m_stats->m_active_exu_warps[m_sid]++;
  }
  void incfpalu_stat(unsigned active_count, double latency) {
    if (m_config->gpgpu_clock_gated_lanes == false) {
      m_stats->m_num_fp_acesses[m_sid] =
          m_stats->m_num_fp_acesses[m_sid] + (double)active_count * latency +
          inactive_lanes_accesses_nonsfu(active_count, latency);
    } else {
      m_stats->m_num_fp_acesses[m_sid] =
          m_stats->m_num_fp_acesses[m_sid] + (double)active_count * latency;
    }
    m_stats->m_active_exu_threads[m_sid] += active_count;
    m_stats->m_active_exu_warps[m_sid]++;
  }
  void incfpmul_stat(unsigned active_count, double latency) {
    // printf("FP MUL stat increament\n");
    if (m_config->gpgpu_clock_gated_lanes == false) {
      m_stats->m_num_fpmul_acesses[m_sid] =
          m_stats->m_num_fpmul_acesses[m_sid] + (double)active_count * latency +
          inactive_lanes_accesses_nonsfu(active_count, latency);
    } else {
      m_stats->m_num_fpmul_acesses[m_sid] =
          m_stats->m_num_fpmul_acesses[m_sid] + (double)active_count * latency;
    }
    m_stats->m_active_exu_threads[m_sid] += active_count;
    m_stats->m_active_exu_warps[m_sid]++;
  }
  void incfpdiv_stat(unsigned active_count, double latency) {
    if (m_config->gpgpu_clock_gated_lanes == false) {
      m_stats->m_num_fpdiv_acesses[m_sid] =
          m_stats->m_num_fpdiv_acesses[m_sid] + (double)active_count * latency +
          inactive_lanes_accesses_sfu(active_count, latency);
    } else {
      m_stats->m_num_fpdiv_acesses[m_sid] =
          m_stats->m_num_fpdiv_acesses[m_sid] + (double)active_count * latency;
    }
    m_stats->m_active_exu_threads[m_sid] += active_count;
    m_stats->m_active_exu_warps[m_sid]++;
  }
  void incdpalu_stat(unsigned active_count, double latency) {
    if (m_config->gpgpu_clock_gated_lanes == false) {
      m_stats->m_num_dp_acesses[m_sid] =
          m_stats->m_num_dp_acesses[m_sid] + (double)active_count * latency +
          inactive_lanes_accesses_nonsfu(active_count, latency);
    } else {
      m_stats->m_num_dp_acesses[m_sid] =
          m_stats->m_num_dp_acesses[m_sid] + (double)active_count * latency;
    }
    m_stats->m_active_exu_threads[m_sid] += active_count;
    m_stats->m_active_exu_warps[m_sid]++;
  }
  void incdpmul_stat(unsigned active_count, double latency) {
    // printf("FP MUL stat increament\n");
    if (m_config->gpgpu_clock_gated_lanes == false) {
      m_stats->m_num_dpmul_acesses[m_sid] =
          m_stats->m_num_dpmul_acesses[m_sid] + (double)active_count * latency +
          inactive_lanes_accesses_nonsfu(active_count, latency);
    } else {
      m_stats->m_num_dpmul_acesses[m_sid] =
          m_stats->m_num_dpmul_acesses[m_sid] + (double)active_count * latency;
    }
    m_stats->m_active_exu_threads[m_sid] += active_count;
    m_stats->m_active_exu_warps[m_sid]++;
  }
  void incdpdiv_stat(unsigned active_count, double latency) {
    if (m_config->gpgpu_clock_gated_lanes == false) {
      m_stats->m_num_dpdiv_acesses[m_sid] =
          m_stats->m_num_dpdiv_acesses[m_sid] + (double)active_count * latency +
          inactive_lanes_accesses_sfu(active_count, latency);
    } else {
      m_stats->m_num_dpdiv_acesses[m_sid] =
          m_stats->m_num_dpdiv_acesses[m_sid] + (double)active_count * latency;
    }
    m_stats->m_active_exu_threads[m_sid] += active_count;
    m_stats->m_active_exu_warps[m_sid]++;
  }

  void incsqrt_stat(unsigned active_count, double latency) {
    if (m_config->gpgpu_clock_gated_lanes == false) {
      m_stats->m_num_sqrt_acesses[m_sid] =
          m_stats->m_num_sqrt_acesses[m_sid] + (double)active_count * latency +
          inactive_lanes_accesses_sfu(active_count, latency);
    } else {
      m_stats->m_num_sqrt_acesses[m_sid] =
          m_stats->m_num_sqrt_acesses[m_sid] + (double)active_count * latency;
    }
    m_stats->m_active_exu_threads[m_sid] += active_count;
    m_stats->m_active_exu_warps[m_sid]++;
  }

  void inclog_stat(unsigned active_count, double latency) {
    if (m_config->gpgpu_clock_gated_lanes == false) {
      m_stats->m_num_log_acesses[m_sid] =
          m_stats->m_num_log_acesses[m_sid] + (double)active_count * latency +
          inactive_lanes_accesses_sfu(active_count, latency);
    } else {
      m_stats->m_num_log_acesses[m_sid] =
          m_stats->m_num_log_acesses[m_sid] + (double)active_count * latency;
    }
    m_stats->m_active_exu_threads[m_sid] += active_count;
    m_stats->m_active_exu_warps[m_sid]++;
  }

  void incexp_stat(unsigned active_count, double latency) {
    if (m_config->gpgpu_clock_gated_lanes == false) {
      m_stats->m_num_exp_acesses[m_sid] =
          m_stats->m_num_exp_acesses[m_sid] + (double)active_count * latency +
          inactive_lanes_accesses_sfu(active_count, latency);
    } else {
      m_stats->m_num_exp_acesses[m_sid] =
          m_stats->m_num_exp_acesses[m_sid] + (double)active_count * latency;
    }
    m_stats->m_active_exu_threads[m_sid] += active_count;
    m_stats->m_active_exu_warps[m_sid]++;
  }

  void incsin_stat(unsigned active_count, double latency) {
    if (m_config->gpgpu_clock_gated_lanes == false) {
      m_stats->m_num_sin_acesses[m_sid] =
          m_stats->m_num_sin_acesses[m_sid] + (double)active_count * latency +
          inactive_lanes_accesses_sfu(active_count, latency);
    } else {
      m_stats->m_num_sin_acesses[m_sid] =
          m_stats->m_num_sin_acesses[m_sid] + (double)active_count * latency;
    }
    m_stats->m_active_exu_threads[m_sid] += active_count;
    m_stats->m_active_exu_warps[m_sid]++;
  }

  void inctensor_stat(unsigned active_count, double latency) {
    if (m_config->gpgpu_clock_gated_lanes == false) {
      m_stats->m_num_tensor_core_acesses[m_sid] =
          m_stats->m_num_tensor_core_acesses[m_sid] +
          (double)active_count * latency +
          inactive_lanes_accesses_sfu(active_count, latency);
    } else {
      m_stats->m_num_tensor_core_acesses[m_sid] =
          m_stats->m_num_tensor_core_acesses[m_sid] +
          (double)active_count * latency;
    }
    m_stats->m_active_exu_threads[m_sid] += active_count;
    m_stats->m_active_exu_warps[m_sid]++;
  }

  void inctex_stat(unsigned active_count, double latency) {
    if (m_config->gpgpu_clock_gated_lanes == false) {
      m_stats->m_num_tex_acesses[m_sid] =
          m_stats->m_num_tex_acesses[m_sid] + (double)active_count * latency +
          inactive_lanes_accesses_sfu(active_count, latency);
    } else {
      m_stats->m_num_tex_acesses[m_sid] =
          m_stats->m_num_tex_acesses[m_sid] + (double)active_count * latency;
    }
    m_stats->m_active_exu_threads[m_sid] += active_count;
    m_stats->m_active_exu_warps[m_sid]++;
  }

  void inc_const_accesses(unsigned active_count) {
    m_stats->m_num_const_acesses[m_sid] =
        m_stats->m_num_const_acesses[m_sid] + active_count;
  }

  void incsfu_stat(unsigned active_count, double latency) {
    m_stats->m_num_sfu_acesses[m_sid] =
        m_stats->m_num_sfu_acesses[m_sid] + (double)active_count * latency;
  }
  void incsp_stat(unsigned active_count, double latency) {
    m_stats->m_num_sp_acesses[m_sid] =
        m_stats->m_num_sp_acesses[m_sid] + (double)active_count * latency;
  }
  void incmem_stat(unsigned active_count, double latency) {
    if (m_config->gpgpu_clock_gated_lanes == false) {
      m_stats->m_num_mem_acesses[m_sid] =
          m_stats->m_num_mem_acesses[m_sid] + (double)active_count * latency +
          inactive_lanes_accesses_nonsfu(active_count, latency);
    } else {
      m_stats->m_num_mem_acesses[m_sid] =
          m_stats->m_num_mem_acesses[m_sid] + (double)active_count * latency;
    }
  }
  void incexecstat(warp_inst_t *&inst);

  void incregfile_reads(unsigned active_count) {
    m_stats->m_read_regfile_acesses[m_sid] =
        m_stats->m_read_regfile_acesses[m_sid] + active_count;
  }
  void incregfile_writes(unsigned active_count) {
    m_stats->m_write_regfile_acesses[m_sid] =
        m_stats->m_write_regfile_acesses[m_sid] + active_count;
  }
  void incnon_rf_operands(unsigned active_count) {
    m_stats->m_non_rf_operands[m_sid] =
        m_stats->m_non_rf_operands[m_sid] + active_count;
  }

  void incspactivelanes_stat(unsigned active_count) {
    m_stats->m_active_sp_lanes[m_sid] =
        m_stats->m_active_sp_lanes[m_sid] + active_count;
  }
  void incsfuactivelanes_stat(unsigned active_count) {
    m_stats->m_active_sfu_lanes[m_sid] =
        m_stats->m_active_sfu_lanes[m_sid] + active_count;
  }
  void incfuactivelanes_stat(unsigned active_count) {
    m_stats->m_active_fu_lanes[m_sid] =
        m_stats->m_active_fu_lanes[m_sid] + active_count;
  }
  void incfumemactivelanes_stat(unsigned active_count) {
    m_stats->m_active_fu_mem_lanes[m_sid] =
        m_stats->m_active_fu_mem_lanes[m_sid] + active_count;
  }

  void inc_simt_to_mem(unsigned n_flits) {
    m_stats->n_simt_to_mem[m_sid] += n_flits;
  }
  bool check_if_non_released_reduction_barrier(warp_inst_t &inst);

 protected:
  unsigned inactive_lanes_accesses_sfu(unsigned active_count, double latency) {
    return (((32 - active_count) >> 1) * latency) +
           (((32 - active_count) >> 3) * latency) +
           (((32 - active_count) >> 3) * latency);
  }
  unsigned inactive_lanes_accesses_nonsfu(unsigned active_count,
                                          double latency) {
    return (((32 - active_count) >> 1) * latency);
  }

  int test_res_bus(int latency);
  address_type next_pc(int tid) const;
  void fetch();
  void register_cta_thread_exit(unsigned cta_num, kernel_info_t *kernel);

  void decode();

  void issue();
  friend class scheduler_unit;  // this is needed to use private issue warp.
  friend class TwoLevelScheduler;
  friend class LooseRoundRobbinScheduler;
  virtual void issue_warp(register_set &warp, const warp_inst_t *pI,
                          const active_mask_t &active_mask, unsigned warp_id,
                          unsigned sch_id);

  void create_front_pipeline();
  void create_schedulers();
  void create_exec_pipeline();

  // pure virtual methods implemented based on the current execution mode
  // (execution-driven vs trace-driven)
  virtual void init_warps(unsigned cta_id, unsigned start_thread,
                          unsigned end_thread, unsigned ctaid, int cta_size,
                          kernel_info_t &kernel);
  virtual void checkExecutionStatusAndUpdate(warp_inst_t &inst, unsigned t,
                                             unsigned tid) = 0;
  virtual void func_exec_inst(warp_inst_t &inst) = 0;

  virtual unsigned sim_init_thread(kernel_info_t &kernel,
                                   ptx_thread_info **thread_info, int sid,
                                   unsigned tid, unsigned threads_left,
                                   unsigned num_threads, core_t *core,
                                   unsigned hw_cta_id, unsigned hw_warp_id,
                                   gpgpu_t *gpu) = 0;

  virtual void create_shd_warp() = 0;

  virtual const warp_inst_t *get_next_inst(unsigned warp_id,
                                           address_type pc) = 0;
  virtual void get_pdom_stack_top_info(unsigned warp_id, const warp_inst_t *pI,
                                       unsigned *pc, unsigned *rpc) = 0;
  virtual const active_mask_t &get_active_mask(unsigned warp_id,
                                               const warp_inst_t *pI) = 0;

  // Returns numbers of addresses in translated_addrs
  unsigned translate_local_memaddr(address_type localaddr, unsigned tid,
                                   unsigned num_shader, unsigned datasize,
                                   new_addr_type *translated_addrs);

  void read_operands();

  void execute();

  void writeback();

  // used in display_pipeline():
  void dump_warp_state(FILE *fout) const;
  void print_stage(unsigned int stage, FILE *fout) const;

  unsigned long long m_last_inst_gpu_sim_cycle;
  unsigned long long m_last_inst_gpu_tot_sim_cycle;

  // general information
  unsigned m_sid;  // shader id
  unsigned m_tpc;  // texture processor cluster id (aka, node id when using
                   // interconnect concentration)
  const shader_core_config *m_config;
  const memory_config *m_memory_config;
  class simt_core_cluster *m_cluster;

  // statistics
  shader_core_stats *m_stats;

  // CTA scheduling / hardware thread allocation
  //当前在此Shader Core上运行的CTA的数量。
  unsigned m_n_active_cta;  // number of Cooperative Thread Arrays (blocks)
                            // currently running on this shader.
  //m_cta_status是Shader Core内的CTA的状态，MAX_CTA_PER_SHADER是每个Shader Core内的最大可并发
  //CTA个数。m_cta_status[i]里保存了第i个CTA中包含的活跃线程总数量，该数量 <= CTA的总线程数量。
  unsigned m_cta_status[MAX_CTA_PER_SHADER];  // CTAs status
  //未完成的线程数（当此核心上的所有线程都完成时，==0）。
  unsigned m_not_completed;  // number of threads to be completed (==0 when all
                             // thread on this core completed)
  std::bitset<MAX_THREAD_PER_SM> m_active_threads;

  // thread contexts
  //m_threadState[i]标识第i号线程是否处于活跃状态。m_threadState是一个数组，它包含着整个Shader
  //Core的所有的线程的状态。
  thread_ctx_t *m_threadState;

  // interconnect interface
  mem_fetch_interface *m_icnt;
  shader_core_mem_fetch_allocator *m_mem_fetch_allocator;

  // fetch
  //处理指令预取的I-Cache。
  read_only_cache *m_L1I;  // instruction cache
  int m_last_warp_fetched;

  // decode/dispatch
  std::vector<shd_warp_t *> m_warp;  // per warp information array
  barrier_set_t m_barriers;
  //指令获取缓冲区。指令获取缓冲区（ifetch_Buffer_t）对指令缓存（I-cache）和SIMT Core之间的接口进行
  //建模。它有一个成员m_valid，用于指示缓冲区是否有有效的指令。它还将指令的warp id记录在m_warp_id中。
  ifetch_buffer_t m_inst_fetch_buffer;
  std::vector<register_set> m_pipeline_reg;
  Scoreboard *m_scoreboard;
  opndcoll_rfu_t m_operand_collector;
  //在此Shader Core中的活跃warp的总数。
  int m_active_warps;
  std::vector<register_set *> m_specilized_dispatch_reg;

  // schedule
  //每个SIMT Core中，都有可配置数量的调度器单元。
  std::vector<scheduler_unit *> schedulers;

  // issue
  unsigned int Issue_Prio;

  // execute
  unsigned m_num_function_units;
  std::vector<unsigned> m_dispatch_port;
  std::vector<unsigned> m_issue_port;
  //m_fu是SIMD功能单元的向量。m_fu包含：
  //  4个SP单元，4个DP单元，4个INT单元，4个SFU单元，4个TC单元，多个或零个specialized_unit，1个LD/ST单元。
  std::vector<simd_function_unit *>
      m_fu;  // stallable pipelines should be last in this array
  ldst_unit *m_ldst_unit;
  static const unsigned MAX_ALU_LATENCY = 512;
  // there are as many result buses as the width of the EX_WB stage
  //结果总线共有m_config->pipe_widths[EX_WB]条。
  //    num_result_bus = m_config->pipe_widths[EX_WB];
  unsigned num_result_bus;
  std::vector<std::bitset<MAX_ALU_LATENCY> *> m_result_bus;

  // used for local address mapping with single kernel launch
  unsigned kernel_max_cta_per_shader;
  unsigned kernel_padded_threads_per_cta;
  // Used for handing out dynamic warp_ids to new warps.
  // the differnece between a warp_id and a dynamic_warp_id
  // is that the dynamic_warp_id is a running number unique to every warp
  // run on this shader, where the warp_id is the static warp slot.
  unsigned m_dynamic_warp_id;

  // Jin: concurrent kernels on a sm
 public:
  bool can_issue_1block(kernel_info_t &kernel);
  bool occupy_shader_resource_1block(kernel_info_t &kernel, bool occupy);
  void release_shader_resource_1block(unsigned hw_ctaid, kernel_info_t &kernel);
  int find_available_hwtid(unsigned int cta_size, bool occupy);

 private:
  unsigned int m_occupied_n_threads;
  unsigned int m_occupied_shmem;
  unsigned int m_occupied_regs;
  unsigned int m_occupied_ctas;
  std::bitset<MAX_THREAD_PER_SM> m_occupied_hwtid;
  std::map<unsigned int, unsigned int> m_occupied_cta_to_hwtid;
};

class exec_shader_core_ctx : public shader_core_ctx {
 public:
  exec_shader_core_ctx(class gpgpu_sim *gpu, class simt_core_cluster *cluster,
                       unsigned shader_id, unsigned tpc_id,
                       const shader_core_config *config,
                       const memory_config *mem_config,
                       shader_core_stats *stats)
      : shader_core_ctx(gpu, cluster, shader_id, tpc_id, config, mem_config,
                        stats) {
    create_front_pipeline();
    //为当前SM创建所有的warp，warp的数量是m_config->max_warps_per_shader确定。
    create_shd_warp();
    create_schedulers();
    create_exec_pipeline();
  }

  virtual void checkExecutionStatusAndUpdate(warp_inst_t &inst, unsigned t,
                                             unsigned tid);
  virtual void func_exec_inst(warp_inst_t &inst);
  virtual unsigned sim_init_thread(kernel_info_t &kernel,
                                   ptx_thread_info **thread_info, int sid,
                                   unsigned tid, unsigned threads_left,
                                   unsigned num_threads, core_t *core,
                                   unsigned hw_cta_id, unsigned hw_warp_id,
                                   gpgpu_t *gpu);
  //为当前SM创建所有的warp，warp的数量是m_config->max_warps_per_shader确定。
  virtual void create_shd_warp();
  virtual const warp_inst_t *get_next_inst(unsigned warp_id, address_type pc);
  virtual void get_pdom_stack_top_info(unsigned warp_id, const warp_inst_t *pI,
                                       unsigned *pc, unsigned *rpc);
  virtual const active_mask_t &get_active_mask(unsigned warp_id,
                                               const warp_inst_t *pI);
};

/*
SIMT Core集群类。
*/
class simt_core_cluster {
 public:
  //构造函数。
  simt_core_cluster(class gpgpu_sim *gpu, unsigned cluster_id,
                    const shader_core_config *config,
                    const memory_config *mem_config, shader_core_stats *stats,
                    memory_stats_t *mstats);

  void core_cycle();
  void icnt_cycle();

  void reinit();
  unsigned issue_block2core();
  void cache_flush();
  void cache_invalidate();
  bool icnt_injection_buffer_full(unsigned size, bool write);
  void icnt_inject_request_packet(class mem_fetch *mf);

  // for perfect memory interface
  //这里的response_queue指的是SIMT Core集群的响应FIFO。响应FIFO是ICNT->SIMT Core集群的数据包队列，
  //该队列接收ICNT的内存请求。
  bool response_queue_full() {
    return (m_response_fifo.size() >= m_config->n_simt_ejection_buffer_size);
  }
  void push_response_fifo(class mem_fetch *mf) {
    m_response_fifo.push_back(mf);
  }

  void get_pdom_stack_top_info(unsigned sid, unsigned tid, unsigned *pc,
                               unsigned *rpc) const;
  unsigned max_cta(const kernel_info_t &kernel);
  unsigned get_not_completed() const;
  void print_not_completed(FILE *fp) const;
  unsigned get_n_active_cta() const;
  //返回SIMT Core集群中的活跃SM的数量。
  unsigned get_n_active_sms() const;
  gpgpu_sim *get_gpu() { return m_gpu; }

  void display_pipeline(unsigned sid, FILE *fout, int print_mem, int mask);
  void print_cache_stats(FILE *fp, unsigned &dl1_accesses,
                         unsigned &dl1_misses) const;

  void get_cache_stats(cache_stats &cs) const;
  void get_L1I_sub_stats(struct cache_sub_stats &css) const;
  void get_L1D_sub_stats(struct cache_sub_stats &css) const;
  void get_L1C_sub_stats(struct cache_sub_stats &css) const;
  void get_L1T_sub_stats(struct cache_sub_stats &css) const;

  void get_icnt_stats(long &n_simt_to_mem, long &n_mem_to_simt) const;
  float get_current_occupancy(unsigned long long &active,
                              unsigned long long &total) const;
  virtual void create_shader_core_ctx() = 0;

 protected:
  unsigned m_cluster_id;
  gpgpu_sim *m_gpu;
  //Shader Core的配置。
  const shader_core_config *m_config;
  shader_core_stats *m_stats;
  memory_stats_t *m_memory_stats;
  //m_core为SIMT Core集群定义的所有SIMT Core，一个二维shader_core_ctx矩阵，第一维代表集群ID，第
  //二维代表SIMT Core ID。
  shader_core_ctx **m_core;
  const memory_config *m_mem_config;

  unsigned m_cta_issue_next_core;
  std::list<unsigned> m_core_sim_order;
  //每个SIMT Core集群都有一个响应FIFO，用于保存从互连网络发出的数据包。数据包被定向到SIMT Core的
  //指令缓存（如果它是为指令获取未命中提供服务的内存响应）或其内存流水线（memory pipeline，LDST 
  //单元）。数据包以先进先出方式拿出。如果SIMT Core无法接受FIFO头部的数据包，则响应FIFO将停止。为
  //了在LDST单元上生成内存请求，每个SIMT Core都有自己的注入端口接入互连网络。但是，注入端口缓冲区
  //由SIMT Core集群所有SIMT Core共享。mem_fetch定义了一个模拟内存请求的通信结构。更像是一个内存
  //请求的行为。
  std::list<mem_fetch *> m_response_fifo;
};

class exec_simt_core_cluster : public simt_core_cluster {
 public:
  exec_simt_core_cluster(class gpgpu_sim *gpu, unsigned cluster_id,
                         const shader_core_config *config,
                         const memory_config *mem_config,
                         class shader_core_stats *stats,
                         class memory_stats_t *mstats)
      : simt_core_cluster(gpu, cluster_id, config, mem_config, stats, mstats) {
    create_shader_core_ctx();
  }

  virtual void create_shader_core_ctx();
};

/*
SM和存储之间的接口。
*/
class shader_memory_interface : public mem_fetch_interface {
 public:
  shader_memory_interface(shader_core_ctx *core, simt_core_cluster *cluster) {
    m_core = core;
    m_cluster = cluster;
  }
  //返回true，如果ICNT的注入缓冲区已满。
  virtual bool full(unsigned size, bool write) const {
    return m_cluster->icnt_injection_buffer_full(size, write);
  }
  //将内存请求包推入ICNT的注入缓冲区。
  virtual void push(mem_fetch *mf) {
    m_core->inc_simt_to_mem(mf->get_num_flits(true));
    m_cluster->icnt_inject_request_packet(mf);
  }

 private:
  shader_core_ctx *m_core;
  simt_core_cluster *m_cluster;
};

class perfect_memory_interface : public mem_fetch_interface {
 public:
  perfect_memory_interface(shader_core_ctx *core, simt_core_cluster *cluster) {
    m_core = core;
    m_cluster = cluster;
  }
  virtual bool full(unsigned size, bool write) const {
    return m_cluster->response_queue_full();
  }
  virtual void push(mem_fetch *mf) {
    if (mf && mf->isatomic())
      mf->do_atomic();  // execute atomic inside the "memory subsystem"
    m_core->inc_simt_to_mem(mf->get_num_flits(true));
    m_cluster->push_response_fifo(mf);
  }

 private:
  shader_core_ctx *m_core;
  simt_core_cluster *m_cluster;
};

inline int scheduler_unit::get_sid() const { return m_shader->get_sid(); }

#endif /* SHADER_H */
