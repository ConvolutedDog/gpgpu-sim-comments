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
�̵߳�״̬�����ġ�
*/
class thread_ctx_t {
 public:
  //���߳�������CTA��ID��
  unsigned m_cta_id;  // hardware CTA this thread belongs

  // per thread stats (ac stands for accumulative).
  //���߳��ڴ����ָ����������
  unsigned n_insn;
  unsigned n_insn_ac;
  unsigned n_l1_mis_ac;
  unsigned n_l1_mrghit_ac;
  unsigned n_l1_access_ac;
  //��ʶ�߳��Ƿ��ڻ�Ծ״̬��
  bool m_active;
};

/*
ģ�����ں���warp��ģ��״̬����һ������ģ������е�warp�Ķ���SIMT Core����һ��shd_warp_t����ļ��ϣ�
��ģ�����ں���ÿ��warp��ģ��״̬���ֲ���<<#Simt-coreͼ>>��ʾ��I-Buffer��ʵ����shader_core_ctx�ڲ���
shd_warp_t�����С�ÿ��shd_warp_t����һ��m_ibuffer��I-Buffer��Ŀ(ibuffer_entry)�����п����õ�ָ����
����һ�������������ȡ�����ָ������⣬shd_warp_t��һЩ��־����Щ��־������������ȷ��warp�ķ����ʸ�
�洢��ibuffer_entry�еĽ���ָ����һ��ָ��warp_inst_t�����ָ�롣warp_inst_t���й�������ָ��Ĳ�����
�ͺ����ò���������Ϣ��
*/
class shd_warp_t {
 public:
  //���캯���������ֱ�Ϊ��
  //    shader_core_ctx *shader��SIMT Core�Ķ���
  //    unsigned warp_size������warp�ڵ��߳�������warp�Ĵ�С��
  shd_warp_t(class shader_core_ctx *shader, unsigned warp_size)
      : m_shader(shader), m_warp_size(warp_size) {
    //��ʼ���ѷ��͵���δȷ�ϵĴ洢������Ϊ�㡣
    m_stores_outstanding = 0;
    //��ʼ������ˮ����ִ�е�ָ����Ϊ�㡣
    m_inst_in_pipeline = 0;
    reset();
  }
  //��ʼ����
  void reset() {
    assert(m_stores_outstanding == 0);
    assert(m_inst_in_pipeline == 0);
    //��ʼ������warp��ָ���δ���ж������״̬Ϊfalse��
    m_imiss_pending = false;
    //warp ID��ʼ��Ϊ-1��
    m_warp_id = (unsigned)-1;
    //��̬warp ID��ʼ��Ϊ-1��
    m_dynamic_warp_id = (unsigned)-1;
    //�����Ѿ���ɵ��̵߳�����Ϊwarp��С�����滹��Ҫ����Ծ���߳���������
    n_completed = m_warp_size;
    //����δ��ɵ�ԭ�Ӳ�����Ϊ�㡣�������n����not��
    m_n_atomic = 0;
    //����warp����memory barrier״̬�ı�ʶΪfalse��
    m_membar = false;
    //�����߳��˳��ı�ʶΪtrue��
    m_done_exit = true;
    //�����ϴ�ȡָ��ʱ�����ڣ�ʱ��ֵΪ��ʱ�̡�
    m_last_fetch = 0;
    //����ָ�������һ����ȡ��ָ��ı��Ϊ�㡣
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
    //���Ѿ���ɵ��̵߳�������ʼֵ����ȥ��Ծ�̵߳���������Ϊ��Ծ�̴߳���������δ���ִ�С�
    n_completed -= active.count();  // active threads are not yet completed
    //���û�Ծ�̵߳�λͼ��Ϊ����active��
    m_active_threads = active;
    //�����߳��˳��ı�ʶΪfalse��
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
  //����warp�Ѿ�ִ����ϵı�־���Ѿ���ɵ��߳�����=warp�Ĵ�Сʱ���ʹ����warp�Ѿ���ɡ�
  bool functional_done() const;
  //����warp�Ƿ����ڣ�warp�Ѿ�ִ��������ڵȴ����ں˳�ʼ����CTA����barrier��memory barrier������δ
  //��ɵ�ԭ�Ӳ������ĸ��������ڵȴ�״̬��
  bool waiting();  // not const due to membar
  //hardware_done()������warp�Ƿ��Ѿ����ִ�в��ҿ��Ի��ա�
  bool hardware_done() const;
  //�����߳��˳��ı�ʶ��
  bool done_exit() const { return m_done_exit; }
  //�����߳��˳��ı�ʶ��
  void set_done_exit() { m_done_exit = true; }

  void print(FILE *fout) const;
  void print_ibuffer(FILE *fout) const;
  //���ص���warp���Ѿ�ִ����ϵ��߳�������
  unsigned get_n_completed() const { return n_completed; }
  //���ӵ���warp���Ѿ�ִ����ϵ��߳�������m_active_threads�ǻ�Ծ�̵߳�λͼ��Ϊ1����һ���̴߳��ڻ�Ծ
  //״̬�����ｫ��resetΪ�㣬������n_completed��
  void set_completed(unsigned lane) {
    assert(m_active_threads.test(lane));
    m_active_threads.reset(lane);
    n_completed++;
  }
  //�����ϴ�ȡָ��ʱ�����ڣ�ʱ��ֵΪ sim_cycle��
  void set_last_fetch(unsigned long long sim_cycle) {
    m_last_fetch = sim_cycle;
  }
  //����δ��ɵ�ԭ�Ӳ�������
  unsigned get_n_atomic() const { return m_n_atomic; }
  //����δ��ɵ�ԭ�Ӳ�������
  void inc_n_atomic() { m_n_atomic++; }
  //����δ��ɵ�ԭ�Ӳ�������
  void dec_n_atomic(unsigned n) { m_n_atomic -= n; }
  //�����ڴ�����״̬�ı�ʶΪtrue��warp����memory barrier���ȴ���
  void set_membar() { m_membar = true; }
  //����ڴ�����״̬�ı�ʶ������Ϊfalse�����˿�warpû����memory barrier���ȴ���
  void clear_membar() { m_membar = false; }
  //�����ڴ�����״̬�ı�ʶ��
  bool get_membar() const { return m_membar; }
  //����warp����һ��Ҫִ�е�ָ���PCֵ��
  virtual address_type get_pc() const { return m_next_pc; }
  //���ذ��ڵ�ǰShader Core��kernel���ں˺�����Ϣ��kernel_info_t����
  virtual kernel_info_t *get_kernel_info() const;
  //����warp����һ��Ҫִ�е�ָ���PCֵ��
  void set_next_pc(address_type pc) { m_next_pc = pc; }
  //������һ����������ָ���ָ�
  void store_info_of_last_inst_at_barrier(const warp_inst_t *pI) {
    m_inst_at_barrier = *pI;
  }
  //������һ����������ָ���ָ�
  warp_inst_t *restore_info_of_last_inst_at_barrier() {
    return &m_inst_at_barrier;
  }
  //��һ����ָ�����I-Bufer������Ĳ�����
  //    unsigned slot������I-Bufer�Ĳ۱�ţ�
  //    warp_inst_t *pI�������ָ�
  void ibuffer_fill(unsigned slot, const warp_inst_t *pI) {
    assert(slot < IBUFFER_SIZE);
    m_ibuffer[slot].m_inst = pI;
    m_ibuffer[slot].m_valid = true;
    //ָ�������һ����ȡ��ָ��ı�š�
    m_next = 0;
  }
  //����I-Bufer�Ƿ�Ϊ�ա�
  bool ibuffer_empty() const {
    //����I-Bufer���вۣ���һ����Ч�Ļ��ͷ���false��
    for (unsigned i = 0; i < IBUFFER_SIZE; i++)
      if (m_ibuffer[i].m_valid) return false;
    return true;
  }
  //���I-Buffer�е����вۡ�
  void ibuffer_flush() {
    for (unsigned i = 0; i < IBUFFER_SIZE; i++) {
      if (m_ibuffer[i].m_valid) dec_inst_in_pipeline();
      m_ibuffer[i].m_inst = NULL;
      m_ibuffer[i].m_valid = false;
    }
  }
  //����I-Buffer�е���һ����ȡ��ָ�
  const warp_inst_t *ibuffer_next_inst() { return m_ibuffer[m_next].m_inst; }
  //����I-Buffer�е���һ����ȡ��ָ���Ƿ���Ч��
  bool ibuffer_next_valid() { return m_ibuffer[m_next].m_valid; }
  //�ͷ�I-Buffer�е���һ����ȡ��ָ��ۡ�
  void ibuffer_free() {
    m_ibuffer[m_next].m_inst = NULL;
    m_ibuffer[m_next].m_valid = false;
  }
  //ˢ��m_next��ֵ��I-Buffer����һ����ȡ��ָ��ۡ�
  void ibuffer_step() { m_next = (m_next + 1) % IBUFFER_SIZE; }
  //����warp�Ƿ���ָ���δ���ж������״̬��ʶ��
  bool imiss_pending() const { return m_imiss_pending; }
  //����warp��ָ���δ���ж������״̬��
  void set_imiss_pending() { m_imiss_pending = true; }
  //���warp��ָ���δ���ж������״̬��
  void clear_imiss_pending() { m_imiss_pending = false; }
  //��������store�ô������Ƿ��Ѿ�ȫ��ִ���꣬�ѷ��͵���δȷ�ϵĴ洢������m_stores_outstanding=0ʱ��
  //��������store�ô������Ѿ�ȫ��ִ���ꡣ
  bool stores_done() const { return m_stores_outstanding == 0; }
  //�����ѷ��͵���δȷ�ϵĴ洢��������
  void inc_store_req() { m_stores_outstanding++; }
  //�����ѷ��͵���δ�յ�дȷ�ϵĴ洢��������
  void dec_store_req() {
    assert(m_stores_outstanding > 0);
    m_stores_outstanding--;
  }
  //����I-Buffer�е���Чָ���������
  unsigned num_inst_in_buffer() const {
    unsigned count = 0;
    for (unsigned i = 0; i < IBUFFER_SIZE; i++) {
      if (m_ibuffer[i].m_valid) count++;
    }
    return count;
  }
  //��������ˮ����ִ�е�ָ������
  unsigned num_inst_in_pipeline() const { return m_inst_in_pipeline; }
  //�����Ѿ����䵽��ˮ���е�ָ���������������ò�Ƽ��㲻�ԣ�Ҳû�б��õ�����ʱ�Ȳ��ܡ�
  unsigned num_issued_inst_in_pipeline() const {
    return (num_inst_in_pipeline() - num_inst_in_buffer());
  }
  //�����Ƿ�����ˮ����������ִ�е�ָ�
  bool inst_in_pipeline() const { return m_inst_in_pipeline > 0; }
  //��������ˮ����ִ�е�ָ������
  void inc_inst_in_pipeline() { m_inst_in_pipeline++; }
  //��������ˮ����ִ�е�ָ������
  void dec_inst_in_pipeline() {
    assert(m_inst_in_pipeline > 0);
    m_inst_in_pipeline--;
  }
  unsigned long long get_streamID() const { return m_streamID; }
  //����warp���ڵ�CTA��ID��
  unsigned get_cta_id() const { return m_cta_id; }
  //���ض�̬warp��ID��
  unsigned get_dynamic_warp_id() const { return m_dynamic_warp_id; }
  //����warp��ID��
  unsigned get_warp_id() const { return m_warp_id; }
  //����warp���ڵ�SIMT Core����
  class shader_core_ctx *get_shader() {
    return m_shader;
  }

 private:
  //����ָ���Ĵ�СΪ2��
  static const unsigned IBUFFER_SIZE = 2;
  //SIMT Core�Ķ���
  class shader_core_ctx *m_shader;
  unsigned long long m_streamID;
  //warp���ڵ�CTA��ID��
  unsigned m_cta_id;
  unsigned m_warp_id;
  //����warp�ڵ��߳�������warp�Ĵ�С��
  unsigned m_warp_size;
  //��̬warp��ID��
  unsigned m_dynamic_warp_id;
  //warp����һ��Ҫִ�е�ָ���PCֵ����shd_warp_t�������ʱ��������Ϊstart_pc��
  address_type m_next_pc;
  //����warp���Ѿ�ִ����ϵ��߳������������߳������ﵽ32ʱ������һ��warpִ����ϡ�
  unsigned n_completed;  // number of threads in warp completed
  //��Ծ�̵߳�λͼ��Ϊ1����һ���̴߳��ڻ�Ծ״̬��
  std::bitset<MAX_WARP_SIZE> m_active_threads;
  //��ʶ�Ƿ���ָ���δ���ж������״̬��
  bool m_imiss_pending;

  //ָ������Ŀ�ṹ��
  struct ibuffer_entry {
    ibuffer_entry() {
      //��ʼ����Ŀ����Чλ��
      m_valid = false;
      //��ʼ����Ŀ�ڴ����ָ�
      m_inst = NULL;
    }
    //��Ŀ�ڴ����ָ�
    const warp_inst_t *m_inst;
    //��Ŀ����Чλ��
    bool m_valid;
  };

  warp_inst_t m_inst_at_barrier;
  //IBUFFER_SIZE��С��ָ��塣
  ibuffer_entry m_ibuffer[IBUFFER_SIZE];
  //I-Buffer����һ����ȡ��ָ��ۡ�
  unsigned m_next;
  //δ��ɵ�ԭ�Ӳ�������
  unsigned m_n_atomic;  // number of outstanding atomic operations
  //�ڴ�����״̬�ı�ʶ�����Ϊtrue����warp����memory barrier���ȴ���
  bool m_membar;        // if true, warp is waiting at memory barrier

  //�߳��˳��ı�ʶ��һ��Ϊ��warp�е��߳�ע�����߳��˳�����Ϊtrue��
  bool m_done_exit;  // true once thread exit has been registered for threads in
                     // this warp

  //�ϴ�ȡָ��ʱ�����ڣ�ʱ��ֵ.
  unsigned long long m_last_fetch;
  //�ѷ��͵���δȷ�ϵĴ洢��������
  unsigned m_stores_outstanding;  // number of store requests sent but not yet
                                  // acknowledged
  //����ˮ����ִ�е�ָ������
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
//����CTA�ڵ�����warp������С��λͼ����������ڶ��ֹ��ܡ�
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
���ȵ�Ԫ�ࡣÿ�����������warp����ѡ��һ�������ָ���������Щָ�����ִ�С�����Shader Core����
�����������ĵ�������Ԫ��������Ĵ�����Կ�������������Ԫ�ں���һ��scoreboard��һ��SIMTջ��һ���ɹ���
��������Ԫ�ٲõ�warp�Ӽ���m_supervised_warps������һ������ָ����sp_out�ȷ�����ڡ���������Ԫ��
���ķ�����cycle()�����ᱻ�����า�ǣ���ʵ�ֲ�ͬ�ĵ��Ȳ��ԡ���������Ԫ�Ĺ��캯���У������ֱ�Ϊ��
    shader_core_stats *stats��SIMT Core��ͳ����Ϣ����
    shader_core_ctx *shader��SIMT Core����
    Scoreboard *scoreboard��SIMT Core�ļǷ��ƶ���
    simt_stack **simt��SIMTջ��
    std::vector<shd_warp_t *> *warp��SIMT Core�ڵ�����warp��
    register_set *sp_out��SP��Ԫ�ķ�����ڣ�
    register_set *dp_out��DP��Ԫ�ķ�����ڣ�
    register_set *sfu_out��SFU��Ԫ�ķ�����ڣ�
    register_set *int_out��INT��Ԫ�ķ�����ڣ�
    register_set *tensor_core_out��Tensor Core��Ԫ�ķ�����ڣ�
    std::vector<register_set *> &spec_cores_out�����⹦�ܵ�Ԫ�ķ�����ڣ�
    register_set *mem_out���洢����Ԫ�ķ�����ڣ�
    int id����������Ԫ��ID��
*/
class scheduler_unit {  // this can be copied freely, so can be used in std
                        // containers.
 public:
  //���캯����
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
  //���ĵ�����cycle()������ָ����������������֮��ͨ�á�����ͨ������m_next_cycle_prioritized_warps
  //�б���������޸ĵ��ȳ������Ϊ��
  void cycle();

  // These are some common ordering fucntions that the
  // higher order schedulers can take advantage of
  //LRR���Ȳ��Եĵ�������Ԫ��order_warps()������Ϊ��ǰ���ȵ�Ԫ�������ֵ���warp��������order_lrr
  //�Ķ���Ϊ��
  //     void scheduler_unit::order_lrr(
  //         std::vector<T> &result_list, const typename std::vector<T> &input_list,
  //         const typename std::vector<T>::const_iterator &last_issued_from_input,
  //         unsigned num_warps_to_add)
  //�����б�
  //result_list��m_next_cycle_prioritized_warps��һ��vector������洢��ǰ���ȵ�Ԫ��ǰ�ľ���warp
  //             ���������һ�ľ������ȼ����ȵ�warp��
  //input_list��m_supervised_warps����һ��vector������洢��ǰ���ȵ�Ԫ����Ҫ�ٲõ�warp��
  //last_issued_from_input����洢�˵�ǰ���ȵ�Ԫ��һ�ĵ��ȹ���warp��
  //num_warps_to_add��m_supervised_warps.size()�����ǵ�ǰ���ȵ�Ԫ����һ����Ҫ���ȵ�warp��Ŀ������
  //                  �����warp��Ŀ���ǵ�ǰ�����������ֵ���warp�Ӽ���m_supervised_warps�Ĵ�С��
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
  //��������Ը��Ǵ˺�������ʹ������Ȳ������m_supervisored_warps��
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
  //m_supervisored_twarps�б��Ǵ˵��ȳ���Ӧ�����������ٲõ�����warps�����ڴ��ڶ��warp��������
  //ϵͳ�зǳ����á��ڵ���������ϵͳ�У���ֻ�Ƿ�����ú��ĵ�����warp��
  std::vector<shd_warp_t *> m_supervised_warps;
  // This is the iterator pointer to the last supervised warp you issued
  std::vector<shd_warp_t *>::const_iterator m_last_supervised_issued;
  shader_core_stats *m_stats;
  shader_core_ctx *m_shader;
  // these things should become accessors: but would need a bigger rearchitect
  // of how shader_core_ctx interacts with its parts.
  //ÿ��SIMT Core����һ���Ƿ��ơ�
  Scoreboard *m_scoreboard;
  //����ÿ����������Ԫ����һ��SIMT��ջ���С�ÿ��SIMT��ջ��Ӧһ��warp��
  simt_stack **m_simt_stack;
  // warp_inst_t** m_pipeline_reg;
  std::vector<shd_warp_t *> *m_warp;
  //m_sp_out, m_sfu_out, m_mem_out��ָ��SP��SFU��Mem��ˮ�߽��յķ���׶κ�ִ�н׶�֮��ĵ�һ��
  //��ˮ�߼Ĵ�����
  //SP��Ԫ�ķ�����ڡ�
  register_set *m_sp_out;
  //DP��Ԫ�ķ�����ڡ�
  register_set *m_dp_out;
  //SFU��Ԫ�ķ�����ڡ�
  register_set *m_sfu_out;
  //INT��Ԫ�ķ�����ڡ�
  register_set *m_int_out;
  //Tensor Core��Ԫ�ķ�����ڡ�
  register_set *m_tensor_core_out;
  //Mem��Ԫ�ķ�����ڡ�
  register_set *m_mem_out;
  std::vector<register_set *> &m_spec_cores_out;
  //��¼��һ�ķ����ָ������
  unsigned m_num_issued_last_cycle;
  //��RRR���Ȳ������õ���Volta�ܹ��в��õ�LRR���Ȳ��ԣ���ʱ���ܡ�
  unsigned m_current_turn_warp;
  //��������Ԫ��Ψһ��ʶID��
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
�������ռ����ࡣOperand Collector Based Register File Unit��ÿ��SM����һ�������Ĳ������ռ�����
Ӣΰ��Ķ���ר��������һ����Ϊ"�������ռ���"�Ľṹ���������ռ�����һ�黺�������ٲ��߼��������ṩһ
��ʵ����ʹ�ö�bank���˿�RAM���ܹ����ֳ���˿ڼĴ����ļ�����ۡ��������Ž�ʡ����Դ�������������
����������Ҫ��AMD��˾Ҳʹ��bankʽ�Ĵ����ļ���������������ȷ����Щ�ļ��ķ��ʲ��ᷢ��bank��ͻ��

��ָ�������ռ�����Ԫ������������ָ���Դ���������ռ�����Ԫ��������ͨ���Ĵ�������������������
�������ԣ�������Ϊһ�ַ��������żĴ������������ʵ�ʱ�䣬�Ա���һ�������ڶ�һ��bank�ķ��ʲ�����һ�Ρ�
������֯�У��ĸ��ռ�����Ԫ�е�ÿһ��������������������Ŀ��ÿ����������Ŀ���ĸ���һ����Чλ��һ��
�Ĵ�����ʶ����һ������λ�Ͳ��������ݡ�ÿ�������������ֶο�������һ����32�����ֽ�Ԫ����ɵ�128�ֽ�Դ
��������warp��ÿ�������߳���һ�����ֽ�ֵ�������⣬�ռ�����Ԫ����һ����ʶ����������ָ�������ĸ�warp��
�ٲ�������һ��ÿ��bank�Ķ�������У��Ա��ַ�������ֱ�����Ǳ���׼��

��һ��ָ��ӽ���׶��յ���������һ���ռ�����Ԫ����ʱ�������������ָ����Ҳ�������warp ID���Ĵ���
��ʶ������Чλ�����á����⣬Դ�������Ķ�ȡ�������ٲ����б��Ŷӡ�Ϊ�˼���ƣ���ִ�е�Ԫд�ص�������
�������ڶ������ٲ���ѡ��һ������ĸ�����ͻ�ķ������������Ĵ����ļ���Ϊ�˼���Crossbar���ռ�����Ԫ��
�����ѡ��ʱÿ���ռ�����Ԫÿ����ֻ����һ����������

��ÿ���������ӼĴ����ļ��ж�����������Ӧ���ռ�����Ԫʱ��һ��"����λ"�����á���󣬵����еĲ�������׼
�����ˣ�ָ��ͱ����䵽SIMDִ�е�Ԫ��

��GPGPU-Simģ���У�ÿ�������ˮ�ߣ�SP��SFU��MEM������һ��ר�õ��ռ�����Ԫ�����ǹ���һ��ͨ���ռ�����
Ԫ�ء�ÿ����ˮ�߿��õĵ�Ԫ������һ�㵥Ԫ�ص������ǿ����õġ�

�õ�Ԫ������
  1. �˿ڣ�m_in_Ports��������������ˮ�߼Ĵ�������ID_OC��������Ĵ�������OC_EX����ID_OC�˿��е�
     warp_inst_t�����������ռ�����Ԫ�����⣬���ռ�����Ԫ������������Դ�Ĵ���ʱ�������ɵ��ȵ�Ԫ
     ���ȵ�����ܵ��Ĵ�������OC_EX����
  2. �ռ�����Ԫ��m_cu����ÿ���ռ�����Ԫһ�ο�������һ��ָ��������ٲ�Ա���Ͷ�Դ�Ĵ���������һ
     ������Դ�Ĵ�����׼�����ˣ����ȵ�Ԫ�Ϳ��Խ�����ȵ������ˮ�߼Ĵ�������OC_EX����
  3. �ٲ�����m_arbiter�����ٲ������ռ�����Ԫ���ն�Դ������������Ȼ�����������С��ٲ�������ÿ��
     ������Ĵ����ļ�������Bank��ͻ����ֵ��ע����ǣ��ٲ��������ڴ���ԼĴ����ѵ�д�أ�����д�ؾ�
     �бȶ�ȡ���ߵ����ȼ���
  4. ���ȵ�Ԫ��m_Dispatch_units����һ���ռ�����Ԫ׼�����������ȵ�Ԫ�����ռ�����Ԫ�е�warp_inst_t
     ���ȵ�OC_EX�Ĵ�������

�������ռ�������ģΪ����ˮ���е�һ���׶Σ��ɺ���shader_core_ctx::cycle()ִ�С����ڲ������ռ����Ľ�
�ڣ���ο�#ALU��ˮ�ߵĸ���ϸ�ڡ�

opndcoll_rfu_t���ǻ��ڲ������ռ����ļĴ����ļ���Ԫ��ģ�͡��������˶��ռ�����Ԫ�����ٲ����͵��ȵ�Ԫ
���г�����ࡣ

opndcoll_rfu_t::allocate_cu(...)����warp_inst_t�������ָ���Ĳ������ռ������еĿ��в������ռ�
����Ԫ��ͬʱ�������ٲ�������ӦBank����Ϊ���е�Դ����������һ����ȡ����

Ȼ����opndcoll_rfu_t::allocate_reads(...)����û�г�ͻ�Ķ����󣬻��仰˵���ڲ�ͬ�Ĵ���Bank�еĶ�
����Ͳ�ȥͬһ���������ռ����Ķ��������ٲ��������е�������˵��д��������ȼ����ڶ�����

����opndcoll_rfu_t::dispatch_ready_cu()��׼���õĲ������ռ����Ĳ������Ĵ��������в����������ռ���
���䵽ִ�н׶Ρ�

����opndcoll_rfu_t::writeback(const warp_inst_t &inst)���ڴ���ˮ�ߵ�д�ؽ׶α����á�������д��
���䡣

��ǰ���warp�����������ﵥ��Sahder Core�ڵ�warp�������ĸ�����gpgpu_num_sched_per_core���ò�����
����Volta�ܹ�ÿ������4��warp��������ÿ���������Ĵ������룺
     schedulers.push_back(new lrr_scheduler(
             m_stats, this, m_scoreboard, m_simt_stack, &m_warp,
             &m_pipeline_reg[ID_OC_SP], &m_pipeline_reg[ID_OC_DP],
             &m_pipeline_reg[ID_OC_SFU], &m_pipeline_reg[ID_OC_INT],
             &m_pipeline_reg[ID_OC_TENSOR_CORE], m_specilized_dispatch_reg,
             &m_pipeline_reg[ID_OC_MEM], i));
�ڷ�������У�warp���������ɷ����ָ�����ָ�����ͷַ�����ͬ�ĵ�Ԫ����Щ��Ԫ����SP/DP/SFU/INT/
TENSOR_CORE/MEM���ڷ��������ɺ���Ҫ���ָ��ͨ���������ռ�����ָ������Ĳ�����ȫ���ռ��롣����һ
��SM����Ӧ��һ���������ռ������������ķ�����̽�ָ����룺
    m_pipeline_reg[ID_OC_SP]��m_pipeline_reg[ID_OC_DP]��m_pipeline_reg[ID_OC_SFU]��
    m_pipeline_reg[ID_OC_INT]��m_pipeline_reg[ID_OC_TENSOR_CORE]��
    m_pipeline_reg[ID_OC_MEM]
�ȼĴ��������У����Բ������ռ������ռ���������
*/
class opndcoll_rfu_t {  // operand collector based register file unit
 public:
  // constructors
  opndcoll_rfu_t() {
    //�Ĵ����ļ���bank��������������ռ���ʾ��ͼ��
    m_num_banks = 0;
    //�ò������ռ����������ĸ�SM��
    m_shader = NULL;
    //�ò������ռ����ĳ�ʼ��״̬��
    m_initialized = false;
  }
  //����collector unit�������������cu_set����Ϊ��
  //    enum { SP_CUS, DP_CUS, SFU_CUS, TENSOR_CORE_CUS, INT_CUS, MEM_CUS, GEN_CUS };
  //������collector unit�ж������Ӧ��SP��Ԫһ������Ӧ��DP��Ԫһ����......��
  void add_cu_set(unsigned cu_set, unsigned num_cu, unsigned num_dispatch);
  
  //port_vector_t�����Ͷ���Ϊ�洢�Ĵ�������register_set��������
  //    typedef std::vector<register_set *> port_vector_t;
  typedef std::vector<register_set *> port_vector_t;
  
  //uint_vector_t�����Ͷ���Ϊ�洢�ռ�����Ԫset_id��������
  //    typedef std::vector<unsigned int> uint_vector_t;
  typedef std::vector<unsigned int> uint_vector_t;
  
  //add_port�ǽ�����׶εļ�����ˮ�߼Ĵ�������ID_OC_SP�ȣ��Լ������������ռ��������ļĴ�������
  //OC_EX_SP�ȣ���Ӧ�����������ռ�����Ԫset_id����ӽ��������ռ����ࡣ
  void add_port(port_vector_t &input, port_vector_t &ouput,
                uint_vector_t cu_sets);
  void init(unsigned num_banks, shader_core_ctx *shader);

  // modifiers
  bool writeback(warp_inst_t &warp);

  //�������ռ�����ǰִ��һ����
  void step() {
    //�������е��ȵ�Ԫ��ÿ����Ԫ�ҵ�һ��׼���õ��ռ�����Ԫ�����е��ȡ�����ܹ��ֱ�Ӹ���������
    //�ҵ�һ������׼���ÿ��Խ��յ��ռ�����Ԫ�Ļ�����ִ�����ķַ�����dispatch()���ú���ִ�е���
    //Ҫ�����ǣ������ռ�����Ԫ�ռ���Դ�������󣬽�ԭ���ݴ����ռ�����Ԫָ���m_warp�е�ָ���Ƴ�
    //��m_output_register�С�
    dispatch_ready_cu();
    //�ٲ���������󣬲����ز�ͬ�Ĵ���Bank�е�op_t�б�������Щ�Ĵ���Bank������Write״̬����
    //�ú����У��ٲ���������󲢷���op_t���б���Щop_tλ�ڲ�ͬ�ļĴ���Bank�У�������Щ�Ĵ���
    //Bank������Write״̬��
    allocate_reads();
    //�˿ڣ�m_in_Ports��������������ˮ�߼Ĵ������ϣ�ID_OC��������Ĵ������ϣ�OC_EX����ID_OC��
    //���е�warp_inst_t�����������ռ�����Ԫ�����⣬���ռ�����Ԫ������������Դ�Ĵ���ʱ��������
    //���ȵ�Ԫ���ȵ�����ܵ��Ĵ�������OC_EX����m_in_ports���ж��input_port_t����ÿ�������
    //���Ӧ��SP/DP/SFU/INT/MEM/TC��Ԫ������һ����Ԫ���ܻ��ж��input_port_t���󣬲���һһ��
    //Ӧ�ģ����������SP��Ԫ��input_port_t����ʱ��
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
    //��ˣ�m_in_ports����
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
    //���������m_in_ports[p]�ǵ�p��input_port_t����
    for (unsigned p = 0; p < m_in_ports.size(); p++) allocate_cu(p);
    //process_banks()����������Bank��״̬ΪNO_ALLOC������״̬��
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

  //���ص�ǰ�������ռ��������ڵ�SM��
  shader_core_ctx *shader_core() { return m_shader; }

 private:
  void process_banks() { m_arbiter.reset_alloction(); }

  //�������е��ȵ�Ԫ��ÿ����Ԫ�ҵ�һ��׼���õ��ռ�����Ԫ�����е��ȡ�����ܹ��ֱ�Ӹ���������
  //�ҵ�һ������׼���ÿ��Խ��յ��ռ�����Ԫ�Ļ�����ִ�����ķַ�����dispatch()���ú���ִ�е�
  //��Ҫ�����ǣ������ռ�����Ԫ�ռ���Դ�������󣬽�ԭ���ݴ����ռ�����Ԫָ���m_warp�е�ָ����
  //����m_output_register�С�
  void dispatch_ready_cu();
  void allocate_cu(unsigned port);
  //�ٲ���������󣬲����ز�ͬ�Ĵ���Bank�е�op_t�б�������Щ�Ĵ���Bank������Write״̬����
  //�ú����У��ٲ���������󲢷���op_t���б���Щop_tλ�ڲ�ͬ�ļĴ���Bank�У�������Щ�Ĵ���
  //Bank������Write״̬��
  void allocate_reads();

  // types

  class collector_unit_t;

  //����Դ���������ࡣop_t�����洢һ��ָ��ĵ���Դ�������������Ҫ��������Դ����������Ҫʹ��
  //op_t*������
  class op_t {
   public:
    //Դ����������Ч״̬��
    op_t() { m_valid = false; }
    //��ʼ����ǰ����������Ҫ�Ĳ���Ϊ��
    //    collector_unit_t *cu����Ӧ���ĸ��ռ�����Ԫ��
    //    unsigned op��Դ����������ָ�����е�Դ�������е�����
    //    unsigned reg��Դ��������Ӧ�ļĴ�����š�
    //register_bank����������������regnum���ڵ�bank����
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
    //���ص�ǰ�������������ռ�����Ԫ��ID��
    unsigned get_oc_id() const { return m_cu->get_id(); }
    //���ص�ǰ������������Bank��
    unsigned get_bank() const { return m_bank; }
    //���ص�ǰ����������ָ�����е�Դ�������е�����
    unsigned get_operand() const { return m_operand; }
    void dump(FILE *fp) const {
      if (m_cu)
        fprintf(fp, " <R%u, CU:%u, w:%02u> ", m_register, m_cu->get_id(),
                m_cu->get_warp_id());
      else if (!m_warp->empty())
        fprintf(fp, " <R%u, wid:%02u> ", m_register, m_warp->warp_id());
    }
    //���ص�ǰ�������ļĴ����ַ�����
    std::string get_reg_string() const {
      char buffer[64];
      snprintf(buffer, 64, "R%u", m_register);
      return std::string(buffer);
    }

    // modifiers
    //���õ�ǰ��������״̬Ϊ��Ч��
    void reset() { m_valid = false; }

   private:
    //��ǰ�������Ƿ���Ч��
    bool m_valid;
    //��ǰ�������������ռ�����Ԫ��
    collector_unit_t *m_cu;
    //��ǰ������������ָ�
    const warp_inst_t *m_warp;
    //��ǰ����������ָ�����е�Դ�������е�����
    unsigned m_operand;  // operand offset in instruction. e.g., add r1,r2,r3;
                         // r2 is oprd 0, r3 is 1 (r1 is dst)
    //��ǰ��������Ӧ�ļĴ�����š�
    unsigned m_register;
    //��ǰ�������������ĸ�Bank��
    unsigned m_bank;
    //��ǰ����������ָ�����ĸ�����������ġ�
    unsigned m_shced_id;  // scheduler id that has issued this inst
  };

  enum alloc_t {
    NO_ALLOC,
    READ_ALLOC,
    WRITE_ALLOC,
  };

  //����һ��Bank��״̬��һ��allocation_t������һ��Bank��״̬��
  class allocation_t {
   public:
    //��ʼ��ʱ������Bank״̬ΪNO_ALLOC������״̬��
    allocation_t() { m_allocation = NO_ALLOC; }
    //���ص�ǰBank�Ƿ��Ƕ�״̬����״̬ʱ��m_allocationΪREAD_ALLOC��
    bool is_read() const { return m_allocation == READ_ALLOC; }
    //���ص�ǰBank�Ƿ���д״̬��д״̬ʱ��m_allocationΪWRITE_ALLOC��
    bool is_write() const { return m_allocation == WRITE_ALLOC; }
    //���ص�ǰBank�Ƿ��ǿ���״̬������״̬ʱ��m_allocationΪNO_ALLOC��
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
    //��ǰBank�ǿ���״̬ʱ���ſ��Խ������Ϊ��״̬�����Ĳ�����Ϊop��
    void alloc_read(const op_t &op) {
      assert(is_free());
      m_allocation = READ_ALLOC;
      m_op = op;
    }
    //��ǰBank�ǿ���״̬ʱ���ſ��Խ������Ϊд״̬��д�Ĳ�����Ϊop��
    void alloc_write(const op_t &op) {
      assert(is_free());
      m_allocation = WRITE_ALLOC;
      m_op = op;
    }
    //���õ�ǰBank��״̬ΪNO_ALLOC������״̬��
    void reset() { m_allocation = NO_ALLOC; }

   private:
    //m_allocation�Ķ���Ϊ��enum alloc_t {NO_ALLOC, READ_ALLOC, WRITE_ALLOC,}; ���洢�˵�
    //ǰBank��״̬�����ǿ���״̬�����Ƕ�״̬������д״̬��
    enum alloc_t m_allocation;
    //����д��ǰBank�Ĳ�������
    op_t m_op;
  };

  //�ٲ������ٲ�����m_arbiter�����ٲ������ռ�����Ԫ���ն�Դ������������Ȼ�����������С�����
  //��ÿ��������Ĵ����ļ�������Bank��ͻ����ֵ��ע����ǣ��ٲ��������ڴ���ԼĴ����ѵ�д�أ���
  //��д�ؾ��бȶ�ȡ���ߵ����ȼ���
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
      //��ǰ�������ռ������ռ�����Ԫ����Ŀ��
      m_num_collectors = num_cu;
      //��ǰ�������ռ����ļĴ����ļ�Bank����
      m_num_banks = num_banks;
      _inmatch = new int[m_num_banks];
      _outmatch = new int[m_num_collectors];
      _request = new int *[m_num_banks];
      //ÿ�η��ʵ�ǰ�������ռ����ļĴ����ļ��ķ��������ռ�����Ԫ�ĸ�������Ϊÿ���ռ�����Ԫ�ռ�һ
      //����������
      for (unsigned i = 0; i < m_num_banks; i++)
        _request[i] = new int[m_num_collectors];
      //add_read_requests��������ռ�����Ԫ��ȡ���е�Դ���������������Ƿ���m_queue[bank]���С�
      //add_read_requests�����Ķ��壺
      //     void add_read_requests(collector_unit_t *cu) {
      //       //��ȡ��������Ԫ�����в�������src[i]�ǵ�i����������
      //       const op_t *src = cu->get_operands();
      //       for (unsigned i = 0; i < MAX_REG_OPERANDS * 2; i++) {
      //         //�����в�����ѭ����
      //         const op_t &op = src[i];
      //         if (op.valid()) {
      //           //�����������Ч�����ȡ���ǵ�Bank��ţ����������m_queue[bank]���С�
      //           unsigned bank = op.get_bank();
      //           m_queue[bank].push_back(op);
      //         }
      //       }
      //     }
      //���Կ�����m_queue��һ����bank�������Ĳ��������У�m_queue[i]�ǵ�i��bank��ȡ�Ĳ�������
      m_queue = new std::list<op_t>[num_banks];
      //���ڴ洢ÿ��Bank��״̬������NO_ALLOC, READ_ALLOC, WRITE_ALLOC��
      m_allocated_bank = new allocation_t[num_banks];
      //m_allocator_rr_head��һ�����ռ�����Ԫ��IDΪ���������飬m_allocator_rr_head[i]�ǵ�i
      //��cu��һ����Ҫ����Bank��Bank ID��cu # -> next bank to check for request (rr-arb)
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

    //���ռ�����Ԫ��ȡ���е�Դ���������������Ƿ���m_queue[bank]���С�
    void add_read_requests(collector_unit_t *cu) {
      //��ȡ��������Ԫ�����в�������src[i]�ǵ�i����������
      const op_t *src = cu->get_operands();
      for (unsigned i = 0; i < MAX_REG_OPERANDS * 2; i++) {
        //�����в�����ѭ����
        const op_t &op = src[i];
        if (op.valid()) {
          //�����������Ч�����ȡ���ǵ�Bank��ţ����������m_queue[bank]���С�
          unsigned bank = op.get_bank();
          //m_queue��һ����bank�������Ĳ��������У�m_queue[i]�ǵ�i��bank��ȡ�Ĳ�������
          m_queue[bank].push_back(op);
        }
      }
    }
    //m_allocated_bank��һ��״̬�������ڸ���ÿ��register bank��״̬����������������״̬��
    //    READ_ALLOC��WRITE_ALLOC��NO_ALLOC
    //m_queue��һ��FIFO���У����ڻ����register bank�����ж�ȡ���󡣻����ϣ�m_allocated_bank
    //��m_queue�е���Ŀ������SM�����еļĴ���������V100Ϊ8����
    bool bank_idle(unsigned bank) const {
      return m_allocated_bank[bank].is_free();
    }
    //�������bank��Bank��д״̬��д�Ĳ�����Ϊop������m_allocated_bank��
    void allocate_bank_for_write(unsigned bank, const op_t &op) {
      assert(bank < m_num_banks);
      m_allocated_bank[bank].alloc_write(op);
    }
    //�������bank��Bank�Ķ�״̬�����Ĳ�����Ϊop������m_allocated_bank��
    void allocate_for_read(unsigned bank, const op_t &op) {
      assert(bank < m_num_banks);
      m_allocated_bank[bank].alloc_read(op);
    }
    //��������Bank��״̬ΪNO_ALLOC������״̬��
    void reset_alloction() {
      for (unsigned b = 0; b < m_num_banks; b++) m_allocated_bank[b].reset();
    }

   private:
    //��ǰ�������ռ����ļĴ�����Ԫ��Bank��Ŀ��
    unsigned m_num_banks;
    //��ǰ�������ռ������ռ�����Ԫ����Ŀ��
    unsigned m_num_collectors;

    //m_allocated_bank��һ��״̬�������ڸ���ÿ��register bank��״̬����������������״̬��
    //    READ_ALLOC��WRITE_ALLOC��NO_ALLOC
    allocation_t *m_allocated_bank;  // bank # -> register that wins
    //m_queue��һ��FIFO���У����ڻ����register bank�����ж�ȡ����m_queue��һ����bank
    //�������Ĳ��������У�m_queue[i]�ǵ�i��bank��ȡ�Ĳ�������
    std::list<op_t> *m_queue;

    //m_allocator_rr_head��һ�����ռ�����Ԫ��IDΪ���������飬m_allocator_rr_head[i]�ǵ�i
    //��cu��һ����Ҫ����Bank��Bank ID��cu # -> next bank to check for request (rr-arb)
    unsigned *
        m_allocator_rr_head;  // cu # -> next bank to check for request (rr-arb)
    unsigned m_last_cu;       // first cu to check while arb-ing banks (rr)

    int *_inmatch;
    int *_outmatch;
    int **_request;
  };

  //����˿��ࡣinput_port_t�Ķ��壺
  //port_vector_t�����Ͷ���Ϊ�洢�Ĵ�������register_set��������
  //    typedef std::vector<register_set *> port_vector_t;
  //uint_vector_t�����Ͷ���Ϊ�洢�ռ�����Ԫset_id��������
  //    typedef std::vector<unsigned int> uint_vector_t;
  //����add_port�ǽ�����׶εļ�����ˮ�߼Ĵ�������ID_OC_SP�ȣ��Լ������������ռ���������
  //�Ĵ�������OC_EX_SP�ȣ���Ӧ�����������ռ�����Ԫset_id����ӽ��������ռ����࣬�����ߵ�
  //��ϱ���input_port_t����
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

  //�ռ�����Ԫ�ࡣ
  class collector_unit_t {
   public:
    // constructors
    //���캯����
    collector_unit_t() {
      m_free = true;
      m_warp = NULL;
      //�����ռ�����Ԫ�ռ���Դ�������󣬽�ָ���Ƴ���m_output_register�С�
      m_output_register = NULL;
      m_src_op = new op_t[MAX_REG_OPERANDS * 2];
      m_not_ready.reset();
      m_warp_id = -1;
      m_num_banks = 0;
    }
    // accessors
    //���ص�ǰ�ռ�����Ԫ�Ƿ�����Դ��������׼�����ˡ�
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

    //m_not_ready�Ķ���Ϊ��
    //    std::bitset<MAX_REG_OPERANDS * 2> m_not_ready;
    //m_not_ready��һ��λ�����������洢һ��ָ�������Դ�������Ƿ��ڷǾ���״̬����������
    //��op��Դ������Ϊ����״̬��
    void collect_operand(unsigned op) { m_not_ready.reset(op); }
    unsigned get_num_operands() const { return m_warp->get_num_operands(); }
    unsigned get_num_regs() const { return m_warp->get_num_regs(); }
    void dispatch();
    bool is_free() { return m_free; }

   private:
    bool m_free;
    unsigned m_cuid;  // collector unit hw id
    unsigned m_warp_id;
    //��һ��ָ������һ��ָ���m_warp�洢����ָ�
    warp_inst_t *m_warp;
    //�����ռ�����Ԫ�ռ���Դ�������󣬽�ָ���Ƴ���m_output_register�С�
    register_set
        *m_output_register;  // pipeline register to issue to when ready
    op_t *m_src_op;
    //m_not_ready�Ķ���Ϊ��
    //    std::bitset<MAX_REG_OPERANDS * 2> m_not_ready;
    //m_not_ready��һ��λ�����������洢m_warpָ�������Դ�������Ƿ��ڷǾ���״̬��
    std::bitset<MAX_REG_OPERANDS * 2> m_not_ready;
    unsigned m_num_banks;
    opndcoll_rfu_t *m_rfu;

    unsigned m_num_banks_per_sched;
    bool m_sub_core_model;
    //����reg_id��ʵ�Ƕ�Ӧ�ĵ�������ID��
    unsigned m_reg_id;  // if sub_core_model enabled, limit regs this cu can r/w
  };

  //һ������˿ڶ�Ӧһ����������
  class dispatch_unit_t {
   public:
    // for now each collector set gets dedicated dispatch units.
    //Ŀǰ��ÿ���ռ���set����ר�õĵ��ȵ�Ԫ����gpgpu_operand_collector_num_out_ports_sp
    //��ȷ������V100�����У�
    //    gpgpu_operand_collector_num_out_ports_sp = 1
    //    gpgpu_operand_collector_num_out_ports_dp = 0
    //    gpgpu_operand_collector_num_out_ports_sfu = 1
    //    gpgpu_operand_collector_num_out_ports_int = 0
    //    gpgpu_operand_collector_num_out_ports_tensor_core = 1
    //    gpgpu_operand_collector_num_out_ports_mem = 1
    //    gpgpu_operand_collector_num_out_ports_gen = 8
    //������ȵ�Ԫ����Ŀ������˿ڵ���Ŀһ�£�����
    //    ��Ӧ��m_cus[SP_CUS         ]��1����������
    //    ��Ӧ��m_cus[DP_CUS         ]��0����������
    //    ��Ӧ��m_cus[SFU_CUS        ]��1����������
    //    ��Ӧ��m_cus[INT_CUS        ]��0����������
    //    ��Ӧ��m_cus[TENSOR_CORE_CUS]��1����������
    //    ��Ӧ��m_cus[MEM_CUS        ]��1����������
    //    ��Ӧ��m_cus[GEN_CUS        ]��8����������
    //�����ǵ������ĳ�ʼ��������ʱ��
    //    for (unsigned i = 0; i < num_dispatch; i++)
    //      m_dispatch_units.push_back(dispatch_unit_t(&m_cus[set_id]));
    //����Ĳ���cus��m_cus[set_id]����Ӧ��set_id���ռ�����Ԫ��
    //    m_cus[SP_CUS         ]��һ��vector���洢��SP         ��Ԫ��4���ռ�����Ԫ��
    //    m_cus[DP_CUS         ]��һ��vector���洢��DP         ��Ԫ��0���ռ�����Ԫ��
    //    m_cus[SFU_CUS        ]��һ��vector���洢��SFU        ��Ԫ��4���ռ�����Ԫ��
    //    m_cus[INT_CUS        ]��һ��vector���洢��INT        ��Ԫ��0���ռ�����Ԫ��
    //    m_cus[TENSOR_CORE_CUS]��һ��vector���洢��TENSOR_CORE��Ԫ��4���ռ�����Ԫ��
    //    m_cus[MEM_CUS        ]��һ��vector���洢��MEM        ��Ԫ��2���ռ�����Ԫ��
    //    m_cus[GEN_CUS        ]��һ��vector���洢��GEN        ��Ԫ��8���ռ�����Ԫ��
    dispatch_unit_t(std::vector<collector_unit_t> *cus) {
      m_last_cu = 0;
      //��Ӧ��set_id���ռ�����Ԫ������
      m_collector_units = cus;
      //��Ӧ��set_id���ռ�����Ԫ�ĸ�����
      m_num_collectors = (*cus).size();
      m_next_cu = 0;
    }

    //��ʼ����
    void init(bool sub_core_model, unsigned num_warp_scheds) {
      //sub_core_modelģʽ��
      m_sub_core_model = sub_core_model;
      //warp������������
      m_num_warp_scheds = num_warp_scheds;
    }

    //�ҵ�һ������׼���ÿ��Խ��յ��ռ�����Ԫ��
    collector_unit_t *find_ready() {
      // With sub-core enabled round robin starts with the next cu assigned to a
      // different sub-core than the one that dispatched last

      //ÿ��warp���������Էֵ����ռ�����Ԫ�ĸ��������磬�ڴ���m_cus[TENSOR_CORE_CUS]ʱ��
      //m_cus[TENSOR_CORE_CUS]�Ĵ�С��Ϊm_num_collectors��m_cus[TENSOR_CORE_CUS]��һ
      //��vector���洢��TENSOR_CORE��Ԫ��4���ռ�����Ԫ����ô����m_num_warp_scheds=4����
      //cusPerSched=4/4=1�����0�ŵ�������ʹ�õ�0���ռ�����Ԫ��1�ŵ�������ʹ�õ�1���ռ���
      //��Ԫ��2�ŵ�������ʹ�õ�2���ռ�����Ԫ��3�ŵ�������ʹ�õ�3���ռ�����Ԫ��
      unsigned cusPerSched = m_num_collectors / m_num_warp_scheds;
      //rr_increment���ڱ�֤��һ��ѡ����cu����һ��ѡ����cu��ͬ����ͬһ��warp�������ķ�
      //Χ�����磬���m_num_collectors=16��m_num_warp_scheds=4��m_last_cu=0����ô����
      //cusPerSched=4��rr_increment=4������һ��ѡ����cu����4��5��6��7��8��9��10��...��
      //�����ӹ���0��1��2��3����4��cu����4��cu����ͬһ��warp��������
      unsigned rr_increment =
          m_sub_core_model ? cusPerSched - (m_last_cu % cusPerSched) : 1;
      for (unsigned n = 0; n < m_num_collectors; n++) {
        unsigned c = (m_last_cu + n + rr_increment) % m_num_collectors;
        //����ռ�����Ԫ׼�����ˣ���ô�ͷ��ظ��ռ�����Ԫ��ע��������ȵ�Ԫ����Ŀ�������
        //�ڵ���Ŀһ�£�����
        //    ��Ӧ��m_cus[SP_CUS         ]��1����������
        //    ��Ӧ��m_cus[DP_CUS         ]��0����������
        //    ��Ӧ��m_cus[SFU_CUS        ]��1����������
        //    ��Ӧ��m_cus[INT_CUS        ]��0����������
        //    ��Ӧ��m_cus[TENSOR_CORE_CUS]��1����������
        //    ��Ӧ��m_cus[MEM_CUS        ]��1����������
        //    ��Ӧ��m_cus[GEN_CUS        ]��8����������
        //�����m_collector_units��һ��ָ�룬ָ��m_cus[set_id]����Ӧ��set_id���ռ�����
        //Ԫ������ֻ���ȶ�Ӧ��set_id���ռ�����Ԫ��(*m_collector_units)[c]�Ƕ�Ӧ��set_id
        //�ĵ�c���ռ�����Ԫ��������ռ�����Ԫ׼�����ˣ���ô�ͷ��ظ��ռ�����Ԫ��
        if ((*m_collector_units)[c].ready()) {
          m_last_cu = c;
          return &((*m_collector_units)[c]);
        }
      }
      return NULL;
    }

   private:
    //��Ӧ��set_id���ռ�����Ԫ�ĸ�����
    unsigned m_num_collectors;
    //��Ӧ��set_id���ռ�����Ԫ������
    std::vector<collector_unit_t> *m_collector_units;
    //��һ�����ȵ��ռ�����Ԫ��
    unsigned m_last_cu;  // dispatch ready cu's rr
    //û���õ����������
    unsigned m_next_cu;  // for initialization
    //sub_core_modelģʽ��
    bool m_sub_core_model;
    //һ��SM��warp�������ĸ�����
    unsigned m_num_warp_scheds;
  };

  // opndcoll_rfu_t data members
  //�Ƿ�������ռ����Ѿ�����ʼ����
  bool m_initialized;
  //û�õ����������
  unsigned m_num_collector_sets;
  // unsigned m_num_collectors;
  //�Ĵ����ļ���bank������V100�����У�m_num_banks����ʼ��Ϊ16��
  unsigned m_num_banks;
  unsigned m_warp_size;
  //�ռ�����Ԫ�б��ռ�����Ԫ��m_cu����ÿ���ռ�����Ԫһ�ο�������һ��ָ������������Ͷ�Դ�Ĵ�
  //��������һ������Դ�Ĵ�����׼�����ˣ����ȵ�Ԫ�Ϳ��Խ�����ȵ������ˮ�߼Ĵ�������OC_EX����
  std::vector<collector_unit_t *> m_cu;
  //�ٲ������ٲ�����m_arbiter�����ٲ������ռ�����Ԫ���ն�Դ������������Ȼ�����������С�����
  //��ÿ��������Ĵ����ļ�������Bank��ͻ����ֵ��ע����ǣ��ٲ��������ڴ���ԼĴ����ѵ�д�أ���
  //��д�ؾ��бȶ�ȡ���ߵ����ȼ���
  // arbiter_t m_arbiter;                                // yangjianchao16 del
  arbiter_t m_arbiter;
  //ÿ��warp���������õ�bank����sub_core_modelģʽ�У�ÿ��warp���������õ�bank������
  //���޵ġ���V100�����У�����4��warp��������0��warp���������õ�bankΪ0-3��1��warp��
  //�������õ�bankΪ4-7��2��warp���������õ�bankΪ8-11��3��warp���������õ�bankΪ12-
  //15��m_num_banks_per_sched = num_banks / shader->get_config()->gpgpu_num_sched_per_core;
  unsigned m_num_banks_per_sched;
  //ÿ��SM��warp�������ĸ�����
  unsigned m_num_warp_scheds;
  //sub_core_modelģʽ��
  bool sub_core_model;

  // unsigned m_num_ports;
  // std::vector<warp_inst_t**> m_input;
  // std::vector<warp_inst_t**> m_output;
  // std::vector<unsigned> m_num_collector_units;
  // warp_inst_t **m_alu_port;

  //�˿ڣ�m_in_Ports��������������ˮ�߼Ĵ������ϣ�ID_OC��������Ĵ������ϣ�OC_EX����ID_OC�˿��е�
  //warp_inst_t�����������ռ�����Ԫ�����⣬���ռ�����Ԫ������������Դ�Ĵ���ʱ�������ɵ��ȵ�Ԫ����
  //������ܵ��Ĵ�������OC_EX����
  std::vector<input_port_t> m_in_ports;
  //id��Ӧ�ռ�����Ԫ�ĵ��ֵ䡣
  typedef std::map<unsigned /* collector set */,
                   std::vector<collector_unit_t> /*collector sets*/>
      cu_sets_t;
  //�������ռ����ļ��ϡ�
  cu_sets_t m_cus;
  //���ȵ�Ԫ�����ȵ�Ԫ��m_Dispatch_units����һ���ռ�����Ԫ׼�����������ȵ�Ԫ�����ռ�����Ԫ�е�warp
  //_inst_t���ȵ�OC_EX�Ĵ�������
  std::vector<dispatch_unit_t> m_dispatch_units;

  // typedef std::map<warp_inst_t**/*port*/,dispatch_unit_t> port_to_du_t;
  // port_to_du_t                     m_dispatch_units;
  // std::map<warp_inst_t**,std::list<collector_unit_t*> > m_free_cu;
  shader_core_ctx *m_shader;
};

/*
barrier�ļ��ϡ�
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
  //Map<CTA ID������CTA�ڵ�����warp������С��λͼ>��
  typedef std::map<unsigned, warp_set_t> cta_to_warp_t;
  //Map<barrier ID������CTA�ڵ�����warp������С��λͼ>��
  typedef std::map<unsigned, warp_set_t>
      bar_id_to_warp_t; /*set of warps reached a specific barrier id*/

  // individual warp hits barrier
  //����warp����barrier��
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
  //Map<CTA ID������CTA�ڵ�����warp������С��λͼ>��
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
ָ���ȡ��������ָ���ȡ��������ifetch_Buffer_t����ָ��棨I-cache����SM Core֮��Ľӿڽ��н�ģ��
����һ����Աm_valid������ָʾ�������Ƿ�����Ч��ָ�������ָ���warp id��¼��m_warp_id�С�
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
  //��ȡ��ָ���PCֵ��
  address_type m_pc;
  unsigned m_nbytes;
  unsigned m_warp_id;
};

class shader_core_config;

/*
simd_function_unit����ʵ����SP��Ԫ��SFU��Ԫ��ALU��ˮ�ߣ���
*/
class simd_function_unit {
 public:
  //���캯����
  simd_function_unit(const shader_core_config *config);
  ~simd_function_unit() { delete m_dispatch_reg; }

  // modifiers
  //issue(warp_inst_t*&)��Ա��������������ˮ�߼Ĵ�������������m_dispatch_reg��
  virtual void issue(register_set &source_reg);
  virtual void cycle() = 0;
  //lane����˼Ϊһ��warp����32���̣߳�������ˮ�߼Ĵ����п����ݴ��˺ܶ���ָ���Щָ���ÿ��Ӧ���߳���
  //���ÿһλ����һ��lane����������ˮ�߼Ĵ����еķǿ�ָ���������ָ��������߳����루����ָ���߳���
  //��Ļ�ֵ����
  virtual void active_lanes_in_pipeline() = 0;

  // accessors
  virtual unsigned clock_multiplier() const { return 1; }
  //�ж�һ��ָ���ܷ��䣬���ж�m_dispatch_reg�Ƿ�Ϊ�գ�����occupied��Ӧ�ı�ʶλ�Ƿ�Ϊ�ա�
  virtual bool can_issue(const warp_inst_t &inst) const {
    return m_dispatch_reg->empty() && !occupied.test(inst.latency);
  }
  virtual bool is_issue_partitioned() = 0;
  //��ȡ����Ĵ�����ID��
  virtual unsigned get_issue_reg_id() = 0;
  virtual bool stallable() const = 0;
  //��ӡSIMD��Ԫ��dispatch�Ĵ�����
  virtual void print(FILE *fp) const {
    fprintf(fp, "%s dispatch= ", m_name.c_str());
    m_dispatch_reg->print(fp);
  }
  //��ȡSIMD��Ԫ�����ơ�
  const char *get_name() { return m_name.c_str(); }

 protected:
  //SIMD��Ԫ�����ơ�
  std::string m_name;
  const shader_core_config *m_config;
  //SIMD��Ԫ��dispatch�Ĵ�����
  warp_inst_t *m_dispatch_reg;
  //���ALUָ����ӳ٣�����ˮ�߼Ĵ���������512���ۡ�
  static const unsigned MAX_ALU_LATENCY = 512;
  //��ˮ�߼Ĵ�������512���۵�λͼ����ʶÿ�����Ƿ�ռ�á�
  std::bitset<MAX_ALU_LATENCY> occupied;
};

/*
SP��Ԫ��SFU��Ԫ��ʱ��ģ����Ҫ�� shader.h �ж���� pipelined_simd_unit ����ʵ�֡�ģ�ⵥԪ�ľ����ࣨ
sp_unit���sfu�ࣩ�Ǵ���������������ģ��ɿ����ص� can_issue() ��Ա������ָ����Ԫ��ִ�е�ָ�����͡�

SP��Ԫͨ��OC_EX_SP��ˮ�߼Ĵ������ӵ������ռ�����Ԫ��SFU��Ԫͨ��OC_EX_SFU��ˮ�߼Ĵ������ӵ��������ռ�
����Ԫ��������Ԫͨ��WB_EX��ˮ�߼Ĵ�������һ����ͬ��д�ؽ׶Ρ�Ϊ�˷�ֹ������Ԫ��д�ؽ׶εĳ�ͻ��ͣ�ͣ�
ÿ�������κ�һ����Ԫ��ָ������ڷ�����Ŀ�굥Ԫ֮ǰ�ڽ�����ߣ�m_result_bus���Ϸ���һ���ۣ���shader
_core_ctx::execute()����

�ֲ�[ALU��ˮ�����ģ��]�е�ͼ�ṩ��һ��������������pipelined_simd_unit���Ϊ��ͬ���͵�ָ���������
���ӳ١�

��ÿ��pipelined_simd_unit�У�issue(warp_inst_t*&)��Ա��������������ˮ�߼Ĵ�������������m_dispatch_
reg��Ȼ��ָ����m_dispatch_reg�ȴ�initiation_interval�����ڡ��ڴ��ڼ䣬û��������ָ����Է��������
Ԫ����������ȴ���ָ�����������ģ�͡��ȴ�֮��ָ��ɷ����ڲ���ˮ�߼Ĵ���m_pipeline_reg�����ӳٽ�
ģ���ɷ���λ����ȷ���ģ�������m_dispatch_reg�л��ѵ�ʱ��Ҳ�������ӳ��С�ÿ�����ڣ�ָ�ͨ����ˮ�߼�
����ǰ�������ս���m_result_port�����ǹ������ˮ�߼Ĵ�����ͨ��SP��SFU��Ԫ�Ĺ�ͬд�ؽ׶Ρ�

����ָ������������ӳ���cuda-sim.cc��ptx_instruction::set_opcode_and_latency()��ָ�������������Ԥ
����ʱ�����á�
*/
class pipelined_simd_unit : public simd_function_unit {
 public:
  pipelined_simd_unit(register_set *result_port,
                      const shader_core_config *config, unsigned max_latency,
                      shader_core_ctx *core, unsigned issue_reg_id);

  // modifiers
  virtual void cycle();
  //issue(warp_inst_t*&)��Ա��������������ˮ�߼Ĵ�������������m_dispatch_reg��
  virtual void issue(register_set &source_reg);
  //lane����˼Ϊһ��warp����32���̣߳�������ˮ�߼Ĵ����п����ݴ��˺ܶ���ָ���Щָ���ÿ��Ӧ���߳���
  //���ÿһλ����һ��lane����������ˮ�߼Ĵ����еķǿ�ָ���������ָ��������߳����루����ָ���߳���
  //��Ļ�ֵ����
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
  //�ж�һ��ָ���ܷ��䣬���ж�m_dispatch_reg�Ƿ�Ϊ�գ�����occupied��Ӧ�ı�ʶλ�Ƿ�Ϊ�ա�
  virtual bool can_issue(const warp_inst_t &inst) const {
    return simd_function_unit::can_issue(inst);
  }
  virtual bool is_issue_partitioned() = 0;
  //��ȡ����Ĵ�����ID��
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
  //��ˮ�ߵ���ȡ�
  unsigned m_pipeline_depth;
  //��ˮ�߼Ĵ�����
  warp_inst_t **m_pipeline_reg;
  //����˿ڡ�
  register_set *m_result_port;
  class shader_core_ctx *m_core;
  //����Ĵ�����ID��
  unsigned m_issue_reg_id;  // if sub_core_model is enabled we can only issue
                            // from a subset of operand collectors

  unsigned active_insts_in_pipeline;
};

/*
���⹦�ܵ�Ԫ�Ķ��塣
*/
class sfu : public pipelined_simd_unit {
 public:
  //SFU���⹦�ܵ�Ԫ�Ĺ��캯������m_name��ͬ��
  sfu(register_set *result_port, const shader_core_config *config,
      shader_core_ctx *core, unsigned issue_reg_id);
  //��������ΪSFU_OP/ALU_SFU_OP�Լ��Լ�������С��29��DP_OP(compute <= 29)�Żᷢ�䵽SFU��
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
DP��Ԫ�Ķ��塣
*/
class dp_unit : public pipelined_simd_unit {
 public:
  dp_unit(register_set *result_port, const shader_core_config *config,
          shader_core_ctx *core, unsigned issue_reg_id);
  //��������ΪDP_OP�Żᷢ�䵽SFU��
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
LDST��Ԫ�ࡣldst_unit��ʵ����Shader��ˮ�ߵ��ڴ�׶Σ�ʵ�������������е�Shader�ڴ棺����m_L1T����
������m_L1C�������ݣ�m_L1D����ldst_unit::cycle()ʵ���˸õ�Ԫ������������ǰ�ƽ�������ÿ���ڱ�����
m_config->mem_warp_parts������

ldst_unit::cycle()�������Ի���������ڴ���Ӧ���洢��m_response_fifo�У�����仺�沢��Ǵ洢Ϊ��ɡ�
�ú�����ʹ�û���������ǰ�ƽ����Ա����ǿ����������緢������Miss�����ݵ����󡣶�ÿ�����͵�L1�洢�Ļ�
����ʷֱ���shared_cycle()��constant_cycle()��texture_cycle()��memory_cycle()����ɡ� 

memory_cycle���ڷ���L1 data cache����Щ�����е�ÿһ���������process_memory_access_queue()������
һ��ͨ�ú�������ָ����ڲ����ʶ����г�ȡһ�����ʣ�������������͵������С���������������������ڲ�
�ܱ�����Ҳ����˵������û�д��Ҳû�����л��棬����ܷ����ڸ���ϵͳ�����Ѿ����˵�����£�����������
lines in a particular way����reserved����û�б�filled������ô������ʽ�����һ�������ٴγ��ԡ�

ֵ��ע����ǣ����������е�ָ��ܵ���õ�Ԫ��д�ؽ׶Ρ����еĴ洢ָ��ͼ���ָ������������Ļ���鱻��
�е�����¶�����cycle()�������˳���ˮ�ߡ�������Ϊ���ǲ���Ҫ�ȴ������������Ӧ�������ƹ�д���߼�����ָ
���������cache lines���Ѿ����ص�cache lines��¼������
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
  //LDST��Ԫissue������
  virtual void issue(register_set &inst);
  bool is_issue_partitioned() { return false; }
  virtual void cycle();

  void fill(mem_fetch *mf);
  void flush();
  void invalidate();
  void writeback();

  // accessors
  //ʱ�ӱ�������һЩ��Ԫ�����ڸ��ߵ�ѭ�����������С�
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
  //��һ����Ҫд�ص�ָ�
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
��ˮ�߽׶�����N_PIPELINE_STAGES�����ǽ׶ε��ܸ�����
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
��ˮ�߽׶�����N_PIPELINE_STAGES�����ǽ׶ε��ܸ�����
*/
const char *const pipeline_stage_name_decode[] = {
    "ID_OC_SP",          "ID_OC_DP",         "ID_OC_INT", "ID_OC_SFU",
    "ID_OC_MEM",         "OC_EX_SP",         "OC_EX_DP",  "OC_EX_INT",
    "OC_EX_SFU",         "OC_EX_MEM",        "EX_WB",     "ID_OC_TENSOR_CORE",
    "OC_EX_TENSOR_CORE", "N_PIPELINE_STAGES"};

/*
��SP/DP/INT/TC/MEM/SFU�ȵ�Ԫ������ؾ��幤����Ԫ����Ϣ������ָ��������������SPECIALIZED_UNIT_NUM
ָ����
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
Shader Core�������ࡣ
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
      //�����ȡ����������ˮ�߽׶εĿ�ȣ���������-gpgpu_pipeline_widths�����ã�
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
  //����Ӳ�����е�SM���ֳ�Shader Core����������
  unsigned num_shader() const {
    return n_simt_clusters * n_simt_cores_per_cluster;
  }
  //����SM��ID����ȡSIMT Core��Ⱥ��ID������SM��ID����sid�����м�Ⱥ������SMһ���ŵġ�
  unsigned sid_to_cluster(unsigned sid) const {
    return sid / n_simt_cores_per_cluster;
  }
  //����SM��ID����ȡSIMT Core��Ⱥ��ID������SM��ID����sid�����м�Ⱥ������SMһ���ŵġ�
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
  //ÿ��Shader Core���߳�����
  unsigned n_thread_per_shader;
  unsigned n_regfile_gating_group;
  unsigned max_warps_per_shader;
  unsigned
      max_cta_per_core;  // Limit on number of concurrent CTAs in shader core
  unsigned max_barriers_per_cta;
  char *gpgpu_scheduler_string;
  //ÿ���߳̿��CTA�Ĺ����ڴ��С��Ĭ��48KB������GPGPU-Sim��-gpgpu_shmem_per_blockѡ�����á�
  unsigned gpgpu_shmem_per_block;
  //ÿ��CTA�����Ĵ���������GPGPU-Sim��-gpgpu_registers_per_blockѡ�����á�
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
  //ÿ��Shader Core�ļĴ�����������CTA����������֮һ����GPGPU-Sim��-gpgpu_shader_registersѡ��
  //���á�
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

  //GPU���õĵ���SIMT Core��Ⱥ��SIMT Core�ĸ�����
  unsigned n_simt_cores_per_cluster;
  //GPU���õ�SIMT Core��Ⱥ�ĸ�����
  unsigned n_simt_clusters;
  //GPU���õ�SIMT Core��Ⱥ�ĵ����������е����ݰ���������������ָ���ǣ�[��������->����������->SIMT 
  //Core��Ⱥ]���м�ڵ㡣
  unsigned n_simt_ejection_buffer_size;
  unsigned ldst_unit_response_queue_size;

  int simt_core_sim_order;

  unsigned smem_latency;

  unsigned mem2device(unsigned memid) const { return memid + n_simt_clusters; }

  // Jin: concurrent kernel on sm
  //֧��SM�ϵĲ����ںˣ�Ĭ��Ϊ���ã���
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
struct shader_core_stats_pod ��һ�����ڼ�¼GPU����ͳ����Ϣ�Ľṹ�壬��GPGPU-Simģ�����а�����Ҫ
��ɫ����GPGPU-Sim�У�struct shader_core_stats_pod ��¼��ÿ��GPU���ĵĴ�����ʱ�䡢ָ��ִ��������
�������������Լ������й�GPU�������ܺ���Դʹ���������Ϣ����Щ��Ϣ������������GPU���ĵ����ܡ����Ǳ
�ڵ�ƿ��������Ż�GPU���ĵ����á�
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
  //�Ѿ���ɵ�CTA������
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
Shader Core��
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
  //���ص�ǰSMδ��ɵ��߳�����
  unsigned get_not_completed() const { return m_not_completed; }
  //���ص�ǰSM�ϵĻ�Ծ�߳̿��������
  unsigned get_n_active_cta() const { return m_n_active_cta; }

  //m_n_active_ctaָ��ǰ�ڴ�Shader Core�����е�CTA���������������������0�������ǰCore�ǻ�Ծ�ģ���֮��
  //�����ǰCore�Ƿǻ�Ծ�ġ�
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
  //��ǰ�ڴ�Shader Core�����е�CTA��������
  unsigned m_n_active_cta;  // number of Cooperative Thread Arrays (blocks)
                            // currently running on this shader.
  //m_cta_status��Shader Core�ڵ�CTA��״̬��MAX_CTA_PER_SHADER��ÿ��Shader Core�ڵ����ɲ���
  //CTA������m_cta_status[i]�ﱣ���˵�i��CTA�а����Ļ�Ծ�߳��������������� <= CTA�����߳�������
  unsigned m_cta_status[MAX_CTA_PER_SHADER];  // CTAs status
  //δ��ɵ��߳��������˺����ϵ������̶߳����ʱ��==0����
  unsigned m_not_completed;  // number of threads to be completed (==0 when all
                             // thread on this core completed)
  std::bitset<MAX_THREAD_PER_SM> m_active_threads;

  // thread contexts
  //m_threadState[i]��ʶ��i���߳��Ƿ��ڻ�Ծ״̬��m_threadState��һ�����飬������������Shader
  //Core�����е��̵߳�״̬��
  thread_ctx_t *m_threadState;

  // interconnect interface
  mem_fetch_interface *m_icnt;
  shader_core_mem_fetch_allocator *m_mem_fetch_allocator;

  // fetch
  //����ָ��Ԥȡ��I-Cache��
  read_only_cache *m_L1I;  // instruction cache
  int m_last_warp_fetched;

  // decode/dispatch
  std::vector<shd_warp_t *> m_warp;  // per warp information array
  barrier_set_t m_barriers;
  //ָ���ȡ��������ָ���ȡ��������ifetch_Buffer_t����ָ��棨I-cache����SIMT Core֮��Ľӿڽ���
  //��ģ������һ����Աm_valid������ָʾ�������Ƿ�����Ч��ָ�������ָ���warp id��¼��m_warp_id�С�
  ifetch_buffer_t m_inst_fetch_buffer;
  std::vector<register_set> m_pipeline_reg;
  Scoreboard *m_scoreboard;
  opndcoll_rfu_t m_operand_collector;
  //�ڴ�Shader Core�еĻ�Ծwarp��������
  int m_active_warps;
  std::vector<register_set *> m_specilized_dispatch_reg;

  // schedule
  //ÿ��SIMT Core�У����п����������ĵ�������Ԫ��
  std::vector<scheduler_unit *> schedulers;

  // issue
  unsigned int Issue_Prio;

  // execute
  unsigned m_num_function_units;
  std::vector<unsigned> m_dispatch_port;
  std::vector<unsigned> m_issue_port;
  //m_fu��SIMD���ܵ�Ԫ��������m_fu������
  //  4��SP��Ԫ��4��DP��Ԫ��4��INT��Ԫ��4��SFU��Ԫ��4��TC��Ԫ����������specialized_unit��1��LD/ST��Ԫ��
  std::vector<simd_function_unit *>
      m_fu;  // stallable pipelines should be last in this array
  ldst_unit *m_ldst_unit;
  static const unsigned MAX_ALU_LATENCY = 512;
  // there are as many result buses as the width of the EX_WB stage
  //������߹���m_config->pipe_widths[EX_WB]����
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
    //Ϊ��ǰSM�������е�warp��warp��������m_config->max_warps_per_shaderȷ����
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
  //Ϊ��ǰSM�������е�warp��warp��������m_config->max_warps_per_shaderȷ����
  virtual void create_shd_warp();
  virtual const warp_inst_t *get_next_inst(unsigned warp_id, address_type pc);
  virtual void get_pdom_stack_top_info(unsigned warp_id, const warp_inst_t *pI,
                                       unsigned *pc, unsigned *rpc);
  virtual const active_mask_t &get_active_mask(unsigned warp_id,
                                               const warp_inst_t *pI);
};

/*
SIMT Core��Ⱥ�ࡣ
*/
class simt_core_cluster {
 public:
  //���캯����
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
  //�����response_queueָ����SIMT Core��Ⱥ����ӦFIFO����ӦFIFO��ICNT->SIMT Core��Ⱥ�����ݰ����У�
  //�ö��н���ICNT���ڴ�����
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
  //����SIMT Core��Ⱥ�еĻ�ԾSM��������
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
  //Shader Core�����á�
  const shader_core_config *m_config;
  shader_core_stats *m_stats;
  memory_stats_t *m_memory_stats;
  //m_coreΪSIMT Core��Ⱥ���������SIMT Core��һ����άshader_core_ctx���󣬵�һά����ȺID����
  //��ά����SIMT Core ID��
  shader_core_ctx **m_core;
  const memory_config *m_mem_config;

  unsigned m_cta_issue_next_core;
  std::list<unsigned> m_core_sim_order;
  //ÿ��SIMT Core��Ⱥ����һ����ӦFIFO�����ڱ���ӻ������緢�������ݰ������ݰ�������SIMT Core��
  //ָ��棨�������Ϊָ���ȡδ�����ṩ������ڴ���Ӧ�������ڴ���ˮ�ߣ�memory pipeline��LDST 
  //��Ԫ�������ݰ����Ƚ��ȳ���ʽ�ó������SIMT Core�޷�����FIFOͷ�������ݰ�������ӦFIFO��ֹͣ��Ϊ
  //����LDST��Ԫ�������ڴ�����ÿ��SIMT Core�����Լ���ע��˿ڽ��뻥�����硣���ǣ�ע��˿ڻ�����
  //��SIMT Core��Ⱥ����SIMT Core����mem_fetch������һ��ģ���ڴ������ͨ�Žṹ��������һ���ڴ�
  //�������Ϊ��
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
SM�ʹ洢֮��Ľӿڡ�
*/
class shader_memory_interface : public mem_fetch_interface {
 public:
  shader_memory_interface(shader_core_ctx *core, simt_core_cluster *cluster) {
    m_core = core;
    m_cluster = cluster;
  }
  //����true�����ICNT��ע�뻺����������
  virtual bool full(unsigned size, bool write) const {
    return m_cluster->icnt_injection_buffer_full(size, write);
  }
  //���ڴ����������ICNT��ע�뻺������
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
