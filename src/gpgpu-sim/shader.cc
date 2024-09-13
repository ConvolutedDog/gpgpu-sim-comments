// Copyright (c) 2009-2021, Tor M. Aamodt, Wilson W.L. Fung, Ali Bakhoda,
// George L. Yuan, Andrew Turner, Inderpreet Singh, Vijay Kandiah, Nikos
// Hardavellas, Mahmoud Khairy, Junrui Pan, Timothy G. Rogers The University of
// British Columbia, Northwestern University, Purdue University All rights
// reserved.
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

/*
shader.cc��SIMT Core��ʱ��ģ�͡�������cudu-sim��һ���ض����߳̽��й���ģ�⣬cuda-sim�����ظ��߳�
���������е���Ϣ��
*/

#include "shader.h"
#include <float.h>
#include <limits.h>
#include <string.h>
#include "../../libcuda/gpgpu_context.h"
#include "../cuda-sim/cuda-sim.h"
#include "../cuda-sim/ptx-stats.h"
#include "../cuda-sim/ptx_sim.h"
#include "../statwrapper.h"
#include "addrdec.h"
#include "dram.h"
#include "gpu-misc.h"
#include "gpu-sim.h"
#include "icnt_wrapper.h"
#include "mem_fetch.h"
#include "mem_latency_stat.h"
#include "shader_trace.h"
#include "stat-tool.h"
#include "traffic_breakdown.h"
#include "visualizer.h"

#define PRIORITIZE_MSHR_OVER_WB 1
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

mem_fetch *shader_core_mem_fetch_allocator::alloc(
    new_addr_type addr, mem_access_type type, unsigned size, bool wr,
    unsigned long long cycle, unsigned long long streamID) const {
  mem_access_t access(type, addr, size, wr, m_memory_config->gpgpu_ctx);
  mem_fetch *mf = new mem_fetch(
      access, NULL, streamID, wr ? WRITE_PACKET_SIZE : READ_PACKET_SIZE, -1,
      m_core_id, m_cluster_id, m_memory_config, cycle);
  return mf;
}

mem_fetch *shader_core_mem_fetch_allocator::alloc(
    new_addr_type addr, mem_access_type type, const active_mask_t &active_mask,
    const mem_access_byte_mask_t &byte_mask,
    const mem_access_sector_mask_t &sector_mask, unsigned size, bool wr,
    unsigned long long cycle, unsigned wid, unsigned sid, unsigned tpc,
    mem_fetch *original_mf, unsigned long long streamID) const {
  mem_access_t access(type, addr, size, wr, active_mask, byte_mask, sector_mask,
                      m_memory_config->gpgpu_ctx);
  mem_fetch *mf = new mem_fetch(
      access, NULL, streamID, wr ? WRITE_PACKET_SIZE : READ_PACKET_SIZE, wid,
      m_core_id, m_cluster_id, m_memory_config, cycle, original_mf);
  return mf;
}
/////////////////////////////////////////////////////////////////////////////
/*
��ȡһ��ָ������д�صļĴ�����ţ����б�ʽ���ء�
*/
std::list<unsigned> shader_core_ctx::get_regs_written(const inst_t &fvt) const {
  std::list<unsigned> result;
  for (unsigned op = 0; op < MAX_REG_OPERANDS; op++) {
    int reg_num = fvt.arch_reg.dst[op];  // this math needs to match that used
                                         // in function_info::ptx_decode_inst
    if (reg_num >= 0)                    // valid register
      result.push_back(reg_num);
  }
  return result;
}

/*
Ϊ��ǰSM�������е�warp��warp��������m_config->max_warps_per_shaderȷ����
*/
void exec_shader_core_ctx::create_shd_warp() {
  m_warp.resize(m_config->max_warps_per_shader);
  for (unsigned k = 0; k < m_config->max_warps_per_shader; ++k) {
    m_warp[k] = new shd_warp_t(this, m_config->warp_size);
  }
}

/*
Ϊ��ǰSM�������е���ˮ�ߵ�Ԫ��
*/
void shader_core_ctx::create_front_pipeline() {
  // pipeline_stages is the sum of normal pipeline stages and specialized_unit
  // stages * 2 (for ID and EX)
  //ȫ������ˮ�߽׶���������ͨ��ˮ�߽׶�������ר�õ�Ԫ����ˮ�߽׶�������2��ID��EX����
  unsigned total_pipeline_stages =
      N_PIPELINE_STAGES + m_config->m_specialized_unit.size() * 2;
  m_pipeline_reg.reserve(total_pipeline_stages);
  //��ˮ�߽׶εĿ��������-gpgpu_pipeline_widths�����ã�
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
  //��V100������Ϊ��-gpgpu_pipeline_widths 4,4,4,4,4,4,4,4,4,4,8,4,4
  //������ˮ�ߵĿ����ʵ������ˮ�߼Ĵ����ڲ��ܹ��洢������ָ���������Ǻ͵�Ԫ����Ŀ��
  //���ġ�
  for (int j = 0; j < N_PIPELINE_STAGES; j++) {
    //����N_PIPELINE_STAGES����ˮ�߽׶Σ��������N_PIPELINE_STAGES����ˮ�߼Ĵ�����
    m_pipeline_reg.push_back(
        register_set(m_config->pipe_widths[j], pipeline_stage_name_decode[j]));
  }
  for (unsigned j = 0; j < m_config->m_specialized_unit.size(); j++) {
    //����Ϊר�õ�Ԫ����ˮ�߽׶�����ID_OC��ˮ�߼Ĵ�����������V100�����У�û������ר�õ�
    //Ԫ��
    m_pipeline_reg.push_back(
        register_set(m_config->m_specialized_unit[j].id_oc_spec_reg_width,
                     m_config->m_specialized_unit[j].name));
    m_config->m_specialized_unit[j].ID_OC_SPEC_ID = m_pipeline_reg.size() - 1;
    m_specilized_dispatch_reg.push_back(
        &m_pipeline_reg[m_pipeline_reg.size() - 1]);
  }
  for (unsigned j = 0; j < m_config->m_specialized_unit.size(); j++) {
    //����Ϊר�õ�Ԫ����ˮ�߽׶�����OC_EX��ˮ�߼Ĵ�����������V100�����У�û������ר�õ�
    //Ԫ��
    m_pipeline_reg.push_back(
        register_set(m_config->m_specialized_unit[j].oc_ex_spec_reg_width,
                     m_config->m_specialized_unit[j].name));
    m_config->m_specialized_unit[j].OC_EX_SPEC_ID = m_pipeline_reg.size() - 1;
  }
  //��subcoreģʽ�£�ÿ��warp�������ڼĴ�����������һ������ļĴ����ɹ�ʹ�ã������
  //�����ɵ�������m_id������
  if (m_config->sub_core_model) {
    // in subcore model, each scheduler should has its own issue register, so
    // ensure num scheduler = reg width
    //����˵����ÿ��SM�ڵ�warp�������ĸ��������еļĴ������ϵĿ����ͬ��
    assert(m_config->gpgpu_num_sched_per_core ==
           m_pipeline_reg[ID_OC_SP].get_size());
    assert(m_config->gpgpu_num_sched_per_core ==
           m_pipeline_reg[ID_OC_SFU].get_size());
    assert(m_config->gpgpu_num_sched_per_core ==
           m_pipeline_reg[ID_OC_MEM].get_size());
    if (m_config->gpgpu_tensor_core_avail)
      assert(m_config->gpgpu_num_sched_per_core ==
             m_pipeline_reg[ID_OC_TENSOR_CORE].get_size());
    if (m_config->gpgpu_num_dp_units > 0)
      assert(m_config->gpgpu_num_sched_per_core ==
             m_pipeline_reg[ID_OC_DP].get_size());
    if (m_config->gpgpu_num_int_units > 0)
      assert(m_config->gpgpu_num_sched_per_core ==
             m_pipeline_reg[ID_OC_INT].get_size());
    for (unsigned j = 0; j < m_config->m_specialized_unit.size(); j++) {
      if (m_config->m_specialized_unit[j].num_units > 0)
        assert(m_config->gpgpu_num_sched_per_core ==
               m_config->m_specialized_unit[j].id_oc_spec_reg_width);
    }
  }

  //�߳�״̬���ϡ�
  m_threadState = (thread_ctx_t *)calloc(sizeof(thread_ctx_t),
                                         m_config->n_thread_per_shader);
  //δ��ɵ��߳���������Shader Core�ϵ������̶߳����ʱ��==0����
  m_not_completed = 0;
  //m_active_threads��Shader Core�ϻ�Ծ�̵߳�λͼ��
  m_active_threads.reset();
  //m_n_active_cta��Shader Core�ϻ�Ծ���߳̿��������
  m_n_active_cta = 0;
  //m_cta_status��Shader Core�ڵ�CTA��״̬��MAX_CTA_PER_SHADER��ÿ��Shader Core�ڵ�
  //���ɲ���CTA������m_cta_status[i]�ﱣ���˵�i��CTA�а����Ļ�Ծ�߳���������������<=
  //CTA�����߳�������
  for (unsigned i = 0; i < MAX_CTA_PER_SHADER; i++) m_cta_status[i] = 0;
  for (unsigned i = 0; i < m_config->n_thread_per_shader; i++) {
    m_thread[i] = NULL;
    m_threadState[i].m_cta_id = -1;
    m_threadState[i].m_active = false;
  }

  // m_icnt = new shader_memory_interface(this,cluster);
  //��V100�����У�m_config->gpgpu_perfect_memΪfalse��
  if (m_config->gpgpu_perfect_mem) {
    m_icnt = new perfect_memory_interface(this, m_cluster);
  } else {
    m_icnt = new shader_memory_interface(this, m_cluster);
  }
  m_mem_fetch_allocator =
      new shader_core_mem_fetch_allocator(m_sid, m_tpc, m_memory_config);

  // fetch
  m_last_warp_fetched = 0;

#define STRSIZE 1024
  char name[STRSIZE];
  snprintf(name, STRSIZE, "L1I_%03d", m_sid);
  m_L1I = new read_only_cache(name, m_config->m_L1I_config, m_sid,
                              get_shader_instruction_cache_id(), m_icnt,
                              IN_L1I_MISS_QUEUE, OTHER_GPU_CACHE, m_gpu);
}

/*
Shader Core�ڴ���warp������������Sahder Core�ڵ�warp�������ĸ�����gpgpu_num_sched_per_core���ò�
��������Volta�ܹ�ÿ������4��warp��������
*/
void shader_core_ctx::create_schedulers() {
  //����һ���Ƿ��ơ�һ��Shader Core��һ���Ƿ��ơ�
  m_scoreboard = new Scoreboard(m_sid, m_config->max_warps_per_shader, m_gpu);

  // scedulers
  // must currently occur after all inputs have been initialized.
  std::string sched_config = m_config->gpgpu_scheduler_string;
  const concrete_scheduler scheduler =
      sched_config.find("lrr") != std::string::npos ? CONCRETE_SCHEDULER_LRR
          //CONCRETE_SCHEDULER_LRR ��ģ�����е�һ����������scheduler��ѡ��֮һ��LRR ��ʾ ��Least 
          //Recently Reused������Ϊ"�������ʹ��"���õ������㷨�����������ʹ��ԭ�����ڹ���ģ����
          //�е�������ȡ���ģ�����У����ڶ��������ָ������ݷ�������������Ҫ����һ����˳�����
          //�����ִ�С�CONCRETE_SCHEDULER_LRR �㷨�������������ʹ�������ȷ����һ��Ҫ���������
          //����ѡ���������ʹ�õ�����ͨ��ʹ���������ʹ���㷨��CONCRETE_SCHEDULER_LRR ��������
          //�Ը��õ����û�����ڴ�ķ���ģʽ��������������ܡ�ͨ�������������ʹ�õ����󣬿��Լ�����
          //����������еĻ����ͻ�;������Ӷ������ӳٲ����Ч�ʡ�
      : sched_config.find("two_level_active") != std::string::npos
          ? CONCRETE_SCHEDULER_TWO_LEVEL_ACTIVE
      : sched_config.find("gto") != std::string::npos ? CONCRETE_SCHEDULER_GTO
      : sched_config.find("rrr") != std::string::npos ? CONCRETE_SCHEDULER_RRR
      : sched_config.find("old") != std::string::npos
          ? CONCRETE_SCHEDULER_OLDEST_FIRST
      : sched_config.find("warp_limiting") != std::string::npos
          ? CONCRETE_SCHEDULER_WARP_LIMITING
          : NUM_CONCRETE_SCHEDULERS;
  assert(scheduler != NUM_CONCRETE_SCHEDULERS);

  //����Sahder Core�ڵ�warp�������ĸ�����gpgpu_num_sched_per_core���ò���������Volta�ܹ�ÿ������
  //4��warp��������
  for (unsigned i = 0; i < m_config->gpgpu_num_sched_per_core; i++) {
    switch (scheduler) {
      //������������Volta�ܹ��е�����������ΪCONCRETE_SCHEDULER_LRR���������ʹ�á�
      case CONCRETE_SCHEDULER_LRR:
        schedulers.push_back(new lrr_scheduler(
            m_stats, this, m_scoreboard, m_simt_stack, &m_warp,
            &m_pipeline_reg[ID_OC_SP], &m_pipeline_reg[ID_OC_DP],
            &m_pipeline_reg[ID_OC_SFU], &m_pipeline_reg[ID_OC_INT],
            &m_pipeline_reg[ID_OC_TENSOR_CORE], m_specilized_dispatch_reg,
            &m_pipeline_reg[ID_OC_MEM], i));
        break;
      case CONCRETE_SCHEDULER_TWO_LEVEL_ACTIVE:
        schedulers.push_back(new two_level_active_scheduler(
            m_stats, this, m_scoreboard, m_simt_stack, &m_warp,
            &m_pipeline_reg[ID_OC_SP], &m_pipeline_reg[ID_OC_DP],
            &m_pipeline_reg[ID_OC_SFU], &m_pipeline_reg[ID_OC_INT],
            &m_pipeline_reg[ID_OC_TENSOR_CORE], m_specilized_dispatch_reg,
            &m_pipeline_reg[ID_OC_MEM], i, m_config->gpgpu_scheduler_string));
        break;
      case CONCRETE_SCHEDULER_GTO:
        schedulers.push_back(new gto_scheduler(
            m_stats, this, m_scoreboard, m_simt_stack, &m_warp,
            &m_pipeline_reg[ID_OC_SP], &m_pipeline_reg[ID_OC_DP],
            &m_pipeline_reg[ID_OC_SFU], &m_pipeline_reg[ID_OC_INT],
            &m_pipeline_reg[ID_OC_TENSOR_CORE], m_specilized_dispatch_reg,
            &m_pipeline_reg[ID_OC_MEM], i));
        break;
      case CONCRETE_SCHEDULER_RRR:
        schedulers.push_back(new rrr_scheduler(
            m_stats, this, m_scoreboard, m_simt_stack, &m_warp,
            &m_pipeline_reg[ID_OC_SP], &m_pipeline_reg[ID_OC_DP],
            &m_pipeline_reg[ID_OC_SFU], &m_pipeline_reg[ID_OC_INT],
            &m_pipeline_reg[ID_OC_TENSOR_CORE], m_specilized_dispatch_reg,
            &m_pipeline_reg[ID_OC_MEM], i));
        break;
      case CONCRETE_SCHEDULER_OLDEST_FIRST:
        schedulers.push_back(new oldest_scheduler(
            m_stats, this, m_scoreboard, m_simt_stack, &m_warp,
            &m_pipeline_reg[ID_OC_SP], &m_pipeline_reg[ID_OC_DP],
            &m_pipeline_reg[ID_OC_SFU], &m_pipeline_reg[ID_OC_INT],
            &m_pipeline_reg[ID_OC_TENSOR_CORE], m_specilized_dispatch_reg,
            &m_pipeline_reg[ID_OC_MEM], i));
        break;
      case CONCRETE_SCHEDULER_WARP_LIMITING:
        schedulers.push_back(new swl_scheduler(
            m_stats, this, m_scoreboard, m_simt_stack, &m_warp,
            &m_pipeline_reg[ID_OC_SP], &m_pipeline_reg[ID_OC_DP],
            &m_pipeline_reg[ID_OC_SFU], &m_pipeline_reg[ID_OC_INT],
            &m_pipeline_reg[ID_OC_TENSOR_CORE], m_specilized_dispatch_reg,
            &m_pipeline_reg[ID_OC_MEM], i, m_config->gpgpu_scheduler_string));
        break;
      default:
        abort();
    };
  }

  //����m_warp�ǻ��ֵ���ǰSM������warp���ϣ��䶨��Ϊ��
  //    std::vector<shd_warp_t *> *m_warp;
  //m_warp[i]�ǻ��ֵ���ǰSM�ĵ�i��warp����δ�����ʵ�ǽ�m_warp�е�����warp���ָ�ÿ������������
  //��ÿ���������Ϳ��ԶԻ��ָ��Լ���warp���е����ˡ���Volta�ܹ��ϣ�ÿ������4��warp�����������ֵ�
  //�����ǣ�warp 0->������0��warp 1->������1��warp 2->������2��warp 3->������3��warp 4->����
  //��0���Դ����ơ���ÿ���������ڲ���һ��ר�Ŵ洢���������ֵ���warp���б���m_supervised_warps��
  //ÿ����������������δ����ｫ����Ϊ�Լ���warp���뵽�Լ���m_supervised_warps�С����б���Ϊ��
  //    std::vector<shd_warp_t *> m_supervised_warps;
  //m_supervisored_twarps�б��Ǵ˵��ȳ���Ӧ�����������ٲõ�����warps�����ڴ��ڶ��warp������
  //��ϵͳ�зǳ����á��ڵ���������ϵͳ�У���ֻ�Ƿ�����ú��ĵ�����warp����������������Ҫ���֣���
  for (unsigned i = 0; i < m_warp.size(); i++) {
    // distribute i's evenly though schedulers;
    //m_supervisored_twarps�б��Ǵ˵��ȳ���Ӧ�����������ٲõ�����warps�����ڴ��ڶ��warp����
    //����ϵͳ�зǳ����á��ڵ���������ϵͳ�У���ֻ�Ƿ�����ú��ĵ�����warp��
    schedulers[i % m_config->gpgpu_num_sched_per_core]->add_supervised_warp_id(
        i);
  }
  
  //done_adding_supervised_warps()�����Ķ���Ϊ��
  //    virtual void done_adding_supervised_warps() {
  //      m_last_supervised_issued = m_supervised_warps.end();
  //    }
  //������ʵ���Ƕ�ÿ����������m_last_supervised_issued���г�ʼ����m_last_supervised_issued��
  //ָ����һ�ε��ȵ�warp���������ʼ��Ϊm_supervised_warps.end()����m_supervised_warps�����
  //һ��m_supervised_warps�е�warp��
  for (unsigned i = 0; i < m_config->gpgpu_num_sched_per_core; ++i) {
    schedulers[i]->done_adding_supervised_warps();
  }
}

void shader_core_ctx::create_exec_pipeline() {
  // op collector configuration
  enum { SP_CUS, DP_CUS, SFU_CUS, TENSOR_CORE_CUS, INT_CUS, MEM_CUS, GEN_CUS };

  opndcoll_rfu_t::port_vector_t in_ports;
  opndcoll_rfu_t::port_vector_t out_ports;
  opndcoll_rfu_t::uint_vector_t cu_sets;

  // configure generic collectors
  m_operand_collector.add_cu_set(
      GEN_CUS, m_config->gpgpu_operand_collector_num_units_gen,
      m_config->gpgpu_operand_collector_num_out_ports_gen);

  for (unsigned i = 0; i < m_config->gpgpu_operand_collector_num_in_ports_gen;
       i++) {
    in_ports.push_back(&m_pipeline_reg[ID_OC_SP]);
    in_ports.push_back(&m_pipeline_reg[ID_OC_SFU]);
    in_ports.push_back(&m_pipeline_reg[ID_OC_MEM]);
    out_ports.push_back(&m_pipeline_reg[OC_EX_SP]);
    out_ports.push_back(&m_pipeline_reg[OC_EX_SFU]);
    out_ports.push_back(&m_pipeline_reg[OC_EX_MEM]);
    if (m_config->gpgpu_tensor_core_avail) {
      in_ports.push_back(&m_pipeline_reg[ID_OC_TENSOR_CORE]);
      out_ports.push_back(&m_pipeline_reg[OC_EX_TENSOR_CORE]);
    }
    if (m_config->gpgpu_num_dp_units > 0) {
      in_ports.push_back(&m_pipeline_reg[ID_OC_DP]);
      out_ports.push_back(&m_pipeline_reg[OC_EX_DP]);
    }
    if (m_config->gpgpu_num_int_units > 0) {
      in_ports.push_back(&m_pipeline_reg[ID_OC_INT]);
      out_ports.push_back(&m_pipeline_reg[OC_EX_INT]);
    }
    if (m_config->m_specialized_unit.size() > 0) {
      for (unsigned j = 0; j < m_config->m_specialized_unit.size(); ++j) {
        in_ports.push_back(
            &m_pipeline_reg[m_config->m_specialized_unit[j].ID_OC_SPEC_ID]);
        out_ports.push_back(
            &m_pipeline_reg[m_config->m_specialized_unit[j].OC_EX_SPEC_ID]);
      }
    }
    cu_sets.push_back((unsigned)GEN_CUS);
    m_operand_collector.add_port(in_ports, out_ports, cu_sets);
    in_ports.clear(), out_ports.clear(), cu_sets.clear();
  }

  if (m_config->enable_specialized_operand_collector) {
    m_operand_collector.add_cu_set(
        SP_CUS, m_config->gpgpu_operand_collector_num_units_sp,
        m_config->gpgpu_operand_collector_num_out_ports_sp);
    m_operand_collector.add_cu_set(
        DP_CUS, m_config->gpgpu_operand_collector_num_units_dp,
        m_config->gpgpu_operand_collector_num_out_ports_dp);
    m_operand_collector.add_cu_set(
        TENSOR_CORE_CUS,
        m_config->gpgpu_operand_collector_num_units_tensor_core,
        m_config->gpgpu_operand_collector_num_out_ports_tensor_core);
    m_operand_collector.add_cu_set(
        SFU_CUS, m_config->gpgpu_operand_collector_num_units_sfu,
        m_config->gpgpu_operand_collector_num_out_ports_sfu);
    m_operand_collector.add_cu_set(
        MEM_CUS, m_config->gpgpu_operand_collector_num_units_mem,
        m_config->gpgpu_operand_collector_num_out_ports_mem);
    m_operand_collector.add_cu_set(
        INT_CUS, m_config->gpgpu_operand_collector_num_units_int,
        m_config->gpgpu_operand_collector_num_out_ports_int);
    //gpgpu_operand_collector_num_in_ports_sp��SP��Ԫ����������ռ���������˿���������ǰ��
    //��warp�����������ﵥ��Sahder Core�ڵ�warp�������ĸ�����gpgpu_num_sched_per_core���ò�
    //��������Volta�ܹ�ÿ������4��warp��������ÿ���������Ĵ������룺
    //     schedulers.push_back(new lrr_scheduler(
    //             m_stats, this, m_scoreboard, m_simt_stack, &m_warp,
    //             &m_pipeline_reg[ID_OC_SP], &m_pipeline_reg[ID_OC_DP],
    //             &m_pipeline_reg[ID_OC_SFU], &m_pipeline_reg[ID_OC_INT],
    //             &m_pipeline_reg[ID_OC_TENSOR_CORE], m_specilized_dispatch_reg,
    //             &m_pipeline_reg[ID_OC_MEM], i));
    //�ڷ�������У�warp���������ɷ����ָ�����ָ�����ͷַ�����ͬ�ĵ�Ԫ����Щ��Ԫ����SP/DP/
    //SFU/INT/TENSOR_CORE/MEM���ڷ��������ɺ���Ҫ���ָ��ͨ���������ռ�����ָ������Ĳ���
    //��ȫ���ռ��롣����һ��SM����Ӧ��һ���������ռ������������ķ�����̽�ָ����룺
    //    m_pipeline_reg[ID_OC_SP]��m_pipeline_reg[ID_OC_DP]��m_pipeline_reg[ID_OC_SFU]��
    //    m_pipeline_reg[ID_OC_INT]��m_pipeline_reg[ID_OC_TENSOR_CORE]��
    //    m_pipeline_reg[ID_OC_MEM]
    //�ȼĴ��������У����Բ������ռ������ռ���������
    for (unsigned i = 0; i < m_config->gpgpu_operand_collector_num_in_ports_sp;
         i++) {
      //m_pipeline_reg�Ķ��壺std::vector<register_set> m_pipeline_reg;
      in_ports.push_back(&m_pipeline_reg[ID_OC_SP]);
      out_ports.push_back(&m_pipeline_reg[OC_EX_SP]);
      cu_sets.push_back((unsigned)SP_CUS);
      cu_sets.push_back((unsigned)GEN_CUS);
      m_operand_collector.add_port(in_ports, out_ports, cu_sets);
      in_ports.clear(), out_ports.clear(), cu_sets.clear();
    }
    //gpgpu_operand_collector_num_in_ports_dp��DP��Ԫ����������ռ���������˿�������
    for (unsigned i = 0; i < m_config->gpgpu_operand_collector_num_in_ports_dp;
         i++) {
      in_ports.push_back(&m_pipeline_reg[ID_OC_DP]);
      out_ports.push_back(&m_pipeline_reg[OC_EX_DP]);
      cu_sets.push_back((unsigned)DP_CUS);
      cu_sets.push_back((unsigned)GEN_CUS);
      m_operand_collector.add_port(in_ports, out_ports, cu_sets);
      in_ports.clear(), out_ports.clear(), cu_sets.clear();
    }
    //gpgpu_operand_collector_num_in_ports_sfu��SFU��Ԫ����������ռ���������˿�������
    for (unsigned i = 0; i < m_config->gpgpu_operand_collector_num_in_ports_sfu;
         i++) {
      in_ports.push_back(&m_pipeline_reg[ID_OC_SFU]);
      out_ports.push_back(&m_pipeline_reg[OC_EX_SFU]);
      cu_sets.push_back((unsigned)SFU_CUS);
      cu_sets.push_back((unsigned)GEN_CUS);
      m_operand_collector.add_port(in_ports, out_ports, cu_sets);
      in_ports.clear(), out_ports.clear(), cu_sets.clear();
    }
    //gpgpu_operand_collector_num_in_ports_tensor_core��TC��Ԫ����������ռ���������˿�������
    for (unsigned i = 0;
         i < m_config->gpgpu_operand_collector_num_in_ports_tensor_core; i++) {
      in_ports.push_back(&m_pipeline_reg[ID_OC_TENSOR_CORE]);
      out_ports.push_back(&m_pipeline_reg[OC_EX_TENSOR_CORE]);
      cu_sets.push_back((unsigned)TENSOR_CORE_CUS);
      cu_sets.push_back((unsigned)GEN_CUS);
      m_operand_collector.add_port(in_ports, out_ports, cu_sets);
      in_ports.clear(), out_ports.clear(), cu_sets.clear();
    }
    //gpgpu_operand_collector_num_in_ports_mem��MEM��Ԫ����������ռ���������˿�������
    for (unsigned i = 0; i < m_config->gpgpu_operand_collector_num_in_ports_mem;
         i++) {
      in_ports.push_back(&m_pipeline_reg[ID_OC_MEM]);
      out_ports.push_back(&m_pipeline_reg[OC_EX_MEM]);
      cu_sets.push_back((unsigned)MEM_CUS);
      cu_sets.push_back((unsigned)GEN_CUS);
      m_operand_collector.add_port(in_ports, out_ports, cu_sets);
      in_ports.clear(), out_ports.clear(), cu_sets.clear();
    }
    //gpgpu_operand_collector_num_in_ports_int��INT��Ԫ����������ռ���������˿�������
    for (unsigned i = 0; i < m_config->gpgpu_operand_collector_num_in_ports_int;
         i++) {
      in_ports.push_back(&m_pipeline_reg[ID_OC_INT]);
      out_ports.push_back(&m_pipeline_reg[OC_EX_INT]);
      cu_sets.push_back((unsigned)INT_CUS);
      cu_sets.push_back((unsigned)GEN_CUS);
      m_operand_collector.add_port(in_ports, out_ports, cu_sets);
      in_ports.clear(), out_ports.clear(), cu_sets.clear();
    }
  }

  //ִ�в������ռ����ĳ�ʼ����
  m_operand_collector.init(m_config->gpgpu_num_reg_banks, this);

  m_num_function_units =
      m_config->gpgpu_num_sp_units + m_config->gpgpu_num_dp_units +
      m_config->gpgpu_num_sfu_units + m_config->gpgpu_num_tensor_core_units +
      m_config->gpgpu_num_int_units + m_config->m_specialized_unit_num +
      1;  // sp_unit, sfu, dp, tensor, int, ldst_unit
  // m_dispatch_port = new enum pipeline_stage_name_t[ m_num_function_units ];
  // m_issue_port = new enum pipeline_stage_name_t[ m_num_function_units ];

  // m_fu = new simd_function_unit*[m_num_function_units];

  for (unsigned k = 0; k < m_config->gpgpu_num_sp_units; k++) {
    m_fu.push_back(new sp_unit(&m_pipeline_reg[EX_WB], m_config, this, k));
    m_dispatch_port.push_back(ID_OC_SP);
    m_issue_port.push_back(OC_EX_SP);
  }

  for (unsigned k = 0; k < m_config->gpgpu_num_dp_units; k++) {
    m_fu.push_back(new dp_unit(&m_pipeline_reg[EX_WB], m_config, this, k));
    m_dispatch_port.push_back(ID_OC_DP);
    m_issue_port.push_back(OC_EX_DP);
  }
  for (unsigned k = 0; k < m_config->gpgpu_num_int_units; k++) {
    m_fu.push_back(new int_unit(&m_pipeline_reg[EX_WB], m_config, this, k));
    m_dispatch_port.push_back(ID_OC_INT);
    m_issue_port.push_back(OC_EX_INT);
  }

  for (unsigned k = 0; k < m_config->gpgpu_num_sfu_units; k++) {
    m_fu.push_back(new sfu(&m_pipeline_reg[EX_WB], m_config, this, k));
    m_dispatch_port.push_back(ID_OC_SFU);
    m_issue_port.push_back(OC_EX_SFU);
  }

  for (unsigned k = 0; k < m_config->gpgpu_num_tensor_core_units; k++) {
    m_fu.push_back(new tensor_core(&m_pipeline_reg[EX_WB], m_config, this, k));
    m_dispatch_port.push_back(ID_OC_TENSOR_CORE);
    m_issue_port.push_back(OC_EX_TENSOR_CORE);
  }

  for (unsigned j = 0; j < m_config->m_specialized_unit.size(); j++) {
    for (unsigned k = 0; k < m_config->m_specialized_unit[j].num_units; k++) {
      m_fu.push_back(new specialized_unit(
          &m_pipeline_reg[EX_WB], m_config, this, SPEC_UNIT_START_ID + j,
          m_config->m_specialized_unit[j].name,
          m_config->m_specialized_unit[j].latency, k));
      m_dispatch_port.push_back(m_config->m_specialized_unit[j].ID_OC_SPEC_ID);
      m_issue_port.push_back(m_config->m_specialized_unit[j].OC_EX_SPEC_ID);
    }
  }

  m_ldst_unit = new ldst_unit(m_icnt, m_mem_fetch_allocator, this,
                              &m_operand_collector, m_scoreboard, m_config,
                              m_memory_config, m_stats, m_sid, m_tpc, m_gpu);
  m_fu.push_back(m_ldst_unit);
  m_dispatch_port.push_back(ID_OC_MEM);
  m_issue_port.push_back(OC_EX_MEM);

  assert(m_num_function_units == m_fu.size() and
         m_fu.size() == m_dispatch_port.size() and
         m_fu.size() == m_issue_port.size());

  // there are as many result buses as the width of the EX_WB stage
  //������߹���m_config->pipe_widths[EX_WB]����
  //��ˮ�߽׶εĿ��������-gpgpu_pipeline_widths�����ã�
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
  //��V100������Ϊ��-gpgpu_pipeline_widths 4,4,4,4,4,4,4,4,4,4,8,4,4
  //������ߵĿ����8����m_config->pipe_widths[EX_WB] = 8��
  num_result_bus = m_config->pipe_widths[EX_WB];
  for (unsigned i = 0; i < num_result_bus; i++) {
    this->m_result_bus.push_back(new std::bitset<MAX_ALU_LATENCY>());
  }
}

shader_core_ctx::shader_core_ctx(class gpgpu_sim *gpu,
                                 class simt_core_cluster *cluster,
                                 unsigned shader_id, unsigned tpc_id,
                                 const shader_core_config *config,
                                 const memory_config *mem_config,
                                 shader_core_stats *stats)
    : core_t(gpu, NULL, config->warp_size, config->n_thread_per_shader),
      m_barriers(this, config->max_warps_per_shader, config->max_cta_per_core,
                 config->max_barriers_per_cta, config->warp_size),
      m_active_warps(0),
      m_dynamic_warp_id(0) {
  m_cluster = cluster;
  m_config = config;
  m_memory_config = mem_config;
  m_stats = stats;
  // unsigned warp_size = config->warp_size;
  Issue_Prio = 0;

  m_sid = shader_id;
  m_tpc = tpc_id;

  if (get_gpu()->get_config().g_power_simulation_enabled) {
    scaling_coeffs = get_gpu()->get_scaling_coeffs();
  }

  m_last_inst_gpu_sim_cycle = 0;
  m_last_inst_gpu_tot_sim_cycle = 0;

  // Jin: for concurrent kernels on a SM
  m_occupied_n_threads = 0;
  m_occupied_shmem = 0;
  m_occupied_regs = 0;
  m_occupied_ctas = 0;
  m_occupied_hwtid.reset();
  m_occupied_cta_to_hwtid.clear();
}

void shader_core_ctx::reinit(unsigned start_thread, unsigned end_thread,
                             bool reset_not_completed) {
  if (reset_not_completed) {
    m_not_completed = 0;
    m_active_threads.reset();

    // Jin: for concurrent kernels on a SM
    m_occupied_n_threads = 0;
    m_occupied_shmem = 0;
    m_occupied_regs = 0;
    m_occupied_ctas = 0;
    m_occupied_hwtid.reset();
    m_occupied_cta_to_hwtid.clear();
    m_active_warps = 0;
  }
  for (unsigned i = start_thread; i < end_thread; i++) {
    m_threadState[i].n_insn = 0;
    m_threadState[i].m_cta_id = -1;
  }
  for (unsigned i = start_thread / m_config->warp_size;
       i < end_thread / m_config->warp_size; ++i) {
    m_warp[i]->reset();
    m_simt_stack[i]->reset();
  }
}

/*
�Ե�cta_id��CTA�У���start_thread��end_thread���߳�����������warp���г�ʼ����
*/
void shader_core_ctx::init_warps(unsigned cta_id, unsigned start_thread,
                                 unsigned end_thread, unsigned ctaid,
                                 int cta_size, kernel_info_t &kernel) {
  //����ģ������У���warp����ʼPCֵ���ø�warp���׸��߳�m_thread[warpId*m_warp_size]->get_pc()��ȡ��
  //�̺߳����߳�������������SIMT��ջ��ÿ��warp����һ��������SIMT��ջ��
  address_type start_pc = next_pc(start_thread);
  unsigned kernel_id = kernel.get_uid();
  if (m_config->model == POST_DOMINATOR) {
    unsigned start_warp = start_thread / m_config->warp_size;
    unsigned warp_per_cta = cta_size / m_config->warp_size;
    unsigned end_warp = end_thread / m_config->warp_size +
                        ((end_thread % m_config->warp_size) ? 1 : 0);
    //��start_warp��ʼ��end_warpѭ����
    for (unsigned i = start_warp; i < end_warp; ++i) {
      //n_active������warp�еĻ�Ծ�̵߳ĸ�����
      unsigned n_active = 0;
      simt_mask_t active_threads;
      //�ֲ��߳�ID����0��31ѭ����
      for (unsigned t = 0; t < m_config->warp_size; t++) {
        //Ӳ���ϵ��߳�ID����ȫ��ID=warp_id*32+�ֲ�ID��
        unsigned hwtid = i * m_config->warp_size + t;
        if (hwtid < end_thread) {
          n_active++;
          assert(!m_active_threads.test(hwtid));
          //����ȫ�ֻ�Ծ�̵߳�λͼ��
          m_active_threads.set(hwtid);
          //���þֲ���Ծ�̵߳�λͼ��
          active_threads.set(t);
        }
      }
      //����ģ������У���warp����ʼPCֵ���ø�warp���׸��߳�m_thread[warpId*m_warp_size]->get_pc()��
      //ȡ���̺߳����߳�������������SIMT��ջ��ÿ��warp����һ��������SIMT��ջ��
      m_simt_stack[i]->launch(start_pc, active_threads);

      //��V100�����У�m_gpu->resume_option��Ĭ������Ϊ0��
      if (m_gpu->resume_option == 1 && kernel_id == m_gpu->resume_kernel &&
          ctaid >= m_gpu->resume_CTA && ctaid < m_gpu->checkpoint_CTA_t) {
        char fname[2048];
        snprintf(fname, 2048, "checkpoint_files/warp_%d_%d_simt.txt",
                 i % warp_per_cta, ctaid);
        unsigned pc, rpc;
        m_simt_stack[i]->resume(fname);
        m_simt_stack[i]->get_pdom_stack_top_info(&pc, &rpc);
        for (unsigned t = 0; t < m_config->warp_size; t++) {
          if (m_thread != NULL) {
            m_thread[i * m_config->warp_size + t]->set_npc(pc);
            m_thread[i * m_config->warp_size + t]->update_pc();
          }
        }
        start_pc = pc;
      }

      //��ȫ�ֵ�i��warp��m_warp[i]���г�ʼ����
      m_warp[i]->init(start_pc, cta_id, i, active_threads, m_dynamic_warp_id,
                      kernel.get_streamID());
      //��һ����̬warp��ID��ÿ��һ��warp��ɻ�Ծ״̬����1��
	  ++m_dynamic_warp_id;
      //m_not_completed��ֵʵ�����Ƿ��ص����Ѿ���ʼ��������warp��������CTA������δ��ɵ��߳�����
	  m_not_completed += n_active;
      //��Ծwarp��������1��
	  ++m_active_warps;
    }
  }
}

// return the next pc of a thread
address_type shader_core_ctx::next_pc(int tid) const {
  if (tid == -1) return -1;
  ptx_thread_info *the_thread = m_thread[tid];
  if (the_thread == NULL) return -1;
  return the_thread
      ->get_pc();  // PC should already be updatd to next PC at this point (was
                   // set in shader_decode() last time thread ran)
}

void gpgpu_sim::get_pdom_stack_top_info(unsigned sid, unsigned tid,
                                        unsigned *pc, unsigned *rpc) {
  unsigned cluster_id = m_shader_config->sid_to_cluster(sid);
  m_cluster[cluster_id]->get_pdom_stack_top_info(sid, tid, pc, rpc);
}

void shader_core_ctx::get_pdom_stack_top_info(unsigned tid, unsigned *pc,
                                              unsigned *rpc) const {
  unsigned warp_id = tid / m_config->warp_size;
  m_simt_stack[warp_id]->get_pdom_stack_top_info(pc, rpc);
}

float shader_core_ctx::get_current_occupancy(unsigned long long &active,
                                             unsigned long long &total) const {
  // To match the achieved_occupancy in nvprof, only SMs that are active are
  // counted toward the occupancy.
  if (m_active_warps > 0) {
    total += m_warp.size();
    active += m_active_warps;
    return float(active) / float(total);
  } else {
    return 0;
  }
}

void shader_core_stats::print(FILE *fout) const {
  unsigned long long thread_icount_uarch = 0;
  unsigned long long warp_icount_uarch = 0;

  for (unsigned i = 0; i < m_config->num_shader(); i++) {
    thread_icount_uarch += m_num_sim_insn[i];
    warp_icount_uarch += m_num_sim_winsn[i];
  }
  fprintf(fout, "gpgpu_n_tot_thrd_icount = %lld\n", thread_icount_uarch);
  fprintf(fout, "gpgpu_n_tot_w_icount = %lld\n", warp_icount_uarch);

  fprintf(fout, "gpgpu_n_stall_shd_mem = %d\n", gpgpu_n_stall_shd_mem);
  fprintf(fout, "gpgpu_n_mem_read_local = %d\n", gpgpu_n_mem_read_local);
  fprintf(fout, "gpgpu_n_mem_write_local = %d\n", gpgpu_n_mem_write_local);
  fprintf(fout, "gpgpu_n_mem_read_global = %d\n", gpgpu_n_mem_read_global);
  fprintf(fout, "gpgpu_n_mem_write_global = %d\n", gpgpu_n_mem_write_global);
  fprintf(fout, "gpgpu_n_mem_texture = %d\n", gpgpu_n_mem_texture);
  fprintf(fout, "gpgpu_n_mem_const = %d\n", gpgpu_n_mem_const);

  fprintf(fout, "gpgpu_n_load_insn  = %d\n", gpgpu_n_load_insn);
  fprintf(fout, "gpgpu_n_store_insn = %d\n", gpgpu_n_store_insn);
  fprintf(fout, "gpgpu_n_shmem_insn = %d\n", gpgpu_n_shmem_insn);
  fprintf(fout, "gpgpu_n_sstarr_insn = %d\n", gpgpu_n_sstarr_insn);
  fprintf(fout, "gpgpu_n_tex_insn = %d\n", gpgpu_n_tex_insn);
  fprintf(fout, "gpgpu_n_const_mem_insn = %d\n", gpgpu_n_const_insn);
  fprintf(fout, "gpgpu_n_param_mem_insn = %d\n", gpgpu_n_param_insn);

  fprintf(fout, "gpgpu_n_shmem_bkconflict = %d\n", gpgpu_n_shmem_bkconflict);
  fprintf(fout, "gpgpu_n_l1cache_bkconflict = %d\n",
          gpgpu_n_l1cache_bkconflict);

  fprintf(fout, "gpgpu_n_intrawarp_mshr_merge = %d\n",
          gpgpu_n_intrawarp_mshr_merge);
  fprintf(fout, "gpgpu_n_cmem_portconflict = %d\n", gpgpu_n_cmem_portconflict);

  fprintf(fout, "gpgpu_stall_shd_mem[c_mem][resource_stall] = %d\n",
          gpu_stall_shd_mem_breakdown[C_MEM][BK_CONF]);
  // fprintf(fout, "gpgpu_stall_shd_mem[c_mem][mshr_rc] = %d\n",
  // gpu_stall_shd_mem_breakdown[C_MEM][MSHR_RC_FAIL]); fprintf(fout,
  // "gpgpu_stall_shd_mem[c_mem][icnt_rc] = %d\n",
  // gpu_stall_shd_mem_breakdown[C_MEM][ICNT_RC_FAIL]); fprintf(fout,
  // "gpgpu_stall_shd_mem[c_mem][data_port_stall] = %d\n",
  // gpu_stall_shd_mem_breakdown[C_MEM][DATA_PORT_STALL]); fprintf(fout,
  // "gpgpu_stall_shd_mem[t_mem][mshr_rc] = %d\n",
  // gpu_stall_shd_mem_breakdown[T_MEM][MSHR_RC_FAIL]); fprintf(fout,
  // "gpgpu_stall_shd_mem[t_mem][icnt_rc] = %d\n",
  // gpu_stall_shd_mem_breakdown[T_MEM][ICNT_RC_FAIL]); fprintf(fout,
  // "gpgpu_stall_shd_mem[t_mem][data_port_stall] = %d\n",
  // gpu_stall_shd_mem_breakdown[T_MEM][DATA_PORT_STALL]);
  fprintf(fout, "gpgpu_stall_shd_mem[s_mem][bk_conf] = %d\n",
          gpu_stall_shd_mem_breakdown[S_MEM][BK_CONF]);
  fprintf(
      fout, "gpgpu_stall_shd_mem[gl_mem][resource_stall] = %d\n",
      gpu_stall_shd_mem_breakdown[G_MEM_LD][BK_CONF] +
          gpu_stall_shd_mem_breakdown[G_MEM_ST][BK_CONF] +
          gpu_stall_shd_mem_breakdown[L_MEM_LD][BK_CONF] +
          gpu_stall_shd_mem_breakdown[L_MEM_ST][BK_CONF]);  // coalescing stall
                                                            // at data cache
  fprintf(
      fout, "gpgpu_stall_shd_mem[gl_mem][coal_stall] = %d\n",
      gpu_stall_shd_mem_breakdown[G_MEM_LD][COAL_STALL] +
          gpu_stall_shd_mem_breakdown[G_MEM_ST][COAL_STALL] +
          gpu_stall_shd_mem_breakdown[L_MEM_LD][COAL_STALL] +
          gpu_stall_shd_mem_breakdown[L_MEM_ST]
                                     [COAL_STALL]);  // coalescing stall + bank
                                                     // conflict at data cache
  fprintf(fout, "gpgpu_stall_shd_mem[gl_mem][data_port_stall] = %d\n",
          gpu_stall_shd_mem_breakdown[G_MEM_LD][DATA_PORT_STALL] +
              gpu_stall_shd_mem_breakdown[G_MEM_ST][DATA_PORT_STALL] +
              gpu_stall_shd_mem_breakdown[L_MEM_LD][DATA_PORT_STALL] +
              gpu_stall_shd_mem_breakdown[L_MEM_ST]
                                         [DATA_PORT_STALL]);  // data port stall
                                                              // at data cache
  // fprintf(fout, "gpgpu_stall_shd_mem[g_mem_ld][mshr_rc] = %d\n",
  // gpu_stall_shd_mem_breakdown[G_MEM_LD][MSHR_RC_FAIL]); fprintf(fout,
  // "gpgpu_stall_shd_mem[g_mem_ld][icnt_rc] = %d\n",
  // gpu_stall_shd_mem_breakdown[G_MEM_LD][ICNT_RC_FAIL]); fprintf(fout,
  // "gpgpu_stall_shd_mem[g_mem_ld][wb_icnt_rc] = %d\n",
  // gpu_stall_shd_mem_breakdown[G_MEM_LD][WB_ICNT_RC_FAIL]); fprintf(fout,
  // "gpgpu_stall_shd_mem[g_mem_ld][wb_rsrv_fail] = %d\n",
  // gpu_stall_shd_mem_breakdown[G_MEM_LD][WB_CACHE_RSRV_FAIL]); fprintf(fout,
  // "gpgpu_stall_shd_mem[g_mem_st][mshr_rc] = %d\n",
  // gpu_stall_shd_mem_breakdown[G_MEM_ST][MSHR_RC_FAIL]); fprintf(fout,
  // "gpgpu_stall_shd_mem[g_mem_st][icnt_rc] = %d\n",
  // gpu_stall_shd_mem_breakdown[G_MEM_ST][ICNT_RC_FAIL]); fprintf(fout,
  // "gpgpu_stall_shd_mem[g_mem_st][wb_icnt_rc] = %d\n",
  // gpu_stall_shd_mem_breakdown[G_MEM_ST][WB_ICNT_RC_FAIL]); fprintf(fout,
  // "gpgpu_stall_shd_mem[g_mem_st][wb_rsrv_fail] = %d\n",
  // gpu_stall_shd_mem_breakdown[G_MEM_ST][WB_CACHE_RSRV_FAIL]); fprintf(fout,
  // "gpgpu_stall_shd_mem[l_mem_ld][mshr_rc] = %d\n",
  // gpu_stall_shd_mem_breakdown[L_MEM_LD][MSHR_RC_FAIL]); fprintf(fout,
  // "gpgpu_stall_shd_mem[l_mem_ld][icnt_rc] = %d\n",
  // gpu_stall_shd_mem_breakdown[L_MEM_LD][ICNT_RC_FAIL]); fprintf(fout,
  // "gpgpu_stall_shd_mem[l_mem_ld][wb_icnt_rc] = %d\n",
  // gpu_stall_shd_mem_breakdown[L_MEM_LD][WB_ICNT_RC_FAIL]); fprintf(fout,
  // "gpgpu_stall_shd_mem[l_mem_ld][wb_rsrv_fail] = %d\n",
  // gpu_stall_shd_mem_breakdown[L_MEM_LD][WB_CACHE_RSRV_FAIL]); fprintf(fout,
  // "gpgpu_stall_shd_mem[l_mem_st][mshr_rc] = %d\n",
  // gpu_stall_shd_mem_breakdown[L_MEM_ST][MSHR_RC_FAIL]); fprintf(fout,
  // "gpgpu_stall_shd_mem[l_mem_st][icnt_rc] = %d\n",
  // gpu_stall_shd_mem_breakdown[L_MEM_ST][ICNT_RC_FAIL]); fprintf(fout,
  // "gpgpu_stall_shd_mem[l_mem_ld][wb_icnt_rc] = %d\n",
  // gpu_stall_shd_mem_breakdown[L_MEM_ST][WB_ICNT_RC_FAIL]); fprintf(fout,
  // "gpgpu_stall_shd_mem[l_mem_ld][wb_rsrv_fail] = %d\n",
  // gpu_stall_shd_mem_breakdown[L_MEM_ST][WB_CACHE_RSRV_FAIL]);

  fprintf(fout, "gpu_reg_bank_conflict_stalls = %d\n",
          gpu_reg_bank_conflict_stalls);

  fprintf(fout, "Warp Occupancy Distribution:\n");
  fprintf(fout, "Stall:%d\t", shader_cycle_distro[2]);
  fprintf(fout, "W0_Idle:%d\t", shader_cycle_distro[0]);
  fprintf(fout, "W0_Scoreboard:%d", shader_cycle_distro[1]);
  for (unsigned i = 3; i < m_config->warp_size + 3; i++)
    fprintf(fout, "\tW%d:%d", i - 2, shader_cycle_distro[i]);
  fprintf(fout, "\n");
  fprintf(fout, "single_issue_nums: ");
  for (unsigned i = 0; i < m_config->gpgpu_num_sched_per_core; i++)
    fprintf(fout, "WS%d:%d\t", i, single_issue_nums[i]);
  fprintf(fout, "\n");
  fprintf(fout, "dual_issue_nums: ");
  for (unsigned i = 0; i < m_config->gpgpu_num_sched_per_core; i++)
    fprintf(fout, "WS%d:%d\t", i, dual_issue_nums[i]);
  fprintf(fout, "\n");

  m_outgoing_traffic_stats->print(fout);
  m_incoming_traffic_stats->print(fout);
}

void shader_core_stats::event_warp_issued(unsigned s_id, unsigned warp_id,
                                          unsigned num_issued,
                                          unsigned dynamic_warp_id) {
  assert(warp_id <= m_config->max_warps_per_shader);
  for (unsigned i = 0; i < num_issued; ++i) {
    if (m_shader_dynamic_warp_issue_distro[s_id].size() <= dynamic_warp_id) {
      m_shader_dynamic_warp_issue_distro[s_id].resize(dynamic_warp_id + 1);
    }
    ++m_shader_dynamic_warp_issue_distro[s_id][dynamic_warp_id];
    if (m_shader_warp_slot_issue_distro[s_id].size() <= warp_id) {
      m_shader_warp_slot_issue_distro[s_id].resize(warp_id + 1);
    }
    ++m_shader_warp_slot_issue_distro[s_id][warp_id];
  }
}

void shader_core_stats::visualizer_print(gzFile visualizer_file) {
  // warp divergence breakdown
  gzprintf(visualizer_file, "WarpDivergenceBreakdown:");
  unsigned int total = 0;
  unsigned int cf =
      (m_config->gpgpu_warpdistro_shader == -1) ? m_config->num_shader() : 1;
  gzprintf(visualizer_file, " %d",
           (shader_cycle_distro[0] - last_shader_cycle_distro[0]) / cf);
  gzprintf(visualizer_file, " %d",
           (shader_cycle_distro[1] - last_shader_cycle_distro[1]) / cf);
  gzprintf(visualizer_file, " %d",
           (shader_cycle_distro[2] - last_shader_cycle_distro[2]) / cf);
  for (unsigned i = 0; i < m_config->warp_size + 3; i++) {
    if (i >= 3) {
      total += (shader_cycle_distro[i] - last_shader_cycle_distro[i]);
      if (((i - 3) % (m_config->warp_size / 8)) ==
          ((m_config->warp_size / 8) - 1)) {
        gzprintf(visualizer_file, " %d", total / cf);
        total = 0;
      }
    }
    last_shader_cycle_distro[i] = shader_cycle_distro[i];
  }
  gzprintf(visualizer_file, "\n");

  gzprintf(visualizer_file, "ctas_completed: %d\n", ctas_completed);
  ctas_completed = 0;
  // warp issue breakdown
  unsigned sid = m_config->gpgpu_warp_issue_shader;
  unsigned count = 0;
  unsigned warp_id_issued_sum = 0;
  gzprintf(visualizer_file, "WarpIssueSlotBreakdown:");
  if (m_shader_warp_slot_issue_distro[sid].size() > 0) {
    for (std::vector<unsigned>::const_iterator iter =
             m_shader_warp_slot_issue_distro[sid].begin();
         iter != m_shader_warp_slot_issue_distro[sid].end(); iter++, count++) {
      unsigned diff = count < m_last_shader_warp_slot_issue_distro.size()
                          ? *iter - m_last_shader_warp_slot_issue_distro[count]
                          : *iter;
      gzprintf(visualizer_file, " %d", diff);
      warp_id_issued_sum += diff;
    }
    m_last_shader_warp_slot_issue_distro = m_shader_warp_slot_issue_distro[sid];
  } else {
    gzprintf(visualizer_file, " 0");
  }
  gzprintf(visualizer_file, "\n");

#define DYNAMIC_WARP_PRINT_RESOLUTION 32
  unsigned total_issued_this_resolution = 0;
  unsigned dynamic_id_issued_sum = 0;
  count = 0;
  gzprintf(visualizer_file, "WarpIssueDynamicIdBreakdown:");
  if (m_shader_dynamic_warp_issue_distro[sid].size() > 0) {
    for (std::vector<unsigned>::const_iterator iter =
             m_shader_dynamic_warp_issue_distro[sid].begin();
         iter != m_shader_dynamic_warp_issue_distro[sid].end();
         iter++, count++) {
      unsigned diff =
          count < m_last_shader_dynamic_warp_issue_distro.size()
              ? *iter - m_last_shader_dynamic_warp_issue_distro[count]
              : *iter;
      total_issued_this_resolution += diff;
      if ((count + 1) % DYNAMIC_WARP_PRINT_RESOLUTION == 0) {
        gzprintf(visualizer_file, " %d", total_issued_this_resolution);
        dynamic_id_issued_sum += total_issued_this_resolution;
        total_issued_this_resolution = 0;
      }
    }
    if (count % DYNAMIC_WARP_PRINT_RESOLUTION != 0) {
      gzprintf(visualizer_file, " %d", total_issued_this_resolution);
      dynamic_id_issued_sum += total_issued_this_resolution;
    }
    m_last_shader_dynamic_warp_issue_distro =
        m_shader_dynamic_warp_issue_distro[sid];
    assert(warp_id_issued_sum == dynamic_id_issued_sum);
  } else {
    gzprintf(visualizer_file, " 0");
  }
  gzprintf(visualizer_file, "\n");

  // overall cache miss rates
  gzprintf(visualizer_file, "gpgpu_n_l1cache_bkconflict: %d\n",
           gpgpu_n_l1cache_bkconflict);
  gzprintf(visualizer_file, "gpgpu_n_shmem_bkconflict: %d\n",
           gpgpu_n_shmem_bkconflict);

  // instruction count per shader core
  gzprintf(visualizer_file, "shaderinsncount:  ");
  for (unsigned i = 0; i < m_config->num_shader(); i++)
    gzprintf(visualizer_file, "%u ", m_num_sim_insn[i]);
  gzprintf(visualizer_file, "\n");
  // warp instruction count per shader core
  gzprintf(visualizer_file, "shaderwarpinsncount:  ");
  for (unsigned i = 0; i < m_config->num_shader(); i++)
    gzprintf(visualizer_file, "%u ", m_num_sim_winsn[i]);
  gzprintf(visualizer_file, "\n");
  // warp divergence per shader core
  gzprintf(visualizer_file, "shaderwarpdiv: ");
  for (unsigned i = 0; i < m_config->num_shader(); i++)
    gzprintf(visualizer_file, "%u ", m_n_diverge[i]);
  gzprintf(visualizer_file, "\n");
}

#define PROGRAM_MEM_START                                      \
  0xF0000000 /* should be distinct from other memory spaces... \
                check ptx_ir.h to verify this does not overlap \
                other memory spaces */

/*
����PCֵ����ȡָ�
*/
const warp_inst_t *exec_shader_core_ctx::get_next_inst(unsigned warp_id,
                                                       address_type pc) {
  // read the inst from the functional model
  //ptx_fetch_inst(pc)�Ĺ���������PCֵ����ȡָ�
  return m_gpu->gpgpu_ctx->ptx_fetch_inst(pc);
}

/*
��ȡwarp_id��Ӧ��SIMT��ջ������PCֵ��RPCֵ��
*/
void exec_shader_core_ctx::get_pdom_stack_top_info(unsigned warp_id,
                                                   const warp_inst_t *pI,
                                                   unsigned *pc,
                                                   unsigned *rpc) {
  //SIMT��ջ��һ��warp��һ����m_simt_stack��ÿ��warp��һ����
  //SIMT��ջ��get_pdom_stack_top_info()�����Ĺ����ǻ�ȡSIMT��ջ������PCֵ��RPCֵ��
  m_simt_stack[warp_id]->get_pdom_stack_top_info(pc, rpc);
}

const active_mask_t &exec_shader_core_ctx::get_active_mask(
    unsigned warp_id, const warp_inst_t *pI) {
  return m_simt_stack[warp_id]->get_active_mask();
}

/*
Shader Core�еĽ���׶���m_inst_fetch_buffer�ṹ������أ����߳䵱��ȡ�ͽ���׶�֮���ͨ�Źܵ�������
�׶���μ������ֲ�˵����
    ������׶�ֻ����shader_core_ctx::m_inst_fetch_buffer������ʼ�������ָ���ǰ����ÿ���������
    ��������ָ��洢��ָ�������Ŀ��m_ibuffer��shd_warp_t::ibuffer_entry�Ķ����У�����Ŀ��Ӧ��
    shader-core_ctx::m_inst_fetch_bbuffer�е�warp����
*/
void shader_core_ctx::decode() {
  //m_inst_fetch_buffer�Ķ���Ϊ��
  //    ifetch_buffer_t m_inst_fetch_buffer;
  //ָ���ȡ��������ָ���ȡ��������ifetch_Buffer_t����ָ��棨I-cache����SIMT Core֮��Ľӿڽ���
  //��ģ������һ����Աm_valid������ָʾ�������Ƿ�����Ч��ָ�������ָ���warp id��¼��m_warp_id�С�
  //��ˣ���m_validΪ0����ָʾ��������ʱû����Ч��ָ���m_validΪ1����ָʾ�������Ѿ�����Ч��ָ�
  if (m_inst_fetch_buffer.m_valid) {
    // decode 1 or 2 instructions and place them into ibuffer.
    //����1~2��ָ��������Ƿŵ�I-Buffer�С�
    //��ȡm_inst_fetch_buffer�д洢��ָ���PCֵ��
    address_type pc = m_inst_fetch_buffer.m_pc;
    //get_next_inst()����PCֵ����ȡָ�
    const warp_inst_t *pI1 = get_next_inst(m_inst_fetch_buffer.m_warp_id, pc);
    //��һ����ָ�����I-Bufer��I-Buffer�������ۣ�pI1�����0��pI2�����1��
    m_warp[m_inst_fetch_buffer.m_warp_id]->ibuffer_fill(0, pI1);
    //��������ˮ����ִ�е�ָ������
    m_warp[m_inst_fetch_buffer.m_warp_id]->inc_inst_in_pipeline();
    if (pI1) {
      //m_num_decoded_insn��SM�Ͻ�����ָ������pI1ָ����Ч�Ļ�������1��
      m_stats->m_num_decoded_insn[m_sid]++;
      if ((pI1->oprnd_type == INT_OP) ||
          (pI1->oprnd_type == UN_OP)) {  // these counters get added up in mcPat
                                         // to compute scheduler power
        m_stats->m_num_INTdecoded_insn[m_sid]++;
      } else if (pI1->oprnd_type == FP_OP) {
        m_stats->m_num_FPdecoded_insn[m_sid]++;
      }
      //��ȡ��һ��ָ�һ��decode����Ž�ibuffer����ָ����������ٻ�ȡһ��ָ�
      const warp_inst_t *pI2 =
          get_next_inst(m_inst_fetch_buffer.m_warp_id, pc + pI1->isize);
      if (pI2) {
        m_warp[m_inst_fetch_buffer.m_warp_id]->ibuffer_fill(1, pI2);
        m_warp[m_inst_fetch_buffer.m_warp_id]->inc_inst_in_pipeline();
        m_stats->m_num_decoded_insn[m_sid]++;
        if ((pI1->oprnd_type == INT_OP) ||
            (pI1->oprnd_type == UN_OP)) {  // these counters get added up in
                                           // mcPat to compute scheduler power
          m_stats->m_num_INTdecoded_insn[m_sid]++;
        } else if (pI2->oprnd_type == FP_OP) {
          m_stats->m_num_FPdecoded_insn[m_sid]++;
        }
      }
    }
    //������Ҫ˵����m_inst_fetch_buffer��m_ibuffer������m_inst_fetch_buffer�����ڻ�ȡ��ָ���
    //���ʣ��ͽ���׶�֮��䵱��ˮ�߼Ĵ�������ÿ��shd_warp_t����һ��m_ibuffer��I-Buffer��Ŀ(ibuffer
    //entry)�����п����õ�ָ��������һ�������������ȡ�����ָ���������Ϻ�m_inst_fetch_buffer
    //����ΪFalse���Ա�����һ�ļ���fetch������
    m_inst_fetch_buffer.m_valid = false;
  }
}

/*
SIMT Core��ȡָ��ʱ�����ڡ�fetch()��������ָ���ڴ����󣬲���һ��ָ������ռ���ȡ��ָ���ȡ��ָ��
������ָ����ȡ�����������µ��߼�Ϊ��
  * ���m_inst_fetch_bufferΪ�գ���Ч����
    ** ���ָ�������ָ���׼����������
      *** ��ָ�����m_inst_fetch_buffer��
    ** ���򣬼����������û�о���ָ�
      *** ��������Ӳ��warp��2048/32�������warp�������У�û�еȴ�instruction cache missing������
          ��ָ�����Ϊ�գ��������ڴ��ȡ����
  * ����m_L1I->cycle()������

������Ҫ˵����m_inst_fetch_buffer��m_ibuffer������m_inst_fetch_buffer�����ڻ�ȡ��ָ�����ʣ�
�ͽ���׶�֮��䵱��ˮ�߼Ĵ�������ÿ��shd_warp_t����һ��m_ibuffer��I-Buffer��Ŀ(ibuffer_entry)����
�п����õ�ָ��������һ�������������ȡ�����ָ����������m_inst_fetch_bufferΪ�գ���Ҫ�ж�L1ָ�
���Ƿ��о�����ָ�����ָ����ڵĻ�Ҫ��L1ָ����л�ȡָ������L1ָ�����û��ָ���ʱ��Ҫ�ж�
һ���Ƿ���Ϊwarp������ϵ���L1ָ�����û��ָ���ˣ������warp�������У���û�еȴ������L1����δ���е�
����״̬����warp��ָ�����Ϊ�գ���ʱ���˵��Ҫȡ��һ��PCֵ��ָ�
*/
void shader_core_ctx::fetch() {
  //m_inst_fetch_buffer�Ķ���Ϊ��
  //    ifetch_buffer_t m_inst_fetch_buffer;
  //ifetch_buffer_t�Ķ���Ϊ��
  //    struct ifetch_buffer_t {
  //       ifetch_buffer_t() { m_valid = false; }
  //       ifetch_buffer_t(address_type pc, unsigned nbytes, unsigned warp_id) {
  //         m_valid = true;
  //         m_pc = pc;
  //         m_nbytes = nbytes;
  //         m_warp_id = warp_id;
  //       }
  //       bool m_valid;
  //       //��ȡ��ָ���PCֵ��
  //       address_type m_pc;
  //       unsigned m_nbytes;
  //       unsigned m_warp_id;
  //     };
  //������Կ���ָ���ȡ��������ifetch_Buffer_t�������ܹ��ݵ��µ���ָ�

  //ָ���ȡ��������ָ���ȡ��������ifetch_Buffer_t����ָ��棨I-cache����SIMT Core֮��Ľӿڽ���
  //��ģ������һ����Աm_valid������ָʾ�������Ƿ�����Ч��ָ�������ָ���warp id��¼��m_warp_id�С�
  //��ˣ���m_validΪ0����ָʾ��������ʱû����Ч��ָ�����Ԥȡ�µ�ָ�ע��Ԥȡ�µ�ָ��ʱ��Ҫ���µ�
  //ָ���½�һ�� ifetch_Buffer_t ���������� ifetch_Buffer_t �ṹ������ÿ��ȡ��ָ����һ��Ϊ�Ľ�ģ��

  //��ÿ��fetch()�����У�m_inst_fetch_buffer���ᷢ��ָ���ڴ���ȡ���󣬲���L1ָ������ռ���ȡ��ָ�
  //��ÿ��decode()�����У�m_inst_fetch_buffer�е������ı�����Ϊwarp_inst_t�Ķ��󣬲��洢���ȴ�������
  //��������ӦӲ��warp��ibuffer�С�
  if (!m_inst_fetch_buffer.m_valid) {
    //m_L1I��ָ��棨I-cache�������ֲ���<<����SIMT Cores>>������I-cache����ϸͼ����������Ѿ�����
    //��MSHR��Ŀ�ķ��ʣ���m_L1I->access_ready()����true������������ڴ���ʴ�����ǣ�I-cache������
    //�Ŀ��Ծ�����ָ���������Ѿ�������MSHR��Ŀ�ķ��ʣ��򷵻�true������Ŀ�ǿ�֤�����Ժϲ��ڴ���ʡ�
    //m_current_response�Ǿ����ڴ���ʵ��б�m_L1I->access_ready()���ص���m_current_response����
    //���о������ڴ���ʡ�m_current_response���洢�˾����ڴ���ʵĵ�ַ��
    if (m_L1I->access_ready()) {
      //��ȡI-cache���´��ڴ���ʣ�������һ���Ѿ�����ķ��ʣ���������һ���Ѿ�������ʵ�����
      mem_fetch *mf = m_L1I->next_access();
      //���ǰ�� mem_fetch *mf �Ѿ���ȡ�˾�����ָ���֤�� mf ���ڵ�warp���ڲ����ڹ����ָ���δ��
      //�е�״̬�����ø�״̬Ϊfalse��mf->get_wid()���ص��ǵ�ǰ�Ѿ�����ָ�����ڵ�warp��ID����ô֤����
      //ǰmf->get_wid()ָʾ��warp�Ѿ��ܻ�ȡ��Чָ��������ڹ����ָ���δ���е�״̬��
      m_warp[mf->get_wid()]->clear_imiss_pending();
      //��������ָ��Ԥȡ��һ��Ϊ�Ķ��󣬴������Ϊ��
      //    address_type pc��m_warp[mf->get_wid()]->get_pc()
      //    unsigned nbytes��mf->get_access_size()
      //    unsigned warp_id��mf->get_wid()
      m_inst_fetch_buffer =
          ifetch_buffer_t(m_warp[mf->get_wid()]->get_pc(),
                          mf->get_access_size(), mf->get_wid());
      assert(m_warp[mf->get_wid()]->get_pc() ==
             (mf->get_addr() -
              PROGRAM_MEM_START));  // Verify that we got the instruction we
                                    // were expecting.
      //����ָʾ�������ڵ�ָ����Ч��
      m_inst_fetch_buffer.m_valid = true;
      //�����ϴ�ȡָ��ʱ�����ڡ�
      m_warp[mf->get_wid()]->set_last_fetch(m_gpu->gpu_sim_cycle);
      delete mf;
    } else {
      // find an active warp with space in instruction buffer that is not
      // already waiting on a cache miss and get next 1-2 instructions from
      // i-cache...
      //����m_L1I->access_ready()����������ָ�����û�о�����ָ���ָ��������ҵ�һ��ָʾ
      //��ָ���ȡ����ռ䣨�����m_inst_fetch_buffer���Ļ�Ծwarp���ÿռ���δ���ڻ���δ���ж���
      //��������I-cache�л�ȡ��һ��1-2��ָ�������һ��warpʱ����ȡ��ѯ���ƣ�
      //    (m_last_warp_fetched + 1 + i) % m_config->max_warps_per_shader;
      for (unsigned i = 0; i < m_config->max_warps_per_shader; i++) {
        //��ѯ���ƻ�ȡ��һ����Ծwarp��
        unsigned warp_id =
            (m_last_warp_fetched + 1 + i) % m_config->max_warps_per_shader;

        // this code checks if this warp has finished executing and can be
        // reclaimed.
        //����Ĵ��������warp�Ƿ��Ѿ����ִ�в��ҿ��Ի��գ���������
        //    m_warp[warp_id]->hardware_done()������warp�Ƿ��Ѿ����ִ�в��ҿ��Ի��գ�
        //    m_scoreboard->pendingWrites(warp_id)���ؼǷ��Ƶ�reg_table���Ƿ��й����д�룻
        //    m_warp[warp_id]->done_exit()�����߳��˳��ı�ʶ��
        //m_scoreboard->pendingWrites(warp_id)���ؼǷ��Ƶ�reg_table���Ƿ��������ڵ�ǰwarpid
        //�Ĺ����д�롣warp idָ���reg_tableΪ�յĻ�������û�й����д�룬����false��[�����
        //д��]��ָwid�Ƿ����ѷ��䵫��δ��ɵ�ָ���Ŀ��Ĵ��������ڼǷ��ƣ���ʱ���warp��δ��
        //��ִ�в����ܻ��ա�
        //shd_warp_t::hardware_done()�У�
        //    functional_done()����warp�Ѿ�ִ����ϵı�־���Ѿ���ɵ��߳�����=warp�Ĵ�Сʱ����
        //    �����warp�Ѿ���ɡ�stores_done()��������store�ô������Ƿ��Ѿ�ȫ��ִ���꣬�ѷ���
        //    ����δȷ�ϵ�д�洢�����ѷ���д����δ�յ�дȷ���ź�ʱ����m_stores_outstanding
        //    =0ʱ����������store�ô������Ѿ�ȫ��ִ���꣬����m_stores_outstanding�ڷ���һ��д
        //    ����ʱ+=1�����յ�һ��дȷ��ʱ-=1��m_inst_fetch_buffer�к���Чָ��ʱ�ҽ���ָ���
        //    �����������warp��m_ibufferʱ����������ˮ���е�ָ����m_inst_in_pipeline��ע��
        //    ������decode()�����л���warp��m_ibuffer����2��ָ�����ָ�����д�ز���ʱ����
        //    ����ˮ���е�ָ����m_inst_in_pipeline��inst_in_pipeline()������ˮ���е�ָ������
        //    m_inst_in_pipeline��
        //    ����һ��warp��ɵı�־������������ɣ��ֱ��ǣ�1��warp�Ѿ�ִ����ϣ�2������store��
        //    �������Ѿ�ȫ��ִ���ꣻ3����ˮ���е�ָ������Ϊ0��
        if (m_warp[warp_id]->hardware_done() &&
            !m_scoreboard->pendingWrites(warp_id) &&
            !m_warp[warp_id]->done_exit()) {
          //did_exit�Ǳ�ʶ��ǰѭ���ڵ�warp�Ƿ��˳����������warp�ĸ����̶߳�ֹͣ�󣬾�����Ϊ�档
          bool did_exit = false;
          //��һ��warp�ڵ������߳̽���ѭ����
          for (unsigned t = 0; t < m_config->warp_size; t++) {
            //tid���̱߳�ţ���������һ��Shader Core�������̵߳���������������warp�ڲ���0~31��
            unsigned tid = warp_id * m_config->warp_size + t;
            //���tid���̴߳��ڻ�Ծ״̬��
            if (m_threadState[tid].m_active == true) {
              //m_threadState[i]��ʶ��i���߳��Ƿ��ڻ�Ծ״̬��m_threadState��һ�����飬������
              //������Shader Core�����е��̵߳�״̬��
              m_threadState[tid].m_active = false;
              //�����߳����ڵ�CTA��ID��
              unsigned cta_id = m_warp[warp_id]->get_cta_id();
              if (m_thread[tid] == NULL) {
                //������߳���ϢΪ�գ���ע����߳��˳���register_cta_thread_exit������ע��cta_id
                //����ʶ��CTA�еĵ����߳��˳�
				register_cta_thread_exit(cta_id,
                                         m_warp[warp_id]->get_kernel_info());
              } else {
                register_cta_thread_exit(cta_id,
                                         &(m_thread[tid]->get_kernel()));
              }
              //δ��ɵ��߳���������Shader Core�ϵ������̶߳����ʱ��==0������������ע�����̵߳�
              //�˳�����˸��߳������ִ�У�δ��ɵ��߳�����Ҫ��ȥ1��
              m_not_completed -= 1;
              //m_active_threads��Shader Core�ϻ�Ծ�̵߳�λͼ��ͬ��tidλ���㣬ȡ����Ծ״̬��
              m_active_threads.reset(tid);
              //did_exit�Ǳ�ʶ��ǰѭ���ڵ�warp�Ѿ��˳���
              did_exit = true;
            }
          }
          //��ǰѭ���ڵ�warp�Ѿ��˳����������˳��ı�־��
          if (did_exit) m_warp[warp_id]->set_done_exit();
          //m_active_warps���ڴ�Shader Core�еĻ�Ծwarp��������Ӧ�ü�1��
          --m_active_warps;
          assert(m_active_warps >= 0);
        }

        // this code fetches instructions from the i-cache or generates memory
        //�˴����I-Cache��ȡָ��������ڴ���ʡ�functional_done()����warp�Ƿ��Ѿ�ִ����ϣ���
        //����ɵ��߳�����=warp�Ĵ�Сʱ���ʹ����warp�Ѿ���ɡ�imiss_pending()����warp�Ƿ���ָ
        //���δ���ж������״̬��ʶ��ibuffer_empty()����I-Bufer�Ƿ�Ϊ�ա�
        //��������ǰ���m_L1I->access_ready()����������ָ�����û�о�����ָ��ҵ�ǰwarp��δ
        //ִ����ϣ��Ҹ�warp��δ����ָ���δ���й����״̬���Ҹ�warp��m_ibufferΪ�գ�û�п���
        //����ִ�е�ָ�����Ҫ���������ڸ�warp����һ��ָ��Ԥȡ��

        //����˵����m_inst_fetch_buffer��m_ibuffer������m_inst_fetch_buffer�����ڻ�ȡ��ָ��
        //������ʣ��ͽ���׶�֮��䵱��ˮ�߼Ĵ�������ÿ��shd_warp_t����һ��m_ibuffer��I-Buffer
        //��Ŀ(ibuffer_entry)�����п����õ�ָ��������һ�������������ȡ�����ָ������ȣ����
        //m_inst_fetch_bufferΪ�գ���Ҫ�ж�L1ָ����Ƿ��о�����ָ�����ָ����ڵĻ�Ҫ��L1ָ
        //����л�ȡָ������L1ָ�����û��ָ���ʱ��Ҫ�ж�һ���Ƿ���Ϊwarp������ϵ���
        //L1ָ�����û��ָ���ˣ������warp�������У���û�еȴ������L1����δ���еĹ���״̬����
        //warp��ָ�����Ϊ�գ���ʱ���˵��Ҫȡ��һ��PCֵ��ָ�
        if (!m_warp[warp_id]->functional_done() &&
            !m_warp[warp_id]->imiss_pending() &&
            m_warp[warp_id]->ibuffer_empty()) {
          address_type pc;
          //����warp����һ��Ҫִ�е�ָ���PCֵ��
          pc = m_warp[warp_id]->get_pc();
          //��һ����ȡ��PCֵ���Ǵ�0��ʼ��ŵģ���Ҫ����ָ�����ڴ��д洢���׵�ַ0xF0000000���ܵ�
          //I-Cache��ȡָ���ΪI-Cache����ʼ��ַ����0xF0000000��
          address_type ppc = pc + PROGRAM_MEM_START;
          unsigned nbytes = 16;
          //offset_in_block��PC��ʶ��ָ����I-cache line�е�ƫ�ơ�
          unsigned offset_in_block =
              pc & (m_config->m_L1I_config.get_line_sz() - 1);
          //������ƫ��+nbytes > I-cache line Size����˵��һ��Cache�治��nbyte��С�����ݣ���
          //������nbytes��ֵ��
          if ((offset_in_block + nbytes) > m_config->m_L1I_config.get_line_sz())
            nbytes = (m_config->m_L1I_config.get_line_sz() - offset_in_block);

          // TODO: replace with use of allocator
          // mem_fetch *mf = m_mem_fetch_allocator->alloc()
          //�����ڴ������Ϊ������󣬸��ڴ���ʶ����������INST_ACC_R����I-Cache�ж�ָ���ַ
          //ΪPCֵ��nbytes�ֽڴ�С����д���ʡ�
          mem_access_t acc(INST_ACC_R, ppc, nbytes, false, m_gpu->gpgpu_ctx);
          mem_fetch *mf = new mem_fetch(
              acc, NULL, m_warp[warp_id]->get_kernel_info()->get_streamID(),
              READ_PACKET_SIZE, warp_id, m_sid, m_tpc, m_memory_config,
              m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle);
          std::list<cache_event> events;

          // Check if the instruction is already in the instruction cache.
          enum cache_request_status status;
          //perfect_inst_const_cache����������inst��const����ģʽ�������е�����inst��const��
          //���У���V100�����ļ��￪����
          if (m_config->perfect_inst_const_cache) {
            status = HIT;
            shader_cache_access_log(m_sid, INSTRUCTION, 0);
          } else
            status = m_L1I->access(
                (new_addr_type)ppc, mf,
                m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle, events);
          //������inst��const����ģʽ�������е�����inst��HIT��
          if (status == MISS) {
            m_last_warp_fetched = warp_id;
            m_warp[warp_id]->set_imiss_pending();
            m_warp[warp_id]->set_last_fetch(m_gpu->gpu_sim_cycle);
          } else if (status == HIT) {
            m_last_warp_fetched = warp_id;
            //��pcֵ��Ӧ��ָ�����m_inst_fetch_buffer������Ҫ�ٴ�˵����m_inst_fetch_buffer��
            //m_ibuffer������m_inst_fetch_buffer�����ڻ�ȡ��ָ�����ʣ��ͽ���׶�֮��䵱
            //��ˮ�߼Ĵ�������ÿ��shd_warp_t����һ��m_ibuffer��I-Buffer��Ŀ(ibuffer_entry)��
            //���п����õ�ָ��������һ�������������ȡ�����ָ���
            m_inst_fetch_buffer = ifetch_buffer_t(pc, nbytes, warp_id);
            m_warp[warp_id]->set_last_fetch(m_gpu->gpu_sim_cycle);
            delete mf;
          } else {
            m_last_warp_fetched = warp_id;
            assert(status == RESERVATION_FAIL);
            delete mf;
          }
          break;
        }
      }
    }
  }
  //ȡָ��ɺ�L1 I-Cache��ǰ�ƽ�һ�ġ�
  m_L1I->cycle();
}

/*
ָ��Ĺ���ִ�С�������ģ�� shader_core_ctx::issue_warp ����ִ��ʱ����ָ��ȷ���ܷ���ʱ��
���øú�����ָ����й���ģ�⡣����ģ��Ĺ����У����ָ����load��storeָ��������ڴ��
�ʵı�Ҫ����Ϣ�����磺���� access_type������һ�� warp ָ��� shared memory �ô��ִ��
���ڵȡ�
*/
void exec_shader_core_ctx::func_exec_inst(warp_inst_t &inst) {
  execute_warp_inst_t(inst);
  if (inst.is_load() || inst.is_store()) {
    inst.generate_mem_accesses();
    // inst.print_m_accessq();
  }
}

/*
����warp�����磺
    m_shader->issue_warp(*m_mem_out, pI, active_mask, warp_id, m_id);
*/
void shader_core_ctx::issue_warp(register_set &pipe_reg_set,
                                 const warp_inst_t *next_inst,
                                 const active_mask_t &active_mask,
                                 unsigned warp_id, unsigned sch_id) {
  //��pipe_reg_set��ˮ�߼Ĵ�����Ѱ��sch_id��Ӧ�Ŀ��еļĴ�������subcoreģʽ�£�ÿ
  //��warp�������ڼĴ�����������һ������ļĴ����ɹ�ʹ�ã�����Ĵ����ɵ�������m_id
  //������
  warp_inst_t **pipe_reg =
      pipe_reg_set.get_free(m_config->sub_core_model, sch_id);
  assert(pipe_reg);
  //�����Ѿ���������ָ���˽�I-Bufer�е�next_inst���ڵĲ��������Ч��
  m_warp[warp_id]->ibuffer_free();
  assert(next_inst->valid());
  **pipe_reg = *next_inst;  // static instruction information
  //(*pipe_reg)->issue()�Ķ������£�
  //    void warp_inst_t::issue(const active_mask_t &mask, unsigned warp_id,
  //                            unsigned long long cycle, int dynamic_warp_id,
  //                            int sch_id) {
  //      m_warp_active_mask = mask;
  //      m_warp_issued_mask = mask;
  //      m_uid = ++(m_config->gpgpu_ctx->warp_inst_sm_next_uid);
  //      m_warp_id = warp_id;
  //      m_dynamic_warp_id = dynamic_warp_id;
  //      issue_cycle = cycle;
  //      cycles = initiation_interval;
  //      m_cache_hit = false;
  //      m_empty = false;
  //      m_scheduler_id = sch_id;
  //    }
  //����ָ�̬��������е�һЩ��Ϣ��
  (*pipe_reg)->issue(
      active_mask, warp_id, m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle,
      m_warp[warp_id]->get_dynamic_warp_id(), sch_id,
      m_warp[warp_id]->get_streamID());  // dynamic instruction information
  m_stats->shader_cycle_distro[2 + (*pipe_reg)->active_count()]++;
  //�����Ѿ�ȷ����ָ��next_inst��ִ��˳��û�����⣬��˿��ԶԸ���ָ����й���ģ�⡣
  func_exec_inst(**pipe_reg);

  // Add LDGSTS instructions into a buffer
  unsigned int ldgdepbar_id = m_warp[warp_id]->m_ldgdepbar_id;
  if (next_inst->m_is_ldgsts) {
    if (m_warp[warp_id]->m_ldgdepbar_buf.size() == ldgdepbar_id + 1) {
      m_warp[warp_id]->m_ldgdepbar_buf[ldgdepbar_id].push_back(*next_inst);
    } else {
      assert(m_warp[warp_id]->m_ldgdepbar_buf.size() < ldgdepbar_id + 1);
      std::vector<warp_inst_t> l;
      l.push_back(*next_inst);
      m_warp[warp_id]->m_ldgdepbar_buf.push_back(l);
    }
    // If the mask of the instruction is all 0, then the address is also 0,
    // so that there's no need to check through the writeback
    if (next_inst->get_active_mask() == 0) {
      (m_warp[warp_id]->m_ldgdepbar_buf.back()).back().pc = -1;
    }
  }
  
  //��������ָ���OP������������ָ��򱣴浱ǰwarp��������ָ��״̬��
  if (next_inst->op == BARRIER_OP) {
    m_warp[warp_id]->store_info_of_last_inst_at_barrier(*pipe_reg);
    m_barriers.warp_reaches_barrier(m_warp[warp_id]->get_cta_id(), warp_id,
                                    const_cast<warp_inst_t *>(next_inst));

  } else if (next_inst->op == MEMORY_BARRIER_OP) {
    m_warp[warp_id]->set_membar();
  } else if (next_inst->m_is_ldgdepbar) {  // Add for LDGDEPBAR
    m_warp[warp_id]->m_ldgdepbar_id++;
    // If there are no added LDGSTS, insert an empty vector
    if (m_warp[warp_id]->m_ldgdepbar_buf.size() != ldgdepbar_id + 1) {
      assert(m_warp[warp_id]->m_ldgdepbar_buf.size() < ldgdepbar_id + 1);
      std::vector<warp_inst_t> l;
      m_warp[warp_id]->m_ldgdepbar_buf.push_back(l);
    }
  } else if (next_inst->m_is_depbar) {  // Add for DEPBAR
    // Set to true immediately when a DEPBAR instruction is met
    m_warp[warp_id]->m_waiting_ldgsts = true;
    m_warp[warp_id]->m_depbar_group =
        next_inst->m_depbar_group_no;  // set in trace_driven.cc

    // Record the last group that's possbily being monitored by this DEPBAR
    // instr
    m_warp[warp_id]->m_depbar_start_id = m_warp[warp_id]->m_ldgdepbar_id - 1;

    // Record the last group that's actually being monitored by this DEPBAR
    // instr
    unsigned int end_group =
        m_warp[warp_id]->m_ldgdepbar_id - m_warp[warp_id]->m_depbar_group;

    // Check for the case that the LDGSTSs monitored have finished when
    // encountering the DEPBAR instruction
    bool done_flag = true;
    for (int i = 0; i < end_group; i++) {
      for (int j = 0; j < m_warp[warp_id]->m_ldgdepbar_buf[i].size(); j++) {
        if (m_warp[warp_id]->m_ldgdepbar_buf[i][j].pc != -1) {
          done_flag = false;
          goto UpdateDEPBAR;
        }
      }
    }

  UpdateDEPBAR:
    if (done_flag) {
      if (m_warp[warp_id]->m_waiting_ldgsts) {
        m_warp[warp_id]->m_waiting_ldgsts = false;
      }
    }
  }

  //����SIMT��ջ��
  updateSIMTStack(warp_id, *pipe_reg);
  //���üƷְ壬����ָ��ʱ������Ŀ��Ĵ�����������ӦӲ��warp�ļǷ����С�
  m_scoreboard->reserveRegisters(*pipe_reg);
  //������һ��ָ���PCֵ��
  m_warp[warp_id]->set_next_pc(next_inst->pc + next_inst->isize);
}

/*
��ÿ��SIMT Core�У����п����������ĵ�������Ԫ������shader_core_ctx::issue()����Щ��Ԫ�Ͻ��е�����
����ÿһ����Ԫ��ִ��scheduler_unit::cycle()���������warp������ѭ����scheduler_unit::cycle()�У�
ָ��ʹ��shader_core_ctx::issue_warp()���������䵽����ʵ�ִ����ˮ�ߡ�����������У�ָ��ͨ������
shader_core_ctx::func_exec_inst()�ڹ����ϱ�ִ�У�SIMT��ջ��m_simt_stack[warp_id]����ͨ������
simt_stack::update()�����¡����⣬����������У�����barrier�Ĵ��ڣ�ͨ��shd_warp_t:set_membar()
��barrier_set_t::warp_reaches_barrier������/�ͷ�warp����һ���棬�Ĵ�����Scoreboard::reserve-
Registers()�������Ա��Ժ󱻼Ƿ����㷨ʹ�á�scheduler_unit::m_sp_out,scheduler_unit::m_sfu_out, 
scheduler_unit::m_mem_outָ��SP��SFU��Mem��ˮ�߽��յķ���׶κ�ִ�н׶�֮��ĵ�һ����ˮ�߼Ĵ�����
�����Ϊʲô��ʹ��shader_core_ctx::issue_warp()������Ӧ����ˮ�߷����κ�ָ��֮ǰҪ������ǡ�

����ָ������º��ӳ�
��ÿ��pipelined_simd_unit�У�issue()��Ա��������������ˮ�߼Ĵ�������������m_dispatch_reg��Ȼ��ָ
����m_dispatch_reg�ȴ�initiation_interval�����ڡ��ڴ��ڼ䣬û��������ָ����Է��������Ԫ��������
���ȴ���ָ�����������ģ�͡��ȴ�֮��ָ��ɷ����ڲ���ˮ�߼Ĵ���m_pipeline_reg�����ӳٽ�ģ����ǲ
��λ����ȷ���ģ�������m_dispatch_reg�л��ѵ�ʱ��Ҳ�������ӳ��С�ÿ�����ڣ�ָ�ͨ����ˮ�߼Ĵ���ǰ
�������ս���m_result_port�����ǹ������ˮ�߼Ĵ�����ͨ��SP��SFU��Ԫ�Ĺ�ͬд�ؽ׶Ρ�ʾ��ͼ��

              m_dispatch_reg    m_pipeline_reg      
                  / |            |---------|
                 /  |----------> |         | 31  --|
                /   |            |---------|       |
Dispatch done every |----------> |         | :     |
Issue_interval cycle|            |---------|       |  Pipeline registers to
to model instruction|----------> |         | 2     |- model instruction latency
throughput          |            |---------|       |
                    |----------> |         | 1     |
                    |            |---------|       |
                    |----------> |         | 0   --|
Dispatch position =              |---------|
Latency - Issue_interval             |
                                    \|/
                                     V
                                m_result_port --> д��
/////////////////////////////////////////////////////////////////////////////////////////
# GPGPU-Sim��ʱ���ƽ�

## �ĸ�ʱ�����ѡ��

ÿ��ʱ����ά�����Եĵ�ǰִ�������������ﲻͬʱ����ĵ���ִ���������ɸ���ʱ�����Ƶ�ʾ�����
```c++
l2_time, icnt_time, dram_time, core_time
```
GPGPU-Simÿ����ǰ�ƽ�һ��ʱ������ʱ����ѡ��һ���򼸸���ǰִ����������С��ʱ��������ƽ���ͬʱ����ʱ
����ά���ĵ�ǰִ������������һ�ģ����ѡ�����ɺ���`gpgpu_sim::next_clock_domain(void)`��ɵġ�
���磬��ǰ�ĸ�ʱ�����ִ�����ڷֱ�Ϊ��
```c++
l2_time=35��, icnt_time=40��, dram_time=50��, core_time=30��
```
��ô��ѡ��`CORE`ʱ������и��£�ͬʱ��`core_time`����һ�ģ���Ϊ31�ģ�����ʱ�����ִ�����������䡣

## ʱ����ĸ���
��ѡ��`CORE`ʱ������и��º󣬻�ѭ������SIMT Core��Ⱥ���ƽ�һ��ʱ�����ڣ�
```c++
if (clock_mask & CORE) {
    //shader core loading (pop from ICNT into core) follows CORE clock.
    //�����е�SIMT Core��Ⱥѭ����m_cluster[i]������һ����Ⱥ��Shader Core���أ���ICNT������Core��
    //��ѭCoreʱ�ӡ�
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
      //simt_core_cluster::icnt_cycle()�������ڴ�����ӻ�����������SIMT���ļ�Ⱥ����ӦFIFO��������
      //FIFO�������󣬲������Ƿ��͵���Ӧ�ں˵�ָ����LDST��Ԫ��ÿ��SIMT Core��Ⱥ����һ����ӦFIFO��
      //���ڱ���ӻ������緢�������ݰ������ݰ�������SIMT Core��ָ��棨�������Ϊָ���ȡδ����
      //�ṩ������ڴ���Ӧ�������ڴ���ˮ�ߣ�memory pipeline��LDST ��Ԫ�������ݰ����Ƚ��ȳ���ʽ�ó���
      //���SIMT Core�޷�����FIFOͷ�������ݰ�������ӦFIFO��ֹͣ��Ϊ����LDST��Ԫ�������ڴ�����ÿ��
      //SIMT Core�����Լ���ע��˿ڽ��뻥�����硣���ǣ�ע��˿ڻ�������SIMT Core��Ⱥ����SIMT Core��
      //��
      m_cluster[i]->icnt_cycle();
}
......
if (clock_mask & CORE) {
    //��GPU�����е�SIMT Core��Ⱥ����ѭ��������ÿ����Ⱥ��״̬��
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
      //���get_not_completed()����1���������SIMT Core��δ��ɣ����get_more_cta_left()Ϊ1��
      //�������SIMT Core����ʣ���CTA��Ҫȡִ�С�m_cluster[i]->get_not_completed()���ص�i��
      //SIMT Core��Ⱥ����δ��ɵ��̸߳�����
      if (m_cluster[i]->get_not_completed() || get_more_cta_left()) {
        //������simt_core_cluster::core_cycle()ʱ�����������������SM�ں˵�ѭ����
        m_cluster[i]->core_cycle();
        //���ӻ�Ծ��SM������get_n_active_sms()����SIMT Core��Ⱥ�еĻ�ԾSM��������active_sms��
        //SIMT Core��Ⱥ�еĻ�ԾSM��������
        *active_sms += m_cluster[i]->get_n_active_sms();
      }
    //��Ҫע�⣬gpu_sim_cycle����COREʱ������ǰ�ƽ�һ��ʱ�Ÿ��£����gpu_sim_cycle��ʾCOREʱ��
    //��ĵ�ǰִ��������
    gpu_sim_cycle++;
    //������SIMT Core��Ⱥ������ѡ��ÿ����Ⱥ�ڵ�һ��SIMT Core�������䷢��һ���߳̿顣��ѡ������
    //��kernelʱ�������gpgpu_sim::select_kernel()�������÷������жϵ�ǰ�Ƿ���δ��ɵ�kernel��
    //���ж��Ƿ��kernel�Ѿ�����������ʱ���Ϊ0�������δ��ɵ�kernel���Ҹ�kernel�Ѿ�����������
    //ʱ���Ϊ0�����ѡ���kernel����ִ�С�
    issue_block2core();
    //m_kernel_TB_latency����ģ��kernel&block�������ӳ�ʱ�䣬�����kernel�������ӳ٣�ÿ���߳̿�
    //�������ӳ١�ÿ�ε���decrement_kernel_latency()ʱ�����Ὣm_kernel_TB_latency��1��ֱ����Ϊ
    //0����˵��kernel���Ա�ѡ����ִ�С�
    decrement_kernel_latency();
}
```

## ����SIMT Core��Ⱥ�Լ�����SIMT Core�ĸ���
����SIMT Core��Ⱥ��ǰ�ƽ�һ��ʱ�����ڣ�����`simt_core_cluster::core_cycle()`������
```c++
void simt_core_cluster::core_cycle() {
  //��SIMT Core��Ⱥ�е�ÿһ��������SIMT Coreѭ����
  for (std::list<unsigned>::iterator it = m_core_sim_order.begin();
       it != m_core_sim_order.end(); ++it) {
    //SIMT Core��Ⱥ�е�ÿһ��������SIMT Core����ǰ�ƽ�һ��ʱ�����ڡ�
    m_core[*it]->cycle();
  }
}
```
����SIMT Core��ǰ�ƽ�һ��ʱ�����ڣ�����`shader_core_ctx::cycle()`������
```c++
  //������SIMT Core���ڷǻ�Ծ״̬�����Ѿ�ִ����ɣ�ʱ�����ڲ���ǰ�ƽ���
  if (!isactive() && get_not_completed() == 0) return;

  //ÿ���ں�ʱ�����ڣ�shader_core_ctx::cycle()�������ã���ģ��SIMT Core��һ�����ڡ������������һ
  //���Ա���������෴��˳��ģ���ں˵���ˮ�߽׶Σ���ģ����ˮ��ЧӦ��
  //     fetch()             ������һִ��
  //     decode()            �����ڶ�ִ��
  //     issue()             ��������ִ��
  //     read_operand()      ��������ִ��
  //     execute()           ��������ִ��
  //     writeback()         ��������ִ��
  writeback();
  execute();
  read_operands();
  issue();
  for (int i = 0; i < m_config->inst_fetch_throughput; ++i) {
    decode();
    fetch();
  }
```

## ����ָ������º��ӳ�
��ÿ��pipelined_simd_unit�У�issue(warp_inst_t*&)��Ա��������������ˮ�߼Ĵ�������������m_dispat
ch_reg��Ȼ��ָ����m_dispatch_reg�ȴ�initiation_interval�����ڡ��ڴ��ڼ䣬û��������ָ����Է�����
����Ԫ����������ȴ���ָ�����������ģ�͡��ȴ�֮��ָ��ɷ����ڲ���ˮ�߼Ĵ���m_pipeline_reg������
�ٽ�ģ����ǲ��λ����ȷ���ģ�������m_dispatch_reg�л��ѵ�ʱ��Ҳ�������ӳ��С�ÿ�����ڣ�ָ�ͨ����ˮ
�߼Ĵ���ǰ�������ս���m_result_port�����ǹ������ˮ�߼Ĵ�����ͨ��SP��SFU��Ԫ�Ĺ�ͬд�ؽ׶Ρ�ʾ��ͼ��
```c++
              m_dispatch_reg    m_pipeline_reg      
                  / |            |---------|
                 /  |----------> |         | 31  --|
                /   |            |---------|       |
Dispatch done every |----------> |         | :     |
Issue_interval cycle|            |---------|       |  Pipeline registers to
to model instruction|----------> |         | 2     |- model instruction latency
throughput          |            |---------|       |
                    |----------> |         | 1     |
                    |            |---------|       |
                    |----------> |         | 0   --|
Dispatch position =              |---------|
Latency - Issue_interval             |
                                    \|/
                                     V
                                m_result_port --> д��
```
/////////////////////////////////////////////////////////////////////////////////////////
*/
void shader_core_ctx::issue() {
  // Ensure fair round robin issu between schedulers
  unsigned j;
  //��Shader Core��Ŀ����������ĵ�������Ԫ���е���������ÿһ����Ԫ��ִ��scheduler_unit::cycle()��
  //������δ�����ʵ����ģ�����������ѭ��Volta�ܹ�ÿ��Shader Core��4������������һ�ĵ���ʱ��ѡ���
  //0������������ǰ�ƽ�һ�ģ�������1��2��3��������ǰ�ƽ�һ�ģ���һ�����Ȱѵ�1����������ǰ�ƽ�һ�ģ���
  //����2��3��0��������ǰ�ƽ�һ�ģ��Դ����ơ�
  for (unsigned i = 0; i < schedulers.size(); i++) {
    j = (Issue_Prio + i) % schedulers.size();
    //��������ǰ�ƽ�һ�ģ�����ִ�д�warp��m_ibuffer��ȡֵ��SIMT��ջ��飬�Ʒְ��飬��ˮ�ߵ�Ԫ��飬
    //���ִ��д���Ӧ����ˮ�ߵ�Ԫǰ�ļĴ������ϣ������й���ģ�⡣��Ҫע�⣬��ִ���ĸ���������һ�ε�
    //�ȣ�������õ�����ѯ���Ե��ȣ�����ѡ��ĳ��������ִ�е�ʱ�������ڸõ��������ĸ�warp��ִ��Ҳ��һ�ε�
    //�ȣ�����V100���ò��õ���LRR������ٱ�ʹ�ò��Ե��ȣ���
    schedulers[j]->cycle();
  }
  Issue_Prio = (Issue_Prio + 1) % schedulers.size();

  // really is issue;
  // for (unsigned i = 0; i < schedulers.size(); i++) {
  //    schedulers[i]->cycle();
  //}
}

shd_warp_t &scheduler_unit::warp(int i) { return *((*m_warp)[i]); }
//LRR���Ȳ��Եĵ�������Ԫ��order_lrr������Ϊ��ǰ���ȵ�Ԫ�������ֵ���warp��������order_lrr�Ķ���
//Ϊ��
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
//��������Ĺ��ܾ��Ǹ�����һ�ĵ��ȹ���warp���ҵ����ڵ�ǰ���ȵ�Ԫ����Ҫ�ٲõ�warp�����е�λ�ã�Ȼ��
//�����λ�ú����warp��ʼ��������ǰ���ȵ�Ԫ����Ҫ�ٲõ�warp���ϣ�������Щwarp����result_list�У�
//ֱ��result_list�е�warp��Ŀ����num_warps_to_add��

/**
 * A general function to order things in a Loose Round Robin way. The simplist
 * use of this function would be to implement a loose RR scheduler between all
 * the warps assigned to this core. A more sophisticated usage would be to order
 * a set of "fetch groups" in a RR fashion. In the first case, the templated
 * class variable would be a simple unsigned int representing the warp_id.  In
 * the 2lvl case, T could be a struct or a list representing a set of warp_ids.
 * @param result_list: The resultant list the caller wants returned.  This list
 * is cleared and then populated in a loose round robin way
 * @param input_list: The list of things that should be put into the
 * result_list. For a simple scheduler this can simply be the m_supervised_warps
 * list.
 * @param last_issued_from_input:  An iterator pointing the last member in the
 * input_list that issued. Since this function orders in a RR fashion, the
 * object pointed to by this iterator will be last in the prioritization list
 * @param num_warps_to_add: The number of warps you want the scheudler to pick
 * between this cycle. Normally, this will be all the warps availible on the
 * core, i.e. m_supervised_warps.size(). However, a more sophisticated scheduler
 * may wish to limit this number. If the number if < m_supervised_warps.size(),
 * then only the warps with highest RR priority will be placed in the
 * result_list.
 */
template <class T>
void scheduler_unit::order_lrr(
    std::vector<T> &result_list, const typename std::vector<T> &input_list,
    const typename std::vector<T>::const_iterator &last_issued_from_input,
    unsigned num_warps_to_add) {
  assert(num_warps_to_add <= input_list.size());
  //�����ǰ���ȵ�Ԫ��һ�ĵ��ȹ���warp���ڵ�ǰ���ȵ�Ԫ����Ҫ�ٲõ�warp�����У�������Ϊ��ǰ���ȵ�
  //Ԫ����Ҫ�ٲõ�warp���ϵĵ�һ��warp���������ǰ���ȵ�Ԫ��һ�ĵ��ȹ���warp�ڵ�ǰ���ȵ�Ԫ����Ҫ��
  //�õ�warp�����У�������Ϊ����˳��������һ��warp��
  result_list.clear();
  typename std::vector<T>::const_iterator iter =
      (last_issued_from_input == input_list.end()) ? input_list.begin()
                                                   : last_issued_from_input + 1;
  //�Ե�ǰ���ȵ�Ԫ����Ҫ�ٲõ�warp���Ͻ��б���������Щwarp���մ��϶δ����ҵ�����һ�ĵ��ȹ���warp��
  //��һ��warp��ʼֱ������warp������һ���˳�򣬷���result_list�С�
  for (unsigned count = 0; count < num_warps_to_add; ++iter, ++count) {
    if (iter == input_list.end()) {
      iter = input_list.begin();
    }
    result_list.push_back(*iter);
  }
}

template <class T>
void scheduler_unit::order_rrr(
    std::vector<T> &result_list, const typename std::vector<T> &input_list,
    const typename std::vector<T>::const_iterator &last_issued_from_input,
    unsigned num_warps_to_add) {
  result_list.clear();

  if (m_num_issued_last_cycle > 0 || warp(m_current_turn_warp).done_exit() ||
      warp(m_current_turn_warp).waiting()) {
    std::vector<shd_warp_t *>::const_iterator iter =
        (last_issued_from_input == input_list.end())
            ? input_list.begin()
            : last_issued_from_input + 1;
    for (unsigned count = 0; count < num_warps_to_add; ++iter, ++count) {
      if (iter == input_list.end()) {
        iter = input_list.begin();
      }
      unsigned warp_id = (*iter)->get_warp_id();
      if (!(*iter)->done_exit() && !(*iter)->waiting()) {
        result_list.push_back(*iter);
        m_current_turn_warp = warp_id;
        break;
      }
    }
  } else {
    result_list.push_back(&warp(m_current_turn_warp));
  }
}
/**
 * A general function to order things in an priority-based way.
 * The core usage of the function is similar to order_lrr.
 * The explanation of the additional parameters (beyond order_lrr) explains the
 * further extensions.
 * @param ordering: An enum that determines how the age function will be treated
 * in prioritization see the definition of OrderingType.
 * @param priority_function: This function is used to sort the input_list.  It
 * is passed to stl::sort as the sorting fucntion. So, if you wanted to sort a
 * list of integer warp_ids with the oldest warps having the most priority, then
 * the priority_function would compare the age of the two warps.
 */
template <class T>
void scheduler_unit::order_by_priority(
    std::vector<T> &result_list, const typename std::vector<T> &input_list,
    const typename std::vector<T>::const_iterator &last_issued_from_input,
    unsigned num_warps_to_add, OrderingType ordering,
    bool (*priority_func)(T lhs, T rhs)) {
  assert(num_warps_to_add <= input_list.size());
  result_list.clear();
  typename std::vector<T> temp = input_list;

  if (ORDERING_GREEDY_THEN_PRIORITY_FUNC == ordering) {
    T greedy_value = *last_issued_from_input;
    result_list.push_back(greedy_value);

    std::sort(temp.begin(), temp.end(), priority_func);
    typename std::vector<T>::iterator iter = temp.begin();
    for (unsigned count = 0; count < num_warps_to_add; ++count, ++iter) {
      if (*iter != greedy_value) {
        result_list.push_back(*iter);
      }
    }
  } else if (ORDERED_PRIORITY_FUNC_ONLY == ordering) {
    std::sort(temp.begin(), temp.end(), priority_func);
    typename std::vector<T>::iterator iter = temp.begin();
    for (unsigned count = 0; count < num_warps_to_add; ++count, ++iter) {
      result_list.push_back(*iter);
    }
  } else {
    fprintf(stderr, "Unknown ordering - %d\n", ordering);
    abort();
  }
}

/*
Shader Core��ĵ�����������Ԫ��ǰ�ƽ�һ�ģ�ִ��scheduler_unit::cycle()��
*/
void scheduler_unit::cycle() {
  SCHED_DPRINTF("scheduler_unit::cycle()\n");
  // These three flags match the valid, ready, and issued state of warps in
  // the scheduler.
  //ibuffer��ȡ����PCֵ��SIMT��ջ�е�PCֵƥ�䣬��˵��û�п���ð�գ�����Ϊ��
  bool valid_inst =
      false;  // there was one warp with a valid instruction to issue (didn't
              // require flush due to control hazard)
  //ָ��ͨ���Ƿְ��飬�Ϳ��Խ�ָ��ready״̬����Ϊtrue��
  bool ready_inst = false;   // of the valid instructions, there was one not
                             // waiting for pending register writes
  //ָ���ɹ�������issued_instΪ�档
  bool issued_inst = false;  // of these we issued one

  //warp�Ǹ���ĳЩ������������ģ����ǲ�ͬ������֮�����Ҫ���𡣶���V100��˵������LRR���Ȳ��ԣ�
  //LRR���Ȳ��Եĵ�������Ԫ��order_warps()������Ϊ��ǰ���ȵ�Ԫ�������ֵ���warp������������
  //�ٵ��ã�
  //order_lrr(m_next_cycle_prioritized_warps, m_supervised_warps,
  //          m_last_supervised_issued, m_supervised_warps.size());
  //������б�Ϊ��
  //    result_list��m_next_cycle_prioritized_warps��һ��vector������洢��ǰ���ȵ�Ԫ��ǰ��
  //                 ����warp���������һ�ľ������ȼ����ȵ�warp��
  //    input_list��m_supervised_warps����һ��vector������洢��ǰ���ȵ�Ԫ����Ҫ�ٲõ�warp��
  //    last_issued_from_input����洢�˵�ǰ���ȵ�Ԫ��һ�ĵ��ȹ���warp��
  //    num_warps_to_add��m_supervised_warps.size()�����ǵ�ǰ���ȵ�Ԫ����һ����Ҫ���ȵ�warp
  //                      ��Ŀ�����������warp��Ŀ���ǵ�ǰ�����������ֵ���warp�Ӽ��ϵĴ�С��
  //��������Ĺ��ܾ��Ǹ�����һ�ĵ��ȹ���warp���ҵ����ڵ�ǰ���ȵ�Ԫ����Ҫ�ٲõ�warp�����е�λ�ã�
  //Ȼ������λ�ú���ĵ�һ��warp��ʼ��һֱ������ǰ���ȵ�Ԫ����Ҫ�ٲõ�warp���ϣ�������Щwarp��
  //��result_list�У�ֱ��result_list�е�warp��Ŀ����num_warps_to_add��
  order_warps();
  //Loop through all the warps based on the order
  //m_next_cycle_prioritized_warps��洢����������һ��Ӧ���ȵ��ȵ�warp˳�򣬱����������ȼ���
  //warp�б����ν��е��ȡ�
  for (std::vector<shd_warp_t *>::const_iterator iter =
           m_next_cycle_prioritized_warps.begin();
       iter != m_next_cycle_prioritized_warps.end(); iter++) {
    // Don't consider warps that are not yet valid
    //���warp������Ч�ģ���û����Ч��ָ�����warp�Ѿ�ִ����ϣ����������ȸ�warp��
    if ((*iter) == NULL || (*iter)->done_exit()) {
      continue;
    }
    SCHED_DPRINTF("Testing (warp_id %u, dynamic_warp_id %u)\n",
                  (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id());
    unsigned warp_id = (*iter)->get_warp_id();
    unsigned checked = 0;
    //��ǰwarp�з����ָ�����ֵ��
    unsigned issued = 0;
    //��ǰwarp�м�¼��һ�η����ָ���ִ�е�Ԫ���͡�
    exec_unit_type_t previous_issued_inst_exec_type = exec_unit_type_t::NONE;
    //ÿ��warp���ε������ָ��������V100����������Ϊ1��
    unsigned max_issue = m_shader->m_config->gpgpu_max_insn_issue_per_warp;
    //��������ͬ��Ӳ����Ԫ˫���䣬��V100����������Ϊ1��diff_exec_units�ǽ�������ͬ��Ӳ����
    //Ԫ˫���䣬��V100����������Ϊ1�����������ͬһ��ͬһ��warp������������ͬһӲ����Ԫ��������
    //������ָ���������Ҫ�ж�һ�µ�ǰ��warp�Ƿ���һ���Ǵ洢ָ����ǵĻ��ſɼ������䡣
    bool diff_exec_units =
        m_shader->m_config
            ->gpgpu_dual_issue_diff_exec_units;  // In tis mode, we only allow
                                                 // dual issue to diff execution
                                                 // units (as in Maxwell and
                                                 // Pascal)

    //����I-Bufer�Ƿ�Ϊ�ա�����һ��warp��һ��I-Bufer��I-Bufer��һ�����У��洢�˵�ǰwarp�еĴ�
    //ִ��ָ�
    if (warp(warp_id).ibuffer_empty())
      SCHED_DPRINTF(
          "Warp (warp_id %u, dynamic_warp_id %u) fails as ibuffer_empty\n",
          (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id());

    //����warp�Ƿ����ڣ�warp�Ѿ�ִ��������ڵȴ����ں˳�ʼ����CTA����barrier��memory barrier��
    //����δ��ɵ�ԭ�Ӳ������ĸ��������ڵȴ�״̬��
    if (warp(warp_id).waiting())
      SCHED_DPRINTF(
          "Warp (warp_id %u, dynamic_warp_id %u) fails as waiting for "
          "barrier\n",
          (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id());

    //checked������ѭ����ѭ�����������ڵ�ǰ�ɵ��ȵ�warp�£�ִ�м�����warp�Ŀɷ���ָ��������Ϊ
    //max_issue����V100������Ϊ1�����������ѭ������û�н�ָ����ȥ���������ٽ��еڶ���ѭ����
    //��Ϊ����warp������Ϊÿ���������һ��ָ�issued�ǵ�ǰwarp�м�¼�ķ����ָ�����ֵ����ֵ��
    //�ܳ�����ǰwarp�����ɷ���ָ����max_issue��ͬʱ��checked�������뱣֤С�ڵ���issued����Ϊ
    //һ��checked��������issued����˵�����Ѿ�������ָ���У����һ��ָ����Ϊĳ��ԭ��û�з����
    //������ʱ�����Ǿ�Ҫ��ͣ��ǰwarp�ĵ��ȣ��Ա�ָ֤��ִ�е���ȷ�ԡ�
    while (!warp(warp_id).waiting() && !warp(warp_id).ibuffer_empty() &&
           (checked < max_issue) && (checked <= issued) &&
           (issued < max_issue)) {
      //��warp_id�����warp����ȡ��ibuffer�е���һ��ָ�
      const warp_inst_t *pI = warp(warp_id).ibuffer_next_inst();
      // Jin: handle cdp latency;
      if (pI && pI->m_is_cdp && warp(warp_id).m_cdp_latency > 0) {
        assert(warp(warp_id).m_cdp_dummy);
        warp(warp_id).m_cdp_latency--;
        break;
      }
      //��ȡibuffer�е���һ��ָ����ո�ȡ����ָ��pI�Ƿ���Ч��
      bool valid = warp(warp_id).ibuffer_next_valid();
      //��־λ��ָʾwarp�Ƿ�����ָ�
      bool warp_inst_issued = false;
      //warp_id��Ӧ��SIMT��ջ������PCֵ��RPCֵ��
      unsigned pc, rpc;
      //ÿ��warp���Լ���SIMT��ջ����ȡwarp_id��Ӧ��SIMT��ջ������PCֵ��RPCֵ��
      m_shader->get_pdom_stack_top_info(warp_id, pI, &pc, &rpc);
      SCHED_DPRINTF(
          "Warp (warp_id %u, dynamic_warp_id %u) has valid instruction (%s)\n",
          (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id(),
          m_shader->m_config->gpgpu_ctx->func_sim->ptx_get_insn_str(pc)
              .c_str());
      //pI�Ǵ�ibufferȡ����ָ������ָ����Ч��
      if (pI) {
        assert(valid);
        //�����֮��������������һ��������Чibuffer�е�warp�������ǵȴ��ϰ�����ȡwarp��
        //��ibuffer��ȡָ�������Ƿ���Ч��������Чָ������pc�뵱ǰSIMT��ջ��pc��ƥ
        //�䣬����ζ�ŷ����˿���Σ�գ�����ibuffer��ˢ�¡�Ȼ������Դ�Ĵ�����Ŀ��Ĵ�����
        //���ݵ��Ƿְ���г�ͻ��顣�����Ҳͨ���˼Ƿְ壬���Ŀ�깦�ܵ�Ԫ��ID_OC��ˮ�߼Ĵ�
        //�����Ƿ��п��в�ۡ�����У�����Է���ָ�ѭ����inital���жϡ���������ǰwarp
        //�е�ָ��δ������������һ��warp����ˣ�ÿ�����ȳ���Ԫÿ������ֻ����һ��ָ�
        if (pc != pI->pc) {
          SCHED_DPRINTF(
              "Warp (warp_id %u, dynamic_warp_id %u) control hazard "
              "instruction flush\n",
              (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id());
          // control hazard
          //��warp��һ��ִ�е�ָ��PCֵ����Ϊ��SIMT��ջ��ȡ����PC��
          warp(warp_id).set_next_pc(pc);
          //ˢ��warp��ibuffer����Ϊibuffer�˿����е�ָ���Ѿ�������ִ�С�
          warp(warp_id).ibuffer_flush();
        } else {
          //���ibuffer��ȡ����PCֵ��SIMT��ջ�е�PCֵƥ�䣬��˵��û�п���ð�գ�����Ϊ�档
          valid_inst = true;
          //���Ʒְ��ð�գ����ĳ��ָ��ʹ�õļĴ����Ƿ񱻱����ڼǷְ��У�����еĻ�����
          //������ WAW �� RAW ð�գ��򲻷������ָ�
          if (!m_scoreboard->checkCollision(warp_id, pI)) {
            SCHED_DPRINTF(
                "Warp (warp_id %u, dynamic_warp_id %u) passes scoreboard\n",
                (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id());
            //һ��ָ��ͨ���Ƿְ��飬�Ϳ��Խ�ָ��ready״̬����Ϊtrue��
            ready_inst = true;
            //��ȡwarp_id��Ӧ��ָ��pI�Ļ�Ծ���롣
            const active_mask_t &active_mask =
                m_shader->get_active_mask(warp_id, pI);

            assert(warp(warp_id).inst_in_pipeline());

            //MEM��ص�ָ���Ҫ���͵�m_mem_out��ˮ�߼Ĵ�����
            if ((pI->op == LOAD_OP) || (pI->op == STORE_OP) ||
                (pI->op == MEMORY_BARRIER_OP) ||
                (pI->op == TENSOR_CORE_LOAD_OP) ||
                (pI->op == TENSOR_CORE_STORE_OP)) {
              //m_id�ǵ�������Ԫ��ID��m_mem_out�ǵ�������Ԫ��m_mem_out��ˮ�߼Ĵ�������
              //��m_mem_out��ˮ�߼Ĵ����п��в�ۣ��Ҳ�֧����ͬ��Ӳ����Ԫ˫���䣬������
              //����ͬ��Ӳ����Ԫ˫���䣬������һ��ָ���ִ�е�Ԫ���Ͳ���MEM��أ������
              //���䡣��subcoreģʽ�£�ÿ��warp�������ڼĴ�����������һ������ļĴ����ɹ�
              //ʹ�ã�����Ĵ����ɵ�������m_id�������������ȥ����m_mem_out�Ĵ��������е�
              //m_id�ŵ���������ʹ�õĵ�m_id�żĴ����Ƿ�Ϊ�տ��ã�ͬʱ��diff_exec_units
              //�ǽ�������ͬ��Ӳ����Ԫ˫���䣬��V100����������Ϊ1�����������ͬһ��ͬһ
              //��warp������������ͬһӲ����Ԫ��������������ָ���������Ҫ�ж�һ�µ�ǰ��
              //warp�Ƿ���һ���Ǵ洢ָ����ǵĻ��ſɼ������䡣
              if (m_mem_out->has_free(m_shader->m_config->sub_core_model,
                                      m_id) &&
                  (!diff_exec_units ||
                   previous_issued_inst_exec_type != exec_unit_type_t::MEM)) {
                //��m_mem_out��ˮ�߼Ĵ�������ָ��pI��
                m_shader->issue_warp(*m_mem_out, pI, active_mask, warp_id,
                                     m_id);
                //�����ָ�����ֵ��һ��
                issued++;
                //ָ���ɹ�������issued_instΪ�档
                issued_inst = true;
                warp_inst_issued = true;
                previous_issued_inst_exec_type = exec_unit_type_t::MEM;
              }
            } else {
              // This code need to be refactored
              if (pI->op != TENSOR_CORE_OP && pI->op != SFU_OP &&
                  pI->op != DP_OP && !(pI->op >= SPEC_UNIT_START_ID)) {
                bool execute_on_SP = false;
                bool execute_on_INT = false;

                bool sp_pipe_avail =
                    (m_shader->m_config->gpgpu_num_sp_units > 0) &&
                    m_sp_out->has_free(m_shader->m_config->sub_core_model,
                                       m_id);
                bool int_pipe_avail =
                    (m_shader->m_config->gpgpu_num_int_units > 0) &&
                    m_int_out->has_free(m_shader->m_config->sub_core_model,
                                        m_id);

                // if INT unit pipline exist, then execute ALU and INT
                // operations on INT unit and SP-FPU on SP unit (like in Volta)
                // if INT unit pipline does not exist, then execute all ALU, INT
                // and SP operations on SP unit (as in Fermi, Pascal GPUs)
                if (m_shader->m_config->gpgpu_num_int_units > 0 &&
                    int_pipe_avail && pI->op != SP_OP &&
                    !(diff_exec_units &&
                      previous_issued_inst_exec_type == exec_unit_type_t::INT))
                  execute_on_INT = true;
                else if (sp_pipe_avail &&
                         (m_shader->m_config->gpgpu_num_int_units == 0 ||
                          (m_shader->m_config->gpgpu_num_int_units > 0 &&
                           pI->op == SP_OP)) &&
                         !(diff_exec_units && previous_issued_inst_exec_type ==
                                                  exec_unit_type_t::SP))
                  execute_on_SP = true;

                if (execute_on_INT || execute_on_SP) {
                  // Jin: special for CDP api
                  if (pI->m_is_cdp && !warp(warp_id).m_cdp_dummy) {
                    assert(warp(warp_id).m_cdp_latency == 0);

                    if (pI->m_is_cdp == 1)
                      warp(warp_id).m_cdp_latency =
                          m_shader->m_config->gpgpu_ctx->func_sim
                              ->cdp_latency[pI->m_is_cdp - 1];
                    else  // cudaLaunchDeviceV2 and cudaGetParameterBufferV2
                      warp(warp_id).m_cdp_latency =
                          m_shader->m_config->gpgpu_ctx->func_sim
                              ->cdp_latency[pI->m_is_cdp - 1] +
                          m_shader->m_config->gpgpu_ctx->func_sim
                                  ->cdp_latency[pI->m_is_cdp] *
                              active_mask.count();
                    warp(warp_id).m_cdp_dummy = true;
                    break;
                  } else if (pI->m_is_cdp && warp(warp_id).m_cdp_dummy) {
                    assert(warp(warp_id).m_cdp_latency == 0);
                    warp(warp_id).m_cdp_dummy = false;
                  }
                }

                if (execute_on_SP) {
                  m_shader->issue_warp(*m_sp_out, pI, active_mask, warp_id,
                                       m_id);
                  issued++;
                  issued_inst = true;
                  warp_inst_issued = true;
                  previous_issued_inst_exec_type = exec_unit_type_t::SP;
                } else if (execute_on_INT) {
                  m_shader->issue_warp(*m_int_out, pI, active_mask, warp_id,
                                       m_id);
                  issued++;
                  issued_inst = true;
                  warp_inst_issued = true;
                  previous_issued_inst_exec_type = exec_unit_type_t::INT;
                }
              } else if ((m_shader->m_config->gpgpu_num_dp_units > 0) &&
                         (pI->op == DP_OP) &&
                         !(diff_exec_units && previous_issued_inst_exec_type ==
                                                  exec_unit_type_t::DP)) {
                bool dp_pipe_avail =
                    (m_shader->m_config->gpgpu_num_dp_units > 0) &&
                    m_dp_out->has_free(m_shader->m_config->sub_core_model,
                                       m_id);

                if (dp_pipe_avail) {
                  m_shader->issue_warp(*m_dp_out, pI, active_mask, warp_id,
                                       m_id);
                  issued++;
                  issued_inst = true;
                  warp_inst_issued = true;
                  previous_issued_inst_exec_type = exec_unit_type_t::DP;
                }
              }  // If the DP units = 0 (like in Fermi archi), then execute DP
                 // inst on SFU unit
              else if (((m_shader->m_config->gpgpu_num_dp_units == 0 &&
                         pI->op == DP_OP) ||
                        (pI->op == SFU_OP) || (pI->op == ALU_SFU_OP)) &&
                       !(diff_exec_units && previous_issued_inst_exec_type ==
                                                exec_unit_type_t::SFU)) {
                bool sfu_pipe_avail =
                    (m_shader->m_config->gpgpu_num_sfu_units > 0) &&
                    m_sfu_out->has_free(m_shader->m_config->sub_core_model,
                                        m_id);

                if (sfu_pipe_avail) {
                  m_shader->issue_warp(*m_sfu_out, pI, active_mask, warp_id,
                                       m_id);
                  issued++;
                  issued_inst = true;
                  warp_inst_issued = true;
                  previous_issued_inst_exec_type = exec_unit_type_t::SFU;
                }
              } else if ((pI->op == TENSOR_CORE_OP) &&
                         !(diff_exec_units && previous_issued_inst_exec_type ==
                                                  exec_unit_type_t::TENSOR)) {
                bool tensor_core_pipe_avail =
                    (m_shader->m_config->gpgpu_num_tensor_core_units > 0) &&
                    m_tensor_core_out->has_free(
                        m_shader->m_config->sub_core_model, m_id);

                if (tensor_core_pipe_avail) {
                  m_shader->issue_warp(*m_tensor_core_out, pI, active_mask,
                                       warp_id, m_id);
                  issued++;
                  issued_inst = true;
                  warp_inst_issued = true;
                  previous_issued_inst_exec_type = exec_unit_type_t::TENSOR;
                }
              } else if ((pI->op >= SPEC_UNIT_START_ID) &&
                         !(diff_exec_units &&
                           previous_issued_inst_exec_type ==
                               exec_unit_type_t::SPECIALIZED)) {
                unsigned spec_id = pI->op - SPEC_UNIT_START_ID;
                assert(spec_id < m_shader->m_config->m_specialized_unit.size());
                register_set *spec_reg_set = m_spec_cores_out[spec_id];
                bool spec_pipe_avail =
                    (m_shader->m_config->m_specialized_unit[spec_id].num_units >
                     0) &&
                    spec_reg_set->has_free(m_shader->m_config->sub_core_model,
                                           m_id);

                if (spec_pipe_avail) {
                  m_shader->issue_warp(*spec_reg_set, pI, active_mask, warp_id,
                                       m_id);
                  issued++;
                  issued_inst = true;
                  warp_inst_issued = true;
                  previous_issued_inst_exec_type =
                      exec_unit_type_t::SPECIALIZED;
                }
              }

            }  // end of else
          } else {
            SCHED_DPRINTF(
                "Warp (warp_id %u, dynamic_warp_id %u) fails scoreboard\n",
                (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id());
          }
        }
      } else if (valid) {
        // this case can happen after a return instruction in diverged warp
        SCHED_DPRINTF(
            "Warp (warp_id %u, dynamic_warp_id %u) return from diverged warp "
            "flush\n",
            (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id());
        warp(warp_id).set_next_pc(pc);
        warp(warp_id).ibuffer_flush();
      }
      if (warp_inst_issued) {
        SCHED_DPRINTF(
            "Warp (warp_id %u, dynamic_warp_id %u) issued %u instructions\n",
            (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id(), issued);
        do_on_warp_issued(warp_id, issued, iter);
      }
      checked++;
    }
    if (issued) {
      // This might be a bit inefficient, but we need to maintain
      // two ordered list for proper scheduler execution.
      // We could remove the need for this loop by associating a
      // supervised_is index with each entry in the
      // m_next_cycle_prioritized_warps vector. For now, just run through until
      // you find the right warp_id
      for (std::vector<shd_warp_t *>::const_iterator supervised_iter =
               m_supervised_warps.begin();
           supervised_iter != m_supervised_warps.end(); ++supervised_iter) {
        if (*iter == *supervised_iter) {
          //m_last_supervised_issued��ָ����һ�ε��ȵ�warp��
          m_last_supervised_issued = supervised_iter;
        }
      }
      //��¼��һ�ķ����ָ������
      m_num_issued_last_cycle = issued;
      if (issued == 1)
        m_stats->single_issue_nums[m_id]++;
      else if (issued > 1)
        m_stats->dual_issue_nums[m_id]++;
      else
        abort();  // issued should be > 0

      break;
    }
  }

  // issue stall statistics:
  //ibuffer��ȡ����PCֵ��SIMT��ջ�е�PCֵƥ�䣬��˵��û�п���ð�գ�����Ϊ�棺
  //    bool valid_inst=False˵��Idle�����ɿ���ð�ա�
  //ָ��ͨ���Ƿְ��飬�Ϳ��Խ�ָ��ready״̬����Ϊtrue��
  //    bool ready_inst=False˵���ȴ�RAWð�գ������������ڴ棩��
  //ָ���ɹ�������issued_instΪ�棺
  //    bool issued_inst=False˵����ˮ��ͣ�١�
  if (!valid_inst)
    m_stats->shader_cycle_distro[0]++;  // idle or control hazard
  else if (!ready_inst)
    m_stats->shader_cycle_distro[1]++;  // waiting for RAW hazards (possibly due
                                        // to memory)
  else if (!issued_inst)
    m_stats->shader_cycle_distro[2]++;  // pipeline stalled
}

void scheduler_unit::do_on_warp_issued(
    unsigned warp_id, unsigned num_issued,
    const std::vector<shd_warp_t *>::const_iterator &prioritized_iter) {
  m_stats->event_warp_issued(m_shader->get_sid(), warp_id, num_issued,
                             warp(warp_id).get_dynamic_warp_id());
  warp(warp_id).ibuffer_step();
}

bool scheduler_unit::sort_warps_by_oldest_dynamic_id(shd_warp_t *lhs,
                                                     shd_warp_t *rhs) {
  if (rhs && lhs) {
    if (lhs->done_exit() || lhs->waiting()) {
      return false;
    } else if (rhs->done_exit() || rhs->waiting()) {
      return true;
    } else {
      return lhs->get_dynamic_warp_id() < rhs->get_dynamic_warp_id();
    }
  } else {
    return lhs < rhs;
  }
}

/*
LRR���Ȳ��Եĵ�������Ԫ��order_warps()������Ϊ��ǰ���ȵ�Ԫ�������ֵ���warp��������order_lrr
�Ķ���Ϊ��
    void scheduler_unit::order_lrr(
        std::vector<T> &result_list, const typename std::vector<T> &input_list,
        const typename std::vector<T>::const_iterator &last_issued_from_input,
        unsigned num_warps_to_add)
�����￴����m_next_cycle_prioritized_warps��һ��vector������洢�˵�ǰ���ȵ�Ԫ��ǰ�ľ���warp
���������һ�ľ������ȼ����ȵ�warp��last_issued_from_input��洢�˵�ǰ���ȵ�Ԫ��һ�ĵ��ȹ���
warp��num_warps_to_add���ǵ�ǰ���ȵ�Ԫ����һ����Ҫ���ȵ�warp��Ŀ�����������warp��Ŀ���ǵ�ǰ��
���������ֵ���warp�Ӽ���m_supervised_warps�Ĵ�С��
*/
void lrr_scheduler::order_warps() {
  order_lrr(m_next_cycle_prioritized_warps, m_supervised_warps,
            m_last_supervised_issued, m_supervised_warps.size());
}
void rrr_scheduler::order_warps() {
  order_rrr(m_next_cycle_prioritized_warps, m_supervised_warps,
            m_last_supervised_issued, m_supervised_warps.size());
}

void gto_scheduler::order_warps() {
  order_by_priority(m_next_cycle_prioritized_warps, m_supervised_warps,
                    m_last_supervised_issued, m_supervised_warps.size(),
                    ORDERING_GREEDY_THEN_PRIORITY_FUNC,
                    scheduler_unit::sort_warps_by_oldest_dynamic_id);
}

void oldest_scheduler::order_warps() {
  order_by_priority(m_next_cycle_prioritized_warps, m_supervised_warps,
                    m_last_supervised_issued, m_supervised_warps.size(),
                    ORDERED_PRIORITY_FUNC_ONLY,
                    scheduler_unit::sort_warps_by_oldest_dynamic_id);
}

void two_level_active_scheduler::do_on_warp_issued(
    unsigned warp_id, unsigned num_issued,
    const std::vector<shd_warp_t *>::const_iterator &prioritized_iter) {
  scheduler_unit::do_on_warp_issued(warp_id, num_issued, prioritized_iter);
  if (SCHEDULER_PRIORITIZATION_LRR == m_inner_level_prioritization) {
    std::vector<shd_warp_t *> new_active;
    order_lrr(new_active, m_next_cycle_prioritized_warps, prioritized_iter,
              m_next_cycle_prioritized_warps.size());
    m_next_cycle_prioritized_warps = new_active;
  } else {
    fprintf(stderr, "Unimplemented m_inner_level_prioritization: %d\n",
            m_inner_level_prioritization);
    abort();
  }
}

void two_level_active_scheduler::order_warps() {
  // Move waiting warps to m_pending_warps
  unsigned num_demoted = 0;
  for (std::vector<shd_warp_t *>::iterator iter =
           m_next_cycle_prioritized_warps.begin();
       iter != m_next_cycle_prioritized_warps.end();) {
    bool waiting = (*iter)->waiting();
    for (int i = 0; i < MAX_INPUT_VALUES; i++) {
      const warp_inst_t *inst = (*iter)->ibuffer_next_inst();
      // Is the instruction waiting on a long operation?
      if (inst && inst->in[i] > 0 &&
          this->m_scoreboard->islongop((*iter)->get_warp_id(), inst->in[i])) {
        waiting = true;
      }
    }

    if (waiting) {
      m_pending_warps.push_back(*iter);
      iter = m_next_cycle_prioritized_warps.erase(iter);
      SCHED_DPRINTF("DEMOTED warp_id=%d, dynamic_warp_id=%d\n",
                    (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id());
      ++num_demoted;
    } else {
      ++iter;
    }
  }

  // If there is space in m_next_cycle_prioritized_warps, promote the next
  // m_pending_warps
  unsigned num_promoted = 0;
  if (SCHEDULER_PRIORITIZATION_SRR == m_outer_level_prioritization) {
    while (m_next_cycle_prioritized_warps.size() < m_max_active_warps) {
      m_next_cycle_prioritized_warps.push_back(m_pending_warps.front());
      m_pending_warps.pop_front();
      SCHED_DPRINTF(
          "PROMOTED warp_id=%d, dynamic_warp_id=%d\n",
          (m_next_cycle_prioritized_warps.back())->get_warp_id(),
          (m_next_cycle_prioritized_warps.back())->get_dynamic_warp_id());
      ++num_promoted;
    }
  } else {
    fprintf(stderr, "Unimplemented m_outer_level_prioritization: %d\n",
            m_outer_level_prioritization);
    abort();
  }
  assert(num_promoted == num_demoted);
}

swl_scheduler::swl_scheduler(shader_core_stats *stats, shader_core_ctx *shader,
                             Scoreboard *scoreboard, simt_stack **simt,
                             std::vector<shd_warp_t *> *warp,
                             register_set *sp_out, register_set *dp_out,
                             register_set *sfu_out, register_set *int_out,
                             register_set *tensor_core_out,
                             std::vector<register_set *> &spec_cores_out,
                             register_set *mem_out, int id, char *config_string)
    : scheduler_unit(stats, shader, scoreboard, simt, warp, sp_out, dp_out,
                     sfu_out, int_out, tensor_core_out, spec_cores_out, mem_out,
                     id) {
  unsigned m_prioritization_readin;
  int ret = sscanf(config_string, "warp_limiting:%d:%d",
                   &m_prioritization_readin, &m_num_warps_to_limit);
  assert(2 == ret);
  m_prioritization = (scheduler_prioritization_type)m_prioritization_readin;
  // Currently only GTO is implemented
  assert(m_prioritization == SCHEDULER_PRIORITIZATION_GTO);
  assert(m_num_warps_to_limit <= shader->get_config()->max_warps_per_shader);
}

void swl_scheduler::order_warps() {
  if (SCHEDULER_PRIORITIZATION_GTO == m_prioritization) {
    order_by_priority(m_next_cycle_prioritized_warps, m_supervised_warps,
                      m_last_supervised_issued,
                      MIN(m_num_warps_to_limit, m_supervised_warps.size()),
                      ORDERING_GREEDY_THEN_PRIORITY_FUNC,
                      scheduler_unit::sort_warps_by_oldest_dynamic_id);
  } else {
    fprintf(stderr, "swl_scheduler m_prioritization = %d\n", m_prioritization);
    abort();
  }
}

/*
ģ��������ռ����ӼĴ����ļ���ȡָ���Դ����������ԭ���ݴ����ռ�����Ԫָ���m_warp�е�ָ���Ƴ���
m_output_register�С�
*/
void shader_core_ctx::read_operands() {
  //m_config->reg_file_port_throughput�ǼĴ����ļ��Ķ˿�������V100�����ļ���gpgpu_reg_file_
  //port_throughput������Ϊ2��
  for (unsigned int i = 0; i < m_config->reg_file_port_throughput; ++i)
    m_operand_collector.step();
}

address_type coalesced_segment(address_type addr,
                               unsigned segment_size_lg2bytes) {
  return (addr >> segment_size_lg2bytes);
}

// Returns numbers of addresses in translated_addrs, each addr points to a 4B
// (32-bit) word
unsigned shader_core_ctx::translate_local_memaddr(
    address_type localaddr, unsigned tid, unsigned num_shader,
    unsigned datasize, new_addr_type *translated_addrs) {
  // During functional execution, each thread sees its own memory space for
  // local memory, but these need to be mapped to a shared address space for
  // timing simulation.  We do that mapping here.

  address_type thread_base = 0;
  unsigned max_concurrent_threads = 0;
  if (m_config->gpgpu_local_mem_map) {
    // Dnew = D*N + T%nTpC + nTpC*C
    // N = nTpC*nCpS*nS (max concurent threads)
    // C = nS*K + S (hw cta number per gpu)
    // K = T/nTpC   (hw cta number per core)
    // D = data index
    // T = thread
    // nTpC = number of threads per CTA
    // nCpS = number of CTA per shader
    //
    // for a given local memory address threads in a CTA map to contiguous
    // addresses, then distribute across memory space by CTAs from successive
    // shader cores first, then by successive CTA in same shader core
    thread_base =
        4 * (kernel_padded_threads_per_cta *
                 (m_sid + num_shader * (tid / kernel_padded_threads_per_cta)) +
             tid % kernel_padded_threads_per_cta);
    max_concurrent_threads =
        kernel_padded_threads_per_cta * kernel_max_cta_per_shader * num_shader;
  } else {
    // legacy mapping that maps the same address in the local memory space of
    // all threads to a single contiguous address region
    thread_base = 4 * (m_config->n_thread_per_shader * m_sid + tid);
    max_concurrent_threads = num_shader * m_config->n_thread_per_shader;
  }
  assert(thread_base < 4 /*word size*/ * max_concurrent_threads);

  // If requested datasize > 4B, split into multiple 4B accesses
  // otherwise do one sub-4 byte memory access
  unsigned num_accesses = 0;

  if (datasize >= 4) {
    // >4B access, split into 4B chunks
    assert(datasize % 4 == 0);  // Must be a multiple of 4B
    num_accesses = datasize / 4;
    assert(num_accesses <= MAX_ACCESSES_PER_INSN_PER_THREAD);  // max 32B
    assert(
        localaddr % 4 ==
        0);  // Address must be 4B aligned - required if accessing 4B per
             // request, otherwise access will overflow into next thread's space
    for (unsigned i = 0; i < num_accesses; i++) {
      address_type local_word = localaddr / 4 + i;
      address_type linear_address = local_word * max_concurrent_threads * 4 +
                                    thread_base + LOCAL_GENERIC_START;
      translated_addrs[i] = linear_address;
    }
  } else {
    // Sub-4B access, do only one access
    assert(datasize > 0);
    num_accesses = 1;
    address_type local_word = localaddr / 4;
    address_type local_word_offset = localaddr % 4;
    assert((localaddr + datasize - 1) / 4 ==
           local_word);  // Make sure access doesn't overflow into next 4B chunk
    address_type linear_address = local_word * max_concurrent_threads * 4 +
                                  local_word_offset + thread_base +
                                  LOCAL_GENERIC_START;
    translated_addrs[0] = linear_address;
  }
  return num_accesses;
}

/////////////////////////////////////////////////////////////////////////////////////////
/*
This function locates a free slot in all the result buses (the slot is free if its bit 
is not set).
�˺��������н�������ж�λһ�����в�ۣ����δ������λ����ò��Ϊ���в�ۣ���
*/
int shader_core_ctx::test_res_bus(int latency) {
  //������߹���m_config->pipe_widths[EX_WB]����
  //��ˮ�߽׶εĿ��������-gpgpu_pipeline_widths�����ã�
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
  //��V100������Ϊ��-gpgpu_pipeline_widths 4,4,4,4,4,4,4,4,4,4,8,4,4
  //������ߵĿ����1����num_result_bus = m_config->pipe_widths[EX_WB] = 8��
  for (unsigned i = 0; i < num_result_bus; i++) {
    if (!m_result_bus[i]->test(latency)) {
      return i;
    }
  }
  return -1;
}

/*
SM��ִ�У������ܵ�Ԫ��ǰ�ƽ�һ�ġ�
*/
void shader_core_ctx::execute() {
  //������߹���m_config->pipe_widths[EX_WB]����
  //��ˮ�߽׶εĿ��������-gpgpu_pipeline_widths�����ã�
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
  //��V100������Ϊ��-gpgpu_pipeline_widths 4,4,4,4,4,4,4,4,4,4,8,4,4
  //������ߵĿ����1����num_result_bus = m_config->pipe_widths[EX_WB] = 8��
  for (unsigned i = 0; i < num_result_bus; i++) {
    *(m_result_bus[i]) >>= 1;
  }
  for (unsigned n = 0; n < m_num_function_units; n++) {
    //m_fu��SIMD���ܵ�Ԫ��������m_num_function_units��SIMD���ܵ�Ԫ��������m_fu������
    //  4��SP��Ԫ��4��DP��Ԫ��4��INT��Ԫ��4��SFU��Ԫ��4��TC��Ԫ����������specialized_unit��
    //  1��LD/ST��Ԫ��
    //��V100�����У�LDST��Ԫ�Լ�����SIMD��Ԫ��m_fu[n]->clock_multiplier()������1��һЩ��Ԫ��
    //���������������ļ��LDST��Ԫ�����Ը��ߵ�ʱ��Ƶ�����У����m_fu[n]->clock_multiplier()
    //���ܷ���2���������SM��ʱ������ǰ�ƽ�һ��ʱ��LDST��Ԫ����ǰ�ƽ����ġ�
    unsigned multiplier = m_fu[n]->clock_multiplier();
    //m_fu[n]��Ԫ��ǰ�ƽ�һ�ġ�����pipelined_simd_unit::cycle()����ˮ�ߵ�Ԫ��ǰ�ƽ�һ�ġ�
    for (unsigned c = 0; c < multiplier; c++) m_fu[n]->cycle();
    //����m_state��һЩ���ܼ�������
    m_fu[n]->active_lanes_in_pipeline();
    //m_issue_port�Ķ������£�
    //    for (unsigned k = 0; k < m_config->gpgpu_num_sp_units; k++)
    //      m_issue_port.push_back(OC_EX_SP);
    //    for (unsigned k = 0; k < m_config->gpgpu_num_dp_units; k++)
    //      m_issue_port.push_back(OC_EX_DP);
    //    for (unsigned k = 0; k < m_config->gpgpu_num_int_units; k++)
    //      m_issue_port.push_back(OC_EX_INT);
    //    for (unsigned k = 0; k < m_config->gpgpu_num_sfu_units; k++)
    //      m_issue_port.push_back(OC_EX_SFU);
    //    for (unsigned k = 0; k < m_config->gpgpu_num_tensor_core_units; k++)
    //      m_issue_port.push_back(OC_EX_TENSOR_CORE);
    //    for (unsigned j = 0; j < m_config->m_specialized_unit.size(); j++)
    //      for (unsigned k = 0; k < m_config->m_specialized_unit[j].num_units; k++)
    //        m_issue_port.push_back(m_config->m_specialized_unit[j].OC_EX_SPEC_ID);
    //    m_issue_port.push_back(OC_EX_MEM);
    unsigned issue_port = m_issue_port[n];
    //����m_issue_port[n]����ȡ��ˮ�߼Ĵ���m_pipeline_reg[issue_port]�е�ָ��Ĵ�������
    //issue_inst��
    register_set &issue_inst = m_pipeline_reg[issue_port];
    unsigned reg_id;
    //��V100�����У�partition_issue������LDST��Ԫ��Ϊfalse�����൥Ԫ��Ϊtrue��
    bool partition_issue =
        m_config->sub_core_model && m_fu[n]->is_issue_partitioned();
    //��partition_issueΪfalseʱ��reg_idΪ0����partition_issueΪtrueʱ��reg_idΪm_fu[n]
    //->get_issue_reg_id()��������Ϊ����˵LDST��Ԫ����һ������issue_reg_idΪ0����������Ԫ
    //�ж����������Ҫͨ��get_issue_reg_id()����ȡ��get_issue_reg_id()��ʵ���صľ��ǵ�ǰ
    //SIMD��Ԫ��m_fu����ͬ���͵�Ԫ�е�����������һ����4��SP��Ԫ����ǰ��Ԫ�Ǵ����ſ�ʼ��3����
    //���Ըú����ͷ��ص���3����
    if (partition_issue) {
      reg_id = m_fu[n]->get_issue_reg_id();
    }
    //����m_fu[n]��Ԫ����ˮ�߼Ĵ���m_pipeline_reg[issue_port]�е�ָ��Ĵ�������issue_inst
    //�ĵ�reg_id����
    warp_inst_t **ready_reg = issue_inst.get_ready(partition_issue, reg_id);
    //����һ���Ĵ���reg_id���жϸüĴ����Ƿ�ǿա�
    if (issue_inst.has_ready(partition_issue, reg_id) &&
        //����**ready_regָ���return m_dispatch_reg->empty()�����ж�m_dispatch_reg�Ƿ�
        //Ϊ�ա�
        m_fu[n]->can_issue(**ready_reg)) {
      //m_fu[n]->stallable()��LDST��Ԫ����true�����෵��false��
      bool schedule_wb_now = !m_fu[n]->stallable();
      int resbus = -1;
      if (schedule_wb_now &&
          //��LDST��Ԫ�������
          //test_res_bus() function locates a free slot in all the result buses (the 
          //slot is free if its bit is not set).
          //test_res_bus���������н�������ж�λһ�����в�ۣ����δ������λ����ò��Ϊ��
          //�в��)��ʵ����test_res_bus()����ģ�����ָ���ӳ١�
          (resbus = test_res_bus((*ready_reg)->latency)) != -1) {
        assert((*ready_reg)->latency < MAX_ALU_LATENCY);
        //m_result_busʵ������ģ��ָ���ӳ١�
        m_result_bus[resbus]->set((*ready_reg)->latency);
        //ִ��simd_function_unit::issue(register_set &source_reg)������
        m_fu[n]->issue(issue_inst);
      } else if (!schedule_wb_now) {
        //LDST��Ԫ�����
        m_fu[n]->issue(issue_inst);
      } else {
        // stall issue (cannot reserve result bus)
      }
    }
  }
}

void ldst_unit::print_cache_stats(FILE *fp, unsigned &dl1_accesses,
                                  unsigned &dl1_misses) {
  if (m_L1D) {
    m_L1D->print(fp, dl1_accesses, dl1_misses);
  }
}

void ldst_unit::get_cache_stats(cache_stats &cs) {
  // Adds stats to 'cs' from each cache
  if (m_L1D) cs += m_L1D->get_stats();
  if (m_L1C) cs += m_L1C->get_stats();
  if (m_L1T) cs += m_L1T->get_stats();
}

void ldst_unit::get_L1D_sub_stats(struct cache_sub_stats &css) const {
  if (m_L1D) m_L1D->get_sub_stats(css);
}
void ldst_unit::get_L1C_sub_stats(struct cache_sub_stats &css) const {
  if (m_L1C) m_L1C->get_sub_stats(css);
}
void ldst_unit::get_L1T_sub_stats(struct cache_sub_stats &css) const {
  if (m_L1T) m_L1T->get_sub_stats(css);
}

// Add this function to unset depbar
void shader_core_ctx::unset_depbar(const warp_inst_t &inst) {
  bool done_flag = true;
  unsigned int end_group = m_warp[inst.warp_id()]->m_depbar_start_id == 0
                               ? m_warp[inst.warp_id()]->m_ldgdepbar_buf.size()
                               : (m_warp[inst.warp_id()]->m_depbar_start_id -
                                  m_warp[inst.warp_id()]->m_depbar_group + 1);

  if (inst.m_is_ldgsts) {
    for (int i = 0; i < m_warp[inst.warp_id()]->m_ldgdepbar_buf.size(); i++) {
      for (int j = 0; j < m_warp[inst.warp_id()]->m_ldgdepbar_buf[i].size();
           j++) {
        if (m_warp[inst.warp_id()]->m_ldgdepbar_buf[i][j].pc == inst.pc) {
          // Handle the case that same pc results in multiple LDGSTS
          // instructions
          if (m_warp[inst.warp_id()]->m_ldgdepbar_buf[i][j].get_addr(0) ==
              inst.get_addr(0)) {
            m_warp[inst.warp_id()]->m_ldgdepbar_buf[i][j].pc = -1;
            goto DoneWB;
          }
        }
      }
    }

  DoneWB:
    for (int i = 0; i < end_group; i++) {
      for (int j = 0; j < m_warp[inst.warp_id()]->m_ldgdepbar_buf[i].size();
           j++) {
        if (m_warp[inst.warp_id()]->m_ldgdepbar_buf[i][j].pc != -1) {
          done_flag = false;
          goto UpdateDEPBAR;
        }
      }
    }

  UpdateDEPBAR:
    if (done_flag) {
      if (m_warp[inst.warp_id()]->m_waiting_ldgsts) {
        m_warp[inst.warp_id()]->m_waiting_ldgsts = false;
      }
    }
  }
}

void shader_core_ctx::warp_inst_complete(const warp_inst_t &inst) {
#if 0
      printf("[warp_inst_complete] uid=%u core=%u warp=%u pc=%#x @ time=%llu \n",
             inst.get_uid(), m_sid, inst.warp_id(), inst.pc,  m_gpu->gpu_tot_sim_cycle +  m_gpu->gpu_sim_cycle);
#endif
  if (inst.op_pipe == SP__OP)
    m_stats->m_num_sp_committed[m_sid]++;
  else if (inst.op_pipe == SFU__OP)
    m_stats->m_num_sfu_committed[m_sid]++;
  else if (inst.op_pipe == MEM__OP)
    m_stats->m_num_mem_committed[m_sid]++;

  if (m_config->gpgpu_clock_gated_lanes == false)
    m_stats->m_num_sim_insn[m_sid] += m_config->warp_size;
  else
    m_stats->m_num_sim_insn[m_sid] += inst.active_count();

  m_stats->m_num_sim_winsn[m_sid]++;
  m_gpu->gpu_sim_insn += inst.active_count();
  inst.completed(m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle);
}

/*
��ˮ�ߵ�д�ؽ׶Ρ���ִ�н׶εĽ��д�ص��Ĵ����ļ��С����ȣ�EX_WB�Ĵ������еľ����۱�ʶ�𲢼�
�ص�preg�������Ч�������m_operated_collector.writeback��Ȼ��Ŀ��Ĵ����ӼǷְ����ͷţ�
EX_WB�Ĵ������еĲ�۱��������һֱ������EX_WB�Ĵ������е����о���ָ���д��Ϊֹ��
*/
void shader_core_ctx::writeback() {
  unsigned max_committed_thread_instructions =
      m_config->warp_size *
      (m_config->pipe_widths[EX_WB]);  // from the functional units
  m_stats->m_pipeline_duty_cycle[m_sid] =
      ((float)(m_stats->m_num_sim_insn[m_sid] -
               m_stats->m_last_num_sim_insn[m_sid])) /
      max_committed_thread_instructions;

  m_stats->m_last_num_sim_insn[m_sid] = m_stats->m_num_sim_insn[m_sid];
  m_stats->m_last_num_sim_winsn[m_sid] = m_stats->m_num_sim_winsn[m_sid];

  //get_ready()��EX_WB��ˮ�߼Ĵ�����ȡһ���ǿռĴ���������ָ���Ƴ�������������ָ�
  //���ldst��Ԫ��ָ���ִ�е������Ϊldst��Ԫ��ָ����ִ�н׶ξ��Ѿ�д���ˡ�
  warp_inst_t **preg = m_pipeline_reg[EX_WB].get_ready();
  warp_inst_t *pipe_reg = (preg == NULL) ? NULL : *preg;
  while (preg and !pipe_reg->empty()) {
    /*
     * Right now, the writeback stage drains all waiting instructions
     * assuming there are enough ports in the register file or the
     * conflicts are resolved at issue.
     */
    /*
    ���ڣ�д�ؽ׶λ�ľ����еȴ���ָ�����Ĵ����ļ������㹻�Ķ˿ڣ����߳�ͻ�Ѿ������
    */
    /*
     * The operand collector writeback can generally generate a stall
     * However, here, the pipelines should be un-stallable. This is
     * guaranteed because this is the first time the writeback function
     * is called after the operand collector's step function, which
     * resets the allocations. There is one case which could result in
     * the writeback function returning false (stall), which is when
     * an instruction tries to modify two registers (GPR and predicate)
     * To handle this case, we ignore the return value (thus allowing
     * no stalling).
     */
    /*
    �������ռ���д��ͨ����������ͣ��Ȼ�����������ˮ��Ӧ����un-stallable�ġ������б�
    ֤�ģ���Ϊ�����ڲ������ռ�����step����֮���״ε���д�غ������ú������÷��䡣��һ����
    �����ܵ���д�غ�������false��stall������ָ����ͼ�޸������Ĵ�����GPR��ν�ʣ���Ϊ�˴�
    ��������������Ǻ��Է���ֵ����˲�����ͣ�ͣ���
    */
    //�������ռ�����Bankд�ء�
    m_operand_collector.writeback(*pipe_reg);
    //��ȡpipe_regָ���warp ID��
    unsigned warp_id = pipe_reg->warp_id();
    // release the register from the scoreboard.
    m_scoreboard->releaseRegisters(pipe_reg);
    m_warp[warp_id]->dec_inst_in_pipeline();
    warp_inst_complete(*pipe_reg);
    m_gpu->gpu_sim_insn_last_update_sid = m_sid;
    m_gpu->gpu_sim_insn_last_update = m_gpu->gpu_sim_cycle;
    m_last_inst_gpu_sim_cycle = m_gpu->gpu_sim_cycle;
    m_last_inst_gpu_tot_sim_cycle = m_gpu->gpu_tot_sim_cycle;
    pipe_reg->clear();
    //ѭ����һ����ˮ�߼Ĵ�������m_pipeline_reg[EX_WB]�е���Чָ��Ĵ�����ִ������ֹͣ
    //����
    preg = m_pipeline_reg[EX_WB].get_ready();
    pipe_reg = (preg == NULL) ? NULL : *preg;
  }
}

/*
�ж�LDST��Ԫ����shared memory�Ƿ��Stall��Stall�Ļ�����0��û��Stall�Ļ�����1��
*/
bool ldst_unit::shared_cycle(warp_inst_t &inst, mem_stage_stall_type &rc_fail,
                             mem_stage_access_type &fail_type) {
  //���ָ���shared memoryָ��򷵻�true��
  if (inst.space.get_type() != shared_space) return true;
  //���ָ��Ļ�Ծ�߳���Ϊ0���򷵻�true��
  if (inst.active_count() == 0) return true;

  //���ظ���ָ�ʣ�¶���initiation_interval��
  if (inst.has_dispatch_delay()) {
    m_stats->gpgpu_n_shmem_bank_access[m_sid]++;
  }
  //��ʼ��ʱ����Ϊcycles = initiation_interval;��ÿ�ε���dispatch_delay()ʱ��cycles
  //��1��ֱ��cycles=0��
  bool stall = inst.dispatch_delay();
  if (stall) {
    //Stall����ΪS_MEM��ԭ����Bank conflict��
    fail_type = S_MEM;
    rc_fail = BK_CONF;
    m_stats->gpgpu_n_shmem_bkconflict++;
  } else
    rc_fail = NO_RC_FAIL;
  
  //Stall�Ļ�����0��û��Stall�Ļ�����1��
  return !stall;
}

mem_stage_stall_type ldst_unit::process_cache_access(
    cache_t *cache, new_addr_type address, warp_inst_t &inst,
    std::list<cache_event> &events, mem_fetch *mf,
    enum cache_request_status status) {
  mem_stage_stall_type result = NO_RC_FAIL;
  bool write_sent = was_write_sent(events);
  bool read_sent = was_read_sent(events);
  if (write_sent) {
    unsigned inc_ack = (m_config->m_L1D_config.get_mshr_type() == SECTOR_ASSOC)
                           ? (mf->get_data_size() / SECTOR_SIZE)
                           : 1;

    for (unsigned i = 0; i < inc_ack; ++i)
      m_core->inc_store_req(inst.warp_id());
  }
  if (status == HIT) {
    assert(!read_sent);
    inst.accessq_pop_back();
    if (inst.is_load()) {
      for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++)
        if (inst.out[r] > 0) m_pending_writes[inst.warp_id()][inst.out[r]]--;

      // release LDGSTS
      if (inst.m_is_ldgsts) {
        m_pending_ldgsts[inst.warp_id()][inst.pc][inst.get_addr(0)]--;
        if (m_pending_ldgsts[inst.warp_id()][inst.pc][inst.get_addr(0)] == 0) {
          m_core->unset_depbar(inst);
        }
      }
    }
    if (!write_sent) delete mf;
  } else if (status == RESERVATION_FAIL) {
    result = BK_CONF;
    assert(!read_sent);
    assert(!write_sent);
    delete mf;
  } else {
    assert(status == MISS || status == HIT_RESERVED);
    // inst.clear_active( access.get_warp_mask() ); // threads in mf writeback
    // when mf returns
    inst.accessq_pop_back();
  }
  if (!inst.accessq_empty() && result == NO_RC_FAIL) result = COAL_STALL;
  return result;
}

mem_stage_stall_type ldst_unit::process_memory_access_queue(cache_t *cache,
                                                            warp_inst_t &inst) {
  mem_stage_stall_type result = NO_RC_FAIL;
  if (inst.accessq_empty()) return result;

  if (!cache->data_port_free()) return DATA_PORT_STALL;

  // const mem_access_t &access = inst.accessq_back();
  mem_fetch *mf = m_mf_allocator->alloc(
      inst, inst.accessq_back(),
      m_core->get_gpu()->gpu_sim_cycle + m_core->get_gpu()->gpu_tot_sim_cycle);
  std::list<cache_event> events;
  enum cache_request_status status = cache->access(
      mf->get_addr(), mf,
      m_core->get_gpu()->gpu_sim_cycle + m_core->get_gpu()->gpu_tot_sim_cycle,
      events);
  return process_cache_access(cache, mf->get_addr(), inst, events, mf, status);
}

/*
�����L1���ݻ�����ڴ���ʡ�
*/
mem_stage_stall_type ldst_unit::process_memory_access_queue_l1cache(
    l1_cache *cache, warp_inst_t &inst) {
  mem_stage_stall_type result = NO_RC_FAIL;
  //inst.accessq_empty()���ص�ǰָ��ķô�������б��Ƿ�Ϊ�ա�
  if (inst.accessq_empty()) return result;

  //��V100�У�m_config->m_L1D_config.l1_latency������Ϊ20�ģ�ָ����L1���ݻ�������ʱ�ķ���������
  if (m_config->m_L1D_config.l1_latency > 0) {
    //��V100�У�m_config->m_L1D_config.l1_banks������Ϊ4��ָ����L1���ݻ����bank������ÿ���ܹ�
    //�������������󲻳���4��
    for (unsigned int j = 0; j < m_config->m_L1D_config.l1_banks;
         j++) {  // We can handle at max l1_banks reqs per cycle

      //inst.accessq_empty()���ص�ǰָ��ķô�������б��Ƿ�Ϊ�ա�
      if (inst.accessq_empty()) return result;

      mem_fetch *mf =
          m_mf_allocator->alloc(inst, inst.accessq_back(),
                                m_core->get_gpu()->gpu_sim_cycle +
                                    m_core->get_gpu()->gpu_tot_sim_cycle);
      unsigned bank_id = m_config->m_L1D_config.set_bank(mf->get_addr());
      assert(bank_id < m_config->m_L1D_config.l1_banks);
      //l1_latency_queue�Ķ���Ϊ��
      //    std::vector<std::deque<mem_fetch *>> l1_latency_queue;
      //��ʼ��Ϊ��
      //    l1_latency_queue.resize(m_config->m_L1D_config.l1_banks);
      //    for (unsigned j = 0; j < m_config->m_L1D_config.l1_banks; j++)
      //      l1_latency_queue[j].resize(m_config->m_L1D_config.l1_latency,
      //                                 (mem_fetch *)NULL);
      //���Ǹ���ά���飬��һά������Ϊbank_id���ڶ�ά�ĳ���Ϊ20��ÿ��Ԫ�ض���(mem_fetch *)����
      //l1_latency_queue����ģ��L1���ݻ���ķ����ӳ٣���V100�������У�m_config->m_L1D_config.
      //l1_latency������Ϊ20�ģ�ָ����L1���ݻ�������ʱ�ķ���������4��Bankÿ��Bank���Դ�������20
      //���������������0��Bank��l1_latency_queue[0]��洢��20���������󣬵��µķ���������
      //ʱ������ֱ�Ӽӵ�l1_latency_queue[0][19]λ�ã�����l1_latency_queue[0][0]�������������
      //��������ϡ�ÿһ�ģ�l1_latency_queue[0]�е����󶼻���ǰ�ƶ�һ��λ�ã���l1_latency_queue
      //[0][0]������ᱻɾ����l1_latency_queue[0][19]������ᱻ�ƶ���l1_latency_queue[0][18]
      //��λ�ã�l1_latency_queue[0][18]������ᱻ�ƶ���l1_latency_queue[0][17]��λ�ã��Դ����ơ�
      if ((l1_latency_queue[bank_id][m_config->m_L1D_config.l1_latency - 1]) ==
          NULL) {
        //���������l1_latency_queue[bank_id][19]Ϊ�յĻ����ͽ��µ�����д�뵽l1_latency_queue
        //[bank_id][19]��λ�á�
        l1_latency_queue[bank_id][m_config->m_L1D_config.l1_latency - 1] = mf;

        //���mf����ָ���Ǵ洢������
        if (mf->get_inst().is_store()) {
          unsigned inc_ack =
              (m_config->m_L1D_config.get_mshr_type() == SECTOR_ASSOC)
                  ? (mf->get_data_size() / SECTOR_SIZE)
                  : 1;

          for (unsigned i = 0; i < inc_ack; ++i)
            m_core->inc_store_req(inst.warp_id());
        }

        inst.accessq_pop_back();
      } else {
        //���������l1_latency_queue[bank_id][19]��Ϊ�յĻ������Ƿ���Bank conflict����Ҫ�ȴ���
        result = BK_CONF;
        m_stats->gpgpu_n_l1cache_bkconflict++;
        delete mf;
        break;  // do not try again, just break from the loop and try the next
                // cycle
      }
    }
    //inst.accessq_empty()���ص�ǰָ��ķô�������б��Ƿ�Ϊ�ա���Ϊ�յĻ����Ҳ�����ΪBank��ͻ
    //����Stall������coalescing stall��
    if (!inst.accessq_empty() && result != BK_CONF) result = COAL_STALL;

    return result;
  } else {
    mem_fetch *mf =
        m_mf_allocator->alloc(inst, inst.accessq_back(),
                              m_core->get_gpu()->gpu_sim_cycle +
                                  m_core->get_gpu()->gpu_tot_sim_cycle);
    std::list<cache_event> events;
    enum cache_request_status status = cache->access(
        mf->get_addr(), mf,
        m_core->get_gpu()->gpu_sim_cycle + m_core->get_gpu()->gpu_tot_sim_cycle,
        events);
    return process_cache_access(cache, mf->get_addr(), inst, events, mf,
                                status);
  }
}

void ldst_unit::L1_latency_queue_cycle() {
  //�����е�L1D��bank����ѭ����
  for (unsigned int j = 0; j < m_config->m_L1D_config.l1_banks; j++) {
    //l1_latency_queue�Ķ���Ϊ��
    //    std::vector<std::deque<mem_fetch *>> l1_latency_queue;
    //��ʼ��Ϊ��
    //    l1_latency_queue.resize(m_config->m_L1D_config.l1_banks);
    //    for (unsigned j = 0; j < m_config->m_L1D_config.l1_banks; j++)
    //      l1_latency_queue[j].resize(m_config->m_L1D_config.l1_latency,
    //                                 (mem_fetch *)NULL);
    //���Ǹ���ά���飬��һά������Ϊbank_id���ڶ�ά�ĳ���Ϊ20��ÿ��Ԫ�ض���(mem_fetch *)����
    //l1_latency_queue����ģ��L1���ݻ���ķ����ӳ٣���V100�������У�m_config->m_L1D_config.
    //l1_latency������Ϊ20�ģ�ָ����L1���ݻ�������ʱ�ķ���������4��Bankÿ��Bank���Դ�������20
    //���������������0��Bank��l1_latency_queue[0]��洢��20���������󣬵��µķ���������
    //ʱ������ֱ�Ӽӵ�l1_latency_queue[0][19]λ�ã�����l1_latency_queue[0][0]�������������
    //��������ϡ�ÿһ�ģ�l1_latency_queue[0]�е����󶼻���ǰ�ƶ�һ��λ�ã���l1_latency_queue
    //[0][0]������ᱻɾ����l1_latency_queue[0][19]������ᱻ�ƶ���l1_latency_queue[0][18]
    //��λ�ã�l1_latency_queue[0][18]������ᱻ�ƶ���l1_latency_queue[0][17]��λ�ã��Դ����ơ�

    //��j��bank�ĵ�0��λ��(l1_latency_queue[j][0])��Ϊ�յĻ���˵����һ���Ѿ���ɵ�L1D Cache��
    //��mf_next��
    if ((l1_latency_queue[j][0]) != NULL) {
      mem_fetch *mf_next = l1_latency_queue[j][0];
      std::list<cache_event> events;
      //m_L1D����״̬��
      enum cache_request_status status =
          m_L1D->access(mf_next->get_addr(), mf_next,
                        m_core->get_gpu()->gpu_sim_cycle +
                            m_core->get_gpu()->gpu_tot_sim_cycle,
                        events);

      //�ж�һϵ�еķ���cache�¼��Ƿ����WRITE_REQUEST_SENT��
      bool write_sent = was_write_sent(events);
      //�ж�һϵ�еķ���cache�¼��Ƿ����READ_REQUEST_SENT��
      bool read_sent = was_read_sent(events);

      if (status == HIT) {
        assert(!read_sent);
        l1_latency_queue[j][0] = NULL;
        if (mf_next->get_inst().is_load()) {
          for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++)
            if (mf_next->get_inst().out[r] > 0) {
              assert(m_pending_writes[mf_next->get_inst().warp_id()]
                                     [mf_next->get_inst().out[r]] > 0);
              unsigned still_pending =
                  --m_pending_writes[mf_next->get_inst().warp_id()]
                                    [mf_next->get_inst().out[r]];
              if (!still_pending) {
                m_pending_writes[mf_next->get_inst().warp_id()].erase(
                    mf_next->get_inst().out[r]);
                m_scoreboard->releaseRegister(mf_next->get_inst().warp_id(),
                                              mf_next->get_inst().out[r]);
                m_core->warp_inst_complete(mf_next->get_inst());
              }
            }

          // release LDGSTS
          if (mf_next->get_inst().m_is_ldgsts) {
            m_pending_ldgsts[mf_next->get_inst().warp_id()]
                            [mf_next->get_inst().pc]
                            [mf_next->get_inst().get_addr(0)]--;
            if (m_pending_ldgsts[mf_next->get_inst().warp_id()]
                                [mf_next->get_inst().pc]
                                [mf_next->get_inst().get_addr(0)] == 0) {
              m_core->unset_depbar(mf_next->get_inst());
            }
          }
        }

        // For write hit in WB policy
        if (mf_next->get_inst().is_store() && !write_sent) {
          unsigned dec_ack =
              (m_config->m_L1D_config.get_mshr_type() == SECTOR_ASSOC)
                  ? (mf_next->get_data_size() / SECTOR_SIZE)
                  : 1;

          mf_next->set_reply();

          for (unsigned i = 0; i < dec_ack; ++i) m_core->store_ack(mf_next);
        }

        if (!write_sent) delete mf_next;

      } else if (status == RESERVATION_FAIL) {
        assert(!read_sent);
        assert(!write_sent);
      } else {
        assert(status == MISS || status == HIT_RESERVED);
        l1_latency_queue[j][0] = NULL;
        if (m_config->m_L1D_config.get_write_policy() != WRITE_THROUGH &&
            mf_next->get_inst().is_store() &&
            (m_config->m_L1D_config.get_write_allocate_policy() ==
                 FETCH_ON_WRITE ||
             m_config->m_L1D_config.get_write_allocate_policy() ==
                 LAZY_FETCH_ON_READ) &&
            !was_writeallocate_sent(events)) {
          unsigned dec_ack =
              (m_config->m_L1D_config.get_mshr_type() == SECTOR_ASSOC)
                  ? (mf_next->get_data_size() / SECTOR_SIZE)
                  : 1;
          mf_next->set_reply();
          for (unsigned i = 0; i < dec_ack; ++i) m_core->store_ack(mf_next);
          if (!write_sent && !read_sent) delete mf_next;
        }
      }
    }

    for (unsigned stage = 0; stage < m_config->m_L1D_config.l1_latency - 1;
         ++stage)
      if (l1_latency_queue[j][stage] == NULL) {
        l1_latency_queue[j][stage] = l1_latency_queue[j][stage + 1];
        l1_latency_queue[j][stage + 1] = NULL;
      }
  }
}

bool ldst_unit::constant_cycle(warp_inst_t &inst, mem_stage_stall_type &rc_fail,
                               mem_stage_access_type &fail_type) {
  if (inst.empty() || ((inst.space.get_type() != const_space) &&
                       (inst.space.get_type() != param_space_kernel)))
    return true;
  if (inst.active_count() == 0) return true;

  mem_stage_stall_type fail;
  if (m_config->perfect_inst_const_cache) {
    fail = NO_RC_FAIL;
    unsigned access_count = inst.accessq_count();
    while (inst.accessq_count() > 0) inst.accessq_pop_back();
    if (inst.is_load()) {
      for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++)
        if (inst.out[r] > 0)
          m_pending_writes[inst.warp_id()][inst.out[r]] -= access_count;
    }
  } else {
    fail = process_memory_access_queue(m_L1C, inst);
  }

  if (fail != NO_RC_FAIL) {
    rc_fail = fail;  // keep other fails if this didn't fail.
    fail_type = C_MEM;
    if (rc_fail == BK_CONF or rc_fail == COAL_STALL) {
      m_stats->gpgpu_n_cmem_portconflict++;  // coal stalls aren't really a bank
                                             // conflict, but this maintains
                                             // previous behavior.
    }
  }
  return inst.accessq_empty();  // done if empty.
}

bool ldst_unit::texture_cycle(warp_inst_t &inst, mem_stage_stall_type &rc_fail,
                              mem_stage_access_type &fail_type) {
  if (inst.empty() || inst.space.get_type() != tex_space) return true;
  if (inst.active_count() == 0) return true;
  mem_stage_stall_type fail = process_memory_access_queue(m_L1T, inst);
  if (fail != NO_RC_FAIL) {
    rc_fail = fail;  // keep other fails if this didn't fail.
    fail_type = T_MEM;
  }
  return inst.accessq_empty();  // done if empty.
}

/*
�ж�LDST��Ԫ����global/local/param memory�Ƿ��Stall��Stall�Ļ�����0��û��Stall�Ļ�����1��
*/
bool ldst_unit::memory_cycle(warp_inst_t &inst,
                             mem_stage_stall_type &stall_reason,
                             mem_stage_access_type &access_type) {
  if (inst.empty() || ((inst.space.get_type() != global_space) &&
                       (inst.space.get_type() != local_space) &&
                       (inst.space.get_type() != param_space_local)))
    return true;
  if (inst.active_count() == 0) return true;
  if (inst.accessq_empty()) return true;

  mem_stage_stall_type stall_cond = NO_RC_FAIL;
  const mem_access_t &access = inst.accessq_back();

  //bypassL1D��ָ�����ݷ����Ƿ��ƹ�L1���ݻ��档
  bool bypassL1D = false;
  if (CACHE_GLOBAL == inst.cache_op || (m_L1D == NULL)) {
    //CACHE_GLOBAL==.cg��Cache at global level��ȫ�ּ����棨L2�����»��棬������L1����ʹ��ld.cgȫ����
    //�ؼ��أ��ƹ�L1���棬����������L2�����С�
    bypassL1D = true;
  } else if (inst.space.is_global()) {  // global memory access
    // skip L1 cache if the option is enabled
    //��V100�����У�gpgpu_gmem_skip_L1D����Ϊ0��
    if (m_core->get_config()->gmem_skip_L1D && (CACHE_L1 != inst.cache_op))
      bypassL1D = true;
  }
  //����ֻ����ʹ��ld.cgʱ�Ż��ƹ�L1���ݻ��档
  if (bypassL1D) {
    // bypass L1 cache
    unsigned control_size =
        inst.is_store() ? WRITE_PACKET_SIZE : READ_PACKET_SIZE;
    unsigned size = access.get_size() + control_size;
    // printf("Interconnect:Addr: %x, size=%d\n",access.get_addr(),size);
    if (m_icnt->full(size, inst.is_store() || inst.isatomic())) {
      //ICNT������
      stall_cond = ICNT_RC_FAIL;
    } else {
      mem_fetch *mf =
          m_mf_allocator->alloc(inst, access,
                                m_core->get_gpu()->gpu_sim_cycle +
                                    m_core->get_gpu()->gpu_tot_sim_cycle);
      //��mf����ICNT�ķ��Ͷ����С�
      m_icnt->push(mf);
      inst.accessq_pop_back();
      // inst.clear_active( access.get_warp_mask() );
      if (inst.is_load()) {
        for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++)
          if (inst.out[r] > 0)
            assert(m_pending_writes[inst.warp_id()][inst.out[r]] > 0);
      } else if (inst.is_store())
        m_core->inc_store_req(inst.warp_id());
    }
  } else {
    // do not bypass L1 cache
    assert(CACHE_UNDEFINED != inst.cache_op);
    //�����L1���ݻ�����ڴ���ʡ�
    stall_cond = process_memory_access_queue_l1cache(m_L1D, inst);
  }
  //inst.accessq_empty()���ص�ǰָ��ķô�������б��Ƿ�Ϊ�ա������ǵ�ǰָ��ķô�������б�Ϊ�գ���
  //û�з���STALL����˵��������Coalescing Stall��
  if (!inst.accessq_empty() && stall_cond == NO_RC_FAIL)
    stall_cond = COAL_STALL;
  if (stall_cond != NO_RC_FAIL) {
    stall_reason = stall_cond;
    bool iswrite = inst.is_store();
    if (inst.space.is_local())
      access_type = (iswrite) ? L_MEM_ST : L_MEM_LD;
    else
      access_type = (iswrite) ? G_MEM_ST : G_MEM_LD;
  }
  return inst.accessq_empty();
}

/*
LD/ST��Ԫ����ӦFIFO�е����ݰ��� >= GPU���õ���Ӧ�����е������Ӧ������������Ҫע����ǣ�LD/ST��ԪҲ��һ��
m_response_fifo����m_response_fifo.size()��ȡ���Ǹ�fifo�Ѿ��洢��mf��Ŀ�������Ŀ�ܹ��жϸ�fifo�Ƿ�������
m_config->ldst_unit_response_queue_size�������õĸ�fifo�����������һ��m_response_fifo.size()��������
������������ͻ᷵��True����ʾ��fifo������
*/
bool ldst_unit::response_buffer_full() const {
  return m_response_fifo.size() >= m_config->ldst_unit_response_queue_size;
}

/*
��mem_fetch *mf����SIMT Core��Ⱥ��������ӦFIFO��
*/
void ldst_unit::fill(mem_fetch *mf) {
  mf->set_status(
      IN_SHADER_LDST_RESPONSE_FIFO, //����ǽ���Shader Core��LDST��Ԫ��ӦFIFO��mf
      m_core->get_gpu()->gpu_sim_cycle + m_core->get_gpu()->gpu_tot_sim_cycle);
  //�����m_response_fifo��SIMT Core��Ⱥ��LDST��Ԫ�ڵ���ӦFIFO������Ϊ��
  //    std::list<mem_fetch *> m_response_fifo;
  m_response_fifo.push_back(mf);
}

void ldst_unit::flush() {
  // Flush L1D cache
  m_L1D->flush();
}

void ldst_unit::invalidate() {
  // Flush L1D cache
  m_L1D->invalidate();
}

/*
issue(warp_inst_t*&)��Ա��������������ˮ�߼Ĵ�������������m_dispatch_reg��Ȼ��ָ����m_dispatch_reg�ȴ�
initiation_interval�����ڡ��ڴ��ڼ䣬û��������ָ����Է��������Ԫ����������ȴ���ָ�����������ģ�͡�
*/
simd_function_unit::simd_function_unit(const shader_core_config *config) {
  m_config = config;
  //m_dispatch_reg��ʵ�Ǹ����壬���������ָ��ȴ�ִ�С���ָ������õ����ܵ�Ԫ��OC_EX�Ĵ�������ʱ������
  //�����ڵ��ȼĴ����С�dispatch register������һ�����͵ļĴ�����dispatch register��¼��ָ����ڱ�����
  //ִ��ʱ�����ݸ�ִ�е�Ԫ��������˵��dispatch register��һ����������ֶεĽṹ�壬���а���ָ���Ŀ�ĵ�ַ��
  //�������͡����������Ϣ��dispatch register���Կ�����ָ����ȹ����д������ݵ���Ҫ�Ĵ�����
  m_dispatch_reg = new warp_inst_t(config);
}

/*
issue(warp_inst_t*&)��Ա��������������ˮ�߼Ĵ�������������m_dispatch_reg��
*/
void simd_function_unit::issue(register_set &source_reg) {
  //��simd_function_unitʵ���У�is_issue_partitioned()�����⺯������LDST��Ԫ����������㵥Ԫ������True��
  //m_config->sub_core_modelΪTrue��
  bool partition_issue =
      m_config->sub_core_model && this->is_issue_partitioned();
  //source_reg��Ϊ��ˮ�߼Ĵ�����Ŀ�����ҵ�һ���ǿյ�ָ���������m_dispatch_reg��
  source_reg.move_out_to(partition_issue, this->get_issue_reg_id(),
                         m_dispatch_reg);
  //����m_dispatch_reg�ı�ʶռ��λͼ��״̬��m_dispatch_reg��warp_inst_t���ͣ��ɻ�ȡ��ָ����ӳ١�
  occupied.set(m_dispatch_reg->latency);
}

/*
SFU���⹦�ܵ�Ԫ�Ĺ��캯������m_name��ͬ��
*/
sfu::sfu(register_set *result_port, const shader_core_config *config,
         shader_core_ctx *core, unsigned issue_reg_id)
    : pipelined_simd_unit(result_port, config, config->max_sfu_latency, core,
                          issue_reg_id) {
  m_name = "SFU";
}

/*
Tensor Core��Ԫ�Ĺ��캯������m_name��ͬ��
*/
tensor_core::tensor_core(register_set *result_port,
                         const shader_core_config *config,
                         shader_core_ctx *core, unsigned issue_reg_id)
    : pipelined_simd_unit(result_port, config, config->max_tensor_core_latency,
                          core, issue_reg_id) {
  m_name = "TENSOR_CORE";
}

/*
SFU���⹦�ܵ�Ԫ�ķ��亯����
*/
void sfu::issue(register_set &source_reg) {
  warp_inst_t **ready_reg =
      source_reg.get_ready(m_config->sub_core_model, m_issue_reg_id);
  // m_core->incexecstat((*ready_reg));

  (*ready_reg)->op_pipe = SFU__OP;
  m_core->incsfu_stat(m_core->get_config()->warp_size, (*ready_reg)->latency);
  pipelined_simd_unit::issue(source_reg);
}

/*
Tensor Core��Ԫ�ķ��亯����
*/
void tensor_core::issue(register_set &source_reg) {
  warp_inst_t **ready_reg =
      source_reg.get_ready(m_config->sub_core_model, m_issue_reg_id);
  // m_core->incexecstat((*ready_reg));

  (*ready_reg)->op_pipe = TENSOR_CORE__OP;
  m_core->incsfu_stat(m_core->get_config()->warp_size, (*ready_reg)->latency);
  pipelined_simd_unit::issue(source_reg);
}

/*
lane����˼Ϊһ��warp����32���̣߳�������ˮ�߼Ĵ����п����ݴ��˺ܶ���ָ���Щָ���ÿ��Ӧ���߳������ÿһ
λ����һ��lane����������ˮ�߼Ĵ����еķǿ�ָ���������ָ��������߳����루����ָ���߳�����Ļ�ֵ����
*/
unsigned pipelined_simd_unit::get_active_lanes_in_pipeline() {
  active_mask_t active_lanes;
  active_lanes.reset();
  if (m_core->get_gpu()->get_config().g_power_simulation_enabled) {
    for (unsigned stage = 0; (stage + 1) < m_pipeline_depth; stage++) {
      if (!m_pipeline_reg[stage]->empty())
        active_lanes |= m_pipeline_reg[stage]->get_active_mask();
    }
  }
  return active_lanes.count();
}

void ldst_unit::active_lanes_in_pipeline() {
  unsigned active_count = pipelined_simd_unit::get_active_lanes_in_pipeline();
  assert(active_count <= m_core->get_config()->warp_size);
  m_core->incfumemactivelanes_stat(active_count);
}

void sp_unit::active_lanes_in_pipeline() {
  unsigned active_count = pipelined_simd_unit::get_active_lanes_in_pipeline();
  assert(active_count <= m_core->get_config()->warp_size);
  m_core->incspactivelanes_stat(active_count);
  m_core->incfuactivelanes_stat(active_count);
  m_core->incfumemactivelanes_stat(active_count);
}
void dp_unit::active_lanes_in_pipeline() {
  unsigned active_count = pipelined_simd_unit::get_active_lanes_in_pipeline();
  assert(active_count <= m_core->get_config()->warp_size);
  // m_core->incspactivelanes_stat(active_count);
  m_core->incfuactivelanes_stat(active_count);
  m_core->incfumemactivelanes_stat(active_count);
}
void specialized_unit::active_lanes_in_pipeline() {
  unsigned active_count = pipelined_simd_unit::get_active_lanes_in_pipeline();
  assert(active_count <= m_core->get_config()->warp_size);
  m_core->incspactivelanes_stat(active_count);
  m_core->incfuactivelanes_stat(active_count);
  m_core->incfumemactivelanes_stat(active_count);
}

void int_unit::active_lanes_in_pipeline() {
  unsigned active_count = pipelined_simd_unit::get_active_lanes_in_pipeline();
  assert(active_count <= m_core->get_config()->warp_size);
  m_core->incspactivelanes_stat(active_count);
  m_core->incfuactivelanes_stat(active_count);
  m_core->incfumemactivelanes_stat(active_count);
}
void sfu::active_lanes_in_pipeline() {
  unsigned active_count = pipelined_simd_unit::get_active_lanes_in_pipeline();
  assert(active_count <= m_core->get_config()->warp_size);
  m_core->incsfuactivelanes_stat(active_count);
  m_core->incfuactivelanes_stat(active_count);
  m_core->incfumemactivelanes_stat(active_count);
}

void tensor_core::active_lanes_in_pipeline() {
  unsigned active_count = pipelined_simd_unit::get_active_lanes_in_pipeline();
  assert(active_count <= m_core->get_config()->warp_size);
  m_core->incsfuactivelanes_stat(active_count);
  m_core->incfuactivelanes_stat(active_count);
  m_core->incfumemactivelanes_stat(active_count);
}

/*
SP��Ԫ�Ĺ��캯������m_name��ͬ��
*/
sp_unit::sp_unit(register_set *result_port, const shader_core_config *config,
                 shader_core_ctx *core, unsigned issue_reg_id)
    : pipelined_simd_unit(result_port, config, config->max_sp_latency, core,
                          issue_reg_id) {
  m_name = "SP ";
}

specialized_unit::specialized_unit(register_set *result_port,
                                   const shader_core_config *config,
                                   shader_core_ctx *core, int supported_op,
                                   char *unit_name, unsigned latency,
                                   unsigned issue_reg_id)
    : pipelined_simd_unit(result_port, config, latency, core, issue_reg_id) {
  m_name = unit_name;
  m_supported_op = supported_op;
}

/*
DP��Ԫ�Ĺ��캯������m_name��ͬ��
*/
dp_unit::dp_unit(register_set *result_port, const shader_core_config *config,
                 shader_core_ctx *core, unsigned issue_reg_id)
    : pipelined_simd_unit(result_port, config, config->max_dp_latency, core,
                          issue_reg_id) {
  m_name = "DP ";
}

/*
INT��Ԫ�Ĺ��캯������m_name��ͬ��
*/
int_unit::int_unit(register_set *result_port, const shader_core_config *config,
                   shader_core_ctx *core, unsigned issue_reg_id)
    : pipelined_simd_unit(result_port, config, config->max_int_latency, core,
                          issue_reg_id) {
  m_name = "INT ";
}

void sp_unit ::issue(register_set &source_reg) {
  warp_inst_t **ready_reg =
      source_reg.get_ready(m_config->sub_core_model, m_issue_reg_id);
  // m_core->incexecstat((*ready_reg));
  (*ready_reg)->op_pipe = SP__OP;
  m_core->incsp_stat(m_core->get_config()->warp_size, (*ready_reg)->latency);
  pipelined_simd_unit::issue(source_reg);
}

void dp_unit ::issue(register_set &source_reg) {
  warp_inst_t **ready_reg =
      source_reg.get_ready(m_config->sub_core_model, m_issue_reg_id);
  // m_core->incexecstat((*ready_reg));
  (*ready_reg)->op_pipe = DP__OP;
  m_core->incsp_stat(m_core->get_config()->warp_size, (*ready_reg)->latency);
  pipelined_simd_unit::issue(source_reg);
}

void specialized_unit ::issue(register_set &source_reg) {
  warp_inst_t **ready_reg =
      source_reg.get_ready(m_config->sub_core_model, m_issue_reg_id);
  // m_core->incexecstat((*ready_reg));
  (*ready_reg)->op_pipe = SPECIALIZED__OP;
  m_core->incsp_stat(m_core->get_config()->warp_size, (*ready_reg)->latency);
  pipelined_simd_unit::issue(source_reg);
}

void int_unit ::issue(register_set &source_reg) {
  warp_inst_t **ready_reg =
      source_reg.get_ready(m_config->sub_core_model, m_issue_reg_id);
  // m_core->incexecstat((*ready_reg));
  (*ready_reg)->op_pipe = INTP__OP;
  m_core->incsp_stat(m_core->get_config()->warp_size, (*ready_reg)->latency);
  pipelined_simd_unit::issue(source_reg);
}

/*
��ˮ�ߵ�Ԫ���캯����
*/
pipelined_simd_unit::pipelined_simd_unit(register_set *result_port,
                                         const shader_core_config *config,
                                         unsigned max_latency,
                                         shader_core_ctx *core,
                                         unsigned issue_reg_id)
    : simd_function_unit(config) {
  m_result_port = result_port;
  m_pipeline_depth = max_latency;
  //m_pipeline_reg��һ�����飬������Ĵ�Сģ����ˮ�ߵ���ȣ�ÿ��Ԫ����һ��warp_inst_t���͵�ָ�롣
  m_pipeline_reg = new warp_inst_t *[m_pipeline_depth];
  for (unsigned i = 0; i < m_pipeline_depth; i++)
    m_pipeline_reg[i] = new warp_inst_t(config);
  m_core = core;
  m_issue_reg_id = issue_reg_id;
  active_insts_in_pipeline = 0;
}

/*
��ˮ�ߵ�Ԫ��ǰ�ƽ�һ�ġ�
m_pipeline_reg�Ķ��壺warp_inst_t **m_pipeline_reg;
�����������飺
1. ���dispatch_reg��Ϊ�գ�����dispatch delay����ɣ��������Ľ��ӵ��ȼĴ����ƶ�����ˮ�ߵĵ����ӳٽ׶Ρ�
2. ���ڲ���ˮ�߼Ĵ���֮���ƶ�ָ�
3. ������һ���ڲ���ˮ�߼Ĵ�����Ϊ�գ����䷢�͵�����˿ڣ�ͨ��ΪEX_WB��ˮ�߼Ĵ���������

Dispatch Delay��
��V100��trace.config�ļ��У�
    #tensor unit
    -specialized_unit_3 1,4,8,4,4,TENSOR
    -trace_opcode_latency_initiation_spec_op_3 8,4      # <latency,initiation>

�ڵڶ����У���������ֵ��8��4��ǰ����latency��������initiation_interval��initiation_interval�ǵ����ӳ٣�
latency�ǹܵ��йܵ��׶ε����������仰˵��ÿ��initiation_interval���������ܵ�Ԫ����ָ�ָ��ͨ�����ܵ�
Ԫ��Ҫ�ȴ����ڡ��˴���start_stage��
    int start_stage = m_dispatch_reg->latency - m_dispatch_reg->initiation_interval;
˵��ָ����ÿ�����ڶ�Ҫ������ˮ�߽׶Σ������ȵ�Ԫ����������initiation_interval�����в��ܸ��ġ�

�� ����V100�У�����spec_op_3���͵�ָ��ִ����Ҫ8�ģ��ڷ���ָ���ָ�����m_dispatch_reg���ȼĴ�����Ȼ��
�ڵ��ȼĴ�����ȴ�4�ģ���4���ڲ�������spec_op_3����ָ�������ȼĴ�����4�ĺ󣬸�ָ�����m_pipeline_reg
�ĵ�8-4=4�Ųۣ�Ȼ����m_pipeline_reg�еȴ�4��->3��->2��->1�۹�4�ĺ󣬸�ָ�����EX_WB��ˮ�߼Ĵ�������
*/
void pipelined_simd_unit::cycle() {
  // pipeline reg 0 is not empty
  //�������move_warp(m_pipeline_reg[stage], m_pipeline_reg[stage + 1])���Կ�����m_pipeline_reg[0]
  //��ģ����ˮ����ȼ�ִ���ӳٵ����һ���ۣ���ˣ����ж�m_pipeline_reg�Ƿ���EX_WB�׶η���ʱ��Ҫ����0�Ų�
  //�Ƿ�Ϊ�ա�
  if (!m_pipeline_reg[0]->empty()) {
    // put m_pipeline_reg[0] to the EX_WB reg
    //��m_pipeline_reg[0]�е�ִ������EX_WB��ˮ�߼Ĵ�������
    m_result_port->move_in(m_pipeline_reg[0]);
    assert(active_insts_in_pipeline > 0);
    //m_pipeline_reg[0]�е�ָ���Ƴ�����ˮ���еĻ�Ծָ������1��
    active_insts_in_pipeline--;
  }
  // move warp_inst_t through out the pipeline
  //m_pipeline_reg��ˮ�߼Ĵ������е�����ָ����ǰ�ƽ�һ�ۣ�ģ��һ�ĵ�ִ�С�
  if (active_insts_in_pipeline) {
    for (unsigned stage = 0; (stage + 1) < m_pipeline_depth; stage++)
      move_warp(m_pipeline_reg[stage], m_pipeline_reg[stage + 1]);
  }
  // If the dispatch_reg is not empty
  //���dispatch_reg��Ϊ�գ���������m_pipeline_reg��ˮ�ߡ�
  //���������ĸ�λ�ã�Ҫ��ָ���ӳټ�ȥ��m_dispatch_reg�еĳ�ʼ����������ʼ���������ָ������������õġ�
  if (!m_dispatch_reg->empty()) {
    // If not dispatch_delay
    if (!m_dispatch_reg->dispatch_delay()) {
      // during dispatch delay, the warp is still moving through the pipeline,
      // though the dispatch_reg cannot be changed
      int start_stage =
          m_dispatch_reg->latency - m_dispatch_reg->initiation_interval;
      if (m_pipeline_reg[start_stage]->empty()) {
      //��m_dispatch_reg����m_pipeline_reg��ˮ�ߡ�
        move_warp(m_pipeline_reg[start_stage], m_dispatch_reg);
      //ָ������m_pipeline_reg����ˮ���еĻ�Ծָ������1��
        active_insts_in_pipeline++;
      }
    }
  }
  //m_dispatch_reg�ı�ʶռ��λͼ��״̬����һλ��ģ��һ�ĵ��ƽ���
  occupied >>= 1;
}

/*
��warp_inst_t���͵�ָ������dispatch�Ĵ�����
*/
void pipelined_simd_unit::issue(register_set &source_reg) {
  // move_warp(m_dispatch_reg,source_reg);
  //sub_core_model: github.com/accel-sim/accel-sim-framework/blob/dev/gpu-simulator/gpgpu-sim4.md
  bool partition_issue =
      m_config->sub_core_model && this->is_issue_partitioned();
  warp_inst_t **ready_reg =
      source_reg.get_ready(partition_issue, m_issue_reg_id);
  m_core->incexecstat((*ready_reg));
  // source_reg.move_out_to(m_dispatch_reg);
  simd_function_unit::issue(source_reg);
}

/*
    virtual void issue( register_set& source_reg )
    {
        //move_warp(m_dispatch_reg,source_reg);
        //source_reg.move_out_to(m_dispatch_reg);
        simd_function_unit::issue(source_reg);
    }
*/
/*
ldst��Ԫ�ĳ�ʼ��������
*/
void ldst_unit::init(mem_fetch_interface *icnt,
                     shader_core_mem_fetch_allocator *mf_allocator,
                     shader_core_ctx *core, opndcoll_rfu_t *operand_collector,
                     Scoreboard *scoreboard, const shader_core_config *config,
                     const memory_config *mem_config, shader_core_stats *stats,
                     unsigned sid, unsigned tpc) {
  m_memory_config = mem_config;
  m_icnt = icnt;
  m_mf_allocator = mf_allocator;
  m_core = core;
  m_operand_collector = operand_collector;
  m_scoreboard = scoreboard;
  m_stats = stats;
  m_sid = sid;
  m_tpc = tpc;
#define STRSIZE 1024
  char L1T_name[STRSIZE];
  char L1C_name[STRSIZE];
  snprintf(L1T_name, STRSIZE, "L1T_%03d", m_sid);
  snprintf(L1C_name, STRSIZE, "L1C_%03d", m_sid);
  //L1�����档
  m_L1T = new tex_cache(L1T_name, m_config->m_L1T_config, m_sid,
                        get_shader_texture_cache_id(), icnt, IN_L1T_MISS_QUEUE,
                        IN_SHADER_L1T_ROB);
  //L1�������档
  m_L1C = new read_only_cache(L1C_name, m_config->m_L1C_config, m_sid,
                              get_shader_constant_cache_id(), icnt,
                              IN_L1C_MISS_QUEUE, OTHER_GPU_CACHE, m_gpu);
  //L1���ݻ��档
  m_L1D = NULL;
  m_mem_rc = NO_RC_FAIL;
  m_num_writeback_clients =
      5;  // = shared memory, global/local (uncached), L1D, L1T, L1C
  m_writeback_arb = 0;
  m_next_global = NULL;
  m_last_inst_gpu_sim_cycle = 0;
  m_last_inst_gpu_tot_sim_cycle = 0;
}

/*
ldst ��Ԫ�Ĺ��캯����
*/
ldst_unit::ldst_unit(mem_fetch_interface *icnt,
                     shader_core_mem_fetch_allocator *mf_allocator,
                     shader_core_ctx *core, opndcoll_rfu_t *operand_collector,
                     Scoreboard *scoreboard, const shader_core_config *config,
                     const memory_config *mem_config, shader_core_stats *stats,
                     unsigned sid, unsigned tpc, gpgpu_sim *gpu)
    : pipelined_simd_unit(NULL, config, config->smem_latency, core, 0),
      m_next_wb(config),
      m_gpu(gpu) {
  assert(config->smem_latency > 1);
  init(icnt, mf_allocator, core, operand_collector, scoreboard, config,
       mem_config, stats, sid, tpc);
  if (!m_config->m_L1D_config.disabled()) {
    char L1D_name[STRSIZE];
    snprintf(L1D_name, STRSIZE, "L1D_%03d", m_sid);
    m_L1D = new l1_cache(L1D_name, m_config->m_L1D_config, m_sid,
                         get_shader_normal_cache_id(), m_icnt, m_mf_allocator,
                         IN_L1D_MISS_QUEUE, core->get_gpu(), L1_GPU_CACHE);

    l1_latency_queue.resize(m_config->m_L1D_config.l1_banks);
    assert(m_config->m_L1D_config.l1_latency > 0);

    for (unsigned j = 0; j < m_config->m_L1D_config.l1_banks; j++)
      l1_latency_queue[j].resize(m_config->m_L1D_config.l1_latency,
                                 (mem_fetch *)NULL);
  }
  m_name = "MEM ";
}

ldst_unit::ldst_unit(mem_fetch_interface *icnt,
                     shader_core_mem_fetch_allocator *mf_allocator,
                     shader_core_ctx *core, opndcoll_rfu_t *operand_collector,
                     Scoreboard *scoreboard, const shader_core_config *config,
                     const memory_config *mem_config, shader_core_stats *stats,
                     unsigned sid, unsigned tpc, l1_cache *new_l1d_cache)
    : pipelined_simd_unit(NULL, config, 3, core, 0),
      m_L1D(new_l1d_cache),
      m_next_wb(config) {
  init(icnt, mf_allocator, core, operand_collector, scoreboard, config,
       mem_config, stats, sid, tpc);
}

/*
LDST ��Ԫ issue ������
*/
void ldst_unit::issue(register_set &reg_set) {
  // �ӼĴ������� reg_set �л�ȡһ���ǿռĴ��������� inst ָ���Ƴ�������������ָ�
  warp_inst_t *inst = *(reg_set.get_ready());

  // record how many pending register writes/memory accesses there are for this
  // instruction
  // ��¼��ָ���ж��ٸ�����ļĴ���д��/�ڴ���ʡ�
  assert(inst->empty() == false);
  // �����ָ���� load ָ����Ҹ�ָ��Ŀռ����Ͳ��ǹ����ڴ�ռ䡣
  if (inst->is_load() and inst->space.get_type() != shared_space) {
    // ��ָ��� warp_id��
    unsigned warp_id = inst->warp_id();
    // ��ָ��ķô������inst->accessq_count() = m_accessq.size()������ m_accessq 
    // �ǵ�ǰָ��ķô�������б�
    unsigned n_accesses = inst->accessq_count();
    // �Ը���ָ�������Ŀ�ļĴ���ѭ����
    for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++) {
      // ����ָ���Ŀ�ļĴ��� ID��inst->out ��¼�˵�ǰָ�������Ŀ�Ĳ������Ĵ��� ID��
      unsigned reg_id = inst->out[r];
      if (reg_id > 0) {
        // m_pending_writes ��һ����ά���飬��һά������ warp_id���ڶ�ά�����ǼĴ�
        // �� ID��ֵ�ǸüĴ��� ID ��Ӧ�ļĴ��������д���������ָ���Ŀ�ļĴ��� ID 
        // ��Ӧ�� m_pending_writes ����Ĺ����д��������ϸ�ָ���д������
        m_pending_writes[warp_id][reg_id] += n_accesses;
      }
    }
    if (inst->m_is_ldgsts) {
      m_pending_ldgsts[warp_id][inst->pc][inst->get_addr(0)] += n_accesses;
    }
  }
  // inst->op_pipe �ǲ����� code (uarch �ɼ�)����ʶ��������ˮ��(SP��SFU �� MEM)��
  inst->op_pipe = MEM__OP;
  // stat collection
  m_core->mem_instruction_stats(*inst);
  m_core->incmem_stat(m_core->get_config()->warp_size, 1);
  //LDST��Ԫ��ˮ�ߵ�Ԫ��ǰ�ƽ�һ�ġ�
  pipelined_simd_unit::issue(reg_set);
}

/*
LDST ��Ԫ��д�ز�����
*/
void ldst_unit::writeback() {
  // process next instruction that is going to writeback
  // ��һ����Ҫд�ص�ָ�
  // m_next_wb => *pipline_reg[0]
  if (!m_next_wb.empty()) {
    // �������ռ����� Bank д�أ����� true��Bank ��ͻʱ���� false��
    // if (m_operand_collector->writeback(m_next_wb)) {                      // yangjianchao16 del
    if (m_operand_collector->writeback(m_next_wb)) {
      bool insn_completed = false;
      // �Ը���ָ�� m_next_wb ������Ŀ�ļĴ���ѭ����
      for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++) {
        if (m_next_wb.out[r] > 0) {
          // ���ָ��д�صĵ�ַ�Ƿ��ǹ����ڴ�ռ䡣
          if (m_next_wb.space.get_type() != shared_space) {
            //ָ��д�صĵ�ַ���ǹ����ڴ�ռ䡣
            assert(m_pending_writes[m_next_wb.warp_id()][m_next_wb.out[r]] > 0);
            //�����m_pending_writes��һ����ά���飬��һά������warp_id���ڶ�ά�����ǼĴ���ID��ֵ�Ǹü�
            //����ID��Ӧ�ļĴ��������д���������������ִ���˲������ռ�����д�ز�����m_next_wb����warp��
            //�üĴ���ID��Ӧ�ļĴ��������д�������1��still_pending�����ڵ�ǰд��ִ����Ϻ�m_next_wb
            //����warp�¸üĴ���ID��Ӧ�ļĴ��������д�������
            //��ĳ��ָ����ʷ�shared memory�ռ�ʱ����Ҫ����ָ�������Ŀ�ļĴ�����д�أ���ִ��ldst_unit::
            //issue()����ʱ����ָ�������Ŀ�ļĴ�����д�ش������ᱻ�ӵ�m_pending_writes�У���ˣ���ָ���
            //����Ŀ�ļĴ���������m_pending_writes���б����Щ�����ֵ���Ǹ�ָ��ķô������
            unsigned still_pending =
                --m_pending_writes[m_next_wb.warp_id()][m_next_wb.out[r]];
            //����ù���д������Ѿ���Ϊ0����˵���Ѿ�û�й����д�룬�üĴ������ڵ�m_pending_writes�еı�
            //����Ա��ͷţ��üĴ���ҲӦ�ôӼƷְ����ͷš�
            if (!still_pending) {
              m_pending_writes[m_next_wb.warp_id()].erase(m_next_wb.out[r]);
              m_scoreboard->releaseRegister(m_next_wb.warp_id(),
                                            m_next_wb.out[r]);
              //����ָ������ɡ�
              insn_completed = true;
            }
          } else {  // shared
            // ָ��д�صĵ�ַ�ǹ����ڴ�ռ䣬û�мĴ������룬��ʱ��ֱ�ӴӼƷְ����ͷŸüĴ������ɡ�
            m_scoreboard->releaseRegister(m_next_wb.warp_id(),
                                          m_next_wb.out[r]);
            insn_completed = true;
          }
        } else if (m_next_wb.m_is_ldgsts) {  // for LDGSTS instructions where no
                                             // output register is used
          m_pending_ldgsts[m_next_wb.warp_id()][m_next_wb.pc]
                          [m_next_wb.get_addr(0)]--;
          if (m_pending_ldgsts[m_next_wb.warp_id()][m_next_wb.pc]
                              [m_next_wb.get_addr(0)] == 0) {
            insn_completed = true;
          }
          break;
        }
      }
      //����������ռ�����д�ز�����ɣ�����ǰָ���Ѿ���ɡ�
      if (insn_completed) {
        m_core->warp_inst_complete(m_next_wb);
        if (m_next_wb.m_is_ldgsts) {
          m_core->unset_depbar(m_next_wb);
        }
      }

      m_next_wb.clear();
      m_last_inst_gpu_sim_cycle = m_core->get_gpu()->gpu_sim_cycle;
      m_last_inst_gpu_tot_sim_cycle = m_core->get_gpu()->gpu_tot_sim_cycle;
    }
  }

  unsigned serviced_client = -1;
  //m_num_writeback_clients = 5��
  //  next_client=0��shared memory,
  //  next_client=1��L1T,   
  //  next_client=2��L1C,
  //  next_client=3��global/local (uncached), 
  //  next_client=4��L1D��
  // �����ǻ�ȡ��һ����Ҫд�ص�ָ�������ѯ��������5��client������ѡ��һ����д�ص�ָ�
  for (unsigned c = 0; m_next_wb.empty() && (c < m_num_writeback_clients);
       c++) {
    // m_writeback_arb ��һ����������������ѯ������һ����Ҫд�صĵ�Ԫ����һ�����ҵĵ�Ԫ��
    // next_client��
    unsigned next_client = (c + m_writeback_arb) % m_num_writeback_clients;
    switch (next_client) {
      // next_client can be chosen as: 
      //     0 - shared memory, 
      //     3 - global/local (uncached), 
      //     4 - L1D, 
      //     1 - L1T, 
      //     2 - L1C.
      case 0:  // shared memory
        // ֱ���ж���ˮ�߼Ĵ������ɡ���Ҫע����ǣ�����ֻ�� shared memory ��ָ��Ż�ŵ� 
        // m_pipeline_reg ��ģ���ӳ٣��������͵ķô�ָ������ L1T/L1C/L1D/global/local 
        // ����ר��ģ�����ǵ�ר�Ų�����
        if (!m_pipeline_reg[0]->empty()) {
          // ����Ҫд�ص�ָ����� m_next_wb �У�����һ�ĵ� ldst_unit::write_back() ��ʼ
          // ��������ָ���д�ز�����
          m_next_wb = *m_pipeline_reg[0];
          // �� cuda-sim.cc �ж���� ptx_thread_info::ptx_exec_inst �����л�ȷ����ǰ
          // ָ��Ĳ������ǲ��� ATOM_OP �����룬�������Ϊ atomic ������
          if (m_next_wb.isatomic()) {
            /* do_atomic ��������Ϊ��
                void warp_inst_t::do_atomic(const active_mask_t &access_mask, 
                                            bool forceDo) {
                  assert(m_isatomic && (!m_empty || forceDo));
                  if (!should_do_atomic) return;
                  for (unsigned i = 0; i < m_config->warp_size; i++) {
                    if (access_mask.test(i)) {
                      dram_callback_t &cb = m_per_scalar_thread[i].callback;
                      if (cb.thread) cb.function(cb.instruction, cb.thread);
                    }
                  }
                }
              ��ǰ�汾�� should_do_atomic ʼ����Ϊ true����������� do_atomic ��Ч����
              gpgpusim ��ʱ��ģ�� atomic ������
            */
            m_next_wb.do_atomic();
            m_core->decrement_atomic_count(m_next_wb.warp_id(),
                                           m_next_wb.active_count());
          }
          m_core->dec_inst_in_pipeline(m_pipeline_reg[0]->warp_id());
          // ����ǰ��� shared memory �ķô�ָ����� m_dispatch_reg ��ģ�⿼�� bank ÿ
          // ��ֻ���ṩһ�� word �ķ��ʣ����� m_pipeline_reg ��ģ����� shared memory 
          // �ķ����ӳ١������� shared memory �ķô��ӳ٣�����ӳ����� gpgpu_l1_latency 
          // ���ò��������ģ���������ӳ���ִ����ˮ��ģ�͵� m_pipeline_reg ��ģ�⡣����һ
          // ���ô� shared memory ָ��� initial interval ��ģ��� bank conflict����
          // �ᱻ�Ž� m_pipeline_reg[gpgpu_l1_latency-1]������������ m_pipeline_reg[0] 
          // ʱ����˵�����ָ���Ѿ�����˷ô����������д���ˡ�

          // ���������д�ز���û�п��� shared memory load �����Ľ����Ҫд���Ĵ�����
          m_pipeline_reg[0]->clear();
          serviced_client = next_client;
        }
        break;
      case 1:  // texture response
        if (m_L1T->access_ready()) {
          // ��ȡ��һ�����ʵ�����ָ�
          mem_fetch *mf = m_L1T->next_access();
          // ����Ҫд�ص�ָ����� m_next_wb �У�����һ�ĵ� ldst_unit::write_back() ��ʼ
          // ��������ָ���д�ز�����
          m_next_wb = mf->get_inst();
          delete mf;
          serviced_client = next_client;
        }
        break;
      case 2:  // const cache response
        if (m_L1C->access_ready()) {
          // ��ȡ��һ�����ʵĳ�������ָ�
          mem_fetch *mf = m_L1C->next_access();
          // ����Ҫд�ص�ָ����� m_next_wb �У�����һ�ĵ� ldst_unit::write_back() ��ʼ
          // ��������ָ���д�ز�����
          m_next_wb = mf->get_inst();
          delete mf;
          serviced_client = next_client;
        }
        break;
      case 3:  // global/local
        if (m_next_global) {
          // ��ȡ��һ�����ʵ�ȫ���ڴ�ָ��� ldst_unit::cycle() �����У��� ldst_unit ��
          // �� m_response_fifo �ǿգ������е� mem_fetch ָ���� global/local �ڴ�ָ��ʱ��
          // �Ὣ��ָ����� m_next_global �С�
          // ����Ҫд�ص�ָ����� m_next_wb �У�����һ�ĵ� ldst_unit::write_back() ��ʼ
          // ��������ָ���д�ز�����
          m_next_wb = m_next_global->get_inst();
          if (m_next_global->isatomic()) {
            m_core->decrement_atomic_count(
                m_next_global->get_wid(),
                m_next_global->get_access_warp_mask().count());
          }
          delete m_next_global;
          m_next_global = NULL;
          serviced_client = next_client;
        }
        break;
      case 4:
        if (m_L1D && m_L1D->access_ready()) {
          // ��ȡ��һ�����ʵ����ݻ���ָ�
          mem_fetch *mf = m_L1D->next_access();
          // ����Ҫд�ص�ָ����� m_next_wb �У�����һ�ĵ� ldst_unit::write_back() ��ʼ
          // ��������ָ���д�ز�����
          m_next_wb = mf->get_inst();
          delete mf;
          serviced_client = next_client;
        }
        break;
      default:
        abort();
    }
  }
  // update arbitration priority only if:
  // 1. the writeback buffer was available
  // 2. a client was serviced
  if (serviced_client != (unsigned)-1) {
    m_writeback_arb = (serviced_client + 1) % m_num_writeback_clients;
  }
}

// ʱ�ӱ�������һЩ��Ԫ�����ڸ��ߵ�ѭ�����������С�
unsigned ldst_unit::clock_multiplier() const {
  // to model multiple read port, we give multiple cycles for the memory units
  // �� V100 �����У�m_config->mem_unit_ports Ĭ��Ϊ 1��
  if (m_config->mem_unit_ports)
    return m_config->mem_unit_ports;
  else
    // m_config->mem_warp_parts: Number of portions a warp is divided into for 
    // shared memory bank conflict check.
    return m_config->mem_warp_parts;
}
/*
void ldst_unit::issue( register_set &reg_set )
{
        warp_inst_t* inst = *(reg_set.get_ready());
   // stat collection
   m_core->mem_instruction_stats(*inst);

   // record how many pending register writes/memory accesses there are for this
instruction assert(inst->empty() == false); if (inst->is_load() and
inst->space.get_type() != shared_space) { unsigned warp_id = inst->warp_id();
      unsigned n_accesses = inst->accessq_count();
      for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++) {
         unsigned reg_id = inst->out[r];
         if (reg_id > 0) {
            m_pending_writes[warp_id][reg_id] += n_accesses;
         }
      }
   }

   pipelined_simd_unit::issue(reg_set);
}
*/
void ldst_unit::cycle() {
  // LDST ��Ԫ��д�ز�����
  writeback();
  // for (int i = 0; i < m_config->reg_file_port_throughput; ++i)
  //   m_operand_collector->step();
  // m_pipeline_reg ��һ�����飬������Ĵ�Сģ����ˮ�ߵ���� m_pipeline_depth��ÿ��Ԫ����
  // һ�� warp_inst_t ���͵�ָ�롣������ m_pipeline_reg ��ˮ�߼Ĵ������е�����ָ����ǰ�ƽ�
  // һ�ۣ�ģ��һ�ĵ�ִ�С�

  for (unsigned stage = 0; (stage + 1) < m_pipeline_depth; stage++)
    if (m_pipeline_reg[stage]->empty() && !m_pipeline_reg[stage + 1]->empty())
      move_warp(m_pipeline_reg[stage], m_pipeline_reg[stage + 1]);

  // ��� LDST ��Ԫ��������Ӧ FIFO ��Ϊ�ա������ m_response_fifo ������ ldst_unit 
  // ���һ����Ա���������ǽ��� SIMT Core Cluster �� m_response_fifo ���͵� mf ��
  // �����ݰ���SIMT Core Cluster ������ L1I Cache ���� INST_ACC_R ���͵ķô����ݰ���
  if (!m_response_fifo.empty()) {
    //��ȡFIFO�еĵ�һ��mem_fetchָ�롣
    mem_fetch *mf = m_response_fifo.front();
    if (mf->get_access_type() == TEXTURE_ACC_R) {
      //����÷ô�mem������ô�������ж��������Ƿ��п��е����˿ڡ�
      if (m_L1T->fill_port_free()) {
        //������ڿ��е����˿ڣ��򽫸�mem_fetchָ��mf�е�ָ�����������档
        m_L1T->fill(mf, m_core->get_gpu()->gpu_sim_cycle +
                            m_core->get_gpu()->gpu_tot_sim_cycle);
        m_response_fifo.pop_front();
      }
    } else if (mf->get_access_type() == CONST_ACC_R) {
      //����÷ô�mem�ǳ����ô�������жϳ��������Ƿ��п��е����˿ڡ�
      if (m_L1C->fill_port_free()) {
        mf->set_status(IN_SHADER_FETCHED,
                       m_core->get_gpu()->gpu_sim_cycle +
                           m_core->get_gpu()->gpu_tot_sim_cycle);
        m_L1C->fill(mf, m_core->get_gpu()->gpu_sim_cycle +
                            m_core->get_gpu()->gpu_tot_sim_cycle);
        m_response_fifo.pop_front();
      }
    } else {
      //mf->get_type()���صĿ������ͣ�
      //    enum mf_type {
      //      READ_REQUEST = 0,
      //      WRITE_REQUEST,
      //      READ_REPLY,  // send to shader
      //      WRITE_ACK
      //    };
      if (mf->get_type() == WRITE_ACK ||
          //m_config->gpgpu_perfect_mem��V100������Ĭ��Ϊ0��
          (m_config->gpgpu_perfect_mem && mf->get_is_write())) {
        //����÷ô�mem��д��Ӧ��ִ��Shader Core��SM����дȷ�ϵĶ���������mf����warp���ѷ��͵���δ�յ�
        //дȷ�ϵĴ洢��������
        m_core->store_ack(mf);
        m_response_fifo.pop_front();
        delete mf;
      } else {
        //����÷ô�mem����д��Ӧ������READ_REQUEST��WRITE_REQUEST��READ_REPLY��
        assert(!mf->get_is_write());  // L1 cache is write evict, allocate line
                                      // on load miss only

        //bypassL1D��ָ�����ݷ����Ƿ��ƹ�L1���ݻ��档
        bool bypassL1D = false;
        if (CACHE_GLOBAL == mf->get_inst().cache_op || (m_L1D == NULL)) {
          //CACHE_GLOBAL==.cg��Cache at global level��ȫ�ּ����棨L2�����»��棬������L1����ʹ��ld.cg
          //ȫ���Եؼ��أ��ƹ�L1���棬����������L2�����С�
          bypassL1D = true;
        } else if (mf->get_access_type() == GLOBAL_ACC_R ||
                   mf->get_access_type() ==
                       GLOBAL_ACC_W) {  // global memory access
          //��V100�����У�gpgpu_gmem_skip_L1D����Ϊ0��
          if (m_core->get_config()->gmem_skip_L1D) bypassL1D = true;
        }
        //����ֻ����ʹ��ld.cgʱ�Ż��ƹ�L1���ݻ��档
        if (bypassL1D) {
          // m_next_global �ݴ���һ�η���ȫ�ִ洢�� mf�����壺mem_fetch *ldst_unit::m_next_global�� 
          if (m_next_global == NULL) {
            //m_next_global����һ�η���ȫ�ִ洢��mf����������У���m_next_global = mf�����еĻ����޷�
            //�����ų��ôη��ʡ�
            mf->set_status(IN_SHADER_FETCHED,
                           m_core->get_gpu()->gpu_sim_cycle +
                               m_core->get_gpu()->gpu_tot_sim_cycle);
            m_response_fifo.pop_front();
            m_next_global = mf;
          }
        } else {
          //������ƹ�L1���ݻ��棬����Ҫ��mfд��L1���ݻ���Ķ˿ڡ�
          if (m_L1D->fill_port_free()) {
            m_L1D->fill(mf, m_core->get_gpu()->gpu_sim_cycle +
                                m_core->get_gpu()->gpu_tot_sim_cycle);
            m_response_fifo.pop_front();
          }
        }
      }
    }
  }

  //L1��������ǰ�ƽ�һ�ġ�
  m_L1T->cycle();
  //L1����������ǰ�ƽ�һ�ġ�
  m_L1C->cycle();
  if (m_L1D) {
    m_L1D->cycle();
    if (m_config->m_L1D_config.l1_latency > 0) L1_latency_queue_cycle();
  }

  // ÿ��ִ�е�Ԫģ�ͣ����� INT/SP �ȣ�����һ�� m_dispatch_reg��m_dispatch_reg ��
  // һ�� warp_inst_t ���͵�ָ�룬warp ����������ָ���Լ���ȡ��������Ϻ󣬷��䵽ÿ
  // ��ִ�е�Ԫ��ָ��ᱻ���뵽��Ӧ��ִ�е�Ԫ�� m_dispatch_reg �У�Ȼ��ָ������
  // m_dispatch_reg �б�ȡ����������ģ��ִ����ˮ���ӳٵ� m_pipeline_reg �С�
  /*
                  m_dispatch_reg    m_pipeline_reg      
                      / |            |---------|
                     /  |----------> |         | 31  --|
                    /   |            |---------|       |
    Dispatch done every |----------> |         | :     |
    Issue_interval cycle|            |---------|       |  Pipeline registers to
    to model instruction|----------> |         | 2     |- model instruction latency
    throughput          |            |---------|       |
                        |----------> |         | 1     |
                        |            |---------|       |
                        |----------> |         | 0   --|
    Dispatch position =              |---------|
    Latency - Issue_interval             |
                                        \|/
                                         V
                                    m_result_port --> write back stage
  */
  warp_inst_t &pipe_reg = *m_dispatch_reg;
  //mem_stage_stall_type�Ķ��壺
  //    enum mem_stage_stall_type { //MEM��ˮ�߲����Stall���͡�
  //      NO_RC_FAIL = 0,           //û��Stall
  //      BK_CONF,                  //Bank conflict
  //      MSHR_RC_FAIL,             //MSHR��Դ��ͻ
  //      ICNT_RC_FAIL,             //Interconnect��Դ��ͻ
  //      COAL_STALL,               //Coalescing Stall
  //      TLB_STALL,                //TLB Stall
  //      DATA_PORT_STALL,          //���ݶ˿�Stall
  //      WB_ICNT_RC_FAIL,          //Writeback Interconnect��Դ��ͻ
  //      WB_CACHE_RSRV_FAIL,       //Writeback Cache��Դ��ͻ
  //      N_MEM_STAGE_STALL_TYPE    //Stall��������
  //    };
  enum mem_stage_stall_type rc_fail = NO_RC_FAIL;
  mem_stage_access_type type;
  bool done = true;
  //�ж�LDST��Ԫ����shared memory�Ƿ��Stall��Stall�Ļ�����0��û��Stall�Ļ�����1��
  done &= shared_cycle(pipe_reg, rc_fail, type);
  done &= constant_cycle(pipe_reg, rc_fail, type);
  done &= texture_cycle(pipe_reg, rc_fail, type);
  //�ж�LDST��Ԫ����global/local/param memory�Ƿ��Stall��Stall�Ļ�����0��û��Stall�Ļ�����1��
  done &= memory_cycle(pipe_reg, rc_fail, type);
  m_mem_rc = rc_fail;

  // ��� done Ϊ 0 ˵��ǰ�淢���� stall������� done ��ʵֻ������ִ�� shared_cycle��
  // constant_cycle��texture_cycle��memory_cycle ���ĸ������е�һ������Ϊ���� LDST
  // ��Ԫ�� m_dispatch_reg �е�ָ����˵����ʵ��ֻ�����������ַô������е�һ�֡������
  // ������ִ�С���ʵ��ָ��������Ӧ��������������Ҫ�����⼸���������⼸���������жϴ����
  // ָ���Ƿ��Ƕ�Ӧ�ķô����ͣ�����ǣ��ͻ�ִ����Ӧ�Ķ�����������ǣ��ͻ�ֱ�ӷ��� true��
  // ��ʾ�����ڵ�ǰ��Ԫִ�У�����ִ�ж�����
  if (!done) {  // log stall types and return
    assert(rc_fail != NO_RC_FAIL);
    m_stats->gpgpu_n_stall_shd_mem++;
    m_stats->gpu_stall_shd_mem_breakdown[type][rc_fail]++;
    return;
  }

  //���ǰ��û�з���Stall��pipe_reg�ǿա�
  if (!pipe_reg.empty()) {
    unsigned warp_id = pipe_reg.warp_id();
    //���pipe_reg��loadָ�
    if (pipe_reg.is_load()) {
      if (pipe_reg.space.get_type() == shared_space) {
        //���pipe_reg��shared memory�ռ��loadָ�
        if (m_pipeline_reg[m_config->smem_latency - 1]->empty()) {
          // new shared memory request
          move_warp(m_pipeline_reg[m_config->smem_latency - 1], m_dispatch_reg);
          m_dispatch_reg->clear();
        }
      } else {
        // if( pipe_reg.active_count() > 0 ) {
        //    if( !m_operand_collector->writeback(pipe_reg) )
        //        return;
        //}
        //���pipe_reg��global/local/param memory�ռ��loadָ�
        bool pending_requests = false;
        for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++) {
          unsigned reg_id = pipe_reg.out[r];
          if (reg_id > 0) {
            if (m_pending_writes[warp_id].find(reg_id) !=
                m_pending_writes[warp_id].end()) {
              // this instruction has pending register writes
              if (m_pending_writes[warp_id][reg_id] > 0) {
                pending_requests = true;
                break;
              } else {
                // this instruction is done already
                m_pending_writes[warp_id].erase(reg_id);
              }
            }
          }
        }
        if (!pending_requests) {
          //�����ָ��û�й����д�룬���ָ��ִ����ϡ�
          m_core->warp_inst_complete(*m_dispatch_reg);
          m_scoreboard->releaseRegisters(m_dispatch_reg);

          // release LDGSTS
          if (m_dispatch_reg->m_is_ldgsts) {
            // m_pending_ldgsts[m_dispatch_reg->warp_id()][m_dispatch_reg->pc][m_dispatch_reg->get_addr(0)]--;
            if (m_pending_ldgsts[m_dispatch_reg->warp_id()][m_dispatch_reg->pc]
                                [m_dispatch_reg->get_addr(0)] == 0) {
              m_core->unset_depbar(*m_dispatch_reg);
            }
          }
        }
        m_core->dec_inst_in_pipeline(warp_id);
        m_dispatch_reg->clear();
      }
    } else {
      //���pipe_reg����loadָ���ִ����ϡ�
      // stores exit pipeline here
      m_core->dec_inst_in_pipeline(warp_id);
      m_core->warp_inst_complete(*m_dispatch_reg);
      m_dispatch_reg->clear();
    }
  }
}

/*
ע��CTA���̵߳��˳���
*/
void shader_core_ctx::register_cta_thread_exit(unsigned cta_num,
                                               kernel_info_t *kernel) {
  assert(m_cta_status[cta_num] > 0);
  //m_cta_status��Shader Core�ڵ�CTA��״̬��MAX_CTA_PER_SHADER��ÿ��Shader Core�ڵ����ɲ���
  //CTA������m_cta_status[i]�ﱣ���˵�i��CTA�а����Ļ�Ծ�߳��������������� <= CTA�����߳�������
  //����������Ҫע�ᵥ���̵߳��˳�����˵�i��CTA�а����Ļ�Ծ�߳�������Ӧ����1��
  m_cta_status[cta_num]--;
  //���m_cta_status[cta_num]=0����cta_num��CTA��û�л�Ծ���̡߳�
  if (!m_cta_status[cta_num]) {
    // Increment the completed CTAs
    //�����Ѿ���ɵ�CTA������
    m_stats->ctas_completed++;
    //�����Ѿ���ɵ�CTA������
    m_gpu->inc_completed_cta();
    //��С��Ծ��CTA������
    m_n_active_cta--;
    m_barriers.deallocate_barrier(cta_num);
    shader_CTA_count_unlog(m_sid, 1);

    SHADER_DPRINTF(
        LIVENESS,
        "GPGPU-Sim uArch: Finished CTA #%u (%lld,%lld), %u CTAs running\n",
        cta_num, m_gpu->gpu_sim_cycle, m_gpu->gpu_tot_sim_cycle,
        m_n_active_cta);
    //һ��û�л�Ծ��CTA�󣬴���ǰkernel�Ѿ�ִ���ꡣ
    if (m_n_active_cta == 0) {
      SHADER_DPRINTF(
          LIVENESS,
          "GPGPU-Sim uArch: Empty (last released kernel %u \'%s\').\n",
          kernel->get_uid(), kernel->name().c_str());
      fflush(stdout);

      // Shader can only be empty when no more cta are dispatched
      if (kernel != m_kernel) {
        assert(m_kernel == NULL || !m_gpu->kernel_more_cta_left(m_kernel));
      }
      //m_kernel�������ڵ�ǰSIMT Core�ϵ��ں˺�����
      m_kernel = NULL;
    }

    // Jin: for concurrent kernels on sm
    release_shader_resource_1block(cta_num, *kernel);
    kernel->dec_running();
    if (!m_gpu->kernel_more_cta_left(kernel)) {
      if (!kernel->running()) {
        SHADER_DPRINTF(LIVENESS,
                       "GPGPU-Sim uArch: GPU detected kernel %u \'%s\' "
                       "finished on shader %u.\n",
                       kernel->get_uid(), kernel->name().c_str(), m_sid);

        if (m_kernel == kernel) m_kernel = NULL;
        m_gpu->set_kernel_done(kernel);
      }
    }
  }
}

void gpgpu_sim::shader_print_runtime_stat(FILE *fout) {
  /*
 fprintf(fout, "SHD_INSN: ");
 for (unsigned i=0;i<m_n_shader;i++)
    fprintf(fout, "%u ",m_sc[i]->get_num_sim_insn());
 fprintf(fout, "\n");
 fprintf(fout, "SHD_THDS: ");
 for (unsigned i=0;i<m_n_shader;i++)
    fprintf(fout, "%u ",m_sc[i]->get_not_completed());
 fprintf(fout, "\n");
 fprintf(fout, "SHD_DIVG: ");
 for (unsigned i=0;i<m_n_shader;i++)
    fprintf(fout, "%u ",m_sc[i]->get_n_diverge());
 fprintf(fout, "\n");

 fprintf(fout, "THD_INSN: ");
 for (unsigned i=0; i<m_shader_config->n_thread_per_shader; i++)
    fprintf(fout, "%d ", m_sc[0]->get_thread_n_insn(i) );
 fprintf(fout, "\n");
 */
}

void gpgpu_sim::shader_print_scheduler_stat(FILE *fout,
                                            bool print_dynamic_info) const {
  fprintf(fout, "ctas_completed %d, ", m_shader_stats->ctas_completed);
  // Print out the stats from the sampling shader core
  const unsigned scheduler_sampling_core =
      m_shader_config->gpgpu_warp_issue_shader;
#define STR_SIZE 55
  char name_buff[STR_SIZE];
  name_buff[STR_SIZE - 1] = '\0';
  const std::vector<unsigned> &distro =
      print_dynamic_info
          ? m_shader_stats->get_dynamic_warp_issue()[scheduler_sampling_core]
          : m_shader_stats->get_warp_slot_issue()[scheduler_sampling_core];
  if (print_dynamic_info) {
    snprintf(name_buff, STR_SIZE - 1, "dynamic_warp_id");
  } else {
    snprintf(name_buff, STR_SIZE - 1, "warp_id");
  }
  fprintf(fout, "Shader %d %s issue ditsribution:\n", scheduler_sampling_core,
          name_buff);
  const unsigned num_warp_ids = distro.size();
  // First print out the warp ids
  fprintf(fout, "%s:\n", name_buff);
  for (unsigned warp_id = 0; warp_id < num_warp_ids; ++warp_id) {
    fprintf(fout, "%d, ", warp_id);
  }

  fprintf(fout, "\ndistro:\n");
  // Then print out the distribution of instuctions issued
  for (std::vector<unsigned>::const_iterator iter = distro.begin();
       iter != distro.end(); iter++) {
    fprintf(fout, "%d, ", *iter);
  }
  fprintf(fout, "\n");
}

void gpgpu_sim::shader_print_cache_stats(FILE *fout) const {
  // L1I
  struct cache_sub_stats total_css;
  struct cache_sub_stats css;

  if (!m_shader_config->m_L1I_config.disabled()) {
    total_css.clear();
    css.clear();
    fprintf(fout, "\n========= Core cache stats =========\n");
    fprintf(fout, "L1I_cache:\n");
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; ++i) {
      m_cluster[i]->get_L1I_sub_stats(css);
      total_css += css;
    }
    fprintf(fout, "\tL1I_total_cache_accesses = %llu\n", total_css.accesses);
    fprintf(fout, "\tL1I_total_cache_misses = %llu\n", total_css.misses);
    if (total_css.accesses > 0) {
      fprintf(fout, "\tL1I_total_cache_miss_rate = %.4lf\n",
              (double)total_css.misses / (double)total_css.accesses);
    }
    fprintf(fout, "\tL1I_total_cache_pending_hits = %llu\n",
            total_css.pending_hits);
    fprintf(fout, "\tL1I_total_cache_reservation_fails = %llu\n",
            total_css.res_fails);
  }

  // L1D
  if (!m_shader_config->m_L1D_config.disabled()) {
    total_css.clear();
    css.clear();
    fprintf(fout, "L1D_cache:\n");
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
      m_cluster[i]->get_L1D_sub_stats(css);

      fprintf(stdout,
              "\tL1D_cache_core[%d]: Access = %llu, Miss = %llu, Miss_rate = "
              "%.3lf, Pending_hits = %llu, Reservation_fails = %llu\n",
              i, css.accesses, css.misses,
              (double)css.misses / (double)css.accesses, css.pending_hits,
              css.res_fails);

      total_css += css;
    }
    fprintf(fout, "\tL1D_total_cache_accesses = %llu\n", total_css.accesses);
    fprintf(fout, "\tL1D_total_cache_misses = %llu\n", total_css.misses);
    if (total_css.accesses > 0) {
      fprintf(fout, "\tL1D_total_cache_miss_rate = %.4lf\n",
              (double)total_css.misses / (double)total_css.accesses);
    }
    fprintf(fout, "\tL1D_total_cache_pending_hits = %llu\n",
            total_css.pending_hits);
    fprintf(fout, "\tL1D_total_cache_reservation_fails = %llu\n",
            total_css.res_fails);
    total_css.print_port_stats(fout, "\tL1D_cache");
  }

  // L1C
  if (!m_shader_config->m_L1C_config.disabled()) {
    total_css.clear();
    css.clear();
    fprintf(fout, "L1C_cache:\n");
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; ++i) {
      m_cluster[i]->get_L1C_sub_stats(css);
      total_css += css;
    }
    fprintf(fout, "\tL1C_total_cache_accesses = %llu\n", total_css.accesses);
    fprintf(fout, "\tL1C_total_cache_misses = %llu\n", total_css.misses);
    if (total_css.accesses > 0) {
      fprintf(fout, "\tL1C_total_cache_miss_rate = %.4lf\n",
              (double)total_css.misses / (double)total_css.accesses);
    }
    fprintf(fout, "\tL1C_total_cache_pending_hits = %llu\n",
            total_css.pending_hits);
    fprintf(fout, "\tL1C_total_cache_reservation_fails = %llu\n",
            total_css.res_fails);
  }

  // L1T
  if (!m_shader_config->m_L1T_config.disabled()) {
    total_css.clear();
    css.clear();
    fprintf(fout, "L1T_cache:\n");
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; ++i) {
      m_cluster[i]->get_L1T_sub_stats(css);
      total_css += css;
    }
    fprintf(fout, "\tL1T_total_cache_accesses = %llu\n", total_css.accesses);
    fprintf(fout, "\tL1T_total_cache_misses = %llu\n", total_css.misses);
    if (total_css.accesses > 0) {
      fprintf(fout, "\tL1T_total_cache_miss_rate = %.4lf\n",
              (double)total_css.misses / (double)total_css.accesses);
    }
    fprintf(fout, "\tL1T_total_cache_pending_hits = %llu\n",
            total_css.pending_hits);
    fprintf(fout, "\tL1T_total_cache_reservation_fails = %llu\n",
            total_css.res_fails);
  }
}

void gpgpu_sim::shader_print_l1_miss_stat(FILE *fout) const {
  unsigned total_d1_misses = 0, total_d1_accesses = 0;
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; ++i) {
    unsigned custer_d1_misses = 0, cluster_d1_accesses = 0;
    m_cluster[i]->print_cache_stats(fout, cluster_d1_accesses,
                                    custer_d1_misses);
    total_d1_misses += custer_d1_misses;
    total_d1_accesses += cluster_d1_accesses;
  }
  fprintf(fout, "total_dl1_misses=%d\n", total_d1_misses);
  fprintf(fout, "total_dl1_accesses=%d\n", total_d1_accesses);
  fprintf(fout, "total_dl1_miss_rate= %f\n",
          (float)total_d1_misses / (float)total_d1_accesses);
  /*
  fprintf(fout, "THD_INSN_AC: ");
  for (unsigned i=0; i<m_shader_config->n_thread_per_shader; i++)
     fprintf(fout, "%d ", m_sc[0]->get_thread_n_insn_ac(i));
  fprintf(fout, "\n");
  fprintf(fout, "T_L1_Mss: "); //l1 miss rate per thread
  for (unsigned i=0; i<m_shader_config->n_thread_per_shader; i++)
     fprintf(fout, "%d ", m_sc[0]->get_thread_n_l1_mis_ac(i));
  fprintf(fout, "\n");
  fprintf(fout, "T_L1_Mgs: "); //l1 merged miss rate per thread
  for (unsigned i=0; i<m_shader_config->n_thread_per_shader; i++)
     fprintf(fout, "%d ", m_sc[0]->get_thread_n_l1_mis_ac(i) -
  m_sc[0]->get_thread_n_l1_mrghit_ac(i)); fprintf(fout, "\n"); fprintf(fout,
  "T_L1_Acc: "); //l1 access per thread for (unsigned i=0;
  i<m_shader_config->n_thread_per_shader; i++) fprintf(fout, "%d ",
  m_sc[0]->get_thread_n_l1_access_ac(i)); fprintf(fout, "\n");

  //per warp
  int temp =0;
  fprintf(fout, "W_L1_Mss: "); //l1 miss rate per warp
  for (unsigned i=0; i<m_shader_config->n_thread_per_shader; i++) {
     temp += m_sc[0]->get_thread_n_l1_mis_ac(i);
     if (i%m_shader_config->warp_size ==
  (unsigned)(m_shader_config->warp_size-1)) { fprintf(fout, "%d ", temp); temp =
  0;
     }
  }
  fprintf(fout, "\n");
  temp=0;
  fprintf(fout, "W_L1_Mgs: "); //l1 merged miss rate per warp
  for (unsigned i=0; i<m_shader_config->n_thread_per_shader; i++) {
     temp += (m_sc[0]->get_thread_n_l1_mis_ac(i) -
  m_sc[0]->get_thread_n_l1_mrghit_ac(i) ); if (i%m_shader_config->warp_size ==
  (unsigned)(m_shader_config->warp_size-1)) { fprintf(fout, "%d ", temp); temp =
  0;
     }
  }
  fprintf(fout, "\n");
  temp =0;
  fprintf(fout, "W_L1_Acc: "); //l1 access per warp
  for (unsigned i=0; i<m_shader_config->n_thread_per_shader; i++) {
     temp += m_sc[0]->get_thread_n_l1_access_ac(i);
     if (i%m_shader_config->warp_size ==
  (unsigned)(m_shader_config->warp_size-1)) { fprintf(fout, "%d ", temp); temp =
  0;
     }
  }
  fprintf(fout, "\n");
  */
}

void warp_inst_t::print(FILE *fout) const {
  if (empty()) {
    fprintf(fout, "bubble\n");
    return;
  } else
    fprintf(fout, "0x%04llx ", pc);
  fprintf(fout, "w%02d[", m_warp_id);
  for (unsigned j = 0; j < m_config->warp_size; j++)
    fprintf(fout, "%c", (active(j) ? '1' : '0'));
  fprintf(fout, "]: ");
  m_config->gpgpu_ctx->func_sim->ptx_print_insn(pc, fout);
  fprintf(fout, "\n");
}
void shader_core_ctx::incexecstat(warp_inst_t *&inst) {
  // Latency numbers for next operations are used to scale the power values
  // for special operations, according observations from microbenchmarking
  // TODO: put these numbers in the xml configuration
  if (get_gpu()->get_config().g_power_simulation_enabled) {
    switch (inst->sp_op) {
      case INT__OP:
        incialu_stat(inst->active_count(), scaling_coeffs->int_coeff);
        break;
      case INT_MUL_OP:
        incimul_stat(inst->active_count(), scaling_coeffs->int_mul_coeff);
        break;
      case INT_MUL24_OP:
        incimul24_stat(inst->active_count(), scaling_coeffs->int_mul24_coeff);
        break;
      case INT_MUL32_OP:
        incimul32_stat(inst->active_count(), scaling_coeffs->int_mul32_coeff);
        break;
      case INT_DIV_OP:
        incidiv_stat(inst->active_count(), scaling_coeffs->int_div_coeff);
        break;
      case FP__OP:
        incfpalu_stat(inst->active_count(), scaling_coeffs->fp_coeff);
        break;
      case FP_MUL_OP:
        incfpmul_stat(inst->active_count(), scaling_coeffs->fp_mul_coeff);
        break;
      case FP_DIV_OP:
        incfpdiv_stat(inst->active_count(), scaling_coeffs->fp_div_coeff);
        break;
      case DP___OP:
        incdpalu_stat(inst->active_count(), scaling_coeffs->dp_coeff);
        break;
      case DP_MUL_OP:
        incdpmul_stat(inst->active_count(), scaling_coeffs->dp_mul_coeff);
        break;
      case DP_DIV_OP:
        incdpdiv_stat(inst->active_count(), scaling_coeffs->dp_div_coeff);
        break;
      case FP_SQRT_OP:
        incsqrt_stat(inst->active_count(), scaling_coeffs->sqrt_coeff);
        break;
      case FP_LG_OP:
        inclog_stat(inst->active_count(), scaling_coeffs->log_coeff);
        break;
      case FP_SIN_OP:
        incsin_stat(inst->active_count(), scaling_coeffs->sin_coeff);
        break;
      case FP_EXP_OP:
        incexp_stat(inst->active_count(), scaling_coeffs->exp_coeff);
        break;
      case TENSOR__OP:
        inctensor_stat(inst->active_count(), scaling_coeffs->tensor_coeff);
        break;
      case TEX__OP:
        inctex_stat(inst->active_count(), scaling_coeffs->tex_coeff);
        break;
      default:
        break;
    }
    if (inst->const_cache_operand)  // warp has const address space load as one
                                    // operand
      inc_const_accesses(1);
  }
}
void shader_core_ctx::print_stage(unsigned int stage, FILE *fout) const {
  m_pipeline_reg[stage].print(fout);
  // m_pipeline_reg[stage].print(fout);
}

void shader_core_ctx::display_simt_state(FILE *fout, int mask) const {
  if ((mask & 4) && m_config->model == POST_DOMINATOR) {
    fprintf(fout, "per warp SIMT control-flow state:\n");
    unsigned n = m_config->n_thread_per_shader / m_config->warp_size;
    for (unsigned i = 0; i < n; i++) {
      unsigned nactive = 0;
      for (unsigned j = 0; j < m_config->warp_size; j++) {
        unsigned tid = i * m_config->warp_size + j;
        int done = ptx_thread_done(tid);
        nactive += (ptx_thread_done(tid) ? 0 : 1);
        if (done && (mask & 8)) {
          unsigned done_cycle = m_thread[tid]->donecycle();
          if (done_cycle) {
            printf("\n w%02u:t%03u: done @ cycle %u", i, tid, done_cycle);
          }
        }
      }
      if (nactive == 0) {
        continue;
      }
      m_simt_stack[i]->print(fout);
    }
    fprintf(fout, "\n");
  }
}

void ldst_unit::print(FILE *fout) const {
  fprintf(fout, "LD/ST unit  = ");
  m_dispatch_reg->print(fout);
  if (m_mem_rc != NO_RC_FAIL) {
    fprintf(fout, "              LD/ST stall condition: ");
    switch (m_mem_rc) {
      case BK_CONF:
        fprintf(fout, "BK_CONF");
        break;
      case MSHR_RC_FAIL:
        fprintf(fout, "MSHR_RC_FAIL");
        break;
      case ICNT_RC_FAIL:
        fprintf(fout, "ICNT_RC_FAIL");
        break;
      case COAL_STALL:
        fprintf(fout, "COAL_STALL");
        break;
      case WB_ICNT_RC_FAIL:
        fprintf(fout, "WB_ICNT_RC_FAIL");
        break;
      case WB_CACHE_RSRV_FAIL:
        fprintf(fout, "WB_CACHE_RSRV_FAIL");
        break;
      case N_MEM_STAGE_STALL_TYPE:
        fprintf(fout, "N_MEM_STAGE_STALL_TYPE");
        break;
      default:
        abort();
    }
    fprintf(fout, "\n");
  }
  fprintf(fout, "LD/ST wb    = ");
  m_next_wb.print(fout);
  fprintf(
      fout,
      "Last LD/ST writeback @ %llu + %llu (gpu_sim_cycle+gpu_tot_sim_cycle)\n",
      m_last_inst_gpu_sim_cycle, m_last_inst_gpu_tot_sim_cycle);
  fprintf(fout, "Pending register writes:\n");
  std::map<unsigned /*warp_id*/,
           std::map<unsigned /*regnum*/, unsigned /*count*/> >::const_iterator
      w;
  for (w = m_pending_writes.begin(); w != m_pending_writes.end(); w++) {
    unsigned warp_id = w->first;
    const std::map<unsigned /*regnum*/, unsigned /*count*/> &warp_info =
        w->second;
    if (warp_info.empty()) continue;
    fprintf(fout, "  w%2u : ", warp_id);
    std::map<unsigned /*regnum*/, unsigned /*count*/>::const_iterator r;
    for (r = warp_info.begin(); r != warp_info.end(); ++r) {
      fprintf(fout, "  %u(%u)", r->first, r->second);
    }
    fprintf(fout, "\n");
  }
  m_L1C->display_state(fout);
  m_L1T->display_state(fout);
  if (!m_config->m_L1D_config.disabled()) m_L1D->display_state(fout);
  fprintf(fout, "LD/ST response FIFO (occupancy = %zu):\n",
          m_response_fifo.size());
  for (std::list<mem_fetch *>::const_iterator i = m_response_fifo.begin();
       i != m_response_fifo.end(); i++) {
    const mem_fetch *mf = *i;
    mf->print(fout);
  }
}

void shader_core_ctx::display_pipeline(FILE *fout, int print_mem,
                                       int mask) const {
  fprintf(fout, "=================================================\n");
  fprintf(fout, "shader %u at cycle %Lu+%Lu (%u threads running)\n", m_sid,
          m_gpu->gpu_tot_sim_cycle, m_gpu->gpu_sim_cycle, m_not_completed);
  fprintf(fout, "=================================================\n");

  dump_warp_state(fout);
  fprintf(fout, "\n");

  m_L1I->display_state(fout);

  fprintf(fout, "IF/ID       = ");
  if (!m_inst_fetch_buffer.m_valid)
    fprintf(fout, "bubble\n");
  else {
    fprintf(fout, "w%2u : pc = 0x%llx, nbytes = %u\n",
            m_inst_fetch_buffer.m_warp_id, m_inst_fetch_buffer.m_pc,
            m_inst_fetch_buffer.m_nbytes);
  }
  fprintf(fout, "\nibuffer status:\n");
  for (unsigned i = 0; i < m_config->max_warps_per_shader; i++) {
    if (!m_warp[i]->ibuffer_empty()) m_warp[i]->print_ibuffer(fout);
  }
  fprintf(fout, "\n");
  display_simt_state(fout, mask);
  fprintf(fout, "-------------------------- Scoreboard\n");
  m_scoreboard->printContents();
  /*
     fprintf(fout,"ID/OC (SP)  = ");
     print_stage(ID_OC_SP, fout);
     fprintf(fout,"ID/OC (SFU) = ");
     print_stage(ID_OC_SFU, fout);
     fprintf(fout,"ID/OC (MEM) = ");
     print_stage(ID_OC_MEM, fout);
  */
  fprintf(fout, "-------------------------- OP COL\n");
  m_operand_collector.dump(fout);
  /* fprintf(fout, "OC/EX (SP)  = ");
     print_stage(OC_EX_SP, fout);
     fprintf(fout, "OC/EX (SFU) = ");
     print_stage(OC_EX_SFU, fout);
     fprintf(fout, "OC/EX (MEM) = ");
     print_stage(OC_EX_MEM, fout);
  */
  fprintf(fout, "-------------------------- Pipe Regs\n");

  for (unsigned i = 0; i < N_PIPELINE_STAGES; i++) {
    fprintf(fout, "--- %s ---\n", pipeline_stage_name_decode[i]);
    print_stage(i, fout);
    fprintf(fout, "\n");
  }

  fprintf(fout, "-------------------------- Fu\n");
  for (unsigned n = 0; n < m_num_function_units; n++) {
    m_fu[n]->print(fout);
    fprintf(fout, "---------------\n");
  }
  fprintf(fout, "-------------------------- other:\n");

  for (unsigned i = 0; i < num_result_bus; i++) {
    std::string bits = m_result_bus[i]->to_string();
    fprintf(fout, "EX/WB sched[%d]= %s\n", i, bits.c_str());
  }
  fprintf(fout, "EX/WB      = ");
  print_stage(EX_WB, fout);
  fprintf(fout, "\n");
  fprintf(
      fout,
      "Last EX/WB writeback @ %llu + %llu (gpu_sim_cycle+gpu_tot_sim_cycle)\n",
      m_last_inst_gpu_sim_cycle, m_last_inst_gpu_tot_sim_cycle);

  if (m_active_threads.count() <= 2 * m_config->warp_size) {
    fprintf(fout, "Active Threads : ");
    unsigned last_warp_id = -1;
    for (unsigned tid = 0; tid < m_active_threads.size(); tid++) {
      unsigned warp_id = tid / m_config->warp_size;
      if (m_active_threads.test(tid)) {
        if (warp_id != last_warp_id) {
          fprintf(fout, "\n  warp %u : ", warp_id);
          last_warp_id = warp_id;
        }
        fprintf(fout, "%u ", tid);
      }
    }
  }
}

/*
�߳̿��SIMT Core�ĵ��ȷ�����shader_core_ctx::issue_block2core(...)��һ��Core�Ͽ�ͬʱ���ȵ������
�̿飨���ΪCTA���������ɺ���shader_core_config::max_cta(...)���㡣����������ݳ���ָ����ÿ���߳̿�
��������ÿ���̼߳Ĵ�����ʹ������������ڴ��ʹ������Լ����õ�ÿ��Core����߳̿����������ƣ�ȷ�����Բ�
�����������SIMT Core������߳̿�����������˵���������ÿ����׼�����������أ���ô���Է����SIMT Core
���߳̿��������������������е���Сֵ���ǿ��Է����SIMT Core������߳̿�����

�ú����Ĳ�����kernel_info_tΪ�ں˺�������Ϣ�ࡣkernel_info_t�������GPU����Ϳ�ά�ȡ����ں���ڵ��
���� function_info �����Լ�Ϊ�ں˲���������ڴ档
*/
unsigned int shader_core_config::max_cta(const kernel_info_t &k) const {
  //k.threads_per_cta()����ÿ���߳̿��е��߳�������threads_per_cta=m_block_dim.x * m_block_dim.y 
  //* m_block_dim.z��������������CUDA�����е�һ���߳̿��ϵ��߳�������
  unsigned threads_per_cta = k.threads_per_cta();
  //k.entry()����kernel����ڵ㣬����ڵ������������
  const class function_info *kernel = k.entry();
  //����һ��CTA/�߳̿��е��̱߳���Ҫ�� warp_size��һ��warp�е��߳����������������������Ҫ����CTA�е�
  //�̣߳���CTA�е��߳������ﵽ warp_size ����������
  unsigned int padded_cta_size = threads_per_cta;
  if (padded_cta_size % warp_size)
    padded_cta_size = ((padded_cta_size / warp_size) + 1) * (warp_size);

  // Limit by n_threads/shader
  //n_thread_per_shader�� -gpgpu_shader_core_pipeline ѡ���еĵ�һ��ֵ���ڶ���ֵ�� warp_size������
  //��V100�е� -gpgpu_shader_core_pipeline 2048:32����һ��Shader Core��SM����ʵ�ֵ������ 64 warps/
  //SM����һ�� Shader Core��SM����ʵ�ֵ����ɲ����߳������� 2048��
  //result_thread��Ϊ������[n_threads/shader]������ɵĵ���SM�����ɲ���CTA������
  unsigned int result_thread = n_thread_per_shader / padded_cta_size;
  //����kernel��kernel_info��
  const struct gpgpu_ptx_sim_info *kernel_info = ptx_sim_kernel_info(kernel);

  // Limit by shmem/shader
  //gpgpu_shmem_sizeΪÿ��SIMT Core��Ҳ��ΪShader Core��SM���Ĺ���洢��С�����ڵ���SM��ÿ����һ��CTA��
  //��Ҫ��ÿ��CTA���ж�������ͬ��С��shared memory����ˣ�result_shmem����shared memory������ɵĵ���
  //SM�����ɲ���CTA������
  unsigned int result_shmem = (unsigned)-1;
  if (kernel_info->smem > 0)
    result_shmem = gpgpu_shmem_size / kernel_info->smem;

  // Limit by register count, rounded up to multiple of 4.
  //gpgpu_shader_registers��ÿ��Shader Core�ļĴ����������ڵ���SM��ÿ����һ��CTA����Ҫ��ÿ��CTA���ж�
  //������ͬ�����ļĴ�����result_regs���ɼĴ���������ɵĵ���SM�����ɲ���CTA������
  unsigned int result_regs = (unsigned)-1;
  if (kernel_info->regs > 0)
    result_regs = gpgpu_shader_registers /
                  (padded_cta_size * ((kernel_info->regs + 3) & ~3));

  // Limit by CTA
  //max_cta_per_core����Ӳ�����õ�SM�ڵ����ɲ���CTA������
  unsigned int result_cta = max_cta_per_core;

  //ѡ�������������£���С��SM�ڵ�CTA����������
  unsigned result = result_thread;
  result = gs_min2(result, result_shmem);
  result = gs_min2(result, result_regs);
  result = gs_min2(result, result_cta);

  //��last_kinfo��ֵΪkernel_info��
  static const struct gpgpu_ptx_sim_info *last_kinfo = NULL;
  if (last_kinfo !=
      kernel_info) {  // Only print out stats if kernel_info struct changes
    last_kinfo = kernel_info;
    printf("GPGPU-Sim uArch: CTA/core = %u, limited by:", result);
    if (result == result_thread) printf(" threads");
    if (result == result_shmem) printf(" shmem");
    if (result == result_regs) printf(" regs");
    if (result == result_cta) printf(" cta_limit");
    printf("\n");
  }

  // gpu_max_cta_per_shader is limited by number of CTAs if not enough to keep
  // all cores busy
  //gpu_max_cta_per_shader��cta���������ƣ���������Ա������к�æµ��
  //k.num_blocks()����CUDA�����е�Grid�е������߳̿��������num_shader()����Ӳ�����е�SM���ֳ�Shader 
  //Core����������������������result*SM���� > Grid�е������߳̿����������result������CTA������ʹ��
  //����SM����æµ״̬�����磬��10��SM���������������޵����CTA������result=5������CUDA��������40��
  //�߳̿飬���� 40 < 5 * 10��Ϊ��������SM��æµ�������Գ���������е�Ӳ����Դ�����Խ���������������
  //�����CTA�������ٽ���һЩ����ƽ�����䵽ÿ��SM�ϡ�
  if (k.num_blocks() < result * num_shader()) {
    result = k.num_blocks() / num_shader();
    if (k.num_blocks() % num_shader()) result++;
  }

  assert(result <= MAX_CTA_PER_SHADER);
  if (result < 1) {
    printf(
        "GPGPU-Sim uArch: ERROR ** Kernel requires more resources than shader "
        "has.\n");
    if (gpgpu_ignore_resources_limitation) {
      printf(
          "GPGPU-Sim uArch: gpgpu_ignore_resources_limitation is set, ignore "
          "the ERROR!\n");
      return 1;
    }
    abort();
  }
  //��V100 GPU�У���֧�� adaptive_cache_config����Ӧ�Ե�cache���ã���ADAPTIVE_VOLTA=1��
  //��Ӧ�Ե�cache���ô�����V100�У���ʣ��Ĳ�ʹ�õ�shared memory����L1 cacheʹ�á�
  //��ʼ״̬ʱ������L1 cache��Ӧ�����õı�־ΪFalse��
  if (adaptive_cache_config && !k.cache_config_set) {
    // For more info about adaptive cache, see
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-7-x
    //���㵱ǰ����SM�е�����CTA�Ĺ���洢��������
    unsigned total_shmem = kernel_info->smem * result;
    assert(total_shmem >= 0 && total_shmem <= shmem_opt_list.back());

    // Unified cache config is in KB. Converting to B
    unsigned total_unified = m_L1D_config.m_unified_cache_size * 1024;

    bool l1d_configured = false;
    unsigned max_assoc = m_L1D_config.get_max_assoc();

    for (std::vector<unsigned>::const_iterator it = shmem_opt_list.begin();
         it < shmem_opt_list.end(); it++) {
      if (total_shmem <= *it) {
        float l1_ratio = 1 - ((float)*(it) / total_unified);
        // make sure the ratio is between 0 and 1
        assert(0 <= l1_ratio && l1_ratio <= 1);
        // round to nearest instead of round down
        m_L1D_config.set_assoc(max_assoc * l1_ratio + 0.5f);
        l1d_configured = true;
        break;
      }
    }

    assert(l1d_configured && "no shared memory option found");

    if (m_L1D_config.is_streaming()) {
      // for streaming cache, if the whole memory is allocated
      // to the L1 cache, then make the allocation to be on_MISS
      // otherwise, make it ON_FILL to eliminate line allocation fails
      // i.e. MSHR throughput is the same, independent on the L1 cache
      // size/associativity
      if (total_shmem == 0) {
        m_L1D_config.set_allocation_policy(ON_MISS);
        printf("GPGPU-Sim: Reconfigure L1 allocation to ON_MISS\n");
      } else {
        m_L1D_config.set_allocation_policy(ON_FILL);
        printf("GPGPU-Sim: Reconfigure L1 allocation to ON_FILL\n");
      }
    }
    printf("GPGPU-Sim: Reconfigure L1 cache to %uKB\n",
           m_L1D_config.get_total_size_inKB());

    k.cache_config_set = true;
  }

  return result;
}

void shader_core_config::set_pipeline_latency() {
  // calculate the max latency  based on the input

  unsigned int_latency[6];
  unsigned fp_latency[5];
  unsigned dp_latency[5];
  unsigned sfu_latency;
  unsigned tensor_latency;

  /*
   * [0] ADD,SUB
   * [1] MAX,Min
   * [2] MUL
   * [3] MAD
   * [4] DIV
   * [5] SHFL
   */
  sscanf(gpgpu_ctx->func_sim->opcode_latency_int, "%u,%u,%u,%u,%u,%u",
         &int_latency[0], &int_latency[1], &int_latency[2], &int_latency[3],
         &int_latency[4], &int_latency[5]);
  sscanf(gpgpu_ctx->func_sim->opcode_latency_fp, "%u,%u,%u,%u,%u",
         &fp_latency[0], &fp_latency[1], &fp_latency[2], &fp_latency[3],
         &fp_latency[4]);
  sscanf(gpgpu_ctx->func_sim->opcode_latency_dp, "%u,%u,%u,%u,%u",
         &dp_latency[0], &dp_latency[1], &dp_latency[2], &dp_latency[3],
         &dp_latency[4]);
  sscanf(gpgpu_ctx->func_sim->opcode_latency_sfu, "%u", &sfu_latency);
  sscanf(gpgpu_ctx->func_sim->opcode_latency_tensor, "%u", &tensor_latency);

  // all div operation are executed on sfu
  // assume that the max latency are dp div or normal sfu_latency
  max_sfu_latency = std::max(dp_latency[4], sfu_latency);
  // assume that the max operation has the max latency
  max_sp_latency = fp_latency[1];
  max_int_latency = std::max(int_latency[1], int_latency[5]);
  max_dp_latency = dp_latency[1];
  max_tensor_core_latency = tensor_latency;
}

/*
Shader Core/SIMT Core��ǰ�ƽ�һ��ʱ�����ڡ�
*/
void shader_core_ctx::cycle() {
  //������SIMT Core���ڷǻ�Ծ״̬�����Ѿ�ִ����ɣ�ʱ�����ڲ���ǰ�ƽ���
  if (!isactive() && get_not_completed() == 0) return;

  m_stats->shader_cycles[m_sid]++;
  // printf("m_stats->shader_cycles[m_sid]: %u\n", m_stats->shader_cycles[m_sid]);
  //ÿ���ں�ʱ�����ڣ�shader_core_ctx::cycle()�������ã���ģ��SIMT Core��һ�����ڡ������������һ
  //���Ա���������෴��˳��ģ���ں˵���ˮ�߽׶Σ���ģ����ˮ��ЧӦ��
  //     fetch()             ������һִ��
  //     decode()            �����ڶ�ִ��
  //     issue()             ��������ִ��
  //     read_operand()      ��������ִ��
  //     execute()           ��������ִ��
  //     writeback()         ��������ִ��
  writeback();
  execute();
  read_operands();
  issue();
  for (unsigned int i = 0; i < m_config->inst_fetch_throughput; ++i) {
    decode();
    fetch();
  }
}

// Flushes all content of the cache to memory

void shader_core_ctx::cache_flush() { m_ldst_unit->flush(); }

void shader_core_ctx::cache_invalidate() { m_ldst_unit->invalidate(); }

// modifiers
/*
�������ռ������ٲ��������ڷ����������
*/
std::list<opndcoll_rfu_t::op_t> opndcoll_rfu_t::arbiter_t::allocate_reads() {
  //create a list of results
  //����һ������б�(a)�ڲ�ͬ�ļĴ����ļ�Bank��(b)��������ͬ�Ĳ������ռ�����
  std::list<op_t>
      result;  // a list of registers that (a) are in different register banks,
               // (b) do not go to the same operand collector

  // input is the register bank
  // output is the collector units
  int input;
  int output;
  //_inputs�ǼĴ����ļ���Bank����
  int _inputs = m_num_banks;
  //_outputs�ǲ������ռ�����������
  int _outputs = m_num_collectors;
  //��Ӧ��MAP�Խ�˳����ͼ�����ά�ȣ��򳤶Ȼ��ȡ�
  int _square = (_inputs > _outputs) ? _inputs : _outputs;
  assert(_square > 0);
  //_pri����һ��ִ��arbiter_t::allocate_reads()����ʱ�����һ����������ռ�����Ԫ����һ����
  //������Ԫ��ID��
  int _pri = (int)m_last_cu;

  // Clear matching: set all the entries to -1
  //_inmatch[i]�ǵ�i��Bank�Ŀ�ƥ��������ռ���ID�������ֵ��Ϊ-1��0����˵�����Ѿ�ƥ�䵽��
  for (int i = 0; i < _inputs; ++i) _inmatch[i] = -1;
  //_outmatch[j]�ǵ�j���������ռ����Ŀ�ƥ��Bank��ID�������ֵ��Ϊ-1��0����˵�����Ѿ�ƥ�䵽��
  for (int j = 0; j < _outputs; ++j) _outmatch[j] = -1;

  //�����еļĴ����ļ�Bank����ѭ����
  for (unsigned i = 0; i < m_num_banks; i++) {
    //_request[m_num_banks][m_num_collectors]��ָĳ���ռ�����Ԫ��ĳ���Ĵ���Bank�Ƿ�������
    //������1��������0�������һ��forѭ�������е��ռ�����Ԫ�����еļĴ���Bank����ѭ����������
    //�е���������Ϊ0��
    for (unsigned j = 0; j < m_num_collectors; j++) {
      assert(i < (unsigned)_inputs);
      assert(j < (unsigned)_outputs);
      //���õ�j���ռ�����Ԫ�Ե�i���Ĵ����ļ�Bank��������0��
      _request[i][j] = 0;
    }
    //m_queue��һ����bank���������Ĳ��������У�m_queue[i]�ǵ�i��bank��ȡ�Ĳ��������С������
    //�����в�Ϊ�գ�˵�����bank�в����������Ѿ�����m_queue��
    if (!m_queue[i].empty()) {
      const op_t &op = m_queue[i].front();
      //op.get_oc_id()���ص�ǰ�������������ռ�����Ԫ��ID��
      int oc_id = op.get_oc_id();
      assert(i < (unsigned)_inputs);
      assert(oc_id < _outputs);
      //��oc_id���ռ�����Ԫ�Ե�i���Ĵ����ļ�Bank������������1��
      _request[i][oc_id] = 1;
    }
    //m_allocated_bank[i]���ڴ洢��i��Bank��״̬������NO_ALLOC, READ_ALLOC, WRITE_ALLOC��
    //�����i��Bank��WRITE_ALLOC��˵�����Bank�Ѿ��������ĳ���ռ�����Ԫ������ռ�����Ԫ����
    //ִ��д��������ˣ����Bank���ܱ�����������ռ�����Ԫ��
    if (m_allocated_bank[i].is_write()) {
      assert(i < (unsigned)_inputs);
      //����i��Bank��WRITE_ALLOCʱ����i��Bank�������е��ռ�����Ԫ��˵�����ɷ���Ϊ������������
      //��������_inmatch�еĵ�i��Bank�Ŀ�ƥ��״̬Ϊ0��д�������и��ߵ����ȼ���
      _inmatch[i] = 0;  // write gets priority
    }
  }

  ///// wavefront allocator from booksim... --->

  // Loop through diagonals of request matrix
  // printf("####\n");

  //�����в������ռ���ѭ�����������˳���ǰ��նԽ��߼��ģ����磬�����5���ռ�����Ԫ��3��Bank��
  //��_square=5��_outputs=5��_inputs=3���������_pri=2�Ļ������������forѭ���ᰴ�������˳��
  //���м�飺
  //    order    input    output
  //      1        0         2
  //      2        1         3
  //      3        2         4
  //      4        0         3
  //      5        1         4
  //      6        2         0
  //      7        0         4
  //      8        1         0
  //      9        2         1
  //      10       0         0
  //      11       1         1
  //      12       2         2
  //      13       0         1
  //      14       1         2
  //      15       2         3
  //��Ӧ��MAP�Խ�˳����ͼΪ��
  //                   _output
  //             0    1    2    3    4
  //           |----|----|----|----|----|
  //         0 | 10 | 13 |  1 |  4 |  7 |
  //           |----|----|----|----|----|
  // _input  1 |  8 | 11 | 14 |  2 |  5 |
  //           |----|----|----|----|----|
  //         2 |  6 |  9 | 12 | 15 |  3 |
  //           |----|----|----|----|----|
  for (int p = 0; p < _square; ++p) {
    //_pri����һ��ִ��arbiter_t::allocate_reads()����ʱ�����һ����������ռ�����Ԫ����һ����
    //������Ԫ��ID�������ǵ�ǰִ��arbiter_t::allocate_reads()����ʱ������һ�����һ���������
    //�ռ�����Ԫ����һ���ռ�����Ԫ��ID��ʼ������RRѭ����
    output = (_pri + p) % _outputs;

    // Step through the current diagonal
    for (input = 0; input < _inputs; ++input) {
      assert(input < _inputs);
      assert(output < _outputs);
      //�����input��Bankû�б������ĳ���ռ�����Ԫ���ҵ�output���ռ�����Ԫ�Ե�input��Bank����
      //����ô�ͷ����input��Bank����output���ռ�����Ԫ������_inmatch�еĵ�input��Bank�Ŀ�ƥ
      //���ռ�����ԪΪoutput������_outmatch�еĵ�output���ռ�����Ԫ�Ŀ�ƥ��BankΪinput��
      if ((output < _outputs) && (_inmatch[input] == -1) &&
          //( _outmatch[output] == -1 ) &&   //allow OC to read multiple reg
          // banks at the same cycle
          (_request[input][output] /*.label != -1*/)) {
        // Grant!
        _inmatch[input] = output;
        _outmatch[output] = input;
        // printf("Register File: granting bank %d to OC %d, schedid %d, warpid
        // %d, Regid %d\n", input, output, (m_queue[input].front()).get_sid(),
        // (m_queue[input].front()).get_wid(),
        // (m_queue[input].front()).get_reg());
      }
      //�������ռ����ı�ŵ�����
      output = (output + 1) % _outputs;
    }
  }

  // Round-robin the priority diagonal
  //_pri����һ��ִ��arbiter_t::allocate_reads()����ʱ�����һ����������ռ�����Ԫ����һ���ռ�
  //����Ԫ��ID��
  _pri = (_pri + 1) % _outputs;

  /// <--- end code from booksim
  //m_last_cu����һ��ִ��arbiter_t::allocate_reads()����ʱ�����һ����������ռ�����Ԫ��ID��
  m_last_cu = _pri;
  //�����еļĴ����ļ�Bank����ѭ����
  for (unsigned i = 0; i < m_num_banks; i++) {
    //������ڷ������i��Bank�Ĳ������ռ�����
    if (_inmatch[i] != -1) {
      //�жϵ�i��Bank�Ƿ��Ѿ��������ĳ���ռ�����Ԫ����д�������������д�����Ļ�����Ϊ����������
      //�򽫶�Ӧ����i��Bank���������m_queue�е��׸��������������result��
      if (!m_allocated_bank[i].is_write()) {
        unsigned bank = (unsigned)i;
        op_t &op = m_queue[bank].front();
        result.push_back(op);
        m_queue[bank].pop_front();
      }
    }
  }

  return result;

}

barrier_set_t::barrier_set_t(shader_core_ctx *shader,
                             unsigned max_warps_per_core,
                             unsigned max_cta_per_core,
                             unsigned max_barriers_per_cta,
                             unsigned warp_size) {
  m_max_warps_per_core = max_warps_per_core;
  m_max_cta_per_core = max_cta_per_core;
  m_max_barriers_per_cta = max_barriers_per_cta;
  m_warp_size = warp_size;
  m_shader = shader;
  if (max_warps_per_core > WARP_PER_CTA_MAX) {
    printf(
        "ERROR ** increase WARP_PER_CTA_MAX in shader.h from %u to >= %u or "
        "warps per cta in gpgpusim.config\n",
        WARP_PER_CTA_MAX, max_warps_per_core);
    exit(1);
  }
  if (max_barriers_per_cta > MAX_BARRIERS_PER_CTA) {
    printf(
        "ERROR ** increase MAX_BARRIERS_PER_CTA in abstract_hardware_model.h "
        "from %u to >= %u or barriers per cta in gpgpusim.config\n",
        MAX_BARRIERS_PER_CTA, max_barriers_per_cta);
    exit(1);
  }
  m_warp_active.reset();
  m_warp_at_barrier.reset();
  for (unsigned i = 0; i < max_barriers_per_cta; i++) {
    m_bar_id_to_warps[i].reset();
  }
}

/*
During cta allocation.
ΪCTA����������Դ����֪����ĳ��CTA�а�������Щwarps���Ϳ��Է�������cta_id��ʶ��CTA���ϵ���Դ������Ĳ�����
    unsigned cta_id��CTA��ţ�
    warp_set_t warps������CTA�ڵ�����warp������С��λͼ��
*/
void barrier_set_t::allocate_barrier(unsigned cta_id, warp_set_t warps) {
  assert(cta_id < m_max_cta_per_core);
  //m_cta_to_warps��Map<CTA ID������CTA�ڵ�����warp������С��λͼ>��
  cta_to_warp_t::iterator w = m_cta_to_warps.find(cta_id);
  //�ڱ���������֮ǰ��cta��Ӧ���Ѿ��ǻ�Ծ�Ļ��Ѿ�������������Դ��
  assert(w == m_cta_to_warps.end());  // cta should not already be active or
                                      // allocated barrier resources
  //��ֵ����m_cta_to_warps��Ϊcta_id��ʶ��CTA��ֵλͼ��
  m_cta_to_warps[cta_id] = warps;
  assert(m_cta_to_warps.size() <=
         m_max_cta_per_core);  // catch cta's that were not properly deallocated

  m_warp_active |= warps;
  m_warp_at_barrier &= ~warps;
  for (unsigned i = 0; i < m_max_barriers_per_cta; i++) {
    m_bar_id_to_warps[i] &= ~warps;
  }
}

// during cta deallocation
void barrier_set_t::deallocate_barrier(unsigned cta_id) {
  cta_to_warp_t::iterator w = m_cta_to_warps.find(cta_id);
  if (w == m_cta_to_warps.end()) return;
  warp_set_t warps = w->second;
  warp_set_t at_barrier = warps & m_warp_at_barrier;
  assert(at_barrier.any() == false);  // no warps stuck at barrier
  warp_set_t active = warps & m_warp_active;
  assert(active.any() == false);  // no warps in CTA still running
  m_warp_active &= ~warps;
  m_warp_at_barrier &= ~warps;

  for (unsigned i = 0; i < m_max_barriers_per_cta; i++) {
    warp_set_t at_a_specific_barrier = warps & m_bar_id_to_warps[i];
    assert(at_a_specific_barrier.any() == false);  // no warps stuck at barrier
    m_bar_id_to_warps[i] &= ~warps;
  }
  m_cta_to_warps.erase(w);
}

// individual warp hits barrier
void barrier_set_t::warp_reaches_barrier(unsigned cta_id, unsigned warp_id,
                                         warp_inst_t *inst) {
  barrier_type bar_type = inst->bar_type;
  unsigned bar_id = inst->bar_id;
  unsigned bar_count = inst->bar_count;
  assert(bar_id != (unsigned)-1);
  cta_to_warp_t::iterator w = m_cta_to_warps.find(cta_id);

  if (w == m_cta_to_warps.end()) {  // cta is active
    printf(
        "ERROR ** cta_id %u not found in barrier set on cycle %llu+%llu...\n",
        cta_id, m_shader->get_gpu()->gpu_tot_sim_cycle,
        m_shader->get_gpu()->gpu_sim_cycle);
    dump();
    abort();
  }
  assert(w->second.test(warp_id) == true);  // warp is in cta

  m_bar_id_to_warps[bar_id].set(warp_id);
  if (bar_type == SYNC || bar_type == RED) {
    m_warp_at_barrier.set(warp_id);
  }
  warp_set_t warps_in_cta = w->second;
  warp_set_t at_barrier = warps_in_cta & m_bar_id_to_warps[bar_id];
  warp_set_t active = warps_in_cta & m_warp_active;
  if (bar_count == (unsigned)-1) {
    if (at_barrier == active) {
      // all warps have reached barrier, so release waiting warps...
      m_bar_id_to_warps[bar_id] &= ~at_barrier;
      m_warp_at_barrier &= ~at_barrier;
      if (bar_type == RED) {
        m_shader->broadcast_barrier_reduction(cta_id, bar_id, at_barrier);
      }
    }
  } else {
    // TODO: check on the hardware if the count should include warp that exited
    if ((at_barrier.count() * m_warp_size) == bar_count) {
      // required number of warps have reached barrier, so release waiting
      // warps...
      m_bar_id_to_warps[bar_id] &= ~at_barrier;
      m_warp_at_barrier &= ~at_barrier;
      if (bar_type == RED) {
        m_shader->broadcast_barrier_reduction(cta_id, bar_id, at_barrier);
      }
    }
  }
}

// warp reaches exit
void barrier_set_t::warp_exit(unsigned warp_id) {
  // caller needs to verify all threads in warp are done, e.g., by checking PDOM
  // stack to see it has only one entry during exit_impl()
  m_warp_active.reset(warp_id);

  // test for barrier release
  cta_to_warp_t::iterator w = m_cta_to_warps.begin();
  for (; w != m_cta_to_warps.end(); ++w) {
    if (w->second.test(warp_id) == true) break;
  }
  warp_set_t warps_in_cta = w->second;
  warp_set_t active = warps_in_cta & m_warp_active;

  for (unsigned i = 0; i < m_max_barriers_per_cta; i++) {
    warp_set_t at_a_specific_barrier = warps_in_cta & m_bar_id_to_warps[i];
    if (at_a_specific_barrier == active) {
      // all warps have reached barrier, so release waiting warps...
      m_bar_id_to_warps[i] &= ~at_a_specific_barrier;
      m_warp_at_barrier &= ~at_a_specific_barrier;
    }
  }
}

// assertions
bool barrier_set_t::warp_waiting_at_barrier(unsigned warp_id) const {
  return m_warp_at_barrier.test(warp_id);
}

void barrier_set_t::dump() {
  printf("barrier set information\n");
  printf("  m_max_cta_per_core = %u\n", m_max_cta_per_core);
  printf("  m_max_warps_per_core = %u\n", m_max_warps_per_core);
  printf(" m_max_barriers_per_cta =%u\n", m_max_barriers_per_cta);
  printf("  cta_to_warps:\n");

  cta_to_warp_t::const_iterator i;
  for (i = m_cta_to_warps.begin(); i != m_cta_to_warps.end(); i++) {
    unsigned cta_id = i->first;
    warp_set_t warps = i->second;
    printf("    cta_id %u : %s\n", cta_id, warps.to_string().c_str());
  }
  printf("  warp_active: %s\n", m_warp_active.to_string().c_str());
  printf("  warp_at_barrier: %s\n", m_warp_at_barrier.to_string().c_str());
  for (unsigned i = 0; i < m_max_barriers_per_cta; i++) {
    warp_set_t warps_reached_barrier = m_bar_id_to_warps[i];
    printf("  warp_at_barrier %u: %s\n", i,
           warps_reached_barrier.to_string().c_str());
  }
  fflush(stdout);
}

void shader_core_ctx::warp_exit(unsigned warp_id) {
  bool done = true;
  for (unsigned i = warp_id * get_config()->warp_size;
       i < (warp_id + 1) * get_config()->warp_size; i++) {
    //		if(this->m_thread[i]->m_functional_model_thread_state &&
    // this->m_thread[i].m_functional_model_thread_state->donecycle()==0) {
    // done = false;
    //		}

    if (m_thread[i] && !m_thread[i]->is_done()) done = false;
  }
  // if (m_warp[warp_id].get_n_completed() == get_config()->warp_size)
  // if (this->m_simt_stack[warp_id]->get_num_entries() == 0)
  if (done) m_barriers.warp_exit(warp_id);
}

bool shader_core_ctx::check_if_non_released_reduction_barrier(
    warp_inst_t &inst) {
  unsigned warp_id = inst.warp_id();
  bool bar_red_op = (inst.op == BARRIER_OP) && (inst.bar_type == RED);
  bool non_released_barrier_reduction = false;
  bool warp_stucked_at_barrier = warp_waiting_at_barrier(warp_id);
  bool single_inst_in_pipeline =
      (m_warp[warp_id]->num_issued_inst_in_pipeline() == 1);
  non_released_barrier_reduction =
      single_inst_in_pipeline and warp_stucked_at_barrier and bar_red_op;
  printf("non_released_barrier_reduction=%u\n", non_released_barrier_reduction);
  return non_released_barrier_reduction;
}

bool shader_core_ctx::warp_waiting_at_barrier(unsigned warp_id) const {
  return m_barriers.warp_waiting_at_barrier(warp_id);
}

bool shader_core_ctx::warp_waiting_at_mem_barrier(unsigned warp_id) {
  if (!m_warp[warp_id]->get_membar()) return false;
  if (!m_scoreboard->pendingWrites(warp_id)) {
    m_warp[warp_id]->clear_membar();
    if (m_gpu->get_config().flush_l1()) {
      // Mahmoud fixed this on Nov 2019
      // Invalidate L1 cache
      // Based on Nvidia Doc, at MEM barrier, we have to
      //(1) wait for all pending writes till they are acked
      //(2) invalidate L1 cache to ensure coherence and avoid reading stall data
      cache_invalidate();
      // TO DO: you need to stall the SM for 5k cycles.
    }
    return false;
  }
  return true;
}

/*
��������ÿSM��CTA����kernel_max_cta_per_shader�����һ�Ҫ�����߳̿���߳������Ƿ��ܶ�warp 
sizeȡģ���㣬������paddedÿCTA���߳�����kernel_padded_threads_per_cta��
*/
void shader_core_ctx::set_max_cta(const kernel_info_t &kernel) {
  // calculate the max cta count and cta size for local memory address mapping
  //��������ÿSM��CTA�������Լ�paddedÿCTA���߳�������
  kernel_max_cta_per_shader = m_config->max_cta(kernel);
  unsigned int gpu_cta_size = kernel.threads_per_cta();
  kernel_padded_threads_per_cta =
      (gpu_cta_size % m_config->warp_size)
          ? m_config->warp_size * ((gpu_cta_size / m_config->warp_size) + 1)
          : gpu_cta_size;
}

void shader_core_ctx::decrement_atomic_count(unsigned wid, unsigned n) {
  assert(m_warp[wid]->get_n_atomic() >= n);
  m_warp[wid]->dec_n_atomic(n);
}

void shader_core_ctx::broadcast_barrier_reduction(unsigned cta_id,
                                                  unsigned bar_id,
                                                  warp_set_t warps) {
  for (unsigned i = 0; i < m_config->max_warps_per_shader; i++) {
    if (warps.test(i)) {
      const warp_inst_t *inst =
          m_warp[i]->restore_info_of_last_inst_at_barrier();
      const_cast<warp_inst_t *>(inst)->broadcast_barrier_reduction(
          inst->get_active_mask());
    }
  }
}

/*
����Ԥȡ��Ԫ��Ӧbuffer�Ƿ�����������һֱ������
*/
bool shader_core_ctx::fetch_unit_response_buffer_full() const { return false; }

/*
SIMT Core����Ԥȡ��ָ�����ݰ��������ݰ���mf����ķô���Ϊ��mf�����mem_access_type����Ϊ��INST_ACC_R��
mem_access_type��������ʱ��ģ�����жԲ�ͬ���͵Ĵ洢�����в�ͬ�ķô����ͣ�
     MA_TUP(GLOBAL_ACC_R),        ��global memory��
     MA_TUP(LOCAL_ACC_R),         ��local memory��
     MA_TUP(CONST_ACC_R),         �ӳ��������
     MA_TUP(TEXTURE_ACC_R),       ���������
     MA_TUP(GLOBAL_ACC_W),        ��global memoryд
     MA_TUP(LOCAL_ACC_W),         ��local memoryд
  ��V100�У�L1 cache��m_write_policyΪWRITE_THROUGH��ʵ����L1_WRBK_ACCҲ�����õ���
     MA_TUP(L1_WRBK_ACC),         L1����write back
  ��V100�У���L2 cacheд������ʱ����ȡlazy_fetch_on_read���ԣ����ҵ�һ��cache block
  ���ʱ��������cache block�Ǳ�MODIFIED������Ҫ�����cache blockд�ص���һ���洢��
  ��˻����L2_WRBK_ACC���ʣ�������ʾ���Ϊ��д�ر������MODIFIED cache block��
     MA_TUP(L2_WRBK_ACC),         L2����write back
     MA_TUP(INST_ACC_R),          ��ָ����
  L1_WR_ALLOC_R/L2_WR_ALLOC_R��V100��������ʱ�ò�����
     MA_TUP(L1_WR_ALLOC_R),       L1����write-allocate��cacheд�����У��������п����cache��
                                  д���cache�飩
  L1_WR_ALLOC_R/L2_WR_ALLOC_R��V100��������ʱ�ò�����
     MA_TUP(L2_WR_ALLOC_R),       L2����write-allocate
*/
void shader_core_ctx::accept_fetch_response(mem_fetch *mf) {
  mf->set_status(IN_SHADER_FETCHED,
                 m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
  //m_L1I��ָ���L1-Cache����mf���뵽m_L1I�С�
  m_L1I->fill(mf, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
}

/*
����LDST��Ԫ��Ӧbuffer�Ƿ�������LD/ST��Ԫ����ӦFIFO�е����ݰ��� >= GPU���õ���Ӧ�����е������Ӧ��������
����Ҫע����ǣ�LD/ST��ԪҲ��һ��m_response_fifo����m_response_fifo.size()��ȡ���Ǹ�fifo�Ѿ��洢��mf��
Ŀ�������Ŀ�ܹ��жϸ�fifo�Ƿ�������m_config->ldst_unit_response_queue_size�������õĸ�fifo�����������
һ��m_response_fifo.size()�������õ�����������ͻ᷵��True����ʾ��fifo������
*/
bool shader_core_ctx::ldst_unit_response_buffer_full() const {
  return m_ldst_unit->response_buffer_full();
}

/*
SIMT Core��Ⱥ����Ԥȡ��data���ݰ���mf�����mem_access_type���Ͳ�ΪINST_ACC_R��
mem_access_type��������ʱ��ģ�����жԲ�ͬ���͵Ĵ洢�����в�ͬ�ķô����ͣ�
     MA_TUP(GLOBAL_ACC_R),        ��global memory��
     MA_TUP(LOCAL_ACC_R),         ��local memory��
     MA_TUP(CONST_ACC_R),         �ӳ��������
     MA_TUP(TEXTURE_ACC_R),       ���������
     MA_TUP(GLOBAL_ACC_W),        ��global memoryд
     MA_TUP(LOCAL_ACC_W),         ��local memoryд
  ��V100�У�L1 cache��m_write_policyΪWRITE_THROUGH��ʵ����L1_WRBK_ACCҲ�����õ���
     MA_TUP(L1_WRBK_ACC),         L1����write back
  ��V100�У���L2 cacheд������ʱ����ȡlazy_fetch_on_read���ԣ����ҵ�һ��cache block
  ���ʱ��������cache block�Ǳ�MODIFIED������Ҫ�����cache blockд�ص���һ���洢��
  ��˻����L2_WRBK_ACC���ʣ�������ʾ���Ϊ��д�ر������MODIFIED cache block��
     MA_TUP(L2_WRBK_ACC),         L2����write back
     MA_TUP(INST_ACC_R),          ��ָ����
  L1_WR_ALLOC_R/L2_WR_ALLOC_R��V100��������ʱ�ò�����
     MA_TUP(L1_WR_ALLOC_R),       L1����write-allocate��cacheд�����У��������п����cache��
                                  д���cache�飩
  L1_WR_ALLOC_R/L2_WR_ALLOC_R��V100��������ʱ�ò�����
     MA_TUP(L2_WR_ALLOC_R),       L2����write-allocate
*/
void shader_core_ctx::accept_ldst_unit_response(mem_fetch *mf) {
  m_ldst_unit->fill(mf);
}

/*
Shader Core��SM����дȷ�ϵĶ���������mf����warp���ѷ��͵���δ�յ�дȷ�ϵĴ洢��������
*/
void shader_core_ctx::store_ack(class mem_fetch *mf) {
  assert(mf->get_type() == WRITE_ACK ||
         (m_config->gpgpu_perfect_mem && mf->get_is_write()));
  unsigned warp_id = mf->get_wid();
  //�����ѷ��͵���δ�յ�дȷ�ϵĴ洢��������
  m_warp[warp_id]->dec_store_req();
}

void shader_core_ctx::print_cache_stats(FILE *fp, unsigned &dl1_accesses,
                                        unsigned &dl1_misses) {
  m_ldst_unit->print_cache_stats(fp, dl1_accesses, dl1_misses);
}

void shader_core_ctx::get_cache_stats(cache_stats &cs) {
  // Adds stats from each cache to 'cs'
  cs += m_L1I->get_stats();          // Get L1I stats
  m_ldst_unit->get_cache_stats(cs);  // Get L1D, L1C, L1T stats
}

void shader_core_ctx::get_L1I_sub_stats(struct cache_sub_stats &css) const {
  if (m_L1I) m_L1I->get_sub_stats(css);
}
void shader_core_ctx::get_L1D_sub_stats(struct cache_sub_stats &css) const {
  m_ldst_unit->get_L1D_sub_stats(css);
}
void shader_core_ctx::get_L1C_sub_stats(struct cache_sub_stats &css) const {
  m_ldst_unit->get_L1C_sub_stats(css);
}
void shader_core_ctx::get_L1T_sub_stats(struct cache_sub_stats &css) const {
  m_ldst_unit->get_L1T_sub_stats(css);
}

void shader_core_ctx::get_icnt_power_stats(long &n_simt_to_mem,
                                           long &n_mem_to_simt) const {
  n_simt_to_mem += m_stats->n_simt_to_mem[m_sid];
  n_mem_to_simt += m_stats->n_mem_to_simt[m_sid];
}

/*
���ذ��ڵ�ǰShader Core��kernel���ں˺�����Ϣ��kernel_info_t����
*/
kernel_info_t *shd_warp_t::get_kernel_info() const {
  return m_shader->get_kernel_info();
}
/*
����warp�Ѿ�ִ����ϵı�־���Ѿ���ɵ��߳�����=warp�Ĵ�Сʱ���ʹ����warp�Ѿ���ɡ�
*/
bool shd_warp_t::functional_done() const {
  return get_n_completed() == m_warp_size;
}

/*
��δ��������warp�Ƿ��Ѿ����ִ�в��ҿ��Ի��ա�
*/
bool shd_warp_t::hardware_done() const {
  //functional_done()����warp�Ѿ�ִ����ϵı�־���Ѿ���ɵ��߳�����=warp�Ĵ�Сʱ���ʹ����warp�Ѿ�
  //��ɡ�stores_done()��������store�ô������Ƿ��Ѿ�ȫ��ִ���꣬�ѷ��͵���δȷ�ϵ�д�洢�����Ѿ���
  //��д���󣬵���û�յ�дȷ���ź�ʱ����m_stores_outstanding=0ʱ����������store�ô������Ѿ�ȫ��ִ��
  //�꣬����m_stores_outstanding�ڷ���һ��д����ʱ+=1�����յ�һ��дȷ��ʱ-=1��
  //m_inst_fetch_buffer�к���Чָ��ʱ�ҽ���ָ��������������warp��m_ibufferʱ����������ˮ���е�
  //ָ����m_inst_in_pipeline��ע��������decode()�����л���warp��m_ibuffer����2��ָ�����ָ����
  //��д�ز���ʱ��������ˮ���е�ָ����m_inst_in_pipeline��inst_in_pipeline()������ˮ���е�ָ������
  //m_inst_in_pipeline��

  //����һ��warp��ɵı�־������������ɣ��ֱ��ǣ�1��warp�Ѿ�ִ����ϣ�2������store�ô������Ѿ�ȫ��ִ
  //���ꣻ3����ˮ���е�ָ������Ϊ0��
  return functional_done() && stores_done() && !inst_in_pipeline();
}

/*
����warp�Ƿ����ڣ�warp�Ѿ�ִ��������ڵȴ����ں˳�ʼ����CTA����barrier��memory barrier������δ���
��ԭ�Ӳ������ĸ��������ڵȴ�״̬��
*/
bool shd_warp_t::waiting() {
  if (functional_done()) {
    // waiting to be initialized with a kernel
    return true;
  } else if (m_shader->warp_waiting_at_barrier(m_warp_id)) {
    // waiting for other warps in CTA to reach barrier
    return true;
  } else if (m_shader->warp_waiting_at_mem_barrier(m_warp_id)) {
    // waiting for memory barrier
    return true;
  } else if (m_n_atomic > 0) {
    // waiting for atomic operation to complete at memory:
    // this stall is not required for accurate timing model, but rather we
    // stall here since if a call/return instruction occurs in the meantime
    // the functional execution of the atomic when it hits DRAM can cause
    // the wrong register to be read.
    return true;
  } else if (m_waiting_ldgsts) {  // Waiting for LDGSTS to finish
    return true;
  }
  return false;
}

/*
debug: dp 0 0 0 (��һ��0����mask�����㼴�ɡ��ڶ���0�����0��shader��������0�����0���ڴ��ӷ���)��
*/
void shd_warp_t::print(FILE *fout) const {
  if (!done_exit()) {
    fprintf(fout,
            "w%02u npc: 0x%04llx, done:%c%c%c%c:%2u i:%u s:%u a:%u (done: ",
            m_warp_id, m_next_pc, (functional_done() ? 'f' : ' '),
            (stores_done() ? 's' : ' '), (inst_in_pipeline() ? ' ' : 'i'),
            (done_exit() ? 'e' : ' '), n_completed, m_inst_in_pipeline,
            m_stores_outstanding, m_n_atomic);
    for (unsigned i = m_warp_id * m_warp_size;
         i < (m_warp_id + 1) * m_warp_size; i++) {
      if (m_shader->ptx_thread_done(i))
        fprintf(fout, "1");
      else
        fprintf(fout, "0");
      if ((((i + 1) % 4) == 0) && (i + 1) < (m_warp_id + 1) * m_warp_size)
        fprintf(fout, ",");
    }
    fprintf(fout, ") ");
    fprintf(fout, " active=%s", m_active_threads.to_string().c_str());
    fprintf(fout, " last fetched @ %5llu", m_last_fetch);
    if (m_imiss_pending) fprintf(fout, " i-miss pending");
    fprintf(fout, "\n");
  }
}

void shd_warp_t::print_ibuffer(FILE *fout) const {
  fprintf(fout, "  ibuffer[%2u] : ", m_warp_id);
  for (unsigned i = 0; i < IBUFFER_SIZE; i++) {
    const inst_t *inst = m_ibuffer[i].m_inst;
    if (inst)
      inst->print_insn(fout);
    else if (m_ibuffer[i].m_valid)
      fprintf(fout, " <invalid instruction> ");
    else
      fprintf(fout, " <empty> ");
  }
  fprintf(fout, "\n");
}

/*
����collector unit�������������set_id����Ϊ��
    enum { SP_CUS, DP_CUS, SFU_CUS, TENSOR_CORE_CUS, INT_CUS, MEM_CUS, GEN_CUS };
������collector unit��7������Ӧ��SP��Ԫһ������Ӧ��DP��Ԫһ����......������num_cu��Ϊ��Ӧ�ڸ�����
Ԫ��collector unit�ı���ڵ���ʱgpgpu_operand_collector_num_units_sp��Ϊ��Ӧ��SP��Ԫ��CU��������
    m_operand_collector.add_cu_set(
            SP_CUS, m_config->gpgpu_operand_collector_num_units_sp,
            m_config->gpgpu_operand_collector_num_out_ports_sp);
������Ĵ����У�m_cus[set_id]�Ƕ�Ӧ��set_idָʾ�ĵ�Ԫ��collector unit�ı����m_cu�����������ռ���
��Ԫ��

��������׻�����m_cus��һ���ֵ䣬�䶨��Ϊ��
  typedef std::map<unsigned collector_set_id,      // �ռ�����Ԫ��set_id
                   std::vector<collector_unit_t>>  // �ռ�����Ԫ������
          cu_sets_t;

��V100�����У�����ÿ��set_id���ռ�����Ԫ����ĿΪ��
   //SP_CUS           gpgpu_operand_collector_num_units_sp = 4
   //DP_CUS           gpgpu_operand_collector_num_units_dp = 0
   //SFU_CUS          gpgpu_operand_collector_num_units_sfu = 4
   //INT_CUS          gpgpu_operand_collector_num_units_int = 0
   //TENSOR_CORE_CUS  gpgpu_operand_collector_num_units_tensor_core = 4
   //MEM_CUS          gpgpu_operand_collector_num_units_mem = 2
   //GEN_CUS          gpgpu_operand_collector_num_units_gen = 8

�����
    m_cus[SP_CUS         ]��һ��vector���洢��SP         ��Ԫ��4���ռ�����Ԫ��
    m_cus[DP_CUS         ]��һ��vector���洢��DP         ��Ԫ��0���ռ�����Ԫ��
    m_cus[SFU_CUS        ]��һ��vector���洢��SFU        ��Ԫ��4���ռ�����Ԫ��
    m_cus[INT_CUS        ]��һ��vector���洢��INT        ��Ԫ��0���ռ�����Ԫ��
    m_cus[TENSOR_CORE_CUS]��һ��vector���洢��TENSOR_CORE��Ԫ��4���ռ�����Ԫ��
    m_cus[MEM_CUS        ]��һ��vector���洢��MEM        ��Ԫ��2���ռ�����Ԫ��
    m_cus[GEN_CUS        ]��һ��vector���洢��GEN        ��Ԫ��8���ռ�����Ԫ��

��m_cu���壺
    collector_unit_t *m_cu;
�洢���������е��ռ�����Ԫ����m_cu��һ��vector���洢�����е��ռ�����Ԫ��
*/
void opndcoll_rfu_t::add_cu_set(unsigned set_id, unsigned num_cu,
                                unsigned num_dispatch) {
  //m_cus��set_id��Ӧ�ռ�����Ԫ�ĵ��ֵ䡣
  m_cus[set_id].reserve(num_cu);  // this is necessary to stop pointers in m_cu
                                  // from being invalid do to a resize;
  for (unsigned i = 0; i < num_cu; i++) {
    //�����ռ�����Ԫ��m_cus[set_id]��set_id��Ӧ�ռ�����Ԫ�������set_id����Ϊ��
    //    enum { SP_CUS, DP_CUS, SFU_CUS, TENSOR_CORE_CUS, INT_CUS, MEM_CUS, GEN_CUS };
    m_cus[set_id].push_back(collector_unit_t());
    //m_cu�������ռ�����Ԫ�ļ��ϣ��������е�m_cus�ֵ��е������ռ�����Ԫ��
    m_cu.push_back(&m_cus[set_id].back());
  }
  // for now each collector set gets dedicated dispatch units.
  //Ŀǰ��ÿ���ռ���set����ר�õĵ��ȵ�Ԫ����gpgpu_operand_collector_num_out_ports_sp��ȷ������
  //V100�����У�
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
  //m_cus[set_id]����Ӧ��set_id���ռ�����Ԫ��
  //    m_cus[SP_CUS         ]��һ��vector���洢��SP         ��Ԫ��4���ռ�����Ԫ��
  //    m_cus[DP_CUS         ]��һ��vector���洢��DP         ��Ԫ��0���ռ�����Ԫ��
  //    m_cus[SFU_CUS        ]��һ��vector���洢��SFU        ��Ԫ��4���ռ�����Ԫ��
  //    m_cus[INT_CUS        ]��һ��vector���洢��INT        ��Ԫ��0���ռ�����Ԫ��
  //    m_cus[TENSOR_CORE_CUS]��һ��vector���洢��TENSOR_CORE��Ԫ��4���ռ�����Ԫ��
  //    m_cus[MEM_CUS        ]��һ��vector���洢��MEM        ��Ԫ��2���ռ�����Ԫ��
  //    m_cus[GEN_CUS        ]��һ��vector���洢��GEN        ��Ԫ��8���ռ�����Ԫ��
  for (unsigned i = 0; i < num_dispatch; i++) {
    m_dispatch_units.push_back(dispatch_unit_t(&m_cus[set_id]));
  }
}

/*
input_port_t�Ķ��壺
    class input_port_t {
    public:
      input_port_t(port_vector_t &input, port_vector_t &output,
                  uint_vector_t cu_sets)
          : m_in(input), m_out(output), m_cu_sets(cu_sets) {
        assert(input.size() == output.size());
        assert(not m_cu_sets.empty());
      }
      // private:
      //
      port_vector_t m_in, m_out;
      uint_vector_t m_cu_sets;
    };
port_vector_t�����Ͷ���Ϊ�洢�Ĵ�������register_set��������
    typedef std::vector<register_set *> port_vector_t;
uint_vector_t�����Ͷ���Ϊ�洢�ռ�����Ԫset_id��������
    typedef std::vector<unsigned int> uint_vector_t;
add_port�ǽ�����׶εļ�����ˮ�߼Ĵ�������ID_OC_SP�ȣ��Լ������������ռ��������ļĴ�����
��OC_EX_SP�ȣ���Ӧ�����������ռ�����Ԫset_id����ӽ��������ռ����ࡣ����:
    for (unsigned i = 0; i < m_config->gpgpu_operand_collector_num_in_ports_sp;
         i++) {
      //m_pipeline_reg�Ķ��壺std::vector<register_set> m_pipeline_reg;
      in_ports.push_back(&m_pipeline_reg[ID_OC_SP]);
      out_ports.push_back(&m_pipeline_reg[OC_EX_SP]);
      cu_sets.push_back((unsigned)SP_CUS);
      cu_sets.push_back((unsigned)GEN_CUS);
      m_operand_collector.add_port(in_ports, out_ports, cu_sets);
      in_ports.clear(), out_ports.clear(), cu_sets.clear();
    }
�϶δ������ΪSP��Ԫ��Ӷ˿ڣ��������õ�gpgpu_operand_collector_num_in_ports_sp��SP��
Ԫ����������ռ����Ķ˿���Ŀ����ΪSP��Ԫ�������˿�Ϊm_pipeline_reg[ID_OC_SP]������˿�
Ϊm_pipeline_reg[OC_EX_SP]���ռ�����Ԫset_idΪSP_CUS�Ķ˿ڣ�������ӽ��Ķ˿ڶ��洢����
��m_in_ports�С�

��ˣ�m_in_ports����
  0-7 -> {{m_pipeline_reg[ID_OC_SP], m_pipeline_reg[ID_OC_SFU], m_pipeline_reg[ID_OC_MEM],
           m_pipeline_reg[ID_OC_TENSOR_CORE], m_pipeline_reg[ID_OC_DP], m_pipeline_reg[ID_OC_INT],
           m_config->m_specialized_unit[0].ID_OC_SPEC_ID, m_config->m_specialized_unit[1].ID_OC_SPEC_ID, 
           m_config->m_specialized_unit[2].ID_OC_SPEC_ID, m_config->m_specialized_unit[3].ID_OC_SPEC_ID,
           m_config->m_specialized_unit[4].ID_OC_SPEC_ID, m_config->m_specialized_unit[5].ID_OC_SPEC_ID,
           m_config->m_specialized_unit[6].ID_OC_SPEC_ID, m_config->m_specialized_unit[7].ID_OC_SPEC_ID},
          {m_pipeline_reg[OC_EX_SP], m_pipeline_reg[OC_EX_SFU], m_pipeline_reg[OC_EX_MEM],
           m_pipeline_reg[OC_EX_TENSOR_CORE], m_pipeline_reg[OC_EX_DP], m_pipeline_reg[OC_EX_INT],
           m_config->m_specialized_unit[0].OC_EX_SPEC_ID, m_config->m_specialized_unit[1].OC_EX_SPEC_ID, 
           m_config->m_specialized_unit[2].OC_EX_SPEC_ID, m_config->m_specialized_unit[3].OC_EX_SPEC_ID,
           m_config->m_specialized_unit[4].OC_EX_SPEC_ID, m_config->m_specialized_unit[5].OC_EX_SPEC_ID,
           m_config->m_specialized_unit[6].OC_EX_SPEC_ID, m_config->m_specialized_unit[7].OC_EX_SPEC_ID},
          GEN_CUS}
    8 -> {m_pipeline_reg[ID_OC_SP], m_pipeline_reg[OC_EX_SP], {SP_CUS, GEN_CUS}}
    9 -> {m_pipeline_reg[ID_OC_SFU], m_pipeline_reg[OC_EX_SFU], {SFU_CUS, GEN_CUS}}
   10 -> {m_pipeline_reg[ID_OC_TENSOR_CORE], m_pipeline_reg[OC_EX_TENSOR_CORE]
   11 -> {m_pipeline_reg[ID_OC_MEM], m_pipeline_reg[OC_EX_MEM], {MEM_CUS, GEN_CUS}}

��ǰ���warp�����������ﵥ��Sahder Core�ڵ�warp�������ĸ�����gpgpu_num_sched_per_core
���ò���������Volta�ܹ�ÿ������4��warp��������ÿ���������Ĵ������룺
     schedulers.push_back(new lrr_scheduler(
             m_stats, this, m_scoreboard, m_simt_stack, &m_warp,
             &m_pipeline_reg[ID_OC_SP], &m_pipeline_reg[ID_OC_DP],
             &m_pipeline_reg[ID_OC_SFU], &m_pipeline_reg[ID_OC_INT],
             &m_pipeline_reg[ID_OC_TENSOR_CORE], m_specilized_dispatch_reg,
             &m_pipeline_reg[ID_OC_MEM], i));
�ڷ�������У�warp���������ɷ����ָ�����ָ�����ͷַ�����ͬ�ĵ�Ԫ����Щ��Ԫ����SP/DP/
SFU/INT/TENSOR_CORE/MEM���ڷ��������ɺ���Ҫ���ָ��ͨ���������ռ�����ָ������Ĳ���
��ȫ���ռ��롣����һ��SM����Ӧ��һ���������ռ������������ķ�����̽�ָ����룺
    m_pipeline_reg[ID_OC_SP]��m_pipeline_reg[ID_OC_DP]��m_pipeline_reg[ID_OC_SFU]��
    m_pipeline_reg[ID_OC_INT]��m_pipeline_reg[ID_OC_TENSOR_CORE]��
    m_pipeline_reg[ID_OC_MEM]
�ȼĴ��������У����Բ������ռ������ռ���������
*/
void opndcoll_rfu_t::add_port(port_vector_t &input, port_vector_t &output,
                              uint_vector_t cu_sets) {
  // m_num_ports++;
  // m_num_collectors += num_collector_units;
  // m_input.resize(m_num_ports);
  // m_output.resize(m_num_ports);
  // m_num_collector_units.resize(m_num_ports);
  // m_input[m_num_ports-1]=input_port;
  // m_output[m_num_ports-1]=output_port;
  // m_num_collector_units[m_num_ports-1]=num_collector_units;
  m_in_ports.push_back(input_port_t(input, output, cu_sets));
}

/*
�������ռ����ĳ�ʼ����num_banks�������ļ��� -gpgpu_num_reg_banks 16 ����ȷ������
V100������Ϊ16��
*/
void opndcoll_rfu_t::init(unsigned num_banks, shader_core_ctx *shader) {
  m_shader = shader;
  m_arbiter.init(m_cu.size(), num_banks);
  // for( unsigned n=0; n<m_num_ports;n++ )
  //    m_dispatch_units[m_output[n]].init( m_num_collector_units[n] );
  
  //��V100�����У�m_num_banks����ʼ��Ϊ16��
  m_num_banks = num_banks;
  //m_warp_size = 32��
  m_warp_size = shader->get_config()->warp_size;

  sub_core_model = shader->get_config()->sub_core_model;
  m_num_warp_scheds = shader->get_config()->gpgpu_num_sched_per_core;
  unsigned reg_id = 0;
  if (sub_core_model) {
    assert(num_banks % shader->get_config()->gpgpu_num_sched_per_core == 0);
    assert(m_num_warp_scheds <= m_cu.size() &&
           m_cu.size() % m_num_warp_scheds == 0);
  }
  //ÿ��warp���������õ�bank����sub_core_modelģʽ�У�ÿ��warp���������õ�bank������
  //���޵ġ���V100�����У�����4��warp��������0��warp���������õ�bankΪ0-3��1��warp��
  //�������õ�bankΪ4-7��2��warp���������õ�bankΪ8-11��3��warp���������õ�bankΪ12-
  //15.
  m_num_banks_per_sched =
      num_banks / shader->get_config()->gpgpu_num_sched_per_core;

  //�ռ�����Ԫ�б��ռ�����Ԫ��m_cu����ÿ���ռ�����Ԫһ�ο�������һ��ָ�����������
  //�Ͷ�Դ�Ĵ���������һ������Դ�Ĵ�����׼�����ˣ����ȵ�Ԫ�Ϳ��Խ�����ȵ������ˮ��
  //�Ĵ�������OC_EX�����ռ�����Ԫm_cu�Ķ��壺
  //    std::vector<collector_unit_t *> m_cu;
  //m_cus[set_id]�Ƕ�Ӧ��set_idָʾ�ĵ�Ԫ��collector unit�ı����m_cu������������
  //������Ԫ��m_cu.size()�������е��ռ�����Ԫ������Ŀ��
  for (unsigned j = 0; j < m_cu.size(); j++) {
    if (sub_core_model) {
      //cusPerSched��ÿ�����������õ��ռ�����Ԫ��Ŀ��
      unsigned cusPerSched = m_cu.size() / m_num_warp_scheds;
      //����reg_id��ʵ�Ƕ�Ӧ�ĵ�������ID��
      reg_id = j / cusPerSched;
    }
    m_cu[j]->init(j, num_banks, shader->get_config(), this, sub_core_model,
                  reg_id, m_num_banks_per_sched);
  }
  for (unsigned j = 0; j < m_dispatch_units.size(); j++) {
    m_dispatch_units[j].init(sub_core_model, m_num_warp_scheds);
  }
  m_initialized = true;
}

/*
��V100�����У�m_num_banks����ʼ��Ϊ16��m_bank_warp_shift����ʼ��Ϊ5�������ڲ�����
�ռ����ļĴ����ļ��У�warp0��r0�Ĵ�������0��bank��...��warp0��r15�Ĵ�������15��bank��
warp0��r16�Ĵ�������0��bank��...��warp0��r31�Ĵ�������15��bank��warp1��r0�Ĵ�������
[0+warp_id]��bank�������Է�sub_core_modelģʽΪ����

����register_bank����������������regnum���ڵ�bank����

Bank0   Bank1   Bank2   Bank3                   ......                  Bank15
w1:r31  w1:r16  w1:r17  w1:r18                  ......                  w1:r30
w1:r15  w1:r0   w1:r1   w1:r2                   ......                  w1:r14
w0:r16  w0:r17  w0:r18  w0:r19                  ......                  w0:r31
w0:r0   w0:r1   w0:r2   w0:r3                   ......                  w0:r15

��sub_core_modelģʽ�У�ÿ��warp���������õ�bank���������޵ġ���V100�����У�����4��
warp��������0��warp���������õ�bankΪ0-3��1��warp���������õ�bankΪ4-7��2��warp����
�����õ�bankΪ8-11��3��warp���������õ�bankΪ12-15��
*/
unsigned register_bank(int regnum, int wid, unsigned num_banks,
                       bool sub_core_model, unsigned banks_per_sched,
                       unsigned sched_id) {
  int bank = regnum;
  //warp��bankƫ�ơ�
  bank += wid;
  //��subcoreģʽ�£�ÿ��warp�������ڼĴ�����������һ������ļĴ����ɹ�ʹ�ã������
  //�����ɵ�������m_id������m_num_banks_per_sched�Ķ���Ϊ��
  //    num_banks / shader->get_config()->gpgpu_num_sched_per_core;
  //��V100�����У�����4��warp��������0��warp���������õ�bankΪ0-3��1��warp��������
  //�õ�bankΪ4-7��2��warp���������õ�bankΪ8-11��3��warp���������õ�bankΪ12-15��
  if (sub_core_model) {
    unsigned bank_num = (bank % banks_per_sched) + (sched_id * banks_per_sched);
    assert(bank_num < num_banks);
    return bank_num;
  } else
    return bank % num_banks;
}

/*
�������ռ�����Bankд�ء�
*/
bool opndcoll_rfu_t::writeback(warp_inst_t &inst) {
  assert(!inst.empty());
  //m_shader->get_regs_written(inst)��ȡһ��ָ��inst����д�صļĴ�����ţ����б�
  //ʽ���ء�
  std::list<unsigned> regs = m_shader->get_regs_written(inst);
  for (unsigned op = 0; op < MAX_REG_OPERANDS; op++) {
    int reg_num = inst.arch_reg.dst[op];  // this math needs to match that used
                                          // in function_info::ptx_decode_inst
    if (reg_num >= 0) {                   // valid register
      //m_bank_warp_shift����ʼ��Ϊ5��
	  unsigned bank =
          register_bank(reg_num, inst.warp_id(), m_num_banks, sub_core_model,
                        m_num_banks_per_sched, inst.get_schd_id());
      if (m_arbiter.bank_idle(bank)) {
        //д�ص��Ĵ���Bank��
        //m_bank_warp_shift����ʼ��Ϊ5��
        m_arbiter.allocate_bank_for_write(
            bank, op_t(&inst, reg_num, m_num_banks, sub_core_model,
                       m_num_banks_per_sched, inst.get_schd_id()));
        inst.arch_reg.dst[op] = -1;
      } else {
        return false;
      }
    }
  }
  for (unsigned i = 0; i < (unsigned)regs.size(); i++) {
    //��V100�����У�gpgpu_clock_gated_reg_fileĬ��Ϊ0��
    if (m_shader->get_config()->gpgpu_clock_gated_reg_file) {
      unsigned active_count = 0;
      for (unsigned i = 0; i < m_shader->get_config()->warp_size;
           i = i + m_shader->get_config()->n_regfile_gating_group) {
        for (unsigned j = 0; j < m_shader->get_config()->n_regfile_gating_group;
             j++) {
          if (inst.get_active_mask().test(i + j)) {
            active_count += m_shader->get_config()->n_regfile_gating_group;
            break;
          }
        }
      }
      m_shader->incregfile_writes(active_count);
    } else {
      //���ӼĴ���д����ĿΪwarp_size��m_shader->get_config()->warp_sizeΪ32��
      m_shader->incregfile_writes(
          m_shader->get_config()->warp_size);  // inst.active_count());
    }
  }
  return true;
}

/*
�������е��ȵ�Ԫ��ÿ����Ԫ�ҵ�һ��׼���õ��ռ�����Ԫ�����е��ȡ�
*/
void opndcoll_rfu_t::dispatch_ready_cu() {
  //Ŀǰÿ���ռ���set����ר�õĵ��ȵ�Ԫ����gpgpu_operand_collector_num_out_ports_sp
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
  //m_dispatch_units��洢�����еĵ������������Ƕ����еĵ���������ѭ����ÿ����������
  //��ǰִ��һ����
  for (unsigned p = 0; p < m_dispatch_units.size(); ++p) {
    //m_dispatch_units[p]�ǵ�p����������
    dispatch_unit_t &du = m_dispatch_units[p];
    //�ӵ�p���������ҵ�һ��׼�������ݵ��ռ�����Ԫ��
    collector_unit_t *cu = du.find_ready();
    if (cu) {
      //�ڶ�PTXָ�������ʱ���м����������Ҫ�ļĴ���������m_operands��ptx_ir.h��
      //ptx_instruction���ж��壺
      //    std::vector<operand_info> m_operands;
      //m_operands����ÿ��ָ�������ʱ�����в���������ӵ����У�������� mad a,b,c 
      //ָ��ʱ���Ὣ a,b,c������������ӽ�m_operands����ÿһ��ָ�������һ����������
      //��m_operands���ù��̶���Ϊ��
      //     if (!m_operands.empty()) {
      //       std::vector<operand_info>::iterator it;
      //       for (it = ++m_operands.begin(); it != m_operands.end(); it++) {
      //         //����������������
      //         num_operands++;
      //         //����������ǼĴ���������ʸ����num_regs������1��
      //         if ((it->is_reg() || it->is_vector())) {
      //           num_regs++;
      //         }
      //       }
      //     }
      //cu->get_num_operands()���ص���num_operandsֵ��cu->get_num_regs()���ص���
      //num_regsֵ��ʵ���ϣ�����һ���������ǼĴ����������ֻ�������������ַ�ȣ�������
      //����num_operands���ڼ���������ֻ�мĴ������������ֵ�ʱ��num_regs�ż�����
      for (unsigned i = 0; i < (cu->get_num_operands() - cu->get_num_regs());
           i++) {
        //����m_shader->get_config()->gpgpu_clock_gated_reg_file��V100��Ϊ0��
        if (m_shader->get_config()->gpgpu_clock_gated_reg_file) {
          unsigned active_count = 0;
          for (unsigned i = 0; i < m_shader->get_config()->warp_size;
               i = i + m_shader->get_config()->n_regfile_gating_group) {
            for (unsigned j = 0;
                 j < m_shader->get_config()->n_regfile_gating_group; j++) {
              if (cu->get_active_mask().test(i + j)) {
                active_count += m_shader->get_config()->n_regfile_gating_group;
                break;
              }
            }
          }
          m_shader->incnon_rf_operands(active_count);
        } else {
          m_shader->incnon_rf_operands(
              m_shader->get_config()->warp_size);  // cu->get_active_count());
        }
      }
      //����ܹ��ӵ�p���������ҵ�һ��׼�������ݵĵ��ռ�����Ԫ�Ļ�����ִ�����ķַ�����
      //dispatch()����Ҫ�����ǣ������ռ�����Ԫ�ռ���Դ�������󣬽�ԭ���ݴ����ռ�����
      //Ԫָ���m_warp�е�ָ���Ƴ���m_output_register�С�
      cu->dispatch();
    }
  }
}

/*
opndcoll_rfu_t::allocate_cu������ID_OC��ˮ�߼Ĵ����е�ָ�������ռ�����Ԫ��
*/
void opndcoll_rfu_t::allocate_cu(unsigned port_num) {
  //�˿ڣ�m_in_Ports��������������ˮ�߼Ĵ������ϣ�ID_OC��������Ĵ������ϣ�OC_EX����
  //ID_OC�˿��е�warp_inst_t�����������ռ�����Ԫ�����⣬���ռ�����Ԫ������������Դ
  //�Ĵ���ʱ�������ɵ��ȵ�Ԫ���ȵ�����ܵ��Ĵ�������OC_EX����m_in_ports�лẬ�ж��
  //input_port_t����ÿ������ֱ��Ӧ��SP/DP/SFU/INT/MEM/TC��Ԫ������һ����Ԫ����
  //���ж��input_port_t���󣬲���һһ��Ӧ�ģ����������SP��Ԫ��input_port_t����ʱ��
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
  //          m_pipeline_reg[m_config->m_specialized_unit[0].ID_OC_SPEC_ID], 
  //          m_pipeline_reg[m_config->m_specialized_unit[1].ID_OC_SPEC_ID], 
  //          m_pipeline_reg[m_config->m_specialized_unit[2].ID_OC_SPEC_ID], 
  //          m_pipeline_reg[m_config->m_specialized_unit[3].ID_OC_SPEC_ID],
  //          m_pipeline_reg[m_config->m_specialized_unit[4].ID_OC_SPEC_ID], 
  //          m_pipeline_reg[m_config->m_specialized_unit[5].ID_OC_SPEC_ID],
  //          m_pipeline_reg[m_config->m_specialized_unit[6].ID_OC_SPEC_ID], 
  //          m_pipeline_reg[m_config->m_specialized_unit[7].ID_OC_SPEC_ID]},
  //         {m_pipeline_reg[OC_EX_SP], m_pipeline_reg[OC_EX_SFU], m_pipeline_reg[OC_EX_MEM],
  //          m_pipeline_reg[OC_EX_TENSOR_CORE], m_pipeline_reg[OC_EX_DP], m_pipeline_reg[OC_EX_INT],
  //          m_pipeline_reg[m_config->m_specialized_unit[0].OC_EX_SPEC_ID], 
  //          m_pipeline_reg[m_config->m_specialized_unit[1].OC_EX_SPEC_ID], 
  //          m_pipeline_reg[m_config->m_specialized_unit[2].OC_EX_SPEC_ID], 
  //          m_pipeline_reg[m_config->m_specialized_unit[3].OC_EX_SPEC_ID],
  //          m_pipeline_reg[m_config->m_specialized_unit[4].OC_EX_SPEC_ID], 
  //          m_pipeline_reg[m_config->m_specialized_unit[5].OC_EX_SPEC_ID],
  //          m_pipeline_reg[m_config->m_specialized_unit[6].OC_EX_SPEC_ID], 
  //          m_pipeline_reg[m_config->m_specialized_unit[7].OC_EX_SPEC_ID]},
  //         GEN_CUS}
  //   8 -> {m_pipeline_reg[ID_OC_SP], m_pipeline_reg[OC_EX_SP], {SP_CUS, GEN_CUS}}
  //   9 -> {m_pipeline_reg[ID_OC_SFU], m_pipeline_reg[OC_EX_SFU], {SFU_CUS, GEN_CUS}}
  //  10 -> {m_pipeline_reg[ID_OC_TENSOR_CORE], m_pipeline_reg[OC_EX_TENSOR_CORE]
  //  11 -> {m_pipeline_reg[ID_OC_MEM], m_pipeline_reg[OC_EX_MEM], {MEM_CUS, GEN_CUS}}
  //���������inp=m_in_ports[port_num]�ǵ�port_num��input_port_t����
  input_port_t &inp = m_in_ports[port_num];
  //��inp������˿ڽ���ѭ����
  for (unsigned i = 0; i < inp.m_in.size(); i++) {
    //�����Ĵ�������(*inp.m_in[i])�Ƿ����һ���ǿռĴ�����׼���á�
    if ((*inp.m_in[i]).has_ready()) {
      // find a free cu
      //������ǰ�˿��ڵ������ռ�����Ԫ���ҵ�һ�����е��ռ�����Ԫ��
      for (unsigned j = 0; j < inp.m_cu_sets.size(); j++) {
        //m_cus��һ���ֵ䣬�洢�����е��ռ�����Ԫ���䶨�壺
        //   //id��Ӧ�ռ�����Ԫ�ĵ��ֵ䡣
        //   typedef std::map<unsigned /* collector set */,
        //                    std::vector<collector_unit_t> /*collector sets*/>
        //       cu_sets_t;
        //   //�������ռ����ļ��ϡ�
        //   cu_sets_t m_cus;
        //���磬inp.m_cu_sets[j]������SP_CUS����ôm_cus[inp.m_cu_sets[j]]���൱����
        //m_cus[SP_CUS]����һ��vector���洢��SP��Ԫ�Ķ���ռ�����Ԫ��
        std::vector<collector_unit_t> &cu_set = m_cus[inp.m_cu_sets[j]];
        bool allocated = false;
        //cuLowerBound�ǵ�ǰ���������õ��ռ�����Ԫ���½硣
        unsigned cuLowerBound = 0;
        //cuUpperBound�ǵ�ǰ���������õ��ռ�����Ԫ���Ͻ硣
        unsigned cuUpperBound = cu_set.size();
        //schd_id�Ƿ��䵱ǰָ��ĵ�����ID��
        unsigned schd_id;
        //��V100�����У�sub_core_modelΪ1��
        if (sub_core_model) {
          // Sub core model only allocates on the subset of CUs assigned to the
          // scheduler that issued
          unsigned reg_id = (*inp.m_in[i]).get_ready_reg_id();
          //��ȡ���䵱ǰָ��ĵ�����ID��
          schd_id = (*inp.m_in[i]).get_schd_id(reg_id);
          assert(cu_set.size() % m_num_warp_scheds == 0 &&
                 cu_set.size() >= m_num_warp_scheds);
          //һ�����������õ��ռ�����Ԫ��Ŀ��
          unsigned cusPerSched = cu_set.size() / m_num_warp_scheds;
          //cuLowerBound�ǵ�ǰ���������õ��ռ�����Ԫ���½硣
          cuLowerBound = schd_id * cusPerSched;
          //cuUpperBound�ǵ�ǰ���������õ��ռ�����Ԫ���Ͻ硣
          cuUpperBound = cuLowerBound + cusPerSched;
          assert(0 <= cuLowerBound && cuUpperBound <= cu_set.size());
        }
        //���cuLowerBound-(cuUpperBound-1)��Χ�ڵ��ռ�����Ԫ�Ƿ��п��е��ռ�����Ԫ��
        for (unsigned k = cuLowerBound; k < cuUpperBound; k++) {
          if (cu_set[k].is_free()) {
            //�ҵ�һ�����е��ռ�����Ԫ��������Ϊk��
            collector_unit_t *cu = &cu_set[k];
            //��ǰ�ռ�����ԪΪ����״̬�Ļ���cu->allocate�Ϳ��Խ�һ���µ�warpָ��ŵ�
            //����ռ�����Ԫ�С�
            allocated = cu->allocate(inp.m_in[i], inp.m_out[i]);
            //���ռ�����Ԫ��ȡ���е�Դ���������������Ƿ���m_queue[bank]���С�
            m_arbiter.add_read_requests(cu);
            break;
          }
        }
        if (allocated) break;  // cu has been allocated, no need to search more.
      }
      // break;  // can only service a single input, if it failed it will fail
      // for
      // others.
    }
  }
}

/*
�ٲ���������󣬲����ز�ͬ�Ĵ���Bank�е�op_t�б�������Щ�Ĵ���Bank������Write״̬��
�ڸú����У��ٲ���������󲢷���op_t���б���Щop_tλ�ڲ�ͬ�ļĴ���Bank�У�������Щ
�Ĵ���Bank������Write״̬��
*/
void opndcoll_rfu_t::allocate_reads() {
  // process read requests that do not have conflicts
  //����û�г�ͻ�Ķ������ڸú����У��ٲ���������󲢷���op_t���б���Щop_tλ�ڲ�
  //ͬ�ļĴ���Bank�У�������Щ�Ĵ���Bank������Write״̬��
  std::list<op_t> allocated = m_arbiter.allocate_reads();
  //read_ops�ֵ䣬�洢��i��Bank�Ķ���������
  std::map<unsigned, op_t> read_ops;
  for (std::list<op_t>::iterator r = allocated.begin(); r != allocated.end();
       r++) {
    const op_t &rr = *r;
    unsigned reg = rr.get_reg();
    unsigned wid = rr.get_wid();
    unsigned bank = register_bank(reg, wid, m_num_banks, sub_core_model,
                                  m_num_banks_per_sched, rr.get_sid());
    //allocate_for_read�����������bank��Bank�Ķ�״̬�����Ĳ�����Ϊop���䶨��Ϊ��
    //    void allocate_for_read(unsigned bank, const op_t &op) {
    //      assert(bank < m_num_banks);
    //      m_allocated_bank[bank].alloc_read(op);
    //    }
    m_arbiter.allocate_for_read(bank, rr);
    read_ops[bank] = rr;
  }
  std::map<unsigned, op_t>::iterator r;
  //����read_ops�ֵ䣬�洢��i��Bank�Ķ����������ֵ䣬�������еĶ���������
  for (r = read_ops.begin(); r != read_ops.end(); ++r) {
    op_t &op = r->second;
    unsigned cu = op.get_oc_id();
    //op.get_operand()���ص�ǰ����������ָ�����е�Դ�������е�����
    unsigned operand = op.get_operand();
    //�����ͷŵ�m_not_readyλ�����ĵ�operandλ��������������ָ��ĵ�operand��Դ��
    //�����Ѿ����ھ���״̬��
    m_cu[cu]->collect_operand(operand);
    //gpgpu_clock_gated_reg_file��V100������Ϊ0��
    if (m_shader->get_config()->gpgpu_clock_gated_reg_file) {
      unsigned active_count = 0;
      for (unsigned i = 0; i < m_shader->get_config()->warp_size;
           i = i + m_shader->get_config()->n_regfile_gating_group) {
        for (unsigned j = 0; j < m_shader->get_config()->n_regfile_gating_group;
             j++) {
          if (op.get_active_mask().test(i + j)) {
            active_count += m_shader->get_config()->n_regfile_gating_group;
            break;
          }
        }
      }
      m_shader->incregfile_reads(active_count);
    } else {
      //����SM�ļĴ������ĸ�����32��
      m_shader->incregfile_reads(
          m_shader->get_config()->warp_size);  // op.get_active_count());
    }
  }
}

/*
���ص�ǰ�ռ�����Ԫ�Ƿ�����Դ��������׼�����ˡ�
*/
bool opndcoll_rfu_t::collector_unit_t::ready() const {
  //�����ռ�����Ԫ�ռ���Դ��������ָ��Ƴ���m_output_register�С������Ǹ��ռ�����Ԫ
  //��û�б�free�����ұ�־����Դ�������Ƿ��Ѿ�׼���õ�λͼm_not_readyΪ�գ�������Դ����
  //������׼���ã���������Ҫ����Ĵ���m_output_register���пռ�����ƽ�ȥ��m_reg_id��ʵ
  //�Ƕ�Ӧ�ĵ�������ID����m_output_register���ҵ�m_reg_id�����������ܹ�ʹ�õĲ��Ƿ���á�
  return (!m_free) && m_not_ready.none() &&
         (*m_output_register).has_free(m_sub_core_model, m_reg_id);
}

void opndcoll_rfu_t::collector_unit_t::dump(
    FILE *fp, const shader_core_ctx *shader) const {
  if (m_free) {
    fprintf(fp, "    <free>\n");
  } else {
    m_warp->print(fp);
    for (unsigned i = 0; i < MAX_REG_OPERANDS * 2; i++) {
      if (m_not_ready.test(i)) {
        std::string r = m_src_op[i].get_reg_string();
        fprintf(fp, "    '%s' not ready\n", r.c_str());
      }
    }
  }
}

/*
�ռ�����Ԫ��ĳ�ʼ����
*/
void opndcoll_rfu_t::collector_unit_t::init(unsigned n, unsigned num_banks,
                                            const core_config *config,
                                            opndcoll_rfu_t *rfu,
                                            bool sub_core_model,
                                            unsigned reg_id,
                                            unsigned banks_per_sched) {
  //�������ĸ��������ռ�����
  m_rfu = rfu;
  //�ռ�����Ԫ��ID��
  m_cuid = n;
  //�������ռ����ļĴ���bank����
  m_num_banks = num_banks;
  assert(m_warp == NULL);
  //�ռ�����Ԫ�洢���ĸ�warpָ��Դ�Ĵ�����
  m_warp = new warp_inst_t(config);
  //sub_core_modelģʽ��ÿ��warp���������õ�bank���������޵ġ�
  m_sub_core_model = sub_core_model;
  m_reg_id = reg_id;
  m_num_banks_per_sched = banks_per_sched;
}

/*
��ǰ�ռ�����ԪΪ����״̬�Ļ����Ϳ��Խ�һ���µ�warpָ��ŵ�����ռ�����Ԫ�С�
*/
bool opndcoll_rfu_t::collector_unit_t::allocate(register_set *pipeline_reg_set,
                                                register_set *output_reg_set) {
  assert(m_free);
  assert(m_not_ready.none());
  m_free = false;
  //�����ռ�����Ԫ�ռ���Դ�������󣬽�ָ���Ƴ���m_output_register�С�
  m_output_register = output_reg_set;
  //pipeline_reg_set->get_ready()Ϊ��ȡһ���ǿռĴ���������ָ���Ƴ�������������ָ�
  warp_inst_t **pipeline_reg = pipeline_reg_set->get_ready();
  if ((pipeline_reg) and !((*pipeline_reg)->empty())) {
    //��ȡpipeline_reg�е�ָ���warp ID��
    m_warp_id = (*pipeline_reg)->warp_id();
    std::vector<int> prev_regs;  // remove duplicate regs within same instr
    //ʵ������£�һ��PTX����SASSָ������кܶ��Դ�Ĵ�����������ЩԴ�Ĵ����п������ظ�
    //�ļĴ�����prev_regs���������洢��Ч��ȥ�ص�Դ�Ĵ����ı�š����������п��ܶ�����
    //ȡ��ͬ�ļĴ�����ֵ�������轫�µ���Ч�Ĵ�����ֵ������prev_regs�С������Ƕ�һ��ָ��
    //������Դ�Ĵ������ѭ�����涨һ��ָ���е�Դ�Ĵ�����Ŀ��󲻳���MAX_REG_OPERANDS=32��
    for (unsigned op = 0; op < MAX_REG_OPERANDS; op++) {
      int reg_num =
          (*pipeline_reg)
              ->arch_reg.src[op];  // this math needs to match that used in
                                   // function_info::ptx_decode_inst
      bool new_reg = true;
      for (auto r : prev_regs) {
        if (r == reg_num) new_reg = false;
          //�������prev_regs�Ѿ����˵�ǰѭ���ļĴ������reg_num����˵��reg_num�Ѿ���
          //��prev_regs�ˣ��ͽ�new_reg��Ϊfalse��
      }
      if (reg_num >= 0 && new_reg) {  // valid register
        //һ���µļĴ�������ʱ��������뵽prev_regs�С�
        prev_regs.push_back(reg_num);
        //op_t�����ڱ���Դ���������Ķ���Ϊ��
        //   op_t(collector_unit_t *cu, unsigned op, 
        //        unsigned reg, unsigned num_banks,
        //        unsigned bank_warp_shift, bool sub_core_model,
        //        unsigned banks_per_sched, unsigned sched_id) {
        //     m_valid = true;
        //     m_warp = NULL;
        //     m_cu = cu;
        //     m_operand = op;
        //     m_register = reg;
        //     m_shced_id = sched_id;
        //     m_bank = register_bank(reg, cu->get_warp_id(), num_banks, 
        //                            bank_warp_shift, sub_core_model, 
        //                            banks_per_sched, sched_id);
        //   }
        //register_bank����������������regnum���ڵ�bank����
        
        //m_src_op��һ��op_t���͵������������洢һ��ָ�������Դ��������m_src_op[0]��
        //����0��Դ��������m_src_op[1]�洢��1��Դ��������...��m_src_op[31]�洢��31��
        //Դ��������
		m_src_op[op] =
            op_t(this, op, reg_num, m_num_banks, m_sub_core_model,
                 m_num_banks_per_sched, (*pipeline_reg)->get_schd_id());
        //m_not_ready�Ķ���Ϊ��
        //    std::bitset<MAX_REG_OPERANDS * 2> m_not_ready;
        //m_not_ready��һ��λ�����������洢һ��ָ�������Դ�������Ƿ��ڷǾ���״̬����
        //�����õ�op��Դ������Ϊ�Ǿ���״̬��
        m_not_ready.set(op);
      } else
        //�����һ���ɵļĴ����Ļ����ͽ����ÿա�
        m_src_op[op] = op_t();
    }
    // move_warp(m_warp,*pipeline_reg);
    //m_warp�Ķ���Ϊ��
    //    warp_inst_t *m_warp;
    //�����ǽ�pipeline_reg�е�ָ���Ƴ������������m_warp�У�m_warp�������ڵ�ǰ�ռ�����
    //Ԫ��һ��ָ��ۣ�
    //    m_warp = new warp_inst_t(config);
    //�����ǽ�����ָ�����ˮ�߼Ĵ������Ƴ����ŵ����ռ�����Ԫ���ݴ棬������ָ����ɵ�ǰ��
    //������Ԫ���������ռ�Դ��������
    pipeline_reg_set->move_out_to(m_warp);
    return true;
  }
  return false;
}

/*
�ַ��������ռ�����Ԫ�ռ���Դ�������󣬽�ԭ���ݴ����ռ�����Ԫָ���m_warp�е�ָ���Ƴ���
m_output_register�С�
*/
void opndcoll_rfu_t::collector_unit_t::dispatch() {
  //ȷ��δ������Դ�������Ѿ�û���ˣ���ɽ�һ����ָ���Ƴ���m_output_register�С�
  assert(m_not_ready.none());
  //�����ռ�����Ԫ�ռ���Դ�������󣬽�ԭ���ݴ����ռ�����Ԫָ���m_warp�е�ָ���Ƴ���
  //m_output_register�С�
  m_output_register->move_in(m_sub_core_model, m_reg_id, m_warp);
  //���õ�ǰ�ռ�����ԪΪ����״̬��
  m_free = true;
  //�����ռ�����Ԫ�ռ���Դ�������󣬽�ԭ���ݴ����ռ�����Ԫָ���m_warp�е�ָ���Ƴ���
  //m_output_register�С�
  m_output_register = NULL;
  //���õ�ǰ�ռ�����Ԫ��Դ������Ϊ�ա�
  for (unsigned i = 0; i < MAX_REG_OPERANDS * 2; i++) m_src_op[i].reset();
}

void exec_simt_core_cluster::create_shader_core_ctx() {
  m_core = new shader_core_ctx *[m_config->n_simt_cores_per_cluster];
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; i++) {
    unsigned sid = m_config->cid_to_sid(i, m_cluster_id);
    m_core[i] = new exec_shader_core_ctx(m_gpu, this, sid, m_cluster_id,
                                         m_config, m_mem_config, m_stats);
    m_core_sim_order.push_back(i);
  }
}

simt_core_cluster::simt_core_cluster(class gpgpu_sim *gpu, unsigned cluster_id,
                                     const shader_core_config *config,
                                     const memory_config *mem_config,
                                     shader_core_stats *stats,
                                     class memory_stats_t *mstats) {
  m_config = config;
  m_cta_issue_next_core = m_config->n_simt_cores_per_cluster -
                          1;  // this causes first launch to use hw cta 0
  m_cluster_id = cluster_id;
  m_gpu = gpu;
  m_stats = stats;
  m_memory_stats = mstats;
  m_mem_config = mem_config;
}

/*
simt_core_cluster��SIMT Core��Ⱥ��ǰ�ƽ�һ��ʱ�����ڡ���V100�����У�ÿ����Ⱥ�ڲ�����1��SIMT Core��
*/
void simt_core_cluster::core_cycle() {
  //��SIMT Core��Ⱥ�е�ÿһ��������SIMT Coreѭ����
  for (std::list<unsigned>::iterator it = m_core_sim_order.begin();
       it != m_core_sim_order.end(); ++it) {
    //SIMT Core��Ⱥ�е�ÿһ��������SIMT Core����ǰ�ƽ�һ��ʱ�����ڡ�
    m_core[*it]->cycle();
  }
  //simt_core_sim_order: Select the simulation order of cores in a cluster.
  //simt_core_sim_order��ѡ��Ⱥ��SIMT Core��ģ��˳��ʱ����������Ĭ��Ϊ1������ѭ�����ȡ�����������
  //��ʱ�������ڣ��Ǵ�m_core_sim_order.begin()��ʼ���ȣ����Ϊ��ʵ����ѯ���ȣ���begin()λ���ƶ�����
  //ĩβ���������´ξ��Ǵ�begin+1λ�õ�SIMT Core��ʼ���ȡ�
  if (m_config->simt_core_sim_order == 1) {
    m_core_sim_order.splice(m_core_sim_order.end(), m_core_sim_order,
                            m_core_sim_order.begin());
  }
}

void simt_core_cluster::reinit() {
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; i++)
    m_core[i]->reinit(0, m_config->n_thread_per_shader, true);
}

unsigned simt_core_cluster::max_cta(const kernel_info_t &kernel) {
  return m_config->n_simt_cores_per_cluster * m_config->max_cta(kernel);
}

/*
���ص�ǰSIMT Core��Ⱥ����δ��ɵ��̸߳�����
*/
unsigned simt_core_cluster::get_not_completed() const {
  unsigned not_completed = 0;
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; i++)
    //m_core[i]->get_not_completed()�Ƿ��ص�ǰSIMT Core��Ⱥ�е�i��SM��δ��ɵ��߳�����
    not_completed += m_core[i]->get_not_completed();
  return not_completed;
}

void simt_core_cluster::print_not_completed(FILE *fp) const {
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; i++) {
    unsigned not_completed = m_core[i]->get_not_completed();
    unsigned sid = m_config->cid_to_sid(i, m_cluster_id);
    fprintf(fp, "%u(%u) ", sid, not_completed);
  }
}

float simt_core_cluster::get_current_occupancy(
    unsigned long long &active, unsigned long long &total) const {
  float aggregate = 0.f;
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; i++) {
    aggregate += m_core[i]->get_current_occupancy(active, total);
  }
  return aggregate / m_config->n_simt_cores_per_cluster;
}

unsigned simt_core_cluster::get_n_active_cta() const {
  unsigned n = 0;
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; i++)
    n += m_core[i]->get_n_active_cta();
  return n;
}

/*
����SIMT Core��Ⱥ�еĻ�ԾSM��������
*/
unsigned simt_core_cluster::get_n_active_sms() const {
  unsigned n = 0;
  //�Լ�Ⱥ�е�����SIMT Core����ѭ����
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; i++)
    //m_core[i]->isactive()Ϊ1ʱ������m_core[i]�ǻ�Ծ�ģ�����SM��Ծ��
    n += m_core[i]->isactive();
  return n;
}

/*
������SIMT Core��Ⱥ������ѡ��ĳ����Ⱥ�ڵ�һ��SIMT Core�����䷢��һ���߳̿顣
*/
unsigned simt_core_cluster::issue_block2core() {
  //��ǰ������SIMT Core��Ⱥ������߳̿�ļ�����
  unsigned num_blocks_issued = 0;
  //������ǰSIMT Core��Ⱥ�ڵ�����SIMT Core��Ϊÿ��SM��һ��kernel���䡣
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; i++) {
    //SIMT Core�ı�š�
    unsigned core =
        (i + m_cta_issue_next_core + 1) % m_config->n_simt_cores_per_cluster;

    kernel_info_t *kernel;
    // Jin: fetch kernel according to concurrent kernel setting
    if (m_config->gpgpu_concurrent_kernel_sm) {  // concurrent kernel on sm
      //֧��SM�ϵĲ����ںˣ�Ĭ��Ϊ���ã�����V100�����н��á�
      // concurrent kernel on sm
      // always select latest issued kernel
      kernel_info_t *k = m_gpu->select_kernel();
      kernel = k;
    } else {
      // first select core kernel, if no more cta, get a new kernel
      // only when core completes
      //����ѡ��һ���ں˺�����������ں˺���û�и����CTA��Ҫִ�У��͵��������һ�����ںˡ�
      kernel = m_core[core]->get_kernel();
      if (!m_gpu->kernel_more_cta_left(kernel)) {
        // wait till current kernel finishes
        //���ص�ǰSIMT Core��δ��ɵ��̸߳����������δ��ɵ��̸߳�������0����˵����SM�ϵ��߳��Ѿ�ȫ
        //����������kernelִ����ϡ�
        if (m_core[core]->get_not_completed() == 0) {
          kernel_info_t *k = m_gpu->select_kernel();
          //��kernel����SM�ϡ�
          if (k) m_core[core]->set_kernel(k);
          kernel = k;
        }
      }
    }
    //���kernel�и����CTA��ִ�У�����ǰSM��δ���kernel��ִ�У�����m_core[core]���Է���һ���ں˺�
    //�������䡣
    if (m_gpu->kernel_more_cta_left(kernel) &&
        //            (m_core[core]->get_n_active_cta() <
        //            m_config->max_cta(*kernel)) ) {
        //can_issue_1block(*kernel)�ж��Ƿ���Է���һ���߳̿飬������Է���һ���߳̿飬�򷵻�true��
        //���򷵻�false��
        m_core[core]->can_issue_1block(*kernel)) {
      //��kernel����SM�ϡ�
      m_core[core]->issue_block2core(*kernel);
      num_blocks_issued++;
      m_cta_issue_next_core = core;
      break;
    }
  }
  return num_blocks_issued;
}

void simt_core_cluster::cache_flush() {
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; i++)
    m_core[i]->cache_flush();
}

void simt_core_cluster::cache_invalidate() {
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; i++)
    m_core[i]->cache_invalidate();
}

bool simt_core_cluster::icnt_injection_buffer_full(unsigned size, bool write) {
  unsigned request_size = size;
  if (!write) request_size = READ_PACKET_SIZE;
  //icnt_has_buffer���жϻ��������Ƿ��п��е����뻺�������������deviceID���豸�µ����ݰ���
  return !::icnt_has_buffer(m_cluster_id, request_size);
}

/*
SIMT Core��Packetsע�뻥������Ľӿڡ�
*/
void simt_core_cluster::icnt_inject_request_packet(class mem_fetch *mf) {
  // stats
  if (mf->get_is_write())
    m_stats->made_write_mfs++;
  else
    m_stats->made_read_mfs++;
  switch (mf->get_access_type()) {
    case CONST_ACC_R:
      m_stats->gpgpu_n_mem_const++;
      break;
    case TEXTURE_ACC_R:
      m_stats->gpgpu_n_mem_texture++;
      break;
    case GLOBAL_ACC_R:
      m_stats->gpgpu_n_mem_read_global++;
      break;
    // case GLOBAL_ACC_R: m_stats->gpgpu_n_mem_read_global++;
    // printf("read_global%d\n",m_stats->gpgpu_n_mem_read_global); break;
    case GLOBAL_ACC_W:
      m_stats->gpgpu_n_mem_write_global++;
      break;
    case LOCAL_ACC_R:
      m_stats->gpgpu_n_mem_read_local++;
      break;
    case LOCAL_ACC_W:
      m_stats->gpgpu_n_mem_write_local++;
      break;
    case INST_ACC_R:
      m_stats->gpgpu_n_mem_read_inst++;
      break;
    //��V100�У�L1 cache��m_write_policyΪWRITE_THROUGH��ʵ����L1_WRBK_ACCҲ�����õ���
    case L1_WRBK_ACC:
      m_stats->gpgpu_n_mem_write_global++;
      break;
    //��V100�У���L2 cacheд������ʱ����ȡlazy_fetch_on_read���ԣ����ҵ�һ��cache block
    //���ʱ��������cache block�Ǳ�MODIFIED������Ҫ�����cache blockд�ص���һ���洢��
    //��˻����L2_WRBK_ACC���ʣ�������ʾ���Ϊ��д�ر������MODIFIED cache block��
    case L2_WRBK_ACC:
      m_stats->gpgpu_n_mem_l2_writeback++;
      break;
    //L1_WR_ALLOC_R/L2_WR_ALLOC_R��V100��������ʱ�ò�����
    case L1_WR_ALLOC_R:
      m_stats->gpgpu_n_mem_l1_write_allocate++;
      break;
    //L1_WR_ALLOC_R/L2_WR_ALLOC_R��V100��������ʱ�ò�����
    case L2_WR_ALLOC_R:
      m_stats->gpgpu_n_mem_l2_write_allocate++;
      break;
    default:
      assert(0);
  }

  // The packet size varies depending on the type of request:
  // - For write request and atomic request, the packet contains the data
  // - For read request (i.e. not write nor atomic), the packet only has control
  // metadata
  unsigned int packet_size = mf->size();
  if (!mf->get_is_write() && !mf->isatomic()) {
    packet_size = mf->get_ctrl_size();
  }
  m_stats->m_outgoing_traffic_stats->record_traffic(mf, packet_size);
  unsigned destination = mf->get_sub_partition_id();
  mf->set_status(IN_ICNT_TO_MEM,
                 m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
  if (!mf->get_is_write() && !mf->isatomic())
    //icnt_push�����ݰ�ѹ�뻥���������뻺������
    ::icnt_push(m_cluster_id, m_config->mem2device(destination), (void *)mf,
                mf->get_ctrl_size());
  else
    //icnt_push�����ݰ�ѹ�뻥���������뻺������
    ::icnt_push(m_cluster_id, m_config->mem2device(destination), (void *)mf,
                mf->size());
}

/*
simt_core_cluster::icnt_cycle()�������ڴ�����ӻ�����������simt���ļ�Ⱥ����ӦFIFO��������FIFO��
�����󣬲������Ƿ��͵���Ӧ�ں˵�ָ����LDST��Ԫ��

ÿ��SIMT Core��Ⱥ����һ����ӦFIFO�����ڱ���ӻ������緢�������ݰ������ݰ�������SIMT Core��ָ�
�棨�������Ϊָ���ȡδ�����ṩ������ڴ���Ӧ�������ڴ���ˮ�ߣ�memory pipeline��LDST ��Ԫ��������
�����Ƚ��ȳ���ʽ�ó������SIMT Core�޷�����FIFOͷ�������ݰ�������ӦFIFO��ֹͣ��Ϊ����LDST��Ԫ������
�ڴ�����ÿ��SIMT Core�����Լ���ע��˿ڽ��뻥�����硣���ǣ�ע��˿ڻ�������SIMT Core��Ⱥ����SIMT 
Core����
*/
void simt_core_cluster::icnt_cycle() {
  //�����ӦFIFO�ǿգ��ſ��Խ�������ICNT�����ݰ��������m_response_fifo��ָSIMT Core��Ⱥ����ӦFIFO��
  if (!m_response_fifo.empty()) {
    //����ӦFIFOͷ���Ƴ�һ�����ݰ� mf��m_response_fifo������Ϊ��
    //    std::list<mem_fetch *> m_response_fifo;
    //mem_fetch������һ��ģ���ڴ������ͨ�Žṹ��������һ���ڴ��������Ϊ�������m_response_fifo��
    //ָSIMT Core��Ⱥ����ӦFIFO��
    mem_fetch *mf = m_response_fifo.front();
    //mf->get_sid()��ȡ�ڴ��������Դ��SIMT Core��ID��m_config��SIMT Core��Ⱥ�е�Shader Core����
    //�á�m_config->sid_to_cid(sid)������SM��ID����ȡSIMT Core��Ⱥ��ID����cidΪSIMT Core��Ⱥ��ID��
    unsigned cid = m_config->sid_to_cid(mf->get_sid());
    //mf->get_access_type()���ضԴ洢�����еķô�����mem_access_type��mem_access_type��������ʱ��
    //ģ�����жԲ�ͬ���͵Ĵ洢�����в�ͬ�ķô����ͣ�
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
    //    MA_TUP(L2_WR_ALLOC_R),       L2����write-allocate
    //    MA_TUP(NUM_MEM_ACCESS_TYPE), �洢�����ʵ���������
    if (mf->get_access_type() == INST_ACC_R) {
      //���mf->get_access_type() = ��ָ����������ָ��Ԥȡ��Ӧ��
      // instruction fetch response.
      //m_coreΪSIMT Core��Ⱥ���������SIMT Core��һ����άshader_core_ctx���󣬵�һά����ȺID��
      //�ڶ�ά����SIMT Core ID��fetch_unit_response_buffer_full()����Ԥȡ��Ԫ��Ӧbuffer�Ƿ�������
      //�����������һֱ�������������ѭ��ʼ��ִ�С�
      if (!m_core[cid]->fetch_unit_response_buffer_full()) {
        //��ָ��Ԥȡ����ӦFIFO����һ�����ݰ��������m_response_fifo��ָSIMT Core��Ⱥ����ӦFIFO��
        m_response_fifo.pop_front();
        //m_core[cid]ָ���SIMT Core��Ⱥ�������Ԥȡ��ָ�����ݰ�����mf�ŵ�cid��ʶ��SIMT Core��Ⱥ
        //��L1 I-Cache����ע�⣬����m_coreΪSIMT Core��Ⱥ���������SM��һ����άshader_core_ctx��
        //�󣬵�һά����ȺID���ڶ�ά����SIMT Core ID���䶨��Ϊ��
        //    shader_core_ctx **m_core;
        //��TITAN V�������У�һ��SIMT Core��Ⱥ���������SM������������SM��ʵ�뻥�����繲��һ������
        //�˿ڣ��Ҵ���δ��뿴������������SM������һ��ָ����LD/ST��Ԫ����֪���Բ��ԣ��������ǻ�
        //�����õ��Ķ��ǵ�SM�����ã��������ﲻ�ع�����ᡣ
        m_core[cid]->accept_fetch_response(mf);
      }
    } else {
      //���mf->get_access_type() �� ��ָ����������������ȡ��Ӧ��
      // data response.
      //ldst_unit_response_buffer_full()����LDST��Ԫ��Ӧbuffer�Ƿ�������
      //����LDST��Ԫ��Ӧbuffer�Ƿ�������LD/ST��Ԫ����ӦFIFO�е����ݰ��� >= GPU���õ���Ӧ�����е���
      //����Ӧ������������Ҫע����ǣ�LD/ST��ԪҲ��һ��m_response_fifo����m_response_fifo.size()
      //��ȡ���Ǹ�fifo�Ѿ��洢��mf��Ŀ��m_config->ldst_unit_response_queue_size�������õĸ�fifo��
      //���������һ��m_response_fifo.size()�������õ�����������ͻ᷵��True����ʾ��fifo������
      if (!m_core[cid]->ldst_unit_response_buffer_full()) {
        //������Ԥȡ����ӦFIFO����һ�����ݰ��������m_response_fifo��ָSIMT Core��Ⱥ����ӦFIFO��
        m_response_fifo.pop_front();
        //ͳ��Memory Latency Statistics.
        m_memory_stats->memlatstat_read_done(mf);
        //m_core[cid]ָ���SIMT Core��Ⱥ�������Ԥȡ��data���ݰ�����ע�⣬����m_coreΪSIMT Core��
        //Ⱥ���������SM��һ����άshader_core_ctx���󣬵�һά����ȺID���ڶ�ά����SIMT Core ID��
        //�䶨��Ϊ��
        //    shader_core_ctx **m_core;
        //��TITAN V�������У�һ��SIMT Core��Ⱥ���������SM������������SM��ʵ�뻥�����繲��һ������
        //�˿ڣ��Ҵ���δ��뿴������������SM������һ��ָ����LD/ST��Ԫ����֪���Բ��ԣ��������ǻ�
        //�����õ��Ķ��ǵ�SM�����ã��������ﲻ�ع�����ᡣ
        m_core[cid]->accept_ldst_unit_response(mf);
      }
    }
  }
  //m_config->n_simt_ejection_buffer_size�ǵ����������е����ݰ�������ʵ�������Ϊ�����SIMT Core��Ⱥ
  //����ӦFIFO����������������ӦFIFO��С < �����������е����ݰ������򵯳����������Լ�����SIMT Core��Ⱥ
  //����ӦFIFO�ﵯ����һ�����ݰ�������������ָ���ǣ�[��������->����������->SIMT Core��Ⱥ]���м�ڵ㡣��
  //��m_response_fifo.size()��ָm_response_fifo�е����ݰ���������m_response_fifoΪ��ʱ��size=0��
  if (m_response_fifo.size() < m_config->n_simt_ejection_buffer_size) {
    //����mem_fetch *mfָ���ǻ������������SIMT Core��Ⱥ����ӦFIFO�ﵯ������һ�����ݰ���
    //icnt_pop�����ݰ������������������������
    mem_fetch *mf = (mem_fetch *)::icnt_pop(m_cluster_id);
    //���û��������˵����������ĵ������������ɻ�������->SIMT Core��Ⱥ��Ϊ�գ���������û���µ����ݰ�Ҫ��
    //SIMT Core��Ⱥ���䡣
    if (!mf) return;
    assert(mf->get_tpc() == m_cluster_id);
    assert(mf->get_type() == READ_REPLY || mf->get_type() == WRITE_ACK);

    // The packet size varies depending on the type of request:
    // - For read request and atomic request, the packet contains the data
    // - For write-ack, the packet only has control metadata
    //���ݰ���С���������Ͷ��죺
    // - ���ڶ�ȡ�����ԭ���������ݰ��������ݣ�
    // - ����дȷ�ϣ����ݰ�ֻ�п���Ԫ���ݡ�
    unsigned int packet_size =
        (mf->get_is_write()) ? mf->get_ctrl_size() : mf->size();
    m_stats->m_incoming_traffic_stats->record_traffic(mf, packet_size);
    mf->set_status(IN_CLUSTER_TO_SHADER_QUEUE,
                   m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
    // m_memory_stats->memlatstat_read_done(mf,m_shader_config->max_warps_per_shader);
    //SIMT Core��Ⱥ����ӦFIFO�����ݰ�mf���뵽FIFO�ײ��������ȳ�˳��
    m_response_fifo.push_back(mf);
    m_stats->n_mem_to_simt[m_cluster_id] += mf->get_num_flits(false);
  }
}

void simt_core_cluster::get_pdom_stack_top_info(unsigned sid, unsigned tid,
                                                unsigned *pc,
                                                unsigned *rpc) const {
  unsigned cid = m_config->sid_to_cid(sid);
  m_core[cid]->get_pdom_stack_top_info(tid, pc, rpc);
}

void simt_core_cluster::display_pipeline(unsigned sid, FILE *fout,
                                         int print_mem, int mask) {
  m_core[m_config->sid_to_cid(sid)]->display_pipeline(fout, print_mem, mask);

  fprintf(fout, "\n");
  fprintf(fout, "Cluster %u pipeline state\n", m_cluster_id);
  fprintf(fout, "Response FIFO (occupancy = %zu):\n", m_response_fifo.size());
  for (std::list<mem_fetch *>::const_iterator i = m_response_fifo.begin();
       i != m_response_fifo.end(); i++) {
    const mem_fetch *mf = *i;
    mf->print(fout);
  }
}

void simt_core_cluster::print_cache_stats(FILE *fp, unsigned &dl1_accesses,
                                          unsigned &dl1_misses) const {
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; ++i) {
    m_core[i]->print_cache_stats(fp, dl1_accesses, dl1_misses);
  }
}

void simt_core_cluster::get_icnt_stats(long &n_simt_to_mem,
                                       long &n_mem_to_simt) const {
  long simt_to_mem = 0;
  long mem_to_simt = 0;
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; ++i) {
    m_core[i]->get_icnt_power_stats(simt_to_mem, mem_to_simt);
  }
  n_simt_to_mem = simt_to_mem;
  n_mem_to_simt = mem_to_simt;
}

void simt_core_cluster::get_cache_stats(cache_stats &cs) const {
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; ++i) {
    m_core[i]->get_cache_stats(cs);
  }
}

void simt_core_cluster::get_L1I_sub_stats(struct cache_sub_stats &css) const {
  struct cache_sub_stats temp_css;
  struct cache_sub_stats total_css;
  temp_css.clear();
  total_css.clear();
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; ++i) {
    m_core[i]->get_L1I_sub_stats(temp_css);
    total_css += temp_css;
  }
  css = total_css;
}
void simt_core_cluster::get_L1D_sub_stats(struct cache_sub_stats &css) const {
  struct cache_sub_stats temp_css;
  struct cache_sub_stats total_css;
  temp_css.clear();
  total_css.clear();
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; ++i) {
    m_core[i]->get_L1D_sub_stats(temp_css);
    total_css += temp_css;
  }
  css = total_css;
}
void simt_core_cluster::get_L1C_sub_stats(struct cache_sub_stats &css) const {
  struct cache_sub_stats temp_css;
  struct cache_sub_stats total_css;
  temp_css.clear();
  total_css.clear();
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; ++i) {
    m_core[i]->get_L1C_sub_stats(temp_css);
    total_css += temp_css;
  }
  css = total_css;
}
void simt_core_cluster::get_L1T_sub_stats(struct cache_sub_stats &css) const {
  struct cache_sub_stats temp_css;
  struct cache_sub_stats total_css;
  temp_css.clear();
  total_css.clear();
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; ++i) {
    m_core[i]->get_L1T_sub_stats(temp_css);
    total_css += temp_css;
  }
  css = total_css;
}

void exec_shader_core_ctx::checkExecutionStatusAndUpdate(warp_inst_t &inst,
                                                         unsigned t,
                                                         unsigned tid) {
  if (inst.isatomic()) m_warp[inst.warp_id()]->inc_n_atomic();
  if (inst.space.is_local() && (inst.is_load() || inst.is_store())) {
    new_addr_type localaddrs[MAX_ACCESSES_PER_INSN_PER_THREAD];
    unsigned num_addrs;
    num_addrs = translate_local_memaddr(
        inst.get_addr(t), tid,
        m_config->n_simt_clusters * m_config->n_simt_cores_per_cluster,
        inst.data_size, (new_addr_type *)localaddrs);
    inst.set_addr(t, (new_addr_type *)localaddrs, num_addrs);
  }
  if (ptx_thread_done(tid)) {
    m_warp[inst.warp_id()]->set_completed(t);
    m_warp[inst.warp_id()]->ibuffer_flush();
  }

  // PC-Histogram Update
  unsigned warp_id = inst.warp_id();
  unsigned pc = inst.pc;
  for (unsigned t = 0; t < m_config->warp_size; t++) {
    if (inst.active(t)) {
      int tid = warp_id * m_config->warp_size + t;
      cflog_update_thread_pc(m_sid, tid, pc);
    }
  }
}
