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
shader.cc是SIMT Core的时序模型。它调用cudu-sim对一个特定的线程进行功能模拟，cuda-sim将返回该线程
的性能敏感的信息。
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
获取一条指令中需写回的寄存器编号，以列表方式返回。
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
为当前SM创建所有的warp，warp的数量是m_config->max_warps_per_shader确定。
*/
void exec_shader_core_ctx::create_shd_warp() {
  m_warp.resize(m_config->max_warps_per_shader);
  for (unsigned k = 0; k < m_config->max_warps_per_shader; ++k) {
    m_warp[k] = new shd_warp_t(this, m_config->warp_size);
  }
}

/*
为当前SM创建所有的流水线单元。
*/
void shader_core_ctx::create_front_pipeline() {
  // pipeline_stages is the sum of normal pipeline stages and specialized_unit
  // stages * 2 (for ID and EX)
  //全部的流水线阶段数等于普通流水线阶段数加上专用单元的流水线阶段数乘以2（ID和EX）。
  unsigned total_pipeline_stages =
      N_PIPELINE_STAGES + m_config->m_specialized_unit.size() * 2;
  m_pipeline_reg.reserve(total_pipeline_stages);
  //流水线阶段的宽度配置在-gpgpu_pipeline_widths中设置：
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
  //在V100中配置为：-gpgpu_pipeline_widths 4,4,4,4,4,4,4,4,4,4,8,4,4
  //这里流水线的宽度其实就是流水线寄存器内部能够存储多少条指令，这个数量是和单元的数目挂
  //钩的。
  for (int j = 0; j < N_PIPELINE_STAGES; j++) {
    //共有N_PIPELINE_STAGES个流水线阶段，因此设置N_PIPELINE_STAGES个流水线寄存器。
    m_pipeline_reg.push_back(
        register_set(m_config->pipe_widths[j], pipeline_stage_name_decode[j]));
  }
  for (unsigned j = 0; j < m_config->m_specialized_unit.size(); j++) {
    //这里为专用单元的流水线阶段设置ID_OC流水线寄存器，但是在V100配置中，没有配置专用单
    //元。
    m_pipeline_reg.push_back(
        register_set(m_config->m_specialized_unit[j].id_oc_spec_reg_width,
                     m_config->m_specialized_unit[j].name));
    m_config->m_specialized_unit[j].ID_OC_SPEC_ID = m_pipeline_reg.size() - 1;
    m_specilized_dispatch_reg.push_back(
        &m_pipeline_reg[m_pipeline_reg.size() - 1]);
  }
  for (unsigned j = 0; j < m_config->m_specialized_unit.size(); j++) {
    //这里为专用单元的流水线阶段设置OC_EX流水线寄存器，但是在V100配置中，没有配置专用单
    //元。
    m_pipeline_reg.push_back(
        register_set(m_config->m_specialized_unit[j].oc_ex_spec_reg_width,
                     m_config->m_specialized_unit[j].name));
    m_config->m_specialized_unit[j].OC_EX_SPEC_ID = m_pipeline_reg.size() - 1;
  }
  //在subcore模式下，每个warp调度器在寄存器集合中有一个具体的寄存器可供使用，这个寄
  //存器由调度器的m_id索引。
  if (m_config->sub_core_model) {
    // in subcore model, each scheduler should has its own issue register, so
    // ensure num scheduler = reg width
    //这里说明了每个SM内的warp调度器的个数与所有的寄存器集合的宽度相同。
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

  //线程状态集合。
  m_threadState = (thread_ctx_t *)calloc(sizeof(thread_ctx_t),
                                         m_config->n_thread_per_shader);
  //未完成的线程数（当此Shader Core上的所有线程都完成时，==0）。
  m_not_completed = 0;
  //m_active_threads是Shader Core上活跃线程的位图。
  m_active_threads.reset();
  //m_n_active_cta是Shader Core上活跃的线程块的数量。
  m_n_active_cta = 0;
  //m_cta_status是Shader Core内的CTA的状态，MAX_CTA_PER_SHADER是每个Shader Core内的
  //最大可并发CTA个数。m_cta_status[i]里保存了第i个CTA中包含的活跃线程总数量，该数量<=
  //CTA的总线程数量。
  for (unsigned i = 0; i < MAX_CTA_PER_SHADER; i++) m_cta_status[i] = 0;
  for (unsigned i = 0; i < m_config->n_thread_per_shader; i++) {
    m_thread[i] = NULL;
    m_threadState[i].m_cta_id = -1;
    m_threadState[i].m_active = false;
  }

  // m_icnt = new shader_memory_interface(this,cluster);
  //在V100配置中，m_config->gpgpu_perfect_mem为false。
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
Shader Core内创建warp调度器。单个Sahder Core内的warp调度器的个数由gpgpu_num_sched_per_core配置参
数决定，Volta架构每核心有4个warp调度器。
*/
void shader_core_ctx::create_schedulers() {
  //创建一个记分牌。一个Shader Core有一个记分牌。
  m_scoreboard = new Scoreboard(m_sid, m_config->max_warps_per_shader, m_gpu);

  // scedulers
  // must currently occur after all inputs have been initialized.
  std::string sched_config = m_config->gpgpu_scheduler_string;
  const concrete_scheduler scheduler =
      sched_config.find("lrr") != std::string::npos ? CONCRETE_SCHEDULER_LRR
          //CONCRETE_SCHEDULER_LRR 是模拟器中的一个调度器（scheduler）选项之一。LRR 表示 “Least 
          //Recently Reused”，意为"最近最少使用"。该调度器算法基于最近最少使用原则，用于管理模拟器
          //中的请求调度。在模拟器中，存在多个请求，如指令和数据访问请求，它们需要按照一定的顺序进行
          //处理和执行。CONCRETE_SCHEDULER_LRR 算法根据请求最近的使用情况来确定下一个要处理的请求，
          //优先选择最近最少使用的请求。通过使用最近最少使用算法，CONCRETE_SCHEDULER_LRR 调度器可
          //以更好地利用缓存和内存的访问模式，以提高整体性能。通过保留最近最少使用的请求，可以减少在
          //请求处理过程中的缓存冲突和竞争，从而降低延迟并提高效率。
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

  //单个Sahder Core内的warp调度器的个数由gpgpu_num_sched_per_core配置参数决定，Volta架构每核心有
  //4个warp调度器。
  for (unsigned i = 0; i < m_config->gpgpu_num_sched_per_core; i++) {
    switch (scheduler) {
      //创建调度器，Volta架构中调度器的类型为CONCRETE_SCHEDULER_LRR，最近最少使用。
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

  //这里m_warp是划分到当前SM的所有warp集合，其定义为：
  //    std::vector<shd_warp_t *> *m_warp;
  //m_warp[i]是划分到当前SM的第i个warp，这段代码其实是将m_warp中的所有warp均分给每个调度器，这
  //样每个调度器就可以对划分给自己的warp进行调度了。在Volta架构上，每核心有4个warp调度器，划分的
  //策略是，warp 0->调度器0，warp 1->调度器1，warp 2->调度器2，warp 3->调度器3，warp 4->调度
  //器0，以此类推。在每个调度器内部有一个专门存储各自所划分到的warp的列表，即m_supervised_warps，
  //每个调度器在下面这段代码里将划分为自己的warp加入到自己的m_supervised_warps中。该列表定义为：
  //    std::vector<shd_warp_t *> m_supervised_warps;
  //m_supervisored_twarps列表是此调度程序应该在其间进行仲裁的所有warps。这在存在多个warp调度器
  //的系统中非常有用。在单个调度器系统中，这只是分配给该核心的所有warp（单个调度器不需要划分）。
  for (unsigned i = 0; i < m_warp.size(); i++) {
    // distribute i's evenly though schedulers;
    //m_supervisored_twarps列表是此调度程序应该在其间进行仲裁的所有warps。这在存在多个warp调度
    //器的系统中非常有用。在单个调度器系统中，这只是分配给该核心的所有warp。
    schedulers[i % m_config->gpgpu_num_sched_per_core]->add_supervised_warp_id(
        i);
  }
  
  //done_adding_supervised_warps()函数的定义为：
  //    virtual void done_adding_supervised_warps() {
  //      m_last_supervised_issued = m_supervised_warps.end();
  //    }
  //这里其实就是对每个调度器的m_last_supervised_issued进行初始化，m_last_supervised_issued是
  //指代上一次调度的warp，在这里初始化为m_supervised_warps.end()，即m_supervised_warps的最后
  //一个m_supervised_warps中的warp。
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
    //gpgpu_operand_collector_num_in_ports_sp是SP单元接入操作数收集器的输入端口数量。在前面
    //的warp调度器代码里单个Sahder Core内的warp调度器的个数由gpgpu_num_sched_per_core配置参
    //数决定，Volta架构每核心有4个warp调度器。每个调度器的创建代码：
    //     schedulers.push_back(new lrr_scheduler(
    //             m_stats, this, m_scoreboard, m_simt_stack, &m_warp,
    //             &m_pipeline_reg[ID_OC_SP], &m_pipeline_reg[ID_OC_DP],
    //             &m_pipeline_reg[ID_OC_SFU], &m_pipeline_reg[ID_OC_INT],
    //             &m_pipeline_reg[ID_OC_TENSOR_CORE], m_specilized_dispatch_reg,
    //             &m_pipeline_reg[ID_OC_MEM], i));
    //在发射过程中，warp调度器将可发射的指令按照其指令类型分发给不同的单元，这些单元包括SP/DP/
    //SFU/INT/TENSOR_CORE/MEM，在发射过程完成后，需要针对指令通过操作数收集器将指令所需的操作
    //数全部收集齐。对于一个SM，对应于一个操作数收集器，调度器的发射过程将指令放入：
    //    m_pipeline_reg[ID_OC_SP]、m_pipeline_reg[ID_OC_DP]、m_pipeline_reg[ID_OC_SFU]、
    //    m_pipeline_reg[ID_OC_INT]、m_pipeline_reg[ID_OC_TENSOR_CORE]、
    //    m_pipeline_reg[ID_OC_MEM]
    //等寄存器集合中，用以操作数收集器来收集操作数。
    for (unsigned i = 0; i < m_config->gpgpu_operand_collector_num_in_ports_sp;
         i++) {
      //m_pipeline_reg的定义：std::vector<register_set> m_pipeline_reg;
      in_ports.push_back(&m_pipeline_reg[ID_OC_SP]);
      out_ports.push_back(&m_pipeline_reg[OC_EX_SP]);
      cu_sets.push_back((unsigned)SP_CUS);
      cu_sets.push_back((unsigned)GEN_CUS);
      m_operand_collector.add_port(in_ports, out_ports, cu_sets);
      in_ports.clear(), out_ports.clear(), cu_sets.clear();
    }
    //gpgpu_operand_collector_num_in_ports_dp是DP单元接入操作数收集器的输入端口数量。
    for (unsigned i = 0; i < m_config->gpgpu_operand_collector_num_in_ports_dp;
         i++) {
      in_ports.push_back(&m_pipeline_reg[ID_OC_DP]);
      out_ports.push_back(&m_pipeline_reg[OC_EX_DP]);
      cu_sets.push_back((unsigned)DP_CUS);
      cu_sets.push_back((unsigned)GEN_CUS);
      m_operand_collector.add_port(in_ports, out_ports, cu_sets);
      in_ports.clear(), out_ports.clear(), cu_sets.clear();
    }
    //gpgpu_operand_collector_num_in_ports_sfu是SFU单元接入操作数收集器的输入端口数量。
    for (unsigned i = 0; i < m_config->gpgpu_operand_collector_num_in_ports_sfu;
         i++) {
      in_ports.push_back(&m_pipeline_reg[ID_OC_SFU]);
      out_ports.push_back(&m_pipeline_reg[OC_EX_SFU]);
      cu_sets.push_back((unsigned)SFU_CUS);
      cu_sets.push_back((unsigned)GEN_CUS);
      m_operand_collector.add_port(in_ports, out_ports, cu_sets);
      in_ports.clear(), out_ports.clear(), cu_sets.clear();
    }
    //gpgpu_operand_collector_num_in_ports_tensor_core是TC单元接入操作数收集器的输入端口数量。
    for (unsigned i = 0;
         i < m_config->gpgpu_operand_collector_num_in_ports_tensor_core; i++) {
      in_ports.push_back(&m_pipeline_reg[ID_OC_TENSOR_CORE]);
      out_ports.push_back(&m_pipeline_reg[OC_EX_TENSOR_CORE]);
      cu_sets.push_back((unsigned)TENSOR_CORE_CUS);
      cu_sets.push_back((unsigned)GEN_CUS);
      m_operand_collector.add_port(in_ports, out_ports, cu_sets);
      in_ports.clear(), out_ports.clear(), cu_sets.clear();
    }
    //gpgpu_operand_collector_num_in_ports_mem是MEM单元接入操作数收集器的输入端口数量。
    for (unsigned i = 0; i < m_config->gpgpu_operand_collector_num_in_ports_mem;
         i++) {
      in_ports.push_back(&m_pipeline_reg[ID_OC_MEM]);
      out_ports.push_back(&m_pipeline_reg[OC_EX_MEM]);
      cu_sets.push_back((unsigned)MEM_CUS);
      cu_sets.push_back((unsigned)GEN_CUS);
      m_operand_collector.add_port(in_ports, out_ports, cu_sets);
      in_ports.clear(), out_ports.clear(), cu_sets.clear();
    }
    //gpgpu_operand_collector_num_in_ports_int是INT单元接入操作数收集器的输入端口数量。
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

  //执行操作数收集器的初始化。
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
  //结果总线共有m_config->pipe_widths[EX_WB]条。
  //流水线阶段的宽度配置在-gpgpu_pipeline_widths中设置：
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
  //在V100中配置为：-gpgpu_pipeline_widths 4,4,4,4,4,4,4,4,4,4,8,4,4
  //结果总线的宽度是8，即m_config->pipe_widths[EX_WB] = 8。
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
对第cta_id个CTA中，从start_thread到end_thread个线程所属的所有warp进行初始化。
*/
void shader_core_ctx::init_warps(unsigned cta_id, unsigned start_thread,
                                 unsigned end_thread, unsigned ctaid,
                                 int cta_size, kernel_info_t &kernel) {
  //功能模拟过程中，用warp的起始PC值（用该warp的首个线程m_thread[warpId*m_warp_size]->get_pc()获取）
  //线程和其线程掩码用于启动SIMT堆栈。每个warp都有一个单独的SIMT堆栈。
  address_type start_pc = next_pc(start_thread);
  unsigned kernel_id = kernel.get_uid();
  if (m_config->model == POST_DOMINATOR) {
    unsigned start_warp = start_thread / m_config->warp_size;
    unsigned warp_per_cta = cta_size / m_config->warp_size;
    unsigned end_warp = end_thread / m_config->warp_size +
                        ((end_thread % m_config->warp_size) ? 1 : 0);
    //从start_warp开始到end_warp循环。
    for (unsigned i = start_warp; i < end_warp; ++i) {
      //n_active是所有warp中的活跃线程的个数。
      unsigned n_active = 0;
      simt_mask_t active_threads;
      //局部线程ID，从0到31循环。
      for (unsigned t = 0; t < m_config->warp_size; t++) {
        //硬件上的线程ID，即全局ID=warp_id*32+局部ID。
        unsigned hwtid = i * m_config->warp_size + t;
        if (hwtid < end_thread) {
          n_active++;
          assert(!m_active_threads.test(hwtid));
          //设置全局活跃线程的位图。
          m_active_threads.set(hwtid);
          //设置局部活跃线程的位图。
          active_threads.set(t);
        }
      }
      //功能模拟过程中，用warp的起始PC值（用该warp的首个线程m_thread[warpId*m_warp_size]->get_pc()获
      //取）线程和其线程掩码用于启动SIMT堆栈。每个warp都有一个单独的SIMT堆栈。
      m_simt_stack[i]->launch(start_pc, active_threads);

      //在V100配置中，m_gpu->resume_option被默认配置为0。
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

      //对全局第i个warp，m_warp[i]进行初始化。
      m_warp[i]->init(start_pc, cta_id, i, active_threads, m_dynamic_warp_id,
                      kernel.get_streamID());
      //下一个动态warp的ID，每有一个warp变成活跃状态就增1。
	  ++m_dynamic_warp_id;
      //m_not_completed的值实际上是返回的是已经初始化的所有warp（即整个CTA）中尚未完成的线程数。
	  m_not_completed += n_active;
      //活跃warp的数量增1。
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
依据PC值，获取指令。
*/
const warp_inst_t *exec_shader_core_ctx::get_next_inst(unsigned warp_id,
                                                       address_type pc) {
  // read the inst from the functional model
  //ptx_fetch_inst(pc)的功能是依据PC值，获取指令。
  return m_gpu->gpgpu_ctx->ptx_fetch_inst(pc);
}

/*
获取warp_id对应的SIMT堆栈顶部的PC值和RPC值。
*/
void exec_shader_core_ctx::get_pdom_stack_top_info(unsigned warp_id,
                                                   const warp_inst_t *pI,
                                                   unsigned *pc,
                                                   unsigned *rpc) {
  //SIMT堆栈是一个warp有一个。m_simt_stack是每个warp有一个。
  //SIMT堆栈的get_pdom_stack_top_info()函数的功能是获取SIMT堆栈顶部的PC值和RPC值。
  m_simt_stack[warp_id]->get_pdom_stack_top_info(pc, rpc);
}

const active_mask_t &exec_shader_core_ctx::get_active_mask(
    unsigned warp_id, const warp_inst_t *pI) {
  return m_simt_stack[warp_id]->get_active_mask();
}

/*
Shader Core中的解码阶段与m_inst_fetch_buffer结构密切相关，后者充当获取和解码阶段之间的通信管道。解码
阶段请参见以下手册说明：
    “解码阶段只需检查shader_core_ctx::m_inst_fetch_buffer，并开始将解码的指令（当前配置每个周期最多
    解码两条指令）存储在指令缓冲区条目（m_ibuffer，shd_warp_t::ibuffer_entry的对象）中，该条目对应于
    shader-core_ctx::m_inst_fetch_bbuffer中的warp。”
*/
void shader_core_ctx::decode() {
  //m_inst_fetch_buffer的定义为：
  //    ifetch_buffer_t m_inst_fetch_buffer;
  //指令获取缓冲区。指令获取缓冲区（ifetch_Buffer_t）对指令缓存（I-cache）和SIMT Core之间的接口进行
  //建模。它有一个成员m_valid，用于指示缓冲区是否有有效的指令。它还将指令的warp id记录在m_warp_id中。
  //因此，当m_valid为0，即指示缓冲区暂时没有有效的指令；当m_valid为1，即指示缓冲区已经有有效的指令。
  if (m_inst_fetch_buffer.m_valid) {
    // decode 1 or 2 instructions and place them into ibuffer.
    //解码1~2条指令，并把它们放到I-Buffer中。
    //获取m_inst_fetch_buffer中存储的指令的PC值。
    address_type pc = m_inst_fetch_buffer.m_pc;
    //get_next_inst()依据PC值，获取指令。
    const warp_inst_t *pI1 = get_next_inst(m_inst_fetch_buffer.m_warp_id, pc);
    //将一条新指令存入I-Bufer。I-Buffer有两个槽，pI1加入槽0，pI2加入槽1。
    m_warp[m_inst_fetch_buffer.m_warp_id]->ibuffer_fill(0, pI1);
    //增加在流水线中执行的指令数。
    m_warp[m_inst_fetch_buffer.m_warp_id]->inc_inst_in_pipeline();
    if (pI1) {
      //m_num_decoded_insn是SM上解码后的指令数，pI1指令有效的话，增加1。
      m_stats->m_num_decoded_insn[m_sid]++;
      if ((pI1->oprnd_type == INT_OP) ||
          (pI1->oprnd_type == UN_OP)) {  // these counters get added up in mcPat
                                         // to compute scheduler power
        m_stats->m_num_INTdecoded_insn[m_sid]++;
      } else if (pI1->oprnd_type == FP_OP) {
        m_stats->m_num_FPdecoded_insn[m_sid]++;
      }
      //获取下一条指令。一次decode解码放进ibuffer两条指令，所以这里再获取一条指令。
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
    //这里需要说明下m_inst_fetch_buffer与m_ibuffer的区别：m_inst_fetch_buffer变量在获取（指令缓存
    //访问）和解码阶段之间充当流水线寄存器；而每个shd_warp_t都有一组m_ibuffer的I-Buffer条目(ibuffer
    //entry)，持有可配置的指令数量（一个周期内允许获取的最大指令）。解码完毕后将m_inst_fetch_buffer
    //设置为False，以便于下一拍继续fetch操作。
    m_inst_fetch_buffer.m_valid = false;
  }
}

/*
SIMT Core的取指令时钟周期。fetch()函数生成指令内存请求，并从一级指令缓存中收集提取的指令。提取的指令
被放入指令提取缓冲区。以下的逻辑为：
  * 如果m_inst_fetch_buffer为空（无效）：
    ** 如果指令缓存中有指令（已准备就绪）：
      *** 将指令放入m_inst_fetch_buffer。
    ** 否则，即如果缓存中没有就绪指令：
      *** 遍历所有硬件warp（2048/32）。如果warp正在运行，没有等待instruction cache missing，并且
          其指令缓冲区为空：则生成内存获取请求。
  * 运行m_L1I->cycle()函数。

这里需要说明下m_inst_fetch_buffer与m_ibuffer的区别：m_inst_fetch_buffer变量在获取（指令缓存访问）
和解码阶段之间充当流水线寄存器；而每个shd_warp_t都有一组m_ibuffer的I-Buffer条目(ibuffer_entry)，持
有可配置的指令数量（一个周期内允许获取的最大指令）。首先如果m_inst_fetch_buffer为空，则要判断L1指令缓
存是否有就绪的指令，就绪指令存在的话要从L1指令缓存中获取指令；而如果L1指令缓存中没有指令，这时候要判断
一下是否因为warp运行完毕导致L1指令缓存中没有指令了，如果是warp正在运行，且没有等待挂起的L1缓存未命中的
挂起状态，且warp的指令缓冲区为空，这时候就说明要取下一条PC值的指令。
*/
void shader_core_ctx::fetch() {
  //m_inst_fetch_buffer的定义为：
  //    ifetch_buffer_t m_inst_fetch_buffer;
  //ifetch_buffer_t的定义为：
  //    struct ifetch_buffer_t {
  //       ifetch_buffer_t() { m_valid = false; }
  //       ifetch_buffer_t(address_type pc, unsigned nbytes, unsigned warp_id) {
  //         m_valid = true;
  //         m_pc = pc;
  //         m_nbytes = nbytes;
  //         m_warp_id = warp_id;
  //       }
  //       bool m_valid;
  //       //获取的指令的PC值。
  //       address_type m_pc;
  //       unsigned m_nbytes;
  //       unsigned m_warp_id;
  //     };
  //这里可以看出指令获取缓冲区（ifetch_Buffer_t）仅仅能够容得下单条指令。

  //指令获取缓冲区。指令获取缓冲区（ifetch_Buffer_t）对指令缓存（I-cache）和SIMT Core之间的接口进行
  //建模。它有一个成员m_valid，用于指示缓冲区是否有有效的指令。它还将指令的warp id记录在m_warp_id中。
  //因此，当m_valid为0，即指示缓冲区暂时没有有效的指令，可以预取新的指令。注意预取新的指令时，要对新的
  //指令新建一个 ifetch_Buffer_t 对象，在这里 ifetch_Buffer_t 结构更像是每次取新指令这一行为的建模。

  //在每个fetch()步骤中，m_inst_fetch_buffer都会发出指令内存提取请求，并从L1指令缓存中收集提取的指令。
  //在每个decode()步骤中，m_inst_fetch_buffer中的上下文被解码为warp_inst_t的对象，并存储到等待调度器
  //发布的相应硬件warp的ibuffer中。
  if (!m_inst_fetch_buffer.m_valid) {
    //m_L1I是指令缓存（I-cache），在手册中<<三、SIMT Cores>>部分有I-cache的详细图。如果存在已经被填
    //入MSHR条目的访问，则m_L1I->access_ready()返回true。这里填入的内存访问代表的是，I-cache含有新
    //的可以就绪的指令。如果存在已经被填入MSHR条目的访问，则返回true，其条目非空证明可以合并内存访问。
    //m_current_response是就绪内存访问的列表，m_L1I->access_ready()返回的是m_current_response中是
    //否有就绪的内存访问。m_current_response仅存储了就绪内存访问的地址。
    if (m_L1I->access_ready()) {
      //获取I-cache的下次内存访问，返回下一个已经填入的访问，即返回下一个已经填入访问的请求。
      mem_fetch *mf = m_L1I->next_access();
      //如果前面 mem_fetch *mf 已经获取了就绪的指令，则证明 mf 所在的warp现在不处于挂起的指令缓冲未命
      //中的状态，设置该状态为false。mf->get_wid()返回的是当前已就绪的指令所在的warp的ID，那么证明当
      //前mf->get_wid()指示的warp已经能获取有效指令而不处于挂起的指令缓冲未命中的状态。
      m_warp[mf->get_wid()]->clear_imiss_pending();
      //创建对新指令预取这一行为的对象，传入参数为：
      //    address_type pc：m_warp[mf->get_wid()]->get_pc()
      //    unsigned nbytes：mf->get_access_size()
      //    unsigned warp_id：mf->get_wid()
      m_inst_fetch_buffer =
          ifetch_buffer_t(m_warp[mf->get_wid()]->get_pc(),
                          mf->get_access_size(), mf->get_wid());
      assert(m_warp[mf->get_wid()]->get_pc() ==
             (mf->get_addr() -
              PROGRAM_MEM_START));  // Verify that we got the instruction we
                                    // were expecting.
      //设置指示缓冲区内的指令有效。
      m_inst_fetch_buffer.m_valid = true;
      //设置上次取指的时钟周期。
      m_warp[mf->get_wid()]->set_last_fetch(m_gpu->gpu_sim_cycle);
      delete mf;
    } else {
      // find an active warp with space in instruction buffer that is not
      // already waiting on a cache miss and get next 1-2 instructions from
      // i-cache...
      //这里m_L1I->access_ready()不成立，即指令缓存中没有就绪的指令。在指令缓冲区中找到一个指示
      //有指令获取缓冲空间（上面的m_inst_fetch_buffer）的活跃warp，该空间尚未由于缓存未命中而等
      //待，并从I-cache中获取下一个1-2条指令。查找下一个warp时，采取轮询机制：
      //    (m_last_warp_fetched + 1 + i) % m_config->max_warps_per_shader;
      for (unsigned i = 0; i < m_config->max_warps_per_shader; i++) {
        //轮询机制获取下一个活跃warp。
        unsigned warp_id =
            (m_last_warp_fetched + 1 + i) % m_config->max_warps_per_shader;

        // this code checks if this warp has finished executing and can be
        // reclaimed.
        //下面的代码检查这个warp是否已经完成执行并且可以回收，各条件：
        //    m_warp[warp_id]->hardware_done()检查这个warp是否已经完成执行并且可以回收；
        //    m_scoreboard->pendingWrites(warp_id)返回记分牌的reg_table中是否有挂起的写入；
        //    m_warp[warp_id]->done_exit()返回线程退出的标识。
        //m_scoreboard->pendingWrites(warp_id)返回记分牌的reg_table中是否有隶属于当前warpid
        //的挂起的写入。warp id指向的reg_table为空的话，代表没有挂起的写入，返回false。[挂起的
        //写入]是指wid是否有已发射但尚未完成的指令，将目标寄存器保留在记分牌，这时候该warp尚未完
        //成执行并不能回收。
        //shd_warp_t::hardware_done()中：
        //    functional_done()返回warp已经执行完毕的标志，已经完成的线程数量=warp的大小时，就
        //    代表该warp已经完成。stores_done()返回所有store访存请求是否已经全部执行完，已发送
        //    但尚未确认的写存储请求（已发出写请求但未收到写确认信号时）数m_stores_outstanding
        //    =0时，代表所有store访存请求已经全部执行完，这里m_stores_outstanding在发出一个写
        //    请求时+=1，在收到一个写确认时-=1。m_inst_fetch_buffer中含有效指令时且将该指令解
        //    码过程中填充进warp的m_ibuffer时，增加在流水线中的指令数m_inst_in_pipeline（注意
        //    这里在decode()函数中会向warp的m_ibuffer填充进2条指令）；在指令完成写回操作时减少
        //    在流水线中的指令数m_inst_in_pipeline。inst_in_pipeline()返回流水线中的指令数量
        //    m_inst_in_pipeline。
        //    这里一个warp完成的标志由三个条件组成，分别是：1、warp已经执行完毕；2、所有store访
        //    存请求已经全部执行完；3、流水线中的指令数量为0。
        if (m_warp[warp_id]->hardware_done() &&
            !m_scoreboard->pendingWrites(warp_id) &&
            !m_warp[warp_id]->done_exit()) {
          //did_exit是标识当前循环内的warp是否退出，待后面该warp的各个线程都停止后，就设置为真。
          bool did_exit = false;
          //对一个warp内的所有线程进行循环。
          for (unsigned t = 0; t < m_config->warp_size; t++) {
            //tid是线程编号，这个编号是一个Shader Core内所有线程的索引，而不仅是warp内部的0~31。
            unsigned tid = warp_id * m_config->warp_size + t;
            //如果tid号线程处于活跃状态。
            if (m_threadState[tid].m_active == true) {
              //m_threadState[i]标识第i号线程是否处于活跃状态。m_threadState是一个数组，它包含
              //着整个Shader Core的所有的线程的状态。
              m_threadState[tid].m_active = false;
              //返回线程所在的CTA的ID。
              unsigned cta_id = m_warp[warp_id]->get_cta_id();
              if (m_thread[tid] == NULL) {
                //如果该线程信息为空，则注册该线程退出。register_cta_thread_exit功能是注册cta_id
                //所标识的CTA中的单个线程退出
				register_cta_thread_exit(cta_id,
                                         m_warp[warp_id]->get_kernel_info());
              } else {
                register_cta_thread_exit(cta_id,
                                         &(m_thread[tid]->get_kernel()));
              }
              //未完成的线程数（当此Shader Core上的所有线程都完成时，==0）。由于这里注册了线程的
              //退出，因此该线程已完成执行，未完成的线程数需要减去1。
              m_not_completed -= 1;
              //m_active_threads是Shader Core上活跃线程的位图，同理将tid位置零，取消活跃状态。
              m_active_threads.reset(tid);
              //did_exit是标识当前循环内的warp已经退出。
              did_exit = true;
            }
          }
          //当前循环内的warp已经退出，设置它退出的标志。
          if (did_exit) m_warp[warp_id]->set_done_exit();
          //m_active_warps是在此Shader Core中的活跃warp的总数，应该减1。
          --m_active_warps;
          assert(m_active_warps >= 0);
        }

        // this code fetches instructions from the i-cache or generates memory
        //此代码从I-Cache获取指令或生成内存访问。functional_done()返回warp是否已经执行完毕，已
        //经完成的线程数量=warp的大小时，就代表该warp已经完成。imiss_pending()返回warp是否因指
        //令缓冲未命中而挂起的状态标识。ibuffer_empty()返回I-Bufer是否为空。
        //这里是在前面的m_L1I->access_ready()不成立，即指令缓存中没有就绪的指令；且当前warp尚未
        //执行完毕；且该warp尚未处于指令缓冲未命中挂起的状态；且该warp的m_ibuffer为空，没有可以
        //后续执行的指令，则需要生成隶属于该warp的下一条指令预取。

        //这里说明下m_inst_fetch_buffer与m_ibuffer的区别：m_inst_fetch_buffer变量在获取（指令
        //缓存访问）和解码阶段之间充当流水线寄存器；而每个shd_warp_t都有一组m_ibuffer的I-Buffer
        //条目(ibuffer_entry)，持有可配置的指令数量（一个周期内允许获取的最大指令）。首先，如果
        //m_inst_fetch_buffer为空，则要判断L1指令缓存是否有就绪的指令，就绪指令存在的话要从L1指
        //令缓存中获取指令；而如果L1指令缓存中没有指令，这时候要判断一下是否因为warp运行完毕导致
        //L1指令缓存中没有指令了，如果是warp正在运行，且没有等待挂起的L1缓存未命中的挂起状态，且
        //warp的指令缓冲区为空，这时候就说明要取下一条PC值的指令。
        if (!m_warp[warp_id]->functional_done() &&
            !m_warp[warp_id]->imiss_pending() &&
            m_warp[warp_id]->ibuffer_empty()) {
          address_type pc;
          //返回warp内下一个要执行的指令的PC值。
          pc = m_warp[warp_id]->get_pc();
          //上一步获取的PC值，是从0开始编号的，需要加上指令在内存中存储的首地址0xF0000000才能到
          //I-Cache中取指令，因为I-Cache的起始地址就是0xF0000000。
          address_type ppc = pc + PROGRAM_MEM_START;
          unsigned nbytes = 16;
          //offset_in_block是PC标识的指令在I-cache line中的偏移。
          unsigned offset_in_block =
              pc & (m_config->m_L1I_config.get_line_sz() - 1);
          //如果这个偏移+nbytes > I-cache line Size，就说明一行Cache存不下nbyte大小的数据，重
          //新设置nbytes的值。
          if ((offset_in_block + nbytes) > m_config->m_L1I_config.get_line_sz())
            nbytes = (m_config->m_L1I_config.get_line_sz() - offset_in_block);

          // TODO: replace with use of allocator
          // mem_fetch *mf = m_mem_fetch_allocator->alloc()
          //创建内存访问行为这个对象，该内存访问对象的类型是INST_ACC_R即从I-Cache中读指令，地址
          //为PC值，nbytes字节大小，非写访问。
          mem_access_t acc(INST_ACC_R, ppc, nbytes, false, m_gpu->gpgpu_ctx);
          mem_fetch *mf = new mem_fetch(
              acc, NULL, m_warp[warp_id]->get_kernel_info()->get_streamID(),
              READ_PACKET_SIZE, warp_id, m_sid, m_tpc, m_memory_config,
              m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle);
          std::list<cache_event> events;

          // Check if the instruction is already in the instruction cache.
          enum cache_request_status status;
          //perfect_inst_const_cache代表完美的inst和const缓存模式，缓存中的所有inst和const都
          //命中，在V100配置文件里开启。
          if (m_config->perfect_inst_const_cache) {
            status = HIT;
            shader_cache_access_log(m_sid, INSTRUCTION, 0);
          } else
            status = m_L1I->access(
                (new_addr_type)ppc, mf,
                m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle, events);
          //完美的inst和const缓存模式，缓存中的所有inst都HIT。
          if (status == MISS) {
            m_last_warp_fetched = warp_id;
            m_warp[warp_id]->set_imiss_pending();
            m_warp[warp_id]->set_last_fetch(m_gpu->gpu_sim_cycle);
          } else if (status == HIT) {
            m_last_warp_fetched = warp_id;
            //将pc值对应的指令放入m_inst_fetch_buffer。这里要再次说明下m_inst_fetch_buffer与
            //m_ibuffer的区别：m_inst_fetch_buffer变量在获取（指令缓存访问）和解码阶段之间充当
            //流水线寄存器；而每个shd_warp_t都有一组m_ibuffer的I-Buffer条目(ibuffer_entry)，
            //持有可配置的指令数量（一个周期内允许获取的最大指令）。
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
  //取指完成后，L1 I-Cache向前推进一拍。
  m_L1I->cycle();
}

/*
指令的功能执行。在性能模型 shader_core_ctx::issue_warp 函数执行时，即指令确定能发射时，
调用该函数对指令进行功能模拟。功能模拟的过程中，如果指令是load或store指令，则生成内存访
问的必要的信息，例如：设置 access_type，计算一个 warp 指令对 shared memory 访存的执行
周期等。
*/
void exec_shader_core_ctx::func_exec_inst(warp_inst_t &inst) {
  execute_warp_inst_t(inst);
  if (inst.is_load() || inst.is_store()) {
    inst.generate_mem_accesses();
    // inst.print_m_accessq();
  }
}

/*
发射warp。例如：
    m_shader->issue_warp(*m_mem_out, pI, active_mask, warp_id, m_id);
*/
void shader_core_ctx::issue_warp(register_set &pipe_reg_set,
                                 const warp_inst_t *next_inst,
                                 const active_mask_t &active_mask,
                                 unsigned warp_id, unsigned sch_id) {
  //在pipe_reg_set流水线寄存器中寻找sch_id对应的空闲的寄存器。在subcore模式下，每
  //个warp调度器在寄存器集合中有一个具体的寄存器可供使用，这个寄存器由调度器的m_id
  //索引。
  warp_inst_t **pipe_reg =
      pipe_reg_set.get_free(m_config->sub_core_model, sch_id);
  assert(pipe_reg);
  //由于已经决定发射指令，因此将I-Bufer中的next_inst所在的槽清除置无效。
  m_warp[warp_id]->ibuffer_free();
  assert(next_inst->valid());
  **pipe_reg = *next_inst;  // static instruction information
  //(*pipe_reg)->issue()的定义如下：
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
  //设置指令动态发射过程中的一些信息。
  (*pipe_reg)->issue(
      active_mask, warp_id, m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle,
      m_warp[warp_id]->get_dynamic_warp_id(), sch_id,
      m_warp[warp_id]->get_streamID());  // dynamic instruction information
  m_stats->shader_cycle_distro[2 + (*pipe_reg)->active_count()]++;
  //由于已经确定了指令next_inst的执行顺序没有问题，因此可以对该条指令进行功能模拟。
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
  
  //如果发射的指令的OP操作码是屏障指令，则保存当前warp处于屏障指令状态。
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

  //更新SIMT堆栈。
  updateSIMTStack(warp_id, *pipe_reg);
  //设置计分板，发射指令时，将其目标寄存器保留在相应硬件warp的记分牌中。
  m_scoreboard->reserveRegisters(*pipe_reg);
  //设置下一条指令的PC值。
  m_warp[warp_id]->set_next_pc(next_inst->pc + next_inst->isize);
}

/*
在每个SIMT Core中，都有可配置数量的调度器单元。函数shader_core_ctx::issue()在这些单元上进行迭代，
其中每一个单元都执行scheduler_unit::cycle()，在这里对warp进行轮循。在scheduler_unit::cycle()中，
指令使用shader_core_ctx::issue_warp()函数被发射到其合适的执行流水线。在这个函数中，指令通过调用
shader_core_ctx::func_exec_inst()在功能上被执行，SIMT堆栈（m_simt_stack[warp_id]）是通过调用
simt_stack::update()被更新。另外，在这个函数中，由于barrier的存在，通过shd_warp_t:set_membar()
和barrier_set_t::warp_reaches_barrier来保持/释放warp。另一方面，寄存器被Scoreboard::reserve-
Registers()保留，以便以后被记分牌算法使用。scheduler_unit::m_sp_out,scheduler_unit::m_sfu_out, 
scheduler_unit::m_mem_out指向SP、SFU和Mem流水线接收的发射阶段和执行阶段之间的第一个流水线寄存器。
这就是为什么在使用shader_core_ctx::issue_warp()向其相应的流水线发出任何指令之前要检查它们。

单条指令的吞吐和延迟
在每个pipelined_simd_unit中，issue()成员函数将给定的流水线寄存器的内容移入m_dispatch_reg。然后指
令在m_dispatch_reg等待initiation_interval个周期。在此期间，没有其他的指令可以发到这个单元，所以这
个等待是指令的吞吐量的模型。等待之后，指令被派发到内部流水线寄存器m_pipeline_reg进行延迟建模。派遣
的位置是确定的，所以在m_dispatch_reg中花费的时间也被计入延迟中。每个周期，指令将通过流水线寄存器前
进，最终进入m_result_port，这是共享的流水线寄存器，通向SP和SFU单元的共同写回阶段。示意图：

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
                                m_result_port --> 写回
/////////////////////////////////////////////////////////////////////////////////////////
# GPGPU-Sim的时钟推进

## 四个时钟域的选择

每个时钟域维护各自的当前执行周期数，这里不同时钟域的单个执行周期数由各自时钟域的频率决定：
```c++
l2_time, icnt_time, dram_time, core_time
```
GPGPU-Sim每拍向前推进一个时钟周期时，会选择一个或几个当前执行周期数最小的时钟域进行推进，同时将该时
钟域维护的当前执行周期数增加一拍，这个选择是由函数`gpgpu_sim::next_clock_domain(void)`完成的。
例如，当前四个时钟域的执行周期分别为：
```c++
l2_time=35拍, icnt_time=40拍, dram_time=50拍, core_time=30拍
```
那么会选择`CORE`时钟域进行更新，同时将`core_time`增加一拍，变为31拍，其他时钟域的执行周期数不变。

## 时钟域的更新
在选择`CORE`时钟域进行更新后，会循环所有SIMT Core集群都推进一个时钟周期：
```c++
if (clock_mask & CORE) {
    //shader core loading (pop from ICNT into core) follows CORE clock.
    //对所有的SIMT Core集群循环，m_cluster[i]是其中一个集群。Shader Core加载（从ICNT弹出到Core）
    //遵循Core时钟。
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
      //simt_core_cluster::icnt_cycle()方法将内存请求从互连网络推入SIMT核心集群的响应FIFO。它还从
      //FIFO弹出请求，并将它们发送到相应内核的指令缓存或LDST单元。每个SIMT Core集群都有一个响应FIFO，
      //用于保存从互连网络发出的数据包。数据包被定向到SIMT Core的指令缓存（如果它是为指令获取未命中
      //提供服务的内存响应）或其内存流水线（memory pipeline，LDST 单元）。数据包以先进先出方式拿出。
      //如果SIMT Core无法接受FIFO头部的数据包，则响应FIFO将停止。为了在LDST单元上生成内存请求，每个
      //SIMT Core都有自己的注入端口接入互连网络。但是，注入端口缓冲区由SIMT Core集群所有SIMT Core共
      //享。
      m_cluster[i]->icnt_cycle();
}
......
if (clock_mask & CORE) {
    //对GPU中所有的SIMT Core集群进行循环，更新每个集群的状态。
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
      //如果get_not_completed()大于1，代表这个SIMT Core尚未完成；如果get_more_cta_left()为1，
      //代表这个SIMT Core还有剩余的CTA需要取执行。m_cluster[i]->get_not_completed()返回第i个
      //SIMT Core集群中尚未完成的线程个数。
      if (m_cluster[i]->get_not_completed() || get_more_cta_left()) {
        //当调用simt_core_cluster::core_cycle()时，它会调用其中所有SM内核的循环。
        m_cluster[i]->core_cycle();
        //增加活跃的SM数量。get_n_active_sms()返回SIMT Core集群中的活跃SM的数量。active_sms是
        //SIMT Core集群中的活跃SM的数量。
        *active_sms += m_cluster[i]->get_n_active_sms();
      }
    //需要注意，gpu_sim_cycle仅在CORE时钟域向前推进一拍时才更新，因此gpu_sim_cycle表示CORE时钟
    //域的当前执行拍数。
    gpu_sim_cycle++;
    //对所有SIMT Core集群遍历，选择每个集群内的一个SIMT Core，并向其发射一个线程块。在选择发射哪
    //个kernel时，会调用gpgpu_sim::select_kernel()方法。该方法会判断当前是否还有未完成的kernel，
    //并判断是否该kernel已经把所有启动时间减为0。如果有未完成的kernel，且该kernel已经把所有启动
    //时间减为0，则会选择该kernel的以执行。
    issue_block2core();
    //m_kernel_TB_latency用于模拟kernel&block的启动延迟时间，这包括kernel的启动延迟，每个线程块
    //的启动延迟。每次调用decrement_kernel_latency()时，都会将m_kernel_TB_latency减1，直到减为
    //0，才说明kernel可以被选择以执行。
    decrement_kernel_latency();
}
```

## 单个SIMT Core集群以及单个SIMT Core的更新
单个SIMT Core集群向前推进一个时钟周期，调用`simt_core_cluster::core_cycle()`方法：
```c++
void simt_core_cluster::core_cycle() {
  //对SIMT Core集群中的每一个单独的SIMT Core循环。
  for (std::list<unsigned>::iterator it = m_core_sim_order.begin();
       it != m_core_sim_order.end(); ++it) {
    //SIMT Core集群中的每一个单独的SIMT Core都向前推进一个时钟周期。
    m_core[*it]->cycle();
  }
}
```
单个SIMT Core向前推进一个时钟周期，调用`shader_core_ctx::cycle()`方法：
```c++
  //如果这个SIMT Core处于非活跃状态，且已经执行完成，时钟周期不向前推进。
  if (!isactive() && get_not_completed() == 0) return;

  //每个内核时钟周期，shader_core_ctx::cycle()都被调用，以模拟SIMT Core的一个周期。这个函数调用一
  //组成员函数，按相反的顺序模拟内核的流水线阶段，以模拟流水线效应：
  //     fetch()             倒数第一执行
  //     decode()            倒数第二执行
  //     issue()             倒数第三执行
  //     read_operand()      倒数第四执行
  //     execute()           倒数第五执行
  //     writeback()         倒数第六执行
  writeback();
  execute();
  read_operands();
  issue();
  for (int i = 0; i < m_config->inst_fetch_throughput; ++i) {
    decode();
    fetch();
  }
```

## 单条指令的吞吐和延迟
在每个pipelined_simd_unit中，issue(warp_inst_t*&)成员函数将给定的流水线寄存器的内容移入m_dispat
ch_reg。然后指令在m_dispatch_reg等待initiation_interval个周期。在此期间，没有其他的指令可以发到这
个单元，所以这个等待是指令的吞吐量的模型。等待之后，指令被派发到内部流水线寄存器m_pipeline_reg进行延
迟建模。派遣的位置是确定的，所以在m_dispatch_reg中花费的时间也被计入延迟中。每个周期，指令将通过流水
线寄存器前进，最终进入m_result_port，这是共享的流水线寄存器，通向SP和SFU单元的共同写回阶段。示意图：
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
                                m_result_port --> 写回
```
/////////////////////////////////////////////////////////////////////////////////////////
*/
void shader_core_ctx::issue() {
  // Ensure fair round robin issu between schedulers
  unsigned j;
  //对Shader Core里的可配置数量的调度器单元进行迭代，其中每一个单元都执行scheduler_unit::cycle()。
  //下面这段代码其实是在模拟调度器的轮循，Volta架构每个Shader Core有4个调度器，第一拍调度时，选择第
  //0个调度器先往前推进一拍，接着是1、2、3个调度器前推进一拍；下一拍则先把第1个调度器往前推进一拍，接
  //着是2、3、0个调度器前推进一拍；以此类推。
  for (unsigned i = 0; i < schedulers.size(); i++) {
    j = (Issue_Prio + i) % schedulers.size();
    //调度器向前推进一拍，包括执行从warp的m_ibuffer中取值，SIMT堆栈检查，计分板检查，流水线单元检查，
    //最后将执行写入对应的流水线单元前的寄存器集合，并进行功能模拟。需要注意，先执行哪个调度器有一次调
    //度（这里采用的是轮询策略调度），在选定某个调度器执行的时候，隶属于该调度器的哪个warp先执行也有一次调
    //度（这里V100配置采用的是LRR最近最少被使用策略调度）。
    schedulers[j]->cycle();
  }
  Issue_Prio = (Issue_Prio + 1) % schedulers.size();

  // really is issue;
  // for (unsigned i = 0; i < schedulers.size(); i++) {
  //    schedulers[i]->cycle();
  //}
}

shd_warp_t &scheduler_unit::warp(int i) { return *((*m_warp)[i]); }
//LRR调度策略的调度器单元的order_lrr函数，为当前调度单元内所划分到的warp进行排序。order_lrr的定义
//为：
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
//这个函数的功能就是根据上一拍调度过的warp，找到它在当前调度单元所需要仲裁的warp集合中的位置，然后
//从这个位置后面的warp起始，遍历当前调度单元所需要仲裁的warp集合，并将这些warp放入result_list中，
//直到result_list中的warp数目等于num_warps_to_add。

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
  //如果当前调度单元上一拍调度过的warp不在当前调度单元所需要仲裁的warp集合中，则将其置为当前调度单
  //元所需要仲裁的warp集合的第一个warp；而如果当前调度单元上一拍调度过的warp在当前调度单元所需要仲
  //裁的warp集合中，则将其置为按照顺序它的下一个warp。
  result_list.clear();
  typename std::vector<T>::const_iterator iter =
      (last_issued_from_input == input_list.end()) ? input_list.begin()
                                                   : last_issued_from_input + 1;
  //对当前调度单元所需要仲裁的warp集合进行遍历，将这些warp按照从上段代码找到的上一拍调度过的warp的
  //下一个warp开始直到所有warp都遍历一遍的顺序，放入result_list中。
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
Shader Core里的单个调度器单元向前推进一拍，执行scheduler_unit::cycle()。
*/
void scheduler_unit::cycle() {
  SCHED_DPRINTF("scheduler_unit::cycle()\n");
  // These three flags match the valid, ready, and issued state of warps in
  // the scheduler.
  //ibuffer中取出的PC值与SIMT堆栈中的PC值匹配，则说明没有控制冒险，设置为真
  bool valid_inst =
      false;  // there was one warp with a valid instruction to issue (didn't
              // require flush due to control hazard)
  //指令通过记分板检查，就可以将指令ready状态设置为true。
  bool ready_inst = false;   // of the valid instructions, there was one not
                             // waiting for pending register writes
  //指令发射成功后，设置issued_inst为真。
  bool issued_inst = false;  // of these we issued one

  //warp是根据某些策略重新排序的，这是不同调度器之间的主要区别。对于V100来说，采用LRR调度策略，
  //LRR调度策略的调度器单元的order_warps()函数，为当前调度单元内所划分到的warp进行排序。它会
  //再调用：
  //order_lrr(m_next_cycle_prioritized_warps, m_supervised_warps,
  //          m_last_supervised_issued, m_supervised_warps.size());
  //其参数列表为：
  //    result_list：m_next_cycle_prioritized_warps是一个vector，里面存储当前调度单元当前拍
  //                 经过warp排序后，在下一拍具有优先级调度的warp。
  //    input_list：m_supervised_warps，是一个vector，里面存储当前调度单元所需要仲裁的warp。
  //    last_issued_from_input：则存储了当前调度单元上一拍调度过的warp。
  //    num_warps_to_add：m_supervised_warps.size()，则是当前调度单元在下一拍需要调度的warp
  //                      数目，在这里这个warp数目就是当前调度器所划分到的warp子集合的大小。
  //这个函数的功能就是根据上一拍调度过的warp，找到它在当前调度单元所需要仲裁的warp集合中的位置，
  //然后从这个位置后面的第一个warp起始，一直遍历当前调度单元所需要仲裁的warp集合，并将这些warp放
  //入result_list中，直到result_list中的warp数目等于num_warps_to_add。
  order_warps();
  //Loop through all the warps based on the order
  //m_next_cycle_prioritized_warps里存储了排序后的下一拍应优先调度的warp顺序，遍历整个优先级的
  //warp列表，依次进行调度。
  for (std::vector<shd_warp_t *>::const_iterator iter =
           m_next_cycle_prioritized_warps.begin();
       iter != m_next_cycle_prioritized_warps.end(); iter++) {
    // Don't consider warps that are not yet valid
    //如果warp不是有效的，即没有有效的指令，或者warp已经执行完毕，则跳过调度该warp。
    if ((*iter) == NULL || (*iter)->done_exit()) {
      continue;
    }
    SCHED_DPRINTF("Testing (warp_id %u, dynamic_warp_id %u)\n",
                  (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id());
    unsigned warp_id = (*iter)->get_warp_id();
    unsigned checked = 0;
    //当前warp中发射的指令计数值。
    unsigned issued = 0;
    //当前warp中记录上一次发射的指令的执行单元类型。
    exec_unit_type_t previous_issued_inst_exec_type = exec_unit_type_t::NONE;
    //每个warp单次的最大发射指令数，在V100配置中设置为1。
    unsigned max_issue = m_shader->m_config->gpgpu_max_insn_issue_per_warp;
    //仅允许向不同的硬件单元双发射，在V100配置中设置为1。diff_exec_units是仅允许向不同的硬件单
    //元双发射，在V100配置中设置为1，因此这里在同一拍同一个warp调度器不能向同一硬件单元发射两条
    //及以上指令，所以这里要判断一下当前的warp是否上一条是存储指令，不是的话才可继续发射。
    bool diff_exec_units =
        m_shader->m_config
            ->gpgpu_dual_issue_diff_exec_units;  // In tis mode, we only allow
                                                 // dual issue to diff execution
                                                 // units (as in Maxwell and
                                                 // Pascal)

    //返回I-Bufer是否为空。这里一个warp有一个I-Bufer，I-Bufer是一个队列，存储了当前warp中的待
    //执行指令。
    if (warp(warp_id).ibuffer_empty())
      SCHED_DPRINTF(
          "Warp (warp_id %u, dynamic_warp_id %u) fails as ibuffer_empty\n",
          (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id());

    //返回warp是否由于（warp已经执行完毕且在等待新内核初始化、CTA处于barrier、memory barrier、
    //还有未完成的原子操作）四个条件处于等待状态。
    if (warp(warp_id).waiting())
      SCHED_DPRINTF(
          "Warp (warp_id %u, dynamic_warp_id %u) fails as waiting for "
          "barrier\n",
          (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id());

    //checked是下面循环的循环次数，即在当前可调度的warp下，执行检测这个warp的可发射指令数至多为
    //max_issue，在V100配置中为1，即无论这个循环中有没有将指令发射出去，都不能再进行第二轮循环，
    //因为单个warp被配置为每次至多调度一条指令。issued是当前warp中记录的发射的指令计数值，该值不
    //能超过当前warp的最大可发射指令数max_issue。同时，checked次数必须保证小于等于issued，因为
    //一旦checked次数大于issued，则说明在已经检查过的指令中，最后一条指令因为某种原因没有发射成
    //功，这时候我们就要暂停当前warp的调度，以保证指令执行的正确性。
    while (!warp(warp_id).waiting() && !warp(warp_id).ibuffer_empty() &&
           (checked < max_issue) && (checked <= issued) &&
           (issued < max_issue)) {
      //对warp_id代表的warp，获取其ibuffer中的下一条指令。
      const warp_inst_t *pI = warp(warp_id).ibuffer_next_inst();
      // Jin: handle cdp latency;
      if (pI && pI->m_is_cdp && warp(warp_id).m_cdp_latency > 0) {
        assert(warp(warp_id).m_cdp_dummy);
        warp(warp_id).m_cdp_latency--;
        break;
      }
      //获取ibuffer中的下一条指令，即刚刚取出的指令pI是否有效。
      bool valid = warp(warp_id).ibuffer_next_valid();
      //标志位，指示warp是否发射了指令。
      bool warp_inst_issued = false;
      //warp_id对应的SIMT堆栈顶部的PC值和RPC值。
      unsigned pc, rpc;
      //每个warp有自己的SIMT堆栈，获取warp_id对应的SIMT堆栈顶部的PC值和RPC值。
      m_shader->get_pdom_stack_top_info(warp_id, pI, &pc, &rpc);
      SCHED_DPRINTF(
          "Warp (warp_id %u, dynamic_warp_id %u) has valid instruction (%s)\n",
          (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id(),
          m_shader->m_config->gpgpu_ctx->func_sim->ptx_get_insn_str(pc)
              .c_str());
      //pI是从ibuffer取出的指令，如果该指令有效。
      if (pI) {
        assert(valid);
        //简而言之，调度器发现了一个具有有效ibuffer中的warp，而不是等待障碍。获取warp后，
        //从ibuffer获取指令并检查其是否有效。对于有效指令，如果其pc与当前SIMT堆栈的pc不匹
        //配，则意味着发生了控制危险，并且ibuffer被刷新。然后，它的源寄存器和目标寄存器被
        //传递到记分板进行冲突检查。如果它也通过了记分板，检查目标功能单元的ID_OC流水线寄存
        //器集是否有空闲插槽。如果有，则可以发出指令，循环的inital将中断。否则，若当前warp
        //中的指令未发出，则检查下一个warp。因此，每个调度程序单元每个周期只发出一条指令。
        if (pc != pI->pc) {
          SCHED_DPRINTF(
              "Warp (warp_id %u, dynamic_warp_id %u) control hazard "
              "instruction flush\n",
              (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id());
          // control hazard
          //将warp下一次执行的指令PC值设置为从SIMT堆栈中取出的PC。
          warp(warp_id).set_next_pc(pc);
          //刷新warp的ibuffer，因为ibuffer此刻已有的指令已经不会再执行。
          warp(warp_id).ibuffer_flush();
        } else {
          //如果ibuffer中取出的PC值与SIMT堆栈中的PC值匹配，则说明没有控制冒险，设置为真。
          valid_inst = true;
          //检测计分板的冒险，检测某个指令使用的寄存器是否被保留在记分板中，如果有的话就是
          //发生了 WAW 或 RAW 冒险，则不发射该条指令。
          if (!m_scoreboard->checkCollision(warp_id, pI)) {
            SCHED_DPRINTF(
                "Warp (warp_id %u, dynamic_warp_id %u) passes scoreboard\n",
                (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id());
            //一旦指令通过记分板检查，就可以将指令ready状态设置为true。
            ready_inst = true;
            //获取warp_id对应的指令pI的活跃掩码。
            const active_mask_t &active_mask =
                m_shader->get_active_mask(warp_id, pI);

            assert(warp(warp_id).inst_in_pipeline());

            //MEM相关的指令，需要发送到m_mem_out流水线寄存器。
            if ((pI->op == LOAD_OP) || (pI->op == STORE_OP) ||
                (pI->op == MEMORY_BARRIER_OP) ||
                (pI->op == TENSOR_CORE_LOAD_OP) ||
                (pI->op == TENSOR_CORE_STORE_OP)) {
              //m_id是调度器单元的ID，m_mem_out是调度器单元的m_mem_out流水线寄存器。如
              //果m_mem_out流水线寄存器有空闲插槽，且不支持向不同的硬件单元双发射，或者允
              //许向不同的硬件单元双发射，但是上一条指令的执行单元类型不是MEM相关，则可以
              //发射。在subcore模式下，每个warp调度器在寄存器集合中有一个具体的寄存器可供
              //使用，这个寄存器由调度器的m_id索引。这里就是去查找m_mem_out寄存器集合中的
              //m_id号调度器仅能使用的第m_id号寄存器是否为空可用，同时，diff_exec_units
              //是仅允许向不同的硬件单元双发射，在V100配置中设置为1，因此这里在同一拍同一
              //个warp调度器不能向同一硬件单元发射两条及以上指令，所以这里要判断一下当前的
              //warp是否上一条是存储指令，不是的话才可继续发射。
              if (m_mem_out->has_free(m_shader->m_config->sub_core_model,
                                      m_id) &&
                  (!diff_exec_units ||
                   previous_issued_inst_exec_type != exec_unit_type_t::MEM)) {
                //向m_mem_out流水线寄存器发射指令pI。
                m_shader->issue_warp(*m_mem_out, pI, active_mask, warp_id,
                                     m_id);
                //发射的指令计数值加一。
                issued++;
                //指令发射成功后，设置issued_inst为真。
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
          //m_last_supervised_issued是指代上一次调度的warp。
          m_last_supervised_issued = supervised_iter;
        }
      }
      //记录上一拍发射的指令数。
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
  //ibuffer中取出的PC值与SIMT堆栈中的PC值匹配，则说明没有控制冒险，设置为真：
  //    bool valid_inst=False说明Idle或者由控制冒险。
  //指令通过记分板检查，就可以将指令ready状态设置为true：
  //    bool ready_inst=False说明等待RAW冒险（可能是由于内存）。
  //指令发射成功后，设置issued_inst为真：
  //    bool issued_inst=False说明流水线停顿。
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
LRR调度策略的调度器单元的order_warps()函数，为当前调度单元内所划分到的warp进行排序。order_lrr
的定义为：
    void scheduler_unit::order_lrr(
        std::vector<T> &result_list, const typename std::vector<T> &input_list,
        const typename std::vector<T>::const_iterator &last_issued_from_input,
        unsigned num_warps_to_add)
从这里看出，m_next_cycle_prioritized_warps是一个vector，里面存储了当前调度单元当前拍经过warp
排序后，在下一拍具有优先级调度的warp。last_issued_from_input则存储了当前调度单元上一拍调度过的
warp。num_warps_to_add则是当前调度单元在下一拍需要调度的warp数目，在这里这个warp数目就是当前调
度器所划分到的warp子集合m_supervised_warps的大小。
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
模拟操作数收集器从寄存器文件读取指令的源操作数，将原先暂存在收集器单元指令槽m_warp中的指令推出到
m_output_register中。
*/
void shader_core_ctx::read_operands() {
  //m_config->reg_file_port_throughput是寄存器文件的端口数。在V100配置文件里gpgpu_reg_file_
  //port_throughput被设置为2。
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
此函数在所有结果总线中定位一个空闲插槽（如果未设置其位，则该插槽为空闲插槽）。
*/
int shader_core_ctx::test_res_bus(int latency) {
  //结果总线共有m_config->pipe_widths[EX_WB]条。
  //流水线阶段的宽度配置在-gpgpu_pipeline_widths中设置：
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
  //在V100中配置为：-gpgpu_pipeline_widths 4,4,4,4,4,4,4,4,4,4,8,4,4
  //结果总线的宽度是1，即num_result_bus = m_config->pipe_widths[EX_WB] = 8。
  for (unsigned i = 0; i < num_result_bus; i++) {
    if (!m_result_bus[i]->test(latency)) {
      return i;
    }
  }
  return -1;
}

/*
SM的执行，各功能单元向前推进一拍。
*/
void shader_core_ctx::execute() {
  //结果总线共有m_config->pipe_widths[EX_WB]条。
  //流水线阶段的宽度配置在-gpgpu_pipeline_widths中设置：
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
  //在V100中配置为：-gpgpu_pipeline_widths 4,4,4,4,4,4,4,4,4,4,8,4,4
  //结果总线的宽度是1，即num_result_bus = m_config->pipe_widths[EX_WB] = 8。
  for (unsigned i = 0; i < num_result_bus; i++) {
    *(m_result_bus[i]) >>= 1;
  }
  for (unsigned n = 0; n < m_num_function_units; n++) {
    //m_fu是SIMD功能单元的向量，m_num_function_units是SIMD功能单元的数量。m_fu包含：
    //  4个SP单元，4个DP单元，4个INT单元，4个SFU单元，4个TC单元，多个或零个specialized_unit，
    //  1个LD/ST单元。
    //在V100配置中，LDST单元以及其他SIMD单元，m_fu[n]->clock_multiplier()均返回1。一些单元，
    //例如在其他配置文件里，LDST单元可能以更高的时钟频率运行，因此m_fu[n]->clock_multiplier()
    //可能返回2。这代表在SM的时钟域向前推进一拍时，LDST单元会向前推进两拍。
    unsigned multiplier = m_fu[n]->clock_multiplier();
    //m_fu[n]单元向前推进一拍。调用pipelined_simd_unit::cycle()，流水线单元向前推进一拍。
    for (unsigned c = 0; c < multiplier; c++) m_fu[n]->cycle();
    //更新m_state的一些性能计数器。
    m_fu[n]->active_lanes_in_pipeline();
    //m_issue_port的定义如下：
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
    //根据m_issue_port[n]，获取流水线寄存器m_pipeline_reg[issue_port]中的指令寄存器集合
    //issue_inst。
    register_set &issue_inst = m_pipeline_reg[issue_port];
    unsigned reg_id;
    //在V100配置中，partition_issue仅有在LDST单元中为false，其余单元均为true。
    bool partition_issue =
        m_config->sub_core_model && m_fu[n]->is_issue_partitioned();
    //当partition_issue为false时，reg_id为0。当partition_issue为true时，reg_id为m_fu[n]
    //->get_issue_reg_id()。这是因为比如说LDST单元仅有一个，其issue_reg_id为0，而其他单元
    //有多个，所以需要通过get_issue_reg_id()来获取。get_issue_reg_id()其实返回的就是当前
    //SIMD单元在m_fu中相同类型单元中的索引（例如一共有4个SP单元，当前单元是从零编号开始第3个，
    //所以该函数就返回的是3）。
    if (partition_issue) {
      reg_id = m_fu[n]->get_issue_reg_id();
    }
    //返回m_fu[n]单元的流水线寄存器m_pipeline_reg[issue_port]中的指令寄存器集合issue_inst
    //的第reg_id个。
    warp_inst_t **ready_reg = issue_inst.get_ready(partition_issue, reg_id);
    //给定一个寄存器reg_id，判断该寄存器是否非空。
    if (issue_inst.has_ready(partition_issue, reg_id) &&
        //返回**ready_reg指令的return m_dispatch_reg->empty()，即判断m_dispatch_reg是否
        //为空。
        m_fu[n]->can_issue(**ready_reg)) {
      //m_fu[n]->stallable()：LDST单元返回true，其余返回false。
      bool schedule_wb_now = !m_fu[n]->stallable();
      int resbus = -1;
      if (schedule_wb_now &&
          //除LDST单元外走这里。
          //test_res_bus() function locates a free slot in all the result buses (the 
          //slot is free if its bit is not set).
          //test_res_bus函数在所有结果总线中定位一个空闲插槽（如果未设置其位，则该插槽为空
          //闲插槽)。实际上test_res_bus()函数模拟的是指令延迟。
          (resbus = test_res_bus((*ready_reg)->latency)) != -1) {
        assert((*ready_reg)->latency < MAX_ALU_LATENCY);
        //m_result_bus实际上是模拟指令延迟。
        m_result_bus[resbus]->set((*ready_reg)->latency);
        //执行simd_function_unit::issue(register_set &source_reg)函数。
        m_fu[n]->issue(issue_inst);
      } else if (!schedule_wb_now) {
        //LDST单元走这里。
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
流水线的写回阶段。将执行阶段的结果写回到寄存器文件中。首先，EX_WB寄存器组中的就绪槽被识别并加
载到preg。如果有效，则调用m_operated_collector.writeback。然后，目标寄存器从记分板上释放，
EX_WB寄存器集中的插槽被清除。它一直持续到EX_WB寄存器集中的所有就绪指令都被写回为止。
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

  //get_ready()从EX_WB流水线寄存器获取一个非空寄存器，将其指令移出，并返回这条指令。
  //这里，ldst单元的指令不会执行到这里，因为ldst单元的指令在执行阶段就已经写回了。
  warp_inst_t **preg = m_pipeline_reg[EX_WB].get_ready();
  warp_inst_t *pipe_reg = (preg == NULL) ? NULL : *preg;
  while (preg and !pipe_reg->empty()) {
    /*
     * Right now, the writeback stage drains all waiting instructions
     * assuming there are enough ports in the register file or the
     * conflicts are resolved at issue.
     */
    /*
    现在，写回阶段会耗尽所有等待的指令，假设寄存器文件中有足够的端口，或者冲突已经解决。
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
    操作数收集器写回通常会生成暂停。然而，在这里，流水线应该是un-stallable的。这是有保
    证的，因为这是在操作数收集器的step函数之后首次调用写回函数，该函数重置分配。有一种情
    况可能导致写回函数返回false（stall），即指令试图修改两个寄存器（GPR和谓词）。为了处
    理这种情况，我们忽略返回值（因此不允许停滞）。
    */
    //操作数收集器的Bank写回。
    m_operand_collector.writeback(*pipe_reg);
    //获取pipe_reg指令的warp ID。
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
    //循环下一个流水线寄存器集合m_pipeline_reg[EX_WB]中的有效指令寄存器，执行它的停止
    //任务。
    preg = m_pipeline_reg[EX_WB].get_ready();
    pipe_reg = (preg == NULL) ? NULL : *preg;
  }
}

/*
判断LDST单元访问shared memory是否会Stall。Stall的话返回0，没有Stall的话返回1。
*/
bool ldst_unit::shared_cycle(warp_inst_t &inst, mem_stage_stall_type &rc_fail,
                             mem_stage_access_type &fail_type) {
  //如果指令不是shared memory指令，则返回true。
  if (inst.space.get_type() != shared_space) return true;
  //如果指令的活跃线程数为0，则返回true。
  if (inst.active_count() == 0) return true;

  //返回该条指令还剩下多少initiation_interval。
  if (inst.has_dispatch_delay()) {
    m_stats->gpgpu_n_shmem_bank_access[m_sid]++;
  }
  //初始化时设置为cycles = initiation_interval;，每次调用dispatch_delay()时，cycles
  //减1，直到cycles=0。
  bool stall = inst.dispatch_delay();
  if (stall) {
    //Stall类型为S_MEM，原因是Bank conflict。
    fail_type = S_MEM;
    rc_fail = BK_CONF;
    m_stats->gpgpu_n_shmem_bkconflict++;
  } else
    rc_fail = NO_RC_FAIL;
  
  //Stall的话返回0，没有Stall的话返回1。
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
处理对L1数据缓存的内存访问。
*/
mem_stage_stall_type ldst_unit::process_memory_access_queue_l1cache(
    l1_cache *cache, warp_inst_t &inst) {
  mem_stage_stall_type result = NO_RC_FAIL;
  //inst.accessq_empty()返回当前指令的访存操作的列表是否为空。
  if (inst.accessq_empty()) return result;

  //在V100中，m_config->m_L1D_config.l1_latency被配置为20拍，指的是L1数据缓存命中时的访问拍数。
  if (m_config->m_L1D_config.l1_latency > 0) {
    //在V100中，m_config->m_L1D_config.l1_banks被配置为4，指的是L1数据缓存的bank数，即每拍能够
    //处理的请求数最大不超过4。
    for (unsigned int j = 0; j < m_config->m_L1D_config.l1_banks;
         j++) {  // We can handle at max l1_banks reqs per cycle

      //inst.accessq_empty()返回当前指令的访存操作的列表是否为空。
      if (inst.accessq_empty()) return result;

      mem_fetch *mf =
          m_mf_allocator->alloc(inst, inst.accessq_back(),
                                m_core->get_gpu()->gpu_sim_cycle +
                                    m_core->get_gpu()->gpu_tot_sim_cycle);
      unsigned bank_id = m_config->m_L1D_config.set_bank(mf->get_addr());
      assert(bank_id < m_config->m_L1D_config.l1_banks);
      //l1_latency_queue的定义为：
      //    std::vector<std::deque<mem_fetch *>> l1_latency_queue;
      //初始化为：
      //    l1_latency_queue.resize(m_config->m_L1D_config.l1_banks);
      //    for (unsigned j = 0; j < m_config->m_L1D_config.l1_banks; j++)
      //      l1_latency_queue[j].resize(m_config->m_L1D_config.l1_latency,
      //                                 (mem_fetch *)NULL);
      //其是个二维数组，第一维的索引为bank_id，第二维的长度为20，每个元素都是(mem_fetch *)对象。
      //l1_latency_queue用于模拟L1数据缓存的访问延迟，在V100的配置中，m_config->m_L1D_config.
      //l1_latency被配置为20拍，指的是L1数据缓存命中时的访问拍数。4个Bank每个Bank可以处理至多20
      //个访问请求，例如第0号Bank，l1_latency_queue[0]里存储了20个访问请求，当新的访问请求到来
      //时，请求直接加到l1_latency_queue[0][19]位置，并且l1_latency_queue[0][0]的请求代表它已
      //经处理完毕。每一拍，l1_latency_queue[0]中的请求都会向前移动一个位置，即l1_latency_queue
      //[0][0]的请求会被删除，l1_latency_queue[0][19]的请求会被移动到l1_latency_queue[0][18]
      //的位置，l1_latency_queue[0][18]的请求会被移动到l1_latency_queue[0][17]的位置，以此类推。
      if ((l1_latency_queue[bank_id][m_config->m_L1D_config.l1_latency - 1]) ==
          NULL) {
        //这里是如果l1_latency_queue[bank_id][19]为空的话，就将新的请求写入到l1_latency_queue
        //[bank_id][19]的位置。
        l1_latency_queue[bank_id][m_config->m_L1D_config.l1_latency - 1] = mf;

        //如果mf所属指令是存储操作。
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
        //这里是如果l1_latency_queue[bank_id][19]不为空的话，则是发生Bank conflict，需要等待。
        result = BK_CONF;
        m_stats->gpgpu_n_l1cache_bkconflict++;
        delete mf;
        break;  // do not try again, just break from the loop and try the next
                // cycle
      }
    }
    //inst.accessq_empty()返回当前指令的访存操作的列表是否为空。不为空的话，且不是因为Bank冲突
    //导致Stall，则发生coalescing stall。
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
  //对所有的L1D的bank进行循环。
  for (unsigned int j = 0; j < m_config->m_L1D_config.l1_banks; j++) {
    //l1_latency_queue的定义为：
    //    std::vector<std::deque<mem_fetch *>> l1_latency_queue;
    //初始化为：
    //    l1_latency_queue.resize(m_config->m_L1D_config.l1_banks);
    //    for (unsigned j = 0; j < m_config->m_L1D_config.l1_banks; j++)
    //      l1_latency_queue[j].resize(m_config->m_L1D_config.l1_latency,
    //                                 (mem_fetch *)NULL);
    //其是个二维数组，第一维的索引为bank_id，第二维的长度为20，每个元素都是(mem_fetch *)对象。
    //l1_latency_queue用于模拟L1数据缓存的访问延迟，在V100的配置中，m_config->m_L1D_config.
    //l1_latency被配置为20拍，指的是L1数据缓存命中时的访问拍数。4个Bank每个Bank可以处理至多20
    //个访问请求，例如第0号Bank，l1_latency_queue[0]里存储了20个访问请求，当新的访问请求到来
    //时，请求直接加到l1_latency_queue[0][19]位置，并且l1_latency_queue[0][0]的请求代表它已
    //经处理完毕。每一拍，l1_latency_queue[0]中的请求都会向前移动一个位置，即l1_latency_queue
    //[0][0]的请求会被删除，l1_latency_queue[0][19]的请求会被移动到l1_latency_queue[0][18]
    //的位置，l1_latency_queue[0][18]的请求会被移动到l1_latency_queue[0][17]的位置，以此类推。

    //第j个bank的第0号位置(l1_latency_queue[j][0])不为空的话，说明有一个已经完成的L1D Cache访
    //问mf_next。
    if ((l1_latency_queue[j][0]) != NULL) {
      mem_fetch *mf_next = l1_latency_queue[j][0];
      std::list<cache_event> events;
      //m_L1D访问状态。
      enum cache_request_status status =
          m_L1D->access(mf_next->get_addr(), mf_next,
                        m_core->get_gpu()->gpu_sim_cycle +
                            m_core->get_gpu()->gpu_tot_sim_cycle,
                        events);

      //判断一系列的访问cache事件是否存在WRITE_REQUEST_SENT。
      bool write_sent = was_write_sent(events);
      //判断一系列的访问cache事件是否存在READ_REQUEST_SENT。
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
判断LDST单元访问global/local/param memory是否会Stall。Stall的话返回0，没有Stall的话返回1。
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

  //bypassL1D是指代数据访问是否绕过L1数据缓存。
  bool bypassL1D = false;
  if (CACHE_GLOBAL == inst.cache_op || (m_L1D == NULL)) {
    //CACHE_GLOBAL==.cg，Cache at global level，全局级缓存（L2及以下缓存，而不是L1）。使用ld.cg全局性
    //地加载，绕过L1缓存，并仅缓存在L2缓存中。
    bypassL1D = true;
  } else if (inst.space.is_global()) {  // global memory access
    // skip L1 cache if the option is enabled
    //在V100配置中，gpgpu_gmem_skip_L1D设置为0。
    if (m_core->get_config()->gmem_skip_L1D && (CACHE_L1 != inst.cache_op))
      bypassL1D = true;
  }
  //所以只有在使用ld.cg时才会绕过L1数据缓存。
  if (bypassL1D) {
    // bypass L1 cache
    unsigned control_size =
        inst.is_store() ? WRITE_PACKET_SIZE : READ_PACKET_SIZE;
    unsigned size = access.get_size() + control_size;
    // printf("Interconnect:Addr: %x, size=%d\n",access.get_addr(),size);
    if (m_icnt->full(size, inst.is_store() || inst.isatomic())) {
      //ICNT堵塞。
      stall_cond = ICNT_RC_FAIL;
    } else {
      mem_fetch *mf =
          m_mf_allocator->alloc(inst, access,
                                m_core->get_gpu()->gpu_sim_cycle +
                                    m_core->get_gpu()->gpu_tot_sim_cycle);
      //将mf放入ICNT的发送队列中。
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
    //处理对L1数据缓存的内存访问。
    stall_cond = process_memory_access_queue_l1cache(m_L1D, inst);
  }
  //inst.accessq_empty()返回当前指令的访存操作的列表是否为空。这里是当前指令的访存操作的列表不为空，且
  //没有发生STALL，则说明发生了Coalescing Stall。
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
LD/ST单元的响应FIFO中的数据包数 >= GPU配置的响应队列中的最大响应包数。这里需要注意的是，LD/ST单元也有一个
m_response_fifo，且m_response_fifo.size()获取的是该fifo已经存储的mf数目，这个数目能够判断该fifo是否已满，
m_config->ldst_unit_response_queue_size则是配置的该fifo的最大容量，一旦m_response_fifo.size()等于配置
的最大容量，就会返回True，表示该fifo已满。
*/
bool ldst_unit::response_buffer_full() const {
  return m_response_fifo.size() >= m_config->ldst_unit_response_queue_size;
}

/*
将mem_fetch *mf放入SIMT Core集群的数据响应FIFO。
*/
void ldst_unit::fill(mem_fetch *mf) {
  mf->set_status(
      IN_SHADER_LDST_RESPONSE_FIFO, //标记是进入Shader Core的LDST单元响应FIFO的mf
      m_core->get_gpu()->gpu_sim_cycle + m_core->get_gpu()->gpu_tot_sim_cycle);
  //这里的m_response_fifo是SIMT Core集群的LDST单元内的响应FIFO，定义为：
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
issue(warp_inst_t*&)成员函数将给定的流水线寄存器的内容移入m_dispatch_reg。然后指令在m_dispatch_reg等待
initiation_interval个周期。在此期间，没有其他的指令可以发到这个单元，所以这个等待是指令的吞吐量的模型。
*/
simd_function_unit::simd_function_unit(const shader_core_config *config) {
  m_config = config;
  //m_dispatch_reg其实是个缓冲，保存输入的指令，等待执行。当指令从设置到功能单元的OC_EX寄存器发出时，它被
  //保存在调度寄存器中。dispatch register是其中一种类型的寄存器。dispatch register记录着指令，会在被后续
  //执行时被传递给执行单元。具体来说，dispatch register是一个包含多个字段的结构体，其中包括指令的目的地址、
  //数据类型、操作码等信息。dispatch register可以看作是指令调度过程中传递数据的重要寄存器。
  m_dispatch_reg = new warp_inst_t(config);
}

/*
issue(warp_inst_t*&)成员函数将给定的流水线寄存器的内容移入m_dispatch_reg。
*/
void simd_function_unit::issue(register_set &source_reg) {
  //在simd_function_unit实现中，is_issue_partitioned()是虚拟函数，除LDST单元外的其他计算单元均返回True。
  //m_config->sub_core_model为True。
  bool partition_issue =
      m_config->sub_core_model && this->is_issue_partitioned();
  //source_reg即为流水线寄存器，目的是找到一个非空的指令，将其移入m_dispatch_reg。
  source_reg.move_out_to(partition_issue, this->get_issue_reg_id(),
                         m_dispatch_reg);
  //设置m_dispatch_reg的标识占用位图的状态，m_dispatch_reg是warp_inst_t类型，可获取该指令的延迟。
  occupied.set(m_dispatch_reg->latency);
}

/*
SFU特殊功能单元的构造函数。仅m_name不同。
*/
sfu::sfu(register_set *result_port, const shader_core_config *config,
         shader_core_ctx *core, unsigned issue_reg_id)
    : pipelined_simd_unit(result_port, config, config->max_sfu_latency, core,
                          issue_reg_id) {
  m_name = "SFU";
}

/*
Tensor Core单元的构造函数。仅m_name不同。
*/
tensor_core::tensor_core(register_set *result_port,
                         const shader_core_config *config,
                         shader_core_ctx *core, unsigned issue_reg_id)
    : pipelined_simd_unit(result_port, config, config->max_tensor_core_latency,
                          core, issue_reg_id) {
  m_name = "TENSOR_CORE";
}

/*
SFU特殊功能单元的发射函数。
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
Tensor Core单元的发射函数。
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
lane的意思为一个warp中有32个线程，而在流水线寄存器中可能暂存了很多条指令，这些指令的每对应的线程掩码的每一
位都是一个lane。即遍历流水线寄存器中的非空指令，返回所有指令的整体线程掩码（所有指令线程掩码的或值）。
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
SP单元的构造函数。仅m_name不同。
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
DP单元的构造函数。仅m_name不同。
*/
dp_unit::dp_unit(register_set *result_port, const shader_core_config *config,
                 shader_core_ctx *core, unsigned issue_reg_id)
    : pipelined_simd_unit(result_port, config, config->max_dp_latency, core,
                          issue_reg_id) {
  m_name = "DP ";
}

/*
INT单元的构造函数。仅m_name不同。
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
流水线单元构造函数。
*/
pipelined_simd_unit::pipelined_simd_unit(register_set *result_port,
                                         const shader_core_config *config,
                                         unsigned max_latency,
                                         shader_core_ctx *core,
                                         unsigned issue_reg_id)
    : simd_function_unit(config) {
  m_result_port = result_port;
  m_pipeline_depth = max_latency;
  //m_pipeline_reg是一个数组，该数组的大小模拟流水线的深度，每个元素是一个warp_inst_t类型的指针。
  m_pipeline_reg = new warp_inst_t *[m_pipeline_depth];
  for (unsigned i = 0; i < m_pipeline_depth; i++)
    m_pipeline_reg[i] = new warp_inst_t(config);
  m_core = core;
  m_issue_reg_id = issue_reg_id;
  active_insts_in_pipeline = 0;
}

/*
流水线单元向前推进一拍。
m_pipeline_reg的定义：warp_inst_t **m_pipeline_reg;
它做以下事情：
1. 如果dispatch_reg不为空，并且dispatch delay已完成，则上下文将从调度寄存器移动到流水线的调度延迟阶段。
2. 在内部流水线寄存器之间移动指令。
3. 如果最后一个内部流水线寄存器不为空，则将其发送到输出端口（通常为EX_WB流水线寄存器集）。

Dispatch Delay：
在V100的trace.config文件中：
    #tensor unit
    -specialized_unit_3 1,4,8,4,4,TENSOR
    -trace_opcode_latency_initiation_spec_op_3 8,4      # <latency,initiation>

在第二行中，它有两个值：8和4。前者是latency，后者是initiation_interval。initiation_interval是调度延迟，
latency是管道中管道阶段的数量。换句话说，每次initiation_interval都可以向功能单元发出指令。指令通过功能单
元需要等待周期。此处的start_stage：
    int start_stage = m_dispatch_reg->latency - m_dispatch_reg->initiation_interval;
说明指令在每个周期都要经过流水线阶段，但调度单元的上下文在initiation_interval周期中不能更改。

☆ 即在V100中，这里spec_op_3类型的指令执行需要8拍，在发射指令后，指令被移入m_dispatch_reg调度寄存器，然后
在调度寄存器里等待4拍，这4拍内不允许别的spec_op_3类型指令进入调度寄存器，4拍后，该指令被移入m_pipeline_reg
的第8-4=4号槽，然后在m_pipeline_reg中等待4槽->3槽->2槽->1槽共4拍后，该指令被移入EX_WB流水线寄存器集。
*/
void pipelined_simd_unit::cycle() {
  // pipeline reg 0 is not empty
  //从下面的move_warp(m_pipeline_reg[stage], m_pipeline_reg[stage + 1])可以看出，m_pipeline_reg[0]
  //是模拟流水线深度即执行延迟的最后一个槽，因此，在判断m_pipeline_reg是否向EX_WB阶段发出时，要看第0号槽
  //是否为空。
  if (!m_pipeline_reg[0]->empty()) {
    // put m_pipeline_reg[0] to the EX_WB reg
    //将m_pipeline_reg[0]中的执行移入EX_WB流水线寄存器集。
    m_result_port->move_in(m_pipeline_reg[0]);
    assert(active_insts_in_pipeline > 0);
    //m_pipeline_reg[0]中的指令移出后，流水线中的活跃指令数减1。
    active_insts_in_pipeline--;
  }
  // move warp_inst_t through out the pipeline
  //m_pipeline_reg流水线寄存器集中的所有指令向前推进一槽，模拟一拍的执行。
  if (active_insts_in_pipeline) {
    for (unsigned stage = 0; (stage + 1) < m_pipeline_depth; stage++)
      move_warp(m_pipeline_reg[stage], m_pipeline_reg[stage + 1]);
  }
  // If the dispatch_reg is not empty
  //如果dispatch_reg不为空，则将其移入m_pipeline_reg流水线。
  //具体移入哪个位置，要用指令延迟减去在m_dispatch_reg中的初始间隔，这个初始间隔是依据指令的吞吐量设置的。
  if (!m_dispatch_reg->empty()) {
    // If not dispatch_delay
    if (!m_dispatch_reg->dispatch_delay()) {
      // during dispatch delay, the warp is still moving through the pipeline,
      // though the dispatch_reg cannot be changed
      int start_stage =
          m_dispatch_reg->latency - m_dispatch_reg->initiation_interval;
      if (m_pipeline_reg[start_stage]->empty()) {
      //从m_dispatch_reg移入m_pipeline_reg流水线。
        move_warp(m_pipeline_reg[start_stage], m_dispatch_reg);
      //指令移入m_pipeline_reg后，流水线中的活跃指令数减1。
        active_insts_in_pipeline++;
      }
    }
  }
  //m_dispatch_reg的标识占用位图的状态右移一位，模拟一拍的推进。
  occupied >>= 1;
}

/*
将warp_inst_t类型的指令移入dispatch寄存器。
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
ldst单元的初始化函数。
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
  //L1纹理缓存。
  m_L1T = new tex_cache(L1T_name, m_config->m_L1T_config, m_sid,
                        get_shader_texture_cache_id(), icnt, IN_L1T_MISS_QUEUE,
                        IN_SHADER_L1T_ROB);
  //L1常量缓存。
  m_L1C = new read_only_cache(L1C_name, m_config->m_L1C_config, m_sid,
                              get_shader_constant_cache_id(), icnt,
                              IN_L1C_MISS_QUEUE, OTHER_GPU_CACHE, m_gpu);
  //L1数据缓存。
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
ldst 单元的构造函数。
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
LDST 单元 issue 函数。
*/
void ldst_unit::issue(register_set &reg_set) {
  // 从寄存器集合 reg_set 中获取一个非空寄存器，将其 inst 指令移出，并返回这条指令。
  warp_inst_t *inst = *(reg_set.get_ready());

  // record how many pending register writes/memory accesses there are for this
  // instruction
  // 记录该指令有多少个挂起的寄存器写入/内存访问。
  assert(inst->empty() == false);
  // 如果该指令是 load 指令，并且该指令的空间类型不是共享内存空间。
  if (inst->is_load() and inst->space.get_type() != shared_space) {
    // 该指令的 warp_id。
    unsigned warp_id = inst->warp_id();
    // 该指令的访存次数。inst->accessq_count() = m_accessq.size()，其中 m_accessq 
    // 是当前指令的访存操作的列表。
    unsigned n_accesses = inst->accessq_count();
    // 对该条指令的所有目的寄存器循环。
    for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++) {
      // 该条指令的目的寄存器 ID。inst->out 记录了当前指令的所有目的操作数寄存器 ID。
      unsigned reg_id = inst->out[r];
      if (reg_id > 0) {
        // m_pending_writes 是一个二维数组，第一维索引是 warp_id，第二维索引是寄存
        // 器 ID，值是该寄存器 ID 对应的寄存器挂起的写入次数。该指令的目的寄存器 ID 
        // 对应的 m_pending_writes 表项的挂起的写入次数加上该指令的写次数。
        m_pending_writes[warp_id][reg_id] += n_accesses;
      }
    }
    if (inst->m_is_ldgsts) {
      m_pending_ldgsts[warp_id][inst->pc][inst->get_addr(0)] += n_accesses;
    }
  }
  // inst->op_pipe 是操作码 code (uarch 可见)，标识操作的流水线(SP、SFU 或 MEM)。
  inst->op_pipe = MEM__OP;
  // stat collection
  m_core->mem_instruction_stats(*inst);
  m_core->incmem_stat(m_core->get_config()->warp_size, 1);
  //LDST单元流水线单元向前推进一拍。
  pipelined_simd_unit::issue(reg_set);
}

/*
LDST 单元的写回操作。
*/
void ldst_unit::writeback() {
  // process next instruction that is going to writeback
  // 下一条需要写回的指令。
  // m_next_wb => *pipline_reg[0]
  if (!m_next_wb.empty()) {
    // 操作数收集器的 Bank 写回，返回 true；Bank 冲突时返回 false。
    // if (m_operand_collector->writeback(m_next_wb)) {                      // yangjianchao16 del
    if (m_operand_collector->writeback(m_next_wb)) {
      bool insn_completed = false;
      // 对该条指令 m_next_wb 的所有目的寄存器循环。
      for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++) {
        if (m_next_wb.out[r] > 0) {
          // 检查指令写回的地址是否是共享内存空间。
          if (m_next_wb.space.get_type() != shared_space) {
            //指令写回的地址不是共享内存空间。
            assert(m_pending_writes[m_next_wb.warp_id()][m_next_wb.out[r]] > 0);
            //这里的m_pending_writes是一个二维数组，第一维索引是warp_id，第二维索引是寄存器ID，值是该寄
            //存器ID对应的寄存器挂起的写入次数。由于这里执行了操作数收集器的写回操作，m_next_wb所在warp下
            //该寄存器ID对应的寄存器挂起的写入次数减1。still_pending则是在当前写回执行完毕后，m_next_wb
            //所在warp下该寄存器ID对应的寄存器挂起的写入次数。
            //在某条指令访问非shared memory空间时，需要将该指令的所有目的寄存器都写回，在执行ldst_unit::
            //issue()函数时，该指令的所有目的寄存器需写回次数都会被加到m_pending_writes中，因此，该指令的
            //所有目的寄存器都会在m_pending_writes中有表项，这些表项的值都是该指令的访存次数。
            unsigned still_pending =
                --m_pending_writes[m_next_wb.warp_id()][m_next_wb.out[r]];
            //如果该挂起写入次数已经减为0，则说明已经没有挂起的写入，该寄存器所在的m_pending_writes中的表
            //项可以被释放，该寄存器也应该从计分板中释放。
            if (!still_pending) {
              m_pending_writes[m_next_wb.warp_id()].erase(m_next_wb.out[r]);
              m_scoreboard->releaseRegister(m_next_wb.warp_id(),
                                            m_next_wb.out[r]);
              //设置指令已完成。
              insn_completed = true;
            }
          } else {  // shared
            // 指令写回的地址是共享内存空间，没有寄存器参与，这时候直接从计分板中释放该寄存器即可。
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
      //如果操作数收集器的写回操作完成，代表当前指令已经完成。
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
  //m_num_writeback_clients = 5：
  //  next_client=0：shared memory,
  //  next_client=1：L1T,   
  //  next_client=2：L1C,
  //  next_client=3：global/local (uncached), 
  //  next_client=4：L1D。
  // 下面是获取下一个需要写回的指令。这里轮询查找上述5个client，从中选择一条需写回的指令。
  for (unsigned c = 0; m_next_wb.empty() && (c < m_num_writeback_clients);
       c++) {
    // m_writeback_arb 是一个计数器，用于轮询查找下一个需要写回的单元。下一个查找的单元是
    // next_client。
    unsigned next_client = (c + m_writeback_arb) % m_num_writeback_clients;
    switch (next_client) {
      // next_client can be chosen as: 
      //     0 - shared memory, 
      //     3 - global/local (uncached), 
      //     4 - L1D, 
      //     1 - L1T, 
      //     2 - L1C.
      case 0:  // shared memory
        // 直接判断流水线寄存器即可。需要注意的是，这里只有 shared memory 的指令才会放到 
        // m_pipeline_reg 中模拟延迟，其他类型的访存指令例如 L1T/L1C/L1D/global/local 
        // 都有专门模拟他们的专门部件。
        if (!m_pipeline_reg[0]->empty()) {
          // 将需要写回的指令放入 m_next_wb 中，在下一拍的 ldst_unit::write_back() 开始
          // 会完成这个指令的写回操作。
          m_next_wb = *m_pipeline_reg[0];
          // 在 cuda-sim.cc 中定义的 ptx_thread_info::ptx_exec_inst 函数中会确定当前
          // 指令的操作码是不是 ATOM_OP 操作码，如果是则为 atomic 操作。
          if (m_next_wb.isatomic()) {
            /* do_atomic 函数定义为：
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
              当前版本将 should_do_atomic 始终置为 true，所以这里的 do_atomic 无效。即
              gpgpusim 暂时不模拟 atomic 操作。
            */
            m_next_wb.do_atomic();
            m_core->decrement_atomic_count(m_next_wb.warp_id(),
                                           m_next_wb.active_count());
          }
          m_core->dec_inst_in_pipeline(m_pipeline_reg[0]->warp_id());
          // 由于前面对 shared memory 的访存指令会在 m_dispatch_reg 中模拟考虑 bank 每
          // 拍只能提供一个 word 的访问，再在 m_pipeline_reg 中模拟对于 shared memory 
          // 的访问延迟。后者是 shared memory 的访存延迟，这个延迟是由 gpgpu_l1_latency 
          // 配置参数决定的，这个固有延迟在执行流水线模型的 m_pipeline_reg 中模拟。即当一
          // 条访存 shared memory 指令经过 initial interval 后（模拟多 bank conflict），
          // 会被放进 m_pipeline_reg[gpgpu_l1_latency-1]，当它出现在 m_pipeline_reg[0] 
          // 时，就说明这个指令已经完成了访存操作，可以写回了。

          // 但是这里的写回操作没有考虑 shared memory load 操作的结果需要写到寄存器。
          m_pipeline_reg[0]->clear();
          serviced_client = next_client;
        }
        break;
      case 1:  // texture response
        if (m_L1T->access_ready()) {
          // 获取下一个访问的纹理指令。
          mem_fetch *mf = m_L1T->next_access();
          // 将需要写回的指令放入 m_next_wb 中，在下一拍的 ldst_unit::write_back() 开始
          // 会完成这个指令的写回操作。
          m_next_wb = mf->get_inst();
          delete mf;
          serviced_client = next_client;
        }
        break;
      case 2:  // const cache response
        if (m_L1C->access_ready()) {
          // 获取下一个访问的常量缓存指令。
          mem_fetch *mf = m_L1C->next_access();
          // 将需要写回的指令放入 m_next_wb 中，在下一拍的 ldst_unit::write_back() 开始
          // 会完成这个指令的写回操作。
          m_next_wb = mf->get_inst();
          delete mf;
          serviced_client = next_client;
        }
        break;
      case 3:  // global/local
        if (m_next_global) {
          // 获取下一个访问的全局内存指令。在 ldst_unit::cycle() 函数中，当 ldst_unit 中
          // 的 m_response_fifo 非空，且其中的 mem_fetch 指令是 global/local 内存指令时，
          // 会将该指令放入 m_next_global 中。
          // 将需要写回的指令放入 m_next_wb 中，在下一拍的 ldst_unit::write_back() 开始
          // 会完成这个指令的写回操作。
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
          // 获取下一个访问的数据缓存指令。
          mem_fetch *mf = m_L1D->next_access();
          // 将需要写回的指令放入 m_next_wb 中，在下一拍的 ldst_unit::write_back() 开始
          // 会完成这个指令的写回操作。
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

// 时钟倍增器：一些单元可能在更高的循环速率下运行。
unsigned ldst_unit::clock_multiplier() const {
  // to model multiple read port, we give multiple cycles for the memory units
  // 在 V100 配置中，m_config->mem_unit_ports 默认为 1。
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
  // LDST 单元的写回操作。
  writeback();
  // for (int i = 0; i < m_config->reg_file_port_throughput; ++i)
  //   m_operand_collector->step();
  // m_pipeline_reg 是一个数组，该数组的大小模拟流水线的深度 m_pipeline_depth，每个元素是
  // 一个 warp_inst_t 类型的指针。这里是 m_pipeline_reg 流水线寄存器集中的所有指令向前推进
  // 一槽，模拟一拍的执行。

  for (unsigned stage = 0; (stage + 1) < m_pipeline_depth; stage++)
    if (m_pipeline_reg[stage]->empty() && !m_pipeline_reg[stage + 1]->empty())
      move_warp(m_pipeline_reg[stage], m_pipeline_reg[stage + 1]);

  // 如果 LDST 单元的数据响应 FIFO 不为空。这里的 m_response_fifo 是属于 ldst_unit 
  // 类的一个成员变量。它是接收 SIMT Core Cluster 的 m_response_fifo 发送的 mf 访
  // 存数据包。SIMT Core Cluster 还会向 L1I Cache 发送 INST_ACC_R 类型的访存数据包。
  if (!m_response_fifo.empty()) {
    //获取FIFO中的第一个mem_fetch指针。
    mem_fetch *mf = m_response_fifo.front();
    if (mf->get_access_type() == TEXTURE_ACC_R) {
      //如果该访存mem是纹理访存读，则判断纹理缓存是否有空闲的填充端口。
      if (m_L1T->fill_port_free()) {
        //如果存在空闲的填充端口，则将该mem_fetch指针mf中的指令移入纹理缓存。
        m_L1T->fill(mf, m_core->get_gpu()->gpu_sim_cycle +
                            m_core->get_gpu()->gpu_tot_sim_cycle);
        m_response_fifo.pop_front();
      }
    } else if (mf->get_access_type() == CONST_ACC_R) {
      //如果该访存mem是常量访存读，则判断常量缓存是否有空闲的填充端口。
      if (m_L1C->fill_port_free()) {
        mf->set_status(IN_SHADER_FETCHED,
                       m_core->get_gpu()->gpu_sim_cycle +
                           m_core->get_gpu()->gpu_tot_sim_cycle);
        m_L1C->fill(mf, m_core->get_gpu()->gpu_sim_cycle +
                            m_core->get_gpu()->gpu_tot_sim_cycle);
        m_response_fifo.pop_front();
      }
    } else {
      //mf->get_type()返回的可能类型：
      //    enum mf_type {
      //      READ_REQUEST = 0,
      //      WRITE_REQUEST,
      //      READ_REPLY,  // send to shader
      //      WRITE_ACK
      //    };
      if (mf->get_type() == WRITE_ACK ||
          //m_config->gpgpu_perfect_mem在V100配置中默认为0。
          (m_config->gpgpu_perfect_mem && mf->get_is_write())) {
        //如果该访存mem是写响应，执行Shader Core（SM）对写确认的动作，减少mf所属warp的已发送但尚未收到
        //写确认的存储请求数。
        m_core->store_ack(mf);
        m_response_fifo.pop_front();
        delete mf;
      } else {
        //如果该访存mem不是写响应，则是READ_REQUEST，WRITE_REQUEST或READ_REPLY。
        assert(!mf->get_is_write());  // L1 cache is write evict, allocate line
                                      // on load miss only

        //bypassL1D是指代数据访问是否绕过L1数据缓存。
        bool bypassL1D = false;
        if (CACHE_GLOBAL == mf->get_inst().cache_op || (m_L1D == NULL)) {
          //CACHE_GLOBAL==.cg，Cache at global level，全局级缓存（L2及以下缓存，而不是L1）。使用ld.cg
          //全局性地加载，绕过L1缓存，并仅缓存在L2缓存中。
          bypassL1D = true;
        } else if (mf->get_access_type() == GLOBAL_ACC_R ||
                   mf->get_access_type() ==
                       GLOBAL_ACC_W) {  // global memory access
          //在V100配置中，gpgpu_gmem_skip_L1D设置为0。
          if (m_core->get_config()->gmem_skip_L1D) bypassL1D = true;
        }
        //所以只有在使用ld.cg时才会绕过L1数据缓存。
        if (bypassL1D) {
          // m_next_global 暂存下一次访问全局存储的 mf，定义：mem_fetch *ldst_unit::m_next_global。 
          if (m_next_global == NULL) {
            //m_next_global是下一次访问全局存储的mf，如果它空闲，则m_next_global = mf，空闲的话就无法
            //继续放出该次访问。
            mf->set_status(IN_SHADER_FETCHED,
                           m_core->get_gpu()->gpu_sim_cycle +
                               m_core->get_gpu()->gpu_tot_sim_cycle);
            m_response_fifo.pop_front();
            m_next_global = mf;
          }
        } else {
          //如果不绕过L1数据缓存，就需要将mf写入L1数据缓存的端口。
          if (m_L1D->fill_port_free()) {
            m_L1D->fill(mf, m_core->get_gpu()->gpu_sim_cycle +
                                m_core->get_gpu()->gpu_tot_sim_cycle);
            m_response_fifo.pop_front();
          }
        }
      }
    }
  }

  //L1纹理缓存向前推进一拍。
  m_L1T->cycle();
  //L1常量缓存向前推进一拍。
  m_L1C->cycle();
  if (m_L1D) {
    m_L1D->cycle();
    if (m_config->m_L1D_config.l1_latency > 0) L1_latency_queue_cycle();
  }

  // 每个执行单元模型（包括 INT/SP 等）都有一个 m_dispatch_reg。m_dispatch_reg 是
  // 一个 warp_inst_t 类型的指针，warp 调度器发射指令以及读取操作数完毕后，发射到每
  // 个执行单元的指令都会被放入到对应的执行单元的 m_dispatch_reg 中，然后指令从这个
  // m_dispatch_reg 中被取出，并置入模拟执行流水线延迟的 m_pipeline_reg 中。
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
  //mem_stage_stall_type的定义：
  //    enum mem_stage_stall_type { //MEM流水线步骤的Stall类型。
  //      NO_RC_FAIL = 0,           //没有Stall
  //      BK_CONF,                  //Bank conflict
  //      MSHR_RC_FAIL,             //MSHR资源冲突
  //      ICNT_RC_FAIL,             //Interconnect资源冲突
  //      COAL_STALL,               //Coalescing Stall
  //      TLB_STALL,                //TLB Stall
  //      DATA_PORT_STALL,          //数据端口Stall
  //      WB_ICNT_RC_FAIL,          //Writeback Interconnect资源冲突
  //      WB_CACHE_RSRV_FAIL,       //Writeback Cache资源冲突
  //      N_MEM_STAGE_STALL_TYPE    //Stall类型数量
  //    };
  enum mem_stage_stall_type rc_fail = NO_RC_FAIL;
  mem_stage_access_type type;
  bool done = true;
  //判断LDST单元访问shared memory是否会Stall。Stall的话返回0，没有Stall的话返回1。
  done &= shared_cycle(pipe_reg, rc_fail, type);
  done &= constant_cycle(pipe_reg, rc_fail, type);
  done &= texture_cycle(pipe_reg, rc_fail, type);
  //判断LDST单元访问global/local/param memory是否会Stall。Stall的话返回0，没有Stall的话返回1。
  done &= memory_cycle(pipe_reg, rc_fail, type);
  m_mem_rc = rc_fail;

  // 如果 done 为 0 说明前面发生了 stall。这里的 done 其实只会真正执行 shared_cycle，
  // constant_cycle，texture_cycle，memory_cycle 这四个函数中的一个，因为对于 LDST
  // 单元的 m_dispatch_reg 中的指令来说，其实它只可能是这四种访存类型中的一种。这里的
  // “真正执行”其实是指不会做相应动作，它还是需要进入这几个函数，这几个函数会判断传入的
  // 指令是否是对应的访存类型：如果是，就会执行相应的动作；如果不是，就会直接返回 true，
  // 表示不会在当前单元执行，即不执行动作。
  if (!done) {  // log stall types and return
    assert(rc_fail != NO_RC_FAIL);
    m_stats->gpgpu_n_stall_shd_mem++;
    m_stats->gpu_stall_shd_mem_breakdown[type][rc_fail]++;
    return;
  }

  //如果前面没有发生Stall且pipe_reg非空。
  if (!pipe_reg.empty()) {
    unsigned warp_id = pipe_reg.warp_id();
    //如果pipe_reg是load指令。
    if (pipe_reg.is_load()) {
      if (pipe_reg.space.get_type() == shared_space) {
        //如果pipe_reg是shared memory空间的load指令。
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
        //如果pipe_reg是global/local/param memory空间的load指令。
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
          //如果该指令没有挂起的写入，则该指令执行完毕。
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
      //如果pipe_reg不是load指令，则执行完毕。
      // stores exit pipeline here
      m_core->dec_inst_in_pipeline(warp_id);
      m_core->warp_inst_complete(*m_dispatch_reg);
      m_dispatch_reg->clear();
    }
  }
}

/*
注册CTA内线程的退出。
*/
void shader_core_ctx::register_cta_thread_exit(unsigned cta_num,
                                               kernel_info_t *kernel) {
  assert(m_cta_status[cta_num] > 0);
  //m_cta_status是Shader Core内的CTA的状态，MAX_CTA_PER_SHADER是每个Shader Core内的最大可并发
  //CTA个数。m_cta_status[i]里保存了第i个CTA中包含的活跃线程总数量，该数量 <= CTA的总线程数量。
  //这里由于需要注册单个线程的退出，因此第i个CTA中包含的活跃线程总数量应当减1。
  m_cta_status[cta_num]--;
  //如果m_cta_status[cta_num]=0即第cta_num号CTA内没有活跃的线程。
  if (!m_cta_status[cta_num]) {
    // Increment the completed CTAs
    //增加已经完成的CTA数量。
    m_stats->ctas_completed++;
    //增加已经完成的CTA数量。
    m_gpu->inc_completed_cta();
    //减小活跃的CTA数量。
    m_n_active_cta--;
    m_barriers.deallocate_barrier(cta_num);
    shader_CTA_count_unlog(m_sid, 1);

    SHADER_DPRINTF(
        LIVENESS,
        "GPGPU-Sim uArch: Finished CTA #%u (%lld,%lld), %u CTAs running\n",
        cta_num, m_gpu->gpu_sim_cycle, m_gpu->gpu_tot_sim_cycle,
        m_n_active_cta);
    //一旦没有活跃的CTA后，代表当前kernel已经执行完。
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
      //m_kernel是运行在当前SIMT Core上的内核函数。
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
线程块对SIMT Core的调度发生在shader_core_ctx::issue_block2core(...)。一个Core上可同时调度的最大线
程块（或称为CTA）的数量由函数shader_core_config::max_cta(...)计算。这个函数根据程序指定的每个线程块
的数量、每个线程寄存器的使用情况、共享内存的使用情况以及配置的每个Core最大线程块数量的限制，确定可以并
发分配给单个SIMT Core的最大线程块数量。具体说，如果上述每个标准都是限制因素，那么可以分配给SIMT Core
的线程块的数量被计算出来。其中的最小值就是可以分配给SIMT Core的最大线程块数。

该函数的参数：kernel_info_t为内核函数的信息类。kernel_info_t对象包含GPU网格和块维度、与内核入口点关
联的 function_info 对象以及为内核参数分配的内存。
*/
unsigned int shader_core_config::max_cta(const kernel_info_t &k) const {
  //k.threads_per_cta()返回每个线程块中的线程数量，threads_per_cta=m_block_dim.x * m_block_dim.y 
  //* m_block_dim.z。就是数定义在CUDA程序中的一个线程块上的线程数量。
  unsigned threads_per_cta = k.threads_per_cta();
  //k.entry()返回kernel的入口点，该入口点就是主函数。
  const class function_info *kernel = k.entry();
  //由于一个CTA/线程块中的线程必须要是 warp_size（一个warp中的线程数量）的整数倍，因此需要补足CTA中的
  //线程，让CTA中的线程数量达到 warp_size 的整数倍。
  unsigned int padded_cta_size = threads_per_cta;
  if (padded_cta_size % warp_size)
    padded_cta_size = ((padded_cta_size / warp_size) + 1) * (warp_size);

  // Limit by n_threads/shader
  //n_thread_per_shader是 -gpgpu_shader_core_pipeline 选项中的第一个值，第二个值是 warp_size。例如
  //在V100中的 -gpgpu_shader_core_pipeline 2048:32，即一个Shader Core（SM）中实现的是最大 64 warps/
  //SM，即一个 Shader Core（SM）中实现的最大可并发线程数量是 2048。
  //result_thread即为，由于[n_threads/shader]限制造成的单个SM内最大可并发CTA数量。
  unsigned int result_thread = n_thread_per_shader / padded_cta_size;
  //返回kernel的kernel_info。
  const struct gpgpu_ptx_sim_info *kernel_info = ptx_sim_kernel_info(kernel);

  // Limit by shmem/shader
  //gpgpu_shmem_size为每个SIMT Core（也称为Shader Core，SM）的共享存储大小。由于单个SM内每分配一个CTA，
  //则要求每个CTA具有独立的相同大小的shared memory，因此，result_shmem是由shared memory限制造成的单个
  //SM内最大可并发CTA数量。
  unsigned int result_shmem = (unsigned)-1;
  if (kernel_info->smem > 0)
    result_shmem = gpgpu_shmem_size / kernel_info->smem;

  // Limit by register count, rounded up to multiple of 4.
  //gpgpu_shader_registers是每个Shader Core的寄存器数。由于单个SM内每分配一个CTA，则要求每个CTA具有独
  //立的相同数量的寄存器，result_regs是由寄存器限制造成的单个SM内最大可并发CTA数量。
  unsigned int result_regs = (unsigned)-1;
  if (kernel_info->regs > 0)
    result_regs = gpgpu_shader_registers /
                  (padded_cta_size * ((kernel_info->regs + 3) & ~3));

  // Limit by CTA
  //max_cta_per_core是由硬件配置的SM内的最大可并发CTA数量。
  unsigned int result_cta = max_cta_per_core;

  //选择多个限制因素下，最小的SM内的CTA并发数量。
  unsigned result = result_thread;
  result = gs_min2(result, result_shmem);
  result = gs_min2(result, result_regs);
  result = gs_min2(result, result_cta);

  //将last_kinfo赋值为kernel_info。
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
  //gpu_max_cta_per_shader受cta数量的限制，如果不足以保持所有核忙碌。
  //k.num_blocks()返回CUDA代码中的Grid中的所有线程块的总数。num_shader()返回硬件所有的SM（又称Shader 
  //Core）的总数。如果上述计算的result*SM数量 > Grid中的所有线程块的总数，即result数量的CTA并不能使得
  //所有SM处于忙碌状态：例如，有10个SM，上面计算出的受限的最大CTA并发数result=5，但是CUDA代码中有40个
  //线程块，所以 40 < 5 * 10。为了让所有SM都忙碌起来，以充分利用所有的硬件资源，可以将上面计算出的受限
  //的最大CTA并发数再降低一些，以平均分配到每个SM上。
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
  //在V100 GPU中，可支持 adaptive_cache_config（适应性的cache配置），ADAPTIVE_VOLTA=1。
  //适应性的cache配置代表：在V100中，将剩余的不使用的shared memory划给L1 cache使用。
  //初始状态时，设置L1 cache适应性配置的标志为False。
  if (adaptive_cache_config && !k.cache_config_set) {
    // For more info about adaptive cache, see
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-7-x
    //计算当前单个SM中的所有CTA的共享存储的总量。
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
Shader Core/SIMT Core向前推进一个时钟周期。
*/
void shader_core_ctx::cycle() {
  //如果这个SIMT Core处于非活跃状态，且已经执行完成，时钟周期不向前推进。
  if (!isactive() && get_not_completed() == 0) return;

  m_stats->shader_cycles[m_sid]++;
  // printf("m_stats->shader_cycles[m_sid]: %u\n", m_stats->shader_cycles[m_sid]);
  //每个内核时钟周期，shader_core_ctx::cycle()都被调用，以模拟SIMT Core的一个周期。这个函数调用一
  //组成员函数，按相反的顺序模拟内核的流水线阶段，以模拟流水线效应：
  //     fetch()             倒数第一执行
  //     decode()            倒数第二执行
  //     issue()             倒数第三执行
  //     read_operand()      倒数第四执行
  //     execute()           倒数第五执行
  //     writeback()         倒数第六执行
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
操作数收集器的仲裁器，用于分配读操作。
*/
std::list<opndcoll_rfu_t::op_t> opndcoll_rfu_t::arbiter_t::allocate_reads() {
  //create a list of results
  //创建一个结果列表，(a)在不同的寄存器文件Bank，(b)不走向相同的操作数收集器。
  std::list<op_t>
      result;  // a list of registers that (a) are in different register banks,
               // (b) do not go to the same operand collector

  // input is the register bank
  // output is the collector units
  int input;
  int output;
  //_inputs是寄存器文件的Bank数。
  int _inputs = m_num_banks;
  //_outputs是操作数收集器的数量。
  int _outputs = m_num_collectors;
  //对应到MAP对角顺序检查图的最大维度，或长度或宽度。
  int _square = (_inputs > _outputs) ? _inputs : _outputs;
  assert(_square > 0);
  //_pri是上一次执行arbiter_t::allocate_reads()函数时，最后一个被分配的收集器单元的下一个收
  //集器单元的ID。
  int _pri = (int)m_last_cu;

  // Clear matching: set all the entries to -1
  //_inmatch[i]是第i个Bank的可匹配操作数收集器ID，如果该值不为-1或0，则说明它已经匹配到。
  for (int i = 0; i < _inputs; ++i) _inmatch[i] = -1;
  //_outmatch[j]是第j个操作数收集器的可匹配Bank的ID，如果该值不为-1或0，则说明它已经匹配到。
  for (int j = 0; j < _outputs; ++j) _outmatch[j] = -1;

  //对所有的寄存器文件Bank进行循环。
  for (unsigned i = 0; i < m_num_banks; i++) {
    //_request[m_num_banks][m_num_collectors]是指某个收集器单元对某个寄存器Bank是否有请求，
    //有则置1，无则置0。下面的一个for循环对所有的收集器单元和所有的寄存器Bank进行循环，设置所
    //有的请求数量为0。
    for (unsigned j = 0; j < m_num_collectors; j++) {
      assert(i < (unsigned)_inputs);
      assert(j < (unsigned)_outputs);
      //设置第j个收集器单元对第i个寄存器文件Bank的请求置0。
      _request[i][j] = 0;
    }
    //m_queue是一个以bank号来索引的操作数队列，m_queue[i]是第i个bank获取的操作数队列。如果这
    //个队列不为空，说明这个bank有操作数请求已经存入m_queue。
    if (!m_queue[i].empty()) {
      const op_t &op = m_queue[i].front();
      //op.get_oc_id()返回当前操作数所属的收集器单元的ID。
      int oc_id = op.get_oc_id();
      assert(i < (unsigned)_inputs);
      assert(oc_id < _outputs);
      //第oc_id个收集器单元对第i个寄存器文件Bank的请求数量置1。
      _request[i][oc_id] = 1;
    }
    //m_allocated_bank[i]用于存储第i个Bank的状态，包括NO_ALLOC, READ_ALLOC, WRITE_ALLOC。
    //如果第i个Bank是WRITE_ALLOC，说明这个Bank已经被分配给某个收集器单元，这个收集器单元正在
    //执行写操作，因此，这个Bank不能被分配给其他收集器单元。
    if (m_allocated_bank[i].is_write()) {
      assert(i < (unsigned)_inputs);
      //当第i个Bank是WRITE_ALLOC时，第i个Bank对于所有的收集器单元来说都不可分配为读操作，所以
      //这里设置_inmatch中的第i个Bank的可匹配状态为0。写操作具有更高的优先级。
      _inmatch[i] = 0;  // write gets priority
    }
  }

  ///// wavefront allocator from booksim... --->

  // Loop through diagonals of request matrix
  // printf("####\n");

  //对所有操作数收集器循环。这里检查的顺序是按照对角线检查的，例如，如果有5个收集器单元，3个Bank，
  //则_square=5，_outputs=5，_inputs=3，如果设置_pri=2的话，下面的两层for循环会按照下面的顺序
  //进行检查：
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
  //对应到MAP对角顺序检查图为：
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
    //_pri是上一次执行arbiter_t::allocate_reads()函数时，最后一个被分配的收集器单元的下一个收
    //集器单元的ID。这里是当前执行arbiter_t::allocate_reads()函数时，从上一次最后一个被分配的
    //收集器单元的下一个收集器单元的ID开始遍历，RR循环。
    output = (_pri + p) % _outputs;

    // Step through the current diagonal
    for (input = 0; input < _inputs; ++input) {
      assert(input < _inputs);
      assert(output < _outputs);
      //如果第input个Bank没有被分配给某个收集器单元，且第output个收集器单元对第input个Bank有请
      //求，那么就分配第input个Bank给第output个收集器单元。设置_inmatch中的第input个Bank的可匹
      //配收集器单元为output，设置_outmatch中的第output个收集器单元的可匹配Bank为input。
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
      //操作数收集器的编号递增。
      output = (output + 1) % _outputs;
    }
  }

  // Round-robin the priority diagonal
  //_pri是上一次执行arbiter_t::allocate_reads()函数时，最后一个被分配的收集器单元的下一个收集
  //器单元的ID。
  _pri = (_pri + 1) % _outputs;

  /// <--- end code from booksim
  //m_last_cu是上一次执行arbiter_t::allocate_reads()函数时，最后一个被分配的收集器单元的ID。
  m_last_cu = _pri;
  //对所有的寄存器文件Bank进行循环。
  for (unsigned i = 0; i < m_num_banks; i++) {
    //如果存在分配给第i号Bank的操作数收集器。
    if (_inmatch[i] != -1) {
      //判断第i号Bank是否已经被分配给某个收集器单元用于写操作，如果不是写操作的话（即为读操作），
      //则将对应读第i号Bank的请求队列m_queue中的首个操作数请求放入result。
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
为CTA分配屏障资源。当知道了某个CTA中包含了哪些warps，就可以分配用于cta_id标识的CTA屏障的资源。传入的参数：
    unsigned cta_id：CTA编号；
    warp_set_t warps：单个CTA内的所有warp数量大小的位图。
*/
void barrier_set_t::allocate_barrier(unsigned cta_id, warp_set_t warps) {
  assert(cta_id < m_max_cta_per_core);
  //m_cta_to_warps是Map<CTA ID，单个CTA内的所有warp数量大小的位图>。
  cta_to_warp_t::iterator w = m_cta_to_warps.find(cta_id);
  //在本函数运行之前，cta不应该已经是活跃的或已经分配了屏障资源。
  assert(w == m_cta_to_warps.end());  // cta should not already be active or
                                      // allocated barrier resources
  //赋值，在m_cta_to_warps里为cta_id标识的CTA赋值位图。
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
计算最大的每SM上CTA数量kernel_max_cta_per_shader，并且还要依据线程块的线程数量是否能对warp 
size取模运算，来计算padded每CTA的线程数量kernel_padded_threads_per_cta。
*/
void shader_core_ctx::set_max_cta(const kernel_info_t &kernel) {
  // calculate the max cta count and cta size for local memory address mapping
  //计算最大的每SM上CTA数量，以及padded每CTA的线程数量。
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
返回预取单元响应buffer是否已满。这里一直非满。
*/
bool shader_core_ctx::fetch_unit_response_buffer_full() const { return false; }

/*
SIMT Core接收预取的指令数据包，该数据包是mf定义的访存行为。mf定义的mem_access_type类型为：INST_ACC_R。
mem_access_type定义了在时序模拟器中对不同类型的存储器进行不同的访存类型：
     MA_TUP(GLOBAL_ACC_R),        从global memory读
     MA_TUP(LOCAL_ACC_R),         从local memory读
     MA_TUP(CONST_ACC_R),         从常量缓存读
     MA_TUP(TEXTURE_ACC_R),       从纹理缓存读
     MA_TUP(GLOBAL_ACC_W),        向global memory写
     MA_TUP(LOCAL_ACC_W),         向local memory写
  在V100中，L1 cache的m_write_policy为WRITE_THROUGH，实际上L1_WRBK_ACC也不会用到：
     MA_TUP(L1_WRBK_ACC),         L1缓存write back
  在V100中，当L2 cache写不命中时，采取lazy_fetch_on_read策略，当找到一个cache block
  逐出时，如果这个cache block是被MODIFIED，则需要将这个cache block写回到下一级存储，
  因此会产生L2_WRBK_ACC访问，这个访问就是为了写回被逐出的MODIFIED cache block。
     MA_TUP(L2_WRBK_ACC),         L2缓存write back
     MA_TUP(INST_ACC_R),          从指令缓存读
  L1_WR_ALLOC_R/L2_WR_ALLOC_R在V100配置中暂时用不到：
     MA_TUP(L1_WR_ALLOC_R),       L1缓存write-allocate（cache写不命中，将主存中块调入cache，
                                  写入该cache块）
  L1_WR_ALLOC_R/L2_WR_ALLOC_R在V100配置中暂时用不到：
     MA_TUP(L2_WR_ALLOC_R),       L2缓存write-allocate
*/
void shader_core_ctx::accept_fetch_response(mem_fetch *mf) {
  mf->set_status(IN_SHADER_FETCHED,
                 m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
  //m_L1I是指令的L1-Cache。将mf填入到m_L1I中。
  m_L1I->fill(mf, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
}

/*
返回LDST单元响应buffer是否已满。LD/ST单元的响应FIFO中的数据包数 >= GPU配置的响应队列中的最大响应包数。这
里需要注意的是，LD/ST单元也有一个m_response_fifo，且m_response_fifo.size()获取的是该fifo已经存储的mf数
目，这个数目能够判断该fifo是否已满，m_config->ldst_unit_response_queue_size则是配置的该fifo的最大容量，
一旦m_response_fifo.size()等于配置的最大容量，就会返回True，表示该fifo已满。
*/
bool shader_core_ctx::ldst_unit_response_buffer_full() const {
  return m_ldst_unit->response_buffer_full();
}

/*
SIMT Core集群接收预取的data数据包。mf定义的mem_access_type类型不为INST_ACC_R。
mem_access_type定义了在时序模拟器中对不同类型的存储器进行不同的访存类型：
     MA_TUP(GLOBAL_ACC_R),        从global memory读
     MA_TUP(LOCAL_ACC_R),         从local memory读
     MA_TUP(CONST_ACC_R),         从常量缓存读
     MA_TUP(TEXTURE_ACC_R),       从纹理缓存读
     MA_TUP(GLOBAL_ACC_W),        向global memory写
     MA_TUP(LOCAL_ACC_W),         向local memory写
  在V100中，L1 cache的m_write_policy为WRITE_THROUGH，实际上L1_WRBK_ACC也不会用到：
     MA_TUP(L1_WRBK_ACC),         L1缓存write back
  在V100中，当L2 cache写不命中时，采取lazy_fetch_on_read策略，当找到一个cache block
  逐出时，如果这个cache block是被MODIFIED，则需要将这个cache block写回到下一级存储，
  因此会产生L2_WRBK_ACC访问，这个访问就是为了写回被逐出的MODIFIED cache block。
     MA_TUP(L2_WRBK_ACC),         L2缓存write back
     MA_TUP(INST_ACC_R),          从指令缓存读
  L1_WR_ALLOC_R/L2_WR_ALLOC_R在V100配置中暂时用不到：
     MA_TUP(L1_WR_ALLOC_R),       L1缓存write-allocate（cache写不命中，将主存中块调入cache，
                                  写入该cache块）
  L1_WR_ALLOC_R/L2_WR_ALLOC_R在V100配置中暂时用不到：
     MA_TUP(L2_WR_ALLOC_R),       L2缓存write-allocate
*/
void shader_core_ctx::accept_ldst_unit_response(mem_fetch *mf) {
  m_ldst_unit->fill(mf);
}

/*
Shader Core（SM）对写确认的动作，减少mf所属warp的已发送但尚未收到写确认的存储请求数。
*/
void shader_core_ctx::store_ack(class mem_fetch *mf) {
  assert(mf->get_type() == WRITE_ACK ||
         (m_config->gpgpu_perfect_mem && mf->get_is_write()));
  unsigned warp_id = mf->get_wid();
  //减少已发送但尚未收到写确认的存储请求数。
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
返回绑定在当前Shader Core的kernel的内核函数信息，kernel_info_t对象。
*/
kernel_info_t *shd_warp_t::get_kernel_info() const {
  return m_shader->get_kernel_info();
}
/*
返回warp已经执行完毕的标志，已经完成的线程数量=warp的大小时，就代表该warp已经完成。
*/
bool shd_warp_t::functional_done() const {
  return get_n_completed() == m_warp_size;
}

/*
这段代码检查这个warp是否已经完成执行并且可以回收。
*/
bool shd_warp_t::hardware_done() const {
  //functional_done()返回warp已经执行完毕的标志，已经完成的线程数量=warp的大小时，就代表该warp已经
  //完成。stores_done()返回所有store访存请求是否已经全部执行完，已发送但尚未确认的写存储请求（已经发
  //出写请求，但还没收到写确认信号时）数m_stores_outstanding=0时，代表所有store访存请求已经全部执行
  //完，这里m_stores_outstanding在发出一个写请求时+=1，在收到一个写确认时-=1。
  //m_inst_fetch_buffer中含有效指令时且将该指令解码过程中填充进warp的m_ibuffer时，增加在流水线中的
  //指令数m_inst_in_pipeline（注意这里在decode()函数中会向warp的m_ibuffer填充进2条指令）；在指令完
  //成写回操作时减少在流水线中的指令数m_inst_in_pipeline。inst_in_pipeline()返回流水线中的指令数量
  //m_inst_in_pipeline。

  //这里一个warp完成的标志由三个条件组成，分别是：1、warp已经执行完毕；2、所有store访存请求已经全部执
  //行完；3、流水线中的指令数量为0。
  return functional_done() && stores_done() && !inst_in_pipeline();
}

/*
返回warp是否由于（warp已经执行完毕且在等待新内核初始化、CTA处于barrier、memory barrier、还有未完成
的原子操作）四个条件处于等待状态。
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
debug: dp 0 0 0 (第一个0代表mask，填零即可。第二个0代表第0个shader，第三个0代表第0个内存子分区)。
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
增加collector unit的数量。这里的set_id定义为：
    enum { SP_CUS, DP_CUS, SFU_CUS, TENSOR_CORE_CUS, INT_CUS, MEM_CUS, GEN_CUS };
这里是collector unit有7个，对应于SP单元一个，对应于DP单元一个，......。这里num_cu即为对应于各个单
元的collector unit的表项，在调用时gpgpu_operand_collector_num_units_sp即为对应于SP单元的CU表项数：
    m_operand_collector.add_cu_set(
            SP_CUS, m_config->gpgpu_operand_collector_num_units_sp,
            m_config->gpgpu_operand_collector_num_out_ports_sp);
在下面的代码中，m_cus[set_id]是对应于set_id指示的单元的collector unit的表项，而m_cu包含了所有收集器
单元。

这里很容易混淆，m_cus是一个字典，其定义为：
  typedef std::map<unsigned collector_set_id,      // 收集器单元的set_id
                   std::vector<collector_unit_t>>  // 收集器单元的向量
          cu_sets_t;

在V100配置中，对于每个set_id，收集器单元的数目为：
   //SP_CUS           gpgpu_operand_collector_num_units_sp = 4
   //DP_CUS           gpgpu_operand_collector_num_units_dp = 0
   //SFU_CUS          gpgpu_operand_collector_num_units_sfu = 4
   //INT_CUS          gpgpu_operand_collector_num_units_int = 0
   //TENSOR_CORE_CUS  gpgpu_operand_collector_num_units_tensor_core = 4
   //MEM_CUS          gpgpu_operand_collector_num_units_mem = 2
   //GEN_CUS          gpgpu_operand_collector_num_units_gen = 8

即这里：
    m_cus[SP_CUS         ]是一个vector，存储了SP         单元的4个收集器单元；
    m_cus[DP_CUS         ]是一个vector，存储了DP         单元的0个收集器单元；
    m_cus[SFU_CUS        ]是一个vector，存储了SFU        单元的4个收集器单元；
    m_cus[INT_CUS        ]是一个vector，存储了INT        单元的0个收集器单元；
    m_cus[TENSOR_CORE_CUS]是一个vector，存储了TENSOR_CORE单元的4个收集器单元；
    m_cus[MEM_CUS        ]是一个vector，存储了MEM        单元的2个收集器单元；
    m_cus[GEN_CUS        ]是一个vector，存储了GEN        单元的8个收集器单元。

而m_cu定义：
    collector_unit_t *m_cu;
存储了上述所有的收集器单元，即m_cu是一个vector，存储了所有的收集器单元。
*/
void opndcoll_rfu_t::add_cu_set(unsigned set_id, unsigned num_cu,
                                unsigned num_dispatch) {
  //m_cus是set_id对应收集器单元的的字典。
  m_cus[set_id].reserve(num_cu);  // this is necessary to stop pointers in m_cu
                                  // from being invalid do to a resize;
  for (unsigned i = 0; i < num_cu; i++) {
    //增加收集器单元。m_cus[set_id]是set_id对应收集器单元。这里的set_id定义为：
    //    enum { SP_CUS, DP_CUS, SFU_CUS, TENSOR_CORE_CUS, INT_CUS, MEM_CUS, GEN_CUS };
    m_cus[set_id].push_back(collector_unit_t());
    //m_cu是所有收集器单元的集合，包括所有的m_cus字典中的所有收集器单元。
    m_cu.push_back(&m_cus[set_id].back());
  }
  // for now each collector set gets dedicated dispatch units.
  //目前，每个收集器set都有专用的调度单元，由gpgpu_operand_collector_num_out_ports_sp等确定。在
  //V100配置中：
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
  //m_cus[set_id]，对应于set_id的收集器单元：
  //    m_cus[SP_CUS         ]是一个vector，存储了SP         单元的4个收集器单元；
  //    m_cus[DP_CUS         ]是一个vector，存储了DP         单元的0个收集器单元；
  //    m_cus[SFU_CUS        ]是一个vector，存储了SFU        单元的4个收集器单元；
  //    m_cus[INT_CUS        ]是一个vector，存储了INT        单元的0个收集器单元；
  //    m_cus[TENSOR_CORE_CUS]是一个vector，存储了TENSOR_CORE单元的4个收集器单元；
  //    m_cus[MEM_CUS        ]是一个vector，存储了MEM        单元的2个收集器单元；
  //    m_cus[GEN_CUS        ]是一个vector，存储了GEN        单元的8个收集器单元。
  for (unsigned i = 0; i < num_dispatch; i++) {
    m_dispatch_units.push_back(dispatch_unit_t(&m_cus[set_id]));
  }
}

/*
input_port_t的定义：
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
port_vector_t的类型定义为存储寄存器集合register_set的向量：
    typedef std::vector<register_set *> port_vector_t;
uint_vector_t的类型定义为存储收集器单元set_id的向量：
    typedef std::vector<unsigned int> uint_vector_t;
add_port是将发射阶段的几个流水线寄存器集合ID_OC_SP等，以及后续操作数收集器发出的寄存器集
合OC_EX_SP等，对应于其所属的收集器单元set_id，添加进操作数收集器类。例如:
    for (unsigned i = 0; i < m_config->gpgpu_operand_collector_num_in_ports_sp;
         i++) {
      //m_pipeline_reg的定义：std::vector<register_set> m_pipeline_reg;
      in_ports.push_back(&m_pipeline_reg[ID_OC_SP]);
      out_ports.push_back(&m_pipeline_reg[OC_EX_SP]);
      cu_sets.push_back((unsigned)SP_CUS);
      cu_sets.push_back((unsigned)GEN_CUS);
      m_operand_collector.add_port(in_ports, out_ports, cu_sets);
      in_ports.clear(), out_ports.clear(), cu_sets.clear();
    }
上段代码就是为SP单元添加端口，根据配置的gpgpu_operand_collector_num_in_ports_sp（SP单
元进入操作数收集器的端口数目），为SP单元添加输入端口为m_pipeline_reg[ID_OC_SP]、输出端口
为m_pipeline_reg[OC_EX_SP]，收集器单元set_id为SP_CUS的端口，所有添加进的端口都存储在向
量m_in_ports中。

因此，m_in_ports对象：
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

在前面的warp调度器代码里单个Sahder Core内的warp调度器的个数由gpgpu_num_sched_per_core
配置参数决定，Volta架构每核心有4个warp调度器。每个调度器的创建代码：
     schedulers.push_back(new lrr_scheduler(
             m_stats, this, m_scoreboard, m_simt_stack, &m_warp,
             &m_pipeline_reg[ID_OC_SP], &m_pipeline_reg[ID_OC_DP],
             &m_pipeline_reg[ID_OC_SFU], &m_pipeline_reg[ID_OC_INT],
             &m_pipeline_reg[ID_OC_TENSOR_CORE], m_specilized_dispatch_reg,
             &m_pipeline_reg[ID_OC_MEM], i));
在发射过程中，warp调度器将可发射的指令按照其指令类型分发给不同的单元，这些单元包括SP/DP/
SFU/INT/TENSOR_CORE/MEM，在发射过程完成后，需要针对指令通过操作数收集器将指令所需的操作
数全部收集齐。对于一个SM，对应于一个操作数收集器，调度器的发射过程将指令放入：
    m_pipeline_reg[ID_OC_SP]、m_pipeline_reg[ID_OC_DP]、m_pipeline_reg[ID_OC_SFU]、
    m_pipeline_reg[ID_OC_INT]、m_pipeline_reg[ID_OC_TENSOR_CORE]、
    m_pipeline_reg[ID_OC_MEM]
等寄存器集合中，用以操作数收集器来收集操作数。
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
操作数收集器的初始化。num_banks由配置文件的 -gpgpu_num_reg_banks 16 参数确定，在
V100中配置为16。
*/
void opndcoll_rfu_t::init(unsigned num_banks, shader_core_ctx *shader) {
  m_shader = shader;
  m_arbiter.init(m_cu.size(), num_banks);
  // for( unsigned n=0; n<m_num_ports;n++ )
  //    m_dispatch_units[m_output[n]].init( m_num_collector_units[n] );
  
  //在V100配置中，m_num_banks被初始化为16。
  m_num_banks = num_banks;
  //m_warp_size = 32。
  m_warp_size = shader->get_config()->warp_size;

  sub_core_model = shader->get_config()->sub_core_model;
  m_num_warp_scheds = shader->get_config()->gpgpu_num_sched_per_core;
  unsigned reg_id = 0;
  if (sub_core_model) {
    assert(num_banks % shader->get_config()->gpgpu_num_sched_per_core == 0);
    assert(m_num_warp_scheds <= m_cu.size() &&
           m_cu.size() % m_num_warp_scheds == 0);
  }
  //每个warp调度器可用的bank。在sub_core_model模式中，每个warp调度器可用的bank数量是
  //有限的。在V100配置中，共有4个warp调度器，0号warp调度器可用的bank为0-3，1号warp调
  //度器可用的bank为4-7，2号warp调度器可用的bank为8-11，3号warp调度器可用的bank为12-
  //15.
  m_num_banks_per_sched =
      num_banks / shader->get_config()->gpgpu_num_sched_per_core;

  //收集器单元列表。收集器单元（m_cu）：每个收集器单元一次可以容纳一条指令。它将向器发
  //送对源寄存器的请求。一旦所有源寄存器都准备好了，调度单元就可以将其调度到输出流水线
  //寄存器集（OC_EX）。收集器单元m_cu的定义：
  //    std::vector<collector_unit_t *> m_cu;
  //m_cus[set_id]是对应于set_id指示的单元的collector unit的表项，而m_cu包含了所有收
  //集器单元。m_cu.size()则是所有的收集器单元的总数目。
  for (unsigned j = 0; j < m_cu.size(); j++) {
    if (sub_core_model) {
      //cusPerSched是每个调度器可用的收集器单元数目。
      unsigned cusPerSched = m_cu.size() / m_num_warp_scheds;
      //这里reg_id其实是对应的调度器的ID。
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
在V100配置中，m_num_banks被初始化为16。m_bank_warp_shift被初始化为5。由于在操作数
收集器的寄存器文件中，warp0的r0寄存器放在0号bank，...，warp0的r15寄存器放在15号bank，
warp0的r16寄存器放在0号bank，...，warp0的r31寄存器放在15号bank；warp1的r0寄存器放在
[0+warp_id]号bank，这里以非sub_core_model模式为例：

这里register_bank函数就是用来计算regnum所在的bank数。

Bank0   Bank1   Bank2   Bank3                   ......                  Bank15
w1:r31  w1:r16  w1:r17  w1:r18                  ......                  w1:r30
w1:r15  w1:r0   w1:r1   w1:r2                   ......                  w1:r14
w0:r16  w0:r17  w0:r18  w0:r19                  ......                  w0:r31
w0:r0   w0:r1   w0:r2   w0:r3                   ......                  w0:r15

在sub_core_model模式中，每个warp调度器可用的bank数量是有限的。在V100配置中，共有4个
warp调度器，0号warp调度器可用的bank为0-3，1号warp调度器可用的bank为4-7，2号warp调度
器可用的bank为8-11，3号warp调度器可用的bank为12-15。
*/
unsigned register_bank(int regnum, int wid, unsigned num_banks,
                       bool sub_core_model, unsigned banks_per_sched,
                       unsigned sched_id) {
  int bank = regnum;
  //warp的bank偏移。
  bank += wid;
  //在subcore模式下，每个warp调度器在寄存器集合中有一个具体的寄存器可供使用，这个寄
  //存器由调度器的m_id索引。m_num_banks_per_sched的定义为：
  //    num_banks / shader->get_config()->gpgpu_num_sched_per_core;
  //在V100配置中，共有4个warp调度器，0号warp调度器可用的bank为0-3，1号warp调度器可
  //用的bank为4-7，2号warp调度器可用的bank为8-11，3号warp调度器可用的bank为12-15。
  if (sub_core_model) {
    unsigned bank_num = (bank % banks_per_sched) + (sched_id * banks_per_sched);
    assert(bank_num < num_banks);
    return bank_num;
  } else
    return bank % num_banks;
}

/*
操作数收集器的Bank写回。
*/
bool opndcoll_rfu_t::writeback(warp_inst_t &inst) {
  assert(!inst.empty());
  //m_shader->get_regs_written(inst)获取一条指令inst中需写回的寄存器编号，以列表方
  //式返回。
  std::list<unsigned> regs = m_shader->get_regs_written(inst);
  for (unsigned op = 0; op < MAX_REG_OPERANDS; op++) {
    int reg_num = inst.arch_reg.dst[op];  // this math needs to match that used
                                          // in function_info::ptx_decode_inst
    if (reg_num >= 0) {                   // valid register
      //m_bank_warp_shift被初始化为5。
	  unsigned bank =
          register_bank(reg_num, inst.warp_id(), m_num_banks, sub_core_model,
                        m_num_banks_per_sched, inst.get_schd_id());
      if (m_arbiter.bank_idle(bank)) {
        //写回到寄存器Bank。
        //m_bank_warp_shift被初始化为5。
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
    //在V100配置中，gpgpu_clock_gated_reg_file默认为0。
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
      //增加寄存器写回数目为warp_size。m_shader->get_config()->warp_size为32。
      m_shader->incregfile_writes(
          m_shader->get_config()->warp_size);  // inst.active_count());
    }
  }
  return true;
}

/*
遍历所有调度单元。每个单元找到一个准备好的收集器单元并进行调度。
*/
void opndcoll_rfu_t::dispatch_ready_cu() {
  //目前每个收集器set都有专用的调度单元，由gpgpu_operand_collector_num_out_ports_sp
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
  //m_dispatch_units里存储了所有的调度器。下面是对所有的调度器进行循环，每个调度器都
  //向前执行一步。
  for (unsigned p = 0; p < m_dispatch_units.size(); ++p) {
    //m_dispatch_units[p]是第p个调度器。
    dispatch_unit_t &du = m_dispatch_units[p];
    //从第p个调度器找到一个准备好数据的收集器单元。
    collector_unit_t *cu = du.find_ready();
    if (cu) {
      //在对PTX指令解析的时候，有计算操作数需要的寄存器个数，m_operands在ptx_ir.h的
      //ptx_instruction类中定义：
      //    std::vector<operand_info> m_operands;
      //m_operands会在每条指令解析的时候将所有操作数都添加到其中，例如解析 mad a,b,c 
      //指令时，会将 a,b,c三个操作数添加进m_operands，即每一条指令对象有一个操作数向
      //量m_operands。该过程定义为：
      //     if (!m_operands.empty()) {
      //       std::vector<operand_info>::iterator it;
      //       for (it = ++m_operands.begin(); it != m_operands.end(); it++) {
      //         //操作数数量计数。
      //         num_operands++;
      //         //如果操作数是寄存器或者是矢量，num_regs数量加1。
      //         if ((it->is_reg() || it->is_vector())) {
      //           num_regs++;
      //         }
      //       }
      //     }
      //cu->get_num_operands()返回的是num_operands值，cu->get_num_regs()返回的是
      //num_regs值。实际上，无论一个操作数是寄存器，向量抑或是立即数，地址等，操作数
      //数量num_operands都在计数，但是只有寄存器，向量出现的时候num_regs才计数。
      for (unsigned i = 0; i < (cu->get_num_operands() - cu->get_num_regs());
           i++) {
        //这里m_shader->get_config()->gpgpu_clock_gated_reg_file在V100中为0。
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
      //如果能够从第p个调度器找到一个准备好数据的的收集器单元的话，就执行它的分发函数
      //dispatch()。主要过程是，经过收集器单元收集完源操作数后，将原先暂存在收集器单
      //元指令槽m_warp中的指令推出到m_output_register中。
      cu->dispatch();
    }
  }
}

/*
opndcoll_rfu_t::allocate_cu函数将ID_OC流水线寄存器中的指令分配给收集器单元。
*/
void opndcoll_rfu_t::allocate_cu(unsigned port_num) {
  //端口（m_in_Ports）：包含输入流水线寄存器集合（ID_OC）和输出寄存器集合（OC_EX）。
  //ID_OC端口中的warp_inst_t将被发布到收集器单元。此外，当收集器单元获得所有所需的源
  //寄存器时，它将由调度单元调度到输出管道寄存器集（OC_EX）。m_in_ports中会含有多个
  //input_port_t对象，每个对象分别对应于SP/DP/SFU/INT/MEM/TC单元（但是一个单元可能
  //会有多个input_port_t对象，不是一一对应的），例如添加SP单元的input_port_t对象时：
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
  //所以这里的inp=m_in_ports[port_num]是第port_num个input_port_t对象。
  input_port_t &inp = m_in_ports[port_num];
  //对inp的输入端口进行循环。
  for (unsigned i = 0; i < inp.m_in.size(); i++) {
    //遍历寄存器集合(*inp.m_in[i])是否存在一个非空寄存器已准备好。
    if ((*inp.m_in[i]).has_ready()) {
      // find a free cu
      //遍历当前端口内的所有收集器单元，找到一个空闲的收集器单元。
      for (unsigned j = 0; j < inp.m_cu_sets.size(); j++) {
        //m_cus是一个字典，存储了所有的收集器单元，其定义：
        //   //id对应收集器单元的的字典。
        //   typedef std::map<unsigned /* collector set */,
        //                    std::vector<collector_unit_t> /*collector sets*/>
        //       cu_sets_t;
        //   //操作数收集器的集合。
        //   cu_sets_t m_cus;
        //例如，inp.m_cu_sets[j]可以是SP_CUS，那么m_cus[inp.m_cu_sets[j]]就相当于是
        //m_cus[SP_CUS]，是一个vector，存储了SP单元的多个收集器单元。
        std::vector<collector_unit_t> &cu_set = m_cus[inp.m_cu_sets[j]];
        bool allocated = false;
        //cuLowerBound是当前调度器可用的收集器单元的下界。
        unsigned cuLowerBound = 0;
        //cuUpperBound是当前调度器可用的收集器单元的上界。
        unsigned cuUpperBound = cu_set.size();
        //schd_id是发射当前指令的调度器ID。
        unsigned schd_id;
        //在V100配置中，sub_core_model为1。
        if (sub_core_model) {
          // Sub core model only allocates on the subset of CUs assigned to the
          // scheduler that issued
          unsigned reg_id = (*inp.m_in[i]).get_ready_reg_id();
          //获取发射当前指令的调度器ID。
          schd_id = (*inp.m_in[i]).get_schd_id(reg_id);
          assert(cu_set.size() % m_num_warp_scheds == 0 &&
                 cu_set.size() >= m_num_warp_scheds);
          //一个调度器可用的收集器单元数目。
          unsigned cusPerSched = cu_set.size() / m_num_warp_scheds;
          //cuLowerBound是当前调度器可用的收集器单元的下界。
          cuLowerBound = schd_id * cusPerSched;
          //cuUpperBound是当前调度器可用的收集器单元的上界。
          cuUpperBound = cuLowerBound + cusPerSched;
          assert(0 <= cuLowerBound && cuUpperBound <= cu_set.size());
        }
        //检查cuLowerBound-(cuUpperBound-1)范围内的收集器单元是否有空闲的收集器单元。
        for (unsigned k = cuLowerBound; k < cuUpperBound; k++) {
          if (cu_set[k].is_free()) {
            //找到一个空闲的收集器单元，其索引为k。
            collector_unit_t *cu = &cu_set[k];
            //当前收集器单元为空闲状态的话，cu->allocate就可以将一个新的warp指令放到
            //这个收集器单元中。
            allocated = cu->allocate(inp.m_in[i], inp.m_out[i]);
            //从收集器单元获取所有的源操作数，并将它们放入m_queue[bank]队列。
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
仲裁器检查请求，并返回不同寄存器Bank中的op_t列表，并且这些寄存器Bank不处于Write状态。
在该函数中，仲裁器检查请求并返回op_t的列表，这些op_t位于不同的寄存器Bank中，并且这些
寄存器Bank不处于Write状态。
*/
void opndcoll_rfu_t::allocate_reads() {
  // process read requests that do not have conflicts
  //处理没有冲突的读请求。在该函数中，仲裁器检查请求并返回op_t的列表，这些op_t位于不
  //同的寄存器Bank中，并且这些寄存器Bank不处于Write状态。
  std::list<op_t> allocated = m_arbiter.allocate_reads();
  //read_ops字典，存储第i个Bank的读操作数。
  std::map<unsigned, op_t> read_ops;
  for (std::list<op_t>::iterator r = allocated.begin(); r != allocated.end();
       r++) {
    const op_t &rr = *r;
    unsigned reg = rr.get_reg();
    unsigned wid = rr.get_wid();
    unsigned bank = register_bank(reg, wid, m_num_banks, sub_core_model,
                                  m_num_banks_per_sched, rr.get_sid());
    //allocate_for_read函数分配给第bank号Bank的读状态，读的操作数为op，其定义为：
    //    void allocate_for_read(unsigned bank, const op_t &op) {
    //      assert(bank < m_num_banks);
    //      m_allocated_bank[bank].alloc_read(op);
    //    }
    m_arbiter.allocate_for_read(bank, rr);
    read_ops[bank] = rr;
  }
  std::map<unsigned, op_t>::iterator r;
  //遍历read_ops字典，存储第i个Bank的读操作数的字典，遍历所有的读操作数。
  for (r = read_ops.begin(); r != read_ops.end(); ++r) {
    op_t &op = r->second;
    unsigned cu = op.get_oc_id();
    //op.get_operand()返回当前操作数在其指令所有的源操作数中的排序。
    unsigned operand = op.get_operand();
    //设置释放掉m_not_ready位向量的第operand位，用来表明该条指令的第operand个源操
    //作数已经处于就绪状态。
    m_cu[cu]->collect_operand(operand);
    //gpgpu_clock_gated_reg_file在V100中配置为0。
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
      //设置SM的寄存器读的个数加32。
      m_shader->incregfile_reads(
          m_shader->get_config()->warp_size);  // op.get_active_count());
    }
  }
}

/*
返回当前收集器单元是否所有源操作数都准备好了。
*/
bool opndcoll_rfu_t::collector_unit_t::ready() const {
  //经过收集器单元收集完源操作数后，指令被推出到m_output_register中。这里是该收集器单元
  //并没有被free掉，且标志所有源操作数是否已经准备好的位图m_not_ready为空（即所有源操作
  //数均已准备好），并且需要输出寄存器m_output_register还有空间可以推进去。m_reg_id其实
  //是对应的调度器的ID，从m_output_register查找第m_reg_id个调度器所能够使用的槽是否可用。
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
收集器单元类的初始化。
*/
void opndcoll_rfu_t::collector_unit_t::init(unsigned n, unsigned num_banks,
                                            const core_config *config,
                                            opndcoll_rfu_t *rfu,
                                            bool sub_core_model,
                                            unsigned reg_id,
                                            unsigned banks_per_sched) {
  //隶属于哪个操作数收集器。
  m_rfu = rfu;
  //收集器单元的ID。
  m_cuid = n;
  //操作数收集器的寄存器bank数。
  m_num_banks = num_banks;
  assert(m_warp == NULL);
  //收集器单元存储了哪个warp指令源寄存器。
  m_warp = new warp_inst_t(config);
  //sub_core_model模式，每个warp调度器可用的bank数量是有限的。
  m_sub_core_model = sub_core_model;
  m_reg_id = reg_id;
  m_num_banks_per_sched = banks_per_sched;
}

/*
当前收集器单元为空闲状态的话，就可以将一个新的warp指令放到这个收集器单元中。
*/
bool opndcoll_rfu_t::collector_unit_t::allocate(register_set *pipeline_reg_set,
                                                register_set *output_reg_set) {
  assert(m_free);
  assert(m_not_ready.none());
  m_free = false;
  //经过收集器单元收集完源操作数后，将指令推出到m_output_register中。
  m_output_register = output_reg_set;
  //pipeline_reg_set->get_ready()为获取一个非空寄存器，将其指令移出，并返回这条指令。
  warp_inst_t **pipeline_reg = pipeline_reg_set->get_ready();
  if ((pipeline_reg) and !((*pipeline_reg)->empty())) {
    //获取pipeline_reg中的指令的warp ID。
    m_warp_id = (*pipeline_reg)->warp_id();
    std::vector<int> prev_regs;  // remove duplicate regs within same instr
    //实际情况下，一条PTX或者SASS指令可能有很多个源寄存器，而且这些源寄存器中可能有重复
    //的寄存器。prev_regs就是用来存储有效的去重的源寄存器的编号。由于这里有可能多次想获
    //取相同的寄存器的值，所以需将新的有效寄存器的值保存在prev_regs中。这里是对一个指令
    //的所有源寄存器编号循环，规定一条指令中的源寄存器数目最大不超过MAX_REG_OPERANDS=32。
    for (unsigned op = 0; op < MAX_REG_OPERANDS; op++) {
      int reg_num =
          (*pipeline_reg)
              ->arch_reg.src[op];  // this math needs to match that used in
                                   // function_info::ptx_decode_inst
      bool new_reg = true;
      for (auto r : prev_regs) {
        if (r == reg_num) new_reg = false;
          //如果发现prev_regs已经有了当前循环的寄存器编号reg_num，则说明reg_num已经存
          //入prev_regs了，就将new_reg置为false。
      }
      if (reg_num >= 0 && new_reg) {  // valid register
        //一个新的寄存器出现时，将其加入到prev_regs中。
        prev_regs.push_back(reg_num);
        //op_t（用于保留源操作数）的定义为：
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
        //register_bank函数就是用来计算regnum所在的bank数。
        
        //m_src_op是一个op_t类型的向量，用来存储一条指令的所有源操作数，m_src_op[0]存
        //储第0个源操作数，m_src_op[1]存储第1个源操作数，...，m_src_op[31]存储第31个
        //源操作数。
		m_src_op[op] =
            op_t(this, op, reg_num, m_num_banks, m_sub_core_model,
                 m_num_banks_per_sched, (*pipeline_reg)->get_schd_id());
        //m_not_ready的定义为：
        //    std::bitset<MAX_REG_OPERANDS * 2> m_not_ready;
        //m_not_ready是一个位向量，用来存储一条指令的所有源操作数是否处于非就绪状态。这
        //里设置第op个源操作数为非就绪状态。
        m_not_ready.set(op);
      } else
        //如果是一个旧的寄存器的话，就将其置空。
        m_src_op[op] = op_t();
    }
    // move_warp(m_warp,*pipeline_reg);
    //m_warp的定义为：
    //    warp_inst_t *m_warp;
    //这里是将pipeline_reg中的指令移出，并将其放入m_warp中，m_warp是隶属于当前收集器单
    //元的一条指令槽：
    //    m_warp = new warp_inst_t(config);
    //这里是将这条指令从流水线寄存器中移出，放到了收集器单元中暂存，即该条指令就由当前寄
    //存器单元来帮助它收集源操作数。
    pipeline_reg_set->move_out_to(m_warp);
    return true;
  }
  return false;
}

/*
分发。经过收集器单元收集完源操作数后，将原先暂存在收集器单元指令槽m_warp中的指令推出到
m_output_register中。
*/
void opndcoll_rfu_t::collector_unit_t::dispatch() {
  //确保未就绪的源操作数已经没有了，便可进一步将指令推出到m_output_register中。
  assert(m_not_ready.none());
  //经过收集器单元收集完源操作数后，将原先暂存在收集器单元指令槽m_warp中的指令推出到
  //m_output_register中。
  m_output_register->move_in(m_sub_core_model, m_reg_id, m_warp);
  //重置当前收集器单元为空闲状态。
  m_free = true;
  //经过收集器单元收集完源操作数后，将原先暂存在收集器单元指令槽m_warp中的指令推出到
  //m_output_register中。
  m_output_register = NULL;
  //重置当前收集器单元的源操作数为空。
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
simt_core_cluster即SIMT Core集群向前推进一个时钟周期。在V100配置中，每个集群内部仅有1个SIMT Core。
*/
void simt_core_cluster::core_cycle() {
  //对SIMT Core集群中的每一个单独的SIMT Core循环。
  for (std::list<unsigned>::iterator it = m_core_sim_order.begin();
       it != m_core_sim_order.end(); ++it) {
    //SIMT Core集群中的每一个单独的SIMT Core都向前推进一个时钟周期。
    m_core[*it]->cycle();
  }
  //simt_core_sim_order: Select the simulation order of cores in a cluster.
  //simt_core_sim_order是选择集群中SIMT Core的模拟顺序时，在配置中默认为1，采用循环调度。这里由于在
  //本时钟周期内，是从m_core_sim_order.begin()开始调度，因此为了实现轮询调度，将begin()位置移动到最
  //末尾。这样，下次就是从begin+1位置的SIMT Core开始调度。
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
返回当前SIMT Core集群中尚未完成的线程个数。
*/
unsigned simt_core_cluster::get_not_completed() const {
  unsigned not_completed = 0;
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; i++)
    //m_core[i]->get_not_completed()是返回当前SIMT Core集群中第i个SM上未完成的线程数。
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
返回SIMT Core集群中的活跃SM的数量。
*/
unsigned simt_core_cluster::get_n_active_sms() const {
  unsigned n = 0;
  //对集群中的所有SIMT Core进行循环。
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; i++)
    //m_core[i]->isactive()为1时，代表m_core[i]是活跃的，即该SM活跃。
    n += m_core[i]->isactive();
  return n;
}

/*
对所有SIMT Core集群遍历，选择某个集群内的一个SIMT Core，向其发射一个线程块。
*/
unsigned simt_core_cluster::issue_block2core() {
  //当前批次向SIMT Core集群发射的线程块的计数。
  unsigned num_blocks_issued = 0;
  //遍历当前SIMT Core集群内的所有SIMT Core，为每个SM找一个kernel发射。
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; i++) {
    //SIMT Core的编号。
    unsigned core =
        (i + m_cta_issue_next_core + 1) % m_config->n_simt_cores_per_cluster;

    kernel_info_t *kernel;
    // Jin: fetch kernel according to concurrent kernel setting
    if (m_config->gpgpu_concurrent_kernel_sm) {  // concurrent kernel on sm
      //支持SM上的并发内核（默认为禁用），在V100配置中禁用。
      // concurrent kernel on sm
      // always select latest issued kernel
      kernel_info_t *k = m_gpu->select_kernel();
      kernel = k;
    } else {
      // first select core kernel, if no more cta, get a new kernel
      // only when core completes
      //首先选择一个内核函数，如果该内核函数没有更多的CTA需要执行，就等其结束后换一个新内核。
      kernel = m_core[core]->get_kernel();
      if (!m_gpu->kernel_more_cta_left(kernel)) {
        // wait till current kernel finishes
        //返回当前SIMT Core尚未完成的线程个数。如果尚未完成的线程个数等于0，则说明该SM上的线程已经全
        //部结束，该kernel执行完毕。
        if (m_core[core]->get_not_completed() == 0) {
          kernel_info_t *k = m_gpu->select_kernel();
          //将kernel部署到SM上。
          if (k) m_core[core]->set_kernel(k);
          kernel = k;
        }
      }
    }
    //如果kernel有更多的CTA待执行，即当前SM尚未完成kernel的执行，但是m_core[core]可以发射一个内核函
    //数，则发射。
    if (m_gpu->kernel_more_cta_left(kernel) &&
        //            (m_core[core]->get_n_active_cta() <
        //            m_config->max_cta(*kernel)) ) {
        //can_issue_1block(*kernel)判断是否可以发射一个线程块，如果可以发射一个线程块，则返回true，
        //否则返回false。
        m_core[core]->can_issue_1block(*kernel)) {
      //将kernel部署到SM上。
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
  //icnt_has_buffer是判断互连网络是否有空闲的输入缓冲可以容纳来自deviceID号设备新的数据包。
  return !::icnt_has_buffer(m_cluster_id, request_size);
}

/*
SIMT Core将Packets注入互连网络的接口。
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
    //在V100中，L1 cache的m_write_policy为WRITE_THROUGH，实际上L1_WRBK_ACC也不会用到。
    case L1_WRBK_ACC:
      m_stats->gpgpu_n_mem_write_global++;
      break;
    //在V100中，当L2 cache写不命中时，采取lazy_fetch_on_read策略，当找到一个cache block
    //逐出时，如果这个cache block是被MODIFIED，则需要将这个cache block写回到下一级存储，
    //因此会产生L2_WRBK_ACC访问，这个访问就是为了写回被逐出的MODIFIED cache block。
    case L2_WRBK_ACC:
      m_stats->gpgpu_n_mem_l2_writeback++;
      break;
    //L1_WR_ALLOC_R/L2_WR_ALLOC_R在V100配置中暂时用不到。
    case L1_WR_ALLOC_R:
      m_stats->gpgpu_n_mem_l1_write_allocate++;
      break;
    //L1_WR_ALLOC_R/L2_WR_ALLOC_R在V100配置中暂时用不到。
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
    //icnt_push是数据包压入互连网络输入缓冲区。
    ::icnt_push(m_cluster_id, m_config->mem2device(destination), (void *)mf,
                mf->get_ctrl_size());
  else
    //icnt_push是数据包压入互连网络输入缓冲区。
    ::icnt_push(m_cluster_id, m_config->mem2device(destination), (void *)mf,
                mf->size());
}

/*
simt_core_cluster::icnt_cycle()方法将内存请求从互连网络推入simt核心集群的响应FIFO。它还从FIFO弹
出请求，并将它们发送到相应内核的指令缓存或LDST单元。

每个SIMT Core集群都有一个响应FIFO，用于保存从互连网络发出的数据包。数据包被定向到SIMT Core的指令缓
存（如果它是为指令获取未命中提供服务的内存响应）或其内存流水线（memory pipeline，LDST 单元）。数据
包以先进先出方式拿出。如果SIMT Core无法接受FIFO头部的数据包，则响应FIFO将停止。为了在LDST单元上生成
内存请求，每个SIMT Core都有自己的注入端口接入互连网络。但是，注入端口缓冲区由SIMT Core集群所有SIMT 
Core共享。
*/
void simt_core_cluster::icnt_cycle() {
  //如果响应FIFO非空，才可以接收来自ICNT的数据包。这里的m_response_fifo是指SIMT Core集群的响应FIFO。
  if (!m_response_fifo.empty()) {
    //从响应FIFO头部推出一个数据包 mf。m_response_fifo被定义为：
    //    std::list<mem_fetch *> m_response_fifo;
    //mem_fetch定义了一个模拟内存请求的通信结构。更像是一个内存请求的行为。这里的m_response_fifo是
    //指SIMT Core集群的响应FIFO。
    mem_fetch *mf = m_response_fifo.front();
    //mf->get_sid()获取内存访问请求源的SIMT Core的ID。m_config是SIMT Core集群中的Shader Core的配
    //置。m_config->sid_to_cid(sid)是依据SM的ID，获取SIMT Core集群的ID。即cid为SIMT Core集群的ID。
    unsigned cid = m_config->sid_to_cid(mf->get_sid());
    //mf->get_access_type()返回对存储器进行的访存类型mem_access_type，mem_access_type定义了在时序
    //模拟器中对不同类型的存储器进行不同的访存类型：
    //    MA_TUP(GLOBAL_ACC_R),        从global memory读
    //    MA_TUP(LOCAL_ACC_R),         从local memory读
    //    MA_TUP(CONST_ACC_R),         从常量缓存读
    //    MA_TUP(TEXTURE_ACC_R),       从纹理缓存读
    //    MA_TUP(GLOBAL_ACC_W),        向global memory写
    //    MA_TUP(LOCAL_ACC_W),         向local memory写
    //在V100中，L1 cache的m_write_policy为WRITE_THROUGH，实际上L1_WRBK_ACC也不会用到。
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
    //    MA_TUP(L2_WR_ALLOC_R),       L2缓存write-allocate
    //    MA_TUP(NUM_MEM_ACCESS_TYPE), 存储器访问的类型总数
    if (mf->get_access_type() == INST_ACC_R) {
      //如果mf->get_access_type() = 从指令缓存读，则是指令预取响应。
      // instruction fetch response.
      //m_core为SIMT Core集群定义的所有SIMT Core，一个二维shader_core_ctx矩阵，第一维代表集群ID，
      //第二维代表SIMT Core ID。fetch_unit_response_buffer_full()返回预取单元响应buffer是否已满。
      //这里这个函数一直非满，即下面的循环始终执行。
      if (!m_core[cid]->fetch_unit_response_buffer_full()) {
        //对指令预取的响应FIFO弹出一个数据包。这里的m_response_fifo是指SIMT Core集群的响应FIFO。
        m_response_fifo.pop_front();
        //m_core[cid]指向的SIMT Core集群接收这个预取的指令数据包，把mf放到cid标识的SIMT Core集群
        //的L1 I-Cache。请注意，这里m_core为SIMT Core集群定义的所有SM，一个二维shader_core_ctx矩
        //阵，第一维代表集群ID，第二维代表SIMT Core ID，其定义为：
        //    shader_core_ctx **m_core;
        //在TITAN V的配置中，一个SIMT Core集群里会有两个SM，但是这两个SM其实与互连网络共享一个公共
        //端口，且从这段代码看起来，这两个SM共用了一套指令缓存和LD/ST单元，不知道对不对，但是我们基
        //本上用到的都是单SM的配置，所以这里不必过多纠结。
        m_core[cid]->accept_fetch_response(mf);
      }
    } else {
      //如果mf->get_access_type() ≠ 从指令缓存读，则是数据提取响应。
      // data response.
      //ldst_unit_response_buffer_full()返回LDST单元响应buffer是否已满。
      //返回LDST单元响应buffer是否已满。LD/ST单元的响应FIFO中的数据包数 >= GPU配置的响应队列中的最
      //大响应包数。这里需要注意的是，LD/ST单元也有一个m_response_fifo，且m_response_fifo.size()
      //获取的是该fifo已经存储的mf数目，m_config->ldst_unit_response_queue_size则是配置的该fifo的
      //最大容量，一旦m_response_fifo.size()等于配置的最大容量，就会返回True，表示该fifo已满。
      if (!m_core[cid]->ldst_unit_response_buffer_full()) {
        //对数据预取的响应FIFO弹出一个数据包。这里的m_response_fifo是指SIMT Core集群的响应FIFO。
        m_response_fifo.pop_front();
        //统计Memory Latency Statistics.
        m_memory_stats->memlatstat_read_done(mf);
        //m_core[cid]指向的SIMT Core集群接收这个预取的data数据包。请注意，这里m_core为SIMT Core集
        //群定义的所有SM，一个二维shader_core_ctx矩阵，第一维代表集群ID，第二维代表SIMT Core ID，
        //其定义为：
        //    shader_core_ctx **m_core;
        //在TITAN V的配置中，一个SIMT Core集群里会有两个SM，但是这两个SM其实与互连网络共享一个公共
        //端口，且从这段代码看起来，这两个SM共用了一套指令缓存和LD/ST单元，不知道对不对，但是我们基
        //本上用到的都是单SM的配置，所以这里不必过多纠结。
        m_core[cid]->accept_ldst_unit_response(mf);
      }
    }
  }
  //m_config->n_simt_ejection_buffer_size是弹出缓冲区中的数据包数。其实可以理解为这就是SIMT Core集群
  //的响应FIFO的最大容量。如果响应FIFO大小 < 弹出缓冲区中的数据包数，则弹出缓冲区可以继续向SIMT Core集群
  //的响应FIFO里弹出下一个数据包。弹出缓冲区指的是，[互连网络->弹出缓冲区->SIMT Core集群]的中间节点。这
  //里m_response_fifo.size()是指m_response_fifo中的数据包数量，当m_response_fifo为空时，size=0。
  if (m_response_fifo.size() < m_config->n_simt_ejection_buffer_size) {
    //这里mem_fetch *mf指的是互连网络继续向SIMT Core集群的响应FIFO里弹出的下一个数据包。
    //icnt_pop是数据包弹出互连网络输出缓冲区。
    mem_fetch *mf = (mem_fetch *)::icnt_pop(m_cluster_id);
    //如果没弹出来，说明互连网络的弹出缓冲区（由互连网络->SIMT Core集群）为空，互连网络没有新的数据包要向
    //SIMT Core集群传输。
    if (!mf) return;
    assert(mf->get_tpc() == m_cluster_id);
    assert(mf->get_type() == READ_REPLY || mf->get_type() == WRITE_ACK);

    // The packet size varies depending on the type of request:
    // - For read request and atomic request, the packet contains the data
    // - For write-ack, the packet only has control metadata
    //数据包大小因请求类型而异：
    // - 对于读取请求和原子请求，数据包包含数据；
    // - 对于写确认，数据包只有控制元数据。
    unsigned int packet_size =
        (mf->get_is_write()) ? mf->get_ctrl_size() : mf->size();
    m_stats->m_incoming_traffic_stats->record_traffic(mf, packet_size);
    mf->set_status(IN_CLUSTER_TO_SHADER_QUEUE,
                   m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
    // m_memory_stats->memlatstat_read_done(mf,m_shader_config->max_warps_per_shader);
    //SIMT Core集群的响应FIFO将数据包mf加入到FIFO底部，先入先出顺序。
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
