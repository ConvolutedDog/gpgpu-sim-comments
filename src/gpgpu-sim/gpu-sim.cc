// Copyright (c) 2009-2021, Tor M. Aamodt, Wilson W.L. Fung, George L. Yuan,
// Ali Bakhoda, Andrew Turner, Ivan Sham, Vijay Kandiah, Nikos Hardavellas,
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

/*
gpu-sim.cc 将GPGPU-Sim中不同的时序模型粘在一起。它包含了支持多个时钟域的实现，并实现了线程块调度器。
*/

#include "gpu-sim.h"

#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include "zlib.h"

#include "dram.h"
#include "mem_fetch.h"
#include "shader.h"
#include "shader_trace.h"

#include <time.h>
#include "addrdec.h"
#include "delayqueue.h"
#include "dram.h"
#include "gpu-cache.h"
#include "gpu-misc.h"
#include "icnt_wrapper.h"
#include "l2cache.h"
#include "shader.h"
#include "stat-tool.h"

#include "../../libcuda/gpgpu_context.h"
#include "../abstract_hardware_model.h"
#include "../cuda-sim/cuda-sim.h"
#include "../cuda-sim/cuda_device_runtime.h"
#include "../cuda-sim/ptx-stats.h"
#include "../cuda-sim/ptx_ir.h"
#include "../debug.h"
#include "../gpgpusim_entrypoint.h"
#include "../statwrapper.h"
#include "../trace.h"
#include "mem_latency_stat.h"
#include "power_stat.h"
#include "stats.h"
#include "visualizer.h"

#ifdef GPGPUSIM_POWER_MODEL
#include "power_interface.h"
#else
class gpgpu_sim_wrapper {};
#endif

#include <stdio.h>
#include <string.h>
#include <iostream>
#include <sstream>
#include <string>

// #define MAX(a, b) (((a) > (b)) ? (a) : (b)) //redefined

bool g_interactive_debugger_enabled = false;

tr1_hash_map<new_addr_type, unsigned> address_random_interleaving;

/* 
Clock Domains
时钟域编码 
*/

#define CORE 0x01
#define L2 0x02
#define DRAM 0x04
#define ICNT 0x08

#define MEM_LATENCY_STAT_IMPL

#include "mem_latency_stat.h"

/*
该函数用于注册GPGPU-Sim中用于控制能耗模型的命令行选项，并将它们添加到OptionParser实例中。该函数接
受一个OptionParser指针作为参数，用于将能耗模型参数添加到OptionParser实例中。
*/
void power_config::reg_options(class OptionParser *opp) {
  //设置gpuwattch功耗评估模型的xml_file存储路径，默认为"gpuwattch.xml"。
  option_parser_register(opp, "-accelwattch_xml_file", OPT_CSTR,
                         &g_power_config_name, "AccelWattch XML file",
                         "accelwattch_sass_sim.xml");
  //设置开启gpuwattch功耗评估开关。
  option_parser_register(opp, "-power_simulation_enabled", OPT_BOOL,
                         &g_power_simulation_enabled,
                         "Turn on power simulator (1=On, 0=Off)", "0");
  //设置Dump功耗评估输出的节拍间隔。
  option_parser_register(opp, "-power_per_cycle_dump", OPT_BOOL,
                         &g_power_per_cycle_dump,
                         "Dump detailed power output each cycle", "0");

  option_parser_register(opp, "-hw_perf_file_name", OPT_CSTR,
                         &g_hw_perf_file_name,
                         "Hardware Performance Statistics file", "hw_perf.csv");

  option_parser_register(
      opp, "-hw_perf_bench_name", OPT_CSTR, &g_hw_perf_bench_name,
      "Kernel Name in Hardware Performance Statistics file", "");

  option_parser_register(opp, "-power_simulation_mode", OPT_INT32,
                         &g_power_simulation_mode,
                         "Switch performance counter input for power "
                         "simulation (0=Sim, 1=HW, 2=HW-Sim Hybrid)",
                         "0");

  option_parser_register(opp, "-dvfs_enabled", OPT_BOOL, &g_dvfs_enabled,
                         "Turn on DVFS for power model", "0");
  option_parser_register(opp, "-aggregate_power_stats", OPT_BOOL,
                         &g_aggregate_power_stats,
                         "Accumulate power across all kernels", "0");

  // Accelwattch Hyrbid Configuration

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_L1_RH", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_L1_RH],
      "Get L1 Read Hits for Accelwattch-Hybrid from Accel-Sim", "0");
  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_L1_RM", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_L1_RM],
      "Get L1 Read Misses for Accelwattch-Hybrid from Accel-Sim", "0");
  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_L1_WH", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_L1_WH],
      "Get L1 Write Hits for Accelwattch-Hybrid from Accel-Sim", "0");
  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_L1_WM", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_L1_WM],
      "Get L1 Write Misses for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_L2_RH", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_L2_RH],
      "Get L2 Read Hits for Accelwattch-Hybrid from Accel-Sim", "0");
  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_L2_RM", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_L2_RM],
      "Get L2 Read Misses for Accelwattch-Hybrid from Accel-Sim", "0");
  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_L2_WH", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_L2_WH],
      "Get L2 Write Hits for Accelwattch-Hybrid from Accel-Sim", "0");
  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_L2_WM", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_L2_WM],
      "Get L2 Write Misses for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_CC_ACC", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_CC_ACC],
      "Get Constant Cache Acesses for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_SHARED_ACC", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_SHRD_ACC],
      "Get Shared Memory Acesses for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(opp, "-accelwattch_hybrid_perfsim_DRAM_RD", OPT_BOOL,
                         &accelwattch_hybrid_configuration[HW_DRAM_RD],
                         "Get DRAM Reads for Accelwattch-Hybrid from Accel-Sim",
                         "0");
  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_DRAM_WR", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_DRAM_WR],
      "Get DRAM Writes for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_NOC", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_NOC],
      "Get Interconnect Acesses for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_PIPE_DUTY", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_PIPE_DUTY],
      "Get Pipeline Duty Cycle Acesses for Accelwattch-Hybrid from Accel-Sim",
      "0");

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_NUM_SM_IDLE", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_NUM_SM_IDLE],
      "Get Number of Idle SMs for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_CYCLES", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_CYCLES],
      "Get Executed Cycles for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_VOLTAGE", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_VOLTAGE],
      "Get Chip Voltage for Accelwattch-Hybrid from Accel-Sim", "0");

  // Output Data Formats
  //设置开启生成功耗跟踪文件。
  option_parser_register(
      opp, "-power_trace_enabled", OPT_BOOL, &g_power_trace_enabled,
      "produce a file for the power trace (1=On, 0=Off)", "0");
  //功耗跟踪输出日志的压缩级别。
  option_parser_register(
      opp, "-power_trace_zlevel", OPT_INT32, &g_power_trace_zlevel,
      "Compression level of the power trace output log (0=no comp, 9=highest)",
      "6");
  //生成稳定功耗电平的文件。
  option_parser_register(
      opp, "-steady_power_levels_enabled", OPT_BOOL,
      &g_steady_power_levels_enabled,
      "produce a file for the steady power levels (1=On, 0=Off)", "0");
  //允许偏差:样本数量。
  option_parser_register(opp, "-steady_state_definition", OPT_CSTR,
                         &gpu_steady_state_definition,
                         "allowed deviation:number of samples", "8:4");
}

/*
该函数用于注册GPGPU-Sim中用于控制存储模型的命令行选项，并将它们添加到OptionParser实例中。该函数接
受一个OptionParser指针作为参数，用于将存储模型参数添加到OptionParser实例中。
*/
void memory_config::reg_options(class OptionParser *opp) {
  //cuda-sim.cc中已经实现了功能性的 memcpy_to_gpu() 函数，这里的 m_perf_sim_memcpy 标志是否执行
  //性能模型中的 perf_memcpy_to_gpu()函数，即功能相同，把数据拷贝到GPU的显存。
  option_parser_register(opp, "-gpgpu_perf_sim_memcpy", OPT_BOOL,
                         &m_perf_sim_memcpy, "Fill the L2 cache on memcpy",
                         "1");
  //默认为0，关闭，后面用到再补充。
  option_parser_register(opp, "-gpgpu_simple_dram_model", OPT_BOOL,
                         &simple_dram_model,
                         "simple_dram_model with fixed latency and BW", "0");
  //GPGPU-Sim对DRAM调度和时序进行建模。GPGPU-Sim实现了两个开放页面模式DRAM调度器：一个FIFO（先进
  //先出）调度器和一个FR-FCFS（First-Row First-Come-First-Served，First-Row 先到先服务）调度器，
  //这两个调度器都在下面描述。可以使用配置选项-gpgpu_dram_scheduler选择这些选项。
  option_parser_register(opp, "-gpgpu_dram_scheduler", OPT_INT32,
                         &scheduler_type, "0 = fifo, 1 = FR-FCFS (defaul)",
                         "1");
  //内存分区中"icnt-to-L2"，"L2-to-dram"，"dram-to-L2"，"L2-to-icnt"四个queue的最大长度。
  //内存请求数据包通过ICNT->L2 queue从互连网络进入内存分区。L2 Cache Bank在每个L2时钟周期从ICNT-> 
  //L2 queue弹出一个请求进行服务。L2生成的芯片外DRAM的任何内存请求都被推入L2->DRAM queue。如果L2 
  //Cache被禁用，数据包将从ICNT->L2 queue弹出，并直接推入L2->DRAM queue，仍然以L2时钟频率。从片外
  //DRAM返回的填充请求从DRAM->L2 queue弹出，并由L2 Cache Bank消耗。从L2到SIMT Core的读响应通过L2
  //->ICNT queue推送。
  option_parser_register(opp, "-gpgpu_dram_partition_queues", OPT_CSTR,
                         &gpgpu_L2_queue_config, "i2$:$2d:d2$:$2i", "8:8:8:8");
  //貌似没有调用过，后面用到再补充。非常理想的l2_cache，总是访存命中。
  option_parser_register(opp, "-l2_ideal", OPT_BOOL, &l2_ideal,
                         "Use a ideal L2 cache that always hit", "0");
  //统一的分Bank的 L2 数据缓存的配置。???
  option_parser_register(opp, "-gpgpu_cache:dl2", OPT_CSTR,
                         &m_L2_config.m_config_string,
                         "unified banked L2 data cache config "
                         " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                         "alloc>,<mshr>:<N>:<merge>,<mq>}",
                         "64:128:8,L:B:m:N,A:16:4,4");
  //是否将 L2 数据缓存仅用于 texture。
  option_parser_register(opp, "-gpgpu_cache:dl2_texture_only", OPT_BOOL,
                         &m_L2_texure_only, "L2 cache used for texture only",
                         "1");
  //gpgpu_n_mem为配置中的内存控制器（DRAM Channel）数量。
  option_parser_register(
      opp, "-gpgpu_n_mem", OPT_UINT32, &m_n_mem,
      "number of memory modules (e.g. memory controllers) in gpu", "8");
  //每个内存模块中的子内存子分区的个数。
  option_parser_register(opp, "-gpgpu_n_sub_partition_per_mchannel", OPT_UINT32,
                         &m_n_sub_partition_per_memory_channel,
                         "number of memory subpartition in each memory module",
                         "1");
  //每个内存控制器的DRAM芯片（也成为DRAM channel）数量由选项-gpgpu_n_mem_per_ctrlr设置。
  option_parser_register(opp, "-gpgpu_n_mem_per_ctrlr", OPT_UINT32,
                         &gpu_n_mem_per_ctrlr,
                         "number of memory chips per memory controller", "1");
  //收集内存延迟统计信息（0x2启用MC，0x4启用队列日志）。
  option_parser_register(opp, "-gpgpu_memlatency_stat", OPT_INT32,
                         &gpgpu_memlatency_stat,
                         "track and display latency statistics 0x2 enables MC, "
                         "0x4 enables queue logs",
                         "0");
  //DRAM FRFCFS调度程序队列大小（0 = unlimited (default); # entries per chip）（FIFO调度程序队
  //列大小固定为2）。
  option_parser_register(opp, "-gpgpu_frfcfs_dram_sched_queue_size", OPT_INT32,
                         &gpgpu_frfcfs_dram_sched_queue_size,
                         "0 = unlimited (default); # entries per chip", "0");
  //DRAM请求返回队列大小（0 = unlimited (default); # entries per chip）。
  option_parser_register(opp, "-gpgpu_dram_return_queue_size", OPT_INT32,
                         &gpgpu_dram_return_queue_size,
                         "0 = unlimited (default); # entries per chip", "0");
  //单个DRAM芯片在命令总线频率下的总线带宽（默认值为4字节（每个命令时钟周期8字节））。每个内存控制器
  //的DRAM芯片数量由选项 -gpgpu_n_mem_per_ctrlr 设置。每个存储器分区具有（gpgpu_dram_buswidth x 
  //gpgpu_n_mem_per_ctrlr）位的DRAM数据总线引脚。例如，Quadro FX5800有一条512位DRAM数据总线，分为
  //8个内存分区。每个存储器分区一个 512/8 = 64 位的DRAM数据总线。该64位总线被分割为每个存储器分区的
  //2个DRAM芯片。每个芯片将具有32位=4字节的DRAM总线宽度。因此，我们将 -gpgpu_dram_buswidth 设置为4。
  option_parser_register(opp, "-gpgpu_dram_buswidth", OPT_UINT32, &busW,
                         "default = 4 bytes (8 bytes per cycle at DDR)", "4");
  //每个DRAM请求的Burst长度（默认值=4个数据时钟周期，在GDDR3中以2倍命令时钟频率运行）。由-gpgpu_dram
  //_burst_length <# burst per DRAM request>配置。
  option_parser_register(
      opp, "-gpgpu_dram_burst_length", OPT_UINT32, &BL,
      "Burst length of each DRAM request (default = 4 data bus cycle)", "4");
  option_parser_register(opp, "-dram_data_command_freq_ratio", OPT_UINT32,
                         &data_command_freq_ratio,
                         "Frequency ratio between DRAM data bus and command "
                         "bus (default = 2 times, i.e. DDR)",
                         "2");
  //DRAM时序参数: ???
  //    [nbk = number of banks]
  //    [tCCD = Column to Column Delay (always = half of burst length)]
  //    [tRRD = Row active to row active delay]
  //    [tRCD = RAW to CAS delay]
  //    [tRAS = Row active time]
  //    [tRP = Row precharge time]
  //    [tRC = Row cycle time]
  //    [CL = CAS latency]
  //    [WL = Write latency]
  //    [tCDLR = ]
  //    [tWR = ]
  //    [nbkgrp = ]
  //    [tCCDL = ]
  //    [tRTPL = ]
  //    [tWTR = Write to read delay]
  option_parser_register(
      opp, "-gpgpu_dram_timing_opt", OPT_CSTR, &gpgpu_dram_timing_opt,
      "DRAM timing parameters = "
      "{nbk:tCCD:tRRD:tRCD:tRAS:tRP:tRC:CL:WL:tCDLR:tWR:nbkgrp:tCCDL:tRTPL}",
      "4:2:8:12:21:13:34:9:4:5:13:1:0:0");
  option_parser_register(opp, "-gpgpu_l2_rop_latency", OPT_UINT32, &rop_latency,
                         "ROP queue latency (default 85)", "85");
  option_parser_register(opp, "-dram_latency", OPT_UINT32, &dram_latency,
                         "DRAM latency (default 30)", "30");
  option_parser_register(opp, "-dram_dual_bus_interface", OPT_UINT32,
                         &dual_bus_interface,
                         "dual_bus_interface (default = 0) ", "0");
  option_parser_register(opp, "-dram_bnk_indexing_policy", OPT_UINT32,
                         &dram_bnk_indexing_policy,
                         "dram_bnk_indexing_policy (0 = normal indexing, 1 = "
                         "Xoring with the higher bits) (Default = 0)",
                         "0");
  option_parser_register(opp, "-dram_bnkgrp_indexing_policy", OPT_UINT32,
                         &dram_bnkgrp_indexing_policy,
                         "dram_bnkgrp_indexing_policy (0 = take higher bits, 1 "
                         "= take lower bits) (Default = 0)",
                         "0");
  option_parser_register(opp, "-dram_seperate_write_queue_enable", OPT_BOOL,
                         &seperate_write_queue_enabled,
                         "Seperate_Write_Queue_Enable", "0");
  option_parser_register(opp, "-dram_write_queue_size", OPT_CSTR,
                         &write_queue_size_opt, "Write_Queue_Size", "32:28:16");
  option_parser_register(
      opp, "-dram_elimnate_rw_turnaround", OPT_BOOL, &elimnate_rw_turnaround,
      "elimnate_rw_turnaround i.e set tWTR and tRTW = 0", "0");
  option_parser_register(opp, "-icnt_flit_size", OPT_UINT32, &icnt_flit_size,
                         "icnt_flit_size", "32");
  m_address_mapping.addrdec_setoption(opp);
}

/*
注册每个Shader Core（SM）的参数设置。
*/
void shader_core_config::reg_options(class OptionParser *opp) {
  //SIMT堆栈处理分支的模式，1代表采用后必经结点模式，其他暂不支持。
  //传统的SIMT Stack（PDOM机制）在线程束分化后采用了一种“unified”机制，令所有分化的线程束“统一地”在
  //条件跳转指令的“immediate post-dominator”处（即IPDOM处）进行汇聚（reconverge）。根据“y is post-
  //dominator of x”的定义：所有路径经由x点则必经由y点，以此可以确保在x点分化出去的所有线程必经过y点；
  //并且“immediate post-dominator”的定义又保证了y点是最早可以汇聚到所有分支线程的点，越早的汇聚则意
  //味着SIMD流水线可以越早地被更充分地利用。
  option_parser_register(opp, "-gpgpu_simd_model", OPT_INT32, &model,
                         "1 = post-dominator", "1");
  //Shader Core Pipeline配置。
  //参数分别是：<每个SM最大可支配线程数>:<定义一个warp有多少线程>
  option_parser_register(
      opp, "-gpgpu_shader_core_pipeline", OPT_CSTR,
      &gpgpu_shader_core_pipeline_opt,
      "shader core pipeline config, i.e., {<nthread>:<warpsize>}", "1024:32");
  //L1 texture cache的配置，后面用到再补充。
  option_parser_register(opp, "-gpgpu_tex_cache:l1", OPT_CSTR,
                         &m_L1T_config.m_config_string,
                         "per-shader L1 texture cache  (READ-ONLY) config "
                         " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                         "alloc>,<mshr>:<N>:<merge>,<mq>:<rf>}",
                         "8:128:5,L:R:m:N,F:128:4,128:2");
  //L1常量缓存（只读）配置。逐出策略：L=LRU，F=FIFO，R=Random。???
  //1. cache_config: 缓存配置，用于指定缓存的类型，包括：L=LRU(Least Recently Used)、F=FIFO(First 
  //   In First Out)、R=Random和Pseudo-LRU。
  //2. cache_size: 缓存大小，用于指定缓存的大小，以字节为单位。
  //3. line_sz: 缓存行大小，用于指定缓存行的大小，以字节为单位。
  //4. associativity: 组相关性，用于指定缓存的组相关性，例如2-way、4-way、8-way等。
  //5. num_banks: 存储器银行数量，用于指定存储器银行的数量，可以是1、2、4等。
  //6. throughput: 缓存吞吐量，用于指定缓存的吞吐量，以每秒字节为单位。
  //7. latency: 缓存延迟，用于指定缓存的延迟，以周期数为单位。
  //！！！错误！！！理解：
  //<nsets>：缓存行的数量，即缓存的容量
  //<bsize>：每个缓存行的字节数
  //<assoc>：缓存的联想度，即一个组中的行数
  //<rep>：替换策略，有LRU，FIFO，Random等
  //<wr>：写策略，有write-back和write-through
  //<alloc>：写分配策略，有write-allocate和no-write-allocate
  //<wr_alloc>：写回分配策略，有write-allocate和no-write-allocate
  //<mshr>：多路复用请求寄存器的大小
  //<N>：每个请求寄存器中最多存储的请求数
  //<merge>：是否启用请求合并，有yes和no
  //<mq>：是否启用队列，有yes和no
  option_parser_register(
      opp, "-gpgpu_const_cache:l1", OPT_CSTR, &m_L1C_config.m_config_string,
      "per-shader L1 constant memory cache  (READ-ONLY) config "
      " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<"
      "merge>,<mq>} ",
      "64:64:2,L:R:f:N,A:2:32,4");
  //L1 instruction cache的配置。
  option_parser_register(opp, "-gpgpu_cache:il1", OPT_CSTR,
                         &m_L1I_config.m_config_string,
                         "shader L1 instruction cache config "
                         " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                         "alloc>,<mshr>:<N>:<merge>,<mq>} ",
                         "4:256:4,L:R:f:N,A:2:32,4");
  //L1 data cache的配置。
  option_parser_register(opp, "-gpgpu_cache:dl1", OPT_CSTR,
                         &m_L1D_config.m_config_string,
                         "per-shader L1 data cache config "
                         " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                         "alloc>,<mshr>:<N>:<merge>,<mq> | none}",
                         "none");
  option_parser_register(opp, "-gpgpu_l1_cache_write_ratio", OPT_UINT32,
                         &m_L1D_config.m_wr_percent, "L1D write ratio", "0");
  //L1 cache的bank数。例如 Volta unified cache 有 4 个banks。
  option_parser_register(opp, "-gpgpu_l1_banks", OPT_UINT32,
                         &m_L1D_config.l1_banks, "The number of L1 cache banks",
                         "1");
  option_parser_register(opp, "-gpgpu_l1_banks_byte_interleaving", OPT_UINT32,
                         &m_L1D_config.l1_banks_byte_interleaving,
                         "l1 banks byte interleaving granularity", "32");
  option_parser_register(opp, "-gpgpu_l1_banks_hashing_function", OPT_UINT32,
                         &m_L1D_config.l1_banks_hashing_function,
                         "l1 banks hashing function", "0");
  option_parser_register(opp, "-gpgpu_l1_latency", OPT_UINT32,
                         &m_L1D_config.l1_latency, "L1 Hit Latency", "1");
  option_parser_register(opp, "-gpgpu_smem_latency", OPT_UINT32, &smem_latency,
                         "smem Latency", "3");
  option_parser_register(opp, "-gpgpu_cache:dl1PrefL1", OPT_CSTR,
                         &m_L1D_config.m_config_stringPrefL1,
                         "per-shader L1 data cache config "
                         " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                         "alloc>,<mshr>:<N>:<merge>,<mq> | none}",
                         "none");
  option_parser_register(opp, "-gpgpu_cache:dl1PrefShared", OPT_CSTR,
                         &m_L1D_config.m_config_stringPrefShared,
                         "per-shader L1 data cache config "
                         " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                         "alloc>,<mshr>:<N>:<merge>,<mq> | none}",
                         "none");
  option_parser_register(opp, "-gpgpu_gmem_skip_L1D", OPT_BOOL, &gmem_skip_L1D,
                         "global memory access skip L1D cache (implements "
                         "-Xptxas -dlcm=cg, default=no skip)",
                         "0");

  option_parser_register(opp, "-gpgpu_perfect_mem", OPT_BOOL,
                         &gpgpu_perfect_mem,
                         "enable perfect memory mode (no cache miss)", "0");
  option_parser_register(
      opp, "-n_regfile_gating_group", OPT_UINT32, &n_regfile_gating_group,
      "group of lanes that should be read/written together)", "4");
  option_parser_register(
      opp, "-gpgpu_clock_gated_reg_file", OPT_BOOL, &gpgpu_clock_gated_reg_file,
      "enable clock gated reg file for power calculations", "0");
  option_parser_register(
      opp, "-gpgpu_clock_gated_lanes", OPT_BOOL, &gpgpu_clock_gated_lanes,
      "enable clock gated lanes for power calculations", "0");
  //每个Shader Core的寄存器数。并发CTA的限制数量。
  //-gpgpu_shader_registers <# registers/shader core, default=8192>
  option_parser_register(opp, "-gpgpu_shader_registers", OPT_UINT32,
                         &gpgpu_shader_registers,
                         "Number of registers per shader core. Limits number "
                         "of concurrent CTAs. (default 8192)",
                         "8192");
  //每个CTA的最大寄存器数（默认值8192）。
  option_parser_register(
      opp, "-gpgpu_registers_per_block", OPT_UINT32, &gpgpu_registers_per_block,
      "Maximum number of registers per CTA. (default 8192)", "8192");
  option_parser_register(opp, "-gpgpu_ignore_resources_limitation", OPT_BOOL,
                         &gpgpu_ignore_resources_limitation,
                         "gpgpu_ignore_resources_limitation (default 0)", "0");
  //Shader Core中并发cta的最大数量。-gpgpu_shader_cta <# CTA/shader core, default=8>
  option_parser_register(
      opp, "-gpgpu_shader_cta", OPT_UINT32, &max_cta_per_core,
      "Maximum number of concurrent CTAs in shader (default 32)", "32");
  option_parser_register(
      opp, "-gpgpu_num_cta_barriers", OPT_UINT32, &max_barriers_per_cta,
      "Maximum number of named barriers per CTA (default 16)", "16");
  option_parser_register(opp, "-gpgpu_n_clusters", OPT_UINT32, &n_simt_clusters,
                         "number of processing clusters", "10");
  option_parser_register(opp, "-gpgpu_n_cores_per_cluster", OPT_UINT32,
                         &n_simt_cores_per_cluster,
                         "number of simd cores per cluster", "3");
  //GPU配置的SIMT Core集群的弹出缓冲区中的数据包数。弹出缓冲区指的是，[互连网络->弹出缓冲区->SIMT 
  //Core集群]的中间节点。
  option_parser_register(opp, "-gpgpu_n_cluster_ejection_buffer_size",
                         OPT_UINT32, &n_simt_ejection_buffer_size,
                         "number of packets in ejection buffer", "8");
  //LD/ST单元弹出缓冲器中的响应包数。
  option_parser_register(
      opp, "-gpgpu_n_ldst_response_buffer_size", OPT_UINT32,
      &ldst_unit_response_queue_size,
      "number of response packets in ld/st unit ejection buffer", "2");
  //每个线程块或CTA的共享内存大小（默认48KB）。
  option_parser_register(
      opp, "-gpgpu_shmem_per_block", OPT_UINT32, &gpgpu_shmem_per_block,
      "Size of shared memory per thread block or CTA (default 48kB)", "49152");
  //每个SIMT Core（也称为Shader Core）的共享存储大小。
  option_parser_register(
      opp, "-gpgpu_shmem_size", OPT_UINT32, &gpgpu_shmem_size,
      "Size of shared memory per shader core (default 16kB)", "16384");
  option_parser_register(opp, "-gpgpu_shmem_option", OPT_CSTR,
                         &gpgpu_shmem_option,
                         "Option list of shared memory sizes", "0");
  option_parser_register(
      opp, "-gpgpu_unified_l1d_size", OPT_UINT32,
      &m_L1D_config.m_unified_cache_size,
      "Size of unified data cache(L1D + shared memory) in KB", "0");
  option_parser_register(opp, "-gpgpu_adaptive_cache_config", OPT_BOOL,
                         &adaptive_cache_config, "adaptive_cache_config", "0");
  option_parser_register(
      opp, "-gpgpu_shmem_sizeDefault", OPT_UINT32, &gpgpu_shmem_sizeDefault,
      "Size of shared memory per shader core (default 16kB)", "16384");
  option_parser_register(
      opp, "-gpgpu_shmem_size_PrefL1", OPT_UINT32, &gpgpu_shmem_sizePrefL1,
      "Size of shared memory per shader core (default 16kB)", "16384");
  option_parser_register(opp, "-gpgpu_shmem_size_PrefShared", OPT_UINT32,
                         &gpgpu_shmem_sizePrefShared,
                         "Size of shared memory per shader core (default 16kB)",
                         "16384");
  option_parser_register(
      opp, "-gpgpu_shmem_num_banks", OPT_UINT32, &num_shmem_bank,
      "Number of banks in the shared memory in each shader core (default 16)",
      "16");
  option_parser_register(
      opp, "-gpgpu_shmem_limited_broadcast", OPT_BOOL, &shmem_limited_broadcast,
      "Limit shared memory to do one broadcast per cycle (default on)", "1");
  option_parser_register(opp, "-gpgpu_shmem_warp_parts", OPT_INT32,
                         &mem_warp_parts,
                         "Number of portions a warp is divided into for shared "
                         "memory bank conflict check ",
                         "2");
  option_parser_register(
      opp, "-gpgpu_mem_unit_ports", OPT_INT32, &mem_unit_ports,
      "The number of memory transactions allowed per core cycle", "1");
  option_parser_register(opp, "-gpgpu_shmem_warp_parts", OPT_INT32,
                         &mem_warp_parts,
                         "Number of portions a warp is divided into for shared "
                         "memory bank conflict check ",
                         "2");
  option_parser_register(
      opp, "-gpgpu_warpdistro_shader", OPT_INT32, &gpgpu_warpdistro_shader,
      "Specify which shader core to collect the warp size distribution from",
      "-1");
  option_parser_register(
      opp, "-gpgpu_warp_issue_shader", OPT_INT32, &gpgpu_warp_issue_shader,
      "Specify which shader core to collect the warp issue distribution from",
      "0");
  option_parser_register(opp, "-gpgpu_local_mem_map", OPT_BOOL,
                         &gpgpu_local_mem_map,
                         "Mapping from local memory space address to simulated "
                         "GPU physical address space (default = enabled)",
                         "1");
  option_parser_register(opp, "-gpgpu_num_reg_banks", OPT_INT32,
                         &gpgpu_num_reg_banks,
                         "Number of register banks (default = 8)", "8");
  option_parser_register(
      opp, "-gpgpu_reg_bank_use_warp_id", OPT_BOOL, &gpgpu_reg_bank_use_warp_id,
      "Use warp ID in mapping registers to banks (default = off)", "0");
  //在subcore模式下，每个warp调度器在寄存器集合中有一个具体的寄存器可供使用，这个寄
  //存器由调度器的m_id索引。
  option_parser_register(opp, "-gpgpu_sub_core_model", OPT_BOOL,
                         &sub_core_model,
                         "Sub Core Volta/Pascal model (default = off)", "0");
  option_parser_register(opp, "-gpgpu_enable_specialized_operand_collector",
                         OPT_BOOL, &enable_specialized_operand_collector,
                         "enable_specialized_operand_collector", "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_sp",
                         OPT_INT32, &gpgpu_operand_collector_num_units_sp,
                         "number of collector units (default = 4)", "4");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_dp",
                         OPT_INT32, &gpgpu_operand_collector_num_units_dp,
                         "number of collector units (default = 0)", "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_sfu",
                         OPT_INT32, &gpgpu_operand_collector_num_units_sfu,
                         "number of collector units (default = 4)", "4");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_int",
                         OPT_INT32, &gpgpu_operand_collector_num_units_int,
                         "number of collector units (default = 0)", "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_tensor_core",
                         OPT_INT32,
                         &gpgpu_operand_collector_num_units_tensor_core,
                         "number of collector units (default = 4)", "4");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_mem",
                         OPT_INT32, &gpgpu_operand_collector_num_units_mem,
                         "number of collector units (default = 2)", "2");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_gen",
                         OPT_INT32, &gpgpu_operand_collector_num_units_gen,
                         "number of collector units (default = 0)", "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_sp",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_sp,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_dp",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_dp,
                         "number of collector unit in ports (default = 0)",
                         "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_sfu",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_sfu,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_int",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_int,
                         "number of collector unit in ports (default = 0)",
                         "0");
  option_parser_register(
      opp, "-gpgpu_operand_collector_num_in_ports_tensor_core", OPT_INT32,
      &gpgpu_operand_collector_num_in_ports_tensor_core,
      "number of collector unit in ports (default = 1)", "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_mem",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_mem,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_gen",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_gen,
                         "number of collector unit in ports (default = 0)",
                         "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_sp",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_sp,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_dp",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_dp,
                         "number of collector unit in ports (default = 0)",
                         "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_sfu",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_sfu,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_int",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_int,
                         "number of collector unit in ports (default = 0)",
                         "0");
  option_parser_register(
      opp, "-gpgpu_operand_collector_num_out_ports_tensor_core", OPT_INT32,
      &gpgpu_operand_collector_num_out_ports_tensor_core,
      "number of collector unit in ports (default = 1)", "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_mem",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_mem,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_gen",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_gen,
                         "number of collector unit in ports (default = 0)",
                         "0");
  option_parser_register(opp, "-gpgpu_coalesce_arch", OPT_INT32,
                         &gpgpu_coalesce_arch,
                         "Coalescing arch (GT200 = 13, Fermi = 20)", "13");
  option_parser_register(opp, "-gpgpu_num_sched_per_core", OPT_INT32,
                         &gpgpu_num_sched_per_core,
                         "Number of warp schedulers per core", "1");
  option_parser_register(opp, "-gpgpu_max_insn_issue_per_warp", OPT_INT32,
                         &gpgpu_max_insn_issue_per_warp,
                         "Max number of instructions that can be issued per "
                         "warp in one cycle by scheduler (either 1 or 2)",
                         "2");
  option_parser_register(opp, "-gpgpu_dual_issue_diff_exec_units", OPT_BOOL,
                         &gpgpu_dual_issue_diff_exec_units,
                         "should dual issue use two different execution unit "
                         "resources (Default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_simt_core_sim_order", OPT_INT32,
                         &simt_core_sim_order,
                         "Select the simulation order of cores in a cluster "
                         "(0=Fix, 1=Round-Robin)",
                         "1");
  option_parser_register(
      opp, "-gpgpu_pipeline_widths", OPT_CSTR, &pipeline_widths_string,
      "Pipeline widths "
      "ID_OC_SP,ID_OC_DP,ID_OC_INT,ID_OC_SFU,ID_OC_MEM,OC_EX_SP,OC_EX_DP,OC_EX_"
      "INT,OC_EX_SFU,OC_EX_MEM,EX_WB,ID_OC_TENSOR_CORE,OC_EX_TENSOR_CORE",
      "1,1,1,1,1,1,1,1,1,1,1,1,1");
  option_parser_register(opp, "-gpgpu_tensor_core_avail", OPT_UINT32,
                         &gpgpu_tensor_core_avail,
                         "Tensor Core Available (default=0)", "0");
  option_parser_register(opp, "-gpgpu_num_sp_units", OPT_UINT32,
                         &gpgpu_num_sp_units, "Number of SP units (default=1)",
                         "1");
  option_parser_register(opp, "-gpgpu_num_dp_units", OPT_UINT32,
                         &gpgpu_num_dp_units, "Number of DP units (default=0)",
                         "0");
  option_parser_register(opp, "-gpgpu_num_int_units", OPT_UINT32,
                         &gpgpu_num_int_units,
                         "Number of INT units (default=0)", "0");
  option_parser_register(opp, "-gpgpu_num_sfu_units", OPT_UINT32,
                         &gpgpu_num_sfu_units, "Number of SF units (default=1)",
                         "1");
  option_parser_register(opp, "-gpgpu_num_tensor_core_units", OPT_UINT32,
                         &gpgpu_num_tensor_core_units,
                         "Number of tensor_core units (default=1)", "0");
  option_parser_register(
      opp, "-gpgpu_num_mem_units", OPT_UINT32, &gpgpu_num_mem_units,
      "Number if ldst units (default=1) WARNING: not hooked up to anything",
      "1");
  option_parser_register(
      opp, "-gpgpu_scheduler", OPT_CSTR, &gpgpu_scheduler_string,
      "Scheduler configuration: < lrr | gto | two_level_active > "
      "If "
      "two_level_active:<num_active_warps>:<inner_prioritization>:<outer_"
      "prioritization>"
      "For complete list of prioritization values see shader.h enum "
      "scheduler_prioritization_type"
      "Default: gto",
      "gto");

  option_parser_register(
      opp, "-gpgpu_concurrent_kernel_sm", OPT_BOOL, &gpgpu_concurrent_kernel_sm,
      "Support concurrent kernels on a SM (default = disabled)", "0");
  option_parser_register(opp, "-gpgpu_perfect_inst_const_cache", OPT_BOOL,
                         &perfect_inst_const_cache,
                         "perfect inst and const cache mode, so all inst and "
                         "const hits in the cache(default = disabled)",
                         "0");
  option_parser_register(
      opp, "-gpgpu_inst_fetch_throughput", OPT_INT32, &inst_fetch_throughput,
      "the number of fetched intruction per warp each cycle", "1");
  //寄存器文件的端口数。在V100配置文件里gpgpu_reg_file_port_throughput被设置为2。
  option_parser_register(opp, "-gpgpu_reg_file_port_throughput", OPT_INT32,
                         &reg_file_port_throughput,
                         "the number ports of the register file", "1");

  for (unsigned j = 0; j < SPECIALIZED_UNIT_NUM; ++j) {
    std::stringstream ss;
    ss << "-specialized_unit_" << j + 1;
    //-specialized_unit_1 0,4,4,4,4,BRA
    //-specialized_unit_2 0,4,4,4,4,BRA
    //-specialized_unit_3 0,4,4,4,4,BRA
    //-specialized_unit_4 0,4,4,4,4,BRA
    //-specialized_unit_5 0,4,4,4,4,BRA
    //-specialized_unit_6 0,4,4,4,4,BRA
    //-specialized_unit_7 0,4,4,4,4,BRA
    //-specialized_unit_8 0,4,4,4,4,BRA
    option_parser_register(opp, ss.str().c_str(), OPT_CSTR,
                           &specialized_unit_string[j],
                           "specialized unit config"
                           " {<enabled>,<num_units>:<latency>:<initiation>,<ID_"
                           "OC_SPEC>:<OC_EX_SPEC>,<NAME>}",
                           "0,4,4,4,4,BRA");
  }
}

/*
GPGPU-Sim 3.x提供了一个通用的命令行选项解析器，允许不同的软件模块通过一个简单的接口来注册他们的选项。
选项解析器在 gpgpusim_entrypoint.cc 的 gpgpu_ptx_sim_init_perf() 中实例化。选项在 reg_options() 
函数中使用函数添加。
*/
void gpgpu_sim_config::reg_options(option_parser_t opp) {
  gpgpu_functional_sim_config::reg_options(opp);
  m_shader_config.reg_options(opp);
  m_memory_config.reg_options(opp);
  power_config::reg_options(opp);
  //在达到最大周期数后尽早终止GPU模拟(0 = no limit)。-gpgpu_max_cycle <# cycles>
  option_parser_register(opp, "-gpgpu_max_cycle", OPT_INT64, &gpu_max_cycle_opt,
                         "terminates gpu simulation early (0 = no limit)", "0");
  //在达到最大指令数后尽早终止GPU模拟(0 = no limit)。-gpgpu_max_insn <# insns>
  option_parser_register(opp, "-gpgpu_max_insn", OPT_INT64, &gpu_max_insn_opt,
                         "terminates gpu simulation early (0 = no limit)", "0");
  //在达到最大CTA并发数后尽早终止GPU模拟(0 = no limit)。
  option_parser_register(opp, "-gpgpu_max_cta", OPT_INT32, &gpu_max_cta_opt,
                         "terminates gpu simulation early (0 = no limit)", "0");
  //在达到最大CTA完成数后尽早终止GPU模拟(0 = no limit)。
  option_parser_register(opp, "-gpgpu_max_completed_cta", OPT_INT32,
                         &gpu_max_completed_cta_opt,
                         "terminates gpu simulation early (0 = no limit)", "0");
  //显示运行时统计信息。-gpgpu_runtime_stat <frequency>:<flag> 
  option_parser_register(
      opp, "-gpgpu_runtime_stat", OPT_CSTR, &gpgpu_runtime_stat,
      "display runtime statistics such as dram utilization {<freq>:<flag>}",
      "10000:0");
  //模拟活动消息之间的最小秒数（0=始终打印）。
  option_parser_register(opp, "-liveness_message_freq", OPT_INT64,
                         &liveness_message_freq,
                         "Minimum number of seconds between simulation "
                         "liveness messages (0 = always print)",
                         "1");
  //最大的设备计算能力。
  option_parser_register(opp, "-gpgpu_compute_capability_major", OPT_UINT32,
                         &gpgpu_compute_capability_major,
                         "Major compute capability version number", "7");
  //最小的设备计算能力。
  option_parser_register(opp, "-gpgpu_compute_capability_minor", OPT_UINT32,
                         &gpgpu_compute_capability_minor,
                         "Minor compute capability version number", "0");
  //在每个内核调用结束时刷新L1缓存。
  option_parser_register(opp, "-gpgpu_flush_l1_cache", OPT_BOOL,
                         &gpgpu_flush_l1_cache,
                         "Flush L1 cache at the end of each kernel call", "0");
  //在每个内核调用结束时刷新L2缓存。
  option_parser_register(opp, "-gpgpu_flush_l2_cache", OPT_BOOL,
                         &gpgpu_flush_l2_cache,
                         "Flush L2 cache at the end of each kernel call", "0");
  //在死锁时停止模拟。-gpgpu_deadlock_detect <0=off, 1=on(default)>
  option_parser_register(
      opp, "-gpgpu_deadlock_detect", OPT_BOOL, &gpu_deadlock_detect,
      "Stop the simulation at deadlock (1=on (default), 0=off)", "1");
  //启用指令分类，如果启用，将对每个内核的ptx指令类型进行分类（现在最多255个内核）。
  //-gpgpu_ptx_instruction_classification <0=off, 1=on (default)>
  option_parser_register(
      opp, "-gpgpu_ptx_instruction_classification", OPT_INT32,
      &(gpgpu_ctx->func_sim->gpgpu_ptx_instruction_classification),
      "if enabled will classify ptx instruction types per kernel (Max 255 "
      "kernels now)",
      "0");
  //在性能或功能模拟之间进行选择（请注意，功能模拟可能会错误地模拟某些ptx代码，这些代码需要warp的每
  //个元素在lock-step中执行）。-gpgpu_ptx_sim_mode <0=performance(default), 1=functional>
  option_parser_register(
      opp, "-gpgpu_ptx_sim_mode", OPT_INT32,
      &(gpgpu_ctx->func_sim->g_ptx_sim_mode),
      "Select between Performance (default) or Functional simulation (1)", "0");
  //以MHz为单位的时钟域频率。
  //-gpgpu_clock_domains <Core Clock>:<Interconnect Clock>:<L2 Clock>:<DRAM Clock>
  option_parser_register(opp, "-gpgpu_clock_domains", OPT_CSTR,
                         &gpgpu_clock_domains,
                         "Clock Domain Frequencies in MhZ {<Core Clock>:<ICNT "
                         "Clock>:<L2 Clock>:<DRAM Clock>}",
                         "500.0:2000.0:2000.0:2000.0");
  //可以在GPU上同时运行的最大内核数。
  option_parser_register(
      opp, "-gpgpu_max_concurrent_kernel", OPT_INT32, &max_concurrent_kernel,
      "maximum kernels that can run concurrently on GPU, set this value "
      "according to max resident grids for your compute capability",
      "32");
  //控制流记录器中每个快照之间的间隔。
  option_parser_register(
      opp, "-gpgpu_cflog_interval", OPT_INT32, &gpgpu_cflog_interval,
      "Interval between each snapshot in control flow logger", "0");
  //打开可视化工具输出（使用AerialVision可视化工具绘制保存在日志中的数据）。
  option_parser_register(opp, "-visualizer_enabled", OPT_BOOL,
                         &g_visualizer_enabled,
                         "Turn on visualizer output (1=On, 0=Off)", "1");
  //指定可视化工具的输出日志文件。
  option_parser_register(opp, "-visualizer_outputfile", OPT_CSTR,
                         &g_visualizer_filename,
                         "Specifies the output log file for visualizer", NULL);
  //可视化工具输出日志的压缩级别（0=无压缩，9=最大压缩）。
  option_parser_register(
      opp, "-visualizer_zlevel", OPT_INT32, &g_visualizer_zlevel,
      "Compression level of the visualizer output log (0=no comp, 9=highest)",
      "6");
  //GPU线程堆栈大小。
  option_parser_register(opp, "-gpgpu_stack_size_limit", OPT_INT32,
                         &stack_size_limit, "GPU thread stack size", "1024");
  //GPU malloc堆大小。
  option_parser_register(opp, "-gpgpu_heap_size_limit", OPT_INT32,
                         &heap_size_limit, "GPU malloc heap size ", "8388608");
  //GPU设备运行时同步深度限制。
  option_parser_register(opp, "-gpgpu_runtime_sync_depth_limit", OPT_INT32,
                         &runtime_sync_depth_limit,
                         "GPU device runtime synchronize depth", "2");
  //GPU设备运行时挂起启动计数限制。
  option_parser_register(opp, "-gpgpu_runtime_pending_launch_count_limit",
                         OPT_INT32, &runtime_pending_launch_count_limit,
                         "GPU device runtime pending launch count", "2048");
  //全局启用或禁用所有trace。如果启用，则打印 trace_components 。
  option_parser_register(opp, "-trace_enabled", OPT_BOOL, &Trace::enabled,
                         "Turn on traces", "0");
  //要启用trance的逗号分隔列表，完整列表可在 src/trace_streams.tup 中找到。
  option_parser_register(opp, "-trace_components", OPT_CSTR, &Trace::config_str,
                         "comma seperated list of traces to enable. "
                         "Complete list found in trace_streams.tup. "
                         "Default none",
                         "none");
  //对于与给定shader core关联的元素（如warp调度器或记分牌），仅打印来自该核心的trace。
  option_parser_register(
      opp, "-trace_sampling_core", OPT_INT32, &Trace::sampling_core,
      "The core which is printed using CORE_DPRINTF. Default 0", "0");
  option_parser_register(opp, "-trace_sampling_memory_partition", OPT_INT32,
                         &Trace::sampling_memory_partition,
                         "The memory partition which is printed using "
                         "MEMPART_DPRINTF. Default -1 (i.e. all)",
                         "-1");
  gpgpu_ctx->stats->ptx_file_line_stats_options(opp);

  // Jin: kernel launch latency
  //内核启动延迟（以周期为单位）。
  option_parser_register(opp, "-gpgpu_kernel_launch_latency", OPT_INT32,
                         &(gpgpu_ctx->device_runtime->g_kernel_launch_latency),
                         "Kernel launch latency in cycles. Default: 0", "0");
  //开启CDP。
  option_parser_register(opp, "-gpgpu_cdp_enabled", OPT_BOOL,
                         &(gpgpu_ctx->device_runtime->g_cdp_enabled),
                         "Turn on CDP", "0");
  //线程块启动延迟（以周期为单位）。
  option_parser_register(opp, "-gpgpu_TB_launch_latency", OPT_INT32,
                         &(gpgpu_ctx->device_runtime->g_TB_launch_latency),
                         "thread block launch latency in cycles. Default: 0",
                         "0");
}

/////////////////////////////////////////////////////////////////////////////

void increment_x_then_y_then_z(dim3 &i, const dim3 &bound) {
  i.x++;
  if (i.x >= bound.x) {
    i.x = 0;
    i.y++;
    if (i.y >= bound.y) {
      i.y = 0;
      if (i.z < bound.z) i.z++;
    }
  }
}


/*
This function launches the kernel specified by kinfo. It sets up the kernel's thread blocks, 
threads, and warps, and then launches them on the device. After the kernel has been launched, 
the function will wait until all of the threads have completed. Once the threads have 
completed, the function will clean up the resources used by the kernel.
此函数启动kinfo指定的内核。它设置内核的线程块、线程和warp，然后在设备上启动它们。内核启动后，该函数将
等待所有线程完成。线程完成后，该函数将清理内核使用的资源。
kernel_info_t类在abstract_hardware_model.h中定义。
*/
void gpgpu_sim::launch(kernel_info_t *kinfo) {
  unsigned kernelID = kinfo->get_uid();
  unsigned long long streamID = kinfo->get_streamID();

  kernel_time_t kernel_time = {gpu_tot_sim_cycle + gpu_sim_cycle, 0};
  if (gpu_kernel_time.find(streamID) == gpu_kernel_time.end()) {
    std::map<unsigned, kernel_time_t> new_val;
    new_val.insert(std::pair<unsigned, kernel_time_t>(kernelID, kernel_time));
    gpu_kernel_time.insert(
        std::pair<unsigned long long, std::map<unsigned, kernel_time_t>>(
            streamID, new_val));
  } else {
    gpu_kernel_time.at(streamID).insert(
        std::pair<unsigned, kernel_time_t>(kernelID, kernel_time));
    ////////// assume same kernel ID do not appear more than once
  }
  //根据内核函数的信息kinfo获取其参数中的每个CTA（线程块）中的线程数。
  unsigned cta_size = kinfo->threads_per_cta();
  //如果程序的每个CTA中的线程数量 > 每个SIMT Core配置的线程数（由-gpgpu_shader配置），输出错误信
  //息。
  if (cta_size > m_shader_config->n_thread_per_shader) {
    printf(
        "Execution error: Shader kernel CTA (block) size is too large for "
        "microarch config.\n");
    printf("                 CTA size (x*y*z) = %u, max supported = %u\n",
           cta_size, m_shader_config->n_thread_per_shader);
    printf(
        "                 => either change -gpgpu_shader argument in "
        "gpgpusim.config file or\n");
    printf(
        "                 modify the CUDA source to decrease the kernel block "
        "size.\n");
    abort();
  }
  //m_running_kernels由gpu-sim.h中的 std::vector<kernel_info_t *> 定义：
  //    std::vector<kernel_info_t *> m_running_kernels;
  //是一组kernel_info_t*组成的向量，它存储着正在运行的内核的信息。下面对这个向量遍历，找到一个空位，
  //加入新的即将运行的内核kinfo。如果向量的某个位置为NULL或者该位置->done()显示该核函数已完成，则将
  //kinfo加入到此位置。
  unsigned n = 0;
  for (n = 0; n < m_running_kernels.size(); n++) {
    if ((NULL == m_running_kernels[n]) || m_running_kernels[n]->done()) {
      m_running_kernels[n] = kinfo;
      break;
    }
  }
  assert(n < m_running_kernels.size());
}

/*
gpgpu_sim::can_start_kernel()是一个函数，它检查当前GPU是否有足够的资源来启动一个新的内核。如果有
足够的资源，该函数将返回true，否则返回false。这里的资源检查，主要是看存储着正在运行的内核的信息的
m_running_kernels向量里是否有位置可以加入新内核，如果向量的某个位置为NULL或者该位置->done()显示该
核函数已完成，则可以加载新内核。
*/
bool gpgpu_sim::can_start_kernel() {
  for (unsigned n = 0; n < m_running_kernels.size(); n++) {
    if ((NULL == m_running_kernels[n]) || m_running_kernels[n]->done())
      return true;
  }
  return false;
}

/*
gpu_max_cta_opt选项是指的是，GPGPU-Sim所能达到最大CTA并发数(0 = no limit)，在配置选项中定义。
gpu_tot_issued_cta即总发出的CTA（Compute Thread Array）亦即线程块数量，加上m_total_cta_launched
即已经启动的CTA数量要大于或等于GPU的最大CTA并发数量（m_config.gpu_max_cta_opt），这是为了确保GPU的
最大性能。
*/
bool gpgpu_sim::hit_max_cta_count() const {
  if (m_config.gpu_max_cta_opt != 0) {
    if ((gpu_tot_issued_cta + m_total_cta_launched) >= m_config.gpu_max_cta_opt)
      return true;
  }
  return false;
}

/*
该函数用于检查指定的内核是否有更多的CTA（Compute Thread Array）需要执行。如果还有更多的CTA需要执行
，则函数返回true；如果没有更多的CTA需要执行，则函数返回false。若已经达到GPU模拟过程中最大的CTA限制
数（由gpgpu_sim::hit_max_cta_count()判断），则没有剩余的CTA，返回False；若kernel非空，且kernel->
no_more_ctas_to_run()为false即kernel自己还可有多余CTA执行，则返回True。no_more_ctas_to_run()函数
指示当前没有更多的CTA（Compute Thread Array）需要执行。 
*/
bool gpgpu_sim::kernel_more_cta_left(kernel_info_t *kernel) const {
  if (hit_max_cta_count()) return false;

  if (kernel && !kernel->no_more_ctas_to_run()) return true;

  return false;
}

/*
该函数用于检查当前是否还有更多的CTA（Compute Thread Array）需要执行。它检查当前活跃的CTA数量，并返
回是否有更多CTA需要执行。如果已达到GPU模拟限制最大的CTA数（由gpgpu_sim::hit_max_cta_count()判断），
则没有剩余的CTA，返回False；如果某个m_running_kernels向量里的kernel非空，且kernel->
no_more_ctas_to_run()为false即kernel自己还可有多余CTA执行，则返回True。no_more_ctas_to_run()函数
指示当前没有更多的CTA（Compute Thread Array）需要执行。
*/
bool gpgpu_sim::get_more_cta_left() const {
  if (hit_max_cta_count()) return false;

  for (unsigned n = 0; n < m_running_kernels.size(); n++) {
    if (m_running_kernels[n] && !m_running_kernels[n]->no_more_ctas_to_run())
      return true;
  }
  return false;
}

/*
该函数用于减少内核延迟（kernel latency），即减少从发出内核命令到内核完成执行的时间（对所有正在运行的
内核的延迟减1）。该函数可以用于模拟GPU内核的性能，以及分析程序的性能。m_kernel_TB_latency表示每一个
线程块的启动延迟之和加上kernel的启动延迟，即从发出线程块到它完成执行的时间。m_kernel_TB_latency是
GPGPU-Sim中用于表示内核启动时间的变量，它表示从内核启动到内核完成执行所需要的时间。
*/
void gpgpu_sim::decrement_kernel_latency() {
  for (unsigned n = 0; n < m_running_kernels.size(); n++) {
    // m_kernel_TB_latency =
    //     entry->gpgpu_ctx->device_runtime->g_kernel_launch_latency +
    //     num_blocks() * entry->gpgpu_ctx->device_runtime->g_TB_launch_latency;
    if (m_running_kernels[n] && m_running_kernels[n]->m_kernel_TB_latency)
      m_running_kernels[n]->m_kernel_TB_latency--;
  }
}

/*
m_last_issued_kernel内部变量表示最近一次发出的内核。它的值是一个指向内核的指针，可以用来跟踪内核的执
行进度。m_running_kernels[m_last_issued_kernel]指向的是上一次发出的内核。该函数用于从当前活动内核
列表中选择最优的内核，以便将其分配给GPU。它返回一个指向内核信息结构的指针，该结构包含有关内核的所有信
息，包括内核名称，参数，线程数，块数等。
*/
kernel_info_t *gpgpu_sim::select_kernel() {
  //如果该内核非空，且该内核有更多的CTA（Compute Thread Array）需要运行，且其m_kernel_TB_latency为
  //零，则m_last_issued_kernel可以被优先选择。如果不满足这些条件，则会从所有正在运行的内核中选择。
  //m_kernel_TB_latency表示从内核启动到内核完成执行所需要的时间，如果这个值不为零，则代表这个内核尚未
  //执行完，则可以被调度执行。
  if (m_running_kernels[m_last_issued_kernel] &&
      !m_running_kernels[m_last_issued_kernel]->no_more_ctas_to_run() &&
      !m_running_kernels[m_last_issued_kernel]->m_kernel_TB_latency) {
    //get_uid()返回一个唯一的32位整数，用于标识不同的GPU内核（每个内核有一个独立的id表示，即uid）。
    unsigned launch_uid = m_running_kernels[m_last_issued_kernel]->get_uid();
    //m_executed_kernel_uids存储了所有已经执行完毕的内核的唯一标识符，即uid。std::find函数如果没在
    //已经执行完毕的内核列表中找到该内核，则说明它还没被执行完，可以被选择执行。
    if (std::find(m_executed_kernel_uids.begin(), m_executed_kernel_uids.end(),
                  launch_uid) == m_executed_kernel_uids.end()) {
      //gpu_tot_sim_cycle变量表示当前的仿真周期，即从仿真开始到当前的总仿真时间。gpu_sim_cycle变量
      //表示执行此内核所需的时钟周期数（以shader core的时钟域为单位）。
      //m_running_kernels[m_last_issued_kernel]->start_cycle变量表示最后一个发布的内核的开始周期，
      //用于跟踪内核的执行情况，以便计算内核的总执行时间。
      //选择此内核执行以后，更新m_running_kernels[m_last_issued_kernel]->start_cycle以便下次选择
      //新内核执行时使用，这样一个内核一个内核的累加时间。
      //同时，m_running_kernels[m_last_issued_kernel]的状态更新为executed，把其uid压入m_executed
      //_kernel_uids，把其name压入m_executed_kernel_names。

      //下面这句话的意思是，gpu_sim_cycle代表着上一次执行的内核的延迟，gpu_tot_sim_cycle代表着上一次
      //执行的内核之前的所有内核的执行时间，因此当前内核的开始启动时间即为二者相加。gpgpu_sim::cycle()
      //每过一拍将gpu_sim_cycle++。
      m_running_kernels[m_last_issued_kernel]->start_cycle =
          gpu_sim_cycle + gpu_tot_sim_cycle;
      m_executed_kernel_uids.push_back(launch_uid);
      m_executed_kernel_names.push_back(
          m_running_kernels[m_last_issued_kernel]->name());
    }
    return m_running_kernels[m_last_issued_kernel];
  }
  //m_last_issued_kernel不满足被优先选择执行的条件时，则从所有正在运行的内核中选择，选择策略是如下方
  //式计算(n + m_last_issued_kernel + 1) % m_config.max_concurrent_kernel。即顺序选择上一次发出的
  //m_last_issued_kernel在m_running_kernels中的下一个编号的内核，依次轮询。max_concurrent_kernel
  //表示模拟的GPU上可能并发执行的最大内核数量。
  for (unsigned n = 0; n < m_running_kernels.size(); n++) {
    unsigned idx =
        (n + m_last_issued_kernel + 1) % m_config.max_concurrent_kernel;
    //如果idx标识的内核有更多的CTA需要执行，且其执行延迟m_kernel_TB_latency尚未减到0，即它还在执行，
    //则可以发出。
    if (kernel_more_cta_left(m_running_kernels[idx]) &&
        !m_running_kernels[idx]->m_kernel_TB_latency) {
      m_last_issued_kernel = idx;
      //gpu_sim_cycle变量表示执行此内核所需的时钟周期数（以shader core的时钟域为单位），gpgpu_sim::
      //cycle()每过一拍将gpu_sim_cycle++。
      m_running_kernels[idx]->start_cycle = gpu_sim_cycle + gpu_tot_sim_cycle;
      // record this kernel for stat print if it is the first time this kernel
      // is selected for execution
      unsigned launch_uid = m_running_kernels[idx]->get_uid();
      assert(std::find(m_executed_kernel_uids.begin(),
                       m_executed_kernel_uids.end(),
                       launch_uid) == m_executed_kernel_uids.end());
      m_executed_kernel_uids.push_back(launch_uid);
      m_executed_kernel_names.push_back(m_running_kernels[idx]->name());

      return m_running_kernels[idx];
    }
  }
  return NULL;
}

/*
从已结束的内核队列m_finished_kernel中选择头部的内核并返回。
*/
unsigned gpgpu_sim::finished_kernel() {
  //m_finished_kernel.empty()为1表示当前暂时无已经结束的内核。
  if (m_finished_kernel.empty()) {
    last_streamID = -1;
    return 0;
  }
  //选择头部的内核。
  unsigned result = m_finished_kernel.front();
  m_finished_kernel.pop_front();
  return result;
}

/*
设置结束的内核的状态，包括将m_running_kernels中该内核剔除掉，以及更新其结束的时钟周期。
*/
void gpgpu_sim::set_kernel_done(kernel_info_t *kernel) {
  unsigned uid = kernel->get_uid();
  last_uid = uid;
  unsigned long long streamID = kernel->get_streamID();
  last_streamID = streamID;
  gpu_kernel_time.at(streamID).at(uid).end_cycle =
      gpu_tot_sim_cycle + gpu_sim_cycle;
  m_finished_kernel.push_back(uid);
  std::vector<kernel_info_t *>::iterator k;
  for (k = m_running_kernels.begin(); k != m_running_kernels.end(); k++) {
    if (*k == kernel) {
      kernel->end_cycle = gpu_sim_cycle + gpu_tot_sim_cycle;
      *k = NULL;
      break;
    }
  }
  assert(k != m_running_kernels.end());
}

/*
停止所有正在运行的内核。
*/
void gpgpu_sim::stop_all_running_kernels() {
  std::vector<kernel_info_t *>::iterator k;
  for (k = m_running_kernels.begin(); k != m_running_kernels.end(); ++k) {
    if (*k != NULL) {       // If a kernel is active
      set_kernel_done(*k);  // Stop the kernel
      assert(*k == NULL);
    }
  }
}

/*
创建SIMT Cluster。m_cluster[...]存储了所有的SM。
  1.m_shader_config：定义shader Core的配置，包括每个shader处理器的指令宽度、数据宽度、指令缓存大小
    等；
  2.m_memory_config：定义存储模块的配置，包括每个存储模块的读写带宽、latency等；
  3.m_shader_stats：定义shader处理器的统计信息，包括每个shader处理器的指令执行次数、指令缓存命中次
    数等；
  4.m_memory_stats：定义存储模块的统计信息，包括每个存储模块的读写次数、cache命中次数等；
*/
void exec_gpgpu_sim::createSIMTCluster() {
  //m_cluster在gpu-sim.h中定义：class simt_core_cluster **m_cluster;
  //n_simt_clusters是配置中的SM的数量，由于配置中可配置每个集群含有多个SM，采用二维数组。
  m_cluster = new simt_core_cluster *[m_shader_config->n_simt_clusters];
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
    m_cluster[i] =
        new exec_simt_core_cluster(this, i, m_shader_config, m_memory_config,
                                   m_shader_stats, m_memory_stats);
}

/*
性能仿真引擎是通过 src/gpgpu-sim 下的文件中定义和实现的许多类来实现的。这些类通过顶层类 gpgpu_sim 
汇集在一起，该类是由 gpgpu_t （其功能仿真对应部分）派生的。在当前版本的GPGPU-Sim中，模拟器中只有一个 
gpgpu_sim 的实例 g_the_gpu。目前不支持同时对多个GPU进行仿真，但在未来的版本中可能会提供。
*/
gpgpu_sim::gpgpu_sim(const gpgpu_sim_config &config, gpgpu_context *ctx)
    : gpgpu_t(config, ctx), m_config(config) {
  //gpgpu_context *ctx在libcuda/gpgpu_context.h中定义。
  gpgpu_ctx = ctx;
  m_shader_config = &m_config.m_shader_config;
  m_memory_config = &m_config.m_memory_config;
  ctx->ptx_parser->set_ptx_warp_size(m_shader_config);
  //m_config.num_shader()返回硬件所有的SM（又称Shader Core）的总数。
  ptx_file_line_stats_create_exposed_latency_tracker(m_config.num_shader());

#ifdef GPGPUSIM_POWER_MODEL
  m_gpgpusim_wrapper = new gpgpu_sim_wrapper(
      config.g_power_simulation_enabled, config.g_power_config_name,
      config.g_power_simulation_mode, config.g_dvfs_enabled);
#endif

  m_shader_stats = new shader_core_stats(m_shader_config);
  m_memory_stats = new memory_stats_t(m_config.num_shader(), m_shader_config,
                                      m_memory_config, this);
  average_pipeline_duty_cycle = (float *)malloc(sizeof(float));
  active_sms = (float *)malloc(sizeof(float));
  m_power_stats =
      new power_stat_t(m_shader_config, average_pipeline_duty_cycle, active_sms,
                       m_shader_stats, m_memory_config, m_memory_stats);
  //在此内核中执行的指令数。
  gpu_sim_insn = 0;
  //迄今为止为所有启动的内核模拟的总周期数（以核心时钟为单位）。
  gpu_tot_sim_insn = 0;
  //即总发出的CTA（Compute Thread Array）亦即线程块数量。
  gpu_tot_issued_cta = 0;
  //已经完成的CTA的总数。
  gpu_completed_cta = 0;
  //已经启动的CTA数量。
  m_total_cta_launched = 0;
  //GPU进入死锁状态的标志。
  gpu_deadlock = false;
  //互连网络输出到DRAM Channel的暂停周期数。请求由icnt发送至L2_queue时，m_icnt_L2_queue没有SECTOR_
  //CHUNCK_SIZE大小的空间可以保存请求信息，因此互连网络的拥塞造成DRAM的停滞次数。
  gpu_stall_dramfull = 0;
  //由于互连拥塞导致DRAM Channel停滞的周期数。在从存储控制器向互连网络弹出时，如果互连网络中有空闲
  //的缓冲区，则将内存请求推入互连网络。但是一旦互连网络中的缓冲区被占满，就会停止推送。由于互连网
  //络缓冲区的大小限制造成的停顿时钟周期数由gpu_stall_icnt2sh计数保存下来。
  gpu_stall_icnt2sh = 0;
  //自模拟器启动的第一个时钟周期开始，partiton_reqs_in_parallel就根据每一拍的内存分区产生的请求开始
  //计数。partiton_reqs_in_parallel表示自模拟器启动的第一个时钟周期开始后所有存储分区产生的请求被推
  //入m_memory_sub_partition的m_icnt_L2_queue的总个数。
  partiton_reqs_in_parallel = 0;
  partiton_reqs_in_parallel_total = 0;
  partiton_reqs_in_parallel_util = 0;
  partiton_reqs_in_parallel_util_total = 0;
  gpu_sim_cycle_parition_util = 0;
  gpu_tot_sim_cycle_parition_util = 0;
  partiton_replys_in_parallel = 0;
  partiton_replys_in_parallel_total = 0;
  last_streamID = -1;

  gpu_kernel_time.clear();

  m_memory_partition_unit =
      new memory_partition_unit *[m_memory_config->m_n_mem];
  m_memory_sub_partition =
      new memory_sub_partition *[m_memory_config->m_n_mem_sub_partition];
  for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
    m_memory_partition_unit[i] =
        new memory_partition_unit(i, m_memory_config, m_memory_stats, this);
    for (unsigned p = 0;
         p < m_memory_config->m_n_sub_partition_per_memory_channel; p++) {
      unsigned submpid =
          i * m_memory_config->m_n_sub_partition_per_memory_channel + p;
      m_memory_sub_partition[submpid] =
          m_memory_partition_unit[i]->get_sub_partition(p);
    }
  }

  //初始化互连网络的配置，指定互连网络的类型以及选择对应的Push/Pop等流程。
  icnt_wrapper_init();
  //创建互连网络。
  icnt_create(m_shader_config->n_simt_clusters,
              m_memory_config->m_n_mem_sub_partition);

  time_vector_create(NUM_MEM_REQ_STAT);
  fprintf(stdout,
          "GPGPU-Sim uArch: performance model initialization complete.\n");

  m_running_kernels.resize(config.max_concurrent_kernel, NULL);
  m_last_issued_kernel = 0;
  m_last_cluster_issue = m_shader_config->n_simt_clusters -
                         1;  // this causes first launch to use simt cluster 0
  *average_pipeline_duty_cycle = 0;
  *active_sms = 0;

  last_liveness_message_time = 0;

  // Jin: functional simulation for CDP
  m_functional_sim = false;
  m_functional_sim_kernel = NULL;
}

/*
获取每个SIMT Core（也称为Shader Core）的共享存储大小。由GPGPU-Sim的-gpgpu_shmem_size选项配置。
*/
int gpgpu_sim::shared_mem_size() const {
  return m_shader_config->gpgpu_shmem_size;
}

/*
获取每个线程块或CTA的共享内存大小（默认48KB）。由GPGPU-Sim的-gpgpu_shmem_per_block选项配置。
*/
int gpgpu_sim::shared_mem_per_block() const {
  return m_shader_config->gpgpu_shmem_per_block;
}

/*
获取每个Shader Core的寄存器数。并发CTA的限制数量。由GPGPU-Sim的-gpgpu_shader_registers选项配置。
*/
int gpgpu_sim::num_registers_per_core() const {
  return m_shader_config->gpgpu_shader_registers;
}

/*
获取每个CTA的最大寄存器数。由GPGPU-Sim的-gpgpu_registers_per_block选项配置。
*/
int gpgpu_sim::num_registers_per_block() const {
  return m_shader_config->gpgpu_registers_per_block;
}

/*
获取一个warp有多少线程数。由GPGPU-Sim的-gpgpu_shader_core_pipeline的第二个选项配置。
选项-gpgpu_shader_core_pipeline的参数分别是：<每个SM最大可支配线程数>:<定义一个warp有多少线程>
*/
int gpgpu_sim::wrp_size() const { return m_shader_config->warp_size; }

/*
获取以MHz为单位的时钟域频率的<Core Clock>。由GPGPU-Sim的-gpgpu_clock_domains的第一个选项配置。
*/
int gpgpu_sim::shader_clock() const { return m_config.core_freq / 1000; }

/*
获取Shader Core中并发cta的最大数量。由GPGPU-Sim的-gpgpu_shader_cta选项配置。
*/
int gpgpu_sim::max_cta_per_core() const {
  return m_shader_config->max_cta_per_core;
}

/*
返回一个Core上可同时调度的最大线程块（或称为CTA）的数量，它由函数shader_core_config::max_cta(...)计
算。max_cta(...)函数根据程序指定的每个线程块的数量、每个线程寄存器的使用情况、共享内存的使用情况以及配
置的每个Core最大线程块数量的限制，确定可以并发分配给单个SIMT Core的最大线程块数量。具体说，如果上述每
个标准都是限制因素，那么可以分配给SIMT Core的线程块的数量被计算出来。其中的最小值就是可以分配给SIMT 
Core的最大线程块数。
*/
int gpgpu_sim::get_max_cta(const kernel_info_t &k) const {
  return m_shader_config->max_cta(k);
}

/*
m_cuda_properties变量是一个结构体，用于存储CUDA设备的性能和功能特性，包括最大线程数、最大块大小、最大
纹理大小等。
*/
void gpgpu_sim::set_prop(cudaDeviceProp *prop) { m_cuda_properties = prop; }

/*
返回最大的设备计算能力。由-gpgpu_compute_capability_major选项配置。
*/
int gpgpu_sim::compute_capability_major() const {
  return m_config.gpgpu_compute_capability_major;
}

/*
返回最小的设备计算能力。由-gpgpu_compute_capability_minor选项配置。
*/
int gpgpu_sim::compute_capability_minor() const {
  return m_config.gpgpu_compute_capability_minor;
}

/*
返回m_cuda_properties变量结构体，用于存储CUDA设备的性能和功能特性。
*/
const struct cudaDeviceProp *gpgpu_sim::get_prop() const {
  return m_cuda_properties;
}

enum divergence_support_t gpgpu_sim::simd_model() const {
  return m_shader_config->model;
}

/*
初始化时钟域。GPGPU-Sim支持四个独立的时钟域：
（1）SIMT Core集群时钟域，core_freq;
（2）互连网络时钟域，icnt_freq;
（3）L2高速缓存时钟域，适用于内存分区单元中除DRAM之外的所有逻辑，l2_freq;
（4）DRAM时钟域，dram_freq。
*/
void gpgpu_sim_config::init_clock_domains(void) {
  sscanf(gpgpu_clock_domains, "%lf:%lf:%lf:%lf", &core_freq, &icnt_freq,
         &l2_freq, &dram_freq);
  core_freq = core_freq MhZ;
  icnt_freq = icnt_freq MhZ;
  l2_freq = l2_freq MhZ;
  dram_freq = dram_freq MhZ;
  //周期。
  core_period = 1 / core_freq;
  icnt_period = 1 / icnt_freq;
  dram_period = 1 / dram_freq;
  l2_period = 1 / l2_freq;
  printf("GPGPU-Sim uArch: clock freqs: %lf:%lf:%lf:%lf\n", core_freq,
         icnt_freq, l2_freq, dram_freq);
  printf("GPGPU-Sim uArch: clock periods: %.20lf:%.20lf:%.20lf:%.20lf\n",
         core_period, icnt_period, l2_period, dram_period);
}

/*
重新初始化下一个上升沿的时刻。即将当前时刻四个时钟域的时刻值重置零。
*/
void gpgpu_sim::reinit_clock_domains(void) {
  core_time = 0;
  dram_time = 0;
  icnt_time = 0;
  l2_time = 0;
}

/*
返回GPGPU-Sim模拟器是否处于活跃状态。
*/
bool gpgpu_sim::active() {
  //gpu_max_cycle_opt选项配置：在达到最大周期数后尽早终止GPU模拟。
  //gpu_sim_cycle是执行当前阶段的指令的延迟，gpgpu_sim::cycle()每过一拍将gpu_sim_cycle++。
  //gpu_tot_sim_cycle是执行当前阶段之前的所有前绪指令的延迟。
  //两项延迟相加 >= gpu_max_cycle_opt说明会达到最大周期数，返回False。
  if (m_config.gpu_max_cycle_opt &&
      (gpu_tot_sim_cycle + gpu_sim_cycle) >= m_config.gpu_max_cycle_opt)
    return false;
  //gpu_max_insn_opt选项配置：在达到最大指令数后尽早终止GPU模拟。
  //gpu_sim_insn是执行当前阶段的指令的总数，比如将各个warp的相加。
  //gpu_tot_sim_insn是执行当前阶段之前的所有前绪指令的总数。
  //两项总数相加 >= gpu_max_insn_opt说明会达到最大指令数，返回False。
  if (m_config.gpu_max_insn_opt &&
      (gpu_tot_sim_insn + gpu_sim_insn) >= m_config.gpu_max_insn_opt)
    return false;
  //gpu_max_cta_opt选项配置：GPGPU-Sim所能达到最大CTA并发数尽早终止GPU模拟。
  //gpu_tot_issued_cta即总发出的CTA（Compute Thread Array）亦即线程块数量。
  //总发出的CTA数量 >= gpu_max_cta_opt，返回False。
  if (m_config.gpu_max_cta_opt &&
      (gpu_tot_issued_cta >= m_config.gpu_max_cta_opt))
    return false;
  //gpu_max_completed_cta_opt选项配置：在达到最大CTA完成数后尽早终止GPU模拟。
  //gpu_completed_cta是已经完成的CTA的总数。
  //已经完成的CTA的总数 >= gpu_max_completed_cta_opt，返回False。
  if (m_config.gpu_max_completed_cta_opt &&
      (gpu_completed_cta >= m_config.gpu_max_completed_cta_opt))
    return false;
  //gpu_deadlock_detect选项配置：在死锁时停止模拟。
  if (m_config.gpu_deadlock_detect && gpu_deadlock) return false;
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
    if (m_cluster[i]->get_not_completed() > 0) return true;
  ;
  for (unsigned i = 0; i < m_memory_config->m_n_mem; i++)
    if (m_memory_partition_unit[i]->busy() > 0) return true;
  ;
  //icnt_busy()判断互连网络是否处于Busy状态。有任意一个子网络处于Busy状态便认为
  //整个互连网络处于Busy状态。
  if (icnt_busy()) return true;
  if (get_more_cta_left()) return true;
  return false;
}

/*
初始化GPGPU-Sim的配置参数。
*/
void gpgpu_sim::init() {
  // run a CUDA grid on the GPU microarchitecture simulator
  //执行当前阶段的指令的延迟，gpgpu_sim::cycle()每过一拍将gpu_sim_cycle++。
  gpu_sim_cycle = 0;
  //执行当前阶段的指令的总数，比如将各个warp的相加。
  gpu_sim_insn = 0;
  last_gpu_sim_insn = 0;
  //已经启动的CTA数量。
  m_total_cta_launched = 0;
  //已经完成的CTA的总数。
  gpu_completed_cta = 0;
  //自模拟器启动的第一个时钟周期开始后所有存储分区产生的请求被推入m_memory_sub_partition的
  //m_icnt_L2_queue的总个数。
  partiton_reqs_in_parallel = 0;
  partiton_replys_in_parallel = 0;
  partiton_reqs_in_parallel_util = 0;
  gpu_sim_cycle_parition_util = 0;

// McPAT initialization function. Called on first launch of GPU
#ifdef GPGPUSIM_POWER_MODEL
  if (m_config.g_power_simulation_enabled) {
    init_mcpat(m_config, m_gpgpusim_wrapper, m_config.gpu_stat_sample_freq,
               gpu_tot_sim_insn, gpu_sim_insn);
  }
#endif

  reinit_clock_domains();
  gpgpu_ctx->func_sim->set_param_gpgpu_num_shaders(m_config.num_shader());
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
    m_cluster[i]->reinit();
  m_shader_stats->new_grid();
  // initialize the control-flow, memory access, memory latency logger
  if (m_config.g_visualizer_enabled) {
    create_thread_CFlogger(gpgpu_ctx, m_config.num_shader(),
                           m_shader_config->n_thread_per_shader, 0,
                           m_config.gpgpu_cflog_interval);
  }
  shader_CTA_count_create(m_config.num_shader(), m_config.gpgpu_cflog_interval);
  if (m_config.gpgpu_cflog_interval != 0) {
    insn_warp_occ_create(m_config.num_shader(), m_shader_config->warp_size);
    shader_warp_occ_create(m_config.num_shader(), m_shader_config->warp_size,
                           m_config.gpgpu_cflog_interval);
    shader_mem_acc_create(m_config.num_shader(), m_memory_config->m_n_mem, 4,
                          m_config.gpgpu_cflog_interval);
    shader_mem_lat_create(m_config.num_shader(), m_config.gpgpu_cflog_interval);
    shader_cache_access_create(m_config.num_shader(), 3,
                               m_config.gpgpu_cflog_interval);
    set_spill_interval(m_config.gpgpu_cflog_interval * 40);
  }

  if (g_network_mode) icnt_init();
}

/*
 * This function updates the statistics of the GPU simulator.
 * 
 * Parameters:
 *    gpu_sim * gpu: pointer to the GPU simulator object
 *    unsigned long long sim_cycle: current simulation cycle
 * 
 * Returns:
 *    void
 */
void gpgpu_sim::update_stats() {
  m_memory_stats->memlatstat_lat_pw();
  gpu_tot_sim_cycle += gpu_sim_cycle;
  gpu_tot_sim_insn += gpu_sim_insn;
  gpu_tot_issued_cta += m_total_cta_launched;
  partiton_reqs_in_parallel_total += partiton_reqs_in_parallel;
  partiton_replys_in_parallel_total += partiton_replys_in_parallel;
  partiton_reqs_in_parallel_util_total += partiton_reqs_in_parallel_util;
  gpu_tot_sim_cycle_parition_util += gpu_sim_cycle_parition_util;
  gpu_tot_occupancy += gpu_occupancy;

  gpu_sim_cycle = 0;
  partiton_reqs_in_parallel = 0;
  partiton_replys_in_parallel = 0;
  partiton_reqs_in_parallel_util = 0;
  gpu_sim_cycle_parition_util = 0;
  gpu_sim_insn = 0;
  m_total_cta_launched = 0;
  gpu_completed_cta = 0;
  gpu_occupancy = occupancy_stats();
}

PowerscalingCoefficients *gpgpu_sim::get_scaling_coeffs() {
  return m_gpgpusim_wrapper->get_scaling_coeffs();
}

void gpgpu_sim::print_stats(unsigned long long streamID) {
  gpgpu_ctx->stats->ptx_file_line_stats_write_file();
  gpu_print_stat(streamID);

  if (g_network_mode) {
    printf(
        "----------------------------Interconnect-DETAILS----------------------"
        "----------\n");
    icnt_display_stats();
    icnt_display_overall_stats();
    printf(
        "----------------------------END-of-Interconnect-DETAILS---------------"
        "----------\n");
  }
}


/* 
This function checks if any deadlock has occurred in the GPGPU-Sim simulator. It performs 
the following steps:
1. Traverse all the active threads in the simulator and check if any of the threads are 
   waiting for a shared resource.
2. If any thread is waiting, check if any other thread is holding the resource.
3. If no thread is holding the resource, then a deadlock has occurred and the simulator 
   should be halted.
4. If a thread is holding the resource, then the simulator should continue running and the 
   deadlock check should be repeated at a later time.
*/
void gpgpu_sim::deadlock_check() {
  if (m_config.gpu_deadlock_detect && gpu_deadlock) {
    fflush(stdout);
    printf(
        "\n\nGPGPU-Sim uArch: ERROR ** deadlock detected: last writeback core "
        "%u @ gpu_sim_cycle %u (+ gpu_tot_sim_cycle %u) (%u cycles ago)\n",
        gpu_sim_insn_last_update_sid, (unsigned)gpu_sim_insn_last_update,
        (unsigned)(gpu_tot_sim_cycle - gpu_sim_cycle),
        (unsigned)(gpu_sim_cycle - gpu_sim_insn_last_update));
    unsigned num_cores = 0;
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
      unsigned not_completed = m_cluster[i]->get_not_completed();
      if (not_completed) {
        if (!num_cores) {
          printf(
              "GPGPU-Sim uArch: DEADLOCK  shader cores no longer committing "
              "instructions [core(# threads)]:\n");
          printf("GPGPU-Sim uArch: DEADLOCK  ");
          m_cluster[i]->print_not_completed(stdout);
        } else if (num_cores < 8) {
          m_cluster[i]->print_not_completed(stdout);
        } else if (num_cores >= 8) {
          printf(" + others ... ");
        }
        num_cores += m_shader_config->n_simt_cores_per_cluster;
      }
    }
    printf("\n");
    for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
      bool busy = m_memory_partition_unit[i]->busy();
      if (busy)
        printf("GPGPU-Sim uArch DEADLOCK:  memory partition %u busy\n", i);
    }
    //icnt_busy()判断互连网络是否处于Busy状态。有任意一个子网络处于Busy状态便认为
    //整个互连网络处于Busy状态。
    if (icnt_busy()) {
      printf("GPGPU-Sim uArch DEADLOCK:  iterconnect contains traffic\n");
      icnt_display_state(stdout);
    }
    printf(
        "\nRe-run the simulator in gdb and use debug routines in .gdbinit to "
        "debug this\n");
    fflush(stdout);
    abort();
  }
}

/// printing the names and uids of a set of executed kernels (usually there is
/// only one)
std::string gpgpu_sim::executed_kernel_info_string() {
  std::stringstream statout;

  statout << "kernel_name = ";
  for (unsigned int k = 0; k < m_executed_kernel_names.size(); k++) {
    statout << m_executed_kernel_names[k] << " ";
  }
  statout << std::endl;
  statout << "kernel_launch_uid = ";
  for (unsigned int k = 0; k < m_executed_kernel_uids.size(); k++) {
    statout << m_executed_kernel_uids[k] << " ";
  }
  statout << std::endl;

  return statout.str();
}

std::string gpgpu_sim::executed_kernel_name() {
  std::stringstream statout;
  if (m_executed_kernel_names.size() == 1)
    statout << m_executed_kernel_names[0];
  else {
    for (unsigned int k = 0; k < m_executed_kernel_names.size(); k++) {
      statout << m_executed_kernel_names[k] << " ";
    }
  }
  return statout.str();
}
void gpgpu_sim::set_cache_config(std::string kernel_name,
                                 FuncCache cacheConfig) {
  m_special_cache_config[kernel_name] = cacheConfig;
}

FuncCache gpgpu_sim::get_cache_config(std::string kernel_name) {
  for (std::map<std::string, FuncCache>::iterator iter =
           m_special_cache_config.begin();
       iter != m_special_cache_config.end(); iter++) {
    std::string kernel = iter->first;
    if (kernel_name.compare(kernel) == 0) {
      return iter->second;
    }
  }
  return (FuncCache)0;
}

bool gpgpu_sim::has_special_cache_config(std::string kernel_name) {
  for (std::map<std::string, FuncCache>::iterator iter =
           m_special_cache_config.begin();
       iter != m_special_cache_config.end(); iter++) {
    std::string kernel = iter->first;
    if (kernel_name.compare(kernel) == 0) {
      return true;
    }
  }
  return false;
}

void gpgpu_sim::set_cache_config(std::string kernel_name) {
  if (has_special_cache_config(kernel_name)) {
    change_cache_config(get_cache_config(kernel_name));
  } else {
    change_cache_config(FuncCachePreferNone);
  }
}

/**
 * @brief Change the cache configuration of the GPGPU-Sim 4.0 version
 * 
 * @param cache_config The cache configuration to be changed
 */
void gpgpu_sim::change_cache_config(FuncCache cache_config) {
  if (cache_config != m_shader_config->m_L1D_config.get_cache_status()) {
    printf("FLUSH L1 Cache at configuration change between kernels\n");
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
      m_cluster[i]->cache_invalidate();
    }
  }

  switch (cache_config) {
    case FuncCachePreferNone:
      m_shader_config->m_L1D_config.init(
          m_shader_config->m_L1D_config.m_config_string, FuncCachePreferNone);
      m_shader_config->gpgpu_shmem_size =
          m_shader_config->gpgpu_shmem_sizeDefault;
      break;
    case FuncCachePreferL1:
      if ((m_shader_config->m_L1D_config.m_config_stringPrefL1 == NULL) ||
          (m_shader_config->gpgpu_shmem_sizePrefL1 == (unsigned)-1)) {
        printf("WARNING: missing Preferred L1 configuration\n");
        m_shader_config->m_L1D_config.init(
            m_shader_config->m_L1D_config.m_config_string, FuncCachePreferNone);
        m_shader_config->gpgpu_shmem_size =
            m_shader_config->gpgpu_shmem_sizeDefault;

      } else {
        m_shader_config->m_L1D_config.init(
            m_shader_config->m_L1D_config.m_config_stringPrefL1,
            FuncCachePreferL1);
        m_shader_config->gpgpu_shmem_size =
            m_shader_config->gpgpu_shmem_sizePrefL1;
      }
      break;
    case FuncCachePreferShared:
      if ((m_shader_config->m_L1D_config.m_config_stringPrefShared == NULL) ||
          (m_shader_config->gpgpu_shmem_sizePrefShared == (unsigned)-1)) {
        printf("WARNING: missing Preferred L1 configuration\n");
        m_shader_config->m_L1D_config.init(
            m_shader_config->m_L1D_config.m_config_string, FuncCachePreferNone);
        m_shader_config->gpgpu_shmem_size =
            m_shader_config->gpgpu_shmem_sizeDefault;
      } else {
        m_shader_config->m_L1D_config.init(
            m_shader_config->m_L1D_config.m_config_stringPrefShared,
            FuncCachePreferShared);
        m_shader_config->gpgpu_shmem_size =
            m_shader_config->gpgpu_shmem_sizePrefShared;
      }
      break;
    default:
      break;
  }
}

void gpgpu_sim::clear_executed_kernel_info() {
  m_executed_kernel_names.clear();
  m_executed_kernel_uids.clear();
}

void gpgpu_sim::gpu_print_stat(unsigned long long streamID) {
  FILE *statfout = stdout;

  std::string kernel_info_str = executed_kernel_info_string();
  fprintf(statfout, "%s", kernel_info_str.c_str());

  printf("kernel_stream_id = %llu\n", streamID);

  //在当前kernel的模拟器模拟期间，模拟器运行的周期数。在运行PyTorch时有多个层的Kernel，或单个程序有可
  //能编译出多个Kernel时，需要多次启动模拟器来执行多个Kernel，这时候在每次模拟器启动时，需要一个全局的
  //记录周期数的变量来记录所有Kernel的执行周期数，因此就用gpu_tot_sim_cycle来表示这一全局的时钟周期数
  //变量。即如果只有一个Kernel执行的话，仅启动一次模拟器，那么gpu_tot_sim_cycle初始化为0，运行过程中
  //的当前的时钟周期数由gpu_sim_cycle记录。
  printf("gpu_sim_cycle = %lld\n", gpu_sim_cycle);
  //在当前kernel的模拟器模拟期间，模拟器运行的指令数。在运行多个Kernel时，与gpu_tot_sim_cycle类似，
  //由gpu_tot_sim_insn维护全局的执行指令总数变量。
  printf("gpu_sim_insn = %lld\n", gpu_sim_insn);
  //在当前kernel的模拟器模拟期间，IPC。
  printf("gpu_ipc = %12.4f\n", (float)gpu_sim_insn / gpu_sim_cycle);
  printf("gpu_tot_sim_cycle = %lld\n", gpu_tot_sim_cycle + gpu_sim_cycle);
  printf("gpu_tot_sim_insn = %lld\n", gpu_tot_sim_insn + gpu_sim_insn);
  //运行多个kernel的模拟期间，IPC。
  printf("gpu_tot_ipc = %12.4f\n", (float)(gpu_tot_sim_insn + gpu_sim_insn) /
                                       (gpu_tot_sim_cycle + gpu_sim_cycle));
  //在当前kernel的模拟器模拟期间，m_total_cta_launched维护当前Kernel的CTA的发射数。在运行多个Kernel
  //的时候，与gpu_tot_sim_cycle类似，gpu_tot_issued_cta维护多个Kernel执行期间的全局的CTA的发射总数。
  printf("gpu_tot_issued_cta = %lld\n",
         gpu_tot_issued_cta + m_total_cta_launched);
  printf("gpu_occupancy = %.4f%% \n", gpu_occupancy.get_occ_fraction() * 100);
  printf("gpu_tot_occupancy = %.4f%% \n",
         (gpu_occupancy + gpu_tot_occupancy).get_occ_fraction() * 100);

  fprintf(statfout, "max_total_param_size = %llu\n",
          gpgpu_ctx->device_runtime->g_max_total_param_size);

  // performance counter for stalls due to congestion.
  //由于拥塞而暂停的性能计数器。
  
  //互连网络输出到DRAM Channel的暂停周期数。请求由icnt发送至L2_queue时，m_icnt_L2_queue没有SECTOR_
  //CHUNCK_SIZE大小的空间可以保存请求信息，因此互连网络的拥塞造成DRAM的停滞次数。
  printf("gpu_stall_dramfull = %d\n", gpu_stall_dramfull);
  //在从存储控制器向互连网络弹出时，如果互连网络中有空闲的缓冲区，则将内存请求推入互连网络。但是一旦
  //互连网络中的缓冲区被占满，就会停止推送。由于互连网络缓冲区的大小限制造成的停顿时钟周期数由gpu_st
  //all_icnt2sh计数保存下来。
  printf("gpu_stall_icnt2sh    = %d\n", gpu_stall_icnt2sh);

  // printf("partiton_reqs_in_parallel = %lld\n", partiton_reqs_in_parallel);
  // printf("partiton_reqs_in_parallel_total    = %lld\n",
  // partiton_reqs_in_parallel_total );
  
  //自模拟器启动的第一个时钟周期开始，partiton_reqs_in_parallel就根据每一拍的内存分区产生的请求开始
  //计数。partiton_reqs_in_parallel表示自模拟器启动的第一个时钟周期开始后所有存储分区产生的请求被推
  //入m_memory_sub_partition的m_icnt_L2_queue的总个数。这里partiton_level_parallism是指在当前的
  //内核执行期间平均每个时钟周期内存储分区产生的请求被推入m_memory_sub_partition的m_icnt_L2_queue
  //的个数。
  printf("partiton_level_parallism = %12.4f\n",
         (float)partiton_reqs_in_parallel / gpu_sim_cycle);
  //partiton_reqs_in_parallel_total是指多个Kernel执行期间所有存储分区产生的请求被推入L2_queue的总个
  //数，这个指标与gpu_tot_sim_cycle类似。
  printf("partiton_level_parallism_total  = %12.4f\n",
         (float)(partiton_reqs_in_parallel + partiton_reqs_in_parallel_total) /
             (gpu_tot_sim_cycle + gpu_sim_cycle));
  // printf("partiton_reqs_in_parallel_util = %lld\n",
  // partiton_reqs_in_parallel_util);
  // printf("partiton_reqs_in_parallel_util_total    = %lld\n",
  // partiton_reqs_in_parallel_util_total ); printf("gpu_sim_cycle_parition_util
  // = %lld\n", gpu_sim_cycle_parition_util);
  // printf("gpu_tot_sim_cycle_parition_util    = %lld\n",
  // gpu_tot_sim_cycle_parition_util );
  printf("partiton_level_parallism_util = %12.4f\n",
         (float)partiton_reqs_in_parallel_util / gpu_sim_cycle_parition_util);
  printf("partiton_level_parallism_util_total  = %12.4f\n",
         (float)(partiton_reqs_in_parallel_util +
                 partiton_reqs_in_parallel_util_total) /
             (gpu_sim_cycle_parition_util + gpu_tot_sim_cycle_parition_util));
  // printf("partiton_replys_in_parallel = %lld\n",
  // partiton_replys_in_parallel); printf("partiton_replys_in_parallel_total =
  // %lld\n", partiton_replys_in_parallel_total );
  printf("L2_BW  = %12.4f GB/Sec\n",
         ((float)(partiton_replys_in_parallel * 32) /
          (gpu_sim_cycle * m_config.icnt_period)) /
             1000000000);
  printf("L2_BW_total  = %12.4f GB/Sec\n",
         ((float)((partiton_replys_in_parallel +
                   partiton_replys_in_parallel_total) *
                  32) /
          ((gpu_tot_sim_cycle + gpu_sim_cycle) * m_config.icnt_period)) /
             1000000000);

  time_t curr_time;
  time(&curr_time);
  unsigned long long elapsed_time =
      MAX(curr_time - gpgpu_ctx->the_gpgpusim->g_simulation_starttime, 1);
  printf("gpu_total_sim_rate=%u\n",
         (unsigned)((gpu_tot_sim_insn + gpu_sim_insn) / elapsed_time));

  // shader_print_l1_miss_stat( stdout );
  shader_print_cache_stats(stdout);

  cache_stats core_cache_stats;
  core_cache_stats.clear();
  for (unsigned i = 0; i < m_config.num_cluster(); i++) {
    m_cluster[i]->get_cache_stats(core_cache_stats);
  }
  printf("\nTotal_core_cache_stats:\n");
  core_cache_stats.print_stats(stdout, streamID,
                               "Total_core_cache_stats_breakdown");
  printf("\nTotal_core_cache_fail_stats:\n");
  core_cache_stats.print_fail_stats(stdout, streamID,
                                    "Total_core_cache_fail_stats_breakdown");
  shader_print_scheduler_stat(stdout, false);

  m_shader_stats->print(stdout);
#ifdef GPGPUSIM_POWER_MODEL
  if (m_config.g_power_simulation_enabled) {
    if (m_config.g_power_simulation_mode > 0) {
      // if(!m_config.g_aggregate_power_stats)
      mcpat_reset_perf_count(m_gpgpusim_wrapper);
      calculate_hw_mcpat(m_config, getShaderCoreConfig(), m_gpgpusim_wrapper,
                         m_power_stats, m_config.gpu_stat_sample_freq,
                         gpu_tot_sim_cycle, gpu_sim_cycle, gpu_tot_sim_insn,
                         gpu_sim_insn, m_config.g_power_simulation_mode,
                         m_config.g_dvfs_enabled, m_config.g_hw_perf_file_name,
                         m_config.g_hw_perf_bench_name, executed_kernel_name(),
                         m_config.accelwattch_hybrid_configuration,
                         m_config.g_aggregate_power_stats);
    }
    m_gpgpusim_wrapper->print_power_kernel_stats(
        gpu_sim_cycle, gpu_tot_sim_cycle, gpu_tot_sim_insn + gpu_sim_insn,
        kernel_info_str, true);
    // if(!m_config.g_aggregate_power_stats)
    mcpat_reset_perf_count(m_gpgpusim_wrapper);
  }
#endif

  // performance counter that are not local to one shader
  m_memory_stats->memlatstat_print(m_memory_config->m_n_mem,
                                   m_memory_config->nbk);
  for (unsigned i = 0; i < m_memory_config->m_n_mem; i++)
    m_memory_partition_unit[i]->print(stdout);

  // L2 cache stats
  if (!m_memory_config->m_L2_config.disabled()) {
    cache_stats l2_stats;
    struct cache_sub_stats l2_css;
    struct cache_sub_stats total_l2_css;
    l2_stats.clear();
    l2_css.clear();
    total_l2_css.clear();

    printf("\n========= L2 cache stats =========\n");
    for (unsigned i = 0; i < m_memory_config->m_n_mem_sub_partition; i++) {
      m_memory_sub_partition[i]->accumulate_L2cache_stats(l2_stats);
      m_memory_sub_partition[i]->get_L2cache_sub_stats(l2_css);

      fprintf(stdout,
              "L2_cache_bank[%d]: Access = %llu, Miss = %llu, Miss_rate = "
              "%.3lf, Pending_hits = %llu, Reservation_fails = %llu\n",
              i, l2_css.accesses, l2_css.misses,
              (double)l2_css.misses / (double)l2_css.accesses,
              l2_css.pending_hits, l2_css.res_fails);

      total_l2_css += l2_css;
    }
    if (!m_memory_config->m_L2_config.disabled() &&
        m_memory_config->m_L2_config.get_num_lines()) {
      // L2c_print_cache_stat();
      printf("L2_total_cache_accesses = %llu\n", total_l2_css.accesses);
      printf("L2_total_cache_misses = %llu\n", total_l2_css.misses);
      if (total_l2_css.accesses > 0)
        printf("L2_total_cache_miss_rate = %.4lf\n",
               (double)total_l2_css.misses / (double)total_l2_css.accesses);
      printf("L2_total_cache_pending_hits = %llu\n", total_l2_css.pending_hits);
      printf("L2_total_cache_reservation_fails = %llu\n",
             total_l2_css.res_fails);
      printf("L2_total_cache_breakdown:\n");
      l2_stats.print_stats(stdout, streamID, "L2_cache_stats_breakdown");
      printf("L2_total_cache_reservation_fail_breakdown:\n");
      l2_stats.print_fail_stats(stdout, streamID,
                                "L2_cache_stats_fail_breakdown");
      total_l2_css.print_port_stats(stdout, "L2_cache");
    }
  }

  if (m_config.gpgpu_cflog_interval != 0) {
    spill_log_to_file(stdout, 1, gpu_sim_cycle);
    insn_warp_occ_print(stdout);
  }
  if (gpgpu_ctx->func_sim->gpgpu_ptx_instruction_classification) {
    StatDisp(gpgpu_ctx->func_sim->g_inst_classification_stat
                 [gpgpu_ctx->func_sim->g_ptx_kernel_count]);
    StatDisp(gpgpu_ctx->func_sim->g_inst_op_classification_stat
                 [gpgpu_ctx->func_sim->g_ptx_kernel_count]);
  }

#ifdef GPGPUSIM_POWER_MODEL
  if (m_config.g_power_simulation_enabled) {
    m_gpgpusim_wrapper->detect_print_steady_state(
        1, gpu_tot_sim_insn + gpu_sim_insn);
  }
#endif

  // Interconnect power stat print
  long total_simt_to_mem = 0;
  long total_mem_to_simt = 0;
  long temp_stm = 0;
  long temp_mts = 0;
  for (unsigned i = 0; i < m_config.num_cluster(); i++) {
    m_cluster[i]->get_icnt_stats(temp_stm, temp_mts);
    total_simt_to_mem += temp_stm;
    total_mem_to_simt += temp_mts;
  }
  printf("\nicnt_total_pkts_mem_to_simt=%ld\n", total_mem_to_simt);
  printf("icnt_total_pkts_simt_to_mem=%ld\n", total_simt_to_mem);

  time_vector_print();
  fflush(stdout);

  clear_executed_kernel_info();
}

// performance counter that are not local to one shader
unsigned gpgpu_sim::threads_per_core() const {
  return m_shader_config->n_thread_per_shader;
}

void shader_core_ctx::mem_instruction_stats(const warp_inst_t &inst) {
  unsigned active_count = inst.active_count();
  // this breaks some encapsulation: the is_[space] functions, if you change
  // those, change this.
  switch (inst.space.get_type()) {
    case undefined_space:
    case reg_space:
      break;
    case shared_space:
      m_stats->gpgpu_n_shmem_insn += active_count;
      break;
    case sstarr_space:
      m_stats->gpgpu_n_sstarr_insn += active_count;
      break;
    case const_space:
      m_stats->gpgpu_n_const_insn += active_count;
      break;
    case param_space_kernel:
    case param_space_local:
      m_stats->gpgpu_n_param_insn += active_count;
      break;
    case tex_space:
      m_stats->gpgpu_n_tex_insn += active_count;
      break;
    case global_space:
    case local_space:
      if (inst.is_store())
        m_stats->gpgpu_n_store_insn += active_count;
      else
        m_stats->gpgpu_n_load_insn += active_count;
      break;
    default:
      abort();
  }
}

/**
 * Checks if the shader core can issue 1 block of instructions for the given kernel.
 * 
 * @param kernel The kernel info object containing information about the kernel.
 * 
 * @return true if the shader core can issue 1 block of instructions, false otherwise.
 */
/*
判断是否可以发射一个线程块，如果可以发射一个线程块，则返回true，否则返回false。
*/
bool shader_core_ctx::can_issue_1block(kernel_info_t &kernel) {
  // Jin: concurrent kernels on one SM
  //支持SM上的并发内核（默认为禁用），在V100配置中禁用。
  if (m_config->gpgpu_concurrent_kernel_sm) {
    if (m_config->max_cta(kernel) < 1) return false;

    return occupy_shader_resource_1block(kernel, false);
  } else {
    //get_n_active_cta()返回当前SM上的活跃线程块的数量，m_config->max_cta(kernel)则是计算kernel的
    //支持的单个SM内的最大线程块数，如果当前SM上的活跃线程块的数量小于kernel支持的单个SM内的最大线程
    //块数，则说明此时发射一个其他kernel的线程块是可行的，返回true，否则返回false。
    return (get_n_active_cta() < m_config->max_cta(kernel));
  }
}

/*
并发内核使用，在V100配置中用不到。
*/
int shader_core_ctx::find_available_hwtid(unsigned int cta_size, bool occupy) {
  unsigned int step;
  for (step = 0; step < m_config->n_thread_per_shader; step += cta_size) {
    unsigned int hw_tid;
    for (hw_tid = step; hw_tid < step + cta_size; hw_tid++) {
      if (m_occupied_hwtid.test(hw_tid)) break;
    }
    if (hw_tid == step + cta_size)  // consecutive non-active
      break;
  }
  if (step >= m_config->n_thread_per_shader)  // didn't find
    return -1;
  else {
    if (occupy) {
      for (unsigned hw_tid = step; hw_tid < step + cta_size; hw_tid++)
        m_occupied_hwtid.set(hw_tid);
    }
    return step;
  }
}

/*
并发内核使用，在V100配置中用不到。
*/
bool shader_core_ctx::occupy_shader_resource_1block(kernel_info_t &k,
                                                    bool occupy) {
  unsigned threads_per_cta = k.threads_per_cta();
  const class function_info *kernel = k.entry();
  unsigned int padded_cta_size = threads_per_cta;
  unsigned int warp_size = m_config->warp_size;
  if (padded_cta_size % warp_size)
    padded_cta_size = ((padded_cta_size / warp_size) + 1) * (warp_size);

  if (m_occupied_n_threads + padded_cta_size > m_config->n_thread_per_shader)
    return false;

  if (find_available_hwtid(padded_cta_size, false) == -1) return false;

  const struct gpgpu_ptx_sim_info *kernel_info = ptx_sim_kernel_info(kernel);

  if (m_occupied_shmem + kernel_info->smem > m_config->gpgpu_shmem_size)
    return false;

  unsigned int used_regs = padded_cta_size * ((kernel_info->regs + 3) & ~3);
  if (m_occupied_regs + used_regs > m_config->gpgpu_shader_registers)
    return false;

  if (m_occupied_ctas + 1 > m_config->max_cta_per_core) return false;

  if (occupy) {
    m_occupied_n_threads += padded_cta_size;
    m_occupied_shmem += kernel_info->smem;
    m_occupied_regs += (padded_cta_size * ((kernel_info->regs + 3) & ~3));
    m_occupied_ctas++;

    SHADER_DPRINTF(LIVENESS,
                   "GPGPU-Sim uArch: Occupied %u threads, %u shared mem, %u "
                   "registers, %u ctas, on shader %d\n",
                   m_occupied_n_threads, m_occupied_shmem, m_occupied_regs,
                   m_occupied_ctas, m_sid);
  }

  return true;
}

void shader_core_ctx::release_shader_resource_1block(unsigned hw_ctaid,
                                                     kernel_info_t &k) {
  if (m_config->gpgpu_concurrent_kernel_sm) {
    unsigned threads_per_cta = k.threads_per_cta();
    const class function_info *kernel = k.entry();
    unsigned int padded_cta_size = threads_per_cta;
    unsigned int warp_size = m_config->warp_size;
    if (padded_cta_size % warp_size)
      padded_cta_size = ((padded_cta_size / warp_size) + 1) * (warp_size);

    assert(m_occupied_n_threads >= padded_cta_size);
    m_occupied_n_threads -= padded_cta_size;

    int start_thread = m_occupied_cta_to_hwtid[hw_ctaid];

    for (unsigned hwtid = start_thread; hwtid < start_thread + padded_cta_size;
         hwtid++)
      m_occupied_hwtid.reset(hwtid);
    m_occupied_cta_to_hwtid.erase(hw_ctaid);

    const struct gpgpu_ptx_sim_info *kernel_info = ptx_sim_kernel_info(kernel);

    assert(m_occupied_shmem >= (unsigned int)kernel_info->smem);
    m_occupied_shmem -= kernel_info->smem;

    unsigned int used_regs = padded_cta_size * ((kernel_info->regs + 3) & ~3);
    assert(m_occupied_regs >= used_regs);
    m_occupied_regs -= used_regs;

    assert(m_occupied_ctas >= 1);
    m_occupied_ctas--;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Launches a cooperative thread array (CTA).
 *
 * @param kernel
 *    object that tells us which kernel to ask for a CTA from
 */
/*
加载一个CTA。函数 ptx_sim_init_thread 初始化标量线程开始，然后使用ptx_exec_inst()在warp中执行标
量线程。即每个线程的功能状态通过调用 ptx_sim_init_thread 进行初始化。这里仅是对单个线程的指令进行
初始化，参数中包括指定某个线程的 unsigned hw_cta_id, unsigned hw_warp_id, 以及 ptx_thread_info 
**thread_info, int sid, unsigned tid。需要注意的是，这类进行的是单个CTA内的所有线程进行循环。

共享内存空间是整个CTA（线程块）所共有的，当每个CTA被分派执行时（在函数 ptx_sim_init_thread() 中），
为其分配一个唯一的 memory_space 对象。当CTA执行完毕后，该对象被取消分配。

ptx_sim_init_thread 在 functionalCoreSim::initializeCTA 函数中被调用，参数的详细说明见该调用处。

函数定义：ptx_sim_init_thread(kernel_info_t &kernel,
                                ptx_thread_info **thread_info, int sid,
                                unsigned tid, unsigned threads_left,
                                unsigned num_threads, core_t *core,
                                unsigned hw_cta_id, unsigned hw_warp_id,
                                gpgpu_t *gpu, bool isInFunctionalSimulationMode)
参数：
  sid=0：SM的index，由于这里执行功能模拟，因此SM的index不重要，可以完全将所有需要执行的线程全部放到
        第0号SM上。
  tid=i：线程的index，在这个循环里将所有需要执行的线程全部放到第0号SM上，则线程的index即为循环变量i。
  threads_left=m_kernel->threads_per_cta()-i：在当前线程之后剩余线程的数量。
  num_threads=m_kernel->threads_per_cta()。
  hw_cta_id=0：由于这里执行功能模拟，因此CTA的index不重要，硬件CTA的index可以始终为0。
  hw_warp_id=i/m_warp_size：由于全都在一个CTA内，硬件的warp的index即为i/m_warp_size。
*/
unsigned exec_shader_core_ctx::sim_init_thread(
    kernel_info_t &kernel, ptx_thread_info **thread_info, int sid, unsigned tid,
    unsigned threads_left, unsigned num_threads, core_t *core,
    unsigned hw_cta_id, unsigned hw_warp_id, gpgpu_t *gpu) {
  return ptx_sim_init_thread(kernel, thread_info, sid, tid, threads_left,
                             num_threads, core, hw_cta_id, hw_warp_id, gpu);
}

/*
SM发射kernel的一个线程块。
*/
void shader_core_ctx::issue_block2core(kernel_info_t &kernel) {
  //支持SM上的并发内核（默认为禁用），在V100配置中禁用。
  if (!m_config->gpgpu_concurrent_kernel_sm)
    //计算最大的每SM上CTA数量kernel_max_cta_per_shader，并且还要依据线程块的线程数量是否能对warp 
    //size取模运算，来计算padded每CTA的线程数量kernel_padded_threads_per_cta。
    set_max_cta(kernel);
  else
    assert(occupy_shader_resource_1block(kernel, true));

  //执行m_num_cores_running++，m_num_cores_running是一个Core计数器，它是一个全局变量，用于跟踪正
  //在运行当前内核函数的Shader Core的数量，并确定GPU是否可以接受新的任务。
  kernel.inc_running();

  // find a free CTA context
  //free_cta_hw_id指没有被占用的CTA ID，下面找一个空闲的CTA。
  unsigned free_cta_hw_id = (unsigned)-1;

  unsigned max_cta_per_core;
  //支持SM上的并发内核（默认为禁用），在V100配置中禁用。
  if (!m_config->gpgpu_concurrent_kernel_sm)
    //kernel_max_cta_per_shader是计算得出的最大的每SM上CTA数量。
    max_cta_per_core = kernel_max_cta_per_shader;
  else
    max_cta_per_core = m_config->max_cta_per_core;
  //对单个SIMT Core中的所有CTA循环，查找处于非活跃状态的CTA，将其编号赋值到free_cta_hw_id。
  for (unsigned i = 0; i < max_cta_per_core; i++) {
    //m_cta_status[i] == 0代表第i个CTA内活跃的线程数量为0，即第i个CTA已经不活跃了，已经结束了。
    if (m_cta_status[i] == 0) {
      free_cta_hw_id = i;
      break;
    }
  }
  assert(free_cta_hw_id != (unsigned)-1);

  // determine hardware threads and warps that will be used for this CTA
  int cta_size = kernel.threads_per_cta();

  // hw warp id = hw thread id mod warp size, so we need to find a range
  // of hardware thread ids corresponding to an integral number of hardware
  // thread ids
  //其实这一步是重复的，因为kernel_padded_threads_per_cta的计算过程与padded_cta_size一致。
  int padded_cta_size = cta_size;
  if (cta_size % m_config->warp_size)
    padded_cta_size =
        ((cta_size / m_config->warp_size) + 1) * (m_config->warp_size);

  //起始线程号和结束线程号。
  unsigned int start_thread, end_thread;

  if (!m_config->gpgpu_concurrent_kernel_sm) {
    //起始线程号。
    start_thread = free_cta_hw_id * padded_cta_size;
    //结束线程号。
    end_thread = start_thread + cta_size;
  } else {
    start_thread = find_available_hwtid(padded_cta_size, true);
    assert((int)start_thread != -1);
    end_thread = start_thread + cta_size;
    assert(m_occupied_cta_to_hwtid.find(free_cta_hw_id) ==
           m_occupied_cta_to_hwtid.end());
    m_occupied_cta_to_hwtid[free_cta_hw_id] = start_thread;
  }

  // reset the microarchitecture state of the selected hardware thread and warp
  // contexts
  //重置所选硬件线程和warp上下文的微架构状态。
  reinit(start_thread, end_thread, false);

  // initalize scalar threads and determine which hardware warps they are
  // allocated to bind functional simulation state of threads to hardware
  // resources (simulation)
  //初始化标量线程并确定它们分配给哪些硬件warp将功能仿真状态绑定到硬件资源（仿真）。
  warp_set_t warps;
  //nthreads_in_block是thread block中的线程数。
  unsigned nthreads_in_block = 0;
  //返回一个kernel的入口函数，m_kernel_entry是 function_info 对象。
  function_info *kernel_func_info = kernel.entry();
  //符号表。
  symbol_table *symtab = kernel_func_info->get_symtab();
  //获取下一个要发射的CTA的索引。CTA的全局索引与CUDA编程模型中的线程块索引类似，其ID算法如下：
  //  ID = m_next_cta.x + m_grid_dim.x * m_next_cta.y +
  //       m_grid_dim.x * m_grid_dim.y * m_next_cta.z;
  unsigned ctaid = kernel.get_next_cta_id_single();
  checkpoint *g_checkpoint = new checkpoint();
  //从隶属于free_cta_hw_id号CTA的起始线程号到结束线程号循环。
  for (unsigned i = start_thread; i < end_thread; i++) {
    //设置线程的CTA ID为free_cta_hw_id。
    m_threadState[i].m_cta_id = free_cta_hw_id;
    //warp_id是线程号除以32。
    unsigned warp_id = i / m_config->warp_size;
    //nthreads_in_block是thread block中的线程数。sim_init_thread函数会返回能否初始化第i个线程。
    nthreads_in_block += sim_init_thread(
        kernel, &m_thread[i], m_sid, i, cta_size - (i - start_thread),
        m_config->n_thread_per_shader, this, free_cta_hw_id, warp_id,
        m_cluster->get_gpu());
    //设置第i个线程为活跃状态。
    m_threadState[i].m_active = true;
    // load thread local memory and register file
    //在V100配置中，m_gpu->resume_option默认配置为0。
    if (m_gpu->resume_option == 1 && kernel.get_uid() == m_gpu->resume_kernel &&
        ctaid >= m_gpu->resume_CTA && ctaid < m_gpu->checkpoint_CTA_t) {
      char fname[2048];
      snprintf(fname, 2048, "checkpoint_files/thread_%d_%d_reg.txt",
               i % cta_size, ctaid);
      m_thread[i]->resume_reg_thread(fname, symtab);
      char f1name[2048];
      snprintf(f1name, 2048, "checkpoint_files/local_mem_thread_%d_%d_reg.txt",
               i % cta_size, ctaid);
      g_checkpoint->load_global_mem(m_thread[i]->m_local_mem, f1name);
    }
    //初始化标量线程并确定它们分配给哪些硬件warp将功能仿真状态绑定到硬件资源（仿真）。这里就是设置
    //哪些warp为活跃。
    warps.set(warp_id);
  }
  assert(nthreads_in_block > 0 &&
         nthreads_in_block <=
             m_config->n_thread_per_shader);  // should be at least one, but
                                              // less than max
  //m_cta_status[i] == 0代表第i个CTA内的活跃线程数量。
  m_cta_status[free_cta_hw_id] = nthreads_in_block;

  //在V100配置中，m_gpu->resume_option默认配置为0。
  if (m_gpu->resume_option == 1 && kernel.get_uid() == m_gpu->resume_kernel &&
      ctaid >= m_gpu->resume_CTA && ctaid < m_gpu->checkpoint_CTA_t) {
    char f1name[2048];
    snprintf(f1name, 2048, "checkpoint_files/shared_mem_%d.txt", ctaid);

    g_checkpoint->load_global_mem(m_thread[start_thread]->m_shared_mem, f1name);
  }
  // now that we know which warps are used in this CTA, we can allocate
  // resources for use in CTA-wide barrier operations
  //既然我们知道了在这个CTA中使用了哪些warp，我们就可以分配用于CTA总数宽的屏障资源。
  m_barriers.allocate_barrier(free_cta_hw_id, warps);

  // initialize the SIMT stacks and fetch hardware
  //初始化SIMT堆栈以及预取硬件。对第cta_id个CTA中，从start_thread到end_thread个线程所属的所有
  //warp进行初始化。
  init_warps(free_cta_hw_id, start_thread, end_thread, ctaid, cta_size, kernel);
  //活跃的CTA数量增1。
  m_n_active_cta++;

  shader_CTA_count_log(m_sid, 1);
  SHADER_DPRINTF(LIVENESS,
                 "GPGPU-Sim uArch: cta:%2u, start_tid:%4u, end_tid:%4u, "
                 "initialized @(%lld,%lld), kernel_uid:%u, kernel_name:%s\n",
                 free_cta_hw_id, start_thread, end_thread, m_gpu->gpu_sim_cycle,
                 m_gpu->gpu_tot_sim_cycle, kernel.get_uid(),
                 kernel.get_name().c_str());
}

///////////////////////////////////////////////////////////////////////////////////////////

void dram_t::dram_log(int task) {
  if (task == SAMPLELOG) {
    StatAddSample(mrqq_Dist, que_length());
  } else if (task == DUMPLOG) {
    printf("Queue Length DRAM[%d] ", id);
    StatDisp(mrqq_Dist);
  }
}

/*
Find next clock domain and increment its time. 找到下一个时钟域并推进它的时间。
*/ 
int gpgpu_sim::next_clock_domain(void) {
  //四个时钟域不同步，每次需要挑选出最小运行时间的一个时钟域进行节拍的推进。
  //例如，四个时钟的初始状态分别为：
  //    core_time = 0，icnt_time = 0，dram_time = 0，l2_time = 0
  //初始状态下smallest=0，mask代表需要更新（时钟周期向前推进）的时钟域的标记。
  //    #define CORE 0b0001
  //    #define L2 0b0010
  //    #define DRAM 0b0100
  //    #define ICNT 0b1000
  //时钟域配置：-gpgpu_clock_domains 1447.0:1447.0:1447.0:850.0
  //通过if后：
  //    core_time+=1/(1447*1000000)
  //    icnt_time+=1/(1447*1000000)
  //    l2_time+=1/(1447*1000000)
  //    dram_time+=1/(850*1000000)
  //    mask=0b1111，即所有时钟域都要更新。
  //下一次运行当前函数，即更新时钟域时：
  //    smallest=core_time=icnt_time=l2_time=1/(1447*1000000)
  //通过if后：
  //    core_time+=1/(1447*1000000)
  //    icnt_time+=1/(1447*1000000)
  //    l2_time+=1/(1447*1000000)
  //    dram_time > 1/(1447*1000000)，故当前不更新
  //    mask=0b1011，即除dram_time外所有时钟域都要更新。
  double smallest = min3(core_time, icnt_time, dram_time);
  int mask = 0x00;
  //初始状态下，smallest为0周期。
  if (l2_time <= smallest) {
    smallest = l2_time;
    mask |= L2;
    l2_time += m_config.l2_period;
  }
  if (icnt_time <= smallest) {
    mask |= ICNT;
    icnt_time += m_config.icnt_period;
  }
  if (dram_time <= smallest) {
    mask |= DRAM;
    dram_time += m_config.dram_period;
  }
  if (core_time <= smallest) {
    mask |= CORE;
    core_time += m_config.core_period;
  }
  return mask;
}

/*
GPU发射线程块。
*/
void gpgpu_sim::issue_block2core() {
  unsigned last_issued = m_last_cluster_issue;
  //基本上，所有SIMT Core集群都被遍历。遍历从最后发射的集群开始。对于每个集群，调用issue_block2core，
  //它返回该集群发射的线程块数。这将增加到成员gpgpu_sim::m_total_cta_launched。
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
    unsigned idx = (i + last_issued + 1) % m_shader_config->n_simt_clusters;
    //m_cluster[idx]发射线程块，返回发射的线程块数。
    unsigned num = m_cluster[idx]->issue_block2core();
    if (num) {
      m_last_cluster_issue = idx;
      m_total_cta_launched += num;
    }
  }
}

unsigned long long g_single_step =
    0;  // set this in gdb to single step the pipeline

/*
发出线程块需要一个分层调用，如下所示：
  gpgpu_sim::cycle()
    gpgpu_sim::issue_block2core()
      simt_core_cluster::issue_block2core()
        shader_core_ctx::issue_block2core()
          trace_shader_core_ctx::init_warps()

在每个模拟周期中，都会调用gpgpu_sim::cycle()，此函数不接受任何参数。

gpgpu_sim::cycle()方法为gpgpu-sim中的所有体系结构组件计时，包括内存分区的队列、DRAM通道和二级缓存。
1. 代码段:
       icnt_push(m_shader_config->mem2device(i), mf->get_tpc(), mf, response_size);
       m_memory_partition_unit[i]->pop();
   将内存请求从内存分区的L2->icnt队列注入到互连网络中。调用tomory_partition_unit::pop()函数执行原
   子指令。请求跟踪器还会丢弃该内存请求的条目，指示内存分区已完成对此请求的服务。
2. 对memory_partition_unit::dram_cycle()的调用将内存请求从L2->dram队列移动到dram通道，dram通道移
   动到dram->L2队列，并循环芯片外GDDR3 dram内存。
3. 对memory_partition_unit::push()的调用从互连网络中弹出数据包，并将其传递到内存分区。请求跟踪器会
   收到该请求的通知。纹理访问被直接推送到icnt->L2队列，而非纹理访问被推送到最小延迟ROP队列。请注意，
   对icnt->L2和ROP队列的推送操作都受到memory_partition_unit::full()方法中定义的icnt->L2-队列大小
   的限制。
4. 对memory_partition_unit::cache_cycle()的调用为二级缓存组计时，并将请求移入或移出二级缓存。下一
   节描述了memory_partition_unit::cache_cycle()的内部结构。
*/
void gpgpu_sim::cycle() {
  //下一个需要推进的时钟域的时钟域掩码。因为每个时钟域是异步的，不是同时更新的。四个时钟的mask标记为：
  //    #define CORE 0b0001
  //    #define L2 0b0010
  //    #define DRAM 0b0100
  //    #define ICNT 0b1000
  //返回的clock_mask如果是0b1011，则代表更新CORE、L2、ICNT三个时钟域。
  int clock_mask = next_clock_domain();
  //SIMT Core时钟域更新。
  if (clock_mask & CORE) {
    // shader core loading (pop from ICNT into core) follows CORE clock.
    //对所有的SIMT Core集群循环，m_cluster[i]是其中一个集群。Shader Core加载（从ICNT弹出到Core）
    //遵循核心时钟。
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
      //simt_core_cluster::icnt_cycle()方法将内存请求从互连网络推入simt核心集群的响应FIFO。它还从
      //FIFO弹出请求，并将它们发送到相应内核的指令缓存或LDST单元。每个SIMT Core集群都有一个响应FIFO，
      //用于保存从互连网络发出的数据包。数据包被定向到SIMT Core的指令缓存（如果它是为指令获取未命中
      //提供服务的内存响应）或其内存流水线（memory pipeline，LDST 单元）。数据包以先进先出方式拿出。
      //如果SIMT Core无法接受FIFO头部的数据包，则响应FIFO将停止。为了在LDST单元上生成内存请求，每个
      //SIMT Core都有自己的注入端口接入互连网络。但是，注入端口缓冲区由SIMT Core集群所有SIMT Core共
      //享。
      //icnt_cycle()实现的主要步骤如下：
      //首先，我们要判断以下SIMT Core集群的m_response_fifo是否为空，如果不为空，则证明这一拍内，必须
      //先将SIMT Core集群的m_response_fifo中的数据包mf推入到SIMT Core的L1指令缓存m_L1I或者LD/ST单
      //元中（实际上，代码中考虑了例如TITAN V配置的单个SIMT Core集群内有多个SM的情况，但是我们这里用
      //到的V100配置每个SIMT Core集群内只有单个SM，所以这里可以认为SIMT Core集群就是单个SM）。其次，
      //判断以下做完上述步骤的SIMT Core集群的m_response_fifo是否还有空间接收新的来自互连网络的数据包，
      //如果不为满，则证明这一拍内，必须接收来自互连网络的数据包。需注意的是上述的先后顺序，实际硬件执
      //行的时候，这两个步骤同步进行，这里我们必须先将SIMT Core集群的m_response_fifo中的数据包mf推入
      //SIMT Core，然后再尝试接收新的来自互连网络的数据包。
      m_cluster[i]->icnt_cycle();
  }
  unsigned partiton_replys_in_parallel_per_cycle = 0;
  
  //更新ICNT时钟域，向前推进一拍。
  if (clock_mask & ICNT) {
    // pop from memory controller to interconnect
    //从存储控制器向互连网络弹出。gpgpu_n_mem为配置中的内存控制器（DRAM Channel）数量，定义为：
    //  option_parser_register(opp, "-gpgpu_n_mem", OPT_UINT32, &m_n_mem,
    //                         "number of memory modules (e.g. memory controllers) in gpu",
    //                         "8");
    //在V100配置中，有32个内存控制器（DRAM Channel），同时每个内存控制器分为了两个子分区，因此，
    //m_n_sub_partition_per_memory_channel为2，定义为：
    //  option_parser_register(opp, "-gpgpu_n_sub_partition_per_mchannel", OPT_UINT32,
    //                         &m_n_sub_partition_per_memory_channel,
    //                         "number of memory subpartition in each memory module",
    //                         "1");
    //而m_n_mem_sub_partition = m_n_mem * m_n_sub_partition_per_memory_channel，代表全部内存子
    //分区的总数。
    for (unsigned i = 0; i < m_memory_config->m_n_mem_sub_partition; i++) {
      //这里需要看手册中的第五章中内存分区的详细细节图，memory_sub_partition向互连网络推出数据包的接
      //口就是L2_icnt_queue->ICNT，因此这里是将内存子分区中的m_L2_icnt_queue队列顶部的数据包弹出并
      //返回。这里是对所有内存子分区循环，将所有内存子分区的m_L2_icnt_queue队列顶部的数据包弹出。这里
      //如果数据包的类型是L2_WRBK_ACC或L1_WRBK_ACC，则返回空数据包，反之返回整个数据包。
      //在V100中，L1 cache的m_write_policy为WRITE_THROUGH，实际上L1_WRBK_ACC也不会用到。
      //在V100中，当L2 cache写不命中时，采取lazy_fetch_on_read策略，当找到一个cache block
      //逐出时，如果这个cache block是被MODIFIED，则需要将这个cache block写回到下一级存储，
      //因此会产生L2_WRBK_ACC访问，这个访问就是为了写回被逐出的MODIFIED cache block。

      //这里实际上 m_memory_sub_partition[i]->top() 即执行 m_L2_icnt_queue->top()，但是当：
      //   mf->get_access_type() == L2_WRBK_ACC 或
      //   mf->get_access_type() == L1_WRBK_ACC
      //时，不会将其再转发到SM了，因为这是由于只是写回。
      mem_fetch *mf = m_memory_sub_partition[i]->top();
      if (mf) {
        // The packet size varies depending on the type of request:
        // - For read request and atomic request, the packet contains the data
        // - For write-ack, the packet only has control metadata
        //数据包大小因请求类型而异：
        // - 对于读取请求和原子请求，数据包包含数据；
        // - 对于写确认，数据包只有控制元数据。
        unsigned response_size =
            mf->get_is_write() ? mf->get_ctrl_size() : mf->size();
        //在从内存子分区向互连网络弹出时，如果互连网络中有空闲的缓冲区，则将内存请求推入互连网络。但是
        //一旦互连网络中的缓冲区被占满，就会停止推送。由于互连网络缓冲区的大小限制造成的停顿时钟周期数
        //由gpu_stall_icnt2sh计数保存下来。
        //icnt_has_buffer是判断互连网络是否有空闲的输入缓冲可以容纳来自deviceID号设备新的数据包。
        if (::icnt_has_buffer(m_shader_config->mem2device(i), response_size)) {
          // if (!mf->get_is_write())
          mf->set_return_timestamp(gpu_sim_cycle + gpu_tot_sim_cycle);
          mf->set_status(IN_ICNT_TO_SHADER, gpu_sim_cycle + gpu_tot_sim_cycle);
          //icnt_push是数据包压入互连网络输入缓冲区。
          ::icnt_push(m_shader_config->mem2device(i), mf->get_tpc(), mf,
                      response_size);
          //m_memory_sub_partition[i]中都有各自的m_icnt_L2_queue队列，这是ICNT给SM数据包的接口。
          m_memory_sub_partition[i]->pop();
          partiton_replys_in_parallel_per_cycle++;
        } else {
          gpu_stall_icnt2sh++;
        }
      } else {
        //如果内存子分区的m_L2_icnt_queue队列顶部的数据包无效，则也将这个失效的数据包也弹出。
        m_memory_sub_partition[i]->pop();
      }
    }
  }
  partiton_replys_in_parallel += partiton_replys_in_parallel_per_cycle;

  if (clock_mask & DRAM) {
    //对每个DRAM通道循环，调用memory_partition_unit::dram_cycle()方法，将内存请求从L2->dram队列移
    //动到DRAM Channel，DRAM Channel到dram->L2队列，并循环片外GDDR3 DRAM内存。
    for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
      //在V100中，simple_dram_model被配置为0。
      if (m_memory_config->simple_dram_model)
        m_memory_partition_unit[i]->simple_dram_model_cycle();
      else
        m_memory_partition_unit[i]
            ->dram_cycle();  // Issue the dram command (scheduler + delay model)
      // Update performance counters for DRAM
      m_memory_partition_unit[i]->set_dram_power_stats(
          m_power_stats->pwr_mem_stat->n_cmd[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_activity[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_nop[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_act[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_pre[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_rd[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_wr[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_wr_WB[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_req[CURRENT_STAT_IDX][i]);
    }
  }

  // L2 operations follow L2 clock domain
  unsigned partiton_reqs_in_parallel_per_cycle = 0;

  //更新L2时钟域。
  if (clock_mask & L2) {
    m_power_stats->pwr_mem_stat->l2_cache_stats[CURRENT_STAT_IDX].clear();
    //gpgpu_n_mem为配置中的内存控制器（DRAM Channel）数量，定义为：
    //  option_parser_register(opp, "-gpgpu_n_mem", OPT_UINT32, &m_n_mem,
    //                         "number of memory modules (e.g. memory controllers) in gpu",
    //                         "8");
    //在V100配置中，有32个内存控制器（DRAM Channel），同时每个内存控制器分为了两个子分区，因此，
    //m_n_sub_partition_per_memory_channel为2，定义为：
    //  option_parser_register(opp, "-gpgpu_n_sub_partition_per_mchannel", OPT_UINT32,
    //                         &m_n_sub_partition_per_memory_channel,
    //                         "number of memory subpartition in each memory module",
    //                         "1");
    //而m_n_mem_sub_partition = m_n_mem * m_n_sub_partition_per_memory_channel，代表全部内存子
    //分区的总数。这里对所有内存分区循环。
    for (unsigned i = 0; i < m_memory_config->m_n_mem_sub_partition; i++) {
      // move memory request from interconnect into memory partition (if not
      // backed up) Note:This needs to be called in DRAM clock domain if there
      // is no L2 cache in the system In the worst case, we may need to push
      // SECTOR_CHUNCK_SIZE requests, so ensure you have enough buffer for them
      //将内存请求从互连移动到内存分区（如果没有备份）注意：如果系统中没有二级缓存，则
      //需要在DRAM时钟域中调用。在最坏的情况下，我们可能需要推送SECTOR_CHUNCK_SIZE大小
      //的请求，因此需要确保有足够的缓冲区来处理它们。
      //m_memory_sub_partition[i]->full的定义为：
      //    bool memory_sub_partition::full(unsigned size) const {
      //      return m_icnt_L2_queue->is_avilable_size(size);
      //    }
      //即请求由icnt发送至L2_queue时，m_icnt_L2_queue没有SECTOR_CHUNCK_SIZE大小的空间
      //可以保存请求信息，因此互连网络的拥塞造成DRAM的停滞。
      
      //这里需要看手册中的第五章中内存分区的详细细节图，互连网络向memory_sub_partition推出数据包的接
      //口就是ICNT->icnt_L2_queue，因此这里是判断内存子分区中的m_icnt_L2_queue队列是否可以放下size
      //大小的数据，可以放下返回False，放不下返回True。SECTOR_CHUNCK_SIZE=4。如果m_icnt_L2_queue队
      //列放不下SECTOR_CHUNCK_SIZE=4大小的数据，则代表由于[icnt_L2_queue的拥塞]造成DRAM的停滞，由
      //gpu_stall_dramfull记录停滞的次数。
      if (m_memory_sub_partition[i]->full(SECTOR_CHUNCK_SIZE)) {
        gpu_stall_dramfull++;
      } else {
        //如果m_memory_sub_partition[i]->full(SECTOR_CHUNCK_SIZE)返回False，代表m_L2_icnt_queue
        //队列放得下SECTOR_CHUNCK_SIZE=4大小的数据，因此可以将数据读请求mf从互连网络推入到内存子分区
        //来进行取数据处理。
        //icnt_pop()是数据包弹出互连网络输出缓冲区。
        mem_fetch *mf = (mem_fetch *)icnt_pop(m_shader_config->mem2device(i));
        //将数据读请求m_req从互连网络推入到内存子分区来进行后续取数据处理。
        m_memory_sub_partition[i]->push(mf, gpu_sim_cycle + gpu_tot_sim_cycle);
        //如果mf不为空，则说明有请求被推入L2_queue，因此当前存储分区有请求产生，而且有
        //partiton_reqs_in_parallel_per_cycle表示当前时钟周期内所有存储分区的并行请求
        //产生的总个数，因此partiton_reqs_in_parallel_per_cycle++表示当前时钟周期内请
        //求被推入L2_queue的总个数加1。
        if (mf) partiton_reqs_in_parallel_per_cycle++;
      }
      //对二级缓存Bank进行节拍推进，并将请求移入或移出二级缓存。????
      m_memory_sub_partition[i]->cache_cycle(gpu_sim_cycle + gpu_tot_sim_cycle);
      if (m_config.g_power_simulation_enabled) {
        m_memory_sub_partition[i]->accumulate_L2cache_stats(
            m_power_stats->pwr_mem_stat->l2_cache_stats[CURRENT_STAT_IDX]);
      }
    }
  }
  //自模拟器启动的第一个时钟周期开始，partiton_reqs_in_parallel就根据每一拍的内存分区产
  //生的请求开始计数。partiton_reqs_in_parallel表示自模拟器启动的第一个时钟周期开始后所
  //有存储分区产生的请求被推入m_memory_sub_partition的m_icnt_L2_queue的总个数。
  partiton_reqs_in_parallel += partiton_reqs_in_parallel_per_cycle;
  if (partiton_reqs_in_parallel_per_cycle > 0) {
    partiton_reqs_in_parallel_util += partiton_reqs_in_parallel_per_cycle;
    gpu_sim_cycle_parition_util++;
  }

  if (clock_mask & ICNT) {
    //互连网络执行路由一拍。
    icnt_transfer();
  }
  //如果推进的是SIMT Core时钟域。
  if (clock_mask & CORE) {
    // L1 cache + shader core pipeline stages
    m_power_stats->pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].clear();
    //对GPU中所有的SIMT Core集群进行循环，更新每个集群的状态。
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
      //如果get_not_completed()大于1，代表这个SIMT Core尚未完成；如果get_more_cta_left()为1，
      //代表这个SIMT Core还有剩余的CTA需要取执行。m_cluster[i]->get_not_completed()返回第i个
      //SIMT Core集群中尚未完成的线程个数。get_more_cta_left()用于检查当前是否还有更多的CTA（
      //Compute Thread Array）需要执行。它检查当前活跃的CTA数量，并返回是否有更多CTA需要执行。
      //如果已达到GPU模拟限制最大的CTA数（由hit_max_cta_count()判断），则没有剩余的CTA，返回
      //False；如果某个m_running_kernels向量里的kernel非空，且kernel->no_more_ctas_to_run()
      //为false即kernel自己还可有多余CTA执行，则返回True。no_more_ctas_to_run()函数指示当前没
      //有更多的CTA（Compute Thread Array）需要执行。
      //这里可以将m_cluster[i]的执行状态分为几类：
      //    1. shader_core_ctx::init_warps中初始化warp时，会设置m_not_completed+=n_active，
      //       因此这里get_not_completed()返回m_not_completed的值实际上是返回的是已经初始化的
      //       所有warp（即整个CTA）中尚未完成的线程数。对于尚未初始化的warp（CTA），是没有记录
      //       的。
      //    2. 因此第二部需要判断是否有尚未初始化的warp（CTA）需要后续执行，只有1和2两个条件同时
      //       满足，才可以断定当前SIMT Core集群上还需要向前推进一拍。
      if (m_cluster[i]->get_not_completed() || get_more_cta_left()) {
        //当调用simt_core_cluster::core_cycle()时，它会调用其中所有SM内核的循环，并实现循环调
        //度SIMT Core的模拟顺序，由于在本时钟周期内，是从m_core_sim_order.begin()开始调度，因
        //此为了实现轮询调度，将begin()位置移动到最末尾。这样下次就是从begin+1位置的SIMT Core开
        //始调度。
        m_cluster[i]->core_cycle();
        //增加活跃的SM数量。get_n_active_sms()返回SIMT Core集群中的活跃SM的数量。active_sms是
        //SIMT Core集群中的活跃SM的数量。get_n_active_sms()会对每个集群内部的SIMT Core进行判断
        //其是否是active()，在V100配置中，每个集群内部仅有1个SIMT Core。
        *active_sms += m_cluster[i]->get_n_active_sms();
      }
      // Update core icnt/cache stats for AccelWattch
      if (m_config.g_power_simulation_enabled) {
        m_cluster[i]->get_icnt_stats(
            m_power_stats->pwr_mem_stat->n_simt_to_mem[CURRENT_STAT_IDX][i],
            m_power_stats->pwr_mem_stat->n_mem_to_simt[CURRENT_STAT_IDX][i]);
        m_cluster[i]->get_cache_stats(
            m_power_stats->pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX]);
        m_cluster[i]->get_current_occupancy(
            gpu_occupancy.aggregate_warp_slot_filled,
            gpu_occupancy.aggregate_theoretical_warp_slots);
      }
    }
    float temp = 0;
    for (unsigned i = 0; i < m_shader_config->num_shader(); i++) {
      temp += m_shader_stats->m_pipeline_duty_cycle[i];
    }
    temp = temp / m_shader_config->num_shader();
    *average_pipeline_duty_cycle = ((*average_pipeline_duty_cycle) + temp);
    // cout<<"Average pipeline duty cycle:
    // "<<*average_pipeline_duty_cycle<<endl;

    //debug。
    if (g_single_step &&
        ((gpu_sim_cycle + gpu_tot_sim_cycle) >= g_single_step)) {
      raise(SIGTRAP);  // Debug breakpoint
    }
    //需要注意，gpu_sim_cycle仅在CORE时钟域向前推进一拍时才更新，因此gpu_sim_cycle表示CORE时钟
    //域的当前执行拍数。
    gpu_sim_cycle++;

    if (g_interactive_debugger_enabled) gpgpu_debug();

      // McPAT main cycle (interface with McPAT)
#ifdef GPGPUSIM_POWER_MODEL
    if (m_config.g_power_simulation_enabled) {
      if (m_config.g_power_simulation_mode == 0) {
        mcpat_cycle(m_config, getShaderCoreConfig(), m_gpgpusim_wrapper,
                    m_power_stats, m_config.gpu_stat_sample_freq,
                    gpu_tot_sim_cycle, gpu_sim_cycle, gpu_tot_sim_insn,
                    gpu_sim_insn, m_config.g_dvfs_enabled);
      }
    }
#endif

    //GPU发射线程块。
    issue_block2core();
    //该函数用于减少内核延迟kernel latency，减少从发出内核命令到内核完成执行的时间（对所有正在运
    //行的内核的延迟减1）。该函数可以用于模拟GPU内核的性能和分析程序的性能。m_kernel_TB_latency
    //表示每一个线程块的启动延迟之和加上kernel的启动延迟，即从发出线程块到它完成执行的时间。
    //m_kernel_TB_latency是GPGPU-Sim中用于表示内核启动时间的变量，它表示从内核启动到内核完成执
    //行所需要的时间。
    //decrement_kernel_latency函数的定义为：
    //   void gpgpu_sim::decrement_kernel_latency() {
    //     //对所有的正在运行的内核函数的内核延迟kernel latency减1。
    //     for (unsigned n = 0; n < m_running_kernels.size(); n++) {
    //       if (m_running_kernels[n] && m_running_kernels[n]->m_kernel_TB_latency)
    //         m_running_kernels[n]->m_kernel_TB_latency--;
    //     }
    //   }
    //这里是检测当前所有已经发射到SM上的kernel的线程块的启动延迟之和加上kernel的启动延迟是否已经
    //归零，当归零后SM才可以取指发射。
    decrement_kernel_latency();

    // Depending on configuration, invalidate the caches once all of threads are
    // completed.
    //标志所有线程都已经结束。
    int all_threads_complete = 1;
    //在V100配置中，m_config.gpgpu_flush_l1_cache被配置为1。
    if (m_config.gpgpu_flush_l1_cache) {
      //对所有的SIMT集群循环。
      for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
        //如果m_cluster[i]->get_not_completed()为0，代表这个SIMT Core集群中的所有线程都已经完
        //成，因此可以将这个SIMT Core集群内所有SM的L1指令缓存和数据缓存进行失效操作。
        if (m_cluster[i]->get_not_completed() == 0)
          m_cluster[i]->cache_invalidate();
        else
          all_threads_complete = 0;
      }
    }

    //在V100配置中，m_config.gpgpu_flush_l2_cache被配置为0。
    if (m_config.gpgpu_flush_l2_cache) {
      if (!m_config.gpgpu_flush_l1_cache) {
        for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
          if (m_cluster[i]->get_not_completed() != 0) {
            all_threads_complete = 0;
            break;
          }
        }
      }

      if (all_threads_complete && !m_memory_config->m_L2_config.disabled()) {
        printf("Flushed L2 caches...\n");
        if (m_memory_config->m_L2_config.get_num_lines()) {
          int dlc = 0;
          for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
            dlc = m_memory_sub_partition[i]->flushL2();
            assert(dlc == 0);  // TODO: need to model actual writes to DRAM here
            printf("Dirty lines flushed from L2 %d is %d\n", i, dlc);
          }
        }
      }
    }

    //以下是一些采样的统计数据。
    if (!(gpu_sim_cycle % m_config.gpu_stat_sample_freq)) {
      time_t days, hrs, minutes, sec;
      time_t curr_time;
      time(&curr_time);
      unsigned long long elapsed_time =
          MAX(curr_time - gpgpu_ctx->the_gpgpusim->g_simulation_starttime, 1);
      if ((elapsed_time - last_liveness_message_time) >=
              m_config.liveness_message_freq &&
          DTRACE(LIVENESS)) {
        days = elapsed_time / (3600 * 24);
        hrs = elapsed_time / 3600 - 24 * days;
        minutes = elapsed_time / 60 - 60 * (hrs + 24 * days);
        sec = elapsed_time - 60 * (minutes + 60 * (hrs + 24 * days));

        unsigned long long active = 0, total = 0;
        for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
          m_cluster[i]->get_current_occupancy(active, total);
        }
        DPRINTFG(LIVENESS,
                 "uArch: inst.: %lld (ipc=%4.1f, occ=%0.4f%% [%llu / %llu]) "
                 "sim_rate=%u (inst/sec) elapsed = %u:%u:%02u:%02u / %s",
                 gpu_tot_sim_insn + gpu_sim_insn,
                 (double)gpu_sim_insn / (double)gpu_sim_cycle,
                 float(active) / float(total) * 100, active, total,
                 (unsigned)((gpu_tot_sim_insn + gpu_sim_insn) / elapsed_time),
                 (unsigned)days, (unsigned)hrs, (unsigned)minutes,
                 (unsigned)sec, ctime(&curr_time));
        fflush(stdout);
        last_liveness_message_time = elapsed_time;
      }
      visualizer_printstat();
      m_memory_stats->memlatstat_lat_pw();
      if (m_config.gpgpu_runtime_stat &&
          (m_config.gpu_runtime_stat_flag != 0)) {
        if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_BW_STAT) {
          for (unsigned i = 0; i < m_memory_config->m_n_mem; i++)
            m_memory_partition_unit[i]->print_stat(stdout);
          printf("maxmrqlatency = %d \n", m_memory_stats->max_mrq_latency);
          printf("maxmflatency = %d \n", m_memory_stats->max_mf_latency);
        }
        if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_SHD_INFO)
          shader_print_runtime_stat(stdout);
        if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_L1MISS)
          shader_print_l1_miss_stat(stdout);
        if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_SCHED)
          shader_print_scheduler_stat(stdout, false);
      }
    }

    if (!(gpu_sim_cycle % 50000)) {
      // deadlock detection
      if (m_config.gpu_deadlock_detect && gpu_sim_insn == last_gpu_sim_insn) {
        gpu_deadlock = true;
      } else {
        last_gpu_sim_insn = gpu_sim_insn;
      }
    }
    try_snap_shot(gpu_sim_cycle);
    spill_log_to_file(stdout, 0, gpu_sim_cycle);

#if (CUDART_VERSION >= 5000)
    // launch device kernel
    gpgpu_ctx->device_runtime->launch_one_device_kernel();
#endif
  }
}

void shader_core_ctx::dump_warp_state(FILE *fout) const {
  fprintf(fout, "\n");
  fprintf(fout, "per warp functional simulation status:\n");
  for (unsigned w = 0; w < m_config->max_warps_per_shader; w++)
    m_warp[w]->print(fout);
}

/*
cuda-sim.cc中已经实现了功能性的 memcpy_to_gpu() 函数，这里实现的是性能模型中的 perf_memcpy_to_gpu()
函数，即功能相同，把数据拷贝到GPU的显存。
*/
void gpgpu_sim::perf_memcpy_to_gpu(size_t dst_start_addr, size_t count) {
  if (m_memory_config->m_perf_sim_memcpy) {
    // if(!m_config.trace_driven_mode)    //in trace-driven mode, CUDA runtime
    // can start nre data structure at any position 	assert (dst_start_addr %
    // 32
    //== 0);

    for (unsigned counter = 0; counter < count; counter += 32) {
      const unsigned wr_addr = dst_start_addr + counter;
      addrdec_t raw_addr;
      mem_access_sector_mask_t mask;
      mask.set(wr_addr % 128 / 32);
      m_memory_config->m_address_mapping.addrdec_tlx(wr_addr, &raw_addr);
      const unsigned partition_id =
          raw_addr.sub_partition /
          m_memory_config->m_n_sub_partition_per_memory_channel;
      m_memory_partition_unit[partition_id]->handle_memcpy_to_gpu(
          wr_addr, raw_addr.sub_partition, mask);
    }
  }
}

void gpgpu_sim::dump_pipeline(int mask, int s, int m) const {
  /*
     You may want to use this function while running GPGPU-Sim in gdb.
     One way to do that is add the following to your .gdbinit file:

        define dp
           call g_the_gpu.dump_pipeline_impl((0x40|0x4|0x1),$arg0,0)
        end

     Then, typing "dp 3" will show the contents of the pipeline for shader
     core 3.
  */

  printf("Dumping pipeline state...\n");
  if (!mask) mask = 0xFFFFFFFF;
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
    if (s != -1) {
      i = s;
    }
    if (mask & 1)
      m_cluster[m_shader_config->sid_to_cluster(i)]->display_pipeline(
          i, stdout, 1, mask & 0x2E);
    if (s != -1) {
      break;
    }
  }
  if (mask & 0x10000) {
    for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
      if (m != -1) {
        i = m;
      }
      printf("DRAM / memory controller %u:\n", i);
      if (mask & 0x100000) m_memory_partition_unit[i]->print_stat(stdout);
      if (mask & 0x1000000) m_memory_partition_unit[i]->visualize();
      if (mask & 0x10000000) m_memory_partition_unit[i]->print(stdout);
      if (m != -1) {
        break;
      }
    }
  }
  fflush(stdout);
}

const shader_core_config *gpgpu_sim::getShaderCoreConfig() {
  return m_shader_config;
}

const memory_config *gpgpu_sim::getMemoryConfig() { return m_memory_config; }

simt_core_cluster *gpgpu_sim::getSIMTCluster() { return *m_cluster; }
