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
gpu-sim.cc ��GPGPU-Sim�в�ͬ��ʱ��ģ��ճ��һ����������֧�ֶ��ʱ�����ʵ�֣���ʵ�����߳̿��������
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
ʱ������� 
*/

#define CORE 0x01
#define L2 0x02
#define DRAM 0x04
#define ICNT 0x08

#define MEM_LATENCY_STAT_IMPL

#include "mem_latency_stat.h"

/*
�ú�������ע��GPGPU-Sim�����ڿ����ܺ�ģ�͵�������ѡ�����������ӵ�OptionParserʵ���С��ú�����
��һ��OptionParserָ����Ϊ���������ڽ��ܺ�ģ�Ͳ�����ӵ�OptionParserʵ���С�
*/
void power_config::reg_options(class OptionParser *opp) {
  //����gpuwattch��������ģ�͵�xml_file�洢·����Ĭ��Ϊ"gpuwattch.xml"��
  option_parser_register(opp, "-accelwattch_xml_file", OPT_CSTR,
                         &g_power_config_name, "AccelWattch XML file",
                         "accelwattch_sass_sim.xml");
  //���ÿ���gpuwattch�����������ء�
  option_parser_register(opp, "-power_simulation_enabled", OPT_BOOL,
                         &g_power_simulation_enabled,
                         "Turn on power simulator (1=On, 0=Off)", "0");
  //����Dump������������Ľ��ļ����
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
  //���ÿ������ɹ��ĸ����ļ���
  option_parser_register(
      opp, "-power_trace_enabled", OPT_BOOL, &g_power_trace_enabled,
      "produce a file for the power trace (1=On, 0=Off)", "0");
  //���ĸ��������־��ѹ������
  option_parser_register(
      opp, "-power_trace_zlevel", OPT_INT32, &g_power_trace_zlevel,
      "Compression level of the power trace output log (0=no comp, 9=highest)",
      "6");
  //�����ȶ����ĵ�ƽ���ļ���
  option_parser_register(
      opp, "-steady_power_levels_enabled", OPT_BOOL,
      &g_steady_power_levels_enabled,
      "produce a file for the steady power levels (1=On, 0=Off)", "0");
  //����ƫ��:����������
  option_parser_register(opp, "-steady_state_definition", OPT_CSTR,
                         &gpu_steady_state_definition,
                         "allowed deviation:number of samples", "8:4");
}

/*
�ú�������ע��GPGPU-Sim�����ڿ��ƴ洢ģ�͵�������ѡ�����������ӵ�OptionParserʵ���С��ú�����
��һ��OptionParserָ����Ϊ���������ڽ��洢ģ�Ͳ�����ӵ�OptionParserʵ���С�
*/
void memory_config::reg_options(class OptionParser *opp) {
  //cuda-sim.cc���Ѿ�ʵ���˹����Ե� memcpy_to_gpu() ����������� m_perf_sim_memcpy ��־�Ƿ�ִ��
  //����ģ���е� perf_memcpy_to_gpu()��������������ͬ�������ݿ�����GPU���Դ档
  option_parser_register(opp, "-gpgpu_perf_sim_memcpy", OPT_BOOL,
                         &m_perf_sim_memcpy, "Fill the L2 cache on memcpy",
                         "1");
  //Ĭ��Ϊ0���رգ������õ��ٲ��䡣
  option_parser_register(opp, "-gpgpu_simple_dram_model", OPT_BOOL,
                         &simple_dram_model,
                         "simple_dram_model with fixed latency and BW", "0");
  //GPGPU-Sim��DRAM���Ⱥ�ʱ����н�ģ��GPGPU-Simʵ������������ҳ��ģʽDRAM��������һ��FIFO���Ƚ�
  //�ȳ�����������һ��FR-FCFS��First-Row First-Come-First-Served��First-Row �ȵ��ȷ��񣩵�������
  //������������������������������ʹ������ѡ��-gpgpu_dram_schedulerѡ����Щѡ�
  option_parser_register(opp, "-gpgpu_dram_scheduler", OPT_INT32,
                         &scheduler_type, "0 = fifo, 1 = FR-FCFS (defaul)",
                         "1");
  //�ڴ������"icnt-to-L2"��"L2-to-dram"��"dram-to-L2"��"L2-to-icnt"�ĸ�queue����󳤶ȡ�
  //�ڴ��������ݰ�ͨ��ICNT->L2 queue�ӻ�����������ڴ������L2 Cache Bank��ÿ��L2ʱ�����ڴ�ICNT-> 
  //L2 queue����һ��������з���L2���ɵ�оƬ��DRAM���κ��ڴ����󶼱�����L2->DRAM queue�����L2 
  //Cache�����ã����ݰ�����ICNT->L2 queue��������ֱ������L2->DRAM queue����Ȼ��L2ʱ��Ƶ�ʡ���Ƭ��
  //DRAM���ص���������DRAM->L2 queue����������L2 Cache Bank���ġ���L2��SIMT Core�Ķ���Ӧͨ��L2
  //->ICNT queue���͡�
  option_parser_register(opp, "-gpgpu_dram_partition_queues", OPT_CSTR,
                         &gpgpu_L2_queue_config, "i2$:$2d:d2$:$2i", "8:8:8:8");
  //ò��û�е��ù��������õ��ٲ��䡣�ǳ������l2_cache�����Ƿô����С�
  option_parser_register(opp, "-l2_ideal", OPT_BOOL, &l2_ideal,
                         "Use a ideal L2 cache that always hit", "0");
  //ͳһ�ķ�Bank�� L2 ���ݻ�������á�???
  option_parser_register(opp, "-gpgpu_cache:dl2", OPT_CSTR,
                         &m_L2_config.m_config_string,
                         "unified banked L2 data cache config "
                         " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                         "alloc>,<mshr>:<N>:<merge>,<mq>}",
                         "64:128:8,L:B:m:N,A:16:4,4");
  //�Ƿ� L2 ���ݻ�������� texture��
  option_parser_register(opp, "-gpgpu_cache:dl2_texture_only", OPT_BOOL,
                         &m_L2_texure_only, "L2 cache used for texture only",
                         "1");
  //gpgpu_n_memΪ�����е��ڴ��������DRAM Channel��������
  option_parser_register(
      opp, "-gpgpu_n_mem", OPT_UINT32, &m_n_mem,
      "number of memory modules (e.g. memory controllers) in gpu", "8");
  //ÿ���ڴ�ģ���е����ڴ��ӷ����ĸ�����
  option_parser_register(opp, "-gpgpu_n_sub_partition_per_mchannel", OPT_UINT32,
                         &m_n_sub_partition_per_memory_channel,
                         "number of memory subpartition in each memory module",
                         "1");
  //ÿ���ڴ��������DRAMоƬ��Ҳ��ΪDRAM channel��������ѡ��-gpgpu_n_mem_per_ctrlr���á�
  option_parser_register(opp, "-gpgpu_n_mem_per_ctrlr", OPT_UINT32,
                         &gpu_n_mem_per_ctrlr,
                         "number of memory chips per memory controller", "1");
  //�ռ��ڴ��ӳ�ͳ����Ϣ��0x2����MC��0x4���ö�����־����
  option_parser_register(opp, "-gpgpu_memlatency_stat", OPT_INT32,
                         &gpgpu_memlatency_stat,
                         "track and display latency statistics 0x2 enables MC, "
                         "0x4 enables queue logs",
                         "0");
  //DRAM FRFCFS���ȳ�����д�С��0 = unlimited (default); # entries per chip����FIFO���ȳ����
  //�д�С�̶�Ϊ2����
  option_parser_register(opp, "-gpgpu_frfcfs_dram_sched_queue_size", OPT_INT32,
                         &gpgpu_frfcfs_dram_sched_queue_size,
                         "0 = unlimited (default); # entries per chip", "0");
  //DRAM���󷵻ض��д�С��0 = unlimited (default); # entries per chip����
  option_parser_register(opp, "-gpgpu_dram_return_queue_size", OPT_INT32,
                         &gpgpu_dram_return_queue_size,
                         "0 = unlimited (default); # entries per chip", "0");
  //����DRAMоƬ����������Ƶ���µ����ߴ���Ĭ��ֵΪ4�ֽڣ�ÿ������ʱ������8�ֽڣ�����ÿ���ڴ������
  //��DRAMоƬ������ѡ�� -gpgpu_n_mem_per_ctrlr ���á�ÿ���洢���������У�gpgpu_dram_buswidth x 
  //gpgpu_n_mem_per_ctrlr��λ��DRAM�����������š����磬Quadro FX5800��һ��512λDRAM�������ߣ���Ϊ
  //8���ڴ������ÿ���洢������һ�� 512/8 = 64 λ��DRAM�������ߡ���64λ���߱��ָ�Ϊÿ���洢��������
  //2��DRAMоƬ��ÿ��оƬ������32λ=4�ֽڵ�DRAM���߿�ȡ���ˣ����ǽ� -gpgpu_dram_buswidth ����Ϊ4��
  option_parser_register(opp, "-gpgpu_dram_buswidth", OPT_UINT32, &busW,
                         "default = 4 bytes (8 bytes per cycle at DDR)", "4");
  //ÿ��DRAM�����Burst���ȣ�Ĭ��ֵ=4������ʱ�����ڣ���GDDR3����2������ʱ��Ƶ�����У�����-gpgpu_dram
  //_burst_length <# burst per DRAM request>���á�
  option_parser_register(
      opp, "-gpgpu_dram_burst_length", OPT_UINT32, &BL,
      "Burst length of each DRAM request (default = 4 data bus cycle)", "4");
  option_parser_register(opp, "-dram_data_command_freq_ratio", OPT_UINT32,
                         &data_command_freq_ratio,
                         "Frequency ratio between DRAM data bus and command "
                         "bus (default = 2 times, i.e. DDR)",
                         "2");
  //DRAMʱ�����: ???
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
ע��ÿ��Shader Core��SM���Ĳ������á�
*/
void shader_core_config::reg_options(class OptionParser *opp) {
  //SIMT��ջ�����֧��ģʽ��1������ú�ؾ����ģʽ�������ݲ�֧�֡�
  //��ͳ��SIMT Stack��PDOM���ƣ����߳����ֻ��������һ�֡�unified�����ƣ������зֻ����߳�����ͳһ�ء���
  //������תָ��ġ�immediate post-dominator��������IPDOM�������л�ۣ�reconverge�������ݡ�y is post-
  //dominator of x���Ķ��壺����·������x����ؾ���y�㣬�Դ˿���ȷ����x��ֻ���ȥ�������̱߳ؾ���y�㣻
  //���ҡ�immediate post-dominator���Ķ����ֱ�֤��y����������Ի�۵����з�֧�̵߳ĵ㣬Խ��Ļ������
  //ζ��SIMD��ˮ�߿���Խ��ر�����ֵ����á�
  option_parser_register(opp, "-gpgpu_simd_model", OPT_INT32, &model,
                         "1 = post-dominator", "1");
  //Shader Core Pipeline���á�
  //�����ֱ��ǣ�<ÿ��SM����֧���߳���>:<����һ��warp�ж����߳�>
  option_parser_register(
      opp, "-gpgpu_shader_core_pipeline", OPT_CSTR,
      &gpgpu_shader_core_pipeline_opt,
      "shader core pipeline config, i.e., {<nthread>:<warpsize>}", "1024:32");
  //L1 texture cache�����ã������õ��ٲ��䡣
  option_parser_register(opp, "-gpgpu_tex_cache:l1", OPT_CSTR,
                         &m_L1T_config.m_config_string,
                         "per-shader L1 texture cache  (READ-ONLY) config "
                         " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                         "alloc>,<mshr>:<N>:<merge>,<mq>:<rf>}",
                         "8:128:5,L:R:m:N,F:128:4,128:2");
  //L1�������棨ֻ�������á�������ԣ�L=LRU��F=FIFO��R=Random��???
  //1. cache_config: �������ã�����ָ����������ͣ�������L=LRU(Least Recently Used)��F=FIFO(First 
  //   In First Out)��R=Random��Pseudo-LRU��
  //2. cache_size: �����С������ָ������Ĵ�С�����ֽ�Ϊ��λ��
  //3. line_sz: �����д�С������ָ�������еĴ�С�����ֽ�Ϊ��λ��
  //4. associativity: ������ԣ�����ָ�������������ԣ�����2-way��4-way��8-way�ȡ�
  //5. num_banks: �洢����������������ָ���洢�����е�������������1��2��4�ȡ�
  //6. throughput: ����������������ָ�����������������ÿ���ֽ�Ϊ��λ��
  //7. latency: �����ӳ٣�����ָ��������ӳ٣���������Ϊ��λ��
  //���������󣡣�����⣺
  //<nsets>�������е������������������
  //<bsize>��ÿ�������е��ֽ���
  //<assoc>�����������ȣ���һ�����е�����
  //<rep>���滻���ԣ���LRU��FIFO��Random��
  //<wr>��д���ԣ���write-back��write-through
  //<alloc>��д������ԣ���write-allocate��no-write-allocate
  //<wr_alloc>��д�ط�����ԣ���write-allocate��no-write-allocate
  //<mshr>����·��������Ĵ����Ĵ�С
  //<N>��ÿ������Ĵ��������洢��������
  //<merge>���Ƿ���������ϲ�����yes��no
  //<mq>���Ƿ����ö��У���yes��no
  option_parser_register(
      opp, "-gpgpu_const_cache:l1", OPT_CSTR, &m_L1C_config.m_config_string,
      "per-shader L1 constant memory cache  (READ-ONLY) config "
      " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<"
      "merge>,<mq>} ",
      "64:64:2,L:R:f:N,A:2:32,4");
  //L1 instruction cache�����á�
  option_parser_register(opp, "-gpgpu_cache:il1", OPT_CSTR,
                         &m_L1I_config.m_config_string,
                         "shader L1 instruction cache config "
                         " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                         "alloc>,<mshr>:<N>:<merge>,<mq>} ",
                         "4:256:4,L:R:f:N,A:2:32,4");
  //L1 data cache�����á�
  option_parser_register(opp, "-gpgpu_cache:dl1", OPT_CSTR,
                         &m_L1D_config.m_config_string,
                         "per-shader L1 data cache config "
                         " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                         "alloc>,<mshr>:<N>:<merge>,<mq> | none}",
                         "none");
  option_parser_register(opp, "-gpgpu_l1_cache_write_ratio", OPT_UINT32,
                         &m_L1D_config.m_wr_percent, "L1D write ratio", "0");
  //L1 cache��bank�������� Volta unified cache �� 4 ��banks��
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
  //ÿ��Shader Core�ļĴ�����������CTA������������
  //-gpgpu_shader_registers <# registers/shader core, default=8192>
  option_parser_register(opp, "-gpgpu_shader_registers", OPT_UINT32,
                         &gpgpu_shader_registers,
                         "Number of registers per shader core. Limits number "
                         "of concurrent CTAs. (default 8192)",
                         "8192");
  //ÿ��CTA�����Ĵ�������Ĭ��ֵ8192����
  option_parser_register(
      opp, "-gpgpu_registers_per_block", OPT_UINT32, &gpgpu_registers_per_block,
      "Maximum number of registers per CTA. (default 8192)", "8192");
  option_parser_register(opp, "-gpgpu_ignore_resources_limitation", OPT_BOOL,
                         &gpgpu_ignore_resources_limitation,
                         "gpgpu_ignore_resources_limitation (default 0)", "0");
  //Shader Core�в���cta�����������-gpgpu_shader_cta <# CTA/shader core, default=8>
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
  //GPU���õ�SIMT Core��Ⱥ�ĵ����������е����ݰ���������������ָ���ǣ�[��������->����������->SIMT 
  //Core��Ⱥ]���м�ڵ㡣
  option_parser_register(opp, "-gpgpu_n_cluster_ejection_buffer_size",
                         OPT_UINT32, &n_simt_ejection_buffer_size,
                         "number of packets in ejection buffer", "8");
  //LD/ST��Ԫ�����������е���Ӧ������
  option_parser_register(
      opp, "-gpgpu_n_ldst_response_buffer_size", OPT_UINT32,
      &ldst_unit_response_queue_size,
      "number of response packets in ld/st unit ejection buffer", "2");
  //ÿ���߳̿��CTA�Ĺ����ڴ��С��Ĭ��48KB����
  option_parser_register(
      opp, "-gpgpu_shmem_per_block", OPT_UINT32, &gpgpu_shmem_per_block,
      "Size of shared memory per thread block or CTA (default 48kB)", "49152");
  //ÿ��SIMT Core��Ҳ��ΪShader Core���Ĺ���洢��С��
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
  //��subcoreģʽ�£�ÿ��warp�������ڼĴ�����������һ������ļĴ����ɹ�ʹ�ã������
  //�����ɵ�������m_id������
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
  //�Ĵ����ļ��Ķ˿�������V100�����ļ���gpgpu_reg_file_port_throughput������Ϊ2��
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
GPGPU-Sim 3.x�ṩ��һ��ͨ�õ�������ѡ�������������ͬ�����ģ��ͨ��һ���򵥵Ľӿ���ע�����ǵ�ѡ�
ѡ��������� gpgpusim_entrypoint.cc �� gpgpu_ptx_sim_init_perf() ��ʵ������ѡ���� reg_options() 
������ʹ�ú�����ӡ�
*/
void gpgpu_sim_config::reg_options(option_parser_t opp) {
  gpgpu_functional_sim_config::reg_options(opp);
  m_shader_config.reg_options(opp);
  m_memory_config.reg_options(opp);
  power_config::reg_options(opp);
  //�ڴﵽ���������������ֹGPUģ��(0 = no limit)��-gpgpu_max_cycle <# cycles>
  option_parser_register(opp, "-gpgpu_max_cycle", OPT_INT64, &gpu_max_cycle_opt,
                         "terminates gpu simulation early (0 = no limit)", "0");
  //�ڴﵽ���ָ����������ֹGPUģ��(0 = no limit)��-gpgpu_max_insn <# insns>
  option_parser_register(opp, "-gpgpu_max_insn", OPT_INT64, &gpu_max_insn_opt,
                         "terminates gpu simulation early (0 = no limit)", "0");
  //�ڴﵽ���CTA������������ֹGPUģ��(0 = no limit)��
  option_parser_register(opp, "-gpgpu_max_cta", OPT_INT32, &gpu_max_cta_opt,
                         "terminates gpu simulation early (0 = no limit)", "0");
  //�ڴﵽ���CTA�����������ֹGPUģ��(0 = no limit)��
  option_parser_register(opp, "-gpgpu_max_completed_cta", OPT_INT32,
                         &gpu_max_completed_cta_opt,
                         "terminates gpu simulation early (0 = no limit)", "0");
  //��ʾ����ʱͳ����Ϣ��-gpgpu_runtime_stat <frequency>:<flag> 
  option_parser_register(
      opp, "-gpgpu_runtime_stat", OPT_CSTR, &gpgpu_runtime_stat,
      "display runtime statistics such as dram utilization {<freq>:<flag>}",
      "10000:0");
  //ģ����Ϣ֮�����С������0=ʼ�մ�ӡ����
  option_parser_register(opp, "-liveness_message_freq", OPT_INT64,
                         &liveness_message_freq,
                         "Minimum number of seconds between simulation "
                         "liveness messages (0 = always print)",
                         "1");
  //�����豸����������
  option_parser_register(opp, "-gpgpu_compute_capability_major", OPT_UINT32,
                         &gpgpu_compute_capability_major,
                         "Major compute capability version number", "7");
  //��С���豸����������
  option_parser_register(opp, "-gpgpu_compute_capability_minor", OPT_UINT32,
                         &gpgpu_compute_capability_minor,
                         "Minor compute capability version number", "0");
  //��ÿ���ں˵��ý���ʱˢ��L1���档
  option_parser_register(opp, "-gpgpu_flush_l1_cache", OPT_BOOL,
                         &gpgpu_flush_l1_cache,
                         "Flush L1 cache at the end of each kernel call", "0");
  //��ÿ���ں˵��ý���ʱˢ��L2���档
  option_parser_register(opp, "-gpgpu_flush_l2_cache", OPT_BOOL,
                         &gpgpu_flush_l2_cache,
                         "Flush L2 cache at the end of each kernel call", "0");
  //������ʱֹͣģ�⡣-gpgpu_deadlock_detect <0=off, 1=on(default)>
  option_parser_register(
      opp, "-gpgpu_deadlock_detect", OPT_BOOL, &gpu_deadlock_detect,
      "Stop the simulation at deadlock (1=on (default), 0=off)", "1");
  //����ָ����࣬������ã�����ÿ���ں˵�ptxָ�����ͽ��з��ࣨ�������255���ںˣ���
  //-gpgpu_ptx_instruction_classification <0=off, 1=on (default)>
  option_parser_register(
      opp, "-gpgpu_ptx_instruction_classification", OPT_INT32,
      &(gpgpu_ctx->func_sim->gpgpu_ptx_instruction_classification),
      "if enabled will classify ptx instruction types per kernel (Max 255 "
      "kernels now)",
      "0");
  //�����ܻ���ģ��֮�����ѡ����ע�⣬����ģ����ܻ�����ģ��ĳЩptx���룬��Щ������Ҫwarp��ÿ
  //��Ԫ����lock-step��ִ�У���-gpgpu_ptx_sim_mode <0=performance(default), 1=functional>
  option_parser_register(
      opp, "-gpgpu_ptx_sim_mode", OPT_INT32,
      &(gpgpu_ctx->func_sim->g_ptx_sim_mode),
      "Select between Performance (default) or Functional simulation (1)", "0");
  //��MHzΪ��λ��ʱ����Ƶ�ʡ�
  //-gpgpu_clock_domains <Core Clock>:<Interconnect Clock>:<L2 Clock>:<DRAM Clock>
  option_parser_register(opp, "-gpgpu_clock_domains", OPT_CSTR,
                         &gpgpu_clock_domains,
                         "Clock Domain Frequencies in MhZ {<Core Clock>:<ICNT "
                         "Clock>:<L2 Clock>:<DRAM Clock>}",
                         "500.0:2000.0:2000.0:2000.0");
  //������GPU��ͬʱ���е�����ں�����
  option_parser_register(
      opp, "-gpgpu_max_concurrent_kernel", OPT_INT32, &max_concurrent_kernel,
      "maximum kernels that can run concurrently on GPU, set this value "
      "according to max resident grids for your compute capability",
      "32");
  //��������¼����ÿ������֮��ļ����
  option_parser_register(
      opp, "-gpgpu_cflog_interval", OPT_INT32, &gpgpu_cflog_interval,
      "Interval between each snapshot in control flow logger", "0");
  //�򿪿��ӻ����������ʹ��AerialVision���ӻ����߻��Ʊ�������־�е����ݣ���
  option_parser_register(opp, "-visualizer_enabled", OPT_BOOL,
                         &g_visualizer_enabled,
                         "Turn on visualizer output (1=On, 0=Off)", "1");
  //ָ�����ӻ����ߵ������־�ļ���
  option_parser_register(opp, "-visualizer_outputfile", OPT_CSTR,
                         &g_visualizer_filename,
                         "Specifies the output log file for visualizer", NULL);
  //���ӻ����������־��ѹ������0=��ѹ����9=���ѹ������
  option_parser_register(
      opp, "-visualizer_zlevel", OPT_INT32, &g_visualizer_zlevel,
      "Compression level of the visualizer output log (0=no comp, 9=highest)",
      "6");
  //GPU�̶߳�ջ��С��
  option_parser_register(opp, "-gpgpu_stack_size_limit", OPT_INT32,
                         &stack_size_limit, "GPU thread stack size", "1024");
  //GPU malloc�Ѵ�С��
  option_parser_register(opp, "-gpgpu_heap_size_limit", OPT_INT32,
                         &heap_size_limit, "GPU malloc heap size ", "8388608");
  //GPU�豸����ʱͬ��������ơ�
  option_parser_register(opp, "-gpgpu_runtime_sync_depth_limit", OPT_INT32,
                         &runtime_sync_depth_limit,
                         "GPU device runtime synchronize depth", "2");
  //GPU�豸����ʱ���������������ơ�
  option_parser_register(opp, "-gpgpu_runtime_pending_launch_count_limit",
                         OPT_INT32, &runtime_pending_launch_count_limit,
                         "GPU device runtime pending launch count", "2048");
  //ȫ�����û��������trace��������ã����ӡ trace_components ��
  option_parser_register(opp, "-trace_enabled", OPT_BOOL, &Trace::enabled,
                         "Turn on traces", "0");
  //Ҫ����trance�Ķ��ŷָ��б������б���� src/trace_streams.tup ���ҵ���
  option_parser_register(opp, "-trace_components", OPT_CSTR, &Trace::config_str,
                         "comma seperated list of traces to enable. "
                         "Complete list found in trace_streams.tup. "
                         "Default none",
                         "none");
  //���������shader core������Ԫ�أ���warp��������Ƿ��ƣ�������ӡ���Ըú��ĵ�trace��
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
  //�ں������ӳ٣�������Ϊ��λ����
  option_parser_register(opp, "-gpgpu_kernel_launch_latency", OPT_INT32,
                         &(gpgpu_ctx->device_runtime->g_kernel_launch_latency),
                         "Kernel launch latency in cycles. Default: 0", "0");
  //����CDP��
  option_parser_register(opp, "-gpgpu_cdp_enabled", OPT_BOOL,
                         &(gpgpu_ctx->device_runtime->g_cdp_enabled),
                         "Turn on CDP", "0");
  //�߳̿������ӳ٣�������Ϊ��λ����
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
�˺�������kinfoָ�����ںˡ��������ں˵��߳̿顢�̺߳�warp��Ȼ�����豸���������ǡ��ں������󣬸ú�����
�ȴ������߳���ɡ��߳���ɺ󣬸ú����������ں�ʹ�õ���Դ��
kernel_info_t����abstract_hardware_model.h�ж��塣
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
  //�����ں˺�������Ϣkinfo��ȡ������е�ÿ��CTA���߳̿飩�е��߳�����
  unsigned cta_size = kinfo->threads_per_cta();
  //��������ÿ��CTA�е��߳����� > ÿ��SIMT Core���õ��߳�������-gpgpu_shader���ã������������
  //Ϣ��
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
  //m_running_kernels��gpu-sim.h�е� std::vector<kernel_info_t *> ���壺
  //    std::vector<kernel_info_t *> m_running_kernels;
  //��һ��kernel_info_t*��ɵ����������洢���������е��ں˵���Ϣ���������������������ҵ�һ����λ��
  //�����µļ������е��ں�kinfo�����������ĳ��λ��ΪNULL���߸�λ��->done()��ʾ�ú˺�������ɣ���
  //kinfo���뵽��λ�á�
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
gpgpu_sim::can_start_kernel()��һ������������鵱ǰGPU�Ƿ����㹻����Դ������һ���µ��ںˡ������
�㹻����Դ���ú���������true�����򷵻�false���������Դ��飬��Ҫ�ǿ��洢���������е��ں˵���Ϣ��
m_running_kernels�������Ƿ���λ�ÿ��Լ������ںˣ����������ĳ��λ��ΪNULL���߸�λ��->done()��ʾ��
�˺�������ɣ�����Լ������ںˡ�
*/
bool gpgpu_sim::can_start_kernel() {
  for (unsigned n = 0; n < m_running_kernels.size(); n++) {
    if ((NULL == m_running_kernels[n]) || m_running_kernels[n]->done())
      return true;
  }
  return false;
}

/*
gpu_max_cta_optѡ����ָ���ǣ�GPGPU-Sim���ܴﵽ���CTA������(0 = no limit)��������ѡ���ж��塣
gpu_tot_issued_cta���ܷ�����CTA��Compute Thread Array���༴�߳̿�����������m_total_cta_launched
���Ѿ�������CTA����Ҫ���ڻ����GPU�����CTA����������m_config.gpu_max_cta_opt��������Ϊ��ȷ��GPU��
������ܡ�
*/
bool gpgpu_sim::hit_max_cta_count() const {
  if (m_config.gpu_max_cta_opt != 0) {
    if ((gpu_tot_issued_cta + m_total_cta_launched) >= m_config.gpu_max_cta_opt)
      return true;
  }
  return false;
}

/*
�ú������ڼ��ָ�����ں��Ƿ��и����CTA��Compute Thread Array����Ҫִ�С�������и����CTA��Ҫִ��
����������true�����û�и����CTA��Ҫִ�У���������false�����Ѿ��ﵽGPUģ�����������CTA����
������gpgpu_sim::hit_max_cta_count()�жϣ�����û��ʣ���CTA������False����kernel�ǿգ���kernel->
no_more_ctas_to_run()Ϊfalse��kernel�Լ������ж���CTAִ�У��򷵻�True��no_more_ctas_to_run()����
ָʾ��ǰû�и����CTA��Compute Thread Array����Ҫִ�С� 
*/
bool gpgpu_sim::kernel_more_cta_left(kernel_info_t *kernel) const {
  if (hit_max_cta_count()) return false;

  if (kernel && !kernel->no_more_ctas_to_run()) return true;

  return false;
}

/*
�ú������ڼ�鵱ǰ�Ƿ��и����CTA��Compute Thread Array����Ҫִ�С�����鵱ǰ��Ծ��CTA����������
���Ƿ��и���CTA��Ҫִ�С�����ѴﵽGPUģ����������CTA������gpgpu_sim::hit_max_cta_count()�жϣ���
��û��ʣ���CTA������False�����ĳ��m_running_kernels�������kernel�ǿգ���kernel->
no_more_ctas_to_run()Ϊfalse��kernel�Լ������ж���CTAִ�У��򷵻�True��no_more_ctas_to_run()����
ָʾ��ǰû�и����CTA��Compute Thread Array����Ҫִ�С�
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
�ú������ڼ����ں��ӳ٣�kernel latency���������ٴӷ����ں�����ں����ִ�е�ʱ�䣨�������������е�
�ں˵��ӳټ�1�����ú�����������ģ��GPU�ں˵����ܣ��Լ�������������ܡ�m_kernel_TB_latency��ʾÿһ��
�߳̿�������ӳ�֮�ͼ���kernel�������ӳ٣����ӷ����߳̿鵽�����ִ�е�ʱ�䡣m_kernel_TB_latency��
GPGPU-Sim�����ڱ�ʾ�ں�����ʱ��ı���������ʾ���ں��������ں����ִ������Ҫ��ʱ�䡣
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
m_last_issued_kernel�ڲ�������ʾ���һ�η������ںˡ�����ֵ��һ��ָ���ں˵�ָ�룬�������������ں˵�ִ
�н��ȡ�m_running_kernels[m_last_issued_kernel]ָ�������һ�η������ںˡ��ú������ڴӵ�ǰ��ں�
�б���ѡ�����ŵ��ںˣ��Ա㽫������GPU��������һ��ָ���ں���Ϣ�ṹ��ָ�룬�ýṹ�����й��ں˵�������
Ϣ�������ں����ƣ��������߳����������ȡ�
*/
kernel_info_t *gpgpu_sim::select_kernel() {
  //������ں˷ǿգ��Ҹ��ں��и����CTA��Compute Thread Array����Ҫ���У�����m_kernel_TB_latencyΪ
  //�㣬��m_last_issued_kernel���Ա�����ѡ�������������Щ���������������������е��ں���ѡ��
  //m_kernel_TB_latency��ʾ���ں��������ں����ִ������Ҫ��ʱ�䣬������ֵ��Ϊ�㣬���������ں���δ
  //ִ���꣬����Ա�����ִ�С�
  if (m_running_kernels[m_last_issued_kernel] &&
      !m_running_kernels[m_last_issued_kernel]->no_more_ctas_to_run() &&
      !m_running_kernels[m_last_issued_kernel]->m_kernel_TB_latency) {
    //get_uid()����һ��Ψһ��32λ���������ڱ�ʶ��ͬ��GPU�ںˣ�ÿ���ں���һ��������id��ʾ����uid����
    unsigned launch_uid = m_running_kernels[m_last_issued_kernel]->get_uid();
    //m_executed_kernel_uids�洢�������Ѿ�ִ����ϵ��ں˵�Ψһ��ʶ������uid��std::find�������û��
    //�Ѿ�ִ����ϵ��ں��б����ҵ����ںˣ���˵������û��ִ���꣬���Ա�ѡ��ִ�С�
    if (std::find(m_executed_kernel_uids.begin(), m_executed_kernel_uids.end(),
                  launch_uid) == m_executed_kernel_uids.end()) {
      //gpu_tot_sim_cycle������ʾ��ǰ�ķ������ڣ����ӷ��濪ʼ����ǰ���ܷ���ʱ�䡣gpu_sim_cycle����
      //��ʾִ�д��ں������ʱ������������shader core��ʱ����Ϊ��λ����
      //m_running_kernels[m_last_issued_kernel]->start_cycle������ʾ���һ���������ں˵Ŀ�ʼ���ڣ�
      //���ڸ����ں˵�ִ��������Ա�����ں˵���ִ��ʱ�䡣
      //ѡ����ں�ִ���Ժ󣬸���m_running_kernels[m_last_issued_kernel]->start_cycle�Ա��´�ѡ��
      //���ں�ִ��ʱʹ�ã�����һ���ں�һ���ں˵��ۼ�ʱ�䡣
      //ͬʱ��m_running_kernels[m_last_issued_kernel]��״̬����Ϊexecuted������uidѹ��m_executed
      //_kernel_uids������nameѹ��m_executed_kernel_names��

      //������仰����˼�ǣ�gpu_sim_cycle��������һ��ִ�е��ں˵��ӳ٣�gpu_tot_sim_cycle��������һ��
      //ִ�е��ں�֮ǰ�������ں˵�ִ��ʱ�䣬��˵�ǰ�ں˵Ŀ�ʼ����ʱ�伴Ϊ������ӡ�gpgpu_sim::cycle()
      //ÿ��һ�Ľ�gpu_sim_cycle++��
      m_running_kernels[m_last_issued_kernel]->start_cycle =
          gpu_sim_cycle + gpu_tot_sim_cycle;
      m_executed_kernel_uids.push_back(launch_uid);
      m_executed_kernel_names.push_back(
          m_running_kernels[m_last_issued_kernel]->name());
    }
    return m_running_kernels[m_last_issued_kernel];
  }
  //m_last_issued_kernel�����㱻����ѡ��ִ�е�����ʱ����������������е��ں���ѡ��ѡ����������·�
  //ʽ����(n + m_last_issued_kernel + 1) % m_config.max_concurrent_kernel����˳��ѡ����һ�η�����
  //m_last_issued_kernel��m_running_kernels�е���һ����ŵ��ںˣ�������ѯ��max_concurrent_kernel
  //��ʾģ���GPU�Ͽ��ܲ���ִ�е�����ں�������
  for (unsigned n = 0; n < m_running_kernels.size(); n++) {
    unsigned idx =
        (n + m_last_issued_kernel + 1) % m_config.max_concurrent_kernel;
    //���idx��ʶ���ں��и����CTA��Ҫִ�У�����ִ���ӳ�m_kernel_TB_latency��δ����0����������ִ�У�
    //����Է�����
    if (kernel_more_cta_left(m_running_kernels[idx]) &&
        !m_running_kernels[idx]->m_kernel_TB_latency) {
      m_last_issued_kernel = idx;
      //gpu_sim_cycle������ʾִ�д��ں������ʱ������������shader core��ʱ����Ϊ��λ����gpgpu_sim::
      //cycle()ÿ��һ�Ľ�gpu_sim_cycle++��
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
���ѽ������ں˶���m_finished_kernel��ѡ��ͷ�����ں˲����ء�
*/
unsigned gpgpu_sim::finished_kernel() {
  //m_finished_kernel.empty()Ϊ1��ʾ��ǰ��ʱ���Ѿ��������ںˡ�
  if (m_finished_kernel.empty()) {
    last_streamID = -1;
    return 0;
  }
  //ѡ��ͷ�����ںˡ�
  unsigned result = m_finished_kernel.front();
  m_finished_kernel.pop_front();
  return result;
}

/*
���ý������ں˵�״̬��������m_running_kernels�и��ں��޳������Լ������������ʱ�����ڡ�
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
ֹͣ�����������е��ںˡ�
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
����SIMT Cluster��m_cluster[...]�洢�����е�SM��
  1.m_shader_config������shader Core�����ã�����ÿ��shader��������ָ���ȡ����ݿ�ȡ�ָ����С
    �ȣ�
  2.m_memory_config������洢ģ������ã�����ÿ���洢ģ��Ķ�д����latency�ȣ�
  3.m_shader_stats������shader��������ͳ����Ϣ������ÿ��shader��������ָ��ִ�д�����ָ������д�
    ���ȣ�
  4.m_memory_stats������洢ģ���ͳ����Ϣ������ÿ���洢ģ��Ķ�д������cache���д����ȣ�
*/
void exec_gpgpu_sim::createSIMTCluster() {
  //m_cluster��gpu-sim.h�ж��壺class simt_core_cluster **m_cluster;
  //n_simt_clusters�������е�SM�����������������п�����ÿ����Ⱥ���ж��SM�����ö�ά���顣
  m_cluster = new simt_core_cluster *[m_shader_config->n_simt_clusters];
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
    m_cluster[i] =
        new exec_simt_core_cluster(this, i, m_shader_config, m_memory_config,
                                   m_shader_stats, m_memory_stats);
}

/*
���ܷ���������ͨ�� src/gpgpu-sim �µ��ļ��ж����ʵ�ֵ��������ʵ�ֵġ���Щ��ͨ�������� gpgpu_sim 
�㼯��һ�𣬸������� gpgpu_t ���书�ܷ����Ӧ���֣������ġ��ڵ�ǰ�汾��GPGPU-Sim�У�ģ������ֻ��һ�� 
gpgpu_sim ��ʵ�� g_the_gpu��Ŀǰ��֧��ͬʱ�Զ��GPU���з��棬����δ���İ汾�п��ܻ��ṩ��
*/
gpgpu_sim::gpgpu_sim(const gpgpu_sim_config &config, gpgpu_context *ctx)
    : gpgpu_t(config, ctx), m_config(config) {
  //gpgpu_context *ctx��libcuda/gpgpu_context.h�ж��塣
  gpgpu_ctx = ctx;
  m_shader_config = &m_config.m_shader_config;
  m_memory_config = &m_config.m_memory_config;
  ctx->ptx_parser->set_ptx_warp_size(m_shader_config);
  //m_config.num_shader()����Ӳ�����е�SM���ֳ�Shader Core����������
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
  //�ڴ��ں���ִ�е�ָ������
  gpu_sim_insn = 0;
  //����ΪֹΪ�����������ں�ģ��������������Ժ���ʱ��Ϊ��λ����
  gpu_tot_sim_insn = 0;
  //���ܷ�����CTA��Compute Thread Array���༴�߳̿�������
  gpu_tot_issued_cta = 0;
  //�Ѿ���ɵ�CTA��������
  gpu_completed_cta = 0;
  //�Ѿ�������CTA������
  m_total_cta_launched = 0;
  //GPU��������״̬�ı�־��
  gpu_deadlock = false;
  //�������������DRAM Channel����ͣ��������������icnt������L2_queueʱ��m_icnt_L2_queueû��SECTOR_
  //CHUNCK_SIZE��С�Ŀռ���Ա���������Ϣ����˻��������ӵ�����DRAM��ͣ�ʹ�����
  gpu_stall_dramfull = 0;
  //���ڻ���ӵ������DRAM Channelͣ�͵����������ڴӴ洢�������������絯��ʱ����������������п���
  //�Ļ����������ڴ��������뻥�����硣����һ�����������еĻ�������ռ�����ͻ�ֹͣ���͡����ڻ�����
  //�绺�����Ĵ�С������ɵ�ͣ��ʱ����������gpu_stall_icnt2sh��������������
  gpu_stall_icnt2sh = 0;
  //��ģ���������ĵ�һ��ʱ�����ڿ�ʼ��partiton_reqs_in_parallel�͸���ÿһ�ĵ��ڴ��������������ʼ
  //������partiton_reqs_in_parallel��ʾ��ģ���������ĵ�һ��ʱ�����ڿ�ʼ�����д洢����������������
  //��m_memory_sub_partition��m_icnt_L2_queue���ܸ�����
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

  //��ʼ��������������ã�ָ����������������Լ�ѡ���Ӧ��Push/Pop�����̡�
  icnt_wrapper_init();
  //�����������硣
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
��ȡÿ��SIMT Core��Ҳ��ΪShader Core���Ĺ���洢��С����GPGPU-Sim��-gpgpu_shmem_sizeѡ�����á�
*/
int gpgpu_sim::shared_mem_size() const {
  return m_shader_config->gpgpu_shmem_size;
}

/*
��ȡÿ���߳̿��CTA�Ĺ����ڴ��С��Ĭ��48KB������GPGPU-Sim��-gpgpu_shmem_per_blockѡ�����á�
*/
int gpgpu_sim::shared_mem_per_block() const {
  return m_shader_config->gpgpu_shmem_per_block;
}

/*
��ȡÿ��Shader Core�ļĴ�����������CTA��������������GPGPU-Sim��-gpgpu_shader_registersѡ�����á�
*/
int gpgpu_sim::num_registers_per_core() const {
  return m_shader_config->gpgpu_shader_registers;
}

/*
��ȡÿ��CTA�����Ĵ���������GPGPU-Sim��-gpgpu_registers_per_blockѡ�����á�
*/
int gpgpu_sim::num_registers_per_block() const {
  return m_shader_config->gpgpu_registers_per_block;
}

/*
��ȡһ��warp�ж����߳�������GPGPU-Sim��-gpgpu_shader_core_pipeline�ĵڶ���ѡ�����á�
ѡ��-gpgpu_shader_core_pipeline�Ĳ����ֱ��ǣ�<ÿ��SM����֧���߳���>:<����һ��warp�ж����߳�>
*/
int gpgpu_sim::wrp_size() const { return m_shader_config->warp_size; }

/*
��ȡ��MHzΪ��λ��ʱ����Ƶ�ʵ�<Core Clock>����GPGPU-Sim��-gpgpu_clock_domains�ĵ�һ��ѡ�����á�
*/
int gpgpu_sim::shader_clock() const { return m_config.core_freq / 1000; }

/*
��ȡShader Core�в���cta�������������GPGPU-Sim��-gpgpu_shader_ctaѡ�����á�
*/
int gpgpu_sim::max_cta_per_core() const {
  return m_shader_config->max_cta_per_core;
}

/*
����һ��Core�Ͽ�ͬʱ���ȵ�����߳̿飨���ΪCTA�������������ɺ���shader_core_config::max_cta(...)��
�㡣max_cta(...)�������ݳ���ָ����ÿ���߳̿��������ÿ���̼߳Ĵ�����ʹ������������ڴ��ʹ������Լ���
�õ�ÿ��Core����߳̿����������ƣ�ȷ�����Բ������������SIMT Core������߳̿�����������˵���������ÿ
����׼�����������أ���ô���Է����SIMT Core���߳̿��������������������е���Сֵ���ǿ��Է����SIMT 
Core������߳̿�����
*/
int gpgpu_sim::get_max_cta(const kernel_info_t &k) const {
  return m_shader_config->max_cta(k);
}

/*
m_cuda_properties������һ���ṹ�壬���ڴ洢CUDA�豸�����ܺ͹������ԣ���������߳����������С�����
�����С�ȡ�
*/
void gpgpu_sim::set_prop(cudaDeviceProp *prop) { m_cuda_properties = prop; }

/*
���������豸������������-gpgpu_compute_capability_majorѡ�����á�
*/
int gpgpu_sim::compute_capability_major() const {
  return m_config.gpgpu_compute_capability_major;
}

/*
������С���豸������������-gpgpu_compute_capability_minorѡ�����á�
*/
int gpgpu_sim::compute_capability_minor() const {
  return m_config.gpgpu_compute_capability_minor;
}

/*
����m_cuda_properties�����ṹ�壬���ڴ洢CUDA�豸�����ܺ͹������ԡ�
*/
const struct cudaDeviceProp *gpgpu_sim::get_prop() const {
  return m_cuda_properties;
}

enum divergence_support_t gpgpu_sim::simd_model() const {
  return m_shader_config->model;
}

/*
��ʼ��ʱ����GPGPU-Sim֧���ĸ�������ʱ����
��1��SIMT Core��Ⱥʱ����core_freq;
��2����������ʱ����icnt_freq;
��3��L2���ٻ���ʱ�����������ڴ������Ԫ�г�DRAM֮��������߼���l2_freq;
��4��DRAMʱ����dram_freq��
*/
void gpgpu_sim_config::init_clock_domains(void) {
  sscanf(gpgpu_clock_domains, "%lf:%lf:%lf:%lf", &core_freq, &icnt_freq,
         &l2_freq, &dram_freq);
  core_freq = core_freq MhZ;
  icnt_freq = icnt_freq MhZ;
  l2_freq = l2_freq MhZ;
  dram_freq = dram_freq MhZ;
  //���ڡ�
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
���³�ʼ����һ�������ص�ʱ�̡�������ǰʱ���ĸ�ʱ�����ʱ��ֵ�����㡣
*/
void gpgpu_sim::reinit_clock_domains(void) {
  core_time = 0;
  dram_time = 0;
  icnt_time = 0;
  l2_time = 0;
}

/*
����GPGPU-Simģ�����Ƿ��ڻ�Ծ״̬��
*/
bool gpgpu_sim::active() {
  //gpu_max_cycle_optѡ�����ã��ڴﵽ���������������ֹGPUģ�⡣
  //gpu_sim_cycle��ִ�е�ǰ�׶ε�ָ����ӳ٣�gpgpu_sim::cycle()ÿ��һ�Ľ�gpu_sim_cycle++��
  //gpu_tot_sim_cycle��ִ�е�ǰ�׶�֮ǰ������ǰ��ָ����ӳ١�
  //�����ӳ���� >= gpu_max_cycle_opt˵����ﵽ���������������False��
  if (m_config.gpu_max_cycle_opt &&
      (gpu_tot_sim_cycle + gpu_sim_cycle) >= m_config.gpu_max_cycle_opt)
    return false;
  //gpu_max_insn_optѡ�����ã��ڴﵽ���ָ����������ֹGPUģ�⡣
  //gpu_sim_insn��ִ�е�ǰ�׶ε�ָ������������罫����warp����ӡ�
  //gpu_tot_sim_insn��ִ�е�ǰ�׶�֮ǰ������ǰ��ָ���������
  //����������� >= gpu_max_insn_opt˵����ﵽ���ָ����������False��
  if (m_config.gpu_max_insn_opt &&
      (gpu_tot_sim_insn + gpu_sim_insn) >= m_config.gpu_max_insn_opt)
    return false;
  //gpu_max_cta_optѡ�����ã�GPGPU-Sim���ܴﵽ���CTA������������ֹGPUģ�⡣
  //gpu_tot_issued_cta���ܷ�����CTA��Compute Thread Array���༴�߳̿�������
  //�ܷ�����CTA���� >= gpu_max_cta_opt������False��
  if (m_config.gpu_max_cta_opt &&
      (gpu_tot_issued_cta >= m_config.gpu_max_cta_opt))
    return false;
  //gpu_max_completed_cta_optѡ�����ã��ڴﵽ���CTA�����������ֹGPUģ�⡣
  //gpu_completed_cta���Ѿ���ɵ�CTA��������
  //�Ѿ���ɵ�CTA������ >= gpu_max_completed_cta_opt������False��
  if (m_config.gpu_max_completed_cta_opt &&
      (gpu_completed_cta >= m_config.gpu_max_completed_cta_opt))
    return false;
  //gpu_deadlock_detectѡ�����ã�������ʱֹͣģ�⡣
  if (m_config.gpu_deadlock_detect && gpu_deadlock) return false;
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
    if (m_cluster[i]->get_not_completed() > 0) return true;
  ;
  for (unsigned i = 0; i < m_memory_config->m_n_mem; i++)
    if (m_memory_partition_unit[i]->busy() > 0) return true;
  ;
  //icnt_busy()�жϻ��������Ƿ���Busy״̬��������һ�������紦��Busy״̬����Ϊ
  //�����������紦��Busy״̬��
  if (icnt_busy()) return true;
  if (get_more_cta_left()) return true;
  return false;
}

/*
��ʼ��GPGPU-Sim�����ò�����
*/
void gpgpu_sim::init() {
  // run a CUDA grid on the GPU microarchitecture simulator
  //ִ�е�ǰ�׶ε�ָ����ӳ٣�gpgpu_sim::cycle()ÿ��һ�Ľ�gpu_sim_cycle++��
  gpu_sim_cycle = 0;
  //ִ�е�ǰ�׶ε�ָ������������罫����warp����ӡ�
  gpu_sim_insn = 0;
  last_gpu_sim_insn = 0;
  //�Ѿ�������CTA������
  m_total_cta_launched = 0;
  //�Ѿ���ɵ�CTA��������
  gpu_completed_cta = 0;
  //��ģ���������ĵ�һ��ʱ�����ڿ�ʼ�����д洢������������������m_memory_sub_partition��
  //m_icnt_L2_queue���ܸ�����
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
    //icnt_busy()�жϻ��������Ƿ���Busy״̬��������һ�������紦��Busy״̬����Ϊ
    //�����������紦��Busy״̬��
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

  //�ڵ�ǰkernel��ģ����ģ���ڼ䣬ģ�������е���������������PyTorchʱ�ж�����Kernel���򵥸������п�
  //�ܱ�������Kernelʱ����Ҫ�������ģ������ִ�ж��Kernel����ʱ����ÿ��ģ��������ʱ����Ҫһ��ȫ�ֵ�
  //��¼�������ı�������¼����Kernel��ִ������������˾���gpu_tot_sim_cycle����ʾ��һȫ�ֵ�ʱ��������
  //�����������ֻ��һ��Kernelִ�еĻ���������һ��ģ��������ôgpu_tot_sim_cycle��ʼ��Ϊ0�����й�����
  //�ĵ�ǰ��ʱ����������gpu_sim_cycle��¼��
  printf("gpu_sim_cycle = %lld\n", gpu_sim_cycle);
  //�ڵ�ǰkernel��ģ����ģ���ڼ䣬ģ�������е�ָ�����������ж��Kernelʱ����gpu_tot_sim_cycle���ƣ�
  //��gpu_tot_sim_insnά��ȫ�ֵ�ִ��ָ������������
  printf("gpu_sim_insn = %lld\n", gpu_sim_insn);
  //�ڵ�ǰkernel��ģ����ģ���ڼ䣬IPC��
  printf("gpu_ipc = %12.4f\n", (float)gpu_sim_insn / gpu_sim_cycle);
  printf("gpu_tot_sim_cycle = %lld\n", gpu_tot_sim_cycle + gpu_sim_cycle);
  printf("gpu_tot_sim_insn = %lld\n", gpu_tot_sim_insn + gpu_sim_insn);
  //���ж��kernel��ģ���ڼ䣬IPC��
  printf("gpu_tot_ipc = %12.4f\n", (float)(gpu_tot_sim_insn + gpu_sim_insn) /
                                       (gpu_tot_sim_cycle + gpu_sim_cycle));
  //�ڵ�ǰkernel��ģ����ģ���ڼ䣬m_total_cta_launchedά����ǰKernel��CTA�ķ������������ж��Kernel
  //��ʱ����gpu_tot_sim_cycle���ƣ�gpu_tot_issued_ctaά�����Kernelִ���ڼ��ȫ�ֵ�CTA�ķ���������
  printf("gpu_tot_issued_cta = %lld\n",
         gpu_tot_issued_cta + m_total_cta_launched);
  printf("gpu_occupancy = %.4f%% \n", gpu_occupancy.get_occ_fraction() * 100);
  printf("gpu_tot_occupancy = %.4f%% \n",
         (gpu_occupancy + gpu_tot_occupancy).get_occ_fraction() * 100);

  fprintf(statfout, "max_total_param_size = %llu\n",
          gpgpu_ctx->device_runtime->g_max_total_param_size);

  // performance counter for stalls due to congestion.
  //����ӵ������ͣ�����ܼ�������
  
  //�������������DRAM Channel����ͣ��������������icnt������L2_queueʱ��m_icnt_L2_queueû��SECTOR_
  //CHUNCK_SIZE��С�Ŀռ���Ա���������Ϣ����˻��������ӵ�����DRAM��ͣ�ʹ�����
  printf("gpu_stall_dramfull = %d\n", gpu_stall_dramfull);
  //�ڴӴ洢�������������絯��ʱ����������������п��еĻ����������ڴ��������뻥�����硣����һ��
  //���������еĻ�������ռ�����ͻ�ֹͣ���͡����ڻ������绺�����Ĵ�С������ɵ�ͣ��ʱ����������gpu_st
  //all_icnt2sh��������������
  printf("gpu_stall_icnt2sh    = %d\n", gpu_stall_icnt2sh);

  // printf("partiton_reqs_in_parallel = %lld\n", partiton_reqs_in_parallel);
  // printf("partiton_reqs_in_parallel_total    = %lld\n",
  // partiton_reqs_in_parallel_total );
  
  //��ģ���������ĵ�һ��ʱ�����ڿ�ʼ��partiton_reqs_in_parallel�͸���ÿһ�ĵ��ڴ��������������ʼ
  //������partiton_reqs_in_parallel��ʾ��ģ���������ĵ�һ��ʱ�����ڿ�ʼ�����д洢����������������
  //��m_memory_sub_partition��m_icnt_L2_queue���ܸ���������partiton_level_parallism��ָ�ڵ�ǰ��
  //�ں�ִ���ڼ�ƽ��ÿ��ʱ�������ڴ洢������������������m_memory_sub_partition��m_icnt_L2_queue
  //�ĸ�����
  printf("partiton_level_parallism = %12.4f\n",
         (float)partiton_reqs_in_parallel / gpu_sim_cycle);
  //partiton_reqs_in_parallel_total��ָ���Kernelִ���ڼ����д洢������������������L2_queue���ܸ�
  //�������ָ����gpu_tot_sim_cycle���ơ�
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
�ж��Ƿ���Է���һ���߳̿飬������Է���һ���߳̿飬�򷵻�true�����򷵻�false��
*/
bool shader_core_ctx::can_issue_1block(kernel_info_t &kernel) {
  // Jin: concurrent kernels on one SM
  //֧��SM�ϵĲ����ںˣ�Ĭ��Ϊ���ã�����V100�����н��á�
  if (m_config->gpgpu_concurrent_kernel_sm) {
    if (m_config->max_cta(kernel) < 1) return false;

    return occupy_shader_resource_1block(kernel, false);
  } else {
    //get_n_active_cta()���ص�ǰSM�ϵĻ�Ծ�߳̿��������m_config->max_cta(kernel)���Ǽ���kernel��
    //֧�ֵĵ���SM�ڵ�����߳̿����������ǰSM�ϵĻ�Ծ�߳̿������С��kernel֧�ֵĵ���SM�ڵ�����߳�
    //��������˵����ʱ����һ������kernel���߳̿��ǿ��еģ�����true�����򷵻�false��
    return (get_n_active_cta() < m_config->max_cta(kernel));
  }
}

/*
�����ں�ʹ�ã���V100�������ò�����
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
�����ں�ʹ�ã���V100�������ò�����
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
����һ��CTA������ ptx_sim_init_thread ��ʼ�������߳̿�ʼ��Ȼ��ʹ��ptx_exec_inst()��warp��ִ�б�
���̡߳���ÿ���̵߳Ĺ���״̬ͨ������ ptx_sim_init_thread ���г�ʼ����������ǶԵ����̵߳�ָ�����
��ʼ���������а���ָ��ĳ���̵߳� unsigned hw_cta_id, unsigned hw_warp_id, �Լ� ptx_thread_info 
**thread_info, int sid, unsigned tid����Ҫע����ǣ�������е��ǵ���CTA�ڵ������߳̽���ѭ����

�����ڴ�ռ�������CTA���߳̿飩�����еģ���ÿ��CTA������ִ��ʱ���ں��� ptx_sim_init_thread() �У���
Ϊ�����һ��Ψһ�� memory_space ���󡣵�CTAִ����Ϻ󣬸ö���ȡ�����䡣

ptx_sim_init_thread �� functionalCoreSim::initializeCTA �����б����ã���������ϸ˵�����õ��ô���

�������壺ptx_sim_init_thread(kernel_info_t &kernel,
                                ptx_thread_info **thread_info, int sid,
                                unsigned tid, unsigned threads_left,
                                unsigned num_threads, core_t *core,
                                unsigned hw_cta_id, unsigned hw_warp_id,
                                gpgpu_t *gpu, bool isInFunctionalSimulationMode)
������
  sid=0��SM��index����������ִ�й���ģ�⣬���SM��index����Ҫ��������ȫ��������Ҫִ�е��߳�ȫ���ŵ�
        ��0��SM�ϡ�
  tid=i���̵߳�index�������ѭ���ｫ������Ҫִ�е��߳�ȫ���ŵ���0��SM�ϣ����̵߳�index��Ϊѭ������i��
  threads_left=m_kernel->threads_per_cta()-i���ڵ�ǰ�߳�֮��ʣ���̵߳�������
  num_threads=m_kernel->threads_per_cta()��
  hw_cta_id=0����������ִ�й���ģ�⣬���CTA��index����Ҫ��Ӳ��CTA��index����ʼ��Ϊ0��
  hw_warp_id=i/m_warp_size������ȫ����һ��CTA�ڣ�Ӳ����warp��index��Ϊi/m_warp_size��
*/
unsigned exec_shader_core_ctx::sim_init_thread(
    kernel_info_t &kernel, ptx_thread_info **thread_info, int sid, unsigned tid,
    unsigned threads_left, unsigned num_threads, core_t *core,
    unsigned hw_cta_id, unsigned hw_warp_id, gpgpu_t *gpu) {
  return ptx_sim_init_thread(kernel, thread_info, sid, tid, threads_left,
                             num_threads, core, hw_cta_id, hw_warp_id, gpu);
}

/*
SM����kernel��һ���߳̿顣
*/
void shader_core_ctx::issue_block2core(kernel_info_t &kernel) {
  //֧��SM�ϵĲ����ںˣ�Ĭ��Ϊ���ã�����V100�����н��á�
  if (!m_config->gpgpu_concurrent_kernel_sm)
    //��������ÿSM��CTA����kernel_max_cta_per_shader�����һ�Ҫ�����߳̿���߳������Ƿ��ܶ�warp 
    //sizeȡģ���㣬������paddedÿCTA���߳�����kernel_padded_threads_per_cta��
    set_max_cta(kernel);
  else
    assert(occupy_shader_resource_1block(kernel, true));

  //ִ��m_num_cores_running++��m_num_cores_running��һ��Core������������һ��ȫ�ֱ��������ڸ�����
  //�����е�ǰ�ں˺�����Shader Core����������ȷ��GPU�Ƿ���Խ����µ�����
  kernel.inc_running();

  // find a free CTA context
  //free_cta_hw_idָû�б�ռ�õ�CTA ID��������һ�����е�CTA��
  unsigned free_cta_hw_id = (unsigned)-1;

  unsigned max_cta_per_core;
  //֧��SM�ϵĲ����ںˣ�Ĭ��Ϊ���ã�����V100�����н��á�
  if (!m_config->gpgpu_concurrent_kernel_sm)
    //kernel_max_cta_per_shader�Ǽ���ó�������ÿSM��CTA������
    max_cta_per_core = kernel_max_cta_per_shader;
  else
    max_cta_per_core = m_config->max_cta_per_core;
  //�Ե���SIMT Core�е�����CTAѭ�������Ҵ��ڷǻ�Ծ״̬��CTA�������Ÿ�ֵ��free_cta_hw_id��
  for (unsigned i = 0; i < max_cta_per_core; i++) {
    //m_cta_status[i] == 0�����i��CTA�ڻ�Ծ���߳�����Ϊ0������i��CTA�Ѿ�����Ծ�ˣ��Ѿ������ˡ�
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
  //��ʵ��һ�����ظ��ģ���Ϊkernel_padded_threads_per_cta�ļ��������padded_cta_sizeһ�¡�
  int padded_cta_size = cta_size;
  if (cta_size % m_config->warp_size)
    padded_cta_size =
        ((cta_size / m_config->warp_size) + 1) * (m_config->warp_size);

  //��ʼ�̺߳źͽ����̺߳š�
  unsigned int start_thread, end_thread;

  if (!m_config->gpgpu_concurrent_kernel_sm) {
    //��ʼ�̺߳š�
    start_thread = free_cta_hw_id * padded_cta_size;
    //�����̺߳š�
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
  //������ѡӲ���̺߳�warp�����ĵ�΢�ܹ�״̬��
  reinit(start_thread, end_thread, false);

  // initalize scalar threads and determine which hardware warps they are
  // allocated to bind functional simulation state of threads to hardware
  // resources (simulation)
  //��ʼ�������̲߳�ȷ�����Ƿ������ЩӲ��warp�����ܷ���״̬�󶨵�Ӳ����Դ�����棩��
  warp_set_t warps;
  //nthreads_in_block��thread block�е��߳�����
  unsigned nthreads_in_block = 0;
  //����һ��kernel����ں�����m_kernel_entry�� function_info ����
  function_info *kernel_func_info = kernel.entry();
  //���ű�
  symbol_table *symtab = kernel_func_info->get_symtab();
  //��ȡ��һ��Ҫ�����CTA��������CTA��ȫ��������CUDA���ģ���е��߳̿��������ƣ���ID�㷨���£�
  //  ID = m_next_cta.x + m_grid_dim.x * m_next_cta.y +
  //       m_grid_dim.x * m_grid_dim.y * m_next_cta.z;
  unsigned ctaid = kernel.get_next_cta_id_single();
  checkpoint *g_checkpoint = new checkpoint();
  //��������free_cta_hw_id��CTA����ʼ�̺߳ŵ������̺߳�ѭ����
  for (unsigned i = start_thread; i < end_thread; i++) {
    //�����̵߳�CTA IDΪfree_cta_hw_id��
    m_threadState[i].m_cta_id = free_cta_hw_id;
    //warp_id���̺߳ų���32��
    unsigned warp_id = i / m_config->warp_size;
    //nthreads_in_block��thread block�е��߳�����sim_init_thread�����᷵���ܷ��ʼ����i���̡߳�
    nthreads_in_block += sim_init_thread(
        kernel, &m_thread[i], m_sid, i, cta_size - (i - start_thread),
        m_config->n_thread_per_shader, this, free_cta_hw_id, warp_id,
        m_cluster->get_gpu());
    //���õ�i���߳�Ϊ��Ծ״̬��
    m_threadState[i].m_active = true;
    // load thread local memory and register file
    //��V100�����У�m_gpu->resume_optionĬ������Ϊ0��
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
    //��ʼ�������̲߳�ȷ�����Ƿ������ЩӲ��warp�����ܷ���״̬�󶨵�Ӳ����Դ�����棩�������������
    //��ЩwarpΪ��Ծ��
    warps.set(warp_id);
  }
  assert(nthreads_in_block > 0 &&
         nthreads_in_block <=
             m_config->n_thread_per_shader);  // should be at least one, but
                                              // less than max
  //m_cta_status[i] == 0�����i��CTA�ڵĻ�Ծ�߳�������
  m_cta_status[free_cta_hw_id] = nthreads_in_block;

  //��V100�����У�m_gpu->resume_optionĬ������Ϊ0��
  if (m_gpu->resume_option == 1 && kernel.get_uid() == m_gpu->resume_kernel &&
      ctaid >= m_gpu->resume_CTA && ctaid < m_gpu->checkpoint_CTA_t) {
    char f1name[2048];
    snprintf(f1name, 2048, "checkpoint_files/shared_mem_%d.txt", ctaid);

    g_checkpoint->load_global_mem(m_thread[start_thread]->m_shared_mem, f1name);
  }
  // now that we know which warps are used in this CTA, we can allocate
  // resources for use in CTA-wide barrier operations
  //��Ȼ����֪���������CTA��ʹ������Щwarp�����ǾͿ��Է�������CTA�������������Դ��
  m_barriers.allocate_barrier(free_cta_hw_id, warps);

  // initialize the SIMT stacks and fetch hardware
  //��ʼ��SIMT��ջ�Լ�ԤȡӲ�����Ե�cta_id��CTA�У���start_thread��end_thread���߳�����������
  //warp���г�ʼ����
  init_warps(free_cta_hw_id, start_thread, end_thread, ctaid, cta_size, kernel);
  //��Ծ��CTA������1��
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
Find next clock domain and increment its time. �ҵ���һ��ʱ�����ƽ�����ʱ�䡣
*/ 
int gpgpu_sim::next_clock_domain(void) {
  //�ĸ�ʱ����ͬ����ÿ����Ҫ��ѡ����С����ʱ���һ��ʱ������н��ĵ��ƽ���
  //���磬�ĸ�ʱ�ӵĳ�ʼ״̬�ֱ�Ϊ��
  //    core_time = 0��icnt_time = 0��dram_time = 0��l2_time = 0
  //��ʼ״̬��smallest=0��mask������Ҫ���£�ʱ��������ǰ�ƽ�����ʱ����ı�ǡ�
  //    #define CORE 0b0001
  //    #define L2 0b0010
  //    #define DRAM 0b0100
  //    #define ICNT 0b1000
  //ʱ�������ã�-gpgpu_clock_domains 1447.0:1447.0:1447.0:850.0
  //ͨ��if��
  //    core_time+=1/(1447*1000000)
  //    icnt_time+=1/(1447*1000000)
  //    l2_time+=1/(1447*1000000)
  //    dram_time+=1/(850*1000000)
  //    mask=0b1111��������ʱ����Ҫ���¡�
  //��һ�����е�ǰ������������ʱ����ʱ��
  //    smallest=core_time=icnt_time=l2_time=1/(1447*1000000)
  //ͨ��if��
  //    core_time+=1/(1447*1000000)
  //    icnt_time+=1/(1447*1000000)
  //    l2_time+=1/(1447*1000000)
  //    dram_time > 1/(1447*1000000)���ʵ�ǰ������
  //    mask=0b1011������dram_time������ʱ����Ҫ���¡�
  double smallest = min3(core_time, icnt_time, dram_time);
  int mask = 0x00;
  //��ʼ״̬�£�smallestΪ0���ڡ�
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
GPU�����߳̿顣
*/
void gpgpu_sim::issue_block2core() {
  unsigned last_issued = m_last_cluster_issue;
  //�����ϣ�����SIMT Core��Ⱥ���������������������ļ�Ⱥ��ʼ������ÿ����Ⱥ������issue_block2core��
  //�����ظü�Ⱥ������߳̿������⽫���ӵ���Աgpgpu_sim::m_total_cta_launched��
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
    unsigned idx = (i + last_issued + 1) % m_shader_config->n_simt_clusters;
    //m_cluster[idx]�����߳̿飬���ط�����߳̿�����
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
�����߳̿���Ҫһ���ֲ���ã�������ʾ��
  gpgpu_sim::cycle()
    gpgpu_sim::issue_block2core()
      simt_core_cluster::issue_block2core()
        shader_core_ctx::issue_block2core()
          trace_shader_core_ctx::init_warps()

��ÿ��ģ�������У��������gpgpu_sim::cycle()���˺����������κβ�����

gpgpu_sim::cycle()����Ϊgpgpu-sim�е�������ϵ�ṹ�����ʱ�������ڴ�����Ķ��С�DRAMͨ���Ͷ������档
1. �����:
       icnt_push(m_shader_config->mem2device(i), mf->get_tpc(), mf, response_size);
       m_memory_partition_unit[i]->pop();
   ���ڴ�������ڴ������L2->icnt����ע�뵽���������С�����tomory_partition_unit::pop()����ִ��ԭ
   ��ָ�������������ᶪ�����ڴ��������Ŀ��ָʾ�ڴ��������ɶԴ�����ķ���
2. ��memory_partition_unit::dram_cycle()�ĵ��ý��ڴ������L2->dram�����ƶ���dramͨ����dramͨ����
   ����dram->L2���У���ѭ��оƬ��GDDR3 dram�ڴ档
3. ��memory_partition_unit::push()�ĵ��ôӻ��������е������ݰ��������䴫�ݵ��ڴ�����������������
   �յ��������֪ͨ��������ʱ�ֱ�����͵�icnt->L2���У�����������ʱ����͵���С�ӳ�ROP���С���ע�⣬
   ��icnt->L2��ROP���е����Ͳ������ܵ�memory_partition_unit::full()�����ж����icnt->L2-���д�С
   �����ơ�
4. ��memory_partition_unit::cache_cycle()�ĵ���Ϊ�����������ʱ����������������Ƴ��������档��һ
   ��������memory_partition_unit::cache_cycle()���ڲ��ṹ��
*/
void gpgpu_sim::cycle() {
  //��һ����Ҫ�ƽ���ʱ�����ʱ�������롣��Ϊÿ��ʱ�������첽�ģ�����ͬʱ���µġ��ĸ�ʱ�ӵ�mask���Ϊ��
  //    #define CORE 0b0001
  //    #define L2 0b0010
  //    #define DRAM 0b0100
  //    #define ICNT 0b1000
  //���ص�clock_mask�����0b1011����������CORE��L2��ICNT����ʱ����
  int clock_mask = next_clock_domain();
  //SIMT Coreʱ������¡�
  if (clock_mask & CORE) {
    // shader core loading (pop from ICNT into core) follows CORE clock.
    //�����е�SIMT Core��Ⱥѭ����m_cluster[i]������һ����Ⱥ��Shader Core���أ���ICNT������Core��
    //��ѭ����ʱ�ӡ�
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
      //simt_core_cluster::icnt_cycle()�������ڴ�����ӻ�����������simt���ļ�Ⱥ����ӦFIFO��������
      //FIFO�������󣬲������Ƿ��͵���Ӧ�ں˵�ָ����LDST��Ԫ��ÿ��SIMT Core��Ⱥ����һ����ӦFIFO��
      //���ڱ���ӻ������緢�������ݰ������ݰ�������SIMT Core��ָ��棨�������Ϊָ���ȡδ����
      //�ṩ������ڴ���Ӧ�������ڴ���ˮ�ߣ�memory pipeline��LDST ��Ԫ�������ݰ����Ƚ��ȳ���ʽ�ó���
      //���SIMT Core�޷�����FIFOͷ�������ݰ�������ӦFIFO��ֹͣ��Ϊ����LDST��Ԫ�������ڴ�����ÿ��
      //SIMT Core�����Լ���ע��˿ڽ��뻥�����硣���ǣ�ע��˿ڻ�������SIMT Core��Ⱥ����SIMT Core��
      //��
      //icnt_cycle()ʵ�ֵ���Ҫ�������£�
      //���ȣ�����Ҫ�ж�����SIMT Core��Ⱥ��m_response_fifo�Ƿ�Ϊ�գ������Ϊ�գ���֤����һ���ڣ�����
      //�Ƚ�SIMT Core��Ⱥ��m_response_fifo�е����ݰ�mf���뵽SIMT Core��L1ָ���m_L1I����LD/ST��
      //Ԫ�У�ʵ���ϣ������п���������TITAN V���õĵ���SIMT Core��Ⱥ���ж��SM���������������������
      //����V100����ÿ��SIMT Core��Ⱥ��ֻ�е���SM���������������ΪSIMT Core��Ⱥ���ǵ���SM������Σ�
      //�ж������������������SIMT Core��Ⱥ��m_response_fifo�Ƿ��пռ�����µ����Ի�����������ݰ���
      //�����Ϊ������֤����һ���ڣ�����������Ի�����������ݰ�����ע������������Ⱥ�˳��ʵ��Ӳ��ִ
      //�е�ʱ������������ͬ�����У��������Ǳ����Ƚ�SIMT Core��Ⱥ��m_response_fifo�е����ݰ�mf����
      //SIMT Core��Ȼ���ٳ��Խ����µ����Ի�����������ݰ���
      m_cluster[i]->icnt_cycle();
  }
  unsigned partiton_replys_in_parallel_per_cycle = 0;
  
  //����ICNTʱ������ǰ�ƽ�һ�ġ�
  if (clock_mask & ICNT) {
    // pop from memory controller to interconnect
    //�Ӵ洢�������������絯����gpgpu_n_memΪ�����е��ڴ��������DRAM Channel������������Ϊ��
    //  option_parser_register(opp, "-gpgpu_n_mem", OPT_UINT32, &m_n_mem,
    //                         "number of memory modules (e.g. memory controllers) in gpu",
    //                         "8");
    //��V100�����У���32���ڴ��������DRAM Channel����ͬʱÿ���ڴ��������Ϊ�������ӷ�������ˣ�
    //m_n_sub_partition_per_memory_channelΪ2������Ϊ��
    //  option_parser_register(opp, "-gpgpu_n_sub_partition_per_mchannel", OPT_UINT32,
    //                         &m_n_sub_partition_per_memory_channel,
    //                         "number of memory subpartition in each memory module",
    //                         "1");
    //��m_n_mem_sub_partition = m_n_mem * m_n_sub_partition_per_memory_channel������ȫ���ڴ���
    //������������
    for (unsigned i = 0; i < m_memory_config->m_n_mem_sub_partition; i++) {
      //������Ҫ���ֲ��еĵ��������ڴ��������ϸϸ��ͼ��memory_sub_partition���������Ƴ����ݰ��Ľ�
      //�ھ���L2_icnt_queue->ICNT����������ǽ��ڴ��ӷ����е�m_L2_icnt_queue���ж��������ݰ�������
      //���ء������Ƕ������ڴ��ӷ���ѭ�����������ڴ��ӷ�����m_L2_icnt_queue���ж��������ݰ�����������
      //������ݰ���������L2_WRBK_ACC��L1_WRBK_ACC���򷵻ؿ����ݰ�����֮�����������ݰ���
      //��V100�У�L1 cache��m_write_policyΪWRITE_THROUGH��ʵ����L1_WRBK_ACCҲ�����õ���
      //��V100�У���L2 cacheд������ʱ����ȡlazy_fetch_on_read���ԣ����ҵ�һ��cache block
      //���ʱ��������cache block�Ǳ�MODIFIED������Ҫ�����cache blockд�ص���һ���洢��
      //��˻����L2_WRBK_ACC���ʣ�������ʾ���Ϊ��д�ر������MODIFIED cache block��

      //����ʵ���� m_memory_sub_partition[i]->top() ��ִ�� m_L2_icnt_queue->top()�����ǵ���
      //   mf->get_access_type() == L2_WRBK_ACC ��
      //   mf->get_access_type() == L1_WRBK_ACC
      //ʱ�����Ὣ����ת����SM�ˣ���Ϊ��������ֻ��д�ء�
      mem_fetch *mf = m_memory_sub_partition[i]->top();
      if (mf) {
        // The packet size varies depending on the type of request:
        // - For read request and atomic request, the packet contains the data
        // - For write-ack, the packet only has control metadata
        //���ݰ���С���������Ͷ��죺
        // - ���ڶ�ȡ�����ԭ���������ݰ��������ݣ�
        // - ����дȷ�ϣ����ݰ�ֻ�п���Ԫ���ݡ�
        unsigned response_size =
            mf->get_is_write() ? mf->get_ctrl_size() : mf->size();
        //�ڴ��ڴ��ӷ����������絯��ʱ����������������п��еĻ����������ڴ��������뻥�����硣����
        //һ�����������еĻ�������ռ�����ͻ�ֹͣ���͡����ڻ������绺�����Ĵ�С������ɵ�ͣ��ʱ��������
        //��gpu_stall_icnt2sh��������������
        //icnt_has_buffer���жϻ��������Ƿ��п��е����뻺�������������deviceID���豸�µ����ݰ���
        if (::icnt_has_buffer(m_shader_config->mem2device(i), response_size)) {
          // if (!mf->get_is_write())
          mf->set_return_timestamp(gpu_sim_cycle + gpu_tot_sim_cycle);
          mf->set_status(IN_ICNT_TO_SHADER, gpu_sim_cycle + gpu_tot_sim_cycle);
          //icnt_push�����ݰ�ѹ�뻥���������뻺������
          ::icnt_push(m_shader_config->mem2device(i), mf->get_tpc(), mf,
                      response_size);
          //m_memory_sub_partition[i]�ж��и��Ե�m_icnt_L2_queue���У�����ICNT��SM���ݰ��Ľӿڡ�
          m_memory_sub_partition[i]->pop();
          partiton_replys_in_parallel_per_cycle++;
        } else {
          gpu_stall_icnt2sh++;
        }
      } else {
        //����ڴ��ӷ�����m_L2_icnt_queue���ж��������ݰ���Ч����Ҳ�����ʧЧ�����ݰ�Ҳ������
        m_memory_sub_partition[i]->pop();
      }
    }
  }
  partiton_replys_in_parallel += partiton_replys_in_parallel_per_cycle;

  if (clock_mask & DRAM) {
    //��ÿ��DRAMͨ��ѭ��������memory_partition_unit::dram_cycle()���������ڴ������L2->dram������
    //����DRAM Channel��DRAM Channel��dram->L2���У���ѭ��Ƭ��GDDR3 DRAM�ڴ档
    for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
      //��V100�У�simple_dram_model������Ϊ0��
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

  //����L2ʱ����
  if (clock_mask & L2) {
    m_power_stats->pwr_mem_stat->l2_cache_stats[CURRENT_STAT_IDX].clear();
    //gpgpu_n_memΪ�����е��ڴ��������DRAM Channel������������Ϊ��
    //  option_parser_register(opp, "-gpgpu_n_mem", OPT_UINT32, &m_n_mem,
    //                         "number of memory modules (e.g. memory controllers) in gpu",
    //                         "8");
    //��V100�����У���32���ڴ��������DRAM Channel����ͬʱÿ���ڴ��������Ϊ�������ӷ�������ˣ�
    //m_n_sub_partition_per_memory_channelΪ2������Ϊ��
    //  option_parser_register(opp, "-gpgpu_n_sub_partition_per_mchannel", OPT_UINT32,
    //                         &m_n_sub_partition_per_memory_channel,
    //                         "number of memory subpartition in each memory module",
    //                         "1");
    //��m_n_mem_sub_partition = m_n_mem * m_n_sub_partition_per_memory_channel������ȫ���ڴ���
    //����������������������ڴ����ѭ����
    for (unsigned i = 0; i < m_memory_config->m_n_mem_sub_partition; i++) {
      // move memory request from interconnect into memory partition (if not
      // backed up) Note:This needs to be called in DRAM clock domain if there
      // is no L2 cache in the system In the worst case, we may need to push
      // SECTOR_CHUNCK_SIZE requests, so ensure you have enough buffer for them
      //���ڴ�����ӻ����ƶ����ڴ���������û�б��ݣ�ע�⣺���ϵͳ��û�ж������棬��
      //��Ҫ��DRAMʱ�����е��á����������£����ǿ�����Ҫ����SECTOR_CHUNCK_SIZE��С
      //�����������Ҫȷ�����㹻�Ļ��������������ǡ�
      //m_memory_sub_partition[i]->full�Ķ���Ϊ��
      //    bool memory_sub_partition::full(unsigned size) const {
      //      return m_icnt_L2_queue->is_avilable_size(size);
      //    }
      //��������icnt������L2_queueʱ��m_icnt_L2_queueû��SECTOR_CHUNCK_SIZE��С�Ŀռ�
      //���Ա���������Ϣ����˻��������ӵ�����DRAM��ͣ�͡�
      
      //������Ҫ���ֲ��еĵ��������ڴ��������ϸϸ��ͼ������������memory_sub_partition�Ƴ����ݰ��Ľ�
      //�ھ���ICNT->icnt_L2_queue������������ж��ڴ��ӷ����е�m_icnt_L2_queue�����Ƿ���Է���size
      //��С�����ݣ����Է��·���False���Ų��·���True��SECTOR_CHUNCK_SIZE=4�����m_icnt_L2_queue��
      //�зŲ���SECTOR_CHUNCK_SIZE=4��С�����ݣ����������[icnt_L2_queue��ӵ��]���DRAM��ͣ�ͣ���
      //gpu_stall_dramfull��¼ͣ�͵Ĵ�����
      if (m_memory_sub_partition[i]->full(SECTOR_CHUNCK_SIZE)) {
        gpu_stall_dramfull++;
      } else {
        //���m_memory_sub_partition[i]->full(SECTOR_CHUNCK_SIZE)����False������m_L2_icnt_queue
        //���зŵ���SECTOR_CHUNCK_SIZE=4��С�����ݣ���˿��Խ����ݶ�����mf�ӻ����������뵽�ڴ��ӷ���
        //������ȡ���ݴ���
        //icnt_pop()�����ݰ������������������������
        mem_fetch *mf = (mem_fetch *)icnt_pop(m_shader_config->mem2device(i));
        //�����ݶ�����m_req�ӻ����������뵽�ڴ��ӷ��������к���ȡ���ݴ���
        m_memory_sub_partition[i]->push(mf, gpu_sim_cycle + gpu_tot_sim_cycle);
        //���mf��Ϊ�գ���˵������������L2_queue����˵�ǰ�洢���������������������
        //partiton_reqs_in_parallel_per_cycle��ʾ��ǰʱ�����������д洢�����Ĳ�������
        //�������ܸ��������partiton_reqs_in_parallel_per_cycle++��ʾ��ǰʱ����������
        //������L2_queue���ܸ�����1��
        if (mf) partiton_reqs_in_parallel_per_cycle++;
      }
      //�Զ�������Bank���н����ƽ�����������������Ƴ��������档????
      m_memory_sub_partition[i]->cache_cycle(gpu_sim_cycle + gpu_tot_sim_cycle);
      if (m_config.g_power_simulation_enabled) {
        m_memory_sub_partition[i]->accumulate_L2cache_stats(
            m_power_stats->pwr_mem_stat->l2_cache_stats[CURRENT_STAT_IDX]);
      }
    }
  }
  //��ģ���������ĵ�һ��ʱ�����ڿ�ʼ��partiton_reqs_in_parallel�͸���ÿһ�ĵ��ڴ������
  //��������ʼ������partiton_reqs_in_parallel��ʾ��ģ���������ĵ�һ��ʱ�����ڿ�ʼ����
  //�д洢������������������m_memory_sub_partition��m_icnt_L2_queue���ܸ�����
  partiton_reqs_in_parallel += partiton_reqs_in_parallel_per_cycle;
  if (partiton_reqs_in_parallel_per_cycle > 0) {
    partiton_reqs_in_parallel_util += partiton_reqs_in_parallel_per_cycle;
    gpu_sim_cycle_parition_util++;
  }

  if (clock_mask & ICNT) {
    //��������ִ��·��һ�ġ�
    icnt_transfer();
  }
  //����ƽ�����SIMT Coreʱ����
  if (clock_mask & CORE) {
    // L1 cache + shader core pipeline stages
    m_power_stats->pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].clear();
    //��GPU�����е�SIMT Core��Ⱥ����ѭ��������ÿ����Ⱥ��״̬��
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
      //���get_not_completed()����1���������SIMT Core��δ��ɣ����get_more_cta_left()Ϊ1��
      //�������SIMT Core����ʣ���CTA��Ҫȡִ�С�m_cluster[i]->get_not_completed()���ص�i��
      //SIMT Core��Ⱥ����δ��ɵ��̸߳�����get_more_cta_left()���ڼ�鵱ǰ�Ƿ��и����CTA��
      //Compute Thread Array����Ҫִ�С�����鵱ǰ��Ծ��CTA�������������Ƿ��и���CTA��Ҫִ�С�
      //����ѴﵽGPUģ����������CTA������hit_max_cta_count()�жϣ�����û��ʣ���CTA������
      //False�����ĳ��m_running_kernels�������kernel�ǿգ���kernel->no_more_ctas_to_run()
      //Ϊfalse��kernel�Լ������ж���CTAִ�У��򷵻�True��no_more_ctas_to_run()����ָʾ��ǰû
      //�и����CTA��Compute Thread Array����Ҫִ�С�
      //������Խ�m_cluster[i]��ִ��״̬��Ϊ���ࣺ
      //    1. shader_core_ctx::init_warps�г�ʼ��warpʱ��������m_not_completed+=n_active��
      //       �������get_not_completed()����m_not_completed��ֵʵ�����Ƿ��ص����Ѿ���ʼ����
      //       ����warp��������CTA������δ��ɵ��߳�����������δ��ʼ����warp��CTA������û�м�¼
      //       �ġ�
      //    2. ��˵ڶ�����Ҫ�ж��Ƿ�����δ��ʼ����warp��CTA����Ҫ����ִ�У�ֻ��1��2��������ͬʱ
      //       ���㣬�ſ��Զ϶���ǰSIMT Core��Ⱥ�ϻ���Ҫ��ǰ�ƽ�һ�ġ�
      if (m_cluster[i]->get_not_completed() || get_more_cta_left()) {
        //������simt_core_cluster::core_cycle()ʱ�����������������SM�ں˵�ѭ������ʵ��ѭ����
        //��SIMT Core��ģ��˳�������ڱ�ʱ�������ڣ��Ǵ�m_core_sim_order.begin()��ʼ���ȣ���
        //��Ϊ��ʵ����ѯ���ȣ���begin()λ���ƶ�����ĩβ�������´ξ��Ǵ�begin+1λ�õ�SIMT Core��
        //ʼ���ȡ�
        m_cluster[i]->core_cycle();
        //���ӻ�Ծ��SM������get_n_active_sms()����SIMT Core��Ⱥ�еĻ�ԾSM��������active_sms��
        //SIMT Core��Ⱥ�еĻ�ԾSM��������get_n_active_sms()���ÿ����Ⱥ�ڲ���SIMT Core�����ж�
        //���Ƿ���active()����V100�����У�ÿ����Ⱥ�ڲ�����1��SIMT Core��
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

    //debug��
    if (g_single_step &&
        ((gpu_sim_cycle + gpu_tot_sim_cycle) >= g_single_step)) {
      raise(SIGTRAP);  // Debug breakpoint
    }
    //��Ҫע�⣬gpu_sim_cycle����COREʱ������ǰ�ƽ�һ��ʱ�Ÿ��£����gpu_sim_cycle��ʾCOREʱ��
    //��ĵ�ǰִ��������
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

    //GPU�����߳̿顣
    issue_block2core();
    //�ú������ڼ����ں��ӳ�kernel latency�����ٴӷ����ں�����ں����ִ�е�ʱ�䣨������������
    //�е��ں˵��ӳټ�1�����ú�����������ģ��GPU�ں˵����ܺͷ�����������ܡ�m_kernel_TB_latency
    //��ʾÿһ���߳̿�������ӳ�֮�ͼ���kernel�������ӳ٣����ӷ����߳̿鵽�����ִ�е�ʱ�䡣
    //m_kernel_TB_latency��GPGPU-Sim�����ڱ�ʾ�ں�����ʱ��ı���������ʾ���ں��������ں����ִ
    //������Ҫ��ʱ�䡣
    //decrement_kernel_latency�����Ķ���Ϊ��
    //   void gpgpu_sim::decrement_kernel_latency() {
    //     //�����е��������е��ں˺������ں��ӳ�kernel latency��1��
    //     for (unsigned n = 0; n < m_running_kernels.size(); n++) {
    //       if (m_running_kernels[n] && m_running_kernels[n]->m_kernel_TB_latency)
    //         m_running_kernels[n]->m_kernel_TB_latency--;
    //     }
    //   }
    //�����Ǽ�⵱ǰ�����Ѿ����䵽SM�ϵ�kernel���߳̿�������ӳ�֮�ͼ���kernel�������ӳ��Ƿ��Ѿ�
    //���㣬�������SM�ſ���ȡָ���䡣
    decrement_kernel_latency();

    // Depending on configuration, invalidate the caches once all of threads are
    // completed.
    //��־�����̶߳��Ѿ�������
    int all_threads_complete = 1;
    //��V100�����У�m_config.gpgpu_flush_l1_cache������Ϊ1��
    if (m_config.gpgpu_flush_l1_cache) {
      //�����е�SIMT��Ⱥѭ����
      for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
        //���m_cluster[i]->get_not_completed()Ϊ0���������SIMT Core��Ⱥ�е������̶߳��Ѿ���
        //�ɣ���˿��Խ����SIMT Core��Ⱥ������SM��L1ָ�������ݻ������ʧЧ������
        if (m_cluster[i]->get_not_completed() == 0)
          m_cluster[i]->cache_invalidate();
        else
          all_threads_complete = 0;
      }
    }

    //��V100�����У�m_config.gpgpu_flush_l2_cache������Ϊ0��
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

    //������һЩ������ͳ�����ݡ�
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
cuda-sim.cc���Ѿ�ʵ���˹����Ե� memcpy_to_gpu() ����������ʵ�ֵ�������ģ���е� perf_memcpy_to_gpu()
��������������ͬ�������ݿ�����GPU���Դ档
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
