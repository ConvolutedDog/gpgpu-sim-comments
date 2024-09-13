// Copyright (c) 2009-2011, Tor M. Aamodt
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The University of British Columbia nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
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

#ifndef CUDASIM_H_INCLUDED
#define CUDASIM_H_INCLUDED

#include <stdlib.h>
#include <map>
#include <string>
#include <vector>
#include "../abstract_hardware_model.h"
#include "../gpgpu-sim/shader.h"
#include "ptx_sim.h"

class gpgpu_context;
class memory_space;
class function_info;
class symbol_table;

extern const char *g_gpgpusim_version_string;
extern int g_debug_execution;

extern void print_splash();

extern void ptxinfo_opencl_addinfo(
    std::map<std::string, function_info *> &kernels);
unsigned ptx_sim_init_thread(kernel_info_t &kernel,
                             class ptx_thread_info **thread_info, int sid,
                             unsigned tid, unsigned threads_left,
                             unsigned num_threads, class core_t *core,
                             unsigned hw_cta_id, unsigned hw_warp_id,
                             gpgpu_t *gpu,
                             bool functionalSimulationMode = false);
const struct gpgpu_ptx_sim_info *ptx_sim_kernel_info(
    const class function_info *kernel);

/*!
 * This class functionally executes a kernel. It uses the basic data structures
 * and procedures in core_t
 * functionalCoreSim 类在功能上执行内核函数。它使用core_t中的基本数据结构和过程。
 */
class functionalCoreSim : public core_t {
 public:
  //构造函数。
  functionalCoreSim(kernel_info_t *kernel, gpgpu_sim *g, unsigned warp_size)
      : core_t(g, kernel, warp_size, kernel->threads_per_cta()) {
    //warp屏障指示器，即每一个warp都有一个指示位，标志该warp是否进入屏障指令执行阶段。
    //m_warpAtBarrier在构造函数中被定义为一个具有 warp总数 长度的bool变量。
    //m_warp_count是一个CTA中的warp总数。
    m_warpAtBarrier = new bool[m_warp_count];
    //每个warp活跃线程数。m_liveThreadCount在构造函数中被定义为一个具有[warp总数]长度的unsigned列表。
    //m_warp_count是一个CTA中的warp总数。
    m_liveThreadCount = new unsigned[m_warp_count];
  }
  //析构函数。
  virtual ~functionalCoreSim() {
    //退出执行所有的warp。m_warp_count是一个CTA中的warp总数。参数0代表的 warp_id 在这里貌似没什么用。
    warp_exit(0);
    delete[] m_liveThreadCount;
    delete[] m_warpAtBarrier;
  }
  //! executes all warps till completion
  //执行所有的 warp。
  void execute(int inst_count, unsigned ctaid_cp);
  //退出执行所有的warp。m_warp_count是一个CTA中的warp总数。m_warp_size是一个warp内有多少线程。warp_id
  //在这里貌似没什么用。
  virtual void warp_exit(unsigned warp_id);
  //warp_waiting_at_barrier 函数用于检测是否一个warp正在屏障指令处等待下一步执行。即当 m_warpAtBarrier
  //[warp_id] = 1 （warp_id所标识的warp正处于屏障指令阶段） 或者 !(m_liveThreadCount[warp_id] > 0)
  //（warp_id所标识的warp此刻没有活跃的线程）时，则表明warp正在屏障指令处等待下一步执行。
  virtual bool warp_waiting_at_barrier(unsigned warp_id) const {
    return (m_warpAtBarrier[warp_id] || !(m_liveThreadCount[warp_id] > 0));
  }

 private:
  //executeWarp函数的功能为执行一个warp。
  //functionalCoreSim::executeWarp函数在functionalCoreSim::execute中的调用为：
  //    executeWarp(i, allAtBarrier, someOneLive);
  //三个参数分别为 warp_id、allAtBarrier、someOneLive。
  //其中，i是一个CTA内的warp的编号，即warp_id，一个CTA内的warp从0开始编号。
  //allAtBarrier初始设置为True，即初始状态默认“不是所有的warp都处于屏障指令状态”。
  //someOneLive初始设置为False，即初始状态默认“所有的warp都暂时没有活跃的指令”。
  void executeWarp(unsigned, bool &, bool &);
  // initializes threads in the CTA block which we are executing
  //执行CTA的初始化。初始化 ctaid_cp 标识的CTA中的所有线程。
  void initializeCTA(unsigned ctaid_cp);
  virtual void checkExecutionStatusAndUpdate(warp_inst_t &inst, unsigned t,
                                             unsigned tid) {
    if (m_thread[tid] == NULL || m_thread[tid]->is_done()) {
      m_liveThreadCount[tid / m_warp_size]--;
    }
  }

  // lunches the stack and set the threads count
  //根据 warpId 创建该warp。
  void createWarp(unsigned warpId);

  // each warp live thread count and barrier indicator
  //每个warp活跃线程数。m_liveThreadCount在构造函数中被定义为一个具有[warp总数]长度的unsigned列表。
  unsigned *m_liveThreadCount;
  //warp屏障指示器，即每一个warp都有一个指示位，标志该warp是否进入屏障指令执行阶段。m_warpAtBarrier
  //在构造函数中被定义为一个具有[warp总数]长度的bool列表。
  bool *m_warpAtBarrier;
};

#define RECONVERGE_RETURN_PC ((address_type)-2)
#define NO_BRANCH_DIVERGENCE ((address_type)-1)
address_type get_return_pc(void *thd);
const char *get_ptxinfo_kname();
void print_ptxinfo();
void clear_ptxinfo();
struct gpgpu_ptx_sim_info get_ptxinfo();

class gpgpu_recon_t;
struct rec_pts {
  gpgpu_recon_t *s_kernel_recon_points;
  int s_num_recon;
};

class cuda_sim {
 public:
  cuda_sim(gpgpu_context *ctx) {
    g_ptx_sim_num_insn = 0;
    g_ptx_kernel_count =
        -1;  // used for classification stat collection purposes
    gpgpu_param_num_shaders = 0;
    g_cuda_launch_blocking = false;
    g_inst_classification_stat = NULL;
    g_inst_op_classification_stat = NULL;
    g_assemble_code_next_pc = 0;
    g_debug_thread_uid = 0;
    g_override_embedded_ptx = false;
    ptx_tex_regs = NULL;
    g_ptx_thread_info_delete_count = 0;
    g_ptx_thread_info_uid_next = 1;
    g_debug_pc = 0xBEEF1518;
    gpgpu_ctx = ctx;
  }
  // global variables
  char *opcode_latency_int;
  char *opcode_latency_fp;
  char *opcode_latency_dp;
  char *opcode_latency_sfu;
  char *opcode_latency_tensor;
  char *opcode_initiation_int;
  char *opcode_initiation_fp;
  char *opcode_initiation_dp;
  char *opcode_initiation_sfu;
  char *opcode_initiation_tensor;
  int cp_count;
  int cp_cta_resume;
  int g_ptxinfo_error_detected;
  unsigned g_ptx_sim_num_insn;
  char *cdp_latency_str;
  int g_ptx_kernel_count;  // used for classification stat collection purposes
  std::map<const void *, std::string>
      g_global_name_lookup;  // indexed by hostVar
  std::map<const void *, std::string>
      g_const_name_lookup;  // indexed by hostVar
  int g_ptx_sim_mode;  // if non-zero run functional simulation only (i.e., no
                       // notion of a clock cycle)
  unsigned gpgpu_param_num_shaders;
  class std::map<function_info *, rec_pts> g_rpts;
  bool g_cuda_launch_blocking;
  void **g_inst_classification_stat;
  void **g_inst_op_classification_stat;
  std::set<std::string> g_globals;
  std::set<std::string> g_constants;
  std::map<unsigned, function_info *> g_pc_to_finfo;
  int gpgpu_ptx_instruction_classification;
  unsigned cdp_latency[5];
  unsigned g_assemble_code_next_pc;
  int g_debug_thread_uid;
  bool g_override_embedded_ptx;
  std::set<unsigned long long> g_ptx_cta_info_sm_idx_used;
  ptx_reg_t *ptx_tex_regs;
  unsigned g_ptx_thread_info_delete_count;
  unsigned g_ptx_thread_info_uid_next;
  addr_t g_debug_pc;
  // backward pointer
  class gpgpu_context *gpgpu_ctx;
  // global functions
  void ptx_opcocde_latency_options(option_parser_t opp);
  void gpgpu_cuda_ptx_sim_main_func(kernel_info_t &kernel, bool openCL = false);
  int gpgpu_opencl_ptx_sim_main_func(kernel_info_t *grid);
  void init_inst_classification_stat();
  kernel_info_t *gpgpu_opencl_ptx_sim_init_grid(class function_info *entry,
                                                gpgpu_ptx_sim_arg_list_t args,
                                                struct dim3 gridDim,
                                                struct dim3 blockDim,
                                                gpgpu_t *gpu);
  void gpgpu_ptx_sim_register_global_variable(void *hostVar,
                                              const char *deviceName,
                                              size_t size);
  void gpgpu_ptx_sim_register_const_variable(void *, const char *deviceName,
                                             size_t size);
  void read_sim_environment_variables();
  void set_param_gpgpu_num_shaders(int num_shaders);
  struct rec_pts find_reconvergence_points(function_info *finfo);
  address_type get_converge_point(address_type pc);
  void gpgpu_ptx_sim_memcpy_symbol(const char *hostVar, const void *src,
                                   size_t count, size_t offset, int to,
                                   gpgpu_t *gpu);
  void ptx_print_insn(address_type pc, FILE *fp);
  std::string ptx_get_insn_str(address_type pc);
  template <int activate_level>
  bool ptx_debug_exec_dump_cond(int thd_uid, addr_t pc);
};

#endif
