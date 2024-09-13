// Copyright (c) 2009-2021, Tor M. Aamodt, Inderpreet Singh, Vijay Kandiah,
// Nikos Hardavellas, Mahmoud Khairy, Junrui Pan, Timothy G. Rogers The
// University of British Columbia, Northwestern University, Purdue University
// All rights reserved.
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

#ifndef ABSTRACT_HARDWARE_MODEL_INCLUDED
#define ABSTRACT_HARDWARE_MODEL_INCLUDED

// Forward declarations
class gpgpu_sim;
class kernel_info_t;
class gpgpu_context;

// Set a hard limit of 32 CTAs per shader [cuda only has 8]
#define MAX_CTA_PER_SHADER 32
#define MAX_BARRIERS_PER_CTA 16

// After expanding the vector input and output operands
#define MAX_INPUT_VALUES 24
#define MAX_OUTPUT_VALUES 8

enum _memory_space_t {
  undefined_space = 0,
  reg_space,
  local_space,
  shared_space,
  sstarr_space,
  param_space_unclassified,
  param_space_kernel, /* global to all threads in a kernel : read-only */
  param_space_local,  /* local to a thread : read-writable */
  const_space,
  tex_space,
  surf_space, // render surfaces 
  global_space,
  generic_space,
  instruction_space
};

#ifndef COEFF_STRUCT
#define COEFF_STRUCT

struct PowerscalingCoefficients {
  double int_coeff;
  double int_mul_coeff;
  double int_mul24_coeff;
  double int_mul32_coeff;
  double int_div_coeff;
  double fp_coeff;
  double dp_coeff;
  double fp_mul_coeff;
  double fp_div_coeff;
  double dp_mul_coeff;
  double dp_div_coeff;
  double sqrt_coeff;
  double log_coeff;
  double sin_coeff;
  double exp_coeff;
  double tensor_coeff;
  double tex_coeff;
};
#endif

enum FuncCache {
  FuncCachePreferNone = 0,
  FuncCachePreferShared = 1,
  FuncCachePreferL1 = 2
};

enum AdaptiveCache { FIXED = 0, ADAPTIVE_CACHE = 1 };

#ifdef __cplusplus

#include <stdio.h>
#include <string.h>
#include <set>

typedef unsigned long long new_addr_type;
typedef unsigned long long cudaTextureObject_t;
typedef unsigned long long address_type;
typedef unsigned long long addr_t;

// the following are operations the timing model can see
#define SPECIALIZED_UNIT_NUM 8
#define SPEC_UNIT_START_ID 100

/*
Type of operation.
操作类型。
*/
enum uarch_op_t {
  NO_OP = -1,
  ALU_OP = 1,
  SFU_OP,
  TENSOR_CORE_OP,
  DP_OP,
  SP_OP,
  INTP_OP,
  ALU_SFU_OP,
  LOAD_OP,
  TENSOR_CORE_LOAD_OP,
  TENSOR_CORE_STORE_OP,
  STORE_OP,
  BRANCH_OP,
  BARRIER_OP,
  MEMORY_BARRIER_OP,
  CALL_OPS,
  RET_OPS,
  EXIT_OPS,
  SPECIALIZED_UNIT_1_OP = SPEC_UNIT_START_ID,
  SPECIALIZED_UNIT_2_OP,
  SPECIALIZED_UNIT_3_OP,
  SPECIALIZED_UNIT_4_OP,
  SPECIALIZED_UNIT_5_OP,
  SPECIALIZED_UNIT_6_OP,
  SPECIALIZED_UNIT_7_OP,
  SPECIALIZED_UNIT_8_OP
};
typedef enum uarch_op_t op_type;

enum uarch_bar_t { NOT_BAR = -1, SYNC = 1, ARRIVE, RED };
typedef enum uarch_bar_t barrier_type;

enum uarch_red_t { NOT_RED = -1, POPC_RED = 1, AND_RED, OR_RED };
typedef enum uarch_red_t reduction_type;

enum uarch_operand_type_t { UN_OP = -1, INT_OP, FP_OP };
typedef enum uarch_operand_type_t types_of_operands;

enum special_operations_t {
  OTHER_OP,
  INT__OP,
  INT_MUL24_OP,
  INT_MUL32_OP,
  INT_MUL_OP,
  INT_DIV_OP,
  FP_MUL_OP,
  FP_DIV_OP,
  FP__OP,
  FP_SQRT_OP,
  FP_LG_OP,
  FP_SIN_OP,
  FP_EXP_OP,
  DP_MUL_OP,
  DP_DIV_OP,
  DP___OP,
  TENSOR__OP,
  TEX__OP
};

typedef enum special_operations_t
    special_ops;  // Required to identify for the power model
enum operation_pipeline_t {
  UNKOWN_OP,
  SP__OP,
  DP__OP,
  INTP__OP,
  SFU__OP,
  TENSOR_CORE__OP,
  MEM__OP,
  SPECIALIZED__OP,
};
typedef enum operation_pipeline_t operation_pipeline;
enum mem_operation_t { NOT_TEX, TEX };
typedef enum mem_operation_t mem_operation;

enum _memory_op_t { no_memory_op = 0, memory_load, memory_store };

#include <assert.h>
#include <stdlib.h>
#include <algorithm>
#include <bitset>
#include <deque>
#include <list>
#include <map>
#include <vector>

#if !defined(__VECTOR_TYPES_H__)
#include "vector_types.h"
#endif
struct dim3comp {
  bool operator()(const dim3 &a, const dim3 &b) const {
    if (a.z < b.z)
      return true;
    else if (a.y < b.y)
      return true;
    else if (a.x < b.x)
      return true;
    else
      return false;
  }
};

void increment_x_then_y_then_z(dim3 &i, const dim3 &bound);

// Jin: child kernel information for CDP
#include "stream_manager.h"
class stream_manager;
struct CUstream_st;
// extern stream_manager * g_stream_manager;
// support for pinned memories added
extern std::map<void *, void **> pinned_memory;
extern std::map<void *, size_t> pinned_memory_size;

/*
内核函数的信息。kernel_info_t 对象包含GPU网格和块维度、与内核入口点关联的 function_info 对象以
及为内核参数分配的内存。
*/
class kernel_info_t {
 public:
  //构造函数 kernel_info_t() 的定义：
  //   kernel_info_t()
  //   {
  //      m_valid=false;
  //      m_kernel_entry=NULL;
  //      m_uid=0;
  //      m_num_cores_running=0;
  //      m_param_mem=NULL;
  //   }
  kernel_info_t(dim3 gridDim, dim3 blockDim, class function_info *entry,
                unsigned long long streamID);
  kernel_info_t(
      dim3 gridDim, dim3 blockDim, class function_info *entry,
      std::map<std::string, const struct cudaArray *> nameToCudaArray,
      std::map<std::string, const struct textureInfo *> nameToTextureInfo);
  ~kernel_info_t();

  //m_num_cores_running是一个Core计数器，它是一个全局变量，用于跟踪正在运行当前内核函数的GPU Core
  //的数量，并确定GPU是否可以接受新的任务。下面的函数中：
  //  inc_running()增加一个正在运行的Core，将m_num_cores_running加1；
  //  dec_running()减少一个正在运行的Core，首先判断m_num_cores_running是否大于0，继而减1；
  //  running()返回是否有正在运行的Core，True 或 False；
  //  done()返回没有更多的CTA需要执行，m_num_cores_running的值也为零，代表所有正在执行的kernel已
  //        经全部执行完。即kernel_info_t已经完成它所有CTA的执行。
  void inc_running() { m_num_cores_running++; }
  void dec_running() {
    assert(m_num_cores_running > 0);
    m_num_cores_running--;
  }
  bool running() const { return m_num_cores_running > 0; }
  bool done() const { return no_more_ctas_to_run() && !running(); }
  //返回一个kernel的入口函数，m_kernel_entry是 function_info 对象。
  class function_info *entry() {
    return m_kernel_entry;
  }
  const class function_info *entry() const { return m_kernel_entry; }

  //返回CUDA代码中的Grid中的所有线程块的总数。
  size_t num_blocks() const {
    return m_grid_dim.x * m_grid_dim.y * m_grid_dim.z;
  }
  //返回每个线程块中的线程数量，threads_per_cta=m_block_dim.x * m_block_dim.y * m_block_dim.z
  size_t threads_per_cta() const {
    return m_block_dim.x * m_block_dim.y * m_block_dim.z;
  }
  //返回CUDA代码中的Grid的三个维度，一个dim3数据类型。
  dim3 get_grid_dim() const { return m_grid_dim; }
  //返回CTA的三个维度，一个dim3数据类型。
  dim3 get_cta_dim() const { return m_block_dim; }
  //在 cuda-sim.cc 中有用到，比如当一个CTA不满足执行条件，因此需要选择下个CTA，通过指定下一个CTA
  //的编号增加来实现，但是要考虑dim3的三维结构，超过其边界时，置零并增加下一维。下面的函数是将[要
  //执行的下一个CTA]的编号向后推一个；在推后时，由于更换了一个CTA，需要将线程ID的三维全部置零。
  void increment_cta_id() {
    increment_x_then_y_then_z(m_next_cta, m_grid_dim);
    m_next_tid.x = 0;
    m_next_tid.y = 0;
    m_next_tid.z = 0;
  }
  //获取下一个要执行的CTA其网格中的ID，一个dim3数据类型。
  dim3 get_next_cta_id() const { return m_next_cta; }
  //获取下一个要发射的CTA的索引。CTA的全局索引与CUDA编程模型中的线程块索引类似，其ID算法如下：
  //ID = m_next_cta.x + m_grid_dim.x * m_next_cta.y +
  //     m_grid_dim.x * m_grid_dim.y * m_next_cta.z;
  unsigned get_next_cta_id_single() const {
    return m_next_cta.x + m_grid_dim.x * m_next_cta.y +
           m_grid_dim.x * m_grid_dim.y * m_next_cta.z;
  }
  //m_next_cta是用于标识下一个要发射的CTA（Compute Thread Array）的变量，它的值是一个全局ID，属
  //于dim3类型，具有.x/.y/.z三个分值。GPU硬件配置的CTA的全局ID的范围为：
  //    m_next_cta.x < m_grid_dim.x &&
  //    m_next_cta.y < m_grid_dim.y &&
  //    m_next_cta.z < m_grid_dim.z
  //因此如果标识下一个要发射的CTA的全局ID的任意一维超过CUDA代码设置的Grid的对应范围，就代表内核函
  //数上已经没有CTA可执行，内核函数的所有线程块均已经执行完毕。
  bool no_more_ctas_to_run() const {
    return (m_next_cta.x >= m_grid_dim.x || m_next_cta.y >= m_grid_dim.y ||
            m_next_cta.z >= m_grid_dim.z);
  }
  //下面的函数是将[要执行的下一个线程]的编号向后推一个。
  void increment_thread_id() {
    increment_x_then_y_then_z(m_next_tid, m_block_dim);
  }
  //获取下一个要执行的线程在其线程块中的ID，一个dim3数据类型。
  dim3 get_next_thread_id_3d() const { return m_next_tid; }
  //返回下一个要执行的线程的索引。
  unsigned get_next_thread_id() const {
    return m_next_tid.x + m_block_dim.x * m_next_tid.y +
           m_block_dim.x * m_block_dim.y * m_next_tid.z;
  }
  //返回当前CTA中是否有更多的线程待执行。
  bool more_threads_in_cta() const {
    return m_next_tid.z < m_block_dim.z && m_next_tid.y < m_block_dim.y &&
           m_next_tid.x < m_block_dim.x;
  }
  //返回当前 kernel_info_t 对象的唯一标识号。
  unsigned get_uid() const { return m_uid; }
  unsigned long long get_streamID() const { return m_streamID; }
 //返回当前 kernel_info_t 内核对象的内核名。
  std::string get_name() const { return name(); }
  //返回当前 kernel_info_t 内核对象的内核名。
  std::string name() const;
  //m_active_threads的定义：
  //    std::list<class ptx_thread_info *> m_active_threads;
  //是一个列表，保存了活跃线程。下面的函数返回该活跃线程的列表。
  std::list<class ptx_thread_info *> &active_threads() {
    return m_active_threads;
  }
  //返回kernel_info_t对象的参数内存空间。
  class memory_space *get_param_memory() {
    return m_param_mem;
  }

  // The following functions access texture bindings present at the kernel's
  // launch

  const struct cudaArray *get_texarray(const std::string &texname) const {
    std::map<std::string, const struct cudaArray *>::const_iterator t =
        m_NameToCudaArray.find(texname);
    assert(t != m_NameToCudaArray.end());
    return t->second;
  }
  //纹理缓存相关，后面用到再补充。
  const struct textureInfo *get_texinfo(const std::string &texname) const {
    std::map<std::string, const struct textureInfo *>::const_iterator t =
        m_NameToTextureInfo.find(texname);
    assert(t != m_NameToTextureInfo.end());
    return t->second;
  }

 private:
  kernel_info_t(const kernel_info_t &);   // disable copy constructor
  void operator=(const kernel_info_t &);  // disable copy operator

  //kernel的入口函数，m_kernel_entry是 function_info 对象。
  class function_info *m_kernel_entry;

  //kernel_info_t对象的唯一标识符
  unsigned m_uid;  // Kernel ID
  unsigned long long m_streamID;

  // These maps contain the snapshot of the texture mappings at kernel launch
  std::map<std::string, const struct cudaArray *> m_NameToCudaArray;
  std::map<std::string, const struct textureInfo *> m_NameToTextureInfo;

  //Grid和Block的维度。
  dim3 m_grid_dim;
  dim3 m_block_dim;
  
  //m_next_cta是用于标识下一个要执行的CTA（Compute Thread Array）的变量。
  dim3 m_next_cta;
  dim3 m_next_tid;

  //正在执行当前kernel的SIMT Core的数量。
  unsigned m_num_cores_running;

  //当前处于活跃状态的线程列表。
  std::list<class ptx_thread_info *> m_active_threads;
  //kernel_info_t对象的参数内存空间。
  class memory_space *m_param_mem;

 public:
  // Jin: parent and child kernel management for CDP
  void set_parent(kernel_info_t *parent, dim3 parent_ctaid, dim3 parent_tid);
  void set_child(kernel_info_t *child);
  void remove_child(kernel_info_t *child);
  bool is_finished();
  bool children_all_finished();
  void notify_parent_finished();
  CUstream_st *create_stream_cta(dim3 ctaid);
  CUstream_st *get_default_stream_cta(dim3 ctaid);
  bool cta_has_stream(dim3 ctaid, CUstream_st *stream);
  void destroy_cta_streams();
  void print_parent_info();
  kernel_info_t *get_parent() { return m_parent_kernel; }

 private:
  kernel_info_t *m_parent_kernel;
  dim3 m_parent_ctaid;
  dim3 m_parent_tid;
  std::list<kernel_info_t *> m_child_kernels;  // child kernel launched
  std::map<dim3, std::list<CUstream_st *>, dim3comp>
      m_cta_streams;  // streams created in each CTA

  // Jin: kernel timing
 public:
  unsigned long long launch_cycle;
  unsigned long long start_cycle;
  unsigned long long end_cycle;
  unsigned m_launch_latency;

  mutable bool cache_config_set;

  //用于模拟kernel&block的启动延迟时间，这包括kernel的启动延迟，每个线程块的启动延迟。
  // m_kernel_TB_latency =
  //     entry->gpgpu_ctx->device_runtime->g_kernel_launch_latency +
  //     num_blocks() * entry->gpgpu_ctx->device_runtime->g_TB_launch_latency;
  unsigned m_kernel_TB_latency;  // this used for any CPU-GPU kernel latency and
                                 // counted in the gpu_cycle
};

/*
Shader Core配置信息类。
*/
class core_config {
 public:
  core_config(gpgpu_context *ctx) {
    gpgpu_ctx = ctx;
    m_valid = false;
    num_shmem_bank = 16;
    shmem_limited_broadcast = false;
    gpgpu_shmem_sizeDefault = (unsigned)-1;
    gpgpu_shmem_sizePrefL1 = (unsigned)-1;
    gpgpu_shmem_sizePrefShared = (unsigned)-1;
  }
  virtual void init() = 0;

  bool m_valid;
  unsigned warp_size;
  // backward pointer
  class gpgpu_context *gpgpu_ctx;

  // off-chip memory request architecture parameters
  //片外存储器请求架构参数。
  int gpgpu_coalesce_arch;

  // shared memory bank conflict checking parameters
  // Limit shared memory to do one broadcast per cycle (default on).
  //将共享内存限制为每个周期执行一次广播（默认设置为打开）。
  bool shmem_limited_broadcast;
  static const address_type WORD_SIZE = 4;
  unsigned num_shmem_bank;
  //根据数据地址，判断其位于哪一个shared memory的Bank。
  unsigned shmem_bank_func(address_type addr) const {
    return ((addr / WORD_SIZE) % num_shmem_bank);
  }
  // m_config->mem_warp_parts: Number of portions a warp is divided into for 
  // shared memory bank conflict check.
  unsigned mem_warp_parts;
  //每个SIMT Core（也称为Shader Core）的共享存储大小。由GPGPU-Sim的-gpgpu_shmem_size选项配置。
  mutable unsigned gpgpu_shmem_size;
  char *gpgpu_shmem_option;
  std::vector<unsigned> shmem_opt_list;
  unsigned gpgpu_shmem_sizeDefault;
  unsigned gpgpu_shmem_sizePrefL1;
  unsigned gpgpu_shmem_sizePrefShared;
  unsigned mem_unit_ports;

  // texture and constant cache line sizes (used to determine number of memory
  // accesses)
  //纹理缓存线大小（用于确定内存访问次数）。
  unsigned gpgpu_cache_texl1_linesize;
  //常量缓存线大小（用于确定内存访问次数）。
  unsigned gpgpu_cache_constl1_linesize;

  unsigned gpgpu_max_insn_issue_per_warp;
  bool gmem_skip_L1D;  // on = global memory access always skip the L1 cache
  //适应性的cache配置代表：在V100中，将剩余的不使用的shared memory划给L1 cache使用。
  bool adaptive_cache_config;
};

// bounded stack that implements simt reconvergence using pdom mechanism from
// MICRO'07 paper
const unsigned MAX_WARP_SIZE = 32;
//用于在处理一个warp内的线程分支，标记每个线程是否执行某个分支。
typedef std::bitset<MAX_WARP_SIZE> active_mask_t;
#define MAX_WARP_SIZE_SIMT_STACK MAX_WARP_SIZE
//线程掩码，位图，大小是32。
typedef std::bitset<MAX_WARP_SIZE_SIMT_STACK> simt_mask_t;
typedef std::vector<address_type> addr_vector_t;

/*
SIMT 堆栈类。每个SIMT Core中，都有可配置数量的调度器单元。对于每个调度器单元，有一个SIMT堆栈阵列。每
个SIMT堆栈对应一个warp。
*/
class simt_stack {
 public:
  simt_stack(unsigned wid, unsigned warpSize, class gpgpu_sim *gpu);

  //清空m_stack里的所有条目。
  void reset();
  //功能模拟过程中，用warp的起始PC值（用该warp的首个线程m_thread[warpId * m_warp_size]->get_pc()获
  //取）线程和其线程掩码用于启动SIMT堆栈。
  void launch(address_type start_pc, const simt_mask_t &active_mask);
  //在scheduler_unit::cycle()中，指令使用shader_core_ctx::issue_warp()函数被发射到其合适的执行流水
  //线。在这个函数中，指令通过调用shader_core_ctx::func_exec_inst()在功能上被执行，SIMT栈（m_simt_
  //stack[warp_id]）通过调用simt_stack::update()被更新。
  void update(simt_mask_t &thread_done, addr_vector_t &next_pc,
              address_type recvg_pc, op_type next_inst_op,
              unsigned next_inst_size, address_type next_inst_pc);

  //返回m_stack队列最末尾加入条目的线程掩码。[最末尾加入条目]即为栈顶top。
  const simt_mask_t &get_active_mask() const;
  //返回m_stack队列最末尾加入条目的PC值和RPC值。[最末尾加入条目]即为栈顶top。
  void get_pdom_stack_top_info(unsigned *pc, unsigned *rpc) const;
  //返回m_stack队列最末尾加入条目的RPC值。[最末尾加入条目]即为栈顶top。
  unsigned get_rp() const;
  //打印SIMT堆栈的每个条目。
  void print(FILE *fp) const;
  //暂时用不到，以后用到再补充。
  void resume(char *fname);
  //打印SIMT堆栈的check point。
  void print_checkpoint(FILE *fout) const;

 protected:
  //warp的ID。
  unsigned m_warp_id;
  //单个warp内的线程数量。
  unsigned m_warp_size;

  //SIMT堆栈条目的类型。在<<通用图形处理器体系结构>>书中有拓展。
  enum stack_entry_type { STACK_ENTRY_TYPE_NORMAL = 0, STACK_ENTRY_TYPE_CALL };

  //SIMT堆栈的条目结构。
  struct simt_stack_entry {
    //下一条需要被执行指令的PC（Next PC，NPC），为该分支内需要执行的指令PC。
    address_type m_pc;
    unsigned int m_calldepth;
    //线程活跃掩码，代表了这条指令的活跃掩码。
    simt_mask_t m_active_mask;
    //分支重聚点的PC（Reconvergence PC，RPC），是直接后必经结点的PC（IPDOM）。
    address_type m_recvg_pc;
    //发生分支的时刻，时钟周期。
    unsigned long long m_branch_div_cycle;
    //SIMT堆栈条目的类型。在<<通用图形处理器体系结构>>书中有拓展。
    stack_entry_type m_type;
    simt_stack_entry()
        : m_pc(-1),
          m_calldepth(0),
          m_active_mask(),
          m_recvg_pc(-1),
          m_branch_div_cycle(0),
          m_type(STACK_ENTRY_TYPE_NORMAL){};
  };
  //m_stack是SIMT堆栈的条目队列，成员是struct simt_stack_entry。
  std::deque<simt_stack_entry> m_stack;

  class gpgpu_sim *m_gpu;
};

// Let's just upgrade to C++11 so we can use constexpr here...
// start allocating from this address (lower values used for allocating globals
// in .ptx file)
const unsigned long long GLOBAL_HEAP_START = 0xC0000000;
// Volta max shmem size is 96kB
const unsigned long long SHARED_MEM_SIZE_MAX = 96 * (1 << 10);
// Volta max local mem is 16kB
const unsigned long long LOCAL_MEM_SIZE_MAX = 1 << 14;
// Volta Titan V has 80 SMs
const unsigned MAX_STREAMING_MULTIPROCESSORS = 80;
// Max 2048 threads / SM
const unsigned MAX_THREAD_PER_SM = 1 << 11;
// MAX 64 warps / SM
const unsigned MAX_WARP_PER_SM = 1 << 6;
const unsigned long long TOTAL_LOCAL_MEM_PER_SM =
    MAX_THREAD_PER_SM * LOCAL_MEM_SIZE_MAX;
const unsigned long long TOTAL_SHARED_MEM =
    MAX_STREAMING_MULTIPROCESSORS * SHARED_MEM_SIZE_MAX;
const unsigned long long TOTAL_LOCAL_MEM =
    MAX_STREAMING_MULTIPROCESSORS * MAX_THREAD_PER_SM * LOCAL_MEM_SIZE_MAX;
const unsigned long long SHARED_GENERIC_START =
    GLOBAL_HEAP_START - TOTAL_SHARED_MEM;
const unsigned long long LOCAL_GENERIC_START =
    SHARED_GENERIC_START - TOTAL_LOCAL_MEM;
const unsigned long long STATIC_ALLOC_LIMIT =
    GLOBAL_HEAP_START - (TOTAL_LOCAL_MEM + TOTAL_SHARED_MEM);

#if !defined(__CUDA_RUNTIME_API_H__)

#include "builtin_types.h"

struct cudaArray {
  void *devPtr;
  int devPtr32;
  struct cudaChannelFormatDesc desc;
  int width;
  int height;
  int size;  // in bytes
  unsigned dimensions;
};

#endif

// Struct that record other attributes in the textureReference declaration
// - These attributes are passed thru __cudaRegisterTexture()
struct textureReferenceAttr {
  const struct textureReference *m_texref;
  int m_dim;
  enum cudaTextureReadMode m_readmode;
  int m_ext;
  textureReferenceAttr(const struct textureReference *texref, int dim,
                       enum cudaTextureReadMode readmode, int ext)
      : m_texref(texref), m_dim(dim), m_readmode(readmode), m_ext(ext) {}
};

class gpgpu_functional_sim_config {
 public:
  void reg_options(class OptionParser *opp);

  void ptx_set_tex_cache_linesize(unsigned linesize);

  unsigned get_forced_max_capability() const {
    return m_ptx_force_max_capability;
  }
  bool convert_to_ptxplus() const { return m_ptx_convert_to_ptxplus; }
  bool use_cuobjdump() const { return m_ptx_use_cuobjdump; }
  bool experimental_lib_support() const { return m_experimental_lib_support; }

  int get_ptx_inst_debug_to_file() const { return g_ptx_inst_debug_to_file; }
  const char *get_ptx_inst_debug_file() const { return g_ptx_inst_debug_file; }
  int get_ptx_inst_debug_thread_uid() const {
    return g_ptx_inst_debug_thread_uid;
  }
  unsigned get_texcache_linesize() const { return m_texcache_linesize; }
  int get_checkpoint_option() const { return checkpoint_option; }
  int get_checkpoint_kernel() const { return checkpoint_kernel; }
  int get_checkpoint_CTA() const { return checkpoint_CTA; }
  int get_resume_option() const { return resume_option; }
  int get_resume_kernel() const { return resume_kernel; }
  int get_resume_CTA() const { return resume_CTA; }
  int get_checkpoint_CTA_t() const { return checkpoint_CTA_t; }
  int get_checkpoint_insn_Y() const { return checkpoint_insn_Y; }

 private:
  // PTX options
  int m_ptx_convert_to_ptxplus;
  int m_ptx_use_cuobjdump;
  int m_experimental_lib_support;
  unsigned m_ptx_force_max_capability;
  int checkpoint_option;
  int checkpoint_kernel;
  int checkpoint_CTA;
  unsigned resume_option;
  unsigned resume_kernel;
  unsigned resume_CTA;
  unsigned checkpoint_CTA_t;
  int checkpoint_insn_Y;
  int g_ptx_inst_debug_to_file;
  char *g_ptx_inst_debug_file;
  int g_ptx_inst_debug_thread_uid;

  unsigned m_texcache_linesize;
};

class gpgpu_t {
 public:
  gpgpu_t(const gpgpu_functional_sim_config &config, gpgpu_context *ctx);
  // backward pointer
  class gpgpu_context *gpgpu_ctx;
  int checkpoint_option;
  int checkpoint_kernel;
  int checkpoint_CTA;
  unsigned resume_option;
  unsigned resume_kernel;
  unsigned resume_CTA;
  unsigned checkpoint_CTA_t;
  int checkpoint_insn_Y;

  // Move some cycle core stats here instead of being global
  //gpu_sim_cycle是执行当前阶段的指令的延迟。
  unsigned long long gpu_sim_cycle;
  //gpu_tot_sim_cycle是执行当前阶段之前的所有前绪指令的延迟。
  unsigned long long gpu_tot_sim_cycle;

  void *gpu_malloc(size_t size);
  void *gpu_mallocarray(size_t count);
  void gpu_memset(size_t dst_start_addr, int c, size_t count);
  void memcpy_to_gpu(size_t dst_start_addr, const void *src, size_t count);
  void memcpy_from_gpu(void *dst, size_t src_start_addr, size_t count);
  void memcpy_gpu_to_gpu(size_t dst, size_t src, size_t count);

  class memory_space *get_global_memory() {
    return m_global_mem;
  }
  class memory_space *get_tex_memory() {
    return m_tex_mem;
  }
  class memory_space *get_surf_memory() {
    return m_surf_mem;
  }

  void gpgpu_ptx_sim_bindTextureToArray(const struct textureReference *texref,
                                        const struct cudaArray *array);
  void gpgpu_ptx_sim_bindNameToTexture(const char *name,
                                       const struct textureReference *texref,
                                       int dim, int readmode, int ext);
  void gpgpu_ptx_sim_unbindTexture(const struct textureReference *texref);
  const char *gpgpu_ptx_sim_findNamefromTexture(
      const struct textureReference *texref);

  const struct textureReference *get_texref(const std::string &texname) const {
    std::map<std::string,
             std::set<const struct textureReference *> >::const_iterator t =
        m_NameToTextureRef.find(texname);
    assert(t != m_NameToTextureRef.end());
    return *(t->second.begin());
  }

  const struct cudaArray *get_texarray(const std::string &texname) const {
    std::map<std::string, const struct cudaArray *>::const_iterator t =
        m_NameToCudaArray.find(texname);
    assert(t != m_NameToCudaArray.end());
    return t->second;
  }

  const struct textureInfo *get_texinfo(const std::string &texname) const {
    std::map<std::string, const struct textureInfo *>::const_iterator t =
        m_NameToTextureInfo.find(texname);
    assert(t != m_NameToTextureInfo.end());
    return t->second;
  }

  const struct textureReferenceAttr *get_texattr(
      const std::string &texname) const {
    std::map<std::string, const struct textureReferenceAttr *>::const_iterator
        t = m_NameToAttribute.find(texname);
    assert(t != m_NameToAttribute.end());
    return t->second;
  }

  const gpgpu_functional_sim_config &get_config() const {
    return m_function_model_config;
  }
  FILE *get_ptx_inst_debug_file() { return ptx_inst_debug_file; }

  //  These maps return the current texture mappings for the GPU at any given
  //  time.
  std::map<std::string, const struct cudaArray *> getNameArrayMapping() {
    return m_NameToCudaArray;
  }
  std::map<std::string, const struct textureInfo *> getNameInfoMapping() {
    return m_NameToTextureInfo;
  }

  virtual ~gpgpu_t() {}

 protected:
  const gpgpu_functional_sim_config &m_function_model_config;
  FILE *ptx_inst_debug_file;

  class memory_space *m_global_mem;
  class memory_space *m_tex_mem;
  class memory_space *m_surf_mem;

  unsigned long long m_dev_malloc;
  //  These maps contain the current texture mappings for the GPU at any given
  //  time.
  std::map<std::string, std::set<const struct textureReference *> >
      m_NameToTextureRef;
  std::map<const struct textureReference *, std::string> m_TextureRefToName;
  std::map<std::string, const struct cudaArray *> m_NameToCudaArray;
  std::map<std::string, const struct textureInfo *> m_NameToTextureInfo;
  std::map<std::string, const struct textureReferenceAttr *> m_NameToAttribute;
};

struct gpgpu_ptx_sim_info {
  // Holds properties of the kernel (Kernel's resource use).
  // These will be set to zero if a ptxinfo file is not present.
  int lmem;
  int smem;
  int cmem;
  int gmem;
  int regs;
  unsigned maxthreads;
  unsigned ptx_version;
  unsigned sm_target;
};

struct gpgpu_ptx_sim_arg {
  gpgpu_ptx_sim_arg() { m_start = NULL; }
  gpgpu_ptx_sim_arg(const void *arg, size_t size, size_t offset) {
    m_start = arg;
    m_nbytes = size;
    m_offset = offset;
  }
  const void *m_start;
  size_t m_nbytes;
  size_t m_offset;
};

typedef std::list<gpgpu_ptx_sim_arg> gpgpu_ptx_sim_arg_list_t;

/*
存储空间的信息，如存储空间的类型和该存储空间的Bank的数量。GPGPU-Sim设置的存储空间的类型有：
    enum _memory_space_t {
      //1. 未定义的空间类型
      undefined_space = 0,
      //2. 寄存器
      reg_space,
      //3. local memory
      local_space,
      //4. shared memory
      shared_space,
      //5. 貌似是 shared static array，其访存的行为与shared memory一致，可以认为其是shared 
      //   memory的一种
      sstarr_space,
      //6. 通用参数存储
      param_space_unclassified,
      //7. 对内核中的所有线程：全局性的，只读的
      param_space_kernel, // global to all threads in a kernel : read-only
      //8. 对某个线程：私有的，可读写的
      param_space_local,  // local to a thread : read-writable
      //9. 常量缓存
      const_space,
      //10.纹理缓存
      tex_space,
      //11.渲染曲面 // render surfaces 
      surf_space,
      //12.全局存储
      global_space,
      //13.通用存储
      generic_space,
      //14.指令存储
      instruction_space
    };
*/
class memory_space_t {
 public:
  //构造函数。初始时，设置存储空间类型为 未定义的空间类型，设置 Bank 数为0。
  memory_space_t() {
    m_type = undefined_space;
    m_bank = 0;
  }
  //构造函数。设置存储空间类型为 传入参数的类型，设置 Bank 数为0。
  memory_space_t(const enum _memory_space_t &from) {
    m_type = from;
    m_bank = 0;
  }
  bool operator==(const memory_space_t &x) const {
    return (m_bank == x.m_bank) && (m_type == x.m_type);
  }
  bool operator!=(const memory_space_t &x) const { return !(*this == x); }
  bool operator<(const memory_space_t &x) const {
    if (m_type < x.m_type)
      return true;
    else if (m_type > x.m_type)
      return false;
    else if (m_bank < x.m_bank)
      return true;
    return false;
  }
  enum _memory_space_t get_type() const { return m_type; }
  void set_type(enum _memory_space_t t) { m_type = t; }
  unsigned get_bank() const { return m_bank; }
  void set_bank(unsigned b) { m_bank = b; }
  bool is_const() const {
    return (m_type == const_space) || (m_type == param_space_kernel);
  }
  bool is_local() const {
    return (m_type == local_space) || (m_type == param_space_local);
  }
  bool is_global() const { return (m_type == global_space); }

 private:
  enum _memory_space_t m_type;
  unsigned m_bank;  // n in ".const[n]"; note .const == .const[0] (see PTX 2.1
                    // manual, sec. 5.1.3)
};

const unsigned MAX_MEMORY_ACCESS_SIZE = 128;
// 用于标记一次访存操作中的数据字节掩码，MAX_MEMORY_ACCESS_SIZE 设置为 128，即每次访
// 存最大数据 128 字节。
typedef std::bitset<MAX_MEMORY_ACCESS_SIZE> mem_access_byte_mask_t;
// 一个 cache line 包含 SECTOR_CHUNCK_SIZE = 4 个 sectors。
const unsigned SECTOR_CHUNCK_SIZE = 4;  // four sectors
const unsigned SECTOR_SIZE = 32;        // sector is 32 bytes width
// 用于标记一次访存操作中的 sector 掩码，4 个 sector，每个 sector 有 32 个字节数据。
typedef std::bitset<SECTOR_CHUNCK_SIZE> mem_access_sector_mask_t;
#define NO_PARTIAL_WRITE (mem_access_byte_mask_t())

// L1_WR_ALLOC_R/L2_WR_ALLOC_R 在 V100 配置中暂时用不到。
// 在 V100 中，L1 cache 的 m_write_policy 为 WRITE_THROUGH，实际上 L1_WRBK_ACC 也
// 不会用到。在 V100 中，当 L2 cache 写不命中时，采取 lazy_fetch_on_read 策略，当找
// 到一个 cache block逐出时，如果这个 cache block 是被 MODIFIED，则需要将这个 cache 
// block 写回到下一级存储，因此会产生 L2_WRBK_ACC 访问，这个访问就是为了写回被逐出的 
// MODIFIED cache block。
/*
enum mem_access_type {
  GLOBAL_ACC_R, LOCAL_ACC_R, CONST_ACC_R, TEXTURE_ACC_R, GLOBAL_ACC_W,
  LOCAL_ACC_W, L1_WRBK_ACC, L2_WRBK_ACC, INST_ACC_R, L1_WR_ALLOC_R,
  L2_WR_ALLOC_R, NUM_MEM_ACCESS_TYPE
};
L2_WR_ALLOC_R: 当前缓存层次是 L2 缓存，如果 mf 的类型是 L2_WR_ALLOC_R，说明 L2 缓存
               发生了写不命中，需要将主存中块调入 L2 缓存再写入该块，因此 mf 的类型是 
               L2_WR_ALLOC_R 时，不能将 mf 再向 ICNT 发送，仅将主存块放入 L2 即可。
L1_WR_ALLOC_R: 同上，发生在 L1 缓存。
*/
#define MEM_ACCESS_TYPE_TUP_DEF                                         \
  MA_TUP_BEGIN(mem_access_type)                                         \
  MA_TUP(GLOBAL_ACC_R), MA_TUP(LOCAL_ACC_R), MA_TUP(CONST_ACC_R),       \
      MA_TUP(TEXTURE_ACC_R), MA_TUP(GLOBAL_ACC_W), MA_TUP(LOCAL_ACC_W), \
      MA_TUP(L1_WRBK_ACC), MA_TUP(L2_WRBK_ACC), MA_TUP(INST_ACC_R),     \
      MA_TUP(L1_WR_ALLOC_R), MA_TUP(L2_WR_ALLOC_R),                     \
      MA_TUP(NUM_MEM_ACCESS_TYPE) MA_TUP_END(mem_access_type)

// 作者又在炫技了。
#define MA_TUP_BEGIN(X) enum X {
#define MA_TUP(X) X
#define MA_TUP_END(X) \
  }                   \
  ;
MEM_ACCESS_TYPE_TUP_DEF
#undef MA_TUP_BEGIN
#undef MA_TUP
#undef MA_TUP_END

const char *mem_access_type_str(enum mem_access_type access_type);

/*
缓存运算符的类型。
PTX ISA 2.0 版在加载和存储指令上引入了可选的缓存运算符。缓存运算符需要 sm_20 或更高的
目标体系结构。加载或存储指令上的缓存运算符仅被视为性能提示。对 ld 或 st 指令使用缓存运
算符不会改变程序的内存一致性行为。对于 sm_20 及更高版本，缓存运算符具有以下定义和行为：
1. 内存 Load 指令的缓存运算符：
 .ca: Cache at all levels，所有级别的缓存，可能会再次访问。
      默认的 Load 指令缓存操作是 ld.ca，它使用正常的逐出策略在所有级别（L1 和 L2）中
      分配缓存线。全局数据在 L2 级是一致的，但多个 L1 缓存对于全局数据来说是不一致的。
      如果一个线程通过一个 L1 缓存存储到全局内存，而第二个线程通过第二个 L1 缓存加载该
      地址，并使用 ld.ca，则第二个可能会得到过时的 L1 缓存数据，而不是第一个线程存储的
      数据。驱动程序必须使并行线程的从属网格之间的全局 L1 缓存线无效。然后，第一个网格
      程序的存储由第二个网格程序正确获取，该程序发出 L1 中缓存的默认 ld.ca 加载。
 .cg  Cache at global level，全局级缓存（L2 及以下缓存，而不是 L1）。
      使用 ld.cg 全局性地加载，绕过 L1 缓存，并仅缓存在 L2 缓存中。
 .cs  Cache streaming，缓存流，可能会被访问一次。
      ld.cs 加载缓存的流操作在 L1 和 L2 分配全局行，采用驱逐优先的策略在 L1 和 L2 分
      配全局行，以限制临时流数据对缓存污染，这些数据可能被访问一次或两次。当 ld.cs 被
      应用到一个本地窗口地址时，它执行 ld.lu 操作。
 .lu  Last use。
      编译器/程序员在恢复溢出的寄存器和弹出函数堆栈框架时可以使用 ld.lu，以避免不必要
      的写回不会再使用的行。ld.lu 指令在全局地址上执行一个加载缓存的流操作（ld.cs）。
 .cv  不要再次缓存和获取（考虑缓存的系统内存线已过时，再次获取）。应用于全局系统内存地
      址的 ld.cv 加载操作使匹配的 L2 行无效（丢弃），并在每次新加载时重新获取该行。
2. 内存 Store 指令的缓存运算符：
 .wb  Cache write-back all coherent levels，缓存写回所有级别一致。
      默认的存储指令缓存操作是 st.wb，它使用正常的逐出策略写回一致缓存级别的缓存行。如
      果一个线程绕过其 L1 缓存存储到全局内存，而另一个 SM 中的第二个线程稍后通过具有 
      ld.ca 的不同 L1 缓存从该地址加载，则第二个可能会命中过时的 L1 缓存数据，而不是从
      第一个线程存储的 L2 或内存中获取数据。驱动程序必须使线程阵列的从属网格之间的全局 
      L1 缓存线无效。然后，第一个网格程序的存储在 L1 中正确丢失，并由发出默认 ld.ca 加
      载的第二个网格程序获取。
 .wt  Cache write-through (to system memory).
      st.wt 存储写入操作应用于通过二级缓存写入的全局系统内存地址。
*/
enum cache_operator_type {
  CACHE_UNDEFINED,

  // loads
  CACHE_ALL,       // .ca
  CACHE_LAST_USE,  // .lu
  CACHE_VOLATILE,  // .cv
  CACHE_L1,        // .nc

  // loads and stores
  CACHE_STREAMING,  // .cs
  CACHE_GLOBAL,     // .cg

  // stores
  CACHE_WRITE_BACK,    // .wb
  CACHE_WRITE_THROUGH  // .wt
};

/*
包含时序模拟器中每个内存访问的信息。该类包含内存访问的类型、请求的地址、数据的大小以及
访问内存的 warp 的活动掩码等信息。该类被用作 mem_fetch 类的参数之一，该类基本上为每个
内存访问实例化。这个类是用于两个不同级别的内存之间的接口，并将两者互连。
*/
class mem_access_t {
 public:
  // 构造函数。mem_access_t 类有一个私有的 gpgpu_context *gpgpu_ctx 对象，初始化该
  // 对象。gpgpu_ctx 用于标识是哪个 gpgpu_context 对象生成了该存储访问的信息。
  mem_access_t(gpgpu_context *ctx) { init(ctx); }
  // 构造函数。
  // new_addr_type 定义：typedef unsigned long long new_addr_type;
  mem_access_t(mem_access_type type, new_addr_type address, unsigned size,
               bool wr, gpgpu_context *ctx) {
    // 初始化 gpgpu_context *gpgpu_ctx 对象。
    init(ctx);
    // m_type 定义：
    //    mem_access_type m_type;
    // mem_access_type 定义了在时序模拟器中对不同类型的存储器进行不同的访存类型：
    //    MA_TUP(GLOBAL_ACC_R), 从 global memory 读
    //    MA_TUP(LOCAL_ACC_R), 从 local memory 读
    //    MA_TUP(CONST_ACC_R), 从常量缓存读
    //    MA_TUP(TEXTURE_ACC_R), 从纹理缓存读
    //    MA_TUP(GLOBAL_ACC_W), 向 global memory 写
    //    MA_TUP(LOCAL_ACC_W), 向 local memory 写
    // 在 V100 中，L1 cache 的 m_write_policy 为WRITE_THROUGH，实际上 L1_WRBK_ACC 
    // 也不会用到：
    //    MA_TUP(L1_WRBK_ACC), L1缓存write back
    // 在 V100 中，当 L2 cache 写不命中时，采取 lazy_fetch_on_read 策略，当找到一个 
    // cache block 逐出时，如果这个 cache block 是被 MODIFIED，则需要将这个 cache 
    // block 写回到下一级存储，因此会产生 L2_WRBK_ACC 访问，这个访问就是为了写回被逐 
    // 出的 MODIFIED cache block。
    //    MA_TUP(L2_WRBK_ACC), L2 缓存 write back
    //    MA_TUP(INST_ACC_R), 从指令缓存读
    // L1_WR_ALLOC_R / L2_WR_ALLOC_R 在 V100 配置中暂时用不到：
    //    MA_TUP(L1_WR_ALLOC_R), L1 缓存 write-allocate（对 cache 写不命中，将主存
    //                           中块调入 cache，写入该 cache 块）
    // L1_WR_ALLOC_R / L2_WR_ALLOC_R 在 V100 配置中暂时用不到：
    //    MA_TUP(L2_WR_ALLOC_R), L2 缓存 write-allocate
    m_type = type;
    // 访存的地址。
    m_addr = address;
    // 访存的数据大小，以字节为单位。
    m_req_size = size;
    // 该访存是写/读，1-写，0-读。
    m_write = wr;
  }
  // 构造函数。
  // active_mask_t 活跃掩码定义：
  //    typedef std::bitset<MAX_WARP_SIZE> active_mask_t; 
  // 用于在处理一个 warp 内的线程分支，标记每个线程是否执行某个分支。
  // mem_access_byte_mask_t 访存数据字节掩码定义：
  //    const unsigned MAX_MEMORY_ACCESS_SIZE = 128;
  //    typedef std::bitset<MAX_MEMORY_ACCESS_SIZE> mem_access_byte_mask_t;
  // 用于标记一次访存操作中的数据字节掩码，MAX_MEMORY_ACCESS_SIZE 设置为 128，即每次
  // 访存最大数据 128 字节。
  // mem_access_sector_mask_t 扇区掩码定义：
  //    const unsigned SECTOR_CHUNCK_SIZE = 4;  // four sectors
  //    typedef std::bitset<SECTOR_CHUNCK_SIZE> mem_access_sector_mask_t;
  // 用于标记一次访存操作中的扇区掩码，4 个扇区，每个扇区 32 个字节数据。
  mem_access_t(mem_access_type type, new_addr_type address, unsigned size,
               bool wr, const active_mask_t &active_mask,
               const mem_access_byte_mask_t &byte_mask,
               const mem_access_sector_mask_t &sector_mask, gpgpu_context *ctx)
      : m_warp_mask(active_mask),
        m_byte_mask(byte_mask),
        m_sector_mask(sector_mask) {
    init(ctx);
    m_type = type;
    m_addr = address;
    m_req_size = size;
    m_write = wr;
  }
  // 返回访存地址。new_addr_type 定义：typedef unsigned long long new_addr_type;
  new_addr_type get_addr() const { return m_addr; }
  // 设置访存地址。
  void set_addr(new_addr_type addr) { m_addr = addr; }
  // 返回访存数据大小，以字节为单位。
  unsigned get_size() const { return m_req_size; }
  // 返回访存的线程活跃掩码。
  const active_mask_t &get_warp_mask() const { return m_warp_mask; }
  // 返回该访存是写/读，1-写，0-读。
  bool is_write() const { return m_write; }
  // 返回对存储器进行的访存类型，见构造函数注释。
  enum mem_access_type get_type() const { return m_type; }
  // 返回访存的数据字节掩码。
  // mem_access_byte_mask_t 访存数据字节掩码定义：
  //    const unsigned MAX_MEMORY_ACCESS_SIZE = 128;
  //    typedef std::bitset<MAX_MEMORY_ACCESS_SIZE> mem_access_byte_mask_t;
  // 用于标记一次访存操作中的数据字节掩码，MAX_MEMORY_ACCESS_SIZE 设置为 128，即每次
  // 访存最大数据 128 字节。
  mem_access_byte_mask_t get_byte_mask() const { return m_byte_mask; }
  // 返回访存的扇区掩码。
  // mem_access_sector_mask_t 扇区掩码定义：
  //    const unsigned SECTOR_CHUNCK_SIZE = 4;  // four sectors
  //    typedef std::bitset<SECTOR_CHUNCK_SIZE> mem_access_sector_mask_t;
  // 用于标记一次访存操作中的扇区掩码，4 个扇区，每个扇区 32 个字节数据。
  mem_access_sector_mask_t get_sector_mask() const { return m_sector_mask; }

  // 将访存的地址、store 或 load、数据大小、访存类型打印到文件。
  void print(FILE *fp) const {
    fprintf(fp, "addr=0x%llx, %s, size=%u, ", m_addr,
            m_write ? "store" : "load ", m_req_size);
    switch (m_type) {
      case GLOBAL_ACC_R:
        fprintf(fp, "GLOBAL_R");
        break;
      case LOCAL_ACC_R:
        fprintf(fp, "LOCAL_R ");
        break;
      case CONST_ACC_R:
        fprintf(fp, "CONST   ");
        break;
      case TEXTURE_ACC_R:
        fprintf(fp, "TEXTURE ");
        break;
      case GLOBAL_ACC_W:
        fprintf(fp, "GLOBAL_W");
        break;
      case LOCAL_ACC_W:
        fprintf(fp, "LOCAL_W ");
        break;
      case L2_WRBK_ACC:
        fprintf(fp, "L2_WRBK ");
        break;
      case INST_ACC_R:
        fprintf(fp, "INST    ");
        break;
      case L1_WRBK_ACC:
        fprintf(fp, "L1_WRBK ");
        break;
      default:
        fprintf(fp, "unknown ");
        break;
    }
  }

  gpgpu_context *gpgpu_ctx;

 private:
  void init(gpgpu_context *ctx);
  // 该次访存操作的唯一 ID。
  unsigned m_uid;
  // 访存地址。new_addr_type 定义：typedef unsigned long long new_addr_type;
  new_addr_type m_addr;  // request address
  // 该访存是写/读，1-写，0-读。
  bool m_write;
  // 访存数据大小，以字节为单位。
  unsigned m_req_size;  // bytes
  // 对不同类型的存储器进行的访存类型，见构造函数注释。
  mem_access_type m_type;
  // 访存的线程活跃掩码。
  active_mask_t m_warp_mask;
  // 访存的数据字节掩码。
  // mem_access_byte_mask_t 访存数据字节掩码定义：
  //    const unsigned MAX_MEMORY_ACCESS_SIZE = 128;
  //    typedef std::bitset<MAX_MEMORY_ACCESS_SIZE> mem_access_byte_mask_t;
  // 用于标记一次访存操作中的数据字节掩码，MAX_MEMORY_ACCESS_SIZE 设置为 128，即每次
  // 访存最大数据 128 字节。
  mem_access_byte_mask_t m_byte_mask;
  // 访存的扇区掩码。
  // mem_access_sector_mask_t 扇区掩码定义：
  //    const unsigned SECTOR_CHUNCK_SIZE = 4;  // four sectors
  //    typedef std::bitset<SECTOR_CHUNCK_SIZE> mem_access_sector_mask_t;
  // 用于标记一次访存操作中的扇区掩码，4 个扇区，每个扇区 32 个字节数据。
  mem_access_sector_mask_t m_sector_mask;
};

class mem_fetch;

/*
对mem访存的接口。
*/
class mem_fetch_interface {
 public:
  virtual bool full(unsigned size, bool write) const = 0;
  virtual void push(mem_fetch *mf) = 0;
};

class mem_fetch_allocator {
 public:
  virtual mem_fetch *alloc(new_addr_type addr, mem_access_type type,
                           unsigned size, bool wr, unsigned long long cycle,
                           unsigned long long streamID) const = 0;
  virtual mem_fetch *alloc(const class warp_inst_t &inst,
                           const mem_access_t &access,
                           unsigned long long cycle) const = 0;
  virtual mem_fetch *alloc(new_addr_type addr, mem_access_type type,
                           const active_mask_t &active_mask,
                           const mem_access_byte_mask_t &byte_mask,
                           const mem_access_sector_mask_t &sector_mask,
                           unsigned size, bool wr, unsigned long long cycle,
                           unsigned wid, unsigned sid, unsigned tpc,
                           mem_fetch *original_mf,
                           unsigned long long streamID) const = 0;
};

// the maximum number of destination, source, or address uarch operands in a
// instruction
#define MAX_REG_OPERANDS 32

struct dram_callback_t {
  dram_callback_t() {
    function = NULL;
    instruction = NULL;
    thread = NULL;
  }
  void (*function)(const class inst_t *, class ptx_thread_info *);

  const class inst_t *instruction;
  class ptx_thread_info *thread;
};

class inst_t {
 public:
  inst_t() {
    m_decoded = false;
    pc = (address_type)-1;
    reconvergence_pc = (address_type)-1;
    op = NO_OP;
    bar_type = NOT_BAR;
    red_type = NOT_RED;
    bar_id = (unsigned)-1;
    bar_count = (unsigned)-1;
    oprnd_type = UN_OP;
    sp_op = OTHER_OP;
    op_pipe = UNKOWN_OP;
    mem_op = NOT_TEX;
    const_cache_operand = 0;
    num_operands = 0;
    num_regs = 0;
    memset(out, 0, sizeof(unsigned));
    memset(in, 0, sizeof(unsigned));
    is_vectorin = 0;
    is_vectorout = 0;
    space = memory_space_t();
    cache_op = CACHE_UNDEFINED;
    latency = 1;
    initiation_interval = 1;
    for (unsigned i = 0; i < MAX_REG_OPERANDS; i++) {
      arch_reg.src[i] = -1;
      arch_reg.dst[i] = -1;
    }
    isize = 0;
  }
  bool valid() const { return m_decoded; }
  virtual void print_insn(FILE *fp) const {
    fprintf(fp, " [inst @ pc=0x%04llx] ", pc);
  }
  //指令的操作码是 LOAD 或 TENSOR_CORE_LOAD 则为load指令。
  bool is_load() const {
    return (op == LOAD_OP || op == TENSOR_CORE_LOAD_OP ||
            memory_op == memory_load);
  }
  //指令的操作码是 STORE 或 TENSOR_CORE_STORE 则为store指令。
  bool is_store() const {
    return (op == STORE_OP || op == TENSOR_CORE_STORE_OP ||
            memory_op == memory_store);
  }

  bool is_fp() const { return ((sp_op == FP__OP)); }  // VIJAY
  bool is_fpdiv() const { return ((sp_op == FP_DIV_OP)); }
  bool is_fpmul() const { return ((sp_op == FP_MUL_OP)); }
  bool is_dp() const { return ((sp_op == DP___OP)); }
  bool is_dpdiv() const { return ((sp_op == DP_DIV_OP)); }
  bool is_dpmul() const { return ((sp_op == DP_MUL_OP)); }
  bool is_imul() const { return ((sp_op == INT_MUL_OP)); }
  bool is_imul24() const { return ((sp_op == INT_MUL24_OP)); }
  bool is_imul32() const { return ((sp_op == INT_MUL32_OP)); }
  bool is_idiv() const { return ((sp_op == INT_DIV_OP)); }
  bool is_sfu() const {
    return ((sp_op == FP_SQRT_OP) || (sp_op == FP_LG_OP) ||
            (sp_op == FP_SIN_OP) || (sp_op == FP_EXP_OP) ||
            (sp_op == TENSOR__OP));
  }
  bool is_alu() const { return (sp_op == INT__OP); }
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
  unsigned get_num_operands() const { return num_operands; }
  unsigned get_num_regs() const { return num_regs; }
  void set_num_regs(unsigned num) { num_regs = num; }
  //没用到。
  void set_num_operands(unsigned num) { num_operands = num; }
  void set_bar_id(unsigned id) { bar_id = id; }
  void set_bar_count(unsigned count) { bar_count = count; }

  address_type pc;  // program counter address of instruction
  unsigned isize;   // size of instruction in bytes
  op_type op;       // opcode (uarch visible)

  barrier_type bar_type;
  reduction_type red_type;
  unsigned bar_id;
  unsigned bar_count;

  types_of_operands oprnd_type;  // code (uarch visible) identify if the
                                 // operation is an interger or a floating point
  special_ops
      sp_op;  // code (uarch visible) identify if int_alu, fp_alu, int_mul ....
  //操作码code(uarch可见)，标识操作的流水线(SP、SFU或MEM)。
  operation_pipeline op_pipe;  // code (uarch visible) identify the pipeline of
                               // the operation (SP, SFU or MEM)
  mem_operation mem_op;        // code (uarch visible) identify memory type
  bool const_cache_operand;    // has a load from constant memory as an operand
  _memory_op_t memory_op;      // memory_op used by ptxplus
  unsigned num_operands;
  unsigned num_regs;  // count vector operand as one register operand

  address_type reconvergence_pc;  // -1 => not a branch, -2 => use function
                                  // return address

  //记录了当前指令的所有目的操作数寄存器ID。
  unsigned out[8];
  //记录了当前指令的所有目的操作数寄存器总数。
  unsigned outcount;
  //记录了当前指令的所有源操作数寄存器ID。
  unsigned in[24];
  //记录了当前指令的所有源操作数寄存器总数。
  unsigned incount;
  unsigned char is_vectorin;
  unsigned char is_vectorout;
  int pred;  // predicate register number
  int ar1, ar2;
  // register number for bank conflict evaluation
  struct {
    int dst[MAX_REG_OPERANDS];
    int src[MAX_REG_OPERANDS];
  } arch_reg;
  // int arch_reg[MAX_REG_OPERANDS]; // register number for bank conflict
  // evaluation
  unsigned latency;  // operation latency
  unsigned initiation_interval;

  unsigned data_size;  // what is the size of the word being operated on?
  //这里是每条指令都对应有自己的一个存储空间，需要在指令解码时设置。
  memory_space_t space;
  cache_operator_type cache_op;

 protected:
  bool m_decoded;
  virtual void pre_decode() {}
};

enum divergence_support_t { POST_DOMINATOR = 1, NUM_SIMD_MODEL };

const unsigned MAX_ACCESSES_PER_INSN_PER_THREAD = 8;

/*
时序仿真中需要的指令数据。每条指令（ptx_instruction）都继承自warp_inst_t，包含用于时序和功能仿真的数据。
ptx_instruction在功能仿真时被填充。在这之后，程序只需要时序信息，所以它将ptx_instruction转为warp_inst_t
（一些数据被释放）用于时序模拟。它持有warp_id、warp内的活动线程掩码、内存访问列表（mem_access_t）和该warp
内线程的信息（per_thread_info）。
*/
class warp_inst_t : public inst_t {
 public:
  // constructors
  //构造函数。
  warp_inst_t() {
    //每条指令都有一个唯一ID。
    m_uid = 0;
    m_streamID = (unsigned long long)-1;
    //m_empty是一个标志变量，用于表示当前warp_inst_t类中的指令是否为空。如果指令为空，则m_empty置为true，
    //否则置为false。warp_inst_t指令初始化时，设置为空。
    m_empty = true;
    //m_config是GPU Shader Core的配置。
    m_config = NULL;

    // Ni:
    m_is_ldgsts = false;
    m_is_ldgdepbar = false;
    m_is_depbar = false;

    m_depbar_group_no = 0;
  }
  //构造函数。
  warp_inst_t(const core_config *config) {
    m_uid = 0;
    m_streamID = (unsigned long long)-1;
    assert(config->warp_size <= MAX_WARP_SIZE);
    //Shader Core的配置。
    m_config = config;
    //m_empty是一个标志变量，用于表示当前warp_inst_t类中的指令是否为空。如果指令为空，则m_empty置为true，
    //否则置为false。warp_inst_t指令初始化时，设置为空。
    m_empty = true;
    //m_isatomic变量是一个布尔值，用于指示指令是否为原子操作（atomic operation）。原子操作是一种特殊的指
    //令，它可以保证在同一时间只有一个线程执行该指令，从而避免多个线程之间的竞争。
    m_isatomic = false;
    m_per_scalar_thread_valid = false;
    //代表是否已经生成访存操作，warp_inst_t指令初始化时，设置为false；生成访存操作后，设置为true。
    m_mem_accesses_created = false;
    m_cache_hit = false;
    //在ptx_ir.h中：API为vprintf时，if (fname == "vprintf") {m_is_printf = true;}
    m_is_printf = false;
    //CDP指的是Cuda Dynamic Parallelism，在ptx_ir.h中：
    //  API为cudaStreamCreateWithFlags时，if (fname == "cudaStreamCreateWithFlags") m_is_cdp = 1;
    //  API为cudaGetParameterBufferV2时，if (fname == "cudaGetParameterBufferV2") m_is_cdp = 2;
    //  API为cudaLaunchDeviceV2时，if (fname == "cudaLaunchDeviceV2") m_is_cdp = 4;
    m_is_cdp = 0;
    //指示GPU是否应该执行原子操作。它用于确保多个线程可以访问和修改共享数据，而不会导致竞争条件或数据损坏。
    should_do_atomic = true;

    // Ni:
    m_is_ldgsts = false;
    m_is_ldgdepbar = false;
    m_is_depbar = false;

    m_depbar_group_no = 0;
  }
  //析构函数。
  virtual ~warp_inst_t() {}

  // modifiers
  void broadcast_barrier_reduction(const active_mask_t &access_mask);
  void do_atomic(bool forceDo = false);
  void do_atomic(const active_mask_t &access_mask, bool forceDo = false);
  void clear() { m_empty = true; }

  void issue(const active_mask_t &mask, unsigned warp_id,
             unsigned long long cycle, int dynamic_warp_id, int sch_id,
             unsigned long long streamID);

  const active_mask_t &get_active_mask() const { return m_warp_active_mask; }
  void completed(unsigned long long cycle)
      const;  // stat collection: called when the instruction is completed

  void set_addr(unsigned n, new_addr_type addr) {
    if (!m_per_scalar_thread_valid) {
      m_per_scalar_thread.resize(m_config->warp_size);
      m_per_scalar_thread_valid = true;
    }
    m_per_scalar_thread[n].memreqaddr[0] = addr;
  }
  void set_addr(unsigned n, new_addr_type *addr, unsigned num_addrs) {
    if (!m_per_scalar_thread_valid) {
      m_per_scalar_thread.resize(m_config->warp_size);
      m_per_scalar_thread_valid = true;
    }
    assert(num_addrs <= MAX_ACCESSES_PER_INSN_PER_THREAD);
    for (unsigned i = 0; i < num_addrs; i++)
      m_per_scalar_thread[n].memreqaddr[i] = addr[i];
  }
  void print_m_accessq() {
    if (accessq_empty())
      return;
    else {
      printf("Printing mem access generated\n");
      std::list<mem_access_t>::iterator it;
      for (it = m_accessq.begin(); it != m_accessq.end(); ++it) {
        printf("MEM_TXN_GEN:%s:%llx, Size:%d \n",
               mem_access_type_str(it->get_type()), it->get_addr(),
               it->get_size());
      }
    }
  }
  struct transaction_info {
    std::bitset<4> chunks;  // bitmask: 32-byte chunks accessed
    mem_access_byte_mask_t bytes;
    active_mask_t active;  // threads in this transaction

    bool test_bytes(unsigned start_bit, unsigned end_bit) {
      for (unsigned i = start_bit; i <= end_bit; i++)
        if (bytes.test(i)) return true;
      return false;
    }
  };

  void generate_mem_accesses();
  void memory_coalescing_arch(bool is_write, mem_access_type access_type);
  void memory_coalescing_arch_atomic(bool is_write,
                                     mem_access_type access_type);
  void memory_coalescing_arch_reduce_and_send(bool is_write,
                                              mem_access_type access_type,
                                              const transaction_info &info,
                                              new_addr_type addr,
                                              unsigned segment_size);

  void add_callback(unsigned lane_id,
                    void (*function)(const class inst_t *,
                                     class ptx_thread_info *),
                    const inst_t *inst, class ptx_thread_info *thread,
                    bool atomic) {
    if (!m_per_scalar_thread_valid) {
      m_per_scalar_thread.resize(m_config->warp_size);
      m_per_scalar_thread_valid = true;
      if (atomic) m_isatomic = true;
    }
    m_per_scalar_thread[lane_id].callback.function = function;
    m_per_scalar_thread[lane_id].callback.instruction = inst;
    m_per_scalar_thread[lane_id].callback.thread = thread;
  }
  void set_active(const active_mask_t &active);

  void clear_active(const active_mask_t &inactive);
  void set_not_active(unsigned lane_id);

  // accessors
  virtual void print_insn(FILE *fp) const {
    fprintf(fp, " [inst @ pc=0x%04llx] ", pc);
    for (int i = (int)m_config->warp_size - 1; i >= 0; i--)
      fprintf(fp, "%c", ((m_warp_active_mask[i]) ? '1' : '0'));
  }
  bool active(unsigned thread) const { return m_warp_active_mask.test(thread); }
  unsigned active_count() const { return m_warp_active_mask.count(); }
  unsigned issued_count() const {
    assert(m_empty == false);
    return m_warp_issued_mask.count();
  }  // for instruction counting
  bool empty() const { return m_empty; }
  unsigned warp_id() const {
    assert(!m_empty);
    return m_warp_id;
  }
  //仅被用于功能模拟。
  unsigned warp_id_func() const  // to be used in functional simulations only
  {
    return m_warp_id;
  }
  unsigned dynamic_warp_id() const {
    assert(!m_empty);
    return m_dynamic_warp_id;
  }
  bool has_callback(unsigned n) const {
    return m_warp_active_mask[n] && m_per_scalar_thread_valid &&
           (m_per_scalar_thread[n].callback.function != NULL);
  }
  new_addr_type get_addr(unsigned n) const {
    assert(m_per_scalar_thread_valid);
    return m_per_scalar_thread[n].memreqaddr[0];
  }
  //m_isatomic变量是一个布尔值，用于指示指令是否为原子操作（atomic operation）。原子操作是一种特殊的指令，
  //它可以保证在同一时间只有一个线程执行该指令，从而避免多个线程之间的竞争。
  bool isatomic() const { return m_isatomic; }

  unsigned warp_size() const { return m_config->warp_size; }

  //返回当前指令的访存操作的列表是否为空。
  bool accessq_empty() const { return m_accessq.empty(); }
  //返回当前指令的访存操作的列表的大小。
  unsigned accessq_count() const { return m_accessq.size(); }
  //返回当前指令的访存操作的列表的表尾元素。
  const mem_access_t &accessq_back() { return m_accessq.back(); }
  //弹出当前指令的访存操作的列表的表尾元素。
  void accessq_pop_back() { m_accessq.pop_back(); }

  // 一般的指令初始化时设置为 cycles = initiation_interval;，每次调用 dispatch_delay() 
  // 时，cycles 减 1，直到 cycles==0。
  // 但是有一个特例，就是当指令是访存 sahred memory 时，为了模拟 bank conflict，需要将
  // initiation_interval 设置为 total_accesses，这样
  // 
  bool dispatch_delay() {
    if (cycles > 0) cycles--;
    return cycles > 0;
  }

  //返回该条指令还剩下多少initiation_interval。
  bool has_dispatch_delay() { return cycles > 0; }

  void print(FILE *fout) const;
  unsigned get_uid() const { return m_uid; }
  unsigned long long get_streamID() const { return m_streamID; }
  unsigned get_schd_id() const { return m_scheduler_id; }
  active_mask_t get_warp_active_mask() const { return m_warp_active_mask; }

 protected:
  unsigned m_uid;
  unsigned long long m_streamID;
  bool m_empty;
  bool m_cache_hit;
  unsigned long long issue_cycle;
  //初始化时设置为cycles = initiation_interval;，每次调用dispatch_delay()时，cycles减1，直到cycles=0。
  unsigned cycles;  // used for implementing initiation interval delay
  bool m_isatomic;
  bool should_do_atomic;
  bool m_is_printf;
  unsigned m_warp_id;
  unsigned m_dynamic_warp_id;
  const core_config *m_config;
  active_mask_t m_warp_active_mask;  // dynamic active mask for timing model
                                     // (after predication)
  active_mask_t
      m_warp_issued_mask;  // active mask at issue (prior to predication test)
                           // -- for instruction counting

  struct per_thread_info {
    per_thread_info() {
      for (unsigned i = 0; i < MAX_ACCESSES_PER_INSN_PER_THREAD; i++)
        memreqaddr[i] = 0;
    }
    dram_callback_t callback;
    //MAX_ACCESSES_PER_INSN_PER_THREAD为单个线程中允许的最大访存次数。设置为8。
    //memreqaddr[]存储了单条指令的所有访存地址。
    new_addr_type
        memreqaddr[MAX_ACCESSES_PER_INSN_PER_THREAD];  // effective address,
                                                       // upto 8 different
                                                       // requests (to support
                                                       // 32B access in 8 chunks
                                                       // of 4B each)
  };
  bool m_per_scalar_thread_valid;
  //m_per_scalar_thread是线程信息的向量，每个warp有一个m_per_scalar_thread。
  std::vector<per_thread_info> m_per_scalar_thread;
  bool m_mem_accesses_created;
  // 当前指令的访存操作的列表。
  std::list<mem_access_t> m_accessq;

  unsigned m_scheduler_id;  // the scheduler that issues this inst

  // Jin: cdp support
 public:
  int m_is_cdp;

  // Ni: add boolean to indicate whether the instruction is ldgsts
  bool m_is_ldgsts;
  bool m_is_ldgdepbar;
  bool m_is_depbar;

  unsigned int m_depbar_group_no;
};

void move_warp(warp_inst_t *&dst, warp_inst_t *&src);

size_t get_kernel_code_size(class function_info *entry);
class checkpoint {
 public:
  checkpoint();
  ~checkpoint() { printf("clasfsfss destructed\n"); }

  void load_global_mem(class memory_space *temp_mem, char *f1name);
  void store_global_mem(class memory_space *mem, char *fname, char *format);
  unsigned radnom;
};
/*
 * This abstract class used as a base for functional and performance and
 * simulation, it has basic functional simulation data structures and
 * procedures.
 */
/*
这个抽象类用作功能和性能仿真的基础，它具有基本的功能仿真数据结构和过程。内核的抽象基类，用于功能和性
能模型。shader_core_ctx（在时序模型中实现SIMT Core的类）来源于这个类。抽象类core_t拥有指令执行功能
上所需的最基本的数据结构和程序。这个类是shader_core_ctx和functionalSimCore的基类，这两个类分别用于
性能模拟和纯功能模拟。core_t最重要的成员是simt_stack和ptx_thread_info类型的对象，它们在功能模拟中
用于跟踪warp分支分歧和处理线程的指令执行。
*/
class core_t {
 public:
  core_t(gpgpu_sim *gpu, kernel_info_t *kernel, unsigned warp_size,
         unsigned threads_per_shader)
      : m_gpu(gpu),
        m_kernel(kernel),
        m_simt_stack(NULL),
        m_thread(NULL),
        m_warp_size(warp_size) {
    m_warp_count = threads_per_shader / m_warp_size;
    // Handle the case where the number of threads is not a
    // multiple of the warp size
    if (threads_per_shader % m_warp_size != 0) {
      m_warp_count += 1;
    }
    assert(m_warp_count * m_warp_size > 0);
    m_thread = (ptx_thread_info **)calloc(m_warp_count * m_warp_size,
                                          sizeof(ptx_thread_info *));
    initilizeSIMTStack(m_warp_count, m_warp_size);

    for (unsigned i = 0; i < MAX_CTA_PER_SHADER; i++) {
      for (unsigned j = 0; j < MAX_BARRIERS_PER_CTA; j++) {
        reduction_storage[i][j] = 0;
      }
    }
  }
  virtual ~core_t() { free(m_thread); }
  virtual void warp_exit(unsigned warp_id) = 0;
  virtual bool warp_waiting_at_barrier(unsigned warp_id) const = 0;
  virtual void checkExecutionStatusAndUpdate(warp_inst_t &inst, unsigned t,
                                             unsigned tid) = 0;
  class gpgpu_sim *get_gpu() {
    return m_gpu;
  }
  void execute_warp_inst_t(warp_inst_t &inst, unsigned warpId = (unsigned)-1);
  bool ptx_thread_done(unsigned hw_thread_id) const;
  virtual void updateSIMTStack(unsigned warpId, warp_inst_t *inst);
  void initilizeSIMTStack(unsigned warp_count, unsigned warps_size);
  void deleteSIMTStack();
  warp_inst_t getExecuteWarp(unsigned warpId);
  void get_pdom_stack_top_info(unsigned warpId, unsigned *pc,
                               unsigned *rpc) const;
  kernel_info_t *get_kernel_info() { return m_kernel; }
  class ptx_thread_info **get_thread_info() {
    return m_thread;
  }
  unsigned get_warp_size() const { return m_warp_size; }
  void and_reduction(unsigned ctaid, unsigned barid, bool value) {
    reduction_storage[ctaid][barid] &= value;
  }
  void or_reduction(unsigned ctaid, unsigned barid, bool value) {
    reduction_storage[ctaid][barid] |= value;
  }
  void popc_reduction(unsigned ctaid, unsigned barid, bool value) {
    reduction_storage[ctaid][barid] += value;
  }
  unsigned get_reduction_value(unsigned ctaid, unsigned barid) {
    return reduction_storage[ctaid][barid];
  }

 protected:
  class gpgpu_sim *m_gpu;
  //运行在当前SIMT Core上的内核函数。
  kernel_info_t *m_kernel;
  simt_stack **m_simt_stack;  // pdom based reconvergence context for each warp
  class ptx_thread_info **m_thread;
  unsigned m_warp_size;
  //warp总数。
  unsigned m_warp_count;
  unsigned reduction_storage[MAX_CTA_PER_SHADER][MAX_BARRIERS_PER_CTA];
};

// register that can hold multiple instructions.
// 单个寄存器集合可以包含多条指令，方便模拟，而非真实硬件结构。
class register_set {
 public:
  //构造函数，用于初始化寄存器集合，寄存器集合中有num个寄存器，每个寄存器含有一条指令。
  //寄存器集合中的所有寄存器用一个向量regs保存：
  //    std::vector<warp_inst_t *> regs;
  register_set(unsigned num, const char *name) {
    for (unsigned i = 0; i < num; i++) {
      regs.push_back(new warp_inst_t());
    }
    //m_name是该寄存器集合的名字。
    m_name = name;
  }
  //获取该寄存器集合的名字。
  const char *get_name() { return m_name; }
  //遍历寄存器集合中的所有寄存器，判断是否有寄存器为空。
  bool has_free() {
    for (unsigned i = 0; i < regs.size(); i++) {
      if (regs[i]->empty()) {
        return true;
      }
    }
    return false;
  }
  //给定一个寄存器id，判断该寄存器是否为空。
  bool has_free(bool sub_core_model, unsigned reg_id) {
    // in subcore model, each sched has a one specific reg to use (based on
    // sched id)
    //在subcore模式下，每个warp调度器在寄存器集合中有一个具体的寄存器可供使用，这个寄
    //存器由调度器的m_id索引。
    if (!sub_core_model) return has_free();

    assert(reg_id < regs.size());
    return regs[reg_id]->empty();
  }
  //遍历是否存在一个非空寄存器已准备好。
  bool has_ready() {
    for (unsigned i = 0; i < regs.size(); i++) {
      if (not regs[i]->empty()) {
        return true;
      }
    }
    return false;
  }
  //给定一个寄存器id，判断该寄存器是否非空。在subcore模式下，每个warp调度器在寄存器集
  //合中有一个具体的寄存器可供使用，这个寄存器由调度器的m_id索引。
  bool has_ready(bool sub_core_model, unsigned reg_id) {
    if (!sub_core_model) return has_ready();
    assert(reg_id < regs.size());
    return (not regs[reg_id]->empty());
  }
  //获取一个非空寄存器的id。
  unsigned get_ready_reg_id() {
    // for sub core model we need to figure which reg_id has the ready warp
    // this function should only be called if has_ready() was true
    assert(has_ready());
    warp_inst_t **ready;
    ready = NULL;
    unsigned reg_id = 0;
    for (unsigned i = 0; i < regs.size(); i++) {
      if (not regs[i]->empty()) {
        if (ready and (*ready)->get_uid() < regs[i]->get_uid()) {
          // ready is oldest
        } else {
          ready = &regs[i];
          reg_id = i;
        }
      }
    }
    return reg_id;
  }
  //获取reg_id号寄存器的m_scheduler_id。
  unsigned get_schd_id(unsigned reg_id) {
    assert(not regs[reg_id]->empty());
    return regs[reg_id]->get_schd_id();
  }
  //获取一个非空寄存器，并将一条指令存入。
  void move_in(warp_inst_t *&src) {
    warp_inst_t **free = get_free();
    move_warp(*free, src);
  }
  // void copy_in( warp_inst_t* src ){
  //   src->copy_contents_to(*get_free());
  //}
  //获取一个空寄存器，并将一条指令存入。在subcore模式下，每个warp调度器在寄存器集合中
  //有一个具体的寄存器可供使用，这个寄存器由调度器的m_id索引。
  void move_in(bool sub_core_model, unsigned reg_id, warp_inst_t *&src) {
    warp_inst_t **free;
    if (!sub_core_model) {
      free = get_free();
    } else {
      assert(reg_id < regs.size());
      free = get_free(sub_core_model, reg_id);
    }
    move_warp(*free, src);
  }
  //获取一个非空寄存器，并将其指令移出到dest。
  void move_out_to(warp_inst_t *&dest) {
    warp_inst_t **ready = get_ready();
    move_warp(dest, *ready);
  }
  //依据寄存器编号reg_id，获取一个非空寄存器，并将其指令移出到dest。在subcore模式下，
  //每个warp调度器在寄存器集合中有一个具体的寄存器可供使用，这个寄存器由调度器的m_id索
  //引。
  void move_out_to(bool sub_core_model, unsigned reg_id, warp_inst_t *&dest) {
    if (!sub_core_model) {
      return move_out_to(dest);
    }
    warp_inst_t **ready = get_ready(sub_core_model, reg_id);
    assert(ready != NULL);
    move_warp(dest, *ready);
  }
  //获取一个非空寄存器，将其指令移出，并返回这条指令。
  warp_inst_t **get_ready() {
    warp_inst_t **ready;
    ready = NULL;
    for (unsigned i = 0; i < regs.size(); i++) {
      if (not regs[i]->empty()) {
        if (ready and (*ready)->get_uid() < regs[i]->get_uid()) {
          // ready is oldest
        } else {
          ready = &regs[i];
        }
      }
    }
    return ready;
  }
  //获取一个非空寄存器，将其指令移出，并返回这条指令。在subcore模式下，每个warp调度器
  //在寄存器集合中有一个具体的寄存器可供使用，这个寄存器由调度器的m_id索引。
  warp_inst_t **get_ready(bool sub_core_model, unsigned reg_id) {
    if (!sub_core_model) return get_ready();
    warp_inst_t **ready;
    ready = NULL;
    assert(reg_id < regs.size());
    if (not regs[reg_id]->empty()) ready = &regs[reg_id];
    return ready;
  }
  //打印寄存器集合中的所有寄存器。
  void print(FILE *fp) const {
    fprintf(fp, "%s : @%p\n", m_name, this);
    for (unsigned i = 0; i < regs.size(); i++) {
      fprintf(fp, "     ");
      regs[i]->print(fp);
      fprintf(fp, "\n");
    }
  }
  //遍历所有寄存器，获取一个空寄存器的id。
  warp_inst_t **get_free() {
    for (unsigned i = 0; i < regs.size(); i++) {
      if (regs[i]->empty()) {
        return &regs[i];
      }
    }
    assert(0 && "No free registers found");
    return NULL;
  }
  //遍历所有寄存器，获取一个空寄存器的地址。在subcore模式下，每个warp调度器在寄存器集
  //合中有一个具体的寄存器可供使用，这个寄存器由调度器的m_id索引。
  warp_inst_t **get_free(bool sub_core_model, unsigned reg_id) {
    // in subcore model, each sched has a one specific reg to use (based on
    // sched id)
    if (!sub_core_model) return get_free();

    assert(reg_id < regs.size());
    if (regs[reg_id]->empty()) {
      return &regs[reg_id];
    }
    assert(0 && "No free register found");
    return NULL;
  }
  //返回寄存器集合的大小。
  unsigned get_size() { return regs.size(); }

 private:
  //将寄存器集合中的所有寄存器用一个向量保存。
  std::vector<warp_inst_t *> regs;
  //该寄存器集合的名字。
  const char *m_name;
};

#endif  // #ifdef __cplusplus

#endif  // #ifndef ABSTRACT_HARDWARE_MODEL_INCLUDED
