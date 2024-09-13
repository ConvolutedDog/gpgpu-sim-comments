// Copyright (c) 2009-2011, The University of British Columbia
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

#ifndef ptx_sim_h_INCLUDED
#define ptx_sim_h_INCLUDED

#include <stdlib.h>
#include "../abstract_hardware_model.h"
#include "../tr1_hash_map.h"
#include "half.h"

#include <assert.h>
#include "opcodes.h"

#include <list>
#include <map>
#include <set>
#include <string>

#include "memory.h"

#define GCC_VERSION \
  (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)

struct param_t {
  const void *pdata;
  int type;
  size_t size;
  size_t offset;
};

#include <stack>

#include "memory.h"

using half_float::half;

/*
PTX指令中的寄存器类。
*/
union ptx_reg_t {
  //构造函数。
  ptx_reg_t() {
    bits.ms = 0;
    bits.ls = 0;
    u128.low = 0;
    u128.lowest = 0;
    u128.highest = 0;
    u128.high = 0;
    s8 = 0;
    s16 = 0;
    s32 = 0;
    s64 = 0;
    u8 = 0;
    u16 = 0;
    u64 = 0;
    f16 = 0;
    f32 = 0;
    f64 = 0;
    pred = 0;
  }
  ptx_reg_t(unsigned x) {
    bits.ms = 0;
    bits.ls = 0;
    u128.low = 0;
    u128.lowest = 0;
    u128.highest = 0;
    u128.high = 0;
    s8 = 0;
    s16 = 0;
    s32 = 0;
    s64 = 0;
    u8 = 0;
    u16 = 0;
    u64 = 0;
    f16 = 0;
    f32 = 0;
    f64 = 0;
    pred = 0;
    u32 = x;
  }
  operator unsigned int() { return u32; }
  operator unsigned short() { return u16; }
  operator unsigned char() { return u8; }
  operator unsigned long long() { return u64; }

  void mask_and(unsigned ms, unsigned ls) {
    bits.ms &= ms;
    bits.ls &= ls;
  }

  void mask_or(unsigned ms, unsigned ls) {
    bits.ms |= ms;
    bits.ls |= ls;
  }
  int get_bit(unsigned bit) {
    if (bit < 32)
      return (bits.ls >> bit) & 1;
    else
      return (bits.ms >> (bit - 32)) & 1;
  }

  signed char s8;
  signed short s16;
  signed int s32;
  signed long long s64;
  unsigned char u8;
  unsigned short u16;
  unsigned int u32;
  unsigned long long u64;
// gcc 4.7.0
#if GCC_VERSION >= 40700
  half f16;
#else
  float f16;
#endif
  float f32;
  double f64;
  struct {
    unsigned ls;
    unsigned ms;
  } bits;
  struct {
    unsigned int lowest;
    unsigned int low;
    unsigned int high;
    unsigned int highest;
  } u128;
  unsigned pred : 4;
};

class ptx_instruction;
class operand_info;
class symbol_table;
class function_info;
class ptx_thread_info;

class ptx_cta_info {
 public:
  ptx_cta_info(unsigned sm_idx, gpgpu_context *ctx);
  void add_thread(ptx_thread_info *thd);
  unsigned num_threads() const;
  void check_cta_thread_status_and_reset();
  void register_thread_exit(ptx_thread_info *thd);
  void register_deleted_thread(ptx_thread_info *thd);
  unsigned get_sm_idx() const;
  unsigned get_bar_threads() const;
  void inc_bar_threads();
  void reset_bar_threads();

 private:
  // backward pointer
  class gpgpu_context *gpgpu_ctx;
  unsigned m_bar_threads;
  unsigned long long m_uid;
  unsigned m_sm_idx;
  std::set<ptx_thread_info *> m_threads_in_cta;
  std::set<ptx_thread_info *> m_threads_that_have_exited;
  std::set<ptx_thread_info *> m_dangling_pointers;
};

class ptx_warp_info {
 public:
  ptx_warp_info();  // add get_core or something, or threads?
  unsigned get_done_threads() const;
  void inc_done_threads();
  void reset_done_threads();

 private:
  unsigned m_done_threads;
};

class symbol;

struct stack_entry {
  stack_entry() {
    m_symbol_table = NULL;
    m_func_info = NULL;
    m_PC = 0;
    m_RPC = -1;
    m_return_var_src = NULL;
    m_return_var_dst = NULL;
    m_call_uid = 0;
    m_valid = false;
  }
  stack_entry(symbol_table *s, function_info *f, unsigned pc, unsigned rpc,
              const symbol *return_var_src, const symbol *return_var_dst,
              unsigned call_uid) {
    m_symbol_table = s;
    m_func_info = f;
    m_PC = pc;
    m_RPC = rpc;
    m_return_var_src = return_var_src;
    m_return_var_dst = return_var_dst;
    m_call_uid = call_uid;
    m_valid = true;
  }

  bool m_valid;
  symbol_table *m_symbol_table;
  function_info *m_func_info;
  unsigned m_PC;
  unsigned m_RPC;
  const symbol *m_return_var_src;
  const symbol *m_return_var_dst;
  unsigned m_call_uid;
};

class ptx_version {
 public:
  ptx_version() {
    m_valid = false;
    m_ptx_version = 0;
    m_ptx_extensions = 0;
    m_sm_version_valid = false;
    m_texmode_unified = true;
    m_map_f64_to_f32 = true;
  }
  ptx_version(float ver, unsigned extensions) {
    m_valid = true;
    m_ptx_version = ver;
    m_ptx_extensions = extensions;
    m_sm_version_valid = false;
    m_texmode_unified = true;
  }
  void set_target(const char *sm_ver, const char *ext, const char *ext2) {
    assert(m_valid);
    m_sm_version_str = sm_ver;
    check_target_extension(ext);
    check_target_extension(ext2);
    sscanf(sm_ver, "%u", &m_sm_version);
    m_sm_version_valid = true;
  }
  float ver() const {
    assert(m_valid);
    return m_ptx_version;
  }
  unsigned target() const {
    assert(m_valid && m_sm_version_valid);
    return m_sm_version;
  }
  unsigned extensions() const {
    assert(m_valid);
    return m_ptx_extensions;
  }

 private:
  void check_target_extension(const char *ext) {
    if (ext) {
      if (!strcmp(ext, "texmode_independent"))
        m_texmode_unified = false;
      else if (!strcmp(ext, "texmode_unified"))
        m_texmode_unified = true;
      else if (!strcmp(ext, "map_f64_to_f32"))
        m_map_f64_to_f32 = true;
      else
        abort();
    }
  }

  bool m_valid;
  float m_ptx_version;
  unsigned m_sm_version_valid;
  std::string m_sm_version_str;
  bool m_texmode_unified;
  bool m_map_f64_to_f32;
  unsigned m_sm_version;
  unsigned m_ptx_extensions;
};

/*
时序模拟器（GPGPU-Sim）通过 ptx_thread_info 类与功能模拟器（CUDA-Sim）连接。 m_thread 成员变量是
SIMT Core类 shader_core_ctx 中 ptx_thread_info 的数组，维护该SIMT Core中所有活动线程的功能状态。
时序模型通过 warp_inst_t 类与功能模型进行通信， warp_inst_t 类表示一个指令的动态实例，正在由一个
warp执行。

时序模型在仿真的以下三个阶段与功能模型进行交流：
1. 解码
    在 shader_core_ctx::decode() 的解码阶段，时序模拟器从功能模拟器中获得指令，给定一个PC。这是通
    过调用 ptx_fetch_inst 函数完成的。
2. 指令执行
    1)功能执行：时序模型通过调用 ptx_thread_info 类的 ptx_exec_inst 方法将线程的功能状态提前一个
      指令。这是在 core_t::execute_warp_inst_t 内完成的。时序模拟器传递要执行的指令的动态实例，而
      功能模型则相应地推进线程的状态。
    2)SIMT堆栈更新：在功能上执行了一条warp的指令后，时序模型通过向功能模型请求更新SIMT堆栈中的下一
      个PC。这发生在 simt_stack::update 里面。
    3)原子回调：如果指令是一个原子操作，那么指令的功能执行就不会在 core_t::execute_warp_inst_t 中
      发生。相反，在功能执行阶段，功能模拟器通过调用 warp_inst_t::add_callback 在 warp_inst_t 对
      象中存储一个指向该原子指令的指针。时序模拟器在请求离开二级缓存时执行这个回调函数。
3. 启动线程块
    当新的线程块在 shader_core_ctx::issue_block2core 中启动时，时序模拟器通过调用功能模型方法 
    ptx_sim_init_thread 初始化每个线程的功能状态。此外，时序模型还通过从功能模型中获取起始PC来初始
    化SIMT堆栈和warp状态。

ptx_thread_info对象包含单个标量线程（OpenCL中的work item）的功能仿真状态。这包括以下内容：
    a. 寄存器值存储
    b. 本地内存存储（OpenCL中的私有内存）
    c. 共享内存存储（OpenCL中的本地内存）。注意，同一线程块/Work Wrap的所有标量线程都
    d. 会访问相同的共享内存存储。
    e. 程序计数器（PC）
    f. 调用堆栈
    g. 线程ID（网格启动中的软件ID，以及表明它在时序模型中占据哪个硬件线程槽的硬件ID)

在函数仿真中使用的动态数据值的存储使用了不同的寄存器和内存空间类。寄存器的值包含在 ptx_thread_info::
m_regs 中，这是一个从符号指针到C语言联合体 ptx_reg_t 的映射。寄存器的访问使用方法 ptx_thread_info::
get_operand_value() ，它使用 operand_info 作为输入。对于内存操作数，该方法返回内存操作数的有效地址。
编程模型中的每个内存空间都包含在一个类型为 memory_space 的对象中。GPU中所有线程可见的内存空间都包含在 
gpgpu_t 中，并通过 ptx_thread_info 中的接口进行访问（例如，ptx_thread_info::get_global_memory）。
*/
class ptx_thread_info {
 public:
  //析构函数。
  ~ptx_thread_info();
  //构造函数。
  ptx_thread_info(kernel_info_t &kernel);

  void init(gpgpu_t *gpu, core_t *core, unsigned sid, unsigned cta_id,
            unsigned wid, unsigned tid, bool fsim) {
    m_gpu = gpu;
    m_core = core;
    //线程所在SM的id。
    m_hw_sid = sid;
    //线程所在CTA的id。
    m_hw_ctaid = cta_id;
    //线程所在warp的id。
    m_hw_wid = wid;
    //线程的id。
    m_hw_tid = tid;
    //功能模拟。
    m_functionalSimulationMode = fsim;
  }
  //解码阶段，定时模拟器从给定PC的函数模拟器获得指令。这是通过调用ptx_fetch_inst函数完成的。
  void ptx_fetch_inst(inst_t &inst) const;
  void ptx_exec_inst(warp_inst_t &inst, unsigned lane_id);

  const ptx_version &get_ptx_version() const;
  void set_reg(const symbol *reg, const ptx_reg_t &value);
  void print_reg_thread(char *fname);
  void resume_reg_thread(char *fname, symbol_table *symtab);
  ptx_reg_t get_reg(const symbol *reg);
  ptx_reg_t get_operand_value(const operand_info &op, operand_info dstInfo,
                              unsigned opType, ptx_thread_info *thread,
                              int derefFlag);
  void set_operand_value(const operand_info &dst, const ptx_reg_t &data,
                         unsigned type, ptx_thread_info *thread,
                         const ptx_instruction *pI);
  void set_operand_value(const operand_info &dst, const ptx_reg_t &data,
                         unsigned type, ptx_thread_info *thread,
                         const ptx_instruction *pI, int overflow, int carry);
  void get_vector_operand_values(const operand_info &op, ptx_reg_t *ptx_regs,
                                 unsigned num_elements);
  void set_vector_operand_values(const operand_info &dst,
                                 const ptx_reg_t &data1, const ptx_reg_t &data2,
                                 const ptx_reg_t &data3,
                                 const ptx_reg_t &data4);
  void set_wmma_vector_operand_values(
      const operand_info &dst, const ptx_reg_t &data1, const ptx_reg_t &data2,
      const ptx_reg_t &data3, const ptx_reg_t &data4, const ptx_reg_t &data5,
      const ptx_reg_t &data6, const ptx_reg_t &data7, const ptx_reg_t &data8);

  function_info *func_info() { return m_func_info; }
  void print_insn(unsigned pc, FILE *fp) const;
  void set_info(function_info *func);
  unsigned get_uid() const { return m_uid; }

  dim3 get_ctaid() const { return m_ctaid; }
  dim3 get_tid() const { return m_tid; }
  dim3 get_ntid() const { return m_ntid; }
  class gpgpu_sim *get_gpu() {
    return (gpgpu_sim *)m_gpu;
  }
  unsigned get_hw_tid() const { return m_hw_tid; }
  unsigned get_hw_ctaid() const { return m_hw_ctaid; }
  unsigned get_hw_wid() const { return m_hw_wid; }
  unsigned get_hw_sid() const { return m_hw_sid; }
  core_t *get_core() { return m_core; }

  unsigned get_icount() const { return m_icount; }
  void set_valid() { m_valid = true; }
  addr_t last_eaddr() const { return m_last_effective_address; }
  memory_space_t last_space() const { return m_last_memory_space; }
  dram_callback_t last_callback() const { return m_last_dram_callback; }
  unsigned long long get_cta_uid() { return m_cta_info->get_sm_idx(); }

  void set_single_thread_single_block() {
    m_ntid.x = 1;
    m_ntid.y = 1;
    m_ntid.z = 1;
    m_ctaid.x = 0;
    m_ctaid.y = 0;
    m_ctaid.z = 0;
    m_tid.x = 0;
    m_tid.y = 0;
    m_tid.z = 0;
    m_nctaid.x = 1;
    m_nctaid.y = 1;
    m_nctaid.z = 1;
    m_gridid = 0;
    m_valid = true;
  }
  void set_tid(dim3 tid) { m_tid = tid; }
  void cpy_tid_to_reg(dim3 tid);
  void set_ctaid(dim3 ctaid) { m_ctaid = ctaid; }
  void set_ntid(dim3 tid) { m_ntid = tid; }
  void set_nctaid(dim3 cta_size) { m_nctaid = cta_size; }

  unsigned get_builtin(int builtin_id, unsigned dim_mod);

  void set_done();
  bool is_done() { return m_thread_done; }
  unsigned donecycle() const { return m_cycle_done; }

  unsigned next_instr() {
    m_icount++;
    m_branch_taken = false;
    return m_PC;
  }
  bool branch_taken() const { return m_branch_taken; }
  unsigned get_pc() const { return m_PC; }
  void set_npc(unsigned npc) { m_NPC = npc; }
  void set_npc(const function_info *f);
  void callstack_push(unsigned npc, unsigned rpc, const symbol *return_var_src,
                      const symbol *return_var_dst, unsigned call_uid);
  bool callstack_pop();
  void callstack_push_plus(unsigned npc, unsigned rpc,
                           const symbol *return_var_src,
                           const symbol *return_var_dst, unsigned call_uid);
  bool callstack_pop_plus();
  void dump_callstack() const;
  std::string get_location() const;
  const ptx_instruction *get_inst() const;
  const ptx_instruction *get_inst(addr_t pc) const;
  bool rpc_updated() const { return m_RPC_updated; }
  bool last_was_call() const { return m_last_was_call; }
  unsigned get_rpc() const { return m_RPC; }
  void clearRPC() {
    m_RPC = -1;
    m_RPC_updated = false;
    m_last_was_call = false;
  }
  unsigned get_return_PC() { return m_callstack.back().m_PC; }
  void update_pc() { m_PC = m_NPC; }
  void dump_regs(FILE *fp);
  void dump_modifiedregs(FILE *fp);
  void clear_modifiedregs() {
    m_debug_trace_regs_modified.back().clear();
    m_debug_trace_regs_read.back().clear();
  }
  function_info *get_finfo() { return m_func_info; }
  const function_info *get_finfo() const { return m_func_info; }
  void push_breakaddr(const operand_info &breakaddr);
  const operand_info &pop_breakaddr();
  void enable_debug_trace() { m_enable_debug_trace = true; }
  unsigned get_local_mem_stack_pointer() const {
    return m_local_mem_stack_pointer;
  }

  memory_space *get_global_memory() { return m_gpu->get_global_memory(); }
  memory_space *get_tex_memory() { return m_gpu->get_tex_memory(); }
  memory_space *get_surf_memory() { return m_gpu->get_surf_memory(); }
  memory_space *get_param_memory() { return m_kernel.get_param_memory(); }
  const gpgpu_functional_sim_config &get_config() const {
    return m_gpu->get_config();
  }
  bool isInFunctionalSimulationMode() { return m_functionalSimulationMode; }
  void exitCore() {
    // m_core is not used in case of functional simulation mode
    if (!m_functionalSimulationMode) m_core->warp_exit(m_hw_wid);
  }

  void registerExit() { m_cta_info->register_thread_exit(this); }
  unsigned get_reduction_value(unsigned ctaid, unsigned barid) {
    return m_core->get_reduction_value(ctaid, barid);
  }
  void and_reduction(unsigned ctaid, unsigned barid, bool value) {
    m_core->and_reduction(ctaid, barid, value);
  }
  void or_reduction(unsigned ctaid, unsigned barid, bool value) {
    m_core->or_reduction(ctaid, barid, value);
  }
  void popc_reduction(unsigned ctaid, unsigned barid, bool value) {
    m_core->popc_reduction(ctaid, barid, value);
  }

  // Jin: get corresponding kernel grid for CDP purpose
  kernel_info_t &get_kernel() { return m_kernel; }

 public:
  addr_t m_last_effective_address;
  bool m_branch_taken;
  memory_space_t m_last_memory_space;
  dram_callback_t m_last_dram_callback;
  memory_space *m_shared_mem;
  memory_space *m_sstarr_mem;
  memory_space *m_local_mem;
  ptx_warp_info *m_warp_info;
  ptx_cta_info *m_cta_info;
  ptx_reg_t m_last_set_operand_value;

 private:
  bool m_functionalSimulationMode;
  unsigned m_uid;
  kernel_info_t &m_kernel;
  core_t *m_core;
  gpgpu_t *m_gpu;
  bool m_valid;
  dim3 m_ntid;
  dim3 m_tid;
  dim3 m_nctaid;
  dim3 m_ctaid;
  unsigned m_gridid;
  bool m_thread_done;
  unsigned m_hw_sid;
  unsigned m_hw_tid;
  unsigned m_hw_wid;
  unsigned m_hw_ctaid;

  unsigned m_icount;
  unsigned m_PC;
  unsigned m_NPC;
  unsigned m_RPC;
  bool m_RPC_updated;
  bool m_last_was_call;
  unsigned m_cycle_done;

  int m_barrier_num;
  bool m_at_barrier;

  symbol_table *m_symbol_table;
  function_info *m_func_info;

  std::list<stack_entry> m_callstack;
  unsigned m_local_mem_stack_pointer;

  typedef tr1_hash_map<const symbol *, ptx_reg_t> reg_map_t;
  std::list<reg_map_t> m_regs;
  std::list<reg_map_t> m_debug_trace_regs_modified;
  std::list<reg_map_t> m_debug_trace_regs_read;
  bool m_enable_debug_trace;

  std::stack<class operand_info, std::vector<operand_info> > m_breakaddrs;
};

addr_t generic_to_local(unsigned smid, unsigned hwtid, addr_t addr);
addr_t generic_to_shared(unsigned smid, addr_t addr);
addr_t generic_to_global(addr_t addr);
addr_t local_to_generic(unsigned smid, unsigned hwtid, addr_t addr);
addr_t shared_to_generic(unsigned smid, addr_t addr);
addr_t global_to_generic(addr_t addr);
bool isspace_local(unsigned smid, unsigned hwtid, addr_t addr);
bool isspace_shared(unsigned smid, addr_t addr);
bool isspace_global(addr_t addr);
memory_space_t whichspace(addr_t addr);

extern unsigned g_ptx_thread_info_uid_next;

#endif
