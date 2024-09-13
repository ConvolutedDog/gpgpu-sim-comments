// Copyright (c) 2009-2011, Tor M. Aamodt, Ali Bakhoda, Wilson W.L. Fung,
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

#ifndef ptx_ir_INCLUDED
#define ptx_ir_INCLUDED

#include "../abstract_hardware_model.h"

#include <assert.h>
#include <cstdlib>
#include <cstring>
#include <list>
#include <map>
#include <string>
#include <vector>

//#include "ptx.tab.h"
#include "ptx_sim.h"

#include "memory.h"

class gpgpu_context;

/*
type_info_key 包含关于数据对象类型的信息（在指令解释期间使用）。
*/
class type_info_key {
 public:
  //构造函数。
  type_info_key() {
    //m_is_non_arch_reg 的意思详见 void set_is_non_arch_reg() 的注释。
    m_is_non_arch_reg = false;
    //标志是否已经完成 type_info_key 对象中的各成员变量的初始化。
    m_init = false;
  }
  //构造函数，以及初始化成员变量。六个参数为：
  //  1.memory_space_t space_spec：GPGPU-Sim设置的存储空间的类型有：
  //    enum _memory_space_t {
  //      //a. 未定义的空间类型
  //      undefined_space = 0,
  //      //b. 寄存器
  //      reg_space,
  //      //c. local memory
  //      local_space,
  //      //d. shared memory
  //      shared_space,
  //      //e. 貌似是 shared static array，其访存的行为与shared memory一致，可以认为其是shared 
  //      //   memory的一种
  //      sstarr_space,
  //      //f. 通用参数存储
  //      param_space_unclassified,
  //      //g. 对内核中的所有线程：全局性的，只读的
  //      param_space_kernel, // global to all threads in a kernel : read-only
  //      //h. 对某个线程：私有的，可读写的
  //      param_space_local,  // local to a thread : read-writable
  //      //i. 常量缓存
  //      const_space,
  //      //j. 纹理缓存
  //      tex_space,
  //      //k. 渲染曲面 // render surfaces 
  //      surf_space,
  //      //l. 全局存储
  //      global_space,
  //      //m. 通用存储
  //      generic_space,
  //      //n. 指令存储
  //      instruction_space
  //    };
  //  2.int scalar_type_spec：在ptx_tab.h定义的yytokentype中枚举，指的是标量数据类型。例如：
  //        enum yytokentype
  //        {
  //          ......
  //          U8_TYPE = 307,
  //          U16_TYPE = 308,
  //          U32_TYPE = 309,
  //          U64_TYPE = 310,
  //          F16_TYPE = 311,
  //          F32_TYPE = 312,
  //          F64_TYPE = 313,
  //          PRED_TYPE = 321,
  //          ......
  //        };
  //  3.int vector_spec：
  //  4.int alignment_spec：
  //  5.int extern_spec：
  //  6.int array_dim：
  type_info_key(memory_space_t space_spec, int scalar_type_spec,
                int vector_spec, int alignment_spec, int extern_spec,
                int array_dim) {
    //m_is_non_arch_reg 的意思详见 void set_is_non_arch_reg() 的注释。 
    m_is_non_arch_reg = false;
    m_init = true;
    m_space_spec = space_spec;
    m_scalar_type_spec = scalar_type_spec;
    m_vector_spec = vector_spec;
    m_alignment_spec = alignment_spec;
    m_extern_spec = extern_spec;
    m_array_dim = array_dim;
    m_is_function = 0;
  }
  //设置一个类型信息关键字（type_info_key）是否是一个函数。它可以用于检测函数是否已被定义，以及函数的参
  //数类型是否正确。
  void set_is_func() {
    //检测当前type_info_key对象是否已经初始化过。
    assert(!m_init);
    m_init = true;
    m_space_spec = undefined_space;
    m_scalar_type_spec = 0;
    m_vector_spec = 0;
    m_alignment_spec = 0;
    m_extern_spec = 0;
    m_array_dim = 0;
    m_is_function = 1;
  }

  void set_array_dim(int array_dim) { m_array_dim = array_dim; }
  int get_array_dim() const {
    assert(m_init);
    return m_array_dim;
  }

  //在处理PTX指令时，有可能会遇到一条指令中存在寄存器："_"（暂时我跑的代码里还没遇到过这种带有"_"寄存
  //器的指令）。m_is_non_arch_reg 用于支持表示为"_"的寄存器的初始设置代码。当未读取或写入指令操作数
  //时，使用此寄存器。然而，解析器必须将其识别为合法的寄存器，但我们不想将其传递给微体系结构寄存器，传递
  //给性能模拟器。为此，我们向符号表中添加一个符号，但将其标记为non_arch_reg，这样不会影响性能模拟。并
  //且，在这里，将带有"_"寄存器的指令中的"_"设置为null_key（null_key就是当前类的一个对象），并设置
  //null_key.set_is_non_arch_reg()。
  void set_is_non_arch_reg() { m_is_non_arch_reg = true; }
  
  //m_is_non_arch_reg 的意思详见 void set_is_non_arch_reg() 的注释。
  bool is_non_arch_reg() const { return m_is_non_arch_reg; }
  bool is_reg() const { return m_space_spec == reg_space; }
  bool is_param_kernel() const { return m_space_spec == param_space_kernel; }
  bool is_param_local() const { return m_space_spec == param_space_local; }
  bool is_param_unclassified() const {
    return m_space_spec == param_space_unclassified;
  }
  bool is_global() const { return m_space_spec == global_space; }
  bool is_local() const { return m_space_spec == local_space; }
  bool is_shared() const { return m_space_spec == shared_space; }
  bool is_const() const { return m_space_spec.get_type() == const_space; }
  bool is_tex() const { return m_space_spec == tex_space; }
  bool is_func_addr() const { return m_is_function ? true : false; }
  int scalar_type() const { return m_scalar_type_spec; }
  int get_alignment_spec() const { return m_alignment_spec; }
  unsigned type_decode(size_t &size, int &t) const;
  static unsigned type_decode(int type, size_t &size, int &t);
  memory_space_t get_memory_space() const { return m_space_spec; }

 private:
  bool m_init;
  memory_space_t m_space_spec;
  int m_scalar_type_spec;
  int m_vector_spec;
  int m_alignment_spec;
  int m_extern_spec;
  int m_array_dim;
  int m_is_function;
  //m_is_non_arch_reg 的意思详见 void set_is_non_arch_reg() 的注释。
  bool m_is_non_arch_reg;

  friend struct type_info_key_compare;
};

class symbol_table;

struct type_info_key_compare {
  bool operator()(const type_info_key &a, const type_info_key &b) const {
    assert(a.m_init && b.m_init);
    if (a.m_space_spec < b.m_space_spec) return true;
    if (a.m_scalar_type_spec < b.m_scalar_type_spec) return true;
    if (a.m_vector_spec < b.m_vector_spec) return true;
    if (a.m_alignment_spec < b.m_alignment_spec) return true;
    if (a.m_extern_spec < b.m_extern_spec) return true;
    if (a.m_array_dim < b.m_array_dim) return true;
    if (a.m_is_function < b.m_is_function) return true;

    return false;
  }
};

/*
type_info 包含关于数据对象类型的信息（在指令解释期间使用）。
*/
class type_info {
 public:
  type_info(symbol_table *scope, type_info_key t) { m_type_info = t; }
  const type_info_key &get_key() const { return m_type_info; }

 private:
  symbol_table *m_scope;
  type_info_key m_type_info;
};

enum operand_type {
  reg_t,
  vector_t,
  builtin_t,
  address_t,
  memory_t,
  float_op_t,
  double_op_t,
  int_t,
  unsigned_t,
  symbolic_t,
  label_t,
  v_reg_t,
  v_float_op_t,
  v_double_op_t,
  v_int_t,
  v_unsigned_t,
  undef_t
};

class operand_info;

class symbol {
 public:
  symbol(const char *name, const type_info *type, const char *location,
         unsigned size, gpgpu_context *ctx) {
    gpgpu_ctx = ctx;
    m_uid = get_uid();
    m_name = name;
    m_decl_location = location;
    m_type = type;
    m_size = size;
    m_address_valid = false;
    m_is_label = false;
    m_is_shared = false;
    m_is_const = false;
    m_is_global = false;
    m_is_local = false;
    m_is_param_local = false;
    m_is_param_kernel = false;
    m_is_tex = false;
    m_is_func_addr = false;
    m_reg_num_valid = false;
    m_function = NULL;
    m_reg_num = (unsigned)-1;
    m_arch_reg_num = (unsigned)-1;
    m_address = (unsigned)-1;
    m_initializer.clear();
    if (type) m_is_shared = type->get_key().is_shared();
    if (type) m_is_const = type->get_key().is_const();
    if (type) m_is_global = type->get_key().is_global();
    if (type) m_is_local = type->get_key().is_local();
    if (type) m_is_param_local = type->get_key().is_param_local();
    if (type) m_is_param_kernel = type->get_key().is_param_kernel();
    if (type) m_is_tex = type->get_key().is_tex();
    if (type) m_is_func_addr = type->get_key().is_func_addr();
  }
  unsigned get_size_in_bytes() const { return m_size; }
  const std::string &name() const { return m_name; }
  const std::string &decl_location() const { return m_decl_location; }
  const type_info *type() const { return m_type; }
  addr_t get_address() const {
    assert(m_is_label ||
           !m_type->get_key().is_reg());  // todo : other assertions
    assert(m_address_valid);
    return m_address;
  }
  function_info *get_pc() const { return m_function; }
  void set_regno(unsigned regno, unsigned arch_regno) {
    m_reg_num_valid = true;
    m_reg_num = regno;
    m_arch_reg_num = arch_regno;
  }

  void set_address(addr_t addr) {
    m_address_valid = true;
    m_address = addr;
  }
  void set_label_address(addr_t addr) {
    m_address_valid = true;
    m_address = addr;
    m_is_label = true;
  }
  void set_function(function_info *func) {
    m_function = func;
    m_is_func_addr = true;
  }

  bool is_label() const { return m_is_label; }
  bool is_shared() const { return m_is_shared; }
  bool is_sstarr() const { return m_is_sstarr; }
  bool is_const() const { return m_is_const; }
  bool is_global() const { return m_is_global; }
  bool is_local() const { return m_is_local; }
  bool is_param_local() const { return m_is_param_local; }
  bool is_param_kernel() const { return m_is_param_kernel; }
  bool is_tex() const { return m_is_tex; }
  bool is_func_addr() const { return m_is_func_addr; }
  bool is_reg() const {
    if (m_type == NULL) {
      return false;
    }
    return m_type->get_key().is_reg();
  }
  bool is_non_arch_reg() const {
    if (m_type == NULL) {
      return false;
    }
    return m_type->get_key().is_non_arch_reg();
  }

  void add_initializer(const std::list<operand_info> &init);
  bool has_initializer() const { return m_initializer.size() > 0; }
  std::list<operand_info> get_initializer() const { return m_initializer; }
  unsigned reg_num() const {
    assert(m_reg_num_valid);
    return m_reg_num;
  }
  unsigned arch_reg_num() const {
    assert(m_reg_num_valid);
    return m_arch_reg_num;
  }
  void print_info(FILE *fp) const;
  unsigned uid() const { return m_uid; }

 private:
  gpgpu_context *gpgpu_ctx;
  unsigned get_uid();
  unsigned m_uid;
  const type_info *m_type;
  unsigned m_size;  // in bytes
  std::string m_name;
  std::string m_decl_location;

  unsigned m_address;
  function_info *m_function;  // used for function symbols

  bool m_address_valid;
  bool m_is_label;
  bool m_is_shared;
  bool m_is_sstarr;
  bool m_is_const;
  bool m_is_global;
  bool m_is_local;
  bool m_is_param_local;
  bool m_is_param_kernel;
  bool m_is_tex;
  bool m_is_func_addr;
  unsigned m_reg_num;
  unsigned m_arch_reg_num;
  bool m_reg_num_valid;

  std::list<operand_info> m_initializer;
};

/*
符号表。包含PTX中一个内存位置的文本表示（e.g., "%r2", "input_data", etc...）到一个包含数据类型和位置
信息的符号对象的映射。
*/
class symbol_table {
 public:
  symbol_table();
  symbol_table(const char *scope_name, unsigned entry_point,
               symbol_table *parent, gpgpu_context *ctx);
  void set_name(const char *name);
  const ptx_version &get_ptx_version() const;
  unsigned get_sm_target() const;
  void set_ptx_version(float ver, unsigned ext);
  void set_sm_target(const char *target, const char *ext, const char *ext2);
  symbol *lookup(const char *identifier);
  std::string get_scope_name() const { return m_scope_name; }
  symbol *add_variable(const char *identifier, const type_info *type,
                       unsigned size, const char *filename, unsigned line);
  void add_function(function_info *func, const char *filename,
                    unsigned linenumber);
  bool add_function_decl(const char *name, int entry_point,
                         function_info **func_info,
                         symbol_table **symbol_table);
  function_info *lookup_function(std::string name);
  type_info *add_type(memory_space_t space_spec, int scalar_type_spec,
                      int vector_spec, int alignment_spec, int extern_spec);
  type_info *add_type(function_info *func);
  type_info *get_array_type(type_info *base_type, unsigned array_dim);
  void set_label_address(const symbol *label, unsigned addr);
  unsigned next_reg_num() { return ++m_reg_allocator; }
  addr_t get_shared_next() { return m_shared_next; }
  addr_t get_sstarr_next() { return m_sstarr_next; }
  addr_t get_global_next() { return m_global_next; }
  addr_t get_local_next() { return m_local_next; }
  addr_t get_tex_next() { return m_tex_next; }
  void alloc_shared(unsigned num_bytes) { m_shared_next += num_bytes; }
  void alloc_sstarr(unsigned num_bytes) { m_sstarr_next += num_bytes; }
  void alloc_global(unsigned num_bytes) { m_global_next += num_bytes; }
  void alloc_local(unsigned num_bytes) { m_local_next += num_bytes; }
  void alloc_tex(unsigned num_bytes) { m_tex_next += num_bytes; }

  typedef std::list<symbol *>::iterator iterator;

  iterator global_iterator_begin() { return m_globals.begin(); }
  iterator global_iterator_end() { return m_globals.end(); }

  iterator const_iterator_begin() { return m_consts.begin(); }
  iterator const_iterator_end() { return m_consts.end(); }

  void dump();

  // Jin: handle instruction group for cdp
  symbol_table *start_inst_group();
  symbol_table *end_inst_group();

  // backward pointer
  class gpgpu_context *gpgpu_ctx;

 private:
  unsigned m_reg_allocator;
  unsigned m_shared_next;
  unsigned m_sstarr_next;
  unsigned m_const_next;
  unsigned m_global_next;
  unsigned m_local_next;
  unsigned m_tex_next;

  symbol_table *m_parent;
  ptx_version m_ptx_version;
  std::string m_scope_name;
  std::map<std::string, symbol *>
      m_symbols;  // map from name of register to pointers to the registers
  std::map<type_info_key, type_info *, type_info_key_compare> m_types;
  std::list<symbol *> m_globals;
  std::list<symbol *> m_consts;
  std::map<std::string, function_info *> m_function_info_lookup;
  std::map<std::string, symbol_table *> m_function_symtab_lookup;

  // Jin: handle instruction group for cdp
  unsigned m_inst_group_id;
  std::map<std::string, symbol_table *> m_inst_group_symtab;
};

/*
一个包含指令源操作数的封装类，可以是寄存器标识符、内存操作数（包括置换模式信息）或即时操作数。
*/
class operand_info {
 public:
  //构造函数。
  operand_info(gpgpu_context *ctx) {
    init(ctx);
    m_is_non_arch_reg = false;
    m_addr_space = undefined_space;
    m_operand_lohi = 0;
    m_double_operand_type = 0;
    m_operand_neg = false;
    m_const_mem_offset = 0;
    m_uid = get_uid();
    m_valid = false;
    m_immediate_address = false;
    m_addr_offset = 0;
    m_value.m_symbolic = NULL;
  }
  operand_info(const symbol *addr, gpgpu_context *ctx) {
    init(ctx);
    m_is_non_arch_reg = false;
    m_addr_space = undefined_space;
    m_operand_lohi = 0;
    m_double_operand_type = 0;
    m_operand_neg = false;
    m_const_mem_offset = 0;
    m_uid = get_uid();
    m_valid = true;
    if (addr->is_label()) {
      m_type = label_t;
    } else if (addr->is_shared()) {
      m_type = symbolic_t;
    } else if (addr->is_const()) {
      m_type = symbolic_t;
    } else if (addr->is_global()) {
      m_type = symbolic_t;
    } else if (addr->is_local()) {
      m_type = symbolic_t;
    } else if (addr->is_param_local()) {
      m_type = symbolic_t;
    } else if (addr->is_param_kernel()) {
      m_type = symbolic_t;
    } else if (addr->is_tex()) {
      m_type = symbolic_t;
    } else if (addr->is_func_addr()) {
      m_type = symbolic_t;
    } else if (!addr->is_reg()) {
      m_type = symbolic_t;
    } else {
      m_type = reg_t;
    }

    m_is_non_arch_reg = addr->is_non_arch_reg();
    m_value.m_symbolic = addr;
    m_addr_offset = 0;
    m_vector = false;
    m_neg_pred = false;
    m_is_return_var = false;
    m_immediate_address = false;
  }
  operand_info(const symbol *addr1, const symbol *addr2, gpgpu_context *ctx) {
    init(ctx);
    m_is_non_arch_reg = false;
    m_addr_space = undefined_space;
    m_operand_lohi = 0;
    m_double_operand_type = 0;
    m_operand_neg = false;
    m_const_mem_offset = 0;
    m_uid = get_uid();
    m_valid = true;
    m_type = memory_t;
    m_value.m_vector_symbolic = new const symbol *[8];
    m_value.m_vector_symbolic[0] = addr1;
    m_value.m_vector_symbolic[1] = addr2;
    m_value.m_vector_symbolic[2] = NULL;
    m_value.m_vector_symbolic[3] = NULL;
    m_value.m_vector_symbolic[4] = NULL;
    m_value.m_vector_symbolic[5] = NULL;
    m_value.m_vector_symbolic[6] = NULL;
    m_value.m_vector_symbolic[7] = NULL;
    m_addr_offset = 0;
    m_vector = false;
    m_neg_pred = false;
    m_is_return_var = false;
    m_immediate_address = false;
  }
  operand_info(int builtin_id, int dim_mod, gpgpu_context *ctx) {
    init(ctx);
    m_is_non_arch_reg = false;
    m_addr_space = undefined_space;
    m_operand_lohi = 0;
    m_double_operand_type = 0;
    m_operand_neg = false;
    m_const_mem_offset = 0;
    m_uid = get_uid();
    m_valid = true;
    m_vector = false;
    m_type = builtin_t;
    m_value.m_int = builtin_id;
    m_addr_offset = dim_mod;
    m_neg_pred = false;
    m_is_return_var = false;
    m_immediate_address = false;
  }
  operand_info(const symbol *addr, int offset, gpgpu_context *ctx) {
    init(ctx);
    m_is_non_arch_reg = false;
    m_addr_space = undefined_space;
    m_operand_lohi = 0;
    m_double_operand_type = 0;
    m_operand_neg = false;
    m_const_mem_offset = 0;
    m_uid = get_uid();
    m_valid = true;
    m_vector = false;
    m_type = address_t;
    m_value.m_symbolic = addr;
    m_addr_offset = offset;
    m_neg_pred = false;
    m_is_return_var = false;
    m_immediate_address = false;
  }
  operand_info(unsigned x, gpgpu_context *ctx) {
    init(ctx);
    m_is_non_arch_reg = false;
    m_addr_space = undefined_space;
    m_operand_lohi = 0;
    m_double_operand_type = 0;
    m_operand_neg = false;
    m_const_mem_offset = 0;
    m_uid = get_uid();
    m_valid = true;
    m_vector = false;
    m_type = unsigned_t;
    m_value.m_unsigned = x;
    m_addr_offset = x;
    m_neg_pred = false;
    m_is_return_var = false;
    m_immediate_address = true;
  }
  operand_info(int x, gpgpu_context *ctx) {
    init(ctx);
    m_is_non_arch_reg = false;
    m_addr_space = undefined_space;
    m_operand_lohi = 0;
    m_double_operand_type = 0;
    m_operand_neg = false;
    m_const_mem_offset = 0;
    m_uid = get_uid();
    m_valid = true;
    m_vector = false;
    m_type = int_t;
    m_value.m_int = x;
    m_addr_offset = 0;
    m_neg_pred = false;
    m_is_return_var = false;
    m_immediate_address = false;
  }
  operand_info(float x, gpgpu_context *ctx) {
    init(ctx);
    m_is_non_arch_reg = false;
    m_addr_space = undefined_space;
    m_operand_lohi = 0;
    m_double_operand_type = 0;
    m_operand_neg = false;
    m_const_mem_offset = 0;
    m_uid = get_uid();
    m_valid = true;
    m_vector = false;
    m_type = float_op_t;
    m_value.m_float = x;
    m_addr_offset = 0;
    m_neg_pred = false;
    m_is_return_var = false;
    m_immediate_address = false;
  }
  operand_info(double x, gpgpu_context *ctx) {
    init(ctx);
    m_is_non_arch_reg = false;
    m_addr_space = undefined_space;
    m_operand_lohi = 0;
    m_double_operand_type = 0;
    m_operand_neg = false;
    m_const_mem_offset = 0;
    m_uid = get_uid();
    m_valid = true;
    m_vector = false;
    m_type = double_op_t;
    m_value.m_double = x;
    m_addr_offset = 0;
    m_neg_pred = false;
    m_is_return_var = false;
    m_immediate_address = false;
  }
  operand_info(const symbol *s1, const symbol *s2, const symbol *s3,
               const symbol *s4, gpgpu_context *ctx) {
    init(ctx);
    m_is_non_arch_reg = false;
    m_addr_space = undefined_space;
    m_operand_lohi = 0;
    m_double_operand_type = 0;
    m_operand_neg = false;
    m_const_mem_offset = 0;
    m_uid = get_uid();
    m_valid = true;
    m_vector = true;
    m_type = vector_t;
    m_value.m_vector_symbolic = new const symbol *[8];
    m_value.m_vector_symbolic[0] = s1;
    m_value.m_vector_symbolic[1] = s2;
    m_value.m_vector_symbolic[2] = s3;
    m_value.m_vector_symbolic[3] = s4;
    m_value.m_vector_symbolic[4] = NULL;
    m_value.m_vector_symbolic[5] = NULL;
    m_value.m_vector_symbolic[6] = NULL;
    m_value.m_vector_symbolic[7] = NULL;
    m_addr_offset = 0;
    m_neg_pred = false;
    m_is_return_var = false;
    m_immediate_address = false;
  }
  operand_info(const symbol *s1, const symbol *s2, const symbol *s3,
               const symbol *s4, const symbol *s5, const symbol *s6,
               const symbol *s7, const symbol *s8, gpgpu_context *ctx) {
    init(ctx);
    m_is_non_arch_reg = false;
    m_addr_space = undefined_space;
    m_operand_lohi = 0;
    m_double_operand_type = 0;
    m_operand_neg = false;
    m_const_mem_offset = 0;
    m_uid = get_uid();
    m_valid = true;
    m_vector = true;
    m_type = vector_t;
    m_value.m_vector_symbolic = new const symbol *[8];
    m_value.m_vector_symbolic[0] = s1;
    m_value.m_vector_symbolic[1] = s2;
    m_value.m_vector_symbolic[2] = s3;
    m_value.m_vector_symbolic[3] = s4;
    m_value.m_vector_symbolic[4] = s5;
    m_value.m_vector_symbolic[5] = s6;
    m_value.m_vector_symbolic[6] = s7;
    m_value.m_vector_symbolic[7] = s8;
    m_addr_offset = 0;
    m_neg_pred = false;
    m_is_return_var = false;
    m_immediate_address = false;
  }

  void init(gpgpu_context *ctx) {
    gpgpu_ctx = ctx;
    m_uid = (unsigned)-1;
    m_valid = false;
    m_vector = false;
    m_type = undef_t;
    m_immediate_address = false;
    m_addr_space = undefined_space;
    m_operand_lohi = 0;
    m_double_operand_type = 0;
    m_operand_neg = false;
    m_const_mem_offset = (unsigned)-1;
    m_value.m_int = 0;
    m_value.m_unsigned = (unsigned)-1;
    m_value.m_float = 0;
    m_value.m_double = 0;
    for (unsigned i = 0; i < 4; i++) {
      m_value.m_vint[i] = 0;
      m_value.m_vunsigned[i] = 0;
      m_value.m_vfloat[i] = 0;
      m_value.m_vdouble[i] = 0;
    }
    m_value.m_symbolic = NULL;
    m_value.m_vector_symbolic = NULL;
    m_addr_offset = 0;
    m_neg_pred = 0;
    m_is_return_var = 0;
    m_is_non_arch_reg = 0;
  }
  void make_memory_operand() { m_type = memory_t; }
  void set_return() { m_is_return_var = true; }
  void set_immediate_addr() { m_immediate_address = true; }
  //返回操作数的名称。例如，pI为一条ptx_instruction *类型的指令，pI->dst().name().c_str()返回目
  //的操作数的名称。
  const std::string &name() const {
    assert(m_type == symbolic_t || m_type == reg_t || m_type == address_t ||
           m_type == memory_t || m_type == label_t);
    return m_value.m_symbolic->name();
  }
  //如果一个操作数是向量，则返回这个向量操作数中的元素个数。例如，pI为一条ptx_instruction *类型的
  //指令，pI->dst().get_vect_nelem()返回目的操作数的元素个数。操作数是向量的例子：
  //wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32
  //                  {%f260, %f261, %f262, %f263, %f264, %f265, %f266, %f267}, 
  //                  {%r144, %r145, %r146, %r147, %r148, %r149, %r150, %r151}, 
  //                  {%r152, %r153, %r154, %r155, %r156, %r157, %r158, %r159}, 
  //                  {%f260, %f261, %f262, %f263, %f264, %f265, %f266, %f267}; 
  //pI->dst().get_vect_nelem()返回值为8。
  unsigned get_vect_nelem() const {
    assert(is_vector());
    if (!m_value.m_vector_symbolic[0]) return 0;
    if (!m_value.m_vector_symbolic[1]) return 1;
    if (!m_value.m_vector_symbolic[2]) return 2;
    if (!m_value.m_vector_symbolic[3]) return 3;
    if (!m_value.m_vector_symbolic[4]) return 4;
    if (!m_value.m_vector_symbolic[5]) return 5;
    if (!m_value.m_vector_symbolic[6]) return 6;
    if (!m_value.m_vector_symbolic[7]) return 7;
    return 8;
  }
  //如果一个操作数是向量，则返回这个向量操作数中的第idx个元素。
  const symbol *vec_symbol(int idx) const {
    assert(idx < 8);
    const symbol *result = m_value.m_vector_symbolic[idx];
    assert(result != NULL);
    return result;
  }
  //如果一个操作数是向量，则返回这个向量操作数中的第0个元素的名称。
  const std::string &vec_name1() const {
    assert(m_type == vector_t);
    return m_value.m_vector_symbolic[0]->name();
  }
  //如果一个操作数是向量，则返回这个向量操作数中的第1个元素的名称。
  const std::string &vec_name2() const {
    assert(m_type == vector_t);
    return m_value.m_vector_symbolic[1]->name();
  }
  //如果一个操作数是向量，则返回这个向量操作数中的第2个元素的名称。
  const std::string &vec_name3() const {
    assert(m_type == vector_t);
    return m_value.m_vector_symbolic[2]->name();
  }
  //如果一个操作数是向量，则返回这个向量操作数中的第3个元素的名称。
  const std::string &vec_name4() const {
    assert(m_type == vector_t);
    return m_value.m_vector_symbolic[3]->name();
  }
  //判断一个操作数是否是寄存器。
  bool is_reg() const {
    if (m_type == reg_t) {
      return true;
    }
    if (m_type != symbolic_t) {
      return false;
    }
    return m_value.m_symbolic->type()->get_key().is_reg();
  }
  bool is_param_local() const {
    if (m_type != symbolic_t) return false;
    return m_value.m_symbolic->type()->get_key().is_param_local();
  }

  bool is_param_kernel() const {
    if (m_type != symbolic_t) return false;
    return m_value.m_symbolic->type()->get_key().is_param_kernel();
  }
  //判断一个操作数是否是向量，如果是则返回true，否则返回false。例如，pI为一条ptx_instruction *类
  //型的指令，pI->dst().is_vector()返回目的操作数是否是向量。操作数是向量的例子：
  //wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32
  //                  {%f260, %f261, %f262, %f263, %f264, %f265, %f266, %f267}, 
  //                  {%r144, %r145, %r146, %r147, %r148, %r149, %r150, %r151}, 
  //                  {%r152, %r153, %r154, %r155, %r156, %r157, %r158, %r159}, 
  //                  {%f260, %f261, %f262, %f263, %f264, %f265, %f266, %f267}; 
  bool is_vector() const {
    if (m_vector) return true;
    return false;
  }
  //返回操作数寄存器编号。
  int reg_num() const { return m_value.m_symbolic->reg_num(); }
  //如果一个操作数是向量，则返回这个向量操作数中的第0个寄存器编号。
  int reg1_num() const { return m_value.m_vector_symbolic[0]->reg_num(); }
  int reg2_num() const { return m_value.m_vector_symbolic[1]->reg_num(); }
  int reg3_num() const {
    return m_value.m_vector_symbolic[2]
               ? m_value.m_vector_symbolic[2]->reg_num()
               : 0;
  }
  int reg4_num() const {
    return m_value.m_vector_symbolic[3]
               ? m_value.m_vector_symbolic[3]->reg_num()
               : 0;
  }
  int reg5_num() const {
    return m_value.m_vector_symbolic[4]
               ? m_value.m_vector_symbolic[4]->reg_num()
               : 0;
  }
  int reg6_num() const {
    return m_value.m_vector_symbolic[5]
               ? m_value.m_vector_symbolic[5]->reg_num()
               : 0;
  }
  int reg7_num() const {
    return m_value.m_vector_symbolic[6]
               ? m_value.m_vector_symbolic[6]->reg_num()
               : 0;
  }
  int reg8_num() const {
    return m_value.m_vector_symbolic[7]
               ? m_value.m_vector_symbolic[7]->reg_num()
               : 0;
  }
  int arch_reg_num() const { return m_value.m_symbolic->arch_reg_num(); }
  int arch_reg_num(unsigned n) const {
    return (m_value.m_vector_symbolic[n])
               ? m_value.m_vector_symbolic[n]->arch_reg_num()
               : -1;
  }
  bool is_label() const { return m_type == label_t; }
  bool is_builtin() const { return m_type == builtin_t; }

  // Memory operand used in ld / st instructions (ex. [__var1])
  bool is_memory_operand() const { return m_type == memory_t; }

  // Memory operand with immediate access (ex. s[0x0004] or g[$r1+=0x0004])
  // This is used by the PTXPlus extension. The operand is assigned an address
  // space during parsing.
  bool is_memory_operand2() const { return (m_addr_space != undefined_space); }

  bool is_immediate_address() const { return m_immediate_address; }

  bool is_literal() const {
    return m_type == int_t || m_type == float_op_t || m_type == double_op_t ||
           m_type == unsigned_t;
  }
  bool is_shared() const {
    if (!(m_type == symbolic_t || m_type == address_t || m_type == memory_t)) {
      return false;
    }
    return m_value.m_symbolic->is_shared();
  }
  bool is_sstarr() const { return m_value.m_symbolic->is_sstarr(); }
  bool is_const() const { return m_value.m_symbolic->is_const(); }
  bool is_global() const { return m_value.m_symbolic->is_global(); }
  bool is_local() const { return m_value.m_symbolic->is_local(); }
  bool is_tex() const { return m_value.m_symbolic->is_tex(); }
  bool is_return_var() const { return m_is_return_var; }

  bool is_function_address() const {
    if (m_type != symbolic_t) {
      return false;
    }
    return m_value.m_symbolic->is_func_addr();
  }

  ptx_reg_t get_literal_value() const {
    ptx_reg_t result;
    switch (m_type) {
      case int_t:
        result.s64 = m_value.m_int;
        break;
      case float_op_t:
        result.f32 = m_value.m_float;
        break;
      case double_op_t:
        result.f64 = m_value.m_double;
        break;
      case unsigned_t:
        result.u32 = m_value.m_unsigned;
        break;
      default:
        assert(0);
        break;
    }
    return result;
  }
  int get_int() const { return m_value.m_int; }
  int get_addr_offset() const { return m_addr_offset; }
  const symbol *get_symbol() const { return m_value.m_symbolic; }
  void set_type(enum operand_type type) { m_type = type; }
  enum operand_type get_type() const { return m_type; }
  void set_neg_pred() {
    assert(m_valid);
    m_neg_pred = true;
  }
  bool is_neg_pred() const { return m_neg_pred; }
  bool is_valid() const { return m_valid; }

  void set_addr_space(enum _memory_space_t set_value) {
    m_addr_space = set_value;
  }
  enum _memory_space_t get_addr_space() const { return m_addr_space; }
  void set_operand_lohi(int set_value) { m_operand_lohi = set_value; }
  int get_operand_lohi() const { return m_operand_lohi; }
  void set_double_operand_type(int set_value) {
    m_double_operand_type = set_value;
  }
  int get_double_operand_type() const { return m_double_operand_type; }
  void set_operand_neg() { m_operand_neg = true; }
  bool get_operand_neg() const { return m_operand_neg; }
  void set_const_mem_offset(addr_t set_value) {
    m_const_mem_offset = set_value;
  }
  addr_t get_const_mem_offset() const { return m_const_mem_offset; }
  bool is_non_arch_reg() const { return m_is_non_arch_reg; }

 private:
  gpgpu_context *gpgpu_ctx;
  unsigned m_uid;
  bool m_valid;
  bool m_vector;
  enum operand_type m_type;
  bool m_immediate_address;
  enum _memory_space_t m_addr_space;
  int m_operand_lohi;
  int m_double_operand_type;
  bool m_operand_neg;
  addr_t m_const_mem_offset;
  union {
    int m_int;
    unsigned int m_unsigned;
    float m_float;
    double m_double;
    int m_vint[4];
    unsigned int m_vunsigned[4];
    float m_vfloat[4];
    double m_vdouble[4];
    const symbol *m_symbolic;
    const symbol **m_vector_symbolic;
  } m_value;

  int m_addr_offset;

  bool m_neg_pred;
  bool m_is_return_var;
  bool m_is_non_arch_reg;

  unsigned get_uid();
};

extern const char *g_opcode_string[];

/*
PTX指令代码块的基本块。
*/
struct basic_block_t {
  basic_block_t(unsigned ID, ptx_instruction *begin, ptx_instruction *end,
                bool entry, bool ex) {
    //basic block的唯一标识。
    bb_id = ID;
    //ptx_begin是该基本块的首条PTX指令。
    ptx_begin = begin;
    //ptx_end是该基本块的末尾PTX指令。
    ptx_end = end;
    //is_entry标志该基本块是否是入口处的基本块。
    is_entry = entry;
    //is_exit标志该基本块是否是出口处的基本块。
    is_exit = ex;
    immediatepostdominator_id = -1;
    immediatedominator_id = -1;
  }

  ptx_instruction *ptx_begin;
  ptx_instruction *ptx_end;
  std::set<int>
      predecessor_ids;  // indices of other basic blocks in m_basic_blocks array
  std::set<int> successor_ids;
  std::set<int> postdominator_ids;
  std::set<int> dominator_ids;
  std::set<int> Tmp_ids;
  int immediatepostdominator_id;
  int immediatedominator_id;
  bool is_entry;
  bool is_exit;
  unsigned bb_id;

  // if this basic block dom B
  bool dom(const basic_block_t *B) {
    return (B->dominator_ids.find(this->bb_id) != B->dominator_ids.end());
  }

  // if this basic block pdom B
  bool pdom(const basic_block_t *B) {
    return (B->postdominator_ids.find(this->bb_id) !=
            B->postdominator_ids.end());
  }
};

struct gpgpu_recon_t {
  address_type source_pc;
  address_type target_pc;
  class ptx_instruction *source_inst;
  class ptx_instruction *target_inst;
};

/*
单条PTX指令类。时序仿真中需要的指令数据。每条指令（ptx_instruction）承自 warp_inst_t，包含用于时序和
功能仿真的数据。 ptx_instruction 在功能仿真时被填充。在这一级之后,程序只需要时序信息,所以它将 ptx_ins
truction 转为 warp_inst_t（一些数据被释放）用于时序模拟。它持有 warp_id、warp内的活动线程掩码、内存
访问列表（mem_access_t）和该warp内线程的信息（per_thread_info）。
*/
class ptx_instruction : public warp_inst_t {
 public:
  ptx_instruction(int opcode, const symbol *pred, int neg_pred, int pred_mod,
                  symbol *label, const std::list<operand_info> &operands,
                  const operand_info &return_var, const std::list<int> &options,
                  const std::list<int> &wmma_options,
                  const std::list<int> &scalar_type, memory_space_t space_spec,
                  const char *file, unsigned line, const char *source,
                  const core_config *config, gpgpu_context *ctx);
  //调用ptx_instruction::print_insn(FILE *fp)，传入参数stdout，打印指令到屏幕。
  void print_insn() const;
  //传入参数 FILE *fp，打印指令到该文件。
  virtual void print_insn(FILE *fp) const;
  //将该指令转换为一个字符串，并返回该字符串。
  std::string to_string() const;
  //返回指令的大小，m_inst_size即为指令的大小，以字节为单位。
  unsigned inst_size() const { return m_inst_size; }
  //UID是 Unique Identifier 的缩写，用于标识特定的对象、任务或者线程。它可以用来跟踪对象、任务或者线
  //程的执行，以及收集相关的性能指标。uid() 用于获取当前指令的唯一ID。
  unsigned uid() const { return m_uid; }
  //获取当前指令的操作码 m_opcode，该操作码是一个 int 类型的值，与 opcodes.def 中定义的操作码对应。
  //例如， opcodes.def 中定义了：
  //    OP_DEF(ABS_OP,abs_impl,"abs",1,1)
  //其操作码为 ABS_OP。
  int get_opcode() const { return m_opcode; }
  //获取当前指令操作码定义中的字符串，例如上面的 opcodes.def 中的定义中，
  //    OP_DEF(ABS_OP,abs_impl,"abs",1,1)
  //其中，"abs"即为g_opcode_string[m_opcode]。
  const char *get_opcode_cstr() const {
    //当 m_opcode != -1 时为正常指令，当m_opcode == -1时为标签。
    if (m_opcode != -1) {
      return g_opcode_string[m_opcode];
    } else {
      return "label";
    }
  }
  //返回该指令所在的源文件。
  const char *source_file() const { return m_source_file.c_str(); }
  //返回该指令所在的源文件的行数。
  unsigned source_line() const { return m_source_line; }
  //返回m_operands变量（一个 std::vector 对象）的大小，即当前指令操作数的数量。
  unsigned get_num_operands() const { return m_operands.size(); }
  //下面的函数返回是否含有谓词寄存器，m_pred 是一个 symbol 类的对象。关于谓词指令，看下面的PTX指令：
  //  1.      asm("{\n\t"
  //  2.          ".reg .s32 b;\n\t"
  //  3.          ".reg .pred p;\n\t"     <======= 声明谓词寄存器p
  //  4.          "add.cc.u32 %1, %1, %2;\n\t"
  //  5.          "addc.s32 b, 0, 0;\n\t"
  //  6.          "sub.cc.u32 %0, %0, %2;\n\t"
  //  7.          "subc.cc.u32 %1, %1, 0;\n\t"
  //  8.          "subc.s32 b, b, 0;\n\t"
  //  9.          "setp.eq.s32 p, b, 1;\n\t"     <======= 给谓词变量绑定具体谓词逻辑
  //  10.         "@p add.cc.u32 %0, %0, 0xffffffff;\n\t"
  //  11.         "@p addc.u32 %1, %1, 0;\n\t"
  //  12.         "}"
  //  13.         : "+r"(x[0]), "+r"(x[1])
  //  14.         : "r"(x[2]));
  //谓词的声明使用 .pred 表示，例如第3行声明了谓词寄存器p。step指令给谓词变量绑定具体谓词逻辑，例如
  //第9行 "setp.eq.s32 p, b, 1"。
  //谓词的使用方法/指令格式为：
  //        @p opcode;
  //        @p opcode a;
  //        @p opcode d, a;
  //        @p opcode d, a, b;
  //        @p opcode d, a, b, c;
  //最左边的 @p是可选的guard predicate，即根据对应谓词结果选择是否执行该条指令。
  //谓词寄存器本质上是虚拟的寄存器，用于处理PTX中的分支（类比其他ISA的条件跳转指令beq等）。
  //SASS指令使用4位条件代码来指定更复杂的谓词行为，而不是PTX中的正常真假谓词系统。因此，PTXPlus使用
  //相同的4位谓词系统。GPGPU-Sim使用decuda的谓词转换表来模拟PTXPlus指令。谓词寄存器的最高位表示溢出
  //标志，后跟进位标志和符号标志。最后也是最低的位是零标志。单独的条件代码可以存储在单独的谓词寄存器中，
  //指令可以指示要使用或修改哪个谓词寄存器。以下指令将寄存器$r0中的值与寄存器$r1中的值相加，并将结果存
  //储在寄存器$r2中。同时，在谓词寄存器$p0中设置适当的标志：
  //        add.u32 $p0|$r2, $r0, $r1;
  //可以对谓词指令使用不同的测试条件。例如，只有当谓词寄存器$p0中的进位标志位被设置时，才执行下一条指令：
  //        @$p0.cf add.u32 $r2, $r0, $r1;
  bool has_pred() const { return m_pred != NULL; }
  //获取谓词。将谓词作为一个 operand_info 对象操作数信息返回。
  operand_info get_pred() const;
  //???
  bool get_pred_neg() const { return m_neg_pred; }
  //???
  int get_pred_mod() const { return m_pred_mod; }
  //返回该指令所在的PTX指令字符串。
  const char *get_source() const { return m_source.c_str(); }

  const std::list<int> get_scalar_type() const { return m_scalar_type; }
  const std::list<int> get_options() const { return m_options; }

  typedef std::vector<operand_info>::const_iterator const_iterator;
  //m_operands变量（是一个 std::vector 对象），即当前指令的操作数。返回其迭代开头。
  const_iterator op_iter_begin() const { return m_operands.begin(); }
  //m_operands变量（是一个 std::vector 对象），即当前指令的操作数。返回其迭代结尾。
  const_iterator op_iter_end() const { return m_operands.end(); }
  //PTX指令一般有0-4个操作数，外加一个可选的判断标志，一般第一个都是目的地址，后面的是源地址，也可以有两
  //个目的地址，比如：
  //    setp.lt.s32 p|q, a, b;          // p = (a < b); q = !(a < b);
  //也可以只有一个目的地址，比如：
  //    mad.rn.f64 d, a, b, c;          // d = a * b + c
  //下面函数获取目的操作数。
  const operand_info &dst() const {
    assert(!m_operands.empty());
    return m_operands[0];
  }
  //一般操作码为 CALL_OP 或 CALLP_OP 时，指令的目的地址为调用函数，这时目的地址是函数的地址 func_addr。
  //    void ptx_recognizer::set_return() {
  //      parse_assert((g_opcode == CALL_OP || g_opcode == CALLP_OP),
  //                  "only call can have return value");
  //      g_operands.front().set_return();
  //      g_return_var = g_operands.front();
  //    }
  //例如，PTX指令对 CALL 操作码指令格式的例子：
  //  //direct call to named function, func is a symbol
  //    call{.uni} (ret-param), func, (param-list);
  //    call{.uni} func, (param-list);
  //    call{.uni} func;
  //其中func_addr可以是 operands[0]，也可以是 operands[1]，但此时没有源操作数。
  const operand_info &func_addr() const {
    assert(!m_operands.empty());
    if (!m_operands[0].is_return_var()) {
      return m_operands[0];
    } else {
      assert(m_operands.size() >= 2);
      return m_operands[1];
    }
  }
  //同上面的 const operand_info &dst() const{...}。
  operand_info &dst() {
    assert(!m_operands.empty());
    return m_operands[0];
  }
  //根据m_operands的长度，获取至多8个源操作数。
  const operand_info &src1() const {
    assert(m_operands.size() > 1);
    return m_operands[1];
  }

  const operand_info &src2() const {
    assert(m_operands.size() > 2);
    return m_operands[2];
  }

  const operand_info &src3() const {
    assert(m_operands.size() > 3);
    return m_operands[3];
  }
  const operand_info &src4() const {
    assert(m_operands.size() > 4);
    return m_operands[4];
  }
  const operand_info &src5() const {
    assert(m_operands.size() > 5);
    return m_operands[5];
  }
  const operand_info &src6() const {
    assert(m_operands.size() > 6);
    return m_operands[6];
  }
  const operand_info &src7() const {
    assert(m_operands.size() > 7);
    return m_operands[7];
  }
  const operand_info &src8() const {
    assert(m_operands.size() > 8);
    return m_operands[8];
  }
  //传入参数 n，返回操作数列表 m_operands 中的第 n 个操作数。
  const operand_info &operand_lookup(unsigned n) const {
    assert(n < m_operands.size());
    return m_operands[n];
  }
  //返回是否该条指令有[返回值]操作数。
  bool has_return() const { return m_return_var.is_valid(); }
  //返回存储空间信息。存储空间的信息，如存储空间的类型和该存储空间的Bank的数量。GPGPU-Sim设置的存储空间
  //的类型有：
  //  enum _memory_space_t {
  //    //1. 未定义的空间类型
  //    undefined_space = 0,
  //    //2. 寄存器
  //    reg_space,
  //    //3. local memory
  //    local_space,
  //    //4. shared memory
  //    shared_space,
  //    //5. 貌似是 shared static array，其访存的行为与shared memory一致，可以认为其是shared 
  //    //   memory的一种
  //    sstarr_space,
  //    //6. 通用参数存储
  //    param_space_unclassified,
  //    //7. 对内核中的所有线程：全局性的，只读的
  //    param_space_kernel, // global to all threads in a kernel : read-only
  //    //8. 对某个线程：私有的，可读写的
  //    param_space_local,  // local to a thread : read-writable
  //    //9. 常量缓存
  //    const_space,
  //    //10.纹理缓存
  //    tex_space,
  //    //11.渲染曲面 // render surfaces 
  //    surf_space,
  //    //12.全局存储
  //    global_space,
  //    //13.通用存储
  //    generic_space,
  //    //14.指令存储
  //    instruction_space
  //  };
  memory_space_t get_space() const { return m_space_spec; }
  unsigned get_vector() const { return m_vector_spec; }
  unsigned get_atomic() const { return m_atomic_spec; }

  int get_wmma_type() const { return m_wmma_type; }
  //warp中的每一个线程都持有矩阵的一部分。warp中线程加载的fragment的分布是未指定的，并且依赖于目标体系
  //结构，因此矩阵中fragment的身份也是未指定的并且依赖于对象体系结构。如果基础矩阵的形状、布局和元素类
  //型匹配，则wmma操作返回的片段可以用作另一个wmma操作的操作数。由于片段布局依赖于体系结构，如果两个函
  //数链接在一起，但针对不同的链接兼容SM体系结构编译，则使用一个函数中的wmma操作返回的片段作为不同函数
  //中wmma操作的操作数可能无法按预期工作。注意，将wmma片段传递给具有弱链接的函数是不安全的，因为在链接
  //时对此类函数的引用可能会解析为不同编译模块中的函数。
  //获取 wmma 指令的 A、B 两个矩阵的layout，index为 0 时为 A 矩阵，index为 1 时为 B 矩阵。
  int get_wmma_layout(int index) const {
    return m_wmma_layout[index];  // 0->Matrix D,1->Matrix C
  }
  //指令字符串只有一个操作数时，直接用get_type()获取即可。
  int get_type() const {
    assert(!m_scalar_type.empty());
    return m_scalar_type.front();
  }
  //例如指令：cvt.frnd2{.relu}.f16x2.f32  d, a, b; 是将FP32类型的源操作数a和b转换为两个FP16类型，并
  //将a转换成的数据放到高16位，将b转换成的数据放到低16位，打包成一个数据放到目的地址d。因此，需要获取指
  //令字符串的第二个操作类型，使用get_type2()来获取。
  int get_type2() const {
    assert(m_scalar_type.size() == 2);
    return m_scalar_type.back();
  }

  void assign_bb(
      basic_block_t *basic_block)  // assign instruction to a basic block
  {
    m_basic_block = basic_block;
  }
  basic_block_t *get_bb() { return m_basic_block; }
  void set_m_instr_mem_index(unsigned index) { m_instr_mem_index = index; }
  //设置当前指令的 PC 值，指令处理过程中，为每条指令分配一个唯一的 PC 并作为参数传入该函数，该函数设置
  //当前指令的 PC 值：m_PC = PC。
  void set_PC(addr_t PC) { m_PC = PC; }
  //获取当前指令的 PC 值，返回 m_PC。
  addr_t get_PC() const { return m_PC; }

  unsigned get_m_instr_mem_index() { return m_instr_mem_index; }
  unsigned get_cmpop() const { return m_compare_op; }
  //获取该条指令的标签，返回的标签是一个 symbol 对象。
  const symbol *get_label() const { return m_label; }

  //is_label() 用于判断指令pI是否含有标签。label即为例如PTX指令块中的$L__BB0_6等：
  //  01.$L__BB0_6: <---- label
  //  02.  .pragma "nounroll";
  //  03.  ld.global.u32 %r28, [%rd32];
  //  04.  ...
  //  ...  ...
  //  12.  @%p5 bra $L__BB0_6; <---- label = $L__BB0_6
  bool is_label() const {
    //m_label 是一个 symbol 类的对象，储存了该条指令的标签，例如上述的 $L__BB0_6。如果 m_label 不为
    //空则代表该条指令是含有标签的。
    if (m_label) {
      assert(m_opcode == -1);
      return true;
    }
    return false;
  }
  bool is_hi() const { return m_hi; }
  bool is_lo() const { return m_lo; }
  bool is_wide() const { return m_wide; }
  bool is_uni() const { return m_uni; }
  bool is_exit() const { return m_exit; }
  bool is_abs() const { return m_abs; }
  bool is_neg() const { return m_neg; }
  bool is_to() const { return m_to_option; }
  unsigned cache_option() const { return m_cache_option; }
  unsigned rounding_mode() const { return m_rounding_mode; }
  unsigned saturation_mode() const { return m_saturation_mode; }
  unsigned clamp_mode() const { return m_clamp_mode; }
  unsigned left_mode() const { return m_left_mode; }
  unsigned dimension() const { return m_geom_spec; }
  unsigned barrier_op() const { return m_barrier_op; }
  unsigned shfl_op() const { return m_shfl_op; }
  unsigned prmt_op() const { return m_prmt_op; }
  enum vote_mode_t { vote_any, vote_all, vote_uni, vote_ballot };
  enum vote_mode_t vote_mode() const { return m_vote_mode; }

  int membar_level() const { return m_membar_level; }

  bool has_memory_read() const {
    if (m_opcode == LD_OP || m_opcode == LDU_OP || m_opcode == TEX_OP ||
        m_opcode == MMA_LD_OP)
      return true;
    // Check PTXPlus operand type below
    // Source operands are memory operands
    ptx_instruction::const_iterator op = op_iter_begin();
    for (int n = 0; op != op_iter_end(); op++, n++) {  // process operands
      if (n > 0 && op->is_memory_operand2())           // source operands only
        return true;
    }
    return false;
  }
  bool has_memory_write() const {
    if (m_opcode == ST_OP || m_opcode == MMA_ST_OP) return true;
    // Check PTXPlus operand type below
    // Destination operand is a memory operand
    ptx_instruction::const_iterator op = op_iter_begin();
    for (int n = 0; (op != op_iter_end() && n < 1);
         op++, n++) {                          // process operands
      if (n == 0 && op->is_memory_operand2())  // source operands only
        return true;
    }
    return false;
  }

 private:
  void set_opcode_and_latency();
  void set_bar_type();
  void set_fp_or_int_archop();
  void set_mul_div_or_other_archop();

  basic_block_t *m_basic_block;
  unsigned m_uid;
  addr_t m_PC;
  std::string m_source_file;
  unsigned m_source_line;
  std::string m_source;

  const symbol *m_pred;
  bool m_neg_pred;
  int m_pred_mod;
  int m_opcode;
  //m_label 是一个 symbol 类的对象，储存了该条指令的标签，例如上述的 $L__BB0_6。如果 m_label 不为空
  //则代表该条指令是含有标签的。
  const symbol *m_label;
  //该条指令的操作数列表 m_operands。
  std::vector<operand_info> m_operands;
  //指令的[返回值]操作数。
  operand_info m_return_var;

  std::list<int> m_options;
  std::list<int> m_wmma_options;
  bool m_wide;
  bool m_hi;
  bool m_lo;
  bool m_exit;
  bool m_abs;
  bool m_neg;
  bool m_uni;  // if branch instruction, this evaluates to true for uniform
               // branches (ie jumps)
  bool m_to_option;
  unsigned m_cache_option;
  int m_wmma_type;
  int m_wmma_layout[2];
  int m_wmma_configuration;
  unsigned m_rounding_mode;
  unsigned m_compare_op;
  unsigned m_saturation_mode;
  unsigned m_clamp_mode;
  unsigned m_left_mode;
  unsigned m_barrier_op;
  unsigned m_shfl_op;
  unsigned m_prmt_op;

  std::list<int> m_scalar_type;
  memory_space_t m_space_spec;
  int m_geom_spec;
  int m_vector_spec;
  int m_atomic_spec;
  enum vote_mode_t m_vote_mode;
  int m_membar_level;
  int m_instr_mem_index;  // index into m_instr_mem array
  unsigned m_inst_size;   // bytes

  virtual void pre_decode();
  friend class function_info;
  // backward pointer
  class gpgpu_context *gpgpu_ctx;
};

class param_info {
 public:
  param_info() {
    m_valid = false;
    m_value_set = false;
    m_size = 0;
    m_is_ptr = false;
  }
  param_info(std::string name, int type, size_t size, bool is_ptr,
             memory_space_t ptr_space) {
    m_valid = true;
    m_value_set = false;
    m_name = name;
    m_type = type;
    m_size = size;
    m_is_ptr = is_ptr;
    m_ptr_space = ptr_space;
  }
  void add_data(param_t v) {
    assert((!m_value_set) ||
           (m_value.size == v.size));  // if this fails concurrent kernel
                                       // launches might execute incorrectly
    m_value_set = true;
    m_value = v;
  }
  void add_offset(unsigned offset) { m_offset = offset; }
  unsigned get_offset() {
    assert(m_valid);
    return m_offset;
  }
  std::string get_name() const {
    assert(m_valid);
    return m_name;
  }
  int get_type() const {
    assert(m_valid);
    return m_type;
  }
  param_t get_value() const {
    assert(m_value_set);
    return m_value;
  }
  size_t get_size() const {
    assert(m_valid);
    return m_size;
  }
  bool is_ptr_shared() const {
    assert(m_valid);
    return (m_is_ptr and m_ptr_space == shared_space);
  }

 private:
  bool m_valid;
  std::string m_name;
  int m_type;
  size_t m_size;
  bool m_value_set;
  param_t m_value;
  unsigned m_offset;
  bool m_is_ptr;
  memory_space_t m_ptr_space;
};

/*
单独的PTX指令在PTX函数中找到，这些函数要么是内核入口点，要么是可以在GPU上调用的子程序。每个PTX函数都有
一个 function_info 对象，例如下述 .ptx 文件中：
    .visible .entry _Z6MatMulPiS_S_i(
      .param .u64 _Z6MatMulPiS_S_i_param_0,
      .param .u64 _Z6MatMulPiS_S_i_param_1,
      .param .u64 _Z6MatMulPiS_S_i_param_2,
      .param .u32 _Z6MatMulPiS_S_i_param_3
      )
    {...}
function_info 对象：
    1. function_info 包含一个可以进行功能模拟的静态PTX指令（ptx_instruction）列表。
    2. 对于内核入口点，将每个内核参数存储在一个映射 m_ptx_kernel_param_info 中；但是，对于OpenCL应用
    程序来说，这可能并不总是这样的。在OpenCL中，相关的常量内存空间可以通过两种方式分配。它可以在声明它
    的ptx文件中显式初始化，或者使用主机上的clCreateBuffer来分配它。在后面这种情况下，.ptx文件将包含一
    个参数的全局声明，但它将有一个未知的数组大小。因此，该符号的地址将不会被设置，需要在执行PTX之前在
    function_info::add_param_data(...) 函数中设置。在这种情况下，内核参数的地址被存储在function_info
    对象中的一个符号表中。
*/
class function_info {
 public:
  //构造函数
  function_info(int entry_point, gpgpu_context *ctx);
  const ptx_version &get_ptx_version() const {
    return m_symtab->get_ptx_version();
  }
  virtual ~function_info() {}
  unsigned get_sm_target() const { return m_symtab->get_sm_target(); }
  bool is_extern() const { return m_extern; }
  void set_name(const char *name) { m_name = name; }
  void set_symtab(symbol_table *symtab) { m_symtab = symtab; }
  std::string get_name() const { return m_name; }
  unsigned print_insn(unsigned pc, FILE *fp) const;
  std::string get_insn_str(unsigned pc) const;
  void add_inst(const std::list<ptx_instruction *> &instructions) {
    m_instructions = instructions;
  }
  std::list<ptx_instruction *>::iterator find_next_real_instruction(
      std::list<ptx_instruction *>::iterator i);
  void create_basic_blocks();

  void print_basic_blocks();

  void print_basic_block_links();
  void print_basic_block_dot();

  operand_info *find_break_target(
      ptx_instruction *p_break_insn);  // find the target of a break instruction
  void connect_basic_blocks();  // iterate across m_basic_blocks of function,
                                // connecting basic blocks together
  bool
  connect_break_targets();  // connecting break instructions with proper targets

  // iterate across m_basic_blocks of function,
  // finding dominator blocks, using algorithm of
  // Muchnick's Adv. Compiler Design & Implemmntation Fig 7.14
  void find_dominators();
  void print_dominators();
  void find_idominators();
  void print_idominators();

  // iterate across m_basic_blocks of function,
  // finding postdominator blocks, using algorithm of
  // Muchnick's Adv. Compiler Design & Implemmntation Fig 7.14
  void find_postdominators();
  void print_postdominators();

  // iterate across m_basic_blocks of function,
  // finding immediate postdominator blocks, using algorithm of
  // Muchnick's Adv. Compiler Design & Implemmntation Fig 7.15
  void find_ipostdominators();
  void print_ipostdominators();
  void do_pdom();  // function to call pdom analysis

  unsigned get_num_reconvergence_pairs();

  void get_reconvergence_pairs(gpgpu_recon_t *recon_points);

  unsigned get_function_size() { return m_instructions.size(); }

  void ptx_assemble();

  unsigned ptx_get_inst_op(ptx_thread_info *thread);
  void add_param(const char *name, struct param_t value) {
    m_kernel_params[name] = value;
  }
  void add_param_name_type_size(unsigned index, std::string name, int type,
                                size_t size, bool ptr, memory_space_t space);
  void add_param_data(unsigned argn, struct gpgpu_ptx_sim_arg *args);
  void add_return_var(const symbol *rv) { m_return_var_sym = rv; }
  void add_arg(const symbol *arg) {
    assert(arg != NULL);
    m_args.push_back(arg);
  }
  void remove_args() { m_args.clear(); }
  unsigned num_args() const { return m_args.size(); }
  unsigned get_args_aligned_size();

  const symbol *get_arg(unsigned n) const {
    assert(n < m_args.size());
    return m_args[n];
  }
  bool has_return() const { return m_return_var_sym != NULL; }
  const symbol *get_return_var() const { return m_return_var_sym; }
  const ptx_instruction *get_instruction(unsigned PC) const {
    unsigned index = PC - m_start_PC;
    if (index < m_instr_mem_size) return m_instr_mem[index];
    return NULL;
  }
  addr_t get_start_PC() const { return m_start_PC; }

  void finalize(memory_space *param_mem);
  void param_to_shared(memory_space *shared_mem, symbol_table *symtab);
  void list_param(FILE *fout) const;
  void ptx_jit_config(std::map<unsigned long long, size_t> mallocPtr_Size,
                      memory_space *param_mem, gpgpu_t *gpu, dim3 gridDim,
                      dim3 blockDim);

  virtual const struct gpgpu_ptx_sim_info *get_kernel_info() const {
    assert(m_kernel_info.maxthreads == maxnt_id);
    return &m_kernel_info;
  }
  //设置 kernel 的信息，m_kernel_info包括：Registers/shmem/etc. used (from ptxas -v), loaded 
  //from ___.ptxinfo along with ___.ptx。例如，cuda_codes/myTest/文件夹下面的PTX文件：
  //    .version 7.5
  //    .target sm_52
  //    .address_size 64
  //    .visible .entry _Z6MatMulPiS_S_i(
  //        .param .u64 _Z6MatMulPiS_S_i_param_0,
  //        .param .u64 _Z6MatMulPiS_S_i_param_1,
  //        .param .u64 _Z6MatMulPiS_S_i_param_2,
  //        .param .u32 _Z6MatMulPiS_S_i_param_3
  //    )
  //    ......
  //使用 ptxas 后，PTXAS文件：
  //    ptxas info    : 0 bytes gmem
  //    ptxas info    : Compiling entry function '_Z6MatMulPiS_S_i' for 'sm_52'
  //    ptxas info    : Function properties for _Z6MatMulPiS_S_i
  //        0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
  //    ptxas info    : Used 32 registers, 348 bytes cmem[0]
  //gpgpu_ptx_sim_info的定义：
  //    struct gpgpu_ptx_sim_info {
  //        // Holds properties of the kernel (Kernel's resource use).
  //        // These will be set to zero if a ptxinfo file is not present.
  //        int lmem;
  //        int smem;
  //        int cmem;
  //        int gmem;
  //        int regs;
  //        unsigned maxthreads;
  //        unsigned ptx_version;
  //        unsigned sm_target;
  //    };
  virtual const void set_kernel_info(const struct gpgpu_ptx_sim_info &info) {
    m_kernel_info = info;
    m_kernel_info.ptx_version = 10 * get_ptx_version().ver();
    m_kernel_info.sm_target = get_ptx_version().target();
    // THIS DEPENDS ON ptxas being called after the PTX is parsed.
    m_kernel_info.maxthreads = maxnt_id;
  }
  symbol_table *get_symtab() { return m_symtab; }

  unsigned local_mem_framesize() const { return m_local_mem_framesize; }
  void set_framesize(unsigned sz) { m_local_mem_framesize = sz; }
  bool is_entry_point() const { return m_entry_point; }
  //pdom_done是检查pdom是否完成的标志，返回这个标志
  bool is_pdom_set() const { return pdom_done; }  // return pdom flag
  //一旦检查pdom完成，设置标志pdom_done = true。
  void set_pdom() { pdom_done = true; }           // set pdom flag

  void add_config_param(size_t size, unsigned alignment) {
    unsigned offset = 0;
    if (m_param_configs.size() > 0) {
      unsigned offset_nom =
          m_param_configs.back().first + m_param_configs.back().second;
      // ensure offset matches alignment requirements
      offset = offset_nom % alignment ? (offset_nom / alignment + 1) * alignment
                                      : offset_nom;
    }
    m_param_configs.push_back(std::pair<size_t, unsigned>(size, offset));
  }

  std::pair<size_t, unsigned> get_param_config(unsigned param_num) const {
    return m_param_configs[param_num];
  }

  void set_maxnt_id(unsigned maxthreads) { maxnt_id = maxthreads; }
  unsigned get_maxnt_id() { return maxnt_id; }
  // backward pointer
  class gpgpu_context *gpgpu_ctx;

 protected:
  // Registers/shmem/etc. used (from ptxas -v), loaded from ___.ptxinfo along
  // with ___.ptx
  struct gpgpu_ptx_sim_info m_kernel_info;

 private:
  unsigned maxnt_id;
  unsigned m_uid;
  unsigned m_local_mem_framesize;
  bool m_entry_point;
  bool m_extern;
  bool m_assembled;
  //pdom_done是检查pdom是否完成的标志
  bool pdom_done;  // flag to check whether pdom is completed or not
  std::string m_name;
  ptx_instruction **m_instr_mem;
  unsigned m_start_PC;
  unsigned m_instr_mem_size;
  std::map<std::string, param_t> m_kernel_params;
  std::map<unsigned, param_info> m_ptx_kernel_param_info;
  std::vector<std::pair<size_t, unsigned> > m_param_configs;
  const symbol *m_return_var_sym;
  std::vector<const symbol *> m_args;
  std::list<ptx_instruction *> m_instructions;
  std::vector<basic_block_t *> m_basic_blocks;
  std::list<std::pair<unsigned, unsigned> > m_back_edges;
  std::map<std::string, unsigned> labels;
  unsigned num_reconvergence_pairs;

  // Registers/shmem/etc. used (from ptxas -v), loaded from ___.ptxinfo along
  // with ___.ptx
  // with ___.ptx

  symbol_table *m_symtab;

  // parameter size for device kernels
  //设备端内核函数的参数大小。
  int m_args_aligned_size;

  addr_t m_n;  // offset in m_instr_mem (used in do_pdom)
};

class arg_buffer_t {
 public:
  arg_buffer_t(gpgpu_context *ctx) : m_src_op(ctx) {
    m_is_reg = false;
    m_is_param = false;
    m_param_value = NULL;
    m_reg_value = ptx_reg_t();
  }
  arg_buffer_t(const arg_buffer_t &another, gpgpu_context *ctx)
      : m_src_op(ctx) {
    make_copy(another);
  }
  void make_copy(const arg_buffer_t &another) {
    m_dst = another.m_dst;
    m_src_op = another.m_src_op;
    m_is_reg = another.m_is_reg;
    m_is_param = another.m_is_param;
    m_reg_value = another.m_reg_value;
    m_param_bytes = another.m_param_bytes;
    if (m_is_param) {
      m_param_value = malloc(m_param_bytes);
      memcpy(m_param_value, another.m_param_value, m_param_bytes);
    }
  }
  void operator=(const arg_buffer_t &another) { make_copy(another); }
  ~arg_buffer_t() {
    if (m_is_param) free(m_param_value);
  }
  arg_buffer_t(const symbol *dst_sym, const operand_info &src_op,
               ptx_reg_t source_value)
      : m_src_op(src_op) {
    m_dst = dst_sym;
    m_reg_value = ptx_reg_t();
    if (dst_sym->is_reg()) {
      m_is_reg = true;
      m_is_param = false;
      assert(src_op.is_reg());
      m_reg_value = source_value;
    } else {
      m_is_param = true;
      m_is_reg = false;
      m_param_value = calloc(sizeof(ptx_reg_t), 1);
      // new (m_param_value) ptx_reg_t(source_value);
      memcpy(m_param_value, &source_value, sizeof(ptx_reg_t));
      m_param_bytes = sizeof(ptx_reg_t);
    }
  }
  arg_buffer_t(const symbol *dst_sym, const operand_info &src_op,
               void *source_param_value_array, unsigned array_size)
      : m_src_op(src_op) {
    m_dst = dst_sym;
    if (dst_sym->is_reg()) {
      m_is_reg = true;
      m_is_param = false;
      assert(src_op.is_param_local());
      assert(dst_sym->get_size_in_bytes() == array_size);
      switch (array_size) {
        case 1:
          m_reg_value.u8 = *(unsigned char *)source_param_value_array;
          break;
        case 2:
          m_reg_value.u16 = *(unsigned short *)source_param_value_array;
          break;
        case 4:
          m_reg_value.u32 = *(unsigned int *)source_param_value_array;
          break;
        case 8:
          m_reg_value.u64 = *(unsigned long long *)source_param_value_array;
          break;
        default:
          printf(
              "GPGPU-Sim PTX: ERROR ** source param size does not match known "
              "register sizes\n");
          break;
      }
    } else {
      // param
      m_is_param = true;
      m_is_reg = false;
      m_param_value = calloc(array_size, 1);
      m_param_bytes = array_size;
      memcpy(m_param_value, source_param_value_array, array_size);
    }
  }

  bool is_reg() const { return m_is_reg; }
  ptx_reg_t get_reg() const {
    assert(m_is_reg);
    return m_reg_value;
  }

  const void *get_param_buffer() const {
    assert(m_is_param);
    return m_param_value;
  }
  size_t get_param_buffer_size() const {
    assert(m_is_param);
    return m_param_bytes;
  }

  const symbol *get_dst() const { return m_dst; }

 private:
  // destination of copy
  const symbol *m_dst;

  // source operand
  operand_info m_src_op;

  // source information
  bool m_is_reg;
  bool m_is_param;

  // source is register
  ptx_reg_t m_reg_value;

  // source is param
  void *m_param_value;
  unsigned m_param_bytes;
};

typedef std::list<arg_buffer_t> arg_buffer_list_t;
arg_buffer_t copy_arg_to_buffer(ptx_thread_info *thread,
                                operand_info actual_param_op,
                                const symbol *formal_param);
void copy_args_into_buffer_list(const ptx_instruction *pI,
                                ptx_thread_info *thread,
                                const function_info *target_func,
                                arg_buffer_list_t &arg_values);
void copy_buffer_list_into_frame(ptx_thread_info *thread,
                                 arg_buffer_list_t &arg_values);
void copy_buffer_to_frame(ptx_thread_info *thread, const arg_buffer_t &a);

struct textureInfo {
  unsigned int texel_size;  // size in bytes, e.g. (channelDesc.x+y+z+w)/8
  unsigned int Tx,
      Ty;  // tiling factor dimensions of layout of texels per 64B cache block
  unsigned int Tx_numbits, Ty_numbits;  // log2(T)
  unsigned int texel_size_numbits;      // log2(texel_size)
};

extern std::map<std::string, symbol_table *> g_sym_name_to_symbol_table;

void gpgpu_ptx_assemble(std::string kname, void *kinfo);
#include "../option_parser.h"
unsigned ptx_kernel_shmem_size(void *kernel_impl);
unsigned ptx_kernel_nregs(void *kernel_impl);

#endif
