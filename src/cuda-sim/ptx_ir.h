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
type_info_key �����������ݶ������͵���Ϣ����ָ������ڼ�ʹ�ã���
*/
class type_info_key {
 public:
  //���캯����
  type_info_key() {
    //m_is_non_arch_reg ����˼��� void set_is_non_arch_reg() ��ע�͡�
    m_is_non_arch_reg = false;
    //��־�Ƿ��Ѿ���� type_info_key �����еĸ���Ա�����ĳ�ʼ����
    m_init = false;
  }
  //���캯�����Լ���ʼ����Ա��������������Ϊ��
  //  1.memory_space_t space_spec��GPGPU-Sim���õĴ洢�ռ�������У�
  //    enum _memory_space_t {
  //      //a. δ����Ŀռ�����
  //      undefined_space = 0,
  //      //b. �Ĵ���
  //      reg_space,
  //      //c. local memory
  //      local_space,
  //      //d. shared memory
  //      shared_space,
  //      //e. ò���� shared static array����ô����Ϊ��shared memoryһ�£�������Ϊ����shared 
  //      //   memory��һ��
  //      sstarr_space,
  //      //f. ͨ�ò����洢
  //      param_space_unclassified,
  //      //g. ���ں��е������̣߳�ȫ���Եģ�ֻ����
  //      param_space_kernel, // global to all threads in a kernel : read-only
  //      //h. ��ĳ���̣߳�˽�еģ��ɶ�д��
  //      param_space_local,  // local to a thread : read-writable
  //      //i. ��������
  //      const_space,
  //      //j. ������
  //      tex_space,
  //      //k. ��Ⱦ���� // render surfaces 
  //      surf_space,
  //      //l. ȫ�ִ洢
  //      global_space,
  //      //m. ͨ�ô洢
  //      generic_space,
  //      //n. ָ��洢
  //      instruction_space
  //    };
  //  2.int scalar_type_spec����ptx_tab.h�����yytokentype��ö�٣�ָ���Ǳ����������͡����磺
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
  //  3.int vector_spec��
  //  4.int alignment_spec��
  //  5.int extern_spec��
  //  6.int array_dim��
  type_info_key(memory_space_t space_spec, int scalar_type_spec,
                int vector_spec, int alignment_spec, int extern_spec,
                int array_dim) {
    //m_is_non_arch_reg ����˼��� void set_is_non_arch_reg() ��ע�͡� 
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
  //����һ��������Ϣ�ؼ��֣�type_info_key���Ƿ���һ�����������������ڼ�⺯���Ƿ��ѱ����壬�Լ������Ĳ�
  //�������Ƿ���ȷ��
  void set_is_func() {
    //��⵱ǰtype_info_key�����Ƿ��Ѿ���ʼ������
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

  //�ڴ���PTXָ��ʱ���п��ܻ�����һ��ָ���д��ڼĴ�����"_"����ʱ���ܵĴ����ﻹû���������ִ���"_"�Ĵ�
  //����ָ���m_is_non_arch_reg ����֧�ֱ�ʾΪ"_"�ļĴ����ĳ�ʼ���ô��롣��δ��ȡ��д��ָ�������
  //ʱ��ʹ�ô˼Ĵ�����Ȼ�������������뽫��ʶ��Ϊ�Ϸ��ļĴ����������ǲ��뽫�䴫�ݸ�΢��ϵ�ṹ�Ĵ���������
  //������ģ������Ϊ�ˣ���������ű������һ�����ţ���������Ϊnon_arch_reg����������Ӱ������ģ�⡣��
  //�ң������������"_"�Ĵ�����ָ���е�"_"����Ϊnull_key��null_key���ǵ�ǰ���һ�����󣩣�������
  //null_key.set_is_non_arch_reg()��
  void set_is_non_arch_reg() { m_is_non_arch_reg = true; }
  
  //m_is_non_arch_reg ����˼��� void set_is_non_arch_reg() ��ע�͡�
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
  //m_is_non_arch_reg ����˼��� void set_is_non_arch_reg() ��ע�͡�
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
type_info �����������ݶ������͵���Ϣ����ָ������ڼ�ʹ�ã���
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
���ű�����PTX��һ���ڴ�λ�õ��ı���ʾ��e.g., "%r2", "input_data", etc...����һ�������������ͺ�λ��
��Ϣ�ķ��Ŷ����ӳ�䡣
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
һ������ָ��Դ�������ķ�װ�࣬�����ǼĴ�����ʶ�����ڴ�������������û�ģʽ��Ϣ����ʱ��������
*/
class operand_info {
 public:
  //���캯����
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
  //���ز����������ơ����磬pIΪһ��ptx_instruction *���͵�ָ�pI->dst().name().c_str()����Ŀ
  //�Ĳ����������ơ�
  const std::string &name() const {
    assert(m_type == symbolic_t || m_type == reg_t || m_type == address_t ||
           m_type == memory_t || m_type == label_t);
    return m_value.m_symbolic->name();
  }
  //���һ�����������������򷵻���������������е�Ԫ�ظ��������磬pIΪһ��ptx_instruction *���͵�
  //ָ�pI->dst().get_vect_nelem()����Ŀ�Ĳ�������Ԫ�ظ����������������������ӣ�
  //wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32
  //                  {%f260, %f261, %f262, %f263, %f264, %f265, %f266, %f267}, 
  //                  {%r144, %r145, %r146, %r147, %r148, %r149, %r150, %r151}, 
  //                  {%r152, %r153, %r154, %r155, %r156, %r157, %r158, %r159}, 
  //                  {%f260, %f261, %f262, %f263, %f264, %f265, %f266, %f267}; 
  //pI->dst().get_vect_nelem()����ֵΪ8��
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
  //���һ�����������������򷵻���������������еĵ�idx��Ԫ�ء�
  const symbol *vec_symbol(int idx) const {
    assert(idx < 8);
    const symbol *result = m_value.m_vector_symbolic[idx];
    assert(result != NULL);
    return result;
  }
  //���һ�����������������򷵻���������������еĵ�0��Ԫ�ص����ơ�
  const std::string &vec_name1() const {
    assert(m_type == vector_t);
    return m_value.m_vector_symbolic[0]->name();
  }
  //���һ�����������������򷵻���������������еĵ�1��Ԫ�ص����ơ�
  const std::string &vec_name2() const {
    assert(m_type == vector_t);
    return m_value.m_vector_symbolic[1]->name();
  }
  //���һ�����������������򷵻���������������еĵ�2��Ԫ�ص����ơ�
  const std::string &vec_name3() const {
    assert(m_type == vector_t);
    return m_value.m_vector_symbolic[2]->name();
  }
  //���һ�����������������򷵻���������������еĵ�3��Ԫ�ص����ơ�
  const std::string &vec_name4() const {
    assert(m_type == vector_t);
    return m_value.m_vector_symbolic[3]->name();
  }
  //�ж�һ���������Ƿ��ǼĴ�����
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
  //�ж�һ���������Ƿ���������������򷵻�true�����򷵻�false�����磬pIΪһ��ptx_instruction *��
  //�͵�ָ�pI->dst().is_vector()����Ŀ�Ĳ������Ƿ��������������������������ӣ�
  //wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32
  //                  {%f260, %f261, %f262, %f263, %f264, %f265, %f266, %f267}, 
  //                  {%r144, %r145, %r146, %r147, %r148, %r149, %r150, %r151}, 
  //                  {%r152, %r153, %r154, %r155, %r156, %r157, %r158, %r159}, 
  //                  {%f260, %f261, %f262, %f263, %f264, %f265, %f266, %f267}; 
  bool is_vector() const {
    if (m_vector) return true;
    return false;
  }
  //���ز������Ĵ�����š�
  int reg_num() const { return m_value.m_symbolic->reg_num(); }
  //���һ�����������������򷵻���������������еĵ�0���Ĵ�����š�
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
PTXָ������Ļ����顣
*/
struct basic_block_t {
  basic_block_t(unsigned ID, ptx_instruction *begin, ptx_instruction *end,
                bool entry, bool ex) {
    //basic block��Ψһ��ʶ��
    bb_id = ID;
    //ptx_begin�Ǹû����������PTXָ�
    ptx_begin = begin;
    //ptx_end�Ǹû������ĩβPTXָ�
    ptx_end = end;
    //is_entry��־�û������Ƿ�����ڴ��Ļ����顣
    is_entry = entry;
    //is_exit��־�û������Ƿ��ǳ��ڴ��Ļ����顣
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
����PTXָ���ࡣʱ���������Ҫ��ָ�����ݡ�ÿ��ָ�ptx_instruction������ warp_inst_t����������ʱ���
���ܷ�������ݡ� ptx_instruction �ڹ��ܷ���ʱ����䡣����һ��֮��,����ֻ��Ҫʱ����Ϣ,�������� ptx_ins
truction תΪ warp_inst_t��һЩ���ݱ��ͷţ�����ʱ��ģ�⡣������ warp_id��warp�ڵĻ�߳����롢�ڴ�
�����б�mem_access_t���͸�warp���̵߳���Ϣ��per_thread_info����
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
  //����ptx_instruction::print_insn(FILE *fp)���������stdout����ӡָ���Ļ��
  void print_insn() const;
  //������� FILE *fp����ӡָ����ļ���
  virtual void print_insn(FILE *fp) const;
  //����ָ��ת��Ϊһ���ַ����������ظ��ַ�����
  std::string to_string() const;
  //����ָ��Ĵ�С��m_inst_size��Ϊָ��Ĵ�С�����ֽ�Ϊ��λ��
  unsigned inst_size() const { return m_inst_size; }
  //UID�� Unique Identifier ����д�����ڱ�ʶ�ض��Ķ�����������̡߳��������������ٶ������������
  //�̵�ִ�У��Լ��ռ���ص�����ָ�ꡣuid() ���ڻ�ȡ��ǰָ���ΨһID��
  unsigned uid() const { return m_uid; }
  //��ȡ��ǰָ��Ĳ����� m_opcode���ò�������һ�� int ���͵�ֵ���� opcodes.def �ж���Ĳ������Ӧ��
  //���磬 opcodes.def �ж����ˣ�
  //    OP_DEF(ABS_OP,abs_impl,"abs",1,1)
  //�������Ϊ ABS_OP��
  int get_opcode() const { return m_opcode; }
  //��ȡ��ǰָ������붨���е��ַ�������������� opcodes.def �еĶ����У�
  //    OP_DEF(ABS_OP,abs_impl,"abs",1,1)
  //���У�"abs"��Ϊg_opcode_string[m_opcode]��
  const char *get_opcode_cstr() const {
    //�� m_opcode != -1 ʱΪ����ָ���m_opcode == -1ʱΪ��ǩ��
    if (m_opcode != -1) {
      return g_opcode_string[m_opcode];
    } else {
      return "label";
    }
  }
  //���ظ�ָ�����ڵ�Դ�ļ���
  const char *source_file() const { return m_source_file.c_str(); }
  //���ظ�ָ�����ڵ�Դ�ļ���������
  unsigned source_line() const { return m_source_line; }
  //����m_operands������һ�� std::vector ���󣩵Ĵ�С������ǰָ���������������
  unsigned get_num_operands() const { return m_operands.size(); }
  //����ĺ��������Ƿ���ν�ʼĴ�����m_pred ��һ�� symbol ��Ķ��󡣹���ν��ָ��������PTXָ�
  //  1.      asm("{\n\t"
  //  2.          ".reg .s32 b;\n\t"
  //  3.          ".reg .pred p;\n\t"     <======= ����ν�ʼĴ���p
  //  4.          "add.cc.u32 %1, %1, %2;\n\t"
  //  5.          "addc.s32 b, 0, 0;\n\t"
  //  6.          "sub.cc.u32 %0, %0, %2;\n\t"
  //  7.          "subc.cc.u32 %1, %1, 0;\n\t"
  //  8.          "subc.s32 b, b, 0;\n\t"
  //  9.          "setp.eq.s32 p, b, 1;\n\t"     <======= ��ν�ʱ����󶨾���ν���߼�
  //  10.         "@p add.cc.u32 %0, %0, 0xffffffff;\n\t"
  //  11.         "@p addc.u32 %1, %1, 0;\n\t"
  //  12.         "}"
  //  13.         : "+r"(x[0]), "+r"(x[1])
  //  14.         : "r"(x[2]));
  //ν�ʵ�����ʹ�� .pred ��ʾ�������3��������ν�ʼĴ���p��stepָ���ν�ʱ����󶨾���ν���߼�������
  //��9�� "setp.eq.s32 p, b, 1"��
  //ν�ʵ�ʹ�÷���/ָ���ʽΪ��
  //        @p opcode;
  //        @p opcode a;
  //        @p opcode d, a;
  //        @p opcode d, a, b;
  //        @p opcode d, a, b, c;
  //����ߵ� @p�ǿ�ѡ��guard predicate�������ݶ�Ӧν�ʽ��ѡ���Ƿ�ִ�и���ָ�
  //ν�ʼĴ���������������ļĴ��������ڴ���PTX�еķ�֧���������ISA��������תָ��beq�ȣ���
  //SASSָ��ʹ��4λ����������ָ�������ӵ�ν����Ϊ��������PTX�е��������ν��ϵͳ����ˣ�PTXPlusʹ��
  //��ͬ��4λν��ϵͳ��GPGPU-Simʹ��decuda��ν��ת������ģ��PTXPlusָ�ν�ʼĴ��������λ��ʾ���
  //��־�������λ��־�ͷ��ű�־�����Ҳ����͵�λ�����־������������������Դ洢�ڵ�����ν�ʼĴ����У�
  //ָ�����ָʾҪʹ�û��޸��ĸ�ν�ʼĴ���������ָ��Ĵ���$r0�е�ֵ��Ĵ���$r1�е�ֵ��ӣ����������
  //���ڼĴ���$r2�С�ͬʱ����ν�ʼĴ���$p0�������ʵ��ı�־��
  //        add.u32 $p0|$r2, $r0, $r1;
  //���Զ�ν��ָ��ʹ�ò�ͬ�Ĳ������������磬ֻ�е�ν�ʼĴ���$p0�еĽ�λ��־λ������ʱ����ִ����һ��ָ�
  //        @$p0.cf add.u32 $r2, $r0, $r1;
  bool has_pred() const { return m_pred != NULL; }
  //��ȡν�ʡ���ν����Ϊһ�� operand_info �����������Ϣ���ء�
  operand_info get_pred() const;
  //???
  bool get_pred_neg() const { return m_neg_pred; }
  //???
  int get_pred_mod() const { return m_pred_mod; }
  //���ظ�ָ�����ڵ�PTXָ���ַ�����
  const char *get_source() const { return m_source.c_str(); }

  const std::list<int> get_scalar_type() const { return m_scalar_type; }
  const std::list<int> get_options() const { return m_options; }

  typedef std::vector<operand_info>::const_iterator const_iterator;
  //m_operands��������һ�� std::vector ���󣩣�����ǰָ��Ĳ������������������ͷ��
  const_iterator op_iter_begin() const { return m_operands.begin(); }
  //m_operands��������һ�� std::vector ���󣩣�����ǰָ��Ĳ������������������β��
  const_iterator op_iter_end() const { return m_operands.end(); }
  //PTXָ��һ����0-4�������������һ����ѡ���жϱ�־��һ���һ������Ŀ�ĵ�ַ���������Դ��ַ��Ҳ��������
  //��Ŀ�ĵ�ַ�����磺
  //    setp.lt.s32 p|q, a, b;          // p = (a < b); q = !(a < b);
  //Ҳ����ֻ��һ��Ŀ�ĵ�ַ�����磺
  //    mad.rn.f64 d, a, b, c;          // d = a * b + c
  //���溯����ȡĿ�Ĳ�������
  const operand_info &dst() const {
    assert(!m_operands.empty());
    return m_operands[0];
  }
  //һ�������Ϊ CALL_OP �� CALLP_OP ʱ��ָ���Ŀ�ĵ�ַΪ���ú�������ʱĿ�ĵ�ַ�Ǻ����ĵ�ַ func_addr��
  //    void ptx_recognizer::set_return() {
  //      parse_assert((g_opcode == CALL_OP || g_opcode == CALLP_OP),
  //                  "only call can have return value");
  //      g_operands.front().set_return();
  //      g_return_var = g_operands.front();
  //    }
  //���磬PTXָ��� CALL ������ָ���ʽ�����ӣ�
  //  //direct call to named function, func is a symbol
  //    call{.uni} (ret-param), func, (param-list);
  //    call{.uni} func, (param-list);
  //    call{.uni} func;
  //����func_addr������ operands[0]��Ҳ������ operands[1]������ʱû��Դ��������
  const operand_info &func_addr() const {
    assert(!m_operands.empty());
    if (!m_operands[0].is_return_var()) {
      return m_operands[0];
    } else {
      assert(m_operands.size() >= 2);
      return m_operands[1];
    }
  }
  //ͬ����� const operand_info &dst() const{...}��
  operand_info &dst() {
    assert(!m_operands.empty());
    return m_operands[0];
  }
  //����m_operands�ĳ��ȣ���ȡ����8��Դ��������
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
  //������� n�����ز������б� m_operands �еĵ� n ����������
  const operand_info &operand_lookup(unsigned n) const {
    assert(n < m_operands.size());
    return m_operands[n];
  }
  //�����Ƿ����ָ����[����ֵ]��������
  bool has_return() const { return m_return_var.is_valid(); }
  //���ش洢�ռ���Ϣ���洢�ռ����Ϣ����洢�ռ�����ͺ͸ô洢�ռ��Bank��������GPGPU-Sim���õĴ洢�ռ�
  //�������У�
  //  enum _memory_space_t {
  //    //1. δ����Ŀռ�����
  //    undefined_space = 0,
  //    //2. �Ĵ���
  //    reg_space,
  //    //3. local memory
  //    local_space,
  //    //4. shared memory
  //    shared_space,
  //    //5. ò���� shared static array����ô����Ϊ��shared memoryһ�£�������Ϊ����shared 
  //    //   memory��һ��
  //    sstarr_space,
  //    //6. ͨ�ò����洢
  //    param_space_unclassified,
  //    //7. ���ں��е������̣߳�ȫ���Եģ�ֻ����
  //    param_space_kernel, // global to all threads in a kernel : read-only
  //    //8. ��ĳ���̣߳�˽�еģ��ɶ�д��
  //    param_space_local,  // local to a thread : read-writable
  //    //9. ��������
  //    const_space,
  //    //10.������
  //    tex_space,
  //    //11.��Ⱦ���� // render surfaces 
  //    surf_space,
  //    //12.ȫ�ִ洢
  //    global_space,
  //    //13.ͨ�ô洢
  //    generic_space,
  //    //14.ָ��洢
  //    instruction_space
  //  };
  memory_space_t get_space() const { return m_space_spec; }
  unsigned get_vector() const { return m_vector_spec; }
  unsigned get_atomic() const { return m_atomic_spec; }

  int get_wmma_type() const { return m_wmma_type; }
  //warp�е�ÿһ���̶߳����о����һ���֡�warp���̼߳��ص�fragment�ķֲ���δָ���ģ�����������Ŀ����ϵ
  //�ṹ����˾�����fragment�����Ҳ��δָ���Ĳ��������ڶ�����ϵ�ṹ����������������״�����ֺ�Ԫ����
  //��ƥ�䣬��wmma�������ص�Ƭ�ο���������һ��wmma�����Ĳ�����������Ƭ�β�����������ϵ�ṹ�����������
  //��������һ�𣬵���Բ�ͬ�����Ӽ���SM��ϵ�ṹ���룬��ʹ��һ�������е�wmma�������ص�Ƭ����Ϊ��ͬ����
  //��wmma�����Ĳ����������޷���Ԥ�ڹ�����ע�⣬��wmmaƬ�δ��ݸ����������ӵĺ����ǲ���ȫ�ģ���Ϊ������
  //ʱ�Դ��ຯ�������ÿ��ܻ����Ϊ��ͬ����ģ���еĺ�����
  //��ȡ wmma ָ��� A��B ���������layout��indexΪ 0 ʱΪ A ����indexΪ 1 ʱΪ B ����
  int get_wmma_layout(int index) const {
    return m_wmma_layout[index];  // 0->Matrix D,1->Matrix C
  }
  //ָ���ַ���ֻ��һ��������ʱ��ֱ����get_type()��ȡ���ɡ�
  int get_type() const {
    assert(!m_scalar_type.empty());
    return m_scalar_type.front();
  }
  //����ָ�cvt.frnd2{.relu}.f16x2.f32  d, a, b; �ǽ�FP32���͵�Դ������a��bת��Ϊ����FP16���ͣ���
  //��aת���ɵ����ݷŵ���16λ����bת���ɵ����ݷŵ���16λ�������һ�����ݷŵ�Ŀ�ĵ�ַd����ˣ���Ҫ��ȡָ
  //���ַ����ĵڶ����������ͣ�ʹ��get_type2()����ȡ��
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
  //���õ�ǰָ��� PC ֵ��ָ�������У�Ϊÿ��ָ�����һ��Ψһ�� PC ����Ϊ��������ú������ú�������
  //��ǰָ��� PC ֵ��m_PC = PC��
  void set_PC(addr_t PC) { m_PC = PC; }
  //��ȡ��ǰָ��� PC ֵ������ m_PC��
  addr_t get_PC() const { return m_PC; }

  unsigned get_m_instr_mem_index() { return m_instr_mem_index; }
  unsigned get_cmpop() const { return m_compare_op; }
  //��ȡ����ָ��ı�ǩ�����صı�ǩ��һ�� symbol ����
  const symbol *get_label() const { return m_label; }

  //is_label() �����ж�ָ��pI�Ƿ��б�ǩ��label��Ϊ����PTXָ����е�$L__BB0_6�ȣ�
  //  01.$L__BB0_6: <---- label
  //  02.  .pragma "nounroll";
  //  03.  ld.global.u32 %r28, [%rd32];
  //  04.  ...
  //  ...  ...
  //  12.  @%p5 bra $L__BB0_6; <---- label = $L__BB0_6
  bool is_label() const {
    //m_label ��һ�� symbol ��Ķ��󣬴����˸���ָ��ı�ǩ������������ $L__BB0_6����� m_label ��Ϊ
    //����������ָ���Ǻ��б�ǩ�ġ�
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
  //m_label ��һ�� symbol ��Ķ��󣬴����˸���ָ��ı�ǩ������������ $L__BB0_6����� m_label ��Ϊ��
  //��������ָ���Ǻ��б�ǩ�ġ�
  const symbol *m_label;
  //����ָ��Ĳ������б� m_operands��
  std::vector<operand_info> m_operands;
  //ָ���[����ֵ]��������
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
������PTXָ����PTX�������ҵ�����Щ����Ҫô���ں���ڵ㣬Ҫô�ǿ�����GPU�ϵ��õ��ӳ���ÿ��PTX��������
һ�� function_info ������������ .ptx �ļ��У�
    .visible .entry _Z6MatMulPiS_S_i(
      .param .u64 _Z6MatMulPiS_S_i_param_0,
      .param .u64 _Z6MatMulPiS_S_i_param_1,
      .param .u64 _Z6MatMulPiS_S_i_param_2,
      .param .u32 _Z6MatMulPiS_S_i_param_3
      )
    {...}
function_info ����
    1. function_info ����һ�����Խ��й���ģ��ľ�̬PTXָ�ptx_instruction���б�
    2. �����ں���ڵ㣬��ÿ���ں˲����洢��һ��ӳ�� m_ptx_kernel_param_info �У����ǣ�����OpenCLӦ��
    ������˵������ܲ������������ġ���OpenCL�У���صĳ����ڴ�ռ����ͨ�����ַ�ʽ���䡣��������������
    ��ptx�ļ�����ʽ��ʼ��������ʹ�������ϵ�clCreateBuffer�����������ں�����������£�.ptx�ļ�������һ
    ��������ȫ����������������һ��δ֪�������С����ˣ��÷��ŵĵ�ַ�����ᱻ���ã���Ҫ��ִ��PTX֮ǰ��
    function_info::add_param_data(...) ���������á�����������£��ں˲����ĵ�ַ���洢��function_info
    �����е�һ�����ű��С�
*/
class function_info {
 public:
  //���캯��
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
  //���� kernel ����Ϣ��m_kernel_info������Registers/shmem/etc. used (from ptxas -v), loaded 
  //from ___.ptxinfo along with ___.ptx�����磬cuda_codes/myTest/�ļ��������PTX�ļ���
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
  //ʹ�� ptxas ��PTXAS�ļ���
  //    ptxas info    : 0 bytes gmem
  //    ptxas info    : Compiling entry function '_Z6MatMulPiS_S_i' for 'sm_52'
  //    ptxas info    : Function properties for _Z6MatMulPiS_S_i
  //        0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
  //    ptxas info    : Used 32 registers, 348 bytes cmem[0]
  //gpgpu_ptx_sim_info�Ķ��壺
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
  //pdom_done�Ǽ��pdom�Ƿ���ɵı�־�����������־
  bool is_pdom_set() const { return pdom_done; }  // return pdom flag
  //һ�����pdom��ɣ����ñ�־pdom_done = true��
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
  //pdom_done�Ǽ��pdom�Ƿ���ɵı�־
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
  //�豸���ں˺����Ĳ�����С��
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
