// Copyright (c) 2009-2021, Tor M. Aamodt, Ali Bakhoda, Wilson W.L. Fung,
// George L. Yuan, Jimmy Kwa, Vijay Kandiah, Nikos Hardavellas,
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

#include "cuda-sim.h"

#include "instructions.h"
#include "ptx_ir.h"
class ptx_recognizer;
typedef void *yyscan_t;
#include <stdio.h>
#include <map>
#include <set>
#include <sstream>
#include "../../libcuda/gpgpu_context.h"
#include "../abstract_hardware_model.h"
#include "../gpgpu-sim/gpu-sim.h"
#include "../gpgpusim_entrypoint.h"
#include "../statwrapper.h"
#include "../stream_manager.h"
#include "cuda_device_runtime.h"
#include "decuda_pred_table/decuda_pred_table.h"
#include "memory.h"
#include "opcodes.h"
#include "ptx-stats.h"
#include "ptx.tab.h"
#include "ptx_loader.h"
#include "ptx_parser.h"
#include "ptx_sim.h"

int g_debug_execution = 0;
// Output debug information to file options

/*
该函数用来解析PTX指令的操作码延迟相关的选项的。该函数接受一个option_parser_t类型的参数，用于解析命
令行参数，并将相应的值存储在一个全局的数据结构中，以便其他函数可以使用。OPT_CSTR为字符串类型变量。
*/
void cuda_sim::ptx_opcocde_latency_options(option_parser_t opp) {
  //整型指令操作码延迟，分别用逗号隔开的是<ADD,MAX,MUL,MAD,DIV,SHFL>指令。
  option_parser_register(
      opp, "-ptx_opcode_latency_int", OPT_CSTR, &opcode_latency_int,
      "Opcode latencies for integers <ADD,MAX,MUL,MAD,DIV,SHFL>"
      "Default 1,1,19,25,145,32",
      "1,1,19,25,145,32");
  //单精度浮点指令操作码延迟，分别用逗号隔开的是<ADD,MAX,MUL,MAD,DIV>指令。
  option_parser_register(opp, "-ptx_opcode_latency_fp", OPT_CSTR,
                         &opcode_latency_fp,
                         "Opcode latencies for single precision floating "
                         "points <ADD,MAX,MUL,MAD,DIV>"
                         "Default 1,1,1,1,30",
                         "1,1,1,1,30");
  //双精度浮点指令操作码延迟，分别用逗号隔开的是<ADD,MAX,MUL,MAD,DIV>指令。
  option_parser_register(opp, "-ptx_opcode_latency_dp", OPT_CSTR,
                         &opcode_latency_dp,
                         "Opcode latencies for double precision floating "
                         "points <ADD,MAX,MUL,MAD,DIV>"
                         "Default 8,8,8,8,335",
                         "8,8,8,8,335");
  //SFU-特殊功能单元指令操作码延迟，所有的Div指令在特殊功能单元执行。
  option_parser_register(opp, "-ptx_opcode_latency_sfu", OPT_CSTR,
                         &opcode_latency_sfu,
                         "Opcode latencies for SFU instructions"
                         "Default 8",
                         "8");
  //Tensor指令操作码延迟。
  option_parser_register(opp, "-ptx_opcode_latency_tesnor", OPT_CSTR,
                         &opcode_latency_tensor,
                         "Opcode latencies for Tensor instructions"
                         "Default 64",
                         "64");
  //整型指令操作码初始化周期，用逗号隔开的是<ADD,MAX,MUL,MAD,DIV,SHFL>指令。对于这个周期数，ALU的
  //输入保持恒定。ALU在此期间不能使用新值。即如果该值为4，则意味着该单元可以每4个周期消耗一次新值。
  option_parser_register(
      opp, "-ptx_opcode_initiation_int", OPT_CSTR, &opcode_initiation_int,
      "Opcode initiation intervals for integers <ADD,MAX,MUL,MAD,DIV,SHFL>"
      "Default 1,1,4,4,32,4",
      "1,1,4,4,32,4");
  //单精度浮点指令操作码初始化周期，用逗号隔开的是<ADD,MAX,MUL,MAD,DIV>指令。
  option_parser_register(opp, "-ptx_opcode_initiation_fp", OPT_CSTR,
                         &opcode_initiation_fp,
                         "Opcode initiation intervals for single precision "
                         "floating points <ADD,MAX,MUL,MAD,DIV>"
                         "Default 1,1,1,1,5",
                         "1,1,1,1,5");
  //双精度浮点指令操作码初始化周期，用逗号隔开的是<ADD,MAX,MUL,MAD,DIV>指令。
  option_parser_register(opp, "-ptx_opcode_initiation_dp", OPT_CSTR,
                         &opcode_initiation_dp,
                         "Opcode initiation intervals for double precision "
                         "floating points <ADD,MAX,MUL,MAD,DIV>"
                         "Default 8,8,8,8,130",
                         "8,8,8,8,130");
  //SFU-特殊功能单元指令操作码初始化周期，所有的Div指令在特殊功能单元执行。
  option_parser_register(opp, "-ptx_opcode_initiation_sfu", OPT_CSTR,
                         &opcode_initiation_sfu,
                         "Opcode initiation intervals for sfu instructions"
                         "Default 8",
                         "8");
  //Tensor指令操作码初始化周期。
  option_parser_register(opp, "-ptx_opcode_initiation_tensor", OPT_CSTR,
                         &opcode_initiation_tensor,
                         "Opcode initiation intervals for tensor instructions"
                         "Default 64",
                         "64");
  //下列CUDA API的延迟：
  //cudaStreamCreateWithFlags, cudaGetParameterBufferV2_init_perWarp, 
  //cudaGetParameterBufferV2_perKernel, cudaLaunchDeviceV2_init_perWarp, 
  //cudaLaunchDevicV2_perKernel>
  option_parser_register(opp, "-cdp_latency", OPT_CSTR, &cdp_latency_str,
                         "CDP API latency <cudaStreamCreateWithFlags, \
cudaGetParameterBufferV2_init_perWarp, cudaGetParameterBufferV2_perKernel, \
cudaLaunchDeviceV2_init_perWarp, cudaLaunchDevicV2_perKernel>"
                         "Default 7200,8000,100,12000,1600",
                         "7200,8000,100,12000,1600");
}

/*
该函数的功能是将纹理变量名绑定到纹理资源上。它是GPGPU-Sim中PTX指令编译器的一部分，用于处理PTX指令中
的纹理变量。它接受一个纹理变量名，以及指向纹理资源的指针，并将这两个参数绑定到一起，以便在运行时可以
使用。
*/
void gpgpu_t::gpgpu_ptx_sim_bindNameToTexture(
    const char *name, const struct textureReference *texref, int dim,
    int readmode, int ext) {
  std::string texname(name);
  if (m_NameToTextureRef.find(texname) == m_NameToTextureRef.end()) {
    m_NameToTextureRef[texname] = std::set<const struct textureReference *>();
  } else {
    const struct textureReference *tr = *m_NameToTextureRef[texname].begin();
    assert(tr != NULL);
    // asserts that all texrefs in set have same fields
    assert(tr->normalized == texref->normalized &&
           tr->filterMode == texref->filterMode &&
           tr->addressMode[0] == texref->addressMode[0] &&
           tr->addressMode[1] == texref->addressMode[1] &&
           tr->addressMode[2] == texref->addressMode[2] &&
           tr->channelDesc.x == texref->channelDesc.x &&
           tr->channelDesc.y == texref->channelDesc.y &&
           tr->channelDesc.z == texref->channelDesc.z &&
           tr->channelDesc.w == texref->channelDesc.w &&
           tr->channelDesc.f == texref->channelDesc.f);
  }
  m_NameToTextureRef[texname].insert(texref);
  m_TextureRefToName[texref] = texname;
  const textureReferenceAttr *texAttr = new textureReferenceAttr(
      texref, dim, (enum cudaTextureReadMode)readmode, ext);
  m_NameToAttribute[texname] = texAttr;
}

const char *gpgpu_t::gpgpu_ptx_sim_findNamefromTexture(
    const struct textureReference *texref) {
  std::map<const struct textureReference *, std::string>::const_iterator t =
      m_TextureRefToName.find(texref);
  assert(t != m_TextureRefToName.end());
  return t->second.c_str();
}

unsigned int intLOGB2(unsigned int v) {
  unsigned int shift;
  unsigned int r;

  r = 0;

  shift = ((v & 0xFFFF0000) != 0) << 4;
  v >>= shift;
  r |= shift;
  shift = ((v & 0xFF00) != 0) << 3;
  v >>= shift;
  r |= shift;
  shift = ((v & 0xF0) != 0) << 2;
  v >>= shift;
  r |= shift;
  shift = ((v & 0xC) != 0) << 1;
  v >>= shift;
  r |= shift;
  shift = ((v & 0x2) != 0) << 0;
  v >>= shift;
  r |= shift;

  return r;
}

/*
此函数用于将纹理引用（texref）绑定到cudaArray（array）上，以实现对纹理的读取操作。
首先，根据texref获取纹理名称texname，然后查找m_NameToCudaArray中是否存在texname，如果存在，则先将
其解绑，再将texname和array绑定到m_NameToCudaArray中。
然后，计算texel_size_bits，texel_size，Tx和Ty，并将Tx，Ty，Tx_numbits，Ty_numbits，texel_size和
texel_size_numbits放入textureInfo中，将textureInfo放入m_NameToTextureInfo中。
*/
void gpgpu_t::gpgpu_ptx_sim_bindTextureToArray(
    const struct textureReference *texref, const struct cudaArray *array) {
  std::string texname = gpgpu_ptx_sim_findNamefromTexture(texref);

  std::map<std::string, const struct cudaArray *>::const_iterator t =
      m_NameToCudaArray.find(texname);
  // check that there's nothing there first
  if (t != m_NameToCudaArray.end()) {
    printf(
        "GPGPU-Sim PTX:   Warning: binding to texref associated with %s, which "
        "was previously bound.\nImplicitly unbinding texref associated to %s "
        "first\n",
        texname.c_str(), texname.c_str());
  }
  m_NameToCudaArray[texname] = array;
  unsigned int texel_size_bits =
      array->desc.w + array->desc.x + array->desc.y + array->desc.z;
  unsigned int texel_size = texel_size_bits / 8;
  unsigned int Tx, Ty;
  int r;

  printf("GPGPU-Sim PTX:   texel size = %d\n", texel_size);
  printf("GPGPU-Sim PTX:   texture cache linesize = %d\n",
         m_function_model_config.get_texcache_linesize());
  // first determine base Tx size for given linesize
  switch (m_function_model_config.get_texcache_linesize()) {
    case 16:
      Tx = 4;
      break;
    case 32:
      Tx = 8;
      break;
    case 64:
      Tx = 8;
      break;
    case 128:
      Tx = 16;
      break;
    case 256:
      Tx = 16;
      break;
    default:
      printf(
          "GPGPU-Sim PTX:   Line size of %d bytes currently not supported.\n",
          m_function_model_config.get_texcache_linesize());
      assert(0);
      break;
  }
  r = texel_size >> 2;
  // modify base Tx size to take into account size of each texel in bytes
  while (r != 0) {
    Tx = Tx >> 1;
    r = r >> 2;
  }
  // by now, got the correct Tx size, calculate correct Ty size
  Ty = m_function_model_config.get_texcache_linesize() / (Tx * texel_size);

  printf(
      "GPGPU-Sim PTX:   Tx = %d; Ty = %d, Tx_numbits = %d, Ty_numbits = %d\n",
      Tx, Ty, intLOGB2(Tx), intLOGB2(Ty));
  printf("GPGPU-Sim PTX:   Texel size = %d bytes; texel_size_numbits = %d\n",
         texel_size, intLOGB2(texel_size));
  printf(
      "GPGPU-Sim PTX:   Binding texture to array starting at devPtr32 = 0x%x\n",
      array->devPtr32);
  printf("GPGPU-Sim PTX:   Texel size = %d bytes\n", texel_size);
  struct textureInfo *texInfo =
      (struct textureInfo *)malloc(sizeof(struct textureInfo));
  texInfo->Tx = Tx;
  texInfo->Ty = Ty;
  texInfo->Tx_numbits = intLOGB2(Tx);
  texInfo->Ty_numbits = intLOGB2(Ty);
  texInfo->texel_size = texel_size;
  texInfo->texel_size_numbits = intLOGB2(texel_size);
  m_NameToTextureInfo[texname] = texInfo;
}

void gpgpu_t::gpgpu_ptx_sim_unbindTexture(
    const struct textureReference *texref) {
  // assumes bind-use-unbind-bind-use-unbind pattern
  std::string texname = gpgpu_ptx_sim_findNamefromTexture(texref);
  m_NameToCudaArray.erase(texname);
  m_NameToTextureInfo.erase(texname);
}

//定义指令最大占用8字节大小==64 bit。
#define MAX_INST_SIZE 8 /*bytes*/

/*
每个内核函数在加载到GPGPU-Sim时都会被分析和预解码。当PTX解析器检测到一个内核函数的结束时，它会调用 
function_info::ptx_assemble()。这个函数做了以下工作：
1. 给函数中的每条指令分配一个唯一的PC
2. 将函数中的每个分支标签解析为对应的指令/PC，即确定每条分支指令的分支目标
3. 为该函数创建控制流图
4. 进行控制流分析
5. 预先对每条指令进行解码（以加快仿真速度）
该函数做了很多事情，包括将指令放入m_instr_mem指令存储、创建PC到指令的映射s_g_pc_to_insn、通过搜索
后必经结点分析基本块信息、分支/分歧信息、计算目标PC的分支指令等。
*/
void function_info::ptx_assemble() {
  //m_assembled是一个布尔类型的变量，用于表示GPGPU-Sim内核是否已经完成编译和装配。如果内核已经完成
  //编译和装配，则m_assembled变量的值为true，否则为false。
  if (m_assembled) {
    return;
  }

  // get the instructions into instruction memory...
  //将指令放入指令存储。
  //num_inst变量存储了指令条数。
  unsigned num_inst = m_instructions.size();
  //m_instr_mem_size变量存储了所有指令存储所需的存储空间大小，以字节为单位。
  m_instr_mem_size = MAX_INST_SIZE * (num_inst + 1);
  //ptx_instruction类是一个描述单条指令的类，在ptx_ir.h中定义，在这里定义的m_instr_mem实际上是一个
  //由类ptx_instruction的对象构成的一维数组。
  m_instr_mem = new ptx_instruction *[m_instr_mem_size];

  //m_name是由nvcc编译器编译后的PTX指令中为内核函数指定的唯一的函数名。
  printf("GPGPU-Sim PTX: instruction assembly for function \'%s\'... ",
         m_name.c_str());
  //使用多个输出函数连续进行多次输出时，有可能发现输出错误。因为下一个数据再上一个数据还没输出完毕，还
  //在输出缓冲区中时，下一个printf就把另一个数据加入输出缓冲区，结果冲掉了原来的数据，出现输出错误。在
  //prinf()后加上fflush(stdout);强制马上输出，避免错误。
  fflush(stdout);
  //定义ptx_instruction类构成的一维列表的迭代器，以用来遍历指令存储中的每条指令。
  std::list<ptx_instruction *>::iterator i;

  addr_t PC =
      gpgpu_ctx->func_sim->g_assemble_code_next_pc;  // globally unique address
                                                     // (across functions)
  // start function on an aligned address
  for (unsigned i = 0; i < (PC % MAX_INST_SIZE); i++)
    gpgpu_ctx->s_g_pc_to_insn.push_back((ptx_instruction *)NULL);
  PC += PC % MAX_INST_SIZE;
  m_start_PC = PC;

  addr_t n = 0;  // offset in m_instr_mem
  // Why s_g_pc_to_insn.size() is needed to reserve additional memory for insts?
  // reserve is cumulative. s_g_pc_to_insn.reserve(s_g_pc_to_insn.size() +
  // MAX_INST_SIZE*m_instructions.size());
  gpgpu_ctx->s_g_pc_to_insn.reserve(MAX_INST_SIZE * m_instructions.size());
  for (i = m_instructions.begin(); i != m_instructions.end(); i++) {
    ptx_instruction *pI = *i;
    //printf("\n!!!!!!!!!!!!!!!!!!INST-%d!!!!!!!!!!!!!!!!!!\n", i);
    //pI->print_insn();
    //fflush(stdout);

    //下面一个if块，首先判断指令pI是否是个标签。如果它是标签，则把它放入标签列表labels；如果不是标签，
    //就说明它是个正常的需要执行的指令，需要为它分配一个唯一的PC，并把它加入到指令存储m_instr_mem中。
    //label即为例如PTX指令块中的$L__BB0_6等：
    //01.$L__BB0_6: <---- label
    //02.  .pragma "nounroll";
    //03.  ld.global.u32 %r28, [%rd32];
    //04.  ...
    //12.  @%p5 bra $L__BB0_6; <---- label
    if (pI->is_label()) {
      //这里pI就是上层for循环的单条指令。
      const symbol *l = pI->get_label();
      //这里labels是一个字典，由std::map定义，即键为标签名例如上述的'$L__BB0_6'，值为当前指令的序号n，
      //这个n是按照指令字节数为一个单位大小编码的，即执行到第n条指令的时候就知道需要把'$L__BB0_6'标签
      //加进来。
      labels[l->name()] = n;
      //printf("\n!!!!=%d, name=%s\n", pI->is_label(), l->name().c_str ());
    } else {
      //将PC值对应的指令信息保存在gpgpu_ctx->func_sim->g_pc_to_finfo中。
      gpgpu_ctx->func_sim->g_pc_to_finfo[PC] = this;
      //将指令pI保存在指令存储m_instr_mem数组中。
      m_instr_mem[n] = pI;
      //s_g_pc_to_insn在gpgpu_context.h中定义，功能是一个直接从PC到指令的映射向量。
      //即访问s_g_pc_to_insn时给它PC，他就返回PC指向的那条指令pI。
      gpgpu_ctx->s_g_pc_to_insn.push_back(pI);
      //检查PC到指令的映射是否正确。
      assert(pI == gpgpu_ctx->s_g_pc_to_insn[PC]);
      //pI的m_instr_mem_index代表在指令存储m_instr_mem数组中的索引。
      pI->set_m_instr_mem_index(n);
      pI->set_PC(PC);
      assert(pI->inst_size() <= MAX_INST_SIZE);
      //MAX_INST_SIZE设置为8字节，这里可能是把没用到的字节全赋值NULL。
      for (unsigned i = 1; i < pI->inst_size(); i++) {
        gpgpu_ctx->s_g_pc_to_insn.push_back((ptx_instruction *)NULL);
        m_instr_mem[n + i] = NULL;
      }
      //n是m_instr_mem中按照指令字节数为一个单位大小编码的，PC也是按照指令字节数为一个单位大小编码的。
      n += pI->inst_size();
      PC += pI->inst_size();
      //printf("!!!!!n=%d,PC=%d\n", n, PC);
    }
  }

  //gpgpu_ctx->func_sim->g_assemble_code_next_pc保存了处理PTX代码过程中的PC值的变化，开始处理第一
  //条指令时，PC=gpgpu_ctx->func_sim->g_assemble_code_next_pc。在所有指令遍历完一遍后，把symbol和
  //非symbol指令区分开以后，就将gpgpu_ctx->func_sim->g_assemble_code_next_pc增至当前PC值，然后再
  //后续处理分支指令，将函数中的每个分支标签解析为对应的指令/PC，即确定每条分支指令的分支目标。
  gpgpu_ctx->func_sim->g_assemble_code_next_pc = PC;
  //下面的循环即为处理分支指令。
  for (unsigned ii = 0; ii < n;
       ii += m_instr_mem[ii]->inst_size()) {  // handle branch instructions
       //再次循环这n字节大小的指令，寻找哪些指令有分支，比如BRA_OP、CALLP_OP、BREAKADDR_OP指令。这些指
	   //令在opcodes.def中对应PTX指令，例如BRA_OP->bra，CALLP_OP->callp，BREAKADDR_OP->breakaddr。
    ptx_instruction *pI = m_instr_mem[ii];
    if (pI->get_opcode() == BRA_OP || pI->get_opcode() == BREAKADDR_OP ||
        pI->get_opcode() == CALLP_OP) {
      // get operand, e.g. target name
      //获取跳转到指令的标签，在ptx_ir.h中定义：return m_operands[0]。
      operand_info &target = pI->dst();  // get operand, e.g. target name
      //printf("!!!!target.name()===%s\n", target.name().c_str());
      //这里是如果该条指令是分支指令，则其最后面跳转到的指令标签应该在前面的labels字典里，因为第一次
      //遍历的时候已经放进去了，所以下面一个if块是判断分支指令的跳转标签是否合法。
      if (labels.find(target.name()) == labels.end()) {
        printf(
            "GPGPU-Sim PTX: Loader error (%s:%u): Branch label \"%s\" does not "
            "appear in assembly code.",
            pI->source_file(), pI->source_line(), target.name().c_str());
        abort();
      }
      //labels[target.name()]获取的是taget这条指令所在的指令存储m_instr_mem中，以8字节为一个单位的
      //第几字节。后续从指令存储m_instr_mem中读一条指令的时候，把n作为索引传进去。这一步即为将标签作
      //为索引传进labels查询它的索引n，再用n查找m_instr_mem查询这条指令的PC值，然后再把m_symtab里的
      //标签替换为对应的PC值。
      unsigned index = labels[target.name()];  // determine address from name
      unsigned PC = m_instr_mem[index]->get_PC();
      m_symtab->set_label_address(target.get_symbol(), PC);
      target.set_type(label_t);
    }
  }
  //offset in m_instr_mem (used in do_pdom)
  //这里后面post-dominators分析要继续在m_instr_mem[n...]后面添加新的指令，所以这里暂存一下m_n。
  m_n = n;
  printf("  done.\n");
  fflush(stdout);

  // disable pdom analysis  here and do it at runtime
  //这里在处理PTX指令时禁用了post-dominators分析，转为在运行时再分析。
#if 0
   printf("GPGPU-Sim PTX: finding reconvergence points for \'%s\'...\n", m_name.c_str() );
   create_basic_blocks();
   connect_basic_blocks();
   bool modified = false; 
   do {
      //寻找一个函数的完整PTX指令中，每一个（基本块）结点的必经结点（dominators）。
      find_dominators();
      //寻找一个函数的完整PTX指令中，每一个（基本块）结点的直接必经结点（immediate dominators）。
      find_idominators();
      modified = connect_break_targets(); 
   } while (modified == true);

   if ( g_debug_execution>=50 ) {
      print_basic_blocks();
      print_basic_block_links();
      print_basic_block_dot();
   }
   if ( g_debug_execution>=2 ) {
      print_dominators();
   }
   //寻找一个函数的完整PTX指令中，每一个（基本块）结点的后必经结点（post-dominators）。
   find_postdominators();
   //寻找一个函数的完整PTX指令中，每一个（基本块）结点的直接后必经结点（immediate post-dominators）。
   find_ipostdominators();
   if ( g_debug_execution>=50 ) {
      print_postdominators();
      print_ipostdominators();
   }

   printf("GPGPU-Sim PTX: pre-decoding instructions for \'%s\'...\n", m_name.c_str() );
   //对m_instr_mem中的每一条指令进行预处理。
   for ( unsigned ii=0; ii < n; ii += m_instr_mem[ii]->inst_size() ) { // handle branch instructions
      ptx_instruction *pI = m_instr_mem[ii];
      pI->pre_decode();
   }
   printf("GPGPU-Sim PTX: ... done pre-decoding instructions for \'%s\'.\n", m_name.c_str() );
   fflush(stdout);

   m_assembled = true;
#endif
}

/*
如果存储器指令未指定状态空间，则使用通用寻址来执行操作。状态空间：.const、.param、.local和.shared被
建模为通用地址空间内的窗口。每个窗口由其window-base和等于该状态空间的大小的window-size来定义。在这种
地址空间划分下，每个线程最多只能有 8kB 的本地内存（LOCAL_MEM_SIZE_MAX）。在CUDA计算能力1.3及以后版
本中，每个线程最多可以有 16kB 的本地内存。在CUDA计算能力2.0的情况下，这一限制增加到 512kB 。用户可以
增加 LOCAL_MEM_SIZE_MAX 来支持每个线程需要超过 8kB 本地内存的应用。然而，应该始终确保：
GLOBAL_HEAP_START > (TOTAL_LOCAL_MEM + TOTAL_SHARED_MEM)。

//GLOBAL_HEAP在GPGPU-Sim中的起始地址为：0x80000000。
GLOBAL_HEAP_START              = 0x80000000
//单个SM所拥有的SHARED_MEM最大空间：64*1024
SHARED_MEM_SIZE_MAX            = 64*1024
//单个线程最多只能有 8kB 的本地内存。
LOCAL_MEM_SIZE_MAX             = 8*1024
//最大SM数量。
MAX_STREAMING_MULTIPROCESSORS  = 64
//每个SM的最大线程数量。
MAX_THREAD_PER_SM              = 2048

TOTAL_LOCAL_MEM_PER_SM         = MAX_THREAD_PER_SM*LOCAL_MEM_SIZE_MAX
TOTAL_SHARED_MEM               = MAX_STREAMING_MULTIPROCESSORS*SHARED_MEM_SIZE_MAX
TOTAL_LOCAL_MEM                = MAX_STREAMING_MULTIPROCESSORS*MAX_THREAD_PER_SM*LOCAL_MEM_SIZE_MAX
//在内存空间中，分为多个部分，首先放global memory-part1，再放local memory-part2，再放shared memory-part3，
//再放global memory-part4；其中：
//global memory-part1的范围为：addr < STATIC_ALLOC_LIMIT
//global memory-part4的范围为：addr >= GLOBAL_HEAP_START
//local memory-part2的范围为：STATIC_ALLOC_LIMIT <= addr < SHARED_GENERIC_START
//shared memory-part3的范围为：SHARED_GENERIC_START <= addr < GLOBAL_HEAP_START

SHARED_GENERIC_START           = GLOBAL_HEAP_START-TOTAL_SHARED_MEM
LOCAL_GENERIC_START            = SHARED_GENERIC_START-TOTAL_LOCAL_MEM
STATIC_ALLOC_LIMIT             = GLOBAL_HEAP_START - (TOTAL_LOCAL_MEM+TOTAL_SHARED_MEM)
*/

/*
传入的参数包括SM的index，和在当前SM内对shared memory的相对地址，即在该SM内，addr的初始值为0。故该函
数为对shared memory的通用寻址，是为了将此相对地址转为在内存空间里的绝对地址。
*/
addr_t shared_to_generic(unsigned smid, addr_t addr) {
  //printf("!!!=smid=%d, addr=%x\n", smid, addr);
  assert(addr < SHARED_MEM_SIZE_MAX);
  return SHARED_GENERIC_START + smid * SHARED_MEM_SIZE_MAX + addr;
}

/*
全局地址采用绝对地址，因此对global memory的通用寻址不需要对地址进行变换。
*/
addr_t global_to_generic(addr_t addr) { return addr; }

/*
传入的参数包括SM的index，和在当前SM内对shared memory的相对地址。shared memory的地址范围为：
SHARED_GENERIC_START<= addr < GLOBAL_HEAP_START。
但该函数是判断地址addr是否是位于id所代表的SM的shared memory。因此，首先，先计算id代表的当前SM的shred 
memory的起始地址start和终止地址end。若start <= addr < end，则在id所代表的SM的shared memory范围内。
*/
bool isspace_shared(unsigned smid, addr_t addr) {
  //计算id代表的当前SM的shred memory的起始地址。
  addr_t start = SHARED_GENERIC_START + smid * SHARED_MEM_SIZE_MAX;
  //计算id代表的当前SM的shred memory的终止地址。
  addr_t end = SHARED_GENERIC_START + (smid + 1) * SHARED_MEM_SIZE_MAX;
  if ((addr >= end) || (addr < start)) return false;
  return true;
}

/*
此函数用于判断地址是否位于global memory的范围内：
global memory-part1的范围为：addr < STATIC_ALLOC_LIMIT
global memory-part4的范围为：addr >= GLOBAL_HEAP_START
*/
bool isspace_global(addr_t addr) {
  return (addr >= GLOBAL_HEAP_START) || (addr < STATIC_ALLOC_LIMIT);
}

/*
此函数用于判断地址是否位于global memory或shared memory或local memory的范围内。
*/
memory_space_t whichspace(addr_t addr) {
  if ((addr >= GLOBAL_HEAP_START) || (addr < STATIC_ALLOC_LIMIT)) {
    return global_space;
  } else if (addr >= SHARED_GENERIC_START) {
    return shared_space;
  } else {
    return local_space;
  }
}

/*
与shared_to_generic函数相反，该函数为对shared memory的相对寻址，是为了将此绝对地址转为在id所代表的SM
内shared memory的相对地址。即addr在绝对地址空间，将其转换为相对地址空间。id所代表的SM内shared memory
的起始地址为(SHARED_GENERIC_START + smid * SHARED_MEM_SIZE_MAX，因此绝对地址addr减去这一起始地址即
转换为相对地址。
*/
addr_t generic_to_shared(unsigned smid, addr_t addr) {
  assert(isspace_shared(smid, addr));
  return addr - (SHARED_GENERIC_START + smid * SHARED_MEM_SIZE_MAX);
}

/*
传入的参数包括SM的index，和在当前SM内且hwtid所代表的线程的local memory的相对地址addr，即在该SM内且
hwtid所代表的线程的local memory内，addr的初始值为0。故该函数为对local memory的通用寻址，是为了将此相
对地址转为在内存空间里的绝对地址。
*/
addr_t local_to_generic(unsigned smid, unsigned hwtid, addr_t addr) {
  assert(addr < LOCAL_MEM_SIZE_MAX);
  return LOCAL_GENERIC_START + (TOTAL_LOCAL_MEM_PER_SM * smid) +
         (LOCAL_MEM_SIZE_MAX * hwtid) + addr;
}

/*
传入的参数包括SM的index，线程index，以及绝对地址。该函数是判断绝对地址addr是否是位于id所代表的SM内，
hwtid所代表的线程的local memory空间内。因此，首先，先计算id代表的当前SM内且hwtid所代表的线程的local 
memory的起始地址start和终止地址end。若start <= addr < end，则在id所代表的SM内且hwtid所代表的线程的
local memory范围内。
*/
bool isspace_local(unsigned smid, unsigned hwtid, addr_t addr) {
  addr_t start = LOCAL_GENERIC_START + (TOTAL_LOCAL_MEM_PER_SM * smid) +
                 (LOCAL_MEM_SIZE_MAX * hwtid);
  addr_t end = LOCAL_GENERIC_START + (TOTAL_LOCAL_MEM_PER_SM * smid) +
               (LOCAL_MEM_SIZE_MAX * (hwtid + 1));
  if ((addr >= end) || (addr < start)) return false;
  return true;
}

/*
与local_to_generic函数相反，该函数为对local memory的相对寻址，是为了将此内存空间内的绝对地址addr转为
在id所代表的SM内且hwtid所代表的线程的local memory的相对地址。
*/
addr_t generic_to_local(unsigned smid, unsigned hwtid, addr_t addr) {
  assert(isspace_local(smid, hwtid, addr));
  return addr - (LOCAL_GENERIC_START + (TOTAL_LOCAL_MEM_PER_SM * smid) +
                 (LOCAL_MEM_SIZE_MAX * hwtid));
}

/*
全局地址采用绝对地址，因此对global memory的通用寻址不需要对地址进行变换。
*/
addr_t generic_to_global(addr_t addr) { return addr; }

/*
在设备端分配一块大小为size的内存区域，分配地址为m_dev_malloc，干函数分配后返回该地址。
*/
void *gpgpu_t::gpu_malloc(size_t size) {
  //m_dev_malloc在abstract_hardware_model.h中的gpgpu_t类中定义，并在abstract_hardware_model.cc中
  //被初始化为GLOBAL_HEAP_START。之后每次分配新的存储空间，就将其增加到下一个256 byte对齐的地址。
  unsigned long long result = m_dev_malloc;
  if (g_debug_execution >= 3) {
    printf(
        "GPGPU-Sim PTX: allocating %zu bytes on GPU starting at address "
        "0x%Lx\n",
        size, m_dev_malloc);
    fflush(stdout);
  }
  m_dev_malloc += size;
  // align to 256 byte boundaries
  //对齐下一个256 byte地址。
  if (size % 256)
    m_dev_malloc += (256 - size % 256);  // align to 256 byte boundaries
  return (void *)result;
}

/*
与gpu_malloc(size_t size)函数相同。
*/
void *gpgpu_t::gpu_mallocarray(size_t size) {
  unsigned long long result = m_dev_malloc;
  if (g_debug_execution >= 3) {
    printf(
        "GPGPU-Sim PTX: allocating %zu bytes on GPU starting at address "
        "0x%Lx\n",
        size, m_dev_malloc);
    fflush(stdout);
  }
  m_dev_malloc += size;
  // align to 256 byte boundaries
  //对齐下一个256 byte地址。
  if (size % 256)
    m_dev_malloc += (256 - size % 256);  // align to 256 byte boundaries
  return (void *)result;
}

/*
将主机端的数据拷贝到设备端，传入参数包括[源]主机端地址src，[目的]设备端地址dst_start_addr，和拷贝的字
节数count。
*/
void gpgpu_t::memcpy_to_gpu(size_t dst_start_addr, const void *src,
                            size_t count) {
  if (g_debug_execution >= 3) {
    printf(
        "GPGPU-Sim PTX: copying %zu bytes from CPU[0x%Lx] to GPU[0x%Lx] ... ",
        count, (unsigned long long)src, (unsigned long long)dst_start_addr);
    fflush(stdout);
  }
  char *src_data = (char *)src;
  for (unsigned n = 0; n < count; n++)
    m_global_mem->write(dst_start_addr + n, 1, src_data + n, NULL, NULL);

  // Copy into the performance model.
  // extern gpgpu_sim* g_the_gpu;
  //将数据拷贝到性能模型中。
  gpgpu_ctx->the_gpgpusim->g_the_gpu->perf_memcpy_to_gpu(dst_start_addr, count);
  if (g_debug_execution >= 3) {
    printf(" done.\n");
    fflush(stdout);
  }
}

/*
将设备端的数据拷贝到主机端，传入参数包括[目的]主机端地址dst，[源]设备端地址src_start_addr，和拷贝的字
节数count。
*/
void gpgpu_t::memcpy_from_gpu(void *dst, size_t src_start_addr, size_t count) {
  if (g_debug_execution >= 3) {
    printf("GPGPU-Sim PTX: copying %zu bytes from GPU[0x%Lx] to CPU[0x%Lx] ...",
           count, (unsigned long long)src_start_addr, (unsigned long long)dst);
    fflush(stdout);
  }
  unsigned char *dst_data = (unsigned char *)dst;
  for (unsigned n = 0; n < count; n++)
    m_global_mem->read(src_start_addr + n, 1, dst_data + n);

  // Copy into the performance model.
  // extern gpgpu_sim* g_the_gpu;
  //将数据拷贝到性能模型中。
  gpgpu_ctx->the_gpgpusim->g_the_gpu->perf_memcpy_to_gpu(src_start_addr, count);
  if (g_debug_execution >= 3) {
    printf(" done.\n");
    fflush(stdout);
  }
}

/*
将设备端的数据拷贝到设备端另一块内存区域，传入参数包括[目的]主机端地址dst，[源]设备端地址src，和拷贝的
字节数count。
*/
void gpgpu_t::memcpy_gpu_to_gpu(size_t dst, size_t src, size_t count) {
  if (g_debug_execution >= 3) {
    printf("GPGPU-Sim PTX: copying %zu bytes from GPU[0x%Lx] to GPU[0x%Lx] ...",
           count, (unsigned long long)src, (unsigned long long)dst);
    fflush(stdout);
  }
  for (unsigned n = 0; n < count; n++) {
    unsigned char tmp;
    m_global_mem->read(src + n, 1, &tmp);
    m_global_mem->write(dst + n, 1, &tmp, NULL, NULL);
  }
  if (g_debug_execution >= 3) {
    printf(" done.\n");
    fflush(stdout);
  }
}

/*
memset是计算机中C/C++语言初始化函数。作用是将某一块内存中的内容全部设置为指定的值，这个函数通常为新申请
的内存做初始化工作。这里gpu_memset的功能是，在dst_start_addr为初始地址的连续count个字节的内存区域里，
将这些count个字节的每个字节的初始值都设置为c。虽然传入参数是int c，但是会将其转换为单字节char类型再写入
每个byte大小的内存区域里。
*/
void gpgpu_t::gpu_memset(size_t dst_start_addr, int c, size_t count) {
  if (g_debug_execution >= 3) {
    printf(
        "GPGPU-Sim PTX: setting %zu bytes of memory to 0x%x starting at "
        "0x%Lx... ",
        count, (unsigned char)c, (unsigned long long)dst_start_addr);
    fflush(stdout);
  }
  unsigned char c_value = (unsigned char)c;
  for (unsigned n = 0; n < count; n++)
    m_global_mem->write(dst_start_addr + n, 1, &c_value, NULL, NULL);
  if (g_debug_execution >= 3) {
    printf(" done.\n");
    fflush(stdout);
  }
}

/*
给定PC值，以字符串的形式打印出该PC对应的指令。fp是错误信息的输出文件。
*/
void cuda_sim::ptx_print_insn(address_type pc, FILE *fp) {
  std::map<unsigned, function_info *>::iterator f = g_pc_to_finfo.find(pc);
  if (f == g_pc_to_finfo.end()) {
    fprintf(fp, "<no instruction at address 0x%llx>", pc);
    return;
  }
  function_info *finfo = f->second;
  assert(finfo);
  finfo->print_insn(pc, fp);
}

/*
给定PC值，以字符串的形式返回该PC对应的指令。fp是错误信息的输出文件。
*/
std::string cuda_sim::ptx_get_insn_str(address_type pc) {
  std::map<unsigned, function_info *>::iterator f = g_pc_to_finfo.find(pc);
  if (f == g_pc_to_finfo.end()) {
#define STR_SIZE 255
    char buff[STR_SIZE];
    buff[STR_SIZE - 1] = '\0';
    snprintf(buff, STR_SIZE, "<no instruction at address 0x%llx>", pc);
    return std::string(buff);
  }
  function_info *finfo = f->second;
  assert(finfo);
  return finfo->get_insn_str(pc);
}

/*
oprnd_type在abstract_hardware_model.h中定义，用于标识操作码m_opcode是整型还是浮点操作。下列XXX_OP在
opcodes.def文件中都有其对应的PTX指令操作码。
下述指令均为简述，详见PTX指令集手册：
1. MEMBAR_OP = membar：强制执行内存操作的排序。membar.level指令保证此线程请求的先前内存访问（ld、st、
atom与red指令）在指定level执行，然后此线程在membar指令之后请求的后续内存操作才会执行。
2. SSY_OP = ssy：最新的PTX手册没有提到这一指令，公开存在的唯一证据是cuobjdump的反汇编输出，处理分支时
使用。
3. BRA_OP = bra：跳转到目标指令。
4. BAR_OP = bar：在CTA内执行屏障同步和通信。
5. RET_OP = ret：调用后从函数返回到指令。将执行返回到调用方的环境。发散返回将挂起线程，直到所有线程都准
备好返回调用方。这允许多个不同的ret指令。
6. RETP_OP = retp：最新的PTX手册没有提到这一指令，应该与ret指令类似的功能。
7. NOP_OP = nop：空指令。
8. EXIT_OP = exit：终止线程。
9. CALLP_OP = callp：声明用于间接调用的原型。定义没有特定函数名的原型，并将原型与标签关联。然后，原型可
用于间接调用指令，其中不完全了解可能的调用目标。
10. CALL_OP = call：调用一个函数，记录返回位置。
11. CVT_OP = cvt：将值从一种类型转换为另一种类型。
12. SET_OP = set：使用关系运算符比较两个数值，并通过应用布尔运算符将此结果与谓词值组合（可选）。
13. SLCT_OP = slct：根据第三个操作数的符号选择一个源操作数。
*/
void ptx_instruction::set_fp_or_int_archop() {
  oprnd_type = UN_OP;
  if ((m_opcode == MEMBAR_OP) || (m_opcode == SSY_OP) || (m_opcode == BRA_OP) ||
      (m_opcode == BAR_OP) || (m_opcode == RET_OP) || (m_opcode == RETP_OP) ||
      (m_opcode == NOP_OP) || (m_opcode == EXIT_OP) || (m_opcode == CALLP_OP) ||
      (m_opcode == CALL_OP)) {
    // do nothing
    //这些指令不涉及操作数的数据类型。
  } else if ((m_opcode == CVT_OP || m_opcode == SET_OP ||
              m_opcode == SLCT_OP)) {
    //例如指令：cvt.frnd2{.relu}.f16x2.f32  d, a, b; 是将FP32类型的源操作数a和b转换为两个FP16类型，并
    //将a转换成的数据放到高16位，将b转换成的数据放到低16位，打包成一个数据放到目的地址d。
    //因此，需要获取指令字符串的第二个操作类型，使用get_type2()来获取。
    if (get_type2() == F16_TYPE || get_type2() == F32_TYPE ||
        get_type2() == F64_TYPE || get_type2() == FF64_TYPE) {
      oprnd_type = FP_OP;
    } else
      oprnd_type = INT_OP;

  } else {
    //其余指令字符串只有一个操作数，因此直接用get_type()获取即可。
    if (get_type() == F16_TYPE || get_type() == F32_TYPE ||
        get_type() == F64_TYPE || get_type() == FF64_TYPE) {
      oprnd_type = FP_OP;
    } else
      oprnd_type = INT_OP;
  }
}

/*
sp_type在abstract_hardware_model.h中定义，它代表特定的指令类型，可以用来标记不同的指令，以便跟踪指令的
执行情况。它主要用于模拟GPGPU上的指令流，以及指令的时间和资源消耗。
*/
void ptx_instruction::set_mul_div_or_other_archop() {
  sp_op = OTHER_OP;
  if ((m_opcode != MEMBAR_OP) && (m_opcode != SSY_OP) && (m_opcode != BRA_OP) &&
      (m_opcode != BAR_OP) && (m_opcode != EXIT_OP) && (m_opcode != NOP_OP) &&
      (m_opcode != RETP_OP) && (m_opcode != RET_OP) && (m_opcode != CALLP_OP) &&
      (m_opcode != CALL_OP)) {
    //这些指令不涉及操作数的数据类型，因此没有乘除或者其他类型的计算操作。
    //用get_type()获取操作数的数据类型，包括F32_TYPE/F64_TYPE/FF64_TYPE。最新的PTX指令集手册里包含的
    //数据类型有：
    //PTX指令集手册里定义的基本类型说明符：
    //Basic Type         |  Fundamental Type Specifiers
    //--------------------------------------------------
    //Signed integer     |  .s8, .s16, .s32, .s64
    //Unsigned integer   |  .u8, .u16, .u32, .u64
    //Floating-point     |  .f16, .f16x2, .f32, .f64
    //Bits (untyped)     |  .b8, .b16, .b32, .b64
    //Predicate          |  .pred
    //其中，F32_TYPE指FP32数据类型，F64_TYPE指FP64数据类型，FF64_TYPE最新的PTX指令集手册没有找到。
    if (get_type() == F64_TYPE || get_type() == FF64_TYPE) {
      switch (get_opcode()) {
        case MUL_OP:
        case MAD_OP:
        case FMA_OP:
          sp_op = DP_MUL_OP;
          break;
        case DIV_OP:
        case REM_OP:
          sp_op = DP_DIV_OP;
          break;
        case RCP_OP:
          sp_op = DP_DIV_OP;
          break;
        case LG2_OP:
          sp_op = FP_LG_OP;
          break;
        case RSQRT_OP:
        case SQRT_OP:
          sp_op = FP_SQRT_OP;
          break;
        case SIN_OP:
        case COS_OP:
          sp_op = FP_SIN_OP;
          break;
        case EX2_OP:
          sp_op = FP_EXP_OP;
          break;
        case MMA_OP:
          sp_op = TENSOR__OP;
          break;
        case TEX_OP:
          sp_op = TEX__OP;
          break;
        default:
          if ((op == DP_OP) || (op == ALU_OP)) sp_op = DP___OP;
          break;
      }
    } else if (get_type() == F16_TYPE || get_type() == F32_TYPE) {
      //get_opcode()在ptx_ir.h中定义，获取指令的操作码。
      switch (get_opcode()) {
        case MUL_OP:   //乘法
        case MAD_OP:   //乘加
        case FMA_OP:
          sp_op = FP_MUL_OP;
          break;
        case DIV_OP:   //除法
        case REM_OP:
          sp_op = FP_DIV_OP;
          break;
        case RCP_OP:
          sp_op = FP_DIV_OP;
          break;
        case LG2_OP:   //对数计算
          sp_op = FP_LG_OP;
          break;
        case RSQRT_OP: //开平方+取倒数
        case SQRT_OP:  //开平方
          sp_op = FP_SQRT_OP;
          break;
        case SIN_OP:   //Sin函数
        case COS_OP:   //Cos函数
          sp_op = FP_SIN_OP;
          break;
        case EX2_OP:   //乘方计算
          sp_op = FP_EXP_OP;
          break;
        case MMA_OP:
          sp_op = TENSOR__OP;
          break;
        case TEX_OP:
          sp_op = TEX__OP;
          break;
        default:
          if ((op == SP_OP) || (op == ALU_OP)) sp_op = FP__OP;
          break;
      }
    } else {
      //除F32_TYPE/F64_TYPE/FF64_TYPE浮点数据类型外，还有些整数/无类型的数据类型。
      switch (get_opcode()) {
        case MUL24_OP: //两个24位整数乘法
        case MAD24_OP: //三个24位整数乘加
          sp_op = INT_MUL24_OP;
          break;
        case MUL_OP:   //乘法
        case MAD_OP:   //乘加
        case FMA_OP:
		  //U32_TYPE指无符号32位整数，S32_TYPE指有符号32位整数，B32_TYPE指无类型的32位数。
          if (get_type() == U32_TYPE || get_type() == S32_TYPE ||
              get_type() == B32_TYPE)
            sp_op = INT_MUL32_OP;
          else
            sp_op = INT_MUL_OP;
          break;
        case DIV_OP:   //除法
        case REM_OP:
          sp_op = INT_DIV_OP;
          break;
        case MMA_OP:
          sp_op = TENSOR__OP;
          break;
        case TEX_OP:
          sp_op = TEX__OP;
          break;
        default:
          if ((op == INTP_OP) || (op == ALU_OP)) sp_op = INT__OP;
          break;
      }
    }
  }
}

/*
不同的屏障操作有不同类型，例如PTX指令集手册9.7.12节中给出了如下并行同步和通信指令：
‣ bar{.cta}, barrier{.cta}
‣ barrier.cluster
‣ bar.warp.sync
‣ membar
‣ atom
‣ red
‣ vote
‣ match.sync
‣ activemask
‣ redux.sync
‣ griddepcontrol
‣ elect.sync
‣ mbarrier.init
‣ mbarrier.inval
‣ mbarrier.arrive
‣ mbarrier.arrive_drop
‣ mbarrier.test_wait
‣ mbarrier.try_wait
‣ mbarrier.pending_count
‣ cp.async.mbarrier.arrive
??? 需要用到再补充。
*/
void ptx_instruction::set_bar_type() {
  if (m_opcode == BAR_OP) {
    switch (m_barrier_op) {
      case SYNC_OPTION:
        bar_type = SYNC;
        break;
      case ARRIVE_OPTION:
        bar_type = ARRIVE;
        break;
      case RED_OPTION:
        bar_type = RED;
        switch (m_atomic_spec) {
          case ATOMIC_POPC:
            red_type = POPC_RED;
            break;
          case ATOMIC_AND:
            red_type = AND_RED;
            break;
          case ATOMIC_OR:
            red_type = OR_RED;
            break;
        }
        break;
      default:
        abort();
    }
  } else if (m_opcode == SST_OP) {
    bar_type = SYNC;
  }
}

/*
设置每种类型指令的操作码和延时。
*/
void ptx_instruction::set_opcode_and_latency() {
  unsigned int_latency[6];
  unsigned fp_latency[5];
  unsigned dp_latency[5];
  unsigned sfu_latency;
  unsigned tensor_latency;
  unsigned int_init[6];
  unsigned fp_init[5];
  unsigned dp_init[5];
  unsigned sfu_init;
  unsigned tensor_init;
  //下列scanf的列表每个元素代表的指令：
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
  //[0]=ADD,SUB  [1]=MAX,Min  [2]=MUL  [3]=MAD  [4]=DIV
  sscanf(gpgpu_ctx->func_sim->opcode_latency_fp, "%u,%u,%u,%u,%u",
         &fp_latency[0], &fp_latency[1], &fp_latency[2], &fp_latency[3],
         &fp_latency[4]);
  //[0]=ADD,SUB  [1]=MAX,Min  [2]=MUL  [3]=MAD  [4]=DIV
  sscanf(gpgpu_ctx->func_sim->opcode_latency_dp, "%u,%u,%u,%u,%u",
         &dp_latency[0], &dp_latency[1], &dp_latency[2], &dp_latency[3],
         &dp_latency[4]);
  //SFU-特殊功能单元指令操作码延迟，所有的Div指令在特殊功能单元执行。
  sscanf(gpgpu_ctx->func_sim->opcode_latency_sfu, "%u", &sfu_latency);
  //Tensor指令操作码延迟。
  sscanf(gpgpu_ctx->func_sim->opcode_latency_tensor, "%u", &tensor_latency);
  //[0]=ADD,SUB  [1]=MAX,Min  [2]=MUL  [3]=MAD  [4]=DIV  [5]=SHFL
  sscanf(gpgpu_ctx->func_sim->opcode_initiation_int, "%u,%u,%u,%u,%u,%u",
         &int_init[0], &int_init[1], &int_init[2], &int_init[3], &int_init[4],
         &int_init[5]);
  //[0]=ADD,SUB  [1]=MAX,Min  [2]=MUL  [3]=MAD  [4]=DIV
  sscanf(gpgpu_ctx->func_sim->opcode_initiation_fp, "%u,%u,%u,%u,%u",
         &fp_init[0], &fp_init[1], &fp_init[2], &fp_init[3], &fp_init[4]);
  //[0]=ADD,SUB  [1]=MAX,Min  [2]=MUL  [3]=MAD  [4]=DIV
  sscanf(gpgpu_ctx->func_sim->opcode_initiation_dp, "%u,%u,%u,%u,%u",
         &dp_init[0], &dp_init[1], &dp_init[2], &dp_init[3], &dp_init[4]);
  //SFU-特殊功能单元指令操作码初始化周期，，所有的Div指令在特殊功能单元执行。
  sscanf(gpgpu_ctx->func_sim->opcode_initiation_sfu, "%u", &sfu_init);
  //Tensor指令操作码初始化周期。
  sscanf(gpgpu_ctx->func_sim->opcode_initiation_tensor, "%u", &tensor_init);
  //下列CUDA API的延迟：
  //cudaStreamCreateWithFlags, cudaGetParameterBufferV2_init_perWarp, 
  //cudaGetParameterBufferV2_perKernel, cudaLaunchDeviceV2_init_perWarp, 
  //cudaLaunchDevicV2_perKernel>
  sscanf(gpgpu_ctx->func_sim->cdp_latency_str, "%u,%u,%u,%u,%u",
         &gpgpu_ctx->func_sim->cdp_latency[0],
         &gpgpu_ctx->func_sim->cdp_latency[1],
         &gpgpu_ctx->func_sim->cdp_latency[2],
         &gpgpu_ctx->func_sim->cdp_latency[3],
         &gpgpu_ctx->func_sim->cdp_latency[4]);
  //这里计算操作数需要的寄存器个数，m_operands在ptx_ir.h的ptx_instruction类中定义：
  //    std::vector<operand_info> m_operands;
  //m_operands会在每条指令解析的时候将所有操作数都添加到其中，例如解析 mad a,b,c 指令时，会将 a,b,c
  //三个操作数添加进m_operands，即每一条指令对象有一个操作数向量m_operands。
  if (!m_operands.empty()) {
    std::vector<operand_info>::iterator it;
    for (it = ++m_operands.begin(); it != m_operands.end(); it++) {
      //操作数数量计数。
      num_operands++;
      //如果操作数是寄存器或者是矢量，寄存器数量加1。
      if ((it->is_reg() || it->is_vector())) {
        num_regs++;
      }
    }
  }
  //先默认op为ALU操作类型，后面依据LOAD/STORE等指令再修改。
  op = ALU_OP;
  //先默认内存操作类型mem_op为非纹理访存类型，后面依据TEX_OP指令再修改。
  mem_op = NOT_TEX;
  //initiation_interval：指令发射间隔，即每次指令发射之间的时间间隔，单位为时钟周期。
  //latency：延迟，指令从发射到执行完毕所需的时间，单位为时钟周期。
  initiation_interval = latency = 1;
  //依据指令的操作码来对op和mem_op修改。
  switch (m_opcode) {
    //mov指令为数据移动指令：has_memory_read()为真时，代表LOAD；has_memory_write()为真时，代表STORE。
    case MOV_OP:       //数据移动指令
      assert(!(has_memory_read() && has_memory_write()));
      if (has_memory_read()) op = LOAD_OP;
      if (has_memory_write()) op = STORE_OP;
      break;
    case LD_OP:        //LOAD指令
      op = LOAD_OP;
      break;
    case MMA_LD_OP:    //Tensor Core上的wmma指令
      op = TENSOR_CORE_LOAD_OP;
      break;
    case LDU_OP:       //从warp中各线程共同的地址加载只读数据
      op = LOAD_OP;
      break;
    case ST_OP:        //STORE指令
      op = STORE_OP;
      break;
    case MMA_ST_OP:    //Tensor Core上的store指令
      op = TENSOR_CORE_STORE_OP;
      break;
    case BRA_OP:       //跳转指令
      op = BRANCH_OP;
      break;
    case BREAKADDR_OP: //跳转指令
      op = BRANCH_OP;
      break;
    case TEX_OP:       //纹理内存查找指令
      op = LOAD_OP;
      mem_op = TEX;
      break;
    case ATOM_OP:      //线程与线程之间通信的Atomic reduction操作指令
      op = LOAD_OP;
      break;
    case BAR_OP:       //屏障指令
      op = BARRIER_OP;
      break;
    case SST_OP:       //???屏障指令
      op = BARRIER_OP;
      break;
    case MEMBAR_OP:    //强制执行内存操作的排序指令
      op = MEMORY_BARRIER_OP;
      break;
    case CALL_OP: {    //调用函数并记录返回位置
      //在ptx_ir.h中定义：API为vprintf时，if (fname == "vprintf") {m_is_printf = true;}
      //在ptx_ir.h中定义：
      // API为cudaStreamCreateWithFlags时，if (fname == "cudaStreamCreateWithFlags") m_is_cdp = 1;
      // API为cudaGetParameterBufferV2时，if (fname == "cudaGetParameterBufferV2") m_is_cdp = 2;
      // API为cudaLaunchDeviceV2时，if (fname == "cudaLaunchDeviceV2") m_is_cdp = 4;
      if (m_is_printf || m_is_cdp) {
        op = ALU_OP;
      } else
        op = CALL_OPS;
      break;
    }
    case CALLP_OP: {
      if (m_is_printf || m_is_cdp) {
        op = ALU_OP;
      } else
        op = CALL_OPS;
      break;
    }
    case RET_OP:
    case RETP_OP:
      op = RET_OPS;
      break;
    case ADD_OP:
    case ADDP_OP:
    case ADDC_OP:
    case SUB_OP:
    case SUBC_OP:
      // ADD,SUB latency
      switch (get_type()) {
        case F32_TYPE:
          latency = fp_latency[0];
          initiation_interval = fp_init[0];
          op = SP_OP;
          break;
        case F64_TYPE:
        case FF64_TYPE:
          latency = dp_latency[0];
          initiation_interval = dp_init[0];
          op = DP_OP;
          break;
        case B32_TYPE:
        case U32_TYPE:
        case S32_TYPE:
        default:  // Use int settings for default
          latency = int_latency[0];
          initiation_interval = int_init[0];
          op = INTP_OP;
          break;
      }
      break;
    case MAX_OP:
    case MIN_OP:
      // MAX,MIN latency
      switch (get_type()) {
        case F32_TYPE:
          latency = fp_latency[1];
          initiation_interval = fp_init[1];
          op = SP_OP;
          break;
        case F64_TYPE:
        case FF64_TYPE:
          latency = dp_latency[1];
          initiation_interval = dp_init[1];
          op = DP_OP;
          break;
        case B32_TYPE:
        case U32_TYPE:
        case S32_TYPE:
        default:  // Use int settings for default
          latency = int_latency[1];
          initiation_interval = int_init[1];
          op = INTP_OP;
          break;
      }
      break;
    case MUL_OP:
      // MUL latency
      switch (get_type()) {
        case F32_TYPE:
          latency = fp_latency[2];
          initiation_interval = fp_init[2];
          op = SP_OP;
          break;
        case F64_TYPE:
        case FF64_TYPE:
          latency = dp_latency[2];
          initiation_interval = dp_init[2];
          op = DP_OP;
          break;
        case B32_TYPE:
        case U32_TYPE:
        case S32_TYPE:
        default:  // Use int settings for default
          latency = int_latency[2];
          initiation_interval = int_init[2];
          op = INTP_OP;
          break;
      }
      break;
    case MAD_OP:
    case MADC_OP:
    case MADP_OP:
    case FMA_OP:
      // MAD latency
      switch (get_type()) {
        case F32_TYPE:
          latency = fp_latency[3];
          initiation_interval = fp_init[3];
          op = SP_OP;
          break;
        case F64_TYPE:
        case FF64_TYPE:
          latency = dp_latency[3];
          initiation_interval = dp_init[3];
          op = DP_OP;
          break;
        case B32_TYPE:
        case U32_TYPE:
        case S32_TYPE:
        default:  // Use int settings for default
          latency = int_latency[3];
          initiation_interval = int_init[3];
          op = INTP_OP;
          break;
      }
      break;
    case MUL24_OP:  // MUL24 is performed on mul32 units (with additional
                    // instructions for bitmasking) on devices with compute
                    // capability >1.x
      latency = int_latency[2] + 1;
      initiation_interval = int_init[2] + 1;
      op = INTP_OP;
      break;
    case MAD24_OP:
      latency = int_latency[3] + 1;
      initiation_interval = int_init[3] + 1;
      op = INTP_OP;
      break;
    case DIV_OP:
    case REM_OP:
      // Floating point only
      op = SFU_OP;
      switch (get_type()) {
        case F32_TYPE:
          latency = fp_latency[4];
          initiation_interval = fp_init[4];
          break;
        case F64_TYPE:
        case FF64_TYPE:
          latency = dp_latency[4];
          initiation_interval = dp_init[4];
          break;
        case B32_TYPE:
        case U32_TYPE:
        case S32_TYPE:
        default:  // Use int settings for default
          latency = int_latency[4];
          initiation_interval = int_init[4];
          break;
      }
      break;
    case SQRT_OP:
    case SIN_OP:
    case COS_OP:
    case EX2_OP:
    case LG2_OP:
    case RSQRT_OP:
    case RCP_OP:
      latency = sfu_latency;
      initiation_interval = sfu_init;
      op = SFU_OP;
      break;
    case MMA_OP:
      latency = tensor_latency;
      initiation_interval = tensor_init;
      op = TENSOR_CORE_OP;
      break;
    case SHFL_OP:
      latency = int_latency[5];
      initiation_interval = int_init[5];
      break;
    default:
      break;
  }
  set_fp_or_int_archop();
  set_mul_div_or_other_archop();
}

/*
解码阶段，定时模拟器从给定PC的函数模拟器获得指令。这是通过调用ptx_fetch_inst函数完成的。
*/
void ptx_thread_info::ptx_fetch_inst(inst_t &inst) const {
  //get_pc()获取当前线程的PC值。
  addr_t pc = get_pc();
  //依据PC值从指令存储m_instr_mem中获取PC值对应的指令。
  const ptx_instruction *pI = m_func_info->get_instruction(pc);
  inst = (const inst_t &)*pI;
  assert(inst.valid());
}

/*
在PTX中，基本类型反映了目标架构支持的本地数据类型。基本类型指定基本类型和大小。寄存器变量总是基本类型的，
指令对这些类型进行操作。变量定义和类型指令使用相同的类型大小说明符，因此有意将它们的名称缩短。大多数指令
都有一个或多个类型说明符，需要这些说明符来完全指定指令行为。根据指令类型检查操作数类型和大小的兼容性。如
果两个基本类型具有相同的基本类型和相同的大小，则它们是兼容的。如果有符号和无符号整数类型具有相同的大小，
则它们是兼容的。位大小类型与具有相同大小的任何基本类型兼容。

PTX指令集手册里定义的基本类型说明符：
Basic Type         |  Fundamental Type Specifiers
--------------------------------------------------
Signed integer     |  .s8, .s16, .s32, .s64
Unsigned integer   |  .u8, .u16, .u32, .u64
Floating-point     |  .f16, .f16x2, .f32, .f64
Bits (untyped)     |  .b8, .b16, .b32, .b64
Predicate          |  .pred
*/
static unsigned datatype2size(unsigned data_type) {
  unsigned data_size;
  switch (data_type) {
    case B8_TYPE: //.b8: 8 bits = 1 byte
    case S8_TYPE: //.s8: 8 bits = 1 byte
    case U8_TYPE: //.u8: 8 bits = 1 byte
      data_size = 1;
      break;
    case B16_TYPE: //.b16: 16 bits = 2 byte
    case S16_TYPE: //.s16: 16 bits = 2 byte
    case U16_TYPE: //.u16: 16 bits = 2 byte
    case F16_TYPE: //.f16: 16 bits = 2 byte
      data_size = 2;
      break;
    case B32_TYPE: //.b32: 32 bits = 4 byte
    case S32_TYPE: //.s32: 32 bits = 4 byte
    case U32_TYPE: //.u32: 32 bits = 4 byte
    case F32_TYPE: //.f32: 32 bits = 4 byte
      data_size = 4;
      break;
    case B64_TYPE:  //.b64: 64 bits = 8 byte
    case BB64_TYPE: //.???: 64 bits = 8 byte
    case S64_TYPE:  //.s64: 64 bits = 8 byte
    case U64_TYPE:  //.u64: 64 bits = 8 byte
    case F64_TYPE:  //.f64: 64 bits = 8 byte
    case FF64_TYPE: //.???: 64 bits = 8 byte
      data_size = 8;
      break;
    case BB128_TYPE: //.???: 128 bits = 16 byte
      data_size = 16;
      break;
    default:
      assert(0);
      break;
  }
  return data_size;
}

/*
PTX指令预解码函数，它的功能是将PTX指令的二进制码解码为可读的字符串，以便进行后续的指令处理和执行。
预解码是通过调用ptx_instruction::pre_decode()对每条指令进行的。它提取了对定时模拟器有用的信息：
1. 检测LD/ST指令。
2. 确定该指令是否写到目标寄存器。
3. 如果这是一条分支指令，获得重合PC。
4. 提取该指令的寄存器操作数。
5. 检测预设指令。
提取的信息存储在与指令对应的 ptx_instruction 对象中。这加快了仿真的速度，因为在一个内核启动中的所有标量
线程都执行相同的内核函数。在函数加载时提取一次这些信息，比在仿真过程中对每个线程重复提取要有效得多。
*/
void ptx_instruction::pre_decode() {
  //PC值以获取当前被处理的指令。
  pc = m_PC;
  //m_inst_size为指令的大小，以byte为单位。
  isize = m_inst_size;
  //MAX_OUTPUT_VALUES和MAX_INPUT_VALUES在abstract_hardware_model.h中定义。out[i]用于保存输出（目标）
  //操作数，in[i]用于保存输入（源）操作数。MAX_OUTPUT_VALUES和MAX_INPUT_VALUES用于指定支持PTX指令中最
  //大的输出操作数数量和输入操作数数量。
  for (unsigned i = 0; i < MAX_OUTPUT_VALUES; i++) {
    out[i] = 0;
  }
  for (unsigned i = 0; i < MAX_INPUT_VALUES; i++) {
    in[i] = 0;
  }
  //outcount是PTX指令中的输出操作数数量，incount是输入操作数数量。
  incount = 0;
  outcount = 0;
  //操作数是向量作为输入。
  is_vectorin = 0;
  //操作数是向量作为输出。
  is_vectorout = 0;
  //MAX_REG_OPERANDS由abstract_hardware_model.h定义，为指令中destination, source, 或address uarch
  //操作数的最大数量。arch_reg在abstract_hardware_model.h中定义为：
  //  struct {
  //        int dst[MAX_REG_OPERANDS];
  //        int src[MAX_REG_OPERANDS];
  //  } arch_reg;
  //是用于保存源操作数和目的操作数的寄存器号。
  std::fill_n(arch_reg.src, MAX_REG_OPERANDS, -1);
  std::fill_n(arch_reg.dst, MAX_REG_OPERANDS, -1);
  //谓词寄存器号。谓词寄存器的介绍和使用见ptx_thread_info::ptx_exec_inst函数。
  pred = 0;
  //指令寄存器，用于记分牌检查：
  //    ar1：指令寄存器1，用于存储指令的第一个参数。
  //    ar2：指令寄存器2，用于存储指令的第二个参数。
  ar1 = 0;
  ar2 = 0;
  space = m_space_spec;
  //对存储的操作，在abstract_hardware_model.h定义：
  //    enum _memory_op_t { no_memory_op = 0, memory_load, memory_store };
  //非对存储操作为no_memory_op，存储load为memory_load，存储store为memory_store。
  memory_op = no_memory_op;
  //对存储访存指令所使用的数据大小，例如ld.f64为64 bit读操作指令。
  data_size = 0;
  if (has_memory_read() || has_memory_write()) {
    //get_type()获取存储访问的类型，包括以下多种可能：
    //    B8_TYPE，S8_TYPE，U8_TYPE，B16_TYPE，S16_TYPE，U16_TYPE，F16_TYPE，B32_TYPE，S32_TYPE，
    //    U32_TYPE，F32_TYPE，B64_TYPE，BB64_TYPE，S64_TYPE，U64_TYPE，F64_TYPE，FF64_TYPE，
    //    BB128_TYPE。
    //详细地内容在datatype2size()函数定义里有注释。
    unsigned to_type = get_type();
    //对存储访存指令所使用的数据大小，例如ld.f64为64 bit读操作指令。
    data_size = datatype2size(to_type);
    //对存储的操作初始时，设为0=no_memory_op，现在重新设置。
    memory_op = has_memory_read() ? memory_load : memory_store;
  }

  //has_dst为True指该指令有目标操作数。
  bool has_dst = false;
  //检测该指令是否有目标操作数，设置has_dst的值。
  switch (get_opcode()) {
#define OP_DEF(OP, FUNC, STR, DST, CLASSIFICATION) \
  case OP:                                         \
    has_dst = (DST != 0);                          \
    break;
#define OP_W_DEF(OP, FUNC, STR, DST, CLASSIFICATION) \
  case OP:                                           \
    has_dst = (DST != 0);                            \
    break;
#include "opcodes.def"
#undef OP_DEF
#undef OP_W_DEF
    default:
      printf("Execution error: Invalid opcode (0x%x)\n", get_opcode());
      break;
  }
  
  //设置缓存一致性策略：
  //PTX ISA 2.0版在加载和存储指令上引入了可选的缓存运算符。缓存运算符需要sm_20或更高的目标体系结构。加
  //载或存储指令上的缓存运算符仅被视为性能提示。对ld或st指令使用缓存运算符不会改变程序的内存一致性行为。
  //对于sm_20及更高版本，缓存运算符具有以下定义和行为：
  //1. 内存Load指令的缓存运算符：
  //  .ca: Cache at all levels，所有级别的缓存，可能会再次访问。
  //       默认的Loaf指令缓存操作是ld.ca，它使用正常的逐出策略在所有级别（L1和L2）中分配缓存线。全局数
  //       据在L2级是一致的，但多个L1缓存对于全局数据来说是不一致的。如果一个线程通过一个L1缓存存储到全
  //       局内存，而第二个线程通过第二个L1缓存加载该地址，并使用ld.ca，则第二个可能会得到过时的L1缓存
  //       数据，而不是第一个线程存储的数据。驱动程序必须使并行线程的从属网格之间的全局L1缓存线无效。然
  //       后，第一个网格程序的存储由第二个网格程序正确获取，该程序发出L1中缓存的默认ld.ca加载。
  //  .cg  Cache at global level，全局级缓存（L2及以下缓存，而不是L1）。
  //       使用ld.cg全局性地加载，绕过L1缓存，并仅缓存在L2缓存中。
  //  .cs  Cache streaming，缓存流，可能会被访问一次。
  //       ld.cs加载缓存的流操作在L1和L2分配全局行，采用驱逐优先的策略在L1和L2分配全局行，以限制临时流
  //       数据对缓存污染，这些数据可能被访问一次或两次。当ld.cs被应用到一个本地窗口地址时，它执行ld.lu
  //       操作。
  //  .lu  Last use。
  //       编译器/程序员在恢复溢出的寄存器和弹出函数堆栈框架时可以使用ld.lu，以避免不必要的写回不会再使
  //       用的行。ld.lu指令在全局地址上执行一个加载缓存的流操作（ld.cs）。
  //  .cv  不要再次缓存和获取（考虑缓存的系统内存线已过时，再次获取）。应用于全局系统内存地址的ld.cv加载
  //       操作使匹配的L2行无效（丢弃），并在每次新加载时重新获取该行。
  //2. 内存Store指令的缓存运算符：
  //  .wb  Cache write-back all coherent levels，缓存写回所有级别一致。
  //       默认的存储指令缓存操作是st.wb，它使用正常的逐出策略写回一致缓存级别的缓存行。如果一个线程绕过
  //       其L1缓存存储到全局内存，而另一个SM中的第二个线程稍后通过具有ld.ca的不同L1缓存从该地址加载，则
  //       第二个可能会命中过时的L1缓存数据，而不是从第一个线程存储的L2或内存中获取数据。驱动程序必须使线
  //       程阵列的从属网格之间的全局L1缓存线无效。然后，第一个网格程序的存储在L1中正确丢失，并由发出默认
  //       ld.ca加载的第二个网格程序获取。
  //  .wt  Cache write-through (to system memory).
  //       st.wt存储写入操作应用于通过二级缓存写入的全局系统内存地址。
  switch (m_cache_option) {
    case CA_OPTION:
      cache_op = CACHE_ALL;
      break;
    case NC_OPTION:
      cache_op = CACHE_L1;
      break;
    case CG_OPTION:
      cache_op = CACHE_GLOBAL;
      break;
    case CS_OPTION:
      cache_op = CACHE_STREAMING;
      break;
    case LU_OPTION:
      cache_op = CACHE_LAST_USE;
      break;
    case CV_OPTION:
      cache_op = CACHE_VOLATILE;
      break;
    case WB_OPTION:
      cache_op = CACHE_WRITE_BACK;
      break;
    case WT_OPTION:
      cache_op = CACHE_WRITE_THROUGH;
      break;
    default:
      // if( m_opcode == LD_OP || m_opcode == LDU_OP )
      //对于MMA_LD_OP/LD_OP/LDU_OP指令，默认选择 CACHE_ALL=.ca 策略。
      if (m_opcode == MMA_LD_OP || m_opcode == LD_OP || m_opcode == LDU_OP)
        cache_op = CACHE_ALL;
      // else if( m_opcode == ST_OP )
      //对于MMA_ST_OP/LST_OP指令，默认选择 CACHE_WRITE_BACK=.wb 策略。
      else if (m_opcode == MMA_ST_OP || m_opcode == ST_OP)
        cache_op = CACHE_WRITE_BACK;
      //对于ATOM_OP指令，默认选择 CACHE_GLOBAL=.cg 策略。
      else if (m_opcode == ATOM_OP)
        cache_op = CACHE_GLOBAL;
      break;
  }
  //设置每种类型指令的操作码和延时。
  set_opcode_and_latency();
  //设置每种并行同步指令的屏障类型。
  set_bar_type();
  // Get register operands
  //获取寄存器操作数。该段代码是GPGPU-Sim模拟器中指令操作数处理函数，主要用于将指令操作数存储在out和in数
  //组中，并将指令操作数的架构寄存器存储在arch_reg.dst和arch_reg.src数组中。其中，n表示操作数的索引，m
  //表示arch register操作数的索引，has_dst表示指令是否有目标操作数，op_iter_begin()和op_iter_end()表
  //示指令操作数的起始和结束位置。
  int n = 0, m = 0;
  //const_iterator在ptx_ir.h中定义：
  //  typedef std::vector<operand_info>::const_iterator const_iterator
  //是一个由operand_info，即操作数信息构成的向量。
  ptx_instruction::const_iterator opr = op_iter_begin();
  //for循环对所有的const_iterator中的操作数循环，处理每个操作数。
  for (; opr != op_iter_end(); opr++, n++) {  // process operands
    //o即获取其中一个操作数，o为存储该操作数信息对象的地址。
    const operand_info &o = *opr;
    //该if判断即为，当该指令具有目标操作数，且n=0时，即n编号所指向的就是目标操作数。
    if (has_dst && n == 0) {
      // Do not set the null register "_" as an architectural register
      if (o.is_reg() && !o.is_non_arch_reg()) {
        //如果o指向的操作数为寄存器，且该寄存器不是为了PTX指令方便使用而临时占位的"_"。
        //out[0]保存该输出（目标）操作数。
        out[0] = o.reg_num();
        arch_reg.dst[0] = o.arch_reg_num();
      } else if (o.is_vector()) {
        //如果o指向的操作数为向量。
        is_vectorin = 1; //操作数是向量作为输入。
        //o.get_vect_nelem()返回o所指向的向量操作数的向量长度。
        unsigned num_elem = o.get_vect_nelem();
        //根据向量操作数的向量长度，将每个向量的元素加入到out[i]中。
        if (num_elem >= 1) out[0] = o.reg1_num();
        if (num_elem >= 2) out[1] = o.reg2_num();
        if (num_elem >= 3) out[2] = o.reg3_num();
        if (num_elem >= 4) out[3] = o.reg4_num();
        if (num_elem >= 5) out[4] = o.reg5_num();
        if (num_elem >= 6) out[5] = o.reg6_num();
        if (num_elem >= 7) out[6] = o.reg7_num();
        if (num_elem >= 8) out[7] = o.reg8_num();
        for (int i = 0; i < num_elem; i++) arch_reg.dst[i] = o.arch_reg_num(i);
      }
    //该else判断即为，如果该指令不具有目标操作数，则其所有操作数均为源操作数。
    } else {
      //如果o指向的操作数为寄存器，且该寄存器不是为了PTX指令方便使用而临时占位的"_"。
      if (o.is_reg() && !o.is_non_arch_reg()) {
        //由于o在循环里往后迭代，先将寄存器号存储到reg_num，后面根据m索引值加入到in[i]中。
        int reg_num = o.reg_num();
        arch_reg.src[m] = o.arch_reg_num();
        switch (m) {
          case 0:
            in[0] = reg_num;
            break;
          case 1:
            in[1] = reg_num;
            break;
          case 2:
            in[2] = reg_num;
            break;
          default:
            break;
        }
        m++;
      } else if (o.is_vector()) {
        // assert(m == 0); //only support 1 vector operand (for textures) right
        // now
        //如果o指向的操作数为向量，与目标操作数为向量时类似。
        is_vectorout = 1;
        unsigned num_elem = o.get_vect_nelem();
        if (num_elem >= 1) in[m + 0] = o.reg1_num();
        if (num_elem >= 2) in[m + 1] = o.reg2_num();
        if (num_elem >= 3) in[m + 2] = o.reg3_num();
        if (num_elem >= 4) in[m + 3] = o.reg4_num();
        if (num_elem >= 5) in[m + 4] = o.reg5_num();
        if (num_elem >= 6) in[m + 5] = o.reg6_num();
        if (num_elem >= 7) in[m + 6] = o.reg7_num();
        if (num_elem >= 8) in[m + 7] = o.reg8_num();
        for (int i = 0; i < num_elem; i++)
          arch_reg.src[m + i] = o.arch_reg_num(i);
        m += num_elem;
      }
    }
  }

  // Setting number of input and output operands which is required for
  // scoreboard check
  //设置记分牌检查所需的输入和输出操作数。outcount是输出操作数数量，incount是输入操作数数量。
  //从这里看，out[i]用于保存输出操作数，in[i]用于保存输入操作数。
  for (int i = 0; i < MAX_OUTPUT_VALUES; i++)
    if (out[i] > 0) outcount++;

  for (int i = 0; i < MAX_INPUT_VALUES; i++)
    if (in[i] > 0) incount++;

  // Get predicate
  //获取谓词寄存器编号。谓词寄存器的介绍和使用见ptx_thread_info::ptx_exec_inst函数。
  if (has_pred()) {
    const operand_info &p = get_pred();
    pred = p.reg_num();
  }

  // Get address registers inside memory operands.
  // Assuming only one memory operand per instruction,
  //  and maximum of two address registers for one memory operand.
  //获取内存操作数内的地址寄存器。假设每条指令只有一个内存操作数，一个内存操作数最多有两个地址寄存器。
  //当指令为读存储指令或者写存储指令，获取内存操作数内的地址寄存器。
  if (has_memory_read() || has_memory_write()) {
    ptx_instruction::const_iterator op = op_iter_begin();
    for (; op != op_iter_end(); op++, n++) {  // process operands
      const operand_info &o = *op;
      //判断o指向的操作数是否是内存操作数。
      if (o.is_memory_operand()) {
        // We do not support the null register as a memory operand
        //GPGPU-Sim暂时不支持将空寄存器（NULL Register）作为内存操作数。
        assert(!o.is_non_arch_reg());

        // Check PTXPlus-type operand
        // memory operand with addressing (ex. s[0x4] or g[$r1])
        if (o.is_memory_operand2()) {
          // memory operand with one address register (ex. g[$r1+0x4] or
          // s[$r2+=0x4])
          if (o.get_double_operand_type() == 0 ||
              o.get_double_operand_type() == 3) {
            ar1 = o.reg_num();
            arch_reg.src[4] = o.arch_reg_num();
            // TODO: address register in $r2+=0x4 should be an output register
            // as well
          }
          // memory operand with two address register (ex. s[$r1+$r1] or
          // g[$r1+=$r2])
          else if (o.get_double_operand_type() == 1 ||
                   o.get_double_operand_type() == 2) {
            ar1 = o.reg1_num();
            arch_reg.src[4] = o.arch_reg_num();
            ar2 = o.reg2_num();
            arch_reg.src[5] = o.arch_reg_num();
            // TODO: first address register in $r1+=$r2 should be an output
            // register as well
          }
        } else if (o.is_immediate_address()) {
        }
        // Regular PTX operand
        else if (o.get_symbol()
                     ->type()
                     ->get_key()
                     .is_reg()) {  // Memory operand contains a register
          ar1 = o.reg_num();
          arch_reg.src[4] = o.arch_reg_num();
        }
      }
    }
  }

  // get reconvergence pc
  //获取分支重聚点PC值。
  reconvergence_pc = gpgpu_ctx->func_sim->get_converge_point(pc);
  //ptx_instruction::pre_decode()函数执行完毕，设置解码状态变量m_decoded = true。
  m_decoded = true;
}

/*
该函数用于向 m_ptx_kernel_param_info 中添加函数名，类型，大小。
对于内核入口，将每个内核参数存储在一个映射 m_ptx_kernel_param_info 中；它在ptx_ir.h中定义：
    std::map<unsigned, param_info> m_ptx_kernel_param_info;
但是，对于OpenCL应用程序来说，可能并不总是这样的。在OpenCL中，相关的常量内存空间可以通过两种方式分配。它
可以在声明它的.ptx文件中显式初始化，或者使用主机上的 clCreateBuffer 来分配它。在后面这种情况下，.ptx文件
将包含一个参数的全局声明，但它将有一个未知的数组大小。因此，该符号的地址将不会被设置，需要在执行PTX之前在：
    function_info::add_param_data(...)
函数中设置。在这种情况下，内核参数的地址被存储在 function_info 对象中的一个符号表中。
*/
void function_info::add_param_name_type_size(unsigned index, std::string name,
                                             int type, size_t size, bool ptr,
                                             memory_space_t space) {
  unsigned parsed_index;
  char buffer[2048];
  //snprintf用于格式化字符串，其原型为：
  //    snprintf(char *str, size_t size, const char *format, ...)
  //用于将format中的size-1个字符复制到str中，但是要注意：
  // (1) 如果格式化后的字符串长度 < size，则将此字符串全部复制到str中，并给其后添加一个字符串结束符('\0')，
  //     该字符串结束符也占用 size 的一个位置；
  // (2) 如果格式化后的字符串长度 >= size，则只将其中的(size-1)个字符复制到str中，并给其后添加一个字符串
  //     结束符('\0')，返回值为欲写入的字符串长度。
  //下面是将 m_name.c_str() + "_param_%%u" 字符串写入 buffer。
  snprintf(buffer, 2048, "%s_param_%%u", m_name.c_str());
  //sscanf用于解析字符串，例如下述函数：
  //    int year, month, day;
  //    int converted = sscanf("20191103", "%04d%02d%02d", &year, &month, &day);
  //    printf("converted=%d, year=%d, month=%d, day=%d/n", converted, year, month, day);
  //将"20191103"字符串按照"%04d%02d%02d"的分割格式解析出并赋值给三个变量year/month/day。返回值为解析成
  //功的变量个数。

  //parsed_index其实存储的是该参数的序号，ntoken返回值可以为0或1。
  int ntokens = sscanf(name.c_str(), buffer, &parsed_index);
  //例如，下面的注释掉的语句：
  //  printf("name.c_str()=%s, buffer=%s, parsed_index=%d\n", name.c_str(), buffer, parsed_index);
  //会输出：
  //  name.c_str()=_Z6MatMulPiS_S_i_param_0, buffer=_Z6MatMulPiS_S_i_param_%u, parsed_index=0
  //  name.c_str()=_Z6MatMulPiS_S_i_param_1, buffer=_Z6MatMulPiS_S_i_param_%u, parsed_index=1
  //  name.c_str()=_Z6MatMulPiS_S_i_param_2, buffer=_Z6MatMulPiS_S_i_param_%u, parsed_index=2
  //  name.c_str()=_Z6MatMulPiS_S_i_param_3, buffer=_Z6MatMulPiS_S_i_param_%u, parsed_index=3
  if (ntokens == 1) {
    //parsed_index序号标明该参数是第几个，将该参数信息=param_info对象加入到m_ptx_kernel_param_info中。
    assert(m_ptx_kernel_param_info.find(parsed_index) ==
           m_ptx_kernel_param_info.end());
    m_ptx_kernel_param_info[parsed_index] =
        param_info(name, type, size, ptr, space);
  } else {
    assert(m_ptx_kernel_param_info.find(index) ==
           m_ptx_kernel_param_info.end());
    m_ptx_kernel_param_info[index] = param_info(name, type, size, ptr, space);
  }
}

/*
对于CUDA和OpenCL，GPGPU-Sim为每个内核参数创建一个 gpgpu_ptx_sim_arg 对象，并维护一个所有内核参数的列表。
在执行内核之前，会调用一个初始化函数来设置GPU网格的尺寸和参数：
    gpgpu_cuda_ptx_sim_init_grid(...) 或者 gpgpu_opencl_ptx_sim_init_grid(...) 。
在这些函数中使用了两个主要的对象， function_info 和 kernel_info_t 对象。在init_grid函数中，内核参数通过
调用：function_info_object::add_param_data(argn, gpgpu_ptx_sim_arg *) 被添加到 function_info 对象中。
*/
void function_info::add_param_data(unsigned argn,
                                   struct gpgpu_ptx_sim_arg *args) {
  const void *data = args->m_start;

  //scratchpad_memory_param=True说明参数在OpenCL local memory中；
  //scratchpad_memory_param=False说明参数在CUDA shared memory中。
  bool scratchpad_memory_param =
      false;  // Is this parameter in CUDA shared memory or OpenCL local memory

  //对于内核入口，将每个内核参数存储在一个映射 m_ptx_kernel_param_info 中；它在ptx_ir.h中定义：
  //  std::map<unsigned, param_info> m_ptx_kernel_param_info;
  //下面是找到m_ptx_kernel_param_info中的第argn个参数信息。
  std::map<unsigned, param_info>::iterator i =
      m_ptx_kernel_param_info.find(argn);
  if (i != m_ptx_kernel_param_info.end()) {
    if (i->second.is_ptr_shared()) {
      //为OpenCL涉及，后面用到再补充。
      assert(
          args->m_start == NULL &&
          "OpenCL parameter pointer to local memory must have NULL as value");
      scratchpad_memory_param = true;
    } else {
      //为CUDA涉及。
      //param_t在ptx_sim.h中定义：
      //  struct param_t {
      //    const void *pdata;
      //    int type;
      //    size_t size;
      //    size_t offset;
      //  };
      param_t tmp;
      tmp.pdata = args->m_start;
      tmp.size = args->m_nbytes;
      tmp.offset = args->m_offset;
      tmp.type = 0;
      i->second.add_data(tmp);
      i->second.add_offset((unsigned)args->m_offset);
    }
  } else {
    scratchpad_memory_param = true;
  }

  //仅为OpenCL使用，后面用到再补充。
  if (scratchpad_memory_param) {
    // This should only happen for OpenCL:
    //
    // The LLVM PTX compiler in NVIDIA's driver (version 190.29)
    // does not generate an argument in the function declaration
    // for __constant arguments.
    //
    // The associated constant memory space can be allocated in two
    // ways. It can be explicitly initialized in the .ptx file where
    // it is declared.  Or, it can be allocated using the clCreateBuffer
    // on the host. In this later case, the .ptx file will contain
    // a global declaration of the parameter, but it will have an unknown
    // array size.  Thus, the symbol's address will not be set and we need
    // to set it here before executing the PTX.

    char buffer[2048];
    snprintf(buffer, 2048, "%s_param_%u", m_name.c_str(), argn);

    symbol *p = m_symtab->lookup(buffer);
    if (p == NULL) {
      printf(
          "GPGPU-Sim PTX: ERROR ** could not locate symbol for \'%s\' : cannot "
          "bind buffer\n",
          buffer);
      abort();
    }
    if (data)
      p->set_address((addr_t) * (size_t *)data);
    else {
      // clSetKernelArg was passed NULL pointer for data...
      // this is used for dynamically sized shared memory on NVIDIA platforms
      bool is_ptr_shared = false;
      if (i != m_ptx_kernel_param_info.end()) {
        is_ptr_shared = i->second.is_ptr_shared();
      }

      if (!is_ptr_shared and !p->is_shared()) {
        printf(
            "GPGPU-Sim PTX: ERROR ** clSetKernelArg passed NULL but arg not "
            "shared memory\n");
        abort();
      }
      unsigned num_bits = 8 * args->m_nbytes;
      printf(
          "GPGPU-Sim PTX: deferred allocation of shared region for \"%s\" from "
          "0x%llx to 0x%llx (shared memory space)\n",
          p->name().c_str(), m_symtab->get_shared_next(),
          m_symtab->get_shared_next() + num_bits / 8);
      fflush(stdout);
      assert((num_bits % 8) == 0);
      addr_t addr = m_symtab->get_shared_next();
      addr_t addr_pad =
          num_bits
              ? (((num_bits / 8) - (addr % (num_bits / 8))) % (num_bits / 8))
              : 0;
      p->set_address(addr + addr_pad);
      m_symtab->alloc_shared(num_bits / 8 + addr_pad);
    }
  }
}

/*
将参数逐个对齐。
*/
unsigned function_info::get_args_aligned_size() {
  //m_args_aligned_size设备端内核函数的参数大小。
  if (m_args_aligned_size >= 0) return m_args_aligned_size;
  //初始时，设置param_address为0，没处理一个参数，就将param_address转到下个参数应放置的地址。
  unsigned param_address = 0;
  //total_size代表在当前处理的指令之前的，所有已处理好（对齐好）的参数一共占用了多少个字节。
  unsigned int total_size = 0;
  for (std::map<unsigned, param_info>::iterator i =
           m_ptx_kernel_param_info.begin();
       i != m_ptx_kernel_param_info.end(); i++) {
    //对m_ptx_kernel_param_info中的所有参数都遍历一遍。
    //获取i指向的参数信息，param_info对象p，p包含了该参数的信息。
    param_info &p = i->second;
    //参数名。
    std::string name = p.get_name();
    //symbol是指令的标签。???
    symbol *param = m_symtab->lookup(name.c_str());
    //p.get_size()获取该参数的位数，再计算成字节数。
    size_t arg_size = p.get_size() / 8;  // size of param in bytes
    total_size = (total_size + arg_size - 1) / arg_size * arg_size;  // aligned
    p.add_offset(total_size);
    param->set_address(param_address + total_size);
    total_size += arg_size;
  }

  m_args_aligned_size = (total_size + 3) / 4 * 4;  // final size aligned to word

  return m_args_aligned_size;
}

/*
在向 function_info 对象添加所有的参数后，调用 function_info::finalize(...) ，它将存储在 
    function_info::m_ptx_kernel_param_info 
映射中的内核参数复制到上面提到的 kernel_info_t 对象中分配的参数内存。如果之前在 
    function_info::add_param_data(...)
中没有做，每个内核参数的地址会被添加到 function_info 对象的符号表中。
*/
void function_info::finalize(memory_space *param_mem) {
  unsigned param_address = 0;
  for (std::map<unsigned, param_info>::iterator i =
           m_ptx_kernel_param_info.begin();
       i != m_ptx_kernel_param_info.end(); i++) {
    param_info &p = i->second;
    if (p.is_ptr_shared())
      continue;  // Pointer to local memory: Should we pass the allocated shared
                 // memory address to the param memory space?
    //param_t在ptx_sim.h中定义：
    //  struct param_t {
    //    const void *pdata;
    //    int type;
    //    size_t size;
    //    size_t offset;
    //  };
    std::string name = p.get_name();     //name
    int type = p.get_type();             //type
    param_t param_value = p.get_value(); //value
    param_value.type = type;
    symbol *param = m_symtab->lookup(name.c_str());
    unsigned xtype = param->type()->get_key().scalar_type();
    assert(xtype == (unsigned)type);
    size_t size;
    size = param_value.size;  // size of param in bytes
    // assert(param_value.offset == param_address);
    if (size != p.get_size() / 8) {
      printf(
          "GPGPU-Sim PTX: WARNING actual kernel paramter size = %zu bytes vs. "
          "formal size = %zu (using smaller of two)\n",
          size, p.get_size() / 8);
      size = (size < (p.get_size() / 8)) ? size : (p.get_size() / 8);
    }
    // copy the parameter over word-by-word so that parameter that crosses a
    // memory page can be copied over
    // Jin: copy parameter using aligned rules
    //逐字复制参数，以便可以复制跨越内存页的参数。
    //使用对齐规则复制参数。
    const type_info *paramtype = param->type();
    int align_amount = paramtype->get_key().get_alignment_spec();
    align_amount = (align_amount == -1) ? size : align_amount;
    param_address = (param_address + align_amount - 1) / align_amount *
                    align_amount;  // aligned

    const size_t word_size = 4;
    // param_address = (param_address + size - 1) / size * size; //aligned with
    // size
    for (size_t idx = 0; idx < size; idx += word_size) {
      const char *pdata = reinterpret_cast<const char *>(param_value.pdata) +
                          idx;  // cast to char * for ptr arithmetic
      param_mem->write(param_address + idx, word_size, pdata, NULL, NULL);
    }
    unsigned offset = p.get_offset();
    assert(offset == param_address);
    param->set_address(param_address);
    param_address += size;
  }
}

/*
支持PTXPlus需要将内核参数复制到共享内存中。内核参数可以通过调用：
    function_info::param_to_shared(shared_mem, symbol_table)
函数从 Param 内存拷贝到 Shared 内存。这个函数遍历存储在 function_info::m_ptx_kernel_param_info 映射
中的内核参数，并将每个参数从 Param 内存拷贝到 shared_mem 指向的 Shared 内存中的适当位置。
*/
void function_info::param_to_shared(memory_space *shared_mem,
                                    symbol_table *symtab) {
  // TODO: call this only for PTXPlus with GT200 models
  // extern gpgpu_sim* g_the_gpu;
  if (not gpgpu_ctx->the_gpgpusim->g_the_gpu->get_config().convert_to_ptxplus())
    return;

  // copies parameters into simulated shared memory
  for (std::map<unsigned, param_info>::iterator i =
           m_ptx_kernel_param_info.begin();
       i != m_ptx_kernel_param_info.end(); i++) {
    param_info &p = i->second;
    if (p.is_ptr_shared())
      continue;  // Pointer to local memory: Should we pass the allocated shared
                 // memory address to the param memory space?
    std::string name = p.get_name();
    int type = p.get_type();
    param_t value = p.get_value();
    value.type = type;
    symbol *param = symtab->lookup(name.c_str());
    unsigned xtype = param->type()->get_key().scalar_type();
    assert(xtype == (unsigned)type);

    int tmp;
    size_t size;
    unsigned offset = p.get_offset();
    type_info_key::type_decode(xtype, size, tmp);

    // Write to shared memory - offset + 0x10
    shared_mem->write(offset + 0x10, size / 8, value.pdata, NULL, NULL);
  }
}

/*
在fout文件中列出所有参数的信息。
*/
void function_info::list_param(FILE *fout) const {
  //遍历param_info对象中的所有参数。
  for (std::map<unsigned, param_info>::const_iterator i =
           m_ptx_kernel_param_info.begin();
       i != m_ptx_kernel_param_info.end(); i++) {
    const param_info &p = i->second;
    //参数名。
    std::string name = p.get_name();
    //获取参数的存放地址。
    symbol *param = m_symtab->lookup(name.c_str());
    addr_t param_addr = param->get_address();
    fprintf(fout, "%s: %#08llx\n", name.c_str(), param_addr);
  }
  fflush(fout);
}

/*
实时（JIT）编译器配置。
GPGPU-Sim模拟NVIDIA使用的并行线程执行（Parallel Thread eXecution，PTX）指令集。PTX是伪汇编指令集；它
不直接在硬件上执行。ptxas是NVIDIA发布的汇编程序，用于将PTX汇编到硬件运行的本机指令集（SASS）中。每一代
硬件都支持不同版本的SASS。因此，PTX被编译成多个版本的SASS，对应于编译时不同的硬件版本。尽管如此，PTX代码
仍然嵌入到二进制文件中，以支持未来的硬件。在运行时，运行时系统根据可用硬件选择要运行的SASS的适当版本。如
果没有，则运行时系统调用嵌入式PTX上的实时（JIT）编译器，将其编译为与可用硬件相对应的SASS。
*/
void function_info::ptx_jit_config(
    std::map<unsigned long long, size_t> mallocPtr_Size,
    memory_space *param_mem, gpgpu_t *gpu, dim3 gridDim, dim3 blockDim) {
  static unsigned long long counter = 0;
  std::vector<std::pair<size_t, unsigned char *> > param_data;
  std::vector<unsigned> offsets;
  std::vector<bool> paramIsPointer;

  char *gpgpusim_path = getenv("GPGPUSIM_ROOT");
  assert(gpgpusim_path != NULL);
  char *wys_exec_path = getenv("WYS_EXEC_PATH");
  assert(wys_exec_path != NULL);
  std::string command =
      std::string("mkdir ") + gpgpusim_path + "/debug_tools/WatchYourStep/data";
  std::string filename(std::string(gpgpusim_path) +
                       "/debug_tools/WatchYourStep/data/params.config" +
                       std::to_string(counter));

  // initialize paramList
  char buff[1024];
  std::string filename_c(filename + "_c");
  snprintf(buff, 1024, "c++filt %s > %s", get_name().c_str(),
           filename_c.c_str());
  assert(system(buff) != 0);
  FILE *fp = fopen(filename_c.c_str(), "r");
  char *ptr = fgets(buff, 1024, fp);
  if (ptr == NULL) {
    printf("can't read file %s \n", filename_c.c_str());
    assert(0);
  }
  fclose(fp);
  std::string fn(buff);
  size_t pos1, pos2;
  pos1 = fn.find_last_of("(");
  pos2 = fn.find(")", pos1);
  assert(pos2 > pos1 && pos1 > 0);
  strcpy(buff, fn.substr(pos1 + 1, pos2 - pos1 - 1).c_str());
  char *tok;
  tok = strtok(buff, ",");
  std::string tmp;
  while (tok != NULL) {
    std::string param(tok);
    if (param.find("<") != std::string::npos) {
      assert(param.find(">") == std::string::npos);
      assert(param.find("*") == std::string::npos);
      tmp = param;
    } else {
      if (tmp.length() > 0) {
        tmp = "";
        assert(param.find(">") != std::string::npos);
        assert(param.find("<") == std::string::npos);
        assert(param.find("*") == std::string::npos);
      }
      printf("%s\n", param.c_str());
      if (param.find("*") != std::string::npos) {
        paramIsPointer.push_back(true);
      } else {
        paramIsPointer.push_back(false);
      }
    }
    tok = strtok(NULL, ",");
  }

  for (std::map<unsigned, param_info>::iterator i =
           m_ptx_kernel_param_info.begin();
       i != m_ptx_kernel_param_info.end(); i++) {
    param_info &p = i->second;
    std::string name = p.get_name();
    symbol *param = m_symtab->lookup(name.c_str());
    addr_t param_addr = param->get_address();
    param_t param_value = p.get_value();
    offsets.push_back((unsigned)p.get_offset());

    if (paramIsPointer[i->first] &&
        (*(unsigned long long *)param_value.pdata != 0)) {
      // is pointer
      assert(param_value.size == sizeof(void *) &&
             "MisID'd this param as pointer");
      size_t array_size = 0;
      unsigned long long param_pointer =
          *(unsigned long long *)param_value.pdata;
      if (mallocPtr_Size.find(param_pointer) != mallocPtr_Size.end()) {
        array_size = mallocPtr_Size[param_pointer];
      } else {
        for (std::map<unsigned long long, size_t>::iterator j =
                 mallocPtr_Size.begin();
             j != mallocPtr_Size.end(); j++) {
          if (param_pointer > j->first &&
              param_pointer < j->first + j->second) {
            array_size = j->first + j->second - param_pointer;
            break;
          }
        }
        assert(array_size > 0 && "pointer was not previously malloc'd");
      }

      unsigned char *val = (unsigned char *)malloc(param_value.size);
      param_mem->read(param_addr, param_value.size, (void *)val);
      unsigned char *array_val = (unsigned char *)malloc(array_size);
      gpu->get_global_memory()->read(*(unsigned *)((void *)val), array_size,
                                     (void *)array_val);
      param_data.push_back(
          std::pair<size_t, unsigned char *>(array_size, array_val));
      paramIsPointer.push_back(true);
    } else {
      unsigned char *val = (unsigned char *)malloc(param_value.size);
      param_mem->read(param_addr, param_value.size, (void *)val);
      param_data.push_back(
          std::pair<size_t, unsigned char *>(param_value.size, val));
      paramIsPointer.push_back(false);
    }
  }

  FILE *fout = fopen(filename.c_str(), "w");
  printf("Writing data to %s ...\n", filename.c_str());
  fprintf(fout, "%s\n", get_name().c_str());
  fprintf(fout, "%u,%u,%u %u,%u,%u\n", gridDim.x, gridDim.y, gridDim.z,
          blockDim.x, blockDim.y, blockDim.z);
  size_t index = 0;
  for (std::vector<std::pair<size_t, unsigned char *> >::const_iterator i =
           param_data.begin();
       i != param_data.end(); i++) {
    if (paramIsPointer[index]) {
      fprintf(fout, "*");
    }
    fprintf(fout, "%lu :", i->first);
    for (size_t j = 0; j < i->first; j++) {
      fprintf(fout, " %u", i->second[j]);
    }
    fprintf(fout, " : %u", offsets[index]);
    free(i->second);
    fprintf(fout, "\n");
    index++;
  }
  fflush(fout);
  fclose(fout);

  // ptx config
  std::string ptx_config_fn(std::string(gpgpusim_path) +
                            "/debug_tools/WatchYourStep/data/ptx.config" +
                            std::to_string(counter));
  snprintf(buff, 1024,
           "grep -rn \".entry %s\" %s/*.ptx | cut -d \":\" -f 1-2 > %s",
           get_name().c_str(), wys_exec_path, ptx_config_fn.c_str());
  if (system(buff) != 0) {
    printf("WARNING: Failed to execute grep to find ptx source \n");
    printf("Problematic call: %s", buff);
    abort();
  }
  FILE *fin = fopen(ptx_config_fn.c_str(), "r");
  char ptx_source[256];
  unsigned line_number;
  int numscanned = fscanf(fin, "%[^:]:%u", ptx_source, &line_number);
  assert(numscanned == 2);
  fclose(fin);
  snprintf(buff, 1024,
           "grep -rn \".version\" %s | cut -d \":\" -f 1 | xargs -I \"{}\" awk "
           "\"NR>={}&&NR<={}+2\" %s > %s",
           ptx_source, ptx_source, ptx_config_fn.c_str());
  if (system(buff) != 0) {
    printf("WARNING: Failed to execute grep to find ptx header \n");
    printf("Problematic call: %s", buff);
    abort();
  }
  fin = fopen(ptx_source, "r");
  assert(fin != NULL);
  printf("Writing data to %s ...\n", ptx_config_fn.c_str());
  fout = fopen(ptx_config_fn.c_str(), "a");
  assert(fout != NULL);
  for (unsigned i = 0; i < line_number; i++) {
    assert(fgets(buff, 1024, fin) != NULL);
    assert(!feof(fin));
  }
  fprintf(fout, "\n\n");
  do {
    fprintf(fout, "%s", buff);
    assert(fgets(buff, 1024, fin) != NULL);
    if (feof(fin)) {
      break;
    }
  } while (strstr(buff, "entry") == NULL);

  fclose(fin);
  fflush(fout);
  fclose(fout);
  counter++;
}

/*
DEBUG相关，用到再补充。
*/
template <int activate_level>
bool cuda_sim::ptx_debug_exec_dump_cond(int thd_uid, addr_t pc) {
  if (g_debug_execution >= activate_level) {
    // check each type of debug dump constraint to filter out dumps
    if ((g_debug_thread_uid != 0) &&
        (thd_uid != (unsigned)g_debug_thread_uid)) {
      return false;
    }
    if ((g_debug_pc != 0xBEEF1518) && (pc != g_debug_pc)) {
      return false;
    }

    return true;
  }

  return false;
}

/*
init_inst_classification_stat函数在下面的ptx_exec_inst函数中调用：
    if (m_gpu->gpgpu_ctx->func_sim->gpgpu_ptx_instruction_classification) {
      m_gpu->gpgpu_ctx->func_sim->init_inst_classification_stat();
      ...
    }
而gpgpu_ptx_instruction_classification在gpgpu-sim.cc中被配置，意为启用指令分类：
    option_parser_register(
      opp, "-gpgpu_ptx_instruction_classification", OPT_INT32,
      &(gpgpu_ctx->func_sim->gpgpu_ptx_instruction_classification),
      "if enabled will classify ptx instruction types per kernel (Max 255 "
      "kernels now)",
      "0");
在配置文件中，"-gpgpu_ptx_instruction_classification"默认设置为0，后面用到再补充。
*/
void cuda_sim::init_inst_classification_stat() {
  static std::set<unsigned> init;
  if (init.find(g_ptx_kernel_count) != init.end()) return;
  init.insert(g_ptx_kernel_count);

#define MAX_CLASS_KER 1024
  char kernelname[MAX_CLASS_KER] = "";
  if (!g_inst_classification_stat)
    g_inst_classification_stat = (void **)calloc(MAX_CLASS_KER, sizeof(void *));
  snprintf(kernelname, MAX_CLASS_KER, "Kernel %d Classification\n",
           g_ptx_kernel_count);
  assert(g_ptx_kernel_count <
         MAX_CLASS_KER);  // a static limit on number of kernels increase it if
                          // it fails!
  g_inst_classification_stat[g_ptx_kernel_count] =
      StatCreate(kernelname, 1, 20);
  if (!g_inst_op_classification_stat)
    g_inst_op_classification_stat =
        (void **)calloc(MAX_CLASS_KER, sizeof(void *));
  snprintf(kernelname, MAX_CLASS_KER, "Kernel %d OP Classification\n",
           g_ptx_kernel_count);
  g_inst_op_classification_stat[g_ptx_kernel_count] =
      StatCreate(kernelname, 1, 100);
}

/*
纹理存储相关，用到再补充。
*/
static unsigned get_tex_datasize(const ptx_instruction *pI,
                                 ptx_thread_info *thread) {
  const operand_info &src1 = pI->src1();  // the name of the texture
  std::string texname = src1.name();

  /*
    For programs with many streams, textures can be bound and unbound
    asynchronously.  This means we need to use the kernel's "snapshot" of
    the state of the texture mappings when it was launched (so that we
    don't try to access the incorrect texture mapping if it's been updated,
    or that we don't access a mapping that has been unbound).
    对于具有许多流的程序，可以异步绑定和解除绑定纹理。这意味着我们需要在启动时使用内核的纹理映射状态的
    “快照”（这样，如果纹理映射已更新，我们就不会尝试访问不正确的纹理映射，或者我们不会访问未绑定的映射）。
   */
  kernel_info_t &k = thread->get_kernel();
  const struct textureInfo *texInfo = k.get_texinfo(texname);

  unsigned data_size = texInfo->texel_size;
  return data_size;
}

/*
判断是否是Tensor Core上的指令，包括 mma,mma.load,mma.store，是则返回1，不是则返回0。
*/
int tensorcore_op(int inst_opcode) {
  if ((inst_opcode == MMA_OP) || (inst_opcode == MMA_LD_OP) ||
      (inst_opcode == MMA_ST_OP))
    return 1;
  else
    return 0;
}

/*
时序模型通过调用 ptx_thread_info 类的 ptx_exec_inst 方法将线程的功能状态提前一个指令。这是在
    core_t::execute_warp_inst_t
内完成的。时序模拟器传递要执行的指令的动态实例，而功能模型则相应地推进线程的状态。

在指令解析之后，用于功能执行的指令被表示为包含在 function_info 对象中的 ptx_instruction 对象（见 
cuda-sim/ptx_ir.{h/cc}）。每个标量线程由一个 ptx_thread_info 对象表示。执行一条指令（功能上）主要是
通过调用 ptx_thread_info::ptx_exec_inst() 完成的。
执行指令简单地从使用函数 ptx_sim_init_thread （在 cuda-sim.cc 中）初始化标量线程开始，然后使用 
function_info::ptx_exec_inst() 在warp中执行标量线程。

ptx_thread_info::ptx_exec_inst 是指令真正被执行的地方。我们检查指令的操作码并调用相应的函数，文件 
opcodes.def 包含用于执行每个指令的函数。每个指令函数都需要两个 ptx_instruction 和 ptx_thread_info 
的参数，这两个参数用于接收指令和执行中的线程的数据。
信息从 ptx_exec_inst 的执行中传回给执行warp的函数，通过修改以引用方式传递给 ptx_exec_inst 的 
warp_inst_t 参数，所以对于原子，我们表示执行的warp指令是原子的，并添加一个回调到 warp_inst_t ，设置
原子标志，然后由warp执行函数检查该标志，以便进行回调，用于执行原子（详见 cuda-sim.cc 中的
functionalCoreSim::executeWarp ）。

ptx_exec_inst(warp_inst_t &inst, unsigned lane_id)函数在abstract_hardware_model.h中引用：
    void core_t::execute_warp_inst_t(warp_inst_t &inst, unsigned warpId) {
      //t是指thread ID号，m_warp_size为warp的大小，即一个warp中线程的数量。
      for (unsigned t = 0; t < m_warp_size; t++) {
        //判断活跃编码，即判断由于分支的情况，该线程对当前指令的执行情况，inst.active(t)为1时，执行；
        //inst.active(t)为0时，不执行。
        if (inst.active(t)) {
          if (warpId == (unsigned(-1))) warpId = inst.warp_id();
          //计算当前线程编号。
          unsigned tid = m_warp_size * warpId + t;
          //让第t个线程执行inst指令。
          m_thread[tid]->ptx_exec_inst(inst, t);

          // virtual function
          checkExecutionStatusAndUpdate(inst, t, tid);
        }
      }
    }

ptx_exec_inst函数的参数 warp_inst_t &inst,unsigned lane_id，指的是，当前执行线程的ID和指令inst。
*/
void ptx_thread_info::ptx_exec_inst(warp_inst_t &inst, unsigned lane_id) {
  //后面检查指令是否为谓词指令用到：如果是，检查是否满足谓词条件，并相应地更新标志[skip]，以指示是否
  //应执行此指令或者跳过执行该指令。
  bool skip = false;
  int op_classification = 0;
  //获取下一条指令的PC值。
  addr_t pc = next_instr();
  assert(pc ==      //确保时序模型和功能模型同步。
         inst.pc);  // make sure timing model and functional model are in sync

  //根据pc值获得当前执行的指令 ptx_instruction 对象 pI。
  const ptx_instruction *pI = m_func_info->get_instruction(pc);

  //设置下一条指令的PC值，NPC=pc+pI指令大小。
  set_npc(pc + pI->inst_size());

  try {
    //这里RPC是指分歧线程的直接重新汇合点PC（reconvergence PC，RPC）。首先要清楚这一个RPC，因为
    //刚刚获取指令。
    clearRPC();
    m_last_set_operand_value.u64 = 0;

    //已经执行完毕。
    if (is_done()) {
      printf(
          "attempted to execute instruction on a thread that is already "
          "done.\n");
      assert(0);
    }

    //DEBUG用，后续用到再补充。
    if (g_debug_execution >= 6 ||
        m_gpu->get_config().get_ptx_inst_debug_to_file()) {
      if ((m_gpu->gpgpu_ctx->func_sim->g_debug_thread_uid == 0) ||
          (get_uid() ==
           (unsigned)(m_gpu->gpgpu_ctx->func_sim->g_debug_thread_uid))) {
        clear_modifiedregs();
        enable_debug_trace();
      }
    }

    //从 m_func_info 对象（类型为 function_info 的成员 ptx_thread_info）中的 m_instr_mem[] 获取
    //ptx_instruction 对象 pI 后，检查 pI 是否为谓词指令，如果是，检查是否满足谓词条件，并相应地更
    //新标志[skip]，以指示是否应执行此指令。
    //关于谓词指令，可以看下面的PTX指令：
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
    
    //如果一条指令具有谓词寄存器。
    if (pI->has_pred()) {
      //获取谓词寄存器，并将其放入 operand_info 对象 pred。
      const operand_info &pred = pI->get_pred();
      //获取谓词寄存器的值。
      ptx_reg_t pred_value = get_operand_value(pred, pred, PRED_TYPE, this, 0);
      //设置[skip]标志，以后续停用线程通道。
      if (pI->get_pred_mod() == -1) {
        skip = (pred_value.pred & 0x0001) ^
               pI->get_pred_neg();  // ptxplus inverts the zero flag
      } else {
        skip = !pred_lookup(pI->get_pred_mod(), pred_value.pred & 0x000F);
      }
    }
    //获取操作码。
    int inst_opcode = pI->get_opcode();

    if (skip) {
      //在谓词测试之后，如果设置了[skip]跳过标志，则停用线程通道，否则执行该条指令。
      inst.set_not_active(lane_id);
    } else { // 正常执行该线程通道。
      const ptx_instruction *pI_saved = pI;
      ptx_instruction *pJ = NULL;
      //VOTE_OP：Vote across thread group.
      //在Warp中的所有活动线程上执行源谓词的Reduction。目标谓词值在warp中的所有线程中都是相同的。
      //Reduction模式包括：
      //    .all：如果warp中所有活动线程的源谓词为True，则为True。对源谓词求反以计算.none。
      //    .any：如果warp中某些活动线程的源谓词为True，则为True。对源谓词求反以计算.not_all。
      //    .uni:如果源谓词在warp中的所有活动线程中具有相同的值，则为True。对源谓词求反也会计算.uni。
      //vote.ballot.b32简单地将谓词从warp中的每个线程复制到目标寄存器 d 的相应位，其中每个位对应于线程
      //的通道id。
      //
      //ACTIVEMASK_OP：Queries the active threads within a warp.
      //activemask查询基于来自执行warp的活动线程，并使用32位整数掩码设置目标 d，其中掩码中的每个位对应于
      //线程的laneid。目标 d 是32位目标寄存器。
      //活动线程将为其在结果中的条目贡献 1，而已退出或非活动或断言的线程将为结果中的其条目贡献 0。
      if (pI->get_opcode() == VOTE_OP || pI->get_opcode() == ACTIVEMASK_OP) {
        pJ = new ptx_instruction(*pI);
        //拷贝活跃掩码。
        *((warp_inst_t *)pJ) = inst;  // copy active mask information
        pI = pJ;
      }

      //Tensor Core中的wmma相关指令：包括MMA，LOAD，STORE。
      if (((inst_opcode == MMA_OP || inst_opcode == MMA_LD_OP ||
            inst_opcode == MMA_ST_OP))) {
        if (inst.active_count() != MAX_WARP_SIZE) {
          printf(
              "Tensor Core operation are warp synchronous operation. All the "
              "threads needs to be active.");
          assert(0);
        }
      }

      // Tensorcore is warp synchronous operation. So these instructions needs
      // to be executed only once. To make the simulation faster removing the
      // redundant tensorcore operation
      //Tensorcore是warp同步操作。所以这些指令只需要执行一次。为了使模拟更快地去除冗余的Tensorcore操
      //作。
      if (!tensorcore_op(inst_opcode) ||
          ((tensorcore_op(inst_opcode)) && (lane_id == 0))) {
        switch (inst_opcode) {
#define OP_DEF(OP, FUNC, STR, DST, CLASSIFICATION) \
  case OP:                                         \
    FUNC(pI, this);                                \
    op_classification = CLASSIFICATION;            \
    break;
#define OP_W_DEF(OP, FUNC, STR, DST, CLASSIFICATION) \
  case OP:                                           \
    FUNC(pI, get_core(), inst);                      \
    op_classification = CLASSIFICATION;              \
    break;
#include "opcodes.def"
#undef OP_DEF
#undef OP_W_DEF
          default:
            printf("Execution error: Invalid opcode (0x%x)\n",
                   pI->get_opcode());
            break;
        }
      }

      //在上面的代码中，实际的指令执行是在switch-case块中模拟的，其中包含一个名为opcodes.def的文件，
      //用于为每种类型的指令指定模拟函数。opcodes.def文件类似于：
      //    OP_DEF(ABS_OP,abs_impl,"abs",1,1)
      //    OP_DEF(ADD_OP,add_impl,"add",1,1)
      //    OP_DEF(ADDP_OP,addp_impl,"addp",1,1)
      //    OP_DEF(ADDC_OP,addc_impl,"addc",1,1)
      //    OP_DEF(AND_OP,and_impl,"and",1,1)
      //    OP_DEF(ANDN_OP,andn_impl,"andn",1,1)
      //    OP_DEF(ATOM_OP,atom_impl,"atom",1,3)
      //    OP_DEF(BAR_OP,bar_sync_impl,"bar.sync",1,3)
      //    OP_DEF(BFE_OP,bfe_impl,"bfe",1,1)
      //    OP_DEF(BFI_OP,bfi_impl,"bfi",1,1)
      //    OP_DEF(BFIND_OP,bfind_impl,"bfind",1,1)
      //    OP_DEF(BRA_OP,bra_impl,"bra",0,3)
      //并且每个OP_DEF宏的第二参数作为指向处理相应指令类型的函数的指针。例如，对于bar指令，处理函数是
      //src/cuda-sim/instructions.cc中定义的bra_impl：
      //    void bra_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
      //        const operand_info &target = pI->dst();
      //        ptx_reg_t target_pc =
      //        thread->get_operand_value(target, target, U32_TYPE, thread, 1);
      //        thread->m_branch_taken = true;
      //        thread->set_npc(target_pc);
      //    }
      //需要注意的是，当前GPGPU-Sim版本中仍有许多指令未实现。

      delete pJ;
      pI = pI_saved;

      // Run exit instruction if exit option included
      //执行exit指令。
      if (pI->is_exit()) exit_impl(pI, this);
    }

    //获取GPU硬件的配置信息。
    const gpgpu_functional_sim_config &config = m_gpu->get_config();

    // Output instruction information to file and stdout
    //输出指令信息。DEBUG用，后续再补充。
    if (config.get_ptx_inst_debug_to_file() != 0 &&
        (config.get_ptx_inst_debug_thread_uid() == 0 ||
         config.get_ptx_inst_debug_thread_uid() == get_uid())) {
      fprintf(m_gpu->get_ptx_inst_debug_file(), "[thd=%u] : (%s:%u - %s)\n",
              get_uid(), pI->source_file(), pI->source_line(),
              pI->get_source());
      // fprintf(ptx_inst_debug_file, "has memory read=%d, has memory
      // write=%d\n", pI->has_memory_read(), pI->has_memory_write());
      fflush(m_gpu->get_ptx_inst_debug_file());
    }

    //DEBUG用，后续再补充。
    if (m_gpu->gpgpu_ctx->func_sim->ptx_debug_exec_dump_cond<5>(get_uid(),
                                                                pc)) {
      dim3 ctaid = get_ctaid();
      dim3 tid = get_tid();
      printf(
          "%u [thd=%u][i=%u] : ctaid=(%u,%u,%u) tid=(%u,%u,%u) icount=%u "
          "[pc=%llu] (%s:%u - %s)  [0x%llx]\n",
          m_gpu->gpgpu_ctx->func_sim->g_ptx_sim_num_insn, get_uid(), pI->uid(),
          ctaid.x, ctaid.y, ctaid.z, tid.x, tid.y, tid.z, get_icount(), pc,
          pI->source_file(), pI->source_line(), pI->get_source(),
          m_last_set_operand_value.u64);
      fflush(stdout);
    }

    addr_t insn_memaddr = 0xFEEBDAED;
    memory_space_t insn_space = undefined_space;
    _memory_op_t insn_memory_op = no_memory_op;
    unsigned insn_data_size = 0;

    //ptx_thread_info::ptx_exec_inst()函数之后的代码主要是调试和统计信息收集代码。
    //然而，ptx_thread_info::ptx_exec_inst()中下述代码段需要解释：
    //下面的代码检查指令是否包含内存读取或写入，如果是，则设置一些变量以供以后使用。
    if ((pI->has_memory_read() || pI->has_memory_write())) {
      if (!((inst_opcode == MMA_LD_OP || inst_opcode == MMA_ST_OP))) {
        insn_memaddr = last_eaddr();
        insn_space = last_space();
        unsigned to_type = pI->get_type();
        insn_data_size = datatype2size(to_type);
        insn_memory_op = pI->has_memory_read() ? memory_load : memory_store;
      }
    }

    if (pI->get_opcode() == BAR_OP && pI->barrier_op() == RED_OPTION) {
      inst.add_callback(lane_id, last_callback().function,
                        last_callback().instruction, this,
                        false /*not atomic*/);
    }

    //下面的代码检查原子指令并设置回调函数来处理原子指令。
    if (pI->get_opcode() == ATOM_OP) {
      insn_memaddr = last_eaddr();
      insn_space = last_space();
      inst.add_callback(lane_id, last_callback().function,
                        last_callback().instruction, this, true /*atomic*/);
      unsigned to_type = pI->get_type();
      insn_data_size = datatype2size(to_type);
    }

    //下面的代码检查纹理操作。
    if (pI->get_opcode() == TEX_OP) {
      inst.set_addr(lane_id, last_eaddr());
      assert(inst.space == last_space());
      insn_data_size = get_tex_datasize(
          pI,
          this);  // texture obtain its data granularity from the texture info
    }

    // Output register information to file and stdout
    if (config.get_ptx_inst_debug_to_file() != 0 &&
        (config.get_ptx_inst_debug_thread_uid() == 0 ||
         config.get_ptx_inst_debug_thread_uid() == get_uid())) {
      dump_modifiedregs(m_gpu->get_ptx_inst_debug_file());
      dump_regs(m_gpu->get_ptx_inst_debug_file());
    }

    if (g_debug_execution >= 6) {
      if (m_gpu->gpgpu_ctx->func_sim->ptx_debug_exec_dump_cond<6>(get_uid(),
                                                                  pc))
        dump_modifiedregs(stdout);
    }
    if (g_debug_execution >= 10) {
      if (m_gpu->gpgpu_ctx->func_sim->ptx_debug_exec_dump_cond<10>(get_uid(),
                                                                   pc))
        dump_regs(stdout);
    }
    update_pc();
    m_gpu->gpgpu_ctx->func_sim->g_ptx_sim_num_insn++;

    // not using it with functional simulation mode
    if (!(this->m_functionalSimulationMode))
      ptx_file_line_stats_add_exec_count(pI);

    if (m_gpu->gpgpu_ctx->func_sim->gpgpu_ptx_instruction_classification) {
      m_gpu->gpgpu_ctx->func_sim->init_inst_classification_stat();
      unsigned space_type = 0;
      switch (pI->get_space().get_type()) {
        case global_space:
          space_type = 10;
          break;
        case local_space:
          space_type = 11;
          break;
        case tex_space:
          space_type = 12;
          break;
        case surf_space:
          space_type = 13;
          break;
        case param_space_kernel:
        case param_space_local:
          space_type = 14;
          break;
        case shared_space:
          space_type = 15;
          break;
        case const_space:
          space_type = 16;
          break;
        default:
          space_type = 0;
          break;
      }
      StatAddSample(m_gpu->gpgpu_ctx->func_sim->g_inst_classification_stat
                        [m_gpu->gpgpu_ctx->func_sim->g_ptx_kernel_count],
                    op_classification);
      if (space_type)
        StatAddSample(m_gpu->gpgpu_ctx->func_sim->g_inst_classification_stat
                          [m_gpu->gpgpu_ctx->func_sim->g_ptx_kernel_count],
                      (int)space_type);
      StatAddSample(m_gpu->gpgpu_ctx->func_sim->g_inst_op_classification_stat
                        [m_gpu->gpgpu_ctx->func_sim->g_ptx_kernel_count],
                    (int)pI->get_opcode());
    }
    if ((m_gpu->gpgpu_ctx->func_sim->g_ptx_sim_num_insn % 100000) == 0) {
      dim3 ctaid = get_ctaid();
      dim3 tid = get_tid();
      DPRINTF(LIVENESS,
              "GPGPU-Sim PTX: %u instructions simulated : ctaid=(%u,%u,%u) "
              "tid=(%u,%u,%u)\n",
              m_gpu->gpgpu_ctx->func_sim->g_ptx_sim_num_insn, ctaid.x, ctaid.y,
              ctaid.z, tid.x, tid.y, tid.z);
      fflush(stdout);
    }

    // "Return values"
    if (!skip) {
      if (!((inst_opcode == MMA_LD_OP || inst_opcode == MMA_ST_OP))) {
        inst.space = insn_space;
        inst.set_addr(lane_id, insn_memaddr);
        inst.data_size = insn_data_size;  // simpleAtomicIntrinsics
        assert(inst.memory_op == insn_memory_op);
      }
    }

  } catch (int x) {
    printf("GPGPU-Sim PTX: ERROR (%d) executing intruction (%s:%u)\n", x,
           pI->source_file(), pI->source_line());
    printf("GPGPU-Sim PTX:       '%s'\n", pI->get_source());
    abort();
  }
}

/*
设置GPU的SIMT Core（Shader Core）的个数，即SM的个数。
*/
void cuda_sim::set_param_gpgpu_num_shaders(int num_shaders) {
  gpgpu_param_num_shaders = num_shaders;
}

/*
返回内核函数的属性，如PTX版本和目标SM等。也保存该内核所使内存和寄存器的数量等。
set_kernel_info()在ptx_ir.h中定义：
    virtual const void set_kernel_info(const struct gpgpu_ptx_sim_info &info) {
        m_kernel_info = info;
        m_kernel_info.ptx_version = 10 * get_ptx_version().ver();
        m_kernel_info.sm_target = get_ptx_version().target();
        // THIS DEPENDS ON ptxas being called after the PTX is parsed.
      m_kernel_info.maxthreads = maxnt_id;
    }
其中gpgpu_ptx_sim_info又在abstract_hardware_model.h中定义：
    struct gpgpu_ptx_sim_info {
        // Holds properties of the kernel (Kernel's resource use).
        // These will be set to zero if a ptxinfo file is not present.
        //保存内核的属性（内核的资源使用）。
        //如果不存在ptxinfo文件，这些值将设置为零。
        int lmem;                 // local memory大小
        int smem;                 // shared memory大小
        int cmem;                 // constant memory大小
        int gmem;                 // global memory大小
        int regs;                 // 寄存器数量
        unsigned maxthreads;      // 最大线程数
        unsigned ptx_version;     // PTX version
        unsigned sm_target;       // 目标 SM
    };
m_kernel_info包括：Registers/shmem/etc. used (from ptxas -v), loaded from ___.ptxinfo along 
with ___.ptx。例如，cuda_codes/myTest/文件夹下面的PTX文件：
     .version 7.5
     .target sm_52
     .address_size 64
     .visible .entry _Z6MatMulPiS_S_i(
         .param .u64 _Z6MatMulPiS_S_i_param_0,
         .param .u64 _Z6MatMulPiS_S_i_param_1,
         .param .u64 _Z6MatMulPiS_S_i_param_2,
         .param .u32 _Z6MatMulPiS_S_i_param_3
     )
     ......
使用 ptxas 后，PTXAS文件：
     ptxas info    : 0 bytes gmem
     ptxas info    : Compiling entry function '_Z6MatMulPiS_S_i' for 'sm_52'
     ptxas info    : Function properties for _Z6MatMulPiS_S_i
         0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
     ptxas info    : Used 32 registers, 348 bytes cmem[0]
*/
const struct gpgpu_ptx_sim_info *ptx_sim_kernel_info(
    const function_info *kernel) {
  return kernel->get_kernel_info();
}

/*
依据PC值，获取指令。
*/
const warp_inst_t *gpgpu_context::ptx_fetch_inst(address_type pc) {
  return pc_to_instruction(pc);
}

/*
执行指令简单地从使用函数 ptx_sim_init_thread 初始化标量线程开始，然后使用上面的 ptx_exec_inst() 在
warp中执行标量线程。即每个线程的功能状态通过调用 ptx_sim_init_thread 进行初始化。这里的函数仅是对单
个线程的指令进行初始化的，参数中包括指定某个线程的 unsigned hw_cta_id, unsigned hw_warp_id, 以及
ptx_thread_info **thread_info, int sid, unsigned tid。需要注意的是，这类进行的是单个CTA内的所有
线程进行循环。

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
unsigned ptx_sim_init_thread(kernel_info_t &kernel,
                             ptx_thread_info **thread_info, int sid,
                             unsigned tid, unsigned threads_left,
                             unsigned num_threads, core_t *core,
                             unsigned hw_cta_id, unsigned hw_warp_id,
                             gpgpu_t *gpu, bool isInFunctionalSimulationMode) {
  //根据给定的内核信息kernel，获取该内核中的活跃线程，返回的是一个活跃线程列表。
  std::list<ptx_thread_info *> &active_threads = kernel.active_threads();
  
  //内存空间是由下列查找表来管理的。
  static std::map<unsigned, memory_space *> shared_memory_lookup;
  static std::map<unsigned, memory_space *> sstarr_memory_lookup;
  static std::map<unsigned, ptx_cta_info *> ptx_cta_lookup;
  static std::map<unsigned, ptx_warp_info *> ptx_warp_lookup;
  static std::map<unsigned, std::map<unsigned, memory_space *> >
      local_memory_lookup;

  //thread_info线程信息非空，线程还保留着上一个执行线程的信息，需要将此信息删除。
  if (*thread_info != NULL) {
    ptx_thread_info *thd = *thread_info;
    assert(thd->is_done());
    if (g_debug_execution == -1) {
      dim3 ctaid = thd->get_ctaid();
      dim3 t = thd->get_tid();
      printf(
          "GPGPU-Sim PTX simulator:  thread exiting ctaid=(%u,%u,%u) "
          "tid=(%u,%u,%u) uid=%u\n",
          ctaid.x, ctaid.y, ctaid.z, t.x, t.y, t.z, thd->get_uid());
      fflush(stdout);
    }
    //挂起thd指向的线程。
    thd->m_cta_info->register_deleted_thread(thd);
    delete thd;
    //删除thread_info的信息。
    *thread_info = NULL;
  }

  //内核中的活跃线程非空，代表后续要执行，需要初始化线程。
  //ptx_thread_info对象包含单个标量线程（OpenCL中的work item）的功能仿真状态。这包括以下内容：
  //  a. 寄存器值存储
  //  b. 本地内存存储（OpenCL中的私有内存）
  //  c. 共享内存存储（OpenCL中的本地内存）。注意，同一线程块/Work Wrap的所有标量线程都会访问相同的
  //     共享内存存储。
  //  e. 程序计数器（PC）
  //  f. 调用堆栈
  //  g. 线程ID（网格启动中的软件ID，以及表明它在时序模型中占据哪个硬件线程槽的硬件ID)
  if (!active_threads.empty()) {
    assert(active_threads.size() <= threads_left);
    //从活跃线程队列中弹出一个。
    ptx_thread_info *thd = active_threads.front();
    active_threads.pop_front();
    *thread_info = thd;
    //初始化线程。
    thd->init(gpu, core, sid, hw_cta_id, hw_warp_id, tid,
              isInFunctionalSimulationMode);
    return 1;
  }

  //没有更多的CTA可执行，内核函数的所有线程块均已经执行完毕。
  if (kernel.no_more_ctas_to_run()) {
    return 0;  // finished!
  }

  if (threads_left < kernel.threads_per_cta()) {
    return 0;
  }

  //DEBUG用，后续再补充。
  if (g_debug_execution == -1) {
    printf("GPGPU-Sim PTX simulator:  STARTING THREAD ALLOCATION --> \n");
    fflush(stdout);
  }

  // initializing new CTA
  //初始化新的CTA。
  ptx_cta_info *cta_info = NULL;
  memory_space *shared_mem = NULL;
  memory_space *sstarr_mem = NULL;

  //根据内核函数的信息kernel获取其参数中的每个CTA（线程块）中的线程数。
  unsigned cta_size = kernel.threads_per_cta();
  //判断该内核函数的执行需要，每个SM最大的线程块数量，num_threads为总线程数，cta_size为每个CTA（线
  //程块）中的线程数。max_cta_per_sm 这个参数其实没啥用，因为它是假设的硬件只有一个SM，后面这个参数
  //也用不到。
  unsigned max_cta_per_sm = num_threads / cta_size;  // e.g., 256 / 48 = 5
  assert(max_cta_per_sm > 0);

  // unsigned sm_idx = (tid/cta_size)*gpgpu_param_num_shaders + sid;
  //gpgpu_param_num_shaders为GPU的SIMT Core的个数，即SM的个数。
  unsigned sm_idx =
      hw_cta_id * gpu->gpgpu_ctx->func_sim->gpgpu_param_num_shaders + sid;

  //printf("<gpu->gpgpu_ctx->func_sim->gpgpu_param_num_shaders>=%d,<hw_cta_id>=%d,<sid>=%d\n", 
  //        gpu->gpgpu_ctx->func_sim->gpgpu_param_num_shaders, hw_cta_id, sid);
  
  if (shared_memory_lookup.find(sm_idx) == shared_memory_lookup.end()) {
    if (g_debug_execution >= 1) {
      printf("  <CTA alloc> : sm_idx=%u sid=%u max_cta_per_sm=%u\n", sm_idx,
             sid, max_cta_per_sm);
    }
    char buf[512];
    snprintf(buf, 512, "shared_%u", sid);
    shared_mem = new memory_space_impl<16 * 1024>(buf, 4);
    shared_memory_lookup[sm_idx] = shared_mem;
    snprintf(buf, 512, "sstarr_%u", sid);
    sstarr_mem = new memory_space_impl<16 * 1024>(buf, 4);
    sstarr_memory_lookup[sm_idx] = sstarr_mem;
    cta_info = new ptx_cta_info(sm_idx, gpu->gpgpu_ctx);
    ptx_cta_lookup[sm_idx] = cta_info;
  } else {
    if (g_debug_execution >= 1) {
      printf("  <CTA realloc> : sm_idx=%u sid=%u max_cta_per_sm=%u\n", sm_idx,
             sid, max_cta_per_sm);
    }
    //依据SM ID，获取当前SM的shared_mem和sstarr_mem，cta_info
    shared_mem = shared_memory_lookup[sm_idx];
    sstarr_mem = sstarr_memory_lookup[sm_idx];
    cta_info = ptx_cta_lookup[sm_idx];
    cta_info->check_cta_thread_status_and_reset();
  }
  
  //依据SM ID，获取当前SM的local_mem。
  std::map<unsigned, memory_space *> &local_mem_lookup =
      local_memory_lookup[sid];
  while (kernel.more_threads_in_cta()) {
    dim3 ctaid3d = kernel.get_next_cta_id();
    unsigned new_tid = kernel.get_next_thread_id();
    dim3 tid3d = kernel.get_next_thread_id_3d();
    kernel.increment_thread_id();
    new_tid += tid;
    ptx_thread_info *thd = new ptx_thread_info(kernel);
    ptx_warp_info *warp_info = NULL;
    if (ptx_warp_lookup.find(hw_warp_id) == ptx_warp_lookup.end()) {
      warp_info = new ptx_warp_info();
      ptx_warp_lookup[hw_warp_id] = warp_info;
    } else {
      warp_info = ptx_warp_lookup[hw_warp_id];
    }
    thd->m_warp_info = warp_info;

    memory_space *local_mem = NULL;
    std::map<unsigned, memory_space *>::iterator l =
        local_mem_lookup.find(new_tid);
    if (l != local_mem_lookup.end()) {
      local_mem = l->second;
    } else {
      char buf[512];
      snprintf(buf, 512, "local_%u_%u", sid, new_tid);
      local_mem = new memory_space_impl<32>(buf, 32);
      local_mem_lookup[new_tid] = local_mem;
    }
    thd->set_info(kernel.entry());
    thd->set_nctaid(kernel.get_grid_dim());
    thd->set_ntid(kernel.get_cta_dim());
    thd->set_ctaid(ctaid3d);
    thd->set_tid(tid3d);
    if (kernel.entry()->get_ptx_version().extensions())
      thd->cpy_tid_to_reg(tid3d);
    thd->set_valid();
    thd->m_shared_mem = shared_mem;
    thd->m_sstarr_mem = sstarr_mem;
    function_info *finfo = thd->func_info();
    symbol_table *st = finfo->get_symtab();
    thd->func_info()->param_to_shared(thd->m_shared_mem, st);
    thd->func_info()->param_to_shared(thd->m_sstarr_mem, st);
    thd->m_cta_info = cta_info;
    cta_info->add_thread(thd);
    thd->m_local_mem = local_mem;
    if (g_debug_execution == -1) {
      printf(
          "GPGPU-Sim PTX simulator:  allocating thread ctaid=(%u,%u,%u) "
          "tid=(%u,%u,%u) @ 0x%Lx\n",
          ctaid3d.x, ctaid3d.y, ctaid3d.z, tid3d.x, tid3d.y, tid3d.z,
          (unsigned long long)thd);
      fflush(stdout);
    }
    active_threads.push_back(thd);
  }

  //DEBUG用，后续再补充。
  if (g_debug_execution == -1) {
    printf("GPGPU-Sim PTX simulator:  <-- FINISHING THREAD ALLOCATION\n");
    fflush(stdout);
  }

  kernel.increment_cta_id();

  assert(active_threads.size() <= threads_left);
  *thread_info = active_threads.front();
  (*thread_info)
      ->init(gpu, core, sid, hw_cta_id, hw_warp_id, tid,
             isInFunctionalSimulationMode);
  active_threads.pop_front();
  return 1;
}

/*
返回内核函数编译后的PTX指令的条数。
*/
size_t get_kernel_code_size(class function_info *entry) {
  return entry->get_function_size();
}

/*
OPENCL用，后续需要再补充。
*/
kernel_info_t *cuda_sim::gpgpu_opencl_ptx_sim_init_grid(
    class function_info *entry, gpgpu_ptx_sim_arg_list_t args,
    struct dim3 gridDim, struct dim3 blockDim, gpgpu_t *gpu) {
  kernel_info_t *result =
      new kernel_info_t(gridDim, blockDim, entry, gpu->getNameArrayMapping(),
                        gpu->getNameInfoMapping());
  unsigned argcount = args.size();
  unsigned argn = 1;
  for (gpgpu_ptx_sim_arg_list_t::iterator a = args.begin(); a != args.end();
       a++) {
    entry->add_param_data(argcount - argn, &(*a));
    argn++;
  }
  entry->finalize(result->get_param_memory());
  g_ptx_kernel_count++;
  fflush(stdout);

  return result;
}

#include "../../version"
#include "detailed_version"

/*
打印GPGPU-Sim内核函数输出文件的第一行信息：
        *** GPGPU-Sim Simulator Version 4.0.0  [build gpgpu-sim_git-commit-2435b16585df2e23f867d605c9f150170b891768_modified_4.0] ***
*/
void print_splash() {
  static int splash_printed = 0;
  if (!splash_printed) {
    fprintf(stdout, "\n\n        *** %s [build %s] ***\n\n\n",
            g_gpgpusim_version_string, g_gpgpusim_build_string);
    splash_printed = 1;
  }
}

/*
此函数只声明但没有调用，后续需要再补充。
*/
void cuda_sim::gpgpu_ptx_sim_register_const_variable(void *hostVar,
                                                     const char *deviceName,
                                                     size_t size) {
  printf("GPGPU-Sim PTX registering constant %s (%zu bytes) to name mapping\n",
         deviceName, size);
  g_const_name_lookup[hostVar] = deviceName;
}

/*
此函数只声明但没有调用，后续需要再补充。
*/
void cuda_sim::gpgpu_ptx_sim_register_global_variable(void *hostVar,
                                                      const char *deviceName,
                                                      size_t size) {
  printf("GPGPU-Sim PTX registering global %s hostVar to name mapping\n",
         deviceName);
  g_global_name_lookup[hostVar] = deviceName;
}

/*
DEBUG用，在stream_manager.cc中有调用，后续需要再补充。
*/
void cuda_sim::gpgpu_ptx_sim_memcpy_symbol(const char *hostVar, const void *src,
                                           size_t count, size_t offset, int to,
                                           gpgpu_t *gpu) {
  printf(
      "GPGPU-Sim PTX: starting gpgpu_ptx_sim_memcpy_symbol with hostVar 0x%p\n",
      hostVar);
  bool found_sym = false;
  memory_space_t mem_region = undefined_space;
  std::string sym_name;

  std::map<const void *, std::string>::iterator c =
      gpu->gpgpu_ctx->func_sim->g_const_name_lookup.find(hostVar);
  if (c != gpu->gpgpu_ctx->func_sim->g_const_name_lookup.end()) {
    found_sym = true;
    sym_name = c->second;
    mem_region = const_space;
  }
  std::map<const void *, std::string>::iterator g =
      gpu->gpgpu_ctx->func_sim->g_global_name_lookup.find(hostVar);
  if (g != gpu->gpgpu_ctx->func_sim->g_global_name_lookup.end()) {
    if (found_sym) {
      printf(
          "Execution error: PTX symbol \"%s\" w/ hostVar=0x%Lx is declared "
          "both const and global?\n",
          sym_name.c_str(), (unsigned long long)hostVar);
      abort();
    }
    found_sym = true;
    sym_name = g->second;
    mem_region = global_space;
  }
  if (g_globals.find(hostVar) != g_globals.end()) {
    found_sym = true;
    sym_name = hostVar;
    mem_region = global_space;
  }
  if (g_constants.find(hostVar) != g_constants.end()) {
    found_sym = true;
    sym_name = hostVar;
    mem_region = const_space;
  }

  if (!found_sym) {
    printf("Execution error: No information for PTX symbol w/ hostVar=0x%Lx\n",
           (unsigned long long)hostVar);
    abort();
  } else
    printf(
        "GPGPU-Sim PTX: gpgpu_ptx_sim_memcpy_symbol: Found PTX symbol w/ "
        "hostVar=0x%Lx\n",
        (unsigned long long)hostVar);
  const char *mem_name = NULL;
  memory_space *mem = NULL;

  std::map<std::string, symbol_table *>::iterator st =
      gpgpu_ctx->ptx_parser->g_sym_name_to_symbol_table.find(sym_name.c_str());
  assert(st != gpgpu_ctx->ptx_parser->g_sym_name_to_symbol_table.end());
  symbol_table *symtab = st->second;

  symbol *sym = symtab->lookup(sym_name.c_str());
  assert(sym);
  unsigned dst = sym->get_address() + offset;
  switch (mem_region.get_type()) {
    case const_space:
      mem = gpu->get_global_memory();
      mem_name = "const";
      break;
    case global_space:
      mem = gpu->get_global_memory();
      mem_name = "global";
      break;
    default:
      abort();
  }
  printf(
      "GPGPU-Sim PTX: gpgpu_ptx_sim_memcpy_symbol: copying %s memory %zu bytes "
      "%s symbol %s+%zu @0x%x ...\n",
      mem_name, count, (to ? " to " : "from"), sym_name.c_str(), offset, dst);
  for (unsigned n = 0; n < count; n++) {
    if (to)
      mem->write(dst + n, 1, ((char *)src) + n, NULL, NULL);
    else
      mem->read(dst + n, 1, ((char *)src) + n);
  }
  fflush(stdout);
}

extern int ptx_debug;

/*
读取一些gpgpu.config环境变量配置。
*/
void cuda_sim::read_sim_environment_variables() {
  ptx_debug = 0;
  g_debug_execution = 0;
  g_interactive_debugger_enabled = false;

  char *mode = getenv("PTX_SIM_MODE_FUNC");
  if (mode) sscanf(mode, "%u", &g_ptx_sim_mode);
  printf(
      "GPGPU-Sim PTX: simulation mode %d (can change with PTX_SIM_MODE_FUNC "
      "environment variable:\n",
      g_ptx_sim_mode);
  printf(
      "               1=functional simulation only, 0=detailed performance "
      "simulator)\n");
  char *dbg_inter = getenv("GPGPUSIM_DEBUG");
  if (dbg_inter && strlen(dbg_inter)) {
    printf("GPGPU-Sim PTX: enabling interactive debugger\n");
    fflush(stdout);
    g_interactive_debugger_enabled = true;
  }
  char *dbg_level = getenv("PTX_SIM_DEBUG");
  if (dbg_level && strlen(dbg_level)) {
    printf("GPGPU-Sim PTX: setting debug level to %s\n", dbg_level);
    fflush(stdout);
    sscanf(dbg_level, "%d", &g_debug_execution);
  }
  char *dbg_thread = getenv("PTX_SIM_DEBUG_THREAD_UID");
  if (dbg_thread && strlen(dbg_thread)) {
    printf("GPGPU-Sim PTX: printing debug information for thread uid %s\n",
           dbg_thread);
    fflush(stdout);
    sscanf(dbg_thread, "%d", &g_debug_thread_uid);
  }
  char *dbg_pc = getenv("PTX_SIM_DEBUG_PC");
  if (dbg_pc && strlen(dbg_pc)) {
    printf(
        "GPGPU-Sim PTX: printing debug information for instruction with PC = "
        "%s\n",
        dbg_pc);
    fflush(stdout);
    sscanf(dbg_pc, "%llu", &g_debug_pc);
  }

#if CUDART_VERSION > 1010
  g_override_embedded_ptx = false;
  char *usefile = getenv("PTX_SIM_USE_PTX_FILE");
  if (usefile && strlen(usefile)) {
    printf(
        "GPGPU-Sim PTX: overriding embedded ptx with ptx file "
        "(PTX_SIM_USE_PTX_FILE is set)\n");
    fflush(stdout);
    g_override_embedded_ptx = true;
  }
  char *blocking = getenv("CUDA_LAUNCH_BLOCKING");
  if (blocking && !strcmp(blocking, "1")) {
    g_cuda_launch_blocking = true;
  }
#else
  g_cuda_launch_blocking = true;
  g_override_embedded_ptx = true;
#endif

  if (g_debug_execution >= 40) {
    ptx_debug = 1;
  }
}

#define MAX(a, b) (((a) > (b)) ? (a) : (b))

/*
一个Core上可同时调度的最大线程块（或CTA或Warp Group）的数量由函数 max_cta(...) 计算。这个函数根据
程序指定的每个线程块的数量、每个线程寄存器的使用情况、共享内存的使用情况以及配置的每个Core最大线程块
数量的限制，确定可以并发分配给单个SIMT Core的最大线程块数量。具体来说，如果上述每个标准都是限制因素，
那么可以分配给SIMT Core的线程块的数量被计算出来。其中的最小值就是可以分配给SIMT Core的最大线程块数。
函数的调用为：
    unsigned max_cta_tot = max_cta(
      kernel_info, 
      kernel.threads_per_cta(),
      gpgpu_ctx->the_gpgpusim->g_the_gpu->getShaderCoreConfig()->warp_size,
      gpgpu_ctx->the_gpgpusim->g_the_gpu->getShaderCoreConfig()->n_thread_per_shader,
      gpgpu_ctx->the_gpgpusim->g_the_gpu->getShaderCoreConfig()->gpgpu_shmem_size,
      gpgpu_ctx->the_gpgpusim->g_the_gpu->getShaderCoreConfig()->gpgpu_shader_registers,
      gpgpu_ctx->the_gpgpusim->g_the_gpu->getShaderCoreConfig()->max_cta_per_core);

该函数的参数为：
    1.const struct gpgpu_ptx_sim_info *kernel_info：内核信息，kernel_info_t对象包含GPU网格和线
      程块的尺寸，与内核入口点相关的function_info对象，以及在参数内存中为内核参数分配的内存。
    2.unsigned threads_per_cta：据内核函数的信息kernel获取其参数中的每个CTA（线程块）中的线程数。
    3.unsigned int warp_size：一个warp中有多少线程。
    4.unsigned int n_thread_per_shader：每个Shader Core的线程数。
    5.unsigned int gpgpu_shmem_size：由GPU配置中的-gpgpu_shmem_size设置，单个Shader Core上的共
      享存储的大小（以字节为单位），V100 GPU有96KB共享存储。
    6.unsigned int gpgpu_shader_registers：由GPU配置中的-gpgpu_shader_registers设置，单个Shader
      Core中的寄存器个数，在V100 GPU中一个Shader Core有65536个寄存器。
    7.unsigned int max_cta_per_core：由GPU配置中的-gpgpu_shader_cta设置，Shader Core中并发CTA的
      最大数量，V100 GPU每个Shader Core有32个CTA，
*/
unsigned max_cta(const struct gpgpu_ptx_sim_info *kernel_info,
                 unsigned threads_per_cta, unsigned int warp_size,
                 unsigned int n_thread_per_shader,
                 unsigned int gpgpu_shmem_size,
                 unsigned int gpgpu_shader_registers,
                 unsigned int max_cta_per_core) {
  //首先计算除以warp_size向上取整的CTA_size。
  unsigned int padded_cta_size = threads_per_cta;
  if (padded_cta_size % warp_size)
    padded_cta_size = ((padded_cta_size / warp_size) + 1) * (warp_size);
  //依据CTA_size，计算理论上每个Shader Core上的并发CTA的个数，一定是可整除的。
  unsigned int result_thread = n_thread_per_shader / padded_cta_size;

  //依据shared memory，计算理论上每个Shader Core上的并发CTA的个数。
  unsigned int result_shmem = (unsigned)-1;
  if (kernel_info->smem > 0)
    //一个线程块占用一个单独的共享存储。
    result_shmem = gpgpu_shmem_size / kernel_info->smem;
  
  //依据寄存器个数，计算理论上每个Shader Core上的并发CTA的个数。
  unsigned int result_regs = (unsigned)-1;
  if (kernel_info->regs > 0)
    //(kernel_info->regs + 3) & ~3)即为kernel_info->regs向上[4的倍数]取值。例如，如果regs等于：
    //    regs | kernel_info->regs + 3 | (kernel_info->regs + 3) & ~3)
    //     3   |   6='b0011+3='b00110  |      &'b11100='b00100=4
    //     4   |   7='b0100+3='b00111  |      &'b11100='b00100=4
    //     5   |   8='b0101+3='b01000  |      &'b11100='b01000=8
    //     6   |   9='b0110+3='b01001  |      &'b11100='b01000=8
    //     7   |  10='b0111+3='b01010  |      &'b11100='b01000=8
    //     8   |  11='b1000+3='b01011  |      &'b11100='b01000=8
    //     9   |  12='b1001+3='b01100  |      &'b11100='b01100=12
    //     10  |  13='b1010+3='b01101  |      &'b11100='b01100=12
    result_regs = gpgpu_shader_registers /
                  (padded_cta_size * ((kernel_info->regs + 3) & ~3));
  printf("padded cta size is %d and %d and %d", padded_cta_size,
         kernel_info->regs, ((kernel_info->regs + 3) & ~3));
  // Limit by CTA
  //理论上Shader Core中并发CTA的最大数量，在不受其他资源限制的情况下。
  unsigned int result_cta = max_cta_per_core;

  //gs_min2 的定义：
  //    #define gs_min2(a, b) (((a) < (b)) ? (a) : (b))
  //即二者比较，选小数。后面代码即为，选择 result_cta/result_shmem/result_regs/result_thread 四者中
  //的最小值。
  unsigned result = result_thread;
  result = gs_min2(result, result_shmem);
  result = gs_min2(result, result_regs);
  result = gs_min2(result, result_cta);

  //输出到命令行，可依据此来判断导致CTA并发不够的原因，或者是受限于CUDA代码中设置的CTA_size，或许是受限
  //于shared memory的大小，或许是受限于SM内寄存器的个数，或许是受限于硬件本身Shader Core中并发CTA的最
  //大限制数量。
  printf("GPGPU-Sim uArch: CTA/core = %u, limited by:", result);
  if (result == result_thread) printf(" threads");
  if (result == result_shmem) printf(" shmem");
  if (result == result_regs) printf(" regs");
  if (result == result_cta) printf(" cta_limit");
  printf("\n");

  return result;
}
/*!
This function simulates the CUDA code functionally, it takes a kernel_info_t
parameter which holds the data for the CUDA kernel to be executed
该函数在功能上模拟CUDA代码，它接受一个 kernel_info_t 参数，该参数保存要执行的CUDA内核的数据。只有在GPU
的配置文件里选择：
    -gpgpu_ptx_sim_mode 1
时，才会执行这一函数，以只进行功能上的模拟，不考虑性能。
*/
void cuda_sim::gpgpu_cuda_ptx_sim_main_func(kernel_info_t &kernel,
                                            bool openCL) {
  printf(
      "GPGPU-Sim: Performing Functional Simulation, executing kernel %s...\n",
      kernel.name().c_str());

  // using a shader core object for book keeping, it is not needed but as most
  // function built for performance simulation need it we use it here
  // extern gpgpu_sim *g_the_gpu;
  // before we execute, we should do PDOM analysis for functional simulation
  // scenario.
  //将Shader Core对象用于book keeping是不需要的，但由于大多数为性能模拟而构建的函数都需要它，所以我们
  //在这里使用它extern gpgpu_sim* g_the_gpu；在执行之前，应该对功能模拟场景进行PDOM（后必经结点）分析。
  
  //kernel_func_info 指向 kernel 的入口。
  function_info *kernel_func_info = kernel.entry();
  //kernel_info 指向 kernel 的PTX分析出的一些参数。例如：
  //    int lmem;                 // local memory大小
  //    int smem;                 // shared memory大小
  //    int cmem;                 // constant memory大小
  //    int gmem;                 // global memory大小
  //    int regs;                 // 寄存器数量
  //    unsigned maxthreads;      // 最大线程数
  //    unsigned ptx_version;     // PTX version
  //    unsigned sm_target;       // 目标 SM
  const struct gpgpu_ptx_sim_info *kernel_info =
      ptx_sim_kernel_info(kernel_func_info);
  //检查点，保存全局存储的状态，暂时没什么用。
  checkpoint *g_checkpoint;
  g_checkpoint = new checkpoint();

  //如果内核函数已经将PDOM（分支处理中的后必经结点检测）检测完成，执行if块代码。
  if (kernel_func_info->is_pdom_set()) {
    printf("GPGPU-Sim PTX: PDOM analysis already done for %s \n",
           kernel.name().c_str());
  } else {
    //内核函数尚未将PDOM（分支处理中的后必经结点检测）检测完成。
    printf("GPGPU-Sim PTX: finding reconvergence points for \'%s\'...\n",
           kernel.name().c_str());
    //执行PDOM检测。 
    kernel_func_info->do_pdom();
    //待上一步执行PDOM检测完成后，设置PDOM检测完成的标志pdom_done = true。
    kernel_func_info->set_pdom();
  }

  //获取一个Shader Core上可同时调度的最大线程块（或CTA或Warp Group）的数量，它或者是受限于CUDA代码中
  //设置的CTA_size，或许是受限于shared memory的大小，或许是受限于SM内寄存器的个数，或许是受限于硬件本
  //身Shader Core中并发CTA的最大限制数量。由 max_cta(...) 来计算。
  unsigned max_cta_tot = max_cta(
      kernel_info, kernel.threads_per_cta(),
      gpgpu_ctx->the_gpgpusim->g_the_gpu->getShaderCoreConfig()->warp_size,
      gpgpu_ctx->the_gpgpusim->g_the_gpu->getShaderCoreConfig()
          ->n_thread_per_shader,
      gpgpu_ctx->the_gpgpusim->g_the_gpu->getShaderCoreConfig()
          ->gpgpu_shmem_size,
      gpgpu_ctx->the_gpgpusim->g_the_gpu->getShaderCoreConfig()
          ->gpgpu_shader_registers,
      gpgpu_ctx->the_gpgpusim->g_the_gpu->getShaderCoreConfig()
          ->max_cta_per_core);
  printf("Max CTA : %d\n", max_cta_tot);

  //在GPGPU-Sim的配置文件中，-checkpoint_option一般不做配置，默认为0，后续用到再补充。
  int cp_op = gpgpu_ctx->the_gpgpusim->g_the_gpu->checkpoint_option;
  //在GPGPU-Sim的配置文件中，-checkpoint_kernel一般不做配置，默认为1，后续用到再补充。
  int cp_kernel = gpgpu_ctx->the_gpgpusim->g_the_gpu->checkpoint_kernel;
  //在GPGPU-Sim的配置文件中，-checkpoint_insn_Y一般不做配置，默认为0，后续用到再补充。
  cp_count = gpgpu_ctx->the_gpgpusim->g_the_gpu->checkpoint_insn_Y;
  //在GPGPU-Sim的配置文件中，-checkpoint_CTA_t一般不做配置，默认为0，后续用到再补充。
  cp_cta_resume = gpgpu_ctx->the_gpgpusim->g_the_gpu->checkpoint_CTA_t;
  //统计已经发射出的CTA总数。
  int cta_launched = 0;

  // we excute the kernel one CTA (Block) at the time, as synchronization
  // functions work block wise
  //我们一次执行内核的一个CTA（块），因为同步函数按块工作。
  //no_more_ctas_to_run()在abstract_hardware_model.h中定义，其函数功能为：
  //    判断kernel内核函数是否还有多余的CTA需要执行。
  //    如果标识下一个要发射的CTA在内核函数设置的全局ID的任意一维，超过了CUDA代码配置的Grid的范围，就
  //    代表kernel内核函数已经没有CTA可执行，内核函数的所有线程块均已经执行完毕。
  while (!kernel.no_more_ctas_to_run()) {
    //获取下一个要发射的CTA的ID。其ID算法如下：
    //    ID = m_next_cta.x + m_grid_dim.x * m_next_cta.y +
    //         m_grid_dim.x * m_grid_dim.y * m_next_cta.z;
    unsigned temp = kernel.get_next_cta_id_single();
    //判断是否CTA满足执行条件。
    //在GPGPU-Sim的配置文件中，cp_op = -checkpoint_option一般不做配置，默认为0。
    //在GPGPU-Sim的配置文件中，cp_kernel = -checkpoint_kernel一般不做配置，默认为1。
    //在GPGPU-Sim的配置文件中，cp_cta_resume = -checkpoint_CTA_t一般不做配置，默认为0。
    if (cp_op == 0 ||
        (cp_op == 1 && cta_launched < cp_cta_resume &&
         kernel.get_uid() == cp_kernel) ||
        kernel.get_uid() < cp_kernel)  // just fro testing
    {
      //functionalCoreSim 类在功能上执行内核函数。它使用core_t中的基本数据结构和过程。
      functionalCoreSim cta(
          &kernel, gpgpu_ctx->the_gpgpusim->g_the_gpu,
          gpgpu_ctx->the_gpgpusim->g_the_gpu->getShaderCoreConfig()->warp_size);
      //CTA执行。
      //在GPGPU-Sim的配置文件中，cp_count = -checkpoint_insn_Y一般不做配置，默认为0。
      cta.execute(cp_count, temp);

#if (CUDART_VERSION >= 5000)
      gpgpu_ctx->device_runtime->launch_all_device_kernels();
#endif
    } else {
      //CTA不满足执行条件，则由于temp标识的CTA已经在执行了，因此需要选择下一个CTA，通过指定下一个CTA
      //的编号增加来实现，但是要考虑m_grid_dim的三维结构，超过其边界时，置零并增加下一维。但是从代码看，
      //这一句没执行到。
      kernel.increment_cta_id();
    }

    //已经发射出的CTA总数增加。
    cta_launched++;
  }

  //创造检查点，保存当前存储的状态。
  //在GPGPU-Sim的配置文件中，cp_op = -checkpoint_option一般不做配置，默认为0。
  //下面的if块暂时用不到，以后用到再补充。
  if (cp_op == 1) {
    char f1name[2048];
    snprintf(f1name, 2048, "checkpoint_files/global_mem_%d.txt",
             kernel.get_uid());
    g_checkpoint->store_global_mem(
        gpgpu_ctx->the_gpgpusim->g_the_gpu->get_global_memory(), f1name,
        (char *)"%08x");
  }

  // registering this kernel as done
  //将此内核注册为已完成。这里的注册完成函数估计没什么用，官方删除了。

  // openCL kernel simulation calls don't register the kernel so we don't
  // register its exit
  //OpenCL内核模拟调用不会注册内核，因此我们不会注册它的退出。
  if (!openCL) {
    // extern stream_manager *g_stream_manager;
    gpgpu_ctx->the_gpgpusim->g_stream_manager->register_finished_kernel(
        kernel.get_uid());
  }

  //******PRINTING*******
  printf("GPGPU-Sim: Done functional simulation (%u instructions simulated).\n",
         g_ptx_sim_num_insn);
  
  //在配置文件中，"-gpgpu_ptx_instruction_classification"默认设置为0，后面用到再补充。
  if (gpgpu_ptx_instruction_classification) {
    StatDisp(g_inst_classification_stat[g_ptx_kernel_count]);
    StatDisp(g_inst_op_classification_stat[g_ptx_kernel_count]);
  }

  // time_t variables used to calculate the total simulation time
  // the start time of simulation is hold by the global variable
  // g_simulation_starttime g_simulation_starttime is initilized by
  // gpgpu_ptx_sim_init_perf() in gpgpusim_entrypoint.cc upon starting gpgpu-sim
  //下面是用于计算总模拟时间的 time_t 变量：
  //模拟的开始时间由全局变量 g_simulation_starttime 保持；启动模拟时，gpgpusim_entrypoint.cc 中的
  //gpgpu_ptx_sim_init_perf() 初始化 g_simulation_starttime。
  time_t end_time, elapsed_time, days, hrs, minutes, sec;
  end_time = time((time_t *)NULL);
  elapsed_time =
      MAX(end_time - gpgpu_ctx->the_gpgpusim->g_simulation_starttime, 1);

  // calculating and printing simulation time in terms of days, hours, minutes
  // and seconds
  //以天、小时、分钟和秒为单位计算和打印模拟时间。
  days = elapsed_time / (3600 * 24);
  hrs = elapsed_time / 3600 - 24 * days;
  minutes = elapsed_time / 60 - 60 * (hrs + 24 * days);
  sec = elapsed_time - 60 * (minutes + 60 * (hrs + 24 * days));

  fflush(stderr);
  printf(
      "\n\ngpgpu_simulation_time = %u days, %u hrs, %u min, %u sec (%u sec)\n",
      (unsigned)days, (unsigned)hrs, (unsigned)minutes, (unsigned)sec,
      (unsigned)elapsed_time);
  printf("gpgpu_simulation_rate = %u (inst/sec)\n",
         (unsigned)(g_ptx_sim_num_insn / elapsed_time));
  fflush(stdout);
}

/*
执行CTA的初始化。
*/
void functionalCoreSim::initializeCTA(unsigned ctaid_cp) {
  //CTA内活跃线程的数量。
  int ctaLiveThreads = 0;
  //m_kernel->entry() 返回m_kernel的入口函数，m_kernel_entry是 function_info 对象。
  symbol_table *symtab = m_kernel->entry()->get_symtab();

  //m_warp_count是一个CTA中的warp总数。每一个warp维护一个m_warpAtBarrier/m_liveThreadCount变量。
  for (int i = 0; i < m_warp_count; i++) {
    m_warpAtBarrier[i] = false;
    m_liveThreadCount[i] = 0;
  }

  //将所有的线程置空，以后续执行初始化。m_warp_count是一个CTA中的warp总数。m_warp_size是单个warp内
  //有多少线程。
  for (int i = 0; i < m_warp_count * m_warp_size; i++) m_thread[i] = NULL;

  // get threads for a cta
  //为一个CTA的所有线程执行初始化任务。
  //m_kernel->threads_per_cta()指的是每个CTA的线程总数，对CTA内的所有的线程遍历。
  //ptx_sim_init_thread的函数原型为上面的：
  //    unsigned ptx_sim_init_thread(kernel_info_t &kernel,
  //                                 ptx_thread_info **thread_info, 
  //                                 int sid,
  //                                 unsigned tid, 
  //                                 unsigned threads_left,
  //                                 unsigned num_threads, 
  //                                 core_t *core,
  //                                 unsigned hw_cta_id, 
  //                                 unsigned hw_warp_id,
  //                                 gpgpu_t *gpu, 
  //                                 bool isInFunctionalSimulationMode)
  //其参数分别为：
  //    1. kernel_info_t &kernel：内核函数信息类。
  //    2. ptx_thread_info **thread_info：m_thread 成员变量是SIMT Core类 shader_core_ctx 中       
  //       ptx_thread_info 的数组，维护该SIMT Core中所有活动线程的功能状态。
  //    3. int sid：这里传入的是0。???
  //    4. unsigned tid：循环体里的 i 即为线程编号 tid。
  //    5. unsigned threads_left：CTA里当前线程往后数，剩余线程的数量，由于CTA内的线程数量为 m_kernel->
  //       threads_per_cta()，且当前线程编号为 i，因此 threads_left = m_kernel->threads_per_cta()-i。
  //    6. unsigned num_threads：CTA里的线程总数，为 m_kernel->threads_per_cta()。
  //    7. core_t *core：class core_t 是内核的抽象基类，用于功能和性能模型。 shader_core_ctx（时间序模型
  //       中实现SIMT Core的类）来源于这个类，即SIMT Core微架构是由 shader.h/cc 中的 shader_core_ctx类
  //       实现的。抽象类 core_t 拥有指令执行功能上所需的最基本的数据结构和程序。 functionalCoreSim类继承
  //       自 core_t 抽象类，它包含了许多纯函数仿真以及性能仿真所使用的函数仿真数据结构和程序。
  //    8. unsigned hw_cta_id：硬件CTA的编号，可能由于每个CTA现在只支持一个线程块的运行，这里传入的是0。???
  //    9. unsigned hw_warp_id：硬件warp的编号，当前线程的编号，除以一个warp的大小即为warp的编号，该warp的
  //       编号为：i/m_warp_size。
  //    10.gpgpu_t *gpu：class gpgpu_t是实现GPU功能模拟器的顶层类。
  //    11.bool isInFunctionalSimulationMode：所有模拟包括功能/性能模拟都要求功能模拟为真。

  //对当前CTA内部的所有线程进行循环。
  for (unsigned i = 0; i < m_kernel->threads_per_cta(); i++) {
    //函数定义：ptx_sim_init_thread(kernel_info_t &kernel,
    //                             ptx_thread_info **thread_info, int sid,
    //                             unsigned tid, unsigned threads_left,
    //                             unsigned num_threads, core_t *core,
    //                             unsigned hw_cta_id, unsigned hw_warp_id,
    //                             gpgpu_t *gpu, bool isInFunctionalSimulationMode)
    //参数：
    //    sid=0：SM的index，由于这里执行功能模拟，因此SM的index不重要，可以完全将所有需要执行的线程全部放到
    //           第0号SM上。
    //    tid=i：线程的index，在这个循环里将所有需要执行的线程全部放到第0号SM上，则线程的index即为循环变量i。
    //    threads_left=m_kernel->threads_per_cta()-i：在当前线程之后剩余线程的数量。
    //    num_threads=m_kernel->threads_per_cta()。
    //    hw_cta_id=0：由于这里执行功能模拟，因此CTA的index不重要，硬件CTA的index可以始终为0。
    //    hw_warp_id=i/m_warp_size：由于全都在一个CTA内，硬件的warp的index即为i/m_warp_size。
    ptx_sim_init_thread(*m_kernel, &m_thread[i], 0, i,
                        m_kernel->threads_per_cta() - i,
                        m_kernel->threads_per_cta(), this, 0, i / m_warp_size,
                        (gpgpu_t *)m_gpu, true);
    assert(m_thread[i] != NULL && !m_thread[i]->is_done());
    char fname[2048];
    snprintf(fname, 2048, "checkpoint_files/thread_%d_0_reg.txt", i);
    //在GPGPU-Sim的配置文件中，cp_cta_resume = -checkpoint_CTA_t一般不做配置，默认为0。
    //下面的if块暂时用不到，以后用到再补充。
    if (m_gpu->gpgpu_ctx->func_sim->cp_cta_resume == 1)
      m_thread[i]->resume_reg_thread(fname, symtab);
    ctaLiveThreads++;
  }
  //创建warp。m_warp_count是一个CTA中的warp总数。
  for (int k = 0; k < m_warp_count; k++) createWarp(k);
}

/*
创建warp。
*/
void functionalCoreSim::createWarp(unsigned warpId) {
  //初始化标识warp内线程通道是否执行的掩码，处理线程分支。simt_mask_t在abstract_hardware_model.h中
  //定义：
  //    typedef std::bitset<MAX_WARP_SIZE_SIMT_STACK> simt_mask_t;
  //它的类型为std::bitset，bitset是用来储存诸多bit，这些元素可以用来表示两种状态：0或1，true或false。
  //可以很方便的用该容器快速实现状态储存。simt_mask_t的大小为 MAX_WARP_SIZE_SIMT_STACK，即一个SIMT
  //堆栈的最大的warp（一个SIMT堆栈控制的所有的warp中的线程数量最多的warp）的线程数量（一般为32）。
  simt_mask_t initialMask;
  //统计活跃线程数量。
  unsigned liveThreadsCount = 0;
  //初始化，将initialMask的所有位均置为1。
  initialMask.set();
  //对当前warp内的所有线程循环，warpID为当前warp的编号，m_warp_size指一个warp内有多少个线程。
  for (int i = warpId * m_warp_size; i < warpId * m_warp_size + m_warp_size;
       i++) {
    //m_thread 成员变量是SIMT Core类 shader_core_ctx 中 ptx_thread_info 的数组，维护该SIMT Core中
    //所有活动线程的功能状态。如果 m_thread[i] == NULL 即第 i 个线程的状态为空，则需要将 initialMask 
    //的对应位置零。
    if (m_thread[i] == NULL)
      //注意，这里 i 是所有warp中的线程统一编号时的线程编号，而initialMask只在当前warp中维护，因此重
      //置时需要注意索引。
      initialMask.reset(i - warpId * m_warp_size);
    else
      //如果第 i 个线程的状态不为空，则说明其为活跃线程，设置活跃线程数量增加1。
      liveThreadsCount++;
  }
  //当前warp中，首个线程必须是活跃的。
  assert(m_thread[warpId * m_warp_size] != NULL);
  //将该warp的起始PC值（用该warp的首个线程m_thread[warpId * m_warp_size]->get_pc()获取）线程和其线
  //程掩码用于启动SIMT堆栈。
  m_simt_stack[warpId]->launch(m_thread[warpId * m_warp_size]->get_pc(),
                               initialMask);
  char fname[2048];
  snprintf(fname, 2048, "checkpoint_files/warp_%d_0_simt.txt", warpId);

  //在GPGPU-Sim的配置文件中，cp_cta_resume = -checkpoint_CTA_t一般不做配置，默认为0。
  //下面的if块暂时用不到，以后用到再补充。
  if (m_gpu->gpgpu_ctx->func_sim->cp_cta_resume == 1) {
    unsigned pc, rpc;
    m_simt_stack[warpId]->resume(fname);
    m_simt_stack[warpId]->get_pdom_stack_top_info(&pc, &rpc);
    for (int i = warpId * m_warp_size; i < warpId * m_warp_size + m_warp_size;
         i++) {
      m_thread[i]->set_npc(pc);
      m_thread[i]->update_pc();
    }
  }
  //设置活跃线程数量全局变量。
  m_liveThreadCount[warpId] = liveThreadsCount;
}

/*
执行一个CTA。
functionalCoreSim::execute函数在gpgpu_cuda_ptx_sim_main_func主函数中的调用为：
    cta.execute(cp_count, temp); temp为下一个要发出的CTA的ID。cp_count=0。
即ctaid_cp为下一个要发射的CTA的ID。其ID算法如下：
    ID = m_next_cta.x + m_grid_dim.x * m_next_cta.y +
         m_grid_dim.x * m_grid_dim.y * m_next_cta.z;
*/
void functionalCoreSim::execute(int inst_count, unsigned ctaid_cp) {
  //在GPGPU-Sim的配置文件中，-checkpoint_insn_Y一般不做配置，默认为0，后续用到再补充。
  m_gpu->gpgpu_ctx->func_sim->cp_count = m_gpu->checkpoint_insn_Y;
  //在GPGPU-Sim的配置文件中，-checkpoint_CTA_t一般不做配置，默认为0，后续用到再补充。
  m_gpu->gpgpu_ctx->func_sim->cp_cta_resume = m_gpu->checkpoint_CTA_t;

  //functionalCoreSim::execute函数在gpgpu_cuda_ptx_sim_main_func主函数中的调用为：
  //    cta.execute(cp_count, temp); temp为下一个要发出的CTA的ID。cp_count=0。
  //即ctaid_cp为下一个要发射的CTA的ID。其ID算法如下：
  //    ID = m_next_cta.x + m_grid_dim.x * m_next_cta.y +
  //         m_grid_dim.x * m_grid_dim.y * m_next_cta.z;
  initializeCTA(ctaid_cp);

  int count = 0;
  while (true) {
    bool someOneLive = false;
    bool allAtBarrier = true;
    //m_warp_count是一个CTA中的warp总数。即下面的i相当于当前CTA内的warp ID。
    for (unsigned i = 0; i < m_warp_count; i++) {
      //执行当前CTA内的第i个warp。
      executeWarp(i, allAtBarrier, someOneLive);
      count++;
    }

    //在GPGPU-Sim的配置文件中，-checkpoint_option一般不做配置，默认为0，后续用到再补充。
    //在GPGPU-Sim的配置文件中，-checkpoint_kernel一般不做配置，默认为1，后续用到再补充。
    //在GPGPU-Sim的配置文件中，-checkpoint_CTA_t一般不做配置，默认为0，后续用到再补充。
    //下面的if块暂时用不到，以后用到再补充。
    if (inst_count > 0 && count > inst_count &&
        (m_kernel->get_uid() == m_gpu->checkpoint_kernel) &&
        (ctaid_cp >= m_gpu->checkpoint_CTA) &&
        (ctaid_cp < m_gpu->checkpoint_CTA_t) && m_gpu->checkpoint_option == 1) {
      someOneLive = false;
      break;
    }
    //m_warp_count是一个CTA中的warp总数。即下面的i相当于当前CTA内的warp ID。
    if (!someOneLive) break;
    if (allAtBarrier) {
      for (unsigned i = 0; i < m_warp_count; i++) m_warpAtBarrier[i] = false;
    }
  }

  checkpoint *g_checkpoint;
  g_checkpoint = new checkpoint();

  ptx_reg_t regval;
  regval.u64 = 123;

  unsigned ctaid = m_kernel->get_next_cta_id_single();

  //在GPGPU-Sim的配置文件中，-checkpoint_option一般不做配置，默认为0，后续用到再补充。
  //在GPGPU-Sim的配置文件中，-checkpoint_kernel一般不做配置，默认为1，后续用到再补充。
  //下面的if块暂时用不到，以后用到再补充。
  if (m_gpu->checkpoint_option == 1 &&
      (m_kernel->get_uid() == m_gpu->checkpoint_kernel) &&
      (ctaid_cp >= m_gpu->checkpoint_CTA) &&
      (ctaid_cp < m_gpu->checkpoint_CTA_t)) {
    char fname[2048];
    snprintf(fname, 2048, "checkpoint_files/shared_mem_%d.txt", ctaid - 1);
    g_checkpoint->store_global_mem(m_thread[0]->m_shared_mem, fname,
                                   (char *)"%08x");
    for (int i = 0; i < 32 * m_warp_count; i++) {
      char fname[2048];
      snprintf(fname, 2048, "checkpoint_files/thread_%d_%d_reg.txt", i,
               ctaid - 1);
      m_thread[i]->print_reg_thread(fname);
      char f1name[2048];
      snprintf(f1name, 2048, "checkpoint_files/local_mem_thread_%d_%d_reg.txt",
               i, ctaid - 1);
      g_checkpoint->store_global_mem(m_thread[i]->m_local_mem, f1name,
                                     (char *)"%08x");
      m_thread[i]->set_done();
      m_thread[i]->exitCore();
      m_thread[i]->registerExit();
    }

    for (int i = 0; i < m_warp_count; i++) {
      char fname[2048];
      snprintf(fname, 2048, "checkpoint_files/warp_%d_%d_simt.txt", i,
               ctaid - 1);
      FILE *fp = fopen(fname, "w");
      assert(fp != NULL);
      m_simt_stack[i]->print_checkpoint(fp);
      fclose(fp);
    }
  }
}

/*
执行一个warp。
functionalCoreSim::executeWarp函数在functionalCoreSim::execute中的调用为：
    executeWarp(i, allAtBarrier, someOneLive);
其中，i是一个CTA内的warp的编号，即warp ID，一个CTA内的warp从0开始编号。
allAtBarrier初始设置为True，即初始状态默认“不是所有的warp都处于屏障指令状态”。
someOneLive初始设置为False，即初始状态默认“所有的warp都暂时没有活跃的指令”。
*/
void functionalCoreSim::executeWarp(unsigned i, bool &allAtBarrier,
                                    bool &someOneLive) {
  //m_warpAtBarrier和m_liveThreadCount在cuda-sim.h中定义：
  //    每个warp活跃线程数
  //    unsigned *m_liveThreadCount;
  //    warp屏障指示器
  //    bool *m_warpAtBarrier;
  //第i个warp不处于屏障状态，且第i个warp内的活跃线程数不为0，需要执行该warp。
  if (!m_warpAtBarrier[i] && m_liveThreadCount[i] != 0) {
    //获取当前CTA内第i个warp的指令。
    warp_inst_t inst = getExecuteWarp(i);
    //执行该条指令。
    execute_warp_inst_t(inst, i);
    //判断是否是原子操作。
    if (inst.isatomic()) inst.do_atomic(true);
    //判断是否指令是屏障指令，并相应地设置对应此warp的屏障指示器。
    if (inst.op == BARRIER_OP || inst.op == MEMORY_BARRIER_OP)
      m_warpAtBarrier[i] = true;
    //更新SIMT堆栈。
    updateSIMTStack(i, &inst);
  }
  //如果该条warp的活跃线程数大于0，则设置someOneLive = true。因为一旦有一个warp的活跃线程数大于0，则
  //指示有某个warp处于活跃状态，就需要设置someOneLive = true。
  if (m_liveThreadCount[i] > 0) someOneLive = true;
  //如果该条线程的活跃线程数大于0，且不处于屏障指令，设置allAtBarrier = false。
  if (!m_warpAtBarrier[i] && m_liveThreadCount[i] > 0) allAtBarrier = false;
}

/*
依据PC值获取该PC对应指令在PTX指令中的行号。
*/
unsigned gpgpu_context::translate_pc_to_ptxlineno(unsigned pc) {
  // this function assumes that the kernel fits inside a single PTX file
  // function_info *pFunc = g_func_info; // assume that the current kernel is
  // the one in query
  const ptx_instruction *pInsn = pc_to_instruction(pc);
  unsigned ptx_line_number = pInsn->source_line();

  return ptx_line_number;
}

// ptxinfo parser
//没有使用到。
extern std::map<unsigned, const char *> get_duplicate();

static char *g_ptxinfo_kname = NULL;
static struct gpgpu_ptx_sim_info g_ptxinfo;
static std::map<unsigned, const char *> g_duplicate;
static const char *g_last_dup_type;

const char *get_ptxinfo_kname() { return g_ptxinfo_kname; }

void print_ptxinfo() {
  if (!get_ptxinfo_kname()) {
    printf("GPGPU-Sim PTX: Binary info : gmem=%u, cmem=%u\n", g_ptxinfo.gmem,
           g_ptxinfo.cmem);
  }
  if (get_ptxinfo_kname()) {
    printf(
        "GPGPU-Sim PTX: Kernel \'%s\' : regs=%u, lmem=%u, smem=%u, cmem=%u\n",
        get_ptxinfo_kname(), g_ptxinfo.regs, g_ptxinfo.lmem, g_ptxinfo.smem,
        g_ptxinfo.cmem);
  }
}

struct gpgpu_ptx_sim_info get_ptxinfo() {
  return g_ptxinfo;
}

std::map<unsigned, const char *> get_duplicate() { return g_duplicate; }

void ptxinfo_linenum(unsigned linenum) {
  g_duplicate[linenum] = g_last_dup_type;
}

void ptxinfo_dup_type(const char *dup_type) { g_last_dup_type = dup_type; }

void ptxinfo_function(const char *fname) {
  clear_ptxinfo();
  g_ptxinfo_kname = strdup(fname);
}

void ptxinfo_regs(unsigned nregs) { g_ptxinfo.regs = nregs; }

void ptxinfo_lmem(unsigned declared, unsigned system) {
  g_ptxinfo.lmem = declared + system;
}

void ptxinfo_gmem(unsigned declared, unsigned system) {
  g_ptxinfo.gmem = declared + system;
}

void ptxinfo_smem(unsigned declared, unsigned system) {
  g_ptxinfo.smem = declared + system;
}

void ptxinfo_cmem(unsigned nbytes, unsigned bank) { g_ptxinfo.cmem += nbytes; }

void clear_ptxinfo() {
  free(g_ptxinfo_kname);
  g_ptxinfo_kname = NULL;
  g_ptxinfo.regs = 0;
  g_ptxinfo.lmem = 0;
  g_ptxinfo.smem = 0;
  g_ptxinfo.cmem = 0;
  g_ptxinfo.gmem = 0;
  g_ptxinfo.ptx_version = 0;
  g_ptxinfo.sm_target = 0;
}

void ptxinfo_opencl_addinfo(std::map<std::string, function_info *> &kernels) {
  if (!g_ptxinfo_kname) {
    printf("GPGPU-Sim PTX: Binary info : gmem=%u, cmem=%u\n", g_ptxinfo.gmem,
           g_ptxinfo.cmem);
    clear_ptxinfo();
    return;
  }

  if (!strcmp("__cuda_dummy_entry__", g_ptxinfo_kname)) {
    // this string produced by ptxas for empty ptx files (e.g., bandwidth test)
    clear_ptxinfo();
    return;
  }
  std::map<std::string, function_info *>::iterator k =
      kernels.find(g_ptxinfo_kname);
  if (k == kernels.end()) {
    printf("GPGPU-Sim PTX: ERROR ** implementation for '%s' not found.\n",
           g_ptxinfo_kname);
    abort();
  } else {
    printf(
        "GPGPU-Sim PTX: Kernel \'%s\' : regs=%u, lmem=%u, smem=%u, cmem=%u\n",
        g_ptxinfo_kname, g_ptxinfo.regs, g_ptxinfo.lmem, g_ptxinfo.smem,
        g_ptxinfo.cmem);
    function_info *finfo = k->second;
    assert(finfo != NULL);
    finfo->set_kernel_info(g_ptxinfo);
  }
  clear_ptxinfo();
}

struct rec_pts cuda_sim::find_reconvergence_points(function_info *finfo) {
  rec_pts tmp;
  std::map<function_info *, rec_pts>::iterator r = g_rpts.find(finfo);

  if (r == g_rpts.end()) {
    int num_recon = finfo->get_num_reconvergence_pairs();

    gpgpu_recon_t *kernel_recon_points =
        (struct gpgpu_recon_t *)calloc(num_recon, sizeof(struct gpgpu_recon_t));
    finfo->get_reconvergence_pairs(kernel_recon_points);
    printf("GPGPU-Sim PTX: reconvergence points for %s...\n",
           finfo->get_name().c_str());
    for (int i = 0; i < num_recon; i++) {
      printf("GPGPU-Sim PTX: %2u (potential) branch divergence @ ", i + 1);
      kernel_recon_points[i].source_inst->print_insn();
      printf("\n");
      printf("GPGPU-Sim PTX:    immediate post dominator      @ ");
      if (kernel_recon_points[i].target_inst)
        kernel_recon_points[i].target_inst->print_insn();
      printf("\n");
    }
    printf("GPGPU-Sim PTX: ... end of reconvergence points for %s\n",
           finfo->get_name().c_str());

    tmp.s_kernel_recon_points = kernel_recon_points;
    tmp.s_num_recon = num_recon;
    g_rpts[finfo] = tmp;
  } else {
    tmp = r->second;
  }
  return tmp;
}

address_type get_return_pc(void *thd) {
  // function call return
  ptx_thread_info *the_thread = (ptx_thread_info *)thd;
  assert(the_thread != NULL);
  return the_thread->get_return_PC();
}

address_type cuda_sim::get_converge_point(address_type pc) {
  // the branch could encode the reconvergence point and/or a bit that indicates
  // the reconvergence point is the return PC on the call stack in the case the
  // branch has no immediate postdominator in the function (i.e., due to
  // multiple return points).

  std::map<unsigned, function_info *>::iterator f = g_pc_to_finfo.find(pc);
  assert(f != g_pc_to_finfo.end());
  function_info *finfo = f->second;
  rec_pts tmp = find_reconvergence_points(finfo);

  int i = 0;
  for (; i < tmp.s_num_recon; ++i) {
    if (tmp.s_kernel_recon_points[i].source_pc == pc) {
      if (tmp.s_kernel_recon_points[i].target_pc == (unsigned)-2) {
        return RECONVERGE_RETURN_PC;
      } else {
        return tmp.s_kernel_recon_points[i].target_pc;
      }
    }
  }
  return NO_BRANCH_DIVERGENCE;
}

/*
退出执行所有的warp。m_warp_count是一个CTA中的warp总数。m_warp_size是一个warp内有多少线程。warp_id
在这里貌似没什么用。
*/
void functionalCoreSim::warp_exit(unsigned warp_id) {
  for (int i = 0; i < m_warp_count * m_warp_size; i++) {
    if (m_thread[i] != NULL) {
      //挂起所有线程。
      m_thread[i]->m_cta_info->register_deleted_thread(m_thread[i]);
      delete m_thread[i];
    }
  }
}
