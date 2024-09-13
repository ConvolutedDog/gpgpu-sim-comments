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

#ifndef MEM_FETCH_H
#define MEM_FETCH_H

#include <bitset>
#include "../abstract_hardware_model.h"
#include "addrdec.h"

/*
mem_fetch内存访问请求的类型，它被定义为：
  enum mf_type {
    READ_REQUEST = 0,  //读请求
    WRITE_REQUEST,     //写请求
    READ_REPLY,        //读响应  // send to shader
    WRITE_ACK          //写确认
  };
*/
enum mf_type {
  READ_REQUEST = 0,
  WRITE_REQUEST,
  READ_REPLY,  // send to shader
  WRITE_ACK
};

#define MF_TUP_BEGIN(X) enum X {
#define MF_TUP(X) X
#define MF_TUP_END(X) \
  }                   \
  ;
#include "mem_fetch_status.tup"
#undef MF_TUP_BEGIN
#undef MF_TUP
#undef MF_TUP_END

class memory_config;

/*
mem_fetch定义了一个模拟内存请求的通信结构。更像是一个内存请求的行为。
*/
class mem_fetch {
 public:
  //构造函数。
  //mem_access_t 包含时序模拟器中每个内存访问的信息。该类包含内存访问的类型、请求的地址、数据的大小
  //以及访问内存的warp的活动掩码等信息。该类被用作mem_fetch类的参数之一，该类基本上为每个内存访问实
  //例化。这个类是用于两个不同级别的内存之间的接口，并将两者互连。
  mem_fetch(const mem_access_t &access, const warp_inst_t *inst,
            unsigned long long streamID, unsigned ctrl_size, unsigned wid,
            unsigned sid, unsigned tpc, const memory_config *config,
            unsigned long long cycle, mem_fetch *original_mf = NULL,
            mem_fetch *original_wr_mf = NULL);
  ~mem_fetch();
  //设置内存请求的状态，和状态变化所处的时钟周期。
  void set_status(enum mem_fetch_status status, unsigned long long cycle);
  //设置内存访问请求响应的类型，内存访问请求中包含四种类型：读请求、写请求、读响应、写确认。这里是设
  //置读响应或者是写确认。
  void set_reply() {
    //在V100中，L1 cache的m_write_policy为WRITE_THROUGH，实际上L1_WRBK_ACC也不会用到。
    //在V100中，当L2 cache写不命中时，采取lazy_fetch_on_read策略，当找到一个cache block
    //逐出时，如果这个cache block是被MODIFIED，则需要将这个cache block写回到下一级存储，
    //因此会产生L2_WRBK_ACC访问，这个访问就是为了写回被逐出的MODIFIED cache block。
    assert(m_access.get_type() != L1_WRBK_ACC &&
           m_access.get_type() != L2_WRBK_ACC);
    //如果内存访问请求的类型是读请求，将其设置为读响应。
    if (m_type == READ_REQUEST) {
      assert(!get_is_write());
      m_type = READ_REPLY;
    //如果内存访问请求的类型是写请求，将其设置为写确认。
    } else if (m_type == WRITE_REQUEST) {
      assert(get_is_write());
      m_type = WRITE_ACK;
    }
  }
  //执行原子操作。
  void do_atomic();

  void print(FILE *fp, bool print_inst = true) const;

  const addrdec_t &get_tlx_addr() const { return m_raw_addr; }
  void set_chip(unsigned chip_id) { m_raw_addr.chip = chip_id; }
  void set_partition(unsigned sub_partition_id) {
    m_raw_addr.sub_partition = sub_partition_id;
  }
  unsigned get_data_size() const { return m_data_size; }
  void set_data_size(unsigned size) { m_data_size = size; }
  //对于写确认，数据包只有控制元数据。当前内存访问请求是互连网络传给SIMT Core集群的写确认，数据包仅
  //包含控制元数据（metadata）。返回该控制元数据的大小。
  unsigned get_ctrl_size() const { return m_ctrl_size; }
  unsigned size() const { return m_data_size + m_ctrl_size; }
  bool is_write() { return m_access.is_write(); }
  void set_addr(new_addr_type addr) { m_access.set_addr(addr); }
  new_addr_type get_addr() const { return m_access.get_addr(); }
  unsigned get_access_size() const { return m_access.get_size(); }
  new_addr_type get_partition_addr() const { return m_partition_addr; }
  unsigned get_sub_partition_id() const { return m_raw_addr.sub_partition; }
  //当前内存访问请求是互连网络传给SIMT Core集群的写确认，数据包仅包含控制元数据（metadata）。
  bool get_is_write() const { return m_access.is_write(); }
  unsigned get_request_uid() const { return m_request_uid; }
  //获取内存访问请求源的SIMT Core的ID。
  unsigned get_sid() const { return m_sid; }
  //获取当前内存访问请求的目的端SIMT Core集群的ID。
  unsigned get_tpc() const { return m_tpc; }
  unsigned get_wid() const { return m_wid; }
  bool istexture() const;
  bool isconst() const;
  //返回对存储器进行的访存类型。
  enum mf_type get_type() const { return m_type; }
  bool isatomic() const;

  void set_return_timestamp(unsigned t) { m_timestamp2 = t; }
  void set_icnt_receive_time(unsigned t) { m_icnt_receive_time = t; }
  unsigned get_timestamp() const { return m_timestamp; }
  unsigned get_return_timestamp() const { return m_timestamp2; }
  unsigned get_icnt_receive_time() const { return m_icnt_receive_time; }
  unsigned long long get_streamID() const { return m_streamID; }
  //m_access是mem_access_t对象，mem_access_t包含时序模拟器中每个内存访问的信息。该类包含内存访问
  //的类型、请求的地址、数据的大小以及访问内存的warp的活动掩码等信息。该类被用作mem_fetch类的参数之
  //一，该类基本上为每个内存访问实例化。这个类是用于两个不同级别的内存之间的接口，并将两者互连。
  //m_access.get_type()返回对存储器进行的访存类型mem_access_type：
  //mem_access_type定义了在时序模拟器中对不同类型的存储器进行不同的访存类型：
  //    MA_TUP(GLOBAL_ACC_R),        从global memory读
  //    MA_TUP(LOCAL_ACC_R),         从local memory读
  //    MA_TUP(CONST_ACC_R),         从常量缓存读
  //    MA_TUP(TEXTURE_ACC_R),       从纹理缓存读
  //    MA_TUP(GLOBAL_ACC_W),        向global memory写
  //    MA_TUP(LOCAL_ACC_W),         向local memory写
  //在V100中，L1 cache的m_write_policy为WRITE_THROUGH，实际上L1_WRBK_ACC也不会用到：
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
  enum mem_access_type get_access_type() const { return m_access.get_type(); }
  const active_mask_t &get_access_warp_mask() const {
    return m_access.get_warp_mask();
  }
  mem_access_byte_mask_t get_access_byte_mask() const {
    return m_access.get_byte_mask();
  }
  mem_access_sector_mask_t get_access_sector_mask() const {
    return m_access.get_sector_mask();
  }

  address_type get_pc() const { return m_inst.empty() ? -1 : m_inst.pc; }
  const warp_inst_t &get_inst() { return m_inst; }
  enum mem_fetch_status get_status() const { return m_status; }

  const memory_config *get_mem_config() { return m_mem_config; }

  unsigned get_num_flits(bool simt_to_mem);

  mem_fetch *get_original_mf() { return original_mf; }
  mem_fetch *get_original_wr_mf() { return original_wr_mf; }

 private:
  // request source information
  //以下是请求源信息。
  //请求的唯一的ID，mem_fetch对象被创建时，赋值为sm_next_mf_request_uid。这个值被初始化为1，每当下
  //一个mem_fetch对象创建时，这个值递增加1。
  unsigned m_request_uid;
  //m_sid表示内存访问请求源的SIMT Core的ID。
  unsigned m_sid;
  //当前内存访问请求的目的端SIMT Core集群的ID。
  unsigned m_tpc;
  //请求的warp ID。
  unsigned m_wid;

  // where is this request now?
  //mem_fetch_status定义了内存请求的状态。m_status变量保存了请求所处的状态。
  enum mem_fetch_status m_status;
  //内存请求状态变化的时刻，变化所处的时钟周期。
  unsigned long long m_status_change;

  // request type, address, size, mask
  //mem_access_t包含时序模拟器中每个内存访问的信息。该类包含内存访问的类型、请求的地址、数据的大小以
  //及访问内存的warp的活动掩码等信息。该类被用作mem_fetch类的参数之一，该类基本上为每个内存访问实例
  //化。这个类是用于两个不同级别的内存之间的接口，并将两者互连。
  mem_access_t m_access;
  //写请求的数据大小，以字节为单位。
  unsigned m_data_size;  // how much data is being written
  //对于写确认，数据包只有控制元数据。当前内存访问请求是互连网络传给SIMT Core集群的写确认，数据包仅
  //包含控制元数据（metadata）。该控制元数据的大小。所有这些元数据在硬件中的大小（不一定与mem_fetch
  //的实际大小匹配）。
  unsigned
      m_ctrl_size;  // how big would all this meta data be in hardware (does not
                    // necessarily match actual size of mem_fetch)
  //DRAM分区内的线性物理地址（partition bank select bits被挤出）。
  new_addr_type
      m_partition_addr;  // linear physical address *within* dram partition
                         // (partition bank select bits squeezed out)
  //原始物理地址（即解码的DRAM chip-row-bank-column地址）。
  addrdec_t m_raw_addr;  // raw physical address (i.e., decoded DRAM
                         // chip-row-bank-column address)
  //mem_fetch内存访问请求的类型，它被定义为：
  //  enum mf_type {
  //    READ_REQUEST = 0,  //读请求
  //    WRITE_REQUEST,     //写请求
  //    READ_REPLY,        //读响应  // send to shader
  //    WRITE_ACK          //写确认
  //  };
  enum mf_type m_type;

  // statistics
  //在mem_fetch对象创建时设置为gpu_sim_cycle+gpu_tot_sim_cycle。
  unsigned
      m_timestamp;  // set to gpu_sim_cycle+gpu_tot_sim_cycle at struct creation
  //当推到icnt到Shader Core上时，设置为gpu_sim_cycle+gpu_tot_sim_cycle；仅用于读取。
  unsigned m_timestamp2;  // set to gpu_sim_cycle+gpu_tot_sim_cycle when pushed
                          // onto icnt to shader; only used for reads
  //启用固定的icnt延迟模式时，设置为gpu_sim_cycle+interconnect_latency。
  unsigned m_icnt_receive_time;  // set to gpu_sim_cycle + interconnect_latency
                                 // when fixed icnt latency mode is enabled

  // requesting instruction (put last so mem_fetch prints nicer in gdb)
  //内存请求的指令（放在最后，以便mem_fetch在gdb中打印得更好）。
  warp_inst_t m_inst;
  //每次内存访问请求都有唯一的ID，这个值被初始化为1，每当下一个mem_fetch对象创建时，这个值递增加1。
  unsigned long long m_streamID;

  static unsigned sm_next_mf_request_uid;
  //内存的配置信息，从gpu-sim.cc中读取gpgpusim.config参数。
  const memory_config *m_mem_config;
  //以字节为单位指定flit_size。这用于根据传递给icnt_push()函数的数据包大小来确定每个数据包的微片数。
  unsigned icnt_flit_size;

  //该指针是在L2 cache中将请求划分为sector requests时设置（如果req size > L2 sector size），此
  //指针指向原始请求。因为实际上只是在性能模拟过程中把mf划分为几个sector，但当mf返回时，需要将这些
  //sector合并为一个mf，这里为了简便，就给每个分割的mf保留一个原始mf的指针。
  mem_fetch
      *original_mf;  // this pointer is set up when a request is divided into
                     // sector requests at L2 cache (if the req size > L2 sector
                     // size), so the pointer refers to the original request
  //当使用fetch-on-write策略时，该指针指向原始写请求。
  mem_fetch *original_wr_mf;  // this pointer refers to the original write req,
                              // when fetch-on-write policy is used
};

#endif
