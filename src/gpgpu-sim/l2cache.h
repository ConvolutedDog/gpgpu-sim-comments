// Copyright (c) 2009-2021, Tor M. Aamodt, Vijay Kandiah, Nikos Hardavellas,
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

#ifndef MC_PARTITION_INCLUDED
#define MC_PARTITION_INCLUDED

#include "../abstract_hardware_model.h"
#include "dram.h"

#include <list>
#include <queue>

class mem_fetch;

/*
用于由内存分区和L2 Cache产生mem_fetch对象（内存请求）。
*/
class partition_mf_allocator : public mem_fetch_allocator {
 public:
  partition_mf_allocator(const memory_config *config) {
    m_memory_config = config;
  }
  virtual mem_fetch *alloc(const class warp_inst_t &inst,
                           const mem_access_t &access,
                           unsigned long long cycle) const {
    abort();
    return NULL;
  }
  virtual mem_fetch *alloc(new_addr_type addr, mem_access_type type,
                           unsigned size, bool wr, unsigned long long cycle,
                           unsigned long long streamID) const;
  virtual mem_fetch *alloc(new_addr_type addr, mem_access_type type,
                           const active_mask_t &active_mask,
                           const mem_access_byte_mask_t &byte_mask,
                           const mem_access_sector_mask_t &sector_mask,
                           unsigned size, bool wr, unsigned long long cycle,
                           unsigned wid, unsigned sid, unsigned tpc,
                           mem_fetch *original_mf,
                           unsigned long long streamID) const;

 private:
  const memory_config *m_memory_config;
};

// Memory partition unit contains all the units assolcated with a single DRAM
// channel.
// - It arbitrates the DRAM channel among multiple sub partitions.
// - It does not connect directly with the interconnection network.
class memory_partition_unit {
 public:
  // 一个memory partition unit实际上是一个存储控制器（DRAM Channel），在V100中m_n_mem配置为32。每个内存
  // 控制器又包含多个子分区，V100中m_n_sub_partition_per_memory_channel配置为2。m_n_mem_sub_partition
  // 实际上是整个GPU上的子分区个数=m_n_mem*m_n_sub_partition_per_memory_channel=64。因此在GPU创建时，对
  // 内存分区和内存子分区有：
  //   m_memory_partition_unit =
  //       new memory_partition_unit *[m_memory_config->m_n_mem];
  //   m_memory_sub_partition =
  //       new memory_sub_partition *[m_memory_config->m_n_mem_sub_partition];
  //   for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
  //     m_memory_partition_unit[i] = // 这里的i是partition_id
  //         new memory_partition_unit(i, m_memory_config, m_memory_stats, this);
  //     for (unsigned p = 0;
  //          p < m_memory_config->m_n_sub_partition_per_memory_channel; p++) {
  //       unsigned submpid =
  //           i * m_memory_config->m_n_sub_partition_per_memory_channel + p;
  //       m_memory_sub_partition[submpid] = // 这里的submpid是sub_partition_id
  //           m_memory_partition_unit[i]->get_sub_partition(p);
  //     }
  //   }
  memory_partition_unit(unsigned partition_id, const memory_config *config,
                        class memory_stats_t *stats, class gpgpu_sim *gpu);
  ~memory_partition_unit();

  bool busy() const;

  void cache_cycle(unsigned cycle);
  void dram_cycle();
  void simple_dram_model_cycle();

  void set_done(mem_fetch *mf);

  void visualizer_print(gzFile visualizer_file) const;
  void print_stat(FILE *fp) { m_dram->print_stat(fp); }
  void visualize() const { m_dram->visualize(); }
  void print(FILE *fp) const;
  void handle_memcpy_to_gpu(size_t dst_start_addr, unsigned subpart_id,
                            mem_access_sector_mask_t mask);

  class memory_sub_partition *get_sub_partition(int sub_partition_id) {
    return m_sub_partition[sub_partition_id];
  }

  // Power model
  void set_dram_power_stats(unsigned &n_cmd, unsigned &n_activity,
                            unsigned &n_nop, unsigned &n_act, unsigned &n_pre,
                            unsigned &n_rd, unsigned &n_wr, unsigned &n_wr_WB,
                            unsigned &n_req) const;

  int global_sub_partition_id_to_local_id(int global_sub_partition_id) const;

  unsigned get_mpid() const { return m_id; }

  class gpgpu_sim *get_mgpu() const {
    return m_gpu;
  }

 private:
  unsigned m_id;
  const memory_config *m_config;
  class memory_stats_t *m_stats;
  class memory_sub_partition **m_sub_partition;
  class dram_t *m_dram;

  class arbitration_metadata {
   public:
    arbitration_metadata(const memory_config *config);

    // check if a subpartition still has credit
    bool has_credits(int inner_sub_partition_id) const;
    // borrow a credit for a subpartition
    void borrow_credit(int inner_sub_partition_id);
    // return a credit from a subpartition
    void return_credit(int inner_sub_partition_id);

    // return the last subpartition that borrowed credit
    int last_borrower() const { return m_last_borrower; }

    void print(FILE *fp) const;

   private:
    // id of the last subpartition that borrowed credit
    int m_last_borrower;

    int m_shared_credit_limit;
    int m_private_credit_limit;

    // credits borrowed by the subpartitions
    std::vector<int> m_private_credit;
    int m_shared_credit;
  };
  arbitration_metadata m_arbitration_metadata;

  // determine wheither a given subpartition can issue to DRAM
  bool can_issue_to_dram(int inner_sub_partition_id);

  // model DRAM access scheduler latency (fixed latency between L2 and DRAM)
  struct dram_delay_t {
    unsigned long long ready_cycle;
    class mem_fetch *req;
  };
  std::list<dram_delay_t> m_dram_latency_queue;

  class gpgpu_sim *m_gpu;
};


/*
L1数据高速缓存在该高速缓存中维护全局存储器地址空间的子集。在一些架构中，L1高速缓存仅包含不被内核修改的位置，
这有助于避免由于GPU上缺乏高速缓存一致性而导致的复杂性。从程序员的角度来看，访问全局存储器时的关键考虑是由给
定线程束的不同线程访问的存储器位置相对于彼此的关系。如果线程束中的所有线程访问落在单个L1数据高速缓存块内的位
置并且该块不存在该高速缓存中，则仅需要向较低级别的高速缓存发送单个请求。这样的访问被称为“合并的”。如果线程束
内的线程访问不同的高速缓存块，则需要生成多个存储器访问。这样的访问被称为未合并的。程序员试图避免存储体冲突和
未合并的访问，但为了简化编程，硬件允许两者。

首先考虑如何处理共享存储器访问，然后考虑合并的高速缓存命中，最后考虑高速缓存未命中和未合并的访问。对所有情况，
存储器访问请求首先从指令流水线内的加载/存储单元发送到L1高速缓存。存储器访问请求由一组存储器地址组成，一个线
程束中的每个线程对应一个存储器地址以及操作类型。

对于共享存储器存取，仲裁器确定线程束内的请求地址是否将引起存储体冲突。如果所请求的地址将导致一个或多个存储体
冲突，则仲裁器将请求分成两个部分。第一部分包括线程束中不具有存储体冲突的线程子集的地址。仲裁器接受原始请求的
这一部分，以供该高速缓存进一步处理。第二部分包含与第一部分中的地址导致存储体冲突的那些地址。原始请求的这一部
分被返回到指令流水线，并且必须再次执行，该后续执行被称为"reply"。在存储原始共享存储器请求的reply部分时存在折
衷。虽然可以通过reply来自指令缓冲器的存储器访问指令来节省面积，但这在访问大寄存器文件时消耗能量。能量效率的
更好替代方案可以是提供有限的缓冲以用于reply加载/存储单元中的存储器访问指令，并且避免在该缓冲器中的空闲空间耗
尽时调度来自指令缓冲器的存储器访问操作。在考虑reply请求会发生什么之前，让我们考虑如何处理存储器请求的接受部分。

共享存储器请求的接受部分绕过标签单元内的标签查找（主要是查询数据是否在L1 Cache中），因为共享存储器被直接映射。
当接受共享存储器加载请求时，仲裁器将写回事件调度到指令流水线内的寄存器文件，因为在没有存储体冲突的情况下直接
映射存储器查找的等待时间是恒定的。标签单元确定每个线程的请求映射到哪个Bank，以便控制地址交叉开关，地址交叉开
关将地址分配到数据阵列内的各个存储体。数据阵列内的每个存储体是32位宽的，并且具有其自己的解码器，允许独立访问
每个存储体中的不同行。数据经由数据交叉开关返回到适当线程的通道以存储在寄存器堆中。只有对应于线程束中的活跃线
程的通道将值写入寄存器文件。

共享存储器请求的reply部分可以在先前接受的部分之后的周期访问L1高速缓存仲裁器。如果该reply部分再次遇到存储体冲
突，则将其进一步细分为接受和reply部分。 <== bank conflict consumes a lot of cycles

接下来，让我们考虑如何处理全局内存空间的加载。由于只有全局存储器空间的子集被缓存在L1高速缓存中，标签单元将需
要检查数据是否存在于该高速缓存中。虽然数据阵列被高度存储以使得能够由各个线程束灵活地访问共享存储器，但对全局
存储器的访问被限制为每个周期单个高速缓存块。该限制有助于减少相对于高速缓存数据量的标签存储开销，并且也是标准
DRAM芯片的标准接口的结果。在费米和开普勒中，L1缓存块大小为128字节，在麦克斯韦和Pascal中，进一步分为四个32字
节扇区。32字节扇区大小对应于可在单次存取中从新近图形DRAM芯片读取的数据的最小大小（例如，GDDR5）。每个128字
节高速缓存块由32个存储体中的每一个中相同行的32位条目组成。

加载/存储单元计算存储器地址并应用合并规则以将线程束的存储器访问分解成单独的合并的访问，这些合并的访问然后被馈
送到仲裁器中。如果没有足够的资源可用，则仲裁器可以拒绝请求。例如，如果访问映射到该高速缓存组中的所有ways都忙
碌，或者在pending request table中没有空闲条目，这将在下面描述。假设有足够的资源可用于处理未命中，仲裁器请求
指令流水线在对应于高速缓存命中的未来固定数量的周期中调度回写到寄存器文件。并行地，仲裁器还请求标签单元检查访
问实际上是否导致高速缓存命中或未命中。在高速缓存命中的情况下，在所有存储体中访问数据阵列的适当行，并且数据被
返回到指令流水线中的寄存器堆。如在共享存储器存取的情况下，仅更新对应于作用中线程的寄存器通道。

当访问标签单元时，如果确定请求触发高速缓存未命中，则仲裁器通知加载/存储单元它必须reply该请求，并且并行地它将
请求信息发送到pending request table（PRT）。pending request table提供的功能与CPU高速缓冲存储器系统中的传
统缺失状态保持寄存器所支持的功能没有什么不同。在NVIDIA专利与图4.1所示的L1缓存体系结构相关的版本看起来有点类
似于传统的MSHR。数据缓存的传统MSHR包含缓存未命中的块地址以及块偏移量和相关寄存器的信息，当块被填充该高速缓存
中时，需要写入。通过记录多个块偏移和寄存器来支持对同一块的多个未命中。图4.1中的PRT支持将两个请求合并到同一块，
并记录信息，以通知指令流水线延迟存储器访问以reply。

![图4.1](../../comments-figs/UnifiedL1DataCacheAndSharedMemory.png)

图4.1所示的L1数据缓存是虚拟索引和虚拟标记的。当与大多采用虚拟索引/物理标记的L1数据高速缓存的现代CPU微架构相
比时，这可能是令人惊讶的。CPU使用这种组织来避免在上下文切换上刷新L1数据缓存的开销。虽然GPU在线程束发出的每个
周期有效地执行上下文切换，但线程束是同一应用的一部分。基于页面的虚拟存储器在GPU内仍然是有利的，即使它被限制为
一次运行单个OS应用程序，因为它有助于简化存储器分配并减少存储器碎片。在PRT中分配条目之后，存储器请求被转发到存
储器管理单元（MMU）以用于虚拟到物理地址转换，并且从那里通过交叉互连转发到适当的存储器分区单元。正如将在4.3节
中扩展的那样，内存分区单元包含一个L2高速缓存库沿着一个内存访问调度器。沿着关于要访问哪个物理存储器地址和要读
取多少字节的信息，存储器请求包含"subid"，当存储器请求返回到核时，该"subid"可以用于查找PRT中包含关于请求的信
息的条目。

一旦针对加载的存储器请求响应被返回到核，它就被MMU传递到填充单元。填充单元进而使用存储器请求中的subid字段来在
PRT中查找关于请求的信息。这包括可以由填充单元经由仲裁器传递到加载/存储单元以重新调度加载的信息，然后通过在高
速缓存中的行已经被放置到数据阵列中之后锁定高速缓存中的行来保证加载命中高速缓存。

图4.1中的L1数据缓存可以支持直写和回写策略。因此，对全局存储器的存储指令（写入）可以以若干方式处理。写入的特定
内存空间确定写入是被视为直写还是回写。在许多GPGPU应用程序中对全局存储器的访问可以预期具有非常差的时间局部性，
因为通常内核以线程在退出之前将数据写出到大阵列的方式编写。对于这样的访问，无写分配的直写策略可能有意义。相比之
下，用于溢出寄存器到堆栈的本地存储器写入可以显示出良好的时间局部性，其中随后的加载证明具有写入分配策略的回写是
合理的。

要写入共享存储器或全局存储器的数据首先被放置在写入数据缓冲器（WDB）中。对于非合并访问或当某些线程被屏蔽时，仅写
入缓存块的一部分。如果块存在该高速缓存中，则可以经由数据交叉开关将数据写入数据阵列。如果数据不存在该高速缓存中，
则必须首先从L2高速缓存或DRAM存储器读取块。如果完全填充高速缓存块的合并写入使高速缓存中的任何陈旧数据的标签无效，
则合并写入可以绕过该高速缓存。

注意图4.1中描述该高速缓存组织不支持缓存一致性。例如，假设在SM1上执行的线程读取存储器位置A并且该值被存储在SM1的
L1数据高速缓存中，然后在SM2上执行的另一线程写入存储器位置A。如果SM1上的任何线程随后在存储器位置A被从SM1的L1数
据高速缓存逐出之前读取存储器位置A，则其将获得旧值而不是新值。为了避免这个问题，从Kepler开始的NVIDIA GPU只允许
寄存器溢出和堆栈数据或只读全局内存数据的本地内存访问被放置在L1数据缓存中。最近的研究已经探索了如何在GPU上启用相
干的L1数据缓存以及对明确定义的GPU存储器一致性模型的需要。

# 32 sets, each 128 bytes 24-way for each memory sub partition (96 KB per memory sub partition). This gives us 6MB L2 cache.
# With previous defined configs: -gpgpu_n_mem 32 and -gpgpu_n_sub_partition_per_mchannel 2, we can calculate that the num of
# sub partition of memories is gpgpu_n_mem*gpgpu_n_sub_partition_per_mchannel = 64. So, with 96 KB per memory sub partition,
# we get 6MB (64*96KB) L2 cache. To match the DRAM atom size of 32 bytes in GDDR5, each cache line inside the slice has four
# 32-byte sectors.
*/
class memory_sub_partition {
 public:
  // 一个memory partition unit实际上是一个存储控制器（DRAM Channel），在V100中m_n_mem配置为32。每个内存
  // 控制器又包含多个子分区，V100中m_n_sub_partition_per_memory_channel配置为2。m_n_mem_sub_partition
  // 实际上是整个GPU上的子分区个数=m_n_mem*m_n_sub_partition_per_memory_channel=64。因此在GPU创建时，对
  // 内存分区和内存子分区有：
  //   m_memory_partition_unit =
  //       new memory_partition_unit *[m_memory_config->m_n_mem];
  //   m_memory_sub_partition =
  //       new memory_sub_partition *[m_memory_config->m_n_mem_sub_partition];
  //   for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
  //     m_memory_partition_unit[i] = // 这里的i是partition_id
  //         new memory_partition_unit(i, m_memory_config, m_memory_stats, this);
  //     for (unsigned p = 0;
  //          p < m_memory_config->m_n_sub_partition_per_memory_channel; p++) {
  //       unsigned submpid =
  //           i * m_memory_config->m_n_sub_partition_per_memory_channel + p;
  //       m_memory_sub_partition[submpid] = // 这里的submpid是sub_partition_id
  //           m_memory_partition_unit[i]->get_sub_partition(p);
  //     }
  //   }
  memory_sub_partition(unsigned sub_partition_id, const memory_config *config,
                       class memory_stats_t *stats, class gpgpu_sim *gpu);
  ~memory_sub_partition();
  //m_sub_partition[dest_spid]->get_id()返回内存子分区的全局ID。
  unsigned get_id() const { return m_id; }

  bool busy() const;

  void cache_cycle(unsigned cycle);

  bool full() const;
  bool full(unsigned size) const;
  void push(class mem_fetch *mf, unsigned long long clock_cycle);
  class mem_fetch *pop();
  class mem_fetch *top();
  void set_done(mem_fetch *mf);

  unsigned flushL2();
  unsigned invalidateL2();

  // interface to L2_dram_queue
  bool L2_dram_queue_empty() const;
  class mem_fetch *L2_dram_queue_top() const;
  void L2_dram_queue_pop();

  // interface to dram_L2_queue
  bool dram_L2_queue_full() const;
  void dram_L2_queue_push(class mem_fetch *mf);

  void visualizer_print(gzFile visualizer_file);
  void print_cache_stat(unsigned &accesses, unsigned &misses) const;
  void print(FILE *fp) const;

  void accumulate_L2cache_stats(class cache_stats &l2_stats) const;
  void get_L2cache_sub_stats(struct cache_sub_stats &css) const;

  // Support for getting per-window L2 stats for AerialVision
  void get_L2cache_sub_stats_pw(struct cache_sub_stats_pw &css) const;
  void clear_L2cache_stats_pw();

  void force_l2_tag_update(new_addr_type addr, unsigned time,
                           mem_access_sector_mask_t mask) {
    m_L2cache->force_tag_access(addr, m_memcpy_cycle_offset + time, mask);
    m_memcpy_cycle_offset += 1;
  }

 private:
  // data
  //内存子分区的全局ID。
  unsigned m_id;  //< the global sub partition ID
  const memory_config *m_config;
  class l2_cache *m_L2cache;
  class L2interface *m_L2interface;
  class gpgpu_sim *m_gpu;
  partition_mf_allocator *m_mf_allocator;

  // model delay of ROP units with a fixed latency
  struct rop_delay_t {
    unsigned long long ready_cycle;
    class mem_fetch *req;
  };
  std::queue<rop_delay_t> m_rop;

  // these are various FIFOs between units within a memory partition
  fifo_pipeline<mem_fetch> *m_icnt_L2_queue;
  fifo_pipeline<mem_fetch> *m_L2_dram_queue;
  fifo_pipeline<mem_fetch> *m_dram_L2_queue;
  fifo_pipeline<mem_fetch> *m_L2_icnt_queue;  // L2 cache hit response queue

  class mem_fetch *L2dramout;
  unsigned long long int wb_addr;

  class memory_stats_t *m_stats;

  std::set<mem_fetch *> m_request_tracker;

  friend class L2interface;

  std::vector<mem_fetch *> breakdown_request_to_sector_requests(mem_fetch *mf);

  // This is a cycle offset that has to be applied to the l2 accesses to account
  // for the cudamemcpy read/writes. We want GPGPU-Sim to only count cycles for
  // kernel execution but we want cudamemcpy to go through the L2. Everytime an
  // access is made from cudamemcpy this counter is incremented, and when the l2
  // is accessed (in both cudamemcpyies and otherwise) this value is added to
  // the gpgpu-sim cycle counters.
  unsigned m_memcpy_cycle_offset;
};

/*
对L2访存的接口。mem_fetch_interface是对mem访存的接口。
*/
class L2interface : public mem_fetch_interface {
 public:
  L2interface(memory_sub_partition *unit) { m_unit = unit; }
  virtual ~L2interface() {}
  //返回L2访存请求队列是否满了。
  virtual bool full(unsigned size, bool write) const {
    // assume read and write packets all same size
    return m_unit->m_L2_dram_queue->full();
  }
  //将新访存请求加入L2访存请求队列。
  virtual void push(mem_fetch *mf) {
    mf->set_status(IN_PARTITION_L2_TO_DRAM_QUEUE, 0 /*FIXME*/);
    m_unit->m_L2_dram_queue->push(mf);
  }

 private:
  memory_sub_partition *m_unit;
};

#endif
