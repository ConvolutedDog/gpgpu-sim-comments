// Copyright (c) 2009-2021, Tor M. Aamodt, Tayler Hetherington, Vijay Kandiah,
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

#ifndef GPU_CACHE_H
#define GPU_CACHE_H

#include <stdio.h>
#include <stdlib.h>
#include "../abstract_hardware_model.h"
#include "../tr1_hash_map.h"
#include "gpu-misc.h"
#include "mem_fetch.h"

#include <iostream>
#include "addrdec.h"

#define MAX_DEFAULT_CACHE_SIZE_MULTIBLIER 4

/*
cache block的状态，包含：
INVALID: Cache block有效，但是其中的byte mask=Cache block[mask]状态INVALID，说明sector
         缺失。
MODIFIED: 如果Cache block[mask]状态是MODIFIED，说明已经被其他线程修改，如果当前访问也是写
          操作的话即为命中，但如果不是写操作则需要判断是否mask标志的块是否修改完毕，修改完毕
          则为命中，修改不完成则为SECTOR_MISS。因为L1 cache与L2 cache写命中时，采用write-
          back策略，只将数据写入该block，并不直接更新下级存储，只有当这个块被替换时，才将数
          据写回下级存储。
VALID: 如果Cache block[mask]状态是VALID，说明已经命中。
RESERVED: 为尚未完成的缓存未命中的数据提供空间。Cache block[mask]状态是RESERVED，说明有其
          他的线程正在读取这个Cache block。挂起的命中访问已命中处于RESERVED状态的缓存行，
          这意味着同一行上已存在由先前缓存未命中发送的flying内存请求。
*/
enum cache_block_state { INVALID = 0, RESERVED, VALID, MODIFIED };

/*
对Cache请求的状态。包括：
HIT，HIT_RESERVED，MISS，RESERVATION_FAIL，SECTOR_MISS，MSHR_HIT六种状态。
*/
enum cache_request_status {
  //命中。
  HIT = 0,
  //保留成功。
  HIT_RESERVED,
  //未命中。
  MISS,
  //保留失败。
  RESERVATION_FAIL,
  //如果Cache block[mask]状态是MODIFIED，说明已经被其他线程修改，如果当前访问也是写
  //操作的话即为命中，但如果不是写操作则需要判断是否mask标志的块是否修改完毕，修改完毕
  //则为命中，修改不完成则为SECTOR_MISS。
  SECTOR_MISS,
  MSHR_HIT,
  NUM_CACHE_REQUEST_STATUS
};

enum cache_reservation_fail_reason {
  LINE_ALLOC_FAIL = 0,  // all line are reserved
  MISS_QUEUE_FULL,      // MISS queue (i.e. interconnect or DRAM) is full
  MSHR_ENRTY_FAIL,
  MSHR_MERGE_ENRTY_FAIL,
  MSHR_RW_PENDING,
  NUM_CACHE_RESERVATION_FAIL_STATUS
};

/*
缓存事件类型。
*/
enum cache_event_type {
  //写回请求。
  WRITE_BACK_REQUEST_SENT,
  //读请求。
  READ_REQUEST_SENT,
  //写请求。
  WRITE_REQUEST_SENT,
  //写分配请求。
  WRITE_ALLOCATE_SENT
};

enum cache_gpu_level {
  L1_GPU_CACHE = 0,
  L2_GPU_CACHE,
  OTHER_GPU_CACHE,
  NUM_CACHE_GPU_LEVELS
};
/*
写回时被逐出的block的信息。
*/
struct evicted_block_info {
  new_addr_type m_block_addr;
  unsigned m_modified_size;
  mem_access_byte_mask_t m_byte_mask;
  mem_access_sector_mask_t m_sector_mask;
  evicted_block_info() {
    m_block_addr = 0;
    m_modified_size = 0;
    m_byte_mask.reset();
    m_sector_mask.reset();
  }
  void set_info(new_addr_type block_addr, unsigned modified_size) {
    m_block_addr = block_addr;
    m_modified_size = modified_size;
  }
  //设置被逐出的cache block的信息。
  void set_info(new_addr_type block_addr, unsigned modified_size,
                mem_access_byte_mask_t byte_mask,
                mem_access_sector_mask_t sector_mask) {
    //地址。
    m_block_addr = block_addr;
    //被modified的sector数量。
    m_modified_size = modified_size;
    //字节mask。
    m_byte_mask = byte_mask;
    //sector mask。
    m_sector_mask = sector_mask;
  }
};

/*
Cache事件，保存了缓存事件类型，和写回时被逐出的block的信息。
*/
struct cache_event {
  //m_cache_event_type保存了缓存事件类型：
  //   enum cache_event_type {
  //     //写回请求。
  //     WRITE_BACK_REQUEST_SENT,
  //     //读请求。
  //     READ_REQUEST_SENT,
  //     //写请求。
  //     WRITE_REQUEST_SENT,
  //     //写分配请求。
  //     WRITE_ALLOCATE_SENT
  //   };
  enum cache_event_type m_cache_event_type;
  //如果当前cache_event是写回事件，就需要更新m_evicted_block。
  evicted_block_info m_evicted_block;  // if it was write_back event, fill the
                                       // the evicted block info

  cache_event(enum cache_event_type m_cache_event) {
    m_cache_event_type = m_cache_event;
  }

  cache_event(enum cache_event_type cache_event,
              evicted_block_info evicted_block) {
    m_cache_event_type = cache_event;
    m_evicted_block = evicted_block;
  }
};

const char *cache_request_status_str(enum cache_request_status status);

/*
Cache block类。
*/
struct cache_block_t {
  //构造函数。
  cache_block_t() {
    //初始化设置cache block的tag位为0。
    //  Memory  |————————|——————————|————————|
    //  Address    Tag       Set    Byte Offset
    m_tag = 0;
    //block的起始地址。
    m_block_addr = 0;
  }
  //已经选定m_lines[idx]作为逐出并reserve新访问的cache line，这里执行对新访问的reserve操作。
  virtual void allocate(new_addr_type tag, new_addr_type block_addr,
                        unsigned time,
                        mem_access_sector_mask_t sector_mask) = 0;
  virtual void fill(unsigned time, mem_access_sector_mask_t sector_mask,
                    mem_access_byte_mask_t byte_mask) = 0;

  virtual bool is_invalid_line() = 0;
  virtual bool is_valid_line() = 0;
  virtual bool is_reserved_line() = 0;
  virtual bool is_modified_line() = 0;

  virtual enum cache_block_state get_status(
      mem_access_sector_mask_t sector_mask) = 0;
  virtual void set_status(enum cache_block_state m_status,
                          mem_access_sector_mask_t sector_mask) = 0;
  virtual void set_byte_mask(mem_fetch *mf) = 0;
  virtual void set_byte_mask(mem_access_byte_mask_t byte_mask) = 0;
  virtual mem_access_byte_mask_t get_dirty_byte_mask() = 0;
  virtual mem_access_sector_mask_t get_dirty_sector_mask() = 0;
  virtual unsigned long long get_last_access_time() = 0;
  //设置当前cache line的最末次访问时间，包括sector的访问时间和line的访问时间。只有
  //访问状态为Hit时才会设置。
  virtual void set_last_access_time(unsigned long long time,
                                    mem_access_sector_mask_t sector_mask) = 0;
  virtual unsigned long long get_alloc_time() = 0;
  //在当前版本的GPGPU-Sim中，set_ignore_on_fill暂时用不到。
  virtual void set_ignore_on_fill(bool m_ignore,
                                  mem_access_sector_mask_t sector_mask) = 0;
  virtual void set_modified_on_fill(bool m_modified,
                                    mem_access_sector_mask_t sector_mask) = 0;
  virtual void set_readable_on_fill(bool readable,
                                    mem_access_sector_mask_t sector_mask) = 0;
  virtual void set_byte_mask_on_fill(bool m_modified) = 0;
  virtual unsigned get_modified_size() = 0;
  virtual void set_m_readable(bool readable,
                              mem_access_sector_mask_t sector_mask) = 0;
  virtual bool is_readable(mem_access_sector_mask_t sector_mask) = 0;
  virtual void print_status() = 0;
  virtual ~cache_block_t() {}

  new_addr_type m_tag;
  new_addr_type m_block_addr;
};

struct line_cache_block : public cache_block_t {
  //构造函数。
  line_cache_block() {
    m_alloc_time = 0;
    m_fill_time = 0;
    m_last_access_time = 0;
    //cache block的状态，包括 INVALID = 0, RESERVED, VALID, MODIFIED。
    m_status = INVALID;
    //在当前版本的GPGPU-Sim中，m_ignore_on_fill_status暂时用不到。
    m_ignore_on_fill_status = false;
    m_set_modified_on_fill = false;
    m_set_readable_on_fill = false;
    m_readable = true;
  }
  //用于为特定的地址空间分配缓存块（cache block）。其参数如下：
  // - tag：缓存块的标记（tag）
  // - block_addr：缓存块的起始地址
  // - time：当前时钟周期数
  // - sector_mask：内存访问的扇区掩码
  //该函数的作用是将指定的地址空间和对应缓存块相关联，并把该缓存块从缓存分区（cache set）中移除。同时
  //会更新缓存统计信息和模拟器内部的时间计数器。
  void allocate(new_addr_type tag, new_addr_type block_addr, unsigned time,
                mem_access_sector_mask_t sector_mask) {
    m_tag = tag;
    m_block_addr = block_addr;
    m_alloc_time = time;
    //上次访问时间
    m_last_access_time = time;
    m_fill_time = 0;
    //cache block的状态
    m_status = RESERVED;
    //在当前版本的GPGPU-Sim中，m_ignore_on_fill_status暂时用不到。
    m_ignore_on_fill_status = false;
    m_set_modified_on_fill = false;
    m_set_readable_on_fill = false;
    m_set_byte_mask_on_fill = false;
  }
  virtual void fill(unsigned time, mem_access_sector_mask_t sector_mask,
                    mem_access_byte_mask_t byte_mask) {
    // if(!m_ignore_on_fill_status)
    //	assert( m_status == RESERVED );

    m_status = m_set_modified_on_fill ? MODIFIED : VALID;

    if (m_set_readable_on_fill) m_readable = true;
    if (m_set_byte_mask_on_fill) set_byte_mask(byte_mask);

    m_fill_time = time;
  }
  //返回cache line的状态。对于Line Cache来说，cache line的状态与cache block的状态一致。
  virtual bool is_invalid_line() { return m_status == INVALID; }
  virtual bool is_valid_line() { return m_status == VALID; }
  virtual bool is_reserved_line() { return m_status == RESERVED; }
  virtual bool is_modified_line() { return m_status == MODIFIED; }

  virtual enum cache_block_state get_status(
      mem_access_sector_mask_t sector_mask) {
    return m_status;
  }
  virtual void set_status(enum cache_block_state status,
                          mem_access_sector_mask_t sector_mask) {
    m_status = status;
  }
  virtual void set_byte_mask(mem_fetch *mf) {
    m_dirty_byte_mask = m_dirty_byte_mask | mf->get_access_byte_mask();
  }
  virtual void set_byte_mask(mem_access_byte_mask_t byte_mask) {
    m_dirty_byte_mask = m_dirty_byte_mask | byte_mask;
  }
  virtual mem_access_byte_mask_t get_dirty_byte_mask() {
    return m_dirty_byte_mask;
  }
  virtual mem_access_sector_mask_t get_dirty_sector_mask() {
    mem_access_sector_mask_t sector_mask;
    if (m_status == MODIFIED) sector_mask.set();
    return sector_mask;
  }
  virtual unsigned long long get_last_access_time() {
    return m_last_access_time;
  }
  virtual void set_last_access_time(unsigned long long time,
                                    mem_access_sector_mask_t sector_mask) {
    m_last_access_time = time;
  }
  virtual unsigned long long get_alloc_time() { return m_alloc_time; }
  //在当前版本的GPGPU-Sim中，set_ignore_on_fill暂时用不到。
  virtual void set_ignore_on_fill(bool m_ignore,
                                  mem_access_sector_mask_t sector_mask) {
    //在当前版本的GPGPU-Sim中，m_ignore_on_fill_status暂时用不到。
    m_ignore_on_fill_status = m_ignore;
  }
  virtual void set_modified_on_fill(bool m_modified,
                                    mem_access_sector_mask_t sector_mask) {
    m_set_modified_on_fill = m_modified;
  }
  virtual void set_readable_on_fill(bool readable,
                                    mem_access_sector_mask_t sector_mask) {
    m_set_readable_on_fill = readable;
  }
  virtual void set_byte_mask_on_fill(bool m_modified) {
    m_set_byte_mask_on_fill = m_modified;
  }
  virtual unsigned get_modified_size() {
    return SECTOR_CHUNCK_SIZE * SECTOR_SIZE;  // i.e. cache line size
  }
  virtual void set_m_readable(bool readable,
                              mem_access_sector_mask_t sector_mask) {
    m_readable = readable;
  }
  virtual bool is_readable(mem_access_sector_mask_t sector_mask) {
    return m_readable;
  }
  virtual void print_status() {
    printf("m_block_addr is %llu, status = %u\n", m_block_addr, m_status);
  }

 private:
  unsigned long long m_alloc_time;
  unsigned long long m_last_access_time;
  unsigned long long m_fill_time;
  cache_block_state m_status;
  //在当前版本的GPGPU-Sim中，m_ignore_on_fill_status暂时用不到。
  bool m_ignore_on_fill_status;
  bool m_set_modified_on_fill;
  bool m_set_readable_on_fill;
  bool m_set_byte_mask_on_fill;
  bool m_readable;
  mem_access_byte_mask_t m_dirty_byte_mask;
};

struct sector_cache_block : public cache_block_t {
  sector_cache_block() { init(); }

  void init() {
    for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; ++i) {
      //第i个sector被分配给新访问reserve的时间。
      m_sector_alloc_time[i] = 0;
      m_sector_fill_time[i] = 0;
      //第i个sector被访问的时间，被访问包括第一次分配时的时间，也包括后续HIT该sector的时间。
      m_last_sector_access_time[i] = 0;
      m_status[i] = INVALID;
      //在当前版本的GPGPU-Sim中，m_ignore_on_fill_status暂时用不到。
      m_ignore_on_fill_status[i] = false;
      //cache block的每个sector都有一个标志位m_set_modified_on_fill[i]，标记着这个cache 
      //block是否被修改，在sector_cache_block::fill()函数调用的时候会使用。
      m_set_modified_on_fill[i] = false;
      m_set_readable_on_fill[i] = false;
      m_readable[i] = true;
    }
    m_line_alloc_time = 0;
    m_line_last_access_time = 0;
    m_line_fill_time = 0;
    m_dirty_byte_mask.reset();
  }
  
  //已经选定m_lines[idx]作为逐出并reserve新访问的cache line，这里执行对新访问的reserve操作。
  virtual void allocate(new_addr_type tag, new_addr_type block_addr,
                        unsigned time, mem_access_sector_mask_t sector_mask) {
    allocate_line(tag, block_addr, time, sector_mask);
  }

  //已经选定m_lines[idx]作为逐出并reserve新访问的cache line，这里执行对新访问的reserve操作。
  void allocate_line(new_addr_type tag, new_addr_type block_addr, unsigned time,
                     mem_access_sector_mask_t sector_mask) {
    // allocate a new line
    // assert(m_block_addr != 0 && m_block_addr != block_addr);
    init();
    m_tag = tag;
    m_block_addr = block_addr;

    unsigned sidx = get_sector_index(sector_mask);

    // set sector stats
    m_sector_alloc_time[sidx] = time;
    //第sidx个sector被访问的时间，这里被访问是第一次分配时的时间。
    m_last_sector_access_time[sidx] = time;
    m_sector_fill_time[sidx] = 0;
    //设置第sidx个sector为RESERVED。
    m_status[sidx] = RESERVED;
    //在当前版本的GPGPU-Sim中，m_ignore_on_fill_status暂时用不到。
    m_ignore_on_fill_status[sidx] = false;
    //cache block的每个sector都有一个标志位m_set_modified_on_fill[i]，标记着这个cache 
    //block是否被修改，在sector_cache_block::fill()函数调用的时候会使用。
    m_set_modified_on_fill[sidx] = false;
    m_set_readable_on_fill[sidx] = false;
    m_set_byte_mask_on_fill = false;

    // set line stats
    m_line_alloc_time = time;  // only set this for the first allocated sector
    m_line_last_access_time = time;
    m_line_fill_time = 0;
  }

  void allocate_sector(unsigned time, mem_access_sector_mask_t sector_mask) {
    // allocate invalid sector of this allocated valid line
    assert(is_valid_line());
    //sector_mask是要访问的sector的mask，例如V100中每个cache block有4个sector，那么这个
    //sector_mask就有可能是0001/0010/0100/1000，这里是判断mask为1的sector属于第几个sector
    //例如0001返回0，0010返回1，0100返回2，1000返回3。
    unsigned sidx = get_sector_index(sector_mask);

    // set sector stats
    //第sidx个sector被分配给新访问reserve的时间。
    m_sector_alloc_time[sidx] = time;
    //第sidx个sector被访问的时间，被访问包括第一次分配时的时间，也包括后续HIT该sector的时间。
    m_last_sector_access_time[sidx] = time;
    m_sector_fill_time[sidx] = 0;
    //cache block的每个sector都有一个标志位m_set_modified_on_fill[i]，标记着这个cache block
    //是否被修改，在sector_cache_block::fill()函数调用的时候会使用。
    if (m_status[sidx] == MODIFIED)  // this should be the case only for
                                     // fetch-on-write policy //TO DO
      m_set_modified_on_fill[sidx] = true;
    else
      m_set_modified_on_fill[sidx] = false;

    m_set_readable_on_fill[sidx] = false;

    m_status[sidx] = RESERVED;
    //在当前版本的GPGPU-Sim中，m_ignore_on_fill_status暂时用不到。
    m_ignore_on_fill_status[sidx] = false;
    // m_set_modified_on_fill[sidx] = false;
    m_readable[sidx] = true;

    // set line stats
    m_line_last_access_time = time;
    m_line_fill_time = 0;
  }

  virtual void fill(unsigned time, mem_access_sector_mask_t sector_mask,
                    mem_access_byte_mask_t byte_mask) {
    //sector_mask是要访问的sector的mask，例如V100中每个cache block有4个sector，那么这个
    //sector_mask就有可能是0001/0010/0100/1000，这里是判断mask为1的sector属于第几个sector
    //例如0001返回0，0010返回1，0100返回2，1000返回3。
    unsigned sidx = get_sector_index(sector_mask);

    //	if(!m_ignore_on_fill_status[sidx])
    //	         assert( m_status[sidx] == RESERVED );
    //cache block的每个sector都有一个标志位m_set_modified_on_fill[i]，标记着这个cache 
    //block是否被修改，在sector_cache_block::fill()函数调用的时候会使用。
    m_status[sidx] = m_set_modified_on_fill[sidx] ? MODIFIED : VALID;

    if (m_set_readable_on_fill[sidx]) {
      m_readable[sidx] = true;
      m_set_readable_on_fill[sidx] = false;
    }
    //在FETCH_ON_READ policy: https://arxiv.org/pdf/1810.07269.pdf 中提到，访问cache发生
    //miss时：
    // In the write-validate policy, no read fetch is required, instead each sector has 
    // a bit-wise write-mask. When a write to a single byte is received, it writes the 
    // byte to the sector, sets the corresponding write bit and sets the sector as valid 
    // and modified. When a modified cache line is evicted, the cache line is written 
    // back to the memory along with the write mask.
    // 在write-validate策略中，不需要read fetch，而是每个扇区都有一个按位写掩码。当收到对单个
    // 字节的写入时，它会将字节写入sector，设置相应的写入位，并将sector设置为有效且已修改。当修
    // 改的缓存行被逐出时，缓存行将与写入掩码一起写回内存。
    //而在FETCH_ON_READ中，需要设置sector的byte mask。这里就是指设置这个byte mask的标志。
    if (m_set_byte_mask_on_fill) set_byte_mask(byte_mask);

    m_sector_fill_time[sidx] = time;
    m_line_fill_time = time;
  }
  // 当这个cache block中存在某个sector不是INVALID时，这个cache block就不是INVALID的。当所有的
  // sector都是INVALID时，这个cache block才是INVALID的。
  virtual bool is_invalid_line() {
    // all the sectors should be invalid
    for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; ++i) {
      if (m_status[i] != INVALID) return false;
    }
    return true;
  }
  // 当这个cache block中存在某个sector不是INVALID时，这个cache block就是VALID的。
  virtual bool is_valid_line() { return !(is_invalid_line()); }
  // 当这个cache block中存在某个sector是RESERVED时，这个cache block就是RESERVED的。
  virtual bool is_reserved_line() {
    // if any of the sector is reserved, then the line is reserved
    for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; ++i) {
      if (m_status[i] == RESERVED) return true;
    }
    return false;
  }
  // 当这个cache block中存在某个sector是MODIFIED时，这个cache block就是MODIFIED的。
  virtual bool is_modified_line() {
    // if any of the sector is modified, then the line is modified
    for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; ++i) {
      if (m_status[i] == MODIFIED) return true;
    }
    return false;
  }
  // 返回cache block的某个sector的状态，这个sector由输入参数sector_mask确定。
  virtual enum cache_block_state get_status(
      mem_access_sector_mask_t sector_mask) {
    //sector_mask是要访问的sector的mask，例如V100中每个cache block有4个sector，那么这个
    //sector_mask就有可能是0001/0010/0100/1000，这里是判断mask为1的sector属于第几个sector
    //例如0001返回0，0010返回1，0100返回2，1000返回3。
    unsigned sidx = get_sector_index(sector_mask);

    return m_status[sidx];
  }
  // 设置cache block的某个sector的状态为传入参数status，这个sector由输入参数sector_mask
  // 确定。
  virtual void set_status(enum cache_block_state status,
                          mem_access_sector_mask_t sector_mask) {
    //sector_mask是要访问的sector的mask，例如V100中每个cache block有4个sector，那么这个
    //sector_mask就有可能是0001/0010/0100/1000，这里是判断mask为1的sector属于第几个sector
    //例如0001返回0，0010返回1，0100返回2，1000返回3。
    unsigned sidx = get_sector_index(sector_mask);
    m_status[sidx] = status;
  }
  // 设置cache block的byte mask。
  virtual void set_byte_mask(mem_fetch *mf) {
    m_dirty_byte_mask = m_dirty_byte_mask | mf->get_access_byte_mask();
  }
  virtual void set_byte_mask(mem_access_byte_mask_t byte_mask) {
    m_dirty_byte_mask = m_dirty_byte_mask | byte_mask;
  }
  virtual mem_access_byte_mask_t get_dirty_byte_mask() {
    return m_dirty_byte_mask;
  }
  virtual mem_access_sector_mask_t get_dirty_sector_mask() {
    mem_access_sector_mask_t sector_mask;
    for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; i++) {
      if (m_status[i] == MODIFIED) sector_mask.set(i);
    }
    return sector_mask;
  }
  virtual unsigned long long get_last_access_time() {
    return m_line_last_access_time;
  }

  //设置当前cache line的最末次访问时间，包括sector的访问时间和line的访问时间。只有
  //访问状态为Hit时才会设置。
  virtual void set_last_access_time(unsigned long long time,
                                    mem_access_sector_mask_t sector_mask) {
    //sector_mask是要访问的sector的mask，例如V100中每个cache block有4个sector，那么这个
    //sector_mask就有可能是0001/0010/0100/1000，这里是判断mask为1的sector属于第几个sector
    //例如0001返回0，0010返回1，0100返回2，1000返回3。
    unsigned sidx = get_sector_index(sector_mask);
    //第sidx个sector被访问的时间，这里被访问是HIT该sector的时间。
    m_last_sector_access_time[sidx] = time;
    m_line_last_access_time = time;
  }

  virtual unsigned long long get_alloc_time() { return m_line_alloc_time; }
  //在当前版本的GPGPU-Sim中，set_ignore_on_fill暂时用不到。
  virtual void set_ignore_on_fill(bool m_ignore,
                                  mem_access_sector_mask_t sector_mask) {
    //sector_mask是要访问的sector的mask，例如V100中每个cache block有4个sector，那么这个
    //sector_mask就有可能是0001/0010/0100/1000，这里是判断mask为1的sector属于第几个sector
    //例如0001返回0，0010返回1，0100返回2，1000返回3。
    unsigned sidx = get_sector_index(sector_mask);
    //在当前版本的GPGPU-Sim中，m_ignore_on_fill_status暂时用不到。
    m_ignore_on_fill_status[sidx] = m_ignore;
  }

  //cache block的每个sector都有一个标志位m_set_modified_on_fill[i]，标记着这个cache 
  //block是否被修改，在sector_cache_block::fill()函数调用的时候会使用。
  virtual void set_modified_on_fill(bool m_modified,
                                    mem_access_sector_mask_t sector_mask) {
    //sector_mask是要访问的sector的mask，例如V100中每个cache block有4个sector，那么这个
    //sector_mask就有可能是0001/0010/0100/1000，这里是判断mask为1的sector属于第几个sector
    //例如0001返回0，0010返回1，0100返回2，1000返回3。
    unsigned sidx = get_sector_index(sector_mask);
    m_set_modified_on_fill[sidx] = m_modified;
  }
  virtual void set_byte_mask_on_fill(bool m_modified) {
    m_set_byte_mask_on_fill = m_modified;
  }

  virtual void set_readable_on_fill(bool readable,
                                    mem_access_sector_mask_t sector_mask) {
    //sector_mask是要访问的sector的mask，例如V100中每个cache block有4个sector，那么这个
    //sector_mask就有可能是0001/0010/0100/1000，这里是判断mask为1的sector属于第几个sector
    //例如0001返回0，0010返回1，0100返回2，1000返回3。
    unsigned sidx = get_sector_index(sector_mask);
    m_set_readable_on_fill[sidx] = readable;
  }
  virtual void set_m_readable(bool readable,
                              mem_access_sector_mask_t sector_mask) {
    //sector_mask是要访问的sector的mask，例如V100中每个cache block有4个sector，那么这个
    //sector_mask就有可能是0001/0010/0100/1000，这里是判断mask为1的sector属于第几个sector
    //例如0001返回0，0010返回1，0100返回2，1000返回3。
    unsigned sidx = get_sector_index(sector_mask);
    m_readable[sidx] = readable;
  }

  virtual bool is_readable(mem_access_sector_mask_t sector_mask) {
    //sector_mask是要访问的sector的mask，例如V100中每个cache block有4个sector，那么这个
    //sector_mask就有可能是0001/0010/0100/1000，这里是判断mask为1的sector属于第几个sector
    //例如0001返回0，0010返回1，0100返回2，1000返回3。
    unsigned sidx = get_sector_index(sector_mask);
    return m_readable[sidx];
  }

  virtual unsigned get_modified_size() {
    unsigned modified = 0;
    for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; ++i) {
      if (m_status[i] == MODIFIED) modified++;
    }
    return modified * SECTOR_SIZE;
  }

  virtual void print_status() {
    printf("m_block_addr is %llu, status = %u %u %u %u\n", m_block_addr,
           m_status[0], m_status[1], m_status[2], m_status[3]);
  }

 private:
  //4个sector被分配给新访问reserve的时间。
  unsigned m_sector_alloc_time[SECTOR_CHUNCK_SIZE];
  //4个sector被访问的时间，被访问包括第一次分配时的时间，也包括后续HIT该sector的时间。
  unsigned m_last_sector_access_time[SECTOR_CHUNCK_SIZE];
  unsigned m_sector_fill_time[SECTOR_CHUNCK_SIZE];
  unsigned m_line_alloc_time;
  unsigned m_line_last_access_time;
  unsigned m_line_fill_time;
  //每个sector的状态，包括INVALID = 0, RESERVED, VALID, MODIFIED。
  cache_block_state m_status[SECTOR_CHUNCK_SIZE];
  //在当前版本的GPGPU-Sim中，m_ignore_on_fill_status暂时用不到。
  bool m_ignore_on_fill_status[SECTOR_CHUNCK_SIZE];
  //cache block的每个sector都有一个标志位m_set_modified_on_fill[i]，标记着这个cache 
  //4个sector是否被修改，在sector_cache_block::fill()函数调用的时候会使用。
  bool m_set_modified_on_fill[SECTOR_CHUNCK_SIZE];
  bool m_set_readable_on_fill[SECTOR_CHUNCK_SIZE];
  bool m_set_byte_mask_on_fill;
  bool m_readable[SECTOR_CHUNCK_SIZE];
  mem_access_byte_mask_t m_dirty_byte_mask;

  //sector_mask是要访问的sector的mask，例如V100中每个cache block有4个sector，那么这个
  //sector_mask就有可能是0001/0010/0100/1000，这里是判断mask为1的sector属于第几个sector
  //例如0001返回0，0010返回1，0100返回2，1000返回3。实际上就是返回sector_mask中的第一个
  //为1的位置，即sector在当前cache line的index。
  unsigned get_sector_index(mem_access_sector_mask_t sector_mask) {
    assert(sector_mask.count() == 1);
    for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; ++i) {
      if (sector_mask.to_ulong() & (1 << i)) return i;
    }
    return SECTOR_CHUNCK_SIZE;  // error
  }
};

enum replacement_policy_t { LRU, FIFO };

enum write_policy_t {
  READ_ONLY,
  WRITE_BACK,
  WRITE_THROUGH,
  WRITE_EVICT,
  LOCAL_WB_GLOBAL_WT
};

enum allocation_policy_t { ON_MISS, ON_FILL, STREAMING };

enum write_allocate_policy_t {
  NO_WRITE_ALLOCATE,
  WRITE_ALLOCATE,
  FETCH_ON_WRITE,
  LAZY_FETCH_ON_READ
};

enum mshr_config_t {
  TEX_FIFO,         // Tex cache
  ASSOC,            // normal cache
  SECTOR_TEX_FIFO,  // Tex cache sends requests to high-level sector cache
  SECTOR_ASSOC      // normal cache sends requests to high-level sector cache
};

enum set_index_function {
  LINEAR_SET_FUNCTION = 0,
  BITWISE_XORING_FUNCTION,
  HASH_IPOLY_FUNCTION,
  FERMI_HASH_SET_FUNCTION,
  CUSTOM_SET_FUNCTION
};

enum cache_type { NORMAL = 0, SECTOR };

#define MAX_WARP_PER_SHADER 64
#define INCT_TOTAL_BUFFER 64
#define L2_TOTAL 64
#define MAX_WARP_PER_SHADER 64
#define MAX_WARP_PER_SHADER 64

/*
Cache配置类。
*/
class cache_config {
 public:
  cache_config() {
    m_valid = false;
    // 是否当前cache被禁用。
    m_disabled = false;
    // m_config_string will be set by option parser, using .config file
    m_config_string = NULL;  // set by option parser
    // 当前这个配置已经停用。
    m_config_stringPrefL1 = NULL;
    // 当前这个配置已经停用。
    m_config_stringPrefShared = NULL;
    // 对cache的access和fill分别需要占用数据端口和填充端口，这里是bandwidth_management
    // 做管理使用。详细的使用看：
    //     bool baseline_cache::bandwidth_management::use_data_port();
    //     bool baseline_cache::bandwidth_management::use_fill_port();
    m_data_port_width = 0;
    // cache_config::hash_function()返回地址在Cache中的set。这个m_set_index_function
    // 是用作计算set的方式，即确定了地址到set的映射方式。
    m_set_index_function = LINEAR_SET_FUNCTION;
    // 流式缓存标志：
    // if (m_L1D_config.is_streaming()) {
    //   // for streaming cache, if the whole memory is allocated
    //   // to the L1 cache, then make the allocation to be on_MISS
    //   // otherwise, make it ON_FILL to eliminate line allocation fails
    //   // i.e. MSHR throughput is the same, independent on the L1 cache
    //   // size/associativity
    //   if (total_shmem == 0) {
    //     m_L1D_config.set_allocation_policy(ON_MISS);
    //     printf("GPGPU-Sim: Reconfigure L1 allocation to ON_MISS\n");
    //   } else {
    //     m_L1D_config.set_allocation_policy(ON_FILL);
    //     printf("GPGPU-Sim: Reconfigure L1 allocation to ON_FILL\n");
    //   }
    // }
    m_is_streaming = false;
    // 在逐出一个cache块时，优先逐出一个干净的块，即没有sector被RESERVED，也没有sector被
    // MODIFIED，来逐出；但是如果dirty的cache line的比例超过m_wr_percent（V100中配置为
    // 25%），也可以不满足MODIFIED的条件。
    m_wr_percent = 0;
  }
  void init(char *config, FuncCache status) {
    cache_status = status;
    assert(config);
    char ct, rp, wp, ap, mshr_type, wap, sif;

    // Cache配置参数：
    //   <sector?>:<nsets>:<bsize>:<assoc>,
    //   <rep>:<wr>:<alloc>:<wr_alloc>:<set_index_fn>,
    //   <mshr>:<N>:<merge>,<mq>:**<fifo_entry>
    // GV100配置示例：
    //   -gpgpu_cache:dl1  S:4:128:64,  L:T:m:L:L, A:512:8, 16:0,32
    //   -gpgpu_cache:dl2  S:32:128:24, L:B:m:L:P, A:192:4, 32:0,32
    //   -gpgpu_cache:il1  N:64:128:16, L:R:f:N:L, S:2:48,  4
    // 在GV100的MSHR type上，L1D为ASSOC，L2D为ASSOC，L1I为SECTOR_ASSOC。
    //   TEX_FIFO,         // Tex cache
    //   ASSOC,            // normal cache
    //   SECTOR_TEX_FIFO,  // Tex cache sends requests to high-level sector cache
    //   SECTOR_ASSOC      // normal cache sends requests to high-level sector cache
    // ct：m_cache_type，包括：
    //     1. NORMAL：cache的每个block被组织成一整个line。
    //     2. SECTOR：cache的每个block被组织成SECTOR_CHUNCK_SIZE个sector。
    // rp：m_replacement_policy，替换策略，分为LRU和FIFO。
    // wp：m_write_policy，写策略，包括：
    //     1. READ_ONLY：已被弃用，TEX Cache和READONLY Cache已经单独成class，这个配置
    //                    已经失效了。
    //     2. WRITE_BACK：写回策略，即当写命中时，只需要将数据单写更新cache，不需要直接将
    //                    数据写入下一级存储。详见data_cache::wr_hit_wb函数。
    //     3. WRITE_THROUGH：写直达策略，与写回策略不同的是，在完成写更新cache，需要直接
    //                    将数据写入下一级存储，通过send_write_request()函数下发写回
    //                    请求，详见data_cache::wr_hit_wt。
    //     4. WRITE_EVICT：写逐出策略，当写命中时，直接逐出当前块（代码中设置该块为无效），
    //                    并通过send_write_request()函数下发写回到下级存储的请求，详见
    //                    data_cache::wr_hit_we。这种策略会导致经常性的cache逐出，因此
    //                    不常使用。
    //     5. LOCAL_WB_GLOBAL_WT：Global write-evict, local write-back: Useful for 
    //                    private caches。即对于GLOBAL_ACC_W请求，采取写逐出策略，其他
    //                    请求采取写回策略。一般来说L2D Cache均采用写回策略，对L1D Cache
    //                    则采用LOCAL_WB_GLOBAL_WT策略。
    // ap：m_alloc_policy，分配策略。对于发送到 L1 D 缓存的请求，如果命中，则立即返回所需
    //                    数据；如果未命中，则分配与缓存未命中相关的资源并将请求转发到 L2 
    //                    缓存。Allocate-on-miss 和 allocateon-fill 是两种缓存行分配策
    //                    略。对于 allocateon-miss，需要为未完成的未命中分配一个缓存行槽、
    //                    一个 MSHR 和一个未命中队列条目。相比之下，allocate-on-fill，当
    //                    未完成的未命中发生时，需要分配一个 MSHR 和一个未命中队列条目，但
    //                    当所需数据从较低内存级别返回时，会选择受害者缓存行槽。在这两种策
    //                    略中，如果任何所需资源不可用，则会发生预留失败，内存管道会停滞。
    //                    分配的 MSHR 会被保留，直到从 L2 缓存/片外内存中获取数据，而未命
    //                    中队列条目会在未命中请求转发到 L2 缓存后被释放。由于 allocate-
    //                    on-fill 在驱逐之前将受害者缓存行保留在缓存中更长时间，并为未完
    //                    成的未命中保留更少的资源，因此它往往能获得更多的缓存命中和更少的
    //                    预留失败，从而比 allocate-on-miss 具有更好的性能。尽管填充时分
    //                    配需要额外的缓冲和流控制逻辑来按顺序将数据填充到缓存中，但按顺序
    //                    执行模型和写入驱逐策略使 GPU L1 D 缓存对填充时分配很友好，因为
    //                    在填充时要驱逐受害者缓存时，没有脏数据写入 L2。
    //                    详见 paper：The Demand for a Sound Baseline in GPU Memory 
    //                    Architecture Research. 
    //                    https://hzhou.wordpress.ncsu.edu/files/2022/12/Hongwen_WDDD2017.pdf
    //
    //                    For streaming cache: (1) we set the alloc policy to be on-
    //                    fill to remove all line_alloc_fail stalls. if the whole me-
    //                    mory is allocated to the L1 cache, then make the allocation 
    //                    to be on_MISS otherwise, make it ON_FILL to eliminate line 
    //                    allocation fails. i.e. MSHR throughput is the same, indepen-
    //                    dent on the L1 cache size/associativity So, we set the allo-
    //                    cation policy per kernel basis, see shader.cc, max_cta() 
    //                    function. (2) We also set the MSHRs to be equal to max allo-
    //                    cated cache lines. This is possible by moving TAG to be sha-
    //                    red between cache line and MSHR enrty (i.e. for each cache 
    //                    line, there is an MSHR rntey associated with it). This is 
    //                    the easiest think we can think of to model (mimic) L1 stream-
    //                    ing cache in Pascal and Volta. For more information about 
    //                    streaming cache, see: 
    //                    http://on-demand.gputechconf.com/gtc/2017/presentation/s7798-luke-durant-inside-volta.pdf
    //                    https://ieeexplore.ieee.org/document/8344474/
    // wap：m_write_alloc_policy，写分配策略，包括：
    //                    NO_WRITE_ALLOCATE：写不分配策略主要是指当发生写缺失时，不将数据
    //                    块加载到缓存，而是直接写到内存中。这种策略适用于数据重用率较低的
    //                    场景，因为将数据加载到缓存中没有实际意义，反而会浪费缓存空间。详见
    //                    代码 enum cache_request_status data_cache::wr_miss_no_wa。
    //                    WRITE_ALLOCATE：这是最早GPGPU-Sim版本的写分配策略，可能要从cache
    //                    里逐出一个块或者一个sector，然后将块加载到缓存再写入。
    //                    FETCH_ON_WRITE：TODO
    //                    ??? TODO
    // sif：m_set_index_function，由访存地址计算映射到的cache块在那个set的哈希方法。包括：
    //                    FERMI_HASH_SET_FUNCTION，HASH_IPOLY_FUNCTION，
    //                    CUSTOM_SET_FUNCTION，LINEAR_SET_FUNCTION，
    //                    BITWISE_XORING_FUNCTION。
    // 
    int ntok =
        sscanf(config, "%c:%u:%u:%u,%c:%c:%c:%c:%c,%c:%u:%u,%u:%u,%u", &ct,
               &m_nset, &m_line_sz, &m_assoc, &rp, &wp, &ap, &wap, &sif,
               &mshr_type, &m_mshr_entries, &m_mshr_max_merge,
               &m_miss_queue_size, &m_result_fifo_entries, &m_data_port_width);

    if (ntok < 12) {
      if (!strcmp(config, "none")) {
        m_disabled = true;
        return;
      }
      exit_parse_error();
    }

    switch (ct) {
      case 'N':
        m_cache_type = NORMAL;
        break;
      case 'S':
        m_cache_type = SECTOR;
        break;
      default:
        exit_parse_error();
    }
    switch (rp) {
      case 'L':
        m_replacement_policy = LRU;
        break;
      case 'F':
        m_replacement_policy = FIFO;
        break;
      default:
        exit_parse_error();
    }
    //在V100配置中，L1 cache为'T'，L2 cache为'B'。
    switch (wp) {
      case 'R':
        m_write_policy = READ_ONLY;
        break;
      case 'B':
        m_write_policy = WRITE_BACK;
        break;
      case 'T':
        m_write_policy = WRITE_THROUGH;
        break;
      case 'E':
        m_write_policy = WRITE_EVICT;
        break;
      case 'L':
        m_write_policy = LOCAL_WB_GLOBAL_WT;
        break;
      default:
        exit_parse_error();
    }
    switch (ap) {
      case 'm':
        m_alloc_policy = ON_MISS;
        break;
      case 'f':
        m_alloc_policy = ON_FILL;
        break;
      case 's':
        m_alloc_policy = STREAMING;
        break;
      default:
        exit_parse_error();
    }
    if (m_alloc_policy == STREAMING) {
      /*
      For streaming cache:
      (1) we set the alloc policy to be on-fill to remove all line_alloc_fail
      stalls. if the whole memory is allocated to the L1 cache, then make the
      allocation to be on_MISS otherwise, make it ON_FILL to eliminate line
      allocation fails. i.e. MSHR throughput is the same, independent on the L1
      cache size/associativity So, we set the allocation policy per kernel
      basis, see shader.cc, max_cta() function

      (2) We also set the MSHRs to be equal to max
      allocated cache lines. This is possible by moving TAG to be shared
      between cache line and MSHR enrty (i.e. for each cache line, there is
      an MSHR rntey associated with it). This is the easiest think we can
      think of to model (mimic) L1 streaming cache in Pascal and Volta

      For more information about streaming cache, see:
      http://on-demand.gputechconf.com/gtc/2017/presentation/s7798-luke-durant-inside-volta.pdf
      https://ieeexplore.ieee.org/document/8344474/
      */
      m_is_streaming = true;
      m_alloc_policy = ON_FILL;
      //m_mshr_entries = m_nset * m_assoc * MAX_DEFAULT_CACHE_SIZE_MULTIBLIER;
      //if (m_cache_type == SECTOR) m_mshr_entries *= SECTOR_CHUNCK_SIZE;
      //m_mshr_max_merge = MAX_WARP_PER_SM;
    }
    switch (mshr_type) {
      case 'F':
        m_mshr_type = TEX_FIFO;
        assert(ntok == 14);
        break;
      case 'T':
        m_mshr_type = SECTOR_TEX_FIFO;
        assert(ntok == 14);
        break;
      case 'A':
        m_mshr_type = ASSOC;
        break;
      case 'S':
        m_mshr_type = SECTOR_ASSOC;
        break;
      default:
        exit_parse_error();
    }
    m_line_sz_log2 = LOGB2(m_line_sz);
    m_nset_log2 = LOGB2(m_nset);
    m_valid = true;
    //cache替换原子操作的粒度，如果cache是SECTOR类型的，粒度为SECTOR_SIZE，否则为line_size。
    m_atom_sz = (m_cache_type == SECTOR) ? SECTOR_SIZE : m_line_sz;
    m_sector_sz_log2 = LOGB2(SECTOR_SIZE);
    original_m_assoc = m_assoc;

    // For more details about difference between FETCH_ON_WRITE and WRITE
    // VALIDAE policies Read: Jouppi, Norman P. "Cache write policies and
    // performance". ISCA 93. WRITE_ALLOCATE is the old write policy in
    // GPGPU-sim 3.x, that send WRITE and READ for every write request
    switch (wap) {
      case 'N':
        m_write_alloc_policy = NO_WRITE_ALLOCATE;
        break;
      case 'W':
        m_write_alloc_policy = WRITE_ALLOCATE;
        break;
      case 'F':
        m_write_alloc_policy = FETCH_ON_WRITE;
        break;
      case 'L':
        // 论文：https://arxiv.org/pdf/1810.07269.pdf
        m_write_alloc_policy = LAZY_FETCH_ON_READ;
        break;
      default:
        exit_parse_error();
    }

    // detect invalid configuration
    if ((m_alloc_policy == ON_FILL || m_alloc_policy == STREAMING) and
        m_write_policy == WRITE_BACK) {
      // A writeback cache with allocate-on-fill policy will inevitably lead to
      // deadlock: The deadlock happens when an incoming cache-fill evicts a
      // dirty line, generating a writeback request.  If the memory subsystem is
      // congested, the interconnection network may not have sufficient buffer
      // for the writeback request.  This stalls the incoming cache-fill.  The
      // stall may propagate through the memory subsystem back to the output
      // port of the same core, creating a deadlock where the wrtieback request
      // and the incoming cache-fill are stalling each other.
      assert(0 &&
             "Invalid cache configuration: Writeback cache cannot allocate new "
             "line on fill. ");
    }

    if ((m_write_alloc_policy == FETCH_ON_WRITE ||
         m_write_alloc_policy == LAZY_FETCH_ON_READ) &&
        m_alloc_policy == ON_FILL) {
      assert(
          0 &&
          "Invalid cache configuration: FETCH_ON_WRITE and LAZY_FETCH_ON_READ "
          "cannot work properly with ON_FILL policy. Cache must be ON_MISS. ");
    }

    if (m_cache_type == SECTOR) {
      bool cond = m_line_sz / SECTOR_SIZE == SECTOR_CHUNCK_SIZE &&
                  m_line_sz % SECTOR_SIZE == 0;
      if (!cond) {
        std::cerr << "error: For sector cache, the simulator uses hard-coded "
                     "SECTOR_SIZE and SECTOR_CHUNCK_SIZE. The line size "
                     "must be product of both values.\n";
        assert(0);
      }
    }

    // default: port to data array width and granularity = line size
    if (m_data_port_width == 0) {
      m_data_port_width = m_line_sz;
    }
    assert(m_line_sz % m_data_port_width == 0);

    switch (sif) {
      //L1D是"L"-LINEAR_SET_FUNCTION，L2D是"P"-HASH_IPOLY_FUNCTION。
      case 'H':
        m_set_index_function = FERMI_HASH_SET_FUNCTION;
        break;
      case 'P':
        m_set_index_function = HASH_IPOLY_FUNCTION;
        break;
      case 'C':
        m_set_index_function = CUSTOM_SET_FUNCTION;
        break;
      case 'L':
        m_set_index_function = LINEAR_SET_FUNCTION;
        break;
      case 'X':
        m_set_index_function = BITWISE_XORING_FUNCTION;
        break;
      default:
        exit_parse_error();
    }
  }
  bool disabled() const { return m_disabled; }
  unsigned get_line_sz() const {
    assert(m_valid);
    return m_line_sz;
  }
  unsigned get_atom_sz() const {
    assert(m_valid);
    return m_atom_sz;
  }
  unsigned get_num_lines() const {
    assert(m_valid);
    return m_nset * m_assoc;
  }
  unsigned get_max_num_lines() const {
    assert(m_valid);
    return get_max_cache_multiplier() * m_nset * original_m_assoc;
  }
  unsigned get_max_assoc() const {
    assert(m_valid);
    return get_max_cache_multiplier() * original_m_assoc;
  }
  //Cache分成多个组(set)，每个组分成多个行(way)，每个行存储字节数是line_size。
  void print(FILE *fp) const {
    fprintf(fp, "Size = %d B (%d Set x %d-way x %d byte line)\n",
            m_line_sz * m_nset * m_assoc, m_nset, m_assoc, m_line_sz);
  }

  virtual unsigned set_index(new_addr_type addr) const;

  virtual unsigned get_max_cache_multiplier() const {
    return MAX_DEFAULT_CACHE_SIZE_MULTIBLIER;
  }

  unsigned hash_function(new_addr_type addr, unsigned m_nset,
                         unsigned m_line_sz_log2, unsigned m_nset_log2,
                         unsigned m_index_function) const;

  //为了便于起见，这里的标记包括index和Tag。这允许更复杂的（可能导致不同的indexes映射到
  //同一set）set index计算，因此需要完整的标签 + 索引来检查命中/未命中。Tag现在与块地址
  //相同。
  new_addr_type tag(new_addr_type addr) const {
    // For generality, the tag includes both index and tag. This allows for more
    // complex set index calculations that can result in different indexes
    // mapping to the same set, thus the full tag + index is required to check
    // for hit/miss. Tag is now identical to the block address.

    // return addr >> (m_line_sz_log2+m_nset_log2);
    //这里实际返回的是除offset位以外的所有位+m_atom_sz'b0，即set index也作为tag的一部分了。
    return addr & ~(new_addr_type)(m_line_sz - 1);
  }
  //返回cache block的地址，该地址即为地址addr的tag位+set index位。即除offset位以外的所
  //有位。
  //|-------|-------------|--------------|
  //   tag     set_index   offset in-line  
  //m_line_sz = SECTOR_SIZE * SECTOR_CHUNCK_SIZE = 32 bytes/sector * 4 sectors = 128 bytes。
  new_addr_type block_addr(new_addr_type addr) const {
    return addr & ~(new_addr_type)(m_line_sz - 1);
  }
  //返回mshr的地址，该地址即为地址addr的tag位+set index位+sector offset位。即除single sector 
  //byte offset位以外的所有位+m_atom_sz'b0。
  //|<----------mshr_addr----------->|
  //                   sector off    byte off in-sector
  //                   |-------------|-----------|
  //                    \                       /
  //                     \                     /
  //|-------|-------------|-------------------|
  //   tag     set_index     offset in-line
  //对于sector cache，m_atom_sz = SECTOR_SIZE = 32 bytes/sector。
  //对于line cache，m_atom_sz = LINE_SIZE。
  new_addr_type mshr_addr(new_addr_type addr) const {
    return addr & ~(new_addr_type)(m_atom_sz - 1);
  }
  enum mshr_config_t get_mshr_type() const { return m_mshr_type; }
  void set_assoc(unsigned n) {
    // set new assoc. L1 cache dynamically resized in Volta
    m_assoc = n;
  }
  //返回cache有多少个set。
  unsigned get_nset() const {
    assert(m_valid);
    return m_nset;
  }
  //以KB为单位，返回整个cache的大小。
  unsigned get_total_size_inKB() const {
    assert(m_valid);
    return (m_assoc * m_nset * m_line_sz) / 1024;
  }
  bool is_streaming() { return m_is_streaming; }
  FuncCache get_cache_status() { return cache_status; }
  void set_allocation_policy(enum allocation_policy_t alloc) {
    m_alloc_policy = alloc;
  }
  char *m_config_string;
  char *m_config_stringPrefL1;
  char *m_config_stringPrefShared;
  FuncCache cache_status;
  unsigned m_wr_percent;
  write_allocate_policy_t get_write_allocate_policy() {
    return m_write_alloc_policy;
  }
  write_policy_t get_write_policy() { return m_write_policy; }

 protected:
  void exit_parse_error() {
    printf("GPGPU-Sim uArch: cache configuration parsing error (%s)\n",
           m_config_string);
    abort();
  }

  bool m_valid;
  bool m_disabled;
  //cache line的大小，以字节为单位。
  unsigned m_line_sz;
  //m_line_sz_log2 = log2(m_line_sz)。
  unsigned m_line_sz_log2;
  //cache有多少个set。
  unsigned m_nset;
  //m_nset_log2 = log2(m_nset)。
  unsigned m_nset_log2;
  //cache有多少way。
  unsigned m_assoc;
  //m_atom_sz = (m_cache_type == SECTOR) ? SECTOR_SIZE : m_line_sz;
  unsigned m_atom_sz;
  //m_sector_sz_log2 = LOGB2(SECTOR_SIZE);
  unsigned m_sector_sz_log2;
  //original assoc (defined in config '-gpgpu_cache:dl1').
  unsigned original_m_assoc;
  //当前cache是否是Streaming Cache。
  bool m_is_streaming;

  //替换策略，分为LRU和FIFO。
  enum replacement_policy_t m_replacement_policy;  // 'L' = LRU, 'F' = FIFO
  enum write_policy_t
      m_write_policy;  // 'T' = write through, 'B' = write back, 'R' = read only
  enum allocation_policy_t
      m_alloc_policy;  // 'm' = allocate on miss, 'f' = allocate on fill
  enum mshr_config_t m_mshr_type;
  enum cache_type m_cache_type;

  write_allocate_policy_t
      m_write_alloc_policy;  // 'W' = Write allocate, 'N' = No write allocate

  union {
    // MSHR Table内的entries的个数。
    unsigned m_mshr_entries;
    unsigned m_fragment_fifo_entries;
  };
  union {
    // MSHR Table内的每个entries的最大可合并地址的个数。
    unsigned m_mshr_max_merge;
    unsigned m_request_fifo_entries;
  };
  union {
    unsigned m_miss_queue_size;
    unsigned m_rob_entries;
  };
  unsigned m_result_fifo_entries;
  unsigned m_data_port_width;  //< number of byte the cache can access per cycle
  enum set_index_function
      m_set_index_function;  // Hash, linear, or custom set index function

  friend class tag_array;
  friend class baseline_cache;
  friend class read_only_cache;
  friend class tex_cache;
  friend class data_cache;
  friend class l1_cache;
  friend class l2_cache;
  friend class memory_sub_partition;
};

class l1d_cache_config : public cache_config {
 public:
  l1d_cache_config() : cache_config() {}
  unsigned set_bank(new_addr_type addr) const;
  void init(char *config, FuncCache status) {
    l1_banks_byte_interleaving_log2 = LOGB2(l1_banks_byte_interleaving);
    l1_banks_log2 = LOGB2(l1_banks);
    cache_config::init(config, status);
  }
  unsigned l1_latency;
  unsigned l1_banks;
  unsigned l1_banks_log2;
  unsigned l1_banks_byte_interleaving;
  unsigned l1_banks_byte_interleaving_log2;
  unsigned l1_banks_hashing_function;
  // In Volta, the authors assign the remaining shared memory to L1 cache,
  // if the assigned shd mem = 0, then L1 cache = 128KB.
  // Defualt config -gpgpu_cache:dl1 is 32KB DL1 and 96KB shared memory.
  // m_unified_cache_size = config '-gpgpu_unified_l1d_size' = shared mem 
  // size + L1 cache size.
  // And the max L1 cache size can be extended to 4 times of the default 
  // config '-gpgpu_cache:dl1', so here the authors defined thid parameter
  // MAX_DEFAULT_CACHE_SIZE_MULTIBLIER = 4, which will be used in function
  // get_max_cache_multiplier().
  unsigned m_unified_cache_size;
  virtual unsigned get_max_cache_multiplier() const {
    // set * assoc * cacheline size. Then convert Byte to KB
    // gpgpu_unified_cache_size is in KB while original_sz is in B
    if (m_unified_cache_size > 0) {
      // Here the authors just calculate the ratio of m_unified_cache_size
      // (config '-gpgpu_unified_l1d_size') and original_m_assoc (defined 
      // in config '-gpgpu_cache:dl1').
      unsigned original_size = m_nset * original_m_assoc * m_line_sz / 1024;
      assert(m_unified_cache_size % original_size == 0);
      return m_unified_cache_size / original_size;
    } else {
      // if m_unified_cache_size is not defined, so just defaultly set the 
      // m_unified_cache_size / original_size to be 4. It means that the 
      // current programe only uses 32KB of L1D, and 96KB of shared memory. 
      return MAX_DEFAULT_CACHE_SIZE_MULTIBLIER;
    }
  }
};

class l2_cache_config : public cache_config {
 public:
  l2_cache_config() : cache_config() {}
  void init(linear_to_raw_address_translation *address_mapping);
  virtual unsigned set_index(new_addr_type addr) const;

 private:
  linear_to_raw_address_translation *m_address_mapping;
};

/*
常量缓存和数据缓存都包含一个成员tag_array对象，实现了保留和替换逻辑。probe()函数检查一个块地址而不影响相
关数据的LRU位置，而access()是为了模拟一个影响LRU位置的查找，是产生未命中和访问统计的函数。纹理缓存没有使
用tag_array，因为它的操作与传统的缓存有很大的不同。
*/
class tag_array {
 public:
  // Use this constructor
  tag_array(cache_config &config, int core_id, int type_id);
  ~tag_array();
  //判断对cache的访问（地址为addr，sector mask为mask）是HIT/HIT_RESERVED/SECTOR_MISS/
  //MISS/RESERVATION_FAIL等状态。
  //对一个cache进行数据访问的时候，调用data_cache::access()函数：
  //- 首先cahe会调用m_tag_array->probe()函数，判断对cache的访问（地址为addr，sector mask
  //  为mask）是HIT/HIT_RESERVED/SECTOR_MISS/MISS/RESERVATION_FAIL等状态。
  //- 然后调用process_tag_probe()函数，根据cache的配置以及上面m_tag_array->probe()函数返
  //  回的cache访问状态，执行相应的操作。
  //  - process_tag_probe()函数中，会根据请求的读写状态，probe()函数返回的cache访问状态，
  //    执行m_wr_hit/m_wr_miss/m_rd_hit/m_rd_miss函数，他们会调用m_tag_array->access()
  //    函数来实现LRU状态的更新。
  enum cache_request_status probe(new_addr_type addr, unsigned &idx,
                                  mem_fetch *mf, bool is_write,
                                  bool probe_mode = false) const;
  //判断对cache的访问（地址为addr，sector mask为mask）是HIT/HIT_RESERVED/SECTOR_MISS/
  //MISS/RESERVATION_FAIL等状态。
  //对一个cache进行数据访问的时候，调用data_cache::access()函数：
  //- 首先cahe会调用m_tag_array->probe()函数，判断对cache的访问（地址为addr，sector mask
  //  为mask）是HIT/HIT_RESERVED/SECTOR_MISS/MISS/RESERVATION_FAIL等状态。
  //- 然后调用process_tag_probe()函数，根据cache的配置以及上面m_tag_array->probe()函数返
  //  回的cache访问状态，执行相应的操作。
  //  - process_tag_probe()函数中，会根据请求的读写状态，probe()函数返回的cache访问状态，
  //    执行m_wr_hit/m_wr_miss/m_rd_hit/m_rd_miss函数，他们会调用m_tag_array->access()
  //    函数来实现LRU状态的更新。
  enum cache_request_status probe(new_addr_type addr, unsigned &idx,
                                  mem_access_sector_mask_t mask, bool is_write,
                                  bool probe_mode = false,
                                  mem_fetch *mf = NULL) const;
  //更新LRU状态。Least Recently Used。返回是否需要写回wb以及逐出的cache line的信息evicted。
  //对一个cache进行数据访问的时候，调用data_cache::access()函数：
  //- 首先cahe会调用m_tag_array->probe()函数，判断对cache的访问（地址为addr，sector mask
  //  为mask）是HIT/HIT_RESERVED/SECTOR_MISS/MISS/RESERVATION_FAIL等状态。
  //- 然后调用process_tag_probe()函数，根据cache的配置以及上面m_tag_array->probe()函数返
  //  回的cache访问状态，执行相应的操作。
  //  - process_tag_probe()函数中，会根据请求的读写状态，probe()函数返回的cache访问状态，
  //    执行m_wr_hit/m_wr_miss/m_rd_hit/m_rd_miss函数，他们会调用m_tag_array->access()
  //    函数来实现LRU状态的更新。
  enum cache_request_status access(new_addr_type addr, unsigned time,
                                   unsigned &idx, mem_fetch *mf);
  //更新LRU状态。Least Recently Used。返回是否需要写回wb以及逐出的cache line的信息evicted。
  //对一个cache进行数据访问的时候，调用data_cache::access()函数：
  //- 首先cahe会调用m_tag_array->probe()函数，判断对cache的访问（地址为addr，sector mask
  //  为mask）是HIT/HIT_RESERVED/SECTOR_MISS/MISS/RESERVATION_FAIL等状态。
  //- 然后调用process_tag_probe()函数，根据cache的配置以及上面m_tag_array->probe()函数返
  //  回的cache访问状态，执行相应的操作。
  //  - process_tag_probe()函数中，会根据请求的读写状态，probe()函数返回的cache访问状态，
  //    执行m_wr_hit/m_wr_miss/m_rd_hit/m_rd_miss函数，他们会调用m_tag_array->access()
  //    函数来实现LRU状态的更新。
  enum cache_request_status access(new_addr_type addr, unsigned time,
                                   unsigned &idx, bool &wb,
                                   evicted_block_info &evicted, mem_fetch *mf);

  void fill(new_addr_type addr, unsigned time, mem_fetch *mf, bool is_write);
  void fill(unsigned idx, unsigned time, mem_fetch *mf);
  void fill(new_addr_type addr, unsigned time, mem_access_sector_mask_t mask,
            mem_access_byte_mask_t byte_mask, bool is_write);

  unsigned size() const { return m_config.get_num_lines(); }
  cache_block_t *get_block(unsigned idx) { return m_lines[idx]; }

  void flush();       // flush all written entries
  void invalidate();  // invalidate all entries
  void new_window();

  void print(FILE *stream, unsigned &total_access,
             unsigned &total_misses) const;
  float windowed_miss_rate() const;
  void get_stats(unsigned &total_access, unsigned &total_misses,
                 unsigned &total_hit_res, unsigned &total_res_fail) const;

  void update_cache_parameters(cache_config &config);
  void add_pending_line(mem_fetch *mf);
  void remove_pending_line(mem_fetch *mf);
  //当一个cache block被MODIFIED时，将其标记为DIRTY，则dirty的数量就应该加1。
  void inc_dirty() { m_dirty++; }

 protected:
  // This constructor is intended for use only from derived classes that wish to
  // avoid unnecessary memory allocation that takes place in the
  // other tag_array constructor
  tag_array(cache_config &config, int core_id, int type_id,
            cache_block_t **new_lines);
  void init(int core_id, int type_id);

 protected:
  cache_config &m_config;

  //cache block的所有集合。
  // For example, 4 sets, 6 ways:
  // |  0  |  1  |  2  |  3  |  4  |  5  |  // set_index 0
  // |  6  |  7  |  8  |  9  |  10 |  11 |  // set_index 1
  // |  12 |  13 |  14 |  15 |  16 |  17 |  // set_index 2
  // |  18 |  19 |  20 |  21 |  22 |  23 |  // set_index 3
  //                |--------> index => cache_block_t *line
  //m_lines[index] = &m_lines[set_index * m_config.m_assoc + way_index]
  cache_block_t **m_lines; /* nbanks x nset x assoc lines in total */

  //对当前tag_array的访问次数。
  unsigned m_access;
  //对当前cache访问的miss次数。
  unsigned m_miss;
  unsigned m_pending_hit;  // number of cache miss that hit a line that is
                           // allocated but not filled
  //Reservation Failed的次数。
  unsigned m_res_fail;
  //Sector Miss的次数。
  unsigned m_sector_miss;
  //Dirty block的个数。
  unsigned m_dirty;

  // performance counters for calculating the amount of misses within a time
  // window
  unsigned m_prev_snapshot_access;
  unsigned m_prev_snapshot_miss;
  unsigned m_prev_snapshot_pending_hit;

  //当前cache所属的Shader Core的ID。
  int m_core_id;  // which shader core is using this
  //什么类型的cahce，包括Normal，Texture，Constant。
  int m_type_id;  // what kind of cache is this (normal, texture, constant)

  //标记当前tag_array所属cache是否被使用过。一旦有access()函数被调用，则说明被使用过。
  bool is_used;  // a flag if the whole cache has ever been accessed before

  //已经弃用了。
  typedef tr1_hash_map<new_addr_type, unsigned> line_table;
  //已经弃用了。
  line_table pending_lines;
};

/*
未命中状态保持寄存器，the miss status holding register，MSHR。MSHR的模型是用mshr_table类来模拟
一个具有有限数量的合并请求的完全关联表。请求通过next_access()函数从MSHR中释放。MSHR表具有固定数量
的MSHR条目。每个MSHR条目可以为单个缓存行（cache line）提供固定数量的未命中请求。MSHR条目的数量和每
个条目的最大请求数是可配置的。

缓存未命中状态保持寄存器。缓存命中后，将立即向寄存器文件发送数据，以满足请求。在缓存未命中时，未命中
处理逻辑将首先检查未命中状态保持寄存器（MSHR），以查看当前是否有来自先前请求的相同请求挂起。如果是，
则此请求将合并到同一条目中，并且不需要发出新的数据请求。否则，将为该数据请求保留一个新的MSHR条目和缓
存行。缓存状态处理程序可能会在资源不可用时失败，例如没有可用的MSHR条目、该集中的所有缓存块都已保留但
尚未填充、未命中队列已满等。
*/
class mshr_table {
 public:
  //构造函数。参数为：
  //    num_entries：MSHR中的条目的个数。
  //    max_merged：MSHR中的单个条目的最大请求数，当一个请求正在运行时，对内存系统的冗余访问被合
  //                并到MSHR中。此请求将合并到同一条目中，并且不需要发出新的数据请求。
  //    m_data：std::unordered_map，是<new_addr_type, mshr_entry>的无序map。
  mshr_table(unsigned num_entries, unsigned max_merged)
      : m_num_entries(num_entries),
        // 这部分实际上是用来初始化 m_data 的桶数（bucket count）的。尽管 std::unordered_map 一
        // 般情况下不需要明确指定桶数，但通过这种方式控制哈希表的初始容量以减少重新哈希的频率，从而
        // 提升性能。
        m_max_merged(max_merged)
#if (tr1_hash_map_ismap == 0)
        ,
        m_data(2 * num_entries)
#endif
  {
  }

  /// Checks if there is a pending request to the lower memory level already
  //检查是否已存在对较低内存级别的挂起请求。即检查m_data中是否存在地址为block_addr的条目。
  bool probe(new_addr_type block_addr) const;
  /// Checks if there is space for tracking a new memory access
  //检查是否有空间处理新的内存访问。首先查找是否MSHR表中有 block_addr 地址的条目。如果存在该条目，
  //看是否有空间合并进该条目。如果不存在该条目，看是否有其他空闲条目添加。
  bool full(new_addr_type block_addr) const;
  /// Add or merge this access
  //添加或合并此访问。通常与m_mshrs.probe和!m_mshrs.full联合使用。如果m_data中存在地址为block_
  //addr，且该条目的m_list.size() < m_max_merged，则将mf添加到该条目的m_list中。否则，将mf作为
  //一个新的条目添加到m_data中。
  void add(new_addr_type block_addr, mem_fetch *mf);
  /// Returns true if cannot accept new fill responses
  //如果无法接受新的填充响应，则返回true。
  bool busy() const { return false; }
  /// Accept a new cache fill response: mark entry ready for processing
  //接受新的缓存填充响应：标记条目以备处理。这个函数会在cache填充响应时调用，用来标记MSHR表中的地
  //址block_addr的条目为就绪状态，即已经有了这个地址对应的数据。
  void mark_ready(new_addr_type block_addr, bool &has_atomic);
  /// Returns true if ready accesses exist
  //如果存在就绪访问，则返回true。m_current_response是就绪内存访问的列表。m_current_response仅
  //存储了就绪内存访问的地址。如果存在已经被填入MSHR条目的访问，则返回true。MSHR的条目非空证明可
  //以合并内存访问。
  //这里m_mshrs.access_ready()返回的是就绪内存访问的列表m_current_response是否非空，就绪内存访
  //问的列表仅存储了就绪内存访问的地址。如果存在已经被填入MSHR条目的访问，则返回true。
  bool access_ready() const { return !m_current_response.empty(); }
  /// Returns next ready access
  //返回下一个就绪访问。通常配合access_ready()一起使用，access_ready用来检查是否存在就绪访问，
  //next_access()用来返回就绪访问：
  //    bool access_ready() const { return !m_current_response.empty(); }
  mem_fetch *next_access();
  void display(FILE *fp) const;
  // Returns true if there is a pending read after write
  //如果存在挂起的写后读请求，返回true。
  bool is_read_after_write_pending(new_addr_type block_addr);

  void check_mshr_parameters(unsigned num_entries, unsigned max_merged) {
    assert(m_num_entries == num_entries &&
           "Change of MSHR parameters between kernels is not allowed");
    assert(m_max_merged == max_merged &&
           "Change of MSHR parameters between kernels is not allowed");
  }

 private:
  // finite sized, fully associative table, with a finite maximum number of
  // merged requests
  //大小有限、完全关联的表，合并请求的最大数量有限。
  const unsigned m_num_entries;
  //MSHR中的每个条目用来合并一个单独的内存访问地址mshr_addr，这个地址算法：
  //  m_atom_sz = (m_cache_type == SECTOR) ? SECTOR_SIZE : m_line_sz; 其中 SECTOR_SIZE =  
  //  const (32 bytes per sector).
  //  1. 如果是SECTOR类型的cache：
  //    mshr_addr函数返回mshr的地址，该地址即为地址addr的tag位+set index位+sector offset位。
  //    即除single sector byte offset位以外的所有位+m_atom_sz'b0。
  //    |<----------mshr_addr----------->|
  //                       sector offset  off in-sector
  //                       |-------------|-----------|
  //                        \                       /
  //                         \                     /
  //    |-------|-------------|-------------------|
  //       tag     set_index     offset in-line
  //  2. 如果不是SECTOR类型的cache：
  //    mshr_addr函数返回mshr的地址，该地址即为地址addr的tag位+set index位。即除single line 
  //    byte offset位以外的所有位+m_atom_sz'b0。
  //    |<----mshr_addr--->|
  //                                line offset
  //                       |-------------------------|
  //                        \                       /
  //                         \                     /
  //    |-------|-------------|-------------------|
  //       tag     set_index     offset in-line
  //
  //  mshr_addr定义：
  //    new_addr_type mshr_addr(new_addr_type addr) const {
  //      return addr & ~(new_addr_type)(m_atom_sz - 1);
  //    }
  //
  //这里的m_num_entries其实是mshr的条目数，即可以合并多个内存访问地址mshr_addr，每个mshr_addr
  //需要占用一个entry，而每个entry不能无限制的合并很多个地址，最大合并数m_max_merged。例如:
  // GV100配置示例：
  //   -gpgpu_cache:dl1  S:4:128:64,  L:T:m:L:L, A:512:8, 16:0,32
  //   -gpgpu_cache:dl2  S:32:128:24, L:B:m:L:P, A:192:4, 32:0,32
  //   -gpgpu_cache:il1  N:64:128:16, L:R:f:N:L, S:2:48,  4
  // L1D、L2D、L1I的配置中，mshr的条目数分别为512、192和2，每个mshr的条目最多可以合并8、4和48个。
  const unsigned m_max_merged;
  //MSHR表中的条目对象。
  struct mshr_entry {
    //单个条目中可以合并的内存访问请求。
    std::list<mem_fetch *> m_list;
    //单个条目是否是原子操作。
    bool m_has_atomic;
    mshr_entry() : m_has_atomic(false) {}
  };
  // #define tr1_hash_map std::unordered_map
  typedef tr1_hash_map<new_addr_type, mshr_entry> table;
  typedef tr1_hash_map<new_addr_type, mshr_entry> line_table;
  table m_data;
  line_table pending_lines;

  // it may take several cycles to process the merged requests
  //处理合并的请求可能需要几个周期。这个变量貌似没有用到。
  bool m_current_response_ready;
  //就绪内存访问的列表。m_current_response仅存储了就绪内存访问的地址。
  std::list<new_addr_type> m_current_response;
};

/***************************************************************** Caches
 * *****************************************************************/
///
/// Simple struct to maintain cache accesses, misses, pending hits, and
/// reservation fails.
///
struct cache_sub_stats {
  unsigned long long accesses;
  unsigned long long misses;
  unsigned long long pending_hits;
  unsigned long long res_fails;

  unsigned long long port_available_cycles;
  unsigned long long data_port_busy_cycles;
  unsigned long long fill_port_busy_cycles;

  cache_sub_stats() { clear(); }
  void clear() {
    accesses = 0;
    misses = 0;
    pending_hits = 0;
    res_fails = 0;
    port_available_cycles = 0;
    data_port_busy_cycles = 0;
    fill_port_busy_cycles = 0;
  }
  cache_sub_stats &operator+=(const cache_sub_stats &css) {
    ///
    /// Overloading += operator to easily accumulate stats
    ///
    accesses += css.accesses;
    misses += css.misses;
    pending_hits += css.pending_hits;
    res_fails += css.res_fails;
    port_available_cycles += css.port_available_cycles;
    data_port_busy_cycles += css.data_port_busy_cycles;
    fill_port_busy_cycles += css.fill_port_busy_cycles;
    return *this;
  }

  cache_sub_stats operator+(const cache_sub_stats &cs) {
    ///
    /// Overloading + operator to easily accumulate stats
    ///
    cache_sub_stats ret;
    ret.accesses = accesses + cs.accesses;
    ret.misses = misses + cs.misses;
    ret.pending_hits = pending_hits + cs.pending_hits;
    ret.res_fails = res_fails + cs.res_fails;
    ret.port_available_cycles =
        port_available_cycles + cs.port_available_cycles;
    ret.data_port_busy_cycles =
        data_port_busy_cycles + cs.data_port_busy_cycles;
    ret.fill_port_busy_cycles =
        fill_port_busy_cycles + cs.fill_port_busy_cycles;
    return ret;
  }

  void print_port_stats(FILE *fout, const char *cache_name) const;
};

// Used for collecting AerialVision per-window statistics
struct cache_sub_stats_pw {
  unsigned accesses;
  unsigned write_misses;
  unsigned write_hits;
  unsigned write_pending_hits;
  unsigned write_res_fails;

  unsigned read_misses;
  unsigned read_hits;
  unsigned read_pending_hits;
  unsigned read_res_fails;

  cache_sub_stats_pw() { clear(); }
  void clear() {
    accesses = 0;
    write_misses = 0;
    write_hits = 0;
    write_pending_hits = 0;
    write_res_fails = 0;
    read_misses = 0;
    read_hits = 0;
    read_pending_hits = 0;
    read_res_fails = 0;
  }
  cache_sub_stats_pw &operator+=(const cache_sub_stats_pw &css) {
    ///
    /// Overloading += operator to easily accumulate stats
    ///
    accesses += css.accesses;
    write_misses += css.write_misses;
    read_misses += css.read_misses;
    write_pending_hits += css.write_pending_hits;
    read_pending_hits += css.read_pending_hits;
    write_res_fails += css.write_res_fails;
    read_res_fails += css.read_res_fails;
    return *this;
  }

  cache_sub_stats_pw operator+(const cache_sub_stats_pw &cs) {
    ///
    /// Overloading + operator to easily accumulate stats
    ///
    cache_sub_stats_pw ret;
    ret.accesses = accesses + cs.accesses;
    ret.write_misses = write_misses + cs.write_misses;
    ret.read_misses = read_misses + cs.read_misses;
    ret.write_pending_hits = write_pending_hits + cs.write_pending_hits;
    ret.read_pending_hits = read_pending_hits + cs.read_pending_hits;
    ret.write_res_fails = write_res_fails + cs.write_res_fails;
    ret.read_res_fails = read_res_fails + cs.read_res_fails;
    return ret;
  }
};

///
/// Cache_stats
/// Used to record statistics for each cache.
/// Maintains a record of every 'mem_access_type' and its resulting
/// 'cache_request_status' : [mem_access_type][cache_request_status]
///
class cache_stats {
 public:
  cache_stats();
  void clear();
  // Clear AerialVision cache stats after each window
  void clear_pw();
  void inc_stats(int access_type, int access_outcome,
                 unsigned long long streamID);
  // Increment AerialVision cache stats
  void inc_stats_pw(int access_type, int access_outcome,
                    unsigned long long streamID);
  void inc_fail_stats(int access_type, int fail_outcome,
                      unsigned long long streamID);
  enum cache_request_status select_stats_status(
      enum cache_request_status probe, enum cache_request_status access) const;
  unsigned long long &operator()(int access_type, int access_outcome,
                                 bool fail_outcome,
                                 unsigned long long streamID);
  unsigned long long operator()(int access_type, int access_outcome,
                                bool fail_outcome,
                                unsigned long long streamID) const;
  cache_stats operator+(const cache_stats &cs);
  cache_stats &operator+=(const cache_stats &cs);
  void print_stats(FILE *fout, unsigned long long streamID,
                   const char *cache_name = "Cache_stats") const;
  void print_fail_stats(FILE *fout, unsigned long long streamID,
                        const char *cache_name = "Cache_fail_stats") const;

  unsigned long long get_stats(enum mem_access_type *access_type,
                               unsigned num_access_type,
                               enum cache_request_status *access_status,
                               unsigned num_access_status) const;
  void get_sub_stats(struct cache_sub_stats &css) const;

  // Get per-window cache stats for AerialVision
  void get_sub_stats_pw(struct cache_sub_stats_pw &css) const;

  void sample_cache_port_utility(bool data_port_busy, bool fill_port_busy);

 private:
  bool check_valid(int type, int status) const;
  bool check_fail_valid(int type, int fail) const;

  // CUDA streamID -> cache stats[NUM_MEM_ACCESS_TYPE]
  std::map<unsigned long long, std::vector<std::vector<unsigned long long>>>
      m_stats;
  // AerialVision cache stats (per-window)
  std::map<unsigned long long, std::vector<std::vector<unsigned long long>>>
      m_stats_pw;
  std::map<unsigned long long, std::vector<std::vector<unsigned long long>>>
      m_fail_stats;

  unsigned long long m_cache_port_available_cycles;
  unsigned long long m_cache_data_port_busy_cycles;
  unsigned long long m_cache_fill_port_busy_cycles;
};

/*
cache的基础类，虚拟函数。
*/
class cache_t {
 public:
  virtual ~cache_t() {}
  virtual enum cache_request_status access(new_addr_type addr, mem_fetch *mf,
                                           unsigned time,
                                           std::list<cache_event> &events) = 0;

  // accessors for cache bandwidth availability
  virtual bool data_port_free() const = 0;
  virtual bool fill_port_free() const = 0;
};

bool was_write_sent(const std::list<cache_event> &events);
bool was_read_sent(const std::list<cache_event> &events);
bool was_writeallocate_sent(const std::list<cache_event> &events);

/// Baseline cache
/// Implements common functions for read_only_cache and data_cache
/// Each subclass implements its own 'access' function
// 基础版Cache。实现read_only_cache和data_cache的通用功能。需要每个子类实现自己的“access”功能。
class baseline_cache : public cache_t {
 public:
  //构造函数。
  baseline_cache(const char *name, cache_config &config, int core_id,
                 int type_id, mem_fetch_interface *memport,
                 enum mem_fetch_status status, enum cache_gpu_level level,
                 gpgpu_sim *gpu)
      : m_config(config),
        m_tag_array(new tag_array(config, core_id, type_id)),
        m_mshrs(config.m_mshr_entries, config.m_mshr_max_merge),
        m_bandwidth_management(config),
        m_level(level),
        m_gpu(gpu) {
    init(name, config, memport, status);
  }

  void init(const char *name, const cache_config &config,
            mem_fetch_interface *memport, enum mem_fetch_status status) {
    m_name = name;
    assert(config.m_mshr_type == ASSOC || config.m_mshr_type == SECTOR_ASSOC);
    //mem_fetch_interface是cache对mem访存的接口，cache将miss请求发送至下一级存储就是通过
    //这个接口来发送，即m_miss_queue中的数据包需要压入m_memport实现发送至下一级存储。
    m_memport = memport;
    m_miss_queue_status = status;
  }

  virtual ~baseline_cache() { delete m_tag_array; }

  void update_cache_parameters(cache_config &config) {
    m_config = config;
    m_tag_array->update_cache_parameters(config);
    m_mshrs.check_mshr_parameters(config.m_mshr_entries,
                                  config.m_mshr_max_merge);
  }

  virtual enum cache_request_status access(new_addr_type addr, mem_fetch *mf,
                                           unsigned time,
                                           std::list<cache_event> &events) = 0;
  /// Sends next request to lower level of memory
  void cycle();
  /// Interface for response from lower memory level (model bandwidth
  /// restictions in caller)
  void fill(mem_fetch *mf, unsigned time);
  /// Checks if mf is waiting to be filled by lower memory level
  bool waiting_for_fill(mem_fetch *mf);
  /// Are any (accepted) accesses that had to wait for memory now ready? (does
  /// not include accesses that "HIT")
  //未命中状态保持寄存器，the miss status holding register，MSHR。MSHR的模型是用mshr_table类来
  //模拟一个具有有限数量的合并请求的完全关联表。请求通过next_access()函数从MSHR中释放。MSHR表具有
  //固定数量的MSHR条目。每个MSHR条目可以为单个缓存行（cache line）提供固定数量的未命中请求。MSHR
  //条目的数量和每个条目的最大请求数是可配置的。
  //缓存未命中状态保持寄存器。缓存命中后，将立即向寄存器文件发送数据，以满足请求。在缓存未命中时，未
  //命中处理逻辑将首先检查未命中状态保持寄存器（MSHR），以查看当前是否有来自先前请求的相同请求挂起。
  //如果是，则此请求将合并到同一条目中，并且不需要发出新的数据请求。否则，将为该数据请求保留一个新的
  //MSHR条目和缓存行。缓存状态处理程序可能会在资源不可用时失败，例如没有可用的MSHR条目、该集中的所
  //有缓存块都已保留但尚未填充、未命中队列已满等。
  //这里m_mshrs.access_ready()返回的是就绪内存访问的列表m_current_response是否非空，就绪内存访问
  //的列表仅存储了就绪内存访问的地址。如果存在已经被填入MSHR条目的访问，则返回true。
  bool access_ready() const { return m_mshrs.access_ready(); }
  /// Pop next ready access (does not include accesses that "HIT")
  mem_fetch *next_access() { return m_mshrs.next_access(); }
  // flash invalidate all entries in cache
  void flush() { m_tag_array->flush(); }
  void invalidate() { m_tag_array->invalidate(); }
  void print(FILE *fp, unsigned &accesses, unsigned &misses) const;
  void display_state(FILE *fp) const;

  // Stat collection
  const cache_stats &get_stats() const { return m_stats; }
  unsigned get_stats(enum mem_access_type *access_type,
                     unsigned num_access_type,
                     enum cache_request_status *access_status,
                     unsigned num_access_status) const {
    return m_stats.get_stats(access_type, num_access_type, access_status,
                             num_access_status);
  }
  void get_sub_stats(struct cache_sub_stats &css) const {
    m_stats.get_sub_stats(css);
  }
  // Clear per-window stats for AerialVision support
  void clear_pw() { m_stats.clear_pw(); }
  // Per-window sub stats for AerialVision support
  void get_sub_stats_pw(struct cache_sub_stats_pw &css) const {
    m_stats.get_sub_stats_pw(css);
  }

  // accessors for cache bandwidth availability
  bool data_port_free() const {
    return m_bandwidth_management.data_port_free();
  }
  bool fill_port_free() const {
    return m_bandwidth_management.fill_port_free();
  }
  void inc_aggregated_stats(cache_request_status status,
                            cache_request_status cache_status, mem_fetch *mf,
                            enum cache_gpu_level level);
  void inc_aggregated_fail_stats(cache_request_status status,
                                 cache_request_status cache_status,
                                 mem_fetch *mf, enum cache_gpu_level level);
  void inc_aggregated_stats_pw(cache_request_status status,
                               cache_request_status cache_status, mem_fetch *mf,
                               enum cache_gpu_level level);

  // This is a gapping hole we are poking in the system to quickly handle
  // filling the cache on cudamemcopies. We don't care about anything other than
  // L2 state after the memcopy - so just force the tag array to act as though
  // something is read or written without doing anything else.
  void force_tag_access(new_addr_type addr, unsigned time,
                        mem_access_sector_mask_t mask) {
    mem_access_byte_mask_t byte_mask;
    m_tag_array->fill(addr, time, mask, byte_mask, true);
  }

 protected:
  // Constructor that can be used by derived classes with custom tag arrays
  baseline_cache(const char *name, cache_config &config, int core_id,
                 int type_id, mem_fetch_interface *memport,
                 enum mem_fetch_status status, tag_array *new_tag_array)
      : m_config(config),
        m_tag_array(new_tag_array),
        m_mshrs(config.m_mshr_entries, config.m_mshr_max_merge),
        m_bandwidth_management(config) {
    init(name, config, memport, status);
  }

 protected:
  std::string m_name;
  cache_config &m_config;
  tag_array *m_tag_array;
  //未命中状态保持寄存器，the miss status holding register，MSHR。MSHR的模型是用mshr_table类来
  //模拟一个具有有限数量的合并请求的完全关联表。请求通过next_access()函数从MSHR中释放。MSHR表具有
  //固定数量的MSHR条目。每个MSHR条目可以为单个缓存行（cache line）提供固定数量的未命中请求。MSHR
  //条目的数量和每个条目的最大请求数是可配置的。
  //缓存未命中状态保持寄存器。缓存命中后，将立即向寄存器文件发送数据，以满足请求。在缓存未命中时，未
  //命中处理逻辑将首先检查未命中状态保持寄存器（MSHR），以查看当前是否有来自先前请求的相同请求挂起。
  //如果是，则此请求将合并到同一条目中，并且不需要发出新的数据请求。否则，将为该数据请求保留一个新的
  //MSHR条目和缓存行。缓存状态处理程序可能会在资源不可用时失败，例如没有可用的MSHR条目、该集中的所
  //有缓存块都已保留但尚未填充、未命中队列已满等。
  mshr_table m_mshrs;
  //在baseline_cache::cycle()中，会将m_miss_queue队首的数据包mf传递给下一层缓存。当遇到miss的请求
  //需要访问下一级存储时，会把miss的请求放到m_miss_queue中。
  std::list<mem_fetch *> m_miss_queue;
  enum mem_fetch_status m_miss_queue_status;
  //mem_fetch_interface是cache对mem访存的接口，cache将miss请求发送至下一级存储就是通过这个接口来发
  //送，即m_miss_queue中的数据包需要压入m_memport实现发送至下一级存储。
  mem_fetch_interface *m_memport;
  cache_gpu_level m_level;
  gpgpu_sim *m_gpu;

  struct extra_mf_fields {
    extra_mf_fields() { m_valid = false; }
    extra_mf_fields(new_addr_type a, new_addr_type ad, unsigned i, unsigned d,
                    const cache_config &m_config) {
      m_valid = true;
      m_block_addr = a;
      m_addr = ad;
      m_cache_index = i;
      m_data_size = d;
      // 当一个 load 请求生成多个 load 事务时，使用此变量。例如，来自非 sectored L1 请求的读取请求
      // 向 sectored L2 发送请求。这里的pending_read是指一个请求需要多少个load事务才能完成，每个事
      // 务会以一个数据包mf的方式返回，当数据包填回cache时，会调用fill()函数，它会将pending_read减
      // 1，当 pending_read 为 0 时，表示所有的load事务都完成了（当来一个数据包，但是经过减1还未清
      // 零，证明还有pending_read个事务尚未返回数据包）。
      // 这里是指非 sectored cache 的请求发送到一个 sectored cache时，需要多少个load事务才能完成，
      // 因此如果 sectored cache 的块大小为m_line_sz，那么则需要m_line_sz / SECTOR_SIZE个load事
      // 务。
      pending_read = m_config.m_mshr_type == SECTOR_ASSOC
                         ? m_config.m_line_sz / SECTOR_SIZE
                         : 0;
    }
    bool m_valid;
    new_addr_type m_block_addr;
    new_addr_type m_addr;
    unsigned m_cache_index;
    unsigned m_data_size;
    // this variable is used when a load request generates multiple load
    // transactions For example, a read request from non-sector L1 request sends
    // a request to sector L2
    // 当加载请求生成多个加载事务时，使用此变量。例如，来自非sectored L1 请求的读取请求向
    // sectored L2 发送请求。
    unsigned pending_read;
  };

  typedef std::map<mem_fetch *, extra_mf_fields> extra_mf_fields_lookup;

  extra_mf_fields_lookup m_extra_mf_fields;

  cache_stats m_stats;

  /// Checks whether this request can be handled on this cycle. num_miss equals
  /// max # of misses to be handled on this cycle
  //检查是否一个miss请求能够在当前时钟周期内被处理，m_miss_queue_size在V100的L1 cache
  //中配置为16，在L2 cache中配置为32，当一个请求的大小大到m_miss_queue放不下时，它就在
  //当前时钟周期内无法处理完毕。这里所说的能否在本时钟周期内处理完毕，仅是指能否将此miss
  //请求放入m_miss_queue。在baseline_cache::cycle()中，会将m_miss_queue队首的数据包
  //mf传递给下一层缓存。至于能否将这个miss请求在本时钟周期内发送至下一层缓存，就不是这里
  //需要考虑的。
  //在baseline_cache::cycle()中，会将m_miss_queue队首的数据包mf传递给下一层缓存。当遇
  //到miss的请求需要访问下一级存储时，会把miss的请求放到m_miss_queue中。
  bool miss_queue_full(unsigned num_miss) {
    return ((m_miss_queue.size() + num_miss) >= m_config.m_miss_queue_size);
  }
  /// Read miss handler without writeback
  void send_read_request(new_addr_type addr, new_addr_type block_addr,
                         unsigned cache_index, mem_fetch *mf, unsigned time,
                         bool &do_miss, std::list<cache_event> &events,
                         bool read_only, bool wa);
  /// Read miss handler. Check MSHR hit or MSHR available
  void send_read_request(new_addr_type addr, new_addr_type block_addr,
                         unsigned cache_index, mem_fetch *mf, unsigned time,
                         bool &do_miss, bool &wb, evicted_block_info &evicted,
                         std::list<cache_event> &events, bool read_only,
                         bool wa);

  /// Sub-class containing all metadata for port bandwidth management
  //cache的子类，包含端口带宽管理的所有元数据。
  class bandwidth_management {
   public:
    bandwidth_management(cache_config &config);

    /// use the data port based on the outcome and events generated by the
    /// mem_fetch request
    //根据mem_fetch请求生成的结果和事件使用数据端口。
    void use_data_port(mem_fetch *mf, enum cache_request_status outcome,
                       const std::list<cache_event> &events);

    /// use the fill port
    //根据mem_fetch请求使用填充端口。
    void use_fill_port(mem_fetch *mf);

    /// called every cache cycle to free up the ports
    void replenish_port_bandwidth();

    /// query for data port availability
    bool data_port_free() const;
    /// query for fill port availability
    bool fill_port_free() const;

   protected:
    const cache_config &m_config;

    int m_data_port_occupied_cycles;  //< Number of cycle that the data port
                                      // remains used
    int m_fill_port_occupied_cycles;  //< Number of cycle that the fill port
                                      // remains used
  };

  bandwidth_management m_bandwidth_management;
};

/// Read only cache
// 只读Cache类。
class read_only_cache : public baseline_cache {
 public:
  read_only_cache(const char *name, cache_config &config, int core_id,
                  int type_id, mem_fetch_interface *memport,
                  enum mem_fetch_status status, enum cache_gpu_level level,
                  gpgpu_sim *gpu)
      : baseline_cache(name, config, core_id, type_id, memport, status, level,
                       gpu) {}

  /// Access cache for read_only_cache: returns RESERVATION_FAIL if request
  /// could not be accepted (for any reason)
  virtual enum cache_request_status access(new_addr_type addr, mem_fetch *mf,
                                           unsigned time,
                                           std::list<cache_event> &events);

  virtual ~read_only_cache() {}

 protected:
  read_only_cache(const char *name, cache_config &config, int core_id,
                  int type_id, mem_fetch_interface *memport,
                  enum mem_fetch_status status, tag_array *new_tag_array)
      : baseline_cache(name, config, core_id, type_id, memport, status,
                       new_tag_array) {}
};

// 数据Cache类。实现 L1 和 L2 数据Cache的常用函数。
/// Data cache - Implements common functions for L1 and L2 data cache
class data_cache : public baseline_cache {
 public:
  data_cache(const char *name, cache_config &config, int core_id, int type_id,
             mem_fetch_interface *memport, mem_fetch_allocator *mfcreator,
             enum mem_fetch_status status, mem_access_type wr_alloc_type,
             mem_access_type wrbk_type, class gpgpu_sim *gpu,
             enum cache_gpu_level level)
      : baseline_cache(name, config, core_id, type_id, memport, status, level,
                       gpu) {
    init(mfcreator);
    m_wr_alloc_type = wr_alloc_type;
    m_wrbk_type = wrbk_type;
    m_gpu = gpu;
  }

  virtual ~data_cache() {}

  virtual void init(mem_fetch_allocator *mfcreator) {
    m_memfetch_creator = mfcreator;

    // Set read hit function
    m_rd_hit = &data_cache::rd_hit_base;

    // Set read miss function
    m_rd_miss = &data_cache::rd_miss_base;

    // Set write hit function
    switch (m_config.m_write_policy) {
      // 在V100配置中，L1 cache为write-through，L2 cache为write-back。
      // READ_ONLY is now a separate cache class, config is deprecated
      case READ_ONLY:
        assert(0 && "Error: Writable Data_cache set as READ_ONLY\n");
        break;
      case WRITE_BACK:
        m_wr_hit = &data_cache::wr_hit_wb;
        break;
      case WRITE_THROUGH:
        m_wr_hit = &data_cache::wr_hit_wt;
        break;
      case WRITE_EVICT:
        m_wr_hit = &data_cache::wr_hit_we;
        break;
      case LOCAL_WB_GLOBAL_WT:
        m_wr_hit = &data_cache::wr_hit_global_we_local_wb;
        break;
      default:
        assert(0 && "Error: Must set valid cache write policy\n");
        break;  // Need to set a write hit function
    }

    // Set write miss function
    //V100中配置为LAZY_FETCH_ON_READ。
    switch (m_config.m_write_alloc_policy) {
      case NO_WRITE_ALLOCATE:
        m_wr_miss = &data_cache::wr_miss_no_wa;
        break;
      case WRITE_ALLOCATE:
        m_wr_miss = &data_cache::wr_miss_wa_naive;
        break;
      case FETCH_ON_WRITE:
        m_wr_miss = &data_cache::wr_miss_wa_fetch_on_write;
        break;
      case LAZY_FETCH_ON_READ:
        m_wr_miss = &data_cache::wr_miss_wa_lazy_fetch_on_read;
        break;
      default:
        assert(0 && "Error: Must set valid cache write miss policy\n");
        break;  // Need to set a write miss function
    }
  }

  virtual enum cache_request_status access(new_addr_type addr, mem_fetch *mf,
                                           unsigned time,
                                           std::list<cache_event> &events);

 protected:
  data_cache(const char *name, cache_config &config, int core_id, int type_id,
             mem_fetch_interface *memport, mem_fetch_allocator *mfcreator,
             enum mem_fetch_status status, tag_array *new_tag_array,
             mem_access_type wr_alloc_type, mem_access_type wrbk_type,
             class gpgpu_sim *gpu)
      : baseline_cache(name, config, core_id, type_id, memport, status,
                       new_tag_array) {
    init(mfcreator);
    m_wr_alloc_type = wr_alloc_type;
    m_wrbk_type = wrbk_type;
    m_gpu = gpu;
  }

  mem_access_type m_wr_alloc_type;  // Specifies type of write allocate request
                                    // (e.g., L1 or L2)
  mem_access_type
      m_wrbk_type;  // Specifies type of writeback request (e.g., L1 or L2)
  class gpgpu_sim *m_gpu;

  //! A general function that takes the result of a tag_array probe
  //  and performs the correspding functions based on the cache configuration
  //  The access fucntion calls this function
  enum cache_request_status process_tag_probe(bool wr,
                                              enum cache_request_status status,
                                              new_addr_type addr,
                                              unsigned cache_index,
                                              mem_fetch *mf, unsigned time,
                                              std::list<cache_event> &events);

 protected:
  mem_fetch_allocator *m_memfetch_creator;

  // Functions for data cache access
  /// Sends write request to lower level memory (write or writeback)
  void send_write_request(mem_fetch *mf, cache_event request, unsigned time,
                          std::list<cache_event> &events);
  void update_m_readable(mem_fetch *mf, unsigned cache_index);
  // Member Function pointers - Set by configuration options
  // to the functions below each grouping
  /******* Write-hit configs *******/
  enum cache_request_status (data_cache::*m_wr_hit)(
      new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
      std::list<cache_event> &events, enum cache_request_status status);
  /// Marks block as MODIFIED and updates block LRU
  enum cache_request_status wr_hit_wb(
      new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
      std::list<cache_event> &events,
      enum cache_request_status status);  // write-back
  enum cache_request_status wr_hit_wt(
      new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
      std::list<cache_event> &events,
      enum cache_request_status status);  // write-through

  /// Marks block as INVALID and sends write request to lower level memory
  enum cache_request_status wr_hit_we(
      new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
      std::list<cache_event> &events,
      enum cache_request_status status);  // write-evict
  enum cache_request_status wr_hit_global_we_local_wb(
      new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
      std::list<cache_event> &events, enum cache_request_status status);
  // global write-evict, local write-back

  /******* Write-miss configs *******/
  enum cache_request_status (data_cache::*m_wr_miss)(
      new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
      std::list<cache_event> &events, enum cache_request_status status);
  /// Sends read request, and possible write-back request,
  //  to lower level memory for a write miss with write-allocate
  enum cache_request_status wr_miss_wa_naive(
      new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
      std::list<cache_event> &events,
      enum cache_request_status
          status);  // write-allocate-send-write-and-read-request
  enum cache_request_status wr_miss_wa_fetch_on_write(
      new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
      std::list<cache_event> &events,
      enum cache_request_status
          status);  // write-allocate with fetch-on-every-write
  enum cache_request_status wr_miss_wa_lazy_fetch_on_read(
      new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
      std::list<cache_event> &events,
      enum cache_request_status status);  // write-allocate with read-fetch-only
  enum cache_request_status wr_miss_wa_write_validate(
      new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
      std::list<cache_event> &events,
      enum cache_request_status
          status);  // write-allocate that writes with no read fetch
  enum cache_request_status wr_miss_no_wa(
      new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
      std::list<cache_event> &events,
      enum cache_request_status status);  // no write-allocate

  // Currently no separate functions for reads
  /******* Read-hit configs *******/
  enum cache_request_status (data_cache::*m_rd_hit)(
      new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
      std::list<cache_event> &events, enum cache_request_status status);
  enum cache_request_status rd_hit_base(new_addr_type addr,
                                        unsigned cache_index, mem_fetch *mf,
                                        unsigned time,
                                        std::list<cache_event> &events,
                                        enum cache_request_status status);

  /******* Read-miss configs *******/
  enum cache_request_status (data_cache::*m_rd_miss)(
      new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
      std::list<cache_event> &events, enum cache_request_status status);
  enum cache_request_status rd_miss_base(new_addr_type addr,
                                         unsigned cache_index, mem_fetch *mf,
                                         unsigned time,
                                         std::list<cache_event> &events,
                                         enum cache_request_status status);
};

/// This is meant to model the first level data cache in Fermi.
/// It is write-evict (global) or write-back (local) at
/// the granularity of individual blocks
/// (the policy used in fermi according to the CUDA manual)
// L1 cache采取的写策略：
//     对L1 cache写不命中时，采用write-allocate策略，将缺失块从下级存储调入L1 cache，
//                          并在L1 cache中修改。
//     对L1 cache写命中时，采用write-back策略，只写入L1 cache，不直接写入下级存储，在
//                          L1 cache的sector被逐出时才将数据写回下级缓存。
// L2 cache采取的写策略：
//     对L2 cache写不命中时，采用write-allocate策略，将缺失块从DRAM调入L2 cache，并在
//                          L2 cache中修改。
//     对L2 cache写命中时，采用write-back策略，只写入L2 cache，并不直接写入DRAM，在L2 
//                          cache的sector被逐出时才将数据写回DRAM。
class l1_cache : public data_cache {
 public:
  //L1_WR_ALLOC_R/L2_WR_ALLOC_R在V100配置中暂时用不到。
  //在V100中，L1 cache的m_write_policy为WRITE_THROUGH，实际上L1_WRBK_ACC也不会用到。
  l1_cache(const char *name, cache_config &config, int core_id, int type_id,
           mem_fetch_interface *memport, mem_fetch_allocator *mfcreator,
           enum mem_fetch_status status, class gpgpu_sim *gpu,
           enum cache_gpu_level level)
      : data_cache(name, config, core_id, type_id, memport, mfcreator, status,
                   L1_WR_ALLOC_R, L1_WRBK_ACC, gpu, level) {}

  virtual ~l1_cache() {}

  virtual enum cache_request_status access(new_addr_type addr, mem_fetch *mf,
                                           unsigned time,
                                           std::list<cache_event> &events);

 protected:
  //L1_WR_ALLOC_R/L2_WR_ALLOC_R在V100配置中暂时用不到。
  //在V100中，L1 cache的m_write_policy为WRITE_THROUGH，实际上L1_WRBK_ACC也不会用到。
  l1_cache(const char *name, cache_config &config, int core_id, int type_id,
           mem_fetch_interface *memport, mem_fetch_allocator *mfcreator,
           enum mem_fetch_status status, tag_array *new_tag_array,
           class gpgpu_sim *gpu)
      : data_cache(name, config, core_id, type_id, memport, mfcreator, status,
                   new_tag_array, L1_WR_ALLOC_R, L1_WRBK_ACC, gpu) {}
};

/// Models second level shared cache with global write-back
/// and write-allocate policies
// L1 cache采取的写策略：
//     对L1 cache写不命中时，采用write-allocate策略，将缺失块从下级存储调入L1 cache，
//                          并在L1 cache中修改。
//     对L1 cache写命中时，采用write-back策略，只写入L1 cache，不直接写入下级存储，在
//                          L1 cache的sector被逐出时才将数据写回下级缓存。
// L2 cache采取的写策略：
//     对L2 cache写不命中时，采用write-allocate策略，将缺失块从DRAM调入L2 cache，并在
//                          L2 cache中修改。
//     对L2 cache写命中时，采用write-back策略，只写入L2 cache，并不直接写入DRAM，在L2 
//                          cache的sector被逐出时才将数据写回DRAM。
class l2_cache : public data_cache {
 public:
  l2_cache(const char *name, cache_config &config, int core_id, int type_id,
           mem_fetch_interface *memport, mem_fetch_allocator *mfcreator,
           enum mem_fetch_status status, class gpgpu_sim *gpu,
           enum cache_gpu_level level)
      //在V100中，当L2 cache写不命中时，采取lazy_fetch_on_read策略，当找到一个cache block
      //逐出时，如果这个cache block是被MODIFIED，则需要将这个cache block写回到下一级存储，
      //因此会产生L2_WRBK_ACC访问，这个访问就是为了写回被逐出的MODIFIED cache block。
      : data_cache(name, config, core_id, type_id, memport, mfcreator, status,
                   L2_WR_ALLOC_R, L2_WRBK_ACC, gpu, level) {}

  virtual ~l2_cache() {}

  virtual enum cache_request_status access(new_addr_type addr, mem_fetch *mf,
                                           unsigned time,
                                           std::list<cache_event> &events);
};

/*****************************************************************************/

// See the following paper to understand this cache model:
//
// Igehy, et al., Prefetching in a Texture Cache Architecture,
// Proceedings of the 1998 Eurographics/SIGGRAPH Workshop on Graphics Hardware
// http://www-graphics.stanford.edu/papers/texture_prefetch/
class tex_cache : public cache_t {
 public:
  tex_cache(const char *name, cache_config &config, int core_id, int type_id,
            mem_fetch_interface *memport, enum mem_fetch_status request_status,
            enum mem_fetch_status rob_status)
      : m_config(config),
        m_tags(config, core_id, type_id),
        m_fragment_fifo(config.m_fragment_fifo_entries),
        m_request_fifo(config.m_request_fifo_entries),
        m_rob(config.m_rob_entries),
        m_result_fifo(config.m_result_fifo_entries) {
    m_name = name;
    assert(config.m_mshr_type == TEX_FIFO ||
           config.m_mshr_type == SECTOR_TEX_FIFO);
    assert(config.m_write_policy == READ_ONLY);
    assert(config.m_alloc_policy == ON_MISS);
    //mem_fetch_interface是cache对mem访存的接口，cache将miss请求发送至下一级存储就是通过
    //这个接口来发送，即m_miss_queue中的数据包需要压入m_memport实现发送至下一级存储。
    m_memport = memport;
    m_cache = new data_block[config.get_num_lines()];
    m_request_queue_status = request_status;
    m_rob_status = rob_status;
  }

  /// Access function for tex_cache
  /// return values: RESERVATION_FAIL if request could not be accepted
  /// otherwise returns HIT_RESERVED or MISS; NOTE: *never* returns HIT
  /// since unlike a normal CPU cache, a "HIT" in texture cache does not
  /// mean the data is ready (still need to get through fragment fifo)
  enum cache_request_status access(new_addr_type addr, mem_fetch *mf,
                                   unsigned time,
                                   std::list<cache_event> &events);
  void cycle();
  /// Place returning cache block into reorder buffer
  void fill(mem_fetch *mf, unsigned time);
  /// Are any (accepted) accesses that had to wait for memory now ready? (does
  /// not include accesses that "HIT")
  bool access_ready() const { return !m_result_fifo.empty(); }
  /// Pop next ready access (includes both accesses that "HIT" and those that
  /// "MISS")
  mem_fetch *next_access() { return m_result_fifo.pop(); }
  void display_state(FILE *fp) const;

  // accessors for cache bandwidth availability - stubs for now
  bool data_port_free() const { return true; }
  bool fill_port_free() const { return true; }

  // Stat collection
  const cache_stats &get_stats() const { return m_stats; }
  unsigned get_stats(enum mem_access_type *access_type,
                     unsigned num_access_type,
                     enum cache_request_status *access_status,
                     unsigned num_access_status) const {
    return m_stats.get_stats(access_type, num_access_type, access_status,
                             num_access_status);
  }

  void get_sub_stats(struct cache_sub_stats &css) const {
    m_stats.get_sub_stats(css);
  }

 private:
  std::string m_name;
  const cache_config &m_config;

  struct fragment_entry {
    fragment_entry() {}
    fragment_entry(mem_fetch *mf, unsigned idx, bool m, unsigned d) {
      m_request = mf;
      m_cache_index = idx;
      m_miss = m;
      m_data_size = d;
    }
    mem_fetch *m_request;    // request information
    unsigned m_cache_index;  // where to look for data
    bool m_miss;             // true if sent memory request
    unsigned m_data_size;
  };

  struct rob_entry {
    rob_entry() {
      m_ready = false;
      m_time = 0;
      m_request = NULL;
    }
    rob_entry(unsigned i, mem_fetch *mf, new_addr_type a) {
      m_ready = false;
      m_index = i;
      m_time = 0;
      m_request = mf;
      m_block_addr = a;
    }
    bool m_ready;
    unsigned m_time;   // which cycle did this entry become ready?
    unsigned m_index;  // where in cache should block be placed?
    mem_fetch *m_request;
    new_addr_type m_block_addr;
  };

  struct data_block {
    data_block() { m_valid = false; }
    bool m_valid;
    new_addr_type m_block_addr;
  };

  // TODO: replace fifo_pipeline with this?
  template <class T>
  class fifo {
   public:
    fifo(unsigned size) {
      m_size = size;
      m_num = 0;
      m_head = 0;
      m_tail = 0;
      m_data = new T[size];
    }
    bool full() const { return m_num == m_size; }
    bool empty() const { return m_num == 0; }
    unsigned size() const { return m_num; }
    unsigned capacity() const { return m_size; }
    unsigned push(const T &e) {
      assert(!full());
      m_data[m_head] = e;
      unsigned result = m_head;
      inc_head();
      return result;
    }
    T pop() {
      assert(!empty());
      T result = m_data[m_tail];
      inc_tail();
      return result;
    }
    const T &peek(unsigned index) const {
      assert(index < m_size);
      return m_data[index];
    }
    T &peek(unsigned index) {
      assert(index < m_size);
      return m_data[index];
    }
    T &peek() const { return m_data[m_tail]; }
    unsigned next_pop_index() const { return m_tail; }

   private:
    void inc_head() {
      m_head = (m_head + 1) % m_size;
      m_num++;
    }
    void inc_tail() {
      assert(m_num > 0);
      m_tail = (m_tail + 1) % m_size;
      m_num--;
    }

    unsigned m_head;  // next entry goes here
    unsigned m_tail;  // oldest entry found here
    unsigned m_num;   // how many in fifo?
    unsigned m_size;  // maximum number of entries in fifo
    T *m_data;
  };

  tag_array m_tags;
  fifo<fragment_entry> m_fragment_fifo;
  fifo<mem_fetch *> m_request_fifo;
  fifo<rob_entry> m_rob;
  data_block *m_cache;
  fifo<mem_fetch *> m_result_fifo;  // next completed texture fetch
  //mem_fetch_interface是cache对mem访存的接口，cache将miss请求发送至下一级存储就是通过
  //这个接口来发送，即m_miss_queue中的数据包需要压入m_memport实现发送至下一级存储。
  mem_fetch_interface *m_memport;
  enum mem_fetch_status m_request_queue_status;
  enum mem_fetch_status m_rob_status;

  struct extra_mf_fields {
    extra_mf_fields() { m_valid = false; }
    extra_mf_fields(unsigned i, const cache_config &m_config) {
      m_valid = true;
      m_rob_index = i;
      pending_read = m_config.m_mshr_type == SECTOR_TEX_FIFO
                         ? m_config.m_line_sz / SECTOR_SIZE
                         : 0;
    }
    bool m_valid;
    unsigned m_rob_index;
    unsigned pending_read;
  };

  cache_stats m_stats;

  typedef std::map<mem_fetch *, extra_mf_fields> extra_mf_fields_lookup;

  extra_mf_fields_lookup m_extra_mf_fields;
};

#endif
