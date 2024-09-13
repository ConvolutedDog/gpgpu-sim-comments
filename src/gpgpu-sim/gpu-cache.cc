// Copyright (c) 2009-2021, Tor M. Aamodt, Tayler Hetherington,
// Vijay Kandiah, Nikos Hardavellas, Mahmoud Khairy, Junrui Pan,
// Timothy G. Rogers
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

#include "gpu-cache.h"
#include <assert.h>
#include "gpu-sim.h"
#include "hashing.h"
#include "stat-tool.h"

// used to allocate memory that is large enough to adapt the changes in cache
// size across kernels

const char *cache_request_status_str(enum cache_request_status status) {
  static const char *static_cache_request_status_str[] = {
      "HIT",         "HIT_RESERVED", "MISS", "RESERVATION_FAIL",
      "SECTOR_MISS", "MSHR_HIT"};

  assert(sizeof(static_cache_request_status_str) / sizeof(const char *) ==
         NUM_CACHE_REQUEST_STATUS);
  assert(status < NUM_CACHE_REQUEST_STATUS);

  return static_cache_request_status_str[status];
}

const char *cache_fail_status_str(enum cache_reservation_fail_reason status) {
  static const char *static_cache_reservation_fail_reason_str[] = {
      "LINE_ALLOC_FAIL", "MISS_QUEUE_FULL", "MSHR_ENRTY_FAIL",
      "MSHR_MERGE_ENRTY_FAIL", "MSHR_RW_PENDING"};

  assert(sizeof(static_cache_reservation_fail_reason_str) /
             sizeof(const char *) ==
         NUM_CACHE_RESERVATION_FAIL_STATUS);
  assert(status < NUM_CACHE_RESERVATION_FAIL_STATUS);

  return static_cache_reservation_fail_reason_str[status];
}

unsigned l1d_cache_config::set_bank(new_addr_type addr) const {
  // For sector cache, we select one sector per bank (sector interleaving)
  // This is what was found in Volta (one sector per bank, sector interleaving)
  // otherwise, line interleaving
  //对于扇区缓存，我们为每个存储体选择一个扇区（扇区交错）。
  //这是在Volta中发现的（每个存储体一个扇区，扇区交错），否则，行交错。

  // 这里是计算 L1 的 bank index，而不是计算 set index。仅复用了 set_index 的哈希函数。
  return cache_config::hash_function(addr, l1_banks,
                                     l1_banks_byte_interleaving_log2,
                                     l1_banks_log2, l1_banks_hashing_function);
}

/*
返回一个地址在Cache中的set。
*/
unsigned cache_config::set_index(new_addr_type addr) const {
  // m_line_sz_log2 = LOGB2(m_line_sz);
  // m_nset_log2 = LOGB2(m_nset);
  // m_set_index_function = L1D是"L"-LINEAR_SET_FUNCTION，L2D是"P"-HASH_IPOLY_FUNCTION。
  return cache_config::hash_function(addr, m_nset, m_line_sz_log2, m_nset_log2,
                                     m_set_index_function);
}

/*
返回一个地址在Cache中的set。
m_line_sz_log2 = LOGB2(m_line_sz);
m_nset_log2 = LOGB2(m_nset);
m_set_index_function = L1D是"L"-LINEAR_SET_FUNCTION，L2D是"P"-HASH_IPOLY_FUNCTION。
*/
unsigned cache_config::hash_function(new_addr_type addr, unsigned m_nset,
                                     unsigned m_line_sz_log2,
                                     unsigned m_nset_log2,
                                     unsigned m_index_function) const {
  unsigned set_index = 0;

  switch (m_index_function) {
    case FERMI_HASH_SET_FUNCTION: {
      /*
       * Set Indexing function from "A Detailed GPU Cache Model Based on Reuse
       * Distance Theory" Cedric Nugteren et al. HPCA 2014
       */
      unsigned lower_xor = 0;
      unsigned upper_xor = 0;

      if (m_nset == 32 || m_nset == 64) {
        // Lower xor value is bits 7-11
        lower_xor = (addr >> m_line_sz_log2) & 0x1F;

        // Upper xor value is bits 13, 14, 15, 17, and 19
        upper_xor = (addr & 0xE000) >> 13;    // Bits 13, 14, 15
        upper_xor |= (addr & 0x20000) >> 14;  // Bit 17
        upper_xor |= (addr & 0x80000) >> 15;  // Bit 19

        set_index = (lower_xor ^ upper_xor);

        // 48KB cache prepends the set_index with bit 12
        if (m_nset == 64) set_index |= (addr & 0x1000) >> 7;

      } else { /* Else incorrect number of sets for the hashing function */
        assert(
            "\nGPGPU-Sim cache configuration error: The number of sets should "
            "be "
            "32 or 64 for the hashing set index function.\n" &&
            0);
      }
      break;
    }

    case BITWISE_XORING_FUNCTION: {
      new_addr_type higher_bits = addr >> (m_line_sz_log2 + m_nset_log2);
      unsigned index = (addr >> m_line_sz_log2) & (m_nset - 1);
      set_index = bitwise_hash_function(higher_bits, index, m_nset);
      break;
    }

    // V100配置的L2D Cache。
    case HASH_IPOLY_FUNCTION: {
      // addr: [m_line_sz_log2+m_nset_log2-1:0]                => set index + byte offset
      // addr: [:m_line_sz_log2+m_nset_log2]                   => Tag
      new_addr_type higher_bits = addr >> (m_line_sz_log2 + m_nset_log2); // higher_bits = Tag
      unsigned index = (addr >> m_line_sz_log2) & (m_nset - 1);
      set_index = ipoly_hash_function(higher_bits, index, m_nset);
      break;
    }
    case CUSTOM_SET_FUNCTION: {
      /* No custom set function implemented */
      break;
    }

    // V100配置的L1D Cache。
    case LINEAR_SET_FUNCTION: {
      // addr: [m_line_sz_log2-1:0]                            => byte offset
      // addr: [m_line_sz_log2+m_nset_log2-1:m_line_sz_log2]   => set index
      set_index = (addr >> m_line_sz_log2) & (m_nset - 1);
      break;
    }

    default: {
      assert("\nUndefined set index function.\n" && 0);
      break;
    }
  }

  // Linear function selected or custom set index function not implemented
  assert((set_index < m_nset) &&
         "\nError: Set index out of bounds. This is caused by "
         "an incorrect or unimplemented custom set index function.\n");

  return set_index;
}

void l2_cache_config::init(linear_to_raw_address_translation *address_mapping) {
  cache_config::init(m_config_string, FuncCachePreferNone);
  m_address_mapping = address_mapping;
}

unsigned l2_cache_config::set_index(new_addr_type addr) const {
  new_addr_type part_addr = addr;

  if (m_address_mapping) {
    // Calculate set index without memory partition bits to reduce set camping
    part_addr = m_address_mapping->partition_address(addr);
  }

  return cache_config::set_index(part_addr);
}

tag_array::~tag_array() {
  unsigned cache_lines_num = m_config.get_max_num_lines();
  for (unsigned i = 0; i < cache_lines_num; ++i) delete m_lines[i];
  delete[] m_lines;
}

tag_array::tag_array(cache_config &config, int core_id, int type_id,
                     cache_block_t **new_lines)
    : m_config(config), m_lines(new_lines) {
  init(core_id, type_id);
}

/*
更新cache_config m_config参数。
*/
void tag_array::update_cache_parameters(cache_config &config) {
  m_config = config;
}

tag_array::tag_array(cache_config &config, int core_id, int type_id)
    : m_config(config) {
  // assert( m_config.m_write_policy == READ_ONLY ); Old assert
  // config.get_max_num_lines() 这里是因为当L1D cache被配置为128KB时，需要的
  // cache block数目较多，这里要保证cache_lines_num足够用，因此用max_num_lines
  // 来计算最坏情况下，需要用多少cache blocks。
  unsigned cache_lines_num = config.get_max_num_lines();
  // 所有的cache blocks都保存在m_lines。m_lines被定义为：
  //   cache_block_t **m_lines; /* nbanks x nset x assoc lines in total */
  // 因此m_lines[...]是指向单个cache block的指针。
  m_lines = new cache_block_t *[cache_lines_num];
  // 这里就开始区分是 line cache 和 sector cache 了。
  if (config.m_cache_type == NORMAL) {
    for (unsigned i = 0; i < cache_lines_num; ++i)
      m_lines[i] = new line_cache_block();
  } else if (config.m_cache_type == SECTOR) {
    for (unsigned i = 0; i < cache_lines_num; ++i)
      m_lines[i] = new sector_cache_block();
  } else
    assert(0);

  // 初始化一些统计参数。
  init(core_id, type_id);
}

void tag_array::init(int core_id, int type_id) {
  //访问当前cache的次数。即tag_array::access()函数被调用的次数。
  m_access = 0;
  //当前cache的miss次数。即tag_array::access()函数返回MISS的次数。
  m_miss = 0;
  //当前cache的pending hit次数。即tag_array::access()函数返回HIT_RESERVED的次数。
  m_pending_hit = 0;
  //当前cache的reservation fail次数。即tag_array::access()函数返回RESERVATION_FAIL的次数。
  m_res_fail = 0;
  //当前cache的sector miss次数。即tag_array::access()函数返回SECTOR_MISS的次数。
  m_sector_miss = 0;
  // initialize snapshot counters for visualizer
  m_prev_snapshot_access = 0;
  m_prev_snapshot_miss = 0;
  m_prev_snapshot_pending_hit = 0;
  //SM_ID。对于L1 cache，这里是core_id，对于L2 cache，这里是-1。
  m_core_id = core_id;
  //Cache类型ID：
  //    enum cache_access_logger_types { NORMALS, TEXTURE, CONSTANT, INSTRUCTION };
  //对于L1 cache，这里是type_id，对于L2 cache，这里是-1。
  m_type_id = type_id;
  //a flag if the whole cache has ever been accessed before
  is_used = false;
  //Dirty block的个数。
  m_dirty = 0;
}

// 已经弃用了。
void tag_array::add_pending_line(mem_fetch *mf) {
  assert(mf);
  new_addr_type addr = m_config.block_addr(mf->get_addr());
  line_table::const_iterator i = pending_lines.find(addr);
  if (i == pending_lines.end()) {
    pending_lines[addr] = mf->get_inst().get_uid();
  }
}

// 已经弃用了。
void tag_array::remove_pending_line(mem_fetch *mf) {
  assert(mf);
  new_addr_type addr = m_config.block_addr(mf->get_addr());
  line_table::const_iterator i = pending_lines.find(addr);
  if (i != pending_lines.end()) {
    pending_lines.erase(addr);
  }
}

/*
判断对cache的访问（地址为addr，sector mask为mask）是HIT/HIT_RESERVED/SECTOR_MISS/MISS
/RESERVATION_FAIL等状态。
对一个cache进行数据访问的时候，调用data_cache::access()函数：
- 首先cahe会调用m_tag_array->probe()函数，判断对cache的访问（地址为addr，sector mask
  为mask）是HIT/HIT_RESERVED/SECTOR_MISS/MISS/RESERVATION_FAIL等状态。
- 然后调用process_tag_probe()函数，根据cache的配置以及上面m_tag_array->probe()函数返
  回的cache访问状态，执行相应的操作。
  - process_tag_probe()函数中，会根据请求的读写状态，probe()函数返回的cache访问状态，
    执行m_wr_hit/m_wr_miss/m_rd_hit/m_rd_miss函数，他们会调用m_tag_array->access()
    函数来实现LRU状态的更新。
*/
enum cache_request_status tag_array::probe(new_addr_type addr, unsigned &idx,
                                           mem_fetch *mf, bool is_write,
                                           bool probe_mode) const {
  mem_access_sector_mask_t mask = mf->get_access_sector_mask();
  return probe(addr, idx, mask, is_write, probe_mode, mf);
}

/*
判断对cache的访问（地址为addr，sector mask为mask）是HIT/HIT_RESERVED/SECTOR_MISS/MISS
/RESERVATION_FAIL等状态。
对一个cache进行数据访问的时候，调用data_cache::access()函数：
- 首先cahe会调用m_tag_array->probe()函数，判断对cache的访问（地址为addr，sector mask
  为mask）是HIT/HIT_RESERVED/SECTOR_MISS/MISS/RESERVATION_FAIL等状态。
- 然后调用process_tag_probe()函数，根据cache的配置以及上面m_tag_array->probe()函数返
  回的cache访问状态，执行相应的操作。
  - process_tag_probe()函数中，会根据请求的读写状态，probe()函数返回的cache访问状态，
    执行m_wr_hit/m_wr_miss/m_rd_hit/m_rd_miss函数，他们会调用m_tag_array->access()
    函数来实现LRU状态的更新。

这里要区分line cache和sector cache的具体执行过程。
*/
enum cache_request_status tag_array::probe(new_addr_type addr, unsigned &idx,
                                           mem_access_sector_mask_t mask,
                                           bool is_write, bool probe_mode,
                                           mem_fetch *mf) const {
  //这里的输入地址addr是cache block的地址，该地址即为地址addr的tag位+set index位。即除
  //offset位以外的所有位。
  //  |-------|-------------|--------------|
  //             set_index   offset in-line
  //  |<--------tag--------> 0 0 0 0 0 0 0 |

  // assert( m_config.m_write_policy == READ_ONLY );
  //返回一个地址addr在Cache中的set index。这里的set index有一整套的映射函数。
  unsigned set_index = m_config.set_index(addr);
  //为了便于起见，这里的标记包括index和Tag。这允许更复杂的（可能导致不同的indexes映射到
  //同一set）set index计算，因此需要完整的标签 + 索引来检查命中/未命中。Tag现在与块地址
  //相同。
  //这里实际返回的是{除offset位以外的所有位, offset'b0}，即set index也作为tag的一部分了。
  new_addr_type tag = m_config.tag(addr);

  unsigned invalid_line = (unsigned)-1;
  unsigned valid_line = (unsigned)-1;
  unsigned long long valid_timestamp = (unsigned)-1;

  bool all_reserved = true;
  // check for hit or pending hit
  //对所有的Cache Ways检查。需要注意这里其实是针对一个set的所有way进行检查，因为给我们一个
  //地址，我们可以确定它所在的set index，然后再通过tag来确定这个地址在哪一个way上。
  for (unsigned way = 0; way < m_config.m_assoc; way++) {
    // For example, 4 sets, 6 ways:
    // |  0  |  1  |  2  |  3  |  4  |  5  |  // set_index 0
    // |  6  |  7  |  8  |  9  |  10 |  11 |  // set_index 1
    // |  12 |  13 |  14 |  15 |  16 |  17 |  // set_index 2
    // |  18 |  19 |  20 |  21 |  22 |  23 |  // set_index 3
    //                |--------> index => cache_block_t *line
    // cache block的索引。
    unsigned index = set_index * m_config.m_assoc + way;
    cache_block_t *line = m_lines[index];
    // Tag相符。m_tag和tag均是：{除offset位以外的所有位, offset'b0}
    if (line->m_tag == tag) {
      // enum cache_block_state { INVALID = 0, RESERVED, VALID, MODIFIED };
      // cache block的状态，包含：
      //   INVALID: Cache block有效，但是其中的byte mask=Cache block[mask]状态INVALID，
      //           说明sector缺失。
      //   MODIFIED: 如果Cache block[mask]状态是MODIFIED，说明已经被其他线程修改，如果当
      //             前访问也是写操作的话即为命中，但如果不是写操作则需要判断是否mask标志的
      //             块是否修改完毕，修改完毕则为命中，修改不完成则为SECTOR_MISS。因为L1 
      //             cache与L2 cache写命中时，采用write-back策略，只将数据写入该block，
      //             并不直接更新下级存储，只有当这个块被替换时，才将数据写回下级存储。
      //   VALID: 如果Cache block[mask]状态是VALID，说明已经命中。
      //   RESERVED: 为尚未完成的缓存未命中的数据提供空间。Cache block[mask]状态RESERVED，
      //             说明有其他的线程正在读取这个Cache block。挂起的命中访问已命中处于RE-
      //             SERVED状态的缓存行，意味着同一行上已存在由先前缓存未命中发送的flying
      //             内存请求。
      if (line->get_status(mask) == RESERVED) {
        //如果Cache block[mask]状态是RESERVED，说明有其他的线程正在读取这个Cache block。
        //挂起的命中访问已命中处于RESERVED状态的缓存行，这意味着同一行上已存在由先前缓存未
        //命中发送的flying内存请求。
        idx = index;
        return HIT_RESERVED;
      } else if (line->get_status(mask) == VALID) {
        //如果Cache block[mask]状态是VALID，说明已经命中。
        idx = index;
        return HIT;
      } else if (line->get_status(mask) == MODIFIED) {
        //如果Cache block[mask]状态是MODIFIED，说明已经被其他线程修改，如果当前访问也是写
        //操作的话即为命中，但如果不是写操作则需要判断是否mask标志的块是否修改完毕，修改完毕
        //则为命中，修改不完成则为SECTOR_MISS。因为L1 cache与L2 cache写命中时，采用write-
        //back策略，只将数据写入该block，并不直接更新下级存储，只有当这个块被替换时，才将数
        //据写回下级存储。
        //is_readable(mask)是判断mask标志的sector是否已经全部写完成，因为在修改cache的过程
        //中，有一个sector被修改即算作当前cache块MODIFIED，但是修改过程可能不是一下就能写完，
        //因此需要判断一下是否全部当前mask标记所读的sector写完才可以算作读命中。
        if ((!is_write && line->is_readable(mask)) || is_write) {
          // 当前line的mask位被修改，如果是写就无所谓，它依然命中，直接覆盖写即可；但是如果
          // 是读，就需要看mask位是否是可读的。如果是可读的，即为命中。
          idx = index;
          return HIT;
        } else {
          // 满足这个分支的条件是：is_write为false，当前访问是读，line->is_readable(mask)
          // 为false，mask位不是可读的，则说明当前读的sector缺失。
          idx = index;
          return SECTOR_MISS;
        }

      } else if (line->is_valid_line() && line->get_status(mask) == INVALID) {
        // 对于line cache不会走这个分支，因为line cache中，line->is_valid_line()返回的是
        // m_status的值，当其为 VALID 时，line cache中line->get_status(mask)也是返回的
        // 也是m_status的值，即为 VALID，因此对于line cache这条分支无效。
        // 但是对于sector cache， 有：
        //   virtual bool is_valid_line() { return !(is_invalid_line()); }
        // 而sector cache中的is_invalid_line()是，只要有一个sector不为INVALID即返回false，
        // 因此is_valid_line()返回的是，只要有一个sector不为INVALID就设置is_valid_line()
        // 为真。所以这条分支对于sector cache是可走的。
        //Cache block有效，但是其中的byte mask=Cache block[mask]状态无效，说明sector缺失。
        idx = index;
        return SECTOR_MISS;
      } else {
        assert(line->get_status(mask) == INVALID);
      }
    }
    
    //每一次循环中能走到这里的，即为当前cache block的line->m_tag!=tag。那么就需要考虑当前这
    //cache block能否被逐出替换，请注意，这个判断是在对每一个way循环的过程中进行的，也就是说，
    //加入第一个cache block没有返回以上访问状态，但有可能直到所有way的最后一个cache block才
    //满足line->m_tag!=tag，但是在对第0~way-2号的cache block循环判断的时候，就需要记录下每
    //一个way的cache block是否能够被逐出。因为如果等到所有way的cache block都没有满足line->
    //m_tag!=tag时，再回过头来循环所有way找最优先被逐出的cache block那就增加了模拟的开销。
    //因此实际上对于所有way中的每一个cache block，只要它不满足line->m_tag!=tag，就在这里判
    //断它能否被逐出。
    // cache block的状态，包含：
    //   INVALID: Cache block有效，但是其中的byte mask=Cache block[mask]状态INVALID，
    //           说明sector缺失。
    //   MODIFIED: 如果Cache block[mask]状态是MODIFIED，说明已经被其他线程修改，如果当
    //             前访问也是写操作的话即为命中，但如果不是写操作则需要判断是否mask标志的
    //             块是否修改完毕，修改完毕则为命中，修改不完成则为SECTOR_MISS。因为L1 
    //             cache与L2 cache写命中时，采用write-back策略，只将数据写入该block，
    //             并不直接更新下级存储，只有当这个块被替换时，才将数据写回下级存储。
    //   VALID: 如果Cache block[mask]状态是VALID，说明已经命中。
    //   RESERVED: 为尚未完成的缓存未命中的数据提供空间。Cache block[mask]状态RESERVED，
    //             说明有其他的线程正在读取这个Cache block。挂起的命中访问已命中处于RE-
    //             SERVED状态的缓存行，意味着同一行上已存在由先前缓存未命中发送的flying
    //             内存请求。
    //line->is_reserved_line()：只要有一个sector是RESERVED，就认为这个Cache Line是RESERVED。
    //这里即整个line没有sector是RESERVED。
    if (!line->is_reserved_line()) {
      // percentage of dirty lines in the cache
      // number of dirty lines / total lines in the cache
      float dirty_line_percentage =
          ((float)m_dirty / (m_config.m_nset * m_config.m_assoc)) * 100;
      // If the cacheline is from a load op (not modified),
      // or the total dirty cacheline is above a specific value,
      // Then this cacheline is eligible to be considered for replacement
      // candidate i.e. Only evict clean cachelines until total dirty cachelines
      // reach the limit.
      //m_config.m_wr_percent在V100中配置为25%。
      //line->is_modified_line()：只要有一个sector是MODIFIED，就认为这个cache line是MODIFIED。
      //这里即整个line没有sector是MODIFIED，或者dirty_line_percentage超过m_wr_percent。
      if (!line->is_modified_line() ||
          dirty_line_percentage >= m_config.m_wr_percent) {
        //一个cache line的状态有：INVALID = 0, RESERVED, VALID, MODIFIED，如果它是VALID，
        //就在上面的代码命中了。
        //因为在逐出一个cache块时，优先逐出一个干净的块，即没有sector被RESERVED，也没有sector
        //被MODIFIED，来逐出；但是如果dirty的cache line的比例超过m_wr_percent（V100中配置为
        //25%），也可以不满足MODIFIED的条件。
        //在缓存管理机制中，优先逐出未被修改（"干净"）的缓存块的策略，是基于几个重要的考虑：
        // 1. 减少写回成本：缓存中的数据通常来源于更低速的后端存储（如主存储器）。当缓存块被修改
        //   （即包含"脏"数据）时，在逐出这些块之前，需要将这些更改写回到后端存储以确保数据一致性。
        //    相比之下，未被修改（"干净"）的缓存块可以直接被逐出，因为它们的内容已经与后端存储一
        //    致，无需进行写回操作。这样就避免了写回操作带来的时间和能量开销。
        // 2. 提高效率：写回操作相对于读取操作来说，是一个成本较高的过程，不仅涉及更多的时间延迟，
        //    还可能占用宝贵的带宽，影响系统的整体性能。通过先逐出那些"干净"的块，系统能够在维持
        //    数据一致性的前提下，减少对后端存储带宽的需求和写回操作的开销。
        // 3. 优化性能：选择逐出"干净"的缓存块还有助于维护缓存的高命中率。理想情况下，缓存应当存
        //    储访问频率高且最近被访问的数据。逐出"脏"数据意味着这些数据需要被写回，这个过程不仅
        //    耗时而且可能导致缓存暂时无法服务其他请求，从而降低缓存效率。
        // 4. 数据安全与完整性：在某些情况下，"脏"缓存块可能表示正在进行的写操作或者重要的数据更
        //    新。通过优先逐出"干净"的缓存块，可以降低因为缓存逐出导致的数据丢失或者完整性破坏的
        //    风险。
        
        //all_reserved被初始化为true，是指所有cache line都没有能够逐出来为新访问提供RESERVE
        //的空间，这里一旦满足上面两个if条件，说明当前line可以被逐出来提供空间供RESERVE新访问，
        //这里all_reserved置为false。而一旦最终all_reserved仍旧保持true的话，就说明当前set里
        //没有哪一个way的cache block可以被逐出，发生RESERVATION_FAIL。
        all_reserved = false;
        //line->is_invalid_line()是所有sector都无效。
        if (line->is_invalid_line()) {
          //当然了，尽管我们有LRU或者FIFO替换策略，但是最理想的情况还是优先替换整个cache block
          //都无效的块。因为这种无效的块不需要写回，能够节省带宽。
          invalid_line = index;
        } else {
          // valid line : keep track of most appropriate replacement candidate
          if (m_config.m_replacement_policy == LRU) {
            //valid_timestamp设置为最近最少被使用的cache line的最末次访问时间。
            //valid_timestamp被初始化为(unsigned)-1，即可以看作无穷大。
            if (line->get_last_access_time() < valid_timestamp) {
              //这里的valid_timestamp是周期数，即最小的周期数具有最大的被逐出优先级，当然这个
              //变量在这里只是找具有最小周期数的cache block，最小周期数意味着离他上次使用才最
              //早，真正标识哪个cache block具有最大优先级被逐出的是valid_line。
              valid_timestamp = line->get_last_access_time();
              //标识当前cache block具有最小的执行周期数，index这个cache block应该最先被逐出。
              valid_line = index;
            }
          } else if (m_config.m_replacement_policy == FIFO) {
            if (line->get_alloc_time() < valid_timestamp) {
              //FIFO按照最早分配时间的cache block最优先被逐出。
              valid_timestamp = line->get_alloc_time();
              valid_line = index;
            }
          }
        }
      }
    } //这里是把当前set里所有的way都循环一遍，如果找到了line->m_tag == tag的块，则已经返回了
      //访问状态，如果没有找到，则也遍历了一遍所有way的cache block，找到了最优先应该被逐出和
      //替换的cache block。
  }
  //Cache访问的状态包含：
  //    HIT，HIT_RESERVED，MISS，RESERVATION_FAIL，SECTOR_MISS，MSHR_HIT六种状态。
  //抛开前面能够确定的HIT，HIT_RESERVED，SECTOR_MISS还能够判断MISS/RESERVATION_FAIL
  //两种状态是否成立。
  //因为在逐出一个cache块时，优先逐出一个干净的块，即没有sector被RESERVED，也没有sector
  //被MODIFIED，来逐出；但是如果dirty的cache line的比例超过m_wr_percent（V100中配置为
  //25%），也可以不满足MODIFIED的条件。
  //all_reserved被初始化为true，是指所有cache line都没有能够逐出来为新访问提供RESERVE
  //的空间，这里一旦满足上面两个if条件，说明cache line可以被逐出来提供空间供RESERVE新访
  //问，这里all_reserved置为false。而一旦最终all_reserved仍旧保持true的话，就说明cache
  //line不可被逐出，发生RESERVATION_FAIL。
  if (all_reserved) {
    //all_reserved为true的话，表明当前set的所有way都没有cache满足被逐出的条件。因此状态
    //返回RESERVATION_FAIL，即all of the blocks in the current set have no enough 
    //space in cache to allocate on miss.
    assert(m_config.m_alloc_policy == ON_MISS);
    return RESERVATION_FAIL;  // miss and not enough space in cache to allocate
                              // on miss
  }

  //如果上面的all_reserved为false，才会到这一步，即cache line可以被逐出来为新访问提供
  //RESERVE。
  if (invalid_line != (unsigned)-1) {
    //尽管我们有LRU或者FIFO替换策略，但是最理想的情况还是优先替换整个cache block都无效
    //的块。因为这种无效的块不需要写回，能够节省带宽。
    idx = invalid_line;
  } else if (valid_line != (unsigned)-1) {
    //没有无效的块，就只能将上面按照LRU或者FIFO确定的cache block作为被逐出的块了。
    idx = valid_line;
  } else
    abort();  // if an unreserved block exists, it is either invalid or
              // replaceable

  //if (probe_mode && m_config.is_streaming()) {
  //  line_table::const_iterator i =
  //      pending_lines.find(m_config.block_addr(addr));
  //  assert(mf);
  //  if (!mf->is_write() && i != pending_lines.end()) {
  //    if (i->second != mf->get_inst().get_uid()) return SECTOR_MISS;
  //  }
  //}

  //如果上面的cache line可以被逐出来reserve新访问，则返回MISS。
  return MISS;
}


/*
更新LRU状态。Least Recently Used。返回是否需要写回wb以及逐出的cache line的信息evicted。
对一个cache进行数据访问的时候，调用data_cache::access()函数：
- 首先cahe会调用m_tag_array->probe()函数，判断对cache的访问（地址为addr，sector mask
  为mask）是HIT/HIT_RESERVED/SECTOR_MISS/MISS/RESERVATION_FAIL等状态。
- 然后调用process_tag_probe()函数，根据cache的配置以及上面m_tag_array->probe()函数返
  回的cache访问状态，执行相应的操作。
  - process_tag_probe()函数中，会根据请求的读写状态，probe()函数返回的cache访问状态，
    执行m_wr_hit/m_wr_miss/m_rd_hit/m_rd_miss函数，他们会调用m_tag_array->access()
    函数来实现LRU状态的更新。
*/
enum cache_request_status tag_array::access(new_addr_type addr, unsigned time,
                                            unsigned &idx, mem_fetch *mf) {
  bool wb = false;
  evicted_block_info evicted;
  enum cache_request_status result = access(addr, time, idx, wb, evicted, mf);
  assert(!wb);
  return result;
}

/*
更新LRU状态。Least Recently Used。返回是否需要写回wb以及逐出的cache line的信息evicted。
对一个cache进行数据访问的时候，调用data_cache::access()函数：
- 首先cahe会调用m_tag_array->probe()函数，判断对cache的访问（地址为addr，sector mask
  为mask）是HIT/HIT_RESERVED/SECTOR_MISS/MISS/RESERVATION_FAIL等状态。
- 然后调用process_tag_probe()函数，根据cache的配置以及上面m_tag_array->probe()函数返
  回的cache访问状态，执行相应的操作。
  - process_tag_probe()函数中，会根据请求的读写状态，probe()函数返回的cache访问状态，
    执行m_wr_hit/m_wr_miss/m_rd_hit/m_rd_miss函数，他们会调用m_tag_array->access()
    函数来实现LRU状态的更新。
*/
enum cache_request_status tag_array::access(new_addr_type addr, unsigned time,
                                            unsigned &idx, bool &wb,
                                            evicted_block_info &evicted,
                                            mem_fetch *mf) {
  // 对当前 tag_array 的访问次数加 1。
  m_access++;
  // 标记当前 tag_array 所属 cache 是否被使用过。一旦有 access() 函数被调用，则
  // 说明被使用过。
  is_used = true;
  shader_cache_access_log(m_core_id, m_type_id, 0);  // log accesses to cache
  // 由于当前函数没有把之前 probe 函数的 cache 访问状态传参进来，这里这个 probe 
  // 单纯的重新获取这个状态。
  enum cache_request_status status = probe(addr, idx, mf, mf->is_write());
  switch (status) {
    // 新访问是 HIT_RESERVED 的话，不执行动作。
    case HIT_RESERVED:
      m_pending_hit++;
    // 新访问是 HIT 的话，设置第 idx 号 cache line 以及 mask 对应的 sector 的最
    // 末此访问时间为当前拍。
    case HIT:
      m_lines[idx]->set_last_access_time(time, mf->get_access_sector_mask());
      break;
    // 新访问是 MISS 的话，说明已经选定 m_lines[idx] 作为逐出并 reserve 新访问的
    // cache line。
    case MISS:
      m_miss++;
      shader_cache_access_log(m_core_id, m_type_id, 1);  // log cache misses
      // For V100, L1 cache and L2 cache are all `allocate on miss`.
      // m_alloc_policy，分配策略：
      //     对于发送到 L1D cache 的请求，如果命中，则立即返回所需数据；如果未命中，
      //     则分配与缓存未命中相关的资源并将请求转至 L2 cache。allocate-on-miss 
      //     和 allocateon-fill 是两种缓存行分配策略。对于 allocateon-miss，需要
      //     为未完成的未命中分配一个缓存行槽、一个 MSHR 和一个未命中队列条目。相比
      //     之下，allocate-on-fill，当未完成的未命中发生时，需要分配一个 MSHR 和
      //     一个未命中队列条目，但当所需数据从较低内存级别返回时，会选择受害者缓存
      //     行槽。在这两种策略中，如果任何所需资源不可用，则会发生预留失败，内存管
      //     道会停滞。分配的 MSHR 会被保留，直到从 L2 缓存/片外内存中获取数据，而
      //     未命中队列条目会在未命中请求转发到 L2 缓存后被释放。由于 allocate-on-
      //     fill 在驱逐之前将受害者缓存行保留在缓存中更长时间，并为未完成的未命中
      //     保留更少的资源，因此它往往能获得更多的缓存命中和更少的预留失败，从而比 
      //     allocate-on-miss 具有更好的性能。尽管填充时分配需要额外的缓冲和流控制
      //     逻辑来按顺序将数据填充到缓存中，但按顺序执行模型和写入驱逐策略使 GPU 
      //     L1D 缓存对填充时分配很友好，因为在填充时要驱逐受害者缓存时，没有脏数据
      //     写入 L2。
      //     详见 paper：
      //     The Demand for a Sound Baseline in GPU Memory Architecture Research. 
      //     https://hzhou.wordpress.ncsu.edu/files/2022/12/Hongwen_WDDD2017.pdf
      //
      //     For streaming cache: (1) we set the alloc policy to be on-fill 
      //     to remove all line_alloc_fail stalls. if the whole memory is 
      //     allocated to the L1 cache, then make the allocation to be on 
      //     MISS, otherwise, make it ON_FILL to eliminate line allocation 
      //     fails. i.e. MSHR throughput is the same, independent on the L1
      //     cache size/associativity So, we set the allocation policy per 
      //     kernel basis, see shader.cc, max_cta() function. (2) We also 
      //     set the MSHRs to be equal to max allocated cache lines. This
      //     is possible by moving TAG to be shared between cache line and 
      //     MSHR enrty (i.e. for each cache line, there is an MSHR entry 
      //     associated with it). This is the easiest think we can think of 
      //     to model (mimic) L1 streaming cache in Pascal and Volta. For 
      //     more information about streaming cache, see: 
      //     https://www2.maths.ox.ac.uk/~gilesm/cuda/lecs/VoltaAG_Oxford.pdf
      //     https://ieeexplore.ieee.org/document/8344474/
      if (m_config.m_alloc_policy == ON_MISS) {
        // 访问时遇到 MISS，说明 probe 确定的 idx 号 cache line 需要被逐出来为新
        // 访问提供 RESERVE 的空间。但是，这里需要判断 idx 号 cache line 是否是 
        // MODIFIED，如果是的话，需要执行写回，设置写回的标志为 wb = true，设置逐
        // 出 cache line 的信息。
        if (m_lines[idx]->is_modified_line()) {
          // m_lines[idx] 作为逐出并 reserve 新访问的 cache line，如果它的某个 
          // sector 已经被 MODIFIED，则需要执行写回操作，设置写回标志为 wb = true，
          // 设置逐出 cache line 的信息。
          wb = true;
          // m_lines[idx]->set_byte_mask(mf);
          evicted.set_info(m_lines[idx]->m_block_addr,
                           m_lines[idx]->get_modified_size(),
                           m_lines[idx]->get_dirty_byte_mask(),
                           m_lines[idx]->get_dirty_sector_mask());
          // 由于执行写回操作，MODIFIED 造成的 m_dirty 数量应该减1。
          m_dirty--;
        }
        // 执行对新访问的 reserve 操作。
        m_lines[idx]->allocate(m_config.tag(addr), m_config.block_addr(addr),
                               time, mf->get_access_sector_mask());
      }
      break;
    // Cache block 有效，但是其中的 byte mask = Cache block[mask] 状态无效，说明
    // sector 缺失。
    case SECTOR_MISS:
      assert(m_config.m_cache_type == SECTOR);
      m_sector_miss++;
      shader_cache_access_log(m_core_id, m_type_id, 1);  // log cache misses
      // For V100, L1 cache and L2 cache are all `allocate on miss`.
      if (m_config.m_alloc_policy == ON_MISS) {
        bool before = m_lines[idx]->is_modified_line();
        // 设置 m_lines[idx] 为新访问分配一个 sector。
        ((sector_cache_block *)m_lines[idx])
            ->allocate_sector(time, mf->get_access_sector_mask());
        if (before && !m_lines[idx]->is_modified_line()) {
          m_dirty--;
        }
      }
      break;
    // probe函数中：
    // all_reserved 被初始化为 true，是指所有 cache line 都没有能够逐出来为新访问
    // 提供 RESERVE 的空间，这里一旦满足函数两个 if 条件，说明 cache line 可以被逐
    // 出来提供空间供 RESERVE 新访问，这里 all_reserved 置为 false。
    // 而一旦最终 all_reserved 仍旧保持 true 的话，就说明 cache line 不可被逐出，
    // 发生 RESERVATION_FAIL。因此这里不执行任何操作。
    case RESERVATION_FAIL:
      m_res_fail++;
      shader_cache_access_log(m_core_id, m_type_id, 1);  // log cache misses
      break;
    default:
      fprintf(stderr,
              "tag_array::access - Error: Unknown"
              "cache_request_status %d\n",
              status);
      abort();
  }
  return status;
}

void tag_array::fill(new_addr_type addr, unsigned time, mem_fetch *mf,
                     bool is_write) {
  fill(addr, time, mf->get_access_sector_mask(), mf->get_access_byte_mask(),
       is_write);
}

void tag_array::fill(new_addr_type addr, unsigned time,
                     mem_access_sector_mask_t mask,
                     mem_access_byte_mask_t byte_mask, bool is_write) {
  // assert( m_config.m_alloc_policy == ON_FILL );
  unsigned idx;
  enum cache_request_status status = probe(addr, idx, mask, is_write);

  if (status == RESERVATION_FAIL) {
    return;
  }

  bool before = m_lines[idx]->is_modified_line();
  // assert(status==MISS||status==SECTOR_MISS); // MSHR should have prevented
  // redundant memory request
  if (status == MISS) {
    m_lines[idx]->allocate(m_config.tag(addr), m_config.block_addr(addr), time,
                           mask);
  } else if (status == SECTOR_MISS) {
    assert(m_config.m_cache_type == SECTOR);
    ((sector_cache_block *)m_lines[idx])->allocate_sector(time, mask);
  }
  if (before && !m_lines[idx]->is_modified_line()) {
    m_dirty--;
  }
  before = m_lines[idx]->is_modified_line();
  m_lines[idx]->fill(time, mask, byte_mask);
  if (m_lines[idx]->is_modified_line() && !before) {
    m_dirty++;
  }
}

void tag_array::fill(unsigned index, unsigned time, mem_fetch *mf) {
  //在V100中，L1 cache与L2 cache均为allocate on miss。
  //allocate-on-miss（简写为on-miss）和allocate-on-fill（简写为on-fill）是两种cache行分配策略：
  //(1) allocate-on-miss：当发生未完成的cache miss时，需要为未完成的miss分配cache line slot、
  //    MSHR和miss队列条目。
  //(2) allocateon-on-fill：当发生未完成的cache miss时，需要为未完成的miss分配MSHR和miss队列条
  //    目，但当所需数据从较低内存级别返回时，会选择受害者cache line slot替换。
  //在这两种策略中，如果任何所需资源不可用，则会发生reservation failure，并且内存流水线停顿。分配
  //的MSHR被保留，直到数据从较低一级内存中取回；而一旦miss请求转发到L2 cache，则释放miss队列条目。
  //allocateon-on-fill往往比allocate-on-miss有更好的性能，因为它保留了受害者cache更长的时间，并
  //且为未完成的miss保留更少的资源，从而享受更多的cache hit和更少的reservation failure。尽管all-
  //ocateon-on-fill需要额外的buffer和流控逻辑来将数据按顺序填充到cache，但按顺序执行模型和write-
  //evict策略使得GPU L1 D-cache对allocateon-on-fill很友好，因为当受害者cache在被evict时，没有脏
  //数据被写入L2。
  assert(m_config.m_alloc_policy == ON_MISS);
  //before是记录在填充之前，m_lines[index]是否是MODIFIED状态。
  bool before = m_lines[index]->is_modified_line();
  m_lines[index]->fill(time, mf->get_access_sector_mask(),
                       mf->get_access_byte_mask());
  //如果填充以后，m_lines[index]是 MODIFIED 状态，但是在填充之前不是 MODIFIED 状态，说明这个cache 
  //line是脏的，m_dirty数量加1。
  if (m_lines[index]->is_modified_line() && !before) {
    m_dirty++;
  }
}

// TODO: we need write back the flushed data to the upper level
void tag_array::flush() {
  if (!is_used) return;

  for (unsigned i = 0; i < m_config.get_num_lines(); i++)
    if (m_lines[i]->is_modified_line()) {
      for (unsigned j = 0; j < SECTOR_CHUNCK_SIZE; j++) {
        m_lines[i]->set_status(INVALID, mem_access_sector_mask_t().set(j));
      }
    }

  m_dirty = 0;
  is_used = false;
}

void tag_array::invalidate() {
  if (!is_used) return;

  for (unsigned i = 0; i < m_config.get_num_lines(); i++)
    for (unsigned j = 0; j < SECTOR_CHUNCK_SIZE; j++)
      m_lines[i]->set_status(INVALID, mem_access_sector_mask_t().set(j));

  m_dirty = 0;
  is_used = false;
}

float tag_array::windowed_miss_rate() const {
  unsigned n_access = m_access - m_prev_snapshot_access;
  unsigned n_miss = (m_miss + m_sector_miss) - m_prev_snapshot_miss;
  // unsigned n_pending_hit = m_pending_hit - m_prev_snapshot_pending_hit;

  float missrate = 0.0f;
  if (n_access != 0) missrate = (float)(n_miss + m_sector_miss) / n_access;
  return missrate;
}

void tag_array::new_window() {
  m_prev_snapshot_access = m_access;
  m_prev_snapshot_miss = m_miss;
  m_prev_snapshot_miss = m_miss + m_sector_miss;
  m_prev_snapshot_pending_hit = m_pending_hit;
}

void tag_array::print(FILE *stream, unsigned &total_access,
                      unsigned &total_misses) const {
  m_config.print(stream);
  fprintf(stream,
          "\t\tAccess = %d, Miss = %d, Sector_Miss = %d, Total_Miss = %d "
          "(%.3g), PendingHit = %d (%.3g)\n",
          m_access, m_miss, m_sector_miss, (m_miss + m_sector_miss),
          (float)(m_miss + m_sector_miss) / m_access, m_pending_hit,
          (float)m_pending_hit / m_access);
  total_misses += (m_miss + m_sector_miss);
  total_access += m_access;
}

void tag_array::get_stats(unsigned &total_access, unsigned &total_misses,
                          unsigned &total_hit_res,
                          unsigned &total_res_fail) const {
  // Update statistics from the tag array
  total_access = m_access;
  total_misses = (m_miss + m_sector_miss);
  total_hit_res = m_pending_hit;
  total_res_fail = m_res_fail;
}

/*
判断一系列的访问cache事件是否存在WRITE_REQUEST_SENT。
缓存事件类型包括：
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
*/
bool was_write_sent(const std::list<cache_event> &events) {
  for (std::list<cache_event>::const_iterator e = events.begin();
       e != events.end(); e++) {
    if ((*e).m_cache_event_type == WRITE_REQUEST_SENT) return true;
  }
  return false;
}

/*
判断一系列的访问cache事件是否存在WRITE_BACK_REQUEST_SENT。
缓存事件类型包括：
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
*/
bool was_writeback_sent(const std::list<cache_event> &events,
                        cache_event &wb_event) {
  for (std::list<cache_event>::const_iterator e = events.begin();
       e != events.end(); e++) {
    if ((*e).m_cache_event_type == WRITE_BACK_REQUEST_SENT) {
      wb_event = *e;
      return true;
    }
  }
  return false;
}

/*
判断一系列的访问cache事件是否存在READ_REQUEST_SENT。
缓存事件类型包括：
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
*/
bool was_read_sent(const std::list<cache_event> &events) {
  for (std::list<cache_event>::const_iterator e = events.begin();
       e != events.end(); e++) {
    if ((*e).m_cache_event_type == READ_REQUEST_SENT) return true;
  }
  return false;
}

/*
判断一系列的访问cache事件是否存在WRITE_ALLOCATE_SENT。
缓存事件类型包括：
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
*/
bool was_writeallocate_sent(const std::list<cache_event> &events) {
  for (std::list<cache_event>::const_iterator e = events.begin();
       e != events.end(); e++) {
    if ((*e).m_cache_event_type == WRITE_ALLOCATE_SENT) return true;
  }
  return false;
}
/****************************************************************** MSHR
 * ******************************************************************/

/// Checks if there is a pending request to the lower memory level already
// 检查是否已存在对较低内存级别的挂起请求。这里实际上是MSHR查找是否已经有block_addr的请求被合并到MSHR。
// typedef new_addr_type unsigned long long.
bool mshr_table::probe(new_addr_type block_addr) const {
  //MSHR表中的数据为std::unordered_map，是<new_addr_type, mshr_entry>的无序map。地址block_addr
  //去查找它是否在表中，如果 a = m_data.end()，则说明表中没有 block_addr；反之，则存在该条目。如果
  //不存在该条目，则返回false；如果存在该条目，返回true，代表存在对较低内存级别的挂起请求。
  table::const_iterator a = m_data.find(block_addr);
  return a != m_data.end();
}

/// Checks if there is space for tracking a new memory access
// 检查是否有空间处理新的内存访问。如果mshr_addr在MSHR中已存在条目，m_mshrs.full检查是否该条目的合并数
// 量已达到最大合并数；如果mshr_addr在MSHR中不存在条目，则检查是否有空闲的MSHR条目可以将mshr_addr插入进MSHR。
bool mshr_table::full(new_addr_type block_addr) const {
  //首先查找是否MSHR表中有 block_addr 地址的条目。
  table::const_iterator i = m_data.find(block_addr);
  if (i != m_data.end())
    //如果存在该条目，看是否有空间合并进该条目。
    return i->second.m_list.size() >= m_max_merged;
  else
    //如果不存在该条目，看是否有其他空闲条目添加。
    return m_data.size() >= m_num_entries;
}

/// Add or merge this access
// 添加或合并此访问。这里假设的是MSHR表中有 block_addr 地址的条目，直接向该条目中添加。
void mshr_table::add(new_addr_type block_addr, mem_fetch *mf) {
  //将 block_addr 地址加入到对应条目内。
  m_data[block_addr].m_list.push_back(mf);
  assert(m_data.size() <= m_num_entries);
  assert(m_data[block_addr].m_list.size() <= m_max_merged);
  // indicate that this MSHR entry contains an atomic operation
  //指示此MSHR条目包含原子操作。
  if (mf->isatomic()) {
    //mem_fetch定义了一个模拟内存请求的通信结构。更像是一个内存请求的行为。如果 mf 代表的内存访问是
    //原子操作，设置原子操作标志位。
    m_data[block_addr].m_has_atomic = true;
  }
}

/// check is_read_after_write_pending
// 检查是否存在挂起的写后读请求。这里假设的是MSHR表中有 block_addr 地址的条目。
bool mshr_table::is_read_after_write_pending(new_addr_type block_addr) {
  std::list<mem_fetch *> my_list = m_data[block_addr].m_list;
  bool write_found = false;
  //在block_addr条目中，查找所有的mem_fetch行为。
  for (std::list<mem_fetch *>::iterator it = my_list.begin();
       it != my_list.end(); ++it) {
    //如果(*it)->is_write()为真，代表it是写行为，写请求正处于挂起状态。
    if ((*it)->is_write())  // Pending Write Request
      write_found = true;
    //如果当前(*it)不是写行为，是读行为，但是write_found又为true，则之前有一个对 block_addr 地址
    //的写行为，因此存在对 block_addr 地址的写后读行为被挂起。
    else if (write_found)  // Pending Read Request and we found previous Write
      return true;
  }

  return false;
}

/// Accept a new cache fill response: mark entry ready for processing
// 接受新的缓存填充响应：标记条目以备处理。这里假设的是MSHR表中有 block_addr 地址的条目。
void mshr_table::mark_ready(new_addr_type block_addr, bool &has_atomic) {
  //busy（）始终返回false，此句无效。
  assert(!busy());
  //查找 block_addr 地址对应的条目。
  //m_data是MSHR的条目表，是<new_addr_type, mshr_entry>的map。
  table::iterator a = m_data.find(block_addr);
  assert(a != m_data.end());
  //当前mark_ready函数会在数据包fill到cache中时调用，当一个数据包到了后，跟随数据包返回的数据已
  //经就绪，因此将其地址加入到m_current_response中。
  //将对 block_addr 地址的访问合并到就绪内存访问列表中。m_current_response是就绪内存访问的列表。
  //m_current_response仅存储了就绪内存访问的地址。
  m_current_response.push_back(block_addr);
  //设置原子标志位。
  has_atomic = a->second.m_has_atomic;
  assert(m_current_response.size() <= m_data.size());
}

/// Returns next ready access
// 返回一个已经填入的就绪访问。通常配合 access_ready() 一起使用，access_ready 用来检查
// 是否存在就绪访问，next_access() 用来返回就绪访问：
//   bool access_ready() const { return !m_current_response.empty(); }
mem_fetch *mshr_table::next_access() {
  // access_ready() 的功能是如果存在就绪访问，则返回 true。这里是假定存在就绪内存访
  // 问。
  assert(access_ready());
  // 返回就绪内存访问列表的首个条目的条目地址。m_current_response 是就绪内存访问的列
  // 表。m_current_response 仅存储了就绪内存访问的地址。
  // 数据包 fill 到 cache 中时，即一个数据包到了后，跟随数据包返回的数据已经就绪，因
  // 此将其 block 地址加入到 m_current_response 中代表这个 block 已经有数据就绪了。
  // 也就是说，在 m_current_response 中存储了已有数据的 block 地址。
  new_addr_type block_addr = m_current_response.front();
  /* m_list 是 mshr_entry 的一个成员，mshr_entry 是一个内存访问请求的列表。m_data 
     是 MSHR 的条目表，是 <new_addr_type, mshr_entry> 的 map。m_data[block_addr]
     因此就是一个 mshr_entry，m_data[block_addr].m_list 是对应 block_addr 这个地
     址的内存访问请求的列表。m_data[block_addr].m_list.front() 是这个列表的首个请
     求。
        struct mshr_entry {
          // 单个条目中可以合并的内存访问请求。
          std::list<mem_fetch *> m_list;
          // 单个条目是否是原子操作。
          bool m_has_atomic;
          mshr_entry() : m_has_atomic(false) {}
        };
  */
  assert(!m_data[block_addr].m_list.empty());
  // 返回 block_addr 的合并的内存访问行为的首个请求，mem_fetch=m_list.front()。
  mem_fetch *result = m_data[block_addr].m_list.front();
  // 将合并的内存访问行为的首个请求从列表里 pop 出去。
  m_data[block_addr].m_list.pop_front();
  // 这里需要注意的是，m_data[block_addr].m_list 存储了 block_addr 地址的内存访问
  // 返回的数据列表，当这个列表为空时，说明这个 block_addr 地址的所有内存访问行为已
  // 经就绪，因此可以将 block_addr 从 m_current_response 中 pop 出去了。
  if (m_data[block_addr].m_list.empty()) {
    // 在将合并的内存访问行为的首个请求从列表里 pop 出去后，列表如果变空即该条目失效，
    // 需要擦除该条目。
    // release entry
    m_data.erase(block_addr);
    // 下一个就绪访问得到后，就绪内存访问列表中把该次就绪访问的地址 pop 出去。
    // m_current_response 仅存储了就绪内存访问的地址。
    m_current_response.pop_front();
  }
  return result;
}

void mshr_table::display(FILE *fp) const {
  fprintf(fp, "MSHR contents\n");
  for (table::const_iterator e = m_data.begin(); e != m_data.end(); ++e) {
    unsigned block_addr = e->first;
    fprintf(fp, "MSHR: tag=0x%06x, atomic=%d %zu entries : ", block_addr,
            e->second.m_has_atomic, e->second.m_list.size());
    if (!e->second.m_list.empty()) {
      mem_fetch *mf = e->second.m_list.front();
      fprintf(fp, "%p :", mf);
      mf->print(fp);
    } else {
      fprintf(fp, " no memory requests???\n");
    }
  }
}
/***************************************************************** Caches
 * *****************************************************************/
cache_stats::cache_stats() {
  m_cache_port_available_cycles = 0;
  m_cache_data_port_busy_cycles = 0;
  m_cache_fill_port_busy_cycles = 0;
}

void cache_stats::clear() {
  ///
  /// Zero out all current cache statistics
  ///
  m_stats.clear();
  m_stats_pw.clear();
  m_fail_stats.clear();

  m_cache_port_available_cycles = 0;
  m_cache_data_port_busy_cycles = 0;
  m_cache_fill_port_busy_cycles = 0;
}

void cache_stats::clear_pw() {
  ///
  /// Zero out per-window cache statistics
  ///
  m_stats_pw.clear();
}

void cache_stats::inc_stats(int access_type, int access_outcome,
                            unsigned long long streamID) {
  ///
  /// Increment the stat corresponding to (access_type, access_outcome) by 1.
  ///
  if (!check_valid(access_type, access_outcome))
    assert(0 && "Unknown cache access type or access outcome");

  if (m_stats.find(streamID) == m_stats.end()) {
    std::vector<std::vector<unsigned long long>> new_val;
    new_val.resize(NUM_MEM_ACCESS_TYPE);
    for (unsigned j = 0; j < NUM_MEM_ACCESS_TYPE; ++j) {
      new_val[j].resize(NUM_CACHE_REQUEST_STATUS, 0);
    }
    m_stats.insert(std::pair<unsigned long long,
                             std::vector<std::vector<unsigned long long>>>(
        streamID, new_val));
  }
  m_stats.at(streamID)[access_type][access_outcome]++;
}

void cache_stats::inc_stats_pw(int access_type, int access_outcome,
                               unsigned long long streamID) {
  ///
  /// Increment the corresponding per-window cache stat
  ///
  if (!check_valid(access_type, access_outcome))
    assert(0 && "Unknown cache access type or access outcome");

  if (m_stats_pw.find(streamID) == m_stats_pw.end()) {
    std::vector<std::vector<unsigned long long>> new_val;
    new_val.resize(NUM_MEM_ACCESS_TYPE);
    for (unsigned j = 0; j < NUM_MEM_ACCESS_TYPE; ++j) {
      new_val[j].resize(NUM_CACHE_REQUEST_STATUS, 0);
    }
    m_stats_pw.insert(std::pair<unsigned long long,
                                std::vector<std::vector<unsigned long long>>>(
        streamID, new_val));
  }
  m_stats_pw.at(streamID)[access_type][access_outcome]++;
}

void cache_stats::inc_fail_stats(int access_type, int fail_outcome,
                                 unsigned long long streamID) {
  if (!check_fail_valid(access_type, fail_outcome))
    assert(0 && "Unknown cache access type or access fail");

  if (m_fail_stats.find(streamID) == m_fail_stats.end()) {
    std::vector<std::vector<unsigned long long>> new_val;
    new_val.resize(NUM_MEM_ACCESS_TYPE);
    for (unsigned j = 0; j < NUM_MEM_ACCESS_TYPE; ++j) {
      new_val[j].resize(NUM_CACHE_RESERVATION_FAIL_STATUS, 0);
    }
    m_fail_stats.insert(std::pair<unsigned long long,
                                  std::vector<std::vector<unsigned long long>>>(
        streamID, new_val));
  }
  m_fail_stats.at(streamID)[access_type][fail_outcome]++;
}

enum cache_request_status cache_stats::select_stats_status(
    enum cache_request_status probe, enum cache_request_status access) const {
  ///
  /// This function selects how the cache access outcome should be counted.
  /// HIT_RESERVED is considered as a MISS in the cores, however, it should be
  /// counted as a HIT_RESERVED in the caches.
  ///
  if (probe == HIT_RESERVED && access != RESERVATION_FAIL)
    return probe;
  else if (probe == SECTOR_MISS && access == MISS)
    return probe;
  else
    return access;
}

unsigned long long &cache_stats::operator()(int access_type, int access_outcome,
                                            bool fail_outcome,
                                            unsigned long long streamID) {
  ///
  /// Simple method to read/modify the stat corresponding to (access_type,
  /// access_outcome) Used overloaded () to avoid the need for separate
  /// read/write member functions
  ///
  if (fail_outcome) {
    if (!check_fail_valid(access_type, access_outcome))
      assert(0 && "Unknown cache access type or fail outcome");

    return m_fail_stats.at(streamID)[access_type][access_outcome];
  } else {
    if (!check_valid(access_type, access_outcome))
      assert(0 && "Unknown cache access type or access outcome");

    return m_stats.at(streamID)[access_type][access_outcome];
  }
}

unsigned long long cache_stats::operator()(int access_type, int access_outcome,
                                           bool fail_outcome,
                                           unsigned long long streamID) const {
  ///
  /// Const accessor into m_stats.
  ///
  if (fail_outcome) {
    if (!check_fail_valid(access_type, access_outcome))
      assert(0 && "Unknown cache access type or fail outcome");

    return m_fail_stats.at(streamID)[access_type][access_outcome];
  } else {
    if (!check_valid(access_type, access_outcome))
      assert(0 && "Unknown cache access type or access outcome");

    return m_stats.at(streamID)[access_type][access_outcome];
  }
}

cache_stats cache_stats::operator+(const cache_stats &cs) {
  ///
  /// Overloaded + operator to allow for simple stat accumulation
  ///
  cache_stats ret;
  for (auto iter = m_stats.begin(); iter != m_stats.end(); ++iter) {
    unsigned long long streamID = iter->first;
    ret.m_stats.insert(std::pair<unsigned long long,
                                 std::vector<std::vector<unsigned long long>>>(
        streamID, m_stats.at(streamID)));
  }
  for (auto iter = m_stats_pw.begin(); iter != m_stats_pw.end(); ++iter) {
    unsigned long long streamID = iter->first;
    ret.m_stats_pw.insert(
        std::pair<unsigned long long,
                  std::vector<std::vector<unsigned long long>>>(
            streamID, m_stats_pw.at(streamID)));
  }
  for (auto iter = m_fail_stats.begin(); iter != m_fail_stats.end(); ++iter) {
    unsigned long long streamID = iter->first;
    ret.m_fail_stats.insert(
        std::pair<unsigned long long,
                  std::vector<std::vector<unsigned long long>>>(
            streamID, m_fail_stats.at(streamID)));
  }
  for (auto iter = cs.m_stats.begin(); iter != cs.m_stats.end(); ++iter) {
    unsigned long long streamID = iter->first;
    if (ret.m_stats.find(streamID) == ret.m_stats.end()) {
      ret.m_stats.insert(
          std::pair<unsigned long long,
                    std::vector<std::vector<unsigned long long>>>(
              streamID, cs.m_stats.at(streamID)));
    } else {
      for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
        for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status) {
          ret.m_stats.at(streamID)[type][status] +=
              cs(type, status, false, streamID);
        }
      }
    }
  }
  for (auto iter = cs.m_stats_pw.begin(); iter != cs.m_stats_pw.end(); ++iter) {
    unsigned long long streamID = iter->first;
    if (ret.m_stats_pw.find(streamID) == ret.m_stats_pw.end()) {
      ret.m_stats_pw.insert(
          std::pair<unsigned long long,
                    std::vector<std::vector<unsigned long long>>>(
              streamID, cs.m_stats_pw.at(streamID)));
    } else {
      for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
        for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status) {
          ret.m_stats_pw.at(streamID)[type][status] +=
              cs(type, status, false, streamID);
        }
      }
    }
  }
  for (auto iter = cs.m_fail_stats.begin(); iter != cs.m_fail_stats.end();
       ++iter) {
    unsigned long long streamID = iter->first;
    if (ret.m_fail_stats.find(streamID) == ret.m_fail_stats.end()) {
      ret.m_fail_stats.insert(
          std::pair<unsigned long long,
                    std::vector<std::vector<unsigned long long>>>(
              streamID, cs.m_fail_stats.at(streamID)));
    } else {
      for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
        for (unsigned status = 0; status < NUM_CACHE_RESERVATION_FAIL_STATUS;
             ++status) {
          ret.m_fail_stats.at(streamID)[type][status] +=
              cs(type, status, true, streamID);
        }
      }
    }
  }
  ret.m_cache_port_available_cycles =
      m_cache_port_available_cycles + cs.m_cache_port_available_cycles;
  ret.m_cache_data_port_busy_cycles =
      m_cache_data_port_busy_cycles + cs.m_cache_data_port_busy_cycles;
  ret.m_cache_fill_port_busy_cycles =
      m_cache_fill_port_busy_cycles + cs.m_cache_fill_port_busy_cycles;
  return ret;
}

cache_stats &cache_stats::operator+=(const cache_stats &cs) {
  ///
  /// Overloaded += operator to allow for simple stat accumulation
  ///
  for (auto iter = cs.m_stats.begin(); iter != cs.m_stats.end(); ++iter) {
    unsigned long long streamID = iter->first;
    if (m_stats.find(streamID) == m_stats.end()) {
      m_stats.insert(std::pair<unsigned long long,
                               std::vector<std::vector<unsigned long long>>>(
          streamID, cs.m_stats.at(streamID)));
    } else {
      for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
        for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status) {
          m_stats.at(streamID)[type][status] +=
              cs(type, status, false, streamID);
        }
      }
    }
  }
  for (auto iter = cs.m_stats_pw.begin(); iter != cs.m_stats_pw.end(); ++iter) {
    unsigned long long streamID = iter->first;
    if (m_stats_pw.find(streamID) == m_stats_pw.end()) {
      m_stats_pw.insert(std::pair<unsigned long long,
                                  std::vector<std::vector<unsigned long long>>>(
          streamID, cs.m_stats_pw.at(streamID)));
    } else {
      for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
        for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status) {
          m_stats_pw.at(streamID)[type][status] +=
              cs(type, status, false, streamID);
        }
      }
    }
  }
  for (auto iter = cs.m_fail_stats.begin(); iter != cs.m_fail_stats.end();
       ++iter) {
    unsigned long long streamID = iter->first;
    if (m_fail_stats.find(streamID) == m_fail_stats.end()) {
      m_fail_stats.insert(
          std::pair<unsigned long long,
                    std::vector<std::vector<unsigned long long>>>(
              streamID, cs.m_fail_stats.at(streamID)));
    } else {
      for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
        for (unsigned status = 0; status < NUM_CACHE_RESERVATION_FAIL_STATUS;
             ++status) {
          m_fail_stats.at(streamID)[type][status] +=
              cs(type, status, true, streamID);
        }
      }
    }
  }
  m_cache_port_available_cycles += cs.m_cache_port_available_cycles;
  m_cache_data_port_busy_cycles += cs.m_cache_data_port_busy_cycles;
  m_cache_fill_port_busy_cycles += cs.m_cache_fill_port_busy_cycles;
  return *this;
}

void cache_stats::print_stats(FILE *fout, unsigned long long streamID,
                              const char *cache_name) const {
  ///
  /// For a given CUDA stream, print out each non-zero cache statistic for every
  /// memory access type and status "cache_name" defaults to "Cache_stats" when
  /// no argument is provided, otherwise the provided name is used. The printed
  /// format is
  /// "<cache_name>[<request_type>][<request_status>] = <stat_value>"
  /// Specify streamID to be -1 to print every stream.

  std::vector<unsigned> total_access;
  std::string m_cache_name = cache_name;
  for (auto iter = m_stats.begin(); iter != m_stats.end(); ++iter) {
    unsigned long long streamid = iter->first;
    // when streamID is specified, skip stats for all other streams, otherwise,
    // print stats from all streams
    if ((streamID != -1) && (streamid != streamID)) continue;
    total_access.clear();
    total_access.resize(NUM_MEM_ACCESS_TYPE, 0);
    for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
      for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status) {
        fprintf(fout, "\t%s[%s][%s] = %llu\n", m_cache_name.c_str(),
                mem_access_type_str((enum mem_access_type)type),
                cache_request_status_str((enum cache_request_status)status),
                m_stats.at(streamid)[type][status]);

        if (status != RESERVATION_FAIL && status != MSHR_HIT)
          // MSHR_HIT is a special type of SECTOR_MISS
          // so its already included in the SECTOR_MISS
          total_access[type] += m_stats.at(streamid)[type][status];
      }
    }
    for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
      if (total_access[type] > 0)
        fprintf(fout, "\t%s[%s][%s] = %u\n", m_cache_name.c_str(),
                mem_access_type_str((enum mem_access_type)type), "TOTAL_ACCESS",
                total_access[type]);
    }
  }
}

void cache_stats::print_fail_stats(FILE *fout, unsigned long long streamID,
                                   const char *cache_name) const {
  std::string m_cache_name = cache_name;
  for (auto iter = m_fail_stats.begin(); iter != m_fail_stats.end(); ++iter) {
    unsigned long long streamid = iter->first;
    // when streamID is specified, skip stats for all other streams, otherwise,
    // print stats from all streams
    if ((streamID != -1) && (streamid != streamID)) continue;
    for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
      for (unsigned fail = 0; fail < NUM_CACHE_RESERVATION_FAIL_STATUS;
           ++fail) {
        if (m_fail_stats.at(streamid)[type][fail] > 0) {
          fprintf(
              fout, "\t%s[%s][%s] = %llu\n", m_cache_name.c_str(),
              mem_access_type_str((enum mem_access_type)type),
              cache_fail_status_str((enum cache_reservation_fail_reason)fail),
              m_fail_stats.at(streamid)[type][fail]);
        }
      }
    }
  }
}

void cache_sub_stats::print_port_stats(FILE *fout,
                                       const char *cache_name) const {
  float data_port_util = 0.0f;
  if (port_available_cycles > 0) {
    data_port_util = (float)data_port_busy_cycles / port_available_cycles;
  }
  fprintf(fout, "%s_data_port_util = %.3f\n", cache_name, data_port_util);
  float fill_port_util = 0.0f;
  if (port_available_cycles > 0) {
    fill_port_util = (float)fill_port_busy_cycles / port_available_cycles;
  }
  fprintf(fout, "%s_fill_port_util = %.3f\n", cache_name, fill_port_util);
}

unsigned long long cache_stats::get_stats(
    enum mem_access_type *access_type, unsigned num_access_type,
    enum cache_request_status *access_status,
    unsigned num_access_status) const {
  ///
  /// Returns a sum of the stats corresponding to each "access_type" and
  /// "access_status" pair. "access_type" is an array of "num_access_type"
  /// mem_access_types. "access_status" is an array of "num_access_status"
  /// cache_request_statuses.
  ///
  unsigned long long total = 0;
  for (auto iter = m_stats.begin(); iter != m_stats.end(); ++iter) {
    unsigned long long streamID = iter->first;
    for (unsigned type = 0; type < num_access_type; ++type) {
      for (unsigned status = 0; status < num_access_status; ++status) {
        if (!check_valid((int)access_type[type], (int)access_status[status]))
          assert(0 && "Unknown cache access type or access outcome");
        total += m_stats.at(streamID)[access_type[type]][access_status[status]];
      }
    }
  }
  return total;
}

void cache_stats::get_sub_stats(struct cache_sub_stats &css) const {
  ///
  /// Overwrites "css" with the appropriate statistics from this cache.
  ///
  struct cache_sub_stats t_css;
  t_css.clear();

  for (auto iter = m_stats.begin(); iter != m_stats.end(); ++iter) {
    unsigned long long streamID = iter->first;
    for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
      for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status) {
        if (status == HIT || status == MISS || status == SECTOR_MISS ||
            status == HIT_RESERVED)
          t_css.accesses += m_stats.at(streamID)[type][status];

        if (status == MISS || status == SECTOR_MISS)
          t_css.misses += m_stats.at(streamID)[type][status];

        if (status == HIT_RESERVED)
          t_css.pending_hits += m_stats.at(streamID)[type][status];

        if (status == RESERVATION_FAIL)
          t_css.res_fails += m_stats.at(streamID)[type][status];
      }
    }
  }

  t_css.port_available_cycles = m_cache_port_available_cycles;
  t_css.data_port_busy_cycles = m_cache_data_port_busy_cycles;
  t_css.fill_port_busy_cycles = m_cache_fill_port_busy_cycles;

  css = t_css;
}

void cache_stats::get_sub_stats_pw(struct cache_sub_stats_pw &css) const {
  ///
  /// Overwrites "css" with the appropriate statistics from this cache.
  ///
  struct cache_sub_stats_pw t_css;
  t_css.clear();

  for (auto iter = m_stats_pw.begin(); iter != m_stats_pw.end(); ++iter) {
    unsigned long long streamID = iter->first;
    for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
      for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status) {
        if (status == HIT || status == MISS || status == SECTOR_MISS ||
            status == HIT_RESERVED)
          t_css.accesses += m_stats_pw.at(streamID)[type][status];

        if (status == HIT) {
          if (type == GLOBAL_ACC_R || type == CONST_ACC_R ||
              type == INST_ACC_R) {
            t_css.read_hits += m_stats_pw.at(streamID)[type][status];
          } else if (type == GLOBAL_ACC_W) {
            t_css.write_hits += m_stats_pw.at(streamID)[type][status];
          }
        }

        if (status == MISS || status == SECTOR_MISS) {
          if (type == GLOBAL_ACC_R || type == CONST_ACC_R ||
              type == INST_ACC_R) {
            t_css.read_misses += m_stats_pw.at(streamID)[type][status];
          } else if (type == GLOBAL_ACC_W) {
            t_css.write_misses += m_stats_pw.at(streamID)[type][status];
          }
        }

        if (status == HIT_RESERVED) {
          if (type == GLOBAL_ACC_R || type == CONST_ACC_R ||
              type == INST_ACC_R) {
            t_css.read_pending_hits += m_stats_pw.at(streamID)[type][status];
          } else if (type == GLOBAL_ACC_W) {
            t_css.write_pending_hits += m_stats_pw.at(streamID)[type][status];
          }
        }

        if (status == RESERVATION_FAIL) {
          if (type == GLOBAL_ACC_R || type == CONST_ACC_R ||
              type == INST_ACC_R) {
            t_css.read_res_fails += m_stats_pw.at(streamID)[type][status];
          } else if (type == GLOBAL_ACC_W) {
            t_css.write_res_fails += m_stats_pw.at(streamID)[type][status];
          }
        }
      }
    }
  }

  css = t_css;
}

bool cache_stats::check_valid(int type, int status) const {
  ///
  /// Verify a valid access_type/access_status
  ///
  if ((type >= 0) && (type < NUM_MEM_ACCESS_TYPE) && (status >= 0) &&
      (status < NUM_CACHE_REQUEST_STATUS))
    return true;
  else
    return false;
}

bool cache_stats::check_fail_valid(int type, int fail) const {
  ///
  /// Verify a valid access_type/access_status
  ///
  if ((type >= 0) && (type < NUM_MEM_ACCESS_TYPE) && (fail >= 0) &&
      (fail < NUM_CACHE_RESERVATION_FAIL_STATUS))
    return true;
  else
    return false;
}

void cache_stats::sample_cache_port_utility(bool data_port_busy,
                                            bool fill_port_busy) {
  m_cache_port_available_cycles += 1;
  if (data_port_busy) {
    m_cache_data_port_busy_cycles += 1;
  }
  if (fill_port_busy) {
    m_cache_fill_port_busy_cycles += 1;
  }
}

baseline_cache::bandwidth_management::bandwidth_management(cache_config &config)
    : m_config(config) {
  m_data_port_occupied_cycles = 0;
  m_fill_port_occupied_cycles = 0;
}

/// use the data port based on the outcome and events generated by the mem_fetch
/// request
/*
cache中模拟了数据端口和填充端口，m_cache->access()使用data_port，m_cache->fill()使用
fill_port。
根据mem_fetch请求生成的结果和事件使用数据端口。因为数据端口的宽度有限，因此当对一个cache
访问时，一个数据包的大小要分割成几拍才能用这个端口。只有
  // query for data port availability
  bool baseline_cache::bandwidth_management::data_port_free() const {
    return (m_data_port_occupied_cycles == 0);
  }

  // query for fill port availability
  bool baseline_cache::bandwidth_management::fill_port_free() const {
    return (m_fill_port_occupied_cycles == 0);
  }
*/
void baseline_cache::bandwidth_management::use_data_port(
    mem_fetch *mf, enum cache_request_status outcome,
    const std::list<cache_event> &events) {
  unsigned data_size = mf->get_data_size();
  unsigned port_width = m_config.m_data_port_width;
  switch (outcome) {
    case HIT: {
      unsigned data_cycles =
          data_size / port_width + ((data_size % port_width > 0) ? 1 : 0);
      m_data_port_occupied_cycles += data_cycles;
    } break;
    case HIT_RESERVED:
    case MISS: {
      // the data array is accessed to read out the entire line for write-back
      // in case of sector cache we need to write bank only the modified sectors
      cache_event ev(WRITE_BACK_REQUEST_SENT);
      if (was_writeback_sent(events, ev)) {
        unsigned data_cycles = ev.m_evicted_block.m_modified_size / port_width;
        m_data_port_occupied_cycles += data_cycles;
      }
    } break;
    case SECTOR_MISS:
    case RESERVATION_FAIL:
      // Does not consume any port bandwidth
      break;
    default:
      assert(0);
      break;
  }
}

/// use the fill port
/*
根据mem_fetch请求使用填充端口。
*/
void baseline_cache::bandwidth_management::use_fill_port(mem_fetch *mf) {
  // assume filling the entire line with the returned request
  unsigned fill_cycles = m_config.get_atom_sz() / m_config.m_data_port_width;
  m_fill_port_occupied_cycles += fill_cycles;
}

/// called every cache cycle to free up the ports
void baseline_cache::bandwidth_management::replenish_port_bandwidth() {
  if (m_data_port_occupied_cycles > 0) {
    m_data_port_occupied_cycles -= 1;
  }
  assert(m_data_port_occupied_cycles >= 0);

  if (m_fill_port_occupied_cycles > 0) {
    m_fill_port_occupied_cycles -= 1;
  }
  assert(m_fill_port_occupied_cycles >= 0);
}

/// query for data port availability
bool baseline_cache::bandwidth_management::data_port_free() const {
  return (m_data_port_occupied_cycles == 0);
}

/// query for fill port availability
bool baseline_cache::bandwidth_management::fill_port_free() const {
  return (m_fill_port_occupied_cycles == 0);
}

/// Sends next request to lower level of memory
/*
cache向前推进一拍。
*/
void baseline_cache::cycle() {
  //如果MISS请求队列中不为空，则将队首的请求发送到下一级内存。
  if (!m_miss_queue.empty()) {
    mem_fetch *mf = m_miss_queue.front();
    if (!m_memport->full(mf->size(), mf->get_is_write())) {
      m_miss_queue.pop_front();
      //mem_fetch_interface是对mem访存的接口。
      //mem_fetch_interface是cache对mem访存的接口，cache将miss请求发送至下一级存储就是通过
      //这个接口来发送，即m_miss_queue中的数据包需要压入m_memport实现发送至下一级存储。
      m_memport->push(mf);
    }
  }
  bool data_port_busy = !m_bandwidth_management.data_port_free();
  bool fill_port_busy = !m_bandwidth_management.fill_port_free();
  m_stats.sample_cache_port_utility(data_port_busy, fill_port_busy);
  m_bandwidth_management.replenish_port_bandwidth();
}

/// Interface for response from lower memory level (model bandwidth restictions
/// in caller)
/*
返回的数据通过baseline_cache::fill填充进cache的tag_array中。
m_config.m_mshr_type定义的MSHR类型：
  TEX_FIFO,         // Tex cache
  ASSOC,            // normal cache
  SECTOR_TEX_FIFO,  // Tex cache sends requests to high-level sector cache
  SECTOR_ASSOC      // normal cache sends requests to high-level sector cache
Cache配置参数：
  <sector?>:<nsets>:<bsize>:<assoc>,
  <rep>:<wr>:<alloc>:<wr_alloc>:<set_index_fn>,
  <mshr>:<N>:<merge>,<mq>:**<fifo_entry>
GV100配置示例：
  -gpgpu_cache:dl1  S:4:128:64,  L:T:m:L:L, A:512:8, 16:0,32
  -gpgpu_cache:dl2  S:32:128:24, L:B:m:L:P, A:192:4, 32:0,32
  -gpgpu_cache:il1  N:64:128:16, L:R:f:N:L, S:2:48,  4
在GV100的MSHR type上，L1D为ASSOC，L2D为ASSOC，L1I为SECTOR_ASSOC。
  TEX_FIFO,         // Tex cache
  ASSOC,            // normal cache
  SECTOR_TEX_FIFO,  // Tex cache sends requests to high-level sector cache
  SECTOR_ASSOC      // normal cache sends requests to high-level sector cache
*/
void baseline_cache::fill(mem_fetch *mf, unsigned time) {
  //如果是sector cache，需要看当前mf是否是一个大mf分割后返回的最后一个小mf；但如果是line cache
  //的话，就不需要考虑这个，因为返回的mf一定是一整个cache block数据。
  if (m_config.m_mshr_type == SECTOR_ASSOC) {
    //mf->get_original_mf()是在L2 cache中将mf划分为sector requests时设置（如果req size > L2
    //sector size），此指针指向原始mf，因为实际上只是在性能模拟过程中把mf划分为几个sector，但当
    //mf返回时，需要将这些sector合并为一个mf，这里为了简便，就给每个分割的mf保留一个原始mf的指针。
    assert(mf->get_original_mf());
    //extra_mf_fields_lookup的定义：
    //  typedef std::map<mem_fetch *, extra_mf_fields> extra_mf_fields_lookup;
    //以L2 cache为例：
    //向cache发出数据请求mf时，如果未命中，且在MSHR中也未命中（没有mf条目），则将其加入到MSHR中，
    //同时，设置m_extra_mf_fields[mf]，意味着如果mf在m_extra_mf_fields中存在，即mf等待着DRAM
    //的数据回到L2缓存填充：
    //m_extra_mf_fields[mf] = extra_mf_fields(
    //      mshr_addr, mf->get_addr(), cache_index, mf->get_data_size(), m_config);
    //L1 Data cache中是等待SM内的m_response_fifo中的数据填充。
    extra_mf_fields_lookup::iterator e =
        m_extra_mf_fields.find(mf->get_original_mf());
    assert(e != m_extra_mf_fields.end());
    //能找到的话，设置m_extra_mf_fields[mf].pending_read减一。
    e->second.pending_read--;

    //如果m_extra_mf_fields[mf].pending_read大于0，说明还在等待其他与mf相同请求的数据，直接丢掉，
    //因为这里的mf其实只是一个被分割开的大mf的一部分，能够直接丢掉是因为后续同属于一个大请求的mf会
    //指向它们共同的原始mf：mf->get_original_mf()。
    if (e->second.pending_read > 0) {
      // wait for the other requests to come back
      delete mf;
      return;
    } else {
      //如果m_extra_mf_fields[mf].pending_read等于0，说明这个mf已经是最后一个分割的mf了，不需要
      //再等待其他与mf相同请求的数据，可以填充到cache中。
      mem_fetch *temp = mf;
      mf = mf->get_original_mf();
      delete temp;
    }
  }
  //现在完成了检查当前mf是否是一个大mf分割后返回的最后一个小mf（不是的话，此函数会直接退出了），接
  //下来就是将mf填充到cache中。

  extra_mf_fields_lookup::iterator e = m_extra_mf_fields.find(mf);
  assert(e != m_extra_mf_fields.end());
  assert(e->second.m_valid);
  mf->set_data_size(e->second.m_data_size);
  mf->set_addr(e->second.m_addr);
  //在V100中，L1 cache与L2 cache均为allocate on miss。
  //allocate-on-miss（简写为on-miss）和allocate-on-fill（简写为on-fill）是两种cache行分配策略：
  //(1) allocate-on-miss：当发生未完成的cache miss时，需要为未完成的miss分配cache line slot、
  //    MSHR和miss队列条目。
  //(2) allocateon-on-fill：当发生未完成的cache miss时，需要为未完成的miss分配MSHR和miss队列条
  //    目，但当所需数据从较低内存级别返回时，会选择受害者cache line slot替换。
  //在这两种策略中，如果任何所需资源不可用，则会发生reservation failure，并且内存流水线停顿。分配
  //的MSHR被保留，直到数据从较低一级内存中取回；而一旦miss请求转发到L2 cache，则释放miss队列条目。
  //allocateon-on-fill往往比allocate-on-miss有更好的性能，因为它保留了受害者cache更长的时间，并
  //且为未完成的miss保留更少的资源，从而享受更多的cache hit和更少的reservation failure。尽管all-
  //ocateon-on-fill需要额外的buffer和流控逻辑来将数据按顺序填充到cache，但按顺序执行模型和write-
  //evict策略使得GPU L1 D-cache对allocateon-on-fill很友好，因为当受害者cache在被evict时，没有脏
  //数据被写入L2。
  if (m_config.m_alloc_policy == ON_MISS)
    m_tag_array->fill(e->second.m_cache_index, time, mf);
  else if (m_config.m_alloc_policy == ON_FILL) {
    m_tag_array->fill(e->second.m_block_addr, time, mf, mf->is_write());
  } else
    abort();
  bool has_atomic = false;
  /*
  Accept a new cache fill response: mark entry ready for processing. 接受新的缓存填充响应：标
  记条目以备处理。
  */
  m_mshrs.mark_ready(e->second.m_block_addr, has_atomic);
  if (has_atomic) {
    assert(m_config.m_alloc_policy == ON_MISS);
    cache_block_t *block = m_tag_array->get_block(e->second.m_cache_index);
    if (!block->is_modified_line()) {
      m_tag_array->inc_dirty();
    }
    block->set_status(MODIFIED,
                      mf->get_access_sector_mask());  // mark line as dirty for
                                                      // atomic operation
    block->set_byte_mask(mf);
  }
  m_extra_mf_fields.erase(mf);
  m_bandwidth_management.use_fill_port(mf);
}

/// Checks if mf is waiting to be filled by lower memory level
/*
检查是否mf正在等待更低的存储层次填充。
*/
bool baseline_cache::waiting_for_fill(mem_fetch *mf) {
  //extra_mf_fields_lookup的定义：
  //  typedef std::map<mem_fetch *, extra_mf_fields> extra_mf_fields_lookup;
  //向cache发出数据请求mf时，如果未命中，且在MSHR中也未命中（没有mf条目），则将其加入到MSHR中，
  //同时，设置m_extra_mf_fields[mf]，意味着如果mf在m_extra_mf_fields中存在，即mf等待着DRAM
  //的数据回到L2缓存填充：
  //m_extra_mf_fields[mf] = extra_mf_fields(
  //      mshr_addr, mf->get_addr(), cache_index, mf->get_data_size(), m_config);
  extra_mf_fields_lookup::iterator e = m_extra_mf_fields.find(mf);
  return e != m_extra_mf_fields.end();
}

void baseline_cache::print(FILE *fp, unsigned &accesses,
                           unsigned &misses) const {
  fprintf(fp, "Cache %s:\t", m_name.c_str());
  m_tag_array->print(fp, accesses, misses);
}

void baseline_cache::display_state(FILE *fp) const {
  fprintf(fp, "Cache %s:\n", m_name.c_str());
  m_mshrs.display(fp);
  fprintf(fp, "\n");
}

void baseline_cache::inc_aggregated_stats(cache_request_status status,
                                          cache_request_status cache_status,
                                          mem_fetch *mf,
                                          enum cache_gpu_level level) {
  if (level == L1_GPU_CACHE) {
    m_gpu->aggregated_l1_stats.inc_stats(
        mf->get_streamID(), mf->get_access_type(),
        m_gpu->aggregated_l1_stats.select_stats_status(status, cache_status));
  } else if (level == L2_GPU_CACHE) {
    m_gpu->aggregated_l2_stats.inc_stats(
        mf->get_streamID(), mf->get_access_type(),
        m_gpu->aggregated_l2_stats.select_stats_status(status, cache_status));
  }
}

void baseline_cache::inc_aggregated_fail_stats(
    cache_request_status status, cache_request_status cache_status,
    mem_fetch *mf, enum cache_gpu_level level) {
  if (level == L1_GPU_CACHE) {
    m_gpu->aggregated_l1_stats.inc_fail_stats(
        mf->get_streamID(), mf->get_access_type(),
        m_gpu->aggregated_l1_stats.select_stats_status(status, cache_status));
  } else if (level == L2_GPU_CACHE) {
    m_gpu->aggregated_l2_stats.inc_fail_stats(
        mf->get_streamID(), mf->get_access_type(),
        m_gpu->aggregated_l2_stats.select_stats_status(status, cache_status));
  }
}

void baseline_cache::inc_aggregated_stats_pw(cache_request_status status,
                                             cache_request_status cache_status,
                                             mem_fetch *mf,
                                             enum cache_gpu_level level) {
  if (level == L1_GPU_CACHE) {
    m_gpu->aggregated_l1_stats.inc_stats_pw(
        mf->get_streamID(), mf->get_access_type(),
        m_gpu->aggregated_l1_stats.select_stats_status(status, cache_status));
  } else if (level == L2_GPU_CACHE) {
    m_gpu->aggregated_l2_stats.inc_stats_pw(
        mf->get_streamID(), mf->get_access_type(),
        m_gpu->aggregated_l2_stats.select_stats_status(status, cache_status));
  }
}

/// Read miss handler without writeback
/*
READ MISS处理函数，检查MSHR是否命中或者MSHR是否可用，依此判断是否需要向下一级存储发送读请求。
*/
void baseline_cache::send_read_request(new_addr_type addr,
                                       new_addr_type block_addr,
                                       unsigned cache_index, mem_fetch *mf,
                                       unsigned time, bool &do_miss,
                                       std::list<cache_event> &events,
                                       bool read_only, bool wa) {
  bool wb = false;
  evicted_block_info e;
  send_read_request(addr, block_addr, cache_index, mf, time, do_miss, wb, e,
                    events, read_only, wa);
}

/// Read miss handler. Check MSHR hit or MSHR available
/*
READ MISS 处理函数，检查 MSHR 是否命中或者 MSHR 是否可用，依此判断是否需要向下一
级存储发送读请求。
*/
void baseline_cache::send_read_request(new_addr_type addr,
                                       new_addr_type block_addr,
                                       unsigned cache_index, mem_fetch *mf,
                                       unsigned time, bool &do_miss, bool &wb,
                                       evicted_block_info &evicted,
                                       std::list<cache_event> &events,
                                       bool read_only, bool wa) {
  // 1. 如果是 Sector Cache：
  //  mshr_addr 函数返回 mshr 的地址，该地址即为地址 addr 的 tag 位 + set index 
  //  位 + sector offset 位。即除 single sector byte offset 位 以外的所有位。
  //  |<----------mshr_addr----------->|
  //                     sector offset  off in-sector
  //                     |-------------|-----------|
  //                      \                       /
  //                       \                     /
  //  |-------|-------------|-------------------|
  //             set_index     offset in-line
  //  |<----tag----> 0 0 0 0|
  // 2. 如果是 Line Cache：
  //  mshr_addr 函数返回 mshr 的地址，该地址即为地址 addr 的 tag 位 + set index 
  //  位。即除 single line byte off-set 位 以外的所有位。
  //  |<----mshr_addr--->|
  //                              line offset
  //                     |-------------------------|
  //                      \                       /
  //                       \                     /
  //  |-------|-------------|-------------------|
  //             set_index     offset in-line
  //  |<----tag----> 0 0 0 0|
  //
  // mshr_addr 定义：
  //   new_addr_type mshr_addr(new_addr_type addr) const {
  //     return addr & ~(new_addr_type)(m_atom_sz - 1);
  //   }
  // m_atom_sz = (m_cache_type == SECTOR) ? SECTOR_SIZE : m_line_sz; 
  // 其中 SECTOR_SIZE = const (32 bytes per sector).
  new_addr_type mshr_addr = m_config.mshr_addr(mf->get_addr());
  // 这里实际上是 MSHR 查找是否已经有 mshr_addr 的请求被合并到 MSHR。如果已经被挂
  // 起则 mshr_hit = true。需要注意，MSHR 中的条目是以 mshr_addr 为索引的，即来自
  // 同一个 line（对于非 Sector Cache）或者来自同一个 sector（对于 Sector Cache）
  // 的事务被合并，因为这种 cache 所请求的最小单位分别是一个 line 或者一个 sector，
  // 因此没必要发送那么多事务，只需要发送一次即可。
  bool mshr_hit = m_mshrs.probe(mshr_addr);
  // 如果 mshr_addr 在 MSHR 中已存在条目，m_mshrs.full 检查是否该条目的合并数量已
  // 达到最大合并数；如果 mshr_addr 在 MSHR 中不存在条目，则检查是否有空闲的 MSHR 
  // 条目可以将 mshr_addr 插入进 MSHR。
  bool mshr_avail = !m_mshrs.full(mshr_addr);
  if (mshr_hit && mshr_avail) {
    // 如果 MSHR 命中，且 mshr_addr 对应条目的合并数量没有达到最大合并数，则将数据
    // 请求 mf 加入到 MSHR 中。
    if (read_only)
      m_tag_array->access(block_addr, time, cache_index, mf);
    else
      // 更新 tag_array 的状态，包括更新 LRU 状态，设置逐出的 block 或 sector 等。
      m_tag_array->access(block_addr, time, cache_index, wb, evicted, mf);

    // 将 mshr_addr 地址的数据请求 mf 加入到 MSHR 中。因为命中 MSHR，说明前面已经
    // 有对该数据的请求发送到下一级缓存了，因此这里只需要等待前面的请求返回即可。
    m_mshrs.add(mshr_addr, mf);
    m_stats.inc_stats(mf->get_access_type(), MSHR_HIT, mf->get_streamID());
    // 标识是否请求被填充进 MSHR 或者 被放到 m_miss_queue 以在下一个周期发送到下一
    // 级存储。
    do_miss = true;

  } else if (!mshr_hit && mshr_avail &&
             (m_miss_queue.size() < m_config.m_miss_queue_size)) {
    // 如果 MSHR 未命中，但有空闲的 MSHR 条目可以将 mshr_addr 插入进 MSHR，则将数
    // 据请求 mf 插入到 MSHR 中。
    // 对于 L1 cache 和 L2 cache，read_only 为 false，对于 read_only_cache 来说，
    // read_only 为true。
    if (read_only)
      m_tag_array->access(block_addr, time, cache_index, mf);
    else
      // 更新 tag_array 的状态，包括更新 LRU 状态，设置逐出的 block 或 sector 等。
      m_tag_array->access(block_addr, time, cache_index, wb, evicted, mf);

    // 将 mshr_addr 地址的数据请求 mf 加入到 MSHR 中。因为没有命中 MSHR，因此还需
    // 要将该数据的请求发送到下一级缓存。
    m_mshrs.add(mshr_addr, mf);
    // if (m_config.is_streaming() && m_config.m_cache_type == SECTOR) {
    //   m_tag_array->add_pending_line(mf);
    // }
    // 设置 m_extra_mf_fields[mf]，意味着如果 mf 在 m_extra_mf_fields 中存在，即 
    // mf 等待着下一级存储的数据回到当前缓存填充。
    m_extra_mf_fields[mf] = extra_mf_fields(
        mshr_addr, mf->get_addr(), cache_index, mf->get_data_size(), m_config);
    mf->set_data_size(m_config.get_atom_sz());
    mf->set_addr(mshr_addr);
    // mf 为 miss 的请求，加入 miss_queue，MISS 请求队列。
    // 在 baseline_cache::cycle() 中，会将 m_miss_queue 队首的数据包 mf 传递给下
    // 一层存储。因为没有命中 MSHR，说明前面没有对该数据的请求发送到下一级缓存，
    // 因此这里需要把该请求发送给下一级存储。
    m_miss_queue.push_back(mf);
    mf->set_status(m_miss_queue_status, time);
    // 在 V100 配置中，wa 对 L1/L2/read_only cache 均为 false。
    if (!wa) events.push_back(cache_event(READ_REQUEST_SENT));
    // 标识是否请求被填充进 MSHR 或者 被放到 m_miss_queue 以在下一个周期发送到下一
    // 级存储。
    do_miss = true;
  } else if (mshr_hit && !mshr_avail)
    // 如果 MSHR 命中，但 mshr_addr 对应条目的合并数量达到了最大合并数。
    m_stats.inc_fail_stats(mf->get_access_type(), MSHR_MERGE_ENRTY_FAIL,
                           mf->get_streamID());
  else if (!mshr_hit && !mshr_avail)
    // 如果 MSHR 未命中，且 mshr_addr 没有空闲的 MSHR 条目可将 mshr_addr 插入。
    m_stats.inc_fail_stats(mf->get_access_type(), MSHR_ENRTY_FAIL,
                           mf->get_streamID());
  else
    assert(0);
}

/// Sends write request to lower level memory (write or writeback)
/*
将数据写请求一同发送至下一级存储。这里需要做的是将写请求类型 WRITE_REQUEST_SENT 或 
WRITE_BACK_REQUEST_SENT 放入 events，并将数据请求 mf 放入 m_miss_queue中，等待下
一时钟周期 baseline_cache::cycle() 将队首的数据请求 mf 发送给下一级存储。
*/
void data_cache::send_write_request(mem_fetch *mf, cache_event request,
                                    unsigned time,
                                    std::list<cache_event> &events) {
  events.push_back(request);
  // 在 baseline_cache::cycle() 中，会将 m_miss_queue 队首的数据包 mf 传递给下
  // 一级存储。
  m_miss_queue.push_back(mf);
  mf->set_status(m_miss_queue_status, time);
}

/*
更新一个cache block的状态为可读。如果所有的byte mask位全都设置为dirty了，则将该sector可
设置为可读，因为当前的sector已经是全部更新为最新值了，是可读的。这个函数对所有的数据请求mf
的所有访问的sector进行遍历，如果mf所访问的所有的byte mask位全都设置为dirty了，则将该cache
block设置为可读。
*/
void data_cache::update_m_readable(mem_fetch *mf, unsigned cache_index) {
  //这里传入的参数是cache block的index。
  // For example, 4 sets, 6 ways:
  // |  0  |  1  |  2  |  3  |  4  |  5  |  // set_index 0
  // |  6  |  7  |  8  |  9  |  10 |  11 |  // set_index 1
  // |  12 |  13 |  14 |  15 |  16 |  17 |  // set_index 2
  // |  18 |  19 |  20 |  21 |  22 |  23 |  // set_index 3
  //                |--------> index => cache_block_t *line
  cache_block_t *block = m_tag_array->get_block(cache_index);
  //对当前cache block的4个sector进行遍历。
  for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; i++) {
    //第i个sector被数据请求mf访问。
    if (mf->get_access_sector_mask().test(i)) {
      //all_set是指所有的byte mask位都被设置成了dirty了。
      bool all_set = true;
      //这里k是隶属于第i个sector的byte的编号。
      for (unsigned k = i * SECTOR_SIZE; k < (i + 1) * SECTOR_SIZE; k++) {
        // If any bit in the byte mask (within the sector) is not set,
        // the sector is unreadble
        //如果第i个sector中有任意一个byte的dirty mask位没有被设置，则all_set就是false。
        if (!block->get_dirty_byte_mask().test(k)) {
          all_set = false;
          break;
        }
      }
      //如果所有的byte mask位全都设置为dirty了，则将该sector可设置为可读，因为当前的
      //sector已经是全部更新为最新值了，是可读的。
      if (all_set) block->set_m_readable(true, mf->get_access_sector_mask());
    }
  }
}

/****** Write-hit functions (Set by config file) ******/

/// Write-back hit: Mark block as modified
/*
若 Write Hit 时采取 write-back 策略，则需要将数据单写入 cache，不需要直接将数据写入
下一级存储。等到新数据 fill 进来时，再将旧数据逐出并写入下一级存储。
*/
cache_request_status data_cache::wr_hit_wb(new_addr_type addr,
                                           unsigned cache_index, mem_fetch *mf,
                                           unsigned time,
                                           std::list<cache_event> &events,
                                           enum cache_request_status status) {
  // m_config.block_addr(addr): 
  //     return addr & ~(new_addr_type)(m_line_sz - 1);
  // |-------|-------------|--------------|
  //            set_index   offset in-line
  // |<--------tag--------> 0 0 0 0 0 0 0 |
  // write-back 策略不需要直接将数据写入下一级存储，因此不需要调用miss_queue_full()
  // 以及 send_write_request() 函数来发送写回请求到下一级存储。
  new_addr_type block_addr = m_config.block_addr(addr);
  // 更新 tag_array 的状态，包括更新 LRU 状态，设置逐出的 block 或 sector 等。
  m_tag_array->access(block_addr, time, cache_index, mf);  // update LRU state
  cache_block_t *block = m_tag_array->get_block(cache_index);
  // 如果 block 不是 modified line，则增加 dirty 计数。因为如果这个时候 block 不是
  // modified line，说明这个 block 是 clean line，而现在要写入数据，因此需要将这个
  // block 设置为 modified line。这样的话，dirty 计数就需要增加。但如果 block 已经
  // 是 modified line，则不需要增加 dirty 计数，因为这个 block 在上次变成 dirty 的
  // 时候，dirty 计数已经增加过了。
  if (!block->is_modified_line()) {
    m_tag_array->inc_dirty();
  }
  // 设置 block 的状态为 modified，即将 block 设置为 MODIFIED。这样的话，下次再有
  // 数据请求访问这个 block 的时候，就可以直接从 cache 中读取数据，而不需要再次访问
  // 下一级存储。当然，当有下次填充进这个 block 的数据请求时（block 的 tag 与请求的
  // tag 不一致），由于这个 block 的状态已经被设置为 modified，因此需要将此 block 
  // 的数据逐出并写回到下一级存储。
  block->set_status(MODIFIED, mf->get_access_sector_mask());
  block->set_byte_mask(mf);
  // 更新一个 cache block 的状态为可读。但需要注意的是，这里的可读是指该 sector 可
  // 读，而不是整个 block 可读。如果一个 sector 内的所有的 byte mask 位全都设置为 
  // dirty 了，则将该sector 可设置为可读，因为当前的 sector 已经是全部更新为最新值
  // 了，是可读的。这个函数对所有的数据请求 mf 的所有访问的 sector 进行遍历，这里的
  // sector 是由 mf 访问的，并由 mf->get_access_sector_mask() 确定。
  update_m_readable(mf, cache_index);

  return HIT;
}

/// Write-through hit: Directly send request to lower level memory
/*
若 Write Hit 时采取 write-through 策略的话，则需要将数据不单单写入 cache，还需要直
接将数据写入下一级存储。
对一个cache进行数据访问的时候，调用data_cache::access()函数：
- 首先cahe会调用m_tag_array->probe()函数，判断对cache的访问（地址为addr，sector mask
  为mask）是HIT/HIT_RESERVED/SECTOR_MISS/MISS/RESERVATION_FAIL等状态。
- 然后调用process_tag_probe()函数，根据cache的配置以及上面m_tag_array->probe()函数返
  回的cache访问状态，执行相应的操作。
  - process_tag_probe()函数中，会根据请求的读写状态，probe()函数返回的cache访问状态，
    执行m_wr_hit/m_wr_miss/m_rd_hit/m_rd_miss函数，他们会调用m_tag_array->access()
    函数来实现LRU状态的更新。
*/
cache_request_status data_cache::wr_hit_wt(new_addr_type addr,
                                           unsigned cache_index, mem_fetch *mf,
                                           unsigned time,
                                           std::list<cache_event> &events,
                                           enum cache_request_status status) {
  // miss_queue_full 检查是否一个 miss 请求能够在当前时钟周期内被处理，当一个请求的
  // 大小大到 m_miss_queue 放不下时即在当前拍内无法处理，发生 RESERVATION_FAIL。
  if (miss_queue_full(0)) {
    m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL,
                           mf->get_streamID());
    // 如果 miss_queue 满了，但由于 write-through 策略要求数据应该直接写入下一级存
    // 储，因此这里返回 RESERVATION_FAIL，表示当前时钟周期内无法处理该请求。
    return RESERVATION_FAIL;  // cannot handle request this cycle
  }
  // m_config.block_addr(addr): 
  //     return addr & ~(new_addr_type)(m_line_sz - 1);
  // |-------|-------------|--------------|
  //            set_index   offset in-line
  // |<--------tag--------> 0 0 0 0 0 0 0 |
  new_addr_type block_addr = m_config.block_addr(addr);
  // 更新 tag_array 的状态，包括更新 LRU 状态，设置逐出的 block 或 sector 等。
  m_tag_array->access(block_addr, time, cache_index, mf);  // update LRU state
  cache_block_t *block = m_tag_array->get_block(cache_index);
  // 如果 block 不是 modified line，则增加 dirty 计数。因为如果这个时候 block 不是
  // modified line，说明这个 block 是 clean line，而现在要写入数据，因此需要将这个
  // block 设置为 modified line。这样的话，dirty 计数就需要增加。但如果 block 已经
  // 是 modified line，则不需要增加 dirty 计数，因为这个 block 在上次变成 dirty 的
  // 时候，dirty 计数已经增加过了。
  if (!block->is_modified_line()) {
    m_tag_array->inc_dirty();
  }
  // 设置 block 的状态为 modified，即将 block 设置为 MODIFIED。这样的话，下次再有
  // 数据请求访问这个 block 的时候，就可以直接从 cache 中读取数据，而不需要再次访问
  // 下一级存储。
  block->set_status(MODIFIED, mf->get_access_sector_mask());
  block->set_byte_mask(mf);
  // 更新一个 cache block 的状态为可读。但需要注意的是，这里的可读是指该 sector 可
  // 读，而不是整个 block 可读。如果一个 sector 内的所有的 byte mask 位全都设置为 
  // dirty 了，则将该sector 可设置为可读，因为当前的 sector 已经是全部更新为最新值
  // 了，是可读的。这个函数对所有的数据请求 mf 的所有访问的 sector 进行遍历，这里的
  // sector 是由 mf 访问的，并由 mf->get_access_sector_mask() 确定。
  update_m_readable(mf, cache_index);

  // generate a write-through
  // write-through 策略需要将数据写入 cache 的同时也直接写入下一级存储。这里需要做
  // 的是将写请求类型 WRITE_REQUEST_SENT 放入 events，并将数据请求放入当前 cache  
  // 的 m_miss_queue 中，等待baseline_cache::cycle() 将 m_miss_queue 队首的数
  // 据写请求 mf 发送给下一级存储。
  send_write_request(mf, cache_event(WRITE_REQUEST_SENT), time, events);

  return HIT;
}

/// Write-evict hit: Send request to lower level memory and invalidate
/// corresponding block
/*
写逐出命中：向下一级存储发送写回请求并直接逐出相应的 cache block 并设置其无效。
*/
cache_request_status data_cache::wr_hit_we(new_addr_type addr,
                                           unsigned cache_index, mem_fetch *mf,
                                           unsigned time,
                                           std::list<cache_event> &events,
                                           enum cache_request_status status) {
  if (miss_queue_full(0)) {
    m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL,
                           mf->get_streamID());
    return RESERVATION_FAIL;  // cannot handle request this cycle
  }

  // generate a write-through/evict
  cache_block_t *block = m_tag_array->get_block(cache_index);
  // write-evict 策略需要将 cache block 直接逐出置为无效的同时也直接写入下一级存
  // 储。这里需要做的是将写请求类型 WRITE_REQUEST_SENT 放入 events，并将数据请求  
  // 放入 m_miss_queue 中，等待baseline_cache::cycle() 将 m_miss_queue 队首的
  // 数据写请求 mf 发送给下一级存储。
  send_write_request(mf, cache_event(WRITE_REQUEST_SENT), time, events);

  // Invalidate block
  // 写逐出，将 cache block 直接逐出置为无效。
  block->set_status(INVALID, mf->get_access_sector_mask());

  return HIT;
}

/// Global write-evict, local write-back: Useful for private caches
/*
全局访存采用写逐出，本地访存采用写回。这种策略适用于私有缓存。这个策略比较简单，即只
需要判断当前的数据请求是全局访存还是本地访存，然后分别采用写逐出和写回策略即可。
*/
enum cache_request_status data_cache::wr_hit_global_we_local_wb(
    new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events, enum cache_request_status status) {
  bool evict = (mf->get_access_type() ==
                GLOBAL_ACC_W);  // evict a line that hits on global memory write
  if (evict)
    return wr_hit_we(addr, cache_index, mf, time, events,
                     status);  // Write-evict
  else
    return wr_hit_wb(addr, cache_index, mf, time, events,
                     status);  // Write-back
}

/****** Write-miss functions (Set by config file) ******/

/// Write-allocate miss: Send write request to lower level memory
// and send a read request for the same block
/*
GPGPU-Sim 3.x版本中的naive写分配策略。wr_miss_wa_naive 策略在写 MISS 时，需要先将 
mf 数据包直接写入下一级存储，即它会将 WRITE_REQUEST_SENT 放入 events，并将数据请求 
mf 放入 m_miss_queue 中，等待下一个周期 baseline_cache::cycle() 将 m_miss_queue 
队首的数据包 mf 发送给下一级存储。其次，wr_miss_wa_naive 策略还会将 addr 地址的数据
读到当前 cache 中，这时候会执行 send_read_request 函数。但是在 send_read_request 
函数中，很有可能这个读请求需要 evict 一个 block 才可以将新的数据读入到 cache 中，这
时候如果 evicted block 是 modified line，则需要将这个 evicted block 写回到下一级
存储，这时候会根据 do_miss 和 wb 的值执行 send_write_request 函数。
*/
enum cache_request_status data_cache::wr_miss_wa_naive(
    new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events, enum cache_request_status status) {
  // m_config.block_addr(addr): 
  //     return addr & ~(new_addr_type)(m_line_sz - 1);
  // |-------|-------------|--------------|
  //            set_index   offset in-line
  // |<--------tag--------> 0 0 0 0 0 0 0 | 
  new_addr_type block_addr = m_config.block_addr(addr);
  // 1. 如果是 Sector Cache：
  //  mshr_addr 函数返回 mshr 的地址，该地址即为地址 addr 的 tag 位 + set index 
  //  位 + sector offset 位。即除 single sector byte offset 位 以外的所有位。
  //  |<----------mshr_addr----------->|
  //                     sector offset  off in-sector
  //                     |-------------|-----------|
  //                      \                       /
  //                       \                     /
  //  |-------|-------------|-------------------|
  //             set_index     offset in-line
  //  |<----tag----> 0 0 0 0|
  // 2. 如果是 Line Cache：
  //  mshr_addr 函数返回 mshr 的地址，该地址即为地址 addr 的 tag 位 + set index 
  //  位。即除 single line byte off-set 位 以外的所有位。
  //  |<----mshr_addr--->|
  //                              line offset
  //                     |-------------------------|
  //                      \                       /
  //                       \                     /
  //  |-------|-------------|-------------------|
  //             set_index     offset in-line
  //  |<----tag----> 0 0 0 0|
  //
  // mshr_addr 定义：
  //   new_addr_type mshr_addr(new_addr_type addr) const {
  //     return addr & ~(new_addr_type)(m_atom_sz - 1);
  //   }
  // m_atom_sz = (m_cache_type == SECTOR) ? SECTOR_SIZE : m_line_sz; 
  // 其中 SECTOR_SIZE = const (32 bytes per sector).
  new_addr_type mshr_addr = m_config.mshr_addr(mf->get_addr());

  // Write allocate, maximum 3 requests (write miss, read request, write back
  // request) Conservatively ensure the worst-case request can be handled this
  // cycle
  // MSHR 的 m_data 的 key 中存储了各个合并的地址，probe() 函数主要检查是否命中，
  // 即主要检查 m_data.keys() 这里面有没有 mshr_addr。
  bool mshr_hit = m_mshrs.probe(mshr_addr);
  // 首先查找是否 MSHR 表中有 block_addr 地址的条目。如果存在该条目（命中 MSHR），
  // 看是否有空间合并进该条目。如果不存在该条目（未命中 MSHR），看是否有其他空间允
  // 许添加 mshr_addr 这一条目。
  bool mshr_avail = !m_mshrs.full(mshr_addr);
  // 在 baseline_cache::cycle() 中，会将 m_miss_queue 队首的数据包 mf 传递给下一
  // 级存储。因此当遇到 miss 的请求或者写回的请求需要访问下一级存储时，会把 miss 的
  // 请求放到 m_miss_queue 中。
  //   bool miss_queue_full(unsigned num_miss) {
  //     return ((m_miss_queue.size() + num_miss) >= m_config.m_miss_queue_size);
  //   }
  
  // 如果 m_miss_queue.size() 已经不能容下三个数据包的话，有可能无法完成后续动作，
  // 因为后面最多需要执行三次 send_write_request，在 send_write_request 里每执行
  // 一次，都需要向 m_miss_queue 添加一个数据包。
  // Write allocate, maximum 3 requests (write miss, read request, write back
  // request) Conservatively ensure the worst-case request can be handled this
  // cycle.
  if (miss_queue_full(2) ||
      // 如果 miss_queue_full(2) 返回 false，有空余空间支持执行三次 send_write_
      // request，那么就需要看 MSHR 是否有可用空间。后面这串判断条件其实可以化简成 
      // if (miss_queue_full(2) || !mshr_avail)。
      // 即符合 RESERVATION_FAIL 的条件：
      //   1. m_miss_queue 不足以放入三个 WRITE_REQUEST_SENT 请求；
      //   2. MSHR 不能合并请求（未命中，或者没有可用空间添加新条目）。
      (!(mshr_hit && mshr_avail) &&
       !(!mshr_hit && mshr_avail &&
         (m_miss_queue.size() < m_config.m_miss_queue_size)))) {
    // check what is the exactly the failure reason
    if (miss_queue_full(2))
      m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL,
                             mf->get_streamID());
    else if (mshr_hit && !mshr_avail)
      m_stats.inc_fail_stats(mf->get_access_type(), MSHR_MERGE_ENRTY_FAIL,
                             mf->get_streamID());
    else if (!mshr_hit && !mshr_avail)
      m_stats.inc_fail_stats(mf->get_access_type(), MSHR_ENRTY_FAIL,
                             mf->get_streamID());
    else
      assert(0);

    // 符合 RESERVATION_FAIL 的条件：
    //   1. m_miss_queue 不足以放入三个 WRITE_REQUEST_SENT 请求；
    //   2. MSHR 不能合并请求（未命中，或者没有可用空间添加新条目）。
    return RESERVATION_FAIL;
  }

  // send_write_request 执行：
  //   events.push_back(request);
  //   // 在 baseline_cache::cycle() 中，会将 m_miss_queue 队首的数据包 mf 传递
  //   // 给下一级存储。
  //   m_miss_queue.push_back(mf);
  //   mf->set_status(m_miss_queue_status, time);
  // wr_miss_wa_naive 策略在写 MISS 时，需要先将 mf 数据包直接写入下一级存储，即它
  // 会将 WRITE_REQUEST_SENT 放入 events，并将数据请求 mf 放入 m_miss_queue 中，
  // 等待下一个周期 baseline_cache::cycle() 将 m_miss_queue 队首的数据包 mf 发送
  // 给下一级存储。其次，wr_miss_wa_naive 策略还会将 addr 地址的数据读到当前 cache
  // 中，这时候会执行 send_read_request 函数。但是在 send_read_request 函数中，很
  // 有可能这个读请求需要 evict 一个 block 才可以将新的数据读入到 cache 中，这时候
  // 如果 evicted block 是 modified line，则需要将这个 evicted block 写回到下一级
  // 存储，这时候会根据 do_miss 和 wb 的值执行 send_write_request 函数。
  send_write_request(mf, cache_event(WRITE_REQUEST_SENT), time, events);
  // Tries to send write allocate request, returns true on success and false on
  // failure
  // if(!send_write_allocate(mf, addr, block_addr, cache_index, time, events))
  //    return RESERVATION_FAIL;

  const mem_access_t *ma =
      new mem_access_t(m_wr_alloc_type, mf->get_addr(), m_config.get_atom_sz(),
                       false,  // Now performing a read
                       mf->get_access_warp_mask(), mf->get_access_byte_mask(),
                       mf->get_access_sector_mask(), m_gpu->gpgpu_ctx);

  mem_fetch *n_mf = new mem_fetch(
      *ma, NULL, mf->get_streamID(), mf->get_ctrl_size(), mf->get_wid(),
      mf->get_sid(), mf->get_tpc(), mf->get_mem_config(),
      m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle);

  // 标识是否请求被填充进 MSHR 或者 被放到 m_miss_queue 以在下一个周期发送到下一
  // 级存储。
  bool do_miss = false;
  // wb 变量标识 tag_array::access() 函数中，如果下面的 send_read_request 函数
  // 发生 MISS，则需要逐出一个 block，并将这个 evicted block 写回到下一级存储。
  // 如果这个 block 已经是 modified line，则 wb 为 true，因为在将其分配给新访问
  // 之前，必须将这个已经 modified 的 block 写回到下一级存储。但如果这个 block 
  // 是 clean line，则 wb 为 false，因为这个 block 不需要写回到下一级存储。这个 
  // evicted block 的信息被设置在 evicted 中。
  bool wb = false;
  evicted_block_info evicted;

  // Send read request resulting from write miss
  send_read_request(addr, block_addr, cache_index, n_mf, time, do_miss, wb,
                    evicted, events, false, true);

  events.push_back(cache_event(WRITE_ALLOCATE_SENT));

  // do_miss 标识是否请求被填充进 MSHR 或者 被放到 m_miss_queue 以在下一个周期
  // 发送到下一级存储。
  if (do_miss) {
    // If evicted block is modified and not a write-through
    // (already modified lower level)
    // wb 变量标识 tag_array::access() 函数中，如果下面的 send_read_request 函
    // 数发生 MISS，则需要逐出一个 block，并将这个 evicted block 写回到下一级存
    // 储。如果这个 block 已经是 modified line，则 wb 为 true，因为在将其分配给
    // 新访问之前，必须将这个已经 modified 的 block 写回到下一级存储。但如果这个  
    // block 是 clean line，则 wb 为 false，因为这个 block 不需要写回到下一级存 
    // 储。这个 evicted block 的信息被设置在 evicted 中。
    if (wb && (m_config.m_write_policy != WRITE_THROUGH)) {
      assert(status ==
             MISS);  // SECTOR_MISS and HIT_RESERVED should not send write back
      mem_fetch *wb = m_memfetch_creator->alloc(
          evicted.m_block_addr, m_wrbk_type, mf->get_access_warp_mask(),
          evicted.m_byte_mask, evicted.m_sector_mask, evicted.m_modified_size,
          true, m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, -1, -1, -1,
          NULL, mf->get_streamID());
      // the evicted block may have wrong chip id when advanced L2 hashing  is
      // used, so set the right chip address from the original mf
      wb->set_chip(mf->get_tlx_addr().chip);
      wb->set_partition(mf->get_tlx_addr().sub_partition);
      // 将 tag_array::access() 函数中逐出的 evicted block 写回到下一级存储。
      send_write_request(wb, cache_event(WRITE_BACK_REQUEST_SENT, evicted),
                         time, events);
    }
    // 如果 do_miss 为 true，表示请求被填充进 MSHR 或者 被放到 m_miss_queue 以在
    // 下一个周期发送到下一级存储。即整个写 MISS 处理函数的所有过程全部完成，返回的
    // 是 write miss 这个原始写请求的状态。
    return MISS;
  }

  // 如果 do_miss 为 false，表示请求未被填充进 MSHR 或者 未被放到 m_miss_queue 以
  // 在下一个周期发送到下一级存储。即整个写 MISS 处理函数没有将读请求发送出去，因此
  // 返回 RESERVATION_FAIL。
  return RESERVATION_FAIL;
}

/*
write_allocated_fetch_on_write 策略，在写入时读取策略中，当写入 sector 的单个字节
时，L2 会读取整个 sector ，然后将写入的部分合并到该 sector ，并将该 sector 设置为已
修改。
*/
enum cache_request_status data_cache::wr_miss_wa_fetch_on_write(
    new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events, enum cache_request_status status) {
  // m_config.block_addr(addr): 
  //     return addr & ~(new_addr_type)(m_line_sz - 1);
  // |-------|-------------|--------------|
  //            set_index   offset in-line
  // |<--------tag--------> 0 0 0 0 0 0 0 | 
  new_addr_type block_addr = m_config.block_addr(addr);
  // 1. 如果是 Sector Cache：
  //  mshr_addr 函数返回 mshr 的地址，该地址即为地址 addr 的 tag 位 + set index 
  //  位 + sector offset 位。即除 single sector byte offset 位 以外的所有位。
  //  |<----------mshr_addr----------->|
  //                     sector offset  off in-sector
  //                     |-------------|-----------|
  //                      \                       /
  //                       \                     /
  //  |-------|-------------|-------------------|
  //             set_index     offset in-line
  //  |<----tag----> 0 0 0 0|
  // 2. 如果是 Line Cache：
  //  mshr_addr 函数返回 mshr 的地址，该地址即为地址 addr 的 tag 位 + set index 
  //  位。即除 single line byte off-set 位 以外的所有位。
  //  |<----mshr_addr--->|
  //                              line offset
  //                     |-------------------------|
  //                      \                       /
  //                       \                     /
  //  |-------|-------------|-------------------|
  //             set_index     offset in-line
  //  |<----tag----> 0 0 0 0|
  //
  // mshr_addr 定义：
  //   new_addr_type mshr_addr(new_addr_type addr) const {
  //     return addr & ~(new_addr_type)(m_atom_sz - 1);
  //   }
  // m_atom_sz = (m_cache_type == SECTOR) ? SECTOR_SIZE : m_line_sz; 
  // 其中 SECTOR_SIZE = const (32 bytes per sector).
  new_addr_type mshr_addr = m_config.mshr_addr(mf->get_addr());

  // 如果请求写入的字节数等于整个 cache line/sector 的大小，那么直接写入 cache，并
  // 将 cache 设置为 MODIFIED，而不需要发送读请求到下一级存储。
  if (mf->get_access_byte_mask().count() == m_config.get_atom_sz()) {
    // if the request writes to the whole cache line/sector, then, write and set
    // cache line Modified. and no need to send read request to memory or
    // reserve mshr
    // 如果 m_miss_queue.size() 已经不能容下一个数据包的话，有可能无法完成后续动作，
    // 因为后面最多需要执行一次 send_write_request，在 send_write_request 里每执行
    // 一次，都需要向 m_miss_queue 添加一个数据包。
    if (miss_queue_full(0)) {
      m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL,
                             mf->get_streamID());
      return RESERVATION_FAIL;  // cannot handle request this cycle
    }
    // wb 变量标识 tag_array::access() 函数中，如果下面的 send_read_request 函数
    // 发生 MISS，则需要逐出一个 block，并将这个 evicted block 写回到下一级存储。
    // 如果这个 block 已经是 modified line，则 wb 为 true，因为在将其分配给新访问
    // 之前，必须将这个已经 modified 的 block 写回到下一级存储。但如果这个 block 
    // 是 clean line，则 wb 为 false，因为这个 block 不需要写回到下一级存储。这个 
    // evicted block 的信息被设置在 evicted 中。
    bool wb = false;
    evicted_block_info evicted;
    // 更新 tag_array 的状态，包括更新 LRU 状态，设置逐出的 block 或 sector 等。
    cache_request_status status =
        m_tag_array->access(block_addr, time, cache_index, wb, evicted, mf);
    assert(status != HIT);
    cache_block_t *block = m_tag_array->get_block(cache_index);
    // 如果 block 不是 modified line，则增加 dirty 计数。因为如果这个时候 block 不
    // 是 modified line，说明这个 block 是 clean line，而现在要写入数据，因此需要将
    // 这个 block 设置为 modified line。这样的话，dirty 计数就需要增加。但若 block 
    // 已经是 modified line，则不需要增加 dirty 计数，这个 block 在上次变成 dirty 
    // 的时候，dirty 计数已经增加过了。
    if (!block->is_modified_line()) {
      m_tag_array->inc_dirty();
    }
    // 设置 block 的状态为 modified，即将 block 设置为 MODIFIED。这样的话，下次再
    // 有数据请求访问这个 block 的时候，就可以直接从 cache 中读取数据，而不需要再次
    // 访问下一级存储。
    block->set_status(MODIFIED, mf->get_access_sector_mask());
    block->set_byte_mask(mf);

    // 暂时不用关心这个函数。
    if (status == HIT_RESERVED)
      block->set_ignore_on_fill(true, mf->get_access_sector_mask());

    // 只要 m_tag_array->access 返回的状态不是 RESERVATION_FAIL，就说明或者发生了
    // HIT_RESERVED，或者 SECTOR_MISS，又或者 MISS。这里只要不是 RESERVATION_FAIL，
    // 就代表有 cache block 被分配了，因此要根据这个被逐出的 cache block 是否需要写
    // 回，将这个 block 写回到下一级存储。
    if (status != RESERVATION_FAIL) {
      // If evicted block is modified and not a write-through
      // (already modified lower level)
      // wb 变量标识 tag_array::access() 函数中，如果上面的 m_tag_array->access 
      // 函数发生 MISS，则需要逐出一个 block，并将这个 evicted block 写回到下一级
      // 存储。如果这个 block 已经是 modified line，则 wb 为 true，因为在将其分配
      // 给新访问之前，必须将这个已经 modified 的 block 写回到下一级存储。但如果这  
      // 个 block 是 clean line，则 wb 为 false，因为这个 block 不需要写回到下一 
      // 级存储。这个 evicted block 的信息被设置在 evicted 中。
      // 这里如果 cache 的写策略为写直达，就不需要在读 miss 时将被逐出的 MODIFIED 
      // cache block 写回到下一级存储，因为这个 cache block 在被 MODIFIED 的时候
      // 已经被 write-through 到下一级存储了。
      if (wb && (m_config.m_write_policy != WRITE_THROUGH)) {
        mem_fetch *wb = m_memfetch_creator->alloc(
            evicted.m_block_addr, m_wrbk_type, mf->get_access_warp_mask(),
            evicted.m_byte_mask, evicted.m_sector_mask, evicted.m_modified_size,
            true, m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, -1, -1, -1,
            NULL, mf->get_streamID());
        // the evicted block may have wrong chip id when advanced L2 hashing  is
        // used, so set the right chip address from the original mf
        wb->set_chip(mf->get_tlx_addr().chip);
        wb->set_partition(mf->get_tlx_addr().sub_partition);
        // 写回 evicted block 到下一级存储。
        send_write_request(wb, cache_event(WRITE_BACK_REQUEST_SENT, evicted),
                           time, events);
      }
      // 整个写 MISS 处理函数的所有过程全部完成，返回的是 write miss 这个原始写请求
      // 的状态。
      return MISS;
    }
    // 整个写 MISS 处理函数没有分配新的 cache block，并将逐出的 block 写回，因此返
    // 回 RESERVATION_FAIL。
    return RESERVATION_FAIL;
  } else {
    // 如果请求写入的字节数小于整个 cache line/sector 的大小，那么需要发送读请求到
    // 下一级存储，然后将写入的部分合并到该 sector ，并将该 sector 设置为已修改。

    // MSHR 的 m_data 的 key 中存储了各个合并的地址，probe() 函数主要检查是否命中，
    // 即主要检查 m_data.keys() 这里面有没有 mshr_addr。
    bool mshr_hit = m_mshrs.probe(mshr_addr);
    // 首先查找是否 MSHR 表中有 block_addr 地址的条目。如果存在该条目（命中 MSHR），
    // 看是否有空间合并进该条目。如果不存在该条目（未命中 MSHR），看是否有其他空间允
    // 许添加 mshr_addr 这一条目。
    bool mshr_avail = !m_mshrs.full(mshr_addr);
    // 如果 m_miss_queue.size() 已经不能容下两个数据包的话，有可能无法完成后续动作，
    // 因为后面最多需要执行一次 send_read_request 和一次 send_write_request，这两
    // 次有可能最多需要向 m_miss_queue 添加两个数据包。
    // 若 miss_queue_full(1) 返回 false，有空余空间支持执行一次 send_write_request
    // 和一次 send_read_request，那么就需要看 MSHR 是否有可用空间。后面这串判断条件
    // 其实可以化简成： 
    //   if (miss_queue_full(1) || !mshr_avail)。
    // 即符合 RESERVATION_FAIL 的条件：
    //   1. m_miss_queue 不足以放入一个读一个写，共两个请求；
    //   2. MSHR 不能合并请求（未命中，或者没有可用空间添加新条目）。
    if (miss_queue_full(1) ||
        (!(mshr_hit && mshr_avail) &&
         !(!mshr_hit && mshr_avail &&
           (m_miss_queue.size() < m_config.m_miss_queue_size)))) {
      // check what is the exactly the failure reason
      if (miss_queue_full(1))
        m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL,
                               mf->get_streamID());
      else if (mshr_hit && !mshr_avail)
        m_stats.inc_fail_stats(mf->get_access_type(), MSHR_MERGE_ENRTY_FAIL,
                               mf->get_streamID());
      else if (!mshr_hit && !mshr_avail)
        m_stats.inc_fail_stats(mf->get_access_type(), MSHR_ENRTY_FAIL,
                               mf->get_streamID());
      else
        assert(0);

      return RESERVATION_FAIL;
    }

    // prevent Write - Read - Write in pending mshr
    // allowing another write will override the value of the first write, and
    // the pending read request will read incorrect result from the second write
    if (m_mshrs.probe(mshr_addr) &&
        m_mshrs.is_read_after_write_pending(mshr_addr) && mf->is_write()) {
      // assert(0);
      m_stats.inc_fail_stats(mf->get_access_type(), MSHR_RW_PENDING,
                             mf->get_streamID());
      return RESERVATION_FAIL;
    }

    const mem_access_t *ma = new mem_access_t(
        m_wr_alloc_type, mf->get_addr(), m_config.get_atom_sz(),
        false,  // Now performing a read
        mf->get_access_warp_mask(), mf->get_access_byte_mask(),
        mf->get_access_sector_mask(), m_gpu->gpgpu_ctx);

    mem_fetch *n_mf = new mem_fetch(
        *ma, NULL, mf->get_streamID(), mf->get_ctrl_size(), mf->get_wid(),
        mf->get_sid(), mf->get_tpc(), mf->get_mem_config(),
        m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, NULL, mf);

    // m_config.block_addr(addr): 
    //     return addr & ~(new_addr_type)(m_line_sz - 1);
    // |-------|-------------|--------------|
    //            set_index   offset in-line
    // |<--------tag--------> 0 0 0 0 0 0 0 | 
    new_addr_type block_addr = m_config.block_addr(addr);
    bool do_miss = false;
    bool wb = false;
    evicted_block_info evicted;
    // 发送读请求到下一级存储，然后将写入的部分合并到该 sector ，并将该 sector 设
    // 置为已修改。
    send_read_request(addr, block_addr, cache_index, n_mf, time, do_miss, wb,
                      evicted, events, false, true);

    cache_block_t *block = m_tag_array->get_block(cache_index);
    // 将 block 设置为在下次 fill 时，置为 MODIFIED。这样的话，下次再有数据请求填
    // 入 fill 时：
    //   m_status = m_set_modified_on_fill ? MODIFIED : VALID; 或
    //   m_status[sidx] = m_set_modified_on_fill[sidx] ? MODIFIED : VALID;
    block->set_modified_on_fill(true, mf->get_access_sector_mask());
    block->set_byte_mask_on_fill(true);

    events.push_back(cache_event(WRITE_ALLOCATE_SENT));

    // do_miss 标识是否请求被填充进 MSHR 或者 被放到 m_miss_queue 以在下一个周期
    // 发送到下一级存储。
    if (do_miss) {
      // If evicted block is modified and not a write-through
      // (already modified lower level)
      // wb 变量标识 tag_array::access() 函数中，如果上面的 m_tag_array->access 
      // 函数发生 MISS，则需要逐出一个 block，并将这个 evicted block 写回到下一级
      // 存储。如果这个 block 已经是 modified line，则 wb 为 true，因为在将其分配
      // 给新访问之前，必须将这个已经 modified 的 block 写回到下一级存储。但如果这  
      // 个 block 是 clean line，则 wb 为 false，因为这个 block 不需要写回到下一 
      // 级存储。这个 evicted block 的信息被设置在 evicted 中。
      // 这里如果 cache 的写策略为写直达，就不需要在读 miss 时将被逐出的 MODIFIED 
      // cache block 写回到下一级存储，因为这个 cache block 在被 MODIFIED 的时候
      // 已经被 write-through 到下一级存储了。
      if (wb && (m_config.m_write_policy != WRITE_THROUGH)) {
        mem_fetch *wb = m_memfetch_creator->alloc(
            evicted.m_block_addr, m_wrbk_type, mf->get_access_warp_mask(),
            evicted.m_byte_mask, evicted.m_sector_mask, evicted.m_modified_size,
            true, m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, -1, -1, -1,
            NULL, mf->get_streamID());
        // the evicted block may have wrong chip id when advanced L2 hashing  is
        // used, so set the right chip address from the original mf
        wb->set_chip(mf->get_tlx_addr().chip);
        wb->set_partition(mf->get_tlx_addr().sub_partition);
        // 写回 evicted block 到下一级存储。
        send_write_request(wb, cache_event(WRITE_BACK_REQUEST_SENT, evicted),
                           time, events);
      }
      // 整个写 MISS 处理函数的所有过程全部完成，返回的是 write miss 这个原始写请求
      // 的状态。
      return MISS;
    }
    // 整个写 MISS 处理函数没有分配新的 cache block，并将逐出的 block 写回，因此返
    // 回 RESERVATION_FAIL。
    return RESERVATION_FAIL;
  }
}

/*
write_allocated_lazy_fetch_on_read 策略。
需要参考 https://arxiv.org/pdf/1810.07269.pdf 论文对 Volta 架构访存行为的解释。
L2 缓存应用了不同的写入分配策略，将其命名为延迟读取读取，这是写入验证和写入时读取之间
的折衷方案。当收到对已修改扇区的扇区读请求时，它首先检查扇区写掩码是否完整，即所有字节
均已写入并且该行完全可读。如果是，则读取该扇区；否则，与写入时读取类似，它生成该扇区的
读取请求并将其与修改后的字节合并。
*/
enum cache_request_status data_cache::wr_miss_wa_lazy_fetch_on_read(
    new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events, enum cache_request_status status) {
  // m_config.block_addr(addr): 
  //     return addr & ~(new_addr_type)(m_line_sz - 1);
  // |-------|-------------|--------------|
  //            set_index   offset in-line
  // |<--------tag--------> 0 0 0 0 0 0 0 | 
  new_addr_type block_addr = m_config.block_addr(addr);

  // if the request writes to the whole cache line/sector, then, write and set
  // cache line Modified. and no need to send read request to memory or reserve
  // mshr

  // FETCH_ON_READ policy: https://arxiv.org/pdf/1810.07269.pdf
  // In literature, there are two different write allocation policies [32], fetch-
  // on-write and write-validate. In fetch-on-write, when we write to a single byte
  // of a sector, the L2 fetches the whole sector then merges the written portion 
  // to the sector and sets the sector as modified. In the write-validate policy, 
  // no read fetch is required, instead each sector has a bit-wise write-mask. When 
  // a write to a single byte is received, it writes the byte to the sector, sets 
  // the corresponding write bit and sets the sector as valid and modified. When a 
  // modified cache line is evicted, the cache line is written back to the memory 
  // along with the write mask. It is important to note that, in a write-validate 
  // policy, it assumes the read and write granularity can be in terms of bytes in 
  // order to exploit the benefits of the write-mask. In fact, based on our micro-
  // benchmark shown in Figure 5, we have observed that the L2 cache applies some-
  // thing similar to write-validate. However, all the reads received by L2 caches 
  // from the coalescer are 32-byte sectored accesses. Thus, the read access granu-
  // larity (32 bytes) is different from the write access granularity (one byte). 
  // To handle this, the L2 cache applies a different write allocation policy, 
  // which we named lazy fetch-on-read, that is a compromise between write-validate 
  // and fetch-on-write. When a sector read request is received to a modified sector, 
  // it first checks if the sector write-mask is complete, i.e. all the bytes have 
  // been written to and the line is fully readable. If so, it reads the sector, 
  // otherwise, similar to fetch-on-write, it generates a read request for this 
  // sector and merges it with the modified bytes.

  // 若 m_miss_queue.size() 已经不能容下一个数据包的话，有可能无法完成后续动作，因
  // 为后面最多需要执行一次 send_write_request，有可能最多需要向 m_miss_queue 添加
  // 两个数据包。虽然后面代码里有两次 send_write_request，但是这两次是不会同时发生。
  if (miss_queue_full(0)) {
    m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL,
                           mf->get_streamID());
    return RESERVATION_FAIL;  // cannot handle request this cycle
  }

  // 在 V100 配置中，L1 cache 为 'T'-write through，L2 cache 为 'B'-write back。
  if (m_config.m_write_policy == WRITE_THROUGH) {
    // 如果是 write through，则需要直接将数据一同写回下一层存储。将数据写请求一同发
    // 送至下一级存储。这里需要做的是将读请求类型 WRITE_REQUEST_SENT 放入 events，
    // 并将数据请求 mf 放入当前 cache 的 m_miss_queue 中，等待 baseline_cache::
    // cycle() 将队首的数据请求 mf 发送给下一级存储。
    send_write_request(mf, cache_event(WRITE_REQUEST_SENT), time, events);
  }

  // wb 变量标识 tag_array::access() 函数中，如果下面的 send_read_request 函数发
  // 生 MISS，则需要逐出一个 block，并将这个 evicted block 写回到下一级存储。如果
  // 这个 block 已经是 modified line，则 wb 为 true，因为在将其分配给新访问之前，
  // 必须将这个已经 modified 的 block 写回到下一级存储。但如果这个 block 是 clean
  // line，则 wb 为 false，因为这个 block 不需要写回到下一级存储。这个 evicted 
  // block 的信息被设置在 evicted 中。
  bool wb = false;
  // evicted 记录着被逐出的 cache block 的信息。
  evicted_block_info evicted;
  // 更新 tag_array 的状态，包括更新 LRU 状态，设置逐出的 block 或 sector 等。
  cache_request_status m_status =
      m_tag_array->access(block_addr, time, cache_index, wb, evicted, mf);

  // Theoretically, the passing parameter status should be the same as the 
  // m_status, if the assertion fails here, go to function :
  //     `wr_miss_wa_lazy_fetch_on_read` 
  // to remove this assertion.
  // assert((m_status == status));
  assert(m_status != HIT);
  // cache_index 是 cache block 的 index。
  cache_block_t *block = m_tag_array->get_block(cache_index);
  // 如果 block 不是 modified line，则增加 dirty 计数。因为如果这个时候 block 不
  // 是 modified line，说明这个 block 是 clean line，而现在要写入数据，因此需要将
  // 这个 block 设置为 modified line。这样的话，dirty 计数就需要增加。但若 block 
  // 已经是 modified line，则不需要增加 dirty 计数，这个 block 在上次变成 dirty 
  // 的时候，dirty 计数已经增加过了。
  if (!block->is_modified_line()) {
    m_tag_array->inc_dirty();
  }
  // 设置 block 的状态为 modified，即将 block 设置为 MODIFIED。这样的话，下次再
  // 有数据请求访问这个 block 的时候，就可以直接从 cache 中读取数据，而不需要再次
  // 访问下一级存储。
  block->set_status(MODIFIED, mf->get_access_sector_mask());
  block->set_byte_mask(mf);
  // 如果 Cache block[mask] 状态是 RESERVED，说明有其他的线程正在读取这个 Cache 
  // block。挂起的命中访问已命中处于 RESERVED 状态的缓存行，这意味着同一行上已存在
  // 由先前缓存未命中发送的 flying 内存请求。
  if (m_status == HIT_RESERVED) {
    // 在当前版本的 GPGPU-Sim 中，set_ignore_on_fill 暂时用不到。
    block->set_ignore_on_fill(true, mf->get_access_sector_mask());
    // cache block 的每个 sector 都有一个标志位 m_set_modified_on_fill[i]，标记
    // 着这个 cache block 是否被修改，在sector_cache_block::fill() 函数调用的时
    // 候会使用。
    // 将 block 设置为在下次 fill 时，置为 MODIFIED。这样的话，下次再有数据请求填
    // 入 fill 时：
    //   m_status = m_set_modified_on_fill ? MODIFIED : VALID; 或
    //   m_status[sidx] = m_set_modified_on_fill[sidx] ? MODIFIED : VALID;
    block->set_modified_on_fill(true, mf->get_access_sector_mask());
    // 在 FETCH_ON_READ policy: https://arxiv.org/pdf/1810.07269.pdf 中提到，
    // 访问 cache 发生 miss 时：
    // In the write-validate policy, no read fetch is required, instead each  
    // sector has a bit-wise write-mask. When a write to a single byte is 
    // received, it writes the byte to the sector, sets the corresponding 
    // write bit and sets the sector as valid and modified. When a modified 
    // cache line is evicted, the cache line is written back to the memory 
    // along with the write mask.
    // 而在 FETCH_ON_READ 中，需要设置 sector 的 byte mask。这里就是指设置这个 
    // byte mask 的标志。
    block->set_byte_mask_on_fill(true);
  }

  // m_config.get_atom_sz() 为 SECTOR_SIZE = 4，即 mf 访问的是一整个 4 字节。
  if (mf->get_access_byte_mask().count() == m_config.get_atom_sz()) {
    // 由于 mf 访问的是整个 sector，因此整个 sector 都是 dirty 的，设置访问的 
    // sector 可读。
    block->set_m_readable(true, mf->get_access_sector_mask());
  } else {
    // 由于 mf 访问的是部分 sector，因此只有 mf 访问的那部分 sector 是 dirty 的，
    // 设置访问的 sector 不可读。但是设置在下次这个 sector 被 fill 时，mf->get_
    // access_sector_mask() 标识的 byte 置为 MODIFIED。
    block->set_m_readable(false, mf->get_access_sector_mask());
    if (m_status == HIT_RESERVED)
      block->set_readable_on_fill(true, mf->get_access_sector_mask());
  }
  // 更新一个 cache block 的状态为可读。如果所有的 byte mask 位全都设置为 dirty 
  // 了，则将该 sector 可设置为可读，因为当前的 sector 已经是全部更新为最新值了，
  // 是可读的。这个函数对所有的数据请求 mf 的所有访问的 sector 进行遍历，如果 mf 
  // 所访问的所有的 byte mask 位全都设置为 dirty 了，则将该 cache block 设置为可
  // 读。
  update_m_readable(mf, cache_index);

  // 只要 m_tag_array->access 返回的状态不是 RESERVATION_FAIL，就说明或者发生了
  // HIT_RESERVED，或者 SECTOR_MISS，又或者 MISS。这里只要不是 RESERVATION_FAIL，
  // 就代表有 cache block 被分配了，因此要根据这个被逐出的 cache block 是否需要写
  // 回，将这个 block 写回到下一级存储。
  if (m_status != RESERVATION_FAIL) {
    // If evicted block is modified and not a write-through
    // (already modified lower level)
    // wb 变量标识 tag_array::access() 函数中，如果上面的 m_tag_array->access 
    // 函数发生 MISS，则需要逐出一个 block，并将这个 evicted block 写回到下一级
    // 存储。如果这个 block 已经是 modified line，则 wb 为 true，因为在将其分配
    // 给新访问之前，必须将这个已经 modified 的 block 写回到下一级存储。但如果这  
    // 个 block 是 clean line，则 wb 为 false，因为这个 block 不需要写回到下一 
    // 级存储。这个 evicted block 的信息被设置在 evicted 中。
    // 这里如果 cache 的写策略为写直达，就不需要在读 miss 时将被逐出的 MODIFIED 
    // cache block 写回到下一级存储，因为这个 cache block 在被 MODIFIED 的时候
    // 已经被 write-through 到下一级存储了。
    if (wb && (m_config.m_write_policy != WRITE_THROUGH)) {
      mem_fetch *wb = m_memfetch_creator->alloc(
          evicted.m_block_addr, m_wrbk_type, mf->get_access_warp_mask(),
          evicted.m_byte_mask, evicted.m_sector_mask, evicted.m_modified_size,
          true, m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, -1, -1, -1,
          NULL, mf->get_streamID());
      // the evicted block may have wrong chip id when advanced L2 hashing  is
      // used, so set the right chip address from the original mf
      wb->set_chip(mf->get_tlx_addr().chip);
      wb->set_partition(mf->get_tlx_addr().sub_partition);
      // 写回 evicted block 到下一级存储。
      send_write_request(wb, cache_event(WRITE_BACK_REQUEST_SENT, evicted),
                         time, events);
    }
    // 整个写 MISS 处理函数的所有过程全部完成，返回的是 write miss 这个原始写请求
    // 的状态。
    return MISS;
  }
  // 整个写 MISS 处理函数没有分配新的 cache block，并将逐出的 block 写回，因此返
  // 回 RESERVATION_FAIL。
  return RESERVATION_FAIL;
}

/// No write-allocate miss: Simply send write request to lower level memory
/*
No write-allocate miss，这个处理函数仅仅简单地将写请求发送到下一级存储。
*/
enum cache_request_status data_cache::wr_miss_no_wa(
    new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events, enum cache_request_status status) {
  // 如果 m_miss_queue.size() 已经不能容下一个数据包的话，有可能无法完成后续动作，
  // 因为后面最多需要执行一次 send_write_request，在 send_write_request 里每执行
  // 一次，都需要向 m_miss_queue 添加一个数据包。
  if (miss_queue_full(0)) {
    m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL,
                           mf->get_streamID());
    return RESERVATION_FAIL;  // cannot handle request this cycle
  }

  // on miss, generate write through (no write buffering -- too many threads for
  // that)
  // send_write_request 执行：
  //   events.push_back(request);
  //   // 在 baseline_cache::cycle() 中，会将 m_miss_queue 队首的数据包 mf 传递
  //   // 给下一级存储。
  //   m_miss_queue.push_back(mf);
  //   mf->set_status(m_miss_queue_status, time);
  // No write-allocate miss 策略在写 MISS 时，直接将 mf 数据包直接写入下一级存储。
  // 这里需要做的是将写请求类型 WRITE_REQUEST_SENT 放入 events，并将数据请求放入  
  // m_miss_queue 中，等待baseline_cache::cycle() 将 m_miss_queue 队首的数据写
  // 请求 mf 发送给下一级存储。
  send_write_request(mf, cache_event(WRITE_REQUEST_SENT), time, events);

  return MISS;
}

/****** Read hit functions (Set by config file) ******/

/// Baseline read hit: Update LRU status of block.
// Special case for atomic instructions -> Mark block as modified
/*
READ HIT 操作。
*/
enum cache_request_status data_cache::rd_hit_base(
    new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events, enum cache_request_status status) {
  // m_config.block_addr(addr): 
  //     return addr & ~(new_addr_type)(m_line_sz - 1);
  // |-------|-------------|--------------|
  //            set_index   offset in-line
  // |<--------tag--------> 0 0 0 0 0 0 0 |
  new_addr_type block_addr = m_config.block_addr(addr);
  // 更新 tag_array 的状态，包括更新 LRU 状态，设置逐出的 block 或 sector 等。
  m_tag_array->access(block_addr, time, cache_index, mf);
  // Atomics treated as global read/write requests - Perform read, mark line as
  // MODIFIED
  // 原子操作是指对全局和共享内存中的32位或者64位数据进行 “读取-修改-覆写” 这一操
  // 作。原子操作可以看作是一种最小单位的执行过程。在其执行过程中，不允许其他并行线
  // 程对该变量进行读取和写入的操作。如果发生竞争，则其他线程必须等待。
  // 原子操作从全局存储取值，计算，并写回相同地址三项事务在同一原子操作中完成，因此
  // 会修改 cache 的状态为 MODIFIED。
  if (mf->isatomic()) {
    assert(mf->get_access_type() == GLOBAL_ACC_R);
    // 获取该原子操作的 cache block，并判断其是否先前已被 MODIFIED，如果先前未被 
    // MODIFIED，此次原子操作做出 MODIFIED，要增加 dirty 数目，如果先前 block 已
    // 经被 MODIFIED，则先前dirty 数目已经增加过了，就不需要再增加了。
    cache_block_t *block = m_tag_array->get_block(cache_index);
    if (!block->is_modified_line()) {
      m_tag_array->inc_dirty();
    }
    // 设置 cache block 的状态为 MODIFIED，以避免其他线程在这个 cache block 上的
    // 读写操作。
    block->set_status(MODIFIED,
                      mf->get_access_sector_mask());  // mark line as
    // 设置 dirty_byte_mask。
    block->set_byte_mask(mf);
  }
  return HIT;
}

/****** Read miss functions (Set by config file) ******/

/// Baseline read miss: Send read request to lower level memory,
// perform write-back as necessary
/*
READ MISS 操作。
*/
enum cache_request_status data_cache::rd_miss_base(
    new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events, enum cache_request_status status) {
  // 读 miss 时，就需要将数据请求发送至下一级存储。这里或许需要真实地向下一级存储发
  // 送读请求，也或许由于 mshr 的存在，可以将数据请求合并进去，这样就不需要真实地向
  // 下一级存储发送读请求。
  // miss_queue_full 检查是否一个 miss 请求能够在当前时钟周期内被处理，当一个请求
  // 的大小大到 m_miss_queue 放不下时即在当前拍内无法处理，发生 RESERVATION_FAIL。
  if (miss_queue_full(1)) {
    // cannot handle request this cycle
    // (might need to generate two requests)
    m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL,
                           mf->get_streamID());
    return RESERVATION_FAIL;
  }

  // m_config.block_addr(addr): 
  //     return addr & ~(new_addr_type)(m_line_sz - 1);
  // |-------|-------------|--------------|
  //            set_index   offset in-line
  // |<--------tag--------> 0 0 0 0 0 0 0 |
  new_addr_type block_addr = m_config.block_addr(addr);
  // 标识是否请求被填充进 MSHR 或者 被放到 m_miss_queue 以在下一个周期发送到下一
  // 级存储。
  bool do_miss = false;
  // wb 代表是否需要写回（当一个被逐出的 cache block 被 MODIFIED 时，需要写回到
  // 下一级存储），evicted 代表被逐出的 cache line 的信息。
  bool wb = false;
  evicted_block_info evicted;
  // READ MISS 处理函数，检查 MSHR 是否命中或者 MSHR 是否可用，依此判断是否需要
  // 向下一级存储发送读请求。
  send_read_request(addr, block_addr, cache_index, mf, time, do_miss, wb,
                    evicted, events, false, false);
  // 如果 send_read_request 中数据请求已经被加入到 MSHR，或是原先存在该条目将请
  // 求合并进去，或是原先不存在该条目将请求插入进去，那么 do_miss 为 true，代表
  // 要将某个 cache block 逐出并接收 mf 从下一级存储返回的数据。
  // m_lines[idx] 作为逐出并 reserve 新访问的 cache line，如果它的某个 sector 
  // 已经被 MODIFIED，则需要执行写回操作，设置写回的标志为 wb = true。
  if (do_miss) {
    // If evicted block is modified and not a write-through
    // (already modified lower level).
    // 这里如果 cache 的写策略为写直达，就不需要在读 miss 时将被逐出的 MODIFIED 
    // cache block 写回到下一级存储，因为这个 cache block 在被 MODIFIED 的时候
    // 已经被 write-through 到下一级存储了。
    if (wb && (m_config.m_write_policy != WRITE_THROUGH)) {
      // 发送写请求，将 MODIFIED 的被逐出的 cache block 写回到下一级存储。
      // 在 V100 中，
      //     m_wrbk_type：L1 cache 为 L1_WRBK_ACC，L2 cache 为 L2_WRBK_ACC。
      //     m_write_policy：L1 cache 为 WRITE_THROUGH。
      mem_fetch *wb = m_memfetch_creator->alloc(
          evicted.m_block_addr, m_wrbk_type, mf->get_access_warp_mask(),
          evicted.m_byte_mask, evicted.m_sector_mask, evicted.m_modified_size,
          true, m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, -1, -1, -1,
          NULL, mf->get_streamID());
      // the evicted block may have wrong chip id when advanced L2 hashing  is
      // used, so set the right chip address from the original mf
      wb->set_chip(mf->get_tlx_addr().chip);
      wb->set_partition(mf->get_tlx_addr().sub_partition);
      // 将数据写请求一同发送至下一级存储。
      // 需要做的是将读请求类型 WRITE_BACK_REQUEST_SENT放入 events，并将数据请
      // 求 mf 放入当前 cache 的 m_miss_queue 中，等 baseline_cache::cycle() 
      // 将队首的数据请求 mf 发送给下一级存储。
      send_write_request(wb, WRITE_BACK_REQUEST_SENT, time, events);
    }
    return MISS;
  }
  return RESERVATION_FAIL;
}

/// Access cache for read_only_cache: returns RESERVATION_FAIL if
// request could not be accepted (for any reason)
/*
read_only_cache 访问，包括 L1I，L1C。
*/
enum cache_request_status read_only_cache::access(
    new_addr_type addr, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events) {
  assert(mf->get_data_size() <= m_config.get_atom_sz());
  assert(m_config.m_write_policy == READ_ONLY);
  assert(!mf->get_is_write());
  // m_config.block_addr(addr): 
  //     return addr & ~(new_addr_type)(m_line_sz - 1);
  // |-------|-------------|--------------|
  //            set_index   offset in-line
  // |<--------tag--------> 0 0 0 0 0 0 0 |
  new_addr_type block_addr = m_config.block_addr(addr);
  //cache_index会返回依据tag位选中的cache block的索引。
  unsigned cache_index = (unsigned)-1;
  //判断对cache的访问（地址为addr，sector mask为mask）是HIT/HIT_RESERVED/SECTOR_MISS/
  //MISS/RESERVATION_FAIL等状态。
  enum cache_request_status status =
      m_tag_array->probe(block_addr, cache_index, mf, mf->is_write());
  enum cache_request_status cache_status = RESERVATION_FAIL;

  if (status == HIT) {
    //仅更新LRU状态。
    cache_status = m_tag_array->access(block_addr, time, cache_index,
                                       mf);  // update LRU state
  } else if (status != RESERVATION_FAIL) {
    //HIT_RESERVED/SECTOR_MISS/MISS状态。
    if (!miss_queue_full(0)) {
      bool do_miss = false;
      //READ MISS处理函数，检查MSHR是否命中或者MSHR是否可用，依此判断是否需要向下一级存储发
      //送读请求。
      send_read_request(addr, block_addr, cache_index, mf, time, do_miss,
                        events, true, false);
      //代表要将某个cache block逐出并接收mf从下一级存储返回的数据。
      if (do_miss)
        cache_status = MISS;
      else
        cache_status = RESERVATION_FAIL;
    } else {
      cache_status = RESERVATION_FAIL;
      m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL,
                             mf->get_streamID());
    }
  } else {
    m_stats.inc_fail_stats(mf->get_access_type(), LINE_ALLOC_FAIL,
                           mf->get_streamID());
  }

  m_stats.inc_stats(mf->get_access_type(),
                    m_stats.select_stats_status(status, cache_status),
                    mf->get_streamID());
  m_stats.inc_stats_pw(mf->get_access_type(),
                       m_stats.select_stats_status(status, cache_status),
                       mf->get_streamID());
  return cache_status;
}

//! A general function that takes the result of a tag_array probe
//  and performs the correspding functions based on the cache configuration
//  The access fucntion calls this function
/*
一个通用函数，它获取tag_array探测的结果并根据缓存配置执行相应的功能。
access函数调用它：
对一个cache进行数据访问的时候，调用data_cache::access()函数：
- 首先cahe会调用m_tag_array->probe()函数，判断对cache的访问（地址为addr，sector mask
  为mask）是HIT/HIT_RESERVED/SECTOR_MISS/MISS/RESERVATION_FAIL等状态。
- 然后调用process_tag_probe()函数，根据cache的配置以及上面m_tag_array->probe()函数返
  回的cache访问状态，执行相应的操作。
  - process_tag_probe()函数中，会根据请求的读写状态，probe()函数返回的cache访问状态，
    执行m_wr_hit/m_wr_miss/m_rd_hit/m_rd_miss函数，他们会调用m_tag_array->access()
    函数来实现LRU状态的更新。
*/
enum cache_request_status data_cache::process_tag_probe(
    bool wr, enum cache_request_status probe_status, new_addr_type addr,
    unsigned cache_index, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events) {
  // Each function pointer ( m_[rd/wr]_[hit/miss] ) is set in the
  // data_cache constructor to reflect the corresponding cache configuration
  // options. Function pointers were used to avoid many long conditional
  // branches resulting from many cache configuration options.
  cache_request_status access_status = probe_status;
  if (wr) {  // Write
    if (probe_status == HIT) {
      //这里会在cache_index中写入cache block的索引。
      access_status =
          (this->*m_wr_hit)(addr, cache_index, mf, time, events, probe_status);
    } else if ((probe_status != RESERVATION_FAIL) ||
               (probe_status == RESERVATION_FAIL &&
                m_config.m_write_alloc_policy == NO_WRITE_ALLOCATE)) {
      access_status =
          (this->*m_wr_miss)(addr, cache_index, mf, time, events, probe_status);
    } else {
      // the only reason for reservation fail here is LINE_ALLOC_FAIL (i.e all
      // lines are reserved)
      m_stats.inc_fail_stats(mf->get_access_type(), LINE_ALLOC_FAIL,
                             mf->get_streamID());
    }
  } else {  // Read
    if (probe_status == HIT) {
      access_status =
          (this->*m_rd_hit)(addr, cache_index, mf, time, events, probe_status);
    } else if (probe_status != RESERVATION_FAIL) {
      access_status =
          (this->*m_rd_miss)(addr, cache_index, mf, time, events, probe_status);
    } else {
      // the only reason for reservation fail here is LINE_ALLOC_FAIL (i.e all
      // lines are reserved)
      m_stats.inc_fail_stats(mf->get_access_type(), LINE_ALLOC_FAIL,
                             mf->get_streamID());
    }
  }

  m_bandwidth_management.use_data_port(mf, access_status, events);
  return access_status;
}

// Both the L1 and L2 currently use the same access function.
// Differentiation between the two caches is done through configuration
// of caching policies.
// Both the L1 and L2 override this function to provide a means of
// performing actions specific to each cache when such actions are implemnted.
/*
L1 和 L2 目前使用相同的访问功能。两个缓存之间的区分是通过配置缓存策略来完成的。
L1 和 L2 都覆盖此函数，以提供在包含此类操作时执行特定于每个缓存的操作的方法。
对cache进行数据访问。

对一个cache进行数据访问的时候，调用data_cache::access()函数：
- 首先cahe会调用m_tag_array->probe()函数，判断对cache的访问（地址为addr，sector mask
  为mask）是HIT/HIT_RESERVED/SECTOR_MISS/MISS/RESERVATION_FAIL等状态。
- 然后调用process_tag_probe()函数，根据cache的配置以及上面m_tag_array->probe()函数返
  回的cache访问状态，执行相应的操作。
  - process_tag_probe()函数中，会根据请求的读写状态，probe()函数返回的cache访问状态，
    执行m_wr_hit/m_wr_miss/m_rd_hit/m_rd_miss函数，他们会调用m_tag_array->access()
    函数来实现LRU状态的更新。
*/
enum cache_request_status data_cache::access(new_addr_type addr, mem_fetch *mf,
                                             unsigned time,
                                             std::list<cache_event> &events) {
  //m_config.get_atom_sz()是cache替换原子操作的粒度，如果cache是SECTOR类型的，粒度为
  //SECTOR_SIZE，否则为line_size。
  assert(mf->get_data_size() <= m_config.get_atom_sz());
  bool wr = mf->get_is_write();
  // m_config.block_addr(addr): 
  //     return addr & ~(new_addr_type)(m_line_sz - 1);
  // |-------|-------------|--------------|
  //            set_index   offset in-line
  // |<--------tag--------> 0 0 0 0 0 0 0 | 
  new_addr_type block_addr = m_config.block_addr(addr);
  //cache_index会返回依据tag位选中的cache block的索引。
  unsigned cache_index = (unsigned)-1;
  //判断对cache的访问（地址为addr，sector mask为mask）是HIT/HIT_RESERVED/SECTOR_MISS/
  //MISS/RESERVATION_FAIL等状态。并且如果返回 MISS，则将需要被替换的cache block的索引
  //写入cache_index。
  enum cache_request_status probe_status =
      m_tag_array->probe(block_addr, cache_index, mf, mf->is_write(), true);
  //主要包括各种状态下的cache访问操作，例如(this->*m_wr_hit)、(this->*m_wr_miss)、
  //(this->*m_rd_hit)、(this->*m_rd_miss)。
  enum cache_request_status access_status =
      process_tag_probe(wr, probe_status, addr, cache_index, mf, time, events);
  m_stats.inc_stats(mf->get_access_type(),
                    m_stats.select_stats_status(probe_status, access_status),
                    mf->get_streamID());
  m_stats.inc_stats_pw(mf->get_access_type(),
                       m_stats.select_stats_status(probe_status, access_status),
                       mf->get_streamID());
  return access_status;
}

/// This is meant to model the first level data cache in Fermi.
/// It is write-evict (global) or write-back (local) at the
/// granularity of individual blocks (Set by GPGPU-Sim configuration file)
/// (the policy used in fermi according to the CUDA manual)
/*
这是为了对Fermi中的第一级数据缓存进行建模。它是单个块粒度的写逐出（global）或写回（local）
（由 GPGPU-Sim 配置文件设置）（根据 CUDA 手册在Fermi中使用的策略）。
*/
enum cache_request_status l1_cache::access(new_addr_type addr, mem_fetch *mf,
                                           unsigned time,
                                           std::list<cache_event> &events) {
  return data_cache::access(addr, mf, time, events);
}

// The l2 cache access function calls the base data_cache access
// implementation.  When the L2 needs to diverge from L1, L2 specific
// changes should be made here.
/*
l2 缓存访问函数调用基本data_cache访问实现。 当L2需要与L1有不一致的功能时，应在此处进行L2
的特定更改。
*/
enum cache_request_status l2_cache::access(new_addr_type addr, mem_fetch *mf,
                                           unsigned time,
                                           std::list<cache_event> &events) {
  return data_cache::access(addr, mf, time, events);
}

/// Access function for tex_cache
/// return values: RESERVATION_FAIL if request could not be accepted
/// otherwise returns HIT_RESERVED or MISS; NOTE: *never* returns HIT
/// since unlike a normal CPU cache, a "HIT" in texture cache does not
/// mean the data is ready (still need to get through fragment fifo)
enum cache_request_status tex_cache::access(new_addr_type addr, mem_fetch *mf,
                                            unsigned time,
                                            std::list<cache_event> &events) {
  if (m_fragment_fifo.full() || m_request_fifo.full() || m_rob.full())
    return RESERVATION_FAIL;

  assert(mf->get_data_size() <= m_config.get_line_sz());

  // at this point, we will accept the request : access tags and immediately
  // allocate line
  // m_config.block_addr(addr): 
  //     return addr & ~(new_addr_type)(m_line_sz - 1);
  // |-------|-------------|--------------|
  //            set_index   offset in-line
  // |<--------tag--------> 0 0 0 0 0 0 0 | 
  new_addr_type block_addr = m_config.block_addr(addr);
  unsigned cache_index = (unsigned)-1;
  enum cache_request_status status =
      m_tags.access(block_addr, time, cache_index, mf);
  enum cache_request_status cache_status = RESERVATION_FAIL;
  assert(status != RESERVATION_FAIL);
  assert(status != HIT_RESERVED);  // as far as tags are concerned: HIT or MISS
  m_fragment_fifo.push(
      fragment_entry(mf, cache_index, status == MISS, mf->get_data_size()));
  if (status == MISS) {
    // we need to send a memory request...
    unsigned rob_index = m_rob.push(rob_entry(cache_index, mf, block_addr));
    m_extra_mf_fields[mf] = extra_mf_fields(rob_index, m_config);
    mf->set_data_size(m_config.get_line_sz());
    m_tags.fill(cache_index, time, mf);  // mark block as valid
    m_request_fifo.push(mf);
    mf->set_status(m_request_queue_status, time);
    events.push_back(cache_event(READ_REQUEST_SENT));
    cache_status = MISS;
  } else {
    // the value *will* *be* in the cache already
    cache_status = HIT_RESERVED;
  }
  m_stats.inc_stats(mf->get_access_type(),
                    m_stats.select_stats_status(status, cache_status),
                    mf->get_streamID());
  m_stats.inc_stats_pw(mf->get_access_type(),
                       m_stats.select_stats_status(status, cache_status),
                       mf->get_streamID());
  return cache_status;
}

/*
TEX Cache 向前推进一拍。
*/
void tex_cache::cycle() {
  // send next request to lower level of memory
  if (!m_request_fifo.empty()) {
    mem_fetch *mf = m_request_fifo.peek();
    if (!m_memport->full(mf->get_ctrl_size(), false)) {
      m_request_fifo.pop();
      // mem_fetch_interface 是 cache 对 mem 访存的接口，cache 将 miss 请求发送至下一级
      // 存储就是通过这个接口来发送，即 m_miss_queue 中的数据包需要压入 m_memport 实现发
      // 送至下一级存储。
      m_memport->push(mf);
    }
  }
  // read ready lines from cache
  if (!m_fragment_fifo.empty() && !m_result_fifo.full()) {
    const fragment_entry &e = m_fragment_fifo.peek();
    if (e.m_miss) {
      // check head of reorder buffer to see if data is back from memory
      unsigned rob_index = m_rob.next_pop_index();
      const rob_entry &r = m_rob.peek(rob_index);
      assert(r.m_request == e.m_request);
      // assert( r.m_block_addr == m_config.block_addr(e.m_request->get_addr())
      // );
      if (r.m_ready) {
        assert(r.m_index == e.m_cache_index);
        m_cache[r.m_index].m_valid = true;
        m_cache[r.m_index].m_block_addr = r.m_block_addr;
        m_result_fifo.push(e.m_request);
        m_rob.pop();
        m_fragment_fifo.pop();
      }
    } else {
      // hit:
      assert(m_cache[e.m_cache_index].m_valid);
      assert(m_cache[e.m_cache_index].m_block_addr ==
             m_config.block_addr(e.m_request->get_addr()));
      m_result_fifo.push(e.m_request);
      m_fragment_fifo.pop();
    }
  }
}

/// Place returning cache block into reorder buffer
void tex_cache::fill(mem_fetch *mf, unsigned time) {
  if (m_config.m_mshr_type == SECTOR_TEX_FIFO) {
    assert(mf->get_original_mf());
    extra_mf_fields_lookup::iterator e =
        m_extra_mf_fields.find(mf->get_original_mf());
    assert(e != m_extra_mf_fields.end());
    e->second.pending_read--;

    if (e->second.pending_read > 0) {
      // wait for the other requests to come back
      delete mf;
      return;
    } else {
      mem_fetch *temp = mf;
      mf = mf->get_original_mf();
      delete temp;
    }
  }

  extra_mf_fields_lookup::iterator e = m_extra_mf_fields.find(mf);
  assert(e != m_extra_mf_fields.end());
  assert(e->second.m_valid);
  assert(!m_rob.empty());
  mf->set_status(m_rob_status, time);

  unsigned rob_index = e->second.m_rob_index;
  rob_entry &r = m_rob.peek(rob_index);
  assert(!r.m_ready);
  r.m_ready = true;
  r.m_time = time;
  assert(r.m_block_addr == m_config.block_addr(mf->get_addr()));
}

void tex_cache::display_state(FILE *fp) const {
  fprintf(fp, "%s (texture cache) state:\n", m_name.c_str());
  fprintf(fp, "fragment fifo entries  = %u / %u\n", m_fragment_fifo.size(),
          m_fragment_fifo.capacity());
  fprintf(fp, "reorder buffer entries = %u / %u\n", m_rob.size(),
          m_rob.capacity());
  fprintf(fp, "request fifo entries   = %u / %u\n", m_request_fifo.size(),
          m_request_fifo.capacity());
  if (!m_rob.empty()) fprintf(fp, "reorder buffer contents:\n");
  for (int n = m_rob.size() - 1; n >= 0; n--) {
    unsigned index = (m_rob.next_pop_index() + n) % m_rob.capacity();
    const rob_entry &r = m_rob.peek(index);
    fprintf(fp, "tex rob[%3d] : %s ", index,
            (r.m_ready ? "ready  " : "pending"));
    if (r.m_ready)
      fprintf(fp, "@%6u", r.m_time);
    else
      fprintf(fp, "       ");
    fprintf(fp, "[idx=%4u]", r.m_index);
    r.m_request->print(fp, false);
  }
  if (!m_fragment_fifo.empty()) {
    fprintf(fp, "fragment fifo (oldest) :");
    fragment_entry &f = m_fragment_fifo.peek();
    fprintf(fp, "%s:          ", f.m_miss ? "miss" : "hit ");
    f.m_request->print(fp, false);
  }
}
/******************************************************************************************************************************************/
