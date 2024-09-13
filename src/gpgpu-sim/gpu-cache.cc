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
  //�����������棬����Ϊÿ���洢��ѡ��һ������������������
  //������Volta�з��ֵģ�ÿ���洢��һ�����������������������н���

  // �����Ǽ��� L1 �� bank index�������Ǽ��� set index���������� set_index �Ĺ�ϣ������
  return cache_config::hash_function(addr, l1_banks,
                                     l1_banks_byte_interleaving_log2,
                                     l1_banks_log2, l1_banks_hashing_function);
}

/*
����һ����ַ��Cache�е�set��
*/
unsigned cache_config::set_index(new_addr_type addr) const {
  // m_line_sz_log2 = LOGB2(m_line_sz);
  // m_nset_log2 = LOGB2(m_nset);
  // m_set_index_function = L1D��"L"-LINEAR_SET_FUNCTION��L2D��"P"-HASH_IPOLY_FUNCTION��
  return cache_config::hash_function(addr, m_nset, m_line_sz_log2, m_nset_log2,
                                     m_set_index_function);
}

/*
����һ����ַ��Cache�е�set��
m_line_sz_log2 = LOGB2(m_line_sz);
m_nset_log2 = LOGB2(m_nset);
m_set_index_function = L1D��"L"-LINEAR_SET_FUNCTION��L2D��"P"-HASH_IPOLY_FUNCTION��
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

    // V100���õ�L2D Cache��
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

    // V100���õ�L1D Cache��
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
����cache_config m_config������
*/
void tag_array::update_cache_parameters(cache_config &config) {
  m_config = config;
}

tag_array::tag_array(cache_config &config, int core_id, int type_id)
    : m_config(config) {
  // assert( m_config.m_write_policy == READ_ONLY ); Old assert
  // config.get_max_num_lines() ��������Ϊ��L1D cache������Ϊ128KBʱ����Ҫ��
  // cache block��Ŀ�϶࣬����Ҫ��֤cache_lines_num�㹻�ã������max_num_lines
  // �����������£���Ҫ�ö���cache blocks��
  unsigned cache_lines_num = config.get_max_num_lines();
  // ���е�cache blocks��������m_lines��m_lines������Ϊ��
  //   cache_block_t **m_lines; /* nbanks x nset x assoc lines in total */
  // ���m_lines[...]��ָ�򵥸�cache block��ָ�롣
  m_lines = new cache_block_t *[cache_lines_num];
  // ����Ϳ�ʼ������ line cache �� sector cache �ˡ�
  if (config.m_cache_type == NORMAL) {
    for (unsigned i = 0; i < cache_lines_num; ++i)
      m_lines[i] = new line_cache_block();
  } else if (config.m_cache_type == SECTOR) {
    for (unsigned i = 0; i < cache_lines_num; ++i)
      m_lines[i] = new sector_cache_block();
  } else
    assert(0);

  // ��ʼ��һЩͳ�Ʋ�����
  init(core_id, type_id);
}

void tag_array::init(int core_id, int type_id) {
  //���ʵ�ǰcache�Ĵ�������tag_array::access()���������õĴ�����
  m_access = 0;
  //��ǰcache��miss��������tag_array::access()��������MISS�Ĵ�����
  m_miss = 0;
  //��ǰcache��pending hit��������tag_array::access()��������HIT_RESERVED�Ĵ�����
  m_pending_hit = 0;
  //��ǰcache��reservation fail��������tag_array::access()��������RESERVATION_FAIL�Ĵ�����
  m_res_fail = 0;
  //��ǰcache��sector miss��������tag_array::access()��������SECTOR_MISS�Ĵ�����
  m_sector_miss = 0;
  // initialize snapshot counters for visualizer
  m_prev_snapshot_access = 0;
  m_prev_snapshot_miss = 0;
  m_prev_snapshot_pending_hit = 0;
  //SM_ID������L1 cache��������core_id������L2 cache��������-1��
  m_core_id = core_id;
  //Cache����ID��
  //    enum cache_access_logger_types { NORMALS, TEXTURE, CONSTANT, INSTRUCTION };
  //����L1 cache��������type_id������L2 cache��������-1��
  m_type_id = type_id;
  //a flag if the whole cache has ever been accessed before
  is_used = false;
  //Dirty block�ĸ�����
  m_dirty = 0;
}

// �Ѿ������ˡ�
void tag_array::add_pending_line(mem_fetch *mf) {
  assert(mf);
  new_addr_type addr = m_config.block_addr(mf->get_addr());
  line_table::const_iterator i = pending_lines.find(addr);
  if (i == pending_lines.end()) {
    pending_lines[addr] = mf->get_inst().get_uid();
  }
}

// �Ѿ������ˡ�
void tag_array::remove_pending_line(mem_fetch *mf) {
  assert(mf);
  new_addr_type addr = m_config.block_addr(mf->get_addr());
  line_table::const_iterator i = pending_lines.find(addr);
  if (i != pending_lines.end()) {
    pending_lines.erase(addr);
  }
}

/*
�ж϶�cache�ķ��ʣ���ַΪaddr��sector maskΪmask����HIT/HIT_RESERVED/SECTOR_MISS/MISS
/RESERVATION_FAIL��״̬��
��һ��cache�������ݷ��ʵ�ʱ�򣬵���data_cache::access()������
- ����cahe�����m_tag_array->probe()�������ж϶�cache�ķ��ʣ���ַΪaddr��sector mask
  Ϊmask����HIT/HIT_RESERVED/SECTOR_MISS/MISS/RESERVATION_FAIL��״̬��
- Ȼ�����process_tag_probe()����������cache�������Լ�����m_tag_array->probe()������
  �ص�cache����״̬��ִ����Ӧ�Ĳ�����
  - process_tag_probe()�����У����������Ķ�д״̬��probe()�������ص�cache����״̬��
    ִ��m_wr_hit/m_wr_miss/m_rd_hit/m_rd_miss���������ǻ����m_tag_array->access()
    ������ʵ��LRU״̬�ĸ��¡�
*/
enum cache_request_status tag_array::probe(new_addr_type addr, unsigned &idx,
                                           mem_fetch *mf, bool is_write,
                                           bool probe_mode) const {
  mem_access_sector_mask_t mask = mf->get_access_sector_mask();
  return probe(addr, idx, mask, is_write, probe_mode, mf);
}

/*
�ж϶�cache�ķ��ʣ���ַΪaddr��sector maskΪmask����HIT/HIT_RESERVED/SECTOR_MISS/MISS
/RESERVATION_FAIL��״̬��
��һ��cache�������ݷ��ʵ�ʱ�򣬵���data_cache::access()������
- ����cahe�����m_tag_array->probe()�������ж϶�cache�ķ��ʣ���ַΪaddr��sector mask
  Ϊmask����HIT/HIT_RESERVED/SECTOR_MISS/MISS/RESERVATION_FAIL��״̬��
- Ȼ�����process_tag_probe()����������cache�������Լ�����m_tag_array->probe()������
  �ص�cache����״̬��ִ����Ӧ�Ĳ�����
  - process_tag_probe()�����У����������Ķ�д״̬��probe()�������ص�cache����״̬��
    ִ��m_wr_hit/m_wr_miss/m_rd_hit/m_rd_miss���������ǻ����m_tag_array->access()
    ������ʵ��LRU״̬�ĸ��¡�

����Ҫ����line cache��sector cache�ľ���ִ�й��̡�
*/
enum cache_request_status tag_array::probe(new_addr_type addr, unsigned &idx,
                                           mem_access_sector_mask_t mask,
                                           bool is_write, bool probe_mode,
                                           mem_fetch *mf) const {
  //����������ַaddr��cache block�ĵ�ַ���õ�ַ��Ϊ��ַaddr��tagλ+set indexλ������
  //offsetλ���������λ��
  //  |-------|-------------|--------------|
  //             set_index   offset in-line
  //  |<--------tag--------> 0 0 0 0 0 0 0 |

  // assert( m_config.m_write_policy == READ_ONLY );
  //����һ����ַaddr��Cache�е�set index�������set index��һ���׵�ӳ�亯����
  unsigned set_index = m_config.set_index(addr);
  //Ϊ�˱������������ı�ǰ���index��Tag������������ӵģ����ܵ��²�ͬ��indexesӳ�䵽
  //ͬһset��set index���㣬�����Ҫ�����ı�ǩ + �������������/δ���С�Tag��������ַ
  //��ͬ��
  //����ʵ�ʷ��ص���{��offsetλ���������λ, offset'b0}����set indexҲ��Ϊtag��һ�����ˡ�
  new_addr_type tag = m_config.tag(addr);

  unsigned invalid_line = (unsigned)-1;
  unsigned valid_line = (unsigned)-1;
  unsigned long long valid_timestamp = (unsigned)-1;

  bool all_reserved = true;
  // check for hit or pending hit
  //�����е�Cache Ways��顣��Ҫע��������ʵ�����һ��set������way���м�飬��Ϊ������һ��
  //��ַ�����ǿ���ȷ�������ڵ�set index��Ȼ����ͨ��tag��ȷ�������ַ����һ��way�ϡ�
  for (unsigned way = 0; way < m_config.m_assoc; way++) {
    // For example, 4 sets, 6 ways:
    // |  0  |  1  |  2  |  3  |  4  |  5  |  // set_index 0
    // |  6  |  7  |  8  |  9  |  10 |  11 |  // set_index 1
    // |  12 |  13 |  14 |  15 |  16 |  17 |  // set_index 2
    // |  18 |  19 |  20 |  21 |  22 |  23 |  // set_index 3
    //                |--------> index => cache_block_t *line
    // cache block��������
    unsigned index = set_index * m_config.m_assoc + way;
    cache_block_t *line = m_lines[index];
    // Tag�����m_tag��tag���ǣ�{��offsetλ���������λ, offset'b0}
    if (line->m_tag == tag) {
      // enum cache_block_state { INVALID = 0, RESERVED, VALID, MODIFIED };
      // cache block��״̬��������
      //   INVALID: Cache block��Ч���������е�byte mask=Cache block[mask]״̬INVALID��
      //           ˵��sectorȱʧ��
      //   MODIFIED: ���Cache block[mask]״̬��MODIFIED��˵���Ѿ��������߳��޸ģ������
      //             ǰ����Ҳ��д�����Ļ���Ϊ���У����������д��������Ҫ�ж��Ƿ�mask��־��
      //             ���Ƿ��޸���ϣ��޸������Ϊ���У��޸Ĳ������ΪSECTOR_MISS����ΪL1 
      //             cache��L2 cacheд����ʱ������write-back���ԣ�ֻ������д���block��
      //             ����ֱ�Ӹ����¼��洢��ֻ�е�����鱻�滻ʱ���Ž�����д���¼��洢��
      //   VALID: ���Cache block[mask]״̬��VALID��˵���Ѿ����С�
      //   RESERVED: Ϊ��δ��ɵĻ���δ���е������ṩ�ռ䡣Cache block[mask]״̬RESERVED��
      //             ˵�����������߳����ڶ�ȡ���Cache block����������з��������д���RE-
      //             SERVED״̬�Ļ����У���ζ��ͬһ�����Ѵ�������ǰ����δ���з��͵�flying
      //             �ڴ�����
      if (line->get_status(mask) == RESERVED) {
        //���Cache block[mask]״̬��RESERVED��˵�����������߳����ڶ�ȡ���Cache block��
        //��������з��������д���RESERVED״̬�Ļ����У�����ζ��ͬһ�����Ѵ�������ǰ����δ
        //���з��͵�flying�ڴ�����
        idx = index;
        return HIT_RESERVED;
      } else if (line->get_status(mask) == VALID) {
        //���Cache block[mask]״̬��VALID��˵���Ѿ����С�
        idx = index;
        return HIT;
      } else if (line->get_status(mask) == MODIFIED) {
        //���Cache block[mask]״̬��MODIFIED��˵���Ѿ��������߳��޸ģ������ǰ����Ҳ��д
        //�����Ļ���Ϊ���У����������д��������Ҫ�ж��Ƿ�mask��־�Ŀ��Ƿ��޸���ϣ��޸����
        //��Ϊ���У��޸Ĳ������ΪSECTOR_MISS����ΪL1 cache��L2 cacheд����ʱ������write-
        //back���ԣ�ֻ������д���block������ֱ�Ӹ����¼��洢��ֻ�е�����鱻�滻ʱ���Ž���
        //��д���¼��洢��
        //is_readable(mask)���ж�mask��־��sector�Ƿ��Ѿ�ȫ��д��ɣ���Ϊ���޸�cache�Ĺ���
        //�У���һ��sector���޸ļ�������ǰcache��MODIFIED�������޸Ĺ��̿��ܲ���һ�¾���д�꣬
        //�����Ҫ�ж�һ���Ƿ�ȫ����ǰmask���������sectorд��ſ������������С�
        if ((!is_write && line->is_readable(mask)) || is_write) {
          // ��ǰline��maskλ���޸ģ������д������ν������Ȼ���У�ֱ�Ӹ���д���ɣ��������
          // �Ƕ�������Ҫ��maskλ�Ƿ��ǿɶ��ġ�����ǿɶ��ģ���Ϊ���С�
          idx = index;
          return HIT;
        } else {
          // ���������֧�������ǣ�is_writeΪfalse����ǰ�����Ƕ���line->is_readable(mask)
          // Ϊfalse��maskλ���ǿɶ��ģ���˵����ǰ����sectorȱʧ��
          idx = index;
          return SECTOR_MISS;
        }

      } else if (line->is_valid_line() && line->get_status(mask) == INVALID) {
        // ����line cache�����������֧����Ϊline cache�У�line->is_valid_line()���ص���
        // m_status��ֵ������Ϊ VALID ʱ��line cache��line->get_status(mask)Ҳ�Ƿ��ص�
        // Ҳ��m_status��ֵ����Ϊ VALID����˶���line cache������֧��Ч��
        // ���Ƕ���sector cache�� �У�
        //   virtual bool is_valid_line() { return !(is_invalid_line()); }
        // ��sector cache�е�is_invalid_line()�ǣ�ֻҪ��һ��sector��ΪINVALID������false��
        // ���is_valid_line()���ص��ǣ�ֻҪ��һ��sector��ΪINVALID������is_valid_line()
        // Ϊ�档����������֧����sector cache�ǿ��ߵġ�
        //Cache block��Ч���������е�byte mask=Cache block[mask]״̬��Ч��˵��sectorȱʧ��
        idx = index;
        return SECTOR_MISS;
      } else {
        assert(line->get_status(mask) == INVALID);
      }
    }
    
    //ÿһ��ѭ�������ߵ�����ģ���Ϊ��ǰcache block��line->m_tag!=tag����ô����Ҫ���ǵ�ǰ��
    //cache block�ܷ�����滻����ע�⣬����ж����ڶ�ÿһ��wayѭ���Ĺ����н��еģ�Ҳ����˵��
    //�����һ��cache blockû�з������Ϸ���״̬�����п���ֱ������way�����һ��cache block��
    //����line->m_tag!=tag�������ڶԵ�0~way-2�ŵ�cache blockѭ���жϵ�ʱ�򣬾���Ҫ��¼��ÿ
    //һ��way��cache block�Ƿ��ܹ����������Ϊ����ȵ�����way��cache block��û������line->
    //m_tag!=tagʱ���ٻع�ͷ��ѭ������way�������ȱ������cache block�Ǿ�������ģ��Ŀ�����
    //���ʵ���϶�������way�е�ÿһ��cache block��ֻҪ��������line->m_tag!=tag������������
    //�����ܷ������
    // cache block��״̬��������
    //   INVALID: Cache block��Ч���������е�byte mask=Cache block[mask]״̬INVALID��
    //           ˵��sectorȱʧ��
    //   MODIFIED: ���Cache block[mask]״̬��MODIFIED��˵���Ѿ��������߳��޸ģ������
    //             ǰ����Ҳ��д�����Ļ���Ϊ���У����������д��������Ҫ�ж��Ƿ�mask��־��
    //             ���Ƿ��޸���ϣ��޸������Ϊ���У��޸Ĳ������ΪSECTOR_MISS����ΪL1 
    //             cache��L2 cacheд����ʱ������write-back���ԣ�ֻ������д���block��
    //             ����ֱ�Ӹ����¼��洢��ֻ�е�����鱻�滻ʱ���Ž�����д���¼��洢��
    //   VALID: ���Cache block[mask]״̬��VALID��˵���Ѿ����С�
    //   RESERVED: Ϊ��δ��ɵĻ���δ���е������ṩ�ռ䡣Cache block[mask]״̬RESERVED��
    //             ˵�����������߳����ڶ�ȡ���Cache block����������з��������д���RE-
    //             SERVED״̬�Ļ����У���ζ��ͬһ�����Ѵ�������ǰ����δ���з��͵�flying
    //             �ڴ�����
    //line->is_reserved_line()��ֻҪ��һ��sector��RESERVED������Ϊ���Cache Line��RESERVED��
    //���Ｔ����lineû��sector��RESERVED��
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
      //m_config.m_wr_percent��V100������Ϊ25%��
      //line->is_modified_line()��ֻҪ��һ��sector��MODIFIED������Ϊ���cache line��MODIFIED��
      //���Ｔ����lineû��sector��MODIFIED������dirty_line_percentage����m_wr_percent��
      if (!line->is_modified_line() ||
          dirty_line_percentage >= m_config.m_wr_percent) {
        //һ��cache line��״̬�У�INVALID = 0, RESERVED, VALID, MODIFIED���������VALID��
        //��������Ĵ��������ˡ�
        //��Ϊ�����һ��cache��ʱ���������һ���ɾ��Ŀ飬��û��sector��RESERVED��Ҳû��sector
        //��MODIFIED����������������dirty��cache line�ı�������m_wr_percent��V100������Ϊ
        //25%����Ҳ���Բ�����MODIFIED��������
        //�ڻ����������У��������δ���޸ģ�"�ɾ�"���Ļ����Ĳ��ԣ��ǻ��ڼ�����Ҫ�Ŀ��ǣ�
        // 1. ����д�سɱ��������е�����ͨ����Դ�ڸ����ٵĺ�˴洢�������洢������������鱻�޸�
        //   ��������"��"���ݣ�ʱ���������Щ��֮ǰ����Ҫ����Щ����д�ص���˴洢��ȷ������һ���ԡ�
        //    ���֮�£�δ���޸ģ�"�ɾ�"���Ļ�������ֱ�ӱ��������Ϊ���ǵ������Ѿ����˴洢һ
        //    �£��������д�ز����������ͱ�����д�ز���������ʱ�������������
        // 2. ���Ч�ʣ�д�ز�������ڶ�ȡ������˵����һ���ɱ��ϸߵĹ��̣������漰�����ʱ���ӳ٣�
        //    ������ռ�ñ���Ĵ���Ӱ��ϵͳ���������ܡ�ͨ���������Щ"�ɾ�"�Ŀ飬ϵͳ�ܹ���ά��
        //    ����һ���Ե�ǰ���£����ٶԺ�˴洢����������д�ز����Ŀ�����
        // 3. �Ż����ܣ�ѡ�����"�ɾ�"�Ļ���黹������ά������ĸ������ʡ���������£�����Ӧ����
        //    ������Ƶ�ʸ�����������ʵ����ݡ����"��"������ζ����Щ������Ҫ��д�أ�������̲���
        //    ��ʱ���ҿ��ܵ��»�����ʱ�޷������������󣬴Ӷ����ͻ���Ч�ʡ�
        // 4. ���ݰ�ȫ�������ԣ���ĳЩ����£�"��"�������ܱ�ʾ���ڽ��е�д����������Ҫ�����ݸ�
        //    �¡�ͨ���������"�ɾ�"�Ļ���飬���Խ�����Ϊ����������µ����ݶ�ʧ�����������ƻ���
        //    ���ա�
        
        //all_reserved����ʼ��Ϊtrue����ָ����cache line��û���ܹ������Ϊ�·����ṩRESERVE
        //�Ŀռ䣬����һ��������������if������˵����ǰline���Ա�������ṩ�ռ乩RESERVE�·��ʣ�
        //����all_reserved��Ϊfalse����һ������all_reserved�Ծɱ���true�Ļ�����˵����ǰset��
        //û����һ��way��cache block���Ա����������RESERVATION_FAIL��
        all_reserved = false;
        //line->is_invalid_line()������sector����Ч��
        if (line->is_invalid_line()) {
          //��Ȼ�ˣ�����������LRU����FIFO�滻���ԣ������������������������滻����cache block
          //����Ч�Ŀ顣��Ϊ������Ч�Ŀ鲻��Ҫд�أ��ܹ���ʡ����
          invalid_line = index;
        } else {
          // valid line : keep track of most appropriate replacement candidate
          if (m_config.m_replacement_policy == LRU) {
            //valid_timestamp����Ϊ������ٱ�ʹ�õ�cache line����ĩ�η���ʱ�䡣
            //valid_timestamp����ʼ��Ϊ(unsigned)-1�������Կ��������
            if (line->get_last_access_time() < valid_timestamp) {
              //�����valid_timestamp��������������С���������������ı�������ȼ�����Ȼ���
              //����������ֻ���Ҿ�����С��������cache block����С��������ζ�������ϴ�ʹ�ò���
              //�磬������ʶ�ĸ�cache block����������ȼ����������valid_line��
              valid_timestamp = line->get_last_access_time();
              //��ʶ��ǰcache block������С��ִ����������index���cache blockӦ�����ȱ������
              valid_line = index;
            }
          } else if (m_config.m_replacement_policy == FIFO) {
            if (line->get_alloc_time() < valid_timestamp) {
              //FIFO�����������ʱ���cache block�����ȱ������
              valid_timestamp = line->get_alloc_time();
              valid_line = index;
            }
          }
        }
      }
    } //�����ǰѵ�ǰset�����е�way��ѭ��һ�飬����ҵ���line->m_tag == tag�Ŀ飬���Ѿ�������
      //����״̬�����û���ҵ�����Ҳ������һ������way��cache block���ҵ���������Ӧ�ñ������
      //�滻��cache block��
  }
  //Cache���ʵ�״̬������
  //    HIT��HIT_RESERVED��MISS��RESERVATION_FAIL��SECTOR_MISS��MSHR_HIT����״̬��
  //�׿�ǰ���ܹ�ȷ����HIT��HIT_RESERVED��SECTOR_MISS���ܹ��ж�MISS/RESERVATION_FAIL
  //����״̬�Ƿ������
  //��Ϊ�����һ��cache��ʱ���������һ���ɾ��Ŀ飬��û��sector��RESERVED��Ҳû��sector
  //��MODIFIED����������������dirty��cache line�ı�������m_wr_percent��V100������Ϊ
  //25%����Ҳ���Բ�����MODIFIED��������
  //all_reserved����ʼ��Ϊtrue����ָ����cache line��û���ܹ������Ϊ�·����ṩRESERVE
  //�Ŀռ䣬����һ��������������if������˵��cache line���Ա�������ṩ�ռ乩RESERVE�·�
  //�ʣ�����all_reserved��Ϊfalse����һ������all_reserved�Ծɱ���true�Ļ�����˵��cache
  //line���ɱ����������RESERVATION_FAIL��
  if (all_reserved) {
    //all_reservedΪtrue�Ļ���������ǰset������way��û��cache���㱻��������������״̬
    //����RESERVATION_FAIL����all of the blocks in the current set have no enough 
    //space in cache to allocate on miss.
    assert(m_config.m_alloc_policy == ON_MISS);
    return RESERVATION_FAIL;  // miss and not enough space in cache to allocate
                              // on miss
  }

  //��������all_reservedΪfalse���Żᵽ��һ������cache line���Ա������Ϊ�·����ṩ
  //RESERVE��
  if (invalid_line != (unsigned)-1) {
    //����������LRU����FIFO�滻���ԣ������������������������滻����cache block����Ч
    //�Ŀ顣��Ϊ������Ч�Ŀ鲻��Ҫд�أ��ܹ���ʡ����
    idx = invalid_line;
  } else if (valid_line != (unsigned)-1) {
    //û����Ч�Ŀ飬��ֻ�ܽ����水��LRU����FIFOȷ����cache block��Ϊ������Ŀ��ˡ�
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

  //��������cache line���Ա������reserve�·��ʣ��򷵻�MISS��
  return MISS;
}


/*
����LRU״̬��Least Recently Used�������Ƿ���Ҫд��wb�Լ������cache line����Ϣevicted��
��һ��cache�������ݷ��ʵ�ʱ�򣬵���data_cache::access()������
- ����cahe�����m_tag_array->probe()�������ж϶�cache�ķ��ʣ���ַΪaddr��sector mask
  Ϊmask����HIT/HIT_RESERVED/SECTOR_MISS/MISS/RESERVATION_FAIL��״̬��
- Ȼ�����process_tag_probe()����������cache�������Լ�����m_tag_array->probe()������
  �ص�cache����״̬��ִ����Ӧ�Ĳ�����
  - process_tag_probe()�����У����������Ķ�д״̬��probe()�������ص�cache����״̬��
    ִ��m_wr_hit/m_wr_miss/m_rd_hit/m_rd_miss���������ǻ����m_tag_array->access()
    ������ʵ��LRU״̬�ĸ��¡�
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
����LRU״̬��Least Recently Used�������Ƿ���Ҫд��wb�Լ������cache line����Ϣevicted��
��һ��cache�������ݷ��ʵ�ʱ�򣬵���data_cache::access()������
- ����cahe�����m_tag_array->probe()�������ж϶�cache�ķ��ʣ���ַΪaddr��sector mask
  Ϊmask����HIT/HIT_RESERVED/SECTOR_MISS/MISS/RESERVATION_FAIL��״̬��
- Ȼ�����process_tag_probe()����������cache�������Լ�����m_tag_array->probe()������
  �ص�cache����״̬��ִ����Ӧ�Ĳ�����
  - process_tag_probe()�����У����������Ķ�д״̬��probe()�������ص�cache����״̬��
    ִ��m_wr_hit/m_wr_miss/m_rd_hit/m_rd_miss���������ǻ����m_tag_array->access()
    ������ʵ��LRU״̬�ĸ��¡�
*/
enum cache_request_status tag_array::access(new_addr_type addr, unsigned time,
                                            unsigned &idx, bool &wb,
                                            evicted_block_info &evicted,
                                            mem_fetch *mf) {
  // �Ե�ǰ tag_array �ķ��ʴ����� 1��
  m_access++;
  // ��ǵ�ǰ tag_array ���� cache �Ƿ�ʹ�ù���һ���� access() ���������ã���
  // ˵����ʹ�ù���
  is_used = true;
  shader_cache_access_log(m_core_id, m_type_id, 0);  // log accesses to cache
  // ���ڵ�ǰ����û�а�֮ǰ probe ������ cache ����״̬���ν������������ probe 
  // ���������»�ȡ���״̬��
  enum cache_request_status status = probe(addr, idx, mf, mf->is_write());
  switch (status) {
    // �·����� HIT_RESERVED �Ļ�����ִ�ж�����
    case HIT_RESERVED:
      m_pending_hit++;
    // �·����� HIT �Ļ������õ� idx �� cache line �Լ� mask ��Ӧ�� sector ����
    // ĩ�˷���ʱ��Ϊ��ǰ�ġ�
    case HIT:
      m_lines[idx]->set_last_access_time(time, mf->get_access_sector_mask());
      break;
    // �·����� MISS �Ļ���˵���Ѿ�ѡ�� m_lines[idx] ��Ϊ����� reserve �·��ʵ�
    // cache line��
    case MISS:
      m_miss++;
      shader_cache_access_log(m_core_id, m_type_id, 1);  // log cache misses
      // For V100, L1 cache and L2 cache are all `allocate on miss`.
      // m_alloc_policy��������ԣ�
      //     ���ڷ��͵� L1D cache ������������У������������������ݣ����δ���У�
      //     ������뻺��δ������ص���Դ��������ת�� L2 cache��allocate-on-miss 
      //     �� allocateon-fill �����ֻ����з�����ԡ����� allocateon-miss����Ҫ
      //     Ϊδ��ɵ�δ���з���һ�������вۡ�һ�� MSHR ��һ��δ���ж�����Ŀ�����
      //     ֮�£�allocate-on-fill����δ��ɵ�δ���з���ʱ����Ҫ����һ�� MSHR ��
      //     һ��δ���ж�����Ŀ�������������ݴӽϵ��ڴ漶�𷵻�ʱ����ѡ���ܺ��߻���
      //     �вۡ��������ֲ����У�����κ�������Դ�����ã���ᷢ��Ԥ��ʧ�ܣ��ڴ��
      //     ����ͣ�͡������ MSHR �ᱻ������ֱ���� L2 ����/Ƭ���ڴ��л�ȡ���ݣ���
      //     δ���ж�����Ŀ����δ��������ת���� L2 ������ͷš����� allocate-on-
      //     fill ������֮ǰ���ܺ��߻����б����ڻ����и���ʱ�䣬��Ϊδ��ɵ�δ����
      //     �������ٵ���Դ������������ܻ�ø���Ļ������к͸��ٵ�Ԥ��ʧ�ܣ��Ӷ��� 
      //     allocate-on-miss ���и��õ����ܡ��������ʱ������Ҫ����Ļ����������
      //     �߼�����˳��������䵽�����У�����˳��ִ��ģ�ͺ�д���������ʹ GPU 
      //     L1D ��������ʱ������Ѻã���Ϊ�����ʱҪ�����ܺ��߻���ʱ��û��������
      //     д�� L2��
      //     ��� paper��
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
        // ����ʱ���� MISS��˵�� probe ȷ���� idx �� cache line ��Ҫ�������Ϊ��
        // �����ṩ RESERVE �Ŀռ䡣���ǣ�������Ҫ�ж� idx �� cache line �Ƿ��� 
        // MODIFIED������ǵĻ�����Ҫִ��д�أ�����д�صı�־Ϊ wb = true��������
        // �� cache line ����Ϣ��
        if (m_lines[idx]->is_modified_line()) {
          // m_lines[idx] ��Ϊ����� reserve �·��ʵ� cache line���������ĳ�� 
          // sector �Ѿ��� MODIFIED������Ҫִ��д�ز���������д�ر�־Ϊ wb = true��
          // ������� cache line ����Ϣ��
          wb = true;
          // m_lines[idx]->set_byte_mask(mf);
          evicted.set_info(m_lines[idx]->m_block_addr,
                           m_lines[idx]->get_modified_size(),
                           m_lines[idx]->get_dirty_byte_mask(),
                           m_lines[idx]->get_dirty_sector_mask());
          // ����ִ��д�ز�����MODIFIED ��ɵ� m_dirty ����Ӧ�ü�1��
          m_dirty--;
        }
        // ִ�ж��·��ʵ� reserve ������
        m_lines[idx]->allocate(m_config.tag(addr), m_config.block_addr(addr),
                               time, mf->get_access_sector_mask());
      }
      break;
    // Cache block ��Ч���������е� byte mask = Cache block[mask] ״̬��Ч��˵��
    // sector ȱʧ��
    case SECTOR_MISS:
      assert(m_config.m_cache_type == SECTOR);
      m_sector_miss++;
      shader_cache_access_log(m_core_id, m_type_id, 1);  // log cache misses
      // For V100, L1 cache and L2 cache are all `allocate on miss`.
      if (m_config.m_alloc_policy == ON_MISS) {
        bool before = m_lines[idx]->is_modified_line();
        // ���� m_lines[idx] Ϊ�·��ʷ���һ�� sector��
        ((sector_cache_block *)m_lines[idx])
            ->allocate_sector(time, mf->get_access_sector_mask());
        if (before && !m_lines[idx]->is_modified_line()) {
          m_dirty--;
        }
      }
      break;
    // probe�����У�
    // all_reserved ����ʼ��Ϊ true����ָ���� cache line ��û���ܹ������Ϊ�·���
    // �ṩ RESERVE �Ŀռ䣬����һ�����㺯������ if ������˵�� cache line ���Ա���
    // �����ṩ�ռ乩 RESERVE �·��ʣ����� all_reserved ��Ϊ false��
    // ��һ������ all_reserved �Ծɱ��� true �Ļ�����˵�� cache line ���ɱ������
    // ���� RESERVATION_FAIL��������ﲻִ���κβ�����
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
  //��V100�У�L1 cache��L2 cache��Ϊallocate on miss��
  //allocate-on-miss����дΪon-miss����allocate-on-fill����дΪon-fill��������cache�з�����ԣ�
  //(1) allocate-on-miss��������δ��ɵ�cache missʱ����ҪΪδ��ɵ�miss����cache line slot��
  //    MSHR��miss������Ŀ��
  //(2) allocateon-on-fill��������δ��ɵ�cache missʱ����ҪΪδ��ɵ�miss����MSHR��miss������
  //    Ŀ�������������ݴӽϵ��ڴ漶�𷵻�ʱ����ѡ���ܺ���cache line slot�滻��
  //�������ֲ����У�����κ�������Դ�����ã���ᷢ��reservation failure�������ڴ���ˮ��ͣ�١�����
  //��MSHR��������ֱ�����ݴӽϵ�һ���ڴ���ȡ�أ���һ��miss����ת����L2 cache�����ͷ�miss������Ŀ��
  //allocateon-on-fill������allocate-on-miss�и��õ����ܣ���Ϊ���������ܺ���cache������ʱ�䣬��
  //��Ϊδ��ɵ�miss�������ٵ���Դ���Ӷ����ܸ����cache hit�͸��ٵ�reservation failure������all-
  //ocateon-on-fill��Ҫ�����buffer�������߼��������ݰ�˳����䵽cache������˳��ִ��ģ�ͺ�write-
  //evict����ʹ��GPU L1 D-cache��allocateon-on-fill���Ѻã���Ϊ���ܺ���cache�ڱ�evictʱ��û����
  //���ݱ�д��L2��
  assert(m_config.m_alloc_policy == ON_MISS);
  //before�Ǽ�¼�����֮ǰ��m_lines[index]�Ƿ���MODIFIED״̬��
  bool before = m_lines[index]->is_modified_line();
  m_lines[index]->fill(time, mf->get_access_sector_mask(),
                       mf->get_access_byte_mask());
  //�������Ժ�m_lines[index]�� MODIFIED ״̬�����������֮ǰ���� MODIFIED ״̬��˵�����cache 
  //line����ģ�m_dirty������1��
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
�ж�һϵ�еķ���cache�¼��Ƿ����WRITE_REQUEST_SENT��
�����¼����Ͱ�����
    enum cache_event_type {
      //д������
      WRITE_BACK_REQUEST_SENT,
      //������
      READ_REQUEST_SENT,
      //д����
      WRITE_REQUEST_SENT,
      //д��������
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
�ж�һϵ�еķ���cache�¼��Ƿ����WRITE_BACK_REQUEST_SENT��
�����¼����Ͱ�����
    enum cache_event_type {
      //д������
      WRITE_BACK_REQUEST_SENT,
      //������
      READ_REQUEST_SENT,
      //д����
      WRITE_REQUEST_SENT,
      //д��������
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
�ж�һϵ�еķ���cache�¼��Ƿ����READ_REQUEST_SENT��
�����¼����Ͱ�����
    enum cache_event_type {
      //д������
      WRITE_BACK_REQUEST_SENT,
      //������
      READ_REQUEST_SENT,
      //д����
      WRITE_REQUEST_SENT,
      //д��������
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
�ж�һϵ�еķ���cache�¼��Ƿ����WRITE_ALLOCATE_SENT��
�����¼����Ͱ�����
    enum cache_event_type {
      //д������
      WRITE_BACK_REQUEST_SENT,
      //������
      READ_REQUEST_SENT,
      //д����
      WRITE_REQUEST_SENT,
      //д��������
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
// ����Ƿ��Ѵ��ڶԽϵ��ڴ漶��Ĺ�����������ʵ������MSHR�����Ƿ��Ѿ���block_addr�����󱻺ϲ���MSHR��
// typedef new_addr_type unsigned long long.
bool mshr_table::probe(new_addr_type block_addr) const {
  //MSHR���е�����Ϊstd::unordered_map����<new_addr_type, mshr_entry>������map����ַblock_addr
  //ȥ�������Ƿ��ڱ��У���� a = m_data.end()����˵������û�� block_addr����֮������ڸ���Ŀ�����
  //�����ڸ���Ŀ���򷵻�false��������ڸ���Ŀ������true��������ڶԽϵ��ڴ漶��Ĺ�������
  table::const_iterator a = m_data.find(block_addr);
  return a != m_data.end();
}

/// Checks if there is space for tracking a new memory access
// ����Ƿ��пռ䴦���µ��ڴ���ʡ����mshr_addr��MSHR���Ѵ�����Ŀ��m_mshrs.full����Ƿ����Ŀ�ĺϲ���
// ���Ѵﵽ���ϲ��������mshr_addr��MSHR�в�������Ŀ�������Ƿ��п��е�MSHR��Ŀ���Խ�mshr_addr�����MSHR��
bool mshr_table::full(new_addr_type block_addr) const {
  //���Ȳ����Ƿ�MSHR������ block_addr ��ַ����Ŀ��
  table::const_iterator i = m_data.find(block_addr);
  if (i != m_data.end())
    //������ڸ���Ŀ�����Ƿ��пռ�ϲ�������Ŀ��
    return i->second.m_list.size() >= m_max_merged;
  else
    //��������ڸ���Ŀ�����Ƿ�������������Ŀ��ӡ�
    return m_data.size() >= m_num_entries;
}

/// Add or merge this access
// ��ӻ�ϲ��˷��ʡ�����������MSHR������ block_addr ��ַ����Ŀ��ֱ�������Ŀ����ӡ�
void mshr_table::add(new_addr_type block_addr, mem_fetch *mf) {
  //�� block_addr ��ַ���뵽��Ӧ��Ŀ�ڡ�
  m_data[block_addr].m_list.push_back(mf);
  assert(m_data.size() <= m_num_entries);
  assert(m_data[block_addr].m_list.size() <= m_max_merged);
  // indicate that this MSHR entry contains an atomic operation
  //ָʾ��MSHR��Ŀ����ԭ�Ӳ�����
  if (mf->isatomic()) {
    //mem_fetch������һ��ģ���ڴ������ͨ�Žṹ��������һ���ڴ��������Ϊ����� mf ������ڴ������
    //ԭ�Ӳ���������ԭ�Ӳ�����־λ��
    m_data[block_addr].m_has_atomic = true;
  }
}

/// check is_read_after_write_pending
// ����Ƿ���ڹ����д�����������������MSHR������ block_addr ��ַ����Ŀ��
bool mshr_table::is_read_after_write_pending(new_addr_type block_addr) {
  std::list<mem_fetch *> my_list = m_data[block_addr].m_list;
  bool write_found = false;
  //��block_addr��Ŀ�У��������е�mem_fetch��Ϊ��
  for (std::list<mem_fetch *>::iterator it = my_list.begin();
       it != my_list.end(); ++it) {
    //���(*it)->is_write()Ϊ�棬����it��д��Ϊ��д���������ڹ���״̬��
    if ((*it)->is_write())  // Pending Write Request
      write_found = true;
    //�����ǰ(*it)����д��Ϊ���Ƕ���Ϊ������write_found��Ϊtrue����֮ǰ��һ���� block_addr ��ַ
    //��д��Ϊ����˴��ڶ� block_addr ��ַ��д�����Ϊ������
    else if (write_found)  // Pending Read Request and we found previous Write
      return true;
  }

  return false;
}

/// Accept a new cache fill response: mark entry ready for processing
// �����µĻ��������Ӧ�������Ŀ�Ա���������������MSHR������ block_addr ��ַ����Ŀ��
void mshr_table::mark_ready(new_addr_type block_addr, bool &has_atomic) {
  //busy����ʼ�շ���false���˾���Ч��
  assert(!busy());
  //���� block_addr ��ַ��Ӧ����Ŀ��
  //m_data��MSHR����Ŀ����<new_addr_type, mshr_entry>��map��
  table::iterator a = m_data.find(block_addr);
  assert(a != m_data.end());
  //��ǰmark_ready�����������ݰ�fill��cache��ʱ���ã���һ�����ݰ����˺󣬸������ݰ����ص�������
  //����������˽����ַ���뵽m_current_response�С�
  //���� block_addr ��ַ�ķ��ʺϲ��������ڴ�����б��С�m_current_response�Ǿ����ڴ���ʵ��б�
  //m_current_response���洢�˾����ڴ���ʵĵ�ַ��
  m_current_response.push_back(block_addr);
  //����ԭ�ӱ�־λ��
  has_atomic = a->second.m_has_atomic;
  assert(m_current_response.size() <= m_data.size());
}

/// Returns next ready access
// ����һ���Ѿ�����ľ������ʡ�ͨ����� access_ready() һ��ʹ�ã�access_ready �������
// �Ƿ���ھ������ʣ�next_access() �������ؾ������ʣ�
//   bool access_ready() const { return !m_current_response.empty(); }
mem_fetch *mshr_table::next_access() {
  // access_ready() �Ĺ�����������ھ������ʣ��򷵻� true�������Ǽٶ����ھ����ڴ��
  // �ʡ�
  assert(access_ready());
  // ���ؾ����ڴ�����б���׸���Ŀ����Ŀ��ַ��m_current_response �Ǿ����ڴ���ʵ���
  // ��m_current_response ���洢�˾����ڴ���ʵĵ�ַ��
  // ���ݰ� fill �� cache ��ʱ����һ�����ݰ����˺󣬸������ݰ����ص������Ѿ���������
  // �˽��� block ��ַ���뵽 m_current_response �д������ block �Ѿ������ݾ����ˡ�
  // Ҳ����˵���� m_current_response �д洢���������ݵ� block ��ַ��
  new_addr_type block_addr = m_current_response.front();
  /* m_list �� mshr_entry ��һ����Ա��mshr_entry ��һ���ڴ����������б�m_data 
     �� MSHR ����Ŀ���� <new_addr_type, mshr_entry> �� map��m_data[block_addr]
     ��˾���һ�� mshr_entry��m_data[block_addr].m_list �Ƕ�Ӧ block_addr �����
     ַ���ڴ����������б�m_data[block_addr].m_list.front() ������б���׸���
     ��
        struct mshr_entry {
          // ������Ŀ�п��Ժϲ����ڴ��������
          std::list<mem_fetch *> m_list;
          // ������Ŀ�Ƿ���ԭ�Ӳ�����
          bool m_has_atomic;
          mshr_entry() : m_has_atomic(false) {}
        };
  */
  assert(!m_data[block_addr].m_list.empty());
  // ���� block_addr �ĺϲ����ڴ������Ϊ���׸�����mem_fetch=m_list.front()��
  mem_fetch *result = m_data[block_addr].m_list.front();
  // ���ϲ����ڴ������Ϊ���׸�������б��� pop ��ȥ��
  m_data[block_addr].m_list.pop_front();
  // ������Ҫע����ǣ�m_data[block_addr].m_list �洢�� block_addr ��ַ���ڴ����
  // ���ص������б�������б�Ϊ��ʱ��˵����� block_addr ��ַ�������ڴ������Ϊ��
  // ����������˿��Խ� block_addr �� m_current_response �� pop ��ȥ�ˡ�
  if (m_data[block_addr].m_list.empty()) {
    // �ڽ��ϲ����ڴ������Ϊ���׸�������б��� pop ��ȥ���б������ռ�����ĿʧЧ��
    // ��Ҫ��������Ŀ��
    // release entry
    m_data.erase(block_addr);
    // ��һ���������ʵõ��󣬾����ڴ�����б��аѸôξ������ʵĵ�ַ pop ��ȥ��
    // m_current_response ���洢�˾����ڴ���ʵĵ�ַ��
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
cache��ģ�������ݶ˿ں����˿ڣ�m_cache->access()ʹ��data_port��m_cache->fill()ʹ��
fill_port��
����mem_fetch�������ɵĽ�����¼�ʹ�����ݶ˿ڡ���Ϊ���ݶ˿ڵĿ�����ޣ���˵���һ��cache
����ʱ��һ�����ݰ��Ĵ�СҪ�ָ�ɼ��Ĳ���������˿ڡ�ֻ��
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
����mem_fetch����ʹ�����˿ڡ�
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
cache��ǰ�ƽ�һ�ġ�
*/
void baseline_cache::cycle() {
  //���MISS��������в�Ϊ�գ��򽫶��׵������͵���һ���ڴ档
  if (!m_miss_queue.empty()) {
    mem_fetch *mf = m_miss_queue.front();
    if (!m_memport->full(mf->size(), mf->get_is_write())) {
      m_miss_queue.pop_front();
      //mem_fetch_interface�Ƕ�mem�ô�Ľӿڡ�
      //mem_fetch_interface��cache��mem�ô�Ľӿڣ�cache��miss����������һ���洢����ͨ��
      //����ӿ������ͣ���m_miss_queue�е����ݰ���Ҫѹ��m_memportʵ�ַ�������һ���洢��
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
���ص�����ͨ��baseline_cache::fill����cache��tag_array�С�
m_config.m_mshr_type�����MSHR���ͣ�
  TEX_FIFO,         // Tex cache
  ASSOC,            // normal cache
  SECTOR_TEX_FIFO,  // Tex cache sends requests to high-level sector cache
  SECTOR_ASSOC      // normal cache sends requests to high-level sector cache
Cache���ò�����
  <sector?>:<nsets>:<bsize>:<assoc>,
  <rep>:<wr>:<alloc>:<wr_alloc>:<set_index_fn>,
  <mshr>:<N>:<merge>,<mq>:**<fifo_entry>
GV100����ʾ����
  -gpgpu_cache:dl1  S:4:128:64,  L:T:m:L:L, A:512:8, 16:0,32
  -gpgpu_cache:dl2  S:32:128:24, L:B:m:L:P, A:192:4, 32:0,32
  -gpgpu_cache:il1  N:64:128:16, L:R:f:N:L, S:2:48,  4
��GV100��MSHR type�ϣ�L1DΪASSOC��L2DΪASSOC��L1IΪSECTOR_ASSOC��
  TEX_FIFO,         // Tex cache
  ASSOC,            // normal cache
  SECTOR_TEX_FIFO,  // Tex cache sends requests to high-level sector cache
  SECTOR_ASSOC      // normal cache sends requests to high-level sector cache
*/
void baseline_cache::fill(mem_fetch *mf, unsigned time) {
  //�����sector cache����Ҫ����ǰmf�Ƿ���һ����mf�ָ�󷵻ص����һ��Сmf���������line cache
  //�Ļ����Ͳ���Ҫ�����������Ϊ���ص�mfһ����һ����cache block���ݡ�
  if (m_config.m_mshr_type == SECTOR_ASSOC) {
    //mf->get_original_mf()����L2 cache�н�mf����Ϊsector requestsʱ���ã����req size > L2
    //sector size������ָ��ָ��ԭʼmf����Ϊʵ����ֻ��������ģ������а�mf����Ϊ����sector������
    //mf����ʱ����Ҫ����Щsector�ϲ�Ϊһ��mf������Ϊ�˼�㣬�͸�ÿ���ָ��mf����һ��ԭʼmf��ָ�롣
    assert(mf->get_original_mf());
    //extra_mf_fields_lookup�Ķ��壺
    //  typedef std::map<mem_fetch *, extra_mf_fields> extra_mf_fields_lookup;
    //��L2 cacheΪ����
    //��cache������������mfʱ�����δ���У�����MSHR��Ҳδ���У�û��mf��Ŀ����������뵽MSHR�У�
    //ͬʱ������m_extra_mf_fields[mf]����ζ�����mf��m_extra_mf_fields�д��ڣ���mf�ȴ���DRAM
    //�����ݻص�L2������䣺
    //m_extra_mf_fields[mf] = extra_mf_fields(
    //      mshr_addr, mf->get_addr(), cache_index, mf->get_data_size(), m_config);
    //L1 Data cache���ǵȴ�SM�ڵ�m_response_fifo�е�������䡣
    extra_mf_fields_lookup::iterator e =
        m_extra_mf_fields.find(mf->get_original_mf());
    assert(e != m_extra_mf_fields.end());
    //���ҵ��Ļ�������m_extra_mf_fields[mf].pending_read��һ��
    e->second.pending_read--;

    //���m_extra_mf_fields[mf].pending_read����0��˵�����ڵȴ�������mf��ͬ��������ݣ�ֱ�Ӷ�����
    //��Ϊ�����mf��ʵֻ��һ�����ָ�Ĵ�mf��һ���֣��ܹ�ֱ�Ӷ�������Ϊ����ͬ����һ���������mf��
    //ָ�����ǹ�ͬ��ԭʼmf��mf->get_original_mf()��
    if (e->second.pending_read > 0) {
      // wait for the other requests to come back
      delete mf;
      return;
    } else {
      //���m_extra_mf_fields[mf].pending_read����0��˵�����mf�Ѿ������һ���ָ��mf�ˣ�����Ҫ
      //�ٵȴ�������mf��ͬ��������ݣ�������䵽cache�С�
      mem_fetch *temp = mf;
      mf = mf->get_original_mf();
      delete temp;
    }
  }
  //��������˼�鵱ǰmf�Ƿ���һ����mf�ָ�󷵻ص����һ��Сmf�����ǵĻ����˺�����ֱ���˳��ˣ�����
  //�������ǽ�mf��䵽cache�С�

  extra_mf_fields_lookup::iterator e = m_extra_mf_fields.find(mf);
  assert(e != m_extra_mf_fields.end());
  assert(e->second.m_valid);
  mf->set_data_size(e->second.m_data_size);
  mf->set_addr(e->second.m_addr);
  //��V100�У�L1 cache��L2 cache��Ϊallocate on miss��
  //allocate-on-miss����дΪon-miss����allocate-on-fill����дΪon-fill��������cache�з�����ԣ�
  //(1) allocate-on-miss��������δ��ɵ�cache missʱ����ҪΪδ��ɵ�miss����cache line slot��
  //    MSHR��miss������Ŀ��
  //(2) allocateon-on-fill��������δ��ɵ�cache missʱ����ҪΪδ��ɵ�miss����MSHR��miss������
  //    Ŀ�������������ݴӽϵ��ڴ漶�𷵻�ʱ����ѡ���ܺ���cache line slot�滻��
  //�������ֲ����У�����κ�������Դ�����ã���ᷢ��reservation failure�������ڴ���ˮ��ͣ�١�����
  //��MSHR��������ֱ�����ݴӽϵ�һ���ڴ���ȡ�أ���һ��miss����ת����L2 cache�����ͷ�miss������Ŀ��
  //allocateon-on-fill������allocate-on-miss�и��õ����ܣ���Ϊ���������ܺ���cache������ʱ�䣬��
  //��Ϊδ��ɵ�miss�������ٵ���Դ���Ӷ����ܸ����cache hit�͸��ٵ�reservation failure������all-
  //ocateon-on-fill��Ҫ�����buffer�������߼��������ݰ�˳����䵽cache������˳��ִ��ģ�ͺ�write-
  //evict����ʹ��GPU L1 D-cache��allocateon-on-fill���Ѻã���Ϊ���ܺ���cache�ڱ�evictʱ��û����
  //���ݱ�д��L2��
  if (m_config.m_alloc_policy == ON_MISS)
    m_tag_array->fill(e->second.m_cache_index, time, mf);
  else if (m_config.m_alloc_policy == ON_FILL) {
    m_tag_array->fill(e->second.m_block_addr, time, mf, mf->is_write());
  } else
    abort();
  bool has_atomic = false;
  /*
  Accept a new cache fill response: mark entry ready for processing. �����µĻ��������Ӧ����
  ����Ŀ�Ա�����
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
����Ƿ�mf���ڵȴ����͵Ĵ洢�����䡣
*/
bool baseline_cache::waiting_for_fill(mem_fetch *mf) {
  //extra_mf_fields_lookup�Ķ��壺
  //  typedef std::map<mem_fetch *, extra_mf_fields> extra_mf_fields_lookup;
  //��cache������������mfʱ�����δ���У�����MSHR��Ҳδ���У�û��mf��Ŀ����������뵽MSHR�У�
  //ͬʱ������m_extra_mf_fields[mf]����ζ�����mf��m_extra_mf_fields�д��ڣ���mf�ȴ���DRAM
  //�����ݻص�L2������䣺
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
READ MISS�����������MSHR�Ƿ����л���MSHR�Ƿ���ã������ж��Ƿ���Ҫ����һ���洢���Ͷ�����
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
READ MISS ����������� MSHR �Ƿ����л��� MSHR �Ƿ���ã������ж��Ƿ���Ҫ����һ
���洢���Ͷ�����
*/
void baseline_cache::send_read_request(new_addr_type addr,
                                       new_addr_type block_addr,
                                       unsigned cache_index, mem_fetch *mf,
                                       unsigned time, bool &do_miss, bool &wb,
                                       evicted_block_info &evicted,
                                       std::list<cache_event> &events,
                                       bool read_only, bool wa) {
  // 1. ����� Sector Cache��
  //  mshr_addr �������� mshr �ĵ�ַ���õ�ַ��Ϊ��ַ addr �� tag λ + set index 
  //  λ + sector offset λ������ single sector byte offset λ ���������λ��
  //  |<----------mshr_addr----------->|
  //                     sector offset  off in-sector
  //                     |-------------|-----------|
  //                      \                       /
  //                       \                     /
  //  |-------|-------------|-------------------|
  //             set_index     offset in-line
  //  |<----tag----> 0 0 0 0|
  // 2. ����� Line Cache��
  //  mshr_addr �������� mshr �ĵ�ַ���õ�ַ��Ϊ��ַ addr �� tag λ + set index 
  //  λ������ single line byte off-set λ ���������λ��
  //  |<----mshr_addr--->|
  //                              line offset
  //                     |-------------------------|
  //                      \                       /
  //                       \                     /
  //  |-------|-------------|-------------------|
  //             set_index     offset in-line
  //  |<----tag----> 0 0 0 0|
  //
  // mshr_addr ���壺
  //   new_addr_type mshr_addr(new_addr_type addr) const {
  //     return addr & ~(new_addr_type)(m_atom_sz - 1);
  //   }
  // m_atom_sz = (m_cache_type == SECTOR) ? SECTOR_SIZE : m_line_sz; 
  // ���� SECTOR_SIZE = const (32 bytes per sector).
  new_addr_type mshr_addr = m_config.mshr_addr(mf->get_addr());
  // ����ʵ������ MSHR �����Ƿ��Ѿ��� mshr_addr �����󱻺ϲ��� MSHR������Ѿ�����
  // ���� mshr_hit = true����Ҫע�⣬MSHR �е���Ŀ���� mshr_addr Ϊ�����ģ�������
  // ͬһ�� line�����ڷ� Sector Cache����������ͬһ�� sector������ Sector Cache��
  // �����񱻺ϲ�����Ϊ���� cache ���������С��λ�ֱ���һ�� line ����һ�� sector��
  // ���û��Ҫ������ô������ֻ��Ҫ����һ�μ��ɡ�
  bool mshr_hit = m_mshrs.probe(mshr_addr);
  // ��� mshr_addr �� MSHR ���Ѵ�����Ŀ��m_mshrs.full ����Ƿ����Ŀ�ĺϲ�������
  // �ﵽ���ϲ�������� mshr_addr �� MSHR �в�������Ŀ�������Ƿ��п��е� MSHR 
  // ��Ŀ���Խ� mshr_addr ����� MSHR��
  bool mshr_avail = !m_mshrs.full(mshr_addr);
  if (mshr_hit && mshr_avail) {
    // ��� MSHR ���У��� mshr_addr ��Ӧ��Ŀ�ĺϲ�����û�дﵽ���ϲ�����������
    // ���� mf ���뵽 MSHR �С�
    if (read_only)
      m_tag_array->access(block_addr, time, cache_index, mf);
    else
      // ���� tag_array ��״̬���������� LRU ״̬����������� block �� sector �ȡ�
      m_tag_array->access(block_addr, time, cache_index, wb, evicted, mf);

    // �� mshr_addr ��ַ���������� mf ���뵽 MSHR �С���Ϊ���� MSHR��˵��ǰ���Ѿ�
    // �жԸ����ݵ������͵���һ�������ˣ��������ֻ��Ҫ�ȴ�ǰ������󷵻ؼ��ɡ�
    m_mshrs.add(mshr_addr, mf);
    m_stats.inc_stats(mf->get_access_type(), MSHR_HIT, mf->get_streamID());
    // ��ʶ�Ƿ��������� MSHR ���� ���ŵ� m_miss_queue ������һ�����ڷ��͵���һ
    // ���洢��
    do_miss = true;

  } else if (!mshr_hit && mshr_avail &&
             (m_miss_queue.size() < m_config.m_miss_queue_size)) {
    // ��� MSHR δ���У����п��е� MSHR ��Ŀ���Խ� mshr_addr ����� MSHR������
    // ������ mf ���뵽 MSHR �С�
    // ���� L1 cache �� L2 cache��read_only Ϊ false������ read_only_cache ��˵��
    // read_only Ϊtrue��
    if (read_only)
      m_tag_array->access(block_addr, time, cache_index, mf);
    else
      // ���� tag_array ��״̬���������� LRU ״̬����������� block �� sector �ȡ�
      m_tag_array->access(block_addr, time, cache_index, wb, evicted, mf);

    // �� mshr_addr ��ַ���������� mf ���뵽 MSHR �С���Ϊû������ MSHR����˻���
    // Ҫ�������ݵ������͵���һ�����档
    m_mshrs.add(mshr_addr, mf);
    // if (m_config.is_streaming() && m_config.m_cache_type == SECTOR) {
    //   m_tag_array->add_pending_line(mf);
    // }
    // ���� m_extra_mf_fields[mf]����ζ����� mf �� m_extra_mf_fields �д��ڣ��� 
    // mf �ȴ�����һ���洢�����ݻص���ǰ������䡣
    m_extra_mf_fields[mf] = extra_mf_fields(
        mshr_addr, mf->get_addr(), cache_index, mf->get_data_size(), m_config);
    mf->set_data_size(m_config.get_atom_sz());
    mf->set_addr(mshr_addr);
    // mf Ϊ miss �����󣬼��� miss_queue��MISS ������С�
    // �� baseline_cache::cycle() �У��Ὣ m_miss_queue ���׵����ݰ� mf ���ݸ���
    // һ��洢����Ϊû������ MSHR��˵��ǰ��û�жԸ����ݵ������͵���һ�����棬
    // ���������Ҫ�Ѹ������͸���һ���洢��
    m_miss_queue.push_back(mf);
    mf->set_status(m_miss_queue_status, time);
    // �� V100 �����У�wa �� L1/L2/read_only cache ��Ϊ false��
    if (!wa) events.push_back(cache_event(READ_REQUEST_SENT));
    // ��ʶ�Ƿ��������� MSHR ���� ���ŵ� m_miss_queue ������һ�����ڷ��͵���һ
    // ���洢��
    do_miss = true;
  } else if (mshr_hit && !mshr_avail)
    // ��� MSHR ���У��� mshr_addr ��Ӧ��Ŀ�ĺϲ������ﵽ�����ϲ�����
    m_stats.inc_fail_stats(mf->get_access_type(), MSHR_MERGE_ENRTY_FAIL,
                           mf->get_streamID());
  else if (!mshr_hit && !mshr_avail)
    // ��� MSHR δ���У��� mshr_addr û�п��е� MSHR ��Ŀ�ɽ� mshr_addr ���롣
    m_stats.inc_fail_stats(mf->get_access_type(), MSHR_ENRTY_FAIL,
                           mf->get_streamID());
  else
    assert(0);
}

/// Sends write request to lower level memory (write or writeback)
/*
������д����һͬ��������һ���洢��������Ҫ�����ǽ�д�������� WRITE_REQUEST_SENT �� 
WRITE_BACK_REQUEST_SENT ���� events�������������� mf ���� m_miss_queue�У��ȴ���
һʱ������ baseline_cache::cycle() �����׵��������� mf ���͸���һ���洢��
*/
void data_cache::send_write_request(mem_fetch *mf, cache_event request,
                                    unsigned time,
                                    std::list<cache_event> &events) {
  events.push_back(request);
  // �� baseline_cache::cycle() �У��Ὣ m_miss_queue ���׵����ݰ� mf ���ݸ���
  // һ���洢��
  m_miss_queue.push_back(mf);
  mf->set_status(m_miss_queue_status, time);
}

/*
����һ��cache block��״̬Ϊ�ɶ���������е�byte maskλȫ������Ϊdirty�ˣ��򽫸�sector��
����Ϊ�ɶ�����Ϊ��ǰ��sector�Ѿ���ȫ������Ϊ����ֵ�ˣ��ǿɶ��ġ�������������е���������mf
�����з��ʵ�sector���б��������mf�����ʵ����е�byte maskλȫ������Ϊdirty�ˣ��򽫸�cache
block����Ϊ�ɶ���
*/
void data_cache::update_m_readable(mem_fetch *mf, unsigned cache_index) {
  //���ﴫ��Ĳ�����cache block��index��
  // For example, 4 sets, 6 ways:
  // |  0  |  1  |  2  |  3  |  4  |  5  |  // set_index 0
  // |  6  |  7  |  8  |  9  |  10 |  11 |  // set_index 1
  // |  12 |  13 |  14 |  15 |  16 |  17 |  // set_index 2
  // |  18 |  19 |  20 |  21 |  22 |  23 |  // set_index 3
  //                |--------> index => cache_block_t *line
  cache_block_t *block = m_tag_array->get_block(cache_index);
  //�Ե�ǰcache block��4��sector���б�����
  for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; i++) {
    //��i��sector����������mf���ʡ�
    if (mf->get_access_sector_mask().test(i)) {
      //all_set��ָ���е�byte maskλ�������ó���dirty�ˡ�
      bool all_set = true;
      //����k�������ڵ�i��sector��byte�ı�š�
      for (unsigned k = i * SECTOR_SIZE; k < (i + 1) * SECTOR_SIZE; k++) {
        // If any bit in the byte mask (within the sector) is not set,
        // the sector is unreadble
        //�����i��sector��������һ��byte��dirty maskλû�б����ã���all_set����false��
        if (!block->get_dirty_byte_mask().test(k)) {
          all_set = false;
          break;
        }
      }
      //������е�byte maskλȫ������Ϊdirty�ˣ��򽫸�sector������Ϊ�ɶ�����Ϊ��ǰ��
      //sector�Ѿ���ȫ������Ϊ����ֵ�ˣ��ǿɶ��ġ�
      if (all_set) block->set_m_readable(true, mf->get_access_sector_mask());
    }
  }
}

/****** Write-hit functions (Set by config file) ******/

/// Write-back hit: Mark block as modified
/*
�� Write Hit ʱ��ȡ write-back ���ԣ�����Ҫ�����ݵ�д�� cache������Ҫֱ�ӽ�����д��
��һ���洢���ȵ������� fill ����ʱ���ٽ������������д����һ���洢��
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
  // write-back ���Բ���Ҫֱ�ӽ�����д����һ���洢����˲���Ҫ����miss_queue_full()
  // �Լ� send_write_request() ����������д��������һ���洢��
  new_addr_type block_addr = m_config.block_addr(addr);
  // ���� tag_array ��״̬���������� LRU ״̬����������� block �� sector �ȡ�
  m_tag_array->access(block_addr, time, cache_index, mf);  // update LRU state
  cache_block_t *block = m_tag_array->get_block(cache_index);
  // ��� block ���� modified line�������� dirty ��������Ϊ������ʱ�� block ����
  // modified line��˵����� block �� clean line��������Ҫд�����ݣ������Ҫ�����
  // block ����Ϊ modified line�������Ļ���dirty ��������Ҫ���ӡ������ block �Ѿ�
  // �� modified line������Ҫ���� dirty ��������Ϊ��� block ���ϴα�� dirty ��
  // ʱ��dirty �����Ѿ����ӹ��ˡ�
  if (!block->is_modified_line()) {
    m_tag_array->inc_dirty();
  }
  // ���� block ��״̬Ϊ modified������ block ����Ϊ MODIFIED�������Ļ����´�����
  // �������������� block ��ʱ�򣬾Ϳ���ֱ�Ӵ� cache �ж�ȡ���ݣ�������Ҫ�ٴη���
  // ��һ���洢����Ȼ�������´�������� block ����������ʱ��block �� tag �������
  // tag ��һ�£���������� block ��״̬�Ѿ�������Ϊ modified�������Ҫ���� block 
  // �����������д�ص���һ���洢��
  block->set_status(MODIFIED, mf->get_access_sector_mask());
  block->set_byte_mask(mf);
  // ����һ�� cache block ��״̬Ϊ�ɶ�������Ҫע����ǣ�����Ŀɶ���ָ�� sector ��
  // �������������� block �ɶ������һ�� sector �ڵ����е� byte mask λȫ������Ϊ 
  // dirty �ˣ��򽫸�sector ������Ϊ�ɶ�����Ϊ��ǰ�� sector �Ѿ���ȫ������Ϊ����ֵ
  // �ˣ��ǿɶ��ġ�������������е��������� mf �����з��ʵ� sector ���б����������
  // sector ���� mf ���ʵģ����� mf->get_access_sector_mask() ȷ����
  update_m_readable(mf, cache_index);

  return HIT;
}

/// Write-through hit: Directly send request to lower level memory
/*
�� Write Hit ʱ��ȡ write-through ���ԵĻ�������Ҫ�����ݲ�����д�� cache������Ҫֱ
�ӽ�����д����һ���洢��
��һ��cache�������ݷ��ʵ�ʱ�򣬵���data_cache::access()������
- ����cahe�����m_tag_array->probe()�������ж϶�cache�ķ��ʣ���ַΪaddr��sector mask
  Ϊmask����HIT/HIT_RESERVED/SECTOR_MISS/MISS/RESERVATION_FAIL��״̬��
- Ȼ�����process_tag_probe()����������cache�������Լ�����m_tag_array->probe()������
  �ص�cache����״̬��ִ����Ӧ�Ĳ�����
  - process_tag_probe()�����У����������Ķ�д״̬��probe()�������ص�cache����״̬��
    ִ��m_wr_hit/m_wr_miss/m_rd_hit/m_rd_miss���������ǻ����m_tag_array->access()
    ������ʵ��LRU״̬�ĸ��¡�
*/
cache_request_status data_cache::wr_hit_wt(new_addr_type addr,
                                           unsigned cache_index, mem_fetch *mf,
                                           unsigned time,
                                           std::list<cache_event> &events,
                                           enum cache_request_status status) {
  // miss_queue_full ����Ƿ�һ�� miss �����ܹ��ڵ�ǰʱ�������ڱ�������һ�������
  // ��С�� m_miss_queue �Ų���ʱ���ڵ�ǰ�����޷��������� RESERVATION_FAIL��
  if (miss_queue_full(0)) {
    m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL,
                           mf->get_streamID());
    // ��� miss_queue ���ˣ������� write-through ����Ҫ������Ӧ��ֱ��д����һ����
    // ����������ﷵ�� RESERVATION_FAIL����ʾ��ǰʱ���������޷����������
    return RESERVATION_FAIL;  // cannot handle request this cycle
  }
  // m_config.block_addr(addr): 
  //     return addr & ~(new_addr_type)(m_line_sz - 1);
  // |-------|-------------|--------------|
  //            set_index   offset in-line
  // |<--------tag--------> 0 0 0 0 0 0 0 |
  new_addr_type block_addr = m_config.block_addr(addr);
  // ���� tag_array ��״̬���������� LRU ״̬����������� block �� sector �ȡ�
  m_tag_array->access(block_addr, time, cache_index, mf);  // update LRU state
  cache_block_t *block = m_tag_array->get_block(cache_index);
  // ��� block ���� modified line�������� dirty ��������Ϊ������ʱ�� block ����
  // modified line��˵����� block �� clean line��������Ҫд�����ݣ������Ҫ�����
  // block ����Ϊ modified line�������Ļ���dirty ��������Ҫ���ӡ������ block �Ѿ�
  // �� modified line������Ҫ���� dirty ��������Ϊ��� block ���ϴα�� dirty ��
  // ʱ��dirty �����Ѿ����ӹ��ˡ�
  if (!block->is_modified_line()) {
    m_tag_array->inc_dirty();
  }
  // ���� block ��״̬Ϊ modified������ block ����Ϊ MODIFIED�������Ļ����´�����
  // �������������� block ��ʱ�򣬾Ϳ���ֱ�Ӵ� cache �ж�ȡ���ݣ�������Ҫ�ٴη���
  // ��һ���洢��
  block->set_status(MODIFIED, mf->get_access_sector_mask());
  block->set_byte_mask(mf);
  // ����һ�� cache block ��״̬Ϊ�ɶ�������Ҫע����ǣ�����Ŀɶ���ָ�� sector ��
  // �������������� block �ɶ������һ�� sector �ڵ����е� byte mask λȫ������Ϊ 
  // dirty �ˣ��򽫸�sector ������Ϊ�ɶ�����Ϊ��ǰ�� sector �Ѿ���ȫ������Ϊ����ֵ
  // �ˣ��ǿɶ��ġ�������������е��������� mf �����з��ʵ� sector ���б����������
  // sector ���� mf ���ʵģ����� mf->get_access_sector_mask() ȷ����
  update_m_readable(mf, cache_index);

  // generate a write-through
  // write-through ������Ҫ������д�� cache ��ͬʱҲֱ��д����һ���洢��������Ҫ��
  // ���ǽ�д�������� WRITE_REQUEST_SENT ���� events����������������뵱ǰ cache  
  // �� m_miss_queue �У��ȴ�baseline_cache::cycle() �� m_miss_queue ���׵���
  // ��д���� mf ���͸���һ���洢��
  send_write_request(mf, cache_event(WRITE_REQUEST_SENT), time, events);

  return HIT;
}

/// Write-evict hit: Send request to lower level memory and invalidate
/// corresponding block
/*
д������У�����һ���洢����д������ֱ�������Ӧ�� cache block ����������Ч��
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
  // write-evict ������Ҫ�� cache block ֱ�������Ϊ��Ч��ͬʱҲֱ��д����һ����
  // ����������Ҫ�����ǽ�д�������� WRITE_REQUEST_SENT ���� events��������������  
  // ���� m_miss_queue �У��ȴ�baseline_cache::cycle() �� m_miss_queue ���׵�
  // ����д���� mf ���͸���һ���洢��
  send_write_request(mf, cache_event(WRITE_REQUEST_SENT), time, events);

  // Invalidate block
  // д������� cache block ֱ�������Ϊ��Ч��
  block->set_status(INVALID, mf->get_access_sector_mask());

  return HIT;
}

/// Global write-evict, local write-back: Useful for private caches
/*
ȫ�ַô����д��������طô����д�ء����ֲ���������˽�л��档������ԱȽϼ򵥣���ֻ
��Ҫ�жϵ�ǰ������������ȫ�ַô滹�Ǳ��طô棬Ȼ��ֱ����д�����д�ز��Լ��ɡ�
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
GPGPU-Sim 3.x�汾�е�naiveд������ԡ�wr_miss_wa_naive ������д MISS ʱ����Ҫ�Ƚ� 
mf ���ݰ�ֱ��д����һ���洢�������Ὣ WRITE_REQUEST_SENT ���� events�������������� 
mf ���� m_miss_queue �У��ȴ���һ������ baseline_cache::cycle() �� m_miss_queue 
���׵����ݰ� mf ���͸���һ���洢����Σ�wr_miss_wa_naive ���Ի��Ὣ addr ��ַ������
������ǰ cache �У���ʱ���ִ�� send_read_request ������������ send_read_request 
�����У����п��������������Ҫ evict һ�� block �ſ��Խ��µ����ݶ��뵽 cache �У���
ʱ����� evicted block �� modified line������Ҫ����� evicted block д�ص���һ��
�洢����ʱ������ do_miss �� wb ��ִֵ�� send_write_request ������
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
  // 1. ����� Sector Cache��
  //  mshr_addr �������� mshr �ĵ�ַ���õ�ַ��Ϊ��ַ addr �� tag λ + set index 
  //  λ + sector offset λ������ single sector byte offset λ ���������λ��
  //  |<----------mshr_addr----------->|
  //                     sector offset  off in-sector
  //                     |-------------|-----------|
  //                      \                       /
  //                       \                     /
  //  |-------|-------------|-------------------|
  //             set_index     offset in-line
  //  |<----tag----> 0 0 0 0|
  // 2. ����� Line Cache��
  //  mshr_addr �������� mshr �ĵ�ַ���õ�ַ��Ϊ��ַ addr �� tag λ + set index 
  //  λ������ single line byte off-set λ ���������λ��
  //  |<----mshr_addr--->|
  //                              line offset
  //                     |-------------------------|
  //                      \                       /
  //                       \                     /
  //  |-------|-------------|-------------------|
  //             set_index     offset in-line
  //  |<----tag----> 0 0 0 0|
  //
  // mshr_addr ���壺
  //   new_addr_type mshr_addr(new_addr_type addr) const {
  //     return addr & ~(new_addr_type)(m_atom_sz - 1);
  //   }
  // m_atom_sz = (m_cache_type == SECTOR) ? SECTOR_SIZE : m_line_sz; 
  // ���� SECTOR_SIZE = const (32 bytes per sector).
  new_addr_type mshr_addr = m_config.mshr_addr(mf->get_addr());

  // Write allocate, maximum 3 requests (write miss, read request, write back
  // request) Conservatively ensure the worst-case request can be handled this
  // cycle
  // MSHR �� m_data �� key �д洢�˸����ϲ��ĵ�ַ��probe() ������Ҫ����Ƿ����У�
  // ����Ҫ��� m_data.keys() ��������û�� mshr_addr��
  bool mshr_hit = m_mshrs.probe(mshr_addr);
  // ���Ȳ����Ƿ� MSHR ������ block_addr ��ַ����Ŀ��������ڸ���Ŀ������ MSHR����
  // ���Ƿ��пռ�ϲ�������Ŀ����������ڸ���Ŀ��δ���� MSHR�������Ƿ��������ռ���
  // ����� mshr_addr ��һ��Ŀ��
  bool mshr_avail = !m_mshrs.full(mshr_addr);
  // �� baseline_cache::cycle() �У��Ὣ m_miss_queue ���׵����ݰ� mf ���ݸ���һ
  // ���洢����˵����� miss ���������д�ص�������Ҫ������һ���洢ʱ����� miss ��
  // ����ŵ� m_miss_queue �С�
  //   bool miss_queue_full(unsigned num_miss) {
  //     return ((m_miss_queue.size() + num_miss) >= m_config.m_miss_queue_size);
  //   }
  
  // ��� m_miss_queue.size() �Ѿ����������������ݰ��Ļ����п����޷���ɺ���������
  // ��Ϊ���������Ҫִ������ send_write_request���� send_write_request ��ÿִ��
  // һ�Σ�����Ҫ�� m_miss_queue ���һ�����ݰ���
  // Write allocate, maximum 3 requests (write miss, read request, write back
  // request) Conservatively ensure the worst-case request can be handled this
  // cycle.
  if (miss_queue_full(2) ||
      // ��� miss_queue_full(2) ���� false���п���ռ�֧��ִ������ send_write_
      // request����ô����Ҫ�� MSHR �Ƿ��п��ÿռ䡣�����⴮�ж�������ʵ���Ի���� 
      // if (miss_queue_full(2) || !mshr_avail)��
      // ������ RESERVATION_FAIL ��������
      //   1. m_miss_queue �����Է������� WRITE_REQUEST_SENT ����
      //   2. MSHR ���ܺϲ�����δ���У�����û�п��ÿռ��������Ŀ����
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

    // ���� RESERVATION_FAIL ��������
    //   1. m_miss_queue �����Է������� WRITE_REQUEST_SENT ����
    //   2. MSHR ���ܺϲ�����δ���У�����û�п��ÿռ��������Ŀ����
    return RESERVATION_FAIL;
  }

  // send_write_request ִ�У�
  //   events.push_back(request);
  //   // �� baseline_cache::cycle() �У��Ὣ m_miss_queue ���׵����ݰ� mf ����
  //   // ����һ���洢��
  //   m_miss_queue.push_back(mf);
  //   mf->set_status(m_miss_queue_status, time);
  // wr_miss_wa_naive ������д MISS ʱ����Ҫ�Ƚ� mf ���ݰ�ֱ��д����һ���洢������
  // �Ὣ WRITE_REQUEST_SENT ���� events�������������� mf ���� m_miss_queue �У�
  // �ȴ���һ������ baseline_cache::cycle() �� m_miss_queue ���׵����ݰ� mf ����
  // ����һ���洢����Σ�wr_miss_wa_naive ���Ի��Ὣ addr ��ַ�����ݶ�����ǰ cache
  // �У���ʱ���ִ�� send_read_request ������������ send_read_request �����У���
  // �п��������������Ҫ evict һ�� block �ſ��Խ��µ����ݶ��뵽 cache �У���ʱ��
  // ��� evicted block �� modified line������Ҫ����� evicted block д�ص���һ��
  // �洢����ʱ������ do_miss �� wb ��ִֵ�� send_write_request ������
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

  // ��ʶ�Ƿ��������� MSHR ���� ���ŵ� m_miss_queue ������һ�����ڷ��͵���һ
  // ���洢��
  bool do_miss = false;
  // wb ������ʶ tag_array::access() �����У��������� send_read_request ����
  // ���� MISS������Ҫ���һ�� block��������� evicted block д�ص���һ���洢��
  // ������ block �Ѿ��� modified line���� wb Ϊ true����Ϊ�ڽ��������·���
  // ֮ǰ�����뽫����Ѿ� modified �� block д�ص���һ���洢���������� block 
  // �� clean line���� wb Ϊ false����Ϊ��� block ����Ҫд�ص���һ���洢����� 
  // evicted block ����Ϣ�������� evicted �С�
  bool wb = false;
  evicted_block_info evicted;

  // Send read request resulting from write miss
  send_read_request(addr, block_addr, cache_index, n_mf, time, do_miss, wb,
                    evicted, events, false, true);

  events.push_back(cache_event(WRITE_ALLOCATE_SENT));

  // do_miss ��ʶ�Ƿ��������� MSHR ���� ���ŵ� m_miss_queue ������һ������
  // ���͵���һ���洢��
  if (do_miss) {
    // If evicted block is modified and not a write-through
    // (already modified lower level)
    // wb ������ʶ tag_array::access() �����У��������� send_read_request ��
    // ������ MISS������Ҫ���һ�� block��������� evicted block д�ص���һ����
    // ���������� block �Ѿ��� modified line���� wb Ϊ true����Ϊ�ڽ�������
    // �·���֮ǰ�����뽫����Ѿ� modified �� block д�ص���һ���洢����������  
    // block �� clean line���� wb Ϊ false����Ϊ��� block ����Ҫд�ص���һ���� 
    // ������� evicted block ����Ϣ�������� evicted �С�
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
      // �� tag_array::access() ����������� evicted block д�ص���һ���洢��
      send_write_request(wb, cache_event(WRITE_BACK_REQUEST_SENT, evicted),
                         time, events);
    }
    // ��� do_miss Ϊ true����ʾ�������� MSHR ���� ���ŵ� m_miss_queue ����
    // ��һ�����ڷ��͵���һ���洢��������д MISS �����������й���ȫ����ɣ����ص�
    // �� write miss ���ԭʼд�����״̬��
    return MISS;
  }

  // ��� do_miss Ϊ false����ʾ����δ������ MSHR ���� δ���ŵ� m_miss_queue ��
  // ����һ�����ڷ��͵���һ���洢��������д MISS ������û�н��������ͳ�ȥ�����
  // ���� RESERVATION_FAIL��
  return RESERVATION_FAIL;
}

/*
write_allocated_fetch_on_write ���ԣ���д��ʱ��ȡ�����У���д�� sector �ĵ����ֽ�
ʱ��L2 ���ȡ���� sector ��Ȼ��д��Ĳ��ֺϲ����� sector �������� sector ����Ϊ��
�޸ġ�
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
  // 1. ����� Sector Cache��
  //  mshr_addr �������� mshr �ĵ�ַ���õ�ַ��Ϊ��ַ addr �� tag λ + set index 
  //  λ + sector offset λ������ single sector byte offset λ ���������λ��
  //  |<----------mshr_addr----------->|
  //                     sector offset  off in-sector
  //                     |-------------|-----------|
  //                      \                       /
  //                       \                     /
  //  |-------|-------------|-------------------|
  //             set_index     offset in-line
  //  |<----tag----> 0 0 0 0|
  // 2. ����� Line Cache��
  //  mshr_addr �������� mshr �ĵ�ַ���õ�ַ��Ϊ��ַ addr �� tag λ + set index 
  //  λ������ single line byte off-set λ ���������λ��
  //  |<----mshr_addr--->|
  //                              line offset
  //                     |-------------------------|
  //                      \                       /
  //                       \                     /
  //  |-------|-------------|-------------------|
  //             set_index     offset in-line
  //  |<----tag----> 0 0 0 0|
  //
  // mshr_addr ���壺
  //   new_addr_type mshr_addr(new_addr_type addr) const {
  //     return addr & ~(new_addr_type)(m_atom_sz - 1);
  //   }
  // m_atom_sz = (m_cache_type == SECTOR) ? SECTOR_SIZE : m_line_sz; 
  // ���� SECTOR_SIZE = const (32 bytes per sector).
  new_addr_type mshr_addr = m_config.mshr_addr(mf->get_addr());

  // �������д����ֽ����������� cache line/sector �Ĵ�С����ôֱ��д�� cache����
  // �� cache ����Ϊ MODIFIED��������Ҫ���Ͷ�������һ���洢��
  if (mf->get_access_byte_mask().count() == m_config.get_atom_sz()) {
    // if the request writes to the whole cache line/sector, then, write and set
    // cache line Modified. and no need to send read request to memory or
    // reserve mshr
    // ��� m_miss_queue.size() �Ѿ���������һ�����ݰ��Ļ����п����޷���ɺ���������
    // ��Ϊ���������Ҫִ��һ�� send_write_request���� send_write_request ��ÿִ��
    // һ�Σ�����Ҫ�� m_miss_queue ���һ�����ݰ���
    if (miss_queue_full(0)) {
      m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL,
                             mf->get_streamID());
      return RESERVATION_FAIL;  // cannot handle request this cycle
    }
    // wb ������ʶ tag_array::access() �����У��������� send_read_request ����
    // ���� MISS������Ҫ���һ�� block��������� evicted block д�ص���һ���洢��
    // ������ block �Ѿ��� modified line���� wb Ϊ true����Ϊ�ڽ��������·���
    // ֮ǰ�����뽫����Ѿ� modified �� block д�ص���һ���洢���������� block 
    // �� clean line���� wb Ϊ false����Ϊ��� block ����Ҫд�ص���һ���洢����� 
    // evicted block ����Ϣ�������� evicted �С�
    bool wb = false;
    evicted_block_info evicted;
    // ���� tag_array ��״̬���������� LRU ״̬����������� block �� sector �ȡ�
    cache_request_status status =
        m_tag_array->access(block_addr, time, cache_index, wb, evicted, mf);
    assert(status != HIT);
    cache_block_t *block = m_tag_array->get_block(cache_index);
    // ��� block ���� modified line�������� dirty ��������Ϊ������ʱ�� block ��
    // �� modified line��˵����� block �� clean line��������Ҫд�����ݣ������Ҫ��
    // ��� block ����Ϊ modified line�������Ļ���dirty ��������Ҫ���ӡ����� block 
    // �Ѿ��� modified line������Ҫ���� dirty ��������� block ���ϴα�� dirty 
    // ��ʱ��dirty �����Ѿ����ӹ��ˡ�
    if (!block->is_modified_line()) {
      m_tag_array->inc_dirty();
    }
    // ���� block ��״̬Ϊ modified������ block ����Ϊ MODIFIED�������Ļ����´���
    // ���������������� block ��ʱ�򣬾Ϳ���ֱ�Ӵ� cache �ж�ȡ���ݣ�������Ҫ�ٴ�
    // ������һ���洢��
    block->set_status(MODIFIED, mf->get_access_sector_mask());
    block->set_byte_mask(mf);

    // ��ʱ���ù������������
    if (status == HIT_RESERVED)
      block->set_ignore_on_fill(true, mf->get_access_sector_mask());

    // ֻҪ m_tag_array->access ���ص�״̬���� RESERVATION_FAIL����˵�����߷�����
    // HIT_RESERVED������ SECTOR_MISS���ֻ��� MISS������ֻҪ���� RESERVATION_FAIL��
    // �ʹ����� cache block �������ˣ����Ҫ�������������� cache block �Ƿ���Ҫд
    // �أ������ block д�ص���һ���洢��
    if (status != RESERVATION_FAIL) {
      // If evicted block is modified and not a write-through
      // (already modified lower level)
      // wb ������ʶ tag_array::access() �����У��������� m_tag_array->access 
      // �������� MISS������Ҫ���һ�� block��������� evicted block д�ص���һ��
      // �洢�������� block �Ѿ��� modified line���� wb Ϊ true����Ϊ�ڽ������
      // ���·���֮ǰ�����뽫����Ѿ� modified �� block д�ص���һ���洢���������  
      // �� block �� clean line���� wb Ϊ false����Ϊ��� block ����Ҫд�ص���һ 
      // ���洢����� evicted block ����Ϣ�������� evicted �С�
      // ������� cache ��д����Ϊдֱ��Ͳ���Ҫ�ڶ� miss ʱ��������� MODIFIED 
      // cache block д�ص���һ���洢����Ϊ��� cache block �ڱ� MODIFIED ��ʱ��
      // �Ѿ��� write-through ����һ���洢�ˡ�
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
        // д�� evicted block ����һ���洢��
        send_write_request(wb, cache_event(WRITE_BACK_REQUEST_SENT, evicted),
                           time, events);
      }
      // ����д MISS �����������й���ȫ����ɣ����ص��� write miss ���ԭʼд����
      // ��״̬��
      return MISS;
    }
    // ����д MISS ������û�з����µ� cache block����������� block д�أ���˷�
    // �� RESERVATION_FAIL��
    return RESERVATION_FAIL;
  } else {
    // �������д����ֽ���С������ cache line/sector �Ĵ�С����ô��Ҫ���Ͷ�����
    // ��һ���洢��Ȼ��д��Ĳ��ֺϲ����� sector �������� sector ����Ϊ���޸ġ�

    // MSHR �� m_data �� key �д洢�˸����ϲ��ĵ�ַ��probe() ������Ҫ����Ƿ����У�
    // ����Ҫ��� m_data.keys() ��������û�� mshr_addr��
    bool mshr_hit = m_mshrs.probe(mshr_addr);
    // ���Ȳ����Ƿ� MSHR ������ block_addr ��ַ����Ŀ��������ڸ���Ŀ������ MSHR����
    // ���Ƿ��пռ�ϲ�������Ŀ����������ڸ���Ŀ��δ���� MSHR�������Ƿ��������ռ���
    // ����� mshr_addr ��һ��Ŀ��
    bool mshr_avail = !m_mshrs.full(mshr_addr);
    // ��� m_miss_queue.size() �Ѿ����������������ݰ��Ļ����п����޷���ɺ���������
    // ��Ϊ���������Ҫִ��һ�� send_read_request ��һ�� send_write_request������
    // ���п��������Ҫ�� m_miss_queue ����������ݰ���
    // �� miss_queue_full(1) ���� false���п���ռ�֧��ִ��һ�� send_write_request
    // ��һ�� send_read_request����ô����Ҫ�� MSHR �Ƿ��п��ÿռ䡣�����⴮�ж�����
    // ��ʵ���Ի���ɣ� 
    //   if (miss_queue_full(1) || !mshr_avail)��
    // ������ RESERVATION_FAIL ��������
    //   1. m_miss_queue �����Է���һ����һ��д������������
    //   2. MSHR ���ܺϲ�����δ���У�����û�п��ÿռ��������Ŀ����
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
    // ���Ͷ�������һ���洢��Ȼ��д��Ĳ��ֺϲ����� sector �������� sector ��
    // ��Ϊ���޸ġ�
    send_read_request(addr, block_addr, cache_index, n_mf, time, do_miss, wb,
                      evicted, events, false, true);

    cache_block_t *block = m_tag_array->get_block(cache_index);
    // �� block ����Ϊ���´� fill ʱ����Ϊ MODIFIED�������Ļ����´���������������
    // �� fill ʱ��
    //   m_status = m_set_modified_on_fill ? MODIFIED : VALID; ��
    //   m_status[sidx] = m_set_modified_on_fill[sidx] ? MODIFIED : VALID;
    block->set_modified_on_fill(true, mf->get_access_sector_mask());
    block->set_byte_mask_on_fill(true);

    events.push_back(cache_event(WRITE_ALLOCATE_SENT));

    // do_miss ��ʶ�Ƿ��������� MSHR ���� ���ŵ� m_miss_queue ������һ������
    // ���͵���һ���洢��
    if (do_miss) {
      // If evicted block is modified and not a write-through
      // (already modified lower level)
      // wb ������ʶ tag_array::access() �����У��������� m_tag_array->access 
      // �������� MISS������Ҫ���һ�� block��������� evicted block д�ص���һ��
      // �洢�������� block �Ѿ��� modified line���� wb Ϊ true����Ϊ�ڽ������
      // ���·���֮ǰ�����뽫����Ѿ� modified �� block д�ص���һ���洢���������  
      // �� block �� clean line���� wb Ϊ false����Ϊ��� block ����Ҫд�ص���һ 
      // ���洢����� evicted block ����Ϣ�������� evicted �С�
      // ������� cache ��д����Ϊдֱ��Ͳ���Ҫ�ڶ� miss ʱ��������� MODIFIED 
      // cache block д�ص���һ���洢����Ϊ��� cache block �ڱ� MODIFIED ��ʱ��
      // �Ѿ��� write-through ����һ���洢�ˡ�
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
        // д�� evicted block ����һ���洢��
        send_write_request(wb, cache_event(WRITE_BACK_REQUEST_SENT, evicted),
                           time, events);
      }
      // ����д MISS �����������й���ȫ����ɣ����ص��� write miss ���ԭʼд����
      // ��״̬��
      return MISS;
    }
    // ����д MISS ������û�з����µ� cache block����������� block д�أ���˷�
    // �� RESERVATION_FAIL��
    return RESERVATION_FAIL;
  }
}

/*
write_allocated_lazy_fetch_on_read ���ԡ�
��Ҫ�ο� https://arxiv.org/pdf/1810.07269.pdf ���Ķ� Volta �ܹ��ô���Ϊ�Ľ��͡�
L2 ����Ӧ���˲�ͬ��д�������ԣ���������Ϊ�ӳٶ�ȡ��ȡ������д����֤��д��ʱ��ȡ֮��
�����Է��������յ������޸�����������������ʱ�������ȼ������д�����Ƿ��������������ֽ�
����д�벢�Ҹ�����ȫ�ɶ�������ǣ����ȡ��������������д��ʱ��ȡ���ƣ������ɸ�������
��ȡ���󲢽������޸ĺ���ֽںϲ���
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

  // �� m_miss_queue.size() �Ѿ���������һ�����ݰ��Ļ����п����޷���ɺ�����������
  // Ϊ���������Ҫִ��һ�� send_write_request���п��������Ҫ�� m_miss_queue ���
  // �������ݰ�����Ȼ��������������� send_write_request�������������ǲ���ͬʱ������
  if (miss_queue_full(0)) {
    m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL,
                           mf->get_streamID());
    return RESERVATION_FAIL;  // cannot handle request this cycle
  }

  // �� V100 �����У�L1 cache Ϊ 'T'-write through��L2 cache Ϊ 'B'-write back��
  if (m_config.m_write_policy == WRITE_THROUGH) {
    // ����� write through������Ҫֱ�ӽ�����һͬд����һ��洢��������д����һͬ��
    // ������һ���洢��������Ҫ�����ǽ����������� WRITE_REQUEST_SENT ���� events��
    // ������������ mf ���뵱ǰ cache �� m_miss_queue �У��ȴ� baseline_cache::
    // cycle() �����׵��������� mf ���͸���һ���洢��
    send_write_request(mf, cache_event(WRITE_REQUEST_SENT), time, events);
  }

  // wb ������ʶ tag_array::access() �����У��������� send_read_request ������
  // �� MISS������Ҫ���һ�� block��������� evicted block д�ص���һ���洢�����
  // ��� block �Ѿ��� modified line���� wb Ϊ true����Ϊ�ڽ��������·���֮ǰ��
  // ���뽫����Ѿ� modified �� block д�ص���һ���洢���������� block �� clean
  // line���� wb Ϊ false����Ϊ��� block ����Ҫд�ص���һ���洢����� evicted 
  // block ����Ϣ�������� evicted �С�
  bool wb = false;
  // evicted ��¼�ű������ cache block ����Ϣ��
  evicted_block_info evicted;
  // ���� tag_array ��״̬���������� LRU ״̬����������� block �� sector �ȡ�
  cache_request_status m_status =
      m_tag_array->access(block_addr, time, cache_index, wb, evicted, mf);

  // Theoretically, the passing parameter status should be the same as the 
  // m_status, if the assertion fails here, go to function :
  //     `wr_miss_wa_lazy_fetch_on_read` 
  // to remove this assertion.
  // assert((m_status == status));
  assert(m_status != HIT);
  // cache_index �� cache block �� index��
  cache_block_t *block = m_tag_array->get_block(cache_index);
  // ��� block ���� modified line�������� dirty ��������Ϊ������ʱ�� block ��
  // �� modified line��˵����� block �� clean line��������Ҫд�����ݣ������Ҫ��
  // ��� block ����Ϊ modified line�������Ļ���dirty ��������Ҫ���ӡ����� block 
  // �Ѿ��� modified line������Ҫ���� dirty ��������� block ���ϴα�� dirty 
  // ��ʱ��dirty �����Ѿ����ӹ��ˡ�
  if (!block->is_modified_line()) {
    m_tag_array->inc_dirty();
  }
  // ���� block ��״̬Ϊ modified������ block ����Ϊ MODIFIED�������Ļ����´���
  // ���������������� block ��ʱ�򣬾Ϳ���ֱ�Ӵ� cache �ж�ȡ���ݣ�������Ҫ�ٴ�
  // ������һ���洢��
  block->set_status(MODIFIED, mf->get_access_sector_mask());
  block->set_byte_mask(mf);
  // ��� Cache block[mask] ״̬�� RESERVED��˵�����������߳����ڶ�ȡ��� Cache 
  // block����������з��������д��� RESERVED ״̬�Ļ����У�����ζ��ͬһ�����Ѵ���
  // ����ǰ����δ���з��͵� flying �ڴ�����
  if (m_status == HIT_RESERVED) {
    // �ڵ�ǰ�汾�� GPGPU-Sim �У�set_ignore_on_fill ��ʱ�ò�����
    block->set_ignore_on_fill(true, mf->get_access_sector_mask());
    // cache block ��ÿ�� sector ����һ����־λ m_set_modified_on_fill[i]�����
    // ����� cache block �Ƿ��޸ģ���sector_cache_block::fill() �������õ�ʱ
    // ���ʹ�á�
    // �� block ����Ϊ���´� fill ʱ����Ϊ MODIFIED�������Ļ����´���������������
    // �� fill ʱ��
    //   m_status = m_set_modified_on_fill ? MODIFIED : VALID; ��
    //   m_status[sidx] = m_set_modified_on_fill[sidx] ? MODIFIED : VALID;
    block->set_modified_on_fill(true, mf->get_access_sector_mask());
    // �� FETCH_ON_READ policy: https://arxiv.org/pdf/1810.07269.pdf ���ᵽ��
    // ���� cache ���� miss ʱ��
    // In the write-validate policy, no read fetch is required, instead each  
    // sector has a bit-wise write-mask. When a write to a single byte is 
    // received, it writes the byte to the sector, sets the corresponding 
    // write bit and sets the sector as valid and modified. When a modified 
    // cache line is evicted, the cache line is written back to the memory 
    // along with the write mask.
    // ���� FETCH_ON_READ �У���Ҫ���� sector �� byte mask���������ָ������� 
    // byte mask �ı�־��
    block->set_byte_mask_on_fill(true);
  }

  // m_config.get_atom_sz() Ϊ SECTOR_SIZE = 4���� mf ���ʵ���һ���� 4 �ֽڡ�
  if (mf->get_access_byte_mask().count() == m_config.get_atom_sz()) {
    // ���� mf ���ʵ������� sector��������� sector ���� dirty �ģ����÷��ʵ� 
    // sector �ɶ���
    block->set_m_readable(true, mf->get_access_sector_mask());
  } else {
    // ���� mf ���ʵ��ǲ��� sector�����ֻ�� mf ���ʵ��ǲ��� sector �� dirty �ģ�
    // ���÷��ʵ� sector ���ɶ��������������´���� sector �� fill ʱ��mf->get_
    // access_sector_mask() ��ʶ�� byte ��Ϊ MODIFIED��
    block->set_m_readable(false, mf->get_access_sector_mask());
    if (m_status == HIT_RESERVED)
      block->set_readable_on_fill(true, mf->get_access_sector_mask());
  }
  // ����һ�� cache block ��״̬Ϊ�ɶ���������е� byte mask λȫ������Ϊ dirty 
  // �ˣ��򽫸� sector ������Ϊ�ɶ�����Ϊ��ǰ�� sector �Ѿ���ȫ������Ϊ����ֵ�ˣ�
  // �ǿɶ��ġ�������������е��������� mf �����з��ʵ� sector ���б�������� mf 
  // �����ʵ����е� byte mask λȫ������Ϊ dirty �ˣ��򽫸� cache block ����Ϊ��
  // ����
  update_m_readable(mf, cache_index);

  // ֻҪ m_tag_array->access ���ص�״̬���� RESERVATION_FAIL����˵�����߷�����
  // HIT_RESERVED������ SECTOR_MISS���ֻ��� MISS������ֻҪ���� RESERVATION_FAIL��
  // �ʹ����� cache block �������ˣ����Ҫ�������������� cache block �Ƿ���Ҫд
  // �أ������ block д�ص���һ���洢��
  if (m_status != RESERVATION_FAIL) {
    // If evicted block is modified and not a write-through
    // (already modified lower level)
    // wb ������ʶ tag_array::access() �����У��������� m_tag_array->access 
    // �������� MISS������Ҫ���һ�� block��������� evicted block д�ص���һ��
    // �洢�������� block �Ѿ��� modified line���� wb Ϊ true����Ϊ�ڽ������
    // ���·���֮ǰ�����뽫����Ѿ� modified �� block д�ص���һ���洢���������  
    // �� block �� clean line���� wb Ϊ false����Ϊ��� block ����Ҫд�ص���һ 
    // ���洢����� evicted block ����Ϣ�������� evicted �С�
    // ������� cache ��д����Ϊдֱ��Ͳ���Ҫ�ڶ� miss ʱ��������� MODIFIED 
    // cache block д�ص���һ���洢����Ϊ��� cache block �ڱ� MODIFIED ��ʱ��
    // �Ѿ��� write-through ����һ���洢�ˡ�
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
      // д�� evicted block ����һ���洢��
      send_write_request(wb, cache_event(WRITE_BACK_REQUEST_SENT, evicted),
                         time, events);
    }
    // ����д MISS �����������й���ȫ����ɣ����ص��� write miss ���ԭʼд����
    // ��״̬��
    return MISS;
  }
  // ����д MISS ������û�з����µ� cache block����������� block д�أ���˷�
  // �� RESERVATION_FAIL��
  return RESERVATION_FAIL;
}

/// No write-allocate miss: Simply send write request to lower level memory
/*
No write-allocate miss����������������򵥵ؽ�д�����͵���һ���洢��
*/
enum cache_request_status data_cache::wr_miss_no_wa(
    new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events, enum cache_request_status status) {
  // ��� m_miss_queue.size() �Ѿ���������һ�����ݰ��Ļ����п����޷���ɺ���������
  // ��Ϊ���������Ҫִ��һ�� send_write_request���� send_write_request ��ÿִ��
  // һ�Σ�����Ҫ�� m_miss_queue ���һ�����ݰ���
  if (miss_queue_full(0)) {
    m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL,
                           mf->get_streamID());
    return RESERVATION_FAIL;  // cannot handle request this cycle
  }

  // on miss, generate write through (no write buffering -- too many threads for
  // that)
  // send_write_request ִ�У�
  //   events.push_back(request);
  //   // �� baseline_cache::cycle() �У��Ὣ m_miss_queue ���׵����ݰ� mf ����
  //   // ����һ���洢��
  //   m_miss_queue.push_back(mf);
  //   mf->set_status(m_miss_queue_status, time);
  // No write-allocate miss ������д MISS ʱ��ֱ�ӽ� mf ���ݰ�ֱ��д����һ���洢��
  // ������Ҫ�����ǽ�д�������� WRITE_REQUEST_SENT ���� events�����������������  
  // m_miss_queue �У��ȴ�baseline_cache::cycle() �� m_miss_queue ���׵�����д
  // ���� mf ���͸���һ���洢��
  send_write_request(mf, cache_event(WRITE_REQUEST_SENT), time, events);

  return MISS;
}

/****** Read hit functions (Set by config file) ******/

/// Baseline read hit: Update LRU status of block.
// Special case for atomic instructions -> Mark block as modified
/*
READ HIT ������
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
  // ���� tag_array ��״̬���������� LRU ״̬����������� block �� sector �ȡ�
  m_tag_array->access(block_addr, time, cache_index, mf);
  // Atomics treated as global read/write requests - Perform read, mark line as
  // MODIFIED
  // ԭ�Ӳ�����ָ��ȫ�ֺ͹����ڴ��е�32λ����64λ���ݽ��� ����ȡ-�޸�-��д�� ��һ��
  // ����ԭ�Ӳ������Կ�����һ����С��λ��ִ�й��̡�����ִ�й����У�����������������
  // �̶Ըñ������ж�ȡ��д��Ĳ�������������������������̱߳���ȴ���
  // ԭ�Ӳ�����ȫ�ִ洢ȡֵ�����㣬��д����ͬ��ַ����������ͬһԭ�Ӳ�������ɣ����
  // ���޸� cache ��״̬Ϊ MODIFIED��
  if (mf->isatomic()) {
    assert(mf->get_access_type() == GLOBAL_ACC_R);
    // ��ȡ��ԭ�Ӳ����� cache block�����ж����Ƿ���ǰ�ѱ� MODIFIED�������ǰδ�� 
    // MODIFIED���˴�ԭ�Ӳ������� MODIFIED��Ҫ���� dirty ��Ŀ�������ǰ block ��
    // ���� MODIFIED������ǰdirty ��Ŀ�Ѿ����ӹ��ˣ��Ͳ���Ҫ�������ˡ�
    cache_block_t *block = m_tag_array->get_block(cache_index);
    if (!block->is_modified_line()) {
      m_tag_array->inc_dirty();
    }
    // ���� cache block ��״̬Ϊ MODIFIED���Ա��������߳������ cache block �ϵ�
    // ��д������
    block->set_status(MODIFIED,
                      mf->get_access_sector_mask());  // mark line as
    // ���� dirty_byte_mask��
    block->set_byte_mask(mf);
  }
  return HIT;
}

/****** Read miss functions (Set by config file) ******/

/// Baseline read miss: Send read request to lower level memory,
// perform write-back as necessary
/*
READ MISS ������
*/
enum cache_request_status data_cache::rd_miss_base(
    new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events, enum cache_request_status status) {
  // �� miss ʱ������Ҫ����������������һ���洢�����������Ҫ��ʵ������һ���洢��
  // �Ͷ�����Ҳ�������� mshr �Ĵ��ڣ����Խ���������ϲ���ȥ�������Ͳ���Ҫ��ʵ����
  // ��һ���洢���Ͷ�����
  // miss_queue_full ����Ƿ�һ�� miss �����ܹ��ڵ�ǰʱ�������ڱ�������һ������
  // �Ĵ�С�� m_miss_queue �Ų���ʱ���ڵ�ǰ�����޷��������� RESERVATION_FAIL��
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
  // ��ʶ�Ƿ��������� MSHR ���� ���ŵ� m_miss_queue ������һ�����ڷ��͵���һ
  // ���洢��
  bool do_miss = false;
  // wb �����Ƿ���Ҫд�أ���һ��������� cache block �� MODIFIED ʱ����Ҫд�ص�
  // ��һ���洢����evicted ��������� cache line ����Ϣ��
  bool wb = false;
  evicted_block_info evicted;
  // READ MISS ����������� MSHR �Ƿ����л��� MSHR �Ƿ���ã������ж��Ƿ���Ҫ
  // ����һ���洢���Ͷ�����
  send_read_request(addr, block_addr, cache_index, mf, time, do_miss, wb,
                    evicted, events, false, false);
  // ��� send_read_request �����������Ѿ������뵽 MSHR������ԭ�ȴ��ڸ���Ŀ����
  // ��ϲ���ȥ������ԭ�Ȳ����ڸ���Ŀ����������ȥ����ô do_miss Ϊ true������
  // Ҫ��ĳ�� cache block ��������� mf ����һ���洢���ص����ݡ�
  // m_lines[idx] ��Ϊ����� reserve �·��ʵ� cache line���������ĳ�� sector 
  // �Ѿ��� MODIFIED������Ҫִ��д�ز���������д�صı�־Ϊ wb = true��
  if (do_miss) {
    // If evicted block is modified and not a write-through
    // (already modified lower level).
    // ������� cache ��д����Ϊдֱ��Ͳ���Ҫ�ڶ� miss ʱ��������� MODIFIED 
    // cache block д�ص���һ���洢����Ϊ��� cache block �ڱ� MODIFIED ��ʱ��
    // �Ѿ��� write-through ����һ���洢�ˡ�
    if (wb && (m_config.m_write_policy != WRITE_THROUGH)) {
      // ����д���󣬽� MODIFIED �ı������ cache block д�ص���һ���洢��
      // �� V100 �У�
      //     m_wrbk_type��L1 cache Ϊ L1_WRBK_ACC��L2 cache Ϊ L2_WRBK_ACC��
      //     m_write_policy��L1 cache Ϊ WRITE_THROUGH��
      mem_fetch *wb = m_memfetch_creator->alloc(
          evicted.m_block_addr, m_wrbk_type, mf->get_access_warp_mask(),
          evicted.m_byte_mask, evicted.m_sector_mask, evicted.m_modified_size,
          true, m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, -1, -1, -1,
          NULL, mf->get_streamID());
      // the evicted block may have wrong chip id when advanced L2 hashing  is
      // used, so set the right chip address from the original mf
      wb->set_chip(mf->get_tlx_addr().chip);
      wb->set_partition(mf->get_tlx_addr().sub_partition);
      // ������д����һͬ��������һ���洢��
      // ��Ҫ�����ǽ����������� WRITE_BACK_REQUEST_SENT���� events������������
      // �� mf ���뵱ǰ cache �� m_miss_queue �У��� baseline_cache::cycle() 
      // �����׵��������� mf ���͸���һ���洢��
      send_write_request(wb, WRITE_BACK_REQUEST_SENT, time, events);
    }
    return MISS;
  }
  return RESERVATION_FAIL;
}

/// Access cache for read_only_cache: returns RESERVATION_FAIL if
// request could not be accepted (for any reason)
/*
read_only_cache ���ʣ����� L1I��L1C��
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
  //cache_index�᷵������tagλѡ�е�cache block��������
  unsigned cache_index = (unsigned)-1;
  //�ж϶�cache�ķ��ʣ���ַΪaddr��sector maskΪmask����HIT/HIT_RESERVED/SECTOR_MISS/
  //MISS/RESERVATION_FAIL��״̬��
  enum cache_request_status status =
      m_tag_array->probe(block_addr, cache_index, mf, mf->is_write());
  enum cache_request_status cache_status = RESERVATION_FAIL;

  if (status == HIT) {
    //������LRU״̬��
    cache_status = m_tag_array->access(block_addr, time, cache_index,
                                       mf);  // update LRU state
  } else if (status != RESERVATION_FAIL) {
    //HIT_RESERVED/SECTOR_MISS/MISS״̬��
    if (!miss_queue_full(0)) {
      bool do_miss = false;
      //READ MISS�����������MSHR�Ƿ����л���MSHR�Ƿ���ã������ж��Ƿ���Ҫ����һ���洢��
      //�Ͷ�����
      send_read_request(addr, block_addr, cache_index, mf, time, do_miss,
                        events, true, false);
      //����Ҫ��ĳ��cache block���������mf����һ���洢���ص����ݡ�
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
һ��ͨ�ú���������ȡtag_array̽��Ľ�������ݻ�������ִ����Ӧ�Ĺ��ܡ�
access������������
��һ��cache�������ݷ��ʵ�ʱ�򣬵���data_cache::access()������
- ����cahe�����m_tag_array->probe()�������ж϶�cache�ķ��ʣ���ַΪaddr��sector mask
  Ϊmask����HIT/HIT_RESERVED/SECTOR_MISS/MISS/RESERVATION_FAIL��״̬��
- Ȼ�����process_tag_probe()����������cache�������Լ�����m_tag_array->probe()������
  �ص�cache����״̬��ִ����Ӧ�Ĳ�����
  - process_tag_probe()�����У����������Ķ�д״̬��probe()�������ص�cache����״̬��
    ִ��m_wr_hit/m_wr_miss/m_rd_hit/m_rd_miss���������ǻ����m_tag_array->access()
    ������ʵ��LRU״̬�ĸ��¡�
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
      //�������cache_index��д��cache block��������
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
L1 �� L2 Ŀǰʹ����ͬ�ķ��ʹ��ܡ���������֮���������ͨ�����û����������ɵġ�
L1 �� L2 �����Ǵ˺��������ṩ�ڰ����������ʱִ���ض���ÿ������Ĳ����ķ�����
��cache�������ݷ��ʡ�

��һ��cache�������ݷ��ʵ�ʱ�򣬵���data_cache::access()������
- ����cahe�����m_tag_array->probe()�������ж϶�cache�ķ��ʣ���ַΪaddr��sector mask
  Ϊmask����HIT/HIT_RESERVED/SECTOR_MISS/MISS/RESERVATION_FAIL��״̬��
- Ȼ�����process_tag_probe()����������cache�������Լ�����m_tag_array->probe()������
  �ص�cache����״̬��ִ����Ӧ�Ĳ�����
  - process_tag_probe()�����У����������Ķ�д״̬��probe()�������ص�cache����״̬��
    ִ��m_wr_hit/m_wr_miss/m_rd_hit/m_rd_miss���������ǻ����m_tag_array->access()
    ������ʵ��LRU״̬�ĸ��¡�
*/
enum cache_request_status data_cache::access(new_addr_type addr, mem_fetch *mf,
                                             unsigned time,
                                             std::list<cache_event> &events) {
  //m_config.get_atom_sz()��cache�滻ԭ�Ӳ��������ȣ����cache��SECTOR���͵ģ�����Ϊ
  //SECTOR_SIZE������Ϊline_size��
  assert(mf->get_data_size() <= m_config.get_atom_sz());
  bool wr = mf->get_is_write();
  // m_config.block_addr(addr): 
  //     return addr & ~(new_addr_type)(m_line_sz - 1);
  // |-------|-------------|--------------|
  //            set_index   offset in-line
  // |<--------tag--------> 0 0 0 0 0 0 0 | 
  new_addr_type block_addr = m_config.block_addr(addr);
  //cache_index�᷵������tagλѡ�е�cache block��������
  unsigned cache_index = (unsigned)-1;
  //�ж϶�cache�ķ��ʣ���ַΪaddr��sector maskΪmask����HIT/HIT_RESERVED/SECTOR_MISS/
  //MISS/RESERVATION_FAIL��״̬������������� MISS������Ҫ���滻��cache block������
  //д��cache_index��
  enum cache_request_status probe_status =
      m_tag_array->probe(block_addr, cache_index, mf, mf->is_write(), true);
  //��Ҫ��������״̬�µ�cache���ʲ���������(this->*m_wr_hit)��(this->*m_wr_miss)��
  //(this->*m_rd_hit)��(this->*m_rd_miss)��
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
����Ϊ�˶�Fermi�еĵ�һ�����ݻ�����н�ģ�����ǵ��������ȵ�д�����global����д�أ�local��
���� GPGPU-Sim �����ļ����ã������� CUDA �ֲ���Fermi��ʹ�õĲ��ԣ���
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
l2 ������ʺ������û���data_cache����ʵ�֡� ��L2��Ҫ��L1�в�һ�µĹ���ʱ��Ӧ�ڴ˴�����L2
���ض����ġ�
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
TEX Cache ��ǰ�ƽ�һ�ġ�
*/
void tex_cache::cycle() {
  // send next request to lower level of memory
  if (!m_request_fifo.empty()) {
    mem_fetch *mf = m_request_fifo.peek();
    if (!m_memport->full(mf->get_ctrl_size(), false)) {
      m_request_fifo.pop();
      // mem_fetch_interface �� cache �� mem �ô�Ľӿڣ�cache �� miss ����������һ��
      // �洢����ͨ������ӿ������ͣ��� m_miss_queue �е����ݰ���Ҫѹ�� m_memport ʵ�ַ�
      // ������һ���洢��
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
