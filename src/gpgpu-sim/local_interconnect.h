// Copyright (c) 2019, Mahmoud Khairy
// Purdue University
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

#ifndef _LOCAL_INTERCONNECT_HPP_
#define _LOCAL_INTERCONNECT_HPP_

#include <iostream>
#include <map>
#include <queue>
#include <vector>
using namespace std;

enum Interconnect_type { REQ_NET = 0, REPLY_NET = 1 };

enum Arbiteration_type { NAIVE_RR = 0, iSLIP = 1 };

/*
ICNT配置类。
*/
struct inct_config {
  // config for local interconnect
  //in_buffers[deviceID]缓冲区最大可容纳数据包个数。
  unsigned in_buffer_limit;
  //out_buffers[deviceID]缓冲区最大可容纳数据包个数。
  unsigned out_buffer_limit;
  //子网络个数。
  unsigned subnets;
  //icnt_arbiter_algo，在V100中配置为1=iSLIP算法。
  Arbiteration_type arbiter_algo;
  //是否输出详细信息。
  unsigned verbose;
  //
  unsigned grant_cycles;
};

/*
子网络类。
*/
class xbar_router {
 public:
  //Xbar路由器。子网络构造函数。
  xbar_router(unsigned router_id, enum Interconnect_type m_type,
              unsigned n_shader, unsigned n_mem,
              const struct inct_config& m_localinct_config);
  ~xbar_router();
  //将数据包推入子网络。
  void Push(unsigned input_deviceID, unsigned output_deviceID, void* data,
            unsigned int size);
  //将数据包从子网络弹出。
  void* Pop(unsigned ouput_deviceID);
  //执行路由一拍。
  void Advance();

  //当所有输入缓冲和输出缓冲都没有数据包时，认为当前子网络处于空闲状态，反之则是Busy
  //状态。
  bool Busy() const;
  //判断当前子网络是否有足够的输入缓冲区能够容纳size大小的新数据包。
  bool Has_Buffer_In(unsigned input_deviceID, unsigned size,
                     bool update_counter = false);
  //判断当前子网络是否有足够的输出缓冲区能够容纳size大小的新数据包。
  bool Has_Buffer_Out(unsigned output_deviceID, unsigned size);

  // some stats
  //互连子网络执行路由的周期总数，无论这一拍有没有路由数据包。
  unsigned long long cycles;
  //conflicts是在整个程序执行期间，数据包的目的设备号有冲突的次数，比如第0号和第1号
  //设备都有数据包发送到第25号设备，那么算冲突一次。
  unsigned long long conflicts;
  //conflicts_util是在[互连子网络的输入buffer中有数据包的周期数]期间，即互连子网络
  //有效利用期间，数据包的目的设备号有冲突的次数，比如第0号和第1号设备都有数据包发送
  //到第25号设备，那么算冲突一次。
  unsigned long long conflicts_util;
  //cycles_util是互连子网络的输入buffer中有数据包的周期数，即互连子网络有效利用的周
  //期数，后面用于统计。
  unsigned long long cycles_util;
  //reqs_util是在[互连子网络的输入buffer中有数据包的周期数]期间，即互连子网络有效利
  //用期间，互连子网络路由的数据包的总数。
  unsigned long long reqs_util;
  //某一个节点的输出缓冲区满了就增加一次，因此统计的是在整个程序执行期间，输出缓冲区
  //满了的总次数。注意两个节点在同一拍满了算两次。
  unsigned long long out_buffer_full;
  unsigned long long out_buffer_util;
  //某一个节点的输入缓冲区满了就增加一次，因此统计的是在整个程序执行期间，输入缓冲区
  //满了的总次数。注意两个节点在同一拍满了算两次。
  unsigned long long in_buffer_full;
  unsigned long long in_buffer_util;
  //整个程序执行周期内，进入子网络的数据包的总个数。
  unsigned long long packets_num;

 private:
  //执行路由一拍。
  void iSLIP_Advance();
  void RR_Advance();

  //数据包类。
  struct Packet {
    Packet(void* m_data, unsigned m_output_deviceID) {
      data = m_data;
      output_deviceID = m_output_deviceID;
    }
    //数据。
    void* data;
    //输出到哪个设备ID。
    unsigned output_deviceID;
  };
  //数据包的输入缓冲区，其大小被设置为节点总数=SM总数量+内存子分区的个数。
  vector<queue<Packet> > in_buffers;
  //数据包的输出缓冲区，其大小被设置为节点总数=SM总数量+内存子分区的个数。
  vector<queue<Packet> > out_buffers;
  //SM总数量，内存子分区的个数，节点总数。
  //total_nodes = _n_shader + _n_mem。
  unsigned _n_shader, _n_mem, total_nodes;
  //in_buffer_limit是in_buffers[deviceID]缓冲区最大可容纳数据包个数。
  //out_buffer_limit是out_buffers[deviceID]缓冲区最大可容纳数据包个数。
  unsigned in_buffer_limit, out_buffer_limit;
  //用于iSLIP算法仲裁。
  vector<unsigned> next_node;  // used for iSLIP arbit
  unsigned next_node_id;       // used for RR arbit
  //子网络的ID。
  unsigned m_id;
  //REQ_NET或REPLY_NET，子网络个数在V100中配置为2，0号自网络负责REQ_NET，1号子网
  //络负责REPLY_NET。
  enum Interconnect_type router_type;
  //如果是REQ_NET（0号子网络），数据包由SM转发至内存子分区，则：
  //    激活的输入缓冲区个数为SM数量。
  //    激活的输出缓冲区个数为内存子分区数量。
  //如果是REPLY_NET（1号子网络），数据包由内存子分区转发至SM，则：
  //    激活的输入缓冲区个数为内存子分区数量。
  //    激活的输出缓冲区个数为SM数量。
  unsigned active_in_buffers, active_out_buffers;
  //仲裁类型，icnt_arbiter_algo，在V100中配置为1=iSLIP算法。
  Arbiteration_type arbit_type;
  unsigned verbose;

  unsigned grant_cycles;
  unsigned grant_cycles_count;

  friend class LocalInterconnect;
};

/*
互连网络。
*/
class LocalInterconnect {
 public:
 //构造函数。
  LocalInterconnect(const struct inct_config& m_localinct_config);
  ~LocalInterconnect();
  //构造函数。
  static LocalInterconnect* New(const struct inct_config& m_inct_config);
  //创建互连网络。
  void CreateInterconnect(unsigned n_shader, unsigned n_mem);

  // node side functions
  void Init();
  //数据包压入互连网络输入缓冲区。
  void Push(unsigned input_deviceID, unsigned output_deviceID, void* data,
            unsigned int size);
  //数据包弹出互连网络输出缓冲区。
  void* Pop(unsigned ouput_deviceID);
  //互连网络执行路由一拍。
  void Advance();
  //判断互连网络是否处于Busy状态。有任意一个子网络处于Busy状态便认为整个互连网络处于
  //Busy状态。
  bool Busy() const;
  //判断互连网络是否有空闲的输入缓冲可以容纳来自deviceID号设备新的数据包。
  bool HasBuffer(unsigned deviceID, unsigned int size) const;
  void DisplayStats() const;
  void DisplayOverallStats() const;
  unsigned GetFlitSize() const;

  void DisplayState(FILE* fp) const;

 protected:
  //互连网络配置。
  const inct_config& m_inct_config;
  //SM节点数量和内存子分区节点数量。
  unsigned n_shader, n_mem;
  //子网络数量。
  unsigned n_subnets;
  //存储子网络的向量，net[REQ_NET]和net[REPLY_NET]。
  vector<xbar_router*> net;
};

#endif
