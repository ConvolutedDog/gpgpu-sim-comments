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

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <utility>

#include "local_interconnect.h"
#include "mem_fetch.h"

/*
Xbar路由器。子网络构造函数。
*/
xbar_router::xbar_router(unsigned router_id, enum Interconnect_type m_type,
                         unsigned n_shader, unsigned n_mem,
                         const struct inct_config& m_localinct_config) {
  //路由器ID。
  m_id = router_id;
  //REQ_NET或REPLY_NET，子网络个数在V100中配置为2，0号自网络负责REQ_NET，1号子
  //网络负责REPLY_NET。
  router_type = m_type;
  //内存子分区的个数。
  _n_mem = n_mem;
  //SM总数量。
  _n_shader = n_shader;
  //总节点数=SM总数量+内存子分区的个数。
  total_nodes = n_shader + n_mem;
  //是否打印详细信息的开关选项。
  verbose = m_localinct_config.verbose;
  grant_cycles = m_localinct_config.grant_cycles;
  grant_cycles_count = m_localinct_config.grant_cycles;
  in_buffers.resize(total_nodes);
  out_buffers.resize(total_nodes);
  next_node.resize(total_nodes, 0);
  //in_buffers[deviceID]缓冲区最大可容纳数据包个数。
  in_buffer_limit = m_localinct_config.in_buffer_limit;
  //out_buffers[deviceID]缓冲区最大可容纳数据包个数。
  out_buffer_limit = m_localinct_config.out_buffer_limit;
  //仲裁类型，icnt_arbiter_algo，在V100中配置为1=iSLIP算法。
  arbit_type = m_localinct_config.arbiter_algo;
  next_node_id = 0;
  if (m_type == REQ_NET) {
    //如果是REQ_NET（0号子网络），数据包由SM转发至内存子分区，则：
    //    激活的输入缓冲区个数为SM数量。
    //    激活的输出缓冲区个数为内存子分区数量。
    active_in_buffers = n_shader;
    active_out_buffers = n_mem;
  } else if (m_type == REPLY_NET) {
    //如果是REPLY_NET（1号子网络），数据包由内存子分区转发至SM，则：
    //    激活的输入缓冲区个数为内存子分区数量。
    //    激活的输出缓冲区个数为SM数量。
    active_in_buffers = n_mem;
    active_out_buffers = n_shader;
  }

  //互连子网络执行路由的周期总数，无论这一拍有没有路由数据包。
  cycles = 0;
  //conflicts是在整个程序执行期间，数据包的目的设备号有冲突的次数，比如第0号和第1号设
  //备都有数据包发送到第25号设备，那么算冲突一次。
  conflicts = 0;
  //某一个节点的输出缓冲区满了就增加一次，因此统计的是在整个程序执行期间，输出缓冲区满
  //了的总次数。注意两个节点在同一拍满了算两次。
  out_buffer_full = 0;
  //某一个节点的输入缓冲区满了就增加一次，因此统计的是在整个程序执行期间，输入缓冲区满
  //了的总次数。注意两个节点在同一拍满了算两次。
  in_buffer_full = 0;
  out_buffer_util = 0;
  in_buffer_util = 0;
  //整个程序执行周期内，进入子网络的数据包的总个数。
  packets_num = 0;
  //conflicts_util是在[互连子网络的输入buffer中有数据包的周期数]期间，即互连子网络有
  //效利用期间，数据包的目的设备号有冲突的次数，比如第0号和第1号设备都有数据包发送到第
  //25号设备，那么算冲突一次。
  conflicts_util = 0;
  //cycles_util是互连子网络的输入buffer中有数据包的周期数，即互连子网络有效利用的周期
  //数，后面用于统计。
  cycles_util = 0;
  //reqs_util是在[互连子网络的输入buffer中有数据包的周期数]期间，即互连子网络有效利用
  //期间，互连子网络路由的数据包的总数。
  reqs_util = 0;
}

xbar_router::~xbar_router() {}

/*
将数据包推入子网络。
*/
void xbar_router::Push(unsigned input_deviceID, unsigned output_deviceID,
                       void* data, unsigned int size) {
  assert(input_deviceID < total_nodes);
  in_buffers[input_deviceID].push(Packet(data, output_deviceID));
  //整个程序执行周期内，进入子网络的数据包的总个数。
  packets_num++;
}

/*
将数据包从子网络弹出。
*/
void* xbar_router::Pop(unsigned ouput_deviceID) {
  assert(ouput_deviceID < total_nodes);
  void* data = NULL;
  //子网络输出缓冲区不为空时，弹出front数据包。
  if (!out_buffers[ouput_deviceID].empty()) {
    data = out_buffers[ouput_deviceID].front().data;
    out_buffers[ouput_deviceID].pop();
  }

  return data;
}

/*
判断当前子网络是否有足够的输入缓冲区能够容纳size大小的新数据包。
*/
bool xbar_router::Has_Buffer_In(unsigned input_deviceID, unsigned size,
                                bool update_counter) {
  assert(input_deviceID < total_nodes);

  //in_buffers[input_deviceID].size()大小是当前已经在输入缓冲区里的数据包数量，
  //如果该数量 + size > 输入缓冲区的大小限制，则代表不能容纳新的size大小的数据包。
  bool has_buffer =
      (in_buffers[input_deviceID].size() + size <= in_buffer_limit);
  //某一个节点的输入缓冲区满了就增加一次，因此统计的是在整个程序执行期间，输入缓冲
  //区满了的总次数。注意两个节点在同一拍满了算两次。
  if (update_counter && !has_buffer) in_buffer_full++;

  return has_buffer;
}

/*
判断当前子网络是否有足够的输出缓冲区能够容纳size大小的新数据包。
*/
bool xbar_router::Has_Buffer_Out(unsigned output_deviceID, unsigned size) {
  //out_buffers[output_deviceID].size()大小是当前已经在输出缓冲区里的数据包数量，
  //如果该数量 + size > 输出缓冲区的大小限制，则代表不能容纳新的size大小的数据包。
  return (out_buffers[output_deviceID].size() + size <= out_buffer_limit);
}

/*
执行路由一拍。
*/
void xbar_router::Advance() {
  //仲裁类型，icnt_arbiter_algo，在V100中配置为1=iSLIP算法。
  if (arbit_type == NAIVE_RR)
    RR_Advance();
  else if (arbit_type == iSLIP)
    iSLIP_Advance();
  else
    assert(0);
}

void xbar_router::RR_Advance() {
  bool active = false;
  vector<bool> issued(total_nodes, false);
  unsigned conflict_sub = 0;
  unsigned reqs = 0;

  for (unsigned i = 0; i < total_nodes; ++i) {
    unsigned node_id = (i + next_node_id) % total_nodes;

    if (!in_buffers[node_id].empty()) {
      active = true;
      Packet _packet = in_buffers[node_id].front();
      // ensure that the outbuffer has space and not issued before in this cycle
      if (Has_Buffer_Out(_packet.output_deviceID, 1)) {
        if (!issued[_packet.output_deviceID]) {
          out_buffers[_packet.output_deviceID].push(_packet);
          in_buffers[node_id].pop();
          issued[_packet.output_deviceID] = true;
          reqs++;
        } else
          conflict_sub++;
      } else {
        out_buffer_full++;

        if (issued[_packet.output_deviceID]) conflict_sub++;
      }
    }
  }
  next_node_id = next_node_id + 1;
  next_node_id = (next_node_id % total_nodes);

  conflicts += conflict_sub;
  if (active) {
    conflicts_util += conflict_sub;
    cycles_util++;
    reqs_util += reqs;
  }

  if (verbose) {
    printf("%d : cycle %llu : conflicts = %d\n", m_id, cycles, conflict_sub);
    printf("%d : cycle %llu : passing reqs = %d\n", m_id, cycles, reqs);
  }

  // collect some stats about buffer util
  for (unsigned i = 0; i < total_nodes; ++i) {
    in_buffer_util += in_buffers[i].size();
    out_buffer_util += out_buffers[i].size();
  }

  cycles++;
}

// iSLIP algorithm
// McKeown, Nick. "The iSLIP scheduling algorithm for input-queued switches."
// IEEE/ACM transactions on networking 2 (1999): 188-201.
// https://www.cs.rutgers.edu/~sn624/552-F18/papers/islip.pdf
/*
执行路由一拍。
*/
void xbar_router::iSLIP_Advance() {
  vector<unsigned> node_tmp;
  bool active = false;

  unsigned conflict_sub = 0;
  //reqs是当前拍，互连子网络路由的数据包的总数。
  unsigned reqs = 0;

  // calcaulte how many conflicts are there for stats
  //这里是遍历所有节点，看它们的输入buffer中，是否有相同的输出目的设备output_deviceID，
  //如果存在则说明有冲突。
  for (unsigned i = 0; i < total_nodes; ++i) {
    if (!in_buffers[i].empty()) {
      Packet _packet_tmp = in_buffers[i].front();
      if (!node_tmp.empty()) {
        if (std::find(node_tmp.begin(), node_tmp.end(),
                      _packet_tmp.output_deviceID) != node_tmp.end()) {
          conflict_sub++;
        } else
          node_tmp.push_back(_packet_tmp.output_deviceID);
      } else {
        node_tmp.push_back(_packet_tmp.output_deviceID);
      }
      active = true;
    }
  }

  //conflicts是在整个程序执行期间，数据包的目的设备号有冲突的次数，比如第0号和第1号设备
  //都有数据包发送到第25号设备，那么算冲突一次。
  conflicts += conflict_sub;
  if (active) {
    //conflicts_util是在[互连子网络的输入buffer中有数据包的周期数]期间，即互连子网络有
    //效利用期间，数据包的目的设备号有冲突的次数，比如第0号和第1号设备都有数据包发送到第
    //25号设备，那么算冲突一次。
    conflicts_util += conflict_sub;
    //cycles_util是互连子网络的输入buffer中有数据包的周期数，即互连子网络有效利用的周期
    //数，后面用于统计。
    cycles_util++;
  }
  // do iSLIP
  //这里遍历所有节点，为这些所有节点的输出缓冲区选择应路由的数据包。
  for (unsigned i = 0; i < total_nodes; ++i) {
    //如果第i号节点的输出缓冲区还可以接收新的数据包。
    if (Has_Buffer_Out(i, 1)) {
      //对所有节点遍历，看这些节点中哪些节点的输入缓冲区里有需要路由到第i号节点的数据包。
      for (unsigned j = 0; j < total_nodes; ++j) {
        //这里是轮盘调度策略。next_node[i]在第i个输出缓冲区接收一个新数据包时，向下旋转
        //一次。
        unsigned node_id = (j + next_node[i]) % total_nodes;
        //下面判断第(j + next_node[i])% total_nodes个节点的输入缓冲里是否有目的节点为
        //i的数据包，如果有的话，则将其从输入缓冲里弹出，并压入第i号节点的输出缓冲区。
        if (!in_buffers[node_id].empty()) {
          Packet _packet = in_buffers[node_id].front();
          if (_packet.output_deviceID == i) {
            out_buffers[_packet.output_deviceID].push(_packet);
            in_buffers[node_id].pop();
            if (verbose)
              printf("%d : cycle %llu : send req from %d to %d\n", m_id, cycles,
                     node_id, i - _n_shader);
            if (grant_cycles_count == 1)
              next_node[i] = (++node_id % total_nodes);
            if (verbose) {
              for (unsigned k = j + 1; k < total_nodes; ++k) {
                unsigned node_id2 = (k + next_node[i]) % total_nodes;
                if (!in_buffers[node_id2].empty()) {
                  Packet _packet2 = in_buffers[node_id2].front();

                  if (_packet2.output_deviceID == i)
                    printf("%d : cycle %llu : cannot send req from %d to %d\n",
                           m_id, cycles, node_id2, i - _n_shader);
                }
              }
            }
            //reqs是当前拍，互连子网络路由的数据包的总数。
            reqs++;
            break;
          }
        }
      }
    } else
      //某一个节点的输出缓冲区满了就增加一次，因此统计的是在整个程序执行期间，输出缓冲
      //区满了的总次数。注意两个节点在同一拍满了算两次。
      out_buffer_full++;
  }

  if (active) {
    //reqs是当前拍，互连子网络路由的数据包的总数。
    //reqs_util是在[互连子网络的输入buffer中有数据包的周期数]期间，即互连子网络有效利
    //用期间，互连子网络路由的数据包的总数。
    reqs_util += reqs;
  }

  if (verbose)
    printf("%d : cycle %llu : grant_cycles = %d\n", m_id, cycles, grant_cycles);

  //在V100配置中，grant_cycles_count始终等于1。
  if (active && grant_cycles_count == 1)
    grant_cycles_count = grant_cycles;
  else if (active)
    grant_cycles_count--;

  if (verbose) {
    printf("%d : cycle %llu : conflicts = %d\n", m_id, cycles, conflict_sub);
    printf("%d : cycle %llu : passing reqs = %d\n", m_id, cycles, reqs);
  }

  // collect some stats about buffer util
  for (unsigned i = 0; i < total_nodes; ++i) {
    in_buffer_util += in_buffers[i].size();
    out_buffer_util += out_buffers[i].size();
  }

  //互连子网络执行路由的周期总数，无论这一拍有没有路由数据包。
  cycles++;
}

/*
当所有输入缓冲和输出缓冲都没有数据包时，认为当前子网络处于空闲状态，反之则是Busy状态。
*/
bool xbar_router::Busy() const {
  for (unsigned i = 0; i < total_nodes; ++i) {
    if (!in_buffers[i].empty()) return true;

    if (!out_buffers[i].empty()) return true;
  }
  return false;
}

////////////////////////////////////////////////////
/////////////LocalInterconnect/////////////////////

// assume all the packets are one flit
// A packet is decomposed into one or more flits. A flit, the smallest unit   
// on which flow control is performed, can advance once buffering in the next 
// switch is available to hold the flit.
#define LOCAL_INCT_FLIT_SIZE 40

/*
构造函数。
*/
LocalInterconnect* LocalInterconnect::New(
    const struct inct_config& m_localinct_config) {
  LocalInterconnect* icnt_interface = new LocalInterconnect(m_localinct_config);

  return icnt_interface;
}

/*
构造函数。
*/
LocalInterconnect::LocalInterconnect(
    const struct inct_config& m_localinct_config)
    : m_inct_config(m_localinct_config) {
  n_shader = 0;
  n_mem = 0;
  n_subnets = m_localinct_config.subnets;
}

LocalInterconnect::~LocalInterconnect() {
  for (unsigned i = 0; i < m_inct_config.subnets; ++i) {
    delete net[i];
  }
}

/*
创建互连网络。
*/
void LocalInterconnect::CreateInterconnect(unsigned m_n_shader,
                                           unsigned m_n_mem) {
  //SM的个数。
  n_shader = m_n_shader;
  //内存子分区的个数。
  n_mem = m_n_mem;
  //子网络个数。在V100中配置为2，0号自网络负责REQ_NET，1号子网络负责REPLY_NET。
  net.resize(n_subnets);
  //创建2个子网络，0号子网络负责REQ_NET，1号子网络负责REPLY_NET。
  for (unsigned i = 0; i < n_subnets; ++i) {
    net[i] = new xbar_router(i, static_cast<Interconnect_type>(i), m_n_shader,
                             m_n_mem, m_inct_config);
  }
}

void LocalInterconnect::Init() {
  // empty
  // there is nothing to do
}

/*
数据包压入互连网络输入缓冲区。
*/
void LocalInterconnect::Push(unsigned input_deviceID, unsigned output_deviceID,
                             void* data, unsigned int size) {
  unsigned subnet;
  //如果互连网络有多个子网络，则SM 0-n_shader 划分给subnet-0。
  if (n_subnets == 1) {
    subnet = 0;
  } else {
    //input_deviceID < n_shader说明是SM侧发来的REQ，应设置子网络号为0，因为0号
    //子网络负责REQ_NET。
    if (input_deviceID < n_shader) {
      subnet = 0;
    } else {
      subnet = 1;
    }
  }

  // it should have free buffer
  // assume all the packets have size of one
  // no flits are implemented
  assert(net[subnet]->Has_Buffer_In(input_deviceID, 1));

  //数据包压入子网络。
  net[subnet]->Push(input_deviceID, output_deviceID, data, size);
}

/*
数据包弹出互连网络输出缓冲区。
*/
void* LocalInterconnect::Pop(unsigned ouput_deviceID) {
  // 0-_n_shader-1 indicates reply(network 1), otherwise request(network 0)
  int subnet = 0;
  //ouput_deviceID < n_shader说明是要向SM侧发出REPLY，应设置子网络号为1，因为1号
  //子网络负责REPLY_NET。
  if (ouput_deviceID < n_shader) subnet = 1;
  //将数据包从子网络弹出。
  return net[subnet]->Pop(ouput_deviceID);
}

/*
互连网络执行路由一拍。
*/
void LocalInterconnect::Advance() {
  for (unsigned i = 0; i < n_subnets; ++i) {
    net[i]->Advance();
  }
}

/*
判断互连网络是否处于Busy状态。有任意一个子网络处于Busy状态便认为整个互连网络处于
Busy状态。
*/
bool LocalInterconnect::Busy() const {
  for (unsigned i = 0; i < n_subnets; ++i) {
    //有任意一个子网络处于Busy状态便认为整个互连网络处于Busy状态。
    if (net[i]->Busy()) return true;
  }
  return false;
}

/*
判断互连网络是否有空闲的输入缓冲可以容纳来自deviceID号设备新的数据包。
*/
bool LocalInterconnect::HasBuffer(unsigned deviceID, unsigned int size) const {
  bool has_buffer = false;
  //设备号 >= SM数量时，属于内存子分区节点，用REPLY_NET子网络。反之属于SM节点，
  //用REQ_NET子网络。
  if ((n_subnets > 1) && deviceID >= n_shader)  // deviceID is memory node
    has_buffer = net[REPLY_NET]->Has_Buffer_In(deviceID, 1, true);
  else
    has_buffer = net[REQ_NET]->Has_Buffer_In(deviceID, 1, true);

  return has_buffer;
}

void LocalInterconnect::DisplayStats() const {
  printf("Req_Network_injected_packets_num = %lld\n",
         net[REQ_NET]->packets_num);
  printf("Req_Network_cycles = %lld\n", net[REQ_NET]->cycles);
  printf("Req_Network_injected_packets_per_cycle = %12.4f \n",
         (float)(net[REQ_NET]->packets_num) / (net[REQ_NET]->cycles));
  printf("Req_Network_conflicts_per_cycle = %12.4f\n",
         (float)(net[REQ_NET]->conflicts) / (net[REQ_NET]->cycles));
  printf("Req_Network_conflicts_per_cycle_util = %12.4f\n",
         (float)(net[REQ_NET]->conflicts_util) / (net[REQ_NET]->cycles_util));
  printf("Req_Bank_Level_Parallism = %12.4f\n",
         (float)(net[REQ_NET]->reqs_util) / (net[REQ_NET]->cycles_util));
  printf("Req_Network_in_buffer_full_per_cycle = %12.4f\n",
         (float)(net[REQ_NET]->in_buffer_full) / (net[REQ_NET]->cycles));
  printf("Req_Network_in_buffer_avg_util = %12.4f\n",
         ((float)(net[REQ_NET]->in_buffer_util) / (net[REQ_NET]->cycles) /
          net[REQ_NET]->active_in_buffers));
  printf("Req_Network_out_buffer_full_per_cycle = %12.4f\n",
         (float)(net[REQ_NET]->out_buffer_full) / (net[REQ_NET]->cycles));
  printf("Req_Network_out_buffer_avg_util = %12.4f\n",
         ((float)(net[REQ_NET]->out_buffer_util) / (net[REQ_NET]->cycles) /
          net[REQ_NET]->active_out_buffers));

  printf("\n");
  printf("Reply_Network_injected_packets_num = %lld\n",
         net[REPLY_NET]->packets_num);
  printf("Reply_Network_cycles = %lld\n", net[REPLY_NET]->cycles);
  printf("Reply_Network_injected_packets_per_cycle =  %12.4f\n",
         (float)(net[REPLY_NET]->packets_num) / (net[REPLY_NET]->cycles));
  printf("Reply_Network_conflicts_per_cycle =  %12.4f\n",
         (float)(net[REPLY_NET]->conflicts) / (net[REPLY_NET]->cycles));
  printf(
      "Reply_Network_conflicts_per_cycle_util = %12.4f\n",
      (float)(net[REPLY_NET]->conflicts_util) / (net[REPLY_NET]->cycles_util));
  printf("Reply_Bank_Level_Parallism = %12.4f\n",
         (float)(net[REPLY_NET]->reqs_util) / (net[REPLY_NET]->cycles_util));
  printf("Reply_Network_in_buffer_full_per_cycle = %12.4f\n",
         (float)(net[REPLY_NET]->in_buffer_full) / (net[REPLY_NET]->cycles));
  printf("Reply_Network_in_buffer_avg_util = %12.4f\n",
         ((float)(net[REPLY_NET]->in_buffer_util) / (net[REPLY_NET]->cycles) /
          net[REPLY_NET]->active_in_buffers));
  printf("Reply_Network_out_buffer_full_per_cycle = %12.4f\n",
         (float)(net[REPLY_NET]->out_buffer_full) / (net[REPLY_NET]->cycles));
  printf("Reply_Network_out_buffer_avg_util = %12.4f\n",
         ((float)(net[REPLY_NET]->out_buffer_util) / (net[REPLY_NET]->cycles) /
          net[REPLY_NET]->active_out_buffers));
}

void LocalInterconnect::DisplayOverallStats() const {}

unsigned LocalInterconnect::GetFlitSize() const { return LOCAL_INCT_FLIT_SIZE; }

void LocalInterconnect::DisplayState(FILE* fp) const {
  fprintf(fp, "GPGPU-Sim uArch: ICNT:Display State: Under implementation\n");
}
