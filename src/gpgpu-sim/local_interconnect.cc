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
Xbar·�����������繹�캯����
*/
xbar_router::xbar_router(unsigned router_id, enum Interconnect_type m_type,
                         unsigned n_shader, unsigned n_mem,
                         const struct inct_config& m_localinct_config) {
  //·����ID��
  m_id = router_id;
  //REQ_NET��REPLY_NET�������������V100������Ϊ2��0�������縺��REQ_NET��1����
  //���縺��REPLY_NET��
  router_type = m_type;
  //�ڴ��ӷ����ĸ�����
  _n_mem = n_mem;
  //SM��������
  _n_shader = n_shader;
  //�ܽڵ���=SM������+�ڴ��ӷ����ĸ�����
  total_nodes = n_shader + n_mem;
  //�Ƿ��ӡ��ϸ��Ϣ�Ŀ���ѡ�
  verbose = m_localinct_config.verbose;
  grant_cycles = m_localinct_config.grant_cycles;
  grant_cycles_count = m_localinct_config.grant_cycles;
  in_buffers.resize(total_nodes);
  out_buffers.resize(total_nodes);
  next_node.resize(total_nodes, 0);
  //in_buffers[deviceID]�����������������ݰ�������
  in_buffer_limit = m_localinct_config.in_buffer_limit;
  //out_buffers[deviceID]�����������������ݰ�������
  out_buffer_limit = m_localinct_config.out_buffer_limit;
  //�ٲ����ͣ�icnt_arbiter_algo����V100������Ϊ1=iSLIP�㷨��
  arbit_type = m_localinct_config.arbiter_algo;
  next_node_id = 0;
  if (m_type == REQ_NET) {
    //�����REQ_NET��0�������磩�����ݰ���SMת�����ڴ��ӷ�������
    //    ��������뻺��������ΪSM������
    //    ������������������Ϊ�ڴ��ӷ���������
    active_in_buffers = n_shader;
    active_out_buffers = n_mem;
  } else if (m_type == REPLY_NET) {
    //�����REPLY_NET��1�������磩�����ݰ����ڴ��ӷ���ת����SM����
    //    ��������뻺��������Ϊ�ڴ��ӷ���������
    //    ������������������ΪSM������
    active_in_buffers = n_mem;
    active_out_buffers = n_shader;
  }

  //����������ִ��·�ɵ�����������������һ����û��·�����ݰ���
  cycles = 0;
  //conflicts������������ִ���ڼ䣬���ݰ���Ŀ���豸���г�ͻ�Ĵ����������0�ź͵�1����
  //���������ݰ����͵���25���豸����ô���ͻһ�Ρ�
  conflicts = 0;
  //ĳһ���ڵ��������������˾�����һ�Σ����ͳ�Ƶ�������������ִ���ڼ䣬�����������
  //�˵��ܴ�����ע�������ڵ���ͬһ�����������Ρ�
  out_buffer_full = 0;
  //ĳһ���ڵ�����뻺�������˾�����һ�Σ����ͳ�Ƶ�������������ִ���ڼ䣬���뻺������
  //�˵��ܴ�����ע�������ڵ���ͬһ�����������Ρ�
  in_buffer_full = 0;
  out_buffer_util = 0;
  in_buffer_util = 0;
  //��������ִ�������ڣ���������������ݰ����ܸ�����
  packets_num = 0;
  //conflicts_util����[���������������buffer�������ݰ���������]�ڼ䣬��������������
  //Ч�����ڼ䣬���ݰ���Ŀ���豸���г�ͻ�Ĵ����������0�ź͵�1���豸�������ݰ����͵���
  //25���豸����ô���ͻһ�Ρ�
  conflicts_util = 0;
  //cycles_util�ǻ��������������buffer�������ݰ�������������������������Ч���õ�����
  //������������ͳ�ơ�
  cycles_util = 0;
  //reqs_util����[���������������buffer�������ݰ���������]�ڼ䣬��������������Ч����
  //�ڼ䣬����������·�ɵ����ݰ���������
  reqs_util = 0;
}

xbar_router::~xbar_router() {}

/*
�����ݰ����������硣
*/
void xbar_router::Push(unsigned input_deviceID, unsigned output_deviceID,
                       void* data, unsigned int size) {
  assert(input_deviceID < total_nodes);
  in_buffers[input_deviceID].push(Packet(data, output_deviceID));
  //��������ִ�������ڣ���������������ݰ����ܸ�����
  packets_num++;
}

/*
�����ݰ��������絯����
*/
void* xbar_router::Pop(unsigned ouput_deviceID) {
  assert(ouput_deviceID < total_nodes);
  void* data = NULL;
  //�����������������Ϊ��ʱ������front���ݰ���
  if (!out_buffers[ouput_deviceID].empty()) {
    data = out_buffers[ouput_deviceID].front().data;
    out_buffers[ouput_deviceID].pop();
  }

  return data;
}

/*
�жϵ�ǰ�������Ƿ����㹻�����뻺�����ܹ�����size��С�������ݰ���
*/
bool xbar_router::Has_Buffer_In(unsigned input_deviceID, unsigned size,
                                bool update_counter) {
  assert(input_deviceID < total_nodes);

  //in_buffers[input_deviceID].size()��С�ǵ�ǰ�Ѿ������뻺����������ݰ�������
  //��������� + size > ���뻺�����Ĵ�С���ƣ�������������µ�size��С�����ݰ���
  bool has_buffer =
      (in_buffers[input_deviceID].size() + size <= in_buffer_limit);
  //ĳһ���ڵ�����뻺�������˾�����һ�Σ����ͳ�Ƶ�������������ִ���ڼ䣬���뻺��
  //�����˵��ܴ�����ע�������ڵ���ͬһ�����������Ρ�
  if (update_counter && !has_buffer) in_buffer_full++;

  return has_buffer;
}

/*
�жϵ�ǰ�������Ƿ����㹻������������ܹ�����size��С�������ݰ���
*/
bool xbar_router::Has_Buffer_Out(unsigned output_deviceID, unsigned size) {
  //out_buffers[output_deviceID].size()��С�ǵ�ǰ�Ѿ������������������ݰ�������
  //��������� + size > ����������Ĵ�С���ƣ�������������µ�size��С�����ݰ���
  return (out_buffers[output_deviceID].size() + size <= out_buffer_limit);
}

/*
ִ��·��һ�ġ�
*/
void xbar_router::Advance() {
  //�ٲ����ͣ�icnt_arbiter_algo����V100������Ϊ1=iSLIP�㷨��
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
ִ��·��һ�ġ�
*/
void xbar_router::iSLIP_Advance() {
  vector<unsigned> node_tmp;
  bool active = false;

  unsigned conflict_sub = 0;
  //reqs�ǵ�ǰ�ģ�����������·�ɵ����ݰ���������
  unsigned reqs = 0;

  // calcaulte how many conflicts are there for stats
  //�����Ǳ������нڵ㣬�����ǵ�����buffer�У��Ƿ�����ͬ�����Ŀ���豸output_deviceID��
  //���������˵���г�ͻ��
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

  //conflicts������������ִ���ڼ䣬���ݰ���Ŀ���豸���г�ͻ�Ĵ����������0�ź͵�1���豸
  //�������ݰ����͵���25���豸����ô���ͻһ�Ρ�
  conflicts += conflict_sub;
  if (active) {
    //conflicts_util����[���������������buffer�������ݰ���������]�ڼ䣬��������������
    //Ч�����ڼ䣬���ݰ���Ŀ���豸���г�ͻ�Ĵ����������0�ź͵�1���豸�������ݰ����͵���
    //25���豸����ô���ͻһ�Ρ�
    conflicts_util += conflict_sub;
    //cycles_util�ǻ��������������buffer�������ݰ�������������������������Ч���õ�����
    //������������ͳ�ơ�
    cycles_util++;
  }
  // do iSLIP
  //����������нڵ㣬Ϊ��Щ���нڵ�����������ѡ��Ӧ·�ɵ����ݰ���
  for (unsigned i = 0; i < total_nodes; ++i) {
    //�����i�Žڵ����������������Խ����µ����ݰ���
    if (Has_Buffer_Out(i, 1)) {
      //�����нڵ����������Щ�ڵ�����Щ�ڵ�����뻺����������Ҫ·�ɵ���i�Žڵ�����ݰ���
      for (unsigned j = 0; j < total_nodes; ++j) {
        //���������̵��Ȳ��ԡ�next_node[i]�ڵ�i���������������һ�������ݰ�ʱ��������ת
        //һ�Ρ�
        unsigned node_id = (j + next_node[i]) % total_nodes;
        //�����жϵ�(j + next_node[i])% total_nodes���ڵ�����뻺�����Ƿ���Ŀ�Ľڵ�Ϊ
        //i�����ݰ�������еĻ�����������뻺���ﵯ������ѹ���i�Žڵ�������������
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
            //reqs�ǵ�ǰ�ģ�����������·�ɵ����ݰ���������
            reqs++;
            break;
          }
        }
      }
    } else
      //ĳһ���ڵ��������������˾�����һ�Σ����ͳ�Ƶ�������������ִ���ڼ䣬�������
      //�����˵��ܴ�����ע�������ڵ���ͬһ�����������Ρ�
      out_buffer_full++;
  }

  if (active) {
    //reqs�ǵ�ǰ�ģ�����������·�ɵ����ݰ���������
    //reqs_util����[���������������buffer�������ݰ���������]�ڼ䣬��������������Ч��
    //���ڼ䣬����������·�ɵ����ݰ���������
    reqs_util += reqs;
  }

  if (verbose)
    printf("%d : cycle %llu : grant_cycles = %d\n", m_id, cycles, grant_cycles);

  //��V100�����У�grant_cycles_countʼ�յ���1��
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

  //����������ִ��·�ɵ�����������������һ����û��·�����ݰ���
  cycles++;
}

/*
���������뻺���������嶼û�����ݰ�ʱ����Ϊ��ǰ�����紦�ڿ���״̬����֮����Busy״̬��
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
���캯����
*/
LocalInterconnect* LocalInterconnect::New(
    const struct inct_config& m_localinct_config) {
  LocalInterconnect* icnt_interface = new LocalInterconnect(m_localinct_config);

  return icnt_interface;
}

/*
���캯����
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
�����������硣
*/
void LocalInterconnect::CreateInterconnect(unsigned m_n_shader,
                                           unsigned m_n_mem) {
  //SM�ĸ�����
  n_shader = m_n_shader;
  //�ڴ��ӷ����ĸ�����
  n_mem = m_n_mem;
  //�������������V100������Ϊ2��0�������縺��REQ_NET��1�������縺��REPLY_NET��
  net.resize(n_subnets);
  //����2�������磬0�������縺��REQ_NET��1�������縺��REPLY_NET��
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
���ݰ�ѹ�뻥���������뻺������
*/
void LocalInterconnect::Push(unsigned input_deviceID, unsigned output_deviceID,
                             void* data, unsigned int size) {
  unsigned subnet;
  //������������ж�������磬��SM 0-n_shader ���ָ�subnet-0��
  if (n_subnets == 1) {
    subnet = 0;
  } else {
    //input_deviceID < n_shader˵����SM�෢����REQ��Ӧ�����������Ϊ0����Ϊ0��
    //�����縺��REQ_NET��
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

  //���ݰ�ѹ�������硣
  net[subnet]->Push(input_deviceID, output_deviceID, data, size);
}

/*
���ݰ������������������������
*/
void* LocalInterconnect::Pop(unsigned ouput_deviceID) {
  // 0-_n_shader-1 indicates reply(network 1), otherwise request(network 0)
  int subnet = 0;
  //ouput_deviceID < n_shader˵����Ҫ��SM�෢��REPLY��Ӧ�����������Ϊ1����Ϊ1��
  //�����縺��REPLY_NET��
  if (ouput_deviceID < n_shader) subnet = 1;
  //�����ݰ��������絯����
  return net[subnet]->Pop(ouput_deviceID);
}

/*
��������ִ��·��һ�ġ�
*/
void LocalInterconnect::Advance() {
  for (unsigned i = 0; i < n_subnets; ++i) {
    net[i]->Advance();
  }
}

/*
�жϻ��������Ƿ���Busy״̬��������һ�������紦��Busy״̬����Ϊ�����������紦��
Busy״̬��
*/
bool LocalInterconnect::Busy() const {
  for (unsigned i = 0; i < n_subnets; ++i) {
    //������һ�������紦��Busy״̬����Ϊ�����������紦��Busy״̬��
    if (net[i]->Busy()) return true;
  }
  return false;
}

/*
�жϻ��������Ƿ��п��е����뻺�������������deviceID���豸�µ����ݰ���
*/
bool LocalInterconnect::HasBuffer(unsigned deviceID, unsigned int size) const {
  bool has_buffer = false;
  //�豸�� >= SM����ʱ�������ڴ��ӷ����ڵ㣬��REPLY_NET�����硣��֮����SM�ڵ㣬
  //��REQ_NET�����硣
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
