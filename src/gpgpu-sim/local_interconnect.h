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
ICNT�����ࡣ
*/
struct inct_config {
  // config for local interconnect
  //in_buffers[deviceID]�����������������ݰ�������
  unsigned in_buffer_limit;
  //out_buffers[deviceID]�����������������ݰ�������
  unsigned out_buffer_limit;
  //�����������
  unsigned subnets;
  //icnt_arbiter_algo����V100������Ϊ1=iSLIP�㷨��
  Arbiteration_type arbiter_algo;
  //�Ƿ������ϸ��Ϣ��
  unsigned verbose;
  //
  unsigned grant_cycles;
};

/*
�������ࡣ
*/
class xbar_router {
 public:
  //Xbar·�����������繹�캯����
  xbar_router(unsigned router_id, enum Interconnect_type m_type,
              unsigned n_shader, unsigned n_mem,
              const struct inct_config& m_localinct_config);
  ~xbar_router();
  //�����ݰ����������硣
  void Push(unsigned input_deviceID, unsigned output_deviceID, void* data,
            unsigned int size);
  //�����ݰ��������絯����
  void* Pop(unsigned ouput_deviceID);
  //ִ��·��һ�ġ�
  void Advance();

  //���������뻺���������嶼û�����ݰ�ʱ����Ϊ��ǰ�����紦�ڿ���״̬����֮����Busy
  //״̬��
  bool Busy() const;
  //�жϵ�ǰ�������Ƿ����㹻�����뻺�����ܹ�����size��С�������ݰ���
  bool Has_Buffer_In(unsigned input_deviceID, unsigned size,
                     bool update_counter = false);
  //�жϵ�ǰ�������Ƿ����㹻������������ܹ�����size��С�������ݰ���
  bool Has_Buffer_Out(unsigned output_deviceID, unsigned size);

  // some stats
  //����������ִ��·�ɵ�����������������һ����û��·�����ݰ���
  unsigned long long cycles;
  //conflicts������������ִ���ڼ䣬���ݰ���Ŀ���豸���г�ͻ�Ĵ����������0�ź͵�1��
  //�豸�������ݰ����͵���25���豸����ô���ͻһ�Ρ�
  unsigned long long conflicts;
  //conflicts_util����[���������������buffer�������ݰ���������]�ڼ䣬������������
  //��Ч�����ڼ䣬���ݰ���Ŀ���豸���г�ͻ�Ĵ����������0�ź͵�1���豸�������ݰ�����
  //����25���豸����ô���ͻһ�Ρ�
  unsigned long long conflicts_util;
  //cycles_util�ǻ��������������buffer�������ݰ�������������������������Ч���õ���
  //��������������ͳ�ơ�
  unsigned long long cycles_util;
  //reqs_util����[���������������buffer�������ݰ���������]�ڼ䣬��������������Ч��
  //���ڼ䣬����������·�ɵ����ݰ���������
  unsigned long long reqs_util;
  //ĳһ���ڵ��������������˾�����һ�Σ����ͳ�Ƶ�������������ִ���ڼ䣬���������
  //���˵��ܴ�����ע�������ڵ���ͬһ�����������Ρ�
  unsigned long long out_buffer_full;
  unsigned long long out_buffer_util;
  //ĳһ���ڵ�����뻺�������˾�����һ�Σ����ͳ�Ƶ�������������ִ���ڼ䣬���뻺����
  //���˵��ܴ�����ע�������ڵ���ͬһ�����������Ρ�
  unsigned long long in_buffer_full;
  unsigned long long in_buffer_util;
  //��������ִ�������ڣ���������������ݰ����ܸ�����
  unsigned long long packets_num;

 private:
  //ִ��·��һ�ġ�
  void iSLIP_Advance();
  void RR_Advance();

  //���ݰ��ࡣ
  struct Packet {
    Packet(void* m_data, unsigned m_output_deviceID) {
      data = m_data;
      output_deviceID = m_output_deviceID;
    }
    //���ݡ�
    void* data;
    //������ĸ��豸ID��
    unsigned output_deviceID;
  };
  //���ݰ������뻺���������С������Ϊ�ڵ�����=SM������+�ڴ��ӷ����ĸ�����
  vector<queue<Packet> > in_buffers;
  //���ݰ�����������������С������Ϊ�ڵ�����=SM������+�ڴ��ӷ����ĸ�����
  vector<queue<Packet> > out_buffers;
  //SM���������ڴ��ӷ����ĸ������ڵ�������
  //total_nodes = _n_shader + _n_mem��
  unsigned _n_shader, _n_mem, total_nodes;
  //in_buffer_limit��in_buffers[deviceID]�����������������ݰ�������
  //out_buffer_limit��out_buffers[deviceID]�����������������ݰ�������
  unsigned in_buffer_limit, out_buffer_limit;
  //����iSLIP�㷨�ٲá�
  vector<unsigned> next_node;  // used for iSLIP arbit
  unsigned next_node_id;       // used for RR arbit
  //�������ID��
  unsigned m_id;
  //REQ_NET��REPLY_NET�������������V100������Ϊ2��0�������縺��REQ_NET��1������
  //�縺��REPLY_NET��
  enum Interconnect_type router_type;
  //�����REQ_NET��0�������磩�����ݰ���SMת�����ڴ��ӷ�������
  //    ��������뻺��������ΪSM������
  //    ������������������Ϊ�ڴ��ӷ���������
  //�����REPLY_NET��1�������磩�����ݰ����ڴ��ӷ���ת����SM����
  //    ��������뻺��������Ϊ�ڴ��ӷ���������
  //    ������������������ΪSM������
  unsigned active_in_buffers, active_out_buffers;
  //�ٲ����ͣ�icnt_arbiter_algo����V100������Ϊ1=iSLIP�㷨��
  Arbiteration_type arbit_type;
  unsigned verbose;

  unsigned grant_cycles;
  unsigned grant_cycles_count;

  friend class LocalInterconnect;
};

/*
�������硣
*/
class LocalInterconnect {
 public:
 //���캯����
  LocalInterconnect(const struct inct_config& m_localinct_config);
  ~LocalInterconnect();
  //���캯����
  static LocalInterconnect* New(const struct inct_config& m_inct_config);
  //�����������硣
  void CreateInterconnect(unsigned n_shader, unsigned n_mem);

  // node side functions
  void Init();
  //���ݰ�ѹ�뻥���������뻺������
  void Push(unsigned input_deviceID, unsigned output_deviceID, void* data,
            unsigned int size);
  //���ݰ������������������������
  void* Pop(unsigned ouput_deviceID);
  //��������ִ��·��һ�ġ�
  void Advance();
  //�жϻ��������Ƿ���Busy״̬��������һ�������紦��Busy״̬����Ϊ�����������紦��
  //Busy״̬��
  bool Busy() const;
  //�жϻ��������Ƿ��п��е����뻺�������������deviceID���豸�µ����ݰ���
  bool HasBuffer(unsigned deviceID, unsigned int size) const;
  void DisplayStats() const;
  void DisplayOverallStats() const;
  unsigned GetFlitSize() const;

  void DisplayState(FILE* fp) const;

 protected:
  //�����������á�
  const inct_config& m_inct_config;
  //SM�ڵ��������ڴ��ӷ����ڵ�������
  unsigned n_shader, n_mem;
  //������������
  unsigned n_subnets;
  //�洢�������������net[REQ_NET]��net[REPLY_NET]��
  vector<xbar_router*> net;
};

#endif
