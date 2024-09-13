// Copyright (c) 2009-2011, Tor M. Aamodt, Inderpreet Singh
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

/*
�����ִ�ͳ�������Լ�⴫ͳCPU��ϵ�ṹ��ָ��֮�������ԣ��Ƿ��ƺͱ���վ������վ�����������������ԣ�
������Թ����߼�����Ҫ���������߼�����������������ǰ���ġ��Ƿ��ƿ������Ϊ֧��˳��ִ�л�����ִ�С�
֧������ִ�еļǷ��ƣ���CDC 6600��ʹ�õļǷ��ƣ�Ҳ�൱���ӡ���һ���棬���߳�˳��ִ�е�CPU�еļǷ�
�Ʒǳ��򵥣��ڼǷ������õ���λ����ʾÿһ���Ĵ�����ÿ��������д�뵽�üĴ�����ָ��ʱ���Ƿ����ж�Ӧ��
����λ���趨Ϊ1���κ���Ҫ��ȡ��д���ڼǷ�������������Ӧλ�ļĴ�����ָ���stall��ֱ��д��Ĵ�����
ָ������˸�λ������Է�ֹд�����д��дð�ա�����Ĵ����ļ���read������Ϊ˳����������CPU�����
�ĵ������������˳��ָ�������ʱ�����ּ򵥵ļǷ��ƿ��Է�ֹ����дð�ա����ǵ�������򵥵���ƣ�
��˽��������ٵ��������Դ��GPUʵ����˳��Ƿ��ơ���֧�ֶ��warpʱ��ʹ��˳��Ƿ���Ҳ����һЩ��ս��

�����򵥵�˳��Ƿ�����Ƶĵ�һ���������ִ�GPU�а����Ĵ����Ĵ���������ÿ��warp���128���Ĵ�����ÿ
���ں����64��warp�����ÿ���ں��ܹ���Ҫ8192λ��ʵ�ּǷ��ơ�

�����򵥵�˳��Ƿ�����Ƶ���һ��ע�����ڣ�������������ָ������ڼǷ������ظ��������������ֱ������
��������ϵ����ǰָ�����д��Ĵ�����Ϊֹ�����ڵ��߳���ƣ��⼸��������������ԡ����ǣ���˳��
���Ķ��̴߳������У����Զ���̵߳�ָ��������ڵȴ�ǰ���ָ����ɡ����������Щָ�������Ƿ��ƣ�
����Ҫ����Ķ�ȡ�˿ڡ������GPU֧��ÿ��SIMT Core���64��warp��������ָ�����ж��4��������������£�
��������warp��ÿ�����ڼ��Ƿ��ƽ���Ҫ256����ȡ�˿ڣ��⽫�Ƿǳ�����ġ�һ���������������ÿ������
���Լ��Ƿ��Ƶ�warp�������������������˿��Կ������ڵ��ȵ�warp�����������ң����������ָ����û��
һ����������Եģ���ʹ���ܱ���������ָ��������������Եģ�Ҳ�����Է���ָ�

ʹ��[Brett W. Coon, Tracking Register Usage During Multithreaded Processing Using a Score-
bard having Separate Memory Regions and Storing Sequential Register Size Indicators]�����
��ƿ��Խ�����������⡣����ư���������λ���������һ���о��й���Ϊ��Լ3��4λ[Ahmad Lashgar, A 
case study in reverse engineering GPGPUs: Outstanding memory handling resources]��������ÿ
һ��Ŀ�ǽ����ѷ��䵫��δ���ִ�е�ָ��д��ļĴ����ı�ʶ�������ڳ����˳��Ƿ��ƶ��ԣ���ָ����
д��ʱ��������ʼǷ��ơ��෴��Coon������ƵļǷ��ƣ���������ָ����õ�ָ�������ʱ�����ߵ�ָ�
����д�뵽�Ĵ�������ʱ���Żᱻ���ʡ�

����ָ����ٻ��棨I-Cache������ȡָ��������ָ�������ʱ������Ӧwarp�ļǷ�����Ŀ���ָ���Դ
�Ĵ�����Ŀ�ļĴ������бȽϡ������̵ܶ�λ���������ڸ�warp���Ƿ����е�ÿ����Ŀһλ�����磬3��4λ����
����Ƿ����еĶ�Ӧ��Ŀ��ָ����κβ�����ƥ�䣬�����ö�Ӧλ��Ȼ�󣬽���λ������ָ��һ���Ƶ�ָ�
�����С�ֱ������λ���������ָ������ʸ�ָ����������Ƿ��䣬�����ͨ����������ÿ��λ���͵�NOR����
ȷ������ָ�����д��Ĵ����ļ�ʱ��ָ������е������λ��������������warp��������Ŀ�������ˣ�
������warp��ͣԤȡ�����߶�����ָ��ұ����ٴλ�ȡ������ִ�е�ָ��׼��д��Ĵ�����ʱ��������Ƿ���
�з����������Ŀ�����һ�����洢��ָ������е�����ͬһwarp���κ�ָ�����Ӧ������λ��

��Accel SIM�У��Ƿ����������б����ڱ����ѷ���ָ���Ŀ��Ĵ�������һ�� reg_table �������е�Ŀ��
�Ĵ������ڶ��� longopregs ֻ���ٴ洢�����ʵ�Ŀ�ļĴ�����һ������һ��warpָ���Ŀ��Ĵ����ͻᱣ
���ڼǷ����С�������regsiter����SIMT Core�ܵ���д�ؽ׶α��ͷš����ָ���Դ�Ĵ�����Ŀ��Ĵ�������
����Ӳ��warp�ļǷ����У����޷�����ָ�
*/

#include <stdio.h>
#include <stdlib.h>
#include <set>
#include <vector>
#include "assert.h"

#ifndef SCOREBOARD_H_
#define SCOREBOARD_H_

#include "../abstract_hardware_model.h"

class Scoreboard {
 public:
  //���캯����
  Scoreboard(unsigned sid, unsigned n_warps, class gpgpu_t *gpu);
  
  //����ָ��ʱ����Ŀ��Ĵ�������������ӦӲ��warp�ļǷ����С�
  void reserveRegisters(const warp_inst_t *inst);
  //��ָ�����д��ʱ����Ŀ��Ĵ��������ͷš�
  void releaseRegisters(const warp_inst_t *inst);
  //������Ŀ��Ĵ����ͷš�
  void releaseRegister(unsigned wid, unsigned regnum);
  
  //���ð�գ����ĳ��ָ��ʹ�õļĴ����Ƿ񱻱����ڼǷְ��У�����еĻ����Ƿ����� WAW �� RAW ð�ա�
  bool checkCollision(unsigned wid, const inst_t *inst) const;
  //���ؼǷ��Ƶ�reg_table���Ƿ��й����д�롣warp idָ���reg_tableΪ�յĻ�������û�й����д�룬��
  //��false��[�����д��]��ָwid�Ƿ����ѷ��䵫��δ��ɵ�ָ���Ŀ��Ĵ��������ڼǷ��ơ�
  bool pendingWrites(unsigned wid) const;
  //��ӡ�Ƿ��Ƶ����ݡ�
  void printContents() const;
  const bool islongop(unsigned warp_id, unsigned regnum);

 private:
  //������Ŀ��Ĵ�����������ӦӲ��warp�ļǷ����С�
  void reserveRegister(unsigned wid, unsigned regnum);
  //����SM��ID��
  int get_sid() const { return m_sid; }
  //SM��ID��
  unsigned m_sid;
  
  //��������������У�
  //    reg_table�����ѷ���ָ������δд�ص�����Ŀ��Ĵ�����
  //    longopregs�����ѷ�����ڴ����ָ������δд�ص�����Ŀ��Ĵ�����
  
  // keeps track of pending writes to registers
  // indexed by warp id, reg_id => pending write count
  //���ٳ��ô�ָ���������е�Ŀ��Ĵ���������ԼĴ�����д�룬�����ĳ������ָ��Ҫд��Ĵ��� r0���ڸ���
  //ָ���ǰ������Ҫ��Ŀ��Ĵ��� r0 ���뵽 reg_table �С�
  //����: warp id=>reg_id=>�����д����������ÿ��warp���Լ���һ�� std::vector reg_table������
  //��˵��ÿ��warp��һ���Ƿ��ơ�
  std::vector<std::set<unsigned> > reg_table;
  // Register that depend on a long operation (global, local or tex memory)
  //���ٴ洢�����ʵ�Ŀ�ļĴ���������ԼĴ�����д�룬�����ĳ���ô�ָ��Ҫд��Ĵ��� r1���ڸ���ָ���
  //ǰ������Ҫ��Ŀ��Ĵ��� r1 ���뵽 longopregs �С�
  //����: warp id=>reg_id=>�����д����������ÿ��warp���Լ���һ�� std::vector longopregs����
  //�仰˵��ÿ��warp��һ���Ƿ��ơ�
  std::vector<std::set<unsigned> > longopregs;

  class gpgpu_t *m_gpu;
};

#endif /* SCOREBOARD_H_ */
