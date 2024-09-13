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
有两种传统方法可以检测传统CPU体系结构中指令之间的相关性：记分牌和保留站。保留站用于消除名称依赖性，
并引入对关联逻辑的需要，而关联逻辑在面积和能量方面是昂贵的。记分牌可以设计为支持顺序执行或乱序执行。
支持乱序执行的记分牌（如CDC 6600中使用的记分牌）也相当复杂。另一方面，单线程顺序执行的CPU中的记分
牌非常简单：在记分牌中用单个位来表示每一个寄存器，每当发出将写入到该寄存器的指令时，记分牌中对应的
单个位被设定为1。任何想要读取或写入在记分牌中设置了相应位的寄存器的指令都会stall，直到写入寄存器的
指令清除了该位。这可以防止写后读和写后写冒险。如果寄存器文件的read被限制为顺序发生，这是CPU设计中
的典型情况，则与顺序指令发射相结合时，这种简单的记分牌可以防止读后写冒险。考虑到这是最简单的设计，
因此将消耗最少的面积和能源，GPU实现了顺序记分牌。在支持多个warp时，使用顺序记分牌也存在一些挑战。

上述简单的顺序记分牌设计的第一个问题是现代GPU中包含的大量寄存器。由于每个warp最多128个寄存器，每
个内核最多64个warp，因此每个内核总共需要8192位来实现记分牌。

上述简单的顺序记分牌设计的另一关注点在于，遇到有依赖的指令必须在记分牌中重复查找其操作数，直到与其
有依赖关系的先前指令将其结果写入寄存器堆为止。对于单线程设计，这几乎不会带来复杂性。但是，在顺序发
出的多线程处理器中，来自多个线程的指令可能正在等待前面的指令完成。如果所有这些指令必须检测记分牌，
则需要额外的读取端口。最近的GPU支持每个SIMT Core多达64个warp，并且在指令中有多达4个操作数的情况下，
允许所有warp在每个周期检测记分牌将需要256个读取端口，这将是非常昂贵的。一种替代方案是限制每个周期
可以检测记分牌的warp的数量，但是这限制了可以考虑用于调度的warp的数量。而且，如果所检查的指令中没有
一个是无相关性的，则即使不能被检查的其它指令碰巧是无相关性的，也不可以发出指令。

使用[Brett W. Coon, Tracking Register Usage During Multithreaded Processing Using a Score-
bard having Separate Memory Regions and Storing Sequential Register Size Indicators]提出的
设计可以解决这两个问题。该设计包含少量的位（在最近的一项研究中估计为大约3或4位[Ahmad Lashgar, A 
case study in reverse engineering GPGPUs: Outstanding memory handling resources]），其中每
一条目是将由已发射但尚未完成执行的指令写入的寄存器的标识符。对于常规的顺序记分牌而言，当指令发射和
写回时，都会访问记分牌。相反，Coon等人设计的记分牌，仅仅当将指令放置到指令缓冲器中时，或者当指令将
其结果写入到寄存器堆中时，才会被访问。

当从指令高速缓存（I-Cache）中提取指令并将其放入指令缓冲器中时，将对应warp的记分牌条目与该指令的源
寄存器和目的寄存器进行比较。这仅需很短的位向量，对于该warp，记分牌中的每个条目一位（例如，3或4位）。
如果记分牌中的对应条目与指令的任何操作数匹配，则设置对应位。然后，将该位向量与指令一起复制到指令缓
冲区中。直到所有位都被清除，指令才有资格被指令调度器考虑发射，这可以通过将向量的每个位馈送到NOR门来
确定。当指令将其结果写入寄存器文件时，指令缓冲区中的相关性位被清除。如果给定warp的所有条目都用完了，
则所有warp暂停预取，或者丢弃该指令并且必须再次获取。当已执行的指令准备写入寄存器堆时，它清除记分牌
中分配给它的条目，并且还清除存储在指令缓冲器中的来自同一warp的任何指令的相应依赖性位。

在Accel SIM中，记分牌有两个列表，用于保留已发射指令的目标寄存器。第一个 reg_table 跟踪所有的目标
寄存器。第二个 longopregs 只跟踪存储器访问的目的寄存器。一旦发出一条warp指令，其目标寄存器就会保
留在记分牌中。保留的regsiter将在SIMT Core管道的写回阶段被释放。如果指令的源寄存器或目标寄存器保留
在其硬件warp的记分牌中，则无法发出指令。
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
  //构造函数。
  Scoreboard(unsigned sid, unsigned n_warps, class gpgpu_t *gpu);
  
  //发射指令时，其目标寄存器将保留在相应硬件warp的记分牌中。
  void reserveRegisters(const warp_inst_t *inst);
  //当指令完成写回时，其目标寄存器将被释放。
  void releaseRegisters(const warp_inst_t *inst);
  //将单个目标寄存器释放。
  void releaseRegister(unsigned wid, unsigned regnum);
  
  //检测冒险，检测某个指令使用的寄存器是否被保留在记分板中，如果有的话就是发生了 WAW 或 RAW 冒险。
  bool checkCollision(unsigned wid, const inst_t *inst) const;
  //返回记分牌的reg_table中是否有挂起的写入。warp id指向的reg_table为空的话，代表没有挂起的写入，返
  //回false。[挂起的写入]是指wid是否有已发射但尚未完成的指令，将目标寄存器保留在记分牌。
  bool pendingWrites(unsigned wid) const;
  //打印记分牌的内容。
  void printContents() const;
  const bool islongop(unsigned warp_id, unsigned regnum);

 private:
  //将单个目标寄存器保留在相应硬件warp的记分牌中。
  void reserveRegister(unsigned wid, unsigned regnum);
  //返回SM的ID。
  int get_sid() const { return m_sid; }
  //SM的ID。
  unsigned m_sid;
  
  //下面的声明代码中：
  //    reg_table保留已发射指令中尚未写回的所有目标寄存器。
  //    longopregs保留已发射的内存访问指令中尚未写回的所有目标寄存器。
  
  // keeps track of pending writes to registers
  // indexed by warp id, reg_id => pending write count
  //跟踪除访存指令以外所有的目标寄存器。挂起对寄存器的写入，即如果某条计算指令要写入寄存器 r0，在该条
  //指令发射前，就需要将目标寄存器 r0 加入到 reg_table 中。
  //索引: warp id=>reg_id=>挂起的写入计数。这里，每个warp有自己的一个 std::vector reg_table。换句
  //话说，每个warp有一个记分牌。
  std::vector<std::set<unsigned> > reg_table;
  // Register that depend on a long operation (global, local or tex memory)
  //跟踪存储器访问的目的寄存器。挂起对寄存器的写入，即如果某条访存指令要写入寄存器 r1，在该条指令发射
  //前，就需要将目标寄存器 r1 加入到 longopregs 中。
  //索引: warp id=>reg_id=>挂起的写入计数。这里，每个warp有自己的一个 std::vector longopregs。换
  //句话说，每个warp有一个记分牌。
  std::vector<std::set<unsigned> > longopregs;

  class gpgpu_t *m_gpu;
};

#endif /* SCOREBOARD_H_ */
