/*

Copyright (c) 2019, North Carolina State University
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. The names “North Carolina State University”, “NCSU” and any trade-name, personal name,
trademark, trade device, service mark, symbol, image, icon, or any abbreviation, contraction or
simulation thereof owned by North Carolina State University must not be used to endorse or promote products derived from this software without prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

// Author: Eric Rotenberg (ericro@ncsu.edu)

#ifndef _PARAMETERS_H_
#define _PARAMETERS_H_

enum class VPTracks
{
    ALL  = 0,
    LoadsOnly,
    LoadsOnlyHitMiss,
    NumTracks
};

extern bool VP_ENABLE;
extern bool VP_PERFECT;
extern uint64_t VP_TRACK;
extern uint64_t WINDOW_SIZE;
extern uint64_t FETCH_WIDTH;
extern uint64_t FETCH_NUM_BRANCH;
extern bool FETCH_STOP_AT_INDIRECT;
extern bool FETCH_STOP_AT_TAKEN;
extern bool FETCH_MODEL_ICACHE;

extern bool PERFECT_BRANCH_PRED;
extern bool PERFECT_INDIRECT_PRED;
extern uint64_t PIPELINE_FILL_LATENCY;
extern uint64_t NUM_LDST_LANES;
extern uint64_t NUM_ALU_LANES;

extern bool PREFETCHER_ENABLE;
extern bool PERFECT_CACHE;
extern bool WRITE_ALLOCATE;

extern uint64_t IC_SIZE;
extern uint64_t IC_ASSOC;
extern uint64_t IC_BLOCKSIZE;

extern uint64_t L1_SIZE;
extern uint64_t L1_ASSOC;
extern uint64_t L1_BLOCKSIZE;
extern uint64_t L1_LATENCY;

extern uint64_t L2_SIZE;
extern uint64_t L2_ASSOC;
extern uint64_t L2_BLOCKSIZE;
extern uint64_t L2_LATENCY;

extern uint64_t L3_SIZE;
extern uint64_t L3_ASSOC;
extern uint64_t L3_BLOCKSIZE;
extern uint64_t L3_LATENCY;

extern uint64_t MAIN_MEMORY_LATENCY;

extern uint64_t DEFAULT_EXEC_LATENCY;
extern uint64_t FP_EXEC_LATENCY;
extern uint64_t SLOW_ALU_EXEC_LATENCY;

extern uint64_t LOG_LEVEL;
extern uint64_t LOG_START_CYCLE;
extern uint64_t LOG_END_CYCLE;

extern uint64_t DQ_LATENCY;
extern uint64_t MISP_REDUCTION_PERC;

extern uint64_t EPOCH_SIZE_INSTS;
extern bool PRINT_PER_EPOCH_STATS;
#endif
