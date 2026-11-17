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


#include <inttypes.h>

bool VP_ENABLE = false;
bool VP_PERFECT = false;
uint64_t VP_TRACK = 0;
uint64_t WINDOW_SIZE = 1024; //old_value = 512;
uint64_t FETCH_WIDTH = 16;
uint64_t FETCH_NUM_BRANCH = 16;     // 0: unlimited; >0: finite
bool FETCH_STOP_AT_INDIRECT = true;
bool FETCH_STOP_AT_TAKEN = true;
bool FETCH_MODEL_ICACHE = true;

bool PERFECT_BRANCH_PRED = false;
bool PERFECT_INDIRECT_PRED = true;    // old_value = false
uint64_t PIPELINE_FILL_LATENCY = 10; // old_value =5;
uint64_t NUM_LDST_LANES = 8;
uint64_t NUM_ALU_LANES = 16;

bool PREFETCHER_ENABLE = true;
bool PERFECT_CACHE = false;
bool WRITE_ALLOCATE = true;

uint64_t IC_SIZE = (1 << 17);
uint64_t IC_ASSOC = 8;
uint64_t IC_BLOCKSIZE = 64;

uint64_t L1_SIZE = (1 << 17); // old_value = (1 << 16);
uint64_t L1_ASSOC = 8;
uint64_t L1_BLOCKSIZE = 64;
uint64_t L1_LATENCY = 3;

uint64_t L2_SIZE = (1 << 22); // old_value = (1 << 20);
uint64_t L2_ASSOC = 8;
uint64_t L2_BLOCKSIZE = 64;
uint64_t L2_LATENCY = 12;

uint64_t L3_SIZE = (1 << 25); // old_value = (1 << 23);
uint64_t L3_ASSOC = 16;
uint64_t L3_BLOCKSIZE = 128;
uint64_t L3_LATENCY = 50; // old_value = 60;

uint64_t MAIN_MEMORY_LATENCY = 150;

uint64_t DEFAULT_EXEC_LATENCY = 1;
uint64_t FP_EXEC_LATENCY = 3;
uint64_t SLOW_ALU_EXEC_LATENCY = 4;

uint64_t LOG_LEVEL = 0;
uint64_t LOG_START_CYCLE = 0;
uint64_t LOG_END_CYCLE = 0;

uint64_t DQ_LATENCY = 2;

uint64_t MISP_REDUCTION_PERC = 0;

uint64_t EPOCH_SIZE_INSTS = 1000000;
bool PRINT_PER_EPOCH_STATS = false;
