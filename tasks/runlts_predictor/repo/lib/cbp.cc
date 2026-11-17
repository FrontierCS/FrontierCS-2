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


#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <assert.h>
#include <string.h>
#include "cbp.h"
#include "trace_reader.h"
#include "fifo.h"
#include "cache.h"
#include "bp.h"
#include "resource_schedule.h"
#include "uarchsim.h"
#include "parameters.h"

uarchsim_t *sim;

int parseargs(int argc, char ** argv) 
{
  int i = 1;

  // read optional flags
  while (i < argc)
  {
     if (!strcmp(argv[i], "-d"))
     {
        PERFECT_CACHE = true;
        i++;
     }
     else if (!strcmp(argv[i], "-b"))
     {
        PERFECT_BRANCH_PRED = true;
        i++;
     }
     //else if (!strcmp(argv[i], "-i"))
     //{
     //   PERFECT_INDIRECT_PRED = true;
     //   i++;
     //}
     else if (!strcmp(argv[i], "-P"))
     {
        PREFETCHER_ENABLE = true;
        i++;
     }
     //else if (!strcmp(argv[i], "-f"))
     //{
     //   i++;
     //   if (i < argc)
     //   {
     //      PIPELINE_FILL_LATENCY = atoi(argv[i]);
     //      i++;
     //   }
     //   else
     //   {
     //      printf("Usage: missing pipeline fill latency: -f <pipeline_fill_latency>.\n");
     //      exit(0);
     //   }
     //}
     else if (!strcmp(argv[i], "-M"))
     {
        i++;
        if (i < argc)
        {
           NUM_LDST_LANES = atoi(argv[i]);
           i++;
        }
        else
        {
           printf("Usage: missing # load/store lanes: -M <num_ldst_lanes>.\n");
           exit(0);
        }
     }
     else if (!strcmp(argv[i], "-A"))
     {
        i++;
        if (i < argc)
        {
           NUM_ALU_LANES = atoi(argv[i]);
           i++;
        }
        else
        {
           printf("Usage: missing # alu lanes: -A <num_alu_lanes>.\n");
           exit(0);
        }
     }
     //else if (!strcmp(argv[i], "-s")) {
     //   WRITE_ALLOCATE = true;
     // i++;
     //}
     else if (!strcmp(argv[i], "-F"))
     {
        i++;
        if (i < argc)
        {
           unsigned int temp1, temp2, temp3, temp4, temp5;
           if (sscanf(argv[i], "%u,%u,%u,%u,%u", &temp1, &temp2, &temp3, &temp4, &temp5) == 5)
           {
              FETCH_WIDTH = (uint64_t)temp1;
              FETCH_NUM_BRANCH = (uint64_t)temp2;
              FETCH_STOP_AT_INDIRECT = (temp3 ? true : false);
              FETCH_STOP_AT_TAKEN = (temp4 ? true : false);
              FETCH_MODEL_ICACHE = (temp5 ? true : false);
           }
           else
           {
              printf("Usage: missing one or more fetch bundle constraints: -F <fetch_width>,<fetch_num_branch>,<fetch_stop_at_indirect>,<fetch_stop_at_taken>,<fetch_model_icache>.\n");
              exit(0);
           }
           i++;
        }
        else
        {
           printf("Usage: missing one or more fetch bundle constraints: -F <fetch_width>,<fetch_num_branch>,<fetch_stop_at_indirect>,<fetch_stop_at_taken>,<fetch_model_icache>.\n");
           exit(0);
        }
     }
     else if (!strcmp(argv[i], "-I"))
     {
        i++;
        if (i < argc)
        {
           unsigned int temp1, temp2, temp3;
           if (sscanf(argv[i], "%u,%u,%u", &temp1, &temp2, &temp3) == 3)
           {
              IC_SIZE = (uint64_t)(1 << temp1);
              IC_ASSOC = (uint64_t)temp2;
              IC_BLOCKSIZE = (uint64_t)temp3;
           }
           else
           {
              printf("Usage: missing one or more I$ parameters: -I <log2_size>,<assoc>,<blocksize>.\n");
              exit(0);
           }
           i++;
        }
        else
        {
           printf("Usage: missing I$ parameters: -I <log2_size>,<assoc>,<blocksize>.\n");
           exit(0);
        }
     }
     else if (!strcmp(argv[i], "-D"))
     {
        i++;
        if (i < argc)
        {
           unsigned int temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9, temp10, temp11, temp12, temp13;
           if (sscanf(argv[i], "%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u",
                      &temp1, &temp2, &temp3, &temp4,
                      &temp5, &temp6, &temp7, &temp8,
                      &temp9, &temp10, &temp11, &temp12,
                      &temp13) == 13)
           {
              L1_SIZE = (uint64_t)(1 << temp1);
              L1_ASSOC = (uint64_t)temp2;
              L1_BLOCKSIZE = (uint64_t)temp3;
              L1_LATENCY = (uint64_t)temp4;

              L2_SIZE = (uint64_t)(1 << temp5);
              L2_ASSOC = (uint64_t)temp6;
              L2_BLOCKSIZE = (uint64_t)temp7;
              L2_LATENCY = (uint64_t)temp8;

              L3_SIZE = (uint64_t)(1 << temp9);
              L3_ASSOC = (uint64_t)temp10;
              L3_BLOCKSIZE = (uint64_t)temp11;
              L3_LATENCY = (uint64_t)temp12;

              MAIN_MEMORY_LATENCY = (uint64_t)temp13;
           }
           else
           {
              printf("Usage: missing one or more L1$, L2$, and L3$ parameters: -D <log2_L1_size>,<L1_assoc>,<L1_blocksize>,<L1_latency>,<log2_L2_size>,<L2_assoc>,<L2_blocksize>,<L2_latency>,<log2_L3_size>,<L3_assoc>,<L3_blocksize>,<L3_latency>,<main_memory_latency>.\n");
              exit(0);
           }
           i++;
        }
        else
        {
           printf("Usage: missing L1$, L2$, and L3$ parameters: -D <log2_L1_size>,<L1_assoc>,<L1_blocksize>,<L1_latency>,<log2_L2_size>,<L2_assoc>,<L2_blocksize>,<L2_latency>,<log2_L3_size>,<L3_assoc>,<L3_blocksize>,<L3_latency>,<main_memory_latency>.\n");
           exit(0);
        }
     }
     else if (!strcmp(argv[i], "-E"))
     {
        i++;
        PRINT_PER_EPOCH_STATS = true;
        if (i < argc)
        {
           uint64_t epoch_size_insts;
           if (sscanf(argv[i], "%lu", &epoch_size_insts) == 1)
           {
              EPOCH_SIZE_INSTS = epoch_size_insts;
           }
           else
           {
              printf("Usage: missing epoch size: -E <epoch_size>\n");
              exit(0);
           }
           i++;
        }
        else
        {
           printf("Usage: missing epoch size: -E <epoch_size>\n");
           exit(0);
        }
     }
     else if (!strcmp(argv[i], "-w"))
     {
        i++;
        if (i < argc)
        {
           WINDOW_SIZE = atoi(argv[i]);
           i++;
        }
        else
        {
           printf("Usage: missing window size: -w <window_size>.\n");
           exit(0);
        }
     }
     else
     {
        break;
     }
  }

  if (i < argc) {
     return(i);
  }
  else {
     //printf("usage:\t%s\n[optional: -d to enable perfect data cache]\n\t[optional: -b to enable perfect branch prediction (all branch types)]\n\t[optional: -i to enable perfect indirect-branch prediction]\n\t[optional: -P to enable stride prefetcher in L1D]\n\t[optional: -f <pipeline_fill_latency>]\n\t[optional: -M <num_ldst_lanes>\n\t[optional: -A <num_alu_lanes>\n\t[optional: -F <fetch_width>,<fetch_num_branch>,<fetch_stop_at_indirect>,<fetch_stop_at_taken>,<fetch_model_icache>]\n\t[optional: -I <log2_ic_size>,<ic_assoc>,<ic_blocksize>]\n\t[optional: -D <log2_L1_size>,<L1_assoc>,<L1_blocksize>,<L1_latency>,<log2_L2_size>,<L2_assoc>,<L2_blocksize>,<L2_latency>,<log2_L3_size>,<L3_assoc>,<L3_blocksize>,<L3_latency>,<main_memory_latency>]\n\t[optional: -w <window_size>]\n\t[REQUIRED: .gz trace file]\n\t[optional: contestant's arguments]\n", argv[0]);
     printf("usage:\t%s\n"
             //"\t[optional: -v to enable value prediction]\n", 
             //"\t[optional: -p to enable perfect value prediction (if -v also specified)]\n",
             "\t[optional: -d to enable perfect data cache]\n"
             "\t[optional: -b to enable perfect branch prediction (all branch types)]\n"
             // "\t[optional: -i to enable perfect indirect-branch prediction]\n"
             "\t[optional: -P to enable stride prefetcher in L1D]\n"
             // "\t[optional: -f <pipeline_fill_latency>]\n"
             "\t[optional: -M <num_ldst_lanes>\n"
             "\t[optional: -A <num_alu_lanes>\n"
             "\t[optional: -F <fetch_width>,<fetch_num_branch>,<fetch_stop_at_indirect>,<fetch_stop_at_taken>,<fetch_model_icache>]\n"
             "\t[optional: -I <log2_ic_size>,<ic_assoc>,<ic_blocksize>]\n"
             "\t[optional: -D <log2_L1_size>,<L1_assoc>,<L1_blocksize>,<L1_latency>,<log2_L2_size>,<L2_assoc>,<L2_blocksize>,<L2_latency>,<log2_L3_size>,<L3_assoc>,<L3_blocksize>,<L3_latency>,<main_memory_latency>]\n"
             "\t[optional: -w <window_size>]\n"
             "\t[optional: -E <epoch_size_insts> to enable dumping per-epoch conditional branch info\n"
             "\t[REQUIRED: .gz trace file]\n", argv[0]);
     exit(0);
  }
}

int main(int argc, char ** argv)
{
  int i = parseargs(argc, argv);
  TraceReader reader(argv[i]);

  // Need to create simulator after parsing arguments (for global parameters).
  sim = new uarchsim_t;
 
  // Get to next (optional) argument after trace filename.
  i++;
  //if (i < argc)
  //   beginPredictor((argc - i), &(argv[i]));
  //else
  //   beginPredictor(0, (char **)NULL);

  //if (i < argc)
  //   beginCondDirPredictor((argc - i), &(argv[i]));
  //else
  //   beginCondDirPredictor(0, (char **)NULL);
  beginCondDirPredictor();

  db_t *inst = reader.get_inst(); 

  //bool dump_activity = true;
  //uint64_t current_fetch_cycle = 0;
  while (inst != nullptr) 
  {
      //const bool logging_activated = (LOG_LEVEL != 0) && (current_fetch_cycle>= LOG_START_CYCLE) && (current_fetch_cycle<=LOG_END_CYCLE);
      //if(logging_activated && dump_activity)
      //{
      //    std::cout<<"======================================================= Begin "<<current_fetch_cycle<<"=======================================================\n";
      //    dump_activity = false;
      //}

      sim->step(inst);

      //const uint64_t next_fetch_cycle = sim->get_current_fetch_cycle();
      //if(logging_activated && next_fetch_cycle != current_fetch_cycle)
      //{
      //    dump_activity = true;
      //    std::cout<<"======================================================= End "<<current_fetch_cycle<<"->"<<next_fetch_cycle<<"=======================================================\n";
      //}
      //current_fetch_cycle = next_fetch_cycle;
      delete inst;
      inst = reader.get_inst();
  }

  endPredictor();
  endCondDirPredictor();
  sim->output();
}
