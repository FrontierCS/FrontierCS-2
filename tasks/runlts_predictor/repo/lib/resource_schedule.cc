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
#include <assert.h>
#include "resource_schedule.h"

resource_schedule::resource_schedule(uint64_t width) {
   base_cycle = 0;
   this->width = width;
   depth = SCHED_DEPTH_INCREMENT;
   sched = new uint64_t[depth];
   for (uint64_t i = 0; i < depth; i++)
      sched[i] = 0;
}

resource_schedule::~resource_schedule() {
}

void resource_schedule::resize(uint64_t new_depth) {
   uint64_t old_depth;
   uint64_t *old;
   uint64_t increments;
   uint64_t i;

   old_depth = depth;
   old = sched;

   increments = (new_depth/SCHED_DEPTH_INCREMENT);
   if (new_depth % SCHED_DEPTH_INCREMENT)
      increments++;
   depth = increments*SCHED_DEPTH_INCREMENT;
   assert(depth >= new_depth);

   sched = new uint64_t[depth];
   for (i = 0; i < old_depth; i++)
      sched[i] = old[i];
   for (i = old_depth; i < depth; i++)
      sched[i] = 0;

   delete[] old;
}

uint64_t resource_schedule::schedule(uint64_t start_cycle, uint64_t max_delta) 
{
   assert(start_cycle >= base_cycle);

   uint64_t limit_cycle = max_delta == MAX_CYCLE ? MAX_CYCLE : start_cycle + max_delta;
   bool found = false;

   while (!found) {
      if ((start_cycle - base_cycle + 1) > depth)
         resize(start_cycle - base_cycle + 1);

      if (sched[MOD_S(start_cycle,depth)] < width) {
         found = true;
         sched[MOD_S(start_cycle,depth)]++;
      }
      else {
         start_cycle++;
         if(start_cycle > limit_cycle)
         {
            return MAX_CYCLE;
         }
      }
   }

   assert(found);
   return(start_cycle);
}

uint64_t resource_schedule::try_schedule(uint64_t try_cycle)
{
   // Calling this assumes all previous events to schedule have been scheduled.
   assert(try_cycle >= base_cycle);

   bool found = false;

   while (!found) {
      if ((try_cycle - base_cycle + 1) > depth)
         resize(try_cycle - base_cycle + 1);

      if (sched[MOD_S(try_cycle,depth)] < width) {
         found = true;
      }
      else {
         try_cycle++;
      }
   }

   assert(found);
   return(try_cycle);
}

void resource_schedule::advance_base_cycle(uint64_t new_base_cycle) {
   assert(new_base_cycle >= base_cycle);
   for (uint64_t i = base_cycle; i < new_base_cycle; i++)
      sched[MOD_S(i,depth)] = 0;
   base_cycle = new_base_cycle;
}

