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


#include <math.h>
#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include "parameters.h"
#include "cache.h"


cache_t::cache_t(uint64_t size, uint64_t assoc, uint64_t blocksize, uint64_t latency, cache_t *next_level) {
   uint64_t num_sets;

   assert(IsPow2(blocksize));
   this->num_offset_bits = log2(blocksize);

   num_sets = size/(assoc*blocksize);
   assert(IsPow2(num_sets));
   this->num_index_bits = log2(num_sets);
   this->index_mask = (num_sets - 1);

   this->assoc = assoc;

   C = new block_t *[num_sets];
   for (uint64_t i = 0; i < num_sets; i++) {
      C[i] = new block_t[assoc];
      for (uint64_t j = 0; j < assoc; j++) {
         C[i][j].valid = false;
     C[i][j].lru = j;
      }
   }

   this->latency = latency;
   this->next_level = next_level;

   accesses = 0;
   misses = 0;
}

cache_t::~cache_t() {
}

bool cache_t::is_hit(uint64_t cycle, uint64_t addr) const {
   uint64_t tag = TAG(addr);
   uint64_t index = INDEX(addr);

   for (uint64_t way = 0; way < assoc; way++) {
      if (C[index][way].valid && (C[index][way].tag == tag)) {
         auto avail = ((C[index][way].timestamp > (cycle + latency)) ? C[index][way].timestamp : (cycle + latency));
         return (cycle + latency >= avail);
      }
   }

   return false;
}

uint64_t cache_t::access(uint64_t cycle, bool read, uint64_t addr, bool pf) {
   uint64_t avail;      // return value: cycle that requested block is available
   uint64_t tag = TAG(addr);
   uint64_t index = INDEX(addr);
   bool hit = false;
   uint64_t way;        // if hit, this is the corresponding way
   uint64_t max_lru_ctr = 0;    // for finding lru block
   uint64_t victim_way;     // if miss, this is the lru/victim way

   accesses+=!pf;
   pf_accesses += pf;

   for (way = 0; way < assoc; way++)
   {
      if (C[index][way].valid && (C[index][way].tag == tag))
      {
         hit = true;
         break;
      }
      else if (C[index][way].lru >= max_lru_ctr)
      {
         max_lru_ctr = C[index][way].lru;
         victim_way = way;
      }
   }

   if (hit) {   // hit
      // determine when the requested block will be available
      avail = ((C[index][way].timestamp > (cycle + latency)) ? C[index][way].timestamp : (cycle + latency));

      update_lru(index, way);   // make "way" the MRU way
   }
   else {   // miss
      misses+= !pf;
      pf_misses += pf;

      assert(max_lru_ctr == (assoc - 1));
      assert(victim_way < assoc);
      
      // TO DO: model writebacks (evictions of dirty blocks)

      // determine when the requested block will be available
      avail = (next_level ? next_level->access((cycle + latency), read, addr, pf) : (cycle + latency + MAIN_MEMORY_LATENCY));

      // replace the victim block with the requested block
      C[index][victim_way].valid = true;
      C[index][victim_way].tag = tag;
      C[index][victim_way].timestamp = avail;
      update_lru(index, victim_way);  // make "victim_way" the MRU way
   }

   return(avail);
}

void cache_t::update_lru(uint64_t index, uint64_t mru_way) {
   for (uint64_t way = 0; way < assoc; way++) {
      if (C[index][way].lru < C[index][mru_way].lru) {
         C[index][way].lru++;
         assert(C[index][way].lru < assoc);
      }
   }
   C[index][mru_way].lru = 0;
}

void cache_t::stats() {
   printf("\taccesses   = %lu\n", accesses);
   printf("\tmisses     = %lu\n", misses);
   printf("\tmiss ratio = %.2f%%\n", 100.0*((double)misses/(double)accesses));
   printf("\tpf accesses   = %lu\n", pf_accesses);
   printf("\tpf misses     = %lu\n", pf_misses);
   printf("\tpf miss ratio = %.2f%%\n", 100.0*((double)pf_misses/(double)pf_accesses));
}
