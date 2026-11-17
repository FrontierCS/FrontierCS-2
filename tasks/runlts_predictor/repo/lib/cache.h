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


struct block_t {
    bool valid;
    //bool dirty;   // TO DO
    uint64_t tag;
    uint64_t timestamp;
    uint64_t lru;
};

#define IsPow2(x)   (((x) & (x-1)) == 0)

#define TAG(addr)   ((addr) >> (num_index_bits + num_offset_bits))
#define INDEX(addr) (((addr) >> num_offset_bits) & index_mask)

class cache_t {
private:
    block_t **C;
    uint64_t num_index_bits;
    uint64_t num_offset_bits;
    uint64_t index_mask;
    uint64_t assoc;

    // latency to search this cache for requested block
    uint64_t latency;

    // pointer to next cache level if applicable
    cache_t *next_level;

    // measurements
    uint64_t accesses;
    uint64_t pf_accesses;
    uint64_t misses;
    uint64_t pf_misses;

    void update_lru(uint64_t index, uint64_t mru_way);

public:
    cache_t(uint64_t size, uint64_t assoc, uint64_t blocksize, uint64_t latency, cache_t *next_level);
    ~cache_t();
    uint64_t access(uint64_t cycle, bool read, uint64_t addr, bool pf = false);
    bool is_hit(uint64_t cycle, uint64_t addr) const;
    void stats();
};
