#ifndef _S_BLOCK_INCLUDED
#define _S_BLOCK_INCLUDED
#include "common_sort.h"
#include "radix_configs.h"
#include "ska_sort.hpp"


class SimpleBlock{
 public:
  sizeT start;
  sizeT size;
  long counts[BUCKETS];
  long bucketEnds[BUCKETS];
  SimpleBlock() {

  }
  void init(sizeT offset, sizeT end){
    this->start = offset;
    this->size = end - offset;
    for(int i = 0; i < BUCKETS; i ++) {
      counts[i] = 0;
    }
  }
};

template <class E, class F>
  void sortSimpleBlock(E* A, SimpleBlock *block, F extract){

  struct metaData meta(extract._offset, true, false, block->counts);
  ska_sort(A+ block->start, A + block->start+block->size, extract._f, meta);
  block->bucketEnds[0] = block->start + block->counts[0];
  for(int i = 1; i < BUCKETS; i++) {
    block->bucketEnds[i] = block->counts[i] + block->bucketEnds[i-1];
  }
}



#endif
