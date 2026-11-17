// This code is part of the Problem Based Benchmark Suite (PBBS)
// Copyright (c) 2010-2016 Guy Blelloch and Harsha Vardhan Simhadri and the PBBS team
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// This file is basically the cache-oblivious sorting algorithm from:
//
// Low depth cache-oblivious algorithms.
// Guy E. Blelloch, Phillip B. Gibbons and  Harsha Vardhan Simhadri.
// Proc. ACM symposium on Parallelism in algorithms and architectures (SPAA), 2010

#pragma once
#include <math.h>
#include <stdio.h>
#include <string.h> 
#include "utilities.h"
#include "sequence_ops.h"
#include "quicksort.h"
#include "transpose.h"

namespace pbbs {

  // the following parameters can be tuned
  constexpr const size_t QUICKSORT_THRESHOLD = 16384;
  constexpr const size_t OVER_SAMPLE = 8;

  // use different parameters for pointer and non-pointer types
  // and depending on size
  template<typename E> bool is_pointer(E x) {return 0;}
  template<typename E> bool is_pointer(E* x) {return 1;}

  // generates counts in Sc for the number of keys in Sa between consecutive
  // values of Sb
  // Sa and Sb must be sorted
  template<typename E, typename BinPred, typename s_size_t>
  void merge_seq (E* sA, E* sB, s_size_t* sC,
		  size_t lA, size_t lB, BinPred f) {
    if (lA==0 || lB==0) return;
    E *eA = sA+lA;
    E *eB = sB+lB;
    for (size_t i=0; i <= lB; i++) sC[i] = 0;
    while(1) {
      while (f(*sA, *sB)) {(*sC)++; if (++sA == eA) return;}
      sB++; sC++;
      if (sB == eB) break;
      if (!(f(*(sB-1),*sB))) {
	while (!f(*sB, *sA)) {(*sC)++; if (++sA == eA) return;}
	sB++; sC++;
	if (sB == eB) break;
      }
    } 
    *sC = eA-sA;
  }

  template<typename E, typename BinPred, typename s_size_t>
    void sample_sort (E* A, s_size_t n, BinPred f) {
    if (n < QUICKSORT_THRESHOLD) {
      quicksort(A, n, f);
    } else {
      size_t bucket_quotient = 5;
      size_t block_quotient = 5;
      if (is_pointer(A[0])) {
	bucket_quotient = 1;
	block_quotient = 2;
      } else if (sizeof(E) > 8) {
	bucket_quotient = block_quotient = 2;
      }
      size_t sqrt = (size_t) ceil(pow(n,0.5));
      size_t num_blocks = 1 << log2_up((sqrt/block_quotient) + 1);
      size_t block_size = ((n-1)/num_blocks) + 1;
      size_t num_buckets = (sqrt/bucket_quotient) + 1;
      size_t sample_set_size = num_buckets * OVER_SAMPLE;
      size_t m = num_blocks*num_buckets;
    
      E* sample_set = new_array<E>(sample_set_size);

      // generate "random" samples with oversampling
      parallel_for (size_t j=0; j< sample_set_size; ++j) 
	sample_set[j] = A[hash64(j)%n];

      // sort the samples
      quicksort(sample_set, sample_set_size, f);

      // subselect samples at even stride
      E* pivots = new_array<E>(num_buckets-1);
      parallel_for (size_t k=0; k < num_buckets-1; ++k)
	pivots[k] = sample_set[OVER_SAMPLE*k];

      delete_array(sample_set,sample_set_size);
    
      // sort each block and merge with samples to get counts for each bucket
      s_size_t *counts = new_array_no_init<s_size_t>(m,1);
      parallel_for_1 (size_t i = 0; i < num_blocks; ++i) {
	size_t offset = i * block_size;
	size_t size =  (i < num_blocks - 1) ? block_size : n - offset;
	quicksort(A+offset, size, f);
	merge_seq(A + offset, pivots, counts + i*num_buckets,
		 size, num_buckets-1, f);
      }

      // move data from blocks to buckets
      E *B = new_array_no_init<E>(n,1);
      size_t* bucket_offsets = pbbsa::transpose_buckets(A, B, counts, n, block_size,
						   num_blocks, num_buckets);
      free(counts);

      // sort within each bucket
      parallel_for_1 (size_t i = 0; i < num_buckets; ++i) {
	size_t start = bucket_offsets[i];
	size_t end = bucket_offsets[i+1];

	// middle buckets need not be sorted if two consecutive pivots
	// are equal
	if (i == 0 || i == num_buckets - 1 || f(pivots[i-1],pivots[i]))
	  quicksort(B+start, end - start, f);

	// move back to A (use memcpy to avoid initializers or overloaded =)
	memcpy((char*) (A+start), (char*) (B+start), (end-start)*sizeof(E));
      }
      delete_array(pivots,num_buckets-1);
      free(bucket_offsets);
      free(B);
    }
  }
}
