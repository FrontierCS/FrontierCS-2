// This code is part of the Problem Based Benchmark Suite (PBBS)
// Copyright (c) 2011 Guy Blelloch and the PBBS team
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

#ifndef _S_RADIX_INCLUDED
#define _S_RADIX_INCLUDED



//#define DEBUG 0

#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include "utils.h"
#include "radix_configs.h"
#include "parallel.h"
#include <vector>
#include <utility>

#include "regionsSort.h"
#include "misc.h"
#include "common_sort.h"

#include "sequence.h"
using namespace std;

template <class E, class F>
  void radixSortOneLevel(E* A, sizeT n, int doneOffset, F f, sizeT K, int depth) {
  if(doneOffset <= 0 || n <= 1) {
    return;
  }

  int bits = MAX_RADIX;
  int buckets = BUCKETS;

  int start = max(doneOffset - bits, 0);


  bool needToRecurse = true;


  long localCounts[BUCKETS];
  needToRecurse = multiBitSwapBasedSort(A, n, buckets, K, localCounts, eBits<E, F>(doneOffset - start, start, f), f, depth, start);
   
  if(start <= 0)
    return;

  if(needToRecurse) {
    long sum = 0;
    for(int i = 0; i < BUCKETS; i++) {
      radixSortOneLevel(A + sum, localCounts[i], start, f, (sizeT)1, depth + 1);
      sum += localCounts[i];

    }
        
  }

}

template <class E, class F>
  void radixSortRoutine(sizeT index, E* A, sizeT n, int doneOffset, F f){ 
  sizeT K; 
#ifdef CYCLE
  K = getWorkers();
#else

  sizeT optimalCache = (n*sizeof(E))/BLOCK_DIVIDE; 
  if(optimalCache >  getWorkers() && optimalCache <= K_BOUND){
    K = optimalCache; 
  }
  else{
    if(optimalCache < getWorkers()) {
      K = getWorkers();
    } else {
      K = K_BOUND;
    }
  }

#endif
  global_K = K;
  global_N = n;

  radixSortOneLevel(A, n, doneOffset, f, K, 0);
          
};


int roundUpToRadixMultiple(int num){
  if(num % MAX_RADIX == 0){
    return num; 
  }
  else{
    return ((num / MAX_RADIX) + 1)  * MAX_RADIX;
  }
}

int roundUpBits(int num){

  int multipleOfRadix = roundUpToRadixMultiple(num); 
#ifdef BITS_HACK
#if MAX_RADIX == 8
  if(multipleOfRadix >= num + 2) {
    return num+2;
  }
#endif
#endif 
  return multipleOfRadix;
}

template <class E, class F, class K>
  void iSort(E *A, sizeT n, K m, F f) {
  int bits = roundUpBits(utils::log2Up(m)); 
  radixSortRoutine((sizeT)0, A, n, bits, f);
}


template <class E, class F, class K>
  K findMaxHelper(E* A, sizeT n, F f, sizeT P, K t){
  
  K* temp = (K*)malloc(sizeof(E) * (P + 1));
  parallel_for(unsigned long i = 0; i <P; i ++) {
    unsigned long start = i * (long)n / P;
    unsigned long end = ((long)(i + 1) * n) / P;
    K local_max = 0;
    for(;start < end; start++){
      K current = f(A[start]);
      if(current > local_max) {
	local_max = current; 
      }

    }
    temp[i] = local_max; 
  }

  K global_max = temp[0]; 

  for(int i = 0; i < P; i++) {
    if(temp[i] > global_max) {
      global_max = temp[i]; 
    }
  }
  free(temp);
  return global_max;
}


template <class E, class F, class K>
  K findMax(E* A, sizeT n, F f, K temp){
  K maxV = findMaxHelper(A, n, f, (sizeT)1000*getWorkers(), temp);
  return maxV; 
}
template <class T, class F>
  static void parallelIntegerSort(T *A, sizeT n, F f) {
  T temp;
  T maxV = findMax(A, n, f, temp);
  iSort(A, n, maxV+1,  f); 
}

template <class T>
static void parallelIntegerSort(T *A, sizeT n) {
  parallelIntegerSort(A, n, utils::identityF<T>());
}

template <class T1, class T2, class F>
  void parallelIntegerSort(pair<T1, T2> *A, sizeT n, F f) {
  T1 temp;
  T1 maxV = findMax(A, n, f, temp);
  iSort(A, n, maxV+1, f);
}

template <class T1, class T2>
  void parallelIntegerSort(pair<T1, T2> *A, sizeT n) {
  parallelIntegerSort(A, n, utils::firstF<T1, T2>());
}

#endif
