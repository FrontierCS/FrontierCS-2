#ifndef _S_GRAPHBASED_INCLUDED
#define _S_GRAPHBASED_INCLUDED
#include <math.h>
#ifdef CYCLE
#include <cyclegraph.h>
#else
#include "edgelistgraph.h"
#endif
#include "triangle.h"
#include "edge.h"
#include "block.h"

#include <atomic>
#include <thread>
#include "radix_configs.h"
#include "ska_sort.hpp"


using namespace std;

template <class E, class F>  
  void radixSortOneLevel(E* A, sizeT n, int doneOffset, F f, sizeT processors, int depth);

struct SortedBucket{
  int bucket;
  sizeT size;
};

sizeT sortedBucketSize(SortedBucket sb) {
  return -sb.size;
}


template <class E, class F, class Z>
  bool multiBitSwapBasedSort(E *A, sizeT n, int buckets, sizeT K, long *internalCounts, F extract, Z f, int depth, int doneOffset) {
 
  if(n < SERIAL_THRESHOLD || K == 1) {
#ifdef INSERTION_COARSEN
    if(n < INSERTION_THRESHOLD) {
      insertionSortSimilarToSTLnoSelfAssignment(A, n, f);
      return false;
    }
#endif
    // if this is the last recursion, we don't care about counts of buckets
    if(extract._offset == 0){
      internalCounts = NULL; 
    }

#ifdef ILP_SORT  
    if(n > 1024){
      struct metaData meta(extract._offset, true, false, internalCounts);
      ska_sort(A, A + n, f,meta);
      return meta.done;
    }else{
      _RadixSort_Unsigned_PowerOf2Radix_1(A, internalCounts, (long)n, BUCKETS, extract);
    }
    return true;
#else 
    return true; 
#endif 

      
  }


  SimpleBlock *blocks = new SimpleBlock[K];
  parallel_for_1(int i = 0; i < K; i ++) {
    unsigned long start = (long)i * n / K;
    unsigned long end = ((long)(i + 1) * n) / K;
    blocks[i].init(start, end);
    sortSimpleBlock(A, &blocks[i], extract);
  }

  parallel_for_1(int bucket = 0; bucket < BUCKETS; bucket ++) {
    internalCounts[bucket] = 0;        
    for(sizeT i = 0; i < K; i ++) {
      internalCounts[bucket] += blocks[i].counts[bucket];
    }
  }

  long countryEnds[BUCKETS];

  countryEnds[0] = internalCounts[0];
  for(int i = 1; i < BUCKETS; i ++) {
    countryEnds[i] = internalCounts[i] + countryEnds[i-1];
  }

  int order[BUCKETS];
  int rank[BUCKETS];
  SortedBucket sbs[BUCKETS];
  for(int i = 0; i < BUCKETS; i ++) {
    sbs[i].bucket = i;
    sbs[i].size = internalCounts[i];
  }
  insertion_sort(sbs, BUCKETS, sortedBucketSize);
  sizeT nextKs[BUCKETS]; 
  int countNonZero = 0; 
  sizeT local = K; 

  const long max_triangles = (K * BUCKETS + BUCKETS + 1) + BUCKETS; 
  Triangle  * triangles = new Triangle  [max_triangles];

  for(int i = 0; i < BUCKETS; i ++) {
    order[i] = sbs[i].bucket;
    rank[order[i]] = i;
    sizeT count = internalCounts[order[i]]; 
    sizeT nextK = (sizeT) ceil( ( 1.0 * (global_K * count)/ global_N) );
    nextKs[order[i]] =  nextK;
    if(count){
      countNonZero++; 
    }
  }

#ifdef CYCLE
  CycleGraph cycleGraph;
  cycleGraph.createCycleGraph(BUCKETS, K, blocks, countryEnds);
  vector<CycleGraph::CyclePlan> cyclePlan(BUCKETS * K + BUCKETS + 1);

  for(int index = 0; index < countNonZero; index ++) {
    int node = order[index];
    CycleGraph::Cycle cycle(node);
    int planc = 0;
    while(cycle.getNextCycle(&cycleGraph, &cyclePlan[planc ++])) {
      cycle.consumeCycle();
    }
    parallel_for_1(int i = 0; i < planc - 1; i ++){
      cyclePlan[i].executeCycle(A);
    }
    sizeT startOfCountry;
    if(node == 0) {
      startOfCountry = 0;
    } else {
      startOfCountry = countryEnds[node - 1];
    }
    sizeT nextK = nextKs[node];
    if(internalCounts[node] < INSERTION_THRESHOLD){	
      radixSortOneLevel(A + startOfCountry, internalCounts[node], doneOffset, f, nextK, depth + 1);
    }
    else{
      cilk_spawn radixSortOneLevel(A + startOfCountry, internalCounts[node], doneOffset, f, nextK, depth + 1);
    }	
  }
  cilk_sync;
  delete blocks;
  return false;
#else


  EdgeListGraph graph(K, rank, order);
  graph.createParallelGraph(BUCKETS, K, blocks, countryEnds);

  int triangles_count;
   

  for(int index = 0; index < countNonZero; index ++) {
    int node = order[index];
    triangles_count = 0;
    graph.extractNode2(node, triangles, &triangles_count);
    parallel_for_1(int i = 0; i < triangles_count; i ++) {
      executeTriangle(A, &triangles[i]);
    }

    sizeT startOfCountry;
    if(node == 0) {
      startOfCountry = 0;
    } else {
      startOfCountry = countryEnds[node - 1];
    }

    sizeT nextK = nextKs[node];
    if(internalCounts[node] < INSERTION_THRESHOLD){
			
      radixSortOneLevel(A + startOfCountry, internalCounts[node], doneOffset, f, nextK, depth + 1);
    }
    else{
      cilk_spawn radixSortOneLevel(A + startOfCountry, internalCounts[node], doneOffset, f, nextK, depth + 1);

    }

    graph.deleteNode(node);

    for(int i = 0; i < triangles_count; i ++) {
      graph.consumeTriangle(&triangles[i]);
    }

  }

  cilk_sync;
    
  delete triangles; 
  delete blocks;
#endif
  return false;
}
#endif
