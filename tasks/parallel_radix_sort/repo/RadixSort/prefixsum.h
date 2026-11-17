#ifndef _S_PREFIXSUM_INCLUDED
#define _S_PREFIXSUM_INCLUDED

#include "edge.h"
#include <vector>
#include "radix_configs.h"

using namespace std;
typedef unsigned char uint8_t;

void getPrefixSum(vector<uint8_t> &input, vector<uint8_t> &res) {
  long sum = 0;
  for(int i = 0; i < input.size(); i ++) {
    res[i] = sum;
    sum += input[i];
  }
  res[input.size()] = sum;
}

void getPrefixSum(uint8_t *input, uint8_t *res, int size) {
  long sum = 0;
  for(int i = 0; i < size; i ++) {
    res[i] = sum;
    sum += input[i];
  }
  res[size] = sum;
}

void getSerialPrefixSum( Edge  ** input,sizeT *res, int size) {
  sizeT sum = 0;
  for(int i = 0; i < size; i ++) {
    res[i] = sum; 
    sum += input[i]->amount; 
  }
  res[size] = sum;
}


void getParallelPrefixSum( Edge  ** input,sizeT *res, int size) {
  parallel_for_swap(int i = 0; i < size; i ++) {
    res[i] = input[i]->amount;
  }
  long total = sequence::plusScan(res, res, size);
  res[size] = total;
}


void getSerialPrefixSumReal( Edge  * input,sizeT *res, int size) {
  sizeT sum = 0;
  for(int i = 0; i < size; i ++) {
    res[i] = sum;
    sum += input[i].amount; 
  }
  res[size] = sum;
}

void getParallelPrefixSumReal( Edge  * input,sizeT *res, int size) {
  parallel_for_swap(int i = 0; i < size; i ++) {
    res[i] = input[i].amount;
  }
  sizeT total = sequence::plusScan(res, res, size);
  res[size] = total;
}

void getPrefixSum(vector<Edge*> &input, vector<sizeT> &res) {
  sizeT sum = 0;
  for(int i = 0; i < input.size(); i ++) {
    res.push_back(sum);
    sum += input[i]->amount;
  }
  res.push_back(sum);
}

void getPrefixSumLong(sizeT *input, sizeT *res, int size) {
  long sum = 0;
  for(int i = 0; i < size; i ++) {
    res[i] = sum;
    sum += input[i];
  }
  res[size] = sum;
} 

void getPrefixSum(sizeT offset, sizeT* count, sizeT* prefixSum, int length) {
  prefixSum[0] = offset + count[0];
  for(int i = 1; i < length; i++) {
    prefixSum[i] = count[i] + prefixSum[i-1];
  }
}


void printPrefixSum(vector<sizeT> ps) {
#ifdef LONG_ARRAY
  printf("ps size %lu\n", ps.size());
  for(int i = 0; i < ps.size(); i++) {
    printf("%ld |",ps[i]);
  }
#else
  printf("ps size %lu\n", ps.size());
  for(int i = 0; i < ps.size(); i++) {
    printf("%d |",ps[i]);
  }
#endif
  printf("\n");
}
#endif
