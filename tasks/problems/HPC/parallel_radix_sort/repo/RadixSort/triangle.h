#ifndef _S_TRIANGLE_INCLUDED
#define _S_TRIANGLE_INCLUDED

#include <stdio.h>
#include "edge.h"
#include "prefixsum.h"
#include "radix_configs.h"
#include <algorithm>
using namespace std;

struct Triangle {

  int from;
  int to;
  sizeT offset1;
  sizeT offset2;
  sizeT amount;
  Triangle() {

  }
 
  void setEmpty() {
    amount = 0;
  }
  bool isEmpty() {
    return amount == 0;
  }

  bool isTrivial() {
    return amount == 0 || to == from;
  }
  void printTriangle() {
    if(isEmpty()) {
      printf("Empty Triangle\n");
    }
#ifndef LONG_ARRAY
    printf("Triangle: (%d -> %d) amount %d \n",from,to,amount);
    printf("offset1 : %d offset2 %d\n",offset1, offset2);
#endif
  }
};

//template<class sizeT>
void makeTriangle(Triangle  *triangle, Edge  *firstEdge, Edge * secondEdge, sizeT firstRemaining, sizeT secondRemaining) {

  triangle->from = firstEdge->from;

  triangle->to = secondEdge->to;

  triangle->offset1 = firstEdge->start + firstEdge->amount - firstRemaining;
  triangle->offset2 = secondEdge->start + secondEdge->amount - secondRemaining;
  sizeT amount = min(firstRemaining, secondRemaining);
  triangle->amount = amount;
}

void makeTrianglefromDynamicEdge(Triangle  *triangle, Edge  *firstEdge, Edge * secondEdge, sizeT amount) {
  triangle->from = firstEdge->from;
  triangle->to = secondEdge->to;
  triangle->offset1 = firstEdge->start;
  triangle->offset2 = secondEdge->start;
  triangle->amount = amount;
}

template <class E>
void serialExecuteTriangle(E *A, Triangle  *triangle) {
  E* start = A + triangle->offset2;
  std::swap_ranges(start, start+triangle->amount, A + triangle->offset1);
}

template <class E>
void parallelExecuteTriangle(E *A, Triangle  *triangle) {
  parallel_for_swap(sizeT index = 0; index < triangle->amount; index ++) {
    sizeT from = triangle->offset2 + index; 
    sizeT to = triangle->offset1 + index; 
    E temp = A[from];
    A[from] = A[to];
    A[to] = temp;
  }

}

template <class E>
void executeTriangle(E *A, Triangle  *triangle) {

  if(triangle->amount < PARALLEL_FOR_THRESHOLD) {
    serialExecuteTriangle(A, triangle);
  } else {
    parallelExecuteTriangle(A, triangle);
  }

}

#endif

