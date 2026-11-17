#ifndef _S_EDGE_INCLUDED
#define _S_EDGE_INCLUDED

#include <stdio.h>

class Edge {
 public:
  sizeT amount;
  sizeT start;
  int from;
  int to;
    
  Edge() {
    this->start = 0;
    this->amount = 0;
    this->from = 0;
    this->to = 0;
  }
    
  Edge(sizeT start, sizeT amount, int from, int to) {
    this->start = start;
    this->amount = amount;
    this->from = from;
    this->to = to;
  }



  void setEmpty() {
    amount = 0;
  }

  bool isTrivial(){
    return (from == to) || (amount <= 0);
  }

  void printEdge() {
#ifdef LONG_ARRAY
    printf("Edge: start: %ld, amount %ld, from %d -> to %d\n", start, amount, from, to);
#else
    printf("Edge: start: %d, amount %d, from %d -> to %d\n", start, amount, from, to);
#endif
  }

  bool compareEdge(Edge *anotherEdge) {
    if(this->from != anotherEdge->from) {
      return false;
    }
    if(this->to != anotherEdge->to) {
      return false;
    }
    if(this->amount != anotherEdge->amount) {
      return false;
    }
    if(this->start != anotherEdge->start) {
      return false;
    }
    return true;
  }
};
#endif
