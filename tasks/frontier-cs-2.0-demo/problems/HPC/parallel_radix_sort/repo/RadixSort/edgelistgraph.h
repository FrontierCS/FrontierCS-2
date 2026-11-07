#ifndef _S_EDGELISTGRAPH_INCLUDED
#define _S_EDGELISTGRAPH_INCLUDED

#include <stdio.h>
#include "edge.h"
#include <vector>
#include "prefixsum.h"
#include "triangle.h"
#include "block.h"
#include "misc.h"
#include "radix_configs.h"
#include "../common/blockRadixSort.h"
#include "../common/blockRadixSort2.h"
#include <algorithm>
using namespace std;

#define MAX_PARALLEL_EDGES (MAXP * BUCKETS + BUCKETS + 5)

class EdgeListGraph{

 public:
  vector< Edge > graphToEdges[BUCKETS];
  vector< Edge > graphFromEdges[BUCKETS];
  int * order;
  int * rank;

  EdgeListGraph(int P, int *rank, int *order) {
    for(int bucket = 0; bucket < BUCKETS; bucket ++) {
      graphToEdges[bucket].clear();
      graphFromEdges[bucket].clear();
    }
    this->rank = rank;
    this->order = order;
  }

  void addEdge( Edge  *e){
    if(e->isTrivial()){
      return;
    }


    if(rank[e->to] < rank[e->from]) {
      graphToEdges[e->to].push_back(*e);
    } else {
      graphFromEdges[e->from].push_back(*e);
    }
  }

  void getFromEdgesSubgraph(int node, vector< Edge *> &res) {
    res.clear();
    for(int i = 0; i < graphFromEdges[node].size(); i++) {
      res.push_back(&graphFromEdges[node][i]);
    }        
  }

  void getToEdgesSubgraph(int node, vector< Edge  *> &res) {
    res.clear();
    for(int i = 0; i < graphToEdges[node].size(); i++) {
      res.push_back(&graphToEdges[node][i]);
    }
  }

  void printGraph() {
    printf("Printing Graph!\n");
    printf("Edges Counts\n");
    vector < Edge  *> parallelFromEdges;
    vector < Edge  *> parallelToEdges;
    for(int node = 0; node < BUCKETS; node ++) {
      printf("vector for node %d\n", node);
      this->getFromEdgesSubgraph(node, parallelFromEdges);
      this->getToEdgesSubgraph(node, parallelToEdges);
    }

  }


  void createParallelGraph(long buckets, long P, SimpleBlock *blocks, const long *countryEnds) {
    int currentCountry = 0;
    long currentRegionStart = 0;
    for(int currentBlock = 0; currentBlock < P; currentBlock ++) {
      for(int currentValue = 0; currentValue < BUCKETS; currentValue ++) {

	while(currentCountry < BUCKETS) {
	  if (blocks[currentBlock].bucketEnds[currentValue] <= countryEnds[currentCountry]) {
	    //if country is a superset of the current bucket inside current block.
	    //both sides of comparison are exclusive.

	    long currentRegionEnd = blocks[currentBlock].bucketEnds[currentValue];
	    Edge  edge =  Edge (currentRegionStart, currentRegionEnd - currentRegionStart, currentCountry, currentValue);



	    addEdge(&edge);
	    currentRegionStart = currentRegionEnd;
	    break;
	  } else {
	    // some parts of the bucket is not included in the country.
	    // Region will include the intersection of the country and the bucket.
	    long currentRegionEnd = countryEnds[currentCountry];

	    Edge  edge =  Edge (currentRegionStart, currentRegionEnd - currentRegionStart, currentCountry, currentValue);
	    addEdge(&edge);
	    currentRegionStart = currentRegionEnd;
	    currentCountry++;
	  }
	}
      }
    }

  }

  static sizeT getAmount( Edge  * e) {
    return e->amount;
  }

  static sizeT getAmount( Edge  &e) {
    return e.amount;
  }

  static  Edge * getL( Edge  *e) {
    return e;
  }
  static  Edge * getL( Edge  &e) {
    return &e;
  }

  // TODO(frontier): implement these planning operations.
  void matchEdgesToTriangle( Edge * fromEdge, Edge * toEdge, Triangle * result);

  template<class T>
  int extractTriangles(vector<T> &fromRegions, int fromSize,
                       vector<T> &toRegions, int toSize,
                       Triangle * triangles, int triangles_count);

  int extractTriangles(vector< Edge > &fromRegions, vector< Edge > &toRegions,
                       Triangle * triangles, int triangles_count);

  int extractTriangles(vector< Edge *> &fromRegions, vector< Edge *> &toRegions,
                       Triangle * triangles, int triangles_count);

  int getEdgesCount(int node) {
    return graphToEdges[node].size() + graphFromEdges[node].size(); 
  }

  void deleteNode(int node) {
    vector< Edge >().swap(graphToEdges[node]);
    vector< Edge >().swap(graphFromEdges[node]);
  }



  void extractNode2(int node, Triangle * triangles, int* triangles_count) {
    // TODO(frontier): implement this planning operation.
    *triangles_count = 0;
  }
  void consumeTriangle(Triangle  *triangle) {
    // TODO(frontier): implement this graph update operation.
  }

  ~EdgeListGraph() {
    for(int bucket = 0; bucket < BUCKETS; bucket ++) {
      graphToEdges[bucket].clear();
      graphFromEdges[bucket].clear();
    }
  }

};


#endif
