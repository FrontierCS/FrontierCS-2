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
namespace radix{
#define MAX_PARALLEL_EDGES (MAXP * R_BUCKETS + R_BUCKETS + 5)

class EdgeListGraph{

    public:
    vector< Edge > graphToEdges[R_BUCKETS];
    vector< Edge > graphFromEdges[R_BUCKETS];
    int * order;
    int * rank;

    EdgeListGraph(int P, int *rank, int *order) {
        for(int bucket = 0; bucket < R_BUCKETS; bucket ++) {
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
        for(int node = 0; node < R_BUCKETS; node ++) {
            printf("vector for node %d\n", node);
            this->getFromEdgesSubgraph(node, parallelFromEdges);
            this->getToEdgesSubgraph(node, parallelToEdges);
        }

    }


    void createParallelGraph(long buckets, long P, SimpleBlock *blocks, const long *countryEnds) {
        int currentCountry = 0;
        long currentRegionStart = 0;
        for(int currentBlock = 0; currentBlock < P; currentBlock ++) {
            for(int currentValue = 0; currentValue < R_BUCKETS; currentValue ++) {

                while(currentCountry < R_BUCKETS) {
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

    void matchEdgesToTriangle( Edge * fromEdge,  Edge  * toEdge, Triangle * result) {
	sizeT toRemaining = toEdge->amount;
	sizeT fromRemaining = fromEdge->amount;
	sizeT min;
	if(toRemaining < fromRemaining) {
	    min = toRemaining;
	} else {
	    min = fromRemaining;
	}
	makeTrianglefromDynamicEdge(result, toEdge, fromEdge, min);
	toEdge->amount -= min;
	toEdge->start += min;
	fromEdge->amount -= min;
	fromEdge->start += min;
    }


template<class T>
    int extractTriangles(vector<T> &fromRegions, int fromSize, vector<T> &toRegions, int toSize, Triangle * triangles, int triangles_count) {

			if(fromSize == 0 || toSize == 0) {
					return triangles_count ;
			}

			int fromP = 0; 
			int toP = 0;

			while(true) {

					for(;toP < toSize && !getAmount(toRegions[toP]);toP++) {

					}

					for(;fromP < fromSize && !getAmount(fromRegions[fromP]);fromP++) {

					}

					if(toP == toSize || fromP == fromSize) {
							break;
					}
		
					matchEdgesToTriangle(getL(fromRegions[fromP]), getL(toRegions[toP]), &triangles[triangles_count]);
					triangles_count = triangles_count + 1;
			}
	
			return triangles_count;	
    }


    int extractTriangles(vector< Edge > &fromRegions, vector< Edge > &toRegions, Triangle * triangles, int triangles_count) {
				return extractTriangles(fromRegions, (int)fromRegions.size(), toRegions, (int)toRegions.size(), triangles, triangles_count);

    }

    int extractTriangles(vector< Edge *> &fromRegions, vector< Edge *> &toRegions, Triangle * triangles, int triangles_count) {
				return extractTriangles(fromRegions, (int)fromRegions.size(), toRegions, (int)toRegions.size(), triangles, triangles_count);

    }

    int getEdgesCount(int node) {
	    return graphToEdges[node].size() + graphFromEdges[node].size(); 
    }

    void deleteNode(int node) {
			vector< Edge >().swap(graphToEdges[node]);
			vector< Edge >().swap(graphFromEdges[node]);
    }



    void extractNode2(int node, Triangle * triangles, int* triangles_count) {
        vector< Edge > fromRegions;
        fromRegions = graphFromEdges[node];
        vector< Edge > toRegions;     
        toRegions = graphToEdges[node];


#ifdef EXTRACT_2CYCLES
///////////////////////EXTRACTING two-cycles START/////////////////////////
	vector< Edge *> edgesPerNodeFrom[R_BUCKETS];
	vector< Edge *> edgesPerNodeTo[R_BUCKETS];
	for(int i = 0; i < fromRegions.size(); i++) {
	    edgesPerNodeFrom[fromRegions[i].to].push_back(&fromRegions[i]);
	}

	for(int i = 0; i < toRegions.size(); i++) {
	    edgesPerNodeTo[toRegions[i].from].push_back(&toRegions[i]);
	}
       	
	if(toRegions.size() < R_BUCKETS - (rank[node] + 1)) {
	        for(int i = 0; i < toRegions.size(); i++) {
		    int bucket = toRegions[i].from;
		    *triangles_count = extractTriangles(edgesPerNodeFrom[bucket], edgesPerNodeTo[bucket], triangles, *triangles_count);
		    edgesPerNodeFrom[bucket].clear();
		    edgesPerNodeTo[bucket].clear();
		}

	} else {
		if(fromRegions.size() < R_BUCKETS - (rank[node] + 1)){
			for(int i = 0; i < fromRegions.size(); i++) {
			    int bucket = fromRegions[i].to;
			    *triangles_count = extractTriangles(edgesPerNodeFrom[bucket], edgesPerNodeTo[bucket], triangles, *triangles_count);
			    edgesPerNodeFrom[bucket].clear();
			    edgesPerNodeTo[bucket].clear();
			}
		}	
		else {
			for(int index = rank[node] + 1; index < R_BUCKETS; index ++) {
			    int bucket = order[index];
			    *triangles_count = extractTriangles(edgesPerNodeFrom[bucket], edgesPerNodeTo[bucket], triangles, *triangles_count);
			}
		}
	}
///////////////////////EXTRACTING two-cycles END/////////////////////////
#endif
        *triangles_count = extractTriangles(fromRegions, toRegions, triangles, *triangles_count);
    }
    void consumeTriangle(Triangle  *triangle) {
	 
				if(triangle->isTrivial()) {
						return;
				}

				 Edge  edge(triangle->offset1 , triangle->amount ,triangle->from, triangle->to); 

				if(rank[triangle->to] < rank[triangle->from]) {
					 graphToEdges[triangle->to].push_back(edge);
				} else {
					 graphFromEdges[triangle->from].push_back(edge);            
				}
    }

    ~EdgeListGraph() {
        for(int bucket = 0; bucket < R_BUCKETS; bucket ++) {
            graphToEdges[bucket].clear();
            graphFromEdges[bucket].clear();
        }
    }

};

}
#endif
