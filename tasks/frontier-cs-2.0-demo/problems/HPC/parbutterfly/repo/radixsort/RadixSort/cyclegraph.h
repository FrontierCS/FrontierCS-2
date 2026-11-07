#ifndef _S_CYCLEGRAPH_INCLUDED
#define _S_CYCLEGRAPH_INCLUDED
#ifndef STABLE
#include <stdio.h>
#include <edge.h>
#include <vector>
#include <prefixsum.h>
#include <triangle.h>
#include <block.h>
#include <misc.h>
#include <radix_configs.h>

using namespace std;
namespace radix{
#define MAX_PARALLEL_EDGES (MAXP * R_BUCKETS + R_BUCKETS + 5)


class CycleGraph{

    public:
    vector<Edge > graphFromEdges[R_BUCKETS];


    CycleGraph() {
        for(int bucket = 0; bucket < R_BUCKETS; bucket ++) {
            graphFromEdges[bucket].clear();
        }
    }

    void addEdge(Edge  *e){
        if(e->isTrivial()){
            return;
        }
        graphFromEdges[e->from].push_back(*e);
    }

    Edge * getNextEdge(int node) {
        int sz = graphFromEdges[node].size();
        while(sz && graphFromEdges[node][sz - 1].amount == 0) {
            sz --;
        }
        if(sz == 0) {
            return NULL;
        }
        graphFromEdges[node].resize(sz);
        return &graphFromEdges[node][sz - 1];
    }

    bool hasEdges(int node) {
        return getNextEdge(node) != NULL;
    }
class CyclePlan {
	    public:
	int length;
	sizeT starts[R_BUCKETS];
	sizeT amount;
	CyclePlan() {
	    length = 0;
	}
	template <class E>
	void executeCycle(E* A) {
	    if(amount < PARALLEL_FOR_THRESHOLD) {
		    for(sizeT pos = 0; pos < amount; pos ++) {
					E save = A[starts[length - 1] + pos];
					for(int i = length - 1; i > 0; i --) {
							A[starts[i] + pos] = A[starts[i - 1] + pos];
					}
					A[starts[0] + pos] = save;
		    }
	    } else {
		    parallel_for_swap(sizeT pos = 0; pos < amount; pos ++) {
					E save = A[starts[length - 1] + pos];
					for(int i = length - 1; i > 0; i --) {
							A[starts[i] + pos] = A[starts[i - 1] + pos];
					}
					A[starts[0] + pos] = save;
		    }
	    }
	}
    };
    class Cycle {
        public:
        Edge  *path[R_BUCKETS];
        int visited[R_BUCKETS];
        int pathIndex;
        sizeT amount;        
        int cycle_start;
        int cycle_length;
	int currentNode;
	CyclePlan cyclePlan;	
        Cycle(int node){
	    currentNode = node;
            pathIndex = 0;
	    cycle_start = 0;
	    cycle_length = 0;
	    amount = 0;
            for(int i = 0; i < R_BUCKETS; i++) {
                visited[i] = -1;
            }
        }

        bool getNextCycle(CycleGraph *graphcycle, CyclePlan *cyclePlan){
            if(!graphcycle->hasEdges(currentNode)) {
                //we are done with this node.
                return false;
            }

            do{
                visited[currentNode] = pathIndex;
                path[pathIndex] = graphcycle->getNextEdge(currentNode);
                currentNode = path[pathIndex]->to;
                pathIndex ++;
            } while(visited[currentNode] == -1);



            cycle_start = visited[currentNode];
            amount = path[cycle_start]->amount;
            for(int i = cycle_start; i < pathIndex; i ++) {
                amount = min(amount, path[i]->amount);
                visited[path[i]->from] = -1;
		cyclePlan->starts[i-cycle_start] = path[i]->start;
            }
	    cyclePlan->amount = amount;
	    cyclePlan->length = pathIndex - cycle_start;
            cycle_length = pathIndex - cycle_start;
            pathIndex = cycle_start;
            return true;
        }
        
        void consumeCycle() {
            parallel_for(int i = cycle_start; i < cycle_start + cycle_length; i ++) {
                path[i]->amount -= amount;
                path[i]->start += amount;
            }
        }

    };

    void createCycleGraph(int buckets, sizeT P, SimpleBlock *blocks, const long *countryEnds) {
        int currentCountry = 0;
        sizeT currentRegionStart = 0;
        for(sizeT currentBlock = 0; currentBlock < P; currentBlock ++) {
            for(int currentValue = 0; currentValue < R_BUCKETS; currentValue ++) {
                while(currentCountry < R_BUCKETS) {
                    if (blocks[currentBlock].bucketEnds[currentValue] <= countryEnds[currentCountry]) {
                    //if country is a superset of the current bucket inside current block.
                    //both sides of comparison are exclusive.

                    sizeT currentRegionEnd = blocks[currentBlock].bucketEnds[currentValue];
                    Edge  edge = Edge (currentRegionStart, currentRegionEnd - currentRegionStart, currentCountry, currentValue);
                    addEdge(&edge);
                    currentRegionStart = currentRegionEnd;
                    break;
                    } else {
                    // some parts of the bucket is not included in the country.
                    // Region will include the intersection of the country and the bucket.
                    sizeT currentRegionEnd = countryEnds[currentCountry];
                    Edge  edge = Edge (currentRegionStart, currentRegionEnd - currentRegionStart, currentCountry, currentValue);
                    addEdge(&edge);
                    currentRegionStart = currentRegionEnd;
                    currentCountry++;
                    }
                }
            }
        }

    }
    void printGraph() {
        printf("No Printing Function!\n");
    }

    void verifyGraph(int node) {
        vector<Edge *> toEdges;
        vector<Edge *> fromEdges; 
        sizeT toSum = 0;
        for(int i = 0; i < toEdges.size(); i ++) {
            if(toEdges[i]->isTrivial())
                continue;
            toSum += toEdges[i]->amount;
        }
        sizeT fromSum = 0;
        for(int i = 0; i < fromEdges.size(); i ++) {
            if(fromEdges[i]->isTrivial())
                continue;        
            fromSum += fromEdges[i]->amount;
        }
        if(toSum != fromSum) {
        printf("Graph Invariants don't hold!\n");
        vector<sizeT> toPrefixSum;
        vector<sizeT> fromPrefixSum;          
        getPrefixSum(toEdges, toPrefixSum);
        getPrefixSum(fromEdges, fromPrefixSum);
        printPrefixSum(toPrefixSum);
        printPrefixSum(fromPrefixSum);
        while(1);
        }

    }


};
}
#endif

#endif

