#ifndef GRIDKDTREE_H
#define GRIDKDTREE_H

#include <vector>
#include <algorithm>
#include <iostream>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <numeric>

#include "DBSCANGeometry.h"
#include "geometry.h"
#include "DBSCAN.h"
#include "Grids.h"
#include "GetNeighborTypeDef.h"

//#define LEFT_MEDIAN // left median could result in leaf size potentially way larger than KDT_LEAF_THRESHOLD, comment out for more accurate leaf sizes
#define KDT_LEAF_THRESHOLD 16 // <= than which must be a leaf

inline void GetGridBoundingBoxParallel(GridsGN *t_grids, OtLongDataType *t_maxCoords, OtLongDataType *t_minCoords) {
  intT P = getWorkers() * 8; // todo numworker

  GdLongDataType blockSize = (t_grids->GetNumGrids() + P - 1) / P;
  OtLongDataType localMaxes[P * t_grids->GetDim()], localMins[P * t_grids->GetDim()];
  for (intT i = 0; i < t_grids->GetDim() * P; ++ i) {
    localMaxes[i] = t_grids->ReadGridVec(0, i % t_grids->GetDim());
    localMins[i] = t_grids->ReadGridVec(0, i % t_grids->GetDim());
  }
  parallel_for(intT i=0; i<P; i++) {
    GdLongDataType start = i*blockSize;
    GdLongDataType end = min((i+1)*blockSize, t_grids->GetNumGrids());
    // each thread find its own bounding box
    for (GdLongDataType j = start; j < end; ++ j) {
      for (intT d = 0; d < t_grids->GetDim(); ++ d) {
        localMaxes[i * t_grids->GetDim() + d] = t_grids->ReadGridVec(j, d) > localMaxes[i * t_grids->GetDim() + d] ? t_grids->ReadGridVec(j, d) : localMaxes[i * t_grids->GetDim() + d];
        localMins[i * t_grids->GetDim() + d] = t_grids->ReadGridVec(j, d) < localMins[i * t_grids->GetDim() + d] ? t_grids->ReadGridVec(j, d) : localMins[i * t_grids->GetDim() + d];
      }
    }
  }

  for (intT d = 0; d < t_grids->GetDim(); ++ d) {
    t_maxCoords[d] = localMaxes[d];
    t_minCoords[d] = localMins[d];
  }
  for(intT i=0;i<P;i++) {
    for (intT d = 0; d < t_grids->GetDim(); ++ d) {
      t_maxCoords[d] = localMaxes[i * t_grids->GetDim() + d] > t_maxCoords[d] ? localMaxes[i * t_grids->GetDim() + d] : t_maxCoords[d];
      t_minCoords[d] = localMins[i * t_grids->GetDim() + d] < t_minCoords[d] ? localMins[i * t_grids->GetDim() + d] : t_minCoords[d];
    }
  }
}


inline void GetGridBoundingBoxSequential(GridsGN *t_grids, OtLongDataType *maxCoords, OtLongDataType *minCoords) {
  OtLongDataType curCoords[t_grids->GetVecLen()];
  for (intT i = 0; i < t_grids->GetDim(); ++ i) {
    minCoords[i] = 0;
    maxCoords[i] = 0;
    curCoords[i] = 0;
  }
  for (intT i = 0; i < t_grids->GetNumGrids(); ++ i) {
    t_grids->GetGridVec(i, curCoords);
    for (intT d = 0; d < t_grids->GetDim(); ++ d) {
if (curCoords[d] < minCoords[d]) {
  minCoords[d] = curCoords[d];
}
if (curCoords[d] > maxCoords[d]) {
  maxCoords[d] = curCoords[d];
}
    }
  }
}

#define GRID_INCLUDE -1
#define GRID_DISJOINT -2
#define GRID_INTERSECT -3

inline bool IsIncluded(OtLongDataType* t_aggressiveMax, OtLongDataType* t_aggressiveMin,
     OtLongDataType *t_passiveMax, OtLongDataType *t_passiveMin, intT t_dim) {
  // all aggressive max >= passive max
  for (intT d = 0; d < t_dim; ++ d) {
    if (t_aggressiveMax[d] < t_passiveMax[d]) {
return false;
    }
  }
  // all aggressive min <= passive min
  for (intT d = 0; d < t_dim; ++ d) {
    if (t_aggressiveMin[d] > t_passiveMin[d]) {
return false;
    }
  }
  return true;
}

inline bool IsDisjoint(OtLongDataType* t_aggressiveMax, OtLongDataType* t_aggressiveMin,
     OtLongDataType *t_passiveMax, OtLongDataType *t_passiveMin, intT t_dim) {
  // one of aggressive min > passive max
  for (intT d = 0; d < t_dim; ++ d) {
    if (t_aggressiveMin[d] > t_passiveMax[d]) {
return true;
    }
  }
  // one of aggressive max < passive min
  for (intT d = 0; d < t_dim; ++ d) {
    if (t_aggressiveMax[d] < t_passiveMin[d]) {
return true;
    }
  }
  return false;
}

inline intT GridRelation(OtLongDataType* t_aggressiveMax, OtLongDataType* t_aggressiveMin,
       OtLongDataType *t_passiveMax, OtLongDataType *t_passiveMin, intT t_dim) {

  if(IsDisjoint(t_aggressiveMax, t_aggressiveMin, t_passiveMax, t_passiveMin, t_dim)) {
    return GRID_DISJOINT;
  } else if (IsIncluded(t_aggressiveMax, t_aggressiveMin, t_passiveMax, t_passiveMin, t_dim)) {
    return GRID_INCLUDE;
  } else {
    return GRID_INTERSECT;
  }
}

/*
  tree node
   - start grid offset int
   - end grid offset int
   - left child offset int (-1 for leaf)
   - right child offset int
   - minCoords (bounding box)
   - maxCoords (bounding box)

   so node size is 4
   since the tree is binary, given n elements, at most n leafs
   at most 2n nodes, 2n - 1 bounds the tree size
 */

// kdt special values
#define EMPTY_VAL -4
#define LEAF -1 // no left or right child

static intT g_nodeSize;

inline OtLongDataType GetMaxTreeSize(PtLongDataType t_n) {
  if (t_n == 0) {
    cout << "getting tree size for 0 nodes, wrong!!!"; exit(1);
  } else {
    return (2 * t_n - 1) * g_nodeSize;
  }
}

// compare grid i and j on a particular dimension
struct gridDimComparator {
  intT m_sortDim;
  GridsGN *m_grids;
  gridDimComparator(GridsGN *t_grids, intT t_sortDim): m_sortDim(t_sortDim), m_grids(t_grids) {
  }
  bool operator() (const GdLongDataType i, GdLongDataType j) {
    return m_grids->ReadGridVec(i, m_sortDim) < m_grids->ReadGridVec(j, m_sortDim);
  }
};

struct GridKDTree {
private:
  GridsGN *m_grids;
  GdLongDataType *m_gridsSorted;
  OtLongDataType *m_tree;
  OtLongDataType m_treeLen;

  /*
    construction:
    first sort by 1st dim, divide into two
    then for each half, sort by second dim, divide
    then for rem half, again sort by the next dimension
    sorting is
     - find pivot
     - divide points
   */

public:

  static const OtLongDataType kdt_par_const = 1000;

  GridKDTree(GridsGN *t_grids) {
    bool t_sequential;
    if (t_grids->GetNumGrids() < kdt_par_const) {
      t_sequential = true;
    } else {
      t_sequential = false;
    }

    m_grids = t_grids;

    // 4 for offset1, offset2, leftchild, rightchild, leftchild -1 for leaf
    g_nodeSize = 4 + t_grids->GetDim() * 2;
    
    // creates grid id array to be sorted
    m_gridsSorted = static_cast<GdLongDataType *>(malloc(sizeof(GdLongDataType) * t_grids->GetNumGrids()));

    if (t_sequential) {
      for (GdLongDataType i = 0; i < t_grids->GetNumGrids(); ++ i) {
	m_gridsSorted[i] = i;
      }
    } else {
      parallel_for (GdLongDataType i = 0; i < t_grids->GetNumGrids(); ++ i) {
	m_gridsSorted[i] = i;
      }
    }

    // todo just for debugging, remove later
    m_treeLen = GetMaxTreeSize(t_grids->GetNumGrids());
    m_tree = static_cast<OtLongDataType *>(malloc(sizeof(OtLongDataType) * m_treeLen ));
    
    // get bounding box
    OtLongDataType minCoords[t_grids->GetDim()];
    OtLongDataType maxCoords[t_grids->GetDim()];
    if (t_sequential) {
      GetGridBoundingBoxSequential(t_grids, maxCoords, minCoords);
    } else {
      GetGridBoundingBoxParallel(t_grids, maxCoords, minCoords);
    }

    // construct tree
    if (t_sequential) {
      TreeConstructSequential(m_tree, 0, t_grids->GetNumGrids(), 0, minCoords, maxCoords, 0);
    } else {
      TreeConstruct(m_tree, 0, t_grids->GetNumGrids(), 0, minCoords, maxCoords, 0);
    }
  }

  inline OtLongDataType *GetTree() {
    return m_tree;
  }

  inline void TreeWrite(OtLongDataType *t_tree, OtLongDataType t_loc, OtLongDataType t_stuff) {
    t_tree[t_loc] = t_stuff;
  }

  // t_tree: buffer for storing the tree
  // t_gridStart: point (grid) start offset
  // t_numGrids: num grids from the grid starting point
  // t_depth: depth of kdtree, starts from 0
  // t_maxCoords: current grid's max coordinates
  // t_minCoords: ...

  void TreeConstruct(OtLongDataType *t_tree, GdLongDataType t_gridStart, GdLongDataType t_numGrids, OtLongDataType t_depth, OtLongDataType *t_minCoords, OtLongDataType *t_maxCoords, OtLongDataType absStart) {

    if (t_numGrids < 1) { // terminates
      cout << "reaching kdt construction with < 1 points"; exit(1);
    } else {

      TreeWrite(t_tree, 0, t_gridStart);
      TreeWrite(t_tree, 1, t_gridStart + t_numGrids);

      
      // if tree size 1, no need more children
      if (t_numGrids <= KDT_LEAF_THRESHOLD) {
        TreeWrite(t_tree, 2, LEAF);
      } else {

        intT sortDim = t_depth % m_grids->GetDim();

        sampleSort(m_gridsSorted + t_gridStart, t_numGrids, gridDimComparator(m_grids, sortDim));

        GdLongDataType median = t_gridStart + static_cast<GdLongDataType>(t_numGrids / 2); // starting offset for right
        GdLongDataType medianVal = m_grids->ReadGridVec(m_gridsSorted[median], sortDim);
	GdLongDataType medianMemorized = median;
  
        // might have duplicate medians
        while ( median - 1 >= 0 && medianVal == m_grids->ReadGridVec(m_gridsSorted[median - 1], sortDim) ) {
          median --;
        }

        // median node belong to the right child
        // left: [start, median)
        // right: [median, end)

        // left size could be zero
        GdLongDataType leftSize = median - t_gridStart;
        GdLongDataType rightSize = t_numGrids - median + t_gridStart;
	
        // child pointers
        if (leftSize <= 0) {
          // if left side has 0 points, current node turns into leaf
	  median = medianMemorized;
	  bool reachEnd = false;
	  if (median >= t_gridStart + t_numGrids - 1) {
	    reachEnd = true;
	  } else {
	    while (medianVal == m_grids->ReadGridVec(m_gridsSorted[median + 1], sortDim) ) {
	      median ++;
	      if (median >= t_gridStart + t_numGrids - 1) {
		reachEnd = true;
		break;
	      }
	    }
	    median ++;
	  }

	  // median node belong to the right child
	  // left: [start, median)
	  // right: [median, end)

	  // make sure new median is found, while loop may exit because there's no more points
	  if (!reachEnd) {
	    leftSize = median - t_gridStart;
	    rightSize = t_numGrids - median + t_gridStart;
	  } else {
	    leftSize = t_numGrids;
	    rightSize = 0;
	    median = -1;
	  }

	  if (rightSize <= 0) { // this dimension is homogeneous
	    // replace current level with the next
	    TreeConstruct(t_tree, t_gridStart, leftSize, t_depth + 1, t_minCoords, t_maxCoords, absStart);
	    return;
	  } else {
	    // right median works
	    medianVal = m_grids->ReadGridVec(m_gridsSorted[median], sortDim);
	    TreeWrite(t_tree, 2, g_nodeSize);
	    TreeWrite(t_tree, 3, t_tree[2] + GetMaxTreeSize(leftSize));
	  }
          
        } else {
          TreeWrite(t_tree, 2, g_nodeSize);
          TreeWrite(t_tree, 3, t_tree[2] + GetMaxTreeSize(leftSize));
        }
        
        // right child pointer
        for (intT d = 0; d < m_grids->GetDim(); ++ d) {
          TreeWrite(t_tree, 4 + d, t_minCoords[d]);
          TreeWrite(t_tree, 4 + m_grids->GetDim() + d, t_maxCoords[d]);
        }

        OtLongDataType *treeMinCoords = t_tree + 4;
        OtLongDataType *treeMaxCoords = t_tree + 4 + m_grids->GetDim();
        intT tmpDim = sortDim;

	OtLongDataType treeMaxCoordsCopyL[m_grids->GetDim()];
	OtLongDataType treeMinCoordsCopyL[m_grids->GetDim()];
	if(leftSize > 0) {
	  for (intT i = 0; i < m_grids->GetDim(); ++ i) {
	    treeMaxCoordsCopyL[i] = treeMaxCoords[i];
	    treeMinCoordsCopyL[i] = treeMinCoords[i];
	  }
	  treeMaxCoordsCopyL[tmpDim] = medianVal - 1;
	  if (leftSize > kdt_par_const) {
	    cilk_spawn TreeConstruct(t_tree + t_tree[2], t_gridStart, leftSize, t_depth + 1, treeMinCoordsCopyL, treeMaxCoordsCopyL, absStart + t_tree[2]);
	  } else {
	    cilk_spawn TreeConstructSequential(t_tree + t_tree[2], t_gridStart, leftSize, t_depth + 1, treeMinCoordsCopyL, treeMaxCoordsCopyL, absStart + t_tree[2]);
	  }
	}
    
	OtLongDataType treeMaxCoordsCopyR[m_grids->GetDim()];
	OtLongDataType treeMinCoordsCopyR[m_grids->GetDim()];
	for (intT i = 0; i < m_grids->GetDim(); ++ i) {
	  treeMaxCoordsCopyR[i] = treeMaxCoords[i];
	  treeMinCoordsCopyR[i] = treeMinCoords[i];
	}
	treeMinCoordsCopyR[tmpDim] = medianVal;
	if (rightSize > kdt_par_const) {
	  TreeConstruct(t_tree + t_tree[3], median, rightSize, t_depth + 1, treeMinCoordsCopyR, treeMaxCoordsCopyR, absStart + t_tree[3]);
	} else {
	  TreeConstructSequential(t_tree + t_tree[3], median, rightSize, t_depth + 1, treeMinCoordsCopyR, treeMaxCoordsCopyR, absStart + t_tree[3]);
	}
	if(leftSize > 0) cilk_sync;
      }
    }
  }

  void TreeConstructSequential(OtLongDataType *t_tree, GdLongDataType t_gridStart, GdLongDataType t_numGrids, OtLongDataType t_depth, OtLongDataType *t_minCoords, OtLongDataType *t_maxCoords, OtLongDataType absStart) {

    if (t_numGrids < 1) { // terminates
      cout << "reaching kdt construction with < 1 points"; exit(1);
    } else {

      TreeWrite(t_tree, 0, t_gridStart);
      TreeWrite(t_tree, 1, t_gridStart + t_numGrids);
      
      // if tree size 1, no need more children
      if (t_numGrids <= KDT_LEAF_THRESHOLD) {
        TreeWrite(t_tree, 2, LEAF);
      } else {

        intT sortDim = t_depth % m_grids->GetDim();

        sampleSort(m_gridsSorted + t_gridStart, t_numGrids, gridDimComparator(m_grids, sortDim));

        GdLongDataType median = t_gridStart + static_cast<GdLongDataType>(t_numGrids / 2); // starting offset for right
        GdLongDataType medianVal = m_grids->ReadGridVec(m_gridsSorted[median], sortDim);
	GdLongDataType medianMemorized = median;
	
        // might have duplicate medians
        while ( median - 1 >= 0 && medianVal == m_grids->ReadGridVec(m_gridsSorted[median - 1], sortDim) ) {
          median --;
        }

        // median node belong to the right child
        // left: [start, median)
        // right: [median, end)

        // left size could be zero
        GdLongDataType leftSize = median - t_gridStart;
        GdLongDataType rightSize = t_numGrids - median + t_gridStart;

        // child pointers
        if (leftSize <= 0) {
          // if left side has 0 points, current node turns into leaf
	  median = medianMemorized;
	  bool reachEnd = false;
	  if (median >= t_gridStart + t_numGrids - 1) {
	    reachEnd = true;
	  } else {
	    while (medianVal == m_grids->ReadGridVec(m_gridsSorted[median + 1], sortDim) ) {
	      median ++;
	      if (median >= t_gridStart + t_numGrids - 1) {
		reachEnd = true;
		break;
	      }
	    }
	    median ++;
	  }

	  // median node belong to the right child
	  // left: [start, median)
	  // right: [median, end)

	  // make sure new median is found, while loop may exit because there's no more points
	  if (!reachEnd) {
	    leftSize = median - t_gridStart;
	    rightSize = t_numGrids - median + t_gridStart;
	  } else {
	    leftSize = t_numGrids;
	    rightSize = 0;
	    median = -1;
	  }

	  if (rightSize <= 0) { // this dimension is homogeneous
	    // replace current level with the next
	    TreeConstruct(t_tree, t_gridStart, leftSize, t_depth + 1, t_minCoords, t_maxCoords, absStart);
	    return;
	  } else {
	    // right median works
	    medianVal = m_grids->ReadGridVec(m_gridsSorted[median], sortDim);
	    TreeWrite(t_tree, 2, g_nodeSize);
	    TreeWrite(t_tree, 3, t_tree[2] + GetMaxTreeSize(leftSize));
	  }
          
        } else {
          TreeWrite(t_tree, 2, g_nodeSize);
          TreeWrite(t_tree, 3, t_tree[2] + GetMaxTreeSize(leftSize));
        }
        
        // right child pointer
        for (intT d = 0; d < m_grids->GetDim(); ++ d) {
          TreeWrite(t_tree, 4 + d, t_minCoords[d]);
          TreeWrite(t_tree, 4 + m_grids->GetDim() + d, t_maxCoords[d]);
        }

        OtLongDataType *treeMinCoords = t_tree + 4;
        OtLongDataType *treeMaxCoords = t_tree + 4 + m_grids->GetDim();
        intT tmpDim = sortDim;

        for (intT s = 0; s < 2; ++ s) {
          if (s == 0 && leftSize > 0) {
            OtLongDataType tmpLeft = treeMaxCoords[tmpDim];
            treeMaxCoords[tmpDim] = medianVal - 1;
            TreeConstructSequential(t_tree + t_tree[2], t_gridStart, leftSize, t_depth + 1, treeMinCoords, treeMaxCoords, absStart + t_tree[2]); // can just pass tmp, work in parallel
            treeMaxCoords[tmpDim] = tmpLeft;
          } else if (s == 1) {
            OtLongDataType tmpRight = treeMinCoords[tmpDim];
            treeMinCoords[tmpDim] = medianVal;
            TreeConstructSequential(t_tree + t_tree[3], median, rightSize, t_depth + 1, treeMinCoords, treeMaxCoords, absStart + t_tree[3]);
            treeMinCoords[tmpDim] = tmpRight;
          }
        }

      }
    }
  }

  void TreeRangeQuery(OtLongDataType *t_tree, OtLongDataType *t_min, OtLongDataType *t_max, NbrContainer *t_results) {
    // leaf
    if (t_tree[2] == LEAF) {
            
      PtLongDataType ptVec[m_grids->GetVecLen()];
      for (OtLongDataType p = t_tree[0]; p < t_tree[1]; ++ p) {
	m_grids->GetGridVec(m_gridsSorted[p], ptVec);
	bool ptDisjoint = IsDisjoint(t_max, t_min, ptVec, ptVec, m_grids->GetDim());

	if (!ptDisjoint) {
	  t_results->push_back(m_gridsSorted[p]);
	}
      }
      
      return;
    }

    // more than 1 point, has range data
    // the range of the current grid being searched
    OtLongDataType *myMin = t_tree + 4;
    OtLongDataType *myMax = t_tree + 4 + m_grids->GetDim();
    intT rel = GridRelation(t_max, t_min, myMax, myMin, m_grids->GetDim());

    if (rel == GRID_INCLUDE) {
      // add all entries
      for (GdLongDataType i = t_tree[0]; i < t_tree[1]; ++ i) {
	t_results->push_back(m_gridsSorted[i]);
      }
    } else if (rel == GRID_DISJOINT) {
      // do nothing
    } else { // GRID_INTERSECT
      // search recursive
      TreeRangeQuery(t_tree + t_tree[2], t_min, t_max, t_results);
      TreeRangeQuery(t_tree + t_tree[3], t_min, t_max, t_results);
      return;
    }
    return;
  }

  ~GridKDTree() {
    free(m_gridsSorted);
    free(m_tree);
  }

};

#endif
