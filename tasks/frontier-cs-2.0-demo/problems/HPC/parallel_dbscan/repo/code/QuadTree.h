#include "DBSCAN.h"

#if !defined(QUADTREE_H)// && !defined(LONG_VARIABLE)
#define QUADTREE_H

#include <vector>
#include <algorithm>
#include <iostream>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <numeric>

#include "DBSCANGeometry.h"
#include "geometry.h"
#include "Grids.h"
#include "blockRadixSort.h"

/* the quadtree supports double datatype with int/long integer types
levelsize pointers, start offset, end offset, radius, center coordinates
double+int: levelsize*int + int + int + double + double*dim
doublt+long: levelsize*ot + ot + ot + double + double*dim

 */

#define APPROX_USE_LARGER_LEAF
#define APPROX_QUADTREE_LEAF_THRESHOLD 32 // <= which must be a leaf

#define EMPTY_NODE -1
#define EMPTY_LEAF -2
#define NONEMPTY_LEAF -3
#define LARGE_LEAF -6 // not b/c size small, but too few points
#define ILLEGAL -4
#define DUMMY -5

template<class IN, class OUT>
  inline OUT *PointerCast(IN *in_pointer) {
  void *tmp = (void *) in_pointer;
  return (OUT *) tmp;
}


// return which quadrant the point lies in inside t_box
// in the 2d case, return 0,1,2,3 for sw,nw,se,ne
inline intT GetQuadrant(pointNd t_pt, PtIntermediateDataType *t_centerCoords) {
  intT q = 0;
  for (intT d = 0; d < t_pt.m_dim; ++ d) {
    q |= (t_pt.m_data[d] > t_centerCoords[d]) << d;
  }
  return q;
}

// given a grid index t_g \in [0,levelSize), and the current grid center coordinate, calculate the new grid center that is part of the larger grid
inline void GetCenterCoords(intT t_dim, OtLongDataType t_g, PtIntermediateDataType t_gridSizeHalved, PtIntermediateDataType *t_centerCoordsOld, PtIntermediateDataType *t_centerCoords) {
  for (intT d = 0; d < t_dim; ++ d) {
    bool direction = (t_g >> d) & (1u);
    if (direction) {
      t_centerCoords[d] = t_centerCoordsOld[d] + t_gridSizeHalved / 2;
    } else {
      t_centerCoords[d] = t_centerCoordsOld[d] - t_gridSizeHalved / 2;
    }
  }
}

struct QuadTree {

private:

  /*
    inline void WriteCheck(intT *t_array, intT t_idx, intT t_value, intT t_depth, intT t_abs) {
    if (t_array[t_idx] != ILLEGAL) { // contains valid value
    for (intT i = 0; i < t_depth; ++ i) {
    cout << ' ';
    }
    cout << "QuadTree: overwriting value @ abs " << t_abs + t_idx << endl;
    exit(1);
    t_array[t_idx] = t_value;
    }
    }*/

#if defined(LONG_VARIABLE)

  // in unit of integers
  inline intT GetNodeSize(intT t_levelSize, intT t_dim) {
    return (t_levelSize*2 + 4) + (2 + t_dim * 2);
  }

  inline OtLongDataType GetNodeStartPtOffset(intT *t_Q, intT t_levelSize) {
    OtLongDataType *result = PointerCast<intT, OtLongDataType>(&t_Q[t_levelSize*2]);
    return result[0];
  }

  inline void WriteNodeStartPtOffset(intT *t_Q, intT t_levelSize, OtLongDataType t_startOffset) {
    OtLongDataType* writeLoc = PointerCast<intT, OtLongDataType>(&t_Q[t_levelSize*2]);
    writeLoc[0] = t_startOffset;
  }

  inline OtLongDataType GetNodeEndPtOffset(intT *t_Q, intT t_levelSize) {
    OtLongDataType *result = PointerCast<intT, OtLongDataType>(&t_Q[t_levelSize*2+2]);
    return result[0];
  }
  
  inline void WriteNodeEndPtOffset(intT *t_Q, intT t_levelSize, OtLongDataType t_endOffset) {
    OtLongDataType* writeLoc =  PointerCast<intT, OtLongDataType>(&t_Q[t_levelSize*2+2]);
    writeLoc[0] = t_endOffset;
  }

  // can be used for both reads and writes
  inline PtIntermediateDataType *GetNodeRadius(intT *t_Q, intT t_levelSize) {
    return PointerCast<intT, PtIntermediateDataType>(&t_Q[t_levelSize*2+4]);
  }

  // can be used for both reads and writes
  inline PtIntermediateDataType *GetNodeCenterCoord(intT *t_Q, intT t_levelSize) {
    return PointerCast<intT, PtIntermediateDataType>(&t_Q[t_levelSize*2+6]);
  }

#else // if not defined long variable

    // in unit of integers
  inline intT GetNodeSize(intT t_levelSize, intT t_dim) {
    return (t_levelSize + 2) + (2 + t_dim * 2);
  }
  
  inline OtLongDataType GetNodeStartPtOffset(intT *t_Q, intT t_levelSize) {
    OtLongDataType *result = PointerCast<intT, OtLongDataType>(&t_Q[t_levelSize]);
    return result[0];
  }
  
  inline void WriteNodeStartPtOffset(intT *t_Q, intT t_levelSize, OtLongDataType t_startOffset) {
    OtLongDataType *writeLoc = PointerCast<intT, OtLongDataType>(&t_Q[t_levelSize]);
    writeLoc[0] = t_startOffset;
  }

  inline OtLongDataType GetNodeEndPtOffset(intT *t_Q, intT t_levelSize) {
    OtLongDataType *result = PointerCast<intT, OtLongDataType>(&t_Q[t_levelSize+1]);
    return result[0];
  }
  
  inline void WriteNodeEndPtOffset(intT *t_Q, intT t_levelSize, OtLongDataType t_endOffset) {
    OtLongDataType* writeLoc =  PointerCast<intT, OtLongDataType>(&t_Q[t_levelSize+1]);
    writeLoc[0] = t_endOffset;
  }
  
  // can be used for both reads and writes
  inline PtIntermediateDataType *GetNodeRadius(intT *t_Q, intT t_levelSize) {
    return PointerCast<intT, PtIntermediateDataType>(&t_Q[t_levelSize+2]);
  }

  // can be used for both reads and writes
  inline PtIntermediateDataType *GetNodeCenterCoord(intT *t_Q, intT t_levelSize) {
    return PointerCast<intT, PtIntermediateDataType>(&t_Q[t_levelSize+4]);
  }
  
#endif

  inline OtLongDataType GetTreeMaxSize(OtLongDataType t_numPts, intT t_dim) {
    if (t_numPts == 0) {
      return 0;
    }
    intT nodeSize = GetNodeSize(m_levelSize, t_dim);
    return (2 * t_numPts - 1) * nodeSize;
  }
  
  // for integer sort
  struct quadrantComparator {
    _seq<pointNd> *t_points;
    PtIntermediateDataType* t_centerCoords;
  quadrantComparator(_seq<pointNd> *_points, PtIntermediateDataType* _centerCoords) : t_points(_points), t_centerCoords(_centerCoords) {}
    inline intT operator() (const OtLongDataType& id) {
      return GetQuadrant(t_points->A[id],t_centerCoords);
    }
  };

  // returns max depth of tree
  void ConstructHelper(intT *Q, _seq<pointNd> *t_points, OtLongDataType *Points, OtLongDataType PointsOffset, PtIntermediateDataType t_threshold, PtIntermediateDataType t_gridSize, PtIntermediateDataType *t_centerCoords, OtLongDataType t_ptCount, OtLongDataType t_depth, OtLongDataType t_absIdx, OtLongDataType t_treeSize) {

    // todo random accesses from t_table into t_points
    // todo shuffle points such that same quadrant points are together
    // semisort
    // can make a new array of core points just for the tree construction part

    if (t_ptCount == 0) {
      cout << "count is 0!" << endl;
      exit(1);
    }

    OtLongDataType *childPointers = PointerCast<intT, OtLongDataType>(Q);
  
    intT dim = t_points->A[0].m_dim;
    intT nodeSize = GetNodeSize(m_levelSize, dim);
  
    if (t_ptCount <= APPROX_QUADTREE_LEAF_THRESHOLD) {
      // store identity as large leaf
      
      // no need to store child pointer
      //WriteCheck(Q, 0, LARGE_LEAF, t_depth, t_absIdx);
      childPointers[0] = LARGE_LEAF;
    
      // store point offsets
      //WriteCheck(Q, m_levelSize, PointsOffset, t_depth, t_absIdx);
      //WriteCheck(Q, m_levelSize + 1, PointsOffset + t_ptCount, t_depth, t_absIdx);
      WriteNodeStartPtOffset(Q, m_levelSize, PointsOffset);
      WriteNodeEndPtOffset(Q, m_levelSize, PointsOffset + t_ptCount);

      PtIntermediateDataType *nodeSizeHalvedEntry = GetNodeRadius(Q, m_levelSize);
      nodeSizeHalvedEntry[0] = t_gridSize / 2;
  
      // store grid coordinate
      PtIntermediateDataType *coordArray = GetNodeCenterCoord(Q, m_levelSize);
      for (intT d = 0; d < dim; ++ d) {
        coordArray[d] = t_centerCoords[d];
      }
      return;
    }
  
    //integer sort by quadrant number. this can be further optimized by
    //setting MAX_RADIX and BUCKETS in blockRadixSort.h to be equal to
    //dimension and 2^dimension. We can make a version of the file that
    //chooses MAX_RADIX and BUCKETS at compile time or run time based on
    //the dimension.
    OtLongDataType* bucketOffsets = newA(OtLongDataType,m_levelSize+1);
    intSort::iSort(Points + PointsOffset,bucketOffsets,t_ptCount,m_levelSize,quadrantComparator(t_points,t_centerCoords));
    bucketOffsets[m_levelSize] = t_ptCount;
  
    OtLongDataType gridSizesPrefix[m_levelSize + 1];//stack
    OtLongDataType numPtsAfter = 0;
    intT nonEmptyQuadrants = 0;
    intT lastNonEmptyG = -1;
    for (intT g = 0; g < m_levelSize; ++ g) {
      OtLongDataType myCount = bucketOffsets[g+1]-bucketOffsets[g];
      //H[g].count();
      numPtsAfter += myCount;
      if (myCount != 0) {
        nonEmptyQuadrants += 1;
        lastNonEmptyG = g;
      }
      gridSizesPrefix[g] = GetTreeMaxSize(myCount, dim);
    }

    if (numPtsAfter != t_ptCount) {
      cout << "point num mismatch" << endl;
      exit(1);
    }

    // when there is 0 or > 1 nonempty quadrants or when the next level grid size is too small, we cannot skip level, because it is either a terminal node or a dividing node
    if (nonEmptyQuadrants == 1 && t_gridSize / 2 >= t_threshold) {

      // SKIP LEVEL (compress)
    
      PtIntermediateDataType centerCoords[dim];//stack
      GetCenterCoords(dim, lastNonEmptyG, t_gridSize/2, t_centerCoords, centerCoords);
  
      ConstructHelper(Q, t_points, Points, PointsOffset, t_threshold, t_gridSize / 2, centerCoords, t_ptCount, t_depth + 1, t_absIdx, t_treeSize);
    
      return;
    }
  
    // NO SKIP LEVEL
#if defined(LONG_VARIABLE)
    OtLongDataType totalSize = sequence::prefixSumL<OtLongDataType>(gridSizesPrefix, 0, m_levelSize);
#else
    OtLongDataType totalSize = sequence::prefixSum<OtLongDataType>(gridSizesPrefix, 0, m_levelSize);
#endif
    gridSizesPrefix[m_levelSize] = totalSize; // todo

    // store point offsets todo, can remove if not large leaf
    //WriteCheck(Q, m_levelSize, PointsOffset, t_depth, t_absIdx);
    //WriteCheck(Q, m_levelSize + 1, PointsOffset + t_ptCount, t_depth, t_absIdx);
    WriteNodeStartPtOffset(Q, m_levelSize, PointsOffset);
    WriteNodeEndPtOffset(Q, m_levelSize, PointsOffset + t_ptCount);


    // store size (radius aka sideHalved)
    PtIntermediateDataType *nodeSizeHalvedEntry = GetNodeRadius(Q, m_levelSize);
    nodeSizeHalvedEntry[0] = t_gridSize / 2;
  
    // store grid coordinate
    PtIntermediateDataType *coordArray = GetNodeCenterCoord(Q, m_levelSize);
    for (intT d = 0; d < dim; ++ d) {
      coordArray[d] = t_centerCoords[d];
    }

    if (t_depth > 4) {
      // store child pointers
      for (intT g = 0; g < m_levelSize; ++ g) {
        if (t_gridSize / 2 < t_threshold) { // basecase
          OtLongDataType quadrantCount = bucketOffsets[g+1]-bucketOffsets[g];
          if (quadrantCount == 0) {
    //WriteCheck(Q, g, EMPTY_LEAF, t_depth, t_absIdx);
	    childPointers[g] = EMPTY_LEAF;
          } else {
    //WriteCheck(Q, g, NONEMPTY_LEAF, t_depth, t_absIdx);
	    childPointers[g] = NONEMPTY_LEAF;
          }
        } else { // recursive case
          OtLongDataType quadrantCount = bucketOffsets[g+1]-bucketOffsets[g];
          if (quadrantCount == 0) {
    //WriteCheck(Q, g, EMPTY_NODE, t_depth, t_absIdx);
	    childPointers[g] = EMPTY_NODE;
          } else {
            OtLongDataType tmp = nodeSize + gridSizesPrefix[g]; // pt offset
            //WriteCheck(Q, g, tmp, t_depth, t_absIdx);
	    childPointers[g] = tmp;
            PtIntermediateDataType centerCoords[dim];//stack
            GetCenterCoords(dim, g, t_gridSize/2, t_centerCoords, centerCoords);
            ConstructHelper(Q + tmp, t_points, Points, PointsOffset+ bucketOffsets[g], t_threshold, t_gridSize / 2, centerCoords, quadrantCount, t_depth + 1, t_absIdx + tmp, gridSizesPrefix[g+1] - gridSizesPrefix[g]);
          }
        }
      }
    } else { // level 0, parallelize one more level below grids
      parallel_for_1 (intT g = 0; g < m_levelSize; ++ g) {
        if (t_gridSize / 2 < t_threshold) { // basecase
          OtLongDataType quadrantCount = bucketOffsets[g+1]-bucketOffsets[g];
          if (quadrantCount == 0) {
    //            WriteCheck(Q, g, EMPTY_LEAF, t_depth, t_absIdx);
	    childPointers[g] = EMPTY_LEAF;
          } else {
    //            WriteCheck(Q, g, NONEMPTY_LEAF, t_depth, t_absIdx);
	    childPointers[g] = NONEMPTY_LEAF;
          }
        } else { // recursive case
          OtLongDataType quadrantCount = bucketOffsets[g+1]-bucketOffsets[g];
          if (quadrantCount == 0) {
    //            WriteCheck(Q, g, EMPTY_NODE, t_depth, t_absIdx);
	    childPointers[g] = EMPTY_NODE;
          } else {
            OtLongDataType tmp = nodeSize + gridSizesPrefix[g]; // pt offset
	    //            WriteCheck(Q, g, tmp, t_depth, t_absIdx);
	    childPointers[g] = tmp;
            PtIntermediateDataType centerCoords[dim];//stack
            GetCenterCoords(dim, g, t_gridSize/2, t_centerCoords, centerCoords);
            ConstructHelper(Q + tmp, t_points, Points, PointsOffset+ bucketOffsets[g], t_threshold, t_gridSize / 2, centerCoords, quadrantCount, t_depth + 1, t_absIdx + tmp, gridSizesPrefix[g+1] - gridSizesPrefix[g]);
          }
        }
      }
    }

    free(bucketOffsets);
    //  free(gridSizesPrefix);
  }

  bool BinApproxRangeQueryHelper(PtDataType *t_pointsData, PtDataType *t_pointData, intT t_dim, OtLongDataType t_treeNodeIdx, PtIntermediateDataType t_radiusSqr1, PtIntermediateDataType t_radiusSqr2, intT *Q) {

    OtLongDataType *childPointers = PointerCast<intT, OtLongDataType>(Q);
    
    if (t_treeNodeIdx == EMPTY_NODE || t_treeNodeIdx == EMPTY_LEAF) {
      return false;
    }

    // NOT EMPTY, CHECK
    PtIntermediateDataType *t_centerCoords = GetNodeCenterCoord(Q, m_levelSize);
    PtIntermediateDataType *t_sideHalved = GetNodeRadius(Q, m_levelSize); // todo dangerous, return ptintdatype
    intT intersect1 = GridCircleIntersect(t_pointData, t_radiusSqr1, t_centerCoords, t_sideHalved[0], t_dim);
    intT intersect2 = GridCircleIntersect(t_pointData, t_radiusSqr2, t_centerCoords, t_sideHalved[0], t_dim);

    if (intersect1 == DISJOINT) {
      return false;
    } else if (intersect2 == COVERED) {
      return true;
    } else {  // INTERSECT
      if (t_treeNodeIdx == NONEMPTY_LEAF) {
        if (intersect1 == INTERSECT) {
          return true;
        } else {
          return false;
        }
      } else if (childPointers[0] == LARGE_LEAF) {
        for (OtLongDataType p = GetNodeStartPtOffset(Q, m_levelSize); p < GetNodeEndPtOffset(Q, m_levelSize); ++ p) {
          if (PointDistNDSqrLeq(t_pointsData + m_points[p] * t_dim, t_pointData, t_dim,  t_radiusSqr2)) {
            return true;
          }
        }
        return false;
      } else { // recursive case
        bool result = false;
        for (intT g = 0; g < m_levelSize;  ++ g) {
          PtIntermediateDataType centerCoords[t_dim];//stack
          GetCenterCoords(t_dim, g, t_sideHalved[0], t_centerCoords, centerCoords);
          result |= BinApproxRangeQueryHelper(t_pointsData, t_pointData, t_dim, childPointers[g], t_radiusSqr1, t_radiusSqr2, Q + childPointers[g]);
          if (result) {
            return result;
          }
        }
        return result;
      }
    }
  }

  // for precise range query
  OtLongDataType RangeQueryHelper(PtDataType *t_pointsData, PtDataType *t_pointData, intT t_dim, OtLongDataType t_treeNodeIdx, PtIntermediateDataType t_radiusSqr1, intT *Q) {
  
    if (t_treeNodeIdx == EMPTY_NODE || t_treeNodeIdx == EMPTY_LEAF) {
      return 0;
    }

    OtLongDataType *childPointers = PointerCast<intT, OtLongDataType>(Q);

    // NOT EMPTY, CHECK
    PtIntermediateDataType *t_centerCoords = GetNodeCenterCoord(Q, m_levelSize);
    PtIntermediateDataType *t_sideHalved = GetNodeRadius(Q, m_levelSize); // todo dangerous, return PtIntermediateDataType
    intT intersect1 = GridCircleIntersect(t_pointData, t_radiusSqr1, t_centerCoords, t_sideHalved[0], t_dim);
    OtLongDataType myTotalPts = GetNodeEndPtOffset(Q, m_levelSize) - GetNodeStartPtOffset(Q, m_levelSize);

    if (intersect1 == DISJOINT) {
      return 0;
    } else if (intersect1 == COVERED) {
      return myTotalPts;
    } else {  // INTERSECT
      OtLongDataType myCount = 0;
      if (t_treeNodeIdx == NONEMPTY_LEAF) {
  if (intersect1 == INTERSECT) {
    return myTotalPts;
  } else {
    return 0;
  }
      } else if (childPointers[0] == LARGE_LEAF) {
  for (OtLongDataType p = GetNodeStartPtOffset(Q, m_levelSize); p < GetNodeEndPtOffset(Q, m_levelSize); ++ p) {
    if (PointDistNDSqrLeq(t_pointsData + m_points[p] * t_dim, t_pointData, t_dim, t_radiusSqr1)) {
      myCount ++;
    }
  }
  return myCount;
      } else { // recursive case
  OtLongDataType result = 0;
  for (intT g = 0; g < m_levelSize;  ++ g) {
    PtIntermediateDataType centerCoords[t_dim];//stack
    GetCenterCoords(t_dim, g, t_sideHalved[0], t_centerCoords, centerCoords);
    result += RangeQueryHelper(t_pointsData, t_pointData, t_dim, childPointers[g], t_radiusSqr1, Q + childPointers[g]);
  }
  return result;
      }
    }
  }

  bool BinRangeQueryHelper(PtDataType *t_pointsData, PtDataType *t_pointData, intT t_dim, OtLongDataType t_treeNodeIdx, PtIntermediateDataType t_radiusSqr1, intT *Q) {
  
    if (t_treeNodeIdx == EMPTY_NODE || t_treeNodeIdx == EMPTY_LEAF) {
      return false;
    }

    OtLongDataType *childPointers = PointerCast<intT, OtLongDataType>(Q);
    
    // NOT EMPTY, CHECK
    PtIntermediateDataType *t_centerCoords = GetNodeCenterCoord(Q, m_levelSize);
    PtIntermediateDataType *t_sideHalved = GetNodeRadius(Q, m_levelSize); // todo dangerous, return PtIntermediateDataType
    intT intersect1 = GridCircleIntersect(t_pointData, t_radiusSqr1, t_centerCoords, t_sideHalved[0], t_dim);

    if (intersect1 == DISJOINT) {
      return false;
    } else if (intersect1 == COVERED) {
      return true;
    } else {  // INTERSECT
      if (t_treeNodeIdx == NONEMPTY_LEAF) {
  if (intersect1 == INTERSECT) {
    return true;
  } else {
    return false;
  }
      } else if (childPointers[0] == LARGE_LEAF) {
  for (OtLongDataType p = GetNodeStartPtOffset(Q, m_levelSize); p < GetNodeEndPtOffset(Q, m_levelSize); ++ p) {
    if (PointDistNDSqrLeq(t_pointsData + m_points[p] * t_dim, t_pointData, t_dim,  t_radiusSqr1)) {
      return true;
    }
  }
  return false;
      } else { // recursive case
  bool result = false;
  for (intT g = 0; g < m_levelSize;  ++ g) {
    PtIntermediateDataType centerCoords[t_dim];//stack
    GetCenterCoords(t_dim, g, t_sideHalved[0], t_centerCoords, centerCoords);
    result |= BinRangeQueryHelper(t_pointsData, t_pointData, t_dim, childPointers[g], t_radiusSqr1, Q + childPointers[g]);
    if (result) {
      return result;
    }
  }
  return result;
      }
    }
  }

  OtLongDataType *m_points;
  intT *m_quadTree;
  OtLongDataType m_quadTreeLen;
  intT m_levelSize;
  intT m_dim;
  bool m_needTree;

public:
  
  // for core points
  QuadTree(_seq<pointNd> *t_points, DbscanParams *t_params, Grids *t_grids, pair<PtLongDataType, intT *> corePointMarkersGrid, bool* isCoreGrid) : m_needTree(true) {

    //    timer t1;
    //    t1.start();

    if (corePointMarkersGrid.first <= 0) {
      m_needTree = false;
      /*
      PrintParser("timer1", 0);
      PrintParser("timer2", 0);
      PrintParser("timer3", 0);
      PrintParser("timer4", 0);
      PrintParser("timer5", 0);
      PrintParser("timer6", 0);
      */
      return;
    }

    intT dim = t_points->A[0].m_dim;
    m_dim = dim;
    m_levelSize = pow(2, dim);
    
    // TODO theoretically there is no need to init the array, but for debugging might want to do so
    PtIntermediateDataType threshold = 0.5 * t_params->m_epsilon * t_params->m_rho / t_grids->GetDimSqrt(); // todo 0.5
  
    OtLongDataType Qlength = GetTreeMaxSize(t_points->n, dim) + t_grids->GetNumGrids();
    m_quadTreeLen = Qlength;

    PtIntermediateDataType *Q_type = static_cast<PtIntermediateDataType *>(malloc(sizeof(PtIntermediateDataType) * Qlength/2 )); // for alignment purpose
    void *Q_tmp = (void *)Q_type;
    intT *Q = (intT *)Q_tmp;
    //    parallel_for (OtLongDataType i = 0; i < Qlength; ++ i) {
    //      Q[i] = ILLEGAL;
    //    }

    //    PrintParser("timer1", t1.next());
  
    // core point markers in correct order, later need to do reduction and prefix sum
    // we need a buffer for prefix sum anyways
    intT *corePointMarkers = (intT *)malloc(sizeof(intT) * t_points->n);
    parallel_for(OtLongDataType g = 0; g < t_grids->GetNumGrids(); ++g) {
      if(isCoreGrid[g]) {
        granular_for (p, t_grids->GetStartOffset(g), t_grids->GetStartOffset(g) + t_grids->GetNumPts(g), 2000, {
          corePointMarkers[p] = corePointMarkersGrid.second[t_grids->GetPt(p)];
        });
      }
    }

    //can parallelize this by counting in parallel using reduce per-grid
    OtLongDataType* PointCounts = newA(OtLongDataType,t_grids->GetNumGrids()+1);
    parallel_for(OtLongDataType g = 0; g < t_grids->GetNumGrids(); ++g) {
      PointCounts[g] = 0;
      if(isCoreGrid[g]) {

#if defined(LONG_VARIABLE)
        PointCounts[g] = sequence::prefixSumL<intT>(corePointMarkers, t_grids->GetStartOffset(g), t_grids->GetStartOffset(g) + t_grids->GetNumPts(g));
#else
        PointCounts[g] = sequence::prefixSum<intT>(corePointMarkers, t_grids->GetStartOffset(g), t_grids->GetStartOffset(g) + t_grids->GetNumPts(g));
#endif

	
      }
    }
    
    //    PrintParser("timer2", t1.next());
    OtLongDataType totalPoints = sequence::plusScan(PointCounts,PointCounts,t_grids->GetNumGrids());
    PointCounts[t_grids->GetNumGrids()] = totalPoints;
    //    PrintParser("timer3", t1.next());

    //can parallelize this by filtering points per core grid in parallel
    m_points = newA(OtLongDataType,totalPoints);
    parallel_for(OtLongDataType g = 0; g < t_grids->GetNumGrids(); ++g) {
      if(isCoreGrid[g]) {
        granular_for (p, t_grids->GetStartOffset(g), t_grids->GetStartOffset(g) + t_grids->GetNumPts(g) - 1, 2000, {
          if (corePointMarkers[p] != corePointMarkers[p + 1]) {
            m_points[PointCounts[g]+corePointMarkers[p]] = t_grids->GetPt(p);
          }
        });
        OtLongDataType p = t_grids->GetStartOffset(g) + t_grids->GetNumPts(g) - 1;
        if (corePointMarkers[p] != PointCounts[g+1]-PointCounts[g]) {
          m_points[PointCounts[g]+corePointMarkers[p]] = t_grids->GetPt(p);
        }
      }
    }
    //    PrintParser("timer4", t1.next());

    //OtLongDataType gridSizesPrefix[t_grids->GetNumGrids() + 1];//stack
    OtLongDataType *gridSizesPrefix = newA(OtLongDataType, t_grids->GetNumGrids() + 1);
    parallel_for (OtLongDataType g = 0; g < t_grids->GetNumGrids(); ++ g) {
      gridSizesPrefix[g] = GetTreeMaxSize(PointCounts[g+1]-PointCounts[g], dim);
    }

#if defined(LONG_VARIABLE)
    OtLongDataType totalSize = sequence::prefixSumL<OtLongDataType>(gridSizesPrefix, 0, t_grids->GetNumGrids());
#else
    OtLongDataType totalSize = sequence::prefixSum<OtLongDataType>(gridSizesPrefix, 0, t_grids->GetNumGrids());
#endif
    gridSizesPrefix[t_grids->GetNumGrids()] = totalSize;

    //    PrintParser("timer5", t1.next());
    OtLongDataType *childPointers = PointerCast<intT, OtLongDataType>(Q);
    parallel_for_1 (OtLongDataType g = 0; g < t_grids->GetNumGrids(); ++ g) {

      if (PointCounts[g+1]-PointCounts[g] == 0) {
        childPointers[g] = EMPTY_NODE;
      } else {

#if defined(LONG_VARIABLE)
	childPointers[g] = t_grids->GetNumGrids()*2 + gridSizesPrefix[g] + GetNodeSize(m_levelSize, dim);
#else
	childPointers[g] = t_grids->GetNumGrids() + gridSizesPrefix[g] + GetNodeSize(m_levelSize, dim);
#endif

	OtLongDataType gridIdx[t_grids->GetVecLen()]; // todo in parallel separate copy //stack
	// trying to get grid idx
	t_grids->GetGridVec(g, gridIdx);
          
	PtIntermediateDataType centerCoords[dim];//stack
	for (intT d = 0; d < dim; ++ d) {
	  centerCoords[d] = t_grids->GetLowCoord(d) + gridIdx[d] * t_grids->GetSize() + t_grids->GetSize() / 2;
	}

	ConstructHelper(Q + childPointers[g], t_points, m_points, PointCounts[g], threshold, t_grids->GetSize(), centerCoords, PointCounts[g+1]-PointCounts[g], 0, childPointers[g], gridSizesPrefix[g + 1] - gridSizesPrefix[g]);
      }

    }

    //    PrintParser("timer6", t1.stop());

    free(gridSizesPrefix);
    free(PointCounts);
    free(corePointMarkers);
    m_quadTree = Q;

  }

  // for all points
  QuadTree(_seq<pointNd> *t_points, DbscanParams *t_params, Grids *t_grids) : m_needTree(true) {

    if (t_points->n <= 0) {
      m_needTree = false;
      return;
    }

    intT dim = t_points->A[0].m_dim;
    m_levelSize = pow(2, dim);
    // TODO theoretically there is no need to init the array, but for debugging might want to do so
    PtIntermediateDataType threshold = 0.5 * t_params->m_epsilon * t_params->m_rho / t_grids->GetDimSqrt(); // todo 0.5
  
    OtLongDataType Qlength = GetTreeMaxSize(t_points->n, dim) + t_grids->GetNumGrids();
    m_quadTreeLen = Qlength;
    
    PtIntermediateDataType *Q_type = static_cast<PtIntermediateDataType *>(malloc(sizeof(PtIntermediateDataType) * Qlength/2 ));
    void *Q_tmp = (void *)Q_type;
    intT *Q = (intT *)Q_tmp;

    /*
    parallel_for (OtLongDataType i = 0; i < Qlength; ++ i) {
      Q[i] = ILLEGAL;
    }
    */

    m_points = (OtLongDataType *) malloc(sizeof(OtLongDataType) * t_points->n);
    parallel_for(OtLongDataType p = 0; p < t_points->n; ++p) {
      m_points[p] = t_grids->GetPt(p);
    }
    
    //OtLongDataType gridNumPtsPrefix[t_grids->GetNumGrids() + 1];//stack
    //OtLongDataType gridSizesPrefix[t_grids->GetNumGrids() + 1];//stack
    OtLongDataType *gridNumPtsPrefix = newA(OtLongDataType, t_grids->GetNumGrids() + 1);
    OtLongDataType *gridSizesPrefix = newA(OtLongDataType, t_grids->GetNumGrids() + 1);
    parallel_for (OtLongDataType g = 0; g < t_grids->GetNumGrids(); ++ g) {
      gridSizesPrefix[g] = GetTreeMaxSize(t_grids->GetNumPts(g), dim);
      gridNumPtsPrefix[g] = t_grids->GetNumPts(g);
    }

#if defined(LONG_VARIABLE)
    OtLongDataType totalSize = sequence::prefixSumL<OtLongDataType>(gridSizesPrefix, 0, t_grids->GetNumGrids());
#else
    OtLongDataType totalSize = sequence::prefixSum<OtLongDataType>(gridSizesPrefix, 0, t_grids->GetNumGrids());
#endif
    gridSizesPrefix[t_grids->GetNumGrids()] = totalSize;

#if defined(LONG_VARIABLE)
    OtLongDataType totalPts = sequence::prefixSumL<OtLongDataType>(gridNumPtsPrefix, 0, t_grids->GetNumGrids());
#else
    OtLongDataType totalPts = sequence::prefixSum<OtLongDataType>(gridNumPtsPrefix, 0, t_grids->GetNumGrids());
#endif
    gridNumPtsPrefix[t_grids->GetNumGrids()] = totalPts;

  
    OtLongDataType nonEmptyLeafCount = 0;
    
    OtLongDataType *childPointers = PointerCast<intT, OtLongDataType>(Q);
    parallel_for (OtLongDataType g = 0; g < t_grids->GetNumGrids(); ++ g) {

      if (t_grids->GetNumPts(g) == 0) {
        Q[g] = EMPTY_NODE;
      } else {

#if defined(LONG_VARIABLE)
	childPointers[g] = t_grids->GetNumGrids()*2 + gridSizesPrefix[g] + GetNodeSize(m_levelSize, dim);
#else
	childPointers[g] = t_grids->GetNumGrids() + gridSizesPrefix[g] + GetNodeSize(m_levelSize, dim);
#endif

        OtLongDataType gridIdx[t_grids->GetVecLen()]; // todo in parallel separate copy //stack
        // trying to get grid idx
        t_grids->GetGridVec(g, gridIdx);
        PtIntermediateDataType centerCoords[dim];//stack
        for (intT d = 0; d < dim; ++ d) {
          centerCoords[d] = t_grids->GetLowCoord(d) + gridIdx[d] * t_grids->GetSize() + t_grids->GetSize() / 2;
        }
        double t_time1 = 0;
        double t_time2 = 0;
        ConstructHelper(Q + childPointers[g], t_points, m_points, gridNumPtsPrefix[g], threshold, t_grids->GetSize(), centerCoords, t_grids->GetNumPts(g), 0, childPointers[g], gridSizesPrefix[g + 1] - gridSizesPrefix[g]);
      }
    }

    //cout << " quadtree avg leafsize " << (double)t_points->n / nonEmptyLeafCount << endl;
  
    m_quadTree = Q;
    free(gridNumPtsPrefix);
    free(gridSizesPrefix);

  }
  
  ~QuadTree() {
    if (m_needTree) {
      free(m_quadTree);
      free(m_points);      
    }
  }

  // for approx connectivity, may want to add core point flag
  bool BinApproxRangeQuery(PtDataType *t_pointsData, PtDataType *t_pointData, intT t_dim, Grids *t_grids, OtLongDataType t_g, PtIntermediateDataType t_radiusSqr1, PtIntermediateDataType t_radiusSqr2) {
    if (!m_needTree) {
      return false;
    }
      
    if (t_radiusSqr1 >= t_radiusSqr2) {
      cout << "error: has core point in range must have radiusSqr1 < radiusSqr2, exiting" << endl;
    }

    OtLongDataType *childPointers = PointerCast<intT, OtLongDataType>(m_quadTree);
    return BinApproxRangeQueryHelper(t_pointsData, t_pointData, t_dim, childPointers[t_g], t_radiusSqr1, t_radiusSqr2, m_quadTree + childPointers[t_g]);
  }

  // for exact markcore
  OtLongDataType RangeQuery(PtDataType *t_pointsData, PtDataType *t_pointData, intT t_dim, Grids *t_grids, OtLongDataType t_g, PtIntermediateDataType t_radiusSqr1) {
    if (!m_needTree) {
      return 0;
    }
    OtLongDataType *childPointers = PointerCast<intT, OtLongDataType>(m_quadTree);
    return RangeQueryHelper(t_pointsData, t_pointData, t_dim, childPointers[t_g], t_radiusSqr1, m_quadTree + childPointers[t_g]);
  }

  // for exact connectivity, in comparison to range query, can have earlier cutoff
  bool BinRangeQuery(PtDataType *t_pointsData, PtDataType *t_pointData, intT t_dim, Grids *t_grids, OtLongDataType t_g, PtIntermediateDataType t_radiusSqr1) {
    if (!m_needTree) {
      return false;
    }
    OtLongDataType *childPointers = PointerCast<intT, OtLongDataType>(m_quadTree);
    return BinRangeQueryHelper(t_pointsData, t_pointData, t_dim, childPointers[t_g], t_radiusSqr1, m_quadTree + childPointers[t_g]);
  }

  inline OtLongDataType GetGridPtStartOffset(OtLongDataType t_g) {
    OtLongDataType *childPointers = PointerCast<intT, OtLongDataType>(m_quadTree);
    return GetNodeStartPtOffset(m_quadTree + childPointers[t_g], m_levelSize);
  }

  inline OtLongDataType GetGridPtEndOffset(OtLongDataType t_g) {
    OtLongDataType *childPointers = PointerCast<intT, OtLongDataType>(m_quadTree);
    return GetNodeEndPtOffset(m_quadTree + childPointers[t_g], m_levelSize);
  }

  inline OtLongDataType GetGridNumPts(OtLongDataType t_g) {
    OtLongDataType *childPointers = PointerCast<intT, OtLongDataType>(m_quadTree);
    return GetNodeEndPtOffset(m_quadTree + childPointers[t_g], m_levelSize) - GetNodeStartPtOffset(m_quadTree + childPointers[t_g], m_levelSize);
  }

  inline OtLongDataType GetPt(OtLongDataType t_offset) {
    return m_points[t_offset];
  }

};

#endif
