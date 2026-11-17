///////////////// CONFIGURATION ///////////////////
// The task retains the reproduced exact execution path.
#define OUR_EXACT
#define USE_BUCKETING
//////////////////////////////////////////////////

// Calling example for exact DBSCAN
// ./DBSCAN -eps 10 -minpts 10 -o clusters.txt dataset.txt

/* Input dataset example, 3 dimensional dataset of 5 points
dim 3
10 20 30
5 2 1
3 3 5
2 20 10
22 10 30
*/

/* Output file meaning
<point index (starting 0)> <cluster id>
same cluster id means belong to the same cluster
*/

/* For best performance (benchmarking)
 - disable LONG_VARIABLE
 - enable USE_JEMALLOC (see more details on github page)
 in DBSCAN.h
 */

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
#include "GetNeighborKDTree.h"
#include "Grids.h"
#include "unionFind.h"
#include "QuadTree.h"

//////////////// PREPROCESSORS ///////////////////
#ifdef USE_JEMALLOC
#include<jemalloc/jemalloc.h>
#include "je_allocator.h"
#endif

#if defined(OUR_EXACT)
#endif

struct pointDistComparator {
  pointDistComparator() {}
  inline intT operator() (const pair<PtIntermediateDataType, PtDataType*>& i, const pair<PtIntermediateDataType, PtDataType*>& j) {
    return i.first < j.first;
  }
};

inline bool PointGridDistanceLeq(intT dim, Grids *t_grids, GdLongDataType g, PtLongDataType p, PtIntermediateDataType t_threshold) {

  OtLongDataType gridDimNum[dim];
  t_grids->GetGridVec(g, gridDimNum);
  PtIntermediateDataType result = 0;
  PtIntermediateDataType dimSmall, dimLarge, pCoord, dimDist;
  for (intT d = 0; d < dim; ++ d) {
    dimSmall = gridDimNum[d] * t_grids->GetSize();
    dimLarge = (gridDimNum[d] + 1) * t_grids->GetSize();
    pCoord = t_grids->GetPtData(p)[d] - t_grids->GetLowCoord(d);
    dimDist = 0;

    if (pCoord >= dimSmall && pCoord <= dimLarge) {
      dimDist = 0;
    } else if (pCoord < dimSmall) {
      dimDist = dimSmall - pCoord;
    } else {
      dimDist = pCoord - dimLarge;
    }
    
    result += dimDist * dimDist;
  }
  
  if (result > t_threshold) {
    return false;
  }
  return true;
}

inline PtIntermediateDataType PointGridDistance(intT dim, Grids *t_grids, GdLongDataType g, PtLongDataType p) {

  OtLongDataType gridDimNum[dim];
  t_grids->GetGridVec(g, gridDimNum);
  PtIntermediateDataType result = 0;
  PtIntermediateDataType dimSmall, dimLarge, pCoord, dimDist;
  for (intT d = 0; d < dim; ++ d) {
    dimSmall = gridDimNum[d] * t_grids->GetSize();
    dimLarge = (gridDimNum[d] + 1) * t_grids->GetSize();
    pCoord = t_grids->GetPtData(p)[d] - t_grids->GetLowCoord(d);
    dimDist = 0;

    if (pCoord >= dimSmall && pCoord <= dimLarge) {
      dimDist = 0;
    } else if (pCoord < dimSmall) {
      dimDist = dimSmall - pCoord;
    } else {
      dimDist = pCoord - dimLarge;
    }
    
    result += dimDist * dimDist;
  }
  
  return result;
}

pair<PtLongDataType, intT *> MarkCorePointGridParallelND(_seq<pointNd> *t_points, DbscanParams *t_params, Grids *t_grids, bool *t_isCoreGrid) {
  intT *corePointMarkers = newA(intT,t_points->n);
  parallel_for (PtLongDataType p = 0; p < t_points->n; ++ p) {
    corePointMarkers[p] = 0;
  }

  intT dim = t_points->A[0].m_dim;
  PtDataType *pointsData = t_points->A[0].m_data;
  
  parallel_for_1 (GdLongDataType g = 0; g < t_grids->GetNumGrids(); ++ g) {
    t_isCoreGrid[g] = false;
    GdPtLongDataType gNumPts = t_grids->GetNumPts(g);

    if (gNumPts >= t_params->m_minPts) {

      t_isCoreGrid[g] = true;
      PtLongDataType iterEnd = t_grids->GetStartOffset(g) + gNumPts;
      granular_for_t (p, t_grids->GetStartOffset(g), iterEnd, 2000, PtLongDataType, {
        corePointMarkers[t_grids->GetPt(p)] = 1;
      });

    } else {
      
      NbrContainer *neighbors = GetNeighborGridsKDTree(g, t_grids);
      
      // compute potential more near points other than own grid by summing up that of
      // every neighboring grid, can just skip the loop when insufficient
      PtLongDataType potentialMoreNearPoints = 0;
      for (GdLongDataType ni = 0; ni < neighbors->size(); ++ ni) {
        GdLongDataType n = neighbors->at(ni);
        if (n < 0 || n == g) continue;
        potentialMoreNearPoints += t_grids->GetNumPts(n);
      }

      if (potentialMoreNearPoints >= t_params->m_minPts - gNumPts) {
        PtLongDataType iterEnd = t_grids->GetStartOffset(g) + gNumPts;
        parallel_for_1 (PtLongDataType p = t_grids->GetStartOffset(g); p < iterEnd; ++p) {
          PtLongDataType totalNearPoints = gNumPts;
          for (GdLongDataType ni = 0; ni < neighbors->size(); ++ ni) {
            GdLongDataType n = neighbors->at(ni);
            if (n < 0 || n == g) continue;

            GdPtLongDataType nNumPts = t_grids->GetNumPts(n);

            PtLongDataType iterEndInner = t_grids->GetStartOffset(n) + nNumPts;
            for (PtLongDataType np = t_grids->GetStartOffset(n); np < iterEndInner; ++np) {
              if (PointDistNDSqrLeq(t_grids->GetPtData(p), t_grids->GetPtData(np), dim, t_params->m_epsilon * t_params->m_epsilon)) { // todo opt comp cast
                totalNearPoints ++;
                if (totalNearPoints >= t_params->m_minPts) {
                  goto decide_core;
                }
              }
            }
          }
          decide_core:
          if (totalNearPoints >= t_params->m_minPts) {
            corePointMarkers[t_grids->GetPt(p)] = 1;
            t_isCoreGrid[g] = true;
          } else {
            corePointMarkers[t_grids->GetPt(p)] = 0;
          }
        } // for pts in g
      } // if has potentially enough points

    }
  }

  PtLongDataType numCore;
  if (sizeof(PtLongDataType) > 4) {
    numCore = sequence::reduceIL(corePointMarkers, (PtLongDataType)t_points->n, utils::addF<intT>());
  } else {
    numCore = sequence::reduce(corePointMarkers, (PtLongDataType)t_points->n, utils::addF<intT>());
  }
  return make_pair(numCore, corePointMarkers);
}

#ifdef QUADTREE_H
pair<PtLongDataType, intT *> MarkCorePointGridParallelND(_seq<pointNd> *t_points, DbscanParams *t_params, Grids *t_grids, QuadTree *t_allPtQuadTree, bool *t_isCoreGrid) {

  intT *corePointMarkers = newA(intT,t_points->n);
  parallel_for (PtLongDataType p = 0; p < t_points->n; ++ p) {
    corePointMarkers[p] = 0;
  }

  intT dim = t_points->A[0].m_dim;
  PtDataType *pointsData = t_points->A[0].m_data;
  
  parallel_for_1 (GdLongDataType g = 0; g < t_grids->GetNumGrids(); ++ g) {
    t_isCoreGrid[g] = false;
    GdPtLongDataType gNumPts = t_grids->GetNumPts(g);

    // If core grid, just mark all.
    if (gNumPts >= t_params->m_minPts) {

      t_isCoreGrid[g] = true;
      PtLongDataType iterEnd = t_grids->GetStartOffset(g) + gNumPts;
      granular_for_t (p, t_grids->GetStartOffset(g), iterEnd, 100, PtLongDataType, {
        corePointMarkers[t_grids->GetPt(p)] = 1;
      });

    } else {
      
      NbrContainer *neighbors = GetNeighborGridsKDTree(g, t_grids);

      PtLongDataType potentialMoreNearPoints = 0;
      for (GdLongDataType ni = 0; ni < neighbors->size(); ++ ni) {
        GdLongDataType n = neighbors->at(ni);
        if (n < 0 || n == g) continue;
        potentialMoreNearPoints += t_grids->GetNumPts(n);
      }
      
      if (potentialMoreNearPoints >= t_params->m_minPts - gNumPts) {
        // Iterate point p in grid.
        PtLongDataType iterEnd = t_grids->GetStartOffset(g) + gNumPts;
        parallel_for_1 (PtLongDataType p = t_grids->GetStartOffset(g); p < iterEnd; ++p) {
          PtLongDataType totalNearPoints = gNumPts;
          
  	      for (GdLongDataType ni = 0; ni < neighbors->size(); ++ ni) {
            GdLongDataType n = neighbors->at(ni);
            if (n < 0 || n == g) continue;
            GdPtLongDataType nNumPts = t_grids->GetNumPts(n);
        	  // instead of itering points in neighbor, do a range query
        	  totalNearPoints += t_allPtQuadTree->RangeQuery(pointsData, pointsData + dim * t_grids->GetPt(p), dim, t_grids, n, t_params->m_epsilon * t_params->m_epsilon);
        	  if (totalNearPoints >= t_params->m_minPts) {
        	    goto decide_core;
        	  }	  
          }
          decide_core:
          if (totalNearPoints >= t_params->m_minPts) {
            corePointMarkers[t_grids->GetPt(p)] = 1;
            // numCore ++;
            t_isCoreGrid[g] = true;
          } else {
            corePointMarkers[t_grids->GetPt(p)] = 0;
          }
        } // for pts in g
      } // if has potentially enough points
    }
  }
  
  PtLongDataType numCore = sequence::reduce(corePointMarkers, t_points->n, utils::addF<intT>());
  return make_pair(numCore, corePointMarkers);
}

#endif // ifdef quadtree h

// Public reconstruction interfaces. The upstream implementations and tuning
// have been removed.
bool edgeAddableBCPND(QuadTree *t_corePtQuadTree, PtDataType *pointsData,
    intT t_dim, DbscanParams *t_params, Grids *t_grids,
    intT *t_corePointMarkers, GdLongDataType t_g1, GdLongDataType t_g2);
bool edgeAddableBCPND(PtDataType *pointsData, intT dim,
    PtIntermediateDataType t_radiusSqr1, Grids *t_grids,
    intT *t_corePointMarkers, GdLongDataType t_g1, GdLongDataType t_g2);
bool edgeAddableBCPNDWithSort(PtDataType *pointsData, intT dim,
    PtIntermediateDataType t_radiusSqr1, Grids *t_grids,
    intT *t_corePointMarkers, GdLongDataType t_g1, GdLongDataType t_g2);
bool edgeAddableBCPNDWithG1Sort(PtDataType *pointsData, intT dim,
    PtIntermediateDataType t_radiusSqr1, Grids *t_grids,
    intT *t_corePointMarkers, GdLongDataType t_g1, GdLongDataType t_g2);
bool edgeAddableBCPNDSimple(PtDataType *pointsData, intT dim,
    PtIntermediateDataType t_radiusSqr1, Grids *t_grids,
    intT *t_corePointMarkers, GdLongDataType t_g1, GdLongDataType t_g2);
inline GdPtLongDataType GetCorePtsParallel(Grids *t_grids,
    intT *t_corePointMarkers, PtDataType **g1Core,
    GdLongDataType t_g1, GdLongDataType t_g2,
    PtIntermediateDataType t_radiusSqr1, intT dim);
bool edgeAddableBCPNDParallel(PtDataType *pointsData, intT dim,
    PtIntermediateDataType t_radiusSqr1, Grids *t_grids,
    intT *t_corePointMarkers, GdLongDataType t_g1, GdLongDataType t_g2);
inline bool edgeAddableBCPNDSmart(PtDataType *pointsData, intT dim,
    PtIntermediateDataType t_radiusSqr1, Grids *t_grids,
    intT *t_corePointMarkers, GdLongDataType t_g1, GdLongDataType t_g2);

/**
 * Returns cluster id for all points, given boxes and core points.
 * @param t_points Sequence of input points.
 * @param t_params DBSCAN parameters.
 * @param t_grids  Grids data structure.
 * @param t_isCoreGrid A boolean vector indicating whether a grid has >=1 core points (core grid).
 * @param t_corePointMarkers pair<numCorePoints, corePointFlags>
 * @return A vector of flag of length t_points->n, containing clusterIds for core points, and -1 for non-assigned (border) points.
 */
GdLongDataType* ClusterCoreBCPGridParallelBucketingND(_seq<pointNd> *t_points, DbscanParams *t_params, Grids *t_grids, bool *isCoreGrid, pair<PtLongDataType, intT *> corePointMarkersGrid) {
  // Reconstruction hook. Return one entry per input point: a component id for
  // each core point and -1 for every other point.
  return NULL;
}

/**
 * Assigns border points after assigning core points to clusters.
 * @param t_points Input sequence, must be after calling assignBox (reordered).
 * @param t_params DBSCAN parameters.
 * @param t_grids  Grid data structure constructed earlier.
 * @param t_clusterIds A vector of cluster ids of all points. -1 means unmarked.
 * @param t_isCoreGrid A boolean vector indicating whether a grid has >=1 core points (core grid).
 * @return Number of border points newly assigned to clusters.
 */
void AssignBorderPointGridParallelND(_seq<pointNd> *t_points, DbscanParams *t_params, Grids *t_grids, GdLongDataType *t_clusterIds, intT *t_corePointMarkers, bool *isCoreGrid) {
  intT dim = t_points->A[0].m_dim;
  PtDataType *pointsData = t_points->A[0].m_data;
  PtIntermediateDataType epsSqr = static_cast<PtIntermediateDataType>(t_params->m_epsilon) * static_cast<PtIntermediateDataType>(t_params->m_epsilon);

  parallel_for_1 (GdLongDataType g = 0; g < t_grids->GetNumGrids(); ++ g) {

    GdPtLongDataType gNumPts = t_grids->GetNumPts(g);
    if (gNumPts < t_params->m_minPts) {

      NbrContainer *neighbors = GetNeighborGridsKDTree(g, t_grids);
      PtLongDataType iterEndOuter = t_grids->GetStartOffset(g) + gNumPts;
      parallel_for (PtLongDataType p = t_grids->GetStartOffset(g); p < iterEndOuter; ++p) {

        if (t_corePointMarkers[t_grids->GetPt(p)] == 0) { // Is a noncore point.

          PtLongDataType nearCore = -1;
          // Let the initial near distance be not as good as qualifying a reachable point.
          PtIntermediateDataType nearDist = epsSqr + 1; // Hack.

          // First look at grid g itself.
          if (isCoreGrid[g]) { // If has no core points, no need to look
	          PtLongDataType iterEndInner = t_grids->GetStartOffset(g) + gNumPts;
            for (PtLongDataType np = t_grids->GetStartOffset(g); np < iterEndInner; ++np) {
              if (t_corePointMarkers[t_grids->GetPt(np)] != 0) { // Core point.
                PtIntermediateDataType myDistSqr = PointDistNDSqr(t_grids->GetPtData(p), t_grids->GetPtData(np), dim);
                if (myDistSqr < nearDist && myDistSqr <= epsSqr ) {
                  nearCore = t_grids->GetPt(np);
                  nearDist = myDistSqr;
                }
              }
            }
          }

          // Then iterate neighboring boxes, find the nearets core to p.
	        for (GdLongDataType ni = 0; ni < neighbors->size(); ++ ni) {
            GdLongDataType n = neighbors->at(ni);
	          if (n < 0 || n == g) continue;

            if (isCoreGrid[n]) {

              GdPtLongDataType nNumPts = t_grids->GetNumPts(n);

              // Iterate through core points of neighboring grids to determine closest core point.
	            PtLongDataType iterEndInner = t_grids->GetStartOffset(n) + nNumPts;
              for (PtLongDataType np = t_grids->GetStartOffset(n); np < iterEndInner; ++np) {
                if (t_corePointMarkers[t_grids->GetPt(np)] != 0) { // Core point.
                  PtIntermediateDataType myDistSqr = PointDistNDSqr(t_grids->GetPtData(p), t_grids->GetPtData(np), dim);
                  if (myDistSqr < nearDist && myDistSqr <= epsSqr ) {
                    nearCore = t_grids->GetPt(np);
                    nearDist = myDistSqr;
                  }
                }
              }
            }
          }

          if (nearCore != -1) {
            t_clusterIds[t_grids->GetPt(p)] = t_clusterIds[nearCore];
          }
        }
      }
    }
  }
}

//void AssignBorderPointMultiGridParallelND(_seq<pointNd> *t_points, DbscanParams *t_params, Grids *t_grids, GdLongDataType *t_clusterIds, intT *t_corePointMarkers, bool *isCoreGrid, Table<hashBorderCluster, OtLongDataType> *t_borderHash) {
void AssignBorderPointMultiGridParallelND(_seq<pointNd> *t_points, DbscanParams *t_params, Grids *t_grids, GdLongDataType *t_clusterIds, intT *t_corePointMarkers, bool *isCoreGrid, ClusterContainer **t_borderHash) {
  intT dim = t_points->A[0].m_dim;
  PtDataType *pointsData = t_points->A[0].m_data;
  PtIntermediateDataType epsSqr = static_cast<PtIntermediateDataType>(t_params->m_epsilon) * static_cast<PtIntermediateDataType>(t_params->m_epsilon);

  parallel_for_1 (GdLongDataType g = 0; g < t_grids->GetNumGrids(); ++ g) {

    GdPtLongDataType gNumPts = t_grids->GetNumPts(g);
    if (gNumPts < t_params->m_minPts) {

      NbrContainer *neighbors = GetNeighborGridsKDTree(g, t_grids);
      PtLongDataType iterEndOuter = t_grids->GetStartOffset(g) + gNumPts;
      parallel_for (PtLongDataType p = t_grids->GetStartOffset(g); p < iterEndOuter; ++p) {

        if (t_corePointMarkers[t_grids->GetPt(p)] == 0) { // Is a noncore point.
	  bool isBorder = false;
	  ClusterContainer *myClusterContainer; // do not initialize yet

          // First look at grid g itself.
          if (isCoreGrid[g]) { // If has no core points, no need to look
	    PtLongDataType iterEndInner = t_grids->GetStartOffset(g) + gNumPts;
            for (PtLongDataType np = t_grids->GetStartOffset(g); np < iterEndInner; ++np) {
              if (t_corePointMarkers[t_grids->GetPt(np)] != 0) { // Core point.
                PtIntermediateDataType myDistSqr = PointDistNDSqr(t_grids->GetPtData(p), t_grids->GetPtData(np), dim);
                if (myDistSqr <= epsSqr ) { // in circle
		  if (!isBorder) {
		    isBorder = true;
		    myClusterContainer = new ClusterContainer();
		  }
		  myClusterContainer->insert(t_clusterIds[t_grids->GetPt(np)]);
                }
              }
            }
          }

          // Then iterate neighboring boxes, find the nearets core to p.
	  for (GdLongDataType ni = 0; ni < neighbors->size(); ++ ni) {
            GdLongDataType n = neighbors->at(ni);
	    if (n < 0 || n == g) continue;

            if (isCoreGrid[n]) {

              GdPtLongDataType nNumPts = t_grids->GetNumPts(n);

              // Iterate through core points of neighboring grids to determine closest core point.
	      PtLongDataType iterEndInner = t_grids->GetStartOffset(n) + nNumPts;
              for (PtLongDataType np = t_grids->GetStartOffset(n); np < iterEndInner; ++np) {
                if (t_corePointMarkers[t_grids->GetPt(np)] != 0) { // Core point.
                  PtIntermediateDataType myDistSqr = PointDistNDSqr(t_grids->GetPtData(p), t_grids->GetPtData(np), dim);
                  if (myDistSqr <= epsSqr ) {
		    if (!isBorder) {
		      isBorder = true;
		      myClusterContainer = new ClusterContainer();
		    }
		    myClusterContainer->insert(t_clusterIds[t_grids->GetPt(np)]);
                  }
                }
              }
            }
          }

          if (isBorder) {
	    t_borderHash[t_grids->GetPt(p)] = myClusterContainer;
            t_clusterIds[t_grids->GetPt(p)] = -2; // -1 means noise, -2 means border
          }
        }
      }
    }
  }
}

/**
 * Computes DBSCAN.
 * @param t_P Sequence of input points, typed point2d (_point2d<PtDataType>). The point order WILL be altered during the computation, and the returned value corresponds to the ALTERED points.
 * @param t_param DBSCAN parameter, defined in DBSCAN.h.
 * @return An array of positive integers indicating cluster ID for each point. -1 means no cluster assigned.
 */
#if defined CLOSEST_BORDER_CLUSTER
pair<intT*, GdLongDataType*> DBSCAN(_seq<pointNd> t_P, DbscanParams t_params)
#else
pair<GdLongDataType*, ClusterContainer** > DBSCAN(_seq<pointNd> t_P, DbscanParams t_params)
#endif
{

  PrintParser("serialalgo", "parallel");
  PrintParser("ndalgo", "nd");
#if defined(APPROX)
  PrintParser("algo", "approx");
#else
  PrintParser("algo", "exact");
#endif
  PrintParser("numprocs", getWorkers());

  int dim = t_P.A[0].m_dim;

#ifdef COMPONENT_TIMER
  startTime();
#endif
  Grids *grids = AssignGridParallelND(&t_P, &t_params, dim);
#ifdef COMPONENT_TIMER
  double assignGridTime = _tm.next();
  PrintParser("assigngridtime", assignGridTime);
#endif

  PrintParser("numgrids", grids->GetNumGrids());

#if defined(ALLPT_QUADTREE)
  QuadTree *allPtQuadTree = new QuadTree(&t_P, &t_params, grids);
#ifdef COMPONENT_TIMER
  double buildAllPtQuadTreeTime = _tm.next();
  cout << " build all pt quadtree: " << buildAllPtQuadTreeTime << endl;
#endif
#endif

  GetNeighborGridKDTreeInit(grids);
#ifdef COMPONENT_TIMER
  double buildGetNbrKdtTime = _tm.next();
  PrintParser("buildgetnbrkdt", buildGetNbrKdtTime);
#endif

  bool *isCoreGrid = newA(bool,grids->GetNumGrids());

  pair<PtLongDataType, intT *> corePointMarkersGrid = MarkCorePointGridParallelND(&t_P, &t_params, grids, isCoreGrid);
  
#ifdef COMPONENT_TIMER
  double markcoreTime = _tm.next();
#if defined(ALLPT_QUADTREE)
  markcoreTime += buildAllPtQuadTreeTime;
#endif
  PrintParser("markcoretime", markcoreTime);
#endif
  
  PrintParser("numcore", corePointMarkersGrid.first);
  PrintParser("numpts", t_P.n);

  GdLongDataType *clusterIds = ClusterCoreBCPGridParallelBucketingND(&t_P, &t_params, grids, isCoreGrid, corePointMarkersGrid);

#ifdef COMPONENT_TIMER
  // Timer code in function
  double clusterCoreTime = _tm.next();
  PrintParser("clustercoretime", clusterCoreTime);
#endif

#if defined CLOSEST_BORDER_CLUSTER
  AssignBorderPointGridParallelND(&t_P, &t_params, grids, clusterIds, corePointMarkersGrid.second, isCoreGrid);
#else
  ClusterContainer **myBorderClusterTable = newA(ClusterContainer *, t_P.n);
  AssignBorderPointMultiGridParallelND(&t_P, &t_params, grids, clusterIds, corePointMarkersGrid.second, isCoreGrid, myBorderClusterTable);
#endif
  
#ifdef COMPONENT_TIMER
  double borderTime = _tm.next();
  PrintParser("assignbordertime", borderTime);
#endif

  delete grids;
  delete g_kdtree;
  free(isCoreGrid);

#ifdef COMPONENT_TIMER
  double cleanupTime = _tm.next();
  PrintParser("cleanuptime", cleanupTime);
#endif

#if defined CLOSEST_BORDER_CLUSTER
  return make_pair(corePointMarkersGrid.second, clusterIds);
#else
  free(corePointMarkersGrid.second);
  return make_pair(clusterIds, myBorderClusterTable);
#endif
}
