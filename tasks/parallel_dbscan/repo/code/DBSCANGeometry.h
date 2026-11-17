#ifndef DBSCANGEOMETRY_H
#define DBSCANGEOMETRY_H

#include <vector>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <string.h>
#include <stdlib.h>
#include <numeric>
#include <iterator>
#include <fstream>

#include "DBSCAN.h"
#include "ndHash.h"
#include "gettime.h"
#include "sampleSort.h"
#include "gettime.h"

#define INTERSECT 1
#define COVERED 2
#define DISJOINT 3

inline PtDataType PointDistNDSqr(PtDataType *t_pointDataA, PtDataType *t_pointDataB, intT t_dim) {
  PtDataType sqrSum = 0;
  PtDataType tmp;
  for (intT d = 0; d < t_dim; ++ d) {
    tmp = t_pointDataA[d] - t_pointDataB[d];
    sqrSum += tmp * tmp;
  }
  return sqrSum;
}

// inline int PointDistNDSqr(int *t_pointDataA, int *t_pointDataB, intT t_dim) {
//   int sqrSum = 0;
//   int tmp;
//   for (intT d = 0; d < t_dim; ++ d) {
//     tmp = t_pointDataA[d] - t_pointDataB[d];
//     sqrSum += tmp * tmp;
//   }
//   return sqrSum;
// }

inline bool PointDistNDSqrLeq(PtDataType *t_pointDataA, PtDataType *t_pointDataB, intT t_dim, PtIntermediateDataType t_threshold) {
  PtDataType sqrSum = 0;
  PtDataType tmp;
  for (intT d = 0; d < t_dim; ++ d) {
    tmp = t_pointDataA[d] - t_pointDataB[d];
    sqrSum += tmp * tmp;
  }
  if (sqrSum > t_threshold) {
    return false;
  }
  return true;
}

// inline bool PointDistNDSqrLeq(int *t_pointDataA, int *t_pointDataB, int t_dim, int t_threshold) {
//   int sqrSum = 0;
//   int tmp;
//   for (intT d = 0; d < t_dim; ++ d) {
//     tmp = t_pointDataA[d] - t_pointDataB[d];
//     sqrSum += tmp * tmp;
//   }
//   if (sqrSum > t_threshold) {
//     return false;
//   }
//   return true;
// }

inline intT GridCircleIntersect(PtDataType *t_center, PtIntermediateDataType t_radiusSqr, PtIntermediateDataType *t_grid, PtIntermediateDataType t_sideHalved, intT t_dim) {
  // find the closest and furthest distances from circle center to hypersphere
  // denoted cf and df respectively
  // let r be circle radius
  // r > fd -> covered
  // r < cd -> disjoint
  PtDataType sumSqrMin = 0;
  PtDataType sumSqrMax = 0;
  for (intT d = 0; d < t_dim; ++ d) {
    PtDataType dist1 = std::abs(t_grid[d] - t_sideHalved - t_center[d]);
    PtDataType dist2 = std::abs(t_grid[d] + t_sideHalved - t_center[d]);
    if (t_center[d] < t_grid[d] - t_sideHalved ||
      t_center[d] > t_grid[d] + t_sideHalved) {
      PtDataType tmpMin = std::min(dist1, dist2);
      sumSqrMin += tmpMin * tmpMin;
    }
    PtDataType tmpMax = std::max(dist1, dist2);
    sumSqrMax += tmpMax * tmpMax;
  }
  if (sumSqrMax <= t_radiusSqr) { // todo opt cast to int
    return COVERED;
  } 
  else if (sumSqrMin > t_radiusSqr) { // todo opt cast to int
    return DISJOINT;
  } 
  else {
    return INTERSECT;
  }
}

// inline intT GridCircleIntersect(int *t_center, double t_radiusSqr, double *t_grid, double t_sideHalved, intT t_dim) {
//   // find the closest and furthest distances from circle center to hypersphere
//   // denoted cf and df respectively
//   // let r be circle radius
//   // r > fd -> covered
//   // r < cd -> disjoint
//   int sumSqrMin = 0;
//   int sumSqrMax = 0;
//   for (intT d = 0; d < t_dim; ++ d) {
//     int dist1 = std::abs(t_grid[d] - t_sideHalved - t_center[d]);
//     int dist2 = std::abs(t_grid[d] + t_sideHalved - t_center[d]);
//     if (t_center[d] < t_grid[d] - t_sideHalved ||
//       t_center[d] > t_grid[d] + t_sideHalved) {
//       int tmpMin = std::min(dist1, dist2);
//       sumSqrMin += tmpMin * tmpMin;
//     }
//     int tmpMax = std::max(dist1, dist2);
//     sumSqrMax += tmpMax * tmpMax;
//   }
//   if (sumSqrMax <= t_radiusSqr) { // todo opt cast to int
//     return COVERED;
//   } 
//   else if (sumSqrMin > t_radiusSqr) { // todo opt cast to int
//     return DISJOINT;
//   } 
//   else {
//     return INTERSECT;
//   }
// }

// *************************************************************
//     Bounding box
// *************************************************************

struct BoundingBoxND {
  PtDataType *m_maxVec;
  PtDataType *m_minVec;
  intT m_dim;

  BoundingBoxND(intT t_dim) {
    m_dim = t_dim;
    m_maxVec = newA(PtDataType, t_dim);
    m_minVec = newA(PtDataType, t_dim);
  }
  
  void del() {
    free(m_maxVec);
    free(m_minVec);
  }
};

inline void GetNDBoundingBox(_seq<pointNd> *t_points, BoundingBoxND *t_out, intT t_dim) {
  PtDataType *pointsData = t_points->A[0].m_data;

  for (intT d = 0; d < t_dim; ++ d) {
    t_out->m_maxVec[d] = t_points->A->m_data[d];
    t_out->m_minVec[d] = t_points->A->m_data[d];
  }
  for (PtLongDataType i = 0; i < t_points->n; ++ i) {
    for (intT d = 0; d < t_dim; ++ d) {
      t_out->m_maxVec[d] = pointsData[i * t_dim + d] > t_out->m_maxVec[d] ? pointsData[i * t_dim + d] : t_out->m_maxVec[d];
      t_out->m_minVec[d] = pointsData[i * t_dim + d] < t_out->m_minVec[d] ? pointsData[i * t_dim + d] : t_out->m_minVec[d];
    }
  }
  t_out->m_dim = t_dim;
}

inline void GetNDBoundingBoxParallel(_seq<pointNd> *t_points, BoundingBoxND *t_out, OtLongDataType t_dim) {
  OtLongDataType P = getWorkers() * 32;
  
  // init
  PtDataType *pointsData = t_points->A[0].m_data;
  t_out->m_dim = t_dim;

  PtLongDataType blockSize = (t_points->n + P - 1) / P;
  PtDataType localMaxes[P * t_dim], localMins[P * t_dim];
  for (OtLongDataType i = 0; i < t_dim * P; ++ i) {
    localMaxes[i] = t_points->A->m_data[i % t_dim];
    localMins[i] = t_points->A->m_data[i % t_dim];
  }

  parallel_for(OtLongDataType i=0; i<P; i++) { // todo try parallel for 1
    PtLongDataType start = i*blockSize;
    PtLongDataType end = min((long)(i+1)*blockSize,t_points->n);
    // each thread find its own bounding box
    for (PtLongDataType j = start; j < end; ++ j) {
      for (OtLongDataType d = 0; d < t_dim; ++ d) {
        localMaxes[i * t_dim + d] = pointsData[j * t_dim + d] > localMaxes[i * t_dim + d] ? pointsData[j * t_dim + d] : localMaxes[i * t_dim + d];
        localMins[i * t_dim + d] = pointsData[j * t_dim + d] < localMins[i * t_dim + d] ? pointsData[j * t_dim + d] : localMins[i * t_dim + d];
      }
    }
  }

  for (OtLongDataType d = 0; d < t_dim; ++ d) {
    t_out->m_maxVec[d] = localMaxes[d];
    t_out->m_minVec[d] = localMins[d];
  }
  for(OtLongDataType i=0;i<P;i++) {
    for (OtLongDataType d = 0; d < t_dim; ++ d) {
      t_out->m_maxVec[d] = localMaxes[i * t_dim + d] > t_out->m_maxVec[d] ? localMaxes[i * t_dim + d] : t_out->m_maxVec[d];
      t_out->m_minVec[d] = localMins[i * t_dim + d] < t_out ->m_minVec[d] ? localMins[i * t_dim + d] : t_out ->m_minVec[d];
    }
  }
}

#endif
