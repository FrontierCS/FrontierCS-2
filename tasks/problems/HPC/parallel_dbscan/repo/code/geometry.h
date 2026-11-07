#ifndef _BENCH_GEOM_INCLUDED
#define _BENCH_GEOM_INCLUDED
#include <iostream>
#include <algorithm>
#include <math.h>
#include <iomanip>
#include "parallel.h"
using namespace std;

// *************************************************************
//    POINTS (Nd)
// *************************************************************

template <class pointDataType>
struct _pointNd {
  intT m_dim;
  pointDataType *m_data;
};

#define NDGCTYPE double
typedef struct _pointNd<NDGCTYPE> pointNd;

// *************************************************************
//    ND Point Subroutines
// *************************************************************

inline pointNd *PointNDArrayAlloc(long t_n, intT t_dim) {
  pointNd *structPt = static_cast<pointNd*>(malloc(t_n * sizeof(pointNd))); //newA(pointNd, t_n);
  NDGCTYPE *pointData = static_cast<NDGCTYPE*>(malloc(t_n * t_dim * sizeof(NDGCTYPE))); //newA(NDGCTYPE, t_n * t_dim);
  parallel_for (long i = 0; i < t_n; ++ i) {
    structPt[i].m_data = &pointData[i * t_dim];
  }
  return structPt;
}

#endif // _BENCH_GEOM_INCLUDED

