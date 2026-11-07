#ifndef GRID_H
#define GRID_H

#include <vector>
#include <algorithm>
#include <iostream>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <numeric>
#include <iterator>
#include <fstream>
#include <set>

#include "DBSCAN.h"
#include "ndHash.h"
#include "gettime.h"
#include "sampleSort.h"
#include "DBSCANGeometry.h"

#if defined USE_JEMALLOC
#include <jemalloc/jemalloc.h>
#include "je_allocator.h"
#endif

#if defined(USE_JEMALLOC)
typedef std::vector<GdLongDataType, je_allocator<GdLongDataType> > NbrContainer;
#else
typedef std::vector<GdLongDataType> NbrContainer;
#endif

extern intT g_dim;

inline bool IsNeighbor(OtLongDataType* g1, OtLongDataType* g2, int t_dim) {

  OtLongDataType sqr_minDist = 0;
  OtLongDataType temp = 0;
  for (int i = 0; i < t_dim; i++) {
    temp = 0;
    if (g1[i] + 1 < g2[i]) {
      temp = (g2[i] - g1[i] - 1);
    } else if (g2[i] + 1 < g1[i]) {
      temp = (g1[i] - g2[i] - 1);
    }
    sqr_minDist += temp * temp;
  }

  return sqr_minDist < t_dim;
}

// *************************************************************
//    Grid HashKey (optional impl, enable at beginning of file)
// *************************************************************

static const uintT PRIME = -5;
static const unsigned int MASK = -1;
static const unsigned int RANGE = (1 << 29);
static NbrContainer MY_RAND_INT;

inline bool GridKeyLt(OtLongDataType* i, OtLongDataType* j) {
  for (intT d = 0; d < g_dim; d++) {
    if (i[d] != j[d]) {
      if (i[d] < j[d]) {
	return true;
      } else {
	return false;
      }
    }
  }
  return false;
}

// eq comparator, has key at end of array
inline bool GridKeyEqual(OtLongDataType *i, OtLongDataType *j, intT t_dim) {
  if (i[t_dim] != j[t_dim]) {
    return false;
  }
  for (intT d = 0; d < t_dim; ++ d) {
    if (i[d] != j[d]) {
      return false;
    }
  }
  return true;
}

struct gridKeyComparator { // for sample sort, lt
  bool operator() (const pair<OtLongDataType *, long> &i, const pair<OtLongDataType *, long> &j) {
    return GridKeyLt(i.first,j.first);
  }
};

// for mod prime hashing
inline void InitializeRandom(intT t_dim) {
  MY_RAND_INT.clear();
  srand(time(NULL));
  for (intT i = 0; i < t_dim; i++) {
    MY_RAND_INT.push_back(rand() % RANGE + 1);
  }
}

OtLongDataType ComputeModPrime(OtLongDataType* gridCoords, intT t_dim);

// key: grid coord seq appended with a primehash
// value: compact grid coord
struct hashGridVec {
  typedef pair<OtLongDataType *, long> eType;
  typedef OtLongDataType* kType;
  eType empty() {
    return pair<OtLongDataType *, long>((OtLongDataType *)NULL,-1);
  }
  kType getKey(eType v) { return v.first; }
  OtLongDataType hash(OtLongDataType *s) {
    return s[g_dim];
  }
  int cmp(OtLongDataType *g1, OtLongDataType *g2) {
    if(g2 == NULL) return 1;
    for (intT i = 0; i < g_dim; i++) {
      if (g1[i] > g2[i]) {
	return 1;
      } else if (g1[i] < g2[i]) {
	return -1;
      }
    }
    return 0;
  }
  bool replaceQ(eType s, eType s2) {return 0;}
};

// given point, compute grid vec including key
inline OtLongDataType * GetPtGridVecExt(PtDataType *t_point, intT t_dim, OtLongDataType *t_vec, BoundingBoxND *t_boundingBox, PtIntermediateDataType t_gridSize) {
  for (intT d = 0; d < t_dim; ++ d) {
    t_vec[d] = static_cast<OtLongDataType>( (t_point[d] - t_boundingBox->m_minVec[d]) / t_gridSize );
  }
  t_vec[t_dim] = ComputeModPrime(t_vec, t_dim); // compute a hash at the of vector
  return t_vec;
}

struct Grids {

private:
  PtIntermediateDataType m_gridSize;
  GdLongDataType m_numValidGrid;
  intT m_dim;
  PtIntermediateDataType m_dimSqrt;
  OtLongDataType *m_numGrid; // Number of grids in each dimension.
  BoundingBoxND *m_boundingBox;
  PtLongDataType *m_gridStartPtsCompact;
  GdPtLongDataType *m_gridNumPtsCompact;

  Table<hashGridVec, OtLongDataType> *m_ndHashTable;
  
  OtLongDataType *m_gridLabelMemCompact;
  pair<OtLongDataType *, long> *m_gridPointPairSorted;
  
  PtDataType *m_ptData;

  bool *m_hasNeighborVec;
  NbrContainer **m_neighborVec;

  PtLongDataType *m_gridPointPointer;

public:

  Grids(OtLongDataType *t_gridLabelMemCompact, PtLongDataType *t_gridStartPtsCompact, GdPtLongDataType *t_gridNumPtsCompact, Table<hashGridVec, OtLongDataType> *t_ndHashTableOrig, pair<OtLongDataType *, long> *t_gridPointPairSortedOrig, GdLongDataType t_numValidGrid, OtLongDataType* t_numGrid, BoundingBoxND *t_boundingBox, PtIntermediateDataType t_gridSize, intT t_dim, PtDataType *t_ptData, PtLongDataType *t_gridPointPointer) {
    m_gridLabelMemCompact = t_gridLabelMemCompact;
    m_gridStartPtsCompact = t_gridStartPtsCompact;
    m_gridNumPtsCompact = t_gridNumPtsCompact;
    m_ndHashTable = t_ndHashTableOrig;
    m_gridPointPairSorted = t_gridPointPairSortedOrig;
    m_numValidGrid = t_numValidGrid;
    m_numGrid = t_numGrid;
    m_boundingBox = t_boundingBox;
    m_gridSize = t_gridSize;
    m_dim = t_dim;
    m_dimSqrt = sqrt(t_dim);
    m_ptData = t_ptData;
    m_gridPointPointer = t_gridPointPointer;

    m_hasNeighborVec = static_cast<bool *>(malloc(sizeof(bool) * t_numValidGrid) );
    m_neighborVec = static_cast<NbrContainer **>(malloc(sizeof(NbrContainer *) * t_numValidGrid) );
    parallel_for (GdLongDataType i = 0; i < t_numValidGrid; ++ i) {
      m_neighborVec[i] = new NbrContainer; // note logic change in neighbor finding
      m_hasNeighborVec[i] = false;
    }
  }

  inline NbrContainer *GetNbrVector(GdLongDataType t_gridId) {
    return m_neighborVec[t_gridId];
  }

  inline bool HasFilledNbrVector(GdLongDataType t_gridId) {
    return m_hasNeighborVec[t_gridId];
  }

  inline void MarkFilledNbrVector(GdLongDataType t_gridId) {
    m_hasNeighborVec[t_gridId] = true;
  }

  inline intT GetDim() {
    return m_dim;
  }

  inline PtIntermediateDataType GetDimSqrt() {
    return m_dimSqrt;
  }

  inline intT GetVecLen() {
    return m_dim + 1;
  }

  inline PtIntermediateDataType GetSize() {
    return m_gridSize;
  }

  inline GdLongDataType GetNumGrids() {
    return m_numValidGrid;
  }

  inline OtLongDataType *GetNDNumGrids() {
    return m_numGrid;
  }

  inline PtLongDataType GetStartOffset(GdLongDataType t_g) {
    return m_gridStartPtsCompact[t_g];
  }

  inline GdPtLongDataType GetNumPts(GdLongDataType t_g) {
    return m_gridNumPtsCompact[t_g];
  }

  // takes point index in grids context
  // return point index in actual point arrary
  inline PtLongDataType GetPt(PtLongDataType t_idx) {
    return m_gridPointPointer[t_idx];
  }

  inline PtDataType* GetPtData(PtLongDataType t_idx) {
    return m_ptData +  m_gridPointPointer[t_idx] * m_dim;
  }

  // takes compact idx
  // return vector representing grid location index
  inline void GetGridVec(GdLongDataType t_g, OtLongDataType *t_gridVec) {
    for (intT d = 0; d < m_dim + 1; ++ d) {
      t_gridVec[d] = m_gridLabelMemCompact[t_g * (m_dim + 1) + d];
    }
  }
  
  // same as above but read only
  inline OtLongDataType ReadGridVec(GdLongDataType t_g, intT t_d) {
    return m_gridLabelMemCompact[t_g * (m_dim + 1) + t_d];
  }

  // return -1 if invalid grid vec
  inline GdLongDataType GetGridId(OtLongDataType *t_gridVec) {
    // .second is hardcoded as long in hashtable to achieve 16 bytes
    return static_cast<GdLongDataType>( m_ndHashTable->find(t_gridVec).second);
  }

  inline OtLongDataType * GetPtGridVec(PtDataType *t_point, OtLongDataType *t_vec) {
    for (intT d = 0; d < m_dim; ++ d) {
      t_vec[d] = static_cast<OtLongDataType>( (t_point[d] - m_boundingBox->m_minVec[d]) / GetSize() );
    }
    t_vec[m_dim] = ComputeModPrime(t_vec, m_dim);
    return t_vec;
  }

  inline void GetOffsetGridVec(OtLongDataType *nbrDimIdx, OtLongDataType *t_meGridIdx, intT *t_nbrOffset) {
    for (intT d = 0; d < m_dim; ++ d) {
      nbrDimIdx[d] = t_meGridIdx[d] + t_nbrOffset[d];
    }
    nbrDimIdx[m_dim] = ComputeModPrime(nbrDimIdx, m_dim);
  }

  inline void CommitGridVec(OtLongDataType *t_gridVec) {
    t_gridVec[m_dim] = ComputeModPrime(t_gridVec, m_dim);
    return;
  }
  
  inline bool IsValidGridVec(OtLongDataType *t_gridVec) {
    for (intT d = 0; d < m_dim; ++ d) {
      if (t_gridVec[d] < 0 || t_gridVec[d] >= m_numGrid[d]) {
	return false;
      }
    }
    return true;
  }

  inline PtDataType GetLowCoord(intT t_d) {
    return m_boundingBox->m_minVec[t_d];
  }
  
  ~Grids() { // todo
    free(m_numGrid);
    // free(m_gridPointPairSorted);
    free(m_gridStartPtsCompact);
    free(m_gridNumPtsCompact);
    free(m_gridPointPointer);
    free(m_gridLabelMemCompact);
    free(m_hasNeighborVec);
    parallel_for (GdLongDataType i = 0; i < m_numValidGrid; ++ i) {
      delete m_neighborVec[i];
    }
    free(m_neighborVec);
    
    m_boundingBox->del();
    m_ndHashTable->del();
  }
};

// function decs
//Grids *AssignGridSerialND(_seq<pointNd> *t_points, DbscanParams *t_params, intT t_dim);
Grids *AssignGridParallelND(_seq<pointNd> *t_points, DbscanParams *t_params, intT t_dim);

#endif
