#include "DBSCAN.h"
#include "Grids.h"
#include "DBSCANGeometry.h"

intT g_dim;

// *************************************************************
//    Grid HashKey (optional impl, enable at beginning of file)
// *************************************************************

// same type as the grid coordinates
OtLongDataType ComputeModPrime(OtLongDataType* gridCoords, intT t_dim) {
  // Compute modPrime value
  unsigned long long temp = 0;
  OtLongDataType key = 0;
  intT len = t_dim;
  for (intT i = 0; i < len; i++) {
    temp = (long long) gridCoords[i] * (long long) MY_RAND_INT[i];
    temp = (temp & MASK) + 5 * (temp >> 32);
    if (temp >= PRIME) {
      temp -= PRIME;
    }
    temp += key;
    if (temp >= PRIME) {
      temp -= PRIME;
    }
    key = (OtLongDataType) temp;
  }
  return key;
}

Grids *AssignGridParallelND(_seq<pointNd> *t_points, DbscanParams *t_params, intT t_dim) {
  g_dim = t_dim;
  
  InitializeRandom(t_dim); // for mod prime hashing

  BoundingBoxND *boundingBox = new BoundingBoxND(t_dim); // todo parallel
  GetNDBoundingBoxParallel(t_points, boundingBox, t_dim);
  
  PtIntermediateDataType gridSize = static_cast<PtIntermediateDataType>(t_params->m_epsilon) / static_cast<PtIntermediateDataType>(sqrt(t_dim));
  PtDataType *pointsData = t_points->A[0].m_data;

  // Compute number of grids along each dimension.
  OtLongDataType *numGridEachDim = newA(OtLongDataType, t_dim);

  for (intT d = 0; d < t_dim; ++ d) {
    numGridEachDim[d] = ceil((boundingBox->m_maxVec[d] - boundingBox->m_minVec[d]) / gridSize);
  }

  OtLongDataType *seqMemOrig = static_cast<OtLongDataType *>(malloc(sizeof(OtLongDataType) * (t_dim + 1) * t_points->n));
  pair<OtLongDataType *, long> *gridPointPairOrig = static_cast<pair<OtLongDataType *, long> *>(malloc(sizeof(pair<OtLongDataType *, long>) * t_points->n));
  parallel_for (PtLongDataType i = 0; i < t_points->n; ++ i) {
    gridPointPairOrig[i].first = GetPtGridVecExt(pointsData + i * t_dim, t_dim, seqMemOrig + i * (t_dim + 1), boundingBox, gridSize);
    gridPointPairOrig[i].second = i; // record the orig point 
  }
  pair<OtLongDataType *, long> *gridPointPairSortedOrig = gridPointPairOrig;
  sampleSort(gridPointPairSortedOrig, t_points->n, gridKeyComparator()); // comparator only uses [0,dim)
  GdLongDataType *gridFlagPrefix = newA(GdLongDataType,t_points->n);
  gridFlagPrefix[0] = 1;

  parallel_for (PtLongDataType i = 1; i < t_points->n; ++ i) {
    if (!GridKeyEqual(gridPointPairSortedOrig[i].first, gridPointPairSortedOrig[i - 1].first, t_dim)) {
      gridFlagPrefix[i] = 1;
    } else {
      gridFlagPrefix[i] = 0;
    }
  }
  
  GdLongDataType numGrids;
  if (sizeof(PtLongDataType) > 4) {
    numGrids = sequence::prefixSumL<GdLongDataType>(gridFlagPrefix, 0, t_points->n);
  } else {
    numGrids = sequence::prefixSum<GdLongDataType>(gridFlagPrefix, 0, t_points->n);
  }

  // compaction
  PtLongDataType *gridStartPtsCompact = newA(PtLongDataType,numGrids);
  GdPtLongDataType *gridNumPtsCompact = newA(GdPtLongDataType,numGrids);
  Table<hashGridVec, OtLongDataType> *myHashTableOrig = new Table<hashGridVec, OtLongDataType>(numGrids, hashGridVec(), 2);

  OtLongDataType *gridLabelCompactMem = static_cast<OtLongDataType *>(malloc(sizeof(OtLongDataType) * (t_dim + 1) * numGrids));
  PtLongDataType *gridPointPointers = static_cast<PtLongDataType *>(malloc(sizeof(PtLongDataType) * t_points->n));

  parallel_for (PtLongDataType i = 0; i < t_points->n - 1; ++ i) {
    gridPointPointers[i] = gridPointPairSortedOrig[i].second;
    if (gridFlagPrefix[i] != gridFlagPrefix[i + 1]) {

      for (intT d = 0; d < t_dim + 1; ++ d) {
        gridLabelCompactMem[gridFlagPrefix[i] * (t_dim + 1) + d] = gridPointPairSortedOrig[i].first[d];
      }
      myHashTableOrig->insert(make_pair(gridLabelCompactMem + gridFlagPrefix[i] * (t_dim + 1), gridFlagPrefix[i]));
      gridStartPtsCompact[gridFlagPrefix[i]] = i;
    }
  }

  PtLongDataType i = t_points->n - 1;
  gridPointPointers[i] = gridPointPairSortedOrig[i].second;
  if (gridFlagPrefix[i] != numGrids) {

    for (intT d = 0; d < t_dim + 1; ++ d) {
      gridLabelCompactMem[gridFlagPrefix[i] * (t_dim + 1) + d] = gridPointPairSortedOrig[i].first[d];
    }
    myHashTableOrig->insert(make_pair(gridLabelCompactMem + gridFlagPrefix[i] * (t_dim + 1), gridFlagPrefix[i]));
    gridStartPtsCompact[gridFlagPrefix[i]] = i;
  }

  granular_for_t (i, 0, numGrids - 1, 2000, GdLongDataType, {
    gridNumPtsCompact[i] = gridStartPtsCompact[i + 1] - gridStartPtsCompact[i];
  });
  gridNumPtsCompact[numGrids - 1] = t_points->n - gridStartPtsCompact[numGrids - 1];

  free(gridFlagPrefix);

  free(seqMemOrig);
  free(gridPointPairOrig);

  Grids* result = new Grids(gridLabelCompactMem, gridStartPtsCompact, gridNumPtsCompact,
       myHashTableOrig, gridPointPairSortedOrig, numGrids, numGridEachDim,
       boundingBox, gridSize, t_dim, pointsData, gridPointPointers);

  return result;
}

