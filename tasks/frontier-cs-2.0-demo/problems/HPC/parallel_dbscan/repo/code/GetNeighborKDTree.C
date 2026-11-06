#include <math.h>
#include "GetNeighborKDTree.h"
#include "GridKDTree.h"
#include "Grids.h"
#include "gettime.h"

void GetNeighborGridKDTreeInit(GridsGN *t_grids) {
  g_kdtree = new GridKDTree(t_grids);
}

// detail benchmarked version
void GetNeighborGridsKDTreeHelper(GdLongDataType me, GridsGN *t_grids, NbrContainer *t_neighbors) {

  t_neighbors->clear();

  OtLongDataType meGridVec[t_grids->GetVecLen()];
  t_grids->GetGridVec(me, meGridVec);

  // drawing a circle around the current grid, this is the radius
  intT stopCondition = ceil(t_grids->GetDimSqrt());
  
  // The range of possible coordinates of neighboring cells.
  OtLongDataType minCoords[g_dim];
  OtLongDataType maxCoords[g_dim];

  // getting the min and mas coordinates, basically the dimensionwise maximum and minimum

  OtLongDataType temp = 0;
  for (intT i = 0; i < g_dim; i++) {
    minCoords[i] = 0;
    maxCoords[i]= 0;
    temp = meGridVec[i] - stopCondition;
    if (temp > 0) { // All the coordinates should be non-negative.
      minCoords[i] = temp;
    }
    maxCoords[i] = temp + (stopCondition << 1);
  }

  g_kdtree->TreeRangeQuery(g_kdtree->GetTree(), minCoords, maxCoords, t_neighbors);
}

NbrContainer *GetNeighborGridsKDTree(GdLongDataType me, GridsGN *t_grids) {
  
  NbrContainer *t_neighbors = t_grids->GetNbrVector(me);
  
  if (t_grids->HasFilledNbrVector(me)) {
    return t_neighbors;
  } else {
    GetNeighborGridsKDTreeHelper(me, t_grids, t_neighbors);
    t_grids->MarkFilledNbrVector(me);
    return t_neighbors;
  }

}

// for bcp, etc where each pair of edge only needs to appear once
NbrContainer GetHalfNeighborGridsKDTree(GdLongDataType me, GridsGN *t_grids) {

  NbrContainer *t_neighbors = t_grids->GetNbrVector(me);
  if (!t_grids->HasFilledNbrVector(me)) {
    GetNeighborGridsKDTreeHelper(me, t_grids, t_neighbors);
    t_grids->MarkFilledNbrVector(me);
  }

  // pruning
  NbrContainer resultNeighbors;
  for (GdLongDataType i = 0; i < t_neighbors->size(); ++ i) {
    if (t_grids->ReadGridVec(t_neighbors->at(i), 0) <= t_grids->ReadGridVec(me, 0) && t_neighbors->at(i) != me) {
      resultNeighbors.push_back(t_neighbors->at(i));
    }
  }
  return resultNeighbors;
}
