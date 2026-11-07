#ifndef GETNEIGHBORKDTREE_H
#define GETNEIGHBORKDTREE_H

#include "Grids.h"
#include "GridKDTree.h"
#include "GetNeighborTypeDef.h"

typedef GridKDTree MyGridKDTreeType;

static MyGridKDTreeType *g_kdtree;
void GetNeighborGridKDTreeInit(GridsGN *t_grids);

NbrContainer *GetNeighborGridsKDTree(GdLongDataType me, GridsGN *t_grids);
NbrContainer GetHalfNeighborGridsKDTree(GdLongDataType me, GridsGN *t_grids);

#endif
