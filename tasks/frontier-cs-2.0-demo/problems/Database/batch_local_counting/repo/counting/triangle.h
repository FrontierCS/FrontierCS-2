//
// Created by anonymous author on 2022/9/27.
//

#ifndef SCOPE_TRIANGLE_H
#define SCOPE_TRIANGLE_H

#include "graph.h"

// for each out edge, store the triangles on it.
// three types of triends: 1. in-in 2. in-out 3. out-out
// offset1[eID] | in-in | offset2 | in-out | offset3 | out-out | offset1[eID+1]
// for each triangle [v1->v2, v3->v2, v3->v1], v1, v2, v3 are vertex1, vertex2, vertex3 respectively
// v1->v2, v3->v2, v3->v1 are edge1, edge2, edge3 respectively

struct Triangle {
    Count triangleCnt;
    VertexID *triend;
    EdgeID *offset1;
    EdgeID *offset2;
    EdgeID *offset3;
    Count *vertex1Cnt;
    Count *vertex2Cnt;
    Count *vertex3Cnt;
    Count *edge1Cnt;
    Count *edge2Cnt;
    Count *edge3Cnt;
    Triangle() = default;

    void EnumTriangle(const DataGraph &din, const DataGraph &dout);
    VertexID *getTriend(EdgeID &e, int type, ui &cnt) const;
    double memoryCost() const;
    void load(const std::string &path, ui m);
    void save(const std::string &path, ui m) const;
};

#endif //SCOPE_TRIANGLE_H
