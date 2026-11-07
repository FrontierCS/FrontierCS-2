//
// Created by anonymous author on 2022/8/8.
//

#ifndef SCOPE_COST_ESTIMATOR_H
#define SCOPE_COST_ESTIMATOR_H

#include "tree.h"
std::vector<std::vector<VertexID>>
generateLocalOrders(const Node &tau, const PatternGraph &p, const std::vector<VertexID> &partitionOrder);
std::vector<ui>
getDegSeqAndSrcPos(const std::vector<VertexID> &localOrder, int &srcPos, const Pattern &p, VertexID *vertices, ui numVertices);
std::vector<ui> evaluateLocalOrder(const std::vector<VertexID> &partitionOrder,
                                   const std::vector<VertexID> &localOrder, int &inNum, const Pattern &p, VertexID *vertices, ui numVertices);
bool firstSequenceLarger(const std::vector<ui> &seq1, const std::vector<ui> &seq2);
void simpleLocalOrder(Tree &t, const Pattern &p);
std::vector<Tree> bestLocalOrder(std::vector<std::vector<std::vector<Tree>>> &minWidth2Tree, const Pattern &p);
Tree bestLocalOrder(std::vector<Tree> &minWidth2Tree, const Pattern &p);
void bestLocalOrder(const Tree &t, VertexID nID, const std::vector<VertexID> &partitionOrder, const Pattern &p,
                    std::vector<VertexID> &localOrder, ui &numIn, ui &edgePos, bool useTriangle);
void
bestLocalOrder(const Tree &t, VertexID nID, const Pattern &p, std::vector<VertexID> &localOrder, ui &numIn, ui &edgePos,
               bool useTriangle);
void conNodeOrder(const ConNode &cn, const Node &tau, const std::vector<VertexID> &prefixOrder, const PatternGraph &p,
                  std::vector<VertexID> &localOrder, ui &numIn, ui &edgePos, bool useTriangle,
                  const std::vector<std::vector<VertexID>> &greaterRules,
                  const std::vector<std::vector<VertexID>> &lessRules);

#endif //SCOPE_COST_ESTIMATOR_H
