//
// Created by anonymous author on 2022/8/19.
//

#ifndef SCOPE_EQUATION_H
#define SCOPE_EQUATION_H

#include "decompose.h"
#include "compute_set_intersection.h"

// key is the canonical value of the colored pattern
static std::map<CanonType, std::vector<Tree>> gCanon2Tree;
static std::map<CanonType, Pattern> gCanon2Pattern;
static std::map<CanonType, std::vector<Pattern>> gCanon2Shrinkage;
extern double gEquationTime;

std::vector<VertexID> mergePair(
        const std::vector<std::vector<VertexID>> &independentPartitions,
        const std::vector<VertexID> &oldPartition,
        Edge nextPair,
        const std::vector<std::vector<bool>> &matrix,
        const Tree &t
);
std::vector<Tree> relabelTrees(const std::vector<Tree> &trees, VertexID *mapping);
void adjustAggreInfo(std::vector<Tree> &trees, bool sign, const VertexID *aggreV, const int *aggreWeight, ui aggreSize,
                     const Pattern &p, std::vector<Tree> &visitedDecomp, bool useTriangle);
bool genEquation(const PatternGraph &p, std::map<int, std::vector<Pattern>> &patterns,
                 std::map<int, std::vector<std::vector<Tree>>> &trees, ConNode &cn, bool useTriangle,
                 bool symmetryBreaking, bool prefix, bool useDirected);
std::vector<VertexID> mergePairHomo(
        const std::vector<std::vector<VertexID>> &independentPartitions,
        const std::vector<VertexID> &oldPartition,
        Edge nextPair,
        const PatternGraph &p
);
std::vector<Pattern> homoShrinkage(const Pattern &p);
std::vector<Pattern> homoShrinkage(const Pattern &p, std::vector<int> &mu);
void homoEquation(const PatternGraph &p, std::map<int, std::vector<Pattern>> &patterns,
                  std::map<int, std::vector<std::vector<Tree>>> &trees, bool prefix);

#endif //SCOPE_EQUATION_H
