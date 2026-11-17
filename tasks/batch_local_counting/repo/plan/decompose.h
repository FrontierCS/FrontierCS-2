//
// Created by anonymous author on 2022/8/2.
//

#ifndef SCOPE_DECOMPOSE_H
#define SCOPE_DECOMPOSE_H

#include "tree.h"
#include "cost_estimator.h"
#include <queue>

extern double gSymmTime;
extern double gOrderTime;

int computeNumRules(Tree &t, const PatternGraph &p);
std::vector<Pattern> computeShrinkages(const Tree &t, const Pattern &p);
std::vector<std::vector<Tree>>
minWidthMaxSymmTrees(const Pattern &p, std::vector<Tree> allTree, bool sign, int &numRules, bool prefix,
                     bool useTriangle);
std::vector<Tree> DISCMinWidth(const Pattern &p, std::vector<Tree> allTree, bool sign);
std::vector<std::vector<Tree>> betterSymmetryDecompositions(const Pattern &p, std::vector<Tree> allTree, ui tw, int oldNumRules);
bool isConnectedOrder(std::vector<VertexID> order, const PatternGraph &p);
std::vector<std::vector<VertexID>> generatePrefixes(VertexID *cut, ui cutSize, const PatternGraph &p);
int getPrefixPos(const std::vector<VertexID> &partitionOrder, const std::vector<std::vector<VertexID>> &prefixes);
bool orderVisited(const std::vector<std::vector<int>> &visitedOrder, const std::vector<int> &current);
std::vector<Tree> setGlobalOrderForT(Tree t, const Pattern &p);
std::vector<Tree>
getBestDecomposition(const Pattern &p, const std::vector<Tree> &allTree, std::vector<Tree> &visitedDecomp, bool sign,
                     bool prefix, bool useTriangle);
std::vector<Tree> analysisFunction(std::vector<std::vector<Tree>> allRootedTree, const Pattern &p, int type);
std::vector<Tree>
DISCDecomposition(const Pattern &p, const std::vector<Tree> &allTree, std::vector<Tree> &visitedDecomp, bool sign,
                  bool useTriangle);
void treeBestOrder(const Pattern &p, Tree &t, std::vector<Tree> &visitedDecomp, bool useTriangle);
void treeBestOrder(const Pattern &p, Tree &t, bool useTriangle, bool prefix);
bool iterationNotIncrease(const Tree &t, const Pattern &p, bool prefix);
int subgraphSymmetry(const PatternGraph &p, const Node &tau, std::vector<std::vector<VertexID>> &greaterRules,
                     std::vector<std::vector<VertexID>> &lessRules, int &numRules, bool symmetry);
void
conNodeDecompose(ConNode &cn, const PatternGraph &p, const Node &tau, bool symmetry, bool prefix, bool useTriangle);

#endif //SCOPE_DECOMPOSE_H
