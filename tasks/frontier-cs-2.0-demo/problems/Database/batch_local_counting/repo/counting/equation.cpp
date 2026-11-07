//
// Created by Qiyan LI on 2022/8/19.
//

#include "equation.h"

double gEquationTime = 0.0;

std::vector<VertexID> mergePair(
        const std::vector<std::vector<VertexID>> &independentPartitions,
        const std::vector<VertexID> &oldPartition,
        Edge nextPair,
        const std::vector<std::vector<bool>> &matrix,
        const Tree &t
) {
    std::vector<VertexID> newPartition;
    // 1. for the next pair (u1, u2), check whether u1 has edges with u' in the partition of u2,
    // whether u1 and u' in the partition of u2 is in the same tree node, and vice versa.
    // If so, next pair is invalid
    VertexID u1 = nextPair.first, u2 = nextPair.second;
    VertexID p1 = oldPartition[u1], p2 = oldPartition[u2];
    if (p1 == p2) return newPartition;
    std::vector<VertexID> verticesInP1;
    for (VertexID i = 0; i < oldPartition.size(); ++i) {
        if (oldPartition[i] == p1)
            verticesInP1.push_back(i);
    }
    std::vector<VertexID> verticesInP2;
    for (VertexID i = 0; i < oldPartition.size(); ++i) {
        if (oldPartition[i] == p2)
            verticesInP2.push_back(i);
    }
    for (VertexID u1: verticesInP1)
        for (VertexID u2 : verticesInP2)
            if (!matrix[u1][u2]) return newPartition;
    // 2. check whether the new partition is not in the independentPartitions
    // to build the new partition, merge verticesInP1 and verticesInP2, set their "partition value"
    // as the smaller value of p1 and p2
    newPartition = oldPartition;
    VertexID pNew = p1 < p2 ? p1 : p2;
    for (VertexID u : verticesInP1)
        newPartition[u] = pNew;
    for (VertexID u : verticesInP2)
        newPartition[u] = pNew;
    for (const auto &partition: independentPartitions)
        if (partition == newPartition) {
            newPartition.clear();
            return newPartition;
        }
    return newPartition;
}

// compute the direct shrinkage, i.e. merging two nodes together.
// shrinkages can be empty and should be skipped during processing
std::vector<Pattern> computeShrinkages(const Tree &t, const Pattern &p) {
#ifdef ONLY_PLAN
    auto start = std::chrono::steady_clock::now();
#endif
    std::vector<Pattern> shrinkage;
    ui n = p.u.getNumVertices();
    std::vector<std::vector<bool>> matrix(n, std::vector<bool>(n, true));
    std::vector<Edge> shrinkPair;   // pair of vertices that can shrinks
    std::vector<std::vector<VertexID>> independentPartitions;   // each trivialPartition is an independent set of p
    // when considering shrinkage, we take directed graphs and undirected
    // vertices in the same node are mapped to distinct data vertices, so they can not shrink
    for (int nID = 0; nID < t.getNumNodes(); ++nID) {
        ui n2;
        VertexID *nodeVertices = t.getVertices(nID, n2);
        for (int i = 0; i < n2; ++i) {
            for (int j = 0; j < n2; ++j) {
                matrix[nodeVertices[i]][nodeVertices[j]] = false;
            }
        }
    }
    for (VertexID i = 0; i < n; ++i) {
        for (VertexID j = i + 1; j < n; ++j) {
            if (matrix[i][j])
                shrinkPair.emplace_back(i, j);
        }
    }
    // return if no there is no shrinkage
    if (shrinkPair.empty()) return shrinkage;
    // build independent partitions
    std::vector<VertexID> trivialPartition(n);
    for (int i = 0; i < n; ++i)
        trivialPartition[i] = i;
    std::vector<std::vector<VertexID>> historyPartition;
    historyPartition.push_back(trivialPartition);
    int i = 0;
    std::vector<int> pos(shrinkPair.size(), 0);
    while (i >= 0) {
        while (i < shrinkPair.size() && pos[i] < shrinkPair.size()) {
            Edge nextPair = shrinkPair[pos[i]];
            ++pos[i];
            std::vector<VertexID> newPartition = mergePair(independentPartitions, historyPartition.back(), nextPair, matrix, t);
            // try to add new Partition
            if (!newPartition.empty()) {
                historyPartition.push_back(newPartition);
                independentPartitions.push_back(newPartition);
                ++i;
                if (i < shrinkPair.size())
                    pos[i] = pos[i - 1];
            }
        }
        historyPartition.pop_back();
        --i;
    }
    // now we have independent partitions. for each independent partition, compute a shrinkage.
    for (const auto &partition : independentPartitions) {
        shrinkage.push_back(p.shrink(partition));
    }
#ifdef ONLY_PLAN
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsedSeconds = end - start;
    gEquationTime += elapsedSeconds.count();
#endif
    return shrinkage;
}

std::vector<Tree> relabelTrees(const std::vector<Tree> &trees, VertexID *mapping) {
    std::vector<Tree> result(trees.size());
    for (int i = 0; i < trees.size(); ++i)
        result[i] = trees[i].relabel(mapping);

    return result;
}

// trees are precomputed rooted trees for the pattern, all tree are all valid decompositions of the pattern
// p is only used for rebuilding
void adjustAggreInfo(std::vector<Tree> &trees, bool sign, const VertexID *aggreV, const int *aggreWeight, ui aggreSize,
                     const Pattern &p, std::vector<Tree> &visitedDecomp, bool useTriangle) {
    int orbitType = trees[0].getOrbitType();
    if (orbitType == 0) {
        trees[0].adjustWeight(0, sign, 99, 99, 1);
        return;
    }

    if (orbitType == 1) {
        for (int i = 0; i < aggreSize; ++i) {
            bool newRoot = true; //check whether roots in trees cover aggreV. If not, trees need to be rebuilt.
            VertexID u = aggreV[i];
            int pos = 99;
            for (Tree &t: trees) {
                if (t.posInAggreV(u, pos)) {
                    newRoot = false;
                    t.adjustWeight(pos, sign, u, 99, aggreWeight[i]);
                    break;
                }
            }
            if (pos == 99) {
                for (Tree &t: trees) {
                    if (t.rootContains(u)) {
                        newRoot = false;
                        t.adjustWeight(pos, sign, u, 99, aggreWeight[i]);
                        break;
                    }
                }
            }
            if (newRoot) {
                // need to build a new tree from the new root.
                VertexID root = trees[0].getNodesContainV(u)[0];
                Tree newT(trees[0]);
                newT.setRoot(root, p.u);
                newT.setAggreInfo(orbitType, u, 99, sign, aggreWeight[i], p);
                // set the global order and local order
                treeBestOrder(p, newT, visitedDecomp, useTriangle);
                trees.push_back(newT);
            }
        }
    }
    else {
        for (int i = 0; i < aggreSize; i = i + 2) {
            bool newRoot = true; //check whether roots in trees cover aggreV. If not, trees need to be rebuilt.
            VertexID u = aggreV[i], v = aggreV[i + 1];
            int pos = 99;
            for (Tree &t: trees) {
                if (t.posInAggreV(u, v, pos)) {
                    newRoot = false;
                    t.adjustWeight(pos, sign, u, v, aggreWeight[i / 2]);
                    break;
                }
            }
            if (pos == 99) {
                for (Tree &t: trees) {
                    if (t.rootContains(u) && t.rootContains(v)) {
                        newRoot = false;
                        t.adjustWeight(pos, sign, u, v, aggreWeight[i / 2]);
                        break;
                    }
                }
            }
            if (newRoot) {
                // need to build a new tree from the new root.
                std::vector<VertexID> uRoots = trees[0].getNodesContainV(u);
                std::vector<VertexID> vRoots = trees[0].getNodesContainV(v);
                VertexID root;
                for (VertexID uRoot: uRoots) {
                    if (std::find(vRoots.begin(), vRoots.end(), uRoot) != vRoots.end()) {
                        root = uRoot;
                        break;
                    }
                }
                Tree newT(trees[0]);
                newT.setRoot(root, p.u);
                newT.setAggreInfo(orbitType, u, v, sign, aggreWeight[i / 2], p);
                // set the global order and local order
                std::vector<Tree> minWidth2Trees = setGlobalOrderForT(newT, p);
                trees.push_back(bestLocalOrder(minWidth2Trees, p));
            }
        }
    }
}

// generate the equation to compute the orbit count for a single query (using DAG)
// the input p is undirected and multi aggregation
bool genEquation(const PatternGraph &p, std::map<int, std::vector<Pattern>> &patterns,
                 std::map<int, std::vector<std::vector<Tree>>> &trees, ConNode &cn, bool useTriangle,
                 bool symmetryBreaking, bool prefix, bool useDirected) {
    int orbitType = p.getOrbitType();
    // visited decompositions for the root pattern
    std::vector<Tree> visitedDecomp;
    // compute directed multi aggregation and decide whether to use directed
    // the patterns that need to enumerate
    bool useDAG = false;
    int numDAGs = 0;
    std::vector<PatternGraph> inGraphs, outGraphs;
    Pattern rootPattern(p);
    // choose the best decomposition for the undirected P
    std::vector<Tree> rootAllTree = getAllTree(rootPattern.u);
    std::vector<Tree> rootBestDecomp = getBestDecomposition(rootPattern, rootAllTree, visitedDecomp, true, prefix,
                                                            useTriangle);
    int uFactor = p.getAutoSize() / rootBestDecomp[0].getMultiFactor() / rootBestDecomp[0].getAggreWeight().size();
    Tree &rt = rootBestDecomp[0];
    rt.rebuildCut();
    if (useTriangle) cn = ConNode(p, rt);
    if (cn.num != 0) {
        int divideFactor = p.getDivideFactor();
        patterns[divideFactor].push_back(rootPattern);
        trees[divideFactor].push_back(rootBestDecomp);
        conNodeDecompose(cn, p, rt.getNode(0), symmetryBreaking, prefix, useTriangle);
        CanonType canonValue = p.getCanonValue();
        gCanon2Pattern[canonValue] = rootPattern;
        gCanon2Tree[canonValue] = rootBestDecomp;
        gCanon2Shrinkage[canonValue] = computeShrinkages(rt, rootPattern);
        return false;
    }
    if (p.getNumVertices() < 6 && uFactor > 1 && symmetryBreaking && rootBestDecomp[0].getFhw() >= 2.0) {
        Pattern rootMulti = rootPattern;
        rootMulti.setMultiAggre();
        rootMulti.u.genDAGs(inGraphs, outGraphs);
        numDAGs = (int)inGraphs.size();
        // decide whether to use directed patterns
        int dFactor = 1;
        for (const auto &pt: outGraphs)
            if (pt.getAutoSize() > dFactor)
                dFactor = pt.getAutoSize();
        if (uFactor > dFactor) useDAG = true;
    }
    visitedDecomp.clear();
    if (useDirected && useDAG) {
//        std::cout << "num DAG: " << numDAGs << std::endl;
        // here trees are relabeled according to canonical labels
        std::map<CanonType, std::vector<Tree>> canon2AllTree;
        // the following maps use the divide factor and the canonical value of the out DAG as key
        std::map<int, std::map<CanonType, VertexID *>> canon2V2L;
        std::map<int, std::map<CanonType, int>> canon2Pos;
        for (int i = 0; i < numDAGs; ++i) {
            Pattern pt(p, inGraphs[i], outGraphs[i]);
            pt.setMultiAggre();
            int divideFactor = outGraphs[i].getAutoSize();
            std::vector<Tree> rootedTrees;
            bool sign = true;   // true: +, false: -
            std::queue<Pattern> q;
            q.push(pt);
            while (!q.empty()) {
                std::vector<Pattern> pts;
                // pop all patterns in queue
                while (!q.empty()) {
                    pts.push_back(q.front());
                    q.pop();
                }
                for (const auto &s: pts) {
                    ui numVertices = s.u.getNumVertices();
                    CanonType directedCanonValue = s.out.getCanonValue();
                    if (canon2Pos.find(divideFactor) != canon2Pos.end()
                        && canon2Pos[divideFactor].find(directedCanonValue) != canon2Pos[divideFactor].end()) {
                        // we find that there is a previously computed pattern that is isomorphic to s
                        // therefore we only need to add the aggregation information to the existing decomposition
                        VertexID *currentV2L = s.out.getV2L();
                        VertexID *oldV2L = canon2V2L[divideFactor][directedCanonValue];
                        // maps current vertices to old vertices
                        VertexID *mapping = new VertexID[numVertices];
                        VertexID *l2OldV = new VertexID[numVertices];
                        for (int j = 0; j < numVertices; ++j)
                            l2OldV[oldV2L[j]] = j;
                        for (int j = 0; j < numVertices; ++j)
                            mapping[j] = l2OldV[currentV2L[j]];
                        int pos = canon2Pos[divideFactor][directedCanonValue];
//                    checkMapping(s, patterns[divideFactor][pos], mapping);
                        ui aggreSize;
                        const VertexID *aggreV = s.out.getAggreV(aggreSize);
                        VertexID *newAggreV = new VertexID[aggreSize];
                        const int *aggreWeight = s.out.getAggreWeight();
                        for (int j = 0; j < aggreSize; ++j)
                            newAggreV[j] = mapping[aggreV[j]];
                        adjustAggreInfo(trees[divideFactor][pos], sign, newAggreV, aggreWeight,
                                        aggreSize, patterns[divideFactor][pos], visitedDecomp, useTriangle);
                        VertexID *reverseMapping = new VertexID[numVertices];
                        for (VertexID j = 0; j < numVertices; ++j) {
                            reverseMapping[mapping[j]] = j;
                        }
                        Tree relabeledTree = trees[divideFactor][pos][0].relabel(reverseMapping);
                        delete[] l2OldV;
                        delete[] mapping;
                        delete[] newAggreV;
                        delete[] reverseMapping;
                        std::vector<Pattern> shrinkage = computeShrinkages(relabeledTree, s);
                        for (auto &shr: shrinkage) {
                            if (shr.u.getNumVertices() != 0) {
                                q.push(shr);
                            }
                        }
                        continue;
                    }
                    std::vector<Tree> allTree;
                    CanonType canonValue = s.u.getCanonValue();
                    if (canon2AllTree.find(canonValue) != canon2AllTree.end()) {
                        VertexID *mapping = new VertexID[numVertices];
                        VertexID *v2l = s.u.getV2L();
                        // mapping is reversing the bijection v2l
                        for (int j = 0; j < numVertices; ++j) {
                            mapping[v2l[j]] = j;
                        }
                        allTree = relabelTrees(canon2AllTree[canonValue], mapping);
                        delete[] mapping;
                    }
                    else {
                        allTree = getAllTree(s.u);
                        std::map<int, CanonType> id2CanonV;
                        std::map<int, int*> id2Orbits;
                        std::map<int, ui> id2AutoSize;
                        for (auto &t: allTree) {
                            for (VertexID nID = 0; nID < t.getNumNodes(); ++nID) {
                                t.setNodeCanon(id2CanonV, id2Orbits, id2AutoSize, nID, s);
                            }
                        }
                        for (auto it = id2Orbits.begin(); it != id2Orbits.end(); ++it) {
                            delete[] it->second;
                        }
                        canon2AllTree[canonValue] = relabelTrees(allTree, s.u.getV2L());
                    }
                    if (canon2V2L.find(divideFactor) == canon2V2L.end() ||
                        canon2V2L[divideFactor].find(directedCanonValue) == canon2V2L[divideFactor].end()) {
                        canon2V2L[divideFactor][directedCanonValue] = new VertexID[numVertices];
                        memcpy(canon2V2L[divideFactor][directedCanonValue], s.out.getV2L(), sizeof(VertexID) * numVertices);
                    }
                    rootedTrees = getBestDecomposition(s, allTree, visitedDecomp, sign, prefix, useTriangle);
                    // the pattern and shrinkages are therefore added to patterns[divideFactor], so as trees.
                    canon2Pos[divideFactor][directedCanonValue] = (int)patterns[divideFactor].size();
                    patterns[divideFactor].push_back(s);
                    trees[divideFactor].push_back(rootedTrees);
                    std::vector<Pattern> shrinkage = computeShrinkages(rootedTrees[0], s);
                    for (auto &shr: shrinkage) {
                        if (shr.u.getNumVertices() != 0) {
                            q.push(shr);
                        }
                    }
                }
                sign = !sign;
            }
        }
        for (auto it1 : canon2V2L) {
            for (auto it2: it1.second)
                delete[] it2.second;
        }
    }
    else {
        bool sign = false;
        int divideFactor = p.getDivideFactor();
        patterns[divideFactor].push_back(rootPattern);
        trees[divideFactor].push_back(rootBestDecomp);
        visitedDecomp.clear();
        std::queue<Pattern> q;
        std::vector<Pattern> rootShrinkage = computeShrinkages(rootBestDecomp[0], rootPattern);
        for (auto &shr: rootShrinkage) {
            if (shr.u.getNumVertices() != 0) {
                shr.u.setSingleAggre();
                q.push(shr);
            }
        }
        while (!q.empty()) {
            std::vector<Pattern> pts;
            // pop all patterns in queue
            while (!q.empty()) {
                pts.push_back(q.front());
                q.pop();
            }
            for (auto &s : pts) {
                // s is always single-aggregation
                ui numVertices = s.u.getNumVertices();
                CanonType canonValue = s.u.getCanonValue();
                bool exists = false;
                for (int i = 0; i < patterns[divideFactor].size(); ++i) {
                    const Pattern pt = patterns[divideFactor][i];
                    if (canonValue == pt.u.getCanonValue()) {
                        exists = true;
                        const std::vector<Tree> bestTree = gCanon2Tree[canonValue];
                        int multiFactor = bestTree[0].getMultiFactor();
                        if (!sign) multiFactor = -multiFactor;
                        trees[divideFactor][i][0].adjustMultiFactor(multiFactor);
                        std::vector<Pattern> shrinkage = gCanon2Shrinkage[canonValue];
                        for (auto &shr: shrinkage) {
                            if (shr.u.getNumVertices() != 0) {
                                q.push(shr);
                            }
                        }
                    }
                    if (exists) break;
                }
                if (exists) continue;
                if (gCanon2Tree.find(canonValue) != gCanon2Tree.end()) {
                    std::vector<Tree> rootedTrees = gCanon2Tree[canonValue];
                    if (!sign) {
                        for (Tree &t: rootedTrees) t.setMultiFactor(-t.getMultiFactor());
                    }
                    patterns[divideFactor].push_back(gCanon2Pattern[canonValue]);
                    trees[divideFactor].push_back(rootedTrees);
                    std::vector<Pattern> shrinkage = gCanon2Shrinkage[canonValue];
                    for (auto &shr: shrinkage) {
                        if (shr.u.getNumVertices() != 0) {
                            shr.u.setSingleAggre();
                            q.push(shr);
                        }
                    }
                }
                else {
                    std::vector<Tree> allTree = getAllTree(s.u);
                    std::vector<Tree> rootedTrees = getBestDecomposition(s, allTree, visitedDecomp, sign, prefix,
                                                                         useTriangle);
                    for (Tree &t : rootedTrees)
                        t.adjustWeight();
                    patterns[divideFactor].push_back(s);
                    trees[divideFactor].push_back(rootedTrees);
                    std::vector<Pattern> shrinkage = computeShrinkages(rootedTrees[0], s);
                    for (auto &shr: shrinkage) {
                        if (shr.u.getNumVertices() != 0) {
                            shr.u.setSingleAggre();
                            q.push(shr);
                        }
                    }
                    gCanon2Pattern[canonValue] = s;
                    for (Tree &t: rootedTrees) {
                        if (t.getMultiFactor() < 0) t.setMultiFactor(-t.getMultiFactor());
                    }
                    gCanon2Tree[canonValue] = rootedTrees;
                    gCanon2Shrinkage[canonValue] = shrinkage;
                }

            }
            sign = !sign;
        }
    }
    for (auto it = patterns.begin(); it != patterns.end(); ++it) {
        int divideFactor = it->first;
        for (int i = 0; i < it->second.size(); ++i) {
            for (int j = 0; j < trees[divideFactor][i].size(); ++j) {
                Tree &t = trees[divideFactor][i][j];
                if (p.getNumVertices() <= 5) {
                    t.setPrefixKeyOrbit(it->second[i]);
                }
                if (!symmetryBreaking) {
                    int multiFactor;
                    if (i != 0)
                        multiFactor = gCanon2Tree[patterns[divideFactor][i].u.getCanonValue()][0].getMultiFactor();
                    else multiFactor = t.getMultiFactor();
                    t.dropRules(patterns[divideFactor][i], useTriangle, multiFactor);
                }
            }
        }
    }
    return useDAG;
}

std::vector<VertexID> mergePairHomo(
        const std::vector<std::vector<VertexID>> &independentPartitions,
        const std::vector<VertexID> &oldPartition,
        Edge nextPair,
        const PatternGraph &p
) {
    std::vector<VertexID> newPartition;
    // 1. for the next pair (u1, u2), check whether u1 has edges with u' in the partition of u2,
    // whether u1 and u' in the partition of u2 is in the same tree node, and vice versa.
    // If so, next pair is invalid
    VertexID u1 = nextPair.first, u2 = nextPair.second;
    VertexID p1 = oldPartition[u1], p2 = oldPartition[u2];
    if (p1 == p2) return newPartition;
    std::vector<VertexID> verticesInP1;
    for (VertexID i = 0; i < oldPartition.size(); ++i) {
        if (oldPartition[i] == p1)
            verticesInP1.push_back(i);
    }
    std::vector<VertexID> verticesInP2;
    for (VertexID i = 0; i < oldPartition.size(); ++i) {
        if (oldPartition[i] == p2)
            verticesInP2.push_back(i);
    }
    for (VertexID u1: verticesInP1)
        for (VertexID u2 : verticesInP2)
            if (p.isEdge(u1, u2)) return newPartition;
    // 2. check whether the new partition is not in the independentPartitions
    // to build the new partition, merge verticesInP1 and verticesInP2, set their "partition value"
    // as the smaller value of p1 and p2
    newPartition = oldPartition;
    VertexID pNew = p1 < p2 ? p1 : p2;
    for (VertexID u : verticesInP1)
        newPartition[u] = pNew;
    for (VertexID u : verticesInP2)
        newPartition[u] = pNew;
    for (const auto &partition: independentPartitions)
        if (partition == newPartition) {
            newPartition.clear();
            return newPartition;
        }
    return newPartition;
}

std::vector<Pattern> homoShrinkage(const Pattern &p) {
    std::vector<Pattern> shrinkage;
    ui n = p.u.getNumVertices();
    std::vector<Edge> shrinkPair;   // pair of vertices that can shrinks
    std::vector<std::vector<VertexID>> independentPartitions;   // each trivialPartition is an independent set of p
    for (VertexID i = 0; i < n; ++i) {
        for (VertexID j = i + 1; j < n; ++j) {
            if (!p.u.isEdge(i, j))
                shrinkPair.emplace_back(i, j);
        }
    }
    // return if no there is no shrinkage
    if (shrinkPair.empty()) return shrinkage;
    // build independent partitions
    std::vector<VertexID> trivialPartition(n);
    for (int i = 0; i < n; ++i)
        trivialPartition[i] = i;
    std::vector<std::vector<VertexID>> historyPartition;
    historyPartition.push_back(trivialPartition);
    int i = 0;
    std::vector<int> pos(shrinkPair.size(), 0);
    while (i >= 0) {
        while (i < shrinkPair.size() && pos[i] < shrinkPair.size()) {
            Edge nextPair = shrinkPair[pos[i]];
            ++pos[i];
            std::vector<VertexID> newPartition = mergePairHomo(independentPartitions, historyPartition.back(), nextPair, p.u);
            // try to add new Partition
            if (!newPartition.empty()) {
                historyPartition.push_back(newPartition);
                independentPartitions.push_back(newPartition);
                ++i;
                if (i < shrinkPair.size())
                    pos[i] = pos[i - 1];
            }
        }
        historyPartition.pop_back();
        --i;
    }
    // now we have independent partitions. for each independent partition, compute a shrinkage.
    for (const auto &partition : independentPartitions) {
        shrinkage.push_back(p.shrink(partition));
    }
    return shrinkage;
}

std::vector<Pattern> homoShrinkage(const Pattern &p, std::vector<int> &mu) {
    std::vector<Pattern> shrinkage;
    ui n = p.u.getNumVertices();
    std::vector<Edge> shrinkPair;   // pair of vertices that can shrinks
    std::vector<std::vector<VertexID>> independentPartitions;   // each trivialPartition is an independent set of p
    for (VertexID i = 0; i < n; ++i) {
        for (VertexID j = i + 1; j < n; ++j) {
            if (!p.u.isEdge(i, j))
                shrinkPair.emplace_back(i, j);
        }
    }
    // return if no there is no shrinkage
    if (shrinkPair.empty()) return shrinkage;
    // build independent partitions
    std::vector<VertexID> trivialPartition(n);
    for (int i = 0; i < n; ++i)
        trivialPartition[i] = i;
    std::vector<std::vector<VertexID>> historyPartition;
    historyPartition.push_back(trivialPartition);
    int i = 0;
    std::vector<int> pos(shrinkPair.size(), 0);
    while (i >= 0) {
        while (i < shrinkPair.size() && pos[i] < shrinkPair.size()) {
            Edge nextPair = shrinkPair[pos[i]];
            ++pos[i];
            std::vector<VertexID> newPartition = mergePairHomo(independentPartitions, historyPartition.back(), nextPair, p.u);
            // try to add new Partition
            if (!newPartition.empty()) {
                historyPartition.push_back(newPartition);
                independentPartitions.push_back(newPartition);
                ++i;
                if (i < shrinkPair.size())
                    pos[i] = pos[i - 1];
            }
        }
        historyPartition.pop_back();
        --i;
    }
    // now we have independent partitions. for each independent partition, compute a shrinkage.
    for (const auto &partition : independentPartitions) {
        Pattern shr = p.shrink(partition);
        int mobius = 1;
        VertexID lastVertex = partition[0];
        int numLattice = 1;
        std::vector<std::vector<int>> groups;
        for (int j = 0; j < partition.size(); ++j) {
            groups.emplace_back();
        }
        for (int j = 0; j < partition.size(); ++j) {
            groups[partition[j]].push_back(j);
        }
        for (int j = 0; j < groups.size(); ++j) {
            int sz = groups[j].size();
            if (sz != 0) {
                if (sz % 2 == 0) mobius *= -1;
                if (sz > 1) {
                    for (int k = 1; k < sz; ++k) {
                        mobius *= k;
                    }
                }
            }
        }
        bool newShr = true;
        for (int j = 0; j < shrinkage.size(); ++j) {
            const Pattern &s = shrinkage[j];
            if (s.useDAG() && s.out.getCanonValue() == shr.out.getCanonValue() &&
                s.out.getOrbit(0) == shr.out.getOrbit(0)) {
                mu[j + 1] += mobius;
                newShr = false;
            }
            else if (s.u.getCanonValue() == shr.u.getCanonValue() &&
                     s.u.getOrbit(0) == shr.u.getOrbit(0)) {
                mu[j + 1] += mobius;
                newShr = false;
            }
        }
        if (newShr) {
            shrinkage.push_back(shr);
            mu.push_back(mobius);
        }

    }
    return shrinkage;
}

void homoEquation(const PatternGraph &p, std::map<int, std::vector<Pattern>> &patterns,
                  std::map<int, std::vector<std::vector<Tree>>> &trees, bool prefix) {
    int orbitType = p.getOrbitType();
    // the following maps use the divide factor and the canonical value of the out DAG as key
    // visited decompositions for the root pattern
    std::vector<Tree> visitedDecomp;
    // compute directed multi aggregation and decide whether to use directed
    // the patterns that need to enumerate
    Pattern rootPattern(p);
    visitedDecomp.clear();
    bool sign = true;
    int divideFactor = p.getDivideFactor();
    visitedDecomp.clear();
    rootPattern.setSingleAggre();
    std::vector<Tree> allTree = getAllTree(rootPattern.u);
    std::vector<Tree> rootedTrees = getBestDecomposition(rootPattern, allTree, visitedDecomp, sign, prefix, true);
    for (Tree &t : rootedTrees)
        t.adjustWeight();
    std::vector<int> mu;
    mu.push_back(1);
    std::vector<Pattern> shrinkage = homoShrinkage(rootPattern, mu);
    patterns[divideFactor].push_back(rootPattern);
    trees[divideFactor].push_back(rootedTrees);
    for (auto &shr: shrinkage) {
        shr.u.setSingleAggre();
        allTree = getAllTree(shr.u);
        rootedTrees = getBestDecomposition(shr, allTree, visitedDecomp, sign, prefix, true);
        for (Tree &t : rootedTrees)
            t.adjustWeight();
        patterns[divideFactor].push_back(shr);
        trees[divideFactor].push_back(rootedTrees);
    }
    for (auto it = patterns.begin(); it != patterns.end(); ++it) {
        int divideFactor = it->first;
        for (int i = 0; i < it->second.size(); ++i) {
            for (int j = 0; j < trees[divideFactor][i].size(); ++j) {
                Tree &t = trees[divideFactor][i][j];
                if (p.getNumVertices() <= 5) {
                    t.setPrefixKeyOrbit(it->second[i]);
                }
                int multiFactor = t.getMultiFactor();
                t.dropRules(patterns[divideFactor][i], true, multiFactor);
                t.setMultiFactor(t.getMultiFactor() * mu[i]);
            }
        }
    }
}
