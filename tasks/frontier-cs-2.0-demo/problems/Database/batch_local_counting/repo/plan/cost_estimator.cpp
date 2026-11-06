//
// Created by anonymous author on 2022/8/8.
//

#include "cost_estimator.h"

std::vector<std::vector<VertexID>>
generateLocalOrders(const Node &tau, const PatternGraph &p, const std::vector<VertexID> &partitionOrder) {
    std::vector<std::vector<VertexID>> result;
    std::vector<VertexID> localVertices;
    ui numVertices = tau.numVertices;
    VertexID *vertices = tau.vertices;
    for (int i = 0; i < numVertices; ++i) {
        if (std::find(partitionOrder.begin(), partitionOrder.end(), vertices[i]) == (partitionOrder.end()))
            localVertices.push_back(vertices[i]);
    }
    do {
        for (int i = 0; i < localVertices.size(); ++i) {
            bool connected = false;
            VertexID u = localVertices[i];
            for (VertexID w : partitionOrder) {
                if (p.isEdge(u, w)) {
                    connected = true;
                    break;
                }
            }
            if (!connected) {
                if (i == 0 && partitionOrder.empty()) connected = true;
                else {
                    for (int j = 0; j < i; ++j) {
                        if (p.isEdge(u, localVertices[j])) {
                            connected = true;
                            break;
                        }
                    }
                }
            }
            if (connected) {
                if (i == numVertices - partitionOrder.size() - 1) result.push_back(localVertices);
                else continue;
            }
            else break;
        }
    } while (std::next_permutation(localVertices.begin(), localVertices.end()));

    return result;
}

// given a local order, get its degree sequence and the position of the source
std::vector<ui> getDegSeqAndSrcPos(const std::vector<VertexID> &localOrder, int &srcPos, const Pattern &p, VertexID *vertices,
                                   ui numVertices) {
    // -1 implies that the source vertex is in cut, or we do not consider directed graph
    srcPos = -1;
    std::vector<int> local2Pos(p.u.getNumVertices());
    for (int i = 0; i < localOrder.size(); ++i) {
        local2Pos[localOrder[i]] = i;
    }
    std::vector<ui> degreeSeq(localOrder.size(), 0);
    for (VertexID u: localOrder) {
        for (int i = 0; i < numVertices; ++i) {
            if (p.u.isEdge(u, vertices[i]))
                ++degreeSeq[local2Pos[u]];
        }
    }
    if (p.useDAG()) {
        for (int i = 0; i < localOrder.size(); ++i) {
            if (p.in.degree(localOrder[i]) == 0) {
                srcPos = i;
                break;
            }
        }
    }

    return degreeSeq;
}

// given a local order, get its degree sequence and the number of in-neighbor lists that need to be used
std::vector<ui> evaluateLocalOrder(const std::vector<VertexID> &partitionOrder,
                                   const std::vector<VertexID> &localOrder, int &inNum, const Pattern &p, VertexID *vertices, ui numVertices) {
    inNum = 0;
    std::vector<int> local2Pos(p.u.getNumVertices());
    for (int i = 0; i < localOrder.size(); ++i) {
        local2Pos[localOrder[i]] = i;
    }
    std::vector<ui> degreeSeq(localOrder.size(), 0);
    for (VertexID u: localOrder) {
        for (int i = 0; i < numVertices; ++i) {
            if (p.u.isEdge(u, vertices[i]))
                ++degreeSeq[local2Pos[u]];
        }
    }
    if (p.useDAG()) {
        for (int i = 0; i < localOrder.size(); ++i) {
            for (int j = 0; j < partitionOrder.size(); ++j) {
                if (p.in.isEdge(partitionOrder[j], localOrder[i]))
                    ++inNum;
            }
            for (int j = 0; j < i; ++j) {
                if (p.in.isEdge(localOrder[j], localOrder[i]))
                    ++inNum;
            }
        }
    }

    return degreeSeq;
}

bool firstSequenceLarger(const std::vector<ui> &seq1, const std::vector<ui> &seq2) {
    if (seq2.empty()) return true;
    for (int i = 0; i < seq1.size(); ++i) {
        if (seq1[i] > seq2[i]) return true;
        if (seq1[i] < seq2[i]) return false;
    }

    return false;
}

// priority: 1. the source vertex, if it is directed and source vertex is not in prefix
// 2. vertex with higher degree. vertices with higher priority should be put earlier.
void simpleLocalOrder(Tree &t, const Pattern &p) {
    const std::vector<VertexID> &postOrder = t.getPostOrder();
    const std::vector<int> &partitionPos = t.getPartitionPos();
    const std::vector<std::vector<VertexID>> &globalOrder = t.getGlobalOrder();
    int numPartition = (int)partitionPos.size();
    std::vector<ui> partitionNumNodes(numPartition);
    for (int i = 0; i < numPartition; ++i) {
        if (i == 0) partitionNumNodes[i] = partitionPos[i] + 1;
        else partitionNumNodes[i] = partitionPos[i] - partitionPos[i - 1];
    }
    for (int i = 0; i < numPartition; ++i) {
        ui numNodes = partitionNumNodes[i];
        int position;
        if (i == 0) position = 0;
        else position = partitionPos[i - 1] + 1;
        for (int j = 0; j < numNodes; ++j) {
            VertexID nID = postOrder[position + j];
            ui numVertices;
            VertexID *vertices = t.getVertices(nID, numVertices);
            std::vector<std::vector<VertexID>> allLocalOrder;
            allLocalOrder = generateLocalOrders(t.getNode(nID), p.u, globalOrder[i]);
            // choose the local order based on priority, break ties randomly
            int minPos = 99, orderPos = 0;
            std::vector<ui> maxDegreeSeq;
            for (int k = 0; k < allLocalOrder.size(); ++k) {
                int srcPos;
                std::vector<ui> degreeSeq = getDegSeqAndSrcPos(allLocalOrder[k], srcPos, p, vertices, numVertices);
                if (srcPos < minPos || firstSequenceLarger(degreeSeq, maxDegreeSeq)) {
                    minPos = srcPos;
                    maxDegreeSeq = degreeSeq;
                    orderPos = k;
                }
            }
            t.setLocalOrder(nID, allLocalOrder[orderPos]);
        }
    }
    t.initPoses(p, false);
}

// for all min width 2 decompositions, choose the one that has the best local order.
// priority: 1. the number of in-neighbors visited 2. key: prefer vertex key than edge key 3. vertex with higher degree
// for each i,j choose a k that has the best local order; for each i,
std::vector<Tree> bestLocalOrder(std::vector<std::vector<std::vector<Tree>>> &minWidth2Tree, const Pattern &p) {
    int firstIndex = -1;
    std::vector<int> thirdIndex;
    int smallestInNum = 99;
    std::vector<ui> bestDegSeq;
    for (int i1 = 0; i1 < minWidth2Tree.size(); ++i1) {
        int i1LargestNum = -1;
        std::vector<int> i1ThirdIndex(minWidth2Tree[i1].size());
        std::vector<int> i1SmallestTreeNum(minWidth2Tree[i1].size(), 99);
        std::vector<int> i1LargestLength(minWidth2Tree[i1].size(), 0);
        for (int i2 = 0; i2 < minWidth2Tree[i1].size(); ++i2) {
            for (int i3 = 0; i3 < minWidth2Tree[i1][i2].size(); ++i3) {
                Tree &t = minWidth2Tree[i1][i2][i3];
                // for every tree node choose the best local order
                const std::vector<VertexID> &postOrder = t.getPostOrder();
                const std::vector<int> &partitionPos = t.getPartitionPos();
                const std::vector<std::vector<VertexID>> &globalOrder = t.getGlobalOrder();
                int numPartition = (int)partitionPos.size();
                std::vector<ui> partitionNumNodes(numPartition);
                for (int i = 0; i < numPartition; ++i) {
                    if (i == 0) partitionNumNodes[i] = partitionPos[i] + 1;
                    else partitionNumNodes[i] = partitionPos[i] - partitionPos[i - 1];
                }
                int treeTotalInNum = 0;
                int treeOrderLength = 0;
                for (int i = 0; i < numPartition; ++i) {
                    ui numNodes = partitionNumNodes[i];
                    int position;
                    if (i == 0) position = 0;
                    else position = partitionPos[i - 1] + 1;
                    treeOrderLength += (int)globalOrder[i].size();
                    for (int j = 0; j < numNodes; ++j) {
                        VertexID nID = postOrder[position + j];
                        ui numVertices;
                        VertexID *vertices = t.getVertices(nID, numVertices);
                        std::vector<std::vector<VertexID>> allLocalOrder;
                        allLocalOrder = generateLocalOrders(t.getNode(nID), p.u, globalOrder[i]);
                        // choose the local order based on priority
                        int minNum = 99, orderPos = 0;
                        std::vector<ui> maxDegreeSeq;
                        for (int k = 0; k < allLocalOrder.size(); ++k) {
                            int inNum;
                            std::vector<ui> degreeSeq = evaluateLocalOrder(globalOrder[i], allLocalOrder[k], inNum, p, vertices, numVertices);
                            if (inNum < minNum || firstSequenceLarger(degreeSeq, maxDegreeSeq)) {
                                minNum = inNum;
                                maxDegreeSeq = degreeSeq;
                                orderPos = k;
                            }
                        }
                        treeTotalInNum += minNum;
                        t.setLocalOrder(nID, allLocalOrder[orderPos]);
                    }
                }
                if (treeTotalInNum < i1SmallestTreeNum[i2] || (treeTotalInNum == i1SmallestTreeNum[i2] &&
                                                               treeOrderLength > i1LargestLength[i2])) {
                    i1SmallestTreeNum[i2] = treeTotalInNum;
                    i1LargestLength[i2] = treeOrderLength;
                    i1ThirdIndex[i2] = i3;
                }
            }
        }
        for (int num: i1SmallestTreeNum)
            if (num > i1LargestNum)
                i1LargestNum = num;
        if (i1LargestNum < smallestInNum) {
            firstIndex = i1;
            thirdIndex = i1ThirdIndex;
            smallestInNum = i1LargestNum;
        }
    }
    std::vector<Tree> result;
    for (int i = 0; i < minWidth2Tree[firstIndex].size(); ++i) {
        minWidth2Tree[firstIndex][i][thirdIndex[i]].initPoses(p, false);
        result.push_back(minWidth2Tree[firstIndex][i][thirdIndex[i]]);
    }
//    printf("the largest number of in-neighbors is: %d\n", smallestInNum);

    return result;
}

Tree bestLocalOrder(std::vector<Tree> &minWidth2Tree, const Pattern &p) {
    int index = -1;
    int smallestInNum = 99;
    int largestLength = 0;
    for (int l = 0; l < minWidth2Tree.size(); ++l) {
        Tree &t = minWidth2Tree[l];
        // for every tree node choose the best local order
        const std::vector<VertexID> &postOrder = t.getPostOrder();
        const std::vector<int> &partitionPos = t.getPartitionPos();
        const std::vector<std::vector<VertexID>> &globalOrder = t.getGlobalOrder();
        int numPartition = (int)partitionPos.size();
        std::vector<ui> partitionNumNodes(numPartition);
        for (int i = 0; i < numPartition; ++i) {
            if (i == 0) partitionNumNodes[i] = partitionPos[i] + 1;
            else partitionNumNodes[i] = partitionPos[i] - partitionPos[i - 1];
        }
        int treeTotalInNum = 0;
        int treeOrderLength = 0;
        for (int i = 0; i < numPartition; ++i) {
            ui numNodes = partitionNumNodes[i];
            int position;
            if (i == 0) position = 0;
            else position = partitionPos[i - 1] + 1;
            treeOrderLength += (int)globalOrder[i].size();
            for (int j = 0; j < numNodes; ++j) {
                VertexID nID = postOrder[position + j];
                ui numVertices;
                VertexID *vertices = t.getVertices(nID, numVertices);
                std::vector<std::vector<VertexID>> allLocalOrder;
                allLocalOrder = generateLocalOrders(t.getNode(nID), p.u, globalOrder[i]);
                // choose the local order based on priority
                int minNum = 99, orderPos = 0;
                std::vector<ui> maxDegreeSeq;
                for (int k = 0; k < allLocalOrder.size(); ++k) {
                    int inNum;
                    std::vector<ui> degreeSeq = evaluateLocalOrder(globalOrder[i], allLocalOrder[k], inNum, p, vertices, numVertices);
                    if (inNum < minNum || firstSequenceLarger(degreeSeq, maxDegreeSeq)) {
                        minNum = inNum;
                        maxDegreeSeq = degreeSeq;
                        orderPos = k;
                    }
                }
                treeTotalInNum += minNum;
                t.setLocalOrder(nID, allLocalOrder[orderPos]);
            }
        }
        if (treeTotalInNum < smallestInNum || (treeTotalInNum == smallestInNum &&
                                               treeOrderLength > largestLength)) {
            smallestInNum = treeTotalInNum;
            index = l;
        }
    }
    minWidth2Tree[index].initPoses(p, false);
    return minWidth2Tree[index];
}

// if no edge key or the edge can be obtained without intersection, the position of the edge key is 0
// priority: if triangle is enabled, 1> deg seq 2> position of edge key 3> number of non-triangle in-neighbors
// if triangle is not enabled, 1> deg seq 2> position of edge key 3> number of in-neighbors
// if use triangle, numIn is computed as the number of in-neighbors in non-triangle intersections
void bestLocalOrder(const Tree &t, VertexID nID, const std::vector<VertexID> &partitionOrder, const Pattern &p,
                    std::vector<VertexID> &localOrder, ui &numIn, ui &edgePos, bool useTriangle) {
    const Node &tau = t.getNode(nID);
    std::vector<VertexID> localVertices;
    ui numVertices = tau.numVertices;
    VertexID *vertices = tau.vertices;
    ui minIn = 99, minEdgePos = 99;
    const std::vector<std::vector<VertexID>> &greaterRules = t.getGreaterRules();
    const std::vector<std::vector<VertexID>> &lessRules = t.getLessRules();
    ui childKey1 = 0, childKey2 = 0, key1 = 0, key2 = 0;
    ui keySize;
    if (nID == t.getRootID() && t.getOrbitType() == 2 && t.getAggreV().size() == 2) {
        key1 = t.getAggreV()[0];
        key2 = t.getAggreV()[1];
    }
    else {
        ui *key = t.getKey(nID, keySize);
        if (keySize == 2) {
            key1 = key[0];
            key2 = key[1];
        }
    }
    if (key1 == 0 && key2 == 0) {
        for (VertexID cID: t.getChild()[nID]) {
            ui *key = t.getKey(cID, keySize);
            if (keySize == 2) {
                childKey1 = key[0];
                childKey2 = key[1];
                break;
            }
        }
    }
    for (int i = 0; i < numVertices; ++i) {
        if (std::find(partitionOrder.begin(), partitionOrder.end(), vertices[i]) == (partitionOrder.end()))
            localVertices.push_back(vertices[i]);
    }
    if (childKey1 == 0 && childKey2 == 0 && key1 == 0 && key2 == 0) edgePos = 0;
    if (localVertices.size() < 2) {
        edgePos = 99;
        numIn = 99;
        localOrder = localVertices;
        return;
    }
    std::vector<ui> degreeInNode(p.u.getNumVertices(), 0);
    for (int i = 0; i < localVertices.size(); ++i) {
        VertexID u1 = localVertices[i];
        for (int j = 0; j < t.getNode(nID).numVertices; ++j) {
            if (p.u.isEdge(u1, t.getNode(nID).vertices[j]))
                ++degreeInNode[u1];
        }
    }
    std::vector<ui> maxDegSeq(localVertices.size(), 0);
    do {
        ui numInNeighbor = 0, ep = 0;
        std::vector<ui> degSeq(localVertices.size(), 0);
        std::vector<VertexID> nodeOrder = partitionOrder;
        for (VertexID u: localVertices) nodeOrder.push_back(u);
        std::vector<std::vector<int>> nodeInPos(nodeOrder.size());
        std::vector<std::vector<int>> nodeOutPos(nodeOrder.size());
        std::vector<std::vector<int>> nodeUnPos(nodeOrder.size());
        std::vector<int> numNeighbor(nodeOrder.size(), 0);
        std::vector<std::pair<int, int>> nodeTriPos(nodeOrder.size());
        for (int i = 0; i < nodeTriPos.size(); ++i) {
            nodeTriPos[i].first = -1;
            nodeTriPos[i].second = -1;
        }
        for (int i = 0; i < nodeOrder.size(); ++i) {
            bool connected = false;
            for (int j = 0; j <= i; ++j) {
                if (p.u.isEdge(nodeOrder[j], nodeOrder[i])) {
                    connected = true;
                    ++numNeighbor[i];
                    if (!p.useDAG()) {
                        if (std::find(lessRules[nodeOrder[j]].begin(), lessRules[nodeOrder[j]].end(),
                                      nodeOrder[i]) != lessRules[nodeOrder[j]].end()) {
                            nodeOutPos[i].push_back(j);
                        }
                        else if (std::find(greaterRules[nodeOrder[j]].begin(), greaterRules[nodeOrder[j]].end(),
                                           nodeOrder[i]) != greaterRules[nodeOrder[j]].end()) {
                            nodeInPos[i].push_back(j);
                        }
                        else nodeUnPos[i].push_back(j);
                    }
                    else if (p.in.isEdge(nodeOrder[j], nodeOrder[i])) {
                        nodeInPos[i].push_back(j);
                    }
                    else  {
                        nodeOutPos[i].push_back(j);
                    }
                }
            }
            if (!connected && i != 0) break;
            // compute ep
            if (!(key1 == 0 && key2 == 0)) {
                if (nodeOrder[i] == key1 || nodeOrder[i] == key2)
                    ep = i;
            }
            else if (!(childKey1 == 0 && childKey2 == 0)) {
                if (nodeOrder[i] == childKey1 || nodeOrder[i] == childKey2)
                    ep = i;
            }
            // if there is no intersection at ep, then in fact we do not need to compute edge key
            if (numNeighbor[ep] < 2) ep = 0;
            // compute numIn and gNumIntersect
            if (useTriangle) {
                // collect triangle edges and choose the one with the smallest position
                std::vector<int> poses = nodeInPos[i];
                for (int pos: nodeOutPos[i]) poses.push_back(pos);
                for (int pos: nodeUnPos[i]) poses.push_back(pos);
                if (poses.size() >= 2) {
                    std::sort(poses.begin(), poses.end());
                    for (int j = 1; j < poses.size(); ++j) {
                        int p2 = poses[j];
                        bool flag = false;
                        for (int k = 0; k < j; ++k) {
                            int p1 = poses[k];
                            if (p.u.isEdge(nodeOrder[p1], nodeOrder[p2]))
                                flag = true;
                            nodeTriPos[i]= std::make_pair(p1, p2);
                            if (flag) break;
                        }
                        if (flag) break;
                    }
                }
                std::vector<int> inCopy = nodeInPos[i];
                std::vector<int> outCopy = nodeOutPos[i];
                std::vector<int> unCopy = nodeUnPos[i];
                nodeInPos[i].clear();
                for (int pos: inCopy) {
                    if (pos != nodeTriPos[i].first && pos != nodeTriPos[i].second)
                        nodeInPos[i].push_back(pos);
                }
                nodeOutPos[i].clear();
                for (int pos: outCopy) {
                    if (pos != nodeTriPos[i].first && pos != nodeTriPos[i].second)
                        nodeOutPos[i].push_back(pos);
                }
                nodeUnPos[i].clear();
                for (int pos: unCopy) {
                    if (pos != nodeTriPos[i].first && pos != nodeTriPos[i].second)
                        nodeUnPos[i].push_back(pos);
                }
            }
            numInNeighbor += nodeInPos[i].size() + nodeUnPos[i].size();
            if (i == nodeOrder.size() - 1) {
                for (int j = 0; j < localVertices.size(); ++j)
                    degSeq[j] = degreeInNode[localVertices[j]];
                if (firstSequenceLarger(maxDegSeq, degSeq)) continue;
                if (firstSequenceLarger(degSeq, maxDegSeq) ||
                    ep < minEdgePos || ep == minEdgePos && numInNeighbor < minIn) {
                    maxDegSeq = degSeq;
                    minIn = numInNeighbor;
                    minEdgePos = ep;
                    numIn = numInNeighbor;
                    edgePos = ep;
                    localOrder = localVertices;
                }
            }
        }
    } while (std::next_permutation(localVertices.begin(), localVertices.end()));
}

void bestLocalOrder(const Tree &t, VertexID nID, const Pattern &p, std::vector<VertexID> &localOrder,
                    ui &numIn, ui &edgePos, bool useTriangle) {
    const Node &tau = t.getNode(nID);
    std::vector<VertexID> localVertices;
    ui numVertices = tau.numVertices;
    VertexID *vertices = tau.vertices;
    ui prefixSize = tau.prefixSize;
    VertexID *prefix = tau.prefix;
    const std::vector<VertexID> child = t.getChild()[nID];
    std::vector<VertexID *> childPrefixes(child.size());
    std::vector<ui> childPrefixSize(child.size());
    std::vector<ui> childLocalSize(child.size());
    for (int i = 0; i < child.size(); ++i) {
        VertexID cID = child[i];
        childPrefixes[i] = t.getNode(cID).prefix;
        childPrefixSize[i] = t.getNode(cID).prefixSize;
        childLocalSize[i] = t.getNode(cID).numVertices - t.getNode(cID).prefixSize;
    }

    ui minIn = 99, minEdgePos = 99;
    const std::vector<std::vector<VertexID>> &greaterRules = t.getGreaterRules();
    const std::vector<std::vector<VertexID>> &lessRules = t.getLessRules();
    ui childKey1 = 0, childKey2 = 0, key1 = 0, key2 = 0;
    ui keySize;
    if (nID == t.getRootID() && t.getOrbitType() == 2 && t.getAggreV().size() == 2) {
        key1 = t.getAggreV()[0];
        key2 = t.getAggreV()[1];
    }
    else {
        ui *key = t.getKey(nID, keySize);
        if (keySize == 2) {
            key1 = key[0];
            key2 = key[1];
        }
    }
    if (key1 == 0 && key2 == 0) {
        for (VertexID cID: child) {
            ui *key = t.getKey(cID, keySize);
            if (keySize == 2) {
                childKey1 = key[0];
                childKey2 = key[1];
                break;
            }
        }
    }
    for (int i = 0; i < numVertices; ++i) {
        if (std::find(prefix, prefix + prefixSize, vertices[i]) == prefix + prefixSize)
            localVertices.push_back(vertices[i]);
    }
    if (childKey1 == 0 && childKey2 == 0 && key1 == 0 && key2 == 0) edgePos = 0;
    std::vector<ui> degreeInNode(p.u.getNumVertices(), 0);
    for (int i = 0; i < localVertices.size(); ++i) {
        VertexID u1 = localVertices[i];
        for (int j = 0; j < tau.numVertices; ++j) {
            if (p.u.isEdge(u1, tau.vertices[j]))
                ++degreeInNode[u1];
        }
    }
    std::vector<ui> maxDegSeq(localVertices.size(), 0);
    ui minChildSize = 99;
    do {
        std::vector<VertexID> nodeOrder(prefix, prefix + prefixSize);
        ui numInNeighbor = 0, ep = 0, childSize = 0;
        std::vector<ui> degSeq(localVertices.size(), 0);
        for (VertexID u: localVertices) nodeOrder.push_back(u);
        std::vector<std::vector<int>> nodeInPos(nodeOrder.size());
        std::vector<std::vector<int>> nodeOutPos(nodeOrder.size());
        std::vector<std::vector<int>> nodeUnPos(nodeOrder.size());
        std::vector<int> numNeighbor(nodeOrder.size(), 0);
        std::vector<std::pair<int, int>> nodeTriPos(nodeOrder.size());
        for (int i = 0; i < nodeTriPos.size(); ++i) {
            nodeTriPos[i].first = -1;
            nodeTriPos[i].second = -1;
        }
        for (int i = 0; i < nodeOrder.size(); ++i) {
            bool connected = false;
            for (int j = 0; j <= i; ++j) {
                if (p.u.isEdge(nodeOrder[j], nodeOrder[i])) {
                    connected = true;
                    ++numNeighbor[i];
                    if (!p.useDAG()) {
                        if (std::find(lessRules[nodeOrder[j]].begin(), lessRules[nodeOrder[j]].end(),
                                      nodeOrder[i]) != lessRules[nodeOrder[j]].end()) {
                            nodeOutPos[i].push_back(j);
                        }
                        else if (std::find(greaterRules[nodeOrder[j]].begin(), greaterRules[nodeOrder[j]].end(),
                                           nodeOrder[i]) != greaterRules[nodeOrder[j]].end()) {
                            nodeInPos[i].push_back(j);
                        }
                        else nodeUnPos[i].push_back(j);
                    }
                    else if (p.in.isEdge(nodeOrder[j], nodeOrder[i])) {
                        nodeInPos[i].push_back(j);
                    }
                    else  {
                        nodeOutPos[i].push_back(j);
                    }
                }
            }
            if (i == 0) connected = true;
            if (!connected && i >= prefixSize) break;
            // compute ep
            if (!(key1 == 0 && key2 == 0)) {
                if (nodeOrder[i] == key1 || nodeOrder[i] == key2)
                    ep = i;
            }
            else if (!(childKey1 == 0 && childKey2 == 0)) {
                if (nodeOrder[i] == childKey1 || nodeOrder[i] == childKey2)
                    ep = i;
            }
            // if there is no intersection at ep, then in fact we do not need to compute edge key
            if (numNeighbor[ep] < 2) ep = 0;
            // compute numIn and gNumIntersect
            if (useTriangle) {
                // collect triangle edges and choose the one with the smallest position
                std::vector<int> poses = nodeInPos[i];
                for (int pos: nodeOutPos[i]) poses.push_back(pos);
                for (int pos: nodeUnPos[i]) poses.push_back(pos);
                if (poses.size() >= 2) {
                    std::sort(poses.begin(), poses.end());
                    for (int j = 1; j < poses.size(); ++j) {
                        int p2 = poses[j];
                        bool flag = false;
                        for (int k = 0; k < j; ++k) {
                            int p1 = poses[k];
                            if (p.u.isEdge(nodeOrder[p1], nodeOrder[p2]))
                                flag = true;
                            nodeTriPos[i]= std::make_pair(p1, p2);
                            if (flag) break;
                        }
                        if (flag) break;
                    }
                }
                std::vector<int> inCopy = nodeInPos[i];
                std::vector<int> outCopy = nodeOutPos[i];
                std::vector<int> unCopy = nodeUnPos[i];
                nodeInPos[i].clear();
                for (int pos: inCopy) {
                    if (pos != nodeTriPos[i].first && pos != nodeTriPos[i].second)
                        nodeInPos[i].push_back(pos);
                }
                nodeOutPos[i].clear();
                for (int pos: outCopy) {
                    if (pos != nodeTriPos[i].first && pos != nodeTriPos[i].second)
                        nodeOutPos[i].push_back(pos);
                }
                nodeUnPos[i].clear();
                for (int pos: unCopy) {
                    if (pos != nodeTriPos[i].first && pos != nodeTriPos[i].second)
                        nodeUnPos[i].push_back(pos);
                }
            }
            numInNeighbor += nodeInPos[i].size() + nodeUnPos[i].size();
            if (i == nodeOrder.size() - 1) {
                for (int j = 0; j < child.size(); ++j) {
                    ui cSize = childLocalSize[j];
                    ui prefixMatchedPos = 0;
                    for (int k = 0; k < childPrefixSize[j]; ++k) {
                        for (int l = 0; l < nodeOrder.size(); ++l) {
                            if (childPrefixes[j][k] == nodeOrder[l]) {
                                if (l + 1 > prefixMatchedPos) prefixMatchedPos = l + 1;
                                break;
                            }
                        }
                    }
                    if (cSize + prefixMatchedPos > childSize) childSize = cSize + prefixMatchedPos;
                }
                if (childSize > minChildSize) continue;
                for (int j = 0; j < localVertices.size(); ++j) degSeq[j] = degreeInNode[localVertices[j]];
                if (childSize == minChildSize && firstSequenceLarger(maxDegSeq, degSeq)) continue;
                if (childSize < minChildSize || firstSequenceLarger(degSeq, maxDegSeq) ||
                    ep < minEdgePos || ep == minEdgePos && numInNeighbor < minIn) {
                    minChildSize = childSize;
                    maxDegSeq = degSeq;
                    minIn = numInNeighbor;
                    minEdgePos = ep;
                    numIn = numInNeighbor;
                    edgePos = ep;
                    localOrder = localVertices;
                }
            }
        }
    } while (std::next_permutation(localVertices.begin(), localVertices.end()));
}

void conNodeOrder(const ConNode &cn, const Node &tau, const std::vector<VertexID> &prefixOrder, const PatternGraph &p,
                  std::vector<VertexID> &localOrder, ui &numIn, ui &edgePos, bool useTriangle,
                  const std::vector<std::vector<VertexID>> &greaterRules,
                  const std::vector<std::vector<VertexID>> &lessRules) {
    std::vector<VertexID> localVertices;
    ui numVertices = tau.numVertices;
    VertexID *vertices = tau.vertices;
    for (int i = 0; i < numVertices; ++i) {
        if (std::find(prefixOrder.begin(), prefixOrder.end(), vertices[i]) == (prefixOrder.end()))
            localVertices.push_back(vertices[i]);
    }
    ui minIn = 99, minEdgePos = 99;
    std::vector<ui> maxDegSeq = std::vector<ui>(numVertices, 0);
    ui key1 = 0, key2 = 0, aggreSize = 0;
    if (cn.cutVertices.size() - prefixOrder.size() == 2) {
        for (VertexID u: cn.cutVertices)
            if (std::find(prefixOrder.begin(), prefixOrder.end(), u) == prefixOrder.end()) {
                key1 = u;
                break;
            }
        for (VertexID u: cn.cutVertices) {
            if (u != key1 && std::find(prefixOrder.begin(), prefixOrder.end(), u) == prefixOrder.end()) {
                key2 = u;
                break;
            }
        }
    }
    else if (p.getOrbitType() == 2) {
        key1 = 0;
        key2 = 1;
    }
    if (key1 == 0 && key2 == 0) edgePos = 0;
    else if (localVertices.size() < 2) {
        edgePos = 99;
        numIn = 99;
        localOrder = localVertices;
        return;
    }
    do {
        ui numInNeighbor = 0, ep = 0;
        std::vector<ui> outDegSeq = std::vector<ui>(numVertices, 0);;
        std::vector<VertexID> nodeOrder = prefixOrder;
        for (VertexID u: localVertices) nodeOrder.push_back(u);
        std::vector<std::vector<int>> nodeInPos(nodeOrder.size());
        std::vector<std::vector<int>> nodeOutPos(nodeOrder.size());
        std::vector<std::vector<int>> nodeUnPos(nodeOrder.size());
        std::vector<int> numNeighbor(nodeOrder.size(), 0);
        std::vector<std::pair<int, int>> nodeTriPos(nodeOrder.size());
        for (int i = 0; i < nodeTriPos.size(); ++i) {
            nodeTriPos[i].first = -1;
            nodeTriPos[i].second = -1;
        }
        for (int i = 0; i < nodeOrder.size(); ++i) {
            bool connected = false;
            for (int j = 0; j <= i; ++j) {
                if (p.isEdge(nodeOrder[j], nodeOrder[i])) {
                    connected = true;
                    ++numNeighbor[i];
                    if (std::find(lessRules[nodeOrder[j]].begin(), lessRules[nodeOrder[j]].end(),
                                  nodeOrder[i]) != lessRules[nodeOrder[j]].end()) {
                        nodeOutPos[i].push_back(j);
                    }
                    else if (std::find(greaterRules[nodeOrder[j]].begin(), greaterRules[nodeOrder[j]].end(),
                                       nodeOrder[i]) != greaterRules[nodeOrder[j]].end()) {
                        nodeInPos[i].push_back(j);
                    }
                    else nodeUnPos[i].push_back(j);
                }
            }
            // compute ep
            if (!(key1 == 0 && key2 == 0)) {
                if (nodeOrder[i] == key1 || nodeOrder[i] == key2)
                    ep = i;
            }
            if (!connected && i != 0) break;
            // if there is no intersection at ep, then in fact we do not need to compute edge key
            if (numNeighbor[ep] < 2) ep = 0;
            outDegSeq[i] = nodeOutPos[i].size();
            // compute numIn and gNumIntersect
            if (useTriangle) {
                // collect triangle edges and choose the one with the smallest position
                std::vector<int> poses = nodeInPos[i];
                for (int pos: nodeOutPos[i]) poses.push_back(pos);
                for (int pos: nodeUnPos[i]) poses.push_back(pos);
                if (poses.size() >= 2) {
                    std::sort(poses.begin(), poses.end());
                    for (int j = 1; j < poses.size(); ++j) {
                        int p2 = poses[j];
                        bool flag = false;
                        for (int k = 0; k < j; ++k) {
                            int p1 = poses[k];
                            if (p.isEdge(nodeOrder[p1], nodeOrder[p2]))
                                flag = true;
                            nodeTriPos[i]= std::make_pair(p1, p2);
                            if (flag) break;
                        }
                        if (flag) break;
                    }
                }
                std::vector<int> inCopy = nodeInPos[i];
                std::vector<int> outCopy = nodeOutPos[i];
                std::vector<int> unCopy = nodeUnPos[i];
                nodeInPos[i].clear();
                for (int pos: inCopy) {
                    if (pos != nodeTriPos[i].first && pos != nodeTriPos[i].second)
                        nodeInPos[i].push_back(pos);
                }
                nodeOutPos[i].clear();
                for (int pos: outCopy) {
                    if (pos != nodeTriPos[i].first && pos != nodeTriPos[i].second)
                        nodeOutPos[i].push_back(pos);
                }
                nodeUnPos[i].clear();
                for (int pos: unCopy) {
                    if (pos != nodeTriPos[i].first && pos != nodeTriPos[i].second)
                        nodeUnPos[i].push_back(pos);
                }
            }
            numInNeighbor += nodeInPos[i].size() + nodeUnPos[i].size();

            if (i == nodeOrder.size() - 1) {
                if (ep < minEdgePos || ep == minEdgePos && numInNeighbor < minIn ||
                    ep == minEdgePos && numInNeighbor == minIn && firstSequenceLarger(outDegSeq, maxDegSeq)) {
                    minIn = numInNeighbor;
                    minEdgePos = ep;
                    outDegSeq = maxDegSeq;
                    numIn = numInNeighbor;
                    edgePos = ep;
                    localOrder = localVertices;
                }
            }
        }
    } while (std::next_permutation(localVertices.begin(), localVertices.end()));
}
