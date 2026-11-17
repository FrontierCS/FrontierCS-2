//
// Created by anonymous author on 2022/8/29.
//

#include "execution.h"

Count gNumIntersect = 0;
Count gNumMatch = 0;
Count gNumIntermediate = 0;
Count gNumEdgeID = 0;
Count gNumUpdate = 0;

void saveCount(const std::string &resultPath, std::vector<HashTable> H, const DataGraph &d, bool batchQuery,
               const std::vector<std::string> &files, const std::vector<int> &orbitTypes) {
    if (batchQuery) {
        for (int i = 0; i < files.size(); ++i)
            saveCount(resultPath + "/" + files[i], H[i], d, orbitTypes[i]);
    }
    else saveCount(resultPath, H[0], d, orbitTypes[0]);
}

void saveCount(const std::string &resultPath, HashTable h, const DataGraph &d, int orbitType) {
    std::ofstream out(resultPath);
    if (orbitType == 0) out << h[0] << std::endl;
    else if (orbitType == 1) {
        for (VertexID i = 0; i < d.getNumVertices(); ++i)
            out << h[i] << std::endl;
    } else {
        for (VertexID u = 0; u < d.getNumVertices(); ++u) {
            ui nbrCnt = 0;
            VertexID *outNeighbors = d.getNeighborsLargerID(u, nbrCnt);
            for (EdgeID j = 0; j < nbrCnt; ++j) {
                VertexID v = outNeighbors[j];
                out << u << "\t\t\t" << v << "\t\t\t" << h[d.getEdgeID(u, v)];
            }
        }
    }
//    std::cout << "save result to " << resultPath << std::endl;
}

// give the current mapping, the (directed) pattern graph, the (directed) data graph, the next pattern vertex(pos),
// return the data vertices that can map to the pattern vertex
// inPos: positions of vertices in dataV that are in-neighbor of dataV[pos]
// note that we should avoid swapping neighbor with tmp
void generateCandidate(
        const DataGraph &din,
        const DataGraph &dout,
        const DataGraph &dun,
        VertexID *dataV,
        int pos,
        const std::vector<int> &inPos,
        const std::vector<int> &outPos,
        const std::vector<int> &unPos,
        VertexID **candidate,
        ui *&candCount,
        VertexID *&tmp
) {
    int num = inPos.size() + outPos.size() + unPos.size();
    ui tmpCount;
    if (num > 2) {
        VertexID **arrays = new VertexID *[num];
        ui *counts = new ui[num];
        int p = 0;
        for (int position: inPos) {
            arrays[p] = din.getNeighbors(dataV[position], counts[p]);
            ++p;
        }
        for (int position: outPos) {
            arrays[p] = dout.getNeighbors(dataV[position], counts[p]);
            ++p;
        }
        for (int position: unPos) {
            arrays[p] = dun.getNeighbors(dataV[position], counts[p]);
            ++p;
        }
        ComputeSetIntersection::LeapfrogJoin(arrays, counts, num, tmp, tmpCount);
        candCount[pos] = tmpCount;
        std::swap(candidate[pos], tmp);
        delete[] counts;
        delete[] arrays;
        return;
    }
    VertexID *firstNeighbor;
    ui firstCnt;
    bool firstFlag = true, secondFlag = true;
    for (int position: inPos) {
        ui nbrCnt;
        VertexID *neighbor = din.getNeighbors(dataV[position], nbrCnt);
        if (firstFlag) {
            firstFlag = false;
            firstCnt = nbrCnt;
            firstNeighbor = neighbor;
        }
        else {
            if (secondFlag) {
                secondFlag = false;
#ifdef COLLECT_STATISTICS
                ++gNumIntersect;
#endif
                ComputeSetIntersection::ComputeCandidates(firstNeighbor, firstCnt, neighbor, nbrCnt, tmp, tmpCount);
                candCount[pos] = tmpCount;
                std::swap(candidate[pos], tmp);
            }
            else {
#ifdef COLLECT_STATISTICS
                ++gNumIntersect;
#endif
                ComputeSetIntersection::ComputeCandidates(candidate[pos], candCount[pos], neighbor, nbrCnt, tmp, tmpCount);
                candCount[pos] = tmpCount;
                std::swap(candidate[pos], tmp);
            }
        }
    }
    for (int position: outPos) {
        ui nbrCnt;
        VertexID *neighbor = dout.getNeighbors(dataV[position], nbrCnt);
        if (firstFlag) {
            firstFlag = false;
            firstCnt = nbrCnt;
            firstNeighbor = neighbor;
        }
        else {
            if (secondFlag) {
                secondFlag = false;
#ifdef COLLECT_STATISTICS
                ++gNumIntersect;
#endif
                ComputeSetIntersection::ComputeCandidates(firstNeighbor, firstCnt, neighbor, nbrCnt, tmp, tmpCount);
                candCount[pos] = tmpCount;
                std::swap(candidate[pos], tmp);
            }
            else {
#ifdef COLLECT_STATISTICS
                ++gNumIntersect;
#endif
                ComputeSetIntersection::ComputeCandidates(candidate[pos], candCount[pos], neighbor, nbrCnt, tmp, tmpCount);
                candCount[pos] = tmpCount;
                std::swap(candidate[pos], tmp);
            }
        }
    }
    for (int position: unPos) {
        ui nbrCnt;
        VertexID *neighbor = dun.getNeighbors(dataV[position], nbrCnt);
        if (firstFlag) {
            firstFlag = false;
            firstCnt = nbrCnt;
            firstNeighbor = neighbor;
        }
        else {
            if (secondFlag) {
                secondFlag = false;
#ifdef COLLECT_STATISTICS
                ++gNumIntersect;
#endif
                ComputeSetIntersection::ComputeCandidates(firstNeighbor, firstCnt, neighbor, nbrCnt, tmp, tmpCount);
                candCount[pos] = tmpCount;
                std::swap(candidate[pos], tmp);
            }
            else {
#ifdef COLLECT_STATISTICS
                ++gNumIntersect;
#endif
                ComputeSetIntersection::ComputeCandidates(candidate[pos], candCount[pos], neighbor, nbrCnt, tmp, tmpCount);
                candCount[pos] = tmpCount;
                std::swap(candidate[pos], tmp);
            }
        }
    }
}

// generate candidate with the help of triangles stored
void generateCandidateT(
        const DataGraph &din,
        const DataGraph &dout,
        const DataGraph &dun,
        const Triangle &tri,
        VertexID *dataV,
        int pos,
        bool isTriangle,
        EdgeID e,
        int type,
        const std::vector<int> &inPos,
        const std::vector<int> &outPos,
        const std::vector<int> &unPos,
        VertexID **candidate,
        ui *&candCount,
        VertexID *&tmp
) {
    if (isTriangle) {
        candidate[pos] = tri.getTriend(e, type, candCount[pos]);
        return;
    }
    int num = inPos.size() + outPos.size() + unPos.size() + (type != 0);
    ui tmpCount;
    if (num > 2) {
        VertexID **arrays = new VertexID *[num];
        ui *counts = new ui[num];
        int p = 0;
        if (type != 0) {
            arrays[p] = tri.getTriend(e, type, counts[p]);
            ++p;
        }
        for (int position: inPos) {
            arrays[p] = din.getNeighbors(dataV[position], counts[p]);
            ++p;
        }
        for (int position: outPos) {
            arrays[p] = dout.getNeighbors(dataV[position], counts[p]);
            ++p;
        }
        for (int position: unPos) {
            arrays[p] = dun.getNeighbors(dataV[position], counts[p]);
            ++p;
        }
        ComputeSetIntersection::LeapfrogJoin(arrays, counts, num, tmp, tmpCount);
        candCount[pos] = tmpCount;
        std::swap(candidate[pos], tmp);
        delete[] counts;
        delete[] arrays;
        return;
    }
    VertexID *firstList;
    ui firstCnt;
    bool firstFlag = true, secondFlag = true;
    if (type != 0) {
        firstList = tri.getTriend(e, type, firstCnt);
        firstFlag = false;
    }
    for (int position: inPos) {
        ui nbrCnt;
        VertexID *neighbor = din.getNeighbors(dataV[position], nbrCnt);
        if (firstFlag) {
            firstFlag = false;
            firstCnt = nbrCnt;
            firstList = neighbor;
        }
        else {
            if (secondFlag) {
                secondFlag = false;
#ifdef COLLECT_STATISTICS
                ++gNumIntersect;
#endif
                ComputeSetIntersection::ComputeCandidates(firstList, firstCnt, neighbor, nbrCnt, tmp, tmpCount);
                candCount[pos] = tmpCount;
                std::swap(candidate[pos], tmp);
            }
            else {
#ifdef COLLECT_STATISTICS
                ++gNumIntersect;
#endif
                ComputeSetIntersection::ComputeCandidates(candidate[pos], candCount[pos], neighbor, nbrCnt, tmp, tmpCount);
                candCount[pos] = tmpCount;
                std::swap(candidate[pos], tmp);
            }
        }
    }
    for (int position: outPos) {
        ui nbrCnt;
        VertexID *neighbor = dout.getNeighbors(dataV[position], nbrCnt);
        if (firstFlag) {
            firstFlag = false;
            firstCnt = nbrCnt;
            firstList = neighbor;
        }
        else {
            if (secondFlag) {
                secondFlag = false;
#ifdef COLLECT_STATISTICS
                ++gNumIntersect;
#endif
                ComputeSetIntersection::ComputeCandidates(firstList, firstCnt, neighbor, nbrCnt, tmp, tmpCount);
                candCount[pos] = tmpCount;
                std::swap(candidate[pos], tmp);
            }
            else {
#ifdef COLLECT_STATISTICS
                ++gNumIntersect;
#endif
                ComputeSetIntersection::ComputeCandidates(candidate[pos], candCount[pos], neighbor, nbrCnt, tmp, tmpCount);
                candCount[pos] = tmpCount;
                std::swap(candidate[pos], tmp);
            }
        }
    }
    for (int position: unPos) {
        ui nbrCnt;
        VertexID *neighbor = dun.getNeighbors(dataV[position], nbrCnt);
        if (firstFlag) {
            firstFlag = false;
            firstCnt = nbrCnt;
            firstList = neighbor;
        }
        else {
            ui tmpCount;
            if (secondFlag) {
                secondFlag = false;
#ifdef COLLECT_STATISTICS
                ++gNumIntersect;
#endif
                ComputeSetIntersection::ComputeCandidates(firstList, firstCnt, neighbor, nbrCnt, tmp, tmpCount);
                candCount[pos] = tmpCount;
                std::swap(candidate[pos], tmp);
            }
            else {
#ifdef COLLECT_STATISTICS
                ++gNumIntersect;
#endif
                ComputeSetIntersection::ComputeCandidates(candidate[pos], candCount[pos], neighbor, nbrCnt, tmp, tmpCount);
                candCount[pos] = tmpCount;
                std::swap(candidate[pos], tmp);
            }
        }
    }
}

ui computeEdgeKey(
        const DataGraph &din,
        const DataGraph &dout,
        const DataGraph &dun,
        EdgeID *outID,
        EdgeID *unID,
        EdgeID *reverseID,
        EdgeID *startOffset,
        int mappingSize,
        ui *pos,
        int edgeType,
        VertexID src,
        VertexID dst
) {
    ui edgeID = 0;
    switch (edgeType) {
        case 1:
            edgeID = startOffset[mappingSize] + pos[mappingSize] - 1;
            break;
        case 2:
            edgeID = outID[startOffset[mappingSize] + pos[mappingSize] - 1];
            break;
        case 3:
#ifdef COLLECT_STATISTICS
            ++gNumEdgeID;
#endif
            edgeID = dout.getEdgeID(src, dst);
            break;
        case 4:
#ifdef COLLECT_STATISTICS
            ++gNumEdgeID;
#endif
            edgeID = dun.getUndirectedEID(src, dst);
            break;
        case 5:
            edgeID = unID[startOffset[mappingSize] + pos[mappingSize] - 1];
            break;
        case 6:
            edgeID = reverseID[startOffset[mappingSize] + pos[mappingSize] - 1];
            break;
    }
    return edgeID;
}

void executeNode(
        VertexID nID,
        const Tree &t,
        const std::vector<VertexID> &child,
        VertexID **candidate,
        ui *candCount,
        HashTable *H,
        const DataGraph &din,
        const DataGraph &dout,
        const DataGraph &dun,
        const Pattern &p,
        bool isRoot,
        EdgeID *outID,
        EdgeID *unID,
        EdgeID *reverseID,
        EdgeID *startOffset,
        VertexID *patternV,
        VertexID *dataV,
        int mappingSize,
        bool *visited,
        ui *pos,
        ui *keyPos,
        ui &keyPosSize,
        ui sizeBound,
        VertexID *&tmp,
        VertexID *allV
) {
    int orbitType = t.getOrbitType();
    HashTable h = H[nID];
    const Node &tau = t.getNode(nID);
    const std::vector<int> &aggrePos = t.getAggrePos(nID);
    const std::vector<bool> &nodeInterPos = t.getNodeInterPos(nID);
    const std::vector<std::vector<int>> &nodeInPos = t.getNodeInPos(nID);
    const std::vector<std::vector<int>> &nodeOutPos = t.getNodeOutPos(nID);
    const std::vector<std::vector<int>> &nodeUnPos = t.getNodeUnPos(nID);
    const std::vector<std::vector<int>> &greaterPos = t.getNodeGreaterPos(nID);
    const std::vector<std::vector<int>> &lessPos = t.getNodeLessPos(nID);
    const std::vector<std::vector<int>> &childKeyPos = t.getChildKeyPos(nID);
    const std::vector<VertexID> &aggreV = t.getAggreV();
    const std::vector<int> &aggreWeight = t.getAggreWeight();
    EdgeID *inOffset = din.getOffsets();
    VertexID *inNbors = din.getNbors();
    EdgeID *outOffset = dout.getOffsets();
    VertexID *outNbors = dout.getNbors();
    EdgeID *unOffset = dun.getOffsets();
    VertexID *unNbors = dun.getNbors();

    ui n = din.getNumVertices();
    if (mappingSize == 0) {
        candidate[0] = allV;
        candCount[0] = n;
        pos[mappingSize] = 0;
    }
    else {
        if (!nodeInterPos[mappingSize]) {
            if (!nodeOutPos[mappingSize].empty()) {
                VertexID v = dataV[nodeOutPos[mappingSize][0]];
                startOffset[mappingSize] = outOffset[v];
                candidate[mappingSize] = outNbors + outOffset[v];
                candCount[mappingSize] = outOffset[v + 1] - outOffset[v];
            }
            else if (!nodeInPos[mappingSize].empty()){
                VertexID v = dataV[nodeInPos[mappingSize][0]];
                startOffset[mappingSize] = inOffset[v];
                candidate[mappingSize] = inNbors + inOffset[v];
                candCount[mappingSize] = inOffset[v + 1] - inOffset[v];
            }
            else {
                VertexID v = dataV[nodeUnPos[mappingSize][0]];
                startOffset[mappingSize] = unOffset[v];
                candidate[mappingSize] = unNbors + unOffset[v];
                candCount[mappingSize] = unOffset[v + 1] - unOffset[v];
            }
        }
        else {
            generateCandidate(din, dout, dun, dataV, mappingSize, nodeInPos[mappingSize],
                              nodeOutPos[mappingSize], nodeUnPos[mappingSize], candidate, candCount, tmp);
        }
        pos[mappingSize] = 0;
        // apply the symmetry breaking rules
        if (!greaterPos[mappingSize].empty()) {
            ui maxTarget = dataV[greaterPos[mappingSize][0]];
            for (int i = 1; i < greaterPos[mappingSize].size(); ++i) {
                if (dataV[greaterPos[mappingSize][i]] > maxTarget)
                    maxTarget = dataV[greaterPos[mappingSize][i]];
            }
            pos[mappingSize] = firstPosGreaterThan(candidate[mappingSize], 0, candCount[mappingSize], maxTarget);
        }
        if (!lessPos[mappingSize].empty()) {
            ui minTarget = dataV[lessPos[mappingSize][0]];
            for (int i = 1; i < lessPos[mappingSize].size(); ++i) {
                if (dataV[lessPos[mappingSize][i]] < minTarget)
                    minTarget = dataV[lessPos[mappingSize][i]];
            }
            candCount[mappingSize] = firstPosGreaterThan(candidate[mappingSize], 0, candCount[mappingSize], minTarget);
        }
    }

    int depth = 0;
    while (depth >= 0) {
        while (pos[mappingSize] < candCount[mappingSize]) {
            VertexID v = candidate[mappingSize][pos[mappingSize]];
            ++pos[mappingSize];
            if (visited[v]) continue;
            visited[v] = true;
            patternV[mappingSize] = tau.nodeOrder[mappingSize];
            dataV[mappingSize] = v;

            if (depth == tau.localOrder.size() - 1) {
#ifdef COLLECT_STATISTICS
                ++gNumMatch;
#endif
                Count cnt = 1;
                // multiply count in children
                for (int j = 0; j < child.size(); ++j) {
                    VertexID cID = child[j];
                    cnt *= H[cID][dataV[childKeyPos[j][0]]];
                }
                if (isRoot) {
                    if (orbitType == 0) h[0] += cnt * aggreWeight[0];
                    else if (orbitType == 1) {
                        for (int j = 0; j < aggreV.size(); ++j) {
#ifdef COLLECT_STATISTICS
                            ++gNumUpdate;
#endif
                            VertexID key = dataV[aggrePos[j]];
                            h[key] += cnt * aggreWeight[j];
                        }
                    }
                }
                else {
#ifdef COLLECT_STATISTICS
                    ++gNumUpdate;
#endif
                    h[dataV[aggrePos[0]]] += cnt;
                    if (keyPosSize < sizeBound) {
                        keyPos[keyPosSize] = dataV[aggrePos[0]];
                        ++keyPosSize;
                    }
                }
                visited[dataV[mappingSize]] = false;
            }
            else {
#ifdef COLLECT_STATISTICS
                ++gNumIntermediate;
#endif
                ++mappingSize;
                ++depth;
                pos[mappingSize] = 0;
                if (!nodeInterPos[mappingSize]) {
                    if (!nodeOutPos[mappingSize].empty()) {
                        VertexID w = dataV[nodeOutPos[mappingSize][0]];
                        startOffset[mappingSize] = outOffset[w];
                        candidate[mappingSize] = outNbors + outOffset[w];
                        candCount[mappingSize] = outOffset[w + 1] - outOffset[w];
                    }
                    else if (!nodeInPos[mappingSize].empty()){
                        VertexID w = dataV[nodeInPos[mappingSize][0]];
                        startOffset[mappingSize] = inOffset[w];
                        candidate[mappingSize] = inNbors + inOffset[w];
                        candCount[mappingSize] = inOffset[w + 1] - inOffset[w];
                    }
                    else {
                        VertexID w = dataV[nodeUnPos[mappingSize][0]];
                        startOffset[mappingSize] = unOffset[w];
                        candidate[mappingSize] = unNbors + unOffset[w];
                        candCount[mappingSize] = unOffset[w + 1] - unOffset[w];
                    }
                }
                else {
                    generateCandidate(din, dout, dun, dataV, mappingSize, nodeInPos[mappingSize],
                                      nodeOutPos[mappingSize], nodeUnPos[mappingSize], candidate, candCount, tmp);
                }
                // apply the symmetry breaking rules
                if (!greaterPos[mappingSize].empty()) {
                    ui maxTarget = dataV[greaterPos[mappingSize][0]];
                    for (int i = 1; i < greaterPos[mappingSize].size(); ++i) {
                        if (dataV[greaterPos[mappingSize][i]] > maxTarget)
                            maxTarget = dataV[greaterPos[mappingSize][i]];
                    }
                    pos[mappingSize] = firstPosGreaterThan(candidate[mappingSize], 0, candCount[mappingSize], maxTarget);
                }
                if (!lessPos[mappingSize].empty()) {
                    ui minTarget = dataV[lessPos[mappingSize][0]];
                    for (int i = 1; i < lessPos[mappingSize].size(); ++i) {
                        if (dataV[lessPos[mappingSize][i]] < minTarget)
                            minTarget = dataV[lessPos[mappingSize][i]];
                    }
                    candCount[mappingSize] = firstPosGreaterThan(candidate[mappingSize], 0, candCount[mappingSize], minTarget);
                }
            }
        }
        --depth;
        if (depth >= 0) {
            --mappingSize;
            visited[dataV[mappingSize]] = false;
        }
    }
}

void executeNodeT(
        VertexID nID,
        const Tree &t,
        const std::vector<VertexID> &child,
        VertexID **candidate,
        ui *candCount,
        HashTable *H,
        const DataGraph &din,
        const DataGraph &dout,
        const DataGraph &dun,
        const Triangle &tri,
        const Pattern &p,
        bool isRoot,
        EdgeID *outID,
        EdgeID *unID,
        EdgeID *reverseID,
        EdgeID *startOffset,
        VertexID *patternV,
        VertexID *dataV,
        int mappingSize,
        bool *visited,
        ui *pos,
        ui *keyPos,
        ui &keyPosSize,
        ui sizeBound,
        VertexID *&tmp,
        VertexID *allV
) {
    int orbitType = t.getOrbitType();
    HashTable h = H[nID];
    const Node &tau = t.getNode(nID);
    const std::vector<int> &aggrePos = t.getAggrePos(nID);
    const std::vector<bool> &nodeInterPos = t.getNodeInterPos(nID);
    const std::vector<bool> &nodeCandPos = t.getNodeCandPos(nID);
    const std::vector<std::vector<int>> &nodeInPos = t.getNodeInPos(nID);
    const std::vector<std::vector<int>> &nodeOutPos = t.getNodeOutPos(nID);
    const std::vector<std::vector<int>> &nodeUnPos = t.getNodeUnPos(nID);
    const std::vector<std::vector<int>> &greaterPos = t.getNodeGreaterPos(nID);
    const std::vector<std::vector<int>> &lessPos = t.getNodeLessPos(nID);
    const std::vector<std::vector<int>> &childKeyPos = t.getChildKeyPos(nID);
    const std::vector<VertexID> &aggreV = t.getAggreV();
    const std::vector<int> &aggreWeight = t.getAggreWeight();
    const std::vector<std::pair<int, int>> &nodeTriPos = t.getNodeTriPos(nID);
    const std::vector<int> &triEdgeType = t.getTriEdgeType(nID);
    const std::vector<int> &triEndType = t.getTriEndType(nID);
    EdgeID *inOffset = din.getOffsets();
    VertexID *inNbors = din.getNbors();
    EdgeID *outOffset = dout.getOffsets();
    VertexID *outNbors = dout.getNbors();
    EdgeID *unOffset = dun.getOffsets();
    VertexID *unNbors = dun.getNbors();

    ui n = din.getNumVertices();
    if (mappingSize == 0) {
        candidate[0] = allV;
        candCount[0] = n;
        pos[mappingSize] = 0;
    }
    else {
        if (!nodeInterPos[mappingSize]) {
            if (!nodeOutPos[mappingSize].empty()) {
                VertexID v = dataV[nodeOutPos[mappingSize][0]];
                startOffset[mappingSize] = outOffset[v];
                candidate[mappingSize] = outNbors + outOffset[v];
                candCount[mappingSize] = outOffset[v + 1] - outOffset[v];
            }
            else if (!nodeInPos[mappingSize].empty()){
                VertexID v = dataV[nodeInPos[mappingSize][0]];
                startOffset[mappingSize] = inOffset[v];
                candidate[mappingSize] = inNbors + inOffset[v];
                candCount[mappingSize] = inOffset[v + 1] - inOffset[v];
            }
            else {
                VertexID v = dataV[nodeUnPos[mappingSize][0]];
                startOffset[mappingSize] = unOffset[v];
                candidate[mappingSize] = unNbors + unOffset[v];
                candCount[mappingSize] = unOffset[v + 1] - unOffset[v];
            }
        }
        else {
            EdgeID e = 0;
            if (triEdgeType[mappingSize] != 0) {
                int pos1 = nodeTriPos[mappingSize].first, pos2 = nodeTriPos[mappingSize].second;
                if (triEdgeType[mappingSize] == 1)
                    e = startOffset[pos2] + pos[pos2] - 1;
                else if (triEdgeType[mappingSize] == 2)
                    e = outID[startOffset[pos2] + pos[pos2] - 1];
                else if (triEdgeType[mappingSize] == 5)
                    e = unID[startOffset[pos2] + pos[pos2] - 1];
                else {
                    VertexID key1 = dataV[pos1], key2 = dataV[pos2];
#ifdef COLLECT_STATISTICS
                    ++gNumEdgeID;
#endif
                    e = dout.getEdgeID(key1, key2);
                }
            }
            generateCandidateT(din, dout, dun, tri, dataV, mappingSize, !nodeCandPos[mappingSize], e, triEndType[mappingSize],
                               nodeInPos[mappingSize], nodeOutPos[mappingSize], nodeUnPos[mappingSize], candidate, candCount, tmp);
        }
        pos[mappingSize] = 0;
        // apply the symmetry breaking rules
        if (!greaterPos[mappingSize].empty()) {
            ui maxTarget = dataV[greaterPos[mappingSize][0]];
            for (int i = 1; i < greaterPos[mappingSize].size(); ++i) {
                if (dataV[greaterPos[mappingSize][i]] > maxTarget)
                    maxTarget = dataV[greaterPos[mappingSize][i]];
            }
            pos[mappingSize] = firstPosGreaterThan(candidate[mappingSize], 0, candCount[mappingSize], maxTarget);
        }
        if (!lessPos[mappingSize].empty()) {
            ui minTarget = dataV[lessPos[mappingSize][0]];
            for (int i = 1; i < lessPos[mappingSize].size(); ++i) {
                if (dataV[lessPos[mappingSize][i]] < minTarget)
                    minTarget = dataV[lessPos[mappingSize][i]];
            }
            candCount[mappingSize] = firstPosGreaterThan(candidate[mappingSize], 0, candCount[mappingSize], minTarget);
        }
    }
    int depth = 0;
    while (depth >= 0) {
        while (pos[mappingSize] < candCount[mappingSize]) {
            VertexID v = candidate[mappingSize][pos[mappingSize]];
            ++pos[mappingSize];
            if (visited[v]) continue;
            visited[v] = true;
            patternV[mappingSize] = tau.nodeOrder[mappingSize];
            dataV[mappingSize] = v;
            if (depth == tau.localOrder.size() - 1) {
#ifdef COLLECT_STATISTICS
                ++gNumMatch;
#endif
                Count cnt = 1;
                // multiply count in children
                for (int j = 0; j < child.size(); ++j) {
                    VertexID cID = child[j];
                    cnt *= H[cID][dataV[childKeyPos[j][0]]];
                }
                if (isRoot) {
                    if (orbitType == 0) h[0] += cnt * aggreWeight[0];
                    else if (orbitType == 1) {
                        for (int j = 0; j < aggreV.size(); ++j) {
                            VertexID key = dataV[aggrePos[j]];
#ifdef COLLECT_STATISTICS
                            ++gNumUpdate;
#endif
                            h[key] += cnt * aggreWeight[j];
                        }
                    }
                }
                else {
#ifdef COLLECT_STATISTICS
                    ++gNumUpdate;
#endif
                    h[dataV[aggrePos[0]]] += cnt;
                    if (keyPosSize < sizeBound) {
                        keyPos[keyPosSize] = dataV[aggrePos[0]];
                        ++keyPosSize;
                    }
                }
                visited[dataV[mappingSize]] = false;
            }
            else {
#ifdef COLLECT_STATISTICS
                ++gNumIntermediate;
#endif
                ++mappingSize;
                ++depth;
                pos[mappingSize] = 0;
                if (!nodeInterPos[mappingSize]) {
                    if (!nodeOutPos[mappingSize].empty()) {
                        VertexID w = dataV[nodeOutPos[mappingSize][0]];
                        startOffset[mappingSize] = outOffset[w];
                        candidate[mappingSize] = outNbors + outOffset[w];
                        candCount[mappingSize] = outOffset[w + 1] - outOffset[w];
                    }
                    else if (!nodeInPos[mappingSize].empty()){
                        VertexID w = dataV[nodeInPos[mappingSize][0]];
                        startOffset[mappingSize] = inOffset[w];
                        candidate[mappingSize] = inNbors + inOffset[w];
                        candCount[mappingSize] = inOffset[w + 1] - inOffset[w];
                    }
                    else {
                        VertexID w = dataV[nodeUnPos[mappingSize][0]];
                        startOffset[mappingSize] = unOffset[w];
                        candidate[mappingSize] = unNbors + unOffset[w];
                        candCount[mappingSize] = unOffset[w + 1] - unOffset[w];
                    }
                }
                else {
                    EdgeID e = 0;
                    if (triEdgeType[mappingSize] != 0) {
                        int pos1 = nodeTriPos[mappingSize].first, pos2 = nodeTriPos[mappingSize].second;
                        if (triEdgeType[mappingSize] == 1)
                            e = startOffset[pos2] + pos[pos2] - 1;
                        else if (triEdgeType[mappingSize] == 2)
                            e = outID[startOffset[pos2] + pos[pos2] - 1];
                        else if (triEdgeType[mappingSize] == 5)
                            e = unID[startOffset[pos2] + pos[pos2] - 1];
                        else{
                            VertexID key1 = dataV[pos1], key2 = dataV[pos2];
#ifdef COLLECT_STATISTICS
                            ++gNumEdgeID;
#endif
                            e = dout.getEdgeID(key1, key2);
                        }
                    }
                    generateCandidateT(din, dout, dun, tri, dataV, mappingSize, !nodeCandPos[mappingSize], e, triEndType[mappingSize],
                                       nodeInPos[mappingSize], nodeOutPos[mappingSize], nodeUnPos[mappingSize], candidate, candCount, tmp);
                }
                // apply the symmetry breaking rules
                if (!greaterPos[mappingSize].empty()) {
                    ui maxTarget = dataV[greaterPos[mappingSize][0]];
                    for (int i = 1; i < greaterPos[mappingSize].size(); ++i) {
                        if (dataV[greaterPos[mappingSize][i]] > maxTarget)
                            maxTarget = dataV[greaterPos[mappingSize][i]];
                    }
                    pos[mappingSize] = firstPosGreaterThan(candidate[mappingSize], 0, candCount[mappingSize], maxTarget);
                }
                if (!lessPos[mappingSize].empty()) {
                    ui minTarget = dataV[lessPos[mappingSize][0]];
                    for (int i = 1; i < lessPos[mappingSize].size(); ++i) {
                        if (dataV[lessPos[mappingSize][i]] < minTarget)
                            minTarget = dataV[lessPos[mappingSize][i]];
                    }
                    candCount[mappingSize] = firstPosGreaterThan(candidate[mappingSize], 0, candCount[mappingSize], minTarget);
                }
            }
        }
        --depth;
        if (depth >= 0) {
            --mappingSize;
            visited[dataV[mappingSize]] = false;
        }
    }
}

void executeNodeEdgeKey(
        VertexID nID,
        const Tree &t,
        const std::vector<VertexID> &child,
        VertexID **candidate,
        ui *candCount,
        HashTable *H,
        const DataGraph &din,
        const DataGraph &dout,
        const DataGraph &dun,
        const Pattern &p,
        bool isRoot,
        EdgeID *outID,
        EdgeID *unID,
        EdgeID *reverseID,
        EdgeID *startOffset,
        VertexID *patternV,
        VertexID *dataV,
        int mappingSize,
        bool *visited,
        ui *pos,
        ui *keyPos,
        ui &keyPosSize,
        ui sizeBound,
        VertexID *&tmp,
        VertexID *allV
) {
    int orbitType = t.getOrbitType();
    HashTable h = H[nID];
    const Node &tau = t.getNode(nID);
    const std::vector<int> &aggrePos = t.getAggrePos(nID);
    const std::vector<bool> &nodeInterPos = t.getNodeInterPos(nID);
    const std::vector<std::vector<int>> &nodeInPos = t.getNodeInPos(nID);
    const std::vector<std::vector<int>> &nodeOutPos = t.getNodeOutPos(nID);
    const std::vector<std::vector<int>> &nodeUnPos = t.getNodeUnPos(nID);
    const std::vector<std::vector<int>> &greaterPos = t.getNodeGreaterPos(nID);
    const std::vector<std::vector<int>> &lessPos = t.getNodeLessPos(nID);
    const std::vector<std::vector<int>> &childKeyPos = t.getChildKeyPos(nID);
    const std::vector<VertexID> &aggreV = t.getAggreV();
    const std::vector<int> &aggreWeight = t.getAggreWeight();
    const std::vector<std::vector<int>> &posChildEdge = t.getPosChildEdge(nID);
    const std::vector<std::vector<int>> &posAggreEdge = t.getPosAggreEdge(nID);
    const std::vector<int> &childEdgeType = t.getChildEdgeType(nID);
    const std::vector<int> &aggreEdgeType = t.getAggreEdgeType(nID);
    EdgeID *inOffset = din.getOffsets();
    VertexID *inNbors = din.getNbors();
    EdgeID *outOffset = dout.getOffsets();
    VertexID *outNbors = dout.getNbors();
    EdgeID *unOffset = dun.getOffsets();
    VertexID *unNbors = dun.getNbors();

    std::vector<EdgeID> childKey(child.size());
    std::vector<EdgeID> aggreKey;
    if (isRoot && orbitType == 2) aggreKey = std::vector<EdgeID>(aggreWeight.size());
    else if (tau.keySize == 2) aggreKey.push_back(0);
    ui n = din.getNumVertices();
    if (mappingSize == 0) {
        candidate[0] = allV;
        candCount[0] = n;
        pos[mappingSize] = 0;
    }
    else {
        if (!nodeInterPos[mappingSize]) {
            if (!nodeOutPos[mappingSize].empty()) {
                VertexID v = dataV[nodeOutPos[mappingSize][0]];
                startOffset[mappingSize] = outOffset[v];
                candidate[mappingSize] = outNbors + outOffset[v];
                candCount[mappingSize] = outOffset[v + 1] - outOffset[v];
            }
            else if (!nodeInPos[mappingSize].empty()){
                VertexID v = dataV[nodeInPos[mappingSize][0]];
                startOffset[mappingSize] = inOffset[v];
                candidate[mappingSize] = inNbors + inOffset[v];
                candCount[mappingSize] = inOffset[v + 1] - inOffset[v];
            }
            else {
                VertexID v = dataV[nodeUnPos[mappingSize][0]];
                startOffset[mappingSize] = unOffset[v];
                candidate[mappingSize] = unNbors + unOffset[v];
                candCount[mappingSize] = unOffset[v + 1] - unOffset[v];
            }
        }
        else {
            generateCandidate(din, dout, dun, dataV, mappingSize, nodeInPos[mappingSize],
                              nodeOutPos[mappingSize], nodeUnPos[mappingSize], candidate, candCount, tmp);
        }
        pos[mappingSize] = 0;
        // apply the symmetry breaking rules
        if (!greaterPos[mappingSize].empty()) {
            ui maxTarget = dataV[greaterPos[mappingSize][0]];
            for (int i = 1; i < greaterPos[mappingSize].size(); ++i) {
                if (dataV[greaterPos[mappingSize][i]] > maxTarget)
                    maxTarget = dataV[greaterPos[mappingSize][i]];
            }
            pos[mappingSize] = firstPosGreaterThan(candidate[mappingSize], 0, candCount[mappingSize], maxTarget);
        }
        if (!lessPos[mappingSize].empty()) {
            ui minTarget = dataV[lessPos[mappingSize][0]];
            for (int i = 1; i < lessPos[mappingSize].size(); ++i) {
                if (dataV[lessPos[mappingSize][i]] < minTarget)
                    minTarget = dataV[lessPos[mappingSize][i]];
            }
            candCount[mappingSize] = firstPosGreaterThan(candidate[mappingSize], 0, candCount[mappingSize], minTarget);
        }
    }
    for (int i = 1; i < mappingSize; ++i) {
        // for child edge keys, compute the key value
        for (int j: posChildEdge[i]) {
            childKey[j] = computeEdgeKey(din, dout, dun, outID, unID, reverseID, startOffset, i, pos,
                                         childEdgeType[j], dataV[childKeyPos[j][0]], dataV[childKeyPos[j][1]]);
        }
        // for aggregation edge keys, compute the key value
        for (int j: posAggreEdge[i]) {
            aggreKey[j] = computeEdgeKey(din, dout, dun, outID, unID, reverseID, startOffset, i, pos,
                                         aggreEdgeType[j], dataV[aggrePos[2 * j]], dataV[aggrePos[2 * j + 1]]);
        }
    }
    int depth = 0;
    while (depth >= 0) {
        while (pos[mappingSize] < candCount[mappingSize]) {
            VertexID v = candidate[mappingSize][pos[mappingSize]];
            ++pos[mappingSize];
            if (visited[v]) continue;
            visited[v] = true;
            patternV[mappingSize] = tau.nodeOrder[mappingSize];
            dataV[mappingSize] = v;
            // for child edge keys, compute the key value
            for (int j: posChildEdge[mappingSize]) {
                childKey[j] = computeEdgeKey(din, dout, dun, outID, unID, reverseID, startOffset, mappingSize, pos,
                                             childEdgeType[j], dataV[childKeyPos[j][0]], dataV[childKeyPos[j][1]]);
            }
            // for aggregation edge keys, compute the key value
            for (int j: posAggreEdge[mappingSize]) {
                aggreKey[j] = computeEdgeKey(din, dout, dun, outID, unID, reverseID, startOffset, mappingSize, pos,
                                             aggreEdgeType[j], dataV[aggrePos[2 * j]], dataV[aggrePos[2 * j + 1]]);
            }
            if (depth == tau.localOrder.size() - 1) {
#ifdef COLLECT_STATISTICS
                ++gNumMatch;
#endif
                Count cnt = 1;
                // multiply count in children
                for (int j = 0; j < child.size(); ++j) {
                    VertexID cID = child[j];
                    if (childKeyPos[j].size() == 1) {
                        cnt *= H[cID][dataV[childKeyPos[j][0]]];
                    }
                    else {
                        cnt *= H[cID][childKey[j]];
                    }
                }
                if (isRoot) {
                    if (orbitType == 0) h[0] += cnt * aggreWeight[0];
                    else if (orbitType == 1) {
                        for (int j = 0; j < aggreV.size(); ++j) {
#ifdef COLLECT_STATISTICS
                            ++gNumUpdate;
#endif
                            VertexID key = dataV[aggrePos[j]];
                            h[key] += cnt * aggreWeight[j];
                        }
                    }
                    else {
                        for (int j = 0; j < aggreKey.size(); ++j) {
#ifdef COLLECT_STATISTICS
                            ++gNumUpdate;
#endif
                            h[aggreKey[j]] += cnt * aggreWeight[j];
                        }
                    }
                }
                else {
                    if (tau.keySize < 2) {
#ifdef COLLECT_STATISTICS
                        ++gNumUpdate;
#endif
                        h[dataV[aggrePos[0]]] += cnt;
                        if (keyPosSize < sizeBound) {
                            keyPos[keyPosSize] = dataV[aggrePos[0]];
                            ++keyPosSize;
                        }
                    }
                    else {
                        EdgeID e = aggreKey[0];
#ifdef COLLECT_STATISTICS
                        ++gNumUpdate;
#endif
                        h[e] += cnt;
                        if (keyPosSize < sizeBound) {
                            keyPos[keyPosSize] = e;
                            ++keyPosSize;
                        }
                    }
                }
                visited[dataV[mappingSize]] = false;
            }
            else {
#ifdef COLLECT_STATISTICS
                ++gNumIntermediate;
#endif
                ++mappingSize;
                ++depth;
                pos[mappingSize] = 0;
                if (!nodeInterPos[mappingSize]) {
                    if (!nodeOutPos[mappingSize].empty()) {
                        VertexID w = dataV[nodeOutPos[mappingSize][0]];
                        startOffset[mappingSize] = outOffset[w];
                        candidate[mappingSize] = outNbors + outOffset[w];
                        candCount[mappingSize] = outOffset[w + 1] - outOffset[w];
                    }
                    else if (!nodeInPos[mappingSize].empty()){
                        VertexID w = dataV[nodeInPos[mappingSize][0]];
                        startOffset[mappingSize] = inOffset[w];
                        candidate[mappingSize] = inNbors + inOffset[w];
                        candCount[mappingSize] = inOffset[w + 1] - inOffset[w];
                    }
                    else {
                        VertexID w = dataV[nodeUnPos[mappingSize][0]];
                        startOffset[mappingSize] = unOffset[w];
                        candidate[mappingSize] = unNbors + unOffset[w];
                        candCount[mappingSize] = unOffset[w + 1] - unOffset[w];
                    }
                }
                else {
                    generateCandidate(din, dout, dun, dataV, mappingSize, nodeInPos[mappingSize],
                                      nodeOutPos[mappingSize], nodeUnPos[mappingSize], candidate, candCount, tmp);
                }
                // apply the symmetry breaking rules
                if (!greaterPos[mappingSize].empty()) {
                    ui maxTarget = dataV[greaterPos[mappingSize][0]];
                    for (int i = 1; i < greaterPos[mappingSize].size(); ++i) {
                        if (dataV[greaterPos[mappingSize][i]] > maxTarget)
                            maxTarget = dataV[greaterPos[mappingSize][i]];
                    }
                    pos[mappingSize] = firstPosGreaterThan(candidate[mappingSize], 0, candCount[mappingSize], maxTarget);
                }
                if (!lessPos[mappingSize].empty()) {
                    ui minTarget = dataV[lessPos[mappingSize][0]];
                    for (int i = 1; i < lessPos[mappingSize].size(); ++i) {
                        if (dataV[lessPos[mappingSize][i]] < minTarget)
                            minTarget = dataV[lessPos[mappingSize][i]];
                    }
                    candCount[mappingSize] = firstPosGreaterThan(candidate[mappingSize], 0, candCount[mappingSize], minTarget);
                }
            }
        }
        --depth;
        if (depth >= 0) {
            --mappingSize;
            visited[dataV[mappingSize]] = false;
        }
    }
}

void executeNodeEdgeKeyT(
        VertexID nID,
        const Tree &t,
        const std::vector<VertexID> &child,
        VertexID **candidate,
        ui *candCount,
        HashTable *H,
        const DataGraph &din,
        const DataGraph &dout,
        const DataGraph &dun,
        const Triangle &tri,
        const Pattern &p,
        bool isRoot,
        EdgeID *outID,
        EdgeID *unID,
        EdgeID *reverseID,
        EdgeID *startOffset,
        VertexID *patternV,
        VertexID *dataV,
        int mappingSize,
        bool *visited,
        ui *pos,
        ui *keyPos,
        ui &keyPosSize,
        ui sizeBound,
        VertexID *&tmp,
        VertexID *allV
) {
    int orbitType = t.getOrbitType();
    HashTable h = H[nID];
    const Node &tau = t.getNode(nID);
    const std::vector<int> &aggrePos = t.getAggrePos(nID);
    const std::vector<bool> &nodeInterPos = t.getNodeInterPos(nID);
    const std::vector<bool> &nodeCandPos = t.getNodeCandPos(nID);
    const std::vector<std::vector<int>> &nodeInPos = t.getNodeInPos(nID);
    const std::vector<std::vector<int>> &nodeOutPos = t.getNodeOutPos(nID);
    const std::vector<std::vector<int>> &nodeUnPos = t.getNodeUnPos(nID);
    const std::vector<std::vector<int>> &greaterPos = t.getNodeGreaterPos(nID);
    const std::vector<std::vector<int>> &lessPos = t.getNodeLessPos(nID);
    const std::vector<std::vector<int>> &childKeyPos = t.getChildKeyPos(nID);
    const std::vector<VertexID> &aggreV = t.getAggreV();
    const std::vector<int> &aggreWeight = t.getAggreWeight();
    const std::vector<std::vector<int>> &posChildEdge = t.getPosChildEdge(nID);
    const std::vector<std::vector<int>> &posAggreEdge = t.getPosAggreEdge(nID);
    const std::vector<int> &childEdgeType = t.getChildEdgeType(nID);
    const std::vector<int> &aggreEdgeType = t.getAggreEdgeType(nID);
    const std::vector<std::pair<int, int>> &nodeTriPos = t.getNodeTriPos(nID);
    const std::vector<int> &triEdgeType = t.getTriEdgeType(nID);
    const std::vector<int> &triEndType = t.getTriEndType(nID);
    EdgeID *inOffset = din.getOffsets();
    VertexID *inNbors = din.getNbors();
    EdgeID *outOffset = dout.getOffsets();
    VertexID *outNbors = dout.getNbors();
    EdgeID *unOffset = dun.getOffsets();
    VertexID *unNbors = dun.getNbors();
    std::vector<EdgeID> childKey(child.size());
    std::vector<EdgeID> aggreKey;
    if (isRoot && orbitType == 2) aggreKey = std::vector<EdgeID>(aggreWeight.size());
    else if (tau.keySize == 2) aggreKey.push_back(0);
    ui n = din.getNumVertices();
    if (mappingSize == 0) {
        candidate[0] = allV;
        candCount[0] = n;
        pos[mappingSize] = 0;
    }
    else {
        if (!nodeInterPos[mappingSize]) {
            if (!nodeOutPos[mappingSize].empty()) {
                VertexID v = dataV[nodeOutPos[mappingSize][0]];
                startOffset[mappingSize] = outOffset[v];
                candidate[mappingSize] = outNbors + outOffset[v];
                candCount[mappingSize] = outOffset[v + 1] - outOffset[v];
            }
            else if (!nodeInPos[mappingSize].empty()){
                VertexID v = dataV[nodeInPos[mappingSize][0]];
                startOffset[mappingSize] = inOffset[v];
                candidate[mappingSize] = inNbors + inOffset[v];
                candCount[mappingSize] = inOffset[v + 1] - inOffset[v];
            }
            else {
                VertexID v = dataV[nodeUnPos[mappingSize][0]];
                startOffset[mappingSize] = unOffset[v];
                candidate[mappingSize] = unNbors + unOffset[v];
                candCount[mappingSize] = unOffset[v + 1] - unOffset[v];
            }
        }
        else {
            EdgeID e = 0;
            if (triEdgeType[mappingSize] != 0) {
                int pos1 = nodeTriPos[mappingSize].first, pos2 = nodeTriPos[mappingSize].second;
                if (triEdgeType[mappingSize] == 1)
                    e = startOffset[pos2] + pos[pos2] - 1;
                else if (triEdgeType[mappingSize] == 2)
                    e = outID[startOffset[pos2] + pos[pos2] - 1];
                else if (triEdgeType[mappingSize] == 5)
                    e = unID[startOffset[pos2] + pos[pos2] - 1];
                else {
                    VertexID key1 = dataV[pos1], key2 = dataV[pos2];
#ifdef COLLECT_STATISTICS
                    ++gNumEdgeID;
#endif
                    e = dout.getEdgeID(key1, key2);
                }
            }
            generateCandidateT(din, dout, dun, tri, dataV, mappingSize, !nodeCandPos[mappingSize], e, triEndType[mappingSize],
                               nodeInPos[mappingSize], nodeOutPos[mappingSize], nodeUnPos[mappingSize], candidate, candCount, tmp);
        }
        pos[mappingSize] = 0;
        // apply the symmetry breaking rules
        if (!greaterPos[mappingSize].empty()) {
            ui maxTarget = dataV[greaterPos[mappingSize][0]];
            for (int i = 1; i < greaterPos[mappingSize].size(); ++i) {
                if (dataV[greaterPos[mappingSize][i]] > maxTarget)
                    maxTarget = dataV[greaterPos[mappingSize][i]];
            }
            pos[mappingSize] = firstPosGreaterThan(candidate[mappingSize], 0, candCount[mappingSize], maxTarget);
        }
        if (!lessPos[mappingSize].empty()) {
            ui minTarget = dataV[lessPos[mappingSize][0]];
            for (int i = 1; i < lessPos[mappingSize].size(); ++i) {
                if (dataV[lessPos[mappingSize][i]] < minTarget)
                    minTarget = dataV[lessPos[mappingSize][i]];
            }
            candCount[mappingSize] = firstPosGreaterThan(candidate[mappingSize], 0, candCount[mappingSize], minTarget);
        }
    }
    for (int i = 1; i < mappingSize; ++i) {
        // for child edge keys, compute the key value
        for (int j: posChildEdge[i]) {
            childKey[j] = computeEdgeKey(din, dout, dun, outID, unID, reverseID, startOffset, i, pos,
                                         childEdgeType[j], dataV[childKeyPos[j][0]], dataV[childKeyPos[j][1]]);
        }
        // for aggregation edge keys, compute the key value
        for (int j: posAggreEdge[i]) {
            aggreKey[j] = computeEdgeKey(din, dout, dun, outID, unID, reverseID, startOffset, i, pos,
                                         aggreEdgeType[j], dataV[aggrePos[2 * j]], dataV[aggrePos[2 * j + 1]]);
        }
    }
    int depth = 0;
    while (depth >= 0) {
        while (pos[mappingSize] < candCount[mappingSize]) {
            VertexID v = candidate[mappingSize][pos[mappingSize]];
            ++pos[mappingSize];
            if (visited[v]) continue;
            visited[v] = true;
            patternV[mappingSize] = tau.nodeOrder[mappingSize];
            dataV[mappingSize] = v;
            // for child edge keys, compute the key value
            for (int j: posChildEdge[mappingSize]) {
                childKey[j] = computeEdgeKey(din, dout, dun, outID, unID, reverseID, startOffset, mappingSize, pos,
                                             childEdgeType[j], dataV[childKeyPos[j][0]], dataV[childKeyPos[j][1]]);
            }
            // for aggregation edge keys, compute the key value
            for (int j: posAggreEdge[mappingSize]) {
                aggreKey[j] = computeEdgeKey(din, dout, dun, outID, unID, reverseID, startOffset, mappingSize, pos,
                                             aggreEdgeType[j], dataV[aggrePos[2 * j]], dataV[aggrePos[2 * j + 1]]);
            }
            if (depth == tau.localOrder.size() - 1) {
#ifdef COLLECT_STATISTICS
                ++gNumMatch;
#endif
                Count cnt = 1;
                // multiply count in children
                for (int j = 0; j < child.size(); ++j) {
                    VertexID cID = child[j];
                    if (childKeyPos[j].size() == 1) {
                        cnt *= H[cID][dataV[childKeyPos[j][0]]];
                    }
                    else {
                        cnt *= H[cID][childKey[j]];
                    }
                }
                if (isRoot) {
                    if (orbitType == 0) h[0] += cnt * aggreWeight[0];
                    else if (orbitType == 1) {
                        for (int j = 0; j < aggreV.size(); ++j) {
                            VertexID key = dataV[aggrePos[j]];
#ifdef COLLECT_STATISTICS
                            ++gNumUpdate;
#endif
                            h[key] += cnt * aggreWeight[j];
                        }
                    }
                    else {
                        for (int j = 0; j < aggreKey.size(); ++j) {
#ifdef COLLECT_STATISTICS
                            ++gNumUpdate;
#endif
                            h[aggreKey[j]] += cnt * aggreWeight[j];
                        }
                    }
                }
                else {
                    if (tau.keySize < 2) {
#ifdef COLLECT_STATISTICS
                        ++gNumUpdate;
#endif
                        h[dataV[aggrePos[0]]] += cnt;
                        if (keyPosSize < sizeBound) {
                            keyPos[keyPosSize] = dataV[aggrePos[0]];
                            ++keyPosSize;
                        }
                    }
                    else {
                        EdgeID e = aggreKey[0];
#ifdef COLLECT_STATISTICS
                        ++gNumUpdate;
#endif
                        h[e] += cnt;
                        if (keyPosSize < sizeBound) {
                            keyPos[keyPosSize] = e;
                            ++keyPosSize;
                        }
                    }
                }
                visited[dataV[mappingSize]] = false;
            }
            else {
#ifdef COLLECT_STATISTICS
                ++gNumIntermediate;
#endif
                ++mappingSize;
                ++depth;
                pos[mappingSize] = 0;
                if (!nodeInterPos[mappingSize]) {
                    if (!nodeOutPos[mappingSize].empty()) {
                        VertexID w = dataV[nodeOutPos[mappingSize][0]];
                        startOffset[mappingSize] = outOffset[w];
                        candidate[mappingSize] = outNbors + outOffset[w];
                        candCount[mappingSize] = outOffset[w + 1] - outOffset[w];
                    }
                    else if (!nodeInPos[mappingSize].empty()){
                        VertexID w = dataV[nodeInPos[mappingSize][0]];
                        startOffset[mappingSize] = inOffset[w];
                        candidate[mappingSize] = inNbors + inOffset[w];
                        candCount[mappingSize] = inOffset[w + 1] - inOffset[w];
                    }
                    else {
                        VertexID w = dataV[nodeUnPos[mappingSize][0]];
                        startOffset[mappingSize] = unOffset[w];
                        candidate[mappingSize] = unNbors + unOffset[w];
                        candCount[mappingSize] = unOffset[w + 1] - unOffset[w];
                    }
                }
                else {
                    EdgeID e = 0;
                    if (triEdgeType[mappingSize] != 0) {
                        int pos1 = nodeTriPos[mappingSize].first, pos2 = nodeTriPos[mappingSize].second;
                        if (triEdgeType[mappingSize] == 1)
                            e = startOffset[pos2] + pos[pos2] - 1;
                        else if (triEdgeType[mappingSize] == 2)
                            e = outID[startOffset[pos2] + pos[pos2] - 1];
                        else if (triEdgeType[mappingSize] == 5)
                            e = unID[startOffset[pos2] + pos[pos2] - 1];
                        else {
                            VertexID key1 = dataV[pos1], key2 = dataV[pos2];
#ifdef COLLECT_STATISTICS
                            ++gNumEdgeID;
#endif
                            e = dout.getEdgeID(key1, key2);
                        }
                    }
                    generateCandidateT(din, dout, dun, tri, dataV, mappingSize, !nodeCandPos[mappingSize], e, triEndType[mappingSize],
                                       nodeInPos[mappingSize], nodeOutPos[mappingSize], nodeUnPos[mappingSize], candidate, candCount, tmp);
                }
                // apply the symmetry breaking rules
                if (!greaterPos[mappingSize].empty()) {
                    ui maxTarget = dataV[greaterPos[mappingSize][0]];
                    for (int i = 1; i < greaterPos[mappingSize].size(); ++i) {
                        if (dataV[greaterPos[mappingSize][i]] > maxTarget)
                            maxTarget = dataV[greaterPos[mappingSize][i]];
                    }
                    pos[mappingSize] = firstPosGreaterThan(candidate[mappingSize], 0, candCount[mappingSize], maxTarget);
                }
                if (!lessPos[mappingSize].empty()) {
                    ui minTarget = dataV[lessPos[mappingSize][0]];
                    for (int i = 1; i < lessPos[mappingSize].size(); ++i) {
                        if (dataV[lessPos[mappingSize][i]] < minTarget)
                            minTarget = dataV[lessPos[mappingSize][i]];
                    }
                    candCount[mappingSize] = firstPosGreaterThan(candidate[mappingSize], 0, candCount[mappingSize], minTarget);
                }
            }
        }
        --depth;
        if (depth >= 0) {
            --mappingSize;
            visited[dataV[mappingSize]] = false;
        }
    }
}

void executePartition(
        VertexID pID,
        const Tree &t,
        VertexID ***candidate,
        ui **candCount,
        HashTable *H,
        const DataGraph &din,
        const DataGraph &dout,
        const DataGraph &dun,
        bool useTriangle,
        const Triangle &tri,
        const Pattern &p,
        EdgeID *outID,
        EdgeID *unID,
        EdgeID *reverseID,
        EdgeID *startOffset,
        VertexID *patternV,
        VertexID *dataV,
        bool *visited,
        ui *pos,
        VertexID *&tmp,
        VertexID *allV
) {
    const std::vector<std::vector<VertexID>> &globalOrder = t.getGlobalOrder();
    const std::vector<std::vector<std::vector<VertexID>>> &nodesAtStep = t.getNodesAtStep();
    const std::vector<VertexID> &partitionOrder = globalOrder[pID];
    const std::vector<std::vector<VertexID>> &child = t.getChild();
    const std::vector<VertexID> &postOrder = t.getPostOrder();
    const std::vector<int> &partitionPos = t.getPartitionPos();
    const std::vector<std::vector<int>> &partitionInPos = t.getPartitionInPos(pID);
    const std::vector<std::vector<int>> &partitionOutPos = t.getPartitionOutPos(pID);
    const std::vector<std::vector<int>> &partitionUnPos = t.getPartitionUnPos(pID);
    const std::vector<bool> &partitionInterPos = t.getPartitionInterPos(pID);
    const std::vector<std::vector<int>> &greaterPos = t.getPartitionGreaterPos(pID);
    const std::vector<std::vector<int>> &lessPos = t.getPartitionLessPos(pID);
    const std::vector<bool> &partitionCandPos = t.getPartitionCandPos(pID);
    const std::vector<std::pair<int, int>> &partitionTriPos = t.getPartitionTriPos(pID);
    const std::vector<int> &triEdgeType = t.getPartitionEdgeType(pID);
    const std::vector<int> &triEndType = t.getPartitionEndType(pID);
    EdgeID *inOffset = din.getOffsets();
    VertexID *inNbors = din.getNbors();
    EdgeID *outOffset = dout.getOffsets();
    VertexID *outNbors = dout.getNbors();
    EdgeID *unOffset = dun.getOffsets();
    EdgeID *unNbors = dun.getNbors();
    int startPos;
    if (pID == 0) startPos = 0;
    else startPos = partitionPos[pID - 1] + 1;
    int endPos = partitionPos[pID] + 1;
    bool isRoot = endPos == postOrder.size();
    ui n = dun.getNumVertices(), m = dun.getNumEdges();
    int mappingSize = 0;
    const std::vector<std::vector<VertexID>> &allChild = t.getChild();
    ui argument = 0;
    if (partitionOrder.empty()) {
        for (int i = startPos; i < endPos; ++i) {
            VertexID nID = postOrder[i];
            isRoot = endPos == postOrder.size() && i == endPos - 1;
            if (!t.nodeEdgeKey(nID) && !useTriangle) {
                executeNode(nID, t, allChild[nID], candidate[nID], candCount[nID], H, din, dout, dun, p, isRoot, outID, unID, reverseID,
                            startOffset, patternV, dataV, 0, visited, pos, nullptr, argument, argument, tmp, allV);
            }
            else if (!t.nodeEdgeKey(nID) && useTriangle) {
                executeNodeT(nID, t, allChild[nID], candidate[nID], candCount[nID], H, din, dout, dun, tri, p, isRoot, outID, unID, reverseID,
                             startOffset, patternV, dataV, 0, visited, pos, nullptr, argument, argument, tmp, allV);

            }
            else if (t.nodeEdgeKey(nID) && !useTriangle) {
                executeNodeEdgeKey(nID, t, allChild[nID], candidate[nID], candCount[nID], H, din, dout, dun, p, isRoot, outID, unID, reverseID,
                                   startOffset, patternV, dataV, 0, visited, pos, nullptr, argument, argument, tmp, allV);
            }
            else {
                executeNodeEdgeKeyT(nID, t, allChild[nID], candidate[nID], candCount[nID], H, din, dout, dun, tri, p, isRoot, outID, unID, reverseID,
                                    startOffset, patternV, dataV, 0, visited, pos, nullptr, argument, argument, tmp, allV);
            }
        }
        return;
    }
    // let candidate points to a memory whose length is the number of vertices in the data graph
    VertexID **partitionCandidate = new VertexID *[partitionOrder.size()];
    for (int i = 0; i < partitionOrder.size(); ++i) {
        if (partitionCandPos[i])
            partitionCandidate[i] = new VertexID[dout.getNumVertices()];
    }
    ui *partitionCandCount = new ui[partitionOrder.size()];
    // for each node, store the keys whose values are not 0
    ui **keyPos = new ui *[t.getNumNodes()];
    for (int i = startPos; i < endPos; ++i) {
        VertexID nID = postOrder[i];
        if (t.getNode(nID).keySize == 0)
            keyPos[nID] = new ui[1];
        else if (t.getNode(nID).keySize == 1)
            keyPos[nID] = new ui[n / 8 + 1];
        else if (t.getNode(nID).keySize == 2)
            keyPos[nID] = new ui[m / 8 + 1];
    }

    ui *keyPosSize = new ui[t.getNumNodes()];
    memset(keyPosSize, 0, sizeof(ui) * (t.getNumNodes()));
    partitionCandidate[0] = allV;
    partitionCandCount[0] = n;
    pos[0] = 0;
    // depth iterate through the pattern vertices in the partition order
    int depth = 0;
    while (depth >= 0) {
        while (pos[depth] < partitionCandCount[depth]) {
            VertexID v = partitionCandidate[depth][pos[depth]];
            ++pos[depth];
            if (visited[v]) continue;
            visited[v] = true;
            patternV[depth] = partitionOrder[depth];
            dataV[depth] = v;
            for (VertexID nID: nodesAtStep[pID][depth]) {
                if (nID == postOrder[endPos - 1]) {
                    if (!t.nodeEdgeKey(nID) && !useTriangle) {
                        executeNode(nID, t, allChild[nID], candidate[nID], candCount[nID], H, din, dout, dun, p, isRoot, outID, unID, reverseID,
                                    startOffset, patternV, dataV, mappingSize + 1, visited, pos, nullptr, argument, argument, tmp, allV);
                    }
                    else if (!t.nodeEdgeKey(nID) && useTriangle) {
                        executeNodeT(nID, t, allChild[nID], candidate[nID], candCount[nID], H, din, dout, dun, tri, p, isRoot, outID, unID, reverseID,
                                     startOffset, patternV, dataV, mappingSize + 1, visited, pos, nullptr, argument, argument, tmp, allV);
                    }
                    else if (t.nodeEdgeKey(nID) && !useTriangle) {
                        executeNodeEdgeKey(nID, t, allChild[nID], candidate[nID], candCount[nID], H, din, dout, dun, p, isRoot, outID, unID, reverseID,
                                           startOffset, patternV, dataV, mappingSize + 1, visited, pos, nullptr, argument, argument, tmp, allV);
                    }
                    else {
                        executeNodeEdgeKeyT(nID, t, allChild[nID], candidate[nID], candCount[nID], H, din, dout, dun, tri, p, isRoot, outID, unID, reverseID,
                                            startOffset, patternV, dataV, mappingSize + 1, visited, pos, nullptr, argument, argument, tmp, allV);
                    }
                }
                else if (t.getNode(nID).keySize < 2) {
                    if (keyPosSize[nID] < n / 8 + 1) {
                        for (int j = 0; j < keyPosSize[nID]; ++j) {
                            H[nID][keyPos[nID][j]] = 0;
                        }
                    }
                    else
                        memset(H[nID], 0, sizeof(Count) * n);
                    keyPosSize[nID] = 0;
                    ui sizeBound = 1;
                    if (t.getNode(nID).keySize == 1) sizeBound = n / 8 + 1;
                    if (t.getNode(nID).keySize == 2) sizeBound = m / 8 + 1;
                    if (!t.nodeEdgeKey(nID) && !useTriangle) {
                        executeNode(nID, t, allChild[nID], candidate[nID], candCount[nID], H, din, dout, dun, p, false, outID, unID, reverseID,
                                    startOffset, patternV, dataV, mappingSize + 1, visited, pos, keyPos[nID], keyPosSize[nID], sizeBound, tmp, allV);
                    }
                    else if (!t.nodeEdgeKey(nID) && useTriangle) {
                        executeNodeT(nID, t, allChild[nID], candidate[nID], candCount[nID], H, din, dout, dun, tri, p, false, outID, unID, reverseID,
                                     startOffset, patternV, dataV, mappingSize + 1, visited, pos, keyPos[nID], keyPosSize[nID], sizeBound, tmp, allV);
                    }
                    else if (t.nodeEdgeKey(nID) && !useTriangle) {
                        executeNodeEdgeKey(nID, t, allChild[nID], candidate[nID], candCount[nID], H, din, dout, dun, p, false, outID, unID, reverseID,
                                           startOffset, patternV, dataV, mappingSize + 1, visited, pos, keyPos[nID], keyPosSize[nID], sizeBound, tmp, allV);
                    }
                    else {
                        executeNodeEdgeKeyT(nID, t, allChild[nID], candidate[nID], candCount[nID], H, din, dout, dun, tri, p, false, outID, unID, reverseID,
                                            startOffset, patternV, dataV, mappingSize + 1, visited, pos, keyPos[nID], keyPosSize[nID], sizeBound, tmp, allV);
                    }
                }
                else {
                    if (keyPosSize[nID] < m / 8) {
                        for (int j = 0; j < keyPosSize[nID]; ++j) {
                            H[nID][keyPos[nID][j]] = 0;
                        }
                    }
                    else
                        memset(H[nID], 0, sizeof(Count) * m);
                    keyPosSize[nID] = 0;
                    if (!useTriangle) {
                        executeNodeEdgeKey(nID, t, allChild[nID], candidate[nID], candCount[nID], H, din, dout, dun, p, false, outID, unID, reverseID, startOffset,
                                           patternV, dataV, mappingSize + 1, visited, pos, keyPos[nID], keyPosSize[nID], m / 8, tmp, allV);
                    }
                    else {
                        executeNodeEdgeKeyT(nID, t, allChild[nID], candidate[nID], candCount[nID], H, din, dout, dun, tri, p, false, outID, unID, reverseID, startOffset,
                                            patternV, dataV, mappingSize + 1, visited, pos, keyPos[nID], keyPosSize[nID], m / 8, tmp, allV);
                    }
                }
            }
            if (depth == partitionOrder.size() - 1) {
                visited[dataV[mappingSize]] = false;
            }
            else {
                ++mappingSize;
                ++depth;
                pos[depth] = 0;
                if (!partitionInterPos[depth]) {
                    if (!partitionOutPos[depth].empty()) {
                        VertexID w = dataV[partitionOutPos[depth][0]];
                        startOffset[mappingSize] = outOffset[w];
                        partitionCandidate[depth] = outNbors + outOffset[w];
                        partitionCandCount[depth] = outOffset[w + 1] - outOffset[w];
                    }
                    else if (!partitionInPos[depth].empty()){
                        VertexID w = dataV[partitionInPos[depth][0]];
                        startOffset[mappingSize] = inOffset[w];
                        partitionCandidate[depth] = inNbors + inOffset[w];
                        partitionCandCount[depth] = inOffset[w + 1] - inOffset[w];
                    }
                    else {
                        VertexID w = dataV[partitionUnPos[depth][0]];
                        startOffset[mappingSize] = unOffset[w];
                        partitionCandidate[depth] = unNbors + unOffset[w];
                        partitionCandCount[depth] = unOffset[w + 1] - unOffset[w];
                    }
                }
                else {
                    if (useTriangle) {
                        EdgeID e = 0;
                        if (triEdgeType[mappingSize] != 0) {
                            int pos1 = partitionTriPos[mappingSize].first, pos2 = partitionTriPos[mappingSize].second;
                            if (triEdgeType[mappingSize] == 1)
                                e = startOffset[pos2] + pos[pos2] - 1;
                            else if (triEdgeType[mappingSize] == 2)
                                e = outID[startOffset[pos2] + pos[pos2] - 1];
                            else if (triEdgeType[mappingSize] == 5)
                                e = unID[startOffset[pos2] + pos[pos2] - 1];
                            else{
                                VertexID key1 = dataV[pos1], key2 = dataV[pos2];
#ifdef COLLECT_STATISTICS
                                ++gNumEdgeID;
#endif
                                e = dout.getEdgeID(key1, key2);
                            }
                        }
                        generateCandidateT(din, dout, dun, tri, dataV, mappingSize, !partitionCandPos[mappingSize], e, triEndType[mappingSize],
                                           partitionInPos[mappingSize], partitionOutPos[mappingSize], partitionUnPos[mappingSize], partitionCandidate, partitionCandCount, tmp);

                    }
                    else {
                        generateCandidate(din, dout, dun, dataV, depth, partitionInPos[depth], partitionOutPos[depth],
                                          partitionUnPos[depth], partitionCandidate, partitionCandCount, tmp);
                    }
                }
                // apply the symmetry breaking rules
                if (!greaterPos[mappingSize].empty()) {
                    ui maxTarget = dataV[greaterPos[mappingSize][0]];
                    for (int i = 1; i < greaterPos[mappingSize].size(); ++i) {
                        if (dataV[greaterPos[mappingSize][i]] > maxTarget)
                            maxTarget = dataV[greaterPos[mappingSize][i]];
                    }
                    pos[mappingSize] = firstPosGreaterThan(partitionCandidate[mappingSize], 0, partitionCandCount[mappingSize], maxTarget);
                }
                if (!lessPos[mappingSize].empty()) {
                    ui minTarget = dataV[lessPos[mappingSize][0]];
                    for (int i = 1; i < lessPos[mappingSize].size(); ++i) {
                        if (dataV[lessPos[mappingSize][i]] < minTarget)
                            minTarget = dataV[lessPos[mappingSize][i]];
                    }
                    partitionCandCount[mappingSize] = firstPosGreaterThan(partitionCandidate[mappingSize], 0, partitionCandCount[mappingSize], minTarget);
                }

            }
        }
        --depth;
        --mappingSize;
        if (depth >= 0)
            visited[dataV[depth]] = false;
    }
    for (int j = 0; j < partitionOrder.size(); ++j) {
        if (partitionCandPos[j])
            delete[] partitionCandidate[j];
    }
    delete[] partitionCandidate;
    delete[] partitionCandCount;
    for (int j = startPos; j < endPos; ++j) {
        VertexID nID = postOrder[j];
        delete[] keyPos[postOrder[j]];
    }
    delete[] keyPos;
    delete[] keyPosSize;
}

void executeTree (
        const Tree &t,
        const DataGraph &din,
        const DataGraph &dout,
        const DataGraph &dun,
        bool useTriangle,
        const Triangle &tri,
        const Pattern &p,
        HashTable *H,
        EdgeID *outID,
        EdgeID *unID,
        EdgeID *reverseID,
        EdgeID *startOffset,
        VertexID *patternV,
        VertexID *dataV,
        bool *visited,
        ui *pos,
        VertexID *&tmp,
        VertexID *allV
) {
    int numNodes = (int)t.getNumNodes();
    VertexID ***candidate = new VertexID **[numNodes];
    ui **candCount = new ui *[numNodes];
    for (VertexID nID = 0; nID < numNodes; ++nID) {
        const Node &tau = t.getNode(nID);
        int partOrderLength = int(tau.nodeOrder.size() - tau.localOrder.size());
        candCount[nID] = new ui[tau.nodeOrder.size()];
        candidate[nID] = new VertexID *[tau.nodeOrder.size()];
        const std::vector<bool> &nodeCandPos = t.getNodeCandPos(nID);
        for (int i = partOrderLength; i < nodeCandPos.size(); ++i) {
            if (nodeCandPos[i])
                candidate[nID][i] = new VertexID[dout.getNumVertices()];
        }
    }

    // for each partition, calls executePartition to build a full hash table
    for (VertexID pID = 0; pID < t.getPartitionPos().size(); ++pID) {
        executePartition(pID, t, candidate, candCount, H, din, dout, dun, useTriangle, tri,
                         p, outID, unID, reverseID, startOffset, patternV, dataV, visited, pos, tmp, allV);
    }
    for (VertexID nID = 0; nID < numNodes; ++nID) {
        const Node &tau = t.getNode(nID);
        int partOrderLength = int(tau.nodeOrder.size() - tau.localOrder.size());
        const std::vector<bool> &nodeCandPos = t.getNodeCandPos(nID);
        for (int i = partOrderLength; i < nodeCandPos.size(); ++i) {
            if (nodeCandPos[i])
                delete[] candidate[nID][i];
        }
        delete[] candidate[nID];
        delete[] candCount[nID];
    }
    delete[] candCount;
    delete[] candidate;
}

void multiJoin(
        VertexID nID,
        const Tree &t,
        const std::vector<VertexID> &child,
        VertexID ***candidates,
        ui **candCounts,
        HashTable *H,
        const DataGraph &din,
        const DataGraph &dout,
        const DataGraph &dun,
        const Triangle &tri,
        const Pattern &p,
        EdgeID *outID,
        EdgeID *unID,
        EdgeID *reverseID,
        EdgeID **startOffsets,
        VertexID **patternVs,
        VertexID **dataVs,
        bool **visits,
        ui **poses,
        ui **keyPoses,
        ui *keyPosSizes,
        ui *sizeBounds,
        VertexID *&tmp,
        VertexID *allV
) {
    int orbitType = t.getOrbitType();
    HashTable h = H[nID];
    const Node &tau = t.getNode(nID);
    const std::vector<int> &aggrePos = t.getAggrePos(nID);
    const std::vector<bool> &nodeInterPos = t.getNodeInterPos(nID);
    const std::vector<bool> &nodeCandPos = t.getNodeCandPos(nID);
    const std::vector<std::vector<int>> &nodeInPos = t.getNodeInPos(nID);
    const std::vector<std::vector<int>> &nodeOutPos = t.getNodeOutPos(nID);
    const std::vector<std::vector<int>> &nodeUnPos = t.getNodeUnPos(nID);
    const std::vector<std::vector<int>> &greaterPos = t.getNodeGreaterPos(nID);
    const std::vector<std::vector<int>> &lessPos = t.getNodeLessPos(nID);
    const std::vector<std::vector<int>> &childKeyPos = t.getChildKeyPos(nID);
    const std::vector<VertexID> &aggreV = t.getAggreV();
    const std::vector<int> &aggreWeight = t.getAggreWeight();
    const std::vector<std::vector<VertexID>> &nodesAtStep = t.getNodesAtStep()[nID];
    const std::vector<std::vector<int>> &prefixPos = t.getPrefixPos();
    EdgeID *inOffset = din.getOffsets();
    VertexID *inNbors = din.getNbors();
    EdgeID *outOffset = dout.getOffsets();
    VertexID *outNbors = dout.getNbors();
    EdgeID *unOffset = dun.getOffsets();
    VertexID *unNbors = dun.getNbors();
    VertexID **candidate = candidates[nID];
    ui *candCount = candCounts[nID];
    ui *pos = poses[nID];
    VertexID *patternV = patternVs[nID];
    VertexID *dataV = dataVs[nID];
    EdgeID *startOffset = startOffsets[nID];
    bool *visited = visits[nID];
    ui *keyPos = keyPoses[nID];
    ui &keyPosSize = keyPosSizes[nID];
    ui sizeBound = sizeBounds[nID];
    bool isRoot = nID == t.getRootID();
    ui n = dun.getNumVertices(), m = dun.getNumEdges();
    if (tau.prefixSize != 0) {
        for (int i = 0; i < nodesAtStep[0].size(); ++i) {
            VertexID cID = nodesAtStep[0][i];
            for (int j = 0; j < prefixPos[cID].size(); ++j) {
                patternVs[cID][j] = patternV[prefixPos[cID][j]];
                dataVs[cID][j] = dataV[prefixPos[cID][j]];
                startOffsets[cID][j] = startOffset[prefixPos[cID][j]];
                poses[cID][j] = pos[prefixPos[cID][j]];
                visits[cID][dataVs[cID][j]] = true;
            }
            if (t.getNode(cID).keySize == 0) H[cID][0] = 0;
            else if (t.getNode(cID).keySize == 1) {
                if (keyPosSizes[cID] < n / 8 + 1) {
                    for (int j = 0; j < keyPosSizes[cID]; ++j) {
                        H[cID][keyPoses[cID][j]] = 0;
                    }
                }
                else memset(H[cID], 0, sizeof(Count) * n);
            }
            else {
                if (keyPosSizes[cID] < m / 8 + 1) {
                    for (int j = 0; j < keyPosSizes[cID]; ++j) {
                        H[cID][keyPoses[cID][j]] = 0;
                    }
                }
                else memset(H[cID], 0, sizeof(Count) * m);
            }
            multiJoinWrapper(cID, t, t.getChild()[cID], candidates, candCounts, H, din, dout, dun, true, tri, p, outID, unID,
                             reverseID, startOffsets, patternVs, dataVs, visits, poses, keyPoses, keyPosSizes, sizeBounds, tmp, allV);
            for (int j = 0; j < prefixPos[cID].size(); ++j) {
                visits[cID][dataVs[cID][j]] = false;
            }
        }
    }
    int mappingSize = tau.prefixSize;
    if (mappingSize == 0) {
        candidate[0] = allV;
        candCount[0] = n;
        pos[0] = 0;
    }
    else {
        if (!nodeInterPos[mappingSize]) {
            if (!nodeOutPos[mappingSize].empty()) {
                VertexID v = dataV[nodeOutPos[mappingSize][0]];
                startOffset[mappingSize] = outOffset[v];
                candidate[mappingSize] = outNbors + outOffset[v];
                candCount[mappingSize] = outOffset[v + 1] - outOffset[v];
            }
            else if (!nodeInPos[mappingSize].empty()){
                VertexID v = dataV[nodeInPos[mappingSize][0]];
                startOffset[mappingSize] = inOffset[v];
                candidate[mappingSize] = inNbors + inOffset[v];
                candCount[mappingSize] = inOffset[v + 1] - inOffset[v];
            }
            else {
                VertexID v = dataV[nodeUnPos[mappingSize][0]];
                startOffset[mappingSize] = unOffset[v];
                candidate[mappingSize] = unNbors + unOffset[v];
                candCount[mappingSize] = unOffset[v + 1] - unOffset[v];
            }
        }
        else {
            generateCandidate(din, dout, dun, dataV, mappingSize, nodeInPos[mappingSize],
                              nodeOutPos[mappingSize], nodeUnPos[mappingSize], candidate, candCount, tmp);
        }
        pos[mappingSize] = 0;
        // apply the symmetry breaking rules
        if (!greaterPos[mappingSize].empty()) {
            ui maxTarget = dataV[greaterPos[mappingSize][0]];
            for (int i = 1; i < greaterPos[mappingSize].size(); ++i) {
                if (dataV[greaterPos[mappingSize][i]] > maxTarget)
                    maxTarget = dataV[greaterPos[mappingSize][i]];
            }
            pos[mappingSize] = firstPosGreaterThan(candidate[mappingSize], 0, candCount[mappingSize], maxTarget);
        }
        if (!lessPos[mappingSize].empty()) {
            ui minTarget = dataV[lessPos[mappingSize][0]];
            for (int i = 1; i < lessPos[mappingSize].size(); ++i) {
                if (dataV[lessPos[mappingSize][i]] < minTarget)
                    minTarget = dataV[lessPos[mappingSize][i]];
            }
            candCount[mappingSize] = firstPosGreaterThan(candidate[mappingSize], 0, candCount[mappingSize], minTarget);
        }
    }
    pos[0] = 0;
    while (mappingSize >= (int)tau.prefixSize) {
        while (pos[mappingSize] < candCount[mappingSize]) {
            VertexID v = candidate[mappingSize][pos[mappingSize]];
            ++pos[mappingSize];
            if (visited[v]) continue;
            visited[v] = true;
            patternV[mappingSize] = tau.nodeOrder[mappingSize];
            dataV[mappingSize] = v;
            for (int i = 0; i < nodesAtStep[mappingSize].size(); ++i) {
                VertexID cID = nodesAtStep[mappingSize][i];
                for (int j = 0; j < prefixPos[cID].size(); ++j) {
                    patternVs[cID][j] = patternV[prefixPos[cID][j]];
                    dataVs[cID][j] = dataV[prefixPos[cID][j]];
                    startOffsets[cID][j] = startOffset[prefixPos[cID][j]];
                    poses[cID][j] = pos[prefixPos[cID][j]];
                    visits[cID][dataVs[cID][j]] = true;
                }
                if (t.getNode(cID).keySize == 0) H[cID][0] = 0;
                else if (t.getNode(cID).keySize == 1) {
                    if (keyPosSizes[cID] < n / 8 + 1) {
                        for (int j = 0; j < keyPosSizes[cID]; ++j) {
                            H[cID][keyPoses[cID][j]] = 0;
                        }
                    }
                    else memset(H[cID], 0, sizeof(Count) * n);
                }
                else {
                    if (keyPosSizes[cID] < m / 8 + 1) {
                        for (int j = 0; j < keyPosSizes[cID]; ++j) {
                            H[cID][keyPoses[cID][j]] = 0;
                        }
                    }
                    else memset(H[cID], 0, sizeof(Count) * m);
                }
                multiJoinWrapper(cID, t, t.getChild()[cID], candidates, candCounts, H, din, dout, dun, true, tri, p, outID, unID,
                                 reverseID, startOffsets, patternVs, dataVs, visits, poses, keyPoses, keyPosSizes, sizeBounds, tmp, allV);
                for (int j = 0; j < prefixPos[cID].size(); ++j) {
                    visits[cID][dataVs[cID][j]] = false;
                }
            }
            if (mappingSize == tau.numVertices - 1) {
#ifdef COLLECT_STATISTICS
                ++gNumMatch;
#endif
                Count cnt = 1;
                // multiply count in children
                for (int j = 0; j < child.size(); ++j) {
                    VertexID cID = child[j];
                    if (childKeyPos[j].empty()) cnt *= H[cID][0];
                    else cnt *= H[cID][dataV[childKeyPos[j][0]]];
                }
                if (isRoot) {
                    if (orbitType == 0) h[0] += cnt * aggreWeight[0];
                    else {
                        for (int j = 0; j < aggreV.size(); ++j) {
                            VertexID key = dataV[aggrePos[j]];
#ifdef COLLECT_STATISTICS
                            ++gNumUpdate;
#endif
                            h[key] += cnt * aggreWeight[j];
                        }
                    }
                }
                else {
                    if (tau.keySize == 0) h[0] += cnt;
                    else {
#ifdef COLLECT_STATISTICS
                        ++gNumUpdate;
#endif
                        h[dataV[aggrePos[0]]] += cnt;
                        if (keyPosSize < sizeBound) {
                            keyPos[keyPosSize] = dataV[aggrePos[0]];
                            ++keyPosSize;
                        }
                    }
                }
                visited[dataV[mappingSize]] = false;
            }
            else {
#ifdef COLLECT_STATISTICS
                ++gNumIntermediate;
#endif
                ++mappingSize;
                pos[mappingSize] = 0;
                if (!nodeInterPos[mappingSize]) {
                    if (!nodeOutPos[mappingSize].empty()) {
                        VertexID w = dataV[nodeOutPos[mappingSize][0]];
                        startOffset[mappingSize] = outOffset[w];
                        candidate[mappingSize] = outNbors + outOffset[w];
                        candCount[mappingSize] = outOffset[w + 1] - outOffset[w];
                    }
                    else if (!nodeInPos[mappingSize].empty()){
                        VertexID w = dataV[nodeInPos[mappingSize][0]];
                        startOffset[mappingSize] = inOffset[w];
                        candidate[mappingSize] = inNbors + inOffset[w];
                        candCount[mappingSize] = inOffset[w + 1] - inOffset[w];
                    }
                    else {
                        VertexID w = dataV[nodeUnPos[mappingSize][0]];
                        startOffset[mappingSize] = unOffset[w];
                        candidate[mappingSize] = unNbors + unOffset[w];
                        candCount[mappingSize] = unOffset[w + 1] - unOffset[w];
                    }
                }
                else {
                    generateCandidate(din, dout, dun, dataV, mappingSize, nodeInPos[mappingSize],
                                      nodeOutPos[mappingSize], nodeUnPos[mappingSize], candidate, candCount, tmp);
                }
                // apply the symmetry breaking rules
                if (!greaterPos[mappingSize].empty()) {
                    ui maxTarget = dataV[greaterPos[mappingSize][0]];
                    for (int i = 1; i < greaterPos[mappingSize].size(); ++i) {
                        if (dataV[greaterPos[mappingSize][i]] > maxTarget)
                            maxTarget = dataV[greaterPos[mappingSize][i]];
                    }
                    pos[mappingSize] = firstPosGreaterThan(candidate[mappingSize], 0, candCount[mappingSize], maxTarget);
                }
                if (!lessPos[mappingSize].empty()) {
                    ui minTarget = dataV[lessPos[mappingSize][0]];
                    for (int i = 1; i < lessPos[mappingSize].size(); ++i) {
                        if (dataV[lessPos[mappingSize][i]] < minTarget)
                            minTarget = dataV[lessPos[mappingSize][i]];
                    }
                    candCount[mappingSize] = firstPosGreaterThan(candidate[mappingSize], 0, candCount[mappingSize], minTarget);
                }
            }
        }
        --mappingSize;
        if (mappingSize >= (int)tau.prefixSize) {
            visited[dataV[mappingSize]] = false;
        }
    }
}

void multiJoinT(
        VertexID nID,
        const Tree &t,
        const std::vector<VertexID> &child,
        VertexID ***candidates,
        ui **candCounts,
        HashTable *H,
        const DataGraph &din,
        const DataGraph &dout,
        const DataGraph &dun,
        const Triangle &tri,
        const Pattern &p,
        EdgeID *outID,
        EdgeID *unID,
        EdgeID *reverseID,
        EdgeID **startOffsets,
        VertexID **patternVs,
        VertexID **dataVs,
        bool **visits,
        ui **poses,
        ui **keyPoses,
        ui *keyPosSizes,
        ui *sizeBounds,
        VertexID *&tmp,
        VertexID *allV
) {
    int orbitType = t.getOrbitType();
    HashTable h = H[nID];
    const Node &tau = t.getNode(nID);
    const std::vector<int> &aggrePos = t.getAggrePos(nID);
    const std::vector<bool> &nodeInterPos = t.getNodeInterPos(nID);
    const std::vector<bool> &nodeCandPos = t.getNodeCandPos(nID);
    const std::vector<std::vector<int>> &nodeInPos = t.getNodeInPos(nID);
    const std::vector<std::vector<int>> &nodeOutPos = t.getNodeOutPos(nID);
    const std::vector<std::vector<int>> &nodeUnPos = t.getNodeUnPos(nID);
    const std::vector<std::vector<int>> &greaterPos = t.getNodeGreaterPos(nID);
    const std::vector<std::vector<int>> &lessPos = t.getNodeLessPos(nID);
    const std::vector<std::vector<int>> &childKeyPos = t.getChildKeyPos(nID);
    const std::vector<VertexID> &aggreV = t.getAggreV();
    const std::vector<int> &aggreWeight = t.getAggreWeight();
    const std::vector<std::pair<int, int>> &nodeTriPos = t.getNodeTriPos(nID);
    const std::vector<int> &triEdgeType = t.getTriEdgeType(nID);
    const std::vector<int> &triEndType = t.getTriEndType(nID);
    const std::vector<std::vector<VertexID>> &nodesAtStep = t.getNodesAtStep()[nID];
    const std::vector<std::vector<int>> &prefixPos = t.getPrefixPos();
    EdgeID *inOffset = din.getOffsets();
    VertexID *inNbors = din.getNbors();
    EdgeID *outOffset = dout.getOffsets();
    VertexID *outNbors = dout.getNbors();
    EdgeID *unOffset = dun.getOffsets();
    VertexID *unNbors = dun.getNbors();
    VertexID **candidate = candidates[nID];
    ui *candCount = candCounts[nID];
    ui *pos = poses[nID];
    VertexID *patternV = patternVs[nID];
    VertexID *dataV = dataVs[nID];
    EdgeID *startOffset = startOffsets[nID];
    bool *visited = visits[nID];
    ui *keyPos = keyPoses[nID];
    ui &keyPosSize = keyPosSizes[nID];
    ui sizeBound = sizeBounds[nID];
    bool isRoot = nID == t.getRootID();
    ui n = dun.getNumVertices(), m = dun.getNumEdges();
    if (tau.prefixSize != 0) {
        for (int i = 0; i < nodesAtStep[0].size(); ++i) {
            VertexID cID = nodesAtStep[0][i];
            for (int j = 0; j < prefixPos[cID].size(); ++j) {
                patternVs[cID][j] = patternV[prefixPos[cID][j]];
                dataVs[cID][j] = dataV[prefixPos[cID][j]];
                startOffsets[cID][j] = startOffset[prefixPos[cID][j]];
                poses[cID][j] = pos[prefixPos[cID][j]];
                visits[cID][dataVs[cID][j]] = true;
            }
            if (t.getNode(cID).keySize == 0) H[cID][0] = 0;
            else {
                if (keyPosSizes[cID] < n / 8 + 1) {
                    for (int j = 0; j < keyPosSizes[cID]; ++j) {
                        H[cID][keyPoses[cID][j]] = 0;
                    }
                }
                else memset(H[cID], 0, sizeof(Count) * n);
            }
            multiJoinWrapper(cID, t, t.getChild()[cID], candidates, candCounts, H, din, dout, dun, true, tri, p, outID, unID,
                             reverseID, startOffsets, patternVs, dataVs, visits, poses, keyPoses, keyPosSizes, sizeBounds, tmp, allV);
            for (int j = 0; j < prefixPos[cID].size(); ++j) {
                visits[cID][dataVs[cID][j]] = false;
            }
        }
    }
    int mappingSize = tau.prefixSize;
    if (mappingSize == 0) {
        candidate[0] = allV;
        candCount[0] = n;
        pos[0] = 0;
    }
    else {
        if (!nodeInterPos[mappingSize]) {
            if (!nodeOutPos[mappingSize].empty()) {
                VertexID v = dataV[nodeOutPos[mappingSize][0]];
                startOffset[mappingSize] = outOffset[v];
                candidate[mappingSize] = outNbors + outOffset[v];
                candCount[mappingSize] = outOffset[v + 1] - outOffset[v];
            }
            else if (!nodeInPos[mappingSize].empty()){
                VertexID v = dataV[nodeInPos[mappingSize][0]];
                startOffset[mappingSize] = inOffset[v];
                candidate[mappingSize] = inNbors + inOffset[v];
                candCount[mappingSize] = inOffset[v + 1] - inOffset[v];
            }
            else {
                VertexID v = dataV[nodeUnPos[mappingSize][0]];
                startOffset[mappingSize] = unOffset[v];
                candidate[mappingSize] = unNbors + unOffset[v];
                candCount[mappingSize] = unOffset[v + 1] - unOffset[v];
            }
        }
        else {
            EdgeID e = 0;
            if (triEdgeType[mappingSize] != 0) {
                int pos1 = nodeTriPos[mappingSize].first, pos2 = nodeTriPos[mappingSize].second;
                if (triEdgeType[mappingSize] == 1)
                    e = startOffset[pos2] + pos[pos2] - 1;
                else if (triEdgeType[mappingSize] == 2)
                    e = outID[startOffset[pos2] + pos[pos2] - 1];
                else if (triEdgeType[mappingSize] == 5)
                    e = unID[startOffset[pos2] + pos[pos2] - 1];
                else {
                    VertexID key1 = dataV[pos1], key2 = dataV[pos2];
#ifdef COLLECT_STATISTICS
                    ++gNumEdgeID;
#endif
                    e = dout.getEdgeID(key1, key2);
                }
            }
            generateCandidateT(din, dout, dun, tri, dataV, mappingSize, !nodeCandPos[mappingSize], e, triEndType[mappingSize],
                               nodeInPos[mappingSize], nodeOutPos[mappingSize], nodeUnPos[mappingSize], candidate, candCount, tmp);
        }
        pos[mappingSize] = 0;
        // apply the symmetry breaking rules
        if (!greaterPos[mappingSize].empty()) {
            ui maxTarget = dataV[greaterPos[mappingSize][0]];
            for (int i = 1; i < greaterPos[mappingSize].size(); ++i) {
                if (dataV[greaterPos[mappingSize][i]] > maxTarget)
                    maxTarget = dataV[greaterPos[mappingSize][i]];
            }
            pos[mappingSize] = firstPosGreaterThan(candidate[mappingSize], 0, candCount[mappingSize], maxTarget);
        }
        if (!lessPos[mappingSize].empty()) {
            ui minTarget = dataV[lessPos[mappingSize][0]];
            for (int i = 1; i < lessPos[mappingSize].size(); ++i) {
                if (dataV[lessPos[mappingSize][i]] < minTarget)
                    minTarget = dataV[lessPos[mappingSize][i]];
            }
            candCount[mappingSize] = firstPosGreaterThan(candidate[mappingSize], 0, candCount[mappingSize], minTarget);
        }
    }
    pos[0] = 0;
    while (mappingSize >= (int)tau.prefixSize) {
        while (pos[mappingSize] < candCount[mappingSize]) {
            VertexID v = candidate[mappingSize][pos[mappingSize]];
            ++pos[mappingSize];
            if (visited[v]) continue;
            visited[v] = true;
            patternV[mappingSize] = tau.nodeOrder[mappingSize];
            dataV[mappingSize] = v;
            for (int i = 0; i < nodesAtStep[mappingSize].size(); ++i) {
                VertexID cID = nodesAtStep[mappingSize][i];
                for (int j = 0; j < prefixPos[cID].size(); ++j) {
                    patternVs[cID][j] = patternV[prefixPos[cID][j]];
                    dataVs[cID][j] = dataV[prefixPos[cID][j]];
                    startOffsets[cID][j] = startOffset[prefixPos[cID][j]];
                    poses[cID][j] = pos[prefixPos[cID][j]];
                    visits[cID][dataVs[cID][j]] = true;
                }
                if (t.getNode(cID).keySize == 0) H[cID][0] = 0;
                else if (t.getNode(cID).keySize == 1) {
                    if (keyPosSizes[cID] < n / 8 + 1) {
                        for (int j = 0; j < keyPosSizes[cID]; ++j) {
                            H[cID][keyPoses[cID][j]] = 0;
                        }
                    }
                    else memset(H[cID], 0, sizeof(Count) * n);
                }
                else {
                    if (keyPosSizes[cID] < m / 8 + 1) {
                        for (int j = 0; j < keyPosSizes[cID]; ++j) {
                            H[cID][keyPoses[cID][j]] = 0;
                        }
                    }
                    else memset(H[cID], 0, sizeof(Count) * m);
                }
                multiJoinWrapper(cID, t, t.getChild()[cID], candidates, candCounts, H, din, dout, dun, true, tri, p, outID, unID,
                                 reverseID, startOffsets, patternVs, dataVs, visits, poses, keyPoses, keyPosSizes, sizeBounds, tmp, allV);
                for (int j = 0; j < prefixPos[cID].size(); ++j) {
                    visits[cID][dataVs[cID][j]] = false;
                }
            }
            if (mappingSize == tau.numVertices - 1) {
#ifdef COLLECT_STATISTICS
                ++gNumMatch;
#endif
                Count cnt = 1;
                // multiply count in children
                for (int j = 0; j < child.size(); ++j) {
                    VertexID cID = child[j];
                    if (childKeyPos[j].empty()) cnt *= H[cID][0];
                    else cnt *= H[cID][dataV[childKeyPos[j][0]]];
                }
                if (isRoot) {
                    if (orbitType == 0) h[0] += cnt * aggreWeight[0];
                    else {
                        for (int j = 0; j < aggreV.size(); ++j) {
                            VertexID key = dataV[aggrePos[j]];
#ifdef COLLECT_STATISTICS
                            ++gNumUpdate;
#endif
                            h[key] += cnt * aggreWeight[j];
                        }
                    }
                }
                else {
                    if (tau.keySize == 0) h[0] += cnt;
                    else {
#ifdef COLLECT_STATISTICS
                        ++gNumUpdate;
#endif
                        h[dataV[aggrePos[0]]] += cnt;
                        if (keyPosSize < sizeBound) {
                            keyPos[keyPosSize] = dataV[aggrePos[0]];
                            ++keyPosSize;
                        }
                    }
                }
                visited[dataV[mappingSize]] = false;
            }
            else {
#ifdef COLLECT_STATISTICS
                ++gNumIntermediate;
#endif
                ++mappingSize;
                pos[mappingSize] = 0;
                if (!nodeInterPos[mappingSize]) {
                    if (!nodeOutPos[mappingSize].empty()) {
                        VertexID w = dataV[nodeOutPos[mappingSize][0]];
                        startOffset[mappingSize] = outOffset[w];
                        candidate[mappingSize] = outNbors + outOffset[w];
                        candCount[mappingSize] = outOffset[w + 1] - outOffset[w];
                    }
                    else if (!nodeInPos[mappingSize].empty()){
                        VertexID w = dataV[nodeInPos[mappingSize][0]];
                        startOffset[mappingSize] = inOffset[w];
                        candidate[mappingSize] = inNbors + inOffset[w];
                        candCount[mappingSize] = inOffset[w + 1] - inOffset[w];
                    }
                    else {
                        VertexID w = dataV[nodeUnPos[mappingSize][0]];
                        startOffset[mappingSize] = unOffset[w];
                        candidate[mappingSize] = unNbors + unOffset[w];
                        candCount[mappingSize] = unOffset[w + 1] - unOffset[w];
                    }
                }
                else {
                    EdgeID e = 0;
                    if (triEdgeType[mappingSize] != 0) {
                        int pos1 = nodeTriPos[mappingSize].first, pos2 = nodeTriPos[mappingSize].second;
                        if (triEdgeType[mappingSize] == 1)
                            e = startOffset[pos2] + pos[pos2] - 1;
                        else if (triEdgeType[mappingSize] == 2)
                            e = outID[startOffset[pos2] + pos[pos2] - 1];
                        else if (triEdgeType[mappingSize] == 5)
                            e = unID[startOffset[pos2] + pos[pos2] - 1];
                        else {
                            VertexID key1 = dataV[pos1], key2 = dataV[pos2];
#ifdef COLLECT_STATISTICS
                            ++gNumEdgeID;
#endif
                            e = dout.getEdgeID(key1, key2);
                        }
                    }
                    generateCandidateT(din, dout, dun, tri, dataV, mappingSize, !nodeCandPos[mappingSize], e, triEndType[mappingSize],
                                       nodeInPos[mappingSize], nodeOutPos[mappingSize], nodeUnPos[mappingSize], candidate, candCount, tmp);
                }
                // apply the symmetry breaking rules
                if (!greaterPos[mappingSize].empty()) {
                    ui maxTarget = dataV[greaterPos[mappingSize][0]];
                    for (int i = 1; i < greaterPos[mappingSize].size(); ++i) {
                        if (dataV[greaterPos[mappingSize][i]] > maxTarget)
                            maxTarget = dataV[greaterPos[mappingSize][i]];
                    }
                    pos[mappingSize] = firstPosGreaterThan(candidate[mappingSize], 0, candCount[mappingSize], maxTarget);
                }
                if (!lessPos[mappingSize].empty()) {
                    ui minTarget = dataV[lessPos[mappingSize][0]];
                    for (int i = 1; i < lessPos[mappingSize].size(); ++i) {
                        if (dataV[lessPos[mappingSize][i]] < minTarget)
                            minTarget = dataV[lessPos[mappingSize][i]];
                    }
                    candCount[mappingSize] = firstPosGreaterThan(candidate[mappingSize], 0, candCount[mappingSize], minTarget);
                }
            }
        }
        --mappingSize;
        if (mappingSize >= (int)tau.prefixSize) {
            visited[dataV[mappingSize]] = false;
        }
    }
}

void multiJoinE(
        VertexID nID,
        const Tree &t,
        const std::vector<VertexID> &child,
        VertexID ***candidates,
        ui **candCounts,
        HashTable *H,
        const DataGraph &din,
        const DataGraph &dout,
        const DataGraph &dun,
        const Triangle &tri,
        const Pattern &p,
        EdgeID *outID,
        EdgeID *unID,
        EdgeID *reverseID,
        EdgeID **startOffsets,
        VertexID **patternVs,
        VertexID **dataVs,
        bool **visits,
        ui **poses,
        ui **keyPoses,
        ui *keyPosSizes,
        ui *sizeBounds,
        VertexID *&tmp,
        VertexID *allV
) {
    int orbitType = t.getOrbitType();
    HashTable h = H[nID];
    const Node &tau = t.getNode(nID);
    const std::vector<int> &aggrePos = t.getAggrePos(nID);
    const std::vector<bool> &nodeInterPos = t.getNodeInterPos(nID);
    const std::vector<bool> &nodeCandPos = t.getNodeCandPos(nID);
    const std::vector<std::vector<int>> &nodeInPos = t.getNodeInPos(nID);
    const std::vector<std::vector<int>> &nodeOutPos = t.getNodeOutPos(nID);
    const std::vector<std::vector<int>> &nodeUnPos = t.getNodeUnPos(nID);
    const std::vector<std::vector<int>> &greaterPos = t.getNodeGreaterPos(nID);
    const std::vector<std::vector<int>> &lessPos = t.getNodeLessPos(nID);
    const std::vector<std::vector<int>> &childKeyPos = t.getChildKeyPos(nID);
    const std::vector<VertexID> &aggreV = t.getAggreV();
    const std::vector<int> &aggreWeight = t.getAggreWeight();
    const std::vector<std::vector<int>> &posChildEdge = t.getPosChildEdge(nID);
    const std::vector<std::vector<int>> &posAggreEdge = t.getPosAggreEdge(nID);
    const std::vector<int> &childEdgeType = t.getChildEdgeType(nID);
    const std::vector<int> &aggreEdgeType = t.getAggreEdgeType(nID);
    const std::vector<std::vector<VertexID>> &nodesAtStep = t.getNodesAtStep()[nID];
    const std::vector<std::vector<int>> &prefixPos = t.getPrefixPos();
    EdgeID *inOffset = din.getOffsets();
    VertexID *inNbors = din.getNbors();
    EdgeID *outOffset = dout.getOffsets();
    VertexID *outNbors = dout.getNbors();
    EdgeID *unOffset = dun.getOffsets();
    VertexID *unNbors = dun.getNbors();
    VertexID **candidate = candidates[nID];
    ui *candCount = candCounts[nID];
    ui *pos = poses[nID];
    VertexID *patternV = patternVs[nID];
    VertexID *dataV = dataVs[nID];
    EdgeID *startOffset = startOffsets[nID];
    bool *visited = visits[nID];
    ui *keyPos = keyPoses[nID];
    ui &keyPosSize = keyPosSizes[nID];
    ui sizeBound = sizeBounds[nID];
    bool isRoot = nID == t.getRootID();
    std::vector<EdgeID> childKey(child.size());
    std::vector<EdgeID> aggreKey;
    if (isRoot && orbitType == 2) aggreKey = std::vector<EdgeID>(aggreWeight.size());
    else if (tau.keySize == 2) aggreKey.push_back(0);
    ui n = dun.getNumVertices(), m = dun.getNumEdges();
    if (tau.prefixSize != 0) {
        for (int i = 0; i < nodesAtStep[0].size(); ++i) {
            VertexID cID = nodesAtStep[0][i];
            for (int j = 0; j < prefixPos[cID].size(); ++j) {
                patternVs[cID][j] = patternV[prefixPos[cID][j]];
                dataVs[cID][j] = dataV[prefixPos[cID][j]];
                startOffsets[cID][j] = startOffset[prefixPos[cID][j]];
                poses[cID][j] = pos[prefixPos[cID][j]];
                visits[cID][dataVs[cID][j]] = true;
            }
            if (t.getNode(cID).keySize == 0) H[cID][0] = 0;
            else if (t.getNode(cID).keySize == 1) {
                if (keyPosSizes[cID] < n / 8 + 1) {
                    for (int j = 0; j < keyPosSizes[cID]; ++j) {
                        H[cID][keyPoses[cID][j]] = 0;
                    }
                }
                else memset(H[cID], 0, sizeof(Count) * n);
            }
            else {
                if (keyPosSizes[cID] < m / 8 + 1) {
                    for (int j = 0; j < keyPosSizes[cID]; ++j) {
                        H[cID][keyPoses[cID][j]] = 0;
                    }
                }
                else memset(H[cID], 0, sizeof(Count) * m);
            }
            multiJoinWrapper(cID, t, t.getChild()[cID], candidates, candCounts, H, din, dout, dun, true, tri, p, outID, unID,
                             reverseID, startOffsets, patternVs, dataVs, visits, poses, keyPoses, keyPosSizes, sizeBounds, tmp, allV);
            for (int j = 0; j < prefixPos[cID].size(); ++j) {
                visits[cID][dataVs[cID][j]] = false;
            }
        }
    }
    int mappingSize = tau.prefixSize;
    for (int i = 1; i < mappingSize; ++i) {
        // for child edge keys, compute the key value
        for (int j: posChildEdge[i]) {
            childKey[j] = computeEdgeKey(din, dout, dun, outID, unID, reverseID, startOffset, i, pos,
                                         childEdgeType[j], dataV[childKeyPos[j][0]], dataV[childKeyPos[j][1]]);
        }
        // for aggregation edge keys, compute the key value
        for (int j: posAggreEdge[i]) {
            aggreKey[j] = computeEdgeKey(din, dout, dun, outID, unID, reverseID, startOffset, i, pos,
                                         aggreEdgeType[j], dataV[aggrePos[2 * j]], dataV[aggrePos[2 * j + 1]]);
        }
    }
    if (mappingSize == 0) {
        candidate[0] = allV;
        candCount[0] = n;
        pos[0] = 0;
    }
    else {
        if (!nodeInterPos[mappingSize]) {
            if (!nodeOutPos[mappingSize].empty()) {
                VertexID v = dataV[nodeOutPos[mappingSize][0]];
                startOffset[mappingSize] = outOffset[v];
                candidate[mappingSize] = outNbors + outOffset[v];
                candCount[mappingSize] = outOffset[v + 1] - outOffset[v];
            }
            else if (!nodeInPos[mappingSize].empty()){
                VertexID v = dataV[nodeInPos[mappingSize][0]];
                startOffset[mappingSize] = inOffset[v];
                candidate[mappingSize] = inNbors + inOffset[v];
                candCount[mappingSize] = inOffset[v + 1] - inOffset[v];
            }
            else {
                VertexID v = dataV[nodeUnPos[mappingSize][0]];
                startOffset[mappingSize] = unOffset[v];
                candidate[mappingSize] = unNbors + unOffset[v];
                candCount[mappingSize] = unOffset[v + 1] - unOffset[v];
            }
        }
        else {
            generateCandidate(din, dout, dun, dataV, mappingSize, nodeInPos[mappingSize],
                              nodeOutPos[mappingSize], nodeUnPos[mappingSize], candidate, candCount, tmp);
        }
        pos[mappingSize] = 0;
        // apply the symmetry breaking rules
        if (!greaterPos[mappingSize].empty()) {
            ui maxTarget = dataV[greaterPos[mappingSize][0]];
            for (int i = 1; i < greaterPos[mappingSize].size(); ++i) {
                if (dataV[greaterPos[mappingSize][i]] > maxTarget)
                    maxTarget = dataV[greaterPos[mappingSize][i]];
            }
            pos[mappingSize] = firstPosGreaterThan(candidate[mappingSize], 0, candCount[mappingSize], maxTarget);
        }
        if (!lessPos[mappingSize].empty()) {
            ui minTarget = dataV[lessPos[mappingSize][0]];
            for (int i = 1; i < lessPos[mappingSize].size(); ++i) {
                if (dataV[lessPos[mappingSize][i]] < minTarget)
                    minTarget = dataV[lessPos[mappingSize][i]];
            }
            candCount[mappingSize] = firstPosGreaterThan(candidate[mappingSize], 0, candCount[mappingSize], minTarget);
        }
    }
    pos[0] = 0;
    while (mappingSize >= (int)tau.prefixSize) {
        while (pos[mappingSize] < candCount[mappingSize]) {
            VertexID v = candidate[mappingSize][pos[mappingSize]];
            ++pos[mappingSize];
            if (visited[v]) continue;
            visited[v] = true;
            patternV[mappingSize] = tau.nodeOrder[mappingSize];
            dataV[mappingSize] = v;
            // for child edge keys, compute the key value
            for (int j: posChildEdge[mappingSize]) {
                childKey[j] = computeEdgeKey(din, dout, dun, outID, unID, reverseID, startOffset, mappingSize, pos,
                                             childEdgeType[j], dataV[childKeyPos[j][0]], dataV[childKeyPos[j][1]]);
            }
            // for aggregation edge keys, compute the key value
            for (int j: posAggreEdge[mappingSize]) {
                aggreKey[j] = computeEdgeKey(din, dout, dun, outID, unID, reverseID, startOffset, mappingSize, pos,
                                             aggreEdgeType[j], dataV[aggrePos[2 * j]], dataV[aggrePos[2 * j + 1]]);
            }
            for (int i = 0; i < nodesAtStep[mappingSize].size(); ++i) {
                VertexID cID = nodesAtStep[mappingSize][i];
                for (int j = 0; j < prefixPos[cID].size(); ++j) {
                    patternVs[cID][j] = patternV[prefixPos[cID][j]];
                    dataVs[cID][j] = dataV[prefixPos[cID][j]];
                    startOffsets[cID][j] = startOffset[prefixPos[cID][j]];
                    poses[cID][j] = pos[prefixPos[cID][j]];
                    visits[cID][dataVs[cID][j]] = true;
                }
                if (t.getNode(cID).keySize == 0) H[cID][0] = 0;
                else if (t.getNode(cID).keySize == 1) {
                    if (keyPosSizes[cID] < n / 8 + 1) {
                        for (int j = 0; j < keyPosSizes[cID]; ++j) {
                            H[cID][keyPoses[cID][j]] = 0;
                        }
                    }
                    else memset(H[cID], 0, sizeof(Count) * n);
                }
                else {
                    if (keyPosSizes[cID] < m / 8 + 1) {
                        for (int j = 0; j < keyPosSizes[cID]; ++j) {
                            H[cID][keyPoses[cID][j]] = 0;
                        }
                    }
                    else memset(H[cID], 0, sizeof(Count) * m);
                }
                multiJoinWrapper(cID, t, t.getChild()[cID], candidates, candCounts, H, din, dout, dun, true, tri, p, outID, unID,
                                 reverseID, startOffsets, patternVs, dataVs, visits, poses, keyPoses, keyPosSizes, sizeBounds, tmp, allV);
                for (int j = 0; j < prefixPos[cID].size(); ++j) {
                    visits[cID][dataVs[cID][j]] = false;
                }
            }
            if (mappingSize == tau.numVertices - 1) {
#ifdef COLLECT_STATISTICS
                ++gNumMatch;
#endif
                Count cnt = 1;
                // multiply count in children
                for (int j = 0; j < child.size(); ++j) {
                    VertexID cID = child[j];
                    if (childKeyPos[j].empty()) cnt *= H[cID][0];
                    else if (childKeyPos[j].size() == 1) {
                        cnt *= H[cID][dataV[childKeyPos[j][0]]];
                    }
                    else {
                        cnt *= H[cID][childKey[j]];
                    }
                }
                if (isRoot) {
                    if (orbitType == 0) h[0] += cnt * aggreWeight[0];
                    else if (orbitType == 1) {
                        for (int j = 0; j < aggreV.size(); ++j) {
                            VertexID key = dataV[aggrePos[j]];
#ifdef COLLECT_STATISTICS
                            ++gNumUpdate;
#endif
                            h[key] += cnt * aggreWeight[j];
                        }
                    }
                    else {
                        for (int j = 0; j < aggreKey.size(); ++j) {
#ifdef COLLECT_STATISTICS
                            ++gNumUpdate;
#endif
                            h[aggreKey[j]] += cnt * aggreWeight[j];
                        }
                    }
                }
                else {
                    if (tau.keySize == 0) h[0] += cnt;
                    else if (tau.keySize == 1) {
#ifdef COLLECT_STATISTICS
                        ++gNumUpdate;
#endif
                        h[dataV[aggrePos[0]]] += cnt;
                        if (keyPosSize < sizeBound) {
                            keyPos[keyPosSize] = dataV[aggrePos[0]];
                            ++keyPosSize;
                        }
                    }
                    else {
                        EdgeID e = aggreKey[0];
#ifdef COLLECT_STATISTICS
                        ++gNumUpdate;
#endif
                        h[e] += cnt;
                        if (keyPosSize < sizeBound) {
                            keyPos[keyPosSize] = e;
                            ++keyPosSize;
                        }
                    }
                }
                visited[dataV[mappingSize]] = false;
            }
            else {
#ifdef COLLECT_STATISTICS
                ++gNumIntermediate;
#endif
                ++mappingSize;
                pos[mappingSize] = 0;
                if (!nodeInterPos[mappingSize]) {
                    if (!nodeOutPos[mappingSize].empty()) {
                        VertexID w = dataV[nodeOutPos[mappingSize][0]];
                        startOffset[mappingSize] = outOffset[w];
                        candidate[mappingSize] = outNbors + outOffset[w];
                        candCount[mappingSize] = outOffset[w + 1] - outOffset[w];
                    }
                    else if (!nodeInPos[mappingSize].empty()){
                        VertexID w = dataV[nodeInPos[mappingSize][0]];
                        startOffset[mappingSize] = inOffset[w];
                        candidate[mappingSize] = inNbors + inOffset[w];
                        candCount[mappingSize] = inOffset[w + 1] - inOffset[w];
                    }
                    else {
                        VertexID w = dataV[nodeUnPos[mappingSize][0]];
                        startOffset[mappingSize] = unOffset[w];
                        candidate[mappingSize] = unNbors + unOffset[w];
                        candCount[mappingSize] = unOffset[w + 1] - unOffset[w];
                    }
                }
                else {
                    generateCandidate(din, dout, dun, dataV, mappingSize, nodeInPos[mappingSize],
                                      nodeOutPos[mappingSize], nodeUnPos[mappingSize], candidate, candCount, tmp);
                }
                // apply the symmetry breaking rules
                if (!greaterPos[mappingSize].empty()) {
                    ui maxTarget = dataV[greaterPos[mappingSize][0]];
                    for (int i = 1; i < greaterPos[mappingSize].size(); ++i) {
                        if (dataV[greaterPos[mappingSize][i]] > maxTarget)
                            maxTarget = dataV[greaterPos[mappingSize][i]];
                    }
                    pos[mappingSize] = firstPosGreaterThan(candidate[mappingSize], 0, candCount[mappingSize], maxTarget);
                }
                if (!lessPos[mappingSize].empty()) {
                    ui minTarget = dataV[lessPos[mappingSize][0]];
                    for (int i = 1; i < lessPos[mappingSize].size(); ++i) {
                        if (dataV[lessPos[mappingSize][i]] < minTarget)
                            minTarget = dataV[lessPos[mappingSize][i]];
                    }
                    candCount[mappingSize] = firstPosGreaterThan(candidate[mappingSize], 0, candCount[mappingSize], minTarget);
                }
            }
        }
        --mappingSize;
        if (mappingSize >= (int)tau.prefixSize) {
            visited[dataV[mappingSize]] = false;
        }
    }
}

void multiJoinET(
        VertexID nID,
        const Tree &t,
        const std::vector<VertexID> &child,
        VertexID ***candidates,
        ui **candCounts,
        HashTable *H,
        const DataGraph &din,
        const DataGraph &dout,
        const DataGraph &dun,
        const Triangle &tri,
        const Pattern &p,
        EdgeID *outID,
        EdgeID *unID,
        EdgeID *reverseID,
        EdgeID **startOffsets,
        VertexID **patternVs,
        VertexID **dataVs,
        bool **visits,
        ui **poses,
        ui **keyPoses,
        ui *keyPosSizes,
        ui *sizeBounds,
        VertexID *&tmp,
        VertexID *allV
) {
    int orbitType = t.getOrbitType();
    HashTable h = H[nID];
    const Node &tau = t.getNode(nID);
    const std::vector<int> &aggrePos = t.getAggrePos(nID);
    const std::vector<bool> &nodeInterPos = t.getNodeInterPos(nID);
    const std::vector<bool> &nodeCandPos = t.getNodeCandPos(nID);
    const std::vector<std::vector<int>> &nodeInPos = t.getNodeInPos(nID);
    const std::vector<std::vector<int>> &nodeOutPos = t.getNodeOutPos(nID);
    const std::vector<std::vector<int>> &nodeUnPos = t.getNodeUnPos(nID);
    const std::vector<std::vector<int>> &greaterPos = t.getNodeGreaterPos(nID);
    const std::vector<std::vector<int>> &lessPos = t.getNodeLessPos(nID);
    const std::vector<std::vector<int>> &childKeyPos = t.getChildKeyPos(nID);
    const std::vector<VertexID> &aggreV = t.getAggreV();
    const std::vector<int> &aggreWeight = t.getAggreWeight();
    const std::vector<std::vector<int>> &posChildEdge = t.getPosChildEdge(nID);
    const std::vector<std::vector<int>> &posAggreEdge = t.getPosAggreEdge(nID);
    const std::vector<int> &childEdgeType = t.getChildEdgeType(nID);
    const std::vector<int> &aggreEdgeType = t.getAggreEdgeType(nID);
    const std::vector<std::pair<int, int>> &nodeTriPos = t.getNodeTriPos(nID);
    const std::vector<int> &triEdgeType = t.getTriEdgeType(nID);
    const std::vector<int> &triEndType = t.getTriEndType(nID);
    const std::vector<std::vector<VertexID>> &nodesAtStep = t.getNodesAtStep()[nID];
    const std::vector<std::vector<int>> &prefixPos = t.getPrefixPos();
    EdgeID *inOffset = din.getOffsets();
    VertexID *inNbors = din.getNbors();
    EdgeID *outOffset = dout.getOffsets();
    VertexID *outNbors = dout.getNbors();
    EdgeID *unOffset = dun.getOffsets();
    VertexID *unNbors = dun.getNbors();
    VertexID **candidate = candidates[nID];
    ui *candCount = candCounts[nID];
    ui *pos = poses[nID];
    VertexID *patternV = patternVs[nID];
    VertexID *dataV = dataVs[nID];
    EdgeID *startOffset = startOffsets[nID];
    bool *visited = visits[nID];
    ui *keyPos = keyPoses[nID];
    ui &keyPosSize = keyPosSizes[nID];
    ui sizeBound = sizeBounds[nID];
    bool isRoot = nID == t.getRootID();
    std::vector<EdgeID> childKey(child.size());
    std::vector<EdgeID> aggreKey;
    if (isRoot && orbitType == 2) aggreKey = std::vector<EdgeID>(aggreWeight.size());
    else if (tau.keySize == 2) aggreKey.push_back(0);
    ui n = dun.getNumVertices(), m = dun.getNumEdges();
    if (tau.prefixSize != 0) {
        for (int i = 0; i < nodesAtStep[0].size(); ++i) {
            VertexID cID = nodesAtStep[0][i];
            for (int j = 0; j < prefixPos[cID].size(); ++j) {
                patternVs[cID][j] = patternV[prefixPos[cID][j]];
                dataVs[cID][j] = dataV[prefixPos[cID][j]];
                startOffsets[cID][j] = startOffset[prefixPos[cID][j]];
                poses[cID][j] = pos[prefixPos[cID][j]];
                visits[cID][dataVs[cID][j]] = true;
            }
            if (t.getNode(cID).keySize == 0) H[cID][0] = 0;
            else if (t.getNode(cID).keySize == 1) {
                if (keyPosSizes[cID] < n / 8 + 1) {
                    for (int j = 0; j < keyPosSizes[cID]; ++j) {
                        H[cID][keyPoses[cID][j]] = 0;
                    }
                }
                else memset(H[cID], 0, sizeof(Count) * n);
            }
            else {
                if (keyPosSizes[cID] < m / 8 + 1) {
                    for (int j = 0; j < keyPosSizes[cID]; ++j) {
                        H[cID][keyPoses[cID][j]] = 0;
                    }
                }
                else memset(H[cID], 0, sizeof(Count) * m);
            }
            multiJoinWrapper(cID, t, t.getChild()[cID], candidates, candCounts, H, din, dout, dun, true, tri, p, outID, unID,
                             reverseID, startOffsets, patternVs, dataVs, visits, poses, keyPoses, keyPosSizes, sizeBounds, tmp, allV);
            for (int j = 0; j < prefixPos[cID].size(); ++j) {
                visits[cID][dataVs[cID][j]] = false;
            }
        }
    }
    int mappingSize = tau.prefixSize;
    for (int i = 1; i < mappingSize; ++i) {
        // for child edge keys, compute the key value
        for (int j: posChildEdge[i]) {
            childKey[j] = computeEdgeKey(din, dout, dun, outID, unID, reverseID, startOffset, i, pos,
                                         childEdgeType[j], dataV[childKeyPos[j][0]], dataV[childKeyPos[j][1]]);
        }
        // for aggregation edge keys, compute the key value
        for (int j: posAggreEdge[i]) {
            aggreKey[j] = computeEdgeKey(din, dout, dun, outID, unID, reverseID, startOffset, i, pos,
                                         aggreEdgeType[j], dataV[aggrePos[2 * j]], dataV[aggrePos[2 * j + 1]]);
        }
    }
    if (mappingSize == 0) {
        candidate[0] = allV;
        candCount[0] = n;
        pos[0] = 0;
    }
    else {
        if (!nodeInterPos[mappingSize]) {
            if (!nodeOutPos[mappingSize].empty()) {
                VertexID v = dataV[nodeOutPos[mappingSize][0]];
                startOffset[mappingSize] = outOffset[v];
                candidate[mappingSize] = outNbors + outOffset[v];
                candCount[mappingSize] = outOffset[v + 1] - outOffset[v];
            }
            else if (!nodeInPos[mappingSize].empty()){
                VertexID v = dataV[nodeInPos[mappingSize][0]];
                startOffset[mappingSize] = inOffset[v];
                candidate[mappingSize] = inNbors + inOffset[v];
                candCount[mappingSize] = inOffset[v + 1] - inOffset[v];
            }
            else {
                VertexID v = dataV[nodeUnPos[mappingSize][0]];
                startOffset[mappingSize] = unOffset[v];
                candidate[mappingSize] = unNbors + unOffset[v];
                candCount[mappingSize] = unOffset[v + 1] - unOffset[v];
            }
        }
        else {
            EdgeID e = 0;
            if (triEdgeType[mappingSize] != 0) {
                int pos1 = nodeTriPos[mappingSize].first, pos2 = nodeTriPos[mappingSize].second;
                if (triEdgeType[mappingSize] == 1)
                    e = startOffset[pos2] + pos[pos2] - 1;
                else if (triEdgeType[mappingSize] == 2)
                    e = outID[startOffset[pos2] + pos[pos2] - 1];
                else if (triEdgeType[mappingSize] == 5)
                    e = unID[startOffset[pos2] + pos[pos2] - 1];
                else {
                    VertexID key1 = dataV[pos1], key2 = dataV[pos2];
#ifdef COLLECT_STATISTICS
                    ++gNumEdgeID;
#endif
                    e = dout.getEdgeID(key1, key2);
                }
            }
            generateCandidateT(din, dout, dun, tri, dataV, mappingSize, !nodeCandPos[mappingSize], e, triEndType[mappingSize],
                               nodeInPos[mappingSize], nodeOutPos[mappingSize], nodeUnPos[mappingSize], candidate, candCount, tmp);
        }
        pos[mappingSize] = 0;
        // apply the symmetry breaking rules
        if (!greaterPos[mappingSize].empty()) {
            ui maxTarget = dataV[greaterPos[mappingSize][0]];
            for (int i = 1; i < greaterPos[mappingSize].size(); ++i) {
                if (dataV[greaterPos[mappingSize][i]] > maxTarget)
                    maxTarget = dataV[greaterPos[mappingSize][i]];
            }
            pos[mappingSize] = firstPosGreaterThan(candidate[mappingSize], 0, candCount[mappingSize], maxTarget);
        }
        if (!lessPos[mappingSize].empty()) {
            ui minTarget = dataV[lessPos[mappingSize][0]];
            for (int i = 1; i < lessPos[mappingSize].size(); ++i) {
                if (dataV[lessPos[mappingSize][i]] < minTarget)
                    minTarget = dataV[lessPos[mappingSize][i]];
            }
            candCount[mappingSize] = firstPosGreaterThan(candidate[mappingSize], 0, candCount[mappingSize], minTarget);
        }
    }
    pos[0] = 0;
    while (mappingSize >= (int)tau.prefixSize) {
        while (pos[mappingSize] < candCount[mappingSize]) {
            VertexID v = candidate[mappingSize][pos[mappingSize]];
            ++pos[mappingSize];
            if (visited[v]) continue;
            visited[v] = true;
            patternV[mappingSize] = tau.nodeOrder[mappingSize];
            dataV[mappingSize] = v;
            // for child edge keys, compute the key value
            for (int j: posChildEdge[mappingSize]) {
                childKey[j] = computeEdgeKey(din, dout, dun, outID, unID, reverseID, startOffset, mappingSize, pos,
                                             childEdgeType[j], dataV[childKeyPos[j][0]], dataV[childKeyPos[j][1]]);
            }
            // for aggregation edge keys, compute the key value
            for (int j: posAggreEdge[mappingSize]) {
                aggreKey[j] = computeEdgeKey(din, dout, dun, outID, unID, reverseID, startOffset, mappingSize, pos,
                                             aggreEdgeType[j], dataV[aggrePos[2 * j]], dataV[aggrePos[2 * j + 1]]);
            }
            for (int i = 0; i < nodesAtStep[mappingSize].size(); ++i) {
                VertexID cID = nodesAtStep[mappingSize][i];
                for (int j = 0; j < prefixPos[cID].size(); ++j) {
                    patternVs[cID][j] = patternV[prefixPos[cID][j]];
                    dataVs[cID][j] = dataV[prefixPos[cID][j]];
                    startOffsets[cID][j] = startOffset[prefixPos[cID][j]];
                    poses[cID][j] = pos[prefixPos[cID][j]];
                    visits[cID][dataVs[cID][j]] = true;
                }
                if (t.getNode(cID).keySize == 0) H[cID][0] = 0;
                else if (t.getNode(cID).keySize == 1) {
                    if (keyPosSizes[cID] < n / 8 + 1) {
                        for (int j = 0; j < keyPosSizes[cID]; ++j) {
                            H[cID][keyPoses[cID][j]] = 0;
                        }
                    }
                    else memset(H[cID], 0, sizeof(Count) * n);
                }
                else {
                    if (keyPosSizes[cID] < m / 8 + 1) {
                        for (int j = 0; j < keyPosSizes[cID]; ++j) {
                            H[cID][keyPoses[cID][j]] = 0;
                        }
                    }
                    else memset(H[cID], 0, sizeof(Count) * m);
                }
                multiJoinWrapper(cID, t, t.getChild()[cID], candidates, candCounts, H, din, dout, dun, true, tri, p, outID, unID,
                                 reverseID, startOffsets, patternVs, dataVs, visits, poses, keyPoses, keyPosSizes, sizeBounds, tmp, allV);
                for (int j = 0; j < prefixPos[cID].size(); ++j) {
                    visits[cID][dataVs[cID][j]] = false;
                }
            }
            if (mappingSize == tau.numVertices - 1) {
#ifdef COLLECT_STATISTICS
                ++gNumMatch;
#endif
                Count cnt = 1;
                // multiply count in children
                for (int j = 0; j < child.size(); ++j) {
                    VertexID cID = child[j];
                    if (childKeyPos[j].empty()) cnt *= H[cID][0];
                    else if (childKeyPos[j].size() == 1) {
                        cnt *= H[cID][dataV[childKeyPos[j][0]]];
                    }
                    else {
                        cnt *= H[cID][childKey[j]];
                    }
                }
                if (isRoot) {
                    if (orbitType == 0) h[0] += cnt * aggreWeight[0];
                    else if (orbitType == 1) {
                        for (int j = 0; j < aggreV.size(); ++j) {
                            VertexID key = dataV[aggrePos[j]];
#ifdef COLLECT_STATISTICS
                            ++gNumUpdate;
#endif
                            h[key] += cnt * aggreWeight[j];
                        }
                    }
                    else {
                        for (int j = 0; j < aggreKey.size(); ++j) {
#ifdef COLLECT_STATISTICS
                            ++gNumUpdate;
#endif
                            h[aggreKey[j]] += cnt * aggreWeight[j];
                        }
                    }
                }
                else {
                    if (tau.keySize == 0) h[0] += cnt;
                    else if (tau.keySize == 1) {
#ifdef COLLECT_STATISTICS
                        ++gNumUpdate;
#endif
                        h[dataV[aggrePos[0]]] += cnt;
                        if (keyPosSize < sizeBound) {
                            keyPos[keyPosSize] = dataV[aggrePos[0]];
                            ++keyPosSize;
                        }
                    }
                    else {
                        EdgeID e = aggreKey[0];
#ifdef COLLECT_STATISTICS
                        ++gNumUpdate;
#endif
                        h[e] += cnt;
                        if (keyPosSize < sizeBound) {
                            keyPos[keyPosSize] = e;
                            ++keyPosSize;
                        }
                    }
                }
                visited[dataV[mappingSize]] = false;
            }
            else {
#ifdef COLLECT_STATISTICS
                ++gNumIntermediate;
#endif
                ++mappingSize;
                pos[mappingSize] = 0;
                if (!nodeInterPos[mappingSize]) {
                    if (!nodeOutPos[mappingSize].empty()) {
                        VertexID w = dataV[nodeOutPos[mappingSize][0]];
                        startOffset[mappingSize] = outOffset[w];
                        candidate[mappingSize] = outNbors + outOffset[w];
                        candCount[mappingSize] = outOffset[w + 1] - outOffset[w];
                    }
                    else if (!nodeInPos[mappingSize].empty()){
                        VertexID w = dataV[nodeInPos[mappingSize][0]];
                        startOffset[mappingSize] = inOffset[w];
                        candidate[mappingSize] = inNbors + inOffset[w];
                        candCount[mappingSize] = inOffset[w + 1] - inOffset[w];
                    }
                    else {
                        VertexID w = dataV[nodeUnPos[mappingSize][0]];
                        startOffset[mappingSize] = unOffset[w];
                        candidate[mappingSize] = unNbors + unOffset[w];
                        candCount[mappingSize] = unOffset[w + 1] - unOffset[w];
                    }
                }
                else {
                    EdgeID e = 0;
                    if (triEdgeType[mappingSize] != 0) {
                        int pos1 = nodeTriPos[mappingSize].first, pos2 = nodeTriPos[mappingSize].second;
                        if (triEdgeType[mappingSize] == 1)
                            e = startOffset[pos2] + pos[pos2] - 1;
                        else if (triEdgeType[mappingSize] == 2)
                            e = outID[startOffset[pos2] + pos[pos2] - 1];
                        else if (triEdgeType[mappingSize] == 5)
                            e = unID[startOffset[pos2] + pos[pos2] - 1];
                        else {
                            VertexID key1 = dataV[pos1], key2 = dataV[pos2];
#ifdef COLLECT_STATISTICS
                            ++gNumEdgeID;
#endif
                            e = dout.getEdgeID(key1, key2);
                        }
                    }
                    generateCandidateT(din, dout, dun, tri, dataV, mappingSize, !nodeCandPos[mappingSize], e, triEndType[mappingSize],
                                       nodeInPos[mappingSize], nodeOutPos[mappingSize], nodeUnPos[mappingSize], candidate, candCount, tmp);
                }
                // apply the symmetry breaking rules
                if (!greaterPos[mappingSize].empty()) {
                    ui maxTarget = dataV[greaterPos[mappingSize][0]];
                    for (int i = 1; i < greaterPos[mappingSize].size(); ++i) {
                        if (dataV[greaterPos[mappingSize][i]] > maxTarget)
                            maxTarget = dataV[greaterPos[mappingSize][i]];
                    }
                    pos[mappingSize] = firstPosGreaterThan(candidate[mappingSize], 0, candCount[mappingSize], maxTarget);
                }
                if (!lessPos[mappingSize].empty()) {
                    ui minTarget = dataV[lessPos[mappingSize][0]];
                    for (int i = 1; i < lessPos[mappingSize].size(); ++i) {
                        if (dataV[lessPos[mappingSize][i]] < minTarget)
                            minTarget = dataV[lessPos[mappingSize][i]];
                    }
                    candCount[mappingSize] = firstPosGreaterThan(candidate[mappingSize], 0, candCount[mappingSize], minTarget);
                }
            }
        }
        --mappingSize;
        if (mappingSize >= (int)tau.prefixSize) {
            visited[dataV[mappingSize]] = false;
        }
    }
}

void multiJoinWrapper(
        VertexID nID,
        const Tree &t,
        const std::vector<VertexID> &child,
        VertexID ***candidates,
        ui **candCounts,
        HashTable *H,
        const DataGraph &din,
        const DataGraph &dout,
        const DataGraph &dun,
        bool useTriangle,
        const Triangle &tri,
        const Pattern &p,
        EdgeID *outID,
        EdgeID *unID,
        EdgeID *reverseID,
        EdgeID **startOffsets,
        VertexID **patternVs,
        VertexID **dataVs,
        bool **visits,
        ui **poses,
        ui **keyPoses,
        ui *keyPosSizes,
        ui *sizeBounds,
        VertexID *&tmp,
        VertexID *allV
) {
    bool edgeKey = t.getNode(nID).edgeKey;
    if (!useTriangle) {
        if (!edgeKey) {
            multiJoin(nID, t, child, candidates, candCounts, H, din, dout, dun, tri, p, outID, unID, reverseID,
                      startOffsets, patternVs, dataVs, visits, poses, keyPoses, keyPosSizes, sizeBounds, tmp, allV);
        }
        else {
            multiJoinE(nID, t, child, candidates, candCounts, H, din, dout, dun, tri, p, outID, unID, reverseID,
                       startOffsets, patternVs, dataVs, visits, poses, keyPoses, keyPosSizes, sizeBounds, tmp, allV);
        }
    }
    else {
        if (!edgeKey) {
            multiJoinT(nID, t, child, candidates, candCounts, H, din, dout, dun, tri, p, outID, unID, reverseID,
                       startOffsets, patternVs, dataVs, visits, poses, keyPoses, keyPosSizes, sizeBounds, tmp, allV);
        }
        else {
            multiJoinET(nID, t, child, candidates, candCounts, H, din, dout, dun, tri, p, outID, unID, reverseID,
                        startOffsets, patternVs, dataVs, visits, poses, keyPoses, keyPosSizes, sizeBounds, tmp, allV);
        }
    }
}

void multiJoinTree(
        const Tree &t,
        const DataGraph &din,
        const DataGraph &dout,
        const DataGraph &dun,
        bool useTriangle,
        const Triangle &tri,
        const Pattern &p,
        HashTable *H,
        EdgeID *outID,
        EdgeID *unID,
        EdgeID *reverseID,
        EdgeID **startOffset,
        VertexID **patternV,
        VertexID **dataV,
        bool **visited,
        VertexID *&tmp,
        VertexID *allV
) {
    int numNodes = (int)t.getNumNodes();
    ui n = dun.getNumVertices(), m = dun.getNumEdges();
    VertexID ***candidate = new VertexID **[numNodes];
    ui **candCount = new ui *[numNodes];
    ui **pos = new ui *[numNodes];
    for (VertexID nID = 0; nID < numNodes; ++nID) {
        const Node &tau = t.getNode(nID);
        candCount[nID] = new ui[tau.numVertices];
        candidate[nID] = new VertexID *[tau.numVertices];
        pos[nID] = new ui[tau.numVertices];
        memset(pos[nID], 0, sizeof(ui) * tau.numVertices);
        const std::vector<bool> &nodeCandPos = t.getNodeCandPos(nID);
        for (ui i = tau.prefixSize; i < tau.numVertices; ++i) {
            if (nodeCandPos[i])
                candidate[nID][i] = new VertexID[dout.getNumVertices()];
        }
    }
    ui **keyPos = new ui *[t.getNumNodes()];
    ui *keyPosSize = new ui[t.getNumNodes()];
    memset(keyPosSize, 0, sizeof(ui) * (t.getNumNodes()));
    ui *sizeBounds = new ui[t.getNumNodes()];
    for (VertexID nID = 0; nID < numNodes; ++nID) {
        if (t.getNode(nID).keySize == 0)
            sizeBounds[nID] = 1;
        else if (t.getNode(nID).keySize == 1)
            sizeBounds[nID] = n / 8 + 1;
        else if (t.getNode(nID).keySize == 2)
            sizeBounds[nID] = m / 8 + 1;
        keyPos[nID] = new ui[sizeBounds[nID]];
    }
    // call simple nodes
    for (VertexID nID: t.getPostOrder()) {
        if (t.getNode(nID).prefixSize == 0) {
            multiJoinWrapper(nID, t, t.getChild()[nID], candidate, candCount, H, din, dout, dun, useTriangle, tri, p,
                             outID, unID, reverseID, startOffset, patternV, dataV, visited,
                             pos, keyPos, keyPosSize, sizeBounds, tmp, allV);
        }
    }
    for (VertexID nID = 0; nID < numNodes; ++nID) {
        const Node &tau = t.getNode(nID);
        const std::vector<bool> &nodeCandPos = t.getNodeCandPos(nID);
        for (ui i = tau.prefixSize; i < tau.numVertices; ++i) {
            if (nodeCandPos[i])
                delete[] candidate[nID][i];
        }
        delete[] candidate[nID];
        delete[] candCount[nID];
        delete[] pos[nID];
        delete[] keyPos[nID];
    }
    delete[] candCount;
    delete[] candidate;
    delete[] pos;
    delete[] keyPos;
}

void executeConNode(
        const ConNode &cn,
        const DataGraph &din,
        const DataGraph &dout,
        const DataGraph &dun,
        bool useTriangle,
        const Triangle &tri,
        const Pattern &p,
        EdgeID *outID,
        EdgeID *unID,
        EdgeID *startOffset,
        VertexID *patternV,
        VertexID *dataV,
        bool *visited,
        ui *pos,
        VertexID *&tmp,
        VertexID *allV
) {
    int orbitType = cn.orbitType;
    EdgeID *inOffset = din.getOffsets();
    VertexID *inNbors = din.getNbors();
    EdgeID *outOffset = dout.getOffsets();
    VertexID *outNbors = dout.getNbors();
    EdgeID *unOffset = dun.getOffsets();
    VertexID *unNbors = dun.getNbors();
    Count *cnt, *cntCopy;
    ui n = dun.getNumVertices(), m = dun.getNumEdges();
    if (cn.childKeyPos.size() == 1) {
        cnt = new Count[n];
        cntCopy = new Count[n];
        memset(cnt, 0, sizeof(Count) * n);
        memset(cntCopy, 0, sizeof(Count) * n);
    }
    else {
        if (p.u.isEOrbitDir()) {
            cnt = new Count[m];
            cntCopy = new Count[m];
            memset(cnt, 0, sizeof(Count) * m);
            memset(cntCopy, 0, sizeof(Count) * m);
        }
        else {
            cnt = new Count[m * 2];
            cntCopy = new Count[m * 2];
            memset(cnt, 0, sizeof(Count) * m * 2);
            memset(cntCopy, 0, sizeof(Count) * m * 2);
        }
    }
    int mappingSize = 0;
    int prefixSize = (int)cn.prefixOrder.size();
    VertexID **candidate = new VertexID *[cn.nodeOrder.size()];
    ui *candCount = new ui[cn.nodeOrder.size()];
    for (int i = 0; i < cn.nodeOrder.size(); ++i) {
        if (cn.candPos[i])
            candidate[i] = new VertexID[n];
    }
    candidate[0] = allV;
    candCount[0] = n;
    pos[0] = 0;
    if (!cn.edgeKey) {
        while (mappingSize >= 0) {
            // matching the prefix
            if (mappingSize < prefixSize) {
                while (pos[mappingSize] < candCount[mappingSize]) {
                    VertexID v = candidate[mappingSize][pos[mappingSize]];
                    ++pos[mappingSize];
                    if (visited[v]) continue;
                    visited[v] = true;
                    patternV[mappingSize] = cn.nodeOrder[mappingSize];
                    dataV[mappingSize] = v;
                    ++mappingSize;
                    pos[mappingSize] = 0;
                    if (!cn.interPos[mappingSize]) {
                        if (!cn.nodeOutPos[mappingSize].empty()) {
                            VertexID w = dataV[cn.nodeOutPos[mappingSize][0]];
                            startOffset[mappingSize] = outOffset[w];
                            candidate[mappingSize] = outNbors + outOffset[w];
                            candCount[mappingSize] = outOffset[w + 1] - outOffset[w];
                        }
                        else if (!cn.nodeInPos[mappingSize].empty()){
                            VertexID w = dataV[cn.nodeInPos[mappingSize][0]];
                            startOffset[mappingSize] = inOffset[w];
                            candidate[mappingSize] = inNbors + inOffset[w];
                            candCount[mappingSize] = inOffset[w + 1] - inOffset[w];
                        }
                        else {
                            VertexID w = dataV[cn.nodeUnPos[mappingSize][0]];
                            startOffset[mappingSize] = unOffset[w];
                            candidate[mappingSize] = unNbors + unOffset[w];
                            candCount[mappingSize] = unOffset[w + 1] - unOffset[w];
                        }
                    }
                    else {
                        EdgeID e = 0;
                        if (cn.triEdgeType[mappingSize] != 0) {
                            int pos1 = cn.triPos[mappingSize].first, pos2 = cn.triPos[mappingSize].second;
                            if (cn.triEdgeType[mappingSize] == 1)
                                e = startOffset[pos2] + pos[pos2] - 1;
                            else if (cn.triEdgeType[mappingSize] == 2)
                                e = outID[startOffset[pos2] + pos[pos2] - 1];
                            else if (cn.triEdgeType[mappingSize] == 5)
                                e = unID[startOffset[pos2] + pos[pos2] - 1];
                            else {
                                VertexID key1 = dataV[pos1], key2 = dataV[pos2];
#ifdef COLLECT_STATISTICS
                                ++gNumEdgeID;
#endif
                                e = dout.getEdgeID(key1, key2);
                            }
                        }
                        generateCandidateT(din, dout, dun, tri, dataV, mappingSize, !cn.candPos[mappingSize], e, cn.triEndType[mappingSize],
                                           cn.nodeInPos[mappingSize], cn.nodeOutPos[mappingSize], cn.nodeUnPos[mappingSize], candidate, candCount, tmp);
                    }
                    // apply the symmetry breaking rules
                    if (!cn.greaterPos[mappingSize].empty()) {
                        ui maxTarget = dataV[cn.greaterPos[mappingSize][0]];
                        for (int i = 1; i < cn.greaterPos[mappingSize].size(); ++i) {
                            if (dataV[cn.greaterPos[mappingSize][i]] > maxTarget)
                                maxTarget = dataV[cn.greaterPos[mappingSize][i]];
                        }
                        pos[mappingSize] = firstPosGreaterThan(candidate[mappingSize], 0, candCount[mappingSize], maxTarget);
                    }
                    if (!cn.lessPos[mappingSize].empty()) {
                        ui minTarget = dataV[cn.lessPos[mappingSize][0]];
                        for (int i = 1; i < cn.lessPos[mappingSize].size(); ++i) {
                            if (dataV[cn.lessPos[mappingSize][i]] < minTarget)
                                minTarget = dataV[cn.lessPos[mappingSize][i]];
                        }
                        candCount[mappingSize] = firstPosGreaterThan(candidate[mappingSize], 0, candCount[mappingSize], minTarget);
                    }
                    if (mappingSize == prefixSize) break;
                }
                if (mappingSize != prefixSize) {
                    --mappingSize;
                    if (mappingSize >= 0)
                        visited[dataV[mappingSize]] = false;
                }
            }
            else {
                // matching the remaining vertices to compute the child table
                while (mappingSize >= prefixSize) {
                    while (pos[mappingSize] < candCount[mappingSize]) {
                        VertexID v = candidate[mappingSize][pos[mappingSize]];
                        ++pos[mappingSize];
                        if (visited[v]) continue;
                        visited[v] = true;
                        patternV[mappingSize] = cn.nodeOrder[mappingSize];
                        dataV[mappingSize] = v;
                        if (mappingSize == cn.nodeOrder.size() - 1) {
#ifdef COLLECT_STATISTICS
                            ++gNumMatch;
#endif
                            ++cnt[dataV[cn.childKeyPos[0]]];
                            cntCopy[dataV[cn.childKeyPos[0]]] = cnt[dataV[cn.childKeyPos[0]]];
                            visited[dataV[mappingSize]] = false;
                        }
                        else {
#ifdef COLLECT_STATISTICS
                            ++gNumIntermediate;
#endif
                            ++mappingSize;
                            pos[mappingSize] = 0;
                            if (!cn.interPos[mappingSize]) {
                                if (!cn.nodeOutPos[mappingSize].empty()) {
                                    VertexID w = dataV[cn.nodeOutPos[mappingSize][0]];
                                    startOffset[mappingSize] = outOffset[w];
                                    candidate[mappingSize] = outNbors + outOffset[w];
                                    candCount[mappingSize] = outOffset[w + 1] - outOffset[w];
                                }
                                else if (!cn.nodeInPos[mappingSize].empty()){
                                    VertexID w = dataV[cn.nodeInPos[mappingSize][0]];
                                    startOffset[mappingSize] = inOffset[w];
                                    candidate[mappingSize] = inNbors + inOffset[w];
                                    candCount[mappingSize] = inOffset[w + 1] - inOffset[w];
                                }
                                else {
                                    VertexID w = dataV[cn.nodeUnPos[mappingSize][0]];
                                    startOffset[mappingSize] = unOffset[w];
                                    candidate[mappingSize] = unNbors + unOffset[w];
                                    candCount[mappingSize] = unOffset[w + 1] - unOffset[w];
                                }
                            }
                            else {
                                EdgeID e = 0;
                                if (cn.triEdgeType[mappingSize] != 0) {
                                    int pos1 = cn.triPos[mappingSize].first, pos2 = cn.triPos[mappingSize].second;
                                    if (cn.triEdgeType[mappingSize] == 1)
                                        e = startOffset[pos2] + pos[pos2] - 1;
                                    else if (cn.triEdgeType[mappingSize] == 2)
                                        e = outID[startOffset[pos2] + pos[pos2] - 1];
                                    else if (cn.triEdgeType[mappingSize] == 5)
                                        e = unID[startOffset[pos2] + pos[pos2] - 1];
                                    else {
                                        VertexID key1 = dataV[pos1], key2 = dataV[pos2];
#ifdef COLLECT_STATISTICS
                                        ++gNumEdgeID;
#endif
                                        e = dout.getEdgeID(key1, key2);
                                    }
                                }
                                generateCandidateT(din, dout, dun, tri, dataV, mappingSize, !cn.candPos[mappingSize], e, cn.triEndType[mappingSize],
                                                   cn.nodeInPos[mappingSize], cn.nodeOutPos[mappingSize], cn.nodeUnPos[mappingSize], candidate, candCount, tmp);
                            }
                            // apply the symmetry breaking rules
                            if (!cn.greaterPos[mappingSize].empty()) {
                                ui maxTarget = dataV[cn.greaterPos[mappingSize][0]];
                                for (int i = 1; i < cn.greaterPos[mappingSize].size(); ++i) {
                                    if (dataV[cn.greaterPos[mappingSize][i]] > maxTarget)
                                        maxTarget = dataV[cn.greaterPos[mappingSize][i]];
                                }
                                pos[mappingSize] = firstPosGreaterThan(candidate[mappingSize], 0, candCount[mappingSize], maxTarget);
                            }
                            if (!cn.lessPos[mappingSize].empty()) {
                                ui minTarget = dataV[cn.lessPos[mappingSize][0]];
                                for (int i = 1; i < cn.lessPos[mappingSize].size(); ++i) {
                                    if (dataV[cn.lessPos[mappingSize][i]] < minTarget)
                                        minTarget = dataV[cn.lessPos[mappingSize][i]];
                                }
                                candCount[mappingSize] = firstPosGreaterThan(candidate[mappingSize], 0, candCount[mappingSize], minTarget);
                            }
                        }
                    }
                    --mappingSize;
                    if (mappingSize >= prefixSize)
                        visited[dataV[mappingSize]] = false;
                }
                // matching the remaining vertices to query the child table and compute orbit count
                ++mappingSize;
                pos[mappingSize] = 0;
                if (!cn.greaterPos[mappingSize].empty()) {
                    ui maxTarget = dataV[cn.greaterPos[mappingSize][0]];
                    for (int i = 1; i < cn.greaterPos[mappingSize].size(); ++i) {
                        if (dataV[cn.greaterPos[mappingSize][i]] > maxTarget)
                            maxTarget = dataV[cn.greaterPos[mappingSize][i]];
                    }
                    pos[mappingSize] = firstPosGreaterThan(candidate[mappingSize], 0, candCount[mappingSize], maxTarget);
                }
                if (!cn.lessPos[mappingSize].empty()) {
                    ui minTarget = dataV[cn.lessPos[mappingSize][0]];
                    for (int i = 1; i < cn.lessPos[mappingSize].size(); ++i) {
                        if (dataV[cn.lessPos[mappingSize][i]] < minTarget)
                            minTarget = dataV[cn.lessPos[mappingSize][i]];
                    }
                    candCount[mappingSize] = firstPosGreaterThan(candidate[mappingSize], 0, candCount[mappingSize], minTarget);
                }
                while (mappingSize >= prefixSize) {
                    while (pos[mappingSize] < candCount[mappingSize]) {
                        VertexID v = candidate[mappingSize][pos[mappingSize]];
                        ++pos[mappingSize];
                        if (visited[v]) continue;
                        visited[v] = true;
                        patternV[mappingSize] = cn.nodeOrder[mappingSize];
                        dataV[mappingSize] = v;
                        if (mappingSize == cn.nodeOrder.size() - 1) {
#ifdef COLLECT_STATISTICS
                            ++gNumMatch;
#endif
                            ui childKey = dataV[cn.childKeyPos[0]];
                            if (orbitType == 0) {
                                cn.hashTables[0][0] += choosec(cnt[childKey], cn.num);
                            }
                            else {
                                for (int j = 0; j < cn.aggrePos.size(); ++j) {
#ifdef COLLECT_STATISTICS
                                    ++gNumUpdate;
#endif
                                    ui aggreKey = dataV[cn.aggrePos[j]];
                                    if (cn.aggreType[j])
                                        cn.hashTables[j][aggreKey] += choosec(cnt[childKey], cn.num);
                                    else
                                        cn.hashTables[j][aggreKey] += choosec(cntCopy[childKey] - 1, cn.num - 1);
                                }
                            }
                            cnt[childKey] = 0;
                            visited[dataV[mappingSize]] = false;
                        }
                        else {
#ifdef COLLECT_STATISTICS
                            ++gNumIntermediate;
#endif
                            ++mappingSize;
                            pos[mappingSize] = 0;
                            if (!cn.interPos[mappingSize]) {
                                if (!cn.nodeOutPos[mappingSize].empty()) {
                                    VertexID w = dataV[cn.nodeOutPos[mappingSize][0]];
                                    startOffset[mappingSize] = outOffset[w];
                                    candidate[mappingSize] = outNbors + outOffset[w];
                                    candCount[mappingSize] = outOffset[w + 1] - outOffset[w];
                                }
                                else if (!cn.nodeInPos[mappingSize].empty()){
                                    VertexID w = dataV[cn.nodeInPos[mappingSize][0]];
                                    startOffset[mappingSize] = inOffset[w];
                                    candidate[mappingSize] = inNbors + inOffset[w];
                                    candCount[mappingSize] = inOffset[w + 1] - inOffset[w];
                                }
                                else {
                                    VertexID w = dataV[cn.nodeUnPos[mappingSize][0]];
                                    startOffset[mappingSize] = unOffset[w];
                                    candidate[mappingSize] = unNbors + unOffset[w];
                                    candCount[mappingSize] = unOffset[w + 1] - unOffset[w];
                                }
                            }
                            else {
                                EdgeID e = 0;
                                if (cn.triEdgeType[mappingSize] != 0) {
                                    int pos1 = cn.triPos[mappingSize].first, pos2 = cn.triPos[mappingSize].second;
                                    if (cn.triEdgeType[mappingSize] == 1)
                                        e = startOffset[pos2] + pos[pos2] - 1;
                                    else if (cn.triEdgeType[mappingSize] == 2)
                                        e = outID[startOffset[pos2] + pos[pos2] - 1];
                                    else if (cn.triEdgeType[mappingSize] == 5)
                                        e = unID[startOffset[pos2] + pos[pos2] - 1];
                                    else {
                                        VertexID key1 = dataV[pos1], key2 = dataV[pos2];
#ifdef COLLECT_STATISTICS
                                        ++gNumEdgeID;
#endif
                                        e = dout.getEdgeID(key1, key2);
                                    }
                                }
                                generateCandidateT(din, dout, dun, tri, dataV, mappingSize, !cn.candPos[mappingSize], e, cn.triEndType[mappingSize],
                                                   cn.nodeInPos[mappingSize], cn.nodeOutPos[mappingSize], cn.nodeUnPos[mappingSize], candidate, candCount, tmp);

                            }
                            // apply the symmetry breaking rules
                            if (!cn.greaterPos[mappingSize].empty()) {
                                ui maxTarget = dataV[cn.greaterPos[mappingSize][0]];
                                for (int i = 1; i < cn.greaterPos[mappingSize].size(); ++i) {
                                    if (dataV[cn.greaterPos[mappingSize][i]] > maxTarget)
                                        maxTarget = dataV[cn.greaterPos[mappingSize][i]];
                                }
                                pos[mappingSize] = firstPosGreaterThan(candidate[mappingSize], 0, candCount[mappingSize], maxTarget);
                            }
                            if (!cn.lessPos[mappingSize].empty()) {
                                ui minTarget = dataV[cn.lessPos[mappingSize][0]];
                                for (int i = 1; i < cn.lessPos[mappingSize].size(); ++i) {
                                    if (dataV[cn.lessPos[mappingSize][i]] < minTarget)
                                        minTarget = dataV[cn.lessPos[mappingSize][i]];
                                }
                                candCount[mappingSize] = firstPosGreaterThan(candidate[mappingSize], 0, candCount[mappingSize], minTarget);
                            }
                        }
                    }
                    --mappingSize;
                    if (mappingSize >= 0)
                        visited[dataV[mappingSize]] = false;
                }
            }
        }
        delete[] cnt;
        delete[] cntCopy;
        return;
    }
    // the edge key case
    EdgeID childKey = 0;
    std::vector<EdgeID> aggreKey;
    if (orbitType == 2) aggreKey = std::vector<EdgeID>(cn.aggreEdgeType.size());
    while (mappingSize >= 0) {
        // matching the prefix
        if (mappingSize < prefixSize) {
            while (pos[mappingSize] < candCount[mappingSize]) {
                VertexID v = candidate[mappingSize][pos[mappingSize]];
                ++pos[mappingSize];
                if (visited[v]) continue;
                visited[v] = true;
                patternV[mappingSize] = cn.nodeOrder[mappingSize];
                dataV[mappingSize] = v;
                ++mappingSize;
                pos[mappingSize] = 0;
                if (!cn.interPos[mappingSize]) {
                    if (!cn.nodeOutPos[mappingSize].empty()) {
                        VertexID w = dataV[cn.nodeOutPos[mappingSize][0]];
                        startOffset[mappingSize] = outOffset[w];
                        candidate[mappingSize] = outNbors + outOffset[w];
                        candCount[mappingSize] = outOffset[w + 1] - outOffset[w];
                    }
                    else if (!cn.nodeInPos[mappingSize].empty()){
                        VertexID w = dataV[cn.nodeInPos[mappingSize][0]];
                        startOffset[mappingSize] = inOffset[w];
                        candidate[mappingSize] = inNbors + inOffset[w];
                        candCount[mappingSize] = inOffset[w + 1] - inOffset[w];
                    }
                    else {
                        VertexID w = dataV[cn.nodeUnPos[mappingSize][0]];
                        startOffset[mappingSize] = unOffset[w];
                        candidate[mappingSize] = unNbors + unOffset[w];
                        candCount[mappingSize] = unOffset[w + 1] - unOffset[w];
                    }
                }
                else {
                    EdgeID e = 0;
                    if (cn.triEdgeType[mappingSize] != 0) {
                        int pos1 = cn.triPos[mappingSize].first, pos2 = cn.triPos[mappingSize].second;
                        if (cn.triEdgeType[mappingSize] == 1)
                            e = startOffset[pos2] + pos[pos2] - 1;
                        else if (cn.triEdgeType[mappingSize] == 2)
                            e = outID[startOffset[pos2] + pos[pos2] - 1];
                        else if (cn.triEdgeType[mappingSize] == 5)
                            e = unID[startOffset[pos2] + pos[pos2] - 1];
                        else {
                            VertexID key1 = dataV[pos1], key2 = dataV[pos2];
#ifdef COLLECT_STATISTICS
                            ++gNumEdgeID;
#endif
                            e = dout.getEdgeID(key1, key2);
                        }
                    }
                    generateCandidateT(din, dout, dun, tri, dataV, mappingSize, !cn.candPos[mappingSize], e, cn.triEndType[mappingSize],
                                       cn.nodeInPos[mappingSize], cn.nodeOutPos[mappingSize], cn.nodeUnPos[mappingSize], candidate, candCount, tmp);
                }
                // apply the symmetry breaking rules
                if (!cn.greaterPos[mappingSize].empty()) {
                    ui maxTarget = dataV[cn.greaterPos[mappingSize][0]];
                    for (int i = 1; i < cn.greaterPos[mappingSize].size(); ++i) {
                        if (dataV[cn.greaterPos[mappingSize][i]] > maxTarget)
                            maxTarget = dataV[cn.greaterPos[mappingSize][i]];
                    }
                    pos[mappingSize] = firstPosGreaterThan(candidate[mappingSize], 0, candCount[mappingSize], maxTarget);
                }
                if (!cn.lessPos[mappingSize].empty()) {
                    ui minTarget = dataV[cn.lessPos[mappingSize][0]];
                    for (int i = 1; i < cn.lessPos[mappingSize].size(); ++i) {
                        if (dataV[cn.lessPos[mappingSize][i]] < minTarget)
                            minTarget = dataV[cn.lessPos[mappingSize][i]];
                    }
                    candCount[mappingSize] = firstPosGreaterThan(candidate[mappingSize], 0, candCount[mappingSize], minTarget);
                }
                if (mappingSize == prefixSize) break;
            }
            if (mappingSize != prefixSize) {
                --mappingSize;
                if (mappingSize >= 0)
                    visited[dataV[mappingSize]] = false;
            }
        }
        else {
            // matching the remaining vertices to compute the child table
            while (mappingSize >= prefixSize) {
                while (pos[mappingSize] < candCount[mappingSize]) {
                    VertexID v = candidate[mappingSize][pos[mappingSize]];
                    ++pos[mappingSize];
                    if (visited[v]) continue;
                    visited[v] = true;
                    patternV[mappingSize] = cn.nodeOrder[mappingSize];
                    dataV[mappingSize] = v;
                    if (mappingSize == cn.nodeOrder.size() - 1) {
#ifdef COLLECT_STATISTICS
                        ++gNumMatch;
#endif
                        if (cn.childEdgeType == 0) childKey = dataV[cn.childKeyPos[0]];
                        else {
                            if (cn.childEdgeType == 1) childKey = startOffset[cn.posChildEdge] + pos[cn.posChildEdge] - 1;
                            else if (cn.childEdgeType == 2) childKey = outID[startOffset[cn.posChildEdge] + pos[cn.posChildEdge] - 1];
                            else {
                                VertexID key1 = dataV[cn.childKeyPos[0]], key2 = dataV[cn.childKeyPos[1]];
                                if (cn.childEdgeType == 3) childKey = dout.getEdgeID(key1, key2);
                                else childKey = dun.getUndirectedEID(key1, key2);
#ifdef COLLECT_STATISTICS
                                ++gNumEdgeID;
#endif
                            }
                        }

                        ++cnt[childKey];
                        cntCopy[childKey] = cnt[childKey];
                        visited[dataV[mappingSize]] = false;
                    }
                    else {
#ifdef COLLECT_STATISTICS
                        ++gNumIntermediate;
#endif
                        ++mappingSize;
                        pos[mappingSize] = 0;
                        if (!cn.interPos[mappingSize]) {
                            if (!cn.nodeOutPos[mappingSize].empty()) {
                                VertexID w = dataV[cn.nodeOutPos[mappingSize][0]];
                                startOffset[mappingSize] = outOffset[w];
                                candidate[mappingSize] = outNbors + outOffset[w];
                                candCount[mappingSize] = outOffset[w + 1] - outOffset[w];
                            }
                            else if (!cn.nodeInPos[mappingSize].empty()){
                                VertexID w = dataV[cn.nodeInPos[mappingSize][0]];
                                startOffset[mappingSize] = inOffset[w];
                                candidate[mappingSize] = inNbors + inOffset[w];
                                candCount[mappingSize] = inOffset[w + 1] - inOffset[w];
                            }
                            else {
                                VertexID w = dataV[cn.nodeUnPos[mappingSize][0]];
                                startOffset[mappingSize] = unOffset[w];
                                candidate[mappingSize] = unNbors + unOffset[w];
                                candCount[mappingSize] = unOffset[w + 1] - unOffset[w];
                            }
                        }
                        else {
                            EdgeID e = 0;
                            if (cn.triEdgeType[mappingSize] != 0) {
                                int pos1 = cn.triPos[mappingSize].first, pos2 = cn.triPos[mappingSize].second;
                                if (cn.triEdgeType[mappingSize] == 1)
                                    e = startOffset[pos2] + pos[pos2] - 1;
                                else if (cn.triEdgeType[mappingSize] == 2)
                                    e = outID[startOffset[pos2] + pos[pos2] - 1];
                                else if (cn.triEdgeType[mappingSize] == 5)
                                    e = unID[startOffset[pos2] + pos[pos2] - 1];
                                else {
                                    VertexID key1 = dataV[pos1], key2 = dataV[pos2];
#ifdef COLLECT_STATISTICS
                                    ++gNumEdgeID;
#endif
                                    e = dout.getEdgeID(key1, key2);
                                }
                            }
                            generateCandidateT(din, dout, dun, tri, dataV, mappingSize, !cn.candPos[mappingSize], e, cn.triEndType[mappingSize],
                                               cn.nodeInPos[mappingSize], cn.nodeOutPos[mappingSize], cn.nodeUnPos[mappingSize], candidate, candCount, tmp);
                        }
                        // apply the symmetry breaking rules
                        if (!cn.greaterPos[mappingSize].empty()) {
                            ui maxTarget = dataV[cn.greaterPos[mappingSize][0]];
                            for (int i = 1; i < cn.greaterPos[mappingSize].size(); ++i) {
                                if (dataV[cn.greaterPos[mappingSize][i]] > maxTarget)
                                    maxTarget = dataV[cn.greaterPos[mappingSize][i]];
                            }
                            pos[mappingSize] = firstPosGreaterThan(candidate[mappingSize], 0, candCount[mappingSize], maxTarget);
                        }
                        if (!cn.lessPos[mappingSize].empty()) {
                            ui minTarget = dataV[cn.lessPos[mappingSize][0]];
                            for (int i = 1; i < cn.lessPos[mappingSize].size(); ++i) {
                                if (dataV[cn.lessPos[mappingSize][i]] < minTarget)
                                    minTarget = dataV[cn.lessPos[mappingSize][i]];
                            }
                            candCount[mappingSize] = firstPosGreaterThan(candidate[mappingSize], 0, candCount[mappingSize], minTarget);
                        }
                    }
                }
                --mappingSize;
                if (mappingSize >= prefixSize)
                    visited[dataV[mappingSize]] = false;
            }
            // matching the remaining vertices to query the child table and compute orbit count
            ++mappingSize;
            pos[mappingSize] = 0;
            if (!cn.greaterPos[mappingSize].empty()) {
                ui maxTarget = dataV[cn.greaterPos[mappingSize][0]];
                for (int i = 1; i < cn.greaterPos[mappingSize].size(); ++i) {
                    if (dataV[cn.greaterPos[mappingSize][i]] > maxTarget)
                        maxTarget = dataV[cn.greaterPos[mappingSize][i]];
                }
                pos[mappingSize] = firstPosGreaterThan(candidate[mappingSize], 0, candCount[mappingSize], maxTarget);
            }
            if (!cn.lessPos[mappingSize].empty()) {
                ui minTarget = dataV[cn.lessPos[mappingSize][0]];
                for (int i = 1; i < cn.lessPos[mappingSize].size(); ++i) {
                    if (dataV[cn.lessPos[mappingSize][i]] < minTarget)
                        minTarget = dataV[cn.lessPos[mappingSize][i]];
                }
                candCount[mappingSize] = firstPosGreaterThan(candidate[mappingSize], 0, candCount[mappingSize], minTarget);
            }
            while (mappingSize >= prefixSize) {
                for (int j: cn.posAggreEdge[mappingSize]) {
                    if (cn.aggreEdgeType[j] == 1) aggreKey[j] = startOffset[mappingSize] + pos[mappingSize] - 1;
                    else if (cn.aggreEdgeType[j] == 2) aggreKey[j] = outID[startOffset[mappingSize] + pos[mappingSize] - 1];
                    else if (cn.aggreEdgeType[j] != 5){
                        VertexID key1 = dataV[cn.aggrePos[2 * j]], key2 = dataV[cn.aggrePos[2 * j + 1]];
                        if (cn.aggreEdgeType[j] == 3) aggreKey[j] = dout.getEdgeID(key1, key2);
                        else aggreKey[j] = dun.getUndirectedEID(key1, key2);
#ifdef COLLECT_STATISTICS
                        ++gNumEdgeID;
#endif
                    }
                    else aggreKey[j] = unID[startOffset[mappingSize] + pos[mappingSize] - 1];
                }
                while (pos[mappingSize] < candCount[mappingSize]) {
                    VertexID v = candidate[mappingSize][pos[mappingSize]];
                    ++pos[mappingSize];
                    if (visited[v]) continue;
                    visited[v] = true;
                    patternV[mappingSize] = cn.nodeOrder[mappingSize];
                    dataV[mappingSize] = v;
                    if (mappingSize == cn.nodeOrder.size() - 1) {
#ifdef COLLECT_STATISTICS
                        ++gNumMatch;
#endif
                        if (cn.childEdgeType == 0) childKey = cn.childKeyPos[0];
                        else {
                            if (cn.childEdgeType == 1) childKey = startOffset[cn.posChildEdge] + pos[cn.posChildEdge] - 1;
                            else if (cn.childEdgeType == 2) childKey = outID[startOffset[cn.posChildEdge] + pos[cn.posChildEdge] - 1];
                            else {
                                VertexID key1 = dataV[cn.childKeyPos[0]], key2 = dataV[cn.childKeyPos[1]];
                                if (cn.childEdgeType == 3) childKey = dout.getEdgeID(key1, key2);
                                else childKey = dun.getUndirectedEID(key1, key2);
#ifdef COLLECT_STATISTICS
                                ++gNumEdgeID;
#endif
                            }
                        }
                        if (orbitType == 0) {
                            cn.hashTables[0][0] += choosec(cnt[childKey], cn.num);
                        }
                        else if (orbitType == 1) {
                            for (int j = 0; j < cn.aggrePos.size(); ++j) {
#ifdef COLLECT_STATISTICS
                                ++gNumUpdate;
#endif
                                VertexID ak = dataV[cn.aggrePos[j]];
                                if (cn.aggreType[j])
                                    cn.hashTables[j][ak] += choosec(cnt[childKey], cn.num);
                                else
                                    cn.hashTables[j][ak] += choosec(cntCopy[childKey] - 1, cn.num - 1);
                            }
                        }
                        else {
                            for (int j = 0; j < aggreKey.size(); ++j) {
#ifdef COLLECT_STATISTICS
                                ++gNumUpdate;
#endif
                                if (cn.aggreType[j])
                                    cn.hashTables[j][aggreKey[j]] += choosec(cnt[childKey], cn.num);
                                else
                                    cn.hashTables[j][aggreKey[j]] += choosec(cntCopy[childKey] - 1, cn.num - 1);
                            }
                        }
                        cnt[childKey] = 0;
                        visited[dataV[mappingSize]] = false;
                    }
                    else {
#ifdef COLLECT_STATISTICS
                        ++gNumIntermediate;
#endif
                        ++mappingSize;
                        pos[mappingSize] = 0;
                        if (!cn.interPos[mappingSize]) {
                            if (!cn.nodeOutPos[mappingSize].empty()) {
                                VertexID w = dataV[cn.nodeOutPos[mappingSize][0]];
                                startOffset[mappingSize] = outOffset[w];
                                candidate[mappingSize] = outNbors + outOffset[w];
                                candCount[mappingSize] = outOffset[w + 1] - outOffset[w];
                            }
                            else if (!cn.nodeInPos[mappingSize].empty()){
                                VertexID w = dataV[cn.nodeInPos[mappingSize][0]];
                                startOffset[mappingSize] = inOffset[w];
                                candidate[mappingSize] = inNbors + inOffset[w];
                                candCount[mappingSize] = inOffset[w + 1] - inOffset[w];
                            }
                            else {
                                VertexID w = dataV[cn.nodeUnPos[mappingSize][0]];
                                startOffset[mappingSize] = unOffset[w];
                                candidate[mappingSize] = unNbors + unOffset[w];
                                candCount[mappingSize] = unOffset[w + 1] - unOffset[w];
                            }
                        }
                        else {
                            EdgeID e = 0;
                            if (cn.triEdgeType[mappingSize] != 0) {
                                int pos1 = cn.triPos[mappingSize].first, pos2 = cn.triPos[mappingSize].second;
                                if (cn.triEdgeType[mappingSize] == 1)
                                    e = startOffset[pos2] + pos[pos2] - 1;
                                else if (cn.triEdgeType[mappingSize] == 2)
                                    e = outID[startOffset[pos2] + pos[pos2] - 1];
                                else if (cn.triEdgeType[mappingSize] == 5)
                                    e = unID[startOffset[pos2] + pos[pos2] - 1];
                                else {
                                    VertexID key1 = dataV[pos1], key2 = dataV[pos2];
#ifdef COLLECT_STATISTICS
                                    ++gNumEdgeID;
#endif
                                    e = dout.getEdgeID(key1, key2);
                                }
                            }
                            generateCandidateT(din, dout, dun, tri, dataV, mappingSize, !cn.candPos[mappingSize], e, cn.triEndType[mappingSize],
                                               cn.nodeInPos[mappingSize], cn.nodeOutPos[mappingSize], cn.nodeUnPos[mappingSize], candidate, candCount, tmp);

                        }
                        // apply the symmetry breaking rules
                        if (!cn.greaterPos[mappingSize].empty()) {
                            ui maxTarget = dataV[cn.greaterPos[mappingSize][0]];
                            for (int i = 1; i < cn.greaterPos[mappingSize].size(); ++i) {
                                if (dataV[cn.greaterPos[mappingSize][i]] > maxTarget)
                                    maxTarget = dataV[cn.greaterPos[mappingSize][i]];
                            }
                            pos[mappingSize] = firstPosGreaterThan(candidate[mappingSize], 0, candCount[mappingSize], maxTarget);
                        }
                        if (!cn.lessPos[mappingSize].empty()) {
                            ui minTarget = dataV[cn.lessPos[mappingSize][0]];
                            for (int i = 1; i < cn.lessPos[mappingSize].size(); ++i) {
                                if (dataV[cn.lessPos[mappingSize][i]] < minTarget)
                                    minTarget = dataV[cn.lessPos[mappingSize][i]];
                            }
                            candCount[mappingSize] = firstPosGreaterThan(candidate[mappingSize], 0, candCount[mappingSize], minTarget);
                        }
                    }
                }
                --mappingSize;
                if (mappingSize >= 0)
                    visited[dataV[mappingSize]] = false;
            }
        }
    }
    delete[] cnt;
    delete[] cntCopy;
}

