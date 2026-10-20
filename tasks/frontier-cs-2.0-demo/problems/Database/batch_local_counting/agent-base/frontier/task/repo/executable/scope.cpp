//
// Created by anonymous author on 2022/8/30.
//

#include "command.h"
#include "execution.h"

int main(int argc, char **argv) {
    Command cmd(argc, argv);
    std::string queryGraphPath = cmd.getQueryGraphPath();
    std::string dataGraphPath = cmd.getDataGraphPath();
    std::string resultPath = cmd.getResultPath();
    std::string trianglePath = cmd.getTrianglePath();
    bool batchQuery = cmd.getBatchQuery();
    bool useTriangle = !trianglePath.empty();
    std::cout << "query graph path: " << queryGraphPath << std::endl;
    std::cout << "data graph path: " << dataGraphPath << std::endl;
    std::cout << "result path: " << resultPath << std::endl;
    std::cout << "using batch query: " << batchQuery << std::endl;
    std::cout << "using triangle: " << useTriangle << std::endl;
    std::cout << "set intersection type: " << SI << std::endl;
    DataGraph dun = DataGraph();
    dun.loadDataGraph(dataGraphPath);
    const DataGraph din = constructDirectedDataGraph(dun, false);
    const DataGraph dout = constructDirectedDataGraph(dun, true);
    specialsparse *sg = (specialsparse *)malloc(sizeof(specialsparse));
    dout.initSpecialSparse(sg);
    ui n = dun.getNumVertices(), m = dun.getNumEdges();
    Triangle triangle;
    if (useTriangle) {
        triangle.load(trianglePath, m / 2);
    }
    auto start = std::chrono::steady_clock::now();
    EdgeID *outID = buildInID2OutID(din, dout);
    EdgeID *unID = buildUnID2OutID(dun, dout);
    EdgeID *reverseID = buildReverseUnID(dun);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsedSeconds = end - start;
    std::cout << "finished building edge ID mappings. time: " << elapsedSeconds.count() << "s" << std::endl;
    std::vector<std::string> files;
    std::vector<PatternGraph> patternGraphs = loadPatternGraph(queryGraphPath, batchQuery, files);
    int patternSize = patternGraphs[0].getNumVertices();
    HashTable factorSum = new Count[m + 1];
    HashTable ht[MAX_NUM_NODE];
    for (auto & h: ht)
        h = new Count[m + 1];
    ui *candPos = new ui[MAX_PATTERN_SIZE];
    memset(candPos, 0, sizeof(ui) * (MAX_PATTERN_SIZE));
    VertexID **patternV = new VertexID *[MAX_NUM_NODE];
    VertexID **dataV = new VertexID *[MAX_NUM_NODE];
    EdgeID **startOffset = new EdgeID *[MAX_NUM_NODE];
    bool **visited = new bool *[MAX_NUM_NODE];
    for (int i = 0; i < MAX_NUM_NODE; ++i) {
        patternV[i] = new VertexID[MAX_PATTERN_SIZE];
        dataV[i] = new VertexID[MAX_PATTERN_SIZE];
        startOffset[i] = new VertexID[MAX_PATTERN_SIZE];
        visited[i] = new bool[n];
        memset(visited[i], false, sizeof(bool) * n);
    }
    VertexID *tmp = new VertexID[n];
    VertexID *allV = new VertexID[n];
    for (VertexID i = 0; i < n; ++i) {
        allV[i] = i;
    }
    VertexID *cliqueVertices = new VertexID[MAX_PATTERN_SIZE + 1];
    std::set<CanonType> visitedNode;

    double totalPlanTime = 0.0, totalExeTime = 0.0;
    ui totalNumPatterns = 0, totalNodes = 0;
    double averageNodeSize = 0.0;
    int numVertexTable = 0, numEdgeTable = 0;
    if (true) {
        std::vector<HashTable> mathCalH(patternGraphs.size());
        std::vector<int> orbitTypes(patternGraphs.size());
#ifdef DEBUG
        HashTable hPattern = new Count[n];
        memset(hPattern, 0, sizeof(Count) * dun.getNumVertices());
#endif
        for (int i = 0; i < patternGraphs.size(); ++i) {
            double exeTime = 0.0;
            std::map<int, std::vector<Pattern>> patterns;
            std::map<int, std::vector<std::vector<Tree>>> trees;
            start = std::chrono::steady_clock::now();
            ConNode cn;
            if (!patternGraphs[i].isClique()) {
                genEquation(patternGraphs[i], patterns, trees, cn, useTriangle, true, true, true);
            }
            end = std::chrono::steady_clock::now();
            elapsedSeconds = end - start;
            totalPlanTime += elapsedSeconds.count();
            start = std::chrono::steady_clock::now();
            HashTable H;
            int orbitType = patternGraphs[i].getOrbitType();
            orbitTypes[i] = orbitType;
            if (orbitType == 0) {
                H = new Count[1];
                H[0] = 0;
            }
            else if (orbitType == 1) {
                H = new Count[dun.getNumVertices()];
                memset(H, 0, sizeof(Count) * n);
            }
            else {
                H = new Count[m];
                memset(H, 0, sizeof(Count) * m);
            }
            if (cn.num != 0) {
                Pattern p(patternGraphs[i]);
                // prepare hash tables for cn
                if (orbitType == 1) {
                    for (int j = 0; j < cn.aggrePos.size(); ++j)
                        cn.hashTables[j] = H;
                }
                else {
                    for (int j = 0; j < cn.aggrePos.size() / 2; ++j)
                        cn.hashTables[j] = H;
                }
                executeConNode(cn, din, dout, dun, useTriangle, triangle, p, outID, unID, startOffset[0], patternV[0],
                               dataV[0], visited[0], candPos, tmp, allV);
                if (!resultPath.empty()) {
                    int divideFactor = cn.divideFactor;
                    if (orbitType == 0) H[0] /= divideFactor;
                    else if (orbitType == 1)
                        for (VertexID l = 0; l < n; ++l)
                            H[l] /= divideFactor;
                    else {
                        for (EdgeID l = 0; l < m ; ++l)
                            H[l] /= divideFactor;
                    }
                }
            }
            if (patternGraphs[i].isClique()) {
                start = std::chrono::steady_clock::now();
                int k = patternGraphs[i].getNumVertices();
                mkspecial(sg, k);
                kclique(k, k, sg, cliqueVertices, H, orbitType);
                freesub(sg, k);
                end = std::chrono::steady_clock::now();
                elapsedSeconds = end - start;
                totalExeTime += elapsedSeconds.count();
            }
            for (auto it = patterns.begin(); it != patterns.end(); ++it) {
                int divideFactor = it->first;
                memset(factorSum, 0, sizeof(Count) * (m + 1));
                for (int j = 0; j < it->second.size(); ++j) {
                    for (int j2 = 0; j2 < trees[divideFactor][j].size(); ++j2) {
                        for (int l = 0; l < trees[divideFactor][j][0].getNumNodes(); ++l) {
                            memset(ht[l], 0, sizeof(Count) * m);
                        }
                        int k = patterns[divideFactor][j].u.getNumVertices();
                        if (patterns[divideFactor][j].u.isClique() && k >= 4) {
                            HashTable h = ht[trees[divideFactor][j][j2].getRootID()];
                            int aggreWeight = trees[divideFactor][j][j2].getAggreWeight()[0];
                            mkspecial(sg, k);
                            kclique(k, k, sg, cliqueVertices, h, orbitType);
                            freesub(sg, k);
                            if (aggreWeight != 1) {
                                if (orbitType == 0) h[0] *= aggreWeight;
                                else if (orbitType == 1) {
                                    for (VertexID v = 0; v < n; ++v)
                                        h[v] *= aggreWeight;
                                }
                                else {
                                    for (EdgeID e = 0; e < m; ++e)
                                        h[e] *= aggreWeight;
                                }
                            }
                        }
                        else {
                            const Tree &t = trees[divideFactor][j][j2];
                            if (t.getExecuteMode()) {
                                executeTree(t, din, dout, dun, useTriangle, triangle, patterns[divideFactor][j],
                                            ht, outID, unID, reverseID, startOffset[0], patternV[0], dataV[0], visited[0], candPos, tmp, allV);
                            }
                            else {
                                multiJoinTree(t, din, dout, dun, useTriangle, triangle, patterns[divideFactor][j],
                                              ht, outID, unID, reverseID, startOffset, patternV, dataV, visited, tmp, allV);
                            }
                        }
                        HashTable h = ht[trees[divideFactor][j][j2].getRootID()];
                        int multiFactor = trees[divideFactor][j][j2].getMultiFactor();
                        if (!resultPath.empty()) {
                            if (orbitType == 0) factorSum[0] += h[0];
                            else if (orbitType == 1)
                                for (VertexID l = 0; l < n; ++l) {
                                    factorSum[l] += h[l] * multiFactor;
                                }
                            else
                                for (EdgeID l = 0; l < m + 1; ++l) {
                                    factorSum[l] += h[l] * multiFactor;
                                }
                        }
#ifdef DEBUG
                        if (divideFactor == 2 && patterns[divideFactor][j].u.getCanonValue() == 281875) {
                        for (VertexID l = 0; l < n; ++l) {
                            hPattern[l] += h[l];
                        }
                    }
#endif
                        for (VertexID nID = 0; nID < trees[divideFactor][j][j2].getNumNodes(); ++nID) {
                            ++totalNodes;
                            averageNodeSize += trees[divideFactor][j][j2].getNode(nID).numVertices;
                            visitedNode.insert(trees[divideFactor][j][j2].getNode(nID).canonValue);
                            if (nID != trees[divideFactor][j][j2].getPostOrder().back()) {
                                if (trees[divideFactor][j][j2].getNode(nID).keySize == 1)
                                    ++numVertexTable;
                                else
                                    ++numEdgeTable;
                            }
                        }
                    }
                }
                if (!resultPath.empty()) {
                    if (orbitType == 0) H[0] += factorSum[0] / divideFactor;
                    else if (orbitType == 1)
                        for (VertexID l = 0; l < n; ++l) {
                            H[l] += factorSum[l] / divideFactor;
                        }
                    else
                        for (EdgeID l = 0; l < m + 1; ++l) {
                            H[l] += factorSum[l] / divideFactor;
                        }
                }
            }
            end = std::chrono::steady_clock::now();
            elapsedSeconds = end - start;
            exeTime = elapsedSeconds.count();
            if (batchQuery)
                std::cout << "file: " << files[i] << ", ";
            std::cout << "execution time: " << exeTime << "s";
            ui numPatterns = 0;
            for (auto it = patterns.begin(); it != patterns.end(); ++it)
                numPatterns += it->second.size();
            std::cout << ", number of patterns: " << numPatterns << std::endl;
            totalExeTime += exeTime;
            totalNumPatterns += numPatterns;
            mathCalH[i] = H;
        }
        if (!resultPath.empty()) saveCount(resultPath, mathCalH, dun, batchQuery, files, orbitTypes);
        for (int i = 0; i < patternGraphs.size(); ++i)
            delete[] mathCalH[i];
        std::cout << "compute equation time: " << gEquationTime << std::endl;
        std::cout << "compute tree decomposition: " << gTDTime << std::endl;
        std::cout << "compute symmetry time: " << gSymmTime << std::endl;
        std::cout << "compute attribute order: " << gOrderTime << std::endl;
        std::cout << "planning time: " << totalPlanTime << ", total execution time: "
                  << totalExeTime << ", total number of patterns: " << totalNumPatterns << "\ntotal number of nodes: "
                  << totalNodes << ", average size of nodes: "<< averageNodeSize / totalNodes
                  << ", number of distinct nodes: " << visitedNode.size() << std::endl
                  << "number of vertex hash tables (except for root node): " << numVertexTable
                  << ", number of edge hash tables (except for root node): " << numEdgeTable << std::endl;
        std::cout << "number of match: " << gNumMatch << ", number of intersect: " << gNumIntersect << std::endl;
    }

    return 0;
}
