//
// reordering the data graph based on core decomposition and store intersection results
//

#include "graph.h"
#include "triangle.h"

// refer to "An O(m) Algorithm for Cores Decomposition of Networks" https://arxiv.org/pdf/cs/0310049.pdf

VertexID *coreDecompOrdering(const DataGraph &g, const std::string &outPath) {
    /* vert contains the set of vertices, sorted by their degrees.
     * Positions of vertices in array vert are stored in array pos.
     * bin contains for each possible degree the position of the first vertex of that degree in vert
     * */
    VertexID n = g.getNumVertices();
    int *pos = new int[n + 1];
    int *bin = new int[n + 1];
    int *deg = new int[n + 1];
    int *vert = new int[n + 1];
    VertexID * old2new = new VertexID[n + 1];
    memset(pos, 0, sizeof(int) * (n + 1));
    memset(bin, 0, sizeof(int) *(n + 1));
    memset(deg, 0, sizeof(int) * (n + 1));
    memset(vert, 0, sizeof(int) * (n + 1));
    int maxDeg = 0;
    for (int v = 1; v <= n; ++v) {
        int degree = int(g.degree(v - 1));
        deg[v] = degree;
        if (degree > maxDeg) maxDeg = degree;
    }
    for (int v = 1; v <= n; ++v)
        ++bin[deg[v]];
    int start = 1;
    for (int d = 0; d <= maxDeg; ++d) {
        int num = bin[d];
        bin[d] = start;
        start += num;
    }
    for (int v = 1; v <= n; ++v) {
        pos[v] = bin[deg[v]];
        vert[pos[v]] = v;
        ++bin[deg[v]];
    }
    for (int d = maxDeg; d > 0; --d)
        bin[d] = bin[d-1];
    bin[0] = 1;
    int numZeroDegree = bin[1] - bin[0];
    for (int i = 1; i <= n; ++i) {
        VertexID v = vert[i];
        // now i is the new vertex id, v is the old id
        old2new[v - 1] = i - 1 - numZeroDegree;
        ui nbrCnt = 0;
        VertexID *neighbors = g.getNeighbors(v - 1, nbrCnt);
        for (ui j = 0; j < nbrCnt; ++j) {
            VertexID u = neighbors[j] + 1;
            if (deg[u] > deg[v]) {
                int du = deg[u], pu = pos[u], pw = bin[du], w = vert[pw];
                if(u != w) {
                    pos[u] = pw;
                    vert[pu] = w;
                    pos[w] = pu;
                    vert[pw] = int(u);
                }
                ++bin[du];
                --deg[u];
            }
        }
    }
    // debug: now deg[] stores core number of vertices
//    std::string file = "../../exp/core.txt";
//    int degeneracy = deg[vert[n - 1]];
//    std::vector<VertexID> *core2v = new std::vector<VertexID>[degeneracy + 1];
//    for (VertexID v = 1; v <= n; ++v) {
//        int core = deg[v];
//        core2v[core].push_back(v - 1);
//    }
//    std::ofstream out(file);
//    out << "v \t core\n";
//    for (int i = 1; i <= degeneracy; ++i) {
//        for (int j = 0; j < core2v[i].size(); ++j)
//            out << core2v[i][j] << "\t" << i << std::endl;
//    }
//    out.close();
    delete[] pos;
    delete[] bin;
    delete[] deg;
    delete[] vert;

    return old2new;
}

void reorderAndStore(const DataGraph &g, const std::string &graphPath, const std::string &trianglePath,
                     const VertexID *old2new) {
    VertexID n = g.getNumVertices();
    EdgeID m = g.getNumEdges() / 2;
    std::vector<std::vector<VertexID>> adjList(n);
    Edge *edgeList = new Edge[m * 2];
    EdgeID edgeCnt = 0;
    for (VertexID u = 0; u < n; ++u) {
        ui nbrCnt = 0;
        VertexID *outNeighbors = g.getNeighborsLargerID(u, nbrCnt);
        for (EdgeID j = 0; j < nbrCnt; ++j) {
            VertexID v = outNeighbors[j];
            VertexID uNew = old2new[u], vNew = old2new[v];
            if (uNew < vNew)
                adjList[uNew].push_back(vNew);
            else
                adjList[vNew].push_back(uNew);
        }
    }

    std::ofstream out(graphPath);
    ui maxOutDegree = 0;
    out << n << " " << m << std::endl;
    for (int i = 0; i < n; ++i) {
        std::sort(adjList[i].begin(), adjList[i].end());
        if (adjList[i].size() > maxOutDegree) maxOutDegree = adjList[i].size();
        for (int j = 0; j < adjList[i].size(); ++j) {
            out << i << " " << adjList[i][j] << std::endl;
            edgeList[edgeCnt] = std::make_pair(i, adjList[i][j]);
            ++edgeCnt;
            edgeList[edgeCnt] = std::make_pair(adjList[i][j], i);
            ++edgeCnt;
        }
    }
    std::cout << "max out degree: " << maxOutDegree << std::endl;
    out.close();
    DataGraph d(n, m * 2);
    d.addDirectedEdges(edgeList, edgeCnt);
    d.buildLargerOffset();
    DataGraph dout = constructDirectedDataGraph(d, true);
    DataGraph din = constructDirectedDataGraph(d, false);
    Triangle t;
    t.EnumTriangle(din, dout);
    t.save(trianglePath, m);
}

int main(int argc, char** argv) {
    DataGraph g = DataGraph();
    if (argc < 3) {
        std::cout << "<raw graph path> <graph output path> <triangle output path>" << std::endl;
        exit(1);
    }
    std::string input_file_path(argv[1]);
    std::string output_file_path(argv[2]);
    std::string triangle_file_path(argv[3]);
    g.loadDataGraph(input_file_path);
    VertexID *old2new = coreDecompOrdering(g, output_file_path);
    reorderAndStore(g, output_file_path, triangle_file_path, old2new);
    delete[] old2new;
    return 0;
}
