//
// Created by anonymous author on 2022/7/19.
//

#ifndef SCOPE_GRAPH_H
#define SCOPE_GRAPH_H

#include "config.h"
#include "utils.h"
#include "clique.h"
#include "glpk.h"
extern "C" {
    #include "automorphism/nauty.h"
    #include "automorphism/gtools.h"
    #include "automorphism/naugroup.h"
};

#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <map>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <dirent.h>
#include <functional>

class Graph {
protected:
    ui _numVertices;           // number of vertices in the graph
    ui _numEdges;              // number of directed edges in the graph.
                               // undirected graph has _numEdges / 2 undirected edges.
    EdgeID *_offsets;          // vertex v's edges are [offset[v], offset[v + 1])
    VertexID *_nbors;          // incoming or outgoing neighbors
    EdgeID *_degree;

public:
    Graph();

    Graph(const Graph& rhs);

    Graph(ui numVertices, ui numEdges);

    Graph& operator=(const Graph& rhs);

    virtual ~Graph();

    inline ui getNumVertices() const {
        return _numVertices;
    }

    inline ui getNumEdges() const {
        return _numEdges;
    }

    inline VertexID *getNeighbors(VertexID v, ui &count) const {
        count = _offsets[v + 1] - _offsets[v];
        return _nbors + _offsets[v];
    }

    inline ui degree(VertexID v) const {
        return _offsets[v + 1] - _offsets[v];
    }

    int getEdgeID(VertexID v, VertexID w) const;
    int getUndirectedEID(VertexID v, VertexID w) const;
    void addDirectedEdges(const Edge *edgeList, ui num);

    EdgeID *getOffsets() const {
        return _offsets;
    }

    VertexID *getNbors() const {
        return _nbors;
    }
};

class DataGraph : public Graph {
private:
    EdgeID *_largerOffsets;    // used for symmetry breaking, the offset of neighbors larger than itself

public:
    DataGraph();

    DataGraph(ui numVertices, ui numEdges);

    DataGraph& operator=(const DataGraph& rhs);

    ~DataGraph() override;
    inline VertexID *getNeighborsLargerID(VertexID u, ui &count) const {
        count = _offsets[u + 1] - _offsets[u] - _largerOffsets[u];
        return _nbors + _offsets[u] + _largerOffsets[u];
    }
    void loadDataGraph(const std::string &file);
    void buildLargerOffset();
    void initSpecialSparse(specialsparse *sg) const;
};

class PatternGraph : public Graph {
private:
    bool **_adjMatrix;
    VertexID *_aggreV;              // aggregation vertices / edges. Note that for shrinkage patterns,
                                    // there could be multiple orbits that needs to be aggregated.
    int *_aggreWeight;              // the weight for each aggregation vertex, 1 by default.
                                    // vertices in the same orbit always has the same weight.
    ui _aggreSize;
    int _orbitType;                 // 0: global counting. 1: vertex orbit. 2: edge orbit
    bool _eOrbitDir;                // used for edge orbit. true: directed edge orbit false: undirected
    VertexID *_coreV;               // vertices with core number \geq 2
    VertexID *_peripheralV;         // other vertices
    ui _coreSize;
    CanonType _canonValue;          // the canonical value of the induced subgraph
    int _autoSize;                  // number of automorphisms
    int _divideFactor;              // _autoSize / auto group size of vertex 0 / edge (0,1)
    int *_v2o;                      // the canonical orbits of vertices in this node
    VertexID *_v2l;                 // the canonical labeling of vertices in this node
    std::vector<std::vector<std::vector<VertexID>>> _candidateRules;

public:
    PatternGraph();

    PatternGraph(ui numVertices, ui numEdges, const Edge *edgeList);

    PatternGraph(const PatternGraph & rhs);

    PatternGraph& operator=(const PatternGraph &rhs);

    ~PatternGraph() override;

    inline bool isEdge(VertexID u1, VertexID u2) const {
        return _adjMatrix[u1][u2];
    }

    // remark. only call this in genDAGs.
    inline void setAggreInfo(VertexID *aggreV, ui aggreSize, int orbitType) {
        _aggreSize = aggreSize;
        _orbitType = orbitType;
        if (orbitType == 0) return;
        _aggreV = new VertexID[aggreSize];
        memcpy(_aggreV, aggreV, sizeof(VertexID) * aggreSize);
        _aggreWeight = new int[aggreSize];
        for (int i = 0; i < aggreSize; ++i)
            _aggreWeight[i] = 1;
    }

    void setCorePeripheral(VertexID *coreV, VertexID *peripheralV, ui coreSize) {
        _coreSize = coreSize;
        _coreV = new VertexID[coreSize];
        memcpy(_coreV, coreV, sizeof(VertexID) * coreSize);
        ui peripheralSize = _numVertices - coreSize;
        _peripheralV = new VertexID[peripheralSize];
        memcpy(_peripheralV, peripheralV, sizeof(VertexID) * peripheralSize);
    }

    inline VertexID *getPeripheralV() const {
        return _peripheralV;
    }

    inline VertexID *getCoreV(ui &coreSize) const {
        coreSize = _coreSize;
        return _coreV;
    }

    inline VertexID *getAggreV(ui &aggreSize) const {
        aggreSize = _aggreSize;
        return _aggreV;
    }

    inline ui getAggreSize() const {
        return _aggreSize;
    }

    inline int *getAggreWeight() const {
        return _aggreWeight;
    }

    inline VertexID *getV2L() const {
        return _v2l;
    }

    inline int getOrbitType() const {
        return _orbitType;
    }

    inline int getOrbit(VertexID u) const {
        return _v2o[u];
    }

    inline int getAutoSize() const {
        return _autoSize;
    }

    inline int getDivideFactor() const {
        return _divideFactor;
    }

    inline CanonType getCanonValue() const {
        return _canonValue;
    }

    inline bool isSingleAggre() const {
        if (_orbitType == 1)
            return _aggreSize == 1;
        else
            return _aggreSize == 2;
    }

    bool isEOrbitDir() const {
        return _eOrbitDir;
    }

    const std::vector<std::vector<std::vector<VertexID>>> &getCandidateRules() const {
        return _candidateRules;
    }

    friend class Tree;
    friend class ConNode;

    void loadPatternGraph(const std::string &file);
    void addEdgeList(const Edge *edgeList, ui num);
    void computeAutoGroup();
    void computeCandidateRules();
    void setSingleAggre();
    void setMultiAggre();
    void genDAGs(std::vector<PatternGraph> &din, std::vector<PatternGraph> &dout) const;
    void buildCorePeripheral();
    Edge *coreUndirectedEdges(ui &num) const;
    PatternGraph shrink(VertexID u, VertexID v) const;
    void printGraph(bool direction = true) const;
    bool isClique() const;
    VertexID *getMultiAggreV(ui &aggreSize) const;
};

typedef struct Pattern {
    PatternGraph u;
    PatternGraph in;
    PatternGraph out;

    explicit Pattern(const PatternGraph &u) : u(u)  {
        in = PatternGraph();
        out = PatternGraph();
    }

    Pattern() = default;

    virtual ~Pattern() = default;

    Pattern(const PatternGraph &u, const PatternGraph &in, const PatternGraph &out) : u(u), in(in), out(out) {}

    Pattern &operator=(const Pattern &rhs) {
        if (this == &rhs)
            return *this;
        u = rhs.u;
        in = rhs.in;
        out = rhs.out;
        return *this;
    }

    inline bool useDAG() const {
        return in.getNumVertices() != 0;
    }

    inline bool empty() const {
        return u.getNumVertices() == 0;
    }

    Pattern shrink(VertexID i, VertexID j, std::vector<VertexID> &old2New) const;
    Pattern shrink(const std::vector<VertexID> &partition) const;
    void setSingleAggre();
    void setMultiAggre();
}Pattern;

// adapted from utility/automorphism/directg.c
#define MAXNV 128
#define MAXMV SETWORDSNEEDED(MAXNV)
#define MAXNE 1024
static int _v0[MAXNE],_v1[MAXNE];
static int edgeno[MAXNV][MAXNV];

#define MAXME SETWORDSNEEDED(2*MAXNE)

static set x[MAXME];
static int ix[2*MAXNE],nix;
static boolean first;
static int me;
static int lastreject[MAXNV];
static boolean lastrejok;
static int rejectlevel;
static unsigned long groupsize;
static unsigned long newgroupsize;
static boolean Gswitch,Vswitch,ntgroup,ntisol;

static int splitlevel,splitmod,splitres,splitcount;
static unsigned long splitcases;
static boolean ismax(int *p, int n);
static void testmax(int *p, int n, int *abort);
static int trythisone(grouprec *group, int ne, int m, int n, std::vector<PatternGraph> &pin, std::vector<PatternGraph> &pout);
static void updatetc(graph *oldtc, graph *newtc, int v, int w, int m, int n);
static void updatetc1(graph *oldtc, graph *newtc, int v, int w, int n);
static int scan_acyclic(int level, int ne, int minarcs, int maxarcs, int sofar,
                        graph *oldtc, grouprec *group, int m, int n, std::vector<PatternGraph> &pin, std::vector<PatternGraph> &pout);
static int scan_acyclic1(int level, int ne, int minarcs, int maxarcs, int sofar,
                         graph *oldtc, grouprec *group, int m, int n, std::vector<PatternGraph> &pin, std::vector<PatternGraph> &pout);
static int scan(int level, int ne, int minarcs, int maxarcs, int sofar,
                boolean oriented, grouprec *group, int m, int n, std::vector<PatternGraph> &pin, std::vector<PatternGraph> &pout);
static void direct(graph *g, int nfixed, long minarcs, long maxarcs, int m, int n, std::vector<PatternGraph> &pin, std::vector<PatternGraph> &pout);
DataGraph constructDirectedDataGraph(const DataGraph &g, bool outDag);
std::vector<PatternGraph> loadPatternGraph(const std::string &path, bool batchQuery, std::vector<std::string> &files);
bool checkMapping(const PatternGraph &p1, const PatternGraph &p2, const std::vector<VertexID> &v2, VertexID *mapping);
bool checkMapping(const PatternGraph &p1, const PatternGraph &p2, const std::vector<VertexID> &v1, const std::vector<VertexID> &v2);
bool checkMapping(const Pattern &p1, const Pattern &p2, const std::vector<VertexID> &v2, VertexID *mapping);
bool checkMapping(const Pattern &p1, const Pattern &p2, const std::vector<VertexID> &v1, const std::vector<VertexID> &v2);
CanonType subgraphCanonValue(const PatternGraph &p, const std::vector<VertexID> &v, int *v2o = nullptr);
bool isSubgraphAutoGroup(const PatternGraph &p, const std::vector<VertexID> &v, const std::vector<VertexID> &group);
EdgeID *buildInID2OutID(const DataGraph &din, const DataGraph &dout);
EdgeID *buildUnID2OutID(const DataGraph &dun, const DataGraph &dout);
EdgeID *buildReverseUnID(const DataGraph &dun);
int numDAG(const PatternGraph &p, const std::vector<VertexID> &v);
bool isConnectedSubgraph(std::vector<VertexID> vertices, const PatternGraph &p);

#endif //SCOPE_GRAPH_H
