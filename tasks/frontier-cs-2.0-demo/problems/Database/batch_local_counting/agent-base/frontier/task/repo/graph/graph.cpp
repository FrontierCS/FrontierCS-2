//
// Created by anonymous author on 2022/7/19.
//

#include "graph.h"

Graph::Graph() {
    _numVertices = 0;
    _numEdges = 0;
    _offsets = nullptr;
    _nbors = nullptr;
    _degree = nullptr;
}

Graph::Graph(ui numVertices, ui numEdges) : _numVertices(numVertices), _numEdges(numEdges) {
    _offsets = new VertexID[_numVertices + 1];
    _nbors = new VertexID[_numEdges];
    _degree = new VertexID[_numVertices];
    memset(_degree, 0, sizeof(VertexID) * _numVertices);
}

Graph::Graph(const Graph& rhs) {
    _numVertices = rhs._numVertices;
    _numEdges = rhs._numEdges;
    if (rhs._offsets == nullptr) _offsets = nullptr;
    else {
        _offsets = new EdgeID[_numVertices + 1];
        memcpy(_offsets, rhs._offsets, sizeof(EdgeID) * (_numVertices + 1));
    }
    if (rhs._nbors == nullptr) _nbors = nullptr;
    else {
        _nbors = new VertexID[_numEdges];
        memcpy(_nbors, rhs._nbors, sizeof(VertexID) * _numEdges);
    }
    if (rhs._degree == nullptr) _degree = nullptr;
    else {
        _degree = new VertexID[_numVertices];
        memcpy(_degree, rhs._degree, sizeof(VertexID) * _numVertices);
    }
}

Graph &Graph::operator=(const Graph &rhs) {
    if (this == &rhs)
        return *this;
    _numVertices = rhs._numVertices;
    _numEdges = rhs._numEdges;
    if (rhs._offsets == nullptr) _offsets = nullptr;
    else {
        delete[] _offsets;
        _offsets = new EdgeID[_numVertices + 1];
        memcpy(_offsets, rhs._offsets, sizeof(EdgeID) * (_numVertices + 1));
    }
    if (rhs._nbors == nullptr) _nbors = nullptr;
    else {
        delete[] _nbors;
        _nbors = new VertexID[_numEdges];
        memcpy(_nbors, rhs._nbors, sizeof(VertexID) * _numEdges);
    }
    if (rhs._degree == nullptr) _degree = nullptr;
    else {
        delete[] _degree;
        _degree = new VertexID[_numVertices];
        memcpy(_degree, rhs._degree, sizeof(VertexID) * _numVertices);
    }

    return *this;
}

Graph::~Graph() {
    delete[] _offsets;
    delete[] _nbors;
    delete[] _degree;
}

int Graph::getEdgeID(VertexID v, VertexID w) const {
    if (v > w) std::swap(v, w);
    int low = (int)_offsets[v];
    int high = (int)_offsets[v + 1] - 1;
    int mid;

    while (high - low >= 16) {
        mid = low + ((high - low) >> 1);
#ifndef __APPLE__
        _mm_prefetch((char *) &_nbors[(mid + 1 + high) / 2], _MM_HINT_T0);
        _mm_prefetch((char *) &_nbors[(mid - 1 + low) / 2], _MM_HINT_T0);
#endif
        if (_nbors[mid] == w) return mid;
        if (_nbors[mid] > w) high = mid - 1;
        else low = mid + 1;
    }

    for (int i = low; i <= high; ++i) {
        if (_nbors[i] == w) return i;
    }

    return -1;
}

int Graph::getUndirectedEID(VertexID v, VertexID w) const {
    int low = (int)_offsets[v];
    int high = (int)_offsets[v + 1] - 1;
    int mid;

    while (high - low >= 16) {
        mid = low + ((high - low) >> 1);
#ifndef __APPLE__
        _mm_prefetch((char *) &_nbors[(mid + 1 + high) / 2], _MM_HINT_T0);
        _mm_prefetch((char *) &_nbors[(mid - 1 + low) / 2], _MM_HINT_T0);
#endif
        if (_nbors[mid] == w) return mid;
        if (_nbors[mid] > w) high = mid - 1;
        else low = mid + 1;
    }

    for (int i = low; i <= high; ++i) {
        if (_nbors[i] == w) return i;
    }

    return -1;
}

void Graph::addDirectedEdges(const Edge *edgeList, ui num) {
    // build CSR graph from the edge list
    delete[] _offsets;
    _offsets = new EdgeID[_numVertices + 1];
    _offsets[0] = 0;
    delete[] _nbors;
    _nbors = new VertexID[num];
    delete[] _degree;
    _degree = new VertexID[_numVertices];
    memset(_degree, 0, sizeof(VertexID) * _numVertices);
    VertexID *nborsOffset = new VertexID[_numVertices + 1];
    memset(nborsOffset, 0, sizeof(VertexID) * (_numVertices + 1));
    for (int i = 0; i < num; ++i) {
        VertexID u1 = edgeList[i].first;
        ++_degree[u1];
    }
    for (int i = 0; i < _numVertices; ++i) {
        _offsets[i + 1] = _offsets[i] + _degree[i];
    }
    for (int i = 0; i < num; ++i) {
        VertexID u1 = edgeList[i].first, u2 = edgeList[i].second;
        EdgeID offset1 = _offsets[u1] + nborsOffset[u1];
        _nbors[offset1] = u2;
        ++nborsOffset[u1];
    }
    // sort neighbors in ascending order of vertex id
    for (int i = 0; i < _numVertices; ++i)
        std::sort(_nbors + _offsets[i], _nbors + _offsets[i + 1]);

    delete[] nborsOffset;
}

void DataGraph::loadDataGraph(const std::string &file) {
    std::ifstream inFile(file);
    if (!inFile.is_open()) {
        std::cout << "Can not open file " << file << "!" << std::endl;
        exit(1);
    }
    // get the number of vertices and edges from the graph file
    inFile >> _numVertices >> _numEdges;
    _numEdges *= 2;
    _offsets = new EdgeID[_numVertices + 1];
    _offsets[0] = 0;
    _nbors = new VertexID[_numEdges];
    _degree = new VertexID[_numVertices];
    memset(_degree, 0, sizeof(VertexID) * _numVertices);
    // read edge list
    EdgeID edgeCnt = 0;
    Edge *edgeList = new Edge[_numEdges];
    for (ui i = 0; i < _numEdges / 2; i++) {
        VertexID _v1, _v2;
        if (inFile >> _v1 >> _v2) {
            edgeList[edgeCnt] = std::make_pair(_v1, _v2);
            ++edgeCnt;
            edgeList[edgeCnt] = std::make_pair(_v2, _v1);
            ++edgeCnt;
        }
        else {
            printf("error: expecting %d edges, getting %d edges.\n", _numEdges / 2, edgeCnt / 2);
            exit(1);
        }
    }
    inFile.close();
    addDirectedEdges(edgeList, _numEdges);
    // build larger offset
    buildLargerOffset();

    delete[] edgeList;
}

DataGraph::DataGraph() {
    _largerOffsets = nullptr;
}

DataGraph::~DataGraph() {
    delete[] _largerOffsets;
}

DataGraph::DataGraph(ui numVertices, ui numEdges) : Graph(numVertices, numEdges) {
    _largerOffsets = nullptr;
}

DataGraph& DataGraph::operator=(const DataGraph& rhs) {
    if (this == &rhs)
        return *this;
    Graph::operator=(rhs);
    delete[] _largerOffsets;
    _largerOffsets = new EdgeID[_numVertices + 1];
    memcpy(_largerOffsets, rhs._offsets, sizeof(EdgeID) * (_numVertices + 1));

    return *this;
}

void DataGraph::buildLargerOffset() {
    _largerOffsets = new EdgeID[_numVertices];
    for (VertexID i = 0; i < _numVertices; ++i) {
        ui temp;
        VertexID *neighbors = getNeighbors(i, temp);
        int begin = 0, end = (int)temp - 1;
        while (begin <= end) {
            int mid = begin + ((end - begin) >> 1);
            if (neighbors[mid] > i) {
                end = mid - 1;
            }
            else {
                begin = mid + 1;
            }
        }
        _largerOffsets[i] = (EdgeID)begin;
    }
}

void DataGraph::initSpecialSparse(specialsparse *sg) const {
    sg->n = _numVertices;
    sg->e = _numEdges;
    sg->edges = (edge *)malloc(sg->e * sizeof(edge));
    for (VertexID u = 0; u < _numVertices; ++u) {
        for (EdgeID e = _offsets[u]; e < _offsets[u + 1]; ++e) {
            VertexID w = _nbors[e];
            sg->edges[e].s = u;
            sg->edges[e].t = w;
        }
    }
}

PatternGraph::PatternGraph() {
    _adjMatrix = nullptr;
    _aggreV = nullptr;
    _aggreWeight = nullptr;
    _aggreSize = 0;
    _orbitType = 0;
    _eOrbitDir = false;
    _coreV = nullptr;
    _peripheralV = nullptr;
    _coreSize = 0;
    _canonValue = 0;
    _autoSize = 0;
    _divideFactor = 1;
    _v2o = nullptr;
    _v2l = nullptr;
}

PatternGraph::PatternGraph(ui numVertices, ui numEdges, const Edge *edgeList) : Graph(
        numVertices, numEdges) {
    _adjMatrix = new bool*[_numVertices];
    _degree = new VertexID[_numVertices];
    memset(_degree, 0, sizeof(VertexID) * _numVertices);
    for (int i = 0; i < _numVertices; ++i) {
        _adjMatrix[i] = new bool[_numVertices];
        memset(_adjMatrix[i], false, sizeof(bool) * (_numVertices));
    }
    _aggreV = nullptr;
    _aggreWeight = nullptr;
    _aggreSize = 0;
    _orbitType = 0;
    _eOrbitDir = false;
    addEdgeList(edgeList, _numEdges);
    _coreV = nullptr;
    _peripheralV = nullptr;
    _coreSize = 0;
    _canonValue = 0;
    _autoSize = 0;
    _divideFactor = 1;
    _v2o = nullptr;
    _v2l = nullptr;
}

PatternGraph::PatternGraph(const PatternGraph & rhs) : Graph(rhs) {
    _aggreSize = rhs._aggreSize;
    _orbitType = rhs._orbitType;
    _coreSize = rhs._coreSize;
    _canonValue = rhs._canonValue;
    _autoSize = rhs._autoSize;
    _divideFactor = rhs._divideFactor;
    _eOrbitDir = rhs._eOrbitDir;
    _candidateRules = rhs._candidateRules;
    if (rhs._adjMatrix == nullptr)
        _adjMatrix = nullptr;
    else {
        _adjMatrix = new bool*[_numVertices];
        for (int i = 0; i < _numVertices; ++i) {
            _adjMatrix[i] = new bool[_numVertices];
            memcpy(_adjMatrix[i], rhs._adjMatrix[i], sizeof(bool) * (_numVertices));
        }
    }
    if (rhs._aggreV == nullptr)
        _aggreV = nullptr;
    else {
        _aggreV = new VertexID[_aggreSize];
        memcpy(_aggreV, rhs._aggreV, sizeof(VertexID) * _aggreSize);
    }
    if (rhs._aggreWeight == nullptr)
        _aggreWeight = nullptr;
    else {
        _aggreWeight = new int[_aggreSize];
        memcpy(_aggreWeight, rhs._aggreWeight, sizeof(int) * _aggreSize);
    }
    if (rhs._coreV == nullptr)
        _coreV = nullptr;
    else {
        _coreV = new VertexID[_coreSize];
        memcpy(_coreV, rhs._coreV, sizeof(VertexID) * _coreSize);
    }
    if (rhs._peripheralV == nullptr)
        _peripheralV = nullptr;
    else {
        ui peripheralSize = _numVertices - _coreSize;
        _peripheralV = new VertexID[peripheralSize];
        memcpy(_peripheralV, rhs._peripheralV, sizeof(VertexID) * peripheralSize);
    }
    if (rhs._v2o == nullptr)
        _v2o = nullptr;
    else {
        _v2o = new int[_numVertices];
        memcpy(_v2o, rhs._v2o, sizeof(int) * _numVertices);
    }
    if (rhs._v2l == nullptr)
        _v2l = nullptr;
    else {
        _v2l = new VertexID[_numVertices];
        memcpy(_v2l, rhs._v2l, sizeof(int) * _numVertices);
    }
}

PatternGraph& PatternGraph::operator=(const PatternGraph &rhs) {
    if (this == &rhs)
        return *this;
    Graph::operator=(rhs);
    _aggreSize = rhs._aggreSize;
    _orbitType = rhs._orbitType;
    _coreSize = rhs._coreSize;
    _canonValue = rhs._canonValue;
    _autoSize = rhs._autoSize;
    _divideFactor = rhs._divideFactor;
    _eOrbitDir = rhs._eOrbitDir;
    _candidateRules = rhs._candidateRules;
    if (rhs._adjMatrix == nullptr)
        _adjMatrix = nullptr;
    else {
        if (_adjMatrix) {
            for (int i = 0; i < _numVertices; ++i) {
                delete[] _adjMatrix[i];
            }
            delete[] _adjMatrix;
        }
        _adjMatrix = new bool*[_numVertices];
        for (int i = 0; i < _numVertices; ++i) {
            _adjMatrix[i] = new bool[_numVertices];
            memcpy(_adjMatrix[i], rhs._adjMatrix[i], sizeof(bool) * (_numVertices));
        }
    }
    if (rhs._aggreV == nullptr)
        _aggreV = nullptr;
    else {
        delete[] _aggreV;
        _aggreV = new VertexID[_aggreSize];
        memcpy(_aggreV, rhs._aggreV, sizeof(VertexID) * _aggreSize);
    }
    if (rhs._aggreWeight == nullptr)
        _aggreWeight = nullptr;
    else {
        delete[] _aggreWeight;
        _aggreWeight = new int[_aggreSize];
        memcpy(_aggreWeight, rhs._aggreWeight, sizeof(int) * _aggreSize);
    }
    if (rhs._coreV == nullptr)
        _coreV = nullptr;
    else {
        delete _coreV;
        _coreV = new VertexID[_coreSize];
        memcpy(_coreV, rhs._coreV, sizeof(VertexID) * _coreSize);
    }
    if (rhs._peripheralV == nullptr)
        _peripheralV = nullptr;
    else {
        delete _peripheralV;
        ui peripheralSize = _numVertices - _coreSize;
        _peripheralV = new VertexID[peripheralSize];
        memcpy(_peripheralV, rhs._peripheralV, sizeof(VertexID) * peripheralSize);
    }
    if (rhs._v2o == nullptr)
        _v2o = nullptr;
    else {
        delete[] _v2o;
        _v2o = new int[_numVertices];
        memcpy(_v2o, rhs._v2o, sizeof(int) * _numVertices);
    }
    if (rhs._v2l == nullptr)
        _v2l = nullptr;
    else {
        delete[] _v2l;
        _v2l = new VertexID[_numVertices];
        memcpy(_v2l, rhs._v2l, sizeof(int) * _numVertices);
    }
    return *this;
}

PatternGraph::~PatternGraph() {
    for (int i = 0; i < _numVertices; ++i) {
        delete[] _adjMatrix[i];
    }
    delete[] _adjMatrix;
    delete[] _aggreV;
    delete[] _aggreWeight;
    delete[] _coreV;
    delete[] _peripheralV;
    delete[] _v2o;
    delete[] _v2l;
}

void PatternGraph::loadPatternGraph(const std::string &file) {
    std::ifstream inFile(file);
    if (!inFile.is_open()) {
        std::cout << "Can not open file " << file << "!" << std::endl;
        exit(1);
    }
    // get the number of vertices and edges from the graph file
    inFile >> _numVertices >> _numEdges;
    _numEdges *= 2;
    // read edge list
    EdgeID edgeCnt = 0;
    Edge *edgeList = new Edge[_numEdges];
    for (ui i = 0; i < _numEdges / 2; i++) {
        VertexID u1, u2;
        if (inFile >> u1 >> u2) {
            edgeList[edgeCnt] = std::make_pair(u1, u2);
            ++edgeCnt;
            edgeList[edgeCnt] = std::make_pair(u2, u1);
            ++edgeCnt;
        }
        else {
            printf("error: expecting %d edges, getting %d edges.\n", _numEdges / 2, edgeCnt / 2);
            exit(1);
        }
    }

    // read the orbit
    VertexID *mapping = new VertexID[_numVertices];
    for (int i = 0; i < _numVertices; ++i) {
        mapping[i] = i;
    }
    VertexID vOrbit, eOrbit1, eOrbit2;
    if (inFile >> _orbitType) {
        // vertex orbit
        if (_orbitType == 1) {
            if (inFile >> vOrbit) {
                // relabel vertices: let the orbit vertex be 0.
                for (int i = 0; i < _numVertices; ++i) {
                    if (i == 0) mapping[i] = vOrbit;
                    else if (i == vOrbit) mapping[i] = 0;
                    else mapping[i] = i;
                }
            }
            else {
                printf("error: expecting a vertex orbit.\n");
                exit(1);
            }
        }
            // edge orbit
        else if (_orbitType == 2) {
            if (inFile >> eOrbit1 >> eOrbit2) {
                // relabel vertices: let the orbit edge be 0, 1.
                for (int i = 0; i < _numVertices; ++i) {
                    if (i == eOrbit1) mapping[i] = 0;
                    else if (i == eOrbit2) mapping[i] = 1;
                }
                int i = 2, j = 0;
                while (i < _numVertices && j < _numVertices) {
                    if (j != eOrbit1 && j != eOrbit2) {
                        mapping[j] = i;
                        ++i;
                    }
                    ++j;
                }
            }
            else {
                printf("error: expecting an edge orbit.\n");
                exit(1);
            }
        }
        else {
            printf("Scope currently supports vertex or edge orbit.\n");
            exit(1);
        }
    }
    // else, the file does not give the orbit, so we do the global counting.
    // relabel the edge list according to mapping
    for (ui i = 0; i < _numEdges; ++i) {
        VertexID u1 = edgeList[i].first, u2 = edgeList[i].second;
        edgeList[i].first = mapping[u1];
        edgeList[i].second = mapping[u2];
    }
    addEdgeList(edgeList,  _numEdges);
    if (_orbitType == 2 && !(isEdge(mapping[eOrbit1], mapping[eOrbit2]))) {
        printf("error: expecting an edge orbit.\n");
        exit(1);
    }
    _aggreSize = 0;
    computeAutoGroup();
    computeCandidateRules();
    if (_orbitType == 1) {
        _aggreV = new VertexID[1];
        _aggreV[0] = 0;
        _aggreSize = 1;
    }
    if (_orbitType == 2) {
        _aggreV = new VertexID[2];
        _aggreV[0] = 0;
        _aggreV[1] = 1;
        _aggreSize = 2;
    }
    _aggreWeight = new int[1];
    _aggreWeight[0] = 1;
    if (_orbitType == 2 && _v2o[0] == _v2o[1]) _eOrbitDir = true;
    else _eOrbitDir = false;
    buildCorePeripheral();
    inFile.close();
    delete[] mapping;
    delete[] edgeList;
}

// should call the constructor with numVertices and numEdges before using this function.
// add a set of directed edges to an empty pattern graph.
void PatternGraph::addEdgeList(const Edge *edgeList, ui num) {
    addDirectedEdges(edgeList, num);
    _adjMatrix = new bool*[_numVertices];
    for (int i = 0; i < _numVertices; ++i) {
        _adjMatrix[i] = new bool[_numVertices];
        memset(_adjMatrix[i], false, sizeof(bool) * (_numVertices));
    }
    for (int i = 0; i < num; ++i) {
        VertexID u1 = edgeList[i].first, u2 = edgeList[i].second;
        _adjMatrix[u1][u2] = true;
    }
}

/*
 * Compute canonical orbits, canonical value and automorphism group size
 * The canonical value corresponds to the graph colored by the query orbit
 * */
void PatternGraph::computeAutoGroup() {
    DYNALLSTAT(graph,g,g_sz);
    DYNALLSTAT(int,lab,lab_sz);
    DYNALLSTAT(int,ptn,ptn_sz);
    DYNALLSTAT(int,orbits,orbits_sz);
    DYNALLSTAT(graph,cg,cg_sz);
    static DEFAULTOPTIONS_GRAPH(options);
    options.getcanon = TRUE;
    options.digraph = TRUE;
    options.defaultptn = FALSE;
    statsblk stats;
    int n = int(_numVertices);
    int m = SETWORDSNEEDED(n);
    nauty_check(WORDSIZE,m,n,NAUTYVERSIONID);
    DYNALLOC2(graph,g,g_sz,m,n,"malloc");
    DYNALLOC2(graph,cg,cg_sz,m,n,"malloc");
    DYNALLOC1(int,lab,lab_sz,n,"malloc");
    DYNALLOC1(int,ptn,ptn_sz,n,"malloc");
    DYNALLOC1(int,orbits,orbits_sz,n,"malloc");
    EMPTYGRAPH(g,m,n);
    // treat pattern as directed
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (isEdge(i, j)) {
                        ADDONEARC(g, i, j, m);
            }
        }
    }
    for (int i = 0; i < n; ++i)
        lab[i] = i;
    for (int i = 0; i < n - 1; ++i)
        ptn[i] = 1;
    ptn[n - 1] = 0;
    densenauty(g,lab,ptn,orbits,&options,&stats,m,n,cg);
    std::map<VertexID, int> canonV2O;
    delete _v2l;
    _v2l = new VertexID[_numVertices];
    for (int i = 0; i < _numVertices; ++i) {
        _v2l[lab[i]] = i;
    }
    // group canonical vertices according to orbits and reorder groups according to the min canonical vertex id
    int o = 0;
    std::vector<std::vector<int>> canonGroup(_numVertices);
    std::vector<int> canonV2Group(_numVertices);
    std::vector<int> group2O(_numVertices, -1);
    for (int i = 0; i < _numVertices; ++i) {
        canonGroup[orbits[i]].push_back(_v2l[i]);
        canonV2Group[_v2l[i]] = orbits[i];
    }
    for (int i = 0; i < _numVertices; ++i) {
        int group = canonV2Group[i];
        if (group2O[group] == -1) {
            group2O[group] = o;
            ++o;
        }
    }
    for (int i = 0; i < _numVertices; ++i) {
        for (int u: canonGroup[i]) {
            canonV2O[u] = group2O[i];
        }
    }
    delete _v2o;
    _v2o = new int[_numVertices];
    for (int i = 0; i < _numVertices; ++i) {
        _v2o[i] = canonV2O[_v2l[i]];
    }
    _autoSize = (int)round(stats.grpsize1) * quickPow10(stats.grpsize2);
    int autoGroupSize = 0;
    if (_orbitType == 0) _divideFactor = _autoSize;
    else if (_orbitType == 1) {
        o = _v2o[0];
        for (VertexID i = 0; i < _numVertices; ++i) {
            if (_v2o[i] == o)
                ++autoGroupSize;
        }
        _divideFactor = _autoSize / autoGroupSize;
    }
    else if (_orbitType == 2) {
        int orbit0 = _v2o[0], orbit1 = _v2o[1];
        for (int u1 = 0; u1 < _numVertices; ++u1) {
            for (int u2 = u1 + 1; u2 < _numVertices; ++u2) {
                if (isEdge(u1, u2) && ((_v2o[u1] == orbit0 && _v2o[u2] == orbit1) ||
                                       (_v2o[u2] == orbit0 && _v2o[u1] == orbit1))) {
                    ++autoGroupSize;
                }
            }
        }
        _divideFactor = _autoSize / autoGroupSize;
    }
    for (int i = 0; i < n; ++i)
        lab[i] = i;
    for (int i = 0; i < n - 1; ++i)
        ptn[i] = 1;
    ptn[n - 1] = 0;
    if (_orbitType == 1)
        ptn[0] = 0;
    else if (_orbitType == 2) {
        ptn[0] = 0;
        ptn[1] = 0;
    }
    densenauty(g,lab,ptn,orbits,&options,&stats,m,n,cg);
    // save the canonical value of the pattern
    int pos = 0;
    _canonValue = 0;
    for (int i = 0; i < m * n; ++i) {
        unsigned long row = cg[i];
        bool current;
        for (unsigned long j = 0; j < n; ++j) {
            if (i != j) {
                if (1 & row >> (sizeof(unsigned long) * 8 - j - 1)) current = true;
                else current = false;
                if (current)
                    _canonValue += 1 << pos;
                ++pos;
            }
        }
    }
}

void PatternGraph::computeCandidateRules() {
    _candidateRules.clear();
    DYNALLSTAT(graph,g,g_sz);
    DYNALLSTAT(int,lab,lab_sz);
    DYNALLSTAT(int,ptn,ptn_sz);
    DYNALLSTAT(int,orbits,orbits_sz);
    static DEFAULTOPTIONS_GRAPH(options);
    options.digraph = TRUE;
    options.defaultptn = FALSE;
    statsblk stats;
    int n = _numVertices;
    // m should be 1
    int m = SETWORDSNEEDED(n);
    nauty_check(WORDSIZE,m,n,NAUTYVERSIONID);
    DYNALLOC2(graph,g,g_sz,m,n,"malloc");
    DYNALLOC1(int,lab,lab_sz,n,"malloc");
    DYNALLOC1(int,ptn,ptn_sz,n,"malloc");
    DYNALLOC1(int,orbits,orbits_sz,n,"malloc");
    EMPTYGRAPH(g,m,n);
    for (VertexID i = 0; i < n; ++i) {
        for (VertexID j = 0; j < n; ++j) {
            if (isEdge(i, j))
                        ADDONEARC(g, i, j, m);
        }
    }
    std::vector<int> initialV;
    std::queue<std::vector<int>> coloredQ;
    std::queue<std::vector<std::vector<VertexID>>> groupQ;
    groupQ.push(std::vector<std::vector<VertexID>>(0));
    coloredQ.push(initialV);
    while (!coloredQ.empty()) {
        std::vector<int> coloredV = coloredQ.front();
        std::vector<std::vector<VertexID>> currentGroup = groupQ.front();
        coloredQ.pop();
        groupQ.pop();
        // reorganize lab and ptn, put colored vertices before uncolored vertices in lab
        for (int i = 0; i < coloredV.size(); ++i) {
            lab[i] = coloredV[i];
            ptn[i] = 0;
        }
        int pos = (int)coloredV.size();
        for (int i = 0; i < n; ++i) {
            if (std::find(coloredV.begin(), coloredV.end(), i) == coloredV.end()) {
                lab[pos] = i;
                ptn[pos] = 1;
                ++pos;
            }
        }
        ptn[n - 1] = 0;
        densenauty(g, lab, ptn, orbits, &options, &stats, m, n, NULL);
        int autoSize = (int)round(stats.grpsize1) * quickPow10(stats.grpsize2);
        if (autoSize != 1) {
            std::vector<std::vector<VertexID>> o2v = std::vector<std::vector<VertexID>>(n);
            for (int i = 0; i < n; ++i)
                o2v[orbits[i]].push_back(i);
            for (const auto &group: o2v) {
                if (group.size() > 1) {
                    // group is a valid rule
                    if (group.size() > 2) {
                        for (int u: group) {
                            std::vector<int> newColoredV = coloredV;
                            std::vector<std::vector<VertexID>> newGroup = currentGroup;
                            newColoredV.push_back(u);
                            std::vector<VertexID> rule;
                            rule.push_back(u);
                            for (int w: group) {
                                if (w != u) rule.push_back(w);
                            }
                            newGroup.push_back(group);
                            coloredQ.push(newColoredV);
                            groupQ.push(newGroup);
                        }
                    }
                    else {
                        std::vector<int> newColoredV = coloredV;
                        std::vector<std::vector<VertexID>> newGroup = currentGroup;
                        newColoredV.push_back(group[0]);
                        newGroup.push_back(group);
                        coloredQ.push(newColoredV);
                        groupQ.push(newGroup);
                    }
                }
            }
        }
        else {
            _candidateRules.push_back(currentGroup);
        }
    }
}

// set the aggregation to single-vertex aggregation.
void PatternGraph::setSingleAggre() {
    if (_aggreSize == 0) {
        return;
    }
    if (_orbitType == 1)
        _aggreSize = 1;
    else
        _aggreSize = 2;
}

// set the aggregation to multi-vertex aggregation.
void PatternGraph::setMultiAggre() {
    if (_aggreSize == 0) {
        return;
    }
    int oldWeight = _aggreWeight[0];
    delete[] _aggreV;
    delete[] _aggreWeight;
    _aggreV = new VertexID[_numEdges];
    _aggreWeight = new int[_numEdges];
    _aggreSize = 0;
    if (_orbitType == 1) {
        int orbit = _v2o[0];
        for (int i = 0; i < _numVertices; ++i) {
            if (_v2o[i] == orbit) {
                _aggreV[_aggreSize] = i;
                _aggreSize++;
            }
        }
    }
    else if(_orbitType == 2) {
        int orbit0 = _v2o[0], orbit1 = _v2o[1];
        for (int u1 = 0; u1 < _numVertices; ++u1) {
            for (int u2 = u1 + 1; u2 < _numVertices; ++u2) {
                if (isEdge(u1, u2) && ((_v2o[u1] == orbit0 && _v2o[u2] == orbit1) ||
                                       (_v2o[u2] == orbit0 && _v2o[u1] == orbit1))) {
                    _aggreV[_aggreSize] = u1;
                    _aggreSize++;
                    _aggreV[_aggreSize] = u2;
                    _aggreSize++;
                }
            }
        }
    }
    for (int i = 0; i < _numEdges; ++i)
        _aggreWeight[i] = oldWeight;
}



// generate DAGs for the core graph of the pattern
// adapted from utility/automorphism/directg.c
void PatternGraph::genDAGs(std::vector<PatternGraph> &din, std::vector<PatternGraph> &dout) const {
    long minarcs = 0, maxarcs = NOLIMIT;
    splitmod = 1;
    splitres = 0;
    int nfixed = 0;
//    if (_orbitType == 1 && _aggreSize == 1) nfixed = 1;
    DYNALLSTAT(graph,g,g_sz);
    static DEFAULTOPTIONS_GRAPH(options);
    statsblk stats;
    int n = int(_numVertices);
    int m = SETWORDSNEEDED(n);
    nauty_check(WORDSIZE,m,n,NAUTYVERSIONID);
    DYNALLOC2(graph,g,g_sz,m,n,"malloc");
    EMPTYGRAPH(g,m,n);
    // build graph from core vertices
    for (int i = 0; i < _numVertices; ++i) {
        for (int j = i + 1; j < _numVertices; ++j) {
            if (isEdge(i, j)) ADDONEEDGE(g, i, j, m);
        }
    }
    // find all dags of the core graph up to isomorphism, store them in din and dout
    direct(g,nfixed,minarcs,maxarcs,m,n,din,dout);
    for (auto &p: din) {
        p.setAggreInfo(_aggreV, _aggreSize, _orbitType);
        p.setCorePeripheral(_coreV, _peripheralV, _coreSize);
        p.computeAutoGroup();
    }
    for (auto &p: dout) {
        p.setAggreInfo(_aggreV, _aggreSize, _orbitType);
        p.setCorePeripheral(_coreV, _peripheralV, _coreSize);
        p.computeAutoGroup();
    }
}

Edge * PatternGraph::coreUndirectedEdges(ui &num) const {
    Edge *edgeList = new Edge[_numEdges / 2];
    num = 0;
    for (ui i = 0; i < _coreSize; ++i) {
        for (ui j = i + 1; j < _coreSize; ++j) {
            VertexID u1 = _coreV[i], u2 = _coreV[j];
            if (isEdge(u1, u2)) {
                edgeList[num] = std::make_pair(u1, u2);
                ++num;
            }
        }
    }

    return edgeList;
}

void PatternGraph::printGraph(bool direction) const {
    printf("graph in CSR:\n");
    printf("vertices: %d, edges: %d\n", _numVertices, _numEdges);
    printf("from   to\n");
    for (VertexID i = 0; i < _numVertices; i++)
        for (EdgeID e = _offsets[i]; e < _offsets[i + 1]; e++) {
            VertexID j = _nbors[e];
            if (direction)
                printf("%4d %4d\n", i, j);
            else
                printf("%4d %4d\n", j, i);
        }
}

bool PatternGraph::isClique() const {
    return _numVertices > 2 && _numVertices * (_numVertices - 1) == _numEdges;
}

VertexID *PatternGraph::getMultiAggreV(ui &aggreSize) const {
    if (_orbitType == 0)
        return nullptr;
    VertexID *aggreV = new VertexID[_numEdges];
    aggreSize = 0;
    if (_orbitType == 1) {
        int orbit = _v2o[0];
        for (int i = 0; i < _numVertices; ++i) {
            if (_v2o[i] == orbit) {
                aggreV[aggreSize] = i;
                aggreSize++;
            }
        }
    }
    else if(_orbitType == 2) {
        int orbit0 = _v2o[0], orbit1 = _v2o[1];
        for (int u1 = 0; u1 < _numVertices; ++u1) {
            for (int u2 = u1 + 1; u2 < _numVertices; ++u2) {
                if (isEdge(u1, u2) && ((_v2o[u1] == orbit0 && _v2o[u2] == orbit1) ||
                                       (_v2o[u2] == orbit0 && _v2o[u1] == orbit1))) {
                    aggreV[aggreSize] = u1;
                    aggreSize++;
                    aggreV[aggreSize] = u2;
                    aggreSize++;
                }
            }
        }
    }
    return aggreV;
}

void PatternGraph::buildCorePeripheral() {
    _coreV = new VertexID[_numVertices];
    _peripheralV = new VertexID[_numVertices];
    ui peripheralSize = 0;
    bool *visited = new bool[_numVertices + 1];
    int *pos = new int[_numVertices + 1];
    int *bin = new int[_numVertices + 1];
    int *deg = new int[_numVertices + 1];
    int *vert = new int[_numVertices + 1];
    memset(visited, false, sizeof(bool) * (_numVertices + 1));
    memset(pos, 0, sizeof(int) * (_numVertices + 1));
    memset(bin, 0, sizeof(int) * (_numVertices + 1));
    memset(deg, 0, sizeof(int) * (_numVertices + 1));
    memset(vert, 0, sizeof(int) * (_numVertices + 1));
    int maxDeg = 0;
    for (int v = 1; v <= _numVertices; ++v) {
        int d = (int)degree(v - 1);
        deg[v] = d;
        if (d > maxDeg) maxDeg = d;
    }
    for (int v = 1; v <= _numVertices; ++v)
        ++bin[deg[v]];
    int start = 1;
    for (int d = 0; d <= maxDeg; ++d) {
        int num = bin[d];
        bin[d] = start;
        start += num;
    }
    for (int v = 1; v <= _numVertices; ++v) {
        pos[v] = bin[deg[v]];
        vert[pos[v]] = v;
        ++bin[deg[v]];
    }
    for (int d = maxDeg; d > 0; --d)
        bin[d] = bin[d-1];
    bin[0] = 1;
    for (int i = 1; i <= _numVertices; ++i) {
        VertexID v = vert[i];
        if (deg[v] < 2){
            // v is a peripheral vertex
            _peripheralV[peripheralSize] = v - 1;
            visited[v - 1] = true;
            ++peripheralSize;
            ui nbrCnt = 0;
            VertexID *neighbors = getNeighbors(v - 1, nbrCnt);
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
        else break;
    }

    _coreSize = _numVertices - peripheralSize;
    ui coreIdx = 0;
    for (VertexID i = 0; i < _numVertices; ++i) {
        if (!visited[i]) {
            _coreV[coreIdx] = i;
            ++coreIdx;
        }
    }
    delete[] visited;
    delete[] pos;
    delete[] bin;
    delete[] deg;
    delete[] vert;
}

// create a new pattern graph, delete v, add v's neighbor to u
PatternGraph PatternGraph::shrink(VertexID u, VertexID v) const {
//    assert(u < v);
    std::vector<VertexID> old2New(_numVertices);
    for (VertexID i = 0; i < _numVertices; ++i)
        old2New[i] = i;

    // relabel vertices according to the shrinkage pair
    for (VertexID w = 0; w < old2New.size(); ++w) {
        if (w < v) old2New[w] = old2New[w];
        else if (w == v) old2New[w] = old2New[u];
        else old2New[w] = old2New[w] - 1;
    }
    Edge *edgeList = new Edge[_numEdges];
    ui shrinkNumEdges = 0;
    for (VertexID i = 0; i < _numVertices; ++i) {
        if (i != u && i != v) {
            for (EdgeID ij = _offsets[i]; ij < _offsets[i + 1]; ++ij) {
                VertexID j = _nbors[ij];
                if (j != v) {
                    edgeList[shrinkNumEdges] = std::make_pair(old2New[i], old2New[j]);
                    ++shrinkNumEdges;
                }
                // if u is not a neighbor of i but v is a neighbor of i, then u becomes a neighbor of i
                if (j == v && !isEdge(i, u)) {
                    edgeList[shrinkNumEdges] = std::make_pair(old2New[i], old2New[j]);
                    ++shrinkNumEdges;
                }
            }
        }
        else if (i == u) {
            for (EdgeID ij = _offsets[i]; ij < _offsets[i + 1]; ++ij) {
                VertexID j = _nbors[ij];
                edgeList[shrinkNumEdges] = std::make_pair(old2New[i], old2New[j]);
                ++shrinkNumEdges;
            }
            // add v's neighbor to u's neighbor
            for (EdgeID vj = _offsets[v]; vj < _offsets[v + 1]; ++vj) {
                VertexID j = _nbors[vj];
                if (!isEdge(i, j)) {
                    edgeList[shrinkNumEdges] = std::make_pair(old2New[i], old2New[j]);
                    ++shrinkNumEdges;
                }
            }
        }
    }

    PatternGraph p(_numVertices - 1, shrinkNumEdges, edgeList);
    p._orbitType = _orbitType;
    p.computeAutoGroup();
    p.computeCandidateRules();
    // set aggregation vertices in p. For each aggregation vertex w in *this,
    // add _aggreWeight[w] to the vertex that w maps to.
    // set aggregation edges similarly.
    if (_aggreSize != 0) {
        p._aggreV = new VertexID[_aggreSize];
        p._aggreWeight = new int[_aggreSize];
    }
    p._aggreSize = 0;
    if (_orbitType == 1) {
        std::vector<bool> visited(_numVertices - 1, false);
        for (int i = 0; i < _aggreSize; ++i) {
            VertexID w = _aggreV[i];
            VertexID newW = old2New[w];
            if (!visited[newW]) {
                visited[newW] = true;
                p._aggreV[p._aggreSize] = newW;
                p._aggreWeight[p._aggreSize] = _aggreWeight[i];
                ++p._aggreSize;
            }
            else {
                for (int j = 0; j < p._aggreSize; ++j) {
                    if (p._aggreV[j] == newW) {
                        p._aggreWeight[j] += _aggreWeight[i];
                        break;
                    }
                }
            }
        }
//        int weight1 = 0, weight2 = 0;
//        for (int i = 0; i < _aggreSize; ++i) weight1 += _aggreWeight[i];
//        for (int i = 0; i < p._aggreSize; ++i) weight2 += p._aggreWeight[i];
//        assert(weight1 == weight2);
    }
    else if (_orbitType == 2) {
        std::map<EdgeID, int> e2Pos; // the position of the edge's weight in p._aggreWeight
        for (int i = 0; i < _aggreSize; i = i + 2) {
            VertexID w1 = _aggreV[i], w2 = _aggreV[i + 1];
            VertexID newW1 = old2New[w1], newW2 = old2New[w2];
            EdgeID e = newW1 < newW2 ? 100 * newW1 + newW2 : 100 * newW2 + newW1;
            if (e2Pos.find(e) == e2Pos.end()) {
                p._aggreWeight[e2Pos[e]] += _aggreWeight[i / 2];
            }
            else {
                e2Pos[e] = p._aggreSize / 2;
                p._aggreWeight[p._aggreSize / 2] = _aggreWeight[i / 2];
                p._aggreV[p._aggreSize] = newW1;
                ++p._aggreSize;
                p._aggreV[p._aggreSize] = newW2;
                ++p._aggreSize;
            }
        }
    }
    delete[] edgeList;
    return p;
}

static boolean ismax(int *p, int n)
/* test if x^p <= x */
{
    int i,j,k;
    set px[MAXME];

    EMPTYSET(px,me);

    for (j = 0; j < nix; ++j)
    {
        i = ix[j];
        k = i >> 1;
        if (i & 1) ADDELEMENT(px,edgeno[p[_v1[k]]][p[_v0[k]]]);
        else       ADDELEMENT(px,edgeno[p[_v0[k]]][p[_v1[k]]]);

        if (px[0] > x[0])
        {
            rejectlevel = k;
            return FALSE;
        }
    }

    rejectlevel = MAXNE+1;

    if (px[0] < x[0]) return TRUE;

    for (i = 1; i < me; ++i)
        if      (px[i] > x[i]) return FALSE;
        else if (px[i] < x[i]) return TRUE;

    ++newgroupsize;
    ntgroup = TRUE;
    return TRUE;
}

/**************************************************************************/

void testmax(int *p, int n, int *abort)
/* Called by allgroup2. */
{
    int i;

    if (first)
    {                       /* only the identity */
        first = FALSE;
        return;
    }

    if (!ismax(p,n))
    {
        *abort = 1;
        for (i = 0; i < n; ++i) lastreject[i] = p[i];
        lastrejok = TRUE;
    }
}

/**************************************************************************/

static int trythisone(grouprec *group, int ne, int m, int n, std::vector<PatternGraph> &pin, std::vector<PatternGraph> &pout)
{
    int i,k;
    boolean accept;
    graph g[MAXNV*MAXMV];

    first = TRUE;

    nix = ne;
    newgroupsize = 1;
    ntgroup = FALSE;

    if (!group || groupsize == 1)
        accept = TRUE;
    else if (lastrejok && !ismax(lastreject,n))
        accept = FALSE;
    else if (lastrejok && groupsize == 2)
        accept = TRUE;
    else
    {
        newgroupsize = 1;
        ntgroup = FALSE;
        if (allgroup2(group,testmax) == 0)
            accept = TRUE;
        else
            accept = FALSE;
    }

    if (accept)
    {
        if (Vswitch && !ntisol && !ntgroup) return MAXNE+1;
        // add the generated digraph to the vector pin and pout
        int edgeCnt = 0;
        Edge* inEdgeList = new Edge[ne];
        Edge* outEdgeList = new Edge[ne];
        VertexID from, to;
        for (i = -1; (i = nextelement(x,me,i)) >= 0; ) {
            k = i >> 1;
            if (i & 1) {
                from = _v1[k];
                to = _v0[k];
            }
            else {
                from = _v0[k];
                to = _v1[k];
            }
            inEdgeList[edgeCnt] = std::make_pair(to, from);
            outEdgeList[edgeCnt] = std::make_pair(from, to);
            ++edgeCnt;
        }
        pin.emplace_back(n, ne, inEdgeList);
        pout.emplace_back(n, ne, outEdgeList);
        delete[] inEdgeList;
        delete[] outEdgeList;
        return MAXNE+1;
    }
    else
        return (rejectlevel < splitlevel ? splitlevel : rejectlevel);
}

/**************************************************************************/

static void updatetc(graph *oldtc, graph *newtc, int v, int w, int m, int n)
/* Update reflexive transitive closure oldtc with the new
 * edge v->w, making newtc.  Loops are essential for this to work. */
{
    int i,j;
    set *gi,*gw;

    for (i = 0; i < m*n; ++i) newtc[i] = oldtc[i];

    gw = newtc + m*w;
    for (i = 0, gi = newtc; i < n; ++i, gi += m)
    {
        if (ISELEMENT(gi,v))
        {
            for (j = 0; j < m; ++j) gi[j] |= gw[j];
        }
    }
}

/**************************************************************************/

static void updatetc1(graph *oldtc, graph *newtc, int v, int w, int n)
/* Update reflexive transitive closure oldtc with the new
 * edge v->w, making newtc.  Loops are essential for this to work.
   Version for m=1. */
{
    int i,j;
    setword gw;

    for (i = 0; i < n; ++i) newtc[i] = oldtc[i];

    gw = newtc[w];
    for (i = 0; i < n; ++i)
        if ((newtc[i] & bit[v])) newtc[i] |= gw;
}

/**************************************************************************/

static int scan_acyclic(int level, int ne, int minarcs, int maxarcs, int sofar,
                        graph *oldtc, grouprec *group, int m, int n, std::vector<PatternGraph> &pin, std::vector<PatternGraph> &pout)
/* Main recursive scan for acyclic orientations;
 * returns the level to return to. */
{
    int k,retlev;
    graph newtc[MAXNV*MAXMV];
    int w0,w1;

    w0 = _v0[level];
    w1 = _v1[level];

    if (level == splitlevel)
    {
        if (splitcount-- != 0) return level-1;
        splitcount = splitmod - 1;
    }

    if (level == ne)
    {
        retlev = trythisone(group, sofar, m, n, pin, pout);
        return retlev;
    }

    if (!ISELEMENT(oldtc+m*w1,w0))
    {
        k = 2*level;        /* edge w0->w1 */
                ADDELEMENT(x,k);
        ix[sofar] = k;
        {
            updatetc(oldtc,newtc,w0,w1,m,n);
            retlev = scan_acyclic(level+1, ne, minarcs, maxarcs,sofar+1, newtc, group, m, n, pin, pout);
        }
                DELELEMENT(x,k);
        if (retlev < level) return retlev;
    }

    if (!ISELEMENT(oldtc+m*w0,w1))
    {
        k = 2*level + 1;      /* edge w1->w0 */
                ADDELEMENT(x,k);
        ix[sofar] = k;
        {
            updatetc(oldtc,newtc,w1,w0,m,n);
            retlev = scan_acyclic(level+1, ne, minarcs, maxarcs,sofar+1, newtc, group, m, n, pin, pout);
        }
                DELELEMENT(x,k);
        if (retlev < level) return retlev;
    }

    return level-1;
}


static int scan_acyclic1(int level, int ne, int minarcs, int maxarcs, int sofar,
                         graph *oldtc, grouprec *group, int n, std::vector<PatternGraph> &din, std::vector<PatternGraph> &dout)
/* Main recursive scan for acyclic orientations;
 * returns the level to return to. Version for m=1. */
{
    int k,retlev;
    graph newtc[MAXNV*MAXMV];
    int w0,w1;

    w0 = _v0[level];
    w1 = _v1[level];

    if (level == splitlevel)
    {
        if (splitcount-- != 0) return level-1;
        splitcount = splitmod - 1;
    }

    if (level == ne)
    {
        retlev = trythisone(group,sofar,1,n,din,dout);
        return retlev;
    }

    if (!(oldtc[w1] & bit[w0]))
    {
        k = 2*level;        /* edge w0->w1 */
                ADDELEMENT(x,k);
        ix[sofar] = k;
        {
            updatetc1(oldtc,newtc,w0,w1,n);
            retlev = scan_acyclic1(level+1,ne,minarcs,maxarcs,sofar+1,newtc,group,n,din,dout);
        }
                DELELEMENT(x,k);
        if (retlev < level) return retlev;
    }

    if (!(oldtc[w0] & bit[w1]))
    {
        k = 2*level + 1;      /* edge w1->w0 */
                ADDELEMENT(x,k);
        ix[sofar] = k;
        {
            updatetc1(oldtc,newtc,w1,w0,n);
            retlev = scan_acyclic1(level+1,ne,minarcs,maxarcs,sofar+1,newtc,group,n,din,dout);
        }
                DELELEMENT(x,k);
        if (retlev < level) return retlev;
    }

    return level-1;
}

/**************************************************************************/

static int scan(int level, int ne, int minarcs, int maxarcs, int sofar,
                boolean oriented, grouprec *group, int m, int n, std::vector<PatternGraph> &pin, std::vector<PatternGraph> &pout)
/* Main recursive scan; returns the level to return to. */
{
    int k,retlev;
    if (level == splitlevel)
    {
        if (splitcount-- != 0) return level-1;
        splitcount = splitmod - 1;
    }

    if (level == ne)
    {
        retlev = trythisone(group, sofar, m, n, pin, pout);
        return retlev;
    }

    if (oriented || sofar + 1 + 2*(ne - level - 1) >= minarcs)
    {
        k = 2*level;
                ADDELEMENT(x,k);
        ix[sofar] = k;
        retlev = scan(level+1, ne, minarcs, maxarcs,sofar+1, oriented, group, m, n, pin, pout);
                DELELEMENT(x,k);
        if (retlev < level) return retlev;
        ++k;
                ADDELEMENT(x,k);
        ix[sofar] = k;
        retlev = scan(level+1, ne, minarcs, maxarcs,sofar+1, oriented, group, m, n, pin, pout);
                DELELEMENT(x,k);
        if (retlev < level) return retlev;
    }

    if (!oriented && sofar + 2 + ne - level - 1 <= maxarcs)
    {
        k = 2*level;
                ADDELEMENT(x,k);
                ADDELEMENT(x,k+1);
        ix[sofar] = k;
        ix[sofar+1] = k+1;
        retlev = scan(level+1, ne, minarcs, maxarcs,sofar+2, oriented, group, m, n, pin, pout);
                DELELEMENT(x,k+1);
                DELELEMENT(x,k);
        if (retlev < level) return retlev;
    }

    return level-1;
}

/**************************************************************************/

static void direct(graph *g, int nfixed, long minarcs, long maxarcs, int m, int n,
                   std::vector<PatternGraph> &pin, std::vector<PatternGraph> &pout)
{
    static DEFAULTOPTIONS_GRAPH(options);
    statsblk stats;
    setword workspace[200];
    grouprec *group;
    long ne;
    int i,j,k,j0,j1,deg;
    int isol0,isol1;  /* isolated vertices before and after nfixed */
    set *gi;
    int lab[MAXNV],ptn[MAXNV],orbits[MAXNV];
    set active[(MAXNV+WORDSIZE-1)/WORDSIZE];
    graph tc[MAXNV*MAXMV];

    nauty_check(WORDSIZE,m,n,NAUTYVERSIONID);

    j0 = -1;  /* last vertex with degree 0 */
    j1 = n;   /* first vertex with degree > 0 */
    isol0 = isol1 = 0;

    ne = 0;
    for (i = 0, gi = g; i < n; ++i, gi += m)
    {
        deg = 0;
        for (j = 0; j < m; ++j) deg += POPCOUNT(gi[j]);
        if (deg == 0)
        {
            lab[++j0] = i;
            if (i < nfixed) ++isol0; else ++isol1;
        }
        else
            lab[--j1] = i;
        ne += deg;
    }
    ne /= 2;
    ntisol = (isol0 >= 2 || isol1 >= 2);

    me = (2*ne + WORDSIZE - 1) / WORDSIZE;
    if (me == 0) me = 1;
    EMPTYSET(x,me);

    splitcount = splitres;
    if (splitmod == 1 || ne <= 6)
        splitlevel = -1;
    else
        splitlevel = (ne <= 36 ? ne/2 : 18);

    if (splitlevel < 0 && splitres > 0) return;
    if (ne == 0 && minarcs <= 0 && (!Vswitch || ntisol) && splitres == 0)
    {
        trythisone(NULL, 0, m, n, pin, pout);    // only in case of 0 edges
        return;
    }

    if (maxarcs < ne || minarcs > ne) return;

    if (n > MAXNV || ne > MAXNE)
    {
        fprintf(stderr,">E directg: MAXNV or MAXNE exceeded\n");
        exit(1);
    }

    for (i = 0; i < n; ++i) ptn[i] = 1;
    ptn[n-1] = 0;
    EMPTYSET(active,m);
            ADDELEMENT(active,0);

    for (i = 0; i <= j0; ++i)
    {
        if (i < n-1) ADDELEMENT(active,i+1);
        ptn[i] = 0;
    }

    for (i = j0+1; i < n; ++i)
        if (lab[i] < nfixed) break;

    if (i != j0+1 && i != n)
    {
        ptn[i-1] = 0;
                ADDELEMENT(active,i);
    }

    options.defaultptn = FALSE;
    options.userautomproc = groupautomproc;
    options.userlevelproc = grouplevelproc;

    nauty(g,lab,ptn,active,orbits,&options,&stats,workspace,200,m,n,NULL);

    if (stats.grpsize2 == 0)
        groupsize = stats.grpsize1 + 0.1;
    else
        groupsize = 0;

    if (Vswitch && groupsize == 1 && !ntisol) return;

    group = groupptr(FALSE);
    makecosetreps(group);
    k = 0;
    for (i = 0, gi = g; i < n; ++i, gi += m)
    {
        for (j = i; (j = nextelement(gi,m,j)) >= 0; )
        {
            _v0[k] = i;
            _v1[k] = j;
            edgeno[i][j] = 2*k;
            edgeno[j][i] = 2*k+1;
            ++k;
        }
    }

    lastrejok = FALSE;
    for (i = 0; i < m*n; ++i) tc[i] = 0;
    for (i = 0; i < n; ++i) ADDELEMENT(tc+i*m,i);
    if (m == 1) scan_acyclic1(0, ne, minarcs, maxarcs, 0, tc, group, n, pin, pout);
    else        scan_acyclic(0, ne, minarcs, maxarcs, 0, tc, group, m, n, pin, pout);
}

// construct directed data graph based on vertex id.
// outDag = true: for each edge (u, v), u -> v if u < v
// else, for each edge (u, v), u <- v if u < v
DataGraph constructDirectedDataGraph(const DataGraph &g, bool outDag) {
    VertexID n = g.getNumVertices();
    EdgeID m = g.getNumEdges();
    DataGraph dg = DataGraph(n, m / 2);
    Edge *edgeList = new Edge[m / 2];
    EdgeID edgeCnt = 0;
    for (VertexID u = 0; u < n; ++u) {
        ui nbrCnt = 0;
        VertexID *outNeighbors = g.getNeighborsLargerID(u, nbrCnt);
        for (EdgeID j = 0; j < nbrCnt; ++j) {
            VertexID v = outNeighbors[j];
            if (outDag)
                edgeList[edgeCnt] = std::make_pair(u, v);
            else
                edgeList[edgeCnt] = std::make_pair(v, u);
            ++edgeCnt;
        }
    }
    dg.addDirectedEdges(edgeList, m / 2);
    dg.buildLargerOffset();
    delete[] edgeList;
    return dg;
}

Pattern Pattern::shrink(VertexID i, VertexID j, std::vector<VertexID> &old2New) const {
    PatternGraph empty = PatternGraph();
    ui n = in.getNumVertices();
    PatternGraph uShrink = u.shrink(old2New[i], old2New[j]);
    if (n == 0) {
        uShrink.buildCorePeripheral();
        // relabel vertices according to the shrinkage pair
        for (VertexID w = 0; w < old2New.size(); ++w) {
            if (w < j) old2New[w] = old2New[w];
            else if (w == j) old2New[w] = old2New[i];
            else old2New[w] = old2New[w] - 1;
        }
        return Pattern(uShrink, in, out);
    }
    PatternGraph inShrink = in.shrink(old2New[i], old2New[j]);
    // 1. check whether there is edge direction conflict
    for (VertexID v = 0; v < n - 1; ++v) {
        ui cnt;
        VertexID *neighbors = inShrink.getNeighbors(v, cnt);
        for (EdgeID vw = 0; vw < cnt; ++vw) {
            VertexID w= neighbors[vw];
            if (inShrink.isEdge(w, v))
                return Pattern(empty, empty, empty);
        }
    }
    // 2. check whether p is acyclic: using topological order
    PatternGraph outShrink = out.shrink(old2New[i], old2New[j]);
    std::vector<ui> inDegree(n - 1);
    int visited = 0;
    std::queue<VertexID> q;
    for (VertexID v = 0; v < n - 1; ++v) {
        inDegree[v] = inShrink.degree(v);
        if (inDegree[v] == 0)
            q.push(v);
    }
    while (!q.empty()) {
        VertexID v = q.front();
        q.pop();
        ++visited;
        ui cnt;
        VertexID *neighbors = outShrink.getNeighbors(v, cnt);
        for (EdgeID vw = 0; vw < cnt; ++vw) {
            VertexID w = neighbors[vw];
            --inDegree[w];
            if (inDegree[w] == 0) q.push(w);
        }
    }
    if (visited < n - 1) return Pattern(empty, empty, empty);
    // the shrinkage is valid
    // relabel vertices according to the shrinkage pair
    uShrink.buildCorePeripheral();
    ui coreSize;
    VertexID *coreV = uShrink.getCoreV(coreSize);
    VertexID *peripheralV = uShrink.getPeripheralV();
    inShrink.setCorePeripheral(coreV, peripheralV, coreSize);
    outShrink.setCorePeripheral(coreV, peripheralV, coreSize);
    for (VertexID w = 0; w < old2New.size(); ++w) {
        if (w < j) old2New[w] = old2New[w];
        else if (w == j) old2New[w] = old2New[i];
        else old2New[w] = old2New[w] - 1;
    }
    return Pattern(uShrink, inShrink, outShrink);
}

Pattern Pattern::shrink(const std::vector<VertexID> &partition) const {
    ui n = u.getNumVertices();
//    assert(n == partition.size());
    Pattern result(*this);
    std::vector<VertexID> old2New(n);
    for (VertexID i = 0; i < n; ++i)
        old2New[i] = i;
    std::vector<std::vector<VertexID>> part(partition.size());
    for (int i = 0; i < n; ++i)
        part[partition[i]].push_back(i);
    for (int i = 0; i < n; ++i) {
        if (part[i].size() < 2) continue;
        for (int j = 1; j < part[i].size(); ++j) {
            result = result.shrink(part[i][0], part[i][j], old2New);
            if (result.empty()) return result;
        }
    }

    return result;
}

void Pattern::setSingleAggre() {
    u.setSingleAggre();
}

void Pattern::setMultiAggre() {
    u.setMultiAggre();
}

std::vector<PatternGraph> loadPatternGraph(const std::string &path, bool batchQuery, std::vector<std::string> &files) {
    std::vector<PatternGraph> patterns;
    if (batchQuery) {
        DIR *dir;
        struct dirent *ent;
        if ((dir = opendir(path.c_str())) != nullptr) {
            while ((ent = readdir (dir)) != nullptr) {
                std::string file = std::string(ent -> d_name);
                if (file[0] == '.') continue;
                files.push_back(file);
            }
        }
        std::sort(files.begin(), files.end(), std::greater<std::string>());
        for (std::string &file : files) {
            PatternGraph pg =  PatternGraph();
            pg.loadPatternGraph(path + "/" + file);
            patterns.push_back(pg);
        }
    }
    else {
        PatternGraph pg =  PatternGraph();
        pg.loadPatternGraph(path);
        files.push_back(path);
        patterns.push_back(pg);
    }

    return patterns;
}

// check whether the mapping (from v2 to v1) is valid
bool checkMapping(const PatternGraph &p1, const PatternGraph &p2, const std::vector<VertexID> &v2, VertexID *mapping) {
    for (int i = 0; i < v2.size(); ++i) {
        for (int j = 0; j < v2.size(); ++j) {
            if (!(p2.isEdge(v2[i], v2[j]) == p1.isEdge(mapping[v2[i]], mapping[v2[j]])))
                return false;
        }
    }

    return true;
}

// the mapping is given by v1 -> v2
bool checkMapping(const PatternGraph &p1, const PatternGraph &p2, const std::vector<VertexID> &v1, const std::vector<VertexID> &v2) {
    for (int i = 0; i < v2.size(); ++i) {
        for (int j = 0; j < v2.size(); ++j) {
            if (!(p2.isEdge(v2[i], v2[j]) == p1.isEdge(v1[i], v1[j])))
                return false;
        }
    }

    return true;
}

bool checkMapping(const Pattern &p1, const Pattern &p2, const std::vector<VertexID> &v2, VertexID *mapping) {
    if (p1.useDAG() != p2.useDAG()) return false;
    if (p1.useDAG()) return checkMapping(p1.out, p2.out, v2, mapping);
    else return checkMapping(p1.u, p2.u, v2, mapping);
}

bool checkMapping(const Pattern &p1, const Pattern &p2, const std::vector<VertexID> &v1, const std::vector<VertexID> &v2) {
    if (p1.useDAG() != p2.useDAG()) return false;
    if (p1.useDAG()) return checkMapping(p1.out, p2.out, v1, v2);
    else return checkMapping(p1.u, p2.u, v1, v2);
}

CanonType subgraphCanonValue(const PatternGraph &p, const std::vector<VertexID> &v, int *v2o) {
    CanonType canonValue = 0;
    DYNALLSTAT(graph,g,g_sz);
    DYNALLSTAT(int,lab,lab_sz);
    DYNALLSTAT(int,ptn,ptn_sz);
    DYNALLSTAT(int,orbits,orbits_sz);
    DYNALLSTAT(graph,cg,cg_sz);
    static DEFAULTOPTIONS_GRAPH(options);
    options.getcanon = TRUE;
    options.digraph = TRUE;
    statsblk stats;
    int n = (int)v.size();
    int m = SETWORDSNEEDED(n);
    nauty_check(WORDSIZE,m,n,NAUTYVERSIONID);
    DYNALLOC2(graph,g,g_sz,m,n,"malloc");
    DYNALLOC2(graph,cg,cg_sz,m,n,"malloc");
    DYNALLOC1(int,lab,lab_sz,n,"malloc");
    DYNALLOC1(int,ptn,ptn_sz,n,"malloc");
    DYNALLOC1(int,orbits,orbits_sz,n,"malloc");
    EMPTYGRAPH(g,m,n);
    for (int i = 0; i < v.size(); ++i) {
        for (int j = 0; j < v.size(); ++j) {
            if (p.isEdge(v[i], v[j]))
                        ADDONEARC(g, i, j, m);
        }
    }
    densenauty(g,lab,ptn,orbits,&options,&stats,m,n,cg);
    // save the canonical value of the pattern
    int pos = 0;
    for (int i = 0; i < m * n; ++i) {
        unsigned long row = cg[i];
        bool current;
        for (unsigned long j = 0; j < n; ++j) {
            if (i != j) {
                if (1 & row >> (sizeof(unsigned long) * 8 - j - 1)) current = true;
                else current = false;
                if (current)
                    canonValue += 1 << pos;
                ++pos;
            }
        }
    }
    if (v2o != nullptr) {
        std::map<VertexID, int> canonV2O;
        int *v2l = new int[n];
        for (int i = 0; i < n; ++i) {
            v2l[lab[i]] = i;
        }
        // group canonical vertices according to orbits and reorder groups according to the min canonical vertex id
        int o = 0;
        std::vector<std::vector<int>> canonGroup(n);
        std::vector<int> canonV2Group(n);
        std::vector<int> group2O(n, -1);
        for (int i = 0; i < n; ++i) {
            canonGroup[orbits[i]].push_back(v2l[i]);
            canonV2Group[v2l[i]] = orbits[i];
        }
        for (int i = 0; i < n; ++i) {
            int group = canonV2Group[i];
            if (group2O[group] == -1) {
                group2O[group] = o;
                ++o;
            }
        }
        for (int i = 0; i < n; ++i) {
            for (int u: canonGroup[i]) {
                canonV2O[u] = group2O[i];
            }
        }
        for (int i = 0; i < n; ++i) {
            v2o[v[i]] = canonV2O[v2l[i]];
        }
        delete[] v2l;
    }
    return canonValue;
}

bool isSubgraphAutoGroup(const PatternGraph &p, const std::vector<VertexID> &v, const std::vector<VertexID> &group) {
    CanonType canonValue = 0;
    DYNALLSTAT(graph,g,g_sz);
    DYNALLSTAT(int,lab,lab_sz);
    DYNALLSTAT(int,ptn,ptn_sz);
    DYNALLSTAT(int,orbits,orbits_sz);
    DYNALLSTAT(graph,cg,cg_sz);
    static DEFAULTOPTIONS_GRAPH(options);
    options.getcanon = TRUE;
    options.digraph = TRUE;
    statsblk stats;
    int n = (int)v.size();
    int m = SETWORDSNEEDED(n);
    nauty_check(WORDSIZE,m,n,NAUTYVERSIONID);
    DYNALLOC2(graph,g,g_sz,m,n,"malloc");
    DYNALLOC2(graph,cg,cg_sz,m,n,"malloc");
    DYNALLOC1(int,lab,lab_sz,n,"malloc");
    DYNALLOC1(int,ptn,ptn_sz,n,"malloc");
    DYNALLOC1(int,orbits,orbits_sz,n,"malloc");
    EMPTYGRAPH(g,m,n);
    for (int i = 0; i < v.size(); ++i) {
        for (int j = 0; j < v.size(); ++j) {
            if (p.isEdge(v[i], v[j]))
                        ADDONEARC(g, i, j, m);
        }
    }
    densenauty(g,lab,ptn,orbits,&options,&stats,m,n,cg);
    std::vector<std::vector<int>> autoGroups(n);
    for (int i = 0; i < n; ++i) {
        autoGroups[orbits[i]].push_back(i);
    }
    for (int i = 0; i < n; ++i) {
        if (autoGroups[i].size() != group.size()) continue;
        std::vector<VertexID> vertices;
        for (int j: autoGroups[i]) {
            vertices.push_back(v[j]);
        }
        if (vertices == group) return true;
    }
    return false;
}

EdgeID *buildInID2OutID(const DataGraph &din, const DataGraph &dout) {
    const EdgeID *inOffset = din.getOffsets();
    const VertexID *inNbors = din.getNbors();
    EdgeID *inID2OutID = new EdgeID[din.getNumEdges()];
    for (VertexID v1 = 0; v1 < din.getNumVertices(); ++v1) {
        for (EdgeID eIn = inOffset[v1]; eIn < inOffset[v1 + 1]; ++eIn) {
            VertexID v2 = inNbors[eIn];
            EdgeID eOut = dout.getEdgeID(v2, v1);
            inID2OutID[eIn] = eOut;
        }
    }

    return inID2OutID;
}

EdgeID *buildUnID2OutID(const DataGraph &dun, const DataGraph &dout) {
    const EdgeID *offset = dun.getOffsets();
    const VertexID *nbors = dun.getNbors();
    EdgeID *unID2OutID = new EdgeID[dun.getNumEdges()];
    for (VertexID v1 = 0; v1 < dun.getNumVertices(); ++v1) {
        for (EdgeID eUn = offset[v1]; eUn < offset[v1 + 1]; ++eUn) {
            VertexID v2 = nbors[eUn];
            EdgeID eOut = dout.getEdgeID(v1, v2);
            unID2OutID[eUn] = eOut;
        }
    }

    return unID2OutID;
}

EdgeID *buildReverseUnID(const DataGraph &dun) {
    const EdgeID *offset = dun.getOffsets();
    const VertexID *nbors = dun.getNbors();
    EdgeID *reverseUnID = new EdgeID[dun.getNumEdges()];
    for (VertexID v1 = 0; v1 < dun.getNumVertices(); ++v1) {
        for (EdgeID e12 = offset[v1]; e12 < offset[v1 + 1]; ++e12) {
            VertexID v2 = nbors[e12];
            if (v2 < v1) continue;
            EdgeID e21 = dun.getUndirectedEID(v2, v1);
            reverseUnID[e12] = e21;
            reverseUnID[e21] = e12;
        }
    }

    return reverseUnID;
}

int numDAG(const PatternGraph &p, const std::vector<VertexID> &v) {
    std::vector<PatternGraph> din;
    std::vector<PatternGraph> dout;
    long minarcs = 0, maxarcs = NOLIMIT;
    splitmod = 1;
    splitres = 0;
    int nfixed = 0;
    DYNALLSTAT(graph,g,g_sz);
    static DEFAULTOPTIONS_GRAPH(options);
    statsblk stats;
    int n = int(v.size());
    int m = SETWORDSNEEDED(n);
    nauty_check(WORDSIZE,m,n,NAUTYVERSIONID);
    DYNALLOC2(graph,g,g_sz,m,n,"malloc");
    EMPTYGRAPH(g,m,n);
    // build graph from core vertices
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (p.isEdge(v[i], v[j]))
                    ADDONEEDGE(g, i, j, m);
        }
    }

    direct(g,nfixed,minarcs,maxarcs,m,n,din,dout);
    return din.size();
}

bool isConnectedSubgraph(std::vector<VertexID> vertices, const PatternGraph &p) {
    std::vector<bool> visited(p.getNumVertices(), false);
    visited[vertices[0]] = true;
    std::queue<VertexID> q;
    q.push(vertices[0]);
    while (!q.empty()) {
        VertexID u = q.front();
        q.pop();
        for (VertexID w: vertices) {
            if (p.isEdge(u, w) && !visited[w]) {
                visited[w] = true;
                q.push(w);
            }
        }
    }
    for (VertexID vertex: vertices) {
        if (!visited[vertex]) return false;
    }

    return true;
}