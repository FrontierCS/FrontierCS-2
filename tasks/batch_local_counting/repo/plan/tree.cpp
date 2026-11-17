//
// Created by anonymous author on 2022/8/2.
//

#include "tree.h"
#include "decompose.h"

double gTDTime = 0.0;

Node::Node() {
    id = 0;
    vertices = nullptr;
    numVertices = 0;
    numSources = 0;
    cut = nullptr;
    cutSize = 0;
    prefix = nullptr;
    prefixSize = 0;
    key = nullptr;
    keySize = 0;
    v2o = nullptr;
    vertexOrbit = nullptr;
    numRules = 0;
    subPatternRules = 0;
    automorphisms = nullptr;
    autoSize = 0;
    canonValue = 0;
    subPatternCanon = 0;
    keyOrbit = 0;
    edgeKey = false;
    unEdgeKey = false;
    numIn = 0;
    fw = 0.0;
}

Node::Node(int id, VertexID *vertices, ui numVertices, ui numSources)
        : id(id), vertices(vertices), numVertices(numVertices), numSources(numSources) {
    cut = nullptr;
    cutSize = 0;
    prefix = nullptr;
    prefixSize = 0;
    key = nullptr;
    keySize = 0;
    v2o = nullptr;
    vertexOrbit = nullptr;
    numRules = 0;
    subPatternRules = 0;
    automorphisms = nullptr;
    autoSize = 0;
    canonValue = 0;
    subPatternCanon = 0;
    keyOrbit = 0;
    edgeKey = false;
    unEdgeKey = false;
    numIn = 0;
    fw = 0.0;
}

Node::Node(const Node &rhs) {
    id = rhs.id;
    numVertices = rhs.numVertices;
    numSources = rhs.numSources;
    cutSize = rhs.cutSize;
    prefixSize = rhs.prefixSize;
    keySize = rhs.keySize;
    localOrder = rhs.localOrder;
    nodeOrder = rhs.nodeOrder;
    autoSize = rhs.autoSize;
    canonValue = rhs.canonValue;
    subPatternCanon = rhs.subPatternCanon;
    keyOrbit = rhs.keyOrbit;
    numRules = rhs.numRules;
    subPatternRules = rhs.subPatternRules;
    prefixOrbit = rhs.prefixOrbit;
    validRules = rhs.validRules;
    candidateRules = rhs.candidateRules;
    edgeKey = rhs.edgeKey;
    unEdgeKey = rhs.unEdgeKey;
    numIn = rhs.numIn;
    fw = rhs.fw;
    if (rhs.vertices == nullptr)
        vertices = nullptr;
    else {
        vertices = new VertexID[numVertices];
        memcpy(vertices, rhs.vertices, sizeof(VertexID) * numVertices);
    }
    if (cutSize == 0)
        cut = nullptr;
    else {
        cut = new VertexID[cutSize];
        memcpy(cut, rhs.cut, sizeof(VertexID) * cutSize);
    }
    if (prefixSize == 0)
        prefix = nullptr;
    else {
        prefix = new VertexID[prefixSize];
        memcpy(prefix, rhs.prefix, sizeof(VertexID) * prefixSize);
    }
    if (keySize == 0)
        key = nullptr;
    else {
        key = new VertexID[keySize];
        memcpy(key, rhs.key, sizeof(VertexID) * keySize);
    }
    if (rhs.automorphisms == nullptr) automorphisms = nullptr;
    else {
        automorphisms = new VertexID *[autoSize];
        for (int i = 0; i < autoSize; ++i) {
            automorphisms[i] = new VertexID[numVertices];
            memcpy(automorphisms[i], rhs.automorphisms[i], sizeof(VertexID) * numVertices);
        }
    }
    if (rhs.v2o == nullptr) v2o = nullptr;
    else {
        v2o = new int[MAX_PATTERN_SIZE];
        memcpy(v2o, rhs.v2o, sizeof(int) * MAX_PATTERN_SIZE);
    }
    if (rhs.vertexOrbit == nullptr) vertexOrbit = nullptr;
    else {
        vertexOrbit = new int[MAX_PATTERN_SIZE];
        memcpy(vertexOrbit, rhs.vertexOrbit, sizeof(int) * MAX_PATTERN_SIZE);
    }
}

Node &Node::operator=(const Node &rhs) {
    if (this == &rhs)
        return *this;
    id = rhs.id;
    numVertices = rhs.numVertices;
    cutSize = rhs.cutSize;
    prefixSize = rhs.prefixSize;
    keySize = rhs.keySize;
    localOrder = rhs.localOrder;
    nodeOrder = rhs.nodeOrder;
    autoSize = rhs.autoSize;
    canonValue = rhs.canonValue;
    subPatternCanon = rhs.subPatternCanon;
    keyOrbit = rhs.keyOrbit;
    numRules = rhs.numRules;
    subPatternRules = rhs.subPatternRules;
    prefixOrbit = rhs.prefixOrbit;
    validRules = rhs.validRules;
    candidateRules = rhs.candidateRules;
    edgeKey = rhs.edgeKey;
    unEdgeKey = rhs.unEdgeKey;
    numIn = rhs.numIn;
    fw = rhs.fw;
    if (rhs.vertices == nullptr)
        vertices = nullptr;
    else {
        delete[] vertices;
        vertices = new VertexID[numVertices];
        memcpy(vertices, rhs.vertices, sizeof(VertexID) * numVertices);
    }
    if (cutSize == 0)
        cut = nullptr;
    else {
        delete[] cut;
        cut = new VertexID[cutSize];
        memcpy(cut, rhs.cut, sizeof(VertexID) * cutSize);
    }
    if (prefixSize == 0)
        prefix = nullptr;
    else {
        delete[] prefix;
        prefix = new VertexID[prefixSize];
        memcpy(prefix, rhs.prefix, sizeof(VertexID) * prefixSize);
    }
    if (keySize == 0)
        key = nullptr;
    else {
        delete[] key;
        key = new VertexID[keySize];
        memcpy(key, rhs.key, sizeof(VertexID) * keySize);
    }
    if (rhs.automorphisms == nullptr) automorphisms = nullptr;
    else {
        if (automorphisms) {
            for (int i = 0; i < autoSize; ++i) {
                delete[] automorphisms[i];
            }
            delete[] automorphisms;
        }
        automorphisms = new VertexID *[autoSize];
        for (int i = 0; i < autoSize; ++i) {
            automorphisms[i] = new VertexID[numVertices];
            memcpy(automorphisms[i], rhs.automorphisms[i], sizeof(VertexID) * numVertices);
        }
    }
    if (rhs.v2o == nullptr) v2o = nullptr;
    else {
        delete[] v2o;
        v2o = new int[MAX_PATTERN_SIZE];
        memcpy(v2o, rhs.v2o, sizeof(int) * MAX_PATTERN_SIZE);
    }
    if (rhs.vertexOrbit == nullptr) vertexOrbit = nullptr;
    else {
        delete[] vertexOrbit;
        vertexOrbit = new int[MAX_PATTERN_SIZE];
        memcpy(vertexOrbit, rhs.vertexOrbit, sizeof(int) * MAX_PATTERN_SIZE);
    }
    return *this;
}

Node::~Node() {
    if (automorphisms) {
        for (int i = 0; i < autoSize; ++i) {
            delete[] automorphisms[i];
        }
        delete[] automorphisms;
    }
    delete[] vertices;
    delete[] cut;
    delete[] prefix;
    delete[] key;
    delete[] v2o;
    delete[] vertexOrbit;
}

// vertices in this and rhs are sorted
// this < rhs: vertices in this is a subset of vertices in rhs
bool Node::operator<(const Node &rhs) const {
    if (rhs.numVertices < numVertices) return false;
    return std::includes(rhs.vertices, rhs.vertices + rhs.numVertices, vertices, vertices + numVertices);
}

// vertices in this and rhs are sorted
// this > rhs: vertices in rhs is a subset of vertices in this
bool Node::operator>(const Node &rhs) const {
    if (rhs.numVertices > numVertices) return false;
    return std::includes(vertices, vertices + numVertices, rhs.vertices, rhs.vertices + rhs.numVertices);
}

// this == rhs: the induces subgraphs are isomorphic,
// cut and key are isomorphic (i.e. the orbits of cut/key is the same)
bool Node::operator==(const Node &rhs) const {
    // check whether induced subgraphs are isomorphic
    if (canonValue != rhs.canonValue) return false;
    // check cut size and key size
    if (cutSize != rhs.cutSize) return false;
    if (keySize != rhs.keySize) return false;
    int maxO = 0;
    for (int i = 0; i < numVertices; ++i) {
        if (v2o[vertices[i]] > maxO) maxO = v2o[vertices[i]];
    }
    int *oCntL = new int[maxO + 1];
    int *oCntR = new int[maxO + 1];
    memset(oCntL, 0, sizeof(int) * (maxO + 1));
    memset(oCntR, 0, sizeof(int) * (maxO + 1));
    // check whether orbits of cut are equivalent
    for (int i = 0; i < cutSize; ++i) {
        VertexID u1 = cut[i], u2 = rhs.cut[i];
        ++oCntL[v2o[u1]];
        ++oCntR[rhs.v2o[u2]];
    }
    for (int i = 0; i < maxO; ++i) {
        if (oCntL[i] != oCntR[i]) {
            delete[] oCntL;
            delete[] oCntR;
            return false;
        }
    }
    // check whether orbits of the key are equivalent
    if (keySize != 0) {
        memset(oCntL, 0, sizeof(int) * maxO);
        memset(oCntR, 0, sizeof(int) * maxO);
        for (int i = 0; i < keySize; ++i) {
            VertexID u1 = key[i], u2 = rhs.key[i];
            ++oCntL[v2o[u1]];
            ++oCntR[rhs.v2o[u2]];
        }
        for (int i = 0; i < maxO; ++i) {
            if (oCntL[i] != oCntR[i]) {
                delete[] oCntL;
                delete[] oCntR;
                return false;
            }
        }
    }

    delete[] oCntL;
    delete[] oCntR;
    return true;
}

bool Node::hasVertex(VertexID u) const {
    for (ui i = 0; i < numVertices; i++)
        if (vertices[i] == u) return true;
    return false;
}

void Node::allgroup(grouprec *grp, const std::function<void(int*)>& action) {
    int i,depth,n;

    depth = grp->depth;
    n = grp->n;
    DYNALLSTAT(int,allp,allp_sz);
    DYNALLSTAT(int,identity,id_sz);
    DYNALLOC1(int,identity,id_sz,n,"malloc");
    for (i = 0; i < n; ++i) identity[i] = i;

    if (depth == 0)
    {
        action(identity);
        return;
    }

    DYNALLOC1(int,allp,allp_sz,n*depth,"malloc");

    groupelts(grp->levelinfo,n,depth-1,action,NULL,allp,identity);
}

void Node::groupelts(levelrec *lr, int n, int level, const std::function<void(int*)>& action,
                     int *before, int *after, int *identity) {
    int i,j,orbsize;
    int *p,*cr;
    cosetrec *coset;

    coset = lr[level].replist;
    orbsize = lr[level].orbitsize;

    for (j = 0; j < orbsize; ++j)
    {
        cr = (coset[j].rep == NULL ? NULL : coset[j].rep->p);
        if (before == NULL)
            p = cr;
        else if (cr == NULL)
            p = before;
        else
        {
            p = after;
            for (i = 0; i < n; ++i) p[i] = cr[before[i]];
        }

        if (level == 0)
            action((p == NULL ? identity : p));
        else
            groupelts(lr, n,level-1, action, p,after+n, identity);
    }
}

void Node::copyAutom(int *p) {
    for (int i = 0; i < numVertices; ++i) {
        automorphisms[autoSize][i] = p[i];
    }
    ++autoSize;
}

// compute the canonical labeling and value and all automorphisms
// handle both directed and undirected cases
// compress the adj matrix of the canonical graph to a single int, stored in canonValue
void Node::getCanonLabel(const PatternGraph &p) {
    // relabel vertices to [0, numVertices - 1]
    ui maxN = p.getNumVertices();
    VertexID *old2new = new VertexID[maxN];
    VertexID *new2old = new VertexID[maxN];
    for (ui i = 0; i < numVertices; ++i) {
        old2new[vertices[i]] = i;
        new2old[i] = vertices[i];
    }
    DYNALLSTAT(graph,g,g_sz);
    DYNALLSTAT(int,lab,lab_sz);
    DYNALLSTAT(int,ptn,ptn_sz);
    DYNALLSTAT(int,orbits,orbits_sz);
    DYNALLSTAT(graph,cg,cg_sz);
    static DEFAULTOPTIONS_GRAPH(options);
    statsblk stats;
    options.getcanon = TRUE;
    options.digraph = TRUE;
    int n = (int)numVertices;
    // m should be 1
    int m = SETWORDSNEEDED(n);
    nauty_check(WORDSIZE,m,n,NAUTYVERSIONID);
    DYNALLOC2(graph,g,g_sz,m,n,"malloc");
    DYNALLOC2(graph,cg,cg_sz,m,n,"malloc");
    DYNALLOC1(int,lab,lab_sz,n,"malloc");
    DYNALLOC1(int,ptn,ptn_sz,n,"malloc");
    DYNALLOC1(int,orbits,orbits_sz,n,"malloc");
    EMPTYGRAPH(g,m,n);
    for (int i = 0; i < numVertices; ++i) {
        for (int j = 0; j < numVertices; ++j) {
            VertexID u1 = vertices[i], u2 = vertices[j];
            if (p.isEdge(u1, u2)) {
                        ADDONEARC(g, old2new[u1], old2new[u2], m);
            }
        }
    }
    densenauty(g,lab,ptn,orbits,&options,&stats,m,n,cg);
    int pos = 0;
    canonValue = 0;
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
    std::map<VertexID, int> canonV2O;
    int *new2Canon = new int[numVertices];
    for (int i = 0; i < numVertices; ++i) {
        new2Canon[lab[i]] = i;
    }
    // group canonical vertices according to orbits and reorder groups according to the min canonical vertex id
    int o = 0;
    std::vector<std::vector<int>> canonGroup(maxN);
    std::vector<int> canonV2Group(maxN);
    std::vector<int> group2O(maxN, -1);
    for (int i = 0; i < numVertices; ++i) {
        canonGroup[orbits[i]].push_back(new2Canon[i]);
        canonV2Group[new2Canon[i]] = orbits[i];
    }
    for (int i = 0; i < numVertices; ++i) {
        int group = canonV2Group[i];
        if (group2O[group] == -1) {
            group2O[group] = o;
            ++o;
        }
    }
    for (int i = 0; i < maxN; ++i) {
        for (int u: canonGroup[i]) {
            canonV2O[u] = group2O[i];
        }
    }
    delete[] v2o;
    v2o = new int[MAX_PATTERN_SIZE];
    for (int i = 0; i < numVertices; ++i) {
        v2o[new2old[i]] = canonV2O[new2Canon[i]];
    }
    autoSize = (int)round(stats.grpsize1) * quickPow10(stats.grpsize2);

    delete[] new2Canon;
    delete[] old2new;
    delete[] new2old;
}

void Node::computeValidRules(const PatternGraph &p, std::set<VertexID> &allCut) {
    std::set<int> nodeRules;
    // compute all rules for the subpattern inside the node
    ui maxN = p.getNumVertices();
    VertexID *old2new = new VertexID[maxN];
    VertexID *new2old = new VertexID[maxN];
    for (ui i = 0; i < numVertices; ++i) {
        old2new[vertices[i]] = i;
        new2old[i] = vertices[i];
    }
    DYNALLSTAT(graph,g,g_sz);
    DYNALLSTAT(int,lab,lab_sz);
    DYNALLSTAT(int,ptn,ptn_sz);
    DYNALLSTAT(int,orbits,orbits_sz);
    static DEFAULTOPTIONS_GRAPH(options);
    statsblk stats;
    options.defaultptn = FALSE;
    options.userautomproc = groupautomproc;
    options.userlevelproc = grouplevelproc;
    int n = (int)numVertices;
    // m should be 1
    int m = SETWORDSNEEDED(n);
    nauty_check(WORDSIZE,m,n,NAUTYVERSIONID);
    DYNALLOC2(graph,g,g_sz,m,n,"malloc");
    DYNALLOC1(int,lab,lab_sz,n,"malloc");
    DYNALLOC1(int,ptn,ptn_sz,n,"malloc");
    DYNALLOC1(int,orbits,orbits_sz,n,"malloc");
    EMPTYGRAPH(g,m,n);
    for (int i = 0; i < numVertices; ++i) {
        for (int j = 0; j < numVertices; ++j) {
            VertexID u1 = vertices[i], u2 = vertices[j];
            if (p.isEdge(u1, u2)) {
                        ADDONEARC(g, old2new[u1], old2new[u2], m);
            }
        }
    }
    std::vector<int> emptyV;     // each vertex in colored V is in a distinct cell (has a distinct color)
    std::queue<std::vector<int>> coloredQ;
    coloredQ.push(emptyV);
    while (!coloredQ.empty()) {
        std::vector<int> coloredV = coloredQ.front();
        coloredQ.pop();
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
        if (coloredV.empty()) {
            // store all automorphisms
            automorphisms = new VertexID *[autoSize];
            for (int i = 0; i < autoSize; ++i) {
                automorphisms[i] = new VertexID[numVertices];
            }
            autoSize = 0;
            grouprec *group;
            group = groupptr(FALSE);
            makecosetreps(group);
            allgroup(group, [this](int* p){ copyAutom(p); });
        }
        int currentAutoSize = (int)round(stats.grpsize1) * quickPow10(stats.grpsize2);
        if (currentAutoSize != 1) {
            std::vector<std::vector<VertexID>> o2v = std::vector<std::vector<VertexID>>(n);
            for (int i = 0; i < n; ++i)
                o2v[orbits[i]].push_back(i);
            for (const auto &group: o2v) {
                if (group.size() > 1) {
                    bool cutFixed = false;
                    // check whether there exists automorphisms fixing all cut
                    std::set<VertexID> tmp = allCut;
                    for (int u: group)
                        tmp.erase(new2old[u]);
                    for (int i = 1; i < autoSize; ++i) {
                        // whether this automorphism fixes u_
                        bool fixed = true;
                        for (VertexID u_: tmp) {
                            if (!(automorphisms[i][old2new[u_]] == old2new[u_])) {
                                fixed = false;
                                break;
                            }
                        }
                        if (!fixed) continue;
                        // whether this automorphism contains a cycle of the group
                        int cycleLength = 1;
                        VertexID u = automorphisms[i][group[0]];
                        while (u != group[0]) {
                            ++cycleLength;
                            u = automorphisms[i][u];
                            if (std::find(group.begin(), group.end(), u) == group.end()) {
                                cycleLength = 0;
                                u = group[0];
                            }
                        }
                        if (cycleLength == group.size()) {
                            cutFixed = true;
                            break;
                        }
                    }
                    int groupID = 0;
                    for (int u: group)
                        groupID += 1 << new2old[u];
                    if (cutFixed) nodeRules.insert(groupID);
                    for (VertexID u : group) {
                        std::vector<int> newColoredV = coloredV;
                        newColoredV.push_back(u);
                        coloredQ.push(newColoredV);
                    }
                }
            }
        }
    }

    validRules = nodeRules;
    delete[] old2new;
    delete[] new2old;
}

void Node::computeCandidateRules(const PatternGraph &p, const std::vector<VertexID *> &childCuts, const std::vector<ui> &cutSizes) {
    if (numVertices == 2) return;
    std::set<VertexID> allCut;
    for (int i = 0; i < cutSize; ++i) {
        allCut.insert(cut[i]);
    }
    for (int i = 0; i < childCuts.size(); ++i) {
        for (int j = 0; j < cutSizes[i]; ++j) {
            allCut.insert(childCuts[i][j]);
        }
    }
    ui maxN = p.getNumVertices();
    VertexID *old2new = new VertexID[maxN];
    VertexID *new2old = new VertexID[maxN];
    for (ui i = 0; i < numVertices; ++i) {
        old2new[vertices[i]] = i;
        new2old[i] = vertices[i];
    }
    DYNALLSTAT(graph,g,g_sz);
    DYNALLSTAT(int,lab,lab_sz);
    DYNALLSTAT(int,ptn,ptn_sz);
    DYNALLSTAT(int,orbits,orbits_sz);
    static DEFAULTOPTIONS_GRAPH(options);
    statsblk stats;
    options.defaultptn = FALSE;
    options.userautomproc = groupautomproc;
    options.userlevelproc = grouplevelproc;
    int n = (int)numVertices;
    // m should be 1
    int m = SETWORDSNEEDED(n);
    nauty_check(WORDSIZE,m,n,NAUTYVERSIONID);
    DYNALLOC2(graph,g,g_sz,m,n,"malloc");
    DYNALLOC1(int,lab,lab_sz,n,"malloc");
    DYNALLOC1(int,ptn,ptn_sz,n,"malloc");
    DYNALLOC1(int,orbits,orbits_sz,n,"malloc");
    EMPTYGRAPH(g,m,n);
    int numEdges = 0;
    for (int i = 0; i < numVertices; ++i) {
        for (int j = 0; j < numVertices; ++j) {
            VertexID u1 = vertices[i], u2 = vertices[j];
            if (p.isEdge(u1, u2)) {
                        ADDONEARC(g, old2new[u1], old2new[u2], m);
                ++numEdges;
            }
        }
    }
    if (numEdges == n * (n - 1)) {
        cliqueNodeRules(p, childCuts, cutSizes);
        delete[] old2new;
        delete[] new2old;
        return;
    }
    std::vector<int> emptyV;     // each vertex in colored V is in a distinct cell (has a distinct color)
    std::queue<std::vector<int>> coloredQ;
    std::queue<std::vector<std::vector<VertexID>>> ruleQ;
    std::vector<std::vector<std::vector<VertexID>>> allRule;
    coloredQ.push(emptyV);
    ruleQ.emplace();
    while (!coloredQ.empty()) {
        std::vector<int> coloredV = coloredQ.front();
        std::vector<std::vector<VertexID>> currentRule = ruleQ.front();
        coloredQ.pop();
        ruleQ.pop();
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
        if (coloredV.empty()) {
            // store all automorphisms
            automorphisms = new VertexID *[autoSize];
            for (int i = 0; i < autoSize; ++i) {
                automorphisms[i] = new VertexID[numVertices];
            }
            autoSize = 0;
            grouprec *group;
            group = groupptr(FALSE);
            makecosetreps(group);
            allgroup(group, [this](int* p){ copyAutom(p); });
        }
        int currentAutoSize = (int)round(stats.grpsize1) * quickPow10(stats.grpsize2);
        if (currentAutoSize != 1) {
            std::vector<std::vector<VertexID>> o2v = std::vector<std::vector<VertexID>>(n);
            for (int i = 0; i < n; ++i)
                o2v[orbits[i]].push_back(i);
            for (const auto &vartheta: o2v) {
                if (vartheta.size() > 1) {
                    bool cutFixed = false;
                    // check whether there exists automorphisms fixing all cut
                    std::set<VertexID> tmp = allCut;
                    for (int u: vartheta)
                        tmp.erase(new2old[u]);
                    for (int i = 1; i < autoSize; ++i) {
                        // whether this automorphism fixes u_
                        bool fixed = true;
                        for (VertexID u_: tmp) {
                            if (!(automorphisms[i][old2new[u_]] == old2new[u_])) {
                                fixed = false;
                                break;
                            }
                        }
                        if (!fixed) continue;
                        // whether this automorphism contains a cycle of the group
                        int cycleLength = 1;
                        VertexID u = automorphisms[i][vartheta[0]];
                        if (std::find(vartheta.begin(), vartheta.end(), u) == vartheta.end()) {
                            cycleLength = 0;
                            u = vartheta[0];
                        }
                        while (u != vartheta[0]) {
                            ++cycleLength;
                            u = automorphisms[i][u];
                            if (std::find(vartheta.begin(), vartheta.end(), u) == vartheta.end()) {
                                cycleLength = 0;
                                u = vartheta[0];
                            }
                        }
                        if (cycleLength == vartheta.size()) {
                            cutFixed = true;
                            break;
                        }
                    }
                    if (vartheta.size() > 2) {
                        for (int u : vartheta) {
                            std::vector<int> newColoredV = coloredV;
                            std::vector<std::vector<VertexID>> newRule = currentRule;
                            newColoredV.push_back(u);
                            std::vector<VertexID> rule;
                            rule.push_back(u);
                            for (int w: vartheta) {
                                if (w != u) rule.push_back(w);
                            }
                            if (cutFixed)
                                newRule.push_back(rule);
                            coloredQ.push(newColoredV);
                            ruleQ.push(newRule);
                        }
                    }
                    else {
                        std::vector<int> newColoredV = coloredV;
                        std::vector<std::vector<VertexID>> newRule = currentRule;
                        newColoredV.push_back(vartheta[0]);
                        if (cutFixed)
                            newRule.push_back(vartheta);
                        coloredQ.push(newColoredV);
                        ruleQ.push(newRule);
                    }
                }
            }
        }
        else {
            if (!currentRule.empty()) allRule.push_back(currentRule);
        }
    }

    // from all Rule, select those with the maximum rule number
    int maxRuleNum = 1;
    for (int i = 0; i < allRule.size(); ++i) {
        int ruleNum = 1;
        for (int j = 0; j < allRule[i].size(); ++j) {
            ruleNum *= (int)allRule[i][j].size();
        }
        if (ruleNum > maxRuleNum) maxRuleNum = ruleNum;
    }
    std::set<std::vector<int>> existingRules;
    for (int i = 0; i < allRule.size(); ++i) {
        int ruleNum = 1;
        for (int j = 0; j < allRule[i].size(); ++j) {
            ruleNum *= (int)allRule[i][j].size();
        }
        if (ruleNum == maxRuleNum) {
            // translate to old id
            std::vector<std::vector<VertexID>> rule(allRule[i].size());
            std::vector<int> ruleID(allRule[i].size());
            for (int j = 0; j < allRule[i].size(); ++j) {
                int varthetaID = 0;
                for (int k = 0; k < allRule[i][j].size(); ++k) {
                    VertexID u = new2old[allRule[i][j][k]];
                    rule[j].push_back(u);
                    ruleID[j] += 1 << u;
                }
            }
            std::sort(ruleID.begin(), ruleID.end());
            if (existingRules.find(ruleID) == existingRules.end()) {
                existingRules.insert(ruleID);
                candidateRules.push_back(rule);
            }
        }
    }
    delete[] old2new;
    delete[] new2old;
}

// make one set of rules for each cut and each non-cut part
void Node::cliqueNodeRules(const PatternGraph &p, const std::vector<VertexID *> &childCuts, const std::vector<ui> &cutSizes) {
    VertexID *current = new VertexID[numVertices];
    bool* used = new bool[numVertices] {false};
    automorphisms = new VertexID *[autoSize];
    for (int i = 0; i < autoSize; ++i) {
        automorphisms[i] = new VertexID[numVertices];
    }
    size_t index = 0;
    generatePermutations(numVertices, automorphisms, current, used, index, 0);
    delete[] current;
    delete[] used;
    candidateRules.clear();
    std::set<VertexID> internalVertices(vertices, vertices + numVertices);
    for (int i = 0; i < cutSize; ++i) {
        internalVertices.erase(cut[i]);
    }
    if (cutSize > 1) {
        std::vector<VertexID> v(cut, cut + cutSize);
        candidateRules = generateCliqueRules(v, candidateRules);
    }
    for (int i = 0; i < childCuts.size(); ++i) {
        for (int j = 0; j < cutSizes[i]; ++j) {
            internalVertices.erase(childCuts[i][j]);
        }
        if (cutSizes[i] > 1) {
            std::vector<VertexID> v(childCuts[i], childCuts[i] + cutSizes[i]);
            candidateRules = generateCliqueRules(v, candidateRules);
        }
    }
    std::vector<VertexID> v(internalVertices.begin(), internalVertices.end());
    candidateRules = generateCliqueRules(v, candidateRules);
}

void Node::readFromStream(std::ifstream &inFile) {
    // Read simple types directly
    inFile.read(reinterpret_cast<char*>(&id), sizeof(id));
    inFile.read(reinterpret_cast<char*>(&numVertices), sizeof(numVertices));
    inFile.read(reinterpret_cast<char*>(&numSources), sizeof(numSources));
    inFile.read(reinterpret_cast<char*>(&cutSize), sizeof(cutSize));
    inFile.read(reinterpret_cast<char*>(&prefixSize), sizeof(prefixSize));
    inFile.read(reinterpret_cast<char*>(&keySize), sizeof(keySize));
    inFile.read(reinterpret_cast<char*>(&autoSize), sizeof(autoSize));
    inFile.read(reinterpret_cast<char*>(&canonValue), sizeof(canonValue));
    inFile.read(reinterpret_cast<char*>(&subPatternCanon), sizeof(subPatternCanon));
    inFile.read(reinterpret_cast<char*>(&keyOrbit), sizeof(keyOrbit));
    inFile.read(reinterpret_cast<char*>(&numRules), sizeof(numRules));
    inFile.read(reinterpret_cast<char*>(&subPatternRules), sizeof(subPatternRules));
    inFile.read(reinterpret_cast<char*>(&edgeKey), sizeof(edgeKey));
    inFile.read(reinterpret_cast<char*>(&unEdgeKey), sizeof(unEdgeKey));
    inFile.read(reinterpret_cast<char*>(&numIn), sizeof(numIn));
    inFile.read(reinterpret_cast<char*>(&fw), sizeof(fw));

    // Read arrays
    readArrayFromStream(inFile, vertices, numVertices);
    readArrayFromStream(inFile, cut, cutSize);
    readArrayFromStream(inFile, prefix, prefixSize);
    readArrayFromStream(inFile, key, keySize);
    readArrayFromStream(inFile, v2o, numVertices);  // Assuming numVertices is the correct size

    // Read vectors
    readVectorFromStream(inFile, localOrder);
    readVectorFromStream(inFile, nodeOrder);
    // Read 2D array (automorphisms)
    read2DArrayFromStream(inFile, automorphisms, autoSize, numVertices);
    // Read complex nested vector (candidateRules)
    read3DVectorFromStream(inFile, candidateRules);
}

void Node::writeToStream(std::ofstream &outFile) {
    if (automorphisms == nullptr) autoSize = 0;
    // Write simple types directly
    outFile.write(reinterpret_cast<const char*>(&id), sizeof(id));
    outFile.write(reinterpret_cast<const char*>(&numVertices), sizeof(numVertices));
    outFile.write(reinterpret_cast<const char*>(&numSources), sizeof(numSources));
    outFile.write(reinterpret_cast<const char*>(&cutSize), sizeof(cutSize));
    outFile.write(reinterpret_cast<const char*>(&prefixSize), sizeof(prefixSize));
    outFile.write(reinterpret_cast<const char*>(&keySize), sizeof(keySize));
    outFile.write(reinterpret_cast<const char*>(&autoSize), sizeof(autoSize));
    outFile.write(reinterpret_cast<const char*>(&canonValue), sizeof(canonValue));
    outFile.write(reinterpret_cast<const char*>(&subPatternCanon), sizeof(subPatternCanon));
    outFile.write(reinterpret_cast<const char*>(&keyOrbit), sizeof(keyOrbit));
    outFile.write(reinterpret_cast<const char*>(&numRules), sizeof(numRules));
    outFile.write(reinterpret_cast<const char*>(&subPatternRules), sizeof(subPatternRules));
    outFile.write(reinterpret_cast<const char*>(&edgeKey), sizeof(edgeKey));
    outFile.write(reinterpret_cast<const char*>(&unEdgeKey), sizeof(unEdgeKey));
    outFile.write(reinterpret_cast<const char*>(&numIn), sizeof(numIn));
    outFile.write(reinterpret_cast<const char*>(&fw), sizeof(fw));

    // Write arrays
    writeArrayToStream(outFile, vertices, numVertices);
    writeArrayToStream(outFile, cut, cutSize);
    writeArrayToStream(outFile, prefix, prefixSize);
    writeArrayToStream(outFile, key, keySize);
    writeArrayToStream(outFile, v2o, numVertices);  // Assuming numVertices is the correct size

    // Write vectors
    writeVectorToStream(outFile, localOrder);
    writeVectorToStream(outFile, nodeOrder);
    // Write 2D array (automorphisms)
    write2DArrayToStream(outFile, automorphisms, autoSize, numVertices);
    // Write complex nested vector (candidateRules)
    write3DVectorToStream(outFile, candidateRules);
}

Tree::Tree() {
    _numVertices = 0;
    _multiFactor = 1;
    _nodes = nullptr;
    _v2n = nullptr;
    _numNodes = 0;
    _numRules = 0;
    _parent = nullptr;
    _orbitType = 0;
    _treeWidth = 0;
    _sumWidth = 0;
    _maxNumSource = 0;
    _fhw = 0.0;
    _executeMode = true;
}

Tree::Tree(ui numVertices): _numVertices(numVertices) {
    _multiFactor = 1;
    _nodes = new Node[_numVertices];
    _v2n = new std::vector<ui>[_numVertices];
    _numNodes = 0;
    _edges = std::vector<std::vector<VertexID>> (_numVertices);
    _numRules = 0;
    _greaterRules = std::vector<std::vector<VertexID>>(numVertices);
    _lessRules = std::vector<std::vector<VertexID>>(numVertices);
    _parent = nullptr;
    _orbitType = 0;
    _treeWidth = 0;
    _sumWidth = 0;
    _maxNumSource = 0;
    _fhw = 0.0;
    _executeMode = true;
}

// only calls this constructor when building a two-level decomposition tree
Tree::Tree(const Node &root, const std::vector<Node> &nodes, ui numVertices): _numVertices(numVertices) {
    _multiFactor = 1;
    _numNodes = nodes.size() + 1;
    _nodes = new Node[_numNodes];
    _v2n = new std::vector<ui>[_numVertices];
    _edges = std::vector<std::vector<VertexID>>(_numNodes);
    _child = std::vector<std::vector<VertexID>>(_numNodes);
    _nodes[0] = root;
    _numRules = 0;
    _greaterRules = std::vector<std::vector<VertexID>>(numVertices);
    _lessRules = std::vector<std::vector<VertexID>>(numVertices);
    _parent = new VertexID[_numNodes];
    _parent[0] = 99;
    for (int i = 0; i < nodes.size(); ++i) {
        _nodes[i + 1] = nodes[i];
        _edges[i + 1].push_back(0);
        _edges[0].push_back(i + 1);
        _parent[i + 1] = 0;
        _child[0].push_back(i + 1);
    }
    _orbitType = 0;
    _treeWidth = 0;
    _sumWidth = 0;
    _maxNumSource = 0;
    _fhw = 0.0;
    _executeMode = true;
}

Tree::Tree(const Tree &rhs) {
    _multiFactor = rhs._multiFactor;
    _numVertices = rhs._numVertices;
    _numNodes = rhs._numNodes;
    _edges = rhs._edges;
    _child = rhs._child;
    _orbitType = rhs._orbitType;
    _numRules = rhs._numRules;
    _greaterRules = rhs._greaterRules;
    _lessRules = rhs._lessRules;
    _aggreV = rhs._aggreV;
    _aggreWeight = rhs._aggreWeight;
    _treeWidth = rhs._treeWidth;
    _sumWidth = rhs._sumWidth;
    _maxNumSource = rhs._maxNumSource;
    _fhw = rhs._fhw;
    _postOrder = rhs._postOrder;
    _partitionPos = rhs._partitionPos;
    _globalOrder = rhs._globalOrder;
    _executeMode = rhs._executeMode;
    _nodesAtStep = rhs._nodesAtStep;
    _prefixPos = rhs._prefixPos;
    _aggrePos = rhs._aggrePos;
    _partitionInterPos = rhs._partitionInterPos;
    _nodeInterPos = rhs._nodeInterPos;
    _nodeInPos = rhs._nodeInPos;
    _nodeOutPos = rhs._nodeOutPos;
    _nodeUnPos = rhs._nodeUnPos;
    _nodeGreaterPos = rhs._nodeGreaterPos;
    _nodeLessPos = rhs._nodeLessPos;
    _partitionInPos = rhs._partitionInPos;
    _partitionOutPos = rhs._partitionOutPos;
    _partitionUnPos = rhs._partitionUnPos;
    _partitionGreaterPos = rhs._partitionGreaterPos;
    _partitionLessPos = rhs._partitionLessPos;
    _childKeyPos = rhs._childKeyPos;
    _posChildEdge = rhs._posChildEdge;
    _posAggreEdge = rhs._posAggreEdge;
    _childEdgeType = rhs._childEdgeType;
    _aggreEdgeType = rhs._aggreEdgeType;
    _nodeCandPos = rhs._nodeCandPos;
    _nodeTriPos = rhs._nodeTriPos;
    _triEdgeType = rhs._triEdgeType;
    _triEndType = rhs._triEndType;
    _partitionCandPos = rhs._partitionCandPos;
    _partitionTriPos = rhs._partitionTriPos;
    _partitionEdgeType = rhs._partitionEdgeType;
    _partitionEndType = rhs._partitionEndType;
    if (rhs._v2n == nullptr) {
        _v2n = nullptr;
    }
    else {
        _v2n = new std::vector<ui>[_numVertices];
        for (ui i = 0; i < _numVertices; ++i)
            _v2n[i] = rhs._v2n[i];
    }
    if (_numVertices == 0)
        _nodes = nullptr;
    else {
        _nodes = new Node[_numVertices];
        for (int i = 0; i < _numNodes; ++i) {
            _nodes[i] = rhs._nodes[i];
        }
    }
    if (rhs._parent == nullptr) {
        _parent = nullptr;
    }
    else {
        _parent = new VertexID[_numNodes];
        memcpy(_parent, rhs._parent, sizeof(VertexID) * _numNodes);
    }
}

Tree &Tree::operator=(const Tree &rhs) {
    if (this == &rhs)
        return *this;
    _multiFactor = rhs._multiFactor;
    _numVertices = rhs._numVertices;
    _numNodes = rhs._numNodes;
    _edges = rhs._edges;
    _child = rhs._child;
    _orbitType = rhs._orbitType;
    _numRules = rhs._numRules;
    _greaterRules = rhs._greaterRules;
    _lessRules = rhs._lessRules;
    _aggreV = rhs._aggreV;
    _aggreWeight = rhs._aggreWeight;
    _treeWidth = rhs._treeWidth;
    _sumWidth = rhs._sumWidth;
    _maxNumSource = rhs._maxNumSource;
    _fhw = rhs._fhw;
    _postOrder = rhs._postOrder;
    _partitionPos = rhs._partitionPos;
    _globalOrder = rhs._globalOrder;
    _executeMode = rhs._executeMode;
    _nodesAtStep = rhs._nodesAtStep;
    _childKeyPos = rhs._childKeyPos;
    _aggrePos = rhs._aggrePos;
    _partitionInterPos = rhs._partitionInterPos;
    _nodeInterPos = rhs._nodeInterPos;
    _nodeInPos = rhs._nodeInPos;
    _nodeOutPos = rhs._nodeOutPos;
    _nodeUnPos = rhs._nodeUnPos;
    _nodeGreaterPos = rhs._nodeGreaterPos;
    _nodeLessPos = rhs._nodeLessPos;
    _partitionInPos = rhs._partitionInPos;
    _partitionOutPos = rhs._partitionOutPos;
    _partitionUnPos = rhs._partitionUnPos;
    _partitionGreaterPos = rhs._partitionGreaterPos;
    _partitionLessPos = rhs._partitionLessPos;
    _childKeyPos = rhs._childKeyPos;
    _posChildEdge = rhs._posChildEdge;
    _posAggreEdge = rhs._posAggreEdge;
    _childEdgeType = rhs._childEdgeType;
    _aggreEdgeType = rhs._aggreEdgeType;
    _nodeCandPos = rhs._nodeCandPos;
    _nodeTriPos = rhs._nodeTriPos;
    _triEdgeType = rhs._triEdgeType;
    _triEndType = rhs._triEndType;
    _partitionCandPos = rhs._partitionCandPos;
    _partitionTriPos = rhs._partitionTriPos;
    _partitionEdgeType = rhs._partitionEdgeType;
    _partitionEndType = rhs._partitionEndType;
    if (rhs._v2n == nullptr) {
        _v2n = nullptr;
    }
    else {
        delete[] _v2n;
        _v2n = new std::vector<ui>[_numVertices];
        for (ui i = 0; i < _numVertices; ++i)
            _v2n[i] = rhs._v2n[i];
    }
    if (_numVertices == 0)
        _nodes = nullptr;
    else {
        delete[] _nodes;
        _nodes = new Node[_numVertices];
        for (int i = 0; i < _numNodes; ++i) {
            _nodes[i] = rhs._nodes[i];
        }
    }
    if (rhs._parent == nullptr) {
        _parent = nullptr;
    }
    else {
        delete[] _parent;
        _parent = new VertexID[_numNodes];
        memcpy(_parent, rhs._parent, sizeof(VertexID) * _numNodes);
    }

    return *this;
}

// check whether two unrooted labeled trees are isomorphic

bool Tree::operator==(const Tree &rhs) const {
    if (_numNodes != rhs._numNodes) return false;
    std::map<CanonType, int> canon2Color;
    std::vector<std::vector<ui>> color2NodeL = std::vector<std::vector<ui>>(_numNodes);
    std::vector<std::vector<ui>> color2NodeR = std::vector<std::vector<ui>>(_numNodes);
    int color = 0;
    // convert canonical value to labels
    for (ui i = 0; i < _numNodes; ++i) {
        CanonType canonV = _nodes[i].canonValue;
        if (canon2Color.find(canonV) == canon2Color.end()) {
            canon2Color[canonV] = color;
            ++color;
        }
        color2NodeL[canon2Color[canonV]].push_back(i);
    }
    for (ui i = 0; i < _numNodes; ++i) {
        CanonType canonV = rhs._nodes[i].canonValue;
        if (canon2Color.find(canonV) == canon2Color.end()) return false;
        color2NodeR[canon2Color[canonV]].push_back(i);
    }
    // check whether the label multiset is equivalent
    for (int i = 0; i < color; ++i) {
        if (color2NodeL[i].size() != color2NodeR[i].size()) return false;
    }
    // flatten color2NodeL into a single vector called new2Old. Then the indices of new2Old follow the order of color
    std::vector<ui> new2Old;
    for (int i = 0; i < color; ++i) {
        for (ui j: color2NodeL[i]) {
            new2Old.push_back(j);
        }
    }
    // find all permutation of nodes in the rhs under the label constraint, fix nodes in the lhs
    // this is affordable since a decomposition tree is usually very small
    std::vector<std::vector<std::vector<ui>>> permutation = std::vector<std::vector<std::vector<ui>>> (color);
    for (int i = 0; i < color; ++i) {
        std::vector<ui> vertices = color2NodeR[i];
        std::sort(vertices.begin(), vertices.end());
        do {
            permutation[i].push_back(vertices);
        } while (std::next_permutation(vertices.begin(), vertices.end()));
    }
    // dfs traversal of all permutation
    int *pos = new int[color];
    // mapping[lu] = ru : mapping nodes in the lhs to the rhs
    ui *mapping = new ui[_numNodes];
    pos[0] = 0;
    int d = 0, pm = 0;  // d iterates all color, pm points to the current position in mapping
    while (d >= 0) {
        while (pos[d] < color2NodeL[d].size()) {
            for (int i = 0; i < color2NodeL[d].size(); ++i) {
                VertexID u = permutation[d][pos[d]][i];
                mapping[new2Old[pm]] = u;
                ++pm;
            }
            ++pos[d];
            if (d == color - 1) {
                pm -= (int)color2NodeL[d].size();
                bool flag = true;
                // check nodes equivalent
                for (int i = 0; i < _numNodes; ++i) {
                    if (!(_nodes[i] == rhs._nodes[mapping[i]])) {
                        flag = false;
                        break;
                    }
                }
                // check whether the edge list is the same
                if (flag) {
                    for (int i = 0; i < _numNodes; ++i) {
                        std::vector<VertexID> permutedEdges;
                        for (ui j : _edges[i]) {
                            permutedEdges.push_back(mapping[j]);
                        }
                        std::sort(permutedEdges.begin(), permutedEdges.end());
                        if (!(permutedEdges == rhs._edges[mapping[i]])) {
                            flag = false;
                            break;
                        }
                    }
                }
                // if there exists one mapping, then two trees are isomorphic
                if (flag) {
                    delete[] pos;
                    delete[] mapping;
                    return true;
                }

            }
            else {
                ++d;
                pos[d] = 0;
            }

        }
        // backtrack
        --d;
        if (d < 0) {
            return false;
        }
        pm -= (int)color2NodeL[d].size();
    }

    delete[] pos;
    delete[] mapping;
    return false;
}

Tree::~Tree() {
    delete[] _nodes;
    delete[] _v2n;
    delete[] _parent;
}

std::vector<VertexID> Tree::getUncovered(const PatternGraph &p) const {
    std::vector<VertexID> result;
    VertexID *coreV = p._coreV;
    for (ui i = 0; i < p._coreSize; ++i) {
        VertexID u = coreV[i];
        if (_v2n[u].empty())
            result.push_back(u);
    }

    return result;
}

// called when building decomposition trees
// decide whether all core vertices and edges are covered by tree nodes
// if valid, then build the cut vertices for nodes
// here p must be the undirected pattern
bool Tree::isValid(const PatternGraph &p) {
    // check all core vertices covered
    if (!getUncovered(p).empty()) return false;

    ui numEdge = 0;
    Edge* edgeList = p.coreUndirectedEdges(numEdge);
    std::set<ui> edgeInCore, edgesInTree;
    for (ui i = 0; i < numEdge; ++i) {
        VertexID u1 = edgeList[i].first, u2 = edgeList[i].second;
        // encode an edge into an integer
        edgeInCore.insert(100 * u1 + u2);
    }
    delete[] edgeList;
    for (ui i = 0; i < _numNodes; ++i) {
        Node tau = _nodes[i];
        // loop over vertex pairs in tau
        for (int j = 0; j < tau.numVertices; ++j) {
            VertexID u1 = tau.vertices[j];
            for (int k = j + 1; k < tau.numVertices; ++k) {
                VertexID u2 = tau.vertices[k];
                if (p.isEdge(u1, u2)) {
                    edgesInTree.insert(100 * u1 + u2);
                }
            }
        }
    }
    if (edgeInCore == edgesInTree)
        return true;
    else
        return false;
}

bool Tree::needPrefix(VertexID nID) const {
    return _nodes[nID].prefixSize != 0;
}

bool Tree::hasSubNodeOf(const Node &tau) const {
    for (int i = 0; i < _numNodes; ++i) {
        if (_nodes[i] < tau)
            return true;
    }

    return false;
}

bool Tree::hasSupNodeOf(const Node &tau) const {
    for (int i = 0; i < _numNodes; ++i) {
        if (_nodes[i] > tau)
            return true;
    }

    return false;
}

std::vector<Tree> Tree::addNode(Node &tau) {
    std::vector<Tree> result;
    if (_numNodes == 0) {
        Tree t(*this);
        t._v2n = new std::vector<VertexID>[t._numVertices];
        t._nodes[0] = tau;
        for (int i = 0; i < tau.numVertices; ++i) {
            VertexID u = tau.vertices[i];
            t._v2n[u].push_back(0);
        }
        t._numNodes = 1;
        result.push_back(t);
        return result;
    }
    // for each existing node that shares some vertices with the new node
    // we add an edge and check whether it is a valid TD
    for (ui i = 0; i < _numNodes; ++i) {
        Tree t(*this);
        t._nodes[t._numNodes] = tau;
        // for each vertex u in tau, add tau to _v2n[u]
        for (int j = 0; j < tau.numVertices; ++j) {
            VertexID u = tau.vertices[j];
            t._v2n[u].push_back(t._numNodes);
        }
        ++t._numNodes;
        bool flag;
        for (int j = 0; j < tau.numVertices; ++j) {
            VertexID u = tau.vertices[j];
            if (t._nodes[i].hasVertex(u)) {
                // add (i, _numNodes - 1) to edges
                flag = true;
                t._edges[i].push_back(t._numNodes - 1);
                t._edges[t._numNodes - 1].push_back(i);
                // check whether t satisfies the TD constraint
                // i.e. for any vertex v, tree nodes containing v are connected
                int k;
                for (k = 0; k < tau.numVertices; ++k) {
                    VertexID _u = tau.vertices[k];
                    std::vector<ui> pos = t._v2n[_u];
                    if (pos.size() == 1) continue;
                    // check whether the subgraph of t._edges induced by pos is connected
                    // start a bfs at pos[0]
                    std::vector<bool> visited(_numVertices, true);
                    for (auto nd : pos)
                        visited[nd] = false;

                    std::queue<VertexID> q;
                    q.push(pos[0]);
                    visited[pos[0]] = true;
                    while (!q.empty()) {
                        VertexID nd1 = q.front();
                        q.pop();
                        for (auto nd2: t._edges[nd1]) {
                            if (!visited[nd2]) {
                                visited[nd2] = true;
                                q.push(nd2);
                            }
                        }
                    }
                    // if there are unvisited nodes, then nodes containing _u is unconnected, and t is invalid
                    for (auto f: visited) {
                        if (!f) flag = false;
                    }
                }
                if (flag)
                    result.push_back(t);
                break;
            }
        }
    }

    return result;
}

void Tree::computeSourceVertices(const PatternGraph &pin) {
    for (VertexID nID = 0; nID < _numNodes; ++nID) {
        std::vector<int> degree(_nodes[nID].numVertices, 0);
        for (int i = 0; i < _nodes[nID].numVertices; ++i) {
            for (int j = 0; j < _nodes[nID].numVertices; ++j) {
                if (pin.isEdge(_nodes[nID].vertices[i], _nodes[nID].vertices[j]))
                    ++degree[i];
            }
        }
        for (int i = 0; i < _nodes[nID].numVertices; ++i) {
            if (degree[i] == 0)
                ++_nodes[nID].numSources;
        }
    }
    for (VertexID nID = 0; nID < _numNodes; ++nID) {
        if (_nodes[nID].numSources > _maxNumSource)
            _maxNumSource = _nodes[nID].numSources;
    }
}

// add the peripheral graph to the tree. p must be undirected.
void Tree::addPeripheral(const Pattern &p) {
    ui coreSize = p.u._coreSize;
    VertexID *coreV = p.u._coreV;
    ui peripheralSize = p.u._numVertices - coreSize;
    VertexID *peripheral = p.u._peripheralV;
    // build the forests by bfs. each forest has exactly one edge connects to the coreV.
    std::vector<bool> visited(p.u._numVertices, false);
    std::queue<Edge> q;
    std::map<VertexID, VertexID> nodeID;
    for (int i = 0; i < peripheralSize; ++i) {
        VertexID u = peripheral[i];
        if (visited[u]) continue;
        for (ui j = 0; j < coreSize; ++j)
            visited[coreV[j]] = false;
        q.emplace(99, u);   // use 99 as a virtual vertex
        // each time we pop an edge, we build a node for this edge and add the link(edge) to the node in the tree
        while (!q.empty()) {
            Edge e = q.front();
            q.pop();
            VertexID u1 = e.first, u2 = e.second;
            visited[u2] = true;
            bool flag = false;
            if (u1 != 99) {
                VertexID *vertices = new VertexID[2];
                vertices[0] = u1 < u2 ? u1 : u2;
                vertices[1] = u1 < u2 ? u2 : u1;
                int id = (1 << u1) + (1 << u2);
                _nodes[_numNodes] = Node(id, vertices, 2, 0);
                if (!p.useDAG()) _nodes[_numNodes].getCanonLabel(p.u);
                else _nodes[_numNodes].getCanonLabel(p.out);
                // update _v2n
                _v2n[u1].push_back(_numNodes);
                _v2n[u2].push_back(_numNodes);
                nodeID[u2] = _numNodes;
                if (nodeID.find(u1) != nodeID.end()) {
                    _edges[_numNodes].push_back(nodeID[u1]);
                    _edges[nodeID[u1]].push_back(_numNodes);
                }
                // check whether u2 is in coreV. If so, build an edge between node containing u2 and the last node
                for (int j = 0; j < coreSize; ++j) {
                    if (coreV[j] == u2) {
                        ui k;
                        for (k = 0; k < _numNodes; ++k) {
                            if (_nodes[k].hasVertex(u2))
                                break;
                        }
                        _edges[_numNodes].push_back(k);
                        _edges[k].push_back(_numNodes);
                        flag = true;
                        break;
                    }
                }
                ++_numNodes;
            }
            // add neighbors only when u2 is not coreV
            if (!flag) {
                ui nbrCnt;
                VertexID *neighbors = p.u.getNeighbors(u2, nbrCnt);
                for (int j = 0; j < nbrCnt; ++j) {
                    VertexID next = neighbors[j];
                    if (!visited[next])
                        q.emplace(u2, next);
                }
            }
        }
    }
}

void
Tree::setNodeCanon(std::map<int, CanonType> &canonV, std::map<int, int *> &id2Orbits, std::map<int, ui> &id2AutoSize,
                   ui nID, const Pattern &p) {
    int id = _nodes[nID].id;
    if (canonV.find(id) != canonV.end()) {
        _nodes[nID].canonValue = canonV[id];
        _nodes[nID].autoSize = id2AutoSize[id];
        delete _nodes[nID].v2o;
        _nodes[nID].v2o = new int[MAX_PATTERN_SIZE];
        memcpy(_nodes[nID].v2o, id2Orbits[id], sizeof(int) * MAX_PATTERN_SIZE);
    }
    else {
        if (p.useDAG()) _nodes[nID].getCanonLabel(p.out);
        else _nodes[nID].getCanonLabel(p.u);
        canonV[id] = _nodes[nID].canonValue;
        id2AutoSize[id] = _nodes[nID].autoSize;
        id2Orbits[id] = new int[MAX_PATTERN_SIZE];
        memcpy(id2Orbits[id], _nodes[nID].v2o, sizeof(int) * MAX_PATTERN_SIZE);
    }
}

void Tree::setNodeSubCanon(const Pattern &p) {
    for (VertexID nID : _postOrder) {
        delete[] _nodes[nID].vertexOrbit;
        _nodes[nID].vertexOrbit = new int[MAX_PATTERN_SIZE];
        std::vector<VertexID> vertices;
        // collect vertices in the subtree of nID
        std::queue<VertexID> q;
        q.push(nID);
        while (!q.empty()) {
            VertexID nodeID = q.front();
            q.pop();
            const Node &tau = _nodes[nodeID];
            for (int i = 0; i < tau.numVertices; ++i) {
                VertexID v = tau.vertices[i];
                if (std::find(vertices.begin(), vertices.end(), v) == vertices.end())
                    vertices.push_back(v);
            }
            for (VertexID cID: _child[nodeID])
                q.push(cID);
        }
        if (p.useDAG()) {
            _nodes[nID].subPatternCanon = subgraphCanonValue(p.out, vertices, _nodes[nID].vertexOrbit);
        }
        else {
            _nodes[nID].subPatternCanon = subgraphCanonValue(p.u, vertices, _nodes[nID].vertexOrbit);
        }
    }
}

void Tree::setPrefixKeyOrbit(const Pattern &p) {
    for (VertexID nID : _postOrder) {
        if (_nodes[nID].keySize == 1)
            _nodes[nID].keyOrbit = _nodes[nID].vertexOrbit[_nodes[nID].key[0]] + 10000 * _nodes[nID].v2o[_nodes[nID].key[0]];
        else if (_nodes[nID].keySize == 2) {
            int orbit1 = _nodes[nID].vertexOrbit[_nodes[nID].key[0]];
            int orbit2 = _nodes[nID].vertexOrbit[_nodes[nID].key[1]];
            if (p.useDAG()) {
                _nodes[nID].keyOrbit = orbit1 < orbit2 ? 100 * orbit1 + orbit2 : 100 * orbit2 + orbit1;
                _nodes[nID].keyOrbit += 100;
            }
            else {
                int pos1 = 0, pos2 = 0;
                for (int i = 0; i < _nodes[nID].nodeOrder.size(); ++i) {
                    if (_nodes[nID].nodeOrder[i] == _nodes[nID].key[0]) pos1 = i;
                    if (_nodes[nID].nodeOrder[i] == _nodes[nID].key[1]) pos2 = i;
                }
                _nodes[nID].keyOrbit = 100 * orbit1 + orbit2;
                _nodes[nID].keyOrbit += 100;
            }
            _nodes[nID].keyOrbit += 10000 * (_nodes[nID].v2o[_nodes[nID].key[0]] + _nodes[nID].v2o[_nodes[nID].key[1]]);
        }
        if (_nodes[nID].prefixSize != 0) {
            for (int i = 0; i < _nodes[nID].nodeOrder.size() - _nodes[nID].localOrder.size(); ++i) {
                _nodes[nID].prefixOrbit.push_back(_nodes[nID].vertexOrbit[_nodes[nID].nodeOrder[i]]);
            }
        }
    }
}

// get rooted trees based on aggregation vertices in the pattern
std::vector<Tree> Tree::getRootedTrees(const PatternGraph &p, bool sign) {
    std::vector<Tree> rootedTrees;
    // if aggregation type is 0 (global counting), any node can be root.
    if (p._orbitType == 0) {
        Tree t(*this);
        int minDeg = 99;
        VertexID id = 0;
        for (VertexID nID = 1; nID < _numNodes; ++nID)
            if (_edges[nID].size() < minDeg) {
                minDeg = _edges[nID].size();
                id = nID;
            }
        t.setRoot(id, p);
        if (sign) t._aggreWeight.push_back(1);
        else t._aggreWeight.push_back(-1);
        rootedTrees.push_back(t);
    }
    // select nodes that cover all aggregation vertices. The number of nodes should be minimized,
    // i.e. Take all aggregation vertices as the universe, use node vertices to cover the universe.
    std::vector<VertexID> universe(p._aggreV, p._aggreV + p._aggreSize);
    _orbitType = p._orbitType;
    if (_orbitType == 1) {
        std::vector<int> u2AggPos(p._numVertices);
        for (int i = 0; i < universe.size(); ++i)
            u2AggPos[universe[i]] = i;
        // if we can use a single node to cover all aggregation vertices, choose the node with the smallest degree
        std::map<VertexID, bool> visited;
        int minDeg = 99;
        bool oneNodeCover = false;
        for (VertexID nID = 0; nID < _numNodes; ++nID) {
            for (VertexID u : universe)
                visited[u] = false;
            int count = 0;
            for (int i = 0; i < _nodes[nID].numVertices; ++i) {
                VertexID u = _nodes[nID].vertices[i];
                if (visited.find(u) != visited.end() && !visited[u]) {
                    visited[u] = true;
                    ++count;
                }
            }
            if (count == p._aggreSize && _edges[nID].size() < minDeg) {
                minDeg = _edges[nID].size();
                oneNodeCover = true;
                // now visited denotes whether an aggregation vertex has added to root
                for (VertexID u : universe)
                    visited[u] = false;
                Tree t(*this);
                t.setRoot(nID, p);
                for (int i = 0; i < _nodes[nID].numVertices; ++i) {
                    VertexID u = _nodes[nID].vertices[i];
                    if (visited.find(u) != visited.end() && !visited[u]) {
                        visited[u] = true;
                        t._aggreV.push_back(u);
                        if (sign) t._aggreWeight.push_back(p._aggreWeight[u2AggPos[u]]);
                        else t._aggreWeight.push_back(-p._aggreWeight[u2AggPos[u]]);
                    }
                }
                rootedTrees = {t};
            }
        }
        if (oneNodeCover) return rootedTrees;
        for (int k = 2; k <= _numNodes; ++k) {
            bool allCovered = false;
            // use k nodes to cover the universe
            std::vector<std::vector<bool>> choices = chooseK(_numNodes, k);
            for (const auto &choice : choices) {
                // visited denotes whether an aggregation vertex has been covered
                for (VertexID u : universe)
                    visited[u] = false;
                int count = 0;
                for (VertexID nID = 0; nID < _numNodes; ++nID) {
                    if (choice[nID]) {
                        for (int i = 0; i < _nodes[nID].numVertices; ++i) {
                            VertexID u = _nodes[nID].vertices[i];
                            if (visited.find(u) != visited.end() && !visited[u]) {
                                visited[u] = true;
                                ++count;
                            }
                        }
                    }
                }
                if (count == p._aggreSize) {
                    allCovered = true;
                    // now visited denotes whether an aggregation vertex has added to root
                    for (VertexID u : universe)
                        visited[u] = false;
                    for (VertexID nID = 0; nID < _numNodes; ++nID) {
                        if (choice[nID]) {
                            Tree t(*this);
                            t.setRoot(nID, p);
                            for (int i = 0; i < _nodes[nID].numVertices; ++i) {
                                VertexID u = _nodes[nID].vertices[i];
                                if (visited.find(u) != visited.end() && !visited[u]) {
                                    visited[u] = true;
                                    t._aggreV.push_back(u);
                                    if (sign) t._aggreWeight.push_back(p._aggreWeight[u2AggPos[u]]);
                                    else t._aggreWeight.push_back(-p._aggreWeight[u2AggPos[u]]);
                                }
                            }
                            rootedTrees.push_back(t);
                        }
                    }
                }
                if (allCovered) break;
            }
            if (allCovered) break;
        }
    }
    else if (p._orbitType == 2) {
        std::map<EdgeID, int> e2AggPos;
        for (int i = 0; i < universe.size(); i = i + 2) {
            VertexID u1 = universe[i], u2 = universe[i + 1];
            EdgeID e = 100 * u1 + u2;
            e2AggPos[e] = i / 2;
        }
        // if we can use a single node to cover all aggregation vertices, choose the node with the largest degree
        std::map<EdgeID, bool> visited;
        int minDeg = 99;
        bool oneNodeCover = false;
        for (VertexID nID = 0; nID < _numNodes; ++nID) {
            for (int i = 0; i < universe.size(); i = i + 2) {
                VertexID u1 = universe[i], u2 = universe[i + 1];
                EdgeID e = 100 * u1 + u2;
                visited[e] = false;
            }
            int count = 0;
            for (int i = 0; i < _nodes[nID].numVertices; ++i) {
                for (int j = i + 1; j < _nodes[nID].numVertices; ++j) {
                    VertexID u1 = _nodes[nID].vertices[i], u2 = _nodes[nID].vertices[j];
                    EdgeID e = 100 * u1 + u2;
                    if (visited.find(e) != visited.end() && !visited[e]) {
                        visited[e] = true;
                        ++count;
                    }
                }
            }
            if (count == p._aggreSize / 2 && _edges[nID].size() < minDeg) {
                minDeg = _edges[nID].size();
                oneNodeCover = true;
                // now visited denotes whether an aggregation vertex has added to root
                for (VertexID u : universe)
                    visited[u] = false;
                Tree t(*this);
                t.setRoot(nID, p);
                for (int i = 0; i < _nodes[nID].numVertices; ++i) {
                    for (int j = i + 1; j < _nodes[nID].numVertices; ++j) {
                        VertexID u1 = _nodes[nID].vertices[i], u2 = _nodes[nID].vertices[j];
                        EdgeID e = 100 * u1 + u2;
                        if (visited.find(e) != visited.end() && !visited[e]) {
                            visited[e] = true;
                            t._aggreV.push_back(u1);
                            t._aggreV.push_back(u2);
                            if (sign) t._aggreWeight.push_back(p._aggreWeight[e2AggPos[e]]);
                            else t._aggreWeight.push_back(-p._aggreWeight[e2AggPos[e]]);
                        }
                    }
                }
                rootedTrees = {t};
            }
        }
        if (oneNodeCover) return rootedTrees;
        for (int k = 2; k <= _numNodes; ++k) {
            bool allCovered = false;
            // use k nodes to cover the universe
            std::vector<std::vector<bool>> choices = chooseK(_numNodes, k);
            for (const auto &choice : choices) {
                // visited denotes whether an aggregation edge has been covered
                for (int i = 0; i < universe.size(); i = i + 2) {
                    VertexID u1 = universe[i], u2 = universe[i + 1];
                    EdgeID e = 100 * u1 + u2;
                    visited[e] = false;
                }
                int count = 0;
                for (VertexID nID = 0; nID < _numNodes; ++nID) {
                    if (choice[nID]) {
                        for (int i = 0; i < _nodes[nID].numVertices; ++i) {
                            for (int j = i + 1; j < _nodes[nID].numVertices; ++j) {
                                VertexID u1 = _nodes[nID].vertices[i], u2 = _nodes[nID].vertices[j];
                                EdgeID e = 100 * u1 + u2;
                                if (visited.find(e) != visited.end() && !visited[e]) {
                                    visited[e] = true;
                                    ++count;
                                }
                            }
                        }
                    }
                }
                if (count == p._aggreSize / 2) {
                    allCovered = true;
                    // now visited denotes whether an aggregation vertex has added to root
                    for (auto & it : visited)
                        it.second = false;
                    for (VertexID nID = 0; nID < _numNodes; ++nID) {
                        if (choice[nID]) {
                            Tree t(*this);
                            t.setRoot(nID, p);
                            for (int i = 0; i < _nodes[nID].numVertices; ++i) {
                                for (int j = i + 1; j < _nodes[nID].numVertices; ++j) {
                                    VertexID u1 = _nodes[nID].vertices[i], u2 = _nodes[nID].vertices[j];
                                    EdgeID e = 100 * u1 + u2;
                                    if (visited.find(e) != visited.end() && !visited[e]) {
                                        visited[e] = true;
                                        t._aggreV.push_back(u1);
                                        t._aggreV.push_back(u2);
                                        if (sign) t._aggreWeight.push_back(p._aggreWeight[e2AggPos[e]]);
                                        else t._aggreWeight.push_back(-p._aggreWeight[e2AggPos[e]]);
                                    }
                                }
                            }
                            rootedTrees.push_back(t);
                        }
                    }
                }
                if (allCovered) break;
            }
            if (allCovered) break;
        }
    }
    return rootedTrees;
}

void Tree::rebuildCut() {
    for (VertexID i = 0; i < _numNodes; ++i) {
        Node &tau = _nodes[i];
        if (tau.cutSize == 0) {
            delete[] tau.cut;
            tau.cut = new VertexID[tau.numVertices];
            std::vector<bool> visited(_numVertices, false);
            for (VertexID j: _edges[i]) {
                VertexID *iVertices = tau.vertices, *jVertices = _nodes[j].vertices;
                int pi = 0, pj = 0;
                while (pi < _nodes[i].numVertices && pj < _nodes[j].numVertices) {
                    VertexID ui = iVertices[pi], uj = jVertices[pj];
                    if (ui == uj) {
                        if (!visited[ui]) {
                            visited[ui] = true;
                            _nodes[i].cut[_nodes[i].cutSize] = ui;
                            ++_nodes[i].cutSize;
                        }
                        ++pi;
                        ++pj;
                    }
                    else if (ui < uj) ++pi;
                    else ++pj;
                }
            }
        }
    }
}

void Tree::setRoot(VertexID root, const PatternGraph &p) {
    delete[] _parent;
    _orbitType = p._orbitType;
    _child.clear();
    _parent = new VertexID[_numNodes];
    _child = std::vector<std::vector<VertexID>> (_numNodes);
    _aggreV.clear();
    _aggreWeight.clear();
    _postOrder.clear();
    _partitionPos.clear();
    _globalOrder.clear();
    std::queue<Edge> q;
    q.emplace(99, root);
    // set _parent and _child
    while (!q.empty()) {
        VertexID parent = q.front().first, current = q.front().second;
        q.pop();
        if (parent != 99) _child[parent].push_back(current);
        _parent[current] = parent;
        for (int i = 0; i < _edges[current].size(); ++i) {
            VertexID next = _edges[current][i];
            if (parent != next) {
                q.emplace(current, next);
            }
        }
    }
    for (VertexID nID = 0; nID < _numNodes; ++nID) {
        // sort children according to id
        std::sort(_child[nID].begin(), _child[nID].end(), [this](const VertexID &lhs, const VertexID &rhs) {
            return (_nodes[lhs].id < _nodes[rhs].id);
        });
    }
    // reset cut as the vertices shared with the parent node
    std::vector<bool> needPrefix(_numNodes, true);
    for (VertexID nID = 0; nID < _numNodes; ++nID) {
        ui j1 = 0, j2 = 0;
        delete[] _nodes[nID].cut;
        _nodes[nID].cut = new VertexID[_nodes[nID].numVertices];
        _nodes[nID].cutSize = 0;
        delete[] _nodes[nID].key;
        _nodes[nID].key = new VertexID[_nodes[nID].numVertices];
        _nodes[nID].keySize = 0;
        VertexID parent = _parent[nID];
        if (parent == 99) continue;
        while (j1 < _nodes[nID].numVertices && j2 < _nodes[parent].numVertices) {
            VertexID u1 = _nodes[nID].vertices[j1], u2 = _nodes[parent].vertices[j2];
            if (u1 == u2) {
                _nodes[nID].cut[_nodes[nID].cutSize] = u1;
                ++_nodes[nID].cutSize;
                ++j1;
                ++j2;
            }
            else if (u1 < u2) ++j1;
            else ++j2;
        }
        if (_nodes[nID].cutSize == 1) {
            _nodes[nID].prefixSize = 0;
            _nodes[nID].keySize = 1;
            _nodes[nID].key[0] = _nodes[nID].cut[0];
            needPrefix[nID] = false;
        }
        else if (_nodes[nID].cutSize == 2 && (p.isEdge(_nodes[nID].cut[0], _nodes[nID].cut[1]) ||
                                              p.isEdge(_nodes[nID].cut[1], _nodes[nID].cut[0]))) {
            _nodes[nID].prefixSize = 0;
            _nodes[nID].keySize = 2;
            _nodes[nID].key[0] = _nodes[nID].cut[0];
            _nodes[nID].key[1] = _nodes[nID].cut[1];
            needPrefix[nID] = false;
        }
    }
    needPrefix[root] = false;
    computePostOrder(root, needPrefix);
    for (int i = 0; i < _postOrder.size(); ++i) {
        if (!needPrefix[_postOrder[i]])
            _partitionPos.push_back(i);
    }
}

void Tree::computePostOrder(VertexID rootID, const std::vector<bool> &needPrefix) {
    // make a simple tree with only simple nodes and a post-order traversal of this simple tree
    std::vector<std::vector<VertexID>> simpleChild(_numNodes);
    std::vector<VertexID> simplePostOrder;
    std::vector<int> poses(_numNodes, 0);
    // vs1: current node, vs2: parent
    std::stack<VertexID> vs1, vs2;
    vs1.push(rootID);
    vs2.push(99);
    while (!vs1.empty()) {
        VertexID nID = vs1.top();
        VertexID parentID = vs2.top();
        if (poses[nID] == _child[nID].size()) {
            if (!needPrefix[nID]) {
                simplePostOrder.push_back(nID);
                if (parentID != 99) simpleChild[parentID].push_back(nID);
            }
            vs1.pop();
            vs2.pop();
        }
        else {
            if (!needPrefix[nID]) parentID = nID;
            vs1.push(_child[nID][poses[nID]]);
            vs2.push(parentID);
            ++poses[nID];
        }
    }
    // make a post-order traversal of the whole tree based on this simple tree
    std::vector<std::vector<VertexID>> subsequences(_numNodes);
    poses = std::vector<int>(_numNodes, 0);
    for (VertexID simpleID: simplePostOrder) {
        for (VertexID cID: simpleChild[simpleID]) {
            for (VertexID u: subsequences[cID]) subsequences[simpleID].push_back(u);
        }
        vs1.push(simpleID);
        while (!vs1.empty()) {
            VertexID nID = vs1.top();
            if (poses[nID] == _child[nID].size()) {
                subsequences[simpleID].push_back(nID);
                vs1.pop();
            }
            else {
                VertexID cID = _child[nID][poses[nID]];
                if (needPrefix[cID]) vs1.push(cID);
                ++poses[nID];
            }
        }
    }
    _postOrder = subsequences[rootID];
}

bool Tree::isLeaf(VertexID i) const {
    return _child[i].empty();
}

bool Tree::twoWidthEqual() const {
    ui maxVertices = 0;
    for (int i = 0; i < _numNodes; ++i) {
        if (_nodes[i].numVertices > maxVertices)
            maxVertices = _nodes[i].numVertices;
    }
//    assert(maxVertices <= _treeWidth);
    if (maxVertices == _treeWidth) return true;
    return false;
}

bool Tree::inTheSameNode(VertexID u1, VertexID u2) const {
    for (int i = 0; i < _numNodes; ++i) {
        if (_nodes[i].hasVertex(u1) && _nodes[i].hasVertex(u2))
            return true;
    }

    return false;
}

VertexID *Tree::getCut(VertexID nID, ui &cnt) const {
    cnt = _nodes[nID].cutSize;
    return _nodes[nID].cut;
}

const std::vector<VertexID> & Tree::getNodesContainV(VertexID u) const {
    return _v2n[u];
}

std::vector<VertexID> Tree::getNodesContainSymV(const std::vector<VertexID> &vartheta) const {
    std::vector<VertexID> result;
    int numNodesContainU = 0;
    int groupID = 0;
    for (const VertexID &w : vartheta)
        groupID += 1 << w;

    for (VertexID nID = 0; nID < _numNodes; ++nID) {
        const Node &tau = _nodes[nID];
        if (std::find(tau.vertices, tau.vertices + tau.numVertices, vartheta[0]) != tau.vertices + tau.numVertices) {
            if (tau.validRules.find(groupID) == tau.validRules.end()) {
                result.clear();
                return result;
                break;
            }
            else result.push_back(nID);
        }
    }
    for (int i = 1; i < vartheta.size(); ++i) {
        int pos = 0;
        for (VertexID nID = 0; nID < _numNodes; ++nID) {
            const Node &tau = _nodes[nID];
            if (std::find(tau.vertices, tau.vertices + tau.numVertices, vartheta[i]) != tau.vertices + tau.numVertices) {
                if (pos == result.size() || result[pos] != nID) {
                    result.clear();
                    return result;
                }
                else ++pos;
            }
        }
    }
    return result;
}

void Tree::setSymmetryRoot(const std::vector<std::vector<VertexID>> &rules) {
    if (_orbitType == 2) {
        // edge orbit
    }
    else {
        // check whether automorphisms fix 0
        const Node &tau = _nodes[_postOrder.back()];
        std::vector<std::vector<VertexID>> notFixZeroRules;
        ui reducedAutoSize1 = 1;
        for (int i = 0; i < rules.size(); ++i) {
            ui pos1 = 0;
            while (tau.vertices[pos1] != rules[i][0]) ++pos1;
            bool fixedZero = false;
            for (int j = 1; j < tau.autoSize; ++j) {
                if (!(tau.automorphisms[j][0] == 0)) continue;
                // whether the automorphism contains a cycle of rule
                int cycleLength = 1;
                ui pos = tau.automorphisms[j][pos1];
                VertexID u = tau.vertices[pos];
                if (std::find(rules[i].begin(), rules[i].end(), u) == rules[i].end()) {
                    cycleLength = 0;
                    u = rules[i][0];
                }
                while (u != rules[i][0]) {
                    ++cycleLength;
                    pos = tau.automorphisms[j][pos];
                    u = tau.vertices[pos];
                    if (std::find(rules[i].begin(), rules[i].end(), u) == rules[i].end()) {
                        cycleLength = 0;
                        u = rules[i][0];
                    }
                }
                if (cycleLength == rules[i].size()) {
                    fixedZero = true;
                    break;
                }
            }
            if (fixedZero) reducedAutoSize1 *= rules[i].size();
            else notFixZeroRules.push_back(rules[i]);
        }
        _multiFactor *= reducedAutoSize1;
        if (notFixZeroRules.empty()) return;
        if (notFixZeroRules.size() == 1 && notFixZeroRules[0][0] == 0) {
            _aggreV = notFixZeroRules[0];
            for (int i = 1; i < _aggreV.size(); ++i) {
                _aggreWeight.push_back(_aggreWeight[0]);
            }
            return;
        }
        int *consecutiveId = new int[MAX_PATTERN_SIZE];
        std::vector<VertexID> zeroOrbit;
        for (int i = 0; i < tau.numVertices; ++i) {
            if (tau.v2o[tau.vertices[i]] == tau.v2o[0])
                zeroOrbit.push_back(tau.vertices[i]);
            consecutiveId[tau.vertices[i]] = i;
        }
        ui reducedAutoSize2 = 1;
        for (int i = 0; i < notFixZeroRules.size(); ++i) {
            reducedAutoSize2 *= notFixZeroRules[i].size();
        }
        // collect automorphisms that do not conflict the rules
        std::vector<int> js;
        for (int j = 0; j < tau.autoSize; ++j) {
            bool valid = true;
            for (int i = 0; i < notFixZeroRules.size(); ++i) {
                int u0 = consecutiveId[notFixZeroRules[i][0]];
                for (int k = 1; k < notFixZeroRules[i].size(); ++k) {
                    int uk = consecutiveId[notFixZeroRules[i][k]];
                    if (tau.automorphisms[j][u0] > tau.automorphisms[j][uk]) {
                        valid = false;
                        break;
                    }
                }
            }
            if (valid) js.push_back(j);
        }
        int multiFactor = 1;
        if (reducedAutoSize2 > zeroOrbit.size()) multiFactor = reducedAutoSize2 / zeroOrbit.size();
        // from these automorphisms, find columns that reconstruct zero orbit
        std::vector<std::vector<bool>> candidates = chooseK(zeroOrbit.size(), (int)reducedAutoSize2);
        for (auto &cand: candidates) {
            if (!cand[0]) continue;
            std::vector<int> orbitCount(MAX_PATTERN_SIZE, 0);
            for (int i = 0; i < cand.size(); ++i) {
                if (cand[i]) {
                    VertexID u = zeroOrbit[i];
                    for (int j : js) ++orbitCount[tau.vertices[tau.automorphisms[j][consecutiveId[u]]]];
                }
            }
            bool valid = true;
            for (int i = 0; i < zeroOrbit.size(); ++i) {
                if (orbitCount[zeroOrbit[i]] != tau.autoSize / zeroOrbit.size() / multiFactor) {
                    valid = false;
                    break;
                }
            }
            if (valid) {
                for (int i = 1; i < cand.size(); ++i) {
                    if (cand[i]) {
                        VertexID u = zeroOrbit[i];
                        _aggreV.push_back(u);
                        _aggreWeight.push_back(_aggreWeight[0]);
                    }
                }
                _multiFactor *= multiFactor;
                delete[] consecutiveId;
                return;
            }
        }
    }
}

void Tree::setSymmetryRules(const std::vector<std::vector<VertexID>> &allRule,
                            const std::vector<std::vector<VertexID>> &allNID) {
    VertexID rootID = getRootID();
    std::vector<std::vector<VertexID>> rootRules;
    for (int i = 0; i < allRule.size(); ++i) {
        if (allRule[i].empty()) continue;
        for (int j = 1; j < allRule[i].size(); ++j) {
            _lessRules[allRule[i][0]].push_back(allRule[i][j]);
            _greaterRules[allRule[i][j]].push_back(allRule[i][0]);
        }
        bool rootRule = false;
        if (_orbitType != 0) {
            for (VertexID nID : allNID[i]) {
                ++_nodes[nID].numRules;
                if (nID == rootID) {
                    rootRule = true;
                    rootRules.push_back(allRule[i]);
                    break;
                }
            }
        }
        if (!rootRule) _multiFactor *= allRule[i].size();
    }
    setSymmetryRoot(rootRules);
    for (VertexID nID : _postOrder) {
        if (_parent[nID] != 99)
            _nodes[_parent[nID]].subPatternRules += _nodes[nID].numRules;
    }
    _numRules = 0;
    for (int i = 0; i < allRule.size(); ++i) {
        if (!allRule[i].empty())
            ++_numRules;
    }
}

int Tree::computeSymmetryRules(const PatternGraph &p) {
    std::vector<VertexID> nodesWithRule;
    for (VertexID nID = 0; nID < _numNodes; ++nID) {
        std::vector<VertexID *> childCuts;
        std::vector<ui> cutSizes;
        for (VertexID cID: _child[nID]) {
            childCuts.push_back(_nodes[cID].cut);
            cutSizes.push_back(_nodes[cID].cutSize);
        }
        _nodes[nID].computeCandidateRules(p, childCuts, cutSizes);
        if (!_nodes[nID].candidateRules.empty()) nodesWithRule.push_back(nID);
    }
    if (nodesWithRule.empty()) return 0;
    int maxRuleNum = 0;
    std::vector<std::vector<VertexID>> bestRule, bestNID;
    std::vector<std::vector<bool>> ruleValid(_numNodes);
    for (int i = 0; i < _numNodes; ++i) {
        ruleValid[i] = std::vector<bool>(_nodes[i].candidateRules.size(), true);
    }
    // each node should either have all nodes the rule, or does not contain any node in the rule
    std::set<int> invalidIDs;
    for (int i = 0; i < _numNodes; ++i) {
        for (const auto &ruleSet: _nodes[i].candidateRules) {
            for (const auto &rule: ruleSet) {
                int ruleID = 0;
                for (const VertexID &w: rule)
                    ruleID += 1 << w;
                if (invalidIDs.find(ruleID) != invalidIDs.end()) continue;
                if (!containOrDisjoint(rule)) invalidIDs.insert(ruleID);
            }
        }
    }
    int depth = 0;
    int *pos = new int[_numNodes];
    memset(pos, 0, sizeof(int) * _numNodes);
    while (depth >= 0) {
        while (pos[depth] < _nodes[nodesWithRule[depth]].candidateRules.size()) {
            ++pos[depth];
            if (depth == nodesWithRule.size() - 1) {
                std::vector<std::vector<VertexID>> allRule, allNID;
                int ruleNum = 0;
                std::set<int> selectedIDs;
                std::set<int> notSelectedIDs;
                // for each selected rule set (_nodes[i].candidateRules[pos[i] - 1]), check whether each rule is valid
                for (int i = 0; i < nodesWithRule.size(); ++i) {
                    for (const auto &rule: _nodes[nodesWithRule[i]].candidateRules[pos[i] - 1]) {
                        int ruleID = 0;
                        for (const VertexID &w: rule)
                            ruleID += 1 << w;
                        if (invalidIDs.find(ruleID) != invalidIDs.end() || selectedIDs.find(ruleID) != selectedIDs.end()
                            || notSelectedIDs.find(ruleID) != notSelectedIDs.end())
                            continue;
                        // collect nodes contain rule, check whether rule is consistent in these nodes and compute rule num
                        bool consistent = true;
                        for (VertexID nID = 0; nID < _numNodes; ++nID) {
                            bool contain = false;
                            if (_nodes[nID].candidateRules.empty()) {
                                for (int j = 0; j < _nodes[nID].numVertices; ++j) {
                                    if (_nodes[nID].vertices[j] == rule[0]) {
                                        contain = true;
                                        break;
                                    }
                                }
                            }
                            if (contain) {
                                consistent = false;
                                break;
                            }
                        }
                        if (!consistent) {
                            notSelectedIDs.insert(ruleID);
                            continue;
                        }
                        std::vector<VertexID> nIDs;
                        nIDs.push_back(nodesWithRule[i]);
                        for (int j = 0; j < nodesWithRule.size(); ++j) {
                            if (j == i) continue;
                            const Node &tau = _nodes[nodesWithRule[j]];
                            bool contain = false;
                            for (int k = 0; k < tau.numVertices; ++k) {
                                if (tau.vertices[k] == rule[0]) {
                                    contain = true;
                                    break;
                                }
                            }
                            if (contain) {
                                nIDs.push_back(nodesWithRule[j]);
                                const std::vector<std::vector<VertexID>> &rules = tau.candidateRules[pos[j] - 1];
                                bool hasThisRule = false;
                                for (const auto & _rule : tau.candidateRules[pos[j] - 1]) {
                                    if (rule == _rule) {
                                        hasThisRule = true;
                                        break;
                                    }
                                }
                                if (!hasThisRule) {
                                    consistent = false;
                                    break;
                                }
                            }
                        }
                        if (!consistent) {
                            notSelectedIDs.insert(ruleID);
                            continue;
                        }
                        if (rule.size() > 4 && nIDs.size() > 1) consistent = autoConsistent(nIDs, rule);
//                        consistent = autoConsistent(nIDs, rule);
                        if (consistent) {
                            selectedIDs.insert(ruleID);
                            allRule.push_back(rule);
                            allNID.push_back(nIDs);
                            for (VertexID nID: nIDs) {
                                ruleNum += rule.size() * _nodes[nID].numVertices;
                            }
                        }
                    }
                }
                if (ruleNum > maxRuleNum) {
                    maxRuleNum = ruleNum;
                    bestRule = allRule;
                    bestNID = allNID;
                }
            }
            else {
                ++depth;
                pos[depth] = 0;
            }
        }
        --depth;
    }
    setSymmetryRules(bestRule, bestNID);
    delete[] pos;
    return bestRule.size();
}

bool Tree::containOrDisjoint(const std::vector<VertexID> &rule) {
    for (VertexID nID = 0; nID < _numNodes; ++nID) {
        bool contain = true, disjoint = true;
        for (VertexID u: rule) {
            int i = 0;
            while (i < _nodes[nID].numVertices) {
                if (_nodes[nID].vertices[i] == u) {
                    disjoint = false;
                    break;
                }
                else ++i;
            }
            if (i == _nodes[nID].numVertices) contain = false;
        }
        if (!(contain || disjoint)) return false;
    }

    return true;
}

bool Tree::autoConsistent(const std::vector<VertexID> &nIDs, const std::vector<VertexID> &rule) {
    // generate the cycles made by rule in the first node
    int numCycle = 0;
    const Node &firstNode = _nodes[nIDs[0]];
    VertexID **cycles = new VertexID *[firstNode.autoSize];
    for (int i = 0; i < firstNode.autoSize; ++i) {
        cycles[i] = new VertexID [rule.size()];
    }
    ui pos1 = 0;
    while (firstNode.vertices[pos1] != rule[0]) ++pos1;
    for (int i = 0; i < firstNode.autoSize; ++i) {
        // whether the automorphism contains a cycle of rule
        cycles[numCycle][0] = rule[0];
        int cycleLength = 0;
        ui pos = firstNode.automorphisms[i][pos1];
        VertexID u = firstNode.vertices[pos];
        cycles[numCycle][cycleLength] = u;
        if (std::find(rule.begin(), rule.end(), u) == rule.end()) {
            cycleLength = 0;
            u = rule[0];
        }
        while (u != rule[0]) {
            ++cycleLength;
            pos = firstNode.automorphisms[i][pos];
            u = firstNode.vertices[pos];
            cycles[numCycle][cycleLength] = u;
            if (std::find(rule.begin(), rule.end(), u) == rule.end()) {
                cycleLength = 0;
                u = rule[0];
            }
        }
        if (cycleLength + 1 == rule.size()) {
            ++numCycle;
        }
    }

    // check whether automorphisms in other nodes have the same cycle
    bool flag = true;
    for (int i = 0; i < numCycle; ++i) {
        flag = true;
        for (int j = 1; j < nIDs.size(); ++j) {
            const Node &tau = _nodes[nIDs[j]];
            int factorial = 1;
            for (int k = 2; k <= tau.numVertices; ++k) {
                factorial *= k;
            }
            if (tau.autoSize == factorial) continue;
            bool hasThisCycle = false;
            for (int k = 0; k < tau.autoSize; ++k) {
                ui pos = 0;
                while (tau.vertices[pos] != rule[0]) ++pos;
                int cycleLength = 0;
                while (cycleLength + 1 < rule.size()) {
                    pos = tau.automorphisms[k][pos];
                    VertexID u = tau.vertices[pos];
                    if (u != cycles[i][cycleLength]) break;
                    else ++cycleLength;
                }
                if (cycleLength + 1 == rule.size()) {
                    hasThisCycle = true;
                    break;
                }
            }
            if (!hasThisCycle) {
                flag = false;
                break;
            }
        }
        if (flag) return true;
    }

    for (int i = 0; i < firstNode.autoSize; ++i) delete[] cycles[i];
    delete[] cycles;
//    std::cout << "autoConsistent returns false" << std::endl;
    return false;
}

Edge *Tree::getRootedSubgraph(VertexID nID, const PatternGraph &p, std::vector<VertexID> &old2New, ui &numVertices, ui &numEdges) const {
    ui n = p.getNumVertices();
    old2New = std::vector<VertexID>(n, 0);
    Edge *edgeList = new Edge[p.getNumEdges()];
    std::set<VertexID> verticesInSubtree;
    std::queue<VertexID> q;
    q.push(nID);
    while (!q.empty()) {
        VertexID nodeID = q.front();
        for (int i = 0; i < _nodes[nodeID].numVertices; ++i) {
            verticesInSubtree.insert(_nodes[nodeID].vertices[i]);
        }
        q.pop();
        for (auto nc: _child[nodeID])
            q.push(nc);
    }
    std::vector<VertexID> new2Old(verticesInSubtree.begin(), verticesInSubtree.end());
    int newSize = (int)new2Old.size();
    numVertices = newSize;
    for (int i = 0; i < newSize; ++i)
        old2New[new2Old[i]] = i;
    numEdges = 0;
    for (VertexID i = 0; i < newSize; ++i) {
        VertexID oldI = new2Old[i];
        for (VertexID j = 0; j < newSize; ++j) {
            VertexID oldJ = new2Old[j];
            if (p.isEdge(oldI, oldJ)) {
                edgeList[numEdges] = std::make_pair(i, j);
                ++numEdges;
            }
        }
    }

    return edgeList;
}

// merge the tree such that there are only two levels (root and root's children)
Tree Tree::mergeToTwoLevel() const {
    VertexID rootID = getRootID();
    std::vector<Node> nodes;
    for (VertexID nID : _child[rootID]) {
        // build a new node for the subgraph rooted at nID and push it to nodes
        std::queue<VertexID> q;
        std::set<VertexID> verticesInSubtree;
        q.push(nID);
        while (!q.empty()) {
            VertexID nodeID = q.front();
            for (int i = 0; i < _nodes[nodeID].numVertices; ++i) {
                verticesInSubtree.insert(_nodes[nodeID].vertices[i]);
            }
            q.pop();
            for (auto nc: _child[nodeID])
                q.push(nc);
        }
        int id = 0;
        int num = 0;
        VertexID *vertices = new VertexID[verticesInSubtree.size()];
        for (auto it = verticesInSubtree.begin(); it != verticesInSubtree.end(); ++it) {
            VertexID u = *it;
            vertices[num] = u;
            ++num;
            id += 1 << u;
        }
        Node nd(id, vertices, num, 0);
        nd.cut = new VertexID[num];
        // build the cut for nd
        VertexID *rootVertices = _nodes[rootID].vertices;
        int p1 = 0, p2 = 0;
        while (p1 < _nodes[rootID].numVertices && p2 < num) {
            VertexID u1 = rootVertices[p1], u2 = vertices[p2];
            if (u1 == u2) {
                nd.cut[nd.cutSize] = u1;
                ++nd.cutSize;
                ++p1;
                ++p2;
            }
            else if (u1 < u2) ++p1;
            else ++p2;
        }
        nodes.push_back(nd);
    }

    return Tree(_nodes[rootID], nodes, _numVertices);
}

void Tree::setPrefix(VertexID nID, std::vector<VertexID> prefix, const Pattern &p) {
    Node &tau = _nodes[nID];
    if (prefix.empty()) {
        tau.prefixSize = 0;
        return;
    }
    delete[] tau.prefix;
    tau.prefixSize = (ui)prefix.size();
    tau.prefix = new VertexID[tau.prefixSize];
    for (int j = 0; j < tau.prefixSize; ++j) {
        tau.prefix[j] = prefix[j];
    }
    // set key according to prefix and cut
    std::sort(prefix.begin(), prefix.end());
    delete[] tau.key;
    tau.key = new VertexID[2];
    tau.keySize = 0;
    int p1 = 0, p2 = 0;
    while (p1 < tau.prefixSize && p2 < tau.cutSize) {
        if (prefix[p1] == tau.cut[p2]) {
            ++p1;
            ++p2;
        }
        else {
            tau.key[tau.keySize] = tau.cut[p2];
            ++tau.keySize;
            ++p2;
        }
    }
    while (p2 < tau.cutSize) {
        tau.key[tau.keySize] = tau.cut[p2];
        ++tau.keySize;
        ++p2;
    }
}

Tree Tree::relabel(VertexID *mapping) const {
    Tree t(*this);
    // relabel vertices in _v2n
    for (ui i = 0; i < t._numVertices; ++i)
        t._v2n[i].clear();
    // relabel vertices in nodes according to the mapping
    for (VertexID nID = 0; nID < t._numNodes; ++nID) {
        int newNodeID = 0;
        for (ui j = 0; j < t._nodes[nID].numVertices; ++j) {
            VertexID newVertexID = mapping[t._nodes[nID].vertices[j]];
            newNodeID += 1 << newVertexID;
            t._nodes[nID].vertices[j] = newVertexID;
            t._v2n[newVertexID].push_back(nID);
        }
        std::sort(t._nodes[nID].vertices, t._nodes[nID].vertices + t._nodes[nID].numVertices);
        for (ui j = 0; j < t._nodes[nID].cutSize; ++j)
            t._nodes[nID].cut[j] = mapping[t._nodes[nID].cut[j]];
        for (ui j = 0; j < t._nodes[nID].prefixSize; ++j)
            t._nodes[nID].prefix[j] = mapping[t._nodes[nID].prefix[j]];
        for (ui j = 0; j < t._nodes[nID].keySize; ++j)
            t._nodes[nID].key[j] = mapping[t._nodes[nID].key[j]];
        for (ui j = 0; j < t._nodes[nID].localOrder.size(); ++j)
            t._nodes[nID].localOrder[j] = mapping[t._nodes[nID].localOrder[j]];
        t._nodes[nID].id = newNodeID;
        int *oldV2O = new int[MAX_PATTERN_SIZE];
        memcpy(oldV2O, t._nodes[nID].v2o, MAX_PATTERN_SIZE * sizeof(int));
        delete t._nodes[nID].v2o;
        t._nodes[nID].v2o = new int[MAX_PATTERN_SIZE];
        for (int j = 0; j < t._nodes[nID].numVertices; ++j)
            t._nodes[nID].v2o[mapping[t._nodes[nID].vertices[j]]] = oldV2O[t._nodes[nID].vertices[j]];
        delete[] oldV2O;
    }

    // relabel vertices in _aggreV
    for (int i = 0; i < t._aggreV.size(); ++i) {
        t._aggreV[i] = mapping[t._aggreV[i]];
    }
    return t;
}

void Tree::adjustWeight(int pos, bool sign, VertexID u, VertexID v, int weight) {
    if (pos >= _aggreWeight.size()) {
        VertexID root = _postOrder.back();
        const std::vector<VertexID> &partitionOrder = _globalOrder.back();
        if (_orbitType == 1) {
            _aggreV.push_back(u);
            // change aggrePos of the root accordingly
            for (int i = 0; i < partitionOrder.size() + _nodes[root].localOrder.size(); ++i) {
                if (i < partitionOrder.size()) {
                    if (partitionOrder[i] == u) {
                        _aggrePos[root].push_back(i);
                        break;
                    }
                }
                else {
                    if (_nodes[root].localOrder[i - partitionOrder.size()] == u) {
                        _aggrePos[root].push_back(i);
                        break;
                    }
                }
            }
        }
        else {
            _aggreV.push_back(u);
            _aggreV.push_back(v);
            // change aggrePos of the root accordingly
            for (int i = 0; i < partitionOrder.size() + _nodes[root].localOrder.size(); ++i) {
                if (i < partitionOrder.size()) {
                    if (partitionOrder[i] == u || partitionOrder[i] == v)
                        _aggrePos[root].push_back(i);
                }
                else {
                    if (_nodes[root].localOrder[i - partitionOrder.size()] == u ||
                        _nodes[root].localOrder[i - partitionOrder.size()] == v)
                        _aggrePos[root].push_back(i);
                }
            }
        }
        if (sign) _aggreWeight.push_back(weight);
        else _aggreWeight.push_back(-weight);
    }
    else {
        if (sign) _aggreWeight[pos] += weight;
        else _aggreWeight[pos] -= weight;
    }
}

void Tree::adjustWeight() {
    int w = _aggreWeight[0];
    if (w == 0) {
        for (int &weight: _aggreWeight)
            weight = 1;
        _multiFactor = 0;
        return;
    }
    _multiFactor *= w;
    for (int &weight: _aggreWeight)
        weight /= w;
}

void Tree::adjustMultiFactor(int value) {
    _multiFactor += value;
}

bool Tree::posInAggreV(VertexID u, int &pos) {
    for (int i = 0; i < _aggreV.size(); ++i) {
        if (_aggreV[i] == u) {
            pos = i;
            return true;
        }
    }

    return false;
}

bool Tree::posInAggreV(VertexID u, VertexID v, int &pos) {
    for (int i = 0; i < _aggreV.size(); i = i + 2) {
        if (_aggreV[i] == u && _aggreV[i + 1] == v) {
            pos = i / 2;
            return true;
        }
    }

    return false;
}

void Tree::setAggreInfo(int orbitType, VertexID u, VertexID v, bool sign, int weight, const Pattern &p) {
    if (orbitType == 1)
        _aggreV.push_back(u);
    else {
        _aggreV.push_back(u);
        _aggreV.push_back(v);
    }
    if (sign) _aggreWeight.push_back(weight);
    else _aggreWeight.push_back(-weight);
}

void Tree::initPoses(const Pattern &p, bool useTriangle) {
    int numPartition = (int)_partitionPos.size();
    // at partition i, when the partitionOrder[length] is matched, nodes in _nodesAtStep[i][length] can be executed,
    _nodesAtStep = std::vector<std::vector<std::vector<VertexID>>> (numPartition);
    _aggrePos = std::vector<std::vector<int>>(_numNodes);
    _partitionInterPos = std::vector<std::vector<bool>>(numPartition);
    _nodeInterPos = std::vector<std::vector<bool>>(_numNodes);
    _nodeInPos = std::vector<std::vector<std::vector<int>>>(_numNodes);
    _nodeOutPos = std::vector<std::vector<std::vector<int>>>(_numNodes);
    _nodeUnPos = std::vector<std::vector<std::vector<int>>>(_numNodes);
    _nodeGreaterPos = std::vector<std::vector<std::vector<int>>>(_numNodes);
    _nodeLessPos = std::vector<std::vector<std::vector<int>>>(_numNodes);
    _partitionInPos = std::vector<std::vector<std::vector<int>>>(numPartition);
    _partitionOutPos = std::vector<std::vector<std::vector<int>>>(numPartition);
    _partitionUnPos = std::vector<std::vector<std::vector<int>>>(numPartition);
    _partitionGreaterPos = std::vector<std::vector<std::vector<int>>>(numPartition);
    _partitionLessPos = std::vector<std::vector<std::vector<int>>>(numPartition);
    _childKeyPos = std::vector<std::vector<std::vector<int>>>(_numNodes);
    _posChildEdge = std::vector<std::vector<std::vector<int>>>(_numNodes);
    _posAggreEdge = std::vector<std::vector<std::vector<int>>>(_numNodes);
    _childEdgeType = std::vector<std::vector<int>>(_numNodes);
    _aggreEdgeType = std::vector<std::vector<int>>(_numNodes);
    _partitionCandPos = std::vector<std::vector<bool>>(numPartition);
    _partitionTriPos = std::vector<std::vector<std::pair<int, int>>>(numPartition);
    _partitionEdgeType = std::vector<std::vector<int>>(numPartition);
    _partitionEndType = std::vector<std::vector<int>>(numPartition);
    bool directed = p.useDAG();
    // some symmetry rules are erased because they are implemented by edge direction
    std::vector<std::vector<std::vector<VertexID>>> erasedGreaterRules(_numNodes);
    std::vector<std::vector<std::vector<VertexID>>> erasedLessRules(_numNodes);
    std::vector<std::vector<std::vector<VertexID>>> remainingGreaterRules(_numNodes);
    std::vector<std::vector<std::vector<VertexID>>> remainingLessRules(_numNodes);
    for (VertexID nID = 0; nID < _numNodes; ++nID) {
        erasedGreaterRules[nID] = std::vector<std::vector<VertexID>>(_greaterRules.size());
        erasedLessRules[nID] = std::vector<std::vector<VertexID>>(_lessRules.size());
        remainingGreaterRules[nID] = std::vector<std::vector<VertexID>>(_greaterRules.size());
        remainingLessRules[nID] = std::vector<std::vector<VertexID>>(_lessRules.size());
    }
    for (int i = 0; i < numPartition; ++i) {
        int startPos;
        if (i == 0) startPos = 0;
        else startPos = _partitionPos[i - 1] + 1;
        int endPos = _partitionPos[i] + 1;
        ui numNodes = endPos - startPos;
        const std::vector<VertexID> &partitionOrder = _globalOrder[i];
        std::vector<bool> prefixCovered(numNodes, false);
        if (partitionOrder.empty()) {
            for (int j = 0; j < numNodes; ++j) {
                VertexID nID = _postOrder[startPos + j];
                _nodes[nID].nodeOrder = _nodes[nID].localOrder;
                // for node nID, decide its aggrePos based on the partition order, local order and its key
                // not root
                if (startPos + j + 1 != _postOrder.size()) {
                    if (_nodes[nID].keySize == 1) {
                        for (int k = 0; k < _nodes[nID].localOrder.size(); ++k) {
                            if (_nodes[nID].localOrder[k] == _nodes[nID].key[0]) {
                                _aggrePos[nID].push_back(k);
                                break;
                            }
                        }
                    }
                    else {
                        VertexID key1 = _nodes[nID].key[0], key2 = _nodes[nID].key[1];
                        for (int k = 0; k < _nodes[nID].localOrder.size(); ++k) {
                            if (_nodes[nID].localOrder[k] == key1) {
                                _aggrePos[nID].push_back(k);
                                break;
                            }
                        }
                        for (int k = 0; k < _nodes[nID].localOrder.size(); ++k) {
                            if (_nodes[nID].localOrder[k] == key2) {
                                _aggrePos[nID].push_back(k);
                                break;
                            }
                        }
                        _nodes[nID].edgeKey = true;
                    }
                }
                    // root
                else {
                    if (_orbitType == 0) {
                        _aggrePos[nID].push_back(0);
                    }
                    else if (_orbitType == 1) {
                        for (int k = 0; k < _aggreV.size(); ++k) {
                            VertexID u = _aggreV[k];
                            for (int l = 0; l < _nodes[nID].localOrder.size(); ++l) {
                                if (_nodes[nID].localOrder[l] == u) {
                                    _aggrePos[nID].push_back(l);
                                    break;
                                }
                            }
                        }
                    }
                    else {
                        for (int k = 0; k < _aggreWeight.size(); ++k) {
                            VertexID u1 = _aggreV[2 * k], u2 = _aggreV[2 * k + 1];
                            for (int l = 0; l < _nodes[nID].localOrder.size(); ++l) {
                                if (_nodes[nID].localOrder[l] == u1) {
                                    _aggrePos[nID].push_back(l);
                                    break;
                                }
                            }
                            for (int l = 0; l < _nodes[nID].localOrder.size(); ++l) {
                                if (_nodes[nID].localOrder[l] == u2) {
                                    _aggrePos[nID].push_back(l);
                                    break;
                                }
                            }
                        }
                        _nodes[nID].edgeKey = true;
                    }
                }
            }
        }
        else {
            _nodesAtStep[i] = std::vector<std::vector<VertexID>>(partitionOrder.size());
            for (int length = 0; length < partitionOrder.size(); ++length) {
                for (int j = 0; j < numNodes; ++j) {
                    VertexID nID = _postOrder[startPos + j];
                    if (!prefixCovered[j]) {
                        bool flag = true;
                        for (VertexID c : _child[nID]) {
                            // find the position of c in the partition
                            int posC = 99;
                            for (int k = 0; k < numNodes; ++k) {
                                if (_postOrder[startPos + k] == c) {
                                    posC = k;
                                    break;
                                }
                            }
                            if (posC != 99 && !prefixCovered[posC]) {
                                flag = false;
                                break;
                            }
                        }
                        if (flag) {
                            if (std::includes(partitionOrder.begin(), partitionOrder.begin() + length + 1,
                                              _nodes[nID].prefix, _nodes[nID].prefix + _nodes[nID].prefixSize)) {
                                prefixCovered[j] = true;
                                _nodesAtStep[i][length].push_back(nID);
                            }
                        }
                    }
                    if (!prefixCovered[j]) break;
                    _nodes[nID].nodeOrder.assign(partitionOrder.begin(), partitionOrder.begin() + length + 1);
                    for (VertexID u: _nodes[nID].localOrder) _nodes[nID].nodeOrder.push_back(u);
                    // for node nID, decide its aggrePos based on the partition order, local order and its key
                    if (startPos + j + 1 != _postOrder.size()) {
                        if (_nodes[nID].keySize == 0) _aggrePos[nID].push_back(0);
                        else if (_nodes[nID].keySize == 1) {
                            for (int k = 0; k < _nodes[nID].nodeOrder.size(); ++k) {
                                if (_nodes[nID].nodeOrder[k] == _nodes[nID].key[0]) {
                                    _aggrePos[nID].push_back(k);
                                    break;
                                }
                            }
                        }
                        else {
                            _nodes[nID].edgeKey = true;
                            VertexID key1 = _nodes[nID].key[0], key2 = _nodes[nID].key[1];
                            for (int k = 0; k < _nodes[nID].nodeOrder.size(); ++k) {
                                if (_nodes[nID].nodeOrder[k] == key1) {
                                    _aggrePos[nID].push_back(k);
                                    break;
                                }
                            }
                            for (int k = 0; k < _nodes[nID].nodeOrder.size(); ++k) {
                                if (_nodes[nID].nodeOrder[k] == key2) {
                                    _aggrePos[nID].push_back(k);
                                    break;
                                }
                            }
                        }
                    }
                    else {
                        if (_orbitType == 0) {
                            _aggrePos[nID].push_back(0);
                        }
                        else if (_orbitType == 1) {
                            for (int k = 0; k < _aggreV.size(); ++k) {
                                VertexID u = _aggreV[k];
                                for (int l = 0; l < _nodes[nID].nodeOrder.size(); ++l) {
                                    if (_nodes[nID].nodeOrder[l] == u) {
                                        _aggrePos[nID].push_back(l);
                                        break;
                                    }
                                }
                            }
                        }
                        else {
                            for (int k = 0; k < _aggreWeight.size(); ++k) {
                                VertexID u1 = _aggreV[2 * k], u2 = _aggreV[2 * k + 1];
                                for (int l = 0; l < _nodes[nID].nodeOrder.size(); ++l) {
                                    if (_nodes[nID].nodeOrder[l] == u1) {
                                        _aggrePos[nID].push_back(l);
                                        break;
                                    }
                                }
                                for (int l = 0; l < _nodes[nID].nodeOrder.size(); ++l) {
                                    if (_nodes[nID].nodeOrder[l] == u2) {
                                        _aggrePos[nID].push_back(l);
                                        break;
                                    }
                                }
                            }
                            _nodes[nID].edgeKey = true;
                        }
                    }
                }
            }
        }
        std::vector<int> numNeighbor(partitionOrder.size(), 0);
        for (int j = 0; j < numNodes; ++j) {
            VertexID nID = _postOrder[startPos + j];
            Node &tau = _nodes[nID];
            _nodeInterPos[nID] = std::vector<bool>(tau.nodeOrder.size(), false);
            _nodeInPos[nID] = std::vector<std::vector<int>>(tau.nodeOrder.size());
            _nodeOutPos[nID] = std::vector<std::vector<int>>(tau.nodeOrder.size());
            _nodeUnPos[nID] = std::vector<std::vector<int>>(tau.nodeOrder.size());
            numNeighbor = std::vector<int>(tau.nodeOrder.size(), 0);
            for (int k = 0; k < tau.nodeOrder.size(); ++k) {
                for (int l = 0; l < k; ++l) {
                    if (p.u.isEdge(tau.nodeOrder[l], tau.nodeOrder[k])) {
                        ++numNeighbor[k];
                        if (!directed) {
                            if (std::find(_greaterRules[tau.nodeOrder[k]].begin(), _greaterRules[tau.nodeOrder[k]].end(),
                                          tau.nodeOrder[l]) != _greaterRules[tau.nodeOrder[k]].end()) {
                                _nodeOutPos[nID][k].push_back(l);
                            }
                            else if (std::find(_lessRules[tau.nodeOrder[k]].begin(), _lessRules[tau.nodeOrder[k]].end(),
                                               tau.nodeOrder[l]) != _lessRules[tau.nodeOrder[k]].end()) {
                                _nodeInPos[nID][k].push_back(l);
                            }
                            else _nodeUnPos[nID][k].push_back(l);
                        }
                        else if (p.in.isEdge(tau.nodeOrder[l], tau.nodeOrder[k])) {
                            _nodeInPos[nID][k].push_back(l);
                        }
                        else {
                            _nodeOutPos[nID][k].push_back(l);
                        }
                    }
                }
            }
            for (int k = 0; k < tau.nodeOrder.size(); ++k) {
                if (numNeighbor[k] > 1)
                    _nodeInterPos[nID][k] = true;
            }
            const std::vector<VertexID> &children = _child[nID];
            _childKeyPos[nID] = std::vector<std::vector<int>>(children.size());
            for (int k = 0; k < children.size(); ++k) {
                VertexID cID = children[k];
                ui keySize = _nodes[cID].keySize;
                if (keySize == 0) {
                    _childKeyPos[nID][k].push_back(0);
                }
                else if (keySize == 1) {
                    VertexID key = _nodes[cID].key[0];
                    for (int l = 0; l < tau.nodeOrder.size(); ++l) {
                        if (tau.nodeOrder[l] == key)
                            _childKeyPos[nID][k].push_back(l);
                    }
                }
                else {
                    tau.edgeKey = true;
                    VertexID key1 = _nodes[cID].key[0], key2 = _nodes[cID].key[1];
                    for (int l = 0; l < tau.nodeOrder.size(); ++l) {
                        if (tau.nodeOrder[l] == key1) {
                            _childKeyPos[nID][k].push_back(l);
                            break;
                        }
                    }
                    for (int l = 0; l < tau.nodeOrder.size(); ++l) {
                        if (tau.nodeOrder[l] == key2) {
                            _childKeyPos[nID][k].push_back(l);
                            break;
                        }
                    }
                }
            }
        }
    }
    // build _posChildEdge, _posAggreEdge, _childEdgeType,
    // edge types: 1: directly use out edge ID or un edge ID; 2: convert in edge ID to out edge ID; 3: get out edge ID
    // 4: get un edge ID: pos1, pos2 5: convert un edge to out edge ID 6: convert un edge ID to reverse un edge ID

    for (VertexID nID = 0; nID < _numNodes; ++nID) {
        _posChildEdge[nID] = std::vector<std::vector<int>>(_nodes[nID].nodeOrder.size());
        std::vector<VertexID> children = _child[nID];
        _childEdgeType[nID] = std::vector<int>(children.size());
        for (int k = 0; k < _childKeyPos[nID].size(); ++k) {
            if (_childKeyPos[nID][k].size() == 2) {
                int pos1 = _childKeyPos[nID][k][0], pos2 = _childKeyPos[nID][k][1];
                int pos = pos1 > pos2 ? pos1 : pos2;
                int smallerPos = pos1 < pos2 ? pos1 : pos2;
                _posChildEdge[nID][pos].push_back(k);
                if (!_nodeInterPos[nID][pos]) {
                    if (!_nodeInPos[nID][pos].empty()) _childEdgeType[nID][k] = 2;
                    else if (!_nodeOutPos[nID][pos].empty()) _childEdgeType[nID][k] = 1;
                    else {
                        if (pos == pos2) _childEdgeType[nID][k] = 1;
                        else _childEdgeType[nID][k] = 6;
                    }
                }
                else {
                    if (std::find(_nodeUnPos[nID][pos].begin(), _nodeUnPos[nID][pos].end(), smallerPos) == _nodeUnPos[nID][pos].end())
                        _childEdgeType[nID][k] = 3;
                    else {
                        _childEdgeType[nID][k] = 4;
                    }
                }
            }
        }
        _posAggreEdge[nID] = std::vector<std::vector<int>>(_nodes[nID].nodeOrder.size());
        // not root
        if (nID != _postOrder.back()) {
            if (_nodes[nID].keySize == 2) {
                int pos1 = _aggrePos[nID][0], pos2 = _aggrePos[nID][1];
                int pos = pos1 > pos2 ? pos1 : pos2;
                int smallerPos = pos1 < pos2 ? pos1 : pos2;
                _posAggreEdge[nID][pos].push_back(0);
                if (!_nodeInterPos[nID][pos]) {
                    if (!_nodeInPos[nID][pos].empty()) _aggreEdgeType[nID].push_back(2);
                    else {
                        _aggreEdgeType[nID].push_back(1);
                        if (!directed && std::find(_nodeUnPos[nID][pos].begin(), _nodeUnPos[nID][pos].end(), smallerPos) != _nodeUnPos[nID][pos].end()) {
                            _nodes[nID].unEdgeKey = true;
                            if (pos != pos2) _aggreEdgeType[nID][0] = 6;
                        }
                    }
                }
                else {
                    if (std::find(_nodeUnPos[nID][pos].begin(), _nodeUnPos[nID][pos].end(), smallerPos) == _nodeUnPos[nID][pos].end())
                        _aggreEdgeType[nID].push_back(3);
                    else {
                        _aggreEdgeType[nID].push_back(4);
                        _nodes[nID].unEdgeKey = true;
                    }
                }
            }
        }
            // root
        else if (_orbitType == 2) {
            _aggreEdgeType[nID] = std::vector<int>(_aggrePos.size() / 2);
            if (p.u.isEOrbitDir()) {
                for (int i = 0; i < _aggrePos.size(); i = i + 2) {
                    int pos1 = _aggrePos[nID][i], pos2 = _aggrePos[nID][i + 1];
                    int pos = pos1 > pos2 ? pos1 : pos2;
                    _posAggreEdge[nID][pos].push_back(i / 2);
                    if (!_nodeInterPos[nID][pos]) {
                        if (!_nodeOutPos[nID][pos].empty()) _aggreEdgeType[nID][i / 2] = 1;
                        else if (!_nodeInPos[nID][pos].empty()) _aggreEdgeType[nID][i / 2] = 2;
                        else _aggreEdgeType[nID][i / 2] = 5;
                    }
                    else _aggreEdgeType[nID][i / 2] = 3;
                }
            }
            else {
                for (int i = 0; i < _aggrePos.size(); i = i + 2) {
                    int pos1 = _aggrePos[nID][i], pos2 = _aggrePos[nID][i + 1];
                    int pos = pos1 > pos2 ? pos1 : pos2;
                    _posAggreEdge[nID][pos].push_back(i / 2);
                    if (!_nodeInterPos[nID][pos] && !_nodeUnPos[nID][pos].empty()) {
                        if (pos == pos2) _aggreEdgeType[nID][i / 2] = 1;
                        else _aggreEdgeType[nID][i / 2] = 6;
                    }
                    else {
                        _aggreEdgeType[nID][i / 2] = 4;
                    }
                }
            }
        }
    }

    // build triangle structures
    _nodeCandPos = _nodeInterPos;
    _nodeTriPos = std::vector<std::vector<std::pair<int, int>>>(_numNodes);
    _triEdgeType = std::vector<std::vector<int>>(_numNodes);
    _triEndType = std::vector<std::vector<int>>(_numNodes);
    for (VertexID nID = 0; nID < _numNodes; ++nID) {
        _nodeTriPos[nID] = std::vector<std::pair<int, int>>(_nodes[nID].nodeOrder.size());
        _triEdgeType[nID] = std::vector<int>(_nodes[nID].nodeOrder.size(), 0);
        _triEndType[nID] = std::vector<int>(_nodes[nID].nodeOrder.size(), 0);
    }
    for (VertexID nID = 0; nID < _numNodes; ++nID) {
        const Node &tau = _nodes[nID];
        for (int i = 0; i < tau.nodeOrder.size(); ++i) {
            // collect triangle edges and choose the one with the smallest position
            std::vector<int> poses = _nodeInPos[nID][i];
            for (int pos : _nodeOutPos[nID][i]) poses.push_back(pos);
            for (int pos: _nodeUnPos[nID][i]) poses.push_back(pos);
            std::sort(poses.begin(), poses.end());
            for (int j = 1; j < poses.size(); ++j) {
                int p2 = poses[j];
                bool flag = false;
                for (int k = 0; k < j; ++k) {
                    int p1 = poses[k];
                    VertexID u1 = tau.nodeOrder[p1], u2 = tau.nodeOrder[p2], u3 = tau.nodeOrder[i];
                    if (p.u.isEdge(u1, u2)) {
                        flag = true;
                        if (!_nodeInterPos[nID][p2]) {
                            if (!directed) {
                                if (std::find(_lessRules[u2].begin(), _lessRules[u2].end(),
                                              u1) != _lessRules[u2].end()) {
                                    _triEdgeType[nID][i] = 2;
                                }
                                else if (std::find(_greaterRules[u2].begin(), _greaterRules[u2].end(),
                                                   u1) != _greaterRules[u2].end())
                                    _triEdgeType[nID][i] = 1;
                                else _triEdgeType[nID][i] = 5;
                            }
                            else if (p.out.isEdge(u1, u2))
                                _triEdgeType[nID][i] = 1;
                            else
                                _triEdgeType[nID][i] = 2;
                        }
                        else {
                            _triEdgeType[nID][i] = 3;
                        }
                        // if undirected pattern, check whether (i,p1), (i,p2) can be directed.
                        // if so, remove the corresponding symmetry rule.
                        if (!directed) {
                            bool u1u3 = std::find(_lessRules[u1].begin(), _lessRules[u1].end(),
                                                  u3) != _lessRules[u1].end();
                            bool u3u1 = std::find(_greaterRules[u1].begin(), _greaterRules[u1].end(),
                                                  u3) != _greaterRules[u1].end();
                            bool u2u3 = std::find(_lessRules[u2].begin(), _lessRules[u2].end(),
                                                  u3) != _lessRules[u2].end();
                            bool u3u2 = std::find(_greaterRules[u2].begin(), _greaterRules[u2].end(),
                                                  u3) != _greaterRules[u2].end();
                            if (u1u3 && u2u3) {
                                _triEndType[nID][i] = 3;
                                erasedLessRules[nID][u1].push_back(u3);
                                erasedLessRules[nID][u2].push_back(u3);
                                erasedGreaterRules[nID][u3].push_back(u1);
                                erasedGreaterRules[nID][u3].push_back(u2);
                            }
                            else if (u3u1 && u3u2) {
                                _triEndType[nID][i] = 1;
                                erasedLessRules[nID][u3].push_back(u1);
                                erasedLessRules[nID][u3].push_back(u2);
                                erasedGreaterRules[nID][u1].push_back(u3);
                                erasedGreaterRules[nID][u2].push_back(u3);
                            }
                            else if (u1u3 && u3u2) {
                                _triEndType[nID][i] = 2;
                                erasedLessRules[nID][u1].push_back(u3);
                                erasedLessRules[nID][u3].push_back(u2);
                                erasedGreaterRules[nID][u3].push_back(u1);
                                erasedGreaterRules[nID][u2].push_back(u3);
                            }
                            else if (u3u1 && u2u3) {
                                _triEndType[nID][i] = 2;
                                erasedLessRules[nID][u3].push_back(u1);
                                erasedLessRules[nID][u2].push_back(u3);
                                erasedGreaterRules[nID][u1].push_back(u3);
                                erasedGreaterRules[nID][u3].push_back(u2);
                            }
                            else {
                                _triEndType[nID][i] = 4;
                            }
                        }
                        else {
                            if (p.out.isEdge(u2, u3) && p.out.isEdge(u1, u3))
                                _triEndType[nID][i] = 3;
                            else if (p.in.isEdge(u2, u3) && p.in.isEdge(u1, u3))
                                _triEndType[nID][i] = 1;
                            else _triEndType[nID][i] = 2;
                        }
                        _nodeTriPos[nID][i]= std::make_pair(p1, p2);
                    }
                    if (flag) break;
                }
                if (flag) break;
            }
            // reset the _nodeInPos[nID][i], _nodeOutPos and _nodeCandPos : remove positions in _nodeTriPos[nID][i]
            if (useTriangle && _nodeInterPos[nID][i] && !(_nodeTriPos[nID][i].first == _nodeTriPos[nID][i].second)) {
                std::vector<int> inCopy = _nodeInPos[nID][i];
                _nodeInPos[nID][i].clear();
                for (int pos: inCopy) {
                    if (pos != _nodeTriPos[nID][i].first && pos != _nodeTriPos[nID][i].second)
                        _nodeInPos[nID][i].push_back(pos);
                }
                std::vector<int> outCopy = _nodeOutPos[nID][i];
                _nodeOutPos[nID][i].clear();
                for (int pos: outCopy) {
                    if (pos != _nodeTriPos[nID][i].first && pos != _nodeTriPos[nID][i].second)
                        _nodeOutPos[nID][i].push_back(pos);
                }
                std::vector<int> unCopy = _nodeUnPos[nID][i];
                _nodeUnPos[nID][i].clear();
                for (int pos: unCopy)  {
                    if (pos != _nodeTriPos[nID][i].first && pos != _nodeTriPos[nID][i].second)
                        _nodeUnPos[nID][i].push_back(pos);
                }
                if ((_nodeInPos[nID][i].empty() && _nodeOutPos[nID][i].empty()) && _nodeUnPos[nID][i].empty()) _nodeCandPos[nID][i] = false;
            }
            // if undirected pattern, check whether (i,p1), (i,p2) can be directed.
            // if so, remove the corresponding symmetry rule.
            if (!directed) {
                for (int pos: _nodeOutPos[nID][i]) {
                    erasedLessRules[nID][tau.nodeOrder[pos]].push_back(tau.nodeOrder[i]);
                    erasedGreaterRules[nID][tau.nodeOrder[i]].push_back(tau.nodeOrder[pos]);
                }
                for (int pos: _nodeInPos[nID][i]) {
                    erasedGreaterRules[nID][tau.nodeOrder[pos]].push_back(tau.nodeOrder[i]);
                    erasedLessRules[nID][tau.nodeOrder[i]].push_back(tau.nodeOrder[pos]);
                }
            }
        }
    }

    // update symmetry rules and nodeGreaterPos, nodeLessPos
    for (VertexID nID = 0; nID < _numNodes; ++nID) {
        for (VertexID i = 0; i < _numVertices; ++i) {
            std::sort(erasedGreaterRules[nID][i].begin(), erasedGreaterRules[nID][i].end());
            std::sort(erasedLessRules[nID][i].begin(), erasedLessRules[nID][i].end());
            std::set_difference(_greaterRules[i].begin(), _greaterRules[i].end(), erasedGreaterRules[nID][i].begin(),
                                erasedGreaterRules[nID][i].end(), std::inserter(remainingGreaterRules[nID][i], remainingGreaterRules[nID][i].begin()));
            std::set_difference(_lessRules[i].begin(), _lessRules[i].end(), erasedLessRules[nID][i].begin(),
                                erasedLessRules[nID][i].end(), std::inserter(remainingLessRules[nID][i], remainingLessRules[nID][i].begin()));
        }
    }

    for (VertexID nID = 0; nID < _numNodes; ++nID) {
        const Node &tau = _nodes[nID];
        _nodeGreaterPos[nID] = std::vector<std::vector<int>>(tau.nodeOrder.size());
        _nodeLessPos[nID] = std::vector<std::vector<int>>(tau.nodeOrder.size());
        for (int i = 0; i < tau.nodeOrder.size(); ++i) {
            std::vector<VertexID> greaterVertices = remainingGreaterRules[nID][tau.nodeOrder[i]];
            std::vector<VertexID> lessVertices = remainingLessRules[nID][tau.nodeOrder[i]];
            for (int j = 0; j < i; ++j) {
                if (std::find(greaterVertices.begin(), greaterVertices.end(), tau.nodeOrder[j]) != greaterVertices.end())
                    _nodeGreaterPos[nID][i].push_back(j);
                if (std::find(lessVertices.begin(), lessVertices.end(), tau.nodeOrder[j]) != lessVertices.end())
                    _nodeLessPos[nID][i].push_back(j);
            }
        }
    }
    for (int i = 0; i < numPartition; ++i) {
        const std::vector<VertexID> &partitionOrder = _globalOrder[i];
        if (partitionOrder.empty()) continue;
        VertexID nID = _nodesAtStep[i][partitionOrder.size() - 1][0];
        _partitionInPos[i] = _nodeInPos[nID];
        _partitionOutPos[i] = _nodeOutPos[nID];
        _partitionUnPos[i] = _nodeUnPos[nID];
        _partitionInterPos[i] = _nodeInterPos[nID];
        _partitionGreaterPos[i] = _nodeGreaterPos[nID];
        _partitionLessPos[i] = _nodeLessPos[nID];
        _partitionCandPos[i] = _nodeCandPos[nID];
        _partitionTriPos[i] = _nodeTriPos[nID];
        _partitionEdgeType[i] = _triEdgeType[nID];
        _partitionEndType[i] = _triEndType[nID];
    }
}

void Tree::initMultiJoinPoses(const Pattern &p, bool useTriangle) {
    int numPartition = (int)_partitionPos.size();
    // at partition i, when the partitionOrder[length] is matched, nodes in _nodesAtStep[i][length] can be executed,
    _aggrePos = std::vector<std::vector<int>>(_numNodes);
    _nodeInterPos = std::vector<std::vector<bool>>(_numNodes);
    _nodeInPos = std::vector<std::vector<std::vector<int>>>(_numNodes);
    _nodeOutPos = std::vector<std::vector<std::vector<int>>>(_numNodes);
    _nodeUnPos = std::vector<std::vector<std::vector<int>>>(_numNodes);
    _nodeGreaterPos = std::vector<std::vector<std::vector<int>>>(_numNodes);
    _nodeLessPos = std::vector<std::vector<std::vector<int>>>(_numNodes);
    _childKeyPos = std::vector<std::vector<std::vector<int>>>(_numNodes);
    _posChildEdge = std::vector<std::vector<std::vector<int>>>(_numNodes);
    _posAggreEdge = std::vector<std::vector<std::vector<int>>>(_numNodes);
    _childEdgeType = std::vector<std::vector<int>>(_numNodes);
    _aggreEdgeType = std::vector<std::vector<int>>(_numNodes);
    // two different data structures
    _nodesAtStep = std::vector<std::vector<std::vector<VertexID>>>(_numNodes);
    _prefixPos = std::vector<std::vector<int>>(_numNodes);
    bool directed = p.useDAG();
    // some symmetry rules are erased because they are implemented by edge direction
    std::vector<std::vector<std::vector<VertexID>>> erasedGreaterRules(_numNodes);
    std::vector<std::vector<std::vector<VertexID>>> erasedLessRules(_numNodes);
    std::vector<std::vector<std::vector<VertexID>>> remainingGreaterRules(_numNodes);
    std::vector<std::vector<std::vector<VertexID>>> remainingLessRules(_numNodes);
    for (VertexID nID = 0; nID < _numNodes; ++nID) {
        erasedGreaterRules[nID] = std::vector<std::vector<VertexID>>(_greaterRules.size());
        erasedLessRules[nID] = std::vector<std::vector<VertexID>>(_lessRules.size());
        remainingGreaterRules[nID] = std::vector<std::vector<VertexID>>(_greaterRules.size());
        remainingLessRules[nID] = std::vector<std::vector<VertexID>>(_lessRules.size());
        _nodesAtStep[nID] = std::vector<std::vector<VertexID>>(_nodes[nID].numVertices);
    }
    for (VertexID nID = 0; nID < _numNodes; ++nID) {
        if (_nodes[nID].prefixSize != 0) continue;
        std::queue<VertexID> nodeQ;
        nodeQ.push(nID);
        while (!nodeQ.empty()) {
            VertexID nID1 = nodeQ.front();
            nodeQ.pop();
            _nodes[nID1].nodeOrder.assign(_nodes[nID1].prefix, _nodes[nID1].prefix + _nodes[nID1].prefixSize);
            for (int k = 0; k < _nodes[nID1].localOrder.size(); ++k) {
                _nodes[nID1].nodeOrder.push_back(_nodes[nID1].localOrder[k]);
            }
            for (VertexID nID2: _child[nID1]) {
                if (_nodes[nID2].prefixSize == 0) continue;
                for (int i = 0; i < _nodes[nID2].prefixSize; ++i) {
                    for (int j = 0; j < _nodes[nID1].nodeOrder.size(); ++j) {
                        if (_nodes[nID2].prefix[i] == _nodes[nID1].nodeOrder[j]) {
                            _prefixPos[nID2].push_back(j);
                        }
                    }
                }
                std::sort(_prefixPos[nID2].begin(), _prefixPos[nID2].end());
                for (int i = 0; i < _nodes[nID2].prefixSize; ++i) {
                    _nodes[nID2].prefix[i] = _nodes[nID1].nodeOrder[_prefixPos[nID2][i]];
                }
                ui maxPos = _prefixPos[nID2].back();
                if (maxPos < _nodes[nID1].prefixSize) maxPos = 0;
                _nodesAtStep[nID1][maxPos].push_back(nID2);
                nodeQ.push(nID2);
            }
        }
    }

    for (int i = 0; i < numPartition; ++i) {
        int startPos;
        if (i == 0) startPos = 0;
        else startPos = _partitionPos[i - 1] + 1;
        int endPos = _partitionPos[i] + 1;
        ui numNodes = endPos - startPos;
        for (int j = 0; j < numNodes; ++j) {
            VertexID nID = _postOrder[startPos + j];
            // for node nID, decide its aggrePos based on the partition order, local order and its key
            // not root
            if (startPos + j + 1 != _postOrder.size()) {
                if (_nodes[nID].keySize == 0) _aggrePos[nID].push_back(0);
                else if (_nodes[nID].keySize == 1) {
                    for (int k = 0; k < _nodes[nID].nodeOrder.size(); ++k) {
                        if (_nodes[nID].nodeOrder[k] == _nodes[nID].key[0]) {
                            _aggrePos[nID].push_back(k);
                            break;
                        }
                    }
                }
                else {
                    _nodes[nID].edgeKey = true;
                    VertexID key1 = _nodes[nID].key[0], key2 = _nodes[nID].key[1];
                    for (int k = 0; k < _nodes[nID].nodeOrder.size(); ++k) {
                        if (_nodes[nID].nodeOrder[k] == key1) {
                            _aggrePos[nID].push_back(k);
                            break;
                        }
                    }
                    for (int k = 0; k < _nodes[nID].nodeOrder.size(); ++k) {
                        if (_nodes[nID].nodeOrder[k] == key2) {
                            _aggrePos[nID].push_back(k);
                            break;
                        }
                    }
                }
            }
                // root
            else {
                if (_orbitType == 0) {
                    _aggrePos[nID].push_back(0);
                }
                else if (_orbitType == 1) {
                    for (int k = 0; k < _aggreV.size(); ++k) {
                        VertexID u = _aggreV[k];
                        for (int l = 0; l < _nodes[nID].nodeOrder.size(); ++l) {
                            if (_nodes[nID].nodeOrder[l] == u) {
                                _aggrePos[nID].push_back(l);
                                break;
                            }
                        }
                    }
                }
                else {
                    for (int k = 0; k < _aggreWeight.size(); ++k) {
                        VertexID u1 = _aggreV[2 * k], u2 = _aggreV[2 * k + 1];
                        for (int l = 0; l < _nodes[nID].nodeOrder.size(); ++l) {
                            if (_nodes[nID].nodeOrder[l] == u1) {
                                _aggrePos[nID].push_back(l);
                                break;
                            }
                        }
                        for (int l = 0; l < _nodes[nID].nodeOrder.size(); ++l) {
                            if (_nodes[nID].nodeOrder[l] == u2) {
                                _aggrePos[nID].push_back(l);
                                break;
                            }
                        }
                    }
                    _nodes[nID].edgeKey = true;
                }
            }
        }
        std::vector<int> numNeighbor;
        for (int j = 0; j < numNodes; ++j) {
            VertexID nID = _postOrder[startPos + j];
            Node &tau = _nodes[nID];
            _nodeInterPos[nID] = std::vector<bool>(tau.nodeOrder.size(), false);
            _nodeInPos[nID] = std::vector<std::vector<int>>(tau.nodeOrder.size());
            _nodeOutPos[nID] = std::vector<std::vector<int>>(tau.nodeOrder.size());
            _nodeUnPos[nID] = std::vector<std::vector<int>>(tau.nodeOrder.size());
            numNeighbor = std::vector<int>(tau.nodeOrder.size(), 0);
            for (int k = 0; k < tau.nodeOrder.size(); ++k) {
                for (int l = 0; l < k; ++l) {
                    if (p.u.isEdge(tau.nodeOrder[l], tau.nodeOrder[k])) {
                        ++numNeighbor[k];
                        if (!directed) {
                            if (std::find(_greaterRules[tau.nodeOrder[k]].begin(), _greaterRules[tau.nodeOrder[k]].end(),
                                          tau.nodeOrder[l]) != _greaterRules[tau.nodeOrder[k]].end()) {
                                _nodeOutPos[nID][k].push_back(l);
                            }
                            else if (std::find(_lessRules[tau.nodeOrder[k]].begin(), _lessRules[tau.nodeOrder[k]].end(),
                                               tau.nodeOrder[l]) != _lessRules[tau.nodeOrder[k]].end()) {
                                _nodeInPos[nID][k].push_back(l);
                            }
                            else _nodeUnPos[nID][k].push_back(l);
                        }
                        else if (p.in.isEdge(tau.nodeOrder[l], tau.nodeOrder[k])) {
                            _nodeInPos[nID][k].push_back(l);
                        }
                        else {
                            _nodeOutPos[nID][k].push_back(l);
                        }
                    }
                }
            }
            for (int k = 0; k < tau.nodeOrder.size(); ++k) {
                if (numNeighbor[k] > 1)
                    _nodeInterPos[nID][k] = true;
            }
            const std::vector<VertexID> &children = _child[nID];
            _childKeyPos[nID] = std::vector<std::vector<int>>(children.size());
            for (int k = 0; k < children.size(); ++k) {
                VertexID cID = children[k];
                ui keySize = _nodes[cID].keySize;
                if (keySize == 1) {
                    VertexID key = _nodes[cID].key[0];
                    for (int l = 0; l < tau.nodeOrder.size(); ++l) {
                        if (tau.nodeOrder[l] == key)
                            _childKeyPos[nID][k].push_back(l);
                    }
                }
                else if (keySize == 2) {
                    tau.edgeKey = true;
                    VertexID key1 = _nodes[cID].key[0], key2 = _nodes[cID].key[1];
                    for (int l = 0; l < tau.nodeOrder.size(); ++l) {
                        if (tau.nodeOrder[l] == key1) {
                            _childKeyPos[nID][k].push_back(l);
                            break;
                        }
                    }
                    for (int l = 0; l < tau.nodeOrder.size(); ++l) {
                        if (tau.nodeOrder[l] == key2) {
                            _childKeyPos[nID][k].push_back(l);
                            break;
                        }
                    }
                }
            }
        }
    }
    // build _posChildEdge, _posAggreEdge, _childEdgeType,
    // edge types: 1: directly use out edge ID or un edge ID; 2: convert in edge ID to out edge ID; 3: get out edge ID
    // 4: get un edge ID: pos1, pos2 5: convert un edge to out edge ID 6: convert un edge ID to reverse un edge ID

    for (VertexID nID = 0; nID < _numNodes; ++nID) {
        _posChildEdge[nID] = std::vector<std::vector<int>>(_nodes[nID].nodeOrder.size());
        std::vector<VertexID> children = _child[nID];
        _childEdgeType[nID] = std::vector<int>(children.size());
        for (int k = 0; k < _childKeyPos[nID].size(); ++k) {
            if (_childKeyPos[nID][k].size() == 2) {
                int pos1 = _childKeyPos[nID][k][0], pos2 = _childKeyPos[nID][k][1];
                int pos = pos1 > pos2 ? pos1 : pos2;
                int smallerPos = pos1 < pos2 ? pos1 : pos2;
                _posChildEdge[nID][pos].push_back(k);
                if (!_nodeInterPos[nID][pos]) {
                    if (!_nodeInPos[nID][pos].empty()) _childEdgeType[nID][k] = 2;
                    else if (!_nodeOutPos[nID][pos].empty()) _childEdgeType[nID][k] = 1;
                    else {
                        if (pos == pos2) _childEdgeType[nID][k] = 1;
                        else _childEdgeType[nID][k] = 6;
                    }
                }
                else {
                    if (std::find(_nodeUnPos[nID][pos].begin(), _nodeUnPos[nID][pos].end(), smallerPos) == _nodeUnPos[nID][pos].end())
                        _childEdgeType[nID][k] = 3;
                    else {
                        _childEdgeType[nID][k] = 4;
                    }
                }
            }
        }
        _posAggreEdge[nID] = std::vector<std::vector<int>>(_nodes[nID].nodeOrder.size());
        // not root
        if (nID != _postOrder.back()) {
            if (_nodes[nID].keySize == 2) {
                int pos1 = _aggrePos[nID][0], pos2 = _aggrePos[nID][1];
                int pos = pos1 > pos2 ? pos1 : pos2;
                int smallerPos = pos1 < pos2 ? pos1 : pos2;
                _posAggreEdge[nID][pos].push_back(0);
                if (!_nodeInterPos[nID][pos]) {
                    if (!_nodeInPos[nID][pos].empty()) _aggreEdgeType[nID].push_back(2);
                    else {
                        _aggreEdgeType[nID].push_back(1);
                        if (!directed && std::find(_nodeUnPos[nID][pos].begin(), _nodeUnPos[nID][pos].end(), smallerPos) != _nodeUnPos[nID][pos].end()) {
                            _nodes[nID].unEdgeKey = true;
                            if (pos != pos2) _aggreEdgeType[nID][0] = 6;
                        }
                    }
                }
                else {
                    if (std::find(_nodeUnPos[nID][pos].begin(), _nodeUnPos[nID][pos].end(), smallerPos) == _nodeUnPos[nID][pos].end())
                        _aggreEdgeType[nID].push_back(3);
                    else {
                        _aggreEdgeType[nID].push_back(4);
                        _nodes[nID].unEdgeKey = true;
                    }
                }
            }
        }
            // root
        else if (_orbitType == 2) {
            _aggreEdgeType[nID] = std::vector<int>(_aggrePos.size() / 2);
            if (p.u.isEOrbitDir()) {
                for (int i = 0; i < _aggrePos.size(); i = i + 2) {
                    int pos1 = _aggrePos[nID][i], pos2 = _aggrePos[nID][i + 1];
                    int pos = pos1 > pos2 ? pos1 : pos2;
                    _posAggreEdge[nID][pos].push_back(i / 2);
                    if (!_nodeInterPos[nID][pos]) {
                        if (!_nodeOutPos[nID][pos].empty()) _aggreEdgeType[nID][i / 2] = 1;
                        else if (!_nodeInPos[nID][pos].empty()) _aggreEdgeType[nID][i / 2] = 2;
                        else _aggreEdgeType[nID][i / 2] = 5;
                    }
                    else _aggreEdgeType[nID][i / 2] = 3;
                }
            }
            else {
                for (int i = 0; i < _aggrePos.size(); i = i + 2) {
                    int pos1 = _aggrePos[nID][i], pos2 = _aggrePos[nID][i + 1];
                    int pos = pos1 > pos2 ? pos1 : pos2;
                    _posAggreEdge[nID][pos].push_back(i / 2);
                    if (!_nodeInterPos[nID][pos] && !_nodeUnPos[nID][pos].empty()) {
                        if (pos == pos2) _aggreEdgeType[nID][i / 2] = 1;
                        else _aggreEdgeType[nID][i / 2] = 6;
                    }
                    else {
                        _aggreEdgeType[nID][i / 2] = 4;
                    }
                }
            }
        }
    }

    // build triangle structures
    _nodeCandPos = _nodeInterPos;
    _nodeTriPos = std::vector<std::vector<std::pair<int, int>>>(_numNodes);
    _triEdgeType = std::vector<std::vector<int>>(_numNodes);
    _triEndType = std::vector<std::vector<int>>(_numNodes);
    for (VertexID nID = 0; nID < _numNodes; ++nID) {
        _nodeTriPos[nID] = std::vector<std::pair<int, int>>(_nodes[nID].nodeOrder.size());
        _triEdgeType[nID] = std::vector<int>(_nodes[nID].nodeOrder.size(), 0);
        _triEndType[nID] = std::vector<int>(_nodes[nID].nodeOrder.size(), 0);
    }
    for (VertexID nID = 0; nID < _numNodes; ++nID) {
        const Node &tau = _nodes[nID];
        for (int i = 0; i < tau.nodeOrder.size(); ++i) {
            // collect triangle edges and choose the one with the smallest position
            std::vector<int> poses = _nodeInPos[nID][i];
            for (int pos : _nodeOutPos[nID][i]) poses.push_back(pos);
            for (int pos: _nodeUnPos[nID][i]) poses.push_back(pos);
            std::sort(poses.begin(), poses.end());
            for (int j = 1; j < poses.size(); ++j) {
                int p2 = poses[j];
                bool flag = false;
                for (int k = 0; k < j; ++k) {
                    int p1 = poses[k];
                    VertexID u1 = tau.nodeOrder[p1], u2 = tau.nodeOrder[p2], u3 = tau.nodeOrder[i];
                    if (p.u.isEdge(u1, u2)) {
                        flag = true;
                        if (!_nodeInterPos[nID][p2]) {
                            if (!directed) {
                                if (std::find(_lessRules[u2].begin(), _lessRules[u2].end(),
                                              u1) != _lessRules[u2].end()) {
                                    _triEdgeType[nID][i] = 2;
                                }
                                else if (std::find(_greaterRules[u2].begin(), _greaterRules[u2].end(),
                                                   u1) != _greaterRules[u2].end())
                                    _triEdgeType[nID][i] = 1;
                                else _triEdgeType[nID][i] = 5;
                            }
                            else if (p.out.isEdge(u1, u2))
                                _triEdgeType[nID][i] = 1;
                            else
                                _triEdgeType[nID][i] = 2;
                        }
                        else {
                            _triEdgeType[nID][i] = 3;
                        }
                        // if undirected pattern, check whether (i,p1), (i,p2) can be directed.
                        // if so, remove the corresponding symmetry rule.
                        if (!directed) {
                            bool u1u3 = std::find(_lessRules[u1].begin(), _lessRules[u1].end(),
                                                  u3) != _lessRules[u1].end();
                            bool u3u1 = std::find(_greaterRules[u1].begin(), _greaterRules[u1].end(),
                                                  u3) != _greaterRules[u1].end();
                            bool u2u3 = std::find(_lessRules[u2].begin(), _lessRules[u2].end(),
                                                  u3) != _lessRules[u2].end();
                            bool u3u2 = std::find(_greaterRules[u2].begin(), _greaterRules[u2].end(),
                                                  u3) != _greaterRules[u2].end();
                            if (u1u3 && u2u3) {
                                _triEndType[nID][i] = 3;
                                erasedLessRules[nID][u1].push_back(u3);
                                erasedLessRules[nID][u2].push_back(u3);
                                erasedGreaterRules[nID][u3].push_back(u1);
                                erasedGreaterRules[nID][u3].push_back(u2);
                            }
                            else if (u3u1 && u3u2) {
                                _triEndType[nID][i] = 1;
                                erasedLessRules[nID][u3].push_back(u1);
                                erasedLessRules[nID][u3].push_back(u2);
                                erasedGreaterRules[nID][u1].push_back(u3);
                                erasedGreaterRules[nID][u2].push_back(u3);
                            }
                            else if (u1u3 && u3u2) {
                                _triEndType[nID][i] = 2;
                                erasedLessRules[nID][u1].push_back(u3);
                                erasedLessRules[nID][u3].push_back(u2);
                                erasedGreaterRules[nID][u3].push_back(u1);
                                erasedGreaterRules[nID][u2].push_back(u3);
                            }
                            else if (u3u1 && u2u3) {
                                _triEndType[nID][i] = 2;
                                erasedLessRules[nID][u3].push_back(u1);
                                erasedLessRules[nID][u2].push_back(u3);
                                erasedGreaterRules[nID][u1].push_back(u3);
                                erasedGreaterRules[nID][u3].push_back(u2);
                            }
                            else {
                                _triEndType[nID][i] = 4;
                            }
                        }
                        else {
                            if (p.out.isEdge(u2, u3) && p.out.isEdge(u1, u3))
                                _triEndType[nID][i] = 3;
                            else if (p.in.isEdge(u2, u3) && p.in.isEdge(u1, u3))
                                _triEndType[nID][i] = 1;
                            else _triEndType[nID][i] = 2;
                        }
                        _nodeTriPos[nID][i]= std::make_pair(p1, p2);
                    }
                    if (flag) break;
                }
                if (flag) break;
            }
            // reset the _nodeInPos[nID][i], _nodeOutPos and _nodeCandPos : remove positions in _nodeTriPos[nID][i]
            if (useTriangle && _nodeInterPos[nID][i] && !(_nodeTriPos[nID][i].first == _nodeTriPos[nID][i].second)) {
                std::vector<int> inCopy = _nodeInPos[nID][i];
                _nodeInPos[nID][i].clear();
                for (int pos: inCopy) {
                    if (pos != _nodeTriPos[nID][i].first && pos != _nodeTriPos[nID][i].second)
                        _nodeInPos[nID][i].push_back(pos);
                }
                std::vector<int> outCopy = _nodeOutPos[nID][i];
                _nodeOutPos[nID][i].clear();
                for (int pos: outCopy) {
                    if (pos != _nodeTriPos[nID][i].first && pos != _nodeTriPos[nID][i].second)
                        _nodeOutPos[nID][i].push_back(pos);
                }
                std::vector<int> unCopy = _nodeUnPos[nID][i];
                _nodeUnPos[nID][i].clear();
                for (int pos: unCopy)  {
                    if (pos != _nodeTriPos[nID][i].first && pos != _nodeTriPos[nID][i].second)
                        _nodeUnPos[nID][i].push_back(pos);
                }
                if ((_nodeInPos[nID][i].empty() && _nodeOutPos[nID][i].empty()) && _nodeUnPos[nID][i].empty()) _nodeCandPos[nID][i] = false;
            }
            // if undirected pattern, check whether (i,p1), (i,p2) can be directed.
            // if so, remove the corresponding symmetry rule.
            if (!directed) {
                for (int pos: _nodeOutPos[nID][i]) {
                    erasedLessRules[nID][tau.nodeOrder[pos]].push_back(tau.nodeOrder[i]);
                    erasedGreaterRules[nID][tau.nodeOrder[i]].push_back(tau.nodeOrder[pos]);
                }
                for (int pos: _nodeInPos[nID][i]) {
                    erasedGreaterRules[nID][tau.nodeOrder[pos]].push_back(tau.nodeOrder[i]);
                    erasedLessRules[nID][tau.nodeOrder[i]].push_back(tau.nodeOrder[pos]);
                }
            }
        }
    }
    // if an edge is in the prefix, its type is related to its parent
    for (auto it = _postOrder.rbegin(); it != _postOrder.rend(); ++it) {
        VertexID nID = *it;
        VertexID pID = _parent[nID];
        if (_nodes[nID].prefixSize == 0) continue;
        for (int i = 0; i < _nodes[nID].numVertices; ++i) {
            if (_triEdgeType[nID][i] != 0) {
                int pos1 = _nodeTriPos[nID][i].first, pos2 = _nodeTriPos[nID][i].second;
                if (pos2 >= _nodes[nID].prefixSize) continue;
                int pPos1 = _prefixPos[nID][pos1], pPos2 = _prefixPos[nID][pos2];
                if (_nodeInterPos[pID][pPos2] || pPos2 < _nodes[pID].prefixSize) _triEdgeType[nID][i] = 3;
            }
        }
        for (int i = 0; i < _childKeyPos[nID].size(); ++i) {
            if (_childKeyPos[nID][i].size() == 2) {
                int pos1 = _childKeyPos[nID][i][0], pos2 = _childKeyPos[nID][i][1];
                int pos = pos1 > pos2 ? pos1 : pos2;
                int smallerPos = pos1 < pos2 ? pos1 : pos2;
                if (pos >= _nodes[nID].prefixSize) continue;
                int pPos = _prefixPos[nID][pos];
                if (_nodeInterPos[pID][pPos] || pPos < _nodes[pID].prefixSize) {
                    if (std::find(_nodeUnPos[nID][pos].begin(), _nodeUnPos[nID][pos].end(), smallerPos) == _nodeUnPos[nID][pos].end()
                        && !_nodeInterPos[nID][pos])
                        _childEdgeType[nID][i] = 3;
                    else {
                        _childEdgeType[nID][i] = 4;
                    }
                }
            }
        }
        if (_nodes[nID].keySize == 2) {
            int pos1 = _aggrePos[nID][0], pos2 = _aggrePos[nID][1];
            int pos = pos1 > pos2 ? pos1 : pos2;
            int smallerPos = pos1 < pos2 ? pos1 : pos2;
            if (pos >= _nodes[nID].prefixSize) continue;
            int pPos = _prefixPos[nID][pos];
            if (_nodeInterPos[pID][pPos] || pPos < _nodes[pID].prefixSize) {
                if (std::find(_nodeUnPos[nID][pos].begin(), _nodeUnPos[nID][pos].end(), smallerPos) == _nodeUnPos[nID][pos].end()
                    && !_nodeInterPos[nID][pos])
                    _aggreEdgeType[nID][0] = 3;
                else {
                    _aggreEdgeType[nID][0] = 4;
                }
            }
        }
    }
    // update symmetry rules and nodeGreaterPos, nodeLessPos
    for (VertexID nID = 0; nID < _numNodes; ++nID) {
        for (VertexID i = 0; i < _numVertices; ++i) {
            std::sort(erasedGreaterRules[nID][i].begin(), erasedGreaterRules[nID][i].end());
            std::sort(erasedLessRules[nID][i].begin(), erasedLessRules[nID][i].end());
            std::set_difference(_greaterRules[i].begin(), _greaterRules[i].end(), erasedGreaterRules[nID][i].begin(),
                                erasedGreaterRules[nID][i].end(), std::inserter(remainingGreaterRules[nID][i], remainingGreaterRules[nID][i].begin()));
            std::set_difference(_lessRules[i].begin(), _lessRules[i].end(), erasedLessRules[nID][i].begin(),
                                erasedLessRules[nID][i].end(), std::inserter(remainingLessRules[nID][i], remainingLessRules[nID][i].begin()));
        }
    }

    for (VertexID nID = 0; nID < _numNodes; ++nID) {
        const Node &tau = _nodes[nID];
        _nodeGreaterPos[nID] = std::vector<std::vector<int>>(tau.nodeOrder.size());
        _nodeLessPos[nID] = std::vector<std::vector<int>>(tau.nodeOrder.size());
        for (int i = 0; i < tau.nodeOrder.size(); ++i) {
            std::vector<VertexID> greaterVertices = remainingGreaterRules[nID][tau.nodeOrder[i]];
            std::vector<VertexID> lessVertices = remainingLessRules[nID][tau.nodeOrder[i]];
            for (int j = 0; j < i; ++j) {
                if (std::find(greaterVertices.begin(), greaterVertices.end(), tau.nodeOrder[j]) != greaterVertices.end())
                    _nodeGreaterPos[nID][i].push_back(j);
                if (std::find(lessVertices.begin(), lessVertices.end(), tau.nodeOrder[j]) != lessVertices.end())
                    _nodeLessPos[nID][i].push_back(j);
            }
        }
    }
}

void Tree::print() const {
    printf("multi factor = %d\n", _multiFactor);
    if (_postOrder.empty()) {
        for (int i = 0; i < _numNodes; ++i) {
            printf("node %d, vertices: ", i);
            for (int j = 0; j < _nodes[i].numVertices; ++j) {
                printf("%d, ", _nodes[i].vertices[j]);
            }
            printf(" fw = %.1f", _nodes[i].fw);
        }
    }
    else {
        for (VertexID nID: _postOrder) {
            printf("node %d, vertices: ", nID);
            for (int j = 0; j < _nodes[nID].numVertices; ++j) {
                printf("%d, ", _nodes[nID].vertices[j]);
            }
            if (_nodes[nID].prefixSize > 0) {
                printf("prefix: ");
                for (int j = 0; j < _nodes[nID].prefixSize; ++j) {
                    printf("%d, ", _nodes[nID].prefix[j]);
                }
            }
            if (_nodes[nID].keySize > 0) {
                printf("key: ");
                for (int j = 0; j < _nodes[nID].keySize; ++j) {
                    printf("%d, ", _nodes[nID].key[j]);
                }
            }
            if (!_nodes[nID].localOrder.empty()) {
                printf("local order: ");
                for (VertexID j : _nodes[nID].localOrder) {
                    printf("%d, ", j);
                }
            }
            printf(" fw = %.1f", _nodes[nID].fw);
            printf("\n");
        }
    }
    printf("edges: ");
    for (int i = 0; i < _numNodes; ++i) {
        for (int j = 0; j < _edges[i].size(); ++j)
            if (_edges[i][j] > i)
                printf("(node %d, node %d) ", i, _edges[i][j]);
    }
    printf("\n");
    if (!_globalOrder.empty()) {
        printf("global order:\n");
        for (int i = 0; i < _globalOrder.size(); ++i) {
            printf("partition %d ", i + 1);
            printf("at node %d: ", _postOrder[_partitionPos[i]]);
            for (VertexID j : _globalOrder[i])
                printf("%d ", j);
            printf("\n");
        }
    }
    if (_parent != nullptr) {
        for (int i = 0; i < _numNodes; ++i) {
            if (_parent[i] == 99)
                printf("root: node %d\n", i);
        }
        printf("aggregation vertices: ");
        if (_orbitType == 1) {
            for (int i = 0; i < _aggreV.size(); ++i) {
                VertexID u = _aggreV[i];
                int weight = _aggreWeight[i];
                printf("id: %d, weight: %d ", u, weight);
            }
        }
        if (_orbitType == 2) {
            for (int i = 0; i < _aggreV.size(); i = i + 2) {
                VertexID u1 = _aggreV[i], u2 = _aggreV[i + 1];
                int weight = _aggreWeight[i / 2];
                printf("id: %d, %d, weight: %d ", u1, u2, weight);
            }
        }
        printf("\n");
    }
}

void Tree::printRules() const {
    for (VertexID nID = 0; nID < _numNodes; ++nID) {
        printf("In node %d:\n", nID);
        const Node &tau = _nodes[nID];
        for (int i = 0; i < tau.nodeOrder.size(); ++i) {
            VertexID u = tau.nodeOrder[i];
            if (!_nodeInPos[nID][i].empty()) {
                printf("in edges to %d: ", u);
                for (int pos: _nodeInPos[nID][i]) {
                    printf("%d, ", tau.nodeOrder[pos]);
                }
                printf("\n");
            }
            if (!_nodeOutPos[nID][i].empty()) {
                printf("out edges to %d: ", u);
                for (int pos: _nodeOutPos[nID][i]) {
                    printf("%d, ", tau.nodeOrder[pos]);
                }
                printf("\n");
            }
        }
    }
    for (VertexID u = 0; u < _numVertices; ++u) {
        if (!_greaterRules[u].empty()) {
            printf("%d > ", u);
            for (VertexID v: _greaterRules[u]) {
                printf("%d, ", v);
            }
            printf("\n");
        }
        if (!_lessRules[u].empty()) {
            printf("%d < ", u);
            for (VertexID v: _lessRules[u]) {
                printf("%d, ", v);
            }
            printf("\n");
        }
    }
}

bool Tree::operator<(const Tree &rhs) const {
    ui lRootSize = 0, rRootSize = 0;
    for (VertexID nID = 0; nID < _numNodes; ++nID) {
        if (_parent[nID] == 99) lRootSize = _nodes[nID].numVertices;
    }
    for (VertexID nID = 0; nID < rhs._numNodes; ++nID) {
        if (rhs._parent[nID] == 99) rRootSize = rhs._nodes[nID].numVertices;
    }
    if (lRootSize > rRootSize) return true;
    else if (lRootSize == rRootSize && _numVertices > rhs._numVertices) return true;
    else return false;
}

void Tree::dropRules(const Pattern &p, bool useTriangle, int multiFactor) {
    for (int i = 0; i < _numVertices; ++i) {
        _greaterRules[i].clear();
        _lessRules[i].clear();
    }
    // set to single-node aggregation
    _aggreV.clear();
    _aggreWeight.clear();
    if (_orbitType == 1)
        _aggreV.push_back(0);
    else if (_orbitType == 2) {
        _aggreV.push_back(0);
        _aggreV.push_back(1);
    }
    _aggreWeight.push_back(1);
    _multiFactor /= multiFactor;
    if (_executeMode) initPoses(p, useTriangle);
    else initMultiJoinPoses(p, useTriangle);
}

void Tree::computeNodeRules(VertexID nID, const PatternGraph &p) {
//    std::set<VertexID> allCut;
//    for (VertexID cID : _child[nID]) {
//        for (int i = 0; i < _nodes[cID].cutSize; ++i) {
//            allCut.insert(_nodes[cID].cut[i]);
//        }
//    }
//    for (int i = 0; i < _nodes[nID].cutSize; ++i)
//        allCut.insert(_nodes[nID].cut[i]);
//    _nodes[nID].computeValidRules(p, allCut);
}

void Tree::computeWidths(const PatternGraph &p) {
    _sumWidth = 0;
    if (_executeMode) {
        for (VertexID nID = 0; nID < _numNodes; ++nID) {
            Node &tau = _nodes[nID];
            if (canonToFW.find(tau.canonValue) == canonToFW.end()) {
                fractionalWidth(tau, p, std::vector<VertexID>());
                canonToFW[tau.canonValue] = tau.fw;
            }
            else {
                tau.fw = canonToFW[tau.canonValue];
            }
            if (tau.fw > _fhw) _fhw = tau.fw;
            ui width = _nodes[nID].numVertices;
            if (width > _treeWidth)
                _treeWidth = width;
            _sumWidth += width;
        }
    }
    else {
        for (VertexID nID = 0; nID < _numNodes; ++nID) {
            if (_nodes[nID].prefixSize != 0) continue;
            std::queue<VertexID> nodeQ;
            std::queue<std::vector<VertexID>> pathQ;
            nodeQ.push(nID);
            pathQ.emplace();
            while (!nodeQ.empty()) {
                VertexID nID1 = nodeQ.front();
                std::vector<VertexID> path = pathQ.front();
                nodeQ.pop();
                pathQ.pop();
                fractionalWidth(_nodes[nID1], p, path);
                if (_nodes[nID1].fw > _fhw) _fhw = _nodes[nID1].fw;
                ui width = _nodes[nID1].localOrder.size() + path.size();
                if (width > _treeWidth) _treeWidth = width;
                _sumWidth += width;
                for (VertexID nID2: _child[nID1]) {
                    if (_nodes[nID2].prefixSize == 0) continue;
                    std::vector<VertexID> newPath = path;
                    int prefixMatchedPos = 0;
                    for (int i = 0; i < _nodes[nID2].prefixSize; ++i) {
                        bool prefixInPath = false;
                        for (int j = 0; j < path.size(); ++j) {
                            if (_nodes[nID2].prefix[i] == path[j]) {
                                prefixInPath = true;
                                break;
                            }
                        }
                        if (!prefixInPath) {
                            for (int j = 0; j < _nodes[nID1].localOrder.size(); ++j) {
                                if (_nodes[nID2].prefix[i] == _nodes[nID1].localOrder[j]) {
                                    if (j + 1 > prefixMatchedPos) prefixMatchedPos = j + 1;
                                    break;
                                }
                            }
                        }
                    }
                    for (int i = 0; i < prefixMatchedPos; ++i) {
                        newPath.push_back(_nodes[nID1].localOrder[i]);
                    }
                    nodeQ.push(nID2);
                    pathQ.push(newPath);
                }
            }
        }
    }
}

int Tree::cutCase(const PatternGraph &p) const {
    bool unConnectedCut = false, twoLongCut = false;
    for (VertexID i = 0; i < _numNodes; ++i) {
        std::vector<int> ccSize;
        std::vector<bool> visited2(_numVertices, true);
        for (int k = 0; k < _nodes[i].cutSize; ++k)
            visited2[_nodes[i].cut[k]] = false;
        std::queue<VertexID> q;
        for (int k = 0; k < _nodes[i].cutSize; ++k) {
            VertexID u = _nodes[i].cut[k];
            if (visited2[u]) continue;
            int num = 0;
            q.push(u);
            while (!q.empty()) {
                VertexID w = q.front();
                q.pop();
                visited2[w] = true;
                ++num;
                ui nbrCnt;
                VertexID *neighbors = p.getNeighbors(w, nbrCnt);
                for (int l = 0; l < nbrCnt; ++l) {
                    VertexID next = neighbors[l];
                    if (!visited2[next])
                        q.push(next);
                }
            }
            ccSize.push_back(num);
        }
        if (ccSize.size() > 1) unConnectedCut = true;
        std::vector<bool> visited(_numVertices, false);
        bool allowLongCut = true;
        for (ui j: _edges[i]) {
            bool cutIncrease = false;
            // add vertices shared by i and j to the cut of i
            VertexID *iVertices = _nodes[i].vertices, *jVertices = _nodes[j].vertices;
            int pi = 0, pj = 0, cutLength = 0;
            while (pi < _nodes[i].numVertices && pj < _nodes[j].numVertices) {
                VertexID ui = iVertices[pi], uj = jVertices[pj];
                if (ui == uj) {
                    if (!visited[ui]) {
                        cutIncrease = true;
                        visited[ui] = true;
                    }
                    ++pi;
                    ++pj;
                    ++cutLength;
                } else if (ui < uj) ++pi;
                else ++pj;
            }
            if (!allowLongCut && cutLength > 2 && cutIncrease) twoLongCut = true;
            if (cutLength > 2 && allowLongCut) allowLongCut = false;
        }
    }
    if (unConnectedCut && !twoLongCut) return 1;
    if (!unConnectedCut && twoLongCut) return 2;
    if (!unConnectedCut && !twoLongCut) return 3;
    else return 4;
}

std::vector<Tree> Tree::realTree(const Pattern &p, bool sign) {
    // rebuild supernodes
    int numPartition = (int)_partitionPos.size();
    std::vector<ui> partitionNumNodes(numPartition);
    for (int i = 0; i < numPartition; ++i) {
        if (i == 0) partitionNumNodes[i] = _partitionPos[i] + 1;
        else partitionNumNodes[i] = _partitionPos[i] - _partitionPos[i - 1];
    }
    for (int i = 0; i < numPartition; ++i) {
        ui numNodes = partitionNumNodes[i];
        if (numNodes == 1) continue;
        std::set<VertexID> cutUnion;
        int position;
        if (i == 0) position = 0;
        else position = _partitionPos[i - 1] + 1;
        for (int j = 0; j < numNodes; ++j) {
            VertexID nID = _postOrder[position + j];
            if (_nodes[nID].cutSize > 2 || (_nodes[nID].cutSize == 2 && !p.u.isEdge(_nodes[nID].cut[0], _nodes[nID].cut[1]))) {
                for (int k = 0; k < _nodes[nID].cutSize; ++k)
                    cutUnion.insert(_nodes[nID].cut[k]);
            }
        }
        std::vector<VertexID> partitionOrder(cutUnion.begin(), cutUnion.end());
        ui coreSize;
        VertexID *coreV = p.u.getCoreV(coreSize);
        if (!isConnectedSubgraph(partitionOrder, p.u)) {
            std::vector<VertexID> verticesToAdd;
            for (int j = 0; j < coreSize; ++j) {
                if (std::find(partitionOrder.begin(), partitionOrder.end(), coreV[j]) == partitionOrder.end())
                    verticesToAdd.push_back(coreV[j]);
            }
            for (int k = 1; k <= verticesToAdd.size(); ++k) {
                std::vector<std::vector<bool>> choices = chooseK(verticesToAdd.size(), k);
                bool flag = false;
                for (const auto &choice : choices) {
                    std::vector<VertexID> enhanced(partitionOrder);
                    for (int j = 0; j < verticesToAdd.size(); ++j) {
                        if (choice[j])
                            enhanced.push_back(verticesToAdd[j]);
                    }
                    if (isConnectedSubgraph(enhanced, p.u)) {
                        partitionOrder = enhanced;
                        flag = true;
                        break;
                    }
                    if (flag) break;
                }
                if (flag) break;
            }
        }
        for (int j = 0; j < numNodes; ++j) {
            VertexID nID = _postOrder[position + j];
            Node &tau = _nodes[nID];
            std::vector<VertexID> newVertices;
            for (VertexID u: partitionOrder) {
                if (std::find(tau.vertices, tau.vertices + tau.numVertices, u) == tau.vertices + tau.numVertices)
                    newVertices.push_back(u);
            }
            if (!newVertices.empty()) {
                std::vector<VertexID> oldVertices(tau.vertices, tau.vertices + tau.numVertices);
                delete[] tau.vertices;
                tau.numVertices += newVertices.size();
                tau.vertices = new VertexID[tau.numVertices];
                int p0 = 0;
                for (VertexID oldVertex : oldVertices) {
                    tau.vertices[p0] = oldVertex;
                    ++p0;
                }
                for (VertexID newVertex : newVertices) {
                    tau.id += 1 << newVertex;
                    tau.vertices[p0] = newVertex;
                    ++p0;
                }
                std::sort(tau.vertices, tau.vertices + tau.numVertices);
                if (!p.useDAG()) tau.getCanonLabel(p.u);
                else tau.getCanonLabel(p.out);
            }
        }
    }
    // remove duplicated supernodes
    std::vector<bool> notSubset(_numNodes, true);
    for (int i = 0; i < _numNodes; ++i) {
        for (int j = i + 1; j < _numNodes; ++j) {
            if (_nodes[i] < _nodes[j]) {
                notSubset[i] = false;
            }
        }
    }
    for (int i = 0; i < _numNodes; ++i) {
        if (!(notSubset[i])) --_numNodes;
    }
    std::vector<Node> newNodes;
    for (int i = 0; i < notSubset.size(); ++i) {
        if (notSubset[i] && _nodes[i].numVertices > 2) {
            newNodes.push_back(_nodes[i]);
        }
    }
    Tree newT = getAllTree(newNodes, p.u)[0];
    newT.addPeripheral(p);
    std::vector<Tree> result = newT.getRootedTrees(p.u, sign);
    for (Tree &t: result) {
        if (p.u.getNumVertices() <= 5)
            t.setNodeSubCanon(p);
    }
    return result;
}

void Tree::writeToFile(const std::string &filename) {
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile) {
        throw std::runtime_error("Unable to open file for writing");
    }

    // Write simple attributes
    outFile.write(reinterpret_cast<const char*>(&_numVertices), sizeof(_numVertices));
    outFile.write(reinterpret_cast<const char*>(&_multiFactor), sizeof(_multiFactor));
    outFile.write(reinterpret_cast<const char*>(&_numNodes), sizeof(_numNodes));
    outFile.write(reinterpret_cast<const char*>(&_numRules), sizeof(_numRules));
    outFile.write(reinterpret_cast<const char*>(&_orbitType), sizeof(_orbitType));
    outFile.write(reinterpret_cast<const char*>(&_treeWidth), sizeof(_treeWidth));
    outFile.write(reinterpret_cast<const char*>(&_sumWidth), sizeof(_sumWidth));
    outFile.write(reinterpret_cast<const char*>(&_maxNumSource), sizeof(_maxNumSource));
    outFile.write(reinterpret_cast<const char*>(&_fhw), sizeof(_fhw));
    outFile.write(reinterpret_cast<const char*>(&_executeMode), sizeof(_executeMode));

    // Write Node array
    for (ui i = 0; i < _numNodes; ++i) {
        _nodes[i].writeToStream(outFile); // Assuming Node::writeToStream is implemented
    }

    // Write vectors and other complex attributes
    for (ui i = 0; i < _numVertices; ++i) {
        writeVectorToStream(outFile, _v2n[i]);
    }
    write2DVectorToStream(outFile, _edges);
    writeArrayToStream(outFile, _parent, _numNodes);
    write2DVectorToStream(outFile, _child);
    write2DVectorToStream(outFile, _greaterRules);
    write2DVectorToStream(outFile, _lessRules);
    writeVectorToStream(outFile, _aggreV);
    writeVectorToStream(outFile, _aggreWeight);
    writeVectorToStream(outFile, _postOrder);
    writeVectorToStream(outFile, _partitionPos);
    write3DVectorToStream(outFile, _nodesAtStep);
    write2DVectorToStream(outFile, _prefixPos);
    write2DVectorToStream(outFile, _globalOrder);
    write2DVectorToStream(outFile, _aggrePos);
    write2DVectorToStream(outFile, _partitionInterPos);
    write2DVectorToStream(outFile, _nodeInterPos);
    write3DVectorToStream(outFile, _nodeInPos);
    write3DVectorToStream(outFile, _nodeOutPos);
    write3DVectorToStream(outFile, _nodeUnPos);
    write3DVectorToStream(outFile, _nodeGreaterPos);
    write3DVectorToStream(outFile, _nodeLessPos);
    write3DVectorToStream(outFile, _partitionInPos);
    write3DVectorToStream(outFile, _partitionOutPos);
    write3DVectorToStream(outFile, _partitionUnPos);
    write3DVectorToStream(outFile, _partitionGreaterPos);
    write3DVectorToStream(outFile, _partitionLessPos);
    write3DVectorToStream(outFile, _childKeyPos);
    write3DVectorToStream(outFile, _posChildEdge);
    write3DVectorToStream(outFile, _posAggreEdge);
    write2DVectorToStream(outFile, _childEdgeType);
    write2DVectorToStream(outFile, _aggreEdgeType);
    write2DVectorToStream(outFile, _nodeCandPos);
    write2DVectorPairToStream(outFile, _nodeTriPos);
    write2DVectorToStream(outFile, _triEdgeType);
    write2DVectorToStream(outFile, _triEndType);
    write2DVectorToStream(outFile, _partitionCandPos);
    write2DVectorPairToStream(outFile, _partitionTriPos);
    write2DVectorToStream(outFile, _partitionEdgeType);
    write2DVectorToStream(outFile, _partitionEndType);

    outFile.close();
}

void Tree::readFromFile(const std::string &filename) {
    std::ifstream inFile(filename, std::ios::binary);
    if (!inFile) {
        throw std::runtime_error("Unable to open file for reading");
    }

    // Read simple attributes
    inFile.read(reinterpret_cast<char*>(&_numVertices), sizeof(_numVertices));
    inFile.read(reinterpret_cast<char*>(&_multiFactor), sizeof(_multiFactor));
    inFile.read(reinterpret_cast<char*>(&_numNodes), sizeof(_numNodes));
    inFile.read(reinterpret_cast<char*>(&_numRules), sizeof(_numRules));
    inFile.read(reinterpret_cast<char*>(&_orbitType), sizeof(_orbitType));
    inFile.read(reinterpret_cast<char*>(&_treeWidth), sizeof(_treeWidth));
    inFile.read(reinterpret_cast<char*>(&_sumWidth), sizeof(_sumWidth));
    inFile.read(reinterpret_cast<char*>(&_maxNumSource), sizeof(_maxNumSource));
    inFile.read(reinterpret_cast<char*>(&_fhw), sizeof(_fhw));
    inFile.read(reinterpret_cast<char*>(&_executeMode), sizeof(_executeMode));

    // Allocate and read Node array
    _nodes = new Node[_numNodes];
    for (ui i = 0; i < _numNodes; ++i) {
        _nodes[i].readFromStream(inFile); // Assuming Node::readFromStream is implemented
    }
    _v2n = new std::vector<VertexID>[_numVertices];
    for (ui i = 0; i < _numVertices; ++i) {
        readVectorFromStream(inFile, _v2n[i]);
    }
    // Read vectors and other complex attributes
    read2DVectorFromStream(inFile, _edges);
    readArrayFromStream(inFile, _parent, _numNodes);
    read2DVectorFromStream(inFile, _child);
    read2DVectorFromStream(inFile, _greaterRules);
    read2DVectorFromStream(inFile, _lessRules);
    readVectorFromStream(inFile, _aggreV);
    readVectorFromStream(inFile, _aggreWeight);
    readVectorFromStream(inFile, _postOrder);
    readVectorFromStream(inFile, _partitionPos);
    read3DVectorFromStream(inFile, _nodesAtStep);
    read2DVectorFromStream(inFile, _prefixPos);
    read2DVectorFromStream(inFile, _globalOrder);
    read2DVectorFromStream(inFile, _aggrePos);
    read2DVectorFromStream(inFile, _partitionInterPos);
    read2DVectorFromStream(inFile, _nodeInterPos);
    read3DVectorFromStream(inFile, _nodeInPos);
    read3DVectorFromStream(inFile, _nodeOutPos);
    read3DVectorFromStream(inFile, _nodeUnPos);
    read3DVectorFromStream(inFile, _nodeGreaterPos);
    read3DVectorFromStream(inFile, _nodeLessPos);
    read3DVectorFromStream(inFile, _partitionInPos);
    read3DVectorFromStream(inFile, _partitionOutPos);
    read3DVectorFromStream(inFile, _partitionUnPos);
    read3DVectorFromStream(inFile, _partitionGreaterPos);
    read3DVectorFromStream(inFile, _partitionLessPos);
    read3DVectorFromStream(inFile, _childKeyPos);
    read3DVectorFromStream(inFile, _posChildEdge);
    read3DVectorFromStream(inFile, _posAggreEdge);
    read2DVectorFromStream(inFile, _childEdgeType);
    read2DVectorFromStream(inFile, _aggreEdgeType);
    read2DVectorFromStream(inFile, _nodeCandPos);
    read2DVectorPairFromStream(inFile, _nodeTriPos);
    read2DVectorFromStream(inFile, _triEdgeType);
    read2DVectorFromStream(inFile, _triEndType);
    read2DVectorFromStream(inFile, _partitionCandPos);
    read2DVectorPairFromStream(inFile, _partitionTriPos);
    read2DVectorFromStream(inFile, _partitionEdgeType);
    read2DVectorFromStream(inFile, _partitionEndType);

    inFile.close();
}

void Tree::wheel9Tree() {
    _multiFactor = 1;
    _numNodes = 6;
    _numVertices = 9;
    _nodes = new Node[_numNodes];
    VertexID vertices[6][4] = {{0, 1, 2, 3}, {0, 1, 3, 8}, {0, 3, 4, 8}, {0, 4, 7, 8}, {0, 4, 5, 7}, {0, 5, 6, 7}};
    _v2n = new std::vector<ui>[_numVertices];
    _edges = std::vector<std::vector<VertexID>>(_numNodes);
    _child = std::vector<std::vector<VertexID>>(_numNodes);
    _numRules = 0;
    _greaterRules = std::vector<std::vector<VertexID>>(_numVertices);
    _lessRules = std::vector<std::vector<VertexID>>(_numVertices);
    _parent = new VertexID[_numNodes];
    _parent[0] = 99;
    _edges[0].push_back(1);
    for (VertexID nID = 0; nID < _numNodes; ++nID) {
        _nodes[nID].numVertices = 4;
        _nodes[nID].vertices = vertices[nID];
        for (int i = 0; i < 4; ++i) {
            _nodes[nID].id += 1 << vertices[nID][i];
        }
        if (nID > 0) {
            _edges[nID].push_back(nID - 1);
            _edges[nID].push_back(nID + 1);
            _parent[nID] = nID - 1;
        }
        if (nID < 5) _child[nID].push_back(nID + 1);
    }
    _orbitType = 1;
    _treeWidth = 4;
    _sumWidth = 24;
    _maxNumSource = 0;
    _fhw = 2.0;
    _executeMode = true;
}

ConNode::ConNode(const PatternGraph &p, const Tree &t) {
    num = 0;
    numRules = 0;
    if (t.getNumNodes() < 2 || !t.getExecuteMode()) return;
    const Node &tau = t.getNode(0);
    VertexID *cut = tau.cut;
    ui cutSize = tau.cutSize;
    if (cutSize != tau.numVertices - 1) return;
    canonValue = tau.canonValue;
    for (int i = 0; i < tau.numVertices; ++i) {
        bool exists = false;
        VertexID u = tau.vertices[i];
        for (int j = 0; j < cutSize; ++j) {
            if (cut[j] == u) {
                exists = true;
                break;
            }
        }
        if (!exists) {
            uIn = u;
            break;
        }
    }
    for (VertexID nID = 1; nID < t.getNumNodes(); ++nID) {
        const Node &tau2 = t.getNode(nID);
        if (tau2.canonValue != canonValue || tau2.cutSize != cutSize) return;
        VertexID *cut2 = tau2.cut;
        for (int i = 0; i < tau2.cutSize; ++i) {
            if (cut[i] != cut2[i]) return;
        }
        // check whether the internal vertex connects to the same cut vertices
        VertexID internalU;
        for (int i = 0; i < tau2.numVertices; ++i) {
            bool exists = false;
            VertexID u = tau2.vertices[i];
            for (int j = 0; j < cutSize; ++j) {
                if (cut[j] == u) {
                    exists = true;
                    break;
                }
            }
            if (!exists) {
                internalU = u;
                break;
            }
        }
        for (int i = 0; i < cutSize; ++i) {
            if (p.isEdge(cut[i], uIn) != p.isEdge(cut[i], internalU))
                return;
        }
    }
    num = t.getNumNodes();
    divideFactor = 1;
    orbitType = p.getOrbitType();
    edgeKey = false;
    posChildEdge = 0;
    childEdgeType = 0;
    cutVertices.assign(tau.cut, tau.cut + tau.cutSize);
}

// call when aggre pos changes
void ConNode::setAggrePoses(const PatternGraph &p) {
    if (orbitType != 2) {
        std::sort(aggrePos.begin(), aggrePos.end());
        posAggreEdge = std::vector<std::vector<int>>(nodeOrder.size());
        aggreType = std::vector<bool>(aggrePos.size(), false);
        for (int i = 0; i < aggrePos.size(); ++i) {
            if (nodeOrder[aggrePos[i]] == uIn) aggreType[i] = false;
            else aggreType[i] = true;
        }
    }
    else {
        sortEOrbitAggrePos(aggrePos);
        aggreType = std::vector<bool>(aggrePos.size() / 2, false);
        for (int i = 0; i < aggrePos.size(); i = i + 2) {
            int u1 = aggrePos[i], u2 = aggrePos[i + 1];
            if (u1 == uIn || u2 == uIn) aggreType[aggrePos[i / 2]] = false;
            else aggreType[aggrePos[i / 2]] = true;
        }
        posAggreEdge = std::vector<std::vector<int>>(nodeOrder.size());
        aggreEdgeType = std::vector<int>(aggrePos.size() / 2);
        if (p.isEOrbitDir()) {
            for (int i = 0; i < aggrePos.size(); i = i + 2) {
                int pos1 = aggrePos[i], pos2 = childKeyPos[i + 1];
                int pos = pos1 > pos2 ? pos1 : pos2;
                posAggreEdge[pos].push_back(i / 2);
                if (!interPos[pos]) {
                    if (!nodeOutPos[pos].empty()) aggreEdgeType[i / 2] = 1;
                    else if (!nodeInPos[pos].empty()) aggreEdgeType[i / 2] = 2;
                    else aggreEdgeType[i / 2] = 5;
                }
                else aggreEdgeType[i / 2] = 3;
            }
        }
        else {
            for (int i = 0; i < aggrePos.size(); i = i + 2) {
                int pos1 = aggrePos[i], pos2 = childKeyPos[i + 1];
                int pos = pos1 > pos2 ? pos1 : pos2;
                posAggreEdge[pos].push_back(i / 2);
                if (!interPos[pos] && !nodeUnPos[pos].empty()) aggreEdgeType[i / 2] = 1;
                else {
                    aggreEdgeType[i / 2] = 4;
                }
            }
        }
    }
}

void ConNode::initPoses(const std::vector<VertexID> &pOrder, const std::vector<VertexID> &localOrder,
                        std::vector<std::vector<VertexID>> &greaterRules, std::vector<std::vector<VertexID>> &lessRules,
                        const PatternGraph &p, bool useTriangle) {
    prefixOrder = pOrder;
    nodeOrder = prefixOrder;
    for (VertexID u: localOrder)
        nodeOrder.push_back(u);
    ui aggreSize;
    VertexID *aggreV = p.getMultiAggreV(aggreSize);
    if (orbitType == 1) {
        for (int i = 0; i < aggreSize; ++i) {
            VertexID u = aggreV[i];
            for (int j = 0; j < nodeOrder.size(); ++j) {
                if (nodeOrder[j] == u) {
                    aggrePos.push_back(j);
                    break;
                }
            }
        }
    }
    if (orbitType == 2) {
        for (int i = 0; i < aggreSize; i = i + 2) {
            VertexID u1 = aggreV[i], u2 = aggreV[i + 1];
            int cnt = 0;
            for (int j = 0; j < nodeOrder.size(); ++j) {
                if (nodeOrder[j] == u1 || nodeOrder[j] == u2) {
                    aggrePos.push_back(j);
                    ++cnt;
                }
            }
            if (cnt == 1) aggrePos.pop_back();
        }
    }
    setAggrePoses(p);
    // build child key pos
    for (VertexID u: cutVertices) {
        if (std::find(prefixOrder.begin(), prefixOrder.end(), u) == prefixOrder.end())
            for (int i = 0; i < nodeOrder.size(); ++i) {
                if (nodeOrder[i] == u) {
                    childKeyPos.push_back(i);
                    break;
                }
            }
    }
    if (childKeyPos.empty()) childKeyPos.push_back(0);
    if (orbitType == 2 || childKeyPos.size() == 2) edgeKey = true;
    int factorial = 1;
    for (int i = 1; i <= num; ++i)
        factorial *= i;
    divideFactor = p.getAutoSize() / factorial / divideFactor;
    if (orbitType == 0) hashTables = std::vector<HashTable>(1, nullptr);
    else if (orbitType == 1) hashTables = std::vector<HashTable>(aggrePos.size(), nullptr);
    else hashTables = std::vector<HashTable>(aggrePos.size() / 2, nullptr);
    nodeInPos = std::vector<std::vector<int>>(nodeOrder.size());
    nodeOutPos = std::vector<std::vector<int>>(nodeOrder.size());
    nodeUnPos = std::vector<std::vector<int>>(nodeOrder.size());
    interPos = std::vector<bool>(nodeOrder.size(), false);
    candPos = std::vector<bool>(nodeOrder.size(), false);
    triPos = std::vector<std::pair<int, int>>(nodeOrder.size());
    triEdgeType = std::vector<int>(nodeOrder.size());
    triEndType = std::vector<int>(nodeOrder.size());
    // some symmetry rules are erased because they are implemented by edge direction
    std::vector<std::vector<VertexID>> erasedGreaterRules(greaterRules.size());
    std::vector<std::vector<VertexID>> erasedLessRules(lessRules.size());
    std::vector<int> numNeighbor(nodeOrder.size(), 0);
    for (int i = 0; i < nodeOrder.size(); ++i) {
        for (int j = 0; j < i; ++j) {
            if (p.isEdge(nodeOrder[i], nodeOrder[j])) {
                ++numNeighbor[i];
                if (std::find(greaterRules[nodeOrder[i]].begin(), greaterRules[nodeOrder[i]].end(), nodeOrder[j])
                    != greaterRules[nodeOrder[i]].end()) {
                    nodeOutPos[i].push_back(j);
                }
                else if (std::find(lessRules[nodeOrder[i]].begin(), lessRules[nodeOrder[i]].end(), nodeOrder[j])
                         != lessRules[nodeOrder[i]].end()) {
                    nodeInPos[i].push_back(j);
                }
                else nodeUnPos[i].push_back(j);
            }
        }
    }
    for (int i = 0; i < nodeOrder.size(); ++i) {
        if (numNeighbor[i] > 1)
            interPos[i] = true;
    }
    candPos = interPos;
    if (childKeyPos.size() == 2) {
        int pos1 = childKeyPos[0], pos2 = childKeyPos[1];
        posChildEdge = pos1 > pos2 ? pos1 : pos2;
        int smallerPos = pos1 < pos2 ? pos1 : pos2;
        if (!interPos[posChildEdge]) {
            if (!nodeInPos[posChildEdge].empty()) childEdgeType = 2;
            else childEdgeType = 1;
        }
        else {
            if (std::find(nodeUnPos[posChildEdge].begin(), nodeUnPos[posChildEdge].end(), smallerPos) == nodeUnPos[posChildEdge].end())
                childEdgeType = 3;
            else {
                childEdgeType = 4;
            }
        }
    }
    // build the triangle structures
    for (int i = 0; i < nodeOrder.size(); ++i) {
        std::vector<int> poses = nodeInPos[i];
        for (int pos: nodeOutPos[i]) poses.push_back(pos);
        for (int pos: nodeUnPos[i]) poses.push_back(pos);
        std::sort(poses.begin(), poses.end());
        for (int j = 1; j < poses.size(); ++j) {
            int p2 = poses[j];
            bool flag = false;
            for (int k = 0; k < j; ++k) {
                int p1 = poses[k];
                VertexID u1 = nodeOrder[p1], u2 = nodeOrder[p2], u3 = nodeOrder[i];
                if (p.isEdge(u1, u2)) {
                    flag = true;
                    if (!interPos[p2]) {
                        if (std::find(lessRules[u2].begin(), lessRules[u2].end(),
                                      u1) != lessRules[u2].end()) {
                            triEdgeType[i] = 2;
                        }
                        else if (std::find(greaterRules[u2].begin(), greaterRules[u2].end(),
                                           u1) != greaterRules[u2].end())
                            triEdgeType[i] = 1;
                        else
                            triEdgeType[i] = 5;
                    }
                    else {
                        triEdgeType[i] = 3;
                    }
                    // if undirected pattern, check whether (i,p1), (i,p2) can be directed.
                    // if so, remove the corresponding symmetry rule.
                    bool u1u3 = std::find(lessRules[u1].begin(), lessRules[u1].end(),
                                          u3) != lessRules[u1].end();
                    bool u3u1 = std::find(greaterRules[u1].begin(), greaterRules[u1].end(),
                                          u3) != greaterRules[u1].end();
                    bool u2u3 = std::find(lessRules[u2].begin(), lessRules[u2].end(),
                                          u3) != lessRules[u2].end();
                    bool u3u2 = std::find(greaterRules[u2].begin(), greaterRules[u2].end(),
                                          u3) != greaterRules[u2].end();
                    if (u1u3 && u2u3) {
                        triEndType[i] = 3;
                        erasedLessRules[u1].push_back(u3);
                        erasedLessRules[u2].push_back(u3);
                        erasedGreaterRules[u3].push_back(u1);
                        erasedGreaterRules[u3].push_back(u2);
                    }
                    else if (u3u1 && u3u2) {
                        triEndType[i] = 1;
                        erasedLessRules[u3].push_back(u1);
                        erasedLessRules[u3].push_back(u2);
                        erasedGreaterRules[u1].push_back(u3);
                        erasedGreaterRules[u2].push_back(u3);
                    }
                    else if (u1u3 && u3u2) {
                        triEndType[i] = 2;
                        erasedLessRules[u1].push_back(u3);
                        erasedLessRules[u3].push_back(u2);
                        erasedGreaterRules[u3].push_back(u1);
                        erasedGreaterRules[u2].push_back(u3);
                    }
                    else if (u3u1 && u2u3) {
                        triEndType[i] = 2;
                        erasedLessRules[u3].push_back(u1);
                        erasedLessRules[u2].push_back(u3);
                        erasedGreaterRules[u1].push_back(u3);
                        erasedGreaterRules[u3].push_back(u2);
                    }
                    else {
                        triEndType[i] = 4;
                    }
                    triPos[i]= std::make_pair(p1, p2);
                }
                if (flag) break;
            }
            if (flag) break;
        }
        // reset the nodeInPos[i], _nodeOutPos and _nodeCandPos : remove positions in triPos[i]
        if (useTriangle && interPos[i] && !(triPos[i].first == triPos[i].second)) {
            std::vector<int> inCopy = nodeInPos[i];
            nodeInPos[i].clear();
            for (int pos: inCopy) {
                if (pos != triPos[i].first && pos != triPos[i].second)
                    nodeInPos[i].push_back(pos);
            }
            std::vector<int> outCopy = nodeOutPos[i];
            nodeOutPos[i].clear();
            for (int pos: outCopy) {
                if (pos != triPos[i].first && pos != triPos[i].second)
                    nodeOutPos[i].push_back(pos);
            }
            std::vector<int> unCopy = nodeUnPos[i];
            nodeUnPos[i].clear();
            for (int pos: unCopy) {
                if (pos != triPos[i].first && pos != triPos[i].second)
                    nodeUnPos[i].push_back(pos);
            }
            if ((nodeInPos[i].empty() && nodeOutPos[i].empty()) && nodeUnPos[i].empty()) candPos[i] = false;
        }
        // if undirected pattern, check whether (i,p1), (i,p2) can be directed.
        // if so, remove the corresponding symmetry rule.
        for (int pos: nodeOutPos[i]) {
            erasedLessRules[nodeOrder[pos]].push_back(nodeOrder[i]);
            erasedGreaterRules[nodeOrder[i]].push_back(nodeOrder[pos]);
        }
        for (int pos: nodeInPos[i]) {
            erasedGreaterRules[nodeOrder[pos]].push_back(nodeOrder[i]);
            erasedLessRules[nodeOrder[i]].push_back(nodeOrder[pos]);
        }
    }
    // update symmetry rules and nodeGreaterPos, nodeLessPos
    for (VertexID i = 0; i < p.getNumVertices(); ++i) {
        std::sort(greaterRules[i].begin(), greaterRules[i].end());
        std::sort(lessRules[i].begin(), lessRules[i].end());
        std::sort(erasedGreaterRules[i].begin(), erasedGreaterRules[i].end());
        std::sort(erasedLessRules[i].begin(), erasedLessRules[i].end());
        std::vector<VertexID> greaterCopy = greaterRules[i];
        std::vector<VertexID> lessCopy = lessRules[i];
        greaterRules[i].clear();
        lessRules[i].clear();
        std::set_difference(greaterCopy.begin(), greaterCopy.end(), erasedGreaterRules[i].begin(),
                            erasedGreaterRules[i].end(), std::inserter(greaterRules[i], greaterRules[i].begin()));
        std::set_difference(lessCopy.begin(), lessCopy.end(), erasedLessRules[i].begin(),
                            erasedLessRules[i].end(), std::inserter(lessRules[i], lessRules[i].begin()));
    }
    greaterPos = std::vector<std::vector<int>>(nodeOrder.size());
    lessPos = std::vector<std::vector<int>>(nodeOrder.size());
    for (int i = 0; i < nodeOrder.size(); ++i) {
        std::vector<VertexID> greaterVertices = greaterRules[nodeOrder[i]];
        std::vector<VertexID> lessVertices = lessRules[nodeOrder[i]];
        for (int j = 0; j < i; ++j) {
            if (std::find(greaterVertices.begin(), greaterVertices.end(), nodeOrder[j]) != greaterVertices.end())
                greaterPos[i].push_back(j);
            if (std::find(lessVertices.begin(), lessVertices.end(), nodeOrder[j]) != lessVertices.end())
                lessPos[i].push_back(j);
        }
    }
    delete[] aggreV;
}

void ConNode::print() const {
    printf("number of nodes compressed: %d\n", num);
    printf("inner vertex: %d\n", uIn);
    printf("node order: ");
    for (VertexID u: nodeOrder)
        printf("%d, ", u);
    printf("\n");
    printf("aggre poses: ");
    if (orbitType == 1) {
        for (int i = 0; i < aggrePos.size(); ++i) {
            bool type = aggreType[i];
            if (type) printf("id : %d, type : 1 ", aggrePos[i]);
            else printf("id : %d, type : 0 ", aggrePos[i]);
        }
    }
    if (orbitType == 2) {
        for (int i = 0; i < aggrePos.size(); ++i) {
            bool type = aggreType[i / 2];
            VertexID u1 = aggrePos[i], u2 = aggrePos[i + 1];
            if (type) printf("id : %d, %d, type : 1 ", u1, u2);
            else printf("id : %d, %d, type : 0 ", u1, u2);
        }
    }
    printf("\n");
    printf("divide factor : %d\n", divideFactor);
}

void ConNode::writeToStream(std::ofstream &outFile) {
    // Write simple types
    outFile.write(reinterpret_cast<const char*>(&num), sizeof(num));
    outFile.write(reinterpret_cast<const char*>(&numRules), sizeof(numRules));
    outFile.write(reinterpret_cast<const char*>(&divideFactor), sizeof(divideFactor));
    outFile.write(reinterpret_cast<const char*>(&edgeKey), sizeof(edgeKey));
    outFile.write(reinterpret_cast<const char*>(&uIn), sizeof(uIn));
    outFile.write(reinterpret_cast<const char*>(&canonValue), sizeof(canonValue));
    outFile.write(reinterpret_cast<const char*>(&orbitType), sizeof(orbitType));
    outFile.write(reinterpret_cast<const char*>(&posChildEdge), sizeof(posChildEdge));
    outFile.write(reinterpret_cast<const char*>(&childEdgeType), sizeof(childEdgeType));

    // Write vectors
    writeVectorToStream(outFile, cutVertices);
    writeVectorToStream(outFile, nodeOrder);
    writeVectorToStream(outFile, prefixOrder);
    writeVectorToStream(outFile, childKeyPos);
    writeVectorToStream(outFile, aggrePos);
    writeVectorToStream(outFile, aggreType);
    write2DVectorToStream(outFile, nodeInPos);
    write2DVectorToStream(outFile, nodeOutPos);
    write2DVectorToStream(outFile, nodeUnPos);
    write2DVectorToStream(outFile, greaterPos);
    write2DVectorToStream(outFile, lessPos);
    writeVectorToStream(outFile, interPos);
    writeVectorToStream(outFile, candPos);
    writeVectorPairToStream(outFile, triPos); // Special function for vector of pairs
    write2DVectorToStream(outFile, posAggreEdge);
    writeVectorToStream(outFile, aggreEdgeType);
    writeVectorToStream(outFile, triEdgeType);
    writeVectorToStream(outFile, triEndType);
}

void ConNode::readFromStream(std::ifstream &inFile) {
    // Read simple types
    inFile.read(reinterpret_cast<char*>(&num), sizeof(num));
    inFile.read(reinterpret_cast<char*>(&numRules), sizeof(numRules));
    inFile.read(reinterpret_cast<char*>(&divideFactor), sizeof(divideFactor));
    inFile.read(reinterpret_cast<char*>(&edgeKey), sizeof(edgeKey));
    inFile.read(reinterpret_cast<char*>(&uIn), sizeof(uIn));
    inFile.read(reinterpret_cast<char*>(&canonValue), sizeof(canonValue));
    inFile.read(reinterpret_cast<char*>(&orbitType), sizeof(orbitType));
    inFile.read(reinterpret_cast<char*>(&posChildEdge), sizeof(posChildEdge));
    inFile.read(reinterpret_cast<char*>(&childEdgeType), sizeof(childEdgeType));

    // Read vectors
    readVectorFromStream(inFile, cutVertices);
    readVectorFromStream(inFile, nodeOrder);
    readVectorFromStream(inFile, prefixOrder);
    readVectorFromStream(inFile, childKeyPos);
    readVectorFromStream(inFile, aggrePos);
    readVectorFromStream(inFile, aggreType);
    read2DVectorFromStream(inFile, nodeInPos);
    read2DVectorFromStream(inFile, nodeOutPos);
    read2DVectorFromStream(inFile, nodeUnPos);
    read2DVectorFromStream(inFile, greaterPos);
    read2DVectorFromStream(inFile, lessPos);
    readVectorFromStream(inFile, interPos);
    readVectorFromStream(inFile, candPos);
    readVectorPairFromStream(inFile, triPos); // Special function for vector of pairs
    read2DVectorFromStream(inFile, posAggreEdge);
    readVectorFromStream(inFile, aggreEdgeType);
    readVectorFromStream(inFile, triEdgeType);
    readVectorFromStream(inFile, triEndType);
}

// get all weakly connected node-induced subgraphs of p
// those subgraphs are candidate tree nodes.
// nodes in the result are ordered by the treewidth

std::vector<Node> getAllNodes(const PatternGraph &p) {
    std::vector<Node> nodes;
    // fill the initial q with cliques
    std::queue<Node> q = findMaximalCliques(p);
    ui coreSize;
    ui maxNodeID = 1 << p.getNumVertices();
    bool *visited = new bool[maxNodeID];
    memset(visited, false, sizeof(bool) * maxNodeID);
    VertexID *coreV = p.getCoreV(coreSize);
    // get core graph edges and push them into q.
    ui numEdge = 0;
    Edge* edgeList = p.coreUndirectedEdges(numEdge);
//    for (int i = 0; i < numEdge; ++i) {
//        VertexID u1 = edgeList[i].first, u2 = edgeList[i].second;
//        int id = (1 << u1) + (1 << u2);
//        visited[id] = true;
//        VertexID *vertices = new VertexID[2];
//        vertices[0] = u1, vertices[1] = u2;
//        q.emplace(id, vertices, 2, 0);
//    }

    while (!q.empty()) {
        Node tau = q.front();
        q.pop();
        ui tauSize = tau.numVertices;
        int id = tau.id;
        VertexID *vertices = tau.vertices;
        // we are only interested in nodes whose width >= 3
        // nodes whose width == 2 are in the non-core part
        if (tauSize > 2) {
            visited[id] = true;
            nodes.push_back(tau);
        }
        for (int i = 0; i < coreSize; ++i) {
            VertexID u1 = coreV[i];
            if (tau.hasVertex(u1)) continue;
            int newID = id + (1 << u1);
            if (visited[newID]) continue;
            // loop over vertices in tau, check whether u1 connects to one of them
            for (int j = 0; j < tauSize; ++j) {
                VertexID u2 = vertices[j];
                if (p.isEdge(u1, u2)) {
                    visited[newID] = true;
                    // vertices + u2 is weakly connected, thus a new node
                    // use insertion sort to keep newV sorted in ascending order
                    VertexID *newV = new VertexID[tauSize + 1];
                    int k;
                    for (k = 0; k < tauSize; ++k) {
                        if (vertices[k] < u1)
                            newV[k] = vertices[k];
                        else
                            break;
                    }
                    newV[k] = u1;
                    for (; k < tauSize; ++k)
                        newV[k + 1] = vertices[k];
                    q.emplace(newID, newV, tauSize + 1, 0);
                    break;
                }
            }
        }
    }

    delete[] edgeList;
    delete[] visited;
    return nodes;
}

// candidate nodes: 1. cover some uncovered vertices 2. not a supergraph of existing nodes
std::vector<Node> getCandidateNodes(const Tree &t, const std::vector<Node> &allNodes, const PatternGraph &p) {
    std::vector<Node> result;
    std::vector<VertexID> unCovered = t.getUncovered(p);
    for (const auto& tau: allNodes) {
        // if tau is neither a supernode nor a subnode of some nodes in t,
        // then tau can be added to tree t
        if (t.hasSubNodeOf(tau) || t.hasSupNodeOf(tau)) continue;
        for (auto u: unCovered)
            if (tau.hasVertex(u)) {
                result.push_back(tau);
                break;
            }
    }
    return result;
}

void findCliquesRecursive(const PatternGraph &graph,
                          std::vector<VertexID> &currentClique,
                          std::vector<VertexID> &potentialClique,
                          std::vector<VertexID> &processedVertices,
                          std::queue<Node> &cliques) {
    if (potentialClique.empty() && processedVertices.empty()) {
        Node node;
        node.numVertices = currentClique.size();
        node.vertices = new VertexID[node.numVertices];
        for (int i = 0; i < currentClique.size(); ++i) {
            node.vertices[i] = currentClique[i];
            node.id += 1 << currentClique[i];
        }
        std::copy(currentClique.begin(), currentClique.end(), node.vertices);
        cliques.push(node);
        return;
    }
    for (int i = 0; i < potentialClique.size(); ++i) {
        std::vector<VertexID> newCurrentClique = currentClique;
        VertexID newVertex = potentialClique[i];
        newCurrentClique.push_back(newVertex);
        std::vector<VertexID> newPotentialClique, newProcessedClique;
        // refine potentialClique and processedVertices
        for (auto & u: potentialClique) {
            if (u > newVertex && graph.isEdge(u, newVertex))
                newPotentialClique.push_back(u);
        }
        for (auto & u: processedVertices) {
            if (graph.isEdge(u, newVertex))
                newProcessedClique.push_back(u);
        }
        findCliquesRecursive(graph, newCurrentClique, newPotentialClique, newProcessedClique, cliques);
        processedVertices.push_back(newVertex);
    }
}

std::queue<Node> findMaximalCliques(const PatternGraph &graph) {
    std::queue<Node> cliques;
    std::vector<VertexID> currentClique, potentialClique(graph.getNumVertices()), processedVertices;

    // Initialize potentialClique with all vertices
    for (ui i = 0; i < graph.getNumVertices(); ++i) {
        potentialClique[i] = i;
    }

    findCliquesRecursive(graph, currentClique, potentialClique, processedVertices, cliques);
    return cliques;
}

std::vector<Tree> getAllTree(const PatternGraph &p) {
#ifdef ONLY_PLAN
    auto start = std::chrono::steady_clock::now();
#endif

    std::vector<Node> allNodes = getAllNodes(p);

#ifdef ONLY_PLAN
    auto result = getAllTree(allNodes, p);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsedSeconds = end - start;
    gTDTime += elapsedSeconds.count();
    return result;
#else
    return getAllTree(allNodes, p);
#endif
}


std::vector<Tree> getAllTree(const std::vector<Node> &allNode, const PatternGraph &p) {
    std::vector<Tree> trees;
    std::queue<Tree> q;
    std::vector<std::set<int>> treeID;
    Pattern pt(p);
    q.emplace(p.getNumVertices());
    while (!q.empty()) {
        Tree t = q.front();
        q.pop();
        if (t.isValid(p)) {
            std::set<int> nodeIDs;
            for (VertexID nID = 0; nID < t.getNumNodes(); ++nID) {
                nodeIDs.insert(t.getNode(nID).id);
            }
            bool idExists = false;
            for (int i = 0; i < treeID.size(); ++i) {
                if (treeID[i] == nodeIDs) {
                    idExists = true;
                    break;
                }
            }
            if (!idExists) {
                trees.push_back(t);
                treeID.push_back(nodeIDs);
                // early termination
                if (p.getNumVertices() > 8 && trees.size() >= 500) return trees;
            }
        }
        else {
            std::vector<Node> candidateNodes = getCandidateNodes(t, allNode, p);
            for (auto tau: candidateNodes) {
                std::vector<Tree> newTrees = t.addNode(tau);
                for (const auto& newT: newTrees) {
                    q.push(newT);
                }
            }
        }
    }

    return trees;
}

void fractionalWidth(struct Node &tau, const PatternGraph &p, const std::vector<VertexID> &pathToTau) {
    glp_term_out(GLP_OFF);

    ui n = 0, m = 0;
    Edge e[100];
    if (!pathToTau.empty()) {
        n = pathToTau.size() + tau.localOrder.size();
        // edges in the path
        for (ui i = 0; i < pathToTau.size(); ++i) {
            for (ui j = i + 1; j < pathToTau.size(); ++j)
                if (p.isEdge(pathToTau[i], pathToTau[j]))
                    e[m++] = std::make_pair(i, j);
        }
        // edges in local order
        for (ui i = pathToTau.size(); i < n; ++i) {
            for (ui j = i + 1; j < n; ++j)
                if (p.isEdge(tau.localOrder[i - pathToTau.size()], tau.localOrder[j - pathToTau.size()]))
                    e[m++] = std::make_pair(i, j);
        }
        // edges between the path and local order
        for (ui i = 0; i < pathToTau.size(); ++i) {
            for (ui j = pathToTau.size(); j < n; ++j)
                if (p.isEdge(pathToTau[i], tau.localOrder[j - pathToTau.size()]))
                    e[m++] = std::make_pair(i, j);
        }
    }
    else {
        n = tau.numVertices;
        for(ui i = 0; i < n; ++i)
            for(ui j = i + 1; j < n; ++j)
                if(p.isEdge(tau.vertices[i], tau.vertices[j]))
                    e[m++] = std::make_pair(i, j);
    }

    int ia[n * (n + m) + 1], ja[n * (n + m) + 1];
    double ar[n * (n + m) + 1], z;

    glp_prob *lp;

    lp = glp_create_prob();
    glp_set_obj_dir(lp, GLP_MIN);
    glp_add_rows(lp, n);

    for(size_t i = 0; i < n; ++i)
        glp_set_row_bnds(lp, i + 1, GLP_LO, 1., 1e8);

    glp_add_cols(lp, m + n);
    for(size_t i = 0; i < n + m; ++i) {
        glp_set_col_bnds(lp, i + 1, GLP_LO, 0., 0.);
        glp_set_obj_coef(lp, i + 1, 1.);
    }

    int cnt = 0;

    for(size_t i = 0; i < n; ++i)
        for(size_t j = 0; j < n; ++j)
            ia[++cnt] = i + 1, ja[cnt] = j + 1, ar[cnt] = (i == j ? 1. : 0);

    for(size_t j = 0; j < m; ++j) {
        size_t u = e[j].first, v = e[j].second;
        for(size_t i = 0; i < n; ++i)
            ia[++cnt] = i + 1, ja[cnt] = n + j + 1, ar[cnt] = ((i == u || i == v) ? 1. : 0);
    }

    glp_load_matrix(lp, cnt, ia, ja, ar);
    glp_simplex(lp, NULL);

    z = glp_get_obj_val(lp);
    glp_delete_prob(lp);

    tau.fw = z;
}
