//
// Created by anonymous author on 2022/8/2.
//

#ifndef SCOPE_TREE_H
#define SCOPE_TREE_H

#include "config.h"
#include "graph.h"
#include <set>
#include <queue>
#include <map>
#include <algorithm>
#include <functional>
#include <stack>

extern double gTDTime;

struct Node {
    int id;
    VertexID *vertices;
    ui numVertices;
    ui numSources;          // number of vertices that have 0 out degree
    VertexID *cut;          // sorted
    ui cutSize;
    VertexID *prefix;       // not sorted
    ui prefixSize;
    VertexID *key;          // sorted
    ui keySize;             // 0: root node, aggregation defined in tree 1: key is vertex 2: key is edge
    std::vector<VertexID> localOrder;
    std::vector<VertexID> nodeOrder;
    VertexID **automorphisms;
    ui autoSize;
    CanonType canonValue;         // the canonical value of the induced subgraph
    CanonType subPatternCanon;      // the canonical value of the directed subpattern rooted at this node
    int keyOrbit;                   // key's orbit in the sub-pattern (both P(\tau) and P(\Gamma[\tau])
    int numRules;                   // number of symmetry rules the node has
    int subPatternRules;            // number of symmetry rules the subpattern rooted as this node has
    std::vector<int> prefixOrbit;   // prefix's orbit in the sub-pattern
    int *v2o;   // the canonical orbits of P(\tau)
    int *vertexOrbit;  // the canonical orbits of P(\Gamma[\tau])
    std::set<int> validRules;   // valid symmetry rules for this node, depreciated; use candidate rules instead
    std::vector<std::vector<std::vector<VertexID>>> candidateRules;
    bool edgeKey;           // whether the key is edge / child key is edge / is the root and orbitType is 2
    bool unEdgeKey;         // whether the key is edge and is undirected
    ui numIn;               // number of in-neighbor lists visited when executing this node
    double fw;

    Node();

    Node(int id, VertexID *vertices, ui numVertices, ui numSources);

    Node(const Node &rhs);

    Node &operator=(const Node &rhs);

    virtual ~Node();

    bool operator<(const Node &rhs) const;

    bool operator>(const Node &rhs) const;

    bool operator==(const Node &rhs) const;

    bool hasVertex(VertexID u) const;
    void allgroup(grouprec *grp, const std::function<void(int*)>& action);
    void groupelts(levelrec *lr, int n, int level, const std::function<void(int*)>& action, int *before, int *after, int *identity);
    void copyAutom(int *p);
    void getCanonLabel(const PatternGraph &p);
    void computeValidRules(const PatternGraph &p, std::set<VertexID> &allCut);
    void computeCandidateRules(const PatternGraph &p, const std::vector<VertexID *> &childCuts, const std::vector<ui> &cutSizes);
    void cliqueNodeRules(const PatternGraph &p, const std::vector<VertexID *> &childCuts, const std::vector<ui> &cutSizes);
    void readFromStream(std::ifstream& inFile);
    void writeToStream(std::ofstream& outFile);
};

class Tree {
private:
    ui _numVertices;
    int _multiFactor;
    Node *_nodes;
    std::vector<VertexID>* _v2n;           // for each pattern vertex, store the nodes that contains it.
    ui _numNodes;
    std::vector<std::vector<VertexID>> _edges;
    ui *_parent;
    std::vector<std::vector<VertexID>> _child;
    int _numRules;
    std::vector<std::vector<VertexID>> _greaterRules; // symmetry breaking rules. greaterRules[i][j] : M[i] > M[j]
    std::vector<std::vector<VertexID>> _lessRules;
    int _orbitType;                         // 0: no orbit, 1: vertex orbit, 2: edge orbit
    std::vector<VertexID> _aggreV;          // aggregation vertices of the root
    std::vector<int> _aggreWeight;          // weight of each aggregation vertex, 1 by default
    ui _treeWidth;                             // maximum size of the subpattern that has to be enumerated
    ui _sumWidth;
    ui _maxNumSource;
    double _fhw;
    std::vector<VertexID> _postOrder;       // postorder of node ID
    bool _executeMode;                      // true: executeTree false: multiJoinTree
    std::vector<int> _partitionPos;         // positions of nodes in the post order that can build full hash tables
    std::vector<std::vector<std::vector<VertexID>>> _nodesAtStep;
    // for partition i, when the jth vertex in _globalOrder[i] is matched, then nodes in _nodesAtStep[i][j] can be executed
    // if use multi join tree, dimension 1 is nID, dimension 2 is mapping size, dimension 3 are the children to call
    std::vector<std::vector<int>> _prefixPos;
    std::vector<std::vector<VertexID>> _globalOrder;
    // for node nID, the position of the aggregation vertices
    std::vector<std::vector<int>> _aggrePos;
    // for some vertices there is no intersection of (in-/out-) neighbors
    // for such vertices, we directly assign them neighbor pointers,
    // for other vertices, they are allocated a candidate list
    std::vector<std::vector<bool>> _partitionInterPos;
    std::vector<std::vector<bool>> _nodeInterPos;
    // for node nID, when the mappingSize is i, the positions of in-neighbors are in _nodeInPos[nID][i]
    std::vector<std::vector<std::vector<int>>> _nodeInPos;
    std::vector<std::vector<std::vector<int>>> _nodeOutPos;
    std::vector<std::vector<std::vector<int>>> _nodeUnPos;
    std::vector<std::vector<std::vector<int>>> _nodeGreaterPos;
    std::vector<std::vector<std::vector<int>>> _nodeLessPos;
    std::vector<std::vector<std::vector<int>>> _partitionInPos;
    std::vector<std::vector<std::vector<int>>> _partitionOutPos;
    std::vector<std::vector<std::vector<int>>> _partitionUnPos;
    std::vector<std::vector<std::vector<int>>> _partitionGreaterPos;
    std::vector<std::vector<std::vector<int>>> _partitionLessPos;
    // for node nID, children i, the position of children's key in the mapping is _childKeyPos[nID][i]
    std::vector<std::vector<std::vector<int>>> _childKeyPos;
    // deal with edge ID computation
    // for node nID, when the mappingSize is i, the child edge key / edge aggregation keys are matched
    std::vector<std::vector<std::vector<int>>> _posChildEdge;
    std::vector<std::vector<std::vector<int>>> _posAggreEdge;
    // for an edge child key / edge aggregation key, the edge ID is given by: 1. edge ID at this position;
    // 2. the edge ID at this position is in-ID, map it to out-ID; 3. need to compute
    std::vector<std::vector<int>> _childEdgeType;
    std::vector<std::vector<int>> _aggreEdgeType;
    // for each node nID, whether each position need to allocate a candidate list
    std::vector<std::vector<bool>> _nodeCandPos;
    // structures for triangle-based intersection
    // for node nID, when the mappingSize is i, the positions of triangle neighbors are in _nodeTriPos[nID][i]
    std::vector<std::vector<std::pair<int, int>>> _nodeTriPos;
    std::vector<std::vector<int>> _triEdgeType; // 1: edge ID at p1 2: outID at p1 3: compute
    // the corresponding triend type is in _nodeTriType[nID][i]. 1: out-out 2: in-out 3: in-in
    std::vector<std::vector<int>> _triEndType;
    std::vector<std::vector<bool>> _partitionCandPos;
    std::vector<std::vector<std::pair<int, int>>> _partitionTriPos;
    std::vector<std::vector<int>> _partitionEdgeType;
    std::vector<std::vector<int>> _partitionEndType;

public:
    Tree();

    explicit Tree(ui numVertices);

    Tree(const Node &root, const std::vector<Node> &nodes, ui numVertices);

    Tree(const Tree &rhs);

    Tree &operator=(const Tree &rhs);

    bool operator==(const Tree &rhs) const;

    virtual ~Tree();

    inline ui getNumNodes() const {
        return _numNodes;
    }

    inline std::vector<VertexID> getPostOrder() const {
        return _postOrder;
    }

    inline std::vector<int> getPartitionPos() const {
        return _partitionPos;
    }

    inline void setTreeWidth(ui treeWidth) {
        _treeWidth = treeWidth;
    }

    void setNumRules(int numRules) {
        _numRules = numRules;
    }

    int getNumRules() const {
        return _numRules;
    }

    inline ui getTreeWidth() const {
        return _treeWidth;
    }

    inline void setGlobalOrder(const std::vector<std::vector<VertexID>> &globalOrder) {
        _globalOrder = globalOrder;
    }

    inline std::vector<std::vector<VertexID>> getGlobalOrder() const {
        return _globalOrder;
    }

    inline VertexID getRootID() const {
        return _postOrder.back();
    }

    inline const std::vector<std::vector<VertexID>> &getChild() const {
        return _child;
    }

    inline VertexID getParent(VertexID nID) const {
        return _parent[nID];
    }

    inline ui nodeNumVertices(VertexID nID) const {
        return _nodes[nID].numVertices;
    }

    inline bool nodeContains(VertexID nID, VertexID u) const {
        return _nodes[nID].hasVertex(u);
    }

    inline void setLocalOrder(VertexID nID, const std::vector<VertexID> &localOrder) {
        _nodes[nID].localOrder = localOrder;
    }

    inline void setMultiFactor(int multiFactor) {
        _multiFactor = multiFactor;
    }

    inline void setNumIn(VertexID nID, ui numIn) {
        _nodes[nID].numIn = numIn;
    }

    bool getExecuteMode() const {
        return _executeMode;
    }

    void setExecuteMode(bool executeMode) {
        _executeMode = executeMode;
    }

    const std::vector<std::vector<std::vector<VertexID>>> &getNodesAtStep() const {
        return _nodesAtStep;
    }

    const std::vector<bool> &getPartitionInterPos(VertexID pID) const {
        return _partitionInterPos[pID];
    }

    const std::vector<VertexID> &getPartitionOrder(VertexID pID) const {
        return _globalOrder[pID];
    }

    const std::vector<bool> &getNodeInterPos(VertexID nID) const {
        return _nodeInterPos[nID];
    }

    const std::vector<int> &getAggrePos(VertexID nID) const {
        return _aggrePos[nID];
    }

    const std::vector<std::vector<int>> &getNodeInPos(VertexID nID) const {
        return _nodeInPos[nID];
    }

    const std::vector<std::vector<int>> &getNodeOutPos(VertexID nID) const {
        return _nodeOutPos[nID];
    }

    const std::vector<std::vector<int>> &getNodeUnPos(VertexID nID) const {
        return _nodeUnPos[nID];
    }

    const std::vector<std::vector<int>> &getNodeGreaterPos(VertexID nID) const {
        return _nodeGreaterPos[nID];
    }

    const std::vector<std::vector<int>> &getNodeLessPos(VertexID nID) const {
        return _nodeLessPos[nID];
    }

    const std::vector<std::vector<int>> &getPartitionInPos(VertexID pID) const {
        return _partitionInPos[pID];
    }

    const std::vector<std::vector<int>> &getPartitionOutPos(VertexID pID) const {
        return _partitionOutPos[pID];
    }

    const std::vector<std::vector<int>> &getPartitionUnPos(VertexID pID) const {
        return _partitionUnPos[pID];
    }

    inline int partitionNumNodes(VertexID pID) const {
        if (pID == 0) return _partitionPos[pID] + 1;
        else return _partitionPos[pID] - _partitionPos[pID - 1];
    }

    const std::vector<std::vector<int>> &getChildKeyPos(VertexID nID) const {
        return _childKeyPos[nID];
    }

    inline const Node &getNode(VertexID nID) const {
        return _nodes[nID];
    }

    inline VertexID *getVertices(VertexID nID, ui &numVertices) const {
        numVertices = _nodes[nID].numVertices;
        return _nodes[nID].vertices;
    }

    inline VertexID *getPrefix(VertexID nID, ui &prefixSize) const {
        prefixSize = _nodes[nID].prefixSize;
        return _nodes[nID].prefix;
    }

    inline VertexID *getKey(VertexID nID, ui &keySize) const {
        keySize = _nodes[nID].keySize;
        return _nodes[nID].key;
    }

    CanonType getSubPatternCanon(VertexID nID) const {
        return _nodes[nID].subPatternCanon;
    }

    inline int getOrbitType() const {
        return _orbitType;
    }

    inline const std::vector<VertexID> &getAggreV() const {
        return _aggreV;
    }

    inline const std::vector<int> &getAggreWeight() const {
        return _aggreWeight;
    }

    const std::vector<std::vector<int>> &getPosChildEdge(VertexID nID) const {
        return _posChildEdge[nID];
    }

    const std::vector<std::vector<int>> &getPosAggreEdge(VertexID nID) const {
        return _posAggreEdge[nID];
    }

    const std::vector<int> &getChildEdgeType(VertexID nID) const {
        return _childEdgeType[nID];
    }

    const std::vector<int> &getAggreEdgeType(VertexID nID) const {
        return _aggreEdgeType[nID];
    }

    inline bool rootContains(VertexID u) const {
        return _nodes[_postOrder.back()].hasVertex(u);
    }

    bool nodeEdgeKey(VertexID nID) const {
        return _nodes[nID].edgeKey;
    }

    const std::vector<bool> &getNodeCandPos(VertexID nID) const {
        return _nodeCandPos[nID];
    }

    const std::vector<std::pair<int, int>> &getNodeTriPos(VertexID nID) const {
        return _nodeTriPos[nID];
    }

    const std::vector<int> &getTriEdgeType(VertexID nID) const {
        return _triEdgeType[nID];
    }

    const std::vector<int> &getTriEndType(VertexID nID) const {
        return _triEndType[nID];
    }

    const std::vector<std::vector<int>> &getPrefixPos() const {
        return _prefixPos;
    }

    const std::vector<std::vector<VertexID>> &getGreaterRules() const {
        return _greaterRules;
    }

    const std::vector<std::vector<VertexID>> &getLessRules() const {
        return _lessRules;
    }

    const std::vector<std::vector<int>> &getPartitionGreaterPos(VertexID pID) const {
        return _partitionGreaterPos[pID];
    }

    const std::vector<std::vector<int>> &getPartitionLessPos(VertexID pID) const {
        return _partitionLessPos[pID];
    }

    const std::vector<bool> &getPartitionCandPos(VertexID pID) const {
        return _partitionCandPos[pID];
    }

    const std::vector<std::pair<int, int>> &getPartitionTriPos(VertexID pID) const {
        return _partitionTriPos[pID];
    }

    const std::vector<int> &getPartitionEdgeType(VertexID pID) const {
        return _partitionEdgeType[pID];
    }

    const std::vector<int> &getPartitionEndType(VertexID pID) const {
        return _partitionEndType[pID];
    }

    ui getSumWidth() const {
        return _sumWidth;
    }

    ui getMaxNumSource() const {
        return _maxNumSource;
    }

    double getFhw() const {
        return _fhw;
    }

    int getMultiFactor() const {
        return _multiFactor;
    }

    bool operator<(const Tree &rhs) const;

    std::vector<VertexID> getUncovered(const PatternGraph &p) const;
    bool isValid(const PatternGraph &p);
    bool needPrefix(VertexID nID) const;
    bool hasSubNodeOf(const Node &tau) const;
    bool hasSupNodeOf(const Node &tau) const;
    std::vector<Tree> addNode(Node &tau);
    void computeSourceVertices(const PatternGraph &pin);
    void addPeripheral(const Pattern &p);
    void
    setNodeCanon(std::map<int, CanonType> &canonV, std::map<int, int *> &id2Orbits, std::map<int, ui> &id2AutoSize,
                 ui nID, const Pattern &p);
    void setNodeSubCanon(const Pattern &p);
    void setPrefixKeyOrbit(const Pattern &p);
    std::vector<Tree> getRootedTrees(const PatternGraph &p, bool sign = true);
    void rebuildCut();
    void setRoot(VertexID root, const PatternGraph &p);
    void computePostOrder(VertexID rootID, const std::vector<bool> &needPrefix);
    bool isLeaf(VertexID i) const;
    bool twoWidthEqual() const;
    bool inTheSameNode(VertexID u1, VertexID u2) const;
    VertexID *getCut(VertexID nID, ui &cnt) const;
    const std::vector<VertexID> & getNodesContainV(VertexID u) const;
    std::vector<VertexID> getNodesContainSymV(const std::vector<VertexID> &vartheta) const;
    void setSymmetryRoot(const std::vector<std::vector<VertexID>> &rules);
    void setSymmetryRules(const std::vector<std::vector<VertexID>> &allRule,
                          const std::vector<std::vector<VertexID>> &allNID);
    int computeSymmetryRules(const PatternGraph &p);
    bool containOrDisjoint(const std::vector<VertexID> &rule);
    bool autoConsistent(const std::vector<VertexID> &nIDs, const std::vector<VertexID> &rule);
    Edge *getRootedSubgraph(VertexID nID, const PatternGraph &p, std::vector<VertexID> &old2New,
                            ui &numVertices, ui &numEdges) const;
    Tree mergeToTwoLevel() const;
    void setPrefix(VertexID nID, std::vector<VertexID> prefix, const Pattern &p);
    Tree relabel(VertexID *mapping) const;
    void adjustWeight(int pos, bool sign, VertexID u, VertexID v, int weight);
    void adjustWeight();
    void adjustMultiFactor(int value);
    bool posInAggreV(VertexID u, int &pos);
    bool posInAggreV(VertexID u, VertexID v, int &pos);
    void setAggreInfo(int orbitType, VertexID u, VertexID v, bool sign, int weight, const Pattern &p);
    void initPoses(const Pattern &p, bool useTriangle = true);
    void initMultiJoinPoses(const Pattern &p, bool useTriangle = true);
    void print() const;
    void printRules() const;
    void dropRules(const Pattern &p, bool useTriangle, int multiFactor);
    void computeNodeRules(VertexID nID, const PatternGraph &p);
    void computeWidths(const PatternGraph &p);
    // the followings should be removed in the future
    int cutCase(const PatternGraph &p) const;
    std::vector<Tree> realTree(const Pattern &p, bool sign);
    void writeToFile(const std::string& filename);
    void readFromFile(const std::string& filename);
    void wheel9Tree();
};

// concentrated node
struct ConNode {
    // the number of nodes compressed into this node
    int num;
    int numRules;
    int divideFactor;
    bool edgeKey;
    std::vector<VertexID> cutVertices;
    VertexID uIn;   // the inner vertex
    std::vector<VertexID> nodeOrder;
    std::vector<VertexID> prefixOrder;
    CanonType canonValue;
    int orbitType;
    // only has one virtual child
    std::vector<int> childKeyPos;
    std::vector<int> aggrePos;
    // true: select num from cnt, false: select num-1 from cnt-1
    // if it has prefix, uIn uses copy, cut vertices use key and set value to 0 after use
    std::vector<bool> aggreType;
    std::vector<HashTable> hashTables;
    std::vector<std::vector<int>> nodeInPos;
    std::vector<std::vector<int>> nodeOutPos;
    std::vector<std::vector<int>> nodeUnPos;
    std::vector<std::vector<int>> greaterPos;
    std::vector<std::vector<int>> lessPos;
    std::vector<bool> interPos;
    std::vector<bool> candPos;
    std::vector<std::pair<int, int>> triPos;
    int posChildEdge;
    std::vector<std::vector<int>> posAggreEdge;
    int childEdgeType;
    std::vector<int> aggreEdgeType;
    std::vector<int> triEdgeType;
    std::vector<int> triEndType;

    ConNode() {
        num = 0;
    };
    ConNode(const PatternGraph &p, const Tree &t);
    void setAggrePoses(const PatternGraph &p);
    void initPoses(const std::vector<VertexID> &pOrder, const std::vector<VertexID> &localOrder,
                   std::vector<std::vector<VertexID>> &greaterRules, std::vector<std::vector<VertexID>> &lessRules,
                   const PatternGraph &p, bool useTriangle);
    void print() const;
    void writeToStream(std::ofstream& outFile);
    void readFromStream(std::ifstream& inFile);
};

static std::map<CanonType, double> canonToFW;
std::vector<Node> getAllNodes(const PatternGraph &p);
std::vector<Node> getCandidateNodes(const Tree &t, const std::vector<Node> &allNodes, const PatternGraph &p);
void findCliquesRecursive(const PatternGraph &graph,
                          std::vector<VertexID> &currentClique,
                          std::vector<VertexID> &potentialClique,
                          std::vector<VertexID> &processedVertices,
                          std::queue<Node> &cliques);
std::queue<Node> findMaximalCliques(const PatternGraph &graph);
std::vector<Tree> getAllTree(const PatternGraph &p);
std::vector<Tree> getAllTree(const std::vector<Node> &allNode, const PatternGraph &p);
void fractionalWidth(struct Node &tau, const PatternGraph &p, const std::vector<VertexID> &pathToTau);

#endif //SCOPE_TREE_H
