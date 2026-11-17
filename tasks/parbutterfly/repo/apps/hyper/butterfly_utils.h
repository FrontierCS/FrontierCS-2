#ifndef _BUTILS_
#define _BUTILS_

#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <string>
#include <assert.h>

#include "parallel.h"
#include "gettime.h"
#include "utils.h"
#include "graph.h"
#include "hypergraphIO.h"
#include "parseCommandLine.h"
#include "vertex.h"
#include "sequence.h"
#include "binary_search.h"

#define MAX_STEP_SIZE 1000

using namespace std;

enum CountType {SORT, ASORT, HASH, AHASH, HIST, AHIST, BATCHS, BATCHWA, SERIAL};
enum RankType {SIDE, COCORE, ACOCORE, DEG, ADEG};
enum PeelType {PSORT, PHASH, PHIST, PBATCHS, PBATCHWA};
enum PerType {VERT, EDGE, TOTAL};
enum SparseType {NOSPARSE, CLRSPARSE, ESPARSE};

// Represents a wedge, where v1 and v2 are the endpoints, u is the center, j is
// the index of u as a neighbor of v1, and k is the index of v2 as a neighbor
// of u. Also, constructors and comparators for UWedge.
struct UWedge {
  uintE v1;
  uintE v2;
  uintE u;
  intT j; intT k;
  UWedge() {}
UWedge(uintE _v1, uintE _v2, uintE _u, intT _j, intT _k) : v1(_v1), v2(_v2), u(_u), j(_j), k(_k) {}
};
struct UWedgeCons {
  inline UWedge operator() (uintE v1, uintE v2, uintE c, intT j, intT k) { return UWedge(v1, v2, c, j, k); }
};
struct UWedgeCmp {
  inline bool operator() (UWedge vs1, UWedge vs2) {
    if (vs1.v1 == vs2.v1) return vs1.v2 < vs2.v2;
    return vs1.v1 < vs2.v1;
  }
};
struct UWedgeEq { inline bool operator() (UWedge vs1, UWedge vs2) { return (vs1.v1 == vs2.v1) && (vs1.v2 == vs2.v2);} };


// Constructs a long representing the endpoints of a wedge
struct UWedgeIntRankCons {
  long nu;
UWedgeIntRankCons(long _nu) : nu(_nu) {}
  inline long operator() (uintE v1, uintE v2, uintE c, intT j, intT k) {
    return ((((long) v1) * nu + ((long) v2 >> 1)) << 1) + (v2 & 0b1);
  }
};
struct UVertexPairIntCons {
  long nu;
UVertexPairIntCons(long _nu) : nu(_nu) {}
  inline long operator() (uintE v1, uintE v2, uintE c, intT j, intT k) {
    return (long) v1 * nu + (long) v2;
  }
};
struct UVertexPairIntRankCons {
  long nu;
UVertexPairIntRankCons(long _nu) : nu(_nu) {}
  inline long operator() (uintE v1, uintE v2, bool b) {
    return ((((long) v1) * nu + (long) v2) << 1) + (b ? 0b1 : 0);
  }
};


// Represents a pair of vertices on one side of a bipartite graph (unordered,
// stored based on constructor order). Also, constructors and comparators for
// UVertexPair.
struct UVertexPair {
  uintE v1;
  uintE v2;
  UVertexPair() {}
UVertexPair(uintE _v1, uintE _v2) : v1(_v1), v2(_v2) {}
};
struct UVertexPairCons {
  inline UVertexPair operator() (uintE v1, uintE v2, uintE c, intT j, intT k) { return UVertexPair(v1, v2); }
};
struct UVertexPairRankCons {
  inline UVertexPair operator() (uintE v1, uintE v2, bool b) { return UVertexPair(v1, (v2 << 1) + b); }
};
struct UVertexPairCmp2{
  UVertexPairCmp2() {}
  inline bool operator() (UVertexPair vs1, UVertexPair vs2) {
    if (vs1.v2 == vs2.v2) return vs1.v1 < vs2.v1;
    return vs1.v2 < vs2.v2;
  }
};
struct UVertexPairCmp {
  UVertexPairCmp() {}
  inline bool operator() (UVertexPair vs1, UVertexPair vs2) {
    if (vs1.v1 == vs2.v1) return vs1.v2 < vs2.v2;
    return vs1.v1 < vs2.v1;
  }
};
struct UVertexPairEq {
  inline bool operator() (UVertexPair vs1, UVertexPair vs2) { return (vs1.v1 == vs2.v1) && (vs1.v2 == vs2.v2);}
};
struct UVertexPairV2{inline uintE operator() (UVertexPair obj) {return obj.v2;}};


// Returns the first and second endpoints in UVertexPair and UWedge
struct UVPFirst { uintE operator() (const UVertexPair& x) {return x.v1;}};
struct UVPSecond { uintE operator() (const UVertexPair& x) {return x.v2;}};
struct UWFirst { uintE operator() (const UWedge& x) {return x.v1;}};
struct UWSecond { uintE operator() (const UWedge& x) {return x.v2;}};

// Tuple functions
template<class T, class X>
  struct tupleFirst {T operator() (tuple<T,X> a) {return get<0>(a);} };
struct nonZeroF{inline bool operator() (tuple<uintE,uintE> &a) {return (get<1>(a) != 0);}};
struct greaterOneLongF{inline bool operator() (tuple<long,uintE> &a) {return (get<1>(a) > 1);}};
struct nonMaxTupleF{
  inline bool operator() (tuple<uintE,uintE> &a) {return (get<1>(a) != UINT_E_MAX || get<0>(a) != UINT_E_MAX);}
};

template<class T, class X>
  struct tupleLt {inline bool operator() (tuple<T,X> a, tuple<T,X> b) {return get<0>(a) < get<0>(b);} };
template<class T, class X>
  struct tupleEq {inline bool operator() (tuple<T,X> a,tuple<T,X> b) {return get<0>(a) == get<0>(b); }};
template<class T, class X>
  struct tupleAdd {
    inline tuple<T,X> operator() (tuple<T,X> a, tuple<T,X> b) const {
      return make_tuple(get<0>(a), get<1>(a) + get<1>(b));
    };
  };

struct uintETupleGet0{inline uintE operator() (tuple<uintE,long> obj) {return get<0>(obj);}};
struct uintECount{inline long operator() (tuple<uintE,long> obj) {return get<1>(obj);}};

// Other functions
template<class T> struct refl{inline T operator() (T obj) {return obj;}};
template<class T> struct reflCount{inline long operator() (T obj) {return 1;}};
struct choose2{inline long operator() (long obj) {return obj*(obj-1)/2;}};

template <class E>
struct writeAddArr {
  E* arr;
writeAddArr(E* _arr) : arr(_arr) {}
  inline void operator() (long idx, E num) {
    writeAdd(&arr[idx], num);
  }
};

template <class E>
struct writeAddSet {
  sparseAdditiveSet<E> set;
writeAddSet(sparseAdditiveSet<E> _set) : set(_set) {}
  inline void operator() (long idx, E num) {
    set.insert(pair<uintE, E>(idx, num));
  }
};

template<class K, class E>
  E getAdd (E curr, tuple<K,E> v) {
  return curr + get<1>(v);
}

template<class K, class E>
  tuple<K,E> getAddReduce (tuple<K,E> curr, tuple<K,E> v) {
  return make_tuple(get<0>(curr),get<1>(curr) + get<1>(v));
}

template<class K, class E>
  pair<K,E> getAddReducePair (pair<K,E> curr, pair<K,E> v) {
  return make_pair(v.first, curr.second + v.second);
}

//***********************************************************************************************
//***********************************************************************************************


// Symmetric compact bipartite graph format (CSR)
struct bipartiteCSR {
  uintT *offsetsV, *offsetsU;
  uintE *edgesV, *edgesU;
  long nv, nu, numEdges;

  bipartiteCSR () {}
bipartiteCSR(uintT* _offsetsV, uintT* _offsetsU, uintE* _edgesV, uintE* _edgesU, long _nv, long _nu, long _ne) :
  offsetsV(_offsetsV), offsetsU(_offsetsU), edgesV(_edgesV), edgesU(_edgesU), nv(_nv), nu(_nu), numEdges(_ne)
  {}
  
  void del() {
    free(offsetsV); free(offsetsU); free(edgesV); free(edgesU);
  }
};

// Symmetric compact graph format (CSR)
struct graphCSR {
  uintT *offsets;
  uintE *edges;
  long n, numEdges;

graphCSR(uintT* _offsets, uintE* _edges, long _n, long _ne) :
  offsets(_offsets), edges(_edges), n(_n), numEdges(_ne)
  {}

  void del() {
    free(offsets); free(edges);
  }
};

// Function to compute the number of wedges a ranked graph produces
template <class E>
struct rankWedgeF { 
  uintT* offsets;
  uintE* edges;
  uintE* rank;
  uintE* orank;

rankWedgeF(uintT* _offsets, uintE* _edges, uintE* _rank, uintE* _orank) :
  offsets(_offsets), edges(_edges), rank(_rank), orank(_orank) {}

  // Each vertex i contributes indegree * outdegree + (indegree choose 2) wedges
  inline E operator() (const uintT& i) const {
    intT v_offset = offsets[i];
    intT v_deg = offsets[i+1]-v_offset;
    intT in_deg = 0; intT out_deg = 0;
    for(intT j=0; j < v_deg; ++j) {
      uintE u = edges[v_offset+j];
      if (orank[u] < rank[i]) in_deg++;
      else out_deg++;
    }
    return (E) ( in_deg * out_deg + ((in_deg * (in_deg-1)) / 2)); 
  }
};

/*
 *  Ranks vertices in bipartite graph G according to the ordering samplesort_f.
 * 
 *  GA          : Bipartite graph in CSR format
 *  samplesort_f: Comparison function that orders U and V vertices (where all U indices come before V indices)
 * 
 *  Returns a tuple of arrays. The first array sorts indices given by U and V by rank (where all U indices come before V
 *  indices). The second array maps V indices to their rank, and the third array maps U indices to their rank.
 */
template<class F>
tuple<uintE*, uintE*, uintE*> getRanks(bipartiteCSR& G, F samplesort_f) {
  uintE* ranks = newA(uintE, G.nv + G.nu);
  uintE* rankV = newA(uintE, G.nv);
  uintE* rankU = newA(uintE, G.nu);
  
  // Sort indices given by U and V by rank
  parallel_for(long v=0; v < G.nv+G.nu; ++v) { ranks[v] = v; }
  sampleSort(ranks, G.nv+G.nu, samplesort_f); 

  // Map each U and V vertex to its rank
  parallel_for(long i=0; i < G.nv + G.nu; ++i) {
    if (ranks[i] >= G.nv) rankU[ranks[i] - G.nv] = i;
    else rankV[ranks[i]] = i;
  }

  return make_tuple(ranks, rankV, rankU);
}

/*
 *  Ranks vertices in bipartite graph G according to approximate degree ordering
 * 
 *  GA: Bipartite graph in CSR format
 * 
 *  Returns a tuple of arrays. The first array sorts indices given by U and V by rank (where all U indices come before V
 *  indices). The second array maps V indices to their rank, and the third array maps U indices to their rank.
 */
tuple<uintE*, uintE*, uintE*> getApproxDegRanks(bipartiteCSR& G) {
  auto samplesort_f = [&] (const uintE a, const uintE b) -> const uintE {
    uintE deg_a = (a >= G.nv) ? G.offsetsU[a-G.nv+1]-G.offsetsU[a-G.nv] : G.offsetsV[a+1]-G.offsetsV[a];
    uintE deg_b = (b >= G.nv) ? G.offsetsU[b-G.nv+1]-G.offsetsU[b-G.nv] : G.offsetsV[b+1]-G.offsetsV[b];
    uintE log_deg_a = deg_a <= 0 ? 0 : (uintE) floor(log2(deg_a)) + 1;
    uintE log_deg_b = deg_b <= 0 ? 0 : (uintE) floor(log2(deg_b)) + 1;
    return log_deg_a > log_deg_b;
  };
  return getRanks(G, samplesort_f);
}

/*
 *  Ranks vertices in bipartite graph G according to degree ordering
 * 
 *  GA: Bipartite graph in CSR format
 * 
 *  Returns a tuple of arrays. The first array sorts indices given by U and V by rank (where all U indices come before V
 *  indices). The second array maps V indices to their rank, and the third array maps U indices to their rank.
 */
tuple<uintE*, uintE*, uintE*> getDegRanks(bipartiteCSR& G) {
  auto samplesort_f = [&] (const uintE a, const uintE b) -> const uintE {
    uintE deg_a = (a >= G.nv) ? G.offsetsU[a-G.nv+1]-G.offsetsU[a-G.nv] : G.offsetsV[a+1]-G.offsetsV[a];
    uintE deg_b = (b >= G.nv) ? G.offsetsU[b-G.nv+1]-G.offsetsU[b-G.nv] : G.offsetsV[b+1]-G.offsetsV[b];
    return deg_a > deg_b;
  };
  return getRanks(G, samplesort_f);
}

// Function that retrieves the degrees of vertices in a bipartite graph, when ordered with all U indices before V
// indices
template <class E>
struct degF { 
  uintT* offsetsV;
  uintT* offsetsU;
  vertexSubset active;
  long nv;
degF(long _nv, uintT* _offsetsV, uintT* _offsetsU, vertexSubset& _active) :
  nv(_nv), offsetsV(_offsetsV), offsetsU(_offsetsU), active(_active) {}
  inline E operator() (const uintT& i) const {
    bool use_v = active.vtx(i) < nv;
    intT idx = use_v ? active.vtx(i) : active.vtx(i) - nv;
    intT deg = (use_v ? offsetsV[idx+1] - offsetsV[idx] : offsetsU[idx+1] - offsetsU[idx]);
    return (E) (deg); 
  }
};

/*
 *  Finds the one-hop neighbors of vertices in V.
 * 
 *  GA: Bipartite graph in CSR format
 *  V : Set of active vertices
 * 
 *  Returns a vertexSubsetData object that wraps an array of all one-hop neighbors of active vertices in V, in the 
 *  form (neighbor, 1).
 */
template <class E, class VS>
  vertexSubsetData<E> edgeMapInduced(bipartiteCSR& GA, VS& V) {
  long nTo = GA.nv + GA.nu;
  uintT m = V.size();
  V.toSparse();

  // Find degrees of vertices in V so that we can produce indices to store
  // all neighbors in parallel
  auto degrees = array_imap<uintT>(m);
  granular_for(i, 0, m, (m > 2000), {
      bool use_v = V.vtx(i) < GA.nv;
      intT idx = use_v ? V.vtx(i) : V.vtx(i) - GA.nv;
      intT deg = (use_v ? GA.offsetsV[idx+1] - GA.offsetsV[idx] : GA.offsetsU[idx+1] - GA.offsetsU[idx]);
      degrees[i] = deg;
    });
  long outEdgeCount = pbbs::scan_add(degrees, degrees);

  if (outEdgeCount == 0) {
    return vertexSubsetData<E>(nTo);
  }
  typedef tuple<uintE, E> VE;
  VE* outEdges = pbbs::new_array_no_init<VE>(outEdgeCount);

  // Retrieve all neighbors of vertices in V
  parallel_for (size_t i = 0; i < m; i++) {
    uintT o = degrees[i];
    bool use_v = V.vtx(i) < GA.nv;
    intT idx = use_v ? V.vtx(i) : V.vtx(i) - GA.nv;
    intT offset  = use_v ? GA.offsetsV[idx] : GA.offsetsU[idx];
    intT deg = (use_v ? GA.offsetsV[idx+1] : GA.offsetsU[idx+1]) - offset;
    granular_for(j,0,deg,deg > 10000, {
	intT nbhr = use_v ? GA.edgesV[offset+j] + GA.nv : GA.edgesU[offset+j];
	// Store the one-hop neighbor of idx
	outEdges[o+j] = make_tuple(nbhr, 1);
      });
  }
  auto vs = vertexSubsetData<E>(nTo, outEdgeCount, outEdges);
  return vs;
}

// Struct to hold logic for retrieving and updating with relation to one-hop
// neighbors of active sets
template <class Val>
struct BipartiteProp {
  using K = uintE; // keys are always uintE's (vertex-identifiers)
  using KV = tuple<K, Val>;
  bipartiteCSR GA;
  pbbs::hist_table<K, Val> ht;

BipartiteProp(bipartiteCSR& _GA, KV _empty, size_t ht_size=numeric_limits<size_t>::max()) : GA(_GA) {
  if (ht_size == numeric_limits<size_t>::max()) {
    ht_size = 1L << pbbs::log2_up(GA.numEdges/20);
  } else { ht_size = 1L << pbbs::log2_up(ht_size); }
  ht = pbbs::hist_table<K, Val>(_empty, ht_size);
}

  /*
   *  Collates the one-hop neighbors of vertices in vs, using reduce_f to
   *  collate (neighbor, 1) pairs and using apply_f to map the (neighbor, count)
   *  pairs to the desired format.
   * 
   *  vs      : Set of active vertices
   *  reduce_f: A function (M current_count, tuple(uintE neighbor, M count)) -> M reduced_val
   *  apply_f : A function (uintE neighbor, M reduced_val) -> O
   * 
   *  Returns a vertexSubsetData object that wraps an array of collated one-hop neighbors of the vertices in vs, in a
   *  format as specified by apply_f.
   */
  template <class O, class M, class Reduce, class Apply, class VS>
    inline vertexSubsetData<O> edgeMapReduce(VS& vs, Reduce& reduce_f, Apply& apply_f) {
    size_t m = vs.size();
    if (m == 0) {
      return vertexSubsetData<O>(GA.nv+GA.nu);
    }
    // Retrieve all one-hop neighbors
    auto oneHop = edgeMapInduced<M, VS>(GA, vs);
    oneHop.toSparse();

    auto get_elm = make_in_imap<tuple<K, M> >(oneHop.size(), [&] (size_t i) { return oneHop.vtxAndData(i); });
    auto get_key = make_in_imap<uintE>(oneHop.size(), [&] (size_t i) -> uintE { return oneHop.vtx(i); });

    // Use a histogram to collate one-hop neighbors
    auto q = [&] (sequentialHT<K, Val>& S, tuple<K, M> v) -> void { S.template insertF<M>(v, reduce_f); };
    auto res = pbbs::histogram_reduce<tuple<K, M>, tuple<K, O> >(get_elm, get_key, oneHop.size(), q, apply_f, ht);
    oneHop.del();
    return vertexSubsetData<O>(GA.nv+GA.nu, res.first, res.second);
  }

  /*
   *  Counts the one-hop neighbors of vertices in vs, using apply_f to map the
   *  (neighbor, count) pairs to the desired format.
   * 
   *  vs      : Set of active vertices
   *  apply_f : A function (uintE neighbor, M reduced_val) -> O
   * 
   *  Returns a vertexSubsetData object that wraps an array of counted one-hop neighbors of the vertices in vs, in a
   *  format as specified by apply_f.
   */
  template <class O, class Apply, class VS>
    inline vertexSubsetData<O> bpedgePropCount(VS& vs, Apply& apply_f) {
    auto reduce_f = [&] (const uintE& cur, const tuple<uintE, uintE>& r) { return cur + 1; };
    return edgeMapReduce<O, uintE>(vs,reduce_f, apply_f);
  }

  ~BipartiteProp() {
    ht.del();
  }
};

/*
 *  Ranks vertices in bipartite graph G according to approximate complement
 *  degeneracy ordering.
 * 
 *  GA: Bipartite graph in CSR format
 * 
 *  Returns a tuple of arrays. The first array sorts indices given by U and V by rank (where all U indices come before V
 *  indices). The second array maps V indices to their rank, and the third array maps U indices to their rank.
 */
tuple<uintE*,uintE*,uintE*> getApproxCoCoreRanks(bipartiteCSR& GA, size_t num_buckets=128) {
  using X = tuple<bool, uintE>;
  long n = GA.nv + GA.nu;
  // Map each vertex to the log of its degree
  auto D = array_imap<uintE>(n, [&] (size_t i) {
      if(i >= GA.nv)
        return GA.offsetsU[i-GA.nv+1] - GA.offsetsU[i-GA.nv] <= 0 ? 0 :
          (uintE) floor(log2(GA.offsetsU[i-GA.nv+1] - GA.offsetsU[i-GA.nv])) + 1;
      return GA.offsetsV[i+1] - GA.offsetsV[i] <= 0 ? 0 : (uintE) floor(log2(GA.offsetsV[i+1] - GA.offsetsV[i])) + 1;
    });
  // Keep a map of each vertex to its degree
  auto D_act = array_imap<uintE>(n, [&] (size_t i) {
      if(i >= GA.nv) return GA.offsetsU[i-GA.nv+1] - GA.offsetsU[i-GA.nv];
      return GA.offsetsV[i+1] - GA.offsetsV[i];
    });

  // Structure to maintain bucket updates
  auto hp = BipartiteProp<uintE>(GA, make_tuple(UINT_E_MAX, 0), (size_t)GA.numEdges/5);
  auto b = make_buckets(n, D, decreasing, num_buckets);

  size_t finished = 0;
  while (finished != n) {
    auto bkt = b.next_bucket();
    // Active vertices (in the bucket that we're processing)
    auto active2 = bkt.identifiers;
    uintE k = bkt.id;
    finished += active2.size();
    // Given updated degrees, function to update our maps and recompute bucket
    auto apply_f = [&] (const tuple<uintE, uintE>& p) -> const Maybe<tuple<uintE, uintE> > {
      uintE v = std::get<0>(p), edgesRemoved = std::get<1>(p);
      uintE deg = D_act.s[v];
      uintE deg_log = D.s[v];
      if (deg_log < k) {
        uintE new_deg = deg - edgesRemoved;
        uintE new_deg_log = deg - edgesRemoved <= 0 ? 0 : (uintE) floor(log2(new_deg)) + 1;
        D_act.s[v] = new_deg;
        D.s[v] = min(new_deg_log, k);
        uintE bkt = b.get_bucket(deg_log, new_deg_log);
        return wrap(v, bkt);
      }
      return Maybe<tuple<uintE, uintE> >();
    };

    // Retrieve updated degrees and update buckets
    vertexSubsetData<uintE> moved = hp.template bpedgePropCount<uintE>(active2, apply_f);
    b.update_buckets(moved.get_fn_repr(), moved.size());
    moved.del(); active2.del();
  }
  b.del();

  // Given the ordering by approx complement degeneracy, compute rankings
  auto samplesort_f = [&] (const uintE a, const uintE b) -> const uintE {
    return D[a] > D[b];
  };
  return getRanks(GA, samplesort_f);
}

/*
 *  Ranks vertices in bipartite graph G according to complement degeneracy
 *  ordering.
 * 
 *  GA: Bipartite graph in CSR format
 * 
 *  Returns a tuple of arrays. The first array sorts indices given by U and V by rank (where all U indices come before V
 *  indices). The second array maps V indices to their rank, and the third array maps U indices to their rank.
 */
tuple<uintE*,uintE*,uintE*> getCoCoreRanks(bipartiteCSR& GA, size_t num_buckets=128) {
  using X = tuple<bool, uintE>;
  long n = GA.nv + GA.nu;
  // Map each vertex to its degree
  auto D = array_imap<uintE>(n, [&] (size_t i) {
      if(i >= GA.nv) return GA.offsetsU[i-GA.nv+1] - GA.offsetsU[i-GA.nv];
      return GA.offsetsV[i+1] - GA.offsetsV[i];
    });

  // Structure to maintain bucket updates
  auto hp = BipartiteProp<uintE>(GA, make_tuple(UINT_E_MAX, 0), (size_t)GA.numEdges/5);
  auto b = make_buckets(n, D, decreasing, num_buckets);

  size_t finished = 0;
  while (finished != n) {
    auto bkt = b.next_bucket();
    // Active vertices (in the bucket that we're processing)
    auto active2 = bkt.identifiers;
    uintE k = bkt.id;
    finished += active2.size();
    // Given updated degrees, function to update our maps and recompute bucket
    auto apply_f = [&] (const tuple<uintE, uintE>& p) -> const Maybe<tuple<uintE, uintE> > {
      uintE v = std::get<0>(p), edgesRemoved = std::get<1>(p);
      uintE deg = D.s[v];
      if (deg < k) {
        uintE new_deg = min(deg - edgesRemoved, k);
        D.s[v] = new_deg;
        uintE bkt = b.get_bucket(deg, new_deg);
        return wrap(v, bkt);
      }
      return Maybe<tuple<uintE, uintE> >();
    };

    // Retrieve updated degrees and update buckets
    vertexSubsetData<uintE> moved = hp.template bpedgePropCount<uintE>(active2, apply_f);
    b.update_buckets(moved.get_fn_repr(), moved.size());
    moved.del(); active2.del();
  }
  b.del();

  // Given the ordering by approx complement degeneracy, compute rankings
  auto samplesort_f = [&] (const uintE a, const uintE b) -> const uintE {
    return D[a] > D[b];
  };
  return getRanks(GA, samplesort_f);
}

//********************************************************************************************
//********************************************************************************************

/*
 *  Computes a graph in CSR format with vertices and adjacency lists ordered
 *  by rank, as given by ranks, rankV, and rankU.
 * 
 *  G    : Bipartite graph in CSR format
 *  use_v: Denotes which bipartition to store counts on, based on which bipartition produces the fewest wedges.
 *         If true, stores counts on U. If false, stores counts on V. This information is encapsulated in the
 *         returned graph
 *  ranks: Sorted U and V vertices by rank (where U indices all come before V indices)
 *  rankV: Array mapping V indices to their rank
 *  rankU: Array mapping U indices to their rank
 * 
 *  Returns a graph in CSR format with vertices and adjacency lists  ordered
 *  by rank.
 */
graphCSR rankGraph(bipartiteCSR& G, bool use_vb, uintE* ranks, uintE* rankV, uintE* rankU) {
  using X = tuple<uintE,uintE>;
  uintT* offsets = newA(uintT,G.nv+G.nu+1);
  offsets[G.nv+G.nu] = 0;
  // Set up offsets
  parallel_for(long i=0; i < G.nv + G.nu; ++i) {
    if (ranks[i] >= G.nv) offsets[i] = G.offsetsU[ranks[i]-G.nv+1]-G.offsetsU[ranks[i]-G.nv];
    else offsets[i] = G.offsetsV[ranks[i]+1]-G.offsetsV[ranks[i]];
  }
  sequence::plusScan(offsets,offsets,G.nv+G.nu+1);

  uintE* edges = newA(uintE,offsets[G.nv+G.nu]);
#ifdef INVERSE
  auto lt = [] (const uintE& l, const uintE& r) { return l < r; };
#else
  auto lt = [] (const uintE& l, const uintE& r) { return l > r; };
#endif

  // Fill in the correct edges using offsets
  parallel_for(long i=0; i < G.nv + G.nu; ++i) {
    bool use_v = (ranks[i] < G.nv);
    // Retrieve the corresponding vertex in G
    intT idx = use_v ? ranks[i] : ranks[i] - G.nv;
    intT offset  = use_v ? G.offsetsV[idx] : G.offsetsU[idx];
    intT deg = (use_v ? G.offsetsV[idx+1] : G.offsetsU[idx+1])-offset;
    granular_for(j,0,deg,deg > 10000, { 
	// Fill in all neighbors of idx using G
	intT nbhr = use_v ? G.edgesV[offset+j] : G.edgesU[offset+j];
	uintE r; 
	// Each vertex is appended with a rightmost bit, which is 1 if 
	// it is on the right bipartition to store butterfly counts on, and 0 
	// otherwise.
	// That is to say, we append a 1 if the vertex is in U and use_vb is true,
	// or if the vertex is in V and use_vb is false. Otherwise, we append 0.
	if (use_vb) r = use_v ? (rankU[nbhr] << 1) + 0b1  : (rankV[nbhr] << 1);
	else r = use_v ? (rankU[nbhr] << 1) : (rankV[nbhr] << 1) + 0b1;
	edges[offsets[i]+j] = r;
      });
    // Sort adjacency list of idx
    sampleSort(&edges[offsets[i]], deg, lt);
  }

  // Create sorted graph
  return graphCSR(offsets,edges,G.nv+G.nu,G.numEdges);
}

/*
 *  Computes a graph in CSR format with vertices and adjacency lists ordered
 *  by rank, as given by ranks, rankV, and rankU. Also, saves a conversion
 *  array that translates edges in the ranked graph to their original
 *  indices in G.
 * 
 *  G    : Bipartite graph in CSR format
 *  use_v: Denotes which bipartition to store counts on, based on which bipartition produces the fewest wedges.
 *         If true, stores counts on U. If false, stores counts on V. This information is encapsulated in the
 *         returned graph
 *  ranks: Sorted U and V vertices by rank (where U indices all come before V indices)
 *  rankV: Array mapping V indices to their rank
 *  rankU: Array mapping U indices to their rank
 * 
 *  Returns a pair, where the first element is a graph in CSR format with
 *  vertices and adjacency lists  ordered by rank. The second element is
 *  an array of tuples that converts edges in the ranked graph to
 *  their original index in GA (the first elements of the tuple form 
 *  the edge list of the ranked graph, and the second element is the
 *  corresponding original index in GA).
 */
pair<graphCSR,tuple<uintE,uintE>*> rankGraphEdges(bipartiteCSR& G, bool use_vb, uintE* ranks, uintE* rankV, uintE* rankU) {
  using X = tuple<uintE,uintE>;

  uintT* offsets = newA(uintT,G.nv+G.nu+1);
  offsets[G.nv+G.nu] = 0;
  // Set up offsets
  parallel_for(long i=0; i < G.nv + G.nu; ++i) {
    if (ranks[i] >= G.nv) offsets[i] = G.offsetsU[ranks[i]-G.nv+1]-G.offsetsU[ranks[i]-G.nv];
    else offsets[i] = G.offsetsV[ranks[i]+1]-G.offsetsV[ranks[i]];
  }

  sequence::plusScan(offsets,offsets,G.nv+G.nu+1);

  uintE* edges = newA(uintE,offsets[G.nv+G.nu]);
  X* edges_convert = newA(X,offsets[G.nv+G.nu]);
#ifdef INVERSE
  auto lt = [] (const uintE& l, const uintE& r) { return l < r; };
  auto ltx = [] (const X& l, const X& r) { return get<0>(l) < get<0>(r); };
#else
  auto lt = [] (const uintE& l, const uintE& r) { return l > r; };
  auto ltx = [] (const X& l, const X& r) { return get<0>(l) > get<0>(r); };
#endif

  // Fill in the correct edges using offsets
  parallel_for(long i=0; i < G.nv + G.nu; ++i) {
    bool use_v = (ranks[i] < G.nv);
    // Retrieve the corresponding vertex in G
    intT idx = use_v ? ranks[i] : ranks[i] - G.nv;
    intT offset  = use_v ? G.offsetsV[idx] : G.offsetsU[idx];
    intT deg = (use_v ? G.offsetsV[idx+1] : G.offsetsU[idx+1])-offset;
    // Fill in all neighbors of idx using G
    granular_for(j,0,deg,deg > 10000, { 
	intT nbhr = use_v ? G.edgesV[offset+j] : G.edgesU[offset+j];
	uintE r;
	// Each vertex is appended with a rightmost bit, which is 1 if 
	// it is on the right bipartition to store butterfly counts on, and 0 
	// otherwise.
	// That is to say, we append a 1 if the vertex is in U and use_vb is true,
	// or if the vertex is in V and use_vb is false. Otherwise, we append 0.
	if (use_vb) r = use_v ? (rankU[nbhr] << 1) + 0b1  : (rankV[nbhr] << 1);
	else r = use_v ? (rankU[nbhr] << 1) : (rankV[nbhr] << 1) + 0b1;
	// Include offset+j to allow for translation back to the original graph
	edges_convert[offsets[i]+j] = make_tuple(r,offset+j);
      });
    // Sort adjacency list of idx
    sampleSort(&edges_convert[offsets[i]], deg, ltx);
    // Set sorted offsets in the edges array for the ranked graph
    granular_for(j,0,deg,deg > 10000, {
	edges[offsets[i]+j] = get<0>(edges_convert[offsets[i]+j]); 
      });
  }

  // Create sorted graph
  return make_pair(graphCSR(offsets,edges,G.nv+G.nu,G.numEdges), edges_convert);
}

//********************************************************************************************
//********************************************************************************************

// Determines if color of an indicated edge is equal to specified color
template <class E>
struct clrF { 
  uintE* edges; uintT offset; uintT color; uintT* colors;
clrF(uintE* _edges, uintT _offset, uintT _color, uintT* _colors) :
  edges(_edges), offset(_offset), color(_color), colors(_colors) {}
  inline E operator() (const uintT& i) const {
    return (E) (colors[edges[offset+i]] == color ? 1 : 0); 
  }
};

// Determines if color of an indicated edges is 0
template <class E>
struct eF { 
  uintT offset; uintE* colors;
eF(uintT _offset, uintE* _colors) : offset(_offset), colors(_colors) {}
  inline E operator() (const uintT& i) const {
    return (E) (colors[offset+i] == 0);
  }
};

// Returns true if value is not max
struct nonMaxUintTF{bool operator() (uintT &a) {return (a != UINT_T_MAX);}};

// Determines if color of an indicated vertex is equal to the specified color
struct isSameColor{
isSameColor(uintT mycolor, uintT *colors) : colors_(colors), me_(mycolor) {};
  bool operator () (uintT v) {return me_ == colors_[v];};
  uintT me_;
  uintT *colors_;
};

/*
 *  Removes all singleton vertices from G.
 * 
 *  G: Bipartite graph in CSR format
 * 
 *  Returns a bipartite graph in CSR format with no singleton vertices.
 */
bipartiteCSR delZeroDeg(bipartiteCSR& G) {
  uintT* offsetsV_f = newA(uintT,G.nv+1);
  uintT* offsetsU_f = newA(uintT,G.nu+1);
  uintT* offsetsV_ff = newA(uintT,G.nv+1);
  uintT* offsetsU_ff = newA(uintT,G.nu+1);
  uintE* edgesV = newA(uintE, G.numEdges);
  uintE* edgesU = newA(uintE, G.numEdges);

  // Mark singleton vertices
  parallel_for(long i=0; i < G.nv; ++i) {
    if (G.offsetsV[i] == G.offsetsV[i+1]) offsetsV_f[i] = UINT_T_MAX; 
    else offsetsV_f[i] = i;
  }
  parallel_for(long i=0; i < G.nu; ++i) {
    if (G.offsetsU[i] == G.offsetsU[i+1]) offsetsU_f[i] = UINT_T_MAX;
    else offsetsU_f[i] = i;
  }

  // Filter out singleton vertices, leaving a list of vertices to keep
  long num_vff = sequence::filter(offsetsV_f,offsetsV_ff,G.nv,nonMaxUintTF());
  long num_uff = sequence::filter(offsetsU_f,offsetsU_ff,G.nu,nonMaxUintTF());

  // Create a mapping to rename vertices, so that indices are incremental
  parallel_for(long i=0; i < num_vff; ++i) {offsetsV_f[offsetsV_ff[i]] = i;}
  parallel_for(long i=0; i < num_uff; ++i) {offsetsU_f[offsetsU_ff[i]] = i;}
  // Rename all neighbors using the new mapping
  parallel_for(long i=0; i < G.numEdges; ++i) { edgesV[i] = offsetsU_f[G.edgesV[i]]; }
  parallel_for(long i=0; i < G.numEdges; ++i) { edgesU[i] = offsetsV_f[G.edgesU[i]]; }

  // Create new offsets list for our renamed vertices
  parallel_for(long i=0; i < num_vff; ++i) {
    offsetsV_ff[i] = G.offsetsV[offsetsV_ff[i]];
  }
  offsetsV_ff[num_vff] = G.offsetsV[G.nv];
  parallel_for(long i=0; i < num_uff; ++i) {
    offsetsU_ff[i] = G.offsetsU[offsetsU_ff[i]];
  }
  offsetsU_ff[num_uff] =  G.offsetsU[G.nu];

  free(offsetsV_f); free(offsetsU_f);
  return bipartiteCSR(offsetsV_ff, offsetsU_ff, edgesV, edgesU, num_vff, num_uff, G.numEdges);
}

/*
 *  Edge-sparsify G.
 * 
 *  G    : Bipartite graph in CSR format
 *  denom: Edges in G are kept with probability 1/denom
 *  seed : Random seed
 * 
 *  Returns a sparsified bipartite graph in CSR format.
 */
bipartiteCSR eSparseBipartite(bipartiteCSR& G, long denom, long seed) {
  // Determine which edges to keep by coloring with denom colors
  uintE numColors = max<uintE>(1,denom);
  uintE* colorsV = newA(uintE,G.numEdges);
  uintE* colorsU = newA(uintE,G.numEdges);
  parallel_for(long i=0; i < G.nv; ++i) {
    uintT v_offset = G.offsetsV[i];
    uintT v_deg = G.offsetsV[i+1] - v_offset;
    parallel_for(long j=0; j < v_deg; ++j) {
      uintT u = G.edgesV[v_offset + j];
      // Create a randomized color
      long concat = i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
      concat ^= u + 0x9e3779b9 + (seed << 6) + (seed >> 2);
      colorsV[v_offset + j] = hashInt((ulong) seed + concat) % numColors;
    }
  }
  // Keep colors indexed from G.U by searching for the corresponding edge
  // indexed from G.V
  parallel_for(intT i=0; i < G.nu; ++i) {
    intT u_offset = G.offsetsU[i];
    intT u_deg = G.offsetsU[i+1] - u_offset;
    granular_for(j,0,u_deg,u_deg>1000,{
	uintE v = G.edgesU[u_offset + j];
	intT v_offset = G.offsetsV[v];
	intT v_deg = G.offsetsV[v+1] - v_offset;
	// Find find_idx such that edgesV[v_offset + find_idx] = i
	auto idx_map = make_in_imap<uintE>(v_deg, [&] (size_t k) { return G.edgesV[v_offset + k]; });
	auto lte = [] (const uintE& l, const uintE& r) { return l < r; };
	size_t find_idx = pbbs::binary_search(idx_map, i, lte);
	colorsU[u_offset + j] = colorsV[v_offset + find_idx];
      });
  }

  // Compute new offsets after filtering out edges that aren't colored 0
  uintT* offsetsV = newA(uintT,G.nv+1);
  uintT* offsetsU = newA(uintT,G.nu+1);
  offsetsV[G.nv] = 0; offsetsU[G.nu] = 0;
  parallel_for(long i=0; i < G.nv; ++i) {
    uintT v_offset = G.offsetsV[i];
    uintT v_deg = G.offsetsV[i+1] - v_offset;
    if (v_deg > 10000) offsetsV[i] = sequence::reduce<uintT>((uintT) 0, v_deg, addF<uintT>(),
                                                             eF<uintT>(v_offset, colorsV));
    else {
      offsetsV[i] = 0;
      for(long j=0; j < v_deg; ++j) {
        if (colorsV[v_offset + j] == 0) offsetsV[i]++;
      }
    }
  }
  parallel_for(long i=0; i < G.nu; ++i) {
    uintT u_offset = G.offsetsU[i];
    uintT u_deg = G.offsetsU[i+1] - u_offset;
    if (u_deg > 10000) offsetsU[i] = sequence::reduce<uintT>((uintT) 0, u_deg, addF<uintT>(),
                                                             eF<uintT>(u_offset, colorsU));
    else {
      offsetsU[i] = 0;
      for(long j=0; j < u_deg; ++j) {
        if (colorsU[u_offset + j] == 0) offsetsU[i]++;
      }
    }
  }
  long mv = sequence::plusScan(offsetsV,offsetsV,G.nv+1);
  long mu = sequence::plusScan(offsetsU,offsetsU,G.nu+1);

  // Compute new adjacency lists after filtering out edges that aren't colored 0
  uintE* edgesV = newA(uintE,mv);
  uintE* edgesU = newA(uintE,mu);
  parallel_for(long i=0; i<G.nv; ++i) {
    uintT v_offset = G.offsetsV[i];
    uintT v_deg = G.offsetsV[i+1] - v_offset;
    uintT v_clr_offset = offsetsV[i];
    long idx = 0;
    for(long j=0; j < v_deg; ++j) {
      uintE u = G.edgesV[v_offset + j];
      if (colorsV[v_offset + j] == 0) {
	edgesV[v_clr_offset + idx] = u;
	idx++;
      }
    }
  }
  parallel_for(long i=0; i<G.nu; ++i) {
    uintT u_offset = G.offsetsU[i];
    uintT u_deg = G.offsetsU[i+1] - u_offset;
    uintT u_clr_offset = offsetsU[i];
    long idx = 0;
    for(long j=0; j < u_deg; ++j) {
      uintE v = G.edgesU[u_offset + j];
      if (colorsU[u_offset + j] == 0) {
	edgesU[u_clr_offset + idx] = v;
	idx++;
      }
    }
  }
  free(colorsV); free(colorsU);

  // Create new bipartite graph with filtered edges
  auto tmp = bipartiteCSR(offsetsV,offsetsU,edgesV,edgesU,G.nv,G.nu,mv);
  // Remove singleton vertices
  auto ret = delZeroDeg(tmp);
  tmp.del();

  return ret;
}

/*
 *  Color-sparsify G.
 * 
 *  G    : Bipartite graph in CSR format
 *  denom: Number of colors used to color G
 *  seed : Random seed
 * 
 *  Returns a sparsified bipartite graph in CSR format.
 */
bipartiteCSR clrSparseBipartite(bipartiteCSR& G, long denom, long seed) {
  double p = 1/denom;
  // Color vertices with denom colors
  uintT numColors = max<uintT>(1,denom);
  uintT* colorsV = newA(uintT,G.nv);
  uintT* colorsU = newA(uintT, G.nu);
  parallel_for(long i=0;i<G.nv;i++) colorsV[i] = hashInt((ulong) seed+i) % numColors;
  parallel_for(long i=0;i<G.nu;i++) colorsU[i] = hashInt((ulong) seed+i) % numColors;

  // Compute new offsets after filtering out edges whose endpoints don't share the same color
  uintT* offsetsV = newA(uintT,G.nv+1);
  uintT* offsetsU = newA(uintT,G.nu+1);
  offsetsV[G.nv] = 0; offsetsU[G.nu] = 0;
  parallel_for(long i=0; i < G.nv; ++i) {
    uintT v_offset = G.offsetsV[i];
    uintT v_deg = G.offsetsV[i+1] - v_offset;
    if (v_deg > 10000) offsetsV[i] = sequence::reduce<uintT>((uintT) 0, v_deg, addF<uintT>(),
                                                             clrF<uintT>(G.edgesV, v_offset, colorsV[i], colorsU));
    else {
      offsetsV[i] = 0;
      for(long j=0; j < v_deg; ++j) {
        if (colorsU[G.edgesV[v_offset + j]] == colorsV[i]) offsetsV[i]++;
      }
    }
  }
  parallel_for(long i=0; i < G.nu; ++i) {
    uintT v_offset = G.offsetsU[i];
    uintT v_deg = G.offsetsU[i+1] - v_offset;
    if (v_deg > 10000) offsetsU[i] = sequence::reduce<uintT>((uintT) 0, v_deg, addF<uintT>(),
                                                             clrF<uintT>(G.edgesU, v_offset, colorsU[i], colorsV));
    else {
      offsetsU[i] = 0;
      for(long j=0; j < v_deg; ++j) {
        uintE u = G.edgesU[v_offset + j];
        if (colorsV[u] == colorsU[i]) offsetsU[i]++;
      }
    }
  }
  long mv = sequence::plusScan(offsetsV,offsetsV,G.nv+1);
  long mu = sequence::plusScan(offsetsU,offsetsU,G.nu+1);

  // Compute new adjacency lists after filtering out edges whose endpoints don't share the same color
  uintE* edgesV = newA(uintE,mv);
  uintE* edgesU = newA(uintE,mu);
  parallel_for(long i=0; i<G.nv; ++i) {
    uintT v_offset = G.offsetsV[i];
    uintT v_deg = G.offsetsV[i+1] - v_offset;
    uintT v_clr_offset = offsetsV[i];
    if (v_deg > 10000) sequence::filter(&G.edgesV[v_offset], &edgesV[v_clr_offset], v_deg,
                                        isSameColor(colorsV[i],colorsU));
    else {
      long idx = 0;
      for(long j=0; j < v_deg; ++j) {
        uintE u = G.edgesV[v_offset + j];
        if (colorsU[u] == colorsV[i]) {
          edgesV[v_clr_offset + idx] = u;
          idx++;
        }
      }
    }
  }
  parallel_for(long i=0; i<G.nu; ++i) {
    uintT v_offset = G.offsetsU[i];
    uintT v_deg = G.offsetsU[i+1] - v_offset;
    uintT v_clr_offset = offsetsU[i];
    if (v_deg > 10000) sequence::filter(&G.edgesU[v_offset], &edgesU[v_clr_offset], v_deg,
                                        isSameColor(colorsU[i],colorsV));
    else {
      long idx = 0;
      for(long j=0; j < v_deg; ++j) {
        uintE u = G.edgesU[v_offset + j];
        if (colorsV[u] == colorsU[i]) {
          edgesU[v_clr_offset + idx] = u;
          idx++;
        }
      }
    }
  }
  free(colorsV); free(colorsU);

  // Create new bipartite graph with filtered edges
  auto tmp = bipartiteCSR(offsetsV,offsetsU,edgesV,edgesU,G.nv,G.nu,mv);
  // Remove singleton vertices
  auto ret = delZeroDeg(tmp);
  tmp.del();

  return ret;
}

//********************************************************************************************
//********************************************************************************************

/*
 *  Read bipartite graph from file fname, in Ligra hypergraph format.
 * 
 *  fname: File containing bipartite graph
 * 
 *  Returns a bipartite graph in CSR format.
 */
bipartiteCSR readBipartite(char* fname) {
  words W;
  _seq<char> S = readStringFromFile(fname);
  W = stringToWords(S.A, S.n);

  if (W.Strings[0] != (string) "AdjacencyHypergraph") {
    cout << "Bad input file" << endl;
    abort();
  }

  long len = W.m -1;
  long nv = atol(W.Strings[1]);
  long mv = atol(W.Strings[2]);
  long nu = atol(W.Strings[3]);
  long mu = atol(W.Strings[4]);

  if ((len != nv + mv + nu + mu + 4) | (mv != mu)) {
    cout << "Bad input file" << endl;
    abort();
  }

  uintT* offsetsV = newA(uintT,nv+1);
  uintT* offsetsU = newA(uintT,nu+1);
  uintE* edgesV = newA(uintE,mv);
  uintE* edgesU = newA(uintE,mu);

  {parallel_for(long i=0; i < nv; i++) offsetsV[i] = atol(W.Strings[i + 5]);}
  offsetsV[nv] = mv;
  
  {parallel_for(long i=0; i<mv; i++) {
      edgesV[i] = atol(W.Strings[i+nv+5]);
      if(edgesV[i] < 0 || edgesV[i] >= nu) {
	cout << "edgesV out of range: nu = " << nu << " edge = " << edgesV[i] << endl; exit(0);
      }
    }}

  {parallel_for(long i=0; i < nu; i++) offsetsU[i] = atol(W.Strings[i + nv + mv + 5]);}
  offsetsU[nu] = mu;
  
  {parallel_for(long i=0; i<mu; i++) {
      edgesU[i] = atol(W.Strings[i+nv+mv+nu+5]);
      if(edgesU[i] < 0 || edgesU[i] >= nv) {
	cout << "edgesU out of range: nv = " << nv << " edge = " << edgesU[i] << endl; exit(0);
      }
    }}

  S.del();
  free(W.Strings);

  return bipartiteCSR(offsetsV,offsetsU,edgesV,edgesU,nv,nu,mv);  
}

// Returns the outdegree choose 2 of a given vertex
template <class E>
struct wedgeF { 
  uintT* offsets;
wedgeF(uintT* _offsets) : offsets(_offsets) {}
  inline E operator() (const uintT& i) const {
    uintE v_deg = offsets[i+1]-offsets[i];
    return (E) ((v_deg * (v_deg-1)) / 2); 
  }
};

/*
 *  Computes the number of wedges given by each bipartition.
 * 
 *  GA: Bipartite graph in CSR format
 * 
 *  Returns a bool indicating which bipartition gives the least number of wedges
 *  when considered as center vertices (true means GA.V, false means GA.U),
 *  and the corresponding number of wedges produced.
 */
tuple<bool,long> cmpWedgeCounts(bipartiteCSR & GA) {
  const long nv = GA.nv, nu = GA.nu;
  long num_wedges_v = sequence::reduce<long>((long) 0, nv, addF<long>(), wedgeF<long>(GA.offsetsV));
  long num_wedges_u = sequence::reduce<long>((long) 0, nu, addF<long>(), wedgeF<long>(GA.offsetsU));

  return make_tuple((num_wedges_v <= num_wedges_u), num_wedges_v <= num_wedges_u ? num_wedges_v : num_wedges_u);
}

//***********************************************************************************************
//***********************************************************************************************

// Function that returns true if given long isn't max
struct nonMaxLongF{bool operator() (long &a) {return (a != LONG_MAX);}};

/*
 *  Given a sorted array objs, retrieve indices of where consecutive objects are
 *  not equal (as given by eq) to give frequency counts of each object in objs.
 *  In other words, if we let arr be the returned array, arr[i+1] - arr[i] is
 *  the frequency of objs[arr[i]] (objs[arr[i]] = ... = objs[arr[i+1]-1]).
 *  Iterating from 0 (inclusive) to the returned length - 1 (exclusive) will
 *  give all frequency counts.
 * 
 *  objs: Objects to count frequencies of
 *  num : Length of objs
 *  cmp : Comparator for T objects
 *  eq  : Equality comparator for T objects
 *  maxL: A count of type L that exceeds the maximum frequency of elements in objs
 *  nonF: Function that filters out all elements equal to maxL
 *  sort: Deprecated
 * 
 *  Returns frequency array and length of array (as described above)
 */
template <class L, class T, class Cmp, class Eq, class F>
  pair<L*, long> getFreqs(T* objs, long num, Cmp cmp, Eq eq, L maxL, F nonF, bool sort=true) {
  L* freqs = newA(L, num + 1);
  freqs[0] = 0;
  freqs[num] = num;

  // Retrieve indices where objects differ
  parallel_for(long i=1; i < num; ++i) {
    if (!eq(objs[i-1],objs[i])) freqs[i] = i;
    else freqs[i] = maxL;
  }
  L* freqs_f = newA(L, num+1);
  long num_freqs_f = sequence::filter(freqs, freqs_f, num+1, nonF);

  free(freqs);

  return make_pair(freqs_f, num_freqs_f);
}

/*
 *  Given a sorted array objs, retrieve indices of where consecutive objects are
 *  not equal (as given by eq) to give frequency counts of each object in objs.
 *  In other words, if we let arr be the returned array,
 *  get<1>(arr[i+1]) - get<1>(arr[i]) is the frequency of get<0>(arr[i]).
 *  Note that the first element of each tuple in the returned array is the
 *  original object, with opt applied. Frequency counts are given using opcount
 *  and opuinte, as described below.
 *  Iterating from 0 (inclusive) to the returned length - 1 (exclusive) will
 *  give all frequency counts.
 * 
 *  objs   : Objects to count frequencies of
 *  num    : Length of objs
 *  cmp    : Comparator for T objects
 *  eq     : Equality comparator for T objects
 *  sort   : Deprecated
 *  opt    : Function applied to each object in objs to save in a tuple returned frequency array
 *  opuinte: Function to transform total frequency count and save in a tuple in returned frequency array
 *  opcount: Function to apply to object to increment frequency counts by (default is to add 1)
 * 
 *  Returns frequency array and length of array (as described above)
 */
template <class L, class S, class T, class Cmp, class Eq, class OpT, class OpuintE, class OpCount>
  pair<tuple<S,L>*, long> getFreqs_seq(T* objs, long num, Cmp cmp, Eq eq, bool sort=true, OpT opt=refl<T>(),
				       OpuintE opuinte=refl<L>(), OpCount opcount=reflCount<T>()) {
  using X = tuple<S,L>;
  X* freqs = newA(X, num);
  T prev = objs[0];
  T curr = objs[0];
  long idx = 0;
  long count = opcount(prev);

  // Retrieve frequency counts of each object
  for(long i=1; i < num; ++i) {
    curr = objs[i];
    if (!eq(prev, curr)) {
      freqs[idx] = make_tuple(opt(prev), opuinte(count));
      idx++;
      count = opcount(curr);
      prev = curr;
    }
    else {
      count += opcount(curr);
    }
  }
  freqs[idx] = make_tuple(opt(curr), opuinte(count));

  return make_pair(freqs, idx + 1);
}

//********************************************************************************************
//********************************************************************************************

/*
 *  Computes indices of all wedges originating from each vertex, so that
 *  they can be retrieved in parallel.
 * 
 *  G: Ranked graph in CSR format
 * 
 *  Returns wedge indices to allow for wedge retrieval in parallel
 */
long* countWedgesScan(graphCSR& G) {
  long* idxs = newA(long, G.n + 1);
  idxs[G.n] = 0;
  using T = long*;
  T* nbhd_idxs = newA(T, G.n);
  // Iterate through each vertex i
  parallel_for(intT i=0; i < G.n; ++i) {
    idxs[i] = 0;
    intT offset = G.offsets[i];
    intT deg = G.offsets[i+1] - offset;

    // Construct a new array to keep track of two-hop degrees
    nbhd_idxs[i] = newA(long, deg + 1);
    (nbhd_idxs[i])[deg] = 0;

    parallel_for(intT j=0; j < deg; ++j) {
      (nbhd_idxs[i])[j] = 0;
      uintE v = G.edges[offset+j] >> 1;
      intT v_offset = G.offsets[v];
      intT v_deg = G.offsets[v+1] - v_offset;
#ifdef INVERSE
      {
#else
      if (v > i) {
#endif
	// Iterate through all two-hop neighbors and increment degrees
	for (intT k = 0; k < v_deg; ++k) {
#ifdef INVERSE
    if ((G.edges[v_offset + k] >> 1) < std::min((uintE) i, v))
#else
	  if ((G.edges[v_offset + k] >> 1) > i)
#endif
      (nbhd_idxs[i][j])++;
	  else break;
	}
      }
    }

    // Compute total number of two-hop neighbors
    idxs[i] = sequence::plusReduce(nbhd_idxs[i], deg + 1);
    free(nbhd_idxs[i]);
  }
  free(nbhd_idxs);

  // Use a scan to compute indices
  sequence::plusScan(idxs, idxs, G.n + 1);
  return idxs;
}

/*
 *  Computes indices of all wedges originating from each vertex, so that
 *  they can be retrieved in parallel.
 * 
 *  GA   : Bipartite graph in CSR format
 *  use_v: Denotes which bipartition produces the fewest wedges. If true, considering two-hop neighbors of U produces
 *         the fewest wedges. If false, considering two-hop neighbors of V produces the fewest wedges.
 *  half : Denotes whether to consider all two-hop neighbors (including duplicates, i.e. u is a two-hop neighbor of v 
 *         and as such v is also a two-hop neighbor of u), or only distinct two-hop neighbors
 * 
 *  Returns wedge indices to allow for wedge retrieval in parallel
 */
long* countWedgesScan(bipartiteCSR& GA, bool use_v, bool half=false) {
  const long nu = use_v ? GA.nu : GA.nv;
  uintT* offsetsV = use_v ? GA.offsetsV : GA.offsetsU;
  uintT* offsetsU = use_v ? GA.offsetsU : GA.offsetsV;
  uintE* edgesV = use_v ? GA.edgesV : GA.edgesU;
  uintE* edgesU = use_v ? GA.edgesU : GA.edgesV;

  long* idxs = newA(long, nu + 1);
  idxs[nu] = 0;

  using T = long*;
  T* nbhd_idxs = newA(T, nu);

  // Iterate through each vertex i
  parallel_for(intT i=0; i < nu; ++i) {
    idxs[i] = 0;
    intT u_offset = offsetsU[i];
    intT u_deg = offsetsU[i+1] - u_offset;

    // Construct a new array to keep track of two-hop degrees
    nbhd_idxs[i] = newA(long, u_deg + 1);
    parallel_for(intT j=0; j < u_deg+1; ++j) {(nbhd_idxs[i])[j] = 0;}

    // Iterate through all two-hop neighbors and increment degrees
    parallel_for(intT j=0; j < u_deg; ++j) {
      if (!half) {
        uintE v = edgesU[u_offset + j];
        (nbhd_idxs[i])[j] = offsetsV[v+1] - offsetsV[v] - 1;
      }
      else {
        (nbhd_idxs[i])[j] = 0;
        uintE v = edgesU[u_offset + j];
        intT v_offset = offsetsV[v];
        intT v_deg = offsetsV[v+1] - v_offset;
        for (intT k = 0; k < v_deg; ++k) {
          if (edgesV[v_offset + k] < i) nbhd_idxs[i][j] ++;
          else break;
        }
      }
    }

    // Compute total number of two-hop neighbors
    idxs[i] = sequence::plusReduce(nbhd_idxs[i], u_deg + 1);
    free(nbhd_idxs[i]);
  }
  free(nbhd_idxs);

  // Use a scan to compute indices
  sequence::plusScan(idxs, idxs, nu + 1);

  return idxs;
}

//********************************************************************************************
//********************************************************************************************

// Allocates and keeps all space needed for edge counting algorithms, so that
// space can be reused between batches
struct CountESpace {
  CountType type; long nu; bool rank;
  // Sort, ASort
  _seq<UWedge> wedges_seq_uw;
  // Sort, Hist
  _seq<tuple<uintE,long>> butterflies_seq_intt;
  // Hist
#ifndef OPENMP
  pbbsa::sequence<tuple<uintE, long>> tmp_uint;
  pbbsa::sequence<tuple<uintE, long>> out_uint;
  // AHist
  pbbsa::sequence<tuple<long, uintE>> tmp;
  pbbsa::sequence<tuple<long, uintE>> out;
#endif
  _seq<long> wedges_seq_int;
  // AHist, Hash, AHash
  sparseAdditiveSet<long, long> wedges_hash;
  // Hash
  _seq<pair<long, long>> wedges_seq_intp;
  sparseAdditiveSet<long, long> butterflies_hash;


CountESpace(CountType _type, long _nu, bool _rank) : type(_type), nu(_nu), rank(_rank) {
  using T = pair<long,long>;
  using X = tuple<uintE,long>;
  using E = pair<long, uintE>;
  if (type == HASH || type == AHASH) {
    wedges_hash = sparseAdditiveSet<long, long>(nu,1,LONG_MAX, LONG_MAX);
    if (type == HASH) {
      wedges_seq_intp = _seq<T>(newA(T, nu), nu);
      butterflies_hash = sparseAdditiveSet<long, long>(nu, 1, LONG_MAX, LONG_MAX);
    }
  }
#ifndef OPENMP
  else if (type == HIST || type == AHIST) {
    tmp = pbbsa::sequence<tuple<long, uintE>>();
    out = pbbsa::sequence<tuple<long, uintE>>();
    wedges_seq_int = _seq<long>(newA(long, nu), nu);
    wedges_hash = sparseAdditiveSet<long, long>(nu,1,LONG_MAX, LONG_MAX);
    if (type == HIST) {
      butterflies_seq_intt = _seq<X>(newA(X, 1), 1);
      tmp_uint = pbbsa::sequence<tuple<uintE, long>>();
      out_uint = pbbsa::sequence<tuple<uintE, long>>();
    }
  }
#endif
  else if (type == SORT || type == ASORT) {
    if (type == SORT) butterflies_seq_intt = _seq<X>(newA(X, 1), 1);
    wedges_seq_uw = _seq<UWedge>(newA(UWedge, nu), nu);
  }
}

  /*
   *  Clears necessary space, to be reused.
   */
  void clear() {
    if (type == HASH || type == AHASH || type == HIST || type == AHIST) wedges_hash.clear();
    if (type == HASH) butterflies_hash.clear();
  }

  /*
   *  Frees all space.
   */
  void del() {
    if (type == HASH || type == AHASH) {
      wedges_hash.del();
      if (type == HASH) { wedges_seq_intp.del(); butterflies_hash.del(); }
    }
    else if (type == HIST || type == AHIST) {
      wedges_seq_int.del();
      wedges_hash.del();
      if (type == HIST) butterflies_seq_intt.del();
    }
    else if (type == SORT || type == ASORT) {
      if (type == SORT) butterflies_seq_intt.del();
      wedges_seq_uw.del();
    }
  }
};

// Only API compatibility remains; contribution implementations are absent.
#include <stdexcept>
struct CountSpace {
CountType type; long nu; bool rank;
  sparseAdditiveSet<long, long> wedges_hash;
  _seq<pair<long,long>> wedges_seq_intp;
  sparseAdditiveSet<long, long> butterflies_hash;
  _seq<pair<long,long>> butterflies_seq_intp;
#ifndef OPENMP
  pbbsa::sequence<tuple<long, uintE>> tmp;
  pbbsa::sequence<tuple<long, uintE>> out;
#endif
  _seq<long> wedges_seq_int;
  _seq<tuple<uintE,long>> butterflies_seq_intt;
#ifndef OPENMP
  pbbsa::sequence<tuple<uintE, long>> tmp_uint;
  pbbsa::sequence<tuple<uintE, long>> out_uint;
#endif
  _seq<UVertexPair> wedges_seq_uvp;
  _seq<UWedge> wedges_seq_uw;
  CountSpace(CountType t, long n, bool r) : type(t), nu(n), rank(r) {
    if (t != SERIAL && t != BATCHS)
      throw std::runtime_error("counting workspace implementation is not supplied");
  }
  void clear() { throw std::runtime_error("counting workspace implementation is not supplied"); }
  void del() {}
};

template<class wedge, class wedgeCons>
  void _getWedges_seq(wedge* wedges, bipartiteCSR& GA, bool use_v, wedgeCons cons, long num_wedges, intT curr_idx,
		      intT next_idx) {
  throw std::runtime_error("wedge batching implementation is not supplied");
}

template<class wedge, class wedgeCons>
  void _getWedges(_seq<wedge>& wedges_seq, bipartiteCSR& GA, bool use_v, wedgeCons cons, long num_wedges,
		  long* wedge_idxs, intT curr_idx=0, intT next_idx=INT_T_MAX) {
  throw std::runtime_error("wedge batching implementation is not supplied");
}

template<class wedgeCons, class T>
  void _getWedgesHash(T& wedges, bipartiteCSR& GA, bool use_v, wedgeCons cons, long num_wedges, intT curr_idx=0,
		      intT next_idx=INT_T_MAX) {
  throw std::runtime_error("wedge batching implementation is not supplied");
}

intT getNextWedgeIdx_seq(bipartiteCSR& GA, bool use_v, long max_wedges, intT curr_idx) {
  throw std::runtime_error("wedge batching implementation is not supplied");
}

intT getNextWedgeIdx(bipartiteCSR& GA, bool use_v, long max_wedges, intT curr_idx, long* wedge_idxs) {
  throw std::runtime_error("wedge batching implementation is not supplied");
}

template<class wedge, class wedgeCons>
  pair<long, intT> getWedges(_seq<wedge>& wedges, bipartiteCSR& GA, bool use_v, wedgeCons cons, long max_wedges,
			     intT curr_idx, long num_wedges, long* wedge_idxs) {
  throw std::runtime_error("wedge batching implementation is not supplied");
}

template<class wedgeCons, class T>
  intT getWedgesHash(T& wedges, bipartiteCSR& GA, bool use_v, wedgeCons cons, long max_wedges, intT curr_idx,
		     long num_wedges, long* wedge_idxs) {
  throw std::runtime_error("wedge batching implementation is not supplied");
}

template<class wedge, class wedgeCons>
  void _getWedges_seq(wedge* wedges, graphCSR& GA, wedgeCons cons, long num_wedges, intT curr_idx, intT next_idx) {
  throw std::runtime_error("wedge batching implementation is not supplied");
}

template<class wedge, class wedgeCons>
  void _getWedges(_seq<wedge>& wedges_seq, graphCSR& GA, wedgeCons cons, long num_wedges, long* wedge_idxs,
		  intT curr_idx=0, intT next_idx=INT_T_MAX) {
  throw std::runtime_error("wedge batching implementation is not supplied");
}

template<class wedgeCons, class T>
  void _getWedgesHash(T& wedges, graphCSR& GA, wedgeCons cons, long num_wedges, intT curr_idx=0,
		      intT next_idx=INT_T_MAX) {
  throw std::runtime_error("wedge batching implementation is not supplied");
}

intT getNextWedgeIdx(graphCSR& GA, long max_wedges, intT curr_idx, long* wedge_idxs) {
  throw std::runtime_error("wedge batching implementation is not supplied");
}

template<class wedgeCons, class T>
  intT getWedgesHash(T& wedges, graphCSR& GA, wedgeCons cons, long max_wedges, intT curr_idx, long num_wedges,
		     long* wedge_idxs) {
  throw std::runtime_error("wedge batching implementation is not supplied");
}

template<class wedge, class wedgeCons>
  pair<long, intT> getWedges(_seq<wedge>& wedges, graphCSR& GA, wedgeCons cons, long max_wedges, intT curr_idx,
			     long num_wedges, long* wedge_idxs) {
  throw std::runtime_error("wedge batching implementation is not supplied");
}

//***************************************************************************************************
//***************************************************************************************************

/*
 *  Construct an array that converts edge indices from one orientation (as
 *  given by use_v) to the other. If use_v is true, then the array
 *  takes edges indexed through GA.edgesU and maps them to edges indexed
 *  through GA.edgesV. Otherwise, the mapping is the opposite way around.
 * 
 *  GA   : Bipartite graph in CSR format
 *  use_v: Denotes which orientation to map edge indices
 * 
 *  Returns an array that converts edge indices from one orientation to the
 *  other, as given by use_v.
 */
uintE* edgeToIdxs(bipartiteCSR& GA, bool use_v) {
  long nu = use_v ? GA.nu : GA.nv;
  long nv = use_v ? GA.nv : GA.nu;
  uintT* offsetsV = use_v ? GA.offsetsV : GA.offsetsU;
  uintT* offsetsU = use_v ? GA.offsetsU : GA.offsetsV;
  uintE* edgesV = use_v ? GA.edgesV : GA.edgesU;
  uintE* edgesU = use_v ? GA.edgesU : GA.edgesV;

  uintE* eti = newA(uintE, GA.numEdges);

  // Iterate through all edges
  parallel_for(intT i=0; i < nu; ++i) {
    intT u_offset = offsetsU[i];
    intT u_deg = offsetsU[i+1] - u_offset;
    granular_for(j,0,u_deg,u_deg>1000,{
	uintE v = edgesU[u_offset + j];
	intT v_offset = offsetsV[v];
	intT v_deg = offsetsV[v+1] - v_offset;
	// Find the corresponding edge from the other orientation
	// Find find_idx such that edgesV[v_offset + k] = i
	auto idx_map = make_in_imap<uintE>(v_deg, [&] (size_t k) { return edgesV[v_offset + k]; });
	auto lte = [] (const uintE& l, const uintE& r) { return l < r; };
	size_t find_idx = pbbs::binary_search(idx_map, i, lte); 
	eti[u_offset+j] = v_offset + find_idx;
      });
  }

  return eti;
}

// Precisely edgeToIdxs, but inverting use_v
uintE* idxsToEdge(bipartiteCSR& G, bool use_v) {
  return edgeToIdxs(G, !use_v);
}

#endif
