#ifndef NOREPEAT_H
#define NOREPEAT_H

#include "DBSCAN.h"

// use union find (common/unionFind.h)

struct unionFind {
  GdLongDataType* parents;

  // initialize with all roots marked with -1
  unionFind(GdLongDataType n) {
    parents = (GdLongDataType*) malloc(sizeof(GdLongDataType)*n);
    parallel_for(GdLongDataType i=0; i < n; i++) parents[i] = -1;
  }

  void del() {free(parents);}

  GdLongDataType find(GdLongDataType i) {
    if (parents[i] < 0) return i;
    GdLongDataType j = parents[i];     
    if (parents[j] < 0) return j;
    do j = parents[j]; 
    while (parents[j] >= 0);
    GdLongDataType tmp;
    while ((tmp = parents[i]) != j) { 
      parents[i] = j;
      i = tmp;
    }
    return j;
  }

  void link(GdLongDataType u, GdLongDataType v) { 
    parents[find(u)] = find(v);}
};

struct parallelUnionFind {

  GdLongDataType *parents;
  GdLongDataType *hooks;

  // initialize with all roots marked with -1
  parallelUnionFind(GdLongDataType n) {
    parents = newA(GdLongDataType,n);
    parallel_for (GdLongDataType i=0; i < n; i++) parents[i] = GDLONG_MAX;
    hooks = newA(GdLongDataType,n);
    parallel_for (GdLongDataType i=0; i < n; i++) hooks[i] = GDLONG_MAX;
  }

  void del() {free(parents);}

  // Assumes root is negative 
  // Not making parent array volatile improves
  // performance and doesn't affect correctness
  inline GdLongDataType find(GdLongDataType i) {
    GdLongDataType j = i;
    if (parents[j] == GDLONG_MAX) return j;
    do j = parents[j];
    while (parents[j] < GDLONG_MAX);
    //note: path compression can happen in parallel in the same tree, so
    //only link from smaller to larger to avoid cycles
    GdLongDataType tmp;
    while((tmp=parents[i])<j){ parents[i]=j; i=tmp;} 
    return j;
  }

  void link(GdLongDataType u, GdLongDataType v) {
    while(1){
      u = find(u);
      v = find(v);
      if(u == v) break;
      if(u > v) swap(u,v);
      //if successful, store the ID of the edge used in hooks[u]
      if(hooks[u] == GDLONG_MAX && __sync_bool_compare_and_swap(&hooks[u],GDLONG_MAX,u)){
        parents[u]=v;
        break;
      }
    }
  }

};

inline GdLongDataType UFFindParallel(GdLongDataType i, GdLongDataType * parent) {
  GdLongDataType j = i;
  if (parent[j] < 0) return j;
  do j = parent[j];
  while (parent[j] >= 0);
  //note: path compression can happen in parallel in the same tree, so
  //only link from smaller to larger to avoid cycles
  GdLongDataType tmp;
  while((tmp=parent[i])<j){ parent[i]=j; i=tmp;}
  return j;
}

inline GdLongDataType *OneShotParallelUnionFind(GdLongDataType *t_edgeIndice, OtLongDataType m, GdLongDataType n){
  GdLongDataType *parents = newA(GdLongDataType,n);
  parallel_for (GdLongDataType i=0; i < n; i++) parents[i] = -1;
  GdLongDataType *hooks = newA(GdLongDataType,n);
  parallel_for (GdLongDataType i=0; i < n; i++) hooks[i] = GDLONG_MAX;
  parallel_for (OtLongDataType i = 0; i < m * 2; i += 2) {
    GdLongDataType u = t_edgeIndice[i], v = t_edgeIndice[i + 1];
    while(1){
      u = UFFindParallel(t_edgeIndice[i], parents);
      v = UFFindParallel(t_edgeIndice[i + 1], parents);
      if(u == v) break;
      if(u > v) swap(u,v);
      //if successful, store the ID of the edge used in hooks[u]
      if(hooks[u] == GDLONG_MAX && __sync_bool_compare_and_swap(&hooks[u],GDLONG_MAX,i)){
        parents[u]=v;
        break;
      }
    }
  }
  free(hooks);
  return parents;
}

#endif

