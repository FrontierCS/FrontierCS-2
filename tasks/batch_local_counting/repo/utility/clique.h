//
// adapted from https://github.com/Gawssin/kCliqueListing/blob/master/DDegCol/DDegCol.c
//

#ifndef SCOPE_CLIQUE_H
#define SCOPE_CLIQUE_H

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include "config.h"

#define NLINKS 100000000 //maximum number of edges for memory allocation, will increase if needed

typedef struct {
    unsigned s;
    unsigned t;
} edge;

typedef struct {
    unsigned node;
    unsigned deg;
} nodedeg;


typedef struct {

    unsigned n;//number of nodes
    unsigned e;//number of edges
    edge *edges;//list of edges

    unsigned *ns;//ns[l]: number of nodes in G_l
    unsigned **d;//d[l]: degrees of G_l
    unsigned *cd, *cdsub;//cumulative degree: (starts with 0) length=n+1
    unsigned *adj, *adjsub;//truncated list of neighbors
    unsigned *rank;//ranking of the nodes according to degeneracy ordering
    //unsigned *map;//oldID newID correspondance

    unsigned char *lab;//lab[i] label of node i
    unsigned **sub;//sub[l]: nodes in G_l

} specialsparse;

typedef struct {
    unsigned id;
    unsigned degree;
} iddegree;

///// CORE ordering /////////////////////

typedef struct {
    unsigned key;
    unsigned value;
} keyvalue;

typedef struct {
    unsigned n_max;	// max number of nodes.
    unsigned n;	// number of nodes.
    unsigned *pt;	// pointers to nodes.
    keyvalue *kv; // nodes.
} bheap;

static iddegree *ig;
static specialsparse *subg;
static int *color, *ind, *loc, *C;
static unsigned *cd0, *adj0, *dsub, *Index;

int cmp(const void* a, const void* b);
int cmpadj(const void* a, const void* b);
void freespecialsparse(specialsparse *g, unsigned char k);
void freesub(specialsparse *g, unsigned char k);
unsigned int max3(unsigned int a, unsigned int b, unsigned int c);
void relabel(specialsparse *g);
bheap *construct(unsigned n_max);
void swap(bheap *heap, unsigned i, unsigned j);
void bubble_up(bheap *heap, unsigned i);
void bubble_down(bheap *heap);
void insert(bheap *heap, keyvalue kv);
void update(bheap *heap, unsigned key);
keyvalue popmin(bheap *heap);
bheap* mkheap(unsigned n, unsigned *v);
void freeheap(bheap *heap);
void ord_core(specialsparse* g);
void mkspecial(specialsparse *g, unsigned char k);
void mkspecial_sub(specialsparse *g, unsigned char k);
void kclique(int l, int K, specialsparse *g, unsigned *vertices, HashTable h, int orbitType);

#endif //SCOPE_CLIQUE_H
