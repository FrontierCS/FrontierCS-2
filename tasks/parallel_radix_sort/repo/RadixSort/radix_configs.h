#ifndef _S_CONFIGS_INCLUDED
#define _S_CONFIGS_INCLUDED

#define ILP_SORT
#define MAXP 100
#define MAXBITS 100
#ifndef MAX_RADIX
#define MAX_RADIX 8

#endif
#if MAX_RADIX > 8
typedef uint16_t radixCustomType; 
#elif MAX_RADIX <= 8 
typedef uint8_t radixCustomType;
#endif

#ifdef LONG_ARRAY
typedef long sizeT;
#else
typedef int sizeT;
#endif


#define BUCKETS (1 << MAX_RADIX)
#define MAXDEPTH (MAXBITS/MAX_RADIX)
#define RANDOM_GRAPH 0

#ifndef BLOCK_DIVIDE
#define BLOCK_DIVIDE 800000
#endif

#ifndef K_BOUND
#define K_BOUND 20000
#endif


#define CYC_ALG 3
//#define CYCLE

#define SERIAL_THRESHOLD 20000


#define INSERTION_COARSEN

#define PARALLEL_FOR_THRESHOLD 4000

#ifdef INSERTION_COARSEN
#define INSERTION_THRESHOLD 32
#endif

#ifndef STABLE
#define EXTRACT_2CYCLES
#endif

//#define LONG

//#define PARALLEL_EXTRACT
//#define PARALLEL_CONSUME

#define MERGE_MAX 100000

#ifndef TIMER
//#define TIMER
#endif

long global_K;
long global_N;

#endif


