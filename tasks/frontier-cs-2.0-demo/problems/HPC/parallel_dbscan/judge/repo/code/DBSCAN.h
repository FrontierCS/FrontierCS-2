#ifndef DBSCAN_H_
#define DBSCAN_H_

#define DOUBLE_DATATYPE // modify geometry.h NDGCTYPE accordingly
//#define FLOAT_DATATYPE // modify geometry.h NDGCTYPE accordingly

#define LONG_VARIABLE
// #define USE_JEMALLOC

#include "utils.h"
#include "sequence.h"
#include "geometry.h"
#if defined (USE_JEMALLOC)
#include "je_allocator.h"
#endif
#include "ndHash.h"
#include <unordered_set>

/////////////// Fixed Configs ///////////////
#define COMPONENT_TIMER
#define PARSER_OUTPUT
// #define CLOSEST_BORDER_CLUSTER // enable mainly for error checker

#if !defined CLOSEST_BORDER_CLUSTER
#define OUTPUT_TO_FILE
#endif

#if defined(DOUBLE_DATATYPE)
typedef double PtDataType;
typedef double PtIntermediateDataType;
#elif defined(FLOAT_DATATYPE)
typedef float PtDataType;
typedef float PtIntermediateDataType;
#define IS_INTSIZE_INTERMEDIATE_DATA
#endif // ifdef double or float

#ifdef LONG_VARIABLE
typedef long PtLongDataType; // must hold number of points
typedef PtLongDataType OtLongDataType;
typedef long GdPtLongDataType; // must hold points in any grid
typedef long GdLongDataType; // must hold number of valid grids
#define GDLONG_MAX LONG_MAX
#else
typedef intT PtLongDataType;
typedef intT GdLongDataType;
typedef intT GdPtLongDataType;
typedef intT OtLongDataType;
#define GDLONG_MAX INT_MAX
#endif // ifdef long var

#ifndef CLOSEST_BORDER_CLUSTER
/* when finding all clusters of a border point (disabled option above), enable to (optionally, controlled by arg -o) output multiple clusters as file and information about the clusters printed on screen. when disabled, the output format will be in identical format to disabling CLOSEST_BORDER_CLUSTER, but border point will be randomly placed in one of its clusters (mostly for debugging of multi-border-cluster code).
for pbbs users, the checker will crash if this option is enabled, this option is meant for end user who do not use the checker */
#define MULTI_CLUSTER_OUTPUT
#endif

/////////////// Fixed Configs ///////////////

// param def
struct DbscanParams {
  PtIntermediateDataType m_epsilon;
  int m_minPts;
  PtIntermediateDataType m_rho; // only used for approximate dbscan
};

// one of the return types
#if defined(USE_JEMALLOC)
typedef std::unordered_set<GdLongDataType, std::hash<GdLongDataType>, std::equal_to<GdLongDataType>, je_allocator<GdLongDataType> > ClusterContainer;
#else
typedef std::unordered_set<GdLongDataType> ClusterContainer;
#endif

/*
struct hashBorderCluster {
  typedef pair<PtLongDataType, ClusterContainer *> eType;
  typedef PtLongDataType kType;
  eType empty() {
    return pair<PtLongDataType, ClusterContainer *>(-1,(ClusterContainer *)NULL);
  }
  kType getKey(eType v) { return v.first; }
  PtLongDataType hash(kType s) {
    return s;
  }
  int cmp(kType g1, kType g2) {
    if (g1 > g2) {
      return 1;
    } else if (g1 < g2) {
      return -1;
    } else {
      return 0;
    }
  }
  bool replaceQ(eType s, eType s2) {return 0;}
};
*/

// pair<CoreFlags, ClusterIds>
#if defined CLOSEST_BORDER_CLUSTER
pair<intT*, GdLongDataType*> DBSCAN(_seq<pointNd> t_P, DbscanParams t_params);
#else
//pair<GdLongDataType*, Table<hashBorderCluster, OtLongDataType>* > DBSCAN(_seq<pointNd> t_P, DbscanParams t_params);
pair<GdLongDataType*, ClusterContainer ** > DBSCAN(_seq<pointNd> t_P, DbscanParams t_params);
#endif

// output a parser recognizable quantity
inline void PrintParser(const std::string& keyword, double content) {
#ifdef PARSER_OUTPUT
  cout << ">" << keyword << "*" << content << "*" << endl;
#endif
  //printf(">%s*%4.3f*\n", keyword.c_str(), content);
}

inline void PrintParser(const std::string& keyword, int content) {
#ifdef PARSER_OUTPUT
  cout << ">" << keyword << "*" << content << "*" << endl;
#endif
}

inline void PrintParser(const std::string& keyword, long content) {
#ifdef PARSER_OUTPUT
  cout << ">" << keyword << "*" << content << "*" << endl;
#endif
}

inline void PrintParser(const std::string& keyword, const std::string& content) {
#ifdef PARSER_OUTPUT
  cout << ">" << keyword << "*" << content << "*" << endl;
#endif
}

#endif /* DBSCAN_H_ */
