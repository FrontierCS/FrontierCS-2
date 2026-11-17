#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <string>
#include <assert.h>

#define HYPER 1
#define MCX16 1
#define VERBOSE 1
#define INVERSE 1

#include "hypergraphIO.h"
#include "index_map.h"
#include "bucket.h"
#include "edgeMapReduce.h"

#include "parallel.h"
#include "gettime.h"
#include "utils.h"
#include "graph.h"
#include "hypergraphIO.h"
#include "parseCommandLine.h"
#include "vertex.h"
#include "sequence.h"
#include "sparseSet.h"
#include "sampleSort.h"
#ifndef OPENMP
#include "../../lib/histogram.h"
#endif

#include "butterfly_count.h"
#include "butterfly_ecount.h"
#include "butterfly_peel.h"
#include "butterfly_epeel.h"
#include "butterfly_count_total.h"

#include <vector>

using namespace std;

string CountTypeToStr(CountType ty) {
  switch(ty) {
  case ASORT: return "Sort"; break;
  case SORT: return "SortCE"; break;
  case AHASH: return "Hash"; break;
  case HASH: return "HashCE"; break;
  case AHIST: return "Hist"; break;
  case HIST: return "HistCE"; break;
  case BATCHS: return "Par"; break;
  case SERIAL: return "Serial"; break;
  case BATCHWA: return "WedgePar"; break;
  default: break;
  }
  return "";
}

string PeelTypeToStr(PeelType ty) {
  switch(ty) {
  case PSORT: return "SortCE Peel"; break;
  case PHASH: return "HashCE Peel"; break;
  case PHIST: return "HistCE Peel"; break;
  case PBATCHS: return "Par Peel"; break;
  case PBATCHWA: return "WedgePar Peel"; break;
  default: break;
  }
  return "";
}

/*
 *  Butterfly counting/peeling framework
 * 
 *  GA: Bipartite graph in CSR format
 *  P : Command line arguments
 */
void Compute(bipartiteCSR& GA, commandLine P) {
  const size_t eltsPerCacheLine = 64/sizeof(long);

  // Type for counting
  long count_type_long = P.getOptionLongValue("-t", LONG_MAX);
  string count_type_str = P.getOptionValue("-countType", "");
  CountType ty;
  if (count_type_long == LONG_MAX) {
    if (count_type_str == "ASORT") ty = ASORT;
    else if (count_type_str == "SORT") ty = SORT;
    else if (count_type_str == "AHASH") ty = AHASH;
    else if (count_type_str == "HASH") ty = HASH;
    else if (count_type_str == "AHIST") ty = AHIST;
    else if (count_type_str == "HIST") ty = HIST;
    else if (count_type_str == "BATCHS") ty = BATCHS;
    else if (count_type_str == "BATCHWA") ty = BATCHWA;
    else ty = SERIAL;
  }
  else {
    switch(count_type_long) {
    case 0: ty = ASORT; break;
    case 1: ty = SORT; break;
    case 2: ty = AHASH; break;
    case 3: ty = HASH; break;
    case 4: ty = AHIST; break;
    case 6: ty = HIST; break;
    case 5: case 7: case 11: ty = BATCHS; break;
    case 9: case 12: ty = SERIAL; break;
    case 8: ty = BATCHWA; break;
    default: break;
    }
  }

  // Type for ranking
  long rank_type_long = P.getOptionLongValue("-w", LONG_MAX);
  string rank_type_str = P.getOptionValue("-rankType", "");
  RankType tw;
  if (rank_type_long == LONG_MAX) {
    if (rank_type_str == "SIDE") tw = SIDE;
    else if (rank_type_str == "COCORE") tw = COCORE;
    else if (rank_type_str == "ACOCORE") tw = ACOCORE;
    else if (rank_type_str == "DEG") tw = DEG;
    else tw = ADEG;
  }
  else {
    switch(rank_type_long) {
    case 0: tw = SIDE; break;
    case 1: tw = COCORE; break;
    case 2: tw = ACOCORE; break;
    case 3: tw = DEG; break;
    case 4: tw = ADEG; break;
    default: break;
    }
  }

  // Type for peeling
  long peel_type_long = P.getOptionLongValue("-tp", LONG_MAX);
  string peel_type_str = P.getOptionValue("-peelType", "");
  PeelType tp;
  bool nopeel = false;
  if (peel_type_long == LONG_MAX) {
    if (peel_type_str == "SORT") tp = PSORT;
    else if (peel_type_str == "HASH") tp = PHASH;
    else if (peel_type_str == "HIST") tp = PHIST;
    else if (peel_type_str == "BATCHS") tp = PBATCHS;
    else if (peel_type_str == "BATCHWA") tp = PBATCHWA;
    else nopeel = true;
  }
  else {
    nopeel = P.getOptionValue("-nopeel");
    switch(peel_type_long) {
    case 0: tp = PHASH; break;
    case 1: tp = PSORT; break;
    case 2: tp = PHIST; break;
    case 3: case 4: tp = PBATCHS; break;
    case 5: tp = PBATCHWA; break;
    default: nopeel = true; break;
    }
  }

  // Type for per vert, edge, or total
  string per_type_str = P.getOptionValue("-per","");
  long te = P.getOptionLongValue("-e", LONG_MAX);
  bool total = P.getOptionValue("-total");
  PerType per_type;
  if (te == LONG_MAX) {
    if (per_type_str == "VERT") per_type = VERT;
    else if (per_type_str == "EDGE") per_type = EDGE;
    else if (per_type_str == "TOTAL") per_type = TOTAL;
    else per_type = VERT;
  } else {
    if (total) per_type = TOTAL;
    else if (te == 0) per_type = VERT;
    else per_type = EDGE;
  }

  // # of max wedges
  long max_wedges = P.getOptionLongValue("-m",2577500000);
  long max_array_size = P.getOptionLongValue("-a",23090996160);

  long denom = P.getOptionLongValue("-d",25);

  long sparse_type_long = P.getOptionLongValue("-s", LONG_MAX);
  string sparse_type_str = P.getOptionValue("-sparseType", "");
  SparseType sparse;
  if (sparse_type_long == LONG_MAX) {
    if (sparse_type_str == "EDGE") sparse = ESPARSE;
    else if (sparse_type_str == "COLOR") sparse = CLRSPARSE;
    else sparse = NOSPARSE;
  }
  else {
    switch(sparse_type_long) {
    case 0: sparse = NOSPARSE; break;
    case 1: sparse = CLRSPARSE; break;
    case 2: sparse = ESPARSE; break;
    default: sparse = NOSPARSE; break;
    }
  }
  if (sparse != NOSPARSE) { per_type = TOTAL; }
  if (per_type == TOTAL) { nopeel = true; }

  // Preprocessing
  timer t1;
  t1.start();
  tuple<bool, long> use_v_tuple = cmpWedgeCounts(GA);
  bool use_v = get<0>(use_v_tuple);
  long num_wedges = get<1>(use_v_tuple);
  t1.reportTotal("preprocess (wedge counts)");

  // Choose per type
  if (per_type == TOTAL) {
    timer t;
    t.start();
    long num_butterflies = CountTotal(GA, use_v, num_wedges, max_wedges, max_array_size, ty, tw);
    t.stop();

    t.reportTotal(CountTypeToStr(ty));

    if (sparse == 1) num_butterflies *= pow(denom, 3);
    else if (sparse == 2) num_butterflies *= pow(denom, 4);
    cout << "number of butterflies: " << num_butterflies << "\n";
    return;
  }
  else if (per_type == VERT) {
    timer t;
    t.start();
    long* butterflies = Count(GA, use_v, num_wedges, max_wedges, max_array_size, ty, tw);
    t.stop();

    t.reportTotal(CountTypeToStr(ty));

    long num_idxs = use_v ? GA.nu : GA.nv;

    auto butterflies_extract_f = [&] (const long i) -> const long {
      return butterflies[eltsPerCacheLine*i];
    };
    
    long total = sequence::reduce<long>((long)0,(long)num_idxs,addF<long>(),butterflies_extract_f);
    
    cout << "number of butterflies: " << total/2 << "\n";

    if(!nopeel) {
      timer t2;
      t2.start();
      auto cores = Peel(GA, use_v, butterflies, max_wedges, tp, max_array_size);
      t2.stop();
      t2.reportTotal(PeelTypeToStr(tp));
    }
    free(butterflies);
  }
  else {
    timer t3,t4;
    t4.start();
    auto eti = edgeToIdxs(GA, use_v);
    t4.reportTotal("edgeToIdxs");
    
    t3.start();
    long* ebutterflies = CountE(eti, GA, use_v, num_wedges, max_wedges, max_array_size, ty, tw);
    t3.stop();
  
    t3.reportTotal(CountTypeToStr(ty));

    auto butterflies_extract_f = [&] (const long i) -> const long {
      return ebutterflies[eltsPerCacheLine*i];
    };
    
    long total = sequence::reduce<long>((long)0,(long)GA.numEdges,addF<long>(),butterflies_extract_f);
    cout << "number of edge butterflies: " << total/4 << "\n";

    if(!nopeel) {
      timer t2;
      t2.start();
      auto ite = idxsToEdge(GA, use_v);
      auto cores = PeelE(eti, ite, GA, use_v, ebutterflies, max_wedges, tp);
      free(ite);      
      t2.stop();
      t2.reportTotal(PeelTypeToStr(tp));
    }
    free(eti);

    free(ebutterflies);
  }
}

/*
 *  Butterfly counting/peeling framework; calls the framework (Compute) over
 *  multiple rounds
 */
int parallel_main(int argc, char* argv[]) {
  commandLine P(argc,argv," <inFile>");
  char* iFile = P.getArgument(0);
  long rounds = P.getOptionLongValue("-r", 3);
  bool nopeel = P.getOptionValue("-nopeel");
  long denom = P.getOptionLongValue("-d", 25);

  string per_type_str = P.getOptionValue("-per","");
  long te = P.getOptionLongValue("-e", LONG_MAX);
  bool total = P.getOptionValue("-total");
  PerType per_type;
  if (te == LONG_MAX) {
    if (per_type_str == "VERT") per_type = VERT;
    else if (per_type_str == "EDGE") per_type = EDGE;
    else if (per_type_str == "TOTAL") per_type = TOTAL;
    else per_type = VERT;
  } else {
    if (total) per_type = TOTAL;
    else if (te == 0) per_type = VERT;
    else per_type = EDGE;
  }

  long sparse_type_long = P.getOptionLongValue("-s", LONG_MAX);
  string sparse_type_str = P.getOptionValue("-sparseType", "");
  SparseType sparse;
  if (sparse_type_long == LONG_MAX) {
    if (sparse_type_str == "EDGE") sparse = ESPARSE;
    else if (sparse_type_str == "COLOR") sparse = CLRSPARSE;
    else sparse = NOSPARSE;
  }
  else {
    switch(sparse_type_long) {
    case 0: sparse = NOSPARSE; break;
    case 1: sparse = CLRSPARSE; break;
    case 2: sparse = ESPARSE; break;
    default: sparse = NOSPARSE; break;
    }
  }

  /*if ((per_type == EDGE || nopeel) && sparse == NOSPARSE) {
    bipartiteCSR G = readBipartite(iFile);
    Compute(G, P);
    for(int r = 0; r < rounds; r++) {
      Compute(G, P);
    }
    G.del();
  }
  else {*/
  for (int r = 0; r < rounds + 1; r++) {
    bipartiteCSR G;
    if (sparse == NOSPARSE) G = readBipartite(iFile);
    else {
      auto tmp = readBipartite(iFile);
      G = sparse == CLRSPARSE ? clrSparseBipartite(tmp, denom, 0) : eSparseBipartite(tmp, denom, 0);
      tmp.del();
    }
    Compute(G, P);
    G.del();
  }

    /*for(int r = 0; r < rounds; r++) {
      if (sparse == NOSPARSE) G = readBipartite(iFile);
      else{
	auto tmp = readBipartite(iFile);
	G = sparse == CLRSPARSE ? clrSparseBipartite(tmp, denom, (r + 1) * (tmp.nv + tmp.nu)) :
    eSparseBipartite(tmp, denom, (r + 1) * (tmp.nv + tmp.nu));
	tmp.del();
      }
      Compute(G, P);
      G.del();
    }*/
  //}
}

