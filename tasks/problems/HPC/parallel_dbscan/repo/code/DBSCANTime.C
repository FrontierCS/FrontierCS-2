// This code is part of the Problem Based Benchmark Suite (PBBS)
// Copyright (c) 2011 Guy Blelloch and the PBBS team
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <stdint.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include "gettime.h"
#include "utils.h"
#include "geometry.h"
#include "geometryIO.h"
#include "parseCommandLine.h"
#include "parallel.h"
#include "DBSCAN.h"

using namespace std;
using namespace benchIO;

// *************************************************************
//  OUTPUT
// *************************************************************

void outputMultiClusterFile(char* t_fileName, pair<GdLongDataType*, ClusterContainer** > t_clusters, PtLongDataType n, bool t_verbose, bool t_writefile) {
  
  ofstream myfile;

  if (t_writefile) {
    myfile.open(t_fileName);
    cout << "Writing output to file " << t_fileName << endl;
  } else {
    cout << "Verbalizing clusters ... " << endl;
  }

  GdLongDataType *clusterIds = t_clusters.first;
  ClusterContainer **borderClusters = t_clusters.second;
  PtLongDataType numClusters = 0;
  PtLongDataType numCorePts = 0;
  PtLongDataType numBorderPts = 0;
  PtLongDataType numNoisePts = 0;
  PtLongDataType totClusterPts = 0;
  GdLongDataType *clusterSizes = newA(GdLongDataType, n);
  parallel_for(PtLongDataType i = 0; i < n; ++ i) clusterSizes[i] = 0;
  ClusterContainer clusterNames;
  
  for (PtLongDataType i = 0; i < n; ++ i) {
    if (clusterIds[i] >= 0) {
      numCorePts ++;
      totClusterPts ++;
      clusterNames.insert(clusterIds[i]);
      clusterSizes[clusterIds[i]] ++;
      if (t_writefile) myfile << i << ' ' << clusterIds[i] << '\n';
    } else if (clusterIds[i] == -2) {
      numBorderPts ++;
      if (t_writefile) myfile << i << ' ';
      ClusterContainer::iterator it = borderClusters[i]->begin();
      while (it != borderClusters[i]->end()) {
	clusterNames.insert(*it);
	clusterSizes[*it] ++;
        totClusterPts ++;
	if (t_writefile) myfile << *it << ' ';
	it ++;
      }
      if (t_writefile) myfile << '\n';
    } else if (clusterIds[i] == -1) {
      numNoisePts ++;
    } else {
      cout << "corrupted dbscan output! stopped" << endl;
      break;
    }
  }
  
  PrintParser("numpts", n);
  PrintParser("numcore",numCorePts);
  PrintParser("numborder",numBorderPts);
  PrintParser("numnoise",numNoisePts);
  PrintParser("numclusterpts",totClusterPts);
  PrintParser("numcluster",(long)clusterNames.size());
  if (t_verbose) {
    cout << ">>>>>>>>>>>>>>>>" << endl;
    cout << "Total points: " << n << endl;
    cout << "Core points: " << numCorePts << endl;
    cout << "Border points: " << numBorderPts << endl;
    cout << "Noise points: " << numNoisePts << endl;
    cout << "Total clustered points: " << totClusterPts << endl;
    cout << "Num clusters: " << clusterNames.size() << endl;
    GdLongDataType limitOutput = 0;
    ClusterContainer::iterator it = clusterNames.begin();
    while (it != clusterNames.end()) {
      cout << "clusterid-" << *it << ": " << clusterSizes[*it] << " points" << endl;
      it ++;
      limitOutput ++;
      if (limitOutput >= 20) {
	cout << " ... see output file for more details" << endl;
	break;
      }
    } 
    cout << ">>>>>>>>>>>>>>>>" << endl;
  }

  free(clusterSizes);

}

void FreeContainer(GdLongDataType *t_ids, ClusterContainer ** t_containers, PtLongDataType n) {
  for (intT i = 0; i < n; ++ i) {
    if (t_ids[i] == -2) delete t_containers[i];
  }
  free(t_containers);
}

// *************************************************************
//  TIMING
// *************************************************************

template <class pointType>
void timeDBSCANNd(_seq<pointType> pts, DbscanParams dbscanParams, intT rounds, char* outFile) {
#if defined CLOSEST_BORDER_CLUSTER
  pair<intT*,GdLongDataType*> Clusters;
#else
  pair<GdLongDataType*, ClusterContainer** > Clusters;
#endif

  PrintParser("roundsplit", "");
  for(intT i=0;i<rounds;i++) {
    startTime();
    Clusters = DBSCAN(pts, dbscanParams);
    //nextTimeN();
    double totalTime = _tm.next();
    PrintParser("totaltime", totalTime);
    //cout << "PBBS-time: " << totalTime << endl;
    PrintParser("roundsplit", "");
    if(i < rounds-1) {
#if defined CLOSEST_BORDER_CLUSTER
      free(Clusters.first);
      free(Clusters.second);
#else
      FreeContainer(Clusters.first, Clusters.second, pts.n);
      free(Clusters.first);
#endif
    }
  }

#if defined CLOSEST_BORDER_CLUSTER
  // dbscan finds only closest cluster for border points
  // each point only has one cluster
  // output pair <coreflags, clusterids>
  
  if (outFile != NULL) {
    GdLongDataType *resultArray = newA(GdLongDataType,pts.n * 2);
    // first might be of a different type, use for loop to copy and cast
    //copy(Clusters.first, Clusters.first + pts.n, resultArray);
    parallel_for (PtLongDataType i = 0; i < pts.n; ++ i) {
      resultArray[i] = (GdLongDataType) Clusters.first[i];
    }
    copy(Clusters.second, Clusters.second + pts.n, resultArray + pts.n);
    if (outFile != NULL) writeIntArrayToFile(resultArray, pts.n * 2, outFile);
    free(resultArray);
  }
  free(Clusters.first);
  free(Clusters.second);
  
#else
  // dbscan finds multiple clsuters for border points
  // output pair <clusterids, // >=0 for corepts, -1 for noise, -2 for border pts
  // unordered_set<id> *> // border point clustering, indexed by normal point order, only initialized for border points

#ifdef MULTI_CLUSTER_OUTPUT // output for enduser, do not use checker on the output
  
  if (outFile != NULL) {
    outputMultiClusterFile(outFile, Clusters, pts.n, true, true);
  } else {
    outputMultiClusterFile(outFile, Clusters, pts.n, true, false);
  }
  FreeContainer(Clusters.first, Clusters.second, pts.n);
  free(Clusters.first);

#else

  if (outFile != NULL) {
    GdLongDataType *resultArray = newA(GdLongDataType,pts.n * 2);
    copy(Clusters.first, Clusters.first + pts.n, resultArray + pts.n);
    parallel_for (PtLongDataType i = 0; i < pts.n; ++ i) {
      if (Clusters.first[i] >= 0) {
	resultArray[i] = 1;
      } else {
        resultArray[i] = 0;
      }
      if (Clusters.first[i] == -2) { // select random cluster for border
        ClusterContainer::iterator it = Clusters.second[i]->begin();
        resultArray[pts.n + i] = *it;
      }
    }
    if (outFile != NULL) writeIntArrayToFile(resultArray, pts.n * 2, outFile);
    free(resultArray);

  }
  FreeContainer(Clusters.first, Clusters.second, pts.n);
  free(Clusters.first);

#endif // MULTI_CLUSTER_OUTPUT
 
#endif 

  return;
}

intT main(intT argc, char* argv[]) {
  commandLine P(argc,argv,"[-o <outFile>] [-r <rounds>] [-eps <p_epsilon>] [-minpts <p_minpts>] <inFile>");
  char* iFile = P.getArgument(0);
  char* oFile = P.getOptionValue("-o");
  intT rounds = P.getOptionIntValue("-r",1);
  PtIntermediateDataType p_epsilon = (PtIntermediateDataType) P.getOptionDoubleValue("-eps",1);
  intT p_minpts = P.getOptionIntValue("-minpts",1);
  PtIntermediateDataType p_rho = (PtIntermediateDataType) P.getOptionDoubleValue("-rho",-1);

#if defined(DOUBLE_DATATYPE)
  if (sizeof(NDGCTYPE) != 8) {
    cout << "error: datatype error, make sure datatype setting in DBSCAN.h and geometry.h NDGCTYPE defn match" << endl;
    exit(1);
  }
#endif

#if defined(FLOAT_DATATYPE)
  if (sizeof(NDGCTYPE) != 4) {
    cout << "error: datatype error, make sure datatype setting in DBSCAN.h and geometry.h NDGCTYPE defn match" << endl;
    exit(1);
  }
#endif
  
  
  cout << "///////////////////////////////////////////////////////////////" << endl;
  cout << "::[DBSCANTime] iFile:  " << iFile << endl;
  if (oFile != NULL) cout << "::[DBSCANTime] oFile:  " << oFile << endl;
  cout << "::[DBSCANTime] rounds:  " << rounds << endl;
  cout << "::[DBSCANTime] EPS:    " << p_epsilon << endl;
  cout << "::[DBSCANTime] MINPTS: " << p_minpts << endl;

  if (p_rho != -1) {
    cout << "::[DBSCANTime] RHO: " << p_rho << endl;
  }

  DbscanParams myParams = {p_epsilon, p_minpts, p_rho};

  intT dim = 0;

  _seq<pointNd> PIn = readNdPointsFromFile(iFile, &dim);
  // for (intT i=0; i<PIn.n; ++i) {
  //   for (intT j=0; j<dim; ++j) {
  //     cout << PIn.A[i].m_data[j] << " ";
  //   }
  //   cout << endl;
  // }

  cout << "::[DBSCANTime] DIM: " << dim << endl;
  
  PrintParser("testsplit","");
  PrintParser("datafile", iFile);
  PrintParser("rounds", rounds);
  PrintParser("eps", p_epsilon);
  PrintParser("minpts", p_minpts);
  if (p_rho != -1) {
    PrintParser("rho", p_rho);
  }
  PrintParser("dim", dim);

  timeDBSCANNd<pointNd>(PIn, myParams, rounds, oFile);
}
