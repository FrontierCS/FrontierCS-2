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


#ifndef _BENCH_GEOMETRY_IO
#define _BENCH_GEOMETRY_IO
#include <string>
#include "IO.h"
#include "parallel.h"
#include "geometry.h"
using namespace benchIO;

namespace benchIO {
  using namespace std;

  void parseNdPoints(char** Str, pointNd* P, long n, int dim) {
    {parallel_for (long i=0; i < n; i++) {
      P[i].m_dim = dim;
      for (intT d = 0; d < dim; ++ d) {
        P[i].m_data[d] = atof(Str[dim * i + d]);
      }
    }}
  }

#ifdef PBBSIO
  inline bool isNumber(char myChar) {
    if (myChar == '0' || myChar == '1' || myChar == '2' ||
	myChar == '3' || myChar == '4' || myChar == '5' ||
	myChar == '6' || myChar == '7' || myChar == '8' ||
	myChar == '9') {
      return true;
    } else {
      return false;
    }
  }
  inline intT extractDim(words *W) {
    int d;
    char *targetString = W->Strings[0];
    intT myPt = 18;
    while (isNumber(targetString[myPt])) myPt ++;
    targetString[myPt] = '\0';
    d = atoi(&targetString[18]);
    return d;
  }
#else
  inline intT extractDim(words *W) {
    int d;
    char *targetString = W->Strings[1];
    d = atoi(&targetString[0]);
    return d;
  }
#endif

  _seq<pointNd> readNdPointsFromFile(char* fname, int *dim) {
    _seq<char> S = readStringFromFile(fname);
    words W = stringToWords(S.A, S.n);
    int d = extractDim(&W);
    long n = (W.m-2)/d;
    pointNd *P = PointNDArrayAlloc(n, d);
    parseNdPoints(W.Strings + 2, P, n, d);
    *dim = d;
    return _seq<pointNd>(P, n);
  }
};

#endif // _BENCH_GEOMETRY_IO
