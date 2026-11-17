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

#ifndef _SEQUENCE_IO
#define _SEQUENCE_IO
//#define HUGEPAGES 

#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include "IO.h"
#include "sequence.h"
#include <sys/mman.h>
#include <errno.h>

namespace benchIO {
  using namespace std;
  typedef pair<intT,intT> intPair;
  typedef pair<char*,intT> stringIntPair;
  typedef pair<intT,char*> intStringPair;
  
  enum elementType { none, intType, intPairT, stringIntPairT, intStringPairT, doubleT, stringT};
  //elementType dataType(long a) { return longT;}
  elementType dataType(intT a) { return intType;}
  elementType dataType(double a) { return doubleT;}
  elementType dataType(char* a) { return stringT;}
  elementType dataType(intPair a) { return intPairT;}
  elementType dataType(stringIntPair a) { return stringIntPairT;}
  elementType dataType(intStringPair a) { return intStringPairT;}
  
  string seqHeader(elementType dt) {
    switch (dt) {
    case intType: return "sequenceInt";
    case doubleT: return "sequenceDouble";
    case stringT: return "sequenceChar";
    case intPairT: return "sequenceIntPair";
    case stringIntPairT: return "sequenceStringIntPair";
    case intStringPairT: return "sequenceIntStringPair";
    default: 
      cout << "writeArrayToFile: type not supported" << endl; 
      abort();
    }
  }

  elementType elementTypeFromString(string s) {
    if (s == "double") return doubleT;
    else if (s == "string") return stringT;
    else if (s == "int") return intType;
    else return none;
  }

  struct seqData { 
    void* A; long n; elementType dt; 
    char* O; // used for strings to store pointer to character array
    seqData(void* _A, long _n, elementType _dt) : 
      A(_A), O(NULL), n(_n), dt(_dt) {}
    seqData(void* _A, char* _O, long _n, elementType _dt) : 
      A(_A), O(_O), n(_n), dt(_dt) {}
    void del() {
      if (O) free(O);
      free(A);
    }
  };

  seqData readSequenceFromFile(char* fileName) {
    _seq<char> S = readStringFromFile(fileName);
    words W = stringToWords(S.A, S.n);
    char* header = W.Strings[0];
    void* bytes;
    long n = W.m-1;
    elementType tp;

    if (header == seqHeader(intType)) {
      tp = intType;
      long size = n*sizeof(intT); 
      #ifdef HUGEPAGES 
      intT* A = (intT*) mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_NORESERVE | MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB , -1, 0);
      //intT * A = (intT*) aligned_alloc(4096, n* sizeof(intT));//new intT[n];
      //int result = madvise(A, n*sizeof(intT), MADV_HUGEPAGE);
      //printf("%p %d \n ", A, n*sizeof(intT)); 
      //std::cout << strerror(errsv)  << "size " <<  size <<  " " << errsv <<  " " << EAGAIN <<  " " << EBADF << " " << " " << EINVAL << endl;
      #else 
      intT * A = new intT[n];
      #endif 
      int errsv = errno;
 
      parallel_for(long i=0; i < n; i++)
	A[i] = atol(W.Strings[i+1]);
      //W.del(); // to deal with performance bug in malloc
      return seqData((void*) A, n, intType);
    } else if (header == seqHeader(doubleT)) {
      double* A = new double[n];
      parallel_for (long i=0; i < n; i++)
	A[i] = atof(W.Strings[i+1]);
      //W.del(); // to deal with performance bug in malloc
      return seqData((void*) A, n, doubleT);
    } else if (header == seqHeader(stringT)) {
      char** A = new char*[n];
      parallel_for (long i=0; i < n; i++)
	A[i] = W.Strings[i+1];
      //free(W.Strings); // to deal with performance bug in malloc
      return seqData((void*) A, W.Chars, n, stringT);
    } else if (header == seqHeader(intPairT)) {
      n = n/2;
      intPair* A = new intPair[n];
      parallel_for (long i=0; i < n; i++) {
	A[i].first = atol(W.Strings[2*i+1]);
	A[i].second = atol(W.Strings[2*i+2]);
      }
      //W.del();  // to deal with perfromance bug in malloc
      return seqData((void*) A, n, intPairT);
    } else if (header == seqHeader(stringIntPairT)) {
      n = n/2;
      stringIntPair* A = new stringIntPair[n];
      parallel_for (long i=0; i < n; i++) {
	A[i].first = W.Strings[2*i+1];
	A[i].second = atol(W.Strings[2*i+2]);
      }
      // free(W.Strings); // to deal with performance bug in malloc
      return seqData((void*) A, W.Chars, n, stringIntPairT);
    } else if (header == seqHeader(intStringPairT)) {
      n = n/2;
      intStringPair* A = new intStringPair[n];
      parallel_for (long i=0; i < n; i++) {
	A[i].first = atol(W.Strings[2*i+1]);
	A[i].second = W.Strings[2*i+2];
      }
      // free(W.Strings); // to deal with performance bug in malloc
      return seqData((void*) A, W.Chars, n, intStringPairT);
    }
    abort();
  }

  template <class T>
  int writeSequenceToFile(T* A, long n, char* fileName) {
    elementType tp = dataType(A[0]);
    return writeArrayToFile(seqHeader(tp), A, n, fileName);
  }

};

#endif // _SEQUENCE_IO
