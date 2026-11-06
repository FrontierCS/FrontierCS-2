#ifndef _UTILITY_H
#define _UTILITY_H

#include <stdlib.h>
#include <algorithm>
#include <stdint.h>
#include <climits>
#include <iostream>
#include <cmath>
#include <vector>
#include <map>
#include <fstream>
#include <cassert>

#include <omp.h>
#include <tbb/scalable_allocator.h>

using namespace std;
#define 	EPSILON   0.001


template <class T>
struct ErrorTolerantEqual:
public binary_function< T, T, bool >
{
   ErrorTolerantEqual(const T & myepsilon):epsilon(myepsilon) {};
   inline bool operator() (const T & a, const T & b) const
   {
   	// According to the IEEE 754 standard, negative zero and positive zero should 
   	// compare as equal with the usual (numerical) comparison operators, like the == operators of C++ 
   
  	if(a == b)      // covers the "division by zero" case as well: max(a,b) can't be zero if it fails
   		return true;    // covered the integral numbers case
   
   	return ( std::abs(a - b) < epsilon || (std::abs(a - b) / max(std::abs(a), std::abs(b))) < epsilon ) ; 
   }
   T epsilon;
};

// Because identify reports ambiguity in PGI compilers
template<typename T>
struct myidentity : public std::unary_function<T, T>
{
    const T operator()(const T& x) const
    {
        return x;
    }
};

template<typename _ForwardIterator, typename _StrictWeakOrdering>
bool my_is_sorted(_ForwardIterator __first, _ForwardIterator __last,  _StrictWeakOrdering __comp)
{
   if (__first == __last)
   	return true;
   
   _ForwardIterator __next = __first;
   for (++__next; __next != __last; __first = __next, ++__next)
   	if (__comp(*__next, *__first))
   		return false;
   	return true;
};

template <typename ITYPE>
ITYPE CumulativeSum (ITYPE * arr, ITYPE size)
{
    ITYPE prev;
    ITYPE tempnz = 0 ;
    for (ITYPE i = 0 ; i < size ; ++i)
    {
		prev = arr[i];  
		arr[i] = tempnz; 
		tempnz += prev ; 
    }
    return (tempnz) ;		    // return sum
}


template<typename _ForwardIter, typename T>
void iota(_ForwardIter __first, _ForwardIter __last, T __value)
{
	while (__first != __last)
     		*__first++ = __value++;
}
	
template<typename T, typename I>
T ** allocate2D(I m, I n)
{
	T ** array = new T*[m];
	for(I i = 0; i<m; ++i) 
		array[i] = new T[n];
	return array;
}

template<typename T, typename I>
void deallocate2D(T ** array, I m)
{
	for(I i = 0; i<m; ++i) 
		delete [] array[i];
	delete [] array;
}


template < typename T >
struct absdiff : binary_function<T, T, T>
{
        T operator () ( T const &arg1, T const &arg2 ) const
        {
                using std::abs;
                return abs( arg1 - arg2 );
        }
};

/* This function will return n % d.
   d must be one of: 1, 2, 4, 8, 16, 32, … */
inline unsigned int getModulo(unsigned int n, unsigned int d)
{
	return ( n & (d-1) );
} 

// Same requirement (d=2^k) here as well
inline unsigned int getDivident(unsigned int n, unsigned int d)
{
	while((d = d >> 1))
		n = n >> 1;
	return n;
}

/*
 * Used for sort function.
 * Elements are sorted in descending order of `first' and ascending order of `second' (to break the tie).
 */
template <typename IT, typename NT>
bool sort_large(const pair<NT, IT> &left,const pair<NT, IT> &right)
{
//  return left.first > right.first;
  if(left.first != right.first) return left.first > right.first;
  return left.second < right.second;
}

// Memory allocation by C++-new / Aligned malloc / scalable malloc
template <typename T>
inline T* my_malloc(size_t array_size)
{
#ifdef CPP
    return (T *)(new T[array_size]);
#elif defined IMM
    return (T *)_mm_malloc(sizeof(T) * array_size, 64);
#elif defined TBB
    return (T *)scalable_malloc(sizeof(T) * array_size);
#else
    return (T *)scalable_malloc(sizeof(T) * array_size);
#endif
}

//template <typename T>
//inline T* my_malloc(int64_t array_size)
//{
//#ifdef CPP
//  return (T *)(new T[array_size]);
//#elif defined IMM
//  return (T *)_mm_malloc(sizeof(T) * array_size, 64);
//#elif defined TBB
//  return (T *)scalable_malloc(sizeof(T) * array_size);
//#else
//  return (T *)scalable_malloc(sizeof(T) * array_size);
//#endif
//}

// Memory deallocation
template <typename T>
inline void my_free(T *a)
{
//  cout << "this is calling\n";
#ifdef CPP
    delete[] a;
#elif defined IMM
    _mm_free(a);
#elif defined TBB
    scalable_free(a);
#else
    scalable_free(a);
#endif
}

// Prefix sum (Sequential)
template <typename T>
void seq_scan(T in, T *out, T N)
{
  out[0] = 0;
  for (T i = 0; i < N - 1; ++i) {
    out[i + 1] = out[i] + in;
  }
}

// Prefix sum (Sequential)
template <typename T>
void seq_scan(T *in, T *out, T N)
{
    out[0] = 0;
    for (T i = 0; i < N - 1; ++i) {
        out[i + 1] = out[i] + in[i];
//        if (((int64_t) out[i] + in[i]) >= (int64_t) INT32_MAX) {
//          cout << i << " " << out[i] << " " << in[i] << endl;
//          cout << i << " " << ((int64_t) out[i] + in[i]) << " " << (out[i] + in[i]) << endl;
//        }
//        assert(((int64_t) out[i] + in[i]) < (int64_t) INT32_MAX && "Sequential cumulative sum goes beyond the INT32_MAX limit!");
    }
}

// Prefix sum (Sequential)
template <typename T>
void seq_scan_V1(const T *in, T *out, T *out1, const T *in_weight, T N)
{
  out[0] = 0;
  out1[0] = 0;
  for (T i = 0; i < N - 1; ++i) {
    out[i + 1] = out[i] + in[i];
    out1[i + 1] = out1[i] + (in[i] * in_weight[i]);
  }
}

// Prefix sum (Sequential)
template <typename T, typename I>
void seq_scan_V2(const I *in, T *out, const I *in_weight, I N)
{
  out[0] = 0;
//  out1[0] = 0;
  for (I i = 0; i < N - 1; ++i) {
//    out[i + 1] = out[i] + in[i];
    out[i + 1] = out[i] + (in[i] * in_weight[i]);
  }
}

// Prefix sum (Sequential)
template <typename T, typename I>
void seq_scan_V3(const I *in, T *out, const I in_weight, I N)
{
  out[0] = 0;
//  out1[0] = 0;
  for (T i = 0; i < N - 1; ++i) {
//    out[i + 1] = out[i] + in[i];
    out[i + 1] = out[i] + (in[i] * in_weight);
  }
}

// Prefix sum (Thread parallel)
template <typename T>
void scan(T *in, T *out, T N)
{
//  seq_scan(in, out, N);
//  return;
    if (N < (1 << 17)) {
        seq_scan(in, out, N);
    }
    else {
        const int requested_threads = std::min(64, omp_get_max_threads());
//        int tnum = 16;


        T *partial_sum = (T *)scalable_malloc(sizeof(T) * requested_threads);
#pragma omp parallel num_threads(requested_threads)
        {
            const int tnum = omp_get_num_threads();
      const T each_n = N / tnum;
      int tid = omp_get_thread_num();
            T start = each_n * tid;
            T end = (tid < tnum - 1)? start + each_n : N;
            out[start] = 0;
            for (T i = start; i < end - 1; ++i) {
                out[i + 1] = out[i] + in[i];
//                assert(((int64_t) out[i] + in[i]) < (int64_t) INT32_MAX && "Cumulative sum goes beyond the INT32_MAX limit!");
            }
            partial_sum[tid] = out[end - 1] + (end < N ? in[end - 1] : 0);
//            assert(((int64_t) out[end - 1] + in[end - 1]) < (int64_t) INT32_MAX && "Partial sum goes beyond the INT32_MAX limit!");
#pragma omp barrier

            T offset = 0;
            for (int ii = 0; ii < tid; ++ii) {
                offset += partial_sum[ii];
            }
            for (T i = start; i < end; ++i) {
//              if (((int64_t) out[i] + offset) >= (int64_t) INT32_MAX) {
//                cout << i << " " << ((int64_t) out[i] + offset) << " " << (out[i] + offset) << endl;
//              }
//              assert(((int64_t) out[i] + offset) < (int64_t) INT32_MAX && "Cumulative sum-2 goes beyond the INT32_MAX limit!");
              out[i] += offset;
            }
        }
        scalable_free(partial_sum);
    }
}

// Prefix sum (Thread parallel)
template <typename T>
void scan(T in, T *out, T N)
{
//  seq_scan(in, out, N);
//  return;
  if (N < (1 << 17)) {
    seq_scan(in, out, N);
  }
  else {
    const int requested_threads = std::min(64, omp_get_max_threads());
//        int tnum = 16;


    T *partial_sum = (T *)scalable_malloc(sizeof(T) * requested_threads);
#pragma omp parallel num_threads(requested_threads)
    {
      const int tnum = omp_get_num_threads();
      const T each_n = N / tnum;
      int tid = omp_get_thread_num();
      T start = each_n * tid;
      T end = (tid < tnum - 1)? start + each_n : N;
      out[start] = 0;
      for (T i = start; i < end - 1; ++i) {
        out[i + 1] = out[i] + in;
//                assert(((int64_t) out[i] + in[i]) < (int64_t) INT32_MAX && "Cumulative sum goes beyond the INT32_MAX limit!");
      }
      partial_sum[tid] = out[end - 1] + in;
//            assert(((int64_t) out[end - 1] + in[end - 1]) < (int64_t) INT32_MAX && "Partial sum goes beyond the INT32_MAX limit!");
#pragma omp barrier

      T offset = 0;
      for (int ii = 0; ii < tid; ++ii) {
        offset += partial_sum[ii];
      }
      for (T i = start; i < end; ++i) {
//              if (((int64_t) out[i] + offset) >= (int64_t) INT32_MAX) {
//                cout << i << " " << ((int64_t) out[i] + offset) << " " << (out[i] + offset) << endl;
//              }
//              assert(((int64_t) out[i] + offset) < (int64_t) INT32_MAX && "Cumulative sum-2 goes beyond the INT32_MAX limit!");
        out[i] += offset;
      }
    }
    scalable_free(partial_sum);
  }
}

// Prefix sum (Thread parallel)
template <typename T>
void scan(const T *in, T *out, T *out1, const T *in_weight, T N)
{
//  seq_scan(in, out, N);
//  return;
  if (N < (1 << 17)) {
    seq_scan_V1(in, out, out1, in_weight, N);
  }
  else {
    const int requested_threads = std::min(64, omp_get_max_threads());


    T *partial_sum = (T *)scalable_malloc(sizeof(T) * requested_threads);
    T *partial_sum1 = (T *)scalable_malloc(sizeof(T) * requested_threads);
#pragma omp parallel num_threads(requested_threads)
    {
      const int tnum = omp_get_num_threads();
      const T each_n = N / tnum;
      int tid = omp_get_thread_num();
      T start = each_n * tid;
      T end = (tid < tnum - 1)? start + each_n : N;
      out[start] = 0;
      out1[start] = 0;
      for (T i = start; i < end - 1; ++i) {
        out[i + 1] = out[i] + in[i];
        out1[i + 1] = out1[i] + (in[i] * in_weight[i]);
      }
      partial_sum[tid] = out[end - 1] + (end < N ? in[end - 1] : 0);
      partial_sum1[tid] = out1[end - 1] + (end < N ? in[end - 1] * in_weight[end - 1] : 0);
#pragma omp barrier

      T offset = 0;
      T offset1 = 0;
      for (int ii = 0; ii < tid; ++ii) {
        offset += partial_sum[ii];
        offset1 += partial_sum1[ii];
      }
      for (T i = start; i < end; ++i) {
        out[i] += offset;
        out1[i] += offset1;
      }
    }
    scalable_free(partial_sum);
    scalable_free(partial_sum1);
  }
}

// Prefix sum (Thread parallel)
template <typename T, typename I>
void scan(const I *in, T *out, const I *in_weight, I N)
{
//  seq_scan(in, out, N);
//  return;
  if (N < (1 << 17)) {
    seq_scan_V2(in, out, in_weight, N);
  }
  else {
    const int requested_threads = std::min(64, omp_get_max_threads());


    T *partial_sum = (T *)scalable_malloc(sizeof(T) * requested_threads);
//    T *partial_sum1 = (T *)scalable_malloc(sizeof(T) * requested_threads);
#pragma omp parallel num_threads(requested_threads)
    {
      const int tnum = omp_get_num_threads();
      const T each_n = N / tnum;
      int tid = omp_get_thread_num();
      T start = each_n * tid;
      T end = (tid < tnum - 1)? start + each_n : N;
      out[start] = 0;
      for (T i = start; i < end - 1; ++i) {
//        out[i + 1] = out[i] + in[i];
        out[i + 1] = out[i] + (in[i] * in_weight[i]);
      }
//      partial_sum[tid] = out[end - 1] + (end < N ? in[end - 1] : 0);
      partial_sum[tid] = out[end - 1] + (end < N ? in[end - 1] * in_weight[end - 1] : 0);
#pragma omp barrier

      T offset = 0;
      T offset1 = 0;
      for (int ii = 0; ii < tid; ++ii) {
        offset += partial_sum[ii];
//        offset1 += partial_sum1[ii];
      }
      for (T i = start; i < end; ++i) {
        out[i] += offset;
//        out1[i] += offset1;
      }
    }
    scalable_free(partial_sum);
//    scalable_free(partial_sum1);
  }
}

// Prefix sum (Thread parallel)
template <typename T, typename I>
void scan(const I *in, T *out, const I in_weight, I N)
{
//  seq_scan(in, out, N);
//  return;
  if (N < (1 << 17)) {
    seq_scan_V3(in, out, in_weight, N);
  }
  else {
    const int requested_threads = std::min(64, omp_get_max_threads());


    T *partial_sum = (T *)scalable_malloc(sizeof(T) * requested_threads);
//    T *partial_sum1 = (T *)scalable_malloc(sizeof(T) * requested_threads);
#pragma omp parallel num_threads(requested_threads)
    {
      const int tnum = omp_get_num_threads();
      const T each_n = N / tnum;
      int tid = omp_get_thread_num();
      T start = each_n * tid;
      T end = (tid < tnum - 1)? start + each_n : N;
      out[start] = 0;
      for (T i = start; i < end - 1; ++i) {
//        out[i + 1] = out[i] + in[i];
        out[i + 1] = out[i] + (in[i] * in_weight);
      }
//      partial_sum[tid] = out[end - 1] + (end < N ? in[end - 1] : 0);
      partial_sum[tid] = out[end - 1] + (end < N ? in[end - 1] * in_weight : 0);
#pragma omp barrier

      T offset = 0;
      T offset1 = 0;
      for (int ii = 0; ii < tid; ++ii) {
        offset += partial_sum[ii];
//        offset1 += partial_sum1[ii];
      }
      for (T i = start; i < end; ++i) {
        out[i] += offset;
//        out1[i] += offset1;
      }
    }
    scalable_free(partial_sum);
//    scalable_free(partial_sum1);
  }
}

// Sort by key
template <typename IT, typename NT>
inline void mergesort(IT *nnz_num, NT *nnz_sorting,
               IT *temp_num, NT *temp_sorting,
               IT left, IT right)
{
    int mid, i, j, k;
  
    if (left >= right) {
        return;
    }

    mid = (left + right) / 2;

    mergesort(nnz_num, nnz_sorting, temp_num, temp_sorting, left, mid);
    mergesort(nnz_num, nnz_sorting, temp_num, temp_sorting, mid + 1, right);

    for (i = left; i <= mid; ++i) {
        temp_num[i] = nnz_num[i];
        temp_sorting[i] = nnz_sorting[i];
    }

    for (i = mid + 1, j = right; i <= right; ++i, --j) {
        temp_sorting[i] = nnz_sorting[j];
        temp_num[i] = nnz_num[j];
    }

    i = left;
    j = right;
  
    for (k = left; k <= right; ++k) {
        if (temp_num[i] <= temp_num[j] && i <= mid) {
            nnz_num[k] = temp_num[i];
            nnz_sorting[k] = temp_sorting[i++];
        }
        else {
            nnz_num[k] = temp_num[j];
            nnz_sorting[k] = temp_sorting[j--];
        }
    }
}

/*Sorting key-value*/
template <typename IT, typename NT>
inline void cpu_sorting_key_value(IT *key, NT *value, IT N)
{
    IT *temp_key;
    NT *temp_value;

    temp_key = my_malloc<IT>(N);
    temp_value = my_malloc<NT>(N);

    mergesort(key, value, temp_key, temp_value, 0, N-1);

    my_free<IT>(temp_key);
    my_free<NT>(temp_value);

}

inline string parseFileNameFromPath(const string &path) {
  // Find the last directory separator
  size_t lastSeparatorIndex = path.find_last_of("/\\");

  // Extract the substring after the last separator
  std::string fileName = path.substr(lastSeparatorIndex + 1);

  return fileName;
}

inline string parseFileNamePrefixFromPath(const string &path) {
  // extract filename from path
  std::string fileName = parseFileNameFromPath(path);

  // Find the last dot and remove the extension
  size_t lastDotIndex = fileName.find_last_of(".");
  if (lastDotIndex != std::string::npos) {
    fileName = fileName.substr(0, lastDotIndex);
  }

  return fileName;
}

inline string parseFileNamePrefixFromPathWithSanitization(const string &path) {
  // Extract filename from path
  std::string fileName = parseFileNameFromPath(path);

  // Remove the extension
  size_t lastDotIndex = fileName.find_last_of(".");
  if (lastDotIndex != std::string::npos) {
    fileName = fileName.substr(0, lastDotIndex);
  }

  // Remove suffixes _rcm or _nd if present at the end
  if (fileName.size() > 4 && fileName.substr(fileName.size() - 4) == "_rcm") {
    fileName = fileName.substr(0, fileName.size() - 4);
  } else if (fileName.size() > 3 && fileName.substr(fileName.size() - 3) == "_nd") {
    fileName = fileName.substr(0, fileName.size() - 3);
  }

  return fileName;
}

inline string create_filename_suffix(char **argv) {
  string suffix = "";
  if (string(argv[1]) == string("gen")) {
    if (string(argv[2]) == string("rmat")) {
      suffix = "rmat_" + string(argv[3]) + "_" + string(argv[4]) + ".txt";
    } else {
      suffix = "er_" + string(argv[3]) + "_" + string(argv[4]) + ".txt";
    }
  } else {
    suffix = parseFileNameFromPath(string(argv[2]));
  }
  return suffix;
}

template <typename IT>
inline void print_cluster_to_file(const map<IT, vector<IT>>& clusters, const string& filename) {
  std::fstream file(filename, std::ios::out);
  for(auto& c : clusters) {
    file << c.first << ":";
    for(auto i : c.second) {
      file << " " << i;
    }
    file << endl;
  }
  file.close();
}

inline int nextPowerOfTwo(int num) {
  if (num <= 0) {
    return 1; // Next power of two for non-positive numbers is 2^0 = 1
  }

  // Calculate the next power of two
  num--;
  num |= num >> 1;
  num |= num >> 2;
  num |= num >> 4;
  num |= num >> 8;
  num |= num >> 16;
  num++;

  return num;
}

inline int get_file_count(const string& dir_path) {
  int file_count = 0;

  try {
    for (const auto& entry : fs::directory_iterator(dir_path)) {
      if (fs::is_regular_file(entry.status())) {
        if (entry.path().extension() == ".mtx") {
          file_count++;
        }
      }
    }
//    std::cout << "Number of files in directory: " << file_count << std::endl;
  } catch (const fs::filesystem_error& e) {
    std::cerr << "Filesystem error: " << e.what() << std::endl;
    exit(-1);
  }
  return file_count;
}

#endif

