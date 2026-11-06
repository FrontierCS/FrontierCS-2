// This code is part of the Problem Based Benchmark Suite (PBBS)
// Copyright (c) 2010-2016 Guy Blelloch and the PBBS team
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

#pragma once
#include <limits>

#include <math.h>
#include <stdio.h>
#include <cstdint>
#include <algorithm>
#include "utilities.h"
#include "counting_sort.h"
#include "sequentialHT.h"
#include "index_map.h"
#include <unordered_map>

namespace pbbsa {

  template <typename s_size_t, typename Seq>
  sequence<s_size_t> seq_histogram(Seq A, size_t m) {
    sequence<s_size_t> counts(m);
    for (size_t i = 0; i < m; i++)
      counts[i] = 0;

    for (size_t i = 0; i < A.size(); i++)
      counts[A[i]]++;
    return counts;
  }

  template <typename Seq, typename CSeq>
  void _seq_count(Seq In, CSeq counts) {
    for (size_t i = 0; i < counts.size(); i++) counts[i] = 0;
    for (size_t j = 0; j < In.size(); j++) counts[In[j]]++;
  }

  template <typename s_size_t, typename Seq>
  sequence<s_size_t> _count(Seq In, size_t num_buckets) {
    sequence<s_size_t> counts(num_buckets);
    size_t n = In.size();

    if (n < ((size_t) 1 << 14)) {
      _seq_count(In, counts);
      return counts;
    }

    size_t num_threads = __cilkrts_get_nworkers();
    size_t num_blocks = std::min((size_t) (1 + n/(num_buckets*32)),
				 num_threads*4);
    size_t block_size = ((n-1)/num_blocks) + 1;
    size_t m = num_blocks * num_buckets;

    sequence<s_size_t> block_counts(m);

    // count each block
    parallel_for_1 (size_t i = 0; i < num_blocks; ++i) {
      size_t start = std::min(i * block_size, n);
      size_t end = std::min((i+1) * block_size, n);
      auto bc = block_counts.slice(i*num_buckets,(i+1)*num_buckets);
      _seq_count(In.slice(start,end), bc);
    }
    if (m >= (1 << 14)) {
      parallel_for (size_t j = 0; j < num_buckets; j++) {
      	size_t sum = 0;
      	for (size_t i = 0; i < num_blocks; i++) {
      	  sum += block_counts[i*num_buckets+j];
        }
      	counts[j] = sum;
      }
    } else {
      for (size_t j = 0; j < num_buckets; j++) {
      	size_t sum = 0;
      	for (size_t i = 0; i < num_blocks; i++) {
      	  sum += block_counts[i*num_buckets+j];
        }
      	counts[j] = sum;
      }
    }
    return counts;
  }


  template <typename E>
  struct get_bucket {
    pair<E,int>* hash_table;
    size_t table_mask;
    size_t low_mask;
    size_t bucket_mask;
    int num_buckets;
    int k;
    E* I;

    pair<E*,int> heavy_hitters(E* A, size_t n, size_t count) {
      E* sample = new E[count];
      for (size_t i = 0; i < count; i++) {
      	sample[i] = A[hash64(i)%n];
      }
      std::sort(sample,sample+count);

      // only keep those with at least two copies
      int k = 0;
      int c = 0;
      for (size_t i = 1; i < count; i++) {
      	if (sample[i] == sample[i-1]) {
      	  if (c++ == 0) {
            sample[k++] = sample[i];
          }
      	} else {
          c = 0;
        }
      }
      return make_pair(sample,k);
    }

    pair<E,int>* make_hash_table(E* entries, size_t n,
				 size_t table_size, size_t table_mask) {
      auto table = new pair<E,int>[table_size];
      for (size_t i=0; i < table_size; i++) {
        table[i] = make_pair(0,-1);
      }
      for (size_t i = 0; i < n; i++) {
        table[hash64(entries[i])&table_mask] = make_pair(entries[i],i);
      }
      return table;
    }

    get_bucket(E* A, size_t n, size_t bits) :I(A) {
      num_buckets = 1 << bits;
      bucket_mask = num_buckets-1;
      low_mask = ~((size_t) 15);
      int count = 2 * num_buckets;
      int table_size = 4 * count;
      table_mask = table_size-1;

      pair<E*,int> heavy = heavy_hitters(A, n, count);
      k = heavy.second;
      E* sample = heavy.first;

      hash_table = make_hash_table(heavy.first,k, table_size, table_mask);
      delete[] sample;
    }

    ~get_bucket() {
      delete[] hash_table; }

    size_t operator() (size_t i) {
      if (k > 0) {
      	pair<E,int> h = hash_table[hash64(I[i])&table_mask];
      	if (h.first == I[i] && h.second != -1) {
      	  return h.second + num_buckets;
        }
      }
      return pbbs::hash64(I[i] & low_mask) & bucket_mask;
    }

  };

  // n = A.size(): number of elements to histogram
  // m: elements are in the range [0, m)
  // Returns a sequence S of size m (dense) where S[i] is the number of times i
  // appeared in A.
  template <typename s_size_t, typename Seq>
  sequence<s_size_t> histogram(Seq A, size_t m) {
    using E = typename Seq::T;
    size_t n = A.size();
    size_t bits;

    if (n < (1 << 27)) bits = (log2_up(n) - 7)/2;
    // for large n selected so each bucket fits into cache
    else bits = (log2_up(n) - 17);
    size_t num_buckets = (1<<bits);
    if (m < n / num_buckets) {
      return  _count<s_size_t>(A, m);
    }
    if (n < (1 << 13)) {
      return seq_histogram<s_size_t>(A , m);
    }

    // generate sample
    get_bucket<E> x(A.as_array(), n, bits-1);
    auto get_buckets = make_sequence<size_t>(n, x);

    // first buckets based on hash, except for low 4 bits
    sequence<size_t> bucket_offsets
      = count_sort(A, A, get_buckets, num_buckets);

    // note that this is cache line alligned
    sequence<s_size_t> counts(m);
    parallel_for (size_t i = 0; i < m; i++) {
      counts[i] = 0;
    }

    // now sequentially process each bucket
    parallel_for_1(size_t i = 0; i < num_buckets; i++) {
      size_t start = bucket_offsets[i];
      size_t end = bucket_offsets[i+1];
      if (i < num_buckets/2) {
        // light
        for (size_t j = start; j < end; j++) {
      	  counts[A[j]]++;
        }
      } else if (end > start) {
        // heavy
        counts[A[start]] = end-start;
      }
    }
    return counts;
  }

  template <typename s_size_t, typename ct_t, typename Seq>
  tuple<size_t, tuple<s_size_t, ct_t>* > seq_sparse_histogram(Seq A, size_t m, sequence<tuple<s_size_t, ct_t>>& out_seq) {
    sequentialHT<s_size_t, ct_t> tab(nullptr, A.size(), 1.0, make_tuple(numeric_limits<s_size_t>::max(), 0));
    for (size_t i = 0; i < A.size(); i++) {
      tab.insertAdd(A[i]);
    }
    out_seq.resize(tab.m);
    size_t out_size = tab.compactInto(out_seq.as_array());
    tab.del();
    return make_tuple(out_size, out_seq.as_array());
  }

  // n = A.size(): number of elements to histogram
  // m: elements are in the range [0, m)
  // Returns a sequence S of pairs of (elm, count) of size <= n.
    template <typename s_size_t, typename ct_t, typename Seq>
  tuple<size_t, tuple<s_size_t, ct_t>* > sparse_histogram(Seq A, size_t m) {
    sequence<tuple<s_size_t, ct_t>> tmp_seq = sequence<tuple<s_size_t, ct_t>>();
    sequence<tuple<s_size_t, ct_t>> out_seq = sequence<tuple<s_size_t, ct_t>>();
    auto out = sparse_histogram<s_size_t, ct_t>(A, m, tmp_seq, out_seq);
    out_seq.allocated = false;
    return out;
  }

  template <typename s_size_t, typename ct_t, typename Seq>
  tuple<size_t, tuple<s_size_t, ct_t>* > sparse_histogram(Seq A, size_t m, 
    sequence<tuple<s_size_t, ct_t>>& tmp_seq, sequence<tuple<s_size_t, ct_t>>& out_seq) {
    using E = typename Seq::T;
    size_t n = A.size();
    size_t bits;

    if (n < (1 << 27)) bits = (log2_up(n) - 7)/2;
    // for large n selected so each bucket fits into cache
    else bits = (log2_up(n) - 17);
    size_t num_buckets = (1<<bits);
    if (n < (1 << 13)) {
      return seq_sparse_histogram<s_size_t, ct_t>(A , m, out_seq);
    }

    // generate sample
    get_bucket<E> x(A.as_array(), n, bits-1);
    auto get_buckets = make_sequence<size_t>(n, x);

    timer t; t.start();
    // first buckets based on hash, except for low 4 bits
    sequence<size_t> bucket_offsets
      = count_sort(A, A, get_buckets, num_buckets);
    t.stop(); // t.total();

    sequence<size_t> offs(num_buckets+1);
    parallel_for_1(size_t i=0; i<num_buckets+1; i++) {
      if (i < num_buckets/2) {
        offs[i] =
          (size_t)(1 << pbbs::log2_up((bucket_offsets[i+1] - bucket_offsets[i]) + 100));
      } else if (bucket_offsets[i+1] > bucket_offsets[i]) {
        offs[i] = 1;
      } else {
        offs[i] = 0;
      }
    }
    offs[num_buckets] = 0;
    scan_add(offs, offs);

    using outT = tuple<s_size_t, ct_t>;

    tmp_seq.resize(offs[num_buckets]);
    outT* tmp = tmp_seq.as_array();
    outT empty = make_tuple(numeric_limits<s_size_t>::max(), 0);

    sequence<size_t> c_offs(num_buckets+1);
    parallel_for(size_t i=0; i<offs[num_buckets]; i++) {
      tmp[i] = empty;
    }

    // now sequentially process each bucket
    parallel_for_1(size_t i = 0; i < num_buckets; i++) {
      size_t start = bucket_offsets[i];
      size_t end = bucket_offsets[i+1];
      size_t t_size = (size_t)(1 << pbbs::log2_up(end - start + 100));
      outT* tab = tmp + offs[i];
      if (i < num_buckets/2) {
        auto table = sequentialHT<s_size_t, ct_t>(tab, t_size, 1, empty);
        // light
        for (size_t j = start; j < end; j++) {
          table.insertAdd(A[j]);
        }
        c_offs[i] = table.n_elms;
      } else if (end > start) {
        // heavy
        *tab = make_tuple(A[start], end - start);
        c_offs[i] = 1;
      } else {
        c_offs[i] = 0;
      }
    }
    c_offs[num_buckets] = 0;
    scan_add(c_offs, c_offs);

    out_seq.resize(c_offs[num_buckets]);
    outT* out = out_seq.as_array();
    parallel_for_1(size_t i=0; i<num_buckets; i++) {
      size_t start = bucket_offsets[i];
      size_t end = bucket_offsets[i+1];
      size_t t_size = (size_t)(1 << pbbs::log2_up(end - start + 100));
      auto tab = tmp + offs[i];
      size_t out_off = c_offs[i];
      if (i < num_buckets/2) {
        auto table = sequentialHT<s_size_t, ct_t>(tab, t_size, 1, empty);
        table.compactInto(out + out_off);
      } else {
        out[out_off] = tmp[offs[i]];
      }
    }
    bucket_offsets.allocated = true;
    bucket_offsets.~sequence();
    return make_tuple(c_offs[num_buckets], out);
  }

//****************************************************************************************

// struct is E; must have field hash
  template <typename E, typename Cmp, typename Eq>
  struct get_bucket_struct {
    pair<E,int>* hash_table;
    size_t table_mask;
    size_t low_mask;
    size_t bucket_mask;
    int num_buckets;
    int k;
    E* I;
    Cmp cmp;
    Eq eq;
    E empty;

    pair<E*,int> heavy_hitters(E* A, size_t n, size_t count) {
      E* sample = new E[count];
      for (size_t i = 0; i < count; i++) {
      	sample[i] = A[hash64(i)%n];
      }
      std::sort(sample,sample+count,cmp);

      // only keep those with at least two copies
      int k = 0;
      int c = 0;
      for (size_t i = 1; i < count; i++) {
      	if (eq(sample[i], sample[i-1])) {
      	  if (c++ == 0) {
            sample[k++] = sample[i];
          }
      	} else {
          c = 0;
        }
      }
      return make_pair(sample,k);
    }

    pair<E,int>* make_hash_table(E* entries, size_t n,
				 size_t table_size, size_t table_mask) {
      auto table = new pair<E,int>[table_size];
      for (size_t i=0; i < table_size; i++) {
        table[i] = make_pair(empty,-1); //TODO lol idk about this
      }
      for (size_t i = 0; i < n; i++) {
        table[hash64(entries[i].hash)&table_mask] = make_pair(entries[i],i);
      }
      return table;
    }

    get_bucket_struct(E* A, size_t n, size_t bits, Cmp _cmp, Eq _eq, E _empty) :I(A), cmp(_cmp), eq(_eq), empty(_empty) {
      num_buckets = 1 << bits;
      bucket_mask = num_buckets-1;
      low_mask = ~((size_t) 15);
      int count = 2 * num_buckets;
      int table_size = 4 * count;
      table_mask = table_size-1;

      pair<E*,int> heavy = heavy_hitters(A, n, count);
      k = heavy.second;
      E* sample = heavy.first;

      hash_table = make_hash_table(heavy.first,k, table_size, table_mask);
      delete[] sample;
    }

    ~get_bucket_struct() {
      free(hash_table); }

    size_t operator() (size_t i) {
      if (k > 0) {
      	pair<E,int> h = hash_table[hash64(I[i].hash)&table_mask];
      	if (h.second != -1) {
      	  return h.second + num_buckets;
        }
      }
      return pbbs::hash64(I[i].hash & low_mask) & bucket_mask;
    }

  };

  template <typename s_size_t, typename ct_t, typename Seq, typename Eq, typename E>
  tuple<size_t, tuple<s_size_t, ct_t>* > seq_sparse_histogram_struct(Seq A, size_t m, Eq eq, E empty) {
    sequentialHTStruct<s_size_t, ct_t, Eq> tab(nullptr, A.size(), 1.0, make_tuple(empty, 0), eq);
    for (size_t i = 0; i < A.size(); i++) {
      tab.insertAdd(A[i]);
    }
    auto out = tab.compact();
    tab.del();
    return out;
  }

  // n = A.size(): number of elements to histogram
  // m: elements are in the range [0, m)
  // Returns a sequence S of pairs of (elm, count) of size <= n.
  template <typename s_size_t, typename ct_t, typename Seq, typename Cmp, typename Eq, typename E>
  tuple<size_t, tuple<s_size_t, ct_t>* > sparse_histogram_struct(Seq A, size_t m, Cmp cmp, Eq eq, E empty_struct) {
    //using E = typename Seq::T;
    size_t n = A.size();
    size_t bits;

    if (n < (1 << 27)) bits = (log2_up(n) - 7)/2;
    // for large n selected so each bucket fits into cache
    else bits = (log2_up(n) - 17);
    size_t num_buckets = (1<<bits);
    if (n < (1 << 13)) {
      return seq_sparse_histogram_struct<s_size_t, ct_t>(A , m, eq, empty_struct);
    }

    // generate sample
    get_bucket_struct<E, Cmp, Eq> x(A.as_array(), n, bits-1, cmp, eq, empty_struct);
    auto get_buckets = make_sequence<size_t>(n, x);

    timer t; t.start();
    // first buckets based on hash, except for low 4 bits
    sequence<size_t> bucket_offsets
      = count_sort(A, A, get_buckets, num_buckets);
    t.stop(); // t.total();

    sequence<size_t> offs(num_buckets+1);
    parallel_for_1(size_t i=0; i<num_buckets+1; i++) {
      if (i < num_buckets/2) {
        offs[i] =
          (size_t)(1 << pbbs::log2_up((bucket_offsets[i+1] - bucket_offsets[i]) + 100));
      } else if (bucket_offsets[i+1] > bucket_offsets[i]) {
        offs[i] = 1;
      } else {
        offs[i] = 0;
      }
    }
    offs[num_buckets] = 0;
    scan_add(offs, offs);

    using outT = tuple<s_size_t, ct_t>;
    outT* tmp = new_array_no_init<outT>(offs[num_buckets]);
    outT empty = make_tuple(empty_struct, 0);

    sequence<size_t> c_offs(num_buckets+1);
    parallel_for(size_t i=0; i<offs[num_buckets]; i++) {
      tmp[i] = empty;
    }

    // now sequentially process each bucket
    parallel_for_1(size_t i = 0; i < num_buckets; i++) {
      size_t start = bucket_offsets[i];
      size_t end = bucket_offsets[i+1];
      size_t t_size = (size_t)(1 << pbbs::log2_up(end - start + 100));
      outT* tab = tmp + offs[i];
      if (i < num_buckets/2) {
        auto table = sequentialHTStruct<s_size_t, ct_t, Eq>(tab, t_size, 1, empty, eq);
        // light
        for (size_t j = start; j < end; j++) {
          table.insertAdd(A[j]);
        }
        c_offs[i] = table.n_elms;
      } else if (end > start) {
        // heavy
        *tab = make_tuple(A[start], end - start);
        c_offs[i] = 1;
      } else {
        c_offs[i] = 0;
      }
    }
    c_offs[num_buckets] = 0;
    scan_add(c_offs, c_offs);

    outT* out = new_array_no_init<outT>(c_offs[num_buckets]);
    parallel_for_1(size_t i=0; i<num_buckets; i++) {
      size_t start = bucket_offsets[i];
      size_t end = bucket_offsets[i+1];
      size_t t_size = (size_t)(1 << pbbs::log2_up(end - start + 100));
      auto tab = tmp + offs[i];
      size_t out_off = c_offs[i];
      if (i < num_buckets/2) {
        auto table = sequentialHTStruct<s_size_t, ct_t, Eq>(tab, t_size, 1, empty, eq);
        table.compactInto(out + out_off);
      } else {
        out[out_off] = tmp[offs[i]];
      }
    }
    free(tmp);
    return make_tuple(c_offs[num_buckets], out);
  }

//****************************************************************************************


template <class E, class V>
struct TupleCmp {
	bool operator() (tuple<E,V> t1, tuple<E,V> t2) {
		return get<0>(t1) < get<0>(t2);
	}
};

  template <typename E, typename V>
  struct get_bucket_f {
    pair<tuple<E,V>,int>* hash_table;
    size_t table_mask;
    size_t low_mask;
    size_t bucket_mask;
    int num_buckets;
    int k;
    tuple<E,V>* I;

    pair<tuple<E,V>*,int> heavy_hitters(tuple<E,V>* A, size_t n, size_t count) {
      tuple<E,V>* sample = new tuple<E,V>[count];
      for (size_t i = 0; i < count; i++) {
      	sample[i] = A[hash64(i)%n];
      }
      std::sort(sample,sample+count,TupleCmp<E,V>());

      // only keep those with at least two copies
      int k = 0;
      int c = 0;
      for (size_t i = 1; i < count; i++) {
      	if (get<0>(sample[i]) == get<0>(sample[i-1])) {
      	  if (c++ == 0) {
            sample[k++] = sample[i];
          }
      	} else {
          c = 0;
        }
      }
      return make_pair(sample,k);
    }

    pair<tuple<E,V>,int>* make_hash_table(tuple<E,V>* entries, size_t n,
				 size_t table_size, size_t table_mask) {
      auto table = new pair<tuple<E,V>,int>[table_size];
      for (size_t i=0; i < table_size; i++) {
        table[i] = make_pair(make_pair(0,0),-1);
      }
      for (size_t i = 0; i < n; i++) {
        table[hash64(get<0>(entries[i]))&table_mask] = make_pair(entries[i],i);
      }
      return table;
    }

    get_bucket_f(tuple<E,V>* A, size_t n, size_t bits) :I(A) {
      num_buckets = 1 << bits;
      bucket_mask = num_buckets-1;
      low_mask = ~((size_t) 15);
      int count = 2 * num_buckets;
      int table_size = 4 * count;
      table_mask = table_size-1;

      pair<tuple<E,V>*,int> heavy = heavy_hitters(A, n, count);
      k = heavy.second;
      tuple<E,V>* sample = heavy.first;

      hash_table = make_hash_table(heavy.first,k, table_size, table_mask);
      delete[] sample;
    }

    ~get_bucket_f() {
      free(hash_table); }

    size_t operator() (size_t i) {
      if (k > 0) {
      	pair<tuple<E,V>,int> h = hash_table[hash64(get<0>(I[i]))&table_mask];
      	if (get<0>(h.first) == get<0>(I[i]) && h.second != -1) {
      	  return h.second + num_buckets;
        }
      }
      return pbbs::hash64(get<0>(I[i]) & low_mask) & bucket_mask;
    }

  };

  template <typename s_size_t, typename ct_t, typename E, typename V, typename Hash>
  tuple<size_t, tuple<s_size_t,sequentialHT<ct_t,uintE>*>* > seq_sparse_histogram_list(sequence<tuple<E,V>> A, size_t m, Hash max) {
    sequentialHTList<s_size_t, ct_t, Hash> tab(nullptr, A.size(), 1.0, make_tuple(numeric_limits<s_size_t>::max(), nullptr), max);
    for (size_t i = 0; i < A.size(); i++) {
      tab.insert(A[i]);
    }
    auto out = tab.compact();
    tab.del();
    return out;
  }

  // n = A.size(): number of elements to histogram
  // m: elements are in the range [0, m)
  // Returns a sequence S of pairs of (elm, count) of size <= n.
  // TODO: maybe not a good idea to have sizes -- just do max size? see which is faster?
  // , tuple<E,uintE>* sizes, size_t total_num
  template <typename s_size_t, typename ct_t, typename E, typename V, typename Hash>
  tuple<size_t, tuple<s_size_t,sequentialHT<ct_t,uintE>*>* > sparse_histogram_list(sequence<tuple<E,V>> A, size_t m, Hash max) {
    //using E = typename Seq::T;
    size_t n = A.size();
    size_t bits;

    if (n < (1 << 27)) bits = (log2_up(n) - 7)/2;
    // for large n selected so each bucket fits into cache
    else bits = (log2_up(n) - 17);
    size_t num_buckets = (1<<bits);
    if (n < (1 << 13)) {
      return seq_sparse_histogram_list<s_size_t, ct_t>(A , m, max);
    }

    // generate sample
    get_bucket_f<E,V> x(A.as_array(), n, bits-1);
    auto get_buckets = make_sequence<size_t>(n, x);

    timer t; t.start();
    // first buckets based on hash, except for low 4 bits
    sequence<size_t> bucket_offsets
      = count_sort(A, A, get_buckets, num_buckets);
    t.stop(); // t.total();

    sequence<size_t> offs(num_buckets+1);
    parallel_for_1(size_t i=0; i<num_buckets+1; i++) {
      if (i < num_buckets/2) {
        offs[i] =
          (size_t)(1 << pbbs::log2_up((bucket_offsets[i+1] - bucket_offsets[i]) + 100));
      } else if (bucket_offsets[i+1] > bucket_offsets[i]) {
        offs[i] = 1;
      } else {
        offs[i] = 0;
      }
    }
    offs[num_buckets] = 0;
    scan_add(offs, offs);

    //using outT = tuple<s_size_t, ct_t>;
    using outT = tuple<s_size_t,sequentialHT<ct_t,uintE>*>;
    outT* tmp = new_array_no_init<outT>(offs[num_buckets]);
    outT empty = make_tuple(numeric_limits<s_size_t>::max(), nullptr);

    sequence<size_t> c_offs(num_buckets+1);
    parallel_for(size_t i=0; i<offs[num_buckets]; i++) {
      tmp[i] = empty;
    }

    // now sequentially process each bucket
    parallel_for_1(size_t i = 0; i < num_buckets; i++) {
      size_t start = bucket_offsets[i];
      size_t end = bucket_offsets[i+1];
      size_t t_size = (size_t)(1 << pbbs::log2_up(end - start + 100));
      outT* tab = tmp + offs[i];
      if (i < num_buckets/2) {
        auto table = sequentialHTList<s_size_t, ct_t, Hash>(tab, t_size, 1, empty, max);
        // light
        for (size_t j = start; j < end; j++) {
          table.insert(A[j]);
        }
        c_offs[i] = table.n_elms;
      } else if (end > start) {
        // heavy
        //*tab = make_tuple(A[start], end - start);

        sequentialHT<ct_t, uintE>* ht = new sequentialHT<ct_t, uintE>(nullptr,end-start,1.0,make_tuple(numeric_limits<ct_t>::max(), 0));
        for (size_t l =start; l < end; ++l) {ht->insertAdd(get<1>(A[l]));}
        *tab = make_tuple(get<0>(A[start]), ht);
        //*tab = make_tuple(get<0>(A[start]), end - start);
        c_offs[i] = 1;
      } else {
        c_offs[i] = 0;
      }
    }
    c_offs[num_buckets] = 0;
    scan_add(c_offs, c_offs);

    outT* out = new_array_no_init<outT>(c_offs[num_buckets]);
    parallel_for_1(size_t i=0; i<num_buckets; i++) {
      size_t start = bucket_offsets[i];
      size_t end = bucket_offsets[i+1];
      size_t t_size = (size_t)(1 << pbbs::log2_up(end - start + 100));
      auto tab = tmp + offs[i];
      size_t out_off = c_offs[i];
      if (i < num_buckets/2) {
        auto table = sequentialHTList<s_size_t, ct_t, Hash>(tab, t_size, 1, empty, max);
        table.compactInto(out + out_off);
      } else {
        out[out_off] = tmp[offs[i]];
      }
    }
    free(tmp);
    return make_tuple(c_offs[num_buckets], out);
  }

//****************************************************************************************
// A should be a sequence of key value tuples

  template <typename s_size_t, typename ct_t, typename E, typename V, typename F>
  tuple<size_t, tuple<s_size_t, ct_t>* > seq_sparse_histogram_f(sequence<tuple<E,V>> A, size_t m, F& f, sequence<tuple<s_size_t, ct_t>>& out_seq) {
    sequentialHT<s_size_t, ct_t> tab(nullptr, A.size(), 1.0, make_tuple(numeric_limits<s_size_t>::max(), 0));
    for (size_t i = 0; i < A.size(); i++) {
      tab.insertF(A[i], f);
    }
    out_seq.resize(tab.m);
    size_t out_size = tab.compactInto(out_seq.as_array());
    tab.del();
    return make_tuple(out_size, out_seq.as_array());
  }

template <typename s_size_t, typename ct_t, typename E, typename V, typename F, typename G>
  tuple<size_t, tuple<s_size_t, ct_t>* > sparse_histogram_f(sequence<tuple<E,V>> A, size_t m, F& f, G& f_reduce) {
    sequence<tuple<s_size_t, ct_t>> tmp_seq = sequence<tuple<s_size_t, ct_t>>();
    sequence<tuple<s_size_t, ct_t>> out_seq = sequence<tuple<s_size_t, ct_t>>();
    auto out =  sparse_histogram_f<s_size_t, ct_t>(A, m, f, f_reduce, tmp_seq, out_seq);
    out_seq.allocated = false;
    return out;
  }

  template <typename s_size_t, typename ct_t, typename E, typename V, typename F, typename G>
  tuple<size_t, tuple<s_size_t, ct_t>* > sparse_histogram_f(sequence<tuple<E,V>> A, size_t m, F& f, G& f_reduce, 
    sequence<tuple<s_size_t, ct_t>>& tmp_seq, sequence<tuple<s_size_t, ct_t>>& out_seq) {
    //using E = typename Seq::T;
    size_t n = A.size();
    size_t bits;

    if (n < (1 << 27)) bits = (log2_up(n) - 7)/2;
    // for large n selected so each bucket fits into cache
    else bits = (log2_up(n) - 17);
    size_t num_buckets = (1<<bits);
    if (n < (1 << 13)) {
      return seq_sparse_histogram_f<s_size_t, ct_t>(A , m, f, out_seq);
    }

    // generate sample
    get_bucket_f<E, V> x(A.as_array(), n, bits-1);
    auto get_buckets = make_sequence<size_t>(n, x);

    timer t; t.start();
    // first buckets based on hash, except for low 4 bits
    sequence<size_t> bucket_offsets
      = count_sort(A, A, get_buckets, num_buckets); //TODO check if this is ok
    t.stop(); // t.total();

    sequence<size_t> offs(num_buckets+1);
    parallel_for_1(size_t i=0; i<num_buckets+1; i++) {
      if (i < num_buckets/2) {
        offs[i] =
          (size_t)(1 << pbbs::log2_up((bucket_offsets[i+1] - bucket_offsets[i]) + 100));
      } else if (bucket_offsets[i+1] > bucket_offsets[i]) {
        offs[i] = 1;
      } else {
        offs[i] = 0;
      }
    }
    offs[num_buckets] = 0;
    scan_add(offs, offs);

    using outT = tuple<s_size_t, ct_t>;
    tmp_seq.resize(offs[num_buckets]);
    outT* tmp = tmp_seq.as_array();
    outT empty = make_tuple(numeric_limits<s_size_t>::max(), 0);

    sequence<size_t> c_offs(num_buckets+1);
    parallel_for(size_t i=0; i<offs[num_buckets]; i++) {
      tmp[i] = empty;
    }

    // now sequentially process each bucket
    parallel_for_1(size_t i = 0; i < num_buckets; i++) {
      size_t start = bucket_offsets[i];
      size_t end = bucket_offsets[i+1];
      size_t t_size = (size_t)(1 << pbbs::log2_up(end - start + 100));
      outT* tab = tmp + offs[i];
      if (i < num_buckets/2) {
      //if (end > start || i < num_buckets / 2) {
        auto table = sequentialHT<s_size_t, ct_t>(tab, t_size, 1, empty);
        // light
        for (size_t j = start; j < end; j++) {
          table.insertF(A[j],f);
        }
        c_offs[i] = table.n_elms;
      } else if (end > start) {
        // heavy
      //  *tab = make_tuple(A[start], end - start);
        *tab = reduce(A.slice(start,end),f_reduce);//make_tuple(get<0>(A[start]), get<1>(reduce(A.slice(start,end),f_reduce)));
        c_offs[i] = 1;
      } else {
        c_offs[i] = 0;
      }
    }
    c_offs[num_buckets] = 0;
    scan_add(c_offs, c_offs);


    out_seq.resize(c_offs[num_buckets]);
    outT* out = out_seq.as_array();
    parallel_for_1(size_t i=0; i<num_buckets; i++) {
      size_t start = bucket_offsets[i];
      size_t end = bucket_offsets[i+1];
      size_t t_size = (size_t)(1 << pbbs::log2_up(end - start + 100));
      auto tab = tmp + offs[i];
      size_t out_off = c_offs[i];
      if (i < num_buckets/2) {
        auto table = sequentialHT<s_size_t, ct_t>(tab, t_size, 1, empty);
        table.compactInto(out + out_off);
      } else {
        out[out_off] = tmp[offs[i]];
      }
    }
    bucket_offsets.allocated = true;
    bucket_offsets.~sequence();
    return make_tuple(c_offs[num_buckets], out);
  }

//****************************************************************************************

  template <typename s_size_t, typename Seq>
  tuple<size_t, s_size_t*> seq_sparse_accumulator_histogram(Seq A, Seq acc, size_t m) {
    s_size_t* out = pbbs::new_array_no_init<s_size_t>(m);
    size_t k = 0;
    for (size_t i = 0; i < A.size(); i++) {
      s_size_t e = A[i];
      if (acc[e] == 0) {
        out[k++] = e;
      }
      acc[e]++;
    }
    return make_tuple(k, out);
  }

  // n = A.size(): number of elements to histogram
  // m: elements are in the range [0, m)
  // Returns a sequence S of pairs of (elm, count) of size <= n.
  // Suppose we're also given as input a cleared array of size m (between [0, max_elm])
  // that we can use to accumulate values. The output is the set of distinct
  // indices, with values written into m.
  template <typename s_size_t, typename Seq>
  tuple<size_t, s_size_t*> sparse_accumulator_histogram(Seq A, Seq acc, size_t m) {
    using E = typename Seq::T;
    size_t n = A.size();
    size_t bits;

    if (n < (1 << 27)) bits = (log2_up(n) - 7)/2;
    // for large n selected so each bucket fits into cache
    else bits = (log2_up(n) - 17);
    size_t num_buckets = (1<<bits);
    if (n < (1 << 13)) {
      return seq_sparse_accumulator_histogram<s_size_t>(A, acc, m);
    }

    // generate sample
    get_bucket<E> x(A.as_array(), n, bits-1);
    auto get_buckets = make_sequence<size_t>(n, x);

    // first buckets based on hash, except for low 4 bits
    sequence<size_t> bucket_offsets
      = count_sort(A, A, get_buckets, num_buckets);

    using OT = s_size_t;

    sequence<size_t> offsets(num_buckets+1);
    parallel_for_1(size_t i = 0; i < num_buckets; i++) {
      size_t start = bucket_offsets[i];
      size_t end = bucket_offsets[i+1];
      if (i < num_buckets/2) {
        offsets[i] = end - start;
      } else if (end > start) {
        offsets[i] = 1;
      } else {
        offsets[i] = 0;
      }
    }
    offsets[num_buckets] = 0;
    scan_add(offsets, offsets);

    OT* tmp = new_array_no_init<OT>(offsets[num_buckets]);
    sequence<size_t> c_offsets(num_buckets+1);

    // now sequentially process each bucket
    parallel_for_1(size_t i = 0; i < num_buckets; i++) {
      size_t start = bucket_offsets[i];
      size_t end = bucket_offsets[i+1];
      OT* T = tmp + offsets[i];
      if (i < num_buckets/2) {
        // light bucket
        size_t k = 0;
        for (size_t j = start; j < end; j++) {
          s_size_t e = A[j];
          if (acc[e] == 0) {
            T[k++] = e;
          }
          acc[e]++;
        }
        c_offsets[i] = k;
      } else if (end > start) {
        // heavy bucket
        *T = A[start];
        c_offsets[i] = 1;
      } else {
        c_offsets[i] = 0;
      }
    }
    c_offsets[num_buckets] = 0;
    scan_add(c_offsets, c_offsets);

    OT* out = new_array_no_init<OT>(c_offsets[num_buckets]);
    parallel_for_1(size_t i=0; i<num_buckets; i++) {
      size_t start = offsets[i];
      size_t sz = c_offsets[i+1] - c_offsets[i];
      OT* T = tmp + start;
      OT* O = out + c_offsets[i];
      if (sz > 0) {
        for (size_t j=0; j<sz; j++) {
          O[j] = T[j];
        }
      }
    }
    free(tmp);
    size_t n_distinct_elms = c_offsets[num_buckets];
    return make_tuple(n_distinct_elms, out);
  }

  template <typename s_size_t, typename ct_t, typename Seq>
  tuple<size_t, tuple<s_size_t, ct_t>*> seq_sparse_accumulator_histogram_compact(Seq A, Seq acc, size_t m) {
    using OT = tuple<s_size_t, ct_t>;
    OT* out = pbbs::new_array_no_init<OT>(m);
    size_t k = 0;
    for (size_t i = 0; i < A.size(); i++) {
      s_size_t e = A[i];
      if (acc[e] == 0) {
        out[k++] = make_tuple(e, 0);
      }
      acc[e]++;
    }
    for (size_t i=0; i<k; i++) {
      s_size_t e = std::get<0>(out[i]);
      std::get<1>(out[i]) = acc[e];
      acc[e] = 0; // clear the accumulator struct
    }
    return make_tuple(k, out);
  }

  // n = A.size(): number of elements to histogram
  // m: elements are in the range [0, m)
  // Returns a sequence S of pairs of (elm, count) of size <= n.
  // Suppose we're also given as input a cleared array of size m (between [0, max_elm])
  // that we can use to accumulate values. The output is an array of pairs of distinct
  // indices, and the number of times they appeared in the input.
  template <typename s_size_t, typename ct_t, typename Seq>
  tuple<size_t, tuple<s_size_t, ct_t>*> sparse_accumulator_histogram_compact(Seq A, Seq acc, size_t m) {
    using E = typename Seq::T;
    size_t n = A.size();
    size_t bits;

    if (n < (1 << 27)) bits = (log2_up(n) - 7)/2;
    // for large n selected so each bucket fits into cache
    else bits = (log2_up(n) - 17);
    size_t num_buckets = (1<<bits);
    if (n < (1 << 13)) {
      return seq_sparse_accumulator_histogram_compact<s_size_t, ct_t>(A, acc, m);
    }

    // generate sample
    get_bucket<E> x(A.as_array(), n, bits-1);
    auto get_buckets = make_sequence<size_t>(n, x);

    // first buckets based on hash, except for low 4 bits
    timer t0; t0.start();
    sequence<size_t> bucket_offsets
      = count_sort(A, A, get_buckets, num_buckets);
    t0.stop(); // t0.total();

    sequence<size_t> offsets(num_buckets+1);
    parallel_for_1(size_t i = 0; i < num_buckets; i++) {
      size_t start = bucket_offsets[i];
      size_t end = bucket_offsets[i+1];
      if (i < num_buckets/2) {
        offsets[i] = end - start;
      } else if (end > start) {
        offsets[i] = 1;
      } else {
        offsets[i] = 0;
      }
    }
    offsets[num_buckets] = 0;
    scan_add(offsets, offsets);

    s_size_t* tmp = new_array_no_init<s_size_t>(offsets[num_buckets]);
    sequence<size_t> c_offsets(num_buckets+1);

    timer t1; t1.start();
    // now sequentially process each bucket
    parallel_for_1(size_t i = 0; i < num_buckets; i++) {
      size_t start = bucket_offsets[i];
      size_t end = bucket_offsets[i+1];
      s_size_t* T = tmp + offsets[i];
      if (i < num_buckets/2) {
        // light bucket
        size_t k = 0;
        for (size_t j = start; j < end; j++) {
          s_size_t e = A[j];
          if (acc[e] == 0) {
            T[k++] = e;
          }
          acc[e]++;
        }
        c_offsets[i] = k;
      } else if (end > start) {
        // heavy bucket
        *T = A[start];
        c_offsets[i] = 1;
      } else {
        c_offsets[i] = 0;
      }
    }
    c_offsets[num_buckets] = 0;
    scan_add(c_offsets, c_offsets);
    t1.stop(); // t1.total();

    using OT = tuple<s_size_t, ct_t>;
    OT* out = new_array_no_init<OT>(c_offsets[num_buckets]);
    parallel_for_1(size_t i=0; i<num_buckets; i++) {
      size_t start = offsets[i];
      size_t sz = c_offsets[i+1] - c_offsets[i];
      s_size_t* T = tmp + start;
      OT* O = out + c_offsets[i];
      if (sz > 0) {
        for (size_t j=0; j<sz; j++) {
          s_size_t e = T[j];
          O[j] = make_tuple(e, acc[e]);
          acc[e] = 0;
        }
      }
    }
    free(tmp);
    size_t n_distinct_elms = c_offsets[num_buckets];
    return make_tuple(n_distinct_elms, out);
  }

  // n = A.size(): number of elements to histogram
  // m: elements are in the range [0, m)
  // Returns a sequence S of pairs of (elm, count) of size <= n.
  // Suppose we're also given as input a cleared array of size m (between [0, max_elm])
  // that we can use to accumulate values. The output is an array of pairs of distinct
  // indices, and the number of times they appeared in the input.
  template <typename s_size_t, typename ct_t, typename Seq>
  tuple<size_t, tuple<s_size_t, ct_t>*> sparse_accumulator_histogram_compact_no_transpose(Seq A, Seq acc, size_t m) {
    using E = typename Seq::T;
    size_t n = A.size();
    size_t bits;

    if (n < (1 << 27)) bits = (log2_up(n) - 7)/2;
    // for large n selected so each bucket fits into cache
    else bits = (log2_up(n) - 17);
    size_t num_buckets = (1<<bits);
    if (n < (1 << 13)) {
      return seq_sparse_accumulator_histogram_compact<s_size_t, ct_t>(A, acc, m);
    }

    // generate sample
    get_bucket<E> x(A.as_array(), n, bits-1);
    auto get_buckets = make_sequence<size_t>(n, x);

    timer t0; t0.start();
    auto ret = _count_sort_no_transpose_size<s_size_t>(A, get_buckets, num_buckets);
    t0.stop(); // t0.total();

    E* B = std::get<0>(ret);
    // Matrix: buckets (rows) by blocks (cols)
    s_size_t* per_block_counts = std::get<1>(ret);
    size_t num_blocks = std::get<2>(ret);
    size_t block_size = ((n-1)/num_blocks) + 1;

    timer t1; t1.start();
    sequence<size_t> offsets(num_buckets+1);
    // Write counts into offsets
    parallel_for_1 (size_t i = 0; i < num_buckets; i++) {
      offsets[i] = 0;
      if (i < num_buckets/2) {
        auto im = make_in_imap<size_t>(num_blocks, [&] (size_t j) { return per_block_counts[j*num_buckets + i + 1] - per_block_counts[j*num_buckets + i]; });
        size_t tot = reduce_add(im);
        offsets[i] = tot;
      } else {
        size_t k = 0;
        if (i == (num_buckets-1)) {
          auto im = make_in_imap<size_t>(num_blocks, [&] (size_t j) {
            size_t start = std::min(j * block_size, n);
            size_t end =  std::min(start + block_size, n);
            return end - start;});
          k += reduce_add(im);
        } else {
          auto im = make_in_imap<size_t>(num_blocks, [&] (size_t j) {
              return per_block_counts[j*num_buckets+i+1] - per_block_counts[j*num_buckets+i];
              });
          k += reduce_add(im);
        }
        if (k > 0) {
          offsets[i] = 1;
        }
      }
    }
    t1.stop(); // t1.total();
    offsets[num_buckets] = 0;
    scan_add(offsets, offsets);

    s_size_t* tmp = new_array_no_init<s_size_t>(offsets[num_buckets]);
    sequence<size_t> c_offsets(num_buckets+1);

    // now sequentially process each bucket
    parallel_for_1(size_t i = 0; i < num_buckets; i++) {
      c_offsets[i] = 0;
      s_size_t* T = tmp + offsets[i];
      if (i < num_buckets/2) {
        size_t n_distinct = 0;
        // light bucket
        for (size_t j=0; j<num_blocks; j++) {
          size_t start = std::min(j * block_size, n);
          size_t off = per_block_counts[j*num_buckets + i];
          size_t ct = per_block_counts[j*num_buckets + i + 1] - off;
          for (size_t k=0; k<ct; k++) {
            s_size_t e = B[start + off + k];
            if (acc[e] == 0) {
              T[n_distinct++] = e;
            }
            acc[e]++;
          }
        }
        c_offsets[i] = n_distinct;
      } else {
        size_t k = 0;
        s_size_t e = 0;
        bool set = false;
        if (i == (num_buckets-1)) {
          for (size_t j=0; j<num_blocks; j++) {
            size_t start = std::min(j * block_size, n);
            size_t end =  std::min(start + block_size, n);
            size_t delta = (end - start) - per_block_counts[j*num_buckets+i];
            if (set && delta > 0) {
              e = B[start];
              set = false;
            }
            k += delta;
          }
        } else {
          for (size_t j=0; j<num_blocks; j++) {
            size_t delta = per_block_counts[j*num_buckets+i+1] - per_block_counts[j*num_buckets+i];
            if (set && delta > 0) {
              e = B[per_block_counts[j*num_buckets+i]];
              set = false;
            }
            k += delta;
          }
        }
        if (k > 0) {
          *T = e;
          c_offsets[i] = 1;
        }
      }
    }


    c_offsets[num_buckets] = 0;
    scan_add(c_offsets, c_offsets);

    using OT = tuple<s_size_t, ct_t>;
    OT* out = new_array_no_init<OT>(c_offsets[num_buckets]);
    parallel_for_1(size_t i=0; i<num_buckets; i++) {
      size_t start = offsets[i];
      size_t sz = c_offsets[i+1] - c_offsets[i];
      s_size_t* T = tmp + start;
      OT* O = out + c_offsets[i];
      if (sz > 0) {
        for (size_t j=0; j<sz; j++) {
          s_size_t e = T[j];
          O[j] = make_tuple(e, acc[e]);
          acc[e] = 0;
        }
      }
    }
    free(tmp);
    size_t n_distinct_elms = c_offsets[num_buckets];
    return make_tuple(n_distinct_elms, out);
  }


  template <class s_size_t, class ct_t>
  std::unordered_map<s_size_t, ct_t> gen_map(size_t k, tuple<s_size_t, ct_t>* arr) {
    std::unordered_map<s_size_t, ct_t> out;
    for (size_t i=0; i<k; i++) {
      auto e = arr[i];
      out[std::get<0>(e)] = std::get<1>(e);
    }
    return out;
  }

  template <class s_size_t, class ct_t, class Seq>
  std::unordered_map<s_size_t, ct_t> gen_map(size_t k, s_size_t* arr, Seq acc) {
    std::unordered_map<s_size_t, ct_t> out;
    for (size_t i=0; i<k; i++) {
      auto e = arr[i];
      out[e] = acc[e];
    }
    return out;
  }


  template <class s_size_t, class ct_t, class Seq>
  std::unordered_map<s_size_t, ct_t> seq_hist(Seq in) {
    std::unordered_map<s_size_t, ct_t> out;
    for (size_t i=0; i<in.size(); i++) {
      auto e = in[i];
      if (out.find(e) == out.end()) {
        out[e] = 1;
      } else {
        out[e]++;
      }
    }
    return out;
  }

  template <class K, class V>
  bool maps_equal(std::unordered_map<K, V>& A, std::unordered_map<K, V>& B) {
    bool correct = true;
    for (const auto& kv : A) {
      auto k = kv.first;
      auto v_a = kv.second;
      auto f_b = B.find(k);
      if (f_b == B.end() || f_b->second != v_a) {
        correct = false;
      }
    }
    for (const auto& kv : B) {
      auto k = kv.first;
      auto v_b = kv.second;
      auto f_a = A.find(k);
      if (f_a == A.end() || f_a->second != v_b) {
        correct = false;
      }
    }
    return correct;
  }

} // namespace pbbs
