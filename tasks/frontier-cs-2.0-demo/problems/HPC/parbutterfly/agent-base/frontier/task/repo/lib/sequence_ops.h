// This code is part of the Problem Based Benchmark Suite (PBBS)
// Copyright (c) 2011-2016 Guy Blelloch and the PBBS team
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

#include <iostream>
#include "utilities.h"
#include "seq.h"

namespace pbbsa {
  using namespace std;

  constexpr const size_t _log_block_size = 12;
  constexpr const size_t _block_size = (1 << _log_block_size);

  inline size_t num_blocks(size_t n, size_t block_size) {
    return (1 + ((n)-1)/(block_size));}

  template <class F>
  void sliced_for(size_t n, size_t block_size, const F& f) {
    size_t l = num_blocks(n, block_size);
    parallel_for_1 (size_t i = 0; i < l; i++) {
      size_t s = i * block_size;
      size_t e = min(s + block_size, n);
      f(i, s, e);
    }
  }

  template <class Seq, class F>
  auto reduce_serial(Seq A, const F& f) -> typename Seq::T {
    using T = typename Seq::T;
    T r = A[0];
    for (size_t j=1; j < A.size(); j++) r = f(r,A[j]);
    return r;
  }

  template <class Seq, class F>
  auto reduce(Seq A, const F& f, flags fl = no_flag)
    -> typename Seq::T
  {
    using T = typename Seq::T;
    size_t n = A.size();
    size_t l = num_blocks(n, _block_size);
    if (l <= 1 || (fl & fl_sequential))
      return reduce_serial(A, f);
    sequence<T> Sums(l);
    sliced_for (n, _block_size,
		[&] (size_t i, size_t s, size_t e)
		{ Sums[i] = reduce_serial(A.slice(s,e), f);});
    T r = reduce_serial(Sums, f);
    return r;
  }

  // sums I with + (can be any type with + defined)
  template <class Seq>
  auto reduce_add(Seq I, flags fl = no_flag)
    -> typename Seq::T
  {
    using T = typename Seq::T;
    auto add = [] (T x, T y) {return x + y;};
    return reduce(I, add, fl);
  }

  const flags fl_scan_inclusive = (1 << 4);

  // serial scan with combining function f and start value zero
  // fl_scan_inclusive indicates it includes the current location
  template <class In_Seq, class Out_Seq, class F>
  auto scan_serial(In_Seq In, Out_Seq Out,
		   const F& f, typename In_Seq::T zero,
		   flags fl = no_flag)  -> typename In_Seq::T
  {
    using T = typename In_Seq::T;
    T r = zero;
    size_t n = In.size();
    bool inclusive = fl & fl_scan_inclusive;
    if (inclusive) {
      for (size_t i = 0; i < n; i++) {
	r = f(r,In[i]);
	Out.update(i,r);
      }
    } else {
      for (size_t i = 0; i < n; i++) {
	T t = In[i];
	Out.update(i,r);
	r = f(r,t);
      }
    }
    return r;
  }

  // parallel version of scan_serial -- see comments above
  template <class In_Seq, class Out_Seq, class F>
  auto scan(In_Seq In, Out_Seq Out,
	    const F& f, typename In_Seq::T zero,
	    flags fl = no_flag)  -> typename In_Seq::T
  {
    using T = typename In_Seq::T;
    size_t n = In.size();
    size_t l = num_blocks(n,_block_size);
    if (l <= 2 || fl & fl_sequential)
      return scan_serial(In, Out, f, zero, fl);
    sequence<T> Sums(l);
    sliced_for (n, _block_size,
		[&] (size_t i, size_t s, size_t e)
		{ Sums[i] = reduce_serial(In.slice(s,e),f);});
    T total = scan_serial(Sums, Sums, f, zero, 0);
    sliced_for (n, _block_size,
		[&] (size_t i, size_t s, size_t e)
		{ scan_serial(In.slice(s,e), Out.slice(s,e), f, Sums[i], fl);});
    return total;
  }

  template <class In_Seq, class Out_Seq>
  auto scan_add(In_Seq In, Out_Seq Out, flags fl = no_flag)
    -> typename In_Seq::T
  {
    using T = typename In_Seq::T;
    auto add = [] (T x, T y) {return x + y;};
    return scan(In, Out, add, (T) 0, fl);
  }

  template <class Seq>
  size_t sum_flags_serial(Seq I) {
    size_t r = 0;
    for (size_t j=0; j < I.size(); j++) r += I[j];
    return r;
  }

  template <class In_Seq, class Bool_Seq>
  auto pack_serial(In_Seq In, Bool_Seq Fl)
    -> sequence<typename In_Seq::T> {
    using T = typename In_Seq::T;
    size_t n = In.size();
    size_t m = sum_flags_serial(Fl);
    T* Out = new_array_no_init<T>(m);
    size_t k = 0;
    for (size_t i=0; i < n; i++)
      if (Fl[i]) assign_uninitialized(Out[k++], In[i]);
    return sequence<T>(Out,m);
  }

  template <class In_Seq, class Bool_Seq>
  void pack_serial_at(In_Seq In, typename In_Seq::T* Out, Bool_Seq Fl) {
    size_t k = 0;
    for (size_t i=0; i < In.size(); i++)
      if (Fl[i]) assign_uninitialized(Out[k++], In[i]);
  }

  template <class In_Seq, class Bool_Seq>
  auto pack(In_Seq In, Bool_Seq Fl, flags fl = no_flag)
    -> sequence<typename In_Seq::T>
  {
    using T = typename In_Seq::T;
    size_t n = In.size();
    size_t l = num_blocks(n,_block_size);
    if (l <= 1 || fl & fl_sequential)
      return pack_serial(In, Fl);
    sequence<size_t> Sums(l);
    sliced_for (n, _block_size,
		[&] (size_t i, size_t s, size_t e)
		{ Sums[i] = sum_flags_serial(Fl.slice(s,e));});
    size_t m = scan_add(Sums, Sums);
    T* Out = new_array_no_init<T>(m);
    sliced_for (n, _block_size,
		[&] (size_t i, size_t s, size_t e)
		{ pack_serial_at(In.slice(s,e),
				 Out + Sums[i],
				 Fl.slice(s,e));});
    return sequence<T>(Out,m);
  }

  template <class Idx_Type, class Bool_Seq>
  sequence<Idx_Type> pack_index(Bool_Seq Fl, flags fl = no_flag) {
    auto identity = [] (size_t i) {return (Idx_Type) i;};
    return pack(make_sequence<Idx_Type>(Fl.size(),identity), Fl, fl);
  }

//  template <class T, class PRED>
//  size_t filter_serial(T* In, T* Out, size_t n, PRED p) {
//    size_t k = 0;
//    for (size_t i = 0; i < n; i++)
//      if (p(In[i])) Out[k++] = In[i];
//    return k;
//  }
//
//   // Faster for a small number in output (about 40% or less)
//   // Destroys the input.   Does not need a bool array.
//   template <class T, class PRED>
//   size_t filterf(T* In, T* Out, size_t n, PRED p) {
//     size_t b = _F_BSIZE;
//     if (n < b)
//       return filter_serial(In, Out, n, p);
//     size_t l = nblocks(n, b);
//     b = nblocks(n, l);
//     size_t *Sums = new_array_no_init<size_t>(l + 1);
//     parallel_for_1 (size_t i = 0; i < l; i++) {
//       size_t s = i * b;
//       size_t e = min(s + b, n);
//       size_t k = s;
//       for (size_t j = s; j < e; j++)
//    if (p(In[j])) In[k++] = In[j];
//       Sums[i] = k - s;
//     }
//     auto isums = array_imap<size_t>(Sums,l);
//     size_t m = scan_add(isums, isums);
//     Sums[l] = m;
//     parallel_for_1 (size_t i = 0; i < l; i++) {
//       T* I = In + i*b;
//       T* O = Out + Sums[i];
//       for (size_t j = 0; j < Sums[i+1]-Sums[i]; j++) {
//        O[j] = I[j];
//       }
//     }
//     free(Sums);
//     return m;
//   }

  //template <class In_Seq, class Out_Seq, class PRED>
  //size_t filter_serial(In_Seq In, Out_Seq Out, PRED p) {
  //   size_t k = 0;
  //   for (size_t i = 0; i < n; i++)
  //     if (p(In[i])) Out[k++] = In[i];
  //   return k;
  // }

  // template <class T, class PRED>
  // size_t filter(T* In, T* Out, size_t n, PRED p, flags fl = no_flag) {
  //   if (n < _F_BSIZE || fl & fl_sequential)
  //     return filter_serial(In, Out, n, p);
  //   bool *Flags = new_array_no_init<bool>(n);
  //   parallel_for (size_t i=0; i < n; i++) Flags[i] = (bool) p(In[i]);
  //   size_t  m = pack(In, Out, Flags, n);
  //   free(Flags);
  //   return m;
  // }

  // // Avoids reallocating the bool array
  // template <class T, class PRED>
  // size_t filter(T* In, T* Out, bool* Flags, size_t n, PRED p,
  // 		flags fl = no_flag) {
  //   if (n < _F_BSIZE || fl & fl_sequential)
  //     return filter_serial(In, Out, n, p);
  //   parallel_for (size_t i=0; i < n; i++) Flags[i] = (bool) p(In[i]);
  //   size_t  m = pack(In, Out, Flags, n);
  //   return m;
  // }

  // template <class T, class PRED>
  // seq<T> filter(T* In, size_t n, PRED p, flags fl = no_flag) {
  //   bool *Fl = new_array_no_init<bool>(n);
  //   parallel_for (size_t i=0; i < n; i++) Fl[i] = (bool) p(In[i]);
  //   seq<T> R = pack(In, Fl, n);
  //   free(Fl);
  //   return R;
  // }

  // // g maps indices to input elements
  // // f maps indices to boolean flags
  // template <class T, class G, class F>
  // seq<T> filter(size_t n, G g, F f) {
  //   bool *Fl = new_array_no_init<bool>(n);
  //   parallel_for (size_t i=0; i < n; i++)
  //     Fl[i] = (bool) f(i);
  //   seq<T> R = pack<T>((T*) NULL, Fl, 0, n, g);
  //   free(Fl);
  //   return R;
  // }

  // // Faster for a small number in output (about 40% or less)
  // // Destroys the input.   Does not need a bool array.
  // template <class T, class PRED>
  // size_t filterf(T* In, T* Out, size_t n, PRED p) {
  //   size_t b = _F_BSIZE;
  //   if (n < b)
  //     return filter_serial(In, Out, n, p);
  //   size_t l = nblocks(n, b);
  //   b = nblocks(n, l);
  //   size_t *Sums = new_array_no_init<size_t>(l + 1);
  //   parallel_for_1 (size_t i = 0; i < l; i++) {
  //     size_t s = i * b;
  //     size_t e = min(s + b, n);
  //     size_t k = s;
  //     for (size_t j = s; j < e; j++)
  // 	if (p(In[j])) In[k++] = In[j];
  //     Sums[i] = k - s;
  //   }
  //   auto isums = array_imap<size_t>(Sums,l);
  //   size_t m = scan_add(isums, isums);
  //   Sums[l] = m;
  //   parallel_for_1 (size_t i = 0; i < l; i++) {
  //     T* I = In + i*b;
  //     T* O = Out + Sums[i];
  //     for (size_t j = 0; j < Sums[i+1]-Sums[i]; j++) {
  // 	O[j] = I[j];
  //     }
  //   }
  //   free(Sums);
  //   return m;
  // }

  // template <class T, class F>
  // void split_serial(T* OutFalse, T* OutTrue, bool* Fl,
  // 		    size_t s, size_t e, const F& f) {
  //   size_t kT = 0;
  //   size_t kF = 0;
  //   for (size_t i=s; i < e; i++)
  //     if (Fl[i]) { assign_uninitialized(OutTrue[kT], f(i)); ++kT; }
  //     else { assign_uninitialized(OutFalse[kF], f(i)); ++kF; }
  // }

  // // Given a boolean array, splits so false (0) elements are at the bottom
  // // and true (1) elements are at the top of the output (of lenght e-s).
  // // As usual s is a start index, e is an end index and
  // // f is a function of type [s,e-1) -> T
  // template <class T, class F>
  //   int split(T* Out, bool*  Fl, size_t s, size_t e, const F& f) {
  //   size_t l = nblocks(e-s, _F_BSIZE);
  //   size_t *sums_true = new_array_no_init<size_t>(l);
  //   blocked_for (i, s, e, _F_BSIZE,
  // 		 sums_true[i] = sum_flags_serial(Fl+s, e-s););
  //   auto sums = array_imap<size_t>(sums_true, l);
  //   size_t num_true = scan_add(sums, sums);
  //   size_t num_false = (e - s) - num_true;
  //   T* out_true = Out + num_false;
  //   blocked_for(i, s, e, _F_BSIZE,
  // 		split_serial(Out + _F_BSIZE*i - sums_true[i],
  // 			    out_true + sums_true[i],
  // 			    Fl, s, e, f););
  //   free(sums_true);
  //   return num_false;
  // }

  // template <class T>
  // size_t split(T* In, T* Out, bool* Fl, size_t n) {
  //   auto get = [In] (size_t i) {return In[i];};
  //   return split(Out, Fl, (size_t) 0, n, get);
  // }
}

