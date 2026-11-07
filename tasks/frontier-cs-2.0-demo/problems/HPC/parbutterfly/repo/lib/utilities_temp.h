
#pragma once

#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include <ctype.h>
#include <memory>
#include <stdlib.h>
#include <type_traits>

#define parallel_for cilk_for
//#define parallel_for_1 _Pragma("cilk_grainsize = 1") cilk_for
#define parallel_for_1 cilk_for
#define parallel_for_16 _Pragma("cilk_grainsize = 16") cilk_for
#define parallel_for_256 _Pragma("cilk_grainsize = 256") cilk_for
size_t nworkers() {
  return __cilkrts_get_nworkers();
}

template <typename Lf, typename Rf>
static void par_do(bool do_parallel, Lf left, Rf right) {
  if (do_parallel) {
    cilk_spawn right();
    left();
    cilk_sync;
  } else {
    left(); right();
  }
}

template <typename F>
static void par_for(size_t start, size_t end, size_t granularity, F f) {
  if ((end - start) <= granularity)
    for (size_t i=start; i < end; i++) f(i);
  else {
    size_t mid = (end+start)/2;
    cilk_spawn par_for(start, mid, granularity, f);
    par_for(mid, end, granularity, f);
    cilk_sync;
  }
}

#include <malloc.h>
struct malloc_init {
  static int i;
  static int j;
};
int malloc_init::i = mallopt(M_MMAP_MAX,0);
int malloc_init::j = mallopt(M_TRIM_THRESHOLD,-1);

namespace pbbsa {

  typedef uint32_t flags;
  const flags no_flag = 0;
  const flags fl_sequential = 1;
  const flags fl_debug = 2;
  const flags fl_time = 4;

  template<typename T>
  inline void assign_uninitialized(T& a, const T& b) {
    new (static_cast<void*>(std::addressof(a))) T(b);
  }

  template<typename T>
  inline void move_uninitialized(T& a, const T& b) {
    new (static_cast<void*>(std::addressof(a))) T(std::move(b));
  }

  // a 32-bit hash function
  uint32_t hash32(uint32_t a) {
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
  }

  // from numerical recipes
  uint64_t hash64(uint64_t u )
  {
    uint64_t v = u * 3935559000370003845 + 2691343689449507681;
    v ^= v >> 21;
    v ^= v << 37;
    v ^= v >>  4;
    v *= 4768777513237032717;
    v ^= v << 20;
    v ^= v >> 41;
    v ^= v <<  5;
    return v;
  }

  // Does not initialize the array
  template<typename E>
  E* new_array_no_init(size_t n, bool touch_pages=false) {
    // pads in case user wants to allign with cache lines
    size_t line_size = 64;
    size_t bytes = ((n * sizeof(E))/line_size + 1)*line_size;
    E* r = (E*) aligned_alloc(line_size, bytes);
    if (r == NULL) {fprintf(stderr, "Cannot allocate space"); exit(1);}
    // a hack to make sure tlb is full for huge pages
    if (touch_pages)
      parallel_for (size_t i = 0; i < bytes; i = i + (1 << 21))
	((bool*) r)[i] = 0;
    return r;
  }

  // Initializes in parallel
  template<typename E>
  E* new_array(size_t n) {
    E* r = new_array_no_init<E>(n);
    if (!std::is_trivially_default_constructible<E>::value) {
      if (n > 2048)
	parallel_for (size_t i = 0; i < n; i++) new ((void*) (r+i)) E;
      else
	for (size_t i = 0; i < n; i++) new ((void*) (r+i)) E;
    }
    return r;
  }

  // Destructs in parallel
  template<typename E>
  void delete_array(E* A, size_t n) {
    // C++14 -- suppored by gnu C++11
    if (!std::is_trivially_destructible<E>::value) {
      if (n > 2048)
	parallel_for (size_t i = 0; i < n; i++) A[i].~E();
      else
	for (size_t i = 0; i < n; i++) A[i].~E();
    }
    free(A);
  }

  template <typename ET>
  inline bool CAS_GCC(ET *ptr, ET oldv, ET newv) {
    return __sync_bool_compare_and_swap(ptr, oldv, newv);
  }

  template <typename E, typename EV>
  inline E fetch_and_add(E *a, EV b) {
    volatile E newV, oldV;
    do {oldV = *a; newV = oldV + b;}
    while (!CAS_GCC(a, oldV, newV));
    return oldV;
  }

  template <typename E, typename EV>
  inline void write_add(E *a, EV b) {
    volatile E newV, oldV;
    do {oldV = *a; newV = oldV + b;}
    while (!CAS_GCC(a, oldV, newV));
  }

  template <typename ET, typename F>
  inline bool write_min(ET *a, ET b, F less) {
    ET c; bool r=0;
    do c = *a;
    while (less(b,c) && !(r=CAS_GCC(a,c,b)));
    return r;
  }

  // returns the log base 2 rounded up (works on ints or longs or unsigned versions)
  template <class T>
  static int log2_up(T i) {
    int a=0;
    T b=i-1;
    while (b > 0) {b = b >> 1; a++;}
    return a;
  }
}


