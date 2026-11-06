#pragma once

#include "utilities.h"
using namespace std;

namespace pbbsa {

template <class K, class V>
class sequentialHT {
 typedef tuple<K,V> T;

 public:
  size_t m;
  size_t mask;
  T empty;
  K max_key;
  T* table;
  bool alloc;
  size_t n_elms;

  inline size_t toRange(size_t h) {return h & mask;}
  inline size_t firstIndex(K v) {return toRange(pbbs::hash64(v));}
  inline size_t incrementIndex(size_t h) {return toRange(h+1);}

  sequentialHT(T* _table, size_t size, float loadFactor, tuple<K, V> _empty) :
    m((size_t) 1 << pbbs::log2_up((size_t)(loadFactor*size))),
    mask(m-1),
    empty(_empty),
    table(_table), alloc(false), n_elms(0) {
      max_key = get<0>(empty);
      if (table == nullptr) {
        table = pbbs::new_array_no_init<T>(m);
        alloc = true;
        for (size_t i=0; i<m; i++) {
          table[i] = empty;
        }
      }
    }

  // m must be a power of two
  sequentialHT(T* _table, size_t _m, tuple<K, V> _empty) :
    m((size_t)_m), mask(m-1), table(_table), empty(_empty), alloc(false), n_elms(0) { max_key = get<0>(empty); }

  inline void del() {
    if (alloc) {
      alloc = false;
      free(table);
    }
  }

  template <class M, class F>
  inline void insertF(tuple<K, M>& v, F& f) {
    K vKey = get<0>(v);
    size_t h = firstIndex(vKey);
    while (1) {
      auto k = get<0>(table[h]);
      if (k == max_key) {
        get<0>(table[h]) = vKey;
        V cur = get<1>(table[h]);
        get<1>(table[h]) = f(cur, v);
        n_elms++;
        return;
      } else if (k == vKey) {
        V cur = get<1>(table[h]);
        get<1>(table[h]) = f(cur, v);
        return;
      }
      h = incrementIndex(h);
    }
  }

  // V must support ++
  inline void insertAdd(K& vKey) {
    size_t h = firstIndex(vKey);
    while (1) {
      auto k = get<0>(table[h]);
      if (k == max_key) {
        table[h] = make_tuple(vKey, 1);
        n_elms++;
        return;
      } else if (k == vKey) {
        get<1>(table[h])++;
        return;
      }
      h = incrementIndex(h);
    }
  }

  // V must support ++, T<1> must be numeric
  inline void insertAdd(T& v) {
    const K& vKey = get<0>(v);
    size_t h = firstIndex(vKey);
    while (1) {
      auto k = get<0>(table[h]);
      if (k == max_key) {
        table[h] = make_tuple(vKey, 1);
        n_elms++;
        return;
      } else if (k == vKey) {
        get<1>(table[h]) += get<1>(v);
        return;
      }
      h = incrementIndex(h);
    }
  }

  inline T find(K& v) {
    size_t h = firstIndex(v);
    T c = table[h];
    while (1) {
      if (get<0>(c) == max_key) {
        return empty;
      } else if (get<0>(c) == v) {
      	return c;
      }
      h = incrementIndex(h);
      c = table[h];
    }
  }

  inline size_t compactInto(T* out) {
    size_t k = 0;
    for (size_t i=0; i<m; i++) {
      auto kv = table[i]; auto key = get<0>(kv);
      if (key != max_key) {
        table[i] = empty;
        out[k++] = kv;
      }
    }
    return k;
  }

  inline tuple<size_t, T*> compact() {
    T* out = pbbs::new_array_no_init<T>(m);
    size_t k = 0;
    for (size_t i=0; i<m; i++) {
      auto kv = table[i]; auto key = get<0>(kv);
      if (key != max_key) {
        table[i] = empty;
        out[k++] = kv;
      }
    }
    return make_tuple(k, out);
  }

};

// Struct is key; must have a hash field
template <class K, class V, class Eq>
class sequentialHTStruct {
 typedef tuple<K,V> T;

 public:
  size_t m;
  size_t mask;
  T empty;
  K max_key;
  T* table;
  bool alloc;
  size_t n_elms;
  Eq eq;

  inline size_t toRange(size_t h) {return h & mask;}
  inline size_t firstIndex(K v) {return toRange(pbbs::hash64(v.hash));}
  inline size_t incrementIndex(size_t h) {return toRange(h+1);}

  sequentialHTStruct(T* _table, size_t size, float loadFactor, tuple<K, V> _empty, Eq _eq) :
    m((size_t) 1 << pbbs::log2_up((size_t)(loadFactor*size))),
    mask(m-1),
    empty(_empty), eq(_eq),
    table(_table), alloc(false), n_elms(0) {
      max_key = get<0>(empty);
      if (table == nullptr) {
        table = pbbs::new_array_no_init<T>(m);
        alloc = true;
        for (size_t i=0; i<m; i++) {
          table[i] = empty;
        }
      }
    }

  inline void del() {
    if (alloc) {
      alloc = false;
      free(table);
    }
  }

  template <class M, class F>
  inline void insertF(tuple<K, M>& v, F& f) {
    K vKey = get<0>(v);
    size_t h = firstIndex(vKey);
    while (1) {
      auto k = get<0>(table[h]);
      if (eq(k, max_key)) {
        get<0>(table[h]) = vKey;
        V cur = get<1>(table[h]);
        get<1>(table[h]) = f(cur, v);
        n_elms++;
        return;
      } else if (eq(k, vKey)) {
        V cur = get<1>(table[h]);
        get<1>(table[h]) = f(cur, v);
        return;
      }
      h = incrementIndex(h);
    }
  }

  // V must support ++
  inline void insertAdd(K& vKey) {
    size_t h = firstIndex(vKey);
    while (1) {
      auto k = get<0>(table[h]);
      if (eq(k, max_key)) {
        table[h] = make_tuple(vKey, 1);
        n_elms++;
        return;
      } else if (eq(k, vKey)) {
        get<1>(table[h])++;
        return;
      }
      h = incrementIndex(h);
    }
  }

  // V must support ++, T<1> must be numeric
  inline void insertAdd(T& v) {
    const K& vKey = get<0>(v);
    size_t h = firstIndex(vKey);
    while (1) {
      auto k = get<0>(table[h]);
      if (eq(k, max_key)) {
        table[h] = make_tuple(vKey, 1);
        n_elms++;
        return;
      } else if (eq(k, vKey)) {
        get<1>(table[h]) += get<1>(v);
        return;
      }
      h = incrementIndex(h);
    }
  }

  inline T find(K& v) {
    size_t h = firstIndex(v);
    T c = table[h];
    while (1) {
      if (eq(get<0>(c), max_key)) {
        return empty;
      } else if (eq(get<0>(c), v)) {
      	return c;
      }
      h = incrementIndex(h);
      c = table[h];
    }
  }

  inline size_t compactInto(T* out) {
    size_t k = 0;
    for (size_t i=0; i<m; i++) {
      auto kv = table[i]; auto key = get<0>(kv);
      if (!eq(key, max_key)) {
        table[i] = empty;
        out[k++] = kv;
      }
    }
    return k;
  }

  inline tuple<size_t, T*> compact() {
    T* out = pbbs::new_array_no_init<T>(m);
    size_t k = 0;
    for (size_t i=0; i<m; i++) {
      auto kv = table[i]; auto key = get<0>(kv);
      if (!eq(key, max_key)) {
        table[i] = empty;
        out[k++] = kv;
      }
    }
    return make_tuple(k, out);
  }

};

template <class K, class V, class Hash>
class sequentialHTList {
 typedef tuple<K,sequentialHT<V,uintE>*> T;
 typedef tuple<tuple<K,V>,size_t> O;

 public:
  size_t m;
  size_t mask;
  T empty;
  K max_key;
  T* table;
  bool alloc;
  size_t n_elms;
  Hash max;

  inline size_t toRange(size_t h) {return h & mask;}
  inline size_t firstIndex(K v) {return toRange(pbbs::hash64(v));}
  inline size_t incrementIndex(size_t h) {return toRange(h+1);}

  sequentialHTList(T* _table, size_t size, float loadFactor, T _empty, Hash _max) ://tuple<K,uintE>* sizes, 
    m((size_t) 1 << pbbs::log2_up((size_t)(loadFactor*size))),
    mask(m-1),
    empty(_empty),
    table(_table), alloc(false), n_elms(0), max(_max) {
      max_key = get<0>(empty);
      if (table == nullptr) {
        table = pbbs::new_array_no_init<T>(m);
        alloc = true;
        for (size_t i=0; i<m; i++) {
          table[i] = empty;
        }
      }
      //for (long i=0; i < size; ++i) { insertInit(sizes[i]); }
    }

  inline void del() {
    if (alloc) {
      alloc = false;
      for (long i=0; i < m; ++i) {
        if (get<0>(table[i]) != max_key) get<1>(table[i])->del();
      }
      free(table);
    }
  }

  // V must support ++, T<1> must be numeric
  inline void insert(tuple<K,V>& v) {
    const K& vKey = get<0>(v);
    size_t h = firstIndex(vKey);
    while (1) {
      auto k = get<0>(table[h]);
      if (k == max_key) {
        sequentialHT<V,uintE>* ht = new sequentialHT<V,uintE>(nullptr, (max.find(vKey)).second, 1.0, make_tuple(numeric_limits<V>::max(), 0));
        table[h] = make_tuple(vKey, ht);
        get<1>(table[h])->insertAdd(get<1>(v));
        n_elms++;
        return;
      } else if (k == vKey) {
        get<1>(table[h])->insertAdd(get<1>(v));
        return;
      }
      h = incrementIndex(h);
    }
  }

  inline size_t compactInto(T* out) {
    size_t k = 0;
    for (size_t i=0; i<m; i++) {
      auto kv = table[i]; auto key = get<0>(kv);
      if (key != max_key) {
        table[i] = empty;
        out[k++] = kv;
      }
    }
    return k;
  }

  inline tuple<size_t, T*> compact() {
    T* out = pbbs::new_array_no_init<T>(m);
    size_t k = 0;
    for (size_t i=0; i<m; i++) {
      auto kv = table[i]; auto key = get<0>(kv);
      if (key != max_key) {
        table[i] = empty;
        out[k++] = kv;
      }
    }
    return make_tuple(k, out);
  }

  /*inline size_t compactInto(T* out) {
    size_t k = 0;
    for (size_t i=0; i<m; i++) {
      auto kv = table[i]; auto key = get<0>(kv);
      if (key != max_key) {
        table[i] = empty;
        out[k++] = kv;
      }
    }
    return k;
  }

  inline tuple<size_t, O*> compact() {
    O* out = pbbs::new_array_no_init<T>(max*m);
    size_t k = 0;
    for (size_t i=0; i<m; i++) {
      auto kv = table[i]; auto key = get<0>(kv);
      if (key != max_key) {
        auto v = get<1>(kv)->compact();
        for (size_t j=0; j < get<0>(v); ++j){ out[k++] = make_tuple(make_tuple(key, get<1>(v)[j]), get<0>(v)); }
        free(get<1>(v));
        get<1>(kv)->del();
        table[i] = empty;
      }
    }
    return make_tuple(k, out);
  }*/

};



} // namespace pbbs

