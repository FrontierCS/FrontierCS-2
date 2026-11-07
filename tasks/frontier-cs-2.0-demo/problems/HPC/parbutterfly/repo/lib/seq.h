#include "utilities.h"

namespace pbbsa {
template <typename E>
struct sequence {
public:
  using T = E;
  bool allocated = false;

  sequence() { s = NULL; e = NULL; allocated = false; }

  sequence(sequence&& b) 
    : s(b.s), e(b.e), allocated(b.allocated) {
    b.s = b.e = NULL; b.allocated = false;}

  sequence(sequence& a) {
    clear(); s = a.s; e = a.e;}

  sequence(const size_t n) 
    : s(new_array<E>(n)), allocated(true) {
    e = s + n;
  };

  sequence(const size_t n, T v) 
    : s(new_array_no_init<E>(n,1)), allocated(true) {
    e = s + n;
    parallel_for (size_t i = 0; i < n; i++) 
      new ((void*) (s+i)) T(v);
  };

  template <typename Func>
  sequence(const size_t n, Func fun) 
    : s(new_array_no_init<E>(n)), allocated(true) {
    e = s + n;
    parallel_for (size_t i = 0; i < n; i++) {
      T x = fun(i);
      new ((void*) (s+i)) T(x);
    }
  }

  sequence(E* s, const size_t n) : s(s), e(s + n) {};

  sequence(E* s, E* e) : s(s), e(e) {};

  ~sequence() { clear();}

  void resize(const size_t n) {
    if (!allocated || n > e-s) {
      clear();
      s = new_array_no_init<E>(n);
      allocated = true;
      e = s + n;
    }
  }

  template <typename X, typename F>
  static sequence<X> tabulate(const size_t n, F f) {
    X* r = new_array_no_init<X>(n);
    parallel_for (size_t i = 0; i < n; i++) 
      new ((void*) (r+i)) X(f(i));
    sequence<X> y(r,n);
    y.allocated = true;
    return y;
  }

  sequence copy(sequence& a) {
    return tabulate(e-s, [&] (size_t i) {return a[i];});
  }
    
  sequence& operator = (const sequence& m) {
    if (this != &m) {
      clear(); s = m.s; e = m.e; allocated = false;}
    return *this;
  }

  sequence& operator = (sequence&& m) {
    if (this != &m) {
      s = m.s; e = m.e; allocated = m.allocated;
      m.s = NULL; m.e = NULL; m.allocated=false;}
    return *this;
  }

  E& operator[] (const size_t i) const {return s[i];}

  sequence slice(size_t ss, size_t ee) { 
    return sequence(s + ss, s + ee);
  }

  void update(const size_t i, T& v) {
    s[i] = v;
  }

  size_t size() { return e - s;}

  T* as_array() {return s;}
  T* start() {return s;}
  T* end() {return e;}

private:
  void clear() {
    if (allocated) {
      delete_array<E>(s,e-s);
    }
    s = e = NULL;
    allocated = false;
  }
  E *s; // = NULL;
  E *e; // = NULL;
};

template <typename E, typename F>
struct func_sequence {
  using T = E;
  func_sequence(size_t n, F& _f) : f(&_f), s(0), e(n) {};
  func_sequence(size_t s, size_t e, F& _f) : f(&_f), s(s), e(e) {};
  T operator[] (const size_t i) {return (*f)(i+s);}
  func_sequence<T,F> slice(size_t ss, size_t ee) {
    return func_sequence<T,F>(s+ss,s+ee,*f); }
  size_t size() { return e - s;}
private:
  F *f;
  size_t s, e;
};

// used so second template argument can be inferred
template <class E, class F>
func_sequence<E,F> make_sequence (size_t n, F& f) {
  return func_sequence<E,F>(n,f);
}
}