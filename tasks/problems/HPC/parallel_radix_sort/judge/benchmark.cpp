#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <sys/resource.h>
#include <utility>
#include <vector>
using Clock = std::chrono::steady_clock;

#ifdef FRONTIER_COMPONENT_TIMER
// Beat spec v3: time spent inside the editable region-graph component (graph
// construction, 2-path planning, node deletion, graph update and destruction)
// during each sort call, summed over Cilk workers. The retained sorter keeps
// calling the same methods; only this harness binary wraps them.
#include "sequence.h"
#include "utils.h"
#include "parallel.h"
#include "radix_configs.h"
#include "edgelistgraph.h"
static std::atomic<long long> component_ns(0);
static inline long long component_now() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
}
struct ComponentDestructionEnd {
  long long start = 0;
  ~ComponentDestructionEnd() { component_ns.fetch_add(component_now() - start); }
};
struct ComponentConstructionStart {
  long long start;
  ComponentConstructionStart() : start(component_now()) {}
};
class TimedEdgeListGraph : private ComponentDestructionEnd, private ComponentConstructionStart, public EdgeListGraph {
 public:
  TimedEdgeListGraph(int P, int *rank, int *order)
      : ComponentDestructionEnd(), ComponentConstructionStart(), EdgeListGraph(P, rank, order) {
    component_ns.fetch_add(component_now() - ComponentConstructionStart::start);
  }
  ~TimedEdgeListGraph() { ComponentDestructionEnd::start = component_now(); }
  void createParallelGraph(long buckets, long P, SimpleBlock *blocks, const long *countryEnds) {
    long long t = component_now(); EdgeListGraph::createParallelGraph(buckets, P, blocks, countryEnds);
    component_ns.fetch_add(component_now() - t);
  }
  void extractNode2(int node, Triangle *triangles, int *triangles_count) {
    long long t = component_now(); EdgeListGraph::extractNode2(node, triangles, triangles_count);
    component_ns.fetch_add(component_now() - t);
  }
  void deleteNode(int node) {
    long long t = component_now(); EdgeListGraph::deleteNode(node);
    component_ns.fetch_add(component_now() - t);
  }
  void consumeTriangle(Triangle *triangle) {
    long long t = component_now(); EdgeListGraph::consumeTriangle(triangle);
    component_ns.fetch_add(component_now() - t);
  }
};
#define EdgeListGraph TimedEdgeListGraph
#endif

#include "sequence.h"
#include "radixSort.h"
using Pair = std::pair<uintT,uintT>;

#ifdef FRONTIER_COMPONENT_TIMER
#define COMPONENT_RESET() component_ns.store(0)
#define COMPONENT_OR(sort_ns) ((long long)component_ns.load())
#else
#define COMPONENT_RESET() ((void)0)
#define COMPONENT_OR(sort_ns) (sort_ns)
#endif

// Allocation accounting is enabled only around parallelIntegerSort.  A header
// records whether each allocation belongs to that interval, so later frees do
// not confuse the concurrent live/peak counters.
struct alignas(std::max_align_t) AllocationHeader { size_t size; bool counted; };
static std::atomic<bool> allocation_enabled(false);
static std::atomic<size_t> allocation_live(0), allocation_peak(0);
extern "C" void *__real_malloc(size_t);
extern "C" void __real_free(void *);

static void *tracked_allocate(size_t size) {
  AllocationHeader *h=(AllocationHeader*)__real_malloc(sizeof(AllocationHeader)+size);
  if(!h) throw std::bad_alloc();
  h->size=size; h->counted=allocation_enabled.load(std::memory_order_relaxed);
  if(h->counted){size_t live=allocation_live.fetch_add(size)+size, peak=allocation_peak.load();
    while(live>peak&&!allocation_peak.compare_exchange_weak(peak,live)){} }
  return h+1;
}
static void tracked_release(void *ptr){if(!ptr)return;AllocationHeader*h=((AllocationHeader*)ptr)-1;
  if(h->counted)allocation_live.fetch_sub(h->size);__real_free(h);}
void *operator new(size_t n){return tracked_allocate(n);} void *operator new[](size_t n){return tracked_allocate(n);}
void operator delete(void*p) noexcept{tracked_release(p);} void operator delete[](void*p) noexcept{tracked_release(p);}
void operator delete(void*p,size_t) noexcept{tracked_release(p);} void operator delete[](void*p,size_t) noexcept{tracked_release(p);}
#define WRAP_VISIBLE __attribute__((used, externally_visible))
extern "C" WRAP_VISIBLE void *__wrap_malloc(size_t n){return tracked_allocate(n);}
extern "C" WRAP_VISIBLE void *__wrap_calloc(size_t n,size_t s){size_t z=n*s;void*p=tracked_allocate(z);std::fill_n((unsigned char*)p,z,0);return p;}
extern "C" WRAP_VISIBLE void __wrap_free(void*p){tracked_release(p);}
extern "C" WRAP_VISIBLE void *__wrap_realloc(void*p,size_t n){if(!p)return tracked_allocate(n);AllocationHeader*h=((AllocationHeader*)p)-1;
  size_t old=h->size;void*q=tracked_allocate(n);std::copy_n((unsigned char*)p,std::min(old,n),(unsigned char*)q);tracked_release(p);return q;}

// Count allocations retained by earlier sort calls until they are actually freed.
static void begin_accounting(){allocation_peak.store(allocation_live.load());allocation_enabled.store(true);}
static size_t end_accounting(){allocation_enabled.store(false);return allocation_peak.load();}

static uint32_t draw(std::mt19937 &g, const std::string &dist, size_t i, size_t n) {
  if (dist == "equal") return 7;
  if (dist == "duplicates") return g() & 1023u;
  if (dist == "zipf") { uint32_t x=g(); return x%100<70 ? x&31u : x&0xffffu; }
  if (dist == "sorted" || dist == "almost") return (uint32_t)i;
  if (dist == "zipf075") {
    // Beat spec v3: Zipf-like keys with exponent 0.75 over [1, 1e9] (the
    // paper's zipf_0.75(1e9, n)) by continuous inverse-CDF sampling:
    // P(key >= x) ~ x^0.25, x = (1 + (1e9^0.25 - 1) u)^4.
    double u = (g() + 0.5) / 4294967296.0;
    double t = 1.0 + 176.82794100389228 * u;
    double x = (t * t) * (t * t);
    return x >= 1e9 ? 1000000000u : (uint32_t)x;
  }
  return g();
}

template<class T, class F>
static int run(size_t n, const std::string &dist, uint32_t seed, int reps, F key) {
  std::mt19937 g(seed); std::vector<T> source(n);
  for(size_t i=0;i<n;i++) source[i]=T(draw(g,dist,i,n));
  if(dist=="almost") for(size_t i=0;i<n/100;i++) std::swap(source[g()%n],source[g()%n]);
  std::vector<T> expected=source; std::sort(expected.begin(),expected.end(),[&](const T&a,const T&b){return key(a)<key(b);});
  std::vector<T> a=source;
  std::vector<long long> times;
  size_t aux=0;
  for(int r=0;r<reps;r++) { a=source; COMPONENT_RESET(); begin_accounting(); auto t0=Clock::now(); parallelIntegerSort(a.data(),(sizeT)a.size(),key); auto t1=Clock::now(); aux=std::max(aux,end_accounting());
    if(a!=expected) { std::cout<<"INVALID\n"; return 2; }
    times.push_back(COMPONENT_OR(std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count()));
  }
  std::sort(times.begin(),times.end());
  std::cout<<"OK "<<times[times.size()/2]<<" "<<aux<<"\n"; return 0;
}

template<class F> static int run_pairs(size_t n,const std::string&d,uint32_t s,int reps,F key){
  std::mt19937 g(s); std::vector<Pair> source(n); for(size_t i=0;i<n;i++) source[i]={draw(g,d,i,n),(uintT)i};
  if(d=="almost") for(size_t i=0;i<n/100;i++) std::swap(source[g()%n],source[g()%n]);
  std::vector<Pair> expected=source; std::sort(expected.begin(),expected.end());
  std::vector<Pair> a=source; std::vector<long long> times; size_t aux=0;
  for(int r=0;r<reps;r++){a=source;COMPONENT_RESET();begin_accounting();auto t0=Clock::now();parallelIntegerSort(a.data(),(sizeT)a.size(),key);auto t1=Clock::now();aux=std::max(aux,end_accounting());
    bool ordered=std::is_sorted(a.begin(),a.end(),[&](const Pair&x,const Pair&y){return key(x)<key(y);}); auto multiset=a; std::sort(multiset.begin(),multiset.end());
    if(!ordered||multiset!=expected){std::cout<<"INVALID\n";return 2;}
    times.push_back(COMPONENT_OR(std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count()));}
  std::sort(times.begin(),times.end());
  std::cout<<"OK "<<times[times.size()/2]<<" "<<aux<<"\n";return 0;
}

int main(int argc,char**argv){if(argc!=6)return 64;std::string type=argv[1],dist=argv[2];size_t n=std::stoull(argv[3]);uint32_t seed=std::stoul(argv[4]);int reps=std::stoi(argv[5]);
  if(type=="int") return run<uintT>(n,dist,seed,reps,utils::identityF<uintT>());
  return run_pairs(n,dist,seed,reps,utils::firstF<uintT,uintT>());
}
