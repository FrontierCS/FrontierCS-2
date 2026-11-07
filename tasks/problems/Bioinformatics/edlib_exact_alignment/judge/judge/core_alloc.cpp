// Trusted, untimed accounting of reconstruction storage. No dynamically
// allocated collector metadata.
//
// Heap: every malloc-family entry point is DEFINED here, so that requests made
// by the reconstruction, by the C++ runtime (operator new) and inside libc on
// its behalf all pass through record()/erase(). Direct references from the
// measured objects to the libc internals behind them (__libc_malloc and
// friends) and to mmap/munmap are redirected here by the linker (--wrap).
// Other ways of obtaining memory (mremap, brk, a raw syscall, the dynamic
// loader) end the process: they cannot be attributed.
//
// Stack: at the outermost core entry the region below the core's frame is
// painted; at exit the lowest overwritten word gives the high-water mark,
// measured from the core's own frame address.
#include <cerrno>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sys/mman.h>
#include <unistd.h>
extern "C" {
void* __real___libc_malloc(size_t);
void* __real___libc_calloc(size_t,size_t);
void* __real___libc_realloc(void*,size_t);
void* __real___libc_memalign(size_t,size_t);
void __real___libc_free(void*);
void* __real_mmap(void*,size_t,int,int,int,off_t);
void* __real_mmap64(void*,size_t,int,int,int,off_t);
int __real_munmap(void*,size_t);
long __real_syscall(long,...);
}
namespace {
constexpr size_t slots=131071;
constexpr size_t stack_window=size_t(4)<<20;      // measurable stack below the core frame
constexpr size_t paint_slack=512;                 // this collector's own frames
constexpr uint64_t paint=0x9d5ec0de7a11f1e5ull;
struct Entry { void* ptr; size_t bytes; unsigned state; };
Entry table[slots];
size_t live_bytes=0,peak_bytes=0,depth=0,entries=0,allocations=0,epochs=0;
uintptr_t frame_top=0,paint_low=0,paint_high=0;
[[noreturn]] void fail() { _exit(86); }
size_t index_of(void* p) { return (reinterpret_cast<uintptr_t>(p)>>3)%slots; }
void record(void* p,size_t n) {
 if(!p || !depth) return;
 size_t start=index_of(p),empty=slots;
 for(size_t j=0;j<slots;j++) {
  size_t i=(start+j)%slots;
  if(table[i].state!=1 && empty==slots) empty=i;
  if(table[i].state==0) break;
  if(table[i].state==1 && table[i].ptr==p) fail();
 }
 if(empty==slots || n>std::numeric_limits<size_t>::max()-live_bytes) fail();
 table[empty]={p,n,1};live_bytes+=n;if(live_bytes>peak_bytes)peak_bytes=live_bytes;++allocations;
}
// Returns the recorded size, or 0 if the block was not recorded in this epoch.
size_t erase(void* p,size_t at_least=0) {
 if(!p) return 0;
 size_t start=index_of(p);
 for(size_t j=0;j<slots;j++) {
  Entry &e=table[(start+j)%slots];if(e.state==0)return 0;
  if(e.state==1 && e.ptr==p) {
   if(e.bytes>live_bytes)fail();
   if(at_least && at_least<e.bytes) return 0;      // partial unmapping stays counted
   live_bytes-=e.bytes;e.state=2;return e.bytes;
  }
 }
 return 0;
}
__attribute__((noinline)) void paint_stack(uintptr_t frame) {
 volatile char here=0;
 uintptr_t high=(reinterpret_cast<uintptr_t>(&here)-paint_slack)&~uintptr_t(7);
 if(frame<reinterpret_cast<uintptr_t>(&here) || frame-reinterpret_cast<uintptr_t>(&here)>stack_window/2) fail();
 uintptr_t low=(frame-stack_window)&~uintptr_t(7);
 for(uintptr_t p=high;p>low;) { p-=8; *reinterpret_cast<volatile uint64_t*>(p)=paint; }
 frame_top=frame;paint_low=low;paint_high=high;
}
__attribute__((noinline)) size_t stack_used() {
 if(*reinterpret_cast<volatile uint64_t*>(paint_low)!=paint) fail();   // deeper than the window
 for(uintptr_t p=paint_low;p<paint_high;p+=8)
  if(*reinterpret_cast<volatile uint64_t*>(p)!=paint) return frame_top-p;
 return frame_top-paint_high;
}
void* aligned(size_t alignment,size_t n) { void* p=__real___libc_memalign(alignment,n);record(p,n);return p; }
}
extern "C" void edlib_core_enter(void* frame) {
 if(depth++==0){
  if(epochs++) fail();                              // one outermost reconstruction per process
  std::memset(table,0,sizeof(table));live_bytes=peak_bytes=entries=allocations=0;
  paint_stack(reinterpret_cast<uintptr_t>(frame));
 }
 ++entries;
}
extern "C" void edlib_core_leave() {
 if(!depth)fail();
 if(--depth==0) {
  const size_t stack=stack_used();
  std::fprintf(stderr,"EDLIB_CORE_ALLOCATION peak=%zu live_at_exit=%zu calls=%zu entries=%zu stack=%zu\n",peak_bytes,live_bytes,allocations,entries,stack);
 }
}
extern "C" {
void* malloc(size_t n){void* p=__real___libc_malloc(n);record(p,n);return p;}
void* calloc(size_t n,size_t s){
 void* p=__real___libc_calloc(n,s);
 if(p) record(p,n*s);                               // success implies n*s did not overflow
 return p;
}
void free(void* p){erase(p);__real___libc_free(p);}
void* realloc(void* p,size_t n){
 void* q=__real___libc_realloc(p,n);
 // Linux/glibc realloc(p,0) releases p; failed nonzero realloc preserves p.
 if(q || n==0){erase(p);record(q,n);}return q;
}
void* reallocarray(void* p,size_t n,size_t s){
 if(s && n>std::numeric_limits<size_t>::max()/s){errno=ENOMEM;return nullptr;}
 return realloc(p,n*s);
}
void* memalign(size_t alignment,size_t n){return aligned(alignment,n);}
void* aligned_alloc(size_t alignment,size_t n){return aligned(alignment,n);}
int posix_memalign(void** out,size_t alignment,size_t n){
 if(alignment%sizeof(void*) || !alignment || (alignment&(alignment-1))) return EINVAL;
 void* p=aligned(alignment,n);if(!p) return ENOMEM;
 *out=p;return 0;
}
void* valloc(size_t n){return aligned(static_cast<size_t>(sysconf(_SC_PAGESIZE)),n);}
void* pvalloc(size_t n){
 const size_t page=static_cast<size_t>(sysconf(_SC_PAGESIZE));
 if(n>std::numeric_limits<size_t>::max()-page){errno=ENOMEM;return nullptr;}
 return aligned(page,(n+page-1)/page*page);
}
// References from the measured objects to the allocator behind malloc.
void* __wrap___libc_malloc(size_t n){return malloc(n);}
void* __wrap___libc_calloc(size_t n,size_t s){return calloc(n,s);}
void* __wrap___libc_realloc(void* p,size_t n){return realloc(p,n);}
void* __wrap___libc_memalign(size_t alignment,size_t n){return aligned(alignment,n);}
void __wrap___libc_free(void* p){free(p);}
void* __wrap_mmap(void* a,size_t n,int prot,int flags,int fd,off_t off){
 void* p=__real_mmap(a,n,prot,flags,fd,off);if(p!=MAP_FAILED)record(p,n);return p;
}
void* __wrap_mmap64(void* a,size_t n,int prot,int flags,int fd,off_t off){
 void* p=__real_mmap64(a,n,prot,flags,fd,off);if(p!=MAP_FAILED)record(p,n);return p;
}
int __wrap_munmap(void* p,size_t n){erase(p,n);return __real_munmap(p,n);}
// Memory that cannot be attributed is not available to the measured objects.
void* __wrap_mremap(void*,size_t,size_t,int,...){fail();}
void* __wrap_sbrk(intptr_t){fail();}
int __wrap_brk(void*){fail();}
// The generic entry point stays usable, except for the calls that obtain or move memory.
long __wrap_syscall(long number,...){
 if(number==9 || number==12 || number==25 || number==30) fail();   // mmap brk mremap shmat
 va_list arguments;va_start(arguments,number);
 long a[6];for(long& value:a) value=va_arg(arguments,long);
 va_end(arguments);
 return __real_syscall(number,a[0],a[1],a[2],a[3],a[4],a[5]);
}
void* __wrap_shmat(int,const void*,int){fail();}
void* __wrap_dlopen(const char*,int){fail();}
void* __wrap_dlsym(void*,const char*){fail();}
void* __wrap_dlvsym(void*,const char*,const char*){fail();}
}
