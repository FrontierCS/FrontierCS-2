/* force-included before every instrumented translation unit: neutralise
   glibc attributes the old CIL front end re-emits out of declaration order */
#include <sys/cdefs.h>
#undef __attr_dealloc
#define __attr_dealloc(dealloc, argno)
#undef __attr_dealloc_free
#define __attr_dealloc_free
