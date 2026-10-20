/* Trusted API bridge. Compile separately from harness.c, without LTO. */
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <limits.h>

typedef union { struct { size_t size; } info; long double alignment; void *pointer; } allocation;
static size_t live_bytes;
static void *tracked_malloc(size_t n) {
  if (n > SIZE_MAX - sizeof(allocation)) return NULL;
  allocation *p = malloc(sizeof(allocation) + n);
  if (!p) return NULL;
  p->info.size = n;
  live_bytes += n;
  return p + 1;
}
static void tracked_free(void *v) {
  if (!v) return;
  allocation *p = (allocation *)v - 1;
  live_bytes -= p->info.size;
  free(p);
}
static void *tracked_calloc(size_t n, size_t s) {
  if (s && n > SIZE_MAX / s) return NULL;
  void *p = tracked_malloc(n*s);
  if (p) memset(p, 0, n*s);
  return p;
}
static void *tracked_realloc(void *v, size_t n) {
  if (!v) return tracked_malloc(n);
  if (!n) { tracked_free(v); return NULL; }
  allocation *p = (allocation *)v - 1;
  void *q = tracked_malloc(n);
  if (!q) return NULL;
  memcpy(q, v, p->info.size < n ? p->info.size : n);
  tracked_free(v);
  return q;
}
#define malloc tracked_malloc
#define calloc tracked_calloc
#define realloc tracked_realloc
#define free tracked_free
#include "binaryfusefilter.h"
#undef malloc
#undef calloc
#undef realloc
#undef free

#define BRIDGE(B) \
void *bf##B##_create(uint64_t *keys, uint32_t n) { \
  binary_fuse##B##_t *f = tracked_calloc(1, sizeof(*f)); \
  if (!f) return NULL; \
  if (!binary_fuse##B##_allocate(n, f) || !binary_fuse##B##_populate(keys, n, f)) { \
    binary_fuse##B##_free(f); tracked_free(f); return NULL; \
  } \
  return f; \
} \
void bf##B##_query(void *v, const uint64_t *keys, uint8_t *out, size_t n) { \
  const binary_fuse##B##_t *f = v; \
  for (size_t i = 0; i < n; ++i) out[i] = binary_fuse##B##_contain(keys[i], f) ? 1 : 0; \
} \
void bf##B##_destroy(void *v) { \
  binary_fuse##B##_free(v); tracked_free(v); \
}
BRIDGE(8)
BRIDGE(16)
size_t bf_live_bytes(void) { return live_bytes; }
