#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <inttypes.h>

void *bf8_create(uint64_t *, uint32_t);
void *bf16_create(uint64_t *, uint32_t);
void bf8_query(void *, const uint64_t *, uint8_t *, size_t);
void bf16_query(void *, const uint64_t *, uint8_t *, size_t);
void bf8_destroy(void *);
void bf16_destroy(void *);
size_t bf_live_bytes(void);

static uint64_t permute(uint64_t x) {
  x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
  return x ^ (x >> 31);
}
/* Each family is injective over every index used; misses cannot be insertions. */
static uint64_t key(uint64_t i, uint64_t seed, int family) {
  if (family == 0) return permute(i + seed);
  if (family == 1) return i * UINT64_C(65537) + seed;
  return (i << 32) ^ seed;
}
static double ns(void) {
  struct timespec t;
  if (clock_gettime(CLOCK_MONOTONIC, &t)) exit(9);
  return (double)t.tv_sec * 1e9 + (double)t.tv_nsec;
}
static void *alloc(size_t n) { void *p = malloc(n); if (!p) exit(8); return p; }

int main(int argc, char **argv) {
  if (argc != 6) return 2;
  int bits = atoi(argv[1]), family = atoi(argv[3]);
  uint32_t n = (uint32_t)strtoul(argv[2], NULL, 10);
  uint64_t seed = strtoull(argv[4], NULL, 10);
  size_t q = strtoul(argv[5], NULL, 10);
  if ((bits != 8 && bits != 16) || family < 0 || family > 2 || n < 2 || n > 1000000 || q < 4 || q % 4 || q > 2097152) return 2;
  void *(*create)(uint64_t *, uint32_t) = bits == 8 ? bf8_create : bf16_create;
  void (*query)(void *, const uint64_t *, uint8_t *, size_t) = bits == 8 ? bf8_query : bf16_query;
  void (*destroy)(void *) = bits == 8 ? bf8_destroy : bf16_destroy;
  uint64_t *keys = alloc(n * sizeof(*keys));
  uint64_t *input = alloc(n * sizeof(*input));
  uint64_t *queries = alloc(q * sizeof(*queries));
  uint8_t *answers = alloc((n > q ? n : q) * sizeof(*answers));
  uint8_t *is_hit = alloc(q * sizeof(*is_hit));
  for (uint32_t i = 0; i < n; i++) keys[i] = key(i, seed, family);
  /* Shuffle insertion order independently of key family. */
  for (uint32_t i = n-1; i; i--) {
    size_t j = permute(seed ^ i) % (i+1);
    uint64_t t = keys[i]; keys[i] = keys[j]; keys[j] = t;
  }
  memcpy(input, keys, n * sizeof(*keys));
  double start = ns();
  void *f = create(input, n);
  double build_ns = ns() - start;
  if (!f) { puts("{\"construction\":false}"); return 0; }
  size_t resident = bf_live_bytes();
  query(f, keys, answers, n);
  size_t fn = 0;
  for (uint32_t i = 0; i < n; i++) fn += answers[i] != 1;
  /* Full independent absent stream for the FPR gate, outside timed lookup. */
  for (size_t i = 0; i < q; i++) queries[i] = key((uint64_t)n + i, seed, family);
  query(f, queries, answers, q);
  size_t fp = 0;
  for (size_t i = 0; i < q; i++) fp += answers[i] != 0;
  double lookup[4]; size_t mixed_fn = 0, mixed_fp[4] = {0,0,0,0};
  for (int mix = 0; mix < 3; mix++) {
    for (size_t i = 0; i < q; i++) {
      int hit = mix == 2 || (mix == 1 && (i & 1));
      uint64_t j = permute(seed + i);
      queries[i] = hit ? keys[j % n] : key((uint64_t)n + q + i, seed, family);
    }
    query(f, queries, answers, q); /* warm cache and branch predictor */
    start = ns();
    query(f, queries, answers, q);
    lookup[mix] = (ns() - start) / q;
    for (size_t i = 0; i < q; i++) {
      int hit = mix == 2 || (mix == 1 && (i & 1));
      if (hit && answers[i] != 1) mixed_fn++;
      if (!hit && answers[i] != 0) mixed_fp[mix]++;
    }
  }
  /* Added paper condition: exactly 25% hits, then shuffle keys AND truth labels.
     Preserve the three original query generators and their ordering above.
     Every miss index is >= n; key() is injective throughout the used range. */
  for (size_t i = 0; i < q; i++) {
    is_hit[i] = i < q / 4;
    queries[i] = is_hit[i] ? keys[permute(seed + i) % n]
                         : key((uint64_t)n + q + i, seed, family);
  }
  for (size_t i = q - 1; i; i--) {
    size_t j = permute(seed ^ UINT64_C(0xd1b54a32d192ed03) ^ i) % (i + 1);
    uint64_t k = queries[i]; queries[i] = queries[j]; queries[j] = k;
    uint8_t h = is_hit[i]; is_hit[i] = is_hit[j]; is_hit[j] = h;
  }
  query(f, queries, answers, q);
  start = ns();
  query(f, queries, answers, q);
  lookup[3] = (ns() - start) / q;
  for (size_t i = 0; i < q; i++) {
    if (is_hit[i] && answers[i] != 1) mixed_fn++;
    if (!is_hit[i] && answers[i] != 0) mixed_fp[3]++;
  }
  /* Precision stream for effective epsilon: eight disjoint q-sized blocks.
     Reuse the original untimed block once; the seven new blocks avoid every
     original query range. Do not pretend repeated executions add query trials.
     This draft's added cost still requires full reference runtime calibration. */
  size_t effective_fp = fp;
  for (size_t block = 1; block < 8; block++) {
    for (size_t i = 0; i < q; i++)
      queries[i] = key((uint64_t)n + (block + 3) * q + i, seed, family);
    query(f, queries, answers, q);
    for (size_t i = 0; i < q; i++) effective_fp += answers[i] != 0;
  }
  /* Caller input may be destroyed after construction; check actual key list again. */
  memset(input, 0, n * sizeof(*input));
  query(f, keys, answers, n);
  for (uint32_t i = 0; i < n; i++) fn += answers[i] != 1;
  /* Two simultaneous instances must not share mutable state. */
  uint64_t small[3] = {key(n+3*q+1, seed, family), key(n+3*q+2, seed, family), key(n+3*q+3, seed, family)};
  uint64_t small_input[3]; memcpy(small_input, small, sizeof(small));
  void *second = create(small_input, 3);
  if (!second) { destroy(f); puts("{\"construction\":false}"); return 0; }
  query(second, small, answers, 3);
  for (int i = 0; i < 3; i++) fn += answers[i] != 1;
  query(f, keys, answers, n);
  for (uint32_t i = 0; i < n; i++) fn += answers[i] != 1;
  destroy(second);
  destroy(f);
  printf("{\"construction\":true,\"false_negatives\":%zu,\"mixed_false_negatives\":%zu,\"false_positives\":%zu,\"mixed_false_positives\":[%zu,%zu,%zu,%zu],\"mixed_absent_queries\":[%zu,%zu,0,%zu],\"shuffled_hit_queries\":%zu,\"absent_queries\":%zu,\"effective_false_positives\":%zu,\"effective_absent_queries\":%zu,\"resident_bytes\":%zu,\"build_ns\":%.0f,\"lookup_ns\":[%.9f,%.9f,%.9f,%.9f],\"leaked_bytes\":%zu}\n",
         fn, mixed_fn, fp, mixed_fp[0], mixed_fp[1], mixed_fp[2], mixed_fp[3], q, q/2, 3*q/4, q/4, q, effective_fp, 8*q, resident, build_ns, lookup[0], lookup[1], lookup[2], lookup[3], bf_live_bytes());
  free(keys); free(input); free(queries); free(answers); free(is_hit);
  return 0;
}
