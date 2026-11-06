/* Public interface. Not editable: the judge links the bench program against it. */
#ifndef TEMPLATE_PRIMES_H
#define TEMPLATE_PRIMES_H
#include <stdint.h>

/* Number of primes p with 2 <= p <= n. Must be exact for every n in [0, 100000000]. */
uint64_t count_primes(uint32_t n);

#endif
