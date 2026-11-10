/* src/ is the editable directory (task.toml submission.allowed_paths): change this file or
 * add more .c files beside it; the judge compiles every .c file in src/. Starter: trial
 * division by odd numbers. Correct, deliberately slow. */
#include "primes.h"

static int is_prime(uint32_t x) {
    if (x < 2) return 0;
    if (x % 2 == 0) return x == 2;
    for (uint32_t d = 3; (uint64_t)d * d <= x; d += 2)
        if (x % d == 0) return 0;
    return 1;
}

uint64_t count_primes(uint32_t n) {
    uint64_t count = 0;
    for (uint32_t x = 2; x <= n && x != 0; x++)
        count += (uint64_t)is_prime(x);
    return count;
}
