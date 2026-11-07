/* Not editable. Prints count_primes(n) for the n given as the only argument. The judge times
 * this whole process. */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include "primes.h"

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: bench N\n");
        return 2;
    }
    unsigned long n = strtoul(argv[1], NULL, 10);
    printf("%" PRIu64 "\n", count_primes((uint32_t)n));
    return 0;
}
