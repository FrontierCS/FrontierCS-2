# Builds bench/bench the way the judge does (same compiler, same flags).
CC ?= cc
CFLAGS ?= -O2 -std=c11 -Wall -Wextra

bench/bench: bench/main.c src/primes.c src/primes.h
	$(CC) $(CFLAGS) -Isrc -o $@ bench/main.c src/primes.c

clean:
	rm -f bench/bench
.PHONY: clean
