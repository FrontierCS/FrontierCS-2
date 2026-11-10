# Builds bench/bench the way the judge does (same compiler, same flags, every src/*.c).
CC ?= cc
CFLAGS ?= -O2 -std=c11 -Wall -Wextra

bench/bench: bench/main.c $(wildcard src/*.c) include/primes.h
	$(CC) $(CFLAGS) -Iinclude -o $@ bench/main.c $(sort $(wildcard src/*.c))

clean:
	rm -f bench/bench
.PHONY: clean
