#!/usr/bin/env python3
"""Deliberately slow baseline for the speed-add task."""

import sys


def add(a: int, b: int) -> int:
    result = 0
    for _ in range(a):
        result += 1
    for _ in range(b):
        result += 1
    return result


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: solution.py <a> <b>")
    a = int(sys.argv[1])
    b = int(sys.argv[2])
    if a < 0 or b < 0:
        raise SystemExit("arguments must be non-negative")
    print(add(a, b))


if __name__ == "__main__":
    main()
