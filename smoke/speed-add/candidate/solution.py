#!/usr/bin/env python3
"""Constant-time reference used only by local verifier tests."""

import sys

def add(a: int, b: int) -> int:
    return a + b


if len(sys.argv) != 3:
    raise SystemExit("usage: solution.py <a> <b>")
print(add(int(sys.argv[1]), int(sys.argv[2])))
