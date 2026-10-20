#!/bin/sh
set -eu
cd "$(dirname "$0")"
git add -A -- my_cond_branch_predictor.cc
if [ -d solver_extensions ]; then git add -A -- solver_extensions; fi
# Reject edits anywhere else, including tracked build/configuration files.
python3 - <<'CHECK'
import subprocess
allowed=lambda p: p=="my_cond_branch_predictor.cc" or p.startswith("solver_extensions/")
generated={"cbp","solution.patch","cond_branch_predictor_interface.o","my_cond_branch_predictor.o","lib/libcbp.a"}
generated.update("lib/"+p+".o" for p in ("cbp","my_value_predictor","parameters","uarchsim","cache","bp","resource_schedule","gzstream"))
others=subprocess.check_output(["git","ls-files","--others","--exclude-standard","-z"]).decode().split("\0")
if any(p and not allowed(p) and p not in generated for p in others):
    raise SystemExit("untracked files outside permitted source surface")
for mode in (["diff","--name-only"],["diff","--cached","--name-only"]):
    names=subprocess.check_output(["git",*mode,"-z"]).decode().split("\0")
    if any(p and not allowed(p) for p in names):
        raise SystemExit("changes outside permitted source surface")
CHECK
git diff --cached --binary -- my_cond_branch_predictor.cc solver_extensions > solution.patch
