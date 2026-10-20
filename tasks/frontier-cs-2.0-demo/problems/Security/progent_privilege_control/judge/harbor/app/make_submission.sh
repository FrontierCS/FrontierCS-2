#!/bin/sh
set -eu
cd /app
git add -A -- . ':!solution.patch'
git diff --cached --binary > solution.patch
