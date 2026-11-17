#!/bin/sh
# Installs the packages it is given, without their recommends, from the sources apt has, and
# leaves no package list behind.
#
#     apt_install.sh PACKAGE...
#
# A PACKAGE is as `apt-get install` takes it (`g++`, `python3-numpy=1:1.26.4+ds-6ubuntu1`). An
# archive snapshot fails a share of requests with server errors, at times more than apt's own
# tries of each file ride out, which leaves packages missing from a build that misses Docker's
# cache; so the update and the download each run again, up to 30 times, 5 seconds apart, keeping
# what earlier runs fetched; then the packages install without downloading.
set -eu
retry() { for attempt in $(seq 30); do "$@" && return; sleep 5; done; return 1; }
retry apt-get update --error-on=any
retry apt-get install --yes --no-install-recommends --download-only "$@"
apt-get install --yes --no-install-recommends --no-download "$@"
rm --recursive --force /var/lib/apt/lists/*
