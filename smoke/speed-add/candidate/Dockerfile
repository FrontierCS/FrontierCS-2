# syntax=docker/dockerfile:1.7

FROM scratch

COPY --chmod=0555 start /opt/frontier/candidate/local-e2e-start
COPY --chmod=0555 run.sh /opt/frontier/candidate/optimized/run.sh
COPY --chmod=0444 solution.py /opt/frontier/candidate/optimized/solution.py
