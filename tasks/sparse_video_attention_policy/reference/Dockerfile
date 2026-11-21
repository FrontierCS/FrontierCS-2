FROM ubuntu:24.04@sha256:fe81560990fae86dc4743202b2cb4d3df98d0bcf949173cecef15ce1127ef852
RUN install --directory --owner=65532 --group=65532 /frontier \
    && install --directory --mode=0555 /opt/frontier/task
COPY --chmod=0444 judge/reference.patch /opt/frontier/task/reference.patch
USER 65532:65532
ENV HOME=/frontier
WORKDIR /frontier
