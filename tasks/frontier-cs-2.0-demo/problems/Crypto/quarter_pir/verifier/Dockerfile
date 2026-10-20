# Preserve the authenticated runtime while replacing reviewed task files.
FROM docker.io/joyemang33/fcs2-task-verifier-quarter_pir@sha256:9d6600f6658a099842e16e105eab1f5a78c61594cfad572d8df8db4337562f77 AS repaired
USER 0:0
# The full-compute-scope judge replaces the predecessor's task directory.
RUN rm -rf /opt/frontier/task
COPY judge /opt/frontier/task
RUN chmod -R a-w,a+rX /opt/frontier/task

# Flatten to exclude historical task files from lower image layers.
FROM scratch
COPY --from=repaired / /
ENV PATH="/usr/local/go/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/sbin:/usr/local/go/bin:/usr/bin:/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/sbin"
ENV GOLANG_VERSION="1.19.13"
ENV GOPATH="/go"
ENV GOMODCACHE="/opt/frontier/go-mod"
ENV HOME="/frontier"
ENV PYTHONDONTWRITEBYTECODE="1"
ENV PYTHONUNBUFFERED="1"
LABEL dev.frontier.generator="frontiersmith2"
LABEL dev.frontier.problem-id="quarter_pir"
LABEL org.opencontainers.image.title="FrontierCS2 verifier quarter_pir"
USER 65532:65532
WORKDIR /frontier
CMD ["bash"]
