# Preserve the authenticated runtime while replacing reviewed task files.
FROM docker.io/joyemang33/fcs2-task-quarter_pir-agent@sha256:b5c518b0defb0db123e4dd2060790f65cefc1381947629ba6a1714d209804a9b AS repaired
USER 0:0
# The full-compute-scope source tree replaces the predecessor's server-only tree.
RUN rm -rf /frontier/task/repo
COPY agent-base/frontier/task/repo /frontier/task/repo
COPY --chmod=0444 agent-base/frontier/task/README.md /frontier/task/README.md
COPY --chmod=0444 agent-base/frontier/task/bin/submission-policy.json /frontier/task/bin/submission-policy.json
RUN chmod -R a-w,a+rX /frontier/task/repo

# Flatten to exclude historical task files from lower image layers.
FROM scratch
COPY --from=repaired / /
ENV PATH="/usr/local/go/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/sbin:/usr/local/go/bin:/usr/bin:/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/sbin"
ENV GOLANG_VERSION="1.19.13"
ENV GOPATH="/go"
ENV GOMODCACHE="/opt/frontier/go-mod"
ENV HOME="/workspace"
ENV TMPDIR="/workspace"
ENV PYTHONDONTWRITEBYTECODE="1"
ENV PYTHONUNBUFFERED="1"
LABEL dev.frontier.generator="frontiersmith2"
LABEL dev.frontier.generator-run="r_top01_uarter_pir"
LABEL dev.frontier.problem-id="quarter_pir"
LABEL org.opencontainers.image.title="FrontierCS2 task quarter_pir"
USER 10001:10001
WORKDIR /workspace
CMD ["bash"]
