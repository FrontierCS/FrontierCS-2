# syntax=docker/dockerfile:1.7
# The shared CPU runner layer (#757): a task's Agent or Verifier image made a Driver CPU machine
# for its [test_sandboxes.<name>] with access = "command" and resources = { cpu_millis,
# memory_bytes }. The build (scripts/build_task_images.py) builds it FROM the task's own image,
# so the machine is exactly that environment plus what the runner protocol needs, the same as a
# GPU runner image has:
#
#   - OpenSSH's sshd and the shared SSH bootstrap (gpu-tools), which the Driver starts as the
#     machine's entrypoint; the runner user `frontier` (non-root; uid 10001 when the image has
#     none) owns /workspace;
#   - /opt/frontier/runtime-python (the image's Python when it has none) and
#     /opt/frontier/task-runtime, which runs a command with the image's ENV (sshd clears it):
#     a step's argv is ["/opt/frontier/task-runtime", <program>, ...] on every runner image.
#
# Nothing else is added: an Agent image's runner holds no judge file (an untrusted sandbox), a
# Verifier image's runner holds the judge it already has (trusted only).
#
# Build arguments (set by the build, never by a task): RUNNER_BASE, the task image by its
# registry digest (a build without a push cannot make it); GPU_TOOLS,
# tasks/shared/bases.json["gpu-tools"], which the task's source identity binds. Context: the task's build context, whose
# .frontier-generated/runner/ holds this recipe and task-runtime.
ARG RUNNER_BASE
ARG GPU_TOOLS
FROM ${GPU_TOOLS} AS frontier_gpu_tools
FROM ${RUNNER_BASE}

USER root
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends openssh-server \
    && rm -rf /var/lib/apt/lists/* \
    && if getent passwd frontier >/dev/null; then test "$(id -u frontier)" != 0; \
       else useradd --create-home --uid 10001 --user-group --shell /bin/sh frontier; fi \
    && usermod --password x frontier \
    && install -d -o frontier -g "$(id -g frontier)" /workspace \
    && install -d -m 0755 /opt/frontier/bin
COPY --from=frontier_gpu_tools /opt/frontier/bin/frontier-gpu-sshd-bootstrap /opt/frontier/bin/frontier-gpu-sshd-bootstrap
COPY .frontier-generated/runner/task-runtime /opt/frontier/task-runtime
RUN chown root:root /opt/frontier/bin/frontier-gpu-sshd-bootstrap /opt/frontier/task-runtime \
    && chmod 0555 /opt/frontier/bin/frontier-gpu-sshd-bootstrap /opt/frontier/task-runtime \
    && if [ ! -e /opt/frontier/runtime-python ]; then ln -s "$(command -v python3)" /opt/frontier/runtime-python; fi \
    && /opt/frontier/runtime-python /opt/frontier/task-runtime --snapshot
WORKDIR /workspace
