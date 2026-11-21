ARG AGENT_BASE
FROM ${AGENT_BASE}

ARG DEBIAN_FRONTEND=noninteractive
# The task's packages, from the same Ubuntu 24.04 snapshot. The agent runs as uid 1000, the base
# image's `ubuntu`, which owns /workspace.
RUN --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    sh /context/fetch/apt_install.sh binutils build-essential ca-certificates dpkg git make perl python3 time \
    && install --directory --owner=1000 --group=1000 /workspace

LABEL org.opencontainers.image.title="FrontierCS2 task cute_crest" \
      dev.frontier.problem-id="cute_crest"

# The solver-visible tree, and nothing else: the evaluator, the graded programs, the reference
# strategy and the beat-reference specification are in the verifier's image only. The
# repository's submission tools replace the base's.
COPY --from=task-tools /agent/bin/ /lib/edit_scope.py /frontier/task/bin/
COPY generated/agent/submission-policy.json /frontier/task/bin/submission-policy.json
COPY --from=readme /README.md /frontier/task/README.md
COPY repo /frontier/task/repo

# The development program, GNU grep 2.2 as distributed with CREST (GPL; its COPYING file stays
# beside the source), as the source fetches it: the CREST
# distribution at its pinned commit, checked by SHA-256, keeping benchmarks/grep-2.2 only.
ARG CREST_URL=https://github.com/jburnim/crest/archive/3e5ececd3793c88f18af47853236af34ffde4569.tar.gz
ARG CREST_SHA256=3ccbe608a23f4ded6937ab2a1886298af0f0415539d230bbed70f1df26ce1dfc
ARG GREP_PREFIX=crest-3e5ececd3793c88f18af47853236af34ffde4569/benchmarks/grep-2.2/
RUN python3 -c 'import hashlib, io, sys, tarfile, urllib.request; from pathlib import Path; url, digest, prefix, out = sys.argv[1:5]; body = urllib.request.urlopen(url, timeout=300).read(); assert hashlib.sha256(body).hexdigest() == digest, "source archive hash mismatch"; tar = tarfile.open(fileobj=io.BytesIO(body)); members = [m for m in tar.getmembers() if m.name.startswith(prefix) and len(m.name) > len(prefix)]; [setattr(m, "name", m.name[len(prefix):]) for m in members]; Path(out).mkdir(parents=True); tar.extractall(out, members=members, filter="data"); assert (Path(out) / "src/grep.c").is_file() and (Path(out) / "COPYING").is_file(), "unexpected source layout"' \
        "$CREST_URL" "$CREST_SHA256" "$GREP_PREFIX" /frontier/task/benchmark-src/grep-2.2 \
    && chmod --recursive a+rX,a-w /frontier/task/benchmark-src

RUN --mount=type=bind,from=task-tools,source=/agent-permissions.sh,target=/agent-permissions.sh \
    sh /agent-permissions.sh '*.sh'

USER 1000:1000
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /workspace
