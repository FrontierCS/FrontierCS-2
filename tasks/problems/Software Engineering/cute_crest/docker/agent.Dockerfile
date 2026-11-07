# Agent image of cute_crest: the shared Agent base plus this task's packages, statement,
# starter repo, development-program source and generated submission policy. Composed with the
# Candidate overlay at Run time, so it must not depend on its own ENTRYPOINT, CMD or named user.
# Build context: this task's root, with .frontier-generated/ added by the build.
ARG AGENT_BASE
FROM ${AGENT_BASE}

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends binutils build-essential ca-certificates dpkg git make perl python3 time \
    && rm -rf /var/lib/apt/lists/*

LABEL org.opencontainers.image.title="FrontierCS2 task cute_crest" \
      dev.frontier.problem-id="cute_crest"

# The solver-visible tree, and nothing else: the evaluator, the graded programs, the reference
# strategy and the beat-reference specification are in the Verifier image only.
COPY .frontier-generated/agent/README.md /frontier/task/README.md
COPY repo /frontier/task/repo
COPY .frontier-generated/agent/submission-policy.json /frontier/task/bin/submission-policy.json

# The development program is fetched here, by pinned URL and SHA-256, rather than committed:
# GNU grep 2.2 as distributed with CREST (GPL; its COPYING file stays beside the source): the
# CREST distribution at its pinned commit, checked by SHA-256, keeping benchmarks/grep-2.2 only.
ARG CREST_URL=https://github.com/jburnim/crest/archive/3e5ececd3793c88f18af47853236af34ffde4569.tar.gz
ARG CREST_SHA256=3ccbe608a23f4ded6937ab2a1886298af0f0415539d230bbed70f1df26ce1dfc
ARG GREP_PREFIX=crest-3e5ececd3793c88f18af47853236af34ffde4569/benchmarks/grep-2.2/
RUN python3 -c 'import hashlib, io, sys, tarfile, urllib.request; from pathlib import Path; url, digest, prefix, out = sys.argv[1:5]; body = urllib.request.urlopen(url, timeout=300).read(); assert hashlib.sha256(body).hexdigest() == digest, "source archive hash mismatch"; tar = tarfile.open(fileobj=io.BytesIO(body)); members = [m for m in tar.getmembers() if m.name.startswith(prefix) and len(m.name) > len(prefix)]; [setattr(m, "name", m.name[len(prefix):]) for m in members]; Path(out).mkdir(parents=True); tar.extractall(out, members=members, filter="data"); assert (Path(out) / "src/grep.c").is_file() and (Path(out) / "COPYING").is_file(), "unexpected source layout"' \
        "$CREST_URL" "$CREST_SHA256" "$GREP_PREFIX" /frontier/task/benchmark-src/grep-2.2 \
    && chmod -R a+rX,a-w /frontier/task/benchmark-src

RUN find /frontier/task -type d -exec chmod 0555 {} + \
    && find /frontier/task/bin -type f -exec chmod 0444 {} + \
    && chmod 0555 /frontier/task/bin/*.sh \
    && find /frontier/task/repo -type f -exec chmod 0444 {} + \
    && find /frontier/task/repo -type f -name '*.sh' -exec chmod 0555 {} + \
    && chmod 0444 /frontier/task/README.md

USER 10001:10001
ENV HOME=/workspace TMPDIR=/workspace PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /workspace
