# The judge's scheduler, a trusted companion of every parent: Poisson timing, the BFCL state and
# the calls to the SWE containers run here, and the Verifier reaches it over its own session.
ARG VERIFIER_BASE
FROM ${VERIFIER_BASE}

USER root
ARG DEBIAN_FRONTEND=noninteractive
# The scheduler's packages, each file pinned by SHA-256 (`requirements.txt` is `uv pip compile` of
# `requirements.in`, the source's list less `cryptography`, which signed what the scheduler's
# token now authenticates, `judge/serving_eval/sidecar_env.py`), in a venv of their own.
COPY requirements.txt /opt/frontier/requirements.txt
RUN --mount=type=bind,source=fetch/apt_install.sh,target=/context/fetch/apt_install.sh \
    sh /context/fetch/apt_install.sh ca-certificates python3 python3-venv \
    && python3 -m venv /opt/frontier/python \
    && /opt/frontier/python/bin/pip install --no-cache-dir --require-hashes --only-binary=:all: --no-deps --requirement /opt/frontier/requirements.txt

COPY judge/instance_manifest.py judge/instance_manifest.json judge/batch_coordinator.py judge/batch_run.py \
     judge/attestation.py judge/sidecar_catalog.py judge/scheduler_service.py judge/scheduler_relay.py \
     judge/scheduler_failure_contract.py judge/supervised_case_timing.py judge/terminal_contract.py \
     judge/terminal_ledger.py judge/containment_policy.py judge/instance_supervisor.py judge/scheduler_stream.py \
     judge/bounded_http.py judge/http_request_worker.py judge/dispatch_broker.py judge/episode_worker.py \
     /opt/frontier/judge/
COPY judge/serving_eval/__init__.py judge/serving_eval/swe_patch_disposition.py judge/serving_eval/swe_case_failure.py \
     judge/serving_eval/agent_runner.py judge/serving_eval/bfcl.py judge/serving_eval/settings.py \
     judge/serving_eval/swe_inputs.py judge/serving_eval/sidecar_env.py \
     /opt/frontier/judge/serving_eval/
COPY judge/serving_eval/bfcl_vendor/ /opt/frontier/judge/serving_eval/bfcl_vendor/
COPY judge/serving_eval/bfcl_data/BFCL_v4_memory.json judge/serving_eval/bfcl_data/NOTICE /opt/frontier/judge/serving_eval/bfcl_data/
COPY judge/serving_eval/bfcl_data/memory_snapshots/ /opt/frontier/judge/serving_eval/bfcl_data/memory_snapshots/
COPY judge/serving_eval/bfcl_data/multi_turn_func_doc/ /opt/frontier/judge/serving_eval/bfcl_data/multi_turn_func_doc/
COPY judge/serving_eval/swe_data/runtime_inputs.json judge/serving_eval/swe_data/NOTICE /opt/frontier/judge/serving_eval/swe_data/
# The BFCL answer key never enters the GPU-side images.
RUN test ! -e /opt/frontier/judge/serving_eval/bfcl_data/possible_answer \
    && find /opt/frontier/judge -type f -exec chmod 0444 {} + \
    && find /opt/frontier/judge -type d -exec chmod 0555 {} + \
    && chmod -R a-w /opt/frontier/python

LABEL org.opencontainers.image.title="FrontierCS2 verifier scheduler vllm_llm_serving_optimization" \
      dev.frontier.problem-id="vllm_llm_serving_optimization"

# A Pumpkin sandbox's /tmp may be read-only, so the scheduler's home and its temporary files are
# below /var/tmp, where the source has them below /tmp.
ENV PYTHONPATH=/opt/frontier/judge PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HOME=/var/tmp TMPDIR=/var/tmp
USER 65532:65532
WORKDIR /var/tmp
