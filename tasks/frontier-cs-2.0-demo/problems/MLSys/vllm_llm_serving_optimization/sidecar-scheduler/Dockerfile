# syntax=docker/dockerfile:1.7
# Trusted scheduler sidecar: the Driver starts it beside every Verifier GPU
# parent from this exact image and hands it the operation's attestation key.
FROM ubuntu@sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517
USER root
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates python3 python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 65532 frontier && useradd --uid 65532 --gid 65532 --no-create-home frontier
COPY sidecar-scheduler/requirements.txt /opt/frontier/requirements.txt
RUN python3 -m venv /opt/frontier/python \
    && /opt/frontier/python/bin/pip install --no-cache-dir -r /opt/frontier/requirements.txt
COPY judge/instance_manifest.py /opt/frontier/judge/instance_manifest.py
COPY judge/instance_manifest.json /opt/frontier/judge/instance_manifest.json
COPY judge/batch_coordinator.py /opt/frontier/judge/batch_coordinator.py
COPY judge/batch_run.py /opt/frontier/judge/batch_run.py
COPY judge/attestation.py /opt/frontier/judge/attestation.py
COPY judge/sidecar_catalog.py /opt/frontier/judge/sidecar_catalog.py
COPY judge/scheduler_service.py /opt/frontier/judge/scheduler_service.py
COPY judge/scheduler_failure_contract.py /opt/frontier/judge/scheduler_failure_contract.py
COPY judge/supervised_case_timing.py /opt/frontier/judge/supervised_case_timing.py
COPY judge/terminal_contract.py /opt/frontier/judge/terminal_contract.py
COPY judge/terminal_ledger.py /opt/frontier/judge/terminal_ledger.py
COPY judge/containment_policy.py /opt/frontier/judge/containment_policy.py
COPY judge/instance_supervisor.py /opt/frontier/judge/instance_supervisor.py
COPY judge/scheduler_stream.py /opt/frontier/judge/scheduler_stream.py
COPY judge/bounded_http.py /opt/frontier/judge/bounded_http.py
COPY judge/http_request_worker.py /opt/frontier/judge/http_request_worker.py
COPY judge/dispatch_broker.py /opt/frontier/judge/dispatch_broker.py
COPY judge/episode_worker.py /opt/frontier/judge/episode_worker.py
COPY judge/serving_eval/__init__.py /opt/frontier/judge/serving_eval/__init__.py
COPY judge/serving_eval/swe_patch_disposition.py /opt/frontier/judge/serving_eval/swe_patch_disposition.py
COPY judge/serving_eval/swe_case_failure.py /opt/frontier/judge/serving_eval/swe_case_failure.py
COPY judge/serving_eval/agent_runner.py /opt/frontier/judge/serving_eval/agent_runner.py
COPY judge/serving_eval/bfcl.py /opt/frontier/judge/serving_eval/bfcl.py
COPY judge/serving_eval/settings.py /opt/frontier/judge/serving_eval/settings.py
COPY judge/serving_eval/swe_inputs.py /opt/frontier/judge/serving_eval/swe_inputs.py
COPY judge/serving_eval/sidecar_env.py /opt/frontier/judge/serving_eval/sidecar_env.py
COPY judge/serving_eval/tool_auth.py /opt/frontier/judge/serving_eval/tool_auth.py
COPY judge/serving_eval/bfcl_vendor/ /opt/frontier/judge/serving_eval/bfcl_vendor/
COPY judge/serving_eval/bfcl_data/BFCL_v4_memory.json /opt/frontier/judge/serving_eval/bfcl_data/BFCL_v4_memory.json
COPY judge/serving_eval/bfcl_data/NOTICE /opt/frontier/judge/serving_eval/bfcl_data/NOTICE
COPY judge/serving_eval/bfcl_data/memory_snapshots/ /opt/frontier/judge/serving_eval/bfcl_data/memory_snapshots/
COPY judge/serving_eval/bfcl_data/multi_turn_func_doc/ /opt/frontier/judge/serving_eval/bfcl_data/multi_turn_func_doc/
COPY judge/serving_eval/swe_data/runtime_inputs.json /opt/frontier/judge/serving_eval/swe_data/runtime_inputs.json
COPY judge/serving_eval/swe_data/NOTICE /opt/frontier/judge/serving_eval/swe_data/NOTICE
# The BFCL answer key never enters the GPU-side images.
RUN test ! -e /opt/frontier/judge/serving_eval/bfcl_data/possible_answer \
    && find /opt/frontier/judge -type f -exec chmod 0444 {} + \
    && find /opt/frontier/judge -type d -exec chmod 0555 {} + \
    && chmod -R a-w /opt/frontier/python
ENV PYTHONPATH=/opt/frontier/judge PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HOME=/tmp TMPDIR=/tmp
USER 65532:65532
WORKDIR /tmp
ENTRYPOINT []
CMD ["/opt/frontier/python/bin/python", "/opt/frontier/judge/scheduler_service.py", "--host", "0.0.0.0", "--port", "8002", "--max-request-bytes", "8388608", "--max-response-bytes", "8388608"]
