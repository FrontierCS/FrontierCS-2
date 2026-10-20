# Preserve qualified GPU execution; remove operator-only reports from visible filesystem.
FROM docker.io/joyemang33/fcs2-task-verifier-flow_opd-gpu@sha256:ee56b3c808f13ac1b350a768af07ea504d0772a32d08fcc6519177e9ee4ce558 AS release_runtime
USER root
RUN rm -f /opt/frontier/task/NOTES.md /opt/frontier/task/RUNTIME_QUALIFICATION.md /opt/frontier/task/STRUCTURAL_CHECK_REPORT.md /opt/frontier/task/SUMMARY.md /opt/frontier/task/result.json /opt/frontier/task/test_evaluator.py /opt/frontier/task/pytest.ini

RUN rm -f /opt/frontier/task/judge/asset_audit.json /opt/frontier/task/judge/corpus_audit.json /opt/frontier/task/judge/leak_inventory.json /opt/frontier/task/judge/phase_audit.json /opt/frontier/task/judge/source_inventory.json /opt/frontier/task/judge/structural_report.json /opt/frontier/task/judge/render_readme.py

# Publish the cleaned runtime without any inherited authoring layers.
FROM scratch
COPY --from=release_runtime / /

# Overlay the released ceiling-aware gate onto the cleaned scientific runtime.
COPY judge/judge/scoring.py /opt/frontier/task/judge/scoring.py
COPY judge/judge/runner.py /opt/frontier/task/judge/runner.py
COPY judge/evaluator.py /opt/frontier/task/evaluator.py
COPY verifier/fcs2_verifier.py /opt/frontier/fcs2_verifier.py
COPY verifier/fcs2_verifier_gpu.py /opt/frontier/fcs2_verifier_gpu.py
COPY judge/judge/qualification.json /opt/frontier/task/judge/qualification.json
COPY judge/beat_spec.json /opt/frontier/task/beat_spec.json
COPY judge/readme /opt/frontier/task/readme
ENV PATH="/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ENV LANG="C.UTF-8"
ENV GPG_KEY="A035C8C19219BA821ECEA86B64E628F8D684696D"
ENV PYTHON_VERSION="3.10.21"
ENV PYTHON_SHA256="a0da1e72132e950154eca0f6f47d5db828454700de20e5113667940d81e0db04"
ENV FRONTIER_MODEL_ROOT="/opt/frontier/models"
ENV HF_HUB_OFFLINE="1"
ENV TRANSFORMERS_OFFLINE="1"
ENV WANDB_MODE="disabled"
ENV PYTHONDONTWRITEBYTECODE="1"
ENV PYTHONUNBUFFERED="1"
ENV FRONTIER_VERIFIER_GPU_TYPE="H100"
ENV FRONTIER_VERIFIER_GPU_SECONDS="3000"
ENV FRONTIER_VERIFIER_GPU_COUNT="1"
LABEL dev.frontier.generator="frontiersmith2"
LABEL dev.frontier.problem-id="flow_opd"
LABEL org.opencontainers.image.title="FrontierCS2 verifier flow_opd"
USER root
WORKDIR /frontier
CMD ["python3"]
