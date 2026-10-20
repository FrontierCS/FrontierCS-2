# Qualified immutable runtime; the public alias preserves the measured image digest.
FROM docker.io/joyemang33/fcs2-task-verifier-ckks_composable_noise@sha256:b6ddcc32dea200caf5a0886815c85cb0a2149023c0ebac9b2fa1012fc1f298bb
USER root
RUN chmod u+w /opt/frontier/task /opt/frontier/task/judge && rm -f /opt/frontier/task/NOTES.md /opt/frontier/task/SUMMARY.md /opt/frontier/task/test_evaluator.py /opt/frontier/task/test_workload_configurations.py /opt/frontier/task/pytest.ini /opt/frontier/task/judge/formal_calibration.json /opt/frontier/task/judge/valid_helper.patch /opt/frontier/task/judge/partial_rounding.patch
USER 65532:65532
COPY judge /opt/frontier/task
COPY verifier/fcs2_verifier.py /opt/frontier/fcs2_verifier.py
COPY verifier/ckks_evidence_transport.py /opt/frontier/ckks_evidence_transport.py
