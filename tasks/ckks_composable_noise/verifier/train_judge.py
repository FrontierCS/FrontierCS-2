"""Rewrites a copy of the judge's tree (the argument) into the train judge's: the frozen
calibration keeps only the cells of the development feedback's workloads, and the three
places that name the final's cells follow. Nothing is measured again, and no value changes.

    python3 train_judge.py <judge directory>
"""

import hashlib
import json
import sys
from pathlib import Path

METRICS = ("avg_log2_prec_l2_abs_discrepancy_bits", "std_log2_prec_l2_abs_discrepancy_bits")


def replaced(path, old, new):
    """Replaces the one `old` in the text of `path` by `new`, and fails unless it is one."""
    text = path.read_text()
    if text.count(old) != 1:
        sys.exit(f"{path}: expected one {old!r}, found {text.count(old)}: the judge changed")
    path.write_text(text.replace(old, new))


def main():
    judge = Path(sys.argv[1])
    release_path = judge / "judge" / "release.json"
    calibration_path = judge / "judge" / "formal_calibration.json"
    release = json.loads(release_path.read_text())
    declared = release["formal_calibration"]["sha256"]
    raw = calibration_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != declared:
        sys.exit(f"{calibration_path}: digest differs from release.json's: the judge changed")
    calibration = json.loads(raw)

    feedback = release["development_feedback"]["workloads"]
    cells = [w + "/" + m for w in feedback for m in METRICS]
    frozen = calibration["frozen"]
    train = {
        "schema": calibration["schema"],
        "status": calibration["status"],
        "measurement_binding": calibration["measurement_binding"],
        "development_workloads": feedback,
        "frozen": {
            "K": frozen["K"],
            "alpha": frozen["alpha"],
            "batch_invocations": frozen["batch_invocations"],
            "bands": {cell: frozen["bands"][cell] for cell in cells},
            "cells": cells,
            "q": frozen["q"],
            "rank": frozen["rank"],
            "scales": {cell: frozen["scales"][cell] for cell in cells},
        },
    }
    body = (json.dumps(train, indent=2) + "\n").encode()
    calibration_path.write_bytes(body)

    replaced(release_path, declared, hashlib.sha256(body).hexdigest())
    # The cells the file must hold are the feedback's, and its development rows may be all of
    # them, where the final's are a strict superset.
    replaced(judge / "evaluator.py", "final_workloads=FINAL_SUITE,", "final_workloads=tuple(DEV_FEEDBACK),")
    replaced(judge / "formal_scoring.py", "not set(development)<set(final_workloads)", "not set(development)<=set(final_workloads)")


main()
