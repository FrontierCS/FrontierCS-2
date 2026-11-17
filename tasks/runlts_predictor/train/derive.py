"""Derives the train judge's copy of the judge from the source's.

    derive.py JUDGE_DIR TABLE_SHA256 EVALUATOR_SHA256

Rewrites `reference_counts.json` and `evaluator.py` in JUDGE_DIR for a judge that holds the 24
development traces and not the 81 final ones, and fails unless the results have the given
SHA-256s.
"""

import hashlib
import json
import sys
from pathlib import Path

OLD_TABLE_SHA256 = "ff6c50e82ea4e78ee193e2a90809a90c6af60aaf0d09da3ab430de9be18ab4f1"
EVALUATOR_EDITS = [
    # The table is the train table below.
    ("PINNED_REFERENCE_SHA256='" + OLD_TABLE_SHA256 + "'", "PINNED_REFERENCE_SHA256='@TABLE@'"),
    # The train table has the development traces' rows alone.
    ("if len(rows)!=105 or len(by_id)!=105:raise pinned_failure('table coverage invalid')",
     "if len(rows)!=24 or len(by_id)!=24:raise pinned_failure('table coverage invalid')"),
    # Only the traces of the suite are checked for presence, size and checksum.
    ("        p=DATA/r['path']\n",
     "    for r in rows if final else [r for r in rows if r['development']]:\n        p=DATA/r['path']\n"),
]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    """Exits with `message` unless `condition` holds; an `if`, not an `assert`, which `-O` skips."""
    if not condition:
        sys.exit(f"derive.py: {message}")


def main():
    judge, want_table, want_evaluator = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    table_path = judge / "reference_counts.json"
    raw = table_path.read_bytes()
    require(sha256(raw) == OLD_TABLE_SHA256, "the source's table is not the one this script derives from")
    table = json.loads(raw)
    manifest = json.loads((judge / "workload_manifest.json").read_text())
    development = [row["id"] for row in manifest["workloads"] if row["development"]]
    require(len(development) == 24, f"the manifest has {len(development)} development traces, not 24")
    rows = {row["id"]: row for row in table["rows"]}
    # The identity is kept: it binds the reference and the whole manifest, which stays whole.
    train = {"identity": table["identity"], "rows": [rows[i] for i in development], "schema": table["schema"]}
    out = (json.dumps(train, indent=1, sort_keys=True) + "\n").encode()
    require(sha256(out) == want_table, f"the train table is {sha256(out)}, not {want_table}")
    table_path.write_bytes(out)

    evaluator_path = judge / "evaluator.py"
    text = evaluator_path.read_text()
    for old, new in EVALUATOR_EDITS:
        require(text.count(old) == 1, f"{old!r} occurs {text.count(old)} times in evaluator.py, not once")
        text = text.replace(old, new.replace("@TABLE@", want_table))
    require(sha256(text.encode()) == want_evaluator,
            f"the train evaluator is {sha256(text.encode())}, not {want_evaluator}")
    evaluator_path.write_text(text)


main()
