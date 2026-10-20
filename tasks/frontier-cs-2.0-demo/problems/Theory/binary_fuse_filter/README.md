# Binary Fuse Filter native task

Manual skill-authored B-group task, source pin
`FastFilter/xor_singleheader@3e10fb185a0c51e5866518d13b403e07db63d8d8`,
Apache-2.0. The authoritative runtime contract is `policy.json`.

This checkout is the **certified non-inferiority successor** (`certified-non-inferiority-r8`)
that resolves an unattainable predecessor rule. The predecessor required the two accuracy metrics
to improve on the exact reference by strictly separated certified intervals; at
16-bit fingerprints that requirement, combined with strictly fewer resident bytes,
lies below the information-theoretic minimum for a key-agnostic filter on at least
24 of the 144 final workloads, so strict PASS was unattainable.

The successor keeps every correctness gate, every workload, the descriptive score
and the strict per-workload wins on resident bytes, construction time and lookup
time. Effective FPR and theoretical space overhead stay in the five-way
conjunction as **certified non-inferiority guards**: a workload clears when the
candidate's certified lower interval bound does not exceed a preregistered
anchor. The anchors are reference-only: for effective FPR the larger of the
fingerprint's nominal rate `2^-b` and the worst retained reference point; for
space overhead the worst retained reference point. They were derived from the
five retained reference rounds and the fingerprint specification before any
successor solver result and are recorded in `judge/reference/statistical-calibration.json`
(bound to the exact measurement sources by `judge/statistical_calibration.json`).
Under the interval's IID assumption the reference and any filter at or below an
anchor are rejected with probability at most the per-construction alpha; the
guard certifies an excess only above roughly 1.6-1.7x the nominal rate at 16-bit
and about 1.04x at 8-bit with 8,388,608 absent queries. See
`judge/accuracy_metrics.py::non_inferior` and `judge/beat_spec.json`.

The r7 documentation-calibration migration and its reuse fence for the r6
statistics are retired: `judge/reference/history/` retains the r6 profile and
runtime receipts, the r7 receipt and migration record and the frozen r6
measurement sources byte-for-byte. Production evaluation still fails closed
until `judge/reference/runtime-qualification.json` binds the current measurement
sources with an exact immutable-image qualification of the private reference.

Public material is solely `agent-base/frontier/task/`. `judge/reference.patch`
restores both headers byte-for-byte to upstream and is never copied into the
Agent image. `judge/reference/` contains the exact private baseline headers.
The unchanged original `make query -> benchmarks/query.c` path runs on the
patched source, alongside the broader workload in `judge/harness.c`.

This task covers segmented 3-wise construction and lookup, not the paper's
4-wise variant, all competing filter families, or universal probabilistic bounds.
Source reproduction, local workload calibration, image runtime qualification,
formal reference acceptance and genuine model evaluation are distinct evidence
levels.
