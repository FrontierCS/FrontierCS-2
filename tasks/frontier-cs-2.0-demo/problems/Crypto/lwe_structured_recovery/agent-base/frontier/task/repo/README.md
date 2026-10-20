# Structured-LWE Public Witness Recovery

## Optimization priority

Your primary objective is **PASS on every required workload and every required
metric**, with all correctness and validity requirements satisfied. The continuous
score is a secondary optimization objective; a higher score cannot compensate for
any failed PASS condition. Do not replace a solution known to satisfy all tested
PASS conditions with one known to fail them merely to increase its score.
Exploratory trials may regress; if no solution passes yet, work on the unmet
conditions and retain a valid submission as a fallback.

Successful verifier execution or a positive score does not mean scientific PASS.
Use the reported per-metric pass/fail progress to assess the tested workloads.
Train feedback covers only its evaluated workloads; passing train does not prove
final PASS. Final independently evaluates the submitted artifact on its specified
suite. The scoring formula, validity rules, and PASS comparisons below are unchanged.

## Problem

Recover valid secret vectors for as many public structured-LWE instances as
you can.  For each instance, the public data define dimensions `n` and `m`, a
modulus `q`, a reproducible matrix `A`, a target vector `b`, public secret and
error distributions, and the exact predicates used for acceptance.  A valid
submission satisfies the public relation

```text
b = A s + e (mod q)
```

with a secret `s` and centered residual `e` that pass those predicates.  Any
valid `s` is accepted; you do not have to recover the particular witness used
when the instance was generated.

## Design space

Write solver code under `/workspace/repo` or elsewhere in `/workspace`. The scored artifact is only `/workspace/payload/solution.json`; it is submitted as a patch against the starter ledger, as described in `/frontier/task/README.md`. You may inspect or experiment with the copied checker locally; final evaluation always uses the immutable checker and catalog in the Verifier. Solver source code is not submitted.

## Workload

The production corpus has 200 public instances, split evenly across ten
structural families (20 per family):

| Family | Public structure |
| --- | --- |
| `DS_BIN`, `DS_TER` | dense uniform matrix; exact-weight binary or signed-ternary secret |
| `DS_SMALL` | dense uniform matrix; dense small-alphabet secret |
| `SA_Q` | sparse matrix over general nonzero residues; unrestricted mod-`q` secret |
| `SA_SMALL` | sparse small-alphabet matrix; unrestricted mod-`q` secret |
| `DA_BIN`, `DA_TER` | dense binary or ternary matrix; unrestricted mod-`q` secret |
| `MIX_Q_SPARSE` | sparse general-residue matrix and sparse small secret |
| `MIX_SMALL_SPARSE` | sparse small-alphabet matrix and sparse small secret |
| `MIX_DENSE_SMALL` | dense small-alphabet matrix and small or sparse small secret |

The public catalog deliberately omits difficulty labels, runtime bins,
calibration estimates, cryptanalysis paths, and reference solutions.  You
should treat the instance IDs as opaque names.  The intended workflow is
incremental: inspect the public algebraic structure, try any attack strategy
you like, validate every candidate with the public facade, and accumulate validated solutions while retaining all earlier ones. Train calls
may batch multiple discoveries; see `/frontier/task/README.md` for the enforced call budget.

### Reproducibility

Every public record contains the complete matrix description, a SHAKE-256 seed
and domain separator, inline `b`, all predicates, and an instance digest.  Thus
any implementation can reconstruct `A` deterministically, and the catalog
sidecar binds the exact JSONL bytes used by the evaluator.  The checked-in
Python facade is the reference implementation; independent implementations
can be compared row-for-row against it.

The current production catalog SHA-256 is
`318f32b68e8a30c6bcd5867c89eb10d87a5472614401f4e6df0f55356cdc98a6`.
Production witnesses and errors are sampled during corpus construction and must
not be retained as evaluator inputs or release assets.  Reproducibility means
reproducing the public instance from its released record, not recovering any
private generation seed.

## Interface

The packaged task contract uses these agent-visible paths:

```text
/frontier/task/repo/catalog.jsonl              public instance catalog
/frontier/task/repo/lwe_instance.py            stable read-only Python facade
/frontier/task/bin/add_solution.py                   cumulative-ledger helper
/workspace/payload/solution.json                 writable submission ledger
```

The production catalog contains exactly 200 instances.  Its entries and
verified digest are authoritative.

Use `repo/lwe_instance.py`; callers do not need to import
`lwe_challenge.*` or parse catalog JSON directly.  This complete example loads
the catalog, streams matrix rows, and checks a candidate:

```python
from pathlib import Path
import sys

repo_dir = Path("/frontier/task/repo")
sys.path.insert(0, str(repo_dir))

from lwe_instance import Catalog

catalog = Catalog.load(repo_dir / "catalog.jsonl")
print(catalog.catalog_id, len(catalog.instances))

instance = catalog.instances[0]
print(instance.instance_id, instance.n, instance.m, instance.q)
first_row = next(instance.iter_rows())

candidate = (0,) * instance.n  # replace with a candidate from your analysis
verdict = instance.validate_secret(candidate)
print(first_row, verdict.ok, verdict.code)
```

`Catalog.instances` preserves public catalog order and `Catalog.get(id)` does
lookup by ID.  Each immutable `Instance` exposes `n`, `m`, `q`, `b`, the matrix
kind/seed/domain/alphabet/row weight, all public secret and error distribution
parameters, all verifier bounds, and the instance digest.  It also provides:

- `iter_rows()` to stream rows without materializing `A`;
- `materialize_row_block(start, stop)` for a bounded row block;
- `materialize_rows()` when the full public matrix fits your memory budget;
- `matvec(secret)` for `A * secret mod q`; and
- `validate_secret(secret)` for the exact public acceptance check.

Prefer row streaming or blocks for large instances.

The declared matrix kinds are `uniform`, `small_alphabet`, `sparse_uniform`,
and `sparse_small_alphabet`.  Secret-generation kinds are `uniform_mod_q`,
`iid_alphabet`, `exact_weight_alphabet`, `balanced_exact_weight_signed`, and
`centered_binomial`; error kinds are `truncated_discrete_gaussian`,
`centered_binomial`, `bounded_uniform`, and `sparse_bounded`.  These labels and
every associated parameter are public.  The balanced signed plant uses an even
exact weight and places equally many `-1` and `1` entries; its public acceptance
predicate remains the separately declared alphabet and nonzero-weight range.

The facade property secret_alphabet is the acceptance alphabet.  For
exact-weight secrets, the `exact_weight_alphabet` generation alphabet is
`secret_alphabet` with zero removed because zero fills positions outside the
exact support.  The `balanced_exact_weight_signed` plant specifically uses
`{-1,1}` on its support even though the verifier also accepts any other vector
meeting the published signed alphabet and exact-weight predicate.  For the
other alphabet-based kinds, `iid_alphabet` and `centered_binomial` generation
and acceptance alphabets coincide.  The `uniform_mod_q` kind instead has no
alphabet and uses all residues `0 <= s_j < q`.

### Cumulative JSON ledger

The only scored artifact is `/workspace/payload/solution.json`. Its strict schema is:

```json
{
  "schema_version": 1,
  "solutions": [
    {"instance_id": "example-id", "secret": [1, 0, -1]}
  ]
}
```

The whole-file rules require an unambiguous UTF-8 JSON object whose top-level
fields are exactly `schema_version` and `solutions`, with integer
`schema_version == 1` and a `solutions` array of at most 200 elements.  No JSON
object may repeat a key, and the encoded file may contain at most 2,000,000
bytes.  The decoder also permits at most 4 levels of JSON nesting and 820,205
decoded nodes.  Exceeding either decoder budget is a whole-file
`invalid_json` error; for example, a nested secret such as `[[0]]` is rejected
before per-record validation.  Therefore, violating a whole-file rule scores
zero even when another record is valid.

Records are checked separately.  A canonical record has exactly `instance_id`
and `secret`; the ID matches `[A-Za-z0-9][A-Za-z0-9._-]{0,63}` and names a
public instance, while `secret` is an integer array whose components have
absolute value at most `2^63 - 1` and whose length is at most 4,096.  A
per-record rejection does not invalidate the whole ledger; unrelated valid
records can still score; the canonical empty ledger is pre-provisioned at
`/workspace/payload/solution.json` as:

```json
{"schema_version":1,"solutions":[]}
```

Use the locked, atomic cumulative helper instead of rebuilding the file:

```bash
python3 /frontier/task/bin/add_solution.py INSTANCE_ID '1,0,-1'
```

The helper reads `/workspace/payload/solution.json`, retains its prior records, adds the new
record, writes canonical sorted JSON atomically, and prints the resulting
record count.  An identical existing witness is an idempotent no-op.  A
different existing witness is rejected unless you pass --replace explicitly:

```bash
python3 /frontier/task/bin/add_solution.py --replace INSTANCE_ID '0,1,-1'
```

A successful add_solution exit and write enforces canonical structural ledger
rules: exact object fields and version, unique regex-valid IDs, integer arrays
of at most 4,096 bounded components, and the 200-record and 2,000,000-byte
caps.  The helper refuses an update that would exceed the 200-record or
2,000,000-byte cap, as well as either component bound.
Success does not prove catalog membership or mathematical witness validity
because the helper does not load the catalog.  Validate with
`instance.validate_secret(candidate)` before adding a record.  After every
successful helper call, the ledger remains cumulative.

Duplicate handling by the evaluator is deliberately strict: every repeated
safe instance_id invalidates every occurrence for that ID, even when the
vectors are identical or one occurrence is malformed.  That ID earns no point.
Here safe means syntactically valid under the instance-ID regex; it does not
mean that the ID occurs in the public catalog.  duplicate_count counts distinct
syntactically valid IDs that occur more than once, not duplicate occurrences.
conflict_count counts distinct IDs with more than one distinct syntactically
valid integer vector, not conflicting pairs or occurrences.  Here a
syntactically valid integer vector is a JSON array of non-boolean integers
within the ledger integer bound and at most 4,096 components; it need not have
the right dimension or pass the mathematical witness predicates.  Every
rejected occurrence contributes to `invalid_count`.

unknown_count counts distinct regex-valid IDs absent from the catalog, even
when an ID is repeated or another field in its record is malformed.  This set
count is independent of the per-occurrence rejection code.  unknown_instance_id
applies only to a unique otherwise syntactically admissible record.  For a
unique admissible record, a syntactically valid but unknown instance_id is a
per-record `unknown_instance_id` rejection.  invalid_record_fields takes
precedence for a unique malformed unknown when its record fields are wrong.
With exact fields but a malformed secret, invalid_secret takes precedence for
a unique malformed unknown over the unknown-ID code.  For repeated IDs,
duplicate_instance_id takes precedence for every occurrence of a repeated ID.
The helper normally prevents duplicate records; use `--replace` instead of
creating a second JSON record.

## Validity

For one public instance, a submitted vector `s` is valid exactly when all of
the following hold:

1. `len(s) == n`, and every component is a JSON/Python integer (booleans are
   not integers for this contract).
2. If `secret_predicate_kind == "alphabet"`, every component belongs to
   `secret_alphabet`.  If it is `"mod_q"`, every component is in
   `0 <= s_j < q`.
3. The number of nonzero components is between `secret_min_nonzero` and
   `secret_max_nonzero`, inclusive.
4. Compute `p = A * s mod q`, then compute each canonical residual as
   `r_i = center_q(b_i - p_i)`.  Equivalently, the evaluator computes
   `center_q(b_i - (A s)_i)`.  The centered representative lies in
   `[-floor(q/2), ceil(q/2) - 1]`.
5. `max_i(abs(r_i)) <= error_max_abs`.
6. When `error_max_l1` is not `None`,
   `sum_i(abs(r_i)) <= error_max_l1`.
7. When `error_max_l2_squared` is not `None`,
   `sum_i(r_i * r_i) <= error_max_l2_squared`.
8. When `error_max_nonzero` is not `None`, the number of nonzero residual
   components is at most `error_max_nonzero`.

The public sampling-distribution fields are analysis inputs; acceptance is
determined by the predicates above.  The evaluator holds no secret, planted
answer, private seed, or private error.  It reconstructs `A` from public data
and checks only the submitted vector against public `(A, b)` and the public
predicates.

The method validate_secret checks mathematical witness validity for one
already-selected instance.  At the submission boundary, ledger admissibility
is a separate check covering the JSON/file limits, exact record shape, ID
handling, and duplicate rules below.  Thus an `ok` mathematical verdict does
not by itself make arbitrary ledger JSON admissible.

## Scoring

Every catalog instance has equal weight.  Let `solved_count` be the number of
unique instance IDs whose submitted secret passes all predicates, and let
`instance_count` be the catalog size:

```text
score = 100 * solved_count / instance_count
score_unbounded = solved_count
```

An invalid record does not erase unrelated valid records.  Whole-ledger format
errors score zero, so keep the helper-produced ledger intact.

The independent continuous score remains available for partial solutions.
Benchmark pass is separate: every one of the 200 public instances must have a
valid submitted secret. Each instance is a workload, with the exact binary metric
valid_witness (1 for a valid admissible solution, 0 otherwise). Pass requires
strict improvement over the frozen empty reference ledger on every workload.
There is no tolerance or cross-instance compensation. Train and final check the
same fully public corpus; there is no hidden final split.

See the operational `/frontier/task/README.md` for public feedback fields and submission commands.

## Feedback and grading

Train returns candidate-only `instance_count`, `solved_count`, `submitted_count`,
`invalid_count`, `duplicate_count`, `conflict_count`, `unknown_count`, and
`rejection_code_counts`. `beat_spec_progress.valid_witness` contains `passed`,
`failed`, `total`, `pass_percent`, and `fail_percent` across all 200 workloads.
The count-valued `score_unbounded` is also returned. Exact per-instance decisions,
reference measurements and benchmark `beats_reference` remain in private evidence;
final metrics are not returned to the Agent. The public checker remains available
for local per-instance diagnosis. Verifier execution success is not benchmark pass.

Train and final use the same public corpus. The selected final ledger is independently evaluated after collection; the last train result is not a final result. See `/frontier/task/bin/README.md` for stored and explicit final selection.

## Resources

The configured environment is CPU-only and provides:

- Ubuntu 24.04 with its distro Python 3.12 runtime;
- 8 CPU cores;
- 32 GiB memory;
- 32 GiB storage;
- 18,000 seconds (5 hours) of Agent answering time, with a separate 120-second final evaluation budget;
- build time is part of the same Agent wall-time budget;
- Ubuntu's generic `fplll-tools` package and standard build dependencies.

Do not rely on a GPU.
