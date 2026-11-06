# Structured-LWE Public Witness Recovery

Recover valid secret vectors for as many of the 200 public structured-LWE instances as you can
and record them in the witness ledger `solution.json`. The score is the fraction solved; beating
the reference requires a valid witness for every instance.

## Task
For each instance the public data define dimensions `n` and `m`, a modulus `q`, a reproducible
matrix `A`, a target vector `b`, public secret and error distributions, and the exact acceptance
predicates. A valid answer satisfies

```text
b = A s + e (mod q)
```

with a secret `s` and centered residual `e` that pass those predicates. Any valid `s` is
accepted; you need not recover the witness used to generate the instance. The judge holds no
secret, planted answer, private seed or private error: it reconstructs `A` from public data and
checks only the submitted vector against public `(A, b)` and the public predicates.

## What you may change
- The only scored artifact is `/workspace/repo/solution.json`, submitted as a patch of the
  starter ledger. A patch that changes any other provided file is refused, so restore
  experiments on the copied checker with `git checkout` before submitting.
- Write solver code elsewhere in `/workspace`, or as new files in `/workspace/repo` (new files
  are never submitted). Solver source is not submitted.
- Final evaluation always uses the immutable checker and catalog in the Verifier.

## Interface
```text
/frontier/task/repo/catalog.jsonl        public instance catalog (exactly 200 instances)
/frontier/task/repo/lwe_instance.py      stable read-only Python facade
/workspace/repo/tools/add_solution.py    cumulative-ledger helper
/workspace/repo/solution.json            writable submission ledger
```

Use `lwe_instance.py`; there is no need to import `lwe_challenge.*` or parse the catalog JSON:

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

`Catalog.instances` keeps catalog order; `Catalog.get(id)` looks up by ID. Each immutable
`Instance` exposes `n`, `m`, `q`, `b`, the matrix kind/seed/domain/alphabet/row weight, every
public secret and error distribution parameter, every verifier bound and the instance digest,
and provides `iter_rows()` (stream rows without materializing `A`), `materialize_row_block(start,
stop)`, `materialize_rows()` (when the full matrix fits in memory), `matvec(secret)` (`A * secret
mod q`) and `validate_secret(secret)` (the exact public acceptance check). Prefer row streaming
or blocks for large instances.

Matrix kinds: `uniform`, `small_alphabet`, `sparse_uniform`, `sparse_small_alphabet`. Secret
generation kinds: `uniform_mod_q`, `iid_alphabet`, `exact_weight_alphabet`,
`balanced_exact_weight_signed`, `centered_binomial`. Error kinds: `truncated_discrete_gaussian`,
`centered_binomial`, `bounded_uniform`, `sparse_bounded`. All labels and parameters are public.
`secret_alphabet` is the acceptance alphabet. For `exact_weight_alphabet` the generation
alphabet is `secret_alphabet` without zero (zero fills positions outside the support). The
`balanced_exact_weight_signed` plant uses an even exact weight with equally many `-1` and `1`
entries, but acceptance is the declared signed alphabet and nonzero-weight range, so any vector
meeting them is accepted. For `iid_alphabet` and `centered_binomial`, generation and acceptance
alphabets coincide; `uniform_mod_q` has no alphabet and uses all residues `0 <= s_j < q`.

The ledger schema (the starter is the canonical empty ledger
`{"schema_version":1,"solutions":[]}`):

```json
{
  "schema_version": 1,
  "solutions": [
    {"instance_id": "example-id", "secret": [1, 0, -1]}
  ]
}
```

Add witnesses with the locked, atomic helper instead of rebuilding the file:

```bash
python3 /workspace/repo/tools/add_solution.py INSTANCE_ID '1,0,-1'
python3 /workspace/repo/tools/add_solution.py --replace INSTANCE_ID '0,1,-1'
```

It keeps prior records, adds the new one, writes canonical sorted JSON atomically and prints the
record count. An identical existing witness is a no-op; a different one is refused unless
`--replace` is given (use it instead of adding a second record). It enforces the structural rules
below and refuses updates past the 200-record or 2,000,000-byte cap or either component bound,
but does not load the catalog: success proves neither catalog membership nor witness validity, so
check with `instance.validate_secret(candidate)` first. An `ok` verdict alone does not make
arbitrary ledger JSON admissible.

## Correctness
Witness validity, for one instance and vector `s`, all of:
1. `len(s) == n` and every component is a JSON/Python integer (booleans are not integers).
2. If `secret_predicate_kind == "alphabet"`, every component is in `secret_alphabet`; if
   `"mod_q"`, every component is in `0 <= s_j < q`.
3. The number of nonzero components is in `[secret_min_nonzero, secret_max_nonzero]`.
4. Residuals `r_i = center_q(b_i - (A s)_i)`, with the centered representative in
   `[-floor(q/2), ceil(q/2) - 1]`.
5. `max_i |r_i| <= error_max_abs`.
6. If `error_max_l1` is not `None`: `sum_i |r_i| <= error_max_l1`.
7. If `error_max_l2_squared` is not `None`: `sum_i r_i^2 <= error_max_l2_squared`.
8. If `error_max_nonzero` is not `None`: at most that many nonzero `r_i`.

The sampling-distribution fields are analysis inputs; only these predicates decide acceptance.

Whole-ledger rules (a violation scores 0 even if other records are valid): an unambiguous UTF-8
JSON object with exactly the fields `schema_version` (integer 1) and `solutions` (an array of at
most 200 elements); no repeated object key; at most 2,000,000 bytes; at most 4 levels of nesting
and 820,205 decoded nodes (exceeding either is `invalid_json`, so e.g. a secret `[[0]]` is
rejected before per-record checks).

Per-record rules (a rejected record does not invalidate unrelated valid ones): exactly the
fields `instance_id` and `secret`; the ID matches `[A-Za-z0-9][A-Za-z0-9._-]{0,63}` and names a
catalog instance; `secret` is an integer array of at most 4,096 components, each of absolute
value at most `2^63 - 1`.

Duplicates are strict: every repeated regex-valid `instance_id` invalidates all its
occurrences, even identical ones or one malformed, and that ID earns no point. Rejection-code
precedence: `duplicate_instance_id` for every occurrence of a repeated ID; otherwise
`invalid_record_fields` if the fields are wrong, then `invalid_secret` for a malformed secret,
then `unknown_instance_id` for an otherwise admissible unknown ID.

## Workload
200 public instances, 20 in each of ten structural families:

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

Train and final check the same fully public corpus; there is no hidden split, and the final
ledger is evaluated independently after collection. The catalog omits difficulty labels, runtime
bins, calibration estimates, cryptanalysis paths and reference solutions; treat instance IDs as
opaque. Each record holds the complete matrix description, a SHAKE-256 seed and domain separator,
inline `b`, all predicates and an instance digest, so any implementation can rebuild `A`
deterministically; the catalog sidecar binds the exact JSONL bytes (production catalog SHA-256
`318f32b68e8a30c6bcd5867c89eb10d87a5472614401f4e6df0f55356cdc98a6`). The checked-in facade is
the reference implementation for row-by-row comparison. Generation witnesses and errors are not
evaluator inputs or release assets; reproducibility means rebuilding the public instance, not
recovering a private seed.

## Scoring

```text
score           = 100 * solved_count / instance_count
score_unbounded = solved_count
```

`solved_count` is the number of unique instance IDs whose submitted secret passes all predicates;
`instance_count` is the catalog size (200). Every instance has equal weight: the empty ledger
scores 0, each solved instance adds 0.5, all 200 score 100. Whole-ledger errors score 0, so keep
the helper-produced ledger intact.

**Beats the reference**: each instance is a workload with the binary metric `valid_witness` (1 for a
valid admissible solution, else 0), which must strictly exceed the frozen empty reference
ledger's (0) on every workload, so all 200 must be solved. No tolerance or cross-instance
compensation.

## Running locally
Validate each candidate with `instance.validate_secret(candidate)` (example above) before adding
it with `tools/add_solution.py`; accumulate validated witnesses, keeping all earlier ones. One
train call may carry many new witnesses.

## Environment
CPU only (no GPU): Ubuntu 24.04 with its distro Python 3.12, Ubuntu's generic `fplll-tools`
package and standard build dependencies. Build time is part of the answering time.
