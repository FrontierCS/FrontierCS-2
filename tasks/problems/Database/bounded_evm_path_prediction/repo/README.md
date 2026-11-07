# Bounded EVM Path Prediction

Supply the branch predictor of SeerEVM so that it predicts the directions of state-dependent
EVM branches during a replay of real Ethereum blocks as accurately as possible. A valid
predictor keeps the interface below, compiles in SeerEVM and lets the unchanged replay finish
with a parseable report; the score rewards branch accuracy and transaction-path satisfaction
above an always-taken predictor.

## Task

SeerEVM (Xiao et al., "Seer: Accelerating Blockchain Transaction Execution by Fine-Grained
Branch Prediction", PVLDB) pre-executes transactions and keeps one online predictor per
state-dependent branch identity; the predictor answers queries and later receives the observed
directions. The predictor implementation in `core/vm/predictor.go` has been removed.

## What you may change

- Only `core/vm/predictor.go`: the package-local predictor state and the entry points below.
  Branch discovery, transaction execution, report generation, command-line handling, direction
  encodings, transaction ordering and the predictor call sites are fixed.
- The file must be valid UTF-8, at most 100,000 bytes, and begin (after optional whitespace)
  with `package vm`. Keep the constant `WARMUP_OBSERVATIONS` defined: `core/vm/var_table.go`
  uses it.
- It may import only `math`, `math/bits`, `sort`, `sync` and `sync/atomic`.
- A violation is rejected as `content_policy` (`too_large` for the size).

## Interface

Keep these package-local definitions in `core/vm/predictor.go`:

- `type predictor` holds the state of one branch identity and must have the fields
  `lastPred []int` (a FIFO of outstanding results, used by callers) and `observationCount int`
  (read by surrounding code).
- `initPredictor() *predictor` returns a predictor ready for queries and observations.
- `(*predictor).predict(train bool) int` returns `uncertain` (1), `taken` (2) or `notTaken` (3).
  Ordinary queries pass `train=false`; when an observation has no queued result the caller gets
  its paired result with `train=true`.
- `(*predictor).update(dir, predDir int)` receives the observed direction and its paired
  prediction; when `lastPred` is nonempty it must remove the oldest queued result.
- `Bool2branchRes(bool) int` maps `true` to `taken`, `false` to `notTaken`;
  `BranchRes2bool(int) bool` is `true` exactly for `taken`.

The fixed `Entry.Predict` caller appends every `predict` result to `lastPred`; `Entry.Update`
passes the oldest queued result (or `predict(true)` if the queue is empty) to `update`:

```go
p := initPredictor()
pred := p.predict(false)               // e.g. 2 (taken)
p.lastPred = append(p.lastPred, pred)  // Entry.Predict
p.update(3, p.lastPred[0])             // Entry.Update: observed notTaken
```

There is no minimum prediction coverage: `uncertain` is always allowed.

## Correctness

The judge builds the complete repository with its vendored modules (`go build`, Go 1.22,
300 s) and runs, from `experiments/`, `seerevm --indicator 1 --start 14650000 --blockNum N`
(420 s). The run scores 0 if the policy above is broken, the build fails, the replay crashes,
exits nonzero, times out, or exceeds the judge's 8 GiB memory (the 600-block reference replay
already peaks at about 4.7 GB resident), the report `prediction_height.txt` (in `experiments/`) is missing or larger than
2,000,000 bytes, the scored epoch has fewer than 20 records, a required epoch or category is
missing, an epoch heading repeats, or a record is malformed: each record is `T && B` with exactly
two decimals, `B` and complex-category `T` in [0.00, 1.00], normal-category `T` in
[0.00, 1.99].

## Workload

The replay reads the authors' snapshot of Ethereum mainnet from block 14,650,000 (the initial
state plus 1,000 canonical blocks). Train replays 200 blocks (`N = 200`) and scores epoch
14,650,001–14,650,200; final replays 600 blocks (`N = 600`), scores only 14,650,201–14,650,400
and grades all three 200-block epochs for beating the reference. Each role is one fixed
sequential replay with no shuffling or seeds.

The report holds, per 200-block epoch, a heading `===== Block height 200 =====` (400, 600), the
complex-category records (receipt gas at least 100,000), a blank line, and the normal-category
records. `T` is the authors' transaction-path satisfaction ratio (for normal transactions it can
exceed 1; it is kept unclamped) and `B` the fraction of satisfied branch directions.

## Scoring

Over the `m` records `(T_i, B_i)` of the scored epoch:

```text
Q     = 100 * (0.8 * mean(B_i) + 0.2 * mean(T_i))
score = 0                                            if Q0 >= 100
score = 100 * max(0, (Q - Q0) / (100 - Q0))^2         otherwise (capped at 100)
```

`Q0` is the quality of the always-taken calibration predictor on the same blocks (43.3410 on the
train epoch, score 0). Every record has equal weight. Runtime only has to stay within the limits.

**Beats the reference** is checked per category (complex, normal) and graded epoch (one in train,
three in final): `branch_accuracy`: the sum of printed branch hundredths exceeds the reference's by at least 1;
`transaction_path_accuracy`: the sum of transaction-path hundredths is at least the
reference's; `report_coverage`: the record count equals the reference's and is positive. The
first two also require equal coverage. Every check must pass in every category and epoch.

## Running locally

```sh
cd /workspace/repo && go build -o /workspace/seerevm . && go vet ./core/vm
```

Modules resolve offline from `/opt/frontier/go-mod`. The Ethereum snapshot is not in the Agent
image, so the replay itself runs only in the judge; use train submissions for it.

## Notes

The replay is a representative segment of mainnet execution: do not recognise or special-case
blocks, branch identities or contracts.
