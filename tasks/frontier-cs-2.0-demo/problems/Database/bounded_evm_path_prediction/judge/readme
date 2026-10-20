# Bounded EVM Path Prediction

## Problem

SeerEVM is a research prototype for accelerating blockchain transaction execution by fine-grained branch prediction. An Ethereum Virtual Machine (EVM) executes the bytecode of smart-contract transactions against mutable blockchain state; a contract's conditional jumps can therefore depend on values written by earlier transactions. SeerEVM is an EVM execution engine that pre-executes transactions and records control-flow and state-access information for use during ordinary block execution. This benchmark reads the authors' approximately 370 MB snapshot of Ethereum mainnet blocks, transactions, and execution state beginning at block 14,650,000. During replay, each state-dependent branch identity has online predictor state; the predictor receives queries and later receives the corresponding observed branch directions. The experiment writes `experiments/prediction_height.txt`, containing per-block, per-category transaction-path satisfaction and branch-direction accuracy records for complex and normal contract transactions. A correct submission must preserve the predictor interface and direction encodings, compile in SeerEVM, finish the unchanged replay, and produce a parseable report containing the required epochs and at least 20 scored observations. The objective is to maximise a bounded score derived from mean branch-direction accuracy and mean transaction-path satisfaction. The implementation of the branch predictor in `core/vm/predictor.go` has been removed and must be supplied.

## Your workspace

`/app` contains the authors' SeerEVM Go repository, based on upstream commit `24aa1bd2210d27f13cd5eabd8e7b59cf0ec458d5`. The Ethereum snapshot and vendored dependencies are installed for the judge rather than included in the editable source tree. The only file that may change is:

```
core/vm/predictor.go
```

The EVM, block replay, branch table, speculative worker, fast-execution path, experiment driver, predictor callers, and harness are fixed; changes to them or to any other file are rejected when the patch is applied. Local binaries, caches, reports, and untracked files are not submitted.

## Design space

The editable scope is the package-local predictor state and the behavior of the entry points listed under Interface. The surrounding branch table retains one predictor instance per branch identity and supplies queries and observations through fixed call sites. A query may return either direction or `uncertain`; there is no minimum prediction coverage requirement. Branch discovery, transaction execution, report generation, command-line handling, direction encodings, transaction ordering, and predictor call sites are outside the editable scope. The edited file must satisfy the declared source, build, replay, memory, and timeout limits without changes to its callers.

## Workload

The judge copies the authors' `ethereumdata_1465_small` snapshot into an isolated run directory, builds the complete patched repository with vendored dependencies, and starts the experiment from `experiments/`. The packaged snapshot occupies 369,941,487 bytes (about 353 MiB) in 221 files. Its database contains the initial block and state at height 14,650,000 plus the 1,000 following canonical blocks through height 14,651,000.

The workload is a fixed replay of this consecutive historical mainnet segment, not generated data and not a collection whose order is shuffled between runs.

The exact invocation is:

```
seerevm --indicator 1 --start 14650000 --blockNum N
```

`--indicator 1` selects the prediction-accuracy experiment, `--start 14650000` identifies the initial block and state, and `--blockNum N` gives the number of following blocks to replay. `N` is `200` for development feedback and `600` for final grading. Each role performs one fixed sequential replay without shuffling or repetition. Development replays and scores blocks 14,650,001 through 14,650,200. Final grading replays blocks 14,650,001 through 14,650,600; its continuous score uses only observations from blocks 14,650,201 through 14,650,400, and the separate beating check grades all three 200-block epochs.

Transactions with receipt gas usage of at least 100,000 are assigned to the complex category; the remaining observed contract transactions are assigned to the normal category. A report record is emitted for a block/category only when that category has the required transaction and branch observations. The fixed 200-block development replay produces 390 parsed records with the calibration implementation.

## Interface

There is no standalone predictor trace file. The replay first opens `../ethereumdata_1465_small/copychain` as an Ethereum LevelDB database. It finds a canonical block hash by number, loads the corresponding Recursive Length Prefix (RLP)-encoded header and body, reconstructs the block, and executes the body's transactions in order. The relevant stored key/value records have this layout:

```
canonical mapping
  key:   0x68 | uint64_be(block_number) | 0x6e
         "h"                                "n"
  value: 32-byte canonical block_hash

header
  key:   0x68 | uint64_be(block_number) | block_hash
  value: RLP header {parentHash, uncleHash, coinbase, stateRoot,
                     transactionsRoot, receiptsRoot, logsBloom, difficulty,
                     number, gasLimit, gasUsed, timestamp, extraData,
                     mixHash, nonce, baseFeePerGas?, withdrawalsRoot?}

body
  key:   0x62 | uint64_be(block_number) | block_hash
         "b"
  value: RLP body {transactions[], uncles[], withdrawals?}
```

For the first replayed block, height 14,650,001 is `0000000000df8a91` as an eight-byte big-endian integer. The actual canonical mapping in the supplied database is:

```
key:   68 | 0000000000df8a91 | 6e
       ^h   ^block number      ^n
value: 90195a15df6b49990d8cea7ecea40e3882d6142c510036ac73cd79164f0e5375
       ^32-byte canonical block hash
```

That hash completes the header key `68 | 0000000000df8a91 | <hash>` and body key `62 | 0000000000df8a91 | <hash>`. The header record is 542 bytes and the body record is 131,110 bytes in this snapshot. Account and storage tries rooted at the header's `stateRoot` provide the state used to execute the transactions.

Predictor input arrives through Go calls rather than a separate trace file. Preserve these package-local entry points and meanings:

- `type predictor` is the state for one branch identity. It must contain `lastPred []int`, which callers use as a FIFO of outstanding results, and `observationCount int`, which surrounding code reads directly.
- `initPredictor() *predictor` returns a new predictor ready to receive queries and observations.
- `(*predictor).predict(train bool) int` returns `uncertain` (`1`), `taken` (`2`), or `notTaken` (`3`). Ordinary queries pass `train=false`. When an observation has no queued ordinary-query result, the caller obtains its paired result by passing `train=true`.
- `(*predictor).update(dir, predDir int)` receives the observed direction in `dir` and its paired prediction in `predDir`, using the same encodings. When `lastPred` is nonempty, the method must remove the oldest queued result corresponding to this update.
- `Bool2branchRes(bool) int` returns `taken` for `true` and `notTaken` for `false`.
- `BranchRes2bool(int) bool` returns `true` exactly when its argument is `taken`.

For example, this legal sequence represents an ordinary query that returned `taken`, followed later by an observed `notTaken` direction:

```go
p := initPredictor()
pred := p.predict(false) // pred is 2 in this example
p.lastPred = append(p.lastPred, pred) // performed by the fixed Entry.Predict caller
p.update(3, p.lastPred[0])            // performed by the fixed Entry.Update caller
```

The fixed `Entry.Predict` caller appends every returned result to `lastPred`. The fixed `Entry.Update` caller passes the oldest queued result to `update`; if the queue is empty, it instead calls `predict(true)` and passes that result. After `update`, the instance must be ready for later calls.

The experiment appends its report to `experiments/prediction_height.txt`. Each observation has this exact layout:

```
0.93 && 0.97
^T          ^B
```

`T` is the authors' transaction-path satisfaction ratio for that block/category, and `B` is the fraction of observed branch directions that were satisfied. `B` and complex-category `T` are in `[0,1]`. Normal-category `T` is not a probability: the authors divide satisfied normal contract calls by total contract calls minus all high-gas transactions, a subtraction that also includes contract creations excluded from the call count. It can therefore exceed `1`. Preserve these printed values without clamping or renormalizing them. The fixed printer emits two decimal places; the judge retains its historical finite `0.xx`/`1.xx` transaction-ratio domain for normal records. Complex-category records precede a blank line and normal-category records follow it within each epoch. Every 200 blocks, the printer writes `===== Block height 200 =====`, `===== Block height 400 =====`, and so on, emits the accumulated records, and resets its accumulators. Development scoring reads the records under the `200` heading. Final scoring requires the `200`, `400` and `600` headings; its continuous score reads only records under `400`, while the separate beating check (below) uses all three epochs. At least 20 records and both transaction categories must be present in the scored epoch, and both categories must be present in every graded epoch. Duplicate epoch headings, malformed or nonfinite records, branch values outside `[0.00,1.00]`, complex transaction ratios outside `[0.00,1.00]`, and normal transaction ratios outside the historical `[0.00,1.99]` domain are rejected. Both columns must have exactly two decimal places.

## Validity

The submission must be a nonempty textual Git patch at `/app/solution.patch`, change only `core/vm/predictor.go`, contain no binary patch content, and be at most 131,072 bytes. After application, the edited file must be valid UTF-8, be at most 100,000 bytes, remain in package `vm`, provide all required symbols and fields, and compile with the complete repository under Go 1.22.

The edited source may import only `math`, `math/bits`, `sort`, `sync`, and `sync/atomic`. A quoted or raw-string token consisting only of letters, digits, underscores, dots, slashes, or hyphens is checked as a package-path token and must also belong to that allowlist. The source may not use operating-system or subprocess access, networking, `unsafe`, judge or environment inspection, Go directives, or an `init` function.

The build timeout is 300 seconds and the replay timeout is 420 seconds. Patch rejection, policy violation, build failure, crash, nonzero replay exit, a missing report, a report larger than 2,000,000 bytes, fewer than 20 parsed observations, a missing required epoch, or a build/replay timeout scores zero before prediction quality is considered.

## Scoring

Maximise the returned score on the 0–100 scale. If the evaluator parses `m` scored records `(T_i, B_i)`, it computes:

```
T = (1/m) * sum(i=1..m, T_i)
B = (1/m) * sum(i=1..m, B_i)
Q = 100 * (0.8 * B + 0.2 * T)
```

Let `Q0` be the raw quality of the judge's always-taken calibration implementation on the same scored block range. The exact bounded scoring rule is:

```
if Q0 >= 100:
    score = 0
else:
    normalized = max(0, (Q - Q0) / (100 - Q0))
    score = max(0, min(100, 100 * normalized * normalized))
```

The `0.8` branch weight, `0.2` transaction-path weight, calibration normalization, and square are challenge constants (not from materials). Every parsed record has equal weight, including records from both transaction categories. Runtime is a validity condition and is not a score term.

## Beating the reference

Beating the reference is checked separately for every graded combination of transaction category (complex, normal) and 200-block reporting epoch. Development grades its single epoch (blocks 14,650,001–14,650,200). Final replays 600 blocks and grades three epochs: the freshly replayed public prefix (blocks 14,650,001–14,650,200) and two held-out epochs (blocks 14,650,201–14,650,400 and 14,650,401–14,650,600). In each category and epoch, improve the sum of printed branch-accuracy hundredths by at least one, preserve the sum of printed transaction-path satisfaction-ratio hundredths without clamping normal values above one, and retain exactly the paired reference observation count. All three checks must pass in every graded category and epoch, so a gain in one interval cannot compensate for a loss in another. The continuous score keeps its existing definition. Train feedback reports only approved absolute candidate summaries and per-metric workload pass/fail counts and percentages; reference measurements and category decisions remain private.

## Calibration

The always-taken calibration implementation has development raw quality `Q0 = 43.3410` on the 200-block replay and therefore receives score `0`. That run produces 390 parsed records. The report printer rounds every `T_i` and `B_i` to two decimal places before the evaluator reads them, and the evaluator averages those printed values. Repeated fixed replays of an unchanged implementation have identical parsed scores in calibration, so no run-to-run score noise was measured; changes may still be quantized by the record format. About `0.1` returned score point is distinguishable in the aggregate development result.

## Feedback and grading

Train submissions return the bounded score, raw quality, mean branch accuracy, mean transaction-path satisfaction ratio, observation count, and `beat_spec_progress` for each of the three metrics. Each progress object reports `passed`, `failed`, `total`, `pass_percent`, and `fail_percent` over exactly two category workloads in development (six category-epoch workloads in final, which returns no Agent-visible metrics). Per-category measurements and decisions, reference values, and margins stay private. Final submissions do not return Agent-visible metrics. At most 50 development submissions are allowed. **The graded suite is not the suite you receive feedback on**: development scores blocks 14,650,001–14,650,200, while the final continuous score uses only blocks 14,650,201–14,650,400. The final beating check additionally grades the replayed development prefix and the held-out blocks 14,650,401–14,650,600. There is no generated case order or undisclosed seed.

## Submission

Submit the Git patch at `/app/solution.patch`. From `/app`, run:

```
bash /app/make_submission.sh
bash /app/submit.sh
```

`make_submission.sh` creates the patch from the working tree and rejects changes outside `core/vm/predictor.go` rather than silently dropping them. The patch must contain a change to that file; an empty patch is invalid. Temporary local files and generated experiment output are not submitted.

## Resources

The evaluation container provides 8 vCPUs, 8,192 MB of memory, and 4,096 MB of storage; the 600-block final replay of the reference predictor peaks at about 4.7 GB resident memory, while the 200-block development replay fits in the 4,096 MB Agent workspace. It uses Go 1.22 with vendored dependencies. The build timeout is 300 seconds, each SeerEVM replay timeout is 420 seconds, and the total evaluation timeout is 1,800 seconds. The submission patch is limited to 131,072 bytes, the edited source is limited to 100,000 bytes, and exactly one changed file is permitted.

Train returns these candidate-only absolute summaries: `raw_quality`, `branch_accuracy`, `transaction_path_accuracy`, `observations`. For each comparison metric, `beat_spec_progress` contains `passed`, `failed`, `total`, `pass_percent`, and `fail_percent`; the denominator is the fixed workload count for the selected suite. No reference measurements, private workload identities, per-workload decisions, or margins are returned.
