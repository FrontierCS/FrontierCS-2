# QuarterPIR full-compute scope

This page records the redesign of Private Set List Decoding (`quarter_pir`)
from a server-handler-only task into a task whose editable scope is all of the
protocol's computation, with online communication as the headline metric.
Measurements and fixture results are in
full-compute-scope-evidence.json.

Paths below are relative to
`tasks/problems/Crypto/quarter_pir/`; `CODE` is
`judge/harbor/app/quarter_pir/code/` (the agent copy under
`agent-base/frontier/task/repo/` is byte-identical).

## What a submission may change

Only the Go package `CODE/pir/`: `client.go` (preprocessing, request
generation, reconstruction, hint refresh), `reduce.go` (request reduction),
`server.go` (server handler), plus new `.go` files. The package is Go source
only; all assembly lives in trusted packages.

The base tree ships the complete reference client and the reference
`ReduceRequest` (identity); the server handler body is a stub, as before.
`judge/reference.patch` fills in the upstream handler. The reference is
therefore one swappable artifact: a hardened reference would be a larger
`reference.patch` and nothing else changes.

## Processes and who holds what

| Process | Built from | Submission code inside | Holds |
| --- | --- | --- | --- |
| evaluator (`judge/evaluator.py`) | trusted | none | seeds, pairing, validity verdict, gate |
| conductor (`CODE/conductor/`) | trusted tree only | none | query stream, trusted reference request generator, transport to the server, all experiment clocks, value oracle, result record |
| server (`CODE/server/`) | tree under test | `pir.HandleQuery` | database, server clock, replay |
| client driver (`CODE/ec24/`) | tree under test | `pir.Client` | coin seed, database stream; no network, no database seed |
| reducer (`CODE/reducer/`) | tree under test | `pir.ReduceRequest` | reference requests only |

The driver and the reducer talk to the conductor over their stdin/stdout
pipes (`CODE/frame/`). Submission code cannot import `os`, so it cannot write
to that channel; `print`/`println` go to stderr, which nothing reads.

## Validity checks and where they are enforced

| Check | Enforced in |
| --- | --- |
| Patch touches only `quarter_pir/code/pir/` | `judge/submission_patch.py` `validate_patch` (config `allow_paths`), then `judge/evaluator.py` `check_protected`: every file outside the package byte-identical to the trusted tree, no symlinks or special files |
| Explicit import allowlist, and for the repository's trusted packages an explicit symbol list; `math/rand` limited to seeded generators; no package-level `var`, no `init`, no bodyless function, no directive except `//go:noinline` / `//go:noescape`, no build constraints, only `.go` files | `CODE/policy/policy.go` on the syntax tree of every file (evaluator builds the tool from its own copy) |
| Same allowlist on the compiled package: `Imports`, exact `GoFiles`, no assembly, cgo, ignored, embedded, test or syso files | `judge/evaluator.py` `check_source_policy` via `go list -json ./pir` |
| Fresh private database, coin and query seeds per experiment, same seeds for candidate and reference | `judge/evaluator.py` `fresh_seeds`, `evaluate` |
| Exactly one database stream; queries issued one at a time after preprocessing; one request per query | `CODE/conductor/conductor.go` (the conductor makes every server call itself) |
| Request privacy by reduction: client's request bytes equal `ReduceRequest(reference request)`; the reduction is what goes on the wire; reduction deterministic | `CODE/conductor/conductor.go` (`bytes.Equal(request, reduced)`, `rawrpc.Call(..., reduced)`), `CODE/reducer/reducer.go` (two calls compared), `CODE/refwire/refwire.go` `Generator` |
| Every retrieved value correct | `CODE/conductor/conductor.go` (`util.GenDBEntry(settings.DBSeed, x)`) |
| State for the planned query budget is retained after the export: additional untimed queries after the measured ones, identical for candidate and reference, each checked like a measured query and no message longer than the longest measured one | `CODE/conductor/probe.go` `planProbe`, `CODE/conductor/conductor.go` (probe phase), `judge/table2_metrics.py` `probe_failure`, `judge/evaluator.py` (category `coverage_probe`, code 11) |
| Submission code is never told the measured query count: `Params.OnlineQueries` is the planned budget in every process | `CODE/refwire/refwire.go` `NewPlannedParams`; `CODE/conductor/conductor.go` `spawn` passes size and threads only |
| Complete hints: canonical export copied inside the preprocessing clock, length exact, hash equal to the paired reference's | `CODE/ec24/ec24_client.go` (export before the `'p'` marker), `CODE/conductor/conductor.go` (length), `judge/table2_metrics.py` `validity_failure` (hash) |
| No work outside the clocks: every trusted call of submission code is followed, before its clock stops and before its result is used, by a wait until every goroutine the call started has exited; ten seconds later the run fails | `CODE/settle/settle.go` `Wait`; client `CODE/ec24/ec24_client.go` `settled` after `NewClient`, `Preprocess`, `ExportHints`, `Query`, `Reconstruct`; server `CODE/server/server.go` `TimedQuery` (live path and replay); reducer `CODE/reducer/reducer.go` `reduce` (both calls) |
| Client storage measured as the live heap after two forced collections, with no submission goroutine alive | `CODE/ec24/ec24_client.go` `liveBytes`, `settled` |
| Incidental limit: client process peak resident memory at most 4 GiB | `CODE/ec24/ec24_client.go` `peakResidentBytes`, `judge/table2_metrics.py` `validity_failure` |
| Reference transcript correct word for word (expanded-set oracle) and at the tight encoding | `CODE/server/replay.go` validate mode, `judge/evaluator.py` |
| Server replay: responses equal the client-verified ones, follow the resident database (perturbation guards before and after the measured passes), database and request unchanged, no goroutine left | `CODE/server/replay.go` |
| Server clock covers request decoding and response encoding | `CODE/server/server.go` `TimedQuery` |

## What the editable package can reach

The editable package contains no assembly and no bodyless declarations. The
only way from it to assembly is `CODE/safeaes/`, whose exported functions take
nothing but pointers to fixed-size arrays; the raw entry points are unexported
there. Everything else it may name is listed here with the reason it is safe.
Methods cannot be restricted by the selector check, so the methods of every
reachable type were audited too.

| Package | Names | Why safe |
| --- | --- | --- |
| `easypir/safeaes` | `RoundKeys`, `ExpandKey`, `MMO`, `Encrypt`, `Xor16` | arguments are `*[16]byte` / `*[44]uint32`; the assembly reads and writes exactly those sizes; no state |
| `easypir/refwire` | `Params` and its methods, `NoLabel`, `FailureProbLog2` | plain data and arithmetic |
| | `EncodeRequest`, `DecodeRequest`, `EncodeResponse`, `DecodeResponse` | pure Go with bounds-checked indexing; malformed input panics or returns `ok=false` |
| | `LongKey`, `NewLongKey`, `Eval`, `SubKey`, `SetOffset` | the expanded key is `*[44]uint32` by type, a nil pointer panics before any assembly; the 16-byte blocks are allocated inside `util` |
| | `Generator`, `NewGenerator`, `LastQuery`; methods `Query`, `DrawReplacements`, `BackupTag` | state is per value; its key field is a `*LongKey`; other fields are slices used with bounds-checked indexing; randomness only from the seeded source it was constructed with |
| `example.com/util` | `DBEntry`, `DBEntryLength`, `DBEntrySize`, `KeyByteLength`, `PrfKey`, `PrfKey128` | types and constants without methods |
| | `ZeroEntry`, `DBEntryXor`, `DBEntryXorFromRaw`, `DBEntryFromSlice`, `EntryIsEqual` | pure Go, bounds-checked |
| | `RandKey` | draws from the caller's seeded `*rand.Rand` |
| standard library | `bytes`, `encoding/binary`, `math`, `math/bits`, `sort`, `sync`, `sync/atomic`, and `math/rand` limited to `New`, `NewSource`, `Rand`, `Source`, `Source64` | memory-safe public API; no clock, file or network; the shared `math/rand` source is excluded |

Not reachable although they sit in imported packages: every slice-based PRF
and cipher helper of `util` (`PRFEvalWithLongKey*`, `GetLongKey`, `PRFEval*`,
`NewCipher`, `AesPrf`, `MembTest*`, `PRSet*`), `util.GenDBEntry`,
`util.DefaultHash`, `util.GenParams` and `refwire.NewParams` (they log).
The reference client and server handler use only the table above.
`util.NewCipher` allocated 4 of the 44 words its assembly writes; the trusted
copy now allocates 44 and `AesPrf.Encrypt` checks its block lengths. Neither
was used by the reference, whose fixed-seed transcript and hint export are
byte-identical before and after this change at both sizes.

## Gate

Conjunction over all workloads. A workload is one experiment, judged on all
five metrics (every workload measures every metric). The first four are
strict `<` with no allowance; client storage is non-inferiority.

| Metric | Decision per experiment | Rule | Clock or measure |
| --- | --- | --- | --- |
| `online_communication_bytes` | on every measured query | `<` | bytes of the request and response blobs the conductor carried |
| `server_compute_time_ns` | on every measured query, median of five replays of the implementation's own request | `<` | `TimedQuery` |
| `preprocessing_client_time_minutes` | once | `<` | conductor: coin seed sent until export complete |
| `end_to_end_online_time_ms` | once | `<` | conductor: sum of `Query`, server round trip, `Reconstruct` over the 1000 measured queries and the depth part of the probe (both runs answer both) |
| `client_storage_bytes` | before and in the volume phase | `<=` reference + 32,768 | live heap (below) |

The additional queries need no byte or server-time measurement of their own:
neither of their messages may be longer than the longest measured one, and the
handler keeps no state between calls (no package variables; the replay must
return exactly the verified responses). Client time and client storage can
move between queries, so both are measured on the additional queries too: the
depth part is clocked and compared paired, the volume part is clocked into the
score (the paired reference run skips it), and storage is read in the volume
phase (Client storage in the volume phase, below).

Amortized offline time is reported in private evidence but not gated: it is
the preprocessing numerator divided by a constant shared by both runs, so it
would always decide exactly as preprocessing time does. Each implementation
replays its own recorded requests for server time, because message formats may
differ.

The reference is counted at an information-tight encoding of its own messages
(`refwire.EncodeRequest`: 16-byte keys, offsets bit-packed at `log2(ChunkSize)`
bits; 16-byte parities), so bit-packing alone cannot pass:

| Entries | Request | Response | Total |
| --- | --- | --- | --- |
| 65,536 | 100 | 128 | 228 |
| 67,108,864 | 736 | 1,024 | 1,760 |

### Client storage

Communication can be bought with client storage (see Attainability), so
storage is gated: the candidate's value must not exceed the paired reference's.

The measure is the live Go heap after two complete collections, taken by the
trusted driver directly after the hint export, minus the same reading before
`NewClient`. At that point `settled` has shown that no submission goroutine
exists, so nothing can sit on another stack, and without package variables,
cgo or `unsafe` the heap is the only place client state can live.

Run-to-run spread of the reference's own value, measured on an idle host:

| Entries | Runs | Minimum | Maximum | Spread |
| --- | --- | --- | --- | --- |
| 65,536 | 14 (6 fresh seeds, 8 fixed coin seed) | 2,144,776 | 2,164,120 | 19,344 |
| 67,108,864 | 6 (3 fresh seeds, 3 fixed coin seed) | 85,261,808 | 85,272,040 | 10,232 |

The spread does not depend on whether seeds are fixed; it comes from runtime
objects whose number depends on scheduling. The gate allows exactly 32,768
bytes, the next power of two above the larger spread. One dropped superblock
key costs 442,368 bytes of parities at the small size and 17,825,792 at the
large size, so the bound is not room for state.

A canonical measure (the client serializes the state it needs for the
remaining query budget, and a fresh process restores from those bytes) would
be exactly deterministic and independent of the runtime. It was not adopted:
it needs a save/restore interface and a process restart inside the experiment,
restore time would have to be charged to a clock, and the reference's size
would become whatever encoding the task author wrote instead of what the
original code allocates.

### Client storage in the volume phase

The readings above stop at the end of the depth part of the probe, about
1,700 queries into an experiment, while the volume part continues to the
planned budget (147,634 queries at 67,108,864 entries). State that grows with
every query (query history, promoted-hint tables, per-query caches) was
therefore never read past that point. The conductor now also reads the client
after one volume query drawn from the private probe seed and after the
second-to-last volume query, while one query is still to come
(`client_storage_volume_bytes`, the larger of the two).

The paired reference run skips the volume phase: the reference spends about
4 ms per query in its own client on top of the trusted generator's time, and
its volume phase took 1,934 s per experiment here, which the 6,000 s deadline
(graded evaluation 4,258 s with the phase skipped) does not hold twice. Its storage only grows (the generator's query history and the client's
per-query cache `localCache`; every other table is updated in place), so its
volume reading is its last reading before the phase plus the reference
client's growth over the phase, `REFERENCE_VOLUME_GROWTH_BYTES`, measured by
running the reference through the whole volume phase:

| Entries | Runs | Growth: minimum | maximum | spread | Constant |
| --- | --- | --- | --- | --- | --- |
| 65,536 | 4 | 80,816 | 85,872 | 5,056 | 83,328 (mean) |
| 67,108,864 | 4 | 9,460,464 | 9,463,376 | 2,912 | 9,461,920 (mean of the first two) |

The storage comparison holds at both points: the candidate's readings before
the volume phase against the reference's, as before, and its volume reading
against the reference's volume reading, each with the same 32,768-byte
jitter. Client storage as reported is the larger of the candidate's two values.

The probe's query plan is unchanged. A client that limits the state it keeps
for some hints is caught when a probe query reaches those hints; the depth
part walks down from the deepest first containing hints of the scanned
chunks, the allotment runs empty four chunks, and the volume part consumes
hints up to the planned budget. Hints beyond that reach cannot be tested by
any query sequence: the first containing hint of an index is geometric with
mean `ChunkSize`, so at 67,108,864 entries the deepest one over the whole
database lies near 16,384 x ln(2^26), about 295,000, and the expected number
of indices whose first containing hint is a fresh hint at 600,000 or beyond
is about 2^26 x e^-36.6, below 1e-8 per experiment. The statement makes such
limits invalid regardless.

### Where the reference's storage goes

Retained bytes of the reference after preprocessing, by structure, against
what the scheme needs:

| Structure | Reference | Needed | Why the difference |
| --- | --- | --- | --- |
| Primary hints | 56 B each | 16 B each | tag is the index until a hint is refreshed; block parity and programmed point exist only for refreshed hints (at most one per query) |
| Backup hints | 56 B each | 48 B each | tag follows from the position |
| Replacements | 24 B each, plus an 8 B duplicate of the index in the generator | 18 B each | the index is a chunk offset below 2^16 |
| Positive labels | 8 B per primary hint | sparse | only refreshed hints have one |
| Tags (generator copy) | 4 B per primary hint | 4 B | kept |

| Entries | Reference measured | Compact layout (computed) | Freed | One dropped superblock costs | Superblocks payable |
| --- | --- | --- | --- | --- | --- |
| 65,536 | 2.15 MB | 0.95 MB | 1.2 MB | 0.44 MB | 2 of 4 |
| 67,108,864 | 85.3 MB | 42.6 MB | 42.7 MB | 17.8 MB | 2 of 32 |

So the conjunction stays attainable: a tighter layout pays for dropping at
least one superblock's keys at each size. The fixture
`compact_dropped_superblock_keys` does exactly this (one superblock at the
small size, two at the large size) and passed the whole gate in a local
development evaluation (Local results).

## Continuous score

Computed from the candidate's own measurements only; no reference measurement
of the run enters. Each value `v` contributes `c/(c+v)` with a fixed scale `c`
that depends only on the database size, so the score is monotone and never
saturates.

```text
score = 100 * exp( 1/2 * mean_w log(cB/(cB+B_w))    online bytes of query w
                 + 1/8 * mean_w log(cS/(cS+S_w))    server ns of query w
                 + 1/8 * mean_e log(cP/(cP+P_e))    preprocessing ns of experiment e
                 + 1/8 * mean_e log(cO/(cO+O_e))    online ns per query of experiment e
                 + 1/8 * mean_e log(cM/(cM+M_e)) )  client storage bytes of experiment e
```

`cB` = 228 or 1,760 (the reference's tight byte count), `cS = 50 ns * SetSize`,
`cP = 5 ns * SetSize * (primary + backup hints)`,
`cO = 50 ns * (ChunkSize + SetSize)`, `cM` = the canonical hint export length
(47,054,848 bytes at 67,108,864 entries) (`judge/table2_metrics.py`
`score_scales`, `continuous_score`). `O_e` is the online time per query over
the uniformly random queries the run answered: the measured ones and the
volume part of the probe (`score_online`). The depth part is chosen
adversarially and is compared paired in the gate instead. `M_e` is the peak
client storage. Communication carries half of the exponent and is the largest
single term; every term moves the same way as its gate comparison, so no gated
metric can be traded away for score without losing that term.

Client storage entered the score when every workload came to measure every metric, so that the gate and the score use
the same metric set. It prices the trade the task is about (fewer bytes for
more client state) without dominating: at 67,108,864 entries 71 MB against
85 MB of storage moves the score by about 1/8 x ln(132/118), 1.4 % of its
value. The reference row of the score (reported only) has no volume part; its
`O_e` is over the measured queries and its `M_e` uses the reference volume
storage of Client storage in the volume phase. `reference_score` and `margin`
therefore cover different query sets: volume queries cost more than measured
ones (more of them hit refreshed hints), and the reference itself run as a
candidate scored 26.2 against a `reference_score` of 28.1. Nothing is decided
on `margin`.

One limit of any such formula here: the byte count can only move by a few
percent under the storage gate, while times can move by integer factors, so in
absolute points a large speed-up still outweighs a byte reduction. The weights
are four constants in one place if a different balance is wanted.

## Why the reduction check gives privacy, and what it does not cover

The reference's requests do not depend on database contents: they are a
deterministic function of the coin seed and the query sequence (measured
before the redesign, and now structural: `refwire.Generator` has no database
input). The conductor computes them itself. `ReduceRequest` runs in a process
that receives only those bytes. Therefore the sequence of candidate requests
is a function of the sequence of reference requests, and any distinguisher
against the candidate's server view is one against the reference's.

- Per-query function versus joint view: a stateless per-request function gives
  a function of the joint view. Even a stateful one would, because its state
  could only come from earlier reference requests; the source policy makes it
  stateless anyway.
- Request count and order are fixed by the conductor. Message lengths can
  depend only on the reference request.
- A panic in `ReduceRequest` or a nondeterministic result invalidates the run.
- Not covered: timing. A client can still make the arrival time of a request
  depend on the index (the reference's hint search already takes
  index-dependent time). The benchmark server does not read time, so there is
  no reward in it, but it is outside the check.
- Not covered: inputs that were not evaluated. This is an equivalence test on
  the evaluated seeds, not a proof.

## Anti-skip: options considered

Wire checks alone do not force the client to compute its hints, because no
request depends on the database. Three mechanisms were considered.

1. **Capacity test** (issue the full planned query budget, or bursts aimed at
   one chunk). Not sufficient as the anti-skip mechanism, but now used in
   addition to the export, to check retention (see Coverage probe). Hole: the scheme's safety margin is deliberate. The first
   matching primary hint is geometrically distributed with mean `ChunkSize`,
   while there are about 40 x `ChunkSize` primary hints; no query sequence the
   harness can find reaches the late ones, so half the primary work can be
   skipped undetected. Bursts catch under-provisioned backup
   groups only with the probability that the cheater skipped that group. The
   full budget is 147,634 queries at the large size, far beyond the time limit.
2. **Spot audit plus storage cap** (ask for the parity of random hints; forbid
   keeping the database). Hole: at 65,536 entries the hints are larger than the
   database, so no cap can stop a client from keeping the database and
   computing audited hints on demand.
3. **Complete canonical export inside the preprocessing clock** (implemented).
   The export is defined on the reference's hint sets, not on the client's
   storage layout, so a client that stores more or different state derives it.
   It forces all hint work into the clock at every size and is compatible with
   redesigned responses (the `dropped_superblock_keys` fixture passes it).
   Hole: it obliges every design to be able to produce the reference's full-set
   parities; a design whose hints do not determine them would be rejected even
   if it retrieved correctly. No such design is known, and the sets themselves
   are already pinned by the reduction check.

The export after the last query was dropped: online hint work is negligible,
so it added constraint without closing a skip.

## Coverage probe: retention after the export

The export proves that the hints were computed; it does not prove that they
were kept. Storage is read directly after the export, and 1,000 uniformly
random queries use only a sliver of the state: at 67,108,864 entries a chunk
sees 0.24 queries on average against an allotment of 108 backup hints and
replacements, and the first containing hint of a random index is beyond
`7 x ChunkSize` with probability 0.1 %. A client could therefore export
everything, discard most of it before the storage reading and spend the
saving on fewer request keys. It was also told the measured query count
(`Params.OnlineQueries` was 1,000), so it could size the remainder exactly.

Two changes close this.

1. `Params.OnlineQueries` is the planned budget in every process
   (`refwire.NewPlannedParams`); the client and reducer drivers no longer take
   a query count. The trusted generator's history is no longer sized by that
   field: it starts with the capacity it always had (1,000 entries) and grows,
   so the reference's storage reading is unchanged.
2. After the measured queries the conductor issues additional untimed queries
   (`conductor/probe.go`). Clocks, byte counts and the transcript are complete
   before the first of them. Each is checked like a measured query, and
   neither of its messages may be longer than the longest measured message in
   its direction. The paired reference run is asked the identical sequence
   (same seeds, plan digest compared); the candidate fails when it fails a
   query the reference answered (`table2_metrics.probe_failure`).

What the reference serves per chunk. Every query consumes one backup hint and
one replacement of its own chunk (`refwire.Generator.Query`: `ReplConsumed`
and `BackupConsumed` both advance by one). With no replacement left the
generator sends a dummy offset and the reference client XORs a zero
replacement value, so the answer is wrong; with no backup hint left the
generator panics. The reference therefore serves exactly `BackupPerGroup`
queries per chunk (66 and 108 at the two sizes) and not one more. The
allotment is `refwire.Params.BackupPerGroup`: `3 x floor(Q/SetSize)` for the
planned budget `Q`, rounded up to a multiple of the thread count. This was
established from the code and by running the reference to the full allotment;
the paper's text was not consulted for it.

How the queries are chosen. The plan is a function of the generator's
bookkeeping after the measured queries and of a private probe seed; it does
not depend on anything the client says.

- The conductor computes, for up to 1,024 randomly chosen chunks, the first
  containing hint of every index under the reference's rule (index order, a
  hint whose promoted point lies in the chunk is skipped), and keeps these
  tables exact while it plans.
- Primary hints: 256 hint indices are drawn uniformly from all primary hints;
  for each, the query is a scanned index whose first containing hint is the
  nearest one at or below the drawn index.
- Backup hints and replacements: four of the scanned chunks are queried until
  their whole allotment is consumed, deepest first containing hints first.
  The last query of such a run uses slot `BackupPerGroup - 1`; nothing beyond
  the allotment is asked, and no other probe query may exhaust a chunk.

`conductor/probe_test.go` checks the plan against an independent
implementation of the selection rule on `refwire` alone: every planned hit is
the reference's, the generator serves every query, each run ends exactly at
the allotment, and the plan is deterministic in the seed.

Limit, and it is a real one. Only hints that are the first containing hint of
some index can be asked for. That depth is about `ChunkSize x ln(indices
scanned)`: 10.7 `ChunkSize` at 65,536 entries (all chunks scanned) and 17.2
`ChunkSize` in the measured run at 67,108,864 entries, of 37 and 41
`ChunkSize` primary hints. The reference keeps the rest for a 2^-40 failure
probability, which no query sequence of feasible length observes (making a
deep hint the first containing one means first consuming every shallower hint
that contains the index, and those are themselves deep). A client that drops
the primary hints beyond about `ChunkSize x (ln N + 2)` passes the probe and
serves the planned budget with a failure probability near 10^-3 instead of
2^-40. Uniformly drawn hint indices, as first specified, cannot do better:
most drawn indices have no index that selects them. Closing this needs a
different kind of check (for example a second export of retained state, or a
canonical storage measure, see Client storage); it was not improvised here.

Measured on this host (fixed seeds, four pinned CPUs, not idle): the probe is
492 queries and 0.6 s at 65,536 entries, 688 queries and 33 s for the
reference at 67,108,864 entries (3.4 s planning, the rest is the reference's
own linear hint search in the conductor's generator and in the client), 21 s
for a fast client. The reference's measured quantities are unchanged: same
hint export and transcript digests, storage 85,262,160 against 85,265,360
before, online 4.03 s against 4.02 s.

## Primary-hint audit (prototype)

Closes the tail the coverage probe cannot reach. After the probe the conductor
draws 512 primary hints that no query has consumed (tag still equal to the
index, no label), uniformly, from the probe seed, and sends them to the client
driver; the driver calls `PrimaryHint(j)` for each, outside every clock, and
compares the 20-byte record (tag, set parity) with the hint export it already
holds. The evaluator pairs the result with the reference run on the same
sample (`table2_metrics.audit_failure`, category `hint_audit`, code 12).

Expected values. An unconsumed primary hint's canonical record is the record
of the in-clock export, whose hash is already required to equal the paired
reference's. No recomputation and no database access is needed, and the
comparison is therefore against the reference's values by construction.
Consumed hints and backup hints are never asked for, so nothing is imposed on
how promoted or backup hints are kept. The comparison happens in the client
process, like the storage reading, and shares its trust assumption.

What it constrains. A client must keep state from which the plain set parity
of every unconsumed primary hint follows. An honest client could instead keep
an invertible re-encoding that answers queries but not the audit: for example
only the weighted sum of per-superblock parities with distinct nonzero
GF(2^8) coefficients, with a server that returns matching weighted sums. That
has the same size as the plain parity and no known advantage, so the audit
excludes it at no cost in storage; it is nevertheless a constraint on the form
of client state that the coverage probe does not impose.

Measured: the audit takes 0.3 ms at 65,536 entries and 1.2 to 1.7 ms at
67,108,864 entries. `discard_primary_tail_after_export` (keeps 20 `ChunkSize`
primary hints, passes every query) is rejected with 242 of 512 records wrong.

## Attainability of the communication gate

Strictly fewer bytes are attainable inside the rules. The fixture
`judge/fixtures/positive/dropped_superblock_keys.patch` keeps, per hint, the
parity of each of the last `d` superblocks; `ReduceRequest` drops those `d`
keys; the server returns the offsets parities and only the remaining key
parities. Each dropped superblock saves 32 bytes per query and costs 16 bytes
per hint of client storage. As it stands that fixture now fails the storage
comparison; `compact_dropped_superblock_keys` pays for the same change with
the bytes a tighter hint layout frees, and stays within the reference's
parities. It also reduces server work. Communication reduction in this family
is paid for in client storage, not in server or online time; with storage
gated, the payment has to come out of the reference's representation slack
(previous section).

No lower bound that would make strict improvement impossible was found. The
offsets and the offsets parities look irreducible for a request that must be a
function of the reference request (the server cannot know which superblock or
chunk matters), but that is an argument, not a proof.

## What is not airtight

- Go data races can break memory safety without any forbidden import. Inside
  the client process that could falsify the storage measurement (taken by the
  driver in the same process), which is now a gated metric. It cannot touch the clocks, the query stream,
  the oracle or the request check, which live in other processes.
- The server replay's "follows the resident database" guard assumes that a
  correct response depends on every entry addressed by the explicit offsets.
- The goroutine rule forbids persistent worker pools; that is a restriction on
  honest designs, accepted to keep background work on the clock. The boundary
  costs about 6 ns per call when nothing is outstanding (measured: 5.6 ns for
  the count and the wait; four started-and-joined workers took 958 ns without
  and 960 ns with it).
- The source policy is an allowlist over the standard library as shipped with
  Go 1.19.13 plus explicit symbol lists for the repository's packages. It was
  reviewed by hand, including the methods of reachable types, which the
  selector check cannot restrict; it is not a formal proof.
- The completion boundary in the live server counts all goroutines of the
  process, including gRPC's. The conductor issues one call at a time on one
  connection, and no interference was observed, but a transport goroutine
  started during a call would be waited for like a handler's.
- At 65,536 entries most of the online time is harness transport, so the
  solver controls a minority of that metric there.
- The coverage probe cannot reach primary hints that no index selects first
  (see Coverage probe); the primary-hint audit covers them.
- The probe's selection logic is in the conductor source, which the solver
  sees; only its seed is private. It does not rely on being unknown.
- The probe's message-length rule compares with the longest measured message.
  A protocol whose message length varies with the request content could
  exceed that on an additional query by chance; no such design is known.

## Local results

All 43 fixtures behaved as declared in `judge/fixtures/expected.json`
when run through the real evaluator in the locally built verifier image
(`judge/fixtures/run_fixtures.py`, development role, both sizes). Runs were
strictly sequential on four pinned CPUs; host load is recorded in the evidence
file.

| Adversarial fixture | Rejected as (code) |
| --- | --- |
| `background_goroutine` | client (4) |
| `bodyless_function` | source_policy (2) |
| `build_tag_file` | source_policy (2) |
| `crypto_aes_import` | source_policy (2) |
| `deferred_work` | hint_export (5) |
| `extra_rpc` | source_policy (2) |
| `fake_metric_record_stdout` | source_policy (2) |
| `import_alias` | source_policy (2) |
| `import_blank_net` | source_policy (2) |
| `import_dot` | source_policy (2) |
| `import_second_file` | source_policy (2) |
| `init_function` | source_policy (2) |
| `keep_db_skip_hints` | hint_export (5) |
| `leak_query_index` | request_reduction (6) |
| `linkname_directive` | source_policy (2) |
| `math_rand_global` | source_policy (2) |
| `new_assembly_file` | source_policy (2) |
| `occasional_wrong_value` | wrong_value (7) |
| `outside_editable_package` | patch (1) |
| `package_variable` | source_policy (2) |
| `process_memory_guard` | process_memory (8) |
| `raw_asm_function_value` | build (3) |
| `raw_asm_symbol_second_file` | build (3) |
| `reducer_lingering_goroutine` | request_reduction (6) |
| `refwire_short_long_key` | build (3) |
| `safeaes_unexported_symbol` | source_policy (2) |
| `seed_special_casing` | source_policy (2) |
| `server_lingering_goroutine` | wrong_value (7) |
| `server_memoization` | source_policy (2) |
| `server_omits_auxiliary_parities` | wrong_value (7) |
| `transcript_replay` | request_reduction (6) |
| `under_provisioned_hints` | hint_export (5) |
| `util_cipher_import` | source_policy (2) |
| `util_long_key_prf_import` | source_policy (2) |

Gate columns are passed/total comparisons in a development evaluation
(2000 queries, 2 experiments).

| Accepted fixture | Verdict | Score | Bytes | Server | Preproc. | Online | Storage | Bytes/query small / large | Storage MB small / large | Transcript identical |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `allocation_free_aes` | valid_identical_transcript | 33.5 | 0/2000 | 2000/2000 | 2/2 | 2/2 | 2/2 | 228 / 1760 | 2.14 / 85.2 | yes |
| `batched_folded_preprocessing` | valid_identical_transcript | 35.1 | 0/2000 | 1995/2000 | 2/2 | 2/2 | 2/2 | 228 / 1760 | 2.13 / 85.2 | yes |
| `compact_dropped_superblock_keys` | valid_beating | 37.7 | 2000/2000 | 2000/2000 | 2/2 | 2/2 | 2/2 | 196 / 1696 | 1.43 / 78.2 | no |
| `dropped_superblock_keys` | valid_fewer_bytes_over_storage | 38.2 | 2000/2000 | 2000/2000 | 2/2 | 2/2 | 0/2 | 196 / 1632 | 2.58 / 156.5 | no |
| `fake_metric_record_builtin_print` | valid_identical_transcript | 28.3 | 0/2000 | 526/2000 | 1/2 | 1/2 | 2/2 | 228 / 1760 | 2.14 / 85.3 | yes |
| `reference_unchanged` | valid_identical_transcript | 28.2 | 0/2000 | 1258/2000 | 1/2 | 0/2 | 2/2 | 228 / 1760 | 2.14 / 85.3 | yes |
| `server_joined_parallel_workers` | valid_identical_transcript | 28.6 | 0/2000 | 1000/2000 | 1/2 | 2/2 | 2/2 | 228 / 1760 | 2.15 / 85.3 | yes |
| `server_returns_before_workers` | valid_identical_transcript | 26.3 | 0/2000 | 5/2000 | 1/2 | 0/2 | 2/2 | 228 / 1760 | 2.15 / 85.3 | yes |
| `storage_for_time` | valid_over_storage | 25.0 | 0/2000 | 1489/2000 | 1/2 | 1/2 | 0/2 | 228 / 1760 | 1075.88 / 1159.0 | yes |

The verifier-entrypoint timings (412 s train, 820 s eval) were last measured on
identity `eb8c18da…` and not rerun for this one; development evaluations in
this fixture run took the wall times in the evidence file.

### Coverage-probe change

Run natively with Go 1.19.13 on four pinned CPUs of a shared host, through
`judge/evaluator.py` in the development role (both sizes, fresh seeds). The
complete fixture set was not rerun for this change; the table above predates
it and now lacks the two `discard_*_after_export` fixtures.

| Patch | Outcome |
| --- | --- |
| `reference.patch` | valid, score 28.0, probe 484/484 and 687/687 answered, 500 s |
| `negative/discard_deep_backups_after_export` | coverage_probe (11), wrong value at probe query 276 of 488 (65,536 entries) |
| `negative/discard_primary_tail_after_export` | coverage_probe (11), wrong value at probe query 0 of 486 (65,536 entries) |
| `positive/compact_dropped_superblock_keys` | valid_beating, storage 1.44 / 78.2 MB, probe answered at both sizes |

`go test ./conductor ./safeaes` passes. The four accepted fixtures that carry
their own copy of the generator were changed with it (history grown instead of
sized by `OnlineQueries`); only `compact_dropped_superblock_keys` was rerun.

## Publication status

The coverage-probe change is not imported: `upstream.lock.json` was refreshed
with `scripts/prepare_quarter_pir_full_compute_export.py --in-place`, but
`config/task-catalog.json` and `config/task-definitions.json` still name
source identity `0e372a1c…`; the new identity is `d9cd8ced…`. The importer has
to be run and the three images built and published before this state is
consistent. The paragraph below describes the previous import.


Imported with `scripts/prepare_quarter_pir_full_compute_export.py` and
`scripts/import_task.py --replace --no-reference` in local build mode. Nothing
was pushed. The catalog entry carries placeholder digests until the three
images are published and the import is completed with a published-images
manifest; see the evidence file for the source identity and image names. The
dataset image must be published under its source-bound tag, not `v1`.
