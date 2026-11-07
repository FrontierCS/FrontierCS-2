# Private Set List Decoding

Send strictly fewer bytes per online query than the provided QuarterPIR implementation, while
also being strictly faster in server compute, client preprocessing and end-to-end online time and
using no more client storage.

## Task

QuarterPIR ("Efficient Pre-processing PIR Without Public-Key Cryptography", EUROCRYPT 2024) is a
preprocessing private-information-retrieval scheme: a client streams the whole database once and
keeps a compact table of hints; afterwards it retrieves single entries with one short request
each, without the server learning which entry it wanted, and refreshes the consumed hint. The
repository is the complete provided implementation (the reference) except the body of the server
request handler, which you write. You may rewrite all client and server computation and the
message formats, under two rules:

- **The request is a deterministic public function of the reference request.** You supply that
  function (`ReduceRequest`); the server receives only its output.
- **The response is free.** It is computed from the database and the request alone; it only has
  to let your client retrieve the right entry.

## What you may change

- Only the package directory `quarter_pir/code/pir/`: edit its files or add `.go` files. Everything
  else is trusted and must stay byte-identical (the trusted side holds the seeds, issues the
  queries, carries every message, runs every clock and checks every value).
- `pir/` may import only `bytes`, `encoding/binary`, `math`, `math/bits`, `math/rand`, `sort`,
  `sync`, `sync/atomic`, `easypir/safeaes`, `easypir/refwire` and `example.com/util`, and from
  the last four only the names listed in `quarter_pir/code/policy/policy.go`. No dot or blank imports,
  package-level variables, `init` functions, bodyless functions, cgo, assembly or non-Go files,
  test files, subdirectories, build constraints, or compiler directives other than
  `//go:noinline` and `//go:noescape`. A violation is rejected as `content_policy`; a change
  outside `pir/` as `out_of_scope`.

## Interface

```go
func NewClient(p refwire.Params, coinSeed int64) *Client
func (c *Client) Preprocess(next func() []uint64)
func (c *Client) ExportHints(emit func([]byte))
func (c *Client) Query(x uint64) []byte
func (c *Client) Reconstruct(response []byte) util.DBEntry
func (c *Client) PrimaryHint(j uint64) (uint32, util.DBEntry)

func ReduceRequest(p refwire.Params, referenceRequest []byte) []byte

func HandleQuery(db []uint64, p refwire.Params, request []byte) []byte
```

`NewClient` may return any type with these methods.

1. **Preprocessing.** The client calls `NewClient`, then `Preprocess`; `next` returns the database chunks in order (`ChunkSize` entries, two `uint64` words each) and `nil` after the last; it stops working when `Preprocess` returns. Directly afterwards the harness calls `ExportHints` once.
2. **Online queries.** The harness hands the client distinct indices `x`, one at a time. The first 1,000 are measured; more follow, and the client is not told where the measured ones end. `Query(x)` returns the request bytes, the server answers with `HandleQuery(db, p, request)` (`db` is read-only), and `Reconstruct(response)` must return entry `x` and leave the client ready for the next query.
3. **Audit.** Between two queries after the measured ones, the harness may call `PrimaryHint(j)` (see Correctness).

`refwire.Params` carries `DBSize`, `ChunkSize`, `SetSize`, `SBSize`, `SBSizeLog2`, `SBNum`, `Threads` (4) and `OnlineQueries`. `OnlineQueries` is the planned query budget of one preprocessing, `floor(sqrt(N)*ln(N))` = 147,634. `util.DBEntry` is `[2]uint64`.

`quarter_pir/code/refwire/refwire.go` defines exactly what the reference client sends; `quarter_pir/code/pir/client.go` is the reference client. The database is split into `SetSize` chunks of `ChunkSize` entries, grouped into `SBNum` superblocks of `SBSize` chunks; a hint with tag `t` holds the entry at offset `refwire.SetOffset(longKey, t, i, p)` of every chunk `i`. A request is `SBNum` 16-byte superblock keys followed by `SBSize` explicit chunk offsets. Key `k` defines the offset `refwire.Eval(refwire.NewLongKey(&k), j) & (ChunkSize-1)` in the `j`-th chunk of its superblock.

The reference response is two arrays of `SBNum` entries: entry `c` of the first is the parity of the set that uses the explicit offsets in superblock `c` and the key-defined offsets everywhere else; entry `b` of the second is the parity of the key-defined entries of superblock `b` alone.

Reference messages are counted at their information-tight encoding (`refwire.EncodeRequest`, `refwire.EncodeResponse`): 736 request plus 1,024 response bytes, 1,760 in total.

## Correctness

Any failed check in any experiment scores zero (seeds are fresh and private):

- **Request reduction:** `ReduceRequest` on the reference request, run twice in an isolated
  process, gives the same bytes, and `Client.Query` returns exactly them; so your client must
  make the reference's choices, including its hint and the coin stream of
  `math/rand.NewSource(coinSeed)`.
- **Retrieval:** every `Reconstruct` result is the database entry.
- **Preprocessing:** `ExportHints` emits the reference's bytes: per primary hint `j` in order, tag
  `j` (uint32) and set parity; per backup hint `(g,k)`, tag `LocalSetNum + g*BackupPerGroup + k`,
  set parity, block parity and programmed-point value; per replacement `(g,k)`, database index
  (uint64) and entry. Little-endian; a parity or entry is two `uint64` words.
- **Planned budget:** serve any sequence of distinct queries up to the planned budget that the
  reference serves, issued or not. The additional queries (first chosen by the harness, possibly
  adversarially, within what the provided implementation answers, then uniformly random ones, up
  to 147,634 in all) are clocked and checked like measured
  ones, and neither message may exceed your longest measured one in that direction.
- **Audit:** `PrimaryHint(j)` returns the canonical (tag, set parity) of any unconsumed primary hint;
  nothing is asked about backup hints or about hints a query has consumed.
- **Memory:** the client process stays below 4 GiB resident.
- **Server:** on replay `HandleQuery` returns exactly the verified response, leaves its inputs
  unchanged and follows the database.
- **Goroutines** settle before any entry point (client, server, `ReduceRequest`) returns; running
  ones stay on its clock, and one alive ten seconds later fails the run.
- **Limits:** each build step 600 s; each experiment run 3,000 s; all within the verification
  time limit.

## Workload

Graded databases have 67,108,864 entries of 16 bytes (`frontier-run.sh` defaults to 65,536 for
quick local checks). An experiment runs the full preprocessing, 1,000 measured queries at
uniformly random distinct indices, and the additional queries, once with your package and once
with the reference on identical seeds.

| suite | experiments |
|---|---|
| train | 1 |
| final | 2 |

Builds use Go 1.19.13, `-trimpath -mod=readonly -buildvcs=false`, `CGO_ENABLED=0`,
`GOMAXPROCS=4`, `GOGC=100`. Measured:

- **Online communication bytes** (per query): request plus response bytes carried by the transport.
- **Server compute time** (per query): from `HandleQuery` receiving the request until the response exists and its goroutines have exited; the median of five replays of your recorded request.
- **Preprocessing client time** (per experiment): `NewClient`, `Preprocess` including receiving the stream, and `ExportHints`.
- **End-to-end online time** (per experiment, reported per query): `Query`, server round trip and `Reconstruct`, summed over the measured queries. The additional queries are clocked the same way.
- **Client storage** (per experiment): The peak live state of your client over the online phase, as live Go heap; garbage does not count. State released before a reading and rebuilt afterwards does count, and so do caches. It must be the state of a client that serves the planned budget and the audit.

## Scoring

The score (0 to 100) uses only your own measurements:

```text
score = 100 * exp( 1/2 * mean_w log(cB/(cB+B_w))      online bytes of query w
                 + 1/8 * mean_w log(cS/(cS+S_w))      server compute ns of query w
                 + 1/8 * mean_e log(cP/(cP+P_e))      preprocessing ns of experiment e
                 + 1/8 * mean_e log(cO/(cO+O_e))      online ns per query of experiment e
                 + 1/8 * mean_e log(cM/(cM+M_e)) )    client storage bytes of experiment e
```

with `cB = 1,760`, `cS = 50 ns * SetSize`, `cP = 5 ns * SetSize * (LocalSetNum + SetSize*BackupPerGroup)`, `cO = 50 ns * (ChunkSize + SetSize)`, `cM = 47,054,848` bytes (the length of the canonical hint export). `O_e` is taken over the measured queries and the additional uniformly random queries of experiment `e`.

Anchors: the starter (no `HandleQuery` body) scores 0; the provided implementation scored about 28
on the authors' development host (the value depends on the hardware).

**Beats the reference** when, in every experiment: online communication bytes and server compute
time are strictly lower than the reference's on every measured query; preprocessing client time
and end-to-end online time are strictly lower; and client storage is at no point of the online
phase above the reference's at that point plus 32,768 bytes of measurement jitter. Equality
fails.

## Running locally

`./frontier-build.sh` builds the tree (server, client driver, reducer, conductor, policy) and
checks the source policy against the pristine tree; `./frontier-run.sh [entries]` runs one complete
local experiment with fresh seeds and prints its result record (no reference comparison).

## Environment

Go 1.19.13 with an offline module cache. The Verifier runs both implementations and all helpers
on 4 vCPUs, 8,192 MiB RAM and 8,192 MiB storage; no GPU.
