# Testbed operations

Read `/frontier/task/README.md` for project payload creation and tar packaging;
read `/frontier/task/repo/README.md` for the mathematics and C# interface.

An accepted `frontier-submit train /workspace/train.tar` consumes one train call,
including failed verification. Use its absolute score, mean gap, aggregate gap
comparison counts and authoritative remaining-call count to guide development.
No reference measurements or case-level decisions are disclosed.

`frontier-submit store-final /workspace/final.tar` keeps one replaceable selected
archive. A successful stored receipt permits continued work; a failed update
preserves the previous selection. Saving has no call-count quota and does not
start evaluation. After feedback, decide whether to rebuild and replace the
selection. A local file change alone never updates it.

On normal exit, answering timeout or an Agent process crash, the Driver collects
the saved selection unless an explicit final was accepted. User cancellation and
platform failures are excluded. Without either selected archive, exiting or
timing out does not produce a final score. The last train is never promoted.

`frontier-submit final /workspace/final.tar` explicitly locks that archive and
has priority over the saved selection. After acceptance, make no further
platform calls and exit. Final starts independently after Agent cleanup.

| Resource | Formal task limit |
| --- | --- |
| Answering | 18,000 seconds (5 hours) |
| Independent final | 3,600 seconds (1 hour) |
| Accepted train calls | 100 |
| Cumulative train evaluation | 18,000 seconds within answering |
| CPU architecture | x64; no specific CPU model guaranteed |
| CPU / RAM / workspace | 4 CPUs / 8 GiB / 8 GiB |
| Project build | 240 seconds |
| Candidate instance | 30 seconds; original watchdog includes 3 seconds of grace |
| Supplied oracle | 256 successful calls per instance |
| Model sessions | 1024 |
| Model request / response | 1 MiB / 8 MiB |
| Total audit bundle | 14.75 GiB |

Build time, local work and train waits consume answering time. Final remains
independently timed. Train counts are distinct from model requests and local
tool calls. There is no GPU or SSH resource pool for this CPU task.
