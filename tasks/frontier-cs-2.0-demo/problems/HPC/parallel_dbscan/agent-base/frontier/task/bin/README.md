# Platform guide

Use `/frontier/task/bin/prepare_workspace.sh` once to create the writable Git
workspace `/workspace/parallel_dbscan`. Keep the original commit as your patch
base. The research statement is `/frontier/task/PROBLEM.md`.

After implementing the reconstruction region, create a patch and its tar:

```sh
/frontier/task/bin/make_submission.sh /workspace/parallel_dbscan /workspace/submission.tar
frontier-submit train /workspace/submission.tar
frontier-submit store-final /workspace/submission.tar
```

The task helper first writes `/workspace/submission.patch`, then an uncompressed
deterministic POSIX tar. Train, store-final and final use the same archive layout:
exactly one regular file `payload/solution.patch`; an optional `payload/` directory
entry is permitted. No links, duplicate paths or additional files are accepted.
The patch is nonempty source text, at most 262144 bytes, changes at most 20 files,
and uses the research statement's editable surface. The complete tar is at most
286720 bytes (patch plus tar headers/padding). No executable bit is required on
the patch. The generic Candidate CLI owns authentication, hashing, upload and
receipts; task helpers only prepare source and package the payload.

Completed train feedback includes score, the approved aggregate candidate elapsed-time median and
p90 for valid submissions, and authoritative remaining calls. Hidden reference
rows, workload identities and final metrics are not returned. Invalid source
submissions score zero. Review feedback and decide whether your selected answer
should be updated; there is no fixed submission schedule.

`store-final` stores a Driver-owned snapshot without training, ending answering
or locking final. A successful replacement supersedes the old snapshot; a failed
replacement preserves it. Editing the local archive does not change the stored
copy. An accepted explicit final takes priority and is irreversible:

```sh
frontier-submit final /workspace/submission.tar
```

Invoke explicit final once and exit. If no explicit final is accepted, normal
exit, answering timeout or a trusted process crash selects a successfully stored
copy. Cancellation and platform failure do not. Neither the last train payload
nor its score is implicitly final. Driver stops Agent/train resources and checks
cleanup before independently evaluating the selected archive. No selected archive
means non-submission. The confirmed protocol has no save-count quota; deployed
CLI/Driver compatibility must be verified for the actual Run.

Formal answering lasts 18000 seconds; independent final has a separate 600-second
budget, including provisioning. Train is limited to 100 calls and 24000 aggregate
verifier seconds. A separately published short trial uses 1800 answering seconds;
its instructions and actual Run quota must identify that version's limit.
Both Agent and Verifier use 8 vCPUs and 4096 MiB memory, with 2048 MiB Agent
workspace; DBSCAN uses eight pinned OpenMP workers, make -j8, a 90-second build
limit and 45-second per-execution limit inside the 240-second train or 600-second final evaluation.
No GPU or task SSH access is allocated. The platform schema requires inert SSH
audit fields (32 sessions, 8388608 bytes per direction); these do not grant a
CPU-only task a GPU runner or an SSH transport. The 512 MiB GPU rollout is not
applicable here.

Network model limits: 1024 requests, 67108864 bytes per request and 8388608 bytes
per response. Total audit bundle: 34359738368 bytes (32 GiB), including the intended
4 GiB platform model-log allowance; actual deployment must provide that allowance.
Verifier evidence is bounded at 1048576 bytes. These are distinct from submission
and workspace limits. The model proxy and final lifecycle come from the Candidate
and Driver; task files contain no credentials or alternative platform clients.
