# Localized Regex Repair

Read [the task statement](repo/README.md) for the problem, editable surface, workloads, and scoring. Read [the testbed guide](bin/README.md) for exact budgets and submission instructions.

Create the writable repository with `/frontier/task/bin/prepare_workspace.sh`, then edit `/workspace/repo`. Generate the patch with `/frontier/task/bin/make_submission.sh`. Package `payload/solution.patch` in an uncompressed POSIX tar of at most 262,144 bytes; train, store-final, and final use the same format.

With an asynchronous Candidate, `frontier-submit train` returns a verification ID immediately; use `frontier-submit status <verification_id>` for completed feedback. A legacy Candidate without `status` waits inside `train` and prints the completed result; see the compatibility note in the testbed guide. `store-final` saves a replaceable answer without running verification or consuming a train call. An accepted explicit `final` is irreversible and takes precedence over the saved selection.

The submission commands are `frontier-submit train`, `frontier-submit status`, `frontier-submit store-final` and `frontier-submit final`; `train`, `store-final` and `final` take the tar containing `payload/solution.patch`, and `status` takes a verification ID. You have at most 100 train calls, within a cumulative 5,400-second training pool and a 7,200-second answering window; the independent final evaluation has its own 3,600-second limit.
