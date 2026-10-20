# Testbed operations

Copy the source with `/frontier/task/prepare_workspace.sh`. Edit only
`src/alg.cpp` under `/workspace/repo`. Use `/frontier/task/make_submission.sh`
to create `/workspace/payload/solution.patch`, then package `payload/` in an
uncompressed POSIX tar. The full archive limit is 65,536 bytes.

`frontier-submit train /workspace/train.tar` spends one of 50 train calls.
Save a usable selection with `frontier-submit store-final /workspace/final.tar`;
that is a checkpoint, not a score. Explicit `frontier-submit final` selects one
final answer and ends answering; evaluation runs independently afterwards.
Follow the remaining-time and checkpoint limits returned by the runtime.
The formal answering budget is 18,000 seconds; independent final has 900 seconds.
No result is accepted merely because it was saved or had a development score.

Model transport audit allows 1,024 sessions, 64 MiB per model request,
8 MiB per response, and a 32 GiB aggregate audit bundle. These audit limits
do not change the CPU, workspace, submission, train, or independent-final quotas.
