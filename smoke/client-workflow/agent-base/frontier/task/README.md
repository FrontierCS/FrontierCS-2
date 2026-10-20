# Native submission workflow

Implement integer addition and demonstrate the installed submission client.
Correctness and completing the workflow are the objectives. Do not optimize
runtime, benchmark implementations, or debug machine code for this task.

Create an executable `/workspace/payload/run.sh`. It receives two decimal,
non-negative integer arguments whose sum fits in an unsigned 64-bit integer.
Write exactly their decimal sum, followed by optional ASCII whitespace.
Python's integer arithmetic is sufficient. You may include supporting files
under `payload/`. The program must not use the network or write outside its
working directory. Check representative cases, including zero and a sum near
the unsigned 64-bit limit, then create a tar containing `payload/run.sh` and
any supporting files. Do not include unrelated workspace files.

Read the installed `frontier-submit` skill. You, the Agent, must perform these
steps through that client:

1. Submit the archive with `frontier-submit train` exactly once. This returns
   an accepted verification ID and runs asynchronously. Read its feedback with
   `frontier-submit status` as documented by the skill.
2. Save your chosen correct archive with `frontier-submit store-final`.
   Saving is not judging and does not end the Agent. Do not wait to save if
   train feedback is still pending.
3. Once ready, call `frontier-submit final` exactly once using your chosen
   archive, then stop as instructed by the skill. Explicit final submission
   is required for this client acceptance; timeout fallback is insufficient.

The Agent window is 1800 seconds, with one train call, a separate 180-second
train pool and a separate 180-second final limit. Finish promptly once your
answer is correct; these are limits, not target durations. Saving does not
have a count quota, but payload, evidence and time limits still apply.

The verifier checks exact answers on multiple cases. A wrong answer, malformed
output, nonzero exit or timeout is rejected. The existing verifier also reports
a runtime-ratio score; it is diagnostic for this workflow acceptance, with no
performance target. A positive score does not replace the required explicit
final, genuine model execution, evidence verification or cleanup checks.

Do not call the Driver API directly. Do not use an operator or scripted
submission fixture to perform train, save or final on your behalf.
