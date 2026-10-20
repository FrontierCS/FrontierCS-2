# Optimize `a + b`

Your goal is to make the supplied integer-addition program as fast as possible
while preserving its behavior.

## Ordinary model smoke budget

For an ordinary model smoke using this task, your Agent budget is **1800 seconds
(30 minutes)**. This is a hard execution limit, not a target work duration.
The separate scripted deployment fixture does not use this model workflow.

First make a simple correct implementation and check representative cases
locally. Submit it with `frontier-submit train` promptly; do not spend the
whole window optimizing before your first submission. Train returns an
accepted verification ID and continues asynchronously. Check its actual
feedback later with `frontier-submit status`; queueing can delay feedback.

Choose and save a correct answer early with `frontier-submit store-final`.
Saving does not require waiting for train to finish and does not run a judge.
You can continue improving your answer and replace the saved selection while
time remains. For this smoke, completing the submission workflow with a
correct answer is more important than maximizing the speed score. Aim to
finish comfortably before the hard limit, ideally within 15 minutes when
verification feedback is available. Call `frontier-submit final` exactly once
when ready, then stop as instructed by the installed submission skill.

## Interface

Your submission must be a tar archive with this entrypoint:

```text
payload/run.sh
```

The entrypoint is invoked as:

```text
payload/run.sh <a> <b>
```

Both arguments are decimal, non-negative integers whose sum fits in an unsigned
64-bit integer. The program must write exactly the decimal value `a + b`,
followed by optional ASCII whitespace, to standard output. It must not require
network access or write outside its working directory.

`payload/run.sh` must be executable. You may include additional files below
`payload/`, but the complete tar must stay below the Task submission limit.

## Starter

The deliberately slow starter is at:

```text
/frontier/task/starter/payload/
```

It starts at zero, increments once `a` times, and then increments once `b`
times. Copy it into `/workspace/payload` and optimize it. Python is available,
but you may use any runtime already present in the image.

## Objective

The verifier first checks exact answers on multiple cases. A wrong answer,
timeout, non-zero exit, or malformed output fails verification.

For a correct submission, `raw_score` is:

```text
sum(reference runtime) / sum(measured median runtime)
```

Higher is better. The score is always a finite non-negative number. Train and
eval use the same verifier image and command, but different datasets. Eval
cases are not present in this Agent image.

## Finish

Create and locally validate the complete submission tree at:

```text
/workspace/payload
```

Read the installed `frontier-submit` skill. Create a tar containing
`payload/run.sh` and any supporting files, then use `frontier-submit train`
for feedback. Training does not select your final answer.

You must explicitly select your answer with `frontier-submit store-final`.
Choose which answer to save based on your own judgment; you may replace it
while your Agent budget remains. Saving has no count quota, does not run a
judge and does not end the Agent. Payload, audit and time limits still apply.
Use `frontier-submit final` once when you are done, then exit. If your Agent
window ends first, the Driver independently judges your most recently saved
selection. Exiting alone never packages or selects an unsaved workspace.

Do not call the Driver API directly. `/frontier/task/check.sh` and
`/frontier/task/submit.sh` are for the separate scripted deployment fixture;
model Candidates use the installed submission skill.
