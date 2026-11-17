"""A smoke workflow for leases of runners, with no model: a scripted agent leases a GPU runner and
a CPU runner and drives each through every lease behaviour, then holds two CPU runners at once,
calls its train function, and submits `agent/probe.py`, a program that tries what a hostile
submission would. In both judges, the train function's target and the final call's, a body
(`calls.star`) runs `judge/judge.py`, which leases a runner of its own, puts the submission there,
has the runner count its bytes, and gets that back; then puts a private tree there, has the runner
print its modes, and gets it back. The final judge also runs the program on an untrusted runner, of
the agent's image and user, and checks each line it prints. It checks the deployment, not a task.

Each judge scores the checks that came out as expected, which its result's `detail` lists. The
train judge's two: its runner counted the payload's bytes, and the private tree's modes, `700`,
`600` and `444`, were those on the runner and those that came back, so it scores 2. The final
judge's 12, so the run scores 12 when everything behaves:

- its GPU runner counted the payload's bytes;
- the private tree's modes on its GPU runner, and after a get, were those it put;
- the program's 8 lines, as a runner has no grants, no model route, no egress and a filesystem
  of its own:
  - `read /pumpkin/home/judge-secret: no`, the file the judge wrote at that path before;
  - `read /opt/smoke/judge.py: no`, a file of the judge's image;
  - `list /pumpkin/home finds judge-secret: no`;
  - `env has SMOKE_JUDGE_MARK: no`, a variable of the judge's command;
  - `connect 1.1.1.1:443: no`;
  - `the gateway at <address> answers HTTP: no`, at the address `sandboxd` was given, or with no
    address found. The connection itself opening is not a failure: the runner's own `sandboxd`
    dials that port, and a connection that presents no registration grant gets no session. The
    line shows what came back, `(connected; read <n> bytes: "<text>")`, with bytes other than
    printable ASCII as `\\xNN`: expected is the gateway's refusal of a hello it does not accept;
  - `the model endpoint answers: no`, at 127.0.0.1:11235, with what came back if an HTTP answer
    did;
  - `write /pumpkin/home/escape: yes`, into the runner's own filesystem;
- the judge doesn't see `/pumpkin/home/escape` in its own sandbox;
- none of the program's lines holds the judge's secret or the variable's value."""

load("//lib/submit.star", "submit")
load("//smoke/lease/calls.star", "final_call", "train_call")

DEADLINE = arg("deadline_seconds", int, default = 900)
GPU_MODEL = arg("gpu_model", str, default = "NVIDIA A100-SXM4-80GB")

AGENT_IMAGE = image("agent/Dockerfile")
RUNNER_IMAGE = image("runner/Dockerfile")
JUDGE_IMAGE = image("judge/Dockerfile")
AGENT_USER = {"uid": 1000, "gid": 1000}
JUDGE_USER = {"uid": 65532, "gid": 65532}
GPU = {"count": 1, "model": GPU_MODEL}

workflow(
    name = "smoke/lease",
    roles = [
        {
            "name": "agent",
            "image": AGENT_IMAGE,
            "user": AGENT_USER,
            "needs": {"cpu_millis": 500, "memory_bytes": 512 * MiB},
            "runtime": {"agent": {
                "command": {"argv": ["python3", "-u", "/opt/smoke/agent.py"], "cwd": "/workspace"},
                "deadline_seconds": DEADLINE,
                "finalize_grace_seconds": 0,
                "grants": [
                    {"name": "gpu", "session": {"runner": "gpu_runner", "instances": 1, "instance_seconds": 300, "cumulative_seconds": 900, "commands": 20, "transfer_bytes": 1 * MiB}},
                    {"name": "cpu", "session": {"runner": "cpu_runner", "instances": 2, "instance_seconds": 300, "cumulative_seconds": 900, "commands": 20, "transfer_bytes": 1 * MiB}},
                    {"name": "train", "job": {"target": "train_judge", "body": train_call, "payload_max_bytes": 8 * KiB, "calls": 2, "call_timeout_seconds": 300}},
                    {"name": "final", "now": {"body": submit, "payload_max_bytes": 8 * KiB}},
                ],
            }},
        },
        {
            "name": "gpu_runner",
            "image": RUNNER_IMAGE,
            "user": AGENT_USER,
            "needs": {"cpu_millis": 2000, "memory_bytes": 4 * GiB, "gpu": GPU},
            "runtime": {"runner": {"cwd": "/workspace"}},
        },
        {
            "name": "cpu_runner",
            "image": RUNNER_IMAGE,
            "user": AGENT_USER,
            "needs": {"cpu_millis": 1000, "memory_bytes": 1 * GiB},
            "runtime": {"runner": {"cwd": "/workspace"}},
        },
        {
            "name": "train_judge",
            "image": JUDGE_IMAGE,
            "user": JUDGE_USER,
            "needs": {"cpu_millis": 500, "memory_bytes": 512 * MiB},
            "runtime": {"target": {
                "cwd": "/pumpkin/home",
                "sessions": [
                    {"name": "tlease", "session": {"runner": "train_runner", "instances": 1, "instance_seconds": 240, "commands": 10, "transfer_bytes": 1 * MiB}},
                ],
            }},
        },
        {
            "name": "train_runner",
            "image": JUDGE_IMAGE,
            "user": JUDGE_USER,
            "needs": {"cpu_millis": 1000, "memory_bytes": 1 * GiB},
            "runtime": {"runner": {"cwd": "/workspace"}},
        },
        {
            "name": "final_judge",
            "image": JUDGE_IMAGE,
            "user": JUDGE_USER,
            "needs": {"cpu_millis": 500, "memory_bytes": 512 * MiB},
            "runtime": {"target": {
                "cwd": "/pumpkin/home",
                "env": ["SMOKE_JUDGE_MARK=judge-only-value"],
                "sessions": [
                    {"name": "slease", "session": {"runner": "score_runner", "instances": 1, "instance_seconds": 240, "commands": 10, "transfer_bytes": 1 * MiB}},
                    {"name": "untrusted", "session": {"runner": "untrusted_runner", "instances": 1, "instance_seconds": 300, "commands": 4, "transfer_bytes": 1 * MiB}},
                ],
            }},
        },
        {
            "name": "score_runner",
            "image": JUDGE_IMAGE,
            "user": JUDGE_USER,
            "needs": {"cpu_millis": 2000, "memory_bytes": 4 * GiB, "gpu": GPU},
            "runtime": {"runner": {"cwd": "/workspace"}},
        },
        {
            # Runs the submission for the final judge: the agent's image and user, so nothing of
            # the judge's is in it.
            "name": "untrusted_runner",
            "image": AGENT_IMAGE,
            "user": AGENT_USER,
            "needs": {"cpu_millis": 1000, "memory_bytes": 1 * GiB},
            "runtime": {"runner": {"cwd": "/workspace"}},
        },
    ],
    final = {"target": "final_judge", "body": final_call, "payload_max_bytes": 8 * KiB, "timeout_seconds": 300},
)
