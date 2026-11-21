"""A smoke workflow for model routes on any role: a scripted agent sends one Chat Completions
request, one Responses request, and one request the route must refuse, through its own route;
the train judge, a call's target, and the final judge, the final call's, each ask the model once
through theirs, and score from the answer. It checks the deployment, not a task."""

load("//lib/submit.star", "FINAL_HELP", "submit")
load("//smoke/routes/calls.star", "final_call", "train_call")

MODEL = arg("model", str, default = "glm-5.3-flash")

# The model the judges ask, which their routes allow, and which the agent asks too.
CHAT_MODEL = arg("chat_model", str, default = "gpt-6.1-terra")

# Room for three model requests and a train judge call, whose sandbox starts on EC2 in under a minute.
DEADLINE = arg("deadline_seconds", int, default = 900)

IMAGE = image("Dockerfile")
USER = {"uid": 1000, "gid": 1000}
NEEDS = {"cpu_millis": 500, "memory_bytes": 512 * MiB}
CHAT = [{"api": "chat", "models": [CHAT_MODEL]}]

JUDGE_RUNTIME = {"target": {"cwd": "/pumpkin/home", "egress": {"model": CHAT}}}

workflow(
    deadline_seconds = DEADLINE,
    name = "smoke/routes",
    roles = [
        {
            "name": "agent",
            "image": IMAGE,
            "user": USER,
            "needs": NEEDS,
            "runtime": {"agent": {
                "command": {"argv": ["python3", "-u", "/opt/routes/routes.py", "agent", MODEL, CHAT_MODEL], "cwd": "/workspace"},
                "finalize_grace_seconds": 0,
                "egress": {"model": [{"api": "responses", "models": [MODEL]}] + CHAT},
                "grants": [
                    {"name": "train", "job": {"target": "train_judge", "body": train_call, "keywords": {"chat_model": CHAT_MODEL}, "payload_max_bytes": 1 * KiB, "calls": 1, "call_timeout_seconds": 300}},
                    {"name": "final", "help": FINAL_HELP, "now": {"body": submit, "payload_max_bytes": 1 * KiB}},
                ],
            }},
        },
        {
            "name": "train_judge",
            "image": IMAGE,
            "user": USER,
            "needs": NEEDS,
            "runtime": JUDGE_RUNTIME,
        },
        {
            "name": "final_judge",
            "image": IMAGE,
            "user": USER,
            "needs": NEEDS,
            "runtime": JUDGE_RUNTIME,
        },
    ],
    final = {"target": "final_judge", "body": final_call, "keywords": {"chat_model": CHAT_MODEL}, "payload_max_bytes": 1 * KiB, "timeout_seconds": 300},
)
