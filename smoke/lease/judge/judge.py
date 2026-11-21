"""Check runner transfers and isolation. The train judge leaves its CPU lease open.
The final body owns the score runner and returns its measurements here before the
untrusted runner's isolation checks.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/opt/smoke")
from ctl import ENDED, SESSIONS, call, open_lease, open_ready, run, say, short, transfer, wait  # noqa: E402

PAYLOAD = "/pumpkin/in/payload/raw"
RESULT = "/pumpkin/out/result.json"
JUDGE = ["python3", "-u", "/opt/smoke/judge.py"]
PROBE = "mkdir -p /workspace/out && python3 -I /workspace/sub/raw > /workspace/out/lines 2>&1"
# The private tree's entries and their modes, in the order the runner prints them.
MODES = (("private", 0o700), ("private/secret", 0o600), ("private/frozen", 0o444))
SECRET = "/pumpkin/home/judge-secret"
ESCAPE = "/pumpkin/home/escape"

# Each line the program prints, by the attempt it names (up to its first colon, past an address),
# and the outcome it should have: the runner reaches nothing of the judge's and nothing beyond
# its own session; what it writes stays in its own filesystem.
EXPECTED = {
    "read /pumpkin/home/judge-secret": "no",
    "read /opt/smoke/judge.py": "no",
    "list /pumpkin/home finds judge-secret": "no",
    "env has SMOKE_JUDGE_MARK": "no",
    "connect 1.1.1.1:443": "no",
    "the gateway": "no",
    "the model endpoint answers": "no",
    "write /pumpkin/home/escape": "yes",
}


def check(checks, what, ok):
    checks.append(("as expected: " if ok else "NOT as expected: ") + what)
    say("check", checks[-1])


def measure():
    """On the runner, count the transferred bytes and write the count as an unsigned binary word."""
    with open("/workspace/sub/raw", "rb") as file:
        count = len(file.read())
    Path("/workspace/res").mkdir()
    Path("/workspace/res/bytes").write_bytes(count.to_bytes(8, "big"))
    try:
        subprocess.run(["nvidia-smi", "--list-gpus"], check=True, stderr=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("no-gpu")
    print(os.uname().nodename)


def make_modes():
    os.makedirs("modes/private")
    for path, _ in MODES[1:]:
        with open("modes/" + path, "w") as file:
            file.write(path + "\n")
    for path, mode in reversed(MODES):
        os.chmod("modes/" + path, mode)


def modes(directory):
    return " ".join("{:o}".format((Path(directory) / path).stat().st_mode & 0o7777)
                    for path, _ in MODES)


def measurements(checks, runner_modes, count_file):
    word = Path(count_file).read_bytes()
    counted = int.from_bytes(word, "big")
    size = os.path.getsize(PAYLOAD)
    check(checks, "the runner counted the payload's {} bytes: {}".format(size, counted),
          len(word) == 8 and counted == size)
    put = " ".join("{:o}".format(mode) for _, mode in MODES)
    try:
        back = modes("modes-back")
    except OSError as error:
        back = str(error)
    check(checks, "modes put {}, on the runner {}, got back {}".format(put, runner_modes, back),
          runner_modes == put and back == put)


def probe(grant, lease, checks):
    """Runs the submission on lease `lease` of `grant`, and checks each line it printed, and that
    its write and the judge's secret stayed where they were."""
    tag = "judge-" + grant
    ready = wait(tag, lease, ENDED, 420)
    if ready.get("state") != "ready":
        check(checks, "the untrusted runner got ready: no ({})".format(ready.get("state")), False)
        return
    os.makedirs("probe")
    shutil.copy(PAYLOAD, "probe/raw")
    transfer(tag, lease, "put", "probe", "/workspace/sub")
    run(tag, lease, ["sh", "-c", PROBE], "--timeout", "60")
    transfer(tag, lease, "get", "/workspace/out", "probe-out")
    try:
        with open("probe-out/lines") as file:
            lines = file.read().splitlines()
    except OSError as error:
        lines = []
        say(tag, "no lines back:", error)
    for attempt, wanted in EXPECTED.items():
        found = [text for text in lines if text.startswith(attempt)]
        seen = found[0].split(" (")[0].rpartition(": ")[2] if found else "missing"
        check(checks, found[0] if found else attempt + ": missing", seen == wanted)
    escaped = os.path.exists(ESCAPE)
    check(checks, "the judge sees the program's {}: {}".format(ESCAPE, "yes" if escaped else "no"), not escaped)
    with open(SECRET) as file:
        secret = file.read().strip()
    leaked = any(secret in text or os.environ["SMOKE_JUDGE_MARK"] in text for text in lines)
    check(checks, "the program's output holds the judge's secret or mark: {}".format("yes" if leaked else "no"),
          not leaked)
    say(tag, "close ->", short(call(lease, "close")))


def write_result(checks, **fields):
    result = {"score": sum(text.startswith("as expected: ") for text in checks),
              "detail": checks, **fields}
    with open(RESULT, "w") as file:
        json.dump(result, file)
    say("judge", "result", short(result, limit=4000))


def train():
    tag = "judge-tlease"
    lease, ready = open_ready(tag, "tlease")
    checks, fields = [], {"lease": lease}
    if lease is not None and ready.get("state") == "ready":
        transfer(tag, lease, "put", "/pumpkin/in/payload", "/workspace/sub")
        end, _ = run(tag, lease, JUDGE + ["measure"])
        fields["host"] = ((end or {}).get("last") or {}).get("stdout", "").strip()
        transfer(tag, lease, "get", "/workspace/res", "res")
        make_modes()
        transfer(tag, lease, "put", "modes", "/workspace/modes")
        end, _ = run(tag, lease, JUDGE + ["show-modes"])
        seen = ((end or {}).get("last") or {}).get("stdout", "").strip()
        transfer(tag, lease, "get", "/workspace/modes", "modes-back")
        measurements(checks, seen, "res/bytes")
        say(tag, "exiting with lease", lease, "open")
    write_result(checks, **fields)


def prepare_final():
    make_modes()
    with open(SECRET, "w") as file:
        file.write(os.urandom(16).hex() + "\n")
    # Both runners can start at once. Only the untrusted lease number crosses judge commands.
    print(json.dumps(open_lease("untrusted").get("lease")))


def finish_final(lease_json, host, runner_modes):
    checks = []
    measurements(checks, runner_modes.strip(), "/pumpkin/out/measurement/bytes")
    lease = json.loads(lease_json)
    if lease is None:
        check(checks, "open untrusted: refused", False)
    else:
        SESSIONS[lease] = "untrusted"
        probe("untrusted", lease, checks)
    write_result(checks, host=host.strip())


if __name__ == "__main__":
    commands = {"measure": measure, "show-modes": lambda: print(modes("/workspace/modes")),
                "train": train, "prepare-final": prepare_final, "finish-final": finish_final}
    commands[sys.argv[1]](*sys.argv[2:])
