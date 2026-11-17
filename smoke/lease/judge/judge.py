"""A judge that leases a runner: it puts the submitted file there, has the runner count its bytes
and name its host, and gets that back. It also puts a private tree there, a `0700` directory
holding a `0600` file and a `0444` file, has the runner print their modes, and gets the tree back:
both the runner's modes and those that came back must be the ones it put. `judge.py <session> keep`
exits with its lease still open, so the call's end closes it; `judge.py <session> close` closes it
first.

`judge.py <session> close <untrusted session>` also runs the submission, a program, on a runner of
the untrusted session, which has the agent's image and user, and gets its lines back; see
`agent/probe.py`. Before, it writes a file only its own sandbox holds at the path the program
tries to read.

The result's `detail` lists each check, as expected or not, a check of a line being the line as
printed, with what an attempt read; the score is how many came out as expected."""

import json
import os
import shutil
import sys

sys.path.insert(0, "/opt/smoke")
from ctl import ENDED, GONE, call, ctl, open_lease, open_ready, run, say, short, transfer, wait  # noqa: E402

PAYLOAD = "/pumpkin/in/payload/raw"
RESULT = "/pumpkin/out/result.json"
COMMAND = ("mkdir -p /workspace/res && wc -c < /workspace/sub/raw > /workspace/res/bytes"
           " && { nvidia-smi -L 2>/dev/null || echo no-gpu; uname -n; } > /workspace/res/host")
PROBE = "mkdir -p /workspace/out && python3 -I /workspace/sub/raw > /workspace/out/lines 2>&1"
# The private tree's entries and their modes, in the order the runner prints them.
MODES = (("private", 0o700), ("private/secret", 0o600), ("private/frozen", 0o444))
STAT = "cd /workspace/modes && stat -c %a " + " ".join(path for path, _ in MODES)
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


def modes(tag, lease, checks):
    """Puts the private tree on lease `lease`, has the runner print its modes, gets it back, and
    checks both against the modes put."""
    os.makedirs("modes/private")
    for path, _ in MODES[1:]:
        with open("modes/" + path, "w") as file:
            file.write(path + "\n")
    for path, mode in reversed(MODES):
        os.chmod("modes/" + path, mode)
    put = " ".join("{:o}".format(mode) for _, mode in MODES)
    transfer(tag, lease, "put", "modes", "/workspace/modes")
    end, _ = run(tag, lease, ["sh", "-c", STAT])
    runner = " ".join(((end or {}).get("last") or {}).get("stdout", "").split())
    transfer(tag, lease, "get", "/workspace/modes", "modes-back")
    try:
        back = " ".join("{:o}".format(os.stat("modes-back/" + path).st_mode & 0o7777) for path, _ in MODES)
    except OSError as error:
        back = str(error)
    check(checks, "modes put {}, on the runner {}, got back {}".format(put, runner, back),
          runner == put and back == put)


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


def main():
    grant, end = sys.argv[1], sys.argv[2]
    untrusted = sys.argv[3] if len(sys.argv) > 3 else None
    tag = "judge-" + grant
    reply = ctl("ops")
    say(tag, "ops ->", short(reply))
    checks = []
    result = {"score": None, "lease": None, "detail": checks}
    other = None
    if untrusted is not None:
        with open(SECRET, "w") as file:
            file.write(os.urandom(16).hex() + "\n")
        # Opened first, so that its runner starts while the other lease's does.
        opened = open_lease(untrusted)
        say(tag, "open", untrusted, "->", short(opened))
        other = opened.get("lease")
        if other is None:
            check(checks, "open " + untrusted + ": refused", False)
    lease, ready = open_ready(tag, grant)
    result["lease"] = lease
    if lease is not None and ready.get("state") == "ready":
        os.makedirs("sub")
        shutil.copy(PAYLOAD, "sub/raw")
        transfer(tag, lease, "put", "sub", "/workspace/sub")
        run(tag, lease, ["sh", "-c", COMMAND])
        transfer(tag, lease, "get", "/workspace/res", "res")
        with open("res/bytes") as file:
            counted = int(file.read().strip())
        size = os.path.getsize(PAYLOAD)
        check(checks, "the runner counted the payload's {} bytes: {}".format(size, counted), counted == size)
        with open("res/host") as file:
            result["host"] = file.read().strip()
        modes(tag, lease, checks)
        if end == "close":
            say(tag, "close ->", short(call(lease, "close")))
            wait(tag, lease, GONE, 120, every=0.2)
        else:
            say(tag, "exiting with lease", lease, "open")
    if other is not None:
        probe(untrusted, other, checks)
    result["score"] = sum(1 for text in checks if text.startswith("as expected: "))
    with open(RESULT, "w") as file:
        json.dump(result, file)
    say(tag, "result", short(result, limit=4000))
    return 0


if __name__ == "__main__":
    sys.exit(main())
