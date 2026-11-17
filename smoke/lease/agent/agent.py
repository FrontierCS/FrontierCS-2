"""A scripted agent that drives two sessions, `gpu` and `cpu`, through every lease behaviour
in parallel, holds two leases of `cpu` at once, calls its train function, whose judge leases a
runner of its own, and submits. What it submits is `probe.py`, a program that tries what a hostile
submission would, which the final judge runs on an untrusted runner.

Each line is `[<seconds>] <grant> <what>`. It needs no model."""

import os
import shutil
import stat
import sys
import threading
import time

sys.path.insert(0, "/opt/smoke")
from ctl import ENDED, GONE, call, ctl, now, open_lease, open_ready, run, say, short, status, transfer, wait  # noqa: E402

GRANTS = ("gpu", "cpu")
RUN_SH = """#!/bin/sh
# Writes a result tree, with an executable tool in it, and says where it ran.
mkdir -p /workspace/out
echo "ran $0 as $(id -u) on $(uname -n)" > /workspace/out/result.txt
printf '#!/bin/sh\\necho tool\\n' > /workspace/out/tool.sh
chmod 755 /workspace/out/tool.sh
cat /workspace/out/result.txt
"""


def ops(tag, grant=None):
    reply = ctl("ops")
    grants = [g for g in reply.get("grants", []) if grant is None or g.get("name") == grant]
    for g in grants:
        say(tag, "ops: remaining_seconds", reply.get("remaining_seconds"), g.get("kind"), g.get("name"),
            "remaining", short(g.get("remaining")))


def tree(grant):
    """A small tree holding an executable script and a plain file."""
    root = "/workspace/t-" + grant
    os.makedirs(root, exist_ok=True)
    with open(root + "/run.sh", "w") as file:
        file.write(RUN_SH)
    os.chmod(root + "/run.sh", 0o755)
    with open(root + "/data.txt", "w") as file:
        file.write("data\n")
    return root


def mode(path):
    try:
        return oct(stat.S_IMODE(os.stat(path).st_mode))
    except OSError as error:
        return str(error)


def sequence(grant, show, open_while_closing=True):
    """Every behaviour of one lease at a time. `open_while_closing` opens another lease while the
    first closes, which a grant of one instance refuses."""
    tag = grant
    lease, end = open_ready(tag, grant)
    if lease is None or end.get("state") != "ready":
        say(tag, "first lease never got ready; stop this grant")
        return
    ops(tag, grant)
    run(tag, lease, show)

    # Trees: put a script, run it directly, put again (refused), replace, get back, get again.
    root = tree(grant)
    transfer(tag, lease, "put", root, "/workspace/t")
    run(tag, lease, ["/workspace/t/run.sh"])
    transfer(tag, lease, "put", root, "/workspace/t")
    transfer(tag, lease, "put", root, "/workspace/t", "--replace")
    got = "/workspace/got-" + grant
    transfer(tag, lease, "get", "/workspace/out", got)
    say(tag, "got:", sorted(os.listdir(got)) if os.path.isdir(got) else "nothing",
        "tool.sh mode", mode(got + "/tool.sh"), "result.txt mode", mode(got + "/result.txt"))
    transfer(tag, lease, "get", "/workspace/out", got)

    # Busy and idle: a 3 s command, then 3 s of nothing, against the status.
    before = status(lease)
    began = now()
    _, took = run(tag, lease, ["sleep", "3"])
    time.sleep(3)
    after = status(lease)
    say(tag, "busy/idle: the script saw {:.2f} s busy (op to status ready), {:.2f} s in all".format(
        took, now() - began),
        "status busy {} -> {}, idle {} -> {}".format(before.get("busy_seconds"), after.get("busy_seconds"),
                                                       before.get("idle_seconds"), after.get("idle_seconds")))

    # A command whose output is past the tail: 20000 lines, 108894 bytes.
    end, _ = run(tag, lease, ["seq", "1", "20000"], lines=0)
    out = ((end or {}).get("last") or {}).get("stdout", "")
    say(tag, "seq tail: {} chars, starts {!r}, ends {!r}".format(len(out), out[:12], out[-12:]))

    # Close, and see `closing` with an open refused while it lasts.
    closed = call(lease, "close")
    say(tag, "close", lease, "->", short(closed))
    seen = set()
    while True:
        reply = status(lease)
        state = reply.get("state")
        if state not in seen:
            seen.add(state)
            say(tag, "lease", lease, "state", state, "elapsed", reply.get("elapsed_seconds"),
                "busy", reply.get("busy_seconds"), "idle", reply.get("idle_seconds"),
                "remaining", short(reply.get("remaining")))
            if state == "closing" and open_while_closing:
                say(tag, "open while closing ->", short(open_lease(grant)))
                ops(tag, grant)
        if state in GONE:
            break
        time.sleep(0.2)

    # A short life: --seconds 15, then its expiry.
    lease, end = open_ready(tag, grant, "--seconds", "15")
    if lease is None or end.get("state") != "ready":
        return
    ready = now()
    end = wait(tag, lease, GONE, 200)
    say(tag, "lease", lease, end.get("state"), "{:.2f} s after the script saw it ready".format(now() - ready))
    ops(tag, grant)


def pair(grant):
    """Two leases of `grant`, which allows two at once: both ready at the same time, a command in
    each, a third open refused while both are held; then both closed, and an open refused while
    they close."""
    tag = grant + "-pair"
    leases = []
    for _ in range(2):
        reply = open_lease(grant)
        say(tag, "open", grant, "->", short(reply))
        if reply.get("kind") == "lease":
            leases.append(reply["lease"])
    ends = [wait(tag, lease, ENDED, 420) for lease in leases]
    if len(leases) != 2 or any(end.get("state") != "ready" for end in ends):
        say(tag, "the two leases never got ready together; close them")
        for lease in leases:
            call(lease, "close")
        return
    say(tag, "leases", leases, "now", [status(lease).get("state") for lease in leases])
    for lease in leases:
        run(tag, lease, ["sh", "-c", 'echo "lease $0 on $(uname -n)"', str(lease)])
    say(tag, "now", [status(lease).get("state") for lease in leases])
    third = open_lease(grant)
    say(tag, "third open while both are held ->", short(third))
    if third.get("kind") == "lease":
        call(third["lease"], "close")
    for lease in leases:
        say(tag, "close", lease, "->", short(call(lease, "close")))
    say(tag, "open while both close ->", short(open_lease(grant)))
    ops(tag, grant)


def cpu():
    sequence("cpu", ["sh", "-c", "nproc; uname -a"], open_while_closing=False)
    pair("cpu")


def judge():
    shutil.copy("/opt/smoke/probe.py", "/workspace/submission")
    reply = ctl("call", "train", "/workspace/submission")
    say("agent", "call train ->", short(reply))
    job = reply.get("job")
    if job is None:
        return
    seen = None
    while True:
        reply = ctl("status", str(job))
        state = reply.get("state")
        if state != seen:
            say("agent", "job", job, "state", state, short(reply))
            seen = state
        if state not in ("queued", "running"):
            break
        time.sleep(1)
    lease = (reply.get("returned") or {}).get("lease")
    if lease is not None:
        # Through a session of the agent's own, which holds no lease of the judge's.
        say("agent", "the judge's lease", lease, "status ->", short(ctl("call", "cpu.status", str(lease))))
        say("agent", "the judge's lease", lease, "close ->", short(ctl("call", "cpu.close", str(lease))))


def main():
    ops("agent")
    threads = [
        threading.Thread(target=sequence, args=("gpu", ["nvidia-smi", "-L"])),
        threading.Thread(target=cpu),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    ops("agent")
    judge()
    say("agent", "submit ->", short(ctl("call", "final", "/workspace/submission", "--arg", "finalize=true")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
