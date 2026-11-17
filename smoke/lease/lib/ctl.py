"""What the smoke's agent and judges share: `sandboxctl` ops as JSON, and a log line with the
seconds since the program started."""

import json
import subprocess
import threading
import time

CTL = "/.pumpkin/bin/sandboxctl"
START = time.monotonic()
PRINT = threading.Lock()
ENDED = {"ready", "closed", "expired", "failed"}
GONE = {"closed", "expired", "failed"}
# The session that opened each lease, by the lease's number.
SESSIONS = {}


def now():
    return time.monotonic() - START


def say(tag, *parts):
    with PRINT:
        print("[{:7.2f}] {}".format(now(), tag), *parts, flush=True)


def ctl(*args):
    done = subprocess.run([CTL, *args], capture_output=True, text=True)
    try:
        return json.loads(done.stdout)
    except ValueError:
        return {"kind": "error", "exit": done.returncode, "stdout": done.stdout, "stderr": done.stderr}


def short(value, limit=600):
    text = json.dumps(value, sort_keys=True)
    return text if len(text) <= limit else text[:limit] + "...({} chars)".format(len(text))


def open_lease(session, *extra):
    """The reply of `<session>.open`, whose lease's session it keeps."""
    reply = ctl("call", session + ".open", *extra)
    if reply.get("kind") == "lease":
        SESSIONS[reply["lease"]] = session
    return reply


def call(lease, op, *args):
    """The reply of op `op` of lease `lease`'s session, on that lease."""
    return ctl("call", SESSIONS[lease] + "." + op, str(lease), *args)


def status(lease):
    return call(lease, "status")


def wait(tag, lease, until, bound, every=0.5):
    """Polls lease `lease`, printing each state it sees, until its state is in `until`."""
    seen = None
    began = time.monotonic()
    while True:
        reply = status(lease)
        state = reply.get("state", reply.get("kind"))
        if state != seen:
            say(tag, "lease", lease, "state", state, "elapsed", reply.get("elapsed_seconds"),
                "busy", reply.get("busy_seconds"), "idle", reply.get("idle_seconds"),
                "remaining", short(reply.get("remaining")), "failure", reply.get("failure"))
            seen = state
        if state in until or reply.get("kind") != "lease_status":
            return reply
        if time.monotonic() - began > bound:
            say(tag, "gave up waiting on lease", lease, "after", bound, "s")
            return reply
        time.sleep(every)


def open_ready(tag, grant, *extra, bound=420):
    """Opens a lease of session `grant` and waits until it is ready; returns its number and
    status."""
    began = time.monotonic()
    reply = open_lease(grant, *extra)
    say(tag, "open", grant, *extra, "->", short(reply))
    if reply.get("kind") != "lease":
        return None, reply
    lease = reply["lease"]
    end = wait(tag, lease, ENDED, bound)
    say(tag, "lease", lease, "open to", end.get("state"), "in {:.2f} s".format(time.monotonic() - began))
    return lease, end


def last(tag, lease, what, bound=150):
    """Waits until lease `lease` has run what it was given, and prints how that ended."""
    began = time.monotonic()
    end = wait(tag, lease, ENDED, bound, every=0.2)
    done = end.get("last") or {}
    say(tag, what, "ended", short({k: v for k, v in done.items() if k not in ("stdout", "stderr")}),
        "after {:.2f} s by the script's clock".format(time.monotonic() - began))
    return end, time.monotonic() - began


def run(tag, lease, argv, *extra, lines=12):
    reply = call(lease, "run", *extra, "--", *argv)
    say(tag, "run", argv, *extra, "->", short(reply))
    if reply.get("kind") != "lease":
        return None, 0.0
    end, took = last(tag, lease, "command")
    done = end.get("last") or {}
    for stream in ("stdout", "stderr"):
        for line in (done.get(stream) or "").splitlines()[-lines:] if lines else ():
            say(tag, "  " + stream + "|", line)
    return end, took


def transfer(tag, lease, verb, *args):
    reply = call(lease, verb, *args)
    say(tag, verb, *args, "->", short(reply))
    if reply.get("kind") != "lease":
        return None
    end, _ = last(tag, lease, verb)
    return end
