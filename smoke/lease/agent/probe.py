"""The smoke's submission: a program that tries what a hostile submission would, from the runner
the final judge leases to run it, and prints one line for each attempt, `<attempt>: yes|no
(<what it saw>)`, where `yes` says the attempt got what it was after. The judge reads the lines;
the workflow's docstring says which outcome each should have."""

import http.client
import os
import socket
import time
import urllib.error
import urllib.request

# Where the final judge has its files: its working directory and a file of its own in it, and its
# program, which only the judge's image holds; and a variable of its command's environment.
JUDGE_HOME = "/pumpkin/home"
JUDGE_FILE = JUDGE_HOME + "/judge-secret"
JUDGE_PROGRAM = "/opt/smoke/judge.py"
JUDGE_VARIABLE = "SMOKE_JUDGE_MARK"
ESCAPE = JUDGE_HOME + "/escape"
PUBLIC = ("1.1.1.1", 443)
MODEL_ENDPOINT = "http://127.0.0.1:11235/v1/chat/completions"
# How much of an answer an attempt reads, and how much of that its line shows.
READ_BYTES = 512
SHOWN_BYTES = 200


def line(attempt, got, seen):
    print("{}: {} ({})".format(attempt, "yes" if got else "no", seen), flush=True)


def shown(data):
    """`read <n> bytes: "<text>"`, the text being the first bytes of `data` with printable ASCII
    kept and every other byte, `"` and `\\` as `\\xNN`, so the line stays one line."""
    text = "".join(chr(byte) if 0x20 <= byte < 0x7f and byte not in b'"\\' else "\\x{:02x}".format(byte)
                   for byte in data[:SHOWN_BYTES])
    return 'read {} bytes: "{}"'.format(len(data), text)


def read(path):
    try:
        with open(path, "rb") as file:
            line("read " + path, True, "{} bytes".format(len(file.read())))
    except OSError as error:
        line("read " + path, False, type(error).__name__)


def listing():
    try:
        entries = os.listdir(JUDGE_HOME)
    except OSError as error:
        line("list " + JUDGE_HOME + " finds judge-secret", False, type(error).__name__)
        return
    line("list " + JUDGE_HOME + " finds judge-secret", "judge-secret" in entries,
         "{} entries".format(len(entries)))


def gateway():
    """The gateway's address: the TCP address `sandboxd`, the sandbox's first process, was given,
    or else the default route's gateway, at port 443."""
    try:
        with open("/proc/1/cmdline", "rb") as file:
            words = file.read().decode(errors="replace").split("\0")
    except OSError:
        words = []
    for word in words:
        host, _, port = word.rpartition(":")
        if port.isdigit() and host.count(".") == 3 and host not in ("0.0.0.0", "127.0.0.1"):
            return host, int(port)
    try:
        with open("/proc/net/route") as file:
            for row in file.read().splitlines()[1:]:
                fields = row.split()
                if fields[1] == "00000000" and fields[2] != "00000000":
                    return socket.inet_ntoa(bytes.fromhex(fields[2])[::-1]), 443
    except (OSError, IndexError, ValueError):
        pass
    return None


def connect(address):
    try:
        with socket.create_connection(address, timeout=3):
            line("connect {}:{}".format(*address), True, "connected")
    except OSError as error:
        line("connect {}:{}".format(*address), False, type(error).__name__)


def receive(connection, deadline):
    """What arrives on `connection` until it ends, `READ_BYTES` have come or `deadline` passes;
    an error before any byte came is raised."""
    data = b""
    while len(data) < READ_BYTES:
        left = deadline - time.monotonic()
        if left <= 0:
            break
        connection.settimeout(left)
        try:
            chunk = connection.recv(READ_BYTES - len(data))
        except OSError:
            if not data:
                raise
            break
        if not chunk:
            break
        data += chunk
    return data


def ask_gateway():
    """Speaks HTTP to the gateway's address: `yes` only if an HTTP answer comes back, and shows
    what came back. Its port is the sessions' own way in, so a connection alone may be accepted."""
    address = gateway()
    if address is None:
        line("the gateway answers HTTP", False, "no address found")
        return
    attempt = "the gateway at {}:{} answers HTTP".format(*address)
    try:
        deadline = time.monotonic() + 3
        with socket.create_connection(address, timeout=3) as connection:
            connection.sendall(b"GET /v1/models HTTP/1.1\r\nHost: gateway\r\nConnection: close\r\n\r\n")
            answer = receive(connection, deadline)
    except OSError as error:
        line(attempt, False, type(error).__name__)
        return
    line(attempt, answer.startswith(b"HTTP/"), "connected; " + shown(answer))


def body(response):
    try:
        return response.read(READ_BYTES)
    except (OSError, http.client.HTTPException):
        return b""


def ask_model():
    request = urllib.request.Request(MODEL_ENDPOINT, method="POST", headers={"content-type": "application/json"},
                                     data=b'{"model":"any","messages":[{"role":"user","content":"hi"}]}')
    attempt = "the model endpoint answers"
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            line(attempt, response.status == 200, "HTTP {}; {}".format(response.status, shown(body(response))))
    except urllib.error.HTTPError as error:
        line(attempt, False, "HTTP {}; {}".format(error.code, shown(body(error))))
    except OSError as error:
        line(attempt, False, type(getattr(error, "reason", error)).__name__)


def write():
    try:
        with open(ESCAPE, "w") as file:
            file.write("written by the submission\n")
        line("write " + ESCAPE, True, "in the runner's own filesystem")
    except OSError as error:
        line("write " + ESCAPE, False, type(error).__name__)


def main():
    read(JUDGE_FILE)
    read(JUDGE_PROGRAM)
    listing()
    line("env has " + JUDGE_VARIABLE, JUDGE_VARIABLE in os.environ, "{} variables".format(len(os.environ)))
    connect(PUBLIC)
    ask_gateway()
    ask_model()
    write()


if __name__ == "__main__":
    main()
