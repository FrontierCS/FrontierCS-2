"""The smoke's agent and judges: model requests through the sandbox's model endpoint, and ops
through `sandboxctl`. `routes.py agent <model> <chat model>` sends a streamed Chat Completions request, a
Responses request and one with a provider-side tool, which the route refuses, then calls the train
function and submits. `routes.py judge <model>` asks for a digit and scores it."""

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request

ENDPOINT = "http://127.0.0.1:11235"
CTL = "/.pumpkin/bin/sandboxctl"
START = time.monotonic()


def say(*parts):
    print("[{:7.2f}]".format(time.monotonic() - START), *parts, flush=True)


def post(path, body):
    """POSTs `body` to the endpoint's `path`, and returns the status and the body's text."""
    request = urllib.request.Request(ENDPOINT + path, data=json.dumps(body).encode(),
                                     headers={"content-type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=240) as response:
            return response.status, response.read().decode(errors="replace")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode(errors="replace")


def chat(model, prompt, **extra):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 400}
    body.update(extra)
    return post("/v1/chat/completions", body)


def ctl(*args):
    done = subprocess.run([CTL, *args], capture_output=True, text=True)
    say("sandboxctl", *args, "->", done.stdout.strip() or done.stderr.strip())
    try:
        return json.loads(done.stdout)
    except ValueError:
        return {}


def text_of_chat(body):
    """The answer's text in a whole (not streamed) Chat Completions response."""
    try:
        return json.loads(body)["choices"][0]["message"]["content"] or ""
    except (ValueError, KeyError, IndexError, TypeError):
        return ""


def agent(model, chat_model):
    status, body = chat(chat_model, "Reply with the single word: pumpkin.", stream=True,
                        stream_options={"include_usage": True})
    say("chat, streamed:", status, body[-600:])
    status, body = post("/responses", {"model": model, "store": False, "input": "Reply with the single word: route.",
                                       "max_output_tokens": 400})
    say("responses:", status, body[-600:])
    status, body = chat(chat_model, "Search the web for pumpkins.", tools=[{"type": "web_search_preview"}])
    say("chat with a provider-side tool:", status, body)
    with open("submission", "w") as file:
        file.write("hello")
    job = ctl("call", "train", "submission").get("job")
    for _ in range(300):
        state = ctl("status", str(job)).get("state")
        if state not in ("queued", "running"):
            break
        time.sleep(1)
    ctl("call", "final", "submission", "--arg", "finalize=true")


def judge(model):
    status, body = chat(model, "Reply with one digit from 1 to 9 and nothing else.")
    answer = text_of_chat(body)
    say("chat:", status, body[-600:])
    digits = [c for c in answer if c.isdigit()]
    score = int(digits[-1]) if digits else 0
    with open("/pumpkin/out/result.json", "w") as file:
        json.dump({"score": score, "answer": answer[-200:]}, file)
    say("score", score)


if __name__ == "__main__":
    {"agent": agent, "judge": judge}[sys.argv[1]](*sys.argv[2:])
