# Antigravity CLI Candidate (Gemini)

Not yet evaluated: no overlay image of this recipe is published.

The overlay runs Google's unmodified Antigravity CLI `agy` **1.4.7** headless
(`--print`, `--output-format stream-json`). The Dockerfile downloads the release
the official installer resolved from its `linux_amd64` manifest on 2026-10-24
and checks the manifest SHA-512. agy is a single native binary; its ELF
interpreter and rpath point at private loader/libc copies in
`/opt/frontier/candidate/agy/lib`, so it runs on any task base.

## Build

```bash
docker build --platform linux/amd64 -t frontier-candidate-agy-gemini:1.4.7 candidates/agy-gemini
```

## Model route

agy's only key-based provider is the native Gemini API
(`settings.json` `"modelProvider": "gemini"`, `GEMINI_API_KEY`,
`GOOGLE_GEMINI_BASE_URL`); it sends
`POST /v1beta/models/{model}:streamGenerateContent?alt=sse`. It has no
OpenAI-compatible provider. The Driver model proxy only serves
`/v1/responses`, `/v1/chat/completions` and `/v1/messages`, so Gemini-native
paths are refused.

The launcher therefore starts `gemini_chat_bridge.py` on a random loopback
port and points agy at it. The bridge:

- translates each Gemini request into one streamed Chat Completions request to
  `FRONTIER_MODEL_API_BASE_URL` with the CandidateVersion alias
  (`FRONTIER_CODEX_MODEL`) for every call, including agy's title generator and
  built-in tool-safety monitors (they ask for flash/flash-lite slugs);
- authenticates to the proxy with `FRONTIER_MODEL_PROXY_TOKEN`. agy authenticates
  to the bridge with a random per-run nonce in `GEMINI_API_KEY`. agy and the
  commands it runs get the platform's environment unchanged, so task tools that
  call the proxy with `FRONTIER_MODEL_PROXY_TOKEN` work;
- replays tool-call ids verbatim. LiteLLM's Gemini translator embeds thought
  signatures in tool-call ids (`call_…__thought__<b64>`), and agy echoes
  `functionCall.id` / `functionResponse.id` unchanged. The bridge also returns
  the signature as `thoughtSignature`;
- maps `reasoning_content` to Gemini thought parts and usage to `usageMetadata`;
- drops `search_web` and `ask_question` from the offered tools, and never
  forwards server-side Gemini tools such as googleSearch.

The LiteLLM route for the alias should be the same as the Pi Gemini
candidate's: LiteLLM's Gemini translator, dropping `reasoning_effort` and the
max-token fields and setting `generationConfig.thinkingConfig.thinkingLevel:
HIGH` explicitly. agy itself is started with `--model gemini-4-argon-high`
(`FRONTIER_AGY_MODEL`), its native slug for Gemini 4 Argon, which also asks
for `thinkingLevel: HIGH`.

## Runtime behaviour

- `--dangerously-skip-permissions` and `toolPermission: always-proceed` are set.
  `--add-dir` covers `/frontier/task` and the skills.
- Telemetry and notifications are off. The self-updater skips because the
  binary directory is read-only. Agent pods have no egress for the
  telemetry flush in any case.
- `HOME`, the XDG directories and `TMPDIR` are a mode-0700 directory under
  `/workspace/.frontier-agy-home`. The root filesystem can stay read-only.
- agy's built-in safety monitors (flash-lite pre-filter, 8 s timeout, then
  `severe_internal_risks_agy`, 10 s timeout, BLOCK mode) stay enabled because
  they are part of the official harness. They go to the same alias. In an
  offline test a monitor timeout failed open.
- Success requires the shared client's accepted final receipt.
- Logs: `agy.log` and `model-bridge.jsonl` hold one metadata line per model
  call, with no bodies or credentials. Both are in the private home.

Parity with the previous Gemini Candidate (`pi-gemini`): the launcher prompt is
byte-identical to pi-gemini's, the alias, route and HIGH thinking level are the
same, and output is uncapped by the same gateway drop rule. The bridge applies
pi's retry policy to the gateway hop (16 retries, 2 s base, 60 s cap, one-hour
request timeout) for connection errors and HTTP 408/409/425/429/5xx, and only
before any byte has reached agy, so an answer is never replayed.

## Tests

```bash
cd candidates/agy-gemini && python3 -m pytest -q test_gemini_chat_bridge.py
```

`offline-smoke.Dockerfile` runs `offline_container_smoke.py` against a built
overlay (`--build-arg FRONTIER_TEST_OVERLAY_IMAGE=<image>`), unprivileged and
network-isolated.
