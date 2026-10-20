const assert = require("node:assert/strict");
process.env.FRONTIER_MODEL_API_BASE_URL = "http://127.0.0.1:8082/v1";
const calls = [];
globalThis.fetch = (...args) => { calls.push(args); return Promise.resolve({ ok: true }); };
require("./fetch-sanitize.cjs");
(async () => {
  const headers = { Authorization: "Bearer run-local-token", "X-Request-ID": "candidate-id",
    "X-LiteLLM-API-Key": "candidate-key", "Content-Type": "application/json", "x-zcode-agent": "glm" };
  const body = JSON.stringify({ messages: [{ role: "user", content: "retained" }] });
  await fetch("http://127.0.0.1:8082/v1/chat/completions", { headers, body, method: "POST" });
  const options = calls[0][1];
  assert.equal(options.headers.get("authorization"), "Bearer run-local-token");
  assert.equal(options.headers.get("x-request-id"), null);
  assert.equal(options.headers.get("x-litellm-api-key"), null);
  assert.equal(options.headers.get("content-type"), "application/json");
  assert.equal(options.headers.get("x-zcode-agent"), "glm");
  assert.equal(options.body, body);
  const request = new Request("http://127.0.0.1:8082/v1/chat/completions", { headers, body, method: "POST" });
  await fetch(request);
  assert.equal(calls[1][0].headers.get("x-request-id"), null);
  assert.equal(await calls[1][0].text(), body);
  await fetch("https://example.test/v1/chat/completions", { headers, body });
  assert.equal(calls[2][1].headers, headers);
  assert.equal(headers["X-Request-ID"], "candidate-id");
  console.log("Driver header normalization: passed");
})().catch(error => { console.error(error); process.exitCode = 1; });
