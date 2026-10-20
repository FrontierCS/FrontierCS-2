// ZCode is untrusted candidate code. Keep provider authority/correlation headers
// out of the run-local model gateway; the Driver owns those values.
const forbidden = [
  "cookie", "set-cookie", "openai-organization", "openai-project", "x-api-key",
  "api-key", "forwarded", "via", "traceparent", "tracestate", "baggage",
  "sentry-trace", "x-request-id", "x-correlation-id", "x-amzn-trace-id",
];
const originalFetch = globalThis.fetch;
const isModelGateway = (input) => {
  const value = typeof input === "string" || input instanceof URL ? input : input && input.url;
  try {
    const url = new URL(value);
    const base = new URL(process.env.FRONTIER_MODEL_API_BASE_URL);
    return url.origin === base.origin && url.pathname.startsWith("/v1/");
  } catch { return false; }
};
const clean = (input, init) => {
  const source = init && init.headers !== undefined
    ? init.headers
    : input instanceof Request ? input.headers : undefined;
  const headers = new Headers(source);
  for (const name of [...headers.keys()]) {
    if (forbidden.includes(name) || /^(?:x-forwarded-|x-litellm-|x-proxy-|proxy-)/.test(name)) {
      headers.delete(name);
    }
  }
  if (init) return [input, { ...init, headers }];
  if (input instanceof Request) return [new Request(input, { headers }), undefined];
  return [input, { headers }];
};
globalThis.fetch = function (input, init) {
  if (!isModelGateway(input)) return originalFetch(input, init);
  const [request, options] = clean(input, init);
  return originalFetch(request, options);
};
