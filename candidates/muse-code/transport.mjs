// Transport prototype for the official Muse host; not an agent loop.
import http from 'node:http';
import { isIP } from 'node:net';

export function createTransport({ driverBaseUrl, agentToken, catalog }) {
  const target = new URL(driverBaseUrl);
  if (target.protocol !== 'http:' || !isIP(target.hostname) ||
      target.pathname !== '/v1' || target.username || target.password ||
      target.search || target.hash) {
    throw new Error('Driver URL must be a literal-IP HTTP endpoint ending in /v1');
  }
  if (typeof agentToken !== 'string' || !agentToken || /[\r\n]/.test(agentToken)) {
    throw new Error('Agent-scoped Driver capability is required');
  }
  if (catalog?.object !== 'list' || catalog.data?.length !== 1 ||
      typeof catalog.data[0]?.id !== 'string' || !catalog.data[0].id ||
      catalog.data[0]?.object !== 'model') {
    throw new Error('Exactly one explicitly configured model catalog row is required');
  }
  // Freeze the reviewed catalog. Do not invent model limits or billing aliases.
  const catalogBody = Buffer.from(JSON.stringify(catalog));
  const active = new Set();
  const server = http.createServer((incoming, outgoing) => {
    if (incoming.method === 'GET' && incoming.url === '/muse-code/models') {
      outgoing.writeHead(200, { 'content-type': 'application/json',
        'content-length': catalogBody.length });
      outgoing.end(catalogBody);
      return;
    }
    if (incoming.method !== 'POST' || incoming.url !== '/responses') {
      outgoing.writeHead(404, { 'content-length': 0 });
      outgoing.end();
      incoming.resume();
      return;
    }
    // Never forward Muse's dummy bearer, cookies, caller-supplied capability,
    // upstream selector or gateway authority. Driver owns the real credentials.
    const headers = { 'x-frontier-agent-token': agentToken };
    for (const key of ['content-type', 'accept', 'content-length']) {
      if (incoming.headers[key] !== undefined) headers[key] = incoming.headers[key];
    }
    const upstream = http.request(new URL('/v1/responses', target), {
      method: 'POST', headers, agent: false,
    });
    active.add(upstream);
    upstream.once('close', () => active.delete(upstream));
    incoming.once('aborted', () => upstream.destroy());
    incoming.once('error', () => upstream.destroy());
    outgoing.once('close', () => {
      if (!outgoing.writableFinished) upstream.destroy();
    });
    upstream.once('error', () => {
      if (outgoing.destroyed) return;
      if (outgoing.headersSent) outgoing.destroy();
      else {
        outgoing.writeHead(502, { 'content-length': 0 });
        outgoing.end();
      }
    });
    upstream.once('response', response => {
      const resultHeaders = {};
      for (const key of ['content-type', 'content-length', 'content-encoding',
        'cache-control', 'retry-after', 'x-request-id']) {
        if (response.headers[key] !== undefined) resultHeaders[key] = response.headers[key];
      }
      outgoing.writeHead(response.statusCode, resultHeaders);
      outgoing.flushHeaders();
      response.once('error', () => outgoing.destroy());
      response.once('aborted', () => outgoing.destroy());
      response.pipe(outgoing);
    });
    // Node streams provide backpressure. No JSON/SSE reconstruction, buffering
    // whole conversations, retry, private model-step cap or alternate provider.
    incoming.pipe(upstream);
  });
  server.requestTimeout = 0;
  return {
    async listen() {
      await new Promise((resolve, reject) => {
        server.once('error', reject);
        server.listen(0, '127.0.0.1', resolve);
      });
      return `http://127.0.0.1:${server.address().port}`;
    },
    async close() {
      for (const request of active) request.destroy();
      server.closeAllConnections();
      await new Promise(resolve => server.close(resolve));
    },
  };
}
