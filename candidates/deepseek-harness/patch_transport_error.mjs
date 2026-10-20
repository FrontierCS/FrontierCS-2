// One reviewed patch to the locked upstream adapter; fail closed on package drift.
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';

export const upstreamSha256 = 'd46dbc30aeb3414190000e6f8aeaaf994246b5e5c651de07639f4a7e92de37ef';
export const transportGuard = String.raw`
	// LiteLLM embeds aiohttp's parser code 400 inside a failed HTTP 200 SSE stream.
	// This exact transport signature must precede the upstream generic 400 matcher.
	if (/^litellm\.APIConnectionError:\s*APIConnectionError:\s*OpenAIException - Response payload is not completed:\s*<TransferEncodingError:\s*400,\s*message=['"]Not enough data to satisfy transfer length header\.['"]>\s*$/.test(message)) return "TRANSPORT";
`;
export function patchAdapter(source) {
  if (createHash('sha256').update(source).digest('hex') !== upstreamSha256)
    throw new Error('Pinned dsh-llm-pi-ai source differs; review the adapter before patching');
  const marker = 'function classifyPiAiError(message) {';
  if (source.split(marker).length !== 2) throw new Error('Adapter classifier is not unique');
  return source.replace(marker, marker + transportGuard);
}
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const path = process.argv[2];
  if (!path) throw new Error('Expected the pinned adapter path');
  writeFileSync(path, patchAdapter(readFileSync(path, 'utf8')));
}
