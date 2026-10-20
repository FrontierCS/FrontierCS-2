import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { patchAdapter } from './patch_transport_error.mjs';

// Exercise the exact locked upstream classifier, not a duplicate implementation.
const original = readFileSync(process.argv[2], 'utf8');
const patched = patchAdapter(original);
const source = patched.match(/function classifyPiAiError\(message\) \{[\s\S]*?\n\}/)[0];
const classify = new Function('isQuotaExceededError', 'QUOTA_EXCEEDED_CODE', source + ';return classifyPiAiError;')(
  () => false, 'QUOTA_EXCEEDED');
const truncated = "litellm.APIConnectionError: APIConnectionError: OpenAIException - Response payload is not completed: <TransferEncodingError: 400, message='Not enough data to satisfy transfer length header.'>";
assert.equal(classify(truncated), 'TRANSPORT');
for (const [message, code] of [
  ['400 invalid_request: unsupported parameter', 'INVALID_REQUEST'],
  ['401 Unauthorized', 'AUTH'], ['403 Forbidden', 'AUTH'],
  ['413 payload too large', 'INVALID_REQUEST'], ['429 rate limit', 'RATE_LIMIT'],
  ['500 internal error', 'SERVER'], ['socket closed', 'TRANSPORT'],
  ['400 invalid_request: ' + truncated, 'INVALID_REQUEST'],
  ['TransferEncodingError: 400 invalid parameter', 'INVALID_REQUEST'],
  [truncated.replace('APIConnectionError', 'BadRequestError'), 'INVALID_REQUEST'],
]) assert.equal(classify(message), code, message);
assert.throws(() => patchAdapter(original + '\n'), /source differs/);
assert.throws(() => patchAdapter(patched), /source differs/);
console.log('Pinned adapter transport classification: 13 assertions passed');
