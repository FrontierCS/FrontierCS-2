"""Lossless private CKKS transport; no scientific calculations.

4 MiB archive = at most 3 MiB compressed measurement member + 1 MiB
for metrics, tar framing and zstd overhead. Metrics retain their 512 KiB cap.
The collector reserves 8 MiB decoded / 2 MiB compressed for a next pair and failure tail.
Bounds are checked again on actual bytes; capacity failure is never success.
"""
import base64
import hashlib
import json
import lzma

MAX_DECODED = 24 * 1024 * 1024
LEGACY_MAX_DECODED = 12 * 1024 * 1024
MAX_MEMBER = 3 * 1024 * 1024
MAX_ARCHIVE = 4 * 1024 * 1024
PAIR_DECODED_RESERVE = 8 * 1024 * 1024
PAIR_COMPRESSED_RESERVE = 2 * 1024 * 1024
MEMBER = 'ckks/calibration.json.xz'
ENCODING = 'lzma-json-member-v2'
WIRE_ENCODING = 'lzma-base64-member-transfer-v2'
KEYS = {'encoding', 'member', 'decoded_bytes', 'sha256', 'compressed_bytes', 'compressed_sha256'}

def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':')).encode()

def packed(value):
    raw = canonical(value)
    if not 0 < len(raw) <= MAX_DECODED:
        raise ValueError('calibration evidence exceeds decoded limit')
    body = lzma.compress(raw, preset=6)
    if len(body) > MAX_MEMBER:
        raise ValueError('calibration evidence exceeds member limit')
    blob = base64.b64encode(body).decode('ascii')
    return {'encoding': WIRE_ENCODING, 'member': MEMBER,
            'decoded_bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
            'compressed_bytes': len(body), 'compressed_sha256': hashlib.sha256(body).hexdigest(),
            'chunks': [blob[i:i+1800] for i in range(0, len(blob), 1800)]}

def descriptor(value):
    result = {key: value[key] for key in KEYS}
    result['encoding'] = ENCODING
    return result

def validate_descriptor(value):
    if (not isinstance(value, dict) or set(value) != KEYS
            or value['encoding'] != ENCODING or value['member'] != MEMBER
            or type(value['decoded_bytes']) is not int
            or not 0 < value['decoded_bytes'] <= MAX_DECODED
            or type(value['compressed_bytes']) is not int
            or not 0 < value['compressed_bytes'] <= MAX_MEMBER):
        raise ValueError('invalid evidence descriptor')
    for key in ('sha256', 'compressed_sha256'):
        if (not isinstance(value[key], str) or len(value[key]) != 64
                or any(c not in '0123456789abcdef' for c in value[key])):
            raise ValueError('invalid evidence digest')

def decode_member(value, body):
    validate_descriptor(value)
    if (not isinstance(body, bytes) or len(body) != value['compressed_bytes']
            or hashlib.sha256(body).hexdigest() != value['compressed_sha256']):
        raise ValueError('invalid compressed evidence integrity')
    dec = lzma.LZMADecompressor(memlimit=128 * 1024 * 1024)
    raw = dec.decompress(body, max_length=value['decoded_bytes'] + 1)
    if (not dec.eof or dec.unused_data or len(raw) != value['decoded_bytes']
            or hashlib.sha256(raw).hexdigest() != value['sha256']):
        raise ValueError('invalid evidence integrity')
    return json.loads(raw)

def extract_transfer(value):
    if (not isinstance(value, dict) or set(value) != KEYS | {'chunks'}
            or value['encoding'] != WIRE_ENCODING):
        raise ValueError('invalid evidence transfer')
    desc = descriptor(value)
    validate_descriptor(desc)
    chunks = value['chunks']
    if (not isinstance(chunks, list) or len(chunks) > (4 * MAX_MEMBER // 3 + 1799) // 1800
            or any(not isinstance(c, str) or not 0 < len(c) <= 1800 for c in chunks)
            or sum(map(len, chunks)) > 4 * ((MAX_MEMBER + 2) // 3)):
        raise ValueError('encoded evidence too large')
    body = base64.b64decode(''.join(chunks), validate=True)
    decode_member(desc, body)
    return desc, body

def unpack(value, member_bytes=None):
    if value.get('encoding') == WIRE_ENCODING:
        desc, body = extract_transfer(value)
        return decode_member(desc, body)
    if value.get('encoding') == ENCODING:
        return decode_member(value, member_bytes)
    # Historical v1 receipts remain readable, under their original bounds.
    if (value.get('encoding') != 'lzma-base64-json-chunks-v1'
            or type(value.get('decoded_bytes')) is not int
            or not 0 < value['decoded_bytes'] <= LEGACY_MAX_DECODED):
        raise ValueError('invalid evidence envelope')
    chunks = value['chunks']
    if (not isinstance(chunks, list) or len(chunks) > 256
            or any(not isinstance(c, str) or len(c) > 1800 for c in chunks)):
        raise ValueError('invalid legacy chunks')
    encoded = ''.join(chunks)
    if len(encoded) > 450 * 1024:
        raise ValueError('encoded evidence too large')
    body = base64.b64decode(encoded, validate=True)
    desc = {'encoding': ENCODING, 'member': MEMBER, 'decoded_bytes': value['decoded_bytes'],
            'sha256': value['sha256'], 'compressed_bytes': len(body),
            'compressed_sha256': hashlib.sha256(body).hexdigest()}
    return decode_member(desc, body)

def reserve_next_pair(value):
    envelope = packed(value)
    if (envelope['decoded_bytes'] > MAX_DECODED - PAIR_DECODED_RESERVE
            or envelope['compressed_bytes'] > MAX_MEMBER - PAIR_COMPRESSED_RESERVE):
        raise ValueError('evidence capacity reserve exhausted; complete prefix retained')

def verify_source_binding(task_dir, name, preregistered_sha):
    actual = hashlib.sha256((task_dir / name).read_bytes()).hexdigest()
    if actual == preregistered_sha:
        return True
    if name != 'calibration.py':
        return False
    binding = json.loads((task_dir / 'judge/transport_binding.json').read_text())
    # This immutable release binding admits only this transport revision; it
    # never accepts arbitrary scientific files or edits the preregistration.
    return (binding['schema'] == 'ckks-transport-source-binding/v1'
            and binding['file'] == name
            and binding['preregistered_sha256'] == preregistered_sha
            and binding['transport_sha256'] == actual)
