"""CPU-controller envelopes; accepted-artifact consumer wiring remains gated."""
import base64
from dataclasses import asdict, dataclass
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile
import tempfile

from terminal_contract import Slot, canonical, validate_record

SCHEMA = 'frontier.vllm-controller-envelope-draft/v1'
MANIFEST_SCHEMA = 'frontier.vllm-controller-manifest-draft/v1'
DOMAIN = b'frontier-vllm-controller-terminal-draft-v1\n'


def digest(value):
    return 'sha256:' + hashlib.sha256(value).hexdigest()


def _bounded(value, maximum):
    if type(maximum) is not int or maximum <= 0:
        raise ValueError('explicit controller byte bound required')
    stack, nodes = [(value, 0)], 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > maximum or depth > 128:
            raise ValueError('controller structure exceeds bound')
        if isinstance(item, str) and len(item) > maximum:
            raise ValueError('controller string exceeds bound')
        if isinstance(item, dict):
            if len(item) > maximum:
                raise ValueError('controller object exceeds bound')
            stack.extend((value, depth + 1) for pair in item.items() for value in pair)
        elif isinstance(item, list):
            if len(item) > maximum:
                raise ValueError('controller array exceeds bound')
            stack.extend((value, depth + 1) for value in item)
    raw = canonical(value)
    if len(raw) > maximum:
        raise ValueError('controller serialization exceeds bound')
    return raw


@dataclass(frozen=True)
class Context:
    run_id: str
    attempt_id: str
    verification_id: str
    task_config_digest: str
    executable_image_digest: str
    source_content_digest: str
    contract_digest: str
    expected_slots_digest: str

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not isinstance(value, str) or not value or len(value.encode()) > 256:
                raise ValueError('bounded controller identity required')
            if name.endswith('_digest') and (len(value) != 71 or not value.startswith('sha256:') or
                                            any(c not in '0123456789abcdef' for c in value[7:])):
                raise ValueError('exact controller SHA256 binding required')


def slots_digest(slots):
    keys = [slot.key() for slot in slots]
    if not keys or len(set(keys)) != len(keys):
        raise ValueError('nonempty unique expected slots required')
    return digest(canonical(sorted(keys)))


def _validate_slots(context, slots, maximum_bytes):
    if type(maximum_bytes) is not int or maximum_bytes <= 0:
        raise ValueError('explicit expected-slot allocation bound required')
    if not isinstance(slots, (tuple, list)) or not 0 < len(slots) <= maximum_bytes:
        raise ValueError('bounded frozen expected-slot sequence required')
    used = 0
    for slot in slots:
        if not isinstance(slot, Slot):
            raise ValueError('typed frozen expected slots required')
        used += len(slot.key().encode())
        if used > maximum_bytes:
            raise ValueError('expected-slot set exceeds allocation bound')
    if slots_digest(slots) != context.expected_slots_digest or any(
            (slot.run_id, slot.attempt_id, slot.verification_id) !=
            (context.run_id, context.attempt_id, context.verification_id) for slot in slots):
        raise ValueError('expected slots differ from controller identity')


def _public(public_key):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    from cryptography.hazmat.primitives.serialization import load_ssh_public_key
    key = load_ssh_public_key(public_key.encode())
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError('controller key must be Ed25519')
    return key


def _key_id(public_key):
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
    return digest(_public(public_key).public_bytes(Encoding.Raw, PublicFormat.Raw))


class ControllerSigner:
    def __init__(self, context, slots, private_key, *, maximum_terminal_bytes,
                 maximum_manifest_bytes):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
        if not isinstance(private_key, Ed25519PrivateKey):
            raise ValueError('CPU-owned Ed25519 private key required')
        _validate_slots(context, slots, maximum_manifest_bytes)
        self.context, self.slots, self.private_key = context, tuple(slots), private_key
        self.maximum_terminal_bytes = maximum_terminal_bytes
        self.maximum_manifest_bytes = maximum_manifest_bytes
        self.public_key = private_key.public_key().public_bytes(Encoding.OpenSSH, PublicFormat.OpenSSH).decode()
        _bounded(self.manifest(), self.maximum_manifest_bytes)

    def manifest(self):
        return {'schema': MANIFEST_SCHEMA, 'context': asdict(self.context),
                'public_key': self.public_key, 'expected_slots': [asdict(slot) for slot in self.slots]}

    def sign(self, record):
        _bounded(record, self.maximum_terminal_bytes)
        validate_record(record)
        if Slot(**record['slot']).key() not in {slot.key() for slot in self.slots}:
            raise ValueError('terminal outside frozen controller slot set')
        payload = {'schema': SCHEMA, 'context': asdict(self.context),
                   'record': record, 'key_id': _key_id(self.public_key)}
        signature = self.private_key.sign(DOMAIN + _bounded(payload, self.maximum_terminal_bytes))
        envelope = {**payload, 'signature': base64.b64encode(signature).decode()}
        _bounded(envelope, self.maximum_terminal_bytes)
        return envelope


def verify_terminal(envelope, *, context, slots, public_key, maximum_bytes):
    """Anchor arguments must come from accepted CPU evidence, never child data."""
    _bounded(envelope, maximum_bytes)
    _validate_slots(context, slots, maximum_bytes)
    if (not isinstance(envelope, dict) or set(envelope) !=
            {'schema', 'context', 'record', 'key_id', 'signature'} or
            envelope['schema'] != SCHEMA or envelope['context'] != asdict(context) or
            envelope['key_id'] != _key_id(public_key)):
        raise ValueError('controller envelope identity mismatch')
    payload = {key: value for key, value in envelope.items() if key != 'signature'}
    signature = base64.b64decode(envelope['signature'], validate=True)
    _public(public_key).verify(signature, DOMAIN + canonical(payload))
    validate_record(envelope['record'])
    if Slot(**envelope['record']['slot']).key() not in {slot.key() for slot in slots}:
        raise ValueError('signed terminal outside expected slot set')
    return envelope['record']


def verify_terminal_bytes(raw, **arguments):
    """Apply the frame allocation cap before JSON parsing/signature validation."""
    maximum = arguments['maximum_bytes']
    if type(maximum) is not int or maximum <= 0 or not isinstance(raw, bytes) or len(raw) > maximum:
        raise ValueError('controller frame exceeds bound')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate controller JSON key')
            result[key] = value
        return result
    try:
        envelope = json.loads(raw.decode('utf-8'), object_pairs_hook=unique)
        return verify_terminal(envelope, **arguments)
    except (UnicodeError, RecursionError) as error:
        raise ValueError('invalid controller JSON frame') from error


def _manifest_value(found, context, maximum_manifest_bytes):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('duplicate controller manifest JSON key')
            value[key] = item
        return value
    value = json.loads(found, object_pairs_hook=unique)
    _bounded(value, maximum_manifest_bytes)
    if (not isinstance(value, dict) or set(value) !=
            {'schema', 'context', 'public_key', 'expected_slots'} or
            value['schema'] != MANIFEST_SCHEMA or value['context'] != asdict(context)):
        raise ValueError('accepted controller manifest identity mismatch')
    if not isinstance(value['expected_slots'], list) or len(value['expected_slots']) > maximum_manifest_bytes:
        raise ValueError('invalid bounded expected-slot set')
    slots = tuple(Slot(**slot) for slot in value['expected_slots'])
    _validate_slots(context, slots, maximum_manifest_bytes)
    _public(value['public_key'])
    return value


def _manifest_path(value):
    if (not isinstance(value, str) or not value or len(value.encode()) > 4096 or
            '\\' in value or '\x00' in value):
        raise ValueError('bounded trusted manifest path required')
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or str(path) != value:
        raise ValueError('manifest path must be canonical and archive-relative')
    return value


def _read_manifest(tar, maximum, manifest_path):
    manifest_path = _manifest_path(manifest_path)
    found = None
    for member in tar:
        if member.name == manifest_path:
            if found is not None or not member.isfile() or not 0 < member.size <= maximum:
                raise ValueError('invalid or duplicate controller manifest')
            with tar.extractfile(member) as stream:
                found = stream.read(maximum + 1)
    if found is None or len(found) > maximum:
        raise ValueError('missing bounded controller manifest')
    return found


def accepted_manifest(archive, *, accepted_sha256, accepted_size, context,
                      maximum_archive_bytes, maximum_manifest_bytes,
                      manifest_path='controller/manifest.json'):
    """Offline plain-tar fixture of the existing opaque Driver evidence binding.

    The accepted digest/size/context are trusted caller inputs. This function
    does not fetch or authenticate them; production accepted-result consumer
    wiring remains an integration gate. The zstd reader below uses the same binding.
    """
    if (any(type(value) is not int or value <= 0 for value in
            (accepted_size, maximum_archive_bytes, maximum_manifest_bytes)) or
            not isinstance(archive, bytes) or len(archive) != accepted_size or
            len(archive) > maximum_archive_bytes or digest(archive) != accepted_sha256):
        raise ValueError('archive differs from accepted Driver artifact')
    with tarfile.open(fileobj=io.BytesIO(archive), mode='r:') as tar:
        found = _read_manifest(tar, maximum_manifest_bytes, manifest_path)
    return _manifest_value(found, context, maximum_manifest_bytes)


def accepted_zstd_manifest(path, *, accepted_sha256, accepted_size, context,
                           maximum_archive_bytes, maximum_expanded_bytes,
                           maximum_manifest_bytes,
                           manifest_path='controller/manifest.json'):
    """Read publisher-format evidence after its opaque accepted binding is checked.

    Caller must obtain digest/size/context from the accepted Driver result. An
    arbitrary downloaded artifact or its embedded key is not a trust anchor.
    manifest_path is derived from the CPU phase layout by the trusted reader;
    it must never be selected from an envelope or other remote child output.
    """
    import zstandard
    if any(type(n) is not int or n <= 0 for n in (accepted_size, maximum_archive_bytes,
            maximum_expanded_bytes, maximum_manifest_bytes)) or accepted_size > maximum_archive_bytes:
        raise ValueError('explicit accepted archive resource bounds required')
    # Decompress only a private snapshot of the exact hashed bytes, including
    # when another process modifies the downloaded file in place.
    with Path(path).open('rb') as source, tempfile.TemporaryFile() as raw:
        hashed, observed = hashlib.sha256(), 0
        while True:
            part = source.read(min(65536, accepted_size + 1 - observed))
            if not part:
                break
            observed += len(part)
            if observed > accepted_size:
                raise ValueError('archive exceeds accepted Driver size')
            hashed.update(part)
            raw.write(part)
        if observed != accepted_size or 'sha256:' + hashed.hexdigest() != accepted_sha256:
            raise ValueError('archive differs from accepted Driver artifact')
        raw.seek(0)
        with zstandard.ZstdDecompressor().stream_reader(raw, closefd=False) as expanded:
            class BoundedReader:
                def __init__(self):
                    self.count = 0
                def read(self, size=-1):
                    remaining = maximum_expanded_bytes - self.count
                    data = expanded.read(min(65536 if size < 0 else size, remaining + 1))
                    self.count += len(data)
                    if self.count > maximum_expanded_bytes:
                        raise ValueError('expanded evidence exceeds bound')
                    return data
            bounded = BoundedReader()
            with tarfile.open(fileobj=bounded, mode='r|') as tar:
                found = _read_manifest(tar, maximum_manifest_bytes, manifest_path)
            while bounded.read(65536):
                pass
    return _manifest_value(found, context, maximum_manifest_bytes)
