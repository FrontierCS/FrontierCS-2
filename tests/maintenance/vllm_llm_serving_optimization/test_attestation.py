"""Real Ed25519 envelope signing and binding; keys are generated per test."""
import copy
import unittest
from unittest.mock import patch

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat, PublicFormat

import attestation


def keypair():
    private = Ed25519PrivateKey.generate()
    return (private.private_bytes(Encoding.PEM, PrivateFormat.OpenSSH, NoEncryption()).decode(),
            private.public_key().public_bytes(Encoding.OpenSSH, PublicFormat.OpenSSH).decode())


IDENTITY = {'run_id': 'run', 'attempt_id': 'attempt', 'operation_id': 'runner_one', 'phase': 'eval',
            'name': 'scheduler', 'profile_digest': 'sha256:' + '1' * 64}


class AttestationTests(unittest.TestCase):
    def setUp(self):
        self.private, self.public = keypair()
        self.signer = attestation.Signer(self.private, self.public, IDENTITY)

    def sign(self, **overrides):
        fields = dict(kind='batch', sequence=1, request_sha256='sha256:' + 'a' * 64,
                      payload={'variant': 'reference', 'batch': {'episodes': [1, 2]}})
        fields.update(overrides)
        return self.signer.sign(**fields)

    def test_trusted_crypto_backend_failure_is_not_a_bad_signature(self):
        signed = self.sign()
        for error_type in (OSError, RuntimeError, MemoryError):
            with self.subTest(error_type=error_type):
                with patch.object(attestation, 'load_ssh_public_key', side_effect=error_type('CPU crypto failed')):
                    with self.assertRaisesRegex(error_type, 'CPU crypto failed'):
                        attestation.verify(signed, self.public)

    def test_roundtrip_binds_operation_phase_name_kind_sequence_and_request(self):
        signed = self.sign()
        envelope = attestation.verify(signed, self.public)
        payload = attestation.bind(envelope, operation_id='runner_one', phase='eval', name='scheduler',
                                   kind='batch', sequence=1, request_sha256='sha256:' + 'a' * 64)
        self.assertEqual(payload['variant'], 'reference')
        for field, value in (('operation_id', 'runner_two'), ('phase', 'train'), ('name', 'swe-000-check'),
                             ('kind', 'correctness'), ('sequence', 2), ('request_sha256', 'sha256:' + 'b' * 64)):
            bound = dict(operation_id='runner_one', phase='eval', name='scheduler', kind='batch', sequence=1,
                         request_sha256='sha256:' + 'a' * 64)
            bound[field] = value
            with self.subTest(field=field), self.assertRaises(attestation.AttestationError):
                attestation.bind(envelope, **bound)

    def test_tampered_payload_signature_or_other_key_is_rejected(self):
        signed = self.sign()
        tampered = copy.deepcopy(signed)
        tampered['envelope']['payload']['batch']['episodes'].append(3)
        with self.assertRaisesRegex(attestation.AttestationError, 'signature'):
            attestation.verify(tampered, self.public)
        resequenced = copy.deepcopy(signed)
        resequenced['envelope']['sequence'] = 2
        with self.assertRaises(attestation.AttestationError):
            attestation.verify(resequenced, self.public)
        _, other_public = keypair()
        with self.assertRaisesRegex(attestation.AttestationError, 'another key'):
            attestation.verify(signed, other_public)
        forged = copy.deepcopy(signed)
        forged['key_id'] = attestation.key_id(other_public)
        with self.assertRaises(attestation.AttestationError):
            attestation.verify(forged, other_public)
        for broken in ({**signed, 'extra': 1}, {**signed, 'signature': signed['signature'] + 'AA'},
                       {**signed, 'envelope': {**signed['envelope'], 'kind': 'other'}}):
            with self.assertRaises(attestation.AttestationError):
                attestation.verify(broken, self.public)

    def test_signer_requires_matching_keys_and_complete_identity(self):
        _, other_public = keypair()
        with self.assertRaises(attestation.AttestationError):
            attestation.Signer(self.private, other_public, IDENTITY)
        for bad in ({**IDENTITY, 'phase': 'other'}, {k: v for k, v in IDENTITY.items() if k != 'run_id'},
                    {**IDENTITY, 'token': 'x'}):
            with self.assertRaises(attestation.AttestationError):
                attestation.Signer(self.private, self.public, bad)
        with self.assertRaises(attestation.AttestationError):
            self.sign(kind='exec')
        with self.assertRaises(attestation.AttestationError):
            self.sign(payload=[1])
        with self.assertRaises(attestation.AttestationError):
            self.sign(request_sha256='abc')

    def test_canonical_bytes_match_core_profile_digests(self):
        # serde_json's canonical object form: sorted keys, no spaces, UTF-8 verbatim.
        value = {'b': [1, 'ü'], 'a': {'y': True, 'x': None}}
        self.assertEqual(attestation.canonical_bytes(value), '{"a":{"x":null,"y":true},"b":[1,"ü"]}'.encode())
        self.assertEqual(attestation.canonical_sha256({})[:7], 'sha256:')


if __name__ == '__main__':
    unittest.main()
