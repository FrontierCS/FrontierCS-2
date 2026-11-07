"""Driver sidecar environment mapping; keys are generated per test."""
import json
import unittest

from serving_eval import sidecar_env
from serving_eval.tool_auth import ToolAuthSigner, ToolAuthVerifier, _key_id
from test_attestation import keypair

IDENTITY = {'run_id': 'run', 'attempt_id': 'attempt', 'operation_id': 'runner_one', 'phase': 'eval',
            'name': 'swe-003-check', 'profile_digest': 'sha256:' + '1' * 64}


class SidecarEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.private, self.public = keypair()

    def environment(self, **extra):
        value = {sidecar_env.IDENTITY_ENV: json.dumps(IDENTITY), sidecar_env.PUBLIC_KEY_ENV: self.public,
                 'PATH': '/usr/bin'}
        value.update(extra)
        return value

    def test_verifier_mapping_feeds_tool_auth_and_removes_driver_variables(self):
        environment = self.environment()
        identity = sidecar_env.install('verifier', environment)
        self.assertEqual(identity, IDENTITY)
        self.assertNotIn(sidecar_env.IDENTITY_ENV, environment)
        self.assertNotIn(sidecar_env.PUBLIC_KEY_ENV, environment)
        self.assertNotIn('FRONTIER_TOOL_AUTH_PRIVATE_KEY_OPENSSH', environment)
        self.assertEqual(environment['FRONTIER_TOOL_AUTH_KEY_ID'], _key_id(self.public))
        context = json.loads(environment['FRONTIER_TOOL_AUTH_CONTEXT_JSON'])
        self.assertEqual(context, {'schema_version': 'frontier.tool-auth-context/v1', 'run_id': 'run',
                                   'attempt_id': 'attempt', 'operation_id': 'runner_one',
                                   'sidecar_id': 'swe-003-check'})
        verifier = ToolAuthVerifier(environment['FRONTIER_TOOL_AUTH_PUBLIC_KEY_OPENSSH'],
                                    environment['FRONTIER_TOOL_AUTH_KEY_ID'],
                                    environment['FRONTIER_TOOL_AUTH_CONTEXT_JSON'])
        signer = ToolAuthSigner(self.private, self.public, _key_id(self.public),
                                {**context, 'sidecar_id': 'scheduler'})
        verifier.verify(signer.headers('swe-003-check', 'POST', '/correctness', b'{}'), 'POST', '/correctness', b'{}')
        sidecar_env.scrub(environment)
        self.assertEqual(set(environment), {'PATH'})

    def test_signer_requires_and_maps_the_private_key(self):
        environment = self.environment(**{sidecar_env.PRIVATE_KEY_ENV: self.private,
                                          sidecar_env.IDENTITY_ENV: json.dumps({**IDENTITY, 'name': 'scheduler'})})
        sidecar_env.install('signer', environment)
        self.assertEqual(environment['FRONTIER_TOOL_AUTH_PRIVATE_KEY_OPENSSH'], self.private)
        self.assertNotIn(sidecar_env.PRIVATE_KEY_ENV, environment)
        with self.assertRaises(sidecar_env.SidecarEnvironmentError):
            sidecar_env.install('signer', self.environment())

    def test_misdeclared_or_malformed_environments_fail_closed(self):
        with self.assertRaisesRegex(sidecar_env.SidecarEnvironmentError, 'private key'):
            sidecar_env.install('verifier', self.environment(**{sidecar_env.PRIVATE_KEY_ENV: self.private}))
        with self.assertRaises(sidecar_env.SidecarEnvironmentError):
            sidecar_env.install('verifier', {sidecar_env.PUBLIC_KEY_ENV: self.public})
        for bad in ({**IDENTITY, 'phase': 'other'}, {**IDENTITY, 'sidecar_id': 'x'},
                    {k: v for k, v in IDENTITY.items() if k != 'name'}, {**IDENTITY, 'name': 'a\nb'}):
            with self.subTest(bad=bad), self.assertRaises(sidecar_env.SidecarEnvironmentError):
                sidecar_env.install('verifier', self.environment(**{sidecar_env.IDENTITY_ENV: json.dumps(bad)}))
        with self.assertRaises(sidecar_env.SidecarEnvironmentError):
            sidecar_env.install('verifier', self.environment(FRONTIER_TOOL_AUTH_KEY_ID='preset'))
        with self.assertRaises(sidecar_env.SidecarEnvironmentError):
            sidecar_env.install('other', self.environment())


if __name__ == '__main__':
    unittest.main()
