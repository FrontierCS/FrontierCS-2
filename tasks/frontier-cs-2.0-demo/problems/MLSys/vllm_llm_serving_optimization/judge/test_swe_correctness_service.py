"""Signed correctness service over real localhost HTTP with a synthetic worker."""
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest import mock
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

from serving_eval import swe_correctness_service as service
from serving_eval import sidecar_env
from serving_eval.accuracy import CorrectnessRuntimeError
from serving_eval.testbed_reset import ResetError
from serving_eval.tool_auth import ToolAuthSigner, ToolAuthVerifier, _key_id
from test_attestation import keypair


class FakeReset:
    def __init__(self):
        self.snapshots = 0
        self.resets = 0
        self.fail = False

    def snapshot(self):
        self.snapshots += 1
        return {'head': 'abc', 'ignored_files': 2}

    def reset(self, *, deadline=None):
        if self.fail:
            raise ResetError('synthetic reset failure')
        self.resets += 1
        return {'head': 'abc', 'removed_changed_or_new_ignored': self.resets}


class FakeWorker:
    calls = []

    def __init__(self, *, instance_id, testbed, max_output_bytes):
        self.instance_id = instance_id

    def run_correctness(self, request):
        FakeWorker.calls.append(request)
        if request.get('candidate_patch') == 'reject':
            raise CorrectnessRuntimeError('request differs from fixed canonical correctness instructions')
        return {'instance_id': self.instance_id, 'request_sha256': request['request_sha256'],
                'execution': 'complete', 'log': 'PASSED test\n'}


class CorrectnessServiceTests(unittest.TestCase):
    def setUp(self):
        FakeWorker.calls = []
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.testbed = Path(self.temp.name)
        (self.testbed / '.git').mkdir()
        private, public = keypair()
        context = {'schema_version': 'frontier.tool-auth-context/v1', 'run_id': 'run', 'attempt_id': 'attempt',
                   'operation_id': 'runner_one', 'sidecar_id': 'scheduler'}
        self.signer = ToolAuthSigner(private, public, _key_id(public), context)
        self.identity = {'operation_id': 'runner_one', 'group_id': 'runner_one', 'placement_id': 'runner_one',
                         'instance_id': 'sympy__sympy-21847', 'child_id': 'swe-000-check', 'phase': 'train'}
        verifier = ToolAuthVerifier(public, _key_id(public), {**context, 'sidecar_id': 'swe-000-check'})
        self.reset = FakeReset()
        self.cleanups = []
        self.service = service.CorrectnessService(self.identity, max_request_bytes=65536, max_output_bytes=65536,
                                                  auth=verifier, testbed=self.testbed, reset=self.reset,
                                                  worker_factory=FakeWorker,
                                                  process_cleanup=lambda: self.cleanups.append(1))
        self.server = service.make_server(self.service, listen_profile='bridge', host='0.0.0.0', port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval': 0.01})
        self.thread.start()
        self.addCleanup(self.close)
        self.endpoint = 'http://127.0.0.1:' + str(self.server.server_port)

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def request(self, method, route, body=None, signed=True):
        data = b'' if body is None else json.dumps(body).encode()
        headers = self.signer.headers('swe-000-check', method, route, data) if signed else {}
        if body is not None:
            headers['Content-Type'] = 'application/json'
        request = Request(self.endpoint + route, data=None if body is None else data, headers=headers, method=method)
        try:
            response = build_opener(ProxyHandler({})).open(request, timeout=5)
        except HTTPError as error:
            response = error
        with response:
            return response.status, json.loads(response.read())

    def body(self, variant, patch=''):
        return {'identity': dict(self.identity), 'variant': variant,
                'request': {'instance_id': 'sympy__sympy-21847', 'candidate_patch': patch,
                            'timeout_seconds': 1200,
                            'request_sha256': 'sha256:' + variant.ljust(64, '0')}}

    def test_candidate_only_first_request_resets_without_fake_prior_runs(self):
        self.service.variants = ('candidate',)
        self.service.prepare()
        self.assertEqual(self.service.completed, [])
        status, result = self.service.correctness(self.body('candidate'))
        self.assertEqual(status, 200)
        self.assertEqual(result['run_index'], 2)
        self.assertIsInstance(result['reset'], dict)
        self.assertEqual(self.service.completed, ['candidate'])
        self.assertEqual(self.service.correctness(self.body('vanilla'))[0], 409)

    def test_three_variants_run_in_order_with_resets_between_them(self):
        self.assertEqual(self.request('GET', '/ready')[1]['ready'], False)
        self.service.prepare()
        self.assertEqual(self.request('GET', '/ready')[1], {'identity': self.identity, 'ready': True,
                                                            'completed_variants': []})
        self.assertEqual(self.request('POST', '/correctness', self.body('reference'))[0], 409)
        status, result = self.request('POST', '/correctness', self.body('vanilla'))
        self.assertEqual((status, result['run_index'], result['reset']), (200, 0, None))
        self.assertEqual(result['receipt']['execution'], 'complete')
        self.assertEqual(self.request('POST', '/correctness', self.body('vanilla'))[0], 409)
        status, result = self.request('POST', '/correctness', self.body('reference'))
        self.assertEqual((status, result['run_index'], result['reset']['removed_changed_or_new_ignored']), (200, 1, 1))
        status, result = self.request('POST', '/correctness', self.body('candidate'))
        self.assertEqual((status, result['run_index'], result['variant']), (200, 2, 'candidate'))
        self.assertEqual(self.request('POST', '/correctness', self.body('candidate'))[0], 409)
        self.assertEqual(self.reset.resets, 2)
        self.assertEqual(len(self.cleanups), 2 + 3, 'strays are killed before every reset and after every run')
        self.assertEqual([request['request_sha256'][-6:] for request in FakeWorker.calls][0][:1], '0')
        self.assertEqual(self.request('GET', '/ready')[1]['completed_variants'], ['vanilla', 'reference', 'candidate'])

    def test_identity_signature_and_request_shape_are_enforced(self):
        self.service.prepare()
        self.assertEqual(self.request('POST', '/correctness', self.body('vanilla'), signed=False)[0], 401)
        wrong = self.body('vanilla')
        wrong['identity']['child_id'] = 'swe-001-check'
        self.assertEqual(self.request('POST', '/correctness', wrong)[0], 409)
        other = self.body('vanilla')
        other['request']['instance_id'] = 'django__django-13925'
        self.assertEqual(self.request('POST', '/correctness', other)[0], 400)
        self.assertEqual(self.request('POST', '/correctness', {'identity': self.identity, 'variant': 'vanilla'})[0], 400)
        self.assertEqual(self.request('POST', '/execute', {'identity': self.identity, 'command': 'true',
                                                           'timeout_seconds': 1})[0], 404)
        self.assertEqual(FakeWorker.calls, [])

    def test_worker_rejection_and_reset_failure_do_not_advance_the_order(self):
        self.service.prepare()
        status, result = self.request('POST', '/correctness', self.body('vanilla', patch='reject'))
        self.assertEqual((status, result['error_type']), (422, 'CorrectnessRuntimeError'))
        self.assertEqual(self.request('GET', '/ready')[1]['completed_variants'], [])
        self.assertEqual(self.request('POST', '/correctness', self.body('vanilla'))[0], 200)
        self.reset.fail = True
        status, result = self.request('POST', '/correctness', self.body('reference'))
        self.assertEqual((status, result['error_type']), (503, 'ResetError'))
        self.assertEqual(self.request('GET', '/ready')[1]['completed_variants'], ['vanilla'])
        self.reset.fail = False
        self.assertEqual(self.request('POST', '/correctness', self.body('reference'))[0], 200)

    def test_service_requires_matching_auth_context_and_known_instance(self):
        private, public = keypair()
        foreign = ToolAuthVerifier(public, _key_id(public), {'schema_version': 'frontier.tool-auth-context/v1',
                                   'run_id': 'run', 'attempt_id': 'attempt', 'operation_id': 'runner_two',
                                   'sidecar_id': 'swe-000-check'})
        with self.assertRaises(ValueError):
            service.CorrectnessService(self.identity, max_request_bytes=1, max_output_bytes=1, auth=foreign,
                                       testbed=self.testbed, reset=self.reset)
        with self.assertRaises(ValueError):
            service.CorrectnessService(dict(self.identity, instance_id='unknown'), max_request_bytes=1,
                                       max_output_bytes=1, auth=self.service.auth, testbed=self.testbed, reset=self.reset)


class CorrectnessStartupTests(unittest.TestCase):
    def test_main_retains_driver_identity_after_auth_environment_is_scrubbed(self):
        _, public = keypair()
        driver_identity = {'run_id': 'run', 'attempt_id': 'attempt', 'operation_id': 'runner',
                           'phase': 'train', 'name': 'swe-001-check', 'profile_digest': 'sha256:' + 'a' * 64}
        expected = {'operation_id': 'runner', 'group_id': 'runner', 'placement_id': 'runner',
                    'instance_id': 'django__django-13925', 'child_id': 'swe-001-check', 'phase': 'train'}
        with tempfile.TemporaryDirectory() as directory:
            identity_file = Path(directory) / 'identity.json'
            argv = ['correctness-service', '--driver-identity-file', str(identity_file),
                    '--instance-id', expected['instance_id'], '--host', '0.0.0.0', '--port', '8001',
                    '--max-request-bytes', '65536', '--max-output-bytes', '65536']
            environment = {sidecar_env.IDENTITY_ENV: json.dumps(driver_identity),
                           sidecar_env.PUBLIC_KEY_ENV: public}
            with mock.patch.dict(os.environ, environment, clear=True), mock.patch.object(sys, 'argv', argv), \
                    mock.patch.object(service, 'CorrectnessService') as factory, \
                    mock.patch.object(service, 'make_server') as server:
                service.main()
                self.assertEqual(json.loads(identity_file.read_text()), expected)
                self.assertEqual(factory.call_args.args[0], expected)
                self.assertEqual(factory.call_args.kwargs['auth'].context['sidecar_id'], expected['child_id'])
                factory.return_value.prepare.assert_called_once_with()
                server.return_value.__enter__.return_value.serve_forever.assert_called_once_with()
                for name in (sidecar_env.IDENTITY_ENV, sidecar_env.PRIVATE_KEY_ENV, sidecar_env.PUBLIC_KEY_ENV,
                             *sidecar_env.TOOL_AUTH_VARIABLES):
                    self.assertNotIn(name, os.environ)


if __name__ == '__main__':
    unittest.main()
