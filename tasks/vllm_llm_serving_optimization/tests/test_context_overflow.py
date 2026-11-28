"""CPU-only: a SWE conversation outgrowing the fixed context window costs one case.

    uv run --no-project --with openai==3.13.0 python -m unittest discover -s tests -v

(run from the task directory). Each episode runs through the real agent loop and
the judges' ModelClient trace, then the verifier's terminal classification and
containment policy.
"""
from __future__ import annotations

import hashlib
import json
import sys
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

TASK = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(TASK / 'judge'), str(TASK / 'verifier')]

import httpx2 as httpx  # noqa: E402  (openai 3.x transport)
import openai  # noqa: E402

from batch_coordinator import ModelClient, _compact_record_trace  # noqa: E402
from containment_policy import classify_failure  # noqa: E402
from relay_client import remote_terminal_classification  # noqa: E402
from serving_eval import agent_runner  # noqa: E402
from serving_eval.settings import EpisodeSettings  # noqa: E402

REQUEST = httpx.Request('POST', 'http://candidate/v1/chat/completions')
# vLLM 0.11.0 serving_engine._validate_input, max_model_len 32768, max_tokens 2048.
OVERFLOW = ("'max_tokens' or 'max_completion_tokens' is too large: 2048. This model's maximum "
            "context length is 32768 tokens and your request has 31000 input tokens "
            "(2048 > 32768 - 31000).")
PATCH = 'diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n'


def status_error(status, message):
    response = httpx.Response(status, request=REQUEST, json={
        'object': 'error', 'message': message, 'type': 'BadRequestError', 'param': None,
        'code': status})
    return openai.OpenAI(api_key='EMPTY', base_url='http://candidate/v1')._make_status_error_from_response(response)


class Completions:
    def __init__(self, failure):
        self.failure, self.calls = failure, 0

    def create(self, **_):
        self.calls += 1
        if self.calls > 1:
            raise self.failure
        message = SimpleNamespace(content='```bash\necho edit\n```')
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], model_dump=lambda: {'ok': True})


class Testbed:
    def __init__(self, patch):
        self.patch = patch

    def run(self, command, *, timeout):
        return 0, ''

    def read_patch(self):
        return self.patch


def episode(failure, patch):
    """One SWE record as `batch_coordinator`'s direct path builds it."""
    trace = []
    client = ModelClient.__new__(ModelClient)
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=Completions(failure)))
    client.api_status_error, client.trace = openai.APIStatusError, trace
    client.chat = SimpleNamespace(completions=client)
    outcome = agent_runner.run_instance({'instance_id': 'repo__case-1', 'problem_statement': 'bug'},
                                        client=client, prepared_testbed=Testbed(patch),
                                        settings=EpisodeSettings())
    record = {'instance_id': 'repo__case-1', 'trace': trace, 'episode': asdict(outcome),
              'patch_sha256': 'sha256:' + hashlib.sha256(outcome.patch.encode()).hexdigest(),
              'requires_error_attribution': any('error_origin' in event for event in trace)}
    _compact_record_trace(record)
    return json.loads(json.dumps(record))  # what the relay receives


def decide(record):
    boundary, failed = remote_terminal_classification('batch', 'swe/high', record)
    decision = classify_failure(SimpleNamespace(), boundary, candidate_input_dispatched=True)
    return boundary, failed, decision


class ContextOverflow(unittest.TestCase):
    def test_vllm_wording_is_recognised(self):
        self.assertTrue(agent_runner._is_context_overflow(status_error(400, OVERFLOW)))
        prompt_only = ("This model's maximum context length is 32768 tokens. However, your "
                       "request has 33000 input tokens. Please reduce the length of the input messages.")
        self.assertTrue(agent_runner._is_context_overflow(status_error(400, prompt_only)))
        self.assertFalse(agent_runner._is_context_overflow(status_error(400, 'bad sampling params')))
        self.assertFalse(agent_runner._is_context_overflow(status_error(500, OVERFLOW)))

    def test_overflow_without_patch_fails_only_its_case(self):
        record = episode(status_error(400, OVERFLOW), '')
        self.assertEqual(record['episode']['exit_status'], 'incomplete')
        self.assertEqual(record['episode']['error'], 'BadRequestError')
        self.assertEqual([e['error_origin'] for e in record['trace'] if 'error_origin' in e],
                         ['model_api_response'])
        boundary, failed, decision = decide(record)
        self.assertTrue(failed)
        # `candidate_projection.timed_case_failures` takes exactly this as a C2 case failure.
        self.assertEqual(boundary.kind, 'swe_case')
        self.assertEqual((decision['disposition'], decision['affected_scope'],
                          decision['scientific_effect']), ('case_fail', 'case', 'case_incorrect'))

    def test_overflow_with_partial_patch_is_limit_with_patch(self):
        record = episode(status_error(400, OVERFLOW), PATCH)
        self.assertEqual(record['episode']['exit_status'], 'limit_with_patch')
        self.assertEqual(record['episode']['patch'], PATCH)
        boundary, failed, decision = decide(record)
        self.assertTrue(failed)
        self.assertEqual((boundary.kind, decision['affected_scope']), ('swe_case', 'case'))

    def test_other_model_api_errors_stay_group_no_credit(self):
        for failure in (status_error(500, 'engine dead'), status_error(400, 'bad sampling params'),
                        openai.APIConnectionError(request=REQUEST)):
            with self.subTest(error=type(failure).__name__):
                record = episode(failure, '')
                self.assertEqual(record['episode']['exit_status'], 'api_error')
                boundary, failed, decision = decide(record)
                self.assertTrue(failed)
                if isinstance(failure, openai.APIStatusError):
                    self.assertEqual(boundary.kind, 'candidate_service')
                    self.assertEqual((decision['affected_scope'], decision['scientific_effect']),
                                     ('workload_group', 'evaluation_fail_affected_scope_no_credit'))
                else:
                    # A client-side transport error is not a model API response; this
                    # change leaves its existing (case-level) classification alone.
                    self.assertEqual(boundary.kind, 'swe_case')


if __name__ == '__main__':
    unittest.main()
