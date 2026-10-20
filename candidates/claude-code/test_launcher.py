import importlib.util
import json
import hashlib
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import tempfile
import os
spec=importlib.util.spec_from_file_location('launcher',Path(__file__).with_name('launcher.py'))
launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)

class Upload:
    def __init__(self,fail=False):self.used=0;self.calls=[];self.fail=fail
    def request(self,method,path,data,headers=None):
        self.calls.append((method,path,data,headers))
        return not self.fail

class LauncherTests(unittest.TestCase):
    def test_redacts_secrets_across_every_read_boundary(self):
        secret=b'synthetic-key-canary'
        raw=b'before '+secret+b' after '+secret+b'\n'
        for cut in range(1,len(raw)):
            r=launcher.Redactor([secret]);out=r.feed(raw[:cut])+r.feed(raw[cut:])+r.feed(b'',final=True)
            self.assertEqual(out,b'before [REDACTED] after [REDACTED]\n')

    def test_chunks_and_completion_match_transmitted_redacted_bytes(self):
        u=Upload();s=launcher.Stream('events',u,[b'canary-secret'])
        s.feed(b'canary-secret\n'+b'x'*100000,final=True);s.complete(7)
        chunks=[c[2] for c in u.calls if c[0]=='PUT']
        self.assertTrue(all(len(c)<=65536 for c in chunks))
        self.assertNotIn(b'canary-secret',b''.join(chunks))
        complete=json.loads(u.calls[-1][2]);self.assertEqual(complete['exit_code'],7)
        self.assertEqual(complete['sha256'],'sha256:'+hashlib.sha256(b''.join(chunks)).hexdigest())
        self.assertEqual(complete['chunks'],len(chunks))

    def test_failed_upload_is_bounded_and_marks_truncated(self):
        u=Upload(fail=True);s=launcher.Stream('stderr',u,[])
        for _ in range(100):s.feed(b'x'*16384)
        self.assertTrue(s.truncated);self.assertLessEqual(len(s.pending),65536)
        self.assertEqual(len(u.calls),1)
        s.complete(1);self.assertTrue(json.loads(u.calls[-1][2])['truncated'])

    def test_capture_separates_stderr_and_preserves_nonzero_exit(self):
        p=subprocess.Popen([sys.executable,'-c','import sys;print(\'{"type":"result","usage":{"input_tokens":4}}\');print("synthetic-secret",file=sys.stderr);sys.exit(9)'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        u=Upload();self.assertEqual(launcher.capture(p,u,[b'synthetic-secret']),9)
        self.assertTrue(any('/events/' in c[1] for c in u.calls))
        self.assertTrue(any('/stderr/' in c[1] for c in u.calls))
        self.assertNotIn(b'synthetic-secret',b''.join(c[2] for c in u.calls))
        self.assertEqual([json.loads(c[2])['exit_code'] for c in u.calls if c[0]=='POST'],[9,9])

    def test_api_key_environment_keeps_auxiliary_caller_auth(self):
        with tempfile.TemporaryDirectory() as workspace, patch.object(launcher, 'WORKSPACE', Path(workspace)), patch.object(Path, 'stat', return_value=SimpleNamespace(st_mode=0o100600, st_size=20)), patch.object(Path, 'read_text', return_value='synthetic-driver-bearer'), patch.dict(os.environ, {
            'ANTHROPIC_API_KEY':'sk-ant-api03-synthetic', 'FRONTIER_CLAUDE_MODEL':'claude-sonnet-4-6',
            'FRONTIER_DRIVER_API_URL':'http://127.0.0.1:8080', 'FRONTIER_MODEL_PROXY_TOKEN':'synthetic-driver-bearer',
            'FRONTIER_LITELLM_INVENTORY':'/run/secrets/litellm/inventory.json',
            'ANTHROPIC_BASE_URL':'https://must-remove.invalid', 'HTTPS_PROXY':'must-remove'
        }, clear=True):
            child, _, _, _, _ = launcher.prepare_environment()
            self.assertEqual(child['FRONTIER_MODEL_PROXY_TOKEN'], 'synthetic-driver-bearer')
            self.assertIn('FRONTIER_LITELLM_INVENTORY',child)
            self.assertEqual(child['ANTHROPIC_API_KEY'], 'sk-ant-api03-synthetic')
            self.assertNotIn('ANTHROPIC_BASE_URL',child)
            self.assertNotIn('HTTPS_PROXY',child)

    def test_command_is_official_cli_with_no_bare_or_api_credentials(self):
        command=launcher.command('claude-sonnet-4-6')
        self.assertNotIn('--bare',command)
        self.assertIn('--include-partial-messages',command)
        self.assertIn('--strict-mcp-config',command)
        self.assertFalse(any('sk-ant' in item for item in command))

    def test_effort_is_optional_and_comes_from_the_candidate_environment(self):
        self.assertNotIn('--effort',launcher.command('claude-opus-5-5'))
        command=launcher.command('claude-opus-5-5','max')
        self.assertEqual(command[command.index('--effort')+1],'max')
        self.assertEqual(command[command.index('--model')+1],'claude-opus-5-5')

if __name__=='__main__':unittest.main()
