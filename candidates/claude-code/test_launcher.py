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

class LimitedUpload(Upload):
    """A Driver whose ordinary room is used up after `room` chunks; only the marked tail still fits."""
    def __init__(self,room,tail=True):super().__init__();self.room=room;self.tail=tail
    def request(self,method,path,data,headers=None):
        self.calls.append((method,path,data,headers))
        if method!='PUT':return True
        if (headers or {}).get('x-frontier-trace-tail')=='1':return self.tail
        self.room-=1;return self.room>=0

RESULT=b'{"type":"result","parent_tool_use_id":null,"usage":{"input_tokens":4,"output_tokens":9},"total_cost_usd":1.5}\n'

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

    def test_timed_flush_sends_the_first_chunk_at_once_then_coalesces_small_ones(self):
        u=Upload();s=launcher.Stream('events',u,[]);now=s.flushed
        s.feed(b'{"type":"system"}\n');s.tick(now)
        self.assertEqual(len(u.calls),1)
        s.feed(b'a'*100);s.tick(s.flushed+2);s.tick(s.flushed+launcher.MAX_DELAY-1)
        self.assertEqual(len(u.calls),1)
        s.tick(s.flushed+launcher.MAX_DELAY)
        self.assertEqual([len(c[2]) for c in u.calls],[18,100])
        s.feed(b'b'*launcher.MIN_FLUSH);s.tick(s.flushed+2)
        self.assertEqual(len(u.calls[-1][2]),launcher.MIN_FLUSH)
        s.feed(b'tail',final=True)
        self.assertEqual(u.calls[-1][2],b'tail');self.assertFalse(s.pending)

    def test_truncated_capture_still_delivers_the_final_cli_result(self):
        u=LimitedUpload(room=2);s=launcher.Stream('events',u,[b'canary-secret'])
        s.feed(b'{"type":"system"}\n'+b'x'*(3*launcher.CHUNK))
        self.assertTrue(s.truncated)
        # Arrives in pieces after the capture stopped, and is still redacted.
        s.feed(b'yyy canary-secret\n'+RESULT[:20]);s.feed(RESULT[20:],final=True);s.complete(0)
        puts=[c for c in u.calls if c[0]=='PUT'];tail=puts[-1]
        self.assertEqual(tail[3]['x-frontier-trace-tail'],'1')
        self.assertEqual(tail[1],'/v1/agent-traces/events/2')
        self.assertEqual(tail[2],b'\n'+RESULT)
        self.assertNotIn(b'canary-secret',b''.join(c[2] for c in puts))
        accepted=puts[:2]+[tail];complete=json.loads(u.calls[-1][2])
        self.assertEqual((complete['chunks'],complete['truncated']),(3,True))
        self.assertEqual(complete['sha256'],'sha256:'+hashlib.sha256(b''.join(c[2] for c in accepted)).hexdigest())

    def test_result_tail_is_sent_only_when_the_result_was_cut_off(self):
        # Not truncated: nothing extra.
        u=Upload();s=launcher.Stream('events',u,[]);s.feed(RESULT,final=True);s.complete(0)
        self.assertEqual([c[2] for c in u.calls if c[0]=='PUT'],[RESULT])
        # Already uploaded before a later chunk failed: no second result.
        u=LimitedUpload(room=1);s=launcher.Stream('events',u,[]);s.feed(RESULT);s.flush()
        s.feed(b'trailing\n',final=True);s.complete(0)
        self.assertTrue(s.truncated)
        self.assertFalse(any((c[3] or {}).get('x-frontier-trace-tail') for c in u.calls))
        # A Driver without tail room: the completion still matches what it accepted.
        u=LimitedUpload(room=0,tail=False);s=launcher.Stream('events',u,[]);s.feed(RESULT,final=True);s.complete(0)
        self.assertEqual(json.loads(u.calls[-1][2])['chunks'],0)
        # stderr never has a tail; subagent results and oversized lines are not the result.
        u=LimitedUpload(room=0);s=launcher.Stream('stderr',u,[]);s.feed(RESULT,final=True);s.complete(1)
        self.assertEqual(len([c for c in u.calls if c[0]=='PUT']),1)
        u=LimitedUpload(room=0);s=launcher.Stream('events',u,[])
        s.feed(b'{"type":"result","parent_tool_use_id":"toolu_1","usage":{}}\n')
        s.feed(b'{"type":"result","pad":"'+b'x'*launcher.TAIL_BYTES+b'"}\n',final=True);s.complete(0)
        self.assertIsNone(s.result);self.assertEqual(len(s.line),0)

    def test_ordinary_chunks_leave_the_tail_reserve_unused(self):
        u=Upload();u.used=launcher.MAX_BYTES-launcher.TAIL_BYTES-10;s=launcher.Stream('events',u,[])
        s.feed(b'x'*11+b'\n'+RESULT,final=True)
        self.assertTrue(s.truncated);self.assertEqual(u.calls,[])
        s.complete(0)
        self.assertEqual(u.calls[0][2],b'\n'+RESULT)

    def test_capture_separates_stderr_and_preserves_nonzero_exit(self):
        p=subprocess.Popen([sys.executable,'-c','import sys;print(\'{"type":"result","usage":{"input_tokens":4}}\');print("synthetic-secret",file=sys.stderr);sys.exit(9)'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        u=Upload();self.assertEqual(launcher.capture(p,u,[b'synthetic-secret']),9)
        self.assertTrue(any('/events/' in c[1] for c in u.calls))
        self.assertTrue(any('/stderr/' in c[1] for c in u.calls))
        self.assertNotIn(b'synthetic-secret',b''.join(c[2] for c in u.calls))
        self.assertEqual([json.loads(c[2])['exit_code'] for c in u.calls if c[0]=='POST'],[9,9])

    def test_result_tail_framing_fits_exact_byte_and_chunk_reserves(self):
        prefix=b'{"type":"result","padding":"';suffix=b'"}'
        for newline in [b'',b'\n']:
            for size in [launcher.TAIL_BYTES-2,launcher.TAIL_BYTES-1,launcher.TAIL_BYTES]:
                with self.subTest(newline=newline,size=size):
                    u=LimitedUpload(room=0);u.used=launcher.MAX_BYTES-launcher.TAIL_BYTES
                    s=launcher.Stream('events',u,[]);s.count=16380
                    result=prefix+b'x'*(size-len(prefix)-len(suffix))+suffix
                    s.feed(result+newline,final=True);s.complete(0)
                    tail=[c for c in u.calls if c[0]=='PUT' and (c[3] or {}).get('x-frontier-trace-tail')=='1']
                    if size==launcher.TAIL_BYTES-2:
                        self.assertEqual(len(tail),4)
                        self.assertEqual(b''.join(c[2] for c in tail),b'\n'+result+b'\n')
                        self.assertEqual((s.count,u.used),(16384,launcher.MAX_BYTES))
                        self.assertEqual(json.loads(b''.join(c[2] for c in tail))['type'],'result')
                    else:
                        self.assertIsNone(s.result);self.assertEqual(tail,[])

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
            self.assertEqual(child['CLAUDE_CODE_MAX_OUTPUT_TOKENS'],'128000')

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

GATEWAY={'ANTHROPIC_BASE_URL':'http://10.42.1.7:8082','ANTHROPIC_AUTH_TOKEN':'synthetic-run-token',
    'FRONTIER_MODEL_API_BASE_URL':'http://10.42.1.7:8082/v1','FRONTIER_MODEL_PROXY_TOKEN':'synthetic-run-token','CODEX_API_KEY':'synthetic-run-token',
    'FRONTIER_CODEX_MODEL':'minimax-m3.1-flash','FRONTIER_DRIVER_API_URL':'http://10.42.1.7:8080',
    **{name:'minimax-m3.1-flash' for name in launcher.GATEWAY_MODEL_VARS}}

class EnvironmentModeTests(unittest.TestCase):
    def prepare(self,env):
        with tempfile.TemporaryDirectory() as workspace, patch.object(launcher,'WORKSPACE',Path(workspace)), patch.object(Path,'stat',return_value=SimpleNamespace(st_mode=0o100600,st_size=20)), patch.object(Path,'read_text',return_value='synthetic-driver-bearer'), patch.dict(os.environ,env,clear=True):
            child,_,secrets,model,effort=launcher.prepare_environment()
            for name in ('HOME','TMPDIR','CLAUDE_CONFIG_DIR'):self.assertTrue(child.pop(name).startswith(workspace))
            return child,secrets,model,effort

    def test_api_key_mode_is_unchanged_and_ignores_gateway_settings(self):
        base={'ANTHROPIC_API_KEY':'sk-ant-api03-synthetic','FRONTIER_CLAUDE_MODEL':'claude-opus-5-5','FRONTIER_DRIVER_API_URL':'http://127.0.0.1:8080','PATH':'/usr/bin','FRONTIER_REASONING_EFFORT':'max'}
        child,secrets,model,effort=self.prepare(base)
        self.assertEqual(child,{'ANTHROPIC_API_KEY':'sk-ant-api03-synthetic','FRONTIER_CLAUDE_MODEL':'claude-opus-5-5','FRONTIER_DRIVER_API_URL':'http://127.0.0.1:8080',
            'FRONTIER_REASONING_EFFORT':'max','PATH':'/opt/frontier/candidate/bin:/usr/bin','DISABLE_AUTOUPDATER':'1','CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC':'1',
            'DISABLE_TELEMETRY':'1','DISABLE_ERROR_REPORTING':'1','CLAUDE_CODE_MAX_OUTPUT_TOKENS':'128000'})
        self.assertEqual((secrets,model,effort),([b'sk-ant-api03-synthetic',b'synthetic-driver-bearer'],'claude-opus-5-5','max'))
        # An API key always wins; gateway variables and limits are stripped/ignored as before.
        mixed=dict(GATEWAY,**base,FRONTIER_MODEL_OUTPUT_TOKENS='32000',FRONTIER_MODEL_CONTEXT_TOKENS='1000000')
        child2,_,model2,_=self.prepare(mixed)
        self.assertEqual(model2,'claude-opus-5-5');self.assertEqual(child2['CLAUDE_CODE_MAX_OUTPUT_TOKENS'],'128000')
        self.assertFalse([k for k in child2 if (k.startswith('ANTHROPIC_') and k != 'ANTHROPIC_API_KEY') or k in ('CODEX_API_KEY','FRONTIER_MODEL_API_BASE_URL','CLAUDE_CODE_MAX_CONTEXT_TOKENS')])

    def test_gateway_mode_keeps_driver_anthropic_settings_and_alias(self):
        child,secrets,model,effort=self.prepare(dict(GATEWAY,FRONTIER_REASONING_EFFORT='max'))
        self.assertEqual((model,effort),('minimax-m3.1-flash','max'))
        self.assertNotIn('ANTHROPIC_API_KEY',child);self.assertNotIn('CODEX_API_KEY',child);self.assertNotIn('FRONTIER_MODEL_API_BASE_URL',child)
        self.assertEqual(child['ANTHROPIC_BASE_URL'],'http://10.42.1.7:8082');self.assertEqual(child['ANTHROPIC_AUTH_TOKEN'],'synthetic-run-token')
        for name in launcher.GATEWAY_MODEL_VARS+('ANTHROPIC_SMALL_FAST_MODEL',):self.assertEqual(child[name],'minimax-m3.1-flash')
        self.assertEqual(child['CLAUDE_CODE_MAX_OUTPUT_TOKENS'],'128000');self.assertNotIn('CLAUDE_CODE_MAX_CONTEXT_TOKENS',child)
        self.assertIn(b'synthetic-run-token',secrets);self.assertIn(b'synthetic-driver-bearer',secrets)
        command=launcher.command(model,effort)
        self.assertEqual(command[command.index('--model')+1],'minimax-m3.1-flash');self.assertEqual(command[command.index('--effort')+1],'max')

    def test_gateway_limits_come_from_the_candidate_environment(self):
        child,_,_,_=self.prepare(dict(GATEWAY,FRONTIER_MODEL_OUTPUT_TOKENS='64000',FRONTIER_MODEL_CONTEXT_TOKENS='1000000',FRONTIER_AUTO_COMPACT_WINDOW='524288'))
        self.assertEqual((child['CLAUDE_CODE_MAX_OUTPUT_TOKENS'],child['CLAUDE_CODE_MAX_CONTEXT_TOKENS'],child['CLAUDE_CODE_AUTO_COMPACT_WINDOW']),('64000','1000000','524288'))

    def test_gateway_mode_fails_closed(self):
        bad=[{'ANTHROPIC_BASE_URL':'http://driver.invalid:8082'},{'ANTHROPIC_BASE_URL':'https://10.42.1.7:8082'},{'ANTHROPIC_BASE_URL':'http://10.42.1.7:8082/v1'},
             {'FRONTIER_MODEL_API_BASE_URL':'http://10.42.1.8:8082/v1'},{'ANTHROPIC_MODEL':'other-alias'},{'ANTHROPIC_DEFAULT_HAIKU_MODEL':'claude-haiku-5-5'},
             {'ANTHROPIC_MODEL':'Bad/Alias','FRONTIER_CODEX_MODEL':'Bad/Alias'},{'FRONTIER_MODEL_OUTPUT_TOKENS':'0'},{'FRONTIER_MODEL_CONTEXT_TOKENS':'1e6'},
             {'ANTHROPIC_AUTH_TOKEN':'x'*16385}]
        for change in bad:
            with self.subTest(change=change),self.assertRaises(launcher.LaunchError):self.prepare(dict(GATEWAY,**change))
        # Neither an API key nor a gateway token: the original API-key error.
        with self.assertRaisesRegex(launcher.LaunchError,'API key'):self.prepare({k:v for k,v in GATEWAY.items() if k!='ANTHROPIC_AUTH_TOKEN'})

if __name__=='__main__':unittest.main()
