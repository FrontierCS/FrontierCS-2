"""Offline official Codex flags and Responses retry protocol qualification."""
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parent
BINARY = Path(os.environ.get('FRONTIER_TEST_CODEX_BINARY', '/opt/frontier/candidate/codex/bin/codex'))
LAUNCHER = ROOT / 'launcher.py'


def response(success=False, code='500', hint=True):
    if success:
        item={'id':'msg_local','type':'message','role':'assistant','content':[{'type':'output_text','text':'LOCAL_PROTOCOL_OK','annotations':[]}]}
        events=[{'type':'response.created','response':{'id':'resp_local','model':'gpt-6-astra','status':'in_progress','output':[]}},
                {'type':'response.output_item.added','output_index':0,'item':item},
                {'type':'response.output_text.delta','item_id':'msg_local','output_index':0,'content_index':0,'delta':'LOCAL_PROTOCOL_OK'},
                {'type':'response.output_item.done','output_index':0,'item':item},
                {'type':'response.completed','response':{'id':'resp_local','model':'gpt-6-astra','status':'completed','output':[item],'usage':{'input_tokens':11,'output_tokens':3,'total_tokens':14}}}]
    else:
        error={'code':code,'message':'Offline protocol fixture; no model request.'}
        if hint:error['headers']={'retry-after':'5'}
        events=[{'type':'response.failed','response':{'status':'failed','error':error}}]
    return ''.join('event: '+e['type']+'\ndata: '+json.dumps(e)+'\n\n' for e in events).encode()


def run_protocol_qualification():
    assert hashlib.sha256(BINARY.read_bytes()).hexdigest() == '628e35487d888a28f0427fc37501da529dd3d81fcbad7a2c3421c1e8670d92e1'
    spec=importlib.util.spec_from_file_location('current_launcher',LAUNCHER)
    launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)
    receipts=[]
    for scenario in ('generic500_then_success','canonical_cyber_policy'):
        requests=[]
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['content-length'])))
                requests.append({'monotonic':time.monotonic(),'path':self.path,'model':body.get('model'),'reasoning':body.get('reasoning'),'stream':body.get('stream'),'header_names':sorted(self.headers.keys()),'body_sha256':hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()})
                data=response(success=len(requests)>1) if scenario=='generic500_then_success' else response(code='cyber_policy',hint=False)
                self.send_response(200);self.send_header('Content-Type','text/event-stream');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data);self.wfile.flush()
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        with tempfile.TemporaryDirectory(prefix='isolated-cli-') as temp:
            temp=Path(temp);workspace=temp/'workspace';workspace.mkdir();home=temp/'home';home.mkdir();(home/'codex').mkdir()
            launcher.CODEX=BINARY;launcher.WORKSPACE=workspace
            old_model=os.environ.get('FRONTIER_CODEX_MODEL');old_effort=os.environ.get('FRONTIER_CODEX_REASONING_EFFORT')
            os.environ['FRONTIER_CODEX_MODEL']='gpt-6-astra';os.environ['FRONTIER_CODEX_REASONING_EFFORT']='xhigh'
            argv=launcher.command('http://127.0.0.1:'+str(server.server_port)+'/v1')
            if old_model is None:os.environ.pop('FRONTIER_CODEX_MODEL')
            else:os.environ['FRONTIER_CODEX_MODEL']=old_model
            if old_effort is None:os.environ.pop('FRONTIER_CODEX_REASONING_EFFORT')
            else:os.environ['FRONTIER_CODEX_REASONING_EFFORT']=old_effort
            env={k:v for k,v in os.environ.items() if not any(w in k.upper() for w in ('KEY','TOKEN','SECRET','PASSWORD','CREDENTIAL'))}
            env.update(HOME=str(home),CODEX_HOME=str(home/'codex'),FRONTIER_MODEL_PROXY_TOKEN='offline-dummy-not-a-credential',NO_PROXY='127.0.0.1,localhost',no_proxy='127.0.0.1,localhost')
            start=time.monotonic()
            try:
                proc=subprocess.run(argv,input='Reply only LOCAL_PROTOCOL_OK. Do not call tools.\n',text=True,capture_output=True,env=env,timeout=35)
                result={'exit_code':proc.returncode,'stdout':proc.stdout,'stderr':proc.stderr}
            except subprocess.TimeoutExpired as error:result={'timeout':True,'stdout':str(error.stdout),'stderr':str(error.stderr)}
            result.update(scenario=scenario,elapsed_seconds=time.monotonic()-start,requests=requests,argv=argv)
            if scenario=='generic500_then_success':
                result['qualified']=len(requests)==2 and 4.5<=requests[1]['monotonic']-requests[0]['monotonic']<=10 and result.get('exit_code')==0
                if len(requests)>=2:result['retry_interval_seconds']=requests[1]['monotonic']-requests[0]['monotonic']
            else:result['qualified']=len(requests)==1 and result.get('exit_code',0)!=0 and not result.get('timeout')
            receipts.append(result)
        server.shutdown();server.server_close()
    out={'observed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Official CLI offline fixture, no paid model requests; exact current launcher arguments','official_binary':{'version':'0.162.1','sha256':hashlib.sha256(BINARY.read_bytes()).hexdigest()},'cases':receipts}
    path=Path(os.environ.get('FRONTIER_TEST_OUTPUT_DIR', tempfile.gettempdir()))/('official-cli-protocol-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json');path.write_text(json.dumps(out,indent=2)+'\n')
    print(path)
    for r in receipts:print(r['scenario'],r['qualified'],r.get('exit_code'),len(r['requests']),r.get('retry_interval_seconds'),r['stderr'][-1400:])
    assert all(r['qualified'] for r in receipts)


if __name__=='__main__':main()
