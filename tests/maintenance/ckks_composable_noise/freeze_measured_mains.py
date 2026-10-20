"""Maintenance: regenerate judge/timed_main_sha256.json (run from the judge directory)."""
import json,hashlib,sys,subprocess,tempfile,shutil,os
sys.path.insert(0,'.')
import workload_configurations as wc
out={}
tmp=tempfile.mkdtemp()
shutil.copytree('harbor/app',tmp+'/r'); subprocess.run(['git','-C',tmp+'/r','init','-q'],check=True)
subprocess.run(['git','-C',tmp+'/r','apply','--whitespace=nowarn',os.path.abspath('private-comment-restoration.patch')],check=True)
real=wc.require; wc.TIMED_MAIN_SHA256={}
for kind,root in (('authored','harbor/app/'),('canonical_restored',tmp+'/r/')):
    for row in wc.configurations():
        if kind=='canonical_restored' and row['circuit'] not in wc.RESTORED_MAIN_SHA256: continue
        src=open(root+row['main_path']).read()
        cfg=wc.configured_main(src,row); assert wc.main_source_binding_kind(cfg,row)==kind,(row['id'],kind)
        wc.require=lambda v,m: real(v,m) if 'frozen form' not in m else None
        d=wc.timed_main(cfg,row); wc.require=real
        out.setdefault(row['id'],{})[kind]=hashlib.sha256(d.encode()).hexdigest()
shutil.rmtree(tmp)
open('judge/timed_main_sha256.json','w').write(json.dumps(out,indent=1,sort_keys=True)+'\n'); print(len(out),sum(len(v) for v in out.values()))
