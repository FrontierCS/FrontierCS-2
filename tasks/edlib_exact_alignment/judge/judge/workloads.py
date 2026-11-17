"""Frozen challenge generator v1; hashes are verified before any evaluation."""
import hashlib, json, random
from pathlib import Path
VERSION='edlib-dna-v1-python-random-mt19937'
ROOT=Path(__file__).resolve().parent

def generate(row):
    r=random.Random(row['seed']); n=row['length']; p=(100-row['similarity'])/100
    core=''.join(r.choice('ACGT') for _ in range(n)); q=[]
    for c in core:
        u=r.random()
        if u < p*.2: continue
        if u < p*.4: q.append(r.choice('ACGT')); q.append(c)
        elif u < p: q.append(r.choice('ACGT'.replace(c,'')))
        else: q.append(c)
    # Two extra insertions ensure unequal-length cases even if indel counts cancel.
    q.extend('AC')
    prefix=''.join(r.choice('ACGT') for _ in range(n//10))
    suffix=''.join(r.choice('ACGT') for _ in range(n//10))
    t=core if row['mode']=='NW' else core+suffix if row['mode']=='SHW' else prefix+core+suffix
    return ''.join(q),t

def digest(s): return hashlib.sha256(s.encode()).hexdigest()
def rows(final=False):
    data=json.loads((ROOT/'workload_manifest.json').read_text())
    return [x for x in data['workloads'] if final or x['split']=='public']
def materialize(row, directory):
    q,t=generate(row)
    if digest(q)!=row['query_sha256'] or digest(t)!=row['target_sha256']: raise ValueError('fixture integrity')
    qp=directory/'query.fa'; tp=directory/'target.fa'
    qp.write_text('>query\n'+q+'\n'); tp.write_text('>target\n'+t+'\n')
    return q,t,qp,tp

if __name__=='__main__':
    entries=[]
    for split,root in [('public',7100321),('heldout',893437129)]:
        for n in [10000,30000]:
            for sim in [60,70,80,90,99]:
                for mode in ['NW','SHW','HW']:
                    row=dict(split=split,length=n,similarity=sim,mode=mode,seed=root+n*101+sim*13+['NW','SHW','HW'].index(mode))
                    row['id']=f'{split}-{n}-{sim}-{mode}'
                    q,t=generate(row); row.update(query_sha256=digest(q),target_sha256=digest(t),query_length=len(q),target_length=len(t)); entries.append(row)
    data=dict(generator=VERSION,compiler='PENDING: actual compiler version must be locked before qualification',flags=['-O3','-DNDEBUG','-std=c++14'],replicates=5,scheduling='sequential alternating C,R / R,C; one pinned CPU',process_timeout_seconds=900,verification_timeout_seconds=600,memory_mb=4096,fixture_limit_bytes=1073741824,mutation_proportions={'substitution':.6,'insertion':.2,'deletion':.2},workloads=entries)
    (ROOT/'workload_manifest.json').write_text(json.dumps(data,indent=2)+'\n')
