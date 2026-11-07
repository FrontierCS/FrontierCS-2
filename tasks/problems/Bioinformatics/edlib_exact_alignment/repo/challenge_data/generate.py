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
    data=json.loads((ROOT/'public_workloads.json').read_text())
    return [x for x in data['workloads'] if final or x['split']=='public']
def materialize(row, directory):
    q,t=generate(row)
    if digest(q)!=row['query_sha256'] or digest(t)!=row['target_sha256']: raise ValueError('fixture integrity')
    qp=directory/'query.fa'; tp=directory/'target.fa'
    qp.write_text('>query\n'+q+'\n'); tp.write_text('>target\n'+t+'\n')
    return q,t,qp,tp

if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Materialize one public Edlib workload')
    parser.add_argument('workload_id');parser.add_argument('output',type=Path)
    args=parser.parse_args()
    selected=[r for r in rows() if r['id']==args.workload_id]
    if len(selected)!=1: parser.error('unknown public workload ID')
    args.output.mkdir(parents=True,exist_ok=True)
    materialize(selected[0],args.output)
