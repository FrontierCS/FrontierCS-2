"""Measure frozen reference distance errors; never infer tolerances or issue PASS."""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=Path('/data'))
    parser.add_argument('--starter', type=Path, default=ROOT/'repo')
    parser.add_argument('--workload', choices=('sift_k10','sift_k50','sift_k100','gist_k10','gist_k50','gist_k100'))
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    from evaluator import _apply_patch, select_suite, validate_data_manifest
    from knn_worker import measure_workload
    config = select_suite(json.loads((ROOT/'scoring_config.json').read_text()), 'final')
    selected = [w for w in config['workloads'] if args.workload is None or w['id']==args.workload]
    manifest = validate_data_manifest(ROOT/'data_manifest.json',args.data_root,[w['dataset'] for w in selected])
    if args.worker:
        assert len(selected)==1
        with tempfile.TemporaryDirectory(prefix='knn-reference-calibration-') as td:
            source=Path(td)/'reference';shutil.copytree(args.starter,source)
            _apply_patch(source,ROOT/'reference.patch')
            row=measure_workload(source,selected[0],config['warmup_iterations'],config['timed_iterations'],ROOT/'refknn.py',args.data_root)
            args.output.write_text(json.dumps(row,indent=2)+'\n')
        return 0
    rows=[]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='knn-reference-results-') as td:
        for workload in selected:
            result=Path(td)/(workload['id']+'.json')
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker','--workload',workload['id'],'--output',str(result),'--data-root',str(args.data_root),'--starter',str(args.starter)],check=True,timeout=3600)
            rows.append(json.loads(result.read_text()))
            args.output.write_text(json.dumps({'status':'partial_measurements_only','rows':rows},indent=2)+'\n')
    report={'schema_version':'frontier.knn-reference-distance-calibration/v1','status':'measurements_only_tolerances_require_user_approval','oracle':config['distance_consistency']['oracle'],'warmup_iterations':config['warmup_iterations'],'timed_iterations':config['timed_iterations'],'dataset_manifest':manifest,'workload_base_seeds':{w['id']:w['base_seed'] for w in selected},'rows':rows,'suggested_tolerances':None,'pass_fail':None,'source_files':{name:'sha256:'+hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('reference.patch','refknn.py','scoring_config.json','knn_worker.py','data_manifest.json')}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
