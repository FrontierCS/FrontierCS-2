#!/usr/bin/env python3
"""Standard local-file Verifier adapter; no provider or Driver client."""
from __future__ import annotations
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tarfile
import tempfile
from beat_spec_feedback import public_metrics

MAX_ARCHIVE = 286720
MAX_PATCH = 262144
MAX_EVIDENCE = 1048576
SOURCE = '6d415b1c92ebb6bcb615c9104ad05bca276bdc26'
VISIBLE = ('candidate_elapsed_seconds_median', 'candidate_elapsed_seconds_p90')


class InvalidSubmission(ValueError):
    pass


def read_payload(path: Path) -> bytes:
    if not 0 < path.stat().st_size <= MAX_ARCHIVE:
        raise InvalidSubmission('invalid archive size')
    try:
        with tarfile.open(path, 'r:') as archive:
            seen = set()
            patch = None
            for member in archive:
                if member.name in seen:
                    raise InvalidSubmission('duplicate archive member')
                seen.add(member.name)
                if member.name == 'payload' and member.isdir():
                    continue
                if member.name != 'payload/solution.patch' or not member.isfile():
                    raise InvalidSubmission('invalid archive member')
                if not 0 < member.size <= MAX_PATCH:
                    raise InvalidSubmission('invalid patch size')
                patch = archive.extractfile(member).read(MAX_PATCH + 1)
            if patch is None:
                raise InvalidSubmission('patch entrypoint missing')
            return patch
    except (tarfile.TarError, EOFError) as exc:
        raise InvalidSubmission('invalid tar') from exc


def validate_dataset(path: Path, mode: str):
    if mode not in ('train', 'eval'):
        raise ValueError('invalid verification mode')
    expected = {'schema':'parallel-dbscan-dataset/v1','problem_id':'parallel_dbscan',
                'mode':mode,'source_commit':SOURCE,
                'source_partition':'development' if mode == 'train' else 'public_plus_hidden'}
    if path.stat().st_size > 4096 or json.loads(path.read_text()) != expected:
        raise ValueError('dataset binding mismatch')


def project(measured: dict, mode: str) -> dict:
    score = measured['score']
    if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 100:
        raise ValueError('invalid score')
    metrics = measured['metrics']
    valid = bool(metrics.get('beat_spec_workload_passes'))
    # Source invalid submissions are scored zero. Do not turn scientific FAIL
    # or a zero score into an infrastructure failure or fabricate a private verdict.
    result = {'status':'passed','raw_score':score,
              'summary':'exact result accepted; performance measured' if valid else 'submission invalid',
              'metrics':{}}
    if valid and mode == 'train':
        result['metrics'] = public_metrics(
            metrics, beat_spec_path=Path(__file__).resolve().parents[1]/'judge/beat_spec.json',
            configured_visible=VISIBLE)
    if valid and mode == 'eval':
        rows = metrics['beat_spec_workload_passes']
        passed = sum(row['metrics']['elapsed_seconds'] for row in rows)
        total = len(rows)
        result['private_metrics'] = {
            'beats_reference':int(metrics['beats_reference']),
            'reference_score':metrics['reference_score'],'margin':metrics['margin'],
            'aggregation':{'kind':'conjunction'},
            'beat_spec_progress':{'elapsed_seconds':{
                'passed':passed,'failed':total-passed,'total':total,
                'pass_percent':100.0*passed/total,'fail_percent':100.0*(total-passed)/total}}}
    return result


def write_evidence(path: Path, audit: dict, work: Path):
    body = json.dumps(audit, sort_keys=True, allow_nan=False).encode()
    raw = work/'evidence.tar'
    with tarfile.open(raw, 'w', format=tarfile.USTAR_FORMAT) as archive:
        member = tarfile.TarInfo('measurement.json')
        member.size, member.mode, member.mtime = len(body), 0o644, 0
        archive.addfile(member, io.BytesIO(body))
    subprocess.run(['zstd','-q','-f',str(raw),'-o',str(path)],check=True,timeout=10)
    if not 0 < path.stat().st_size <= MAX_EVIDENCE:
        raise ValueError('evidence exceeds bound')


def evaluator_environment(work: Path, mode: str) -> dict:
    # Only task execution settings reach compiled/submitted code. The platform
    # wrapper's authentication and any operator environment are not inherited.
    return {'PATH':'/usr/local/bin:/usr/bin:/bin','LANG':'C.UTF-8','LC_ALL':'C.UTF-8',
            'HOME':str(work),'TMPDIR':str(work),'PYTHONDONTWRITEBYTECODE':'1',
            'FRONTIER_SUBMISSION_ROLE':'final' if mode=='eval' else 'dev',
            'GIT_CONFIG_NOSYSTEM':'1','GIT_CONFIG_GLOBAL':'/dev/null'}


def verify(payload: Path, dataset: Path, mode: str, result_path: Path,
           evidence_path: Path, *, judge: Path, workspace: Path):
    # All task-owned writable state belongs below the wrapper's /frontier mount.
    workspace.mkdir(parents=True,exist_ok=True)
    result_path.parent.mkdir(parents=True,exist_ok=True)
    evidence_path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='dbscan-', dir=workspace) as td:
        work = Path(td)
        try:
            validate_dataset(dataset, mode)
            payload_hash = hashlib.sha256(payload.read_bytes()).hexdigest() if payload.stat().st_size <= MAX_ARCHIVE else None
            try:
                patch = read_payload(payload)
            except InvalidSubmission:
                measured = {'score':0.0,'score_unbounded':0.0,'message':'invalid submission archive','metrics':{}}
                patch = None
            if patch is not None:
                # Source evaluator creates work below TASK_DIR and system TMPDIR.
                # Copy the immutable image content into this disposable mount first.
                local_judge = work/'judge'
                shutil.copytree(judge,local_judge)
                for item in local_judge.rglob('*'):
                    item.chmod(0o755 if item.is_dir() else 0o644)
                local_judge.chmod(0o755)
                patch_path = work/'solution.patch'
                patch_path.write_bytes(patch)
                env = evaluator_environment(work, mode)
                output = work/'evaluation.stdout'
                with output.open('wb') as stdout, (work/'evaluation.stderr').open('wb') as stderr:
                    process = subprocess.Popen(['python3',str(local_judge/'evaluator.py'),str(patch_path)],
                                               cwd=work,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
                    try:
                        returncode = process.wait(timeout=600 if mode == 'eval' else 240)
                    finally:
                        # Kill compiler/DBSCAN descendants too, including after an
                        # evaluator exits while submitted code leaves children.
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.wait()
                if returncode or output.stat().st_size > MAX_EVIDENCE:
                    raise RuntimeError('evaluator process failed')
                measured = json.loads(output.read_text())
            result = project(measured, mode)
            audit = {'schema':'parallel-dbscan-evidence/v1','source_commit':SOURCE,
                     'mode':mode,'payload_sha256':payload_hash,
                     'dataset_sha256':hashlib.sha256(dataset.read_bytes()).hexdigest(),
                     'measurement':measured,
                     'science_status':'M004 public-plus-hidden final and strict comparison; M010 fixed20ms reference-independent continuous; beat spec v3 paper-regime seed-spreader workloads'}
            write_evidence(evidence_path,audit,work)
        except Exception as exc:
            evidence_path.unlink(missing_ok=True)
            result = {'status':'failed','summary':'verification runtime failed','metrics':{},
                      'failure':{'code':'verification_runtime_error',
                                 'message':type(exc).__name__,'retryable':False}}
        result_path.write_text(json.dumps(result,sort_keys=True,allow_nan=False)+'\n')
        return result


def main():
    result = verify(Path(os.environ['FRONTIER_PAYLOAD_PATH']),
                   Path(os.environ['FRONTIER_DATASET_PATH']),
                   os.environ['FRONTIER_VERIFICATION_MODE'],
                   Path(os.environ['FRONTIER_RESULT_PATH']),
                   Path(os.environ['FRONTIER_EVIDENCE_PATH']),
                   judge=Path('/opt/frontier/judge'),workspace=Path('/frontier/work'))
    return 0 if result['status']=='passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
