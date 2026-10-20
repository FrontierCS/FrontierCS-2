"""Native local-file adapter. Driver owns credentials, transport and final selection."""
from __future__ import annotations
import hashlib
import io
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import tempfile

_MODULE_DIR = Path(__file__).resolve().parent
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))
from beat_spec_feedback import public_metrics

MAX_TAR_BYTES = 276480
MAX_PATCH_BYTES = 262144
TASK = Path(__file__).resolve().parents[1]
JUDGE = TASK / 'judge'
REPO = TASK / 'repo'
if not REPO.is_dir():
    REPO = TASK / 'agent-base/frontier/task/repo'
COMPILER_SHA256 = '13f10181b8e4a26f9e452a02ddbbf77620929b69f9d3eac09f3b3a27adae73b3'
METRICS = ('sort_time_ns', 'region_graph_ns', 'auxiliary_bytes')
VISIBLE = ['dev_candidate_median_sort_time_ns', 'dev_candidate_max_auxiliary_bytes',
           'dev_candidate_median_region_graph_ns']

class InvalidSubmission(ValueError):
    pass

class RuntimeFailure(RuntimeError):
    pass


def validate_selector(path: Path, mode: str) -> None:
    expected = {'schema_version':'frontier.parallel-radix-sort-dataset/v1',
                'problem_id':'parallel_radix_sort','mode':mode,
                'suite_revision':'pr29-ac19add-public-plus-hidden-v1',
                'suite':'DEV' if mode == 'train' else 'DEV+HIDDEN'}
    if mode not in {'train', 'eval'} or path.stat().st_size > 4096:
        raise RuntimeFailure('dataset mode or size mismatch')
    if json.loads(path.read_text()) != expected:
        raise RuntimeFailure('dataset selection mismatch')


def read_patch(path: Path) -> bytes:
    if not 0 < path.stat().st_size <= MAX_TAR_BYTES:
        raise InvalidSubmission('archive size invalid')
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
                if (member.name != 'payload/solution.patch' or not member.isfile()
                        or member.pax_headers or member.size > MAX_PATCH_BYTES):
                    raise InvalidSubmission('invalid archive member')
                stream = archive.extractfile(member)
                if stream is None:
                    raise InvalidSubmission('missing patch bytes')
                patch = stream.read(MAX_PATCH_BYTES + 1)
            if patch is None or len(patch) > MAX_PATCH_BYTES:
                raise InvalidSubmission('missing or oversized patch')
            return patch
    except (tarfile.TarError, EOFError) as exc:
        raise InvalidSubmission('malformed uncompressed tar') from exc


def project(raw: list, mode: str) -> tuple[dict, dict]:
    if not isinstance(raw, list) or len(raw) != 4:
        raise RuntimeFailure('invalid evaluator result')
    score, unbounded, message, metrics = raw
    if (isinstance(score, bool) or not isinstance(score, (int, float))
            or not math.isfinite(score) or not 0 <= score <= 100
            or not isinstance(metrics, dict)):
        raise RuntimeFailure('invalid score or metrics')
    result = {'status':'passed','raw_score':score,'summary':'submission evaluated','metrics':{}}
    audit = {'score':score,'score_unbounded':unbounded,'metrics':metrics}
    if message != 'scored complete suite':
        if score != 0:
            raise RuntimeFailure('incomplete evaluator score')
        result['summary'] = 'submission invalid; score zero'
        audit['outcome'] = 'candidate_invalid'
        # Source exception text may contain hidden identities or submitted output.
        return result, audit
    rows = metrics.get('beat_spec_workload_passes')
    prefixes = ['d'] if mode == 'train' else ['d','f']
    identities = [prefix+x for prefix in prefixes for x in ['-int-uniform','-pair-zipf','-int-duplicates','-pair-almost']]
    # Beat spec v3: paper-regime 16M-key rows follow the original rows.
    identities += [prefix+x for prefix in prefixes for x in ['-int-zipf075','-int-uniform16m']]
    count = len(identities)
    if (not isinstance(rows,list) or len(rows) != count
            or [r.get('workload') for r in rows] != identities
            or any(set(r.get('metrics',{})) != set(METRICS)
                   or any(type(r['metrics'][name]) is not bool for name in METRICS) for r in rows)):
        raise RuntimeFailure('incomplete workload verdict')
    progress = {}
    for name in METRICS:
        passed = sum(r['metrics'][name] for r in rows)
        progress[name] = {'passed':passed,'failed':count-passed,'total':count,
                          'pass_percent':100.0*passed/count,'fail_percent':100.0*(count-passed)/count}
    every = all(value for r in rows for value in r['metrics'].values())
    if type(metrics.get('beats_reference')) is not int or metrics['beats_reference'] != int(every):
        raise RuntimeFailure('inconsistent reference verdict')
    if mode == 'train':
        for key in VISIBLE:
            value = metrics[key]
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value < 0:
                raise RuntimeFailure('invalid candidate aggregate')
        result['metrics'] = public_metrics(
            metrics, beat_spec_path=JUDGE/'beat_spec.json',
            configured_visible=VISIBLE)
    else:
        for key in ['reference_score','margin']:
            value = metrics.get(key)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
                raise RuntimeFailure('invalid reference score')
        result['private_metrics'] = {'beats_reference':metrics['beats_reference'],
            'reference_score':metrics['reference_score'],'margin':metrics['margin'],
            'beat_spec_progress':progress,
            'aggregation':{'objective':'geomean_std_sort_speedup','workloads':4*len(prefixes),'graded_workloads':count,
                           'geomean_speedup':metrics['geomean_speedup'],
                           'log_geomean_speedup':metrics['log_geomean_speedup']}}
        if any(not isinstance(metrics.get(key),(int,float)) or isinstance(metrics[key],bool) or not math.isfinite(metrics[key]) for key in ['geomean_speedup','log_geomean_speedup']):
            raise RuntimeFailure('invalid continuous audit')
        if metrics['geomean_speedup']<=0:
            raise RuntimeFailure('invalid speedup')
    return result, audit


def preflight() -> None:
    archive = REPO / 'frontier-gcc7.tar.xz'
    if not archive.is_file() or hashlib.sha256(archive.read_bytes()).hexdigest() != COMPILER_SHA256:
        raise RuntimeFailure('pinned GCC7 compiler unavailable')
    if not {0,1,2,3}.issubset(os.sched_getaffinity(0)):
        raise RuntimeFailure('source CPU affinity 0-3 is unavailable')


def evaluate(patch: Path, mode: str, scratch: Path) -> list:
    output = scratch / 'measurement.json'
    # Deliberate allowlist: no wrapper/provider credentials enter submitted C++.
    env = {'PATH':'/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
           'LANG':'C.UTF-8','TMPDIR':str(scratch),'HOME':str(scratch),
           'GIT_CONFIG_NOSYSTEM':'1','GIT_CONFIG_GLOBAL':'/dev/null'}
    with (scratch/'evaluator.log').open('wb') as log:
        proc = subprocess.Popen(['python3',str(JUDGE/'evaluator.py'),str(patch),mode,str(output)],
            cwd=scratch,env=env,stdout=log,stderr=log,start_new_session=True)
        try:
            proc.wait(timeout=900)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            return [0,0,'candidate evaluation timeout',{}]
        finally:
            # Kill any child left by a candidate or a timed-out source subprocess.
            try: os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError: pass
    if proc.returncode != 0 or not output.is_file() or output.stat().st_size > 65536:
        raise RuntimeFailure('evaluator runtime failure')
    return json.loads(output.read_text())


def write_evidence(path: Path, evidence: dict) -> None:
    raw = json.dumps(evidence,sort_keys=True,allow_nan=False).encode()
    if len(raw) > 131072:
        raise RuntimeFailure('evidence size exceeded')
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w',format=tarfile.USTAR_FORMAT) as archive:
        member = tarfile.TarInfo('measurement.json'); member.size = len(raw); member.mode = 0o600
        archive.addfile(member,io.BytesIO(raw))
    compressed = subprocess.run(['zstd','-q','-c'],input=buffer.getvalue(),stdout=subprocess.PIPE,check=True).stdout
    if len(compressed) > 1048576:
        raise RuntimeFailure('compressed evidence exceeds policy')
    path.write_bytes(compressed)


def main() -> None:
    result_path = Path(os.environ['FRONTIER_RESULT_PATH'])
    evidence_path = Path(os.environ['FRONTIER_EVIDENCE_PATH'])
    for path in [result_path,evidence_path]:
        if not path.resolve().is_relative_to('/frontier/output'):
            raise RuntimeFailure('output path outside platform mount')
        path.parent.mkdir(parents=True,exist_ok=True)
    try:
        mode = os.environ['FRONTIER_VERIFICATION_MODE']
        dataset = Path(os.environ['FRONTIER_DATASET_PATH'])
        validate_selector(dataset,mode)
        payload = Path(os.environ['FRONTIER_PAYLOAD_PATH'])
        # Verify infrastructure before classifying submitted code failures.
        preflight()
        digest = hashlib.sha256()
        with payload.open('rb') as stream:
            for block in iter(lambda: stream.read(65536), b''): digest.update(block)
        payload_hash = digest.hexdigest()
        try:
            patch_bytes = read_patch(payload)
            with tempfile.TemporaryDirectory(prefix='radix-',dir='/frontier') as directory:
                scratch = Path(directory); patch = scratch/'solution.patch';patch.write_bytes(patch_bytes)
                raw = evaluate(patch,mode,scratch)
                result,audit = project(raw,mode)
        except InvalidSubmission:
            result = {'status':'passed','raw_score':0,'summary':'submission invalid; score zero','metrics':{}}
            audit = {'outcome':'invalid_archive'}
        audit.update(mode=mode,payload_sha256=payload_hash,dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest())
        write_evidence(evidence_path,audit)
    except Exception:
        result = {'status':'failed','summary':'verifier runtime failure','metrics':{},
                  'failure':{'code':'verifier_runtime_error','message':'task runtime or dataset is unavailable','retryable':False}}
    result_path.write_text(json.dumps(result,sort_keys=True,allow_nan=False)+'\n')

if __name__ == '__main__':
    main()
