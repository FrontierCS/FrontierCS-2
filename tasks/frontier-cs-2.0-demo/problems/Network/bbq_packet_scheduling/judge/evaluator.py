"""CPU evaluator for the transparent author-corrected BBQ source/reference.

Cycles come from the Verilator trace; the clock period comes from the fixed
synthesis and timing model in harbor/app/frontier_cpu/timing.py.
"""
from __future__ import annotations
import base64
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import zlib
import yaml
from submission_patch import apply_submission_patch

sys.dont_write_bytecode = True
TASK = Path(__file__).resolve().parent
CONFIG = yaml.safe_load((TASK/'config.yaml').read_text())
spec = importlib.util.spec_from_file_location('bbq_cpu_runner', TASK/'harbor/app/frontier_cpu/runner.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
assert runner.ENTRYPOINT == "hardware/bbq/generator/bbq.py"
DEV = json.loads((TASK/'harbor/app/frontier_cpu/public_cases.json').read_text())
# Private held-out seeds and membership. Final always freshly executes DEV as well.
HELD = [dict(case, id='hidden_'+str(i), seed=seed)
        for i,(case,seed) in enumerate(zip(DEV,[524287,65537,104729,999983]))]
HELD += [dict(id='hidden_scale',depth=3,width=16,capacity=255,seed=390581)]
METRICS = ('throughput_ops_per_second','latency_seconds')
# One lookup-table level between registers: the timing model's shortest period.
FLOOR_PS = runner.timing.REGISTER_PS + runner.timing.LUT_PS
RESOURCES = ('period_ps','lut_count','register_count','memory_bits')
# Private evidence must fit the verifier's 512 KiB metrics limit with the rest
# of the metrics, and its strings the verifier's 4096-character cap.
EVIDENCE_BYTES = 400_000
EVIDENCE_CHUNK = 4000


def efficiency(row):
    if not row.get('valid'):
        return 0.0
    # Packet rate as a percentage of the model ceiling: the zero-latency offered
    # schedule at the timing model's one-level period. Proportional to operations
    # per second on the finite trace; no private reference value is involved.
    return 100.0 * (row['offered_span'] * FLOOR_PS) / (row['cycles'] * row['period_ps'])


def summarize(candidate, reference, cases):
    score = sum(efficiency(candidate.get(c['id'], {})) for c in cases)/len(cases)
    refscore = sum(efficiency(reference[c['id']]) for c in cases)/len(cases)
    audit=[]
    for case in cases:
        c, r = candidate.get(case['id'], {}), reference[case['id']]
        valid = c.get('valid',False) and c.get('count') == r['count'] and r['count'] > 0
        # Exact integer comparisons of count/(cycles*period) and latency*period.
        audit.append({'workload':case['id'], 'metrics':{
            'throughput_ops_per_second': bool(valid and
                c['count']*r['cycles']*r['period_ps'] > r['count']*c['cycles']*c['period_ps']),
            'latency_seconds': bool(valid and
                c['latency_cycles']*c['period_ps'] <= r['latency_cycles']*r['period_ps'])}})
    devrows = [candidate.get(c['id'], {}) for c in DEV]
    allvalid = all(r.get('valid') for r in devrows)
    # Operator-only cost diagnostics; never listed in solver_feedback.
    resources = {side: {case['id']: {k: rows[case['id']][k] for k in RESOURCES}
                        for case in cases if rows.get(case['id'], {}).get('valid')}
                 for side, rows in (('candidate', candidate), ('reference', reference))}
    metrics = dict(beat_spec_workload_passes=audit,
        beats_reference=int(all(v for row in audit for v in row['metrics'].values())),
        reference_score=refscore, margin=score-refscore,
        dev_min_throughput_ops_per_second=min(r['throughput_ops_per_second'] for r in devrows) if allvalid else 0,
        dev_max_latency_picoseconds=max(r['latency_ps'] for r in devrows) if allvalid else -1,
        dev_max_clock_period_picoseconds=max(r['period_ps'] for r in devrows) if allvalid else -1,
        dev_min_throughput_ops_per_cycle=min(r['throughput_ops_per_cycle'] for r in devrows) if allvalid else 0,
        dev_max_latency_cycles=max(r['latency_cycles'] for r in devrows) if allvalid else -1,
        dev_valid_fraction=sum(bool(r.get('valid')) for r in devrows)/len(devrows),
        resource_diagnostics=resources)
    return score, score, 'CPU cycle and clock-period evaluation complete', metrics


def _reason(error):
    return (type(error).__name__ + ': ' + str(error))[:300]


def _digest(data):
    return 'sha256:' + hashlib.sha256(data).hexdigest()


def _blob(data):
    """Compressed bytes in strings short enough for the verifier's 4096-character cap."""
    text = base64.b64encode(zlib.compress(data, 9)).decode('ascii')
    return {'sha256': _digest(data), 'bytes': len(data),
            'zlib_base64': [text[i:i+EVIDENCE_CHUNK] for i in range(0, len(text), EVIDENCE_CHUNK)]}


def _tail(path, limit=EVIDENCE_CHUNK):
    try:
        return path.read_bytes()[-limit:].decode('utf-8', errors='replace')
    except OSError:
        return None


def _errors(path, limit=EVIDENCE_CHUNK):
    """Tool error lines (Verilator `%Error`), which a log tail can miss."""
    try:
        lines = path.read_text(errors='replace').splitlines()
    except OSError:
        return None
    return '\n'.join(line for line in lines if line.startswith('%Error'))[:limit]


def _build_evidence(build):
    """Generated RTL, Yosys check/statistics output and the mapped netlist of one build."""
    timing_dir = build/'timing'
    entry, rtl, netlist = {}, None, None
    for name, path in (('simulated_rtl', build/'bbq.sv'), ('synthesized_input', timing_dir/'design.v')):
        if path.is_file():
            entry[name+'_sha256'] = _digest(path.read_bytes())
    if (build/'bbq.sv').is_file():
        rtl = _blob((build/'bbq.sv').read_bytes())
    entry['verilator_log_tail'] = _tail(build/'trace_tb'/'compile.log')
    entry['verilator_errors'] = _errors(build/'trace_tb'/'compile.log')
    entry['yosys_check_log'] = _tail(timing_dir/'check.log')
    entry['yosys_stat_log'] = _tail(timing_dir/'stat.log')
    try:
        declared = json.loads((timing_dir/'declared.json').read_text())
        # Instantiated primitive -> whether the elaborated design reads its output.
        entry['declared_memories'] = {name: bool(cell.get('read'))
                                      for name, cell in declared.items()}
    except (OSError, ValueError, AttributeError):
        pass
    try:
        raw = (timing_dir/'netlist.json').read_bytes()
        module = json.loads(raw)['modules']['bbq']
        entry['netlist_sha256'], entry['netlist_bytes'] = _digest(raw), len(raw)
        cells = module['cells']
        entry['netlist_memories'] = sorted(name for name, cell in cells.items()
                                           if cell['type'] in ('bram_simple2port', 'sc_fifo'))
        histogram = {}
        for cell in cells.values():
            histogram[cell['type']] = histogram.get(cell['type'], 0) + 1
        entry['netlist_cell_types'] = histogram
        netlist = _blob(json.dumps({'ports': module['ports'], 'cells': {
            name: {k: cell.get(k, {}) for k in ('type', 'parameters', 'connections')}
            for name, cell in cells.items()}}, sort_keys=True, separators=(',', ':')).encode())
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return entry, rtl, netlist


def private_evidence(root, failures):
    """Bounded per-configuration audit record; never solver-visible.

    Retains the generated RTL, both tool logs and, while the evidence budget
    allows, the mapped netlist without wire names or source attributes.
    """
    evidence = {'failures': dict(failures), 'builds': {}}
    blobs = []
    if 'official' in failures:
        ran = [root/'official'/('TEST_'+name) for name in runner.OFFICIAL
               if (root/'official'/('TEST_'+name)).is_dir()]
        entry = evidence['official_failure'] = {'test': ran[-1].name if ran else None}
        if ran:
            entry['compile_log_tail'] = _tail(ran[-1]/'tb_bbq'/'compile.log')
            entry['compile_errors'] = _errors(ran[-1]/'tb_bbq'/'compile.log')
            entry['run_log_tail'] = _tail(ran[-1]/'run.log')
        if (root/'official'/'bbq.sv').is_file():
            blobs.append((entry, 'rtl', _blob((root/'official'/'bbq.sv').read_bytes())))
    for build in sorted(root.glob('build_*')):
        entry, rtl, netlist = _build_evidence(build)
        evidence['builds'][build.name] = entry
        blobs += [(entry, 'rtl', rtl), (entry, 'netlist', netlist)]
    # RTL before netlists, smaller first, while the budget allows; a summary
    # and the hashes remain for anything left out.
    size = len(json.dumps(evidence))
    blobs = [blob for blob in blobs if blob[2] is not None]
    for entry, name, blob in sorted(blobs, key=lambda b: (b[1] != 'rtl', len(json.dumps(b[2])))):
        cost = len(json.dumps(blob)) + len(name) + 6
        if size + cost <= EVIDENCE_BYTES:
            entry[name] = blob
            size += cost
    return evidence


def evaluate(solution_path: str) -> tuple[float,float,str,dict]:
    candidate_patch = reference_patch = None
    work = None
    try:
        # Keep every copy/build on disk, including submission_patch's temporary tree.
        base = Path(os.environ.get('BBQ_SCRATCH', str(TASK/'scratch'))).resolve()
        base.mkdir(parents=True,exist_ok=True)
        tempfile.tempdir = str(base)
        cases = DEV + HELD if os.environ.get('FRONTIER_SUBMISSION_ROLE') == 'final' else DEV
        candidate_patch = apply_submission_patch(solution_path, TASK, CONFIG)
        if not candidate_patch.ok:
            return 0.,0.,'Invalid patch submission',{}
        reference_patch = apply_submission_patch(TASK/'reference.patch', TASK, CONFIG)
        if not reference_patch.ok:
            raise RuntimeError('reference unavailable')
        work = Path(tempfile.mkdtemp(prefix='evaluation-',dir=base))
        # Paired measurements: no hardcoded reference cycles and no persistent cache.
        reference = runner.suite(reference_patch.workdir, work/'reference', cases)
        candidate = {}
        builds, clocks, failures = {}, {}, {}
        try:
            runner.official(candidate_patch.workdir, work/'candidate'/'official')
            for case in cases:
                key=tuple(case[k] for k in ('depth','width','capacity'))
                build=work/'candidate'/('build_'+'_'.join(map(str,key)))
                try:
                    if key not in builds:
                        builds[key]=None
                        builds[key]=runner.build_trace(candidate_patch.workdir,build,case)
                    if builds[key] is None:
                        raise ValueError('no simulation build')
                    if key not in clocks:
                        # A design outside the timed design rules stays invalid.
                        clocks[key]=None
                        clocks[key]=runner.clock(build,case,
                            expected_memories=reference[case['id']]['memory_inventory'])
                    if clocks[key] is None:
                        raise ValueError('no clock period')
                    candidate[case['id']]=runner.physical(
                        runner.measure(builds[key], work/'candidate'/case['id'],case),clocks[key])
                except Exception as error:
                    failures[case['id']]=_reason(error)
                    candidate[case['id']]={'valid':False}
        except Exception as error:
            failures['official']=_reason(error)
            candidate={}
        result = summarize(candidate,reference,cases)
        # Operator-only audit record of what was simulated and what was timed.
        try:
            result[3]['private_evidence'] = private_evidence(work/'candidate', failures)
        except Exception as error:
            result[3]['private_evidence'] = {'failures': failures, 'error': _reason(error)}
        # Detailed event streams never enter returned metrics or messages.
        return result
    except Exception:
        return 0.,0.,'Evaluation failed: build, runtime, or reference validation',{}
    finally:
        for patched in (candidate_patch,reference_patch):
            if patched and patched.cleanup_root:
                shutil.rmtree(patched.cleanup_root,ignore_errors=True)
        if work:
            shutil.rmtree(work,ignore_errors=True)


if __name__ == '__main__':
    result=evaluate(sys.argv[1] if len(sys.argv)>1 else '')
    print(json.dumps(dict(zip(('score','score_unbounded','message','metrics'),result)),allow_nan=False))
