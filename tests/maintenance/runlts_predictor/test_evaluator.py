"""Offline contract tests; admitted full105 tests are explicitly opt-in.

Unit measurements below are invented parser/arithmetic inputs, NEVER task results.
No patched program is run against synthetic traces. The judge is the Verifier image's
(the judge bundle, starter at pristine/); patches go through the shared edit scope and the
judge's check_tree hook as the shared verifier runs them, and evaluate() receives the judged
tree. Tests marked CLANG reach the libclang part of the source audit and need the verifier
image's clang Python bindings.
"""
import deployment_test_guard  # noqa: F401  # isort: skip

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from decimal import Decimal
import pytest
from tests.maintenance.runlts_predictor.runtime_imports import (
    DESIGN_SPEC, EDIT_SCOPE, FAILURE_STAGES, FEEDBACK, MODULES, PRISTINE, REPOSITORY_ROOT, RULES, TASK,
    failure_metrics, judged, task_manifest)
e = MODULES["evaluator"]
CODE=Path('.')
policy = MODULES["source_policy"]
BITS=policy.REFERENCE_BITS
PROVENANCE=REPOSITORY_ROOT/'docs/task-authoring/runlts_predictor/provenance'
STATEMENT=TASK/'repo/README.md'
CLANG=pytest.mark.skipif(importlib.util.find_spec('clang') is None,
                         reason="needs libclang's Python bindings of the verifier image")

@pytest.fixture
def restored(tmp_path):
    """The pristine predictor with judge/reference.patch, as the evaluator restores it."""
    return e.reference_tree(tmp_path)

def patched_tree(scratch,name):
    """A predictor copy of the judged tree of a judge-owned patch (shared edit scope)."""
    with judged((e.ROOT/name).read_bytes()) as (scope,verdict):
        assert scope.ok and verdict is None,(name,scope.category,verdict)
        return e.predictor_copy(scope.tree,scratch,Path(name).stem)

def measure(b,w,n=1000000):
    return e.Measurement(n,b,w,2*n-1,Decimal(f'{1000*b/n:.4f}'),Decimal(f'{1000*w/n:.4f}'))

def test_reference_is_exact_source_and_budget(restored):
    digest=hashlib.sha256((restored/CODE/'my_cond_branch_predictor.cc').read_bytes()).hexdigest()
    assert digest=='dd673059e19faa9d6902490e6a2d511aec51092ed1de9e0b7568080b2d0590d5'
    assert policy.audit(restored)['logical_bits']==1570890
    assert policy.REFERENCE_BITS<=policy.CAP_BITS
    # The predictor tree the audit and the reference identity cover has no statement.
    assert not (restored/'README.md').exists() and (PRISTINE/'README.md').is_file()

@pytest.mark.parametrize('name',['baseline_naive.patch',pytest.param('partial_example.patch',marks=CLANG)])
def test_naive_and_plausible_partial_are_accepted(name,tmp_path):
    # Accepted by the shared edit scope and by check_tree's source audit.
    assert policy.audit(patched_tree(tmp_path,name))['logical_bits']<=policy.CAP_BITS

def test_reference_repeated_missing_data_is_explicit(monkeypatch,restored):
    def missing(_):raise e.EvaluationError('required trace asset missing')
    monkeypatch.setattr(e,'manifest',missing)
    for _ in range(3):
        with pytest.raises(e.EvaluationError,match='required trace asset missing'):
            evaluate_without_telemetry(str(restored))

def test_unexpected_infrastructure_failure_is_not_a_scored_zero(monkeypatch,restored):
    def failed(_):raise OSError('private fixture diagnostic')
    monkeypatch.setattr(e,'_evaluate',failed)
    with pytest.raises(e.EvaluationError,match='internal evaluator infrastructure failure'):
        evaluate_without_telemetry(str(restored))

def test_invalid_patch_never_crashes(tmp_path):
    # The patch is the shared verifier's: an unreadable one is its scored zero, `malformed`.
    with judged(b'invalid patch\n') as (scope,verdict):
        assert not scope.ok and scope.category=='malformed' and verdict is None
    # evaluate() only ever receives a tree; a missing one is infrastructure, not a score.
    with pytest.raises(e.EvaluationError,match='internal evaluator infrastructure failure'):
        evaluate_without_telemetry(str(tmp_path/'missing-tree'))

def test_arithmetic_order_and_headroom():
    weaker=[measure(8000,300000)];partial=[measure(5000,210000)];stronger=[measure(3000,140000)]
    assert 0<e.score_of(weaker)<e.score_of(partial)<e.score_of(stronger)<100
    assert e.score_of([measure(2999,139999)])>e.score_of(stronger)
    # Independent of any reference. Not an empirical naive/reference witness.

def test_strict_integer_comparison_and_complete_private_metrics():
    workloads=[{'id':'unit-a'},{'id':'unit-b'}]
    reference={w['id']:measure(3000,140000) for w in workloads}
    candidate={w['id']:measure(2999,139999) for w in workloads}
    r=e.grade(workloads,candidate,reference,False,logical_storage_bits=BITS)
    assert 0<=r[0]<=100 and math.isfinite(r[0])
    assert {'beats_reference','beat_spec_workload_passes','reference_score','margin'}<=r[3].keys()
    assert r[3]['beats_reference']==1
    for row in r[3]['beat_spec_workload_passes']:
        assert set(row['metrics'])==set(e.METRICS)
        assert all(type(v) is bool for v in row['metrics'].values())
    tied=e.grade(workloads,reference,reference,False,logical_storage_bits=BITS)
    assert tied[3]['beats_reference']==0 and tied[3]['margin']==0 and tied[0]>0
    candidate['unit-b']=measure(2999,140000)
    assert e.grade(workloads,candidate,reference,False,logical_storage_bits=BITS)[3]['beats_reference']==0

def test_sub_display_resolution_comparisons():
    n=1000000000
    r=measure(1000000,2000000,n);c=measure(999999,1999999,n)
    assert c.printed_br==r.printed_br and c.printed_wp==r.printed_wp
    m=e.grade([{'id':'unit'}],{'unit':c},{'unit':r},False,logical_storage_bits=BITS)[3]
    assert m['beats_reference']==1

def test_zero_reference_counts_do_not_pass():
    r=measure(0,0)
    assert e.grade([{'id':'unit'}],{'unit':r},{'unit':r},False,logical_storage_bits=BITS)[3]['beats_reference']==0

def test_final_covers_public_again_and_keeps_feedback_private():
    manifest=json.loads((e.ROOT/'workload_manifest.json').read_text())['workloads']
    c={r['id']:measure(2999,139999) for r in manifest};ref={r['id']:measure(3000,140000) for r in manifest}
    public=next(r['id'] for r in manifest if r['development']);c[public]=ref[public]
    metrics=e.grade(manifest,c,ref,True,logical_storage_bits=BITS)[3]
    assert len(metrics['beat_spec_workload_passes'])==105
    assert metrics['beats_reference']==0
    assert 'train_br_mis_pki_mean' not in metrics
    # Of the train feedback a final result carries only candidate_failure, 0 for a measured result.
    assert set(e.CONFIG['evaluation']['solver_visible_metrics'])&metrics.keys()=={'candidate_failure'}
    assert metrics['candidate_failure']==0
    with pytest.raises(e.EvaluationError):e.grade(manifest[:-1],c,ref,True,logical_storage_bits=BITS)

def test_partition_membership_and_rendered_budget():
    rows=json.loads((e.ROOT/'workload_manifest.json').read_text())['workloads']
    dev=sorted(r['path'] for r in rows if r['development'])
    assert len(rows)==105 and len(dev)==24
    assert hashlib.sha256(('\n'.join(dev)+'\n').encode()).hexdigest()=='b3c171aa40be1d08363fa0322707ecbcd8c8c5ad7bbe363e5dd67c8b7afea1cf'
    # The train-call ceiling is task.toml's, rendered into the agent README.
    assert e.CONFIG['evaluation']['max_dev_submissions']==task_manifest.read_task(TASK)['quota']['max_train_calls']
    assert f"**{e.CONFIG['evaluation']['max_dev_submissions']} train calls**" in task_manifest.agent_readme(TASK)
    # The train-visible metrics are task.toml [feedback]: the design record's, plus candidate_failure.
    spec=json.loads(DESIGN_SPEC.read_text())
    assert e.CONFIG['evaluation']['solver_visible_metrics']==FEEDBACK==list(e.FEEDBACK)
    assert FEEDBACK==[r['name'] for r in spec['solver_feedback']]+['candidate_failure']
    assert not set(FEEDBACK)&set(spec['withheld'])

def output_fixture():
    header='       Instr       Cycles      IPC      NumBr     MispBr BrPerCyc MispBrPerCyc        MR     MPKI      CycWP   CycWPAvg   CycWPPKI'
    row='1000000 500000 2.0000 100000 3000 0.2000 0.0060 3.0000% 3.0000 140000 46.6667 140.0000'
    return '\n'.join(['--DIRECT CONDITIONAL BRANCH PREDICTION MEASUREMENTS (50 Perc instructions)--',header,row,'--DIRECT CONDITIONAL BRANCH PREDICTION MEASUREMENTS (Full Simulation i.e. Counts Not Reset When Warmup Ends)--',header,row])

def test_parser_completeness_and_spoof_rejection():
    text=output_fixture();expected={'full_instr':1000000,'measured_instr':1000000}
    m=e.parse_output(text,expected);assert m.misp_br==3000 and m.cyc_wp==140000
    for bad in [text+text,text[:100],text.replace('3.0000 140000','2.0000 140000')]:
        with pytest.raises(e.CandidateError):e.parse_output(bad,expected)
    with pytest.raises(e.CandidateError):e.parse_output(text,{'full_instr':2000000,'measured_instr':1000000})

def test_retained_sc_is_protected(restored):
    p=restored/CODE/'my_cond_branch_predictor.cc';p.write_text(p.read_text().replace('#define LOGB 17','#define LOGB 18'))
    with pytest.raises(policy.InvalidSource):policy.audit(restored)

@pytest.mark.parametrize('added',['printf("fake measurement");',
    pytest.param('std::array<uint64_t, 200000> giant = {};',marks=CLANG)])
def test_extra_io_and_oversize_are_rejected(added,tmp_path):
    tree=e.predictor_copy(PRISTINE,tmp_path,'tree')
    p=tree/CODE/'my_cond_branch_predictor.cc';base=p.read_text()
    p.write_text(base.replace('namespace nRUNLTS {','namespace nRUNLTS {\n'+added,1))
    with pytest.raises(policy.InvalidSource):policy.audit(tree)

@pytest.mark.skipif(os.environ.get('RUNLTS_ADMITTED_FULL105_TESTS')!='1',reason='requires separately admitted full105 assets and compute; prohibited in local authoring phase')
def test_admitted_reference_scores_repeats_and_real_naive_gap():
    # Three genuine full-suite calls; opt-in resource plan must fund all of them.
    old=os.environ.get('FRONTIER_SUBMISSION_ROLE');os.environ['FRONTIER_SUBMISSION_ROLE']='final'
    try:
        scratch=Path(tempfile.mkdtemp(prefix='runlts-admitted-'))
        reference=e.reference_tree(scratch)
        results=[evaluate_without_telemetry(str(reference)) for _ in range(3)]
        for score,_,_,m in results:
            assert m.get('evaluation_valid') is True
            assert math.isfinite(score) and 0<=score<=100
            assert {'beats_reference','beat_spec_workload_passes','reference_score','margin'}<=m.keys()
            assert len(m['beat_spec_workload_passes'])==105
            assert m['beats_reference']==int(all(v for r in m['beat_spec_workload_passes'] for v in r['metrics'].values()))
        assert results[0][0]==results[1][0]==results[2][0]
        assert results[0][3]['paired_measurements']==results[1][3]['paired_measurements']==results[2][3]['paired_measurements']
        naive=evaluate_without_telemetry(str(patched_tree(scratch,'baseline_naive.patch')))
        assert naive[3].get('evaluation_valid') is True and naive[0]<results[0][0]
        partial=evaluate_without_telemetry(str(patched_tree(scratch,'partial_example.patch')))
        assert partial[3].get('evaluation_valid') is True
        shutil.rmtree(scratch)
    finally:
        if old is None:os.environ.pop('FRONTIER_SUBMISSION_ROLE',None)
        else:os.environ['FRONTIER_SUBMISSION_ROLE']=old

@pytest.mark.parametrize('added',[
    '%:define LOGB 20\n',
    '// splice\\\n',
    '__builtin_printf("fake");\n',
    'template<int I> int bank(){static int v=0;return v;}\n',
    '#include "solver_extensions/../../private.h"\n',
])
def test_reviewed_lexical_bypasses_rejected(added,tmp_path):
    tree=e.predictor_copy(PRISTINE,tmp_path,'tree')
    p=tree/CODE/'my_cond_branch_predictor.cc';s=p.read_text()
    p.write_text(s.replace('namespace nRUNLTS {\n','namespace nRUNLTS {\n'+added,1))
    with pytest.raises(policy.InvalidSource):policy.audit(tree)

@CLANG
def test_retained_input_write_rejected(tmp_path):
    tree=patched_tree(tmp_path,'partial_example.patch')
    p=tree/CODE/'my_cond_branch_predictor.cc';s=p.read_text()
    p.write_text(s.replace('        ready.at(dst_reg) = false;','        dst_reg = 0;\n        ready.at(dst_reg) = false;'))
    with pytest.raises(policy.InvalidSource):policy.audit(tree)

def test_missing_manifest_is_infrastructure_error(monkeypatch,restored):
    def bad_manifest(final):raise FileNotFoundError('private path')
    monkeypatch.setattr(e,'manifest',bad_manifest)
    with pytest.raises(e.EvaluationError,match='internal evaluator infrastructure failure') as exc:
        evaluate_without_telemetry(str(restored))
    assert 'private path' not in str(exc.value)

@pytest.mark.parametrize('added',[
    'auto f = [](auto tag) { static std::array<int,1024> bank{}; return bank.at(0); };\n',
    'constexpr int x = sizeof(R"x("/*)x");\n',
    '/* unclosed comment\n',
])
def test_additional_critic_bypasses_rejected(added,tmp_path):
    tree=e.predictor_copy(PRISTINE,tmp_path,'tree')
    p=tree/CODE/'my_cond_branch_predictor.cc';s=p.read_text()
    p.write_text(s.replace('namespace nRUNLTS {\n','namespace nRUNLTS {\n'+added,1))
    with pytest.raises(policy.InvalidSource):policy.audit(tree)

@CLANG
def test_reference_field_alias_rejected(tmp_path):
    tree=patched_tree(tmp_path,'partial_example.patch')
    p=tree/CODE/'my_cond_branch_predictor.cc';s=p.read_text()
    p.write_text(s.replace('        ready.at(dst_reg) = false;','        struct Alias { decltype(dst_reg)& r; }; Alias a{dst_reg}; a.r = 0;\n        ready.at(dst_reg) = false;'))
    with pytest.raises(policy.InvalidSource):policy.audit(tree)

def test_submission_helper_includes_new_source_and_ignores_build_products(tmp_path):
    # The starter's own make_submission.sh (repo/), for a local check.
    tree=tmp_path/'repo';shutil.copytree(PRISTINE,tree)
    def git(*args):return subprocess.check_output(['git','-C',str(tree),*args])
    git('init','-q');git('add','-A');git('-c','user.name=Fixture','-c','user.email=fixture@localhost','commit','-qm','base')
    (tree/CODE/'solver_extensions').mkdir();(tree/CODE/'solver_extensions/utility.h').write_text('// solver-authored utility\n')
    (tree/CODE/'cbp').write_text('generated executable placeholder')
    subprocess.run(['bash','make_submission.sh'],cwd=tree,check=True,capture_output=True)
    patch=(tree/'solution.patch').read_text()
    assert 'solver_extensions/utility.h' in patch and 'diff --git a/cbp ' not in patch
    # The shared edit scope accepts what it writes (check_tree's audit of it needs libclang).
    scope=EDIT_SCOPE.check(patch.encode(),PRISTINE,RULES)
    try:assert scope.ok and sorted(c.path for c in scope.changes)==['solver_extensions/utility.h']
    finally:scope.dispose()
    (tree/'outside.cc').write_text('unexpected new source')
    assert subprocess.run(['bash','make_submission.sh'],cwd=tree,capture_output=True).returncode!=0

def test_artifact_restoration_scope_and_frozen_wiring(restored):
    manifest=json.loads((PROVENANCE/'artifact_manifest.json').read_text())
    assert manifest['upstream_root']=='runlts_predictor/code'
    assert manifest['authentic_execution']['entrypoint']=='lib/cbp.cc'
    assert manifest['solution_surface']['modify']+manifest['solution_surface']['add_under']==e.CONFIG['edit_scope']['allowed_paths']
    for rel,digest in json.loads((e.ROOT/'integrity.json').read_text()).items():
        assert hashlib.sha256((restored/rel).read_bytes()).hexdigest()==digest
    assert not any(p.name=='.git' for p in PRISTINE.rglob('*'))

def test_scientific_samples_never_enter_manifest():
    rows=json.loads((e.ROOT/'workload_manifest.json').read_text())['workloads']
    assert all('sample' not in r['path'] for r in rows)

def test_cpp_entrypoint_adapter_dispatches_original_command(tmp_path,monkeypatch):
    tree=e.predictor_copy(PRISTINE,tmp_path,'nested')
    calls=[]
    def execution(argv,cwd,timeout,deadline,read_paths=()):
        calls.append((argv,cwd,timeout,deadline,read_paths))
        return output_fixture()
    monkeypatch.setattr(e,'run_sandboxed',execution)
    before=Path.cwd()
    row={'path':'fixture/fixture.gz','full_instr':1000000,'measured_instr':1000000}
    result=e.run_trace(tree,row,12345)
    assert result.misp_br==3000
    # In its isolate instance, with read access to that one trace.
    trace=os.path.realpath(e.DATA/row['path'])
    assert calls==[(['./cbp',trace],tree/CODE,e.TRACE_TIMEOUT,12345,[trace])]
    assert Path.cwd()==before
    (tree/CODE/'lib/cbp.cc').unlink()
    with pytest.raises(e.CandidateError):e.run_trace(tree,row,12345)
    assert len(calls)==1  # no fallback when the real retained target is missing
    assert Path.cwd()==before

def test_run_ceiling_and_scientific_timeout():
    count=e.CONFIG['evaluation']['max_dev_submissions']
    cap=e.CONFIG['runtime']['timeout_seconds']
    assert count == 100
    assert 0 < e.CONFIG['evaluation']['max_train_wall_seconds'] <= e.CONFIG['evaluation']['answering_timeout_seconds'] <= 604800
    assert cap <= 604800
    # Limits revision final2h: a 2 h outer deadline, about twice the measured
    # official finals on 8 workers (3,247 to 3,608 s). It no longer covers every
    # simulation at its watchdog; it still covers the slowest measured final plus
    # both builds and one more trace at its watchdog.
    # See docs/acceptance/runlts_predictor/final-timeout-2h.md.
    assert e.TRACE_WORKERS==8
    assert cap==7200
    assert 3608+e.TRACE_TIMEOUT+2*e.BUILD_TIMEOUT<=cap
    assert e.CONFIG['evaluation']['solver_visible_metrics']==['train_br_mis_pki_mean','train_cyc_wp_pki_mean','all_metrics_progress','category_means','candidate_failure']


def test_reference_command_failure_retains_bounded_private_cause(tmp_path):
    import sys,time
    private_marker='private-trace-diagnostic'
    with pytest.raises(e.CommandFailure) as caught:
        e.run_command([sys.executable,'-c',"import sys;print('x'*9000+'private-trace-diagnostic');sys.exit(7)"],
                      tmp_path,5,time.monotonic()+10)
    error=caught.value
    assert str(error)=='build or simulator execution failed'
    assert error.diagnostic['returncode']==7 and error.diagnostic['kind']=='process_exit'
    assert len(error.diagnostic['output_tail'])<=4096
    failure=e.reference_failure('execution',error)
    assert private_marker not in str(failure)
    assert private_marker in failure.private_execution_evidence['command_failure']['output_tail']
    assert failure.private_execution_evidence['reference_phase']=='execution'


def test_reference_watchdog_has_explicit_cause_and_reaps_child(tmp_path):
    import sys,time,signal
    with pytest.raises(e.CommandFailure) as caught:
        e.run_command([sys.executable,'-c','import time;time.sleep(5)'],tmp_path,.05,time.monotonic()+10)
    assert caught.value.diagnostic['kind']=='timeout'
    assert caught.value.diagnostic['returncode']==-signal.SIGKILL
    assert caught.value.diagnostic['elapsed_seconds']<2
    assert str(caught.value)=='build or trace timeout'


def test_reference_unknown_exception_keeps_only_type():
    failure=e.reference_failure('execution',OSError('private path/secret'))
    assert failure.private_execution_evidence=={'execution_valid':False,
        'reference_phase':'execution','error_type':'OSError',
        'exception_chain':{'exceptions':[{'error_type':'OSError'}],'truncated':False}}


def test_reference_parser_failure_is_distinct_from_command_failure():
    failure=e.reference_failure('execution',e.CandidateError('incomplete or substituted trace execution'))
    assert failure.private_execution_evidence['parser_failure']=='incomplete or substituted trace execution'
    assert 'command_failure' not in failure.private_execution_evidence


# ---------------------------------------------------------------- rejection categories
# A source-audit rejection of scope or source policy is check_tree's: one of the shared
# edit-scope categories (SHARED_CATEGORY), reported by the shared verifier. A candidate failure
# found later (storage, build, simulation) is evaluate()'s scored zero whose train message is one
# fixed sentence (SENTENCES). The stages below are driven with invented outputs; no predictor is
# built or run.
import ast
import re
import sys
import types

ROW={'id':'unit','path':'fixture/fixture.gz','full_instr':1000000,'measured_instr':1000000}
EDITED=[EDIT_SCOPE.Change('M','my_cond_branch_predictor.cc')]

def rejected(category):
    # The message stays private (operators); the Agent sees the metrics through [feedback].
    return (0.0,0.0,f'Rejected: {e.SENTENCES[category]}.',failure_metrics(category))

def raise_calls(path,name):
    tree=ast.parse(Path(path).read_text())
    return [n.exc for n in ast.walk(tree) if isinstance(n,ast.Raise) and isinstance(n.exc,ast.Call)
            and isinstance(n.exc.func,ast.Name) and n.exc.func.id==name]

def test_every_source_policy_reason_is_bucketed():
    reasons={c.args[0].value for c in raise_calls(e.ROOT/'source_policy.py','InvalidSource')}
    assert len(reasons)==33 and reasons==set(e.SOURCE_POLICY_CATEGORIES)
    assert set(e.SOURCE_POLICY_CATEGORIES.values())==set(e.SHARED_CATEGORY)|{'semantic_audit','storage_cap'}
    assert set(e.SHARED_CATEGORY.values())<=set(EDIT_SCOPE.CATEGORIES)
    assert not set(e.SHARED_CATEGORY)&set(e.SENTENCES)

def test_every_candidate_failure_site_names_a_category_and_none_is_unused():
    used=set(e.SOURCE_POLICY_CATEGORIES.values());unnamed=[]
    for call in raise_calls(e.ROOT/'evaluator.py','CandidateError'):
        if len(call.args)==2 and isinstance(call.args[1],ast.Constant):used.add(call.args[1].value)
        else:unnamed.append(ast.unparse(call))
    # The deadline check is named by the stage it interrupts; the audit reason by the table.
    assert sorted(unnamed)==["CandidateError('evaluation timeout')",
        "CandidateError(reason, SOURCE_POLICY_CATEGORIES.get(reason) if isinstance(reason, str) else None)"]
    tree=ast.parse((e.ROOT/'evaluator.py').read_text())
    used|={n.args[1].value for n in ast.walk(tree) if isinstance(n,ast.Call)
           and isinstance(n.func,ast.Name) and n.func.id=='in_stage' and isinstance(n.args[1],ast.Constant)}
    assert used==set(e.SENTENCES)|set(e.SHARED_CATEGORY)

def test_train_sentences_are_fixed():
    # No field can carry a measurement, path, symbol, reference value or trace identity: the
    # only numbers in the sentences are the statement's public constants.
    assert tuple(e.SENTENCES)==FAILURE_STAGES
    # The candidate_failure codes are those task.toml [feedback] documents, in that order.
    assert e.FAILURE_CODES=={**{name:i+1 for i,name in enumerate(FAILURE_STAGES)},None:8}
    described=task_manifest.read_task(TASK)['feedback']['candidate_failure']
    assert described.startswith('0 for a measured result; ')
    assert [int(n) for n in re.findall(r'(?:: |, )(\d) ',described)]==list(range(1,9)),described
    for category,sentence in e.SENTENCES.items():
        assert sentence==sentence.strip() and sentence[0].islower() and not sentence.endswith('.'),category
        assert set(re.findall(r'\d[\d,]*',sentence))<={'1,572,864','192','17'},category   # C++17
        assert not re.search(r'/|\.(cc|h|gz)\b|fp_|int_',sentence),category

def test_statement_states_every_rejection():
    text=STATEMENT.read_text()
    # A scope or source-policy rejection: bytes between the regions and other files (the latter
    # the shared edit scope's out_of_scope), and the audit's exact forbidden words
    # (source_policy's list, less the `__` and `exec` prefixes).
    assert 'every byte between them\n  (whitespace and line endings included) is protected' in text
    assert 'Changing bytes outside the regions, or another\nretained file, is also rejected.' in text
    source=(e.ROOT/'source_policy.py').read_text()
    words=re.search(r"forbidden=r'\\b\(\?:([^)]*)\)\\b'",source)[1].split('|')
    stated=re.search(r'the whole words (.*?);',' '.join(text.split()))[1]
    assert re.findall(r'`([^`]+)`',stated)==[w for w in words if not w.startswith(('__','exec'))]
    assert 'exec\\w*' in words and 'starting with `__` or with `exec`' in text
    # The audit's limits: the 60 s timeout of source_audit and every command's address space.
    assert "str(ROOT/'source_policy.py'),str(tree)],tree,60,deadline)" in (e.ROOT/'evaluator.py').read_text()
    assert f'its 60 s or {e.MEMORY_BYTES//2**20:,} MiB limit' in text
    # Each fixed train sentence is a score-0 condition of the Correctness section.
    for needle in ("C++ errors in the audit's C++17 parse",'exceeds 1,572,864 bits (192 KiB',
                   'the source audit exceeds its 60 s',f'the build fails or exceeds {e.BUILD_TIMEOUT} s',
                   f'exceeds its\n{e.TRACE_TIMEOUT:,} s watchdog','killed by a resource limit',
                   'exactly one well-formed copy of each original measurement\nsection with consistent counters',
                   "instruction counts differ from the trace's expected\ncounts"):
        assert needle in text,needle

@pytest.fixture
def stages(tmp_path,monkeypatch):
    """_evaluate with every external step replaced; each test overrides one step."""
    trees={}
    for name in ('candidate','reference'):
        tree=tmp_path/name;(tree/'lib').mkdir(parents=True);(tree/'lib/cbp.cc').write_text('// fixture\n')
        (tree/'my_cond_branch_predictor.cc').write_text('// '+name+'\n')  # a candidate that is not the reference
        trees[name]=tree
    state=types.SimpleNamespace(trees=trees,audit={'valid':True,'logical_bits':BITS,'method':'fixture'},
        outputs={'candidate':output_fixture().replace(' 3000 ',' 2999 ').replace('3.0000 140000 46.6667 140.0000','2.9990 139999 46.6667 139.9990'),
                 'reference':output_fixture()},fail=set())
    def command(argv,cwd,timeout,deadline):
        side='reference' if Path(cwd)==trees['reference'] else 'candidate'
        step='audit' if argv[0]==sys.executable else 'build' if argv[0]=='make' else 'trace'
        if (side,step) in state.fail:
            raise e.CommandFailure('timeout' if step=='trace' else 'process_exit',elapsed_seconds=1,timeout_seconds=timeout,
                                   returncode=1,output_tail='private compiler or simulator output')
        if step=='audit':return json.dumps(state.audit)
        return state.outputs[side] if step=='trace' else ''
    monkeypatch.setattr(e,'predictor_copy',lambda source,scratch,name:trees['candidate'])
    monkeypatch.setattr(e,'reference_tree',lambda scratch:trees['reference'])
    monkeypatch.setattr(e,'run_command',command)
    monkeypatch.setattr(e,'run_sandboxed',lambda argv,cwd,timeout,deadline,**sandbox:command(argv,cwd,timeout,deadline))
    monkeypatch.setattr(e,'manifest',lambda final:[ROW])
    # The pinned table itself is covered by test_pinned_reference.py.
    monkeypatch.setattr(e,'pinned_reference',lambda suite,tree:{'unit':e.parse_output(output_fixture(),ROW)})
    monkeypatch.delenv('FRONTIER_SUBMISSION_ROLE',raising=False)
    monkeypatch.delenv(e.REFERENCE_AUDIT_ENV,raising=False)
    return state

def test_valid_development_evaluation_is_unchanged_by_the_stage_fixture(stages):
    assert e.check_tree('candidate-tree',str(PRISTINE),EDITED) is None
    score,_,message,metrics=evaluate_without_telemetry('candidate-tree')
    assert score>0 and message=='evaluation complete' and metrics['beats_reference']==1
    assert 'rejected' not in metrics and metrics['candidate_failure']==0
    assert metrics['all_metrics_progress']=={'passed':1,'failed':0,'total':1,'pass_percent':100.0,'fail_percent':0.0}

@pytest.mark.parametrize('name,category',[(None,'malformed'),('lib/cbp.cc','out_of_scope'),
                                           ('Makefile','out_of_scope'),('README.md','out_of_scope')])
def test_patch_rejection_is_the_shared_edit_scope(tmp_path,name,category):
    # The patch surface is the shared verifier's (task.toml [submission]), not the evaluator's.
    patch=b'invalid patch\n' if name is None else real_patch(tmp_path,lambda s:s+'\n// edit\n',name)
    with judged(patch) as (scope,verdict):
        assert not scope.ok and scope.category==category and verdict is None

@pytest.mark.parametrize('reason,category',sorted(e.SOURCE_POLICY_CATEGORIES.items()))
def test_each_source_policy_reason_reaches_its_category(stages,reason,category):
    stages.audit={'valid':False,'reason':reason}
    if category in e.SHARED_CATEGORY:
        # check_tree: a shared rejection naming the changed files; evaluate() never runs.
        assert e.check_tree('candidate-tree',str(PRISTINE),EDITED)==(e.SHARED_CATEGORY[category],['my_cond_branch_predictor.cc'])
        assert e.check_tree('candidate-tree',str(PRISTINE),[])==(e.SHARED_CATEGORY[category],['my_cond_branch_predictor.cc'])
    else:
        assert e.check_tree('candidate-tree',str(PRISTINE),EDITED) is None
        result=evaluate_without_telemetry('candidate-tree')
        assert result==rejected(category)
        assert reason not in json.dumps(result) or reason in e.SENTENCES[category]

def test_audit_resource_limit_is_source_audit_incomplete(stages):
    stages.fail={('candidate','audit')}
    assert e.check_tree('candidate-tree',str(PRISTINE),EDITED) is None   # evaluate() reports it
    assert evaluate_without_telemetry('candidate-tree')==rejected('source_audit_incomplete')

def test_storage_over_cap_after_audit_is_storage_cap(stages):
    stages.audit={'valid':True,'logical_bits':policy.CAP_BITS+1}
    assert evaluate_without_telemetry('candidate-tree')==rejected('storage_cap')

def test_build_failure_is_build_failed(stages):
    stages.fail={('candidate','build')}
    result=evaluate_without_telemetry('candidate-tree')
    assert result==rejected('build_failed') and 'private' not in json.dumps(result)

def test_trace_failure_or_timeout_is_simulator_run_failed(stages):
    stages.fail={('candidate','trace')}
    assert evaluate_without_telemetry('candidate-tree')==rejected('simulator_run_failed')

def test_deadline_during_candidate_stages_is_named_by_the_stage(stages,monkeypatch):
    def command(argv,cwd,timeout,deadline):
        if Path(cwd)==stages.trees['candidate'] and argv[0]=='make':raise e.CandidateError('evaluation timeout')
        if argv[0]==sys.executable:return json.dumps(stages.audit)
        return ''
    monkeypatch.setattr(e,'run_command',command)
    monkeypatch.setattr(e,'run_sandboxed',lambda argv,cwd,timeout,deadline,**sandbox:command(argv,cwd,timeout,deadline))
    assert evaluate_without_telemetry('candidate-tree')==rejected('build_failed')

@pytest.mark.parametrize('damage',[
    lambda text:text+'\n'+text,
    lambda text:text[:100],
    lambda text:text.replace('MispBr','Mispredicts'),
    lambda text:text.replace(' 2999 ',' x ',1),
    lambda text:text.replace('2.9990 139999','2.0000 139999',1),
])
def test_malformed_candidate_output_is_output_format(stages,damage):
    stages.outputs['candidate']=damage(stages.outputs['candidate'])
    assert evaluate_without_telemetry('candidate-tree')==rejected('output_format')

def test_candidate_instruction_counts_are_instruction_count_mismatch(stages):
    # A well-formed, self-consistent output for twice the trace's pinned instruction count.
    stages.outputs['candidate']=stages.outputs['candidate'].replace('1000000 500000','2000000 500000').replace(' 2999 ',' 5998 ').replace(' 139999 ',' 279998 ')
    assert evaluate_without_telemetry('candidate-tree')==rejected('instruction_count_mismatch')
    workloads=[{'id':'unit'}]
    with pytest.raises(e.CandidateError) as caught:
        e.grade(workloads,{'unit':measure(2999,139999,999999)},{'unit':measure(3000,140000)},False,logical_storage_bits=BITS)
    assert caught.value.category=='instruction_count_mismatch'

def test_missing_retained_entrypoint_is_refused_before_any_simulation(stages):
    # The judged tree is pristine plus the allowed files (the shared edit scope), so through
    # the shared verifier the entrypoint is always there. Should evaluate() see a tree without
    # it, the trace stage refuses it with a private scored zero, before any simulator runs.
    (stages.trees['candidate']/'lib/cbp.cc').unlink()
    assert evaluate_without_telemetry('candidate-tree')==(0.0,0.0,'retained simulator entrypoint missing',failure_metrics(None))

def test_unlisted_audit_reason_keeps_the_previous_empty_result(stages):
    stages.audit={'valid':False,'reason':'a reason no table lists'}
    assert e.check_tree('candidate-tree',str(PRISTINE),EDITED) is None
    assert evaluate_without_telemetry('candidate-tree')==(0.0,0.0,'a reason no table lists',failure_metrics(None))
    stages.audit={'valid':False,'reason':['not','text']}
    assert e.check_tree('candidate-tree',str(PRISTINE),EDITED) is None
    assert evaluate_without_telemetry('candidate-tree')[3]==failure_metrics(None)

def test_final_rejections_keep_the_previous_private_result(stages,monkeypatch):
    monkeypatch.setenv('FRONTIER_SUBMISSION_ROLE','final')
    stages.audit={'valid':False,'reason':'C++ source failed semantic audit'}
    assert evaluate_without_telemetry('candidate-tree')==(0.0,0.0,'C++ source failed semantic audit',{})
    stages.audit={'valid':True,'logical_bits':BITS};stages.fail={('candidate','build')}
    assert evaluate_without_telemetry('candidate-tree')==(0.0,0.0,'build or simulator execution failed',{})
    stages.fail={('candidate','trace')}
    assert evaluate_without_telemetry('candidate-tree')==(0.0,0.0,'build or trace timeout',{})

@pytest.mark.parametrize('step,phase',[('build','build'),('trace','execution')])
def test_reference_side_failures_stay_evaluator_errors(stages,monkeypatch,step,phase):
    # The reference is built and simulated only when the live audit is requested.
    stages.fail={('reference',step)}
    assert evaluate_without_telemetry('candidate-tree')[3]['beats_reference']==1
    monkeypatch.setenv(e.REFERENCE_AUDIT_ENV,'live')
    with pytest.raises(e.EvaluationError,match='private reference '+phase+' failed'):evaluate_without_telemetry('candidate-tree')

def test_reference_output_and_audit_infrastructure_stay_evaluator_errors(stages,monkeypatch):
    stages.outputs['reference']=stages.outputs['reference'][:100]
    monkeypatch.setenv(e.REFERENCE_AUDIT_ENV,'live')
    with pytest.raises(e.EvaluationError,match='private reference execution failed'):evaluate_without_telemetry('candidate-tree')
    monkeypatch.delenv(e.REFERENCE_AUDIT_ENV)
    stages.audit={'infrastructure_error':True}
    with pytest.raises(e.EvaluationError,match='source audit infrastructure unavailable'):evaluate_without_telemetry('candidate-tree')
    # check_tree does not turn missing audit infrastructure into a verdict either.
    with pytest.raises(e.EvaluationError,match='source audit infrastructure unavailable'):
        e.check_tree('candidate-tree',str(PRISTINE),EDITED)

def real_patch(scratch,edit,name='my_cond_branch_predictor.cc'):
    tree=scratch/'repo';shutil.copytree(PRISTINE,tree)
    def git(*args):return subprocess.check_output(['git','-C',str(tree),'-c','user.name=Fixture','-c','user.email=fixture@localhost',*args])
    git('init','-q');git('add','-A');git('commit','-qm','base')
    p=tree/name;p.write_text(edit(p.read_text()))
    patch=git('diff');shutil.rmtree(tree)
    return patch

@pytest.mark.parametrize('edit',[
    lambda s:s+'\n',                                                             # blank line after the last region
    lambda s:s.replace('namespace nRUNLTS {\n','namespace nRUNLTS { \n',1),       # trailing space on an anchor
    lambda s:s.replace('static constexpr int LogMaxNewlyCounters = 16;','static constexpr int  LogMaxNewlyCounters = 16;',1),
])
def test_real_whitespace_change_outside_the_regions_is_named(tmp_path,edit):
    # The real shared edit scope and the real source audit; rejected before any trace is needed.
    with judged(real_patch(tmp_path,edit)) as (scope,verdict):
        assert scope.ok and verdict==('out_of_scope',['my_cond_branch_predictor.cc'])

def test_real_forbidden_word_in_a_region_is_named(tmp_path):
    patch=real_patch(tmp_path,lambda s:s.replace('namespace nRUNLTS {\n\n','namespace nRUNLTS {\n\nstatic int read = 0;\n',1))
    with judged(patch) as (scope,verdict):
        assert scope.ok and verdict==('content_policy',['my_cond_branch_predictor.cc'])


def evaluate_without_telemetry(tree):
    score, unbounded, message, metrics = e.evaluate(tree)
    metrics = dict(metrics)
    metrics.pop("private_runtime_telemetry", None)
    return score, unbounded, message, metrics
