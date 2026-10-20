import deployment_test_guard  # noqa: F401
import sys
from pathlib import Path
TASK = Path(__file__).resolve().parents[3] / "tasks/frontier-cs-2.0-demo/problems/Bioinformatics/edlib_exact_alignment"
sys.path.insert(0, str(TASK / "judge"))
"""Source-phase contracts only. No compilation, reference replay or timing run."""
import shutil
import pytest
import evaluator as e

CLI=('Using NW alignment mode.\nReading queries...\nRead 1 queries, 4 residues total.\n'
     'Reading target fasta file...\nRead target, 4 residues.\n\nComparing queries to target...\n'
     '\nQuery #0 (4 residues): score = 1\nCigar:\n2=1X1=\n\nCpu time of searching: 0.000001234\n')
ALLOCATION='EDLIB_CORE_ALLOCATION peak=64 live_at_exit=4 calls=3 entries=1 stack=96\n'
LOCATION=('Using HW alignment mode.\nReading queries...\nRead 1 queries, 2 residues total.\n'
     'Reading target fasta file...\nRead target, 6 residues.\n\nComparing queries to target...\n'
     '0/1\r1/1\nScores:\n<query number>: <score>, <num_locations>, [(<start_location_in_target>, <end_location_in_target>)]\n'
     '#0: 0  2  [ (2, 3) (4, 5) ]\n\nCpu time of searching: 0.000001234\n')

def test_complete_cli_output_and_cigar():
    distance,ops,seconds=e.parse_cli_run(CLI,'NW','ACGT','ACCT')
    assert distance==1 and ops=='0030' and seconds=='0.000001234'
    e.replay('ACGT','ACCT',0,3,ops,distance)
    assert e.parse_cli_run(ALLOCATION+CLI,'NW','ACGT','ACCT',instrumented=True)[1]=='0030'
    with pytest.raises(e.Invalid): e.cigar_ops('2=trailing')

@pytest.mark.parametrize('output',[
    CLI+'Cpu time of searching: 0.000000001\n',                       # a second timer line
    'Cpu time of searching: 0.000000001\n'+CLI,                       # an earlier timer line
    CLI.replace('\nCpu time','note from the submission\n\nCpu time'),  # anything extra
    CLI.replace('Cigar:\n2=1X1=\n',''),                               # no path printed
    CLI.replace('0.000001234','1e-9'),                                 # not the CLI's %.9f
    CLI.replace('4 residues total','5 residues total'),                # another input
    ALLOCATION+CLI,                                                    # heap evidence in a timed run
    '0/1\r1/1\n\nCpu time of searching: 0.000001234\n',               # the silent mode
    '',
])
def test_cli_output_must_be_exact(output):
    with pytest.raises(e.Invalid): e.parse_cli_run(output,'NW','ACGT','ACCT')

def test_instrumented_run_needs_heap_evidence_and_mode_must_match():
    with pytest.raises(e.Invalid): e.parse_cli_run(CLI,'NW','ACGT','ACCT',instrumented=True)
    with pytest.raises(e.Invalid): e.parse_cli_run(CLI,'HW','ACGT','ACCT')

def test_path_validity_accepts_ties_and_rejects_wrong_cost():
    e.replay('AA','A',0,0,'01',1)
    e.replay('AA','A',0,0,'10',1)
    with pytest.raises(e.Invalid): e.replay('AA','A',0,0,'00',0)
    with pytest.raises(e.Invalid): e.replay('A','T',0,0,'3',0)
    e.replay('N','A',0,0,'0',0,True)

def test_small_independent_distance_oracle():
    assert e.dp('ACGT','ACCT','NW')==(1,[3])
    assert e.dp('AC','TTACGG','HW')==(0,[3])
    assert e.dp('AC','ACGG','SHW')==(0,[1])

def test_strict_per_workload_predicate_and_score_separation():
    workloads=[{'id':'a','length':10000},{'id':'b','length':100000}]
    candidate={'a':1.,'b':2.}; reference={'a':2.,'b':2.}
    decisions=e.audit(workloads,True,True,candidate,reference)
    assert [x['metrics']['cpu_time_seconds'] for x in decisions]==[True,False]
    assert not all(x['metrics']['cpu_time_seconds'] for x in decisions)
    assert not any(x['metrics']['cpu_time_seconds'] for x in e.audit(workloads,False,True,candidate,reference))
    assert all(set(x['metrics'])=={'cpu_time_seconds','core_peak_live_bytes'} for x in decisions)
    # A difference that is not positive is not a reconstruction time: it never passes,
    # and nothing passes against a reference difference that is not positive.
    assert not any(x['metrics']['cpu_time_seconds'] for x in e.audit(workloads,True,True,{'a':0.,'b':-1.},reference))
    assert not any(x['metrics']['cpu_time_seconds'] for x in e.audit(workloads,True,True,{'a':1.,'b':1.},{'a':0.,'b':-2.}))
    assert not any(x['metrics']['cpu_time_seconds'] for x in e.audit(workloads,True,True,reference,reference))
    assert e.reconstruction_seconds([.5,.3,.4],[.25,.2,.9])==.3-.2
    # Storage: equality passes, one byte more fails. Process RSS is not a criterion.
    heap=e.audit(workloads,True,True,candidate,reference,{'a':8,'b':9},{'a':8,'b':8})
    assert [x['metrics']['core_peak_live_bytes'] for x in heap]==[True,False]
    assert 0<e.score(candidate,workloads)<100
    assert e.score({'a':.5,'b':1.},workloads)>e.score(candidate,workloads)

def test_frozen_complete_suite_metadata():
    public=e.rows(False); final=e.rows(True)
    assert len(public)==30 and len(final)==60
    assert {x['id'] for x in public} < {x['id'] for x in final}
    assert len({x['id'] for x in final})==60
    for split in ('public','heldout'):
        assert {(x['length'],x['similarity'],x['mode']) for x in final if x['split']==split}=={
            (n,s,m) for n in (10000,30000) for s in (60,70,80,90,99) for m in ('NW','SHW','HW')}
    assert all(len(x['query_sha256'])==64 and len(x['target_sha256'])==64 for x in final)

def test_unqualified_draft_never_starts_work(monkeypatch):
    monkeypatch.setattr(e,'load_qualification',lambda:{'execution_admitted':False,'reference_qualification_complete':False})
    def forbidden(*a,**k): raise AssertionError('draft must not start executable work')
    monkeypatch.setattr(e,'build',forbidden)
    monkeypatch.setattr(e,'apply_submission_patch',forbidden)
    with pytest.raises(e.IncompleteEvaluation) as failure:
        e.evaluate('not-read.patch')
    evidence=failure.value.private_execution_evidence
    assert 'qualification pending' in evidence['message']
    assert evidence['reference_qualification_complete'] is False

def test_reference_patch_changes_only_the_body():
    # File application only, never execute the restored program.
    result=e.apply_submission_patch(e.ROOT/'reference.patch',e.ROOT,e.CONFIG)
    try:
        assert result.ok,result.message
        e.check_surface(result.workdir)
        import hashlib
        assert hashlib.sha256((result.workdir/'edlib/src/edlib.cpp').read_bytes()).hexdigest()=='b7491bc8bb3feafc23ce6e477d35e3b24ad99d02bd0fd3c77117ef494d76aece'
    finally:
        if result.cleanup_root: shutil.rmtree(result.cleanup_root)

def test_native_filter_is_an_allow_list():
    import native_cmake as n
    rows=n.native_filter_instructions()
    assert rows[-1]==(0x06,0,0,0x00050000|38)                 # default: ENOSYS
    allowed={k for code,jt,jf,k in rows if code==0x15 and (jt,jf)==(0,1)}
    for number in (56,57,58,203,435): assert (0x15,0,1,number) in rows   # still fatal
    # Nothing that creates, renames, removes or re-labels a file, and no IPC or sockets.
    for number in (2,257,85,82,83,84,86,87,88,90,92,76,77,133,188,190,319,29,30,31,64,68,240,41,49,16,72,217,89,62,101,310,311,437,425):
        assert number not in n.NATIVE_ALLOWED_SYSCALLS, number
    assert n.NATIVE_ENV=={'LC_ALL':'C','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}

def test_preprocessor_spellings_are_screened(tmp_path):
    import json
    surface=e.SURFACE
    path=tmp_path/surface['path'];path.parent.mkdir(parents=True)
    for body in ('\n#define X 1\n','\n%:define X 1\n','\n  ??=define X 1\n','\n fopen ("x","w");\n'):
        path.write_text(surface['prefix']+body+surface['suffix'])
        with pytest.raises(e.Invalid): e.check_surface(tmp_path)

def test_self_reported_time_is_only_a_bounded_diagnostic(monkeypatch):
    expected={'distance':1,'starts':[0],'ends':[3]}
    monkeypatch.setattr(e,'run_native',lambda args,deadline:(CLI,1000,0.5))
    out,elapsed,reported,rss=e.checked_cli_run('aligner','NW','ACGT','ACCT','q','t',expected,0)
    assert elapsed==0.5 and reported==0.000001234      # the compared value is the verifier's
    monkeypatch.setattr(e,'run_native',lambda args,deadline:(CLI,1000,0.000001))
    with pytest.raises(e.Invalid,match='exceeds verifier-measured'): e.checked_cli_run('aligner','NW','ACGT','ACCT','q','t',expected,0)
    monkeypatch.setattr(e,'run_native',lambda args,deadline:(CLI.replace('0.000001234','0.000000000'),1000,0.5))
    with pytest.raises(e.Invalid,match='self-reported'): e.checked_cli_run('aligner','NW','ACGT','ACCT','q','t',expected,0)

def test_location_only_output_must_be_exact_and_complete():
    assert e.parse_location_run(LOCATION,'HW','AC','TTACAC')==(0,[(2,3),(4,5)],'0.000001234')
    for output in (LOCATION+'x',LOCATION.replace('  2  [','  3  ['),LOCATION.replace('0/1','note\n0/1'),CLI,
                   LOCATION.replace('6 residues','7 residues'),LOCATION.replace('0.000001234','1e-9'),ALLOCATION+LOCATION,''):
        with pytest.raises(e.Invalid): e.parse_location_run(output,'HW','AC','TTACAC')
    with pytest.raises(e.Invalid): e.parse_location_run(LOCATION,'NW','AC','TTACAC')

def test_location_only_run_is_checked_against_the_reference_result(monkeypatch):
    expected={'distance':0,'starts':[2,4],'ends':[3,5]}
    monkeypatch.setattr(e,'run_native',lambda args,deadline:(LOCATION,1000,0.5))
    assert e.checked_location_run('aligner','HW','AC','TTACAC','q','t',expected,0)[1:3]==(0.5,0.000001234)
    for wrong in ({'distance':1,'starts':[2,4],'ends':[3,5]},{'distance':0,'starts':[2],'ends':[3]},{'distance':0,'starts':[2,4],'ends':[3,4]}):
        with pytest.raises(e.Invalid,match='location'): e.checked_location_run('aligner','HW','AC','TTACAC','q','t',wrong,0)
    monkeypatch.setattr(e,'run_native',lambda args,deadline:(LOCATION,1000,0.000001))
    with pytest.raises(e.Invalid,match='exceeds verifier-measured'): e.checked_location_run('aligner','HW','AC','TTACAC','q','t',expected,0)

@pytest.mark.parametrize('body,inside',[
    ('\n    return 0;\n',True),
    ('\n    if (a) { b(); } else { c(); }\n    char x = \'}\'; const char* y = "}{\\"}";  // }\n    /* } */ int z = 1\'000;\n',True),
    ('\n    return 0;\n}\nstatic void later() {\n',False),                  # closes the function
    ('\n    return 0;\n%>\nstatic void later() <%\n',False),                # the same with digraphs
    ('\n    int z = 1\'0; }\nstatic int w = 0\'1; static void later() {\n',False),   # digit separators are not quotes
    ('\n    const char* s = R"x(}{)x";\n',False),                           # raw strings are not interpreted
    ('\n    int a = 1; \\\n    int b = 2;\n',False),                         # line splices
    ('\n    const char* s = "what??/";\n',False),                           # trigraphs
    ('\n    {\n',False),
])
def test_body_must_stay_inside_its_function(body,inside):
    assert e.body_stays_in_function(body) is inside

def test_scope_escape_is_rejected_by_the_surface_check(tmp_path):
    surface=e.SURFACE
    path=tmp_path/surface['path'];path.parent.mkdir(parents=True)
    path.write_text(surface['prefix']+'\n    return 0;\n}\n__attribute__((constructor)) static void early() {}\nstatic int tail() {\n    return 0;\n'+surface['suffix'])
    with pytest.raises(e.Invalid,match='only reconstruction body may change'): e.check_surface(tmp_path)

def test_storage_evidence_is_one_line_with_a_stack_measurement():
    import core_measurement as c
    assert c.parse_allocation(ALLOCATION+CLI)==64+96
    for output in (CLI,ALLOCATION+ALLOCATION+CLI,ALLOCATION.replace(' stack=96','')+CLI,ALLOCATION.replace('stack=96','stack=0')+CLI,
                   ALLOCATION.replace('peak=64','peak=2')+CLI):
        with pytest.raises(ValueError): c.parse_allocation(output)

# ---- text outside the body: blank lines and line endings only ----

BODY='\n    *alignment = NULL;\n    return EDLIB_STATUS_OK;\n'
PREFIX=e.SURFACE['prefix'];SUFFIX=e.SURFACE['suffix']

def surface_tree(tmp_path,text,newline=None):
    path=tmp_path/e.SURFACE['path'];path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline=newline) as stream: stream.write(text)
    return tmp_path

def category_of(tmp_path,text):
    with pytest.raises(e.Invalid) as failure: e.check_surface(surface_tree(tmp_path,text))
    return failure.value.category

def replace_once(text,old,new):
    assert text.count(old)==1,old
    return text.replace(old,new)

def test_pristine_outside_text_has_nothing_whose_line_structure_carries_meaning():
    # Otherwise locate_body compares bytes and the cases below would not be reached.
    assert not e.UNSAFE_OUTSIDE.search(PREFIX) and not e.UNSAFE_OUTSIDE.search(SUFFIX)
    assert '\r' not in PREFIX+SUFFIX and not PREFIX.endswith('\n') and SUFFIX.startswith('}\n\n\n/**\n')
    assert e.locate_body(PREFIX+BODY+SUFFIX,PREFIX,SUFFIX)==(len(PREFIX),len(PREFIX)+len(BODY))

@pytest.mark.parametrize('before,after',[
    (lambda p:p,lambda s:replace_once(s,'}\n\n\n/**\n * Takes char query','}\n/**\n * Takes char query')),      # the rejected run: fewer blank lines after the body
    (lambda p:p,lambda s:replace_once(s,'}\n\n\n/**\n * Takes char query','}\n\n\n\n\n \t\n/**\n * Takes char query')),
    (lambda p:p,lambda s:s+'\n\n'),
    (lambda p:p,lambda s:s[:-1]),                                                # no newline at end of file
    (lambda p:'\n\n'+p,lambda s:s),
    (lambda p:p.replace('\n\n','\n'),lambda s:s.replace('\n\n','\n')),          # every blank line removed
    (lambda p:p.replace('\n','\n\n'),lambda s:s.replace('\n','\n\n')),          # a blank line after every line
    (lambda p:p.replace(';\n',';  \t\n'),lambda s:s.replace(';\n','; \n')),     # trailing spaces and tabs
    (lambda p:p.replace('\n','\r\n'),lambda s:s.replace('\n','\r\n')),          # CRLF
])
def test_blank_lines_and_line_endings_outside_the_body_are_ignored(tmp_path,before,after):
    prefix=before(PREFIX);suffix=after(SUFFIX)
    assert (prefix,suffix)!=(PREFIX,SUFFIX)
    start,end=e.locate_body(prefix+BODY+suffix,PREFIX,SUFFIX)
    assert (prefix+BODY+suffix)[start:end]==BODY          # the body is the same text, wherever it moved
    e.check_surface(surface_tree(tmp_path,prefix+BODY+suffix,newline=''))

def test_crlf_file_is_accepted(tmp_path):
    e.check_surface(surface_tree(tmp_path,(PREFIX+BODY+SUFFIX).replace('\n','\r\n'),newline=''))

def test_blank_line_only_patch_is_accepted_through_patch_application(tmp_path):
    # The shape of the rejected run: a real diff whose only change outside the body is blank lines.
    import difflib
    pristine=(e.ROOT/'harbor/app'/e.SURFACE['path']).read_text()
    changed=PREFIX+BODY+replace_once(SUFFIX,'}\n\n\n/**\n * Takes char query','}\n\n/**\n * Takes char query')
    name=e.SURFACE['path']
    patch=tmp_path/'solution.patch'
    patch.write_text(''.join(difflib.unified_diff(pristine.splitlines(True),changed.splitlines(True),'a/'+name,'b/'+name)))
    result=e.apply_submission_patch(patch,e.ROOT,e.CONFIG)
    try:
        assert result.ok,result.message
        assert (result.workdir/name).read_text()==changed
        e.check_surface(result.workdir)
    finally:
        if result.cleanup_root: shutil.rmtree(result.cleanup_root)

@pytest.mark.parametrize('side,old,new,category',[
    # a changed line that is not blank
    ('prefix','#include "edlib.h"','#include "edlib.h" // note','outside_before_body'),
    ('prefix','#include "edlib.h"\n','','outside_before_body'),
    ('suffix',' * Takes char query',' * takes char query','outside_after_body'),
    ('suffix','\n/**\n * Takes char query','\nstatic int smuggled = 0;\n/**\n * Takes char query','outside_after_body'),
    # spacing inside a line
    ('prefix','#include "edlib.h"','#include  "edlib.h"','outside_before_body'),
    ('prefix','#include "edlib.h"',' #include "edlib.h"','outside_before_body'),
    ('prefix','#include "edlib.h"','#include "edlib.h"\f','outside_before_body'),
    ('suffix',' * Takes char query','  * Takes char query','outside_after_body'),
    ('suffix',' * Takes char query',' * Takes  char query','outside_after_body'),
    # a line joined to, or split from, its neighbour
    ('suffix','\n/**\n * Takes char query','\n/** * Takes char query','outside_after_body'),
    ('suffix',' * Takes char query',' * Takes char\n query','outside_after_body'),
])
def test_any_other_difference_outside_the_body_is_rejected(tmp_path,side,old,new,category):
    prefix=replace_once(PREFIX,old,new) if side=='prefix' else PREFIX
    suffix=replace_once(SUFFIX,old,new) if side=='suffix' else SUFFIX
    assert category_of(tmp_path,prefix+BODY+suffix)==category

def test_the_signature_line_and_the_closing_brace_are_not_whitespace(tmp_path):
    opening=PREFIX[PREFIX.rfind('\n')+1:]
    assert opening.startswith('        unsigned char** const alignment') and opening.endswith(') {')
    assert category_of(tmp_path,PREFIX[:-len(opening)]+opening.lstrip()+BODY+SUFFIX)=='outside_before_body'
    assert category_of(tmp_path,PREFIX[:-1]+' {'+BODY+SUFFIX)=='outside_before_body'
    assert category_of(tmp_path,PREFIX[:-1]+BODY+SUFFIX)=='outside_before_body'
    assert category_of(tmp_path,PREFIX+BODY+SUFFIX[1:])=='outside_after_body'
    assert category_of(tmp_path,PREFIX+BODY+'};'+SUFFIX[1:])=='outside_after_body'
    assert category_of(tmp_path,PREFIX)=='outside_after_body' and category_of(tmp_path,SUFFIX)=='outside_before_body'
    assert category_of(tmp_path,'')=='outside_before_body'

def test_code_added_after_the_function_is_rejected(tmp_path):
    added='__attribute__((constructor)) static void early() {}\n'
    assert category_of(tmp_path,PREFIX+BODY+'}\nstatic int smuggled = 0;'+SUFFIX[1:])=='outside_after_body'
    # Ending in a brace it reads as body text that closes the function early: still rejected, as a scope escape.
    assert category_of(tmp_path,PREFIX+BODY+'}\n'+added+SUFFIX[1:])=='body_scope'
    assert category_of(tmp_path,PREFIX+BODY+SUFFIX+added)=='outside_after_body'
    assert category_of(tmp_path,PREFIX+BODY+SUFFIX+'\n\n'+added)=='outside_after_body'
    assert category_of(tmp_path,added+PREFIX+BODY+SUFFIX)=='outside_before_body'
    # The boundary is counted from the end of the file, so an earlier copy of the closing text is body text.
    assert category_of(tmp_path,PREFIX+BODY+SUFFIX+SUFFIX)=='body_scope'

@pytest.mark.parametrize('prefix,suffix',[
    (PREFIX,SUFFIX),
    ('\n'+PREFIX.replace('\n\n','\n'),replace_once(SUFFIX,'}\n\n\n/**','}  \n\n\n\n\n/**')),
])
def test_closing_the_function_early_is_a_scope_escape_whatever_the_blank_lines(tmp_path,prefix,suffix):
    # Textually inside the body region: closes the function, defines code at namespace scope, reopens one.
    escape='\n    return 0;\n}\n__attribute__((constructor)) static void early() {}\nstatic int tail() {\n    return 0;\n'
    with pytest.raises(e.Invalid,match='only reconstruction body may change') as failure:
        e.check_surface(surface_tree(tmp_path,prefix+escape+suffix))
    assert failure.value.category=='body_scope'
    start,end=e.locate_body(prefix+escape+suffix,PREFIX,SUFFIX)
    assert (prefix+escape+suffix)[start:end]==escape and not e.body_stays_in_function(escape)

def test_byte_exact_comparison_where_line_structure_can_carry_meaning():
    for unsafe in ('#define TWO 1 \\\n  + 1\n','const char* s = R"x(\n\n)x";\n','// why??/\n'):
        prefix='int before;\n'+unsafe+'void f() {';suffix='}\n\nint after;\n'
        assert e.locate_body(prefix+'x;'+suffix,prefix,suffix)==(len(prefix),len(prefix)+2)
        for changed in (prefix.replace('int before;\n','int before;\n\n')+'x;'+suffix,prefix+'x;'+suffix.replace('\n\n','\n')):
            with pytest.raises(e.Invalid): e.locate_body(changed,prefix,suffix)
    prefix='int before;\nvoid f() {';suffix='}\n\nint after;\n'
    assert e.locate_body('int before;\n\nvoid f() {x;}\nint after;',prefix,suffix)==(23,25)

# ---- what a rejected development submission is told ----

def rejected_evaluation(monkeypatch,tmp_path,text,final,reference=None):
    """_evaluate up to the static checks; anything later is a test failure."""
    import submission_patch
    trees=[surface_tree(tmp_path/'candidate',text),surface_tree(tmp_path/'reference',reference or PREFIX+BODY+SUFFIX)]
    monkeypatch.setattr(e,'load_qualification',lambda:{'execution_admitted':True,'reference_qualification_complete':True})
    monkeypatch.setattr(e,'protect_verifier_process',lambda:None)
    monkeypatch.setattr(e,'apply_submission_patch',lambda *a,**k:submission_patch.PatchResult(True,workdir=trees.pop(0)))
    def forbidden(*a,**k): raise AssertionError('a statically rejected submission must not be built')
    monkeypatch.setattr(e,'build',forbidden)
    return e._evaluate('unused.patch',final)

@pytest.mark.parametrize('text,category',[
    (PREFIX+BODY+SUFFIX.replace(' * Takes char query',' * takes char query'),'outside_after_body'),
    (PREFIX.replace('#include "edlib.h"','#include  "edlib.h"')+BODY+SUFFIX,'outside_before_body'),
    (PREFIX+'\n    return 0;\n}\nstatic int tail() {\n'+SUFFIX,'body_scope'),
    (PREFIX+'\n#define X 1\n'+SUFFIX,'body_forbidden_construct'),
])
def test_train_rejection_returns_only_a_fixed_category(monkeypatch,tmp_path,text,category):
    import candidate_failure
    score,unbounded,message,metrics=rejected_evaluation(monkeypatch,tmp_path,text,False)
    assert (score,unbounded)==(0.,0.) and set(metrics)=={'candidate_diagnostic'}
    visible=metrics['candidate_diagnostic']
    assert visible==candidate_failure.diagnostic(category)==candidate_failure.public_diagnostic(visible)
    assert message==visible['message']=='Rejected: '+candidate_failure.CATEGORIES[category]+'.'
    if category.startswith('outside'): assert message.startswith('Rejected: patch modifies source outside the permitted function body (')

def test_final_rejection_and_reference_faults_keep_the_private_contract(monkeypatch,tmp_path):
    changed=PREFIX+BODY+SUFFIX.replace(' * Takes char query',' * takes char query')
    with pytest.raises(e.IncompleteEvaluation) as failure: rejected_evaluation(monkeypatch,tmp_path/'final',changed,True)
    assert failure.value.private_execution_evidence['message']=='only reconstruction body may change'
    # The same fault in the reference lane is a task fault, never a statement about the submission.
    with pytest.raises(e.IncompleteEvaluation): rejected_evaluation(monkeypatch,tmp_path/'train',PREFIX+BODY+SUFFIX,False,reference=changed)

def test_unaccepted_patch_and_launch_surface_categories(monkeypatch,tmp_path):
    import candidate_failure,submission_patch
    monkeypatch.setattr(e,'load_qualification',lambda:{'execution_admitted':True,'reference_qualification_complete':True})
    monkeypatch.setattr(e,'protect_verifier_process',lambda:None)
    monkeypatch.setattr(e,'apply_submission_patch',lambda *a,**k:submission_patch.PatchResult(False,'patch touches secret/path.cpp'))
    _,_,message,metrics=e._evaluate('unused.patch',False)
    assert metrics=={'candidate_diagnostic':candidate_failure.diagnostic('patch_policy')} and 'secret' not in message
    trees=[surface_tree(tmp_path/name,PREFIX+BODY+SUFFIX) for name in ('candidate','reference')]
    monkeypatch.setattr(e,'apply_submission_patch',lambda *a,**k:submission_patch.PatchResult(True,workdir=trees.pop(0)))
    monkeypatch.setattr(e,'build',lambda *a,**k:(None,None))
    def differs(*a,**k): raise e.Invalid('submission changes program start-up or runtime symbols: private_symbol_name','launch_surface')
    monkeypatch.setattr(e,'check_launch_surface',differs)
    _,_,message,metrics=e._evaluate('unused.patch',False)
    assert metrics=={'candidate_diagnostic':candidate_failure.diagnostic('launch_surface')} and 'private_symbol_name' not in message
    # A failure that is not one of the static categories is unchanged.
    def uncategorised(*a,**k): raise e.Invalid('CLI output malformed')
    monkeypatch.setattr(e,'apply_submission_patch',uncategorised)
    with pytest.raises(e.IncompleteEvaluation): e._evaluate('unused.patch',False)

def test_diagnostic_table_is_closed_and_pinned():
    import candidate_failure,hashlib,json
    assert set(candidate_failure.CATEGORIES)=={'patch_policy','outside_before_body','outside_after_body','body_scope','body_forbidden_construct','launch_surface'}
    for wrong in ('other',None,3,['outside_before_body']):
        with pytest.raises(ValueError): candidate_failure.diagnostic(wrong)
    good=candidate_failure.diagnostic('outside_after_body')
    assert good=={'schema':'edlib.candidate-diagnostic/v1','category':'outside_after_body',
                  'message':'Rejected: patch modifies source outside the permitted function body (after the body).'}
    for bad in (dict(good,message='Rejected: }\n\n\n/**'),dict(good,extra=1),dict(good,schema='x'),{k:v for k,v in good.items() if k!='message'},'x'):
        with pytest.raises(ValueError): candidate_failure.public_diagnostic(bad)
    declared=json.loads((e.ROOT/'candidate-diagnostic.json').read_text())
    assert declared=={'schema':'frontier.candidate-diagnostic-validator/v1','diagnostic_schema':candidate_failure.SCHEMA,
                      'helper_sha256':hashlib.sha256((e.ROOT/'candidate_failure.py').read_bytes()).hexdigest()}
    assert e.CONFIG['evaluation']['solver_visible_metrics']==['candidate_diagnostic']
    assert [x['name'] for x in json.loads((e.ROOT/'beat_spec.json').read_text())['solver_feedback']]==['candidate_diagnostic']
