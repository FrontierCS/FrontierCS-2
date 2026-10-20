"""Private judge.

Gate, per workload: correctness, cpu_time_seconds strictly below the reference
and core_peak_live_bytes no greater than the reference.

cpu_time_seconds is a difference of two verifier-measured elapsed times of the
same executable on the same input: the fastest of TIMING_ROUNDS path-printing
runs minus the fastest of TIMING_ROUNDS location-only runs. The location-only
run does everything the path run does up to the reconstruction call, with
none of the editable code, so start-up, input, the retained distance and
location search and the build's own code placement cancel. Every run of both
kinds is validated before its time is used; the program's own timer line is
only a diagnostic. Because the subtraction is only meaningful if nothing of a
submission runs in the location-only mode, the body must stay inside its
function (check_surface) and the built executable must have the reference
build's constructor tables and runtime symbol definitions (elf_surface).

core_peak_live_bytes is requested live heap and mapped bytes plus the stack
high-water mark inside the outermost reconstruction call, plus the writable
and thread-local static storage of the translation unit.

Process peak RSS is recorded for evidence only. Every native execution runs
under the native guard of native_cmake.py, which leaves it no way to create or
modify a file.
"""
from pathlib import Path
import json, math, os, re, shutil, statistics, tempfile, time
from core_measurement import build_core, parse_allocation, static_bytes, WRAPPED
from elf_surface import ElfError, require_same_launch_surface
from native_cmake import build_cmake_entrypoint, build_native_guard, protect_verifier_process, run_bounded, NativeError
import yaml
import candidate_failure
from submission_patch import apply_submission_patch
from judge.workloads import rows, materialize
ROOT=Path(__file__).resolve().parent
AUTHENTIC_ENTRYPOINT='apps/aligner/aligner.cpp'
CONFIG=yaml.safe_load((ROOT/'config.yaml').read_text())
SURFACE=json.loads((ROOT/'judge/surface.json').read_text())
# Operator measurement hooks are never populated by the submission interface.
BEFORE_PROCESS=None
PROCESS_RECEIPTS=None
# Set for the duration of one evaluation: the preloaded stage-2 guard and the
# empty read-only directory every native process starts in.
NATIVE_GUARD=None
NATIVE_CWD=None
RSS_NOTE=('peak_rss_bytes is wait4 ru_maxrss of the native process. It inherits the verifier '
          "process's own pre-exec high-water mark, so it is reported for evidence and is not "
          'part of the comparison.')
# Interleaved rounds per workload; each round is one path run and one
# location-only run of each lane. The per-lane statistic is the minimum.
TIMING_ROUNDS=15
# How the linker names a --wrap redirection depends on whether the name is referenced.
WRAP_ALIASES=frozenset(prefix+name for name in WRAPPED for prefix in ('','__wrap_','__real_'))
GUARD_SELFTEST_EXPECTED='EDLIB_NATIVE_GUARD control_read_only_open control_clock control_heap\n'

class Invalid(Exception):
    """`category`, when set, is a key of candidate_failure.CATEGORIES: the static rejection class."""
    def __init__(self,message,category=None):
        super().__init__(message);self.category=category

class IncompleteEvaluation(RuntimeError):
    """Use the native adapter's failed-execution evidence contract, not scores."""
    def __init__(self, cause, final, timing_failure=None):
        super().__init__('exact alignment evaluation did not complete')
        self.private_execution_evidence = {
            'schema': 'edlib.incomplete-evaluation/v1',
            'mode': 'final' if final else 'train',
            'exception_type': type(cause).__name__[:128],
            'message': str(cause)[:512],
            'reference_qualification_complete': False,
        }


        if timing_failure is not None:
            self.private_execution_evidence['timing_failure'] = timing_failure

def timing_failure_evidence(output, wid, lane, rep, stage, value):
    """Failure-only allowlist; never echo arbitrary process output or environment.

    The runner already decodes UTF-8 with replacement. Only an ASCII timer line
    can therefore be retained byte-exactly; everything else is omitted.
    """
    line = None
    truncated = False
    line_status = 'missing'
    for candidate in output.split('\n'):
        if candidate.startswith('Cpu time of searching: '):
            # Bound retained bytes, but validate the WHOLE line before retaining
            # any prefix. A suffix containing a credential is not a timer line.
            if re.fullmatch(r'Cpu time of searching: [0-9.eE+\-naifNAIF ]*', candidate):
                line = candidate[:128]  # ASCII allowlist: chars == original bytes.
                truncated = len(candidate) > 128
                line_status = 'retained'
            else:
                line_status = 'rejected_non_numeric_line'
            break
    resolution = None
    try:
        measured = time.clock_getres(time.CLOCK_MONOTONIC)
        if math.isfinite(measured) and measured > 0:
            resolution = measured
    except (AttributeError, OSError, ValueError):
        pass
    return {
        'schema': 'edlib.timing-failure/v1',
        'workload': wid if re.fullmatch(r'(public|heldout)-(10000|30000)-(60|70|80|90|99)-(NW|SHW|HW)', wid) else None,
        'lane': lane if lane in ('C', 'R') else None,
        'replicate': rep if type(rep) is int and 0 <= rep < TIMING_ROUNDS else None,
        'stage': stage if stage in ('match', 'parse', 'positive_finite', 'external_bound') else None,
        'raw_timing_line': line,
        'raw_timing_line_status': line_status,
        'raw_timing_line_truncated': truncated,
        'parsed_value': value if type(value) is float and math.isfinite(value) else None,
        'clock': {
            'timer_api': 'clock_gettime(CLOCK_MONOTONIC), CLI self-report (diagnostic; the compared time is verifier-measured)',
            'format': '%.9f (nine fractional digits)',
            'reported_monotonic_resolution_seconds': resolution,
            'resolution_scope': 'verifier process; not measured effective precision of CLI clock_gettime',
            'native_clocks_per_sec': None,
            'libc_identity': 'unavailable',
            'runtime_identity': 'unavailable',
            'clocksource': 'unavailable',
        },
    }

def load_qualification():
    return json.loads((ROOT/'qualification.json').read_text())

def run(args,cwd,deadline):
    return run_bounded(args,cwd,deadline,before_process=BEFORE_PROCESS,receipts=PROCESS_RECEIPTS)

def run_native(args,deadline):
    """Every candidate/reference executable: guarded, read-only directory, fixed environment."""
    if NATIVE_GUARD is None or NATIVE_CWD is None: raise NativeError('native guard is not prepared')
    return run_bounded(args,NATIVE_CWD,deadline,before_process=BEFORE_PROCESS,receipts=PROCESS_RECEIPTS,preload=NATIVE_GUARD)

def prepare_native_guard(work,deadline):
    """Build the guard, then prove on this runtime that it denies every write channel."""
    global NATIVE_GUARD,NATIVE_CWD
    tools=work/'native-guard';tools.mkdir()
    guard,selftest=build_native_guard(tools,deadline,before_process=BEFORE_PROCESS,receipts=PROCESS_RECEIPTS)
    cwd=work/'native-cwd';cwd.mkdir();cwd.chmod(0o555)
    canary=tools/'canary';canary.write_text('edlib native guard canary\n')
    NATIVE_GUARD,NATIVE_CWD=guard,cwd
    before=sorted(x.name for x in tools.iterdir())
    out,_,_=run_native([selftest,canary],deadline)
    if out!=GUARD_SELFTEST_EXPECTED or canary.read_text()!='edlib native guard canary\n' \
            or sorted(x.name for x in tools.iterdir())!=before or any(cwd.iterdir()):
        raise NativeError('native guard self-test failed')

def body_stays_in_function(body):
    """True iff, read as C++ tokens, the body never closes the function it belongs to.

    Line splices, trigraphs and raw string literals are refused outright rather
    than interpreted, so that this reading cannot differ from the compiler's.
    """
    if '??' in body or re.search(r'\\[ \t]*\r?\n',body): return False
    i=0;n=len(body);depth=0
    while i<n:
        c=body[i]
        if body.startswith('//',i):
            j=body.find('\n',i);i=n if j<0 else j
        elif body.startswith('/*',i):
            j=body.find('*/',i+2)
            if j<0: return False
            i=j+2
        elif c.isalpha() or c=='_':
            j=i
            while j<n and (body[j].isalnum() or body[j]=='_'): j+=1
            if j<n and body[j]=='"' and body[i:j] in ('R','LR','uR','UR','u8R'): return False
            i=j
        elif c.isdigit() or (c=='.' and i+1<n and body[i+1].isdigit()):
            j=i+1                      # preprocessing number, including digit separators
            while j<n:
                if body[j].isalnum() or body[j] in '._': j+=1
                elif body[j]=="'" and j+1<n and (body[j+1].isalnum() or body[j+1]=='_'): j+=2
                elif body[j] in '+-' and body[j-1] in 'eEpP': j+=1
                else: break
            i=j
        elif c in '"\'':
            j=i+1
            while j<n and body[j]!=c:
                if body[j]=='\n': return False
                j+=2 if body[j]=='\\' else 1
            if j>=n: return False
            i=j+1
        elif c=='{' or body.startswith('<%',i):
            depth+=1;i+=1 if c=='{' else 2
        elif c=='}' or body.startswith('%>',i):
            depth-=1
            if depth<0: return False
            i+=1 if c=='}' else 2
        else: i+=1
    return depth==0

OUTSIDE_LINE_END=re.compile(r'[ \t]*\r?$')
UNSAFE_OUTSIDE=re.compile(r'\\[ \t\r]*\n|R"|\?\?/')

def outside_lines(text):
    """(start, text) of every line of `text` that is not blank, without its line ending.

    A line ending is any run of spaces and tabs, then at most one carriage
    return, at the end of a line. A line that is empty without it is blank and
    is dropped. Nothing inside a line is touched.
    """
    lines=[];start=0
    for piece in text.split('\n'):
        line=OUTSIDE_LINE_END.sub('',piece)
        if line: lines.append((start,line))
        start+=len(piece)+1
    return lines

def locate_body(s,prefix,suffix):
    """(start, end) of the editable body in `s`; Invalid naming the side that differs.

    Outside the body, `s` must have exactly the non-blank lines of the pristine
    text, in order, each equal up to its line ending (see outside_lines), so
    only the number of blank lines, trailing spaces and tabs, and CRLF against
    LF may differ. The body still starts immediately after the pristine opening
    line of the function and ends immediately before the brace that begins the
    pristine suffix; both are found by counting non-blank lines from the two
    ends of the file, never by searching for text. If the pristine text has a
    line splice, a `??/` trigraph or a raw string literal outside the body,
    where line structure can carry meaning, the comparison is byte-exact.
    """
    before=Invalid('only reconstruction body may change','outside_before_body')
    after=Invalid('only reconstruction body may change','outside_after_body')
    if UNSAFE_OUTSIDE.search(prefix) or UNSAFE_OUTSIDE.search(suffix):
        if not s.startswith(prefix): raise before
        if len(s)<len(prefix)+len(suffix) or not s.endswith(suffix): raise after
        return len(prefix),len(s)-len(suffix)
    lines=outside_lines(s)
    # The prefix ends inside the line that opens the function: that last piece
    # must be the start of the matching line, byte for byte.
    cut=prefix.rfind('\n')+1
    expected=[line for _,line in outside_lines(prefix[:cut])];opening=prefix[cut:]
    if [line for _,line in lines[:len(expected)]]!=expected or len(lines)<=len(expected): raise before
    start,line=lines[len(expected)]
    if not opening or not s.startswith(opening,start): raise before
    start+=len(opening)
    # The suffix begins inside the line that closes the function: that first
    # piece must be the end of the matching line, up to its line ending.
    cut=suffix.find('\n')
    if cut<0: raise after
    closing=suffix[:cut];expected=[line for _,line in outside_lines(suffix[cut+1:])]
    count=len(expected)
    if len(lines)<=count or [line for _,line in lines[len(lines)-count:]]!=expected: raise after
    at,line=lines[len(lines)-count-1]
    if not closing or not line.endswith(closing): raise after
    end=at+len(line)-len(closing)
    if end<start: raise after
    return start,end

def check_surface(tree):
    s=(tree/SURFACE['path']).read_text()
    start,end=locate_body(s,SURFACE['prefix'],SURFACE['suffix'])
    body=s[start:end]
    # A courtesy screen for obvious mistakes, not the enforcement: the native guard
    # is what denies files, processes and state to the running executable.
    if re.search(r'^\s*(?:#|%:|\?\?=)',body,re.M) or re.search(r'\b(?:system|popen|fork|execve|fopen|freopen|clock|setenv|getenv|asm)\s*\(',body):
        raise Invalid('reconstruction may not access environment, processes, files or timing','body_forbidden_construct')
    # The only editable text is one function body: nothing at namespace scope.
    if not body_stays_in_function(body): raise Invalid('only reconstruction body may change','body_scope')

def check_launch_surface(candidate,reference,ignore=()):
    """Nothing of a submission may run outside the reconstruction call (see elf_surface)."""
    try:
        require_same_launch_surface(candidate.resolve_for_execution(),reference.resolve_for_execution(),ignore)
    except ElfError as error:
        raise Invalid('submission changes program start-up or runtime symbols: '+str(error),'launch_surface') from error

def build(tree,deadline,sanitize=False):
    if not (tree/AUTHENTIC_ENTRYPOINT).is_file():
        raise Invalid('original FASTA aligner entrypoint missing')
    b=tree/('sanitized' if sanitize else 'build')
    aligner=build_cmake_entrypoint(tree,b,AUTHENTIC_ENTRYPOINT,'edlib-aligner',deadline,
        sanitize=sanitize,before_process=BEFORE_PROCESS,receipts=PROCESS_RECEIPTS)
    probe=b/'bin/api-probe'
    run(['c++','-std=c++14','-O3','-I',tree/'edlib/include',ROOT/'judge/api_probe.cpp',b/'lib/libedlib.a','-o',probe],tree,deadline)
    return aligner,probe

def equal(a,b,amb): return a==b or (amb and ('N'==a and b in 'ACGT' or 'N'==b and a in 'ACGT'))
def dp(q,t,mode,amb=False):
    prev=[0]* (len(t)+1) if mode=='HW' else list(range(len(t)+1))
    for i,a in enumerate(q,1):
        cur=[i]
        for j,b in enumerate(t,1): cur.append(min(prev[j]+1,cur[-1]+1,prev[j-1]+(not equal(a,b,amb))))
        prev=cur
    d=prev[-1] if mode=='NW' else min(prev)
    ends=[len(t)-1] if mode=='NW' else [j-1 for j,v in enumerate(prev) if v==d]
    return d,ends

def replay(q,t,start,end,ops,d,amb=False):
    i=0;j=start;cost=0
    for op in ops:
        if op in '03':
            if i>=len(q) or j>end or (op=='0')!=equal(q[i],t[j],amb): raise Invalid('invalid path operation')
            i+=1;j+=1;cost+=op=='3'
        elif op=='1': i+=1;cost+=1
        elif op=='2': j+=1;cost+=1
        else: raise Invalid('invalid path alphabet')
        if i>len(q) or j>end+1: raise Invalid('path overrun')
    if i!=len(q) or j!=end+1 or cost!=d: raise Invalid('path is missing or nonoptimal')

def cigar_ops(cigar):
    parts=re.findall(r'([1-9][0-9]*)([=XID])',cigar)
    if ''.join(a+b for a,b in parts)!=cigar: raise Invalid('malformed CIGAR')
    if sum(int(a) for a,b in parts)>4000000: raise Invalid('oversized CIGAR')
    return ''.join({'=':'0','I':'1','D':'2','X':'3'}[b]*int(a) for a,b in parts)

CLI_HEADER=(r'Using (?P<mode>NW|SHW|HW) alignment mode\.\n'
    r'Reading queries\.\.\.\n'
    r'Read 1 queries, (?P<query>\d+) residues total\.\n'
    r'Reading target fasta file\.\.\.\n'
    r'Read target, (?P<target>\d+) residues\.\n'
    r'\nComparing queries to target\.\.\.\n')
ALLOCATION_LINE=r'EDLIB_CORE_ALLOCATION peak=\d+ live_at_exit=\d+ calls=\d+ entries=\d+ stack=\d+\n'
CLI_OUTPUT=re.compile(
    r'(?P<allocation>(?:'+ALLOCATION_LINE+r')?)'+CLI_HEADER+
    r'\nQuery #0 \((?P<residues>\d+) residues\): score = (?P<distance>\d+)\n'
    r'Cigar:\n(?P<cigar>[0-9=IDX]+)\n'
    r'\nCpu time of searching: (?P<seconds>[0-9]+\.[0-9]{9})\n')

def parse_cli_run(output,mode,q,t,instrumented=False):
    """The complete output of one path-printing CLI run, and nothing else.

    Any extra, missing, repeated or reordered byte is invalid, so a second
    timer line or a note printed by the submission cannot be read as the result.
    Returns (distance, operations, seconds text).
    """
    match=CLI_OUTPUT.fullmatch(output)
    if not match: raise Invalid('CLI output malformed')
    if bool(match['allocation'])!=instrumented: raise Invalid('CLI output malformed')
    if match['mode']!=mode or int(match['query'])!=len(q) or int(match['residues'])!=len(q) or int(match['target'])!=len(t):
        raise Invalid('CLI output does not describe this workload')
    return int(match['distance']),cigar_ops(match['cigar']),match['seconds']

LOCATION_OUTPUT=re.compile(CLI_HEADER+
    r'0/1\r1/1\nScores:\n'
    r'<query number>: <score>, <num_locations>, \[\(<start_location_in_target>, <end_location_in_target>\)\]\n'
    r'#0: (?P<distance>\d+)  (?P<count>\d+)  \[(?P<locations>(?: \(\d+, -?\d+\))+) \]\n'
    r'\nCpu time of searching: (?P<seconds>[0-9]+\.[0-9]{9})\n')

def parse_location_run(output,mode,q,t):
    """The complete output of one location-only CLI run, and nothing else.

    Returns (distance, [(start, end), ...], seconds text).
    """
    match=LOCATION_OUTPUT.fullmatch(output)
    if not match: raise Invalid('CLI output malformed')
    if match['mode']!=mode or int(match['query'])!=len(q) or int(match['target'])!=len(t):
        raise Invalid('CLI output does not describe this workload')
    locations=[(int(a),int(b)) for a,b in re.findall(r'\((\d+), (-?\d+)\)',match['locations'])]
    if len(locations)!=int(match['count']): raise Invalid('CLI output malformed')
    return int(match['distance']),locations,match['seconds']

def checked_duration(out,seconds,elapsed):
    """The program's own timer line: well formed, positive, no greater than the verifier's time."""
    reported=float(seconds)
    if not math.isfinite(reported) or reported<=0:
        error=Invalid('invalid self-reported duration');error.cli_output=out;error.stage='positive_finite';error.value=reported
        raise error
    if reported>elapsed:
        error=Invalid('self-reported duration exceeds verifier-measured elapsed time');error.cli_output=out;error.stage='external_bound';error.value=reported
        raise error
    return reported

def checked_location_run(binary,mode,q,t,qp,tp,expected,deadline):
    """Run the location-only invocation and validate what it printed.

    `expected` is the reference API location result for this workload. The
    printed distance and every printed (start, end) pair must equal it. Returns
    (output, verifier elapsed seconds, self-reported seconds, peak RSS).
    """
    out,rss,elapsed=run_native([binary,'-m',mode,'-l','-r','1','-k','-1',qp,tp],deadline)
    try:
        distance,locations,seconds=parse_location_run(out,mode,q,t)
    except Invalid as error:
        error.cli_output=out
        raise
    if distance!=expected['distance'] or locations!=list(zip(expected['starts'],expected['ends'])):
        raise Invalid('CLI distance or location disagreement')
    return out,elapsed,checked_duration(out,seconds,elapsed),rss

def checked_cli_run(binary,mode,q,t,qp,tp,expected,deadline,instrumented=False):
    """Run the one CLI invocation used everywhere and validate what it printed.

    `expected` is the reference API result for this workload, already
    cross-checked against the distance-only engine. The printed distance must
    equal it and the printed path must replay on the same target interval with
    cost equal to that distance. The program's own timer line is a diagnostic:
    it must be well formed, positive and no greater than the elapsed time the
    verifier measured around the whole process, which is the value returned
    for comparison. Returns (output, verifier elapsed seconds, self-reported
    seconds, peak RSS).
    """
    out,rss,elapsed=run_native([binary,'-m',mode,'-p','-f','CIG_EXT','-r','1','-k','-1',qp,tp],deadline)
    try:
        distance,ops,seconds=parse_cli_run(out,mode,q,t,instrumented)
    except Invalid as error:
        error.cli_output=out
        raise
    if distance!=expected['distance']: raise Invalid('CLI distance disagreement')
    replay(q,t,expected['starts'][0],expected['ends'][0],ops,distance)
    return out,elapsed,checked_duration(out,seconds,elapsed),rss

def probe(exe,qp,tp,mode,task,k,amb,deadline):
    out,_,_=run_native([exe,qp,tp,['NW','SHW','HW'].index(mode),task,k,int(amb)],deadline)
    return json.loads(out)

def validate(c,r,q,t,mode,task,k,amb):
    if c['status']!=0 or r['status']!=0: raise Invalid('API status')
    if c['distance']!=r['distance'] or c['ends']!=r['ends'] or c['starts']!=r['starts']: raise Invalid('distance or location contract')
    d=c['distance']
    if len(q)*len(t)<=100000 and q and t:
        exact,ends=dp(q,t,mode,amb)
        expected=-1 if k>=0 and exact>k else exact
        if d!=expected or (d>=0 and c['ends']!=ends): raise Invalid('independent oracle mismatch')
    # Upstream empty-input fast return supplies distance/ends but no path/starts,
    # and does not apply finite k. Preserve that exact pinned API behavior.
    if task==2 and d>=0 and q and t:
        if not c['starts'] or not c['ends'] or len(c['ops'])!=c['length']: raise Invalid('missing path')
        replay(q,t,c['starts'][0],c['ends'][0],c['ops'],d,amb)
        if cigar_ops(c['cigar'])!=c['ops']: raise Invalid('CIGAR does not encode path')
    elif c['ops']!=r['ops'] or c['length']!=r['length']: raise Invalid('unexpected path output')

def correctness(cprobe,rprobe,work,final,deadline):
    import random
    seeds=[18421,940771] if final else [18421]
    pairs=[('',''),('A',''),('','T'),('ACGT','ACCT'),('AAAA','AAAAAAAA'),('NNAC','TTAC')]
    for seed in seeds:
        rng=random.Random(seed)
        for n in [1,2,63,64,65,127,128,129,1792,1856,1920,4097]:
            q=''.join(rng.choice('ACGTN') for _ in range(n)); t='GT'+q[1:]+'AC'
            pairs.append((q,t))
    for idx,(q,t) in enumerate(pairs):
        qp=work/'q.fa';tp=work/'t.fa';qp.write_text('>q\n'+q+'\n');tp.write_text('>t\n'+t+'\n')
        for mode in ['NW','SHW','HW']:
            for task in [0,1,2]:
                for k in [-1,0,2, max(len(q),len(t))]:
                    for amb in [False,True]:
                        r=probe(rprobe,qp,tp,mode,task,k,amb,deadline); c=probe(cprobe,qp,tp,mode,task,k,amb,deadline)
                        validate(c,r,q,t,mode,task,k,amb)

def score(times,workloads):
    # Absolute challenge scales avoid disclosing reference measurements via score.
    return 100*statistics.mean((w['length']/100000)**2 / ((w['length']/100000)**2+times[w['id']]) for w in workloads)

def reconstruction_seconds(path_samples,location_samples):
    """Fastest path run minus fastest location-only run of one executable on one input."""
    return min(path_samples)-min(location_samples)

def audit(workloads,qualified=False,correct=False,ct=None,rt=None,ca=None,ra=None):
    """The whole comparison: reconstruction time strictly lower, core storage no greater.

    A candidate difference that is not positive is not a measurement of a
    reconstruction and never passes; neither does any row whose reference
    difference is not positive.
    """
    ct=ct or {};rt=rt or {};ca=ca or {};ra=ra or {}
    return [{'workload':w['id'],'metrics':{'cpu_time_seconds':bool(qualified and correct and w['id'] in ct and w['id'] in rt and 0<ct[w['id']]<rt[w['id']]),'core_peak_live_bytes':bool(qualified and correct and w['id'] in ca and w['id'] in ra and 0 <= ca[w['id']] <= ra[w['id']])}} for w in workloads]

def evaluate(solution_path):
    return _evaluate(solution_path,os.environ.get('FRONTIER_SUBMISSION_ROLE')=='final')

def measure_reference(final=False):
    """Operator-only local fit measurement; cannot qualify a submission or award PASS.

    Exercises the identical complete evaluator with the reference in both
    lanes. It leaves qualification.json false and the public entrypoint closed.
    """
    return _evaluate(ROOT/'reference.patch',final,resource_measurement=True)

def _evaluate(solution_path,final,resource_measurement=False):
    workloads=rows(final)
    global NATIVE_GUARD,NATIVE_CWD
    metrics={'beats_reference':0,'beat_spec_workload_passes':audit(workloads),'reference_score':None,'margin':None,'development_cpu_time_seconds_median':None,'development_peak_rss_bytes_max':None,'reference_qualification_complete':False,'peak_rss_bytes_gated':False,'peak_rss_bytes_note':RSS_NOTE}
    patched=[]
    timing_failure=None
    candidate_static=True
    try:
        qualification=load_qualification()
        if not resource_measurement and not qualification.get('execution_admitted',False):
            raise Invalid('reference qualification pending; evaluation is not admitted')
        deadline=time.monotonic()+CONFIG['runtime']['timeout_seconds']
        protect_verifier_process()
        for path in [solution_path,ROOT/'reference.patch']:
            if BEFORE_PROCESS is not None: BEFORE_PROCESS()
            result=apply_submission_patch(path,ROOT,CONFIG)
            if not result.ok: raise Invalid('invalid submission patch','patch_policy')
            patched.append(result); check_surface(result.workdir)
            # A category names a fault of the submission, never of the reference.
            candidate_static=False
        cb,cp=build(patched[0].workdir,deadline); rb,rp=build(patched[1].workdir,deadline)
        check_launch_surface(cb,rb)
        ac=build_core(patched[0].workdir,deadline,before_process=BEFORE_PROCESS,receipts=PROCESS_RECEIPTS)
        ar=build_core(patched[1].workdir,deadline,before_process=BEFORE_PROCESS,receipts=PROCESS_RECEIPTS)
        check_launch_surface(ac,ar,WRAP_ALIASES)
        static={'C':static_bytes(ac),'R':static_bytes(ar)}
        with tempfile.TemporaryDirectory(prefix='edlib-eval-') as wd:
            work=Path(wd);prepare_native_guard(work,deadline);correctness(cp,rp,work,final,deadline)
            ct={};rt={};pt={};rssc={};rssr={};samples={};location_samples={};distances={};rss_samples={};own_samples={};allocc={};allocr={};alloc_samples={}
            for w in workloads:
                q,t,qp,tp=materialize(w,work); mode=w['mode']; wid=w['id']
                r=probe(rp,qp,tp,mode,2,-1,False,deadline); c=probe(cp,qp,tp,mode,2,-1,False,deadline)
                validate(c,r,q,t,mode,2,-1,False)
                # Retained distance-only engine certifies the large optimum.
                rd=probe(rp,qp,tp,mode,0,-1,False,deadline)
                if rd['distance']!=r['distance']: raise Invalid('large distance disagreement')
                # Reference API location result: what every location-only run must print.
                rl=probe(rp,qp,tp,mode,1,-1,False,deadline)
                if rl['distance']!=r['distance'] or rl['starts'][:1]!=r['starts'][:1] or rl['ends'][:1]!=r['ends'][:1]:
                    raise Invalid('large location disagreement')
                distances[wid]={'distance':r['distance'],'actual_similarity':1-r['distance']/min(len(q),len(t))}
                # One untimed path execution per lane, then TIMING_ROUNDS interleaved
                # rounds of a path run and a location-only run per lane. Path runs
                # share one command line with the untimed and instrumented ones, so
                # no execution can tell that it is a timed one, and every run of
                # either kind is validated in full before its time is accepted.
                for binary in (cb,rb): checked_cli_run(binary,mode,q,t,qp,tp,r,deadline)
                values={'C':[],'R':[]};located={'C':[],'R':[]};mem={'C':[],'R':[]};own={'C':[],'R':[]}
                for rep_index in range(TIMING_ROUNDS):
                    lanes=[('C',cb),('R',rb)] if rep_index%2==0 else [('R',rb),('C',cb)]
                    for kind in ('path','location'):
                        for label,binary in lanes:
                            try:
                                if kind=='path':
                                    out,value,reported,rss=checked_cli_run(binary,mode,q,t,qp,tp,r,deadline)
                                    values[label].append(value);mem[label].append(rss);own[label].append(reported)
                                else:
                                    out,value,reported,rss=checked_location_run(binary,mode,q,t,qp,tp,rl,deadline)
                                    located[label].append(value)
                            except Invalid as error:
                                # Diagnostics must never replace the original failure.
                                try:
                                    timing_failure=timing_failure_evidence(getattr(error,'cli_output',''),wid,label,rep_index,getattr(error,'stage','match'),getattr(error,'value',None))
                                except Exception:
                                    pass
                                raise
                ct[wid]=reconstruction_seconds(values['C'],located['C']);rt[wid]=reconstruction_seconds(values['R'],located['R'])
                pt[wid]=statistics.median(values['C'])
                rssc[wid]=max(mem['C']);rssr[wid]=max(mem['R']);samples[wid]=values;location_samples[wid]=located;rss_samples[wid]=mem;own_samples[wid]=own
                allocation={'C':[],'R':[]}
                for rep in range(5):
                    for label,binary in ([('C',ac),('R',ar)] if rep%2==0 else [('R',ar),('C',ac)]):
                        out,_,_,_=checked_cli_run(binary,mode,q,t,qp,tp,r,deadline,instrumented=True)
                        allocation[label].append(parse_allocation(out)+static[label])
                allocc[wid]=max(allocation['C']);allocr[wid]=max(allocation['R']);alloc_samples[wid]=allocation
            qualification=load_qualification()
            qualified=not resource_measurement and qualification.get('reference_qualification_complete') is True
            reference_path={wid:statistics.median(samples[wid]['R']) for wid in samples}
            candidate_score=score(pt,workloads);reference_score=score(reference_path,workloads)
            decisions=audit(workloads,qualified,True,ct,rt,allocc,allocr)
            metrics.update(beats_reference=int(all(all(x['metrics'].values()) for x in decisions)),beat_spec_workload_passes=decisions,reference_score=reference_score,margin=candidate_score-reference_score,reference_qualification_complete=qualified,cpu_time_seconds_candidate_by_workload=ct,cpu_time_seconds_reference_by_workload=rt,cpu_time_seconds_replicates_by_workload=samples,location_seconds_replicates_by_workload=location_samples,path_seconds_median_candidate_by_workload=pt,path_seconds_median_reference_by_workload=reference_path,timing_rounds=TIMING_ROUNDS,static_storage_bytes={'candidate':static['C'],'reference':static['R']},self_reported_seconds_replicates_by_workload=own_samples,peak_rss_bytes_candidate_by_workload=rssc,peak_rss_bytes_reference_by_workload=rssr,peak_rss_bytes_replicates_by_workload=rss_samples,core_peak_live_bytes_candidate_by_workload=allocc,core_peak_live_bytes_reference_by_workload=allocr,core_peak_live_bytes_replicates_by_workload=alloc_samples,private_expected_distances_endpoints_and_paths=distances)
            if not final:
                metrics.update(development_cpu_time_seconds_median=statistics.median(pt.values()),development_peak_rss_bytes_max=max(rssc.values()))
            return candidate_score,candidate_score,'valid; scientific comparison qualified' if qualified else 'valid; reference qualification pending',metrics
    except Exception as exc:
        if resource_measurement:
            import traceback
            metrics['operator_exception']=traceback.format_exc()
            message=str(exc) if isinstance(exc,(Invalid,NativeError)) else 'evaluation failed safely'
            return 0.,0.,message,metrics
        # A development submission rejected by a static check is told which
        # one: a fixed category and sentence (candidate_failure.py), which the
        # verifier admits to train feedback only as the sole metric of a scored
        # zero. Final keeps the private failed-execution contract below.
        category=getattr(exc,'category',None)
        if not final and category is not None and (candidate_static or category=='launch_surface'):
            visible=candidate_failure.diagnostic(category)
            return 0.,0.,visible['message'],{'candidate_diagnostic':visible}
        # Incomplete measurement has no finite reference_score or margin. A
        # scored-zero tuple would enter final beat-spec validation and mask the
        # actual failure with "invalid private beat-spec metrics". The existing
        # adapter records this bounded cause privately and emits a failed result
        # without a numerical score. Completed-suite scoring above is unchanged.
        raise IncompleteEvaluation(exc, final, timing_failure) from exc
    finally:
        NATIVE_GUARD=NATIVE_CWD=None
        for result in patched:
            if result.cleanup_root: shutil.rmtree(result.cleanup_root,ignore_errors=True)

if __name__=='__main__':
    import sys
    print(json.dumps(evaluate(sys.argv[1])))
