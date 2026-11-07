"""Task-owned CPU build and external-event checker; contains no queue RTL."""
from __future__ import annotations
import argparse
from collections import defaultdict, deque
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time

# Challenge constants, not material parameters. Independent of candidate metadata.
ENTRYPOINT = "hardware/bbq/generator/bbq.py"
DRAIN = 128
OFFICIAL = ('BASIC_ENQUE', 'BASIC_DEQUE_MIN', 'BASIC_DEQUE_MAX', 'HEAP_PROPERTY',
            'CAPACITY_LIMITS', 'PIPELINING_ENQUE_ENQUE', 'PIPELINING_ENQUE_DEQUE',
            'PIPELINING_DEQUE_DEQUE_MIN', 'PIPELINING_DEQUE_DEQUE_MAX', 'DEQUE_FIFO', 'RESET')
# Verilator warnings that mean simulation and synthesis may see different
# hardware: one variable written by several processes, or by both blocking and
# non-blocking assignments. Every other warning stays non-fatal.
SIMULATION_ERRORS = ('-Werror-MULTIDRIVEN', '-Werror-BLKANDNBLK')


def _load_timing():
    spec = importlib.util.spec_from_file_location(
        'bbq_timing', Path(__file__).resolve().parent/'timing.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


timing = _load_timing()


def run(command, cwd, log, timeout=120):
    start = time.monotonic()
    command = list(map(str, command))
    # Use the standard platform sandbox. Child processes receive no parent tokens.
    # The log's parent is writable in both Agent and Verifier task workspaces.
    with tempfile.TemporaryDirectory(prefix='bbq-process-',
                                     dir=Path(log).resolve().parent) as scratch:
        environment = {
            'PATH': os.pathsep.join((str(Path(sys.executable).parent),
                                    '/usr/local/bin', '/usr/bin', '/bin')),
            'HOME': scratch, 'TMPDIR': scratch, 'LANG': 'C.UTF-8',
            'LC_ALL': 'C.UTF-8', 'PYTHONDONTWRITEBYTECODE': '1',
        }
        with Path(log).open('w') as stream:
            result = subprocess.run(command, cwd=cwd, stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=timeout,
                                    env=environment, close_fds=True)
    if result.returncode:
        raise ValueError('build or simulation failed')
    return time.monotonic() - start


def generate(root, out, depth):
    out.mkdir(parents=True, exist_ok=True)
    rtl = out / 'bbq.sv'
    run([sys.executable, '-B', root/ENTRYPOINT, depth], root, rtl, 20)
    if rtl.stat().st_size > 8_000_000:
        raise ValueError('generated source exceeds limit')
    source = rtl.read_text()
    # Hardware submissions have no file/process/DPI access or other testbench hooks.
    allowed = {'clog2', 'display', 'error', 'finish', 'bits', 'signed', 'unsigned',
               'countones', 'onehot', 'onehot0', 'isunknown', 'size', 'high', 'low'}
    if set(re.findall(r'\$([A-Za-z_][A-Za-z_0-9]*)', source)) - allowed:
        raise ValueError('non-hardware system task in generated source')
    if (re.search(r'\b(?:DPI|force|release|bind|program|interface)\b', source)
            or re.search(r'`(?:include|define|undef)', source.replace('// `define DEBUG', ''))
            or re.search(r'\bmodule\s+(?!bbq\b)', source)
            or source.count('endmodule') != 1 or 'trace_tb' in source
            or re.search(r'\bimport\s+(?!heap_ops::)', source)):
        raise ValueError('generated source violates hardware boundary')
    return rtl


def compile_tb(root, out, rtl, tb, top, flags=()):
    build = out / top
    build.mkdir(exist_ok=True)
    exe = shutil.which('verilator-cli') or shutil.which('verilator')
    if not exe:
        raise RuntimeError('Verilator unavailable')
    # A variable written by more than one process is resolved one way by the
    # simulator and another by synthesis; such a design is invalid.
    run([exe, '--binary', '--timing', '-Wno-fatal', *SIMULATION_ERRORS, '--top-module', top,
         '-j', '2', '--Mdir', build, '-DSIM', *flags,
         root/'hardware/bbq/src/heap_ops.sv', root/'hardware/bbq/src/ffs.sv',
         root/'frontier_cpu/simulation_memory.sv', rtl, tb], out, build/'compile.log')
    return build / ('V' + top)


def build_trace(root, out, case):
    rtl = generate(root, out, case['depth'])
    return compile_tb(root, out, rtl, root/'frontier_cpu/trace_tb.sv', 'trace_tb',
                      [f'-GD={case["depth"]}', f'-GW={case["width"]}',
                       f'-GN={case["capacity"]}'])


def official(root, out):
    rtl = generate(root, out, 2)
    text = (root/'hardware/bbq/tb/bbq/tb_bbq.sv').read_text()
    for name in ('NUM_PIPELINE_STAGES', 'HEAP_NUM_LEVELS'):
        match = re.search(rf'localparam {name}\s*=\s*(\d+)\s*[,;]', rtl.read_text())
        if not match or text.count('bbq_inst.' + name) != 1:
            raise ValueError('upstream testbench portability boundary invalid')
        text = text.replace('bbq_inst.' + name, match[1])
    tb = out/'tb_bbq.sv'
    tb.write_text(text)
    for suffix in OFFICIAL:
        name = 'TEST_' + suffix
        case_out = out/name
        case_out.mkdir(exist_ok=True)
        binary = compile_tb(root, case_out, rtl, tb, 'tb_bbq', [f'-DTEST_CASE="{name}"'])
        log = case_out/'run.log'
        run([binary], case_out, log, 10)
        output = log.read_text()
        if f'PASS {name}' not in output or 'FAIL' in output:
            raise ValueError('official regression failed')


def schedule(case):
    """Construct legal offered schedules with ample free-list reserve.

    Correctness probes are outside the timed interval. IDs never repeat.
    The host model determines priority, never models the RTL's bitmap algorithm.
    """
    rng = random.Random(case['seed'])
    n, priorities = case['capacity'], case['width'] ** case['depth']
    rows, timed = [], []
    size, serial = 0, 0
    def idle(count=DRAIN):
        rows.extend([(0, 0, 0, 0, 0)] * count)
    def reset():
        nonlocal size
        rows.append((1, 0, 0, 0, 0))
        idle(max(n, priorities) + DRAIN)
        size = 0
    def op(kind, priority=0, timing=False, reject=False):
        nonlocal size, serial
        serial += 1
        cycle = len(rows)
        rows.append((0, 1, kind, serial if kind == 0 else 0, priority))
        if not reject:
            size += 1 if kind == 0 else -1
            assert 0 <= size <= n
        if timing:
            timed.append(cycle)
    def drain():
        nonlocal size
        idle()
        while size:
            op(1)
            idle(2)  # isolated FIFO contract
        idle()
    reset()
    # Timed: consecutive subtrees, repeated buckets, mixed ends, then empty flips.
    for _ in range(3):
        count = min(24, n // 3)
        for i in range(count):
            p = rng.choice([0, priorities-1, rng.randrange(priorities)])
            op(0, p, True)
        for i in range(count):
            op(rng.choice([1, 2]), timing=True)
        # Safe refill interval, counted in timed duration if between timed requests.
        idle(24)
        for i in range(24):
            op(0, rng.randrange(priorities), True)
            op(rng.choice([1, 2]), timing=True)
        idle(24)
    idle()
    drain()
    # Rejected empty request and over-capacity request must produce no response.
    op(1, reject=True)
    idle()
    for i in range(n):
        op(0, i % priorities)
    idle()
    op(0, 0, reject=True)
    idle()
    drain()
    # Isolated same-priority FIFO and consecutive end alternation.
    for i in range(20):
        op(0, priorities // 2)
    idle()
    for i in range(10):
        op(1)
        idle(2)
    for i in range(10):
        op(2)
    idle()
    # Cancel an outstanding input at reset, then verify clean reinitialization.
    op(0, 0)
    reset()
    op(2, reject=True)
    idle()
    op(0, priorities-1)
    idle()
    op(1)
    idle()
    return rows, timed


def check(rows, timed, events, capacity):
    accepted, outputs = {}, []
    done = False
    for line in events.splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == 'A' and len(fields) == 3:
            cycle, ready = map(int, fields[1:])
            if cycle in accepted:
                raise ValueError('duplicate acceptance event')
            accepted[cycle] = ready
        elif fields[0] == 'O' and len(fields) == 5:
            outputs.append(tuple(map(int, fields[1:])))
        elif fields == ['DONE', str(len(rows))]:
            done = True
        else:
            raise ValueError('malformed observation')
    if not done or set(accepted) != {i for i,r in enumerate(rows) if r[1] and not r[0]}:
        raise ValueError('missing observation')
    if any(v != 1 for v in accepted.values()):
        raise ValueError('request offered before initialization')
    bycycle = {}
    for output in outputs:
        if output[0] in bycycle or not 0 <= output[0] < len(rows):
            raise ValueError('multiple or out-of-range responses')
        bycycle[output[0]] = output
    buckets = defaultdict(deque)
    size, previous_cycle, previous_priority, previous_blue = 0, -2, -1, False
    expected = deque()
    latencies, completions = [], []
    timed_set = set(timed)
    count = 0
    for cycle, row in enumerate(rows):
        rst, valid, kind, data, p = row
        if rst:
            if any(item[0] in timed_set for item in expected):
                raise ValueError('timed operation cancelled')
            expected.clear()
            buckets.clear()
            size = 0
            previous_cycle = -2
        elif valid:
            legal = size < capacity if kind == 0 else size > 0
            if not legal and cycle in timed_set:
                raise ValueError('illegal timed schedule')
            if legal:
                if kind == 0:
                    buckets[p].append(data)
                    want_data = data
                    blue = True
                    size += 1
                else:
                    p = (min if kind == 1 else max)(p for p,b in buckets.items() if b)
                    blue = (cycle == previous_cycle + 1 and p == previous_priority
                            and not previous_blue)
                    want_data = buckets[p].pop() if blue else buckets[p].popleft()
                    size -= 1
                expected.append((cycle, kind, want_data, p))
                previous_cycle, previous_priority, previous_blue = cycle, p, blue
        if cycle in bycycle:
            if rst or not expected:
                raise ValueError('extra response or reset cancellation failure')
            offered, want_kind, want_data, want_p = expected.popleft()
            _, got_kind, got_data, got_p = bycycle[cycle]
            if (got_kind, got_data, got_p) != (want_kind, want_data, want_p):
                raise ValueError(f'queue semantics failed at cycle {cycle}: expected {(want_kind,want_data,want_p)}, got {(got_kind,got_data,got_p)}')
            if not 0 <= cycle - offered <= DRAIN:
                raise ValueError('response outside trusted latency bound')
            if offered in timed_set:
                count += 1
                latencies.append(cycle-offered)
                completions.append(cycle)
    if expected or size or count != len(timed) or not count:
        raise ValueError('missing response or conservation failure')
    duration = max(completions) - min(timed) + 1
    if duration < count:
        raise ValueError('interface ceiling exceeded')
    return dict(valid=True, count=count, cycles=duration,
                throughput_ops_per_cycle=count/duration, latency_cycles=max(latencies),
                offered_span=max(timed)-min(timed)+1)


def measure(binary, out, case):
    out.mkdir(parents=True, exist_ok=True)
    rows, timed = schedule(case)
    inputs, events = out/'inputs.txt', out/'events.txt'
    inputs.write_text(''.join(' '.join(map(str,row))+'\n' for row in rows))
    run([binary, '+INPUT='+str(inputs), '+EVENTS='+str(events)], out, out/'run.log', 15)
    return check(rows, timed, events.read_text(), case['capacity'])


def clock(build, case, *, expected_memories=None):
    """Estimated minimum clock period of exactly the RTL that was simulated."""
    return timing.measure(build/'bbq.sv', build/'timing', case['width'], case['capacity'],
                          expected_memories=expected_memories)


def physical(row, clocked):
    """Cycle measurements in time units under the fixed timing model."""
    period = clocked['period_ps']
    return dict(row, period_ps=period,
                throughput_ops_per_second=row['count']*1e12/(row['cycles']*period),
                latency_ps=row['latency_cycles']*period,
                lut_count=clocked['lut_count'], register_count=clocked['register_count'],
                memory_bits=clocked['memory_bits'],
                memory_inventory=clocked['memory_inventory'])


def suite(root, out, cases, regressions=True):
    if regressions:
        official(root, out/'official')
    results, builds, clocks = {}, {}, {}
    for case in cases:
        key = tuple(case[k] for k in ('depth','width','capacity'))
        if key not in builds:
            build = out/('build_'+'_'.join(map(str,key)))
            builds[key] = build_trace(root, build, case)
            clocks[key] = clock(build, case)
        results[case['id']] = physical(measure(builds[key], out/case['id'], case), clocks[key])
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--official', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cases = json.loads((root/'frontier_cpu/public_cases.json').read_text())
    print(json.dumps(suite(root, args.workdir.resolve(), cases, args.official), indent=2))
