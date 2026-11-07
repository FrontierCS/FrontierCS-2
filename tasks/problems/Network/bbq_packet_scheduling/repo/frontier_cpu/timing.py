"""Task-owned clock-period estimate for generated BBQ RTL; contains no queue RTL.

The generated SystemVerilog is converted by sv2v, synthesized and mapped to
6-input lookup tables by a fixed Yosys/ABC script, and timed with a fixed
unit-delay model. No placement, routing, vendor tool or random seed is used,
so the result is a deterministic function of the generated source.

    period_ps = REGISTER_PS + max over timing paths of
                (start offset + LUT_PS * lookup tables on the path)

Paths start at clocked register outputs, memory-primitive outputs and module
inputs, and end at register inputs (data, enable, synchronous reset),
memory-primitive inputs and module outputs. Every structural path counts:
there are no multicycle or false-path exceptions.

The timed netlist must be the simulated circuit: a signal driven from more
than one place, or a RAM/FIFO primitive whose output the design reads but
which synthesis removes, makes the design invalid.
"""
from __future__ import annotations
import argparse
import atexit
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
STUBS = HERE / 'timing_memory.v'
SCRIPT = HERE / 'timing_lut6.abc'

# Fixed delay model, in integer picoseconds.
REGISTER_PS = 300           # clock-to-output plus setup of the capturing element
LUT_PS = 500                # one 6-input lookup table including its routing
MEMORY_READ_PS = 1000       # extra clock-to-output of a RAM without output register
FIFO_STATUS_PS = 500        # FIFO empty/full/usedw flags
ENTRY_DWIDTH = 64           # the trace harness instantiates 64-bit entries

MEMORIES = re.compile(r'(free_list|heap_entries|next_pointers|previous_pointers'
                      r'|priority_buckets|bm_l[1-9]|counters_l[1-9])\Z')
# Positive-edge registers, optionally with clock enable and synchronous reset.
REGISTERS = re.compile(r'\$_(DFF_P|DFFE_P[NP]|SDFF_P[NP][01]|SDFFE_P[NP][01][NP]'
                       r'|SDFFCE_P[NP][01][NP])_\Z')
TASKS = re.compile(r'\$(?:display|error|finish)\b\s*(?:\((?:[^()"]|"[^"\n]*")*\))?\s*;')


class TimingError(ValueError):
    """The generated design is outside the timed design rules."""


def check_source(source: str) -> None:
    """Rules that keep simulation and synthesis looking at the same hardware."""
    source = re.sub(r'//[^\n]*|/\*.*?\*/', ' ', source, flags=re.S)
    for directive, argument in re.findall(r'`([A-Za-z_]\w*)(?:[ \t]+(\w+))?', source):
        if directive == 'endif' or (directive == 'ifdef' and argument in ('DEBUG', 'SIM')):
            continue
        raise TimingError('compiler directive outside the fixed generator set')
    if re.search(r"'[sS]?(?:[bB][01_]*|[oO][0-7_]*|[dD]|[hH][0-9a-fA-F_]*)?[xXzZ?]", source):
        raise TimingError('unknown or high-impedance literals are not permitted')
    if re.search(r'\(\*(?!\s*\))', source):
        raise TimingError('synthesis attributes are not permitted')
    if re.search(r'\b(?:negedge|always_latch|specify|primitive|deassign)\b', source):
        raise TimingError('construct outside single-clock synchronous logic')


def _tool(name: str) -> str:
    found = shutil.which(name, path=os.pathsep.join(
        (str(Path(sys.executable).parent), '/usr/local/bin', '/usr/bin', '/bin')))
    if not found:
        raise RuntimeError(name + ' unavailable')
    return found


_scratch_cache = None


def _run(command, cwd, timeout, cache=None):
    environment = {
        'PATH': os.pathsep.join((str(Path(sys.executable).parent),
                                '/usr/local/bin', '/usr/bin', '/bin')),
        'HOME': str(cwd), 'TMPDIR': str(cwd), 'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8',
        'PYTHONDONTWRITEBYTECODE': '1',
        # Confine the WebAssembly tool to its working directory.
        'YOWASP_MOUNT': f'/work={cwd}',
    }
    if cache:
        environment['YOWASP_CACHE_DIR'] = str(cache)
    return subprocess.run(list(map(str, command)), cwd=cwd, capture_output=True,
                          text=True, timeout=timeout, env=environment, close_fds=True)


def _private_cache() -> str:
    """A writable module cache for this process, removed at exit."""
    global _scratch_cache
    _scratch_cache = tempfile.mkdtemp(prefix='yowasp-cache-')
    atexit.register(shutil.rmtree, _scratch_cache, ignore_errors=True)
    return _scratch_cache


def _yosys(script: str, cwd: Path) -> None:
    """Run the pinned Yosys; reuse a precompiled module cache when one is usable."""
    command = [_tool('yowasp-yosys'), '-q', '-p', script]
    shared = not _scratch_cache
    cache = _scratch_cache or os.environ.get('YOWASP_CACHE_DIR') or next(
        (path for path in (Path('/opt/yowasp-cache'), Path.home() / '.cache' / 'YoWASP')
         if path.is_dir()), None) or _private_cache()
    result = _run(command, cwd, 600, cache)
    if result.returncode and shared and 'Preparing to run' in result.stderr:
        # The shared cache is unusable or read-only on this host: compile once
        # into a private cache and keep it for the rest of this process.
        result = _run(command, cwd, 600, _scratch_cache or _private_cache())
    if result.returncode:
        raise TimingError('synthesis failed')


def synthesize(rtl: Path, out: Path, width: int, capacity: int) -> dict:
    """Return the flattened LUT6 netlist of module bbq as Yosys JSON."""
    out.mkdir(parents=True, exist_ok=True)
    source = Path(rtl).read_text()
    check_source(source)
    # Simulation-only reporting tasks have no hardware; SIM stays defined so the
    # synthesized text is the simulated text.
    (out / 'design.sv').write_text(TASKS.sub(';', source))
    for name in ('heap_ops.sv', 'ffs.sv'):
        shutil.copyfile(HERE.parent / 'hardware/bbq/src' / name, out / name)
    shutil.copyfile(STUBS, out / 'memory.v')
    shutil.copyfile(SCRIPT, out / 'lut6.abc')
    converted = _run([_tool('sv2v'), '-DSIM', 'heap_ops.sv', 'ffs.sv', 'design.sv'], out, 60)
    if converted.returncode or len(converted.stdout) > 16_000_000:
        raise TimingError('SystemVerilog conversion failed')
    (out / 'design.v').write_text(converted.stdout)
    for name in ('check.log', 'elaborated.json', 'declared.json', 'stat.log', 'netlist.json'):
        (out / name).unlink(missing_ok=True)
    elaborate = ('read_verilog /work/memory.v; read_verilog -defer /work/design.v; '
                 f'chparam -set HEAP_BITMAP_WIDTH {int(width)} '
                 f'-set HEAP_ENTRY_DWIDTH {ENTRY_DWIDTH} '
                 f'-set HEAP_MAX_NUM_ENTRIES {int(capacity)} bbq; hierarchy -top bbq; ')
    # After elaboration and process conversion, before any optimisation: driver
    # conflicts and the instantiated RAM/FIFO primitives with their readers. A
    # separate invocation, so the synthesis run below (and its period) is unchanged.
    _yosys(elaborate + 'proc; tee -q -o /work/check.log check; write_json /work/elaborated.json',
           out)
    _yosys(elaborate + 'synth -flatten; '
           'abc -lut 6 -script /work/lut6.abc; opt_clean; write_json /work/netlist.json; '
           'tee -q -o /work/stat.log stat', out)
    elaborated = out / 'elaborated.json'
    if elaborated.stat().st_size > 256_000_000:
        raise TimingError('elaborated design exceeds limit')
    declared = declared_memories(json.loads(elaborated.read_text()))
    elaborated.unlink()
    (out / 'declared.json').write_text(json.dumps(declared, indent=1, sort_keys=True))
    check_drivers((out / 'check.log').read_text())
    module = json.loads((out / 'netlist.json').read_text())['modules']['bbq']
    check_memories(declared, module)
    return module


def check_drivers(log: str) -> None:
    """Simulation and synthesis resolve a doubly driven signal differently."""
    if re.search(r'multiple conflicting drivers for', log):
        raise TimingError('signal driven from more than one place')


def _identity(cell):
    """Type and integer parameters, independent of Yosys's parameter encoding."""
    parameters = {}
    for name in sorted(cell.get('parameters', {})):
        try:
            parameters[name] = _parameter(cell, name)
        except ValueError:
            raise TimingError('memory primitive parameters') from None
    return {'type': cell['type'], 'parameters': parameters}


MEMORY_OUTPUTS = ('q', 'empty', 'full', 'usedw')


def declared_memories(design: dict) -> dict:
    """RAM/FIFO primitives of the elaborated design, before optimisation.

    `read` records whether any cell or module output consumes one of the
    primitive's outputs, i.e. whether simulated behaviour can depend on it.
    """
    found = {}
    for module_name, module in design.get('modules', {}).items():
        for name, cell in module.get('cells', {}).items():
            if cell['type'] not in ('bram_simple2port', 'sc_fifo'):
                continue
            if module_name != 'bbq' or name in found:
                raise TimingError('memory primitive outside the generator set')
            found[name] = _identity(cell)
    module = design.get('modules', {}).get('bbq', {})
    consumed = set()
    for name, cell in module.get('cells', {}).items():
        directions = cell.get('port_directions', {})
        for pin, bits in cell.get('connections', {}).items():
            if directions.get(pin, 'input') != 'output' and not (
                    name in found and pin in MEMORY_OUTPUTS):
                consumed.update(bit for bit in bits if isinstance(bit, int))
    for port in module.get('ports', {}).values():
        if port.get('direction') != 'input':
            consumed.update(bit for bit in port['bits'] if isinstance(bit, int))
    for name in found:
        connections = module['cells'][name].get('connections', {})
        found[name]['read'] = any(bit in consumed for pin in MEMORY_OUTPUTS
                                  for bit in connections.get(pin, []))
    return found


def check_memories(declared: dict, module: dict) -> None:
    """A RAM/FIFO primitive the design reads survives synthesis unchanged.

    If synthesis removes a primitive whose output the elaborated design reads,
    the timed netlist is not the simulated circuit. A primitive whose outputs
    nothing reads cannot affect simulated behaviour and may be removed.
    """
    surviving = {name: _identity(cell) for name, cell in module['cells'].items()
                 if cell['type'] in ('bram_simple2port', 'sc_fifo')}
    if any(entry['read'] and name not in surviving for name, entry in declared.items()):
        raise TimingError('memory primitive removed by synthesis')
    for name, identity in surviving.items():
        if name not in declared or {k: declared[name][k] for k in identity} != identity:
            raise TimingError('memory primitive outside the generator set')


def _parameter(cell, name):
    value = cell.get('parameters', {}).get(name, 0)
    return int(value, 2) if isinstance(value, str) else int(value)


def analyse(module: dict, detail: dict | None = None, *, expected_memories=None) -> dict:
    """Design-rule check and longest-path timing of a mapped netlist."""
    clock = module['ports']['clk']['bits']
    names = {}
    for name, net in module['netnames'].items():
        for index, bit in enumerate(net['bits']):
            if isinstance(bit, int) and (bit not in names
                                         or name.count('$') < names[bit].count('$')):
                names[bit] = f'{name}[{index}]'
    arrival, drivers, ends = {}, {}, []
    for name, port in module['ports'].items():
        for bit in port['bits']:
            if not isinstance(bit, int):
                continue
            if port['direction'] == 'input':
                arrival[bit] = 0
            elif port['direction'] == 'output':
                ends.append((bit, name))
            else:
                raise TimingError('bidirectional port')
    luts = registers = memory_bits = 0
    seen = set()
    memories = {}
    for name, cell in module['cells'].items():
        kind = cell['type']
        if kind == '$scopeinfo':
            continue
        directions = cell['port_directions']
        pins = [(pin, bit) for pin, bits in cell['connections'].items()
                for bit in bits if isinstance(bit, int)]
        inputs = [(pin, bit) for pin, bit in pins if directions[pin] == 'input']
        outputs = [(pin, bit) for pin, bit in pins if directions[pin] == 'output']
        if kind == '$lut':
            luts += 1
            for _, bit in outputs:
                drivers[bit] = [b for _, b in inputs]
            continue
        if REGISTERS.match(kind):
            registers += 1
            if cell['connections']['C'] != clock:
                raise TimingError('register not clocked by clk')
            offsets = {}
        elif kind in ('bram_simple2port', 'sc_fifo') and MEMORIES.match(name):
            identity = {'type': kind, 'parameters': cell.get('parameters', {})}
            if expected_memories is not None and expected_memories.get(name) != identity:
                raise TimingError('memory primitive differs from the trusted configuration')
            memories[name] = identity
            if name in seen or cell['connections']['clock'] != clock:
                raise TimingError('memory primitive outside the generator set')
            seen.add(name)
            depth, width = _parameter(cell, 'DEPTH'), _parameter(cell, 'DWIDTH')
            if not 0 < depth <= 1 << 20 or not 0 < width <= 4096:
                raise TimingError('memory primitive parameters')
            memory_bits += depth * width
            if kind == 'sc_fifo':
                if _parameter(cell, 'IS_SHOWAHEAD') != 0 or _parameter(cell, 'IS_OUTDATA_REG') != 1:
                    raise TimingError('unsupported FIFO mode')
                offsets = {'empty': FIFO_STATUS_PS, 'full': FIFO_STATUS_PS,
                           'usedw': FIFO_STATUS_PS}
            else:
                offsets = {} if _parameter(cell, 'IS_OUTDATA_REG') else {'q': MEMORY_READ_PS}
        else:
            # Latches, negative-edge or asynchronous registers, unknown modules,
            # tristates and any cell the flow could not map.
            raise TimingError('cell outside single-clock synchronous logic')
        for pin, bit in outputs:
            arrival[bit] = offsets.get(pin, 0)
        label = names.get(outputs[0][1], name) if REGISTERS.match(kind) and outputs else name
        for pin, bit in inputs:
            if pin not in ('C', 'clock'):
                ends.append((bit, f'{label}.{pin}'))
    # Iterative longest path; an unfinished node met again is a combinational loop.
    for root, _ in ends:
        stack = [root]
        while stack:
            bit = stack[-1]
            if bit in arrival:
                stack.pop()
                continue
            if bit not in drivers:
                raise TimingError('undriven or unmapped net')
            pending = [b for b in drivers[bit] if b not in arrival]
            if pending:
                if arrival.get(('open', bit)):
                    raise TimingError('combinational loop')
                arrival[('open', bit)] = True
                stack.extend(pending)
            else:
                arrival[bit] = LUT_PS + max((arrival[b] for b in drivers[bit]), default=0)
                stack.pop()
    if detail is not None:
        detail.update(arrival=arrival, ends=ends, names=names, drivers=drivers)
    if any(arrival[bit] for bit in clock):
        raise TimingError('clock is not a primary input')
    for fanin in drivers.values():
        if any(bit in clock for bit in fanin):
            raise TimingError('clock used as data')
    worst_bit, worst_end = max(ends, key=lambda end: arrival[end[0]], default=(None, ''))
    path_ps = arrival[worst_bit] if worst_bit is not None else 0
    bit = worst_bit
    while bit in drivers and drivers[bit]:
        bit = max(drivers[bit], key=lambda b: arrival[b])
    return dict(period_ps=REGISTER_PS + path_ps, path_ps=path_ps,
                critical_start=names.get(bit, ''), critical_endpoint=worst_end,
                lut_count=luts,
                register_count=registers, memory_bits=memory_bits,
                memory_inventory=memories)


def measure(rtl: Path, out: Path, width: int, capacity: int, *, expected_memories=None) -> dict:
    return analyse(synthesize(Path(rtl), Path(out), width, capacity),
                   expected_memories=expected_memories)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--rtl', type=Path, required=True, help='generated bbq.sv')
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--capacity', type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(measure(args.rtl.resolve(), args.workdir.resolve(),
                             args.width, args.capacity), indent=2))
