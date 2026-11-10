#!/usr/bin/env python3
"""Operator-only local physical proof; not accepted native or scientific evidence.

Run only after review of the exact source, existing image digest and command.
Uses one disposable restricted Docker container, never a builder or task run.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid


def command(arguments, *, timeout=5):
    return subprocess.run(arguments, check=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, timeout=timeout).stdout


def inspect(container):
    value = json.loads(command(['docker', 'inspect', container], timeout=2))
    if not isinstance(value, list) or len(value) != 1:
        raise ValueError('exact disposable container inspect required')
    return value[0]


def counters(path):
    result = {}
    for line in path.read_text().splitlines():
        key, value = line.split()
        result[key] = int(value)
    return result


def cgroup_snapshot(root):
    return {name: (counters(root / name) if name in {'memory.events', 'memory.events.local', 'cgroup.events'}
                   else (root / name).read_text().strip())
            for name in ('memory.events', 'memory.events.local', 'memory.current', 'memory.peak',
                         'memory.max', 'memory.swap.max', 'cgroup.events')}


def child_program(case):
    if case == 'oom':
        # Keep stdout/stderr silent: malformed output must not short-circuit
        # the real memcg kill before its independent runtime observation.
        allocator = 'blocks=[]\nwhile True:\n blocks.append(bytearray(1048576))\n'
        # A tiny nonallocating PID1 retains the actual child memcg event for
        # five seconds. Exit status derives from the real allocator process;
        # local oom_kill delta AND Docker OOMKilled remain mandatory.
        return ('import os,subprocess,sys,time\ntime.sleep(3)\n'
                f'child=subprocess.Popen([sys.executable,"-B","-c",{allocator!r}],'
                'stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\n'
                'status=child.wait()\ntime.sleep(5)\nos._exit(128-status if status<0 else status)\n')
    if case == 'watchdog':
        return ('import os,signal,time\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\n'
                'os.close(1)\nos.close(2)\nwhile True: time.sleep(1)\n')
    if case == 'dribble':
        return ('import os,signal,struct,time\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\n'
                'os.write(1,struct.pack("!I",65000))\n'
                'while True:\n os.write(1,b"x")\n time.sleep(1)\n')
    if case == 'healthy':
        return ('import json,os,struct\nb=json.dumps({"schema":"frontier.vllm-child-result-draft/v1",'
                '"result":{"healthy":True}},separators=(",",":")).encode()\n'
                'os.write(1,struct.pack("!I",len(b))+b)\n')
    raise ValueError('unknown physical proof case')


def assess(case, record, *, before, after, runtime, elapsed, cleanup, tolerance):
    reasons = []
    if not cleanup['container_absent'] or not cleanup['cgroup_empty_or_removed']:
        reasons.append('physical cleanup was not independently confirmed')
    if elapsed > 25 + tolerance:
        reasons.append('owned20s work plus5s cleanup allocation exceeded reviewed scheduling tolerance')
    if case == 'oom':
        if before is None or after is None:
            reasons.append('memcg before/after observation missing')
        else:
            delta = after['memory.events.local']['oom_kill'] - before['memory.events.local']['oom_kill']
            if delta <= 0:
                reasons.append('no physical local oom_kill delta')
        if runtime.get('OOMKilled') is not True or runtime.get('ExitCode') != 137:
            reasons.append('runtime did not confirm memcg OOMKilled/137')
        if record['status'] != 'runtime_error' or record['cause'] != 'child_exit':
            reasons.append('real supervisor did not retain runtime_error')
    elif case in {'watchdog', 'dribble'}:
        if record['status'] != 'timeout' or record['cause'] != 'deadline':
            reasons.append('absolute source watchdog did not select deadline')
    elif record['status'] != 'complete' or record['result'] != {'healthy': True}:
        reasons.append('subsequent healthy sibling did not complete')
    return reasons


def run_case(args, case, source, output, identity):
    from containment_policy import Boundary
    from cpu_controller import CPUController
    from controller_identity import ENVIRONMENT, load_limits
    from instance_supervisor import Limits, run
    from terminal_contract import Slot, canonical
    output.mkdir(mode=0o700)
    program = child_program(case)
    image = args.image
    name = 'frontier-local-physical-' + uuid.uuid4().hex
    create = ['docker', 'create', '--pull=never', '--name', name, '--network', 'none', '--memory', '128m',
              '--memory-swap', '128m', '--cpus', '1', '--pids-limit', '64', '--read-only',
              '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', image,
              'python3', '-B', '-c', program]
    slot = Slot(identity['run_id'], identity['attempt_id'], identity['verification_id'], name,
                'eval', 'local-physical/' + case, 'candidate', 'one', args.source_content_digest,
                'sha256:' + hashlib.sha256(program.encode()).hexdigest())
    started = time.monotonic()
    controller = CPUController((slot,), environment={v: identity[k] for k, v in ENVIRONMENT.items()},
        content_contract=source / 'controller-contract.json', source_root=source,
        evidence=output / 'cpu-controller', maximum_terminal_bytes=65536,
        maximum_manifest_bytes=65536, deadline=started + 20, started=started,
        **load_limits(source / 'controller-contract-limits.json'))
    if (controller.signer.context.source_content_digest != args.source_content_digest
            or controller.signer.context.contract_digest != args.contract_digest):
        raise ValueError('physical proof differs from independent source/contract pins')
    before = after = cgroup = runtime = None
    container, create_attempted = None, False
    cleanup = {'container_absent': False, 'cgroup_empty_or_removed': False}
    original_error = None
    record = None
    try:
        # Inspect local presence prevents docker create from implicitly pulling.
        command(['docker', 'image', 'inspect', image])
        create_attempted = True
        container = command(create).decode().strip()
        if not re.fullmatch('[0-9a-f]{64}', container):
            raise ValueError('exact newly created Docker container required')
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            remaining = controller.deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('local controller work allocation expired before child dispatch')
            future = pool.submit(run, controller.ledger, slot, ['docker', 'start', '--attach', container],
                Limits(remaining, 5, 65536, 65536, 32), boundary=Boundary('candidate_service', 'execution'))
            while not future.done():
                state = inspect(container)['State']
                pid = state['Pid']
                if cgroup is None and pid:
                    entries = Path(f'/proc/{pid}/cgroup').read_text().splitlines()
                    unified = [line[3:] for line in entries if line.startswith('0::/')]
                    if len(unified) != 1:
                        raise ValueError('exact unified child cgroup required')
                    candidate = Path('/sys/fs/cgroup') / unified[0].lstrip('/')
                    if not candidate.resolve().is_relative_to(Path('/sys/fs/cgroup')):
                        raise ValueError('child cgroup escaped host controller')
                    cgroup = candidate
                    parent_groups = Path('/proc/self/cgroup').read_text().splitlines()
                    if '0::' + unified[0] in parent_groups:
                        raise ValueError('CPU parent shares disposable child cgroup')
                    before = cgroup_snapshot(cgroup)
                    if before['memory.max'] != str(128 * 1024 * 1024) or before['memory.swap.max'] != '0':
                        raise ValueError('Docker did not enforce reviewed memcg bounds')
                if cgroup is not None and cgroup.exists():
                    after = cgroup_snapshot(cgroup)
                time.sleep(0.05)
            record = future.result()
            elapsed = time.monotonic() - started
        runtime = inspect(container)['State']
        controller._sign(slot, record)
        process_cleanup = controller.ledger.cleanup(slot)
        if process_cleanup is None or process_cleanup['confirmed'] is not True:
            raise RuntimeError('actual supervisor process cleanup was not confirmed')
    except BaseException as error:
        original_error = error
    finally:
        if create_attempted:
            # Runtime cleanup is separate from process-group/CLI cleanup.
            try:
                selected = container or name
                command(['docker', 'rm', '--force', selected], timeout=5)
                listing = command(['docker', 'ps', '-aq', '--no-trunc', '--filter', 'name=^/' + name + '$'])
                cleanup['container_absent'] = not listing.strip()
                cleanup['cgroup_empty_or_removed'] = (cgroup is None or not cgroup.exists()
                    or counters(cgroup / 'cgroup.events').get('populated') == 0)
            except BaseException as error:
                cleanup['error_type'] = type(error).__name__
                if original_error is None:
                    original_error = error
    elapsed = time.monotonic() - started
    reasons = ([f'trusted measurement blocker: {type(original_error).__name__}'] if original_error else
        assess(case, record, before=before, after=after, runtime=runtime, elapsed=elapsed,
               cleanup=cleanup, tolerance=args.scheduling_tolerance_seconds))
    report = {'scope': 'local_physical_operator_measurement_not_accepted_native',
        'case': case, 'status': 'BLOCKED' if reasons else 'LOCAL_PHYSICAL_CASE_VERIFIED', 'reasons': reasons,
        'cpu_parent_pid': os.getpid(), 'cpu_parent_cgroup': Path('/proc/self/cgroup').read_text(),
        'cpu_parent_limit': 'not_enforced_by_this_harness',
        'identity_scope': 'synthetic_local_fixture_not_Driver_injected_or_accepted; context executable_image_digest names child image, not host CPU owner',
        'cpu_parent_alive_after': True, 'create_command': create, 'container_id': container,
        'image': image, 'source_content_digest': args.source_content_digest,
        'contract_digest': args.contract_digest, 'controller_context': controller.signer.manifest()['context'],
        'child_cgroup': str(cgroup) if cgroup else None, 'before': before, 'after': after,
        'runtime_state': runtime, 'elapsed_seconds': elapsed, 'cleanup': cleanup,
        'record': record, 'scientific_write': False}
    with (output / 'measurement.json').open('xb') as stream:
        stream.write(canonical(report))
        stream.flush()
        os.fsync(stream.fileno())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--image', required=True, help='Existing local digest-pinned reviewed Python image')
    parser.add_argument('--source-content-digest', required=True)
    parser.add_argument('--contract-digest', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scheduling-tolerance-seconds', type=float, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[^\s]+@sha256:[0-9a-f]{64}', args.image):
        raise ValueError('immutable reviewed local image digest required')
    if not 0 <= args.scheduling_tolerance_seconds <= 5:
        raise ValueError('explicit reviewed scheduling tolerance from0to5 seconds required')
    source = args.source_root.resolve(strict=True)
    sys.path.insert(0, str(source / 'judge'))
    output = args.output.absolute()
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    identity = dict(run_id='local_physical_' + uuid.uuid4().hex, attempt_id='local_attempt',
                    verification_id='local_verification', task_config_digest='sha256:' + '0' * 64,
                    executable_image_digest=args.image.split('@')[1])
    for case in ('oom', 'healthy', 'watchdog', 'healthy', 'dribble', 'healthy'):
        directory = output / (str(len(list(output.iterdir()))) + '-' + case)
        report = run_case(args, case, source, directory, identity)
        print(json.dumps({'case': case, 'status': report['status'], 'output': str(directory)}), flush=True)
        if report['status'] == 'BLOCKED':
            raise RuntimeError('physical prerequisite blocked; inspect private measurement')


if __name__ == '__main__':
    main()
