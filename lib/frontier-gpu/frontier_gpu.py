#!/usr/bin/env python3
'''Use this task's GPU runners, leased through a session: /opt/frontier/candidate/skills/frontier-gpu/SKILL.md says how.'''

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time

# The session on the runners, and their GPU's type, which the agent's command sets.
GRANT = os.environ['FRONTIER_GPU_GRANT']
GPU_TYPE = os.environ['FRONTIER_GPU_TYPE']
HOME = Path('/workspace/.frontier-gpu')
RUNNERS = HOME / 'runners.json'
WORKSPACE = Path('/workspace')
SANDBOXCTL = '/.pumpkin/bin/sandboxctl'
RUNNER_ID = re.compile(r'^gpurunner_[0-9A-HJKMNP-TV-Z]{26}$')
ALPHABET = '0123456789ABCDEFGHJKMNPQRSTVWXYZ'
# Where a runner keeps each command's whole output: `<n>/command.stdout` and `command.stderr`.
LOG = PurePosixPath('/pumpkin/log')
# The most of each output stream that a status gives, as JSON writes it.
TAIL_BYTES = 21 << 10
# States of a lease that has ended; `closing` is one whose runner is still being removed.
ENDED = ('closing', 'closed', 'expired', 'failed')
# When a wait first says it still waits, and how often after that.
FIRST_NOTE_SECONDS = 5
NOTE_SECONDS = 30
# How long `create` waits for the slot of a runner that has ended but is still being removed.
SLOT_SECONDS = 180


class GpuError(RuntimeError):
    pass


class Refused(GpuError):
    pass


def ctl(*args):
    done = subprocess.run([SANDBOXCTL, *map(str, args)], capture_output=True, text=True)
    try:
        reply = json.loads(done.stdout)
    except json.JSONDecodeError:
        raise GpuError(done.stderr.strip() or f'sandboxctl exited with {done.returncode}') from None
    if reply.get('kind') == 'refused':
        raise Refused(reply['message'])
    return reply


def call(op, *args):
    # The session's op `op`: `sandboxctl call <session>.<op>`.
    return ctl('call', f'{GRANT}.{op}', *args)


def runner_id(lease):
    digits = ''
    for _ in range(26):
        lease, digit = divmod(lease, 32)
        digits = ALPHABET[digit] + digits
    return 'gpurunner_' + digits


def lease_of(runner):
    if not RUNNER_ID.fullmatch(runner):
        raise GpuError(f'`{runner}` is no GPU runner ID: give one that `create` or `list` printed')
    lease = 0
    for digit in runner[len('gpurunner_'):]:
        lease = lease * 32 + ALPHABET.index(digit)
    return lease


def grant():
    for held in ctl('ops')['grants']:
        if held.get('kind') == 'session' and held['name'] == GRANT:
            return held
    raise GpuError(f'this run holds no `{GRANT}` session: `sandboxctl ops` lists what it holds')


def known():
    # The runners created here, by lease: rebuilt from the host, without their options, if the
    # file is gone or unreadable.
    try:
        entries = json.loads(RUNNERS.read_text())['runners']
        return {int(entry['lease']): entry for entry in entries}
    except (OSError, ValueError, KeyError, TypeError):
        pass
    entries, lease = {}, 0
    while True:
        try:
            call('status', lease)
        except Refused:
            break
        entries[lease] = {'lease': lease}
        lease += 1
    save(entries)
    return entries


def save(entries):
    HOME.mkdir(parents=True, exist_ok=True)
    temporary = RUNNERS.with_name(RUNNERS.name + '.new')
    temporary.write_text(json.dumps({'runners': sorted(entries.values(), key=lambda e: e['lease'])},
                                    indent=1, sort_keys=True))
    os.replace(temporary, RUNNERS)


def status(lease):
    try:
        return call('status', lease)
    except Refused as refused:
        raise GpuError(f'{refused}; `frontier-gpu list` shows your runners') from None


def last(reply):
    # What the runner ran last, a command or a transfer, with its number.
    if not reply.get('last'):
        return None
    done = dict(reply['last'])
    done['number'] = done.pop('command', done.pop('transfer', None))
    return done


def view(reply):
    shown = {'runner_id': runner_id(reply['lease']), 'state': reply['state'],
             'gpu_type': GPU_TYPE, 'gpu_count': 1, 'elapsed_seconds': reply['elapsed_seconds'],
             'remaining': reply['remaining']}
    if reply.get('failure'):
        shown['failure'] = reply['failure']
    done = last(reply)
    if done:
        shown['last'] = {key: done[key] for key in ('kind', 'number', 'end') if key in done}
        if done['kind'] == 'command':
            shown['last']['log'] = str(LOG / str(done['number']))
    return shown


def next_step(reply):
    runner, state = runner_id(reply['lease']), reply['state']
    if state == 'starting':
        return (f"{runner} has been starting for {reply['elapsed_seconds']} s: the host is "
                'getting it a GPU and starting it, which can take minutes, and costs no GPU '
                'seconds; exec, push and pull wait for it')
    if state == 'ready':
        return f'{runner} is ready: push, exec and pull use it, and it keeps its files between them'
    if state == 'running':
        return (f'{runner} is running its last command or transfer: the next waits for it, or '
                f'delete {runner} ends it')
    if state == 'failed':
        return (f"{runner} failed: {reply.get('failure') or 'no reason given'}. That is the "
                "provider's answer, not your fault, and its start cost no GPU seconds: create "
                'another')
    if state == 'expired':
        return f'{runner} expired, as its time, or the GPU seconds in all, ran out: create another'
    return f'{runner} is {state}: create another to go on'


def note(text):
    print('frontier-gpu: ' + text, file=sys.stderr, flush=True)


def pause(waited):
    time.sleep(1 if waited < 10 else 3)


def wait_idle(lease):
    # Waits until the runner can take a command or a transfer, and returns its status.
    began, noted = time.monotonic(), None
    while True:
        reply = status(lease)
        state = reply['state']
        if state == 'ready':
            return reply
        if state in ENDED:
            raise GpuError(next_step(reply))
        waited = time.monotonic() - began
        if waited >= FIRST_NOTE_SECONDS and (noted is None or waited - noted >= NOTE_SECONDS):
            note(next_step(reply) + ('' if state == 'starting' else '; waiting for it to end'))
            noted = waited
        pause(waited)


def follow(lease, number):
    # Waits until the runner's command or transfer `number` has ended, or the runner has, and
    # returns the status and what it ran last.
    began = time.monotonic()
    while True:
        reply = status(lease)
        done = last(reply)
        if done is None or done['number'] != number:
            raise GpuError(f'{runner_id(lease)} shows no command or transfer {number}: run it again')
        if reply['state'] != 'running':
            return reply, done
        pause(time.monotonic() - began)


def left(reply):
    remaining = reply['remaining']
    parts = []
    if remaining.get('commands') is not None:
        parts.append(f"{remaining['commands']} commands and transfers")
    if remaining.get('cumulative_seconds') is not None:
        parts.append(f"{remaining['cumulative_seconds']} GPU seconds")
    return (f"the `{GRANT}` grant has {' and '.join(parts) or 'no limit'} left, and "
            f"{runner_id(reply['lease'])} {remaining['instance_seconds']} s")


def started(lease, verb, *args):
    # Sends op `verb` to the runner once it is idle, and returns the number it runs as.
    before = last(wait_idle(lease))
    number = 0 if before is None else before['number'] + 1
    try:
        call(verb, lease, *args)
    except Refused as refused:
        raise GpuError(str(refused)) from None
    return number


def workspace_path(raw, existing):
    path = Path(raw)
    if not path.is_absolute():
        raise GpuError('path must be absolute')
    resolved = path.resolve(strict=existing)
    try:
        resolved.relative_to(WORKSPACE)
    except ValueError:
        raise GpuError('path must stay below /workspace') from None
    return resolved


def remote_path(raw, directory=False):
    path = PurePosixPath(raw)
    if not path.is_absolute() or '..' in path.parts:
        raise GpuError("remote path must be absolute and must not contain '..'")
    if not directory and path == PurePosixPath('/'):
        raise GpuError('remote source path must not be the filesystem root')
    return path


def tree_bytes(source):
    # The bytes of the files below `source`, a directory; refuses links and special files, as
    # FrontierCS does.
    if not source.is_dir():
        raise GpuError(f'{source} is not a directory, and push copies a directory: put the file '
                       'in one and push that')
    total = 0
    for path in [source, *(Path(top) / name for top, dirs, names in os.walk(source)
                            for name in [*dirs, *names])]:
        metadata = path.lstat()
        if stat.S_ISREG(metadata.st_mode):
            total += metadata.st_size
        elif not stat.S_ISDIR(metadata.st_mode):
            raise GpuError(f'push rejects links and special files: {path}')
    return total


def create(args):
    if args.gpu_type != GPU_TYPE:
        raise GpuError(f"this task's runners are of type {GPU_TYPE}: give --type {GPU_TYPE}")
    if args.count != 1:
        raise GpuError('a runner has one GPU: give --count 1')
    held = grant()
    most = held['instance_seconds']
    if not 0 < args.seconds <= most:
        raise GpuError(f'give --seconds from 1 to {most}, the most that a runner lasts')
    entries = known()
    if args.idempotency_key:
        same = [e for e in entries.values() if e.get('idempotency_key') == args.idempotency_key]
        if same:
            reply = status(same[0]['lease'])
            print(json.dumps(view(reply), sort_keys=True))
            note(next_step(reply))
            return 0
    began = time.monotonic()
    while True:
        try:
            lease = call('open', '--seconds', args.seconds)['lease']
            break
        except Refused as refused:
            live = [lease for lease in entries if status(lease)['state'] not in ENDED]
            if 'leases open' not in str(refused) or time.monotonic() - began > SLOT_SECONDS:
                raise GpuError(str(refused)) from None
            if live:
                raise GpuError(f'{refused}: {runner_id(live[0])} has not ended; use it, or end it '
                               f'with `frontier-gpu delete {runner_id(live[0])}` first') from None
            note('the last runner has ended and is still being removed; waiting for its slot')
            time.sleep(3)
    entries[lease] = {'lease': lease, 'seconds': args.seconds,
                      'idempotency_key': args.idempotency_key,
                      'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    save(entries)
    runner = runner_id(lease)
    note(f'created {runner}; it lasts --seconds {args.seconds} from when it is ready; waiting for it '
         'to start (Ctrl-C leaves it starting)')
    try:
        reply = wait_idle(lease)
    except KeyboardInterrupt:
        note(f'{runner} goes on starting: `frontier-gpu status {runner}` shows it')
        return 130
    print(json.dumps(view(reply), sort_keys=True))
    note(next_step(reply))
    return 0


def show(args):
    reply = status(lease_of(args.runner_id))
    print(json.dumps(view(reply), sort_keys=True))
    note(next_step(reply))
    return 0


def listed(args):
    print(json.dumps({'runners': [view(status(lease)) for lease in sorted(known())]}, sort_keys=True))
    return 0


def execute(args):
    lease = lease_of(args.runner_id)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        raise GpuError('exec requires a command after --')
    try:
        number = started(lease, 'run', '--', *command)
        began = time.monotonic()
        reply, done = follow(lease, number)
    except KeyboardInterrupt:
        note(f'interrupted; the command goes on to its end on {args.runner_id}, and the next one '
             f'waits for it: `frontier-gpu status {args.runner_id}` shows it, and `delete` ends it')
        return 130
    took = round(time.monotonic() - began)
    end = done.get('end')
    sys.stdout.write(done.get('stdout', ''))
    sys.stdout.flush()
    sys.stderr.write(done.get('stderr', ''))
    if done.get('stderr') and not done['stderr'].endswith('\n'):
        print('', file=sys.stderr)
    notes, log = [], LOG / str(number)
    for name in ('stdout', 'stderr'):
        if len(done.get(name, '').encode()) >= TAIL_BYTES - 3:
            notes.append(f'{name} may be longer than its last 21 KiB shown: the whole is '
                         f'{log}/command.{name} on the runner, which pull copies back')
    if end is None:
        code = 255
        what = f"did not finish, as the runner ended under it"
        notes.append(next_step(reply))
    elif end == 'unstarted':
        code = 127
        what = 'did not start, as its stderr says'
    elif 'code' in end['exited']:
        code = end['exited']['code']
        what = f'exited with {code}'
    else:
        code = 128 + end['exited']['signal']
        what = f"was killed by signal {end['exited']['signal']}"
    note(f'command {number} on {args.runner_id} {what}, after about {took} s; '
         + '; '.join(notes + [left(reply)]))
    return code


def transfer(lease, verb, source, to, *flags):
    number = started(lease, verb, source, to, *flags)
    reply, done = follow(lease, number)
    end = done.get('end')
    if end is None:
        raise GpuError(f"the runner ended under the transfer ({reply['state']}): what arrived of "
                       f'it stays at {to}; ' + next_step(reply))
    if 'failed' in end:
        note(left(reply))
        raise GpuError(end['failed'])
    moved = end['moved']
    note(f"copied {source} to {to}: {moved['entries']} entries, {moved['bytes']} bytes; {left(reply)}")
    return 0


def push(args):
    lease = lease_of(args.runner_id)
    source = workspace_path(args.source, existing=True)
    to = remote_path(args.remote_directory, directory=True) / source.name
    size, most = tree_bytes(source), grant().get('transfer_bytes')
    if most is not None and size > most:
        raise GpuError(f'{source} holds {size} bytes of files, over the {most} that one transfer '
                       'may move: push less')
    return transfer(lease, 'put', source, to, '--replace')


def pull(args):
    lease = lease_of(args.runner_id)
    remote = remote_path(args.remote_path)
    destination = workspace_path(args.destination, existing=False)
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    # The tree arrives in a fresh directory, then is copied over DESTINATION, so that what it
    # holds and the tree lacks stays.
    HOME.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='pull-', dir=HOME))
    try:
        transfer(lease, 'get', remote, staging / remote.name)
        shutil.copytree(staging / remote.name, destination / remote.name, dirs_exist_ok=True)
    except GpuError as error:
        if 'Not a directory' in str(error):
            raise GpuError(f'{error}. pull copies a directory: pull the one that holds the file') from None
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return 0


def delete(args):
    lease = lease_of(args.runner_id)
    try:
        call('close', lease)
    except Refused as refused:
        raise GpuError(str(refused)) from None
    reply = status(lease)
    print(json.dumps(view(reply), sort_keys=True))
    note(f'{args.runner_id} is {reply["state"]}: it runs nothing more and costs no more GPU seconds')
    return 0


def quota(args):
    held = grant()
    remaining = held['remaining']
    def counted(limit, rest):
        return {'limit': limit, 'used': None if limit is None else limit - rest, 'remaining': rest}
    print(json.dumps({
        'gpu_type': GPU_TYPE,
        'runners_at_a_time': held['instances'],
        'runners_open_now': held['instances'] - remaining['instances'],
        'max_runner_seconds': held['instance_seconds'],
        'gpu_seconds': counted(held.get('cumulative_seconds'), remaining.get('cumulative_seconds')),
        'commands_and_transfers': counted(held.get('commands'), remaining.get('commands')),
        'max_transfer_bytes': held.get('transfer_bytes'),
    }, sort_keys=True))
    return 0


def ssh(args):
    raise GpuError('there is no interactive ssh here: run each command with exec, on a runner that '
                   'keeps its files between commands')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    made = sub.add_parser('create')
    made.add_argument('--type', required=True, dest='gpu_type')
    made.add_argument('--count', type=int, default=1)
    made.add_argument('--seconds', type=int, required=True)
    made.add_argument('--idempotency-key', default=None)
    made.set_defaults(run=create)
    sub.add_parser('list').set_defaults(run=listed)
    sub.add_parser('quota').set_defaults(run=quota)
    for name, run in (('status', show), ('delete', delete), ('ssh', ssh)):
        command = sub.add_parser(name)
        command.add_argument('runner_id')
        command.set_defaults(run=run)
    command = sub.add_parser('exec')
    command.add_argument('runner_id')
    command.add_argument('command', nargs=argparse.REMAINDER)
    command.set_defaults(run=execute)
    command = sub.add_parser('push')
    command.add_argument('runner_id')
    command.add_argument('source')
    command.add_argument('remote_directory')
    command.set_defaults(run=push)
    command = sub.add_parser('pull')
    command.add_argument('runner_id')
    command.add_argument('remote_path')
    command.add_argument('destination')
    command.set_defaults(run=pull)
    args = parser.parse_args(argv)
    try:
        return args.run(args)
    except (GpuError, OSError, ValueError) as error:
        print(f'frontier-gpu: ERROR: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
