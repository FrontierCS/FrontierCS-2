"""Durable local cutoffs for operator calibration; no networking or credentials.

A grant is bounded by Driver's owner clocks before provisioning. Anchoring it to
our earliest create attempt and subtracting at least one second covers Driver's
integer rounding under the conventional comparable elapsed-seconds assumption.
This does not expose an exact owner deadline or guarantee real-time completion.
A separately supported elapsed-clock error allowance belongs in margin_seconds.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
import time
from urllib.parse import urlsplit

SCHEMA = 'frontier.vllm-calibration-clock/v1'
_MAX_MARKER_BYTES = 65536
_COMPONENT = re.compile(r'[A-Za-z0-9_-]{1,256}\Z')
_DIGEST = re.compile(r'sha256:[0-9a-f]{64}\Z')


class CalibrationClockError(ValueError):
    """Unsafe marker, binding or clock; no new create should be attempted."""


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _component(value):
    return isinstance(value, str) and _COMPONENT.fullmatch(value) is not None


def _binding(value):
    fields = {'driver_origin', 'verification_id', 'request_key', 'request',
              'config_sha256', 'plan_sha256'}
    if not isinstance(value, dict) or set(value) != fields:
        raise CalibrationClockError('closed credential-free clock binding required')
    try:
        origin = value['driver_origin']
        parsed = urlsplit(origin)
        port = parsed.port
        valid_origin = (isinstance(origin, str) and parsed.scheme in {'http', 'https'}
                        and parsed.hostname and not parsed.username and not parsed.password
                        and not parsed.query and not parsed.fragment and parsed.path in ('', '/')
                        and not any(c.isspace() for c in origin) and len(origin) <= 2048
                        and (port is None or 0 < port <= 65535))
    except (ValueError, TypeError, AttributeError):
        valid_origin = False
    if (not valid_origin or not _component(value['verification_id'])
            or not _component(value['request_key'])
            or any(not isinstance(value[k], str) or not _DIGEST.fullmatch(value[k])
                   for k in ('config_sha256', 'plan_sha256'))):
        raise CalibrationClockError('invalid clock identity or Driver origin')
    request = value['request']
    if (not isinstance(request, dict)
            or set(request) != {'gpu_type', 'gpu_count', 'max_duration_seconds'}
            or not _component(request['gpu_type'])
            or any(type(request[k]) is not int or request[k] <= 0
                   for k in ('gpu_count', 'max_duration_seconds'))):
        raise CalibrationClockError('fixed credential-free GPU create body required')
    return copy.deepcopy(value)


def _budgets(cap_seconds, cleanup_seconds, capture_seconds, margin_seconds):
    values = {'cap_seconds': cap_seconds, 'cleanup_seconds': cleanup_seconds,
              'capture_seconds': capture_seconds, 'margin_seconds': margin_seconds}
    if (any(not _finite(v) or v <= 0 for v in values.values()) or margin_seconds < 1
            or cap_seconds <= cleanup_seconds + capture_seconds + margin_seconds):
        raise CalibrationClockError('cap must fund positive cleanup/capture reserves and margin >= 1 second')
    return {k: float(v) for k, v in values.items()}


def _clock_domain():
    """Boot plus current time namespace and offset; never a remote clock claim."""
    try:
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        namespace = os.readlink('/proc/self/ns/time')
        offsets = [line.split() for line in Path('/proc/self/timens_offsets').read_text().splitlines()]
        monotonic = [row for row in offsets if row and row[0] == 'monotonic']
        if (not re.fullmatch(r'[0-9a-f-]{36}', boot)
                or not re.fullmatch(r'time:\[[0-9]+\]', namespace)
                or len(monotonic) != 1 or len(monotonic[0]) != 3):
            raise ValueError()
        offset = int(monotonic[0][1]) * 1_000_000_000 + int(monotonic[0][2])
        implementation = time.get_clock_info('monotonic').implementation
    except (OSError, ValueError) as error:
        raise CalibrationClockError('local monotonic clock domain unavailable') from error
    return {'boot_id': boot, 'time_namespace': namespace,
            'monotonic_offset_ns': offset, 'implementation': implementation}


@contextmanager
def _lock(path):
    # Separate stable inode: replacing the marker must not replace its lock.
    descriptor = os.open(str(path) + '.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise CalibrationClockError('clock lock must be a regular file')
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise CalibrationClockError('clock marker is in use') from None
        yield
    finally:
        os.close(descriptor)


def _sync_directory(path):
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, 'rb') as stream:
        metadata = os.fstat(stream.fileno())
        if (not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o077
                or not 0 < metadata.st_size <= _MAX_MARKER_BYTES):
            raise CalibrationClockError('clock marker must be a bounded private regular file')
        raw = stream.read(_MAX_MARKER_BYTES + 1)
    try:
        return json.loads(raw)
    except (ValueError, UnicodeError):
        raise CalibrationClockError('incomplete or invalid clock marker') from None


def _write(path, record, *, new):
    raw = (json.dumps(record, sort_keys=True, allow_nan=False) + '\n').encode()
    if len(raw) > _MAX_MARKER_BYTES:
        raise CalibrationClockError('clock marker exceeds its local bound')
    pending = None
    if new:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    else:
        descriptor, name = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.pending', dir=path.parent)
        pending = Path(name)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if pending is not None:
            os.replace(pending, path)
        _sync_directory(path)
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)


def _validate_record(record, binding, budgets, domain):
    fields = {'schema', 'binding', 'budgets', 'clock_domain', 't0', 'local_cap_deadline',
              'runner_id', 'granted_seconds'}
    if (not isinstance(record, dict) or set(record) != fields or record['schema'] != SCHEMA
            or record['binding'] != binding or record['budgets'] != budgets
            or record['clock_domain'] != domain):
        raise CalibrationClockError('clock marker binding, budgets or domain changed')
    # Python considers True == 1; equality alone must not accept type drift.
    _binding(record['binding'])
    _budgets(**record['budgets'])
    t0, end = record['t0'], record['local_cap_deadline']
    if (not _finite(t0) or not _finite(end) or end != t0 + budgets['cap_seconds']
            or t0 > time.monotonic()):
        raise CalibrationClockError('clock marker timeline is invalid')
    runner, grant = record['runner_id'], record['granted_seconds']
    if (runner is None) != (grant is None):
        raise CalibrationClockError('incomplete runner grant binding')
    if runner is not None:
        _validate_grant(runner, grant, binding)


def _validate_grant(runner_id, granted_seconds, binding):
    if (not _component(runner_id) or type(granted_seconds) is not int or granted_seconds <= 0
            or granted_seconds > binding['request']['max_duration_seconds']):
        raise CalibrationClockError('invalid Driver runner identity or resource grant')


class CalibrationClock:
    def __init__(self, path, record):
        self.path = path
        self._record = copy.deepcopy(record)

    @property
    def t0(self):
        return self._record['t0']

    @property
    def local_cap_deadline(self):
        return self._record['local_cap_deadline']

    @property
    def runner_id(self):
        return self._record['runner_id']

    @property
    def granted_seconds(self):
        return self._record['granted_seconds']

    def _deadlines(self, end):
        budgets = self._record['budgets']
        delete = end - budgets['capture_seconds']
        return {'work_deadline': delete - budgets['cleanup_seconds'],
                'delete_deadline': delete, 'capture_deadline': end}

    @property
    def initial_deadlines(self):
        """Create bound; a resumed accepted grant can only shorten it."""
        return self.deadlines

    @property
    def deadlines(self):
        end = self.local_cap_deadline
        if self.granted_seconds is not None:
            end = min(end, self.t0 + self.granted_seconds - self._record['budgets']['margin_seconds'])
        return self._deadlines(end)

    def preview_grant(self, runner_id, granted_seconds):
        """Pure cutoff calculation for cleanup before fallible persistence.

        This is not grant acceptance or replay authority. The caller must still
        persist through accept_grant before starting work, and retain these
        stricter cutoffs if that persistence fails.
        """
        _validate_grant(runner_id, granted_seconds, self._record['binding'])
        end = min(self.deadlines['capture_deadline'],
                  self.t0 + granted_seconds - self._record['budgets']['margin_seconds'])
        return self._deadlines(end)

    def accept_grant(self, runner_id, granted_seconds):
        """Persist before returning cutoffs, including already-expired cutoffs.

        A delayed/short grant does not reset time. Clamp the caller's deadlines
        first, then reject further work if expired so cleanup retains its bound.
        """
        _validate_grant(runner_id, granted_seconds, self._record['binding'])
        with _lock(self.path):
            record = _read(self.path)
            _validate_record(record, self._record['binding'], self._record['budgets'], _clock_domain())
            if (record['t0'], record['local_cap_deadline'], record['clock_domain']) != (
                    self.t0, self.local_cap_deadline, self._record['clock_domain']):
                raise CalibrationClockError('earliest clock marker changed')
            if record['runner_id'] is not None and (record['runner_id'], record['granted_seconds']) != (runner_id, granted_seconds):
                raise CalibrationClockError('logical create replay changed its runner or grant')
            record.update(runner_id=runner_id, granted_seconds=granted_seconds)
            _write(self.path, record, new=False)
            self._record = record
        return self.deadlines


def begin(binding, path, *, cap_seconds, cleanup_seconds, capture_seconds,
          margin_seconds=1.0, resume=False):
    """Fsync the earliest-create marker before the caller may send a request.

    The parent directory must already exist. Existing markers require explicit
    resume; missing/corrupt markers cannot resume. A marker is immutable in its
    binding, clock domain and original cap. No tokens or arbitrary metadata fit
    the closed binding schema. Clock-rate comparability remains an assumption.
    """
    t0 = time.monotonic()
    binding = _binding(binding)
    budgets = _budgets(cap_seconds, cleanup_seconds, capture_seconds, margin_seconds)
    if type(resume) is not bool:
        raise CalibrationClockError('resume must be explicit Boolean')
    domain = _clock_domain()
    path = Path(path)
    with _lock(path):
        if resume:
            try:
                record = _read(path)
            except FileNotFoundError:
                raise CalibrationClockError('explicit resume requires the original clock marker') from None
            _validate_record(record, binding, budgets, domain)
            # Also establish durability after recovery from an interrupted fsync.
            _write(path, record, new=False)
        else:
            record = {'schema': SCHEMA, 'binding': binding, 'budgets': budgets,
                      'clock_domain': domain, 't0': t0, 'local_cap_deadline': t0 + budgets['cap_seconds'],
                      'runner_id': None, 'granted_seconds': None}
            _validate_record(record, binding, budgets, domain)
            try:
                _write(path, record, new=True)
            except FileExistsError:
                raise CalibrationClockError('existing logical clock marker requires explicit resume') from None
    return CalibrationClock(path, record)
