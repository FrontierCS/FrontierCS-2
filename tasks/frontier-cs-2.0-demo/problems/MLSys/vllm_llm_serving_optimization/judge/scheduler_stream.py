"""Bounded incremental framing for the existing scheduler SSH operation."""
import json
import struct

SCHEMA = 'frontier.vllm-scheduler-stream-draft/v1'


class FrameError(ValueError):
    pass


def frame(value, maximum):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if not 0 < len(raw) <= maximum:
        raise ValueError('scheduler frame exceeds publisher allocation')
    return struct.pack('!I', len(raw)) + raw


class Decoder:
    def __init__(self, *, maximum_frame_bytes, maximum_total_bytes, accept):
        if (any(type(n) is not int or n <= 0 for n in (maximum_frame_bytes, maximum_total_bytes))
                or maximum_frame_bytes > maximum_total_bytes or not callable(accept)):
            raise ValueError('explicit bounded scheduler frame observer required')
        self.maximum_frame_bytes, self.maximum_total_bytes = maximum_frame_bytes, maximum_total_bytes
        self.accept, self.buffer, self.total = accept, bytearray(), 0
        self.expected, self.closed = None, False

    def feed(self, chunk):
        if self.closed or not isinstance(chunk, bytes):
            raise FrameError('invalid scheduler stream lifecycle')
        self.total += len(chunk)
        if self.total > self.maximum_total_bytes:
            raise FrameError('scheduler stream exceeds total publisher allocation')
        # Consume only the next header/body; never allocate an untrusted declared
        # body until the injected per-frame cap has been checked.
        offset = 0
        while offset < len(chunk):
            target = 4 if self.expected is None else self.expected
            take = min(target - len(self.buffer), len(chunk) - offset)
            self.buffer.extend(chunk[offset:offset + take])
            offset += take
            if len(self.buffer) != target:
                continue
            if self.expected is None:
                self.expected, = struct.unpack('!I', self.buffer)
                self.buffer.clear()
                if not 0 < self.expected <= self.maximum_frame_bytes:
                    raise FrameError('scheduler frame length exceeds publisher allocation')
            else:
                raw = bytes(self.buffer)
                self.buffer.clear()
                self.expected = None
                self.accept(raw)

    def finish(self):
        self.closed = True
        if self.buffer or self.expected is not None:
            raise FrameError('scheduler stream ended inside a frame')


def decode(raw):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('duplicate scheduler JSON key')
            value[key] = item
        return value
    def finite(value):
        raise ValueError('nonfinite scheduler JSON value')
    return json.loads(raw, object_pairs_hook=unique, parse_constant=finite)
