"""One clean exec owns episode parsing/state; no tool signer enters this worker."""
from dataclasses import asdict
import json
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batch_coordinator import ModelClient, _compact_record_trace
from serving_eval import agent_runner, bfcl
from serving_eval.settings import EpisodeSettings

RPC_SCHEMA = 'frontier.vllm-episode-rpc-draft/v1'


def read_frame(maximum):
    header = sys.stdin.buffer.read(4)
    if len(header) != 4:
        raise ValueError('episode broker frame is absent')
    length, = struct.unpack('!I', header)
    if not 0 < length <= maximum:
        raise ValueError('episode broker frame exceeds allocation')
    raw = sys.stdin.buffer.read(length)
    if len(raw) != length:
        raise ValueError('episode broker frame is incomplete')
    return json.loads(raw)


def send(value, maximum):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if not 0 < len(raw) <= maximum:
        raise ValueError('episode worker frame exceeds allocation')
    sys.stdout.buffer.write(struct.pack('!I', len(raw)) + raw)
    sys.stdout.buffer.flush()


class TestbedProxy:
    def __init__(self, maximum):
        self.maximum, self.sequence = maximum, 0

    def call(self, method, arguments):
        send({'schema': RPC_SCHEMA, 'sequence': self.sequence,
              'method': method, 'arguments': arguments}, self.maximum)
        value = read_frame(self.maximum)
        if (not isinstance(value, dict) or set(value) != {'schema', 'sequence', 'result'}
                or value['schema'] != RPC_SCHEMA or value['sequence'] != self.sequence):
            raise ValueError('episode broker reply does not bind request')
        self.sequence += 1
        return value['result']

    def run(self, command, *, timeout):
        result = self.call('run', {'command': command, 'timeout': timeout})
        if (not isinstance(result, list) or len(result) != 2 or type(result[0]) is not int
                or not isinstance(result[1], str)):
            raise ValueError('invalid broker command result')
        return result[0], result[1]

    def read_patch(self):
        patch = self.call('read_patch', {})
        if not isinstance(patch, str):
            raise ValueError('invalid broker patch result')
        return patch


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--maximum-frame-bytes', required=True, type=int)
    args = parser.parse_args()
    document = read_frame(args.maximum_frame_bytes)
    if (not isinstance(document, dict) or set(document) !=
            {'workload', 'instance', 'endpoint', 'request_timeout_seconds', 'settings'}
            or document['workload'] not in {'swe', 'bfcl'}):
        raise ValueError('invalid fixed episode worker request')
    settings = EpisodeSettings(**document['settings'])
    trace = []
    client = ModelClient(document['endpoint'], document['request_timeout_seconds'], trace)
    try:
        if document['workload'] == 'swe':
            outcome = agent_runner.run_instance(document['instance'], client=client,
                prepared_testbed=TestbedProxy(args.maximum_frame_bytes), settings=settings)
        else:
            outcome = bfcl.run_memory_instance(document['instance'], client=client,
                settings=settings, tool_trace=trace)
    finally:
        client.close()
    result = {'episode': asdict(outcome), 'trace': trace,
              'requires_error_attribution': any('error_origin' in event for event in trace)}
    _compact_record_trace(result)
    send({'schema': 'frontier.vllm-child-result-draft/v1', 'result': result}, args.maximum_frame_bytes)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        print('episode worker failed: ' + type(error).__name__, file=sys.stderr)
        raise SystemExit(1)
