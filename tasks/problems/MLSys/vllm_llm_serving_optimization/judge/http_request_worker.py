"""Fresh exec for one request-bound capability; never loads a signing key."""
import argparse
import json
from pathlib import Path
import struct
import sys
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request-file', required=True)
    parser.add_argument('--maximum-frame-bytes', required=True, type=int)
    args = parser.parse_args()
    document = json.loads(Path(args.request_file).read_bytes())
    if (set(document) != {'url', 'method', 'headers', 'body', 'timeout_seconds', 'maximum_body_bytes'}
            or document['method'] not in {'GET', 'POST'}
            or type(document['maximum_body_bytes']) is not int
            or document['maximum_body_bytes'] <= 0):
        raise ValueError('invalid fixed signed HTTP transport request')
    body = None if document['body'] is None else document['body'].encode()
    request = urllib.request.Request(document['url'], data=body,
        headers=document['headers'], method=document['method'])
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
            request, timeout=document['timeout_seconds']) as response:
        raw = response.read(document['maximum_body_bytes'] + 1)
        if len(raw) > document['maximum_body_bytes']:
            raise ValueError('HTTP response exceeds publisher allocation')
    result = json.loads(raw)
    if not isinstance(result, dict):
        raise ValueError('HTTP response is not an object')
    framed = json.dumps({'schema': 'frontier.vllm-child-result-draft/v1', 'result': result},
        sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if len(framed) > args.maximum_frame_bytes:
        raise ValueError('HTTP result exceeds publisher frame allocation')
    sys.stdout.buffer.write(struct.pack('!I', len(framed)) + framed)
    sys.stdout.buffer.flush()


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        print('bounded HTTP worker failed: ' + type(error).__name__, file=sys.stderr)
        raise SystemExit(1)
