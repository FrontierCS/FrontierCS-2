"""Generate exact image-local CPU source allocation artifacts at image build."""
import argparse
from pathlib import Path
import sys

from controller_identity import content_contract, LIMITS_SCHEMA
from terminal_contract import canonical


def emit(root, source_trees, sources, contract_path, limits_path):
    root = Path(root)
    paths = list(sources)
    for tree in source_trees:
        directory = root / tree
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError('immutable source tree must be a real directory')
        paths.extend(str(path.relative_to(root)) for path in directory.rglob('*')
                     if path.suffix in {'.py', '.json'} and '__pycache__' not in path.parts)
    if len(set(paths)) != len(paths):
        raise ValueError('overlapping immutable source projections')
    # The first pass measures build inputs; the second hashes under precisely
    # that measured allocation and detects changes while generating artifacts.
    source_bytes = sum((root / path).stat().st_size for path in paths)
    contract = content_contract(root, paths, maximum_contract_bytes=sys.maxsize,
                                maximum_source_bytes=source_bytes)
    encoded = canonical(contract)
    checked = content_contract(root, paths, maximum_contract_bytes=len(encoded),
                               maximum_source_bytes=source_bytes)
    if checked != contract:
        raise ValueError('source changed while generating immutable contract')
    limits = {'schema': LIMITS_SCHEMA, 'maximum_contract_bytes': len(encoded),
              'maximum_source_bytes': source_bytes}
    destinations = [Path(contract_path), Path(limits_path)]
    if any(path.exists() for path in destinations):
        raise ValueError('contract allocation artifacts must be generated once')
    for path, data in zip(destinations, (encoded, canonical(limits))):
        with path.open('xb') as stream:
            stream.write(data)
        path.chmod(0o444)
    return limits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--source-tree', action='append', default=[])
    parser.add_argument('--source', action='append', default=[])
    parser.add_argument('--contract', required=True)
    parser.add_argument('--limits', required=True)
    args = parser.parse_args()
    emit(args.root, args.source_tree, args.source, args.contract, args.limits)


if __name__ == '__main__':
    main()
