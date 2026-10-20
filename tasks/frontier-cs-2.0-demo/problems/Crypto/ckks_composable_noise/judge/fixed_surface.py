"""The part of the source tree a submission may not change, byte for byte.

These files produce the clear target, the test vectors, the encrypted run and
the reported statistics. Everything else below estimator/ is editable. The two
protected declarations of dft.go are checked by judge/source_policy.go.
"""
from pathlib import Path

FIXED_FILES = ('estimator/stats.go', 'estimator/utils.go', 'estimator/go.mod', 'estimator/go.sum')
FIXED_DIRECTORIES = ('estimator/experiments',)
EDITABLE_ROOT = 'estimator'


def _files(root, relative):
    base = Path(root)/relative
    return {str(p.relative_to(root)) for p in base.rglob('*') if p.is_file()} if base.is_dir() else set()


def fixed_surface_violations(base, tree):
    """Paths whose bytes differ from the pristine tree, are missing, or are extra."""
    base = Path(base); tree = Path(tree); bad = []
    expected = set(FIXED_FILES)
    for directory in FIXED_DIRECTORIES:
        expected |= _files(base, directory)
        bad += sorted(_files(tree, directory) - _files(base, directory))
    # Nothing outside estimator/ belongs to the submission either.
    for path in sorted(p for p in _files(base, '.') if not p.startswith(EDITABLE_ROOT+'/') and '.git/' not in p and not p.startswith('.git')):
        expected.add(path)
    for path in sorted(expected):
        a = base/path; b = tree/path
        if b.is_symlink() or not b.is_file() or a.read_bytes() != b.read_bytes():
            bad.append(path)
    for path in sorted(_files(tree, EDITABLE_ROOT)):
        p = tree/path
        if p.is_symlink() or not path.endswith('.go') and path not in expected or path.endswith('_test.go'):
            bad.append(path)
    return sorted(set(bad))
