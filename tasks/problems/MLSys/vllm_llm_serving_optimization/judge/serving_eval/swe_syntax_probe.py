"""Compile changed existing sources without importing them; supports testbed Python 3.5+."""
import hashlib
import json
import os
import subprocess
import sys
import warnings


def probe(base, root):
    # Legal syntax warnings belong in the original test log; they must not
    # corrupt this probe's machine-readable evidence channel.
    warnings.simplefilter('ignore', SyntaxWarning)
    def git(*args):
        return subprocess.check_output(['git', '-c', 'core.fileMode=false', '-c', 'safe.directory=' + root, *args], cwd=root)

    tracked = set(git('ls-tree', '-r', '--name-only', '-z', base).split(b'\0'))
    paths = set(git('diff', '--name-only', '-z', base).split(b'\0'))
    paths.update(git('ls-files', '--others', '--exclude-standard', '-z').split(b'\0'))
    evidence, checks, skipped = [], [], []
    for raw in sorted(paths):
        name = os.fsdecode(raw)
        path = os.path.join(root, name)
        if not name.endswith('.py'):
            continue
        if (os.path.islink(path) or not os.path.isfile(path)
                or os.path.commonpath([os.path.realpath(root), os.path.realpath(path)]) != os.path.realpath(root)):
            skipped.append(name)
            continue
        check = {'path': name, 'baseline_exists': raw in tracked,
                 'baseline_compiles': None, 'interpreter': sys.version.split()[0]}
        with open(path, 'rb') as handle:
            changed = handle.read()
        if raw in tracked:
            original = git('show', base + ':' + name)
            if original == changed:
                # Parent images may have packaging-only executable-mode commits.
                continue
            check['baseline_sha256'] = hashlib.sha256(original).hexdigest()
            try:
                compile(original, path, 'exec', dont_inherit=True)
                check['baseline_compiles'] = True
            except (SyntaxError, ValueError):
                check['baseline_compiles'] = False
        check['patched_sha256'] = hashlib.sha256(changed).hexdigest()
        try:
            compile(changed, path, 'exec', dont_inherit=True)
            check['patched_compiles'] = True
        except (SyntaxError, ValueError) as error:
            check.update(patched_compiles=False, error_type=type(error).__name__,
                         lineno=getattr(error, 'lineno', None))
            # Existing invalid parser fixtures and added files do not provide a
            # baseline comparison; retain their audit without assigning blame.
            if check['baseline_compiles'] is True and isinstance(error, SyntaxError):
                evidence.append(check)
        checks.append(check)
    return {'checks': checks, 'errors': evidence, 'skipped_paths': skipped}


if __name__ == '__main__':
    print(json.dumps(probe(sys.argv[1], sys.argv[2])))
