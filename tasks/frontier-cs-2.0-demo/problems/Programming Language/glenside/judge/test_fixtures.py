import difflib
import shutil
import subprocess
from pathlib import Path

import evaluator


def copy_writable_tree(source: Path, destination: Path) -> Path:
    shutil.copytree(source, destination)
    # Image inputs are sealed; fixtures mutate only their owned copies.
    for path in [destination, *destination.rglob('*')]:
        path.chmod(path.stat().st_mode | (0o700 if path.is_dir() else 0o600))
    return destination


def function(text, name):
    start = text.index('fn ' + name + '(')
    if text[max(0, start - 4):start] == 'pub ':
        start -= 4
    cursor = text.index('{', start) + 1
    depth = 1
    while depth:
        depth += (text[cursor] == '{') - (text[cursor] == '}')
        cursor += 1
    return text[start:cursor]


def controlled_patch(directory: Path, variant: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    root = restored_tree(directory / 'tree')
    public = (root / evaluator.REWRITES).read_text()
    if variant == 'partial':
        public = public.replace('if is_access())', 'if constrain_access("?access".parse().unwrap(), |access| access.shape.ndim() <= 1 || access.shape.slice()[0] == 1))')
    elif variant == 'selective':
        public = public.replace('if is_access())', 'if constrain_access("?access".parse().unwrap(), |access| access.shape.ndim() > 1 || access.item_shape.ndim() > 1))')
    elif variant != 'exact':
        raise ValueError(variant)
    (root / evaluator.REWRITES).write_text(public)
    patch = directory / (variant + '.patch')
    write_patch(root, patch)
    return patch


def restored_tree(destination: Path) -> Path:
    root = copy_writable_tree(evaluator.APP, destination)
    subprocess.run(['patch', '--batch', '--fuzz=0', '-p1', '-i', str(evaluator.REFERENCE_PATCH)],
                   cwd=root, check=True, capture_output=True, text=True)
    return root


def write_patch(root: Path, destination: Path) -> None:
    sections = []
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        original = evaluator.APP / relative
        before = original.read_text() if original.exists() else ''
        after = path.read_text()
        if before != after:
            sections.append('diff --git a/' + relative + ' b/' + relative + '\n')
            sections.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                fromfile='a/' + relative if original.exists() else '/dev/null', tofile='b/' + relative))
    destination.write_text(''.join(sections))
