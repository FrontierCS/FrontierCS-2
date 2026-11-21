"""Prints the agent's README: `python3 -I join.py <task> <platform> '<JSON metric pairs>'`.

In `<task>/README.md`, a line containing only `{metrics}` expands to the task's ordered public
metric names and Markdown descriptions. Adjacent names with the same description share a bullet.
Each other line that is only `{<name>}` is replaced by the file
`<name>.md`, joined the same way: the task's own, `<task>/readme/<name>.md`, if it has one, else
the platform's, `<platform>/<name>.md`. So the platform's text holds what every task's README
says, and a task gives its values, and the parts its source words otherwise, as files of its
own, which win, as the including file's keys win over an `include` in lm-evaluation-harness
(https://github.com/EleutherAI/lm-evaluation-harness/blob/d6de81643928d653435c431bae19945d41d32520/lm_eval/tasks/_yaml_loader.py#L190-L208).
A README with no such line is printed as it is. As nixpkgs' `replaceVars` does
(https://github.com/NixOS/nixpkgs/blob/06c561b7cdfb8ea2f98ac13811a4f2da5f16f931/pkgs/build-support/replace-vars/replace-vars.nix#L3-L15),
it fails on a name that neither directory has a file for, and on a file of the task's that no
line names, so that a misspelt name stops the build.
"""

import json
import re
import sys
from itertools import groupby
from pathlib import Path

# A line that is a file's name in braces, and nothing else.
SLOT = re.compile(r"\{([a-z]+)\}\n")


def lines(path):
    """The lines of the file at `path`, each with its newline, which the file's last line must have."""
    text = path.read_bytes().decode()
    if not text.endswith("\n"):
        sys.exit(f"join.py: {path} does not end with a newline: end its last line")
    return re.findall(r"[^\n]*\n", text)


def metric_bullets(metrics):
    if not metrics:
        return "- No task-specific metrics.\n"
    bullets = []
    for description, group in groupby(metrics, key=lambda pair: pair[1]):
        names = ", ".join(f"`{name}`" for name, _ in group)
        bullets.append(f"- {names}: {description}\n")
    return "".join(bullets)


def main(task, platform, metrics):
    own = task / "readme"
    used = set()

    def joined(path):
        out = []
        for line in lines(path):
            if line == "{metrics}\n":
                out.append(metric_bullets(metrics))
                continue
            slot = SLOT.fullmatch(line)
            if not slot:
                out.append(line)
                continue
            name = slot[1] + ".md"
            if (own / name).is_file():
                used.add(name)
                out.append(joined(own / name))
            elif (platform / name).is_file():
                out.append(joined(platform / name))
            else:
                sys.exit(f"join.py: {path} names `{slot[1]}`, which neither {own} nor {platform} has: add {own / name}")
        return "".join(out)

    readme = joined(task / "README.md")
    unused = sorted({p.name for p in own.glob("*")} - used) if own.is_dir() else []
    if unused:
        sys.exit(f"join.py: no line names {', '.join(unused)} of {own}: name each, or remove it")
    sys.stdout.buffer.write(readme.encode())


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), json.loads(sys.argv[3]))
