"""Prints the agent's `/frontier/task/README.md` of a task: `python3 -I join.py <task> <platform>`.

It is the task's `<task>/README.md` with each line that is only `{<name>}` replaced by the file
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

import re
import sys
from pathlib import Path

# A line that is a file's name in braces, and nothing else.
SLOT = re.compile(r"\{([a-z]+)\}\n")


def lines(path):
    """The lines of the file at `path`, each with its newline, which the file's last line must have."""
    text = path.read_bytes().decode()
    if not text.endswith("\n"):
        sys.exit(f"join.py: {path} does not end with a newline: end its last line")
    return re.findall(r"[^\n]*\n", text)


def main(task, platform):
    own = task / "readme"
    used = set()

    def joined(path):
        out = []
        for line in lines(path):
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
    main(Path(sys.argv[1]), Path(sys.argv[2]))
