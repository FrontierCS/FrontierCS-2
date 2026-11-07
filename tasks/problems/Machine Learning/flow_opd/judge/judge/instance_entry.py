"""The first command of a workload's isolate instance (judge-owned, staged as an input).

usage: instance_entry.py <in dir> <out dir> <run output dir>

Points the run profile at the instance's own copy of the corpus, runs the unchanged shell
launcher in the patched tree (the instance's work dir, the cwd), and copies the run's reports to
the instance's out/ dir, from where the judge collects them; every other file of the run is
deleted with the instance, so out/files.json lists their sizes. Exits with the launcher's status.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

ENTRYPOINT = 'scripts/single_node/sd3_opd_example.sh'
# The run reports the judge reads (judge/judge/shell_workload.py REPORTS).
REPORTS = ('progress.json', 'effective_config.json', 'runtime_adaptations.json')


def main(inputs, out, output):
    inputs, out, output = Path(inputs), Path(out), Path(output)
    profile = json.loads((inputs/'run-profile.json').read_text())
    profile['dataset'] = str(inputs/'corpus')
    output.mkdir(parents=True, exist_ok=True)
    staged = output.parent/'run-profile.json'
    staged.write_text(json.dumps(profile))
    os.environ['FRONTIER_RUN_PROFILE'] = str(staged)
    os.environ['FRONTIER_RUN_OUTPUT'] = str(output)
    returncode = subprocess.call(['bash', ENTRYPOINT])
    for name in REPORTS:
        path = output/name
        if path.is_file() and not path.is_symlink():
            (out/name).write_bytes(path.read_bytes())
    files = {}
    for path in sorted(output.rglob('*')):
        if path.is_file() and not path.is_symlink():
            files[str(path.relative_to(output))] = path.stat().st_size
    (out/'files.json').write_text(json.dumps(files))
    return returncode


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
