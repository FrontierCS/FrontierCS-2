"""trim.py <judge directory>: leaves the final suite out of a copy of the judge, in place, for
the train judge's image.

A train call runs the four feedback workloads alone (`evaluator.py:324-332,364`: the graded
suite only when the role is `final`). The final suite is the twelve graded workloads' seeds and
their cached dense and reference outputs, so this:

- deletes `dense_targets/*graded-*.mp4`, the graded outputs;
- removes the graded workloads' records from `dense_targets/manifest.json`, written as the source
  writes it (`json.dumps(..., indent=2, sort_keys=True)`, one newline at the end), which the script
  checks first; `_load_manifest` checks only its schema and models (`evaluator.py:447-468`);
- sets every seed of `GRADED_SUITE` in `evaluator.py` to 0, keeping its twelve rows, models and
  prompts, and nothing else of the file.
"""
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
targets = root / "dense_targets"

graded = sorted(targets.glob("*graded-*.mp4"))
assert len(graded) == 24, len(graded)
for path in graded:
    path.unlink()

text = (targets / "manifest.json").read_text(encoding="utf-8")
manifest = json.loads(text)
assert json.dumps(manifest, indent=2, sort_keys=True) + "\n" == text
names = [name for name in manifest["targets"] if "_graded_" in name]
assert len(names) == 12, names
for name in names:
    del manifest["targets"][name]
(targets / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

source = (root / "evaluator.py").read_text(encoding="utf-8")
block = re.search(r"^GRADED_SUITE = \(\n(.*?)\n\)\n", source, re.MULTILINE | re.DOTALL)
assert block and source.count("GRADED_SUITE = (") == 1
rows = re.subn(r'\(("(?:wan|hunyuan)"), (\d+), \d+\)', r"(\1, \2, 0)", block.group(1))
assert rows[1] == 12, rows[1]
(root / "evaluator.py").write_text(source[:block.start(1)] + rows[0] + source[block.end(1):], encoding="utf-8")
