"""trim.py <judge directory>: leaves the final suite's articles out of a copy of the judge, in
place, for the train judge's image.

The articles that `evaluator.py` calls `HIDDEN_ARTICLES` are removed from: that tuple (it becomes
empty, so the final suite is the public one), the frozen subset's rows, the reference targets'
`workloads`, and the article lists of the targets' `calibration_captures`. Nothing a train call
reads changes: it selects `PUBLIC_ARTICLES` alone.
"""
import ast
import json
import re
import sys
from pathlib import Path

judge = Path(sys.argv[1])
source = (judge / "evaluator.py").read_text(encoding="utf-8")
match = re.search(r"^HIDDEN_ARTICLES = \(.*?\)\n", source, re.DOTALL | re.MULTILINE)
assert match and len(re.findall(r"^HIDDEN_ARTICLES = ", source, re.MULTILINE)) == 1, "HIDDEN_ARTICLES"
hidden = set(ast.literal_eval(match.group(0).split("=", 1)[1].strip()))
assert len(hidden) == 12, hidden
(judge / "evaluator.py").write_text(
    source[:match.start()] + "HIDDEN_ARTICLES: tuple[str, ...] = ()\n" + source[match.end():], encoding="utf-8")

subset = judge / "data" / "quality_subset.jsonl"
rows = subset.read_text(encoding="utf-8").splitlines(keepends=True)
kept = [row for row in rows if json.loads(row)["article_id"] not in hidden]
assert len(kept) == len(rows) // 2 and all(json.loads(row)["article_id"] not in hidden for row in kept)
subset.write_text("".join(kept), encoding="utf-8")

path = judge / "targets" / "manifest.json"
manifest = json.loads(path.read_text(encoding="utf-8"))
assert hidden <= set(manifest["workloads"])
manifest["workloads"] = {key: value for key, value in manifest["workloads"].items() if key not in hidden}
for capture in manifest["calibration_captures"]:
    capture["articles"] = [article for article in capture["articles"] if article not in hidden]
path.write_text(json.dumps(manifest, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")
