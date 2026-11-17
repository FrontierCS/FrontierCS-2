"""trim.py <judge directory>: leaves the final suite out of a copy of the judge, in place, for the
train judge's image.

The final suite is `judge_data/private/final/`, which the script removes whole, and its entries in
the two records of the private inputs next to it: the `final/...` keys of `files` in
`assets-manifest.json`, and the images of `attribution.json` whose `suite` is `final`. The
evaluator reads only the suite its mode selects (`judge/evaluator.py`'s `_suite_root`), so a train
call, which selects `development`, reads none of what is removed. Both records are written as
the source writes them (`json.dumps(..., indent=2)` and a newline), and nothing else in them
changes.
"""
import json
import shutil
import sys
from pathlib import Path

private = Path(sys.argv[1]) / "judge_data" / "private"
shutil.rmtree(private / "final")

manifest_path = private / "assets-manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
manifest["files"] = {name: row for name, row in manifest["files"].items() if not name.startswith("final/")}
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

attribution_path = private / "attribution.json"
attribution = json.loads(attribution_path.read_text(encoding="utf-8"))
attribution["images"] = [image for image in attribution["images"] if image["suite"] != "final"]
attribution_path.write_text(json.dumps(attribution, indent=2) + "\n", encoding="utf-8")
