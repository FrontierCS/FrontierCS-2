"""trim.py <judge directory>: leaves the final workloads' photographs, models, batch sizes and
seeds out of a copy of the judge, in place, for the train judge's image.

The final workloads of `judge_assets/flower_photos/manifest.json` become `hidden_0` to `hidden_2`
with the first development workload's model, batch size, seed and shard sizes, and their 90 files
are replaced by placeholders of distinct bytes and generic names, one for each row the manifest
keeps. These are the shapes `evaluator.py`'s `_load_manifest` checks (six workloads, three of each role, every
file present with its SHA-256, no file reused, four training and two validation files for each
class). A train call selects the development workloads alone, so nothing it reads changes.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1]) / "judge_assets" / "flower_photos"
path = root / "manifest.json"
manifest = json.loads(path.read_text(encoding="utf-8"))

development = [row for row in manifest["workloads"] if row["role"] == "development"]
final = [row for row in manifest["workloads"] if row["role"] == "final"]
assert len(development) == 3 and len(final) == 3, manifest["workloads"]

for row in final:
    shutil.rmtree(root / "images" / row["id"])

renamed = {row["id"]: f"hidden_{index}" for index, row in enumerate(final)}
manifest["workloads"] = development + [
    {**development[0], "id": renamed[row["id"]], "role": "final"} for row in final]

files = []
counts = {}
for row in manifest["files"]:
    if row["role"] != "final":
        files.append(row)
        continue
    identity = renamed[row["workload"]]
    key = (identity, row["split"], row["class"])
    counts[key] = counts.get(key, 0) + 1
    asset = f"images/{identity}/{row['split']}/{row['class']}/placeholder_{counts[key]}.jpg"
    data = f"placeholder {asset}\n".encode()
    (root / asset).parent.mkdir(parents=True, exist_ok=True)
    (root / asset).write_bytes(data)
    files.append({**row, "workload": identity, "source_path": asset, "asset_path": asset,
                  "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
assert len(files) == 180 and not any((root / "images").glob("final_*")), len(files)
manifest["files"] = files
path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
