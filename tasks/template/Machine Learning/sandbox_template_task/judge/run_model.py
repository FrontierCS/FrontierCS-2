"""The fit driver on a class-1 machine (judge/evaluator.py uploads it with the tree and the data).

    run_model.py TREE DATA OUT

For each workload DATA/<index>/{train,queries}.json: model.fit on the training points, then
model.predict at the query x; OUT/<index>/params.json and OUT/<index>/predictions.json. It runs
beside the candidate's code on the candidate's machine, so the judge trusts nothing it writes: it
validates the files and recomputes the error itself.
"""

import json
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    tree, data, out = (Path(arg) for arg in argv)
    sys.path.insert(0, str(tree))
    import model

    for folder in sorted(data.iterdir()):
        train = json.loads((folder / "train.json").read_text())
        queries = json.loads((folder / "queries.json").read_text())
        params = model.fit(train["x"], train["y"])
        target = out / folder.name
        target.mkdir()
        (target / "params.json").write_text(json.dumps(params))
        (target / "predictions.json").write_text(json.dumps(model.predict(params, queries["x"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
