"""Prepare the released public statistics and offline model cache."""

import argparse
import hashlib
import json
from pathlib import Path
import pickle


class ArrayUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {("numpy", "ndarray"), ("numpy", "dtype")}
        for prefix in ("numpy.core", "numpy._core"):
            allowed.update((prefix + ".multiarray", item)
                           for item in ("_reconstruct", "scalar"))
            allowed.add((prefix + ".numeric", "_frombuffer"))
        if (module, name) not in allowed:
            raise pickle.UnpicklingError("unexpected statistics object")
        return super().find_class(module, name)


def prepare(model_root, output_root):
    import numpy as np

    model_root = Path(model_root).resolve()
    output_root = Path(output_root).resolve()
    assets = json.loads(Path(__file__).with_name("frontier_public_assets.json").read_text())
    stats_item = next(item for item in assets if item["path"].endswith("paper_ref_stats.pkl"))
    bundle_path = model_root / stats_item["installed_path"]
    digest = hashlib.sha256()
    with bundle_path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        if digest.hexdigest() != stats_item["sha256"]:
            raise ValueError("released statistics bundle hash mismatch")
        stream.seek(0)
        bundle = ArrayUnpickler(stream).load()
    if not isinstance(bundle, dict):
        raise ValueError("invalid released statistics bundle")
    stats = output_root / "stats"
    stats.mkdir(parents=True, exist_ok=True)
    for name, arrays in bundle.items():
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".npz"):
            raise ValueError("invalid statistics filename")
        if not isinstance(arrays, dict) or not {"mu", "sigma"} <= arrays.keys():
            raise ValueError("missing released moments")
        for value in arrays.values():
            if not isinstance(value, np.ndarray) or value.dtype.kind not in "fiu" or not np.isfinite(value).all():
                raise ValueError("invalid released statistics array")
        dim = arrays["mu"].size
        if arrays["mu"].shape != (dim,) or arrays["sigma"].shape != (dim, dim):
            raise ValueError("invalid released moment shapes")
        if not np.allclose(arrays["sigma"], arrays["sigma"].T, atol=1e-10):
            raise ValueError("asymmetric released covariance")
        np.savez(stats / name, **arrays)

    # The image stages repository files directly; timm expects HF snapshot paths.
    for item in assets:
        source = model_root / item["installed_path"]
        if not source.is_file() or source.stat().st_size != item["size_bytes"]:
            raise ValueError("missing or wrong-size public asset: " + str(source))
        if not item.get("repo_id", "").startswith("timm/"):
            continue
        hub = output_root / "cache/hub" / ("models--" + item["repo_id"].replace("/", "--"))
        target = hub / "snapshots" / item["revision"] / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink() and target.resolve() == source.resolve():
            pass
        elif target.exists() or target.is_symlink():
            raise ValueError("conflicting offline cache entry: " + str(target))
        else:
            target.symlink_to(source)
        (hub / "refs").mkdir(exist_ok=True)
        (hub / "refs/main").write_text(item["revision"])
    return {"stats": str(stats), "cache": str(output_root / "cache"),
            "statistics_files": sorted(bundle)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=Path("/opt/frontier/models"))
    parser.add_argument("--output-root", type=Path, default=Path("/workspace/fd-assets"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.model_root, args.output_root), sort_keys=True))
