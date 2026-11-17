"""Run the retained reconstruction entrypoint inside its isolate instance and pack what it wrote.

usage: method_runner.py <result.tar> <output_dir> [--ema-set <root> <list>] <entrypoint> <args...>

The judge stages this file read-only into the instance (isolate.run) next to the workload and
runs it with the instance's copy of the tree as the working directory. It runs the entrypoint as
`python <entrypoint> <args...>` would (the tree first on sys.path), then writes one uncompressed
tar of eval_summary.txt and the epoch_001_use_ema=False/True PNG directories found under
<output_dir> (names relative to it) to <result.tar>, the instance's declared output; the judge
recomputes every metric from those PNGs.

With --ema-set (the graded suite) the EMA evaluation receives a loader over a second held-out
validation set, built exactly like the retained validation loader (same dataset class, transform,
sampler, batch size and workers), so no result computed for one evaluation can be looked up in
the other. Everything else, including the ordinary evaluation, is the unchanged entrypoint.
"""
from __future__ import annotations

import runpy
import sys
import tarfile
from pathlib import Path

PACKED_DIRS = ("epoch_001_use_ema=False", "epoch_001_use_ema=True")


def _ema_loader(loader, root: str, data_list: str):
    import torch.utils.data
    from utils.loader import ListDataset

    dataset = ListDataset(
        root, data_list=data_list, transform=loader.dataset.transform, loader_name="img_loader",
        return_label=False, return_index=True, should_flip=False,
    )
    sampler = torch.utils.data.DistributedSampler(
        dataset, num_replicas=loader.sampler.num_replicas, rank=loader.sampler.rank, shuffle=False,
    )
    return torch.utils.data.DataLoader(
        dataset, sampler=sampler, batch_size=loader.batch_size, num_workers=loader.num_workers,
        pin_memory=loader.pin_memory, drop_last=False,
    )


def install(root: str, data_list: str) -> None:
    import utils.train_utils as train_utils

    original = train_utils.evaluate_tokenizer

    def evaluate_tokenizer(args, model, ema_model, data_loader_val, *rest, **kwargs):
        use_ema = kwargs["use_ema"] if "use_ema" in kwargs else (rest[2] if len(rest) > 2 else True)
        if use_ema:
            data_loader_val = _ema_loader(data_loader_val, root, data_list)
        return original(args, model, ema_model, data_loader_val, *rest, **kwargs)

    train_utils.evaluate_tokenizer = evaluate_tokenizer


def pack(output_dir: Path, result: Path) -> None:
    """eval_summary.txt files and the two PNG directories, regular files only."""
    files = sorted(output_dir.rglob("eval_summary.txt"))
    for name in PACKED_DIRS:
        for directory in output_dir.rglob(name):
            files.extend(sorted(p for p in directory.iterdir()))
    with tarfile.open(result, "w", format=tarfile.PAX_FORMAT) as tar:
        for path in files:
            if path.is_file() and not path.is_symlink():
                tar.add(path, arcname=str(path.relative_to(output_dir)), recursive=False)


def main(argv: list[str]) -> None:
    result, output_dir, *rest = argv
    ema = None
    if rest[:1] == ["--ema-set"]:
        ema, rest = rest[1:3], rest[3:]
    entrypoint, *args = rest
    entry = Path(entrypoint).resolve()
    here = Path(__file__).resolve().parent
    # Same import path as `python main_reconstruction.py`: the tree first, never this directory.
    sys.path[:] = [str(entry.parent)] + [p for p in sys.path[1:] if p and Path(p).resolve() != here]
    if ema:
        install(*ema)
    sys.argv = [str(entry), *args]
    try:
        runpy.run_path(str(entry), run_name="__main__")
    except SystemExit as stop:
        if stop.code not in (None, 0):
            raise
    pack(Path(output_dir), Path(result))


if __name__ == "__main__":
    main(sys.argv[1:])
