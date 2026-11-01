"""Run the retained reconstruction entrypoint with the EMA evaluation on its own image set.

usage: final_launcher.py <ema_image_root> <ema_list> <tree>/main_reconstruction.py <entrypoint args...>

The retained entrypoint evaluates the ordinary and the EMA model one after the other. Here the
EMA call receives a loader over a second held-out validation set, built exactly like the
retained validation loader (same dataset class, transform, sampler, batch size and workers),
so no result computed for one evaluation can be looked up in the other. Everything else,
including the ordinary evaluation, is the unchanged entrypoint.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path


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


def main(argv: list[str]) -> None:
    root, data_list, entrypoint, *args = argv
    entry = Path(entrypoint).resolve()
    judge = Path(__file__).resolve().parent
    # Same import path as `python main_reconstruction.py`: the tree first, never the judge directory.
    sys.path[:] = [str(entry.parent)] + [p for p in sys.path[1:] if p and Path(p).resolve() != judge]
    install(root, data_list)
    sys.argv = [str(entry), *args]
    runpy.run_path(str(entry), run_name="__main__")


if __name__ == "__main__":
    main(sys.argv[1:])
