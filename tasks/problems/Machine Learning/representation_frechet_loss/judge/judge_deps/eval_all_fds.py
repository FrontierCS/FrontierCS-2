"""Evaluate Frechet Distance across multiple repr models, cfg scales, and EMA copies.

Generates images for every (cfg, ema_label) combination and computes FD using
InceptionV3, DINOv2-L, MAE-L, CLIP-L, etc. Results saved to final_eval_summary.csv.

Usage (model-based):
    torchrun --nproc_per_node=8 eval_all_fds.py \
        --model pMF_B --rope_2d --learned_pe \
        --load_from checkpoints/post-trained/pMF-B_FD-SIM.pth \
        --num_sampling_steps 1 --interval_min 0.1 --interval_max 0.7 \
        --eval_bsz 256 --num_images 50000 \
        --cfg_list 2.0 4.0 6.0 7.0 8.0 8.5 9.0 10.0 12.0 14.0 \
        --models inception vit_large_patch16_224.mae vit_large_patch14_dinov2.lvd142m

Usage (folder-based — no model checkpoint needed):
    python eval_all_fds.py \
        --image_folder data/imagenet/gt-image50000 \
        --models inception vit_large_patch16_224.mae \
                 vit_large_patch14_dinov2.lvd142m vit_large_patch14_clip_224.openai

    torchrun --nproc_per_node=8 eval_all_fds.py \
        --image_folder data/imagenet/gt-image50000 --no_prc
"""

import argparse
import csv
import datetime
import logging
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist


from tqdm import tqdm

from frechet_distance.metrics import compute_fid as np_fid, compute_isc as np_isc
from frechet_distance.datasets import ImageFolderDataset, ImageListDataset, build_dataloader
from frechet_distance.repr_models import load_repr_model, model_short_name
from utils.distributed_util import (
    broadcast_scalar, get_global_rank, get_world_size, is_enabled,
)
from utils.eval_util import get_start_end_indices, _prepare_eval_classes
from utils.vis_util import visualize
from frechet_distance.evaluator import extract_ref_features, gather_features
from frechet_distance.metrics import compute_precision_recall, compute_mmd

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

INCEPTION_STATS = [
    ("FID(JiT)",     "data/fid_stats/jit_in256_stats.npz"),
    ("FID(ADM)",     "data/fid_stats/guided_diffusion_stats.npz"),
]

TARGET_SIZE = 256

DEFAULT_MODELS = [
    "inception",
    "vit_large_patch16_224.mae",
    "vit_large_patch14_dinov2.lvd142m",
    "vit_large_patch14_clip_224.openai",
    "vit_so400m_patch16_siglip_256.v2_webli",
    "convnext",
]

# Per-model target size overrides (model name -> target size).
# Models not listed here use the global TARGET_SIZE.
DEFAULT_TARGET_SIZES = {
    "vit_large_patch16_224.mae": 224,
    "vit_so400m_patch16_siglip_256.v2_webli": 224,
}

# Validation-set raw FD values used to convert raw FD to FDr.
# FDr-X = rawFD-X / valFD-X. FDr-6 is the arithmetic mean over these six
# representation spaces.
FDR_VALIDATION_FD = {
    "FID(ADM)": 1.68,
    "convnext": 56.87,
    "dinov2_cls": 14.19,
    "mae_cls": 0.04,
    "siglip_cls": 0.60,
    "clip_cls": 5.60,
}

FDR6_MODELS = (
    "FID(ADM)",
    "convnext",
    "dinov2_cls",
    "mae_cls",
    "siglip_cls",
    "clip_cls",
)



logger = logging.getLogger("FD_loss")


# ---------------------------------------------------------------------------
# Core: accumulate features, reduce, compute FD
# ---------------------------------------------------------------------------

def _make_accumulators(repr_models, device):
    """Create accumulators, sharing them for entries with the same (model, pool_type)."""
    accumulators = []
    acc_map = {}  # (model_id, pool_type) -> accumulator dict
    inception_idx = None
    for i, repr_entry in enumerate(repr_models):
        key = (id(repr_entry["model"]), repr_entry.get("pool_type", "cls"))
        if key in acc_map:
            accumulators.append(acc_map[key])  # shared reference
        else:
            dim = repr_entry["feat_dim"]
            acc = {
                "feat_sum": torch.zeros(dim, dtype=torch.float64, device=device),
                "feat_outer": torch.zeros(dim, dim, dtype=torch.float64, device=device),
                "count": 0,
            }
            acc_map[key] = acc
            accumulators.append(acc)
        if repr_entry["has_logits"] and inception_idx is None:
            inception_idx = i
    return accumulators, inception_idx


@torch.inference_mode()
def accumulate_batch(images, repr_models, accumulators, inception_logits,
                      local_prc_feats):
    """Extract features from one batch and accumulate into sufficient stats.

    images must be in [0, 1] range (model-ready).  Caches forward passes per
    model object so that cls/avg entries sharing the same model don't trigger
    redundant computation.  Accumulators are shared for entries with the same
    (model, pool_type), so updates happen only once.
    """
    fwd_cache = {}  # id(model) -> (output_0, output_1)
    updated = set()  # accumulator ids already updated this batch

    for i, repr_entry in enumerate(repr_models):
        model = repr_entry["model"]
        model_id = id(model)
        if model_id not in fwd_cache:
            if repr_entry["has_logits"]:
                fwd_cache[model_id] = model(images)
            else:
                with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                    fwd_cache[model_id] = model(images)

        out_0, out_1 = fwd_cache[model_id]

        if repr_entry["has_logits"]:
            feats = out_0
        else:
            pool_type = repr_entry.get("pool_type", "cls")
            feats = out_0 if pool_type == "cls" else out_1

        if repr_entry["name"] in local_prc_feats:
            local_prc_feats[repr_entry["name"]].append(feats.float().cpu())

        # Only accumulate once per shared accumulator
        acc_id = id(accumulators[i])
        if acc_id not in updated:
            feats64 = feats.double()
            accumulators[i]["feat_sum"].add_(feats64.sum(0))
            accumulators[i]["feat_outer"].addmm_(feats64.T, feats64)
            accumulators[i]["count"] += feats.shape[0]
            updated.add(acc_id)

    # Collect inception logits once per batch (all inception entries share the model)
    for repr_entry in repr_models:
        if repr_entry["has_logits"]:
            _, logits = fwd_cache[id(repr_entry["model"])]
            inception_logits.append(logits.cpu())
            break


def _reduce_and_compute(repr_models, accumulators, inception_logits,
                        inception_idx, local_feat_lists, *,
                        prc_ref_features=None, prc_model_names=None,
                        mmd_ref_features=None, mmd_model_names=None,
                        prc_k=3, prc_batch_size=5000):
    """Reduce across ranks, compute FD / IS / P&R / CMMD.

    ``local_feat_lists`` collects per-sample features for the union of
    P&R and CMMD models.  Each metric uses its own reference features.
    """
    world_size, rank = get_world_size(), get_global_rank()
    device = torch.device("cuda")
    distributed = is_enabled()
    prc_ref_features = prc_ref_features or {}
    prc_model_names = set(prc_model_names or [])
    mmd_ref_features = mmd_ref_features or {}
    mmd_model_names = set(mmd_model_names or [])
    feat_model_names = prc_model_names | mmd_model_names

    # Reduce each unique accumulator once, then compute FD per entry
    reduced = {}  # id(acc) -> (mu, sigma) or None (non-rank-0)
    results = {}
    for i, repr_entry in enumerate(repr_models):
        acc = accumulators[i]
        acc_id = id(acc)

        if acc_id not in reduced:
            if distributed:
                for key in ("feat_sum", "feat_outer"):
                    dist.reduce(acc[key], dst=0, op=dist.ReduceOp.SUM)
                count_t = torch.tensor([acc["count"]], dtype=torch.long, device=device)
                dist.reduce(count_t, dst=0, op=dist.ReduceOp.SUM)
                total = count_t.item()
            else:
                total = acc["count"]

            if rank == 0:
                feat_sum = acc["feat_sum"].cpu().numpy()
                feat_outer = acc["feat_outer"].cpu().numpy()
                mu = (feat_sum / total).astype(np.float64)
                sigma = ((feat_outer - np.outer(feat_sum, feat_sum) / total)
                         / (total - 1)).astype(np.float64)
                reduced[acc_id] = (mu, sigma)
            else:
                reduced[acc_id] = None

        if rank == 0:
            mu, sigma = reduced[acc_id]
            fd_val = float(np_fid(mu, sigma,
                                  repr_entry["mu_ref"].cpu().numpy(),
                                  repr_entry["sigma_ref"].cpu().numpy()))
        else:
            fd_val = 0.0
        results[repr_entry["name"]] = broadcast_scalar(fd_val)

    # Inception Score
    is_val = None
    if inception_idx is not None and inception_logits:
        local_logits = torch.cat(inception_logits, dim=0).to(device)
        if distributed:
            all_logits = gather_features(local_logits, world_size, rank, device)
            is_val = float(np_isc(all_logits.cpu())[0]) if rank == 0 else 0.0
            is_val = broadcast_scalar(is_val)
        else:
            is_val, _ = np_isc(local_logits.cpu())

    # Per-sample metrics (P&R, CMMD) — require gathered features + reference.
    # NOTE: feat_model_names is the same on all ranks; ref dicts are only
    # populated on rank 0.  Guard on feat_model_names (not ref dicts) to
    # ensure all ranks participate in the collective gather/broadcast.
    prc_results: dict[str, tuple[float, float]] = {}
    mmd_results: dict[str, float] = {}
    for feat_name in sorted(feat_model_names):
        if not local_feat_lists.get(feat_name):
            continue
        local_feats = torch.cat(local_feat_lists.pop(feat_name), dim=0).to(device)
        all_gen = (gather_features(local_feats, world_size, rank, device)
                   if distributed else local_feats)
        del local_feats

        do_prc = feat_name in prc_model_names
        do_mmd = feat_name in mmd_model_names

        if rank == 0:
            gen_gpu = all_gen.float()
            del all_gen
            metrics_str = []

            if do_prc:
                prc_ref = prc_ref_features[feat_name].to(device=device, dtype=torch.float32)
                logger.info(f"P&R ({feat_name}): ref={prc_ref.shape[0]}, "
                            f"gen={gen_gpu.shape[0]}, k={prc_k}")
                prec, rec = compute_precision_recall(prc_ref, gen_gpu, k=prc_k,
                                                     batch_size=prc_batch_size)
                del prc_ref
                metrics_str += [f"Precision={prec:.4f}", f"Recall={rec:.4f}"]
            else:
                prec, rec = None, None

            if do_mmd:
                mmd_ref = mmd_ref_features[feat_name].to(device=device, dtype=torch.float32)
                mmd_val = compute_mmd(mmd_ref, gen_gpu)
                del mmd_ref
                metrics_str.append(f"CMMD={mmd_val:.4f}")
            else:
                mmd_val = None

            del gen_gpu
            if metrics_str:
                logger.info(f"  {feat_name}: {'  '.join(metrics_str)}")
        else:
            prec, rec, mmd_val = None, None, None
            del all_gen

        if do_prc:
            prc_results[feat_name] = (
                broadcast_scalar(prec if prec is not None else 0.0),
                broadcast_scalar(rec if rec is not None else 0.0),
            )
        if do_mmd:
            mmd_results[feat_name] = broadcast_scalar(
                mmd_val if mmd_val is not None else 0.0)
        torch.cuda.empty_cache()

    return results, is_val, prc_results, mmd_results


# ---------------------------------------------------------------------------
# Image sources: generate or load from folder
# ---------------------------------------------------------------------------

@torch.inference_mode()
def _generate_and_evaluate(args, model, ema_model, repr_models, cfg, ema_label,
                           num_images, *, tokenizer=None,
                           prc_ref_features=None, prc_model_names=None,
                           mmd_ref_features=None, mmd_model_names=None):
    """Generate images and compute FD/IS/P&R/CMMD."""
    from utils.data_util import save_image, to_uint8_numpy
    from utils.sampling_util import generate_images

    world_size, rank = get_world_size(), get_global_rank()
    device = torch.device("cuda")
    prc_model_names = prc_model_names or []
    mmd_model_names = mmd_model_names or []
    feat_model_names = sorted(set(prc_model_names) | set(mmd_model_names))

    start_idx, end_idx = get_start_end_indices(num_images, world_size, rank)
    local_n = end_idx - start_idx
    bsz = min(args.eval_bsz, local_n)
    rank_classes = _prepare_eval_classes(args, num_images, start_idx, end_idx)

    save_images = getattr(args, "save_eval_images", False)
    eval_dir = None
    if save_images:
        eval_dir = os.path.join(
            args.log_dir, "eval_images",
            f"ema={ema_label}-cfg={cfg}-steps={args.num_sampling_steps}-"
            f"interval_min={args.interval_min}-interval_max={args.interval_max}",
        )
        if rank == 0:
            os.makedirs(eval_dir, exist_ok=True)

    accumulators, inception_idx = _make_accumulators(repr_models, device)
    inception_logits: list[torch.Tensor] = []
    local_feat_lists: dict[str, list[torch.Tensor]] = {n: [] for n in feat_model_names}

    generated = 0
    t0 = time.perf_counter()
    with ema_model.swap(model, label=ema_label):
        while generated < local_n:
            b = min(bsz, local_n - generated)
            y = torch.from_numpy(rank_classes[generated:generated + b]).long().to(device)
            images = generate_images(args, model, labels=y, cfg=cfg, tokenizer=tokenizer)

            if save_images and eval_dir is not None:
                imgs_np = to_uint8_numpy(images)

            accumulate_batch(images, repr_models, accumulators, inception_logits,
                              local_feat_lists)

            if save_images and eval_dir is not None:
                for j, img in enumerate(imgs_np):
                    save_image(img, f"{eval_dir}/{start_idx + generated + j:06d}.png")
                del imgs_np

            del images
            generated += b

    gen_time = time.perf_counter() - t0
    torch.cuda.empty_cache()

    results, is_val, prc_results, mmd_results = _reduce_and_compute(
        repr_models, accumulators, inception_logits, inception_idx,
        local_feat_lists,
        prc_ref_features=prc_ref_features, prc_model_names=prc_model_names,
        mmd_ref_features=mmd_ref_features, mmd_model_names=mmd_model_names,
        prc_k=args.prc_k, prc_batch_size=args.prc_batch_size,
    )

    # Cleanup eval images
    if save_images and eval_dir and not getattr(args, "keep_eval_folder", False):
        for idx in range(start_idx, end_idx):
            try:
                os.remove(f"{eval_dir}/{idx:06d}.png")
            except FileNotFoundError:
                pass
        if rank == 0:
            try:
                if not os.listdir(eval_dir):
                    os.rmdir(eval_dir)
            except OSError:
                pass
    elif save_images:
        logger.info(f"Saved images to: {eval_dir}")

    return results, is_val, gen_time, prc_results, mmd_results


@torch.inference_mode()
def _evaluate_from_folder(repr_models, image_dir=None, *,
                          dataset=None, img_size=256, batch_size=64,
                          num_workers=8,
                          prc_ref_features=None, prc_model_names=None,
                          mmd_ref_features=None, mmd_model_names=None,
                          prc_k=3, prc_batch_size=5000):
    """Load images from a folder (or pre-built dataset) and compute FD/IS/P&R/CMMD.

    Handles any image naming convention (png, jpg, jpeg, webp).
    Images are center-cropped to img_size and normalized to [0, 1].
    """
    world_size, rank = get_world_size(), get_global_rank()
    device = torch.device("cuda")
    prc_model_names = prc_model_names or []
    mmd_model_names = mmd_model_names or []
    feat_model_names = sorted(set(prc_model_names) | set(mmd_model_names))

    if dataset is None:
        dataset = ImageFolderDataset(image_dir, img_size=img_size)
    num_images = len(dataset)
    logger.info(f"Evaluating {num_images} images")
    loader = build_dataloader(dataset, batch_size=batch_size, num_workers=num_workers,
                              distributed=world_size > 1)

    accumulators, inception_idx = _make_accumulators(repr_models, device)
    inception_logits: list[torch.Tensor] = []
    local_feat_lists: dict[str, list[torch.Tensor]] = {n: [] for n in feat_model_names}

    count = 0
    t0 = time.perf_counter()
    pbar = tqdm(loader, desc="  Extracting features", disable=(rank != 0))
    for batch in pbar:
        images = batch.to(device)  # already [0, 1] from ImageFolderDataset
        accumulate_batch(images, repr_models, accumulators, inception_logits,
                          local_feat_lists)
        count += images.shape[0]
        pbar.set_postfix(images=count * world_size)
        del images

    # Reduce local count across ranks
    if world_size > 1:
        count_t = torch.tensor([count], dtype=torch.long, device=device)
        dist.reduce(count_t, dst=0, op=dist.ReduceOp.SUM)
        total_count = count_t.item()
    else:
        total_count = count

    elapsed = time.perf_counter() - t0
    logger.info(f"  Extracted {total_count} images in {elapsed:.1f}s")
    torch.cuda.empty_cache()

    results, is_val, prc_results, mmd_results = _reduce_and_compute(
        repr_models, accumulators, inception_logits, inception_idx,
        local_feat_lists,
        prc_ref_features=prc_ref_features, prc_model_names=prc_model_names,
        mmd_ref_features=mmd_ref_features, mmd_model_names=mmd_model_names,
        prc_k=prc_k, prc_batch_size=prc_batch_size,
    )
    return results, is_val, elapsed, total_count, prc_results, mmd_results


# ---------------------------------------------------------------------------
# Repr model & P&R setup
# ---------------------------------------------------------------------------

def _load_repr_models(model_names, img_size, target_size=TARGET_SIZE,
                      target_size_overrides=None):
    """Load repr models and reference stats. Returns list of repr-entry dicts."""
    rank = get_global_rank()
    target_size_overrides = target_size_overrides or {}
    repr_models = []
    for name in model_names:
        if rank == 0:
            logger.info(f"Loading repr model '{name}' ...")
        ts_override = target_size_overrides.get(name, target_size)
        repr_model, feat_dim, has_logits, ts = load_repr_model(
            name, target_size=ts_override,
        )

        if name == "inception":
            # Inception uses multiple reference datasets with custom labels
            for label, stats_path in INCEPTION_STATS:
                if not os.path.exists(stats_path):
                    logger.warning(f"  Skipping {label}: {stats_path} not found")
                    continue
                ref = np.load(stats_path)
                repr_models.append({
                    "name": label, "model": repr_model, "feat_dim": feat_dim,
                    "has_logits": has_logits,
                    "mu_ref": torch.tensor(ref["mu"], device="cuda", dtype=torch.float64),
                    "sigma_ref": torch.tensor(ref["sigma"], device="cuda", dtype=torch.float64),
                    "pool_type": "cls",
                })
                logger.info(f"  '{label}': feat_dim={feat_dim}")
        else:
            stats_name = name
            safe_name = stats_name.replace("/", "_").replace(".", "_")
            if img_size == 512:
                img_size = 256
                
            stats_path = f"data/fid_stats/{safe_name}_in{img_size}_t{ts}_stats.npz"
            short = model_short_name(name)
            # if not os.path.exists(stats_path):
            ref = np.load(stats_path)

            pools = [("cls", "mu", "sigma")]
            has_avg = "avg_mu" in ref
            if has_avg:
                pools.append(("avg", "avg_mu", "avg_sigma"))

            for pool, mu_key, sig_key in pools:
                label = f"{short}_{pool}" if has_avg else short
                repr_models.append({
                    "name": label, "model": repr_model, "feat_dim": feat_dim,
                    "has_logits": has_logits,
                    "mu_ref": torch.tensor(ref[mu_key], device="cuda", dtype=torch.float64),
                    "sigma_ref": torch.tensor(ref[sig_key], device="cuda", dtype=torch.float64),
                    "pool_type": pool,
                })
                logger.info(f"  '{label}': feat_dim={feat_dim}, pool={pool}")
    return repr_models


def _load_prc_refs(prc_model_names, repr_models, prc_ref_dir):
    """Load P&R reference features using all GPUs.

    For dual-output models (cls + avg), runs one forward pass and caches
    both outputs to avoid redundant computation.
    """
    rank = get_global_rank()
    prc_ref_features, prc_names = {}, []
    # Track which models we've already extracted: model_id -> {"cls": T, "avg": T|None}
    extracted = {}

    for prc_raw in prc_model_names:
        short = model_short_name(prc_raw)
        if prc_raw == "inception":
            inception_labels = {label for label, _ in INCEPTION_STATS}
            match = next((entry for entry in repr_models
                          if entry["name"] in inception_labels), None)
            if match is None:
                logger.warning(f"P&R model '{prc_raw}' not in models; skipping")
                continue
            entries_to_register = [(match, "inception", "cls")]
        else:
            entries_to_register = []
            bare_entry = next((e for e in repr_models if e["name"] == short), None)
            if bare_entry is not None:
                entries_to_register.append((bare_entry, short, "cls"))
            else:
                for pool in ("cls", "avg"):
                    entry = next((e for e in repr_models
                                  if e["name"] == f"{short}_{pool}"), None)
                    if entry is not None:
                        entries_to_register.append((entry, f"{short}_{pool}", pool))
            if not entries_to_register:
                logger.warning(f"P&R model '{prc_raw}' not in models; skipping")
                continue

        repr_model = entries_to_register[0][0]["model"]
        model_id = id(repr_model)

        if model_id not in extracted:
            if not os.path.isdir(prc_ref_dir):
                logger.warning(f"No P&R ref dir for {short}; skipping")
                continue

            has_logits = entries_to_register[0][0]["has_logits"]
            feat_dim = entries_to_register[0][0]["feat_dim"]
            has_dual = len(entries_to_register) > 1

            # Check if all individual caches exist — if so, load from cache
            cache_paths = {
                pool: f"{prc_ref_dir.rstrip('/')}_{suffix}.pt"
                for _, suffix, pool in entries_to_register
            }
            all_cached = all(os.path.exists(cp) for cp in cache_paths.values())

            if all_cached and rank == 0:
                result = {}
                for pool, cp in cache_paths.items():
                    result[pool] = torch.load(cp, map_location="cpu", weights_only=True)
                    logger.info(f"Loaded cached P&R ref from {cp} ({result[pool].shape})")
                extracted[model_id] = result
            elif all_cached:
                extracted[model_id] = {pool: None for pool in cache_paths}
            else:
                # One forward pass for both cls and avg
                def both_fn(x, m=repr_model, logits=has_logits, dual=has_dual):
                    if logits:
                        return m(x)[0]
                    with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                        out0, out1 = m(x)
                    if dual and out1 is not None:
                        return torch.cat([out0, out1], dim=1)
                    return out0

                all_feats = extract_ref_features(
                    both_fn, prc_ref_dir, cache_path=None)

                if rank == 0 and all_feats.numel() > 0:
                    if has_dual and all_feats.shape[1] > feat_dim:
                        cls_feats = all_feats[:, :feat_dim]
                        avg_feats = all_feats[:, feat_dim:]
                    else:
                        cls_feats = all_feats
                        avg_feats = None
                    extracted[model_id] = {"cls": cls_feats, "avg": avg_feats}
                    # Cache individually
                    for _, suffix, pool in entries_to_register:
                        f = extracted[model_id].get(pool)
                        if f is not None:
                            cp = cache_paths[pool]
                            os.makedirs(os.path.dirname(cp) or ".", exist_ok=True)
                            torch.save(f, cp)
                            logger.info(f"Cached P&R ref ({suffix}): {f.shape} -> {cp}")
                else:
                    extracted[model_id] = {"cls": None, "avg": None}

        for match, _, pool in entries_to_register:
            prc_name = match["name"]
            prc_names.append(prc_name)
            if rank == 0 and extracted.get(model_id, {}).get(pool) is not None:
                prc_ref_features[prc_name] = extracted[model_id][pool]

    return prc_ref_features, prc_names


# ---------------------------------------------------------------------------
# CSV & summary output
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "model", "fd", "fdr", "fdr6", "is", "precision", "recall", "cmmd", "n",
    "ema_label", "cfg", "step", "interval_min", "interval_max",
    "num_sampling_steps", "timestamp",
]


def _append_csv(csv_path, rows):
    """Append list of dicts to CSV."""
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _compute_fdr6(results):
    vals = []
    for name in FDR6_MODELS:
        if name not in results:
            return None
        vals.append(results[name] / FDR_VALIDATION_FD[name])
    return float(np.mean(vals))


def _resolve_prc(results, is_val, prc_results, mmd_results=None):
    """Resolve per-row derived metrics (inception rows share IS/P&R/CMMD)."""
    mmd_results = mmd_results or {}
    inception_labels = {label for label, _ in INCEPTION_STATS}
    inception_prc = next(((p, r) for name, (p, r) in prc_results.items()
                          if name in inception_labels), (None, None))
    inception_mmd = next((v for name, v in mmd_results.items()
                          if name in inception_labels), None)
    fdr6 = _compute_fdr6(results)
    rows = []
    for rname, fd_val in results.items():
        is_inception = rname in inception_labels
        prec, rec = inception_prc if is_inception else prc_results.get(rname, (None, None))
        mmd_val = inception_mmd if is_inception else mmd_results.get(rname, None)
        fdr = fd_val / FDR_VALIDATION_FD[rname] if rname in FDR_VALIDATION_FD else None
        rows.append({
            "model": rname, "fd": fd_val, "fdr": fdr, "fdr6": fdr6,
            "is": is_val if is_inception else None,
            "precision": prec, "recall": rec, "cmmd": mmd_val,
        })
    return rows


def _build_csv_rows(results, is_val, prc_results, num_images, context,
                    mmd_results=None):
    """Build wide-format CSV rows (one row per model) for _append_csv."""
    def _round(v, n):
        return round(v, n) if v is not None else None
    return [{
        "model": e["model"], "fd": _round(e["fd"], 6),
        "fdr": _round(e["fdr"], 6), "fdr6": _round(e["fdr6"], 6),
        "n": num_images,
        "is": _round(e["is"], 4), "precision": _round(e["precision"], 6),
        "recall": _round(e["recall"], 6), "cmmd": _round(e["cmmd"], 6),
        **context,
    } for e in _resolve_prc(results, is_val, prc_results, mmd_results)]


def _log_summary_table(results, is_val, prc_results, num_images, title,
                       mmd_results=None):
    """Print a formatted summary table to the logger."""
    entries = _resolve_prc(results, is_val, prc_results, mmd_results)
    has_prc = any(e["precision"] is not None for e in entries)
    has_mmd = any(e["cmmd"] is not None for e in entries)
    has_fdr = any(e["fdr"] is not None for e in entries)
    has_fdr6 = any(e["fdr6"] is not None for e in entries)

    logger.info(f"\n{'='*60}\n{title}\n{'='*60}")
    cols = ["Model", "FD", "IS"]
    if has_fdr:
        cols += ["FDr"]
    if has_fdr6:
        cols += ["FDr-6"]
    if has_prc:
        cols += ["Prec", "Recall"]
    if has_mmd:
        cols += ["CMMD"]
    cols += ["N"]
    logger.info("  " + "  ".join(f"{c:>14}" for c in cols))
    logger.info("  " + "  ".join("-" * 14 for _ in cols))
    for e in entries:
        vals = [f"{e['model']:<14}",
                f"{e['fd']:14.4f}",
                f"{e['is']:14.2f}" if e["is"] is not None else f"{'N/A':>14}"]
        if has_fdr:
            vals.append(f"{e['fdr']:14.4f}" if e["fdr"] is not None else f"{'N/A':>14}")
        if has_fdr6:
            vals.append(f"{e['fdr6']:14.4f}" if e["fdr6"] is not None else f"{'N/A':>14}")
        if has_prc:
            for v in (e["precision"], e["recall"]):
                vals.append(f"{v:14.4f}" if v is not None else f"{'N/A':>14}")
        if has_mmd:
            vals.append(f"{e['cmmd']:14.4f}" if e["cmmd"] is not None else f"{'N/A':>14}")
        vals.append(f"{num_images:>14}")
        logger.info("  " + "  ".join(vals))
    logger.info("")


def _print_csv_summary(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path, on_bad_lines="skip")
    df["fd"] = pd.to_numeric(df["fd"], errors="coerce")
    if "fdr" in df:
        df["fdr"] = pd.to_numeric(df["fdr"], errors="coerce")
    if "fdr6" in df:
        df["fdr6"] = pd.to_numeric(df["fdr6"], errors="coerce")

    logger.info("\n" + "=" * 70)
    logger.info("BEST FD PER MODEL")
    logger.info("=" * 70)
    for model_name in df["model"].unique():
        sub = df[df["model"] == model_name].dropna(subset=["fd"])
        if sub.empty:
            continue
        best = sub.loc[sub["fd"].idxmin()]
        prec_str = f"  P={best['precision']:.4f}  R={best['recall']:.4f}" if pd.notna(best.get("precision")) else ""
        fdr_str = f"  FDr={best['fdr']:.4f}" if pd.notna(best.get("fdr")) else ""
        fdr6_str = f"  FDr-6={best['fdr6']:.4f}" if pd.notna(best.get("fdr6")) else ""
        logger.info(f"  {model_name:>20s}: FD={best['fd']:.4f}{fdr_str}{fdr6_str}"
                     f"{prec_str}  cfg={best['cfg']}  ema={best['ema_label']}")

    logger.info("\n" + "=" * 70)
    logger.info("FULL RESULTS")
    logger.info("=" * 70)
    pivot = df.pivot_table(index=["ema_label", "cfg"], columns="model",
                           values="fd", aggfunc="first")
    logger.info("\n" + pivot.to_string())


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def main_folder(args):
    """Evaluate FD/IS/P&R from one or more image folders.

    Supports multiple folders in a single invocation to amortize model
    loading time:
        torchrun --nproc_per_node=8 eval_all_fds.py \
            --image_folder dir1 dir2 dir3 \
            --output_csv out1.csv out2.csv out3.csv
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for name in ("httpx", "timm", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.WARNING)

    # Distributed setup
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    rank = get_global_rank()
    if rank != 0:
        logger.setLevel(logging.WARNING)

    # Normalise folder / csv lists
    folders = args.image_folder
    if isinstance(folders, str):
        folders = [folders]
    csv_list = args.output_csv
    if csv_list is None:
        csv_list = [None] * len(folders)
    elif isinstance(csv_list, str):
        csv_list = [csv_list]
    assert len(folders) == len(csv_list), (
        f"--image_folder ({len(folders)}) and --output_csv ({len(csv_list)}) "
        "must have the same number of entries"
    )

    repr_models = _load_repr_models(args.models, args.img_size,
                                     target_size_overrides=DEFAULT_TARGET_SIZES)

    eval_prc = not args.no_prc
    eval_mmd = getattr(args, "eval_mmd", False)
    prc_model_list = getattr(args, "prc_models", None) or args.models

    prc_ref_features, prc_names = {}, []
    if eval_prc:
        prc_ref_features, prc_names = _load_prc_refs(
            prc_model_list, repr_models, args.prc_ref_dir)

    mmd_ref_features, mmd_names = {}, []
    if eval_mmd:
        mmd_ref_features, mmd_names = _load_prc_refs(
            prc_model_list, repr_models, args.mmd_ref_dir)

    for folder, out_csv in zip(folders, csv_list):
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating folder: {folder}")
        logger.info(f"{'='*60}")

        results, is_val, elapsed, num_images, prc_results, mmd_results = _evaluate_from_folder(
            repr_models, folder,
            img_size=args.img_size, batch_size=args.batch_size,
            num_workers=args.num_workers,
            prc_ref_features=prc_ref_features, prc_model_names=prc_names,
            mmd_ref_features=mmd_ref_features, mmd_model_names=mmd_names,
            prc_k=args.prc_k, prc_batch_size=args.prc_batch_size,
        )

        if rank == 0 and results:
            _log_summary_table(results, is_val, prc_results, num_images,
                               f"Summary: {folder}", mmd_results)
            if out_csv:
                os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
                context = {"timestamp": datetime.datetime.now().strftime("%Y-%m-%d-%H:%M"),
                            "step": 0, "ema_label": "folder", "cfg": 0.0,
                            "interval_min": 0.0, "interval_max": 0.0,
                            "num_sampling_steps": 0}
                _append_csv(out_csv,
                            _build_csv_rows(results, is_val, prc_results, num_images,
                                            context, mmd_results))
                logger.info(f"Results saved to {out_csv}")

        torch.cuda.empty_cache()

    if is_enabled():
        dist.destroy_process_group()


def main_random_train(args):
    """Sample random subsets from training set and evaluate FD/IS/P&R."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for name in ("httpx", "timm", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.WARNING)

    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    rank = get_global_rank()
    if rank != 0:
        logger.setLevel(logging.WARNING)

    # Load training image list
    train_list_path = args.train_list
    data_root = args.data_root
    with open(train_list_path) as f:
        lines = [l.strip().split()[0] for l in f if l.strip()]
    all_paths = [os.path.join(data_root, "train", p) for p in lines]
    logger.info(f"Training set: {len(all_paths)} images")

    repr_models = _load_repr_models(args.models, args.img_size,
                                     target_size_overrides=DEFAULT_TARGET_SIZES)

    eval_prc = not args.no_prc
    eval_mmd = getattr(args, "eval_mmd", False)

    prc_ref_features, prc_names = {}, []
    if eval_prc:
        prc_ref_features, prc_names = _load_prc_refs(
            args.models, repr_models, args.prc_ref_dir)

    mmd_ref_features, mmd_names = {}, []
    if eval_mmd:
        mmd_ref_features, mmd_names = _load_prc_refs(
            args.models, repr_models, args.mmd_ref_dir)

    base_csv = args.output_csv[0] if isinstance(args.output_csv, list) else args.output_csv
    num_samples = args.num_samples
    seeds = list(range(args.num_trials))

    for seed in seeds:
        logger.info(f"\n{'='*60}")
        logger.info(f"Trial seed={seed}: sampling {num_samples} images")
        logger.info(f"{'='*60}")

        rng = random.Random(seed)
        sampled = rng.sample(all_paths, num_samples)
        dataset = ImageListDataset(sampled, img_size=args.img_size)

        results, is_val, elapsed, num_images, prc_results, mmd_results = _evaluate_from_folder(
            repr_models, dataset=dataset,
            img_size=args.img_size, batch_size=args.batch_size,
            num_workers=args.num_workers,
            prc_ref_features=prc_ref_features, prc_model_names=prc_names,
            mmd_ref_features=mmd_ref_features, mmd_model_names=mmd_names,
            prc_k=args.prc_k, prc_batch_size=args.prc_batch_size,
        )

        if rank == 0 and results:
            _log_summary_table(results, is_val, prc_results, num_images,
                               f"Summary: train subset seed={seed}", mmd_results)
            if base_csv:
                csv_path = f"{os.path.splitext(base_csv)[0]}_seed{seed}.csv"
                os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
                context = {
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d-%H:%M"),
                    "step": 0, "ema_label": f"train_seed{seed}", "cfg": 0.0,
                    "interval_min": 0.0, "interval_max": 0.0,
                    "num_sampling_steps": 0,
                }
                _append_csv(csv_path,
                            _build_csv_rows(results, is_val, prc_results,
                                            num_images, context, mmd_results))
                logger.info(f"Results saved to {csv_path}")

    if is_enabled():
        dist.destroy_process_group()


def main_gen_only(args):
    """Generate images across cfg/ema grid and save to disk (no evaluation)."""
    from utils.builders import create_generation_model, create_tokenizer
    from utils.checkpoint_util import ckpt_resume
    from utils.setup_util import setup
    from utils.data_util import save_image, to_uint8_numpy
    from utils.sampling_util import generate_images

    wandb_logger = setup(args)
    rank = get_global_rank()
    world_size = get_world_size()

    tokenizer = create_tokenizer(args)
    model, ema_model = create_generation_model(args)
    ckpt_resume(args, model, optimizer=None, model_ema=ema_model)

    if not args.disable_vis:
        visualize(args, model, ema_model, args.current_step, tokenizer=tokenizer)
        if args.vis_only:
            exit()

    device = torch.device("cuda")
    cfg_list = sorted(args.cfg_list)
    ema_labels = args.eval_ema_labels if args.eval_ema_labels else ["online"]
    n_total = len(cfg_list) * len(ema_labels)

    logger.info(f"Gen-only grid: {len(cfg_list)} cfgs x {len(ema_labels)} emas "
                f"= {n_total} combos, {args.num_images} images each")
    logger.info(f"  cfgs={cfg_list}  emas={ema_labels}")

    done = 0
    total_start = time.perf_counter()
    for ema_label in ema_labels:
        for cfg_val in cfg_list:
            done += 1
            logger.info(f"[{done}/{n_total}] ema={ema_label} cfg={cfg_val} "
                        f"n={args.num_images} ...")

            start_idx, end_idx = get_start_end_indices(args.num_images, world_size, rank)
            local_n = end_idx - start_idx
            bsz = min(args.eval_bsz, local_n)
            rank_classes = _prepare_eval_classes(args, args.num_images, start_idx, end_idx)

            eval_dir = os.path.join(
                args.log_dir, "gen_images",
                f"ema={ema_label}-cfg={cfg_val}-steps={args.num_sampling_steps}-"
                f"interval_min={args.interval_min}-interval_max={args.interval_max}",
            )
            if rank == 0:
                os.makedirs(eval_dir, exist_ok=True)
            if world_size > 1:
                dist.barrier()

            generated = 0
            t0 = time.perf_counter()
            with ema_model.swap(model, label=ema_label):
                while generated < local_n:
                    b = min(bsz, local_n - generated)
                    y = torch.from_numpy(rank_classes[generated:generated + b]).long().to(device)
                    images = generate_images(args, model, labels=y, cfg=cfg_val, tokenizer=tokenizer)
                    imgs_np = to_uint8_numpy(images)
                    for j, img in enumerate(imgs_np):
                        save_image(img, f"{eval_dir}/{start_idx + generated + j:06d}.png")
                    del images, imgs_np
                    generated += b

            gen_time = time.perf_counter() - t0
            torch.cuda.empty_cache()

            elapsed = time.perf_counter() - total_start
            eta = elapsed / done * (n_total - done)
            ips = args.num_images / gen_time if gen_time > 0 else 0
            logger.info(f"    gen={gen_time:.1f}s ({ips:.0f} img/s) "
                        f"saved to {eval_dir}")
            logger.info(f"    elapsed={datetime.timedelta(seconds=int(elapsed))} "
                        f"eta={datetime.timedelta(seconds=int(eta))}")

    logger.info("All generation complete.")
    if is_enabled():
        dist.destroy_process_group()


def main_generate(args):
    """Generate images across cfg/ema grid and evaluate FD/IS/P&R."""
    from utils.builders import create_generation_model, create_tokenizer
    from utils.checkpoint_util import ckpt_resume
    from utils.setup_util import setup

    wandb_logger = setup(args)
    rank = get_global_rank()

    tokenizer = create_tokenizer(args)
    model, ema_model = create_generation_model(args)
    ckpt_resume(args, model, optimizer=None, model_ema=ema_model)

    if not args.disable_vis:
        visualize(args, model, ema_model, args.current_step, tokenizer=tokenizer)
        if args.vis_only:
            exit()

    repr_models = _load_repr_models(args.models, args.img_size,
                                     target_size_overrides=DEFAULT_TARGET_SIZES)

    eval_prc = not args.no_prc
    eval_mmd = getattr(args, "eval_mmd", False)
    prc_models = getattr(args, "prc_models", None) or list(args.models)

    prc_ref_features, prc_names = {}, []
    if eval_prc:
        prc_ref_features, prc_names = _load_prc_refs(
            prc_models, repr_models, args.prc_ref_dir)

    mmd_ref_features, mmd_names = {}, []
    if eval_mmd:
        mmd_ref_features, mmd_names = _load_prc_refs(
            prc_models, repr_models, args.mmd_ref_dir)

    cfg_list = sorted(args.cfg_list)
    ema_labels = args.eval_ema_labels if args.eval_ema_labels else ["online"]
    repr_names = [entry["name"] for entry in repr_models]
    n_total = len(cfg_list) * len(ema_labels)

    logger.info(f"Eval grid: {len(cfg_list)} cfgs x {len(ema_labels)} emas "
                f"x {len(repr_models)} models = {n_total * len(repr_models)} FD evals")
    logger.info(f"  cfgs={cfg_list}  emas={ema_labels}  "
                f"models={repr_names}  images={args.num_images}")

    csv_path = os.path.join(args.log_dir, "final_eval_summary.csv")

    # Load existing results to skip already-computed entries
    existing = set()  # (ema_label, cfg, model, step)
    if rank == 0 and os.path.exists(csv_path):
        import pandas as _pd
        _df = _pd.read_csv(csv_path, on_bad_lines="skip")
        for _, row in _df.iterrows():
            existing.add((str(row["ema_label"]), float(row["cfg"]), str(row["model"]), int(row["step"])))
        logger.info(f"Loaded {len(existing)} cached entries from {csv_path}")
    n_existing = broadcast_scalar(float(len(existing)))

    done = 0
    total_start = time.perf_counter()
    for ema_label in ema_labels:
        for cfg_val in cfg_list:
            done += 1
            if n_existing > 0:
                # Must call broadcast_scalar for ALL models (it's a collective);
                # do NOT use all() which short-circuits and causes deadlock.
                cached_flags = [
                    broadcast_scalar(float(rank == 0 and (ema_label, cfg_val, rn, args.current_step) in existing))
                    for rn in repr_names
                ]
                if all(cached_flags):
                    logger.info(f"[{done}/{n_total}] CACHED ema={ema_label} cfg={cfg_val}")
                    continue

            logger.info(f"[{done}/{n_total}] ema={ema_label} cfg={cfg_val} "
                        f"n={args.num_images} ...")

            results, is_val, gen_time, prc_results, mmd_results = _generate_and_evaluate(
                args, model, ema_model, repr_models,
                cfg=cfg_val, ema_label=ema_label, num_images=args.num_images,
                tokenizer=tokenizer,
                prc_ref_features=prc_ref_features, prc_model_names=prc_names,
                mmd_ref_features=mmd_ref_features, mmd_model_names=mmd_names,
            )

            context = {
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d-%H:%M"),
                "step": args.current_step, "ema_label": ema_label,
                "cfg": round(cfg_val, 2), "interval_min": args.interval_min,
                "interval_max": args.interval_max,
                "num_sampling_steps": args.num_sampling_steps,
            }
            csv_rows = _build_csv_rows(results, is_val, prc_results,
                                        args.num_images, context, mmd_results)
            if rank == 0:
                _append_csv(csv_path, csv_rows)

            elapsed = time.perf_counter() - total_start
            eta = elapsed / done * (n_total - done)
            ips = args.num_images / gen_time if gen_time > 0 else 0
            logger.info(f"    gen={gen_time:.1f}s ({ips:.0f} img/s) "
                        f"elapsed={datetime.timedelta(seconds=int(elapsed))} "
                        f"eta={datetime.timedelta(seconds=int(eta))}")

    logger.info(f"All evaluations complete. CSV: {csv_path}")
    if rank == 0:
        _print_csv_summary(csv_path)
    if is_enabled():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Args & __main__
# ---------------------------------------------------------------------------

def _get_folder_parser():
    """Lightweight parser for standalone folder evaluation."""
    parser = argparse.ArgumentParser(
        description="Compute FD, IS, Precision & Recall from an image folder")
    parser.add_argument("--image_folder", type=str, nargs="+", default=None,
                        help="One or more image folders to evaluate")
    parser.add_argument("--eval_random_train_set", action="store_true",
                        help="Sample random subsets from training set and evaluate")
    parser.add_argument("--train_list", type=str, default="data/train.txt")
    parser.add_argument("--data_root", type=str, default="data/imagenet")
    parser.add_argument("--num_samples", type=int, default=50000)
    parser.add_argument("--num_trials", type=int, default=5)
    parser.add_argument("--models", type=str, nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--no_prc", action="store_true")
    parser.add_argument("--prc_models", type=str, nargs="+", default=None,
                        help="models for P&R (default: same as --models)")
    parser.add_argument("--eval_mmd", action="store_true",
                        help="compute CMMD (MMD with RBF kernel) for all models")
    parser.add_argument("--prc_ref_dir", type=str, default="./data/imagenet-val-prc")
    parser.add_argument("--mmd_ref_dir", type=str, default="./data/imagenet/val",
                        help="reference image dir for CMMD (default: validation set)")
    parser.add_argument("--prc_k", type=int, default=3)
    parser.add_argument("--prc_batch_size", type=int, default=10000)
    parser.add_argument("--output_csv", type=str, nargs="+", default=None,
                        help="One or more output CSV paths (must match --image_folder count)")
    return parser


def get_args_parser():
    """Full parser for model-based evaluation (inherits from main_fd)."""
    from main_fd import get_args_parser as _fd_parser
    parent = _fd_parser()
    parser = argparse.ArgumentParser(
        "Multi-model FD evaluation",
        parents=[parent], add_help=True, conflict_handler="resolve",
    )
    parser.add_argument("--models", type=str, nargs="+", default=DEFAULT_MODELS,
                        dest="models")
    parser.add_argument("--cfg_list", type=float, nargs="+",
                        default=[1.0, 2.0, 3.0, 4.0, 5.0, 6.5, 8.0, 8.5, 10.0, 12.0, 14.0])
    parser.add_argument("--gen_only", action="store_true",
                        help="Generate and save images only, skip evaluation")
    parser.add_argument("--save_eval_images", action="store_true")
    parser.add_argument("--keep_eval_folder", action="store_true")
    parser.add_argument("--no_prc", action="store_true",
                        help="disable Precision & Recall computation")
    parser.add_argument("--eval_mmd", action="store_true",
                        help="compute CMMD (MMD with RBF kernel) for all models")
    parser.add_argument("--mmd_ref_dir", type=str, default="./data/imagenet/val",
                        help="reference image dir for CMMD (default: validation set)")
    parser.add_argument("--prc_models", type=str, nargs="+", default=None,
                        help="models for P&R (default: same as --models)")
    parser.add_argument("--prc_ref_dir", type=str, default="./data/imagenet-val-prc")
    parser.add_argument("--prc_k", type=int, default=3)
    parser.add_argument("--prc_batch_size", type=int, default=10000)
    parser.add_argument("--enable_vis", action="store_false", dest="disable_vis",
                        help="generate visualization grids before evaluation")
    parser.set_defaults(disable_vis=True)
    return parser


if __name__ == "__main__":
    import sys
    if "--eval_random_train_set" in sys.argv:
        main_random_train(_get_folder_parser().parse_args())
    elif "--image_folder" in sys.argv:
        main_folder(_get_folder_parser().parse_args())
    elif "--gen_only" in sys.argv:
        main_gen_only(get_args_parser().parse_args())
    else:
        main_generate(get_args_parser().parse_args())
