"""
Unified experiment runner. Loads config YAML, builds mixing + encoder,
calls engine.train_and_evaluate, saves standardized .pt output.

Usage:
    python run.py --config configs/2d.yaml --run spiral --seed 1337
    python run.py --config configs/ablation.yaml --run spiral_lejepa --seed 1337
    python run.py --config configs/scaling.yaml --N 16 --seed 0
    python run.py --config configs/grid.yaml --lamb 0.01 --rho 0.9 --seed 0
"""

import argparse, os, json, yaml
import torch
import numpy as np

from lejepa_id.mixing import MIXINGS_2D, make_coupling_mixing
from lejepa_id.models import make_mlp_encoder, make_matched_encoder
from lejepa_id.data import sample_latents, ou_augment
from lejepa_id.metrics import compute_all_metrics, compute_recovery_metrics
from lejepa_id.engine import train_and_evaluate


def _jsonify(obj):
    """Convert numpy types to Python natives for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_jsonify(v) for v in obj]
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def build_mixing(mixing_name, N, n_layers=4, seed=1337, device="cuda"):
    """Build mixing function from name."""
    if mixing_name in MIXINGS_2D:
        return MIXINGS_2D[mixing_name]
    elif mixing_name in ("nvp", "coupling"):
        return make_coupling_mixing(N, n_layers=n_layers, seed=seed, device=device)
    else:
        raise ValueError(f"Unknown mixing: {mixing_name}")


def build_encoder(encoder_type, N, hidden=256, n_layers=4, seed=42, device="cuda"):
    """Build encoder from type string."""
    if encoder_type == "mlp":
        return make_mlp_encoder(N, hidden=hidden, device=device)
    elif encoder_type == "matched":
        return make_matched_encoder(N, n_layers=n_layers, seed=seed, device=device)
    else:
        raise ValueError(f"Unknown encoder: {encoder_type}")


def resolve_run_spec(cfg, args):
    """Resolve the full run specification from config + CLI args.
    Returns a dict with all parameters needed for one training run."""
    experiment = cfg["experiment"]

    # Start with config-level defaults
    spec = {
        "experiment": experiment,
        "N": cfg.get("N", 2),
        "source_dist": cfg.get("source_dist", "gaussian"),
        "num_eval": cfg.get("num_eval", 10000),
        "steps": cfg.get("steps", 10000),
        "lr": cfg.get("lr", 3e-3),
        "batch_size": cfg.get("batch_size", 256),
        "rho": cfg.get("rho", 0.95),
        "lamb": cfg.get("lamb"),
        "sigma": cfg.get("sigma", 1.0),
        "source_alpha": cfg.get("source_alpha"),   # NEW
        "log_every": cfg.get("log_every", 100),
        "encoder": cfg.get("encoder", "mlp"),
        "hidden": cfg.get("hidden", 256),
        "n_layers": cfg.get("n_layers", 4),
        "mixing": cfg.get("mixing", "spiral"),
        "mode": cfg.get("mode", "lejepa"),
        "seed": args.seed,
    }

    if experiment in ("2d", "ablation"):
        # Look up run-specific overrides
        run_name = args.run
        run_cfg = cfg["runs"][run_name]
        spec["run_name"] = run_name
        for k in ("mixing", "encoder", "hidden", "n_layers", "mode", "lamb", "sigma"):
            if k in run_cfg:
                spec[k] = run_cfg[k]

    elif experiment == "scaling":
        N = args.N
        spec["N"] = N
        spec["mixing"] = "coupling"
        if args.mode is not None:
            spec["mode"] = args.mode
        # Mode-specific lamb (whiten uses different default)
        if spec["mode"] == "whiten":
            spec["lamb"] = cfg.get("lamb_whiten", 0.5)
        spec["run_name"] = f"N={N}_{spec['mode']}"

    elif experiment == "grid":
        spec["lamb"] = args.lamb
        spec["rho"] = args.rho
        spec["run_name"] = f"lamb={args.lamb:.0e}_rho={args.rho:.2f}"

    elif experiment == "gennorm":
        if args.alpha is None:
            raise ValueError("--alpha required for gennorm experiment")
        spec["source_dist"] = "gennorm"
        spec["source_alpha"] = args.alpha
        run_name = args.run
        run_cfg = cfg["runs"][run_name]
        for k in ("mixing", "encoder", "hidden", "n_layers", "mode", "lamb", "sigma"):
            if k in run_cfg:
                spec[k] = run_cfg[k]
        spec["run_name"] = f"{run_name}_alpha={args.alpha:g}"

    return spec


def run_single(spec, device):
    """Execute one training run from a resolved spec. Returns result dict."""
    N = spec["N"]
    seed = spec["seed"]

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build mixing
    mix_seed = seed
    n_layers = spec.get("n_layers", 4)
    mix_fn = build_mixing(spec["mixing"], N, n_layers=n_layers,
                          seed=mix_seed, device=device)

    # Build encoder (different seed from mixing)
    enc_seed = seed + 77777
    encoder = build_encoder(spec["encoder"], N, hidden=spec.get("hidden", 256),
                            n_layers=n_layers, seed=enc_seed, device=device)

    # Fixed eval set
    z_eval = sample_latents(spec["num_eval"], N, dist=spec["source_dist"],
                            device=device, alpha=spec.get("source_alpha"))

    # Train
    encoder, log = train_and_evaluate(
        encoder, mix_fn,
        N=N, rho=spec["rho"], lamb=spec["lamb"], mode=spec["mode"],
        source_dist=spec["source_dist"],
        source_alpha=spec.get("source_alpha"),
        sigma=spec["sigma"],
        steps=spec["steps"], batch_size=spec["batch_size"], lr=spec["lr"],
        z_eval=z_eval, log_every=spec["log_every"], device=device,
    )
    
    # Final metrics from 10k eval set
    encoder.eval()
    with torch.no_grad():
        x_eval = mix_fn(z_eval)
        h_eval = encoder(x_eval)
        # z_prime = ou_augment(z_eval, spec["rho"], n_views=1).squeeze(0)
        z_prime = ou_augment(
            z_eval, spec["rho"], n_views=1,
            dist=spec["source_dist"],
            alpha=spec.get("source_alpha")
        ).squeeze(0)
        h_prime = encoder(mix_fn(z_prime))

    final_metrics = compute_all_metrics(
        z_eval, mix_fn(z_eval), h_eval, h_prime, spec["rho"], N,
    )

    # Fixed-grid evaluation (cross-distribution comparable, only for 2D)
    if N == 2:
        with torch.no_grad():
            g = torch.linspace(-3.0, 3.0, 100, device=device)
            z_grid = torch.stack(torch.meshgrid(g, g, indexing='ij'), dim=-1).reshape(-1, N)
            h_grid = encoder(mix_fn(z_grid))
        final_metrics.update(compute_recovery_metrics(z_grid, h_grid, N, suffix="_grid"))

    # Large scatter data for plotting (only for 2d/ablation)
    if spec["experiment"] in ("2d", "ablation"):
        with torch.no_grad():
            z_plot = sample_latents(100000, N, dist=spec["source_dist"],
                                    device=device, alpha=spec.get("source_alpha"))
            x_plot = mix_fn(z_plot)
            h_chunks = []
            for i in range(0, len(z_plot), 10000):
                h_chunks.append(encoder(x_plot[i:i+10000]))
            h_plot = torch.cat(h_chunks, dim=0)
        z_np = z_plot.cpu().numpy()
        x_np = x_plot.cpu().numpy()
        h_np = h_plot.cpu().numpy()
    else:
        z_np, x_np, h_np = None, None, None
    
    # JSON-serializable result (scalars + training curves)
    result = {
        # Identity
        "experiment": spec["experiment"],
        "run_name": spec["run_name"],
        "mixing": spec["mixing"],
        "encoder": spec["encoder"],
        "mode": spec["mode"],
        "source_dist": spec["source_dist"],
        "source_alpha": spec.get("source_alpha"),
        "seed": seed,
        "N": N,
        # Hyperparameters
        "lamb": spec["lamb"],
        "rho": spec["rho"],
        "lr": spec["lr"],
        "steps": spec["steps"],
        "batch_size": spec["batch_size"],
        "n_layers": n_layers,
        "hidden": spec.get("hidden", None),
        # Final metrics
        **final_metrics,
        "final_align": log["align"][-1],
        "final_sigreg": log["sigreg"][-1],
        "final_whiten": log["whiten"][-1],
        "final_loss": log["total"][-1],
        # Training curves
        "log": log,
    }

    # Heavy data (arrays + model) — only saved as .pt for 2d/ablation
    arrays = {
        "z": z_np, "x": x_np, "h": h_np,
        "model_state_dict": encoder.state_dict(),
    }

    return result, arrays


def save_result(result, arrays, out_dir, fname_base, save_pt=False):
    """Save JSON always; .pt with arrays/model only when requested."""
    # JSON
    json_path = os.path.join(out_dir, fname_base + ".json")
    with open(json_path, "w") as f:
        json.dump(_jsonify(result), f, indent=2)
    print(f"Saved {fname_base}.json")

    # .pt (arrays + model) for 2d/ablation scatter plots
    if save_pt:
        pt_path = os.path.join(out_dir, fname_base + ".pt")
        torch.save({**result, **arrays}, pt_path)
        print(f"Saved {fname_base}.pt")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    # Sweep variables (CLI overrides)
    p.add_argument("--run", type=str, default=None, help="Run name (2d/ablation)")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--N", type=int, default=None, help="Latent dim (scaling)")
    p.add_argument("--lamb", type=float, default=None, help="Lambda (grid)")
    p.add_argument("--rho", type=float, default=None, help="Rho (grid)")
    p.add_argument("--alpha", type=float, default=None, help="Gennorm shape (gennorm)")
    p.add_argument("--mode", type=str, default=None,
               help="Override mode (lejepa/whiten/infonce)")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    spec = resolve_run_spec(cfg, args)
    out_dir = cfg["out"]
    os.makedirs(out_dir, exist_ok=True)

    experiment = cfg["experiment"]
    save_pt = experiment in ("2d", "ablation")

    if experiment == "scaling":
        # For small N, train K encoders, pick best
        K = cfg.get("K", 1)
        # For large N, all converge
        if spec["N"] > 32:
            K = 1
        best_result = None
        best_arrays = None
        best_loss = float("inf")

        for k in range(K):
            spec_k = dict(spec)
            spec_k["seed"] = spec["seed"] + k * 1000
            print(f"\n  Encoder {k+1}/{K} (seed={spec_k['seed']})")
            result, arrays = run_single(spec_k, device)
            print(f"    R²(h->z)={result['r2_hz']:.4f} "
                  f"orth={result['orth_err']:.4f} "
                  f"loss={result['final_loss']:.6f}")

            if result["final_loss"] < best_loss:
                best_loss = result["final_loss"]
                best_result = result
                best_arrays = arrays

        best_result["K"] = K
        best_result["seed"] = spec["seed"]  # original seed
        fname = f"{spec['run_name']}_seed={spec['seed']}"
        save_result(best_result, best_arrays, out_dir, fname, save_pt=False)
        print(f"  R²(h->z)={best_result['r2_hz']:.4f} orth={best_result['orth_err']:.4f}")

    else:
        # Single run
        print(f"\n{'='*50}")
        print(f"{spec['run_name']} seed={spec['seed']}")
        print(f"{'='*50}")
        result, arrays = run_single(spec, device)

        fname = f"{spec['run_name']}_seed={spec['seed']}"
        save_result(result, arrays, out_dir, fname, save_pt=save_pt)
        print(f"  R²(z->h)={result['r2_zh']:.4f} R²(h->z)={result['r2_hz']:.4f} "
              f"orth={result['orth_err']:.4f}")


if __name__ == "__main__":
    main()
