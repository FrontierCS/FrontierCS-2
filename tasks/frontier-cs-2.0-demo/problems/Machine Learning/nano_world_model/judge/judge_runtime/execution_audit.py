"""Observe the real upstream path without replacing any of its computations."""

import json
from pathlib import Path
import sys

ENTRYPOINT = "src/main.py"
PIPELINE = {
    "entrypoint": (ENTRYPOINT, "main"),
    "factory": ("src/experiments/__init__.py", "build_experiment"),
    "dispatch": ("src/experiments/base.py", "BaseExperiment.exec"),
    "evaluation": ("src/experiments/train_experiment.py", "TrainExperiment.evaluate"),
    "checkpoint": ("src/experiments/train_experiment.py", "TrainExperiment._load_checkpoint"),
    "dataset": ("src/wm_datasets/world_model_dataset.py", "WorldModelDataset.__getitem__"),
    "frames": ("src/wm_datasets/data_source/dino_wm/dino_world_model.py",
               "DinoWorldModelDataSource.load_visual_frames"),
    "encode": ("src/latent_codecs/sd_vae.py", "SDVAELatentCodec.encode"),
    "images": ("src/experiments/train_experiment.py", "NanoWMTrainingModule.log_images"),
    "latents": ("src/experiments/train_experiment.py", "NanoWMTrainingModule.log_latents"),
    "decode": ("src/latent_codecs/sd_vae.py", "SDVAELatentCodec.decode"),
    "writer": ("src/utils/metrics.py", "Evaluator.evaluate_all"),
}
REFERENCE_SAMPLER = {
    "sampler": ("src/diffusion/df_sample.py", "dfot_sample"),
    "loop": ("src/diffusion/gaussian_diffusion.py", "GaussianDiffusion.dfot_sample_loop"),
    "transition": ("src/diffusion/gaussian_diffusion.py", "GaussianDiffusion.dfot_ddim_sample"),
    "model": ("src/models/nanowm.py", "NanoWM.forward"),
}


def required_steps(isolated):
    # Candidate sampling is separately audited across the sealed process bridge.
    return PIPELINE if isolated else {**PIPELINE, **REFERENCE_SAMPLER}


def install(entrypoint, destination, *, isolated):
    entrypoint = Path(entrypoint).resolve()
    if entrypoint.parts[-2:] != ("src", "main.py") or not entrypoint.is_file():
        raise ValueError("expected the retained src/main.py entrypoint")
    repo = entrypoint.parent.parent
    pending = {(str(repo / path), symbol): name
               for name, (path, symbol) in required_steps(isolated).items()}
    observed = set()

    def profile(frame, event, arg):
        if event != "call":
            return
        code = frame.f_code
        name = pending.pop((code.co_filename, code.co_qualname), None)
        if name is not None:
            observed.add(name)
        if not pending:
            sys.setprofile(None)
            Path(destination).write_text(json.dumps({
                "entrypoint": ENTRYPOINT, "isolated": isolated,
                "observed": sorted(observed),
            }), encoding="utf-8")

    sys.setprofile(profile)


def validate(path, *, isolated):
    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        valid = (record["entrypoint"] == ENTRYPOINT
                 and record["isolated"] is isolated
                 and set(record["observed"]) == set(required_steps(isolated)))
    except (OSError, ValueError, KeyError, TypeError):
        valid = False
    if not valid:
        raise RuntimeError("native src/main.py execution path was not completed")
