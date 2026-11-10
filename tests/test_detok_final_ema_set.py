"""detok_latent_denoising: the graded EMA evaluation uses its own held-out images.

The ordinary and EMA evaluations of one process used to reconstruct the same 64 images in the
same order, so per-input state kept by the first evaluation could serve the second. In the graded
(final) suite the EMA evaluation now reconstructs a second, disjoint held-out set; development
evaluation is unchanged.
"""

import deployment_test_guard  # noqa: F401  # isort: skip

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

from test_detok_warmup import JUDGE, load_evaluator, sandbox_result

PRIVATE = JUDGE / "judge_data/private"
FINAL = [PRIVATE / "final" / str(i) for i in range(3)]
DEVELOPMENT = [PRIVATE / "development" / str(i) for i in range(3)]
LICENSE = "https://creativecommons.org/licenses/by/2.0/"


def names(path: Path) -> list[str]:
    return [line.split()[0] for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FinalEmaSetData(unittest.TestCase):
    def test_each_final_workload_has_a_disjoint_ema_set(self):
        seen_ids, seen_content = {}, {}
        for workload in DEVELOPMENT + FINAL:
            for kind in ("train", "val", "val_ema"):
                listing = workload / f"{kind}.txt"
                if not listing.exists():
                    self.assertEqual(kind, "val_ema")
                    self.assertIn(workload, DEVELOPMENT)
                    self.assertFalse((workload / kind).exists())
                    continue
                self.assertIn(workload, FINAL if kind == "val_ema" else DEVELOPMENT + FINAL)
                listed = names(listing)
                self.assertEqual(len(listed), 64)
                self.assertTrue(all(line.endswith(".jpg 0") for line in listing.read_text().splitlines()))
                self.assertEqual(sorted(listed), sorted(p.name for p in (workload / kind).iterdir()))
                for name in listed:
                    where = f"{workload.parent.name}/{workload.name}/{kind}"
                    self.assertNotIn(name, seen_ids, (where, seen_ids.get(name)))
                    seen_ids[name] = where
                    digest = sha256(workload / kind / name)
                    self.assertNotIn(digest, seen_content, (where, seen_content.get(digest)))
                    seen_content[digest] = where
        self.assertEqual(len(seen_ids), 768 + 3 * 64)

    def test_ema_images_are_attributed_and_manifested(self):
        manifest = json.loads((PRIVATE / "assets-manifest.json").read_text())
        self.assertEqual(manifest["attribution_sha256"], sha256(PRIVATE / "attribution.json"))
        attribution = {x["image_id"]: x for x in json.loads((PRIVATE / "attribution.json").read_text())["images"]}
        self.assertEqual(len(attribution), 768 + 3 * 64)
        for i, workload in enumerate(FINAL):
            for rel in (f"final/{i}/val_ema.txt",
                        *(f"final/{i}/val_ema/{name}" for name in names(workload / "val_ema.txt"))):
                self.assertEqual(manifest["files"][rel]["sha256"], sha256(PRIVATE / rel), rel)
                self.assertEqual(manifest["files"][rel]["bytes"], (PRIVATE / rel).stat().st_size, rel)
            for name in names(workload / "val_ema.txt"):
                entry = attribution[name.removesuffix(".jpg")]
                self.assertEqual((entry["suite"], entry["workload"], entry["role"]), ("final", str(i), "ema_reconstruction"))
                self.assertEqual(entry["license"], LICENSE)
                self.assertEqual(entry["license_evidence"]["license"], LICENSE)
                self.assertEqual(entry["bytes"], (workload / "val_ema" / name).stat().st_size)
        files = {str(p.relative_to(PRIVATE)) for p in PRIVATE.rglob("*") if p.is_file()}
        self.assertEqual(files - {"assets-manifest.json", "attribution.json"}, set(manifest["files"]))


def has(*modules):
    return all(importlib.util.find_spec(m) for m in modules)


@unittest.skipUnless(has("numpy", "PIL"), "decoding the private photographs needs numpy and Pillow")
class FinalEmaSetPixels(unittest.TestCase):
    def test_decoded_inputs_never_repeat_across_the_two_evaluations(self):
        """The audited per-batch key (shape and three pixels) and an exact per-image key never hit."""
        import numpy as np
        evaluator = load_evaluator()

        def batch_keys(targets):
            keys = set()
            for i in range(0, len(targets), 4):
                x = np.stack(targets[i:i + 4]).transpose(0, 3, 1, 2).astype(np.float32) / 127.5 - 1.0
                h, w = x.shape[2:]
                keys.add((x.shape, float(x[0, 0, 0, 0]), float(x[0, 1, h // 2, w // 2]), float(x[-1, -1, -1, -1])))
            return keys

        for workload in FINAL:
            ordinary = evaluator._targets(workload / "val", workload / "val.txt")
            ema = evaluator._targets(workload / "val_ema", workload / "val_ema.txt")
            self.assertEqual((len(ordinary), len(ema)), (64, 64))
            self.assertFalse(batch_keys(ordinary) & batch_keys(ema), workload)
            exact = {hashlib.sha256(t.tobytes()).hexdigest() for t in ordinary}
            self.assertFalse(exact & {hashlib.sha256(t.tobytes()).hexdigest() for t in ema}, workload)


# A stand-in for the retained tree: the same module layout and call shapes as
# main_reconstruction.py / utils.train_utils.evaluate_tokenizer / utils.loader.ListDataset,
# torch.utils.data replaced by a minimal loader, and a candidate model whose module-level cache is
# keyed on input content and filled by whichever evaluation runs first.
FAKE_TREE = {
    "torch/__init__.py": "",
    "torch/utils/__init__.py": "",
    "torch/utils/data.py": """
        class DistributedSampler:
            def __init__(self, dataset, num_replicas, rank, shuffle):
                self.num_replicas, self.rank, self.shuffle = num_replicas, rank, shuffle
                self.indices = list(range(len(dataset)))
        class DataLoader:
            def __init__(self, dataset, sampler, batch_size, num_workers, pin_memory, drop_last):
                self.dataset, self.sampler, self.batch_size = dataset, sampler, batch_size
                self.num_workers, self.pin_memory, self.drop_last = num_workers, pin_memory, drop_last
            def __len__(self):
                return (len(self.sampler.indices) + self.batch_size - 1) // self.batch_size
            def __iter__(self):
                idx = self.sampler.indices
                for i in range(0, len(idx), self.batch_size):
                    yield [self.dataset[j] for j in idx[i:i + self.batch_size]]
    """,
    "utils/__init__.py": "",
    "utils/loader.py": """
        import os
        class ListDataset:
            def __init__(self, data_root, data_list, transform=None, loader_name="npz_loader", return_path=False,
                         return_label=True, return_index=False, should_flip=True, class_of_interest=None):
                self.data_root, self.transform = data_root, transform
                assert loader_name == "img_loader" and return_index and not return_label and not should_flip
                with open(data_list) as f:
                    self.samples = [line.split()[0] for line in f if line.strip()]
            def __len__(self):
                return len(self.samples)
            def __getitem__(self, index):
                with open(os.path.join(self.data_root, self.samples[index]), "rb") as f:
                    return {"index": index, "img": self.transform(f.read())}
    """,
    "utils/train_utils.py": """
        import json, os
        def evaluate_tokenizer(args, model, ema_model, data_loader_val, epoch=0, wandb_logger=None, use_ema=True):
            before = dict(model.stats)
            for batch in data_loader_val:
                model.reconstruct([item["img"] for item in batch])
            record = {"use_ema": use_ema, "images": len(data_loader_val.dataset),
                      "hits": model.stats["hits"] - before["hits"], "misses": model.stats["misses"] - before["misses"],
                      "loader_id": id(data_loader_val), "batch_size": data_loader_val.batch_size,
                      "num_workers": data_loader_val.num_workers, "pin_memory": data_loader_val.pin_memory,
                      "shuffle": data_loader_val.sampler.shuffle, "transform_id": id(data_loader_val.dataset.transform)}
            with open(os.environ["FAKE_DETOK_RECORD"], "a") as f:
                f.write(json.dumps(record) + "\\n")
    """,
    "models/__init__.py": "",
    "models/detok.py": """
        import hashlib
        _CACHE = {}
        class DeTok:
            def __init__(self):
                self.stats = {"hits": 0, "misses": 0}
            def reconstruct(self, images):
                for image in images:
                    key = hashlib.sha256(image).hexdigest()
                    if key in _CACHE:
                        self.stats["hits"] += 1
                    else:
                        _CACHE[key] = image
                        self.stats["misses"] += 1
    """,
    "main_reconstruction.py": """
        import argparse, json, os, sys
        import torch.utils.data
        from models.detok import DeTok
        from utils.loader import ListDataset
        from utils.train_utils import evaluate_tokenizer
        parser = argparse.ArgumentParser()
        parser.add_argument("--data_path")
        parser.add_argument("--eval_bsz", type=int)
        args = parser.parse_args()
        dataset = ListDataset(args.data_path.replace("train", "val"), data_list="data/val.txt",
                              transform=lambda raw: raw, loader_name="img_loader",
                              return_label=False, return_index=True, should_flip=False)
        loader = torch.utils.data.DataLoader(
            dataset, sampler=torch.utils.data.DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=False),
            batch_size=args.eval_bsz, num_workers=4, pin_memory=True, drop_last=False)
        with open(os.environ["FAKE_DETOK_RECORD"], "a") as f:
            f.write(json.dumps({"sys_path": sys.path, "argv": sys.argv, "loader_id": id(loader),
                                "transform_id": id(dataset.transform)}) + "\\n")
        model = DeTok()
        for use_ema in [False, True]:
            evaluate_tokenizer(args, model, None, loader, 1, None, use_ema)
        sys.exit(0)
    """,
}


class FinalEmaSetProcess(unittest.TestCase):
    def run_process(self, workload: Path, launcher: bool) -> list[dict]:
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        tree = Path(scratch.name) / "repo"
        for rel, source in FAKE_TREE.items():
            (tree / rel).parent.mkdir(parents=True, exist_ok=True)
            (tree / rel).write_text(textwrap.dedent(source))
        (tree / "data").mkdir()
        (tree / "data/val.txt").write_text((workload / "val.txt").read_text())
        record = Path(scratch.name) / "record.jsonl"
        ema = ["--ema-set", str(workload / "val_ema"), str(workload / "val_ema.txt")]
        command = [sys.executable, str(JUDGE / "method_runner.py"), str(Path(scratch.name) / "result.tar"),
                   str(tree / "output"), *(ema if launcher else []), str(tree / "main_reconstruction.py"),
                   "--data_path", str(workload / "train"), "--eval_bsz", "4"]
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env.update(FAKE_DETOK_RECORD=str(record), PYTHONDONTWRITEBYTECODE="1")
        proc = subprocess.run(command, cwd=tree, env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return [json.loads(line) for line in record.read_text().splitlines()], tree

    def test_content_keyed_cache_gains_nothing_in_the_graded_suite(self):
        for workload in FINAL:
            (start, ordinary, ema), tree = self.run_process(workload, launcher=True)
            self.assertEqual((ordinary["use_ema"], ordinary["images"], ordinary["hits"], ordinary["misses"]),
                             (False, 64, 0, 64))
            self.assertEqual((ema["use_ema"], ema["images"], ema["hits"], ema["misses"]), (True, 64, 0, 64))
            # The ordinary evaluation keeps the retained loader; the EMA loader is built like it.
            self.assertEqual(ordinary["loader_id"], start["loader_id"])
            self.assertNotEqual(ema["loader_id"], start["loader_id"])
            for key in ("batch_size", "num_workers", "pin_memory", "shuffle"):
                self.assertEqual(ema[key], ordinary[key], key)
            self.assertEqual(ema["transform_id"], start["transform_id"])
            # Same import path and argv as running the entrypoint directly; no judge module is importable.
            self.assertEqual(Path(start["sys_path"][0]), tree)
            self.assertNotIn(str(JUDGE), [str(Path(p).resolve()) for p in start["sys_path"] if p])
            self.assertEqual(Path(start["argv"][0]), tree / "main_reconstruction.py")
            self.assertEqual(start["argv"][1:], ["--data_path", str(workload / "train"), "--eval_bsz", "4"])

    def test_without_the_ema_set_every_ema_input_is_a_cache_hit(self):
        """Control: the same candidate on the development wiring (one image set for both)."""
        (_, ordinary, ema), _ = self.run_process(FINAL[0], launcher=False)
        self.assertEqual((ordinary["hits"], ordinary["misses"]), (0, 64))
        self.assertEqual((ema["hits"], ema["misses"]), (64, 0))


class RetainedCallShape(unittest.TestCase):
    def test_launcher_matches_the_retained_evaluation_call(self):
        """The launcher reads use_ema as the 7th positional argument, as the retained entrypoint passes it."""
        import ast
        repo = JUDGE.parent / "repo"
        tree = ast.parse((repo / "utils/train_utils.py").read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "evaluate_tokenizer")
        self.assertEqual([a.arg for a in function.args.args],
                         ["args", "model", "ema_model", "data_loader_val", "epoch", "wandb_logger", "use_ema"])
        self.assertEqual(ast.literal_eval(function.args.defaults[-1]), True)
        calls = [n for n in ast.walk(ast.parse((repo / "main_reconstruction.py").read_text()))
                 if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "evaluate_tokenizer"]
        self.assertTrue(calls)
        for call in calls:
            self.assertEqual((len(call.args), call.keywords), (7, []))
            self.assertEqual(call.args[6].id, "use_ema")
        loader = (repo / "utils/builders.py").read_text()
        self.assertIn('data_list="data/val.txt"', loader)


class FinalEmaSetCommand(unittest.TestCase):
    def capture(self, role: str, with_ema_set: bool = True) -> list[str]:
        evaluator = load_evaluator()
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        root = Path(scratch.name)
        tree, workload, models = root / "tree", root / "workload", root / "models"
        tree.mkdir()
        (tree / "main_reconstruction.py").write_text("# intercepted\n")
        for name in ("train", "val") + (("val_ema",) if with_ema_set else ()):
            (workload / name).mkdir(parents=True)
            (workload / f"{name}.txt").write_text("photo.jpg 0\n")
        (workload / "fid_stats.npz").write_bytes(b"fixture")
        (workload / "seed.txt").write_text("11\n")
        for name in ("detok/detok-BB-gamm3.0-m0.7.pth", "lpips/vgg.pth", "torch/hub/checkpoints/vgg16-397923af.pth",
                     "torch/hub/checkpoints/convnext_small-0c510722.pth",
                     "torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth"):
            (models / name).parent.mkdir(parents=True, exist_ok=True)
            (models / name).write_bytes(b"fixture")
        calls = []

        def run(command, **kwargs):
            calls.append((command, kwargs))
            return sandbox_result(returncode=1)

        with mock.patch.object(evaluator, "OFFLINE_MODEL_ROOT", models), \
                mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0", "FRONTIER_SUBMISSION_ROLE": role}), \
                mock.patch.object(evaluator.isolate, "run", side_effect=run):
            try:
                evaluator._run(tree, workload, root / "output")
            except RuntimeError as error:
                if not calls:
                    raise
                self.assertIn("method process failed", str(error))
        command, kwargs = calls[0]
        # The instance gets this workload only (copied), the runner, and the model assets in place.
        self.assertEqual(kwargs["inputs"], {"method_runner.py": evaluator.METHOD_RUNNER, "workload": workload})
        self.assertEqual(kwargs["tree"], tree)
        self.assertEqual(kwargs["read_paths"], [str(models)])
        self.assertEqual(kwargs["devices"], ["nvidia"])
        self.assertFalse(any(k.startswith("FRONTIER_") for k in kwargs["env"]))
        return command, tree, workload, evaluator

    def test_development_runs_the_entrypoint_directly(self):
        command, tree, workload, evaluator = self.capture("development", with_ema_set=False)
        self.assertEqual(command[:6], [os.path.realpath(sys.executable), "{in}/method_runner.py", "{out}/result.tar",
                                       "{work}/output", "{work}/main_reconstruction.py", "--epochs"])
        self.assertNotIn("--ema-set", command)

    def test_final_runs_the_ema_evaluation_on_its_own_set(self):
        development, *_ = self.capture("development", with_ema_set=False)
        final, tree, workload, evaluator = self.capture("final")
        self.assertEqual(final[3:8], ["{work}/output", "--ema-set", "{in}/workload/val_ema",
                                      "{in}/workload/val_ema.txt", "{work}/main_reconstruction.py"])
        # Same arguments after the entrypoint (host paths differ only by scratch directory).
        self.assertEqual([x for x in final[8:] if not x.startswith("/")],
                         [x for x in development[5:] if not x.startswith("/")])

    def test_final_requires_the_ema_set(self):
        with self.assertRaisesRegex(RuntimeError, "private workload is incomplete"):
            self.capture("final", with_ema_set=False)


if __name__ == "__main__":
    unittest.main()
