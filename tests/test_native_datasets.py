from __future__ import annotations

# Install isolation before project imports.
import deployment_test_guard  # noqa: F401  # isort: skip

import copy
import hashlib
import io
import json
import shutil
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_task_images as builder
import import_task as importer
import native_datasets as datasets
import render_task_definition as renderer
import validate_task_layout as layout
from preview import publication


class NativeDatasetsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "minimal_task"
        shutil.copytree(ROOT / "tests/fixtures/export-minimal", self.root)
        self.policy = json.loads((self.root / "policy.json").read_text())

    def tar(self, mode, members=None):
        body = io.BytesIO()
        with tarfile.open(fileobj=body, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            for name, kind in members or [("workload.bin", tarfile.REGTYPE)]:
                item = tarfile.TarInfo(name)
                item.type = kind
                data = (mode + "\x00original workload").encode()
                item.size = len(data) if kind == tarfile.REGTYPE else 0
                item.linkname = "workload.bin" if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE) else ""
                archive.addfile(item, io.BytesIO(data) if item.size else None)
        path = self.root / "datasets" / f"{mode}.tar"
        path.write_bytes(body.getvalue())
        self.policy["metadata"]["datasets"][mode] = {
            "source": "task", "path": f"datasets/{mode}.tar",
            "media_type": "application/x-tar", "bytes": path.stat().st_size,
            "version": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        return path

    def prepare(self):
        for mode in ("train", "eval"):
            self.tar(mode)
            (self.root / "datasets" / f"{mode}.json").unlink()
        self.save()

    def save(self):
        (self.root / "policy.json").write_text(json.dumps(self.policy))

    def test_original_json_definition_is_unchanged(self):
        definition = renderer.render(self.root, self.root.name)
        self.assertEqual(definition["dataset_extension"], "json")
        self.assertEqual(definition["datasets"], {
            f"{mode}_sha256": self.policy["metadata"]["datasets"][mode]["version"]
            for mode in ("train", "eval")
        })

    def test_tar_export_render_build_and_registry_verify_preserve_exact_bytes(self):
        self.prepare()
        before = {str(p.relative_to(self.root)): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        definition = importer.validate_export(self.root)
        self.assertEqual(definition["dataset_extension"], "tar")
        context = Path(self.temp.name) / "context"
        context.mkdir()
        expected = builder.stage_datasets(self.root, context, definition)
        for mode in ("train", "eval"):
            digest = definition["datasets"][f"{mode}_sha256"]
            member = f"datasets/{mode}/sha256/{digest[7:]}.tar"
            self.assertEqual(expected[member], (self.root / f"datasets/{mode}.tar").read_bytes())
            uri = publication.task_dataset_uri(self.root.name, mode, digest, definitions={self.root.name: definition})
            self.assertTrue(uri.endswith(f"/{digest[7:]}.tar"))
        # Real verification entrypoint, fake registry transport only; inspect exact
        # expected layer members rather than claiming any live publication.
        image = {"repository": "docker.io/test/fcs2-task-minimal_task-datasets", "tag": "v1", "digest": "sha256:" + "a" * 64}
        entry = {"datasets": definition["datasets"], "dataset_image": image}
        with mock.patch.object(importer.resolver, "task_images", return_value=[(image, True)]), mock.patch.object(importer.resolver, "resolve_digest", return_value=image["digest"]) as resolve:
            importer.verify_images(entry, self.root, None)
        self.assertEqual(resolve.call_args_list[0].kwargs["expected_members"], expected)
        self.assertEqual(before, {str(p.relative_to(self.root)): p.read_bytes() for p in self.root.rglob("*") if p.is_file()})

    def test_tar_declared_path_media_size_source_and_pair_are_required(self):
        self.prepare()
        mutations = [
            ("path", "../datasets/train.tar"), ("path", "datasets/eval.tar"),
            ("path", "datasets/train.tar.gz"), ("path", "/datasets/train.tar"),
            ("media_type", "application/json"), ("media_type", None),
            ("bytes", True), ("bytes", 0), ("bytes", 1), ("source", "external"),
        ]
        for key, value in mutations:
            policy = copy.deepcopy(self.policy)
            policy["metadata"]["datasets"]["train"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                datasets.dataset_hashes(self.root, policy)
        policy = copy.deepcopy(self.policy)
        policy["metadata"]["datasets"]["eval"] = {"version": "sha256:" + "b" * 64}
        with self.assertRaisesRegex(ValueError, "formats differ"):
            datasets.dataset_extension(policy)

    def test_changed_body_or_declared_version_is_rejected(self):
        self.prepare()
        policy = copy.deepcopy(self.policy)
        policy["metadata"]["datasets"]["train"]["version"] = "sha256:" + "0" * 64
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            datasets.dataset_hashes(self.root, policy)
        path = self.root / "datasets/train.tar"
        body = bytearray(path.read_bytes()); body[512] ^= 1; path.write_bytes(body)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            datasets.dataset_hashes(self.root, self.policy)

    def test_symlink_and_unknown_mode_extension_are_rejected(self):
        self.prepare()
        path = self.root / "datasets/train.tar"
        path.unlink(); path.symlink_to("eval.tar")
        with self.assertRaisesRegex(ValueError, "regular dataset"):
            datasets.dataset_hashes(self.root, self.policy)
        for mode, extension in [("../train", "tar"), ("train", "../tar"), ("test", "json")]:
            with self.subTest(mode=mode, extension=extension), self.assertRaises(ValueError):
                datasets.dataset_path(self.root, mode, extension)

    def test_tar_links_paths_duplicates_and_special_members_are_rejected(self):
        cases = [
            [("../escape", tarfile.REGTYPE)], [("/absolute", tarfile.REGTYPE)],
            [("a\\b", tarfile.REGTYPE)], [("a/./b", tarfile.REGTYPE)],
            [("data", tarfile.REGTYPE), ("data", tarfile.REGTYPE)],
            [("link", tarfile.SYMTYPE)], [("link", tarfile.LNKTYPE)],
            [("fifo", tarfile.FIFOTYPE)], [("dev", tarfile.CHRTYPE)],
        ]
        for members in cases:
            with self.subTest(members=members), self.assertRaises(ValueError):
                datasets.check_tar_dataset(self.tar("train", members))

    def test_tar_truncated_compressed_empty_and_nonarchive_are_rejected(self):
        path = self.tar("train")
        good = path.read_bytes()
        import gzip
        for body in [good[:520], gzip.compress(good), bytes(10240), b"not a tar"]:
            path.write_bytes(body)
            with self.subTest(length=len(body)), self.assertRaises(ValueError):
                datasets.check_tar_dataset(path)

    def test_layout_accepts_tar_with_complete_lock_but_keeps_identity_and_science_gates(self):
        task = "batch_local_counting"
        source = renderer.problem_directory(layout.DEFAULT_PROBLEMS_ROOT, task)
        self.root = Path(self.temp.name) / task
        shutil.copytree(source, self.root)
        self.policy = json.loads((self.root / "policy.json").read_text())
        self.prepare()
        lock_path = self.root / "upstream.lock.json"
        lock = json.loads(lock_path.read_text())
        def lock_files():
            lock["files"] = [{"path": p.relative_to(self.root).as_posix(), "bytes": p.stat().st_size, "sha256": layout.digest(p)} for p in sorted(self.root.rglob("*")) if p.is_file() and p != lock_path]
            lock["file_count"] = len(lock["files"])
            lock_path.write_text(json.dumps(lock))
        lock_files()
        self.assertEqual(layout.validate_task(self.root), "patch-workspace")
        self.policy["quota"]["max_train_calls"] += 1; self.save(); lock_files()
        with self.assertRaisesRegex(layout.LayoutFailure, "task-authored"):
            layout.validate_task(self.root)

    def test_builder_rejects_changed_body_after_render(self):
        self.prepare()
        definition = renderer.render(self.root, self.root.name)
        (self.root / "datasets/train.tar").write_bytes(b"changed")
        context = Path(self.temp.name) / "context"; context.mkdir()
        with self.assertRaisesRegex(builder.BuildFailure, "dataset selector changed"):
            builder.stage_datasets(self.root, context, definition)


if __name__ == "__main__":
    unittest.main()
