"""Offline signature/accepted-artifact fixtures, never live Driver evidence."""

import deployment_test_guard  # noqa: F401

import io
import json
import sys
import tarfile
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from vllm_task import REPO_ROOT  # noqa: E402

JUDGE = REPO_ROOT / "tasks/problems/MLSys/vllm_llm_serving_optimization/judge"
sys.path.insert(0, str(JUDGE))
from controller_envelope import (
    Context,
    ControllerSigner,
    accepted_manifest,
    accepted_zstd_manifest,
    digest,
    slots_digest,
    verify_terminal,
    verify_terminal_bytes,
)
from terminal_contract import Slot, canonical, terminal


class ControllerEnvelopeTests(unittest.TestCase):
    def setUp(self):
        self.slot = Slot(
            "run",
            "attempt",
            "verification",
            "operation",
            "eval",
            "swe/public",
            "candidate",
            "case",
            "sha256:" + "a" * 64,
            "sha256:" + "b" * 64,
        )
        self.slots = (self.slot,)
        self.context = Context(
            "run",
            "attempt",
            "verification",
            *["sha256:" + c * 64 for c in "cdef"],
            slots_digest(self.slots),
        )
        self.signer = ControllerSigner(
            self.context,
            self.slots,
            Ed25519PrivateKey.generate(),
            maximum_terminal_bytes=16384,
            maximum_manifest_bytes=8192,
        )
        self.record = terminal(
            self.slot, status="runtime_error", cause="child_exit", elapsed=1, cap=2
        )
        self.envelope = self.signer.sign(self.record)

    def verify(self, envelope=None, **changes):
        args = dict(
            context=self.context,
            slots=self.slots,
            public_key=self.signer.public_key,
            maximum_bytes=16384,
        )
        args.update(changes)
        return verify_terminal(self.envelope if envelope is None else envelope, **args)

    def archive(
        self, manifest=None, duplicate=False, manifest_path="controller/manifest.json"
    ):
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w") as tar:
            raw = canonical(self.signer.manifest() if manifest is None else manifest)
            for _ in range(2 if duplicate else 1):
                item = tarfile.TarInfo(manifest_path)
                item.size = len(raw)
                tar.addfile(item, io.BytesIO(raw))
        return stream.getvalue()

    def accepted(self, archive, **changes):
        args = dict(
            accepted_sha256=digest(archive),
            accepted_size=len(archive),
            context=self.context,
            maximum_archive_bytes=32768,
            maximum_manifest_bytes=8192,
        )
        args.update(changes)
        return accepted_manifest(archive, **args)

    def test_distinct_controller_signature_roundtrip(self):
        self.assertEqual(self.verify(), self.record)
        self.assertNotIn("public_key", self.envelope)
        self.assertNotIn("private_key", json.dumps(self.signer.manifest()))

    def test_accepted_artifact_then_manifest_then_signature(self):
        archive = self.archive()
        manifest = self.accepted(archive)
        self.assertEqual(self.verify(public_key=manifest["public_key"]), self.record)

    def test_modified_record_signature_is_rejected(self):
        value = json.loads(json.dumps(self.envelope))
        value["record"]["elapsed_seconds"] = 0
        with self.assertRaises(InvalidSignature):
            self.verify(value)

    def test_foreign_key_and_embedded_replacement_key_are_rejected(self):
        other = ControllerSigner(
            self.context,
            self.slots,
            Ed25519PrivateKey.generate(),
            maximum_terminal_bytes=16384,
            maximum_manifest_bytes=8192,
        )
        with self.assertRaises(ValueError):
            self.verify(other.sign(self.record))
        with self.assertRaises(ValueError):
            self.verify(dict(self.envelope, public_key=other.public_key))

    def test_run_attempt_verification_task_contract_replay_is_rejected(self):
        for field in [
            "run_id",
            "attempt_id",
            "verification_id",
            "task_config_digest",
            "executable_image_digest",
            "source_content_digest",
            "contract_digest",
        ]:
            value = "other" if field.endswith("_id") else "sha256:" + "9" * 64
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.verify(context=replace(self.context, **{field: value}))

    def test_operation_instance_source_request_or_variant_cannot_change_slot_set(self):
        for field, value in [
            ("operation_id", "other"),
            ("instance_id", "other"),
            ("source_digest", "sha256:" + "f" * 64),
            ("request_digest", "sha256:" + "f" * 64),
            ("variant", "reference"),
        ]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.verify(slots=(replace(self.slot, **{field: value}),))

    def test_foreign_slot_cannot_be_signed(self):
        record = terminal(
            replace(self.slot, operation_id="other"),
            status="runtime_error",
            cause="unknown_exception",
            elapsed=1,
            cap=2,
        )
        with self.assertRaises(ValueError):
            self.signer.sign(record)

    def test_artifact_substitution_or_wrong_accepted_size_is_rejected(self):
        archive = self.archive()
        manifest = self.signer.manifest()
        manifest["public_key"] = ControllerSigner(
            self.context,
            self.slots,
            Ed25519PrivateKey.generate(),
            maximum_terminal_bytes=16384,
            maximum_manifest_bytes=8192,
        ).public_key
        replacement = self.archive(manifest)
        with self.assertRaises(ValueError):
            self.accepted(replacement, accepted_sha256=digest(archive))
        with self.assertRaises(ValueError):
            self.accepted(archive, accepted_size=len(archive) + 1)

    def test_duplicate_manifest_and_byte_caps_are_rejected(self):
        with self.assertRaises(ValueError):
            self.accepted(self.archive(duplicate=True))
        with self.assertRaises(ValueError):
            self.accepted(self.archive(), maximum_manifest_bytes=32)
        with self.assertRaises(ValueError):
            self.verify(maximum_bytes=32)

    def test_trusted_nested_cpu_phase_manifest_path(self):
        path = "train/group-00/relay/candidate-batch/cpu-controller/manifest.json"
        archive = self.archive(manifest_path=path)
        self.assertEqual(
            self.accepted(archive, manifest_path=path)["public_key"],
            self.signer.public_key,
        )
        with self.assertRaises(ValueError):
            self.accepted(archive)
        with self.assertRaises(ValueError):
            self.accepted(
                self.archive(duplicate=True, manifest_path=path), manifest_path=path
            )
        for invalid in [
            "../controller/manifest.json",
            "/controller/manifest.json",
            "./controller/manifest.json",
            "controller//manifest.json",
            "controller/../manifest.json",
            "controller\\manifest.json",
        ]:
            with self.subTest(path=invalid), self.assertRaises(ValueError):
                self.accepted(archive, manifest_path=invalid)

    def test_frame_byte_nesting_and_duplicate_keys_fail_before_signature_use(self):
        args = dict(
            context=self.context,
            slots=self.slots,
            public_key=self.signer.public_key,
            maximum_bytes=16384,
        )
        self.assertEqual(
            verify_terminal_bytes(canonical(self.envelope), **args), self.record
        )
        for raw in [
            b"x" * 16385,
            b'{"schema":1,"schema":2}',
            b"[" * 990 + b"0" + b"]" * 990,
            b"\xff",
        ]:
            with (
                self.subTest(length=len(raw)),
                self.assertRaises((ValueError, RecursionError)),
            ):
                verify_terminal_bytes(raw, **args)
        with self.assertRaises(ValueError):
            self.verify(dict(self.envelope, extra="x" * 20000))

    def test_accepted_artifact_context_or_expected_membership_mismatch_is_rejected(
        self,
    ):
        with self.assertRaises(ValueError):
            self.accepted(
                self.archive(), context=replace(self.context, verification_id="other")
            )
        manifest = self.signer.manifest()
        manifest["expected_slots"][0]["operation_id"] = "other"
        with self.assertRaises(ValueError):
            self.accepted(self.archive(manifest))

    def test_signer_bounds_expected_slots_before_materializing_manifest(self):
        with self.assertRaises(ValueError):
            ControllerSigner(
                self.context,
                self.slots,
                Ed25519PrivateKey.generate(),
                maximum_terminal_bytes=16384,
                maximum_manifest_bytes=32,
            )
        with self.assertRaises(ValueError):
            ControllerSigner(
                self.context,
                iter(self.slots),
                Ed25519PrivateKey.generate(),
                maximum_terminal_bytes=16384,
                maximum_manifest_bytes=8192,
            )

    def test_actual_publisher_zstd_wrapper_chain_and_resource_caps(self):
        verifier = JUDGE.parent / "verifier"
        sys.path.insert(0, str(verifier))
        from evidence import write_evidence

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "evidence"
            (data / "controller").mkdir(parents=True)
            (data / "controller/manifest.json").write_bytes(
                canonical(self.signer.manifest())
            )
            (data / "controller/terminal.json").write_bytes(canonical(self.envelope))
            archive = root / "evidence.tar.zst"
            write_evidence(data, archive, 32768)
            raw = archive.read_bytes()
            args = dict(
                accepted_sha256=digest(raw),
                accepted_size=len(raw),
                context=self.context,
                maximum_archive_bytes=32768,
                maximum_expanded_bytes=32768,
                maximum_manifest_bytes=8192,
            )
            manifest = accepted_zstd_manifest(archive, **args)
            self.assertEqual(
                self.verify(public_key=manifest["public_key"]), self.record
            )
            for field, value in [
                ("maximum_expanded_bytes", 512),
                ("maximum_manifest_bytes", 32),
                ("accepted_sha256", "sha256:" + "0" * 64),
                ("accepted_size", len(raw) - 1),
            ]:
                with self.subTest(field=field), self.assertRaises(ValueError):
                    accepted_zstd_manifest(archive, **dict(args, **{field: value}))

    def test_zstd_duplicate_manifest_rejected(self):
        import zstandard

        with tempfile.TemporaryDirectory() as directory:
            raw = zstandard.ZstdCompressor().compress(self.archive(duplicate=True))
            path = Path(directory) / "duplicate.tar.zst"
            path.write_bytes(raw)
            with self.assertRaises(ValueError):
                accepted_zstd_manifest(
                    path,
                    accepted_sha256=digest(raw),
                    accepted_size=len(raw),
                    context=self.context,
                    maximum_archive_bytes=32768,
                    maximum_expanded_bytes=32768,
                    maximum_manifest_bytes=8192,
                )
