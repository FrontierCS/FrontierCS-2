"""task.toml [test_sandboxes.<name>] (one sandbox or a per-instance template): expansion into the platform's `sidecars`
mapping, the image recipes and build args behind them, and their validation."""

from __future__ import annotations

import deployment_test_guard  # noqa: F401  # isort: skip

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_task_images  # noqa: E402
import task_manifest  # noqa: E402
import validate_task_layout  # noqa: E402

TEMPLATE = ROOT / "tasks/template/Theory/template_task"
BASE_A = "docker.io/swebench/sweb.eval.x86_64.a@sha256:" + "a" * 64
BASE_B = "docker.io/swebench/sweb.eval.x86_64.b@sha256:" + "b" * 64

GPU = """
[gpu.verifier]
max_gpu_instances = 1
max_gpu_seconds = 3600

[gpu_parent_resources.verifier]
cpu_millis = 8000
memory_bytes = 17179869184
"""

SERVICES = """
[test_sandboxes.scheduler]
trust = "trusted"
dockerfile = "docker/test-sandbox-scheduler.Dockerfile"
args = ["/py", "scheduler.py", "--port", "8002"]
attestation_key = true
phases = ["train", "eval"]
workdir = "/tmp"
"""

SANDBOXES = """
[test_sandboxes.swe]
dockerfile = "docker/test-sandbox.Dockerfile"
instances = { dataset_key = "swe_ids", manifest = "judge/instances.json" }
workdir = "/testbed"

[test_sandboxes.swe.roles.candidate]
args = ["/py", "-m", "tool", "--instance-id", "{instance}", "--port", "8001"]

[test_sandboxes.swe.roles.check]
args = ["/py", "-m", "check", "--instance-id={instance}"]
"""

MANIFEST = {
    "schema": "any task schema",
    "test_sandboxes": {"swe": {
        "repo__b-2": {"index": "002", "base": BASE_B},
        "repo__a-1": {"index": "000", "base": BASE_A,
                      "build_args": {"TEST_SANDBOX_SOURCE": "restored"}},
    }},
}

SANDBOX_RECIPE = """\
# syntax=docker/dockerfile:1.7
ARG TEST_SANDBOX_BASE
ARG TEST_SANDBOX_SOURCE=declared
FROM docker.io/astral/uv@sha256:%s AS uv
FROM ${TEST_SANDBOX_BASE} AS prepared
ARG TEST_SANDBOX_INSTANCE
COPY --from=uv /uv /opt/uv
RUN echo "$TEST_SANDBOX_INSTANCE"
FROM prepared AS source-declared
FROM prepared AS source-restored
RUN true
FROM source-${TEST_SANDBOX_SOURCE}
""" % ("c" * 64)

SERVICE_RECIPE = "FROM docker.io/library/ubuntu@sha256:%s\n" % ("d" * 64)

EXPECTED = {
    "scheduler": {
        "args": ["/py", "scheduler.py", "--port", "8002"], "attestation_key": True,
        "image": "scheduler", "phases": ["train", "eval"], "workdir": "/tmp"},
    "swe-000-candidate": {
        "args": ["/py", "-m", "tool", "--instance-id", "repo__a-1", "--port", "8001"],
        "image": "swe-000", "phases": ["train", "eval"], "workdir": "/testbed"},
    "swe-000-check": {
        "args": ["/py", "-m", "check", "--instance-id=repo__a-1"],
        "image": "swe-000", "phases": ["train", "eval"], "workdir": "/testbed"},
    "swe-002-candidate": {
        "args": ["/py", "-m", "tool", "--instance-id", "repo__b-2", "--port", "8001"],
        "image": "swe-002", "phases": ["eval"], "workdir": "/testbed"},
    "swe-002-check": {
        "args": ["/py", "-m", "check", "--instance-id=repo__b-2"],
        "image": "swe-002", "phases": ["eval"], "workdir": "/testbed"},
}


class Fixture(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="test-sandboxes-"))
        self.addCleanup(shutil.rmtree, self.tmp)
        self.root = self.tmp / "Theory" / "template_task"
        shutil.copytree(TEMPLATE, self.root)

    def datasets(self, train=("repo__a-1",), eval=("repo__a-1", "repo__b-2")) -> None:
        for mode, ids in (("train", train), ("eval", eval)):
            path = self.root / "datasets" / f"{mode}.json"
            selector = json.loads(path.read_text())
            selector["swe_ids"] = list(ids)
            path.write_text(json.dumps(selector))

    def manifest(self, document=None) -> None:
        (self.root / "judge/instances.json").write_text(json.dumps(document or MANIFEST))

    def write(self, extra: str) -> Path:
        if "test_sandboxes" in extra:
            if not (self.root / "judge/instances.json").exists():
                self.manifest()
            if "swe_ids" not in (self.root / "datasets/train.json").read_text():
                self.datasets()
        manifest = self.root / "task.toml"
        manifest.write_text((TEMPLATE / "task.toml").read_text(encoding="utf-8") + extra,
                            encoding="utf-8")
        return self.root

    def recipes(self, services: bool = True, sandboxes: bool = True) -> None:
        if services:
            (self.root / "docker/test-sandbox-scheduler.Dockerfile").write_text(SERVICE_RECIPE)
        if sandboxes:
            (self.root / "docker/test-sandbox.Dockerfile").write_text(SANDBOX_RECIPE)


class ExpansionTest(Fixture):
    def test_expands_into_the_exact_legacy_sidecars_mapping_in_order(self) -> None:
        policy = task_manifest.policy_from_task(task_manifest.read_task(
            self.write(GPU + SERVICES + SANDBOXES)))
        self.assertEqual(policy["sidecars"], EXPECTED)
        self.assertEqual(list(policy["sidecars"]), list(EXPECTED))
        for name, profile in policy["sidecars"].items():
            self.assertEqual(list(profile), list(EXPECTED[name]), name)

    def test_no_sidecars_key_without_services_or_sandboxes(self) -> None:
        policy = task_manifest.load_policy(self.root)
        self.assertNotIn("sidecars", policy)

    def test_images_and_build_args(self) -> None:
        task = task_manifest.read_task(self.write(GPU + SERVICES + SANDBOXES))
        self.assertEqual(task["sidecar_images"], {
            "scheduler": ("docker/test-sandbox-scheduler.Dockerfile", {}),
            "swe-000": ("docker/test-sandbox.Dockerfile",
                        {"TEST_SANDBOX_BASE": BASE_A, "TEST_SANDBOX_INSTANCE": "repo__a-1",
                         "TEST_SANDBOX_SOURCE": "restored"}),
            "swe-002": ("docker/test-sandbox.Dockerfile",
                        {"TEST_SANDBOX_BASE": BASE_B, "TEST_SANDBOX_INSTANCE": "repo__b-2"}),
        })
        paths = task_manifest.layout(self.root)
        self.assertEqual(paths.dockerfile("sidecar-swe-002"),
                         self.root / "docker/test-sandbox.Dockerfile")
        self.assertEqual(paths.dockerfile("sidecar-scheduler"),
                         self.root / "docker/test-sandbox-scheduler.Dockerfile")
        self.assertEqual(paths.build_args("sidecar-swe-000")["TEST_SANDBOX_INSTANCE"], "repo__a-1")
        self.assertEqual(paths.build_args("sidecar-scheduler"), {})
        self.assertEqual(paths.build_args("verifier"), {})
        with self.assertRaises(task_manifest.ManifestError):
            paths.dockerfile("sidecar-swe-001")

    def refused(self, extra: str, message: str) -> None:
        with self.assertRaisesRegex(task_manifest.ManifestError, message):
            task_manifest.read_task(self.write(extra))

    def broken_manifest(self, change) -> dict:
        document = json.loads(json.dumps(MANIFEST))
        change(document["test_sandboxes"]["swe"])
        return document

    def test_refuses_repeated_index(self) -> None:
        self.manifest(self.broken_manifest(lambda i: i["repo__b-2"].update(index="000")))
        self.refused(GPU + SANDBOXES, "repeated")

    def test_refuses_unpinned_base(self) -> None:
        self.manifest(self.broken_manifest(
            lambda i: i["repo__b-2"].update(base="docker.io/swebench/b:latest")))
        self.refused(GPU + SANDBOXES, "digest")

    def test_refuses_build_args_that_override_base_or_instance(self) -> None:
        self.manifest(self.broken_manifest(
            lambda i: i["repo__b-2"].update(build_args={"TEST_SANDBOX_BASE": "x"})))
        self.refused(GPU + SANDBOXES, "build_args")

    def test_manifest_must_cover_exactly_the_selected_instances(self) -> None:
        self.manifest(self.broken_manifest(lambda i: i.pop("repo__b-2")))
        self.refused(GPU + SANDBOXES, "exactly the ids")
        self.manifest()
        self.datasets(eval=("repo__a-1", "repo__b-2", "repo__c-3"))
        self.refused(GPU + SANDBOXES, "exactly the ids")

    def test_phases_follow_dataset_membership(self) -> None:
        self.datasets(train=("repo__b-2",), eval=("repo__a-1",))
        policy = task_manifest.load_policy(self.write(GPU + SANDBOXES))
        self.assertEqual(policy["sidecars"]["swe-000-check"]["phases"], ["eval"])
        self.assertEqual(policy["sidecars"]["swe-002-check"]["phases"], ["train"])

    def test_refuses_role_without_placeholder(self) -> None:
        self.refused(GPU + SANDBOXES.replace('"--instance-id={instance}"', '"--x"'), "{instance}")

    def test_refuses_other_instance_sources_and_unknown_keys(self) -> None:
        self.refused(GPU + SANDBOXES.replace('dataset_key', 'key'), "unknown keys")
        self.refused(GPU + SANDBOXES.replace('workdir = "/testbed"',
                                             'workdir = "/testbed"\nargs = ["/x"]'), "never both")
        self.refused(GPU + SERVICES.replace("workdir", "cwd"), "unknown keys")
        self.refused(GPU + SERVICES.replace('phases = ["train", "eval"]', 'phases = ["final"]'),
                     "phases")

    def test_refuses_name_clash(self) -> None:
        self.refused(GPU + SERVICES.replace("test_sandboxes.scheduler",
                                            "test_sandboxes.swe-000-check")
                     + SANDBOXES, "share the names")

    def test_attestation_key_needs_a_trusted_sandbox(self) -> None:
        self.refused(GPU + SERVICES.replace('trust = "trusted"\n', ''), "trusted")
        self.refused(GPU + SERVICES.replace('trust = "trusted"', 'trust = "untrusted"'),
                     "trusted")
        self.refused(GPU + SERVICES.replace('trust = "trusted"', 'trust = "half"'), "trust")
        self.refused(GPU + SANDBOXES.replace('workdir = "/testbed"',
                                             'workdir = "/testbed"\nattestation_key = true'),
                     "trusted")
        self.refused(GPU + SERVICES.replace("test-sandbox-scheduler", "service-scheduler"),
                     "dockerfile")


class BuildTest(Fixture):
    def test_from_build_arg_is_resolved_and_checked(self) -> None:
        path = self.root / "docker/test-sandbox.Dockerfile"
        path.write_text(SANDBOX_RECIPE)
        arguments = {"TEST_SANDBOX_BASE": BASE_A, "TEST_SANDBOX_INSTANCE": "repo__a-1"}
        self.assertEqual(build_task_images.dockerfile_bases(path, arguments)[1], BASE_A)
        with self.assertRaisesRegex(build_task_images.BuildFailure, "unreviewed"):
            build_task_images.dockerfile_bases(path)
        with self.assertRaisesRegex(build_task_images.BuildFailure, "unreviewed"):
            build_task_images.dockerfile_bases(
                path, {**arguments, "TEST_SANDBOX_BASE": "docker.io/x/y:latest"})
        bases = {"docker.io/astral/uv@sha256:" + "c" * 64: "x", BASE_A: BASE_A}
        self.assertEqual(
            build_task_images.freeze_bases(path, bases, None, arguments),
            ["--build-arg", f"TEST_SANDBOX_BASE={BASE_A}",
             "--build-arg", "TEST_SANDBOX_INSTANCE=repo__a-1"])

    def test_build_problem_builds_one_image_per_sandbox_with_its_args(self) -> None:
        self.write(GPU + SERVICES + SANDBOXES)
        self.recipes()
        (self.root / "docker/verifier-gpu.Dockerfile").write_text(SERVICE_RECIPE)
        definition = {"image_tag": "source-x", "sidecars": EXPECTED,
                      "gpu": {"verifier": {"max_gpu_instances": 1}},
                      "source": {}, "datasets": {}, "dataset_extension": "json"}
        commands = []
        with mock.patch.object(build_task_images, "run",
                               side_effect=lambda c, **k: commands.append(list(c)) or ""), \
                mock.patch.object(build_task_images, "stage_datasets", return_value={}), \
                mock.patch.object(build_task_images, "verify_dataset_image"), \
                mock.patch.object(build_task_images, "registry_digest", return_value=None), \
                mock.patch.object(build_task_images, "verifier_smoke", return_value=""), \
                mock.patch.object(build_task_images, "pull_digest_reference",
                                  side_effect=lambda source: source), \
                mock.patch.object(build_task_images, "REPOSITORY_ROOT", self.tmp):
            built = build_task_images.build_problem(
                "template_task", "docker.io/p", push=False, bases={}, root=self.root,
                definition=definition)
        self.assertEqual(sorted(k for k in built["images"] if k.startswith("sidecar-")),
                         ["sidecar-scheduler", "sidecar-swe-000", "sidecar-swe-002"])
        builds = {c[c.index("--tag") + 1]: c for c in commands if c[:2] == ["docker", "build"]}
        sandbox = builds["docker.io/p/fcs2-task-template_task-sidecar-swe-002:source-x"]
        self.assertTrue(sandbox[sandbox.index("-f") + 1].endswith("docker/test-sandbox.Dockerfile"))
        self.assertIn(f"TEST_SANDBOX_BASE={BASE_B}", sandbox)
        self.assertIn("TEST_SANDBOX_INSTANCE=repo__b-2", sandbox)
        service = builds["docker.io/p/fcs2-task-template_task-sidecar-scheduler:source-x"]
        self.assertNotIn("--build-arg", service)


class ValidatorTest(Fixture):
    def test_recipes_must_match_task_toml(self) -> None:
        self.write(GPU + SERVICES + SANDBOXES)
        self.recipes()
        validate_task_layout.check_native_sidecar_recipes(self.root)
        (self.root / "docker/test-sandbox-extra.Dockerfile").write_text(SERVICE_RECIPE)
        with self.assertRaisesRegex(validate_task_layout.LayoutFailure, "test-sandbox-extra"):
            validate_task_layout.check_native_sidecar_recipes(self.root)

    def test_missing_sandbox_recipe_or_base_arg_is_refused(self) -> None:
        self.write(GPU + SERVICES + SANDBOXES)
        self.recipes(sandboxes=False)
        with self.assertRaisesRegex(validate_task_layout.LayoutFailure, "test-sandbox"):
            validate_task_layout.check_native_sidecar_recipes(self.root)
        (self.root / "docker/test-sandbox.Dockerfile").write_text(
            SANDBOX_RECIPE.replace("${TEST_SANDBOX_BASE}", BASE_A))
        with self.assertRaisesRegex(validate_task_layout.LayoutFailure, "TEST_SANDBOX_BASE"):
            validate_task_layout.check_native_sidecar_recipes(self.root)

    def test_sandbox_recipe_may_copy_from_its_own_stages_only(self) -> None:
        self.write(GPU + SANDBOXES)
        self.recipes(services=False)
        validate_task_layout.check_native_dockerfile(self.root, "sidecar-swe-000")
        path = self.root / "docker/test-sandbox.Dockerfile"
        path.write_text(SANDBOX_RECIPE + "COPY --from=elsewhere /x /x\n")
        with self.assertRaisesRegex(validate_task_layout.LayoutFailure, "elsewhere"):
            validate_task_layout.check_native_dockerfile(self.root, "sidecar-swe-000")


class StandaloneVerifierTest(Fixture):
    def with_command(self) -> None:
        text = (TEMPLATE / "task.toml").read_text(encoding="utf-8")
        head, tail = text.split("[verifier]\n", 1)
        (self.root / "task.toml").write_text(
            head + "[verifier]\ncommand = [\"/opt/frontier/python/bin/python\", "
            "\"/opt/frontier/verifier/verify.py\"]\n" + tail, encoding="utf-8")

    def test_template_passes_and_refuses_a_verifier_dir(self) -> None:
        validate_task_layout.validate_task(self.root)
        (self.root / "verifier").mkdir()
        (self.root / "verifier/verify.py").write_text("")
        with self.assertRaisesRegex(validate_task_layout.LayoutFailure, "forks shared"):
            validate_task_layout.validate_task(self.root)

    def test_own_command_needs_and_allows_verifier_verify_py(self) -> None:
        self.with_command()
        with self.assertRaisesRegex(validate_task_layout.LayoutFailure, "standalone Verifier"):
            validate_task_layout.validate_task(self.root)
        (self.root / "verifier").mkdir()
        (self.root / "verifier/verify.py").write_text("")
        with self.assertRaisesRegex(validate_task_layout.LayoutFailure, "docstring"):
            validate_task_layout.validate_task(self.root)
        (self.root / "verifier/verify.py").write_text(
            '"""Scoring note."""\nRESULT = {"beats_reference": 0}\n')
        with self.assertRaisesRegex(validate_task_layout.LayoutFailure, "beat_spec_progress"):
            validate_task_layout.validate_task(self.root)
        # A standalone Verifier names the template's [feedback] metric itself.
        (self.root / "verifier/verify.py").write_text(
            '"""Scoring note."""\nRESULT = {"beats_reference": 0, "beat_spec_progress": {},'
            ' "wall_seconds_total": 0.0}\n')
        validate_task_layout.validate_task(self.root)


if __name__ == "__main__":
    unittest.main()
