from __future__ import annotations

# Install isolation before project imports.
import deployment_test_guard  # noqa: F401  # isort: skip

import copy
import hashlib
import importlib.machinery
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.dont_write_bytecode = True
import render_task_definition as renderer
import validate_task_layout as native_layout
from preview import images as incremental
from preview import infra

SPEC = importlib.util.spec_from_file_location(
    "build_task_images", ROOT / "scripts" / "build_task_images.py"
)
assert SPEC is not None and SPEC.loader is not None
build_task_images = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_task_images)


class DemoImageBuildTests(unittest.TestCase):
    def test_reviewed_metaopt_mcr_sdk_is_exact_and_pinned(self):
        approved = "mcr.microsoft.com/dotnet/sdk:8.0-bookworm-slim@sha256:3ee581addaa35c8c790f5f62441b189810522bb32e43a8db9d4283b3f649cc24"
        self.assertTrue(build_task_images.approved_base(approved))
        rejected = (
            approved.split("@", 1)[0],
            approved.replace("8.0-bookworm-slim", "9.0-bookworm-slim"),
            approved[:-1] + "0",
            approved.replace("/sdk:", "/runtime:"),
            approved.replace("mcr.microsoft.com", "private.example"),
            approved.replace("mcr.microsoft.com", "mcr.microsoft.com.evil.example"),
        )
        with tempfile.TemporaryDirectory() as directory:
            recipe = Path(directory) / "Dockerfile"
            recipe.write_text("FROM " + approved + "\n")
            self.assertEqual(build_task_images.dockerfile_bases(recipe), [approved])
            for source in rejected:
                with self.subTest(source=source):
                    self.assertFalse(build_task_images.approved_base(source))
                    recipe.write_text("FROM " + source + "\n")
                    with self.assertRaises(build_task_images.BuildFailure):
                        build_task_images.dockerfile_bases(recipe)

    def test_imports_do_not_read_catalog(self):
        code = """
from pathlib import Path
from unittest import mock
original = Path.open
def checked_open(path, *args, **kwargs):
    if path.name == 'task-catalog.json':
        raise AssertionError('catalog read during import')
    return original(path, *args, **kwargs)
with mock.patch.object(Path, 'open', checked_open):
    import build_task_images
    import import_task
"""
        result = subprocess.run(
            [sys.executable, "-B", "-c", code],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((result.stdout, result.stderr), ("", ""))

    def test_missing_catalog_allows_help_but_refuses_build(self):
        with (
            mock.patch.object(
                renderer,
                "catalog_task_names",
                side_effect=ValueError("cannot read task catalog: missing"),
            ),
            mock.patch.object(build_task_images, "build_problem") as build,
            mock.patch("sys.stdout", new_callable=io.StringIO) as stdout,
        ):
            with self.assertRaises(SystemExit) as exit_status:
                build_task_images.main(["--help"])
            self.assertEqual(exit_status.exception.code, 0)
            self.assertIn("--problem {all,cpu}", stdout.getvalue())
            with self.assertRaisesRegex(
                build_task_images.BuildFailure, "cannot read task catalog"
            ):
                build_task_images.main(["--problem", "all"])
        build.assert_not_called()

    def test_escape_directive_cannot_change_continuation_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Dockerfile"
            path.write_text(
                "# escape=`\nFROM ubuntu:24.04\nRUN echo ok \\\nFROM private.example/image:latest\n"
            )
            with self.assertRaisesRegex(
                build_task_images.BuildFailure, "unsupported parser directive"
            ):
                build_task_images.dockerfile_bases(path)
            path.write_text(
                "# escape=\\\n# syntax=docker/dockerfile:1\nFROM ubuntu:24.04\n"
            )
            self.assertEqual(build_task_images.dockerfile_bases(path), ["ubuntu:24.04"])

    def test_comment_backslash_cannot_hide_an_unapproved_from(self):
        for newline in ("\n", "\r\n"):
            with (
                self.subTest(newline=newline),
                tempfile.TemporaryDirectory() as directory,
            ):
                path = Path(directory) / "Dockerfile"
                path.write_bytes(
                    newline.join(
                        [
                            "FROM scratch",
                            "# comment \\",
                            "FROM private.example/image:latest",
                            "",
                        ]
                    ).encode()
                )
                with mock.patch.object(
                    build_task_images, "pull_digest_reference"
                ) as pull:
                    with self.assertRaisesRegex(
                        build_task_images.BuildFailure, "private.example/image:latest"
                    ):
                        build_task_images.freeze_bases(path, {}, None)
                pull.assert_not_called()

    def test_from_continuations_platform_aliases_and_continued_comments(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Dockerfile"
            path.write_text(
                "  FrOm \\\n# comment \\\n  --platform=linux/amd64 \\\n"
                "  python:3.11-slim-bookworm aS base\nFROM base AS final\n"
            )
            self.assertEqual(
                build_task_images.dockerfile_bases(path), ["python:3.11-slim-bookworm"]
            )
            # Docker joins continued instruction bytes without inserting spaces.
            path.write_text("FROM ubun\\\ntu:24.04\n")
            self.assertEqual(build_task_images.dockerfile_bases(path), ["ubuntu:24.04"])

    def test_heredoc_bodies_are_skipped_for_run_copy_and_add(self):
        for command in ("RUN", "COPY", "ADD"):
            for marker, terminator in (
                ("<<EOF", "EOF"),
                ("<<-EOF", "\tEOF"),
                ("<<'EOF'", "EOF"),
                ('<<"EOF"', "EOF"),
                ("<<-'EOF'", "\tEOF"),
            ):
                with (
                    self.subTest(command=command, marker=marker),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    path = Path(directory) / "Dockerfile"
                    text = (
                        "FROM python:3.11-slim-bookworm AS base\n"
                        f"{command} {marker}"
                        + (" /file" if command != "RUN" else "")
                        + "\n"
                        "from pathlib import Path\nFROM private.example/in-heredoc:latest\n"
                        "# body comment \\\n" + terminator + "\nFROM base AS final\n"
                    )
                    path.write_text(text)
                    self.assertEqual(
                        build_task_images.dockerfile_bases(path),
                        ["python:3.11-slim-bookworm"],
                    )
                    path.write_text(
                        text + "FROM private.example/after-heredoc:latest\n"
                    )
                    with self.assertRaisesRegex(
                        build_task_images.BuildFailure, "private.example/after-heredoc"
                    ):
                        build_task_images.dockerfile_bases(path)

    def test_multiple_heredocs_and_quoted_literals(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Dockerfile"
            path.write_text(
                "FROM scratch\nCOPY <<'FIRST' \\\n<<-SECOND /files/\n"
                "from pathlib import Path\nFIRST\nFROM ignored\n\tSECOND\nFROM ubuntu:24.04\n"
            )
            self.assertEqual(
                build_task_images.dockerfile_bases(path), ["scratch", "ubuntu:24.04"]
            )
            for literal in ("'<<EOF'", '"<<EOF"', r"\<<EOF", "prefix<<EOF"):
                path.write_text(
                    f"FROM scratch\nRUN echo {literal}\nFROM private.example/image:latest\nEOF\n"
                )
                with (
                    self.subTest(literal=literal),
                    self.assertRaisesRegex(
                        build_task_images.BuildFailure, "private.example"
                    ),
                ):
                    build_task_images.dockerfile_bases(path)
            path.write_text("FROM scratch\nRUN <<EOF\nfrom pathlib import Path\n")
            with self.assertRaisesRegex(
                build_task_images.BuildFailure, "unterminated heredoc"
            ):
                build_task_images.dockerfile_bases(path)

    def test_empty_catalog_is_rejected_before_building(self):
        read_catalog = renderer.catalog_task_names
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            path.write_text('{"tasks": []}')
            with mock.patch.object(
                renderer, "catalog_task_names", side_effect=lambda: read_catalog(path)
            ):
                module = importlib.util.module_from_spec(SPEC)
                SPEC.loader.exec_module(module)
                with mock.patch.object(module, "build_problem") as build:
                    with self.assertRaisesRegex(
                        module.BuildFailure, "task catalog lists no tasks"
                    ):
                        module.main(["--problem", "all"])
                build.assert_not_called()

    def test_malformed_definition_type_fails_before_building(self):
        definitions = json.loads(renderer.DEFINITIONS.read_text())
        name = "batch_local_counting"
        for value in (None, [], "gpu", 1, False):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "definitions.json"
                path.write_text(json.dumps({**definitions, name: value}))
                with (
                    mock.patch.object(renderer, "DEFINITIONS", path),
                    mock.patch.object(build_task_images, "build_problem") as build,
                ):
                    with self.assertRaisesRegex(
                        build_task_images.BuildFailure, f"catalog task {name}.*object"
                    ):
                        build_task_images.main(["--problem", "cpu"])
                build.assert_not_called()

    def test_agent_gpu_file_is_not_a_gpu_directory_in_check_or_build(self):
        name = "a_task"
        definition = {"image_tag": "source-test", "source": {}, "datasets": {}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            problem = root / "problems" / name
            problem.mkdir(parents=True)
            (problem / "agent-gpu").write_text("not a directory")
            (problem / "agent-base").mkdir()
            (problem / "verifier").mkdir()
            definitions = root / "definitions.json"
            definitions.write_text(json.dumps({name: definition}))
            manifest = root / "manifest.json"
            with (
                mock.patch.object(build_task_images, "TASK_ROOT", root),
                mock.patch.object(renderer, "catalog_task_names", return_value=[name]),
                mock.patch.object(renderer, "DEFINITIONS", definitions),
                mock.patch.object(renderer, "render", return_value=definition),
                mock.patch.object(build_task_images, "gpu_ssh_base") as gpu_base,
                mock.patch.object(build_task_images, "freeze_bases") as freeze,
                mock.patch.object(build_task_images, "stage_datasets", return_value={}),
                mock.patch.object(build_task_images, "verify_dataset_image"),
                mock.patch.object(build_task_images, "run"),
                mock.patch("sys.stdout", new_callable=io.StringIO),
            ):
                self.assertEqual(
                    build_task_images.main(
                        ["--problem", "cpu", "--manifest", str(manifest)]
                    ),
                    0,
                )
            self.assertEqual(
                set(json.loads(manifest.read_text())["tasks"][name]["images"]),
                {"agent", "verifier", "datasets"},
            )
            gpu_base.assert_not_called()
            self.assertEqual(freeze.call_count, 2)

    def test_verifier_only_gpu_build_and_cpu_selection_follow_enabled_roles(self):
        name = "verifier_only"
        definition = {
            "image_tag": "source-test",
            "source": {},
            "datasets": {},
            "gpu": {"verifier": {"max_gpu_instances": 1, "max_gpu_seconds": 60}},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            problem = root / "problems" / name
            for role in ("agent-base", "verifier", "verifier-gpu"):
                (problem / role).mkdir(parents=True)
                (problem / role / "Dockerfile").write_text("FROM scratch\n")
            definitions = root / "definitions.json"
            definitions.write_text(json.dumps({name: definition}))
            manifest = root / "manifest.json"
            with (
                mock.patch.object(build_task_images, "TASK_ROOT", root),
                mock.patch.object(renderer, "catalog_task_names", return_value=[name]),
                mock.patch.object(renderer, "DEFINITIONS", definitions),
                mock.patch.object(renderer, "render", return_value=definition),
                mock.patch.object(build_task_images, "gpu_ssh_base") as gpu_base,
                mock.patch.object(build_task_images, "freeze_bases", return_value=[]),
                mock.patch.object(build_task_images, "stage_datasets", return_value={}),
                mock.patch.object(build_task_images, "verify_dataset_image"),
                mock.patch.object(build_task_images, "run") as run,
            ):
                build_task_images.main(
                    ["--problem", "all", "--manifest", str(manifest)]
                )
                self.assertEqual(
                    set(json.loads(manifest.read_text())["tasks"][name]["images"]),
                    {"agent", "verifier", "verifier-gpu", "datasets"},
                )
                gpu_base.assert_not_called()
                run.reset_mock()
                build_task_images.main(
                    ["--problem", "cpu", "--manifest", str(manifest)]
                )
                run.assert_not_called()

    def test_all_ignores_extra_directory_with_one_warning(self):
        paths = list((build_task_images.TASK_ROOT / "problems").iterdir())
        names = [
            task["name"]
            for task in json.loads(build_task_images.TASK_CATALOG.read_text())["tasks"]
        ]
        with tempfile.TemporaryDirectory() as directory:
            extra = Path(directory) / "extra_task"
            extra.mkdir()
            output = Path(directory) / "manifest.json"
            with (
                mock.patch.object(Path, "iterdir", return_value=iter([*paths, extra])),
                mock.patch.object(
                    build_task_images, "build_problem", return_value={}
                ) as build,
                mock.patch("sys.stderr", new_callable=io.StringIO) as stderr,
            ):
                self.assertEqual(
                    build_task_images.main(
                        ["--problem", "all", "--manifest", str(output)]
                    ),
                    0,
                )
            self.assertEqual([call.args[0] for call in build.call_args_list], names)
            self.assertEqual(set(json.loads(output.read_text())["tasks"]), set(names))
            self.assertEqual(
                stderr.getvalue().splitlines(),
                [f"{extra}: ignored: not in config/task-catalog.json"],
            )

    def test_missing_catalog_directory_fails_before_building(self):
        with (
            mock.patch.object(
                renderer, "catalog_task_names", return_value=["missing_task"]
            ),
            mock.patch.object(build_task_images, "build_problem") as build,
        ):
            with self.assertRaisesRegex(
                build_task_images.BuildFailure,
                "catalog task missing_task has no problem directory",
            ):
                build_task_images.main(["--problem", "all"])
        build.assert_not_called()

    def test_gpu_definition_directory_disagreement_fails_before_building(self):
        definitions = json.loads(renderer.DEFINITIONS.read_text())
        for name in renderer.catalog_task_names():
            changed = json.loads(json.dumps(definitions))
            if "gpu" in changed[name]:
                del changed[name]["gpu"]
            else:
                changed[name]["gpu"] = {}
            with self.subTest(task=name), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "definitions.json"
                path.write_text(json.dumps(changed))
                with (
                    mock.patch.object(renderer, "DEFINITIONS", path),
                    mock.patch.object(build_task_images, "build_problem") as build,
                ):
                    with self.assertRaisesRegex(
                        build_task_images.BuildFailure,
                        f"GPU definition disagrees.*{name}",
                    ):
                        build_task_images.main(["--problem", "cpu"])
                build.assert_not_called()

    def test_native_gpu_quotas_and_ssh_environment_fix_are_preserved(self) -> None:
        root = renderer.problem_directory(
            build_task_images.TASK_ROOT / "problems", build_task_images.SPARSE_VIDEO
        )
        policy = json.loads((root / "policy.json").read_text())
        self.assertEqual(policy["gpu"]["agent"]["max_instances"], 1)
        self.assertEqual(policy["gpu"]["verifier"]["max_gpu_instances"], 4)
        self.assertEqual(
            policy["network"]["audit"]["max_ssh_bytes_per_direction"], 512 * 1024**2
        )
        self.assertIn(
            "FRONTIER_VERIFIER_GPU_COUNT=4", (root / "verifier/Dockerfile").read_text()
        )
        self.assertIn(
            "/opt/frontier/runtime-python", (root / "agent-gpu/Dockerfile").read_text()
        )
        self.assertFalse((root / "agent-base/frontier/task/bin/frontier-gpu").exists())
        self.assertIn(
            "task-runtime --snapshot", (root / "agent-gpu/Dockerfile").read_text()
        )
        self.assertIn(
            "COPY agent-base/frontier/task/repo /frontier/task/repo",
            (root / "agent-gpu/Dockerfile").read_text(),
        )
        readme = (root / "agent-base/frontier/task/README.md").read_text()
        self.assertIn("/opt/frontier/candidate/bin/frontier-gpu", readme)
        self.assertIn("frontier-gpu exec", readme)

    def test_pinned_pull_preserves_requested_reference(self) -> None:
        reference = "docker.io/example/ssh@sha256:" + "a" * 64
        with mock.patch.object(build_task_images, "run") as run:
            self.assertEqual(
                build_task_images.pull_digest_reference(reference), reference
            )
        run.assert_called_once_with(
            ["docker", "pull", "--platform", "linux/amd64", reference]
        )

    def test_digest_pinned_bases_need_no_context_overrides(self):
        rust = "rust:1.64-bullseye@sha256:" + "a" * 64
        debian = "debian:bookworm-slim@sha256:" + "b" * 64
        with tempfile.TemporaryDirectory() as directory:
            dockerfile = Path(directory) / "Dockerfile"
            dockerfile.write_text(
                f"FROM {rust} AS toolchain\nFROM {debian}\n"
                "COPY --from=toolchain /usr/local /usr/local\n"
            )
            before = dockerfile.read_bytes()
            bases = {}
            with mock.patch.object(build_task_images, "run") as run:
                self.assertEqual(
                    build_task_images.freeze_bases(dockerfile, bases, None), []
                )
            self.assertEqual(bases, {rust: rust, debian: debian})
            self.assertEqual(
                run.call_args_list,
                [
                    mock.call(["docker", "pull", "--platform", "linux/amd64", source])
                    for source in (rust, debian)
                ],
            )
            self.assertEqual(dockerfile.read_bytes(), before)

    def test_mixed_bases_override_only_mutable_tags(self):
        rust = "rust:1.64-bullseye@sha256:" + "a" * 64
        ubuntu = "ubuntu@sha256:" + "b" * 64
        with tempfile.TemporaryDirectory() as directory:
            dockerfile = Path(directory) / "Dockerfile"
            dockerfile.write_text(
                f"FROM {rust} AS toolchain\nFROM ubuntu:24.04\n"
                "COPY --from=toolchain /usr/local /usr/local\n"
            )
            before = dockerfile.read_bytes()
            bases = {}
            with mock.patch.object(
                build_task_images, "run", return_value=ubuntu
            ) as run:
                self.assertEqual(
                    build_task_images.freeze_bases(dockerfile, bases, None),
                    ["--build-context", "ubuntu:24.04=docker-image://" + ubuntu],
                )
            self.assertEqual(bases, {rust: rust, "ubuntu:24.04": ubuntu})
            self.assertEqual(
                run.call_args_list,
                [
                    mock.call(["docker", "pull", "--platform", "linux/amd64", rust]),
                    mock.call(
                        ["docker", "pull", "--platform", "linux/amd64", "ubuntu:24.04"]
                    ),
                    mock.call(
                        [
                            "docker",
                            "image",
                            "inspect",
                            "--format",
                            "{{index .RepoDigests 0}}",
                            "ubuntu:24.04",
                        ],
                        capture=True,
                    ),
                ],
            )
            self.assertEqual(dockerfile.read_bytes(), before)

    def test_verifier_only_gpu_resolves_ssh_bootstrap_when_referenced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for role in ("agent-base", "verifier", "verifier-gpu"):
                (root / role).mkdir()
                (root / role / "Dockerfile").write_text("FROM scratch\n")
            definition = {
                "image_tag": "source-fixture",
                "source": {},
                "datasets": {},
                "gpu": {"verifier": {"max_gpu_instances": 1}},
            }
            ssh = "docker.io/example/ssh@sha256:" + "a" * 64
            for needs_ssh in (True, False):
                (root / "verifier-gpu/Dockerfile").write_text(
                    "  from --platform=linux/amd64 ${FRONTIER_GPU_SSH_BASE_IMAGE} AS ssh\n"
                    if needs_ssh
                    else "FROM scratch\n"
                )
                with (
                    self.subTest(needs_ssh=needs_ssh),
                    mock.patch.object(
                        build_task_images, "gpu_ssh_base", return_value=ssh
                    ) as bootstrap,
                    mock.patch.object(
                        build_task_images, "pull_digest_reference", return_value=ssh
                    ) as pull,
                    mock.patch.object(build_task_images, "run") as run,
                    mock.patch.object(
                        build_task_images, "stage_datasets", return_value={}
                    ),
                    mock.patch.object(build_task_images, "verify_dataset_image"),
                ):
                    result = build_task_images.build_problem(
                        "verifier_only",
                        "example",
                        push=False,
                        bases={},
                        root=root,
                        definition=definition,
                    )
                self.assertEqual(bootstrap.call_count, int(needs_ssh))
                self.assertEqual(pull.call_count, int(needs_ssh))
                self.assertIn("verifier-gpu", result["images"])
                self.assertNotIn("agent-gpu", result["images"])
                command = next(
                    c.args[0]
                    for c in run.call_args_list
                    if str(root / "verifier-gpu/Dockerfile") in c.args[0]
                )
                self.assertEqual(
                    "FRONTIER_GPU_SSH_BASE_IMAGE=" + ssh in command, needs_ssh
                )

    def test_native_bootstrap_rejects_private_or_mutable_input(self) -> None:
        for repository, digest in (
            (
                "123456789012.dkr.ecr.us-west-1.amazonaws.com/agent",
                "sha256:" + "a" * 64,
            ),
            ("docker.io/example/agent", "latest"),
        ):
            catalog = {
                "tasks": [
                    {
                        "name": build_task_images.SPARSE_VIDEO,
                        "agent_gpu_image": {"repository": repository, "digest": digest},
                    }
                ]
            }
            with mock.patch.object(Path, "read_text", return_value=json.dumps(catalog)):
                with self.assertRaises(build_task_images.BuildFailure):
                    build_task_images.gpu_ssh_base()

    def test_build_catalog_with_native_contexts_names_and_frozen_bases(self):
        commands = []
        catalog_bytes = build_task_images.TASK_CATALOG.read_bytes()
        catalog = json.loads(catalog_bytes)
        catalog_tasks = {task["name"]: task for task in catalog["tasks"]}
        self.assertEqual(list(renderer.catalog_task_names()), list(catalog_tasks))
        definitions = json.loads(renderer.DEFINITIONS.read_text())
        bases = {}

        def fake_run(command, **kwargs):
            commands.append(command)
            if command[:3] == ["docker", "image", "inspect"]:
                return command[-1].split(":")[0] + "@sha256:" + "a" * 64
            return ""

        with (
            mock.patch.object(build_task_images, "run", side_effect=fake_run),
            mock.patch.object(
                renderer, "render", side_effect=lambda root, name: definitions[name]
            ),
            mock.patch.object(build_task_images, "verify_dataset_image") as verify,
        ):
            for name in renderer.catalog_task_names():
                result = build_task_images.build_problem(
                    name, "joyemang33", push=False, bases=bases
                )
                self.assertEqual(
                    len(result["images"]),
                    sum(key.endswith("_image") for key in catalog_tasks[name])
                    + len(catalog_tasks[name].get("sidecar_images", {})),
                )
        self.assertEqual(verify.call_count, len(catalog_tasks))
        required_bases = set()
        for name in renderer.catalog_task_names():
            root = renderer.problem_directory(renderer.TASK_ROOT / "problems", name)
            for role in ("agent-base", "verifier", "agent-gpu", "verifier-gpu"):
                recipe = root / role / "Dockerfile"
                if recipe.is_file():
                    required_bases.update(build_task_images.dockerfile_bases(recipe))
        self.assertTrue((required_bases - {"scratch"}).issubset(bases))
        self.assertTrue(any(key.startswith("pytorch/") for key in bases))
        self.assertIn("${FRONTIER_GPU_SSH_BASE_IMAGE}", bases)
        builds = [c for c in commands if c[:2] == ["docker", "build"]]
        self.assertEqual(
            len(builds),
            sum(
                sum(key.endswith("_image") for key in task)
                + len(task.get("sidecar_images", {}))
                for task in catalog_tasks.values()
            ),
        )
        catalog = json.loads(build_task_images.TASK_CATALOG.read_text())
        expected = {
            image["repository"].removeprefix("docker.io/")
            + ":"
            + (
                "v1"
                if key == "dataset_image"
                else definitions[task["name"]]["image_tag"]
            )
            for task in catalog["tasks"]
            for key, image in task.items()
            if key.endswith("_image")
        } | {
            # Declared sidecar images build under the same reviewed source tag.
            image["repository"].removeprefix("docker.io/")
            + ":"
            + definitions[task["name"]]["image_tag"]
            for task in catalog["tasks"]
            for image in task.get("sidecar_images", {}).values()
        }
        for task in catalog["tasks"]:
            source_tag = definitions[task["name"]]["image_tag"]
            self.assertEqual(
                source_tag,
                "source-"
                + definitions[task["name"]]["source"]["source_set_sha256"].removeprefix(
                    "sha256:"
                ),
            )
            for key, image in task.items():
                if key.endswith("_image") and not (
                    key == "dataset_image" and image["tag"] == "v1"
                ):
                    self.assertTrue(
                        infra.source_image_tag_matches(
                            image["tag"], source_tag, image["digest"]
                        )
                    )
            for image in task.get("sidecar_images", {}).values():
                self.assertTrue(
                    infra.source_image_tag_matches(
                        image["tag"], source_tag, image["digest"]
                    )
                )
        self.assertEqual(catalog_bytes, build_task_images.TASK_CATALOG.read_bytes())
        self.assertFalse(any(c[:2] == ["docker", "push"] for c in commands))
        self.assertEqual({c[c.index("--tag") + 1] for c in builds}, expected)
        for command in builds:
            self.assertIn("--provenance=false", command)
            self.assertIn("linux/amd64", command)
            dockerfile = Path(command[command.index("-f") + 1])
            if "datasets" not in command[command.index("--tag") + 1]:
                self.assertEqual(command[-1], str(dockerfile.parent.parent))
                sources = build_task_images.dockerfile_bases(dockerfile)
                # Already-pinned FROM images need no BuildKit override.
                expected_contexts = [
                    f"{source}=docker-image://{bases[source]}"
                    for source in sources
                    if source not in {"scratch", build_task_images.SSH_BASE_ARGUMENT}
                    and not build_task_images.DIGEST_REFERENCE_RE.fullmatch(source)
                ]
                contexts = [
                    command[index + 1]
                    for index, value in enumerate(command)
                    if value == "--build-context"
                ]
                self.assertEqual(contexts, expected_contexts)
                if build_task_images.SSH_BASE_ARGUMENT in sources:
                    self.assertIn(
                        "FRONTIER_GPU_SSH_BASE_IMAGE="
                        + build_task_images.gpu_ssh_base(),
                        command,
                    )
        self.assertFalse(any(c[1] in {"push", "login"} for c in commands))

    def test_dataset_staging_and_single_layer_member_bodies(self):
        name = "batch_local_counting"
        definition = json.loads(renderer.DEFINITIONS.read_text())[name]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = build_task_images.stage_datasets(
                renderer.problem_directory(
                    build_task_images.TASK_ROOT / "problems", name
                ),
                root,
                definition,
            )
            self.assertEqual(
                (root / "Dockerfile").read_text(),
                "FROM scratch\nCOPY datasets/ /datasets/\n",
            )
            for path, body in expected.items():
                self.assertEqual((root / path).read_bytes(), body)
            for variant in (
                "valid",
                "wrong-body",
                "missing",
                "two-layers",
                "symlink",
                "duplicate",
            ):
                layer_bytes = io.BytesIO()
                with tarfile.open(fileobj=layer_bytes, mode="w") as layer:
                    for index, (path, body) in enumerate(expected.items()):
                        if variant == "missing" and index == 0:
                            continue
                        member = tarfile.TarInfo(path)
                        body = b"x" * len(body) if variant == "wrong-body" else body
                        member.size = len(body)
                        if variant == "symlink":
                            member.type = tarfile.SYMTYPE
                        layer.addfile(member, io.BytesIO(body))
                        if variant == "duplicate":
                            layer.addfile(member, io.BytesIO(body))
                manifest = [
                    {"Layers": ["layer.tar"] * (2 if variant == "two-layers" else 1)}
                ]
                archive = root / "image.tar"
                with tarfile.open(archive, "w") as saved:
                    for path, body in (
                        ("layer.tar", layer_bytes.getvalue()),
                        ("manifest.json", json.dumps(manifest).encode()),
                    ):
                        member = tarfile.TarInfo(path)
                        member.size = len(body)
                        saved.addfile(member, io.BytesIO(body))
                if variant == "valid":
                    build_task_images.verify_dataset_image(archive, expected)
                else:
                    with (
                        self.subTest(variant=variant),
                        self.assertRaises(build_task_images.BuildFailure),
                    ):
                        build_task_images.verify_dataset_image(archive, expected)

    def test_native_sources_are_unchanged_and_gpu_argument_is_required(self):
        path = (
            renderer.problem_directory(
                build_task_images.TASK_ROOT / "problems", build_task_images.SPARSE_VIDEO
            )
            / "verifier-gpu/Dockerfile"
        )
        before = path.read_bytes()
        for value in (None, "", "ubuntu:24.04"):
            with self.assertRaises(build_task_images.BuildFailure):
                build_task_images.freeze_bases(path, {}, value)
        with mock.patch.object(
            build_task_images, "pull_digest_reference", side_effect=lambda x: x
        ):
            build_task_images.freeze_bases(
                path, {}, "docker.io/example/ssh@sha256:" + "a" * 64
            )
        self.assertEqual(path.read_bytes(), before)

    def test_push_is_opt_in_and_after_dataset_verification(self):
        definitions = json.loads(renderer.DEFINITIONS.read_text())
        for fails in (False, True):
            with (
                mock.patch.object(
                    renderer, "render", return_value=definitions["batch_local_counting"]
                ),
                mock.patch.object(build_task_images, "freeze_bases", return_value=[]),
                mock.patch.object(build_task_images, "run") as run,
                mock.patch.object(
                    build_task_images,
                    "verify_dataset_image",
                    side_effect=build_task_images.BuildFailure() if fails else None,
                ),
            ):
                if fails:
                    with self.assertRaises(build_task_images.BuildFailure):
                        build_task_images.build_problem(
                            "batch_local_counting", "example", push=True, bases={}
                        )
                else:
                    build_task_images.build_problem(
                        "batch_local_counting", "example", push=True, bases={}
                    )
                self.assertEqual(
                    sum(
                        c.args[0][:2] == ["docker", "push"] for c in run.call_args_list
                    ),
                    0 if fails else 3,
                )

    def test_selections_manifest_and_prefix(self):
        names = [
            task["name"]
            for task in json.loads(build_task_images.TASK_CATALOG.read_text())["tasks"]
        ]
        definitions = json.loads(renderer.DEFINITIONS.read_text())
        cpu = [name for name in names if "gpu" not in definitions[name]]
        for selection in (
            [],
            ["--problem", "cpu"],
            ["--problem", "join_tree_rewriting"],
        ):
            with (
                tempfile.TemporaryDirectory() as directory,
                mock.patch.object(
                    build_task_images, "build_problem", return_value={}
                ) as build,
            ):
                path = Path(directory) / "manifest.json"
                build_task_images.main([*selection, "--manifest", str(path)])
                self.assertEqual(
                    [call.args[0] for call in build.call_args_list],
                    names
                    if not selection
                    else cpu
                    if selection[-1] == "cpu"
                    else [selection[-1]],
                )
                self.assertEqual(
                    json.loads(path.read_text())["platform"], "linux/amd64"
                )
                self.assertTrue(
                    all(call.args[1] == "joyemang33" for call in build.call_args_list)
                )
        for value in ("repo;echo", "repo:latest", "UPPER/repo", "repo name"):
            self.assertIsNone(build_task_images.IMAGE_PREFIX_RE.fullmatch(value))

    def test_native_copy_sources_exist_and_agents_exclude_judge(self):
        for name in renderer.catalog_task_names():
            root = renderer.problem_directory(
                build_task_images.TASK_ROOT / "problems", name
            )
            for directory in ("agent-base", "verifier", "agent-gpu", "verifier-gpu"):
                path = root / directory / "Dockerfile"
                if not path.exists():
                    continue
                # Use the same COPY option/source/visibility rules as the
                # official native layout gate (for example COPY --chmod=0644).
                native_layout.check_dockerfile(root, directory)

    def test_native_copy_flags_keep_source_and_agent_visibility_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "native_copy_flags"
            public = root / "agent-base/frontier/task"
            public.mkdir(parents=True)
            (public / "README.md").write_text("Public task inputs.\n")
            verifier = root / "verifier"
            verifier.mkdir()
            (verifier / "verify.py").write_text("pass\n")
            private = root / "judge"
            private.mkdir()
            (private / "private.py").write_text("Private judge.\n")
            agent_recipe = root / "agent-base/Dockerfile"
            public_copy = (
                'COPY --chmod=0755 "agent-base/frontier/task" /frontier/task\n'
            )
            agent_recipe.write_text("FROM scratch\n" + public_copy)
            recipe = verifier / "Dockerfile"
            recipe.write_text(
                "FROM scratch\nCOPY --chown=65532:65532 --chmod=0644 "
                "verifier/verify.py /opt/verifier/verify.py\n"
            )
            with (
                mock.patch.object(
                    renderer, "catalog_task_names", return_value=[root.name]
                ),
                mock.patch.object(renderer, "problem_directory", return_value=root),
            ):
                self.test_native_copy_sources_exist_and_agents_exclude_judge()
                for copy_line, message in (
                    (
                        "COPY --chmod=0644 verifier/missing.py /opt/verifier/missing.py\n",
                        "missing verifier COPY source",
                    ),
                    (
                        "COPY --chmod=0644 ../private.py /opt/verifier/private.py\n",
                        "unsafe source or payload path",
                    ),
                ):
                    with self.subTest(copy_line=copy_line):
                        recipe.write_text("FROM scratch\n" + copy_line)
                        with self.assertRaisesRegex(
                            native_layout.LayoutFailure, message
                        ):
                            self.test_native_copy_sources_exist_and_agents_exclude_judge()
                recipe.write_text(
                    "FROM scratch\nCOPY --chmod=0644 verifier/verify.py /opt/verifier.py\n"
                )
                agent_recipe.write_text(
                    "FROM scratch\n"
                    + public_copy
                    + "COPY --chmod=0644 judge/private.py /frontier/task/private.py\n"
                )
                with self.assertRaisesRegex(
                    native_layout.LayoutFailure, "non-public Agent COPY source"
                ):
                    self.test_native_copy_sources_exist_and_agents_exclude_judge()


def lock_tree(root: Path) -> None:
    records = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": incremental.sha(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "upstream.lock.json"
    ]
    (root / "upstream.lock.json").write_text(
        json.dumps(
            {"problem_id": root.name, "file_count": len(records), "files": records}
        )
    )


class IncrementalImageTests(unittest.TestCase):
    def test_lfs_materialization_requires_exact_bytes_and_digest(self):
        body = b"local LFS payload\x00\xff"
        digest = hashlib.sha256(body).hexdigest()
        pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{digest}\nsize {len(body)}\n".encode()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current, target = root / "current", root / "target"
            current.write_bytes(b"x" * len(body))
            with self.assertRaisesRegex(
                incremental.IncrementalFailure, "missing or corrupt"
            ):
                incremental.materialize_lfs(pointer, target, current, root / "objects")
            self.assertFalse(target.exists())
            stored = root / "objects" / digest[:2] / digest[2:4] / digest
            stored.parent.mkdir(parents=True)
            stored.write_bytes(body)
            incremental.materialize_lfs(pointer, target, current, root / "objects")
            self.assertEqual(target.read_bytes(), body)
            stored.unlink()
            current.write_bytes(body)
            incremental.materialize_lfs(pointer, target, current, root / "objects")
            self.assertEqual(target.read_bytes(), body)

    def test_only_reviewed_source_changes_are_incremental(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = Path(directory) / "before/task"
            current = Path(directory) / "after/task"
            for name in (
                "judge/evaluator.py",
                "judge/reference.patch",
                "agent-base/frontier/task/README.md",
                "agent-base/frontier/task/repo/solver.py",
                "datasets/train.json",
                "verifier/Dockerfile",
            ):
                path = previous / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("original\n")
            lock_tree(previous)
            for name in (
                "judge/reference.patch",
                "agent-base/frontier/task/repo/solver.py",
                "datasets/train.json",
                "verifier/Dockerfile",
                "judge/unknown.py",
                "judge/SUMMARY.py",
            ):
                with self.subTest(path=name):
                    shutil.copytree(previous, current, dirs_exist_ok=True)
                    (current / name).write_text("changed\n")
                    with self.assertRaisesRegex(
                        incremental.IncrementalFailure, "unhandled incremental"
                    ):
                        incremental.changed_paths(previous, current)
                    shutil.rmtree(current)
            shutil.copytree(previous, current)
            (current / "judge/evaluator.py").chmod(0o755)
            with self.assertRaisesRegex(
                incremental.IncrementalFailure, "executable mode"
            ):
                incremental.changed_paths(previous, current)
            (current / "judge/evaluator.py").chmod(0o644)
            (current / "judge/reference.patch").unlink()
            with self.assertRaisesRegex(
                incremental.IncrementalFailure, "removed task files"
            ):
                incremental.changed_paths(previous, current)
            shutil.copyfile(
                previous / "judge/reference.patch", current / "judge/reference.patch"
            )
            (current / "judge/evaluator.py").write_text("corrected evaluator\n")
            (current / "judge/beat_spec.json").write_text("{}\n")
            (current / "judge/SUMMARY.md").write_text(
                "restored private task documentation\n"
            )
            (current / "judge/test_evaluator.py").write_text(
                "updated evaluator tests\n"
            )
            (current / "judge/test_beat_spec.py").write_text("new beat-spec tests\n")
            instructions = current / "agent-base/frontier/task/bin/README.md"
            instructions.parent.mkdir(parents=True)
            instructions.write_text("Formal answering: 18000 seconds\n")
            self.assertEqual(
                incremental.changed_paths(previous, current),
                [
                    "agent-base/frontier/task/bin/README.md",
                    "judge/SUMMARY.md",
                    "judge/beat_spec.json",
                    "judge/evaluator.py",
                    "judge/test_beat_spec.py",
                    "judge/test_evaluator.py",
                ],
            )

    def test_oci_layer_contains_exact_copy_targets_permissions_and_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = {
                "agent-base/frontier/task/README.md": "task statement",
                "agent-base/frontier/task/bin/README.md": "Formal answering: 18000 seconds",
                "agent-base/frontier/task/repo/README.md": "problem statement",
                "judge/evaluator.py": "evaluator",
                "judge/SUMMARY.md": "private source documentation",
                "judge/beat_spec.json": "{}",
                "verifier/beat_spec_feedback.py": "feedback",
            }
            for name, value in files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(value)
                path.chmod(0o755 if name == "judge/evaluator.py" else 0o644)
            recipes = {
                "agent-base": "COPY agent-base/frontier/task /frontier/task\n",
                "agent-gpu": "COPY agent-base/frontier/task/repo /frontier/task/repo\n",
                "verifier": "COPY verifier/beat_spec_feedback.py /opt/frontier/beat_spec_feedback.py\n",
                "verifier-gpu": "COPY judge /opt/frontier/task\nCOPY agent-base/frontier/task/repo /opt/frontier/task/harbor/app\n"
                "COPY verifier/beat_spec_feedback.py /opt/frontier/beat_spec_feedback.py\n",
            }
            for role, recipe in recipes.items():
                (root / role).mkdir(exist_ok=True)
                (root / role / "Dockerfile").write_text(recipe)
            expected = {
                "agent": {
                    "/frontier/task/README.md": 0o444,
                    "/frontier/task/repo/README.md": 0o444,
                    "/frontier/task/bin/README.md": 0o444,
                },
                "agent-gpu": {"/frontier/task/repo/README.md": 0o444},
                "verifier": {"/opt/frontier/beat_spec_feedback.py": 0o644},
                "verifier-gpu": {
                    "/opt/frontier/task/SUMMARY.md": 0o444,
                    "/opt/frontier/task/evaluator.py": 0o555,
                    "/opt/frontier/task/beat_spec.json": 0o444,
                    "/opt/frontier/task/harbor/app/README.md": 0o444,
                    "/opt/frontier/beat_spec_feedback.py": 0o644,
                },
            }
            for role in recipes:
                component = "agent" if role == "agent-base" else role
                records = incremental.mapped_files(root, component, list(files))
                self.assertEqual(
                    {record["image_path"]: record["mode"] for record in records},
                    expected[component],
                )
                path = root / f"{role}.tar"
                first_digest = incremental.write_layer(root, records, path)
                self.assertEqual(
                    first_digest, incremental.write_layer(root, records, path)
                )
                with tarfile.open(path) as archive:
                    self.assertEqual(
                        archive.getnames(),
                        [record["image_path"].lstrip("/") for record in records],
                    )
                    for member, record in zip(archive, records, strict=False):
                        self.assertTrue(member.isfile())
                        self.assertEqual(
                            (member.mode, member.uid, member.gid, member.mtime),
                            (record["mode"], 0, 0, 0),
                        )
                        self.assertEqual(
                            archive.extractfile(member).read(),
                            files[record["source_path"]].encode(),
                        )

    def test_incremental_directory_copy_matches_native_trailing_slash_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "agent-base/frontier/task"
            source.mkdir(parents=True)
            (source / "README.md").write_text("updated instructions\n")
            (root / "runtime-context/wheelhouse").mkdir(parents=True)
            recipe = root / "agent-base/Dockerfile"
            changed = ["agent-base/frontier/task/README.md"]
            records = []
            for suffix in ("", "/"):
                recipe.write_text(
                    f"COPY agent-base/frontier/task{suffix} /frontier/task\n"
                    "COPY runtime-context/wheelhouse/ /opt/wheels\n"
                )
                records.append(incremental.mapped_files(root, "agent", changed))
                self.assertEqual(
                    incremental.mapped_files(root, "agent", ["upstream.lock.json"]),
                    [],
                )
            self.assertEqual(records[0], records[1])
            self.assertEqual(len(records[1]), 1)
            self.assertEqual(records[1][0]["image_path"], "/frontier/task/README.md")
            self.assertEqual(records[1][0]["mode"], 0o444)
            self.assertEqual(records[1][0]["sha256"], incremental.sha(source / "README.md"))

    def test_incremental_copy_rejects_unsafe_and_missing_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "agent-base/frontier/task"
            source.mkdir(parents=True)
            (source / "README.md").write_text("instructions\n")
            (root / "alias").symlink_to(source, target_is_directory=True)
            recipe = root / "agent-base/Dockerfile"
            for invalid in (
                "/etc",
                "agent-base/../agent-base/frontier/task",
                "agent-base//frontier/task",
                "alias/",
                "alias/README.md",
                "agent-base/frontier/task/README.md/",
                "missing/",
            ):
                with self.subTest(source=invalid):
                    recipe.write_text(f"COPY {invalid} /frontier/task\n")
                    with self.assertRaises(native_layout.LayoutFailure):
                        incremental.mapped_files(
                            root, "agent", ["agent-base/frontier/task/README.md"]
                        )

    def test_policy_only_release_accepts_unchanged_chmod_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "verifier").mkdir()
            (root / "verifier/Dockerfile").write_text(
                "COPY --chmod=0644 verifier/beat_spec_feedback.py /opt/frontier/beat_spec_feedback.py\n"
            )
            (root / "verifier/beat_spec_feedback.py").write_text(
                "# immutable runtime\n"
            )
            self.assertEqual(
                incremental.mapped_files(
                    root, "verifier", ["policy.json", "upstream.lock.json"]
                ),
                [],
            )
            with self.assertRaisesRegex(
                incremental.IncrementalFailure, "cannot override COPY --chmod"
            ):
                incremental.mapped_files(
                    root, "verifier", ["verifier/beat_spec_feedback.py"]
                )

    def test_published_image_must_preserve_runtime_and_complete_base_layer_chain(self):
        base_manifest = {"layers": [{"digest": "sha256:" + "a" * 64, "size": 3000}]}
        base_config = {
            "architecture": "amd64",
            "os": "linux",
            "created": "2026-09-01T00:00:00Z",
            "config": {
                "User": "10001:10001",
                "Entrypoint": ["/frontier/run"],
                "Env": ["OFFLINE=1"],
            },
            "rootfs": {"type": "layers", "diff_ids": ["sha256:" + "b" * 64]},
            "history": [{"created_by": "base"}],
        }
        diff_id = "sha256:" + "c" * 64
        manifest = {
            "layers": [
                *base_manifest["layers"],
                {"digest": "sha256:" + "d" * 64, "size": 2000},
            ]
        }
        config = copy.deepcopy(base_config)
        config["rootfs"]["diff_ids"].append(diff_id)
        config["history"].append({})
        incremental.check_appended_image(
            base_manifest, base_config, manifest, config, diff_id
        )
        incremental.check_appended_image(
            base_manifest, base_config, base_manifest, base_config, None
        )
        mutations = [
            (
                lambda m, c: m["layers"][0].update(digest="sha256:" + "0" * 64),
                "base layer chain",
            ),
            (lambda m, c: c["config"].update(User="root"), "image configuration"),
            (
                lambda m, c: c["config"].update(Entrypoint=["unexpected"]),
                "image configuration",
            ),
            (
                lambda m, c: c["rootfs"]["diff_ids"].__setitem__(
                    -1, "sha256:" + "0" * 64
                ),
                "verified source layer",
            ),
            (lambda m, c: c["history"][0].update(created_by="replacement"), "history"),
        ]
        for mutate, error in mutations:
            with self.subTest(error=error):
                changed_manifest, changed_config = (
                    copy.deepcopy(manifest),
                    copy.deepcopy(config),
                )
                mutate(changed_manifest, changed_config)
                with self.assertRaisesRegex(incremental.IncrementalFailure, error):
                    incremental.check_appended_image(
                        base_manifest,
                        base_config,
                        changed_manifest,
                        changed_config,
                        diff_id,
                    )

    def test_versioned_dataset_retag_is_idempotent_and_rejects_collision(self):
        import import_task

        digest = "sha256:" + "a" * 64
        dataset = {
            "repository": "docker.io/owner/fcs2-task-test-datasets",
            "tag": "source-" + "b" * 64,
            "digest": digest,
        }
        plan = {
            "source": {"source_set_sha256": "sha256:" + "b" * 64},
            "components": {},
            "dataset_image": dataset,
        }
        for existing in (None, digest, "sha256:" + "c" * 64):
            with (
                self.subTest(existing=existing),
                tempfile.TemporaryDirectory() as directory,
                mock.patch.object(
                    incremental.renderer,
                    "problem_directory",
                    return_value=Path(directory),
                ),
                mock.patch.object(
                    incremental.renderer,
                    "source_sha256",
                    return_value=plan["source"]["source_set_sha256"],
                ),
                mock.patch.object(
                    import_task,
                    "remote_digest",
                    side_effect=[existing, existing, digest],
                ),
                mock.patch.object(incremental, "command") as command,
            ):
                if existing not in (None, digest):
                    plan["components"] = {"agent": {}}
                    with self.assertRaisesRegex(
                        incremental.IncrementalFailure, "collision"
                    ):
                        incremental.publish_plan(Path(directory), "test", plan, "crane")
                    command.assert_not_called()
                else:
                    incremental.publish_plan(Path(directory), "test", plan, "crane")
                    if existing is None:
                        command.assert_called_once_with(
                            [
                                "crane",
                                "tag",
                                dataset["repository"] + "@" + digest,
                                dataset["tag"],
                            ],
                            cwd=Path(directory),
                        )
                    else:
                        command.assert_not_called()

    def test_no_registry_mutation_before_all_selected_sources_validate(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(incremental, "resolve_commit", return_value="a" * 40),
            mock.patch.object(
                incremental,
                "plan_problem",
                side_effect=[{}, incremental.IncrementalFailure("bad second task")],
            ),
            mock.patch.object(incremental, "publish_plan") as publish,
        ):
            with self.assertRaisesRegex(
                incremental.IncrementalFailure, "bad second task"
            ):
                build_task_images.main(
                    [
                        "--problem",
                        "cpu",
                        "--incremental-from",
                        "d3d21d1",
                        "--push",
                        "--manifest",
                        str(Path(directory) / "receipt.json"),
                    ]
                )
            publish.assert_not_called()

    def test_incremental_without_push_only_plans(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(incremental, "resolve_commit", return_value="a" * 40),
            mock.patch.object(incremental, "plan_problem", return_value={}),
            mock.patch.object(incremental, "publish_plan") as publish,
            mock.patch.object(build_task_images, "build_problem") as docker_build,
        ):
            receipt = Path(directory) / "receipt.json"
            build_task_images.main(
                [
                    "--problem",
                    "cpu",
                    "--incremental-from",
                    "d3d21d1",
                    "--manifest",
                    str(receipt),
                ]
            )
            self.assertEqual(json.loads(receipt.read_text())["status"], "planned")
            publish.assert_not_called()
            docker_build.assert_not_called()


class TaskImageMakeCompatibilityTests(unittest.TestCase):
    def run_make(self, *arguments, environment=None):
        env = {
            key: value
            for key, value in os.environ.items()
            if key
            not in {
                "TASK_PROBLEM",
                "TASK_IMAGE_PREFIX",
                "DEMO_PROBLEM",
                "DEMO_IMAGE_PREFIX",
                "MAKEFLAGS",
                "MFLAGS",
                "MAKEOVERRIDES",
            }
        }
        env.update(environment or {})
        # Echo the builder invocation: no Docker, registry, task or state writes.
        return subprocess.run(
            ["make", "--no-print-directory", "-s", "PYTHON=echo", *arguments],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )

    def test_new_target_and_defaults(self):
        result = self.run_make("build-task-images")
        self.assertEqual((result.returncode, result.stderr), (0, ""))
        self.assertEqual(
            result.stdout.strip(),
            "scripts/build_task_images.py --problem all --image-prefix joyemang33",
        )

    def test_legacy_target_and_variables_preserve_arguments_and_warn_once(self):
        for target in ("build-demo-images", "build-task-images"):
            for via_environment in (False, True):
                with self.subTest(target=target, environment=via_environment):
                    old = {"DEMO_PROBLEM": "cpu", "DEMO_IMAGE_PREFIX": "example"}
                    arguments = (
                        [] if via_environment else [f"{k}={v}" for k, v in old.items()]
                    )
                    result = self.run_make(
                        target, *arguments, environment=old if via_environment else None
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(
                        result.stdout.strip(),
                        "scripts/build_task_images.py --problem cpu --image-prefix example",
                    )
                    self.assertEqual(len(result.stderr.splitlines()), 1)
                    self.assertIn("deprecated:", result.stderr)
        result = self.run_make("build-demo-images")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(result.stderr.splitlines()), 1)

    def test_new_variables_win_and_both_targets_build_once(self):
        result = self.run_make(
            "build-demo-images",
            "build-task-images",
            "DEMO_PROBLEM=all",
            "DEMO_IMAGE_PREFIX=old",
            "TASK_PROBLEM=cpu",
            "TASK_IMAGE_PREFIX=new",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.strip(),
            "scripts/build_task_images.py --problem cpu --image-prefix new",
        )
        self.assertEqual(len(result.stderr.splitlines()), 1)

    def test_builder_environment_alias_and_explicit_flag_precedence(self):
        for env, flags, prefix, warning in (
            ({"DEMO_IMAGE_PREFIX": "old"}, [], "old", True),
            (
                {"DEMO_IMAGE_PREFIX": "old", "TASK_IMAGE_PREFIX": "new"},
                [],
                "new",
                False,
            ),
            (
                {"DEMO_IMAGE_PREFIX": "old"},
                ["--image-prefix", "explicit"],
                "explicit",
                True,
            ),
        ):
            with (
                self.subTest(env=env, flags=flags),
                tempfile.TemporaryDirectory() as directory,
            ):
                with (
                    mock.patch.dict(os.environ, env, clear=True),
                    mock.patch.object(
                        build_task_images, "build_problem", return_value={}
                    ) as build,
                    mock.patch("sys.stdout", new_callable=io.StringIO),
                    mock.patch("sys.stderr", new_callable=io.StringIO) as stderr,
                ):
                    build_task_images.main(
                        [
                            "--problem",
                            "cpu",
                            "--manifest",
                            str(Path(directory) / "build.json"),
                            *flags,
                        ]
                    )
                self.assertTrue(build.called)
                self.assertTrue(
                    all(call.args[1] == prefix for call in build.call_args_list)
                )
                self.assertEqual(len(stderr.getvalue().splitlines()), int(warning))


if __name__ == "__main__":
    unittest.main()
