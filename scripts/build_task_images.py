#!/usr/bin/env python3
"""Build the bundled CPU and native GPU demo images locally.

This entrypoint deliberately has no AWS, Kubernetes, registry-login, or secret
handling.  A push is opt-in and relies on the operator's existing Docker
credential configuration without reading or copying it.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Sequence

import image_namespace
import render_task_definition as renderer
import source_tree
import task_layers
import task_manifest
import verifier_layout
from native_datasets import dataset_path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SPARSE_VIDEO = "sparse_video_attention_policy"
IMAGE_PREFIX_RE = re.compile(
    r"^(?:[a-z0-9]+(?:[._-][a-z0-9]+)*(?::[0-9]+)?/)*"
    r"[a-z0-9]+(?:[._-][a-z0-9]+)*$"
)
DIGEST_REFERENCE_RE = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")


class BuildFailure(RuntimeError):
    """A demo image could not be built with frozen inputs."""


def run(
    command: Sequence[str], *, capture: bool = False, input_text: str | None = None
) -> str:
    result = subprocess.run(
        list(command),
        cwd=REPOSITORY_ROOT,
        check=False,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        text=True,
        input=input_text,
    )
    if result.returncode != 0:
        detail = result.stderr[-2000:].strip() if capture and result.stderr else ""
        raise BuildFailure(
            f"{command[0]} failed with exit {result.returncode}"
            + (f": {detail}" if detail else "")
        )
    return result.stdout.strip() if capture else ""


def pull_digest_reference(source: str) -> str:
    run(["docker", "pull", "--platform", "linux/amd64", source])
    if DIGEST_REFERENCE_RE.fullmatch(source):
        # A local image can have multiple RepoDigests. Preserve the exact
        # requested immutable identity, not an unrelated first alias.
        return source
    reference = run(
        ["docker", "image", "inspect", "--format", "{{index .RepoDigests 0}}", source],
        capture=True,
    )
    if DIGEST_REFERENCE_RE.fullmatch(reference) is None:
        raise BuildFailure(f"Docker did not resolve {source} to one immutable digest")
    return reference


def gpu_ssh_base() -> str:
    """The current shared gpu-tools image (tasks/shared/bases.json), which carries the GPU SSH
    bootstrap at the path legacy GPU Dockerfiles copy as FRONTIER_GPU_SSH_BASE_IMAGE; it
    replaces reading one task's published Agent GPU image from the catalog. A task.toml GPU
    task names gpu-tools by digest in its own Dockerfile instead."""
    try:
        reference = task_manifest.base_reference("gpu-tools")
    except (OSError, ValueError, task_manifest.ManifestError) as exc:
        raise BuildFailure("cannot resolve the shared gpu-tools image from bases.json") from exc
    if (
        not reference.startswith("docker.io/")
        or DIGEST_REFERENCE_RE.fullmatch(reference) is None
    ):
        raise BuildFailure(
            "GPU SSH bootstrap must be a public Docker Hub digest reference"
        )
    return reference


BASE_IMAGES = {"ubuntu:24.04", "buildpack-deps:noble", "python:3.11-slim-bookworm"}
# Exact reviewed public SDK used by MetaOpt; no general MCR registry approval.
REVIEWED_PUBLIC_BASES = {
    "mcr.microsoft.com/dotnet/sdk:8.0-bookworm-slim@sha256:3ee581addaa35c8c790f5f62441b189810522bb32e43a8db9d4283b3f649cc24",
}
SSH_BASE_ARGUMENT = "${FRONTIER_GPU_SSH_BASE_IMAGE}"
# A recipe names a shared image by build argument (task_manifest.SHARED_BASE_ARGUMENTS):
# `ARG VERIFIER_BASE` then `FROM ${VERIFIER_BASE}` (or `FROM ${GPU_TOOLS} AS frontier_gpu_tools`
# + COPY --from=<stage>). The build injects tasks/shared/bases.json.
SHARED_BASE_ARGUMENTS = task_manifest.SHARED_BASE_ARGUMENTS


def shared_base_argument(source: str) -> str | None:
    """The build argument a `FROM ${NAME}` names when NAME is a shared base, else None."""
    match = re.fullmatch(r"\$\{([A-Z_]+)\}", source)
    return match.group(1) if match and match.group(1) in SHARED_BASE_ARGUMENTS else None


def shared_base_reference(argument: str) -> str:
    """bases.json's digest-pinned Docker Hub reference for a shared base build argument."""
    try:
        reference = task_manifest.base_reference(SHARED_BASE_ARGUMENTS[argument])
    except (OSError, ValueError, task_manifest.ManifestError) as exc:
        raise BuildFailure(f"cannot resolve {argument} from tasks/shared/bases.json") from exc
    if not reference.startswith("docker.io/") or DIGEST_REFERENCE_RE.fullmatch(reference) is None:
        raise BuildFailure(f"bases.json {argument} must be a Docker Hub digest reference")
    return reference


def approved_base(source: str) -> bool:
    """Accept reviewed tags or digest-pinned Docker Hub names, including shorthand."""
    if source in REVIEWED_PUBLIC_BASES:
        return True
    name = source.removeprefix("docker.io/")
    repository = name.split("@", 1)[0].rsplit("/", 1)
    # A registry hostname/port must never be interpreted as a Hub namespace.
    if len(repository) == 2 and (
        "." in repository[0] or ":" in repository[0] or repository[0] == "localhost"
    ):
        return False
    if (
        re.fullmatch(
            r"(?:[a-z0-9_-]+/)?[a-z0-9._-]+(?::[A-Za-z0-9_.-]+)?"
            r"(?:@sha256:[0-9a-f]{64})?",
            name,
        )
        is None
    ):
        return False
    return name.removeprefix("library/") in BASE_IMAGES or bool(
        DIGEST_REFERENCE_RE.fullmatch(name)
    )


def dockerfile_bases(
    dockerfile: Path, build_args: dict[str, str] | None = None
) -> list[str]:
    """Parse and validate every FROM before either caller can pull or build. A FROM may
    name a build arg (`${NAME}`) the build passes, e.g. a test sandbox's
    TEST_SANDBOX_BASE; it is validated as the image it stands for."""
    build_args = dict(build_args or {})
    sources, stages = [], set()
    for line in dockerfile_instructions(dockerfile):
        # A global ARG default (before the first FROM) applies unless the build passes it.
        default = re.fullmatch(r"\s*ARG\s+([A-Za-z_][A-Za-z0-9_]*)=(\S*)\s*", line, re.IGNORECASE)
        if default is not None and not sources and not stages:
            build_args.setdefault(default.group(1), default.group(2))
        if re.match(r"^\s*FROM(?:\s|$)", line, re.IGNORECASE) is None:
            continue
        match = re.fullmatch(
            r"\s*FROM\s+(?:--platform=\S+\s+)?(\S+)"
            r"(?:\s+AS\s+([a-zA-Z0-9_.-]+))?\s*",
            line,
            re.IGNORECASE,
        )
        if match is None:
            raise BuildFailure(f"invalid FROM in {dockerfile}: {line.strip()}")
        source, alias = match.groups()
        source = re.sub(
            r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}",
            lambda name: build_args.get(name.group(1), name.group(0)),
            source,
        )
        if source.lower() not in stages:
            if (source not in {"scratch", SSH_BASE_ARGUMENT}
                    and shared_base_argument(source) is None and not approved_base(source)):
                raise BuildFailure(
                    f"unreviewed or private base image in {dockerfile}: {source}"
                )
            sources.append(source)
        if alias:
            stages.add(alias.lower())
    if not sources:
        raise BuildFailure(f"missing FROM in {dockerfile}")
    return sources


def dockerfile_instructions(dockerfile: Path):
    """Join instruction continuations, removing comments and heredoc bodies."""
    text = dockerfile.read_text(encoding="utf-8")
    # Parser directives precede everything else and can change the escape
    # character; this parser only understands the default backslash.
    for line in text.splitlines():
        directive = re.fullmatch(r"#\s*([A-Za-z]+)\s*=\s*(.*?)\s*", line.strip())
        if directive is None:
            break
        if directive.group(1).lower() == "escape" and directive.group(2) != "\\":
            raise BuildFailure(
                f"unsupported parser directive in {dockerfile}: {line.strip()}"
            )
    lines = iter(text.splitlines())
    instruction = ""
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        line = line.rstrip()
        if line.endswith("\\"):
            instruction += line[:-1]
            continue
        instruction += line
        yield instruction
        if re.match(r"^\s*(?:RUN|COPY|ADD)\s", instruction, re.IGNORECASE):
            # Docker recognizes whole heredoc words (with an optional file
            # descriptor). Preserve quotes/escapes until the marker is identified:
            # echo '<<EOF' and echo prefix<<EOF must not hide later instructions.
            words = re.findall(
                r"(?:[^\s'\"\\]+|'[^']*'|\"(?:\\.|[^\"\\])*\"|\\.)+", instruction
            )
            for word in words:
                match = re.fullmatch(r"\d*<<(?P<tabs>-?)(?P<word>[^<]+)", word)
                if match is None:
                    continue
                terminator = shlex.split(match.group("word"))[0]
                for body in lines:
                    if (
                        body.lstrip("\t") if match.group("tabs") else body
                    ) == terminator:
                        break
                else:
                    raise BuildFailure(
                        f"unterminated heredoc in {dockerfile}: {terminator}"
                    )
        instruction = ""
    if instruction:
        yield instruction


def sidecar_roles(definition: dict) -> list[str]:
    """Image roles behind the declared sidecars; several sidecars may share one."""
    profiles = definition.get("sidecars") or {}
    return sorted({profile.get("image", name) for name, profile in profiles.items()})


def image_names(
    prefix: str, problem: str, tag: str, sidecars: tuple[str, ...] | list[str] = ()
) -> dict[str, str]:
    names = {
        "agent": f"{prefix}/fcs2-task-{problem}-agent:{tag}",
        "verifier": f"{prefix}/fcs2-task-verifier-{problem}:{tag}",
        "agent-gpu": f"{prefix}/fcs2-task-{problem}-agent-gpu:{tag}",
        "verifier-gpu": f"{prefix}/fcs2-task-verifier-{problem}-gpu:{tag}",
        # CPU machines of command test sandboxes: the Agent and Verifier images with the shared
        # runner layer (tasks/shared/images/cpu-runner.Dockerfile), published beside them.
        "agent-cpu-runner": f"{prefix}/fcs2-task-{problem}-agent-runner:{tag}",
        "verifier-cpu-runner": f"{prefix}/fcs2-task-verifier-{problem}-runner:{tag}",
        # Every image, the dataset image included, is tagged by the task's source identity, so
        # an existing tag is the image this build would produce (build_problem reuses it).
        "datasets": f"{prefix}/fcs2-task-{problem}-datasets:{tag}",
    }
    # Declared sidecars are task-owned images too; Modal imports them beside
    # the GPU runner, so they follow the GPU image naming and source tag.
    for sidecar in sidecars:
        names[f"sidecar-{sidecar}"] = f"{prefix}/fcs2-task-{problem}-sidecar-{sidecar}:{tag}"
    return names


def freeze_bases(
    dockerfile: Path,
    bases: dict[str, str],
    ssh_base: str | None,
    build_args: dict[str, str] | None = None,
) -> list[str]:
    arguments = []
    for name, value in sorted((build_args or {}).items()):
        arguments.extend(["--build-arg", f"{name}={value}"])
    for source in dockerfile_bases(dockerfile, build_args):
        if source == "scratch":
            continue
        if source == "${FRONTIER_GPU_SSH_BASE_IMAGE}":
            if (
                not ssh_base
                or DIGEST_REFERENCE_RE.fullmatch(ssh_base) is None
                or not approved_base(ssh_base)
            ):
                raise BuildFailure(
                    "FRONTIER_GPU_SSH_BASE_IMAGE must be non-empty and digest pinned"
                )
            bases[source] = ssh_base
            arguments.extend(["--build-arg", f"FRONTIER_GPU_SSH_BASE_IMAGE={ssh_base}"])
            continue
        argument = shared_base_argument(source)
        if argument is not None:
            bases[source] = shared_base_reference(argument)
            if f"{argument}={bases[source]}" not in arguments:  # one per argument
                arguments.extend(["--build-arg", f"{argument}={bases[source]}"])
            continue
        if source not in bases:
            bases[source] = pull_digest_reference(source)
        # BuildKit image contexts override FROM resolution without editing the
        # Dockerfile or the task tree whose exact bytes define the release.
        if not DIGEST_REFERENCE_RE.fullmatch(source):
            arguments.extend(
                ["--build-context", f"{source}=docker-image://{bases[source]}"]
            )
    return arguments


def stage_datasets(root: Path, context: Path, definition: dict) -> dict[str, bytes]:
    members = {}
    extension = definition["dataset_extension"]
    for mode in ("train", "eval"):
        digest = definition["datasets"][f"{mode}_sha256"].removeprefix("sha256:")
        member = f"datasets/{mode}/sha256/{digest}.{extension}"
        body = dataset_path(root, mode, extension).read_bytes()
        if hashlib.sha256(body).hexdigest() != digest:
            raise BuildFailure(f"dataset selector changed: {mode}")
        path = context / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        members[member] = body
    (context / "Dockerfile").write_text("FROM scratch\nCOPY datasets/ /datasets/\n")
    return members


def verify_dataset_image(archive_path: Path, expected: dict[str, bytes]) -> None:
    with tarfile.open(archive_path) as archive:
        manifest_file = archive.extractfile("manifest.json")
        if manifest_file is None:
            raise BuildFailure("dataset image has no save manifest")
        manifest = json.load(manifest_file)
        if len(manifest) != 1 or len(manifest[0]["Layers"]) != 1:
            raise BuildFailure("dataset image must have exactly one layer")
        layer_file = archive.extractfile(manifest[0]["Layers"][0])
        if layer_file is None:
            raise BuildFailure("dataset image layer missing")
        with tarfile.open(fileobj=layer_file) as layer:
            observed = {}
            for member in layer:
                if member.isdir():
                    continue
                name = member.name.removeprefix("./")
                if not member.isfile() or name not in expected or name in observed:
                    raise BuildFailure(f"unexpected dataset layer member: {name}")
                if member.size != len(expected[name]):
                    raise BuildFailure(f"dataset member size differs: {name}")
                stream = layer.extractfile(member)
                assert stream is not None
                observed[name] = stream.read()
            if observed != expected:
                raise BuildFailure(
                    "dataset image member bodies differ from the selectors"
                )


def build_context(root: Path, paths: "task_manifest.Layout", scratch: Path) -> Path:
    """The docker build context: a copy of the task's git-tracked files only (source_tree;
    an untracked or ignored file under the task fails the build, so no stray __pycache__ or
    scratch file reaches an image), plus for a task.toml task the files generated from
    task.toml (never committed; the source identity binds their hashes)."""
    context = scratch
    if context.exists():
        shutil.rmtree(context)
    context.parent.mkdir(parents=True, exist_ok=True)
    source_tree.stage(root, context)
    if not paths.native:
        return context
    for relative, body in task_manifest.generated_files(root).items():
        target = context / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
    return context


# Run inside a GPU task's built control Verifier image: the shared verifier and the GPU adapter
# must both report the task.toml placement. A verifier base without _gpu_placement (fast_kv on
# v9) would silently judge on one runner.
PLACEMENT_CHECK = """
import sys
import fcs2_verifier, fcs2_verifier_gpu
expected = sys.argv[1]
reported = getattr(fcs2_verifier, "_gpu_placement", None)
assert reported is not None, "fcs2_verifier.py predates the GPU placements (no _gpu_placement)"
verifier, adapter = reported(), fcs2_verifier_gpu.placement()
assert verifier == adapter == expected, (
    f"placement: task.toml {expected!r}, fcs2_verifier {verifier!r}, fcs2_verifier_gpu {adapter!r}")
print(f"placement {expected}: fcs2_verifier and fcs2_verifier_gpu agree")
"""
# A CPU task whose judge leases Driver CPU machines (verifier.json `sandboxes`) cannot judge
# without the Driver: its smoke checks that the image's shared adapter reads exactly the
# command test sandboxes task.toml declares.
SANDBOX_CHECK = """
import json, sys
import fcs2_verifier_gpu
expected = json.loads(sys.argv[1])
reported = fcs2_verifier_gpu.sandbox_parameters()
assert reported == expected, f"sandboxes: task.toml {expected!r}, the image {reported!r}"
print("command test sandboxes " + ", ".join(sorted(reported)) + ": the image's adapter reads task.toml's")
"""
SMOKE_RECORD = REPOSITORY_ROOT / "config" / "verifier-smoke.json"
SMOKE_SCHEMA = "frontier.verifier-smoke/v1"
STARTER = "the starter (empty patch)"


def task_placement(root: Path) -> str | None:
    """The `[gpu.verifier] placement` a task.toml GPU task's Verifier must report: None when
    the Verifier is not the shared one (a legacy policy.json task, or a standalone Verifier with
    its own verifier.command), which has no shared GPU adapter to report it."""
    if not task_manifest.is_native(root):
        return None
    task = task_manifest.read_task(root)
    if task["verifier"]["command"] != task_manifest.defaults()["verifier"]["command"]:
        return None
    gpu = task.get("gpu", {}).get("verifier")
    return gpu.get("placement") if isinstance(gpu, dict) else None


def smoke_records() -> dict:
    """config/verifier-smoke.json: per task, the payload the smoke judged when it was recorded
    and its train raw_score and metric names in the production layout (the tolerance applies to
    every task), or
    `skip`: why the task's Verifier cannot judge offline on a build host (it needs the model
    gateway, or more dedicated CPUs than a build host has)."""
    try:
        document = json.loads(SMOKE_RECORD.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise BuildFailure(f"unreadable {SMOKE_RECORD.name}: {error}") from error
    if document.get("schema_version") != SMOKE_SCHEMA:
        raise BuildFailure(f"{SMOKE_RECORD.name} must have schema_version {SMOKE_SCHEMA}")
    return document


def smoke_payload(root: Path, definition: dict) -> tuple[str, bytes]:
    """The payload tar the smoke judges: the task's judge/reference.patch, else the empty patch
    (the unchanged starter), as the submission entrypoint member."""
    reference = root / "judge" / "reference.patch"
    label, patch = ("judge/reference.patch", reference.read_bytes()) if reference.is_file() \
        else (STARTER, b"")
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        member = tarfile.TarInfo(definition["submission"]["entrypoint"])
        member.size, member.mode, member.mtime = len(patch), 0o644, 0
        archive.addfile(member, io.BytesIO(patch))
    return label, stream.getvalue()


def check_smoke_result(problem: str, result: dict | None, label: str, payload: bytes,
                       records: dict) -> str:
    """The smoke verdict: the payload must be judged (status passed, not an edit-scope
    rejection) and, when the task has a record for this payload, return the recorded train
    metric names and score within the tolerance of the recorded production-layout score."""
    if not isinstance(result, dict):
        raise BuildFailure("the Verifier wrote no result.json")
    if result.get("status") != "passed":
        failure = result.get("failure") or {}
        raise BuildFailure(f"status {result.get('status')!r}: "
                           f"{failure.get('code', '')} {result.get('summary', '')}".strip())
    if "rejected" in result.get("metrics", {}):
        raise BuildFailure(f"{label} was rejected ({result['metrics']['rejected']}): "
                           f"{result.get('summary', '')}")
    score = result.get("raw_score")
    record = records["tasks"].get(problem)
    if record is None:
        return f"{label}: passed, raw_score {score} (no recorded score)"
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    if record["payload_sha256"] != digest:
        raise BuildFailure(f"{label} ({digest}) is not the payload recorded in "
                           f"{SMOKE_RECORD.name}; re-record with --record-smoke")
    keys = sorted(result.get("metrics", {}))
    if keys != record["train_metric_keys"]:
        # A starter that scores zero either way (metaopt) still loses its feedback metrics
        # when the evaluator falls into a failure path (build_failed).
        raise BuildFailure(f"{label}: train metrics {keys}, recorded "
                           f"{record['train_metric_keys']}: {result.get('summary', '')}")
    expected = record["train_raw_score"]
    tolerance = max(records["tolerance"]["abs"], records["tolerance"]["rel"] * abs(expected))
    if not isinstance(score, (int, float)) or abs(score - expected) > tolerance:
        raise BuildFailure(f"{label}: raw_score {score}, recorded {expected} "
                           f"(tolerance {tolerance:g}): {result.get('summary', '')}")
    return f"{label}: passed, raw_score {score} (recorded {expected} +/- {tolerance:g})"


def record_smoke(problem: str, payload: bytes, result: dict) -> None:
    document = smoke_records()
    document["tasks"][problem] = {
        "payload_sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
        "train_raw_score": result["raw_score"],
        "train_metric_keys": sorted(result.get("metrics", {})),
    }
    document["tasks"] = dict(sorted(document["tasks"].items()))
    SMOKE_RECORD.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


# The registry tool (crane) the build asks whether a content-tagged image already exists.
CRANE = os.environ.get("FRONTIER_REGISTRY_COPY_TOOL", "crane")


def registry_digest(reference: str) -> str | None:
    """The digest the registry holds for `reference`, or None when it has no such tag. Image
    tags are derived from content (the task's source identity), so an existing tag is the image
    this build would produce."""
    result = subprocess.run([CRANE, "digest", reference], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def kubectl(*arguments: str, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(["kubectl", "-n", verifier_layout.SMOKE_NAMESPACE, *arguments],
                          capture_output=True, text=True, **kwargs)


def run_in_pod(image: str, verifier: dict, mode: str, work: Path, command: list[str] | None,
               *, timeout: int, label: str) -> subprocess.CompletedProcess:
    """Run `command` (default: the task's verifier.command) in the production Verifier layout on
    the cluster's Verifier nodes: a Pod from verifier_layout.pod_manifest (read-only root, an
    emptyDir /frontier, uid 65532, a user namespace, the task's own CPU and memory, offline),
    the inputs staged in `work`/input copied to /frontier/input, the command exec'd with the
    production Pod environment, and result.json and evidence copied back to `work`/output. The
    image must be pullable by digest."""
    name = f"verifier-smoke-{os.getpid()}-{abs(hash((label, image))) % 10**8}"
    manifest = verifier_layout.pod_manifest(
        name, image, verifier, mode, node_selector=verifier_layout.NODE_SELECTOR,
        labels={"app": "frontier-verifier-smoke"})
    applied = kubectl("apply", "-f", "-", input=json.dumps(manifest))
    if applied.returncode != 0:
        raise BuildFailure(f"cannot create the smoke Pod: {applied.stderr.strip()}")
    try:
        ready = kubectl("wait", "--for=condition=Ready", f"pod/{name}", "--timeout=1800s")
        if ready.returncode != 0:
            raise BuildFailure(f"the smoke Pod did not start: {ready.stderr.strip()}")
        kubectl("exec", name, "--", "mkdir", "-p", f"{verifier_layout.WORK}/input",
                f"{verifier_layout.WORK}/output", check=True)
        for path in sorted((work / "input").iterdir()):
            copied = kubectl("cp", str(path), f"{name}:{verifier_layout.WORK}/input/{path.name}")
            if copied.returncode != 0:
                raise BuildFailure(f"cannot copy {path.name}: {copied.stderr.strip()}")
        try:
            completed = kubectl("exec", name, "--", *(command or verifier["command"]),
                                timeout=timeout)
        except subprocess.TimeoutExpired as error:
            raise BuildFailure(f"the {mode} run exceeded {timeout} s") from error
        for output in ("result.json", "evidence.tar.zst"):
            kubectl("cp", f"{name}:{verifier_layout.WORK}/output/{output}",
                    str(work / "output" / output))
        return completed
    finally:
        kubectl("delete", "pod", name, "--wait=false")


# Run inside the built Verifier image (production layout): the whole shared verifier entry
# (fcs2_verifier.main: payload, edit scope, check_tree, result bounds, train feedback / eval
# private metrics, result and evidence writing) with the task's evaluator replaced by a
# recorded evaluator output. For a task whose evaluator needs a GPU, this is the only local
# run of the result path its real output takes on the platform.
FIXTURE_CHECK = """
import json, sys
from pathlib import Path
import fcs2_verifier
recorded = json.loads(Path(sys.argv[1]).read_text())
load = fcs2_verifier.load_evaluator
def load_evaluator(task_dir):
    check_tree = load(task_dir)[1]
    def evaluate(tree):
        return (recorded["raw_score"], recorded["score_unbounded"], recorded["message"],
                recorded["metrics"])
    return evaluate, check_tree
fcs2_verifier.load_evaluator = load_evaluator
parameters = json.loads(Path("/opt/frontier/verifier.json").read_text())
raise SystemExit(fcs2_verifier.main(Path("/opt/frontier/task"), parameters["entrypoint"],
                                    sandboxes="sandboxes" in parameters))
"""
FIXTURES = REPOSITORY_ROOT / "tests" / "fixtures" / "verifier-metrics"
# fcs2_verifier._bounded refuses metrics nested deeper than this (the top dict is level 0).
MAX_METRICS_DEPTH = 6


def metrics_fixtures(problem: str) -> dict[str, dict]:
    """tests/fixtures/verifier-metrics/<task>/<mode>.json: an evaluator output (raw_score,
    score_unbounded, message, metrics) recorded from a real verification of the task."""
    fixtures = {}
    for mode in ("train", "eval"):
        path = FIXTURES / problem / f"{mode}.json"
        if path.is_file():
            fixture = json.loads(path.read_text(encoding="utf-8"))
            if not {"raw_score", "score_unbounded", "message", "metrics"} <= set(fixture):
                raise BuildFailure(f"{path} is not an evaluator output")
            fixtures[mode] = fixture
    return fixtures


def metrics_depth(value: object, level: int = 0) -> int:
    if isinstance(value, dict):
        return max([metrics_depth(item, level + 1) for item in value.values()] or [level])
    if isinstance(value, list):
        return max([metrics_depth(item, level + 1) for item in value] or [level])
    return level


def evidence_metrics(path: Path) -> dict | None:
    """The evaluator metrics the shared verifier kept in its evidence archive."""
    try:
        import zstandard

        body = zstandard.ZstdDecompressor().decompress(path.read_bytes(),
                                                      max_output_size=1 << 27)
        with tarfile.open(fileobj=io.BytesIO(body)) as archive:
            member = archive.extractfile("metrics.json")
            return json.load(member).get("metrics") if member else None
    except (ImportError, OSError, ValueError, KeyError, tarfile.TarError):
        return None


def depth_note(metrics: dict | None) -> str:
    if not isinstance(metrics, dict):
        return ""
    depth = metrics_depth(metrics)
    return (f"; metrics depth {depth} of {MAX_METRICS_DEPTH}"
            + (" (AT THE LIMIT)" if depth >= MAX_METRICS_DEPTH else ""))


def run_in_layout(image: str, problem: str, definition: dict, work: Path, mode: str,
                  command: list[str] | None = None) -> tuple[dict | None, dict | None, str]:
    """Run `command` (default: the task's verifier.command) in the production layout on the
    inputs staged in `work`; return result.json, the evidence metrics and the end of the
    Verifier's stderr (the private evaluator exception tail a failed verdict needs)."""
    verifier = definition["verifier"]
    timeout = definition["quota"]["eval_timeout_seconds"] + 300
    completed = run_in_pod(image, verifier, mode, work, command, timeout=timeout,
                           label=f"{problem}-{mode}")
    try:
        result = json.loads((work / "output" / "result.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        result = None
    if completed.returncode != 0 and result is None:
        raise BuildFailure(f"the Verifier exited {completed.returncode}: "
                           f"{completed.stderr[-2000:].strip()}")
    return (result, evidence_metrics(work / "output" / "evidence.tar.zst"),
            completed.stderr[-2000:].strip())


def fixture_smoke(image: str, problem: str, root: Path, definition: dict, directory: Path,
                  ) -> list[str]:
    """Each recorded evaluator output through the shared verifier's result path in the image."""
    verdicts = []
    for mode, fixture in metrics_fixtures(problem).items():
        work = directory / f"fixture-{mode}"
        try:
            label, payload = smoke_payload(root, definition)
            dataset = dataset_path(root, mode, definition["dataset_extension"]).read_bytes()
            verifier_layout.stage_work(work, payload, dataset)
            (work / "input" / "fixture.json").write_text(json.dumps(fixture))
            result, metrics, log = run_in_layout(
                image, problem, definition, work, mode,
                command=["/opt/frontier/runtime-python", "-c", FIXTURE_CHECK,
                         f"{verifier_layout.WORK}/input/fixture.json"])
            try:
                check_smoke_result(problem, result, f"recorded {mode} metrics", payload,
                                   {"tasks": {}})
            except BuildFailure as error:
                raise BuildFailure(f"{label} with the recorded {mode} evaluator output "
                                   f"({FIXTURES.name}/{problem}/{mode}.json): {error}"
                                   + (f"\n{log}" if log else "")) from error
            verdicts.append(f"recorded {mode} metrics: passed{depth_note(metrics)}")
        finally:
            if work.exists():
                shutil.rmtree(work, ignore_errors=True)
    return verdicts


def verifier_smoke(image: str, problem: str, root: Path, definition: dict, *,
                   record: bool = False) -> str:
    """Fail unless the Verifier image (pushed: `image` names its digest) works in the production
    layout on the production Verifier nodes (run_in_pod: read-only root, an empty /frontier of
    its own, uid 65532, the Driver's environment, no network). A CPU task's Verifier judges its smoke payload in train mode through the whole
    shared verifier (edit scope, evaluator, result bounds); a GPU task's control image (it
    reaches the GPU only through the Driver) must report the task.toml placement. Any task's
    recorded evaluator outputs (tests/fixtures/verifier-metrics) also go through the shared
    verifier's result path in the image. A legacy policy.json GPU task has none of these."""
    verifier = definition["verifier"]
    gpu = definition.get("gpu", {}).get("verifier", {}).get("max_gpu_instances", 0) > 0
    sandboxes = (task_manifest.verifier_parameters(task_manifest.read_task(root)).get("sandboxes")
                 if not gpu and "verifier_cpu" in definition else None)
    with tempfile.TemporaryDirectory(prefix="frontier-verifier-smoke-",
                                     ignore_cleanup_errors=True) as directory:
        work = Path(directory) / "frontier"
        try:
            if gpu:
                placement = task_placement(root)
                if placement is None:
                    return ("no control-image placement check: the task's Verifier does not "
                            "use the shared GPU adapter (legacy policy.json or standalone)")
                verifier_layout.stage_work(work, b"", b"")
                command = ["/opt/frontier/runtime-python", "-c", PLACEMENT_CHECK, placement]
                try:
                    completed = run_in_pod(image, verifier, "train", work, command,
                                           timeout=1800, label=f"{problem}-placement")
                    if completed.returncode != 0:
                        raise BuildFailure(f"exit {completed.returncode}: "
                                           f"{completed.stderr[-2000:].strip()}")
                    verdicts = [completed.stdout.strip()]
                except BuildFailure as error:
                    raise BuildFailure(f"{image} does not report the task.toml GPU placement "
                                       f"{placement!r} in the production layout: {error}") \
                        from error
            elif sandboxes is not None:
                verifier_layout.stage_work(work, b"", b"")
                command = ["/opt/frontier/runtime-python", "-c", SANDBOX_CHECK,
                           json.dumps(sandboxes)]
                completed = run_in_pod(image, verifier, "train", work, command,
                                       timeout=1800, label=f"{problem}-sandboxes")
                if completed.returncode != 0:
                    raise BuildFailure(f"{image} does not read task.toml's command test sandboxes "
                                       f"in the production layout: exit {completed.returncode}: "
                                       f"{completed.stderr[-2000:].strip()}")
                verdicts = [completed.stdout.strip()]
            else:
                records = smoke_records()
                label, payload = smoke_payload(root, definition)
                dataset = dataset_path(root, "train",
                                       definition["dataset_extension"]).read_bytes()
                verifier_layout.stage_work(work, payload, dataset)
                log = ""
                try:
                    result, metrics, log = run_in_layout(image, problem, definition, work,
                                                         "train")
                    if record:
                        check_smoke_result(problem, result, label, payload,
                                           {**records, "tasks": {}})
                        record_smoke(problem, payload, result)
                        verdict = f"{label}: recorded raw_score {result['raw_score']}"
                    else:
                        verdict = check_smoke_result(problem, result, label, payload,
                                                     records)
                except BuildFailure as error:
                    raise BuildFailure(f"{image} fails in the production layout: {error}"
                                       + (f"\n{log}" if log else "")) from error
                verdicts = [verdict + depth_note(metrics)]
        finally:
            if work.exists():
                shutil.rmtree(work, ignore_errors=True)
        try:
            verdicts += fixture_smoke(image, problem, root, definition, Path(directory))
        except BuildFailure as error:
            raise BuildFailure(f"{image} fails in the production layout: {error}") from error
    return "; ".join(verdicts)


def build_problem(
    problem: str,
    prefix: str,
    *,
    push: bool,
    bases: dict[str, str],
    root: Path | None = None,
    definition: dict | None = None,
) -> dict:
    """Build the task's images from its git-tracked sources, skipping each image whose
    content-derived tag the registry already holds (the image this build would produce). With
    `push`, push the built images, then smoke a newly built Verifier image by its pushed digest
    in the production layout (run_in_pod); without it, the smoke runs on the published catalog
    image (--smoke-catalog) once the image is pushed."""
    root = root or renderer.task_directory(problem)
    definition = definition or renderer.render(root, problem)
    sidecars = sidecar_roles(definition)
    names = image_names(prefix, problem, definition["image_tag"], sidecars)
    gpu = definition.get("gpu", {})
    agent_gpu = gpu.get("agent", {}).get("max_instances", 0) > 0
    verifier_gpu = gpu.get("verifier", {}).get("max_gpu_instances", 0) > 0
    paths = task_manifest.layout(root)
    gpu_roles = [
        role
        for role, enabled in (("agent-gpu", agent_gpu), ("verifier-gpu", verifier_gpu))
        if enabled
    ] + [f"sidecar-{sidecar}" for sidecar in sidecars]
    needs_ssh = any(
        SSH_BASE_ARGUMENT
        in dockerfile_bases(paths.dockerfile(role), paths.build_args(role))
        for role in gpu_roles
    )
    ssh_base = pull_digest_reference(gpu_ssh_base()) if needs_ssh else None
    built, reused = {}, {}
    context = build_context(root, paths, REPOSITORY_ROOT / ".generated/build-contexts" / problem)
    for directory, component in (
        ("agent-base", "agent"),
        ("verifier", "verifier"),
        ("agent-gpu", "agent-gpu"),
        ("verifier-gpu", "verifier-gpu"),
        *((f"sidecar-{sidecar}", f"sidecar-{sidecar}") for sidecar in sidecars),
    ):
        if (component == "agent-gpu" and not agent_gpu) or (
            component == "verifier-gpu" and not verifier_gpu
        ):
            continue
        if not paths.has_role(directory):
            raise BuildFailure(f"missing enabled image recipe: {paths.dockerfile(directory)}")
        digest = registry_digest(names[component])
        if digest is not None:
            reused[component] = digest
            continue
        dockerfile = context / paths.dockerfile(directory).relative_to(root)
        arguments = freeze_bases(
            dockerfile, bases, ssh_base, paths.build_args(directory)
        )
        run(
            [
                "docker",
                "build",
                "--provenance=false",
                "--platform",
                "linux/amd64",
                "-f",
                str(dockerfile),
                *arguments,
                "--tag",
                names[component],
                str(context),
            ]
        )
        built[component] = names[component]
    digest = registry_digest(names["datasets"])
    if digest is not None:
        reused["datasets"] = digest
    else:
        build_dataset_image(root, definition, names["datasets"])
        built["datasets"] = names["datasets"]
    if push:
        for image in built.values():
            run(["docker", "push", image])
    # CPU runner images: FROM the task's own Agent/Verifier image by its registry digest (the
    # reused one, or the one the push above gave) with the shared runner layer the build context
    # carries; freeze_bases pins gpu-tools from bases.json like every shared base. A newly built
    # task image without a push has no registry digest, so its runner cannot be built.
    runner_bases = {"agent-cpu-runner": "agent", "verifier-cpu-runner": "verifier"}
    for component in (task_manifest.cpu_runner_roles(task_manifest.read_task(root))
                      if task_manifest.is_native(root) else ()):
        digest = registry_digest(names[component])
        if digest is not None:
            reused[component] = digest
            continue
        base = runner_bases[component]
        base_digest = reused.get(base) or registry_digest(names[base])
        if base_digest is None:
            raise BuildFailure(f"{component} is built FROM the {base} image by its registry "
                               f"digest: build with --push")
        dockerfile = context / task_manifest.GENERATED / "runner" / "cpu-runner.Dockerfile"
        arguments = freeze_bases(dockerfile, bases, None, {
            "RUNNER_BASE": names[base].rsplit(":", 1)[0] + "@" + base_digest})
        run(["docker", "build", "--provenance=false", "--platform", "linux/amd64",
             "-f", str(dockerfile), *arguments, "--tag", names[component], str(context)])
        built[component] = names[component]
        if push:
            run(["docker", "push", names[component]])
    if push:
        if "verifier" in built:
            pushed = registry_digest(names["verifier"])
            if pushed is None:
                raise BuildFailure(f"cannot read back the pushed {names['verifier']}")
            reference = names["verifier"].rsplit(":", 1)[0] + "@" + pushed
            print(verifier_smoke(reference, problem, root, definition))
    return {
        "source": definition["source"],
        "datasets": definition["datasets"],
        "images": {**built, **{component: names[component] for component in reused}},
        "reused": reused,
    }


def build_dataset_image(root: Path, definition: dict, name: str) -> None:
    with tempfile.TemporaryDirectory(prefix="frontier-datasets-") as directory:
        context = Path(directory) / "context"
        context.mkdir()
        members = stage_datasets(root, context, definition)
        run(
            [
                "docker",
                "build",
                "--provenance=false",
                "--platform",
                "linux/amd64",
                "-f",
                str(context / "Dockerfile"),
                "--tag",
                name,
                str(context),
            ]
        )
        archive = Path(directory) / "datasets.tar"
        run(["docker", "image", "save", "--output", str(archive), name])
        verify_dataset_image(archive, members)


def smoke_catalog(problems: Sequence[str], *, record: bool = False) -> int:
    """The production-layout smoke on the published catalog Verifier images: a list of the
    tasks that are broken in production right now."""
    catalog = {task["name"]: task for task in task_layers.load_catalog()["tasks"]}
    failures, unchecked = [], []
    for problem in problems:
        image = catalog[problem]["verifier_image"]
        reference = image["repository"] + "@" + image["digest"]
        try:
            root = renderer.task_directory(problem)
            verdict = verifier_smoke(reference, problem, root, renderer.render(root, problem),
                                     record=record).strip()
            checked = not verdict.startswith("no control-image placement check")
            if not checked:
                unchecked.append(problem)
            print(f"{'pass' if checked else 'n/a '} {problem}: {verdict}",
                  flush=True)
        except (BuildFailure, OSError, ValueError) as error:
            failures.append(problem)
            print(f"FAIL {problem}: {error}", flush=True)
    print(f"{len(problems) - len(failures) - len(unchecked)} of {len(problems)} catalog Verifier "
          f"images pass the production-layout smoke, {len(unchecked)} without a shared-adapter "
          "check"
          + (": failing " + ", ".join(failures) if failures else ""))
    return 1 if failures else 0


def main(arguments: Sequence[str] | None = None) -> int:
    catalog_error = None
    try:
        catalog_problems = tuple(renderer.catalog_task_names())
    except ValueError as exc:
        # Help remains available in an incomplete checkout. A build still
        # requires valid membership and fails before any build side effects.
        catalog_problems, catalog_error = (), exc
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--problem",
        action="append",
        choices=("all", "cpu", *catalog_problems),
        default=[],
    )
    parser.add_argument(
        "--image-namespace",
        default=image_namespace.namespace(),
        help="<registry>/<namespace> to name the images under "
        "(default: config/image-namespace.json)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPOSITORY_ROOT / ".generated/native-build.json",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="push after verification; never performs docker login",
    )
    parser.add_argument(
        "--incremental-from",
        help="verify this previous Git commit and plan source-only OCI layers; --push publishes them",
    )
    parser.add_argument(
        "--crane",
        default=os.environ.get("FRONTIER_REGISTRY_COPY_TOOL", "crane"),
        help="crane executable: registry lookups of existing images and incremental publication",
    )
    parser.add_argument(
        "--smoke-catalog",
        action="store_true",
        help="build nothing: run the production-layout smoke on each selected task's "
        "published catalog Verifier image (pulled by digest) and report every failure",
    )
    parser.add_argument(
        "--record-smoke",
        action="store_true",
        help="with --smoke-catalog: record each passing train raw_score in "
        "config/verifier-smoke.json instead of comparing with it",
    )
    options = parser.parse_args(arguments)
    global CRANE
    CRANE = options.crane
    if catalog_error is not None:
        raise BuildFailure(str(catalog_error)) from catalog_error
    if IMAGE_PREFIX_RE.fullmatch(options.image_namespace) is None:
        raise BuildFailure("--image-namespace is not a bounded Docker repository prefix")
    if any(options.manifest.resolve().is_relative_to(layer.problems.resolve())
           for layer in task_layers.layers()):
        raise BuildFailure("build manifest must live outside task trees")
    requested = options.problem or ["all"]
    try:
        directories = renderer.catalog_problem_directories(catalog_problems)
        definitions = task_layers.load_definitions()
        for name, root in directories.items():
            if name not in definitions:
                raise ValueError(f"catalog task {name} has no rendered definition")
            if not isinstance(definitions[name], dict):
                raise ValueError(
                    f"catalog task {name} rendered definition must be an object"
                )
            if task_manifest.is_task_dir(root):
                policy = task_manifest.load_policy(root)
                if definitions[name].get("gpu") != policy.get("gpu"):
                    raise ValueError(
                        f"GPU definition disagrees with native policy: {name}"
                    )
            gpu = definitions[name].get("gpu", {})
            for role, quota in (
                ("agent", "max_instances"),
                ("verifier", "max_gpu_instances"),
            ):
                if (
                    gpu.get(role, {}).get(quota, 0) > 0
                    and not task_manifest.layout(root).has_role(f"{role}-gpu")
                ):
                    raise ValueError(
                        f"GPU definition disagrees with {role}-gpu/ directory: {name}"
                    )
        cpu = tuple(
            name
            for name in catalog_problems
            if not definitions[name]
            .get("gpu", {})
            .get("agent", {})
            .get("max_instances", 0)
            and not definitions[name]
            .get("gpu", {})
            .get("verifier", {})
            .get("max_gpu_instances", 0)
        )
    except (OSError, ValueError) as exc:
        raise BuildFailure(str(exc)) from exc
    problems = (
        catalog_problems
        if "all" in requested
        else tuple(
            dict.fromkeys(
                name
                for selection in requested
                for name in (cpu if selection == "cpu" else (selection,))
            )
        )
    )
    if options.record_smoke and not options.smoke_catalog:
        raise BuildFailure("--record-smoke records only with --smoke-catalog")
    if options.smoke_catalog:
        return smoke_catalog(problems, record=options.record_smoke)
    if options.incremental_from:
        from preview import images as incremental

        commit = incremental.resolve_commit(REPOSITORY_ROOT, options.incremental_from)
        directory = options.manifest.with_suffix(".layers")
        manifest = {
            "schema_version": "frontier.preview-incremental-build/v1",
            "platform": "linux/amd64",
            "previous_commit": commit,
            "status": "planned",
            "tasks": {},
        }
        # Every selected task must validate before the first registry mutation.
        for problem in problems:
            manifest["tasks"][problem] = incremental.plan_problem(
                REPOSITORY_ROOT, problem, options.image_namespace, commit, directory
            )
        options.manifest.parent.mkdir(parents=True, exist_ok=True)
        options.manifest.write_text(renderer.json_text(manifest), encoding="utf-8")
        for problem in problems:
            if options.push:
                incremental.publish_plan(
                    REPOSITORY_ROOT, problem, manifest["tasks"][problem], options.crane
                )
                options.manifest.write_text(
                    renderer.json_text(manifest), encoding="utf-8"
                )
            print("pushed" if options.push else "planned", problem)
        manifest["status"] = "published" if options.push else "planned"
        options.manifest.write_text(renderer.json_text(manifest), encoding="utf-8")
        return 0
    manifest = {
        "schema_version": "frontier.preview-native-build/v1",
        "platform": "linux/amd64",
        "bases": {},
        "tasks": {},
    }
    for problem in problems:
        manifest["tasks"][problem] = build_problem(
            problem,
            options.image_namespace,
            push=options.push,
            bases=manifest["bases"],
        )
        options.manifest.parent.mkdir(parents=True, exist_ok=True)
        options.manifest.write_text(renderer.json_text(manifest), encoding="utf-8")
        print(("pushed" if options.push else "built"), problem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
