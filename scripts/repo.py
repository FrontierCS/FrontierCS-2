"""Prepare pinned tools and task inputs for explicit repository workflows."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


class ToolError(RuntimeError):
    pass


def host_platform() -> str:
    system = {"Darwin": "darwin", "Linux": "linux"}.get(platform.system())
    arch = {
        "x86_64": "amd64",
        "AMD64": "amd64",
        "arm64": "arm64",
        "aarch64": "arm64",
    }.get(platform.machine())
    if not system or not arch:
        raise ToolError(
            f"unsupported host: {platform.system()} {platform.machine()}; see README.md"
        )
    return f"{system}-{arch}"


def cache_directory() -> Path:
    return Path(
        os.environ.get(
            "FRONTIER_TOOL_CACHE",
            str(
                Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
                / "frontier-preview-tools"
            ),
        )
    )


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@contextmanager
def cache_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def download(url: str, destination: Path) -> None:
    if not url.startswith("https://"):
        raise ToolError(f"tool downloads require HTTPS: {url}")
    print(f"Downloading {url}", file=sys.stderr, flush=True)
    request = Request(url)
    parts = urlsplit(url)
    if (
        parts.hostname == "ghcr.io"
        and parts.path.startswith("/v2/")
        and "/blobs/" in parts.path
    ):
        repository = parts.path[4:].split("/blobs/", 1)[0]
        token_url = "https://ghcr.io/token?" + urlencode(
            {"service": "ghcr.io", "scope": f"repository:{repository}:pull"}
        )
        # Public anonymous pull only. Never consult Docker credentials or persist tokens.
        with urlopen(token_url, timeout=30) as response:
            token = json.load(response)["token"]
        request.add_header("Authorization", "Bearer " + token)
    with urlopen(request, timeout=120) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def unpack(archive: Path, destination: Path, kind: str) -> None:
    if kind in {"tar.gz", "tar.xz"}:
        with tarfile.open(archive, "r:*") as source:
            source.extractall(destination, filter="data")
    elif kind == "zip":
        with zipfile.ZipFile(archive) as source:
            for entry in source.infolist():
                path = destination / entry.filename
                if not path.resolve().is_relative_to(destination.resolve()):
                    raise ToolError(f"unsafe archive member: {entry.filename}")
                mode = entry.external_attr >> 16
                if stat.S_ISLNK(mode):
                    target = source.read(entry).decode()
                    if (
                        not (path.parent / target)
                        .resolve()
                        .is_relative_to(destination.resolve())
                    ):
                        raise ToolError(f"unsafe archive link: {entry.filename}")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.symlink_to(target)
                else:
                    source.extract(entry, destination)
                    if mode & 0o777:
                        path.chmod(mode & 0o777)
    elif kind == "conda":
        import zstandard

        with zipfile.ZipFile(archive) as package:
            members = [
                name
                for name in package.namelist()
                if name.startswith("pkg-") and name.endswith(".tar.zst")
            ]
            if len(members) != 1:
                raise ToolError("Conda archive must contain one package payload")
            with (
                package.open(members[0]) as compressed,
                zstandard.ZstdDecompressor().stream_reader(compressed) as stream,
                tarfile.open(fileobj=stream, mode="r|") as source,
            ):
                source.extractall(destination, filter="data")
    elif kind == "file":
        shutil.copyfile(archive, destination / "binary")
    else:
        raise ToolError(f"unsupported tool archive format: {kind}")


class Toolchain:
    def __init__(
        self,
        root: Path = ROOT,
        *,
        cache: Path | None = None,
        host: str | None = None,
        offline: bool = False,
    ):
        self.root = root
        self.cache = cache if cache is not None else cache_directory()
        self.host = host if host is not None else host_platform()
        self.offline = offline
        self.manifest = json.loads((root / "config/toolchain.json").read_text())
        if self.manifest.get("schema_version") != 1:
            raise ToolError("unsupported config/toolchain.json schema")

    def artifact(self, name: str) -> dict:
        try:
            tool = self.manifest["tools"][name]
            return {"version": tool["version"], **tool["platforms"][self.host]}
        except KeyError as error:
            raise ToolError(f"no pinned {name} artifact for {self.host}") from error

    def location(self, name: str) -> Path:
        artifact = self.artifact(name)
        identity = hashlib.sha256(
            json.dumps(artifact, sort_keys=True).encode()
        ).hexdigest()
        return self.cache / name / artifact["version"] / self.host / identity

    def installed(self, name: str) -> Path | None:
        artifact = self.artifact(name)
        directory = self.location(name)
        executable = directory / artifact["executable"]
        receipt = directory / ".complete.json"
        if receipt.is_file() and executable.is_file():
            try:
                record = json.loads(receipt.read_text())
                if (
                    isinstance(record, dict)
                    and record.get("artifact") == artifact
                    and os.access(executable, os.X_OK)
                    and record.get("executable_sha256") == sha256(executable)
                    and all(
                        sha256(directory / path) == digest
                        for path, digest in record.get("files", {}).items()
                    )
                ):
                    return executable
            except (OSError, ValueError):
                pass
        return None

    def archive(self, name: str, artifact: dict) -> Path:
        digest = artifact["sha256"]
        downloads = self.cache / "downloads"
        downloads.mkdir(parents=True, exist_ok=True)
        archive = downloads / digest
        if not archive.is_file() or sha256(archive) != digest:
            if self.offline:
                raise ToolError(f"{name} is not cached; rerun ./x bootstrap online")
            with tempfile.TemporaryDirectory(
                prefix="download-", dir=downloads
            ) as scratch:
                incoming = Path(scratch) / "archive"
                download(artifact["url"], incoming)
                if sha256(incoming) != digest:
                    raise ToolError(
                        f"SHA-256 mismatch for {name}; downloaded artifact was not installed"
                    )
                incoming.replace(archive)
        return archive

    def ensure(self, name: str) -> Path:
        if executable := self.installed(name):
            return executable
        artifact = self.artifact(name)
        directory = self.location(name)
        with cache_lock(directory.parent / (directory.name + ".lock")):
            if executable := self.installed(name):
                return executable
            archive = self.archive(name, artifact)
            with tempfile.TemporaryDirectory(
                prefix="unpack-", dir=directory.parent
            ) as scratch:
                staged = Path(scratch) / "tool"
                staged.mkdir()
                unpack(archive, staged, artifact["format"])
                files = {}
                for dependency in artifact.get("files", []):
                    with tempfile.TemporaryDirectory(
                        prefix="dependency-", dir=directory.parent
                    ) as dependency_dir:
                        dependency_root = Path(dependency_dir)
                        unpack(
                            self.archive(name, dependency),
                            dependency_root,
                            dependency["format"],
                        )
                        target = staged / dependency["destination"]
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(dependency_root / dependency["member"], target)
                        files[dependency["destination"]] = sha256(target)
                executable = staged / artifact["executable"]
                if source := artifact.get("source_executable"):
                    executable.parent.mkdir(parents=True, exist_ok=True)
                    executable.symlink_to(
                        os.path.relpath(staged / source, executable.parent)
                    )
                if not executable.is_file() or not executable.resolve().is_relative_to(
                    staged.resolve()
                ):
                    raise ToolError(
                        f"{name} archive does not contain {artifact['executable']}"
                    )
                executable.chmod(executable.stat().st_mode | 0o755)
                (staged / ".complete.json").write_text(
                    json.dumps(
                        {
                            "artifact": artifact,
                            "executable_sha256": sha256(executable),
                            "files": files,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                if directory.exists():
                    shutil.rmtree(directory)
                staged.replace(directory)
            return directory / artifact["executable"]


TOOLS = ("python", "uv", "ruff", "git-lfs", "tar", "make", "shellcheck")
CONTRIBUTOR_TOOLS = ("tar", "make", "shellcheck")
# Existing application parsers own every argument after the workflow name.
WORKFLOWS = {
    "import-task": ("import_task.py", ("tar", "make"), ()),
    "validate": ("frontier_deploy.py", (), ("validate",)),
}


def run(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    stdout=None,
) -> None:
    print("+ " + shlex.join(command), file=sys.stderr, flush=True)
    subprocess.run(command, cwd=cwd, env=env, stdout=stdout, check=True)


def require_git() -> str:
    git = shutil.which("git")
    if git is None:
        raise ToolError("Git is a bootstrap host requirement; see README.md")
    return git


def lfs_missing(root: Path, lfs: Path) -> bool:
    result = subprocess.run(
        [str(lfs), "ls-files", "--json"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return any(not row["checkout"] for row in json.loads(result.stdout)["files"])


def hydrate_inputs(chain: Toolchain) -> None:
    require_git()
    lfs = chain.ensure("git-lfs")
    # Scope filters to these commands, preserving user config and worktree hooks.
    environment = os.environ.copy()
    count = int(environment.get("GIT_CONFIG_COUNT", "0"))
    filters = {
        "filter.lfs.process": f"{shlex.quote(str(lfs))} filter-process",
        "filter.lfs.clean": f"{shlex.quote(str(lfs))} clean -- %f",
        "filter.lfs.smudge": f"{shlex.quote(str(lfs))} smudge -- %f",
        "filter.lfs.required": "true",
    }
    for index, (key, value) in enumerate(filters.items(), start=count):
        environment[f"GIT_CONFIG_KEY_{index}"] = key
        environment[f"GIT_CONFIG_VALUE_{index}"] = value
    environment["GIT_CONFIG_COUNT"] = str(count + len(filters))
    if lfs_missing(chain.root, lfs):
        # Another worktree may already have fetched the required objects.
        run([str(lfs), "checkout"], cwd=chain.root, env=environment, stdout=sys.stderr)
        if lfs_missing(chain.root, lfs):
            if chain.offline:
                raise ToolError(
                    "Git LFS inputs are missing; run ./x bootstrap --profile contributor online"
                )
            run([str(lfs), "pull"], cwd=chain.root, env=environment, stdout=sys.stderr)
        if lfs_missing(chain.root, lfs):
            raise ToolError("Git LFS still contains pointer files after hydration")


def prepare(
    chain: Toolchain, names: tuple[str, ...] = (), *, task_inputs: bool = False
) -> dict[str, str]:
    directories = [str(Path(sys.executable).parent)]
    directories.extend(str(chain.ensure(name).parent) for name in names)
    if task_inputs:
        hydrate_inputs(chain)
        directories.append(str(chain.ensure("git-lfs").parent))
    environment = os.environ.copy()
    environment["PATH"] = os.pathsep.join(
        [*dict.fromkeys(directories), environment.get("PATH", os.defpath)]
    )
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def doctor(chain: Toolchain, profile: str) -> int:
    failures = []
    expected = (chain.root / ".python-version").read_text().strip()
    actual = ".".join(map(str, sys.version_info[:3]))
    print(f"Python: {actual} ({sys.executable})")
    if actual != expected:
        failures.append(
            f"expected Python {expected}; invoke ./x instead of a system interpreter"
        )
    for name in ("jsonschema", "pytest", "zstandard", "ruff"):
        try:
            print(f"{name}: {importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError:
            failures.append(
                f"{name} is missing; invoke ./x to synchronize the locked environment"
            )
    if profile == "contributor":
        try:
            print(f"Git: {require_git()}")
            for name in (*CONTRIBUTOR_TOOLS, "git-lfs"):
                executable = chain.installed(name)
                if executable is None:
                    failures.append(
                        f"{name} is not prepared; run ./x bootstrap --profile contributor"
                    )
                else:
                    print(f"{name}: {chain.artifact(name)['version']} ({executable})")
                    if name == "git-lfs" and lfs_missing(chain.root, executable):
                        failures.append(
                            "Git LFS inputs are missing; run ./x bootstrap --profile contributor"
                        )
        except (ToolError, subprocess.CalledProcessError, OSError, ValueError) as error:
            failures.append(str(error))
    for error in failures:
        print(f"FAIL: {error}", file=sys.stderr)
    return int(bool(failures))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="./x", description=__doc__, allow_abbrev=False
    )
    result.add_argument(
        "--offline",
        action="store_true",
        default=os.environ.get("FRONTIER_BOOTSTRAP_OFFLINE") == "1",
        help="prepare from cache only; forwarded commands may still use the network",
    )
    commands = result.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("bootstrap", "prepare pinned tools and selected inputs"),
        ("doctor", "diagnose the local environment without cloud access"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument(
            "--profile", choices=("tooling", "contributor"), default="contributor"
        )
    commands.add_parser(
        "lint-shell", help="run pinned ShellCheck on the shell launchers"
    )
    commands.add_parser("check", help="run all local contributor checks")
    commands.add_parser("fmt", help="apply safe Python lint fixes and formatting")
    command = commands.add_parser(
        "run", help="run a command with the selected managed environment"
    )
    command.add_argument(
        "--profile", choices=("tooling", "contributor"), default="contributor"
    )
    command.add_argument("arguments", nargs=argparse.REMAINDER)
    for name in (*TOOLS, *WORKFLOWS, "test"):
        kind = "managed tool" if name in TOOLS else "repository workflow"
        command = commands.add_parser(name, help=f"run {kind} {name}", add_help=False)
        command.add_argument("arguments", nargs=argparse.REMAINDER)
    return result


def forwarding_argv(arguments: list[str]) -> list[str]:
    for index, argument in enumerate(arguments):
        if not argument.startswith("-"):
            if argument in (*TOOLS, *WORKFLOWS, "test"):
                # The inserted separator is removed once; an original -- survives.
                return [*arguments[: index + 1], "--", *arguments[index + 1 :]]
            break
    return arguments


def dispatch_tool(chain: Toolchain, name: str, arguments: list[str]) -> None:
    if name == "python":
        executable = Path(sys.executable)
    elif name == "ruff":
        executable = Path(sys.executable).with_name("ruff")
    elif name == "uv":
        if not (uv := os.environ.get("FRONTIER_UV")):
            raise ToolError("invoke ./x to select the pinned uv")
        executable = Path(uv)
    else:
        executable = chain.ensure(name)
    environment = os.environ.copy()
    environment["PATH"] = os.pathsep.join(
        [
            str(Path(sys.executable).parent),
            str(executable.parent),
            environment.get("PATH", os.defpath),
        ]
    )
    # Raw tools inherit caller cwd, argv, streams, and exit status.
    os.execve(executable, [str(executable), *arguments], environment)


def workflow_command(name: str, arguments: list[str]) -> list[str]:
    if name == "test":
        return [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-p",
            "test_*.py",
            *arguments,
        ]
    script, _, prefix = WORKFLOWS[name]
    return [sys.executable, str(ROOT / "scripts" / script), *prefix, *arguments]


def check(chain: Toolchain) -> int:
    environment = prepare(chain, CONTRIBUTOR_TOOLS, task_inputs=True)
    gates = [
        ("Python lint", [[sys.executable, "-m", "ruff", "check"]]),
        ("Python formatting", [[sys.executable, "-m", "ruff", "format", "--check"]]),
        (
            "Shell lint",
            [
                [
                    str(chain.ensure("shellcheck")),
                    "python",
                    "x",
                    "scripts/bootstrap-uv.sh",
                ]
            ],
        ),
        (
            "Tests",
            [
                [sys.executable, "scripts/check_test_prerequisites.py"],
                workflow_command("test", []),
            ],
        ),
        ("Task layouts", [[sys.executable, "scripts/validate_task_layout.py"]]),
        (
            "Example configuration",
            [workflow_command("validate", ["--config", "config/site.example.json"])],
        ),
    ]
    if (ROOT / "config/site.json").is_file():
        gates.append(
            (
                "Local configuration",
                [workflow_command("validate", ["--config", "config/site.json"])],
            )
        )
    status = 0
    for name, commands in gates:
        print(f"Checking {name}", file=sys.stderr, flush=True)
        try:
            for command in commands:
                run(command, cwd=ROOT, env=environment)
        except (subprocess.CalledProcessError, OSError) as error:
            code = (
                error.returncode
                if isinstance(error, subprocess.CalledProcessError)
                else 1
            )
            status = status or max(code, 1)
            print(f"FAIL: {name}: {error}", file=sys.stderr, flush=True)
        else:
            print(f"PASS: {name}", file=sys.stderr, flush=True)
    return status


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    options = parser().parse_args(forwarding_argv(arguments))
    forwarded = getattr(options, "arguments", [])
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    try:
        if options.command in (*WORKFLOWS, "test"):
            command = workflow_command(options.command, forwarded)
            cwd = ROOT if options.command == "test" else None
            # Application help is available before tool or task-input preparation.
            if forwarded in (["--help"], ["-h"]):
                run(command, cwd=cwd)
                return 0
            chain = Toolchain(offline=options.offline)
            names = (
                ("tar", "make")
                if options.command == "test"
                else WORKFLOWS[options.command][1]
            )
            if options.command == "import-task" and {"--plan", "--check"}.intersection(
                forwarded
            ):
                names = ("tar",)
            environment = prepare(chain, names, task_inputs=True)
            if options.command == "test":
                run(
                    [sys.executable, str(ROOT / "scripts/check_test_prerequisites.py")],
                    cwd=ROOT,
                    env=environment,
                )
            elif options.command == "validate":
                run(
                    [sys.executable, str(ROOT / "scripts/validate_task_layout.py")],
                    env=environment,
                )
            run(command, cwd=cwd, env=environment)
            return 0
        chain = Toolchain(offline=options.offline)
        if options.command in TOOLS:
            dispatch_tool(chain, options.command, forwarded)
            return 0
        if options.command == "doctor":
            return doctor(chain, options.profile)
        if options.command == "check":
            return check(chain)
        if options.command == "fmt":
            for arguments in (("check", "--fix-only"), ("format",), ("check",)):
                run([sys.executable, "-m", "ruff", *arguments], cwd=ROOT)
            return 0
        if options.command == "lint-shell":
            run(
                [
                    str(chain.ensure("shellcheck")),
                    *(
                        str(ROOT / path)
                        for path in ("python", "x", "scripts/bootstrap-uv.sh")
                    ),
                ]
            )
            return 0
        if options.command == "run":
            if not forwarded:
                raise ToolError("run requires a command after --")
            contributor = options.profile == "contributor"
            environment = prepare(
                chain, CONTRIBUTOR_TOOLS if contributor else (), task_inputs=contributor
            )
            run(forwarded, env=environment)
            return 0
        if options.profile == "contributor":
            prepare(chain, CONTRIBUTOR_TOOLS, task_inputs=True)
            print("Contributor inputs: ready (Git LFS hydrated)", file=sys.stderr)
        print("Tooling environment: ready", file=sys.stderr)
        return 0
    except (ToolError, OSError, ValueError) as error:
        print(f"x: {error}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as error:
        return error.returncode if error.returncode > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
