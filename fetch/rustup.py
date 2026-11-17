#!/usr/bin/env python3
"""Installs a Rust toolchain from static.rust-lang.org, each fetched byte pinned by a digest.

    rustup.py RUSTUP_VERSION RUSTUP_INIT_SHA256 TOOLCHAIN TOOLCHAIN_TREE_SHA256

`rustup-init` of RUSTUP_VERSION must have the digest, which is the file's SHA-256. It then
installs TOOLCHAIN with the minimal profile, where `RUSTUP_HOME` and `CARGO_HOME` say, and the
toolchain's directory must have the tree digest: the SHA-256 of one line for each file and link
below it, in path order, `<path> <x if executable, else -> <file's SHA-256, or link's target>`; no other
mode counts, since rustup creates some files with the umask of whoever runs it. The channel manifest
that rustup rewrites, `lib/rustlib/multirust-channel-manifest.toml`, is left out, since its bytes
vary from one install to the next. A mismatch fails and
says the digest it found, which is how to write one for a new toolchain.
"""

import hashlib
import os
import stat
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

HOST = "x86_64-unknown-linux-gnu"
CHANNEL_MANIFEST = "lib/rustlib/multirust-channel-manifest.toml"
URL = "https://static.rust-lang.org/rustup/archive/{version}/" + HOST + "/rustup-init"


def tree_digest(root):
    """The tree digest of the directory `root`, as the module doc says."""
    lines = []
    for directory, subdirectories, names in os.walk(root):
        subdirectories.sort()
        # `os.walk` lists a link to a directory among the subdirectories, and never enters it.
        links = [name for name in subdirectories if Path(directory, name).is_symlink()]
        for name in names + links:
            path = Path(directory, name)
            if path.relative_to(root).as_posix() == CHANNEL_MANIFEST:
                continue
            executable = "x" if stat.S_IMODE(path.lstat().st_mode) & 0o111 else "-"
            if path.is_symlink():
                what = "-> " + os.readlink(path)
            else:
                what = hashlib.sha256(path.read_bytes()).hexdigest()
            lines.append(f"{path.relative_to(root).as_posix()} {executable} {what}\n")
    return hashlib.sha256("".join(sorted(lines)).encode()).hexdigest()


def main():
    version, init_sha256, toolchain, wanted = sys.argv[1:]
    body = urllib.request.urlopen(URL.format(version=version), timeout=120).read()
    found = hashlib.sha256(body).hexdigest()
    if found != init_sha256:
        sys.exit(f"rustup-init {version} has the SHA-256 {found}, not {init_sha256}: pin the digest of the file you mean")
    with tempfile.TemporaryDirectory() as scratch:
        init = Path(scratch, "rustup-init")
        init.write_bytes(body)
        init.chmod(0o755)
        subprocess.run([init, "-y", "--no-modify-path", "--profile", "minimal",
                        "--default-toolchain", toolchain, "--default-host", HOST], check=True)
    directory = Path(os.environ["RUSTUP_HOME"], "toolchains", f"{toolchain}-{HOST}")
    found = tree_digest(directory)
    if found != wanted:
        sys.exit(f"the toolchain {toolchain} has the tree digest {found}, not {wanted}: pin the digest of the toolchain you mean")


if __name__ == "__main__":
    main()
