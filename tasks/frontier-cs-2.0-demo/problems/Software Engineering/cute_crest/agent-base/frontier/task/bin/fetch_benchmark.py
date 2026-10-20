#!/usr/bin/env python3
"""Image-build step: fetch the development program's source by pinned URL.

Downloads the CREST distribution at its pinned commit, checks its SHA-256,
keeps benchmarks/grep-2.2 (GNU grep 2.2 with CREST's symbolic-input driver,
GNU GPL version 2 or later; see COPYING there) and discards the rest.
"""
import hashlib
from pathlib import Path
import sys
import tarfile
import urllib.request

URL = 'https://github.com/jburnim/crest/archive/3e5ececd3793c88f18af47853236af34ffde4569.tar.gz'
SHA256 = '3ccbe608a23f4ded6937ab2a1886298af0f0415539d230bbed70f1df26ce1dfc'
PREFIX = 'crest-3e5ececd3793c88f18af47853236af34ffde4569/benchmarks/grep-2.2/'


def main(destination):
    destination = Path(destination)
    with urllib.request.urlopen(URL, timeout=300) as response:
        body = response.read()
    if hashlib.sha256(body).hexdigest() != SHA256:
        raise SystemExit('source archive hash mismatch')
    archive = destination.parent / 'crest.tar.gz'
    destination.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(body)
    try:
        with tarfile.open(archive) as tar:
            members = []
            for member in tar.getmembers():
                if member.name.startswith(PREFIX) and len(member.name) > len(PREFIX):
                    member.name = member.name[len(PREFIX):]
                    members.append(member)
            tar.extractall(destination, members=members, filter='data')
    finally:
        archive.unlink()
    if not (destination / 'src/grep.c').is_file() or not (destination / 'COPYING').is_file():
        raise SystemExit('unexpected source layout')


if __name__ == '__main__':
    main(sys.argv[1])
