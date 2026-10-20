"""Extract the one granted immutable suite; no dataset body lives in an image."""
import hashlib
import os
from pathlib import Path
import shutil
import tarfile


class HashingReader:
    def __init__(self, stream, check_deadline):
        self.stream = stream
        self.check_deadline = check_deadline
        self.digest = hashlib.sha256()

    def read(self, size):
        self.check_deadline()
        if size < 0 or size > 4 * 1024**2:
            raise ValueError("unbounded archive read")
        data = self.stream.read(size)
        self.digest.update(data)
        return data


def install(root, mode, config, members, check_deadline=lambda: None):
    descriptor = config["dataset_objects"][mode]
    path = Path(os.environ["FRONTIER_DATASET_PATH"])
    if not path.is_absolute() or path.is_symlink() or not path.is_file() or path.stat().st_size != descriptor["bytes"]:
        raise ValueError("selected dataset size/type mismatch")
    destination = root / "judge_data"
    destination.mkdir(mode=0o700)
    seen = set()
    try:
        with path.open("rb") as raw:
            reader = HashingReader(raw, check_deadline)
            with tarfile.open(fileobj=reader, mode="r|") as archive:
                for member in archive:
                    if member.name not in members or member.name in seen or not member.isfile():
                        raise ValueError("unexpected, duplicate or nonregular dataset member")
                    if member.pax_headers or member.size != members[member.name]["bytes"]:
                        raise ValueError("dataset member metadata mismatch")
                    parts = member.name.split("/")
                    if any(p in {"", ".", ".."} for p in parts) or "\\" in member.name:
                        raise ValueError("unsafe dataset member path")
                    output = destination / member.name
                    output.parent.mkdir(parents=True, exist_ok=True)
                    digest = hashlib.sha256()
                    with archive.extractfile(member) as source, output.open("xb") as target:
                        total = 0
                        while block := source.read(1024**2):
                            total += len(block)
                            if total > member.size: raise ValueError("oversized dataset member")
                            target.write(block); digest.update(block)
                    if total != member.size or digest.hexdigest() != members[member.name]["sha256"]:
                        raise ValueError("dataset member identity mismatch")
                    seen.add(member.name)
            while reader.read(1024**2): pass
            if "sha256:" + reader.digest.hexdigest() != descriptor["version"] or seen != set(members):
                raise ValueError("selected dataset body or coverage mismatch")
    except BaseException:
        shutil.rmtree(destination)
        raise
    return destination
