"""Public debug entrypoint using the exact trusted batched adaptation."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

from .adapter import build_trusted_host, compile_cubin
from .protocol import batch_count

SOURCE = Path("src/block_gemm/KAMI/1d/fp64/block_gemm_1d_double_mma.cu")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--k", type=int, default=24)
    parser.add_argument("--active-ranks", type=int, default=1)
    parser.add_argument("--allocated-ranks", type=int, default=1)
    parser.add_argument("--input-profile", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0x4B414D4932563201)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()

    repo = args.repo.resolve()
    count = batch_count(args.m, args.n, args.k)
    harness_dir = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="kami-public-debug-") as raw:
        scratch = Path(raw)
        cubin = scratch / "candidate.cubin"
        rendered, compile_proc = compile_cubin(
            repo / SOURCE,
            cubin,
            m=args.m,
            n=args.n,
            k=args.k,
            active_ranks=args.active_ranks,
            allocated_ranks=args.allocated_ranks,
        )
        if compile_proc.returncode != 0:
            print(compile_proc.stderr, end="")
            return compile_proc.returncode
        host = scratch / "trusted_host"
        host_proc = build_trusted_host(harness_dir, host)
        if host_proc.returncode != 0:
            print(host_proc.stderr, end="")
            return host_proc.returncode
        print(
            "trusted batched adapter compiled: "
            f"entry=frontier_batched_block_gemm_entry batch={count} "
            f"NUM_ITER=1 beta=0.5 rendered_sha256={rendered.rendered_sha256}"
        )
        if args.compile_only:
            return 0
        command = [
            str(host),
            "--candidate", str(cubin),
            "--m", str(args.m), "--n", str(args.n), "--k", str(args.k),
            "--active-ranks", str(args.active_ranks),
            "--allocated-ranks", str(args.allocated_ranks),
            "--batch", str(count),
            "--profile", str(args.input_profile),
            "--seed", str(args.seed),
            "--repetitions", "1",
        ]
        return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
