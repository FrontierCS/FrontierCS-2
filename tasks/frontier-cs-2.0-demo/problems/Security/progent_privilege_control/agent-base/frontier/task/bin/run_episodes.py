#!/usr/bin/env python3
"""Run AgentDojo episodes locally with your policy code, through the judge's own harness.

Each episode is the undefended AgentDojo agent on the task model with your
``secagent`` governing every tool call, exactly as the verifier runs it, and is
billed to the run's ``explore`` key. Examples:

  run_episodes.py --suite banking --user-task user_task_0 --injection all
  run_episodes.py --suite slack --injection none --reference      # upstream Progent-LLM
  run_episodes.py --suite travel --user-task user_task_3 --injection injection_task_5 --reps 3

Rows (one JSON object per episode) go to --out; a per-suite summary is printed.
The local harness keeps AgentDojo's published attacker identities; final
evaluation draws new ones per evaluation.

The AgentDojo copy in this environment contains only development user tasks:
the held-out final tasks, and the combined development tasks built from one of
them, are not included. Every injection task is included.
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HARNESS = Path("/frontier/task/harness")
sys.path.insert(0, str(HARNESS))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tree", default="/workspace/repo", help="source tree with your secagent/ (default /workspace/repo)")
    parser.add_argument("--reference", action="store_true", help="run the pristine upstream tree instead")
    parser.add_argument("--suite", required=True, choices=["banking", "slack", "travel", "workspace"])
    parser.add_argument("--user-task", action="append", help="repeatable; default: every user task of the suite")
    parser.add_argument("--injection", action="append", default=None,
                        help="injection task id, 'all' or 'none' (repeatable; default: none)")
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--out", default="/workspace/episodes.jsonl")
    parser.add_argument("--key", default="explore", help="additional_keys name (default explore)")
    args = parser.parse_args()

    tasks = json.loads((HARNESS / "suite_tasks.json").read_text())[args.suite]
    users = args.user_task or tasks["user"]
    wanted = args.injection or ["none"]
    injections = []
    for item in wanted:
        if item == "none":
            injections.append(None)
        elif item == "all":
            injections.extend(tasks["injection"])
        else:
            injections.append(item)
    tree = "/frontier/task/repo" if args.reference else str(Path(args.tree).resolve())
    if os.environ.get("PROGENT_UNSANDBOXED") != "1" and not _landlock_available():
        print("note: Landlock ABI 6 is unavailable here; set PROGENT_UNSANDBOXED=1 to run the worker "
              "unconfined (the verifier always confines it)", file=sys.stderr)
        return 2

    import episodes
    jobs = [{"tag": "reference" if args.reference else "candidate", "tree": tree, "user_task": u, "injection_task": i}
            for _ in range(args.reps) for u in users for i in injections]
    rows = []
    with ProcessPoolExecutor(max_workers=args.concurrency, initializer=episodes.init_runner,
                             initargs=(args.suite, args.key)) as pool, open(args.out, "a") as out:
        for future in as_completed([pool.submit(episodes.run_episode, job) for job in jobs]):
            row = future.result()
            rows.append(row)
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(f"{row['user_task']:>14} {str(row['injection_task']):>18} utility={row['utility']} "
                  f"breached={row['breached']} tokens={row['tokens']} worker_failure={row['worker_failure']}")
    attacked = [r for r in rows if r["injection_task"]]
    benign = [r for r in rows if not r["injection_task"]]
    summary = {"episodes": len(rows), "breached": sum(bool(r["breached"]) for r in attacked),
               "utility_under_attack": sum(r["utility"] for r in attacked) / len(attacked) if attacked else None,
               "benign_utility": sum(r["utility"] for r in benign) / len(benign) if benign else None,
               "tokens": sum(r["tokens"] for r in rows)}
    print(json.dumps(summary))
    return 0


def _landlock_available():
    import ctypes
    try:
        return ctypes.CDLL(None, use_errno=True).syscall(444, 0, 0, 1) >= 6
    except OSError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
