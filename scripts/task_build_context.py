#!/usr/bin/env python3
"""Write the docker build context of a task.toml task: a copy of the task plus the files
generated from task.toml (never committed). For building a task's images locally:

  scripts/task_build_context.py TASK_DIR OUT_DIR
  docker build -f OUT_DIR/docker/verifier.Dockerfile -t local/<task>-verifier OUT_DIR
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_task_images  # noqa: E402
import task_manifest  # noqa: E402

if len(sys.argv) != 3:
    sys.exit(__doc__)
root = Path(sys.argv[1]).resolve()
print(build_task_images.build_context(root, task_manifest.layout(root), Path(sys.argv[2]).resolve()))
