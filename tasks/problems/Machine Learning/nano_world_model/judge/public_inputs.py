"""Build the public Point Maze data a candidate's sampling machine receives (control image build).

Usage: public_inputs.py <judge_data> <output>. For each suite (dev, final) it writes
<output>/<suite>/subset.json (the suite manifest) and <output>/<suite>/point_maze/ in the native
layout: actions.pth and seq_lengths.pth unchanged (the action normalisation uses every record),
states.pth zeroed (the evaluation does not normalise or condition on states), and one
obses/episode_<n>.pth per episode of the suite holding only the clips' context frames
(start_frame); every other frame is zero. The native pipeline then encodes the same context
latents and actions as on the real data, and no scored frame is present.
"""

import json
import shutil
import sys
from pathlib import Path

import torch

FRAME_INTERVAL = 5   # configs/dataset/dino_wm/point_maze.yaml
NUM_FRAMES = 4       # one context frame and three scored frames per clip


def build(judge_data: Path, output: Path) -> None:
    source = judge_data / "point_maze"
    for suite in ("dev", "final"):
        manifest = judge_data / f"{suite}_subset.json"
        slices = json.loads(manifest.read_text(encoding="utf-8"))["slices"]
        context = {(s["traj_idx"], s["start_frame"]) for s in slices}
        scored = {(s["traj_idx"], s["start_frame"] + FRAME_INTERVAL * k)
                  for s in slices for k in range(1, NUM_FRAMES)}
        if context & scored:
            raise ValueError(f"{suite}: a context frame is a scored frame of another clip")
        target = output / suite / "point_maze"
        (target / "obses").mkdir(parents=True)
        shutil.copyfile(manifest, output / suite / "subset.json")
        for name in ("actions.pth", "seq_lengths.pth"):
            shutil.copyfile(source / name, target / name)
        states = torch.load(source / "states.pth", weights_only=True)
        torch.save(torch.zeros_like(states), target / "states.pth")
        for episode in sorted({index for index, _ in context}):
            name = f"episode_{episode:03d}.pth"
            frames = torch.load(source / "obses" / name, weights_only=True)
            public = torch.zeros_like(frames)
            for index, start in context:
                if index == episode:
                    public[start] = frames[start]
            torch.save(public, target / "obses" / name)


if __name__ == "__main__":
    build(Path(sys.argv[1]), Path(sys.argv[2]))
