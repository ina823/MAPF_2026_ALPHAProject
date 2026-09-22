"""NavHint IL policy -- thin wrapper around UNMODIFIED Alpha 1 logic.

No new algorithm, heuristic, fallback or action correction lives here. This
module only assembles two pieces of Alpha 1 code so that rollouts can be run
from one place (Alpha 1 provenance: origin/analysis == origin/IL):

  * ``predict``           <- scripts/eval_cbs_vs_il.py::predict (lines 60-71),
                             copied verbatim.
                             normalisation (goal_dir - gmean) / gstd
                             -> NavHintCNN((grid, goal)) -> argmax
  * ``NavHintPolicy.reset`` / ``.act``
                          <- scripts/eval_flow.py::rollout (lines 40-47):
                             ``dist = {aid: bfs_dist(grid, goal)}`` once per
                             episode, then every step the simulator's straight
                             ``goal_dir`` is overwritten by
                             ``make_goal_dir(dist[aid], position, goal, hint_mode)``
                             (``hint_mode`` = "flowdist" for cnn_navhint.pt).

Model loading is ``src.il.model_navhint.load_navhint`` unchanged (including its
``torch.load`` call). Positions are passed in by the caller (Alpha 1 reads
``sim._positions`` in eval_flow.py; the same is done in scripts/run_il.py).
"""

from __future__ import annotations

import numpy as np
import torch

from src.il.model_navhint import load_navhint
from src.il.nav_hint import bfs_dist, make_goal_dir


@torch.no_grad()
def predict(m, obs):
    # verbatim: scripts/eval_cbs_vs_il.py::predict
    acts = {}
    for aid, st in obs.items():
        g = torch.from_numpy(st["grid"]).float().unsqueeze(0)
        d = (torch.from_numpy(st["goal_dir"]).float().unsqueeze(0) - m["gmean"]) / m["gstd"]
        if m["mode"] == "mlp":
            logits = m["model"](torch.cat([g.reshape(1, -1), d], dim=1))
        else:
            logits = m["model"]((g, d))
        acts[aid] = int(logits.argmax(1))
    return acts


class NavHintPolicy:
    """load_navhint() + the eval_flow.py goal_dir overwrite + predict()."""

    def __init__(self, ckpt_path, device="cpu"):
        self.model = load_navhint(ckpt_path, device)   # dict: model/gmean/gstd/goal_dim/hint_mode/...
        self.hint_mode = self.model["hint_mode"]
        self._goals = {}
        self._dist = {}

    def reset(self, grid, goals):
        """Once per episode (eval_flow.py: ``dist = {aid: bfs_dist(grid, gd[aid]) for aid in gd}``)."""
        self._goals = goals
        self._dist = {aid: bfs_dist(grid, goals[aid]) for aid in goals}

    def act(self, obs, positions):
        """One step. ``obs`` is mutated in place exactly as eval_flow.py does."""
        for aid in obs:
            obs[aid]["goal_dir"] = np.asarray(
                make_goal_dir(self._dist[aid], positions[aid], self._goals[aid], self.hint_mode),
                dtype=np.float32)
        return predict(self.model, obs)
