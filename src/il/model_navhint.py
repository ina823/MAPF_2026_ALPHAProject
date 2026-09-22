"""Self-contained CNN for the navigation-hint prototype.

Identical to model_cnn.ActionCNN except the goal feature dimension is a
constructor argument (2 for flow/flowdist, 4 for the straight+flow "both" mode),
so we can try the new representation without touching the original v0.3 model.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.common import spec


class NavHintCNN(nn.Module):
    def __init__(self, goal_dim: int = 4, p_drop: float = 0.1):
        super().__init__()
        self.goal_dim = goal_dim
        self.conv = nn.Sequential(
            nn.Conv2d(spec.GRID_C, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.goal_fc = nn.Sequential(nn.Linear(goal_dim, 32), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(64 + 32, 128), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(128, spec.NUM_ACTIONS),
        )

    def forward(self, inp):
        grid, goal = inp                        # (B,3,5,5), (B,goal_dim)
        g = self.conv(grid).flatten(1)
        v = self.goal_fc(goal)
        return self.head(torch.cat([g, v], 1))


def load_navhint(path, device="cpu"):
    """Load a NavHintCNN checkpoint. Returns dict like eval_cbs_vs_il.load_model."""
    ck = torch.load(path, map_location=device)
    gdim = int(ck["goal_mean"].shape[-1])
    model = NavHintCNN(goal_dim=gdim)
    model.load_state_dict(ck["model_state"]); model.eval()
    return {"name": ck.get("label", "navhint"), "mode": "cnn", "model": model,
            "gmean": ck["goal_mean"].cpu(), "gstd": ck["goal_std"].cpu(),
            "goal_dim": gdim, "hint_mode": ck.get("hint_mode", "both")}
