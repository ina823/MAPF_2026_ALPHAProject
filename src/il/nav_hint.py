"""Single-agent navigation hint via BFS over the static map.

The v0.3 policy's only global signal is ``goal_dir`` = the straight-line vector
to the goal. In a maze that vector points into a wall, so the policy has no way
to know which way to detour. This module computes, from the goal outward, a BFS
distance field over the free cells, and from it the *next step toward the goal
around walls* — a "flow" hint that replaces the crow-flies direction.

It ignores other agents on purpose: navigation (get around walls) is solved here
cheaply and deterministically; coordination (avoid other robots) stays the CNN's
job via the local grid channels.
"""

from __future__ import annotations

from collections import deque

import numpy as np

_NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))  # up, down, left, right


def bfs_dist(grid: np.ndarray, goal) -> np.ndarray:
    """BFS distance (in steps) from ``goal`` to every free cell. -1 = wall/unreachable.

    grid: 2-D, 0=free, 1=wall. goal: (row, col)."""
    grid = np.asarray(grid)
    h, w = grid.shape
    dist = -np.ones((h, w), dtype=np.int32)
    gr, gc = int(goal[0]), int(goal[1])
    if not (0 <= gr < h and 0 <= gc < w) or grid[gr, gc] == 1:
        return dist
    dist[gr, gc] = 0
    dq = deque([(gr, gc)])
    while dq:
        r, c = dq.popleft()
        for dr, dc in _NEIGHBORS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == 0 and dist[nr, nc] < 0:
                dist[nr, nc] = dist[r, c] + 1
                dq.append((nr, nc))
    return dist


def flow_step(dist: np.ndarray, cell) -> tuple[int, int]:
    """Greedy descent on the BFS field: (drow, dcol) toward the neighbor closest
    to the goal. (0,0) at the goal or if boxed in. Guaranteed to reach the goal
    if the cell is in the goal's connected component."""
    r, c = int(cell[0]), int(cell[1])
    h, w = dist.shape
    if dist[r, c] == 0:
        return (0, 0)
    best = None
    for dr, dc in _NEIGHBORS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and dist[nr, nc] >= 0:
            if best is None or dist[nr, nc] < best[0]:
                best = (dist[nr, nc], dr, dc)
    if best is None:
        return (0, 0)
    return (best[1], best[2])


def flow_dir(grid: np.ndarray, goal, cell) -> np.ndarray:
    """Convenience: flow step at ``cell`` as a float32 (2,) vector, same shape as
    the v0.3 ``goal_dir`` so it is a drop-in replacement."""
    step = flow_step(bfs_dist(grid, goal), cell)
    return np.asarray(step, dtype=np.float32)


def make_goal_dir(dist: np.ndarray, cell, goal, mode: str = "both") -> list[float]:
    """Build the goal feature at ``cell`` for a given hint mode.

    - "straight": [drow, dcol] to goal (the v0.3 baseline).
    - "flow"    : flow unit step (direction only).
    - "flowdist": flow unit * BFS distance (direction + distance packed in 2-D).
    - "both"    : [drow, dcol, flow_drow, flow_dcol] (4-D: straight vector kept for
                  open-map coordination + flow direction for maze detours)."""
    r, c = int(cell[0]), int(cell[1])
    sdr, sdc = float(goal[0] - r), float(goal[1] - c)
    if mode == "straight":
        return [sdr, sdc]
    fdr, fdc = flow_step(dist, cell)
    if mode == "flow":
        return [float(fdr), float(fdc)]
    if mode == "flowdist":
        d = dist[r, c]
        mag = float(d) if d >= 0 else float(dist.max() + 1)
        return [fdr * mag, fdc * mag]
    if mode == "both":
        return [sdr, sdc, float(fdr), float(fdc)]
    raise ValueError(f"unknown mode: {mode}")


if __name__ == "__main__":
    # quick self-test: straight line points into a wall; flow routes around it.
    g = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ])
    goal = (0, 4)          # top-right
    cell = (0, 0)          # top-left; wall column at c=2 rows 1..3
    d = bfs_dist(g, goal)
    print("dist field (from goal):\n", d)
    print("straight goal_dir:", [goal[0] - cell[0], goal[1] - cell[1]])  # [0, 4] -> 'right'
    print("flow_dir at (0,0):", flow_dir(g, goal, cell))  # should be 'right' along open top row
    cell2 = (2, 0)         # blocked from goal by the wall column -> must go up or down first
    print("straight at (2,0):", [goal[0] - cell2[0], goal[1] - cell2[1]])  # [-2,4] up-right
    print("flow at (2,0):", flow_dir(g, goal, cell2))
