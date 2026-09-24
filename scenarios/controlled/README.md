# Controlled scenarios (offline failure analysis)

Four minimal instances for probing coordination behaviour of the IL-navhint
baseline. Format is the same as `scenarios/smoke/`: `map_<slug>.npy`
(0 = free, 1 = wall, int64) + `scenario_<slug>.json` (agent `start`/`goal` are
`[x, y]` = `[col, row]`; `scenario_loader` converts to `(row, col)`).

Each instance is **solvable**: `tests/test_failure_analysis.py` replays a
hand-written joint plan in the unmodified `MAPFStepSimulator` and requires all
agents at goal with zero vertex/edge/wall conflicts. A failure of the policy on
these is therefore a policy failure, not an impossible instance.
Running them says nothing about how often such failures occur in general.

| slug | map | agents | coordination problem | why it is solvable |
|---|---|---|---|---|
| `corridor_bay_n2` | 5x11, 1-wide corridor on row 2 + one bay cell (1,5) | 2, swap ends (head-on) | head-on in a 1-wide corridor: someone must yield into the bay | agent 1 steps into the bay while agent 0 passes, then returns (12 steps) |
| `intersection_n2` | 7x7 plus shape, centre (3,3) | 2, perpendicular crossing, both 2 cells from the centre | simultaneous arrival at the centre cell (vertex conflict, symmetric) | one agent waits a step; the other crosses first (5 steps) |
| `bottleneck_n4` | 5x9, two 5x4 rooms joined by one door (2,4) | 4, two east-bound + two west-bound | single-cell capacity, opposing flows | rooms give waiting space; east-bound pair passes first, then the west-bound pair (17 steps) |
| `cycle_ring_n4` | 3x3 ring around a wall, 8 free cells | 4 corner agents, each goal = next agent's start | circular dependency (A wants B's cell, B wants C's, ...) | 4 free ring cells; all rotate clockwise in lockstep, each target is vacated as it is entered (2 steps). The cycle is policy-induced (waiting on the occupant), not physical |
