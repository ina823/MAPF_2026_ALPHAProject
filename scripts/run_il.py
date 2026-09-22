"""Roll out the Alpha 1 NavHint IL policy on one scenario JSON.

Baseline-reproduction runner (no new features). The episode loop mirrors
scripts/eval_flow.py::rollout from Alpha 1:

    sim = MAPFStepSimulator(cbs_solver_root=None, max_steps=max_steps)
    obs = sim.reset(grid, starts, goals)
    loop: actions = policy(obs)  ->  obs, done, info = sim.step(actions)

Usage (from the repository root):
    python -m scripts.run_il
    python -m scripts.run_il --scenario scenarios/smoke/scenario_empty_s11_n3.json --verbose
    python -m scripts.run_il --out outputs/logs/run_il_empty_s11_n3.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.scenario_loader import load_map_generator_scenario
from src.il.policy import NavHintPolicy
from src.simulator.mapf_step_simulator import MAPFStepSimulator

DEFAULT_CKPT = PROJECT_ROOT / "checkpoints" / "cnn_navhint.pt"
DEFAULT_SCENARIO = PROJECT_ROOT / "scenarios" / "smoke" / "scenario_empty_s11_n3.json"
DEFAULT_MAX_STEPS = 160  # same default as Alpha 1 scripts/eval_flow.py


def run_episode(policy, grid, starts, goals, max_steps):
    """Run one episode. starts/goals: {agent_id: (row, col)}.

    Returns a dict with steps / done / all_at_goal / timed_out, the final
    positions, and the per-step trajectory and actions (for logging/tests).
    Success is ``info["all_at_goal"]`` (see docs/alpha1_baseline.md, KI-2)."""
    sim = MAPFStepSimulator(cbs_solver_root=None, max_steps=max_steps)
    obs = sim.reset(grid, starts, goals)
    policy.reset(grid, goals)

    trajectory = [{aid: tuple(pos) for aid, pos in sim._positions.items()}]
    actions_log = []
    info = {"t": 0, "all_at_goal": False, "timed_out": False}
    done = False
    for _ in range(max_steps):
        actions = policy.act(obs, sim._positions)
        obs, done, info = sim.step(actions)
        actions_log.append(dict(actions))
        trajectory.append({aid: tuple(pos) for aid, pos in sim._positions.items()})
        if done:
            break

    final = {aid: tuple(pos) for aid, pos in sim._positions.items()}
    return {
        "steps": info["t"],
        "done": bool(done),
        "all_at_goal": bool(info["all_at_goal"]),
        "timed_out": bool(info["timed_out"]),
        "final_positions": final,
        "goals": {aid: tuple(g) for aid, g in goals.items()},
        "trajectory": trajectory,
        "actions": actions_log,
    }


def load_scenario(path):
    """Scenario JSON -> (grid, starts dict, goals dict); (row, col) internally."""
    grid, starts, goals = load_map_generator_scenario(path)
    return grid, {i: starts[i] for i in range(len(starts))}, {i: goals[i] for i in range(len(goals))}


def _jsonable(result):
    out = dict(result)
    for key in ("final_positions", "goals"):
        out[key] = {str(a): list(p) for a, p in result[key].items()}
    out["trajectory"] = [{str(a): list(p) for a, p in f.items()} for f in result["trajectory"]]
    out["actions"] = [{str(a): int(v) for a, v in f.items()} for f in result["actions"]]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    ap.add_argument("--verbose", action="store_true", help="print positions at every step")
    ap.add_argument("--out", type=Path, default=None, help="optional JSON log path")
    args = ap.parse_args(argv)

    policy = NavHintPolicy(args.ckpt)
    grid, starts, goals = load_scenario(args.scenario)
    result = run_episode(policy, grid, starts, goals, args.max_steps)

    print(f"checkpoint : {args.ckpt.name}  (hint_mode={policy.hint_mode}, goal_dim={policy.model['goal_dim']})")
    print(f"scenario   : {args.scenario.name}  map={grid.shape}  agents={len(goals)}")
    if args.verbose:
        for t, frame in enumerate(result["trajectory"]):
            print(f"  t={t:3d}  {frame}")
    print(f"steps={result['steps']}  all_at_goal={result['all_at_goal']}  timed_out={result['timed_out']}")
    print(f"final positions == goals : {result['final_positions'] == result['goals']}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(_jsonable(result), indent=2), encoding="utf-8")
        print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
