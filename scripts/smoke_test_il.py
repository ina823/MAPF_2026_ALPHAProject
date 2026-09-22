"""Smoke test: reproduce the Alpha 1 IL-navhint baseline inside Alpha 2.

Run from the repository root:
    python -m scripts.smoke_test_il
    python -m scripts.smoke_test_il --strict-baseline     # also FAIL on reference-metadata drift

Checks (exit code 0 only if there is no FAIL):
  1  checkpoint file exists and its sha256 matches checkpoints/MANIFEST.json
  2  load_navhint(): NavHintCNN, goal_dim=2, hint_mode=flowdist, eval mode, (1,2) normalisation stats
  3  spec constants and action convention 0=UP 1=DOWN 2=LEFT 3=RIGHT 4=WAIT
     (spec / simulator / nav_hint agree)
  4  scenario_empty_s11_n3 loads; [x,y] -> (row,col) conversion
  5  observation: local grid (3,5,5) float32, goal_dir (2,)
  6  logits shape (1,5) with the flowdist goal feature
  7  episode: terminates within max_steps, all_at_goal, not timed_out,
     final positions == goals, all actions in 0..4, steps >= BFS lower bound
  8  determinism: a second episode gives the identical trajectory
  9  reference metadata (tests/baselines/*.json): INFO/WARN only, FAIL with --strict-baseline

There is deliberately NO hard assertion on the exact number of steps (9 was
observed on Alpha 1). 9 equals the makespan lower bound of this scenario, so it
is expected here, but it is not a general invariant of the policy.
Console output is ASCII only (Windows cp949 consoles).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CKPT = PROJECT_ROOT / "checkpoints" / "cnn_navhint.pt"
MANIFEST = PROJECT_ROOT / "checkpoints" / "MANIFEST.json"
SCENARIO = PROJECT_ROOT / "scenarios" / "smoke" / "scenario_empty_s11_n3.json"
REFERENCE = PROJECT_ROOT / "tests" / "baselines" / "alpha1_il_navhint_reference.json"
DEFAULT_MAX_STEPS = 160

EXPECTED_ACTION_DELTA = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)}  # UP DOWN LEFT RIGHT WAIT


class Result:
    def __init__(self, name, status, detail=""):
        self.name, self.status, self.detail = name, status, detail

    def __repr__(self):
        return f"Result({self.name!r}, {self.status!r})"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pos_key(frame):
    return {int(a): tuple(int(x) for x in p) for a, p in frame.items()}


def run_all(max_steps=DEFAULT_MAX_STEPS, strict_baseline=False):
    """Run every check; returns a list of Result (never raises for check failures)."""
    results = []
    ctx = {}

    def check(name, fn):
        try:
            status, detail = fn()
        except Exception as exc:  # a crashing check is a FAIL, later checks still run
            status, detail = "FAIL", f"{type(exc).__name__}: {exc}"
        results.append(Result(name, status, detail))

    # ---- 1. checkpoint file + sha256 -------------------------------------------------
    def c1():
        assert CKPT.is_file(), f"missing {CKPT}"
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        want = manifest["checkpoints"]["cnn_navhint.pt"]["sha256"]
        got = _sha256(CKPT)
        assert got == want, f"sha256 mismatch: got {got}, manifest {want}"
        return "PASS", f"{CKPT.name} {CKPT.stat().st_size} bytes, sha256 {got[:16]}..."
    check("1 checkpoint present + sha256", c1)

    # ---- 2. model load -----------------------------------------------------------------
    def c2():
        from src.il.policy import NavHintPolicy
        policy = NavHintPolicy(CKPT)
        m = policy.model
        assert type(m["model"]).__name__ == "NavHintCNN", type(m["model"]).__name__
        assert m["goal_dim"] == 2, m["goal_dim"]
        assert m["hint_mode"] == "flowdist", m["hint_mode"]
        assert m["mode"] == "cnn", m["mode"]
        assert tuple(m["gmean"].shape) == (1, 2) and tuple(m["gstd"].shape) == (1, 2)
        assert m["model"].training is False, "model must be in eval() mode (Dropout off)"
        ctx["policy"] = policy
        return "PASS", (f"model=NavHintCNN goal_dim={m['goal_dim']} hint_mode={m['hint_mode']} "
                        f"label={m['name']!r}")
    check("2 load_navhint()", c2)

    # ---- 3. constants / action convention ------------------------------------------------
    def c3():
        from src.common import spec
        from src.il import nav_hint
        from src.simulator import mapf_step_simulator as ms
        assert spec.NUM_ACTIONS == 5
        assert (spec.GRID_C, spec.GRID_H, spec.GRID_W) == (3, 5, 5)
        assert spec.ACTION_DELTA == EXPECTED_ACTION_DELTA, spec.ACTION_DELTA
        assert ms._STEP_ACTION_DELTA == spec.ACTION_DELTA, "simulator action table differs from spec"
        assert nav_hint._NEIGHBORS == tuple(spec.ACTION_DELTA[i] for i in range(4)), "nav_hint neighbour order"
        assert (ms._CH_WALL, ms._CH_OTHER_ROBOT, ms._CH_OTHER_GOAL) == \
            (spec.CH_WALL, spec.CH_OTHER_ROBOT, spec.CH_OTHER_GOAL)
        return "PASS", "0=UP 1=DOWN 2=LEFT 3=RIGHT 4=WAIT ; NUM_ACTIONS=5 ; grid (3,5,5) ; spec==simulator==nav_hint"
    check("3 constants + action convention", c3)

    # ---- 4. scenario -----------------------------------------------------------------------
    def c4():
        from scripts.run_il import load_scenario
        grid, starts, goals = load_scenario(SCENARIO)
        assert grid.shape == (11, 11), grid.shape
        assert set(int(v) for v in set(grid.ravel().tolist())) <= {0, 1}
        assert len(starts) == len(goals) == 3
        assert starts[0] == (4, 3) and goals[0] == (0, 8), "JSON [x,y] must become (row,col)"
        ctx.update(grid=grid, starts=starts, goals=goals)
        return "PASS", f"{SCENARIO.name}: map {grid.shape}, 3 agents, starts(row,col)={starts}, goals={goals}"
    check("4 scenario_empty_s11_n3", c4)

    # ---- 5. observation --------------------------------------------------------------------------
    def c5():
        from src.simulator.mapf_step_simulator import MAPFStepSimulator
        sim = MAPFStepSimulator(cbs_solver_root=None, max_steps=max_steps)
        obs = sim.reset(ctx["grid"], ctx["starts"], ctx["goals"])
        for aid, st in obs.items():
            assert st["grid"].shape == (3, 5, 5), st["grid"].shape
            assert str(st["grid"].dtype) == "float32", st["grid"].dtype
            assert st["goal_dir"].shape == (2,), st["goal_dir"].shape
        ctx["obs0"], ctx["sim0"] = obs, sim
        return "PASS", "local grid (3,5,5) float32 ; simulator goal_dir (2,) (straight vector, overwritten by flowdist)"
    check("5 observation shape", c5)

    # ---- 6. logits ----------------------------------------------------------------------------------
    def c6():
        import torch
        from src.il.nav_hint import bfs_dist, make_goal_dir
        policy = ctx["policy"]
        m = policy.model
        aid = 0
        dist = bfs_dist(ctx["grid"], ctx["goals"][aid])
        gd = make_goal_dir(dist, ctx["sim0"]._positions[aid], ctx["goals"][aid], m["hint_mode"])
        assert len(gd) == 2
        with torch.no_grad():
            g = torch.from_numpy(ctx["obs0"][aid]["grid"]).float().unsqueeze(0)
            d = (torch.tensor([gd], dtype=torch.float32) - m["gmean"]) / m["gstd"]
            logits = m["model"]((g, d))
        assert tuple(logits.shape) == (1, 5), tuple(logits.shape)
        assert bool(torch.isfinite(logits).all())
        action = int(logits.argmax(1))
        assert 0 <= action <= 4
        return "PASS", f"logits shape (1,5) ; agent0 flowdist={gd} ; first action={action}"
    check("6 logits shape (1,5)", c6)

    # ---- 7. episode -------------------------------------------------------------------------------------
    def c7():
        from scripts.run_il import run_episode
        from src.il.nav_hint import bfs_dist
        res = run_episode(ctx["policy"], ctx["grid"], ctx["starts"], ctx["goals"], max_steps)
        ctx["episode"] = res
        assert res["done"], "episode did not terminate within max_steps"
        assert res["steps"] <= max_steps
        assert res["all_at_goal"] is True, "all_at_goal is not True"
        assert res["timed_out"] is False, "timed_out is not False"
        assert res["final_positions"] == res["goals"], "final positions != goals"
        used = {a for f in res["actions"] for a in f.values()}
        assert used <= {0, 1, 2, 3, 4}, used
        lower = max(int(bfs_dist(ctx["grid"], ctx["goals"][a])[ctx["starts"][a]]) for a in ctx["goals"])
        assert res["steps"] >= lower, f"steps {res['steps']} < BFS lower bound {lower}"
        distinct = all(len(set(f.values())) == len(f) for f in res["trajectory"])
        assert distinct, "two agents shared a cell (see docs/alpha1_baseline.md KI-1)"
        return "PASS", (f"steps={res['steps']} (max_steps={max_steps}, BFS lower bound={lower}) "
                        f"all_at_goal=True timed_out=False final==goals actions_used={sorted(used)}")
    check("7 episode terminates, all_at_goal, no timeout", c7)

    # ---- 8. determinism -------------------------------------------------------------------------------------
    def c8():
        from scripts.run_il import run_episode
        again = run_episode(ctx["policy"], ctx["grid"], ctx["starts"], ctx["goals"], max_steps)
        assert again["trajectory"] == ctx["episode"]["trajectory"], "second run differs from first run"
        return "PASS", "second episode reproduces the identical trajectory"
    if "episode" in ctx:
        check("8 determinism", c8)
    else:
        results.append(Result("8 determinism", "FAIL", "skipped: episode check did not produce a result"))

    # ---- 9. reference metadata (observed values, not hard assertions) ---------------------------------------------
    def c9():
        ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
        obs = ref["observed_alpha1"]
        res = ctx["episode"]
        notes, drift = [], False
        if res["steps"] != obs["steps"]:
            drift = True
            notes.append(f"steps {res['steps']} != reference {obs['steps']}")
        ref_traj = ref.get("numpy_crosscheck", {}).get("trajectory")
        if ref_traj is not None:
            same = [_pos_key(f) for f in ref_traj] == [_pos_key(f) for f in res["trajectory"]]
            if not same:
                drift = True
                notes.append("trajectory differs from numpy cross-check reference")
        if drift:
            return ("FAIL" if strict_baseline else "WARN"), "; ".join(notes)
        return "INFO", f"matches reference metadata (steps={obs['steps']}, trajectory identical)"
    if "episode" in ctx:
        check("9 reference metadata (soft)", c9)

    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    ap.add_argument("--strict-baseline", action="store_true",
                    help="treat drift from tests/baselines/*.json as FAIL (default: WARN)")
    args = ap.parse_args(argv)

    try:
        import torch  # noqa: F401
    except ImportError:
        print("[FAIL] torch is not installed (pip install -r requirements.txt)")
        return 1

    results = run_all(args.max_steps, args.strict_baseline)
    print("=== Alpha 2 smoke test: Alpha 1 IL-navhint baseline ===")
    for r in results:
        print(f"[{r.status}] {r.name} - {r.detail}")
    failed = [r for r in results if r.status == "FAIL"]
    print("RESULT:", "FAIL" if failed else "PASS", f"({len(results) - len(failed)}/{len(results)} checks without FAIL)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
