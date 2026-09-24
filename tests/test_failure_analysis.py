"""Tests for scripts/analyze_failure_logs.py and the controlled scenarios.

  * TestFailureAnalysis     -- small synthetic logs written with the real
                               EpisodeLogger (no torch, deterministic).
  * TestControlledScenarios -- each scenarios/controlled/ instance is solvable:
                               a hand-written joint plan is replayed in the
                               unmodified MAPFStepSimulator with zero conflicts
                               (no torch).
  * TestAnalyzeRealRollout  -- end-to-end on the smoke scenario; skipped
                               without torch (same pattern as test_episode_logger).
"""

from __future__ import annotations

import csv
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.analyze_failure_logs import (
    EPISODE_FIELDS, EVENT_FIELDS, analyze_episode, analyze_paths, read_steps, read_summary,
)
from src.common.episode_logger import EpisodeLogger
from src.common.scenario_loader import load_map_generator_scenario

try:
    import torch  # noqa: F401
except ImportError:  # pragma: no cover
    torch = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]

UP, DOWN, LEFT, RIGHT, WAIT = 0, 1, 2, 3, 4


def row(t, aid, cur, action, intended, nxt, before, after, ctype="none"):
    return dict(timestep=t, agent_id=aid, current=cur, action_id=action, intended=intended,
                next_pos=nxt, goal_distance_before=before, goal_distance_after=after,
                conflict_type=ctype)


def wait_row(t, aid, cell, dist):
    return row(t, aid, cell, WAIT, cell, cell, dist, dist)


class TestFailureAnalysis(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def analyze(self, rows, *, success=False, timed_out=None, conflicts=(0, 0), **params):
        """Write rows with the real EpisodeLogger, read them back, analyze."""
        logger = EpisodeLogger("run_syn", scenario="scenario_syn.json", mode="IL", out_dir=self.tmp)
        t_last = max(r["timestep"] for r in rows)
        for r in rows:
            last = r["timestep"] == t_last
            logger.log_step(done=last, success=success and last,
                            failure_reason="timeout" if (last and not success) else "", **r)
        logger.log_timestep_conflicts(*conflicts)
        n = len({r["agent_id"] for r in rows})
        logger.write_steps()
        logger.write_summary(num_agents=n, steps=t_last, success=success,
                             timed_out=(not success) if timed_out is None else timed_out,
                             makespan=t_last)
        return analyze_episode(read_steps(self.tmp / "run_syn_steps.csv"),
                               read_summary(self.tmp / "run_syn_summary.csv"), **params)

    # ---- WAIT ------------------------------------------------------------
    def test_wait_runs(self):
        a, b = (0, 0), (0, 1)
        rows = [wait_row(1, 0, a, 5), wait_row(2, 0, a, 5), wait_row(3, 0, a, 5),
                row(4, 0, a, RIGHT, b, b, 5, 4), wait_row(5, 0, b, 4)]
        events, ep = self.analyze(rows)
        runs = sorted(e["length"] for e in events if e["event_type"] == "wait_run")
        self.assertEqual(runs, [1, 3])
        self.assertEqual(ep["wait_count_active"], 4)
        self.assertEqual(ep["max_consecutive_wait"], 3)
        self.assertEqual(ep["max_consecutive_stationary"], 3)

    def test_wait_of_agent_already_at_goal_is_not_counted(self):
        # agent 1 sits on its goal (distance 0) and WAITs all episode.
        rows = []
        for t in range(1, 6):
            rows.append(row(t, 0, (0, t - 1), RIGHT, (0, t), (0, t), 6 - t, 5 - t))
            rows.append(wait_row(t, 1, (4, 4), 0))
        events, ep = self.analyze(rows, success=True, timed_out=False)
        self.assertEqual(ep["wait_count_active"], 0)
        self.assertEqual(ep["wait_count_at_goal"], 5)
        self.assertEqual(ep["max_consecutive_wait"], 0)
        self.assertEqual(ep["max_consecutive_stationary"], 0)
        self.assertEqual(ep["no_progress_steps"], 0)
        self.assertFalse([e for e in events if e["event_type"] in ("wait_run", "stationary_run")])

    # ---- blocked ---------------------------------------------------------
    def test_blocked_split_by_conflict_type(self):
        rows = [
            row(1, 0, (0, 0), RIGHT, (0, 1), (0, 0), 3, 3, "wall"),
            row(1, 1, (2, 0), RIGHT, (2, 1), (2, 0), 3, 3, "vertex"),
            row(1, 2, (4, 0), RIGHT, (4, 1), (4, 0), 3, 3, "edge"),
        ]
        events, ep = self.analyze(rows)
        self.assertEqual((ep["blocked_by_wall"], ep["blocked_by_vertex"], ep["blocked_by_edge"]), (1, 1, 1))
        self.assertEqual(ep["blocked_count"], 3)
        self.assertEqual(ep["blocked_by_other"], 0)
        self.assertEqual(sum(1 for e in events if e["event_type"] == "blocked"), 3)
        # blocked is not WAIT: it must not inflate wait_count_active
        self.assertEqual(ep["wait_count_active"], 0)
        self.assertEqual(ep["max_consecutive_stationary"], 1)

    # ---- no progress / BFS stagnation -------------------------------------
    def test_no_progress_span_and_goal_distance_stagnation(self):
        a, b, c = (0, 0), (0, 1), (0, 2)
        rows = [wait_row(1, 0, a, 5), wait_row(2, 0, a, 5),
                row(3, 0, a, RIGHT, b, b, 5, 4), wait_row(4, 0, b, 4),
                row(5, 0, b, RIGHT, c, c, 4, 3)]
        events, ep = self.analyze(rows)
        self.assertEqual(ep["no_progress_steps"], 3)
        self.assertEqual(ep["max_no_progress_span"], 2)
        spans = sorted(e["length"] for e in events if e["event_type"] == "no_progress_span")
        self.assertEqual(spans, [1, 2])
        # total BFS distance 5,5,5,4,4,3 -> new minima at t=3 and t=5
        self.assertEqual(ep["goal_distance_stagnation_max_span"], 2)
        self.assertEqual(ep["goal_distance_stagnation_tail_span"], 0)
        self.assertEqual((ep["initial_total_goal_distance"], ep["final_total_goal_distance"]), (5, 3))
        self.assertEqual(ep["progress_ratio"], 0.4)

    def test_tail_stagnation_uses_only_optional_window(self):
        a = (0, 0)
        rows = [row(1, 0, a, RIGHT, (0, 1), (0, 1), 5, 4)] + \
               [wait_row(t, 0, (0, 1), 4) for t in range(2, 7)]
        _, ep = self.analyze(rows)
        self.assertEqual(ep["goal_distance_stagnation_tail_span"], 5)
        # no window given -> no windowed signal (a frozen agent does repeat its joint
        # state, which is the threshold-free signal, so the episode is still a candidate)
        self.assertNotIn("goal_distance_stagnation_tail", ep["candidate_signals"])
        self.assertEqual(ep["candidate_signals"], "joint_state_repeat")
        _, ep = self.analyze(rows, no_progress_window=4)
        self.assertIn("goal_distance_stagnation_tail>=4", ep["candidate_signals"])
        self.assertEqual(ep["stagnation_candidate"], "yes")

    # ---- oscillation -------------------------------------------------------
    def test_a_b_a_oscillation(self):
        a, b = (0, 0), (0, 1)
        rows = [row(1, 0, a, RIGHT, b, b, 4, 3), row(2, 0, b, LEFT, a, a, 3, 4),
                row(3, 0, a, RIGHT, b, b, 4, 3), row(4, 0, b, LEFT, a, a, 3, 4)]
        events, ep = self.analyze(rows)
        self.assertEqual(ep["oscillation_count"], 3)            # A B A, B A B, A B A
        self.assertEqual(ep["oscillation_max_run"], 3)
        self.assertEqual(ep["oscillation_count_excl_goal"], 3)
        self.assertEqual([e["length"] for e in events if e["event_type"] == "oscillation"], [3])
        _, ep3 = self.analyze(rows, oscillation_min_repeats=3)
        self.assertIn("oscillation_run>=3", ep3["candidate_signals"])
        _, ep4 = self.analyze(rows, oscillation_min_repeats=4)
        self.assertNotIn("oscillation_run", ep4["candidate_signals"])   # run of 3 < 4 repeats

    def test_wait_between_moves_is_not_oscillation(self):
        a, b = (0, 0), (0, 1)
        rows = [row(1, 0, a, RIGHT, b, b, 4, 3), wait_row(2, 0, b, 3),
                row(3, 0, b, LEFT, a, a, 3, 4)]
        _, ep = self.analyze(rows)
        self.assertEqual(ep["oscillation_count"], 0)

    # ---- joint-state repeat -----------------------------------------------
    def test_joint_state_repeat(self):
        a0, b0, a1, b1 = (0, 0), (0, 1), (2, 0), (2, 1)
        rows = []
        for t in range(1, 4):
            if t % 2:  # A -> B
                rows += [row(t, 0, a0, RIGHT, b0, b0, 4, 3), row(t, 1, a1, RIGHT, b1, b1, 4, 3)]
            else:      # B -> A
                rows += [row(t, 0, b0, LEFT, a0, a0, 3, 4), row(t, 1, b1, LEFT, a1, a1, 3, 4)]
        events, ep = self.analyze(rows)
        # states: t0 S0, t1 S1, t2 S0 (first repeat, period 2), t3 S1
        self.assertEqual(ep["joint_state_first_repeat_timestep"], 2)
        self.assertEqual(ep["joint_state_period"], 2)
        self.assertEqual(ep["joint_state_repeat_count"], 2)
        self.assertEqual(ep["candidate_signals"], "joint_state_repeat")   # threshold-free
        self.assertEqual(ep["stagnation_candidate"], "yes")
        self.assertEqual(sum(1 for e in events if e["event_type"] == "joint_state_repeat"), 1)

    def test_no_repeat_when_state_never_recurs(self):
        rows = [row(t, 0, (0, t - 1), RIGHT, (0, t), (0, t), 5 - t + 1, 5 - t) for t in range(1, 4)]
        _, ep = self.analyze(rows)
        self.assertEqual(ep["joint_state_repeat_count"], 0)
        self.assertEqual(ep["joint_state_first_repeat_timestep"], "")

    # ---- summary is authoritative for conflict totals ------------------------
    def test_summary_conflict_totals_are_used_reconstruction_is_separate(self):
        rows = [row(1, aid, (aid * 2, 0), RIGHT, (3, 3), (aid * 2, 0), 4, 4, "vertex") for aid in range(3)]
        events, ep = self.analyze(rows, conflicts=(7, 2))
        self.assertEqual((ep["vertex_conflicts"], ep["edge_conflicts"]), (7, 2))          # summary
        self.assertEqual((ep["vertex_conflicts_reconstructed"], ep["edge_conflicts_reconstructed"]), (1, 0))
        self.assertFalse(ep["conflict_reconstruction_matches_summary"])
        vertex = [e for e in events if e["event_type"] == "vertex_conflict_candidate"]
        self.assertEqual(len(vertex), 1)                     # 3 tagged rows, ONE event
        self.assertTrue(vertex[0]["reconstructed"])
        _, ep_ok = self.analyze(rows, conflicts=(1, 0))
        self.assertTrue(ep_ok["conflict_reconstruction_matches_summary"])

    def test_vertex_event_records_stationary_occupant(self):
        # agent 0 moves into (0,1); agent 1 WAITs there -> 1 vertex event, occupant recorded
        rows = [row(1, 0, (0, 0), RIGHT, (0, 1), (0, 0), 4, 4, "vertex"), wait_row(1, 1, (0, 1), 6)]
        events, ep = self.analyze(rows, conflicts=(1, 0))
        self.assertEqual(ep["vertex_conflicts_reconstructed"], 1)
        ev = [e for e in events if e["event_type"] == "vertex_conflict_candidate"][0]
        self.assertEqual(ev["agents"], "0;1")
        self.assertIn("occupants=1", ev["detail"])

    def test_edge_swap_reconstructed_as_one_event(self):
        a, b = (0, 0), (0, 1)
        rows = [row(1, 0, a, RIGHT, b, a, 4, 4, "edge"), row(1, 1, b, LEFT, a, b, 4, 4, "edge")]
        events, ep = self.analyze(rows, conflicts=(0, 1))
        self.assertEqual(ep["edge_conflicts_reconstructed"], 1)
        self.assertTrue(ep["conflict_reconstruction_matches_summary"])
        self.assertEqual(sum(1 for e in events if e["event_type"] == "edge_conflict_candidate"), 1)
        self.assertEqual(ep["blocked_by_edge"], 2)           # two agents blocked, one event

    def test_repeated_conflict_is_only_a_candidate(self):
        a, b = (0, 0), (0, 1)
        rows = []
        for t in (1, 2, 3):
            rows += [row(t, 0, a, RIGHT, b, a, 4, 4, "edge"), row(t, 1, b, LEFT, a, b, 4, 4, "edge")]
        _, ep = self.analyze(rows, conflicts=(0, 3))
        self.assertEqual(ep["repeated_conflict_candidate_max"], 3)
        self.assertEqual(ep["repeated_conflict_candidate_max_consecutive"], 3)
        self.assertEqual(ep["repeated_conflict_candidate_keys"], 1)

    # ---- candidate semantics --------------------------------------------------
    def test_success_episode_is_never_a_stagnation_candidate(self):
        a, b = (0, 0), (0, 1)
        rows = [row(1, 0, a, RIGHT, b, b, 4, 3), row(2, 0, b, LEFT, a, a, 3, 4),
                row(3, 0, a, RIGHT, b, b, 4, 3), row(4, 0, b, LEFT, a, a, 3, 4),
                row(5, 0, a, RIGHT, (0, 1), (0, 1), 4, 3),
                row(6, 0, (0, 1), RIGHT, (0, 2), (0, 2), 3, 0)]
        _, ep = self.analyze(rows, success=True, timed_out=False,
                             oscillation_min_repeats=2, no_progress_window=1, stationary_window=1)
        self.assertIn("oscillation_run>=2", ep["candidate_signals"])   # signal is still visible...
        self.assertEqual(ep["stagnation_candidate"], "no")             # ...but it succeeded
        self.assertEqual(ep["agents_at_goal"], 1)

    def test_failed_episode_without_signal_is_undetermined(self):
        rows = [row(t, 0, (0, t - 1), RIGHT, (0, t), (0, t), 9 - t + 1, 9 - t) for t in range(1, 4)]
        _, ep = self.analyze(rows)                                      # timed out, still progressing
        self.assertEqual(ep["stagnation_candidate"], "undetermined")

    def test_overlap_is_reported(self):
        rows = [row(1, 0, (0, 0), RIGHT, (0, 1), (0, 1), 4, 3), row(1, 1, (0, 2), LEFT, (0, 1), (0, 1), 4, 3)]
        events, ep = self.analyze(rows)
        self.assertEqual(ep["overlap_timesteps"], 1)
        self.assertEqual(sum(1 for e in events if e["event_type"] == "position_overlap"), 1)

    # ---- I/O ------------------------------------------------------------------
    def test_analyze_paths_writes_separate_event_and_episode_csv(self):
        rows = [wait_row(1, 0, (0, 0), 3), wait_row(2, 0, (0, 0), 3)]
        self.analyze(rows)
        out = self.tmp / "out"
        episode, events_path, episode_path = analyze_paths(self.tmp / "run_syn_steps.csv", out_dir=out)
        self.assertEqual(events_path.name, "run_syn_events.csv")
        self.assertEqual(episode_path.name, "run_syn_episode.csv")
        with events_path.open(newline="", encoding="utf-8") as fh:
            self.assertEqual(csv.DictReader(fh).fieldnames, list(EVENT_FIELDS))
        with episode_path.open(newline="", encoding="utf-8") as fh:
            rd = csv.DictReader(fh)
            self.assertEqual(rd.fieldnames, list(EPISODE_FIELDS))
            self.assertEqual(len(list(rd)), 1)
        self.assertEqual(episode["summary_source"], "summary_csv")

    def test_steps_only_leaves_conflict_totals_unavailable(self):
        self.analyze([wait_row(1, 0, (0, 0), 3)])
        (self.tmp / "run_syn_summary.csv").unlink()
        episode, _, _ = analyze_paths(self.tmp / "run_syn_steps.csv", out_dir=self.tmp / "out2")
        self.assertEqual(episode["summary_source"], "steps_only")
        self.assertEqual(episode["vertex_conflicts"], "")
        self.assertEqual(episode["edge_conflicts"], "")


# hand-written joint plans (U/D/L/R/W), replayed in the unmodified simulator
PLANS = {
    # agent 1 pulls into the bay (1,5) while agent 0 passes, then returns
    "corridor_bay_n2": {0: "RRRRWRRRRRR", 1: "LLLLLUDLLLLL"},
    # agent 1 yields one step, then crosses after agent 0 has left the centre
    "intersection_n2": {0: "RRRR", 1: "WDDDD"},
    # 0 and 1 go east through the door first, then 2 and 3 go west
    "bottleneck_n4": {0: "RRDRRRRU", 1: "WWRRURRRRD",
                      2: "WWWWWWLDDLLLLUUL", 3: "WWWWWWWLUULLLLDDL"},
    # all four rotate clockwise in lockstep; every target cell is vacated as it is entered
    "cycle_ring_n4": {0: "RR", 1: "DD", 2: "LL", 3: "UU"},
}
_ACTION = {"U": UP, "D": DOWN, "L": LEFT, "R": RIGHT, "W": WAIT}


class TestControlledScenarios(unittest.TestCase):
    DIR = PROJECT_ROOT / "scenarios" / "controlled"

    def load(self, slug):
        grid, starts, goals = load_map_generator_scenario(self.DIR / f"scenario_{slug}.json")
        return grid, dict(enumerate(starts)), dict(enumerate(goals))

    def test_every_scenario_has_a_plan_and_every_plan_a_scenario(self):
        on_disk = {p.stem[len("scenario_"):] for p in self.DIR.glob("scenario_*.json")}
        self.assertEqual(on_disk, set(PLANS))

    def test_instances_are_well_formed(self):
        from src.il.nav_hint import bfs_dist
        for slug in PLANS:
            with self.subTest(slug=slug):
                grid, starts, goals = self.load(slug)
                self.assertEqual(set(np.unique(grid)) - {0, 1}, set())
                self.assertEqual(len(set(starts.values())), len(starts), "duplicate starts")
                self.assertEqual(len(set(goals.values())), len(goals), "duplicate goals")
                for aid in starts:
                    self.assertEqual(grid[starts[aid]], 0)
                    self.assertEqual(grid[goals[aid]], 0)
                    self.assertGreaterEqual(int(bfs_dist(grid, goals[aid])[starts[aid]]), 1)

    def test_each_scenario_is_solvable_by_a_conflict_free_joint_plan(self):
        from src.simulator.mapf_step_simulator import MAPFStepSimulator
        for slug, plan in PLANS.items():
            with self.subTest(slug=slug):
                grid, starts, goals = self.load(slug)
                horizon = max(len(p) for p in plan.values())
                sim = MAPFStepSimulator(cbs_solver_root=None, max_steps=horizon + 5)
                sim.reset(grid, starts, goals)
                info = {"all_at_goal": False}
                for t in range(horizon):
                    actions = {aid: _ACTION[p[t]] if t < len(p) else WAIT for aid, p in plan.items()}
                    _, _, info = sim.step(actions)
                    self.assertEqual(set(sim._last_conflict_types.values()), {"none"},
                                     f"t={t + 1}: {sim._last_conflict_types}")
                    self.assertEqual((sim._last_vertex_conflict_count, sim._last_edge_conflict_count), (0, 0))
                    self.assertEqual(len(set(sim._positions.values())), len(sim._positions), "overlap")
                self.assertTrue(info["all_at_goal"], f"{slug}: plan does not reach all goals")

    def test_corridor_head_on_needs_the_bay(self):
        # Without the bay the same head-on instance has no solution (only one agent
        # can ever be in the corridor cell between them) -- the bay is what makes
        # the scenario a policy test rather than an impossible instance.
        grid, _, _ = self.load("corridor_bay_n2")
        self.assertEqual(int(grid[1, 5]), 0)
        self.assertEqual(int((grid == 0).sum()), grid.shape[1] + 1)   # corridor row + one bay cell


@unittest.skipIf(torch is None, "torch is not installed")
class TestAnalyzeRealRollout(unittest.TestCase):
    SCENARIO = PROJECT_ROOT / "scenarios" / "smoke" / "scenario_empty_s11_n3.json"
    CKPT = PROJECT_ROOT / "checkpoints" / "cnn_navhint.pt"

    def test_smoke_rollout_analysis(self):
        from scripts.run_il import load_scenario, run_episode
        from src.il.policy import NavHintPolicy

        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        policy = NavHintPolicy(self.CKPT)
        grid, starts, goals = load_scenario(self.SCENARIO)
        logger = EpisodeLogger("run_real", scenario=self.SCENARIO.name, mode="IL", out_dir=tmp)
        result = run_episode(policy, grid, starts, goals, 160, logger=logger)
        logger.write_steps()
        logger.write_summary(num_agents=len(goals), steps=result["steps"], success=result["all_at_goal"],
                             timed_out=result["timed_out"], makespan=result["steps"])
        _, ep = analyze_episode(read_steps(tmp / "run_real_steps.csv"), read_summary(tmp / "run_real_summary.csv"))
        self.assertTrue(ep["success"])
        self.assertEqual(ep["steps"], 9)
        self.assertEqual(ep["agents_at_goal"], 3)
        self.assertEqual(ep["final_total_goal_distance"], 0)
        self.assertEqual(ep["stagnation_candidate"], "no")
        self.assertTrue(ep["conflict_reconstruction_matches_summary"])
        self.assertEqual(ep["joint_state_repeat_count"], 0)
        self.assertEqual(ep["overlap_timesteps"], 0)


if __name__ == "__main__":
    unittest.main()
