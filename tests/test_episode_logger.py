"""Tests for src/common/episode_logger.py and its wiring into scripts/run_il.py.

Split in two:
  * TestEpisodeLoggerUnit    -- numpy/csv only, exercises EpisodeLogger directly
                                 with synthetic rows (no torch required).
  * TestRunILWithLogging     -- end-to-end: runs the real IL policy with
                                 logging on/off via scripts.run_il.run_episode.
                                 Skipped when torch is not installed, same
                                 pattern as tests/test_smoke_il.py.
"""

from __future__ import annotations

import csv
import shutil
import tempfile
import unittest
from pathlib import Path

from src.common.episode_logger import (
    ACTION_NAMES,
    STEP_FIELDS,
    SUMMARY_FIELDS,
    EpisodeLogger,
    _scenario_slug,
    make_run_id,
)

try:
    import torch  # noqa: F401
except ImportError:  # pragma: no cover
    torch = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestRunId(unittest.TestCase):
    def test_scenario_slug_strips_prefix_and_extension(self):
        self.assertEqual(_scenario_slug("scenario_empty_s11_n3.json"), "empty_s11_n3")

    def test_make_run_id_format(self):
        import datetime as dt
        ts = dt.datetime(2026, 9, 22, 14, 25, 0)
        run_id = make_run_id("scenario_empty_s11_n3.json", "IL", timestamp=ts)
        self.assertEqual(run_id, "20260922_142500_empty_s11_n3_IL")


class TestEpisodeLoggerUnit(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp_dir, ignore_errors=True)
        self.logger = EpisodeLogger("run_unit_test", scenario="scenario_unit.json", mode="IL", out_dir=self.tmp_dir)

    def test_action_id_action_name_convention(self):
        self.assertEqual(ACTION_NAMES, {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "WAIT"})

    def test_wait_action_sets_is_wait_true_and_blocked_false(self):
        row = self.logger.log_step(
            timestep=1, agent_id=0, current=(2, 2), action_id=4, intended=(2, 2), next_pos=(2, 2),
            goal_distance_before=3, goal_distance_after=3, conflict_type="none", done=False, success=False,
        )
        self.assertTrue(row["is_wait"])
        self.assertFalse(row["blocked"])
        self.assertEqual(row["action_name"], "WAIT")

    def test_blocked_when_move_action_but_next_equals_current(self):
        # current=(3,3) action=RIGHT(3) intended=(3,4) next=(3,3) -> blocked, not WAIT
        row = self.logger.log_step(
            timestep=1, agent_id=0, current=(3, 3), action_id=3, intended=(3, 4), next_pos=(3, 3),
            goal_distance_before=5, goal_distance_after=5, conflict_type="wall", done=False, success=False,
        )
        self.assertFalse(row["is_wait"])
        self.assertTrue(row["blocked"])
        self.assertEqual(row["action_name"], "RIGHT")
        self.assertEqual((row["intended_row"], row["intended_col"]), (3, 4))
        self.assertEqual((row["next_row"], row["next_col"]), (3, 3))

    def test_unblocked_move_next_equals_intended(self):
        row = self.logger.log_step(
            timestep=1, agent_id=0, current=(3, 3), action_id=3, intended=(3, 4), next_pos=(3, 4),
            goal_distance_before=5, goal_distance_after=4, conflict_type="none", done=False, success=False,
        )
        self.assertFalse(row["blocked"])
        self.assertEqual((row["next_row"], row["next_col"]), (3, 4))

    def test_progress_is_before_minus_after(self):
        row = self.logger.log_step(
            timestep=1, agent_id=0, current=(3, 3), action_id=3, intended=(3, 4), next_pos=(3, 4),
            goal_distance_before=5, goal_distance_after=4, conflict_type="none", done=False, success=False,
        )
        self.assertEqual(row["progress"], 1)
        self.assertIsInstance(row["progress"], int)
        self.assertIsInstance(row["goal_distance_before"], int)
        self.assertIsInstance(row["goal_distance_after"], int)

    def test_write_steps_creates_csv_with_required_columns(self):
        self.logger.log_step(
            timestep=1, agent_id=0, current=(0, 0), action_id=4, intended=(0, 0), next_pos=(0, 0),
            goal_distance_before=0, goal_distance_after=0, conflict_type="none", done=True, success=True,
        )
        path = self.logger.write_steps()
        self.assertTrue(path.is_file())
        self.assertEqual(path.name, "run_unit_test_steps.csv")
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            self.assertEqual(reader.fieldnames, list(STEP_FIELDS))
            rows = list(reader)
        self.assertEqual(len(rows), 1)

    def test_write_summary_creates_csv_with_required_columns(self):
        path = self.logger.write_summary(num_agents=1, steps=1, success=True, timed_out=False, makespan=1)
        self.assertTrue(path.is_file())
        self.assertEqual(path.name, "run_unit_test_summary.csv")
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            self.assertEqual(reader.fieldnames, list(SUMMARY_FIELDS))
            rows = list(reader)
        self.assertEqual(len(rows), 1)

    def test_wait_and_blocked_counts_are_per_agent_step_sums(self):
        self.logger.log_step(
            timestep=1, agent_id=0, current=(0, 0), action_id=4, intended=(0, 0), next_pos=(0, 0),
            goal_distance_before=0, goal_distance_after=0, conflict_type="none", done=False, success=False,
        )
        self.logger.log_step(
            timestep=1, agent_id=1, current=(1, 1), action_id=3, intended=(1, 2), next_pos=(1, 1),
            goal_distance_before=2, goal_distance_after=2, conflict_type="wall", done=False, success=False,
        )
        path = self.logger.write_summary(num_agents=2, steps=1, success=False, timed_out=True, makespan=1)
        with path.open(newline="", encoding="utf-8") as fh:
            row = next(csv.DictReader(fh))
        self.assertEqual(row["wait_count"], "1")
        self.assertEqual(row["blocked_count"], "1")

    def test_vertex_edge_conflicts_are_events_not_row_counts(self):
        # 3 agents tagged "vertex" in the same timestep (one contested cell,
        # i.e. one real event) must NOT be reported as 3 in the summary.
        for aid in (0, 1, 2):
            self.logger.log_step(
                timestep=1, agent_id=aid, current=(aid, 0), action_id=3, intended=(aid, 1), next_pos=(aid, 0),
                goal_distance_before=5, goal_distance_after=5, conflict_type="vertex", done=False, success=False,
            )
        self.logger.log_timestep_conflicts(vertex_events=1, edge_events=0)
        path = self.logger.write_summary(num_agents=3, steps=1, success=False, timed_out=False, makespan=1)
        with path.open(newline="", encoding="utf-8") as fh:
            row = next(csv.DictReader(fh))
        self.assertEqual(row["vertex_conflicts"], "1")  # not 3
        self.assertEqual(row["edge_conflicts"], "0")

    def test_conflicts_default_to_zero_without_log_timestep_conflicts(self):
        path = self.logger.write_summary(num_agents=1, steps=0, success=False, timed_out=False, makespan=0)
        with path.open(newline="", encoding="utf-8") as fh:
            row = next(csv.DictReader(fh))
        self.assertEqual(row["vertex_conflicts"], "0")
        self.assertEqual(row["edge_conflicts"], "0")


@unittest.skipIf(torch is None, "torch is not installed")
class TestRunILWithLogging(unittest.TestCase):
    SCENARIO = PROJECT_ROOT / "scenarios" / "smoke" / "scenario_empty_s11_n3.json"
    CKPT = PROJECT_ROOT / "checkpoints" / "cnn_navhint.pt"

    def setUp(self):
        from scripts.run_il import load_scenario
        from src.il.policy import NavHintPolicy

        self.tmp_dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp_dir, ignore_errors=True)
        self.policy = NavHintPolicy(self.CKPT)
        self.grid, self.starts, self.goals = load_scenario(self.SCENARIO)

    def test_logging_off_reproduces_baseline(self):
        from scripts.run_il import run_episode
        result = run_episode(self.policy, self.grid, self.starts, self.goals, 160, logger=None)
        self.assertTrue(result["all_at_goal"])
        self.assertFalse(result["timed_out"])
        self.assertEqual(result["steps"], 9)  # same as tests/baselines reference

    def test_logging_on_runs_episode_and_writes_both_csvs(self):
        from scripts.run_il import run_episode
        from src.common.episode_logger import EpisodeLogger, make_run_id

        run_id = make_run_id(self.SCENARIO.name, "IL")
        logger = EpisodeLogger(run_id, scenario=self.SCENARIO.name, mode="IL", out_dir=self.tmp_dir)
        result = run_episode(self.policy, self.grid, self.starts, self.goals, 160, logger=logger)

        # Logging on must not change the underlying rollout vs. logging off.
        self.assertTrue(result["all_at_goal"])
        self.assertEqual(result["steps"], 9)

        steps_path = logger.write_steps()
        summary_path = logger.write_summary(
            num_agents=len(self.goals), steps=result["steps"], success=result["all_at_goal"],
            timed_out=result["timed_out"], makespan=result["steps"],
        )
        self.assertTrue(steps_path.is_file())
        self.assertTrue(summary_path.is_file())

        with steps_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            self.assertEqual(reader.fieldnames, list(STEP_FIELDS))
            rows = list(reader)

        # one row per (timestep, agent)
        self.assertEqual(len(rows), result["steps"] * len(self.goals))
        self.assertEqual({int(r["timestep"]) for r in rows}, set(range(1, result["steps"] + 1)))
        self.assertEqual({int(r["agent_id"]) for r in rows}, set(self.goals.keys()))

        for r in rows:
            action_id = int(r["action_id"])
            self.assertIn(action_id, range(5))
            self.assertEqual(r["action_name"], ACTION_NAMES[action_id])
            is_wait = r["is_wait"] == "True"
            self.assertEqual(is_wait, action_id == 4)
            blocked = r["blocked"] == "True"
            next_eq_current = (r["next_row"], r["next_col"]) == (r["current_row"], r["current_col"])
            self.assertEqual(blocked, (not is_wait) and next_eq_current)
            # goal distance / progress: integers, progress == before - after
            before, after, progress = int(r["goal_distance_before"]), int(r["goal_distance_after"]), int(r["progress"])
            self.assertEqual(progress, before - after)
            self.assertIn(r["conflict_type"], {"none", "wall", "vertex", "edge"})
            self.assertIn(r["run_id"], {run_id})
            self.assertEqual(r["scenario"], self.SCENARIO.name)
            self.assertEqual(r["mode"], "IL")

        # final timestep is the only one marked done/success
        final_rows = [r for r in rows if int(r["timestep"]) == result["steps"]]
        self.assertTrue(all(r["done"] == "True" for r in final_rows))
        self.assertTrue(all(r["success"] == "True" for r in final_rows))
        non_final_rows = [r for r in rows if int(r["timestep"]) != result["steps"]]
        self.assertTrue(all(r["done"] == "False" for r in non_final_rows))

        with summary_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            self.assertEqual(reader.fieldnames, list(SUMMARY_FIELDS))
            summary_row = next(reader)
        self.assertEqual(summary_row["run_id"], run_id)
        self.assertEqual(int(summary_row["num_agents"]), len(self.goals))
        self.assertEqual(int(summary_row["steps"]), result["steps"])
        self.assertEqual(summary_row["success"], "True")
        self.assertEqual(summary_row["timed_out"], "False")
        self.assertEqual(int(summary_row["makespan"]), result["steps"])
        self.assertEqual(int(summary_row["wait_count"]), sum(1 for r in rows if r["is_wait"] == "True"))
        self.assertEqual(int(summary_row["blocked_count"]), sum(1 for r in rows if r["blocked"] == "True"))
        self.assertGreaterEqual(int(summary_row["vertex_conflicts"]), 0)
        self.assertGreaterEqual(int(summary_row["edge_conflicts"]), 0)


if __name__ == "__main__":
    unittest.main()
