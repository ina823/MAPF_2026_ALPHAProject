"""Reusable step/episode CSV logging for MAPF rollouts.

Generic on purpose: this module knows nothing about the IL policy, the CBS
adapter, or MAPFStepSimulator. A caller (currently scripts/run_il.py; later
CBS / Global Hybrid / Selective Hybrid runners) extracts the per-step values
from its own simulator/policy objects and hands them to ``EpisodeLogger``.
Logging is opt-in: a caller that never constructs an ``EpisodeLogger`` (or
passes ``logger=None`` down its call chain) does no extra work and produces
no CSV output, so the non-logging code path is unaffected byte-for-byte.

Two CSVs per episode, sharing one run_id:
    outputs/logs/<run_id>_steps.csv    -- one row per (timestep, agent)
    outputs/logs/<run_id>_summary.csv  -- one row for the whole episode

``trigger_reason`` / ``cbs_latency_ms`` are placeholders for the future
Deadlock Monitor / CBS hybrid work (not implemented here); they are always
written but are empty/None for IL runs.

Summary metric semantics (read this before adding a caller):
    wait_count / blocked_count
        Agent-step counts, summed over the whole episode: e.g. wait_count=12
        means WAIT was the chosen action in 12 (agent, timestep) rows total
        across however many agents, not "N agents used WAIT at least once".
        These are unambiguous because is_wait/blocked are per-agent-per-step
        properties by definition -- there is no separate "event" they could
        be miscounting. Derived directly from the collected step rows.
    vertex_conflicts / edge_conflicts
        TRUE conflict-EVENT counts, NOT a count of step rows whose
        conflict_type == "vertex"/"edge". A single vertex conflict where 3
        agents converge on one cell produces 3 per-agent rows tagged
        "vertex" but is exactly 1 event; a single edge swap produces 2 rows
        tagged "edge" but is exactly 1 event. Counting rows would therefore
        over-count. These totals are instead accumulated by
        ``log_timestep_conflicts()``, called once per timestep (not once per
        agent) with the simulator's own per-timestep event counts
        (MAPFStepSimulator._last_vertex_conflict_count /
        _last_edge_conflict_count), which are exact because they read the
        simulator's existing collision-resolution bookkeeping (landing-cell
        groups / swapping pairs) that already exists for the unrelated
        purpose of deciding who bounces back -- no new collision logic, no
        change to collision behaviour, see mapf_step_simulator.py::step().
        If a caller never calls log_timestep_conflicts() (e.g. a future
        caller without that instrumentation available), both totals stay 0;
        this is a known limitation, not a silent wrong answer -- such a
        caller must not rely on these two summary fields until it wires up
        the equivalent per-timestep event counts.
"""

from __future__ import annotations

import csv
from pathlib import Path
from datetime import datetime
from typing import Any, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "logs"

# Action convention (0=UP 1=DOWN 2=LEFT 3=RIGHT 4=WAIT) is defined once, and
# only, in src/common/spec.py (ACTION_DELTA). This module never redefines or
# reinterprets it -- these are just English display labels for the CSV.
ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "WAIT"}
WAIT_ACTION_ID = 4

STEP_FIELDS: Sequence[str] = (
    "run_id", "scenario", "mode", "timestep", "agent_id",
    "current_row", "current_col",
    "action_id", "action_name", "is_wait",
    "intended_row", "intended_col",
    "next_row", "next_col",
    "goal_distance_before", "goal_distance_after", "progress",
    "conflict_type", "blocked",
    "done", "success", "failure_reason",
    "trigger_reason", "cbs_latency_ms",
)

SUMMARY_FIELDS: Sequence[str] = (
    "run_id", "scenario", "mode", "num_agents", "steps", "success",
    "timed_out", "makespan", "wait_count", "blocked_count",
    "vertex_conflicts", "edge_conflicts",
)


def _scenario_slug(scenario_name: str) -> str:
    """'scenario_empty_s11_n3.json' -> 'empty_s11_n3' (strip dir/ext/prefix)."""
    stem = Path(scenario_name).stem
    if stem.startswith("scenario_"):
        stem = stem[len("scenario_"):]
    return stem


def make_run_id(scenario_name: str, mode: str, timestamp: Optional[datetime] = None) -> str:
    """'<YYYYmmdd_HHMMSS>_<scenario_slug>_<mode>', e.g. 20260922_142500_empty_s11_n3_IL."""
    ts = (timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{_scenario_slug(scenario_name)}_{mode}"


class EpisodeLogger:
    """Collects step rows for one episode, then writes the two CSVs.

    See the module docstring for what wait_count/blocked_count vs.
    vertex_conflicts/edge_conflicts actually mean.
    """

    def __init__(self, run_id: str, scenario: str, mode: str, out_dir: Path | str = DEFAULT_OUT_DIR):
        self.run_id = run_id
        self.scenario = scenario
        self.mode = mode
        self.out_dir = Path(out_dir)
        self._step_rows: list[dict[str, Any]] = []
        self._vertex_conflict_events = 0
        self._edge_conflict_events = 0

    def log_step(
        self,
        *,
        timestep: int,
        agent_id: int,
        current: tuple[int, int],
        action_id: int,
        intended: tuple[int, int],
        next_pos: tuple[int, int],
        goal_distance_before: int,
        goal_distance_after: int,
        conflict_type: str,
        done: bool,
        success: bool,
        failure_reason: str = "",
        trigger_reason: str = "",
        cbs_latency_ms: Optional[float] = None,
    ) -> dict[str, Any]:
        """Append one step-level row (one call per agent per timestep).

        Returns the row dict (mainly for tests).

        is_wait / blocked are derived here (single source of truth, shared by
        the step CSV and the wait_count/blocked_count summary totals):
            is_wait = action_id == WAIT
            blocked = action_id != WAIT and next_pos == current
        conflict_type is the agent's OWN classification for this step (one
        of "none"/"wall"/"vertex"/"edge") -- see log_timestep_conflicts() for
        the separate, event-level vertex_conflicts/edge_conflicts totals.
        """
        is_wait = action_id == WAIT_ACTION_ID
        blocked = (not is_wait) and (tuple(next_pos) == tuple(current))
        row = {
            "run_id": self.run_id,
            "scenario": self.scenario,
            "mode": self.mode,
            "timestep": timestep,
            "agent_id": agent_id,
            "current_row": current[0],
            "current_col": current[1],
            "action_id": action_id,
            "action_name": ACTION_NAMES[action_id],
            "is_wait": is_wait,
            "intended_row": intended[0],
            "intended_col": intended[1],
            "next_row": next_pos[0],
            "next_col": next_pos[1],
            "goal_distance_before": goal_distance_before,
            "goal_distance_after": goal_distance_after,
            "progress": goal_distance_before - goal_distance_after,
            "conflict_type": conflict_type,
            "blocked": blocked,
            "done": bool(done),
            "success": bool(success),
            "failure_reason": failure_reason,
            "trigger_reason": trigger_reason,
            "cbs_latency_ms": cbs_latency_ms,
        }
        self._step_rows.append(row)
        return row

    def log_timestep_conflicts(self, vertex_events: int, edge_events: int) -> None:
        """Accumulate TRUE conflict-event counts for one timestep.

        Call this exactly once per timestep (not once per agent), with the
        simulator's own per-timestep event counts -- for
        MAPFStepSimulator this is
        (sim._last_vertex_conflict_count, sim._last_edge_conflict_count).
        See the module docstring for why row-counting by conflict_type would
        over-count and why these per-timestep totals are exact instead.
        """
        self._vertex_conflict_events += vertex_events
        self._edge_conflict_events += edge_events

    @property
    def step_rows(self) -> list[dict[str, Any]]:
        return list(self._step_rows)

    def write_steps(self) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"{self.run_id}_steps.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=STEP_FIELDS)
            writer.writeheader()
            writer.writerows(self._step_rows)
        return path

    def write_summary(
        self,
        *,
        num_agents: int,
        steps: int,
        success: bool,
        timed_out: bool,
        makespan: int,
    ) -> Path:
        """wait_count/blocked_count: summed per-agent-step counts from the
        collected step rows (unambiguous, see module docstring).
        vertex_conflicts/edge_conflicts: accumulated event totals from
        log_timestep_conflicts(), NOT derived from step rows (see module
        docstring) -- 0 if log_timestep_conflicts() was never called.
        """
        wait_count = sum(1 for r in self._step_rows if r["is_wait"])
        blocked_count = sum(1 for r in self._step_rows if r["blocked"])

        row = {
            "run_id": self.run_id,
            "scenario": self.scenario,
            "mode": self.mode,
            "num_agents": num_agents,
            "steps": steps,
            "success": bool(success),
            "timed_out": bool(timed_out),
            "makespan": makespan,
            "wait_count": wait_count,
            "blocked_count": blocked_count,
            "vertex_conflicts": self._vertex_conflict_events,
            "edge_conflicts": self._edge_conflict_events,
        }
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"{self.run_id}_summary.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerow(row)
        return path
