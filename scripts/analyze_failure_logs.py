"""Offline failure analysis of Common Logging CSVs (evidence collection only).

Reads the step CSV (+ summary CSV) written by src/common/episode_logger.py and
computes raw Event-level and Episode-level metrics describing stagnation /
coordination behaviour of a rollout. It is READ-ONLY with respect to the
simulator, the policy and the logs: nothing here changes a rollout, and it is
NOT a Deadlock Detector, Wait-for Graph or recovery mechanism.

Outputs (outputs/failure_analysis/ by default), one pair per run_id:
    <run_id>_events.csv    Event-level: one row per event / interval
    <run_id>_episode.csv   Episode-level: exactly one row

Usage (from the repository root):
    python -m scripts.analyze_failure_logs outputs/logs/<run_id>_steps.csv
    python -m scripts.analyze_failure_logs outputs/logs            # every *_steps.csv
    python -m scripts.analyze_failure_logs <steps.csv> --no-progress-window 8 \\
        --stationary-window 6 --oscillation-min-repeats 3

Principles
  * Thresholds are NOT fixed. Every raw metric (max_consecutive_wait,
    max_no_progress_span, ...) is always written so it can be swept later.
    --no-progress-window / --stationary-window / --oscillation-min-repeats are
    optional and only decide which entries appear in ``candidate_signals``.
  * ``stagnation_candidate`` is NOT a deadlock label. It says whether any raw
    signal was observed in an episode that did not succeed:
        "no"           episode succeeded
        "yes"          episode failed AND >= 1 signal observed
        "undetermined" episode failed and no signal observed (e.g. slow progress)
  * Progress is measured with the BFS goal distances the logger recorded
    (goal_distance_before/after), never with straight-line goal_dir.
  * Agents already at their goal (goal_distance_before == 0) are ACTIVE=False:
    their WAITs / zero progress are not counted as coordination failure.
  * vertex_conflicts / edge_conflicts (episode) come from the summary CSV,
    which is authoritative. Anything derived from step rows is a
    RECONSTRUCTION, marked reconstructed=True (events) or *_reconstructed
    (episode). The logger has no event ID, so repeated conflicts are only a
    ``repeated_conflict_candidate`` (same type + cell + agent set), not exact.
  * joint_state_repeat: the whole-team position tuple at time t equals one seen
    earlier. For a deterministic, stateless policy (the IL-navhint baseline)
    that implies the rollout cycles forever from there; this reasoning does NOT
    transfer to modes with internal state (CBS / hybrid).
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "failure_analysis"

EVENT_FIELDS = (
    "run_id", "event_type", "scope", "agents", "t_start", "t_end", "length",
    "location", "reconstructed", "detail",
)

EPISODE_FIELDS = (
    "run_id", "scenario", "mode", "num_agents", "success", "timed_out", "steps",
    "agents_at_goal", "initial_total_goal_distance", "final_total_goal_distance",
    "progress_ratio",
    # conflict totals: summary CSV is authoritative; *_reconstructed are derived
    "vertex_conflicts", "edge_conflicts",
    "vertex_conflicts_reconstructed", "edge_conflicts_reconstructed",
    "conflict_reconstruction_matches_summary",
    "repeated_conflict_candidate_max", "repeated_conflict_candidate_max_consecutive",
    "repeated_conflict_candidate_keys",
    # wait / blocked (active = not yet at goal)
    "wait_count_active", "wait_count_at_goal", "blocked_count",
    "blocked_by_wall", "blocked_by_vertex", "blocked_by_edge", "blocked_by_other",
    "max_consecutive_wait", "max_consecutive_stationary",
    # progress (BFS goal distance)
    "no_progress_steps", "max_no_progress_span",
    "goal_distance_stagnation_max_span", "goal_distance_stagnation_tail_span",
    "goal_departures",
    # cycling
    "oscillation_count", "oscillation_count_excl_goal", "oscillation_max_run",
    "joint_state_repeat_count", "joint_state_first_repeat_timestep", "joint_state_period",
    # integrity
    "overlap_timesteps", "invalid_distance_rows",
    # candidate output (NOT a deadlock label)
    "candidate_signals", "stagnation_candidate",
    "analysis_params", "summary_source",
)


# --------------------------------------------------------------------------- reading

def _to_bool(value) -> bool:
    return str(value).strip().lower() in ("true", "1")


def read_steps(path: Path | str) -> list[dict]:
    """Step CSV -> typed row dicts (positions become (row, col) tuples)."""
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append({
                "run_id": r["run_id"], "scenario": r["scenario"], "mode": r["mode"],
                "timestep": int(r["timestep"]), "agent_id": int(r["agent_id"]),
                "current": (int(r["current_row"]), int(r["current_col"])),
                "intended": (int(r["intended_row"]), int(r["intended_col"])),
                "next": (int(r["next_row"]), int(r["next_col"])),
                "action_id": int(r["action_id"]),
                "is_wait": _to_bool(r["is_wait"]), "blocked": _to_bool(r["blocked"]),
                "before": int(r["goal_distance_before"]), "after": int(r["goal_distance_after"]),
                "progress": int(r["progress"]),
                "conflict_type": r["conflict_type"],
                "done": _to_bool(r["done"]), "success": _to_bool(r["success"]),
                "failure_reason": r["failure_reason"],
            })
    return rows


def read_summary(path: Path | str) -> dict:
    with Path(path).open(newline="", encoding="utf-8") as fh:
        r = next(csv.DictReader(fh))
    return {
        "run_id": r["run_id"], "scenario": r["scenario"], "mode": r["mode"],
        "num_agents": int(r["num_agents"]), "steps": int(r["steps"]),
        "success": _to_bool(r["success"]), "timed_out": _to_bool(r["timed_out"]),
        "vertex_conflicts": int(r["vertex_conflicts"]), "edge_conflicts": int(r["edge_conflicts"]),
    }


# --------------------------------------------------------------------------- helpers

def _active(row) -> bool:
    """Agent has not reached its goal yet (BFS distance before the step > 0)."""
    return row["before"] > 0


def _runs(rows, pred):
    """Maximal runs of consecutive-timestep rows satisfying ``pred``."""
    out, cur = [], []
    for r in rows:
        if pred(r) and (not cur or r["timestep"] == cur[-1]["timestep"] + 1):
            cur.append(r)
        else:
            if cur:
                out.append(cur)
            cur = [r] if pred(r) else []
    if cur:
        out.append(cur)
    return out


def _cell(pos) -> str:
    return f"{pos[0]},{pos[1]}"


def _ids(agents) -> str:
    return ";".join(str(a) for a in sorted(agents))


def _event(run_id, event_type, scope, agents, t_start, t_end, length, location="",
           reconstructed=False, detail=""):
    return {
        "run_id": run_id, "event_type": event_type, "scope": scope, "agents": _ids(agents),
        "t_start": t_start, "t_end": t_end, "length": length, "location": location,
        "reconstructed": reconstructed, "detail": detail,
    }


def _max_consecutive(sorted_times) -> int:
    best = run = 0
    prev = None
    for t in sorted_times:
        run = run + 1 if prev is not None and t == prev + 1 else 1
        best = max(best, run)
        prev = t
    return best


# --------------------------------------------------------------------------- core

def analyze_episode(step_rows, summary=None, *, no_progress_window=None,
                    stationary_window=None, oscillation_min_repeats=None):
    """Compute (events, episode) from typed step rows and an optional summary dict.

    Pure function: no file I/O. ``episode`` has exactly EPISODE_FIELDS.
    """
    if not step_rows:
        raise ValueError("no step rows to analyze")
    run_id = step_rows[0]["run_id"]
    by_agent = defaultdict(list)
    by_time = defaultdict(list)
    for r in step_rows:
        by_agent[r["agent_id"]].append(r)
        by_time[r["timestep"]].append(r)
    for rows in by_agent.values():
        rows.sort(key=lambda r: r["timestep"])
    agents = sorted(by_agent)
    t_first, t_last = min(by_time), max(by_time)
    events: list[dict] = []

    # ---- wait / stationary / blocked / no-progress (per agent, ACTIVE rows only)
    wait_count_active = wait_count_at_goal = 0
    blocked_by = {"wall": 0, "vertex": 0, "edge": 0, "none": 0}
    no_progress_steps = 0
    max_wait = max_stationary = max_no_progress = 0
    goal_departures = invalid_rows = 0
    for aid in agents:
        rows = by_agent[aid]
        for r in rows:
            if r["before"] < 0 or r["after"] < 0:
                invalid_rows += 1
            if r["is_wait"]:
                if _active(r):
                    wait_count_active += 1
                elif r["before"] == 0:
                    wait_count_at_goal += 1
            if r["blocked"]:
                blocked_by[r["conflict_type"] if r["conflict_type"] in blocked_by else "none"] += 1
                events.append(_event(
                    run_id, "blocked", "agent", [aid], r["timestep"], r["timestep"], 1,
                    _cell(r["current"]),
                    detail=f"conflict_type={r['conflict_type']};intended={_cell(r['intended'])}"))
            if r["before"] == 0 and r["after"] > 0:
                goal_departures += 1
            if _active(r) and r["progress"] <= 0:
                no_progress_steps += 1
        for run in _runs(rows, lambda r: _active(r) and r["is_wait"]):
            max_wait = max(max_wait, len(run))
            events.append(_event(run_id, "wait_run", "agent", [aid], run[0]["timestep"],
                                 run[-1]["timestep"], len(run), _cell(run[0]["current"]),
                                 detail=f"goal_distance_at_start={run[0]['before']}"))
        for run in _runs(rows, lambda r: _active(r) and (r["is_wait"] or r["blocked"])):
            max_stationary = max(max_stationary, len(run))
            n_blocked = sum(1 for r in run if r["blocked"])
            events.append(_event(run_id, "stationary_run", "agent", [aid], run[0]["timestep"],
                                 run[-1]["timestep"], len(run), _cell(run[0]["current"]),
                                 detail=f"wait_steps={len(run) - n_blocked};blocked_steps={n_blocked}"))
        for run in _runs(rows, lambda r: _active(r) and r["progress"] <= 0):
            max_no_progress = max(max_no_progress, len(run))
            events.append(_event(run_id, "no_progress_span", "agent", [aid], run[0]["timestep"],
                                 run[-1]["timestep"], len(run), _cell(run[0]["current"]),
                                 detail="progress<=0 on active rows (BFS goal distance)"))

    # ---- positions over time: pos[aid][t]; t_first-1 is the start position
    pos = {aid: {} for aid in agents}
    for aid in agents:
        rows = by_agent[aid]
        pos[aid][rows[0]["timestep"] - 1] = rows[0]["current"]
        for r in rows:
            pos[aid][r["timestep"]] = r["next"]

    # ---- oscillation: real A->B->A moves (WAIT / blocked steps are not moves)
    osc_total = osc_excl_goal = osc_max_run = osc_runs_ge_min = 0
    for aid in agents:
        rows = by_agent[aid]
        goal_cells = {r["current"] for r in rows if r["before"] == 0} | \
                     {r["next"] for r in rows if r["after"] == 0}
        times = sorted(pos[aid])
        seq = [pos[aid][t] for t in times]
        rev = [i for i in range(len(seq) - 2)
               if seq[i + 2] == seq[i] and seq[i + 1] != seq[i]
               and times[i + 2] == times[i] + 2]
        i = 0
        while i < len(rev):
            j = i
            while j + 1 < len(rev) and rev[j + 1] == rev[j] + 1:
                j += 1
            length = j - i + 1
            a, b = seq[rev[i]], seq[rev[i] + 1]
            involves_goal = bool(goal_cells & {a, b})
            osc_total += length
            osc_max_run = max(osc_max_run, length)
            if not involves_goal:
                osc_excl_goal += length
            if oscillation_min_repeats is not None and length >= oscillation_min_repeats:
                osc_runs_ge_min += 1
            events.append(_event(
                run_id, "oscillation", "agent", [aid], times[rev[i]], times[rev[j] + 2], length,
                f"{_cell(a)}|{_cell(b)}", detail=f"repeats={length};involves_goal_cell={involves_goal}"))
            i = j + 1

    # ---- system potential: total BFS goal distance over time, stagnation spans
    phi = {}
    first_rows = by_time[t_first]
    phi[t_first - 1] = sum(max(r["before"], 0) for r in first_rows)
    for t in sorted(by_time):
        phi[t] = sum(max(r["after"], 0) for r in by_time[t])
    best, best_t = phi[t_first - 1], t_first - 1
    stretches = []  # (t_start, t_end, length, is_tail)
    for t in range(t_first, t_last + 1):
        if t not in phi:
            continue
        if phi[t] < best:
            if t - best_t - 1 > 0:
                stretches.append((best_t + 1, t - 1, t - best_t - 1, False))
            best, best_t = phi[t], t
    tail = t_last - best_t
    if tail > 0:
        stretches.append((best_t + 1, t_last, tail, True))
    stagnation_max = max((s[2] for s in stretches), default=0)
    for t0, t1, length, is_tail in stretches:
        events.append(_event(run_id, "goal_distance_stagnation", "system", agents, t0, t1, length,
                             detail=f"no new minimum of total BFS goal distance;tail={is_tail}"))

    # ---- joint-state repeat (whole-team position tuple seen before)
    seen, repeat_count, first_repeat, period = {}, 0, None, None
    for t in range(t_first - 1, t_last + 1):
        if any(t not in pos[aid] for aid in agents):
            continue
        state = tuple(pos[aid][t] for aid in agents)
        if state in seen:
            repeat_count += 1
            if first_repeat is None:
                first_repeat, period = t, t - seen[state]
                first_seen = seen[state]
        else:
            seen[state] = t
    if repeat_count:
        events.append(_event(run_id, "joint_state_repeat", "system", agents, first_seen, first_repeat,
                             period, detail=f"period={period};repeat_count={repeat_count}"))

    # ---- overlap integrity (two agents on one cell after a step)
    overlap_timesteps = 0
    for t, rows in sorted(by_time.items()):
        cells = defaultdict(list)
        for r in rows:
            cells[r["next"]].append(r["agent_id"])
        dup = {c: a for c, a in cells.items() if len(a) > 1}
        if dup:
            overlap_timesteps += 1
            for c, a in dup.items():
                events.append(_event(run_id, "position_overlap", "cell", a, t, t, 1, _cell(c),
                                     reconstructed=True, detail="two agents share a cell after this step"))

    # ---- conflict RECONSTRUCTION from step rows (summary is authoritative)
    conflict_keys = defaultdict(list)  # key -> timesteps
    vertex_rec = edge_rec = 0
    for t, rows in sorted(by_time.items()):
        cells = defaultdict(list)
        for r in rows:
            if r["conflict_type"] == "vertex":
                cells[r["intended"]].append(r)
        for cell, movers in sorted(cells.items()):
            occupants = [r["agent_id"] for r in rows
                         if r["current"] == cell and r["next"] == cell
                         and r["agent_id"] not in {m["agent_id"] for m in movers}]
            involved = [m["agent_id"] for m in movers] + occupants
            vertex_rec += 1
            conflict_keys[("vertex", cell, frozenset(involved))].append(t)
            events.append(_event(
                run_id, "vertex_conflict_candidate", "cell", involved, t, t, 1, _cell(cell),
                reconstructed=True,
                detail=f"movers={_ids([m['agent_id'] for m in movers])};occupants={_ids(occupants)}"))
        edge_rows = [r for r in rows if r["conflict_type"] == "edge"]
        for i, a in enumerate(edge_rows):
            for b in edge_rows[i + 1:]:
                if a["intended"] == b["current"] and b["intended"] == a["current"]:
                    edge_rec += 1
                    key = ("edge", frozenset({a["current"], b["current"]}),
                           frozenset({a["agent_id"], b["agent_id"]}))
                    conflict_keys[key].append(t)
                    events.append(_event(
                        run_id, "edge_conflict_candidate", "pair", [a["agent_id"], b["agent_id"]], t, t, 1,
                        f"{_cell(a['current'])}|{_cell(b['current'])}", reconstructed=True,
                        detail="swap attempt reconstructed from two rows tagged edge"))
    rep_max = max((len(ts) for ts in conflict_keys.values()), default=0)
    rep_consec = max((_max_consecutive(sorted(ts)) for ts in conflict_keys.values()), default=0)
    rep_keys = sum(1 for ts in conflict_keys.values() if len(ts) >= 2)

    # ---- outcome / totals
    last_rows = by_time[t_last]
    agents_at_goal = sum(1 for r in last_rows if r["after"] == 0)
    initial_total, final_total = phi[t_first - 1], phi[t_last]
    if summary is not None:
        scenario, mode = summary["scenario"], summary["mode"]
        num_agents, steps = summary["num_agents"], summary["steps"]
        success, timed_out = summary["success"], summary["timed_out"]
        vertex_total, edge_total = summary["vertex_conflicts"], summary["edge_conflicts"]
        matches = (vertex_total == vertex_rec and edge_total == edge_rec)
        summary_source = "summary_csv"
    else:  # steps only: conflict totals are unavailable, not silently reconstructed
        scenario, mode = step_rows[0]["scenario"], step_rows[0]["mode"]
        num_agents, steps = len(agents), t_last
        success = any(r["success"] for r in last_rows)
        timed_out = any(r["failure_reason"] == "timeout" for r in last_rows)
        vertex_total = edge_total = matches = ""
        summary_source = "steps_only"

    # ---- candidate signals (only what was asked for + threshold-free joint repeat)
    signals = []
    if repeat_count:
        signals.append("joint_state_repeat")
    if no_progress_window is not None:
        if max_no_progress >= no_progress_window:
            signals.append(f"no_progress_span>={no_progress_window}")
        if tail >= no_progress_window:
            signals.append(f"goal_distance_stagnation_tail>={no_progress_window}")
    if stationary_window is not None and max_stationary >= stationary_window:
        signals.append(f"stationary_run>={stationary_window}")
    if oscillation_min_repeats is not None and osc_max_run >= oscillation_min_repeats:
        signals.append(f"oscillation_run>={oscillation_min_repeats}")
    if success:
        candidate = "no"
    else:
        candidate = "yes" if signals else "undetermined"

    episode = {
        "run_id": run_id, "scenario": scenario, "mode": mode, "num_agents": num_agents,
        "success": success, "timed_out": timed_out, "steps": steps,
        "agents_at_goal": agents_at_goal,
        "initial_total_goal_distance": initial_total, "final_total_goal_distance": final_total,
        "progress_ratio": round((initial_total - final_total) / initial_total, 4) if initial_total > 0 else "",
        "vertex_conflicts": vertex_total, "edge_conflicts": edge_total,
        "vertex_conflicts_reconstructed": vertex_rec, "edge_conflicts_reconstructed": edge_rec,
        "conflict_reconstruction_matches_summary": matches,
        "repeated_conflict_candidate_max": rep_max,
        "repeated_conflict_candidate_max_consecutive": rep_consec,
        "repeated_conflict_candidate_keys": rep_keys,
        "wait_count_active": wait_count_active, "wait_count_at_goal": wait_count_at_goal,
        "blocked_count": sum(blocked_by.values()),
        "blocked_by_wall": blocked_by["wall"], "blocked_by_vertex": blocked_by["vertex"],
        "blocked_by_edge": blocked_by["edge"], "blocked_by_other": blocked_by["none"],
        "max_consecutive_wait": max_wait, "max_consecutive_stationary": max_stationary,
        "no_progress_steps": no_progress_steps, "max_no_progress_span": max_no_progress,
        "goal_distance_stagnation_max_span": stagnation_max,
        "goal_distance_stagnation_tail_span": tail,
        "goal_departures": goal_departures,
        "oscillation_count": osc_total, "oscillation_count_excl_goal": osc_excl_goal,
        "oscillation_max_run": osc_max_run,
        "joint_state_repeat_count": repeat_count,
        "joint_state_first_repeat_timestep": first_repeat if first_repeat is not None else "",
        "joint_state_period": period if period is not None else "",
        "overlap_timesteps": overlap_timesteps, "invalid_distance_rows": invalid_rows,
        "candidate_signals": ";".join(signals), "stagnation_candidate": candidate,
        "analysis_params": (f"no_progress_window={no_progress_window};"
                            f"stationary_window={stationary_window};"
                            f"oscillation_min_repeats={oscillation_min_repeats}"),
        "summary_source": summary_source,
    }
    events.sort(key=lambda e: (e["t_start"], e["event_type"], e["agents"]))
    return events, episode


# --------------------------------------------------------------------------- I/O

def _summary_path_for(steps_path: Path) -> Path | None:
    name = steps_path.name
    if name.endswith("_steps.csv"):
        cand = steps_path.with_name(name[: -len("_steps.csv")] + "_summary.csv")
        if cand.is_file():
            return cand
    return None


def analyze_paths(steps_path, summary_path=None, out_dir=DEFAULT_OUT_DIR, **params):
    """Analyze one run and write <run_id>_events.csv / <run_id>_episode.csv."""
    steps_path = Path(steps_path)
    rows = read_steps(steps_path)
    summary_path = Path(summary_path) if summary_path else _summary_path_for(steps_path)
    summary = read_summary(summary_path) if summary_path else None
    events, episode = analyze_episode(rows, summary, **params)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = episode["run_id"]
    events_path = out_dir / f"{run_id}_events.csv"
    episode_path = out_dir / f"{run_id}_episode.csv"
    with events_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=EVENT_FIELDS)
        w.writeheader()
        w.writerows(events)
    with episode_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=EPISODE_FIELDS)
        w.writeheader()
        w.writerow(episode)
    return episode, events_path, episode_path


def _collect_inputs(paths):
    out = []
    for p in map(Path, paths):
        out.extend(sorted(p.glob("*_steps.csv")) if p.is_dir() else [p])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path, help="*_steps.csv file(s) or a directory of them")
    ap.add_argument("--summary", type=Path, default=None,
                    help="summary CSV (single input only; default: <run_id>_summary.csv next to the steps CSV)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--no-progress-window", type=int, default=None, metavar="N",
                    help="optional: flag no_progress_span / goal-distance stagnation tail >= N steps")
    ap.add_argument("--stationary-window", type=int, default=None, metavar="N",
                    help="optional: flag max_consecutive_stationary >= N")
    ap.add_argument("--oscillation-min-repeats", type=int, default=None, metavar="K",
                    help="optional: flag an A->B->A run of >= K repeats")
    args = ap.parse_args(argv)

    inputs = _collect_inputs(args.inputs)
    if not inputs:
        print("[FAIL] no *_steps.csv input found")
        return 1
    if args.summary is not None and len(inputs) != 1:
        print("[FAIL] --summary needs exactly one steps CSV")
        return 1
    for steps_path in inputs:
        episode, events_path, episode_path = analyze_paths(
            steps_path, args.summary, args.out_dir,
            no_progress_window=args.no_progress_window,
            stationary_window=args.stationary_window,
            oscillation_min_repeats=args.oscillation_min_repeats)
        print(f"{episode['run_id']}: success={episode['success']} steps={episode['steps']} "
              f"agents_at_goal={episode['agents_at_goal']}/{episode['num_agents']} "
              f"candidate={episode['stagnation_candidate']} signals=[{episode['candidate_signals']}]")
        print(f"  events : {events_path}")
        print(f"  episode: {episode_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
