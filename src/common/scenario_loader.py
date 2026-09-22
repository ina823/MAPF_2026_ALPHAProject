"""Load map generator scenario JSON files for the CBS adapter.

The map generator stores agent start/goal coordinates as ``[x, y]`` in JSON.
The CBS adapter accepts project-internal ``(row, col)`` coordinates, so this
module converts coordinates before they cross into ``CBSAdapter.plan()``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

GridLocation = tuple[int, int]

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def xy_to_rowcol(pos: Sequence[int]) -> GridLocation:
    """Convert map-generator ``[x, y]`` to internal ``(row, col)``."""
    if not isinstance(pos, (list, tuple)):
        raise ValueError(f"Position must be [x, y], got {pos!r}.")
    if len(pos) != 2:
        raise ValueError(f"Position must have exactly 2 values, got {pos!r}.")
    x, y = pos
    return (int(y), int(x))


def load_map_generator_scenario(
    scenario_json_path: str | Path,
) -> tuple[np.ndarray, list[GridLocation], list[GridLocation]]:
    """Load a map-generator scenario JSON and return ``grid_map, starts, goals``.

    Returned ``starts`` and ``goals`` are internal ``(row, col)`` coordinates.
    """
    scenario_path = Path(scenario_json_path).resolve()
    with scenario_path.open("r", encoding="utf-8") as handle:
        scenario = json.load(handle)

    if not isinstance(scenario, dict):
        raise ValueError(f"Scenario JSON must be an object: {scenario_path}")

    map_file = scenario.get("map_file")
    if not map_file:
        raise ValueError(f"Scenario JSON is missing 'map_file': {scenario_path}")

    map_path = _resolve_map_file(map_file, scenario_path)
    grid_map = np.load(map_path)

    agents = scenario.get("agents")
    if not isinstance(agents, list) or not agents:
        raise ValueError(f"Scenario JSON must include a non-empty 'agents' list: {scenario_path}")

    starts: list[GridLocation] = []
    goals: list[GridLocation] = []
    for agent_index, agent in enumerate(agents):
        if not isinstance(agent, dict):
            raise ValueError(f"Agent {agent_index} must be an object.")
        if "start" not in agent or "goal" not in agent:
            raise ValueError(f"Agent {agent_index} must include 'start' and 'goal'.")
        starts.append(xy_to_rowcol(agent["start"]))
        goals.append(xy_to_rowcol(agent["goal"]))

    return grid_map, starts, goals


def _resolve_map_file(map_file: str | Path, scenario_path: Path) -> Path:
    map_path = Path(map_file)
    candidates = [map_path] if map_path.is_absolute() else [
        PROJECT_ROOT / map_path,
        scenario_path.parent / map_path,
    ]

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved

    raise FileNotFoundError(
        "Could not find scenario map_file. Tried: "
        + ", ".join(str(candidate.resolve()) for candidate in candidates)
    )
