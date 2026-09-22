"""Conventions shared by the Alpha 1 baseline code. numpy only.

Action convention : 0=UP 1=DOWN 2=LEFT 3=RIGHT 4=WAIT, deltas are (drow, dcol)
Coordinates       : internal (row, col); scenario JSON is [x, y] = [col, row]
Local observation : (3, 5, 5) = wall/outside, other robot, other agent's goal
"""

import unittest
from pathlib import Path

from src.common import spec
from src.common.scenario_loader import load_map_generator_scenario, xy_to_rowcol
from src.il import nav_hint
from src.simulator import mapf_step_simulator as ms

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestActionConvention(unittest.TestCase):
    def test_spec_action_delta(self):
        self.assertEqual(spec.ACTION_DELTA, {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)})
        self.assertEqual(spec.NUM_ACTIONS, 5)

    def test_simulator_uses_the_same_table(self):
        self.assertEqual(ms._STEP_ACTION_DELTA, spec.ACTION_DELTA)

    def test_nav_hint_neighbour_order_matches_actions_0_to_3(self):
        self.assertEqual(nav_hint._NEIGHBORS, tuple(spec.ACTION_DELTA[i] for i in range(4)))


class TestObservationSpec(unittest.TestCase):
    def test_grid_spec(self):
        self.assertEqual((spec.GRID_C, spec.GRID_H, spec.GRID_W), (3, 5, 5))
        self.assertEqual((spec.CH_WALL, spec.CH_OTHER_ROBOT, spec.CH_OTHER_GOAL), (0, 1, 2))
        self.assertEqual((ms._CH_WALL, ms._CH_OTHER_ROBOT, ms._CH_OTHER_GOAL), (0, 1, 2))
        self.assertEqual(ms._FOV_RADIUS, 2)


class TestCoordinates(unittest.TestCase):
    def test_xy_to_rowcol(self):
        self.assertEqual(xy_to_rowcol([3, 4]), (4, 3))       # [x, y] -> (row, col)

    def test_smoke_scenario_loads_in_row_col(self):
        grid, starts, goals = load_map_generator_scenario(
            PROJECT_ROOT / "scenarios" / "smoke" / "scenario_empty_s11_n3.json")
        self.assertEqual(grid.shape, (11, 11))
        self.assertEqual(int(grid.sum()), 0)
        self.assertEqual(starts, [(4, 3), (9, 2), (5, 5)])
        self.assertEqual(goals, [(0, 8), (1, 1), (9, 5)])


if __name__ == "__main__":
    unittest.main()
