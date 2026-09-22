"""Unit tests for src/il/nav_hint.py (Alpha 1 BFS flow / flowdist). numpy only."""

import unittest

import numpy as np

from src.il.nav_hint import bfs_dist, flow_step, make_goal_dir


class TestBfsDist(unittest.TestCase):
    def test_open_grid_is_manhattan(self):
        d = bfs_dist(np.zeros((10, 10), dtype=int), (0, 0))
        self.assertEqual(d[0, 0], 0)
        self.assertEqual(d[9, 0], 9)
        self.assertEqual(d[9, 9], 18)

    def test_wall_is_minus_one_and_detour_is_counted(self):
        g = np.array([[0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0]])
        d = bfs_dist(g, (0, 4))
        self.assertEqual(d[1, 2], -1)
        self.assertEqual(d[0, 0], 4)
        self.assertEqual(d[2, 0], 6)

    def test_goal_inside_wall_gives_all_minus_one(self):
        g = np.zeros((3, 3), dtype=int)
        g[1, 1] = 1
        self.assertTrue((bfs_dist(g, (1, 1)) == -1).all())


class TestFlow(unittest.TestCase):
    def test_flow_routes_around_wall(self):
        g = np.array([[0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0]])
        d = bfs_dist(g, (0, 4))
        self.assertEqual(flow_step(d, (0, 0)), (0, 1))     # open top row -> right
        self.assertEqual(flow_step(d, (2, 0)), (-1, 0))    # blocked by the wall column -> up first

    def test_flow_is_zero_at_goal(self):
        d = bfs_dist(np.zeros((4, 4), dtype=int), (2, 2))
        self.assertEqual(flow_step(d, (2, 2)), (0, 0))

    def test_tie_break_order_is_up_down_left_right(self):
        d = bfs_dist(np.zeros((10, 10), dtype=int), (0, 0))
        # from (9,9): up (8,9) and left (9,8) are both 17 away -> the first neighbour in order wins (up)
        self.assertEqual(flow_step(d, (9, 9)), (-1, 0))


class TestMakeGoalDir(unittest.TestCase):
    def setUp(self):
        self.grid = np.zeros((10, 10), dtype=int)
        self.goal = (0, 0)
        self.dist = bfs_dist(self.grid, self.goal)

    def test_flowdist_is_flow_unit_times_bfs_distance(self):
        # flow direction (-1,0) and remaining BFS distance 9 -> (-9, 0)
        self.assertEqual(make_goal_dir(self.dist, (9, 0), self.goal, "flowdist"), [-9.0, 0.0])

    def test_other_modes(self):
        self.assertEqual(make_goal_dir(self.dist, (9, 0), self.goal, "straight"), [-9.0, 0.0])
        self.assertEqual(make_goal_dir(self.dist, (9, 0), self.goal, "flow"), [-1.0, 0.0])
        self.assertEqual(make_goal_dir(self.dist, (9, 0), self.goal, "both"), [-9.0, 0.0, -1.0, 0.0])

    def test_flowdist_differs_from_straight_when_wall_blocks(self):
        g = np.array([[0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0]])
        d = bfs_dist(g, (0, 4))
        self.assertEqual(make_goal_dir(d, (2, 0), (0, 4), "straight"), [-2.0, 4.0])
        self.assertEqual(make_goal_dir(d, (2, 0), (0, 4), "flowdist"), [-6.0, 0.0])

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            make_goal_dir(self.dist, (9, 0), self.goal, "nope")


if __name__ == "__main__":
    unittest.main()
