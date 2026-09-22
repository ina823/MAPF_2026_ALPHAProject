"""Runs scripts/smoke_test_il.py inside the test runner. Skipped when torch is not installed."""

import unittest

try:
    import torch  # noqa: F401
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class TestSmokeIL(unittest.TestCase):
    def test_alpha1_baseline_reproduces(self):
        from scripts.smoke_test_il import run_all

        results = run_all()
        failed = [f"{r.name}: {r.detail}" for r in results if r.status == "FAIL"]
        self.assertEqual(failed, [], "\n".join(failed))


if __name__ == "__main__":
    unittest.main()
