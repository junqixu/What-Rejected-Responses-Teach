from __future__ import annotations

import unittest

from rebuttal.analysis.statistical_tests import exact_mcnemar, holm_adjust, paired_bootstrap_ci


class StatisticalTests(unittest.TestCase):
    def test_mcnemar_and_holm(self) -> None:
        result = exact_mcnemar([True, True, False, True], [False, True, True, False])
        self.assertEqual(result["left_only"], 2)
        self.assertEqual(result["right_only"], 1)
        adjusted = holm_adjust([0.01, 0.04, 0.03])
        self.assertEqual(adjusted, [0.03, 0.06, 0.06])

    def test_bootstrap_is_reproducible(self) -> None:
        first = paired_bootstrap_ci([1, 1, 0, 1], [0, 1, 0, 0], samples=200, seed=7)
        second = paired_bootstrap_ci([1, 1, 0, 1], [0, 1, 0, 0], samples=200, seed=7)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
