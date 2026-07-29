from __future__ import annotations

import unittest

import numpy as np

from rebuttal.identifiability.build_design_matrix import build_matrix
from rebuttal.identifiability.matrix_diagnostics import diagnose_matrix


class DesignMatrixTests(unittest.TestCase):
    def test_confounded_pair_has_rank_one(self) -> None:
        rows = [
            {"error_labels": ["missing_node", "disorder"]},
            {"error_labels": ["missing_node", "disorder"]},
        ]
        matrix = build_matrix(rows, "binary", "delta")
        report, nullspace = diagnose_matrix(matrix)
        self.assertEqual(report["matrix_rank"], 1)
        self.assertEqual(report["nullspace_dimension"], 7)
        self.assertEqual(nullspace.shape, (8, 7))

    def test_orthogonal_pair_has_rank_two(self) -> None:
        rows = [
            {"error_labels": ["missing_nodes"]},
            {"error_type": "disorder"},
        ]
        matrix = build_matrix(rows, "binary", "exposure")
        report, _nullspace = diagnose_matrix(matrix)
        self.assertEqual(report["matrix_rank"], 2)
        np.testing.assert_equal(matrix[:, 4:7], [[1, 0, 0], [0, 0, 1]])


if __name__ == "__main__":
    unittest.main()
