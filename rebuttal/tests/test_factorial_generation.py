from __future__ import annotations

import unittest

from rebuttal.identifiability.build_design_matrix import build_matrix
from rebuttal.identifiability.generate_factorial_pairs import generate_regimes
from rebuttal.identifiability.matrix_diagnostics import diagnose_matrix


TRACE = (
    "Define alpha as A; so A = 2. "
    "Define beta as B; so B = A + 3 = 2 + 3 = 5. "
    "Define gamma as C; so C = B * 2 = 5 * 2 = 10. "
    "Answer: 10."
)


class FactorialGenerationTests(unittest.TestCase):
    def test_required_regimes_and_ranks(self) -> None:
        rows = [
            {"id": f"p{index}", "prompt": f"problem {index}", "solution": TRACE, "op": 10}
            for index in range(10)
        ]
        generated, failures = generate_regimes(
            rows,
            "missing_node",
            "disorder",
            ["confounded", "orthogonal_size", "orthogonal_exposure", "factorial"],
            414,
        )
        self.assertEqual(len(failures), 0)
        self.assertEqual(len(generated["confounded"]), 10)
        self.assertEqual(len(generated["orthogonal_size"]), 10)
        self.assertEqual(len(generated["orthogonal_exposure"]), 20)
        confounded, _ = diagnose_matrix(build_matrix(generated["confounded"], "binary", "exposure"))
        orthogonal, _ = diagnose_matrix(build_matrix(generated["orthogonal_size"], "binary", "exposure"))
        self.assertEqual(confounded["matrix_rank"], 1)
        self.assertEqual(orthogonal["matrix_rank"], 2)
        self.assertEqual(
            {row["prompt_id"] for row in generated["confounded"]},
            {row["prompt_id"] for row in generated["orthogonal_size"]},
        )


if __name__ == "__main__":
    unittest.main()
