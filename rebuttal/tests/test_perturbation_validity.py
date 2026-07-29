from __future__ import annotations

import unittest

from rebuttal.identifiability.feature_schema import ERROR_FEATURES
from rebuttal.taxonomy.base import apply_perturbation, validate_perturbation
from rebuttal.taxonomy.validate_perturbations import validate


def make_trace(index: int) -> str:
    first = index + 2
    second = first + 3
    final = second * 2
    return (
        f"Define alpha as A; so A = {first}. "
        f"Define beta as B; so B = A + 3 = {first} + 3 = {second}. "
        f"Define gamma as C; so C = B * 2 = {second} * 2 = {final}. "
        f"Answer: {final}."
    )


class PerturbationValidityTests(unittest.TestCase):
    def test_all_taxonomy_axes_apply_and_validate(self) -> None:
        for index in range(20):
            trace = make_trace(index)
            for error_type in ERROR_FEATURES:
                with self.subTest(index=index, error_type=error_type):
                    result = apply_perturbation(trace, error_type)
                    self.assertIsNotNone(result)
                    assert result is not None
                    valid, reasons = validate_perturbation(trace, result)
                    self.assertTrue(valid, reasons)
                    self.assertNotEqual(trace, result.text)
                    if error_type == "spurious_node":
                        self.assertIn("downstream_target", result.metadata)

    def test_dataset_validator_applies_type_specific_rules(self) -> None:
        unused = make_trace(0).replace(
            "Answer: 10.", "Define spare as z; so z = 0. Answer: 10."
        )
        report, failures = validate(
            [
                {
                    "id": "p:spurious_node",
                    "prompt_id": "p",
                    "split": "train",
                    "chosen": make_trace(0),
                    "rejected": unused,
                    "error_type": "spurious_node",
                }
            ]
        )
        self.assertFalse(report["valid"])
        self.assertIn("structural:spurious_node_not_used", failures[0]["reasons"])


if __name__ == "__main__":
    unittest.main()
