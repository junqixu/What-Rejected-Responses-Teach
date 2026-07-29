from __future__ import annotations

import unittest

from rebuttal.generation.deepseek_eight_types import _parse_env_assignment, _source_prompt_id, validate_candidate
from rebuttal.taxonomy.base import apply_perturbation


TRACE = (
    "Define alpha as A; so A = 2. "
    "Define beta as B; so B = A + 3 = 2 + 3 = 5. "
    "Define gamma as C; so C = B * 2 = 5 * 2 = 10. "
    "Answer: 10."
)


class DeepSeekGenerationValidationTests(unittest.TestCase):
    def test_deterministic_references_pass_api_validators(self) -> None:
        for error_type in (
            "operation_substitution",
            "dependency_redirect",
            "extra_edge",
            "missing_node",
            "spurious_node",
            "disorder",
            "wrong_target",
        ):
            with self.subTest(error_type=error_type):
                result = apply_perturbation(TRACE, error_type)
                self.assertIsNotNone(result)
                assert result is not None
                valid, reasons = validate_candidate(error_type, TRACE, result.text)
                self.assertTrue(valid, reasons)

        computation = apply_perturbation(TRACE, "computation")
        assert computation is not None
        valid, reasons = validate_candidate("computation", TRACE, computation.text)
        self.assertFalse(valid)
        self.assertIn("computation_final_answer_unchanged", reasons)

    def test_identical_candidate_is_rejected(self) -> None:
        valid, reasons = validate_candidate("computation", TRACE, TRACE)
        self.assertFalse(valid)
        self.assertIn("identical_to_chosen", reasons)

    def test_unused_spurious_node_is_rejected(self) -> None:
        rejected = TRACE.replace("Answer: 10.", "Define spare as z; so z = 0. Answer: 10.")
        valid, reasons = validate_candidate("spurious_node", TRACE, rejected)
        self.assertFalse(valid)
        self.assertIn("spurious_node_not_used", reasons)

    def test_prompt_id_uses_content_not_legacy_id(self) -> None:
        first = {"id": "shared", "problem": "p1", "question": "q"}
        second = {"id": "shared", "problem": "p2", "question": "q"}
        self.assertNotEqual(_source_prompt_id(first), _source_prompt_id(second))

    def test_bash_style_env_assignment_is_supported(self) -> None:
        self.assertEqual(
            _parse_env_assignment('export DEEPSEEK_API_KEY="secret"'),
            ("DEEPSEEK_API_KEY", "secret"),
        )


if __name__ == "__main__":
    unittest.main()
