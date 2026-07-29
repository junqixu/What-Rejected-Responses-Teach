from __future__ import annotations

import unittest

from rebuttal.evaluation.eval_answer_at_k import extract_answer, summarize


class AnswerAtKTests(unittest.TestCase):
    def test_extracts_last_number(self) -> None:
        self.assertEqual(extract_answer("1 + 2 = 3. Answer: 3"), 3.0)
        self.assertIsNone(extract_answer("no numeric answer"))

    def test_summary_uses_fixed_candidate_prefix(self) -> None:
        results = [
            {
                "op": 10,
                "candidates": [
                    {"correct": False},
                    {"correct": True},
                ],
            }
        ]
        report = summarize(results, [1, 2])
        self.assertEqual(report["overall"]["answer@1"], 0.0)
        self.assertEqual(report["overall"]["answer@2"], 1.0)


if __name__ == "__main__":
    unittest.main()
