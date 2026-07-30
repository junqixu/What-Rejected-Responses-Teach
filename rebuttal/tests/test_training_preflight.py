from __future__ import annotations

import unittest

from rebuttal.common import sha256_text
from rebuttal.train_dpo import _validate_rows, build_parser


def row(record_id: str, prompt_id: str, rejected: str) -> dict[str, str]:
    chosen = "A = 1. Answer: 1."
    return {
        "id": record_id,
        "prompt_id": prompt_id,
        "prompt": "problem",
        "chosen": chosen,
        "rejected": rejected,
        "chosen_hash": sha256_text(chosen),
        "rejected_hash": sha256_text(rejected),
        "error_type": "computation",
    }


class TrainingPreflightTests(unittest.TestCase):
    def test_reports_prompt_and_error_counts(self) -> None:
        summary = _validate_rows([row("a", "p", "A = 2. Answer: 2.")])
        self.assertEqual(summary["n_records"], 1)
        self.assertEqual(summary["n_prompts"], 1)
        self.assertEqual(summary["error_type_counts"], {"computation": 1})

    def test_rejects_duplicate_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate training record id"):
            _validate_rows(
                [
                    row("same", "p1", "A = 2. Answer: 2."),
                    row("same", "p2", "A = 3. Answer: 3."),
                ]
            )

    def test_error_type_filter_is_available(self) -> None:
        args = build_parser().parse_args(
            [
                "--train_file",
                "train.jsonl",
                "--output_dir",
                "out",
                "--error_types",
                "operation_substitution,wrong_target",
            ]
        )
        self.assertEqual(args.error_types, "operation_substitution,wrong_target")
        self.assertEqual(args.save_strategy, "no")


if __name__ == "__main__":
    unittest.main()
