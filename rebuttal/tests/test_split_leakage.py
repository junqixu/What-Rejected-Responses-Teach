from __future__ import annotations

import unittest

from rebuttal.common import prompt_split_leakage


class SplitLeakageTests(unittest.TestCase):
    def test_detects_prompt_reuse_across_splits(self) -> None:
        rows = [
            {"prompt_id": "p1", "split": "train"},
            {"prompt_id": "p1", "split": "test"},
            {"prompt_id": "p2", "split": "train"},
        ]
        self.assertEqual(prompt_split_leakage(rows), {"p1": ["test", "train"]})

    def test_multiple_corruptions_in_one_split_are_allowed(self) -> None:
        rows = [
            {"prompt_id": "p1", "split": "train", "error_type": "missing_node"},
            {"prompt_id": "p1", "split": "train", "error_type": "disorder"},
        ]
        self.assertEqual(prompt_split_leakage(rows), {})


if __name__ == "__main__":
    unittest.main()
