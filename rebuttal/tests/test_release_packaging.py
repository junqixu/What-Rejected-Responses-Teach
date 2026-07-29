from __future__ import annotations

import unittest

from rebuttal.identifiability.feature_schema import ERROR_FEATURES
from rebuttal.generation.package_release import validate_complete_split


class ReleasePackagingTests(unittest.TestCase):
    def test_requires_exactly_one_record_per_axis(self) -> None:
        rows = [
            {
                "id": f"p:{error_type}",
                "prompt_id": "p",
                "split": "train",
                "prompt": "problem",
                "chosen": "A = 1. Answer: 1.",
                "rejected": "A = 2. Answer: 2.",
                "error_type": error_type,
            }
            for error_type in ERROR_FEATURES
        ]
        _packed, problems = validate_complete_split(rows, "train")
        self.assertTrue(any("strict structural validation failed" in problem for problem in problems))
        _packed, missing_problems = validate_complete_split(rows[:-1], "train")
        self.assertTrue(any("expected" in problem for problem in missing_problems))


if __name__ == "__main__":
    unittest.main()
