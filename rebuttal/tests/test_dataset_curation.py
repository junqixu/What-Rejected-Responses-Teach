from __future__ import annotations

import unittest

from rebuttal.common import sha256_text
from rebuttal.generation.curate_deepseek_dataset import curate_type_rows
from rebuttal.taxonomy.base import apply_perturbation


TRACE = (
    "Define alpha as A; so A = 2. "
    "Define beta as B; so B = A + 3 = 2 + 3 = 5. "
    "Define gamma as C; so C = B * 2 = 5 * 2 = 10. "
    "Answer: 10."
)


def make_row(rejected: str, prompt_id: str) -> dict[str, object]:
    return {
        "id": f"{prompt_id}:spurious_node",
        "prompt_id": prompt_id,
        "chosen": TRACE,
        "rejected": rejected,
        "chosen_hash": sha256_text(TRACE),
        "rejected_hash": sha256_text(rejected),
        "error_type": "spurious_node",
        "split": "train",
        "generator_version": "deepseek-rebuttal-v2",
    }


class DatasetCurationTests(unittest.TestCase):
    def test_keeps_used_spurious_and_rejects_unused_spurious(self) -> None:
        valid_result = apply_perturbation(TRACE, "spurious_node")
        assert valid_result is not None
        unused = TRACE.replace("Answer: 10.", "Define spare as z; so z = 0. Answer: 10.")
        kept, rejected, reasons = curate_type_rows(
            [make_row(valid_result.text, "good"), make_row(unused, "bad")],
            "spurious_node",
            expected_split="train",
        )
        self.assertEqual([row["prompt_id"] for row in kept], ["good"])
        self.assertEqual(rejected[0]["prompt_id"], "bad")
        self.assertEqual(reasons["spurious_node_not_used"], 1)


if __name__ == "__main__":
    unittest.main()
