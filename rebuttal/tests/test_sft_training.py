from __future__ import annotations

import unittest

from rebuttal.train_sft import _validate_rows, format_prompt, tokenize_example


class FakeTokenizer:
    eos_token_id = 99

    def __call__(self, text: str, add_special_tokens: bool) -> dict[str, list[int]]:
        prefix = [1] if add_special_tokens else []
        return {"input_ids": prefix + list(range(10, 10 + len(text.split())))}


def clean_row(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "problem": "A equals 2.",
        "question": "What is A?",
        "solution": "A = 2. Answer: 2.",
        "prompt_id": "p1",
        "dag_family_id": "f1",
        "split": "train",
        "op": 10,
        "mode": "normalforward",
        "template": "unit",
    }
    row.update(updates)
    return row


class SftValidationTests(unittest.TestCase):
    def test_accepts_clean_train_op10(self) -> None:
        summary = _validate_rows([clean_row()])
        self.assertEqual(summary["n_records"], 1)
        self.assertEqual(summary["n_dag_families"], 1)

    def test_rejects_heldout_or_perturbed_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "train only"):
            _validate_rows([clean_row(split="validation")])
        with self.assertRaisesRegex(ValueError, "OP=10"):
            _validate_rows([clean_row(op=2)])

    def test_completion_only_labels_mask_prompt(self) -> None:
        tokenizer = FakeTokenizer()
        row = clean_row()
        tokenized = tokenize_example(tokenizer, row, max_length=128)
        prompt_tokens = tokenizer(format_prompt(row), add_special_tokens=True)["input_ids"]
        self.assertEqual(tokenized["labels"][: len(prompt_tokens)], [-100] * len(prompt_tokens))
        self.assertTrue(all(label != -100 for label in tokenized["labels"][len(prompt_tokens) :]))
        self.assertEqual(tokenized["input_ids"][-1], tokenizer.eos_token_id)

if __name__ == "__main__":
    unittest.main()
