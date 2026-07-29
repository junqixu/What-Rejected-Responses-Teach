from __future__ import annotations

import unittest

from rebuttal.generation.prepare_source_splits import assign_split, family_id, prompt_id


class SourceSplitTests(unittest.TestCase):
    def test_assignment_is_family_deterministic(self) -> None:
        first = assign_split("family-a", 414, 0.8, 0.1)
        second = assign_split("family-a", 414, 0.8, 0.1)
        self.assertEqual(first, second)

    def test_prompt_and_family_ids_have_distinct_roles(self) -> None:
        first = {"id": "family-a", "problem": "p1", "question": "q"}
        second = {"id": "family-a", "problem": "p2", "question": "q"}
        self.assertEqual(family_id(first), family_id(second))
        self.assertNotEqual(prompt_id(first), prompt_id(second))


if __name__ == "__main__":
    unittest.main()
