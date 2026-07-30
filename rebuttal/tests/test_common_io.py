from __future__ import annotations

import unittest
from pathlib import Path

from rebuttal.common import iter_records


class CommonIoTests(unittest.TestCase):
    def test_streams_json_array_and_jsonl_with_json_suffix(self) -> None:
        fixtures = Path(__file__).parent / "fixtures"
        array_records = iter_records(fixtures / "records_array.json")
        array_first = next(array_records)
        array_second = next(array_records)
        array_records.close()
        self.assertIsInstance(array_first, dict)
        self.assertNotEqual(array_first.get("id"), array_second.get("id"))

        jsonl_records = iter_records(fixtures / "records_jsonl_with_json_suffix.json")
        jsonl_first = next(jsonl_records)
        jsonl_second = next(jsonl_records)
        jsonl_records.close()
        self.assertIsInstance(jsonl_first, dict)
        self.assertNotEqual(jsonl_first.get("id"), jsonl_second.get("id"))


if __name__ == "__main__":
    unittest.main()
