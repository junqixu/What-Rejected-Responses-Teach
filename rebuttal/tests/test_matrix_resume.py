import json
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.run_eval_matrix import _evaluation_complete
from scripts.run_train_matrix import _completed


class MatrixResumeTests(unittest.TestCase):
    def test_training_resume_requires_matching_completed_manifest(self) -> None:
        output = Path("unused-output")
        manifest = {"status": "complete", "checkpoint": "/models/a", "seed": 414}
        with (
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "read_text", return_value=json.dumps(manifest)),
        ):
            self.assertTrue(_completed(output, {"checkpoint": "/models/a", "seed": 414}))
            self.assertFalse(_completed(output, {"checkpoint": "/models/b", "seed": 414}))

    def test_evaluation_resume_requires_metrics_and_matching_provenance(self) -> None:
        output = Path("unused-output")
        expected = {"checkpoint": "/runs/model", "split": "validation"}
        with (
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "read_text", return_value=json.dumps(expected)),
        ):
            self.assertTrue(_evaluation_complete(output, expected))
            self.assertFalse(_evaluation_complete(output, {**expected, "split": "test"}))


if __name__ == "__main__":
    unittest.main()
