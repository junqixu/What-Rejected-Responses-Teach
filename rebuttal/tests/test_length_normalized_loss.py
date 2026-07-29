from __future__ import annotations

import unittest

import numpy as np

from rebuttal.controls.length_normalized_dpo import reduce_masked_logps


class LengthNormalizedLogpTests(unittest.TestCase):
    def test_sum_and_mean_ignore_padding(self) -> None:
        logps = np.asarray([[-1.0, -2.0, -99.0], [-3.0, -3.0, -3.0]])
        mask = np.asarray([[True, True, False], [True, True, True]])
        np.testing.assert_allclose(reduce_masked_logps(logps, mask, "sum"), [-3.0, -9.0])
        np.testing.assert_allclose(reduce_masked_logps(logps, mask, "mean"), [-1.5, -3.0])

    def test_mean_rejects_empty_response(self) -> None:
        with self.assertRaises(ValueError):
            reduce_masked_logps([[0.0]], [[False]], "mean")


if __name__ == "__main__":
    unittest.main()
