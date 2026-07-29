from __future__ import annotations

from typing import Any

import numpy as np


def _json_number(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def diagnose_matrix(matrix: np.ndarray, tolerance: float | None = None) -> tuple[dict[str, Any], np.ndarray]:
    x = np.asarray(matrix, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("design matrix must be two-dimensional")
    n_samples, n_features = x.shape
    if n_samples == 0:
        raise ValueError("design matrix contains no samples")

    _u, singular_values, vh = np.linalg.svd(x, full_matrices=True)
    if tolerance is None:
        leading = singular_values[0] if singular_values.size else 0.0
        tolerance = max(x.shape) * np.finfo(np.float64).eps * leading
    rank = int(np.sum(singular_values > tolerance))

    positive = singular_values[singular_values > tolerance]
    if positive.size:
        probabilities = positive / positive.sum()
        effective_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    else:
        effective_rank = 0.0

    if rank < min(x.shape) or positive.size == 0:
        condition_number = float("inf")
    else:
        condition_number = float(positive[0] / positive[-1])

    variances = np.var(x, axis=0)
    correlations: list[list[float | None]] = []
    for left in range(n_features):
        row: list[float | None] = []
        for right in range(n_features):
            if left == right and variances[left] > 0:
                row.append(1.0)
            elif variances[left] == 0 or variances[right] == 0:
                row.append(None)
            else:
                row.append(float(np.corrcoef(x[:, left], x[:, right])[0, 1]))
        correlations.append(row)

    nullspace = vh[rank:, :].T.copy()
    report: dict[str, Any] = {
        "n_samples": n_samples,
        "n_features": n_features,
        "matrix_rank": rank,
        "effective_rank": effective_rank,
        "singular_values": [float(value) for value in singular_values],
        "condition_number": _json_number(condition_number),
        "condition_number_is_infinite": not np.isfinite(condition_number),
        "nullspace_dimension": int(n_features - rank),
        "pairwise_feature_correlation": correlations,
        "variance_per_feature": [float(value) for value in variances],
        "svd_tolerance": float(tolerance),
    }
    return report, nullspace
