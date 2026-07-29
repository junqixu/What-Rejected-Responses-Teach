from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


def paired_bootstrap_ci(
    left: Sequence[float],
    right: Sequence[float],
    *,
    samples: int = 10_000,
    seed: int = 414,
    confidence: float = 0.95,
) -> dict[str, float]:
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or len(a) == 0:
        raise ValueError("paired arrays must be non-empty, one-dimensional, and equally sized")
    differences = a - b
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(a), size=(samples, len(a)))
    estimates = differences[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean_difference": float(differences.mean()),
        "ci_low": float(np.quantile(estimates, alpha)),
        "ci_high": float(np.quantile(estimates, 1.0 - alpha)),
        "paired_standardized_effect": float(differences.mean() / differences.std(ddof=1)) if len(a) > 1 and differences.std(ddof=1) > 0 else 0.0,
    }


def exact_mcnemar(left_correct: Sequence[bool], right_correct: Sequence[bool]) -> dict[str, float | int]:
    if len(left_correct) != len(right_correct) or not left_correct:
        raise ValueError("paired correctness arrays must be non-empty and equally sized")
    left_only = sum(bool(left) and not bool(right) for left, right in zip(left_correct, right_correct))
    right_only = sum(not bool(left) and bool(right) for left, right in zip(left_correct, right_correct))
    discordant = left_only + right_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, value) for value in range(0, min(left_only, right_only) + 1)) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {"left_only": left_only, "right_only": right_only, "discordant": discordant, "p_value": p_value}


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    values = [float(value) for value in p_values]
    if any(value < 0 or value > 1 for value in values):
        raise ValueError("p-values must lie in [0, 1]")
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [0.0] * len(values)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (total - rank) * values[index]))
        adjusted[index] = running
    return adjusted
