from __future__ import annotations

from typing import Any

import numpy as np


REDUCTIONS = ("sum", "mean")


def reduce_masked_logps(
    per_token_logps: Any, loss_mask: Any, reduction: str = "sum"
) -> np.ndarray:
    """Reduce response-token log probabilities without counting padding tokens."""
    if reduction not in REDUCTIONS:
        raise ValueError(f"reduction must be one of {REDUCTIONS}")
    values = np.asarray(per_token_logps, dtype=np.float64)
    mask = np.asarray(loss_mask, dtype=bool)
    if values.shape != mask.shape:
        raise ValueError("per_token_logps and loss_mask must have identical shapes")
    totals = np.where(mask, values, 0.0).sum(axis=-1)
    if reduction == "sum":
        return totals
    counts = mask.sum(axis=-1)
    if np.any(counts == 0):
        raise ValueError("mean reduction received a sequence with no response tokens")
    return totals / counts


def make_reduction_trainer(base_trainer: type, reduction: str) -> type:
    """Create a TRL DPOTrainer subclass while keeping imports optional for offline tools."""
    if reduction not in REDUCTIONS:
        raise ValueError(f"reduction must be one of {REDUCTIONS}")

    class ReductionDPOTrainer(base_trainer):  # type: ignore[misc, valid-type]
        sequence_logp_reduction = reduction

        @staticmethod
        def get_batch_logps(logits: Any, labels: Any, *args: Any, **kwargs: Any) -> Any:
            # TRL 0.14 calls this with average_log_prob=False for standard DPO.
            # For the control, forcing this flag changes only response-token
            # aggregation; the pairwise objective and beta remain unchanged.
            positional = list(args)
            if positional:
                positional[0] = reduction == "mean"
                kwargs.pop("average_log_prob", None)
            else:
                kwargs["average_log_prob"] = reduction == "mean"
            return base_trainer.get_batch_logps(logits, labels, *positional, **kwargs)

    ReductionDPOTrainer.__name__ = f"DPOTrainer_{reduction}_logp"
    return ReductionDPOTrainer
