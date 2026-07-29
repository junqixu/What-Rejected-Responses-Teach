from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from typing import Any

import numpy as np

from rebuttal.common import add_common_cli_args, configure_logging, ensure_output_dir, load_records, write_json
from rebuttal.controls.surface_audit import NUMERIC_FIELDS, audit_record


FEATURES = tuple(field for field in NUMERIC_FIELDS if field not in {"prompt_tokens"}) + ("op", "dag_depth")


def _is_test_group(prompt_id: str, seed: int, test_fraction: float) -> bool:
    digest = hashlib.sha256(f"{seed}:{prompt_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    return value < test_fraction


def _matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    values = np.asarray(
        [[float(row.get(field)) if isinstance(row.get(field), (int, float)) else np.nan for field in FEATURES] for row in rows],
        dtype=np.float64,
    )
    medians = np.asarray(
        [np.median(column[~np.isnan(column)]) if np.any(~np.isnan(column)) else 0.0 for column in values.T]
    )
    locations = np.where(np.isnan(values))
    values[locations] = medians[locations[1]]
    return values


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _macro_f1(truth: np.ndarray, prediction: np.ndarray, n_classes: int) -> float:
    scores: list[float] = []
    for label in range(n_classes):
        true_positive = int(np.sum((truth == label) & (prediction == label)))
        false_positive = int(np.sum((truth != label) & (prediction == label)))
        false_negative = int(np.sum((truth == label) & (prediction != label)))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append((2 * true_positive / denominator) if denominator else 0.0)
    return float(np.mean(scores))


def train_classifier(
    rows: list[dict[str, Any]], seed: int, test_fraction: float, iterations: int, learning_rate: float
) -> dict[str, Any]:
    labels = sorted({str(row["error_type"]) for row in rows})
    if len(labels) < 2:
        raise ValueError("metadata classifier requires at least two error types")
    label_index = {label: index for index, label in enumerate(labels)}
    x = _matrix(rows)
    y = np.asarray([label_index[str(row["error_type"])] for row in rows], dtype=np.int64)
    test_mask = np.asarray([_is_test_group(str(row["prompt_id"]), seed, test_fraction) for row in rows])
    if not np.any(test_mask) or np.all(test_mask):
        raise ValueError("group split produced an empty train or test set")
    train_x, test_x = x[~test_mask], x[test_mask]
    train_y, test_y = y[~test_mask], y[test_mask]
    missing_train = [labels[index] for index in range(len(labels)) if index not in set(train_y)]
    if missing_train:
        raise ValueError(f"classes absent from training split: {missing_train}")

    center = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale == 0] = 1.0
    train_x = (train_x - center) / scale
    test_x = (test_x - center) / scale
    train_x = np.column_stack((train_x, np.ones(len(train_x))))
    test_x = np.column_stack((test_x, np.ones(len(test_x))))

    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 0.001, size=(train_x.shape[1], len(labels)))
    targets = np.eye(len(labels))[train_y]
    regularization = 1e-4
    for _ in range(iterations):
        probabilities = _softmax(train_x @ weights)
        gradient = train_x.T @ (probabilities - targets) / len(train_x)
        gradient[:-1] += regularization * weights[:-1]
        weights -= learning_rate * gradient

    prediction = np.argmax(test_x @ weights, axis=1)
    confusion = np.zeros((len(labels), len(labels)), dtype=int)
    for truth, predicted in zip(test_y, prediction):
        confusion[truth, predicted] += 1
    return {
        "model": "multinomial_logistic_regression_numpy",
        "features": list(FEATURES),
        "labels": labels,
        "group_key": "prompt_id",
        "seed": seed,
        "test_fraction": test_fraction,
        "n_train": int(len(train_y)),
        "n_test": int(len(test_y)),
        "accuracy": float(np.mean(test_y == prediction)),
        "macro_f1": _macro_f1(test_y, prediction, len(labels)),
        "confusion_matrix": confusion.tolist(),
        "train_class_counts": dict(Counter(labels[index] for index in train_y)),
        "test_class_counts": dict(Counter(labels[index] for index in test_y)),
        "coefficients": weights[:-1].tolist(),
        "intercepts": weights[-1].tolist(),
        "feature_center": center.tolist(),
        "feature_scale": scale.tolist(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = load_records(args.data, args.max_samples)
    rows = [audit_record(row) for row in source]
    if args.dry_run:
        report = {
            "n_samples": len(rows),
            "classes": dict(Counter(str(row["error_type"]) for row in rows)),
            "n_prompt_groups": len({row["prompt_id"] for row in rows}),
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return report
    report = train_classifier(rows, args.seed, args.test_fraction, args.iterations, args.learning_rate)
    output = ensure_output_dir(args.out, args.overwrite)
    write_json(output / "metadata_classifier.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict error type from non-model surface metadata.")
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--test_fraction", type=float, default=0.2)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    add_common_cli_args(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.log_level)
    run(args)


if __name__ == "__main__":
    main()
