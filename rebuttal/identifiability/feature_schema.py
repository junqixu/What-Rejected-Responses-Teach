from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


ERROR_FEATURES: tuple[str, ...] = (
    "computation",
    "operation_substitution",
    "dependency_redirect",
    "extra_edge",
    "missing_node",
    "spurious_node",
    "disorder",
    "wrong_target",
)

ERROR_ALIASES = {
    "computation": "computation",
    "computation_error": "computation",
    "operation": "operation_substitution",
    "operation_substitution": "operation_substitution",
    "dependency": "dependency_redirect",
    "dependency_mismatch": "dependency_redirect",
    "dependency_redirect": "dependency_redirect",
    "extra_edge": "extra_edge",
    "missing": "missing_node",
    "missing_nodes": "missing_node",
    "missing_node": "missing_node",
    "spurious": "spurious_node",
    "spurious_node": "spurious_node",
    "disorder": "disorder",
    "wrong_target": "wrong_target",
}

VALUE_MODES: tuple[str, ...] = ("binary", "count", "severity")


def canonical_error_name(value: str) -> str:
    key = value.strip().lower().replace("-", "_").replace(" ", "_")
    if key not in ERROR_ALIASES:
        raise ValueError(f"unknown error label: {value!r}")
    return ERROR_ALIASES[key]


@dataclass(frozen=True)
class FeatureValue:
    binary: float = 0.0
    count: float = 0.0
    severity: float = 0.0

    @classmethod
    def from_raw(cls, value: Any) -> "FeatureValue":
        if isinstance(value, Mapping):
            count = float(value.get("count", value.get("binary", 0.0)) or 0.0)
            binary = float(value.get("binary", count > 0) or 0.0)
            severity = float(value.get("severity", count) or 0.0)
            return cls(binary=binary, count=count, severity=severity)
        number = float(value or 0.0)
        return cls(binary=float(number != 0.0), count=number, severity=abs(number))

    def select(self, mode: str) -> float:
        if mode not in VALUE_MODES:
            raise ValueError(f"unsupported value mode: {mode}")
        return float(getattr(self, mode))


def empty_feature_map() -> dict[str, FeatureValue]:
    return {name: FeatureValue() for name in ERROR_FEATURES}


def _labels_from_record(record: Mapping[str, Any]) -> Sequence[str]:
    labels = record.get("error_labels")
    if isinstance(labels, str):
        return [labels]
    if isinstance(labels, Sequence):
        return [str(label) for label in labels]
    error_type = record.get("error_type")
    return [str(error_type)] if error_type else []


def feature_map_from_record(record: Mapping[str, Any]) -> dict[str, FeatureValue]:
    result = empty_feature_map()
    raw_vector = record.get("error_vector")
    if isinstance(raw_vector, Mapping):
        for raw_name, value in raw_vector.items():
            name = canonical_error_name(str(raw_name))
            result[name] = FeatureValue.from_raw(value)
    for raw_label in _labels_from_record(record):
        name = canonical_error_name(raw_label)
        if result[name].binary == 0.0:
            severity_map = record.get("error_severity")
            severity = 1.0
            if isinstance(severity_map, Mapping):
                severity = float(severity_map.get(raw_label, severity) or severity)
            result[name] = FeatureValue(binary=1.0, count=1.0, severity=severity)
    return result


def difference_vector(
    record: Mapping[str, Any], mode: str = "binary", orientation: str = "delta"
) -> list[float]:
    if orientation not in {"delta", "exposure"}:
        raise ValueError("orientation must be 'delta' or 'exposure'")

    chosen_raw = record.get("chosen_error_vector")
    rejected_raw = record.get("rejected_error_vector")
    if isinstance(chosen_raw, Mapping) or isinstance(rejected_raw, Mapping):
        chosen = feature_map_from_record({"error_vector": chosen_raw or {}})
        rejected = feature_map_from_record({"error_vector": rejected_raw or {}})
        delta = [chosen[name].select(mode) - rejected[name].select(mode) for name in ERROR_FEATURES]
    else:
        rejected = feature_map_from_record(record)
        delta = [-rejected[name].select(mode) for name in ERROR_FEATURES]

    return [-value for value in delta] if orientation == "exposure" else delta


def feature_names(mode: str = "binary") -> list[str]:
    if mode not in VALUE_MODES:
        raise ValueError(f"unsupported value mode: {mode}")
    return [f"{name}.{mode}" for name in ERROR_FEATURES]
