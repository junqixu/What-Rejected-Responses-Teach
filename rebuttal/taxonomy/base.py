from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterable

from rebuttal.identifiability.feature_schema import canonical_error_name


SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
ASSIGNMENT = re.compile(r"\b([A-Za-z])\s*=\s*([^;,.]+)")
ARITHMETIC = re.compile(r"(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)")
ANSWER = re.compile(r"(Answer\s*:\s*)(-?\d+(?:\.\d+)?)", re.IGNORECASE)
VARIABLE = re.compile(r"\b([A-Za-z])\b")


@dataclass(frozen=True)
class TraceGraph:
    steps: tuple[str, ...]
    definitions: tuple[tuple[str, int], ...]
    edges: tuple[tuple[str, str], ...]
    forward_references: tuple[tuple[str, int], ...]

    @property
    def node_count(self) -> int:
        return len({name for name, _index in self.definitions})

    @property
    def edge_count(self) -> int:
        return len(self.edges)


@dataclass(frozen=True)
class PerturbationResult:
    error_type: str
    text: str
    changed_step_indices: tuple[int, ...]
    metadata: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def split_trace(text: str) -> list[str]:
    return [part.strip() for part in SENTENCE_BOUNDARY.split(text.strip()) if part.strip()]


def _assignments(step: str) -> list[tuple[str, str]]:
    return [(match.group(1), match.group(2)) for match in ASSIGNMENT.finditer(step)]


def parse_trace(text: str) -> TraceGraph:
    steps = split_trace(text)
    definitions: list[tuple[str, int]] = []
    first_definition: dict[str, int] = {}
    edge_set: set[tuple[str, str]] = set()
    forward: set[tuple[str, int]] = set()

    all_defined = {name for step in steps for name, _rhs in _assignments(step)}
    for index, step in enumerate(steps):
        for target, rhs in _assignments(step):
            definitions.append((target, index))
            for source in VARIABLE.findall(rhs):
                if source == target or source.lower() in {"e", "x"} and source not in all_defined:
                    continue
                if source in all_defined:
                    edge_set.add((source, target))
                    if source not in first_definition:
                        forward.add((source, index))
            first_definition.setdefault(target, index)

    return TraceGraph(
        steps=tuple(steps),
        definitions=tuple(definitions),
        edges=tuple(sorted(edge_set)),
        forward_references=tuple(sorted(forward)),
    )


def _join_steps(steps: Iterable[str]) -> str:
    return " ".join(step.strip() for step in steps if step.strip())


def _dependency_candidates(graph: TraceGraph) -> list[tuple[int, int, str]]:
    first: dict[str, int] = {}
    for name, index in graph.definitions:
        first.setdefault(name, index)
    candidates: list[tuple[int, int, str]] = []
    for later_index, step in enumerate(graph.steps):
        for _target, rhs in _assignments(step):
            for source in VARIABLE.findall(rhs):
                if source in first and first[source] < later_index:
                    candidates.append((first[source], later_index, source))
    return candidates


def _replace_match(text: str, match: re.Match[str], replacement: str) -> str:
    return text[: match.start()] + replacement + text[match.end() :]


def _computation(text: str) -> PerturbationResult | None:
    steps = split_trace(text)
    for index, step in enumerate(steps):
        match = ARITHMETIC.search(step)
        if not match:
            continue
        old_value = float(match.group(4))
        new_value = old_value + 1.0
        rendered = str(int(new_value)) if new_value.is_integer() else str(new_value)
        replacement = match.group(0)[: match.start(4) - match.start()] + rendered
        steps[index] = _replace_match(step, match, replacement)
        return PerturbationResult("computation", _join_steps(steps), (index,), {"old_result": match.group(4), "new_result": rendered})
    return None


def _operation_substitution(text: str) -> PerturbationResult | None:
    replacements = {"+": "-", "-": "+", "*": "+", "/": "*"}
    steps = split_trace(text)
    for index, step in enumerate(steps):
        match = ARITHMETIC.search(step)
        if not match:
            continue
        operator_start = match.start(2)
        old = match.group(2)
        steps[index] = step[:operator_start] + replacements[old] + step[operator_start + 1 :]
        return PerturbationResult(
            "operation_substitution",
            _join_steps(steps),
            (index,),
            {"old_operator": old, "new_operator": replacements[old]},
        )
    return None


def _missing_node(text: str) -> PerturbationResult | None:
    graph = parse_trace(text)
    for defining, later, variable in _dependency_candidates(graph):
        if later == defining:
            continue
        steps = list(graph.steps)
        removed = steps.pop(defining)
        return PerturbationResult(
            "missing_node",
            _join_steps(steps),
            (defining,),
            {"removed_variable": variable, "removed_step": removed},
        )
    return None


def _disorder(text: str) -> PerturbationResult | None:
    graph = parse_trace(text)
    for defining, later, variable in _dependency_candidates(graph):
        if later - defining < 1:
            continue
        steps = list(graph.steps)
        steps[defining], steps[later] = steps[later], steps[defining]
        return PerturbationResult(
            "disorder",
            _join_steps(steps),
            (defining, later),
            {"dependency_variable": variable, "swapped_steps": [defining, later]},
        )
    return None


def _dependency_redirect(text: str) -> PerturbationResult | None:
    graph = parse_trace(text)
    defined_so_far: list[str] = []
    steps = list(graph.steps)
    for index, step in enumerate(steps):
        for target, rhs in _assignments(step):
            used = [name for name in VARIABLE.findall(rhs) if name in defined_so_far]
            alternatives = [name for name in defined_so_far if name not in used and name != target]
            if used and alternatives:
                source, replacement = used[0], alternatives[0]
                match = re.search(rf"\b{re.escape(source)}\b", step)
                if match:
                    steps[index] = _replace_match(step, match, replacement)
                    return PerturbationResult(
                        "dependency_redirect",
                        _join_steps(steps),
                        (index,),
                        {"old_dependency": source, "new_dependency": replacement},
                    )
        defined_so_far.extend(name for name, _rhs in _assignments(step))
    return None


def _extra_edge(text: str) -> PerturbationResult | None:
    graph = parse_trace(text)
    defined_so_far: list[str] = []
    steps = list(graph.steps)
    for index, step in enumerate(steps):
        assignments = list(ASSIGNMENT.finditer(step))
        if assignments and defined_so_far:
            match = assignments[-1]
            target, rhs = match.group(1), match.group(2)
            candidates = [name for name in defined_so_far if name != target and name not in VARIABLE.findall(rhs)]
            if candidates:
                source = candidates[0]
                replacement = f"{target} = {rhs} + 0 * {source}"
                steps[index] = _replace_match(step, match, replacement)
                return PerturbationResult(
                    "extra_edge",
                    _join_steps(steps),
                    (index,),
                    {"inserted_dependency": source, "target": target},
                )
        defined_so_far.extend(name for name, _rhs in _assignments(step))
    return None


def _spurious_node(text: str) -> PerturbationResult | None:
    graph = parse_trace(text)
    used = {name for name, _index in graph.definitions}
    variable = next((name for name in "zqruvw" if name not in used), None)
    if variable is None or not graph.steps:
        return None
    steps = list(graph.steps)
    target_index = next(
        (index for index in range(len(steps) - 1, -1, -1) if ASSIGNMENT.search(steps[index])),
        None,
    )
    if target_index is None:
        return None
    target_step = steps[target_index]
    assignment = ASSIGNMENT.search(target_step)
    if assignment is None:
        return None
    target, rhs = assignment.group(1), assignment.group(2)
    first_expression, separator, remainder = rhs.partition("=")
    augmented_rhs = f"{first_expression.rstrip()} + {variable}"
    if separator:
        augmented_rhs += f" = {remainder.lstrip()}"
    steps[target_index] = _replace_match(
        target_step,
        assignment,
        f"{target} = {augmented_rhs}",
    )
    insert_at = target_index
    steps.insert(insert_at, f"Define a redundant zero as {variable}; so {variable} = 0.")
    return PerturbationResult(
        "spurious_node",
        _join_steps(steps),
        (insert_at, insert_at + 1),
        {"inserted_variable": variable, "downstream_target": target},
    )


def _wrong_target(text: str) -> PerturbationResult | None:
    answer = ANSWER.search(text)
    if not answer:
        return None
    prior_numbers = re.findall(r"-?\d+(?:\.\d+)?", text[: answer.start()])
    replacement = next((value for value in reversed(prior_numbers) if value != answer.group(2)), None)
    if replacement is None:
        return None
    changed = text[: answer.start(2)] + replacement + text[answer.end(2) :]
    return PerturbationResult(
        "wrong_target",
        changed,
        (len(split_trace(text)) - 1,),
        {"old_answer": answer.group(2), "new_answer": replacement},
    )


PERTURBERS = {
    "computation": _computation,
    "operation_substitution": _operation_substitution,
    "dependency_redirect": _dependency_redirect,
    "extra_edge": _extra_edge,
    "missing_node": _missing_node,
    "spurious_node": _spurious_node,
    "disorder": _disorder,
    "wrong_target": _wrong_target,
}


def apply_perturbation(text: str, error_type: str) -> PerturbationResult | None:
    canonical = canonical_error_name(error_type)
    return PERTURBERS[canonical](text)


def validate_perturbation(chosen: str, result: PerturbationResult) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not result.text.strip():
        reasons.append("empty_rejected")
    if result.text.strip() == chosen.strip():
        reasons.append("identical_to_chosen")
    if not result.changed_step_indices:
        reasons.append("missing_changed_step_indices")
    before = parse_trace(chosen)
    after = parse_trace(result.text)
    if not before.steps or not after.steps:
        reasons.append("unparseable_trace")
    if result.error_type == "missing_node" and len(after.steps) >= len(before.steps):
        reasons.append("missing_node_did_not_remove_step")
    if result.error_type == "spurious_node":
        new_nodes = {name for name, _ in after.definitions} - {name for name, _ in before.definitions}
        used_as_dependency = {source for source, _target in after.edges}
        if len(after.steps) <= len(before.steps) or len(new_nodes) != 1:
            reasons.append("spurious_node_did_not_insert_one_node")
        if not new_nodes or not any(name in used_as_dependency for name in new_nodes):
            reasons.append("spurious_node_not_used_downstream")
    if result.error_type == "extra_edge" and after.edge_count <= before.edge_count:
        reasons.append("extra_edge_did_not_add_dependency")
    if result.error_type == "disorder" and not after.forward_references:
        reasons.append("disorder_did_not_create_forward_reference")
    return not reasons, reasons
