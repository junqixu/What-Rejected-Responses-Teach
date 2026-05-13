#!/usr/bin/env python3
"""
Stage 1: DAG Corruptor
=======================
Takes a parsed ReasoningDAG and produces a structurally "wrong" version that
serves as the DPO *rejected* response. Four corruption strategies are implemented:

  1. computer_error
     - Find a step with an explicit number and equation (e.g. "so a = b + 5;")
     - Change the last number by ±1 or ±2
     - Leaves the DAG topology unchanged; only the numerical content differs

  2. dependency_mismatch
     - Find a non-source node (has dependencies)
     - In its step text, replace one referenced variable with a different one
       that was defined earlier but is not a true dependency
     - Keeps step text plausible but violates the true dependency edge

  3. missing_nodes
     - Delete 1–2 intermediate steps that are referenced by later steps
     - The remaining trace contains dangling references to undefined variables
     - Represents an incomplete reasoning chain

  4. disorder
     - Pick a real dependency edge u → v
     - Move the step for v immediately before the step for u
     - This guarantees a topological-order violation

All methods return ``None`` if the corruption cannot be safely applied,
which signals the caller to discard that sample.

Author: Reasoning-DPO Pipeline
"""

from __future__ import annotations

import re
import random
from typing import Optional, List

from dag_parser import ReasoningDAG, perturb_integer_str, _find_all_vars_in_text


# ---------------------------------------------------------------------------
# Corruption strategies
# ---------------------------------------------------------------------------

class DAGCorruptor:
    """Applies structural corruption to a ReasoningDAG."""

    def __init__(self, dag: ReasoningDAG) -> None:
        self.dag = dag

    # ------------------------------------------------------------------
    # Strategy 1: computer_error
    # ------------------------------------------------------------------
    def make_computer_error(self) -> Optional[str]:
        """
        Perturb the last integer in one numerical step.
        Preference: non-source nodes first (they have local effects only).
        """
        candidates = []
        for v in self.dag.order:
            step = self.dag.nodes[v].step_text
            if "Answer:" in step or "Define a as " in step:
                continue
            if "=" in step and re.search(r"\b\d+\b", step):
                candidates.append(v)

        if not candidates:
            return None

        # Prefer non-source nodes
        candidates.sort(key=lambda x: (self.dag.indegree(x) == 0, self.dag.nodes[x].step_idx))

        chosen_var = random.choice(candidates)
        step = self.dag.nodes[chosen_var].step_text

        matches = list(re.finditer(r"\b\d+\b", step))
        if not matches:
            return None

        # Change the last number (usually the result of an intermediate computation)
        target = matches[-1]
        old_num = target.group(0)
        new_num = perturb_integer_str(old_num)

        new_step = step[:target.start()] + new_num + step[target.end():]

        steps = self.dag.ordered_steps()
        steps[self.dag.nodes[chosen_var].step_idx] = new_step
        return self.dag.reconstruct_solution(custom_steps=steps)

    # ------------------------------------------------------------------
    # Strategy 2: dependency_mismatch
    # ------------------------------------------------------------------
    def make_dependency_mismatch(self) -> Optional[str]:
        """
        Replace one correct variable reference with another earlier variable.
        Tries two patterns:

        Pattern A (direct):   "so e = j = 5"  →  "so e = C = 5"
        Pattern B (body):      "so a = b + c"  →  "so a = C + c"
        """
        # Collect non-source nodes
        candidate_vars = [
            v for v in self.dag.order
            if len(self.dag.nodes[v].dependencies) > 0
        ]
        if not candidate_vars:
            return None

        random.shuffle(candidate_vars)

        # ---- Pattern A: "so target = old_source = ..." ----
        for chosen_var in candidate_vars:
            node = self.dag.nodes[chosen_var]
            step = node.step_text
            earlier_vars = self.dag.order[: node.step_idx]
            if not earlier_vars:
                continue

            direct_pattern = (
                rf"(so\s+{re.escape(chosen_var)}\s*=\s*)"
                rf"([A-Za-z][A-Za-z0-9_]*)"
                rf"(\s*=)"
            )
            m = re.search(direct_pattern, step)
            if m:
                old_source = m.group(2)
                replacement_pool = [
                    v for v in earlier_vars
                    if v != old_source and v != chosen_var
                ]
                if not replacement_pool:
                    continue
                new_source = random.choice(replacement_pool)
                new_step = (
                    step[: m.start(2)]
                    + new_source
                    + step[m.end(2):]
                )
                steps = self.dag.ordered_steps()
                steps[node.step_idx] = new_step
                return self.dag.reconstruct_solution(custom_steps=steps)

        # ---- Pattern B: replace a dependency reference in the body ----
        random.shuffle(candidate_vars)
        for chosen_var in candidate_vars:
            node = self.dag.nodes[chosen_var]
            step = node.step_text
            earlier_vars = self.dag.order[: node.step_idx]

            replacement_pool_global = [x for x in earlier_vars if x != chosen_var]
            if not replacement_pool_global:
                continue

            deps = list(node.dependencies)
            random.shuffle(deps)

            for old_dep in deps:
                replacement_pool = [x for x in replacement_pool_global if x != old_dep]
                if not replacement_pool:
                    continue

                new_dep = random.choice(replacement_pool)
                pattern = rf"\b{re.escape(old_dep)}\b"
                if not re.search(pattern, step):
                    continue

                new_step = re.sub(pattern, new_dep, step, count=1)
                if new_step == step:
                    continue

                steps = self.dag.ordered_steps()
                steps[node.step_idx] = new_step
                return self.dag.reconstruct_solution(custom_steps=steps)

        return None

    # ------------------------------------------------------------------
    # Strategy 3: missing_nodes
    # ------------------------------------------------------------------
    def make_missing_nodes(self) -> Optional[str]:
        """
        Delete 1–2 intermediate steps that are referenced by later steps.
        This creates dangling references in the remaining text.
        """
        # First try: prefer nodes used by later steps (outdegree > 0)
        # but not the very last or first step
        candidates = []
        for v in self.dag.order:
            node = self.dag.nodes[v]
            outdeg = self.dag.outdegree(v)
            if 0 < node.step_idx < len(self.dag.order) - 1 and outdeg > 0:
                candidates.append(v)

        # Fallback: any middle node
        if not candidates:
            for v in self.dag.order:
                node = self.dag.nodes[v]
                if 0 < node.step_idx < len(self.dag.order) - 1:
                    candidates.append(v)

        if not candidates:
            return None

        # Decide how many steps to delete
        n_delete = random.choice([1, 2]) if len(candidates) >= 2 else 1

        # Pick the steps to delete, preferring lower outdegree first
        chosen = random.sample(candidates, min(n_delete, len(candidates)))
        chosen.sort(key=lambda v: self.dag.nodes[v].step_idx, reverse=True)

        steps = self.dag.ordered_steps()
        for var in chosen:
            idx = self.dag.nodes[var].step_idx
            if 0 <= idx < len(steps):
                steps.pop(idx)

        return self.dag.reconstruct_solution(custom_steps=steps)

    # ------------------------------------------------------------------
    # Strategy 4: disorder
    # ------------------------------------------------------------------
    def make_disorder(self) -> Optional[str]:
        """
        Pick a real dependency edge u → v and move step v before step u.
        This guarantees a topological-order violation (use-before-define).
        """
        edges = self.dag.edges()
        if not edges:
            return None

        random.shuffle(edges)
        for u, v in edges:
            u_idx = self.dag.nodes[u].step_idx
            v_idx = self.dag.nodes[v].step_idx
            if u_idx >= v_idx:
                continue

            steps = self.dag.ordered_steps()
            v_step = steps[v_idx]

            steps.pop(v_idx)
            steps.insert(u_idx, v_step)
            return self.dag.reconstruct_solution(custom_steps=steps)

        return None

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------
    def corrupt(self, error_type: str) -> Optional[str]:
        """
        Apply the named corruption strategy.
        
        Parameters
        ----------
        error_type : str
            One of {"computer_error", "dependency_mismatch", "missing_nodes", "disorder"}.
        
        Returns
        -------
        str or None
            The corrupted solution text, or None if corruption could not be applied.
        """
        dispatch = {
            "computer_error": self.make_computer_error,
            "dependency_mismatch": self.make_dependency_mismatch,
            "missing_nodes": self.make_missing_nodes,
            "disorder": self.make_disorder,
        }
        if error_type not in dispatch:
            raise ValueError(f"Unknown error_type: {error_type!r}")
        return dispatch[error_type]()


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

CORRUPTION_STRATEGIES = list(DAGCorruptor(DAGParser=None).corrupt.__code__.co_names[:0] == [] and ["computer_error", "dependency_mismatch", "missing_nodes", "disorder"])
# Note: the above is a quirky idiom; we expose the list directly below

CORRUPTION_TYPES = [
    "computer_error",
    "dependency_mismatch",
    "missing_nodes",
    "disorder",
]


def corrupt_solution(solution: str, error_type: str, seed: int = 42) -> Optional[str]:
    """
    One-shot corruption of a single solution string.
    
    Returns None if parsing or corruption fails.
    """
    random.seed(seed)
    from dag_parser import DAGParser
    parser = DAGParser()
    try:
        dag = parser.parse(solution)
    except Exception:
        return None
    corruptor = DAGCorruptor(dag)
    return corruptor.corrupt(error_type)


# ---------------------------------------------------------------------------
# CLI (standalone test)
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse, json, sys
    from dag_parser import load_json_or_jsonl

    ap = argparse.ArgumentParser(description="DAGCorruptor standalone test")
    ap.add_argument("--input_path", required=True)
    ap.add_argument("--error_type", default="disorder",
                    choices=CORRUPTION_TYPES)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=5, help="Number of samples to corrupt")
    args = ap.parse_args()

    random.seed(args.seed)
    data = load_json_or_jsonl(args.input_path)

    from dag_parser import DAGParser
    parser = DAGParser()
    corruptor_class = DAGCorruptor

    corruptors_run = 0
    successes = 0
    for record in data[: args.n]:
        solution = record.get("solution", "").strip()
        if not solution:
            continue
        try:
            dag = parser.parse(solution)
        except Exception as e:
            print(f"[{record.get('id','?')}] parse error: {e}", file=sys.stderr)
            continue
        corruptor = corruptor_class(dag)
        result = corruptor.corrupt(args.error_type)
        corruptors_run += 1
        if result is not None:
            successes += 1
            print(f"\n=== [{record.get('id','?')}] SUCCESS ===")
            print("CHOSEN :", solution[:300])
            print("REJECTED:", result[:300])
        else:
            print(f"\n=== [{record.get('id','?')}] FAILED to corrupt ===")

    print(f"\n{successes}/{corruptors_run} successful corruptions")


if __name__ == "__main__":
    main()
