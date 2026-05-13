#!/usr/bin/env python3
"""
Stage 1: DAG Parser
===================
Parses a structured math reasoning solution (using "Define X as ...;" pattern)
into a directed acyclic graph (DAG), where each node represents one reasoning step
and edges encode variable-level data dependencies.

The resulting ReasoningDAG powers the corruption engine (stage1_dag_corruptor.py),
which introduces four types of reasoning errors:
  1. computer_error     — perturb one intermediate number
  2. dependency_mismatch— replace one variable reference with another
  3. missing_nodes      — delete 1–2 critical intermediate steps
  4. disorder           — violate topological order by swapping two steps

Author: Reasoning-DPO Pipeline
"""

from __future__ import annotations

import re
import copy
import random
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Node:
    """One reasoning step in the DAG."""
    var: str                    # variable name introduced by this step
    desc: str                   # natural-language description of what is defined
    step_text: str              # full text of the step, e.g. "Define a as 5;"
    dependencies: List[str]      # names of previously-defined variables used
    step_idx: int               # index in the original order

    def __repr__(self) -> str:
        deps = ", ".join(self.dependencies) if self.dependencies else "∅"
        return f"Node({self.var} | deps=[{deps}] | {self.step_text[:40]})"


@dataclass
class ReasoningDAG:
    """
    Full DAG representation of one solution.
    
    The DAG is extracted from the preamble + ordered step list.
    Each node's `dependencies` field encodes the edges.
    """
    preamble: str          # text before the first "Define" block
    nodes: Dict[str, Node] # keyed by variable name
    order: List[str]       # topologically-sorted variable names
    tail: str              # text after the last "Define" block

    # ----- derived views -----
    def edges(self) -> List[Tuple[str, str]]:
        """Return all (dependency, variable) edges."""
        out = []
        for v in self.order:
            for u in self.nodes[v].dependencies:
                out.append((u, v))
        return out

    def indegree(self, var: str) -> int:
        return len(self.nodes[var].dependencies)

    def outdegree(self, var: str) -> int:
        cnt = 0
        for v in self.order:
            if var in self.nodes[v].dependencies:
                cnt += 1
        return cnt

    def ordered_steps(self) -> List[str]:
        """Return step texts in topological order."""
        return [self.nodes[v].step_text for v in self.order]

    def reconstruct_solution(self, custom_steps: Optional[List[str]] = None) -> str:
        """
        Reconstruct the full solution text.
        If custom_steps is provided, use those instead of the stored steps
        (used by the corruptor to build mutated versions).
        """
        pieces = []
        if self.preamble.strip():
            pieces.append(self.preamble.strip())
        steps = custom_steps if custom_steps is not None else self.ordered_steps()
        for s in steps:
            s = s.strip()
            if s:
                pieces.append(s)
        if self.tail.strip():
            pieces.append(self.tail.strip())
        return " ".join(pieces).strip()

    def to_dict(self) -> dict:
        """Serialise for JSON / debugging."""
        return {
            "preamble": self.preamble,
            "order": self.order,
            "tail": self.tail,
            "nodes": {k: asdict(v) for k, v in self.nodes.items()},
        }

    def summary(self) -> str:
        lines = [
            f"=== ReasoningDAG ===",
            f"  Preamble: {self.preamble[:80]!r}",
            f"  Steps ({len(self.order)}):",
        ]
        for v in self.order:
            node = self.nodes[v]
            lines.append(f"    [{node.step_idx}] {node.var}  deps={node.dependencies}")
        lines.append(f"  Tail: {self.tail[:60]!r}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class DAGParser:
    """
    Heuristic DAG parser for structured math-CoT solutions.
    
    Expected solution format::
    
        <preamble>
        Define <desc> as <var>;
        Define <desc> as <var>;
        ...
        Define <desc> as <var>;
        <tail>
    
    Dependencies are inferred by checking which previously-defined variable
    names appear in the current step text.
    """

    # Split the solution into individual Define blocks
    DEFINE_BLOCK_RE = re.compile(
        r"(Define\s+.*?)(?=(?:\s+Define\s+|\s+We\s+know\s+|\s+Answer:|$))",
        re.DOTALL,
    )

    # Parse one block: "Define <desc> as <var>;"  or  "Define <var> = <expr>;"
    DEFINE_HEAD_RE = re.compile(
        r"^Define\s+(.*?)\s+as\s+([A-Za-z][A-Za-z0-9_]*)\s*;\s*(.*)$",
        re.DOTALL,
    )

    # Alternative: "Define X = <expr>;"
    DEFINE_ALT_RE = re.compile(
        r"^Define\s+([A-Za-z][A-Za-z0-9_]*)\s*=\s*(.+)\s*;?\s*$",
        re.DOTALL,
    )

    def split_solution(self, solution: str) -> Tuple[str, List[str], str]:
        """
        Split a raw solution into (preamble, define_blocks, tail).
        
        - preamble: text before the first "Define"
        - define_blocks: list of raw "Define ..." text snippets
        - tail: text after the last Define block
        """
        solution = solution.strip()
        first_define = solution.find("Define ")
        if first_define == -1:
            raise ValueError("No 'Define' block found in solution.")

        preamble = solution[:first_define].strip()

        define_region = solution[first_define:]
        define_blocks = [m.group(1).strip() for m in self.DEFINE_BLOCK_RE.finditer(define_region)]
        define_blocks = [b for b in define_blocks if b]

        if not define_blocks:
            raise ValueError("No valid define blocks parsed.")

        # The tail is whatever follows the last block
        last_start = solution.find(define_blocks[-1])
        last_end = last_start + len(define_blocks[-1])
        tail = solution[last_end:].strip()

        return preamble, define_blocks, tail

    def parse(self, solution: str) -> ReasoningDAG:
        """
        Parse a raw solution string into a ReasoningDAG.
        
        Raises
        ------
        ValueError
            If the solution does not conform to the expected format.
        """
        preamble, define_blocks, tail = self.split_solution(solution)

        nodes: Dict[str, Node] = {}
        order: List[str] = []
        defined_so_far: List[str] = []

        for step_idx, block in enumerate(define_blocks):
            # Primary pattern: "Define <desc> as <var>;"
            m = self.DEFINE_HEAD_RE.match(block)
            if m:
                desc = m.group(1).strip()
                var = m.group(2).strip()
                # dependencies are variables used in the body (after ";")
                body = m.group(3).strip()
            else:
                # Fallback: "Define <var> = <expr>;"
                m2 = self.DEFINE_ALT_RE.match(block)
                if not m2:
                    continue
                desc = ""
                var = m2.group(1).strip()
                body = m2.group(2).strip()

            # Collect dependencies: any previously-defined variable that appears in body
            deps = _find_all_vars_in_text(body, defined_so_far)
            deps = [d for d in deps if d != var]  # no self-loops

            node = Node(
                var=var,
                desc=desc,
                step_text=block.strip(),
                dependencies=deps,
                step_idx=step_idx,
            )
            nodes[var] = node
            order.append(var)
            defined_so_far.append(var)

        if not order:
            raise ValueError("No nodes parsed from solution.")

        return ReasoningDAG(
            preamble=preamble,
            nodes=nodes,
            order=order,
            tail=tail,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_all_vars_in_text(text: str, candidate_vars: List[str]) -> List[str]:
    """
    Return all candidate variable names that appear as whole words in `text`.
    Preserves the order of first appearance.
    """
    seen = set()
    out = []
    for v in candidate_vars:
        if re.search(rf"\b{re.escape(v)}\b", text):
            if v not in seen:
                seen.add(v)
                out.append(v)
    return out


def perturb_integer_str(n: str) -> str:
    """
    Perturb a positive integer string by ±1 or ±2.
    Never returns the same value.
    """
    v = int(n)
    if v == 0:
        return "1"
    delta = random.choice([-2, -1, 1, 2])
    new_v = v + delta
    if new_v <= 0:
        new_v = v + abs(delta)
    if new_v == v:
        new_v = v + 1
    return str(new_v)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_json_or_jsonl(path: str) -> List[dict]:
    """Load data from a .json (list) or .jsonl file."""
    path = Path(path)
    if path.suffix == ".jsonl":
        data = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(__import__("json").loads(line))
        return data
    elif path.suffix == ".json":
        with open(path, encoding="utf-8") as f:
            obj = __import__("json").load(f)
        if isinstance(obj, list):
            return obj
        raise ValueError("JSON file must contain a list of examples.")
    else:
        raise ValueError("Only .json and .jsonl are supported.")


def save_jsonl(data: List[dict], path: str) -> None:
    """Write a list of dicts as JSON Lines."""
    with open(path, "w", encoding="utf-8") as f:
        for ex in data:
            f.write(__import__("json").dumps(ex, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _debug_one_record(parser: DAGParser, record: dict) -> None:
    """Pretty-print the parsed DAG for the first record (--debug flag)."""
    solution = record.get("solution", "").strip()
    if not solution:
        print("[DEBUG] No solution field found.")
        return
    try:
        dag = parser.parse(solution)
        print(dag.summary())
        print("\nEdges:")
        for u, v in dag.edges():
            print(f"  {u} → {v}")
    except Exception as e:
        print(f"[DEBUG] Parse error: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="DAG Parser — parse math solutions into DAGs")
    parser.add_argument("--input_path", required=True, help="Path to .json or .jsonl source data")
    parser.add_argument("--debug_first", action="store_true", help="Pretty-print parsed DAG for the first record")
    args = parser.parse_args()

    data = load_json_or_jsonl(args.input_path)
    dag_parser = DAGParser()

    if args.debug_first and data:
        print(f"\n[DEBUG] Inspecting first record: {data[0].get('id', 'unknown')}")
        _debug_one_record(dag_parser, data[0])
    else:
        stats = {"ok": 0, "fail": 0}
        for record in data:
            try:
                dag_parser.parse(record["solution"])
                stats["ok"] += 1
            except Exception:
                stats["fail"] += 1
        total = len(data)
        print(f"Parsed {total} records: {stats['ok']} ok, {stats['fail']} failed "
              f"({stats['fail']/max(total,1):.1%} error rate)")


if __name__ == "__main__":
    main()
