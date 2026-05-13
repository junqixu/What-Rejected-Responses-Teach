"""
Tests for Stage 1 — DAG Parser and Corruptor
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage1_dag_preference"))

from dag_parser import DAGParser, ReasoningDAG, load_json_or_jsonl, perturb_integer_str
from dag_corruptor import DAGCorruptor, CORRUPTION_TYPES


# ---- Fixtures ----

SAMPLE_SOLUTION = (
    "Define a as 4. Define b as 3. so c = a + b. so d = c * 2. Answer: 14"
)

PARSEABLE_SOLUTION = (
    "We have two numbers. Define first_num as 7. "
    "Define second_num as 5. so sum_val = first_num + second_num. "
    "so product_val = sum_val * 2. Answer: 24"
)

MULTI_DEFINE_SOLUTION = (
    "Define x as 10. Define y as 3. so z = x / y. so w = z * 2. "
    "Define t as 5. so u = w + t. Answer: 11.67"
)


class TestDAGParser:
    def test_parse_basic(self):
        dag = DAGParser().parse(PARSEABLE_SOLUTION)
        assert len(dag.order) >= 2, f"Expected >= 2 nodes, got {len(dag.order)}"
        assert dag.preamble.strip() != "", "Expected non-empty preamble"
        assert all(isinstance(n.var, str) for n in dag.nodes.values())

    def test_parse_multi_define(self):
        dag = DAGParser().parse(MULTI_DEFINE_SOLUTION)
        assert len(dag.order) >= 4
        edges = dag.edges()
        assert len(edges) >= 2, "Expected at least 2 edges"

    def test_edges(self):
        dag = DAGParser().parse(PARSEABLE_SOLUTION)
        edges = dag.edges()
        assert isinstance(edges, list)
        for e in edges:
            assert len(e) == 2

    def test_reconstruct(self):
        dag = DAGParser().parse(PARSEABLE_SOLUTION)
        recon = dag.reconstruct_solution()
        assert isinstance(recon, str)
        assert len(recon) > 0

    def test_to_dict(self):
        dag = DAGParser().parse(PARSEABLE_SOLUTION)
        d = dag.to_dict()
        assert "preamble" in d
        assert "order" in d
        assert "nodes" in d
        assert "tail" in d

    def test_indegree_outdegree(self):
        dag = DAGParser().parse(PARSEABLE_SOLUTION)
        for var in dag.order:
            assert dag.indegree(var) >= 0
            assert dag.outdegree(var) >= 0

    def test_parse_fails_on_bad_input(self):
        parser = DAGParser()
        try:
            parser.parse("This has no Define blocks at all")
            assert False, "Expected ValueError"
        except ValueError:
            pass


class TestCorruptor:
    def test_computer_error(self):
        dag = DAGParser().parse(PARSEABLE_SOLUTION)
        corruptor = DAGCorruptor(dag)
        result = corruptor.make_computer_error()
        assert result is not None, "computer_error should succeed on this solution"
        assert result != PARSEABLE_SOLUTION

    def test_dependency_mismatch(self):
        dag = DAGParser().parse(MULTI_DEFINE_SOLUTION)
        corruptor = DAGCorruptor(dag)
        result = corruptor.make_dependency_mismatch()
        assert result is not None, "dependency_mismatch should succeed"
        assert result != MULTI_DEFINE_SOLUTION

    def test_missing_nodes(self):
        dag = DAGParser().parse(MULTI_DEFINE_SOLUTION)
        corruptor = DAGCorruptor(dag)
        result = corruptor.make_missing_nodes()
        assert result is not None, "missing_nodes should succeed"
        assert len(result) <= len(MULTI_DEFINE_SOLUTION)

    def test_disorder(self):
        dag = DAGParser().parse(MULTI_DEFINE_SOLUTION)
        corruptor = DAGCorruptor(dag)
        result = corruptor.make_disorder()
        assert result is not None, "disorder should succeed"
        assert result != MULTI_DEFINE_SOLUTION

    def test_all_corruption_types_valid(self):
        dag = DAGParser().parse(MULTI_DEFINE_SOLUTION)
        corruptor = DAGCorruptor(dag)
        for et in CORRUPTION_TYPES:
            result = corruptor.corrupt(et)
            # At least some should succeed; none should raise
            assert isinstance(result, (str, type(None)))

    def test_bad_error_type_raises(self):
        import pytest
        dag = DAGParser().parse(PARSEABLE_SOLUTION)
        corruptor = DAGCorruptor(dag)
        try:
            corruptor.corrupt("nonexistent_error")
            assert False, "Should raise ValueError"
        except ValueError:
            pass


class TestUtilityFunctions:
    def test_perturb_integer_str(self):
        # Should never return same value for positive integers
        for n in ["5", "10", "100", "999"]:
            new_n = perturb_integer_str(n)
            assert new_n != n or n == "1"
            assert int(new_n) > 0

    def test_perturb_zero(self):
        # 0 should be perturbed to a positive number
        new_n = perturb_integer_str("0")
        assert int(new_n) >= 1

    def test_corruption_types_list(self):
        assert len(CORRUPTION_TYPES) == 4
        assert "computer_error" in CORRUPTION_TYPES
        assert "dependency_mismatch" in CORRUPTION_TYPES
        assert "missing_nodes" in CORRUPTION_TYPES
        assert "disorder" in CORRUPTION_TYPES
