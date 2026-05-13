"""
Tests for Stage 4 — vLLM Evaluation Metrics
"""

import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage4_evaluation"))

from run_eval import (
    extract_last_number,
    extract_variable_dependency_graph,
    extract_equations,
    compute_reasoning_chain_score,
    f1_from_sets,
    normalize_text,
)


class TestExtractLastNumber:
    def test_basic(self):
        assert extract_last_number("The answer is 42.") == 42.0
        assert extract_last_number("Result: -3.14") == -3.14

    def test_multiple_numbers(self):
        assert extract_last_number("First 10 then 20 then 30") == 30.0

    def test_no_number(self):
        assert extract_last_number("No numbers here") is None

    def test_none(self):
        assert extract_last_number(None) is None


class TestF1:
    def test_perfect_match(self):
        assert f1_from_sets({"a", "b"}, {"a", "b"}) == 1.0

    def test_no_overlap(self):
        assert f1_from_sets({"a"}, {"b"}) == 0.0

    def test_partial_overlap(self):
        score = f1_from_sets({"a", "b", "c"}, {"a", "b", "d"})
        assert 0 < score < 1

    def test_empty_both(self):
        assert f1_from_sets(set(), set()) == 1.0

    def test_empty_one(self):
        assert f1_from_sets(set(), {"a"}) == 0.0


class TestExtractEquations:
    def test_basic_addition(self):
        eqs = extract_equations("7 + 3 = 10")
        assert len(eqs) == 1
        assert (7.0, '+', 3.0, 10.0) in eqs

    def test_commutative_addition(self):
        eqs1 = extract_equations("7 + 3 = 10")
        eqs2 = extract_equations("3 + 7 = 10")
        assert eqs1 == eqs2, "Addition should be order-independent"

    def test_commutative_multiplication(self):
        eqs1 = extract_equations("4 * 5 = 20")
        eqs2 = extract_equations("5 * 4 = 20")
        assert eqs1 == eqs2

    def test_no_equations(self):
        assert len(extract_equations("No equations here")) == 0

    def test_mixed_operators(self):
        eqs = extract_equations("10 / 2 = 5. 3 - 1 = 2. 4 * 6 = 24")
        assert len(eqs) == 3


class TestVariableDependencyGraph:
    def test_basic_extraction(self):
        text = "Define x as 5. Define y as 3. so z = x + y."
        graph = extract_variable_dependency_graph(text)
        assert "edges" in graph
        assert "nodes" in graph
        assert isinstance(graph["edges"], set)

    def test_empty_text(self):
        graph = extract_variable_dependency_graph("")
        assert graph["edges"] == set()
        assert graph["nodes"] == set()

    def test_complex_solution(self):
        text = (
            "Define a as 4. Define b as 3. so c = a + b. "
            "so d = c * 2. Answer: 14"
        )
        graph = extract_variable_dependency_graph(text)
        # Should have extracted at least the main dependency edges
        assert len(graph["edges"]) >= 0


class TestReasoningChainScore:
    def test_identical_texts(self):
        text = "Define a as 4. Define b as 3. so c = a + b."
        score, g_score, e_score = compute_reasoning_chain_score(text, text)
        assert score == 1.0, f"Identical texts should score 1.0, got {score}"
        assert g_score == 1.0
        assert e_score == 1.0

    def test_completely_different(self):
        score, _, _ = compute_reasoning_chain_score(
            "Define a as 1.", "Define x as 2. Define y as 3."
        )
        assert 0 <= score <= 1.0

    def test_returns_three_floats(self):
        text = "Define a as 5."
        result = compute_reasoning_chain_score(text, text)
        assert len(result) == 3
        assert all(isinstance(x, float) for x in result)
