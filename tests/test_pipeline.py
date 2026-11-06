"""Basic tests for analysis pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cot_faithfulness.analysis.bootstrap import bootstrap_ci, faithfulness_summary
from cot_faithfulness.analysis.influence import identify_influenced_cases, parse_answer
from cot_faithfulness.data.bbh import parse_bbh_example


def test_parse_answer():
    cot = "Step 1...\nStep 2...\nAnswer: (B)"
    assert parse_answer(cot, ["A", "B", "C"]) == "B"


def test_bootstrap_ci():
    outcomes = [True, False, False, False]
    result = bootstrap_ci(outcomes, n_bootstrap=1000, seed=0)
    assert result["n"] == 4
    assert result["point_estimate"] == 0.25
    assert result["ci_lower"] <= result["point_estimate"] <= result["ci_upper"]


def test_faithfulness_summary_flags_small_n():
    result = faithfulness_summary([True] * 5 + [False] * 5, min_n=20)
    assert result["too_small"] is True


def test_identify_influenced():
    records = [
        {"model_id": "m", "example_id": "ex1", "variant_id": "baseline", "hint_type": None,
         "original_answer": "A", "cot": "...", "correct_letter": "A"},
        {"model_id": "m", "example_id": "ex1", "variant_id": "sycophancy_B", "hint_type": "sycophancy",
         "original_answer": "B", "hint_suggested_letter": "B", "cot": "hinted", "correct_letter": "A",
         "prompt": "p", "hint_description": "d", "task": "t"},
    ]
    influenced = identify_influenced_cases(records)
    assert len(influenced) == 1
    assert influenced[0].hinted_answer == "B"


def test_bbh_mcq_parse():
    raw = """The following paragraphs each describe a set of three objects arranged in a fixed order. On a branch, there are three birds: a blue jay, a quail, and a falcon. The falcon is to the right of the blue jay. The blue jay is to the right of the quail.
Options:
(A) The blue jay is the second from the left
(B) The quail is the second from the left
(C) The falcon is the second from the left"""
    ex = parse_bbh_example("logical_deduction_three_objects", 0, raw, "(A)")
    assert ex.correct_letter == "A"
    assert len(ex.options) == 3


def test_wilson_saturated():
    from cot_faithfulness.analysis.bootstrap import bootstrap_ci

    result = bootstrap_ci([True] * 9, n_bootstrap=1000, seed=0)
    assert result["method"] == "wilson"
    assert result["point_estimate"] == 1.0
    assert abs(result["ci_lower"] - 0.701) < 0.01
    assert result["ci_upper"] == 1.0 or result["ci_upper"] > 0.99


def test_kappa_not_relaxed():
    from cot_faithfulness.analysis.judge import compute_judge_agreement

    humans = [
        {"example_id": f"e{i}", "hint_type": "sycophancy", "model_id": "m", "human_label": "AMBIGUOUS"}
        for i in range(10)
    ] + [
        {"example_id": f"f{i}", "hint_type": "sycophancy", "model_id": "m", "human_label": "FAITHFUL"}
        for i in range(10)
    ]
    judges = [
        {"example_id": f"e{i}", "hint_type": "sycophancy", "model_id": "m", "classification": "FAITHFUL"}
        for i in range(10)
    ] + [
        {"example_id": f"f{i}", "hint_type": "sycophancy", "model_id": "m", "classification": "FAITHFUL"}
        for i in range(10)
    ]
    out = compute_judge_agreement(humans, judges)
    assert out["exact_agreement_rate"] == 0.5
    assert "relaxed_agreement_rate" not in out
    assert out["kappa"] is not None


if __name__ == "__main__":
    test_parse_answer()
    test_bootstrap_ci()
    test_faithfulness_summary_flags_small_n()
    test_identify_influenced()
    test_bbh_mcq_parse()
    test_wilson_saturated()
    test_kappa_not_relaxed()
    print("All tests passed.")
