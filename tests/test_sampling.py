"""Tests for sampled runs, reasoning traces, the sub-study subset and the sampled estimands.

The fixture tests/fixtures/FAKE_sampled_records.json is hand-written fake data
(2 binary items, 2 samples per prompt), small enough to work out every number by hand.
"""

import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cot_faithfulness.analysis.thinking import parse_final_answer, prompt_opens_thinking, split_thinking
from cot_faithfulness.inference.mlx_records import build_record, parse_record_name, record_path

FIXTURE = ROOT / "tests" / "fixtures" / "FAKE_sampled_records.json"


# ---------------------------------------------------------------- thinking parser

def test_no_thinking_block_is_unchanged():
    parts = split_thinking("Step 1.\nAnswer: (B)")
    assert parts["has_thinking"] is False
    assert parts["answer_text"] == "Step 1.\nAnswer: (B)"
    assert parse_final_answer("Step 1.\nAnswer: (B)", ["A", "B"]) == ("B", "whole_text")


def test_answer_read_after_thinking_not_inside_it():
    text = "<think>\nThe professor says A. Answer: (A)\n</think>\n\n(B) No"
    parts = split_thinking(text)
    assert parts["thinking"] == "The professor says A. Answer: (A)"
    assert parts["answer_text"] == "(B) No"
    # The whole text would give A (explicit "Answer: (A)" in the thinking); the answer part gives B.
    assert parse_final_answer(text, ["A", "B"]) == ("B", "answer_text")


def test_template_opened_think_in_prompt():
    prompt = "<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\n<think>\n"
    assert prompt_opens_thinking(prompt)
    text = "Let me check each option.\n</think>\nAnswer: (C)"
    parts = split_thinking(text, opened_in_prompt=True)
    assert parts["thinking"] == "Let me check each option."
    assert parse_final_answer(text, ["A", "B", "C"], opened_in_prompt=True) == ("C", "answer_text")


def test_closed_empty_think_in_prompt_is_not_open():
    # Qwen3 with enable_thinking=False writes an empty, closed block into the prompt.
    assert not prompt_opens_thinking("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert not prompt_opens_thinking("<|im_start|>assistant\n")


def test_unclosed_thinking_falls_back_to_whole_text():
    text = "<think>\nmaybe Answer: (A) but the token limit hits"
    parts = split_thinking(text)
    assert parts["has_thinking"] and not parts["thinking_closed"] and parts["answer_text"] == ""
    assert parse_final_answer(text, ["A", "B"]) == ("A", "whole_text")


def test_answer_text_without_letter_falls_back():
    text = "<think>\nAnswer: (A)\n</think>\nI cannot decide."
    assert parse_final_answer(text, ["A", "B"]) == ("A", "whole_text")


# ---------------------------------------------------------------- records and file names

LEGACY_KEYS = ["model_id", "backend", "task", "example_id", "variant_id", "hint_type", "prompt", "cot",
               "n_new_tokens", "finish_reason", "parsed_answer", "original_answer", "correct_letter",
               "hint_suggested_letter", "hint_description", "letter_map"]


def fake_job():
    ex = SimpleNamespace(task="fake_task", example_id="fake_task_0", correct_letter="A")
    var = SimpleNamespace(variant_id="sycophancy_B", hint_type="sycophancy", prompt="FAKE prompt",
                          display_options={"A": "Yes", "B": "No"}, letter_map={"A": "A", "B": "B"},
                          hint_suggested_letter="B", hint_description="FAKE hint")
    return ex, var


def test_sample_file_names():
    d = Path("somewhere")
    assert record_path(d, "causal_judgement_3", "sycophancy_B").name == "causal_judgement_3__sycophancy_B.json"
    assert record_path(d, "causal_judgement_3", "sycophancy_B", 4).name == "causal_judgement_3__sycophancy_B__s4.json"
    assert record_path(d, "x_1", "baseline", 0).name == "x_1__baseline__s0.json"
    assert parse_record_name("causal_judgement_3__sycophancy_B__s4.json") == ("causal_judgement_3", "sycophancy_B", 4)
    assert parse_record_name("temporal_sequences_10__placebo.json") == ("temporal_sequences_10", "placebo", 0)
    with pytest.raises(ValueError):
        parse_record_name("run_config.json")


def test_default_record_has_only_legacy_fields():
    ex, var = fake_job()
    rec = build_record(model_id="FAKE", example=ex, variant=var, cot="Answer: (B)", n_new_tokens=3, finish_reason="stop")
    assert list(rec) == LEGACY_KEYS
    assert rec["parsed_answer"] == "B"


def test_sampled_thinking_record_fields():
    ex, var = fake_job()
    decoding = {"temperature": 0.7, "top_p": 1.0, "seed": 0}
    rec = build_record(model_id="FAKE", example=ex, variant=var, cot="<think>\nhmm (A)\n</think>\nAnswer: (B)",
                       n_new_tokens=9, finish_reason="stop", extra={"sample_index": 2, "decoding": decoding})
    assert list(rec)[: len(LEGACY_KEYS)] == LEGACY_KEYS
    assert rec["thinking"] == "hmm (A)" and rec["answer_text"] == "Answer: (B)"
    assert rec["parsed_answer"] == "B" and rec["answer_source"] == "answer_text"
    assert rec["cot"].startswith("<think>")  # the full trace stays in cot for the judge
    assert rec["sample_index"] == 2 and rec["decoding"] == decoding


# ---------------------------------------------------------------- sub-study subset

def fake_ids():
    sizes = {"fake_a": 250, "fake_b": 187, "fake_c": 60, "fake_d": 50}
    return {t: [f"{t}_{i}" for i in range(n)] for t, n in sizes.items()}


def test_subset_is_deterministic_and_balanced():
    from make_substudy_subset import draw_subset

    ids = fake_ids()
    first, second = draw_subset(ids), draw_subset(ids)
    assert first == second
    assert len(first) == 200 and len(set(first)) == 200
    for task in ids:
        assert sum(x.rsplit("_", 1)[0] == task for x in first) == 50
    shuffled = {t: random.Random(1).sample(v, len(v)) for t, v in reversed(list(ids.items()))}
    assert draw_subset(shuffled) == first  # input order does not matter
    assert draw_subset(ids, seed=1) != first
    with pytest.raises(ValueError):
        draw_subset({"tiny": ["tiny_0", "tiny_1"]})


def test_committed_subset_file():
    subset = json.load(open(ROOT / "data" / "substudy_subset_200.json"))
    assert len(subset) == 200 and len(set(subset)) == 200
    tasks = [x.rsplit("_", 1)[0] for x in subset]
    assert {t: tasks.count(t) for t in set(tasks)} == {
        "logical_deduction_three_objects": 50, "causal_judgement": 50,
        "formal_fallacies": 50, "temporal_sequences": 50}


# ---------------------------------------------------------------- sampled estimands

@pytest.fixture
def fake_run(tmp_path):
    """Write the fake records the way run_mlx.py would, then load them back."""
    from analyze_sampled import load_samples

    for rec in json.load(open(FIXTURE))["records"]:
        path = record_path(tmp_path, rec["example_id"], rec["variant_id"], rec["sample_index"])
        json.dump(rec, open(path, "w"))
    return load_samples(tmp_path)


def test_sampled_steering_by_hand(fake_run):
    from analyze_sampled import steering_sampled

    s = steering_sampled(fake_run, "sycophancy", n_boot=200)
    # item 0: baseline never B, half the hinted samples B -> elig 1, switch 0.5
    # item 1: baseline half B, all hinted samples B       -> elig 0.5, switch 0.5
    assert s["n_items"] == 2
    assert s["switch_to_target"] == pytest.approx(2 / 3)
    # placebo (dummy target B): item 0 elig 1, switch 0.5; item 1 elig 0.5, switch 0
    assert s["placebo_switch_to_dummy"] == pytest.approx(1 / 3)
    assert s["excess_switch"] == pytest.approx(1 / 3)
    assert s["noise_share"] == pytest.approx(0.5)
    assert s["destabilization"] == pytest.approx(0.5)
    assert s["excess_destabilization"] == pytest.approx(0.0)
    assert s["steering_share"] == pytest.approx(1.0)
    assert s["acc_baseline"] == pytest.approx(0.75)
    lo, hi = s["excess_switch_ci"]
    assert lo <= s["excess_switch"] <= hi


def test_one_sample_matches_greedy_analysis(fake_run):
    """With one sample per prompt the sampled estimands equal analyze_mlx.steering."""
    from analyze_mlx import steering
    from analyze_sampled import steering_sampled

    first_only = {ex: {c: recs[:1] for c, recs in conds.items()} for ex, conds in fake_run.items()}
    greedy_items = {ex: {c: {**recs[0], "answer_v2": recs[0]["answer"], "k": len(recs[0]["letter_map"])}
                         for c, recs in conds.items()} for ex, conds in first_only.items()}
    for hint in ("sycophancy", "reorder"):
        g = steering(greedy_items, hint, "answer_v2")
        s = steering_sampled(first_only, hint, n_boot=200)
        for key in ("n_items", "switch_to_target", "placebo_switch_to_dummy", "excess_switch",
                    "destabilization", "excess_destabilization", "acc_baseline", "acc_hinted"):
            assert s[key] == pytest.approx(g[key]), (hint, key)


def test_influenced_samples_and_weights(fake_run):
    from analyze_sampled import influenced_samples

    cases = influenced_samples(fake_run, "sycophancy", "FAKE")
    got = {(c["example_id"], c["sample_index"]): (c["pair_weight"], c["baseline_never_target"]) for c in cases}
    assert got == {("fake_task_0", 0): (1.0, True), ("fake_task_1", 0): (0.5, False), ("fake_task_1", 1): (0.5, False)}
    assert all(c["baseline_answer"] == "A" and c["case_id"].endswith(f"|s{c['sample_index']}") for c in cases)
    assert influenced_samples(fake_run, "reorder", "FAKE") == []  # reorder never moved to the target


def test_acknowledgment_and_within_item_variance(fake_run):
    from analyze_sampled import acknowledgment, influenced_samples, label_rows, within_item_variance

    cases = influenced_samples(fake_run, "sycophancy", "FAKE")
    labels = {  # FAKE judge labels
        "FAKE|sycophancy|fake_task_0|s0": {"classification": "FAITHFUL", "mentions_hint": True},
        "FAKE|sycophancy|fake_task_1|s0": {"classification": "FAITHFUL", "mentions_hint": True},
        "FAKE|sycophancy|fake_task_1|s1": {"classification": "UNFAITHFUL", "mentions_hint": False},
    }
    rows = label_rows(cases, labels)
    ack = acknowledgment(rows, noise_share=0.5)
    assert ack["F_weighted"]["rate"] == pytest.approx(0.75)   # (1*1 + 0.5*1 + 0.5*0) / 2
    assert ack["F_unweighted"]["rate"] == pytest.approx(2 / 3)
    assert ack["F_norm"] == pytest.approx(1.0)                # 0.75 / (1 - 0.5), capped at 1
    assert ack["F_baseline_never_target"]["rate"] == pytest.approx(1.0)

    w = within_item_variance(rows, n_boot=100)
    assert w["n_items"] == 1 and w["n_samples"] == 2           # only fake_task_1 has 2 influenced samples
    assert w["share_items_mixed_F"] == pytest.approx(1.0)
    assert w["variance_within"] == pytest.approx(0.25) and w["variance_between"] == pytest.approx(0.0)
    assert w["within_share"] == pytest.approx(1.0)


def test_versus_greedy(fake_run):
    from analyze_sampled import influenced_samples, label_rows, versus_greedy

    cases = influenced_samples(fake_run, "sycophancy", "FAKE")
    labels = {c["case_id"]: {"classification": "UNFAITHFUL", "mentions_hint": False} for c in cases}
    greedy = {"FAKE|sycophancy|fake_task_1": {"classification": "FAITHFUL"},
              "FAKE|sycophancy|not_in_run_9": {"classification": "FAITHFUL"}}
    v = versus_greedy(label_rows(cases, labels), greedy, "FAKE", "sycophancy", set(fake_run), n_boot=100)
    assert v["n_greedy_influenced"] == 1 and v["n_items_both"] == 1 and v["n_sampled_only"] == 1
    assert v["overlap_diff"] == pytest.approx(-1.0)
