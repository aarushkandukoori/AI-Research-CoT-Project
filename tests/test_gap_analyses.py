"""Tests for scripts/gap_analyses.py on small made-up inputs (no real data needed)."""

import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2_contingency

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import gap_analyses as ga  # noqa: E402


def test_mantel_haenszel_single_stratum_matches_textbook():
    a, b, c, d = 12, 38, 5, 45
    mh = ga.mantel_haenszel([(a, b, c, d)])
    assert math.isclose(mh["odds_ratio"], a * d / (b * c))
    # CMH chi-square on one table = (n - 1) / n * Pearson chi-square
    n = a + b + c + d
    pearson = chi2_contingency([[a, b], [c, d]], correction=False)[0]
    assert math.isclose(mh["z"] ** 2, (n - 1) / n * pearson, rel_tol=1e-9)
    assert mh["z"] > 0 and mh["p_one_sided"] < 0.5


def test_mantel_haenszel_identical_strata_keep_the_odds_ratio_and_skip_empty_ones():
    t = (6, 14, 3, 17)
    mh = ga.mantel_haenszel([t, t, (0, 10, 0, 12)])  # last stratum: nobody acknowledges
    assert math.isclose(mh["odds_ratio"], 6 * 17 / (14 * 3))
    assert mh["informative_strata"] == 2
    lo, hi = mh["or_ci"]
    assert lo < mh["odds_ratio"] < hi


def test_permutation_centre_and_direction():
    rng = np.random.default_rng(0)
    tasks = np.array(["x"] * 20 + ["y"] * 20)
    large = np.array([True, False] * 20)
    ack = np.zeros(40, dtype=bool)
    ack[[0, 2, 4, 6]] = True  # all four acknowledging cases are 7B, in task x
    res = ga.within_task_permutation(tasks, large, ack, rng, n_perm=2000)
    assert res["expected"] == 2.0 and res["observed"] == 4
    assert res["p_one_sided"] < 0.1
    # flip the labels: the same cases now sit in the 1.5B group
    res2 = ga.within_task_permutation(tasks, ~large, ack, np.random.default_rng(0), n_perm=2000)
    assert res2["p_one_sided"] > 0.9


def test_min_detectable_difference_has_the_stated_power():
    mdd = ga.min_detectable_diff(0.10, 224, 258)
    assert math.isclose(ga.power_two_prop(0.10, 0.10 + mdd, 224, 258), 0.8, abs_tol=1e-6)
    assert ga.min_detectable_diff(0.10, 2240, 2580) < mdd  # more data, smaller detectable gap


def test_eta_fnorm_by_hand():
    # 10 items. Hint: 4 eligible, 2 switch (S_h = .5). Placebo: 5 eligible, 1 switch (S_plc = .2).
    sw = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], float)
    el = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], float)
    swp = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0], float)
    elp = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 0], float)
    in_I = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 1], float)
    faithful = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], float)
    eta, F, fnorm = ga.eta_fnorm(sw, el, swp, elp, faithful, in_I)
    assert math.isclose(eta, 0.4) and math.isclose(F, 1 / 3) and math.isclose(fnorm, (1 / 3) / 0.6)


def _probe_row(ex, target, hinted, baseline, pa0, pc0):
    def pt(p):
        return {"p": {"A": p, "B": 1 - p}}
    return {"example_id": ex, "target": target, "hinted_answer": hinted, "baseline_answer": baseline,
            "A_hint_prompt_hint_cot": [pt(pa0), pt(0.5), pt(0.9)],
            "C_base_prompt_base_cot": [pt(pc0), pt(0.5), pt(0.1)]}


def test_probe_sets_and_readout():
    rows = [
        _probe_row("i1", "A", "A", "B", 0.8, 0.3),   # influenced
        _probe_row("i2", "A", "A", None, 0.7, 0.4),  # hinted = target, baseline unparsed: not in I_h
        _probe_row("i3", "A", "A", "A", 0.9, 0.9),   # baseline already at target
        _probe_row("r1", "A", "B", "B", 0.6, 0.2),   # reference, target on top before the CoT
        _probe_row("r2", "A", None, "B", 0.3, 0.2),  # reference, hinted answer unparsed
    ]
    s = ga.probe_sets(rows)
    assert [r["example_id"] for r in s["influenced"]] == ["i1"]
    assert [r["example_id"] for r in s["naive_influenced"]] == ["i1", "i2"]
    assert len(s["baseline_at_target"]) == 1 and len(s["reference"]) == 2
    out = ga.probe_readout(s["reference"], np.random.default_rng(0))
    assert out["f0"]["argmax_A"] == 1 and out["f0"]["argmax_C"] == 0
    assert math.isclose(out["f0"]["diff"], ((0.6 - 0.2) + (0.3 - 0.2)) / 2)
    assert out["f1"]["argmax_A"] == 2


def test_formatting_matches_make_paper_assets():
    import make_paper_assets as mpa
    for x in (0.1234, -0.00001, 0.5, None):
        assert ga.pct(x) == mpa.pct(x) and ga.pct(x, 0) == mpa.pct(x, 0)
    assert ga.ci_pct(0.1, 0.2) == mpa.ci_pct(0.1, 0.2)
