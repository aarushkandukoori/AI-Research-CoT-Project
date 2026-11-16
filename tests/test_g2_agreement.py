"""Tests for scripts/g2_agreement.py (gate G2), on synthetic data only.

tests/fixtures/fake_g2_labels.csv holds made-up labels for made-up case ids
("fake_task_NN"), written in the same CSV format the labeling page downloads. It is not
a human label of any real case. The fake judge labels are built in fake_judge() below,
and every expected number is worked out by hand in the comments.
"""

import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import g2_agreement as g2  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "fake_g2_labels.csv"
SIZES = ["1.5B", "3B", "7B"]


def fake_judge(judge_dir: Path) -> None:
    """Fake final judge labels, per model size:
    sycophancy: 40 judged cases = 8 flagged (5 FAITHFUL + 3 AMBIGUOUS, all mention the hint)
                + 32 not flagged (UNFAITHFUL, no mention). Cases 01-10 are in the sheet:
                01-03 FAITHFUL, 04-05 AMBIGUOUS, 06-10 UNFAITHFUL.
    reorder:    20 judged cases, all UNFAITHFUL with no mention. Cases 01-10 are in the sheet.
    """
    judge_dir.mkdir()
    for size in SIZES:
        labels = {}
        for k in range(1, 41):
            if k <= 10:
                cls = "FAITHFUL" if k <= 3 else "AMBIGUOUS" if k <= 5 else "UNFAITHFUL"
            else:  # judged but not drawn for labeling
                cls = "FAITHFUL" if k <= 12 else "AMBIGUOUS" if k == 13 else "UNFAITHFUL"
            labels[f"{size}|sycophancy|fake_task_{k:02d}"] = {"classification": cls, "mentions_hint": cls != "UNFAITHFUL"}
        for k in range(1, 21):
            labels[f"{size}|reorder|fake_task_{k:02d}"] = {"classification": "UNFAITHFUL", "mentions_hint": False}
        (judge_dir / f"{size}.json").write_text(json.dumps(labels))


def run(tmp_path: Path, labels: Path = FIXTURE, extra: tuple = ()) -> dict:
    """Run the script on fake data and return the JSON it wrote."""
    judge_dir = tmp_path / "judge"
    if not judge_dir.exists():
        fake_judge(judge_dir)
    todo = tmp_path / "todo.json"  # the cases "drawn" for labeling: the 60 in the sheet
    todo.write_text(json.dumps([{"case_id": f"{s}|{h}|fake_task_{k:02d}"}
                                for s in SIZES for h in ("sycophancy", "reorder") for k in range(1, 11)]))
    out = tmp_path / "g2.json"
    g2.main([str(labels), "--judge-dir", str(judge_dir), "--todo", str(todo), "--out", str(out),
             "--boot", "200", *extra])
    return json.loads(out.read_text())


def test_kappa_small_examples():
    ones = [1.0] * 4
    # 3 of 4 agree. Human says F half the time, judge a quarter of the time:
    # p_e = .5 x .25 + .5 x .75 = .5, so kappa = (.75 - .5) / (1 - .5) = .5
    agree, kappa = g2.agreement_and_kappa([("F", "F"), ("F", "U"), ("U", "U"), ("U", "U")], ones)
    assert agree == 0.75 and kappa == pytest.approx(0.5)
    # both raters say U every time: 100% agreement, and kappa is undefined (not 1.0)
    assert g2.agreement_and_kappa([("U", "U")] * 4, ones) == (1.0, None)
    # the judge says U every time but the human does not: kappa is 0
    assert g2.agreement_and_kappa([("F", "U"), ("U", "U"), ("U", "U"), ("U", "U")], ones)[1] == pytest.approx(0.0)
    # weights: the disagreeing case counted 3 times, so 3 of 6 agree
    assert g2.agreement_and_kappa([("F", "F"), ("F", "U"), ("U", "U"), ("U", "U")], [1, 3, 1, 1])[0] == pytest.approx(0.5)


def test_fake_sheet_end_to_end(tmp_path):
    r = run(tmp_path)
    assert r["complete"] is True and r["n_labeled"] == 60

    syc = r["by_condition"]["sycophancy"]
    assert syc["n"] == 30
    # Per size, (human, judge) = (F,F) x3, (A,A) x2, (U,U) x4, (F,U) x1, so 9 of 10 agree.
    # Human shares F .4, A .2, U .4; judge F .3, A .2, U .5: p_e = .12 + .04 + .20 = .36,
    # kappa = (.9 - .36) / (1 - .36) = .84375. Every size is the same, so the pooled value is too.
    assert syc["three_way"]["agreement"] == pytest.approx(0.9)
    assert syc["three_way"]["kappa"] == pytest.approx(0.84375, abs=1e-6)
    assert syc["ambiguous_as_no"]["kappa"] == pytest.approx(18 / 23, abs=1e-6)   # (.9 - .54) / .46
    assert syc["ambiguous_as_yes"]["kappa"] == pytest.approx(0.8, abs=1e-6)      # (.9 - .5) / .5
    assert syc["mention"]["agreement"] == pytest.approx(0.9)
    assert syc["mention"]["kappa"] == pytest.approx(0.8, abs=1e-6)
    assert syc["three_way"]["confusion"]["FAITHFUL"]["UNFAITHFUL"] == 3          # one per size
    lo, hi = syc["three_way"]["kappa_ci95"]
    assert lo <= 0.84375 <= hi
    assert r["gate"]["sycophancy"]["status"] == "met"

    # Reweighting: flagged cases count 8/5 = 1.6 each, the others 32/5 = 6.4 each.
    w = r["weights"]["7B|sycophancy|not_flagged"]
    assert (w["judged_cases"], w["labeled"], w["weight"]) == (32, 5, 6.4)
    # agreement = (5 x 1.6 + 4 x 6.4) / (5 x 1.6 + 5 x 6.4) = 33.6 / 40 = .84
    assert syc["three_way"]["agreement_reweighted"] == pytest.approx(0.84)
    # human FAITHFUL rate = (3 x 1.6 + 1 x 6.4) / 40 = .28; the judge's over all 120 judged = 15/120
    assert syc["ambiguous_as_no"]["human_rate_reweighted"] == pytest.approx(0.28)
    assert syc["ambiguous_as_no"]["judge_rate_all_judged"] == pytest.approx(0.125)
    # Of the 5 not-flagged cases per size (15 in all), the human called case 10 FAITHFUL each time.
    assert syc["judge_not_flagged"] == {"labeled": 15, "human_faithful": 3, "human_ambiguous": 0,
                                        "human_mentions_hint": 3}

    # Reorder: the judge said UNFAITHFUL every time, so kappa is 0 (or undefined) whatever the human says.
    order = r["by_condition"]["reorder"]
    assert order["three_way"]["agreement"] == pytest.approx(29 / 30)
    assert order["three_way"]["kappa"] == pytest.approx(0.0)
    assert r["gate"]["reorder"]["status"] == "not met"
    assert "same label" in r["gate"]["reorder"]["reason"]
    assert r["by_cell"]["3B|reorder"]["three_way"]["kappa"] is None   # both raters all UNFAITHFUL

    # All 60 cases, each counted for its stratum: 3 x 33.6 + (18 + 20 + 20) = 158.8 of 180.
    assert r["overall"]["all"]["three_way"]["agreement_reweighted"] == pytest.approx(158.8 / 180)


def test_unlabeled_cases_stop_the_run(tmp_path, capsys):
    rows = list(csv.DictReader(open(FIXTURE, newline="")))
    rows[0]["label"] = ""           # one case without a label
    rows[1]["mentions_hint"] = ""   # one without the mention answer
    partial = tmp_path / "fake_partial.csv"
    with open(partial, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(SystemExit) as stop:
        run(tmp_path, partial)
    assert "2 case(s) still need" in str(stop.value)
    assert "allow-partial" not in str(stop.value)   # no nudge toward peeking at numbers
    assert not (tmp_path / "g2.json").exists()

    capsys.readouterr()
    r = run(tmp_path, partial, ("--allow-partial",))
    assert r["complete"] is False and r["n_labeled"] == 58
    printed = capsys.readouterr().out
    assert "kappa" not in printed and "agreement" not in printed   # the labeler stays blind
    assert g2.paper_macros(r) == []   # a partial file never reaches the paper


def test_bad_label_is_rejected(tmp_path):
    bad = tmp_path / "fake_bad.csv"
    bad.write_text(FIXTURE.read_text().replace('"FAITHFUL","yes"', '"FAITHFULL","yes"', 1))
    with pytest.raises(SystemExit) as stop:
        run(tmp_path, bad)
    assert "FAITHFULL" in str(stop.value)


def test_paper_macros_and_table(tmp_path):
    r = run(tmp_path)
    macros = dict(g2.paper_macros(r))
    assert macros["GTwoN"] == "60"
    assert macros["GTwoKappaSyc"] == "0.84"
    assert macros["GTwoGateSyc"] == "met"
    assert macros["GTwoAgreeSyc"] == "90.0"
    assert macros["GTwoKappaOrd"] == "0.00"
    assert macros["GTwoGateOrd"] == "not met"
    assert macros["GTwoKappaOrdThree"] == "--"   # undefined kappa prints as a dash
    assert macros["GTwoMissedSyc"] == "3"
    assert macros["GTwoHumanFSyc"] == "28.0"
    assert (macros["GTwoNotFlaggedNSyc"], macros["GTwoNotFlaggedFSyc"]) == ("15", "3")
    assert (macros["GTwoNotFlaggedNOrd"], macros["GTwoNotFlaggedAOrd"]) == ("30", "1")
    assert all(name.isalpha() for name in macros)  # LaTeX macro names cannot contain digits
    table = g2.paper_table(r)
    assert table.count("\\\\") == 1 + 8  # header + (3 sizes + pooled) x 2 conditions
