#!/usr/bin/env python3
"""Does the pre-reasoning readout predict which hinted items switch? (suggested by Hyojung Han)

The paper reports that at 7B the hint target is the top option before any reasoning in 92% of
influenced items. That share is conditioned on switching. Here the items that switch and the
items that resist get the same readout, and we ask how well it separates them (AUROC). This
measures predictive value only. It is not a test of when the model commits.

No new generations. Same loaders and set definitions as gap_analyses.py:
  switched  influenced set I_h: hinted answer = target, baseline parsed and off the target
            (every item; n = 224 / 206 / 258)
  resisted  matched resisted sample: hinted answer parsed and not the target, same base
            (a random sample; n = 77 / 83 / 90). AUROC does not depend on group sizes, so a
            random sample of the resisted items gives an unbiased estimate.

Scores (probability of the target letter in the forced answer, from probe_mediation.py):
  A0        hinted prompt, no CoT (f = 0): the readout behind the 92%
  C0        baseline prompt, no CoT: the item's own lean toward the target, no hint
  A0-C0     how far the hint moves the immediate answer
  D0        placebo prompt, no CoT
  topA0     1 if the target is the top option under A at f = 0 (the 92% / 59% readout as a rule)
  A_f       hinted prompt + hinted CoT cut at f = .25, .5, .75, 1
  B_f       baseline prompt + hinted CoT cut at f (the CoT text without the hint sentence)

AUROC = P(score of a random switched item > score of a random resisted item), ties count one
half. "Pooled" compares every switched item with every resisted item. "Within task" compares only
pairs from the same BBH task, so the option count (2 to 4 letters, chance 1/K) cannot drive it.
Intervals: percentile bootstrap, items resampled within each group, B = 10,000, fixed seed. The
A0-vs-C0 and A0-vs-D0 differences reuse the same resamples (paired).

    python scripts/auroc_switch.py      # writes outputs/analysis/auroc_switch.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gap_analyses import ARM_A, ARM_C, SIZES, is_matched, load_inputs, probe_sets, top  # noqa: E402

ARM_B, ARM_D = "B_base_prompt_hint_cot", "D_plc_prompt_hint_cot"
FRACS = [0.0, 0.25, 0.5, 0.75, 1.0]
B = 10_000
SEED = 20261114
OUT = Path(__file__).resolve().parent.parent / "outputs" / "analysis" / "auroc_switch.json"

# counts the paper already reports (gap_analyses macros); the script stops if they differ
EXPECT = {"1.5B": (224, 77, 191, 47), "3B": (206, 83, 154, 31), "7B": (258, 90, 238, 53)}


def p_target(r: dict, arm: str, i: int) -> float:
    cut = r[arm][i]
    if cut["frac"] != FRACS[i]:
        raise SystemExit(f"{r['example_id']}: {arm} cut {i} is f = {cut['frac']}")
    return float(cut["p"][r["target"]])


def score_table(rows: list[dict]) -> dict[str, np.ndarray]:
    s = {
        "A0": [p_target(r, ARM_A, 0) for r in rows],
        "C0": [p_target(r, ARM_C, 0) for r in rows],
        "A0-C0": [p_target(r, ARM_A, 0) - p_target(r, ARM_C, 0) for r in rows],
        "D0": [p_target(r, ARM_D, 0) for r in rows],
        "topA0": [float(top(r[ARM_A][0]) == r["target"]) for r in rows],
    }
    for i in range(1, len(FRACS)):
        s[f"A_{FRACS[i]:g}"] = [p_target(r, ARM_A, i) for r in rows]
        s[f"B_{FRACS[i]:g}"] = [p_target(r, ARM_B, i) for r in rows]
    return {k: np.array(v, dtype=float) for k, v in s.items()}


def ci(x: np.ndarray) -> list[float]:
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def analyze(pos_rows: list[dict], neg_rows: list[dict], rng: np.random.Generator) -> dict:
    sp, sn = score_table(pos_rows), score_table(neg_rows)
    n1, n0 = len(pos_rows), len(neg_rows)
    same = (np.array([r["task"] for r in pos_rows])[:, None]
            == np.array([r["task"] for r in neg_rows])[None, :]).astype(float)
    # one set of resamples for every score, so differences between scores are paired
    c1 = rng.multinomial(n1, np.full(n1, 1 / n1), size=B).astype(float)
    c0 = rng.multinomial(n0, np.full(n0, 1 / n0), size=B).astype(float)
    den_within = ((c1 @ same) * c0).sum(axis=1)

    out, boots = {}, {}
    for key in sp:
        d = sp[key][:, None] - sn[key][None, :]
        m = (d > 0) + 0.5 * (d == 0)
        pooled = float(m.mean())
        within = float((m * same).sum() / same.sum())
        b_pooled = ((c1 @ m) * c0).sum(axis=1) / (n1 * n0)
        b_within = ((c1 @ (m * same)) * c0).sum(axis=1) / den_within
        boots[key] = (b_pooled, b_within)
        out[key] = {"pooled": pooled, "pooled_ci": ci(b_pooled),
                    "within_task": within, "within_task_ci": ci(b_within),
                    "mean_switched": float(sp[key].mean()), "mean_resisted": float(sn[key].mean())}
    for a, b in (("A0", "C0"), ("A0", "D0")):
        out[f"diff_{a}_minus_{b}"] = {
            "pooled": out[a]["pooled"] - out[b]["pooled"],
            "pooled_ci": ci(boots[a][0] - boots[b][0]),
            "within_task": out[a]["within_task"] - out[b]["within_task"],
            "within_task_ci": ci(boots[a][1] - boots[b][1]),
        }
    # per task, point estimates only (groups are small)
    per_task = {}
    for t in sorted({r["task"] for r in pos_rows + neg_rows}):
        ip = [i for i, r in enumerate(pos_rows) if r["task"] == t]
        ineg = [j for j, r in enumerate(neg_rows) if r["task"] == t]
        row = {"n_switched": len(ip), "n_resisted": len(ineg)}
        if ip and ineg:
            for key in ("A0", "C0"):
                d = sp[key][ip][:, None] - sn[key][ineg][None, :]
                row[key] = float(((d > 0) + 0.5 * (d == 0)).mean())
        per_task[t] = row
    out["per_task"] = per_task
    return out


def main() -> None:
    _, _, probes, _ = load_inputs()
    rng = np.random.default_rng(SEED)
    res = {"seed": SEED, "B": B, "sizes": {}}
    for size in SIZES:
        ps = probe_sets(probes[size])
        pos = ps["influenced"]
        neg = [x for x in ps["reference"] if is_matched(x)]
        top_pos = sum(top(r[ARM_A][0]) == r["target"] for r in pos)
        top_neg = sum(top(r[ARM_A][0]) == r["target"] for r in neg)
        got = (len(pos), len(neg), top_pos, top_neg)
        if got != EXPECT[size]:
            raise SystemExit(f"{size}: counts {got} differ from the paper's {EXPECT[size]}")
        # arm B at f = 0 is the baseline prompt with no CoT, the same input as arm C at f = 0
        gap = max(abs(p_target(r, ARM_B, 0) - p_target(r, ARM_C, 0)) for r in pos + neg)
        r = analyze(pos, neg, rng)
        r.update(n_switched=len(pos), n_resisted=len(neg), topA0_switched=top_pos, topA0_resisted=top_neg,
                 max_abs_B0_minus_C0=gap)
        res["sizes"][size] = r
    OUT.write_text(json.dumps(res, indent=1))
    for size, r in res["sizes"].items():
        print(f"\n{size}: switched {r['n_switched']}, resisted {r['n_resisted']} "
              f"(target on top at f=0: {r['topA0_switched']} vs {r['topA0_resisted']}); "
              f"max |B0-C0| = {r['max_abs_B0_minus_C0']:.2g}")
        for key in ["topA0", "A0", "C0", "A0-C0", "D0"] + [f"A_{f:g}" for f in FRACS[1:]] + [f"B_{f:g}" for f in FRACS[1:]]:
            x = r[key]
            print(f"  {key:7s} pooled {x['pooled']:.3f} [{x['pooled_ci'][0]:.3f}, {x['pooled_ci'][1]:.3f}]   "
                  f"within-task {x['within_task']:.3f} [{x['within_task_ci'][0]:.3f}, {x['within_task_ci'][1]:.3f}]   "
                  f"mean P(target) {x['mean_switched']:.3f} vs {x['mean_resisted']:.3f}")
        for key in ("diff_A0_minus_C0", "diff_A0_minus_D0"):
            x = r[key]
            print(f"  {key}: pooled {x['pooled']:+.3f} [{x['pooled_ci'][0]:+.3f}, {x['pooled_ci'][1]:+.3f}]   "
                  f"within-task {x['within_task']:+.3f} [{x['within_task_ci'][0]:+.3f}, {x['within_task_ci'][1]:+.3f}]")
        print("  per task:", json.dumps(r["per_task"]))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
