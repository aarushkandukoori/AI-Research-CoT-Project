#!/usr/bin/env python3
"""Does the hinted readout add predictive value beyond the unhinted one? (suggested by Hyojung Han)

Exploratory, added after the main analysis. auroc_switch.py compares the AUROC of P_A(0) with the
AUROC of P_C(0). That compares two standalone scores. It does not test whether the hinted readout
carries information about switching beyond the unhinted readout. Hyojung Han suggested this check:

  base   logistic model of switching on P_C(0), the item's own lean toward the target (no hint)
  shift  logistic model on P_C(0) and the shift P_A(0) - P_C(0), how far the hint moves the
         immediate answer (the same model as one on P_C(0) and P_A(0))

Both models are scored on the same held-out folds: stratified 5-fold cross-validation, repeated
20 times with fixed seeds. AUROC = chance that a held-out switched item scores above a held-out
resisted item (ties count one half), counting only pairs held out in the same fold, so both items in
a pair are scored by the same fitted model. (Pooling held-out scores across folds would compare
items scored by different fits; most P_C(0) values sit near zero, so small differences between the
folds' intercepts would then decide the order and pull the base model well below the AUROC of
P_C(0) itself.) The 20 repeats are averaged. "Within task" also requires both items of a pair to
come from the same BBH task, as in auroc_switch.py. Gain = AUROC(shift model) - AUROC(base model).

Paired uncertainty: bootstrap over items, resampled within each group (B = 2,000). Every draw
refits both models on the same folds (5 repeats per draw) and records both AUROCs, so the gain's
interval is paired. Folds are dealt over the original items, so the copies of an item drawn twice
never sit in both the training and the test part.
Check: the faster bootstrap that keeps the held-out predictions fixed (B = 10,000, as in
auroc_switch.py). It ignores refitting noise, so its interval should be a bit narrower.
Sensitivity: the same comparison with log-odds in place of probabilities.

Items: the same as auroc_switch.py. Switched = the influenced set I_h (224 / 206 / 258 at
1.5B / 3B / 7B); resisted = the matched random sample (77 / 83 / 90). AUROC does not depend on
group sizes. Sampling one outcome group mainly moves a logistic model's intercept, which does not
change how it ranks items within a fold.

No new generations. Readouts come from outputs/probes (probe_mediation.py), loaded with the same
helpers as gap_analyses.py.

    python scripts/auroc_conditional.py      # writes outputs/analysis/auroc_conditional.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gap_analyses import ARM_A, ARM_C, SIZES, is_matched, load_inputs, probe_sets  # noqa: E402

K = 5               # folds
REPEATS = 20        # cross-validation repeats for the point estimate
BOOT_REPEATS = 5    # cross-validation repeats inside each bootstrap draw
B_REFIT = 2_000     # bootstrap draws that refit both models
B_FIXED = 10_000    # bootstrap draws that keep the held-out predictions fixed (check only)
SEED = 20261115
EPS = 1e-6          # probabilities are clipped to [EPS, 1 - EPS] before taking log-odds
OUT = Path(__file__).resolve().parent.parent / "outputs" / "analysis" / "auroc_conditional.json"

# set sizes the paper reports (auroc_switch.py checks the same); the script stops if they differ
EXPECT = {"1.5B": (224, 77), "3B": (206, 83), "7B": (258, 90)}


def readouts(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """P(target) with no CoT shown (f = 0): baseline prompt (arm C) and hinted prompt (arm A)."""
    for r in rows:
        if r[ARM_A][0]["frac"] != 0 or r[ARM_C][0]["frac"] != 0:
            raise SystemExit(f"{r['example_id']}: first cut is not f = 0")
    pc = np.array([r[ARM_C][0]["p"][r["target"]] for r in rows], dtype=float)
    pa = np.array([r[ARM_A][0]["p"][r["target"]] for r in rows], dtype=float)
    return pc, pa


def log_odds(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def design(pc: np.ndarray, pa: np.ndarray, use_log_odds: bool) -> dict[str, np.ndarray]:
    """Feature matrices (with an intercept column) for the two models."""
    if use_log_odds:
        pc, pa = log_odds(pc), log_odds(pa)
    one = np.ones_like(pc)
    return {"base": np.column_stack([one, pc]), "shift": np.column_stack([one, pc, pa - pc])}


def fit_logistic(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Unpenalized logistic regression by Newton's method. w = item weights (bootstrap counts)."""
    beta = np.zeros(X.shape[1])
    for _ in range(100):
        mu = 1.0 / (1.0 + np.exp(-(X @ beta)))
        grad = X.T @ (w * (y - mu))
        hess = X.T @ (X * (w * mu * (1.0 - mu))[:, None])
        step = np.linalg.solve(hess, grad)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-9:
            return beta
    raise RuntimeError("logistic fit did not converge")


def stratified_folds(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Deal each group's items, in random order, into K folds (fold sizes differ by at most one)."""
    fold = np.empty(len(y), dtype=int)
    for group in (0, 1):
        idx = rng.permutation(np.flatnonzero(y == group))
        fold[idx] = np.arange(len(idx)) % K
    return fold


def held_out(X: np.ndarray, y: np.ndarray, w: np.ndarray, fold: np.ndarray) -> np.ndarray:
    """Each item's score (log-odds of switching) from the model fit on the other K - 1 folds."""
    s = np.empty(len(y))
    for k in range(K):
        test = fold == k
        s[test] = X[test] @ fit_logistic(X[~test], y[~test], w[~test])
    return s


def wins(s: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pair matrix, switched items x resisted items: 1 if the switched item scores higher, 1/2 if tied."""
    d = s[y == 1][:, None] - s[y == 0][None, :]
    return (d > 0) + 0.5 * (d == 0)


def auroc(m: np.ndarray, w: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    """AUROC from a pair matrix. Pairs are weighted by item weights; mask keeps same-task pairs (or all)."""
    pw = np.outer(w[y == 1], w[y == 0]) * mask
    return float((m * pw).sum() / pw.sum())


def ci(x: np.ndarray) -> list[float]:
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def analyze(pos: list[dict], neg: list[dict], use_log_odds: bool, rng: np.random.Generator) -> dict:
    pc1, pa1 = readouts(pos)
    pc0, pa0 = readouts(neg)
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    X = design(np.concatenate([pc1, pc0]), np.concatenate([pa1, pa0]), use_log_odds)
    tasks = np.array([r["task"] for r in pos + neg])
    masks = {"pooled": np.ones((len(pos), len(neg))),
             "within_task": (tasks[y == 1][:, None] == tasks[y == 0][None, :]).astype(float)}
    ones = np.ones(len(y))

    def same_fold(fold: np.ndarray) -> np.ndarray:
        return (fold[y == 1][:, None] == fold[y == 0][None, :]).astype(float)

    # 1. Point estimate: REPEATS x stratified K-fold, the same folds for both models.
    per_repeat = {name: {kind: [] for kind in masks} for name in X}
    num = {name: np.zeros((len(pos), len(neg))) for name in X}  # wins summed over repeats (same-fold pairs)
    den = np.zeros((len(pos), len(neg)))                         # times each pair was held out together
    for _ in range(REPEATS):
        fold = stratified_folds(y, rng)
        sf = same_fold(fold)
        den += sf
        for name, Xm in X.items():
            m = wins(held_out(Xm, y, ones, fold), y)
            num[name] += m * sf
            for kind, mask in masks.items():
                per_repeat[name][kind].append(auroc(m, ones, y, mask * sf))

    # 2. Paired bootstrap over items that refits both models on the same folds in every draw.
    n1, n0 = len(pos), len(neg)
    draws = {name: {kind: np.empty(B_REFIT) for kind in masks} for name in X}
    for b in range(B_REFIT):
        w = np.concatenate([rng.multinomial(n1, np.full(n1, 1 / n1)),
                            rng.multinomial(n0, np.full(n0, 1 / n0))]).astype(float)
        total = {name: {kind: 0.0 for kind in masks} for name in X}
        for _ in range(BOOT_REPEATS):
            fold = stratified_folds(y, rng)
            sf = same_fold(fold)
            for name, Xm in X.items():
                m = wins(held_out(Xm, y, w, fold), y)
                for kind, mask in masks.items():
                    total[name][kind] += auroc(m, w, y, mask * sf) / BOOT_REPEATS
        for name in X:
            for kind in masks:
                draws[name][kind][b] = total[name][kind]

    # 3. Check: bootstrap with the held-out predictions fixed (wins from step 1, same-fold pairs only).
    c1 = rng.multinomial(n1, np.full(n1, 1 / n1), size=B_FIXED).astype(float)
    c0 = rng.multinomial(n0, np.full(n0, 1 / n0), size=B_FIXED).astype(float)
    fixed = {}
    for kind, mask in masks.items():
        pairs = ((c1 @ (den * mask)) * c0).sum(axis=1)
        fixed[kind] = {name: ((c1 @ (num[name] * mask)) * c0).sum(axis=1) / pairs for name in X}

    out = {}
    for name in X:
        out[name] = {}
        for kind in masks:
            vals = np.array(per_repeat[name][kind])
            out[name][kind] = float(vals.mean())
            out[name][f"{kind}_ci"] = ci(draws[name][kind])
            out[name][f"{kind}_repeat_range"] = [float(vals.min()), float(vals.max())]
    out["gain"] = {}
    for kind in masks:
        gain_repeats = np.array(per_repeat["shift"][kind]) - np.array(per_repeat["base"][kind])
        out["gain"][kind] = float(gain_repeats.mean())
        out["gain"][f"{kind}_ci"] = ci(draws["shift"][kind] - draws["base"][kind])
        out["gain"][f"{kind}_ci_fixed_predictions"] = ci(fixed[kind]["shift"] - fixed[kind]["base"])
        out["gain"][f"{kind}_repeat_range"] = [float(gain_repeats.min()), float(gain_repeats.max())]
    # coefficients on all items, for the sign of the shift term (not used for any AUROC)
    out["coef_all_items"] = {name: fit_logistic(Xm, y, ones).tolist() for name, Xm in X.items()}
    return out


def main() -> None:
    _, _, probes, _ = load_inputs()
    res = {"seed": SEED, "folds": K, "repeats": REPEATS, "boot_refit": B_REFIT,
           "boot_repeats_per_draw": BOOT_REPEATS, "boot_fixed_predictions": B_FIXED,
           "models": {"base": "logistic on [1, P_C(0)]", "shift": "logistic on [1, P_C(0), P_A(0) - P_C(0)]"},
           "scales": {"probability": "features are probabilities (primary)",
                      "log_odds": f"features are log-odds, probabilities clipped to [{EPS}, 1 - {EPS}] (sensitivity)"},
           "sizes": {}}
    for i, size in enumerate(SIZES):
        ps = probe_sets(probes[size])
        pos = ps["influenced"]
        neg = [x for x in ps["reference"] if is_matched(x)]
        if (len(pos), len(neg)) != EXPECT[size]:
            raise SystemExit(f"{size}: counts {(len(pos), len(neg))} differ from the paper's {EXPECT[size]}")
        r = {"n_switched": len(pos), "n_resisted": len(neg)}
        for j, scale in enumerate(("probability", "log_odds")):
            rng = np.random.default_rng([SEED, i, j])  # own stream per size and scale
            r[scale] = analyze(pos, neg, scale == "log_odds", rng)
        res["sizes"][size] = r
        for scale in ("probability", "log_odds"):
            x = r[scale]
            print(f"{size} {scale:11s} base {x['base']['pooled']:.3f} {fmt(x['base']['pooled_ci'])}  "
                  f"shift {x['shift']['pooled']:.3f} {fmt(x['shift']['pooled_ci'])}  "
                  f"gain {x['gain']['pooled']:+.3f} {fmt(x['gain']['pooled_ci'])} "
                  f"(fixed {fmt(x['gain']['pooled_ci_fixed_predictions'])})  "
                  f"within-task gain {x['gain']['within_task']:+.3f} {fmt(x['gain']['within_task_ci'])}", flush=True)
    OUT.write_text(json.dumps(res, indent=1))
    print(f"wrote {OUT}")


def fmt(v: list[float]) -> str:
    return f"[{v[0]:+.3f}, {v[1]:+.3f}]"


if __name__ == "__main__":
    main()
