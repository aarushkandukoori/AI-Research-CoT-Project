#!/usr/bin/env python3
"""Extra analyses on the released runs, no new generations (Nov 2026 review checks).

1. Probe readout on the reference items: the 100 randomly drawn sycophancy items per size
   where the hinted answer is NOT the hint target. Is the target already the top option
   before any reasoning (f = 0) with the hint (arm A) and without it (arm C)? Same at f = 1,
   and the same numbers for the influenced items, so the two groups can be compared.
2. Exact sizes of the sets that the paper and the probe file count.
3. Paired item bootstrap intervals for the noise share eta and for F_norm (sycophancy).
4. H2 (acknowledgment 7B vs 1.5B) stratified by task: Mantel-Haenszel common odds ratio
   and test, plus a permutation test that shuffles model size within each task.
   A robustness check next to the preregistered pooled z-test, not a replacement.
5. Minimum detectable difference for H2 at the achieved denominators.
6. Reordering excess switch-to-target: bootstrap interval and p-value per size.

Item-level data are rebuilt from the greedy generation cache (read only) and checked
against outputs/analysis/summary.json before anything is computed, so the two cannot drift.

make_paper_assets.py --gap-analyses calls compute() and macros(). Run alone to print the JSON:

    python scripts/gap_analyses.py
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cot_faithfulness.analysis.bootstrap import wilson_ci  # noqa: E402

SIZES = ["1.5B", "3B", "7B"]
WORD = {"1.5B": "OneFive", "3B": "Three", "7B": "Seven"}
B = 10_000
SEED = 20261110  # fixed; every bootstrap/permutation below draws from its own generator with this seed
ARM_A, ARM_C = "A_hint_prompt_hint_cot", "C_base_prompt_base_cot"


# ---------------------------------------------------------------- formatting (same as make_paper_assets)
def pct(x: float | None, digits: int = 1) -> str:
    if x is None:
        return "--"
    s = f"{100 * x:.{digits}f}"
    return s[1:] if s.startswith("-") and float(s) == 0 else s


def ci_pct(lo: float, hi: float, digits: int = 1) -> str:
    return f"[{pct(lo, digits)}, {pct(hi, digits)}]"


def ptxt(p: float) -> str:
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


# ---------------------------------------------------------------- item-level rows
def item_rows(complete: dict, hint: str) -> list[dict]:
    """One row per complete item, with the same flags analyze_mlx.steering() and
    influenced_cases() use. in_steer: all three answers parsed (the steering table's items)."""
    from analyze_mlx import placebo_target

    rows = []
    for ex_id, conds in sorted(complete.items()):
        base, rec, plc = conds["baseline"], conds[hint], conds["placebo"]
        a0, ah, ap = base["answer_v2"], rec["answer_v2"], plc["answer_v2"]
        tau = rec["hint_suggested_letter"]
        identity = hint == "reorder" and tau == "A"
        in_steer = None not in (a0, ah, ap) and not identity
        tau_p = placebo_target(plc)
        row = {"example_id": ex_id, "task": rec["task"], "in_steer": in_steer,
               "elig": False, "switch": False, "elig_p": False, "switch_p": False,
               "in_I": a0 is not None and ah is not None and a0 != ah and ah == tau and not identity,
               "plc_unparsed": ap is None}
        if in_steer:
            row.update(elig=a0 != tau, switch=a0 != tau and ah == tau,
                       elig_p=a0 != tau_p, switch_p=a0 != tau_p and ap == tau_p)
        rows.append(row)
    return rows


def check_rows(rows: list[dict], st: dict, n_infl: int) -> None:
    """Stop if the rebuilt rows do not reproduce summary.json exactly."""
    steer = [r for r in rows if r["in_steer"]]
    elig = [r for r in steer if r["elig"]]
    elig_p = [r for r in steer if r["elig_p"]]
    got = (len(steer), len(elig), sum(r["switch"] for r in elig) / len(elig),
           sum(r["switch_p"] for r in elig_p) / len(elig_p), sum(r["in_I"] for r in rows))
    want = (st["n_items"], st["switch_to_target_n"], st["switch_to_target"], st["placebo_switch_to_dummy"], n_infl)
    if got[:2] != want[:2] or got[4] != want[4] or not np.isclose(got[2], want[2]) or not np.isclose(got[3], want[3]):
        raise SystemExit(f"rebuilt rows do not match summary.json: {got} vs {want}")


# ---------------------------------------------------------------- 1. probe readout
def top(pt: dict) -> str:
    return max(pt["p"], key=pt["p"].get)


def boot_mean(d: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    idx = rng.integers(0, len(d), size=(B, len(d)))
    b = d[idx].mean(axis=1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def probe_readout(rows: list[dict], rng: np.random.Generator) -> dict:
    """Argmax and mean P(target) at f = 0 and f = 1 under arms A and C."""
    out = {"n": len(rows)}
    for label, i in (("f0", 0), ("f1", -1)):
        pa = np.array([r[ARM_A][i]["p"][r["target"]] for r in rows])
        pc = np.array([r[ARM_C][i]["p"][r["target"]] for r in rows])
        lo, hi = boot_mean(pa - pc, rng)
        out[label] = {
            "argmax_A": int(sum(top(r[ARM_A][i]) == r["target"] for r in rows)),
            "argmax_C": int(sum(top(r[ARM_C][i]) == r["target"] for r in rows)),
            "mean_A": float(pa.mean()), "mean_C": float(pc.mean()),
            "diff": float((pa - pc).mean()), "diff_ci": [lo, hi],
        }
    return out


def probe_sets(probe_rows: list[dict]) -> dict:
    """Split one size's probe file the way probe_mediation.py drew it."""
    hinted_target = [r for r in probe_rows if r["hinted_answer"] == r["target"]]
    reference = [r for r in probe_rows if r["hinted_answer"] != r["target"]]
    infl = [r for r in hinted_target if r["baseline_answer"] is not None and r["baseline_answer"] != r["target"]]
    return {
        "rows": probe_rows, "hinted_target": hinted_target, "reference": reference, "influenced": infl,
        # plain reading of the file: baseline answer != target, an unparsed baseline counts as "!="
        "naive_influenced": [r for r in hinted_target if r["baseline_answer"] != r["target"]],
        "baseline_unparsed": [r for r in hinted_target if r["baseline_answer"] is None],
        "baseline_at_target": [r for r in hinted_target if r["baseline_answer"] == r["target"]],
    }


def reference_pool(complete: dict) -> int:
    """Items the 100 reference items were drawn from (probe_mediation.py: hinted answer,
    parsed on the displayed letters, is not the target; includes unparsed hinted answers)."""
    from cot_faithfulness.analysis.influence import parse_answer_v2

    n = 0
    for conds in complete.values():
        syc = conds["sycophancy"]
        n += parse_answer_v2(syc["cot"], sorted(syc["letter_map"])) != syc["hint_suggested_letter"]
    return n


# ---------------------------------------------------------------- 3. eta and F_norm
def eta_fnorm(sw, el, swp, elp, faithful, in_I):
    """Noise share eta = min(1, S_plc / S_h) and F_norm = min(1, F / (1 - eta)), from item flags.
    Works on 1-D arrays (one estimate) or 2-D arrays (one estimate per bootstrap row)."""
    s_h = sw.sum(-1) / np.maximum(el.sum(-1), 1)
    s_p = swp.sum(-1) / np.maximum(elp.sum(-1), 1)
    eta = np.minimum(1.0, np.divide(s_p, s_h, out=np.ones_like(s_p, dtype=float), where=s_h > 0))
    F = faithful.sum(-1) / np.maximum(in_I.sum(-1), 1)
    fnorm = np.where(eta < 1, np.minimum(1.0, F / np.where(eta < 1, 1 - eta, 1)), np.nan)
    return eta, F, fnorm


def boot_eta_fnorm(rows: list[dict], faithful_ids: set[str], rng: np.random.Generator) -> dict:
    """Paired bootstrap over complete items: steering flags and judge labels move together."""
    arr = {k: np.array([r[k] for r in rows], dtype=float) for k in ("switch", "elig", "switch_p", "elig_p", "in_I")}
    fa = np.array([r["in_I"] and r["example_id"] in faithful_ids for r in rows], dtype=float)
    eta0, F0, fn0 = eta_fnorm(arr["switch"], arr["elig"], arr["switch_p"], arr["elig_p"], fa, arr["in_I"])
    idx = rng.integers(0, len(rows), size=(B, len(rows)))
    eta, F, fn = eta_fnorm(arr["switch"][idx], arr["elig"][idx], arr["switch_p"][idx], arr["elig_p"][idx],
                           fa[idx], arr["in_I"][idx])
    ok = ~np.isnan(fn)
    return {"n_items": len(rows), "eta": float(eta0), "F": float(F0), "fnorm": float(fn0),
            "eta_ci": [float(np.percentile(eta, 2.5)), float(np.percentile(eta, 97.5))],
            "fnorm_ci": [float(np.percentile(fn[ok], 2.5)), float(np.percentile(fn[ok], 97.5))],
            "draws_eta_capped": int((eta >= 1).sum()), "draws_fnorm_capped": int((fn[ok] >= 1).sum()),
            "draws_fnorm_undefined": int((~ok).sum())}


# ---------------------------------------------------------------- 4. stratified H2
def mantel_haenszel(tables: list[tuple[int, int, int, int]]) -> dict:
    """tables: (a, b, c, d) per stratum = (7B ack, 7B not, 1.5B ack, 1.5B not).
    Common odds ratio (7B vs 1.5B) with the Robins-Breslow-Greenland 95% CI, and the
    Cochran-Mantel-Haenszel z (no continuity correction) with one-sided p for 7B > 1.5B."""
    R = S = PR = PS_QR = QS = 0.0
    obs = exp = var = 0.0
    informative = 0
    for a, b, c, d in tables:
        n = a + b + c + d
        if n < 2:
            continue
        m1, n1, n0 = a + c, a + b, c + d  # acknowledging total, 7B total, 1.5B total
        if m1 in (0, n):
            continue  # no variation in this stratum: adds nothing to the test or the odds ratio
        informative += 1
        r, s = a * d / n, b * c / n
        p, q = (a + d) / n, (b + c) / n
        R += r; S += s; PR += p * r; PS_QR += p * s + q * r; QS += q * s
        obs += a
        exp += n1 * m1 / n
        var += n1 * n0 * m1 * (n - m1) / (n * n * (n - 1))
    z = (obs - exp) / math.sqrt(var) if var else 0.0
    out = {"informative_strata": informative, "z": z, "p_one_sided": float(1 - norm.cdf(z)),
           "p_two_sided": float(2 * (1 - norm.cdf(abs(z)))), "observed": obs, "expected": exp}
    if R > 0 and S > 0:
        orr = R / S
        se = math.sqrt(PR / (2 * R * R) + PS_QR / (2 * R * S) + QS / (2 * S * S))
        out.update(odds_ratio=orr, or_ci=[orr * math.exp(-1.96 * se), orr * math.exp(1.96 * se)])
    else:
        out.update(odds_ratio=None, or_ci=None)
    return out


def within_task_permutation(tasks: np.ndarray, is_large: np.ndarray, ack: np.ndarray,
                            rng: np.random.Generator, n_perm: int = B) -> dict:
    """Shuffle the model-size label within each task. Statistic: acknowledging cases in the
    7B group (equivalent to the pooled 7B - 1.5B rate difference, since group sizes per task
    are fixed). One-sided p for 7B > 1.5B; two-sided p on |statistic - its permutation mean|."""
    obs = int(ack[is_large].sum())
    stats = np.zeros(n_perm)
    for t in np.unique(tasks):
        sel = np.where(tasks == t)[0]
        k_large = int(is_large[sel].sum())
        a = ack[sel]
        perm = np.argsort(rng.random((n_perm, len(sel))), axis=1)[:, :k_large]
        stats += a[perm].sum(axis=1)
    centre = sum(ack[tasks == t].sum() * is_large[tasks == t].mean() for t in np.unique(tasks))
    return {"observed": obs, "expected": float(centre), "n_perm": n_perm,
            "p_one_sided": float((1 + (stats >= obs).sum()) / (n_perm + 1)),
            "p_two_sided": float((1 + (np.abs(stats - centre) >= abs(obs - centre) - 1e-9).sum()) / (n_perm + 1))}


# ---------------------------------------------------------------- 5. minimum detectable difference
def power_two_prop(p1: float, p2: float, n1: int, n2: int, alpha: float = 0.05) -> float:
    """Power of the one-sided pooled two-proportion z-test (the preregistered H2 test) for p2 > p1."""
    pbar = (n1 * p1 + n2 * p2) / (n1 + n2)
    se0 = math.sqrt(pbar * (1 - pbar) * (1 / n1 + 1 / n2))
    se1 = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return float(norm.cdf(((p2 - p1) - norm.ppf(1 - alpha) * se0) / se1))


def min_detectable_diff(p1: float, n1: int, n2: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest increase p2 - p1 that the test detects with the given power (bisection)."""
    lo, hi = 0.0, 1.0 - p1
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if power_two_prop(p1, p1 + mid, n1, n2, alpha) < power else (lo, mid)
    return hi


# ---------------------------------------------------------------- 6. excess switch-to-target
def boot_excess(rows: list[dict], rng: np.random.Generator) -> dict:
    """Paired bootstrap over the steering items, same estimator as analyze_mlx.steering()."""
    steer = [r for r in rows if r["in_steer"]]
    e, s = np.array([r["elig"] for r in steer]), np.array([r["switch"] for r in steer])
    ep, sp = np.array([r["elig_p"] for r in steer]), np.array([r["switch_p"] for r in steer])
    point = s.sum() / e.sum() - sp.sum() / ep.sum()
    idx = rng.integers(0, len(steer), size=(B, len(steer)))
    d = s[idx].sum(1) / np.maximum(e[idx].sum(1), 1) - sp[idx].sum(1) / np.maximum(ep[idx].sum(1), 1)
    share_ge0, share_le0 = float((d >= 0).mean()), float((d <= 0).mean())
    return {"n": len(steer), "excess": float(point),
            "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "share_draws_ge0": share_ge0,
            # two-sided percentile-bootstrap p; floor at 1/B when no draw crosses zero
            "p_two_sided": max(2 * min(share_ge0, share_le0), 1 / B)}


# ---------------------------------------------------------------- driver
def compute(summary: dict, labels: dict, probes: dict, complete_by_size: dict) -> dict:
    res = {"seed": SEED, "B": B, "sizes": {}}
    rows_syc, faithful_ids = {}, {}
    for size in SIZES:
        if size not in summary["models"] or size not in complete_by_size:
            continue
        mm = summary["models"][size]
        complete = complete_by_size[size]
        rows = {h: item_rows(complete, h) for h in ("sycophancy", "reorder")}
        for h in rows:
            check_rows(rows[h], mm["steering_answer_v2"][h], mm["n_influenced"][h])
        rows_syc[size] = rows["sycophancy"]
        lab = labels.get(size, {})
        faithful_ids[size] = {k.split("|", 2)[2] for k, v in lab.items()
                              if k.startswith(f"{size}|sycophancy|") and v["classification"] == "FAITHFUL"}
        in_I = {r["example_id"] for r in rows["sycophancy"] if r["in_I"]}
        judged = {k.split("|", 2)[2] for k in lab if k.startswith(f"{size}|sycophancy|")}
        if judged != in_I:
            raise SystemExit(f"{size}: judged sycophancy cases differ from the rebuilt influenced set")
        r = res["sizes"][size] = {}

        # 2. set sizes
        steer = [x for x in rows["sycophancy"] if x["in_steer"]]
        r["sets"] = {
            "complete_items": len(complete),
            "influenced_I_h": len(in_I),
            "influenced_placebo_unparsed": sum(x["in_I"] and x["plc_unparsed"] for x in rows["sycophancy"]),
            "steering_items": len(steer),
            "switch_eligible": sum(x["elig"] for x in steer),
            "switch_count": sum(x["switch"] for x in steer),
        }
        if size in probes:
            ps = probe_sets(probes[size])
            if {x["example_id"] for x in ps["influenced"]} != in_I:
                raise SystemExit(f"{size}: probe influenced set differs from I_h")
            r["sets"].update({
                "probe_rows": len(ps["rows"]), "probe_hinted_target": len(ps["hinted_target"]),
                "probe_naive_influenced": len(ps["naive_influenced"]),
                "probe_baseline_unparsed": len(ps["baseline_unparsed"]),
                "probe_baseline_at_target": len(ps["baseline_at_target"]),
                "probe_reference": len(ps["reference"]),
            })
            # 1. probe readout
            ref = ps["reference"]
            pool = reference_pool(complete)
            rng = np.random.default_rng(SEED)
            r["probe_reference"] = probe_readout(ref, rng)
            r["probe_reference"].update(
                pool=pool,
                hinted_unparsed=sum(x["hinted_answer"] is None for x in ref),
                baseline_at_target=sum(x["baseline_answer"] == x["target"] for x in ref),
            )
            k = r["probe_reference"]["f0"]["argmax_A"]
            lo, hi = wilson_ci(k, len(ref))
            r["probe_reference"]["f0"]["argmax_A_share_wilson"] = [lo, hi]
            r["probe_reference"]["f0"]["argmax_A_pool_estimate"] = [k / len(ref) * pool, lo * pool, hi * pool]
            r["probe_influenced"] = probe_readout(ps["influenced"], rng)

        # 3. eta and F_norm
        r["eta_fnorm"] = boot_eta_fnorm(rows["sycophancy"], faithful_ids[size], np.random.default_rng(SEED))
        # 6. reordering excess (and sycophancy for reference)
        r["excess"] = {h: boot_excess(rows[h], np.random.default_rng(SEED)) for h in ("sycophancy", "reorder")}

    # 4. stratified H2 and 5. MDD (1.5B vs 7B, sycophancy)
    if "1.5B" in rows_syc and "7B" in rows_syc:
        recs = []
        for size in ("1.5B", "7B"):
            for x in rows_syc[size]:
                if x["in_I"]:
                    recs.append((x["task"], size == "7B", x["example_id"] in faithful_ids[size]))
        tasks = np.array([t for t, _, _ in recs]); large = np.array([l for _, l, _ in recs]); ack = np.array([a for _, _, a in recs])
        by_task, tables = {}, []
        for t in sorted(set(tasks)):
            s7, s1 = (tasks == t) & large, (tasks == t) & ~large
            a, c = int(ack[s7].sum()), int(ack[s1].sum())
            tables.append((a, int(s7.sum()) - a, c, int(s1.sum()) - c))
            by_task[t] = {"7B": [a, int(s7.sum())], "1.5B": [c, int(s1.sum())]}
        n1, n2 = int((~large).sum()), int(large.sum())
        k1 = int(ack[~large].sum())
        res["h2"] = {"by_task": by_task, "mantel_haenszel": mantel_haenszel(tables),
                     "permutation": within_task_permutation(tasks, large, ack, np.random.default_rng(SEED)),
                     "note": "Same BBH questions at both sizes; like the preregistered pooled test, cells are treated as independent."}
        mdd = min_detectable_diff(k1 / n1, n1, n2)
        res["mdd"] = {"p1": k1 / n1, "n1": n1, "n2": n2, "alpha_one_sided": 0.05, "power": 0.8,
                      "mdd": mdd, "p2_detectable": k1 / n1 + mdd,
                      "power_check": power_two_prop(k1 / n1, k1 / n1 + mdd, n1, n2)}
    return res


def macros(res: dict) -> list[tuple[str, str]]:
    out = []
    add = lambda n, v: out.append((n, v))
    for size, r in res["sizes"].items():
        w = WORD[size]
        st = r["sets"]
        add(f"SetComplete{w}", str(st["complete_items"]))
        add(f"SetSteerSyc{w}", str(st["steering_items"]))
        add(f"SetEligSyc{w}", str(st["switch_eligible"]))
        add(f"SetSwitchSyc{w}", str(st["switch_count"]))
        if "probe_rows" in st:
            add(f"SetProbeRows{w}", str(st["probe_rows"]))
            add(f"SetProbeHintedTarget{w}", str(st["probe_hinted_target"]))
            add(f"SetProbeNaive{w}", str(st["probe_naive_influenced"]))
            add(f"SetProbeBaseUnparsed{w}", str(st["probe_baseline_unparsed"]))
            add(f"SetProbeBaseAtTarget{w}", str(st["probe_baseline_at_target"]))
        if "probe_reference" in r:
            ref, inf = r["probe_reference"], r["probe_influenced"]
            add(f"RefN{w}", str(ref["n"]))
            add(f"RefPool{w}", str(ref["pool"]))
            add(f"RefHintedUnparsed{w}", str(ref["hinted_unparsed"]))
            add(f"RefBaseAtTarget{w}", str(ref["baseline_at_target"]))
            for fl, tag in (("f0", "Zero"), ("f1", "Final")):
                for src, pre in ((ref, "Ref"), (inf, "Infl")):
                    v = src[fl]
                    add(f"{pre}ArgmaxA{tag}{w}", str(v["argmax_A"]))
                    add(f"{pre}ArgmaxC{tag}{w}", str(v["argmax_C"]))
                    add(f"{pre}ArgmaxA{tag}Pct{w}", pct(v["argmax_A"] / src["n"], 0))
                    add(f"{pre}ArgmaxC{tag}Pct{w}", pct(v["argmax_C"] / src["n"], 0))
                    add(f"{pre}ProbeA{tag}{w}", pct(v["mean_A"]))
                    add(f"{pre}ProbeC{tag}{w}", pct(v["mean_C"]))
                    add(f"{pre}Diff{tag}{w}", pct(v["diff"]))
                    add(f"{pre}Diff{tag}CI{w}", ci_pct(*v["diff_ci"]))
            lo, hi = ref["f0"]["argmax_A_share_wilson"]
            add(f"RefArgmaxAZeroWilson{w}", ci_pct(lo, hi, 0))
            est, elo, ehi = ref["f0"]["argmax_A_pool_estimate"]
            add(f"RefLeadEst{w}", f"{est:.0f}")
            add(f"RefLeadEstCI{w}", f"[{elo:.0f}, {ehi:.0f}]")
        ef = r["eta_fnorm"]
        add(f"NoiseShareCISyc{w}", ci_pct(*ef["eta_ci"], 0))
        add(f"FnormCISyc{w}", ci_pct(*ef["fnorm_ci"]))
        for h, tag in (("sycophancy", "Syc"), ("reorder", "Ord")):
            ex = r["excess"][h]
            add(f"ExcessBootCI{tag}{w}", ci_pct(*ex["ci"]))
            add(f"ExcessBootP{tag}{w}", ptxt(ex["p_two_sided"]))
    if "h2" in res:
        mh, pm = res["h2"]["mantel_haenszel"], res["h2"]["permutation"]
        add("HTwoMHOR", "--" if mh["odds_ratio"] is None else f"{mh['odds_ratio']:.2f}")
        add("HTwoMHORCI", "--" if mh["or_ci"] is None else f"[{mh['or_ci'][0]:.2f}, {mh['or_ci'][1]:.2f}]")
        add("HTwoMHZ", f"{mh['z']:.2f}")
        add("HTwoMHP", f"{mh['p_one_sided']:.3f}")
        add("HTwoMHPTwo", f"{mh['p_two_sided']:.3f}")
        add("HTwoMHStrata", str(mh["informative_strata"]))
        add("HTwoPermP", f"{pm['p_one_sided']:.3f}")
        add("HTwoPermPTwo", f"{pm['p_two_sided']:.3f}")
        add("HTwoPermN", f"{pm['n_perm']:,}")
        md = res["mdd"]
        add("HTwoMDD", pct(md["mdd"]))
        add("HTwoMDDTarget", pct(md["p2_detectable"]))
    add("GapBoot", f"{B:,}")
    return out


def bytask_table(res: dict) -> str | None:
    if "h2" not in res:
        return None
    rows = []
    for t, v in res["h2"]["by_task"].items():
        (a, n7), (c, n1) = v["7B"], v["1.5B"]
        rows.append(f"\\code{{{t.replace('_', chr(92) + '_')}}} & {c}/{n1} & {pct(c / n1) if n1 else '--'} & "
                    f"{a}/{n7} & {pct(a / n7) if n7 else '--'} \\\\")
    return ("\\begin{tabular}{lrrrr}\n\\toprule\n"
            " & \\multicolumn{2}{c}{1.5B} & \\multicolumn{2}{c}{7B} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
            "Task & $k/n$ & $\\Fhat$ & $k/n$ & $\\Fhat$ \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def load_inputs() -> tuple[dict, dict, dict, dict]:
    from analyze_mlx import MODELS, load_model_records
    from cot_faithfulness.config import OUTPUT_DIR

    summary = json.load(open(OUTPUT_DIR / "analysis" / "summary.json"))
    labels, probes, complete = {}, {}, {}
    for size in SIZES:
        p = OUTPUT_DIR / "judge_labels" / f"{size}.json"
        if p.exists():
            labels[size] = json.load(open(p))
        p = OUTPUT_DIR / "probes" / f"{MODELS[size].replace('/', '__')}.jsonl"
        if p.exists():
            probes[size] = [json.loads(line) for line in open(p)]
        items = load_model_records(MODELS[size])
        complete[size] = {k: v for k, v in items.items() if len(v) == 4}
    return summary, labels, probes, complete


if __name__ == "__main__":
    print(json.dumps(compute(*load_inputs()), indent=1))
