#!/usr/bin/env python3
"""Gate G2: do the judge's labels match the human labels?

Input: the CSV that the labeling page (sheet.html) downloads, sheet_labeled.csv. It has one
row per case with the columns case_id, label (FAITHFUL / UNFAITHFUL / AMBIGUOUS),
mentions_hint (yes / no) and notes. The other columns are ignored.

Each row is joined, by case_id, with the final judge label in outputs/judge_labels/<size>.json.
The script writes outputs/analysis/g2_agreement.json with, for each condition, each condition
at each model size, each model size, and all cases together:
  - n, raw agreement and Cohen's kappa on the 3-way label
  - kappa with AMBIGUOUS merged, both ways: as "not acknowledging" (how the paper's F counts
    it) and as "acknowledging" (how F+ counts it)
  - agreement on whether the CoT mentions the hint at all
  - confusion matrices (rows = human label, columns = judge label)
  - bootstrap 95% CIs
  - agreement reweighted to all judged cases: the labeling set oversampled the cases the judge
    flagged, so each labeled case is counted as the N/n judged cases it stands for
and the preregistered gate G2 for each condition: Cohen's kappa >= 0.6 on >= 30 human labels.

    python3 scripts/g2_agreement.py ~/random/cot-paper-latex/g2-labeling/sheet_labeled.csv \
        --paper-out ~/random/cot-paper-latex/generated

--paper-out also reruns make_paper_assets.py into that folder, so the paper's macros pick up
the G2 numbers. If any case is still unlabeled the script stops and lists it. With
--allow-partial it scores the labeled cases only and writes g2_agreement_partial.json, which
the paper never reads.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cot_faithfulness.config import BOOTSTRAP_SAMPLES, HUMAN_LABEL_TARGET, MIN_KAPPA, OUTPUT_DIR

SIZES = ["1.5B", "3B", "7B"]
CONDITIONS = ["sycophancy", "reorder"]
LABELS = ["FAITHFUL", "UNFAITHFUL", "AMBIGUOUS"]

# Four ways to compare the two raters, with the answers each one allows.
# The gate uses "three_way", the preregistered label.
VIEWS = {
    "three_way": LABELS,
    "ambiguous_as_no": ["FAITHFUL", "NOT_FAITHFUL"],               # AMBIGUOUS = not acknowledging, like F
    "ambiguous_as_yes": ["FAITHFUL_OR_AMBIGUOUS", "UNFAITHFUL"],   # AMBIGUOUS = acknowledging, like F+
    "mention": ["yes", "no"],                                      # does the CoT refer to the hint at all
}
# For the yes/no views: which answer counts as "yes" when we compute a rate.
POSITIVE = {"ambiguous_as_no": "FAITHFUL", "ambiguous_as_yes": "FAITHFUL_OR_AMBIGUOUS", "mention": "yes"}

# Names used in LaTeX macros (same as make_paper_assets.py). Macro names cannot hold digits.
WORD = {"1.5B": "OneFive", "3B": "Three", "7B": "Seven"}
HINT_WORD = {"sycophancy": "Syc", "reorder": "Ord"}


# ---------------------------------------------------------------- reading the labels

def read_view(label: str, mention: str, view: str) -> str:
    """One rater's answer for one case, read in one of the four ways in VIEWS."""
    if view == "three_way":
        return label
    if view == "ambiguous_as_no":
        return "FAITHFUL" if label == "FAITHFUL" else "NOT_FAITHFUL"
    if view == "ambiguous_as_yes":
        return "UNFAITHFUL" if label == "UNFAITHFUL" else "FAITHFUL_OR_AMBIGUOUS"
    return mention  # view == "mention"


def judge_flagged(judge: dict) -> bool:
    """The rule export_human_labels.py used to oversample a case: the judge did not call it
    UNFAITHFUL, or the judge found a mention of the hint."""
    return judge["classification"] != "UNFAITHFUL" or bool(judge["mentions_hint"])


def stratum_of(case_id: str, judge: dict) -> str:
    """The sampling stratum of a case: model size | hint | whether the judge flagged it."""
    size, hint, _ = case_id.split("|", 2)
    return f"{size}|{hint}|{'flagged' if judge_flagged(judge) else 'not_flagged'}"


def read_sheet(path: Path) -> list[dict]:
    """All rows of the downloaded CSV. utf-8-sig also reads a file that a spreadsheet app
    re-saved with a byte-order mark."""
    if not path.exists():
        raise SystemExit(f"Can't find {path}. The labeling page saves sheet_labeled.csv to your "
                         "Downloads folder; move it next to sheet.html or pass its real path.")
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path} has no rows.")
    missing = {"case_id", "label", "mentions_hint"} - set(rows[0])
    if missing:
        raise SystemExit(f"{path} is missing the column(s) {sorted(missing)}.")
    return rows


def load_judge(judge_dir: Path) -> dict[str, dict]:
    """Final judge labels for every judged case, keyed by case_id."""
    judge = {}
    for size in SIZES:
        path = judge_dir / f"{size}.json"
        if path.exists():
            judge.update(json.load(open(path)))
    if not judge:
        raise SystemExit(f"No judge labels found in {judge_dir}.")
    return judge


def join_labels(rows: list[dict], judge: dict) -> tuple[list[dict], list[str], list[str]]:
    """Pair every labeled row with its judge label.

    Returns (cases, unlabeled, problems). `unlabeled` lists rows that still miss the label or
    the mention answer; `problems` lists anything that makes the file unusable.
    """
    cases, unlabeled, problems, seen = [], [], [], set()
    for line, row in enumerate(rows, start=2):  # line 1 is the header
        name = f"case {row['row']}" if row.get("row") else f"line {line}"  # "case N" = "Case N of 90" on the page
        case_id = (row.get("case_id") or "").strip()
        label = (row.get("label") or "").strip().upper()
        mention = (row.get("mentions_hint") or "").strip().lower()
        if case_id in seen:
            problems.append(f"{name}: case_id {case_id} appears twice")
        seen.add(case_id)
        if case_id not in judge:
            problems.append(f"{name}: case_id {case_id!r} has no judge label")
            continue
        size, hint, _ = case_id.split("|", 2)
        if row.get("hint_type") and row["hint_type"].strip() != hint:
            problems.append(f"{name}: hint_type {row['hint_type']!r} does not match case_id {case_id}")
        if not label or not mention:
            unlabeled.append(name)
            continue
        if label not in LABELS:
            problems.append(f"{name}: label {row['label']!r} is not FAITHFUL, UNFAITHFUL or AMBIGUOUS")
            continue
        if mention not in ("yes", "no"):
            problems.append(f"{name}: mentions_hint {row['mentions_hint']!r} is not yes or no")
            continue
        j = judge[case_id]
        judge_mention = "yes" if j["mentions_hint"] else "no"
        cases.append({
            "case_id": case_id, "size": size, "hint": hint, "name": name,
            "stratum": stratum_of(case_id, j),
            # (human answer, judge answer) under each of the four views
            "pairs": {v: (read_view(label, mention, v), read_view(j["classification"], judge_mention, v))
                      for v in VIEWS},
        })
    return cases, unlabeled, problems


# ---------------------------------------------------------------- the statistics

def agreement_and_kappa(pairs: list[tuple[str, str]], weights: list[float]) -> tuple[float, float | None]:
    """Raw agreement and Cohen's kappa, with each case counted `weight` times.

    With every weight equal to 1 this is the usual Cohen's kappa:
      p_o   = share of cases where human and judge give the same answer
      p_e   = agreement expected by chance = sum over answers of (human share) x (judge share)
      kappa = (p_o - p_e) / (1 - p_e)
    If both raters give every case one and the same answer, p_e = 1 and kappa is undefined, so
    None is returned. (cohens_kappa() in cot_faithfulness returns 1.0 there, which would pass
    the gate on no information, so it is not used here.)
    """
    total = sum(weights)
    agree = 0.0
    human, judge = defaultdict(float), defaultdict(float)
    for (h, j), w in zip(pairs, weights):
        if h == j:
            agree += w
        human[h] += w
        judge[j] += w
    p_o = agree / total
    p_e = sum(human[x] * judge[x] for x in human) / (total * total)
    if p_e > 1 - 1e-9:
        return p_o, None
    return p_o, (p_o - p_e) / (1 - p_e)


def share(answers: list[str], weights: list[float], positive: str) -> float:
    """Weighted share of answers equal to `positive`."""
    return sum(w for a, w in zip(answers, weights) if a == positive) / sum(weights)


def confusion(pairs: list[tuple[str, str]], answers: list[str]) -> dict:
    """Counts of each (human, judge) pair: rows = human answer, columns = judge answer."""
    table = {h: {j: 0 for j in answers} for h in answers}
    for h, j in pairs:
        table[h][j] += 1
    return table


def point_stats(cases: list[dict]) -> dict:
    """The numbers that get a bootstrap CI, for one list of labeled cases."""
    ones = [1.0] * len(cases)
    weights = [c["weight"] for c in cases]
    out = {}
    for view in VIEWS:
        pairs = [c["pairs"][view] for c in cases]
        agree, kappa = agreement_and_kappa(pairs, ones)
        agree_rw, kappa_rw = agreement_and_kappa(pairs, weights)
        out[view] = {"agreement": agree, "kappa": kappa,
                     "agreement_reweighted": agree_rw, "kappa_reweighted": kappa_rw}
        if view in POSITIVE:
            # the human "yes" rate, reweighted to all judged cases (what the rate would be if
            # the human had labeled every judged case)
            out[view]["human_rate_reweighted"] = share([h for h, _ in pairs], weights, POSITIVE[view])
    return out


def bootstrap_cis(cases: list[dict], rounds: int, seed: int) -> dict:
    """95% percentile intervals for every number in point_stats().

    The labeling set was drawn separately inside each sampling stratum, so each round redraws,
    with replacement, as many cases from each stratum as it holds. Rounds where a kappa is
    undefined are left out of that kappa's interval and counted.
    """
    rng = random.Random(seed)
    strata = defaultdict(list)
    for c in cases:
        strata[c["stratum"]].append(c)
    draws = defaultdict(list)  # (view, statistic) -> its value in every round
    for _ in range(rounds):
        sample = [rng.choice(members) for members in strata.values() for _ in members]
        for view, stats in point_stats(sample).items():
            for name, value in stats.items():
                if value is not None:
                    draws[(view, name)].append(value)
    cis = {}
    for key, values in draws.items():
        lo, hi = np.percentile(values, [2.5, 97.5])
        cis[key] = {"ci95": [float(lo), float(hi)], "undefined_rounds": rounds - len(values)}
    return cis


def summarize(cases: list[dict], population: list[dict], rounds: int, seed: int) -> dict:
    """Every agreement number for one group of labeled cases (for example 7B sycophancy).

    `population` holds the judge labels of every judged case in the same group, used for the
    judge's own rate over all of them.
    """
    point = point_stats(cases)
    cis = bootstrap_cis(cases, rounds, seed)
    out = {"n": len(cases)}
    for view, answers in VIEWS.items():
        pairs = [c["pairs"][view] for c in cases]
        block = {}
        for name, value in point[view].items():
            block[name] = value
            ci = cis.get((view, name))
            block[name + "_ci95"] = ci["ci95"] if ci else None
            if ci and ci["undefined_rounds"]:
                block[name + "_undefined_rounds"] = ci["undefined_rounds"]
        block["confusion"] = confusion(pairs, answers)
        block["human_answers_used"] = sorted({h for h, _ in pairs})
        block["judge_answers_used"] = sorted({j for _, j in pairs})
        if view in POSITIVE:
            judge_answers = [read_view(j["classification"], "yes" if j["mentions_hint"] else "no", view)
                             for j in population]
            block["judge_rate_all_judged"] = share(judge_answers, [1.0] * len(judge_answers), POSITIVE[view])
        out[view] = block
    # The cases the judge did not flag (UNFAITHFUL, no mention) are most of the judged set.
    # Plain counts of what the human said about the labeled ones.
    quiet = [c for c in cases if c["stratum"].endswith("|not_flagged")]
    out["judge_not_flagged"] = {
        "labeled": len(quiet),
        "human_faithful": sum(c["pairs"]["three_way"][0] == "FAITHFUL" for c in quiet),
        "human_ambiguous": sum(c["pairs"]["three_way"][0] == "AMBIGUOUS" for c in quiet),
        "human_mentions_hint": sum(c["pairs"]["mention"][0] == "yes" for c in quiet),
    }
    return out


def gate(group: dict) -> dict:
    """The preregistered G2 rule for one condition: Cohen's kappa >= 0.6 on >= 30 human labels
    (3-way label, model sizes pooled, unweighted, as preregistered)."""
    n, tw = group["n"], group["three_way"]
    kappa = tw["kappa"]
    if n < HUMAN_LABEL_TARGET:
        status, reason = "not met", f"only {n} labeled cases; the gate needs at least {HUMAN_LABEL_TARGET}"
    elif kappa is None:
        status, reason = "not computable", "human and judge gave every case the same label, so agreement is 100% but kappa is undefined"
    elif kappa >= MIN_KAPPA:
        status, reason = "met", f"kappa {kappa:.2f} is at least {MIN_KAPPA}"
    else:
        status, reason = "not met", f"kappa {kappa:.2f} is below {MIN_KAPPA}"
    if len(tw["judge_answers_used"]) == 1 and kappa is not None:
        reason += ("; the judge gave every case in this condition the same label, so kappa is 0 whatever "
                   "the human labels are, and raw agreement is the informative number")
    return {"status": status, "reason": reason, "n": n, "kappa": kappa, "kappa_ci95": tw["kappa_ci95"],
            "agreement": tw["agreement"]}


# ---------------------------------------------------------------- putting it together

def group_cells() -> dict:
    """Which (size, hint) cells make up each reported group."""
    groups = {}
    for hint in CONDITIONS:
        groups[("by_condition", hint)] = {(s, hint) for s in SIZES}
        for size in SIZES:
            groups[("by_cell", f"{size}|{hint}")] = {(size, hint)}
    for size in SIZES:
        groups[("by_size", size)] = {(size, h) for h in CONDITIONS}
    groups[("overall", "all")] = {(s, h) for s in SIZES for h in CONDITIONS}
    return groups


def tidy(x):
    """Round floats for the JSON file and turn -0.0 into 0.0."""
    if isinstance(x, float):
        x = round(x, 6)
        return 0.0 if x == 0 else x
    if isinstance(x, dict):
        return {k: tidy(v) for k, v in x.items()}
    if isinstance(x, list):
        return [tidy(v) for v in x]
    return x


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gate G2: judge labels vs human labels.")
    parser.add_argument("labels_csv", help="the CSV the labeling page downloaded (sheet_labeled.csv)")
    parser.add_argument("--judge-dir", default=str(OUTPUT_DIR / "judge_labels"))
    parser.add_argument("--todo", default=str(OUTPUT_DIR / "human_labels_todo.json"),
                        help="the drawn cases (export_human_labels.py); used to check that none is missing")
    parser.add_argument("--out", help="output JSON (default outputs/analysis/g2_agreement.json, "
                                      "or g2_agreement_partial.json with --allow-partial)")
    parser.add_argument("--allow-partial", action="store_true", help="score whatever is labeled so far")
    parser.add_argument("--boot", type=int, default=BOOTSTRAP_SAMPLES, help="bootstrap rounds")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--paper-out", help="after scoring, rerun make_paper_assets.py into this folder")
    args = parser.parse_args(argv)

    labels_path = Path(args.labels_csv).expanduser()
    rows = read_sheet(labels_path)
    judge = load_judge(Path(args.judge_dir).expanduser())
    cases, unlabeled, problems = join_labels(rows, judge)

    # Every drawn case should be in the sheet (the todo file is git-ignored, so it may be absent).
    todo_path = Path(args.todo).expanduser()
    missing_from_sheet = []
    if todo_path.exists():
        drawn = [t["case_id"] for t in json.load(open(todo_path))]
        in_sheet = {(r.get("case_id") or "").strip() for r in rows}
        missing_from_sheet = [cid for cid in drawn if cid not in in_sheet]
        problems += [f"case_id {cid} is in the sheet but was never drawn" for cid in sorted(in_sheet - set(drawn))]
    else:
        print(f"note: {todo_path} not found, so the check that every drawn case is in the sheet was skipped")

    if problems:
        raise SystemExit("Fix these in the labels file first:\n  " + "\n  ".join(problems))
    complete = not unlabeled and not missing_from_sheet
    if not complete and not args.allow_partial:
        lines = [f"{len(unlabeled)} case(s) still need a label and a mentions_hint answer: {', '.join(unlabeled)}"]
        if missing_from_sheet:
            lines.append(f"{len(missing_from_sheet)} drawn case(s) are not in the file at all")
        lines.append("Finish them on the page and download again, or add --allow-partial for a progress check.")
        raise SystemExit("\n".join(lines))
    if not cases:
        raise SystemExit("No case is fully labeled yet.")

    # Oversampling weights: a labeled case stands for (judged cases in its stratum) / (labeled cases in it).
    population = Counter(stratum_of(cid, j) for cid, j in judge.items())
    labeled = Counter(c["stratum"] for c in cases)
    for c in cases:
        c["weight"] = population[c["stratum"]] / labeled[c["stratum"]]
    weights = {s: {"judged_cases": population[s], "labeled": labeled[s], "weight": population[s] / labeled[s]}
               for s in sorted(labeled)}
    uncovered = {s: population[s] for s in sorted(population) if s not in labeled}

    result = {
        "what": "Gate G2: final judge labels vs human labels from the G2 labeling sheet",
        "complete": complete,
        "labels_file": labels_path.name,
        "labels_file_sha256": sha256(labels_path),
        "judge_labels_sha256": {p.name: sha256(p) for p in sorted(Path(args.judge_dir).expanduser().glob("*.json"))},
        "n_rows_in_sheet": len(rows),
        "n_labeled": len(cases),
        "unlabeled_cases": unlabeled,
        "drawn_cases_missing_from_sheet": missing_from_sheet,
        "gate_rule": {
            "statistic": "Cohen's kappa on the 3-way label (FAITHFUL / UNFAITHFUL / AMBIGUOUS), "
                         "computed per condition with model sizes pooled, unweighted",
            "min_kappa": MIN_KAPPA,
            "min_labels_per_condition": HUMAN_LABEL_TARGET,
        },
        "gate": {},
        "by_condition": {}, "by_cell": {}, "by_size": {}, "overall": {},
        "weights": weights,
        "strata_without_labels": uncovered,
        "bootstrap": {"rounds": args.boot, "seed": args.seed,
                      "method": "percentile; cases redrawn with replacement inside each sampling stratum"},
        "notes": [
            "confusion: rows are the human answer, columns the judge answer",
            "*_reweighted: each labeled case counts as (judged cases in its stratum) / (labeled cases in its stratum)",
            "human_rate_reweighted: share of 'yes' answers (FAITHFUL; FAITHFUL or AMBIGUOUS; mentions the hint) "
            "in the human labels, reweighted to all judged cases",
            "judge_rate_all_judged: the same share in the judge labels over every judged case of the group",
            "judge_not_flagged: what the human said about the labeled cases the judge called UNFAITHFUL with no mention",
            "a bootstrap round can only redraw answers seen in a stratum, so when a stratum has none of some answer "
            "(for example no human FAITHFUL among the not-flagged cases) the reweighted intervals are too narrow; "
            "report the judge_not_flagged counts next to them",
            "by_size and overall pool conditions; the gate is per condition only",
        ],
    }
    for (kind, name), cells in group_cells().items():
        members = [c for c in cases if (c["size"], c["hint"]) in cells]
        if members:
            group_population = [j for cid, j in judge.items() if tuple(cid.split("|", 2)[:2]) in cells]
            result[kind][name] = summarize(members, group_population, args.boot, args.seed)
    for hint in CONDITIONS:
        if hint in result["by_condition"]:
            result["gate"][hint] = gate(result["by_condition"][hint])

    analysis_dir = OUTPUT_DIR / "analysis"
    default_name = "g2_agreement.json" if complete else "g2_agreement_partial.json"
    out_path = Path(args.out).expanduser() if args.out else analysis_dir / default_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(tidy(result), open(out_path, "w"), indent=1)

    # Short report in plain words.
    print(f"G2: {len(cases)} of {len(rows)} cases in the sheet are labeled"
          + ("" if complete else " (PARTIAL: progress check only, the paper ignores this file)"))
    for hint, g in result["gate"].items():
        ci = g["kappa_ci95"]
        kappa_txt = "undefined" if g["kappa"] is None else f"{g['kappa']:.2f}" + (f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "")
        print(f"  {hint:<10} n={g['n']:<3} agreement {100 * g['agreement']:.1f}%  kappa {kappa_txt}  ->  gate {g['status']} ({g['reason']})")
    overall = result["overall"]["all"]["three_way"]
    print(f"  all cases, reweighted to every judged case: agreement {100 * overall['agreement_reweighted']:.1f}%")
    if uncovered:
        print(f"  warning: strata with judged cases but no labels (left out of reweighting): {uncovered}")
    print(f"wrote {out_path}")

    if args.paper_out:
        if not complete:
            print("not updating the paper: the labels are not complete")
        else:
            subprocess.run([sys.executable, str(ROOT / "scripts" / "make_paper_assets.py"),
                            "--out", str(Path(args.paper_out).expanduser()), "--g2", str(out_path)], check=True)
    return 0


# ---------------------------------------------------------------- for the paper (make_paper_assets.py)

def pct(x: float | None) -> str:
    """Percent with one decimal, "--" when missing (same as make_paper_assets.pct)."""
    if x is None:
        return "--"
    s = f"{100 * x:.1f}"
    return s[1:] if s.startswith("-") and float(s) == 0 else s


def kap(x: float | None) -> str:
    """Kappa with two decimals, "--" when undefined."""
    if x is None:
        return "--"
    s = f"{x:.2f}"
    return s[1:] if s.startswith("-") and float(s) == 0 else s


def ci_text(ci: list | None, fmt) -> str:
    return "--" if ci is None else f"[{fmt(ci[0])}, {fmt(ci[1])}]"


def paper_macros(g2: dict) -> list[tuple[str, str]]:
    """LaTeX macro (name, value) pairs for the G2 numbers. Empty unless every case is labeled,
    so the paper can never print a number from a partial file."""
    if not g2.get("complete"):
        return []
    out = [("GTwoN", str(g2["n_labeled"])),
           ("GTwoMinKappa", str(g2["gate_rule"]["min_kappa"])),
           ("GTwoMinLabels", str(g2["gate_rule"]["min_labels_per_condition"]))]

    def group_macros(grp: dict, tag: str) -> list[tuple[str, str]]:
        tw, no, yes, me = grp["three_way"], grp["ambiguous_as_no"], grp["ambiguous_as_yes"], grp["mention"]
        return [
            (f"GTwoN{tag}", str(grp["n"])),
            (f"GTwoAgree{tag}", pct(tw["agreement"])),
            (f"GTwoKappa{tag}", kap(tw["kappa"])),
            (f"GTwoKappaCI{tag}", ci_text(tw["kappa_ci95"], kap)),
            (f"GTwoKappaAmbNo{tag}", kap(no["kappa"])),
            (f"GTwoKappaAmbYes{tag}", kap(yes["kappa"])),
            (f"GTwoMentionAgree{tag}", pct(me["agreement"])),
            (f"GTwoMentionKappa{tag}", kap(me["kappa"])),
            (f"GTwoAgreeRew{tag}", pct(tw["agreement_reweighted"])),
            (f"GTwoAgreeRewCI{tag}", ci_text(tw["agreement_reweighted_ci95"], pct)),
            # human FAITHFUL rate reweighted to all judged cases, next to the judge's rate on all of them
            (f"GTwoHumanF{tag}", pct(no["human_rate_reweighted"])),
            (f"GTwoHumanFCI{tag}", ci_text(no["human_rate_reweighted_ci95"], pct)),
            (f"GTwoJudgeF{tag}", pct(no["judge_rate_all_judged"])),
            # cases the human called FAITHFUL and the judge did not, and the reverse
            (f"GTwoMissed{tag}", str(no["confusion"]["FAITHFUL"]["NOT_FAITHFUL"])),
            (f"GTwoExtra{tag}", str(no["confusion"]["NOT_FAITHFUL"]["FAITHFUL"])),
            # labeled cases the judge did not flag, and what the human found in them
            (f"GTwoNotFlaggedN{tag}", str(grp["judge_not_flagged"]["labeled"])),
            (f"GTwoNotFlaggedF{tag}", str(grp["judge_not_flagged"]["human_faithful"])),
            (f"GTwoNotFlaggedA{tag}", str(grp["judge_not_flagged"]["human_ambiguous"])),
            (f"GTwoNotFlaggedMention{tag}", str(grp["judge_not_flagged"]["human_mentions_hint"])),
        ]

    for hint, hw in HINT_WORD.items():
        if hint in g2["by_condition"]:
            out += group_macros(g2["by_condition"][hint], hw)
            out.append((f"GTwoGate{hw}", g2["gate"][hint]["status"]))
        for size, sw in WORD.items():
            if f"{size}|{hint}" in g2["by_cell"]:
                out += group_macros(g2["by_cell"][f"{size}|{hint}"], hw + sw)
    if "all" in g2["overall"]:
        out += group_macros(g2["overall"]["all"], "All")
    return out


def paper_table(g2: dict) -> str:
    """A LaTeX tabular with one row per model size and condition, plus pooled rows per condition."""
    rows = []
    for hint, tex in (("sycophancy", r"\syc"), ("reorder", r"\ord")):
        names = [(size, f"{size}|{hint}", g2["by_cell"]) for size in SIZES] + [("All", hint, g2["by_condition"])]
        for label, key, source in names:
            grp = source.get(key)
            if not grp:
                continue
            tw = grp["three_way"]
            kappa_cell = "--" if tw["kappa"] is None else f"{kap(tw['kappa'])} {ci_text(tw['kappa_ci95'], kap)}"
            rows.append(f"{label} & {tex} & {grp['n']} & {pct(tw['agreement'])} & {kappa_cell} & "
                        f"{kap(grp['ambiguous_as_no']['kappa'])} & {kap(grp['ambiguous_as_yes']['kappa'])} & "
                        f"{pct(grp['mention']['agreement'])} & {kap(grp['mention']['kappa'])} \\\\")
        rows.append(r"\midrule")
    return ("\\begin{tabular}{llrrlrrrr}\n\\toprule\n"
            "Model & Hint & $n$ & Agree & $\\kappa$ [95\\% CI] & $\\kappa$, F vs.\\ rest & $\\kappa$, U vs.\\ rest & "
            "Mention agree & Mention $\\kappa$ \\\\\n\\midrule\n"
            + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")


if __name__ == "__main__":
    sys.exit(main())
