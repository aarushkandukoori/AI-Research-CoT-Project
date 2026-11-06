#!/usr/bin/env python3
"""Merge the two blind judge passes (+ tie-break) into final labels and agreement stats.

    python scripts/finalize_judge.py outputs/judge_raw/*.json
Writes outputs/judge_labels/<size>.json and outputs/analysis/judge_agreement.json.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cot_faithfulness.analysis.judge import cohens_kappa
from cot_faithfulness.config import OUTPUT_DIR


def majority(values: list[str], fallback: str) -> str:
    top, count = Counter(values).most_common(1)[0]
    return top if count >= 2 or len(values) == 1 else fallback


def main() -> None:
    p1, p2, p3 = {}, {}, {}
    for path in sys.argv[1:]:
        raw = json.load(open(path))
        p1.update(raw["pass1"]); p2.update(raw["pass2"]); p3.update(raw.get("tiebreak", {}))

    inputs = {}
    for path in (OUTPUT_DIR / "judge_inputs").glob("*.json"):
        for case in json.load(open(path)):
            inputs[case["case_id"]] = case

    final: dict[str, dict] = defaultdict(dict)
    pairs = defaultdict(list)          # (size|hint) -> [(p1 class, p2 class)]
    mention_pairs = defaultdict(list)  # (size|hint) -> [(p1 mention, p2 mention)]
    lexical_pairs = defaultdict(list)  # size -> [(lexical mention, final judge mention)] for sycophancy
    for case_id in sorted(set(p1) | set(p2)):
        votes = [p[case_id] for p in (p1, p2, p3) if case_id in p]
        size, hint, _ = case_id.split("|", 2)
        cell = f"{size}|{hint}"
        # Three-way splits (F/U/A) fall back to AMBIGUOUS, which the F-/F+ bracket covers.
        cls = majority([v["classification"] for v in votes], "AMBIGUOUS")
        mentioned = majority([str(v["mention_type"] != "no_mention") for v in votes], "True") == "True"
        mtypes = [v["mention_type"] for v in votes if (v["mention_type"] != "no_mention") == mentioned]
        final[size][case_id] = {
            "classification": cls,
            "mentions_hint": mentioned,
            "mention_type": Counter(mtypes).most_common(1)[0][0],
            "n_votes": len(votes),
            "evidence": next((v["evidence"] for v in votes if v["evidence"]), ""),
        }
        if case_id in p1 and case_id in p2:
            pairs[cell].append((p1[case_id]["classification"], p2[case_id]["classification"]))
            mention_pairs[cell].append((p1[case_id]["mention_type"] != "no_mention", p2[case_id]["mention_type"] != "no_mention"))
        if hint == "sycophancy" and case_id in inputs:
            lexical_pairs[size].append((inputs[case_id]["lexical_mention"], mentioned))

    out_dir = OUTPUT_DIR / "judge_labels"
    out_dir.mkdir(parents=True, exist_ok=True)
    for size, labels in final.items():
        json.dump(labels, open(out_dir / f"{size}.json", "w"), indent=1)

    def agree(pp):
        return {"n": len(pp), "exact": sum(a == b for a, b in pp) / len(pp),
                "kappa": cohens_kappa([(str(a), str(b)) for a, b in pp])}

    agreement = {
        "pass1_vs_pass2_classification": {c: agree(v) for c, v in pairs.items()},
        "pass1_vs_pass2_mention": {c: agree(v) for c, v in mention_pairs.items()},
        "lexical_vs_judge_mention_sycophancy": {
            s: {**agree(v),
                "lexical_only": sum(a and not b for a, b in v),
                "judge_only": sum(b and not a for a, b in v)}
            for s, v in lexical_pairs.items()
        },
        "n_tiebreak": len(p3),
    }
    json.dump(agreement, open(OUTPUT_DIR / "analysis" / "judge_agreement.json", "w"), indent=1)
    print(json.dumps(agreement, indent=1))


if __name__ == "__main__":
    main()
