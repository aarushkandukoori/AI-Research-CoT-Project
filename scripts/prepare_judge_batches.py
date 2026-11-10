#!/usr/bin/env python3
"""Split influenced cases into judge batches, twice with different shuffles.

Each case is shown in the frame the model saw (for reorder, display letters),
with option text next to every letter so the judge never has to translate.

    python scripts/prepare_judge_batches.py 1.5B 7B 3B          # greedy run (outputs/)
    python scripts/prepare_judge_batches.py --root outputs/substudy_t07 1.5B 3B 7B

--root is the folder that holds judge_inputs/ (written by analyze_mlx.py or
analyze_sampled.py); batches go to <root>/judge_batches/. The default root is
outputs/, which is the original behavior.

After both passes are labeled and saved as <root>/judge_raw/<size>.json
({"pass1": {...}, "pass2": {...}}), write the tie-break batch with
    python scripts/prepare_judge_batches.py --root <root> --tiebreak <root>/judge_raw/1.5B.json 1.5B
It holds the cases where the passes disagree on the classification or on
mention vs no_mention, and lists cases that a pass left unlabeled.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cot_faithfulness.config import OUTPUT_DIR, PROJECT_ROOT

BATCH_SIZE = 20
OPTION_RE = re.compile(r"^\(([A-Z])\)\s*(.+)$", re.MULTILINE)


def judge_view(case: dict) -> dict:
    """What the judge sees: hint, answers (with option text), and the CoT."""
    options_block = case["hinted_prompt"].split("Options:", 1)[1]
    display = {l: t.strip() for l, t in OPTION_RE.findall(options_block)}
    to_display = {orig: disp for disp, orig in case["letter_map"].items()}

    def show(orig_letter: str) -> str:
        d = to_display[orig_letter]
        return f"({d}) {display[d]}"

    if case["hint_type"] == "sycophancy":
        hint = case["hint_description"]
    else:
        hint = ("The answer options were reordered so that a wrong option was moved to the first "
                "position, (A). The question text is unchanged.")
    return {
        "case_id": case["case_id"],
        "hint_type": case["hint_type"],
        "hint": hint,
        "hint_target": show(case["hint_suggested"]),
        "model_final_answer": show(case["hinted_answer"]),
        "correct_answer": show(case["correct_letter"]),
        "answer_without_hint": f"{display[to_display[case['baseline_answer']]]}",
        "chain_of_thought": case["hinted_cot"],
    }


def disagreements(raw: dict) -> tuple[list[str], list[str]]:
    """Case ids for the tie-break vote, and case ids that one pass skipped."""
    p1, p2 = raw["pass1"], raw["pass2"]
    split = [c for c in sorted(set(p1) & set(p2))
             if p1[c]["classification"] != p2[c]["classification"]
             or (p1[c]["mention_type"] == "no_mention") != (p2[c]["mention_type"] == "no_mention")]
    return split, sorted(set(p1) ^ set(p2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sizes", nargs="+", help="labels of judge_inputs/<label>.json files")
    parser.add_argument("--root", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--tiebreak", type=Path, default=None, help="judge_raw file with pass1 and pass2 votes")
    args = parser.parse_args()
    out_root = args.root / "judge_batches"

    if args.tiebreak:
        (size,) = args.sizes  # one size per raw file
        split, skipped = disagreements(json.load(open(args.tiebreak)))
        cases = {c["case_id"]: c for c in json.load(open(args.root / "judge_inputs" / f"{size}.json"))}
        path = out_root / "tiebreak" / f"{size}_tb000.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        json.dump([judge_view(cases[c]) for c in split], open(path, "w"), indent=1)
        print(f"{size}: {len(split)} cases need a tie-break vote -> {path}")
        if skipped:
            print(f"{len(skipped)} cases are labeled in only one pass; re-label them in the pass that missed them:",
                  skipped)
        return

    files = {"pass1": [], "pass2": []}
    for size in args.sizes:
        cases = [judge_view(c) for c in json.load(open(args.root / "judge_inputs" / f"{size}.json"))]
        for pass_name, seed in (("pass1", 1), ("pass2", 2)):
            order = cases[:]
            random.Random(f"{seed}:{size}").shuffle(order)
            for i in range(0, len(order), BATCH_SIZE):
                path = out_root / pass_name / f"{size}_b{i // BATCH_SIZE:03d}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                json.dump(order[i : i + BATCH_SIZE], open(path, "w"), indent=1)
                files[pass_name].append(os.path.relpath(path.resolve(), PROJECT_ROOT))  # repo-relative
        print(f"{size}: {len(cases)} cases")
    json.dump(files, open(out_root / f"manifest_{'_'.join(args.sizes)}.json", "w"), indent=1)
    print({k: len(v) for k, v in files.items()}, "batches")


if __name__ == "__main__":
    main()
