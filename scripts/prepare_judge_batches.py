#!/usr/bin/env python3
"""Split influenced cases into judge batches, twice with different shuffles.

Each case is shown in the frame the model saw (for reorder, display letters),
with option text next to every letter so the judge never has to translate.

    python scripts/prepare_judge_batches.py 1.5B 7B 3B
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cot_faithfulness.config import OUTPUT_DIR

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


def main() -> None:
    sizes = sys.argv[1:]
    out_root = OUTPUT_DIR / "judge_batches"
    files = {"pass1": [], "pass2": []}
    for size in sizes:
        cases = [judge_view(c) for c in json.load(open(OUTPUT_DIR / "judge_inputs" / f"{size}.json"))]
        for pass_name, seed in (("pass1", 1), ("pass2", 2)):
            order = cases[:]
            random.Random(f"{seed}:{size}").shuffle(order)
            for i in range(0, len(order), BATCH_SIZE):
                path = out_root / pass_name / f"{size}_b{i // BATCH_SIZE:03d}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                json.dump(order[i : i + BATCH_SIZE], open(path, "w"), indent=1)
                files[pass_name].append(str(path.relative_to(OUTPUT_DIR.parent)))  # repo-relative
        print(f"{size}: {len(cases)} cases")
    json.dump(files, open(out_root / f"manifest_{'_'.join(sizes)}.json", "w"), indent=1)
    print({k: len(v) for k, v in files.items()}, "batches")


if __name__ == "__main__":
    main()
