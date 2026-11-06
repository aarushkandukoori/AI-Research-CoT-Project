#!/usr/bin/env python3
"""Stratified set of influenced cases for human labeling (gate G2).

Positives are rare, so cases the judge called FAITHFUL or AMBIGUOUS, or that
mention the hint, are oversampled. Judge labels are NOT written to the file, so
labelers stay blind to them.

    python scripts/export_human_labels.py   # -> outputs/human_labels_todo.json
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cot_faithfulness.config import OUTPUT_DIR

PER_CELL = {"sycophancy": 20, "reorder": 10}


def main() -> None:
    rng = random.Random(2026)
    picked = []
    for inputs_path in sorted((OUTPUT_DIR / "judge_inputs").glob("*.json")):
        size = inputs_path.stem
        labels_path = OUTPUT_DIR / "judge_labels" / f"{size}.json"
        if not labels_path.exists():
            continue
        labels = json.load(open(labels_path))
        cases = json.load(open(inputs_path))
        for hint, n in PER_CELL.items():
            cell = [c for c in cases if c["hint_type"] == hint and c["case_id"] in labels]
            flagged = [c for c in cell if labels[c["case_id"]]["classification"] != "UNFAITHFUL"
                       or labels[c["case_id"]]["mentions_hint"]]
            rest = [c for c in cell if c not in flagged]
            take = rng.sample(flagged, min(len(flagged), n // 2))
            take += rng.sample(rest, min(len(rest), n - len(take)))
            picked += take
    rng.shuffle(picked)
    out = [{
        "case_id": c["case_id"], "hint_type": c["hint_type"], "hint_description": c["hint_description"],
        "hint_target": c["hint_suggested"], "model_answer": c["hinted_answer"],
        "correct_answer": c["correct_letter"], "answer_without_hint": c["baseline_answer"],
        "chain_of_thought": c["hinted_cot"],
        "human_label": None,        # FAITHFUL | UNFAITHFUL | AMBIGUOUS
        "human_mentions_hint": None,  # true | false
        "notes": "",
    } for c in picked]
    path = OUTPUT_DIR / "human_labels_todo.json"
    json.dump(out, open(path, "w"), indent=1)
    print(f"{len(out)} cases -> {path}")


if __name__ == "__main__":
    main()
