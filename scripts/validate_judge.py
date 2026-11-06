#!/usr/bin/env python3
"""
Validate LLM judge against human labels (calibration gate).

Usage:
  1. Run experiment and analysis with --export-labels --skip-judge
  2. Fill human_label in outputs/human_labels.json (>=30, stratified by model x hint)
  3. Run this script — aborts if any condition has kappa < 0.6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cot_faithfulness.analysis.judge import (
    judge_case,
    compute_judge_agreement,
    passes_kappa_gate,
)
from cot_faithfulness.config import OUTPUT_DIR, JUDGE_MODEL, HUMAN_LABEL_TARGET, MIN_KAPPA


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate judge against human labels")
    parser.add_argument("--labels", type=Path, default=OUTPUT_DIR / "human_labels.json")
    parser.add_argument("--judge-model", default=JUDGE_MODEL)
    parser.add_argument("--min-labels", type=int, default=HUMAN_LABEL_TARGET)
    parser.add_argument("--min-kappa", type=float, default=MIN_KAPPA)
    parser.add_argument(
        "--allow-heuristic-judge",
        action="store_true",
        help="DEV ONLY — not valid for confirmatory calibration",
    )
    args = parser.parse_args()

    if not args.labels.exists():
        print(f"Missing {args.labels}. Copy human_labels_template.json and fill human_label fields.")
        raise SystemExit(1)

    with open(args.labels) as f:
        cases = json.load(f)

    labeled = [c for c in cases if c.get("human_label")]
    if len(labeled) < args.min_labels:
        print(f"Warning: only {len(labeled)} labeled cases (target: {args.min_labels})")

    judge_results = []
    for case in labeled:
        judge_input = {
            "model_id": case["model_id"],
            "task": case["task"],
            "example_id": case["example_id"],
            "hint_type": case["hint_type"],
            "baseline_answer": case["baseline_answer"],
            "hinted_answer": case["hinted_answer"],
            "hint_suggested": case["hint_suggested"],
            "correct_letter": case["correct_letter"],
            "hinted_cot": case["hinted_cot"],
            "hint_description": case.get("hint_description"),
        }
        result = judge_case(
            judge_input,
            model=args.judge_model,
            use_heuristic_fallback=args.allow_heuristic_judge,
        )
        result.update({k: case[k] for k in ["example_id", "hint_type", "model_id"]})
        judge_results.append(result)

    agreement = compute_judge_agreement(labeled, judge_results)

    print(f"\n=== Judge Validation (n={agreement['n']}) ===")
    print(f"Exact agreement: {agreement.get('exact_agreement_rate', 'N/A')}")
    print(f"Cohen's kappa:   {agreement.get('kappa', 'N/A')}")
    print(f"Per-condition:   {agreement.get('kappa_by_condition', {})}")

    if agreement["n"] > 0:
        print("\nDisagreements:")
        for pair in agreement["pairs"]:
            if not pair["match"]:
                print(f"  human={pair['human']}, judge={pair['judge']}")

    out_path = OUTPUT_DIR / "judge_validation.json"
    with open(out_path, "w") as f:
        json.dump(agreement, f, indent=2)
    print(f"\nSaved to {out_path}")

    ok, msg = passes_kappa_gate(agreement, args.min_kappa)
    print(msg)
    if not ok:
        print("ABORT: do not proceed to confirmatory adjudication.")
        raise SystemExit(2)
    print("Calibration gate passed — safe to run scripts/analyze.py")


if __name__ == "__main__":
    main()
