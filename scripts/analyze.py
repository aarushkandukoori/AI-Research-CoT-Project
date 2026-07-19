#!/usr/bin/env python3
"""Analyze cached generations: influence detection, judging, report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cot_faithfulness.analysis.influence import identify_influenced_cases
from cot_faithfulness.analysis.judge import load_or_judge_case, compute_judge_agreement
from cot_faithfulness.analysis.report import build_report
from cot_faithfulness.config import CACHE_DIR, OUTPUT_DIR, JUDGE_MODEL, HUMAN_LABEL_TARGET
from cot_faithfulness.inference.cache import iter_cached_records


def load_human_labels(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def export_influenced_for_labeling(influenced: list, output_path: Path, n: int = 40) -> None:
    """Export cases for hand-labeling."""
    cases = []
    for i, case in enumerate(influenced[:n]):
        cases.append({
            "id": i + 1,
            "example_id": case.example_id,
            "model_id": case.model_id,
            "hint_type": case.hint_type,
            "task": case.task,
            "baseline_answer": case.baseline_answer,
            "hinted_answer": case.hinted_answer,
            "hint_suggested": case.hint_suggested,
            "correct_letter": case.correct_letter,
            "hint_description": case.hint_description,
            "hinted_cot": case.hinted_cot,
            "human_label": None,  # FAITHFUL | UNFAITHFUL | AMBIGUOUS
            "human_notes": "",
        })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(cases, f, indent=2)
    print(f"Exported {len(cases)} cases for labeling to {output_path}")


def case_to_judge_dict(case) -> dict:
    return {
        "model_id": case.model_id,
        "task": case.task,
        "example_id": case.example_id,
        "hint_type": case.hint_type,
        "baseline_answer": case.baseline_answer,
        "hinted_answer": case.hinted_answer,
        "hint_suggested": case.hint_suggested,
        "correct_letter": case.correct_letter,
        "hinted_cot": case.hinted_cot,
        "hint_description": case.hint_description,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze CoT faithfulness experiment")
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--judge-model", default=JUDGE_MODEL)
    parser.add_argument("--human-labels", type=Path, default=None)
    parser.add_argument("--export-labels", action="store_true", help="Export influenced cases for labeling")
    parser.add_argument("--skip-judge", action="store_true", help="Skip LLM judge (metrics only)")
    args = parser.parse_args()

    records = iter_cached_records(args.cache_dir)
    if not records:
        print("No cached records found. Run scripts/run_experiment.py first.")
        return

    print(f"Loaded {len(records)} cached generation records")
    influenced = identify_influenced_cases(records)
    print(f"Found {len(influenced)} hint-influenced answer flips")

    export_path = args.output_dir / "human_labels_template.json"
    if args.export_labels or not (args.output_dir / "human_labels.json").exists():
        export_influenced_for_labeling(influenced, export_path, n=max(40, HUMAN_LABEL_TARGET + 10))

    human_labels_path = args.human_labels or (args.output_dir / "human_labels.json")
    human_labels = load_human_labels(human_labels_path)
    labeled = [h for h in human_labels if h.get("human_label")]

    judge_results = []
    if not args.skip_judge and influenced:
        print(f"Running LLM judge ({args.judge_model}) on influenced cases...")
        for case in influenced:
            result = load_or_judge_case(
                case_to_judge_dict(case),
                args.cache_dir,
                judge_model=args.judge_model,
            )
            judge_results.append(result)

    judge_validation = None
    if labeled and judge_results:
        # Match human labels to judge results
        judge_for_validation = []
        human_map = {(h["example_id"], h["hint_type"], h["model_id"]): h for h in labeled}
        for jr in judge_results:
            key = (jr["example_id"], jr["hint_type"], jr["model_id"])
            if key in human_map:
                judge_for_validation.append(jr)
        judge_validation = compute_judge_agreement(labeled, judge_for_validation)
        print(f"Judge-human agreement: {judge_validation.get('exact_agreement_rate', 'N/A')} (n={judge_validation.get('n', 0)})")

        val_path = args.output_dir / "judge_validation.json"
        with open(val_path, "w") as f:
            json.dump(judge_validation, f, indent=2)

    # Load device probe if available
    probe_path = args.output_dir / "device_probe.json"
    device_probe = []
    if probe_path.exists():
        with open(probe_path) as f:
            device_probe = json.load(f)

    # Infer config from records
    config = {
        "tasks": sorted({r["task"] for r in records}),
        "models": sorted({r["model_id"] for r in records}),
        "max_samples_per_task": None,
    }

    report = build_report(
        records=records,
        influenced_cases=influenced,
        judge_results=judge_results,
        device_probe=device_probe,
        judge_validation=judge_validation,
        config=config,
        output_dir=args.output_dir,
    )

    print(f"\nReport written to {args.output_dir / 'REPORT.md'}")
    for model_id, m in report["metrics_by_model"].items():
        if m["n"] > 0:
            print(
                f"  {model_id}: {m['point_estimate']:.1%} faithful "
                f"[{m['ci_lower']:.1%}, {m['ci_upper']:.1%}] (n={m['n']})"
            )


if __name__ == "__main__":
    main()
