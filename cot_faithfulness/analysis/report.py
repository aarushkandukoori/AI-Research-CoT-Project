"""Generate experiment report with metrics and limitations."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from cot_faithfulness.analysis.bootstrap import faithfulness_summary
from cot_faithfulness.config import MIN_INFLUENCED_FOR_CI


def _case_to_dict(case) -> dict:
    if isinstance(case, dict):
        return case
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


def build_report(
    records: list[dict],
    influenced_cases: list,
    judge_results: list[dict],
    device_probe: list[dict],
    judge_validation: dict | None,
    config: dict,
    output_dir: Path,
) -> dict:
    """Assemble full experiment report."""
    influenced_dicts = [_case_to_dict(c) for c in influenced_cases]

    # Per-model, per-hint-type metrics
    cells = {}
    for jr in judge_results:
        key = (jr["model_id"], jr["hint_type"])
        cells.setdefault(key, []).append(jr)

    metrics = {}
    for (model_id, hint_type), judges in cells.items():
        faithful = [j["classification"] == "FAITHFUL" for j in judges]
        summary = faithfulness_summary(faithful, min_n=MIN_INFLUENCED_FOR_CI)
        metrics[f"{model_id}__{hint_type}"] = {
            "model_id": model_id,
            "hint_type": hint_type,
            "n_influenced": summary["n"],
            "n_faithful": summary["faithful_count"],
            "n_unfaithful_or_ambiguous": summary["unfaithful_count"],
            **{k: summary[k] for k in ["point_estimate", "ci_lower", "ci_upper", "too_small"]},
        }

    # Overall per model
    model_metrics = {}
    for model_id in {r["model_id"] for r in records}:
        model_judges = [j for j in judge_results if j["model_id"] == model_id]
        faithful = [j["classification"] == "FAITHFUL" for j in model_judges]
        model_metrics[model_id] = faithfulness_summary(faithful, min_n=MIN_INFLUENCED_FOR_CI)

    # Sample sizes
    sample_sizes = {}
    for rec in records:
        key = (rec["model_id"], rec["task"], rec.get("hint_type") or "baseline")
        sample_sizes[str(key)] = sample_sizes.get(str(key), 0) + 1

    n_influenced_total = len(influenced_dicts)
    influence_rate = n_influenced_total / max(
        len({(r["model_id"], r["example_id"]) for r in records if r.get("hint_type")}), 1
    )

    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "config": config,
        "device_probe": device_probe,
        "sample_sizes": sample_sizes,
        "total_generations": len(records),
        "total_influenced_cases": n_influenced_total,
        "influence_rate": influence_rate,
        "metrics_by_cell": metrics,
        "metrics_by_model": {
            k: {kk: vv for kk, vv in v.items() if kk != "too_small" or True}
            for k, v in model_metrics.items()
        },
        "judge_validation": judge_validation,
        "limitations": _build_limitations(config, judge_validation, metrics, device_probe),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    md_path = output_dir / "REPORT.md"
    with open(md_path, "w") as f:
        f.write(_render_markdown(report, influenced_dicts))

    return report


def _build_limitations(
    config: dict,
    judge_validation: dict | None,
    metrics: dict,
    device_probe: list[dict],
) -> list[str]:
    limitations = [
        "Dataset contamination: BBH tasks are public and likely appeared in model pretraining "
        "corpora, so absolute accuracy and some faithfulness patterns may not generalize.",
        "Small sample sizes: this is a small-scale replication; cells with <20 hint-influenced "
        "cases are flagged as too noisy for reliable CI estimation.",
        "Judge reliability ceiling: LLM-as-judge agreement with human labels is imperfect; "
        "faithfulness rates inherit this measurement error.",
        "Hint surface form: we test only sycophancy (authority) and option-reordering hints; "
        "other bias mechanisms (e.g., token priors, subtle framing) are not covered.",
        "Parsing fragility: answer extraction from free-form CoT may misclassify influence "
        "if the model uses non-standard answer formatting.",
    ]

    if judge_validation and judge_validation.get("n", 0) >= 10:
        rate = judge_validation.get("exact_agreement_rate")
        if rate is not None and rate < 0.8:
            limitations.append(
                f"Judge-human agreement ({rate:.0%}) is below 80%; interpret automated "
                "faithfulness labels cautiously."
            )

    noisy_cells = [k for k, v in metrics.items() if v.get("too_small")]
    if noisy_cells:
        limitations.append(
            f"Noisy cells (n<20 influenced): {', '.join(noisy_cells)}. "
            "Point estimates shown but CIs should not be over-interpreted."
        )

    for probe in device_probe:
        if not probe.get("probe_success"):
            limitations.append(
                f"Model {probe['model_id']} failed GPU probe: {probe.get('error', 'unknown')}. "
                "Results may be from fallback hardware/settings."
            )

    max_samples = config.get("max_samples_per_task")
    if max_samples and max_samples < 100:
        limitations.append(
            f"Only {max_samples} samples per BBH subtask used; larger samples needed for "
            "stable size-comparison conclusions."
        )

    return limitations


def _render_markdown(report: dict, influenced: list[dict]) -> str:
    lines = [
        "# CoT Faithfulness Experiment Report",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Summary",
        "",
        f"- Total generations cached: {report['total_generations']}",
        f"- Hint-influenced answer flips: {report['total_influenced_cases']}",
        f"- Influence rate (among hinted prompts): {report['influence_rate']:.1%}",
        "",
        "## Faithfulness by Model and Hint Type",
        "",
        "| Model | Hint | n influenced | Faithful | Rate [95% CI] | Noisy? |",
        "|-------|------|-------------|----------|---------------|--------|",
    ]

    for key, m in report["metrics_by_cell"].items():
        rate = m.get("point_estimate")
        if rate is not None:
            ci = f"{rate:.1%} [{m['ci_lower']:.1%}, {m['ci_upper']:.1%}]"
        else:
            ci = "N/A"
        noisy = "⚠️ yes" if m.get("too_small") else "no"
        lines.append(
            f"| {m['model_id']} | {m['hint_type']} | {m['n_influenced']} | "
            f"{m['n_faithful']} | {ci} | {noisy} |"
        )

    lines.extend(["", "## Model-Level Faithfulness", ""])
    for model_id, m in report["metrics_by_model"].items():
        if m["n"] == 0:
            lines.append(f"- **{model_id}**: no influenced cases")
            continue
        lines.append(
            f"- **{model_id}**: {m['point_estimate']:.1%} faithful "
            f"[{m['ci_lower']:.1%}, {m['ci_upper']:.1%}] "
            f"(n={m['n']}, faithful={m['faithful_count']})"
        )

    if report.get("judge_validation"):
        jv = report["judge_validation"]
        lines.extend([
            "",
            "## Judge Validation (Human Labels)",
            "",
            f"- Labeled cases: {jv.get('n', 0)}",
            f"- Exact agreement: {jv.get('exact_agreement_rate', 'N/A')}",
            f"- Relaxed agreement (incl. AMBIGUOUS): {jv.get('relaxed_agreement_rate', 'N/A')}",
        ])

    lines.extend(["", "## Hardware Probe", ""])
    for probe in report.get("device_probe", []):
        status = "✓" if probe.get("probe_success") else "✗"
        lines.append(f"- {status} {probe['model_id']}: {probe.get('strategy', {}).get('note', '')}")

    lines.extend(["", "## Limitations", ""])
    for lim in report.get("limitations", []):
        lines.append(f"- {lim}")

    lines.extend(["", "## Example Influenced Cases", ""])
    for case in influenced[:5]:
        lines.append(f"### {case['example_id']} ({case['hint_type']})")
        lines.append(f"- Baseline → Hinted: ({case['baseline_answer']}) → ({case['hinted_answer']})")
        lines.append(f"- Hint suggested: ({case['hint_suggested']}), Correct: ({case['correct_letter']})")
        lines.append(f"- CoT excerpt: {case['hinted_cot'][:300]}...")
        lines.append("")

    return "\n".join(lines)
