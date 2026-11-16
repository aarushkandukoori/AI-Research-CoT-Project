#!/usr/bin/env python3
"""Quick health check of one run folder (from run_vllm.py or run_mlx.py).

    python scripts/check_run.py cache/generations_vllm/think_on/Qwen__Qwen3-8B \\
        --items-file data/substudy_subset_200.json

Prints, per condition: record count, share with no parsed answer, share that hit
the token cap, share whose thinking block never closed, and mean new tokens.
Exits with code 1 if any (item, condition) is missing a sample, so it can gate a
tar-and-send step. The expected item list is --items-file (else every item seen
in the folder); the expected samples per prompt come from run_config.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cot_faithfulness.analysis.thinking import split_thinking
from cot_faithfulness.inference.mlx_records import parse_record_name

CONDITIONS = ["baseline", "sycophancy", "reorder", "placebo"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path, help="one model folder, e.g. cache/generations_vllm/think_on/Qwen__Qwen3-8B")
    parser.add_argument("--items-file", type=Path, default=None, help="JSON list of example_ids the run should cover")
    args = parser.parse_args()

    config_path = args.run_dir / "run_config.json"
    n_samples = json.load(open(config_path)).get("n_samples", 1) if config_path.exists() else 1
    rows = defaultdict(list)  # condition -> one dict per record
    seen = defaultdict(set)  # (example_id, condition) -> sample indices on disk
    for path in sorted(args.run_dir.glob("*__*.json")):
        ex_id, _, k = parse_record_name(path.name)
        rec = json.load(open(path))
        cond = rec["hint_type"] or "baseline"
        parts = split_thinking(rec["cot"], rec.get("prompt_opens_thinking", False))
        rows[cond].append({
            "unparsed": rec["parsed_answer"] is None,
            "capped": rec["finish_reason"] == "length",
            "unclosed": parts["has_thinking"] and not parts["thinking_closed"],
            "tokens": rec["n_new_tokens"],
        })
        seen[ex_id, cond].add(k)

    items = set(json.load(open(args.items_file))) if args.items_file else {ex for ex, _ in seen}
    expected = len(items) * n_samples
    print(f"{args.run_dir}: {len(items)} items x {n_samples} sample(s) = {expected} records expected per condition")
    print(f"{'condition':12s}{'records':>9s}{'unparsed':>10s}{'at cap':>9s}{'unclosed':>10s}{'mean tok':>10s}")
    complete = True
    for cond in CONDITIONS:
        r = rows.get(cond, [])
        n = len(r) or 1
        share = lambda key: sum(x[key] for x in r) / n  # noqa: E731
        print(f"{cond:12s}{len(r):9d}{share('unparsed'):10.1%}{share('capped'):9.1%}{share('unclosed'):10.1%}"
              f"{sum(x['tokens'] for x in r) / n:10.0f}")
        missing = [ex for ex in items if len(seen.get((ex, cond), ())) < n_samples]
        if missing:
            complete = False
            print(f"  missing samples for {len(missing)} items, e.g. {sorted(missing)[:3]}")
    if not complete:
        sys.exit("INCOMPLETE: rerun the same command; it resumes and fills the gaps")
    print("complete")


if __name__ == "__main__":
    main()
