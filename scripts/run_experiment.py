#!/usr/bin/env python3
"""Run CoT faithfulness experiment: generate and cache model outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

# Allow running as script from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cot_faithfulness.analysis.influence import parse_answer, map_to_original_letter
from cot_faithfulness.config import ExperimentConfig, MAX_NEW_TOKENS, TEMPERATURE
from cot_faithfulness.data.bbh import load_all_tasks
from cot_faithfulness.inference.cache import load_cached, save_cached
from cot_faithfulness.inference.model_runner import ModelRunner, probe_model_loadable
from cot_faithfulness.prompts.builder import build_all_variants


def run_generation_pass(config: ExperimentConfig) -> list[dict]:
    """Generate and cache all prompt variants for configured models."""
    device_probes = []
    loadable_models = []

    for model_id in config.models:
        probe = probe_model_loadable(model_id)
        device_probes.append(probe)
        print(f"Probe {model_id}: loadable={probe['loadable']}, note={probe.get('strategy', {}).get('note')}")
        if probe["loadable"]:
            loadable_models.append(model_id)

    probe_path = config.output_dir / "device_probe.json"
    with open(probe_path, "w") as f:
        json.dump(device_probes, f, indent=2)

    if config.probe_only:
        print(f"Probe-only mode. Results saved to {probe_path}")
        return []

    if not loadable_models:
        raise RuntimeError("No models loadable on current hardware. See device_probe.json")

    examples = load_all_tasks(config.tasks, config.max_samples_per_task)
    print(f"Loaded {len(examples)} BBH MCQ examples across {len(config.tasks)} tasks")

    all_records = []
    for model_id in loadable_models:
        print(f"\n=== Loading {model_id} ===")
        runner = ModelRunner(model_id)
        try:
            for example in tqdm(examples, desc=model_id):
                variants = build_all_variants(example, config.hint_types)
                for variant in variants:
                    if not config.force_regenerate:
                        cached = load_cached(
                            config.cache_dir, model_id, example.example_id, variant.variant_id
                        )
                        if cached:
                            all_records.append(cached)
                            continue

                    cot = runner.generate(
                        variant.prompt,
                        max_new_tokens=MAX_NEW_TOKENS,
                        temperature=TEMPERATURE,
                    )
                    display_answer = parse_answer(cot, list(variant.display_options.keys()))
                    original_answer = map_to_original_letter(display_answer, variant.letter_map)

                    record = {
                        "model_id": model_id,
                        "task": example.task,
                        "example_id": example.example_id,
                        "variant_id": variant.variant_id,
                        "hint_type": variant.hint_type,
                        "prompt": variant.prompt,
                        "cot": cot,
                        "parsed_answer": display_answer,
                        "original_answer": original_answer,
                        "correct_letter": example.correct_letter,
                        "hint_suggested_letter": variant.hint_suggested_letter,
                        "hint_description": variant.hint_description,
                        "letter_map": variant.letter_map,
                    }
                    save_cached(config.cache_dir, record)
                    all_records.append(record)
        finally:
            runner.unload()

    manifest_path = config.output_dir / "generations_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({"n_records": len(all_records), "models": loadable_models}, f, indent=2)

    print(f"\nDone. {len(all_records)} records. Manifest: {manifest_path}")
    return all_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CoT faithfulness generation pass")
    parser.add_argument("--max-samples", type=int, default=75, help="Max samples per BBH task")
    parser.add_argument("--models", nargs="+", default=None, help="Model IDs to run")
    parser.add_argument("--tasks", nargs="+", default=None, help="BBH task names")
    parser.add_argument("--force", action="store_true", help="Regenerate even if cached")
    parser.add_argument("--probe-only", action="store_true", help="Only probe GPU/model loadability")
    args = parser.parse_args()

    config = ExperimentConfig(
        max_samples_per_task=args.max_samples,
        force_regenerate=args.force,
        probe_only=args.probe_only,
    )
    if args.models:
        config.models = args.models
    if args.tasks:
        config.tasks = args.tasks

    run_generation_pass(config)


if __name__ == "__main__":
    main()
