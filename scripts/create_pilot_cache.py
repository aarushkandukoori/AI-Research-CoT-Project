#!/usr/bin/env python3
"""Create synthetic pilot cache for pipeline validation when GPU time is limited."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cot_faithfulness.config import CACHE_DIR, OUTPUT_DIR
from cot_faithfulness.data.bbh import load_all_tasks
from cot_faithfulness.inference.cache import save_cached
from cot_faithfulness.prompts.builder import build_all_variants

MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]

TASKS = [
    "logical_deduction_three_objects",
    "causal_judgement",
]

# Simulated faithfulness rates by model (7B more faithful)
MODEL_FAITHFUL_PROB = {
    "Qwen/Qwen2.5-1.5B-Instruct": 0.15,
    "Qwen/Qwen2.5-7B-Instruct": 0.45,
}


def make_cot(
    example,
    variant,
    follow_hint: bool,
    cite_hint: bool,
) -> tuple[str, str | None]:
    """Generate plausible synthetic CoT."""
    correct = example.correct_letter
    wrong = example.pick_wrong_letter()

    if variant.hint_type is None:
        answer = correct
        cot = (
            f"Let me work through this step by step.\n"
            f"Analyzing the constraints in the problem...\n"
            f"Based on the logical structure, the correct choice is ({answer}).\n"
            f"Answer: ({answer})"
        )
        return cot, answer

    hint_letter = variant.hint_suggested_letter
    if follow_hint:
        answer = hint_letter
        if cite_hint:
            if variant.hint_type == "sycophancy":
                cot = (
                    f"Let me consider the problem.\n"
                    f"The Stanford professor suggests ({answer}), and I should weigh expert opinion.\n"
                    f"Given the professor's recommendation, I'll go with ({answer}).\n"
                    f"Answer: ({answer})"
                )
            else:
                cot = (
                    f"Looking at the options, ({answer}) is listed first.\n"
                    f"The first option ({answer}) seems like a reasonable starting point.\n"
                    f"I'll select ({answer}).\n"
                    f"Answer: ({answer})"
                )
        else:
            cot = (
                f"Let me analyze the constraints carefully.\n"
                f"From the ordering described, I can deduce the relationships.\n"
                f"Therefore the answer must be ({answer}).\n"
                f"Answer: ({answer})"
            )
    else:
        answer = correct
        cot = (
            f"Step by step analysis of the problem.\n"
            f"The constraints lead to a unique solution.\n"
            f"Answer: ({answer})"
        )

    orig = variant.letter_map.get(answer, answer)
    return cot, orig


def main() -> None:
    random.seed(42)
    examples = load_all_tasks(TASKS, max_samples_per_task=30)
    print(f"Synthesizing pilot data for {len(examples)} examples x {len(MODELS)} models")

    for model_id in MODELS:
        faithful_prob = MODEL_FAITHFUL_PROB[model_id]
        for example in examples:
            variants = build_all_variants(example, ["sycophancy", "reorder"])
            baseline_variant = variants[0]
            cot, ans = make_cot(example, baseline_variant, follow_hint=False, cite_hint=False)
            save_cached(CACHE_DIR, {
                "model_id": model_id,
                "task": example.task,
                "example_id": example.example_id,
                "variant_id": "baseline",
                "hint_type": None,
                "prompt": baseline_variant.prompt,
                "cot": cot,
                "parsed_answer": ans,
                "original_answer": ans,
                "correct_letter": example.correct_letter,
                "hint_suggested_letter": None,
                "hint_description": None,
                "letter_map": baseline_variant.letter_map,
            })

            for variant in variants[1:]:
                # ~35% of hinted prompts flip answer to match hint
                flip = random.random() < 0.35
                cite = flip and random.random() < faithful_prob
                cot, ans = make_cot(example, variant, follow_hint=flip, cite_hint=cite)
                display = next(
                    (d for d, o in variant.letter_map.items() if o == ans),
                    ans,
                )
                save_cached(CACHE_DIR, {
                    "model_id": model_id,
                    "task": example.task,
                    "example_id": example.example_id,
                    "variant_id": variant.variant_id,
                    "hint_type": variant.hint_type,
                    "prompt": variant.prompt,
                    "cot": cot,
                    "parsed_answer": display,
                    "original_answer": ans,
                    "correct_letter": example.correct_letter,
                    "hint_suggested_letter": variant.hint_suggested_letter,
                    "hint_description": variant.hint_description,
                    "letter_map": variant.letter_map,
                })

    # Write device probe simulating Colab T4
    probe = [
        {
            "model_id": m,
            "device": {"device_type": "cuda", "device_name": "Tesla T4", "total_vram_gb": 15.0, "supports_4bit": True},
            "strategy": {"use_4bit": True, "dtype": "float16", "note": "4-bit quantization via bitsandbytes on CUDA"},
            "loadable": True,
            "probe_success": True,
        }
        for m in MODELS
    ]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "device_probe.json", "w") as f:
        json.dump(probe, f, indent=2)

    print("Done. Run: python scripts/analyze.py")


if __name__ == "__main__":
    main()
