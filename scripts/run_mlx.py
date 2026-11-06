#!/usr/bin/env python3
"""Generate CoT outputs with MLX on Apple Silicon (batched, greedy, resumable).

Same prompts, hints and answer parser as run_experiment.py, but uses mlx-lm
instead of transformers so 1.5B-7B models run fast on an M-series Mac.

Example:
    python scripts/run_mlx.py --model mlx-community/Qwen2.5-7B-Instruct-4bit
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import BatchGenerator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cot_faithfulness.analysis.influence import map_to_original_letter, parse_answer
from cot_faithfulness.config import BBH_TASKS, CACHE_DIR, MAX_NEW_TOKENS
from cot_faithfulness.data.bbh import load_all_tasks
from cot_faithfulness.prompts.builder import build_all_variants

HINT_TYPES = ["sycophancy", "reorder", "placebo"]


def record_path(out_dir: Path, example_id: str, variant_id: str) -> Path:
    return out_dir / f"{example_id}__{variant_id}.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="MLX model repo, e.g. mlx-community/Qwen2.5-7B-Instruct-4bit")
    parser.add_argument("--max-samples", type=int, default=250, help="items per BBH task (preregistered: 250)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--limit", type=int, default=None, help="only run the first N prompts (smoke test)")
    args = parser.parse_args()

    out_dir = CACHE_DIR / "generations_mlx" / args.model.replace("/", "__")
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = load_all_tasks(BBH_TASKS, args.max_samples)
    print(f"Loaded {len(examples)} BBH items", flush=True)

    # One job per (item, condition); skip anything already on disk.
    jobs = []
    for ex in examples:
        for variant in build_all_variants(ex, HINT_TYPES):
            if not record_path(out_dir, ex.example_id, variant.variant_id).exists():
                jobs.append((ex, variant))
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"{len(jobs)} generations to run", flush=True)
    if not jobs:
        return

    model, tokenizer = load(args.model)
    prompts = []
    for _, variant in jobs:
        messages = [{"role": "user", "content": variant.prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(tokenizer.encode(text, add_special_tokens=False))

    # Greedy decoding (default sampler) with continuous batching.
    gen = BatchGenerator(
        model,
        stop_tokens=[[t] for t in tokenizer.eos_token_ids],
        completion_batch_size=args.batch_size,
        prefill_batch_size=min(8, args.batch_size),
    )
    uids = gen.insert(prompts, [args.max_tokens] * len(prompts))
    job_of = dict(zip(uids, jobs))
    tokens = {uid: [] for uid in uids}

    done, start = 0, time.time()
    while responses := gen.next_generated():
        for r in responses:
            if r.finish_reason != "stop":
                tokens[r.uid].append(r.token)
            if r.finish_reason is None:
                continue
            ex, variant = job_of[r.uid]
            cot = tokenizer.decode(tokens[r.uid]).strip()
            display_answer = parse_answer(cot, list(variant.display_options.keys()))
            record = {
                "model_id": args.model,
                "backend": "mlx",
                "task": ex.task,
                "example_id": ex.example_id,
                "variant_id": variant.variant_id,
                "hint_type": variant.hint_type,
                "prompt": variant.prompt,
                "cot": cot,
                "n_new_tokens": len(tokens[r.uid]),
                "finish_reason": r.finish_reason,  # "length" means it hit max_tokens
                "parsed_answer": display_answer,
                "original_answer": map_to_original_letter(display_answer, variant.letter_map),
                "correct_letter": ex.correct_letter,
                "hint_suggested_letter": variant.hint_suggested_letter,
                "hint_description": variant.hint_description,
                "letter_map": variant.letter_map,
            }
            with open(record_path(out_dir, ex.example_id, variant.variant_id), "w") as f:
                json.dump(record, f, indent=2)
            del tokens[r.uid]
            done += 1
            if done % 50 == 0 or done == len(jobs):
                rate = done / (time.time() - start)
                eta = (len(jobs) - done) / rate / 60
                print(f"{done}/{len(jobs)} done, {rate:.2f} gen/s, ETA {eta:.0f} min, "
                      f"peak mem {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)
    gen.close()


if __name__ == "__main__":
    main()
