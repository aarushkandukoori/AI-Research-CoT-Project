#!/usr/bin/env python3
"""Generate CoT outputs with MLX on Apple Silicon (batched, resumable).

Same prompts, hints and answer parser as run_experiment.py, but uses mlx-lm
instead of transformers so 1.5B-7B models run fast on an M-series Mac.

Default (no new flags) = the November 2026 greedy run, unchanged:
    python scripts/run_mlx.py --model mlx-community/Qwen2.5-7B-Instruct-4bit

Stochastic sub-study (T = 0.7, 5 samples per prompt, 200-item subset):
    python scripts/run_mlx.py --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \\
        --temperature 0.7 --n-samples 5 --seed 0 --items-file data/substudy_subset_200.json

Reasoning model with its thinking switched on:
    python scripts/run_mlx.py --model <mlx model id> --max-tokens 4096 --batch-size 8 \\
        --chat-template-kwargs '{"enable_thinking": true}' --out-root cache/generations_reasoning

Sampling notes:
  * Sample k of a prompt is saved as <example_id>__<variant_id>__s<k>.json and the
    decoding settings are stored in the record ("decoding").
  * --seed seeds MLX's global random state once. With continuous batching the
    random draws a sequence gets depend on which other sequences share its batch,
    so a rerun with the same seed reproduces samples only if the same jobs run in
    the same order with the same batch size. Resuming a half-finished run gives
    different (still valid) samples for the remaining jobs.
  * A non-default run writes run_config.json next to its records and refuses to
    resume into a folder made with different settings.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx_lm
from mlx_lm import load
from mlx_lm.generate import BatchGenerator
from mlx_lm.sample_utils import make_sampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cot_faithfulness.analysis.thinking import prompt_opens_thinking
from cot_faithfulness.config import BBH_TASKS, CACHE_DIR, MAX_NEW_TOKENS
from cot_faithfulness.data.bbh import load_all_tasks
from cot_faithfulness.inference.mlx_records import build_record, record_path
from cot_faithfulness.prompts.builder import build_all_variants

HINT_TYPES = ["sycophancy", "reorder", "placebo"]
GREEDY_ROOT = CACHE_DIR / "generations_mlx"  # the released November runs; never touched by other settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="MLX model repo, e.g. mlx-community/Qwen2.5-7B-Instruct-4bit")
    parser.add_argument("--max-samples", type=int, default=250, help="items per BBH task (preregistered: 250)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--limit", type=int, default=None, help="only run the first N generations (smoke test)")
    parser.add_argument("--temperature", type=float, default=0.0, help="0 = greedy (default)")
    parser.add_argument("--top-p", type=float, default=1.0, help="nucleus sampling; 1.0 = off")
    parser.add_argument("--n-samples", type=int, default=1, help="samples per prompt (needs --temperature > 0)")
    parser.add_argument("--seed", type=int, default=0, help="MLX random seed for sampled runs")
    parser.add_argument("--items-file", type=Path, default=None, help="JSON list of example_ids to run")
    parser.add_argument("--out-root", type=Path, default=None,
                        help="folder that gets one subfolder per model (default: cache/generations_mlx for plain "
                             "greedy, cache/generations_mlx_sampled for sampling, cache/generations_reasoning "
                             "when --chat-template-kwargs is set)")
    parser.add_argument("--chat-template-kwargs", type=json.loads, default=None,
                        help='JSON passed to apply_chat_template, e.g. \'{"enable_thinking": true}\'')
    args = parser.parse_args()

    sampled = args.temperature > 0
    template_kwargs = args.chat_template_kwargs or {}
    if args.n_samples < 1 or (args.n_samples > 1 and not sampled):
        parser.error("--n-samples > 1 needs --temperature > 0")
    default_run = not sampled and not template_kwargs  # the original greedy setting

    out_root = args.out_root
    if out_root is None:
        out_root = GREEDY_ROOT if default_run else (
            CACHE_DIR / "generations_mlx_sampled" if sampled else CACHE_DIR / "generations_reasoning")
    if not default_run and out_root.resolve() == GREEDY_ROOT.resolve():
        parser.error("cache/generations_mlx holds the released greedy runs; pick another --out-root")
    out_dir = out_root / args.model.replace("/", "__")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Settings stored in every non-default record and checked before resuming.
    decoding = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed if sampled else None,
        "n_samples": args.n_samples,
        "max_tokens": args.max_tokens,
        "chat_template_kwargs": template_kwargs,
        "mlx_lm_version": mlx_lm.__version__,
    }
    if not default_run:
        config_path = out_dir / "run_config.json"
        if config_path.exists():
            old = json.load(open(config_path))
            keys = ["temperature", "top_p", "max_tokens", "chat_template_kwargs"]
            if any(old.get(k) != decoding[k] for k in keys):
                sys.exit(f"{out_dir} was made with other settings ({config_path}); use another --out-root")
        else:
            json.dump(decoding, open(config_path, "w"), indent=2)

    examples = load_all_tasks(BBH_TASKS, args.max_samples)
    if args.items_file:
        wanted = set(json.load(open(args.items_file)))
        missing = wanted - {ex.example_id for ex in examples}
        if missing:
            sys.exit(f"{len(missing)} ids in {args.items_file} are not BBH items, e.g. {sorted(missing)[:3]}")
        examples = [ex for ex in examples if ex.example_id in wanted]
    print(f"Loaded {len(examples)} BBH items", flush=True)

    # One job per (item, condition, sample); skip anything already on disk.
    sample_ids = list(range(args.n_samples)) if sampled else [None]  # None = greedy file name
    jobs = []
    for ex in examples:
        for variant in build_all_variants(ex, HINT_TYPES):
            for k in sample_ids:
                if not record_path(out_dir, ex.example_id, variant.variant_id, k).exists():
                    jobs.append((ex, variant, k))
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"{len(jobs)} generations to run", flush=True)
    if not jobs:
        return

    t_load = time.time()
    model, tokenizer = load(args.model)
    print(f"model loaded in {time.time() - t_load:.1f} s", flush=True)
    prompts, opens_think = [], []
    for _, variant, _ in jobs:
        messages = [{"role": "user", "content": variant.prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **template_kwargs)
        prompts.append(tokenizer.encode(text, add_special_tokens=False))
        opens_think.append(prompt_opens_thinking(text))

    # Greedy decoding (default sampler) or temperature sampling, with continuous batching.
    sampler = None
    if sampled:
        mx.random.seed(args.seed)
        sampler = make_sampler(temp=args.temperature, top_p=args.top_p)
    gen = BatchGenerator(
        model,
        stop_tokens=[[t] for t in tokenizer.eos_token_ids],
        sampler=sampler,  # None = greedy, same as before
        completion_batch_size=args.batch_size,
        prefill_batch_size=min(8, args.batch_size),
    )
    uids = gen.insert(prompts, [args.max_tokens] * len(prompts))
    job_of = dict(zip(uids, range(len(jobs))))
    tokens = {uid: [] for uid in uids}

    done, n_tokens, start = 0, 0, time.time()
    while responses := gen.next_generated():
        for r in responses:
            if r.finish_reason != "stop":
                tokens[r.uid].append(r.token)
            if r.finish_reason is None:
                continue
            i = job_of[r.uid]
            ex, variant, k = jobs[i]
            cot = tokenizer.decode(tokens[r.uid]).strip()
            extra = None if default_run else {"sample_index": k or 0, "decoding": decoding}
            record = build_record(
                model_id=args.model, example=ex, variant=variant, cot=cot,
                n_new_tokens=len(tokens[r.uid]), finish_reason=r.finish_reason,
                prompt_opens_think=opens_think[i], extra=extra,
            )
            with open(record_path(out_dir, ex.example_id, variant.variant_id, k), "w") as f:
                json.dump(record, f, indent=2)
            n_tokens += len(tokens[r.uid])
            del tokens[r.uid]
            done += 1
            if done % 50 == 0 or done == len(jobs):
                rate = done / (time.time() - start)
                eta = (len(jobs) - done) / rate / 60
                print(f"{done}/{len(jobs)} done, {rate:.2f} gen/s, ETA {eta:.0f} min, "
                      f"peak mem {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)
    gen.close()
    elapsed = time.time() - start
    print(f"finished {done} generations, {n_tokens} new tokens in {elapsed:.1f} s: "
          f"{done / elapsed:.3f} gen/s, {n_tokens / elapsed:.1f} tok/s", flush=True)


if __name__ == "__main__":
    main()
