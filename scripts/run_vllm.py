#!/usr/bin/env python3
"""Generate CoT outputs with vLLM on NVIDIA GPUs (the GPU twin of run_mlx.py).

Same items, prompts, chat template, answer parser and record files as run_mlx.py,
so analyze_sampled.py and the judge scripts read the output unchanged. The only
differences are the backend ("vllm"), the weights (bf16 instead of 4-bit) and the
"decoding" block (dtype and vLLM version added). Commands for every model are in
docs/GPU_RUNBOOK.md. Examples:

    # Reasoning model, thinking on (addendum B: T 0.7, top-p 0.95, 4,096 tokens, 1 sample)
    python scripts/run_vllm.py --model Qwen/Qwen3-8B --temperature 0.7 --top-p 0.95 --max-tokens 4096 \\
        --items-file data/substudy_subset_200.json --chat-template-kwargs '{"enable_thinking": true}' \\
        --out-root cache/generations_vllm/think_on

    # CPU smoke test without a GPU (Hugging Face transformers instead of vLLM)
    python scripts/run_vllm.py --backend hf --model Qwen/Qwen2.5-0.5B-Instruct --limit 4 --max-tokens 48 \\
        --out-root /tmp/gpu-smoke

How the generation matches run_mlx.py:
  * The prompt is the builder's text inside the model's own chat template
    (tokenizer.apply_chat_template with --chat-template-kwargs), tokenized with
    add_special_tokens=False and handed to vLLM as token ids, so vLLM cannot add
    a second BOS token.
  * End tokens are collected by the same rule as mlx-lm (generation_config.json,
    else config.json, plus the tokenizer's). The ids can differ from the Mac runs:
    the mlx-community 4-bit Qwen repos ship no generation_config.json, so MLX stopped
    only at <|im_end|> (151645), while the original Qwen repos list <|im_end|> and
    <|endoftext|> (151643), and vLLM stops at the generation_config ids in any case.
    No released greedy MLX record contains <|endoftext|>, so this should not bite.
    The ids used are stored in each record ("decoding" -> "stop_token_ids").
    The end token is dropped and the rest is decoded with the same tokenizer call.
  * No top-k, min-p or repetition penalty: the model's generation_config.json
    sampling defaults are switched off (generation_config="vllm").

Seeds: every prompt gets its own seed, sha256("<seed>:<example_id>:<variant_id>"),
and vLLM gives sample k of that prompt the seed + k. So a sample does not depend on
which other prompts share its batch, and a resumed run gives the same samples
(up to GPU floating point differences).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cot_faithfulness.analysis.thinking import prompt_opens_thinking
from cot_faithfulness.config import BBH_TASKS, CACHE_DIR, MAX_NEW_TOKENS
from cot_faithfulness.data.bbh import load_all_tasks
from cot_faithfulness.inference.mlx_records import build_record, record_path
from cot_faithfulness.prompts.builder import build_all_variants

HINT_TYPES = ["sycophancy", "reorder", "placebo"]
# Settings that must match before we resume into an existing run folder.
CHECKED_KEYS = ["backend", "dtype", "temperature", "top_p", "max_tokens", "chat_template_kwargs", "seed",
                "n_samples"]
GREEDY_ROOT = CACHE_DIR / "generations_mlx"  # the released greedy MLX runs; never written to from here
SEED_RULE = "prompt seed = int(sha256('<seed>:<example_id>:<variant_id>')[:8], 16); sample k uses prompt seed + k"


def prompt_seed(seed: int, example_id: str, variant_id: str) -> int:
    """Seed of one prompt (vLLM adds k for sample k)."""
    return int(hashlib.sha256(f"{seed}:{example_id}:{variant_id}".encode()).hexdigest()[:8], 16)


def stop_token_ids(model_id: str, tokenizer) -> list[int]:
    """End tokens, collected the way mlx-lm does: generation_config.json (else config.json) plus the tokenizer's."""
    from transformers import AutoConfig, GenerationConfig

    try:
        eos = GenerationConfig.from_pretrained(model_id).eos_token_id
    except OSError:
        eos = None
    if eos is None:
        eos = AutoConfig.from_pretrained(model_id).eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos or [])
    if tokenizer.eos_token_id is not None:
        eos.append(tokenizer.eos_token_id)
    return sorted(set(eos))


def make_record(*, model_id, example, variant, token_ids, finish_reason, stop_ids, decode,
                opens_think, sample_index, decoding) -> dict:
    """The record run_mlx.py would write for these tokens.

    run_mlx.py never keeps the end token, so we drop it here, decode the rest with
    the same tokenizer call and count the same tokens. Only "backend" and the
    "decoding" block differ from an MLX record.
    """
    token_ids = list(token_ids)
    if finish_reason == "stop" and token_ids and token_ids[-1] in stop_ids:
        token_ids = token_ids[:-1]
    return build_record(
        model_id=model_id, example=example, variant=variant, cot=decode(token_ids).strip(),
        n_new_tokens=len(token_ids), finish_reason=finish_reason, prompt_opens_think=opens_think,
        extra={"backend": decoding["backend"], "sample_index": sample_index, "decoding": decoding},
    )


def load_vllm(args, max_model_len: int, stop_ids: list[int]):
    """Return generate(prompt_ids, seeds) -> per prompt, a list of (token_ids, finish_reason), one per sample."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max_model_len,
        seed=args.seed,
        generation_config="vllm",  # ignore the model card's sampling defaults (top-k 20 etc.)
    )
    sampled = args.temperature > 0

    def generate(prompt_ids, seeds):
        params = [SamplingParams(
            n=args.n_samples,
            temperature=args.temperature,  # 0 = greedy
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            seed=s if sampled else None,
            stop_token_ids=stop_ids,
            skip_special_tokens=False,  # we decode ourselves anyway
        ) for s in seeds]
        outs = llm.generate([{"prompt_token_ids": p} for p in prompt_ids], params, use_tqdm=False)
        return [[(list(c.token_ids), c.finish_reason) for c in sorted(o.outputs, key=lambda c: c.index)]
                for o in outs]

    return generate


def load_hf(args, stop_ids: list[int], pad_id: int):
    """Hugging Face transformers on CPU, one prompt at a time. For smoke tests without a GPU."""
    import torch
    from transformers import AutoModelForCausalLM

    dtype = "auto" if args.dtype == "auto" else getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to("cpu").eval()
    sampled = args.temperature > 0

    def generate(prompt_ids, seeds):
        results = []
        for ids, seed in zip(prompt_ids, seeds):
            if sampled:
                torch.manual_seed(seed)
            # Every sampling knob is set here so the model's generation_config.json defaults
            # (Qwen2.5: top-k 20, repetition penalty 1.05) do not leak in.
            knobs = dict(temperature=args.temperature, top_p=args.top_p, top_k=0) if sampled else \
                dict(temperature=None, top_p=None, top_k=None)
            with torch.no_grad():
                out = model.generate(
                    torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                    max_new_tokens=args.max_tokens, do_sample=sampled, num_return_sequences=args.n_samples,
                    repetition_penalty=1.0, eos_token_id=stop_ids, pad_token_id=pad_id, **knobs,
                )
            samples = []
            for row in out[:, len(ids):].tolist():
                end = next((i for i, t in enumerate(row) if t in stop_ids), None)
                samples.append((row, "length") if end is None else (row[: end + 1], "stop"))
            results.append(samples)
        return results

    return generate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Hugging Face repo, e.g. Qwen/Qwen3-8B")
    parser.add_argument("--max-samples", type=int, default=250, help="items per BBH task (preregistered: 250)")
    parser.add_argument("--max-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--limit", type=int, default=None,
                        help="only run the first N prompts (each gives --n-samples generations; smoke test)")
    parser.add_argument("--temperature", type=float, default=0.0, help="0 = greedy (default)")
    parser.add_argument("--top-p", type=float, default=1.0, help="nucleus sampling; 1.0 = off")
    parser.add_argument("--n-samples", type=int, default=1, help="samples per prompt (needs --temperature > 0)")
    parser.add_argument("--seed", type=int, default=0, help="base seed; each prompt's seed is derived from it")
    parser.add_argument("--items-file", type=Path, default=None, help="JSON list of example_ids to run")
    parser.add_argument("--out-root", type=Path, default=CACHE_DIR / "generations_vllm",
                        help="folder that gets one subfolder per model (default: cache/generations_vllm)")
    parser.add_argument("--chat-template-kwargs", type=json.loads, default=None,
                        help='JSON passed to apply_chat_template, e.g. \'{"enable_thinking": true}\'')
    parser.add_argument("--backend", choices=["vllm", "hf"], default="vllm", help="hf = transformers on CPU")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32", "auto"])
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="GPUs per model (vllm)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90, help="share of GPU memory vLLM may use")
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="prompt + output length vLLM reserves (default: longest prompt + --max-tokens)")
    parser.add_argument("--chunk-size", type=int, default=512,
                        help="prompts per vLLM call; records are written after each chunk")
    args = parser.parse_args()

    sampled = args.temperature > 0
    template_kwargs = args.chat_template_kwargs or {}
    if args.n_samples < 1 or (args.n_samples > 1 and not sampled):
        parser.error("--n-samples > 1 needs --temperature > 0")
    if args.out_root.resolve() == GREEDY_ROOT.resolve():
        parser.error("cache/generations_mlx holds the released greedy runs; pick another --out-root")
    out_dir = args.out_root / args.model.replace("/", "__")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Settings stored in every record and in run_config.json, checked before resuming
    # (run_config.json is written once the end tokens are known, just before generating).
    decoding = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed if sampled else None,
        "n_samples": args.n_samples,
        "max_tokens": args.max_tokens,
        "chat_template_kwargs": template_kwargs,
        "backend": args.backend,
        "dtype": args.dtype,
        # hf (smoke tests only) seeds torch with the prompt seed and draws all samples in one call
        "seed_rule": (SEED_RULE if args.backend == "vllm" else "torch.manual_seed(prompt seed)") if sampled else None,
        "transformers_version": version("transformers"),
        "torch_version": version("torch"),
    }
    if args.backend == "vllm":
        decoding["vllm_version"] = version("vllm")
    config_path = out_dir / "run_config.json"
    if config_path.exists():
        old = json.load(open(config_path))
        if any(old.get(k) != decoding[k] for k in CHECKED_KEYS):
            sys.exit(f"{out_dir} was made with other settings ({config_path}); use another --out-root")

    examples = load_all_tasks(BBH_TASKS, args.max_samples)
    if args.items_file:
        wanted = set(json.load(open(args.items_file)))
        missing = wanted - {ex.example_id for ex in examples}
        if missing:
            sys.exit(f"{len(missing)} ids in {args.items_file} are not BBH items, e.g. {sorted(missing)[:3]}")
        examples = [ex for ex in examples if ex.example_id in wanted]
    print(f"Loaded {len(examples)} BBH items", flush=True)

    # One job per (item, condition) prompt that still misses at least one sample file.
    sample_ids = list(range(args.n_samples)) if sampled else [None]  # None = greedy file name
    jobs = []
    for ex in examples:
        for variant in build_all_variants(ex, HINT_TYPES):
            todo = [k for k in sample_ids if not record_path(out_dir, ex.example_id, variant.variant_id, k).exists()]
            if todo:
                jobs.append((ex, variant, todo))
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"{len(jobs)} prompts to run ({args.n_samples} sample(s) each)", flush=True)
    if not jobs:
        return

    # Render prompts exactly like run_mlx.py: the model's chat template, then token ids.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    stop_ids = stop_token_ids(args.model, tokenizer)
    decoding["stop_token_ids"] = stop_ids
    if not config_path.exists():
        json.dump(decoding, open(config_path, "w"), indent=2)
    prompt_ids, opens_think, seeds = [], [], []
    for ex, variant, _ in jobs:
        messages = [{"role": "user", "content": variant.prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **template_kwargs)
        prompt_ids.append(tokenizer.encode(text, add_special_tokens=False))
        opens_think.append(prompt_opens_thinking(text))
        seeds.append(prompt_seed(args.seed, ex.example_id, variant.variant_id))
    print(f"end tokens {stop_ids}; prompts open thinking: {sum(opens_think)}/{len(jobs)}", flush=True)

    t_load = time.time()
    if args.backend == "vllm":
        max_model_len = args.max_model_len or max(map(len, prompt_ids)) + args.max_tokens
        print(f"max_model_len {max_model_len}", flush=True)
        generate = load_vllm(args, max_model_len, stop_ids)
    else:
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else stop_ids[0]
        generate = load_hf(args, stop_ids, pad_id)
    print(f"model loaded in {time.time() - t_load:.1f} s", flush=True)

    done, n_tokens, start = 0, 0, time.time()
    for lo in range(0, len(jobs), args.chunk_size):
        hi = min(lo + args.chunk_size, len(jobs))
        results = generate(prompt_ids[lo:hi], seeds[lo:hi])
        for i, samples in zip(range(lo, hi), results):
            ex, variant, todo = jobs[i]
            for k in todo:  # only the files that are missing
                token_ids, finish_reason = samples[k or 0]
                record = make_record(
                    model_id=args.model, example=ex, variant=variant, token_ids=token_ids,
                    finish_reason=finish_reason, stop_ids=stop_ids, decode=tokenizer.decode,
                    opens_think=opens_think[i], sample_index=k or 0, decoding=decoding,
                )
                with open(record_path(out_dir, ex.example_id, variant.variant_id, k), "w") as f:
                    json.dump(record, f, indent=2)
                n_tokens += record["n_new_tokens"]
                done += 1
        rate = hi / (time.time() - start)
        print(f"{hi}/{len(jobs)} prompts done, {done} records, {n_tokens / (time.time() - start):.0f} tok/s, "
              f"ETA {(len(jobs) - hi) / rate / 60:.0f} min", flush=True)
    elapsed = time.time() - start
    print(f"finished {done} generations, {n_tokens} new tokens in {elapsed:.1f} s: "
          f"{done / elapsed:.3f} gen/s, {n_tokens / elapsed:.1f} tok/s", flush=True)


if __name__ == "__main__":
    main()
