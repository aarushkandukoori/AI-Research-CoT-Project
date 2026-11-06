#!/usr/bin/env python3
"""Does the sycophancy hint act THROUGH the written reasoning, or around it?

For every item with a sycophancy run, we cut the chain-of-thought (CoT) at
0%, 25%, 50%, 75% and 100% of its tokens, append "Answer: (", and read the
model's probability of each option letter. Three versions per item:

  A  hinted prompt   + hinted CoT    (what actually happened)
  B  baseline prompt + hinted CoT    (hint sentence removed, hinted reasoning kept)
  C  baseline prompt + baseline CoT  (no hint anywhere)
  D  placebo prompt  + hinted CoT    (like B, but a neutral preamble replaces the hint,
                                       so any preamble effect is held fixed)

P(hint target) under A vs B tells how much the hint sentence still matters once
the reasoning is fixed (a "direct" effect). B vs C tells how much the hinted
reasoning text alone pushes toward the target (an effect carried by the CoT).
Only forward passes, no sampling.

Example:
    python scripts/probe_mediation.py --model mlx-community/Qwen2.5-7B-Instruct-4bit
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cot_faithfulness.analysis.influence import ANSWER_PATTERNS_V2, parse_answer_v2
from cot_faithfulness.config import CACHE_DIR, OUTPUT_DIR

FRACTIONS = [0.0, 0.25, 0.5, 0.75, 1.0]
FINAL_ANSWER_RE = re.compile(r"Answer:\s*\(", re.IGNORECASE)


def cot_body(cot: str) -> str:
    """CoT text before its final 'Answer: (' line."""
    matches = list(FINAL_ANSWER_RE.finditer(cot))
    return cot[: matches[-1].start()].rstrip() if matches else cot.rstrip()


def before_first_answer(body: str) -> str:
    """Body cut just before its FIRST explicit answer statement ("the answer is (B)" etc.).

    At f=1 the model can simply copy a conclusion the CoT already wrote down. This
    cut keeps the reasoning but drops every stated conclusion, so a transplant
    result there is not just copying.
    """
    starts = [m.start() for pat in ANSWER_PATTERNS_V2 for m in pat.finditer(body)]
    if not starts:
        return body
    cut = min(starts)
    # back up to the start of that sentence/line
    sentence_start = max(body.rfind("\n", 0, cut), body.rfind(". ", 0, cut))
    return body[: sentence_start + 1].rstrip() if sentence_start > 0 else body[:cut].rstrip()


def letter_curve(model, tokenizer, prompt: str, cot: str, letters: list[str]) -> dict:
    """P(letter) after each CoT cut, reusing one KV cache for the whole CoT."""
    messages = [{"role": "user", "content": prompt}]
    prefix = tokenizer.encode(
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
        add_special_tokens=False,
    )
    body_text = cot_body(cot)
    body = tokenizer.encode(body_text, add_special_tokens=False)
    pre_answer_cut = min(len(body), len(tokenizer.encode(before_first_answer(body_text), add_special_tokens=False)))
    letter_ids = [tokenizer.encode(l, add_special_tokens=False)[0] for l in letters]

    # (label, number of CoT tokens shown), fed in increasing order
    cuts = [(frac, round(frac * len(body))) for frac in FRACTIONS] + [("pre_answer", pre_answer_cut)]
    cuts.sort(key=lambda c: (c[1], c[0] == "pre_answer"))

    cache = make_prompt_cache(model)
    model(mx.array([prefix]), cache=cache)
    fed = 0
    points = {}
    for label, cut in cuts:
        if cut > fed:
            model(mx.array([body[fed:cut]]), cache=cache)
            fed = cut
        suffix = tokenizer.encode("\nAnswer: (" if cut else "Answer: (", add_special_tokens=False)
        logits = model(mx.array([suffix]), cache=cache)[0, -1].astype(mx.float32)
        probs = mx.softmax(logits)
        letter_p = [probs[i].item() for i in letter_ids]
        total = sum(letter_p)
        points[label] = {
            "frac": label,
            "n_cot_tokens": cut,
            "letter_mass": total,  # sanity check: should be close to 1
            "p": {l: p / total for l, p in zip(letters, letter_p)},
        }
        trim_prompt_cache(cache, len(suffix))
    return {"curve": [points[f] for f in FRACTIONS], "pre_answer": points["pre_answer"], "n_body_tokens": len(body)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    gen_dir = CACHE_DIR / "generations_mlx" / args.model.replace("/", "__")
    out_path = OUTPUT_DIR / "probes" / f"{args.model.replace('/', '__')}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        done = {json.loads(line)["example_id"] for line in open(out_path)}

    pairs = []
    for syc_path in sorted(gen_dir.glob("*__sycophancy_*.json")):
        syc = json.load(open(syc_path))
        base_path = gen_dir / f"{syc['example_id']}__baseline.json"
        plc_path = gen_dir / f"{syc['example_id']}__placebo.json"
        if syc["example_id"] in done or not base_path.exists() or not plc_path.exists():
            continue
        pairs.append((json.load(open(base_path)), syc, json.load(open(plc_path))))
    # Probe items where the hinted answer is the target (that is where mediation is
    # defined), plus 100 random other items as a reference set.
    def hinted_is_target(syc):
        return parse_answer_v2(syc["cot"], sorted(syc["letter_map"])) == syc["hint_suggested_letter"]
    others = [p for p in pairs if not hinted_is_target(p[1])]
    controls = {p[1]["example_id"] for p in random.Random(0).sample(others, min(100, len(others)))}
    pairs = [p for p in pairs if hinted_is_target(p[1]) or p[1]["example_id"] in controls]
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"{len(pairs)} items to probe", flush=True)

    model, tokenizer = load(args.model)
    start = time.time()
    with open(out_path, "a") as out:
        for i, (base, syc, plc) in enumerate(pairs, 1):
            letters = sorted(base["letter_map"].keys())
            row = {
                "model_id": args.model,
                "task": syc["task"],
                "example_id": syc["example_id"],
                "target": syc["hint_suggested_letter"],
                "correct": syc["correct_letter"],
                # fixed parser (D7); syc and baseline prompts keep the canonical letter order
                "baseline_answer": parse_answer_v2(base["cot"], letters),
                "hinted_answer": parse_answer_v2(syc["cot"], letters),
            }
            arms = {
                "A_hint_prompt_hint_cot": (syc["prompt"], syc["cot"]),
                "B_base_prompt_hint_cot": (base["prompt"], syc["cot"]),
                "C_base_prompt_base_cot": (base["prompt"], base["cot"]),
                "D_plc_prompt_hint_cot": (plc["prompt"], syc["cot"]),
            }
            for key, (prompt, cot) in arms.items():
                res = letter_curve(model, tokenizer, prompt, cot, letters)
                row[key] = res["curve"]
                row[key + "_pre_answer"] = res["pre_answer"]
                row[key + "_n_tokens"] = res["n_body_tokens"]
            out.write(json.dumps(row) + "\n")
            if i % 100 == 0 or i == len(pairs):
                rate = i / (time.time() - start)
                print(f"{i}/{len(pairs)} probed, {rate:.2f} items/s", flush=True)


if __name__ == "__main__":
    main()
