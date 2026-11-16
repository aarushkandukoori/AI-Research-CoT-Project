# GPU runbook (vLLM)

For running the larger models on NVIDIA GPUs. Every setting below is copied from the preregistration
addendum ([`PREREG_ADDENDUM_2026-11-10.md`](PREREG_ADDENDUM_2026-11-10.md)); please run the commands as they are.
`scripts/run_vllm.py` writes the same JSON records as the Mac runs, so the analysis needs no changes.

## 1. Set up (about 10 minutes)

```bash
git clone https://github.com/aarushkandukoori/AI-Research-CoT-Project.git
cd AI-Research-CoT-Project && git checkout planned-work-2026-11
python3.12 -m venv .venv && source .venv/bin/activate      # vLLM 0.32.0 needs Python 3.11-3.14
pip install vllm==0.32.0                                    # pins torch 2.13.0; CUDA wheels from PyPI
pip install -r requirements.txt                             # datasets, pandas etc. (mlx is skipped on Linux)
python -m pytest -q tests                                   # about 3 s, no GPU needed
```

The models and the BBH data (`lukaemon/bbh`) download from the Hugging Face Hub on first use; none are gated.

## 2. Runs

Every run covers the 200 frozen items in all four conditions (baseline, sycophancy, reordering, placebo).
Set this once:

```bash
S="--items-file data/substudy_subset_200.json --seed 0"
R="--temperature 0.7 --top-p 0.95 --max-tokens 4096"        # addendum B: reasoning settings, 1 sample
G=cache/generations_vllm
```

| Run | Command |
|---|---|
| Qwen3, thinking on | `for m in 4B 8B; do python scripts/run_vllm.py --model Qwen/Qwen3-$m $S $R --chat-template-kwargs '{"enable_thinking": true}' --out-root $G/think_on; done` |
| Qwen3, thinking off | same, with `'{"enable_thinking": false}'` and `--out-root $G/think_off` |
| R1 distills | `for m in 7B 14B 32B; do python scripts/run_vllm.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-$m $S $R --out-root $G/r1; done` (no template flag: the template already opens `<think>`) |
| Same-base instruct (for the 14B/32B distills) | `for m in 14B 32B; do python scripts/run_vllm.py --model Qwen/Qwen2.5-$m-Instruct $S $R --out-root $G/instruct; done` |
| bf16 rerun of the sub-study (addendum A) | `for m in 1.5B 3B 7B; do python scripts/run_vllm.py --model Qwen/Qwen2.5-$m-Instruct $S --temperature 0.7 --top-p 1.0 --n-samples 5 --max-tokens 768 --out-root $G/sampled_t07; done` |
| If compute allows | Qwen3-14B and Qwen3-32B, thinking on and off, same commands as the Qwen3 rows |

Order of priority: the Qwen3 4B/8B pairs, R1-7B/14B, the bf16 rerun, then the 32B models.
Each command resumes where it stopped: rerun it after a crash or preemption.

**Memory flags** (bf16 weights; defaults are `--tensor-parallel-size 1 --gpu-memory-utilization 0.90`, and
`--max-model-len` is set to the longest prompt + `--max-tokens`, so leave it alone):

| Size | 80 GB card (A100/H100) | 40-48 GB cards |
|---|---|---|
| 1.5B-8B | defaults | defaults |
| 14B (28 GB of weights) | defaults | `--tensor-parallel-size 2` |
| 32B (64 GB of weights) | `--tensor-parallel-size 2` (one card works with `--gpu-memory-utilization 0.95`, but leaves little KV cache and is about 2x slower) | `--tensor-parallel-size 4` |

If vLLM stops at start-up with an out-of-memory or "not enough KV cache" error, add tensor parallelism.

## 3. Cost (estimates, not measured)

From the throughput assumptions in our planning notes: A100-80GB with vLLM in bf16 at about 3,000 tok/s for
7-9B, 2,500 for 14B and 650 for 32B on one card; H100 about 1.7x faster. Thinking-on traces are assumed to
average about 1,500 tokens (all 800 hitting 4,096 is the worst case); thinking off and instruct about 300.

| Run | Generations | A100-hours (worst case) |
|---|---|---|
| Qwen3-4B or 8B, thinking on; R1-7B | 800 | 0.1 each (0.3) |
| R1-14B, Qwen3-14B on | 800 | 0.15 (0.4) |
| R1-32B, Qwen3-32B on (one card) | 800 | 0.5 (1.4) |
| Any thinking-off or instruct run | 800 | under 0.1 |
| bf16 sub-study, each of 1.5B/3B/7B | 4,000 | 0.1 |

Add about 5-10 minutes per run for download and model load. The whole list: about 3-4 A100-hours, at most
about 8 if every trace hits the cap.

## 4. Check a run

```bash
python scripts/check_run.py cache/generations_vllm/think_on/Qwen__Qwen3-8B --items-file data/substudy_subset_200.json
```

It prints, per condition, the record count, the share with no parsed answer, the share at the token cap,
the share whose thinking never closed, and mean tokens. It exits non-zero if any sample is missing (rerun the
same command to fill gaps). Each folder should hold 800 records (4,000 for the sub-study) plus
`run_config.json`. Please tell us if a thinking-on run has more than 10% at the cap; do not raise
`--max-tokens` yourself (a longer cap would be a separate, labeled run).

## 5. Send results back

One `.tar.gz` per run root, run from the repo folder:

```bash
for d in think_on think_off r1 instruct sampled_t07; do tar czf vllm_$d.tar.gz -C cache/generations_vllm $d; done
```

Expected sizes (estimates): a thinking-on model folder is about 10-15 MB of JSON, under 5 MB compressed; thinking
off, instruct and each sub-study model are about 2-12 MB raw and 1-3 MB compressed. Shared Drive or any file
link is fine. Please include the console log of each run if you kept it.

## Do not change

- Prompts, items, chat templates or the code under `cot_faithfulness/`. No system prompt.
- `--temperature`, `--top-p`, `--max-tokens`, `--n-samples`, `--seed`, `--dtype` (bfloat16). No top-k, min-p
  or repetition penalty: the script turns off the model card defaults on purpose, so thinking on and off use
  the same decoding.
- The `--out-root` names above (the analysis looks for `<root>/<org>__<model>/`). A different setting must go
  to a new `--out-root`; the script refuses to mix settings in one folder.
- Do not edit or delete record files; send everything, including failed or truncated traces.

Tensor parallelism, `--gpu-memory-utilization`, `--chunk-size` and the vLLM patch version only
change speed, so adjust them freely (and tell us the vLLM version if it is not 0.32.0).
