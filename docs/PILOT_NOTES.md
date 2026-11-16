# Experiment Status and Deviation Log

## Protocol patches (D1-D6), applied 2026-07-22

| Defect | Fix |
|--------|-----|
| D1 reorder `(a)` trigger | Removed; causal-position signals only; judge rubric updated |
| D2 influence denominator | `(model, item, condition)` triples |
| D3 silent heuristic | Off by default; missing API key raises |
| D4 target confound | Uniform wrong-option sampling, independent per condition |
| D5 saturated CI | Wilson when rate ∈ {0,1} |
| D6 relaxed agreement | Replaced with Cohen's κ (per condition) + F± bracket |
| Placebo arm | Added (`placebo`) |
| Sample size default | 250 / task |
| Parse failures | Logged in analysis |

## Real runs (done, November 2026)

Qwen2.5-1.5B, 3B and 7B Instruct on all 937 items, each under baseline, sycophancy, reordering and placebo prompts (11,244 generations). Results, data layout and commands are in the [README](../README.md#real-runs-november-2026).

## Deviations from the July 2026 plan

Recorded here in November 2026; they were not logged with timestamps as they happened.

| | Deviation | Reason |
|---|---|---|
| Hardware | 4-bit MLX conversions on an Apple M4 (`scripts/run_mlx.py`) instead of 4-bit NF4 on a T4 | speed; quantization is still 4-bit, the scheme differs |
| Models | Qwen2.5-3B-Instruct added between the two planned sizes | same-family sweep |
| Judge | Claude Opus 5.5, two blind passes + tie-break, with a mention field and evidence quote added to the output ([`judge_prompt.md`](judge_prompt.md)) instead of gpt-4o | gpt-4o key unavailable |
| D7 (2026-10-29) | Answer extraction battery replaced (`ANSWER_PATTERNS_V2` in `cot_faithfulness/analysis/influence.py`) before any acknowledgment rate was computed | the planned battery missed "the correct answer is (B)" and fell back to the last "option X" in the final 500 characters |
| G1 | Steering gate uses the excess switch-to-target rate over the placebo; the binomial steering-share test is still reported for 3- and 4-option tasks | the binomial test is undefined on binary tasks |
| G2 | Human calibration not done yet; the judge labeled every case first, so human labels will come after it (`scripts/export_human_labels.py` draws the set, `scripts/g2_agreement.py` scores it) | open |
| G3 | Extraction checked by comparing two batteries on every trace instead of a hand check of 100 traces | |
| Sample | 937 items per cell instead of 1,000 (causal judgement has 187 test items); reordering analyses drop the 353 items whose target was already (A), since that prompt equals the baseline | |
| Probes | Mediation probes (`scripts/probe_mediation.py`) added after reading related work | not preregistered |
| Sub-study | Stochastic sub-study at T=0.7 not run | |

## Planned runs (code ready 2026-11-10, nothing run yet)

Pipeline support for the two planned experiments; no results exist yet.

- **Stochastic sub-study (preregistered: T = 0.7, n = 5 per prompt, 200-item subset).** The plan did not say how to pick the 200 items. Fixed on 2026-11-10, before any sampled generation: 50 items per task, uniform without replacement, `random.Random("20261110:<task>")` (`scripts/make_substudy_subset.py` -> `data/substudy_subset_200.json`). top-p is left off (1.0); the plan did not mention it.
- **Sampled estimands** (`scripts/analyze_sampled.py`): every (baseline sample, hinted sample) pair is treated as one greedy comparison, so with one sample per prompt all numbers equal `analyze_mlx.py` (checked on the released greedy runs: same S_h, S_plc, E_h, influenced sets and acknowledgment rates). A hinted sample enters the influenced set if it hits the target and at least one baseline sample did not; it is weighted by the share of baseline samples that did not. Intervals resample items.
- **Reasoning models**: `run_mlx.py --chat-template-kwargs` passes options such as `{"enable_thinking": true}` to the chat template; the answer is read after `</think>` and the full trace stays in `cot` for the judge.

```bash
python scripts/make_substudy_subset.py
for m in 1.5B 3B 7B; do
  python scripts/run_mlx.py --model mlx-community/Qwen2.5-$m-Instruct-4bit --temperature 0.7 --n-samples 5 --seed 0 \
      --items-file data/substudy_subset_200.json            # -> cache/generations_mlx_sampled/
done
python scripts/analyze_sampled.py --items-file data/substudy_subset_200.json   # -> outputs/substudy_t07/
python scripts/prepare_judge_batches.py --root outputs/substudy_t07 1.5B 3B 7B
# judge pass1/pass2 as in docs/judge_prompt.md, save outputs/substudy_t07/judge_raw/<size>.json, then per size:
python scripts/prepare_judge_batches.py --root outputs/substudy_t07 --tiebreak outputs/substudy_t07/judge_raw/1.5B.json 1.5B
python scripts/finalize_judge.py --root outputs/substudy_t07 outputs/substudy_t07/judge_raw/*.json
python scripts/analyze_sampled.py --items-file data/substudy_subset_200.json   # adds acknowledgment and within-item variance
```

## Do not trust

Charts in `docs/figures/` and numbers in `docs/SAMPLE_REPORT.md` remain **synthetic** pipeline demos.
