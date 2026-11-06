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
| G2 | Human calibration not done yet; the judge labeled every case first, so human labels will come after it (`scripts/export_human_labels.py` draws the set) | open |
| G3 | Extraction checked by comparing two batteries on every trace instead of a hand check of 100 traces | |
| Sample | 937 items per cell instead of 1,000 (causal judgement has 187 test items); reordering analyses drop the 353 items whose target was already (A), since that prompt equals the baseline | |
| Probes | Mediation probes (`scripts/probe_mediation.py`) added after reading related work | not preregistered |
| Sub-study | Stochastic sub-study at T=0.7 not run | |

## Do not trust

Charts in `docs/figures/` and numbers in `docs/SAMPLE_REPORT.md` remain **synthetic** pipeline demos.
