# Preregistration addendum, 2026-11-10

Written and committed before any sampled or reasoning-model generation was run. The July 2026 plan
(first version of the paper) stays in force; this file fixes the details it left open and adds the
reasoning-model extension. Code for everything below is on branch `planned-work-2026-11`.

## A. Sampled-decoding sub-study (preregistered in July, details fixed now)

July plan: "A secondary run at T = 0.7 with n = 5 samples per prompt on a 200-item subset, reporting
within-item variance in adjudicated faithfulness."

Fixed now:
- **Items:** `data/substudy_subset_200.json`, 50 items per BBH task, drawn uniformly with seed
  `random.Random("20261110:<task>")` (`scripts/make_substudy_subset.py`).
- **Models:** the same three checkpoints as the main run (mlx-community Qwen2.5-1.5B/3B/7B-Instruct-4bit),
  same prompts, chat template, 768-token cap and the final extraction battery (after deviation D7).
- **Decoding:** temperature 0.7, top-p off (1.0, the plan names no top-p), 5 samples for every prompt in
  all four conditions (baseline, sycophancy, reordering, placebo).
- **Steering:** S_h, S_plc, E_h and the noise share are computed over all (baseline sample, hinted
  sample) pairs of an item, with a bootstrap over items (B = 10,000).
- **Influenced set:** a hinted sample is influenced if it gives the target and at least one baseline
  sample of the same item did not; it is weighted by the share of baseline samples that did not give the
  target. With one sample per prompt this is exactly Eq. 1 of the paper. Sensitivity: only items where no
  baseline sample gave the target.
- **Judge:** the same rubric and prompt (`docs/judge_prompt.md`), two blind passes and a tie-break vote,
  run the same way as for the main run. The "answer without the hint" shown to the judge is the most
  common non-target baseline answer.
- **Outcomes (descriptive, with intervals):** (1) among items with two or more influenced samples, the
  share whose samples get different acknowledgment labels, and a between-item vs within-item variance
  split; (2) the acknowledgment rate under sampling vs greedy on the same items; (3) steering under
  sampling vs greedy on the same items. No new hypothesis test.

## B. Models trained to reason at length (new)

- **Models (local, 4-bit MLX):** Qwen3-1.7B with thinking on and with thinking off (the same weights;
  `enable_thinking` in the chat template), and DeepSeek-R1-Distill-Qwen-1.5B.
- **Items and conditions:** the 200-item subset above, all four conditions, one sample per prompt.
- **Decoding:** temperature 0.7, top-p 0.95, at most 4,096 new tokens, for both modes and both models.
- **Answer:** parsed from the text after the last `</think>`; if the thinking block never closes, from the
  whole text (counted and reported).
- **Judge:** the same rubric and procedure on the full output (thinking and final answer together). We
  also report separately whether the hint is mentioned in the thinking and in the final answer.
- **Primary comparison:** sycophancy acknowledgment rate for Qwen3-1.7B with thinking on vs off,
  two-proportion test, two-sided, alpha 0.05. Prediction: higher with thinking on. Steering is estimated
  against the placebo in each mode as in the main run, and a condition must pass G1 before its
  acknowledgment rate is interpreted.
- **Larger models (planned, on NVIDIA GPUs with vLLM, same prompts and settings):** Qwen3 4B/8B (and
  14B/32B if compute allows), thinking on vs off; DeepSeek-R1-Distill-Qwen 7B/14B/32B; Qwen2.5-14B/32B-
  Instruct as same-base comparisons for the 14B/32B distills. GPU results are reported apart from the
  4-bit MLX results.

## C. Reporting

Sample sizes are fixed above, with no optional stopping. All results are reported whatever their
direction, including a failed G1 or a reversed prediction. Anything decided after seeing results is
listed as a deviation.
