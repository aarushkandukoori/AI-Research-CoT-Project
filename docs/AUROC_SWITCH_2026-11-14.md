# Does the readout before reasoning predict which hinted items switch? (2026-11-14)

Suggested by Hyojung Han (ThakiCloud), who is thanked in the paper. The 7B result "target on top
before any reasoning in 92% of influenced items (238/258)" only looks at items that switched. This
compares them with hinted items that do not switch (AUROC), kept separate from any claim about when
the model commits.

- Script: `scripts/auroc_switch.py`, output `outputs/analysis/auroc_switch.json`. No new generations.
- Sets: same loaders and definitions as `gap_analyses.py`. The script stops if the counts differ
  from the paper's.
  - Switched: the influenced set I_h, every item (224 / 206 / 258 at 1.5B / 3B / 7B).
  - Resisted: the matched random sample (77 / 83 / 90). A random sample of one group leaves
    AUROC unbiased.
- Score: probability of the hint target in the forced answer (`probe_mediation.py` arms).
- AUROC: chance that a random switched item scores above a random resisted item (ties count 1/2).
  - "Within task" compares pairs from the same BBH task only, so option count cannot drive it.
  - 95% intervals: percentile bootstrap, items resampled within each group, B = 10,000.
  - Differences between readouts use the same resamples (paired).

## Before any reasoning (pooled AUROC, 95% CI)

| Readout | 1.5B | 3B | 7B |
|---|---|---|---|
| Hinted prompt, no CoT (A, f = 0) | 0.59 [0.51, 0.67] | 0.69 [0.62, 0.76] | 0.78 [0.72, 0.83] |
| Baseline prompt, no CoT (C, f = 0) | 0.64 [0.57, 0.70] | 0.64 [0.56, 0.71] | 0.72 [0.65, 0.78] |
| Placebo prompt, no CoT (D, f = 0) | 0.62 [0.55, 0.69] | 0.64 [0.57, 0.71] | 0.70 [0.64, 0.77] |
| Hint lift, A minus C | 0.51 [0.43, 0.59] | 0.63 [0.56, 0.69] | 0.61 [0.54, 0.68] |
| Target on top under A (the 92% / 59% rule) | 0.62 [0.56, 0.68] | 0.69 [0.63, 0.75] | 0.67 [0.61, 0.72] |
| Gain of A over C (paired), pooled | -0.04 [-0.11, +0.02] | +0.05 [-0.01, +0.11] | +0.06 [-0.00, +0.12] |
| Gain of A over C (paired), within task | -0.02 [-0.10, +0.05] | +0.07 [+0.01, +0.14] | +0.04 [-0.03, +0.11] |

## Along the CoT (hinted prompt + hinted CoT cut at fraction f, pooled AUROC)

| f | 0 | .25 | .5 | .75 | 1 |
|---|---|---|---|---|---|
| 1.5B | 0.59 | 0.60 | 0.61 | 0.69 | 0.99 |
| 3B | 0.69 | 0.62 | 0.67 | 0.75 | 1.00 |
| 7B | 0.78 | 0.75 | 0.74 | 0.81 | 1.00 |

Arm B, the hinted CoT under the baseline prompt (the CoT text without the hint sentence), is
close to arm A. At 7B it reads 0.69, 0.70, 0.81, 0.98 at f = .25 to 1. Full intervals and
within-task values are in the JSON.

## What it says

1. **The readout before reasoning predicts switching only moderately.** The hinted readout gives
   AUROC 0.59 / 0.69 / 0.78. The 92% vs 59% "target on top" rule at 7B is 0.67.
2. **Most of that is there without the hint.** The model's own unhinted lean toward the same
   letter predicts switching nearly as well (0.64 / 0.64 / 0.72).
   - Adding the hint gains -0.04 / +0.05 / +0.06, with a 95% interval that includes zero at
     every size pooled. Within task, only 3B excludes zero.
   - The hint's lift alone is a weak predictor (0.51 / 0.63 / 0.61).
3. **So the 92% partly reflects selection.** Items that switch were already leaning toward the
   target before the hint. It is weak evidence that the answer is settled before the reasoning.
   (The first version said "mostly reflects selection". That is stronger
   than these numbers support: see the 2026-11-15 update below.)
4. **Predictive value barely moves through the first half of the CoT.**
   - It rises in the last quarter (f = .75: 0.69 / 0.75 / 0.81).
   - At f = 1 it reaches about 1, where the readout is essentially the answer.
   - This fits the lead-vs-commitment result (`LEAD_VS_COMMITMENT_2026-11-11.md`): reversals
     mostly happen in the last quarter.

## Limits

- The readout is still a forced answer appended after a cut (a modified continuation).
- This is predictive value, not a test of commitment.
- The resisted group is a sample of 77 to 90 items, so intervals are wide.
- Generations are greedy, so each item has one outcome.
- Per-task values are noisy. For example, formal fallacies at 1.5B gives A = 0.35 on 23 vs 21
  items.
- Post hoc, not preregistered.

## Update 2026-11-15 (revised reading of Section 5.3 and Table 10)

- **What the "gain" row shows.** AUROC(A0) minus AUROC(C0) compares two standalone scores. It
  does not test whether the hinted readout carries information beyond the unhinted one. "No clear
  evidence of extra predictive value" is the most these rows support, and not "the separation is
  mostly the original preference".
- **The shift itself is associated with switching** at 3B (0.63 [0.56, 0.69]) and 7B
  (0.61 [0.54, 0.68]), not at 1.5B (0.51 [0.43, 0.59]). That is an association only.
- **New check, exploratory: `scripts/auroc_conditional.py`** -> `outputs/analysis/auroc_conditional.json`.
  - Two logistic models of switching, P_C(0) alone vs P_C(0) + shift (same as P_C(0) + P_A(0)).
  - Same held-out folds for both: stratified 5-fold, repeated 20 times, fixed seeds.
  - AUROC counts only pairs held out in the same fold, so one fitted model scores both items.
    In a first test run, pooling held-out scores across folds pulled the P_C(0)-only model well
    below the AUROC of P_C(0) itself at 7B (most P_C(0) values are near zero, so fold-to-fold
    intercept noise decided the order). So the script compares within folds.
  - Paired bootstrap over items (within each group, B = 2,000), refitting both models on the same
    folds in every draw (5 repeats per draw). Check: bootstrap with fixed held-out predictions
    (B = 10,000). Sensitivity: log-odds features.

| Held-out AUROC (95% CI) | 1.5B | 3B | 7B |
|---|---|---|---|
| P_C(0) alone | 0.64 [0.56, 0.70] | 0.64 [0.55, 0.71] | 0.72 [0.65, 0.78] |
| P_C(0) + shift | 0.64 [0.53, 0.71] | 0.68 [0.60, 0.76] | 0.77 [0.68, 0.83] |
| Gain from the shift | +0.01 [-0.05, +0.03] | +0.04 [-0.01, +0.14] | +0.05 [+0.003, +0.09] |
| Gain, within task | +0.02 [-0.05, +0.05] | +0.06 [+0.001, +0.16] | +0.04 [-0.02, +0.09] |
| Gain, fixed-prediction bootstrap | [-0.02, +0.04] | [-0.02, +0.10] | [+0.02, +0.09] |
| Gain, log-odds features | -0.00 [-0.06, +0.01] | +0.05 [-0.01, +0.14] | +0.06 [+0.02, +0.11] |

- **Reading.** At 7B the gain's interval sits just above zero under all three versions, so there is
  weak evidence that the hinted readout adds some predictive value at 7B. At 3B the pooled
  interval includes zero (within task it narrowly excludes it). At 1.5B nothing. Post hoc,
  borderline, no multiplicity correction. It cannot separate the hint's content from copying of the
  target letter, and it says nothing about commitment.
- In the paper draft (Section 5.3, Table 10 last panel, Appendix G) since 2026-11-15.
