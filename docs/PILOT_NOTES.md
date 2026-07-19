# Pilot Run Notes

## What was run locally

| Component | Status |
|-----------|--------|
| Pipeline code | Complete |
| Unit tests | Passing |
| GPU probe | 1.5B loadable on MPS (fp16); 7B requires CUDA + 4-bit |
| Synthetic pilot | 60 examples × 2 models → `outputs/REPORT.md` |
| Real inference | 2 BBH examples × 3 variants on Qwen2.5-1.5B (MPS) |
| Judge validation | 40 hand-labeled cases, 80% exact agreement with heuristic judge |

## Pilot results (synthetic cache — illustrative only)

These numbers demonstrate the pipeline, **not** real model behavior:

| Model | Faithfulness rate [95% CI] | n influenced |
|-------|---------------------------|--------------|
| Qwen2.5-1.5B | 40.0% [22.9%, 57.1%] | 35 |
| Qwen2.5-7B | 64.3% [46.4%, 82.1%] | 28 |

The synthetic data was constructed with ~15% / ~45% cite-hint rates for 1.5B / 7B respectively, so the size-scaling pattern is by design, not empirical.

## Next steps for a real experiment

1. Run on **Colab T4** (`notebooks/cot_faithfulness_colab.ipynb`) for 4-bit on both model sizes
2. Set `OPENAI_API_KEY` and re-run judge with gpt-4o (heuristic used in pilot)
3. Hand-label ≥30 influenced cases in `outputs/human_labels.json`
4. Re-run `python scripts/analyze.py` — cached generations won't regenerate

## Honest limitations

- **Dataset contamination:** BBH is public; models may have memorized tasks
- **Judge ceiling:** 80% agreement on pilot; expect lower on real ambiguous CoTs
- **Small n cells:** reorder hint cells had n<20 in pilot — flagged as noisy
- **Hardware:** Full 75-sample × 4-task run ≈ 3-4 hours on T4 per model
