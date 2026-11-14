# Phase 2 experiments (Task 12)

Every run: 30% of the training films (`sample_frac: 0.3`), seed 42, up to 5 epochs, patience 3. Early stopping
checks a fixed sample of 2,000 validation films, drawn so the share with revenue is kept. When a run ends, its best
checkpoint is scored once on the whole validation split (5,144 films with a target). Those full-split numbers are the
ones compared below.

Table produced by `python -m llm_movies.eval.compare_runs --runs e1_small,e1_large`:

Metrics are the best checkpoint's on the full validation split (`full_validation` in `best/state.json`); check pinball is the early-stopping metric on the run's validation sample. — means not available yet.

| run | best step | stopped | val pinball | revenue mae_log | revenue within_2x | revenue coverage_80 | score mae | score coverage_80 | check pinball |
|---|---|---|---|---|---|---|---|---|---|
| e1_small | 558 | patience | 0.1911 | 1.567 | 0.302 | 0.724 | 0.567 | 0.698 | 0.1931 |
| e1_large | 372 | patience | 0.1836 | 1.579 | 0.266 | 0.672 | 0.529 | 0.763 | 0.1848 |

## E1: model size (Qwen3-0.6B-Base vs Qwen3-1.7B-Base)

**Rule (fixed before the runs):** keep the larger model only if it improves revenue `mae_log` by ≥ 0.03 **or** score
`mae` by ≥ 0.02, **and** it runs batch 13 at max_len 768 in ≤ 300 s on CPU.

| Test | 0.6B | 1.7B | 1.7B minus 0.6B | Threshold | Met |
|---|---|---|---|---|---|
| revenue `mae_log` | 1.567 | 1.579 | +0.012 (worse) | ≤ −0.03 | no |
| score `mae` | 0.567 | 0.529 | −0.039 (better) | ≤ −0.02 | **yes** |
| CPU, batch 13 at 768 (Task 10) | 100.5 s | 249.3 s | | ≤ 300 s | **yes** |

**Result under the rule: keep the larger model, Qwen3-1.7B-Base**, because the score target improved enough and it passes
the CPU limit.

What the rule does not weigh:
- **The two targets move in opposite directions.**
  - The 1.7B model is better on score: `mae` −0.039, and 80% coverage 0.763 against 0.698.
  - It is worse on revenue: within 2× 0.266 against 0.302, and 80% coverage 0.672 against 0.724.
  - Its revenue `mae_log` is 0.012 worse, which is small next to the 0.03 threshold.
  - Overall validation pinball improves by 3.9% relative (0.1836 against 0.1911).
- **Coverage here is before calibration** (calibration is Phase 3). The coverage gaps show how wide each model's raw
  ranges are, not how good the final product will be.
- **Single seed, 30% of the films.** A score `mae` difference of 0.039 clears the threshold by about twice. This
  comparison can't show how much of it is noise.
- **Cost:**
  - The 1.7B model stopped earlier (best at step 372 of about 1,240, against 558), but took about 3 Colab sessions
    against 1.
  - Serving on CPU should take about 270 s at 832 tokens, against about 110 s.
  - E2–E5 on 1.7B cost about 4 × 3 sessions.

## E2–E5

Not run yet. The owner confirmed the E1 result on 2026-11-14, so they run on Qwen3-1.7B-Base: each config extends
`e1_large.yaml` and changes only what its experiment tests. `e1_large` is their reference row.
