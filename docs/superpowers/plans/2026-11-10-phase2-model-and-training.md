# Phase 2: Base Model and Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Choose the base model, then train the quantile-head predictor (LoRA + 6-output head) on Kaggle/Colab with resumable Hub checkpoints, run experiments E1–E5, and finish with one fully trained, validated predictor.

**Architecture:** A `llm_movies/model/` package: a dataset that renders each `ResolvedFilm` with random field hiding, a `MoviePredictor` (frozen base LM + PEFT LoRA + quantile head on the last kept token), a masked pinball loss, and a training loop with quarter-epoch validation, early stopping and Hub checkpoints. `llm_movies/eval/` gains metrics, probes (memorization, cutoff), a CPU speed check and a run comparison. Notebooks only launch modules.

**Tech Stack:** PyTorch, Transformers, PEFT, Weights & Biases (optional), huggingface_hub, Kaggle (T4) and Colab (T4, CPU runtime for the speed check).

**Spec:** `docs/superpowers/specs/2026-11-05-llm-movie-predictor-design.md` (sections 7, 8, 10). Roadmap: `docs/superpowers/plans/2026-11-05-roadmap.md`. State before this phase: `PROGRESS.md` (Phase 1 done 2026-11-10).

## Global Constraints

- Everything in Phase 1's Global Constraints still holds (pre-release inputs only, leak safety, locked test sets, offline tests, secrets from the environment, notebooks only launch modules).
- Training reads only `films.parquet` (train, calibration, validation). `locked.parquet` is never read in this phase. Early stopping, experiments and every decision use **validation (2016–2018)** only; calibration rows are not used yet (Phase 3).
- Quantiles `QUANTILES = (0.1, 0.5, 0.9)`; targets in this order `TARGETS = ("revenue", "score")`; predictions are tensors of shape `(batch, 2, 3)`.
- Up to `epochs: 5`; validation every `eval_every: 0.25` epoch; stop after `patience: 3` checks without improvement of mean validation pinball loss; keep the best checkpoint (spec D11).
- Random hiding probability `hide_p: 0.3` per hideable group, training only; validation renders with nothing hidden except an experiment's `drop_groups`.
- Decisions from 2026-11-10: outcome phrases in tagline and synopsis are masked in `render` (training and serving alike; dollar figures are not masked); `source_strength` is kept, randomly hidden in training, and tested in E5; leak check E4 drops `release_type`, `distributor`, `themes`.
- Base weights frozen; LoRA rank 16, alpha 32, dropout 0.05, on all attention and MLP projection layers; head and LoRA in fp32; base in fp16 on CUDA, fp32 on CPU.
- Checkpoints live in the private Hub model repo `Tsomaros/llm-movies-runs` under `runs/<run_name>/{best,last}/`; uploads go through `llm_movies.data.hub_store` (refuses public repos). `WANDB_API_KEY` set → log to W&B project `llm-movies`; unset → log to stdout and `metrics.jsonl` only.
- Torch is not a core dependency: the optional extra `train` holds transformers, peft, accelerate, wandb; torch comes from one of two conflicting extras, `cpu` (PyTorch CPU index: tests, CI, the Windows venv, CPU notebooks) or `cu132` (PyTorch CUDA 13.2 index: GPU notebooks; the only CUDA build of torch 2.15 that still supports the T4). Always pair `train` with one of them (`uv sync --extra train --extra cpu|cu132`). (Changed 2026-11-11: the uv venv shadows the platform's preinstalled torch, so "use the preinstalled CUDA torch" gave CPU torch on a T4.)
- Tests build a tiny random causal LM and tokenizer in memory (no downloads): every model test runs on CPU in seconds.
- `CLAUDE.md` hard rules: code-reviewer after each task's code, test-auditor on each new test file.

## Review Focus

1. A prompt longer than `max_len` → right truncation keeps the start, and pooling reads the last *kept* token, never a pad (Task 4 test `test_truncated_prompt_pools_last_kept_token`).
2. A batch with no revenue rows or no score rows → finite loss, the missing target contributes 0 (Task 3 test `test_loss_with_empty_target_mask`).
3. A Kaggle session killed during a checkpoint upload → resume loads the previous complete checkpoint, never a half-written one (Task 7 test `test_interrupted_upload_resumes_from_previous`).
4. fp16 overflow or a NaN loss → the step is skipped and counted; 10 NaN steps in a row stop the run with a clear error (Task 8 test `test_nan_steps_skipped_then_abort`).
5. Probe answers in odd shapes ("I don't know", "$1.2 billion", "1,234,567", "$85M", "two hundred") → parsed or recorded as unknown, never a crash (Task 9 test `test_parse_money_answer_table`).

---

## File Structure

```
pyproject.toml                         optional extra `train`; dev group gains torch (CPU index), tokenizers
llm_movies/prompt/outcome_words.py     OUTCOME_PATTERNS, mask_outcomes()           (Task 1)
llm_movies/prompt/render.py            masking, HIDEABLE, RANDOM_HIDE, anonymize   (Tasks 1-2)
llm_movies/data/report.py              audit uses OUTCOME_PATTERNS + its own dollar pattern (Task 1)
llm_movies/model/config.py             TrainConfig + load_train_config (YAML with `extends`)
llm_movies/model/head.py               QuantileHead
llm_movies/model/loss.py               pinball_loss
llm_movies/model/predictor.py          MoviePredictor, build_predictor
llm_movies/model/data.py               TargetScaler, FilmDataset, Collator, revenue_sampler
llm_movies/model/checkpoint.py         save/load/resume on the Hub
llm_movies/model/train.py              train() + CLI
llm_movies/eval/metrics.py             revenue_metrics, score_metrics, pinball
llm_movies/eval/probe.py               parse_money, revenue_probe, director_probe, empirical_cutoff + CLI
llm_movies/eval/speed.py               cpu_forward_seconds + CLI
llm_movies/eval/tokens.py              token_lengths + CLI
llm_movies/eval/compare_runs.py        experiments table + CLI
configs/train/base.yaml, e1_*.yaml, e2_anon.yaml, e3_notext.yaml, e4_noleak.yaml, e5_nosource.yaml, final.yaml
notebooks/10_model_choice.ipynb, 11_train.ipynb (Kaggle/Colab), 12_speed.ipynb (Colab CPU)
docs/model_choice.md, reports/experiments.md, reports/phase2_results.md
tests/prompt/test_outcome_words.py, tests/prompt/test_render.py (extended), tests/model/*, tests/eval/test_metrics.py,
tests/eval/test_probe.py, tests/conftest.py (tiny model + tokenizer fixtures)
```

---

### Task 1: Mask outcome phrases

**Files:** Create `llm_movies/prompt/outcome_words.py`; Modify `llm_movies/prompt/render.py`, `llm_movies/data/report.py`; Test `tests/prompt/test_outcome_words.py`, `tests/data/test_report.py`

**Interfaces:**
- Produces: `OUTCOME_PATTERNS: tuple[tuple[str, re.Pattern], ...]` = the report's current audit patterns **except** "dollar figure" (box office, gross/grossed/grossing, Oscar(s), Academy Award(s), award-winning, blockbuster(s), Golden Globe(s), record-breaking, smash hit(s)), word-bounded, case-insensitive; `MASK = "[…]"`; `mask_outcomes(text: str | None) -> str | None`.
- `render` applies `mask_outcomes` to tagline and synopsis (synopsis is masked before the 120-word cut). `report` builds its audit from `OUTCOME_PATTERNS` plus its own dollar pattern, so the two can never drift.

- [x] **Step 1: Failing tests:** `test_each_pattern_masked` (one sentence per pattern → contains `[…]`, not the phrase), `test_case_and_hyphen_variants` ("Award Winning", "box-office"), `test_word_bounds` ("grossly", "Oscarsson" unchanged), `test_dollar_figures_not_masked`, `test_none_passthrough`, `test_render_masks_tagline_and_synopsis`, `test_report_audit_uses_shared_patterns` (every `OUTCOME_PATTERNS` label appears in the report table).
- [x] **Step 2–4:** FAIL → implement → PASS. Golden prompts: run `test_render_matches_golden`; if a golden changes, inspect it and regenerate with `UPDATE_GOLDEN=1` only when the diff is exactly the masking.
- [x] **Step 5:** Commit `feat(prompt): mask outcome phrases`.

### Task 2: Render options for training and experiments

**Files:** Modify `llm_movies/prompt/render.py`; Test `tests/prompt/test_render.py`

**Interfaces:**
- Produces: `HIDEABLE = OPTIONAL_GROUPS + ("source_strength", "release_type", "themes")`; `RANDOM_HIDE = OPTIONAL_GROUPS + ("source_strength",)` (release type and themes are never hidden at random); `render(r, hidden: frozenset[str] = frozenset(), *, anonymize: bool = False) -> str`. `hidden` accepts names from `HIDEABLE`. `anonymize=True` replaces every person's name with `Person N` (numbered by first appearance in line order: director, writers, cast, producers, composer), every company with `Company N`, the distributor with `Distributor`, and the franchise name with `GENERIC_FRANCHISE`; records (films, money, score, awards) stay.
- [x] **Step 1: Failing tests:** `test_hide_release_type_and_themes`, `test_source_strength_hideable_alone` (source material still shown), `test_unknown_group_still_raises`, `test_anonymize_removes_every_name` (no name from the golden film's ResolvedFilm appears; `Person 1` is the director), `test_anonymize_same_person_same_number` (a director who also writes is `Person 1` in both lines), `test_anonymize_keeps_records`, `test_random_hide_excludes_release_type_and_themes`.
- [x] **Step 2–4:** FAIL → implement → PASS. **Step 5:** Commit `feat(prompt): hideable groups and anonymized names`.

### Task 3: Training dependencies, tiny-model fixtures, head and loss

**Files:** Modify `pyproject.toml`, `.github/workflows/ci.yml`; Create `llm_movies/model/__init__.py`, `llm_movies/model/head.py`, `llm_movies/model/loss.py`, `tests/conftest.py`; Test `tests/model/test_head.py`, `tests/model/test_loss.py`

**Interfaces:**
- `pyproject.toml`: `[project.optional-dependencies] train = ["transformers", "peft", "accelerate", "wandb"]` (latest at lock time, lower bounds only); dev group adds `torch` from the PyTorch CPU index (`[tool.uv.sources]` + `[[tool.uv.index]] url = "https://download.pytorch.org/whl/cpu", explicit = true`) and `tokenizers`; CI runs `uv sync --locked --extra train`.
- `tests/conftest.py` fixtures: `tiny_tokenizer` (a `PreTrainedTokenizerFast` from a word-level `tokenizers` model trained in memory on the three golden prompts, with `pad`, `eos` tokens, right padding) and `tiny_lm` (`LlamaForCausalLM(LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4, vocab_size=len(tokenizer)))`, seeded).
- `QUANTILES`, `TARGETS` constants in `llm_movies/model/__init__.py`.
- `QuantileHead(in_dim: int, hidden: int = 256)`: `forward(h: Tensor[B, in_dim]) -> Tensor[B, 2, 3]`; Linear → GELU → Linear(6); per target `(m, a, b)` → `(m - softplus(a), m, m + softplus(b))`.
- `pinball_loss(pred: Tensor[B,2,3], target: Tensor[B,2], mask: Tensor[B,2] (bool), weight: Tensor[B,2]) -> Tensor[]`: per target, weighted mean over available rows of the mean pinball over the 3 quantiles; the score weights are normalised to mean 1 over the available rows; result = revenue term + score term; a target with no rows contributes 0.
- [x] **Step 1: Failing tests:** `test_quantiles_never_cross` (random inputs ×1000 → q10 ≤ q50 ≤ q90), `test_head_shape`, `test_pinball_known_values` (hand-worked: target 1.0, preds (0, 1, 3) → per-quantile 0.1, 0, 0.2 → mean 0.1), `test_loss_with_empty_target_mask`, `test_score_weights_normalised` (scaling all weights ×10 leaves the loss unchanged), `test_masked_rows_ignored` (changing a masked target's value leaves the loss unchanged).
- [x] **Step 2–4:** FAIL → implement → PASS; `uv sync --locked --extra train` works on Windows and in CI. **Step 5:** Commit `feat(model): quantile head and pinball loss`.

### Task 4: The predictor module

**Files:** Create `llm_movies/model/config.py`, `llm_movies/model/predictor.py`, `configs/train/base.yaml`; Test `tests/model/test_config.py`, `tests/model/test_predictor.py`

**Interfaces:**
- `TrainConfig` (frozen dataclass) with defaults from Global Constraints: `run_name, base_model, max_len=768, lora_r=16, lora_alpha=32, lora_dropout=0.05, lr_lora=2e-4, lr_head=1e-3, weight_decay=0.0, batch_size=8, grad_accum=4, epochs=5, eval_every=0.25, patience=3, hide_p=0.3, revenue_oversample=1.5, sample_frac=1.0, drop_groups: tuple[str, ...]=(), anonymize=False, seed=42, save_every_steps=200, hub_repo="Tsomaros/llm-movies-runs", data_repo="Tsomaros/llm-movies-data", wandb_project="llm-movies"`. `load_train_config(path) -> TrainConfig` supports `extends: base.yaml` (child keys override); unknown keys and `drop_groups` outside `HIDEABLE` raise.
- `MoviePredictor(nn.Module)`: holds `lm` (PEFT-wrapped base, `output_hidden_states=True`) and `head`; `forward(input_ids, attention_mask) -> Tensor[B,2,3]`; pooling = last hidden layer at index `attention_mask.sum(1) - 1` (right padding).
- `build_predictor(cfg: TrainConfig, lm: PreTrainedModel | None = None) -> MoviePredictor` (loads `cfg.base_model` when `lm` is None; LoRA `target_modules="all-linear"`; enables gradient checkpointing on CUDA); `trainable_parameters(model) -> int`.
- [x] **Step 1: Failing tests:** `test_extends_overrides`, `test_unknown_key_raises`, `test_bad_drop_group_raises`, `test_output_shape` (tiny model), `test_only_lora_and_head_trainable` (every base weight `requires_grad=False`), `test_padding_does_not_change_prediction` (same prompt alone vs in a batch with a longer prompt → equal outputs, atol 1e-5), `test_truncated_prompt_pools_last_kept_token`.
- [x] **Step 2–4:** FAIL → implement → PASS. **Step 5:** Commit `feat(model): predictor with LoRA and pooling`.

### Task 5: Dataset, collator and sampler

**Files:** Create `llm_movies/model/data.py`; Test `tests/model/test_data.py`; fixture `tests/fixtures/films_small.parquet` (built by a test helper from the mini DB through `llm_movies.data.export`, not committed if it can be built in the test)

**Interfaces:**
- `load_frames(parquet: Path, cfg) -> tuple[DataFrame, DataFrame]` (train, validation; `sample_frac` samples train with `cfg.seed`; reads only `films.parquet`).
- `TargetScaler.fit(train_df) -> TargetScaler` (mean/std of `revenue_log` and `score` over non-null train rows); `.transform(df) -> (target[N,2], mask[N,2])`; `.inverse(pred: Tensor[...,2,3]) -> Tensor` ; `.to_json()/from_json()`.
- `FilmDataset(df, scaler, cfg, train: bool)` with `.set_epoch(epoch: int)`: `__getitem__` returns `{"text", "target", "mask", "weight"}`; text = `render(ResolvedFilm.from_json(row.resolved_json), hidden, anonymize=cfg.anonymize)` where hidden = `drop_groups` ∪ (training only) each `RANDOM_HIDE` group with probability `hide_p` (a per-item RNG seeded from `(cfg.seed, epoch, index)` so a resumed run renders the same text); weight = `[1, score_weight]`.
- `Collator(tokenizer, max_len)` → `input_ids, attention_mask, target, mask, weight` (right padding, right truncation).
- `revenue_sampler(df, factor, seed, epoch) -> torch.utils.data.Sampler` (weights `1 + (factor - 1) * has_revenue`, `len(df)` draws per epoch, with replacement).
- [x] **Step 1: Failing tests:** `test_validation_renders_without_random_hiding` (same text twice, no `not given` added beyond missing fields), `test_drop_groups_always_hidden`, `test_hiding_rate_close_to_hide_p` (2,000 renders → each group hidden 30% ± 4%), `test_same_seed_epoch_index_same_text`, `test_scaler_roundtrip` (inverse(transform) ≈ original), `test_locked_file_never_opened` (call with a dir holding a `locked.parquet` that raises on read → no error), `test_sampler_oversamples_revenue` (factor 2 → revenue share ≈ 2r/(1+r) ± 0.03).
- [x] **Step 2–4:** FAIL → implement → PASS. **Step 5:** Commit `feat(model): dataset, collator and sampler`.

### Task 6: Metrics

**Files:** Create `llm_movies/eval/metrics.py`; Test `tests/eval/test_metrics.py`

**Interfaces:**
- `revenue_metrics(pred: ndarray[N,3], true_log: ndarray[N]) -> dict` with `mae_log` (|q50 − true|), `within_2x` (share with |q50 − true| ≤ ln 2), `coverage_80` (share with q10 ≤ true ≤ q90), `median_width_x` (median of exp(q90 − q10)), `pinball`.
- `score_metrics(pred: ndarray[N,3], true: ndarray[N], weight: ndarray[N]) -> dict` with weighted `mae`, `coverage_80`, `median_width`, `pinball` (same definitions as the app's evaluation).
- [x] **Step 1: Failing tests:** hand-worked values for each metric on 4 rows (working shown in comments), `test_exactly_2x_counts_within`, `test_weights_change_score_mae`.
- [x] **Step 2–4:** FAIL → implement → PASS. **Step 5:** Commit `feat(eval): prediction metrics`.

### Task 7: Hub checkpoints and resume

**Files:** Create `llm_movies/model/checkpoint.py`; Test `tests/model/test_checkpoint.py`

**Interfaces:**
- `save_checkpoint(model, scaler, cfg, state: dict, kind: Literal["best","last"], api=None) -> None`: writes to a local temp folder (`adapter/` via `save_pretrained`, `head.pt`, `scaler.json`, `train_config.yaml`, `state.json`; `last` also `optimizer.pt`, `grad_scaler.pt`, `lr_scheduler.pt`, `rng.pt` with the torch, numpy and Python RNG states), then uploads `runs/<run_name>/<kind>/` in **one commit** through `hub_store`; `state.json` (`step, epoch, best_metric, bad_checks, metrics`) is written last within the commit.
- `load_checkpoint(cfg, kind, api=None, lm=None) -> tuple[MoviePredictor, TargetScaler, dict] | None` (None when absent).
- `resume_state(cfg, api=None) -> dict | None` (the `last` state, or None for a fresh run).
- [x] **Step 1: Failing tests** (fake Hub api over a temp dir, tiny model): `test_save_load_roundtrip_same_predictions`, `test_upload_is_one_commit`, `test_interrupted_upload_resumes_from_previous` (fake commit raises → the Hub still holds the earlier complete `last`), `test_refuses_public_repo`, `test_missing_run_returns_none`.
- [x] **Step 2–4:** FAIL → implement → PASS. **Step 5:** Commit `feat(model): Hub checkpoints and resume`.

### Task 8: Training loop

**Files:** Create `llm_movies/model/train.py`; Test `tests/model/test_train.py`

**Interfaces:**
- `train(cfg: TrainConfig, data_dir: Path, *, api=None, lm=None, tokenizer=None, device=None) -> dict` (final summary: `best_metric`, `best_step`, `stopped_reason` ∈ {"patience", "epochs", "nan"}, validation metrics of the best checkpoint).
- Behaviour: AdamW with two parameter groups (`lr_lora`, `lr_head`), linear warmup 3% then cosine decay; `grad_accum`; on CUDA fp16 autocast + `GradScaler`; validation every `eval_every` epoch on the full validation frame → mean pinball (both targets, scaled units) + `revenue_metrics`/`score_metrics` in original units; improvement → `save_checkpoint(..., "best")`; `last` every `save_every_steps` and after each validation; resumes from `resume_state`; `patience` checks without improvement → stop; a non-finite loss skips the step, 10 in a row → stop with `stopped_reason="nan"`; logs to W&B when `WANDB_API_KEY` is set, always to `metrics.jsonl`.
- CLI: `python -m llm_movies.model.train --config configs/train/<name>.yaml [--data DIR]` (downloads `films.parquet` from `cfg.data_repo` when `--data` is absent).
- [x] **Step 1: Failing tests** (tiny model, 64-film frame from the mini DB, CPU): `test_loss_decreases` (validation pinball after 3 epochs < before training), `test_early_stopping_by_patience` (frozen learning rate 0 → stops after `patience` checks), `test_resume_continues_same_run` (stop after step k, resume → same final state as an uninterrupted run, atol 1e-5), `test_nan_steps_skipped_then_abort` (inject NaN loss), `test_best_checkpoint_is_best_metric`.
- [x] **Step 2–4:** FAIL → implement → PASS. **Step 5:** Commit `feat(model): training loop`.

### Task 9: Probes, token lengths and CPU speed

**Files:** Create `llm_movies/eval/probe.py`, `llm_movies/eval/tokens.py`, `llm_movies/eval/speed.py`; Test `tests/eval/test_probe.py`, `tests/eval/test_tokens.py`, `tests/eval/test_speed.py`

**Interfaces:**
- `parse_money(text: str) -> float | None`: first number in the answer with optional `,` thousands, decimal, and scale word or suffix (`thousand|K`, `million|M|mn`, `billion|B|bn`); words without digits → None.
- `revenue_probe(generate: Callable[[list[str]], list[str]], films: DataFrame) -> DataFrame`: prompt `The {year} film "{title}" earned a worldwide box office gross of $`; columns `tmdb_id, answer, guess_usd, nominal_usd, known` (`known` = guess within 2× of the nominal revenue, `nominal = revenue_2025 * cpi[year] / cpi[2025]`). Films come from the database's **validation** split with revenue (open data).
- `director_probe(generate, films) -> DataFrame`: prompt `The {year} film "{title}" was directed by`; `known` = the database director's last name appears in the answer (case-insensitive, accents folded). Uses titles and directors only (inputs, never targets), so it may sample films of any year, including 2019+.
- `empirical_cutoff(results: DataFrame) -> date`: recall per release quarter; baseline = mean recall of 2019–2020; cutoff = last day of the last quarter whose recall is ≥ 50% of the baseline.
- `make_generator(model_id: str, device: str, max_new_tokens: int = 12) -> Callable` (greedy, batched).
- `token_lengths(parquet, tokenizer_id) -> dict` (`p50, p90, p99, max` over open-split prompts rendered with nothing hidden); `recommended_max_len = ceil(p99 / 64) * 64`.
- `cpu_forward_seconds(model_id, max_len, batch=13, int8=True) -> float` (median of 5 forward passes of `batch` prompts on CPU with `torch.ao.quantization.quantize_dynamic` on Linear layers; batch 13 = one prediction + 12 effect groups).
- CLIs: `python -m llm_movies.eval.probe revenue|director --model ID --db PATH --n N --out CSV`, `python -m llm_movies.eval.tokens --tokenizer ID`, `python -m llm_movies.eval.speed --model ID --max-len L`.
- [x] **Step 1: Failing tests:** `test_parse_money_answer_table` (the five Review Focus shapes + "$1.2 billion" → 1.2e9, "$85M" → 8.5e7, "1,234,567" → 1234567, "I don't know" → None, "two hundred" → None), `test_revenue_probe_known_within_2x` (fake generator), `test_director_probe_accents_folded`, `test_empirical_cutoff_on_synthetic_curve`, `test_token_lengths_tiny_tokenizer`, `test_speed_runs_on_tiny_model` (returns a positive float).
- [x] **Step 2–4:** FAIL → implement → PASS. **Step 5:** Commit `feat(eval): probes, token lengths and CPU speed`.

### Task 10: Choose the base model (Colab + decision)

**Files:** Create `notebooks/10_model_choice.ipynb`, `notebooks/12_speed.ipynb`, `docs/model_choice.md`; Modify `config/data.yaml`, `tests/test_config.py`

Candidates: `Qwen/Qwen3-0.6B-Base`, `Qwen/Qwen3-1.7B-Base`, `meta-llama/Llama-3.2-1B`, `google/gemma-3-1b-pt`, `HuggingFaceTB/SmolLM2-1.7B`; add Gemma 4 E2B only if its licence allows fine-tuning plus a public Space and its CPU time fits. Use base (pretrained), not instruct, checkpoints.

- [x] **Step 1:** For each candidate, from the official model card: licence, stated training cutoff (or "not stated"), parameter count. Record with links in `docs/model_choice.md`.
- [x] **Step 2:** Colab T4 (`10_model_choice`): `director_probe` on 150 films per quarter 2019Q1–2026Q2 (films with ≥100 votes) and `revenue_probe` on 300 validation films, per candidate → `empirical_cutoff` and memorization rate. Colab CPU runtime (`12_speed`): `cpu_forward_seconds` at `max_len` 768.
- [x] **Step 3:** Fill the table in `docs/model_choice.md`: licence · stated cutoff · empirical cutoff · director recall 2019–20 · revenue known % · CPU seconds (batch 13). Rule for the cutoff used: the later of stated and empirical.
- [x] **Step 4: Decision with Dimitris:** one family and two sizes for E1 (the smaller must run batch 13 in ≤ 20 s on CPU so prediction + explanation stays near the 30 s target; **relaxed to ≤ 300 s on 2026-11-11**, see `docs/model_choice.md`). Record it in `docs/model_choice.md` and `PROGRESS.md`.
- [x] **Step 5:** Set `base_model_cutoff` in `config/data.yaml` to the chosen family's cutoff; update `tests/test_config.py` to the new value; rebuild and export in Colab (`03_build_and_export`, no re-pull); check the report shows a non-zero holdout. Commit `data: base model chosen, cutoff set, holdout built`.

### Task 11: Token lengths and launchers

**Files:** Create `notebooks/11_train.ipynb`, configs `configs/train/base.yaml` (finalised), `e1_small.yaml`, `e1_large.yaml`, `e2_anon.yaml`, `e3_notext.yaml`, `e4_noleak.yaml`, `e5_nosource.yaml`; Modify `notebooks/README.md`

- [x] **Step 1:** Run `python -m llm_movies.eval.tokens` with the chosen tokenizer; set `max_len` in `base.yaml` to `recommended_max_len`; record p50/p99 in `PROGRESS.md` (spec target 400–600 tokens; note if exceeded).
- [x] **Step 2:** Configs: `base.yaml` = chosen smaller model, `sample_frac: 0.3`. E1 small/large differ only in `base_model`. E2 `anonymize: true`. E3 `drop_groups: [synopsis, tagline]`. E4 `drop_groups: [release_type, distributor, themes]`. E5 `drop_groups: [source_strength]`. Each `run_name` unique.
- [x] **Step 3:** `11_train.ipynb`: works on Kaggle (Secrets `HF_TOKEN`, `WANDB_API_KEY`) and Colab; installs `-e ".[train]"` on the platform's torch; runs `python -m llm_movies.model.train --config $CONFIG`; rerunning the same config resumes.
- [x] **Step 4:** Smoke run on T4 with `sample_frac: 0.01`, `epochs: 1`: a `best` checkpoint appears on the Hub and resumes after a forced restart. Commit `feat(notebooks): training launcher and experiment configs`.

### Task 12: Experiments E1–E5

**Files:** Create `llm_movies/eval/compare_runs.py`, `reports/experiments.md`; Test `tests/eval/test_compare_runs.py`

**Interfaces:**
- `compare(run_names: list[str], api=None) -> str` (markdown: one row per run from each `best/state.json`: validation pinball, revenue `mae_log`, `within_2x`, `coverage_80`, score `mae`, `coverage_80`, best step, stopped reason); CLI `python -m llm_movies.eval.compare_runs --runs a,b,... --out reports/experiments.md`.
- [ ] **Step 1:** Failing test `test_compare_table_from_fake_states` → implement → PASS → commit `feat(eval): compare runs`.
- [ ] **Step 2:** Run E1 (both sizes), then E2–E5 on the chosen size (Dimitris on Kaggle, several sessions).
- [ ] **Step 3:** Write `reports/experiments.md` with the table and these pre-set decision rules:
  - E1: keep the larger model only if it improves revenue `mae_log` by ≥ 0.03 **or** score `mae` by ≥ 0.02 **and** its CPU time passes Task 10's limit (≤ 300 s for batch 13 at max_len 768).
  - E2 (anonymized names), E3 (no text): analysis only; record the gap. Names and text stay unless removing them improves validation pinball.
  - E4: if removing `release_type`/`distributor`/`themes` worsens validation pinball by > 2% relative, look at which group and whether that gain is plausible before release; decide with Dimitris to keep, fix or drop it.
  - E5: keep `source_strength` only if it improves validation pinball by ≥ 1% relative; otherwise add it to `drop_groups` permanently.
- [ ] **Step 4:** Decisions recorded in `reports/experiments.md` and `PROGRESS.md`; `configs/train/final.yaml` written from them (`sample_frac: 1.0`). Commit `docs: experiment results and final config`.

### Task 13: Full training run

**Files:** Create `reports/phase2_results.md`

- [ ] **Step 1:** Run `final.yaml` on Kaggle (up to 5 epochs, several sessions, resuming).
- [ ] **Step 2:** `reports/phase2_results.md`: validation metrics of `runs/final/best` (uncalibrated, so coverage is expected to miss 80%; calibration is Phase 3), stopped reason, training time, W&B link, and the app's LightGBM validation numbers from its docs for context only (the fair head-to-head is Phase 4).
- [ ] **Step 3:** Update `PROGRESS.md`; commit `docs: Phase 2 results`. Phase 2 done; write the Phase 3 plan.
