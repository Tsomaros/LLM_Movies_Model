# Progress

## Now
- 2026-11-14: **Phase 2 Task 12, E1 runs done (owner, Colab T4). Under the rule, the larger model wins; owner to confirm.**
  - Both runs stopped on patience: `e1_small` with its best at step 558 of about 1,240, and `e1_large` at step 372
    (3 Colab sessions, resumed across days).
  - Both have `full_validation` (5,144 films) and `val_films` 2000, so the end-of-run pass works on a GPU.
  - On the full split, 0.6B against 1.7B:
    - val pinball 0.1911 against 0.1836;
    - revenue `mae_log` 1.567 against 1.579 (1.7B 0.012 worse);
    - score `mae` 0.567 against 0.529 (1.7B 0.039 better);
    - CPU time 100.5 s against 249.3 s.
  - **E1 rule:** score improved by at least 0.02 and CPU time is at most 300 s, so keep `Qwen/Qwen3-1.7B-Base`.
  - Caveats are in `reports/experiments.md`:
    - revenue gets worse (within 2× 0.302 to 0.266, coverage 0.724 to 0.672);
    - one seed only;
    - about 3 times the Colab time;
    - serving should take about 270 s.
  - **Open:** E2-E5 run on the chosen size. The configs still extend `base.yaml` (0.6B), so switching them to 1.7B is
    waiting on the owner's go-ahead.
- 2026-11-11: **Validation question resolved (`val_sample`), and Phase 2 Task 12 Step 1 done (`compare_runs`).**
  - **`TrainConfig.val_sample`** (positive int or null, default null). `base.yaml` sets **2000**, so every experiment
    inherits it. `final.yaml` (Task 12 Step 4) will use null.
  - **The sample** is `validation_sample(val_df, n, seed)` in `llm_movies/model/data.py`:
    - drawn after sorting by tmdb_id, so the same seed gives the same films whatever the row order, and E1-E5 (all
      seed 42) check on identical films;
    - stratified so the share of films with revenue is kept (rounded).
    - **Corrected 2026-11-14:** training's validation frame has **5,144** films with a target. The 95,216 in the data
      report counts every film in the split. A 2,000-film check is therefore about 2.5 times cheaper, not 50.
  - **`full_validation`:**
    - When training ends, `train()` loads the best weights into the model it holds and scores them once on the whole
      validation frame.
    - It writes `full_validation` (`step`, `films`, `val_loss`, revenue and score metrics) and `stopped_reason` into
      `best/state.json`, in a commit that replaces only that file.
    - With no sample, the best check's own metrics are copied instead. That happens only when the check's recorded
      `val_films` was the whole frame, so a run resumed under a changed `val_sample` is still scored for real.
    - A rerun skips the pass when `full_validation` exists for the current best step. A session lost during the pass
      redoes only that pass.
    - The summary and the W&B summary carry it too. A NaN-stopped run that has a best still gets it.
  - **`compare_runs`** (`python -m llm_movies.eval.compare_runs --runs a,b --out reports/experiments.md [--repo ...]`):
    - one markdown row per run, with every metric column taken from `full_validation`;
    - "check pinball" is the sample metric, shown for reference only;
    - a run still training shows dashes instead of its sample numbers, and a missing run shows "no best checkpoint".
  - **Reviews:**
    - The code-reviewer's main point was the `val_films` guard, now added. A follow-up review found nothing new.
    - The test-auditor added a stale-`full_validation` test and ran 5 mutants through monkeypatch. I ran the 2 inline
      mutants it couldn't (no stale-step check; compare falling back to sample metrics), and both turn tests red.
    - 1267 tests pass, ruff check is clean.
  - **Unverified:** loading the best weights into a CUDA fp16 model at the end of a run is covered only by CPU tests.
    On the first T4 run, check that `best/state.json` gets `full_validation`.
  - Next: Task 12 Step 2, the owner's E1 runs on Kaggle (`e1_small`, `e1_large`).
- 2026-11-11: **Phase 2 Task 11 Step 4 done: the T4 smoke run worked and resumed. Task 11 is complete.**
  - The owner ran `configs/train/smoke.yaml` on a T4.
  - All 9 of 9 steps finished, with validation at steps 2, 4, 6, 8 and 9.
  - Validation loss fell at every check (final 0.355), so `best` moved each time.
  - `best` and `last` are at `runs/smoke/{best,last}/` in `Tsomaros/llm-movies-runs`. The earlier note saying
    `smoke/` was wrong: `checkpoint.py` saves under `runs/<run_name>/`.
  - `last` holds the optimizer, scheduler, GradScaler and RNG states.
  - **Resume:** the owner restarted the runtime after step 2. The Hub commits then go step 2 once, then 4, 6, 8, 9. A
    fresh start would have saved step 2 again, so the run resumed from `last`.
  - Step-9 metrics only show the code runs end to end: revenue mae_log 3.67, 80% coverage 0.73 at a p10-p90 width of
    about 87,000×; score mae 0.87, coverage 0.85.
  - **Validation is slow.** Each check takes about 9-10 min, because `load_frames` doesn't sample the validation split
    and every check scores all of 2016-2018. For `base` (about 270 steps, 4 checks) that adds roughly 40 min per run.
    Open question before E1-E5: validate on a fixed sample?
  - Next: Task 12, only when the owner asks.
- 2026-11-11: **Phase 2 Task 11 Steps 1-3 done. Step 4, the T4 smoke run, is the owner's.**
  - **Step 1, token lengths** (Qwen3-0.6B-Base tokenizer, open-split films, nothing hidden): p50 663, p90 759,
    p99 829, max 924.
    - `max_len` is **832** (`recommended_max_len`) in `base.yaml`, and `test_repo_base_config_loads` pins it.
    - **This exceeds the spec target of 400-600 tokens** (p50 663).
    - About 1% of prompts are cut on the right, which trims only the end of the synopsis.
    - Not measured: lengths with `anonymize` on (E2).
    - The CPU rule stays defined at 768. At 832 it should still fit under 300 s (an estimate, in
      `docs/model_choice.md`).
    - The local HF token was refreshed (`hf auth login --force`).
  - **Configs** (`configs/train/`):
    - `base`: Qwen3-0.6B-Base, `sample_frac` 0.3.
    - `smoke`: base with `sample_frac` 0.01 and `epochs` 1, under its own run name.
    - `e1_small`, `e1_large` (1.7B-Base), `e2_anon`, `e3_notext`, `e4_noleak`, `e5_nosource`.
    - All of them use `extends: base.yaml`. Tests pin each one as base plus only its own change.
  - **Notebook** `notebooks/11_train.ipynb`, for Kaggle or Colab T4:
    - The install is `uv sync --frozen --extra train --extra cu132`, not the platform's torch as the plan said, matching
      the cpu/cu132 constraint.
    - One secrets cell per platform. `HF_TOKEN` and `WANDB_API_KEY` are both required, because the cell fails on a
      missing secret.
    - A `CONFIG` dropdown, which you edit by hand on Kaggle, then `llm_movies.model.train --config {CONFIG}`. Rerunning
      resumes.
    - The README warns never to edit a config mid-run.
  - **Unverified on Kaggle:** its driver must report CUDA ≥ 13.2, and the clone goes to `/content`.
  - **Reviews:** code-reviewer found that a smoke run under `base` would make the real `base` run stop at once
    (hence `smoke.yaml`), and that the "optional" W&B secret wasn't optional. Both are fixed. The test-auditor added
    negative cases for the widened launcher-line patterns. 1246 tests pass, ruff is clean.
  - **Step 4 (owner, T4):** run `smoke`. Check that `best` appears under `smoke/` on the Hub, force a restart, and check
    that it resumes.
- 2026-11-11: **Checked whether Qwen3's low director recall is a probe artifact: it is real.**
  - The Colab CSVs weren't downloaded, because the Hugging Face token cached on Windows is rejected (401). Qwen3-0.6B-Base,
    which is public, was rerun on CPU through `make_generator` instead.
  - Its answers are clean text, with no leading newlines.
  - Batch 1 and batch 8 give the same answers in fp32 and in fp16.
  - The name match is correct, including TMDB's "Bong Joon Ho" against "Joon-ho".
  - The model got only 3 of 16 famous 2019-20 films right; the rest were confident wrong names.
  - No code change. The decision and cutoff stand. Details are in `docs/model_choice.md`.
  - **Open:** refresh the local HF token if the CSVs or a Llama comparison are wanted.
- 2026-11-11: **Phase 2 Task 10 done.** The owner reran notebook 03 (no re-pull) with `base_model_cutoff` 2025-04-29.
  - `reports/data_summary.md` now shows test 281,690, holdout 57,017 and later 28,744. These add up to the old test count
    of 367,451.
  - The open splits are unchanged.
  - **Task 11 waits for the owner's go-ahead.**
- 2026-11-11: **Task 10 Step 4 decided; Step 5's code part done. Waiting on the Colab rebuild.**
  - Decisions (Dimitris):
    - Qwen3: `Qwen/Qwen3-0.6B-Base` (base config, E1 small) and `Qwen/Qwen3-1.7B-Base` (E1 large).
    - `base_model_cutoff: 2025-04-29`, Qwen3's release date: Qwen3 states no cutoff and the probes were noise.
    - CPU rule relaxed to batch 13 in ≤ 300 s, which both sizes pass. Also written into the E1 rule in Task 12.
    - Gemma 3 is not rerun.
  - Code and tests:
    - `config/data.yaml` sets the cutoff. `DataConfig`'s code default stays None.
    - Updated tests: `tests/test_config.py`, `tests/data/test_build_db.py` (Oppenheimer is now test, Secret Wars later)
      and `tests/features/test_computed.py`.
    - `configs/train/base.yaml` uses the `-Base` id.
    - 1187 tests pass and ruff is clean. code-reviewer approved, on Windows.
  - **Next (owner, Colab CPU):** rerun `notebooks/03_build_and_export.ipynb` with no re-pull, so the Hub data gets the
    new splits.
    - The old build has holdout 0, and `export` stops on a split mismatch.
    - Check that `reports/data_summary.md` shows holdout > 0 and test < 367,451. The reviewer estimates holdout covers
      about 2025-04-30 to 2026-05-06.
    - Then commit `data: base model chosen, cutoff set, holdout built` with the report, tick Step 5, and start Task 11.
- 2026-11-11: **Phase 2 Task 10 Steps 2-3 done: the Colab probe and speed results are in `docs/model_choice.md`.**
  Step 4, the decision, is open; Step 5 waits for it.
  - Findings:
    - The gemma-3-1b-pt probe run is invalid. It scored 0 on everything, most likely from fp16 overflow in
      `make_generator`; the CSVs were not inspected.
    - The Qwen empirical cutoffs are noise. 2019-20 recall is only 0.006 and 0.022, and 1.7B's computed date, 2026-06,
      is after Qwen3's release on 2025-04-29.
    - No model meets the 20 s CPU rule. Qwen3-0.6B takes 100 s for batch 13 on 1 thread at 768 tokens.
    - Gemma 4 E2B gave no speed result, probably out of memory.
    - Only Qwen3 offers one family with two measured sizes.
  - Open questions for Dimitris:
    - the family and its two sizes;
    - the Qwen cutoff source (release date or another rule);
    - the CPU rule, which also feeds the E1 rule in Task 12;
    - whether to rerun Gemma 3 in fp32.
- 2026-11-11: **Hotfix 2: notebook 12 was killed on Colab's CPU runtime (most likely out of memory).** `cpu_forward_seconds` ran
  the full CausalLM, so it also computed 13 × 768 × vocab logits (~6 GB for Qwen) next to an fp32 model and a second int8 copy.
  It now times only the decoder (`get_decoder()`, `last_hidden_state`, `use_cache=False`), the same path as `MoviePredictor`
  (serving never computes logits), loads with `dtype=float32, low_cpu_mem_usage=True` and quantizes in place when loading by id.
  `torch_dtype=` → `dtype=` in speed and probe. Gemma 4 E2B (~20 GB in fp32) will likely still not fit in 12 GB. 1186 tests pass,
  ruff clean; code-reviewer (approved) and test-auditor run on Windows. Deferred: README wording on the lm_head (tied embeddings
  free nothing) and the notebook-12 FLOP estimate is now an upper bound.
- 2026-11-11: **Hotfix after the first Colab run of notebook 10 part B ("No GPU found" on a T4).** `uv sync --frozen` installed the
  dev group's CPU torch and no `train` extra, so the uv venv hid the T4. Torch is now two conflicting extras: `cpu` (CPU index)
  and `cu132` (CUDA 13.2 index; torch 2.15.0 has CUDA wheels only for cu132/cu134, and 13.2 still supports the T4's sm_75).
  Always pair `train` with one of them. **Local/Windows and the Linux side: `uv sync --extra train --extra cpu`** (Linux:
  `UV_PROJECT_ENVIRONMENT=$HOME/.venv-llm uv sync --extra train --extra cpu`); CI runs `uv sync --locked --extra train --extra cpu`.
  Notebook 10: part A (CPU) `uv sync --frozen` (no torch); part B (T4) `uv sync --frozen --extra train --extra cu132` in its own
  cell, after the shared Setup and Secrets cells. Notebook 12: `--extra train --extra cpu`. Before part B, check `!nvidia-smi`
  in a scratch cell: "CUDA Version" must be ≥ 13.2 (unverified for Colab's T4 driver); don't re-run part A's sync after part B's
  (exact sync removes torch). The plan's Global Constraint on torch is updated. 1182 tests pass, ruff clean; code-reviewer and
  test-auditor run on Windows. The owner's db/movies.db upload (part A) already succeeded.
- 2026-11-11: **Phase 2 Task 10 prepared** (Step 1 done; Steps 2–5 wait for Colab runs and the decision). `docs/model_choice.md`
  holds the official model-card facts: Qwen3-0.6B-Base and 1.7B-Base (Apache 2.0, ungated, cutoff not stated; the plan's ids
  without `-Base` are the post-trained models), Llama-3.2-1B (gated, cutoff December 2023, "Built with Llama" and a "Llama..." name
  for a published fine-tune), gemma-3-1b-pt (gated, cutoff August 2024 per Google's model card, not the HF card; Prohibited Use
  Policy passes to Space users), SmolLM2-1.7B (Apache 2.0, cutoff not stated), and Gemma 4 E2B as a conditional last candidate
  (Apache 2.0 per its licence page, cutoff January 2025, 5.1B with embeddings; base vs instruct and gating unconfirmed).
  To run: `notebooks/10_model_choice.ipynb` part A on a CPU runtime once (`db_snapshot ensure`: builds movies.db from the raw shards
  and uploads `db/movies.db` to the raw repo unless present; run it only after the pull is complete; the snapshot is frozen and
  for Task 10 only), part B on a T4 (`db_snapshot fetch` then `model_choice run`: director probe 150 films per quarter 2019Q1–2026Q2
  with ≥100 votes, revenue probe 300 validation films, per candidate in `model_choice.CANDIDATES`; CSVs and `summary.json`/`.md`
  go to `model_choice/` in the raw repo after each model; rerun resumes, a gated model without an accepted licence is recorded as
  failed and the rest go on), then `notebooks/12_speed.ipynb` on a CPU runtime (one `eval.speed` call per candidate, max_len 768,
  batch 13). Before part B, accept the Llama 3.2 and Gemma 3 licences on huggingface.co with the HF_TOKEN account. Run times in
  `notebooks/README.md` are estimates only; by its own arithmetic a batch-13 pass may take minutes on Colab's 2 vCPUs even for
  0.6B (the plan's target is ≤ 20 s), and Gemma 4 E2B may not fit in 12 GB. `tests/test_notebooks.py` now covers notebooks 10 and
  12 (HF_TOKEN only, one chain per runtime part). 1157 tests pass, ruff clean; code-reviewer (opus, approved; it caught the missed
  Gemma 3 cutoff) and test-auditor run on Windows. Next: run notebook 10 (A, then B) and 12, fill the measurements table, decide.
- 2026-11-10: **Phase 2 Task 9 done** (probes, token lengths, CPU speed) in `llm_movies/eval/probe.py`, `tokens.py`, `speed.py`.
  CLIs for Task 10: `python -m llm_movies.eval.probe revenue|director --model ID --db PATH --n N --out CSV [--min-votes 100]`,
  `python -m llm_movies.eval.tokens --tokenizer ID [--parquet DIR]` (the `--parquet` default `data/export` is a guess; pass it),
  `python -m llm_movies.eval.speed --model ID --max-len L` (prints the thread count). Leak rules: the revenue probe reads only
  validation films; the director probe may sample any year but returns only id, title, year, release date and director (vote_count
  is a sampling filter only); `token_lengths` reads only films.parquet. `parse_money` takes the first number (atomic, so "1980s" is
  1980), a range's first number inherits the tail's scale ("$1.2-1.5 billion" → 1.2e9), scale words K/M/mn/mil/mln/B/bn/thousand/
  million/billion; other currencies are read as dollars. `known` = within 2× of nominal dollars (revenue_2025 × CPI ratio).
  `empirical_cutoff` = last day of the last quarter with recall ≥ 50% of the 2019–2020 mean (a lucky tiny quarter can push it
  late; use enough films per quarter). `make_generator` is greedy, batched, left-padded, new text only. `cpu_forward_seconds` = median
  of 5 int8 forward passes after one warm-up (`torch.ao.quantization.quantize_dynamic`, deprecated but working on torch 2.15).
  1106 tests pass, ruff clean; code-reviewer (two Important findings, fixed in one round) and test-auditor (it found the "1980s"
  backtracking bug) run on Windows. Deferred: "85 and 3 billion" parses as 8.5e10 ("and"/"or" connectors). Next: Task 10 (Colab).
- 2026-11-10: **Phase 2 Task 8 done** (training loop) in `llm_movies/model/train.py`: `train(cfg, data_dir, *, api, lm, tokenizer,
  device, out_dir) -> summary` (`best_metric`, `best_step`, `stopped_reason` patience/epochs/nan, best validation metrics) and the
  CLI `python -m llm_movies.model.train --config configs/train/<name>.yaml [--data DIR]`, which downloads only films.parquet
  (`hub_store.download_file`). AdamW with LoRA/head groups, 3% warmup then cosine, grad accumulation (loss divided by the real
  micro-batch count), fp16 autocast + GradScaler on CUDA. Validation metric = `pinball_loss` over the whole validation frame ÷ 2
  (scaled units), every round(steps_per_epoch × eval_every) steps; `best` on improvement, `last` every `save_every_steps` and after
  each check, using the Task 7 API as recorded (plus an optional `device=` keyword added to `load_checkpoint` and `build_predictor`,
  so a CPU device stays fp32). Resume maps `state["step"]` to epoch = step // steps_per_epoch and skips the epoch's done steps; the
  test shows an interrupted + resumed run equals an uninterrupted one (weights and AdamW state, atol 1e-5). Non-finite loss skips
  the step; 10 in a row stop with "nan". The head now runs with autocast disabled (fp32 on CUDA, per the Global Constraints).
  W&B (only with `WANDB_API_KEY`) resumes by `id=run_name` and gets `train_loss`/`lr` per step; `metrics.jsonl` (validation lines
  only) goes to `outputs/<run_name>/` (git-ignored). `tests/hub_fakes.py` holds the shared fake Hub. 1051 tests pass, ruff clean;
  code-reviewer (opus, spec ✅, approved; five pre-Kaggle fixes came from its minors in one round) and test-auditor run on Windows.
  Untested here: the CUDA/GradScaler path and real Hub calls; the first T4 run should be a short smoke run that checks non-zero LoRA
  gradients, a `last` upload and a resume. Reusing a `run_name` merges W&B runs. Next: Task 9.
- 2026-11-10: **Phase 2 Task 7 done** (Hub checkpoints and resume) in `llm_movies/model/checkpoint.py`. API for Task 8:
  `save_checkpoint(model, scaler, cfg, state, kind, api=None, *, optimizer=None, grad_scaler=None, lr_scheduler=None)`,
  `load_checkpoint(cfg, kind, api=None, lm=None) -> (model, scaler, state) | None`, `resume_state(cfg, api=None)`,
  `restore_training_state(cfg, api=None, *, optimizer=None, grad_scaler=None, lr_scheduler=None) -> bool` (downloads `last` again).
  Checkpoints go to the private model repo `runs/<run_name>/{best,last}/` in one commit with `state.json` last; a folder without
  `state.json` counts as absent, so a killed upload leaves the previous checkpoint loadable. `last` must get an optimizer and an lr
  scheduler, and always writes `grad_scaler.pt` (marker `{"enabled": False}` without a scaler), so no stale file survives an add-only
  commit; pass `grad_scaler=None` on CPU, not a disabled GradScaler. RNG states (torch CPU/CUDA, numpy, Python) load with
  `weights_only=True`. `hub_store` gained keyword-only `repo_type="dataset"` on `ensure_private_repo`, `upload_files`, `download`,
  `read_json` (dataset behaviour unchanged; a public model repo is refused). Only fakes are tested; the real Hub path is unexercised
  until the first Kaggle run (the reviewer checked every call against huggingface_hub 2.4.0). 1040 tests pass, ruff clean;
  code-reviewer (spec ✅, approved; the same-file-set fix came from its minor, one fix round) and test-auditor run on Windows.
  Next: Task 8.
- 2026-11-10: **Phase 2 Task 6 done** (metrics) in `llm_movies/eval/metrics.py`. `revenue_metrics`: `mae_log`, `within_2x`
  (ln 2 inclusive, the app's 1e-12 tolerance), `coverage_80`, `median_width_x`, `pinball`. `score_metrics`: weighted `mae`,
  unweighted `coverage_80` (as the app), `median_width`, weighted `pinball`. Pinball follows `model/loss.py`. Differences from the
  app's `ml/eval/metrics.py` (the plan's formulas win): score width is a median (app: mean), `mae` is the weighted one (app reports
  both), pinball is new, key names differ. Empty input raises. 1027 tests pass, ruff clean; code-reviewer (spec ✅, approved) and
  test-auditor (it made median vs mean, boundaries and weights distinguishable) run on Windows. Next: Task 7. Deferred minors: NaN or
  negative weights and NaN predictions are not rejected; no shape-mismatch tests.
- 2026-11-10: **Phase 2 Task 5 done** (dataset, collator, sampler) in `llm_movies/model/data.py`. `load_frames` takes the data dir
  or the films.parquet path and reads only films.parquet (train sampled by `sample_frac` with the seed, validation; calibration
  dropped). `TargetScaler` (mean 0 / std 1 fallback for empty or near-constant columns, JSON round trip). `FilmDataset` renders with
  `drop_groups` plus, in training only, each `RANDOM_HIDE` group at `hide_p` from an RNG seeded by (seed, epoch, index); missing
  targets are 0.0 with mask False and a missing score weight is 0.0. `Collator` works on a deep copy of the tokenizer, forces right
  padding and right truncation whatever its default, falls back to `pad_token = eos_token` (Llama 3.2 base has no pad token), and
  raises on an empty text or an all-zero attention row. `revenue_sampler` oversamples revenue films with replacement. 1021 tests pass,
  ruff clean; code-reviewer (spec ✅, approved; the pad-token and NaN-std fixes came from its minors, one fix round) and test-auditor
  (three passes) run on Windows. Next: Task 6. Deferred minors: `TargetScaler.fit` does not coerce with `pd.to_numeric` as
  `transform` does.
- 2026-11-10: **Phase 2 Task 4 done** (predictor). `llm_movies/model/config.py`: `TrainConfig` (frozen, plan defaults) and
  `load_train_config` (`extends` relative to the child file; unknown keys and `drop_groups` outside `HIDEABLE` raise).
  `llm_movies/model/predictor.py`: `MoviePredictor` / `build_predictor` (frozen base, LoRA r16/a32/0.05 on all-linear, fp32 LoRA and
  head, fp16 base + gradient checkpointing on CUDA only) and `trainable_parameters`. Pooling runs the PEFT-wrapped decoder
  (`get_base_model().get_decoder()`, last hidden state at `attention_mask.sum(1) - 1`) instead of the full LM with
  `output_hidden_states`, so no vocab-size logits are computed; a test proves gradients reach every LoRA B. `configs/train/base.yaml`
  holds `run_name: base` and `base_model: Qwen/Qwen3-0.6B-Base` (chosen in Task 10; Task 11 finalises the rest). Config tests are in
  `tests/model/test_train_config.py` (name clash with `tests/test_config.py`). 1002 tests pass, ruff clean; code-reviewer (spec ✅,
  approved) and test-auditor (it tightened the truncation test with a padded mixed batch) run on Windows. Next: Task 5.
  For Task 5: pooling assumes right padding; the tokenizer/Collator must force `padding_side="right"` (Qwen may default left) and test
  it; an all-zero mask row would pool position -1. For Task 10: Gemma 3 overflows in fp16, and a multimodal Gemma keeps `hidden_size`
  on `text_config`. For the first T4 run: the CUDA path is untested here; check one backward step gives non-zero LoRA gradients.
- 2026-11-10: Invisible whitespace in `tests/prompt/test_render.py` parametrize lists replaced by `\u` escapes. `main` pushed at ec927ab;
  CI passed with the CPU-torch install (https://github.com/Tsomaros/LLM_Movies_Model/actions/runs/42847879800). `gh` is not installed
  on the Windows PC; CI was checked through the public GitHub API.
- 2026-11-10: **Phase 2 Task 3 done** (training dependencies, tiny-model fixtures, head and loss). `train` extra: transformers 5.22.2,
  peft 0.21.2, accelerate 1.15.0, wandb 0.31.0 (lower bounds); dev group: CPU torch 2.15.0 from the explicit PyTorch CPU index, tokenizers
  0.23.3. CI runs `uv sync --locked --extra train`; on Windows run `uv sync --extra train` once (the Linux side likewise with its own
  env). `llm_movies/model/`: `QUANTILES`, `TARGETS`, `QuantileHead` (non-crossing by softplus), `pinball_loss` (computes in fp32; masked
  cells are fully ignored, even NaN/inf weights or targets; an empty target adds 0 with zero gradients). `tests/conftest.py` builds
  `tiny_tokenizer`/`tiny_lm` offline (transformers imported inside the fixtures). The test-auditor's tests found the masked NaN-weight
  bug, fixed in one round. 990 tests pass, ruff clean; code-reviewer (spec ✅, approved) and test-auditor run on Windows. Next: Task 4.
  Deferred minors: the loss also normalises revenue weights (same result while they are 1); no test feeds an fp16 `pred`. During the
  run the Unicode-whitespace parametrize values in `tests/prompt/test_render.py` were blanked once by an unknown cause (restored; the
  committed file is intact). If it recurs, suspect a hook or editor rewriting that file.
- 2026-11-10: Free-text anonymize fix (E2): with `anonymize=True`, this film's own person, company, distributor and franchise names
  inside tagline and synopsis become the same labels as the record lines (case-sensitive, `(?<!\w)`/`(?!\w)` bounds, one regex longest
  first, after outcome masking and before the 120-word cut). Names of hidden groups still get the next free numbers. A distributor or
  franchise name equal to a company name reads `Company N` in free text. Surnames alone and other-case mentions are not replaced.
- 2026-11-10: **Phase 2 Task 2 done** (hideable groups, anonymized names). `render(r, hidden, *, anonymize=False)`; `HIDEABLE` adds
  `source_strength`, `release_type`, `themes` to `OPTIONAL_GROUPS`; `RANDOM_HIDE` adds only `source_strength`. Anonymize: people
  `Person N` by first appearance (director, writers, cast, producers, composer), identified by cleaned name since ResolvedFilm has no
  ids, each nameless person a fresh number; companies `Company N`; distributor `Distributor`; franchise `GENERIC_FRANCHISE = "Franchise"`;
  records unchanged (free text: see the anonymize fix above). Default render unchanged (goldens untouched). 964 tests pass, ruff clean; code-reviewer (spec ✅, approved) and
  test-auditor run on Windows. Next: Task 3. Deferred minors: the render docstring's long anonymize paragraph, and an inline
  distributor lambda in the f-string.
- 2026-11-10: Task 1 follow-up fix: `mask_outcomes` makes one pass with one combined regex, longest first, plus compounds of
  Oscar / Academy Award / Golden Globe followed by `[\s-]winning`, so "Academy Award-winning" becomes one `[…]` (no fragment).
  `OUTCOME_PATTERNS` and the report audit are unchanged. Deferred: the compound allows a single space or hyphen only (an en dash or
  "Oscar - winning" still leaves "winning"; the audit has the same limit).
- 2026-11-10: **Phase 2 Task 1 done** (mask outcome phrases). `llm_movies/prompt/outcome_words.py` holds `OUTCOME_PATTERNS` (the
  report's nine audit patterns, without the dollar figure), `MASK = "[…]"` and `mask_outcomes`. `render` masks tagline and synopsis
  (synopsis before the 120-word cut); `report` audits with the shared patterns plus its own dollar pattern (row order unchanged).
  Golden prompts unchanged. 953 tests pass, ruff clean; code-reviewer (spec ✅, approved) and test-auditor run on Windows.
  Deferred minors: `report.py` keeps the dollar row third by slicing `OUTCOME_PATTERNS[:2]`; `test_outcome_words.py` imports
  `film` from `test_render.py`.
- 2026-11-10: **Phase 2 plan written:** `docs/superpowers/plans/2026-11-10-phase2-model-and-training.md` (13 tasks). Next: Task 1 (mask outcome phrases).
  Decisions with the owner: outcome phrases in tagline/synopsis are masked in `render` (training and serving; dollar figures not masked);
  `source_strength` kept, randomly hidden in training, tested in E5; leak-check experiment E4 drops `release_type`, `distributor`, `themes`.
- Phase 1 plan: `docs/superpowers/plans/2026-11-05-phase1-foundation-and-data.md`. **Phase 1 done** (2026-11-10). Next: write the Phase 2 plan from the roadmap.
- 2026-11-07: two dry-run fixes done. TMDB pull runs `--workers` threads (default 8) under the shared 40/s cap, writing in id
  order on the main thread (expected about 10 h instead of 70 h; the real speed is unmeasured until Colab). `release_type`: the
  country is chosen among those with a cinema entry (US first), only TMDB type 4 counts as streaming-only and only from 2007;
  types 5/6 (DVD, TV) never do. Festival-first left as is: the 7 cases in the 50-film local sample were real festival films.
  Known trade-off: a US digital-only film with a cinema run abroad now reads limited/wide, not streaming-only. 429s back off
  per thread (no shared pause); revisit if Colab shows many 429s.
- 2026-11-08: the first Colab run stopped on a TMDB token with a trailing line break, and the `requests` error printed the whole
  token. Tokens are now stripped, and a token with inner whitespace or odd characters is refused without being shown (TMDB in
  `TmdbClient`, Hugging Face in `hub_store.hf_token`). The owner was told to replace the exposed TMDB token and clear the Colab output.
- 2026-11-08: Task 15 Step 2 done: the 500-film sample went through notebooks 01, 02 and 03 in Colab (export pushed, report written; 01 took under a minute). It held 499 train and 1 calibration film, so real locked films are still unexercised until the full run.
- 2026-11-10: Task 15 Step 4 done: the owner ran the full pull (5 chunks took 16 min, about 26 films/s), full Wikidata (no
  pruning) and notebook 03 in Colab; report copied to `reports/data_summary.md`. 1,081,349 films: train 565,901, calibration
  52,781, validation 95,216, test 367,451, holdout 0 (old build: `base_model_cutoff` was unset then; it is now 2025-04-29 and notebook 03 must rebuild, so every film from 2019 on is test until
  Phase 2 picks the base model and sets it, which needs a rebuild). Open-split dataset: 34,354 films (17,312 with revenue,
  28,907 with score at min_votes 50). The Hub manifest was not checked from here; the owner reported the run complete.
- 2026-11-10: Task 15 Step 5: the owner kept `min_votes` at 50 (28,907 score films; 20 would give 53,217 with noisier scores,
  and scores are already weighted by log1p(votes)). No config change, so the exported dataset on the Hub stands as built.
- Open for Phase 2: the outcome-word audit finds about 170 open films whose tagline or synopsis mentions awards or box office
  (Oscar 56, award-winning 50, Academy Award 24, gross 14, blockbuster 12, box office 8, Golden Globe 4, record-breaking 4,
  smash hit 2; overlaps possible), plus 223 dollar figures (mostly plot money in the samples). Decide whether the prompt masks
  them or those films are dropped.
- Execution: subagent-driven, alternating between Claude Code (Windows) and the claude.ai cloud session.
- Remote `origin` is https://github.com/Tsomaros/LLM_Movies_Model (public). `main` pushed at 55e5fc5; its CI run passed
  (https://github.com/Tsomaros/LLM_Movies_Model/actions/runs/42055792680).

## Done
- 2026-11-05: Design spec approved; roadmap and Phase 1 plan written; `CLAUDE.md`, the three subagents (adapted) and the
  ponytail skill copied from the Movie Prediction App.
- 2026-11-05: Task 1 (repo foundation, config, CI). 7 tests pass, ruff clean, code-reviewer and test-auditor run on Windows.
- 2026-11-05: Task 2 (money conversion, FX file). 21 tests pass, ruff clean, code-reviewer and test-auditor run on Windows.
- 2026-11-05: Task 3 (resumable JSONL shards, manifest). 38 tests pass, ruff clean, code-reviewer and test-auditor run on Windows.
- 2026-11-05: Task 4 (TMDB IDs, client, resumable pull, CLI), except Step 5 and the real fixture. 77 tests pass, ruff clean,
  code-reviewer and test-auditor run on Windows.
- 2026-11-05: Task 5 (source categories, award families). Code-reviewer and test-auditor run on Windows.
- 2026-11-05: Task 6 (Wikidata queries, parsers, CLI), including its live check. 220 tests pass, ruff clean, code-reviewer and
  test-auditor run on Windows.
- 2026-11-05: Task 7 (cleaning rules). Code-reviewer and test-auditor run on Windows. `clean.py` was written before its tests,
  and RED was recorded by moving it aside; the audit found the tests were not shaped to the code.
- 2026-11-05: Task 4 finished with a token: live `--limit 50` check passed (50 pulled, 0 missing, manifest unchanged, rerun added
  nothing), real Joker fixture saved. `clean_movie` ran on 150 real films with no crash or shape mismatch, which led to the two
  rule changes in Notes.
- 2026-11-05: Task 8 (SQLite build, real mini shard set from `python -m tests.fixtures.make_mini`, 37 films, 328 KB). Code-reviewer
  and test-auditor run on Windows.
- 2026-11-05: Task 9 (time splits, test lock, `movies.split` filled). 422 tests pass, ruff clean, code-reviewer and test-auditor
  run on Windows.
- 2026-11-05: Task 10 (form contract). 550 tests pass, ruff clean, code-reviewer and test-auditor run on Windows.
- 2026-11-05: Task 11 (leak-safe track records). 576 tests pass, ruff clean, code-reviewer and test-auditor run on Windows; the
  audit ran 56 leak mutants against the tests.
- 2026-11-05: Task 12 (the six computed inputs, `source_sitelinks` column). 609 tests pass, ruff clean, code-reviewer and test-auditor
  run on Windows.
- 2026-11-05: Task 13 (resolve and render, `collection_name` column, three golden prompts). 727 tests pass, ruff clean, code-reviewer
  and test-auditor run on Windows; the audit ran 104 mutants.
- 2026-11-05: Task 14 (dataset export, data report). 759 tests pass, ruff clean, code-reviewer and test-auditor run on Windows. CLIs run
  on the mini DB; no real Hugging Face push yet.

## Open questions
- Possible outcome leaks to check before training (Phase 1 data report or Phase 2): `source_sitelinks` is today's count, so a hit film
  can raise its source's page count (try training with and without it); `release_type` comes from TMDB entries added after release,
  so "streaming-only" may mark films that flopped; `distributor` may list later rights holders; themes and keywords may hold outcome
  words such as "cult". `money_suspect` comes from revenue, so History's budget filter has a small leak for films under 182 days old.
- `source_sitelinks` at serving: the form has no such field, so user forms print "not given" while training films show a count. Drop the
  line in Phase 2 if the leak check or this skew says so.
- `min_votes` for the score target: decided from the Phase 1 data report.
- Base model and its training cutoff: decided in Phase 2.

## Notes
- Reviews run on the cloud side are marked "(cloud review, no hooks)".
- `tests/test_config.py` pins `config/data.yaml` to the defaults (`min_votes` 50; `base_model_cutoff` 2025-04-29 in the yaml, None as the code default). Changing either
  value (Task 15 Step 5, Phase 2) turns `test_defaults_match_spec` and `test_yaml_and_dataclass_defaults_agree` red, so those
  tests must change with the decision.
- Layout decisions in Task 1: flat `llm_movies/` at the repo root (`module-root = ""`), `requires-python = ">=3.11,<3.12"`,
  no `paths` key in `config/data.yaml` until a task needs one.
- `data/fx_usd_annual.csv` is FRED only, so years FRED lacks are absent and `to_usd` returns None: everything before 1971,
  INR and ESP before 1973, CNY and KRW before 1981, EUR before 1999. 2026 repeats 2025. FRF, DEM, ITL and ESP have no annual
  FRED series, so up to 1998 they are means of the monthly series. No build script is kept; the file's comments name the series.
- Shards: a record counts only once its newline is written. `ShardWriter` repairs the last line on open, so call `done_keys`
  either way but expect the unterminated or invalid tail to be refetched. An invalid line anywhere else raises `ValueError`:
  that is real corruption and the file needs a look. `done_keys` raises `KeyError` on a record without the key.
- TMDB pull: a `--limit` run never marks the chunk done (`pull_chunk(..., mark_done=False)`), or the real run would skip the
  rest of it. `missing_NNNN.txt` holds `{"id": n}` lines written through `ShardWriter`. Manifest stats are chunk totals. Only a
  real 404 records a film as missing, and that file is never rechecked; failures raise and the chunk resumes. A huge
  `Retry-After` is honoured in full. `tests/fixtures/tmdb_ids_sample.json.gz` mixes real export lines with hand-written ones,
  because the real export has no `adult: true` lines.
- `llm_movies/data/awards.json` is a Wikidata snapshot from 2026-11-05 (171 items, no build script kept). It lists film categories
  directly because Task 6 fetches no parent classes, so `award_family(qid, set())` matches on the item. The Golden Globes and
  SAG root classes also cover TV, so passing real parent classes later would over-count; drop those roots if that ever happens.
- Source categories are a best guess when Wikidata is vague: "literary work", "written work" and "book" map to novel, so a
  non-fiction source or a musical typed that way reads as novel, and a sequel linked to a film by P144 reads as remake. Films based
  on a character, a short or a documentary fall back to keywords. Check `source_material` in the Phase 1 data report.
- Wikidata: queries use UNION branches and skip deprecated statements. `AwardEvent.when` is the last day an imprecise date could
  mean, so Task 12 must count only `when < as_of` and skip None. Every id gets a record (`facts: null` means no item). `run_pull`
  prints nothing until it ends, about 2 hours on the full id list. Live check, 500 ids: chunk 0 had 495 items, distributors 92%,
  P144 37%; top-500 popular films had 457 items, distributors 82%, P144 39%.
- Cleaning overrules the plan in two places. `distributor` is the family name from a table of majors in `clean.py` (Joker gives
  "Warner Bros."; 81% of the 952 sampled Wikidata films resolve to one), retail storefronts are dropped by exact name, and the
  first sorted label is only the fallback. CPI always uses the release year, because Wikidata figures are often undated.
- Real data changed two more cleaning rules. `festival-first` needs the premiere at least 30 days before the first theatrical date
  (`FESTIVAL_GAP_DAYS`): TMDB counts red-carpet premieres, so the plan's rule read 46% of popular films as festival-first (12% of
  150 real films now). A streamer counts as distributor only on a streaming-only, limited or festival-first film from its start
  year on, because Wikidata lists later rights (Inception read Netflix; Fight Club read Disney+, a time-travel leak).
- Distributor is still weak, so check it in the Phase 1 data report: None on 32% of popular films and 80% of old ones, and with only
  regional labels on Wikidata a Hollywood film reads a regional name (Inception reads InterCom). SkyShowtime and a bare
  "Prime Video" are not gated as streamers. `FilmFacts.from_dict(None)` raises, so skip `facts: null` records.
- SQLite build: `awards` has an `award_qid` column beyond the plan, so same-night wins in different categories count separately;
  Task 12 counts rows and skips NULL `when`. A given Wikidata dir without its shard raises, and only `None` builds without
  Wikidata (the CLI always needs both). `--pulled-at` defaults to the latest in the TMDB shards. A record without `id` stops the build.
  Film facts stay in memory, an estimated 1-2 GB on a full build, not measured: check it in Task 15's run.
- Splits: `assign_split(release, cfg, pulled_at)` takes `pulled_at`, beyond the plan's signature, because holdout needs it. `DataConfig`
  rejects non-contiguous year bands. Locked rows sit in `movies` with a plain `split` value, so the lock guards against accidents
  only. For Task 14: the plan's single parquet holds every split, so decide how locked rows are stored and read, keep `later` films
  out of training, and don't print ids or results for test and holdout in the report.
- Form contract: `llm_movies/contract/form.py` copies the app's rules. It differs in returning errors as a list of `{field, message}`,
  loading themes itself, reading blank strings as absent, not crashing on huge integers (the app's `_number` still has that bug),
  and `distributor` being name-only text, not the plan's ref. It owns `CERTIFICATIONS`; `SOURCE_TYPES` stays in `source_material.py`.
- For Task 12: the plan's Review Focus names `test_award_on_release_day_not_counted` under Task 10; it belongs with `prior_awards`.
- Track records follow the app's design. Revenue counts only when not suspect or incomplete, score only with at least `min_votes`
  votes, budget only when not suspect. `n` in the shrinkage formula is the count of usable values, so it can differ from `films`.
  An earlier film's score is its current TMDB rating, not its rating then (same limit as the app). On 200k films: `from_db` 2 s,
  225 MB peak, `person` 47 us; not yet measured at 1.2M, where export calling it 25 times per film is about 25 minutes.
- Computed inputs: `competition` looks forward by design and counts every budget above 100M, suspect or not. `collaborations` skips
  the director among the actors. A database built before Task 12 lacks `source_sitelinks`, so rebuild first. On 200k films
  `Context.from_db` took 13.6 s and 502 MB; estimated about 3 GB at 1.2M, so for serving plan a packed snapshot, not `from_db` on each
  cold start.
- For Task 13, a hard-rule leak: a first film has a collection only because sequels came later, so installment 1 tells the model a
  sequel got made. Drop the franchise when `installment == 1`, in one place used by training and serving alike. A user's known
  franchise with no earlier film then reads as standalone.
- For Task 13: track-record values can be NaN on any record, new or not, when no earlier film has a usable value yet. Print
  "not given" and keep the JSON round trip valid. Also, never run `validate` on training forms. `film_to_form` emits the clean shape directly with TMDB ids and `resolve`
  accepts it. Films with under 5 actors or no themes must still resolve, a `None` release type from `clean.py` needs handling, and
  actors cap at 10, writers and producers at 3.
- Prompts: `llm_movies/prompt/` builds one fixed layout. Extras beyond the spec's layout, because spec 4.2 lists them as inputs: director and
  lead age, collaborations, source strength, wins beside nominations, company records. A first franchise entry shows no franchise, in
  training and serving alike. A nameless person prints `Unnamed` with the real record. A training film's budget ignores `money_suspect`.
  `release_type` None prints "not given", never wide. Regenerate goldens only with `UPDATE_GOLDEN=1`.
- Export and report: `films.parquet` holds train, calibration, validation; `locked.parquet` holds test and holdout; `later` films go in
  neither, beyond the plan's single file. Phase 2 training reads only `films.parquet`. `export` re-checks every film's split against
  the current config and stops on a mismatch, so changing `base_model_cutoff` or the years means a rebuild first. `push` refuses a public
  repo and missing files. `report` uses open splits only, with locked splits as film counts, and audits free text for outcome words; on
  the mini set all patterns show 0, so judge them on the real report. Mini run: 1.9 ms per film resolved, 1.8 KB of parquet per film;
  projected at most 38 minutes and 2.2 GB at 1.2M films, unmeasured at scale. The longest real mini prompt is 2,160 characters; a
  synthetic worst case reached 2,929 with long names.
- Colab and storage (owner's decision, 2026-11-05): no Google Drive. Raw data lives in the private Hugging Face dataset repo
  `Tsomaros/llm-movies-raw`; the final dataset in `Tsomaros/llm-movies-data`. `chunk_size` is 5,000 (241 chunks). `tmdb_pull --all --repo`
  uploads each finished chunk in one commit; Wikidata uploads its shard every 30 minutes and at the end; rerunning a notebook resumes.
  Each notebook's steps are one `&&` chain, so a failed step stops the rest. Guards: the manifest records chunk size and count and `--all`
  refuses a mismatch or a Hub manifest with chunks the local one lacks; the id export on the Hub is never replaced without `--replace`;
  `shard_ids` refuses an unfinished pull unless `--partial`; uploads refuse a public repo and never send `.env`, `.git`, `.venv` or temp
  files. `HF_TOKEN` needs write access. Real Hugging Face calls are only tested against fakes so far; Step 2 is their first real run.
- Local dry run (2026-11-07, 1,000 films from chunks 0 and 5 of the old 50,000-id chunks): 995 movies built, 492 exported, all open
  splits, so the locked path hasn't met real data yet. Raw TMDB is about 48 KB per film for the oldest ids and 6 KB for later ones, so
  at least 7 GB in all. Wikidata had facts for 670 of 1,000 films and 13,816 of 24,408 people.
- For Phase 2: measure real tokens, since digit-heavy person lines may exceed the spec's 400-600. A hidden `release_day` still leaves
  day-based values from the true day, a small conservative skew from serving's 1st of the month. An anonymised-names experiment must
  anonymise the franchise name too. Later franchise names and rare synopsis details can identify a film. `render` imports `resolve.py` and
  so pandas; split `ResolvedFilm` out if the Space needs it lighter. Themes come from keywords TMDB fills in after release.
- For later tasks: `revenue_2025` can be set on a film not yet 182 days old or not yet released, so training filters must
  use `revenue_incomplete`. Pre-1960 films keep their row but have no money, since the CPI file starts in 1960. A missing
  top-level `id` raises on purpose, so decide in Task 8 whether a build stops or skips. `pick_distributor` relies on Wikidata's
  labels being sorted, which `wikidata.py` guarantees.
- For Task 10: `clean.py` holds a local `CERTIFICATIONS`; use one definition.
- `clean_amount` handles NaN, but `pd.NA` raises in `math.isnan`. Convert to plain floats or None before calling it.
