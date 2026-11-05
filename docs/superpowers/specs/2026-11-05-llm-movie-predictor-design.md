# LLM Movie Predictor: Design

Date: 2026-11-05 · Status: draft for review · Author: Dimitris Vlachogiannis (with Claude)

## 1. Purpose and success

**Purpose.** A learning project that becomes a portfolio piece: fine-tune a small open LLM to predict a planned
film's **worldwide revenue** and **audience score** as calibrated ranges, and explain in plain English which inputs
raise or lower them.

**Success means:**
1. An end-to-end pipeline the author built and understands: data (TMDB + Wikidata) → prompts → LoRA fine-tuning with a
   regression head → calibration → faithful explanations → a public demo.
2. An honest comparison with the existing Movie Prediction App's LightGBM and baselines B0–B2 on the same films,
   including how much the LLM's pretraining memory inflates its results. Losing to LightGBM is an acceptable,
   reportable outcome.
3. A public Hugging Face Space and a model card anyone can try and read.

**Relationship to the Movie Prediction App.** This is a fully separate project in its own repo. The app is not changed.
Only a read-only script loads the app's frozen LightGBM models to score shared test films. Merging into the app is a
decision for the end (section 12).

## 2. Decisions summary

| # | Decision | Choice |
|---|---|---|
| D1 | Purpose | Learning first, then portfolio |
| D2 | Model approach | LLM backbone + LoRA + regression head with quantile outputs (approach B) |
| D3 | Output | p10/p50/p90 ranges, calibrated to ~80% coverage, for revenue and score |
| D4 | Explanation | Measured counterfactual effects per input group, written in English by a second LoRA adapter, with a faithfulness check and template fallback |
| D5 | Inputs | App's 14 form fields + 8 new optional fields + 6 computed inputs (section 4) |
| D6 | Data | Own pipeline: as many TMDB films as possible + Wikidata enrichment; filtering decided later |
| D7 | Compute | Data pipeline in Colab (CPU runtime); training and evaluation on Kaggle (long runs) and Colab (experiments); Hugging Face Hub for datasets, checkpoints and models |
| D8 | Serving | Hugging Face Docker Space on free CPU (2 vCPU, 16 GB); model ≤ ~1.5B, 8-bit |
| D9 | Front end | React + Vite copy of the app's creator form (plus new fields) and results layout, served by the Space |
| D10 | Testing honesty | Two test sets (2019→base-model cutoff, and a clean post-cutoff holdout) + a memorization probe |
| D11 | Training length | Up to 5 epochs with early stopping on validation loss, checked every quarter epoch |
| D12 | Explanation language | English |

## 3. Architecture

```
Colab (CPU)                      Kaggle / Colab (GPU T4)              Hugging Face
data/  pull TMDB + Wikidata  ─►  model/ train predictor + writer  ─►  Hub: private dataset, checkpoints, models
       clean, features,          eval/  calibrate, evaluate             Space (Docker, free CPU):
       prompts → HF dataset                                              FastAPI /predict /options /health
                                                                         + React form and results page
```

**Repo layout** (`LLM_Movies_Prediction_Model`, Python 3.11, uv):

| Package | One job |
|---|---|
| `data/` | TMDB pull (resumable, chunked), Wikidata SPARQL enrichment, cleaning, SQLite build |
| `features/` | The 6 computed inputs and track-record stats; leak-safe "as of release date" |
| `contract/` | The form: every field, type and rule (one source of truth for validation) |
| `prompt/` | `build_prompt(form, as_of, ctx) -> str`, shared by training and serving |
| `model/` | Base model + LoRA + quantile head; training loop; writer adapter |
| `calibrate/` | Conformalized quantile regression widths |
| `explain/` | Neutral placeholders, batched effect measurement, paragraph writing, faithfulness check |
| `eval/` | Splits (test locked), metrics, baselines, memorization probe, LightGBM comparison, report |
| `serve/` | FastAPI service |
| `web/` | React + Vite front end |
| `notebooks/` | Thin Colab/Kaggle launchers: clone repo, set secrets, run `python -m ...` |

Rule: all logic lives in tested modules; notebooks only launch them.

## 4. Inputs

### 4.1 Form fields

Existing 14 (same rules as the app's `ml/features/contract.py`): genres (1–5), release year, release month, runtime,
language, US certification, director, actors (5–10, billing order), themes (1–5), companies (0–5), countries,
franchise, synopsis, budget (2025 USD, optional).

New 8 (all optional):

| Field | Type | Rules |
|---|---|---|
| Writers | people autocomplete | 0–3 |
| Producers | people autocomplete | 0–3 |
| Composer | people autocomplete | 0–1 |
| Distributor | autocomplete | 0–1, options from `GET /options` |
| Source material | choice | original, novel, comic, true story, remake, video game, play, TV series, manga |
| Release day | int | 1–31, valid for the month |
| Release type | choice | wide (default), limited, festival-first, streaming-only |
| Tagline | text | ≤150 characters |

### 4.2 Computed inputs (never typed by the user)

1. Competition: number of films with budget > $100M (2025 USD) released within ±14 days.
2. Franchise installment number.
3. Prior collaborations between director and lead actors.
4. Age at release of the lead actor and the director.
5. Prior award wins and nominations (Oscars, BAFTA, Golden Globes, major festivals) of director, writers and cast, dated before release.
6. Source IP strength: number of Wikipedia language editions of the source work's Wikidata item.

Plus track-record stats per person/company (earlier films, median earlier log revenue and score, shrunk towards the
typical film), as in the app.

### 4.3 Excluded

Trailer views, theater counts, marketing spend (no honest historical pre-release source); post-release keywords
(leakage). Stretch goals: Wikipedia pageviews before release, poster image, Greek explanations.

## 5. Data pipeline (Colab CPU runtime)

1. **TMDB pull.** IDs from TMDB's daily movie ID export; one call per film with `append_to_response=credits,keywords,release_dates`.
   Skip adult and video entries. No vote filter at pull time. Raw JSONL shards per chunk (e.g. 50k IDs per session),
   saved to Google Drive; a manifest records finished chunks so any session resumes. TMDB token in Colab Secrets.
2. **Wikidata.** SPARQL in batches of ~500 TMDB IDs (P4947) with a descriptive project User-Agent and pauses.
   Films: based on (P144) and the source's type and sitelink count, distributed by (P750), screenwriter (P58),
   producer (P162), composer (P86), part of the series (P179) / follows (P155), box office (P2142), cost (P2130).
   People via TMDB person ID (P4985): award received (P166), nominated for (P1411) with point in time (P585), birth date.
3. **Cleaning.** App rules: TMDB `0` = unknown; money to 2025 USD via the CPI table; suspicious-money checks.
   Wikidata money: currency conversion, worldwide figures only; TMDB wins unless empty.
4. **Leak safety.** Only films released strictly before a film's date count; an earlier film's revenue and score count
   once it is 182 days old (budget from release); awards only if dated before release; vocabularies fit on train years only.
5. **Targets.** Revenue: worldwide, 2025 USD, log scale. Score: TMDB `vote_average`, weighted by log(1+votes);
   minimum vote count decided after the pull.
6. **Build.** SQLite built on the Colab disk from the shards (never written directly on Drive), then copied to Drive.
   Output: a private Hugging Face dataset of examples `{prompt, revenue_log, score, score_weight, split, tmdb_id}`.

## 6. Splits

| Split | Release dates | Use |
|---|---|---|
| Train | ≤ 2013 | Fit LoRA and head |
| Calibration | 2014–2015 | Conformal widths |
| Validation | 2016–2018 | Early stopping, model choice, experiments |
| Test | 2019 → base-model cutoff | Locked; opened once at the end |
| Clean holdout | after base-model cutoff, released ≥182 days before the pull | Locked; opened once at the end |

The test sets are locked in code (as in the app's `TestSetLocked`), enforced by a test.

## 7. Prompt

`build_prompt` writes one compact block (~400–600 tokens): release date, weekday, release type, distributor; genres,
runtime, rating, language, countries; director, writers, cast (billing order), producers, composer, each as
`Name [earlier films · median revenue · median score · award nominations before]`; source and franchise
(installment); themes; budget or "not given"; competition; tagline; synopsis truncated to ~120 words.

- Missing optional fields are written as "not given". During training, optional fields are hidden at random (~30%).
- No title field.

## 8. Model and training

- **Base model.** Shortlist of ~0.5B and ~1–1.7B open models (e.g. Qwen, Llama 3.2 1B, Gemma, SmolLM families).
  Chosen by: licence allows fine-tuning and a public Space; known training cutoff (earlier = bigger clean holdout);
  memorization probe; validation accuracy; CPU speed on the Space.
- **Architecture.** Frozen base model in fp16, LoRA (rank ~16) on attention and MLP; last-token hidden state → MLP
  head → 6 outputs. Each target predicts the median plus two softplus offsets, so p10 ≤ p50 ≤ p90.
- **Loss.** Pinball loss per quantile; multi-task masks (a film trains only the targets it has); score weighted by
  log(1+votes); revenue films oversampled.
- **Training.** Transformers + PEFT, fp16 mixed precision, gradient checkpointing, separate learning rates for LoRA
  and head. Up to 5 epochs; validation every quarter epoch; stop after 3 checks with no improvement; keep the best
  checkpoint. YAML configs, fixed seeds. Checkpoints to the Hub every N steps, automatic resume. Weights & Biases tracking.
- **Experiments** (on a data sample first; full run only for the final model):
  E1 size (~0.5B vs ~1.5B); E2 names + stats vs anonymized names + stats; E3 with vs without synopsis and tagline.
- **Final model.** After all choices are frozen and the one-time test run is done, retrain on all years before the clean holdout with the same settings
  and step count; keep the calibration widths.

## 9. Calibration and explanations

- **Calibration.** Conformalized quantile regression on the calibration split: one width adjustment per target so
  p10–p90 covers 80%. Coverage is checked on validation.
- **Effects.** ~12 input groups (director; cast; writers; producers + composer; distributor; budget; release timing;
  genres + themes; source + franchise; rating + runtime; language + countries; synopsis + tagline). Each group is
  replaced by a neutral version (generic name with typical training stats, or "not given"); all versions run in one
  batch. Effect = change in p50: revenue as a multiplier, score in points. Report the top 3 raising and top 3
  lowering per target; effects are "on its own", not additive.
- **Writer.** A second LoRA adapter on the same base turns `{effects, key form facts}` into an 80–120 word paragraph.
  Training data: ~5k examples from real training films' effect lists, paragraphs from varied templates (no paid API).
- **Faithfulness check.** Every input the paragraph mentions must be in the effect list, with the right direction and
  roughly the right size; otherwise the template paragraph is returned.
- **Response.** `{revenue {p10,p50,p90}, score {p10,p50,p90}, effects [{group, revenue_x, score_delta}], explanation, model_version}`.

## 10. Evaluation

- **Metrics.** Revenue: MAE of log revenue, % within 2×, p10–p90 coverage and typical width. Score: MAE, coverage,
  width. Pinball loss. Bootstrap confidence intervals.
- **Comparisons on shared films** (present in both datasets): LLM vs app LightGBM (read-only script loading the
  frozen models from the Movie Prediction App folder) vs baselines B0–B2 (same definitions as the app).
- **Two test sets** (section 6), each opened once.
- **Memorization probe.** Ask the base model directly for the box office of test films; report how many it "knows"
  (within 2×) and the LLM's error on known vs unknown films.
- **Explanations.** Faithfulness pass rate; direction sanity checks (e.g. neutralizing a star usually lowers revenue);
  author reads 30 paragraphs.
- **Deliverables.** Results report in the repo; Hugging Face model card.

## 11. Serving and front end

- **Space.** Docker Space on free CPU. FastAPI: `POST /predict`, `GET /options`, `GET /health`. 8-bit model, batched
  effect runs. Target ≤ ~30 s for prediction + explanation once awake. Fallbacks in order: template paragraph, 0.5B model.
  The Space sleeps after 48 h idle; the front end shows a "waking up" state.
- **Front end.** React + Vite, copied from the app: the creator form with the 8 new fields, and a results page with
  the two ranges, the effects and the paragraph. Only these pages are copied.

## 12. Possible later merge into the app (not in scope)

Add the new fields to the app's form, a backend client calling the Space, and an "LLM prediction" panel next to
LightGBM. Decided at the end of this project.

## 13. Testing

pytest, run in GitHub Actions on CPU:
- Data: cleaning rules; Wikidata parsing from saved sample responses; leak-safety time-travel test (cut the DB at a
  date, features of later films must not change); golden prompts.
- Model: quantiles never cross; loss masks; calibration maths; effect measurement and the full
  train → predict → explain path on a tiny random model.
- Explanations: faithfulness checker cases.
- Service: API contract tests with a fake model.
- Front end: component tests (Vitest), as in the app.

## 14. Risks

| Risk | Mitigation |
|---|---|
| LLM worse than LightGBM | Acceptable result; E2 and the memorization analysis explain why |
| Messy or sparse Wikidata | Strict rules, TMDB wins on conflict, coverage reported per field |
| Colab/Kaggle quotas and disconnects | Chunked resumable steps, Hub checkpoints, experiments on samples |
| CPU latency on the Space | 8-bit, batching, short prompts, fallbacks |
| Licences and terms | TMDB attribution; Wikidata CC0; base-model licence checked first |
