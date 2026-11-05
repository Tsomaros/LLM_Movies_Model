# Phase 1: Foundation and Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A tested repo whose Colab-run pipeline pulls every TMDB film plus Wikidata facts, cleans them, computes the leak-safe inputs, and publishes a private Hugging Face dataset of resolved films ready for prompt rendering.

**Architecture:** One Python package `llm_movies/` with one sub-package per spec unit (`data`, `features`, `contract`, `prompt`, `eval`). Network steps write raw JSONL shards (resumable, manifest-driven); a pure build step turns shards into SQLite; a resolver turns each film into a `ResolvedFilm` (form + computed stats as of release), and a pure renderer turns that into prompt text. The dataset stores `ResolvedFilm` JSON so training can re-render prompts with random field hiding without a database.

**Tech Stack:** Python 3.11, uv, pytest, ruff, pandas, pyarrow, requests, huggingface_hub, SQLite, GitHub Actions, Google Colab (CPU runtime).

**Spec:** `docs/superpowers/specs/2026-11-05-llm-movie-predictor-design.md` (Roadmap: `docs/superpowers/plans/2026-11-05-roadmap.md`)

## Global Constraints

- Python 3.11 only; dependencies managed with uv; `uv run pytest` must pass offline (no network in tests: use saved fixtures).
- Pre-release inputs only: `popularity`, `vote_count`, `vote_average`, `revenue` are never inputs.
- TMDB `0` budget or revenue means unknown (`None`). Values below `MIN_MONEY = 1_000` USD are placeholders → `None`.
- Money in 2025 USD via `data/cpi_us_annual.csv` (copied from the Movie Prediction App, includes a 2026 row equal to 2025).
- Leak safety: only films released strictly before `as_of` count; an earlier film's revenue and score count once it is `MATURE_DAYS = 182` days old; its budget from release; awards only if dated before `as_of`.
- Track-record shrinkage: `(n * own + SHRINK_FILMS * typical) / (n + SHRINK_FILMS)` with `SHRINK_FILMS = 4`, typical film built from earlier films only.
- Splits by release date: train ≤2013, calibration 2014–2015, validation 2016–2018, test 2019→cutoff, holdout after cutoff (cutoff = `base_model_cutoff` in `config/data.yaml`, `null` until Phase 2; while `null`, every film from 2019 is `test`). Test and holdout are locked: loading them raises `TestSetLocked` unless `final=True`.
- Score target: `vote_average`, weight `log(1 + vote_count)`; minimum vote count is a config value `min_votes` (default 50) decided after the data report.
- Wikidata requests send `User-Agent: LLMMoviesPredictionModel/0.1 (https://github.com/Tsomaros/LLM_Movies_Prediction_Model)` and never an email address.
- Secrets (TMDB token, HF token) come from environment variables `TMDB_READ_TOKEN`, `HF_TOKEN` (Colab Secrets in notebooks, `.env` locally); never committed.
- Notebooks only launch `python -m llm_movies....` commands; no logic in notebooks.
- SQLite is built on local disk, never written directly on Google Drive.
- `CLAUDE.md` hard rules apply to every task: after the task's code is written, run the **code-reviewer** subagent; run the **test-auditor** subagent on every new test file; run the **ui-ux-reviewer** on every new frontend component (none in Phase 1). Apply their accepted findings before the task's commit.

## Review Focus

1. A Colab session dies mid-chunk → rerunning the same chunk resumes after the last written line and never duplicates a film (Task 4 test `test_pull_chunk_resumes_without_duplicates`).
2. A truncated last line in a shard (killed while writing) → the reader skips it and the pull refetches that id (Task 4 test `test_truncated_last_line_is_refetched`).
3. TMDB returns 429 or 5xx → retried with backoff; 404 → recorded as missing, not retried forever (Task 4 tests `test_retries_429_then_succeeds`, `test_404_is_recorded_missing`).
4. Wikidata gives several conflicting values (two distributors, two box-office figures, a non-USD figure) → deterministic choice, never a crash (Task 6 test `test_multiple_values_pick_deterministically`).
5. A person's award or a franchise entry dated on or after the film's release → not counted (Task 10 test `test_award_on_release_day_not_counted`, Task 11 test `test_time_travel_identical_inputs`).

---

## File Structure

```
pyproject.toml, uv.lock, .gitignore, .env.example, README.md, ruff config in pyproject
.github/workflows/ci.yml
config/data.yaml                      split years, cutoff, min_votes, chunk size, paths
data/cpi_us_annual.csv                copied from the app
data/fx_usd_annual.csv                annual USD rates for the currencies Wikidata uses most
llm_movies/__init__.py
llm_movies/config.py                  load config/data.yaml -> DataConfig
llm_movies/data/money.py              CPI + FX conversion, placeholder rules
llm_movies/data/shards.py             JSONL shard writer/reader + manifest (shared by TMDB and Wikidata pulls)
llm_movies/data/tmdb_ids.py           daily ID export download + filtering + chunking
llm_movies/data/tmdb_pull.py          TMDB client + pull_chunk + CLI
llm_movies/data/wikidata.py           SPARQL client, film and people queries, parsers + CLI
llm_movies/data/source_material.py    P144/keyword -> source category
llm_movies/data/awards.py             award family whitelist (awards.json) + matcher
llm_movies/data/awards.json
llm_movies/data/clean.py              raw TMDB + Wikidata -> cleaned rows
llm_movies/data/build_db.py           shards -> SQLite + CLI
llm_movies/eval/splits.py             assign_split, TestSetLocked, load_split
llm_movies/contract/form.py           Field table, GENRES, CERTIFICATIONS, SOURCE_TYPES, RELEASE_TYPES, validate()
llm_movies/contract/themes.json       copied from the app (ml/features/themes.json)
llm_movies/features/history.py        leak-safe track records (people, companies, distributors)
llm_movies/features/computed.py       the 6 computed inputs
llm_movies/prompt/resolve.py          ResolvedFilm, resolve(form, as_of, ctx), film_to_form(row)
llm_movies/prompt/render.py           render(resolved, hidden=frozenset()) -> str
llm_movies/data/export.py             SQLite -> parquet dataset -> private HF dataset + CLI
llm_movies/data/report.py             coverage and counts report + CLI
notebooks/01_pull_tmdb.ipynb, 02_pull_wikidata.ipynb, 03_build_and_export.ipynb
tests/... mirrors the package; tests/fixtures/ holds saved TMDB/Wikidata responses and a mini shard set
```

**Execution note:** the shell on Dimitris's computer cannot delete files by default, and git deletes its own lock files. Before Task 1, ask for delete permission on `C:\Users\dimbl\Desktop\LLM_Movies_Prediction_Model` (the device delete-permission tool), or have Dimitris run the git commands himself.

---

### Task 1: Repo foundation and CI

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.env.example`, `README.md`, `config/data.yaml`, `llm_movies/__init__.py`, `llm_movies/config.py`, `.github/workflows/ci.yml`, `data/cpi_us_annual.csv` (copy from `C:\Users\dimbl\Desktop\Movies_Final_project\data\cpi_us_annual.csv`)
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `DataConfig` (frozen dataclass: `train_max_year=2013`, `calibration_years=(2014, 2015)`, `validation_years=(2016, 2018)`, `test_from_year=2019`, `base_model_cutoff: date | None`, `min_votes=50`, `chunk_size=50_000`, `mature_days=182`, `shrink_films=4`) and `load_config(path: Path = Path("config/data.yaml")) -> DataConfig`.

- [x] **Step 1:** `git init`; `uv init --package --name llm_movies --python 3.11`; add deps `pandas pyarrow requests pyyaml huggingface_hub python-dotenv`, dev deps `pytest ruff`. `.gitignore`: `.env`, `.venv/`, `raw/`, `build/`, `*.db`, `*.parquet`, `__pycache__/`, `.ipynb_checkpoints/`.
- [x] **Step 2: Write failing tests**

```python
def test_defaults_match_spec():
    c = load_config(Path("config/data.yaml"))
    assert (c.train_max_year, c.calibration_years, c.validation_years, c.test_from_year) == (2013, (2014, 2015), (2016, 2018), 2019)
    assert c.base_model_cutoff is None and c.min_votes == 50 and c.mature_days == 182 and c.shrink_films == 4

def test_cutoff_parses_as_date(tmp_path):
    p = tmp_path / "d.yaml"; p.write_text(Path("config/data.yaml").read_text().replace("base_model_cutoff: null", "base_model_cutoff: 2024-06-30"))
    assert load_config(p).base_model_cutoff == date(2024, 6, 30)
```

- [x] **Step 3:** Run `uv run pytest tests/test_config.py -v` → FAIL (no module).
- [x] **Step 4:** Implement `config/data.yaml` and `load_config`.
- [x] **Step 5:** Run → PASS. Add CI workflow: `astral-sh/setup-uv`, `uv sync`, `uv run ruff check .`, `uv run pytest`. Run `uv run ruff check .` locally → clean.
- [x] **Step 6:** Commit `chore: repo foundation, config and CI` (also add `CLAUDE.md`, `.claude/` and the spec and plans already in `docs/`).

### Task 2: Money conversion

**Files:** Create `llm_movies/data/money.py`, `data/fx_usd_annual.csv`; Test `tests/data/test_money.py`

**Interfaces:**
- Produces: `MIN_MONEY = 1_000`; `load_cpi(path) -> dict[int, float]`; `load_fx(path) -> dict[tuple[str, int], float]` (units of currency per 1 USD); `clean_amount(amount: float | None) -> float | None` (0, None, < MIN_MONEY → None); `to_usd(amount, currency: str, year: int, fx) -> float | None` (None for an unknown currency/year); `to_2025_usd(amount_usd, year, cpi) -> float | None`.

- [ ] **Step 1:** Build `data/fx_usd_annual.csv` (`currency,year,per_usd`) for EUR, GBP, JPY, INR, CNY, KRW, AUD, CAD, FRF, DEM, ITL, ESP, 1960–2026, from FRED annual exchange-rate series (legacy currencies via their fixed euro conversion rates after 1999). Record the source in a header comment row `# source: ...`.
- [ ] **Step 2: Write failing tests**

```python
def test_zero_and_placeholder_are_unknown():
    assert clean_amount(0) is None and clean_amount(7) is None and clean_amount(None) is None and clean_amount(1_000) == 1_000
def test_2025_dollars_scale_by_cpi():
    cpi = {2000: 172.2, 2025: 322.0}
    assert to_2025_usd(100.0, 2000, cpi) == pytest.approx(100 * 322.0 / 172.2)
def test_unknown_currency_is_none():
    assert to_usd(1e6, "XYZ", 2010, {("EUR", 2010): 0.75}) is None
def test_eur_converts():
    assert to_usd(75.0, "EUR", 2010, {("EUR", 2010): 0.75}) == pytest.approx(100.0)
def test_cpi_file_has_1960_to_2026():
    cpi = load_cpi(Path("data/cpi_us_annual.csv")); assert min(cpi) == 1960 and max(cpi) >= 2026
```

- [ ] **Step 3:** Run → FAIL. **Step 4:** Implement. **Step 5:** Run → PASS. **Step 6:** Commit `feat(data): money conversion`.

### Task 3: Shards and manifest

**Files:** Create `llm_movies/data/shards.py`; Test `tests/data/test_shards.py`

**Interfaces:**
- Produces: `ShardWriter(path: Path)` context manager with `.write(record: dict)` (one JSON line, flushed); `read_shard(path) -> Iterator[dict]` (skips a truncated/invalid last line); `done_keys(path, key: str) -> set` (keys of valid records); `Manifest(path)` with `.is_done(chunk: int) -> bool`, `.mark_done(chunk: int, stats: dict)` (written atomically: temp file + `os.replace`).

- [ ] **Step 1: Write failing tests:** `test_roundtrip`, `test_truncated_last_line_is_skipped` (write two records, append `'{"id": 3, "ti'`, assert reader yields 2 and `done_keys == {1, 2}`), `test_manifest_marks_and_survives_reload`.
- [ ] **Step 2–4:** FAIL → implement → PASS. **Step 5:** Commit `feat(data): resumable JSONL shards`.

### Task 4: TMDB IDs and resumable pull

**Files:** Create `llm_movies/data/tmdb_ids.py`, `llm_movies/data/tmdb_pull.py`; Test `tests/data/test_tmdb_ids.py`, `tests/data/test_tmdb_pull.py`; fixtures `tests/fixtures/tmdb_ids_sample.json.gz`, `tests/fixtures/tmdb_movie_475557.json` (trimmed real response with credits, keywords, release_dates)

**Interfaces:**
- Consumes: Task 3 shards and manifest.
- Produces:
  - `export_url(day: date) -> str` = `https://files.tmdb.org/p/exports/movie_ids_{MM}_{DD}_{YYYY}.json.gz`; `read_ids(path: Path) -> list[int]` (skips `adult: true` and `video: true`, sorted ascending); `chunk_ids(ids, size) -> list[list[int]]`.
  - `TmdbClient(token: str, session: requests.Session | None = None, max_per_second: float = 40, max_retries: int = 6)`; `.movie(tmdb_id: int) -> dict | None` calling `/3/movie/{id}?append_to_response=credits,keywords,release_dates`; 429/5xx → exponential backoff honouring `Retry-After`; 404 → `None`.
  - `pull_chunk(client, chunk: int, ids: list[int], out_dir: Path, manifest: Manifest) -> dict` writing `out_dir/tmdb_{chunk:04d}.jsonl` (full record + `pulled_at`) and `out_dir/missing_{chunk:04d}.txt`; skips ids already in the shard or missing file; marks the chunk done with `{"pulled": n, "missing": m}`.
  - CLI `python -m llm_movies.data.tmdb_pull --ids PATH --chunk N --out DIR [--limit K]`.

- [ ] **Step 1: Write failing tests** with a fake session (records calls, returns scripted responses):
  - `test_read_ids_skips_adult_and_video`
  - `test_retries_429_then_succeeds` (429 with `Retry-After: 0`, then 200 → dict, 2 calls)
  - `test_404_is_recorded_missing` (id in `missing_0000.txt`, not refetched on rerun)
  - `test_pull_chunk_resumes_without_duplicates` (first run raises after 2 ids; rerun → shard has each id once)
  - `test_truncated_last_line_is_refetched`
  - `test_done_chunk_is_skipped` (manifest done → 0 HTTP calls)
- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5:** Manual check (needs `TMDB_READ_TOKEN`): `uv run python -m llm_movies.data.tmdb_pull --ids <export> --chunk 0 --out raw/tmdb --limit 50` → 50 lines (minus missing), manifest updated.
- [ ] **Step 6:** Commit `feat(data): resumable TMDB pull`.

### Task 5: Source material and award families

**Files:** Create `llm_movies/data/source_material.py`, `llm_movies/data/awards.py`, `llm_movies/data/awards.json`; Test `tests/data/test_source_material.py`, `tests/data/test_awards.py`

**Interfaces:**
- Produces:
  - `SOURCE_TYPES = ("original", "novel", "comic", "true story", "remake", "video game", "play", "TV series", "manga")`.
  - `source_from_wikidata(based_on_types: set[str]) -> str | None` (Wikidata QIDs of the P144 target's P31 classes → category); `source_from_keywords(names: set[str]) -> str | None` using exactly: `based on novel or book`, `based on young adult novel` → novel; `based on comic`, `superhero` → comic; `based on true story`, `biography` → true story; `remake`, `live action remake` → remake; `based on video game` → video game; `based on play or musical` → play; `based on tv series` → TV series; `based on manga` → manga. `source_category(wd, keywords) -> str` = Wikidata first, then keywords, else `"original"`.
  - `awards.json`: `{family: [QIDs of the award items/classes]}` for `oscar`, `bafta_film`, `golden_globe`, `cannes`, `venice`, `berlin`, `sag` (QIDs looked up on wikidata.org and listed with their labels). `award_family(award_qid: str, parent_qids: set[str]) -> str | None`.
- [ ] **Step 1: Failing tests:** keyword mapping table (each keyword above → its category), `test_wikidata_wins_over_keywords`, `test_no_signal_is_original`, `test_awards_json_has_seven_families_with_labels`, `test_award_family_matches_parent_class`.
- [ ] **Step 2–4:** FAIL → implement → PASS. **Step 5:** Commit `feat(data): source material and award families`.

### Task 6: Wikidata enrichment

**Files:** Create `llm_movies/data/wikidata.py`; Test `tests/data/test_wikidata.py`; fixtures `tests/fixtures/wd_films.json`, `tests/fixtures/wd_people.json` (saved SPARQL JSON results for ~5 films incl. Joker 475557 and their people)

**Interfaces:**
- Consumes: Tasks 2, 3, 5.
- Produces:
  - `ENDPOINT = "https://query.wikidata.org/sparql"`, `USER_AGENT` (Global Constraints), `BATCH = 500`, `PAUSE_S = 1.0`.
  - `film_query(tmdb_ids: list[int]) -> str`: `VALUES ?tmdb {...}` on P4947; OPTIONAL P144 (+ target's P31 and `wikibase:sitelinks`), P750, P58, P162, P86, P179, P155, P2142 (with P3005 place, P585 date, unit), P2130 (unit).
  - `people_query(tmdb_person_ids: list[int]) -> str`: P4985; P569 birth; P166 and P1411 with P585.
  - `parse_films(sparql_json) -> dict[int, FilmFacts]`; `FilmFacts` dataclass: `qid, source_types: set[str], source_sitelinks: int | None, distributors: list[str]` (labels, sorted), `writers_qids, producers_qids, composer_qids, series_qid, follows_qid, box_office: list[Money], cost: list[Money]`; `Money(amount, currency, place_qid, year)`.
  - `parse_people(sparql_json) -> dict[int, PersonFacts]`; `PersonFacts(birth: date | None, awards: list[AwardEvent])`; `AwardEvent(award_qid, kind: Literal["won","nominated"], when: date | None)`.
  - `pick_worldwide_usd(values: list[Money], fx) -> tuple[float | None, int | None]`: keep place = worldwide (`Q13780930`, verify the QID while saving the fixture) or no place; convert with `to_usd`; pick the latest dated figure, ties → largest; returns (usd, year).
  - `SparqlClient(session=None)` with `.run(query) -> dict` (retry on 429/5xx, honour `Retry-After`).
  - CLI `python -m llm_movies.data.wikidata films|people --ids PATH --out DIR` writing shards via Task 3 (one record per id, `{tmdb_id, facts}`), batch-resumable.
- [ ] **Step 1: Failing tests:** `test_film_query_contains_values_and_properties`, `test_parse_joker_distributor_and_source`, `test_multiple_values_pick_deterministically` (two distributors → sorted list; two box-office values → latest year wins; non-USD converted; unknown currency ignored), `test_non_worldwide_box_office_ignored`, `test_parse_people_awards_with_dates`, `test_user_agent_has_no_email` (`"@" not in USER_AGENT`).
- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5:** Manual check: run `films` on 500 ids from chunk 0 → shard written; print coverage of distributors and P144.
- [ ] **Step 6:** Commit `feat(data): Wikidata enrichment`.

### Task 7: Cleaning

**Files:** Create `llm_movies/data/clean.py`; Test `tests/data/test_clean.py`

**Interfaces:**
- Consumes: Tasks 2, 5, 6.
- Produces: `clean_movie(raw: dict, wd: FilmFacts | None, cpi, fx, pulled_at: date) -> MovieRow | None` (None when no release date or status not Released/Post Production/In Production/Planned). `MovieRow` dataclass fields: `tmdb_id, title, release_date, year, month, day, runtime, language, countries, genres, certification` (US, mapped like the app: NR/missing → `"unrated/unknown"`), `keywords, overview, tagline, collection_id, budget_2025, revenue_2025, revenue_source ("tmdb"|"wikidata"|None), money_suspect: bool, revenue_incomplete: bool, vote_count, vote_average, release_type, source_material, distributor, wd_qid`.
  - Revenue/budget: TMDB first (after `clean_amount`); Wikidata only when TMDB is None.
  - `money_suspect`: revenue/budget < `LOW_RATIO = 0.0001`.
  - `revenue_incomplete`: `(pulled_at - release_date).days < 182`.
  - `release_type` from US (else earliest country) `release_dates`: type 1 premiere before type 2/3 → `festival-first`; type 3 → `wide`; only type 2 → `limited`; only 4/5/6 → `streaming-only`; none → `None`.
  - `distributor`: first of Wikidata's sorted list, else None.
- [ ] **Step 1: Failing tests:** `test_tmdb_money_wins_over_wikidata`, `test_wikidata_fills_missing_revenue`, `test_low_ratio_flags_suspect`, `test_release_type_rules` (one case per rule), `test_unreleased_without_date_dropped`, `test_post_release_fields_kept_only_as_targets` (MovieRow has vote fields; Task 13 asserts they never reach the resolved form).
- [ ] **Step 2–4:** FAIL → implement → PASS. **Step 5:** Commit `feat(data): cleaning rules`.

### Task 8: SQLite build

**Files:** Create `llm_movies/data/build_db.py`; Test `tests/data/test_build_db.py`; fixture `tests/fixtures/mini/` (3 TMDB shards of ~30 films with credits, 1 Wikidata films shard, 1 people shard; generated by `python -m tests.fixtures.make_mini` from real responses, committed)

**Interfaces:**
- Consumes: Tasks 3, 6, 7.
- Produces: `build_db(tmdb_dir: Path, wd_films_dir: Path, wd_people_dir: Path, out: Path, pulled_at: date) -> dict` (counts). Tables: `movies` (MovieRow columns + `split`), `people(id, name, birth)`, `credits(movie_id, person_id, role, ord)` with role in `director, actor, writer, producer, composer` (TMDB jobs `Writer`/`Screenplay` → writer), `companies`, `movie_companies`, `keywords`, `movie_keywords`, `awards(person_id, family, kind, when)`, `meta(key, value)`. Builds into `out.with_suffix(".tmp")` then `os.replace`. CLI `python -m llm_movies.data.build_db --tmdb DIR --wd-films DIR --wd-people DIR --out PATH`.
- [ ] **Step 1: Failing tests:** `test_counts_match_fixture`, `test_writer_and_screenplay_map_to_writer`, `test_duplicate_film_across_shards_kept_once`, `test_rebuild_is_identical` (same row hashes twice), `test_awards_only_whitelisted_families`.
- [ ] **Step 2–4:** FAIL → implement → PASS. **Step 5:** Commit `feat(data): SQLite build`.

### Task 9: Splits and the test lock

**Files:** Create `llm_movies/eval/splits.py`; Test `tests/eval/test_splits.py`

**Interfaces:**
- Consumes: Task 1 `DataConfig`.
- Produces: `assign_split(release: date, cfg: DataConfig) -> Literal["train","calibration","validation","test","holdout"]`; `class TestSetLocked(Exception)`; `load_split(df: pd.DataFrame, name: str, final: bool = False) -> pd.DataFrame` (raises for test/holdout unless `final`). Holdout requires `release <= pulled_at - 182 days`; later films get `"later"` (never used).
- [ ] **Step 1: Failing tests:** boundaries (2013-12-31 train, 2014-01-01 calibration, 2016-01-01 validation, 2019-01-01 test), `test_null_cutoff_puts_all_2019plus_in_test`, `test_cutoff_moves_later_films_to_holdout`, `test_unsettled_after_cutoff_is_later`, `test_locked_without_final`, and `test_only_final_module_passes_final_true` (grep the package: `final=True` appears only in `llm_movies/eval/final.py`, which does not exist yet → zero occurrences allowed now).
- [ ] **Step 2–4:** FAIL → implement → PASS. Wire `build_db` to fill `movies.split`. **Step 5:** Commit `feat(eval): time splits and test lock`.

### Task 10: Form contract

**Files:** Create `llm_movies/contract/form.py`, copy `llm_movies/contract/themes.json` from the app's `ml/features/themes.json`; Test `tests/contract/test_form.py`

**Interfaces:**
- Produces: `GENRES`, `CERTIFICATIONS`, `MIN_YEAR=1960`, `MAX_YEAR=2035`, `MIN_RUNTIME=40`, `MAX_RUNTIME=300`, `MIN_BUDGET=1_000`, `MAX_BUDGET=1_000_000_000`, `MAX_SYNOPSIS=2_000`, `MIN_ACTORS=5`, `MAX_ACTORS=10` (copied from the app's contract), plus `RELEASE_TYPES = ("wide","limited","festival-first","streaming-only")`, `MAX_TAGLINE = 150`; `FIELDS` (the app's 14 + writers refs 0–3, producers refs 0–3, composer ref 0–1, distributor ref 0–1, source_material choice of `SOURCE_TYPES`, release_day int 1–31, release_type choice default `"wide"`, tagline text ≤150); `validate(form: dict) -> tuple[dict, list[dict]]` (errors as `{"field", "message"}`; unknown fields are errors; release_day checked against year+month with `calendar.monthrange`).
- [ ] **Step 1: Failing tests:** `test_minimal_valid_form`, `test_required_fields_missing`, `test_release_day_31_in_april_rejected`, `test_writers_max_three`, `test_tagline_over_150_rejected`, `test_release_type_defaults_to_wide`, `test_unknown_field_rejected`, `test_post_release_fields_rejected` (`vote_average`, `popularity`, `revenue` → errors).
- [ ] **Step 2–4:** FAIL → implement → PASS. **Step 5:** Commit `feat(contract): creator form with the new fields`.

### Task 11: Leak-safe track records

**Files:** Create `llm_movies/features/history.py`; Test `tests/features/test_history.py`

**Interfaces:**
- Consumes: Task 8 database.
- Produces: `History.from_db(con) -> History`; `History.person(person_id: int | None, role: str, as_of: date) -> PersonRecord` with `films: int, years_active: int, revenue_log: float, score: float, budget_log: float, is_new: bool` (shrunk with `SHRINK_FILMS=4` towards `History.typical(as_of)`; revenue and score only from films ≥182 days old at `as_of`, budget from release); `History.company(company_id, as_of)` and `History.distributor(name, as_of)` returning the same record shape; `History.typical(as_of) -> tuple[float, float, float]` (median log revenue, score, log budget of earlier films). People names resolve with `History.resolve(ref: int | str | dict | None, role) -> int | None` (id wins; exact name match, most films wins; unknown → None = new).
- [ ] **Step 1: Failing tests:** `test_film_never_counts_for_itself`, `test_revenue_waits_182_days_budget_does_not`, `test_shrinkage_formula` (one earlier film → `(1*own + 4*typical)/5`), `test_unknown_person_is_new_with_typical_values`, `test_time_travel_identical_inputs` (cut the mini DB at a date: records for every film released before the cut are identical with and without later films).
- [ ] **Step 2–4:** FAIL → implement → PASS. **Step 5:** Commit `feat(features): leak-safe track records`.

### Task 12: The six computed inputs

**Files:** Create `llm_movies/features/computed.py`; Test `tests/features/test_computed.py`

**Interfaces:**
- Consumes: Task 11 `History`, Task 8 database.
- Produces: `Context.from_db(con) -> Context` (holds History, release calendar, collections, awards, births, source sitelinks) and
  - `competition(ctx, release: date, exclude_id: int | None) -> int`: films with `budget_2025 > 100_000_000` released within ±14 days (calendar known in advance, so later films count).
  - `installment(ctx, collection_id: int | None, release: date) -> int | None`: 1 + earlier entries in the collection.
  - `collaborations(ctx, director_id, actor_ids: list[int | None], as_of) -> int`: earlier films shared by the director and any of the first 3 actors.
  - `age_at(ctx, person_id, as_of) -> int | None`.
  - `prior_awards(ctx, person_ids, as_of) -> dict[str, int]` → `{"wins": n, "nominations": m}` counting only events with `when < as_of` (undated events are not counted).
  - `source_strength(ctx, sitelinks: int | None) -> int | None` (passes the sitelink count through; None for original).
- [ ] **Step 1: Failing tests:** `test_competition_counts_both_sides_and_excludes_self`, `test_first_entry_is_installment_1`, `test_collaborations_only_before`, `test_award_on_release_day_not_counted`, `test_undated_award_not_counted`, `test_age_none_without_birth`.
- [ ] **Step 2–4:** FAIL → implement → PASS. **Step 5:** Commit `feat(features): computed inputs`.

### Task 13: Resolve and render prompts

**Files:** Create `llm_movies/prompt/resolve.py`, `llm_movies/prompt/render.py`; Test `tests/prompt/test_resolve.py`, `tests/prompt/test_render.py`, golden `tests/golden/prompts/*.txt` (3 films from the mini DB)

**Interfaces:**
- Consumes: Tasks 10, 11, 12.
- Produces:
  - `ResolvedFilm` (frozen dataclass, JSON round-trip with `to_json() -> str` / `from_json(s)`): the validated form values plus per-person `{name, films, revenue_usd, score, nominations}` for director, writers, cast, producers, composer; distributor record; `competition, installment, collaborations, director_age, lead_age, source_strength`.
  - `film_to_form(row: MovieRow-like mapping, ctx) -> dict` (a training film written as a form: top 10 cast in billing order, first director, up to 3 writers/producers in credit order, first composer; themes from the film's keywords via `contract/themes.json` (the app's keyword→theme grouping), synopsis = overview, franchise = collection; never vote/popularity/revenue fields).
  - `resolve(form: dict, as_of: date, ctx: Context) -> ResolvedFilm` (as_of = release date for training films; 1st of the release month for user forms unless release_day is given).
  - `OPTIONAL_GROUPS = ("writers","producers","composer","distributor","source_material","release_day","tagline","synopsis","budget","companies","countries","franchise")`.
  - `render(r: ResolvedFilm, hidden: frozenset[str] = frozenset()) -> str`: the layout in spec section 7, line order fixed; hidden or missing optional fields print `not given`; synopsis cut to 120 words; money as `$480M`/`$1.2B`; no title.
- [ ] **Step 1: Failing tests:** `test_render_matches_golden` (×3), `test_hidden_fields_print_not_given`, `test_resolved_json_roundtrip`, `test_no_post_release_values_in_prompt` (vote_average/popularity/revenue of the film itself never appear: render with sentinel values 9.87 / 123456.7 → not in text), `test_synopsis_truncated_to_120_words`, `test_prompt_under_700_tokens_proxy` (≤ 2,800 characters for the longest golden).
- [ ] **Step 2–4:** FAIL → implement → PASS. Regenerate goldens only with `UPDATE_GOLDEN=1`. **Step 5:** Commit `feat(prompt): resolve and render`.

### Task 14: Dataset export and data report

**Files:** Create `llm_movies/data/export.py`, `llm_movies/data/report.py`; Test `tests/data/test_export.py`, `tests/data/test_report.py`

**Interfaces:**
- Consumes: Tasks 8, 9, 12, 13.
- Produces:
  - `export(db: Path, out_dir: Path, cfg: DataConfig) -> Path` writing `out_dir/films.parquet` with columns `tmdb_id, release_date, split, resolved_json, revenue_log` (natural log of `revenue_2025`; null when missing, suspect or incomplete), `score` (null when `vote_count < cfg.min_votes`), `score_weight = log1p(vote_count)`. Rows with neither target are dropped.
  - `push(out_dir, repo_id="Tsomaros/llm-movies-data", private=True)` via `huggingface_hub.HfApi.upload_folder`.
  - `report(db, parquet) -> str` (markdown): films per split; with revenue / with score; coverage % of every new field (writers, producers, composer, distributor, source_material ≠ original, release_type, tagline) and computed input; score-film counts at min_votes 10/20/50/100 (the table for the vote-threshold decision).
  - CLIs: `python -m llm_movies.data.export --db PATH --out DIR [--push]`, `python -m llm_movies.data.report --db PATH --parquet PATH --out reports/data_summary.md`.
- [ ] **Step 1: Failing tests:** `test_targets_null_rules` (suspect/incomplete/low-votes → null), `test_rows_without_targets_dropped`, `test_resolved_json_loads`, `test_report_has_threshold_table`.
- [ ] **Step 2–4:** FAIL → implement → PASS. **Step 5:** Commit `feat(data): dataset export and report`.

### Task 15: Colab launchers and the full run

**Files:** Create `notebooks/01_pull_tmdb.ipynb`, `notebooks/02_pull_wikidata.ipynb`, `notebooks/03_build_and_export.ipynb`, `notebooks/README.md`

Each notebook: mount Drive; clone the GitHub repo; `uv sync`; read `TMDB_READ_TOKEN`/`HF_TOKEN` from Colab Secrets into env; run one CLI; nothing else.
- 01: download today's ID export to Drive once, then `--chunk N` (N typed in a form cell); shards and manifest on `Drive/llm_movies/raw/tmdb/`.
- 02: films then people (person ids from the TMDB shards' credits), shards on `Drive/llm_movies/raw/wd_*`.
- 03: copy shards to `/content/`, `build_db` to `/content/build/movies.db`, copy the DB to Drive, `export --push`, `report`.

- [ ] **Step 1:** Write the notebooks and README (run order, CPU runtime, how to resume).
- [ ] **Step 2:** Run 01 for chunk 0 with `--limit 500`, then 02 and 03 on that sample → a tiny HF dataset and report exist. Fix anything that breaks.
- [ ] **Step 3:** Commit `feat(notebooks): Colab launchers`.
- [ ] **Step 4:** Full run (Dimitris, over several sessions): all chunks of 01, then 02, then 03. Verification: manifest lists every chunk; `reports/data_summary.md` committed.
- [ ] **Step 5:** Decision with Dimitris from the report: `min_votes` → update `config/data.yaml`, re-run export. Commit `data: first full dataset and summary`. Phase 1 done; write the Phase 2 plan.
