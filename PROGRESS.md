# Progress

## Now
- Phase 1 plan: `docs/superpowers/plans/2026-11-05-phase1-foundation-and-data.md`. Tasks 1-3 done; Task 4 done except its manual check. Next: Task 5 (source material and award families).
- Task 4 still open, needs `TMDB_READ_TOKEN` in the environment or `.env`: Step 5, the `--limit 50` check against the live API
  (expect 50 lines minus missing, manifest unchanged; the plan's "manifest updated" was overruled, see Notes), and saving the
  trimmed real response `tests/fixtures/tmdb_movie_475557.json`. `TmdbClient` has only met hand-built responses so far.
  Task 8's mini fixtures also come from real responses, so the token is needed by then at the latest.
- Execution: subagent-driven, alternating between Claude Code (Windows) and the claude.ai cloud session.
- No GitHub remote yet, so `.github/workflows/ci.yml` has not run. Check the first run after the first push.

## Done
- 2026-11-05: Design spec approved; roadmap and Phase 1 plan written; `CLAUDE.md`, the three subagents (adapted) and the
  ponytail skill copied from the Movie Prediction App.
- 2026-11-05: Task 1 (repo foundation, config, CI). 7 tests pass, ruff clean, code-reviewer and test-auditor run on Windows.
- 2026-11-05: Task 2 (money conversion, FX file). 21 tests pass, ruff clean, code-reviewer and test-auditor run on Windows.
- 2026-11-05: Task 3 (resumable JSONL shards, manifest). 38 tests pass, ruff clean, code-reviewer and test-auditor run on Windows.
- 2026-11-05: Task 4 (TMDB IDs, client, resumable pull, CLI), except Step 5 and the real fixture. 77 tests pass, ruff clean,
  code-reviewer and test-auditor run on Windows.

## Open questions
- `min_votes` for the score target: decided from the Phase 1 data report.
- Base model and its training cutoff: decided in Phase 2.

## Notes
- Reviews run on the cloud side are marked "(cloud review, no hooks)".
- `tests/test_config.py` pins `config/data.yaml` to the defaults (`min_votes` 50, `base_model_cutoff` null). Changing either
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
- For Task 7: `clean_amount` handles NaN, but `pd.NA` raises in `math.isnan`. Convert to plain floats or None before calling it.
