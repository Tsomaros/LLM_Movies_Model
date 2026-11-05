# Progress

## Now
- Phase 1 plan: `docs/superpowers/plans/2026-11-05-phase1-foundation-and-data.md`. Tasks 1-7 done. Next: Task 8 (SQLite build).
- Execution: subagent-driven, alternating between Claude Code (Windows) and the claude.ai cloud session.
- Remote `origin` is https://github.com/Tsomaros/LLM_Movies_Model (public). `main` pushed at fed3002; its CI run passed
  (https://github.com/Tsomaros/LLM_Movies_Model/actions/runs/42027850257).

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
- For Task 8 and later: `revenue_2025` can be set on a film not yet 182 days old or not yet released, so training filters must
  use `revenue_incomplete`. Pre-1960 films keep their row but have no money, since the CPI file starts in 1960. A missing
  top-level `id` raises on purpose, so decide in Task 8 whether a build stops or skips. `pick_distributor` relies on Wikidata's
  labels being sorted, which `wikidata.py` guarantees.
- For Task 10: `clean.py` holds a local `CERTIFICATIONS`; use one definition.
- `clean_amount` handles NaN, but `pd.NA` raises in `math.isnan`. Convert to plain floats or None before calling it.
