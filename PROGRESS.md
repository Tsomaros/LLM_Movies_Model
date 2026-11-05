# Progress

## Now
- Phase 1 plan: `docs/superpowers/plans/2026-11-05-phase1-foundation-and-data.md`. Tasks 1-2 done. Next: Task 3 (shards and manifest).
- Execution: subagent-driven, alternating between Claude Code (Windows) and the claude.ai cloud session.
- No GitHub remote yet, so `.github/workflows/ci.yml` has not run. Check the first run after the first push.

## Done
- 2026-11-05: Design spec approved; roadmap and Phase 1 plan written; `CLAUDE.md`, the three subagents (adapted) and the
  ponytail skill copied from the Movie Prediction App.
- 2026-11-05: Task 1 (repo foundation, config, CI). 7 tests pass, ruff clean, code-reviewer and test-auditor run on Windows.
- 2026-11-05: Task 2 (money conversion, FX file). 21 tests pass, ruff clean, code-reviewer and test-auditor run on Windows.

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
- For Task 7: `clean_amount` handles NaN, but `pd.NA` raises in `math.isnan`. Convert to plain floats or None before calling it.
