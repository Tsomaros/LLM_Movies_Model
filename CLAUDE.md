# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

An LLM movie predictor: a small open LLM, fine-tuned with LoRA and a quantile regression head, predicts a planned film's
**worldwide revenue** and **audience score** as calibrated p10/p50/p90 ranges (~80% coverage) and explains in English
which inputs raise or lower them. A learning and portfolio project, fully separate from the Movie Prediction App
(`C:\Users\dimbl\Desktop\Movies_Final_project`), which is never changed from here; a merge is decided at the end.

- Spec: `docs/superpowers/specs/2026-11-05-llm-movie-predictor-design.md` (source of truth for every decision)
- Roadmap: `docs/superpowers/plans/2026-11-05-roadmap.md`; one detailed plan per phase in the same folder
- Python package `llm_movies/` (Python 3.11, uv); React + Vite front end in `web/`; tests in `tests/`
- Data pipeline runs in Colab (CPU runtime), training and evaluation on Kaggle/Colab (T4), models and datasets on the
  Hugging Face Hub, serving on a free-CPU Hugging Face Docker Space. Notebooks only launch `python -m llm_movies....`.

## Hard rules

- When you finish a task that changes or creates code, always use the **code-reviewer** subagent.
- Whenever a new test is created, always use the **test-auditor** subagent on that test.
- Whenever a new frontend component is created, use the **ui-ux-reviewer** subagent to get feedback and improve the new component.

## Project rules (from the spec)

- **Pre-release inputs only.** `popularity`, `vote_count`, `vote_average`, `revenue` are never inputs. The form in
  `llm_movies/contract/form.py` defines every allowed input.
- **Leak safety.** Only films released strictly before a film's date count; an earlier film's revenue and score count
  once it is 182 days old (its budget from release); awards count only if dated before the release.
- **Time splits:** train ≤2013, calibration 2014–2015, validation 2016–2018, test 2019→base-model cutoff, clean
  holdout after the cutoff. Test and holdout are locked (`TestSetLocked`) and opened once, at the end; nothing may
  change because of their results.
- TMDB `0` budget/revenue means unknown. Money is compared in 2025 dollars via `data/cpi_us_annual.csv`.
- Tests run offline with saved fixtures. Secrets (`TMDB_READ_TOKEN`, `HF_TOKEN`) come from the environment, never git.

## Working from two places

Work alternates between Claude Code on this Windows PC and a claude.ai cloud session that reaches this folder through
a Linux shell. Both sides share state only through this folder:

- **Progress lives in the files:** tick the plan's checkboxes (`- [x]`) as steps finish, and keep `PROGRESS.md`
  current (last finished task, what is in progress, open questions). Read both before starting.
- **Commit at the end of every task**, so the other side starts from a clean `git status`.
- **Never share a virtual environment across operating systems:** `.venv/` belongs to Windows. The Linux shell uses
  `UV_PROJECT_ENVIRONMENT=$HOME/.venv-llm uv ...` so it never touches `.venv/`.
- Line endings are fixed by `.gitattributes` (`* text=auto eol=lf`; `*.ps1 text eol=crlf`).
- The three subagents and their PowerShell hooks run only in Claude Code on Windows. On the cloud side, reviews follow
  the same agent files, run by a fresh agent, without hook enforcement; note that in `PROGRESS.md`.
