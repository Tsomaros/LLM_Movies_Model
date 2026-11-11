"""Run the memorization probes for every candidate base model and keep the results in the private raw repo.

python -m llm_movies.eval.model_choice run --db PATH --out DIR [--repo REPO] [--models a,b] [--per-quarter 150]
    [--quarters 2019Q1:2026Q2] [--revenue-n 300] [--min-votes 100] [--seed 0] [--device cuda|cpu]

For each model: the director probe on up to `--per-quarter` films from every release quarter in `--quarters`, and the
revenue probe on `--revenue-n` validation films (the same films for every model: one seed). Writes `<slug>_director.csv`,
`<slug>_revenue.csv`, `summary.json` and `summary.md` in `--out`, and uploads them to `model_choice/` in the raw repo after
EACH model (one commit), so a lost session loses at most one model. A model already "ok" in the repo's summary.json (with the
same settings) is skipped, so a rerun resumes. A model that fails (a licence not accepted, out of memory) is recorded as
failed with its error (the token is never written) and the run goes on. Without a GPU the run stops unless --device cpu.

CANDIDATES is the one list of models; the notebooks and docs follow it. Gated models come after the open ones, and the
conditional one (checked only if its CPU time allows) comes last.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from llm_movies.data import hub_store
from llm_movies.eval.probe import (
    BASELINE_YEARS,
    director_probe,
    empirical_cutoff,
    load_director_films_by_quarter,
    load_revenue_films,
    make_generator,
    revenue_probe,
)

# Base (pretrained) checkpoints. Qwen ids without -Base are the post-trained models: not used.
CANDIDATES = (
    "Qwen/Qwen3-0.6B-Base",
    "Qwen/Qwen3-1.7B-Base",
    "HuggingFaceTB/SmolLM2-1.7B",
    "meta-llama/Llama-3.2-1B",  # gated: accept the licence on huggingface.co first
    "google/gemma-3-1b-pt",  # gated: accept the licence on huggingface.co first
    "google/gemma-4-E2B",  # conditional (Apache 2.0): only if its CPU time allows; may be multimodal and fail to load
)
GATED = ("meta-llama/Llama-3.2-1B", "google/gemma-3-1b-pt")
CONDITIONAL = ("google/gemma-4-E2B",)
FOLDER = "model_choice"
SUMMARY_JSON = f"{FOLDER}/summary.json"
ERROR_LIMIT = 500


def slug(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9.-]+", "_", model_id).lower()


def parse_quarters(text: str) -> tuple[tuple[int, int], tuple[int, int]]:
    """'2019Q1:2026Q2' as ((2019, 1), (2026, 2))."""
    match = re.fullmatch(r"(\d{4})Q([1-4]):(\d{4})Q([1-4])", text.strip())
    if match is None:
        raise ValueError(f"Quarters must look like 2019Q1:2026Q2, got {text!r}.")
    first, last = (int(match[1]), int(match[2])), (int(match[3]), int(match[4]))
    if first > last:
        raise ValueError(f"The first quarter {first} is after the last {last}.")
    return first, last


def _free_memory() -> None:
    gc.collect()
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _clean_error(error: BaseException) -> str:
    text = f"{type(error).__name__}: {error}"
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        text = text.replace(token, "***")
    return re.sub(r"hf_[A-Za-z0-9]{8,}", "***", text)[:ERROR_LIMIT]


def _entry(director: pd.DataFrame, revenue: pd.DataFrame, settings: dict) -> dict:
    baseline = director[director["year"].isin(BASELINE_YEARS)]
    recall = float(baseline["known"].mean()) if len(baseline) else None
    cutoff = "n/a"
    if recall:  # a model that names no 2019-2020 director has no cutoff to find
        try:
            cutoff = empirical_cutoff(director).isoformat()
        except ValueError:
            pass
    by_year = director.groupby("year")["known"].mean()
    return {
        "status": "ok",
        "empirical_cutoff": cutoff,
        "director_recall_2019_2020": recall,
        "recall_by_year": {str(int(y)): float(v) for y, v in by_year.items()},
        "revenue_known_pct": float(revenue["known"].mean() * 100) if len(revenue) else None,
        "films_asked": {"director": len(director), "revenue": len(revenue)},
        "settings": settings,
    }


def _fmt(value, pct: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}%" if pct else f"{value:.3f}"


def render_markdown(summary: dict) -> str:
    lines = [
        "# Base-model choice: memorization probes",
        "",
        "| model | status | empirical cutoff | director recall 2019-2020 | revenue known (within 2x) | films asked (director / revenue) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for model, e in summary["models"].items():
        if e["status"] != "ok":
            lines.append(f"| {model} | failed: {e['error']} | | | | |")
            continue
        asked = e["films_asked"]
        lines.append(
            f"| {model} | ok | {e['empirical_cutoff']} | {_fmt(e['director_recall_2019_2020'])} | "
            f"{_fmt(e['revenue_known_pct'], pct=True)} | {asked['director']} / {asked['revenue']} |"
        )
    lines += ["", "## Director recall per year", ""]
    for model, e in summary["models"].items():
        if e["status"] == "ok":
            lines.append(f"- {model}: " + ", ".join(f"{y} {v:.3f}" for y, v in e["recall_by_year"].items()))
    return "\n".join(lines) + "\n"


def _order(model: str) -> tuple[int, str]:
    return (CANDIDATES.index(model), model) if model in CANDIDATES else (len(CANDIDATES), model)


def run(
    db: Path,
    out: Path,
    repo: str = hub_store.DEFAULT_REPO,
    models: tuple[str, ...] = CANDIDATES,
    per_quarter: int = 150,
    first: tuple[int, int] = (2019, 1),
    last: tuple[int, int] = (2026, 2),
    revenue_n: int = 300,
    min_votes: int = 100,
    seed: int = 0,
    *,
    device: str | None = None,
    api=None,
    load_generator: Callable[[str, str], Callable[[list[str]], list[str]]] | None = None,
) -> dict:
    """Probe each model not yet ok in the repo's summary; upload after each one. Returns the final summary."""
    if device is None:
        import torch

        if not torch.cuda.is_available():
            raise SystemExit("No GPU found. Switch the runtime to a T4 GPU (or pass --device cpu to run on the CPU).")
        device = "cuda"
    load_generator = load_generator or (lambda model_id, dev: make_generator(model_id, dev))
    api = api or hub_store.make_api()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    settings = {
        "per_quarter": per_quarter, "first": f"{first[0]}Q{first[1]}", "last": f"{last[0]}Q{last[1]}",
        "revenue_n": revenue_n, "min_votes": min_votes, "seed": seed,
    }
    summary = hub_store.read_json(repo, SUMMARY_JSON, api=api) or {}
    summary.setdefault("models", {})
    director_films = load_director_films_by_quarter(db, per_quarter, first, last, seed, min_votes)
    revenue_films = load_revenue_films(db, revenue_n, seed)
    if director_films.empty or revenue_films.empty:
        raise SystemExit("The database has no films to ask about for these settings.")
    print(f"{len(director_films)} director films, {len(revenue_films)} revenue films, device {device}")
    for model in models:
        previous = summary["models"].get(model)
        if previous and previous.get("status") == "ok" and previous.get("settings") == settings:
            print(f"{model}: already ok in {repo}, skipped")
            continue
        print(f"{model}: probing")
        generate = None
        files: dict[str, Path] = {}
        try:
            generate = load_generator(model, device)
            director = director_probe(generate, director_films)
            revenue = revenue_probe(generate, revenue_films)
            summary["models"][model] = _entry(director, revenue, settings)
            for kind, result in (("director", director), ("revenue", revenue)):
                path = out / f"{slug(model)}_{kind}.csv"
                result.to_csv(path, index=False)
                files[f"{FOLDER}/{path.name}"] = path
        except Exception as error:  # noqa: BLE001 - any load or probe failure is recorded and the run goes on
            summary["models"][model] = {"status": "failed", "error": _clean_error(error), "settings": settings}
        finally:
            generate = None
            _free_memory()
        summary["models"] = {m: summary["models"][m] for m in sorted(summary["models"], key=_order)}
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (out / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
        files[SUMMARY_JSON] = out / "summary.json"
        files[f"{FOLDER}/summary.md"] = out / "summary.md"
        status = summary["models"][model]["status"]
        hub_store.upload_files(files, repo, api=api, message=f"model choice: {model} {status}")
        print(f"{model}: {status}, uploaded")
    print(render_markdown(summary))
    return summary


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("action", choices=("run",))
    p.add_argument("--db", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--repo", default=hub_store.DEFAULT_REPO)
    p.add_argument("--models", default=",".join(CANDIDATES), help="comma-separated Hub ids; default: all candidates")
    p.add_argument("--per-quarter", type=int, default=150)
    p.add_argument("--quarters", default="2019Q1:2026Q2")
    p.add_argument("--revenue-n", type=int, default=300)
    p.add_argument("--min-votes", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)
    first, last = parse_quarters(args.quarters)
    models = tuple(m.strip() for m in args.models.split(",") if m.strip())
    run(
        args.db, args.out, args.repo, models, per_quarter=args.per_quarter, first=first, last=last,
        revenue_n=args.revenue_n, min_votes=args.min_votes, seed=args.seed, device=args.device,
    )


if __name__ == "__main__":
    main()
