"""One markdown table comparing training runs, from each run's `runs/<name>/best/state.json` on the runs repo.

    python -m llm_movies.eval.compare_runs --runs e1_small,e1_large --out reports/experiments.md

Every metric column comes from `full_validation`, the best checkpoint scored once on the whole validation split when its
run ended, so runs that stopped early on different validation samples are compared on the same films. "check pinball" is
the early-stopping metric on the run's own check frame (its validation sample), shown for reference only. A run still
training has no full validation yet: its metric cells stay empty rather than show sample numbers as full ones.
"""

import argparse
from pathlib import Path

from llm_movies.data import hub_store
from llm_movies.model.config import TrainConfig

DEFAULT_REPO = TrainConfig.hub_repo  # the dataclass default
NONE = "—"
HEADER = (
    "run", "best step", "stopped", "val pinball", "revenue mae_log", "revenue within_2x", "revenue coverage_80",
    "score mae", "score coverage_80", "check pinball",
)
# (target, metric, decimals) in column order, after val pinball
COLUMNS = (("revenue", "mae_log", 3), ("revenue", "within_2x", 3), ("revenue", "coverage_80", 3), ("score", "mae", 3), ("score", "coverage_80", 3))


def _num(value, decimals: int) -> str:
    return NONE if value is None else f"{value:.{decimals}f}"


def _row(name: str, state: dict | None) -> list[str]:
    if state is None:
        return [name, NONE, "no best checkpoint", *[NONE] * (len(HEADER) - 3)]
    full = state.get("full_validation") or {}
    cells = [name, str(state.get("step", NONE)), state.get("stopped_reason", NONE), _num(full.get("val_loss"), 4)]
    cells += [_num(full.get(target, {}).get(metric), d) for target, metric, d in COLUMNS]
    return [*cells, _num(state.get("best_metric"), 4)]


def compare(run_names: list[str], api=None, repo: str = DEFAULT_REPO) -> str:
    """Markdown: a note on where the numbers come from, then one row per run in the given order."""
    api = api or hub_store.make_api()
    note = (
        "Metrics are the best checkpoint's on the full validation split (`full_validation` in `best/state.json`); "
        "check pinball is the early-stopping metric on the run's validation sample. — means not available yet."
    )
    lines = [
        note,
        "",
        "| " + " | ".join(HEADER) + " |",
        "|" + "|".join("---" for _ in HEADER) + "|",
    ]
    for name in run_names:
        state = hub_store.read_json(repo, f"runs/{name}/best/state.json", api=api, repo_type="model")
        lines.append("| " + " | ".join(_row(name, state)) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare training runs from their best checkpoints on the Hub.")
    parser.add_argument("--runs", required=True, help="comma-separated run names, e.g. e1_small,e1_large")
    parser.add_argument("--out", type=Path, help="write the table here as well (e.g. reports/experiments.md)")
    parser.add_argument("--repo", default=DEFAULT_REPO, help=f"runs repo (default {DEFAULT_REPO})")
    args = parser.parse_args(argv)
    names = [n.strip() for n in args.runs.split(",") if n.strip()]
    table = compare(names, repo=args.repo)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(table, encoding="utf-8", newline="\n")
    print(table)


if __name__ == "__main__":
    main()
