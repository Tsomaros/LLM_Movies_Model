"""The data report: what the dataset holds, and the numbers behind the `min_votes` decision.

python -m llm_movies.data.report --db PATH --parquet PATH --out reports/data_summary.md

    report(db, parquet, cfg=None) -> markdown

Only film counts per split cover every split. Everything else is the open splits (train, calibration, validation)
alone, so the report never shows a locked film's id, score or revenue. `parquet` is films.parquet; a split column
other than the open ones is skipped even if a file holds one, and locked.parquet is never opened. The parquet is read
in batches. The outcome-word audit only reports: nothing in the data changes because of it.
"""

import argparse
import json
import re
import sqlite3
from pathlib import Path

import pyarrow.parquet as pq

from llm_movies.config import DataConfig, load_config
from llm_movies.prompt.outcome_words import OUTCOME_PATTERNS

BATCH_ROWS = 50_000
OPEN_SPLITS = ("train", "calibration", "validation")
SPLITS = ("train", "calibration", "validation", "test", "holdout", "later")
THRESHOLDS = (10, 20, 50, 100)
EXAMPLES = 5

# (label, test on the resolved film), in table order
_COVERAGE = (
    ("writers", lambda f: bool(f["writers"])),
    ("producers", lambda f: bool(f["producers"])),
    ("composer", lambda f: f["composer"] is not None),
    ("distributor", lambda f: f["distributor"] is not None),
    ("source_material (not original)", lambda f: f["source_material"] not in (None, "original")),
    ("release_type", lambda f: f["release_type"] is not None),
    ("tagline", lambda f: bool((f["tagline"] or "").strip())),
    ("competition (above 0)", lambda f: f["competition"] > 0),
    ("installment", lambda f: f["installment"] is not None),
    ("collaborations (above 0)", lambda f: f["collaborations"] > 0),
    ("director_age", lambda f: f["director_age"] is not None),
    ("lead_age", lambda f: f["lead_age"] is not None),
    ("source_strength", lambda f: f["source_strength"] is not None),
)

# The outcome patterns are shared with the prompt's masking; the dollar figure is audited here only (kept third in the table)
_DOLLAR = ("dollar figure", re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:million|billion|thousand)\b)?", re.IGNORECASE))
_AUDIT = (*OUTCOME_PATTERNS[:2], _DOLLAR, *OUTCOME_PATTERNS[2:])


def _percent(count: int, total: int) -> str:
    return f"{100 * count / total:.1f}%" if total else "0.0%"


def _database_tables(db: Path) -> tuple[list[str], list[str]]:
    con = sqlite3.connect(db)
    try:
        counts = dict(con.execute("SELECT split, COUNT(*) FROM movies GROUP BY split"))
        rows = ["| split | films |", "| --- | --- |"] + [f"| {s} | {counts.get(s, 0)} |" for s in SPLITS]
        marks = ",".join("?" * len(OPEN_SPLITS))
        thresholds = ["| min_votes | train | calibration | validation | total |", "| --- | --- | --- | --- | --- |"]
        for t in THRESHOLDS:
            found = dict(con.execute(
                f"SELECT split, COUNT(*) FROM movies WHERE split IN ({marks}) AND vote_average IS NOT NULL AND vote_count >= ? GROUP BY split",
                (*OPEN_SPLITS, t),
            ))
            per = [found.get(s, 0) for s in OPEN_SPLITS]
            thresholds.append(f"| {t} | " + " | ".join(map(str, per)) + f" | {sum(per)} |")
        return rows, thresholds
    finally:
        con.close()


def _scan_parquet(parquet: Path) -> tuple[int, int, int, list[int], dict[str, tuple[int, list[int]]]]:
    """Open-split rows only: (rows, with revenue, with score, coverage counts, audit hits per pattern)."""
    rows = with_revenue = with_score = 0
    covered = [0] * len(_COVERAGE)
    hits = {label: (0, []) for label, _ in _AUDIT}
    columns = ["tmdb_id", "split", "resolved_json", "revenue_log", "score"]
    for batch in pq.ParquetFile(parquet).iter_batches(batch_size=BATCH_ROWS, columns=columns):
        data = batch.to_pydict()
        for i, split in enumerate(data["split"]):
            if split not in OPEN_SPLITS:
                continue
            rows += 1
            with_revenue += data["revenue_log"][i] is not None
            with_score += data["score"][i] is not None
            film = json.loads(data["resolved_json"][i])
            for k, (_, test) in enumerate(_COVERAGE):
                covered[k] += bool(test(film))
            text = f"{film.get('tagline') or ''}\n{film.get('synopsis') or ''}"
            for label, pattern in _AUDIT:
                if pattern.search(text):
                    count, ids = hits[label]
                    hits[label] = (count + 1, sorted(ids + [data["tmdb_id"][i]])[:EXAMPLES])
    return rows, with_revenue, with_score, covered, hits


def report(db: Path, parquet: Path, cfg: DataConfig | None = None) -> str:
    cfg = cfg if cfg is not None else load_config()
    split_rows, thresholds = _database_tables(db)
    rows, with_revenue, with_score, covered, hits = _scan_parquet(parquet)
    out = ["# Data summary", "", "## Films per split", "", "Counts from the database; all splits.", "", *split_rows, ""]
    out += [
        "## Open splits only", "",
        "Everything below covers the open splits (train, calibration, validation) only. Locked films are never shown.", "",
        f"Films in the dataset (open splits, with at least one target): {rows}", "",
        "| target | films | share |", "| --- | --- | --- |",
        f"| with revenue | {with_revenue} | {_percent(with_revenue, rows)} |",
        f"| with score | {with_score} | {_percent(with_score, rows)} |", "",
        "## Coverage", "",
        "| field | films | coverage |", "| --- | --- | --- |",
        *[f"| {label} | {n} | {_percent(n, rows)} |" for (label, _), n in zip(_COVERAGE, covered)], "",
        "## Score films by min_votes", "",
        "Open-split films with a vote_average and at least this many votes: the table for the vote-threshold decision.",
        "The threshold table comes from the database; `with score` comes from the parquet (at the current min_votes).", "",
        *thresholds, "", f"Current setting: min_votes = {cfg.min_votes}", "",
        "## Outcome-word audit (open splits only)", "",
        ("Films whose tagline or synopsis matches the pattern (word-bounded, case-insensitive), with up to "
         f"{EXAMPLES} example film ids. Report only: nothing in the data is changed."), "",
        "| pattern | films | example ids |", "| --- | --- | --- |",
        *[f"| {label} | {hits[label][0]} | {', '.join(map(str, hits[label][1])) or '-'} |" for label, _ in _AUDIT], "",
    ]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write the data summary report.")
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, required=True, help="films.parquet (the open splits)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    text = report(args.db, args.parquet, load_config())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8", newline="\n")
    print(text)


if __name__ == "__main__":
    main()
