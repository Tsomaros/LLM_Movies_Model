"""The dataset training reads: one parquet file of open-split films, one of locked-split films.

python -m llm_movies.data.export --db PATH --out DIR [--push]

    export(db, out_dir, cfg, batch_size=50_000) -> Path of films.parquet
    push(out_dir, repo_id, private=True, api=None)

Two files, so that opening the usual one never opens the locked sets:
- films.parquet  holds the train, calibration and validation films;
- locked.parquet holds the test and holdout films (same columns). Only the final evaluation may read it.
- "later" films go in neither. A film's split is the `movies.split` column the database was built with.

Columns: tmdb_id, release_date, split, resolved_json (ResolvedFilm.to_json, and no other film data), revenue_log
(natural log of revenue_2025; null when missing, money_suspect or revenue_incomplete), score (vote_average; null when
vote_count < cfg.min_votes or missing), score_weight (log1p(vote_count); null exactly when score is null).
Films with neither target are dropped before resolving, because resolving is the slow step.

Films are read in tmdb_id order and written in batches, so memory does not grow with the film count. Each file is
written under a temporary name and moved into place at the end, so a failure leaves no half file and keeps an earlier
export in place (the two moves are separate: a kill between them can leave a new films.parquet with an old locked.parquet).
"""

import argparse
import math
import os
import sqlite3
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv

from llm_movies.config import DataConfig, load_config
from llm_movies.data.hub_store import ensure_private_repo, hf_token, make_api
from llm_movies.eval.splits import assign_split
from llm_movies.features.computed import Context
from llm_movies.prompt.resolve import film_row, film_to_form, form_as_of, resolve

FILMS_FILE = "films.parquet"
LOCKED_FILE = "locked.parquet"
BATCH_ROWS = 50_000
DEFAULT_REPO = "Tsomaros/llm-movies-data"
OPEN_SPLITS = ("train", "calibration", "validation")
LOCKED_SPLITS = ("test", "holdout")

SCHEMA = pa.schema([
    ("tmdb_id", pa.int64()),
    ("release_date", pa.date32()),
    ("split", pa.string()),
    ("resolved_json", pa.string()),
    ("revenue_log", pa.float64()),
    ("score", pa.float64()),
    ("score_weight", pa.float64()),
])

_STALE = (
    "The database's stored splits do not match the current config (year ranges, cutoff or pulled_at). "
    "Rebuild the database with the current config (rebuild, then export again); nothing was written."
)

_SELECT = (
    "SELECT tmdb_id, release_date, split, revenue_2025, money_suspect, revenue_incomplete, vote_count, vote_average "
    "FROM movies WHERE release_date IS NOT NULL ORDER BY tmdb_id"
)


def _targets(revenue, suspect, incomplete, votes, average, min_votes):
    """(revenue_log, score, score_weight), each None when it does not apply."""
    revenue_log = math.log(revenue) if revenue is not None and revenue > 0 and not suspect and not incomplete else None
    has_score = average is not None and votes is not None and votes >= min_votes
    return revenue_log, (float(average) if has_score else None), (math.log1p(votes) if has_score else None)


class _Sink:
    """A parquet file written in batches under a temporary name."""

    def __init__(self, path: Path, batch_size: int):
        self.path, self.tmp, self.batch_size = path, path.with_name(path.name + ".tmp"), batch_size
        self.rows: list[dict] = []
        self.writer = pq.ParquetWriter(self.tmp, SCHEMA)

    def add(self, row: dict) -> None:
        self.rows.append(row)
        if len(self.rows) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if self.rows:
            self.writer.write_table(pa.Table.from_pylist(self.rows, schema=SCHEMA))
            self.rows = []

    def close(self) -> None:
        self.flush()
        self.writer.close()

    def abort(self) -> None:
        try:
            self.writer.close()
        finally:
            self.tmp.unlink(missing_ok=True)


def _pulled_at(con: sqlite3.Connection) -> date:
    found = con.execute("SELECT value FROM meta WHERE key = 'pulled_at'").fetchone()
    if found is None:
        raise ValueError("The database has no pulled_at in its meta table, so its splits cannot be checked. Rebuild it.")
    return date.fromisoformat(found[0])


def export(db: Path, out_dir: Path, cfg: DataConfig, batch_size: int = BATCH_ROWS) -> Path:
    """Write films.parquet (train, calibration, validation) and locked.parquet (test, holdout) into `out_dir`.
    Returns the films.parquet path: the open file only. The locked file is never returned or read here.

    Every film's stored split is checked against `assign_split` under `cfg` and the database's pulled_at. Any mismatch
    (the config changed after the build) raises ValueError without naming a film, and nothing is written: a stale split
    could otherwise put a test film in the open file."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sinks = {name: _Sink(out_dir / name, batch_size) for name in (FILMS_FILE, LOCKED_FILE)}
    con = sqlite3.connect(db)
    try:
        pulled_at = _pulled_at(con)
        ctx = Context.from_db(con, cfg)
        seen = 0
        for tmdb_id, released, split, revenue, suspect, incomplete, votes, average in con.execute(_SELECT):
            if assign_split(date.fromisoformat(released), cfg, pulled_at) != split:
                raise ValueError(_STALE)  # before anything else, so a dropped row is checked too
            if split in OPEN_SPLITS:
                sink = sinks[FILMS_FILE]
            elif split in LOCKED_SPLITS:
                sink = sinks[LOCKED_FILE]
            else:
                continue  # "later": too recent to have settled
            revenue_log, score, weight = _targets(revenue, suspect, incomplete, votes, average, cfg.min_votes)
            if revenue_log is None and score is None:
                continue  # no target: not worth resolving
            form = film_to_form(film_row(con, tmdb_id))
            resolved = resolve(form, form_as_of(form), ctx)
            sink.add({
                "tmdb_id": tmdb_id, "release_date": date.fromisoformat(released), "split": split,
                "resolved_json": resolved.to_json(), "revenue_log": revenue_log, "score": score, "score_weight": weight,
            })
            seen += 1
            if seen % batch_size == 0:
                print(f"{seen} films resolved", flush=True)
        for sink in sinks.values():
            sink.close()
    except BaseException:
        for sink in sinks.values():
            sink.abort()
        raise
    finally:
        con.close()
    for sink in sinks.values():
        os.replace(sink.tmp, sink.path)
    print(f"{seen} films written to {out_dir}", flush=True)
    return out_dir / FILMS_FILE


def push(out_dir: Path, repo_id: str = DEFAULT_REPO, private: bool = True, api=None) -> None:
    """Upload the two parquet files in `out_dir` to a Hugging Face dataset repo (created private if missing).
    The token is HF_TOKEN from the environment; `api` is for tests. Nothing else in the folder is uploaded.
    Both files must exist. An existing repo is never changed: if `private` is asked and the repo is public (create_repo
    leaves an existing repo as it is), this raises and uploads nothing, since locked.parquet holds the test labels."""
    missing = [name for name in (FILMS_FILE, LOCKED_FILE) if not (Path(out_dir) / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Nothing pushed: {', '.join(missing)} not found in {out_dir}")
    if api is None:
        api = make_api()
    ensure_private_repo(api, repo_id, private)
    api.upload_folder(
        folder_path=str(out_dir), repo_id=repo_id, repo_type="dataset", allow_patterns=[FILMS_FILE, LOCKED_FILE]
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export the training dataset (parquet) from the SQLite database.")
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--push", action="store_true", help="upload to the Hugging Face dataset repo (needs HF_TOKEN)")
    args = parser.parse_args(argv)
    load_dotenv()
    if args.push:
        hf_token()  # before the export, so a bad token costs nothing
    export(args.db, args.out, load_config())
    if args.push:
        push(args.out)


if __name__ == "__main__":
    main()
