"""TMDB daily id export: which films exist, split into fixed-size chunks for the resumable pull.

python -m llm_movies.data.tmdb_ids --out PATH   downloads the day's export (yesterday's if today's is not out yet)
"""

import argparse
import gzip
import json
import os
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import requests


def export_url(day: date) -> str:
    return f"https://files.tmdb.org/p/exports/movie_ids_{day:%m_%d_%Y}.json.gz"


def read_ids(path: Path) -> list[int]:
    """Unique film ids, ascending, from a gzip JSON-lines export; skips adult and video entries.
    A missing `adult` or `video` field counts as false."""
    ids = set()
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("adult") or row.get("video"):
                continue
            ids.add(row["id"])
    return sorted(ids)


def chunk_ids(ids: list[int], size: int) -> list[list[int]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    return [ids[i : i + size] for i in range(0, len(ids), size)]


def download_export(path: Path, today: date | None = None, get=requests.get) -> Path:
    """Download the day's export to `path` unless it is already there. TMDB publishes each day's file in the
    morning, so a 404 falls back one day; any other failure raises. The file is streamed to a `.part` file and
    replaced only when complete, so `path` never holds half an export. `get` is for tests."""
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    today = today or datetime.now(UTC).date()
    tmp = path.with_name(path.name + ".part")
    try:
        for day in (today, today - timedelta(days=1)):
            response = get(export_url(day), stream=True, timeout=60)
            try:
                if response.status_code == 404:
                    continue
                response.raise_for_status()
                with open(tmp, "wb") as f:
                    f.writelines(response.iter_content(chunk_size=1 << 20))
            finally:
                response.close()
            os.replace(tmp, path)
            return path
        raise FileNotFoundError(f"No TMDB id export for {today} or the day before.")
    finally:
        tmp.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download the TMDB movie id export (once).")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    print(download_export(args.out))


if __name__ == "__main__":
    main()
