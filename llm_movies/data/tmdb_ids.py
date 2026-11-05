"""TMDB daily id export: which films exist, split into fixed-size chunks for the resumable pull."""

import gzip
import json
from datetime import date
from pathlib import Path


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
