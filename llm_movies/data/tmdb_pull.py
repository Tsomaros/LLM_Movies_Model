"""Resumable TMDB pull: one JSONL shard per chunk of ids, safe to kill and rerun at any time.

python -m llm_movies.data.tmdb_pull --ids PATH --chunk N --out DIR [--limit K]
"""

import argparse
import json
import math
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

from llm_movies.config import load_config
from llm_movies.data.shards import Manifest, ShardWriter, done_keys
from llm_movies.data.tmdb_ids import chunk_ids, read_ids

API = "https://api.themoviedb.org/3/movie/{id}"
APPEND = "credits,keywords,release_dates"
MAX_BACKOFF = 60.0


class TmdbError(RuntimeError):
    """A TMDB request failed for good. Messages carry the id and status, never the token."""


class TmdbClient:
    def __init__(self, token: str, session: requests.Session | None = None, max_per_second: float = 40, max_retries: int = 6):
        self._token = token
        self.session = session or requests.Session()
        self.max_retries = max_retries
        self._interval = 1 / max_per_second
        self._last = None

    def _throttle(self) -> None:
        if self._last is not None:
            wait = self._last + self._interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
        self._last = time.monotonic()

    def movie(self, tmdb_id: int) -> dict | None:
        """The film's full record, or None only for a real 404. Anything else that fails raises, so a
        failed chunk stays unmarked and resumes."""
        for attempt in range(self.max_retries + 1):
            backoff = min(MAX_BACKOFF, 2.0**attempt)
            self._throttle()
            try:
                response = self.session.get(
                    API.format(id=tmdb_id),
                    params={"append_to_response": APPEND},
                    headers={"Authorization": f"Bearer {self._token}"},
                    timeout=30,
                )
            except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError):
                if attempt == self.max_retries:
                    raise
                time.sleep(backoff)
                continue
            status = response.status_code
            if status == 200:
                return response.json()
            if status == 404:
                return None
            if status != 429 and status < 500:
                raise TmdbError(f"TMDB returned HTTP {status} for movie {tmdb_id}")
            if attempt == self.max_retries:
                raise TmdbError(f"TMDB returned HTTP {status} for movie {tmdb_id} after {self.max_retries} retries")
            time.sleep(self._delay(response, backoff))

    @staticmethod
    def _delay(response, backoff: float) -> float:
        try:
            seconds = float(response.headers.get("Retry-After"))
        except (TypeError, ValueError):
            return backoff
        if not math.isfinite(seconds):
            return backoff
        return max(seconds, 0)


def pull_chunk(client, chunk: int, ids: list[int], out_dir: Path, manifest: Manifest, *, mark_done: bool = True) -> dict:
    """Fetch every id not yet in the chunk's shard or missing file. Returns chunk totals
    ({"pulled", "missing"}), so a resumed run never understates. `mark_done=False` is for partial runs."""
    if manifest.is_done(chunk):
        return {"pulled": 0, "missing": 0}
    out_dir = Path(out_dir)
    shard = out_dir / f"tmdb_{chunk:04d}.jsonl"
    missing_file = out_dir / f"missing_{chunk:04d}.txt"
    pulled = done_keys(shard, "id")
    missing = done_keys(missing_file, "id")
    with ShardWriter(shard) as shard_writer, ShardWriter(missing_file) as missing_writer:
        for tmdb_id in ids:
            if tmdb_id in pulled or tmdb_id in missing:
                continue
            record = client.movie(tmdb_id)
            if record is None:
                missing_writer.write({"id": tmdb_id})
                missing.add(tmdb_id)
                continue
            # Key the shard by the id we asked for: if TMDB answers with another id (a redirect or merge),
            # resume would otherwise never see this id as done and refetch it forever.
            record = {**record, "id": tmdb_id, "pulled_at": datetime.now(UTC).isoformat()}
            shard_writer.write(record)
            pulled.add(tmdb_id)
    stats = {"pulled": len(pulled), "missing": len(missing)}
    if mark_done:
        manifest.mark_done(chunk, stats)
    return stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Pull one chunk of TMDB films into a JSONL shard.")
    parser.add_argument("--ids", type=Path, required=True, help="TMDB movie id export (.json.gz)")
    parser.add_argument("--chunk", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="pull only the first K ids of the chunk; leaves the chunk unmarked")
    args = parser.parse_args(argv)

    load_dotenv()
    token = os.environ.get("TMDB_READ_TOKEN")
    if not token:
        raise SystemExit("TMDB_READ_TOKEN is not set; put it in the environment or a .env file.")
    chunks = chunk_ids(read_ids(args.ids), load_config().chunk_size)
    if not 0 <= args.chunk < len(chunks):
        raise SystemExit(f"Chunk {args.chunk} is out of range: {args.ids} has {len(chunks)} chunks (0..{len(chunks) - 1}).")
    ids = chunks[args.chunk]
    if args.limit is not None:
        ids = ids[: args.limit]
    stats = pull_chunk(TmdbClient(token), args.chunk, ids, args.out, Manifest(args.out / "manifest.json"), mark_done=args.limit is None)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
