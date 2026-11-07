"""Resumable TMDB pull: one JSONL shard per chunk of ids, safe to kill and rerun at any time.

python -m llm_movies.data.tmdb_pull --ids PATH --chunk N --out DIR [--limit K]
python -m llm_movies.data.tmdb_pull --ids PATH --all --repo REPO --out DIR [--limit K] [--max-chunks M]

`--chunk N` pulls one chunk into the local folder. `--all` goes through every chunk the manifest does not list, in order,
and after each finished chunk uploads its shard, its missing file and the manifest to the Hub repo in one commit
(under `tmdb/`). A dead session then loses only the chunk in progress: download `tmdb/`, run again.
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
from llm_movies.data import hub_store
from llm_movies.data.shards import Manifest, ShardWriter, done_keys
from llm_movies.data.tmdb_ids import chunk_ids, read_ids

API = "https://api.themoviedb.org/3/movie/{id}"
APPEND = "credits,keywords,release_dates"
MAX_BACKOFF = 60.0
REPO_DIR = "tmdb"  # the folder of shards in the Hub repo


class ManifestError(RuntimeError):
    """The local or the Hub's manifest does not match this run, so writing to the Hub could overwrite good state."""


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


def _check_state(manifest: Manifest, meta: dict | None, remote: dict | None) -> None:
    if meta is not None:
        for where, found in (("local manifest", manifest.meta()), ("Hub manifest", (remote or {}).get(Manifest.META_KEY))):
            if found is not None and found != meta:
                raise ManifestError(
                    f"The {where} was made with {found} but this run has {meta}: the chunk size or the id list changed, so chunks would "
                    "be cut differently. Nothing was pulled or uploaded."
                )
    if remote is not None:
        lost = sorted({int(k) for k in remote if k != Manifest.META_KEY} - manifest.done_chunks())
        if lost:
            raise ManifestError(
                f"The Hub's manifest lists chunks {lost[:5]}{'...' if len(lost) > 5 else ''} as done but the local manifest does not: "
                "the download of tmdb/ probably failed. Uploading now would overwrite the Hub's list. Run the download again."
            )


def pull_all(
    client,
    chunks: list[list[int]],
    out_dir: Path,
    manifest: Manifest,
    upload,
    *,
    limit: int | None = None,
    max_chunks: int | None = None,
    meta: dict | None = None,
    remote: dict | None = None,
) -> list[int]:
    """Pull every chunk not done in the manifest, in order, and `upload({repo path: file or bytes})` after each one.
    A chunk enters the manifest (in the upload, and on disk only once the upload worked) when it is fully pulled, so a
    failed upload leaves it open and the rerun uploads it without refetching. With `limit` a chunk is a sample: its shard
    and missing file are uploaded, the manifest is not touched. Stops after `max_chunks` chunks. Returns the chunks handled.

    `meta` ({"chunk_size", "chunks"}) is recorded in the manifest, and the run refuses to start if the local manifest was made
    with other sizes. `remote` is the Hub's manifest (None when there is none yet): the run refuses to start if the Hub lists a
    done chunk that the local manifest lacks (the download of `tmdb/` failed, and uploading would overwrite the Hub's list)
    or was made with other sizes. Both checks run before any request."""
    out_dir = Path(out_dir)
    _check_state(manifest, meta, remote)
    handled = []
    for chunk, ids in enumerate(chunks):
        if max_chunks is not None and len(handled) >= max_chunks:
            break
        if manifest.is_done(chunk):
            continue
        stats = pull_chunk(client, chunk, ids if limit is None else ids[:limit], out_dir, manifest, mark_done=False)
        files = {
            f"{REPO_DIR}/tmdb_{chunk:04d}.jsonl": out_dir / f"tmdb_{chunk:04d}.jsonl",
            f"{REPO_DIR}/missing_{chunk:04d}.txt": out_dir / f"missing_{chunk:04d}.txt",
        }
        if limit is None:
            current = json.loads(manifest.path.read_text(encoding="utf-8")) if manifest.path.exists() else {}
            recorded = {Manifest.META_KEY: meta} if meta is not None else {}
            files[f"{REPO_DIR}/manifest.json"] = json.dumps({**current, **recorded, str(chunk): stats}).encode("utf-8")
        upload(files)
        if limit is None:
            if meta is not None:
                manifest.set_meta(meta)
            manifest.mark_done(chunk, stats)
        handled.append(chunk)
    return handled


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Pull one chunk of TMDB films into a JSONL shard.")
    parser.add_argument("--ids", type=Path, required=True, help="TMDB movie id export (.json.gz)")
    parser.add_argument("--chunk", type=int)
    parser.add_argument("--all", action="store_true", help="every chunk not done yet, uploading to --repo after each")
    parser.add_argument("--repo", help="Hub dataset repo for --all, e.g. Tsomaros/llm-movies-raw (needs HF_TOKEN)")
    parser.add_argument("--max-chunks", type=int, help="with --all: stop after this many chunks")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="pull only the first K ids of the chunk; leaves the chunk unmarked")
    args = parser.parse_args(argv)
    if args.all == (args.chunk is not None):
        parser.error("give exactly one of --chunk N or --all")
    if args.all and not args.repo:
        parser.error("--all needs --repo")
    if args.max_chunks is not None and not args.all:
        parser.error("--max-chunks needs --all")

    load_dotenv()
    token = os.environ.get("TMDB_READ_TOKEN")
    if not token:
        raise SystemExit("TMDB_READ_TOKEN is not set; put it in the environment or a .env file.")
    chunks = chunk_ids(read_ids(args.ids), load_config().chunk_size)
    manifest = Manifest(args.out / "manifest.json")
    if args.all:
        api = hub_store.make_api()

        def upload(files):
            hub_store.upload_files(files, args.repo, api=api, message=f"tmdb: {', '.join(sorted(files))}")

        meta = {"chunk_size": load_config().chunk_size, "chunks": len(chunks)}
        remote = hub_store.read_json(args.repo, f"{REPO_DIR}/manifest.json", api=api)
        try:
            handled = pull_all(
                TmdbClient(token), chunks, args.out, manifest, upload, limit=args.limit, max_chunks=args.max_chunks, meta=meta, remote=remote
            )
        except ManifestError as error:
            raise SystemExit(str(error)) from None
        print(json.dumps({"chunks": handled}))
        return
    if not 0 <= args.chunk < len(chunks):
        raise SystemExit(f"Chunk {args.chunk} is out of range: {args.ids} has {len(chunks)} chunks (0..{len(chunks) - 1}).")
    ids = chunks[args.chunk]
    if args.limit is not None:
        ids = ids[: args.limit]
    stats = pull_chunk(TmdbClient(token), args.chunk, ids, args.out, manifest, mark_done=args.limit is None)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
