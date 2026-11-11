"""One copy of movies.db in the private raw repo, so the GPU notebook never rebuilds it.

python -m llm_movies.data.db_snapshot ensure --raw-dir DIR --out PATH [--repo REPO]
python -m llm_movies.data.db_snapshot fetch --out PATH [--repo REPO]

`ensure`: when `db/movies.db` is already in the repo, say so and do nothing (no shard download, no build). Otherwise download
the three shard folders (tmdb, wd_films, wd_people; not the id export or the reports), build the database with `build_db`, and
upload it as `db/movies.db` in one commit. `fetch` downloads only that one file. The token is HF_TOKEN (write access).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from llm_movies.data import hub_store
from llm_movies.data.build_db import build_db, latest_pulled_at

DB_IN_REPO = "db/movies.db"
SHARD_FOLDERS = ("tmdb", "wd_films", "wd_people")


def ensure(raw_dir: Path, out: Path, repo: str = hub_store.DEFAULT_REPO, *, api=None) -> bool:
    """Make sure `db/movies.db` is in the repo. Returns True when it was built and uploaded, False when it was already there."""
    api = api or hub_store.make_api()
    if hub_store.has_file(repo, DB_IN_REPO, api=api):
        print(f"{DB_IN_REPO} is already in {repo}: nothing to download, build or upload")
        return False
    raw_dir, out = Path(raw_dir), Path(out)
    counts = {folder: hub_store.download(raw_dir / folder, repo, folder, api=api) for folder in SHARD_FOLDERS}
    if not counts["tmdb"]:
        raise SystemExit(f"No tmdb shards in {repo}: run notebooks 01 and 02 first.")
    pulled_at = latest_pulled_at(raw_dir / "tmdb")
    if pulled_at is None:
        raise SystemExit("No pulled_at found in the TMDB shards.")
    built = build_db(raw_dir / "tmdb", raw_dir / "wd_films", raw_dir / "wd_people", out, pulled_at)
    print(json.dumps(built))
    hub_store.upload_files({DB_IN_REPO: out}, repo, api=api, message="add movies.db built from the raw shards")
    print(f"uploaded {DB_IN_REPO} to {repo}")
    return True


def fetch(out: Path, repo: str = hub_store.DEFAULT_REPO, *, api=None) -> Path:
    """Download `db/movies.db` from the repo to `out`."""
    api = api or hub_store.make_api()
    out = Path(out)
    if not hub_store.has_file(repo, DB_IN_REPO, api=api):
        raise SystemExit(f"{DB_IN_REPO} is not in {repo}: run part A of notebook 10 (CPU runtime) first.")
    stage = out.parent / f".{out.name}.stage"
    try:
        source = hub_store.download_file(repo, DB_IN_REPO, stage, api=api)
        out.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, out)
    finally:
        import shutil

        shutil.rmtree(stage, ignore_errors=True)
    print(f"downloaded {DB_IN_REPO} from {repo} to {out}")
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Keep one built movies.db in the private raw repo (db/movies.db).")
    parser.add_argument("action", choices=("ensure", "fetch"))
    parser.add_argument("--raw-dir", type=Path, help="ensure: where the shards are downloaded to")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo", default=hub_store.DEFAULT_REPO)
    args = parser.parse_args(argv)
    if args.action == "ensure" and args.raw_dir is None:
        parser.error("ensure needs --raw-dir")
    api = hub_store.make_api()
    if args.action == "ensure":
        ensure(args.raw_dir, args.out, args.repo, api=api)
    else:
        fetch(args.out, args.repo, api=api)


if __name__ == "__main__":
    main()
