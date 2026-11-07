"""Id lists from the TMDB shards, for the Wikidata pull.

python -m llm_movies.data.shard_ids films|people --tmdb DIR --out FILE [--partial]

One id per line, unique and sorted. `films` are the shard records' ids; `people` are the cast and the crew in the
jobs build_db maps to a role, found by the same code that builds the credits.

The lists are only complete once the TMDB pull is: the command refuses if the manifest beside the shards lists fewer done chunks
than there are shards (or than the manifest's recorded chunk count), unless `--partial` says the run is a sample.
"""

import argparse
from pathlib import Path

from llm_movies.data.build_db import _credits, _tmdb_shards
from llm_movies.data.shards import Manifest, read_shard


def collect_ids(kind: str, tmdb_dir: Path) -> list[int]:
    ids = set()
    for shard in _tmdb_shards(tmdb_dir):
        for record in read_shard(shard):
            if kind == "films":
                ids.add(record["id"])
            else:
                ids.update(person_id for person_id, _, _, _ in _credits(record))
    return sorted(ids)


def check_complete(tmdb_dir: Path) -> None:
    """Exit unless every chunk of the pull is done: the manifest beside the shards lists as many done chunks as there are
    shards, and as many as the chunk count it recorded."""
    shards = len(_tmdb_shards(tmdb_dir))
    manifest_path = Path(tmdb_dir) / "manifest.json"
    manifest = Manifest(manifest_path)
    done = len(manifest.done_chunks())
    expected = max(shards, (manifest.meta() or {}).get("chunks", 0))
    if not manifest_path.exists():
        raise SystemExit(f"No manifest.json beside the shards in {tmdb_dir}: the TMDB pull is not finished. Pass --partial for a sample.")
    if done < expected:
        raise SystemExit(
            f"The TMDB pull is not finished: the manifest lists {done} done chunks of {expected}. "
            "Finish notebook 01 first, or pass --partial if this is a sample."
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="List the film or person ids in the TMDB shards, one per line.")
    parser.add_argument("kind", choices=("films", "people"))
    parser.add_argument("--tmdb", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--partial", action="store_true", help="allow an unfinished pull (a sample run)")
    args = parser.parse_args(argv)
    if not args.partial:
        check_complete(args.tmdb)
    ids = collect_ids(args.kind, args.tmdb)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".part")
    tmp.write_text("".join(f"{i}\n" for i in ids), encoding="utf-8", newline="\n")
    tmp.replace(args.out)
    print(f"{len(ids)} {args.kind} ids written to {args.out}")


if __name__ == "__main__":
    main()
