import json
from pathlib import Path

import pytest

from llm_movies.data import shard_ids
from llm_movies.data.shard_ids import collect_ids
from llm_movies.data.shards import Manifest

MINI = Path(__file__).parent.parent / "fixtures" / "mini" / "tmdb"


def write(path: Path, records: list[dict], torn: str = "") -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in records) + torn, encoding="utf-8")


def film(fid, cast=(), crew=()):
    return {"id": fid, "credits": {"cast": [{"id": c} for c in cast], "crew": [{"id": p, "job": j} for p, j in crew]}}


def test_films_are_unique_sorted_ids_across_shards(tmp_path):
    write(tmp_path / "tmdb_0001.jsonl", [film(30), film(10)])
    write(tmp_path / "tmdb_0000.jsonl", [film(20), film(10)])
    write(tmp_path / "missing_0000.txt", [{"id": 99}])
    assert collect_ids("films", tmp_path) == [10, 20, 30]


def test_people_are_cast_plus_mapped_crew_jobs_only(tmp_path):
    write(
        tmp_path / "tmdb_0000.jsonl",
        [
            film(1, cast=[7, 5], crew=[(3, "Director"), (4, "Writer"), (6, "Screenplay"), (8, "Producer"), (9, "Original Music Composer")]),
            film(2, cast=[5], crew=[(100, "Editor"), (101, "Casting"), (102, "Executive Producer")]),
        ],
    )
    assert collect_ids("people", tmp_path) == [3, 4, 5, 6, 7, 8, 9]


def test_torn_last_line_is_skipped(tmp_path):
    write(tmp_path / "tmdb_0000.jsonl", [film(1, cast=[5])], torn='{"id": 2, "credits": {"cast": [{"id": 6')
    assert collect_ids("films", tmp_path) == [1]
    assert collect_ids("people", tmp_path) == [5]


def test_records_without_credits_or_with_odd_entries_are_tolerated(tmp_path):
    write(tmp_path / "tmdb_0000.jsonl", [{"id": 1}, {"id": 2, "credits": {"cast": [{"name": "no id"}, "junk", {"id": 4}], "crew": None}}])
    assert collect_ids("people", tmp_path) == [4]
    assert collect_ids("films", tmp_path) == [1, 2]


def test_empty_folder_gives_nothing(tmp_path):
    assert collect_ids("films", tmp_path) == []


def test_mini_set_counts_by_plain_count():
    films, people = set(), set()
    for shard in sorted(MINI.glob("tmdb_*.jsonl")):
        for line in shard.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            films.add(record["id"])
            credits = record.get("credits") or {}
            people.update(p["id"] for p in credits.get("cast") or [])
            people.update(
                p["id"]
                for p in credits.get("crew") or []
                if p.get("job") in ("Director", "Writer", "Screenplay", "Producer", "Original Music Composer")
            )
    assert len(films) > 0 and len(people) > 0
    assert collect_ids("films", MINI) == sorted(films)
    assert collect_ids("people", MINI) == sorted(people)


def test_main_writes_one_id_per_line(tmp_path, capsys):
    write(tmp_path / "tmdb_0000.jsonl", [film(2), film(1)])
    out = tmp_path / "ids" / "films.txt"
    shard_ids.main(["films", "--tmdb", str(tmp_path), "--out", str(out), "--partial"])
    assert out.read_text(encoding="utf-8") == "1\n2\n"
    assert "2" in capsys.readouterr().out
    assert list(out.parent.iterdir()) == [out]


def test_main_rejects_unknown_kind(tmp_path):
    with pytest.raises(SystemExit):
        shard_ids.main(["studios", "--tmdb", str(tmp_path), "--out", str(tmp_path / "x")])


# ---- notebook 02 must not start before 01 is finished ----


def shards_and_manifest(tmp_path, shards=3, done=(), meta=None):
    for n in range(shards):
        write(tmp_path / f"tmdb_{n:04d}.jsonl", [film(n + 1, cast=[n + 10])])
    manifest = Manifest(tmp_path / "manifest.json")
    if meta:
        manifest.set_meta(meta)
    for n in done:
        manifest.mark_done(n, {"pulled": 1, "missing": 0})
    return tmp_path


def run(tmp_path, *extra, kind="films"):
    out = tmp_path / "out" / "ids.txt"
    shard_ids.main([kind, "--tmdb", str(tmp_path), "--out", str(out), *extra])
    return out


def test_refuses_when_the_manifest_lists_fewer_done_chunks_than_there_are_shards(tmp_path):
    shards_and_manifest(tmp_path, shards=3, done=(0, 1))
    with pytest.raises(SystemExit, match="partial"):
        run(tmp_path)
    assert not (tmp_path / "out").exists()


def test_refuses_without_a_manifest(tmp_path):
    shards_and_manifest(tmp_path, shards=2)
    with pytest.raises(SystemExit, match="manifest"):
        run(tmp_path, kind="people")


def test_refuses_when_the_manifest_says_more_chunks_exist_than_are_pulled(tmp_path):
    # every shard is done, but the pull was meant to have 241 chunks
    shards_and_manifest(tmp_path, shards=3, done=(0, 1, 2), meta={"chunk_size": 5000, "chunks": 241})
    with pytest.raises(SystemExit, match="241"):
        run(tmp_path)


def test_passes_when_every_chunk_is_done(tmp_path):
    shards_and_manifest(tmp_path, shards=3, done=(0, 1, 2), meta={"chunk_size": 5000, "chunks": 3})
    assert run(tmp_path).read_text() == "1\n2\n3\n"
    assert run(tmp_path, kind="people").read_text() == "10\n11\n12\n"


def test_partial_allows_a_sample(tmp_path):
    shards_and_manifest(tmp_path, shards=3, done=(), meta={"chunk_size": 5000, "chunks": 241})
    assert run(tmp_path, "--partial").read_text() == "1\n2\n3\n"
