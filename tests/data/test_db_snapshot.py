from pathlib import Path

import pytest
from hub_fakes import FakeHub

from llm_movies.data import db_snapshot, hub_store

MINI = Path(__file__).parent.parent / "fixtures" / "mini"
REPO = "Tsomaros/llm-movies-raw"


def raw_hub(with_db=False):
    hub = FakeHub()
    for f in MINI.rglob("*"):
        if f.is_file():
            hub.files[f.relative_to(MINI).as_posix()] = f.read_bytes()
    hub.files["ids/movie_ids.json.gz"] = b"ids"
    hub.files["reports/data_summary.md"] = b"report"
    if with_db:
        hub.files["db/movies.db"] = b"SQLite format 3\x00 already here"
    return hub


def test_ensure_skips_when_the_database_is_in_the_repo(tmp_path, monkeypatch, capsys):
    hub = raw_hub(with_db=True)

    def no_build(*a, **k):
        raise AssertionError("must not build")

    monkeypatch.setattr(db_snapshot, "build_db", no_build)
    assert db_snapshot.ensure(tmp_path / "raw", tmp_path / "build" / "movies.db", REPO, api=hub) is False
    assert hub.downloads == [] and hub.commits == []
    assert not (tmp_path / "raw").exists() and not (tmp_path / "build" / "movies.db").exists()
    assert "db/movies.db" in capsys.readouterr().out


def test_ensure_builds_from_the_shards_and_uploads_once(tmp_path):
    hub = raw_hub()
    out = tmp_path / "build" / "movies.db"
    assert db_snapshot.ensure(tmp_path / "raw", out, REPO, api=hub) is True
    assert hub.commits == [["db/movies.db"]]
    assert hub.files["db/movies.db"].startswith(b"SQLite format 3") and hub.files["db/movies.db"] == out.read_bytes()
    assert hub.downloads and all(d.split("/")[0] in {"tmdb", "wd_films", "wd_people"} for d in hub.downloads)  # no ids/, reports/
    before = len(hub.downloads)
    assert db_snapshot.ensure(tmp_path / "raw2", tmp_path / "build2" / "movies.db", REPO, api=hub) is False  # second run: nothing
    assert len(hub.downloads) == before and len(hub.commits) == 1


def test_ensure_refuses_a_repo_without_tmdb_shards(tmp_path):
    hub = FakeHub()
    with pytest.raises(SystemExit, match="tmdb"):
        db_snapshot.ensure(tmp_path / "raw", tmp_path / "movies.db", REPO, api=hub)
    assert hub.commits == []


def test_fetch_downloads_only_the_database(tmp_path):
    hub = raw_hub(with_db=True)
    out = tmp_path / "build" / "movies.db"
    db_snapshot.fetch(out, REPO, api=hub)
    assert hub.downloads == ["db/movies.db"] and out.read_bytes() == hub.files["db/movies.db"]
    assert not any(p.name.startswith(".") for p in out.parent.iterdir())  # no staging leftovers


def test_fetch_without_the_database_says_to_run_part_a(tmp_path):
    with pytest.raises(SystemExit, match="part A"):
        db_snapshot.fetch(tmp_path / "movies.db", REPO, api=raw_hub())


def test_main_wires_ensure_and_fetch(tmp_path, monkeypatch):
    hub = raw_hub(with_db=True)
    monkeypatch.setattr(hub_store, "make_api", lambda: hub)
    out = tmp_path / "movies.db"
    db_snapshot.main(["ensure", "--raw-dir", str(tmp_path / "raw"), "--out", str(out), "--repo", REPO])
    assert hub.downloads == []
    db_snapshot.main(["fetch", "--out", str(out), "--repo", REPO])
    assert out.exists()
    with pytest.raises(SystemExit):
        db_snapshot.main(["ensure", "--out", str(out)])  # --raw-dir is needed to build


def test_has_file(tmp_path):
    hub = raw_hub(with_db=True)
    assert hub_store.has_file(REPO, "db/movies.db", api=hub) and not hub_store.has_file(REPO, "db/other", api=hub)
