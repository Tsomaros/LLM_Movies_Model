import json
import math
import re
import shutil
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from llm_movies.config import DataConfig
from llm_movies.data import export as export_module
from llm_movies.data import hub_store
from llm_movies.data.build_db import build_db
from llm_movies.data.export import FILMS_FILE, LOCKED_FILE, export, push
from llm_movies.prompt.render import render
from llm_movies.prompt.resolve import ResolvedFilm

MINI = Path(__file__).parent.parent / "fixtures" / "mini"
PULLED_AT = date(2026, 11, 4)
CFG = DataConfig(base_model_cutoff=date(2020, 12, 31))  # so a holdout exists in the mini set
OPEN, LOCKED = ("train", "calibration", "validation"), ("test", "holdout")


@pytest.fixture(scope="module")
def mini_db(tmp_path_factory):
    out = tmp_path_factory.mktemp("export_mini") / "mini.db"
    build_db(MINI / "tmdb", MINI / "wd_films", MINI / "wd_people", out, PULLED_AT, cfg=CFG)
    return out


@pytest.fixture
def edited_db(tmp_path, mini_db):
    """A copy of the mini database; call it with SQL statements to change the copy."""
    copy = tmp_path / "edited.db"
    shutil.copy(mini_db, copy)

    def edit(*statements):
        con = sqlite3.connect(copy)
        for statement in statements:
            con.execute(statement)
        con.commit()
        con.close()
        return copy

    return edit


def read(path):
    return pq.read_table(path).to_pylist()


def by_id(path):
    return {row["tmdb_id"]: row for row in read(path)}


def db_rows(db, where="1=1"):
    con = sqlite3.connect(db)
    rows = con.execute(f"SELECT tmdb_id, split FROM movies WHERE {where}").fetchall()
    con.close()
    return rows


def test_targets_null_rules(edited_db, tmp_path):
    db = edited_db(
        "UPDATE movies SET money_suspect = 1 WHERE tmdb_id = 24",
        "UPDATE movies SET revenue_incomplete = 1 WHERE tmdb_id = 77",
        "UPDATE movies SET vote_count = 49 WHERE tmdb_id = 155",
        "UPDATE movies SET vote_average = NULL WHERE tmdb_id = 272",
    )
    rows = by_id(export(db, tmp_path / "out", CFG))
    assert rows[24]["revenue_log"] is None and rows[24]["score"] is not None  # suspect
    assert rows[77]["revenue_log"] is None and rows[77]["score"] is not None  # incomplete
    assert rows[155]["score"] is None and rows[155]["revenue_log"] is not None  # 49 < min_votes 50
    assert rows[272]["score"] is None and rows[272]["revenue_log"] is not None  # no vote_average
    assert rows[680]["revenue_log"] is not None and rows[680]["score"] == pytest.approx(8.48)
    con = sqlite3.connect(db)
    (revenue,) = con.execute("SELECT revenue_2025 FROM movies WHERE tmdb_id = 680").fetchone()
    assert rows[680]["revenue_log"] == pytest.approx(math.log(revenue))


def test_vote_threshold_comes_from_the_config(edited_db, tmp_path):
    db = edited_db("UPDATE movies SET vote_count = 60 WHERE tmdb_id = 155")
    kept = by_id(export(db, tmp_path / "a", DataConfig(base_model_cutoff=CFG.base_model_cutoff, min_votes=60)))
    stricter = by_id(export(db, tmp_path / "b", DataConfig(base_model_cutoff=CFG.base_model_cutoff, min_votes=61)))
    assert kept[155]["score"] == pytest.approx(8.5) and stricter[155]["score"] is None


def test_rows_without_targets_dropped(edited_db, tmp_path):
    db = edited_db(
        "UPDATE movies SET money_suspect = 1, vote_count = 3 WHERE tmdb_id = 272",  # neither target
        "UPDATE movies SET revenue_2025 = NULL, vote_count = 3 WHERE tmdb_id = 24",  # neither target
        "UPDATE movies SET revenue_2025 = NULL WHERE tmdb_id = 77",  # score only: kept
    )
    out = tmp_path / "out"
    export(db, out, CFG)
    everything = by_id(out / FILMS_FILE) | by_id(out / LOCKED_FILE)
    assert 272 not in everything and 24 not in everything
    assert everything[77]["revenue_log"] is None and everything[77]["score"] is not None
    assert all(r["revenue_log"] is not None or r["score"] is not None for r in everything.values())


def test_score_weight_is_null_with_the_score_and_worked_by_hand(edited_db, tmp_path):
    db = edited_db("UPDATE movies SET vote_count = 10 WHERE tmdb_id = 155")
    rows = by_id(export(db, tmp_path / "out", CFG))
    assert rows[155]["score"] is None and rows[155]["score_weight"] is None
    assert rows[24]["score_weight"] == pytest.approx(math.log1p(19678))
    assert all((r["score"] is None) == (r["score_weight"] is None) for r in rows.values())


def test_two_files_one_for_the_open_splits_one_for_the_locked(mini_db, tmp_path):
    out = tmp_path / "out"
    path = export(mini_db, out, CFG)
    assert path == out / FILMS_FILE
    films, locked = read(out / FILMS_FILE), read(out / LOCKED_FILE)
    assert pq.read_schema(out / FILMS_FILE).equals(pq.read_schema(out / LOCKED_FILE))
    assert {r["split"] for r in films} == set(OPEN) and {r["split"] for r in locked} == set(LOCKED)
    expected = {s: {i for i, sp in db_rows(mini_db) if sp == s} for s in OPEN + LOCKED}
    assert {r["tmdb_id"] for r in films} == expected["train"] | expected["calibration"] | expected["validation"]
    assert {r["tmdb_id"] for r in locked} == expected["test"] | expected["holdout"]
    assert expected["holdout"] and expected["test"]
    assert not ({r["tmdb_id"] for r in films} & {r["tmdb_id"] for r in locked})
    later = {i for i, sp in db_rows(mini_db) if sp == "later"}
    assert later and not (later & ({r["tmdb_id"] for r in films} | {r["tmdb_id"] for r in locked}))
    assert sorted(r["tmdb_id"] for r in films) == [r["tmdb_id"] for r in films]  # tmdb_id order
    assert sorted(f.name for f in out.iterdir()) == [FILMS_FILE, LOCKED_FILE]


def test_revenue_rule_at_181_and_182_days(tmp_path):
    shard_dir = tmp_path / "tmdb"
    shard_dir.mkdir()
    line = next(l for l in (MINI / "tmdb" / "tmdb_0000.jsonl").read_text(encoding="utf-8").splitlines() if '"id": 77,' in l)
    (shard_dir / "tmdb_0000.jsonl").write_text(line + "\n", encoding="utf-8")
    released = date(2000, 10, 11)
    got = {}
    for days in (181, 182):
        db = tmp_path / f"d{days}.db"
        build_db(shard_dir, None, None, db, released + timedelta(days=days), cfg=DataConfig())
        got[days] = by_id(export(db, tmp_path / f"o{days}", DataConfig()))[77]
    assert got[181]["revenue_log"] is None and got[181]["score"] is not None
    assert got[182]["revenue_log"] is not None


def test_resolved_json_loads_and_renders(mini_db, tmp_path):
    out = tmp_path / "out"
    export(mini_db, out, CFG)
    for row in read(out / FILMS_FILE) + read(out / LOCKED_FILE):
        film = ResolvedFilm.from_json(row["resolved_json"])
        assert render(film)
        assert film.release_year == row["release_date"].year


def test_no_outcome_value_reaches_resolved_json(edited_db, tmp_path):
    db = edited_db("UPDATE movies SET vote_average = 6.54321, revenue_2025 = 123456789.123, vote_count = 77777 WHERE tmdb_id = 24")
    row = by_id(export(db, tmp_path / "out", CFG))[24]
    assert row["score"] == pytest.approx(6.54321)  # the sentinel really is the target
    text = row["resolved_json"]
    assert "6.54321" not in text and "123456789" not in text and "77777" not in text
    assert not ({"vote_average", "vote_count", "revenue", "popularity", "revenue_2025"} & set(json.loads(text)))


def test_two_exports_are_identical(mini_db, tmp_path):
    export(mini_db, tmp_path / "a", CFG)
    export(mini_db, tmp_path / "b", CFG)
    for name in (FILMS_FILE, LOCKED_FILE):
        assert read(tmp_path / "a" / name) == read(tmp_path / "b" / name)


def test_batch_size_does_not_change_the_rows(mini_db, tmp_path):
    export(mini_db, tmp_path / "big", CFG)
    export(mini_db, tmp_path / "small", CFG, batch_size=3)
    for name in (FILMS_FILE, LOCKED_FILE):
        assert read(tmp_path / "big" / name) == read(tmp_path / "small" / name)
    assert pq.ParquetFile(tmp_path / "small" / FILMS_FILE).num_row_groups > 1  # the small batches really split it


def test_a_failure_leaves_no_half_file_and_keeps_an_earlier_one(mini_db, tmp_path, monkeypatch):
    fresh = tmp_path / "fresh"
    real = export_module.resolve
    calls = []

    def fail_on_fifth(*args, **kwargs):
        calls.append(1)
        if len(calls) == 5:
            raise RuntimeError("killed")
        return real(*args, **kwargs)

    monkeypatch.setattr(export_module, "resolve", fail_on_fifth)
    with pytest.raises(RuntimeError):
        export(mini_db, fresh, CFG, batch_size=2)
    assert not fresh.exists() or list(fresh.iterdir()) == []

    monkeypatch.setattr(export_module, "resolve", real)
    export(mini_db, tmp_path / "kept", CFG)
    before = {n: (tmp_path / "kept" / n).read_bytes() for n in (FILMS_FILE, LOCKED_FILE)}
    calls.clear()
    monkeypatch.setattr(export_module, "resolve", fail_on_fifth)
    with pytest.raises(RuntimeError):
        export(mini_db, tmp_path / "kept", CFG, batch_size=2)
    assert {n: (tmp_path / "kept" / n).read_bytes() for n in (FILMS_FILE, LOCKED_FILE)} == before
    assert sorted(f.name for f in (tmp_path / "kept").iterdir()) == [FILMS_FILE, LOCKED_FILE]


def test_dropped_rows_are_not_resolved(edited_db, tmp_path, monkeypatch):
    db = edited_db("UPDATE movies SET money_suspect = 1, vote_count = 3 WHERE tmdb_id IN (24, 272)")
    real, calls = export_module.resolve, []
    monkeypatch.setattr(export_module, "resolve", lambda *a, **k: calls.append(1) or real(*a, **k))
    out = tmp_path / "out"
    export(db, out, CFG)
    written = len(read(out / FILMS_FILE)) + len(read(out / LOCKED_FILE))
    not_later = [i for i, sp in db_rows(db) if sp != "later"]
    assert len(calls) == written == len(not_later) - 2  # 24 and 272 dropped, later films skipped, nothing else lost


class FakeApi:
    def __init__(self, repo_private=True):
        self.calls, self.repo_private = [], repo_private

    def create_repo(self, **kwargs):
        self.calls.append(("create_repo", kwargs))

    def dataset_info(self, repo_id):
        self.calls.append(("dataset_info", repo_id))
        return type("Info", (), {"private": self.repo_private})()

    def upload_folder(self, **kwargs):
        self.calls.append(("upload_folder", kwargs))

    def names(self):
        return [name for name, _ in self.calls]


@pytest.fixture
def two_files(tmp_path):
    folder = tmp_path / "ready"
    folder.mkdir()
    for name in (FILMS_FILE, LOCKED_FILE):
        (folder / name).write_bytes(b"x")
    return folder


def test_push_is_private_dataset_and_only_the_two_files(two_files):
    api = FakeApi()
    push(two_files, api=api)
    assert api.names() == ["create_repo", "dataset_info", "upload_folder"]
    create, upload = api.calls[0][1], api.calls[2][1]
    assert create["repo_id"] == "Tsomaros/llm-movies-data" and create["private"] is True
    assert create["repo_type"] == "dataset" and create["exist_ok"] is True
    assert upload["repo_type"] == "dataset" and upload["repo_id"] == "Tsomaros/llm-movies-data"
    assert Path(upload["folder_path"]) == two_files
    assert sorted(upload["allow_patterns"]) == [FILMS_FILE, LOCKED_FILE]


def test_push_never_uploads_to_a_public_repo(two_files):
    api = FakeApi(repo_private=False)
    with pytest.raises(RuntimeError, match="private"):
        push(two_files, api=api)
    assert "upload_folder" not in api.names()


@pytest.mark.parametrize("present, absent", [(FILMS_FILE, LOCKED_FILE), (LOCKED_FILE, FILMS_FILE), (None, FILMS_FILE)])
def test_push_needs_both_files_before_anything(tmp_path, present, absent):
    folder = tmp_path / "half"
    folder.mkdir()
    if present:
        (folder / present).write_bytes(b"x")
    api = FakeApi()
    with pytest.raises(FileNotFoundError, match=absent):
        push(folder, api=api)
    assert api.calls == []
    with pytest.raises(FileNotFoundError):
        push(tmp_path / "missing", api=FakeApi())


def test_a_database_that_does_not_match_the_config_is_refused(mini_db, tmp_path):
    other_ranges = DataConfig(base_model_cutoff=CFG.base_model_cutoff, validation_years=(2016, 2019), test_from_year=2020)
    for name, cfg in {"no cutoff": DataConfig(), "other years": other_ranges}.items():
        out = tmp_path / name
        with pytest.raises(ValueError, match="rebuild") as stop:
            export(mini_db, out, cfg)
        assert not re.search(r"\d", str(stop.value))  # no id, count or year in the message
        assert not out.exists() or list(out.iterdir()) == []


def test_a_mismatch_on_a_row_that_would_be_dropped_is_still_refused(edited_db, tmp_path):
    db = edited_db("UPDATE movies SET money_suspect = 1, vote_count = 3, split = 'test' WHERE tmdb_id = 272")
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="rebuild"):
        export(db, out, CFG)
    assert not out.exists() or list(out.iterdir()) == []


@pytest.mark.parametrize("extra", ["", ", money_suspect = 1, vote_count = 3"])  # kept row, then a row that would be dropped
def test_a_stored_open_split_that_is_really_locked_is_refused(edited_db, tmp_path, extra):
    # film 272 is stored as 'train' but released 2019-06-10: with the 2020-12-31 cutoff that is the test split
    db = edited_db(f"UPDATE movies SET release_date = '2019-06-10'{extra} WHERE tmdb_id = 272")
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="rebuild"):
        export(db, out, CFG)
    assert not out.exists() or list(out.iterdir()) == []


def test_a_database_without_pulled_at_is_refused(edited_db, tmp_path):
    db = edited_db("DELETE FROM meta WHERE key = 'pulled_at'")
    with pytest.raises(ValueError, match="pulled_at"):
        export(db, tmp_path / "out", CFG)


def test_progress_follows_the_batch_size(mini_db, tmp_path, capsys):
    export(mini_db, tmp_path / "out", CFG, batch_size=10)
    assert "10 films resolved" in capsys.readouterr().out


def test_cli_exits_on_missing_token_before_doing_anything(tmp_path, monkeypatch, mini_db):
    monkeypatch.setattr(export_module, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(hub_store, "load_dotenv", lambda *a, **k: None)  # the token is read there; a real .env must not fill it
    monkeypatch.delenv("HF_TOKEN", raising=False)
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as stop:
        export_module.main(["--db", str(mini_db), "--out", str(out), "--push"])
    assert "HF_TOKEN" in str(stop.value)
    assert not out.exists()


def test_cli_exports_and_pushes_with_a_fake(tmp_path, monkeypatch, mini_db):
    monkeypatch.setattr(export_module, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    monkeypatch.setattr(export_module, "load_config", lambda: CFG)
    pushed = []
    monkeypatch.setattr(export_module, "push", lambda out_dir, **k: pushed.append(Path(out_dir)))
    out = tmp_path / "out"
    export_module.main(["--db", str(mini_db), "--out", str(out), "--push"])
    assert pushed == [out] and (out / FILMS_FILE).is_file() and (out / LOCKED_FILE).is_file()


def test_push_removes_a_line_break_around_the_token(tmp_path, monkeypatch):
    import huggingface_hub

    (tmp_path / FILMS_FILE).write_bytes(b"x")
    (tmp_path / LOCKED_FILE).write_bytes(b"x")
    tokens = []

    class Stop(Exception):
        pass

    def fake_api(token):
        tokens.append(token)
        raise Stop

    monkeypatch.setattr(huggingface_hub, "HfApi", fake_api)
    monkeypatch.setenv("HF_TOKEN", "hf_abc\r\n")
    with pytest.raises(Stop):
        export_module.push(tmp_path)
    assert tokens == ["hf_abc"]


def test_cli_treats_a_blank_token_as_missing(tmp_path, monkeypatch, mini_db):
    monkeypatch.setattr(export_module, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(hub_store, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("HF_TOKEN", " \r\n")
    with pytest.raises(SystemExit, match="HF_TOKEN"):
        export_module.main(["--db", str(mini_db), "--out", str(tmp_path / "out"), "--push"])


def test_cli_refuses_a_bad_token_before_exporting_without_showing_it(tmp_path, monkeypatch, mini_db):
    monkeypatch.setattr(export_module, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(hub_store, "load_dotenv", lambda: None)
    monkeypatch.setenv("HF_TOKEN", "secret\ntoken")
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as stop:
        export_module.main(["--db", str(mini_db), "--out", str(out), "--push"])
    assert "secret" not in str(stop.value)
    assert not out.exists()
