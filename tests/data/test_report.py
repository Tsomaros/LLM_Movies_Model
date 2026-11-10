import sqlite3
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from llm_movies.config import DataConfig
from llm_movies.data import report as report_module
from llm_movies.data.build_db import build_db
from llm_movies.data.export import FILMS_FILE, SCHEMA, export
from llm_movies.data.report import report
from llm_movies.prompt.resolve import ResolvedFilm

MINI = Path(__file__).parent.parent / "fixtures" / "mini"
PERSON = {"name": "P", "films": 1, "revenue_usd": 1.0, "score": 7.0, "wins": 0, "nominations": 0}
LOCKED_ID = 987654


def film(**overrides) -> str:
    base = {
        "genres": [], "release_year": 2000, "release_month": 1, "release_day": 1, "runtime": None, "language": None,
        "certification": None, "release_type": None, "countries": [], "themes": [], "budget": None, "synopsis": None,
        "tagline": None, "source_material": None, "franchise": None, "installment": None, "director": None,
        "writers": [], "cast": [], "producers": [], "composer": None, "distributor": None, "companies": [],
        "competition": 0, "collaborations": 0, "director_age": None, "lead_age": None, "source_strength": None,
    }
    return ResolvedFilm(**{**base, **overrides}).to_json()


def parquet_of(path, rows):
    """rows: (tmdb_id, split, resolved_json, revenue_log, score)."""
    table = pa.Table.from_pylist(
        [
            {"tmdb_id": i, "release_date": date(2000, 1, 1), "split": s, "resolved_json": j, "revenue_log": r, "score": v,
             "score_weight": None if v is None else 1.0}
            for i, s, j, r, v in rows
        ],
        schema=SCHEMA,
    )
    pq.write_table(table, path)
    return path


def db_of(path, rows):
    """rows: (tmdb_id, split, vote_count, vote_average)."""
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE movies (tmdb_id INTEGER PRIMARY KEY, split TEXT, vote_count INTEGER, vote_average REAL)")
    con.executemany("INSERT INTO movies VALUES (?, ?, ?, ?)", rows)
    con.commit()
    con.close()
    return path


def line(text, prefix):
    return next(l for l in text.splitlines() if l.startswith(prefix))


CFG = DataConfig(min_votes=50)


@pytest.fixture
def hand_db(tmp_path):
    return db_of(tmp_path / "h.db", [
        # train votes 5, 10, 19, 20, 49, 50, 99, 100, 500; 1000 votes but no average
        *[(i, "train", v, 7.0) for i, v in enumerate([5, 10, 19, 20, 49, 50, 99, 100, 500], start=1)],
        (10, "train", 1000, None),
        (11, "calibration", 30, 6.0), (12, "validation", 200, 6.0),
        (13, "test", 100000, 9.0), (14, "holdout", 100000, 9.0), (15, "later", 100000, 9.0),
    ])


def test_report_has_threshold_table(hand_db, tmp_path):
    text = report(hand_db, parquet_of(tmp_path / "f.parquet", []), CFG)
    assert "| min_votes | train | calibration | validation | total |" in text
    # worked by hand: votes >= t with an average, open splits only (the locked 100000-vote films are not counted)
    assert "| 10 | 8 | 1 | 1 | 10 |" in text
    assert "| 20 | 6 | 1 | 1 | 8 |" in text
    assert "| 50 | 4 | 0 | 1 | 5 |" in text
    assert "| 100 | 2 | 0 | 1 | 3 |" in text
    assert "min_votes = 50" in text  # the current setting
    assert "threshold table comes from the database" in text and "`with score` comes from the parquet" in text


def test_films_per_split_are_counts_only_for_all_splits(hand_db, tmp_path):
    text = report(hand_db, parquet_of(tmp_path / "f.parquet", []), CFG)
    for name, count in {"train": 10, "calibration": 1, "validation": 1, "test": 1, "holdout": 1, "later": 1}.items():
        assert f"| {name} | {count} |" in text


def test_coverage_is_worked_by_hand_on_the_open_splits(hand_db, tmp_path):
    full = film(
        writers=[PERSON], producers=[PERSON], composer=PERSON, distributor=PERSON, source_material="novel",
        release_type="wide", tagline="A", competition=2, installment=2, collaborations=1, director_age=40,
        lead_age=30, source_strength=5,
    )
    two = film(producers=[PERSON], distributor=PERSON, source_material="original", lead_age=30)
    three = film(writers=[PERSON], composer=PERSON, release_type="wide", tagline="", competition=1, installment=3, director_age=50)
    four = film()
    path = parquet_of(tmp_path / "f.parquet", [
        (1, "train", full, 1.0, 7.0), (2, "train", two, 1.0, 7.0), (3, "calibration", three, 1.0, None), (4, "validation", four, None, 7.0),
        (LOCKED_ID, "test", full, 1.0, 9.0),  # a locked row in the file must change nothing
    ])
    text = report(hand_db, path, CFG)
    assert "with revenue | 3 | 75.0%" in text and "with score | 3 | 75.0%" in text
    expected = {
        "writers": "2 | 50.0%", "producers": "2 | 50.0%", "composer": "2 | 50.0%", "distributor": "2 | 50.0%",
        "source_material (not original)": "1 | 25.0%", "release_type": "2 | 50.0%", "tagline": "1 | 25.0%",
        "competition (above 0)": "2 | 50.0%", "installment": "2 | 50.0%", "collaborations (above 0)": "1 | 25.0%",
        "director_age": "2 | 50.0%", "lead_age": "2 | 50.0%", "source_strength": "1 | 25.0%",
    }
    for field, tail in expected.items():
        assert line(text, f"| {field} |").endswith(f"| {tail} |"), field
    assert "open splits" in text.lower()


def test_report_leaves_locked_films_out(hand_db, tmp_path):
    outcome = film(tagline="A blockbuster at the box office", synopsis="A Golden Globe smash hit")
    path = parquet_of(tmp_path / "f.parquet", [(1, "train", film(tagline="Calm"), 1.0, 7.0), (LOCKED_ID, "test", outcome, 5.0, 9.0)])
    text = report(hand_db, path, CFG)
    assert str(LOCKED_ID) not in text
    assert line(text, "| blockbuster |").split("|")[2].strip() == "0"
    assert line(text, "| Golden Globe |").split("|")[2].strip() == "0"
    assert line(text, "| smash hit |").split("|")[2].strip() == "0"
    assert "with score | 1 |" in text  # the locked 9.0 score is not counted


def test_audit_word_boundaries_counts_and_five_examples(hand_db, tmp_path):
    rows = [
        (21, "train", film(tagline="It GROSSED $100 million at the Box Office", synopsis="Calm"), 1.0, 7.0),
        (22, "train", film(synopsis="Grossly overrated; the grossing film"), 1.0, 7.0),
        (23, "calibration", film(tagline="An Oscar winner, Academy Awards and award-winning"), 1.0, 7.0),
        (24, "validation", film(tagline="The year's top blockbuster, boxoffice no"), 1.0, 7.0),
        *[(30 + i, "train", film(synopsis="A record-breaking run"), 1.0, 7.0) for i in range(7)],
        (25, "train", film(tagline="Won a Golden Globe, a smash hit"), 1.0, 7.0),
        # near misses only: each contains a pattern's letters but not as a whole word, so no count may change
        (40, "train", film(
            tagline="Engrossing, grossly Oscarish, Boscars Academy Awardsx blockbusterish Golden Globex",
            synopsis="preaward-winning and award-winningly record-breakingly, smash hitter, Golden Globeless, nonblockbuster, "
            "xgolden globe, unrecord-breaking, thesmash hit",
        ), 1.0, 7.0),
    ]
    text = report(hand_db, parquet_of(tmp_path / "f.parquet", rows), CFG)

    def cells(prefix):
        parts = [c.strip() for c in line(text, prefix).strip("|").split("|")]
        return parts[1], parts[2]

    assert cells("| box office |") == ("1", "21")
    assert cells("| gross") == ("2", "21, 22")  # grossed and grossing; grossly is not a match
    assert cells("| dollar figure |") == ("1", "21")
    assert cells("| Oscar |") == ("1", "23")
    assert cells("| Academy Award |") == ("1", "23")
    assert cells("| award-winning |") == ("1", "23")
    assert cells("| blockbuster |") == ("1", "24")
    assert cells("| Golden Globe |") == ("1", "25")
    assert cells("| smash hit |") == ("1", "25")
    assert cells("| record-breaking |") == ("7", "30, 31, 32, 33, 34")  # up to five examples
    assert "| sequel |" not in text  # the installment is a legitimate input, so it is not audited


def test_report_reads_the_file_in_batches(hand_db, tmp_path, monkeypatch):
    rows = [(i, "train", film(tagline="a smash hit"), 1.0, 7.0) for i in range(1, 8)]
    path = parquet_of(tmp_path / "f.parquet", rows)
    monkeypatch.setattr(report_module, "BATCH_ROWS", 2)
    assert line(report(hand_db, path, CFG), "| smash hit |").split("|")[2].strip() == "7"


def test_report_on_the_mini_export_never_names_a_locked_film(tmp_path):
    cfg = DataConfig(base_model_cutoff=date(2020, 12, 31))
    db = tmp_path / "mini.db"
    build_db(MINI / "tmdb", MINI / "wd_films", MINI / "wd_people", db, date(2026, 11, 4), cfg=cfg)
    export(db, tmp_path / "out", cfg)
    text = report(db, tmp_path / "out" / FILMS_FILE, cfg)
    con = sqlite3.connect(db)
    locked_ids = [r[0] for r in con.execute("SELECT tmdb_id FROM movies WHERE split IN ('test', 'holdout', 'later')")]
    assert locked_ids
    audited = text.split("Outcome-word audit")[1]
    assert not any(f" {i}," in audited or f" {i} " in audited or f"| {i}" in audited for i in locked_ids)
    assert "| min_votes |" in text and "| holdout | 2 |" in text


def test_report_audit_uses_shared_patterns(hand_db, tmp_path, monkeypatch):
    import importlib
    import re

    from llm_movies.prompt import outcome_words

    # Add a probe pattern to the shared tuple and re-import report.py: if it audits with its own copy, the probe is absent.
    probe = ("zzprobe", re.compile(r"\bzzprobe\b", re.IGNORECASE))
    monkeypatch.setattr(outcome_words, "OUTCOME_PATTERNS", (*outcome_words.OUTCOME_PATTERNS, probe))
    try:
        reloaded = importlib.reload(report_module)
        rows = [(41, "train", film(tagline="a ZZProbe here"), 1.0, 7.0)]
        text = reloaded.report(hand_db, parquet_of(tmp_path / "f.parquet", rows), CFG)
    finally:
        monkeypatch.undo()
        importlib.reload(report_module)
    audited = text.split("Outcome-word audit")[1]
    assert line(audited, "| zzprobe |").split("|")[2].strip() == "1"
    assert "| dollar figure |" in audited
