import hashlib
import json
import shutil
import sqlite3
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from llm_movies.config import DataConfig
from llm_movies.data.awards import FAMILIES
from llm_movies.data.build_db import build_db, main

MINI = Path(__file__).parent.parent / "fixtures" / "mini"
AWARDS_JSON = Path(__file__).parents[2] / "llm_movies" / "data" / "awards.json"
PULLED_AT = date(2026, 11, 4)
TABLES = ("movies", "people", "credits", "companies", "movie_companies", "keywords", "movie_keywords", "awards", "meta")


MINI_DIR = object()  # "use the mini set's directory"; None means deliberately no Wikidata


def build(tmp_path, name="out.db", tmdb=None, wd_films=MINI_DIR, wd_people=MINI_DIR, pulled_at=PULLED_AT):
    out = tmp_path / name
    counts = build_db(
        tmdb if tmdb is not None else MINI / "tmdb",
        MINI / "wd_films" if wd_films is MINI_DIR else wd_films,
        MINI / "wd_people" if wd_people is MINI_DIR else wd_people,
        out,
        pulled_at,
    )
    return out, counts


def rows(path, sql, *args):
    con = sqlite3.connect(path)
    try:
        return con.execute(sql, args).fetchall()
    finally:
        con.close()


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def film(tmdb_id, **extra):
    base = {
        "id": tmdb_id,
        "title": f"Film {tmdb_id}",
        "status": "Released",
        "release_date": "2015-06-01",
        "pulled_at": "2026-11-04T00:00:00+00:00",
    }
    return {**base, **extra}


def crew(person_id, name, job):
    return {"id": person_id, "name": name, "job": job}


def tmdb_dir(tmp_path, *shards):
    root = tmp_path / "tmdb"
    for n, records in enumerate(shards):
        write_jsonl(root / f"tmdb_{n:04d}.jsonl", records)
    return root


def build_records(tmp_path, *shards, **kwargs):
    return build(tmp_path, tmdb=tmdb_dir(tmp_path, *shards), **kwargs)


def all_records():
    return [r for f in sorted((MINI / "tmdb").glob("tmdb_*.jsonl")) for r in jsonl(f)]


# ---- the mini set ----


def test_counts_match_fixture(tmp_path):
    out, counts = build(tmp_path)
    # Plain counts over the JSONL, written without project code.
    kept_statuses = {"Released", "Post Production", "In Production", "Planned"}
    records = all_records()
    assert len({r["id"] for r in records}) == len(records)  # the mini set has no duplicates
    kept = [r for r in records if r.get("release_date") and r.get("status") in kept_statuses]
    jobs = {
        "Director": "director",
        "Writer": "writer",
        "Screenplay": "writer",
        "Producer": "producer",
        "Original Music Composer": "composer",
    }
    credit_keys, person_ids = set(), set()
    for r in kept:
        for c in r["credits"]["cast"]:
            credit_keys.add((r["id"], c["id"], "actor"))
            person_ids.add(c["id"])
        for c in r["credits"]["crew"]:
            if c["job"] in jobs:
                credit_keys.add((r["id"], c["id"], jobs[c["job"]]))
                person_ids.add(c["id"])
    company_links = {(r["id"], c["id"]) for r in kept for c in r["production_companies"]}
    keyword_links = {(r["id"], k["id"]) for r in kept for k in r["keywords"]["keywords"]}
    expected = {
        "movies": len(kept),
        "dropped": len(records) - len(kept),
        "duplicates": 0,
        "people": len(person_ids),
        "credits": len(credit_keys),
        "companies": len({c for _, c in company_links}),
        "keywords": len({k for _, k in keyword_links}),
    }
    assert expected["movies"] >= 30
    assert {k: counts[k] for k in expected} == expected
    assert rows(out, "SELECT COUNT(*) FROM movies") == [(expected["movies"],)]
    assert rows(out, "SELECT COUNT(*) FROM people") == [(expected["people"],)]
    assert rows(out, "SELECT COUNT(*) FROM credits") == [(expected["credits"],)]
    assert rows(out, "SELECT COUNT(*) FROM companies") == [(expected["companies"],)]
    assert rows(out, "SELECT COUNT(*) FROM movie_companies") == [(len(company_links),)]
    assert rows(out, "SELECT COUNT(*) FROM keywords") == [(expected["keywords"],)]
    assert rows(out, "SELECT COUNT(*) FROM movie_keywords") == [(len(keyword_links),)]


def test_mini_set_covers_what_later_tasks_need():
    records = {r["id"]: r for r in all_records()}
    assert {r["status"] for r in records.values()} >= {"Released", "Planned", "In Production"}
    assert any(r["budget"] == 0 for r in records.values()) and any(r["revenue"] == 0 for r in records.values())
    assert any(r["original_language"] != "en" for r in records.values())
    years = {int(r["release_date"][:4]) for r in records.values()}
    assert min(years) <= 2013 and years & {2014, 2015} and years & {2016, 2017, 2018} and max(years) >= 2019
    assert 475557 in records  # Joker


def test_writer_and_screenplay_map_to_writer(tmp_path):
    out, _ = build(tmp_path)
    # Barbie (346698): TMDB lists Gerwig (45400) and Baumbach (5656) as Writer. The Wolf of Wall Street (106646): Terence Winter
    # (62635) is the Screenplay credit. Both jobs are the writer role, and no other role holds them.
    assert rows(out, "SELECT person_id FROM credits WHERE movie_id = 346698 AND role = 'writer' ORDER BY person_id") == [(5656,), (45400,)]
    assert rows(out, "SELECT person_id FROM credits WHERE movie_id = 106646 AND role = 'writer'") == [(62635,)]
    assert not rows(out, "SELECT 1 FROM credits WHERE role NOT IN ('director','actor','writer','producer','composer')")

    # One person with both jobs on one film gets one writer row; a second writer keeps its own position.
    record = film(1, credits={"cast": [], "crew": [crew(7, "Ann", "Screenplay"), crew(8, "Bob", "Writer"), crew(7, "Ann", "Writer")]})
    out2, _ = build_records(tmp_path, [record], name="hand.db")
    assert rows(out2, "SELECT person_id, role, ord FROM credits ORDER BY person_id") == [(7, "writer", 0), (8, "writer", 1)]


def test_credit_roles_and_order(tmp_path):
    record = film(
        1,
        credits={
            "cast": [{"id": 20, "name": "B", "order": 1}, {"id": 10, "name": "A", "order": 0}, {"id": 10, "name": "A", "order": 5}],
            "crew": [
                crew(30, "Dir", "Director"),
                crew(31, "Prod", "Producer"),
                crew(32, "Exec", "Executive Producer"),
                crew(33, "Comp", "Original Music Composer"),
                crew(34, "Cam", "Director of Photography"),
                crew(35, "Prod2", "Producer"),
            ],
        },
    )
    out, _ = build_records(tmp_path, [record])
    got = rows(out, "SELECT person_id, role, ord FROM credits")
    assert sorted(got) == sorted(
        [(10, "actor", 0), (20, "actor", 1), (30, "director", 0), (31, "producer", 0), (35, "producer", 1), (33, "composer", 0)]
    )
    assert rows(out, "SELECT id FROM people ORDER BY id") == [(10,), (20,), (30,), (31,), (33,), (35,)]  # 32 and 34 have no mapped job


def test_one_person_in_two_roles_and_cast_without_order(tmp_path):
    # Person 7 directs and writes the same film: two rows, one per role. Cast entries with no "order" take their list position.
    record = film(
        1,
        credits={
            "cast": [{"id": 20, "name": "B"}, {"id": 21, "name": "C"}],
            "crew": [crew(7, "Ann", "Director"), crew(7, "Ann", "Writer")],
        },
    )
    out, _ = build_records(tmp_path, [record])
    assert sorted(rows(out, "SELECT person_id, role, ord FROM credits")) == [
        (7, "director", 0),
        (7, "writer", 0),
        (20, "actor", 0),
        (21, "actor", 1),
    ]
    assert rows(out, "SELECT COUNT(*) FROM people WHERE id = 7") == [(1,)]


def test_duplicate_film_across_shards_kept_once(tmp_path):
    tmdb = tmp_path / "tmdb"
    shutil.copytree(MINI / "tmdb", tmdb)
    first = jsonl(tmdb / "tmdb_0000.jsonl")[0]
    clone = {**first, "title": "A later copy", "credits": {"cast": [{"id": 999999, "name": "Ghost", "order": 0}], "crew": []}}
    with open(tmdb / "tmdb_0002.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(clone) + "\n")
    out, counts = build(tmp_path, tmdb=tmdb)
    _, clean_counts = build(tmp_path, name="clean.db")
    assert counts["duplicates"] == 1 and clean_counts["duplicates"] == 0
    assert counts["movies"] == clean_counts["movies"]
    assert rows(out, "SELECT COUNT(*), MIN(title) FROM movies WHERE tmdb_id = ?", first["id"]) == [(1, first["title"])]
    assert rows(out, "SELECT COUNT(*) FROM people WHERE id = 999999") == [(0,)]


def table_hashes(path):
    con = sqlite3.connect(path)
    try:
        out = {}
        for table in TABLES:
            digest = hashlib.sha256()
            # Stored order (rowid), not sorted: an unstable insertion order is a difference too.
            for row in con.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall():
                digest.update(repr(row).encode())
            out[table] = digest.hexdigest()
        return out
    finally:
        con.close()


def test_rebuild_is_identical(tmp_path):
    a, _ = build(tmp_path, name="a.db")
    b, _ = build(tmp_path, name="b.db")
    assert table_hashes(a) == table_hashes(b)
    assert all(rows(a, f"SELECT COUNT(*) FROM {t}")[0][0] > 0 for t in TABLES)


def test_awards_only_whitelisted_families(tmp_path):
    out, _ = build(tmp_path)
    got = {f for (f,) in rows(out, "SELECT DISTINCT family FROM awards")}
    assert got and got <= set(FAMILIES) and len(FAMILIES) == 7
    assert {k for (k,) in rows(out, "SELECT DISTINCT kind FROM awards")} <= {"won", "nominated"}

    # Independent expectation: every whitelisted raw event of a person in the people table is one row, keeping its award QID.
    family_of = {item["qid"]: fam for fam, items in json.loads(AWARDS_JSON.read_text(encoding="utf-8")).items() for item in items}
    people = {p for (p,) in rows(out, "SELECT id FROM people")}
    expected = []
    for rec in jsonl(MINI / "wd_people" / "wd_people.jsonl"):
        if rec["facts"] and rec["tmdb_id"] in people:
            for a in rec["facts"]["awards"]:
                if a["award_qid"] in family_of:
                    expected.append((rec["tmdb_id"], a["award_qid"], family_of[a["award_qid"]], a["kind"], a["when"]))
    assert len(expected) > 50
    stored = rows(out, 'SELECT person_id, award_qid, family, kind, "when" FROM awards')
    assert len(stored) == len(expected)  # no event is lost to a collapse
    assert sorted(stored, key=repr) == sorted(expected, key=repr)


def test_same_night_different_awards_stay_two_rows(tmp_path):
    oscars = [item["qid"] for item in json.loads(AWARDS_JSON.read_text(encoding="utf-8"))["oscar"]]
    assert len(oscars) >= 2
    first, second = oscars[:2]
    wd_people = tmp_path / "wd_people"
    event = {"kind": "won", "when": "2020-02-09"}
    write_jsonl(
        wd_people / "wd_people.jsonl",
        [
            {
                "tmdb_id": 7,
                "facts": {
                    "birth": None,
                    "awards": [{"award_qid": first, **event}, {"award_qid": second, **event}, {"award_qid": first, **event}],
                },
            }
        ],
    )
    record = film(1, credits={"cast": [{"id": 7, "name": "Ann", "order": 0}], "crew": []})
    out, counts = build_records(tmp_path, [record], wd_people=wd_people)
    got = rows(out, 'SELECT person_id, award_qid, family, kind, "when" FROM awards ORDER BY award_qid')
    assert got == sorted([(7, first, "oscar", "won", "2020-02-09"), (7, second, "oscar", "won", "2020-02-09")], key=lambda r: r[1])
    assert counts["awards"] == 2


def test_hand_built_person_with_non_whitelisted_award_gets_no_row(tmp_path):
    oscar_qid = json.loads(AWARDS_JSON.read_text(encoding="utf-8"))["oscar"][0]["qid"]
    wd_people = tmp_path / "wd_people"
    write_jsonl(
        wd_people / "wd_people.jsonl",
        [
            {
                "tmdb_id": 7,
                "facts": {
                    "birth": "1970-02-03",
                    "awards": [
                        {"award_qid": "Q1", "kind": "won", "when": "2000-01-01"},  # not a whitelisted award
                        {"award_qid": oscar_qid, "kind": "nominated", "when": None},  # undated stays NULL
                        {"award_qid": oscar_qid, "kind": "nominated", "when": None},  # identical: once
                    ],
                },
            },
            {
                "tmdb_id": 8,
                "facts": {"birth": None, "awards": [{"award_qid": oscar_qid, "kind": "won", "when": "2001-03-25"}]},
            },  # no credit
        ],
    )
    record = film(1, credits={"cast": [{"id": 7, "name": "Ann", "order": 0}], "crew": []})
    out, _ = build_records(tmp_path, [record], wd_people=wd_people)
    assert rows(out, 'SELECT person_id, award_qid, family, kind, "when" FROM awards') == [(7, oscar_qid, "oscar", "nominated", None)]
    assert rows(out, "SELECT birth FROM people WHERE id = 7") == [("1970-02-03",)]


# ---- the rules ----


def test_failure_leaves_out_untouched(tmp_path):
    out = tmp_path / "out.db"
    out.write_bytes(b"previous build")
    stale = out.with_suffix(".tmp")
    stale.write_bytes(b"stale junk")
    tmdb = tmdb_dir(tmp_path, [film(1)], [{"title": "no id", "release_date": "2015-01-01", "status": "Released"}])
    with pytest.raises(ValueError):
        build_db(tmdb, None, None, out, PULLED_AT)
    assert out.read_bytes() == b"previous build"
    assert not stale.exists()


def test_record_without_id_stops_with_shard_name(tmp_path):
    tmdb = tmdb_dir(tmp_path, [film(1)], [film(2), {"title": "no id", "release_date": "2015-01-01", "status": "Released"}])
    with pytest.raises(ValueError, match=r"tmdb_0001\.jsonl"):
        build_db(tmdb, None, None, tmp_path / "out.db", PULLED_AT)
    assert not (tmp_path / "out.db").exists()


def test_facts_null_is_handled(tmp_path):
    wd_films, wd_people = tmp_path / "wf", tmp_path / "wp"
    write_jsonl(wd_films / "wd_films.jsonl", [{"tmdb_id": 1, "facts": None}])
    write_jsonl(wd_people / "wd_people.jsonl", [{"tmdb_id": 7, "facts": None}])
    record = film(1, credits={"cast": [{"id": 7, "name": "Ann", "order": 0}], "crew": []})
    out, counts = build_records(tmp_path, [record], wd_films=wd_films, wd_people=wd_people)
    assert counts["movies"] == 1 and counts["awards"] == 0
    assert rows(out, "SELECT wd_qid FROM movies") == [(None,)]
    assert rows(out, "SELECT name, birth FROM people") == [("Ann", None)]


def test_dropped_film_adds_nothing(tmp_path):
    credits = {"cast": [{"id": 50, "name": "Only Here", "order": 0}], "crew": [crew(51, "Dir", "Director")]}
    extras = {"production_companies": [{"id": 60, "name": "Co"}], "keywords": {"keywords": [{"id": 70, "name": "kw"}]}, "credits": credits}
    dropped = [film(1, status="Rumored", **extras), film(2, release_date="", **extras), film(3, status="Canceled", **extras)]
    kept = film(4, credits={"cast": [{"id": 52, "name": "Kept", "order": 0}], "crew": []})
    out, counts = build_records(tmp_path, dropped + [kept])
    assert counts["movies"] == 1 and counts["dropped"] == 3
    assert rows(out, "SELECT id FROM people") == [(52,)]
    assert rows(out, "SELECT movie_id FROM credits") == [(4,)]
    for table in ("companies", "movie_companies", "keywords", "movie_keywords"):
        assert rows(out, f"SELECT COUNT(*) FROM {table}") == [(0,)]


def test_first_name_seen_wins(tmp_path):
    a = film(1, credits={"cast": [{"id": 7, "name": "First Name", "order": 0}], "crew": []})
    b = film(2, credits={"cast": [{"id": 7, "name": "Other Name", "order": 0}], "crew": []})
    out, _ = build_records(tmp_path, [a], [b])
    assert rows(out, "SELECT name FROM people WHERE id = 7") == [("First Name",)]


def test_build_works_without_wikidata(tmp_path):
    out, counts = build(tmp_path, wd_films=None, wd_people=None)
    assert counts["movies"] > 30 and counts["awards"] == 0
    assert counts["wd_films"] == 0 and counts["wd_people"] == 0
    assert rows(out, "SELECT COUNT(*) FROM people WHERE birth IS NOT NULL") == [(0,)]
    assert rows(out, "SELECT COUNT(*) FROM movies WHERE wd_qid IS NOT NULL") == [(0,)]
    out2, _ = build(tmp_path, name="two.db", wd_people=None)
    assert rows(out2, "SELECT COUNT(*) FROM movies WHERE wd_qid IS NOT NULL")[0][0] > 0


def test_mistyped_wikidata_dir_raises(tmp_path):
    out = tmp_path / "out.db"
    with pytest.raises(FileNotFoundError, match="nowhere"):
        build_db(MINI / "tmdb", tmp_path / "nowhere", MINI / "wd_people", out, PULLED_AT)
    with pytest.raises(FileNotFoundError, match="wd_people.jsonl"):
        build_db(MINI / "tmdb", MINI / "wd_films", tmp_path, out, PULLED_AT)  # a real directory without the shard
    assert not out.exists() and not out.with_suffix(".tmp").exists()


def test_wikidata_record_counts_in_meta(tmp_path):
    out, counts = build(tmp_path)
    films = len(jsonl(MINI / "wd_films" / "wd_films.jsonl"))
    people = len(jsonl(MINI / "wd_people" / "wd_people.jsonl"))
    assert (counts["wd_films"], counts["wd_people"]) == (films, people) and films > 0 and people > 0
    meta = dict(rows(out, "SELECT key, value FROM meta"))
    assert (meta["wd_films"], meta["wd_people"]) == (str(films), str(people))


def test_wikidata_counts_are_records_read_not_rows_kept(tmp_path):
    # Hand-built shards: 3 film records (one `facts: null`) and 4 people records (one null, one with no credit, one more).
    # The counts are the records in the shards, whatever the build keeps.
    wd_films, wd_people = tmp_path / "wf", tmp_path / "wp"
    write_jsonl(wd_films / "wd_films.jsonl", [{"tmdb_id": 1, "facts": None}, {"tmdb_id": 2, "facts": None}, {"tmdb_id": 99, "facts": None}])
    write_jsonl(
        wd_people / "wd_people.jsonl",
        [
            {"tmdb_id": 7, "facts": None},
            {"tmdb_id": 8, "facts": {"birth": "1980-01-01", "awards": []}},  # credited
            {"tmdb_id": 9, "facts": {"birth": "1981-01-01", "awards": []}},  # no credit in any film
            {"tmdb_id": 10, "facts": None},  # no credit and no facts
        ],
    )
    record = film(1, credits={"cast": [{"id": 7, "name": "Ann", "order": 0}, {"id": 8, "name": "Bo", "order": 1}], "crew": []})
    out, counts = build_records(tmp_path, [record], wd_films=wd_films, wd_people=wd_people)
    assert (counts["wd_films"], counts["wd_people"]) == (3, 4)
    meta = dict(rows(out, "SELECT key, value FROM meta"))
    assert (meta["wd_films"], meta["wd_people"]) == ("3", "4")
    assert rows(out, "SELECT id, birth FROM people ORDER BY id") == [(7, None), (8, "1980-01-01")]  # 9 is never added


def test_movies_columns(tmp_path):
    out, _ = build(tmp_path)
    columns = [r[1] for r in rows(out, "PRAGMA table_info(movies)")]
    assert "popularity" not in columns
    assert {"vote_count", "vote_average", "split", "release_date", "budget_2025", "revenue_2025"} <= set(columns)
    # Post-release fields (popularity, votes, revenue) may exist only as target columns of movies, in no other table.
    allowed = {"vote_count", "vote_average", "revenue_2025", "revenue_source", "revenue_incomplete"}
    for table in TABLES:
        for column in (r[1] for r in rows(out, f"PRAGMA table_info({table})")):
            if any(word in column for word in ("popularity", "vote", "revenue")):
                assert table == "movies" and column in allowed, f"{table}.{column}"
    countries, genres, release_date = rows(out, "SELECT countries, genres, release_date FROM movies WHERE tmdb_id = 475557")[0]
    assert json.loads(countries) and "Crime" in json.loads(genres)
    assert release_date == "2019-10-01"
    assert {v for (v,) in rows(out, "SELECT DISTINCT money_suspect FROM movies")} <= {0, 1}


def test_movies_split_filled(tmp_path):
    out, _ = build(tmp_path)  # the default config: train <= 2013, calibration 2014-15, validation 2016-18, test 2019+
    assert rows(out, "SELECT COUNT(*) FROM movies WHERE split IS NULL") == [(0,)]
    splits = dict(rows(out, "SELECT tmdb_id, split FROM movies"))
    assert splits[77] == "train"  # Memento, 2000
    assert splits[106646] == "train"  # The Wolf of Wall Street, 2013-12-25
    assert splits[157336] == "calibration"  # Interstellar, 2014
    assert splits[273248] == "calibration"  # The Hateful Eight, 2015-12-25
    assert splits[279641] == "validation"  # Whiskey Tango Foxtrot, 2016
    assert splits[374720] == "validation"  # Dunkirk, 2017
    assert splits[475557] == "test"  # Joker, 2019
    assert splits[577922] == "test"  # Tenet, 2020
    assert splits[1003598] == "test"  # Avengers: Secret Wars, 2027 (no cutoff configured)


def test_movies_split_uses_given_config_and_pulled_at(tmp_path):
    cfg = replace(DataConfig(), base_model_cutoff=date(2022, 12, 31))
    out = tmp_path / "cut.db"
    build_db(MINI / "tmdb", MINI / "wd_films", MINI / "wd_people", out, PULLED_AT, cfg=cfg)
    splits = dict(rows(out, "SELECT tmdb_id, split FROM movies"))
    assert splits[577922] == "test"  # Tenet, 2020: before the cutoff
    assert splits[872585] == "holdout"  # Oppenheimer, 2023-07-19: after the cutoff, over 182 days old
    assert splits[791523] == "later"  # Wildwood, 2026-10-19: 16 days old
    assert splits[1003598] == "later"  # Avengers: Secret Wars, 2027: not yet released
    assert splits[157336] == "calibration"


def test_pulled_at_drives_incomplete_flag_and_meta(tmp_path):
    out, _ = build(tmp_path)
    assert rows(out, "SELECT revenue_incomplete FROM movies WHERE tmdb_id = 791523") == [(1,)]  # Wildwood, 16 days old
    assert rows(out, "SELECT revenue_incomplete FROM movies WHERE tmdb_id = 475557") == [(0,)]  # Joker
    meta = dict(rows(out, "SELECT key, value FROM meta"))
    assert meta["pulled_at"] == "2026-11-04"
    assert int(meta["movies"]) == rows(out, "SELECT COUNT(*) FROM movies")[0][0]


def test_indexes_exist(tmp_path):
    out, _ = build(tmp_path)
    plan = " ".join(r[3] for r in rows(out, "EXPLAIN QUERY PLAN SELECT * FROM credits WHERE person_id = 1"))
    assert "USING INDEX" in plan or "USING COVERING INDEX" in plan
    assert len(rows(out, "SELECT name FROM sqlite_master WHERE type = 'index' AND name NOT LIKE 'sqlite_%'")) >= 4


# ---- the command line ----


def run_cli(tmp_path, *extra):
    out = tmp_path / "cli.db"
    main(
        [
            "--tmdb",
            str(MINI / "tmdb"),
            "--wd-films",
            str(MINI / "wd_films"),
            "--wd-people",
            str(MINI / "wd_people"),
            "--out",
            str(out),
            *extra,
        ]
    )
    return out


def test_cli_prints_counts(tmp_path, capsys):
    run_cli(tmp_path)
    printed = capsys.readouterr().out
    assert "movies" in printed and "awards" in printed


def test_cli_pulled_at_defaults_to_latest_in_shards(tmp_path):
    tmdb = tmp_path / "tmdb"
    write_jsonl(tmdb / "tmdb_0000.jsonl", [film(1, pulled_at="2026-01-02T10:00:00+00:00")])
    write_jsonl(tmdb / "tmdb_0001.jsonl", [film(2, pulled_at="2026-03-04T01:00:00+00:00"), film(3, pulled_at="2026-02-01T00:00:00+00:00")])
    out = tmp_path / "x.db"
    write_jsonl(tmp_path / "n1" / "wd_films.jsonl", [])
    write_jsonl(tmp_path / "n2" / "wd_people.jsonl", [])
    main(["--tmdb", str(tmdb), "--wd-films", str(tmp_path / "n1"), "--wd-people", str(tmp_path / "n2"), "--out", str(out)])
    assert dict(rows(out, "SELECT key, value FROM meta"))["pulled_at"] == "2026-03-04"


def test_cli_on_mini_set_uses_its_fixed_date(tmp_path):
    out = run_cli(tmp_path)
    assert dict(rows(out, "SELECT key, value FROM meta"))["pulled_at"] == "2026-11-04"


def test_cli_pulled_at_option_wins(tmp_path):
    out = run_cli(tmp_path, "--pulled-at", "2027-05-06")
    assert dict(rows(out, "SELECT key, value FROM meta"))["pulled_at"] == "2027-05-06"
    assert rows(out, "SELECT revenue_incomplete FROM movies WHERE tmdb_id = 791523") == [(0,)]
