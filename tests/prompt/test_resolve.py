import json
import math
import shutil
import sqlite3
from datetime import date
from pathlib import Path

import pytest

from llm_movies.config import DataConfig
from llm_movies.contract.form import validate
from llm_movies.features.computed import Context
from llm_movies.prompt.render import render
from llm_movies.prompt.resolve import ResolvedFilm, film_row, film_to_form, form_as_of, resolve

CFG = DataConfig()
TOY_STORY, TOY_STORY_2, TOY_STORY_4, TENET, RUPTURE, DUNKIRK, PARASITE, MEMENTO = 862, 863, 301528, 577922, 345925, 374720, 496243, 77
MINI = Path(__file__).parents[1] / "fixtures" / "mini"
THEMES_JSON = Path(__file__).parents[2] / "llm_movies" / "contract" / "themes.json"
POST_RELEASE = ("vote_average", "vote_count", "popularity", "revenue_2025", "revenue_source", "revenue_incomplete", "money_suspect", "split")


def resolved(con, ctx, movie_id):
    form = film_to_form(film_row(con, movie_id))
    return resolve(form, form_as_of(form), ctx)


# ---- film_row / film_to_form ----


def test_film_to_form_takes_the_billing_order_and_limits(mini_con, mini_ctx):
    form = film_to_form(film_row(mini_con, TOY_STORY_4))
    assert form["director"] == {"id": 84496, "name": "Josh Cooley"}
    assert [a["name"] for a in form["actors"]] == [
        "Tom Hanks", "Tim Allen", "Annie Potts", "Tony Hale", "Keegan-Michael Key",
        "Madeleine McGraw", "Christina Hendricks", "Jordan Peele", "Keanu Reeves", "Ally Maki",
    ]  # the ten billed, in billing order
    assert [w["name"] for w in form["writers"]] == ["Stephany Folsom", "Andrew Stanton"]
    assert [p["name"] for p in form["producers"]] == ["Jonas Rivera", "Mark Nielsen", "Galyn Susman"]
    assert form["composer"] == {"id": 7885, "name": "Randy Newman"}
    assert form["companies"] == [{"id": 3, "name": "Pixar"}]
    assert form["genres"] == ["Family", "Comedy", "Animation", "Adventure"]
    assert form["countries"] == ["US"]
    assert (form["release_year"], form["release_month"], form["release_day"]) == (2019, 6, 19)
    assert form["distributor"] == "Disney"
    assert form["tagline"] == "The adventure of a lifetime."
    assert form["synopsis"].startswith("Woody has always been confident")
    assert form["film_id"] == TOY_STORY_4
    assert form["franchise"] == {"id": 10194, "name": "Toy Story Collection"}


def test_film_to_form_caps_producers_and_keeps_credit_order(mini_con, mini_ctx):
    form = film_to_form(film_row(mini_con, RUPTURE))  # five producers are credited, in this order
    assert [p["name"] for p in form["producers"]] == ["Andrea Iervolino", "Monika Bacardi", "Andrew Lazar"]
    assert form["countries"] == ["CA", "US"]
    assert [c["name"] for c in form["companies"]] == ["Tango Pictures", "Splendid Film", "BondIt", "AMBI Group"]


def test_film_to_form_themes_follow_the_file_order(mini_con, mini_ctx):
    themes = json.loads(THEMES_JSON.read_text(encoding="utf-8"))["themes"]
    for movie_id in (TENET, TOY_STORY_4, MEMENTO):
        keywords = {k.casefold() for k in film_row(mini_con, movie_id)["keywords"]}
        # a film has a theme when any of its keywords is listed under it; in the file's order, at most 5
        expected = [t["name"] for t in themes if keywords & {k["name"].casefold() for k in t["keywords"]}][:5]
        assert film_to_form(film_row(mini_con, movie_id))["themes"] == expected
    assert film_to_form(film_row(mini_con, TENET))["themes"], "Tenet has at least one theme"


def test_film_to_form_caps_every_list_at_the_form_limits(mini_con):
    # limits from llm_movies/contract/form.py: actors 10, writers 3, producers 3, companies 5, countries 5, genres 5, themes 5.
    # No film in the mini set exceeds all of them, so a long list is built by hand; names say the position in the credit order.
    def refs(kind, n):
        return [{"id": 1000 + i, "name": f"{kind} {i}"} for i in range(n)]

    themes = json.loads(THEMES_JSON.read_text(encoding="utf-8"))["themes"]
    row = film_row(mini_con, TOY_STORY_4)
    row["credits"] = {"actor": refs("Actor", 12), "writer": refs("Writer", 5), "producer": refs("Producer", 5),
                      "director": refs("Director", 2), "composer": refs("Composer", 2)}
    row["companies"] = refs("Company", 8)
    row["genres"] = [f"G{i}" for i in range(7)]
    row["countries"] = [f"C{i}" for i in range(7)]
    # the keyword of seven different themes, listed in reverse file order and in capitals: the form lists them in file order
    row["keywords"] = [t["keywords"][0]["name"].upper() for t in reversed(themes[:7])]
    form = film_to_form(row)
    assert [a["name"] for a in form["actors"]] == [f"Actor {i}" for i in range(10)]
    assert [w["name"] for w in form["writers"]] == ["Writer 0", "Writer 1", "Writer 2"]
    assert [p["name"] for p in form["producers"]] == ["Producer 0", "Producer 1", "Producer 2"]
    assert [c["name"] for c in form["companies"]] == [f"Company {i}" for i in range(5)]
    assert form["genres"] == [f"G{i}" for i in range(5)] and form["countries"] == [f"C{i}" for i in range(5)]
    assert form["director"]["name"] == "Director 0" and form["composer"]["name"] == "Composer 0"  # the first one
    assert form["themes"] == [t["name"] for t in themes[:5]]


def test_film_to_form_none_stays_none_and_never_wide(mini_con, mini_ctx):
    form = film_to_form(film_row(mini_con, RUPTURE))  # no stored release type, no budget, no distributor, no collection
    assert form["release_type"] is None
    assert form["budget"] is None
    assert form["distributor"] is None
    assert form["franchise"] is None


def test_film_row_carries_the_collection_name(mini_con):
    row = film_row(mini_con, TOY_STORY_4)
    assert row["collection_id"] == 10194
    assert row["collection_name"] == "Toy Story Collection"


def test_film_row_unknown_film_raises(mini_con):
    with pytest.raises(KeyError):
        film_row(mini_con, 999_999_999)


def test_film_row_never_reads_post_release_columns(mini_con):
    row = film_row(mini_con, TENET)
    for key in POST_RELEASE:
        assert key not in row


def test_suspect_budget_still_reaches_the_form_and_prompt(mini_con, mini_ctx):
    row = film_row(mini_con, TENET)
    row["money_suspect"] = 1  # the flag comes from the film's own revenue; the budget is a pre-release fact
    form = film_to_form(row)
    assert form["budget"] == pytest.approx(255172171.9710523)  # Tenet's budget_2025 in the database
    assert "Budget: $255M" in render(resolve(form, form_as_of(form), mini_ctx))


def test_film_to_form_ignores_post_release_fields(mini_con, mini_ctx):
    # Rupture has no release type, budget, distributor or collection: the places where an "or <post-release field>" fallback would flow
    for movie_id in (TENET, RUPTURE, TOY_STORY_4):
        clean = film_row(mini_con, movie_id)
        poisoned = dict(clean, vote_average=9.87, vote_count=5551234, popularity=123456.7, revenue_2025=777_333_111.0,
                        revenue_source="tmdb", revenue_incomplete=1, money_suspect=1, split="train")
        assert film_to_form(poisoned) == film_to_form(clean), movie_id


def test_no_post_release_values_in_prompt(mini_con, mini_ctx):
    for movie_id in (TOY_STORY_4, TENET, RUPTURE):
        row = film_row(mini_con, movie_id)
        row.update(vote_average=9.87, vote_count=5551234, popularity=123456.7, revenue_2025=777_333_111.0)
        form = film_to_form(row)
        r = resolve(form, form_as_of(form), mini_ctx)
        for text in (render(r), r.to_json()):
            for sentinel in ("9.87", "5551234", "123456.7", "777333111", "777M", "123K"):
                assert sentinel not in text, (movie_id, sentinel)


# ---- form_as_of ----


def test_form_as_of_uses_the_first_of_the_month_without_a_day():
    assert form_as_of({"release_year": 2024, "release_month": 3, "release_day": None}) == date(2024, 3, 1)
    assert form_as_of({"release_year": 2024, "release_month": 3}) == date(2024, 3, 1)
    assert form_as_of({"release_year": 2024, "release_month": 3, "release_day": 17}) == date(2024, 3, 17)


# ---- resolve ----


def test_person_record_excludes_the_film_itself(mini_con, mini_ctx):
    r = resolved(mini_con, mini_ctx, TENET)
    assert r.director["name"] == "Christopher Nolan"
    assert r.director["films"] == 7  # Memento to Dunkirk; Tenet itself and Oppenheimer do not count
    assert (r.director["wins"], r.director["nominations"]) == (0, 5)  # awards dated before 2020-08-22 (hand-read from the table)


def test_director_awards_count_only_events_dated_before_the_release(mini_con, mini_ctx):
    # an independent count in SQL, strictly before the release date
    for (movie_id,) in mini_con.execute("SELECT tmdb_id FROM movies").fetchall():
        row = mini_con.execute(
            "SELECT m.release_date, c.person_id FROM movies m JOIN credits c ON c.movie_id = m.tmdb_id AND c.role = 'director' "
            "WHERE m.tmdb_id = ? ORDER BY c.ord LIMIT 1", (movie_id,)).fetchone()
        if row is None:
            continue
        release, director_id = row
        want = {kind: mini_con.execute(
            'SELECT COUNT(*) FROM awards WHERE person_id = ? AND kind = ? AND "when" IS NOT NULL AND "when" < ?',
            (director_id, kind, release)).fetchone()[0] for kind in ("won", "nominated")}
        r = resolved(mini_con, mini_ctx, movie_id)
        assert (r.director["wins"], r.director["nominations"]) == (want["won"], want["nominated"]), movie_id


def test_award_dated_on_the_release_day_or_later_does_not_count(mini_db, mini_con, tmp_path):
    # the saved set holds no award dated on a release day, so add three to a copy: Nolan (person 525) on Tenet's release day
    # (2020-08-22) and the day after, and one with no date
    before = resolved(mini_con, Context.from_db(mini_con, CFG), TENET).director
    copy_path = tmp_path / "awards.db"
    shutil.copy(mini_db, copy_path)
    con = sqlite3.connect(copy_path)
    con.executemany('INSERT INTO awards (person_id, award_qid, family, kind, "when") VALUES (525, ?, ?, ?, ?)',
                    [("Q1", "x", "won", "2020-08-22"), ("Q2", "x", "nominated", "2020-08-23"), ("Q3", "x", "won", None)])
    con.commit()
    after = resolved(con, Context.from_db(con, CFG), TENET).director
    con.close()
    assert (after["wins"], after["nominations"]) == (before["wins"], before["nominations"]) == (0, 5)


def test_resolve_record_values_and_shape(mini_con, mini_ctx):
    r = resolved(mini_con, mini_ctx, TENET)
    person_keys = {"name", "films", "revenue_usd", "score", "wins", "nominations"}
    for person in [r.director, *r.writers, *r.cast, *r.producers, r.composer]:
        assert set(person) == person_keys
    for company in [*r.companies, r.distributor]:
        assert set(company) == person_keys - {"wins", "nominations"}
    assert [c["name"] for c in r.companies] == ["Warner Bros. Pictures", "Syncopy"]
    assert r.distributor["name"] == "Warner Bros."
    assert r.distributor["films"] == 6  # The Dark Knight, Tarzan, Suicide Squad, Dunkirk, Joker, Birds of Prey
    assert r.director_age == 50  # born 1970-07-30, Tenet on 2020-08-22
    assert r.lead_age == 36  # John David Washington, born 1984-07-28
    assert r.competition == 0
    assert (r.release_year, r.release_month, r.release_day) == (2020, 8, 22)


def test_revenue_usd_is_exp_of_the_log(mini_con, mini_ctx):
    form = film_to_form(film_row(mini_con, TENET))
    as_of = form_as_of(form)
    r = resolve(form, as_of, mini_ctx)
    assert r.director["revenue_usd"] == pytest.approx(math.exp(mini_ctx.history.person(525, "director", as_of).revenue_log))


def test_first_franchise_entry_shows_no_franchise_and_no_installment(mini_con, mini_ctx):
    r = resolved(mini_con, mini_ctx, TOY_STORY)  # in collection 10194, with no earlier entry
    assert r.franchise is None and r.installment is None
    text = render(r)
    assert "nstallment" not in text and "Franchise: not given" in text


def test_later_entries_print_the_franchise_and_installment(mini_con, mini_ctx):
    r = resolved(mini_con, mini_ctx, TOY_STORY_2)
    assert r.installment == 2 and r.franchise == "Toy Story Collection"
    assert "Franchise: Toy Story Collection (installment 2)" in render(r)
    assert resolved(mini_con, mini_ctx, TOY_STORY_4).installment == 4  # after 1995, 1999 and 2010


def test_same_day_entry_is_not_earlier(mini_con, mini_ctx):
    form = film_to_form(film_row(mini_con, TOY_STORY_2))
    form["release_year"], form["release_month"], form["release_day"] = 1995, 11, 22  # Toy Story's own release day
    r = resolve(form, form_as_of(form), mini_ctx)
    assert r.installment is None and r.franchise is None


def test_user_form_with_a_known_collection_and_no_earlier_entry_reads_as_standalone(mini_ctx):
    # the same first-film rule for a user form: Toy Story Collection (id 10194) has no entry before 1990
    form, errors = validate({
        "genres": ["Family"], "release_year": 1990, "release_month": 6, "runtime": 100, "language": "en", "certification": "G",
        "director": "A Director", "actors": list("abcde"), "themes": ["based on a book"],
        "franchise": {"id": 10194, "name": "Toy Story Collection"},
    })
    assert errors == []
    r = resolve(form, form_as_of(form), mini_ctx)
    assert r.franchise is None and r.installment is None
    text = render(r)
    assert "Franchise: not given" in text and "Toy Story" not in text and "nstallment" not in text


def test_competition_counts_big_films_within_14_days_and_not_the_film_itself(mini_con, mini_ctx):
    # an independent count in SQL: budget_2025 above 100M, released within 14 days either way, not the film itself
    ids = [r[0] for r in mini_con.execute("SELECT tmdb_id FROM movies")]
    counts = {}
    for movie_id in ids:
        counts[movie_id] = mini_con.execute(
            "SELECT COUNT(*) FROM movies o, movies m WHERE m.tmdb_id = ? AND o.tmdb_id != m.tmdb_id AND o.budget_2025 > 100000000 "
            "AND ABS(julianday(o.release_date) - julianday(m.release_date)) <= 14", (movie_id,)).fetchone()[0]
        assert resolved(mini_con, mini_ctx, movie_id).competition == counts[movie_id], movie_id
    assert counts[346698] == 1 and counts[872585] == 1  # two big films opened on 2023-07-19
    assert counts[TENET] == 0  # a big film itself, alone in its window: it must not count itself


def test_resolved_film_is_frozen(mini_con, mini_ctx):
    r = resolved(mini_con, mini_ctx, TENET)
    with pytest.raises(AttributeError):
        r.competition = 5


def test_resolved_json_roundtrip(mini_con, mini_ctx):
    for movie_id in (TOY_STORY_4, TOY_STORY, RUPTURE, PARASITE, TENET):
        r = resolved(mini_con, mini_ctx, movie_id)
        assert ResolvedFilm.from_json(r.to_json()) == r
        assert render(ResolvedFilm.from_json(r.to_json())) == render(r)


def test_nan_becomes_none_and_json_is_strict(mini_ctx):
    # 1900 is before every film in the database: no typical value exists, so History hands back NaN
    form = {"release_year": 1900, "release_month": 1, "genres": ["Drama"], "director": {"id": None, "name": "Nobody"}}
    r = resolve(form, form_as_of(form), mini_ctx)
    assert r.director["films"] == 0
    assert r.director["revenue_usd"] is None and r.director["score"] is None
    text = r.to_json()
    json.loads(text, parse_constant=lambda c: pytest.fail(f"non-strict JSON constant {c}"))
    assert "NaN" not in text
    assert "Nobody [0 films · not given · not given" in render(r)


def test_missing_keys_read_as_absent(mini_ctx):
    r = resolve({"release_year": 2020, "release_month": 5}, date(2020, 5, 1), mini_ctx)
    assert r.director is None and r.cast == [] and r.writers == [] and r.composer is None and r.distributor is None
    assert r.franchise is None and r.installment is None and r.source_strength is None
    assert r.collaborations == 0 and r.director_age is None and r.lead_age is None
    render(r)


def test_user_form_path_with_a_new_person_and_a_new_franchise(mini_ctx):
    form, errors = validate({
        "genres": ["Drama"], "release_year": 2021, "release_month": 6, "runtime": 100, "language": "en", "certification": "PG-13",
        "director": {"name": "Christopher Nolan"},  # known, by name
        "actors": ["Brand New Actor", "Tom Hanks", "Cate New", "Dev Two", "Fifth Person"],
        "themes": ["based on a book"], "franchise": "Brand New Saga",
    })
    assert errors == []
    r = resolve(form, form_as_of(form), mini_ctx)
    assert r.director["name"] == "Christopher Nolan" and r.director["films"] == 8
    assert (r.cast[0]["name"], r.cast[0]["films"], r.cast[0]["wins"], r.cast[0]["nominations"]) == ("Brand New Actor", 0, 0, 0)
    assert r.cast[1]["name"] == "Tom Hanks" and r.cast[1]["films"] == 4  # Toy Story 1 to 4, all before 2021-06-01
    assert r.franchise is None and r.installment is None  # a new franchise reads as standalone
    assert r.source_strength is None
    assert form_as_of(form) == date(2021, 6, 1)
    text = render(r)
    assert "Brand New Actor [0 films" in text and "Franchise: not given" in text


def test_user_form_franchise_with_earlier_entries_prints(mini_ctx):
    form, errors = validate({
        "genres": ["Family"], "release_year": 2030, "release_month": 6, "runtime": 100, "language": "en", "certification": "G",
        "director": "A Director", "actors": list("abcde"), "themes": ["based on a book"],
        "franchise": {"id": 10194, "name": "Toy Story Collection"},
    })
    assert errors == []
    r = resolve(form, form_as_of(form), mini_ctx)
    assert r.installment == 5 and r.franchise == "Toy Story Collection"  # Toy Story 1 to 4 are earlier
    assert "Toy Story Collection" in render(r)


def test_film_with_three_actors_and_no_themes_resolves_and_renders(mini_con, mini_ctx):
    form = film_to_form(film_row(mini_con, DUNKIRK))
    form["actors"] = form["actors"][:3]
    form["themes"] = []
    r = resolve(form, form_as_of(form), mini_ctx)
    assert len(r.cast) == 3 and r.themes == []
    assert "Themes: not given" in render(r)


def test_collaborations_and_source_strength_flow_through(mini_con, mini_ctx):
    assert resolved(mini_con, mini_ctx, DUNKIRK).collaborations == 2  # Nolan with Dunkirk's first three actors before 2017-07-19
    assert resolved(mini_con, mini_ctx, MEMENTO).source_strength == 5  # source_sitelinks 5 in wd_films.jsonl
    assert resolved(mini_con, mini_ctx, TENET).source_strength is None


# ---- time travel ----


def test_resolution_ignores_later_films(mini_db, mini_con, mini_ctx, tmp_path):
    cut_path = tmp_path / "cut.db"
    shutil.copy(mini_db, cut_path)
    con = sqlite3.connect(cut_path)
    con.execute("DELETE FROM movies WHERE release_date >= '2016-01-01'")
    con.execute("DELETE FROM credits WHERE movie_id NOT IN (SELECT tmdb_id FROM movies)")
    con.execute("DELETE FROM movie_companies WHERE movie_id NOT IN (SELECT tmdb_id FROM movies)")
    con.commit()
    cut_ctx = Context.from_db(con, CFG)
    ids = [r[0] for r in con.execute("SELECT tmdb_id FROM movies")]
    assert 10 < len(ids) < 37
    for movie_id in ids:
        form = film_to_form(film_row(con, movie_id))
        assert film_to_form(film_row(mini_con, movie_id)) == form
        as_of = form_as_of(form)
        full, cut = resolve(form, as_of, mini_ctx).__dict__, resolve(form, as_of, cut_ctx).__dict__
        # competition looks forward by design: the release calendar is known in advance
        assert {**full, "competition": 0} == {**cut, "competition": 0}, movie_id
    con.close()


def test_later_entry_prints_the_name_stored_in_the_shard(mini_con, mini_ctx):
    shard_names = {}
    for shard in sorted((MINI / "tmdb").glob("tmdb_*.jsonl")):  # a plain read of the raw files, not through clean_movie
        for line in shard.read_text(encoding="utf-8").splitlines():
            raw = json.loads(line)
            if raw["id"] in (TOY_STORY, TOY_STORY_4):
                shard_names[raw["id"]] = raw["belongs_to_collection"]["name"]
    assert shard_names[TOY_STORY_4] == "Toy Story Collection"
    assert f"Franchise: {shard_names[TOY_STORY_4]} (installment 4)" in render(resolved(mini_con, mini_ctx, TOY_STORY_4))
    first = resolved(mini_con, mini_ctx, TOY_STORY)  # the first entry's name must not leak that a franchise followed
    assert shard_names[TOY_STORY] not in render(first) and shard_names[TOY_STORY] not in first.to_json()
    assert "Toy Story" not in first.to_json().split('"synopsis"')[0]


def test_real_collection_without_a_name_does_not_crash(mini_con, mini_ctx):
    row = film_row(mini_con, TOY_STORY_4)
    row["collection_name"] = None
    r = resolve(film_to_form(row), date(2019, 6, 19), mini_ctx)
    assert r.installment == 4 and r.franchise == "a film series"
    assert "Franchise: a film series (installment 4)" in render(r)


def test_id_only_person_prints_unnamed_with_the_real_record(mini_ctx):
    form = {"release_year": 2020, "release_month": 8, "release_day": 22, "director": {"id": 525, "name": None},
            "actors": [{"id": 1117313, "name": None}], "companies": [{"id": 174, "name": None}]}
    r = resolve(form, date(2020, 8, 22), mini_ctx)
    assert r.director["name"] is None and r.director["films"] == 7
    lines = render(r).splitlines()
    director = next(x for x in lines if x.startswith("Director:"))
    assert director.startswith("Director: Unnamed [7 films · ") and "0 wins, 5 noms]" in director
    assert next(x for x in lines if x.startswith("Cast:")).startswith("Cast: Unnamed [")
    assert next(x for x in lines if x.startswith("Companies:")).startswith("Companies: Unnamed [")
    assert "Unknown" not in render(r)


def test_resolve_gives_a_blank_franchise_name_the_generic_label(mini_ctx):
    form = {"release_year": 2030, "release_month": 6, "franchise": {"id": 10194, "name": "  "}}
    r = resolve(form, form_as_of(form), mini_ctx)
    assert r.franchise == "a film series" and r.installment == 5
    assert "Franchise: a film series (installment 5)" in render(r)
