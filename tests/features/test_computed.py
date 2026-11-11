import json
import shutil
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import pytest

from llm_movies.config import DataConfig, load_config
from llm_movies.data.build_db import _SCHEMA, build_db
from llm_movies.features.computed import (
    Context,
    age_at,
    collaborations,
    competition,
    installment,
    prior_awards,
    source_strength,
)

MINI = Path(__file__).parent.parent / "fixtures" / "mini"
PULLED_AT = date(2026, 11, 4)
CFG = DataConfig()
BIG = 100_000_000


def d(text):
    return date.fromisoformat(text)


# ---- hand-built databases, through the real tables ----


def movie(movie_id, released, budget=None, suspect=0, collection=None, sitelinks=None):
    """One movie row; `budget` is stored as given (2025 dollars)."""
    return {
        "tmdb_id": movie_id,
        "release_date": released,
        "budget_2025": budget,
        "money_suspect": suspect,
        "collection_id": collection,
        "source_sitelinks": sitelinks,
    }


def make_db(movies, credits=(), people=(), awards=()):
    """credits: (movie_id, person_id, role); people: (id, birth or None); awards: (person_id, award_qid, kind, when or None)."""
    con = sqlite3.connect(":memory:")
    for statement in _SCHEMA:
        con.execute(statement)
    for m in movies:
        columns = ", ".join(m)
        con.execute(f"INSERT INTO movies ({columns}) VALUES ({', '.join('?' * len(m))})", list(m.values()))
    con.executemany("INSERT INTO credits (movie_id, person_id, role, ord) VALUES (?, ?, ?, 0)", credits)
    con.executemany("INSERT INTO people (id, name, birth) VALUES (?, 'x', ?)", people)
    con.executemany("INSERT INTO awards (person_id, award_qid, family, kind, \"when\") VALUES (?, ?, 'oscar', ?, ?)", awards)
    con.commit()
    return con


def context(*args, **kwargs):
    return Context.from_db(make_db(*args, **kwargs), CFG)


# ---- competition ----


def test_competition_counts_both_sides_and_excludes_self():
    # Release R = 2020-06-15. Big budgets (> 100M) inside the window of +-14 days count, on either side.
    movies = [
        movie(1, "2020-06-01", 200_000_000),  # R - 14: counts (inclusive)
        movie(2, "2020-05-31", 200_000_000),  # R - 15: outside
        movie(3, "2020-06-29", 200_000_000),  # R + 14: counts (a later film: the calendar is known in advance)
        movie(4, "2020-06-30", 200_000_000),  # R + 15: outside
        movie(5, "2020-06-15", 300_000_000),  # R itself: counts unless it is the film asked about
        movie(6, "2020-06-15", BIG),  # exactly 100M: not strictly more
        movie(7, "2020-06-15", BIG + 1),  # counts
        # money_suspect comes from the film's revenue, and a rival can open up to 14 days after this film, so filtering on
        # it would leak the rival's outcome: a suspect-flagged big budget counts like any other.
        movie(8, "2020-06-15", 500_000_000, suspect=1),
        movie(9, "2020-06-15", None),  # unknown budget
        movie(10, "2020-06-15", 50_000_000),  # small
    ]
    ctx = context(movies)
    r = d("2020-06-15")
    assert competition(ctx, r, None) == 5  # films 1, 3, 5, 7, 8
    assert competition(ctx, r, 5) == 4  # the film itself leaves
    assert competition(ctx, r, 1) == 4  # an in-window big film can be excluded too
    assert competition(ctx, r, 6) == 5  # excluding a film that was never counted changes nothing
    assert competition(ctx, r, 2) == 5  # nor one outside the window
    assert competition(ctx, r, 999) == 5  # nor an unknown id
    assert competition(ctx, d("2030-01-01"), None) == 0


def test_competition_window_edges_each_side():
    ctx = context([movie(1, "2020-06-15", 200_000_000)])
    for offset, expected in ((-15, 0), (-14, 1), (0, 1), (14, 1), (15, 0)):
        assert competition(ctx, d("2020-06-15") + timedelta(days=offset), None) == expected, offset


def test_competition_on_an_empty_database():
    assert competition(context([]), d("2020-01-01"), None) == 0


# ---- installment ----


def test_first_entry_is_installment_1():
    ctx = context([movie(1, "2000-01-01", collection=7), movie(2, "2003-01-01", collection=7)])
    assert installment(ctx, 7, d("2000-01-01")) == 1


def test_installment_counts_earlier_entries_only():
    movies = [
        movie(1, "2000-01-01", collection=7),
        movie(2, "2003-01-01", collection=7),
        movie(3, "2006-01-01", collection=7),
        movie(4, "2006-01-01", collection=7),  # same day as film 3
        movie(5, "2001-01-01", collection=8),  # another collection
    ]
    ctx = context(movies)
    assert installment(ctx, 7, d("2003-01-01")) == 2
    assert installment(ctx, 7, d("2006-01-01")) == 3  # entries 1 and 2 are earlier; the same-day entry is not
    assert installment(ctx, 7, d("2006-01-02")) == 5  # all four are earlier
    assert installment(ctx, 8, d("2010-01-01")) == 2
    assert installment(ctx, 7, d("1990-01-01")) == 1


def test_installment_none_without_a_collection_and_one_for_an_unknown_collection():
    ctx = context([movie(1, "2000-01-01", collection=7)])
    assert installment(ctx, None, d("2010-01-01")) is None
    assert installment(ctx, 999, d("2010-01-01")) == 1  # no earlier entries: the first one


# ---- collaborations ----

D, A, B, C, E = 1, 10, 11, 12, 13  # director, three first actors, a fourth-billed actor


def collab_ctx():
    movies = [
        movie(1, "2000-01-01"),  # D directs; A and B act: one film, counted once
        movie(2, "2001-01-01"),  # D directs; E (fourth) acts
        movie(3, "2002-01-01"),  # D directs; C acts
        movie(4, "2003-01-01"),  # someone else directs; A acts
        movie(5, "2004-01-01"),  # D acts (not directs), with A
        movie(6, "2005-01-01"),  # D directs; A acts
    ]
    credits = [
        (1, D, "director"), (1, A, "actor"), (1, B, "actor"),
        (2, D, "director"), (2, E, "actor"),
        (3, D, "director"), (3, C, "actor"),
        (4, 2, "director"), (4, A, "actor"),
        (5, D, "actor"), (5, A, "actor"),
        (6, D, "director"), (6, A, "actor"),
    ]  # fmt: skip
    return context(movies, credits)


def test_collaborations_only_before():
    ctx = collab_ctx()
    actors = [A, B, C, E]
    assert collaborations(ctx, D, actors, d("2010-01-01")) == 3  # films 1, 3, 6 (2 is the fourth actor, 4 and 5 lack D as director)
    assert collaborations(ctx, D, actors, d("2005-01-01")) == 2  # film 6 is on the date: not counted
    assert collaborations(ctx, D, actors, d("2005-01-02")) == 3
    assert collaborations(ctx, D, actors, d("2000-01-01")) == 0  # as_of on the shared film's own date
    assert collaborations(ctx, D, actors, d("2000-01-02")) == 1


def test_collaborations_two_actors_on_one_film_count_once():
    ctx = collab_ctx()
    assert collaborations(ctx, D, [A, B], d("2000-06-01")) == 1  # film 1 has both A and B


def test_collaborations_fourth_actor_does_not_count():
    ctx = collab_ctx()
    assert collaborations(ctx, D, [99, 98, 97, E], d("2010-01-01")) == 0
    assert collaborations(ctx, D, [E], d("2010-01-01")) == 1  # E alone, in the first slot, does count
    assert collaborations(ctx, D, [A, B, C, E], d("2001-06-01")) == 1  # film 2 (E) is not counted, film 1 is


def test_collaborations_skip_the_director_among_the_actors():
    # F directs and acts in films 1 and 2; film 2 also has G; film 3 is directed by F with G only; film 4 is by F with H (4th).
    F, G, H = 20, 21, 22
    movies = [movie(i, f"200{i}-01-01") for i in (1, 2, 3, 4)]
    credits = [
        (1, F, "director"), (1, F, "actor"),
        (2, F, "director"), (2, F, "actor"), (2, G, "actor"),
        (3, F, "director"), (3, G, "actor"),
        (4, F, "director"), (4, H, "actor"),
    ]  # fmt: skip
    ctx = context(movies, credits)
    as_of = d("2010-01-01")
    assert collaborations(ctx, F, [F], as_of) == 0  # his own filmography is not a collaboration
    assert collaborations(ctx, F, [F, G], as_of) == 2  # films 2 and 3 with G; film 1 has only himself
    assert collaborations(ctx, F, [F, 98, 97, H], as_of) == 0  # himself still takes one of the first three slots; H is fourth
    assert collaborations(ctx, F, [G], as_of) == 2


def test_collaborations_ignores_none_ids_and_handles_unknowns():
    ctx = collab_ctx()
    as_of = d("2010-01-01")
    assert collaborations(ctx, None, [A, B, C], as_of) == 0
    assert collaborations(ctx, D, [None, None, None], as_of) == 0
    assert collaborations(ctx, D, [], as_of) == 0
    assert collaborations(ctx, D, [None, A], as_of) == 2  # A with D: films 1 and 6
    assert collaborations(ctx, D, [None, None, None, A], as_of) == 0  # A is fourth: the None slots still use the three
    assert collaborations(ctx, 999, [A], as_of) == 0
    assert collaborations(ctx, D, [999], as_of) == 0
    assert collaborations(ctx, D, [A, A], as_of) == 2  # the same actor twice


# ---- age_at ----


def test_age_none_without_birth():
    ctx = context([], people=[(1, None), (2, "1990-05-15")])
    assert age_at(ctx, 1, d("2020-01-01")) is None  # a person row with no birth
    assert age_at(ctx, None, d("2020-01-01")) is None
    assert age_at(ctx, 999, d("2020-01-01")) is None  # an unknown person


def test_age_goes_up_on_the_birthday():
    ctx = context([], people=[(1, "1990-05-15")])
    assert age_at(ctx, 1, d("2020-05-14")) == 29
    assert age_at(ctx, 1, d("2020-05-15")) == 30
    assert age_at(ctx, 1, d("2020-05-16")) == 30
    assert age_at(ctx, 1, d("1990-05-15")) == 0  # born that day
    assert age_at(ctx, 1, d("1991-05-14")) == 0
    assert age_at(ctx, 1, d("1991-05-15")) == 1


def test_age_before_birth_is_none():
    ctx = context([], people=[(1, "1990-05-15")])
    assert age_at(ctx, 1, d("1990-05-14")) is None
    assert age_at(ctx, 1, d("1980-01-01")) is None


def test_age_of_a_leap_day_birth():
    # Born 2000-02-29: in a common year the birthday counts from 1 March.
    ctx = context([], people=[(1, "2000-02-29")])
    assert age_at(ctx, 1, d("2021-02-28")) == 20
    assert age_at(ctx, 1, d("2021-03-01")) == 21
    assert age_at(ctx, 1, d("2024-02-28")) == 23
    assert age_at(ctx, 1, d("2024-02-29")) == 24
    assert age_at(ctx, 1, d("2000-02-29")) == 0


# ---- prior_awards ----


def awards_ctx():
    awards = [
        (1, "Q1", "won", "2010-01-01"),
        (1, "Q2", "nominated", "2010-01-01"),  # another category the same night: counts too
        (1, "Q3", "won", "2010-03-09"),
        (1, "Q4", "nominated", "2010-03-10"),
        (1, "Q5", "won", "2010-03-11"),
        (1, "Q6", "nominated", None),  # undated
        (1, "Q7", "won", None),  # undated
        (2, "Q1", "won", "2000-05-05"),  # three categories the same night
        (2, "Q2", "won", "2000-05-05"),
        (2, "Q3", "won", "2000-05-05"),
    ]
    return context([], awards=awards)


def test_award_on_release_day_not_counted():
    ctx = awards_ctx()
    assert prior_awards(ctx, [1], d("2010-03-10")) == {"wins": 2, "nominations": 1}  # the 03-10 nomination is on the date
    assert prior_awards(ctx, [1], d("2010-03-11")) == {"wins": 2, "nominations": 2}  # the day after, it counts
    assert prior_awards(ctx, [1], d("2010-03-09")) == {"wins": 1, "nominations": 1}  # the 03-09 win is on the date
    assert prior_awards(ctx, [1], d("2010-03-12")) == {"wins": 3, "nominations": 2}


def test_award_the_day_before_counts_and_before_the_first_does_not():
    ctx = awards_ctx()
    assert prior_awards(ctx, [1], d("2010-01-02")) == {"wins": 1, "nominations": 1}  # both of 2010-01-01
    assert prior_awards(ctx, [1], d("2010-01-01")) == {"wins": 0, "nominations": 0}


def test_undated_award_not_counted():
    ctx = awards_ctx()
    assert prior_awards(ctx, [1], d("2100-01-01")) == {"wins": 3, "nominations": 2}  # Q6 and Q7 never count


def test_awards_sum_over_people_and_count_a_repeated_id_once():
    ctx = awards_ctx()
    as_of = d("2010-03-10")
    assert prior_awards(ctx, [2], as_of) == {"wins": 3, "nominations": 0}  # three categories one night: all three
    assert prior_awards(ctx, [1, 2], as_of) == {"wins": 5, "nominations": 1}
    assert prior_awards(ctx, [1, 1], as_of) == prior_awards(ctx, [1], as_of)
    assert prior_awards(ctx, [1, 2, 1, 2], as_of) == {"wins": 5, "nominations": 1}


def test_awards_ignore_none_unknown_and_empty():
    ctx = awards_ctx()
    as_of = d("2010-03-10")
    zero = {"wins": 0, "nominations": 0}
    assert prior_awards(ctx, [None, 1], as_of) == {"wins": 2, "nominations": 1}
    assert prior_awards(ctx, [None], as_of) == zero
    assert prior_awards(ctx, [], as_of) == zero
    assert prior_awards(ctx, [999], as_of) == zero


def test_awards_count_wins_and_nominations_separately():
    ctx = context([], awards=[(1, "Q1", "won", "2000-01-01"), (1, "Q2", "nominated", "2000-01-01"), (1, "Q3", "nominated", "2000-01-02")])
    assert prior_awards(ctx, [1], d("2001-01-01")) == {"wins": 1, "nominations": 2}


# ---- source_strength ----


def test_source_strength_passes_the_count_through():
    ctx = context([])
    assert source_strength(ctx, 12) == 12
    assert source_strength(ctx, 0) == 0
    assert source_strength(ctx, None) is None


def test_sitelinks_of_hand_built():
    ctx = context([movie(1, "2000-01-01", sitelinks=42), movie(2, "2000-01-01")])
    assert ctx.sitelinks_of(1) == 42
    assert ctx.sitelinks_of(2) is None  # a film with no source count
    assert ctx.sitelinks_of(999) is None  # not a film
    assert ctx.sitelinks_of(None) is None


# ---- the Context ----


def test_context_works_after_the_connection_closes():
    con = make_db(
        [movie(1, "2000-01-01", 200_000_000, collection=7, sitelinks=3), movie(2, "2001-01-01", collection=7)],
        credits=[(1, D, "director"), (1, A, "actor")],
        people=[(D, "1970-01-01")],
        awards=[(D, "Q1", "won", "1999-01-01")],
    )
    ctx = Context.from_db(con, CFG)
    con.close()
    assert competition(ctx, d("2000-01-05"), None) == 1
    assert installment(ctx, 7, d("2001-01-01")) == 2
    assert collaborations(ctx, D, [A], d("2005-01-01")) == 1
    assert age_at(ctx, D, d("2000-01-01")) == 30
    assert prior_awards(ctx, [D], d("2005-01-01")) == {"wins": 1, "nominations": 0}
    assert ctx.sitelinks_of(1) == 3
    assert ctx.history.person(D, "director", d("2005-01-01")).films == 1


def test_context_default_config_is_loaded():
    ctx = Context.from_db(make_db([movie(1, "2000-01-01")]))
    assert ctx.history.cfg == load_config(Path("config/data.yaml"))
    assert ctx.history.cfg.base_model_cutoff == date(2025, 4, 29)  # the yaml, not the code default (None)


def test_empty_database_never_raises():
    ctx = context([])
    assert competition(ctx, d("2020-01-01"), 1) == 0
    assert installment(ctx, 1, d("2020-01-01")) == 1
    assert collaborations(ctx, 1, [2], d("2020-01-01")) == 0
    assert age_at(ctx, 1, d("2020-01-01")) is None
    assert prior_awards(ctx, [1], d("2020-01-01")) == {"wins": 0, "nominations": 0}
    assert ctx.sitelinks_of(1) is None


# ---- the mini set, real data ----


@pytest.fixture(scope="module")
def mini_db(tmp_path_factory):
    out = tmp_path_factory.mktemp("mini") / "mini.db"
    build_db(MINI / "tmdb", MINI / "wd_films", MINI / "wd_people", out, PULLED_AT)
    return out


@pytest.fixture(scope="module")
def mini(mini_db):
    con = sqlite3.connect(mini_db)
    ctx = Context.from_db(con, CFG)
    con.close()
    return ctx


def test_sitelinks_of_real_films_match_the_wikidata_shard(mini, mini_db):
    expected = {}
    with open(MINI / "wd_films" / "wd_films.jsonl", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            facts = record["facts"]
            expected[record["tmdb_id"]] = facts["source_sitelinks"] if facts else None
    con = sqlite3.connect(mini_db)
    ids = [r[0] for r in con.execute("SELECT tmdb_id FROM movies")]
    con.close()
    assert len(ids) == 37
    values = [mini.sitelinks_of(i) for i in ids]
    assert values == [expected.get(i) for i in ids]
    assert any(v is not None for v in values) and any(v is None for v in values)
    assert mini.sitelinks_of(77) == 5  # Memento: source_sitelinks 5 in wd_films.jsonl


def test_dunkirk_collaborations_of_nolan_and_its_first_three_actors(mini):
    # Dunkirk (374720, 2017-07-19): director Christopher Nolan (525); actors by billing: Fionn Whitehead (1687041),
    # Tom Hardy (2524), Mark Rylance (40900), then Kenneth Branagh (11181). Read from the TMDB shards: Nolan's earlier
    # films with any of the first three are Inception (2010-07-15, Hardy) and The Dark Knight Rises (2012-07-17, Hardy);
    # Whitehead and Rylance are in no earlier Nolan film.
    actors = [1687041, 2524, 40900, 11181]
    assert collaborations(mini, 525, actors, d("2017-07-19")) == 2
    assert collaborations(mini, 525, actors, d("2012-07-17")) == 1  # The Dark Knight Rises is on the date
    assert collaborations(mini, 525, actors, d("2012-07-18")) == 2
    assert collaborations(mini, 525, actors, d("2010-07-15")) == 0
    assert collaborations(mini, 525, [1687041, 40900, 11181], d("2017-07-19")) == 0


def test_nolan_prior_awards_and_age(mini):
    # Nolan's events in wd_people.jsonl that fall in an award family of data/awards.json (all three are Oscars:
    # Q41417 Original Screenplay, Q102427 Best Picture, Q103360 Best Director): nominated 2002-02-12 (Q41417), nominated
    # 2011-01-25 (Q102427 and Q41417), nominated 2018-01-23 (Q102427 and Q103360); no Oscar win before 2024. His other
    # events are undated, in no family (Q1056240 ...), or in a family but dated 2024 (Q107258, Q787131, Q586356), so none
    # falls before the dates asked here. On 2018-01-23 itself the two 2018 nominations do not count.
    assert prior_awards(mini, [525], d("2018-01-23")) == {"wins": 0, "nominations": 3}
    assert prior_awards(mini, [525], d("2018-01-24")) == {"wins": 0, "nominations": 5}
    assert prior_awards(mini, [525], d("2011-01-25")) == {"wins": 0, "nominations": 1}
    assert prior_awards(mini, [525], d("2002-02-12")) == {"wins": 0, "nominations": 0}
    # Born 1970-07-30 (wd_people.jsonl).
    assert age_at(mini, 525, d("2020-08-22")) == 50
    assert age_at(mini, 525, d("2020-07-29")) == 49
    assert age_at(mini, 525, d("2020-07-30")) == 50


def test_real_installments_and_competition(mini):
    # Batman Begins 2005-06-10, The Dark Knight 2008-07-16, The Dark Knight Rises 2012-07-17 share collection 263.
    assert installment(mini, 263, d("2005-06-10")) == 1
    assert installment(mini, 263, d("2012-07-17")) == 3
    assert installment(mini, 10194, d("2019-06-19")) == 4  # Toy Story 4: after 1995, 1999 and 2010
    # Barbie (346698) and Oppenheimer (872585) both open 2023-07-19 with budgets over 100M (2025 dollars).
    assert competition(mini, d("2023-07-19"), None) == 2
    assert competition(mini, d("2023-07-19"), 346698) == 1
    # Toy Story 3 (2010-06-16) is 15 days before 2010-07-01, Inception (2010-07-15) exactly 14 days after it.
    assert competition(mini, d("2010-07-01"), None) == 1


def test_time_travel_identical_inputs(tmp_path, mini_db):
    con = sqlite3.connect(mini_db)
    films = con.execute("SELECT tmdb_id, release_date, collection_id FROM movies").fetchall()
    credits = con.execute("SELECT movie_id, person_id, role, ord FROM credits").fetchall()
    con.close()
    full = Context.from_db(sqlite3.connect(mini_db), CFG)
    compared = 0
    seen_awards = 0
    for cut in ("2005-01-01", "2012-01-01", "2016-01-01", "2020-01-01"):
        cut_db = tmp_path / f"cut_{cut}.db"
        shutil.copy(mini_db, cut_db)
        con = sqlite3.connect(cut_db)
        later = [f[0] for f in films if f[1] >= cut]
        assert later and len(later) < len(films)
        for table, column in (
            ("credits", "movie_id"),
            ("movie_companies", "movie_id"),
            ("movie_keywords", "movie_id"),
            ("movies", "tmdb_id"),
        ):
            con.executemany(f"DELETE FROM {table} WHERE {column} = ?", [(m,) for m in later])
        # Awards are dated facts too: drop those from the cut on, so a prior_awards that ignores the date would differ.
        con.execute('DELETE FROM awards WHERE "when" >= ?', (cut,))
        con.commit()
        cutc = Context.from_db(con, CFG)
        con.close()
        for movie_id, released, collection in (f for f in films if f[1] < cut):
            directors = [c[1] for c in credits if c[0] == movie_id and c[2] == "director"]
            cast = [c[1] for c in sorted((c for c in credits if c[0] == movie_id and c[2] == "actor"), key=lambda c: c[3])]
            people = directors + cast
            for as_of in (d(released), d(cut)):
                assert installment(full, collection, as_of) == installment(cutc, collection, as_of), (cut, movie_id)
                for director in directors or [None]:
                    assert collaborations(full, director, cast[:3], as_of) == collaborations(cutc, director, cast[:3], as_of)
                    compared += 1
                assert prior_awards(full, people, as_of) == prior_awards(cutc, people, as_of), (cut, movie_id)
                seen_awards += sum(prior_awards(full, people, as_of).values())
                for person in people:
                    assert age_at(full, person, as_of) == age_at(cutc, person, as_of)
                    compared += 1
        # The test can see a leak: after the cut the two databases differ.
        far = d("2030-01-01")
        if cut < "2020-01-01":  # Toy Story 4 (2019) and Dunkirk (2017) are lost by these cuts
            assert collaborations(full, 525, [2037], far) != collaborations(cutc, 525, [2037], far)
            assert installment(full, 10194, far) != installment(cutc, 10194, far)
        # Nolan's events dated 2011, 2018 and 2024 are dropped from the 2005, 2012 and 2016 cuts.
        assert prior_awards(full, [525], far) != prior_awards(cutc, [525], far)
        # competition is exempt by design: later films count, so the cut database sees fewer of them.
        # Tenet (2020-08-22) is the only big film within 14 days of 2020-08-15, and it is after the 2020 cut.
        if cut == "2020-01-01":
            assert competition(full, d("2020-08-15"), None) == 1
            assert competition(cutc, d("2020-08-15"), None) == 0
    assert compared > 500
    assert seen_awards > 0  # the award comparison saw real counts, not only zeros
