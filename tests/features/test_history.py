import math
import shutil
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import pytest

from llm_movies.config import DataConfig
from llm_movies.data.build_db import _SCHEMA, build_db
from llm_movies.features.history import History, PersonRecord

MINI = Path(__file__).parent.parent / "fixtures" / "mini"
PULLED_AT = date(2026, 11, 4)
NAN = float("nan")
CFG = DataConfig()  # min_votes 50, mature_days 182, shrink_films 4


# ---- hand-built databases, through the real tables ----


def film(movie_id, released, rev=None, bud=None, votes=None, score=None, suspect=0, incomplete=0, distributor=None):
    """One movie row. `rev` and `bud` are LOG values (natural log) so the hand arithmetic stays small; stored as exp()."""
    return {
        "tmdb_id": movie_id,
        "release_date": released,
        "revenue_2025": math.exp(rev) if rev is not None else None,
        "budget_2025": math.exp(bud) if bud is not None else None,
        "vote_count": votes,
        "vote_average": score,
        "money_suspect": suspect,
        "revenue_incomplete": incomplete,
        "distributor": distributor,
    }


def make_db(films, credits=(), companies=(), people=(), company_names=(), unique_credits=True):
    """credits: (movie_id, person_id, role); companies: (movie_id, company_id); people: (id, name); company_names: (id, name)."""
    con = sqlite3.connect(":memory:")
    for statement in _SCHEMA:
        if not unique_credits and statement.startswith("CREATE TABLE credits"):
            statement = statement.replace(", PRIMARY KEY (movie_id, person_id, role)", "")
        con.execute(statement)
    for f in films:
        columns = ", ".join(f)
        con.execute(f"INSERT INTO movies ({columns}) VALUES ({', '.join('?' * len(f))})", list(f.values()))
    con.executemany("INSERT INTO credits (movie_id, person_id, role, ord) VALUES (?, ?, ?, 0)", credits)
    con.executemany("INSERT INTO movie_companies (movie_id, company_id, ord) VALUES (?, ?, 0)", companies)
    con.executemany("INSERT INTO people (id, name) VALUES (?, ?)", people)
    con.executemany("INSERT INTO companies (id, name) VALUES (?, ?)", company_names)
    con.commit()
    return con


def history(*args, **kwargs):
    return History.from_db(make_db(*args, **kwargs), CFG)


def d(text):
    return date.fromisoformat(text)


def same(a, b):
    """Exact equality, NaN equal to NaN (repr shows the exact float)."""
    return repr(a) == repr(b)


# ---- the five named tests ----


def test_film_never_counts_for_itself():
    # Director 1 directs F1 (2015-01-01) and, on one day, F2 and F3 (2016-06-01). F2 and F3 have huge values.
    h = history(
        [
            film(1, "2015-01-01", rev=10, bud=8, votes=100, score=6),
            film(2, "2016-06-01", rev=30, bud=30, votes=100, score=9),
            film(3, "2016-06-01", rev=30, bud=30, votes=100, score=9),
        ],
        [(1, 1, "director"), (2, 1, "director"), (3, 1, "director")],
    )
    # On the first film's own date nothing is earlier: new, and no earlier film exists, so typical values are NaN.
    first = h.person(1, "director", d("2015-01-01"))
    assert first.films == 0 and first.is_new and first.years_active == 0
    assert math.isnan(first.revenue_log) and math.isnan(first.score) and math.isnan(first.budget_log)
    # On the day of F2 and F3 only F1 counts: not F2 or F3 themselves, and not each other.
    rec = h.person(1, "director", d("2016-06-01"))
    assert rec.films == 1 and not rec.is_new
    # F1 is 517 days old (>= 182): its values count. Only F1 is earlier, so typical = F1's own values:
    # revenue (1*10 + 4*10)/5 = 10, score (6 + 4*6)/5 = 6, budget (8 + 4*8)/5 = 8. F2/F3 (30, 9) would show.
    assert (rec.revenue_log, rec.score, rec.budget_log) == pytest.approx((10, 6, 8))
    assert h.typical(d("2016-06-01")) == pytest.approx((10, 6, 8))
    # The day after, both same-day films count (films = 3).
    assert h.person(1, "director", d("2016-06-02")).films == 3


def test_revenue_waits_182_days_budget_does_not():
    # H (other person, 2000-01-01): log revenue 5, score 5, log budget 3. F (director 1, 2020-01-01): rev 11, score 7, bud 9.
    # 2020-01-01 + 181 days = 2020-06-30; + 182 days = 2020-07-01.
    h = history(
        [film(1, "2000-01-01", rev=5, bud=3, votes=100, score=5), film(2, "2020-01-01", rev=11, bud=9, votes=100, score=7)],
        [(1, 9, "director"), (2, 1, "director")],
    )
    release = d("2020-01-01")
    day0 = h.person(1, "director", release)
    assert day0.films == 0 and day0.is_new  # not on its own release day, even for the budget
    # Day 1: budget counts. Typical budget {3, 9} -> 6; own budget n=1: (1*9 + 4*6)/5 = 33/5 = 6.6.
    # Revenue and score: n=0, so typical from H only: 5 and 5.
    early = h.person(1, "director", release + timedelta(days=1))
    assert early.films == 1
    assert early.budget_log == pytest.approx(6.6)
    assert (early.revenue_log, early.score) == pytest.approx((5, 5))
    # 181 days: still not mature. Films counts, revenue and score are still the typical ones.
    day181 = h.person(1, "director", release + timedelta(days=181))
    assert day181.films == 1 and day181.budget_log == pytest.approx(6.6)
    assert (day181.revenue_log, day181.score) == pytest.approx((5, 5))
    assert h.typical(release + timedelta(days=181))[:2] == pytest.approx((5, 5))
    # 182 days: F counts. Typical revenue {5, 11} -> 8, score {5, 7} -> 6.
    # revenue (1*11 + 4*8)/5 = 43/5 = 8.6; score (1*7 + 4*6)/5 = 31/5 = 6.2; budget unchanged 6.6.
    day182 = h.person(1, "director", release + timedelta(days=182))
    assert day182.films == 1
    assert (day182.revenue_log, day182.score, day182.budget_log) == pytest.approx((8.6, 6.2, 6.6))
    assert h.typical(release + timedelta(days=182)) == pytest.approx((8, 6, 6))


def test_shrinkage_formula():
    # Films (all mature by 2020-01-01); P = person 1 (director) has A, B, C; person 2 has D.
    #   A 2010: log rev 10, score 6 (100 votes), log bud 8
    #   B 2011: no revenue,  score 8 (100 votes), log bud 12      <- budget but no revenue
    #   C 2012: log rev 14, score 7 (60 votes),  log bud 10
    #   D 2009 (person 2): log rev 6, score 4 (100 votes), log bud 4
    # Typical: revenue {10, 14, 6} -> 10; score {6, 8, 7, 4} -> (6+7)/2 = 6.5; budget {8, 12, 10, 4} -> (8+10)/2 = 9.
    # P: revenue n=2 {10, 14} median 12 -> (2*12 + 4*10)/6 = 64/6 = 10.6667
    #    score n=3 {6, 8, 7} median 7 -> (3*7 + 4*6.5)/7 = 47/7 = 6.7143
    #    budget n=3 {8, 12, 10} median 10 -> (3*10 + 4*9)/7 = 66/7 = 9.4286   (n is 2, 3, 3: not films)
    # Person 2, one film D: revenue (1*6 + 4*10)/5 = 9.2, score (4 + 4*6.5)/5 = 6.0, budget (4 + 4*9)/5 = 8.0
    h = history(
        [
            film(1, "2010-01-01", rev=10, bud=8, votes=100, score=6),
            film(2, "2011-01-01", bud=12, votes=100, score=8),
            film(3, "2012-01-01", rev=14, bud=10, votes=60, score=7),
            film(4, "2009-01-01", rev=6, bud=4, votes=100, score=4),
        ],
        [(1, 1, "director"), (2, 1, "director"), (3, 1, "director"), (4, 2, "director")],
    )
    as_of = d("2020-01-01")
    assert h.typical(as_of) == pytest.approx((10, 6.5, 9))
    p = h.person(1, "director", as_of)
    assert p.films == 3 and not p.is_new
    assert p.revenue_log == pytest.approx(64 / 6) and p.score == pytest.approx(47 / 7) and p.budget_log == pytest.approx(66 / 7)
    q = h.person(2, "director", as_of)
    assert q.films == 1
    assert (q.revenue_log, q.score, q.budget_log) == pytest.approx((9.2, 6.0, 8.0))


def test_unknown_person_is_new_with_typical_values():
    h = history(
        [film(1, "2000-01-01", rev=5, bud=3, votes=100, score=5), film(2, "2001-01-01", rev=7, bud=5, votes=100, score=7)],
        [(1, 1, "director")],
    )
    as_of = d("2010-01-01")
    # Typical: revenue {5, 7} -> 6, score {5, 7} -> 6, budget {3, 5} -> 4.
    expected = PersonRecord(films=0, years_active=0, revenue_log=6.0, score=6.0, budget_log=4.0, is_new=True)
    for record in (h.person(999, "director", as_of), h.person(None, "director", as_of), h.person(1, "actor", as_of)):
        assert record == expected


def test_time_travel_identical_inputs(tmp_path):
    full = tmp_path / "full.db"
    build_db(MINI / "tmdb", MINI / "wd_films", MINI / "wd_people", full, PULLED_AT)
    con = sqlite3.connect(full)
    films = con.execute("SELECT tmdb_id, release_date, distributor FROM movies").fetchall()
    credits = con.execute("SELECT movie_id, person_id, role FROM credits").fetchall()
    links = con.execute("SELECT movie_id, company_id FROM movie_companies").fetchall()
    con.close()
    h_full = History.from_db(sqlite3.connect(full), CFG)
    compared = 0
    for cut in ("2005-01-01", "2012-01-01", "2016-01-01", "2020-01-01"):
        cut_db = tmp_path / f"cut_{cut}.db"
        shutil.copy(full, cut_db)
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
        con.commit()
        h_cut = History.from_db(con, CFG)
        date_of = {f[0]: d(f[1]) for f in films}
        for movie_id, released, distributor in (f for f in films if f[1] < cut):
            when = d(released)
            for as_of in (when, d(cut)):
                assert same(h_full.typical(as_of), h_cut.typical(as_of)), (cut, movie_id, as_of)
                for _, person, role in (c for c in credits if c[0] == movie_id):
                    assert same(h_full.person(person, role, as_of), h_cut.person(person, role, as_of)), (cut, movie_id, person, role)
                    compared += 1
                for _, company in (k for k in links if k[0] == movie_id):
                    assert same(h_full.company(company, as_of), h_cut.company(company, as_of)), (cut, movie_id, company)
                    compared += 1
                assert same(h_full.distributor(distributor, as_of), h_cut.distributor(distributor, as_of)), (cut, movie_id)
                compared += 1
        # The test can see a leak: after the cut the two databases differ for somebody, and for typical.
        far = d("2030-01-01")
        assert not same(h_full.typical(far), h_cut.typical(far))
        assert any(not same(h_full.person(p, r, far), h_cut.person(p, r, far)) for m, p, r in credits if date_of[m] >= d(cut))
    assert compared > 1000


# ---- values that must not count ----


def test_score_excluded_below_min_votes():
    # Person 1: X 49 votes (score 3), Y 50 votes (score 7), Z 0 votes (score 0.0), V 100 votes but no vote_average.
    # Person 2: H 100 votes, score 5. Only Y (>= min_votes 50) and H count. Typical {7, 5} -> 6.
    # Person 1 score: n=1, (1*7 + 4*6)/5 = 6.2. Counting X would give n=2 median 5; counting Z would drag it down.
    # X and Z are chosen so that counting them (alone or together) moves typical off 6: {3,5,7} -> 5, {0,5,7} -> 5,
    # {0,3,5,7} -> 4 (with 9 instead of 3 the last was {0,5,7,9} -> 6, which hid a 0-vote film in typical).
    h = history(
        [
            film(1, "2010-01-01", votes=49, score=3),
            film(2, "2010-02-01", votes=50, score=7),
            film(3, "2010-03-01", votes=0, score=0.0),
            film(4, "2010-04-01", votes=100, score=None),
            film(5, "2009-01-01", votes=100, score=5),
        ],
        [(1, 1, "director"), (2, 1, "director"), (3, 1, "director"), (4, 1, "director"), (5, 2, "director")],
    )
    as_of = d("2020-01-01")
    assert h.typical(as_of)[1] == pytest.approx(6.0)
    rec = h.person(1, "director", as_of)
    assert rec.films == 4 and rec.score == pytest.approx(6.2)


def test_suspect_and_incomplete_excluded():
    # Person 1: S money_suspect (rev 20, bud 20, score 9 from 100 votes), I revenue_incomplete (rev 20, bud 20),
    # O ok (rev 10, bud 8), N with no revenue and no budget. Person 2: H rev 6, bud 4, score 5.
    # Revenue usable: O 10, H 6 -> typical 8; person 1 n=1: (10 + 4*8)/5 = 8.4.
    # Budget usable: S excluded (suspect); I 20, O 8, H 4 -> typical 8; person 1 n=2 {20, 8} median 14: (2*14 + 4*8)/6 = 10.
    # Score ignores the money flags: S 9 and H 5 -> typical 7; person 1 n=1: (9 + 4*7)/5 = 7.4.
    h = history(
        [
            film(1, "2010-01-01", rev=20, bud=20, votes=100, score=9, suspect=1),
            film(2, "2010-02-01", rev=20, bud=20, incomplete=1),
            film(3, "2010-03-01", rev=10, bud=8),
            film(4, "2010-04-01"),
            film(5, "2009-01-01", rev=6, bud=4, votes=100, score=5),
        ],
        [(1, 1, "director"), (2, 1, "director"), (3, 1, "director"), (4, 1, "director"), (5, 2, "director")],
    )
    as_of = d("2020-01-01")
    assert h.typical(as_of) == pytest.approx((8, 7, 8))
    rec = h.person(1, "director", as_of)
    assert rec.films == 4
    assert (rec.revenue_log, rec.score, rec.budget_log) == pytest.approx((8.4, 7.4, 10.0))


def test_median_of_an_even_count():
    # Person 1 revenues 2, 4, 6, 100 -> median (4+6)/2 = 5 (a mean would be 28). Person 2 has an earlier film with revenue 0,
    # so typical {0, 2, 4, 6, 100} -> 4. Shrunk: (4*5 + 4*4)/8 = 4.5.
    h = history(
        [
            film(1, "2010-01-01", rev=2),
            film(2, "2010-02-01", rev=4),
            film(3, "2010-03-01", rev=6),
            film(4, "2010-04-01", rev=100),
            film(5, "2009-01-01", rev=0),
        ],
        [(1, 1, "director"), (2, 1, "director"), (3, 1, "director"), (4, 1, "director"), (5, 2, "director")],
    )
    assert h.typical(d("2020-01-01"))[0] == pytest.approx(4.0)
    assert h.person(1, "director", d("2020-01-01")).revenue_log == pytest.approx(4.5)


def test_roles_are_separate_tracks():
    # Person 1 directs film 1 (rev 10) and acts in films 2 and 3 (rev 20, 30); film 4 (rev 0) is by person 2 for typical.
    # Typical revenue {10, 20, 30, 0} -> 15.
    h = history(
        [film(1, "2010-01-01", rev=10), film(2, "2011-01-01", rev=20), film(3, "2012-01-01", rev=30), film(4, "2009-01-01", rev=0)],
        [(1, 1, "director"), (2, 1, "actor"), (3, 1, "actor"), (4, 2, "writer")],
    )
    as_of = d("2020-01-01")
    director, actor = h.person(1, "director", as_of), h.person(1, "actor", as_of)
    assert (director.films, actor.films) == (1, 2)
    assert director.revenue_log == pytest.approx((10 + 4 * 15) / 5)  # 14.0
    assert actor.revenue_log == pytest.approx((2 * 25 + 4 * 15) / 6)  # median 25 -> 110/6
    for role in ("writer", "producer", "composer"):
        rec = h.person(1, role, as_of)
        assert rec.is_new and rec.films == 0 and rec.revenue_log == pytest.approx(15)
    assert h.person(2, "writer", as_of).films == 1 and h.person(2, "director", as_of).is_new


def test_each_film_counts_once_per_track_even_with_two_credits():
    rows = [(1, 1, "director"), (1, 1, "director"), (2, 1, "director")]
    h = history([film(1, "2010-01-01", rev=1), film(2, "2011-01-01", rev=3)], rows, unique_credits=False)
    rec = h.person(1, "director", d("2020-01-01"))
    # Two films, revenue {1, 3} -> median 2, typical {1, 3} -> 2: (2*2 + 4*2)/6 = 2. A double count of film 1 would give n=3.
    assert rec.films == 2 and rec.revenue_log == pytest.approx(2.0)
    links = [(1, 7), (1, 7), (2, 7)]  # the same company twice on film 1
    con = make_db([film(1, "2010-01-01"), film(2, "2011-01-01")])
    con.execute("DROP TABLE movie_companies")
    con.execute("CREATE TABLE movie_companies (movie_id INTEGER, company_id INTEGER, ord INTEGER)")
    con.executemany("INSERT INTO movie_companies VALUES (?, ?, 0)", links)
    assert History.from_db(con, CFG).company(7, d("2020-01-01")).films == 2


def test_company_record():
    # Company 7 made films 1 (rev 10) and 2 (rev 20); company 8 only film 2; film 3 (rev 0, nobody's company) sets typical.
    # Typical {10, 20, 0} -> 10. Company 7: n=2 median 15 -> (2*15 + 4*10)/6 = 70/6. Company 8: n=1 -> (20 + 40)/5 = 12.
    h = history(
        [film(1, "2010-01-01", rev=10), film(2, "2011-01-01", rev=20), film(3, "2009-01-01", rev=0)],
        companies=[(1, 7), (2, 7), (2, 8)],
    )
    as_of = d("2020-01-01")
    seven, eight = h.company(7, as_of), h.company(8, as_of)
    assert (seven.films, eight.films) == (2, 1)
    assert seven.revenue_log == pytest.approx(70 / 6) and eight.revenue_log == pytest.approx(12.0)
    assert seven.years_active == 9 and not seven.is_new  # 2010-01-01 to 2020-01-01 is 3652 days = 9.9986 years
    new = h.company(999, as_of)
    assert same(new, h.company(None, as_of)) and new.is_new and new.revenue_log == pytest.approx(10)
    # Company 8 had no earlier film before 2011-01-01 (film 2 is the same day): new.
    assert h.company(8, d("2011-01-01")).is_new


def test_distributor_record():
    # "Universal" (stored with different case and spaces) made films 1 (rev 10) and 2 (rev 20). Film 3 (rev 0, no distributor),
    # film 4 (rev 60) and film 5 (rev 70) belong to "Other" or nobody. Typical {0, 10, 20, 60, 70} -> 20.
    # Universal: n=2 median 15 -> (2*15 + 4*20)/6 = 110/6.
    h = history(
        [
            film(1, "2010-01-01", rev=10, distributor="Universal"),
            film(2, "2011-01-01", rev=20, distributor=" universal "),
            film(3, "2009-01-01", rev=0),
            film(4, "2012-01-01", rev=60, distributor="Other"),
            film(5, "2013-01-01", rev=70, distributor="Other"),
        ],
    )
    as_of = d("2020-01-01")
    assert h.typical(as_of)[0] == pytest.approx(20)
    for spelling in ("Universal", "  UNIVERSAL ", "universal"):
        rec = h.distributor(spelling, as_of)
        assert rec.films == 2 and rec.revenue_log == pytest.approx(110 / 6) and rec.years_active == 9
    for unknown in (None, "No Such Studio"):
        rec = h.distributor(unknown, as_of)
        assert rec.is_new and rec.films == 0 and rec.revenue_log == pytest.approx(20)
    assert h.distributor("Universal", d("2010-01-01")).is_new  # film 1's own day


# ---- resolve ----

PEOPLE = [(1, "Ann Lee"), (2, "Bob Roy"), (3, "Zoë Fox"), (4, "Zoe Fox"), (5, "Twin"), (6, "Twin"), (7, "Twin"), (8, "Ann Lee")]
FILMS = [film(i, f"20{10 + i}-01-01") for i in range(1, 6)]
CREDITS = [
    (1, 1, "director"),
    (2, 1, "actor"),  # Ann Lee (1): one directing film, one acting film
    (3, 2, "director"),
    (4, 3, "director"),
    (5, 4, "director"),  # Zoë and Zoe: different people
    (1, 5, "director"),
    (2, 6, "director"),
    (3, 6, "director"),  # Twin 6 has 2 films, Twin 5 has 1
    (4, 7, "director"),  # Twin 7 has 1: ties with Twin 5, loses to Twin 6
    (5, 8, "director"),  # a second Ann Lee, a director too, with one film, like Ann Lee (1)
]
COMPANY_NAMES = [(10, "Big Studio"), (11, "Big Studio"), (12, "Tiny Co")]
COMPANY_LINKS = [(1, 10), (2, 11), (3, 11), (4, 12)]


@pytest.fixture
def named():
    return History.from_db(make_db(FILMS, CREDITS, COMPANY_LINKS, PEOPLE, COMPANY_NAMES), CFG)


def test_resolve_known_id_wins_over_name(named):
    assert named.resolve({"id": 2, "name": "Ann Lee"}, "director") == 2
    assert named.resolve(2, "director") == 2


def test_resolve_unknown_id_falls_back_to_name(named):
    assert named.resolve({"id": 999, "name": "Bob Roy"}, "director") == 2
    # id 2 is a director only; as an actor it is unknown, so the name is used (and nobody acts as "Bob Roy").
    assert named.resolve({"id": 2, "name": "Ann Lee"}, "actor") == 1


def test_resolve_name_is_trimmed_and_casefolded(named):
    assert named.resolve("  BOB roy ", "director") == 2
    assert named.resolve({"name": "bob ROY"}, "director") == 2


def test_resolve_name_is_exact_and_accents_matter(named):
    assert named.resolve("Zoë Fox", "director") == 3
    assert named.resolve("Zoe Fox", "director") == 4
    assert named.resolve("Zoe", "director") is None
    assert named.resolve("Bob", "director") is None


def test_resolve_most_films_wins_ties_lowest_id(named):
    assert named.resolve("Twin", "director") == 6  # 2 films beats 1 film
    assert named.resolve("Ann Lee", "director") == 1  # 1 film each: the lowest id


def test_resolve_roles_are_separate(named):
    assert named.resolve("Ann Lee", "actor") == 1
    assert named.resolve("Bob Roy", "actor") is None
    assert named.resolve("Twin", "writer") is None


def test_resolve_unknown_gives_none(named):
    assert named.resolve(None, "director") is None
    assert named.resolve(12345, "director") is None
    assert named.resolve("Nobody", "director") is None
    assert named.resolve({"id": None, "name": None}, "director") is None
    assert named.resolve({"id": 12345, "name": "  "}, "director") is None


def test_resolve_company_by_id_and_name(named):
    assert named.resolve(12, "company") == 12
    assert named.resolve(" big STUDIO", "company") == 11  # 2 films beats 1
    assert named.resolve({"id": 999, "name": "Tiny Co"}, "company") == 12
    assert named.resolve("Nobody", "company") is None


def test_resolve_and_person_reject_unknown_role(named):
    with pytest.raises(ValueError):
        named.resolve("Ann Lee", "gaffer")
    with pytest.raises(ValueError):
        named.resolve(None, "distributor")
    with pytest.raises(ValueError):
        named.person(1, "gaffer", d("2020-01-01"))


# ---- years_active, NaN, empty ----


def test_years_active_is_whole_years_from_first_earlier_film():
    h = history([film(1, "2000-01-01"), film(2, "2005-06-01")], [(1, 1, "director"), (2, 1, "director")])
    # 2000-01-01 to 2010-01-01 is 3653 days = 10.0014 years -> 10; one day less is 3652 days = 9.9986 -> 9.
    assert h.person(1, "director", d("2010-01-01")).years_active == 10
    assert h.person(1, "director", d("2009-12-31")).years_active == 9
    assert h.person(1, "director", d("2000-01-02")).years_active == 0  # one film, one day
    assert h.person(1, "director", d("2000-01-01")).years_active == 0  # new
    assert h.person(1, "director", d("2000-01-01")).is_new
    assert h.person(1, "director", d("2000-01-02")).films == 1 and not h.person(1, "director", d("2000-01-02")).is_new
    assert h.person(1, "director", d("2001-01-01")).years_active == 1  # 366 days / 365.25 = 1.002


def test_typical_is_nan_where_no_earlier_film_has_a_value():
    h = history([film(1, "2010-01-01", rev=5), film(2, "2012-01-01", bud=4, votes=100, score=6)])
    typical = h.typical(d("2010-01-01"))
    assert all(math.isnan(v) for v in typical)
    # 2010-07-01 is 181 days after film 1, 2010-07-02 is 182 days: the revenue counts only on the last.
    assert math.isnan(h.typical(d("2010-07-01"))[0]) and h.typical(d("2010-07-02"))[0] == pytest.approx(5)
    # Film 1 has no score or budget at all: those stay NaN. Film 2 (2012-01-01) is 152 days old at 2012-06-01, so its score
    # (6) is not available yet but its budget (4) is.
    after = h.typical(d("2012-06-01"))
    assert math.isnan(after[1]) and after[2] == pytest.approx(4)
    assert h.typical(d("2012-07-01"))[1] == pytest.approx(6)  # 182 days after 2012-01-01


def test_empty_database_never_raises():
    h = history([])
    rec = h.person(1, "director", d("2020-01-01"))
    assert rec.is_new and rec.films == 0 and math.isnan(rec.revenue_log)
    assert h.company(1, d("2020-01-01")).is_new and h.distributor("x", d("2020-01-01")).is_new
    assert all(math.isnan(v) for v in h.typical(d("2020-01-01")))
    assert h.resolve("x", "director") is None


def test_default_config_is_loaded():
    h = History.from_db(make_db([film(1, "2010-01-01", rev=5)]))
    assert h.typical(d("2010-07-02"))[0] == pytest.approx(5)  # mature_days 182 from config/data.yaml


# ---- the mini set, real data ----


def test_nolan_as_of_tenet(tmp_path):
    # Christopher Nolan (TMDB 525) directed, before Tenet (2020-08-22), in the mini shards: Memento 2000, Batman Begins 2005,
    # The Dark Knight 2008, Inception 2010, The Dark Knight Rises 2012, Interstellar 2014, Dunkirk 2017 = 7 films.
    # (Tenet itself and Oppenheimer 2023 do not count.) 2000-10-11 to 2020-08-22 is 7255 days = 19.86 years -> 19.
    out = tmp_path / "mini.db"
    build_db(MINI / "tmdb", MINI / "wd_films", MINI / "wd_people", out, PULLED_AT)
    h = History.from_db(sqlite3.connect(out), CFG)
    rec = h.person(525, "director", date(2020, 8, 22))
    assert rec.films == 7 and not rec.is_new and rec.years_active == 19
    assert h.resolve("christopher nolan", "director") == 525
    assert h.person(525, "actor", date(2020, 8, 22)).films == 0
    assert h.person(525, "director", date(2000, 10, 11)).is_new
