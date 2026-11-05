import json
from dataclasses import fields
from datetime import date, timedelta
from pathlib import Path

import pytest

from llm_movies.data.clean import (
    FESTIVAL_GAP_DAYS,
    LOW_RATIO,
    STREAMER_START_YEARS,
    MovieRow,
    clean_movie,
    pick_distributor,
)
from llm_movies.data.wikidata import FilmFacts, Money

CPI = {2000: 50.0, 2001: 80.0, 2025: 100.0}  # 2000 dollars x 2 = 2025 dollars
PULLED = date(2026, 1, 1)


def raw_film(**over) -> dict:
    raw = {
        "id": 1,
        "title": "Test Film",
        "status": "Released",
        "release_date": "2000-06-01",
        "runtime": 100,
        "original_language": "en",
        "production_countries": [{"iso_3166_1": "US"}],
        "genres": [{"name": "Drama"}],
        "keywords": {"keywords": [{"name": "based on novel or book"}]},
        "overview": "An overview.",
        "tagline": "A tagline.",
        "belongs_to_collection": {"id": 77},
        "budget": 0,
        "revenue": 0,
        "vote_count": 10,
        "vote_average": 6.5,
        "release_dates": {"results": []},
    }
    raw.update(over)
    return raw


def facts(**over) -> FilmFacts:
    base = {
        "qid": "Q1",
        "source_types": set(),
        "source_sitelinks": None,
        "distributors": [],
        "writers_qids": [],
        "producers_qids": [],
        "composer_qids": [],
        "series_qid": None,
        "follows_qid": None,
        "box_office": [],
        "cost": [],
    }
    base.update(over)
    return FilmFacts(**base)


def clean(raw=None, wd=None, cpi=CPI, pulled_at=PULLED) -> MovieRow:
    row = clean_movie(raw or raw_film(), wd, cpi, {}, pulled_at)
    assert row is not None
    return row


def rd(kind: int, day: str, cert: str = "") -> dict:
    return {"type": kind, "release_date": f"{day}T00:00:00.000Z", "certification": cert}


def country(iso: str, *dates: dict) -> dict:
    return {"iso_3166_1": iso, "release_dates": list(dates)}


def with_dates(*results: dict) -> dict:
    return raw_film(release_dates={"results": list(results)})


# ---- drop rule ----


@pytest.mark.parametrize(
    "over",
    [
        {"release_date": ""},
        {"release_date": None},
        {"release_date": "not-a-date"},
        {"release_date": "2000-13-45"},
        {"status": "Rumored"},
        {"status": "Canceled"},
        {"status": None},
    ],
)
def test_unreleased_without_date_dropped(over):
    assert clean_movie(raw_film(**over), None, CPI, {}, PULLED) is None


@pytest.mark.parametrize("status", ["Released", "Post Production", "In Production", "Planned"])
def test_four_statuses_kept(status):
    assert clean(raw_film(status=status)).year == 2000


def test_post_release_fields_kept_only_as_targets():
    row = clean()
    assert (row.vote_count, row.vote_average) == (10, 6.5)


def test_popularity_absent_from_movie_row():
    assert "popularity" not in {f.name for f in fields(MovieRow)}
    assert not hasattr(clean(raw_film(popularity=99.0)), "popularity")


def test_empty_values_become_none():
    row = clean(raw_film(title="", overview="", tagline="  ", original_language="", runtime=0, belongs_to_collection=None))
    assert (row.title, row.overview, row.tagline, row.language, row.runtime, row.collection_id) == (None,) * 6


def test_plain_fields_copied():
    row = clean()
    assert (row.tmdb_id, row.title, row.year, row.month, row.day) == (1, "Test Film", 2000, 6, 1)
    assert (row.runtime, row.language, row.countries, row.genres) == (100, "en", ["US"], ["Drama"])
    assert (row.keywords, row.collection_id, row.wd_qid) == (["based on novel or book"], 77, None)
    assert row.source_material == "novel"


def test_every_nested_key_missing_or_null_does_not_crash():
    bare = {"id": 5, "status": "Released", "release_date": "2000-06-01"}
    nulls = {k: None for k in raw_film() if k not in ("id", "status", "release_date")}
    for raw in (bare, {**bare, **nulls}):
        row = clean(raw)
        assert (row.runtime, row.language, row.countries, row.genres, row.keywords) == (None, None, [], [], [])
        assert (row.certification, row.release_type, row.budget_2025, row.revenue_2025) == ("unrated/unknown", None, None, None)
        assert (row.source_material, row.distributor, row.collection_id) == ("original", None, None)
    odd = raw_film(
        production_countries=[None, {}],
        genres=[{"name": None}],
        keywords={"keywords": None},
        release_dates={"results": [None, {"iso_3166_1": "US", "release_dates": None}, {"release_dates": [None, {}]}]},
    )
    assert clean(odd).release_type is None


# ---- money ----


def test_tmdb_money_wins_over_wikidata():
    wd = facts(box_office=[Money(9e9, "USD", None, 2000)], cost=[Money(9e9, "USD", None, 2000)])
    row = clean(raw_film(revenue=100_000_000, budget=40_000_000), wd)
    assert row.revenue_2025 == pytest.approx(200_000_000)
    assert row.budget_2025 == pytest.approx(80_000_000)
    assert row.revenue_source == "tmdb"


@pytest.mark.parametrize("figure_year", [None, 2001])
def test_wikidata_fills_missing_revenue(figure_year):
    # Release year 2000: CPI 2025 / CPI 2000 = 100 / 50 = 2, so 200M x 2 = 400M. The figure's own year (2001, CPI 80)
    # would give 250M and an undated figure none at all: the release year is always used.
    wd = facts(box_office=[Money(200_000_000, "USD", None, figure_year)], cost=[Money(30_000_000, "USD", None, None)])
    row = clean(raw_film(revenue=0, budget=0), wd)
    assert row.revenue_2025 == pytest.approx(400_000_000)
    assert row.budget_2025 == pytest.approx(60_000_000)
    assert row.revenue_source == "wikidata"


def test_wikidata_not_used_when_tmdb_present_but_cpi_missing():
    wd = facts(box_office=[Money(200_000_000, "USD", None, None)])
    row = clean(raw_film(release_date="1950-01-01", revenue=100_000_000), wd)
    assert (row.revenue_2025, row.revenue_source) == (None, None)


def test_revenue_source_none_when_cpi_missing_and_row_stays():
    wd = facts(box_office=[Money(200_000_000, "USD", None, None)])
    row = clean(raw_film(release_date="1950-01-01", revenue=0, budget=5_000_000), wd)
    assert (row.revenue_2025, row.budget_2025, row.revenue_source, row.year) == (None, None, None, 1950)
    assert row.money_suspect is False


def test_revenue_source_none_when_no_revenue_anywhere():
    assert clean(raw_film(revenue=0), facts()).revenue_source is None


def test_tmdb_placeholder_below_minimum_is_unknown():
    wd = facts(box_office=[Money(200_000_000, "USD", None, None)])
    row = clean(raw_film(revenue=12), wd)  # a placeholder like 12 means unknown, so Wikidata fills in
    assert (row.revenue_source, row.revenue_2025) == ("wikidata", pytest.approx(400_000_000))


@pytest.mark.parametrize("field", ["budget", "revenue"])
@pytest.mark.parametrize(
    ("amount", "expected"),
    [
        (0, None),  # TMDB 0 means unknown
        (None, None),
        (float("nan"), None),
        (999, None),  # below 1,000 is a placeholder
        (1_000, 2_000.0),  # the 1,000 floor itself is real: x2 (CPI 100 / 50)
    ],
)
def test_money_zero_nan_and_placeholder_boundary(field, amount, expected):
    row = clean(raw_film(**{field: amount}))
    got = getattr(row, f"{field}_2025")
    assert got == (pytest.approx(expected) if expected is not None else None)


def test_wikidata_placeholder_below_minimum_is_unknown():
    wd = facts(box_office=[Money(999, "USD", None, None)], cost=[Money(500, "USD", None, 2000)])
    row = clean(raw_film(), wd)
    assert (row.revenue_2025, row.budget_2025, row.revenue_source) == (None, None, None)


def test_low_ratio_flags_suspect():
    assert LOW_RATIO == 0.0001
    low = clean(raw_film(budget=100_000_000, revenue=9_000))  # ratio 0.00009 < LOW_RATIO
    assert low.money_suspect is True


def test_ratio_at_or_above_low_ratio_not_suspect():
    assert clean(raw_film(budget=100_000_000, revenue=20_000)).money_suspect is False  # ratio 0.0002
    assert clean(raw_film(budget=100_000_000, revenue=100_000_000)).money_suspect is False


def test_ratio_exactly_low_ratio_not_suspect():
    # 10,000 / 100,000,000 = 0.0001 exactly; the rule is strictly below
    assert clean(raw_film(budget=100_000_000, revenue=10_000)).money_suspect is False


def test_suspect_needs_both_values():
    assert clean(raw_film(budget=0, revenue=5_000)).money_suspect is False
    assert clean(raw_film(budget=100_000_000, revenue=0)).money_suspect is False


def test_revenue_incomplete_boundary():
    released = date(2000, 6, 1)
    assert clean(pulled_at=released + timedelta(days=181)).revenue_incomplete is True
    assert clean(pulled_at=released + timedelta(days=182)).revenue_incomplete is False


def test_release_after_pulled_at_is_incomplete():
    # released 2000-06-01: one month and more than 182 days (about 17 months) before it; both are not yet earned
    assert clean(pulled_at=date(2000, 5, 1)).revenue_incomplete is True
    assert clean(pulled_at=date(1998, 12, 1)).revenue_incomplete is True


# ---- festival gap ----


def test_festival_gap_boundary():
    assert FESTIVAL_GAP_DAYS == 30
    thirty = with_dates(country("US", rd(1, "2000-05-02"), rd(3, "2000-06-01")))
    twenty_nine = with_dates(country("US", rd(1, "2000-05-03"), rd(3, "2000-06-01")))
    assert clean(thirty).release_type == "festival-first"
    assert clean(twenty_nine).release_type == "wide"


def test_festival_gap_below_threshold_falls_through_to_limited():
    short = with_dates(country("US", rd(1, "2000-05-20"), rd(2, "2000-06-01")))
    assert clean(short).release_type == "limited"


def test_gap_runs_from_the_earliest_premiere():
    # a festival debut on 1 March, then a red-carpet premiere a week before opening: 92 days from the earliest
    film = with_dates(country("US", rd(1, "2000-03-01"), rd(1, "2000-05-25"), rd(3, "2000-06-01")))
    assert clean(film).release_type == "festival-first"


def test_gap_runs_to_the_first_theatrical_date():
    # premiere 1 May, limited opening 10 May (9 days), wide expansion 15 July (75 days): the first opening counts
    film = with_dates(country("US", rd(1, "2000-05-01"), rd(2, "2000-05-10"), rd(3, "2000-07-15")))
    assert clean(film).release_type == "wide"


def test_true_festival_film_stays_festival_first():
    # Cannes in May, wide release in October: a gap of well over 30 days
    film = with_dates(country("US", rd(1, "2000-05-15"), rd(2, "2000-09-01"), rd(3, "2000-10-06")))
    assert clean(film).release_type == "festival-first"


# ---- streamer labels ----


def streamer_film(label_year: int, results: list[dict], distributors: list[str]) -> MovieRow:
    raw = raw_film(release_date=f"{label_year}-06-01", release_dates={"results": results})
    return clean(raw, facts(distributors=distributors), cpi={label_year: 90.0, 2025: 100.0})


@pytest.mark.parametrize(
    "dates",
    [
        [rd(4, "2022-06-01")],  # streaming-only
        [rd(2, "2022-06-01")],  # limited
        [rd(1, "2022-01-01"), rd(2, "2022-06-01")],  # festival-first
    ],
)
def test_streamer_kept_for_streaming_limited_festival_films(dates):
    assert streamer_film(2022, [country("US", *dates)], ["Netflix"]).distributor == "Netflix"


def test_streamer_dropped_on_wide_film():
    row = streamer_film(2022, [country("US", rd(3, "2022-06-01"))], ["Netflix"])
    assert (row.release_type, row.distributor) == ("wide", None)


def test_streamer_dropped_without_release_type():
    assert streamer_film(2022, [], ["Netflix"]).distributor is None


def test_streamer_dropped_before_its_start_year_even_when_streaming_only():
    # Disney+ launched in November 2019: a 2018 film cannot have been released there (Fight Club in the real data)
    row = streamer_film(2018, [country("US", rd(4, "2018-06-01"))], ["Disney+"])
    assert (row.release_type, row.distributor) == ("streaming-only", None)


def test_streamer_kept_in_its_start_year():
    row = streamer_film(2019, [country("US", rd(4, "2019-06-01"))], ["Disney+"])
    assert row.distributor == "Disney+"


def test_wide_film_with_streamer_and_studio_gives_studio():
    row = streamer_film(2022, [country("US", rd(3, "2022-06-01"))], ["Netflix", "Warner Bros. Pictures"])
    assert row.distributor == "Warner Bros."


def test_streamer_dropped_falls_to_other_labels():
    row = streamer_film(2022, [country("US", rd(3, "2022-06-01"))], ["InterCom", "Netflix"])
    assert row.distributor == "InterCom"


def test_streamer_start_years_are_set_for_every_streamer_family():
    assert STREAMER_START_YEARS == {
        "Netflix": 2007,
        "Disney+": 2019,
        "HBO Max": 2020,
        "Amazon Prime Video": 2006,
        "Paramount+": 2021,
        "Apple TV": 2019,
        "Hulu": 2007,
        "Starz": 1994,
    }


# ---- distributor ----


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        (["Warner Bros. Entertainment"], "Warner Bros."),  # Joker
        (["InterCom", "Warner Bros. Entertainment"], "Warner Bros."),  # a major beats the sorted first
        (["InterCom"], "InterCom"),  # no major: the first label
        (["Fandango", "InterCom"], "InterCom"),  # a storefront is not a distributor
        (["FANDANGO", "FandangoNow", "Xfinity Streampix"], None),  # only storefronts: nothing
        (["Fandango", "Netflix"], "Netflix"),
        (["Fandango Media"], "Fandango Media"),  # the blocklist is exact-name
        (["Lionsgate+"], "Lionsgate+"),  # the word boundary applies to every prefix of a family
        (["New Line Cinemax"], "New Line Cinemax"),
        (["Walt Disneyland"], "Walt Disneyland"),
        (["Apple TV+"], "Apple TV"),
        (["Sony Pictures"], "Sony"),
        (["Sony Pictures Motion Picture Group"], "Sony"),
        (["Sony Pictures Releasing France"], "Sony"),
        (["Sony Pictures Television"], "Sony Pictures Television"),  # a TV arm, left apart
        (["MGM"], "Metro-Goldwyn-Mayer"),
        (["MGM+"], "MGM+"),
        (["Metro-Goldwyn-Mayer"], "Metro-Goldwyn-Mayer"),
        (["Amazon MGM"], "Amazon MGM Studios"),
        (["Amazon MGM Studios Distribution"], "Amazon MGM Studios"),
        ([], None),
        (["Netflix", "Walt Disney Studios Motion Pictures"], "Disney"),  # families are ordered, studios before streamers
        (["Disney+", "Netflix"], "Netflix"),
        (["Disney+"], "Disney+"),
        (["Paramount+"], "Paramount+"),  # not caught by the Paramount studio prefix
        (["Paramount Vantage"], "Paramount Vantage"),  # specialty arm, left apart
        (["Sony Pictures Classics"], "Sony Pictures Classics"),
        (["Columbia Pictures"], "Sony"),
        (["Warner Bros. France"], "Warner Bros."),
        (["warner bros. pictures"], "Warner Bros."),
        (["Universal Pictures"], "Universal"),
        (["Universal Express"], "Universal Express"),  # a prefix of the family is not enough
        (["Tohoku Films"], "Tohoku Films"),  # word boundary
        (["20th Century Studios"], "20th Century"),
    ],
)
def test_distributor_table(labels, expected):
    assert pick_distributor(labels, "streaming-only", 2022) == expected  # a context where streamer labels count


def test_blocklist_runs_before_the_major_table(monkeypatch):
    # no real storefront matches a family, so make one that would: a blocked name must never reach the table
    monkeypatch.setattr("llm_movies.data.clean.STOREFRONTS", frozenset({"netflix"}))
    assert pick_distributor(["Netflix"], "streaming-only", 2022) is None
    assert pick_distributor(["Netflix", "InterCom"], "streaming-only", 2022) == "InterCom"


def test_distributor_comes_from_wikidata_facts():
    assert clean(wd=facts(distributors=["InterCom", "Warner Bros. Entertainment"])).distributor == "Warner Bros."
    assert clean(wd=facts(distributors=["InterCom"])).distributor == "InterCom"
    assert clean(wd=None).distributor is None


def test_wikidata_qid_and_source_material():
    row = clean(raw_film(keywords=None), facts(qid="Q42", source_types={"Q1004"}))
    assert (row.wd_qid, row.source_material) == ("Q42", "comic")


def test_source_sitelinks_come_from_wikidata_facts():
    assert clean(wd=facts(source_sitelinks=12)).source_sitelinks == 12
    assert clean(wd=facts(source_sitelinks=None)).source_sitelinks is None
    assert clean(wd=None).source_sitelinks is None


# ---- release type ----

WIDE_LIMITED = [rd(2, "2000-06-01"), rd(3, "2000-06-10")]


@pytest.mark.parametrize(
    ("results", "expected"),
    [
        ([country("US", rd(1, "2000-05-01"), rd(3, "2000-06-01"))], "festival-first"),
        ([country("US", rd(1, "2000-05-01"), rd(2, "2000-06-01"))], "festival-first"),
        ([country("US", rd(1, "2000-06-01"), rd(3, "2000-06-01"))], "wide"),  # same day is not "strictly earlier"
        ([country("US", rd(3, "2000-06-01"))], "wide"),
        ([country("US", *WIDE_LIMITED)], "wide"),
        ([country("US", rd(2, "2000-06-01"))], "limited"),
        ([country("US", rd(4, "2000-06-01"))], "streaming-only"),
        ([country("US", rd(5, "2000-06-01"), rd(6, "2000-07-01"))], "streaming-only"),
        ([country("US", rd(1, "2000-05-01"), rd(4, "2000-06-01"))], "streaming-only"),  # even after a premiere
        ([country("US", rd(1, "2000-05-01"))], None),  # a premiere alone
        ([country("US", rd(1, "2000-05-01"), rd(4, "2000-06-01"), rd(3, "2000-07-01"))], "festival-first"),
        ([country("US")], None),
        ([], None),
        # a US entry with no dates counts as absent: the earliest country is used
        ([country("US"), country("FR", rd(2, "2000-06-01"))], "limited"),
        # earliest country when there is no US entry; GB and FR tie, FR is first alphabetically
        ([country("GB", rd(3, "2000-06-01")), country("FR", rd(2, "2000-06-01"))], "limited"),
        ([country("DE", rd(3, "2000-07-01")), country("FR", rd(2, "2000-06-01"))], "limited"),
        # a premiere after the theatrical release is not festival-first
        ([country("US", rd(1, "2000-07-01"), rd(3, "2000-06-01"))], "wide"),
        ([country("US", rd(1, "2000-07-01"), rd(2, "2000-06-01"))], "limited"),
        # any wide beats limited; theatrical beats streaming
        ([country("US", rd(2, "2000-06-01"), rd(3, "2000-06-02"), rd(4, "2000-06-03"))], "wide"),
        ([country("US", rd(4, "2000-06-01"), rd(3, "2000-06-02"))], "wide"),
        ([country("US", rd(4, "2000-06-01"), rd(2, "2000-06-02"))], "limited"),
        # the earliest premiere counts against the earliest theatrical date
        ([country("US", rd(1, "2000-07-01"), rd(1, "2000-05-01"), rd(3, "2000-06-01"))], "festival-first"),
        # a premiere in another country does not make a US release festival-first
        ([country("FR", rd(1, "2000-01-01")), country("US", rd(3, "2000-06-01"))], "wide"),
        # US wins even when another country is earlier
        ([country("FR", rd(3, "1999-06-01")), country("US", rd(2, "2000-06-01"))], "limited"),
        # unparseable dates are ignored
        ([country("US", {"type": 3, "release_date": "garbage"}, rd(2, "2000-06-01"))], "limited"),
        ([country("US", {"type": 3, "release_date": ""})], None),
    ],
)
def test_release_type_rules(results, expected):
    assert clean(with_dates(*results)).release_type == expected


# ---- certification ----


def test_certification_prefers_theatrical_in_date_order():
    results = country("US", rd(4, "2000-01-01", "PG"), rd(3, "2000-06-10", "R"), rd(2, "2000-06-01", "PG-13"))
    assert clean(with_dates(results)).certification == "PG-13"  # earliest theatrical date, not list order or digital


def test_certification_skips_empty_theatrical_values():
    results = country("US", rd(2, "2000-06-01", ""), rd(3, "2000-06-10", "R"))
    assert clean(with_dates(results)).certification == "R"


def test_certification_falls_back_to_any_type_in_date_order():
    results = country("US", rd(5, "2000-08-01", "PG"), rd(4, "2000-07-01", "G"))
    assert clean(with_dates(results)).certification == "G"


def test_certification_ignores_other_countries():
    results = [country("GB", rd(3, "2000-06-01", "15")), country("US", rd(3, "2000-06-02", "R"))]
    assert clean(with_dates(*results)).certification == "R"
    assert clean(with_dates(country("GB", rd(3, "2000-06-01", "R")))).certification == "unrated/unknown"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("G", "G"),
        ("PG", "PG"),
        ("PG-13", "PG-13"),
        ("R", "R"),
        ("NC-17", "NC-17"),
        ("NR", "unrated/unknown"),
        ("Unrated", "unrated/unknown"),
        ("X", "unrated/unknown"),
        ("", "unrated/unknown"),
    ],
)
def test_certification_mapping(value, expected):
    assert clean(with_dates(country("US", rd(3, "2000-06-01", value)))).certification == expected


# ---- a real TMDB response ----


def test_real_joker_response():
    raw = json.loads((Path(__file__).parents[1] / "fixtures" / "tmdb_movie_475557.json").read_text(encoding="utf-8"))
    assert (raw["budget"], raw["revenue"]) == (55_000_000, 1_078_958_629)  # the working below uses these raw values
    wd = facts(qid="Q60", distributors=["Warner Bros. Entertainment"])
    cpi = {2019: 80.0, 2025: 100.0}  # 2019 dollars x 1.25 = 2025 dollars
    row = clean_movie(raw, wd, cpi, {}, date(2026, 1, 1))
    assert row is not None

    assert (row.tmdb_id, row.title, row.release_date) == (475557, "Joker", date(2019, 10, 1))
    assert (row.year, row.month, row.day) == (2019, 10, 1)
    assert (row.language, row.certification, row.genres) == ("en", "R", ["Crime", "Thriller", "Drama"])
    assert row.collection_id == 987044
    assert (row.distributor, row.wd_qid) == ("Warner Bros.", "Q60")

    # By hand, CPI 2025 / CPI 2019 = 100 / 80 = 1.25:
    #   budget  55,000,000 x 1.25 =    68,750,000
    #   revenue 1,078,958,629 x 1.25 = 1,348,698,286.25
    assert row.budget_2025 == pytest.approx(68_750_000)
    assert row.revenue_2025 == pytest.approx(1_348_698_286.25)
    assert (row.revenue_source, row.money_suspect, row.revenue_incomplete) == ("tmdb", False, False)

    # US entries: a type 1 premiere on 2019-09-28 (note "Hollywood, California") and the type 3 wide release on
    # 2019-10-04, a gap of 6 days. That is below FESTIVAL_GAP_DAYS (30), so it is a red-carpet premiere, not a
    # festival debut, and the film reads "wide".
    assert row.release_type == "wide"
