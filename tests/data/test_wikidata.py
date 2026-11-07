import json
import math
import re
from datetime import date
from pathlib import Path

import pytest
import requests

from llm_movies.data import wikidata
from llm_movies.data.money import load_fx
from llm_movies.data.shards import done_keys, read_shard
from llm_movies.data.tmdb_ids import read_ids
from llm_movies.data.wikidata import (
    USER_AGENT,
    WORLDWIDE_QID,
    AwardEvent,
    FilmFacts,
    Money,
    PersonFacts,
    SparqlClient,
    SparqlError,
    film_query,
    parse_films,
    parse_people,
    people_query,
    pick_worldwide_usd,
    run_pull,
)

FIXTURES = Path(__file__).parent.parent / "fixtures"
# Real saved results of film_query / people_query (retrieved live from query.wikidata.org, unedited).
WD_FILMS = json.loads((FIXTURES / "wd_films.json").read_text(encoding="utf-8"))
WD_PEOPLE = json.loads((FIXTURES / "wd_people.json").read_text(encoding="utf-8"))
SAMPLE_IDS = FIXTURES / "tmdb_ids_sample.json.gz"

FX = {("EUR", 2005): 0.5, ("JPY", 2010): 100.0}  # hand-built: units per 1 USD

JOKER, DARK_KNIGHT, PARASITE, AMELIE, TITANIC = 475557, 155, 496243, 194, 597
PHOENIX, LEDGER, NOLAN = 73421, 1810, 525

WD = "http://www.wikidata.org/entity/"


# ---- hand-built SPARQL result helpers ----


def lit(value, datatype=None):
    out = {"type": "literal", "value": str(value)}
    if datatype:
        out["datatype"] = datatype
    return out


def uri(qid):
    return {"type": "uri", "value": WD + qid}


def row(tmdb, item, kind, **cols):
    out = {"tmdb": lit(tmdb), "item": uri(item), "kind": lit(kind)}
    for key, value in cols.items():
        out[key] = value
    return out


def sparql(*rows):
    return {"head": {"vars": []}, "results": {"bindings": list(rows)}}


def money_row(tmdb, item, kind, amount, unit="Q4917", place=None, when=None, prec="11"):
    cols = {"v": lit(amount, "http://www.w3.org/2001/XMLSchema#decimal"), "unit": uri(unit)}
    if place:
        cols["place"] = uri(place)
    if when:
        cols["date"] = lit(when, "http://www.w3.org/2001/XMLSchema#dateTime")
        cols["prec"] = lit(prec, "http://www.w3.org/2001/XMLSchema#integer")
    return row(tmdb, item, kind, **cols)


# ---- queries ----


def test_film_query_contains_values_and_properties():
    query = film_query([475557, 155])
    assert 'VALUES ?tmdb { "475557" "155" }' in query  # P4947 holds strings
    assert "wdt:P4947 ?tmdb" in query
    for prop in ("P144", "P750", "P58", "P162", "P86", "P179", "P155", "P2142", "P2130", "P3005", "P585", "P31"):
        assert prop in query, prop
    assert "wikibase:sitelinks" in query
    assert "UNION" in query
    assert "OPTIONAL { ?item" not in query  # no chained OPTIONALs on the film: they multiply and time out
    for statement_part in ("p:P2142", "psv:P2142", "pq:P3005", "pqv:P585", "wikibase:quantityUnit", "wikibase:timePrecision"):
        assert statement_part in query, statement_part
    assert "SERVICE wikibase:label" not in query


def _branch(query, kind):
    """The one UNION branch that labels its rows `kind`."""
    found = [b for b in query.split(" UNION ") if f'BIND("{kind}" AS ?kind)' in b]
    assert len(found) == 1, kind
    return found[0]


def test_film_query_each_branch_carries_its_own_property():
    # Each fact must come from its own Wikidata property (names: wikidata.org property pages) in its own branch;
    # a property that only appears elsewhere in the query would not catch a dropped or swapped branch.
    query = film_query([1])
    for kind, needed in {
        "source": ("wdt:P144", "wdt:P31", "wikibase:sitelinks"),
        "distributor": ("wdt:P750", "rdfs:label"),
        "writer": ("wdt:P58 ",),
        "producer": ("wdt:P162 ",),
        "composer": ("wdt:P86 ",),
        "series": ("wdt:P179 ",),
        "follows": ("wdt:P155 ",),
        "box_office": ("p:P2142", "psv:P2142", "pq:P3005", "pqv:P585", "wikibase:quantityUnit", "wikibase:timePrecision"),
        "cost": ("p:P2130", "psv:P2130", "pqv:P585", "wikibase:quantityUnit", "wikibase:timePrecision"),
    }.items():
        branch = _branch(query, kind)
        for part in needed:
            assert part in branch, (kind, part)
    assert "P2142" not in _branch(query, "cost") and "P2130" not in _branch(query, "box_office")


def test_money_and_award_branches_drop_deprecated_rank():
    for query, kinds in ((film_query([1]), ("box_office", "cost")), (people_query([1]), ("won", "nominated"))):
        for kind in kinds:
            branch = _branch(query, kind)
            assert "?st wikibase:rank ?r" in branch and "FILTER(?r != wikibase:DeprecatedRank)" in branch, kind


def test_people_query_each_branch_carries_its_own_property():
    query = people_query([1])
    assert "wdt:P569" in _branch(query, "birth")
    for kind, prop in (("won", "P166"), ("nominated", "P1411")):
        branch = _branch(query, kind)
        for part in (f"p:{prop}", f"ps:{prop}", "pqv:P585", "wikibase:timeValue", "wikibase:timePrecision"):
            assert part in branch, (kind, part)
    assert "P1411" not in _branch(query, "won") and "P166" not in _branch(query, "nominated")


def test_people_query_contains_values_and_properties():
    query = people_query([73421, 1810])
    assert 'VALUES ?tmdb { "73421" "1810" }' in query
    assert "wdt:P4985 ?tmdb" in query
    for prop in ("P569", "P166", "P1411", "P585"):
        assert prop in query, prop
    assert "wikibase:timePrecision" in query


def test_user_agent_has_no_email():
    assert "@" not in USER_AGENT
    assert USER_AGENT == "LLMMoviesPredictionModel/0.1 (https://github.com/Tsomaros/LLM_Movies_Model)"


def test_worldwide_qid_is_the_worldwide_item():
    # Checked on wikidata.org: Q13780930 is "worldwide" (a special value for place of publication / operating area).
    assert WORLDWIDE_QID == "Q13780930"


# ---- parsing real saved results ----


def test_parse_joker_distributor_and_source():
    films = parse_films(WD_FILMS)
    assert set(films) == {JOKER, DARK_KNIGHT, PARASITE, AMELIE, TITANIC}
    joker = films[JOKER]
    assert joker.qid == "Q42759035"  # wikidata.org: Joker (2019 film)
    # wikidata.org: Joker is based on the character The Joker (Q217533), an instance of these classes
    # (comics character, video game character, fictional human, animated character, television character,
    # film character, anime character), and the page has 68 Wikipedia language editions when saved.
    assert joker.source_types == {"Q1114461", "Q1569167", "Q15632617", "Q15711870", "Q15773317", "Q15773347", "Q80447738"}
    assert joker.source_sitelinks == 68
    assert joker.distributors == ["InterCom", "Warner Bros. Entertainment"]
    assert joker.cost == [Money(55_000_000.0, "USD", None, None)]
    assert pick_worldwide_usd(joker.box_office, FX) == (1_074_458_282.0, None)  # worldwide figure, undated


def test_film_without_source_has_empty_source_fields():
    films = parse_films(WD_FILMS)
    assert films[TITANIC].source_types == set()
    assert films[TITANIC].source_sitelinks is None
    assert films[TITANIC].series_qid is None
    assert films[TITANIC].follows_qid is None


def test_series_and_follows_take_the_lowest_numeric_qid():
    dark_knight = parse_films(WD_FILMS)[DARK_KNIGHT]
    # Three series on the item (Q2111133, Q12859908, Q26705935): lowest number wins, not the lexically smallest.
    assert dark_knight.series_qid == "Q2111133"
    assert dark_knight.follows_qid == "Q166262"  # Batman Begins


def test_dated_worldwide_figure_and_deprecated_statement_dropped():
    parasite = parse_films(WD_FILMS)[PARASITE]
    # Parasite also had a 204,581,001 worldwide figure dated 2020-02-23 that is deprecated rank on Wikidata; the
    # query leaves it out, so only the 2022 figure remains (fixture re-saved after adding the rank filter).
    assert [(m.amount, m.year) for m in parasite.box_office if m.place_qid == WORLDWIDE_QID] == [(263_136_741.0, 2022)]
    assert 204_581_001.0 not in [m.amount for m in parasite.box_office]
    assert pick_worldwide_usd(parasite.box_office, FX) == (263_136_741.0, 2022)


def test_non_worldwide_box_office_ignored():
    # The Dark Knight on wikidata.org: 158.4M and 535.0M carry place Q49 (North America); only 1.006B is worldwide.
    dark_knight = parse_films(WD_FILMS)[DARK_KNIGHT]
    places = {m.place_qid for m in dark_knight.box_office}
    assert places == {"Q49", WORLDWIDE_QID}
    assert pick_worldwide_usd(dark_knight.box_office, FX) == (1_006_234_167.0, None)
    only_regional = [m for m in dark_knight.box_office if m.place_qid != WORLDWIDE_QID]
    assert pick_worldwide_usd(only_regional, FX) == (None, None)
    assert pick_worldwide_usd([Money(5_000_000.0, "USD", "Q30", 2010)], FX) == (None, None)


def test_figure_without_place_counts_as_worldwide():
    amelie = parse_films(WD_FILMS)[AMELIE]
    # wikidata.org: Amelie's dated worldwide figure is 174,118,254 (2022); a placeless 173.9M figure (deprecated rank) is gone.
    assert pick_worldwide_usd(amelie.box_office, FX) == (174_118_254.0, 2022)
    assert pick_worldwide_usd([Money(173_900_000.0, "USD", None, None)], FX) == (173_900_000.0, None)


# ---- parsing hand-built results: conflicts and bad values ----


def test_multiple_values_pick_deterministically():
    result = sparql(
        row("1", "Q10", "item"),
        row("1", "Q10", "distributor", v=lit("Warner Bros.")),
        row("1", "Q10", "distributor", v=lit("Universal")),
        row("1", "Q10", "distributor", v=lit("Universal")),  # duplicate row
        row("1", "Q10", "writer", v=uri("Q9")),
        row("1", "Q10", "writer", v=uri("Q100")),
        row("1", "Q10", "writer", v=uri("Q9")),
        row("1", "Q10", "source", n=lit(30), v=uri("Q1")),
        row("1", "Q10", "source", n=lit(75), v=uri("Q2")),
        money_row("1", "Q10", "box_office", 100_000_000, place=WORLDWIDE_QID, when="2008-06-01T00:00:00Z"),
        money_row("1", "Q10", "box_office", 90_000_000, place=WORLDWIDE_QID, when="2005-06-01T00:00:00Z"),
        money_row("1", "Q10", "box_office", 800_000_000, unit="Q4916", place=WORLDWIDE_QID, when="2005-06-01T00:00:00Z"),
        money_row("1", "Q10", "box_office", 5_000_000_000, unit="Q99999999", place=WORLDWIDE_QID, when="2009-01-01T00:00:00Z"),
    )
    film = parse_films(result)[1]
    assert film.distributors == ["Universal", "Warner Bros."]
    assert film.writers_qids == ["Q100", "Q9"]  # sorted, deduplicated
    assert film.source_sitelinks == 75  # largest across sources
    assert film.source_types == {"Q1", "Q2"}
    # The 2009 figure has an unknown currency and is ignored, so 2008 wins.
    assert pick_worldwide_usd(film.box_office, FX) == (100_000_000.0, 2008)
    # Of the 2005 figures: EUR 800M at 0.5 EUR/USD = 1600M USD beats USD 90M (ties take the largest).
    older = [m for m in film.box_office if m.year == 2005]
    assert pick_worldwide_usd(older, FX) == (1_600_000_000.0, 2005)


def test_unknown_currency_keeps_its_unit_qid_and_no_unit_gives_none():
    result = sparql(
        money_row("1", "Q10", "cost", 1_000_000, unit="Q99999999"),
        row("1", "Q10", "cost", v=lit(2_000_000)),  # no unit at all
    )
    cost = parse_films(result)[1].cost
    assert Money(1_000_000.0, "Q99999999", None, None) in cost
    assert Money(2_000_000.0, None, None, None) in cost
    assert pick_worldwide_usd(cost, FX) == (None, None)


@pytest.mark.parametrize(
    "qid, code",
    [
        ("Q4917", "USD"),
        ("Q4916", "EUR"),
        ("Q25224", "GBP"),
        ("Q8146", "JPY"),
        ("Q80524", "INR"),
        ("Q39099", "CNY"),
        ("Q202040", "KRW"),
        ("Q259502", "AUD"),
        ("Q1104069", "CAD"),
        ("Q184172", "FRF"),
        ("Q16068", "DEM"),
        ("Q204992", "ITL"),
        ("Q189097", "ESP"),
    ],
)
def test_currency_units_map_to_iso_codes(qid, code):
    films = parse_films(sparql(money_row("1", "Q10", "cost", 5_000_000, unit=qid)))
    assert films[1].cost == [Money(5_000_000.0, code, None, None)]


def test_currency_table_matches_the_fx_file_currencies():
    fx = load_fx(Path(wikidata.__file__).parents[2] / "data" / "fx_usd_annual.csv")
    codes = {currency for currency, _ in fx}
    assert set(wikidata.CURRENCY_BY_UNIT.values()) == codes | {"USD"}


def test_undated_usd_ranks_below_any_dated_and_undated_non_usd_is_dropped():
    values = [
        Money(900_000_000.0, "USD", None, None),
        Money(50_000_000.0, "USD", WORLDWIDE_QID, 1999),
        Money(2_000_000_000.0, "EUR", None, None),  # cannot be converted without a year
    ]
    assert pick_worldwide_usd(values, FX) == (50_000_000.0, 1999)
    assert pick_worldwide_usd(values[::-1], FX) == (50_000_000.0, 1999)
    assert pick_worldwide_usd([values[0], values[2]], FX) == (900_000_000.0, None)
    assert pick_worldwide_usd([values[2]], FX) == (None, None)


def test_pick_worldwide_ties_take_the_largest_and_never_crash():
    tie = [Money(10_000_000.0, "USD", None, 2015), Money(30_000_000.0, "USD", None, 2015), Money(20_000_000.0, "USD", None, 2015)]
    assert pick_worldwide_usd(tie, FX) == (30_000_000.0, 2015)
    assert pick_worldwide_usd([], FX) == (None, None)
    assert pick_worldwide_usd(None, FX) == (None, None)
    junk = [
        None,
        Money(None, "USD", None, 2015),
        Money(math.nan, "USD", None, 2015),
        Money("lots", "USD", None, 2015),
        Money(-5.0, "USD", None, 2015),
        Money(0.0, "USD", None, 2015),
        Money(7_000_000.0, None, None, 2015),
        Money(7_000_000.0, "JPY", None, 1900),  # no rate for that year
    ]
    assert pick_worldwide_usd(junk, FX) == (None, None)
    assert pick_worldwide_usd(junk + [Money(1_000_000.0, "USD", None, 2001)], FX) == (1_000_000.0, 2001)


def test_placeholder_money_below_min_is_ignored_even_when_latest():
    older = Money(5_000_000.0, "USD", None, 2010)
    assert pick_worldwide_usd([older, Money(500.0, "USD", None, 2021)], FX) == (5_000_000.0, 2010)
    assert pick_worldwide_usd([Money(500.0, "USD", None, 2021)], FX) == (None, None)
    # the check is in USD after conversion: EUR 600 at 0.5 EUR/USD = 1200 USD is kept, EUR 400 = 800 USD is not
    assert pick_worldwide_usd([Money(600.0, "EUR", None, 2005)], FX) == (1200.0, 2005)
    assert pick_worldwide_usd([older, Money(400.0, "EUR", None, 2005)], FX) == (5_000_000.0, 2010)
    # the boundary: MIN_MONEY = 1,000 USD itself is a real figure, one cent below is a placeholder
    assert pick_worldwide_usd([Money(1000.0, "USD", None, 2021)], FX) == (1000.0, 2021)
    assert pick_worldwide_usd([Money(999.99, "USD", None, 2021)], FX) == (None, None)
    assert pick_worldwide_usd([Money(500.0, "USD", None, None)], FX) == (None, None)  # undated placeholder too
    assert pick_worldwide_usd([Money(500.0, "USD", WORLDWIDE_QID, 2021)], FX) == (None, None)  # explicit worldwide too


def test_year_comes_from_date_precision():
    result = sparql(
        money_row("1", "Q10", "cost", 1_000_000, when="2005-01-01T00:00:00Z", prec="9"),
        money_row("1", "Q10", "cost", 2_000_000, when="1990-01-01T00:00:00Z", prec="8"),  # decade: no usable year
        money_row("1", "Q10", "cost", 3_000_000, when="-0100-01-01T00:00:00Z", prec="9"),  # BCE
    )
    cost = parse_films(result)[1].cost
    assert Money(1_000_000.0, "USD", None, 2005) in cost
    assert Money(2_000_000.0, "USD", None, None) in cost
    assert Money(3_000_000.0, "USD", None, None) in cost


def test_duplicate_tmdb_id_uses_the_lowest_numeric_qid_and_merges_nothing():
    result = sparql(
        row("1", "Q100", "distributor", v=lit("From Q100")),
        row("1", "Q100", "item"),
        row("1", "Q9", "item"),
        row("1", "Q9", "distributor", v=lit("From Q9")),
        money_row("1", "Q100", "cost", 9_000_000),
    )
    film = parse_films(result)[1]
    assert film.qid == "Q9"  # numeric, not lexical (lexically "Q100" < "Q9")
    assert film.distributors == ["From Q9"]
    assert film.cost == []


def test_bad_film_values_never_crash():
    bnode = {"type": "bnode", "value": "t12"}
    genid = {"type": "uri", "value": "http://www.wikidata.org/.well-known/genid/abc123"}  # "unknown value"
    result = sparql(
        row("1", "Q10", "item"),
        row("1", "Q10", "writer", v=bnode),
        row("1", "Q10", "writer", v=genid),
        row("1", "Q10", "writer", v=lit("not a qid")),
        row("1", "Q10", "series", v=genid),
        row("1", "Q10", "source", n=lit("many"), v=genid),
        money_row("1", "Q10", "box_office", "twelve", place=WORLDWIDE_QID),
        money_row("1", "Q10", "box_office", "nan", place=WORLDWIDE_QID),
        money_row("1", "Q10", "box_office", 1_000_000, unit="Q4917", place=WORLDWIDE_QID, when="garbage"),
        row("x", "Q11", "item"),  # tmdb id that is not a number
        row("2", "not-an-item", "item"),
        {"kind": lit("item")},  # missing columns
        row("3", "Q12", "mystery_kind", v=uri("Q1")),
    )
    films = parse_films(result)
    assert set(films) == {1, 3}
    film = films[1]
    assert film.qid == "Q10"
    assert film.writers_qids == []
    assert film.series_qid is None
    assert film.source_types == set()
    assert film.source_sitelinks is None
    assert film.box_office == [Money(1_000_000.0, "USD", WORLDWIDE_QID, None)]
    assert parse_films(sparql()) == {}
    assert parse_films({}) == {}


# ---- people ----


def test_parse_people_awards_with_dates():
    people = parse_people(WD_PEOPLE)
    assert people[PHOENIX].birth == date(1974, 10, 28)
    assert people[LEDGER].birth == date(1979, 4, 4)
    assert people[NOLAN].birth == date(1970, 7, 30)
    # wikidata.org: Q103916 = Academy Award for Best Actor. Phoenix won it on 2020-02-09 (day precision).
    assert AwardEvent("Q103916", "won", date(2020, 2, 9)) in people[PHOENIX].awards
    # He was nominated for it on 2020-01-13, 2013-01-10 and 2006-01-31.
    nominated = {a.when for a in people[PHOENIX].awards if a.award_qid == "Q103916" and a.kind == "nominated"}
    assert nominated == {date(2020, 1, 13), date(2013, 1, 10), date(2006, 1, 31)}
    # Golden Globe (Q181883) 2006 is dated to the year only: it counts from the last day of that year.
    assert AwardEvent("Q181883", "won", date(2006, 12, 31)) in people[PHOENIX].awards
    # Q106291 = Academy Award for Best Supporting Actor, won by Ledger on 2009-02-22.
    assert AwardEvent("Q106291", "won", date(2009, 2, 22)) in people[LEDGER].awards
    # An award with no date at all is kept with when=None (never counted).
    assert AwardEvent("Q1260789", "won", None) in people[LEDGER].awards
    assert people[PHOENIX].awards == sorted(set(people[PHOENIX].awards), key=lambda a: (a.award_qid, a.kind, a.when or date.min))


@pytest.mark.parametrize(
    "when, prec, expected",
    [
        ("2006-01-01T00:00:00Z", "9", date(2006, 12, 31)),  # year precision: last day of the year
        ("2024-02-01T00:00:00Z", "10", date(2024, 2, 29)),  # month precision: last day of a leap February
        ("2023-02-01T00:00:00Z", "10", date(2023, 2, 28)),
        ("2023-12-01T00:00:00Z", "10", date(2023, 12, 31)),
        ("2020-02-09T00:00:00Z", "11", date(2020, 2, 9)),  # day precision: the day itself
        ("2020-02-09T00:00:00Z", "12", date(2020, 2, 9)),  # finer than a day: the day itself
        ("2020-02-09T00:00:00Z", "14", date(2020, 2, 9)),
        ("2010-01-01T00:00:00Z", "8", None),  # decade: not counted
        ("2000-01-01T00:00:00Z", "7", None),  # century
        ("2020-02-09T00:00:00Z", None, None),  # precision unknown: not counted
        ("-0100-01-01T00:00:00Z", "9", None),  # BCE
        ("garbage", "11", None),
    ],
)
def test_award_date_is_the_last_day_an_imprecise_date_could_mean(when, prec, expected):
    cols = {"v": uri("Q103916"), "date": lit(when)}
    if prec is not None:
        cols["prec"] = lit(prec)
    people = parse_people(sparql(row("5", "Q50", "item"), row("5", "Q50", "won", **cols)))
    assert people[5].awards == [AwardEvent("Q103916", "won", expected)]


def test_bad_people_values_never_crash():
    bnode = {"type": "bnode", "value": "t0"}
    genid = {"type": "uri", "value": "http://www.wikidata.org/.well-known/genid/zzz"}
    result = sparql(
        row("5", "Q50", "item"),
        row("5", "Q50", "birth", v=lit("-0450-03-04T00:00:00Z")),  # BCE
        row("5", "Q50", "won", v=bnode),
        row("5", "Q50", "nominated", v=genid, date=lit("2001-01-01T00:00:00Z"), prec=lit("11")),
        row("6", "Q51", "item"),
        row("6", "Q51", "birth", v=lit("oops")),
        row("7", "Q52", "item"),
        row("7", "Q52", "birth", v=genid),
        row("8", "Q53", "item"),
        row("8", "Q53", "birth", v=lit("1970-00-00T00:00:00Z")),  # month and day zero: unusable as given
        row("9", "Q54", "item"),
        row("9", "Q54", "birth", v=lit("1970-05-17T00:00:00Z")),
        row("9", "Q54", "won", v=uri("Q103916"), date=lit("2001-01-01T00:00:00Z"), prec=lit("many")),
    )
    people = parse_people(result)
    assert people[5] == PersonFacts(None, [])
    assert people[6].birth is None and people[7].birth is None and people[8].birth is None
    assert people[9].birth == date(1970, 5, 17)
    assert people[9].awards == [AwardEvent("Q103916", "won", None)]


# ---- to_dict / from_dict ----


def test_round_trip_through_json_equals_the_original():
    for film in parse_films(WD_FILMS).values():
        assert FilmFacts.from_dict(json.loads(json.dumps(film.to_dict()))) == film
    for person in parse_people(WD_PEOPLE).values():
        assert PersonFacts.from_dict(json.loads(json.dumps(person.to_dict()))) == person
    # a record with nothing set also survives
    empty = FilmFacts("Q1", set(), None, [], [], [], [], None, None, [], [])
    assert FilmFacts.from_dict(json.loads(json.dumps(empty.to_dict()))) == empty
    assert PersonFacts.from_dict(json.loads(json.dumps(PersonFacts(None, []).to_dict()))) == PersonFacts(None, [])


def test_to_dict_is_json_safe_and_sorted():
    joker = parse_films(WD_FILMS)[JOKER].to_dict()
    json.dumps(joker)  # no sets, no dates
    assert joker["source_types"] == sorted(joker["source_types"])
    phoenix = parse_people(WD_PEOPLE)[PHOENIX].to_dict()
    assert phoenix["birth"] == "1974-10-28"
    assert {"award_qid": "Q103916", "kind": "won", "when": "2020-02-09"} in phoenix["awards"]


# ---- SparqlClient ----


class FakeResponse:
    def __init__(self, status: int, body=None, headers: dict | None = None):
        self.status_code = status
        self._body = body
        self.headers = headers or {}

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class FakeSession:
    def __init__(self, script: list):
        self.script = list(script)
        self.calls: list[dict] = []

    def post(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def sleeps(monkeypatch):
    recorded: list[float] = []
    monkeypatch.setattr("time.sleep", recorded.append)
    return recorded


OK = {"results": {"bindings": []}}


def test_sparql_request_shape(sleeps):
    session = FakeSession([FakeResponse(200, OK)])
    assert SparqlClient(session=session).run("SELECT 1") == OK
    call = session.calls[0]
    assert call["url"] == wikidata.ENDPOINT
    assert call["data"] == {"query": "SELECT 1"}  # POSTed, never in the URL
    assert call["headers"]["User-Agent"] == USER_AGENT
    assert call["headers"]["Accept"] == "application/sparql-results+json"
    assert call["timeout"] >= 70  # WDQS's own limit is 60 s
    # No email anywhere in what is sent: headers (any, e.g. From), body, URL.
    assert "@" not in repr([call["url"], call["data"], call["headers"]])
    assert set(call["headers"]) == {"User-Agent", "Accept"}
    assert sleeps == []


def test_sparql_retry_honours_retry_after_in_full(sleeps):
    session = FakeSession([FakeResponse(429, headers={"Retry-After": "300"}), FakeResponse(503), FakeResponse(200, OK)])
    assert SparqlClient(session=session).run("q") == OK
    assert len(session.calls) == 3
    assert sleeps[0] == 300  # beyond the backoff cap
    assert sleeps[1] == 2.0**1  # plain backoff for the second attempt


def test_sparql_backoff_is_capped(sleeps):
    session = FakeSession([FakeResponse(500)] * 8)
    with pytest.raises(SparqlError):
        SparqlClient(session=session, max_retries=7).run("q")
    assert max(sleeps) == wikidata.MAX_BACKOFF
    assert len(session.calls) == 8


def test_sparql_retries_connection_errors_then_raises(sleeps):
    session = FakeSession([requests.ConnectionError("down"), requests.Timeout("slow"), FakeResponse(200, OK)])
    assert SparqlClient(session=session).run("q") == OK
    session = FakeSession([requests.ConnectionError("down")] * 3)
    with pytest.raises(requests.ConnectionError):
        SparqlClient(session=session, max_retries=2).run("q")
    assert len(session.calls) == 3


def test_sparql_other_4xx_raise_at_once(sleeps):
    session = FakeSession([FakeResponse(400)])
    with pytest.raises(SparqlError, match="400"):
        SparqlClient(session=session).run("q")
    assert len(session.calls) == 1 and sleeps == []


def test_sparql_gives_up_after_the_retries(sleeps):
    session = FakeSession([FakeResponse(503)] * 3)
    with pytest.raises(SparqlError, match="503"):
        SparqlClient(session=session, max_retries=2).run("q")
    assert len(session.calls) == 3


def test_sparql_body_without_results_is_retried_then_raises(sleeps):
    session = FakeSession(
        [FakeResponse(200, {}), FakeResponse(200, {"error": "x"}), FakeResponse(200, {"results": {}}), FakeResponse(200, OK)]
    )
    assert SparqlClient(session=session).run("q") == OK
    assert len(session.calls) == 4
    session = FakeSession([FakeResponse(200, {})] * 3)
    with pytest.raises(SparqlError) as error:
        SparqlClient(session=session, max_retries=2).run("q")
    assert error.value.retryable and len(session.calls) == 3


def test_a_bad_200_body_writes_no_record_for_its_ids(tmp_path, sleeps):
    for n, body in enumerate(({}, {"results": {"bindings": None}}, {"results": {"bindings": {}}}, {"results": []}, [1])):
        out = tmp_path / str(n)
        client = SparqlClient(session=FakeSession([FakeResponse(200, body)] * 50), max_retries=1)
        with pytest.raises(SparqlError):
            run_pull("films", [1, 2, 3], out, client, batch=3)
        shard = out / "wd_films.jsonl"
        assert not shard.exists() or done_keys(shard, "tmdb_id") == set()


def test_sparql_malformed_body_is_retried(sleeps):
    session = FakeSession([FakeResponse(200, ValueError("bad json")), FakeResponse(200, OK)])
    assert SparqlClient(session=session).run("q") == OK


# ---- the pull: batches, halving, resume ----


class FakeWikidata:
    """A client whose run() answers a films or people query from hand-built rows. `fail_over` makes any query
    for more than that many ids fail, as WDQS does when a batch is too heavy."""

    def __init__(self, films: dict[int, str | None], fail_over: int | None = None, fail_ids: set[int] = frozenset()):
        self.items = films  # tmdb id -> QID, or None for "no Wikidata item"
        self.fail_over = fail_over
        self.fail_ids = fail_ids
        self.queries: list[list[int]] = []

    def run(self, query: str) -> dict:
        import re

        ids = [int(i) for i in re.search(r"VALUES \?tmdb \{([^}]*)\}", query).group(1).replace('"', "").split()]
        self.queries.append(ids)
        if (self.fail_over is not None and len(ids) > self.fail_over) or self.fail_ids & set(ids):
            raise SparqlError("WDQS returned HTTP 500", retryable=True)
        rows = []
        for i in ids:
            if self.items.get(i):
                rows.append(row(str(i), self.items[i], "item"))
                rows.append(row(str(i), self.items[i], "distributor", v=lit(f"Studio {i}")))
        return sparql(*rows)


def read_records(path):
    return list(read_shard(path))


def test_run_pull_writes_one_record_per_id_including_no_item(tmp_path, sleeps):
    client = FakeWikidata({1: "Q1", 2: None, 3: "Q3"})
    stats = run_pull("films", [1, 2, 3], tmp_path, client, batch=2, pause=1.0)
    records = read_records(tmp_path / "wd_films.jsonl")
    assert [r["tmdb_id"] for r in records] == [1, 2, 3]
    assert records[0]["facts"]["qid"] == "Q1" and records[0]["facts"]["distributors"] == ["Studio 1"]
    assert records[1] == {"tmdb_id": 2, "facts": None}  # remembered, so resume does not ask again
    assert records[2]["facts"]["qid"] == "Q3"
    assert client.queries == [[1, 2], [3]]
    assert stats == {"queried": 3, "skipped": 0}
    assert sleeps == [1.0]  # between the two batches only


def test_run_pull_resumes_without_asking_again(tmp_path, sleeps):
    first = FakeWikidata({1: "Q1", 2: None, 3: "Q3"})
    run_pull("films", [1, 2], tmp_path, first, batch=10)
    second = FakeWikidata({1: "Q1", 2: None, 3: "Q3"})
    stats = run_pull("films", [1, 2, 3], tmp_path, second, batch=10)
    assert second.queries == [[3]]
    assert stats == {"queried": 1, "skipped": 2}
    assert done_keys(tmp_path / "wd_films.jsonl", "tmdb_id") == {1, 2, 3}
    third = FakeWikidata({})
    assert run_pull("films", [1, 2, 3], tmp_path, third, batch=10) == {"queried": 0, "skipped": 3}
    assert third.queries == []
    assert len(read_records(tmp_path / "wd_films.jsonl")) == 3


def test_run_pull_dedupes_ids_and_ignores_unrequested_results(tmp_path, sleeps):
    client = FakeWikidata({1: "Q1"})
    run_pull("films", [1, 1, 1], tmp_path, client, batch=10)
    assert client.queries == [[1]]
    assert len(read_records(tmp_path / "wd_films.jsonl")) == 1


def test_a_failing_batch_is_halved_down_to_single_ids(tmp_path, sleeps):
    client = FakeWikidata({i: f"Q{i}" for i in range(1, 6)}, fail_over=1)
    run_pull("films", [1, 2, 3, 4, 5], tmp_path, client, batch=5)
    assert [r["tmdb_id"] for r in read_records(tmp_path / "wd_films.jsonl")] == [1, 2, 3, 4, 5]
    assert client.queries[0] == [1, 2, 3, 4, 5]
    assert client.queries[1] == [1, 2, 3]  # halves: first 3, then last 2
    assert [q for q in client.queries if len(q) == 1]  # went down to single ids


def test_a_single_id_that_still_fails_raises_and_keeps_earlier_work(tmp_path, sleeps):
    client = FakeWikidata({i: f"Q{i}" for i in range(1, 5)}, fail_ids={3})
    with pytest.raises(SparqlError):
        run_pull("films", [1, 2, 3, 4], tmp_path, client, batch=4)
    # The sub-batch that succeeded before the failure is kept; the resume asks only for the rest.
    assert done_keys(tmp_path / "wd_films.jsonl", "tmdb_id") == {1, 2}
    client2 = FakeWikidata({i: f"Q{i}" for i in range(1, 5)})
    run_pull("films", [1, 2, 3, 4], tmp_path, client2, batch=4)
    assert client2.queries == [[3, 4]]
    assert done_keys(tmp_path / "wd_films.jsonl", "tmdb_id") == {1, 2, 3, 4}


def test_halving_does_not_retry_a_non_retryable_error(tmp_path, sleeps):
    class Rejecting:
        calls = 0

        def run(self, query):
            Rejecting.calls += 1
            raise SparqlError("WDQS returned HTTP 400", retryable=False)

    with pytest.raises(SparqlError):
        run_pull("films", [1, 2, 3, 4], tmp_path, Rejecting(), batch=4)
    assert Rejecting.calls == 1


def test_people_pull_uses_person_ids_and_its_own_shard(tmp_path, sleeps):
    class PeopleClient:
        def run(self, query):
            assert "P4985" in query
            return sparql(row("77", "Q7", "item"), row("77", "Q7", "birth", v=lit("1980-01-02T00:00:00Z")))

    run_pull("people", [77, 78], tmp_path, PeopleClient(), batch=10)
    records = read_records(tmp_path / "wd_people.jsonl")
    assert records[0] == {"tmdb_id": 77, "facts": {"birth": "1980-01-02", "awards": []}}
    assert records[1] == {"tmdb_id": 78, "facts": None}
    assert not (tmp_path / "wd_films.jsonl").exists()


def test_cli_reads_text_and_gzip_id_files(tmp_path, monkeypatch, sleeps):
    clients = []

    class Recording(FakeWikidata):
        def __init__(self):
            super().__init__({i: f"Q{i}" for i in (3924, 8773, 25449, 7, 8)})
            clients.append(self)

    monkeypatch.setattr(wikidata, "SparqlClient", Recording)
    wikidata.main(["films", "--ids", str(SAMPLE_IDS), "--out", str(tmp_path / "gz")])
    # a .json.gz goes through read_ids: adult and video entries dropped
    expected = read_ids(SAMPLE_IDS)
    assert expected[:3] == [3924, 8773, 25449] and 900001 not in expected and 31975 not in expected
    assert clients[0].queries == [expected]
    assert [r["tmdb_id"] for r in read_records(tmp_path / "gz" / "wd_films.jsonl")] == expected

    ids = tmp_path / "ids.txt"
    ids.write_text("7\n\n8\n7\n", encoding="utf-8")
    wikidata.main(["films", "--ids", str(ids), "--out", str(tmp_path / "txt")])
    assert clients[1].queries == [[7, 8]]
    wikidata.main(["films", "--ids", str(ids), "--out", str(tmp_path / "txt")])  # rerun: nothing to ask
    assert clients[2].queries == []


# ---- syncing the shard to the Hub during the pull ----


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class SyncRecorder:
    """sync() copies the shard's current line count, as an upload would snapshot it."""

    def __init__(self, shard, clock=None, advance=0.0, fail_on=None):
        self.shard, self.clock, self.advance, self.fail_on = shard, clock, advance, fail_on
        self.snapshots: list[int] = []

    def __call__(self):
        if self.fail_on == len(self.snapshots) + 1:
            raise RuntimeError("hub down")
        self.snapshots.append(len(self.shard.read_text(encoding="utf-8").splitlines()))


class SlowClient(FakeWikidata):
    """Every query takes `seconds` on the injected clock."""

    def __init__(self, items, clock, seconds):
        super().__init__(items)
        self.clock, self.seconds = clock, seconds

    def run(self, query):
        self.clock.now += self.seconds
        return super().run(query)


def test_sync_runs_every_interval_and_once_at_the_end(tmp_path, sleeps):
    clock = Clock()
    shard = tmp_path / "wd_films.jsonl"
    sync = SyncRecorder(shard)
    client = SlowClient({i: f"Q{i}" for i in range(1, 7)}, clock, seconds=400)
    run_pull("films", [1, 2, 3, 4, 5, 6], tmp_path, client, batch=1, sync=sync, sync_every=600, clock=clock)
    # 400 s per batch: syncs after batches 2 (800 s), 4 (1600 s... counted from the last sync) and 6, then the final one
    assert sync.snapshots == [2, 4, 6, 6]


def test_the_final_sync_happens_even_when_the_last_batch_ended_between_syncs(tmp_path, sleeps):
    clock = Clock()
    sync = SyncRecorder(tmp_path / "wd_films.jsonl")
    client = SlowClient({1: "Q1", 2: "Q2", 3: "Q3"}, clock, seconds=10)
    run_pull("films", [1, 2, 3], tmp_path, client, batch=1, sync=sync, sync_every=600, clock=clock)
    assert sync.snapshots == [3]  # no periodic sync (30 s run), one final upload with everything


def test_no_sync_given_means_no_upload_and_the_old_behaviour(tmp_path, sleeps):
    stats = run_pull("films", [1], tmp_path, FakeWikidata({1: "Q1"}))
    assert stats == {"queried": 1, "skipped": 0}


def test_a_failed_sync_raises_and_keeps_the_records_written_so_far(tmp_path, sleeps):
    clock = Clock()
    shard = tmp_path / "wd_films.jsonl"
    sync = SyncRecorder(shard, fail_on=1)
    client = SlowClient({i: f"Q{i}" for i in range(1, 5)}, clock, seconds=400)
    with pytest.raises(RuntimeError, match="hub down"):
        run_pull("films", [1, 2, 3, 4], tmp_path, client, batch=1, sync=sync, sync_every=600, clock=clock)
    assert done_keys(shard, "tmdb_id") == {1, 2}  # stopped at the first failed sync, never swallowed


def test_a_failed_final_sync_raises(tmp_path, sleeps):
    sync = SyncRecorder(tmp_path / "wd_films.jsonl", fail_on=1)
    with pytest.raises(RuntimeError, match="hub down"):
        run_pull("films", [1], tmp_path, FakeWikidata({1: "Q1"}), sync=sync, clock=Clock())


def test_resume_after_a_crash_asks_only_for_the_missing_ids(tmp_path, sleeps):
    # session 1 syncs, then dies on id 3; the Hub holds what was synced (ids 1 and 2)
    clock = Clock()
    hub = {}
    shard = tmp_path / "s1" / "wd_films.jsonl"
    client = SlowClient({1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4"}, clock, seconds=400)
    original = client.run

    def dying(query):
        if '"3"' in re.search(r"VALUES \?tmdb \{([^}]*)\}", query).group(1):
            raise SparqlError("WDQS returned HTTP 400")  # not retryable: the run stops
        return original(query)

    client.run = dying

    def sync():
        hub["wd_films.jsonl"] = shard.read_bytes()

    with pytest.raises(SparqlError):
        run_pull("films", [1, 2, 3, 4], shard.parent, client, batch=1, sync=sync, sync_every=600, clock=clock)
    # session 2: empty disk, download the shard, run again
    second = tmp_path / "s2"
    second.mkdir()
    (second / "wd_films.jsonl").write_bytes(hub["wd_films.jsonl"])
    client2 = FakeWikidata({i: f"Q{i}" for i in range(1, 5)})
    run_pull("films", [1, 2, 3, 4], second, client2, batch=1, clock=Clock())
    assert client2.queries == [[3], [4]]
    assert [r["tmdb_id"] for r in read_records(second / "wd_films.jsonl")] == [1, 2, 3, 4]


def test_cli_repo_syncs_through_hub_store(tmp_path, monkeypatch, sleeps):
    from llm_movies.data import hub_store

    uploads, seen_every = [], []
    real_run_pull = wikidata.run_pull

    def spying_run_pull(*args, **kwargs):
        seen_every.append(kwargs["sync_every"])
        return real_run_pull(*args, **kwargs)

    monkeypatch.setattr(wikidata, "run_pull", spying_run_pull)
    monkeypatch.setattr(wikidata, "SparqlClient", lambda: FakeWikidata({7: "Q7", 8: "Q8"}))
    monkeypatch.setattr(hub_store, "make_api", lambda: "api")
    monkeypatch.setattr(hub_store, "upload", lambda folder, repo, path, **kw: uploads.append((folder, repo, path, kw)))
    ids = tmp_path / "ids.txt"
    ids.write_text("7\n8\n", encoding="utf-8")
    out = tmp_path / "wd_films"
    wikidata.main(["films", "--ids", str(ids), "--out", str(out), "--repo", "a/b", "--sync-minutes", "5"])
    assert uploads == [(out, "a/b", "wd_films", {"api": "api", "allow_patterns": ["wd_films.jsonl"]})]
    assert seen_every == [300.0]  # --sync-minutes 5 is 300 seconds
    wikidata.main(["people", "--ids", str(ids), "--out", str(tmp_path / "wd_people"), "--repo", "a/b"])
    assert uploads[1][2] == "wd_people" and uploads[1][3]["allow_patterns"] == ["wd_people.jsonl"]


def test_cli_sync_interval_defaults_to_30_minutes(tmp_path, monkeypatch):
    from llm_movies.data import hub_store

    seen = []
    monkeypatch.setattr(wikidata, "SparqlClient", lambda: None)
    monkeypatch.setattr(hub_store, "make_api", lambda: "api")
    monkeypatch.setattr(wikidata, "run_pull", lambda *a, **k: seen.append(k["sync_every"]) or {})
    ids = tmp_path / "ids.txt"
    ids.write_text("7\n", encoding="utf-8")
    wikidata.main(["films", "--ids", str(ids), "--out", str(tmp_path / "o"), "--repo", "a/b"])
    wikidata.main(["films", "--ids", str(ids), "--out", str(tmp_path / "o"), "--repo", "a/b", "--sync-minutes", "5"])
    assert seen == [1800.0, 300.0]
