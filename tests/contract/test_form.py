import itertools
import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from llm_movies.contract import form
from llm_movies.contract.form import CERTIFICATIONS, FIELDS, RELEASE_TYPES, describe, validate
from llm_movies.data.clean import _certification, release_type
from llm_movies.data.source_material import SOURCE_TYPES

THEMES_FILE = Path(form.__file__).with_name("themes.json")
REAL_THEMES = [t["name"] for t in json.loads(THEMES_FILE.read_text(encoding="utf-8"))["themes"]]
POST_RELEASE = ("vote_average", "vote_count", "popularity", "revenue")


def good_form(**overrides) -> dict:
    base = {
        "genres": ["Crime", "Thriller"], "release_year": 2027, "release_month": 7, "runtime": 118,
        "language": "en", "certification": "R", "director": "A Director",
        "actors": [f"Actor {i}" for i in range(5)], "themes": [REAL_THEMES[0]],
    }
    return {**base, **overrides}


def fields_in(errors) -> list[str]:
    return [e["field"] for e in errors]


def test_constants_match_spec_and_brief():
    assert form.GENRES == (
        "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", "Drama", "Family", "Fantasy", "History",
        "Horror", "Music", "Mystery", "Romance", "Science Fiction", "TV Movie", "Thriller", "War", "Western",
    )
    assert CERTIFICATIONS == ("G", "PG", "PG-13", "R", "NC-17", "unrated/unknown")
    assert form.UNRATED == "unrated/unknown"
    assert RELEASE_TYPES == ("wide", "limited", "festival-first", "streaming-only")
    assert SOURCE_TYPES == ("original", "novel", "comic", "true story", "remake", "video game", "play", "TV series", "manga")
    assert (form.MIN_YEAR, form.MAX_YEAR) == (1960, 2035)
    assert (form.MIN_RUNTIME, form.MAX_RUNTIME) == (40, 300)
    assert (form.MIN_BUDGET, form.MAX_BUDGET) == (1_000, 1_000_000_000)
    assert form.MAX_SYNOPSIS == 2_000
    assert (form.MIN_ACTORS, form.MAX_ACTORS) == (5, 10)
    assert form.MAX_TAGLINE == 150


def test_field_names_exactly_the_22_in_form_order():
    assert [f.name for f in FIELDS] == [
        "genres", "release_year", "release_month", "runtime", "language", "certification", "director", "actors",
        "themes", "companies", "countries", "franchise", "synopsis", "budget",
        "writers", "producers", "composer", "distributor", "source_material", "release_day", "release_type", "tagline",
    ]


def test_minimal_valid_form():
    clean, errors = validate(good_form())
    assert errors == []
    assert clean["director"] == {"id": None, "name": "A Director"}
    assert clean["companies"] == [] and clean["writers"] == [] and clean["producers"] == []
    assert clean["composer"] is None and clean["distributor"] is None and clean["source_material"] is None
    assert clean["release_day"] is None and clean["tagline"] is None and clean["budget"] is None
    assert set(clean) == {f.name for f in FIELDS}


def test_required_fields_missing():
    clean, errors = validate({})
    assert sorted(fields_in(errors)) == sorted(f.name for f in FIELDS if f.required)
    assert all(e["message"] == "required" for e in errors)
    assert clean["release_type"] == "wide"


def test_release_day_31_in_april_rejected():
    _, errors = validate(good_form(release_month=4, release_day=31))
    assert fields_in(errors) == ["release_day"]
    _, errors = validate(good_form(release_month=4, release_day=30))
    assert errors == []


def test_release_day_leap_year():
    _, errors = validate(good_form(release_year=2024, release_month=2, release_day=29))
    assert errors == []
    _, errors = validate(good_form(release_year=2023, release_month=2, release_day=29))
    assert fields_in(errors) == ["release_day"]


def test_release_day_with_bad_month_gives_only_the_month_error():
    _, errors = validate(good_form(release_month=13, release_day=31))
    assert fields_in(errors) == ["release_month"]
    _, errors = validate(good_form(release_year=1900, release_day=31))
    assert fields_in(errors) == ["release_year"]


@pytest.mark.parametrize("day", [0, 32, 1.5, "5", True])
def test_release_day_out_of_range_or_wrong_type(day):
    _, errors = validate(good_form(release_day=day))
    assert fields_in(errors) == ["release_day"]


def test_writers_max_three():
    names = [f"W{i}" for i in range(4)]
    clean, errors = validate(good_form(writers=names[:3]))
    assert errors == [] and len(clean["writers"]) == 3
    _, errors = validate(good_form(writers=names))
    assert fields_in(errors) == ["writers"]


def test_producers_max_three():
    names = [f"P{i}" for i in range(4)]
    _, errors = validate(good_form(producers=names[:3]))
    assert errors == []
    _, errors = validate(good_form(producers=names))
    assert fields_in(errors) == ["producers"]


@pytest.mark.parametrize("field, value, ok", [
    # limits are literals from the brief / spec 4.1, both sides of each edge
    ("release_year", 1959, False), ("release_year", 1960, True), ("release_year", 2035, True), ("release_year", 2036, False),
    ("runtime", 39, False), ("runtime", 40, True), ("runtime", 300, True), ("runtime", 301, False),
    ("budget", 999, False), ("budget", 1_000, True), ("budget", 1_000_000_000, True), ("budget", 1_000_000_001, False),
    ("synopsis", "x" * 2000, True), ("synopsis", "x" * 2001, False),
    ("release_day", 1, True), ("release_day", 31, True),  # July has 31 days
    ("genres", ["Action", "Adventure", "Comedy", "Crime", "Drama"], True),
    ("genres", ["Action", "Adventure", "Comedy", "Crime", "Drama", "Family"], False),
    ("companies", ["A", "B", "C", "D", "E"], True), ("companies", ["A", "B", "C", "D", "E", "F"], False),
    ("countries", ["US", "GB", "FR", "DE", "IT"], True), ("countries", ["US", "GB", "FR", "DE", "IT", "ES"], False),
])
def test_range_and_count_edges(field, value, ok):
    _, errors = validate(good_form(**{field: value}))
    assert (errors == []) is ok
    if not ok:
        assert fields_in(errors) == [field]


@pytest.mark.parametrize("count, ok", [(4, False), (5, True), (10, True), (11, False)])
def test_actors_boundaries(count, ok):
    _, errors = validate(good_form(actors=[f"A{i}" for i in range(count)]))
    assert (errors == []) is ok


@pytest.mark.parametrize("count, ok", [(0, False), (1, True), (5, True), (6, False)])
def test_theme_count_boundaries(count, ok):
    _, errors = validate(good_form(themes=REAL_THEMES[:count]))
    assert (errors == []) is ok


def test_theme_must_come_from_themes_json():
    _, errors = validate(good_form(themes=["not a theme"]))
    assert fields_in(errors) == ["themes"]
    _, errors = validate(good_form(themes=[REAL_THEMES[-1]]))
    assert errors == []


def test_tagline_over_150_rejected():
    clean, errors = validate(good_form(tagline="x" * 150))
    assert errors == [] and clean["tagline"] == "x" * 150
    _, errors = validate(good_form(tagline="x" * 151))
    assert fields_in(errors) == ["tagline"]


def test_release_type_defaults_to_wide():
    clean, errors = validate(good_form())
    assert errors == [] and clean["release_type"] == "wide"
    clean, errors = validate(good_form(release_type="limited"))
    assert errors == [] and clean["release_type"] == "limited"


@pytest.mark.parametrize("value", ["Wide", "imax", 3, ["wide"]])
def test_invalid_release_type_is_an_error(value):
    _, errors = validate(good_form(release_type=value))
    assert fields_in(errors) == ["release_type"]


def test_clean_release_types_are_all_allowed():
    """Every release type clean.py can produce (training data) is a valid form value."""
    produced = set()
    for r in range(1, 7):
        for combo in itertools.combinations(range(1, 7), r):
            for gap in (0, 29, 30, 90):
                entries = [(date(2020, 1, 1) + timedelta(days=gap if k != 1 else 0), k, "") for k in combo]
                produced.add(release_type(entries))
    produced.add(release_type([]))
    produced.discard(None)
    assert produced == set(RELEASE_TYPES)


@pytest.mark.parametrize("raw, expected", [
    ("G", "G"), ("PG", "PG"), ("PG-13", "PG-13"), ("R", "R"), ("NC-17", "NC-17"),
    ("NR", "unrated/unknown"), ("X", "unrated/unknown"), ("", "unrated/unknown"),
])
def test_clean_certifications_are_valid_form_values(raw, expected):
    """What clean.py stores as a certification is accepted by the form (training and serving agree)."""
    produced = _certification({"US": [(date(2020, 1, 1), 3, raw)]})
    assert produced == expected
    _, errors = validate(good_form(certification=produced))
    assert errors == []


def test_source_material_choice():
    clean, errors = validate(good_form(source_material="manga"))
    assert errors == [] and clean["source_material"] == "manga"
    _, errors = validate(good_form(source_material="rumor"))
    assert fields_in(errors) == ["source_material"]


def test_unknown_field_rejected():
    clean, errors = validate(good_form(colour="red"))
    assert errors == [{"field": "colour", "message": "unknown field"}]
    assert "colour" not in clean


@pytest.mark.parametrize("name", POST_RELEASE)
def test_post_release_fields_rejected(name):
    clean, errors = validate(good_form(**{name: 5}))
    assert errors == [{"field": name, "message": "unknown field"}]
    assert name not in clean
    assert name not in {f.name for f in FIELDS}


def test_describe_has_no_post_release_name():
    text = json.dumps(describe())
    assert not any(name in text for name in POST_RELEASE)
    assert [d["name"] for d in describe()] == [f.name for f in FIELDS]


@pytest.mark.parametrize("field, value", [
    ("release_year", True), ("runtime", True), ("budget", True), ("release_month", False), ("distributor", False),
    ("director", True), ("composer", True), ("franchise", True),
])
def test_bools_rejected_as_numbers_and_refs(field, value):
    _, errors = validate(good_form(**{field: value}))
    assert fields_in(errors) == [field]


def test_bools_rejected_inside_ref_lists_and_ids():
    _, errors = validate(good_form(writers=[True]))
    assert fields_in(errors) == ["writers"]
    _, errors = validate(good_form(director={"id": True}))
    assert fields_in(errors) == ["director"]


@pytest.mark.parametrize("field", ["release_year", "release_month", "runtime", "budget", "release_day"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nan_and_infinity_rejected(field, value):
    _, errors = validate(good_form(**{field: value}))
    assert errors == [{"field": field, "message": "must be a number"}]


def test_person_ref_as_id_name_or_both():
    clean, errors = validate(good_form(director=7, composer="Hans"))
    assert errors == []
    assert clean["director"] == {"id": 7, "name": None}
    assert clean["composer"] == {"id": None, "name": "Hans"}
    clean, errors = validate(good_form(composer={"id": 9, "name": " Zimmer "}))
    assert errors == [] and clean["composer"] == {"id": 9, "name": "Zimmer"}
    _, errors = validate(good_form(director={"id": 0}))
    assert fields_in(errors) == ["director"]
    _, errors = validate(good_form(director={"id": form.MAX_REF_ID + 1}))
    assert fields_in(errors) == ["director"]
    _, errors = validate(good_form(director={"name": "  "}))
    assert fields_in(errors) == ["director"]


def test_composer_given_as_list_rejected():
    _, errors = validate(good_form(composer=["One"]))
    assert fields_in(errors) == ["composer"]
    _, errors = validate(good_form(composer=[{"id": 1}]))
    assert fields_in(errors) == ["composer"]


def test_distributor_is_a_trimmed_name():
    clean, errors = validate(good_form(distributor="  A24 "))
    assert errors == [] and clean["distributor"] == "A24"
    clean, errors = validate(good_form(distributor="d" * form.MAX_DISTRIBUTOR))
    assert errors == [] and len(clean["distributor"]) == 200
    _, errors = validate(good_form(distributor="d" * (form.MAX_DISTRIBUTOR + 1)))
    assert fields_in(errors) == ["distributor"]
    assert form.MAX_DISTRIBUTOR == 200


@pytest.mark.parametrize("value", [{"id": 9}, {"name": "A24"}, 9, ["A24"], 1.5])
def test_distributor_non_text_rejected(value):
    _, errors = validate(good_form(distributor=value))
    assert fields_in(errors) == ["distributor"]


@pytest.mark.parametrize("field", ["int_field", "float_field"])
@pytest.mark.parametrize("value", [10**400, -(10**400)])
def test_huge_integers_give_a_range_error_not_an_exception(field, value):
    name = {"int_field": "runtime", "float_field": "budget"}[field]
    clean, errors = validate(good_form(**{name: value}))
    assert len(errors) == 1 and errors[0]["field"] == name and errors[0]["message"].startswith("must be between")
    assert clean[name] is None


@pytest.mark.parametrize("field", ["tagline", "synopsis", "distributor"])
@pytest.mark.parametrize("blank", ["", "  ", "\t\n "])
def test_blank_optional_text_is_absent(field, blank):
    clean, errors = validate(good_form(**{field: blank}))
    assert errors == [] and clean[field] is None


@pytest.mark.parametrize("field", ["language", "certification", "director"])
def test_blank_required_string_is_required(field):
    _, errors = validate(good_form(**{field: "   "}))
    assert errors == [{"field": field, "message": "required"}]


def test_blank_name_in_a_dict_is_still_an_error():
    _, errors = validate(good_form(composer={"name": "  "}))
    assert errors == [{"field": "composer", "message": "needs an id or a name"}]


def test_errors_have_exactly_field_and_message():
    _, errors = validate({"nope": 1, "runtime": 5})
    assert errors
    assert all(set(e) == {"field", "message"} and all(isinstance(v, str) for v in e.values()) for e in errors)


@pytest.mark.parametrize("field, value", [
    ("genres", ["Cooking"]), ("release_year", 1950), ("release_month", 13), ("runtime", 118.5),
    ("language", "English"), ("certification", "NR"), ("actors", ["Only", "Four", "Actors", "Here"]),
    ("countries", ["usa"]), ("language", "en\n"), ("budget", 50), ("synopsis", "x" * 2001),
])
def test_bad_values_rejected(field, value):
    _, errors = validate(good_form(**{field: value}))
    assert fields_in(errors) == [field]
