import os
import re
from pathlib import Path

import pytest

from llm_movies.prompt.render import GENERIC_FRANCHISE, HIDEABLE, OPTIONAL_GROUPS, RANDOM_HIDE, money, render
from llm_movies.prompt.resolve import ResolvedFilm, film_row, film_to_form, form_as_of, resolve

GOLDEN = Path(__file__).parent.parent / "golden" / "prompts"
GOLDEN_FILMS = {"toy_story_4": 301528, "toy_story": 862, "rupture": 345925}
# later franchise entry; first franchise entry (no franchise shown); no budget, no release type, no distributor


def person(name, films=12, revenue=480_000_000.0, score=7.1, wins=2, noms=5):
    return {"name": name, "films": films, "revenue_usd": revenue, "score": score, "wins": wins, "nominations": noms}


def company(name, films=3, revenue=90_000_000.0, score=6.5):
    return {"name": name, "films": films, "revenue_usd": revenue, "score": score}


def film(**over):
    base = {
        "genres": ["Drama", "Thriller"], "release_year": 2019, "release_month": 6, "release_day": 19, "runtime": 100, "language": "en",
        "certification": "PG-13", "release_type": "wide", "countries": ["US", "CA"], "themes": ["heist", "family"], "budget": 50_000_000.0,
        "synopsis": "A short plot.", "tagline": "Tag line.", "source_material": "novel", "franchise": "Saga", "installment": 4,
        "director": person("Dee Rector"), "writers": [person("Wri Ter"), person("Ann Other", 1, 2_000_000.0, 6.0, 1, 1)],
        "cast": [person("Lead Actor"), person("Second Actor", 0, 17_000_000.0, 6.8, 0, 0)], "producers": [person("Pro Ducer")],
        "composer": person("Com Poser"), "distributor": company("Big Dist"), "companies": [company("Studio One"), company("Studio Two")],
        "competition": 2, "collaborations": 3, "director_age": 54, "lead_age": 62, "source_strength": 5,
    }
    return ResolvedFilm(**{**base, **over})


def prompt_of(con, ctx, movie_id):
    form = film_to_form(film_row(con, movie_id))
    return render(resolve(form, form_as_of(form), ctx))


# ---- goldens ----


@pytest.mark.parametrize("name", GOLDEN_FILMS)
def test_render_matches_golden(name, mini_con, mini_ctx):
    text = prompt_of(mini_con, mini_ctx, GOLDEN_FILMS[name])
    path = GOLDEN / f"{name}.txt"
    if os.environ.get("UPDATE_GOLDEN") == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
    assert path.exists(), f"missing golden {path}; run with UPDATE_GOLDEN=1 and read it"
    assert text == path.read_text(encoding="utf-8")


def test_golden_first_entry_shows_no_franchise_and_later_entry_does(mini_con, mini_ctx):
    assert "Franchise: not given" in prompt_of(mini_con, mini_ctx, 862)
    assert "(installment 4)" in prompt_of(mini_con, mini_ctx, 301528)


# ---- layout, by hand ----


def test_render_layout_and_line_order():
    expected = "Release: 19 June 2019, Wednesday\nRelease type: wide\nDistributor: Big Dist [3 films · $90M · 6.5]\nGenres: Drama, Thriller\nRuntime: 100 min · Rating: PG-13 · Language: en · Countries: US, CA\nDirector: Dee Rector [12 films · $480M · 7.1 · 2 wins, 5 noms] · age 54\nWriters: Wri Ter [12 films · $480M · 7.1 · 2 wins, 5 noms]; Ann Other [1 film · $2M · 6.0 · 1 win, 1 nom]\nCast: Lead Actor [12 films · $480M · 7.1 · 2 wins, 5 noms]; Second Actor [0 films · $17M · 6.8 · 0 wins, 0 noms]\nLead actor age: 62\nCollaborations: 3 earlier films (director with top 3 actors)\nProducers: Pro Ducer [12 films · $480M · 7.1 · 2 wins, 5 noms]\nComposer: Com Poser [12 films · $480M · 7.1 · 2 wins, 5 noms]\nCompanies: Studio One [3 films · $90M · 6.5]; Studio Two [3 films · $90M · 6.5]\nSource: novel\nSource strength: 5\nFranchise: Saga (installment 4)\nThemes: heist, family\nBudget: $50M\nCompetition: 2 big-budget films within 14 days\nTagline: Tag line.\nSynopsis: A short plot."
    assert render(film()) == expected


def test_no_title_line_and_no_id_in_text(mini_con, mini_ctx):
    for movie_id in (301528, 862, 345925, 577922):
        text = prompt_of(mini_con, mini_ctx, movie_id)
        assert not any(line.startswith("Title") for line in text.splitlines())
        row = film_row(mini_con, movie_id)
        form = film_to_form(row)
        ids = [movie_id, row["collection_id"], form["director"]["id"]] + [p["id"] for p in form["actors"]]
        for i in ids:
            if i is not None and i >= 10_000:  # smaller ids could collide with counts such as "31 films"
                assert str(i) not in text, (movie_id, i)


def test_singular_counts():
    text = render(film(competition=1, collaborations=1))
    assert "Collaborations: 1 earlier film (director" in text
    assert "Competition: 1 big-budget film within 14 days" in text


def test_none_values_print_not_given():
    text = render(film(release_type=None, runtime=None, language=None, certification=None, director_age=None, lead_age=None,
                       source_strength=None, genres=[], themes=[]))
    assert "Release type: not given" in text
    assert "Runtime: not given · Rating: not given · Language: not given" in text
    assert "age not given" in text and "Lead actor age: not given" in text
    assert "Source strength: not given" in text
    assert "Genres: not given" in text and "Themes: not given" in text


def test_empty_strings_print_not_given():
    text = render(film(release_type="", language="", certification="", tagline="", synopsis="", source_material=""))
    assert "Release type: not given" in text and "Runtime: 100 min · Rating: not given · Language: not given" in text
    assert "Tagline: not given" in text and "Synopsis: not given" in text and "Source: not given" in text


def test_release_type_none_prints_not_given(mini_con, mini_ctx):
    text = prompt_of(mini_con, mini_ctx, 345925)  # Rupture has no stored release type
    assert "Release type: not given" in text and "wide" not in text
    assert "Budget: not given" in text and "Distributor: not given" in text


def test_person_record_formats():
    text = render(film(director=person("Solo", films=1, revenue=950_000.0, score=7.0, wins=1, noms=0), cast=[person("Zed", 0, None, None, 0, 0)]))
    assert "Solo [1 film · $950K · 7.0 · 1 win, 0 noms]" in text
    assert "Zed [0 films · not given · not given · 0 wins, 0 noms]" in text


# ---- hidden groups ----

HIDDEN_LINES = {
    "writers": ["Writers: not given"],
    "producers": ["Producers: not given"],
    "composer": ["Composer: not given"],
    "distributor": ["Distributor: not given"],
    "source_material": ["Source: not given", "Source strength: not given"],
    "release_day": ["Release: June 2019, weekday not given"],
    "tagline": ["Tagline: not given"],
    "synopsis": ["Synopsis: not given"],
    "budget": ["Budget: not given"],
    "companies": ["Companies: not given"],
    "countries": ["Countries: not given"],
    "franchise": ["Franchise: not given"],
}
HIDDEN_ABSENT = {  # what must not survive hiding
    "writers": ["Wri Ter"], "producers": ["Pro Ducer"], "composer": ["Com Poser"], "distributor": ["Big Dist"],
    "source_material": ["novel", "Source strength: 5"], "release_day": ["Wednesday", "19 June"], "tagline": ["Tag line"],
    "synopsis": ["A short plot"], "budget": ["$50M"], "companies": ["Studio One"], "countries": ["US, CA"],
    "franchise": ["Saga", "installment"],
}
MISSING = {  # the same film with the field missing instead of hidden
    "writers": {"writers": []}, "producers": {"producers": []}, "composer": {"composer": None}, "distributor": {"distributor": None},
    "source_material": {"source_material": None, "source_strength": None}, "release_day": {"release_day": None},
    "tagline": {"tagline": None}, "synopsis": {"synopsis": None}, "budget": {"budget": None}, "companies": {"companies": []},
    "countries": {"countries": []}, "franchise": {"franchise": None, "installment": None},
}


def test_optional_groups_are_the_agreed_twelve():
    assert OPTIONAL_GROUPS == (
        "writers", "producers", "composer", "distributor", "source_material", "release_day", "tagline", "synopsis", "budget",
        "companies", "countries", "franchise",
    )
    assert set(HIDDEN_LINES) == set(HIDDEN_ABSENT) == set(MISSING) == set(OPTIONAL_GROUPS)


@pytest.mark.parametrize("group", OPTIONAL_GROUPS)
def test_hidden_fields_print_not_given(group):
    text = render(film(), hidden=frozenset({group}))
    for line in HIDDEN_LINES[group]:
        assert line in text, (group, line)
    for gone in HIDDEN_ABSENT[group]:
        assert gone not in text, (group, gone)


@pytest.mark.parametrize("group", OPTIONAL_GROUPS)
def test_hidden_and_missing_look_identical(group):
    assert render(film(), hidden=frozenset({group})) == render(film(**MISSING[group]))


def test_hiding_nothing_changes_nothing_and_hiding_leaves_other_lines():
    full = render(film())
    for group in OPTIONAL_GROUPS:
        assert len(render(film(), hidden=frozenset({group})).splitlines()) == len(full.splitlines())  # the line order is fixed


def test_unknown_hidden_group_raises():
    with pytest.raises(ValueError, match="nonsense"):
        render(film(), hidden=frozenset({"nonsense"}))
    with pytest.raises(ValueError):
        render(film(), hidden=frozenset({"director"}))  # required inputs cannot be hidden


# ---- synopsis, money, weekday ----


def test_synopsis_truncated_to_120_words():
    words = [f"w{i}" for i in range(130)]
    line = next(x for x in render(film(synopsis="  ".join(words))).splitlines() if x.startswith("Synopsis: "))
    kept = line.removeprefix("Synopsis: ").split()
    assert len(kept) == 120
    assert kept[0] == "w0" and kept[-1].startswith("w119") and "w120" not in line
    exactly = " ".join(words[:120])
    assert render(film(synopsis=exactly)).endswith("Synopsis: " + exactly)  # 120 words are kept whole
    assert "w119" in render(film(synopsis=exactly))


def test_synopsis_newlines_do_not_break_the_layout():
    text = render(film(synopsis="First line.\n\nSecond   line."))
    assert text.endswith("Synopsis: First line. Second line.")


@pytest.mark.parametrize("amount, expected", [
    (None, "not given"), (0, "$0"), (999, "$999"), (999.4, "$999"), (999.5, "$1K"), (1_000, "$1K"), (950_000, "$950K"), (999_499, "$999K"), (999_500, "$1M"),
    (1_000_000, "$1M"), (480_000_000, "$480M"), (999_499_999, "$999M"), (999_500_000, "$1.0B"), (1_000_000_000, "$1.0B"),
    (1_200_000_000, "$1.2B"), (2_750_000_000, "$2.8B"),
])
def test_money_formats_at_boundaries(amount, expected):
    assert money(amount) == expected


@pytest.mark.parametrize("year, month, day, weekday", [(2019, 6, 19, "Wednesday"), (2000, 1, 1, "Saturday"), (2024, 2, 29, "Thursday")])
def test_weekday_on_known_dates(year, month, day, weekday):
    # checked by hand: Toy Story 4 opened Friday 21 June 2019, so the 19th was a Wednesday; 1 January 2000 was a Saturday;
    # 29 February 2024 was a Thursday (1 March 2024 was a Friday)
    first = render(film(release_year=year, release_month=month, release_day=day)).splitlines()[0]
    assert first.endswith(f", {weekday}")


# ---- size ----


def test_every_real_mini_film_fits_the_proxy_limit(mini_con, mini_ctx):
    # the plan's limit is for real films (goldens included); the hand-built worst case below only bounds the layout
    ids = [r[0] for r in mini_con.execute("SELECT tmdb_id FROM movies")]
    assert len(ids) == 37
    for movie_id in ids:
        assert len(prompt_of(mini_con, mini_ctx, movie_id)) <= 2800, movie_id


def test_prompt_under_700_tokens_proxy():
    # worst case of the form: 10 actors, 3 writers, 3 producers, 5 companies, a full tagline, a 120-word synopsis. The names are
    # real ones from the mini set; every record is as wide as a busy career makes it (three-digit films, two-digit wins)
    def named(name):
        return person(name, films=145, revenue=480_000_000.0, score=7.6, wins=12, noms=48)

    cast = ["Tom Hanks", "Tim Allen", "Annie Potts", "Tony Hale", "Keegan-Michael Key", "Madeleine McGraw", "Christina Hendricks",
            "Jordan Peele", "Keanu Reeves", "Ally Maki"]  # Toy Story 4's ten billed actors
    story = ["a", "retired", "courier", "discovers", "that", "the", "quiet", "harbour", "town", "hides", "an", "old", "conspiracy", "and", "must", "trust", "a", "stranger", "before", "the", "winter", "storm", "closes", "every", "road", "and", "the", "last", "ferry", "leaves", "without", "anyone", "aboard"]
    synopsis = " ".join((story * 10)[:200])  # 200 words: cut to 120
    worst = film(
        genres=["Science Fiction", "Action", "Adventure", "Thriller", "Fantasy"], countries=["US", "GB", "FR", "DE", "CA"],
        themes=["time travel", "heist", "revenge", "friendship", "survival"], director=named("Steven Shainberg"),
        writers=[named(n) for n in ("Stephany Folsom", "Andrew Stanton", "Brian Nelson")], cast=[named(n) for n in cast],
        producers=[named(n) for n in ("Jonas Rivera", "Mark Nielsen", "Galyn Susman")], composer=named("Ludwig Goransson"),
        distributor=company("Twentieth Century Studios", 45, 480_000_000.0, 7.6),
        companies=[company(n, 45, 480_000_000.0, 7.6) for n in ("Warner Bros. Pictures", "Tango Pictures", "Splendid Film", "BondIt", "AMBI Group")],
        tagline="T" * 149, synopsis=synopsis, source_material="video game", franchise="Extraordinary Adventures Collection",
        installment=12, competition=9, collaborations=14, director_age=99, lead_age=99, source_strength=250,
    )
    text = render(worst)
    assert len(text.split("Synopsis: ")[1].split()) == 120
    assert len(text) <= 2800  # 700 tokens at about 4 characters per token


def test_newlines_in_free_text_cannot_add_lines():
    clean = render(film())
    hostile = render(film(
        tagline="a\nBudget: $1B", source_material="novel\nBudget: $2B", franchise="Saga\nBudget: $3B", genres=["Drama\nBudget: $4B"],
        themes=["heist\r\nBudget: $5B"], countries=["US\nBudget: $6B"], language="en\nBudget: $7B", certification="R\nBudget: $8B",
        release_type="wide\nBudget: $9B",
        director=person("Bad\nName"), writers=[person("Wri\nTer")], cast=[person("Cast\tMember\nBudget: $1M")],
        distributor=company("Big\nDist"), companies=[company("Studio\nOne")],
    ))
    assert len(hostile.splitlines()) == len(clean.splitlines())
    assert sum(line.startswith("Budget:") for line in hostile.splitlines()) == 1
    assert "Tagline: a Budget: $1B" in hostile and "Director: Bad Name [" in hostile
    assert "Distributor: Big Dist [" in hostile and "Studio One [" in hostile


def test_whitespace_only_name_prints_unnamed():
    text = render(film(director=person(" \n "), cast=[person("")]))
    assert "Director: Unnamed [" in text and "Cast: Unnamed [" in text


@pytest.mark.parametrize("blank", [None, "", " \n ", " ", "\r\n\t"])
def test_unnamed_applies_to_every_person_and_company_and_keeps_the_record(blank):
    text = render(film(
        director=person(blank, films=7, revenue=12_000_000.0, score=6.0, wins=0, noms=5), writers=[person(blank, 1, 2_000_000.0, 6.0, 1, 1)],
        cast=[person(blank, 0, None, None, 0, 0)], producers=[person(blank, 4, 950_000.0, 5.5, 2, 3)], composer=person(blank, 9, 3_000_000.0, 7.0, 0, 1),
        distributor=company(blank, 45, 480_000_000.0, 7.6), companies=[company(blank, 3, 90_000_000.0, 6.5), company("Studio Two")],
    ))
    lines = dict(line.split(": ", 1) for line in text.splitlines())
    assert lines["Director"] == "Unnamed [7 films · $12M · 6.0 · 0 wins, 5 noms] · age 54"
    assert lines["Writers"] == "Unnamed [1 film · $2M · 6.0 · 1 win, 1 nom]"
    assert lines["Cast"] == "Unnamed [0 films · not given · not given · 0 wins, 0 noms]"
    assert lines["Producers"] == "Unnamed [4 films · $950K · 5.5 · 2 wins, 3 noms]"
    assert lines["Composer"] == "Unnamed [9 films · $3M · 7.0 · 0 wins, 1 nom]"
    assert lines["Distributor"] == "Unnamed [45 films · $480M · 7.6]"
    assert lines["Companies"] == "Unnamed [3 films · $90M · 6.5]; Studio Two [3 films · $90M · 6.5]"


@pytest.mark.parametrize("sep", ["\n", "\r\n", "\r", "\t", " ", " ", " ", "\x85", "\x0b", "\x0c", "\x1c", "  \n  "])
def test_any_whitespace_collapses_to_one_space_in_every_free_text_field(sep):
    j = lambda a, b: f"{a}{sep}{b}"
    clean = render(film())
    text = render(film(
        tagline=j("a", "b"), source_material=j("c", "d"), franchise=j("e", "f"), genres=[j("g", "h")], themes=[j("i", "k")],
        countries=[j("l", "m")], language=j("n", "o"), certification=j("p", "q"), release_type=j("r", "s"),
        director=person(j("Dee", "Rector")), writers=[person(j("W", "T"))], cast=[person(j("C", "M"))], producers=[person(j("P", "D"))],
        composer=person(j("Co", "Po")), distributor=company(j("Big", "Dist")), companies=[company(j("Stu", "One"))],
    ))
    assert len(text.splitlines()) == len(clean.splitlines())
    for fragment in ("Tagline: a b", "Source: c d", "Franchise: e f (", "Genres: g h", "Themes: i k", "Countries: l m", "Language: n o",
                     "Rating: p q", "Release type: r s", "Director: Dee Rector [", "Writers: W T [", "Cast: C M [", "Producers: P D [",
                     "Composer: Co Po [", "Distributor: Big Dist [", "Companies: Stu One ["):
        assert fragment in text, fragment


def test_blank_franchise_name_prints_not_given():
    for name in ("", "  ", "\n"):  # the whole line, so no "(installment 3)" may trail the label
        lines = dict(line.split(": ", 1) for line in render(film(franchise=name, installment=3)).splitlines())
        assert lines["Franchise"] == "not given", repr(name)
    assert "Franchise: Saga (installment 3)" in render(film(franchise=" Saga\n", installment=3))


def test_list_drops_blank_entries():
    text = render(film(genres=["  "], themes=["", "Drama", " \n"], countries=["", " \t", " "]))
    lines = dict(line.split(": ", 1) for line in text.splitlines())
    assert lines["Genres"] == "not given"
    assert lines["Themes"] == "Drama"  # exact: a leftover separator such as "Drama, " must fail
    assert lines["Runtime"].endswith("Countries: not given")  # countries sit on the Runtime line; ", , " must fail


# ---- hideable extras and anonymized names ----


def test_hide_release_type_and_themes():
    assert HIDEABLE == OPTIONAL_GROUPS + ("source_strength", "release_type", "themes")
    text = render(film(), hidden=frozenset({"release_type", "themes"}))
    assert "Release type: not given" in text and "Themes: not given" in text
    assert "wide" not in text and "heist" not in text
    assert len(text.splitlines()) == len(render(film()).splitlines())


def test_source_strength_hideable_alone():
    text = render(film(), hidden=frozenset({"source_strength"}))
    assert "Source: novel" in text and "Source strength: not given" in text


def test_unknown_group_still_raises():
    with pytest.raises(ValueError, match="HIDEABLE"):
        render(film(), hidden=frozenset({"nonsense"}))
    with pytest.raises(ValueError):
        render(film(), hidden=frozenset({"director"}))


def test_random_hide_excludes_release_type_and_themes():
    assert RANDOM_HIDE == OPTIONAL_GROUPS + ("source_strength",)
    assert "release_type" not in RANDOM_HIDE and "themes" not in RANDOM_HIDE
    assert set(RANDOM_HIDE) <= set(HIDEABLE)


def test_anonymize_removes_every_name(mini_con, mini_ctx):
    form = film_to_form(film_row(mini_con, 301528))
    r = resolve(form, form_as_of(form), mini_ctx)
    text = render(r, anonymize=True)
    names = [p["name"] for p in [r.director, r.composer, *r.writers, *r.cast, *r.producers, *r.companies] if p and p["name"]]
    names.append(r.distributor["name"])
    names.append(r.franchise)
    names = [n for n in names if n]
    assert names
    lines = dict(line.split(": ", 1) for line in text.splitlines())
    non_free = "\n".join(f"{k}: {v}" for k, v in lines.items() if k not in ("Tagline", "Synopsis"))
    for name in names:
        assert name not in non_free, name
    assert lines["Director"].startswith("Person 1 [")
    assert lines["Distributor"].startswith("Distributor [")
    assert lines["Franchise"] == f"{GENERIC_FRANCHISE} (installment {r.installment})"


def test_anonymize_same_person_same_number():
    text = render(film(writers=[person("Dee Rector"), person("Wri Ter")], cast=[person("Wri Ter"), person("New Face")],
                       companies=[company("Studio One"), company("Studio One"), company("Studio Two")]), anonymize=True)
    lines = dict(line.split(": ", 1) for line in text.splitlines())
    assert lines["Director"].startswith("Person 1 [")
    assert lines["Writers"].startswith("Person 1 [") and "; Person 2 [" in lines["Writers"]
    assert lines["Cast"].startswith("Person 2 [") and "; Person 3 [" in lines["Cast"]
    assert lines["Producers"].startswith("Person 4 [") and lines["Composer"].startswith("Person 5 [")
    assert lines["Companies"].startswith("Company 1 [") and "; Company 1 [" in lines["Companies"] and "; Company 2 [" in lines["Companies"]


def test_anonymize_nameless_people_get_fresh_numbers():
    text = render(film(cast=[person(""), person(" ")]), anonymize=True)
    cast = dict(line.split(": ", 1) for line in text.splitlines())["Cast"]
    assert cast.startswith("Person ") and "Unnamed" not in cast
    first, second = (c.split(" [")[0] for c in cast.split("; "))
    assert first != second


def test_anonymize_keeps_records(mini_con, mini_ctx):
    form = film_to_form(film_row(mini_con, 301528))
    r = resolve(form, form_as_of(form), mini_ctx)
    plain, anon = render(r).splitlines(), render(r, anonymize=True).splitlines()
    assert len(plain) == len(anon)
    brackets = lambda line: re.findall(r"\[[^\]]*\]", line)
    for a, b in zip(plain, anon):
        if a.split(": ", 1)[0] in ("Director", "Writers", "Cast", "Producers", "Composer", "Companies", "Distributor"):
            assert brackets(a) == brackets(b), a
        elif a.split(": ", 1)[0] != "Franchise":
            assert a == b, a
    assert "age 54" in render(film(), anonymize=True) and "Budget: $50M" in render(film(), anonymize=True)
