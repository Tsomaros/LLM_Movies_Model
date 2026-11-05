"""Turn a film into the facts the model reads, with nothing from after the film's release.

    film_row(con, tmdb_id)         -> what film_to_form needs, read from the database
    film_to_form(row)             -> a training film written as a form, in the shape validate() returns
    form_as_of(form)               -> the release date (the 1st of the month without release_day)
    resolve(form, as_of, ctx)      -> ResolvedFilm: the form values plus every track record and computed input
    ResolvedFilm.to_json() / from_json(text)

This is the last guard before training data, so:
- film_to_form never reads vote_count, vote_average, popularity, revenue_*, money_suspect or split. film_row does not even select them.
- The budget is a pre-release fact: it is passed on whether or not money_suspect is set (the flag comes from the film's own revenue).
- First-film rule, for training films and user forms alike: a collection with no entry strictly before `as_of` gives
  franchise=None and installment=None. A first film has a collection only because sequels came later, so showing it would
  leak that a sequel got made. A user's new franchise reads as standalone too.
- A collection has an id and a name (movies.collection_name, which can be NULL). A later entry prints GENERIC_FRANCHISE when the form
  has an id but no name; a name with no id is a new franchise and reads as standalone.
- Every NaN from History becomes None, so to_json is strict JSON.
- film_to_form output is never passed through validate(): a film with 3 actors or no themes must still resolve and render.
"""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

from llm_movies.features.computed import (
    Context,
    age_at,
    collaborations,
    competition,
    installment,
    prior_awards,
    source_strength,
)

GENERIC_FRANCHISE = "a film series"
MAX_ACTORS, MAX_CREW, MAX_GENRES, MAX_COMPANIES, MAX_COUNTRIES, MAX_THEMES = 10, 3, 5, 5, 5, 5

_THEMES_PATH = Path(__file__).resolve().parents[1] / "contract" / "themes.json"
# [(theme name, casefolded keyword names)] in the file's order
_THEMES = [
    (t["name"], {k["name"].casefold() for k in t["keywords"]})
    for t in json.loads(_THEMES_PATH.read_text(encoding="utf-8"))["themes"]
]

_MOVIE_COLUMNS = (
    "tmdb_id", "release_date", "runtime", "language", "certification", "genres", "countries", "keywords", "overview",
    "tagline", "collection_id", "collection_name", "budget_2025", "release_type", "source_material", "distributor", "source_sitelinks",
)
_JSON_COLUMNS = ("genres", "countries", "keywords")


# ---------------------------------------------------------------- the database side


def film_row(con: sqlite3.Connection, tmdb_id: int) -> dict[str, Any]:
    """The movie row (only the columns the form can use), its credits in `ord` order per role, companies in `ord` order,
    keyword names. Raises KeyError for an unknown film."""
    found = con.execute(f"SELECT {', '.join(_MOVIE_COLUMNS)} FROM movies WHERE tmdb_id = ?", (tmdb_id,)).fetchone()
    if found is None:
        raise KeyError(tmdb_id)
    row = dict(zip(_MOVIE_COLUMNS, found))
    for column in _JSON_COLUMNS:
        row[column] = json.loads(row[column]) if row[column] else []
    credits: dict[str, list[dict[str, Any]]] = {}
    for role, person_id, name in con.execute(
        "SELECT c.role, c.person_id, p.name FROM credits c LEFT JOIN people p ON p.id = c.person_id "
        "WHERE c.movie_id = ? ORDER BY c.role, c.ord, c.person_id",
        (tmdb_id,),
    ):
        credits.setdefault(role, []).append({"id": person_id, "name": name})
    row["credits"] = credits
    row["companies"] = [
        {"id": company_id, "name": name}
        for company_id, name in con.execute(
            "SELECT mc.company_id, co.name FROM movie_companies mc LEFT JOIN companies co ON co.id = mc.company_id "
            "WHERE mc.movie_id = ? ORDER BY mc.ord, mc.company_id",
            (tmdb_id,),
        )
    ]
    return row


def film_to_form(row: dict[str, Any]) -> dict[str, Any]:
    """A training film as a form, in the clean shape validate() returns. The row already holds everything, so there is no ctx. Adds two non-form keys: film_id and source_sitelinks."""
    released = date.fromisoformat(row["release_date"]) if isinstance(row["release_date"], str) else row["release_date"]
    credits = row.get("credits", {})
    keywords = {k.casefold() for k in row.get("keywords") or []}
    themes = [name for name, listed in _THEMES if keywords & listed][:MAX_THEMES]
    collection_id = row.get("collection_id")
    return {
        "genres": list(row.get("genres") or [])[:MAX_GENRES],
        "release_year": released.year,
        "release_month": released.month,
        "release_day": released.day,
        "runtime": row.get("runtime"),
        "language": row.get("language"),
        "certification": row.get("certification"),
        "director": (credits.get("director") or [None])[0],
        "actors": list(credits.get("actor") or [])[:MAX_ACTORS],
        "themes": themes,
        "companies": list(row.get("companies") or [])[:MAX_COMPANIES],
        "countries": list(row.get("countries") or [])[:MAX_COUNTRIES],
        "franchise": None if collection_id is None else {"id": collection_id, "name": row.get("collection_name")},
        "synopsis": row.get("overview"),
        "budget": row.get("budget_2025"),  # whether or not money_suspect: that flag comes from the film's own revenue
        "writers": list(credits.get("writer") or [])[:MAX_CREW],
        "producers": list(credits.get("producer") or [])[:MAX_CREW],
        "composer": (credits.get("composer") or [None])[0],
        "distributor": row.get("distributor"),
        "source_material": row.get("source_material"),
        "release_type": row.get("release_type"),
        "tagline": row.get("tagline"),
        "film_id": row.get("tmdb_id"),
        "source_sitelinks": row.get("source_sitelinks"),
    }


def form_as_of(form: dict[str, Any]) -> date:
    """The release date; the 1st of the month when the form has no release_day."""
    return date(form["release_year"], form["release_month"], form.get("release_day") or 1)


# ---------------------------------------------------------------- resolved film


@dataclass(frozen=True)
class ResolvedFilm:
    genres: list[str]
    release_year: int
    release_month: int
    release_day: int | None
    runtime: int | None
    language: str | None
    certification: str | None
    release_type: str | None
    countries: list[str]
    themes: list[str]
    budget: float | None
    synopsis: str | None
    tagline: str | None
    source_material: str | None
    franchise: str | None  # None for a first entry (see the module docstring)
    installment: int | None
    director: dict[str, Any] | None  # {name, films, revenue_usd, score, wins, nominations}
    writers: list[dict[str, Any]]
    cast: list[dict[str, Any]]
    producers: list[dict[str, Any]]
    composer: dict[str, Any] | None
    distributor: dict[str, Any] | None  # {name, films, revenue_usd, score}
    companies: list[dict[str, Any]]
    competition: int
    collaborations: int
    director_age: int | None
    lead_age: int | None
    source_strength: int | None

    def to_json(self) -> str:
        return json.dumps(asdict(self), allow_nan=False, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> ResolvedFilm:
        return cls(**json.loads(text))


def _number(value: float | None) -> float | None:
    return None if value is None or math.isnan(value) else float(value)


def _name(ref: Any) -> str | None:
    if isinstance(ref, dict):
        name = ref.get("name")
    else:
        name = ref if isinstance(ref, str) else None
    return name.strip() or None if isinstance(name, str) else None


def _id(ref: Any) -> int | None:
    return ref.get("id") if isinstance(ref, dict) else (ref if isinstance(ref, int) and not isinstance(ref, bool) else None)


def _record(name: str | None, record) -> dict[str, Any]:
    revenue = _number(record.revenue_log)
    return {
        "name": name,
        "films": record.films,
        "revenue_usd": None if revenue is None else math.exp(revenue),
        "score": _number(record.score),
    }


def _person(ctx: Context, ref: Any, role: str, as_of: date) -> tuple[dict[str, Any], int | None]:
    person_id = ctx.history.resolve(ref, role)
    record = _record(_name(ref), ctx.history.person(person_id, role, as_of))
    awards = prior_awards(ctx, [person_id], as_of)
    return {**record, "wins": awards["wins"], "nominations": awards["nominations"]}, person_id


def _refs(form: dict[str, Any], key: str) -> list[Any]:
    value = form.get(key)
    return list(value) if isinstance(value, (list, tuple)) else []


def resolve(form: dict[str, Any], as_of: date, ctx: Context) -> ResolvedFilm:
    """Everything the prompt shows. `form` is film_to_form's shape or validate()'s clean output; missing keys read as absent.
    `as_of` is the caller's: the release date for training films, form_as_of(form) for user forms."""
    director, director_id = (None, None)
    if form.get("director") is not None:
        director, director_id = _person(ctx, form["director"], "director", as_of)
    writers = [_person(ctx, ref, "writer", as_of)[0] for ref in _refs(form, "writers")]
    producers = [_person(ctx, ref, "producer", as_of)[0] for ref in _refs(form, "producers")]
    composer = _person(ctx, form["composer"], "composer", as_of)[0] if form.get("composer") is not None else None
    cast_pairs = [_person(ctx, ref, "actor", as_of) for ref in _refs(form, "actors")]
    cast = [record for record, _ in cast_pairs]
    actor_ids = [person_id for _, person_id in cast_pairs]

    companies = []
    for ref in _refs(form, "companies"):
        company_id = ctx.history.resolve(ref, "company")
        companies.append(_record(_name(ref), ctx.history.company(company_id, as_of)))
    distributor_name = form.get("distributor")
    distributor = None
    if isinstance(distributor_name, str) and distributor_name.strip():
        distributor = _record(distributor_name.strip(), ctx.history.distributor(distributor_name, as_of))

    franchise = form.get("franchise")
    entry = installment(ctx, _id(franchise), as_of)
    if entry is None or entry == 1:  # no collection, or a first entry: nothing earlier to show
        franchise_name, entry = None, None
    else:
        franchise_name = _name(franchise) or GENERIC_FRANCHISE

    return ResolvedFilm(
        genres=list(form.get("genres") or []),
        release_year=form["release_year"],
        release_month=form["release_month"],
        release_day=form.get("release_day"),
        runtime=form.get("runtime"),
        language=form.get("language"),
        certification=form.get("certification"),
        release_type=form.get("release_type"),
        countries=list(form.get("countries") or []),
        themes=list(form.get("themes") or []),
        budget=_number(form.get("budget")),
        synopsis=form.get("synopsis"),
        tagline=form.get("tagline"),
        source_material=form.get("source_material"),
        franchise=franchise_name,
        installment=entry,
        director=director,
        writers=writers,
        cast=cast,
        producers=producers,
        composer=composer,
        distributor=distributor,
        companies=companies,
        competition=competition(ctx, as_of, form.get("film_id")),
        collaborations=collaborations(ctx, director_id, actor_ids, as_of),
        director_age=age_at(ctx, director_id, as_of),
        lead_age=age_at(ctx, actor_ids[0], as_of) if actor_ids else None,
        source_strength=source_strength(ctx, form.get("source_sitelinks")),
    )
