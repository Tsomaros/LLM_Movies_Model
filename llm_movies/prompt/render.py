"""Write a ResolvedFilm as the exact text the model reads (spec section 7).

    render(resolved, hidden=frozenset(), *, anonymize=False) -> str
    money(amount)                        -> "$480M", "$1.2B", "$950K"; "not given" for None

One line per item, in a fixed order (hidden or missing items keep their line and print "not given"):

    Release, Release type, Distributor, Genres, Runtime/Rating/Language/Countries, Director, Writers, Cast,
    Lead actor age, Collaborations, Producers, Composer, Companies, Source, Source strength, Franchise, Themes,
    Budget, Competition, Tagline, Synopsis

A person reads `Name [12 films · $480M · 7.1 · 2 wins, 5 noms]`; companies and the distributor have no awards.
`hidden` holds names from HIDEABLE = OPTIONAL_GROUPS + source_strength, release_type, themes (an unknown name raises
ValueError). RANDOM_HIDE (OPTIONAL_GROUPS + source_strength) is what training hides at random; release type and themes are
only hidden on purpose, in experiments. Hiding `franchise` also hides the installment, hiding `source_material` also hides
source strength (hiding `source_strength` alone keeps the source), hiding `release_day` prints the month and year with the
weekday "not given".
`anonymize=True` swaps names for labels, keeping every record: a person becomes `Person N` (numbered by first appearance in
line order; the same cleaned name keeps its number, a nameless person gets a fresh one), a company `Company N` (same rule),
the distributor `Distributor`, the franchise `Franchise`. Tagline and synopsis are free text and are left as they are. A hidden group and a missing one produce the same text. There is no title, and no id is ever stored in a
ResolvedFilm, so none can reach the text.
"""

from __future__ import annotations

import calendar
from collections.abc import Callable
from datetime import date
from typing import Any

from llm_movies.prompt.outcome_words import mask_outcomes
from llm_movies.prompt.resolve import ResolvedFilm

OPTIONAL_GROUPS = (
    "writers", "producers", "composer", "distributor", "source_material", "release_day", "tagline", "synopsis", "budget",
    "companies", "countries", "franchise",
)
HIDEABLE = OPTIONAL_GROUPS + ("source_strength", "release_type", "themes")
RANDOM_HIDE = OPTIONAL_GROUPS + ("source_strength",)
GENERIC_FRANCHISE = "Franchise"
NOT_GIVEN = "not given"
SYNOPSIS_WORDS = 120
UNNAMED = "Unnamed"  # a person or company with an id but no name
_SEP = " · "


def money(amount: float | None) -> str:
    if amount is None:
        return NOT_GIVEN
    if amount >= 999_500_000:  # thresholds sit where the next unit rounds up, so "$1000M" never appears
        return f"${amount / 1e9:.1f}B"
    if amount >= 999_500:
        return f"${amount / 1e6:.0f}M"
    if amount >= 999.5:
        return f"${amount / 1e3:.0f}K"
    return f"${amount:.0f}"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _clean(value: Any) -> str:
    """Free text on one line: any run of whitespace (newlines too) becomes one space, so no value can add a line."""
    return " ".join(str(value).split())


def _value(value: Any) -> str:
    return _clean(value) or NOT_GIVEN if value is not None else NOT_GIVEN


def _list(values: list[str]) -> str:
    cleaned = [c for c in map(_clean, values) if c]
    return ", ".join(cleaned) or NOT_GIVEN


def _score(score: float | None) -> str:
    return NOT_GIVEN if score is None else f"{score:.1f}"


class _Namer:
    """Anonymous labels: the same cleaned name keeps its number; a blank name gets a fresh number each time."""

    def __init__(self, prefix: str) -> None:
        self.prefix, self.seen, self.count = prefix, {}, 0

    def __call__(self, name: Any) -> str:
        key = _clean(name or "")
        if not key or key not in self.seen:
            self.count += 1
            if key:
                self.seen[key] = self.count
            return f"{self.prefix} {self.count}"
        return f"{self.prefix} {self.seen[key]}"


def _record(record: dict[str, Any], awards: bool, namer: Callable[[Any], str] | None = None) -> str:
    parts = [_plural(record["films"], "film"), money(record["revenue_usd"]), _score(record["score"])]
    if awards:
        parts.append(f"{_plural(record['wins'], 'win')}, {record['nominations']} {'nom' if record['nominations'] == 1 else 'noms'}")
    name = namer(record["name"]) if namer else _clean(record["name"] or "") or UNNAMED
    return f"{name} [{_SEP.join(parts)}]"


def _people(records: list[dict[str, Any]], awards: bool = True, namer: Callable[[Any], str] | None = None) -> str:
    return "; ".join(_record(r, awards, namer) for r in records) if records else NOT_GIVEN


def _release(r: ResolvedFilm, hidden: frozenset[str]) -> str:
    month = calendar.month_name[r.release_month]
    if "release_day" in hidden or r.release_day is None:
        return f"Release: {month} {r.release_year}, weekday {NOT_GIVEN}"
    weekday = calendar.day_name[date(r.release_year, r.release_month, r.release_day).weekday()]
    return f"Release: {r.release_day} {month} {r.release_year}, {weekday}"


def _synopsis(text: str | None) -> str:
    words = (mask_outcomes(text) or "").split()
    if not words:
        return NOT_GIVEN
    if len(words) > SYNOPSIS_WORDS:
        words = words[:SYNOPSIS_WORDS]
        words[-1] += "..."
    return " ".join(words)


def render(r: ResolvedFilm, hidden: frozenset[str] = frozenset(), *, anonymize: bool = False) -> str:
    hidden = frozenset(hidden)
    unknown = sorted(hidden - set(HIDEABLE))
    if unknown:
        raise ValueError(f"unknown hidden group(s) {unknown}; expected names from HIDEABLE {HIDEABLE}")
    show = lambda group: group not in hidden
    people, companies = (_Namer("Person"), _Namer("Company")) if anonymize else (None, None)

    director = NOT_GIVEN
    if r.director is not None:
        director = f"{_record(r.director, True, people)}{_SEP}age {_value(r.director_age)}"
    franchise = NOT_GIVEN
    if show("franchise") and r.installment is not None and r.franchise is not None and _clean(r.franchise):
        franchise = f"{GENERIC_FRANCHISE if anonymize else _clean(r.franchise)} (installment {r.installment})"
    source = r.source_material if show("source_material") else None
    strength = r.source_strength if show("source_material") and show("source_strength") and source is not None else None

    lines = [
        _release(r, hidden),
        f"Release type: {_value(r.release_type) if show('release_type') else NOT_GIVEN}",
        f"Distributor: {_record(r.distributor, False, (lambda _: 'Distributor') if anonymize else None) if show('distributor') and r.distributor else NOT_GIVEN}",
        f"Genres: {_list(r.genres)}",
        _SEP.join([
            f"Runtime: {NOT_GIVEN if r.runtime is None else f'{r.runtime} min'}",
            f"Rating: {_value(r.certification)}",
            f"Language: {_value(r.language)}",
            f"Countries: {_list(r.countries) if show('countries') else NOT_GIVEN}",
        ]),
        f"Director: {director}",
        f"Writers: {_people(r.writers, namer=people) if show('writers') else NOT_GIVEN}",
        f"Cast: {_people(r.cast, namer=people)}",
        f"Lead actor age: {_value(r.lead_age)}",
        f"Collaborations: {_plural(r.collaborations, 'earlier film')} (director with top 3 actors)",
        f"Producers: {_people(r.producers, namer=people) if show('producers') else NOT_GIVEN}",
        f"Composer: {_record(r.composer, True, people) if show('composer') and r.composer else NOT_GIVEN}",
        f"Companies: {_people(r.companies, False, companies) if show('companies') else NOT_GIVEN}",
        f"Source: {_value(source)}",
        f"Source strength: {_value(strength)}",
        f"Franchise: {franchise}",
        f"Themes: {_list(r.themes) if show('themes') else NOT_GIVEN}",
        f"Budget: {money(r.budget) if show('budget') else NOT_GIVEN}",
        f"Competition: {_plural(r.competition, 'big-budget film')} within 14 days",
        f"Tagline: {_value(mask_outcomes(r.tagline)) if show('tagline') else NOT_GIVEN}",
        f"Synopsis: {_synopsis(r.synopsis) if show('synopsis') else NOT_GIVEN}",
    ]
    return "\n".join(lines)
