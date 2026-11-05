"""Cleaning: one raw TMDB record plus Wikidata facts becomes one clean row for the SQLite build."""

import re
from dataclasses import dataclass
from datetime import date

from llm_movies.contract.form import CERTIFICATIONS, UNRATED
from llm_movies.data.money import clean_amount, to_2025_usd
from llm_movies.data.source_material import source_category
from llm_movies.data.wikidata import FilmFacts, pick_worldwide_usd

LOW_RATIO = 0.0001  # revenue / budget below this means the revenue is probably a partial or wrong figure
INCOMPLETE_DAYS = 182  # an earlier film's revenue and score count once it is this old
KEPT_STATUSES = {"Released", "Post Production", "In Production", "Planned"}

# Release types in TMDB's release_dates: 1 premiere, 2 theatrical (limited), 3 theatrical, 4 digital, 5 physical, 6 TV.
_PREMIERE, _LIMITED, _WIDE = 1, 2, 3

# TMDB type 1 also covers red-carpet premieres days before opening, so a premiere counts as a festival debut only
# when it is at least this many days before the first theatrical date.
FESTIVAL_GAP_DAYS = 30

# Streaming families and the year each service launched (Wikipedia: Netflix streaming 16 Jan 2007, Amazon Unbox
# 7 Sep 2006, Hulu 29 Oct 2007, Disney+ 12 Nov 2019, Apple TV+ 1 Nov 2019, HBO Max 27 May 2020, Paramount+ 4 Mar
# 2021, Starz 1 Feb 1994). Wikidata lists these for later streaming rights, so a label only counts for a film that
# was released that way and no earlier than the launch.
STREAMER_START_YEARS = {
    "Netflix": 2007,
    "Disney+": 2019,
    "HBO Max": 2020,
    "Amazon Prime Video": 2006,
    "Paramount+": 2021,
    "Apple TV": 2019,
    "Hulu": 2007,
    "Starz": 1994,
}
_STREAMER_RELEASE_TYPES = {"streaming-only", "limited", "festival-first"}
_STREAMING_TYPES = {4, 5, 6}

# Distributor families, in priority order: the first family with any matching label wins and gives its canonical
# name. Built from the Wikidata distributor labels of 1,000 sampled films (counts in the task-7 report). Only a
# company's own arms that share its track record are merged; specialty arms and streaming services stay apart.
# A prefix matches a whole label or a label continuing after a non-word character (not "+"), so "Paramount Pictures"
# does not catch "Paramount+" and "Toho" does not catch "Tohoku".
DISTRIBUTOR_FAMILIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Warner Bros.", ("Warner Bros.", "New Line Cinema", "Warner Independent Pictures")),
    ("Universal", ("Universal Pictures",)),
    ("Paramount", ("Paramount Pictures", "Paramount Home Entertainment")),
    ("Sony Pictures Classics", ("Sony Pictures Classics",)),  # before Sony, whose "Sony Pictures" would swallow it
    ("Sony Pictures Television", ("Sony Pictures Television",)),  # before Sony, whose "Sony Pictures" would swallow it
    ("Sony", ("Sony Pictures", "Columbia Pictures", "TriStar Pictures", "Screen Gems")),
    ("Disney", ("Walt Disney", "The Walt Disney Company", "Touchstone Pictures", "Buena Vista", "Hollywood Pictures")),
    ("20th Century", ("20th Century",)),
    ("Lionsgate", ("Lionsgate", "Summit Entertainment")),
    ("Metro-Goldwyn-Mayer", ("Metro-Goldwyn-Mayer", "MGM")),
    ("United Artists", ("United Artists",)),
    ("Amazon MGM Studios", ("Amazon MGM",)),
    ("Focus Features", ("Focus Features",)),
    ("Searchlight Pictures", ("Searchlight Pictures", "Fox Searchlight")),
    ("Miramax", ("Miramax",)),
    ("DreamWorks", ("DreamWorks",)),
    ("Orion Pictures", ("Orion Pictures",)),
    ("A24", ("A24",)),
    ("Gaumont", ("Gaumont",)),
    ("StudioCanal", ("StudioCanal",)),
    ("Constantin Film", ("Constantin Film",)),
    ("Toho", ("Toho",)),
    ("The Weinstein Company", ("The Weinstein Company",)),
    ("Netflix", ("Netflix",)),
    ("Disney+", ("Disney+",)),
    ("HBO Max", ("HBO Max",)),
    ("Amazon Prime Video", ("Amazon Prime Video",)),
    ("Paramount+", ("Paramount+",)),
    ("Apple TV", ("Apple TV", "Apple TV+", "Apple Original Films")),
    ("Hulu", ("Hulu",)),
    ("Starz", ("Starz",)),
)


def _compile(prefixes: tuple[str, ...]) -> re.Pattern:
    return re.compile("(?:" + "|".join(re.escape(p) for p in prefixes) + r")(?![\w+])", re.IGNORECASE)


# Retail storefronts, not distributors: exact names, case-insensitive, dropped before any matching.
STOREFRONTS = frozenset({"fandango", "fandangonow", "xfinity streampix"})

_FAMILY_PATTERNS = tuple((name, _compile(prefixes)) for name, prefixes in DISTRIBUTOR_FAMILIES)


@dataclass
class MovieRow:
    tmdb_id: int
    title: str | None
    release_date: date
    year: int
    month: int
    day: int
    runtime: int | None
    language: str | None
    countries: list[str]
    genres: list[str]
    certification: str
    keywords: list[str]
    overview: str | None
    tagline: str | None
    collection_id: int | None
    collection_name: str | None  # TMDB's name for the collection, when it gives one
    budget_2025: float | None
    revenue_2025: float | None
    revenue_source: str | None  # "tmdb" | "wikidata" | None
    money_suspect: bool
    revenue_incomplete: bool
    vote_count: int | None  # a target only, never an input
    vote_average: float | None  # a target only, never an input
    release_type: str | None
    source_material: str
    distributor: str | None
    wd_qid: str | None
    source_sitelinks: int | None  # Wikipedias covering the source work (today's count); None without Wikidata or a source


def _dicts(value) -> list[dict]:
    return [v for v in value if isinstance(v, dict)] if isinstance(value, list) else []


def _dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def _text(value) -> str | None:
    return value.strip() or None if isinstance(value, str) else None


def _names(items, key: str) -> list[str]:
    return [t for d in _dicts(items) if (t := _text(d.get(key)))]


def _parse_date(value) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value.strip()[:10])
    except ValueError:
        return None


def pick_distributor(labels: list[str], release_type: str | None, year: int | None) -> str | None:
    """The canonical name of the first major family with a matching label, else the first sorted label, else None.
    Storefronts are dropped first; a streamer label is dropped unless the film was released as streaming-only,
    limited or festival-first and in or after the service's launch year."""
    kept = []
    for label in labels or []:
        if not isinstance(label, str) or not label.strip() or label.strip().lower() in STOREFRONTS:
            continue
        if _streamer_of(label) is not None and not _streamer_counts(_streamer_of(label), release_type, year):
            continue
        kept.append(label)
    for name, pattern in _FAMILY_PATTERNS:
        if any(pattern.match(label.strip()) for label in kept):
            return name
    return kept[0] if kept else None


def _streamer_of(label: str) -> str | None:
    return next((n for n, p in _FAMILY_PATTERNS if n in STREAMER_START_YEARS and p.match(label.strip())), None)


def _streamer_counts(streamer: str, release_type: str | None, year: int | None) -> bool:
    return release_type in _STREAMER_RELEASE_TYPES and year is not None and year >= STREAMER_START_YEARS[streamer]


def _release_entries(raw: dict) -> dict[str, list[tuple[date, int, str]]]:
    """Country -> its parseable (date, type, certification) entries; countries without any are left out."""
    by_country: dict[str, list[tuple[date, int, str]]] = {}
    for result in _dicts(_dict(raw.get("release_dates")).get("results")):
        country = _text(result.get("iso_3166_1"))
        if not country:
            continue
        for entry in _dicts(result.get("release_dates")):
            when = _parse_date(entry.get("release_date"))
            try:
                kind = int(entry.get("type"))
            except (TypeError, ValueError):
                continue
            if when is not None:
                by_country.setdefault(country, []).append((when, kind, _text(entry.get("certification")) or ""))
    return by_country


def _chosen_country(by_country: dict[str, list]) -> list:
    if by_country.get("US"):
        return by_country["US"]
    if not by_country:
        return []
    earliest = min(by_country, key=lambda c: (min(e[0] for e in by_country[c]), c))
    return by_country[earliest]


def release_type(entries: list[tuple[date, int, str]]) -> str | None:
    kinds = {k for _, k, _ in entries}
    theatrical = [d for d, k, _ in entries if k in (_LIMITED, _WIDE)]
    premieres = [d for d, k, _ in entries if k == _PREMIERE]
    if theatrical and premieres and (min(theatrical) - min(premieres)).days >= FESTIVAL_GAP_DAYS:
        return "festival-first"
    if _WIDE in kinds:
        return "wide"
    if _LIMITED in kinds:
        return "limited"
    if kinds & _STREAMING_TYPES:
        return "streaming-only"
    return None


def _certification(by_country: dict[str, list]) -> str:
    us = sorted(by_country.get("US", []), key=lambda e: e[0])  # stable: same-day entries keep their order
    theatrical = [c for _, k, c in us if k in (_LIMITED, _WIDE) and c]
    anyone = [c for _, _, c in us if c]
    found = (theatrical or anyone or [UNRATED])[0]
    return found if found in CERTIFICATIONS else UNRATED


def _money(tmdb_amount, wikidata_values, year: int, cpi: dict, fx: dict) -> tuple[float | None, str | None]:
    """(2025 dollars, source). TMDB first; Wikidata only when TMDB has no usable figure. CPI uses the release year."""
    number = tmdb_amount if isinstance(tmdb_amount, (int, float)) and not isinstance(tmdb_amount, bool) else None
    cleaned = clean_amount(number)
    source = "tmdb"
    if cleaned is None:
        cleaned, _ = pick_worldwide_usd(wikidata_values, fx)
        source = "wikidata"
    value = to_2025_usd(cleaned, year, cpi)
    return (value, source) if value is not None else (None, None)


def clean_movie(raw: dict, wd: FilmFacts | None, cpi: dict, fx: dict, pulled_at: date) -> MovieRow | None:
    released = _parse_date(raw.get("release_date"))
    if released is None or raw.get("status") not in KEPT_STATUSES:
        return None

    budget, _ = _money(raw.get("budget"), wd.cost if wd else [], released.year, cpi, fx)
    revenue, revenue_source = _money(raw.get("revenue"), wd.box_office if wd else [], released.year, cpi, fx)

    by_country = _release_entries(raw)
    keywords = _names(_dict(raw.get("keywords")).get("keywords"), "name")
    kind = release_type(_chosen_country(by_country))
    runtime = raw.get("runtime")
    belongs = _dict(raw.get("belongs_to_collection"))
    collection = belongs.get("id")
    return MovieRow(
        tmdb_id=raw["id"],  # required: a missing top-level id raises KeyError on purpose
        title=_text(raw.get("title")),
        release_date=released,
        year=released.year,
        month=released.month,
        day=released.day,
        runtime=runtime if isinstance(runtime, int) and runtime > 0 else None,
        language=_text(raw.get("original_language")),
        countries=_names(raw.get("production_countries"), "iso_3166_1"),
        genres=_names(raw.get("genres"), "name"),
        certification=_certification(by_country),
        keywords=keywords,
        overview=_text(raw.get("overview")),
        tagline=_text(raw.get("tagline")),
        collection_id=collection,
        collection_name=_text(belongs.get("name")),
        budget_2025=budget,
        revenue_2025=revenue,
        revenue_source=revenue_source,
        money_suspect=budget is not None and revenue is not None and revenue / budget < LOW_RATIO,
        revenue_incomplete=(pulled_at - released).days < INCOMPLETE_DAYS,
        vote_count=raw.get("vote_count"),
        vote_average=raw.get("vote_average"),
        release_type=kind,
        source_material=source_category(wd.source_types if wd else set(), set(keywords)),
        distributor=pick_distributor(wd.distributors if wd else [], kind, released.year),
        wd_qid=wd.qid if wd else None,
        source_sitelinks=wd.source_sitelinks if wd else None,
    )
