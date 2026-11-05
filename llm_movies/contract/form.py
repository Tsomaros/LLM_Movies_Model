"""The creator form: every input a prediction may use, its type and allowed values.

    FIELDS         -> one Field per form input, in form order
    describe()     -> FIELDS as JSON-ready dicts
    validate(form) -> (clean_form, errors); errors is a list of {"field", "message"}, empty when the form is usable

Only pre-release facts appear here. Post-release fields (popularity, vote_count, vote_average, revenue) are not
fields, so validate() rejects them as unknown and no feature can be built from them.

People, companies and the franchise are references: an id picked from autocomplete, or free text.
"""

from __future__ import annotations

import calendar
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from llm_movies.data.source_material import SOURCE_TYPES

GENRES = (
    "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", "Drama", "Family", "Fantasy",
    "History", "Horror", "Music", "Mystery", "Romance", "Science Fiction", "TV Movie", "Thriller", "War",
    "Western",
)  # TMDB's 19 movie genres
UNRATED = "unrated/unknown"
CERTIFICATIONS = ("G", "PG", "PG-13", "R", "NC-17", UNRATED)  # NR and missing both map to UNRATED
RELEASE_TYPES = ("wide", "limited", "festival-first", "streaming-only")
DEFAULT_RELEASE_TYPE = "wide"

MIN_YEAR, MAX_YEAR = 1960, 2035  # the pull starts in 1960; creators plan up to ten years ahead
MIN_RUNTIME, MAX_RUNTIME = 40, 300  # minutes; feature-length films
MIN_BUDGET, MAX_BUDGET = 1_000, 1_000_000_000  # 2025 USD
MAX_SYNOPSIS = 2_000  # characters
MAX_TAGLINE = 150  # characters
MAX_DISTRIBUTOR = 200  # characters
MIN_ACTORS, MAX_ACTORS = 5, 10
MAX_REF_ID = 2**31 - 1  # TMDB ids are far below this; keeps ids inside sqlite's integer range

LANGUAGE_CODE = re.compile(r"^[a-z]{2}$")  # TMDB uses ISO 639-1 (plus a few of its own, e.g. "cn", "xx")
COUNTRY_CODE = re.compile(r"^[A-Z]{2}$")  # ISO 3166-1 alpha-2

# themes.json: {"about": ..., "themes": [{"name": ..., "keywords": [...]}, ...], "dropped": {...}}; the form allows the names.
THEMES = tuple(t["name"] for t in json.loads(Path(__file__).with_name("themes.json").read_text(encoding="utf-8"))["themes"])


@dataclass(frozen=True)
class Field:
    name: str
    kind: str  # int | float | text | choice | choices | code | codes | ref | refs
    required: bool
    help: str
    min: float | None = None  # int / float: smallest value; text: shortest length
    max: float | None = None  # int / float: largest value; text: longest length
    choices: tuple[str, ...] | None = None  # choice / choices
    min_items: int | None = None  # choices / codes / refs
    max_items: int | None = None
    pattern: str | None = None  # code / codes
    default: str | None = None  # what validate() uses when an optional choice is left out


FIELDS = (
    Field("genres", "choices", True, "One or more TMDB genres", choices=GENRES, min_items=1, max_items=5),
    Field("release_year", "int", True, "Planned release year", min=MIN_YEAR, max=MAX_YEAR),
    Field("release_month", "int", True, "Planned release month (1-12)", min=1, max=12),
    Field("runtime", "int", True, "Runtime in minutes", min=MIN_RUNTIME, max=MAX_RUNTIME),
    Field("language", "code", True, "Original language, two-letter code (en, fr, hi...)", pattern=LANGUAGE_CODE.pattern),
    Field("certification", "choice", True, "US certification", choices=CERTIFICATIONS),
    Field("director", "ref", True, "Director: an id from autocomplete or a name"),
    Field("actors", "refs", True, "Top-billed actors, in billing order", min_items=MIN_ACTORS, max_items=MAX_ACTORS),
    Field("themes", "choices", True, "Story themes", choices=THEMES, min_items=1, max_items=5),
    Field("companies", "refs", False, "Production companies", min_items=0, max_items=5),
    Field("countries", "codes", False, "Production countries, two-letter codes (US, GB...)",
          min_items=0, max_items=5, pattern=COUNTRY_CODE.pattern),
    Field("franchise", "ref", False, "Franchise (TMDB collection) the film belongs to; a new name starts one"),
    Field("synopsis", "text", False, "Short plot summary", min=1, max=MAX_SYNOPSIS),
    Field("budget", "float", False, "Planned budget in 2025 US dollars", min=MIN_BUDGET, max=MAX_BUDGET),
    Field("writers", "refs", False, "Writers", min_items=0, max_items=3),
    Field("producers", "refs", False, "Producers", min_items=0, max_items=3),
    Field("composer", "ref", False, "Composer: an id from autocomplete or a name"),
    Field("distributor", "text", False, "Distributor name, as listed in the options", min=1, max=MAX_DISTRIBUTOR),
    Field("source_material", "choice", False, "What the story is based on", choices=SOURCE_TYPES),
    Field("release_day", "int", False, "Planned release day of the month (1-31), valid for the month", min=1, max=31),
    Field("release_type", "choice", False, "How the film is released", choices=RELEASE_TYPES, default=DEFAULT_RELEASE_TYPE),
    Field("tagline", "text", False, "Tagline", min=1, max=MAX_TAGLINE),
)
FIELDS_BY_NAME = {f.name: f for f in FIELDS}


def describe() -> list[dict[str, Any]]:
    return [{k: v for k, v in asdict(f).items() if v is not None} for f in FIELDS]


# ---------------------------------------------------------------- validation


def _empty(value: Any) -> bool:
    if isinstance(value, str):
        return not value.strip()  # whitespace-only counts as left out
    return value is None or (isinstance(value, (list, tuple)) and len(value) == 0)


def _number(f: Field, value: Any, cast: type) -> tuple[Any, str | None]:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or (isinstance(value, float) and not math.isfinite(value)):
        return None, "must be a number"  # NaN and infinity too: JSON parsers let them through
    if cast is int and value != int(value):
        return None, "must be a whole number"
    if not f.min <= value <= f.max:  # before the cast: float(10**400) would raise OverflowError
        low, high = (f"{x:,.0f}" if x >= 10_000 else f"{x:g}" for x in (f.min, f.max))
        return None, f"must be between {low} and {high}"
    return cast(value), None


def _text(value: Any) -> str | None:
    """Trimmed text, or None when value is not text or is blank."""
    return (value.strip() or None) if isinstance(value, str) else None


def _ref(value: Any) -> tuple[dict[str, Any] | None, str | None]:
    """An int is an id, a string is a name; a dict may carry both."""
    if isinstance(value, bool):
        return None, "must be an id or a name"
    if isinstance(value, int):
        value = {"id": value}
    elif isinstance(value, str):
        value = {"name": value}
    if not isinstance(value, dict) or set(value) - {"id", "name"}:
        return None, "must be an id or a name"
    ref_id, name = value.get("id"), _text(value.get("name"))
    if ref_id is not None and (isinstance(ref_id, bool) or not isinstance(ref_id, int) or not 0 < ref_id <= MAX_REF_ID):
        return None, "id must be a positive whole number"
    if ref_id is None and name is None:
        return None, "needs an id or a name"
    return {"id": ref_id, "name": name}, None


def _items(f: Field, value: Any) -> tuple[list[Any] | None, str | None]:
    if isinstance(value, (str, int, dict)) and f.kind in ("codes", "refs"):
        value = [value]  # a single company or country is fine
    if not isinstance(value, (list, tuple)):
        return None, "must be a list"
    if not f.min_items <= len(value) <= f.max_items:
        return None, f"needs {f.min_items} to {f.max_items} items"
    return list(value), None


def _field(f: Field, value: Any) -> tuple[Any, str | None]:
    if f.kind == "int":
        return _number(f, value, int)
    if f.kind == "float":
        return _number(f, value, float)
    if f.kind == "text":
        text = _text(value)
        if text is None:
            return None, "must be text"
        return (text, None) if len(text) <= f.max else (None, f"must be at most {f.max:g} characters")
    if f.kind == "choice":
        return (value, None) if value in f.choices else (None, f"must be one of {', '.join(f.choices)}")
    if f.kind == "code":
        return (value, None) if isinstance(value, str) and re.fullmatch(f.pattern, value) else (None, "must be a two-letter code")
    if f.kind == "ref":
        return _ref(value)

    items, error = _items(f, value)
    if error:
        return None, error
    if f.kind == "choices":
        bad = [repr(v) for v in items if v not in f.choices]
        if bad:
            return None, f"unknown value(s): {', '.join(bad)}"
        return list(dict.fromkeys(items)), None  # drop repeats, keep order
    if f.kind == "codes":
        if any(not isinstance(v, str) or not re.fullmatch(f.pattern, v) for v in items):
            return None, "must be two-letter codes"
        return list(dict.fromkeys(items)), None
    refs = []
    for i, v in enumerate(items, 1):
        ref, error = _ref(v)
        if error:
            return None, f"item {i}: {error}"
        refs.append(ref)
    return refs, None


def validate(form: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Check a creator form against FIELDS.

    Returns (clean_form, errors). clean_form has every field: optional fields left out become None (or [] for
    lists, "wide" for release_type); refs become {"id": int | None, "name": str | None}. Unknown fields are
    errors and never appear in clean_form. errors is a list of {"field", "message"}.
    """
    errors = [{"field": name, "message": "unknown field"} for name in form if name not in FIELDS_BY_NAME]
    clean: dict[str, Any] = {}
    for f in FIELDS:
        value = form.get(f.name)
        if _empty(value):
            if f.required:
                errors.append({"field": f.name, "message": "required"})
            clean[f.name] = [] if f.kind in ("choices", "codes", "refs") else f.default
            continue
        clean[f.name], error = _field(f, value)
        if error:
            errors.append({"field": f.name, "message": error})
    day, year, month = clean["release_day"], clean["release_year"], clean["release_month"]
    # a failed or missing year or month is None here, so a bad month gives one error, not two
    if day is not None and year is not None and month is not None and day > calendar.monthrange(year, month)[1]:
        errors.append({"field": "release_day", "message": f"day {day} does not exist in {year}-{month:02d}"})
        clean["release_day"] = None
    return clean, errors
