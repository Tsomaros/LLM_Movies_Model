"""Wikidata enrichment: facts TMDB lacks (distributors, writers, source material, box office, awards).

python -m llm_movies.data.wikidata films|people --ids PATH --out DIR
"""

import argparse
import calendar
import math
import re
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal, Self

import requests

from llm_movies.data.money import clean_amount, to_usd
from llm_movies.data.shards import ShardWriter, done_keys
from llm_movies.data.tmdb_ids import read_ids
from llm_movies.data.tmdb_pull import MAX_BACKOFF, TmdbClient

ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "LLMMoviesPredictionModel/0.1 (https://github.com/Tsomaros/LLM_Movies_Model)"
BATCH = 500
PAUSE_S = 1.0
TIMEOUT_S = 70  # WDQS cuts a query off at 60 s

# Wikidata item "worldwide": a special value for "place of publication" and "operating area" (verified on wikidata.org).
WORLDWIDE_QID = "Q13780930"

# Wikidata unit item -> ISO 4217 code, each verified live on query.wikidata.org (label and ISO 4217 code, P498).
# An unknown unit keeps its QID as the currency and no unit gives None, so `to_usd` returns None and the figure is ignored.
CURRENCY_BY_UNIT = {
    "Q4917": "USD",  # United States dollar
    "Q4916": "EUR",  # euro
    "Q25224": "GBP",  # pound sterling
    "Q8146": "JPY",  # yen
    "Q80524": "INR",  # Indian rupee
    "Q39099": "CNY",  # renminbi
    "Q202040": "KRW",  # won
    "Q259502": "AUD",  # Australian dollar
    "Q1104069": "CAD",  # Canadian dollar
    "Q184172": "FRF",  # French franc
    "Q16068": "DEM",  # Deutsche Mark
    "Q204992": "ITL",  # Italian lira
    "Q189097": "ESP",  # peseta
}

# ---- queries ----
# Every query returns the same columns, one value per row (UNION branches, never chained OPTIONALs, which multiply
# writers x producers x distributors x money and hit WDQS's 60 s timeout on 500 films).
_COLUMNS = "?tmdb ?item ?kind ?v ?n ?unit ?place ?date ?prec"

_FILM_BRANCHES = [
    '{ BIND("item" AS ?kind) }',
    # source material: the P144 target's classes (P31) and how many Wikipedias cover it
    '{ ?item wdt:P144 ?src . ?src wikibase:sitelinks ?n . OPTIONAL { ?src wdt:P31 ?v } BIND("source" AS ?kind) }',
    '{ ?item wdt:P750 ?d . ?d rdfs:label ?v . FILTER(LANG(?v) = "en") BIND("distributor" AS ?kind) }',
    '{ ?item wdt:P58 ?v . BIND("writer" AS ?kind) }',
    '{ ?item wdt:P162 ?v . BIND("producer" AS ?kind) }',
    '{ ?item wdt:P86 ?v . BIND("composer" AS ?kind) }',
    '{ ?item wdt:P179 ?v . BIND("series" AS ?kind) }',
    '{ ?item wdt:P155 ?v . BIND("follows" AS ?kind) }',
    # money through statement nodes, so unit, place, date and date precision are all available; p:/ps:/psv: include
    # deprecated-rank statements (wdt: does not), so those are filtered out in the money and award branches
    (
        "{ ?item p:P2142 ?st . ?st psv:P2142 ?amt . ?amt wikibase:quantityAmount ?v ; wikibase:quantityUnit ?unit . "
        "?st wikibase:rank ?r . FILTER(?r != wikibase:DeprecatedRank) "
        "OPTIONAL { ?st pq:P3005 ?place } "
        "OPTIONAL { ?st pqv:P585 ?dv . ?dv wikibase:timeValue ?date ; wikibase:timePrecision ?prec } "
        'BIND("box_office" AS ?kind) }'
    ),
    (
        "{ ?item p:P2130 ?st . ?st psv:P2130 ?amt . ?amt wikibase:quantityAmount ?v ; wikibase:quantityUnit ?unit . "
        "?st wikibase:rank ?r . FILTER(?r != wikibase:DeprecatedRank) "
        "OPTIONAL { ?st pqv:P585 ?dv . ?dv wikibase:timeValue ?date ; wikibase:timePrecision ?prec } "
        'BIND("cost" AS ?kind) }'
    ),
]

_PEOPLE_BRANCHES = [
    '{ BIND("item" AS ?kind) }',
    '{ ?item wdt:P569 ?v . BIND("birth" AS ?kind) }',
    (
        "{ ?item p:P166 ?st . ?st ps:P166 ?v . "
        "?st wikibase:rank ?r . FILTER(?r != wikibase:DeprecatedRank) "
        "OPTIONAL { ?st pqv:P585 ?dv . ?dv wikibase:timeValue ?date ; wikibase:timePrecision ?prec } "
        'BIND("won" AS ?kind) }'
    ),
    (
        "{ ?item p:P1411 ?st . ?st ps:P1411 ?v . "
        "?st wikibase:rank ?r . FILTER(?r != wikibase:DeprecatedRank) "
        "OPTIONAL { ?st pqv:P585 ?dv . ?dv wikibase:timeValue ?date ; wikibase:timePrecision ?prec } "
        'BIND("nominated" AS ?kind) }'
    ),
]


def _query(prop: str, ids: list[int], branches: list[str]) -> str:
    values = " ".join(f'"{int(i)}"' for i in ids)  # P4947 and P4985 hold strings
    union = " UNION ".join(branches)
    return f"SELECT {_COLUMNS} WHERE {{ VALUES ?tmdb {{ {values} }} ?item wdt:{prop} ?tmdb . {union} }}"


def film_query(tmdb_ids: list[int]) -> str:
    return _query("P4947", tmdb_ids, _FILM_BRANCHES)


def people_query(tmdb_person_ids: list[int]) -> str:
    return _query("P4985", tmdb_person_ids, _PEOPLE_BRANCHES)


# ---- the facts ----


@dataclass(frozen=True)
class Money:
    amount: float | None
    currency: str | None  # ISO code; an unknown unit keeps its QID, no unit gives None
    place_qid: str | None
    year: int | None

    def to_dict(self) -> dict:
        return {"amount": self.amount, "currency": self.currency, "place_qid": self.place_qid, "year": self.year}

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        return cls(d["amount"], d["currency"], d["place_qid"], d["year"])


@dataclass(frozen=True)
class AwardEvent:
    award_qid: str
    kind: Literal["won", "nominated"]
    when: date | None  # the last day the (possibly imprecise) date could mean; None when it cannot be placed

    def to_dict(self) -> dict:
        return {"award_qid": self.award_qid, "kind": self.kind, "when": self.when.isoformat() if self.when else None}

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        return cls(d["award_qid"], d["kind"], date.fromisoformat(d["when"]) if d["when"] else None)


@dataclass
class FilmFacts:
    qid: str
    source_types: set[str]
    source_sitelinks: int | None
    distributors: list[str]
    writers_qids: list[str]
    producers_qids: list[str]
    composer_qids: list[str]
    series_qid: str | None
    follows_qid: str | None
    box_office: list[Money]
    cost: list[Money]

    def to_dict(self) -> dict:
        return {
            "qid": self.qid,
            "source_types": sorted(self.source_types),
            "source_sitelinks": self.source_sitelinks,
            "distributors": list(self.distributors),
            "writers_qids": list(self.writers_qids),
            "producers_qids": list(self.producers_qids),
            "composer_qids": list(self.composer_qids),
            "series_qid": self.series_qid,
            "follows_qid": self.follows_qid,
            "box_office": [m.to_dict() for m in self.box_office],
            "cost": [m.to_dict() for m in self.cost],
        }

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        return cls(
            d["qid"],
            set(d["source_types"]),
            d["source_sitelinks"],
            list(d["distributors"]),
            list(d["writers_qids"]),
            list(d["producers_qids"]),
            list(d["composer_qids"]),
            d["series_qid"],
            d["follows_qid"],
            [Money.from_dict(m) for m in d["box_office"]],
            [Money.from_dict(m) for m in d["cost"]],
        )


@dataclass
class PersonFacts:
    birth: date | None
    awards: list[AwardEvent]

    def to_dict(self) -> dict:
        return {"birth": self.birth.isoformat() if self.birth else None, "awards": [a.to_dict() for a in self.awards]}

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        return cls(date.fromisoformat(d["birth"]) if d["birth"] else None, [AwardEvent.from_dict(a) for a in d["awards"]])


# ---- parsing ----
# Wikidata can return "unknown value" blank nodes (or .well-known/genid URIs), BCE dates and malformed literals.
# Every reader below returns None for those, and a None is simply skipped.

_QID = re.compile(r"Q[1-9]\d*")
_DATETIME = re.compile(r"(-?\d{4,})-(\d{2})-(\d{2})T")


def _cell(binding: dict, name: str) -> str | None:
    cell = binding.get(name)
    if not isinstance(cell, dict) or cell.get("type") == "bnode":
        return None
    value = cell.get("value")
    return value if isinstance(value, str) else None


def _qid(binding: dict, name: str) -> str | None:
    value = _cell(binding, name)
    if not value or ".well-known/genid" in value:
        return None
    tail = value.rsplit("/", 1)[-1]
    return tail if _QID.fullmatch(tail) else None


def _qnum(qid: str) -> int:
    return int(qid[1:])


def _number(binding: dict, name: str) -> float | None:
    try:
        number = float(_cell(binding, name))
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _int(binding: dict, name: str) -> int | None:
    number = _number(binding, name)
    return int(number) if number is not None and number == int(number) else None


def _datetime_parts(binding: dict, name: str) -> tuple[int, int, int] | None:
    """(year, month, day) of an xsd:dateTime; None for BCE, year 0, years beyond 9999 and anything malformed."""
    value = _cell(binding, name)
    match = _DATETIME.match(value) if value else None
    if not match:
        return None
    year, month, day = (int(g) for g in match.groups())
    return (year, month, day) if 1 <= year <= 9999 else None


def _birth(binding: dict) -> date | None:
    """The date as given; None when it is not a real calendar date (BCE, month or day 00, malformed)."""
    parts = _datetime_parts(binding, "v")
    try:
        return date(*parts) if parts else None
    except ValueError:
        return None


def _year(binding: dict) -> int | None:
    parts = _datetime_parts(binding, "date")
    precision = _int(binding, "prec")
    return parts[0] if parts and precision is not None and precision >= 9 else None


def _award_when(binding: dict) -> date | None:
    """The last day the date could mean, so an imprecise date never counts as earlier than it was:
    year precision (9) -> 31 December, month (10) -> the month's last day, day or finer -> that day;
    coarser than a year, or no precision -> None."""
    parts = _datetime_parts(binding, "date")
    precision = _int(binding, "prec")
    if not parts or precision is None or precision < 9:
        return None
    year, month, day = parts
    try:
        if precision == 9:
            return date(year, 12, 31)
        if precision == 10:
            return date(year, month, calendar.monthrange(year, month)[1])
        return date(year, month, day)
    except (ValueError, calendar.IllegalMonthError):
        return None


def _money(binding: dict) -> Money | None:
    amount = _number(binding, "v")
    if amount is None:
        return None
    unit = _qid(binding, "unit")
    return Money(amount, CURRENCY_BY_UNIT.get(unit, unit) if unit else None, _qid(binding, "place"), _year(binding))


def _money_key(m: Money) -> tuple:
    return (m.year is None, m.year or 0, m.amount, m.currency or "", m.place_qid or "")


def _group(sparql_json: dict) -> dict[int, tuple[str, list[dict]]]:
    """tmdb id -> (item QID, that item's rows). If two items share a TMDB id, the lowest numeric QID supplies
    everything and the others are dropped."""
    try:
        bindings = sparql_json["results"]["bindings"]
    except (KeyError, TypeError):
        return {}
    by_item: dict[int, dict[str, list[dict]]] = {}
    for b in bindings:
        if not isinstance(b, dict):
            continue
        tmdb = _int(b, "tmdb")
        item = _qid(b, "item")
        if tmdb is None or item is None:
            continue
        by_item.setdefault(tmdb, {}).setdefault(item, []).append(b)
    return {tmdb: min(items.items(), key=lambda kv: _qnum(kv[0])) for tmdb, items in by_item.items()}


def _qids(rows: list[dict], kind: str) -> set[str]:
    return {q for b in rows if _cell(b, "kind") == kind and (q := _qid(b, "v"))}


def _lowest(rows: list[dict], kind: str) -> str | None:
    found = _qids(rows, kind)
    return min(found, key=_qnum) if found else None


def parse_films(sparql_json: dict) -> dict[int, FilmFacts]:
    films = {}
    for tmdb, (qid, rows) in _group(sparql_json).items():
        sitelinks = [n for b in rows if _cell(b, "kind") == "source" and (n := _int(b, "n")) is not None]
        distributors = {label for b in rows if _cell(b, "kind") == "distributor" and (label := _cell(b, "v"))}
        money = {kind: {m for b in rows if _cell(b, "kind") == kind and (m := _money(b))} for kind in ("box_office", "cost")}
        films[tmdb] = FilmFacts(
            qid=qid,
            source_types=_qids(rows, "source"),
            source_sitelinks=max(sitelinks) if sitelinks else None,
            distributors=sorted(distributors),
            writers_qids=sorted(_qids(rows, "writer")),
            producers_qids=sorted(_qids(rows, "producer")),
            composer_qids=sorted(_qids(rows, "composer")),
            series_qid=_lowest(rows, "series"),
            follows_qid=_lowest(rows, "follows"),
            box_office=sorted(money["box_office"], key=_money_key),
            cost=sorted(money["cost"], key=_money_key),
        )
    return films


def parse_people(sparql_json: dict) -> dict[int, PersonFacts]:
    people = {}
    for tmdb, (_, rows) in _group(sparql_json).items():
        births = sorted(d for b in rows if _cell(b, "kind") == "birth" and (d := _birth(b)))
        awards = {
            AwardEvent(award, kind, _award_when(b))
            for b in rows
            if (kind := _cell(b, "kind")) in ("won", "nominated") and (award := _qid(b, "v"))
        }
        ordered = sorted(awards, key=lambda a: (a.award_qid, a.kind, a.when or date.min))
        people[tmdb] = PersonFacts(births[0] if births else None, ordered)
    return people


def pick_worldwide_usd(values: list[Money], fx: dict) -> tuple[float | None, int | None]:
    """The worldwide figure in USD and its year: place worldwide or none; the latest year wins, ties take the
    largest. An undated USD figure is usable but ranks below any dated one; an undated figure in another
    currency cannot be converted and is dropped. Nothing usable gives (None, None)."""
    best = None
    for m in values or []:
        if not isinstance(m, Money) or m.place_qid not in (None, WORLDWIDE_QID) or not m.currency:
            continue
        try:
            amount = float(m.amount)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(amount):
            continue
        if m.year is None and m.currency != "USD":
            continue
        usd = clean_amount(to_usd(amount, m.currency, m.year, fx))  # below MIN_MONEY in USD is a placeholder
        if usd is None:
            continue
        rank = (m.year is not None, m.year or 0, usd)
        if best is None or rank > best[0]:
            best = (rank, usd, m.year)
    return (best[1], best[2]) if best else (None, None)


# ---- the client ----


def _has_bindings(body) -> bool:
    return isinstance(body, dict) and isinstance(body.get("results"), dict) and isinstance(body["results"].get("bindings"), list)


class SparqlError(RuntimeError):
    """A WDQS request failed for good. `retryable` says whether a smaller batch might still succeed."""

    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class SparqlClient:
    def __init__(self, session: requests.Session | None = None, max_retries: int = 5):
        self.session = session or requests.Session()
        self.max_retries = max_retries

    def run(self, query: str) -> dict:
        """The SPARQL JSON result. 429, 5xx, connection errors and unreadable bodies are retried (Retry-After
        honoured in full, plain backoff capped); other 4xx raise at once."""
        for attempt in range(self.max_retries + 1):
            backoff = min(MAX_BACKOFF, 2.0**attempt)
            last = attempt == self.max_retries
            try:
                response = self.session.post(
                    ENDPOINT,
                    data={"query": query},
                    headers={"User-Agent": USER_AGENT, "Accept": "application/sparql-results+json"},
                    timeout=TIMEOUT_S,
                )
            except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError):
                if last:
                    raise
                time.sleep(backoff)
                continue
            status = response.status_code
            if status == 200:
                try:
                    body = response.json()
                except ValueError:
                    body = None
                if _has_bindings(body):
                    return body
                # a cut-off body or an error JSON: retry like a server error, never read as "no item"
                if last:
                    raise SparqlError("WDQS returned an unreadable body", retryable=True)
                time.sleep(backoff)
                continue
            if status != 429 and status < 500:
                raise SparqlError(f"WDQS returned HTTP {status}")
            if last:
                raise SparqlError(f"WDQS returned HTTP {status} after {self.max_retries} retries", retryable=True)
            time.sleep(TmdbClient._delay(response, backoff))
        raise AssertionError("unreachable")


# ---- the pull ----

_KINDS = {
    "films": (film_query, parse_films, "wd_films.jsonl"),
    "people": (people_query, parse_people, "wd_people.jsonl"),
}


def run_pull(kind: str, ids: list[int], out_dir: Path, client, batch: int = BATCH, pause: float = PAUSE_S) -> dict:
    """Ask Wikidata about every id not yet in the shard and write one record per id, `{"tmdb_id", "facts"}`,
    with facts None for an id that has no Wikidata item (so a resume does not ask again). A batch that still
    fails after the client's retries is halved, down to one id; a single id that still fails raises."""
    query_for, parse, filename = _KINDS[kind]
    shard = Path(out_dir) / filename
    done = done_keys(shard, "tmdb_id")
    unique = list(dict.fromkeys(ids))
    todo = [i for i in unique if i not in done]
    asked = False

    def fetch(chunk: list[int]) -> None:
        nonlocal asked
        if asked:
            time.sleep(pause)
        asked = True
        try:
            facts = parse(client.run(query_for(chunk)))
        except (SparqlError, requests.RequestException) as error:
            if len(chunk) == 1 or (isinstance(error, SparqlError) and not error.retryable):
                raise
            middle = (len(chunk) + 1) // 2
            fetch(chunk[:middle])
            fetch(chunk[middle:])
            return
        for i in chunk:
            found = facts.get(i)
            writer.write({"tmdb_id": i, "facts": found.to_dict() if found else None})

    with ShardWriter(shard) as writer:
        for start in range(0, len(todo), batch):
            fetch(todo[start : start + batch])
    return {"queried": len(todo), "skipped": len(unique) - len(todo)}


def _read_text_ids(path: Path) -> list[int]:
    return [int(token) for token in Path(path).read_text(encoding="utf-8").split()]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Pull Wikidata facts for TMDB films or people into a JSONL shard.")
    parser.add_argument("kind", choices=sorted(_KINDS))
    parser.add_argument("--ids", type=Path, required=True, help="TMDB id export (.json.gz) or a text file with one id per line")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    ids = read_ids(args.ids) if args.ids.name.endswith(".json.gz") else _read_text_ids(args.ids)
    print(run_pull(args.kind, ids, args.out, SparqlClient()))


if __name__ == "__main__":
    main()
