"""Track records for people, companies and distributors, from earlier films only.

    History.from_db(con, cfg=None)                  -> every film's dated credits, loaded once
    history.person(person_id, role, as_of)          -> PersonRecord; role: director, actor, writer, producer, composer
    history.company(company_id, as_of)              -> PersonRecord, from movie_companies
    history.distributor(name, as_of)                -> PersonRecord, from movies.distributor (casefolded, trimmed)
    history.typical(as_of)                          -> (median log revenue, median score, median log budget) of earlier films
    history.resolve(ref, role)                      -> id for a form reference (int, str, {"id", "name"} or None), or None

Leak-safety rules (`as_of` is a date; day numbers are used inside):
- A film counts only if released strictly before `as_of`: a film never counts for itself, nor does a same-day film.
- An earlier film's revenue and score count once `release + cfg.mature_days <= as_of` (exactly 182 days counts); before
  that it adds to `films` only. Its budget counts from strictly after its release (the same strictly-before rule).
- `typical(as_of)` follows the same timing over all earlier films, so later films change nothing.
- Values are shrunk medians towards the typical film: (n * median + cfg.shrink_films * typical) / (n + cfg.shrink_films),
  with n the number of usable values (not `films`); with n == 0 the value is the typical one.
- Usable: log revenue needs revenue_2025, not money_suspect, not revenue_incomplete; score needs vote_count >=
  cfg.min_votes and a vote_average; log budget needs budget_2025 and not money_suspect. Natural log, 2025 dollars.
- Each film counts once per track (a person in one role, a company, a distributor), however many credits it has.

Where no earlier film has a value, `typical` (and so a new record's value) is NaN, as in the app; nothing raises.
A record with `films > 0` can also carry NaN, when no earlier film has a usable value of that kind yet (for example a
first film that is still under 182 days old, with no other film in the database to give a typical value). Consumers must
handle NaN on any record, not only on new ones.

Known limitation: an earlier film's score is its current TMDB rating, not its rating at `as_of`.

Storage is packed: for every kind of track, one set of flat numpy arrays sorted by (key, day), instead of an object per
person (the app needed this at about 350k tracks; this database has several times more).
"""

import heapq
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from llm_movies.config import DataConfig, load_config

ROLES = ("director", "actor", "writer", "producer", "composer")
_YEAR_DAYS = 365.25
_EPOCH_ORDINAL = date(1970, 1, 1).toordinal()


@dataclass(frozen=True)
class PersonRecord:
    films: int  # earlier films in the role (all of them, usable values or not)
    years_active: int  # whole years since the first earlier film; 0 when new
    revenue_log: float  # shrunk median log revenue (2025 dollars)
    score: float  # shrunk median score
    budget_log: float  # shrunk median log budget (2025 dollars)
    is_new: bool  # no earlier film


def _day(when: date) -> int:
    return when.toordinal() - _EPOCH_ORDINAL


# ---------------------------------------------------------------- packed storage


@dataclass
class _Values:
    """One kind of value for every track, packed: track t's values are values[offsets[t]:offsets[t + 1]], by day."""

    days: np.ndarray  # int32
    values: np.ndarray  # float64
    offsets: np.ndarray  # int64, tracks + 1

    @classmethod
    def pack(cls, track_of_row: np.ndarray, n_tracks: int, days: np.ndarray, values: np.ndarray) -> "_Values":
        ok = ~np.isnan(values)
        counts = np.bincount(track_of_row[ok], minlength=n_tracks)
        return cls(days[ok], values[ok], np.concatenate([[0], np.cumsum(counts)]).astype(np.int64))

    def until(self, track: int, day: int, side: str) -> np.ndarray:
        """The track's values dated before `day` (side "left") or on or before it (side "right")."""
        s, e = self.offsets[track], self.offsets[track + 1]
        return self.values[s : s + int(np.searchsorted(self.days[s:e], day, side))]


class _Tracks:
    """The films of every key (a person in one role, a company, a distributor), packed and sorted by (key, day)."""

    def __init__(self, keys: np.ndarray, film: np.ndarray, films: "_Films") -> None:
        day = films.day[film]
        order = np.lexsort((film, day, keys))  # key, then day, then film: a repeated (key, film) pair is adjacent
        keys, film, day = keys[order], film[order], day[order]
        keep = np.ones(len(keys), dtype=bool)
        keep[1:] = (keys[1:] != keys[:-1]) | (film[1:] != film[:-1])
        keys, film, day = keys[keep], film[keep], day[keep]
        new = np.ones(len(keys), dtype=bool)
        new[1:] = keys[1:] != keys[:-1]
        starts = np.flatnonzero(new)
        track_of_row = np.cumsum(new) - 1
        self.keys = keys[starts]
        self.offsets = np.append(starts, len(keys)).astype(np.int64)
        self.days = day
        self.revenue = _Values.pack(track_of_row, len(starts), day, films.log_revenue[film])
        self.score = _Values.pack(track_of_row, len(starts), day, films.score[film])
        self.budget = _Values.pack(track_of_row, len(starts), day, films.log_budget[film])

    def find(self, key) -> int | None:
        """The index of the key's track, or None if the key has no film."""
        if key is None:
            return None
        i = int(np.searchsorted(self.keys, key))
        return i if i < len(self.keys) and self.keys[i] == key else None

    def films_of(self) -> np.ndarray:
        return np.diff(self.offsets)


class _NameMap:
    """Exact lookups of a casefolded name -> id, packed: names in one sorted bytes blob, found by binary search.
    A dict of millions of names costs hundreds of MB; this costs about the size of the names."""

    def __init__(self, keys: Iterable[str], ids: Iterable[int]) -> None:
        encoded, ids = [k.encode("utf-8") for k in keys], list(ids)
        order = sorted(range(len(encoded)), key=encoded.__getitem__)
        self.blob = b"".join(encoded[i] for i in order)
        self.offsets = np.concatenate([[0], np.cumsum([len(encoded[i]) for i in order])]).astype(np.int64)
        self.ids = np.array([ids[i] for i in order], dtype=np.int64)

    def _key(self, i: int) -> bytes:
        return self.blob[self.offsets[i] : self.offsets[i + 1]]

    def get(self, key: str) -> int | None:
        target, lo, hi = key.encode("utf-8"), 0, len(self.ids)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._key(mid) < target:
                lo = mid + 1
            else:
                hi = mid
        return int(self.ids[lo]) if lo < len(self.ids) and self._key(lo) == target else None


class _ExpandingMedian:
    """Median of all values that became available on or before a day, for any day."""

    def __init__(self, available: np.ndarray, values: np.ndarray) -> None:
        order = np.argsort(available, kind="stable")
        available, values = available[order].tolist(), values[order].tolist()
        low: list[float] = []  # max-heap (negated) of the lower half
        high: list[float] = []  # min-heap of the upper half
        days, medians = [], []
        for i, (day, v) in enumerate(zip(available, values)):
            if not low or v <= -low[0]:
                heapq.heappush(low, -v)
            else:
                heapq.heappush(high, v)
            if len(low) > len(high) + 1:
                heapq.heappush(high, -heapq.heappop(low))
            elif len(high) > len(low):
                heapq.heappush(low, -heapq.heappop(high))
            if i + 1 == len(available) or available[i + 1] != day:  # last value of this day
                days.append(day)
                medians.append(-low[0] if len(low) > len(high) else (-low[0] + high[0]) / 2)
        self.days, self.medians = np.array(days, dtype=np.int64), np.array(medians)

    def at(self, day: int) -> float:
        i = int(np.searchsorted(self.days, day, "right")) - 1
        return float(self.medians[i]) if i >= 0 else float("nan")


@dataclass
class _Films:
    """Per film (same order): release day and the values it may contribute (NaN when unusable)."""

    day: np.ndarray  # int32
    log_revenue: np.ndarray
    score: np.ndarray
    log_budget: np.ndarray


def _flag(column: pd.Series) -> np.ndarray:
    return column.fillna(0).to_numpy(dtype=np.int64) != 0


def _log_where(amount: np.ndarray, usable: np.ndarray) -> np.ndarray:
    out = np.full(len(amount), np.nan)
    ok = usable & (amount > 0)
    out[ok] = np.log(amount[ok])
    return out


def _films_from(movies: pd.DataFrame, cfg: DataConfig) -> _Films:
    day = pd.to_datetime(movies["release_date"], format="%Y-%m-%d").to_numpy().astype("datetime64[D]").astype(np.int64)
    suspect = _flag(movies["money_suspect"])
    revenue = movies["revenue_2025"].to_numpy(dtype=float)
    budget = movies["budget_2025"].to_numpy(dtype=float)
    votes = movies["vote_count"].to_numpy(dtype=float)
    average = movies["vote_average"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        score_ok = (votes >= cfg.min_votes) & ~np.isnan(average)
    return _Films(
        day=day.astype(np.int32),
        log_revenue=_log_where(revenue, ~suspect & ~_flag(movies["revenue_incomplete"])),
        score=np.where(score_ok, average, np.nan),
        log_budget=_log_where(budget, ~suspect),
    )


def _normalised(names: pd.Series) -> pd.Series:
    """Trimmed and casefolded; empty becomes missing."""
    text = names.str.strip().str.casefold()
    return text.where(text != "")


# ---------------------------------------------------------------- track records


class History:
    def __init__(
        self, cfg: DataConfig, films: _Films, tracks: dict[str, _Tracks], names: dict[str, _NameMap], distributors: dict[str, int]
    ):
        self.cfg = cfg
        self._tracks = tracks  # role names, "company" and "distributor"
        self._names = names  # role names and "company"
        self._distributor_codes = distributors
        mature, delay_budget = cfg.mature_days, 1  # budget is usable from the day after release
        self._revenue = _ExpandingMedian(*self._available(films, films.log_revenue, mature))
        self._score = _ExpandingMedian(*self._available(films, films.score, mature))
        self._budget = _ExpandingMedian(*self._available(films, films.log_budget, delay_budget))

    @staticmethod
    def _available(films: _Films, values: np.ndarray, delay: int) -> tuple[np.ndarray, np.ndarray]:
        have = ~np.isnan(values)
        return films.day[have].astype(np.int64) + delay, values[have]

    @classmethod
    def from_db(cls, con: sqlite3.Connection, cfg: DataConfig | None = None) -> "History":
        cfg = cfg if cfg is not None else load_config()
        movies = pd.read_sql_query(
            "SELECT tmdb_id, release_date, budget_2025, revenue_2025, money_suspect, revenue_incomplete, vote_count, vote_average, "
            "distributor FROM movies WHERE release_date IS NOT NULL",
            con,
        )
        films = _films_from(movies, cfg)
        people = pd.read_sql_query("SELECT id, name FROM people WHERE name IS NOT NULL", con)  # read once, used by every role
        ids = movies["tmdb_id"].to_numpy(dtype=np.int64)
        by_id = np.argsort(ids, kind="stable")
        sorted_ids = ids[by_id]

        def positions(movie_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """Film position of each movie id, and which of them are films at all."""
            at = np.minimum(np.searchsorted(sorted_ids, movie_ids), max(len(sorted_ids) - 1, 0))
            known = (sorted_ids[at] == movie_ids) if len(sorted_ids) else np.zeros(len(movie_ids), dtype=bool)
            return by_id[at][known] if len(sorted_ids) else at[:0], known

        tracks: dict[str, _Tracks] = {}
        names: dict[str, _NameMap] = {}
        for role in ROLES:
            frame = pd.read_sql_query("SELECT movie_id, person_id FROM credits WHERE role = ?", con, params=(role,))
            film, known = positions(frame["movie_id"].to_numpy(dtype=np.int64))
            tracks[role] = _Tracks(frame["person_id"].to_numpy(dtype=np.int64)[known], film, films)
            names[role] = cls._name_map(people, tracks[role])
        frame = pd.read_sql_query("SELECT movie_id, company_id FROM movie_companies", con)
        film, known = positions(frame["movie_id"].to_numpy(dtype=np.int64))
        tracks["company"] = _Tracks(frame["company_id"].to_numpy(dtype=np.int64)[known], film, films)
        companies = pd.read_sql_query("SELECT id, name FROM companies WHERE name IS NOT NULL", con)
        names["company"] = cls._name_map(companies, tracks["company"])

        codes, uniques = pd.factorize(_normalised(movies["distributor"]))
        has = codes >= 0
        tracks["distributor"] = _Tracks(codes[has].astype(np.int64), np.flatnonzero(has), films)
        return cls(cfg, films, tracks, names, {name: i for i, name in enumerate(uniques)})

    @staticmethod
    def _name_map(named: pd.DataFrame, tracks: _Tracks) -> _NameMap:
        """Casefolded name -> the id with the most films in the track, ties to the lowest id."""
        counts = pd.DataFrame({"id": tracks.keys, "n": tracks.films_of()})
        named = counts.merge(named, on="id")
        named["key"] = _normalised(named["name"])
        named = named.dropna(subset=["key"]).sort_values(["key", "n", "id"], ascending=[True, False, True])
        named = named.drop_duplicates("key")
        return _NameMap(named["key"], named["id"])

    # ------------------------------------------------------------ queries

    def typical(self, as_of: date) -> tuple[float, float, float]:
        """(median log revenue, median score, median log budget) of the earlier films, by the same timing as a record.
        NaN for a value that no earlier film has."""
        return self._typical(_day(as_of))

    def _typical(self, day: int) -> tuple[float, float, float]:
        return (self._revenue.at(day), self._score.at(day), self._budget.at(day))

    def person(self, person_id: int | None, role: str, as_of: date) -> PersonRecord:
        if role not in ROLES:
            raise ValueError(f"unknown role {role!r}; expected one of {ROLES}")
        return self._record(self._tracks[role], self._tracks[role].find(person_id), _day(as_of))

    def company(self, company_id: int | None, as_of: date) -> PersonRecord:
        return self._record(self._tracks["company"], self._tracks["company"].find(company_id), _day(as_of))

    def distributor(self, name: str | None, as_of: date) -> PersonRecord:
        code = self._distributor_codes.get(name.strip().casefold()) if isinstance(name, str) else None
        return self._record(self._tracks["distributor"], self._tracks["distributor"].find(code), _day(as_of))

    def _record(self, tracks: _Tracks, track: int | None, day: int) -> PersonRecord:
        typical = self._typical(day)
        n = 0
        if track is not None:
            s, e = tracks.offsets[track], tracks.offsets[track + 1]
            n = int(np.searchsorted(tracks.days[s:e], day, "left"))  # strictly before as_of
        if n == 0:
            return PersonRecord(0, 0, typical[0], typical[1], typical[2], True)
        mature = day - self.cfg.mature_days  # release + mature_days <= as_of
        k = self.cfg.shrink_films
        return PersonRecord(
            films=n,
            years_active=int((day - int(tracks.days[s])) / _YEAR_DAYS),
            revenue_log=_shrunk(tracks.revenue.until(track, mature, "right"), typical[0], k),
            score=_shrunk(tracks.score.until(track, mature, "right"), typical[1], k),
            budget_log=_shrunk(tracks.budget.until(track, day, "left"), typical[2], k),
            is_new=False,
        )

    def resolve(self, ref: int | str | dict | None, role: str) -> int | None:
        """The id for a form reference. A known id in the role wins; an unknown id falls back to the name; a name matches
        exactly after trim and casefold (accents matter), the id with the most films in the role winning, ties the lowest
        id. Unknown gives None (a new person). `role` may also be "company", matched on companies.name.
        The tie-break counts all films, regardless of date. It only chooses which id a name means; it never changes the
        values of any record."""
        if role not in ROLES and role != "company":
            raise ValueError(f"unknown role {role!r}; expected one of {ROLES + ('company',)}")
        if isinstance(ref, dict):
            ref_id, name = ref.get("id"), ref.get("name")
        elif isinstance(ref, str):
            ref_id, name = None, ref
        else:
            ref_id, name = ref, None
        if ref_id is not None:
            try:
                ref_id = int(ref_id)
            except (TypeError, ValueError):
                ref_id = None
            if ref_id is not None and self._tracks[role].find(ref_id) is not None:
                return ref_id
        key = name.strip().casefold() if isinstance(name, str) else ""
        return self._names[role].get(key) if key else None


def _shrunk(values: np.ndarray, typical: float, shrink_films: int) -> float:
    n = len(values)
    if n == 0:
        return typical
    return (n * float(np.median(values)) + shrink_films * typical) / (n + shrink_films)
