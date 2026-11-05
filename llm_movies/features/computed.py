"""The six inputs that are worked out from the database, not typed by the user.

    ctx = Context.from_db(con, cfg=None)                  -> everything below, loaded once; keeps no connection
    competition(ctx, release, exclude_id)                 -> big-budget films opening within 14 days either way
    installment(ctx, collection_id, release)              -> 1 + earlier entries of the collection; None without one
    collaborations(ctx, director_id, actor_ids, as_of)    -> earlier films of the director with any of the first 3 actors
    age_at(ctx, person_id, as_of)                         -> whole years by birthday; None without a birth
    prior_awards(ctx, person_ids, as_of)                  -> {"wins": n, "nominations": m} from events dated before as_of
    source_strength(ctx, sitelinks)                       -> the source work's sitelink count; None for an original
    ctx.sitelinks_of(movie_id)                            -> a film's stored sitelink count, or None

Leak-safety rules (`as_of` and `release` are dates; day numbers are used inside):
- competition is the one input that looks forward: the release calendar is known in advance, so later films count.
  A film counts when its budget_2025 is strictly above 100 million (money_suspect is ignored: it comes from revenue).
- installment, collaborations, prior_awards and age_at use only what is dated strictly before the date given: a film
  never counts for itself or for a same-day film, and an award dated on the day does not count. Undated awards never count.

Unknown ids and collections never raise: they give zero, None, or (installment of an unknown collection) 1.

Storage is packed, as in History: flat numpy arrays sorted by key and day, searched with `searchsorted`.
"""

import sqlite3
from collections.abc import Iterable
from datetime import date

import numpy as np

from llm_movies.config import DataConfig, load_config
from llm_movies.features.history import History, _day

BIG_BUDGET = 100_000_000  # 2025 dollars, strictly more
COMPETITION_DAYS = 14
ACTORS_CONSIDERED = 3
_EPOCH_ORDINAL = date(1970, 1, 1).toordinal()


def _days(texts: list[str]) -> np.ndarray:
    """ISO dates (any year 1-9999) as days since 1970-01-01."""
    return np.array(texts, dtype="datetime64[D]").astype(np.int64) if texts else np.zeros(0, dtype=np.int64)


def _grouped(keys: np.ndarray, days: np.ndarray, *extra: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """Rows sorted by (key, day): the unique keys, the offsets of each key's rows, the days and the extra columns."""
    order = np.lexsort((days, keys))
    keys, days = keys[order], days[order]
    new = np.ones(len(keys), dtype=bool)
    new[1:] = keys[1:] != keys[:-1]
    starts = np.flatnonzero(new)
    return keys[starts], np.append(starts, len(keys)).astype(np.int64), days, [e[order] for e in extra]


def _find(keys: np.ndarray, key) -> int | None:
    if key is None:
        return None
    i = int(np.searchsorted(keys, key))
    return i if i < len(keys) and keys[i] == key else None


class Context:
    """The History and the packed tables the computed inputs read. Built once by `from_db`; no connection is kept."""

    def __init__(self, history: History, movies, collections, births, awards) -> None:
        self.history = history
        self._big_days, self._big_day_of = movies["big_days"], movies["big_day_of"]
        self._sitelink_ids, self._sitelinks = movies["sitelink_ids"], movies["sitelinks"]
        self._collection_keys, self._collection_offsets, self._collection_days = collections
        self._birth_ids, self._birth_days = births
        self._award_keys, self._award_offsets, self._award_days, self._award_wins_cum, self._award_noms_cum = awards

    @classmethod
    def from_db(cls, con: sqlite3.Connection, cfg: DataConfig | None = None) -> "Context":
        cfg = cfg if cfg is not None else load_config()
        history = History.from_db(con, cfg)

        rows = con.execute(
            "SELECT tmdb_id, release_date, budget_2025, collection_id, source_sitelinks "
            "FROM movies WHERE release_date IS NOT NULL"
        ).fetchall()
        ids = np.array([r[0] for r in rows], dtype=np.int64)
        days = _days([r[1] for r in rows])
        big = np.array([r[2] is not None and r[2] > BIG_BUDGET for r in rows], dtype=bool)
        order = np.argsort(days[big], kind="stable")
        big_ids, big_days = ids[big][order], days[big][order]
        movies = {
            "big_days": big_days,
            "big_day_of": dict(zip(big_ids.tolist(), big_days.tolist())),
            "sitelink_ids": np.sort(ids),
            "sitelinks": np.array([r[4] if r[4] is not None else -1 for r in rows], dtype=np.int64)[np.argsort(ids)],
        }

        in_collection = np.array([r[3] is not None for r in rows], dtype=bool)
        collection_ids = np.array([r[3] for r in rows if r[3] is not None], dtype=np.int64)
        keys, offsets, coll_days, _ = _grouped(collection_ids, days[in_collection])
        collections = (keys, offsets, coll_days)

        born = [(r[0], r[1]) for r in con.execute("SELECT id, birth FROM people WHERE birth IS NOT NULL ORDER BY id")]
        births = (np.array([b[0] for b in born], dtype=np.int64), _days([b[1] for b in born]))

        events = con.execute(
            "SELECT person_id, kind, \"when\" FROM awards WHERE \"when\" IS NOT NULL AND person_id IS NOT NULL AND kind IN ('won', 'nominated')"
        ).fetchall()
        person = np.array([e[0] for e in events], dtype=np.int64)
        won = np.array([e[1] == "won" for e in events], dtype=np.int64)
        keys, offsets, award_days, (won,) = _grouped(person, _days([e[2] for e in events]), won)
        wins_cum = np.concatenate([[0], np.cumsum(won)])  # prefix sums, so a person's wins before a day are two lookups
        noms_cum = np.concatenate([[0], np.cumsum(1 - won)])
        awards = (keys, offsets, award_days, wins_cum, noms_cum)
        return cls(history, movies, collections, births, awards)

    def sitelinks_of(self, movie_id: int | None) -> int | None:
        """The film's stored source_sitelinks, or None (no source work, no Wikidata, or not a film).
        Known limit: it is the count today, not as of release (see `source_strength`)."""
        i = _find(self._sitelink_ids, movie_id)
        if i is None or self._sitelinks[i] < 0:
            return None
        return int(self._sitelinks[i])


def competition(ctx: Context, release: date, exclude_id: int | None) -> int:
    """Big-budget films (budget_2025 strictly above 100M) released within 14 days of `release`, either way, 14 days inclusive.
    Later films count: the calendar is known in advance. `exclude_id` (the film itself) is left out.
    `money_suspect` is deliberately not used: it is derived from revenue, and a rival opening up to 14 days later would leak
    its outcome into this film's input."""
    day = _day(release)
    low = int(np.searchsorted(ctx._big_days, day - COMPETITION_DAYS, "left"))
    high = int(np.searchsorted(ctx._big_days, day + COMPETITION_DAYS, "right"))
    count = high - low
    own = ctx._big_day_of.get(exclude_id) if exclude_id is not None else None
    if own is not None and abs(own - day) <= COMPETITION_DAYS:
        count -= 1
    return count


def installment(ctx: Context, collection_id: int | None, release: date) -> int | None:
    """1 + the entries of the collection released strictly before `release`; same-day entries do not count.
    None when `collection_id` is None.

    Known limit for training: a first film's collection membership in TMDB is usually known only because later sequels
    exist, so a first film's `collection_id` is itself a leak of the future. Task 13 must drop the franchise whenever
    `installment == 1`, in one place used by training and serving alike."""
    if collection_id is None:
        return None
    track = _find(ctx._collection_keys, collection_id)
    if track is None:
        return 1
    s, e = ctx._collection_offsets[track], ctx._collection_offsets[track + 1]
    return 1 + int(np.searchsorted(ctx._collection_days[s:e], _day(release), "left"))


def collaborations(ctx: Context, director_id: int | None, actor_ids: list[int | None], as_of: date) -> int:
    """Distinct films released strictly before `as_of` on which the director is credited as director and any of the first
    three `actor_ids` as actor. A film with two of those actors counts once. None ids are ignored (a None director gives
    0); a fourth actor never counts. The director is skipped among the actors (a director-star's own films are not
    collaborations), but he still occupies his slot: the first three stay the first three billed."""
    if director_id is None:
        return 0
    directed = ctx.history.films_before(director_id, "director", as_of)
    if len(directed) == 0:
        return 0
    actors = [a for a in actor_ids[:ACTORS_CONSIDERED] if a is not None and a != director_id]
    acted = [ctx.history.films_before(a, "actor", as_of) for a in actors]
    if not acted:
        return 0
    return len(np.intersect1d(directed, np.concatenate(acted)))  # intersect1d de-duplicates both sides


def age_at(ctx: Context, person_id: int | None, as_of: date) -> int | None:
    """Whole years by birthday on `as_of`: it goes up on the birthday itself (a 29 February birth is counted from
    1 March in a common year). None without a birth date, or when `as_of` is before the birth."""
    i = _find(ctx._birth_ids, person_id)
    if i is None:
        return None
    born = date.fromordinal(int(ctx._birth_days[i]) + _EPOCH_ORDINAL)
    if as_of < born:
        return None
    return as_of.year - born.year - ((as_of.month, as_of.day) < (born.month, born.day))


def prior_awards(ctx: Context, person_ids: Iterable[int | None], as_of: date) -> dict[str, int]:
    """Wins and nominations dated strictly before `as_of`, summed over the people (a repeated id counts once, None is
    ignored). Undated events never count; every dated event counts, also several in one night."""
    day = _day(as_of)
    wins = nominations = 0
    for person in {p for p in person_ids if p is not None}:
        track = _find(ctx._award_keys, person)
        if track is None:
            continue
        s, e = int(ctx._award_offsets[track]), int(ctx._award_offsets[track + 1])
        end = s + int(np.searchsorted(ctx._award_days[s:e], day, "left"))
        wins += int(ctx._award_wins_cum[end] - ctx._award_wins_cum[s])
        nominations += int(ctx._award_noms_cum[end] - ctx._award_noms_cum[s])
    return {"wins": wins, "nominations": nominations}


def source_strength(ctx: Context, sitelinks: int | None) -> int | None:
    """The sitelink count of the source work (how many Wikipedias cover it), passed through; None for an original.
    Known limit: the count is today's, not as of the film's release."""
    return None if sitelinks is None else int(sitelinks)
