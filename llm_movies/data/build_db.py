"""SQLite build: raw TMDB shards plus Wikidata shards become one database.

python -m llm_movies.data.build_db --tmdb DIR --wd-films DIR --wd-people DIR --out PATH [--pulled-at YYYY-MM-DD]

The build is deterministic (no wall-clock time is stored) and atomic: it writes `out.with_suffix(".tmp")` and replaces
`out` only when everything succeeded. `movies.split` is left NULL; the split step fills it.
"""

import argparse
import dataclasses
import json
import os
import sqlite3
from datetime import date
from pathlib import Path

from llm_movies.data.awards import award_family
from llm_movies.data.clean import MovieRow, clean_movie
from llm_movies.data.money import load_cpi, load_fx
from llm_movies.data.shards import read_shard
from llm_movies.data.wikidata import FilmFacts, PersonFacts

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
BATCH_FILMS = 5000

_JSON_COLUMNS = {"countries", "genres", "keywords"}
_BOOL_COLUMNS = {"money_suspect", "revenue_incomplete"}
_SQL_TYPES = {"int": "INTEGER", "float": "REAL", "str": "TEXT", "date": "TEXT", "bool": "INTEGER", "list": "TEXT"}

_CREW_ROLES = {
    "Director": "director",
    "Writer": "writer",
    "Screenplay": "writer",
    "Producer": "producer",
    "Original Music Composer": "composer",
}


def _movie_columns() -> list[tuple[str, str]]:
    columns = []
    for f in dataclasses.fields(MovieRow):
        text = str(f.type)
        kind = next(t for t in ("bool", "int", "float", "list", "str", "date") if t in text)
        columns.append((f.name, _SQL_TYPES[kind]))
    return columns


_MOVIE_COLUMNS = _movie_columns()

_SCHEMA = (
    "CREATE TABLE movies ("
    + ", ".join(f"{name} {kind}" + (" PRIMARY KEY" if name == "tmdb_id" else "") for name, kind in _MOVIE_COLUMNS)
    + ", split TEXT)",
    "CREATE TABLE people (id INTEGER PRIMARY KEY, name TEXT, birth TEXT)",
    "CREATE TABLE credits (movie_id INTEGER, person_id INTEGER, role TEXT, ord INTEGER, PRIMARY KEY (movie_id, person_id, role))",
    "CREATE TABLE companies (id INTEGER PRIMARY KEY, name TEXT)",
    "CREATE TABLE movie_companies (movie_id INTEGER, company_id INTEGER, ord INTEGER, PRIMARY KEY (movie_id, company_id))",
    "CREATE TABLE keywords (id INTEGER PRIMARY KEY, name TEXT)",
    "CREATE TABLE movie_keywords (movie_id INTEGER, keyword_id INTEGER, PRIMARY KEY (movie_id, keyword_id))",
    'CREATE TABLE awards (person_id INTEGER, award_qid TEXT, family TEXT, kind TEXT, "when" TEXT)',
    "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)",
)

_INDEXES = (
    "CREATE INDEX credits_person ON credits (person_id)",
    "CREATE INDEX movie_companies_company ON movie_companies (company_id)",
    "CREATE INDEX movies_release_date ON movies (release_date)",
    "CREATE INDEX awards_person ON awards (person_id)",
)

_INSERT_MOVIE = f"INSERT INTO movies VALUES ({', '.join('?' * (len(_MOVIE_COLUMNS) + 1))})"
_UPSERT_PERSON = "INSERT INTO people (id, name) VALUES (?, ?) ON CONFLICT (id) DO UPDATE SET name = excluded.name WHERE people.name IS NULL"


def _tmdb_shards(tmdb_dir: Path) -> list[Path]:
    return sorted(Path(tmdb_dir).glob("tmdb_*.jsonl"))


def _movie_values(row: MovieRow) -> tuple:
    values = []
    for name, _ in _MOVIE_COLUMNS:
        value = getattr(row, name)
        if name in _JSON_COLUMNS:
            value = json.dumps(value, sort_keys=True, ensure_ascii=False)
        elif name in _BOOL_COLUMNS:
            value = int(value)
        elif isinstance(value, date):
            value = value.isoformat()
        values.append(value)
    return (*values, None)


def _int_id(entry) -> int | None:
    value = entry.get("id") if isinstance(entry, dict) else None
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _entries(value) -> list[dict]:
    return [e for e in value if isinstance(e, dict)] if isinstance(value, list) else []


def _name(entry: dict) -> str | None:
    name = entry.get("name")
    return name.strip() or None if isinstance(name, str) else None


def _credits(raw: dict) -> list[tuple[int, str | None, str, int]]:
    """(person id, name, role, ord) for the film, one entry per (person, role) with the lowest ord."""
    credits = raw.get("credits") if isinstance(raw.get("credits"), dict) else {}
    found: dict[tuple[int, str], tuple[str | None, int]] = {}

    def add(person: dict, role: str, order: int) -> None:
        pid = _int_id(person)
        if pid is None:
            return
        key = (pid, role)
        if key not in found or order < found[key][1]:
            found[key] = (_name(person) or (found[key][0] if key in found else None), order)

    for position, person in enumerate(_entries(credits.get("cast"))):
        order = person.get("order")
        add(person, "actor", order if isinstance(order, int) and not isinstance(order, bool) else position)
    seen_in_role: dict[str, int] = {}
    for person in _entries(credits.get("crew")):
        role = _CREW_ROLES.get(person.get("job"))
        if role is None:
            continue
        add(person, role, seen_in_role.get(role, 0))
        seen_in_role[role] = seen_in_role.get(role, 0) + 1
    return [(pid, name, role, order) for (pid, role), (name, order) in found.items()]


def _wd_shard(directory: Path | None, filename: str) -> Path | None:
    """The Wikidata shard in `directory`; None means deliberately no Wikidata. A given directory must hold the shard."""
    if directory is None:
        return None
    path = Path(directory) / filename
    if not path.is_file():
        raise FileNotFoundError(f"Wikidata shard not found: {path}")
    return path


def _read_films(path: Path | None) -> tuple[dict[int, FilmFacts], int]:
    """tmdb id -> facts (`facts: null` skipped) and the number of records in the shard."""
    films, records = {}, 0
    if path is not None:
        for record in read_shard(path):
            records += 1
            if record.get("facts") is not None:
                films[record["tmdb_id"]] = FilmFacts.from_dict(record["facts"])
    return films, records


class _Buffers:
    def __init__(self, con: sqlite3.Connection):
        self.con = con
        self.sql = {
            "movies": _INSERT_MOVIE,
            "people": _UPSERT_PERSON,
            "credits": "INSERT INTO credits VALUES (?, ?, ?, ?)",
            "companies": "INSERT OR IGNORE INTO companies VALUES (?, ?)",
            "movie_companies": "INSERT INTO movie_companies VALUES (?, ?, ?)",
            "keywords": "INSERT OR IGNORE INTO keywords VALUES (?, ?)",
            "movie_keywords": "INSERT INTO movie_keywords VALUES (?, ?)",
        }
        self.rows: dict[str, list] = {table: [] for table in self.sql}

    def add(self, table: str, row: tuple) -> None:
        self.rows[table].append(row)

    def flush(self) -> None:
        for table, rows in self.rows.items():
            if rows:
                self.con.executemany(self.sql[table], rows)
                rows.clear()


def _add_film(buffers: _Buffers, raw: dict, row: MovieRow) -> None:
    movie_id = row.tmdb_id
    buffers.add("movies", _movie_values(row))
    for person_id, name, role, order in _credits(raw):
        buffers.add("people", (person_id, name))
        buffers.add("credits", (movie_id, person_id, role, order))
    linked = set()
    for position, company in enumerate(_entries(raw.get("production_companies"))):
        company_id = _int_id(company)
        if company_id is not None and company_id not in linked:
            linked.add(company_id)
            buffers.add("companies", (company_id, _name(company)))
            buffers.add("movie_companies", (movie_id, company_id, position))
    keywords = raw.get("keywords") if isinstance(raw.get("keywords"), dict) else {}
    linked = set()
    for keyword in _entries(keywords.get("keywords")):
        keyword_id = _int_id(keyword)
        if keyword_id is not None and keyword_id not in linked:
            linked.add(keyword_id)
            buffers.add("keywords", (keyword_id, _name(keyword)))
            buffers.add("movie_keywords", (movie_id, keyword_id))


def _fill_films(con, tmdb_dir, films, pulled_at, cpi, fx) -> dict:
    buffers = _Buffers(con)
    seen: set[int] = set()
    dropped = duplicates = 0
    shards = _tmdb_shards(tmdb_dir)
    if not shards:
        raise FileNotFoundError(f"no tmdb_*.jsonl shards in {tmdb_dir}")
    pending = 0
    for shard in shards:
        for raw in read_shard(shard):
            tmdb_id = raw.get("id")
            if tmdb_id is None:
                raise ValueError(f"{shard.name}: a record has no id")
            if tmdb_id in seen:
                duplicates += 1
                continue
            seen.add(tmdb_id)
            row = clean_movie(raw, films.get(tmdb_id), cpi, fx, pulled_at)
            if row is None:
                dropped += 1
                continue
            _add_film(buffers, raw, row)
            pending += 1
            if pending >= BATCH_FILMS:
                buffers.flush()
                pending = 0
    buffers.flush()
    return {"dropped": dropped, "duplicates": duplicates}


def _fill_people(con, path: Path | None) -> int:
    """Stream the people shard once; only people with a credit get a birth date and awards. Returns the record count."""
    if path is None:
        return 0
    kept = {r[0] for r in con.execute("SELECT id FROM people")}
    records, births, awards, done = 0, [], set(), set()
    for record in read_shard(path):
        records += 1
        person_id = record.get("tmdb_id")
        if record.get("facts") is None or person_id not in kept or person_id in done:
            continue
        done.add(person_id)
        facts = PersonFacts.from_dict(record["facts"])
        if facts.birth is not None:
            births.append((facts.birth.isoformat(), person_id))
        for event in facts.awards:
            family = award_family(event.award_qid, set())
            if family is not None:
                awards.add((person_id, event.award_qid, family, event.kind, event.when.isoformat() if event.when else None))
    con.executemany("UPDATE people SET birth = ? WHERE id = ?", births)
    con.executemany(
        'INSERT INTO awards (person_id, award_qid, family, kind, "when") VALUES (?, ?, ?, ?, ?)',
        sorted(awards, key=lambda a: (a[0], a[1], a[3], a[4] or "")),
    )
    return records


def build_db(
    tmdb_dir: Path,
    wd_films_dir: Path | None,
    wd_people_dir: Path | None,
    out: Path,
    pulled_at: date,
    cpi: dict | None = None,
    fx: dict | None = None,
) -> dict:
    """Build the database at `out` and return its counts. CPI and FX default to the files in data/.

    `out` must be on local disk: SQLite is never built on Google Drive. A Wikidata directory of None means deliberately
    no Wikidata; a given directory without its shard raises FileNotFoundError. The Wikidata record counts are in the
    counts and in `meta`, so a thin build is visible."""
    out = Path(out)
    cpi = cpi if cpi is not None else load_cpi(DATA_DIR / "cpi_us_annual.csv")
    fx = fx if fx is not None else load_fx(DATA_DIR / "fx_usd_annual.csv")
    films_path = _wd_shard(wd_films_dir, "wd_films.jsonl")
    people_path = _wd_shard(wd_people_dir, "wd_people.jsonl")
    films, wd_films = _read_films(films_path)

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp)
    try:
        con.execute("PRAGMA journal_mode = OFF")
        con.execute("PRAGMA synchronous = OFF")
        for statement in _SCHEMA:
            con.execute(statement)
        counts = _fill_films(con, tmdb_dir, films, pulled_at, cpi, fx)
        wd_people = _fill_people(con, people_path)
        for statement in _INDEXES:
            con.execute(statement)
        for table in ("movies", "people", "credits", "companies", "keywords", "awards"):
            counts[table] = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        counts["wd_films"], counts["wd_people"] = wd_films, wd_people
        meta = {"pulled_at": pulled_at.isoformat(), **{k: str(v) for k, v in counts.items()}}
        con.executemany("INSERT INTO meta VALUES (?, ?)", sorted(meta.items()))
        con.commit()
        con.close()
        os.replace(tmp, out)
    except BaseException:
        con.close()
        tmp.unlink(missing_ok=True)
        raise
    return counts


def latest_pulled_at(tmdb_dir: Path) -> date | None:
    """The latest `pulled_at` date across the TMDB shards (one pass), or None if no record has one."""
    latest = None
    for shard in _tmdb_shards(tmdb_dir):
        for raw in read_shard(shard):
            stamp = raw.get("pulled_at")
            if isinstance(stamp, str):
                try:
                    day = date.fromisoformat(stamp[:10])
                except ValueError:
                    continue
                latest = day if latest is None or day > latest else latest
    return latest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the SQLite database from TMDB and Wikidata shards.")
    parser.add_argument("--tmdb", type=Path, required=True)
    parser.add_argument("--wd-films", type=Path, required=True)
    parser.add_argument("--wd-people", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pulled-at", type=date.fromisoformat, help="YYYY-MM-DD; default: the latest pulled_at in the TMDB shards")
    args = parser.parse_args(argv)
    pulled_at = args.pulled_at or latest_pulled_at(args.tmdb)
    if pulled_at is None:
        raise SystemExit("No pulled_at found in the TMDB shards; pass --pulled-at YYYY-MM-DD.")
    print(json.dumps(build_db(args.tmdb, args.wd_films, args.wd_people, args.out, pulled_at)))


if __name__ == "__main__":
    main()
