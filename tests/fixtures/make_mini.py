"""Build the mini shard set in tests/fixtures/mini/ from real TMDB and Wikidata answers.

python -m tests.fixtures.make_mini

Needs TMDB_READ_TOKEN (environment or .env) and the network. The output is committed; tests never call this.
Records are trimmed to what the build reads and keep a fixed `pulled_at`, so a rerun changes only what the sources changed.
"""

import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

from llm_movies.data.shards import ShardWriter
from llm_movies.data.tmdb_pull import TmdbClient
from llm_movies.data.wikidata import SparqlClient, run_pull

OUT = Path(__file__).parent / "mini"
PULLED_AT = "2026-11-04T00:00:00+00:00"  # fixed: a film released after 2026-05-06 is under 182 days old

# id -> why it is here. Order is the shard order (three shards, filled in turn).
FILMS: dict[int, str] = {
    # Christopher Nolan (director), ids 155/272/49026 are the Batman collection (first film: Batman Begins)
    77: "Memento 2000, Nolan's earliest",
    272: "Batman Begins 2005, first of the Dark Knight collection",
    155: "The Dark Knight 2008",
    27205: "Inception 2010",
    49026: "The Dark Knight Rises 2012, last train-band film in the collection",
    157336: "Interstellar 2014, calibration band",
    374720: "Dunkirk 2017, validation band",
    577922: "Tenet 2020, test band",
    872585: "Oppenheimer 2023, test band, many awards",
    # Quentin Tarantino
    680: "Pulp Fiction 1994",
    24: "Kill Bill Vol. 1 2003, a collection",
    16869: "Inglourious Basterds 2009",
    68718: "Django Unchained 2012",
    273248: "The Hateful Eight 2015, calibration band",
    466272: "Once Upon a Time in Hollywood 2019, test band, Robbie + Tarantino",
    # Greta Gerwig
    391713: "Lady Bird 2017, validation band",
    331482: "Little Women 2019",
    346698: "Barbie 2023, Gerwig + Robbie",
    # Margot Robbie: films less and more than 182 days apart
    106646: "The Wolf of Wall Street 2013, train band, Robbie's start",
    279641: "Whiskey Tango Foxtrot 2016-03, about 150 days before Suicide Squad",
    258489: "The Legend of Tarzan 2016-06, about 35 days before Suicide Squad",
    297761: "Suicide Squad 2016-08",
    389015: "I, Tonya 2017-12",
    525661: "Bombshell 2019-12, 142 days after Once Upon a Time in Hollywood",
    495764: "Birds of Prey 2020-02, 54 days after Bombshell",
    # Franchise: Toy Story (first film 862), Toy Story 4 is in the test band
    862: "Toy Story 1995, first of its collection",
    863: "Toy Story 2 1999",
    10193: "Toy Story 3 2010",
    301528: "Toy Story 4 2019",
    # Others
    475557: "Joker 2019, test band",
    496243: "Parasite 2019, non-English (Korean), many awards",
    296523: "Man Down 2016, budget 0 with revenue",
    345925: "Rupture 2016, budget 0 and revenue 0",
    1147572: "Narnia: The Magician's Nephew, budget known and revenue 0, Post Production",
    791523: "Wildwood 2026-10-19, released under 182 days before pulled_at",
    1003598: "Avengers: Secret Wars, Planned",
    1384758: "Ice Age: Boiling Point, In Production",
}

CAST_KEPT = 10
KEYWORDS_KEPT = 15
CREW_JOBS = {"Director", "Writer", "Screenplay", "Producer", "Original Music Composer"}
OTHER_COUNTRIES = 2
SHARDS = 3


def trim(record: dict) -> dict:
    out = dict(record)
    credits = record.get("credits") or {}
    cast = sorted(credits.get("cast") or [], key=lambda c: c.get("order", 10**6))[:CAST_KEPT]
    crew = [c for c in credits.get("crew") or [] if c.get("job") in CREW_JOBS]
    out["credits"] = {
        "cast": [{"id": c["id"], "name": c["name"], "order": c["order"]} for c in cast],
        "crew": [{"id": c["id"], "name": c["name"], "job": c["job"]} for c in crew],
    }
    results = (record.get("release_dates") or {}).get("results") or []
    us = [r for r in results if r.get("iso_3166_1") == "US"]
    others = sorted((r for r in results if r.get("iso_3166_1") != "US"), key=lambda r: r["iso_3166_1"])[:OTHER_COUNTRIES]
    out["release_dates"] = {"results": us + others}
    out["keywords"] = {"keywords": ((record.get("keywords") or {}).get("keywords") or [])[:KEYWORDS_KEPT]}
    for key in ("homepage", "backdrop_path", "poster_path", "spoken_languages", "origin_country"):
        out.pop(key, None)
    out["pulled_at"] = PULLED_AT
    return out


def main() -> None:
    load_dotenv()
    token = os.environ.get("TMDB_READ_TOKEN")
    if not token:
        raise SystemExit("TMDB_READ_TOKEN is not set; put it in the environment or a .env file.")
    if OUT.exists():
        shutil.rmtree(OUT)
    client = TmdbClient(token)
    records = []
    for tmdb_id, why in FILMS.items():
        record = client.movie(tmdb_id)
        if record is None:
            raise SystemExit(f"TMDB has no movie {tmdb_id} ({why})")
        records.append({**trim(record), "id": tmdb_id})
    per_shard = -(-len(records) // SHARDS)
    for n in range(SHARDS):
        with ShardWriter(OUT / "tmdb" / f"tmdb_{n:04d}.jsonl") as writer:
            for record in records[n * per_shard : (n + 1) * per_shard]:
                writer.write(record)

    person_ids = sorted({p["id"] for r in records for group in r["credits"].values() for p in group})
    sparql = SparqlClient()
    run_pull("films", list(FILMS), OUT / "wd_films", sparql)
    run_pull("people", person_ids, OUT / "wd_people", sparql)
    print(f"{len(records)} films, {len(person_ids)} people")


if __name__ == "__main__":
    main()
