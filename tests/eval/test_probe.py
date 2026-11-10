import copy
import sqlite3
from datetime import date

import pandas as pd
import pytest

from llm_movies.eval.probe import (
    director_probe,
    empirical_cutoff,
    load_director_films,
    load_revenue_films,
    make_generator,
    parse_money,
    revenue_probe,
)

CPI = {2010: 80.0, 2017: 90.0, 2025: 100.0}


@pytest.mark.parametrize(
    "text, expected",
    [
        ("$1.2 billion", 1.2e9),
        ("$85M", 8.5e7),
        ("1,234,567", 1234567),
        ("1.2bn", 1.2e9),
        ("85 million dollars", 8.5e7),
        ("$85 Million", 8.5e7),
        ("300K", 3e5),
        ("12 thousand", 12000),
        ("2.5mn", 2.5e6),
        ("3B", 3e9),
        (" 480", 480),
        ("about $1.2-1.5 billion", 1.2e9),  # a range's first number inherits the second one's scale
        ("1.2 to 1.5 billion", 1.2e9),
        ("$300-400 thousand", 3e5),
        ("85 - 3", 85),
        ("$300 mil", 3e8),
        ("$2 mln", 2e6),
        ("$1,200,000,000 worldwide", 1.2e9),
        ("$.5 million", 5e5),
        ("I don't know", None),
        ("two hundred", None),
        ("", None),
        ("$", None),
        ("   ", None),
        ("million", None),
        ("85 Mondays", 85),  # "M" followed by letters is not a scale
        ("12,345.67", 12345.67),
        ("1.5 Billion", 1.5e9),
        ("grossed 85M.", 8.5e7),
        ("5 B", 5e9),
        ("1980s", 1980),  # trailing letters must not make the regex give back digits
        ("10x", 10),
    ],
)
def test_parse_money_answer_table(text, expected):
    got = parse_money(text)
    if expected is None:
        assert got is None
    else:
        assert got is not None and got == pytest.approx(expected, rel=1e-12)  # exact up to float rounding


@pytest.mark.parametrize("text", ["1e999999999", "9" * 400, "\x00\u200b$", None])
def test_revenue_probe_missing_cpi_year_is_named():
    films = pd.DataFrame({"tmdb_id": [1], "title": ["A"], "year": [1999], "revenue_2025": [1e8]})
    with pytest.raises(ValueError, match="1999"):
        revenue_probe(lambda ps: ["1"] * len(ps), films, CPI)


def test_parse_money_never_raises(text):
    parse_money(text)


def test_revenue_probe_known_within_2x():
    films = pd.DataFrame({
        "tmdb_id": [1, 2, 3, 4, 5, 6], "title": ["A", "B", "C", "D", "E", "F"], "year": [2010, 2010, 2017, 2017, 2010, 2017],
        # nominal = revenue_2025 * cpi[year] / cpi[2025]: 100M*0.8=80M, 100M*0.8=80M, 200M*0.9=180M, 200M*0.9=180M,
        # 100M*0.8=80M, 200M*0.9=180M
        "revenue_2025": [1e8, 1e8, 2e8, 2e8, 1e8, 2e8],
    })
    seen = []
    answers = {
        "2010 film \"A\"": "150 million", "2010 film \"B\"": "170 million", "2017 film \"C\"": "$90M",
        "2017 film \"D\"": "no idea", "2010 film \"E\"": "$160 million", "2017 film \"F\"": "$89M",
    }

    def generate(prompts):
        seen.extend(prompts)
        return [next(v for k, v in answers.items() if k in p) for p in prompts]

    out = revenue_probe(generate, films, CPI)
    assert seen[0] == 'The 2010 film "A" earned a worldwide box office gross of $'
    assert list(out["tmdb_id"]) == [1, 2, 3, 4, 5, 6]
    assert list(out.columns) == ["tmdb_id", "answer", "guess_usd", "nominal_usd", "known"]
    assert out["nominal_usd"].tolist() == pytest.approx([8e7, 8e7, 1.8e8, 1.8e8, 8e7, 1.8e8])
    # 150M is within 2x of 80M (<=160M); 170M is not; 90M is exactly the lower 2x edge of 180M; 160M exactly the upper
    # edge of 80M; 89M is just under the lower edge; None is never known
    assert out["known"].tolist() == [True, False, True, False, True, False]
    assert out["guess_usd"].isna().tolist() == [False, False, False, True, False, False]
    assert out["guess_usd"].dropna().tolist() == [1.5e8, 1.7e8, 9e7, 1.6e8, 8.9e7]


def _directors(rows):
    return pd.DataFrame(rows, columns=["tmdb_id", "title", "year", "release_date", "directors"])


def test_director_probe_accents_folded():
    films = _directors([
        (1, "Roma", 2018, date(2018, 11, 21), ["Alfonso Cuarón"]),
        (2, "Elle", 2016, date(2016, 5, 25), ["Paul Verhoeven"]),
        (3, "Duo", 2017, date(2017, 1, 1), ["Ann Smith", "Bob Łukasz"]),
        (4, "Jr", 2012, date(2012, 1, 1), ["Robert Downey Jr."]),
        (5, "Sub", 2012, date(2012, 1, 1), ["Ann Lee"]),
        (6, "Three", 2012, date(2012, 1, 1), ["Henry Ford III"]),
        (7, "Senior", 2012, date(2012, 1, 1), ["Tom Gray Sr."]),
        (8, "Second", 2012, date(2012, 1, 1), ["Al Roe II"]),
        (9, "Reverse", 2012, date(2012, 1, 1), ["Pedro Almodovar"]),  # accent in the answer, none in the database
        (10, "Suffix", 2012, date(2012, 1, 1), ["Jo Brown Jr."]),  # the suffix is not the last name
    ])
    answers = {
        "Roma": " ALFONSO CUARON.", "Elle": " Someone Else", "Duo": " Bob Lukasz and friends", "Jr": " Downey",
        "Sub": " Leeds", "Three": " Ford", "Senior": " Gray!", "Second": " Roe", "Reverse": " Pedro Almodóvar",
        "Suffix": " Jr.",
    }
    out = director_probe(lambda ps: [next(v for k, v in answers.items() if f'"{k}"' in p) for p in ps], films)
    assert out["known"].tolist() == [True, False, True, True, False, True, True, True, True, False]  # "Leeds" lacks the word "lee"
    # no revenue, score or vote column, and the film's own year and date pass through
    assert list(out.columns) == ["tmdb_id", "year", "release_date", "answer", "known"]
    assert out["release_date"].tolist() == list(films["release_date"])


def _results(per_quarter):
    """per_quarter: {(year, quarter): (n_known, n_total)} -> one row per film."""
    rows = []
    for (year, q), (k, n) in per_quarter.items():
        for i in range(n):
            rows.append({"release_date": date(year, 3 * q - 1, 15), "known": i < k})
    return pd.DataFrame(rows)


def test_empirical_cutoff_on_synthetic_curve():
    curve = {
        (2019, 1): (8, 10), (2019, 3): (8, 10), (2020, 2): (8, 10), (2020, 4): (8, 10),  # baseline 0.8
        (2021, 1): (5, 10),  # 0.5 >= 0.4
        (2021, 2): (4, 10),  # exactly 0.4 (the threshold) counts
        (2021, 3): (1, 10),  # collapse
        (2021, 4): (0, 10),
        (2022, 1): (3, 10),  # below 0.4
    }
    assert empirical_cutoff(_results(curve)) == date(2021, 6, 30)
    # a later quarter that recovers still counts: the cutoff is the LAST passing quarter
    curve[(2022, 4)] = (6, 10)
    assert empirical_cutoff(_results(curve)) == date(2022, 12, 31)


def test_empirical_cutoff_baseline_and_quarter_ends():
    # baseline quarters average (1.0 + 0.2) / 2 = 0.6, so the threshold is 0.3; later or earlier quarters are not baseline
    base = {(2019, 1): (10, 10), (2020, 2): (2, 10), (2018, 1): (0, 10), (2021, 1): (3, 10), (2021, 3): (2, 10)}
    assert empirical_cutoff(_results(base)) == date(2021, 3, 31)  # Q1 ends on the 31st of March; Q3 (0.2) is below 0.3
    assert empirical_cutoff(_results({**base, (2021, 3): (3, 10)})) == date(2021, 9, 30)

def test_empirical_cutoff_needs_baseline():
    with pytest.raises(ValueError):
        empirical_cutoff(_results({(2015, 1): (5, 10)}))


def _mini_db(path):
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE movies (tmdb_id INTEGER PRIMARY KEY, title TEXT, release_date TEXT, year INTEGER, revenue_2025 REAL,
            money_suspect INTEGER, revenue_incomplete INTEGER, vote_count INTEGER, vote_average REAL, split TEXT);
        CREATE TABLE people (id INTEGER PRIMARY KEY, name TEXT, birth TEXT);
        CREATE TABLE credits (movie_id INTEGER, person_id INTEGER, role TEXT, ord INTEGER);
        INSERT INTO people VALUES (1, 'Dina Dir', NULL), (2, 'Eli Dir', NULL), (3, 'Wanda Wri', NULL);
        """
    )
    rows = [
        # id, title, date, year, revenue, suspect, incomplete, votes, avg, split
        (1, "Val One", "2017-05-01", 2017, 5e8, 0, 0, 900, 7.1, "validation"),
        (2, "Train One", "2010-05-01", 2010, 4e8, 0, 0, 900, 6.1, "train"),
        (3, "Test One", "2020-05-01", 2020, 3e8, 0, 0, 900, 6.5, "test"),
        (4, "Holdout One", "2024-05-01", 2024, 2e8, 0, 0, 900, 6.9, "holdout"),
        (5, "Val Suspect", "2017-06-01", 2017, 9e8, 1, 0, 900, 6.0, "validation"),
        (6, "Val Unknown", "2017-07-01", 2017, None, 0, 0, 900, 6.0, "validation"),
        (7, "Few Votes", "2012-07-01", 2012, 1e7, 0, 0, 5, 6.0, "train"),
    ]
    con.executemany("INSERT INTO movies VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    credits = [(m[0], 1, "director", 0) for m in rows] + [(1, 2, "director", 1), (1, 3, "writer", 0)]
    con.executemany("INSERT INTO credits VALUES (?,?,?,?)", credits)
    con.commit()
    con.close()
    return path


def test_load_revenue_films_reads_validation_only(tmp_path):
    db = _mini_db(tmp_path / "m.db")
    films = load_revenue_films(db, n=100)
    assert films["tmdb_id"].tolist() == [1]  # no train/test/holdout, no suspect, no unknown revenue
    assert set(films.columns) == {"tmdb_id", "title", "year", "revenue_2025"}


def test_load_director_films_has_no_targets_and_filters_votes(tmp_path):
    db = _mini_db(tmp_path / "m.db")
    films = load_director_films(db, n=100, min_votes=100)
    assert set(films.columns) == {"tmdb_id", "title", "year", "release_date", "directors"}
    assert sorted(films["tmdb_id"]) == [1, 2, 3, 4, 5, 6]  # film 7 has too few votes; any year is allowed
    assert films.set_index("tmdb_id").loc[1, "directors"] == ["Dina Dir", "Eli Dir"]  # writers are not directors
    assert len(load_director_films(db, n=2, min_votes=100)) == 2


def test_make_generator_left_pads_and_returns_only_new_text(tiny_lm, tiny_tokenizer, golden_prompts):
    tok = copy.deepcopy(tiny_tokenizer)
    tok.padding_side = "right"
    tok.pad_token = None  # a missing pad token falls back to eos
    generate = make_generator((tiny_lm, tok), "cpu", max_new_tokens=4, batch_size=2)
    short, long_ = golden_prompts[0].split()[0], " ".join(golden_prompts[0].split()[:12])
    together = generate([short, long_, short])
    assert len(together) == 3
    assert tok.padding_side == "right" and tok.pad_token is None  # the caller's tokenizer is not changed

    def alone(prompt):  # greedy continuation of one unpadded prompt, new tokens only, computed without make_generator
        enc = tiny_tokenizer(prompt, return_tensors="pt")
        ids = tiny_lm.generate(**enc, max_new_tokens=4, do_sample=False, pad_token_id=tiny_tokenizer.pad_token_id)
        return tiny_tokenizer.decode(ids[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    # left padding keeps a prompt's answer equal to its unpadded answer, whatever its batch neighbours
    assert together == [alone(short), alone(long_), alone(short)]
    # the prompt itself is not echoed: at most max_new_tokens tokens come back
    assert all(len(tiny_tokenizer(t)["input_ids"]) <= 4 for t in together)
