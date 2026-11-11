"""Memorization probes that choose the base model (spec: base model selection).

python -m llm_movies.eval.probe revenue|director --model ID --db PATH --n N --out CSV [--seed S] [--min-votes V]

    parse_money(text)                       -> float | None
    revenue_probe(generate, films, cpi)     -> tmdb_id, answer, guess_usd, nominal_usd, known
    director_probe(generate, films)         -> tmdb_id, year, release_date, answer, known
    empirical_cutoff(results)               -> date; needs a `release_date` and a boolean `known` column
    make_generator(model_id, device, ...)   -> generate(list[str]) -> list[str] (greedy, batched, left padded)
    load_revenue_films(db, n, seed)         -> validation films with a usable revenue (never test or holdout)
    load_director_films(db, n, seed, min_votes) -> any-year films with title, year, release_date, directors
    load_director_films_by_quarter(db, per_quarter, first, last, seed, min_votes) -> the same, capped per release quarter

Leak rules: the revenue probe reads only the validation split. The director probe selects id, title, release date and
director names only; `vote_count` may pick which films to ask about but is never returned.
"""

from __future__ import annotations

import argparse
import copy
import re
import sqlite3
import unicodedata
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from llm_movies.data.money import load_cpi

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
REVENUE_PROMPT = 'The {year} film "{title}" earned a worldwide box office gross of $'
DIRECTOR_PROMPT = 'The {year} film "{title}" was directed by'
BASELINE_YEARS = (2019, 2020)
CUTOFF_FRACTION = 0.5

_SCALES = {
    "thousand": 1e3, "k": 1e3, "million": 1e6, "mil": 1e6, "mln": 1e6, "m": 1e6, "mn": 1e6,
    "billion": 1e9, "b": 1e9, "bn": 1e9,
}
# The number is atomic: "1980s" is 1980 and "10x" is 10, never a shorter prefix of the digits.
_NUM = r"(?>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)"
_SCALE = r"(?:thousand|million|billion|mil|mln|mn|bn|k|m|b)(?![A-Za-z])"
_MONEY = re.compile(rf"({_NUM})(?:\s*({_SCALE}))?", re.IGNORECASE)
_RANGE_TAIL = re.compile(rf"\s*(?:-|–|—|to|or|and)\s*\$?\s*{_NUM}\s*({_SCALE})", re.IGNORECASE)
_SUFFIXES = {"jr", "sr", "ii", "iii"}


def parse_money(text: str) -> float | None:
    """The first number in `text` in dollars (commas, decimals, a leading $, scale words or suffixes); None without digits."""
    try:
        match = _MONEY.search(text or "")
        if match is None:
            return None
        value = float(match.group(1).replace(",", ""))
        scale = match.group(2)
        if scale is None:  # a range such as "1.2-1.5 billion" shares the scale of its second number
            tail = _RANGE_TAIL.match(text, match.end())
            scale = tail.group(1) if tail else None
        return value * _SCALES[scale.lower()] if scale else value
    except (ValueError, OverflowError, AttributeError, TypeError):
        return None


# letters NFKD does not decompose (no base letter plus accent)
_EXTRA_FOLD = str.maketrans({"ł": "l", "Ł": "l", "ø": "o", "Ø": "o", "đ": "d", "Đ": "d", "ð": "d", "Ð": "d", "ı": "i", "þ": "th", "Þ": "th"})


def _fold(text: str) -> str:
    """Accents removed (NFKD), casefolded (ß -> ss), and the few letters NFKD leaves alone mapped to ASCII."""
    stripped = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    return stripped.translate(_EXTRA_FOLD).casefold().replace("æ", "ae").replace("œ", "oe")


def last_name(name: str) -> str | None:
    """Last token of the name, after dropping a trailing Jr./Sr./II/III; accents folded, casefolded."""
    tokens = _fold(name).replace(",", " ").split()
    while len(tokens) > 1 and tokens[-1].strip(".") in _SUFFIXES:
        tokens.pop()
    return tokens[-1].strip(".") if tokens else None


def _generate(generate, prompts: list[str]) -> list[str]:
    answers = list(generate(prompts))
    if len(answers) != len(prompts):
        raise ValueError(f"The generator returned {len(answers)} answers for {len(prompts)} prompts.")
    return answers


def revenue_probe(
    generate: Callable[[list[str]], list[str]], films: pd.DataFrame, cpi: dict[int, float] | None = None
) -> pd.DataFrame:
    """Ask for each film's worldwide gross. `films` has tmdb_id, title, year and revenue_2025 (2025 dollars).

    `known` means the parsed guess is within 2x of the nominal (year-of-release) revenue."""
    cpi = load_cpi(DATA_DIR / "cpi_us_annual.csv") if cpi is None else cpi
    films = films.reset_index(drop=True)
    prompts = [REVENUE_PROMPT.format(year=int(r.year), title=r.title) for r in films.itertuples()]
    answers = _generate(generate, prompts)
    rows = []
    for r, answer in zip(films.itertuples(), answers):
        if int(r.year) not in cpi or 2025 not in cpi:
            raise ValueError(f"The CPI table has no entry for {int(r.year) if int(r.year) not in cpi else 2025}.")
        nominal = float(r.revenue_2025) * cpi[int(r.year)] / cpi[2025]
        guess = parse_money(answer)
        known = guess is not None and nominal / 2 <= guess <= nominal * 2
        rows.append(
            {"tmdb_id": int(r.tmdb_id), "answer": answer, "guess_usd": guess, "nominal_usd": nominal, "known": bool(known)}
        )
    return pd.DataFrame(rows, columns=["tmdb_id", "answer", "guess_usd", "nominal_usd", "known"])


def director_probe(generate: Callable[[list[str]], list[str]], films: pd.DataFrame) -> pd.DataFrame:
    """Ask who directed each film. `films` has tmdb_id, title, year, release_date and directors (a list of names).

    `known` means any director's last name appears in the answer as a word (accents folded, case-insensitive)."""
    films = films.reset_index(drop=True)
    prompts = [DIRECTOR_PROMPT.format(year=int(r.year), title=r.title) for r in films.itertuples()]
    answers = _generate(generate, prompts)
    rows = []
    for r, answer in zip(films.itertuples(), answers):
        folded = _fold(answer)
        names = [n for n in (last_name(d) for d in r.directors) if n]
        known = any(re.search(rf"(?<!\w){re.escape(n)}(?!\w)", folded) for n in names)
        rows.append(
            {"tmdb_id": int(r.tmdb_id), "year": int(r.year), "release_date": r.release_date, "answer": answer, "known": bool(known)}
        )
    return pd.DataFrame(rows, columns=["tmdb_id", "year", "release_date", "answer", "known"])


def _quarter_end(year: int, quarter: int) -> date:
    month = quarter * 3
    return date(year + month // 12, month % 12 + 1, 1) - timedelta(days=1)


def empirical_cutoff(results: pd.DataFrame) -> date:
    """Last day of the last release quarter whose recall is at least half the mean 2019-2020 quarterly recall."""
    dates = pd.to_datetime(results["release_date"])
    df = pd.DataFrame({"year": dates.dt.year, "quarter": dates.dt.quarter, "known": results["known"].astype(float)})
    recall = df.groupby(["year", "quarter"])["known"].mean()
    baseline_rows = recall[[y in BASELINE_YEARS for y, _ in recall.index]]
    if baseline_rows.empty:
        raise ValueError("empirical_cutoff needs probe results from 2019-2020 to set the baseline.")
    threshold = CUTOFF_FRACTION * float(baseline_rows.mean())
    passing = [yq for yq, value in recall.items() if value >= threshold]
    if not passing:
        raise ValueError("No release quarter reaches half the 2019-2020 recall.")
    year, quarter = max(passing)
    return _quarter_end(int(year), int(quarter))


def make_generator(
    model_id, device: str, max_new_tokens: int = 12, batch_size: int = 16
) -> Callable[[list[str]], list[str]]:
    """Greedy, batched text continuation. `model_id` is a Hub id, or a ready (model, tokenizer) pair (tests).

    Prompts are LEFT padded (decoder-only generation); only the newly generated text is returned."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if isinstance(model_id, str):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    else:
        model, tokenizer = model_id
        tokenizer = copy.deepcopy(tokenizer)  # padding side and pad token are set below: leave the caller's alone
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = model.to(device).eval()

    @torch.no_grad()
    def generate(prompts: list[str]) -> list[str]:
        out: list[str] = []
        for i in range(0, len(prompts), batch_size):
            batch = tokenizer(prompts[i:i + batch_size], return_tensors="pt", padding=True).to(device)
            ids = model.generate(
                **batch, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id
            )
            out.extend(tokenizer.batch_decode(ids[:, batch["input_ids"].shape[1]:], skip_special_tokens=True))
        return out

    return generate


def load_revenue_films(db: Path, n: int, seed: int = 0) -> pd.DataFrame:
    """Up to `n` validation-split films with a trusted revenue: tmdb_id, title, year, revenue_2025. Nothing else is read."""
    with sqlite3.connect(db) as con:
        df = pd.read_sql_query(
            "SELECT tmdb_id, title, year, revenue_2025 FROM movies WHERE split = 'validation' AND title IS NOT NULL "
            "AND revenue_2025 IS NOT NULL AND revenue_2025 > 0 AND money_suspect = 0 AND revenue_incomplete = 0 "
            "ORDER BY tmdb_id",
            con,
        )
    return df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)


def _director_films(db: Path, min_votes: int) -> pd.DataFrame:
    """Every film with a director, vote_count >= `min_votes`: tmdb_id, title, year, release_date, directors."""
    with sqlite3.connect(db) as con:
        df = pd.read_sql_query(
            "SELECT m.tmdb_id, m.title, m.year, m.release_date, p.name AS director FROM movies m "
            "JOIN credits c ON c.movie_id = m.tmdb_id AND c.role = 'director' JOIN people p ON p.id = c.person_id "
            "WHERE m.title IS NOT NULL AND m.year IS NOT NULL AND m.release_date IS NOT NULL AND p.name IS NOT NULL AND m.vote_count >= ? "
            "ORDER BY m.tmdb_id, c.ord",
            con,
            params=(min_votes,),
        )
    films = df.groupby(["tmdb_id", "title", "year", "release_date"], sort=True)["director"].apply(list)
    return films.rename("directors").reset_index()


def load_director_films(db: Path, n: int, seed: int = 0, min_votes: int = 100) -> pd.DataFrame:
    """Up to `n` films of any year: tmdb_id, title, year, release_date, directors.

    `min_votes` only picks which films to ask about; vote_count is not returned."""
    films = _director_films(db, min_votes)
    return films.sample(n=min(n, len(films)), random_state=seed).reset_index(drop=True)


def load_director_films_by_quarter(
    db: Path,
    per_quarter: int,
    first: tuple[int, int] = (2019, 1),
    last: tuple[int, int] = (2026, 2),
    seed: int = 0,
    min_votes: int = 100,
) -> pd.DataFrame:
    """Up to `per_quarter` films from each release quarter `first`..`last` (year, quarter; both included), same columns and
    leak rules as `load_director_films`. A quarter with fewer films keeps all of them. Sorted by release date and id."""
    films = _director_films(db, min_votes)
    dates = pd.to_datetime(films["release_date"], errors="coerce")
    index = dates.dt.year * 4 + dates.dt.quarter - 1
    keep = (index >= first[0] * 4 + first[1] - 1) & (index <= last[0] * 4 + last[1] - 1)
    films, index = films[keep], index[keep]
    parts = [g.sample(n=min(per_quarter, len(g)), random_state=seed) for _, g in films.groupby(index)]
    if not parts:
        return films.iloc[0:0].reset_index(drop=True)
    return pd.concat(parts).sort_values(["release_date", "tmdb_id"]).reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("kind", choices=("revenue", "director"))
    p.add_argument("--model", required=True)
    p.add_argument("--db", type=Path, required=True)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-votes", type=int, default=100)
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    generate = make_generator(args.model, device)
    if args.kind == "revenue":
        result = revenue_probe(generate, load_revenue_films(args.db, args.n, args.seed))
        print(f"revenue memorization rate (within 2x): {result['known'].mean():.3f} on {len(result)} films")
    else:
        result = director_probe(generate, load_director_films(args.db, args.n, args.seed, args.min_votes))
        print(f"director recall: {result['known'].mean():.3f} on {len(result)} films")
        print(f"empirical cutoff: {empirical_cutoff(result)}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
