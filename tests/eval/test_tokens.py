from datetime import date

import pandas as pd
import pytest

from llm_movies.eval.tokens import token_lengths
from llm_movies.prompt.render import render
from llm_movies.prompt.resolve import ResolvedFilm


def _film(synopsis):
    return ResolvedFilm(
        genres=["Drama"], release_year=2010, release_month=5, release_day=14, runtime=110, language="en",
        certification="PG-13", release_type="Wide", countries=["US"], themes=["Heist"], budget=5e7, synopsis=synopsis,
        tagline=None, source_material="Original", franchise=None, installment=None, director=None, writers=[], cast=[],
        producers=[], composer=None, distributor=None, companies=[], competition=3, collaborations=1,
        director_age=None, lead_age=None, source_strength=None,
    )


def test_token_lengths_tiny_tokenizer(tmp_path, tiny_tokenizer):
    films = [_film("A crew plans one last job. " * k) for k in (1, 2, 3, 4, 5)]
    path = tmp_path / "films.parquet"
    pd.DataFrame({
        "tmdb_id": range(5), "release_date": [date(2010, 5, 14)] * 5, "split": ["train"] * 5,
        "resolved_json": [f.to_json() for f in films],
    }).to_parquet(path)
    expected = sorted(len(tiny_tokenizer(render(f))["input_ids"]) for f in films)  # nothing hidden
    out = token_lengths(path, tiny_tokenizer)
    assert out["max"] == expected[-1] and out["p50"] == expected[2]
    assert expected[0] < out["max"]
    # linear-interpolated percentiles of 5 sorted values, worked out by hand: rank = q * 4
    assert out["p90"] == pytest.approx(expected[3] + 0.6 * (expected[4] - expected[3]))  # rank 3.6
    assert out["p99"] == pytest.approx(expected[3] + 0.96 * (expected[4] - expected[3]))  # rank 3.96
    assert out["recommended_max_len"] == -(-out["p99"] // 64) * 64  # ceil(p99 / 64) * 64, written as a negated floor
    assert out["p99"] <= out["recommended_max_len"] < out["p99"] + 64
    assert token_lengths(tmp_path, tiny_tokenizer) == out  # a folder holding films.parquet works too


def test_token_lengths_ignores_locked_file_and_rounds_up(tmp_path, tiny_tokenizer):
    open_film = _film("A crew plans one last job. ")
    n = len(tiny_tokenizer(render(open_film))["input_ids"])
    pd.DataFrame({
        "tmdb_id": [1], "release_date": [date(2010, 5, 14)], "split": ["train"], "resolved_json": [open_film.to_json()],
    }).to_parquet(tmp_path / "films.parquet")
    huge = _film("A crew plans one last job. " * 400)
    pd.DataFrame({
        "tmdb_id": [2], "release_date": [date(2020, 5, 14)], "split": ["test"], "resolved_json": [huge.to_json()],
    }).to_parquet(tmp_path / "locked.parquet")
    out = token_lengths(tmp_path, tiny_tokenizer)
    assert out["max"] == n and out["p50"] == n and out["p99"] == n  # the locked film is never read
    assert out["recommended_max_len"] == ((n + 63) // 64) * 64  # one film: a length just over a multiple goes up
