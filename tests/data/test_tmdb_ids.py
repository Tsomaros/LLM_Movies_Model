from datetime import date
from pathlib import Path

import pytest

from llm_movies.data.tmdb_ids import chunk_ids, export_url, read_ids

SAMPLE = Path(__file__).parent.parent / "fixtures" / "tmdb_ids_sample.json.gz"


def test_export_url_zero_pads_month_and_day():
    assert export_url(date(2026, 1, 5)) == "https://files.tmdb.org/p/exports/movie_ids_01_05_2026.json.gz"


def test_read_ids_skips_adult_and_video():
    # Fixture: 3 plain films, 1 real video:true, hand-written adult:true, video:true, absent flags,
    # an out-of-order id and a duplicate of 3924. Missing adult/video count as false.
    assert read_ids(SAMPLE) == [3924, 8773, 25449, 900000, 900003, 900004]


def test_chunk_ids_splits_in_order_with_short_last_chunk():
    assert chunk_ids([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]


def test_chunk_ids_empty_and_exact_multiple():
    assert chunk_ids([], 3) == []
    assert chunk_ids([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]


def test_chunk_ids_rejects_non_positive_size():
    with pytest.raises(ValueError):
        chunk_ids([1], 0)
