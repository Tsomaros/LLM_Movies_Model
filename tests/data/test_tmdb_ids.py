from datetime import date
from pathlib import Path

import pytest
import requests

from llm_movies.data import tmdb_ids
from llm_movies.data.tmdb_ids import chunk_ids, download_export, export_url, read_ids

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


# ---- download of the day's export ----


class FakeResponse:
    def __init__(self, status=200, chunks=(b"abc", b"def"), fail_after=None):
        self.status_code = status
        self._chunks = chunks
        self._fail_after = fail_after

    def iter_content(self, chunk_size=None):
        for i, chunk in enumerate(self._chunks):
            if self._fail_after is not None and i >= self._fail_after:
                raise requests.ConnectionError("dropped")
            yield chunk

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def close(self):
        pass


class FakeGet:
    def __init__(self, responses):
        self.responses = responses  # url -> FakeResponse
        self.urls = []

    def __call__(self, url, **kwargs):
        self.urls.append(url)
        return self.responses.get(url, FakeResponse(404))


TODAY = date(2026, 11, 5)
URL_TODAY = export_url(TODAY)
URL_YESTERDAY = export_url(date(2026, 11, 4))


def test_download_writes_todays_export(tmp_path):
    out = tmp_path / "raw" / "ids.json.gz"
    get = FakeGet({URL_TODAY: FakeResponse(chunks=(b"abc", b"def"))})
    assert download_export(out, today=TODAY, get=get) == out
    assert out.read_bytes() == b"abcdef"
    assert get.urls == [URL_TODAY]
    assert [p.name for p in out.parent.iterdir()] == ["ids.json.gz"]


def test_download_falls_back_one_day_on_404(tmp_path):
    out = tmp_path / "ids.json.gz"
    get = FakeGet({URL_YESTERDAY: FakeResponse(chunks=(b"old",))})
    download_export(out, today=TODAY, get=get)
    assert get.urls == [URL_TODAY, URL_YESTERDAY]
    assert out.read_bytes() == b"old"


def test_download_fails_when_both_days_are_missing(tmp_path):
    out = tmp_path / "ids.json.gz"
    with pytest.raises(FileNotFoundError):
        download_export(out, today=TODAY, get=FakeGet({}))
    assert list(tmp_path.iterdir()) == []


def test_download_does_not_fall_back_on_other_errors(tmp_path):
    out = tmp_path / "ids.json.gz"
    get = FakeGet({URL_TODAY: FakeResponse(status=500)})
    with pytest.raises(requests.HTTPError):
        download_export(out, today=TODAY, get=get)
    assert get.urls == [URL_TODAY]
    assert list(tmp_path.iterdir()) == []


def test_download_skips_when_the_file_exists(tmp_path):
    out = tmp_path / "ids.json.gz"
    out.write_bytes(b"keep")
    get = FakeGet({URL_TODAY: FakeResponse()})
    download_export(out, today=TODAY, get=get)
    assert get.urls == []
    assert out.read_bytes() == b"keep"


def test_failed_download_leaves_no_half_file_and_keeps_nothing_at_path(tmp_path):
    out = tmp_path / "ids.json.gz"
    get = FakeGet({URL_TODAY: FakeResponse(chunks=(b"abc", b"def"), fail_after=1)})
    with pytest.raises(requests.ConnectionError):
        download_export(out, today=TODAY, get=get)
    assert list(tmp_path.iterdir()) == []


def test_failed_download_does_not_replace_nothing_then_retry_works(tmp_path):
    out = tmp_path / "ids.json.gz"
    with pytest.raises(requests.ConnectionError):
        download_export(out, today=TODAY, get=FakeGet({URL_TODAY: FakeResponse(fail_after=0)}))
    download_export(out, today=TODAY, get=FakeGet({URL_TODAY: FakeResponse()}))
    assert out.read_bytes() == b"abcdef"


def test_main_passes_the_path_to_download(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(tmdb_ids, "download_export", lambda path, **kw: seen.append(path))
    tmdb_ids.main(["--out", str(tmp_path / "x.json.gz")])
    assert seen == [tmp_path / "x.json.gz"]
