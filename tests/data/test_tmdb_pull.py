import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import requests

from llm_movies.data import tmdb_pull
from llm_movies.data.shards import Manifest, ShardWriter, done_keys, read_shard
from llm_movies.data.tmdb_pull import TmdbClient, TmdbError, pull_chunk

SAMPLE = Path(__file__).parent.parent / "fixtures" / "tmdb_ids_sample.json.gz"
TOKEN = "secret-token-123"


class FakeResponse:
    def __init__(self, status: int, body: dict | None = None, headers: dict | None = None):
        self.status_code = status
        self._body = body
        self.headers = headers or {}

    def json(self):
        return self._body


class FakeSession:
    """Records every call and returns scripted responses (or raises scripted exceptions) in order."""

    def __init__(self, script: list):
        self.script = list(script)
        self.calls: list[dict] = []

    def get(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def sleeps(monkeypatch):
    recorded: list[float] = []
    monkeypatch.setattr("time.sleep", recorded.append)
    return recorded


def make_client(script, **kwargs):
    session = FakeSession(script)
    kwargs.setdefault("max_per_second", float("inf"))  # no throttle sleeps unless a test asks for them
    return TmdbClient(TOKEN, session=session, **kwargs), session


# ---- TmdbClient ----


def test_movie_request_shape(sleeps):
    client, session = make_client([FakeResponse(200, {"id": 7})])
    assert client.movie(7) == {"id": 7}
    call = session.calls[0]
    assert call["url"] == "https://api.themoviedb.org/3/movie/7"
    assert call["params"] == {"append_to_response": "credits,keywords,release_dates"}
    assert call["headers"]["Authorization"] == f"Bearer {TOKEN}"
    assert call["timeout"] == 30


def test_retries_429_then_succeeds(sleeps):
    client, session = make_client([FakeResponse(429, headers={"Retry-After": "0"}), FakeResponse(200, {"id": 7})])
    assert client.movie(7) == {"id": 7}
    assert len(session.calls) == 2
    assert sleeps == [0]


def test_retry_after_seconds_is_honoured(sleeps):
    client, _ = make_client([FakeResponse(429, headers={"Retry-After": "7"}), FakeResponse(200, {"id": 7})])
    client.movie(7)
    assert sleeps == [7]


def test_non_numeric_retry_after_falls_back_to_backoff(sleeps):
    date_value = "Wed, 21 Oct 2026 07:28:00 GMT"
    client, _ = make_client([FakeResponse(429, headers={"Retry-After": date_value}), FakeResponse(200, {"id": 7})])
    assert client.movie(7) == {"id": 7}
    assert sleeps == [1]


def test_backoff_grows_exponentially(sleeps):
    client, session = make_client([FakeResponse(500)] * 4 + [FakeResponse(200, {"id": 7})], max_retries=4)
    assert client.movie(7) == {"id": 7}
    assert len(session.calls) == 5
    assert sleeps == [1, 2, 4, 8]


def test_backoff_is_capped_at_60_seconds(sleeps):
    # 2**attempt for attempt 0..7 is 1,2,4,8,16,32,64,128; the cap holds the last two at 60.
    client, _ = make_client([FakeResponse(500)] * 8 + [FakeResponse(200, {"id": 7})], max_retries=8)
    client.movie(7)
    assert sleeps == [1, 2, 4, 8, 16, 32, 60, 60]


@pytest.mark.parametrize(
    "retry_after, expected",
    # honoured in full; negative becomes 0; NaN, +inf and -inf fall back to backoff 1 s (attempt 0 = 2**0)
    [("9999", 9999), ("-5", 0), ("nan", 1), ("inf", 1), ("-inf", 1)],
)
def test_retry_after_is_honoured_in_full_negative_is_zero_nan_falls_back(sleeps, retry_after, expected):
    client, _ = make_client([FakeResponse(429, headers={"Retry-After": retry_after}), FakeResponse(200, {"id": 7})])
    client.movie(7)
    assert sleeps == [expected]


def test_retry_after_300_is_slept_in_full(sleeps):
    client, session = make_client([FakeResponse(429, headers={"Retry-After": "300"}), FakeResponse(200, {"id": 7})])
    assert client.movie(7) == {"id": 7}
    assert sleeps == [300]
    assert len(session.calls) == 2


def test_429_raises_after_max_retries(sleeps):
    client, session = make_client([FakeResponse(429, headers={"Retry-After": "0"})] * 3, max_retries=2)
    with pytest.raises(TmdbError, match="429"):
        client.movie(7)
    assert len(session.calls) == 3


def test_5xx_raises_after_max_retries(sleeps):
    client, session = make_client([FakeResponse(503)] * 3, max_retries=2)
    with pytest.raises(TmdbError) as err:
        client.movie(7)
    assert len(session.calls) == 3  # first try + 2 retries
    assert len(sleeps) == 2
    assert "503" in str(err.value)
    assert TOKEN not in str(err.value)


def test_401_raises_at_once_without_leaking_token(sleeps):
    client, session = make_client([FakeResponse(401)])
    with pytest.raises(TmdbError) as err:
        client.movie(7)
    assert len(session.calls) == 1
    assert sleeps == []
    assert "401" in str(err.value)
    assert TOKEN not in str(err.value)


def test_404_returns_none_without_retry(sleeps):
    client, session = make_client([FakeResponse(404)])
    assert client.movie(7) is None
    assert len(session.calls) == 1


@pytest.mark.parametrize(
    "exc",
    [requests.ConnectionError("down"), requests.Timeout("slow"), requests.exceptions.ChunkedEncodingError("dropped")],
)
def test_network_errors_are_retried(sleeps, exc):
    client, session = make_client([exc, FakeResponse(200, {"id": 7})])
    assert client.movie(7) == {"id": 7}
    assert len(session.calls) == 2
    assert sleeps == [1]


def test_persistent_connection_error_raises_after_max_retries(sleeps):
    client, session = make_client([requests.ConnectionError("down")] * 3, max_retries=2)
    with pytest.raises(requests.ConnectionError):
        client.movie(7)
    assert len(session.calls) == 3


def test_requests_are_spaced_by_max_per_second(monkeypatch, sleeps):
    monkeypatch.setattr("time.monotonic", lambda: 100.0)  # frozen clock: the second call must wait one interval
    client, _ = make_client([FakeResponse(200, {"id": 1}), FakeResponse(200, {"id": 2})], max_per_second=10)
    client.movie(1)
    assert sleeps == []
    client.movie(2)
    assert sleeps == [pytest.approx(0.1)]


# ---- pull_chunk ----


class FakeClient:
    """Stands in for TmdbClient: records requested ids; `missing` ids give None; raises after `fail_after` calls."""

    def __init__(self, missing=(), fail_after=None, response_id=None):
        self.requested: list[int] = []
        self.missing = set(missing)
        self.fail_after = fail_after
        self.response_id = response_id

    def movie(self, tmdb_id):
        if self.fail_after is not None and len(self.requested) >= self.fail_after:
            raise RuntimeError("session died")
        self.requested.append(tmdb_id)
        if tmdb_id in self.missing:
            return None
        return {"id": self.response_id or tmdb_id, "title": f"Film {tmdb_id}"}


def shard_ids(out: Path, chunk: int = 0) -> list[int]:
    return [r["id"] for r in read_shard(out / f"tmdb_{chunk:04d}.jsonl")]


def test_pull_chunk_writes_records_with_pulled_at_and_marks_done(tmp_path):
    manifest = Manifest(tmp_path / "manifest.json")
    stats = pull_chunk(FakeClient(missing={2}), 0, [1, 2, 3], tmp_path, manifest)
    assert stats == {"pulled": 2, "missing": 1}
    assert manifest.is_done(0)
    assert Manifest(tmp_path / "manifest.json").is_done(0)
    records = list(read_shard(tmp_path / "tmdb_0000.jsonl"))
    assert [r["id"] for r in records] == [1, 3]
    assert records[0]["title"] == "Film 1"
    stamp = datetime.fromisoformat(records[0]["pulled_at"])
    assert stamp.utcoffset() == timedelta(0)
    assert abs(datetime.now(stamp.tzinfo) - stamp) < timedelta(minutes=1)


def test_404_is_recorded_missing(tmp_path):
    first = FakeClient(missing={2})
    pull_chunk(first, 0, [1, 2, 3], tmp_path, Manifest(tmp_path / "m1.json"), mark_done=False)
    assert done_keys(tmp_path / "missing_0000.txt", "id") == {2}
    assert shard_ids(tmp_path) == [1, 3]

    second = FakeClient(missing={2})  # fresh manifest: only the missing file can stop the refetch
    stats = pull_chunk(second, 0, [1, 2, 3], tmp_path, Manifest(tmp_path / "m2.json"))
    assert second.requested == []
    assert stats == {"pulled": 2, "missing": 1}


def test_pull_chunk_resumes_without_duplicates(tmp_path):
    manifest = Manifest(tmp_path / "manifest.json")
    with pytest.raises(RuntimeError):
        pull_chunk(FakeClient(fail_after=2), 0, [1, 2, 3, 4], tmp_path, manifest)
    assert not manifest.is_done(0)
    assert shard_ids(tmp_path) == [1, 2]

    second = FakeClient()
    stats = pull_chunk(second, 0, [1, 2, 3, 4], tmp_path, manifest)
    assert second.requested == [3, 4]  # ids already written are not requested again
    assert sorted(shard_ids(tmp_path)) == [1, 2, 3, 4]
    assert stats == {"pulled": 4, "missing": 0}
    assert manifest.is_done(0)


def test_resume_after_crash_does_not_refetch_missing_ids(tmp_path):
    manifest = Manifest(tmp_path / "manifest.json")
    with pytest.raises(RuntimeError):
        pull_chunk(FakeClient(missing={2}, fail_after=3), 0, [1, 2, 3, 4], tmp_path, manifest)
    second = FakeClient(missing={2})
    stats = pull_chunk(second, 0, [1, 2, 3, 4], tmp_path, manifest)
    assert second.requested == [4]  # 1 and 3 are in the shard, 2 in the missing file
    assert sorted(shard_ids(tmp_path)) == [1, 3, 4]
    assert stats == {"pulled": 3, "missing": 1}


def test_truncated_missing_file_tail_is_refetched(tmp_path):
    with ShardWriter(tmp_path / "missing_0000.txt") as w:
        w.write({"id": 1})
    with open(tmp_path / "missing_0000.txt", "ab") as f:
        f.write(b'{"id": 2')  # crash mid-write
    client = FakeClient(missing={1, 2})
    stats = pull_chunk(client, 0, [1, 2], tmp_path, Manifest(tmp_path / "manifest.json"))
    assert client.requested == [2]
    assert done_keys(tmp_path / "missing_0000.txt", "id") == {1, 2}
    assert stats == {"pulled": 0, "missing": 2}


def test_truncated_last_line_is_refetched(tmp_path):
    shard = tmp_path / "tmdb_0000.jsonl"
    with ShardWriter(shard) as w:
        w.write({"id": 1, "title": "Film 1"})
        w.write({"id": 2, "title": "Film 2"})
    with open(shard, "ab") as f:
        f.write(b'{"id": 3, "title": "Fi')  # crash mid-write

    client = FakeClient()
    stats = pull_chunk(client, 0, [1, 2, 3], tmp_path, Manifest(tmp_path / "manifest.json"))
    assert client.requested == [3]
    assert sorted(shard_ids(tmp_path)) == [1, 2, 3]
    assert stats == {"pulled": 3, "missing": 0}


def test_done_chunk_is_skipped(tmp_path):
    manifest = Manifest(tmp_path / "manifest.json")
    manifest.mark_done(0, {"pulled": 5, "missing": 1})
    client = FakeClient()
    assert pull_chunk(client, 0, [1, 2, 3], tmp_path, manifest) == {"pulled": 0, "missing": 0}
    assert client.requested == []
    assert list(tmp_path.glob("*.jsonl")) == []  # files are not even opened for writing
    assert list(tmp_path.glob("missing_*")) == []


def test_limited_run_does_not_mark_chunk_done(tmp_path):
    manifest = Manifest(tmp_path / "manifest.json")
    limited = FakeClient()
    pull_chunk(limited, 0, [1, 2], tmp_path, manifest, mark_done=False)
    assert limited.requested == [1, 2]
    assert not manifest.is_done(0)

    full = FakeClient()
    stats = pull_chunk(full, 0, [1, 2, 3, 4], tmp_path, manifest)
    assert full.requested == [3, 4]
    assert manifest.is_done(0)
    assert stats == {"pulled": 4, "missing": 0}


def test_response_id_mismatch_is_keyed_by_requested_id(tmp_path):
    pull_chunk(FakeClient(response_id=999), 0, [5], tmp_path, Manifest(tmp_path / "manifest.json"))
    assert shard_ids(tmp_path) == [5]
    again = FakeClient()
    pull_chunk(again, 0, [5], tmp_path, Manifest(tmp_path / "other.json"))
    assert again.requested == []


# ---- CLI ----


@pytest.fixture
def no_dotenv(monkeypatch):
    monkeypatch.setattr(tmdb_pull, "load_dotenv", lambda *a, **k: None)


def test_cli_exits_without_token_before_any_network(monkeypatch, tmp_path, no_dotenv):
    monkeypatch.delenv("TMDB_READ_TOKEN", raising=False)

    def boom(*a, **k):
        raise AssertionError("network used")

    monkeypatch.setattr(requests.Session, "get", boom)
    with pytest.raises(SystemExit) as err:
        tmdb_pull.main(["--ids", str(SAMPLE), "--chunk", "0", "--out", str(tmp_path)])
    assert "TMDB_READ_TOKEN" in str(err.value)
    assert list(tmp_path.iterdir()) == []


def test_cli_chunk_out_of_range_exits_with_message(monkeypatch, tmp_path, no_dotenv):
    monkeypatch.setenv("TMDB_READ_TOKEN", TOKEN)
    with pytest.raises(SystemExit) as err:
        tmdb_pull.main(["--ids", str(SAMPLE), "--chunk", "99", "--out", str(tmp_path)])
    assert "99" in str(err.value)
    assert list(tmp_path.iterdir()) == []


def test_cli_limit_pulls_prefix_and_leaves_chunk_open(monkeypatch, tmp_path, no_dotenv, capsys):
    monkeypatch.setenv("TMDB_READ_TOKEN", TOKEN)
    client = FakeClient()
    monkeypatch.setattr(tmdb_pull, "TmdbClient", lambda token: client)
    tmdb_pull.main(["--ids", str(SAMPLE), "--chunk", "0", "--out", str(tmp_path), "--limit", "2"])
    assert client.requested == [3924, 8773]
    assert not Manifest(tmp_path / "manifest.json").is_done(0)
    assert json.loads(capsys.readouterr().out) == {"pulled": 2, "missing": 0}


def test_cli_full_run_marks_chunk_done(monkeypatch, tmp_path, no_dotenv, capsys):
    monkeypatch.setenv("TMDB_READ_TOKEN", TOKEN)
    client = FakeClient()
    monkeypatch.setattr(tmdb_pull, "TmdbClient", lambda token: client)
    tmdb_pull.main(["--ids", str(SAMPLE), "--chunk", "0", "--out", str(tmp_path)])
    assert client.requested == [3924, 8773, 25449, 900000, 900003, 900004]
    assert Manifest(tmp_path / "manifest.json").is_done(0)
    assert json.loads(capsys.readouterr().out) == {"pulled": 6, "missing": 0}
