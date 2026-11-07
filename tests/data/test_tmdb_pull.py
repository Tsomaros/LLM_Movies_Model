import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
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
    tmdb_pull.main(["--ids", str(SAMPLE), "--chunk", "0", "--out", str(tmp_path), "--limit", "2", "--workers", "1"])
    assert client.requested == [3924, 8773]
    assert not Manifest(tmp_path / "manifest.json").is_done(0)
    assert json.loads(capsys.readouterr().out) == {"pulled": 2, "missing": 0}


def test_cli_full_run_marks_chunk_done(monkeypatch, tmp_path, no_dotenv, capsys):
    monkeypatch.setenv("TMDB_READ_TOKEN", TOKEN)
    client = FakeClient()
    monkeypatch.setattr(tmdb_pull, "TmdbClient", lambda token: client)
    tmdb_pull.main(["--ids", str(SAMPLE), "--chunk", "0", "--out", str(tmp_path), "--workers", "1"])
    assert client.requested == [3924, 8773, 25449, 900000, 900003, 900004]
    assert Manifest(tmp_path / "manifest.json").is_done(0)
    assert json.loads(capsys.readouterr().out) == {"pulled": 6, "missing": 0}


# ---- pull_all: every chunk in order, uploading after each ----


class FakeHub:
    """The raw repo as a dict of repo path -> bytes. `fail_on` makes the nth upload (1-based) raise."""

    def __init__(self, fail_on=None):
        self.files: dict[str, bytes] = {}
        self.commits: list[list[str]] = []
        self.attempts = 0
        self.fail_on = fail_on

    def upload(self, files):
        self.attempts += 1
        if self.attempts == self.fail_on:
            raise RuntimeError("hub down")
        self.commits.append(sorted(files))
        for name, source in files.items():
            self.files[name] = source if isinstance(source, bytes) else Path(source).read_bytes()

    def restore(self, out: Path) -> None:
        """A fresh session: download tmdb/ from the repo."""
        for name, data in self.files.items():
            target = out / name.removeprefix("tmdb/")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)


CHUNKS = [[1, 2], [3, 4], [5, 6]]


def test_pull_all_uploads_one_commit_after_each_chunk(tmp_path):
    hub, client = FakeHub(), FakeClient(missing={4})
    done = tmdb_pull.pull_all(client, CHUNKS, tmp_path, Manifest(tmp_path / "manifest.json"), hub.upload)
    assert done == [0, 1, 2] and client.requested == [1, 2, 3, 4, 5, 6]
    assert hub.commits == [
        ["tmdb/manifest.json", f"tmdb/missing_{c:04d}.txt", f"tmdb/tmdb_{c:04d}.jsonl"] for c in range(3)
    ]
    manifest = json.loads(hub.files["tmdb/manifest.json"])
    assert manifest == {"0": {"pulled": 2, "missing": 0}, "1": {"pulled": 1, "missing": 1}, "2": {"pulled": 2, "missing": 0}}
    assert json.loads((tmp_path / "manifest.json").read_text()) == manifest


def test_each_commit_carries_only_the_chunks_done_so_far(tmp_path):
    seen = []

    def upload(files):
        seen.append(sorted(json.loads(files["tmdb/manifest.json"])))

    tmdb_pull.pull_all(FakeClient(), CHUNKS, tmp_path, Manifest(tmp_path / "manifest.json"), upload)
    assert seen == [["0"], ["0", "1"], ["0", "1", "2"]]


def test_a_dead_session_then_a_rerun_gives_every_id_once(tmp_path):
    hub = FakeHub()
    first = tmp_path / "session1"
    with pytest.raises(RuntimeError, match="session died"):
        tmdb_pull.pull_all(FakeClient(fail_after=3), CHUNKS, first, Manifest(first / "manifest.json"), hub.upload)
    assert "tmdb/tmdb_0001.jsonl" not in hub.files  # the chunk in progress is the only thing lost
    # the new session starts with an empty disk and only what the repo holds
    second = tmp_path / "session2"
    hub.restore(second)
    client = FakeClient()
    done = tmdb_pull.pull_all(client, CHUNKS, second, Manifest(second / "manifest.json"), hub.upload)
    assert done == [1, 2] and client.requested == [3, 4, 5, 6]
    every_id = [i for c in range(3) for i in shard_ids(second, c)]
    assert every_id == [1, 2, 3, 4, 5, 6]
    assert sorted(json.loads(hub.files["tmdb/manifest.json"])) == ["0", "1", "2"]


def test_a_rerun_does_not_ask_again_for_an_id_that_was_a_404(tmp_path):
    # chunk 1 is [3, 4]; 3 is gone (404, recorded in the missing file) and the session dies on the call for 4
    manifest = Manifest(tmp_path / "manifest.json")
    with pytest.raises(RuntimeError, match="session died"):
        tmdb_pull.pull_all(FakeClient(missing={3}, fail_after=3), CHUNKS, tmp_path, manifest, FakeHub().upload)
    client = FakeClient(missing={3})
    tmdb_pull.pull_all(client, CHUNKS, tmp_path, Manifest(tmp_path / "manifest.json"), FakeHub().upload)
    assert client.requested == [4, 5, 6]  # 3 is not asked for again
    assert json.loads((tmp_path / "manifest.json").read_text())["1"] == {"pulled": 1, "missing": 1}


def test_rerun_in_the_same_session_does_not_duplicate(tmp_path):
    manifest = Manifest(tmp_path / "manifest.json")
    with pytest.raises(RuntimeError):
        tmdb_pull.pull_all(FakeClient(fail_after=3), CHUNKS, tmp_path, manifest, FakeHub().upload)
    client = FakeClient()
    tmdb_pull.pull_all(client, CHUNKS, tmp_path, Manifest(tmp_path / "manifest.json"), FakeHub().upload)
    assert client.requested == [4, 5, 6]
    assert [i for c in range(3) for i in shard_ids(tmp_path, c)] == [1, 2, 3, 4, 5, 6]


def test_done_chunks_are_skipped_without_a_request_or_an_upload(tmp_path):
    manifest = Manifest(tmp_path / "manifest.json")
    manifest.mark_done(0, {"pulled": 2, "missing": 0})
    hub, client = FakeHub(), FakeClient()
    done = tmdb_pull.pull_all(client, CHUNKS, tmp_path, manifest, hub.upload)
    assert done == [1, 2] and client.requested == [3, 4, 5, 6] and len(hub.commits) == 2


def test_limit_leaves_chunks_unmarked_uploads_them_and_max_chunks_stops(tmp_path):
    hub, client = FakeHub(), FakeClient()
    manifest = Manifest(tmp_path / "manifest.json")
    done = tmdb_pull.pull_all(client, CHUNKS, tmp_path, manifest, hub.upload, limit=1, max_chunks=2)
    assert done == [0, 1] and client.requested == [1, 3]
    assert hub.commits == [["tmdb/missing_0000.txt", "tmdb/tmdb_0000.jsonl"], ["tmdb/missing_0001.txt", "tmdb/tmdb_0001.jsonl"]]
    assert not (tmp_path / "manifest.json").exists() and "tmdb/manifest.json" not in hub.files
    # a later full run finishes the chunk without asking for the id it already has
    full = FakeClient()
    tmdb_pull.pull_all(full, CHUNKS, tmp_path, Manifest(tmp_path / "manifest.json"), hub.upload)
    assert full.requested == [2, 4, 5, 6]
    assert [i for c in range(3) for i in shard_ids(tmp_path, c)] == [1, 2, 3, 4, 5, 6]


def test_max_chunks_without_limit_stops_after_that_many_finished_chunks(tmp_path):
    client = FakeClient()
    done = tmdb_pull.pull_all(client, CHUNKS, tmp_path, Manifest(tmp_path / "manifest.json"), FakeHub().upload, max_chunks=1)
    assert done == [0] and client.requested == [1, 2]


def test_a_failed_upload_stops_the_run_and_the_manifest_does_not_claim_the_chunk(tmp_path):
    hub = FakeHub(fail_on=2)
    manifest = Manifest(tmp_path / "manifest.json")
    with pytest.raises(RuntimeError, match="hub down"):
        tmdb_pull.pull_all(FakeClient(), CHUNKS, tmp_path, manifest, hub.upload)
    assert json.loads((tmp_path / "manifest.json").read_text()) == {"0": {"pulled": 2, "missing": 0}}
    assert not Manifest(tmp_path / "manifest.json").is_done(1)
    # the retry pulls nothing new for chunk 1 (its ids are on disk) and uploads it
    client = FakeClient()
    tmdb_pull.pull_all(client, CHUNKS, tmp_path, Manifest(tmp_path / "manifest.json"), hub.upload)
    assert client.requested == [5, 6]
    assert sorted(json.loads(hub.files["tmdb/manifest.json"])) == ["0", "1", "2"]


def test_cli_all_needs_a_repo_and_max_chunks_needs_all(tmp_path, no_dotenv, monkeypatch):
    monkeypatch.setenv("TMDB_READ_TOKEN", TOKEN)
    with pytest.raises(SystemExit):
        tmdb_pull.main(["--ids", str(SAMPLE), "--all", "--out", str(tmp_path)])
    with pytest.raises(SystemExit):
        tmdb_pull.main(["--ids", str(SAMPLE), "--chunk", "0", "--max-chunks", "2", "--out", str(tmp_path)])
    with pytest.raises(SystemExit):
        tmdb_pull.main(["--ids", str(SAMPLE), "--out", str(tmp_path)])
    with pytest.raises(SystemExit):
        tmdb_pull.main(["--ids", str(SAMPLE), "--all", "--chunk", "0", "--repo", "a/b", "--out", str(tmp_path)])


def test_cli_all_pulls_the_sample_and_uploads_through_hub_store(tmp_path, no_dotenv, monkeypatch, capsys):
    from llm_movies.data import hub_store

    monkeypatch.setenv("TMDB_READ_TOKEN", TOKEN)
    client, uploads = FakeClient(), []
    monkeypatch.setattr(tmdb_pull, "TmdbClient", lambda token: client)
    monkeypatch.setattr(hub_store, "make_api", lambda: "api")
    monkeypatch.setattr(hub_store, "read_json", lambda repo, path, **kw: None)
    monkeypatch.setattr(hub_store, "upload_files", lambda files, repo, **kw: uploads.append((sorted(files), repo, kw["api"])))
    tmdb_pull.main(["--ids", str(SAMPLE), "--all", "--repo", "a/b", "--out", str(tmp_path), "--limit", "2", "--max-chunks", "1", "--workers", "1"])
    assert client.requested == [3924, 8773]
    assert uploads == [(["tmdb/missing_0000.txt", "tmdb/tmdb_0000.jsonl"], "a/b", "api")]
    assert json.loads(capsys.readouterr().out) == {"chunks": [0]}


# ---- a wrong state must never be written: sizes, the Hub's manifest, a failed download ----

META = {"chunk_size": 2, "chunks": 3}


def test_the_manifest_records_chunk_size_and_chunk_count(tmp_path):
    hub = FakeHub()
    manifest = Manifest(tmp_path / "manifest.json")
    tmdb_pull.pull_all(FakeClient(), CHUNKS, tmp_path, manifest, hub.upload, meta=META)
    sent = json.loads(hub.files["tmdb/manifest.json"])
    assert sent["_meta"] == META and sorted(k for k in sent if k != "_meta") == ["0", "1", "2"]
    assert Manifest(tmp_path / "manifest.json").meta() == META
    assert Manifest(tmp_path / "manifest.json").done_chunks() == {0, 1, 2}


def test_a_local_manifest_with_another_chunk_size_or_count_refuses_before_any_request(tmp_path):
    for old in ({"chunk_size": 5000, "chunks": 3}, {"chunk_size": 2, "chunks": 9}):
        manifest = Manifest(tmp_path / "manifest.json")
        manifest.set_meta(old)
        hub, client = FakeHub(), FakeClient()
        with pytest.raises(tmdb_pull.ManifestError, match="chunk"):
            tmdb_pull.pull_all(client, CHUNKS, tmp_path, manifest, hub.upload, meta=META)
        assert client.requested == [] and hub.commits == []


def test_a_hub_manifest_listing_done_chunks_the_local_one_lacks_refuses(tmp_path):
    # the download of tmdb/ failed, so the local manifest is empty while the Hub knows chunks 0 and 1
    remote = {"_meta": META, "0": {"pulled": 2, "missing": 0}, "1": {"pulled": 2, "missing": 0}}
    hub, client = FakeHub(), FakeClient()
    with pytest.raises(tmdb_pull.ManifestError, match="download"):
        tmdb_pull.pull_all(client, CHUNKS, tmp_path, Manifest(tmp_path / "manifest.json"), hub.upload, meta=META, remote=remote)
    assert client.requested == [] and hub.commits == [] and hub.files == {}
    assert not (tmp_path / "manifest.json").exists()


def test_a_hub_manifest_with_other_sizes_refuses(tmp_path):
    remote = {"_meta": {"chunk_size": 5000, "chunks": 241}}
    with pytest.raises(tmdb_pull.ManifestError, match="chunk"):
        tmdb_pull.pull_all(FakeClient(), CHUNKS, tmp_path, Manifest(tmp_path / "manifest.json"), FakeHub().upload, meta=META, remote=remote)


@pytest.mark.parametrize("other", [{"chunk_size": 5000, "chunks": 3}, {"chunk_size": 2, "chunks": 9}])
def test_a_hub_manifest_differing_in_only_one_size_refuses(tmp_path, other):
    # each of the two numbers alone is enough to cut the chunks differently
    with pytest.raises(tmdb_pull.ManifestError, match="chunk"):
        tmdb_pull.pull_all(
            FakeClient(), CHUNKS, tmp_path, Manifest(tmp_path / "manifest.json"), FakeHub().upload, meta=META, remote={"_meta": other}
        )


def test_a_hub_manifest_with_the_same_number_of_done_chunks_but_other_ones_refuses(tmp_path):
    # the Hub lists chunks 0 and 1; the local disk holds 0 and 2: equal counts, yet chunk 1 would be overwritten
    manifest = Manifest(tmp_path / "manifest.json")
    manifest.mark_done(0, {"pulled": 2, "missing": 0})
    manifest.mark_done(2, {"pulled": 2, "missing": 0})
    remote = {"0": {"pulled": 2, "missing": 0}, "1": {"pulled": 2, "missing": 0}}
    hub, client = FakeHub(), FakeClient()
    with pytest.raises(tmdb_pull.ManifestError, match="download"):
        tmdb_pull.pull_all(client, CHUNKS, tmp_path, manifest, hub.upload, meta=META, remote=remote)
    assert client.requested == [] and hub.commits == []


def test_a_hub_manifest_that_the_local_one_covers_is_fine(tmp_path):
    hub = FakeHub()
    manifest = Manifest(tmp_path / "manifest.json")
    manifest.set_meta(META)
    manifest.mark_done(0, {"pulled": 2, "missing": 0})
    remote = {"_meta": META, "0": {"pulled": 2, "missing": 0}}
    client = FakeClient()
    done = tmdb_pull.pull_all(client, CHUNKS, tmp_path, manifest, hub.upload, meta=META, remote=remote)
    assert done == [1, 2] and client.requested == [3, 4, 5, 6]
    assert sorted(json.loads(hub.files["tmdb/manifest.json"])) == ["0", "1", "2", "_meta"]


def test_the_first_run_has_no_hub_manifest_and_passes(tmp_path):
    done = tmdb_pull.pull_all(FakeClient(), CHUNKS, tmp_path, Manifest(tmp_path / "manifest.json"), FakeHub().upload, meta=META, remote=None)
    assert done == [0, 1, 2]


class DownFakeApi:
    """The Hub holds a manifest, but every download fails: Colab's `!cmd` would carry on to the pull."""

    def __init__(self):
        self.files = {"tmdb/manifest.json": json.dumps({"_meta": {"chunk_size": 5000, "chunks": 1}, "0": {"pulled": 5000, "missing": 0}}).encode()}
        self.commits = []

    def repo_exists(self, repo_id, repo_type=None):
        return True

    def list_repo_files(self, repo_id, repo_type=None):
        return sorted(self.files)

    def hf_hub_download(self, repo_id, filename, repo_type=None, local_dir=None):
        if self.fail_downloads:
            raise requests.ConnectionError("down")
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.files[filename])
        return str(target)

    fail_downloads = True

    def create_repo(self, **kw):
        pass

    def dataset_info(self, repo_id):
        return type("I", (), {"private": True})()

    def create_commit(self, **kw):
        self.commits.append(kw)


def test_failed_tmdb_download_then_all_leaves_the_hub_manifest_alone_and_stops(tmp_path, no_dotenv, monkeypatch, sleeps):
    from llm_movies.data import hub_store

    api = DownFakeApi()
    monkeypatch.setenv("TMDB_READ_TOKEN", TOKEN)
    monkeypatch.setattr(hub_store, "make_api", lambda: api)
    client = FakeClient()
    monkeypatch.setattr(tmdb_pull, "TmdbClient", lambda token: client)
    out = tmp_path / "raw" / "tmdb"
    with pytest.raises(requests.ConnectionError):  # the download step raises, so a `&&` chain stops here
        hub_store.main(["download", "--dir", str(out), "--path-in-repo", "tmdb"])
    # run by hand while the Hub is still down: reading the Hub's manifest fails too, so the run stops
    with pytest.raises(requests.ConnectionError):
        tmdb_pull.main(["--ids", str(SAMPLE), "--all", "--repo", "a/b", "--out", str(out)])
    # run by hand once the Hub is back but without the download: --all sees chunk 0 listed as done and refuses to start
    api.fail_downloads = False
    with pytest.raises(SystemExit, match="download"):
        tmdb_pull.main(["--ids", str(SAMPLE), "--all", "--repo", "a/b", "--out", str(out)])
    assert api.commits == [] and client.requested == [] and not (out / "manifest.json").exists()


def test_cli_all_passes_meta_and_the_hub_manifest(tmp_path, no_dotenv, monkeypatch, capsys):
    from llm_movies.config import load_config
    from llm_movies.data import hub_store

    monkeypatch.setenv("TMDB_READ_TOKEN", TOKEN)
    client, uploads = FakeClient(), []
    monkeypatch.setattr(tmdb_pull, "TmdbClient", lambda token: client)
    monkeypatch.setattr(hub_store, "make_api", lambda: "api")
    monkeypatch.setattr(hub_store, "read_json", lambda repo, path, **kw: None)
    monkeypatch.setattr(hub_store, "upload_files", lambda files, repo, **kw: uploads.append(files))
    tmdb_pull.main(["--ids", str(SAMPLE), "--all", "--repo", "a/b", "--out", str(tmp_path)])
    sent = json.loads(uploads[0]["tmdb/manifest.json"])
    assert sent["_meta"] == {"chunk_size": load_config().chunk_size, "chunks": 1}


# ---- several requests in flight at once ----


class ThreadedClient:
    """A thread-safe stand-in: answers arrive in scrambled order; `barrier` makes each request wait until that many
    are in flight at once; ids in `missing` give None; the id `fail_on` raises."""

    def __init__(self, missing=(), fail_on=None, barrier=None):
        self.lock = threading.Lock()
        self.requested: list[int] = []
        self.missing, self.fail_on, self.barrier = set(missing), fail_on, barrier

    def movie(self, tmdb_id):
        with self.lock:
            self.requested.append(tmdb_id)
        if self.barrier is not None:
            self.barrier.wait(timeout=5)  # raises BrokenBarrierError if the requests are not concurrent
        time.sleep((tmdb_id * 7 % 5) / 1000)  # later ids often finish first
        if tmdb_id == self.fail_on:
            raise RuntimeError("session died")
        return None if tmdb_id in self.missing else {"id": tmdb_id, "title": f"film {tmdb_id}"}


def assert_titles_match_ids(out: Path) -> None:
    """pull_chunk overwrites `id` with the id it asked for, so a record paired with the wrong id shows only in the rest
    of the record."""
    assert all(r["title"] == f"film {r['id']}" for r in read_shard(out / "tmdb_0000.jsonl"))


def test_workers_run_requests_at_the_same_time(tmp_path):
    client = ThreadedClient(barrier=threading.Barrier(4))
    stats = pull_chunk(client, 0, [1, 2, 3, 4], tmp_path, Manifest(tmp_path / "manifest.json"), workers=4)
    assert shard_ids(tmp_path) == [1, 2, 3, 4]
    assert stats == {"pulled": 4, "missing": 0}


def test_workers_write_in_id_order_across_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(tmdb_pull, "WINDOW", 7)
    ids = list(range(1, 31))
    stats = pull_chunk(ThreadedClient(missing={5, 17}), 0, ids, tmp_path, Manifest(tmp_path / "manifest.json"), workers=8)
    assert shard_ids(tmp_path) == [i for i in ids if i not in (5, 17)]
    assert_titles_match_ids(tmp_path)
    assert sorted(done_keys(tmp_path / "missing_0000.txt", "id")) == [5, 17]
    assert stats == {"pulled": 28, "missing": 2}


def test_a_failure_with_workers_keeps_the_ids_before_it_and_resumes_without_duplicates(tmp_path, monkeypatch):
    monkeypatch.setattr(tmdb_pull, "WINDOW", 5)
    ids = list(range(1, 21))
    manifest = Manifest(tmp_path / "manifest.json")
    with pytest.raises(RuntimeError, match="session died"):
        pull_chunk(ThreadedClient(fail_on=12), 0, ids, tmp_path, manifest, workers=4)
    assert shard_ids(tmp_path) == list(range(1, 12))  # everything before the failure, nothing after
    assert not manifest.is_done(0)
    again = ThreadedClient()
    stats = pull_chunk(again, 0, ids, tmp_path, manifest, workers=4)
    assert sorted(again.requested) == list(range(12, 21))
    assert shard_ids(tmp_path) == ids
    assert_titles_match_ids(tmp_path)
    assert stats == {"pulled": 20, "missing": 0}


def test_duplicate_ids_in_a_chunk_are_fetched_once(tmp_path):
    client = ThreadedClient()
    pull_chunk(client, 0, [1, 2, 1, 3, 2], tmp_path, Manifest(tmp_path / "manifest.json"), workers=3)
    assert sorted(client.requested) == [1, 2, 3]
    assert shard_ids(tmp_path) == [1, 2, 3]


def test_threads_together_keep_to_max_per_second(monkeypatch, sleeps):
    monkeypatch.setattr("time.monotonic", lambda: 100.0)  # frozen clock: every request books the next free slot
    client, session = make_client([FakeResponse(200, {"id": i}) for i in range(6)], max_per_second=10)
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(client.movie, range(6)))
    assert len(session.calls) == 6
    assert sorted(sleeps) == [pytest.approx(0.1 * n) for n in range(1, 6)]  # one slot each, none shared


def test_each_thread_gets_its_own_session_when_none_is_given():
    client = TmdbClient(TOKEN)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = pool.map(lambda _: (time.sleep(0.05), client.session)[1], range(2))
    assert first is not second
    assert client.session is client.session  # one per thread, reused


def test_cli_defaults_to_eight_workers_and_rejects_zero(monkeypatch, tmp_path, no_dotenv):
    monkeypatch.setenv("TMDB_READ_TOKEN", TOKEN)
    seen = {}
    monkeypatch.setattr(tmdb_pull, "TmdbClient", lambda token: FakeClient())
    monkeypatch.setattr(tmdb_pull, "pull_chunk", lambda *a, **kw: seen.update(kw) or {"pulled": 0, "missing": 0})
    tmdb_pull.main(["--ids", str(SAMPLE), "--chunk", "0", "--out", str(tmp_path)])
    assert seen["workers"] == 8
    with pytest.raises(SystemExit):
        tmdb_pull.main(["--ids", str(SAMPLE), "--chunk", "0", "--out", str(tmp_path), "--workers", "0"])


def test_a_failure_cancels_the_queued_requests_of_the_window(tmp_path):
    # 2 workers, a window of 200 ids, id 3 fails at once: the ids still queued must never be requested
    client = ThreadedClient(fail_on=3)
    with pytest.raises(RuntimeError, match="session died"):
        pull_chunk(client, 0, list(range(1, 201)), tmp_path, Manifest(tmp_path / "manifest.json"), workers=2)
    assert len(client.requested) < 20  # not the 200 the window holds
    assert shard_ids(tmp_path) == [1, 2]
