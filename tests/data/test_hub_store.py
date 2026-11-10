from pathlib import Path

import pytest
import requests

from llm_movies.data import hub_store
from llm_movies.data.hub_store import DEFAULT_REPO, download, download_file, ensure_private_repo, upload, upload_files


def http_error(status):
    error = requests.HTTPError(f"HTTP {status}")
    error.response = type("Response", (), {"status_code": status})()
    return error


class FakeApi:
    """A Hub in a dict: repo files by path. `fail` scripts exceptions for named calls, in order."""

    def __init__(self, exists=True, private=True, files=None, fail=None):
        self.exists, self.private = exists, private
        self.files = dict(files or {})
        self.fail = {name: list(errors) for name, errors in (fail or {}).items()}
        self.calls = []

    def _call(self, name, **kwargs):
        self.calls.append((name, kwargs))
        if self.fail.get(name):
            raise self.fail[name].pop(0)

    def names(self):
        return [name for name, _ in self.calls]

    def create_repo(self, **kwargs):
        self._call("create_repo", **kwargs)
        self.exists = True

    def dataset_info(self, repo_id):
        self._call("dataset_info", repo_id=repo_id)
        return type("Info", (), {"private": self.private})()

    def repo_exists(self, repo_id, repo_type=None):
        self._call("repo_exists", repo_id=repo_id, repo_type=repo_type)
        return self.exists

    def upload_folder(self, **kwargs):
        self._call("upload_folder", **kwargs)

    def create_commit(self, **kwargs):
        self._call("create_commit", **kwargs)

    def list_repo_files(self, repo_id, repo_type=None):
        self._call("list_repo_files", repo_id=repo_id)
        return sorted(self.files)

    def hf_hub_download(self, repo_id, filename, repo_type=None, local_dir=None):
        self._call("hf_hub_download", filename=filename)
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.files[filename])
        return str(target)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    slept = []
    monkeypatch.setattr("time.sleep", slept.append)
    return slept


@pytest.fixture
def folder(tmp_path):
    d = tmp_path / "shards"
    d.mkdir()
    (d / "a.jsonl").write_text("1\n")
    (d / "b.tmp").write_text("half")
    return d


# ---- upload ----


def test_first_run_creates_a_private_dataset_repo_and_uploads(folder):
    api = FakeApi(exists=False)
    upload(folder, DEFAULT_REPO, "tmdb", api=api)
    assert api.names() == ["create_repo", "dataset_info", "upload_folder"]
    create, up = api.calls[0][1], api.calls[2][1]
    assert create == {"repo_id": "Tsomaros/llm-movies-raw", "private": True, "repo_type": "dataset", "exist_ok": True}
    assert up["repo_id"] == DEFAULT_REPO and up["repo_type"] == "dataset" and up["path_in_repo"] == "tmdb"
    assert "*.tmp" in up["ignore_patterns"]


def test_a_public_repo_is_refused_and_nothing_is_uploaded(folder):
    api = FakeApi(private=False)
    with pytest.raises(RuntimeError, match="private"):
        upload(folder, DEFAULT_REPO, "tmdb", api=api)
    with pytest.raises(RuntimeError, match="private"):
        upload_files({"x": b"1"}, DEFAULT_REPO, api=api)
    assert "upload_folder" not in api.names() and "create_commit" not in api.names()


def test_the_shared_check_is_the_one_export_push_uses():
    from llm_movies.data import export

    assert export.ensure_private_repo is ensure_private_repo
    api = FakeApi(private=False)
    with pytest.raises(RuntimeError, match="public"):
        ensure_private_repo(api, "a/b", True)
    ensure_private_repo(api, "a/b", False)  # asking for a public repo skips the check


def test_tmp_files_are_never_uploaded_even_if_asked(folder):
    api = FakeApi()
    upload(folder, DEFAULT_REPO, "tmdb", api=api, allow_patterns=["*"])
    up = api.calls[-1][1]
    assert "*.tmp" in up["ignore_patterns"] and "*.part" in up["ignore_patterns"] and up["allow_patterns"] == ["*"]


def test_upload_files_is_one_commit_of_named_files(tmp_path):
    shard = tmp_path / "s.jsonl"
    shard.write_text("1\n")
    api = FakeApi()
    upload_files({"tmdb/tmdb_0000.jsonl": shard, "tmdb/manifest.json": b"{}"}, DEFAULT_REPO, api=api, message="chunk 0")
    commits = [kw for name, kw in api.calls if name == "create_commit"]
    assert len(commits) == 1 and commits[0]["repo_type"] == "dataset" and commits[0]["commit_message"] == "chunk 0"
    assert sorted(op.path_in_repo for op in commits[0]["operations"]) == ["tmdb/manifest.json", "tmdb/tmdb_0000.jsonl"]


def test_a_5xx_is_retried_then_succeeds(folder, no_sleep):
    api = FakeApi(fail={"upload_folder": [http_error(503), requests.ConnectionError("reset")]})
    upload(folder, DEFAULT_REPO, "tmdb", api=api)
    assert api.names().count("upload_folder") == 3 and len(no_sleep) == 2


def test_retries_run_out_and_raise(folder, no_sleep):
    api = FakeApi(fail={"upload_folder": [http_error(500)] * 10})
    with pytest.raises(requests.HTTPError):
        upload(folder, DEFAULT_REPO, "tmdb", api=api)
    assert api.names().count("upload_folder") == 7  # one try and six retries
    assert no_sleep == [2, 4, 8, 16, 32, 64]  # doubling from 2 s: a 1-2 minute window for a Hub hiccup


def test_a_4xx_is_not_retried(folder, no_sleep):
    api = FakeApi(fail={"upload_folder": [http_error(403)]})
    with pytest.raises(requests.HTTPError):
        upload(folder, DEFAULT_REPO, "tmdb", api=api)
    assert api.names().count("upload_folder") == 1 and no_sleep == []


def test_a_missing_folder_is_an_error_before_any_call(tmp_path):
    api = FakeApi()
    with pytest.raises(FileNotFoundError):
        upload(tmp_path / "nope", DEFAULT_REPO, "tmdb", api=api)
    assert api.calls == []


# ---- download ----


def test_download_of_a_repo_that_does_not_exist_is_a_first_run(tmp_path):
    api = FakeApi(exists=False)
    assert download(tmp_path / "tmdb", DEFAULT_REPO, "tmdb", api=api) == 0
    assert "hf_hub_download" not in api.names()


def test_download_takes_only_the_path_and_strips_the_prefix(tmp_path):
    api = FakeApi(files={"tmdb/tmdb_0000.jsonl": b"a\n", "tmdb/manifest.json": b"{}", "wd_films/wd_films.jsonl": b"b\n", "tmdb2/x": b"c"})
    out = tmp_path / "raw" / "tmdb"
    assert download(out, DEFAULT_REPO, "tmdb", api=api) == 2
    assert sorted(p.name for p in out.iterdir()) == ["manifest.json", "tmdb_0000.jsonl"]
    assert (out / "tmdb_0000.jsonl").read_bytes() == b"a\n"
    assert [p.name for p in out.parent.iterdir()] == ["tmdb"]  # no staging folder left behind


def test_download_with_no_path_takes_everything(tmp_path):
    api = FakeApi(files={"ids/m.gz": b"1", "tmdb/s": b"2"})
    assert download(tmp_path / "all", DEFAULT_REPO, "", api=api) == 2
    assert (tmp_path / "all" / "ids" / "m.gz").read_bytes() == b"1"


def test_download_overwrites_an_older_local_copy(tmp_path):
    out = tmp_path / "tmdb"
    out.mkdir()
    (out / "manifest.json").write_text("old")
    download(out, DEFAULT_REPO, "tmdb", api=FakeApi(files={"tmdb/manifest.json": b"new"}))
    assert (out / "manifest.json").read_text() == "new"


def test_download_file_fetches_only_the_named_file(tmp_path):
    api = FakeApi(files={"films.parquet": b"films", "locked.parquet": b"secret"})
    path = download_file("me/data", "films.parquet", tmp_path / "d", api=api)
    assert path == tmp_path / "d" / "films.parquet" and path.read_bytes() == b"films"
    assert [kw["filename"] for name, kw in api.calls if name == "hf_hub_download"] == ["films.parquet"]
    assert [p.name for p in (tmp_path / "d").iterdir()] == ["films.parquet"]  # locked.parquet never touched


def test_download_retries_a_5xx(tmp_path, no_sleep):
    api = FakeApi(files={"tmdb/a": b"1"}, fail={"list_repo_files": [http_error(502)]})
    assert download(tmp_path / "t", DEFAULT_REPO, "tmdb", api=api) == 1 and len(no_sleep) == 1


# ---- the command line ----


def test_missing_token_exits_clearly(monkeypatch, folder):
    monkeypatch.setattr(hub_store, "load_dotenv", lambda: None)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(SystemExit, match="HF_TOKEN"):
        hub_store.main(["upload", "--dir", str(folder)])


def test_cli_wires_the_arguments(monkeypatch, folder, tmp_path):
    seen = []
    monkeypatch.setattr(hub_store, "make_api", lambda: "api")
    monkeypatch.setattr(hub_store, "upload", lambda *a, **k: seen.append(("upload", a, k)))
    monkeypatch.setattr(hub_store, "download", lambda *a, **k: seen.append(("download", a, k)) or 3)
    hub_store.main(["upload", "--dir", str(folder), "--path-in-repo", "reports"])
    hub_store.main(["download", "--dir", str(tmp_path), "--repo", "me/other"])
    assert seen[0] == ("upload", (folder, DEFAULT_REPO, "reports"), {"api": "api"})
    assert seen[1] == ("download", (tmp_path, "me/other", ""), {"api": "api"})


# ---- never upload secrets or repo internals, even from a wrong --dir ----


def test_never_upload_patterns_exclude_env_git_venv_and_caches_by_the_hubs_own_filter():
    from huggingface_hub.utils import filter_repo_objects

    paths = [
        "a.jsonl", "tmdb_0000.jsonl", "b.tmp", "c.part", ".env", ".env.local", "sub/.env", "sub/deeper/.env.production",
        ".git/config", ".git/objects/ab/cdef", ".venv/lib/site.py", "x/.cache/huggingface/download/y", ".cache/z",
    ]
    kept = list(filter_repo_objects(paths, ignore_patterns=hub_store.NEVER_UPLOAD))
    assert kept == ["a.jsonl", "tmdb_0000.jsonl"]


def test_a_folder_with_an_env_file_is_uploaded_with_the_ignore_patterns(tmp_path):
    (tmp_path / ".env").write_text("HF_TOKEN=x")
    (tmp_path / "a.jsonl").write_text("1")
    api = FakeApi()
    upload(tmp_path, DEFAULT_REPO, "tmdb", api=api)
    patterns = api.calls[-1][1]["ignore_patterns"]
    for needed in (".env*", "**/.env*", ".git/**", ".venv/**", "**/.cache/**", "*.tmp"):
        assert needed in patterns


def test_upload_files_refuses_a_secret_by_name():
    with pytest.raises(ValueError, match="env"):
        upload_files({"tmdb/.env": b"HF_TOKEN=x"}, DEFAULT_REPO, api=FakeApi())


# ---- the id export in the repo is never silently replaced ----

IDS = "ids/movie_ids.json.gz"


def ids_folder(tmp_path, data=b"new export"):
    d = tmp_path / "ids"
    d.mkdir()
    (d / "movie_ids.json.gz").write_bytes(data)
    return d


def test_an_existing_different_id_export_is_refused_and_nothing_is_uploaded(tmp_path):
    api = FakeApi(files={IDS: b"old export"})
    with pytest.raises(RuntimeError, match="--replace"):
        upload(ids_folder(tmp_path), DEFAULT_REPO, "ids", api=api)
    with pytest.raises(RuntimeError, match="--replace"):
        upload_files({IDS: b"new export"}, DEFAULT_REPO, api=api)
    assert "upload_folder" not in api.names() and "create_commit" not in api.names()


def test_the_same_id_export_is_a_no_op_so_a_rerun_of_the_notebook_works(tmp_path):
    api = FakeApi(files={IDS: b"new export"})
    upload(ids_folder(tmp_path), DEFAULT_REPO, "ids", api=api)
    up = [kw for name, kw in api.calls if name == "upload_folder"]
    assert not up or "movie_ids.json.gz" in up[0]["ignore_patterns"]  # nothing is committed for the unchanged file


def test_a_first_upload_of_the_id_export_goes_through(tmp_path):
    api = FakeApi(exists=False)
    upload(ids_folder(tmp_path), DEFAULT_REPO, "ids", api=api)
    assert api.names().count("upload_folder") == 1
    assert "movie_ids.json.gz" not in api.calls[-1][1]["ignore_patterns"]


def test_replace_overwrites_the_id_export(tmp_path):
    api = FakeApi(files={IDS: b"old export"})
    upload(ids_folder(tmp_path), DEFAULT_REPO, "ids", api=api, replace=True)
    assert api.names().count("upload_folder") == 1


def test_other_folders_never_trigger_the_id_export_guard(folder):
    api = FakeApi(files={"tmdb/a.jsonl": b"old"})
    upload(folder, DEFAULT_REPO, "tmdb", api=api)
    assert "hf_hub_download" not in api.names()


def test_cli_replace_flag_reaches_upload(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(hub_store, "make_api", lambda: "api")
    monkeypatch.setattr(hub_store, "upload", lambda *a, **k: seen.append(k))
    hub_store.main(["upload", "--dir", str(tmp_path), "--path-in-repo", "ids", "--replace"])
    hub_store.main(["upload", "--dir", str(tmp_path), "--path-in-repo", "ids"])
    assert seen == [{"api": "api", "replace": True}, {"api": "api"}]


# ---- reading the repo's manifest ----


def test_read_json_returns_none_for_a_missing_repo_or_file_and_the_data_otherwise():
    assert hub_store.read_json(DEFAULT_REPO, "tmdb/manifest.json", api=FakeApi(exists=False)) is None
    assert hub_store.read_json(DEFAULT_REPO, "tmdb/manifest.json", api=FakeApi(files={"tmdb/x": b"1"})) is None
    api = FakeApi(files={"tmdb/manifest.json": b'{"0": {"pulled": 1}}'})
    assert hub_store.read_json(DEFAULT_REPO, "tmdb/manifest.json", api=api) == {"0": {"pulled": 1}}


def test_read_json_raises_when_the_hub_keeps_failing(no_sleep):
    api = FakeApi(files={"tmdb/manifest.json": b"{}"}, fail={"list_repo_files": [http_error(500)] * 20})
    with pytest.raises(requests.HTTPError):
        hub_store.read_json(DEFAULT_REPO, "tmdb/manifest.json", api=api)


def test_a_line_break_around_the_token_is_removed(monkeypatch):
    monkeypatch.setattr(hub_store, "load_dotenv", lambda: None)
    monkeypatch.setenv("HF_TOKEN", "hf_abc\r\n")
    assert hub_store.make_api().token == "hf_abc"


def test_a_blank_token_counts_as_missing(monkeypatch):
    monkeypatch.setattr(hub_store, "load_dotenv", lambda: None)
    monkeypatch.setenv("HF_TOKEN", " \r\n")
    with pytest.raises(SystemExit, match="HF_TOKEN"):
        hub_store.make_api()


@pytest.mark.parametrize("bad", ["secret\ntoken", "secret\r\ntoken", "secret token", "secret\ttoken", "secret\x7ftoken", "sécret"])
def test_a_bad_token_is_refused_without_showing_it(monkeypatch, bad):
    monkeypatch.setattr(hub_store, "load_dotenv", lambda: None)
    monkeypatch.setenv("HF_TOKEN", bad)
    with pytest.raises(SystemExit) as stop:
        hub_store.make_api()
    assert "HF_TOKEN" in str(stop.value) and "secret" not in str(stop.value) and "sécret" not in str(stop.value)


def test_public_model_repo_is_refused_and_model_info_is_used():
    class ModelApi(FakeApi):
        def model_info(self, repo_id):
            self._call("model_info", repo_id=repo_id)
            return type("Info", (), {"private": self.private})()

    api = ModelApi(private=False)
    with pytest.raises(RuntimeError, match="public"):
        upload_files({"runs/a.json": b"{}"}, "me/runs", api=api, repo_type="model")
    assert api.names() == ["create_repo", "model_info"]
    assert api.calls[0][1]["repo_type"] == "model"


def test_model_repo_commit_download_and_read_json_forward_the_repo_type(tmp_path):
    seen = []

    class ModelApi(FakeApi):
        def model_info(self, repo_id):
            return type("Info", (), {"private": True})()

        def repo_exists(self, repo_id, repo_type=None):
            seen.append(("repo_exists", repo_type))
            return True

        def list_repo_files(self, repo_id, repo_type=None):
            seen.append(("list_repo_files", repo_type))
            return sorted(self.files)

        def hf_hub_download(self, repo_id, filename, repo_type=None, local_dir=None):
            seen.append(("hf_hub_download", repo_type))
            return super().hf_hub_download(repo_id, filename, repo_type, local_dir)

    api = ModelApi(files={"runs/a/state.json": b'{"step": 3}'})
    upload_files({"runs/a/state.json": b"{}"}, "me/runs", api=api, repo_type="model")
    commit = [kw for name, kw in api.calls if name == "create_commit"]
    assert len(commit) == 1 and commit[0]["repo_type"] == "model"
    assert download(tmp_path / "a", "me/runs", "runs/a", api=api, repo_type="model") == 1
    assert hub_store.read_json("me/runs", "runs/a/state.json", api=api, repo_type="model") == {"step": 3}
    assert {name for name, _ in seen} == {"repo_exists", "list_repo_files", "hf_hub_download"}
    assert {t for _, t in seen} == {"model"}  # never the dataset default
