"""The Hugging Face dataset repo that holds the raw shards, so no Colab session keeps state of its own.

python -m llm_movies.data.hub_store download|upload --repo REPO --dir DIR [--path-in-repo PATH]

Layout of the repo: ids/movie_ids.json.gz, tmdb/, wd_films/, wd_people/, reports/. The repo is a private dataset,
created on the first upload. An upload to a public repo is refused (`ensure_private_repo`, shared with `export.push`).
The token is HF_TOKEN from the environment (write access). Temporary files, `.env*`, `.git`, `.venv` and cache folders
are never uploaded, so a wrong --dir cannot leak a secret. Network errors, 429 and 5xx are retried for about two minutes
and then raised: a lost session is recoverable by design, so the caller just runs again. The id export
(`ids/movie_ids.json.gz`) is never replaced by a different file unless `--replace` is given: the chunks are cut from it,
so a new export would shift every id while the manifest still lists chunks as done. `api` is for tests.
"""

import argparse
import fnmatch
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import CommitOperationAdd

DEFAULT_REPO = "Tsomaros/llm-movies-raw"
NEVER_UPLOAD = ["*.tmp", "*.part", ".env*", "**/.env*", ".git/**", ".venv/**", ".cache/**", "**/.cache/**"]
PROTECTED = ("ids/movie_ids.json.gz",)
RETRIES = 6  # delays 2, 4, 8, 16, 32, 64 s


def hf_token() -> str:
    """HF_TOKEN without the line break a pasted secret often ends in. Exits on a token requests would refuse, because
    requests quotes the bad header, token included, in its error; this message never shows it."""
    load_dotenv()
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise SystemExit("HF_TOKEN is not set (a Hugging Face token with write access); put it in the environment or a .env file.")
    if not re.fullmatch(r"[!-~]+", token):
        raise SystemExit("HF_TOKEN has spaces, line breaks or other odd characters inside it (not shown here).")
    return token


def make_api():
    from huggingface_hub import HfApi

    return HfApi(token=hf_token())


def ensure_private_repo(api, repo_id: str, private: bool = True, *, repo_type: str = "dataset") -> None:
    """Create the repo (`repo_type` "dataset" or "model") if absent. If `private` is asked and the repo is public
    (create_repo leaves an existing repo as it is), raise before anything is uploaded."""
    api.create_repo(repo_id=repo_id, private=private, repo_type=repo_type, exist_ok=True)
    info = api.model_info(repo_id) if repo_type == "model" else api.dataset_info(repo_id)
    if private and info.private is not True:
        raise RuntimeError(f"The {repo_type} repo {repo_id} is public. Make it private on Hugging Face, then push again. Nothing was uploaded.")


def _retryable(error: Exception) -> bool:
    status = getattr(getattr(error, "response", None), "status_code", None)
    if isinstance(status, int):
        return status == 429 or status >= 500
    # a dropped connection or a timeout, from requests, the standard library or the Hub client's HTTP library
    return isinstance(error, ConnectionError | TimeoutError) or type(error).__module__.split(".")[0] in {"requests", "httpx", "httpx2"}


def _retry(call):
    for attempt in range(RETRIES + 1):
        try:
            return call()
        except Exception as error:
            if attempt == RETRIES or not _retryable(error):
                raise
            time.sleep(2.0 * 2**attempt)


def _ignored(name: str) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in NEVER_UPLOAD)


def _sha256(source: Path | bytes) -> str:
    digest = hashlib.sha256()
    if isinstance(source, bytes):
        digest.update(source)
    else:
        with open(source, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _unchanged_protected(api, repo_id: str, files: dict[str, Path | bytes], replace: bool) -> set[str]:
    """Repo paths among `files` that are protected and already in the repo with the same content (nothing to upload).
    A protected file that is in the repo with different content raises, unless `replace`."""
    same = set()
    protected = [name for name in files if name in PROTECTED]
    if replace or not protected:
        return same
    present = set(_retry(lambda: api.list_repo_files(repo_id, repo_type="dataset")))
    for name in protected:
        if name not in present:
            continue
        with tempfile.TemporaryDirectory() as stage:
            remote = _retry(lambda name=name, stage=stage: api.hf_hub_download(repo_id, name, repo_type="dataset", local_dir=stage))
            if _sha256(Path(remote)) == _sha256(files[name]):
                same.add(name)
            else:
                raise RuntimeError(
                    f"{name} is already in {repo_id} and differs from the local file. Replacing it would shift every chunk while the "
                    "manifest still lists chunks as done. Nothing was uploaded; pass --replace only if you mean to start the pull over."
                )
    return same


def upload(
    folder: Path,
    repo_id: str = DEFAULT_REPO,
    path_in_repo: str = "",
    *,
    api=None,
    allow_patterns: list[str] | None = None,
    replace: bool = False,
) -> None:
    """Upload `folder` to `path_in_repo` in one commit, or only the files matching `allow_patterns`."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Nothing to upload: {folder} is not a folder")
    api = api or make_api()
    ensure_private_repo(api, repo_id)
    prefix = path_in_repo.strip("/") + "/" if path_in_repo.strip("/") else ""
    local = {
        prefix + f.relative_to(folder).as_posix(): f
        for f in folder.rglob("*")
        if f.is_file() and not _ignored(f.relative_to(folder).as_posix())
    }
    unchanged = _unchanged_protected(api, repo_id, local, replace)
    kwargs = {"allow_patterns": allow_patterns} if allow_patterns else {}
    _retry(
        lambda: api.upload_folder(
            folder_path=str(folder),
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo=path_in_repo,
            ignore_patterns=[*NEVER_UPLOAD, *(name.removeprefix(prefix) for name in unchanged)],
            **kwargs,
        )
    )


def upload_files(
    files: dict[str, Path | bytes],
    repo_id: str = DEFAULT_REPO,
    *,
    api=None,
    message: str = "update",
    replace: bool = False,
    repo_type: str = "dataset",
) -> None:
    """Upload named files (repo path -> local file or bytes) as one commit."""
    for name in files:
        if _ignored(name) or _ignored(name.rsplit("/", 1)[-1]):
            raise ValueError(f"Refusing to upload {name}: env files, temporary files and repo internals are never uploaded")
    api = api or make_api()
    ensure_private_repo(api, repo_id, repo_type=repo_type)
    unchanged = _unchanged_protected(api, repo_id, files, replace)
    operations = [
        CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(src) if isinstance(src, Path) else src)
        for name, src in files.items()
        if name not in unchanged
    ]
    if not operations:
        return
    _retry(lambda: api.create_commit(repo_id=repo_id, repo_type=repo_type, operations=operations, commit_message=message))


def download(folder: Path, repo_id: str = DEFAULT_REPO, path_in_repo: str = "", *, api=None, repo_type: str = "dataset") -> int:
    """Download the repo's `path_in_repo` into `folder` (the prefix is stripped) and return the file count. A repo that
    does not exist yet is a first run: 0 files, no error. Existing local files are overwritten."""
    folder = Path(folder)
    api = api or make_api()
    if not _retry(lambda: api.repo_exists(repo_id, repo_type=repo_type)):
        return 0
    prefix = path_in_repo.strip("/") + "/" if path_in_repo.strip("/") else ""
    names = [n for n in _retry(lambda: api.list_repo_files(repo_id, repo_type=repo_type)) if n.startswith(prefix)]
    stage = folder.parent / f".{folder.name}.stage"
    shutil.rmtree(stage, ignore_errors=True)
    try:
        for name in names:
            source = _retry(lambda name=name: api.hf_hub_download(repo_id, name, repo_type=repo_type, local_dir=stage))
            target = folder / name[len(prefix) :]
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, target)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return len(names)


def download_file(repo_id: str, name: str, folder: Path, *, repo_type: str = "dataset", api=None) -> Path:
    """Download the one file `name` into `folder` (created) and return its path; nothing else in the repo is fetched."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    api = api or make_api()
    return Path(_retry(lambda: api.hf_hub_download(repo_id, name, repo_type=repo_type, local_dir=folder)))


def read_json(repo_id: str, path_in_repo: str, *, api=None, repo_type: str = "dataset"):
    """The JSON file at `path_in_repo`, or None when the repo or the file does not exist yet (a first run)."""
    api = api or make_api()
    if not _retry(lambda: api.repo_exists(repo_id, repo_type=repo_type)):
        return None
    if path_in_repo not in _retry(lambda: api.list_repo_files(repo_id, repo_type=repo_type)):
        return None
    with tempfile.TemporaryDirectory() as stage:
        path = _retry(lambda: api.hf_hub_download(repo_id, path_in_repo, repo_type=repo_type, local_dir=stage))
        return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Move shard folders to and from the private raw-data repo on Hugging Face.")
    parser.add_argument("action", choices=("download", "upload"))
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--path-in-repo", default="", help="folder inside the repo, e.g. tmdb")
    parser.add_argument("--replace", action="store_true", help="upload: allow replacing ids/movie_ids.json.gz with a different file")
    args = parser.parse_args(argv)
    api = make_api()
    if args.action == "upload":
        options = {"replace": True} if args.replace else {}
        upload(args.dir, args.repo, args.path_in_repo, api=api, **options)
        print(f"uploaded {args.dir} to {args.repo}/{args.path_in_repo}")
    else:
        count = download(args.dir, args.repo, args.path_in_repo, api=api)
        print(f"downloaded {count} files from {args.repo}/{args.path_in_repo}" if count else f"nothing to download from {args.repo}/{args.path_in_repo}")


if __name__ == "__main__":
    main()
