import csv
import json
import sqlite3

import pytest
from hub_fakes import FakeHub

from llm_movies.data import hub_store
from llm_movies.eval import model_choice
from llm_movies.eval.model_choice import CANDIDATES, CONDITIONAL, GATED, parse_quarters, run, slug

REPO = "Tsomaros/llm-movies-raw"
GOOD, BAD, OTHER = "fake/good", "fake/bad", "fake/other"


def _db(path):
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE movies (tmdb_id INTEGER PRIMARY KEY, title TEXT, release_date TEXT, year INTEGER, revenue_2025 REAL,
            money_suspect INTEGER, revenue_incomplete INTEGER, vote_count INTEGER, vote_average REAL, split TEXT);
        CREATE TABLE people (id INTEGER PRIMARY KEY, name TEXT, birth TEXT);
        CREATE TABLE credits (movie_id INTEGER, person_id INTEGER, role TEXT, ord INTEGER);
        INSERT INTO people VALUES (1, 'Dina Dir', NULL);
        """
    )
    films = [
        # id, title, date, year, revenue, split
        (1, "Val One", "2017-05-01", 2017, 1e8, "validation"),
        (2, "Val Two", "2018-05-01", 2018, 1e8, "validation"),
        (3, "T19a", "2019-01-10", 2019, 3e8, "test"),
        (4, "T19b", "2019-02-10", 2019, 3e8, "test"),
        (5, "T19c", "2019-03-10", 2019, 3e8, "test"),
        (6, "T20a", "2020-05-10", 2020, 3e8, "test"),
        (7, "T20b", "2020-05-11", 2020, 3e8, "test"),
        (8, "H21", "2021-05-10", 2021, 2e8, "holdout"),
        (9, "Old", "2012-05-10", 2012, 2e8, "train"),
    ]
    con.executemany("INSERT INTO movies VALUES (?,?,?,?,?,0,0,500,6.0,?)", films)
    con.executemany("INSERT INTO credits VALUES (?,1,'director',0)", [(f[0],) for f in films])
    con.commit()
    con.close()
    return path


class Factory:
    """Stands in for make_generator: GOOD knows every answer, BAD fails to load, OTHER knows nothing."""

    def __init__(self):
        self.loaded, self.prompts = [], {}

    def __call__(self, model_id, device):
        if model_id == BAD:
            raise RuntimeError("401 gated repo, token hf_secret123456 was rejected")
        self.loaded.append(model_id)
        log = self.prompts.setdefault(model_id, [])

        def generate(prompts):
            log.extend(prompts)
            if model_id == GOOD:
                return [" Dina Dir" if "directed by" in p else "$90 million" for p in prompts]
            return ["I do not know" for _ in prompts]

        return generate


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret123456")
    freed = []
    monkeypatch.setattr(model_choice, "_free_memory", lambda: freed.append(1))
    return _db(tmp_path / "m.db"), tmp_path / "out", FakeHub(), Factory(), freed


def go(setup, models=(GOOD, OTHER), **kw):
    db, out, hub, factory, _ = setup
    options = {"per_quarter": 2, "revenue_n": 5, "min_votes": 100, "device": "cpu", **kw}
    run(db, out, REPO, models, api=hub, load_generator=factory, **options)
    return json.loads(hub.files["model_choice/summary.json"])


def test_candidates_are_base_checkpoints_gated_then_conditional_last():
    assert CANDIDATES == (
        "Qwen/Qwen3-0.6B-Base",
        "Qwen/Qwen3-1.7B-Base",
        "HuggingFaceTB/SmolLM2-1.7B",
        "meta-llama/Llama-3.2-1B",
        "google/gemma-3-1b-pt",
        "google/gemma-4-E2B",
    )
    assert GATED == ("meta-llama/Llama-3.2-1B", "google/gemma-3-1b-pt")
    assert CONDITIONAL == ("google/gemma-4-E2B",)
    assert CANDIDATES[-3:] == (*GATED, *CONDITIONAL)


def test_parse_quarters():
    assert parse_quarters("2019Q1:2026Q2") == ((2019, 1), (2026, 2))
    for bad in ("2019Q5:2020Q1", "2019:2020", "2021Q1:2019Q1", "x"):
        with pytest.raises(ValueError):
            parse_quarters(bad)


def test_slug_is_a_safe_file_name():
    assert slug("Qwen/Qwen3-0.6B-Base") == "qwen_qwen3-0.6b-base"


def test_one_commit_per_model_with_its_files_and_the_summary(setup):
    summary = go(setup)
    _, out, hub, _, freed = setup
    assert hub.commits == [
        [f"model_choice/{slug(m)}_director.csv", f"model_choice/{slug(m)}_revenue.csv", "model_choice/summary.json", "model_choice/summary.md"]
        for m in (GOOD, OTHER)
    ]
    assert len(freed) == 2
    good, other = summary["models"][GOOD], summary["models"][OTHER]
    assert good["status"] == "ok" and good["director_recall_2019_2020"] == 1.0 and good["revenue_known_pct"] == 100.0
    assert good["recall_by_year"] == {"2019": 1.0, "2020": 1.0, "2021": 1.0}
    # 2019Q1 has 3 films (cap 2), 2019Q2 none, 2020Q2 has 2, 2021Q2 has 1; two validation films for revenue
    assert good["films_asked"] == {"director": 5, "revenue": 2}
    assert good["empirical_cutoff"] == "2021-06-30"
    assert other["director_recall_2019_2020"] == 0.0 and other["revenue_known_pct"] == 0.0
    assert other["empirical_cutoff"] == "n/a"  # a model that knows no 2019-2020 director has no cutoff to find
    assert (out / f"{slug(GOOD)}_director.csv").exists() and GOOD in (out / "summary.md").read_text()


def _rows(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_revenue_probe_asks_validation_films_only_and_files_hold_no_labels(setup):
    go(setup)
    out = setup[1]
    revenue = _rows(out / f"{slug(GOOD)}_revenue.csv")
    assert sorted(int(r["tmdb_id"]) for r in revenue) == [1, 2]
    director = _rows(out / f"{slug(GOOD)}_director.csv")
    assert set(director[0]) == {"tmdb_id", "year", "release_date", "answer", "known"}
    assert not {r["tmdb_id"] for r in director} & {"1", "2", "9"}  # validation and train films are outside 2019+


def test_every_model_sees_the_same_films(setup):
    go(setup)
    factory = setup[3]
    assert factory.prompts[GOOD] == factory.prompts[OTHER] and len(factory.prompts[GOOD]) == 7


def test_an_ok_model_in_the_repo_summary_is_skipped_and_kept(setup):
    first = go(setup, models=(GOOD,))
    factory = setup[3]
    factory.loaded.clear()
    second = go(setup, models=(GOOD, OTHER))
    assert factory.loaded == [OTHER]
    assert second["models"][GOOD] == first["models"][GOOD]
    assert len(setup[2].commits) == 2  # nothing is uploaded for the skipped model


def test_an_ok_model_is_rerun_when_the_settings_differ(setup):
    go(setup, models=(GOOD,))
    setup[3].loaded.clear()
    go(setup, models=(GOOD,), per_quarter=1)
    assert setup[3].loaded == [GOOD]


def test_a_failing_model_is_recorded_without_the_token_and_the_run_goes_on(setup):
    summary = go(setup, models=(BAD, GOOD))
    _, out, hub, factory, freed = setup
    bad = summary["models"][BAD]
    assert bad["status"] == "failed" and "401 gated repo" in bad["error"]
    assert "hf_secret123456" not in json.dumps(summary) and "hf_secret123456" not in (out / "summary.md").read_text()
    assert summary["models"][GOOD]["status"] == "ok"
    assert hub.commits[0] == ["model_choice/summary.json", "model_choice/summary.md"]  # no probe files for a failed model
    assert len(freed) == 2
    factory.loaded.clear()
    again = go(setup, models=(BAD, GOOD))
    assert factory.loaded == []  # a failed model is tried again (and fails again); GOOD stays skipped
    assert again["models"][BAD]["status"] == "failed"
    assert len(hub.commits) == 3


def test_the_exact_token_is_scrubbed_even_when_it_does_not_look_like_an_hf_token(setup, monkeypatch):
    # the hf_... pattern alone would miss a token of another shape
    monkeypatch.setenv("HF_TOKEN", "plain-secret-value")

    def leaky(model_id, device):
        raise RuntimeError("rejected token plain-secret-value for the repo")

    db, out, hub, _, _ = setup
    run(db, out, REPO, ("x/y",), per_quarter=2, revenue_n=5, device="cpu", api=hub, load_generator=leaky)
    entry = json.loads(hub.files["model_choice/summary.json"])["models"]["x/y"]
    assert entry["status"] == "failed" and "rejected token" in entry["error"]
    assert "plain-secret-value" not in json.dumps(entry) and "plain-secret-value" not in hub.files["model_choice/summary.md"].decode()


def test_the_seed_picks_the_films_and_is_the_same_for_every_model(setup):
    # 2019Q1 has 3 films and the cap is 2, so the seed decides which two are asked; both models must be asked the same two
    seen = set()
    for seed in range(8):
        for model in (GOOD, OTHER):
            setup[3].prompts.pop(model, None)
        go(setup, seed=seed)
        assert setup[3].prompts[GOOD] == setup[3].prompts[OTHER], seed
        seen.add(frozenset(p for p in setup[3].prompts[GOOD] if "directed by" in p))  # the director films only
    assert len(seen) > 1  # a hard-coded seed would give one fixed choice


def test_a_model_not_in_this_run_keeps_its_entry_in_the_summary(setup):
    go(setup, models=(GOOD,))
    summary = go(setup, models=(OTHER,))
    assert set(summary["models"]) == {GOOD, OTHER} and summary["models"][GOOD]["status"] == "ok"


def test_a_probe_error_after_loading_is_a_failure_too(setup):
    def broken(model_id, device):
        def generate(prompts):
            raise MemoryError("CUDA out of memory")

        return generate

    db, out, hub, _, _ = setup
    run(db, out, REPO, ("x/y",), per_quarter=2, revenue_n=5, device="cpu", api=hub, load_generator=broken)
    entry = json.loads(hub.files["model_choice/summary.json"])["models"]["x/y"]
    assert entry["status"] == "failed" and "out of memory" in entry["error"]


def test_no_baseline_years_gives_na_not_a_crash(setup):
    summary = go(setup, models=(GOOD,), first=(2021, 1), last=(2021, 4))
    entry = summary["models"][GOOD]
    assert entry["status"] == "ok" and entry["empirical_cutoff"] == "n/a" and entry["director_recall_2019_2020"] is None


def test_an_upload_failure_stops_the_run(setup):
    setup[2].fail_commit = True
    with pytest.raises(RuntimeError, match="session killed"):
        go(setup)


def test_without_a_gpu_the_run_refuses_unless_cpu_is_asked(setup, monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit, match="GPU"):
        run(setup[0], setup[1], REPO, (GOOD,), api=setup[2], load_generator=setup[3])
    assert setup[3].loaded == [] and setup[2].commits == []


def test_main_passes_the_arguments_through(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(model_choice, "run", lambda *a, **k: seen.update(a=a, k=k))
    model_choice.main(
        ["run", "--db", str(tmp_path / "m.db"), "--out", str(tmp_path), "--repo", REPO, "--models", "a/b,c/d",
         "--per-quarter", "7", "--quarters", "2020Q1:2020Q4", "--revenue-n", "9", "--min-votes", "50"]
    )
    assert seen["a"][2] == REPO and seen["a"][3] == ("a/b", "c/d")
    assert seen["k"]["per_quarter"] == 7 and seen["k"]["first"] == (2020, 1) and seen["k"]["last"] == (2020, 4)
    assert seen["k"]["revenue_n"] == 9 and seen["k"]["min_votes"] == 50
    model_choice.main(["run", "--db", "d", "--out", "o"])
    assert seen["a"][3] == CANDIDATES and seen["a"][2] == hub_store.DEFAULT_REPO
