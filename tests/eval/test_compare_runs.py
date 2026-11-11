import json

from hub_fakes import FakeHub

from llm_movies.data import hub_store
from llm_movies.eval import compare_runs
from llm_movies.eval.compare_runs import compare

REPO = "me/runs"


def _metrics(mae_log, within, rcov, smae, scov):
    return {
        "revenue": {"mae_log": mae_log, "within_2x": within, "coverage_80": rcov, "median_width_x": 9.0, "pinball": 1.0},
        "score": {"mae": smae, "coverage_80": scov, "median_width": 2.0, "pinball": 0.3},
    }


def _state(step, check_loss, full=None, reason=None):
    state = {"step": step, "best_step": step, "best_metric": check_loss, "best_metrics": _metrics(9, 9, 9, 9, 9)}
    if full is not None:
        state["full_validation"] = {"step": step, "films": 95216, **full}
    if reason is not None:
        state["stopped_reason"] = reason
    return state


def _hub(states):
    hub = FakeHub()
    for name, state in states.items():
        hub.files[f"runs/{name}/best/state.json"] = json.dumps(state).encode()
    return hub


HUB_STATES = {
    "e1_small": _state(270, 0.31, {"val_loss": 0.32514, **_metrics(1.23456, 0.4, 0.71, 0.6543, 0.8)}, "epochs"),
    "e1_large": _state(180, 0.30, {"val_loss": 0.30001, **_metrics(1.1, 0.45, 0.7, 0.6, 0.81)}, "patience"),
    "e2_anon": _state(90, 0.40),  # still training: no full validation yet
}


def test_compare_table_from_fake_states():
    table = compare(["e1_small", "e1_large", "e2_anon", "e9_missing"], api=_hub(HUB_STATES), repo=REPO)
    rows = [line for line in table.strip().splitlines() if line.startswith("|")]
    assert rows[0] == (
        "| run | best step | stopped | val pinball | revenue mae_log | revenue within_2x | revenue coverage_80 "
        "| score mae | score coverage_80 | check pinball |"
    )
    assert rows[1].startswith("|---")
    assert rows[2] == "| e1_small | 270 | epochs | 0.3251 | 1.235 | 0.400 | 0.710 | 0.654 | 0.800 | 0.3100 |"
    assert rows[3] == "| e1_large | 180 | patience | 0.3000 | 1.100 | 0.450 | 0.700 | 0.600 | 0.810 | 0.3000 |"
    assert rows[4] == "| e2_anon | 90 | — | — | — | — | — | — | — | 0.4000 |"  # sample metrics never pass as full ones
    assert rows[5] == "| e9_missing | — | no best checkpoint | — | — | — | — | — | — | — |"
    assert len(rows) == 6 and rows[1] == "|" + "---|" * 10
    assert "full validation" in table.lower()


def test_cli_writes_the_table(tmp_path, monkeypatch, capsys):
    hub = _hub(HUB_STATES)
    monkeypatch.setattr(hub_store, "make_api", lambda: hub)
    out = tmp_path / "reports" / "experiments.md"
    compare_runs.main(["--runs", "e1_small,e1_large", "--out", str(out), "--repo", REPO])
    expected = compare(["e1_small", "e1_large"], api=hub, repo=REPO)
    assert out.read_text(encoding="utf-8") == expected
    assert "| e1_large | 180 |" in capsys.readouterr().out
