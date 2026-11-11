import copy
import io
import json
from dataclasses import replace
from datetime import date

import pandas as pd
import pytest
import torch
from hub_fakes import FakeHub
from test_data import _film

from llm_movies.model import train as train_module
from llm_movies.model.checkpoint import load_checkpoint, resume_state
from llm_movies.model.config import TrainConfig
from llm_movies.model.data import FilmDataset, TargetScaler, load_frames, validation_sample
from llm_movies.model.predictor import build_predictor
from llm_movies.model.train import evaluate, train

CFG = TrainConfig(
    run_name="t", base_model="unused", hub_repo="me/runs", max_len=48, batch_size=8, grad_accum=2, epochs=3,
    eval_every=1.0, patience=5, lr_lora=1e-2, lr_head=3e-2, save_every_steps=2, seed=3,
)  # 40 train films -> 5 micro-batches -> 3 optimizer steps per epoch (the last one a partial accumulation)


def _frame(n_train=40, n_val=16):
    rows = []
    for i in range(n_train + n_val):
        film = replace(_film(2000 + i % 14), runtime=80 + i, budget=1e7 * (1 + i % 7))
        rows.append(
            {
                "tmdb_id": i + 1, "release_date": date(2000 + i % 14, 5, 14), "split": "train" if i < n_train else "validation",
                "resolved_json": film.to_json(), "revenue_log": None if i % 9 == 4 else 16.0 + 0.05 * (i % 40),
                "score": None if i % 11 == 5 else 5.0 + (i % 5) * 0.5, "score_weight": None if i % 11 == 5 else 3.0 + i % 4,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def data_dir(tmp_path):
    _frame().to_parquet(tmp_path / "films.parquet")
    return tmp_path


@pytest.fixture
def base(tiny_lm):
    return copy.deepcopy(tiny_lm)  # pristine copy: train() wraps the lm it gets in place


def _run(cfg, data_dir, tmp_path, base, tok, hub, name="out"):
    return train(cfg, data_dir, api=hub, lm=copy.deepcopy(base), tokenizer=tok, device="cpu", out_dir=tmp_path / name)


def _trainable(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def _val_loss(model, scaler, cfg, data_dir, tok):
    _, val = load_frames(data_dir, cfg)
    return evaluate(model, FilmDataset(val, scaler, cfg, False), scaler, tok, cfg, torch.device("cpu"))[0]


def _lines(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def test_loss_decreases(data_dir, tmp_path, base, tiny_tokenizer):
    torch.manual_seed(CFG.seed)
    initial = build_predictor(CFG, lm=copy.deepcopy(base))
    train_df, _ = load_frames(data_dir, CFG)
    before = _val_loss(initial, TargetScaler.fit(train_df), CFG, data_dir, tiny_tokenizer)
    hub = FakeHub()
    summary = _run(CFG, data_dir, tmp_path, base, tiny_tokenizer, hub)
    model, scaler, state = load_checkpoint(CFG, "last", api=hub, lm=copy.deepcopy(base))
    after = _val_loss(model, scaler, CFG, data_dir, tiny_tokenizer)
    assert after < before - 0.01
    assert summary["stopped_reason"] == "epochs" and state["step"] == 9 and state["epoch"] == 3
    lines = _lines(tmp_path / "out" / "metrics.jsonl")
    assert [x["step"] for x in lines] == [3, 6, 9]  # eval_every 1.0 epoch = 3 steps, none repeated at the end
    assert lines[-1]["val_loss"] == pytest.approx(after, abs=1e-5) and summary["best_metric"] <= lines[0]["val_loss"]


def test_early_stopping_by_patience(data_dir, tmp_path, base, tiny_tokenizer):
    cfg = replace(CFG, lr_lora=0.0, lr_head=0.0, patience=2, epochs=10)
    hub = FakeHub()
    summary = _run(cfg, data_dir, tmp_path, base, tiny_tokenizer, hub)
    assert len(_lines(tmp_path / "out" / "metrics.jsonl")) == 3  # the first check sets the best, the next two bring nothing
    assert summary["stopped_reason"] == "patience" and summary["best_step"] == 3
    assert resume_state(cfg, api=hub)["step"] == 9  # stopped well before 10 epochs, and saved where it stopped


def test_best_checkpoint_is_best_metric(data_dir, tmp_path, base, tiny_tokenizer):
    cfg = replace(CFG, lr_lora=1e-2, lr_head=0.02, epochs=4)  # measured: val loss falls, then rises again, so the best is a middle check
    hub = FakeHub()
    summary = _run(cfg, data_dir, tmp_path, base, tiny_tokenizer, hub)
    values = [x["val_loss"] for x in _lines(tmp_path / "out" / "metrics.jsonl")]
    assert 0 < values.index(min(values)) < len(values) - 1  # precondition: best is neither first nor last, so "save the latest" fails
    assert summary["best_metric"] == pytest.approx(min(values))
    model, scaler, state = load_checkpoint(cfg, "best", api=hub, lm=copy.deepcopy(base))
    assert state["best_metric"] == pytest.approx(min(values)) and state["step"] == 3 * (1 + values.index(min(values)))
    assert _val_loss(model, scaler, cfg, data_dir, tiny_tokenizer) == pytest.approx(min(values), abs=1e-5)  # the saved weights score it
    assert summary["best_step"] == state["step"] and "revenue" in summary["metrics"] and "score" in summary["metrics"]


class Killed(Exception):
    pass


class KillingHub(FakeHub):
    """Stores the `last` commit of `kill_at` optimizer steps, then 'loses the session'."""

    def __init__(self, kill_at):
        super().__init__()
        self.kill_at = kill_at

    def create_commit(self, repo_id, repo_type, operations, commit_message):
        super().create_commit(repo_id, repo_type, operations, commit_message)
        if self.files.get("runs/t/last/state.json") and json.loads(self.files["runs/t/last/state.json"])["step"] == self.kill_at:
            raise Killed


def test_resume_continues_same_run(data_dir, tmp_path, base, tiny_tokenizer):
    straight = FakeHub()
    s1 = _run(CFG, data_dir, tmp_path, base, tiny_tokenizer, straight, "straight")
    hub = KillingHub(kill_at=4)  # step 4 = epoch 1, one step in: resume must skip that epoch's first 2 micro-batches
    with pytest.raises(Killed):
        _run(CFG, data_dir, tmp_path, base, tiny_tokenizer, hub, "resumed")
    assert resume_state(CFG, api=hub)["step"] == 4
    hub.kill_at = -1
    s2 = _run(CFG, data_dir, tmp_path, base, tiny_tokenizer, hub, "resumed")
    a = load_checkpoint(CFG, "last", api=straight, lm=copy.deepcopy(base))
    b = load_checkpoint(CFG, "last", api=hub, lm=copy.deepcopy(base))
    assert a[2]["step"] == b[2]["step"] == 9
    pa, pb = _trainable(a[0]), _trainable(b[0])
    assert pa.keys() == pb.keys() and any(p.abs().sum() > 0 for p in pa.values())
    for name in pa:
        torch.testing.assert_close(pa[name], pb[name], atol=1e-5, rtol=0)
    assert s1["best_metric"] == pytest.approx(s2["best_metric"], abs=1e-5) and s1["best_step"] == s2["best_step"]
    assert (a[1].mean, a[1].std) == (b[1].mean, b[1].std)
    oa, ob = (torch.load(io.BytesIO(h.files["runs/t/last/optimizer.pt"]), weights_only=True) for h in (straight, hub))
    assert oa["state"].keys() == ob["state"].keys() and oa["state"]
    for k in oa["state"]:  # AdamW moments and step counts after 9 steps
        assert float(oa["state"][k]["step"]) == float(ob["state"][k]["step"]) == 9
        for moment in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(oa["state"][k][moment], ob["state"][k][moment], atol=1e-5, rtol=0)
    steps = [[x["step"] for x in _lines(tmp_path / d / "metrics.jsonl")] for d in ("straight", "resumed")]
    assert steps == [[3, 6, 9], [3, 6, 9]]


def _nan_in_training(monkeypatch, nan_calls):
    """pinball_loss gives NaN on the first `nan_calls` training calls (those whose predictions need grad)."""
    real, seen = train_module.pinball_loss, {"n": 0}

    def fake(pred, target, mask, weight):
        loss = real(pred, target, mask, weight)
        if pred.requires_grad:
            seen["n"] += 1
            if seen["n"] <= nan_calls:
                return loss * float("nan")
        return loss

    monkeypatch.setattr(train_module, "pinball_loss", fake)


def test_nan_steps_skipped_then_abort(data_dir, tmp_path, base, tiny_tokenizer, monkeypatch):
    # the first micro-batch of each of the first three steps is NaN: those steps are skipped and training goes on
    _nan_in_training(monkeypatch, nan_calls=3)
    hub = FakeHub()
    summary = _run(CFG, data_dir, tmp_path, base, tiny_tokenizer, hub, "some")
    assert summary["stopped_reason"] == "epochs" and summary["nan_steps"] == 3
    assert resume_state(CFG, api=hub)["step"] == 9  # a skipped step still counts, so schedule and data position stay aligned
    # every step bad: 10 in a row stop the run with a message, no exception
    monkeypatch.undo()
    _nan_in_training(monkeypatch, nan_calls=10**9)
    cfg = replace(CFG, epochs=10)
    summary = _run(cfg, data_dir, tmp_path, base, tiny_tokenizer, FakeHub(), "all")
    assert summary["stopped_reason"] == "nan" and summary["nan_steps"] == 10
    assert "non-finite" in summary["error"] and summary["step"] == 10  # stopped right at the 10th bad step


def test_nan_steps_do_not_update_weights(data_dir, tmp_path, base, tiny_tokenizer, monkeypatch):
    _nan_in_training(monkeypatch, nan_calls=10**9)
    cfg = replace(CFG, epochs=1)  # 3 steps, all NaN: fewer than 10, so the run finishes
    hub = FakeHub()
    summary = _run(cfg, data_dir, tmp_path, base, tiny_tokenizer, hub)
    assert summary["stopped_reason"] == "epochs" and summary["nan_steps"] == 3
    torch.manual_seed(cfg.seed)
    initial = _trainable(build_predictor(cfg, lm=copy.deepcopy(base)))
    after = _trainable(load_checkpoint(cfg, "last", api=hub, lm=copy.deepcopy(base))[0])
    assert initial.keys() == after.keys()
    for name in initial:
        torch.testing.assert_close(after[name], initial[name], atol=0, rtol=0)  # no step ran: not even weight decay


def test_a_good_step_resets_the_nan_counter(data_dir, tmp_path, base, tiny_tokenizer, monkeypatch):
    """Every other optimizer step is NaN (15 of 30, never two in a row): the run must not abort on the 10th bad step."""
    real, pos = train_module.pinball_loss, {"step": 0, "micro": 0}

    def fake(pred, target, mask, weight):
        loss = real(pred, target, mask, weight)
        if not pred.requires_grad:
            return loss
        size = 1 if pos["step"] % 3 == 2 else 2  # micro-batches in this step: the last step of an epoch has one
        if pos["step"] % 2 == 0:  # NaN breaks the step at its first micro-batch
            pos.update(step=pos["step"] + 1, micro=0)
            return loss * float("nan")
        pos["micro"] += 1
        if pos["micro"] == size:
            pos.update(step=pos["step"] + 1, micro=0)
        return loss

    monkeypatch.setattr(train_module, "pinball_loss", fake)
    summary = _run(replace(CFG, epochs=10), data_dir, tmp_path, base, tiny_tokenizer, FakeHub())
    assert summary["stopped_reason"] == "epochs" and summary["step"] == 30 and summary["nan_steps"] == 15


# ---- val_sample: early stopping on a fixed sample, the best scored once on the whole validation frame ----


def _spy_evaluate(monkeypatch, fail_on_full=False):
    """Record the size of every frame train() evaluates; with `fail_on_full`, the first whole-frame (16 films) call dies."""
    real, calls = train_module.evaluate, []

    def spy(model, dataset, *args, **kwargs):
        calls.append(len(dataset))
        if fail_on_full and len(dataset) == 16 and calls.count(16) == 1:
            raise Killed
        return real(model, dataset, *args, **kwargs)

    monkeypatch.setattr(train_module, "evaluate", spy)
    return calls


def _best_state(hub):
    return json.loads(hub.files["runs/t/best/state.json"])


def test_checks_use_the_val_sample_and_the_best_is_scored_on_the_full_frame(data_dir, tmp_path, base, tiny_tokenizer, monkeypatch):
    cfg = replace(CFG, val_sample=6, epochs=4, lr_head=0.02)  # 16 validation films; 4 checks
    calls = _spy_evaluate(monkeypatch)
    hub = FakeHub()
    summary = _run(cfg, data_dir, tmp_path, base, tiny_tokenizer, hub)
    assert calls == [6, 6, 6, 6, 16]  # every check on the sample, then the whole frame exactly once
    state = _best_state(hub)
    assert state["step"] < summary["step"]  # precondition: best is not the last weights, so scoring "last" would fail
    full = state["full_validation"]
    assert full["step"] == state["step"] == summary["best_step"] and full["films"] == 16
    assert state["stopped_reason"] == summary["stopped_reason"] == "epochs"
    model, scaler, _ = load_checkpoint(cfg, "best", api=hub, lm=copy.deepcopy(base))
    _, val = load_frames(data_dir, cfg)
    cpu = torch.device("cpu")
    on_full = evaluate(model, FilmDataset(val, scaler, cfg, False), scaler, tiny_tokenizer, cfg, cpu)
    sample = FilmDataset(validation_sample(val, 6, cfg.seed), scaler, cfg, False)
    on_sample = evaluate(model, sample, scaler, tiny_tokenizer, cfg, cpu)
    assert full["val_loss"] == pytest.approx(on_full[0], abs=1e-5)
    assert full["revenue"]["mae_log"] == pytest.approx(on_full[1]["revenue"]["mae_log"], abs=1e-5)
    assert full["score"]["mae"] == pytest.approx(on_full[1]["score"]["mae"], abs=1e-5)
    assert state["best_metric"] == pytest.approx(on_sample[0], abs=1e-5)  # the checks used that very sample
    assert abs(full["val_loss"] - state["best_metric"]) > 1e-4  # precondition: sample and whole frame differ
    assert summary["full_validation"] == full
    assert [x["step"] for x in _lines(tmp_path / "out" / "metrics.jsonl")] == [3, 6, 9, 12]  # checks only


def test_without_val_sample_the_best_check_is_the_full_validation(data_dir, tmp_path, base, tiny_tokenizer, monkeypatch):
    calls = _spy_evaluate(monkeypatch)
    hub = FakeHub()
    summary = _run(CFG, data_dir, tmp_path, base, tiny_tokenizer, hub)
    assert calls == [16, 16, 16]  # no extra pass: the checks already scored the whole frame
    state = _best_state(hub)
    assert state["full_validation"] == {
        "step": state["step"], "films": 16, "val_loss": state["best_metric"], **state["best_metrics"]
    }
    assert summary["full_validation"] == state["full_validation"] and state["stopped_reason"] == "epochs"


def test_rerun_of_a_finished_run_does_not_score_again(data_dir, tmp_path, base, tiny_tokenizer, monkeypatch):
    cfg = replace(CFG, val_sample=6)
    hub = FakeHub()
    first = _run(cfg, data_dir, tmp_path, base, tiny_tokenizer, hub)
    commits, stored = len(hub.commits), _best_state(hub)
    calls = _spy_evaluate(monkeypatch)
    again = _run(cfg, data_dir, tmp_path, base, tiny_tokenizer, hub, "again")
    assert calls == [] and len(hub.commits) == commits and _best_state(hub) == stored
    assert again["full_validation"] == first["full_validation"] == stored["full_validation"]


def test_a_full_validation_of_an_older_best_is_scored_again(data_dir, tmp_path, base, tiny_tokenizer, monkeypatch):
    cfg = replace(CFG, val_sample=6)
    hub = FakeHub()
    first = _run(cfg, data_dir, tmp_path, base, tiny_tokenizer, hub)
    stale = _best_state(hub)
    stale["full_validation"] = {**stale["full_validation"], "step": stale["step"] - 1, "val_loss": 123.0}  # an earlier best's score
    hub.files["runs/t/best/state.json"] = json.dumps(stale).encode()
    calls = _spy_evaluate(monkeypatch)
    again = _run(cfg, data_dir, tmp_path, base, tiny_tokenizer, hub, "again")
    assert calls == [16]  # the stored score belongs to another step: scored once more
    assert again["full_validation"]["step"] == stale["step"] and again["full_validation"]["val_loss"] != 123.0
    assert again["full_validation"] == first["full_validation"] == _best_state(hub)["full_validation"]


def test_a_session_lost_during_full_validation_scores_it_on_the_rerun(data_dir, tmp_path, base, tiny_tokenizer, monkeypatch):
    cfg = replace(CFG, val_sample=6)
    calls = _spy_evaluate(monkeypatch, fail_on_full=True)
    hub = FakeHub()
    with pytest.raises(Killed):
        _run(cfg, data_dir, tmp_path, base, tiny_tokenizer, hub)
    assert "full_validation" not in _best_state(hub) and calls == [6, 6, 6, 16]
    summary = _run(cfg, data_dir, tmp_path, base, tiny_tokenizer, hub, "again")
    assert calls == [6, 6, 6, 16, 16]  # training was finished: only the full pass runs again
    assert _best_state(hub)["full_validation"]["films"] == 16 and summary["full_validation"]["step"] == summary["best_step"]


def test_a_best_checked_on_a_sample_is_scored_even_if_the_config_now_checks_everything(data_dir, tmp_path, base, tiny_tokenizer, monkeypatch):
    sampled = replace(CFG, val_sample=6)
    hub = FakeHub()
    _run(sampled, data_dir, tmp_path, base, tiny_tokenizer, hub)
    lost = _best_state(hub)
    assert lost["val_films"] == 6  # the best check scored the 6-film sample
    del lost["full_validation"]  # as if the session died before the full pass
    hub.files["runs/t/best/state.json"] = json.dumps(lost).encode()
    calls = _spy_evaluate(monkeypatch)
    again = _run(CFG, data_dir, tmp_path, base, tiny_tokenizer, hub, "again")  # resumed with val_sample null (a config edit)
    assert calls == [16]  # a real pass: the sample's metrics are never copied as full ones
    assert again["full_validation"]["val_loss"] != pytest.approx(lost["best_metric"], abs=1e-6)


def test_a_nan_stopped_run_with_a_best_still_gets_full_validation(data_dir, tmp_path, base, tiny_tokenizer, monkeypatch):
    real, seen = train_module.pinball_loss, {"n": 0}

    def fake(pred, target, mask, weight):  # training is fine for 3 steps (one check, a best), then NaN for good
        loss = real(pred, target, mask, weight)
        if pred.requires_grad:
            seen["n"] += 1
            if seen["n"] > 5:  # 3 steps = 5 micro-batches
                return loss * float("nan")
        return loss

    monkeypatch.setattr(train_module, "pinball_loss", fake)
    hub = FakeHub()
    summary = _run(replace(CFG, val_sample=6, epochs=10), data_dir, tmp_path, base, tiny_tokenizer, hub)
    assert summary["stopped_reason"] == "nan" and summary["error"] and summary["best_step"] == 3
    state = _best_state(hub)
    assert state["stopped_reason"] == "nan" and state["full_validation"]["step"] == 3 and state["full_validation"]["films"] == 16
