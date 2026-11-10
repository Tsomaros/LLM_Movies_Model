import copy
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from huggingface_hub import CommitOperationAdd

from llm_movies.model.checkpoint import load_checkpoint, restore_training_state, resume_state, save_checkpoint
from llm_movies.model.config import TrainConfig
from llm_movies.model.data import TargetScaler
from llm_movies.model.predictor import build_predictor

CFG = TrainConfig(run_name="run1", base_model="unused", hub_repo="me/runs")
STATE = {"step": 10, "epoch": 1, "best_metric": 0.5, "bad_checks": 0, "metrics": {"loss": 1.0}}


class FakeHub:
    """A private model repo in a dict. create_commit applies all operations or none."""

    def __init__(self, private=True):
        self.private, self.files, self.commits, self.fail_commit = private, {}, [], False

    def create_repo(self, **kw):
        pass

    def model_info(self, repo_id):
        return type("Info", (), {"private": self.private})()

    def repo_exists(self, repo_id, repo_type=None):
        return True

    def list_repo_files(self, repo_id, repo_type=None):
        return sorted(self.files)

    def hf_hub_download(self, repo_id, filename, repo_type=None, local_dir=None):
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.files[filename])
        return str(target)

    def create_commit(self, repo_id, repo_type, operations, commit_message):
        assert repo_type == "model"
        if self.fail_commit:
            raise RuntimeError("session killed")
        staged = {}
        for op in operations:
            assert isinstance(op, CommitOperationAdd)
            src = op.path_or_fileobj
            staged[op.path_in_repo] = src if isinstance(src, bytes) else Path(src).read_bytes()
        self.files.update(staged)
        self.commits.append([op.path_in_repo for op in operations])


@pytest.fixture
def base(tiny_lm):
    return copy.deepcopy(tiny_lm)  # build_predictor wraps its argument in place, so copy before the model fixture


@pytest.fixture
def model(tiny_lm, base):
    m = build_predictor(CFG, lm=tiny_lm).eval()
    with torch.no_grad():  # LoRA B starts at zero and the head is random: make every trainable weight non-trivial
        for p in m.parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p) * 0.1)
    return m


@pytest.fixture
def scaler():
    return TargetScaler([1.0, 2.0], [3.0, 4.0])


def _predict(m, tok, prompts):
    enc = tok(prompts, return_tensors="pt", padding=True)
    with torch.no_grad():
        return m(enc["input_ids"], enc["attention_mask"])


def _opt(model):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    return {"optimizer": opt, "lr_scheduler": torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)}


def test_save_load_roundtrip_same_predictions(model, scaler, base, tiny_tokenizer, golden_prompts):
    hub = FakeHub()
    save_checkpoint(model, scaler, CFG, STATE, "best", api=hub)
    fresh = build_predictor(CFG, lm=copy.deepcopy(base)).eval()  # what a load that restored nothing would give
    assert not torch.equal(_predict(model, tiny_tokenizer, golden_prompts[:3]), _predict(fresh, tiny_tokenizer, golden_prompts[:3]))
    loaded, loaded_scaler, state = load_checkpoint(CFG, "best", api=hub, lm=base)
    prompts = golden_prompts[:3]
    expected = _predict(model, tiny_tokenizer, prompts)
    assert expected.abs().sum() > 0
    assert torch.equal(expected, _predict(loaded.eval(), tiny_tokenizer, prompts))
    assert (loaded_scaler.mean, loaded_scaler.std) == (scaler.mean, scaler.std)
    assert state == STATE
    assert not any(n.endswith(("optimizer.pt", "rng.pt")) for n in hub.files)  # best is for predicting only
    assert yaml.safe_load(hub.files["runs/run1/best/train_config.yaml"])["run_name"] == "run1"


def test_upload_is_one_commit(model, scaler):
    hub = FakeHub()
    save_checkpoint(model, scaler, CFG, STATE, "last", api=hub, **_opt(model))
    assert len(hub.commits) == 1
    names = hub.commits[0]
    assert names[-1] == "runs/run1/last/state.json"
    assert all(n.startswith("runs/run1/last/") for n in names)
    for part in ("adapter/adapter_model.safetensors", "head.pt", "scaler.json", "train_config.yaml", "optimizer.pt", "lr_scheduler.pt", "grad_scaler.pt", "rng.pt"):
        assert f"runs/run1/last/{part}" in names
    assert json.loads(hub.files["runs/run1/last/state.json"]) == STATE


def test_interrupted_upload_resumes_from_previous(model, scaler, base, tiny_tokenizer, golden_prompts):
    hub = FakeHub()
    save_checkpoint(model, scaler, CFG, STATE, "last", api=hub, **_opt(model))
    before = dict(hub.files)
    expected = _predict(model, tiny_tokenizer, golden_prompts[:2])
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(1.0)
    hub.fail_commit = True
    with pytest.raises(RuntimeError, match="killed"):
        save_checkpoint(model, scaler, CFG, {**STATE, "step": 20}, "last", api=hub, **_opt(model))
    assert hub.files == before
    assert resume_state(CFG, api=hub) == STATE
    loaded, _, state = load_checkpoint(CFG, "last", api=hub, lm=base)
    assert state["step"] == 10
    assert torch.equal(expected, _predict(loaded.eval(), tiny_tokenizer, golden_prompts[:2]))


def test_refuses_public_repo(model, scaler):
    hub = FakeHub(private=False)
    with pytest.raises(RuntimeError, match="public"):
        save_checkpoint(model, scaler, CFG, STATE, "last", api=hub, **_opt(model))
    assert hub.commits == [] and hub.files == {}


def test_missing_run_returns_none(base):
    hub = FakeHub()
    assert load_checkpoint(CFG, "last", api=hub, lm=base) is None
    assert resume_state(CFG, api=hub) is None


def test_folder_without_state_json_counts_as_absent(model, scaler, base):
    hub = FakeHub()
    save_checkpoint(model, scaler, CFG, STATE, "last", api=hub, **_opt(model))
    del hub.files["runs/run1/last/state.json"]
    assert load_checkpoint(CFG, "last", api=hub, lm=base) is None
    assert resume_state(CFG, api=hub) is None


def test_restore_training_state_brings_back_optimizer_and_rngs(model, scaler):
    hub = FakeHub()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
    gs = torch.amp.GradScaler("cpu", enabled=False)
    sum(p.sum() for p in params).backward()
    opt.step()
    sched.step()
    random.seed(1), np.random.seed(2), torch.manual_seed(3)
    save_checkpoint(model, scaler, CFG, STATE, "last", api=hub, optimizer=opt, lr_scheduler=sched, grad_scaler=gs)
    expected = (random.random(), np.random.rand(), torch.rand(1).item())
    random.seed(9), np.random.seed(9), torch.manual_seed(9)

    opt2 = torch.optim.AdamW(params, lr=1e-3)
    sched2 = torch.optim.lr_scheduler.StepLR(opt2, step_size=1, gamma=0.5)
    assert restore_training_state(CFG, api=hub, optimizer=opt2, lr_scheduler=sched2, grad_scaler=gs)
    assert (random.random(), np.random.rand(), torch.rand(1).item()) == expected
    s1, s2 = opt.state_dict()["state"], opt2.state_dict()["state"]
    assert s1.keys() == s2.keys() and len(s1) == len(params)
    for i in s1:  # the AdamW moments and step count, not just which parameters have state
        assert s1[i]["step"] == s2[i]["step"] == 1
        assert torch.equal(s1[i]["exp_avg"], s2[i]["exp_avg"]) and torch.equal(s1[i]["exp_avg_sq"], s2[i]["exp_avg_sq"])
        assert s2[i]["exp_avg"].abs().sum() > 0
    assert sched2.last_epoch == sched.last_epoch and opt2.param_groups[0]["lr"] == opt.param_groups[0]["lr"]
    assert restore_training_state(CFG, api=FakeHub()) is False


@pytest.mark.parametrize("missing", ["optimizer", "lr_scheduler"])
def test_last_without_optimizer_or_scheduler_raises(model, scaler, missing):
    hub = FakeHub()
    kwargs = {**_opt(model), missing: None}
    with pytest.raises(ValueError, match="optimizer and the lr_scheduler"):
        save_checkpoint(model, scaler, CFG, STATE, "last", api=hub, **kwargs)
    assert hub.commits == []


def test_last_without_grad_scaler_leaves_no_stale_scaler_state(model, scaler):
    hub = FakeHub()
    scaler_a = torch.amp.GradScaler("cpu", enabled=True, init_scale=1234.0)
    save_checkpoint(model, scaler, CFG, STATE, "last", api=hub, grad_scaler=scaler_a, **_opt(model))
    save_checkpoint(model, scaler, CFG, {**STATE, "step": 20}, "last", api=hub, **_opt(model))  # CPU run: no scaler
    assert sorted(hub.commits[0]) == sorted(hub.commits[1])  # same file set
    fresh = torch.amp.GradScaler("cpu", enabled=True, init_scale=7.0)
    assert restore_training_state(CFG, api=hub, grad_scaler=fresh)
    assert fresh.state_dict()["scale"] == 7.0  # the marker was seen, not the old 1234.0


def test_restore_names_the_missing_file(model, scaler):
    hub = FakeHub()
    save_checkpoint(model, scaler, CFG, STATE, "last", api=hub, **_opt(model))
    del hub.files["runs/run1/last/lr_scheduler.pt"]
    with pytest.raises(FileNotFoundError, match="lr_scheduler.pt"):
        restore_training_state(CFG, api=hub, lr_scheduler=_opt(model)["lr_scheduler"])
