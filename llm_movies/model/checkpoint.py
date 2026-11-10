"""Checkpoints of a training run on the private Hub model repo, `runs/<run_name>/{best,last}/`.

`last` holds everything to resume exactly: adapter/, head.pt, scaler.json, train_config.yaml, optimizer.pt,
grad_scaler.pt, lr_scheduler.pt, rng.pt and state.json. `best` holds what is needed to predict (no optimizer, grad
scaler, lr scheduler or RNG). All files of one checkpoint go up in ONE commit with state.json last, so a session killed
mid-upload leaves the previous complete checkpoint in place; a folder without state.json counts as absent.

Task 8's loop: `save_checkpoint(model, scaler, cfg, state, "last", optimizer=opt, lr_scheduler=sched, grad_scaler=gs)`
every `save_every_steps`; on start `ck = load_checkpoint(cfg, "last")` gives (model, scaler, state), then
`restore_training_state(cfg, optimizer=opt, lr_scheduler=sched, grad_scaler=gs)` puts optimizer, scheduler, grad scaler
and all RNG states back (build the optimizer/scheduler from the loaded model first). `resume_state(cfg)` is the cheap
"is there a run to resume" check: it reads only state.json. Files are loaded with `torch.load(weights_only=True)`, so
the RNG states are stored as tensors, lists and numbers only.
"""

import json
import random
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import yaml
from peft import load_peft_weights, set_peft_model_state_dict

from llm_movies.data import hub_store
from llm_movies.model.config import TrainConfig
from llm_movies.model.data import TargetScaler
from llm_movies.model.predictor import MoviePredictor, build_predictor

Kind = Literal["best", "last"]


def _prefix(cfg: TrainConfig, kind: str) -> str:
    return f"runs/{cfg.run_name}/{kind}"


def rng_state() -> dict:
    """All RNG states as tensors, lists and numbers (loadable with weights_only=True)."""
    _, keys, pos, has_gauss, cached = np.random.get_state()
    version, internal, gauss = random.getstate()
    state = {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy": {"keys": torch.from_numpy(keys.astype(np.int64)), "pos": int(pos), "has_gauss": int(has_gauss), "cached": float(cached)},
        "python": {"version": version, "internal": list(internal), "gauss": gauss},
    }
    return state


def set_rng_state(state: dict) -> None:
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available() and len(state["cuda"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(state["cuda"])
    n = state["numpy"]
    np.random.set_state(("MT19937", n["keys"].numpy().astype(np.uint32), n["pos"], n["has_gauss"], n["cached"]))
    p = state["python"]
    random.setstate((p["version"], tuple(p["internal"]), p["gauss"]))


def save_checkpoint(
    model: MoviePredictor,
    scaler: TargetScaler,
    cfg: TrainConfig,
    state: dict,
    kind: Kind,
    api=None,
    *,
    optimizer=None,
    grad_scaler=None,
    lr_scheduler=None,
) -> None:
    """Upload one checkpoint as one commit. `state` is `step, epoch, best_metric, bad_checks, metrics`. For "last" the
    optimizer, grad scaler and lr scheduler (whichever the run has) and the current RNG states are saved too."""
    if kind not in ("best", "last"):
        raise ValueError(f"kind must be 'best' or 'last', got {kind!r}")
    api = api or hub_store.make_api()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        model.lm.save_pretrained(root / "adapter")
        torch.save(model.head.state_dict(), root / "head.pt")
        (root / "scaler.json").write_text(scaler.to_json(), encoding="utf-8")
        (root / "train_config.yaml").write_text(yaml.safe_dump(asdict(cfg), sort_keys=True), encoding="utf-8")
        if kind == "last":
            for name, obj in (("optimizer", optimizer), ("grad_scaler", grad_scaler), ("lr_scheduler", lr_scheduler)):
                if obj is not None:
                    torch.save(obj.state_dict(), root / f"{name}.pt")
            torch.save(rng_state(), root / "rng.pt")
        (root / "state.json").write_text(json.dumps(state), encoding="utf-8")
        names = sorted(f.relative_to(root).as_posix() for f in root.rglob("*") if f.is_file())
        names.remove("state.json")
        names.append("state.json")  # last in the commit's list
        files = {f"{_prefix(cfg, kind)}/{n}": root / n for n in names}
        hub_store.upload_files(files, cfg.hub_repo, api=api, message=f"{cfg.run_name} {kind} step {state.get('step')}", repo_type="model")


def _fetch(cfg: TrainConfig, kind: str, api, folder: Path) -> bool:
    """Download the checkpoint into `folder`; False when it is absent (no files, or no state.json)."""
    api = api or hub_store.make_api()
    hub_store.download(folder, cfg.hub_repo, _prefix(cfg, kind), api=api, repo_type="model")
    return (folder / "state.json").is_file()


def load_checkpoint(
    cfg: TrainConfig, kind: Kind, api=None, lm=None
) -> tuple[MoviePredictor, TargetScaler, dict] | None:
    """The saved model, scaler and state, or None when the run has no such checkpoint. `lm` is the base model (default:
    downloaded from `cfg.base_model`); the adapter and head weights are loaded into it."""
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "ck"
        if not _fetch(cfg, kind, api, folder):
            return None
        model = build_predictor(cfg, lm=lm)
        result = set_peft_model_state_dict(model.lm, load_peft_weights(str(folder / "adapter"), device="cpu"))
        if getattr(result, "unexpected_keys", None):
            raise RuntimeError(f"adapter has keys the model does not: {result.unexpected_keys[:3]}")
        model.head.load_state_dict(torch.load(folder / "head.pt", map_location="cpu", weights_only=True))
        scaler = TargetScaler.from_json((folder / "scaler.json").read_text(encoding="utf-8"))
        state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
    return model, scaler, state


def resume_state(cfg: TrainConfig, api=None) -> dict | None:
    """The `last` state (step, epoch, best_metric, bad_checks, metrics), or None for a fresh run."""
    api = api or hub_store.make_api()
    return hub_store.read_json(cfg.hub_repo, f"{_prefix(cfg, 'last')}/state.json", api=api, repo_type="model")


def restore_training_state(cfg: TrainConfig, api=None, *, optimizer=None, grad_scaler=None, lr_scheduler=None) -> bool:
    """Load optimizer, grad scaler, lr scheduler (those given) and the RNG states from the Hub `last`. False if absent."""
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "ck"
        if not _fetch(cfg, "last", api, folder):
            return False
        for name, obj in (("optimizer", optimizer), ("grad_scaler", grad_scaler), ("lr_scheduler", lr_scheduler)):
            if obj is not None:
                obj.load_state_dict(torch.load(folder / f"{name}.pt", map_location="cpu", weights_only=True))
        set_rng_state(torch.load(folder / "rng.pt", map_location="cpu", weights_only=True))
    return True


