from dataclasses import dataclass, fields
from pathlib import Path

import yaml

from llm_movies.prompt.render import HIDEABLE


@dataclass(frozen=True)
class TrainConfig:
    run_name: str
    base_model: str
    max_len: int = 768
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lr_lora: float = 2e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 8
    grad_accum: int = 4
    epochs: int = 5
    eval_every: float = 0.25
    patience: int = 3
    hide_p: float = 0.3
    revenue_oversample: float = 1.5
    sample_frac: float = 1.0
    drop_groups: tuple[str, ...] = ()
    anonymize: bool = False
    seed: int = 42
    save_every_steps: int = 200
    hub_repo: str = "Tsomaros/llm-movies-runs"
    data_repo: str = "Tsomaros/llm-movies-data"
    wandb_project: str = "llm-movies"

    def __post_init__(self):
        bad = sorted(set(self.drop_groups) - set(HIDEABLE))
        if bad:
            raise ValueError(f"unknown drop_groups {bad}; expected names from HIDEABLE {HIDEABLE}")


def _load_raw(path: Path, seen: tuple[Path, ...] = ()) -> dict:
    path = Path(path).resolve()
    if path in seen:
        raise ValueError(f"circular extends: {path}")
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    return {**_load_raw(path.parent / parent, seen + (path,)), **raw}  # relative to this file; child overrides


def load_train_config(path: str | Path) -> TrainConfig:
    raw = _load_raw(Path(path))
    unknown = sorted(set(raw) - {f.name for f in fields(TrainConfig)})
    if unknown:
        raise ValueError(f"unknown config key(s) {unknown} in {path}")
    if "drop_groups" in raw:
        raw["drop_groups"] = tuple(raw["drop_groups"])
    return TrainConfig(**raw)
