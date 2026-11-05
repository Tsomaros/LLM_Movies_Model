from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DataConfig:
    train_max_year: int = 2013
    calibration_years: tuple[int, int] = (2014, 2015)
    validation_years: tuple[int, int] = (2016, 2018)
    test_from_year: int = 2019
    base_model_cutoff: date | None = None
    min_votes: int = 50
    chunk_size: int = 50_000
    mature_days: int = 182
    shrink_films: int = 4


def load_config(path: Path = Path("config/data.yaml")) -> DataConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    for key in ("calibration_years", "validation_years"):
        raw[key] = tuple(raw[key])
    return DataConfig(**raw)
