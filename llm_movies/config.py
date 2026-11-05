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

    def __post_init__(self):
        (c0, c1), (v0, v1) = self.calibration_years, self.validation_years
        if not (c0 == self.train_max_year + 1 and v0 == c1 + 1 and self.test_from_year == v1 + 1 and c0 <= c1 and v0 <= v1):
            raise ValueError(
                "The split bands must be contiguous and ordered: calibration starts the year after train_max_year, "
                "validation the year after calibration ends, test_from_year the year after validation ends."
            )


def load_config(path: Path = Path("config/data.yaml")) -> DataConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    for key in ("calibration_years", "validation_years"):
        raw[key] = tuple(raw[key])
    return DataConfig(**raw)
