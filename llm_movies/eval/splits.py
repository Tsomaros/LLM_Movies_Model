"""Time splits and the lock on the test and holdout sets.

A film's split comes from its release date alone. The test and holdout sets are opened once, at the end, by the
final evaluation module; every other caller gets `TestSetLocked`.
"""

from datetime import date, timedelta
from typing import Literal

import pandas as pd

from llm_movies.config import DataConfig

Split = Literal["train", "calibration", "validation", "test", "holdout", "later"]

_OPEN = ("train", "calibration", "validation")
_LOCKED = ("test", "holdout")


class TestSetLocked(Exception):
    __test__ = False  # not a pytest test class, even when a test module imports it


def assign_split(release: date, cfg: DataConfig, pulled_at: date) -> Split:
    """The split of a film released on `release`. "later" films are too recent to have settled and are never used."""
    year = release.year
    if year <= cfg.train_max_year:
        return "train"
    if cfg.calibration_years[0] <= year <= cfg.calibration_years[1]:
        return "calibration"
    if cfg.validation_years[0] <= year <= cfg.validation_years[1]:
        return "validation"
    if cfg.base_model_cutoff is None or release <= cfg.base_model_cutoff:
        return "test"
    return "holdout" if release <= pulled_at - timedelta(days=cfg.mature_days) else "later"


def load_split(df: pd.DataFrame, name: str, final: bool = False) -> pd.DataFrame:
    """The rows of `df` in split `name`. The test and holdout sets need the explicit final flag."""
    if name in _LOCKED:
        if not final:
            raise TestSetLocked(f"The {name} set is locked until the final evaluation.")
    elif name not in _OPEN:
        raise ValueError(f"Unknown or unusable split: {name!r}")
    return df[df["split"] == name].copy()
