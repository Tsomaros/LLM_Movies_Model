import ast
import re
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from llm_movies.config import DataConfig
from llm_movies.eval.splits import TestSetLocked, assign_split, load_split

PULLED_AT = date(2026, 11, 4)
CUTOFF = date(2023, 6, 30)
NO_CUTOFF = DataConfig()
WITH_CUTOFF = replace(DataConfig(), base_model_cutoff=CUTOFF)
PACKAGE = Path(__file__).parents[2] / "llm_movies"


@pytest.mark.parametrize(
    "release, expected",
    [
        (date(1900, 1, 1), "train"),
        (date(2013, 12, 31), "train"),
        (date(2014, 1, 1), "calibration"),
        (date(2015, 12, 31), "calibration"),
        (date(2016, 1, 1), "validation"),
        (date(2018, 12, 31), "validation"),
        (date(2019, 1, 1), "test"),
    ],
)
def test_band_boundaries(release, expected):
    assert assign_split(release, NO_CUTOFF, PULLED_AT) == expected


def test_null_cutoff_puts_all_2019plus_in_test():
    # Even a film released yesterday and one in the far future stay in test when there is no cutoff.
    for release in (date(2019, 1, 1), date(2024, 5, 5), PULLED_AT - timedelta(days=1), date(2030, 1, 1)):
        assert assign_split(release, NO_CUTOFF, PULLED_AT) == "test"


def test_cutoff_moves_later_films_to_holdout():
    assert assign_split(CUTOFF, WITH_CUTOFF, PULLED_AT) == "test"  # exactly the cutoff day
    assert assign_split(CUTOFF - timedelta(days=1), WITH_CUTOFF, PULLED_AT) == "test"
    assert assign_split(CUTOFF + timedelta(days=1), WITH_CUTOFF, PULLED_AT) == "holdout"
    assert assign_split(date(2019, 1, 1), WITH_CUTOFF, PULLED_AT) == "test"
    assert assign_split(date(2015, 6, 1), WITH_CUTOFF, PULLED_AT) == "calibration"  # the cutoff leaves earlier bands alone


def test_unsettled_after_cutoff_is_later():
    last_holdout = PULLED_AT - timedelta(days=182)
    assert assign_split(last_holdout, WITH_CUTOFF, PULLED_AT) == "holdout"
    assert assign_split(last_holdout + timedelta(days=1), WITH_CUTOFF, PULLED_AT) == "later"  # 181 days old
    assert assign_split(PULLED_AT, WITH_CUTOFF, PULLED_AT) == "later"
    assert assign_split(date(2027, 12, 15), WITH_CUTOFF, PULLED_AT) == "later"


def test_unsettled_film_before_cutoff_is_still_test():
    cutoff = PULLED_AT - timedelta(days=10)
    cfg = replace(DataConfig(), base_model_cutoff=cutoff)
    assert assign_split(cutoff, cfg, PULLED_AT) == "test"


def test_config_values_drive_the_bands():
    cfg = DataConfig(train_max_year=2000, calibration_years=(2001, 2002), validation_years=(2003, 2004), test_from_year=2005)
    assert [assign_split(date(y, 6, 1), cfg, PULLED_AT) for y in (2000, 2001, 2002, 2003, 2004, 2005)] == [
        "train",
        "calibration",
        "calibration",
        "validation",
        "validation",
        "test",
    ]
    short = replace(WITH_CUTOFF, mature_days=10)
    assert assign_split(PULLED_AT - timedelta(days=10), short, PULLED_AT) == "holdout"
    assert assign_split(PULLED_AT - timedelta(days=9), short, PULLED_AT) == "later"


SPLITS = ["train", "calibration", "validation", "test", "holdout", "later"]


def frame():
    return pd.DataFrame({"tmdb_id": range(len(SPLITS) * 2), "split": SPLITS * 2})


@pytest.mark.parametrize("name", ["test", "holdout"])
def test_locked_without_final(name):
    df = frame()
    with pytest.raises(TestSetLocked) as err:
        load_split(df, name)
    message = str(err.value)
    assert not re.search(r"\d", message.replace(name, ""))  # no counts or ids in the message
    with pytest.raises(TestSetLocked):
        load_split(df, name, final=False)
    opened = load_split(df, name, final=True)
    assert set(opened["split"]) == {name} and len(opened) == 2


def test_locked_check_happens_before_touching_rows():
    with pytest.raises(TestSetLocked):
        load_split(pd.DataFrame({"tmdb_id": [1]}), "test")  # no split column: still the lock error, not KeyError


@pytest.mark.parametrize("name", ["train", "calibration", "validation"])
def test_open_splits_return_a_filtered_copy(name):
    df = frame()
    out = load_split(df, name)
    assert set(out["split"]) == {name} and len(out) == 2
    out["split"] = "changed"
    assert set(df["split"]) == set(SPLITS)  # the original is untouched


@pytest.mark.parametrize("name", ["later", "nonsense", ""])
@pytest.mark.parametrize("final", [False, True])
def test_later_and_unknown_names_raise_value_error(name, final):
    with pytest.raises(ValueError):
        load_split(frame(), name, final=final)


def test_missing_split_column_raises_key_error():
    with pytest.raises(KeyError):
        load_split(pd.DataFrame({"tmdb_id": [1]}), "train")


def _opens_locked_sets(source: str) -> bool:
    """True if the code passes a final flag that is not literally False to load_split, in any call form."""
    if re.search(r"final\s*=\s*True", source):
        return True
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (getattr(func, "id", None) == "load_split" or getattr(func, "attr", None) == "load_split"):
            continue
        if len(node.args) >= 3:  # final passed by position
            return True
        for kw in node.keywords:
            if kw.arg is None or (kw.arg == "final" and not (isinstance(kw.value, ast.Constant) and kw.value.value is False)):
                return True  # final=<anything but False>, or **kwargs that could hide it
    return False


def _stray_files(root: Path, allowed: Path) -> list[str]:
    return [
        str(p.relative_to(root))
        for p in root.rglob("*.py")
        if p.resolve() != allowed.resolve() and _opens_locked_sets(p.read_text(encoding="utf-8"))
    ]


@pytest.mark.parametrize(
    "line",
    [
        "load_split(df, 'test', final=True)",
        "load_split(df, 'test', final = True)",
        "load_split(df, 'holdout', True)",
        "splits.load_split(df, 'test', True)",
        "load_split(df, 'test', final=flag)",
        "load_split(df, 'test', **opts)",
    ],
)
def test_scan_catches_every_way_to_open_the_locked_sets(tmp_path, line):
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "final.py").write_text(line + "\n", encoding="utf-8")  # the one allowed file
    (tmp_path / "stray.py").write_text(line + "\n", encoding="utf-8")
    assert _stray_files(tmp_path, tmp_path / "eval" / "final.py") == ["stray.py"]


def test_scan_leaves_ordinary_calls_alone(tmp_path):
    (tmp_path / "ok.py").write_text("load_split(df, 'train')\nload_split(df, 'train', final=False)\n", encoding="utf-8")
    assert _stray_files(tmp_path, tmp_path / "eval" / "final.py") == []


def test_only_final_module_passes_final_true():
    visited = [p.name for p in PACKAGE.rglob("*.py")]
    assert "splits.py" in visited and "build_db.py" in visited
    assert _stray_files(PACKAGE, PACKAGE / "eval" / "final.py") == []
