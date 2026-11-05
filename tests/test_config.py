import dataclasses
from datetime import date
from pathlib import Path

import pytest

from llm_movies.config import DataConfig, load_config


def test_defaults_match_spec():
    c = load_config(Path("config/data.yaml"))
    assert (c.train_max_year, c.calibration_years, c.validation_years, c.test_from_year) == (2013, (2014, 2015), (2016, 2018), 2019)
    assert c.base_model_cutoff is None and c.min_votes == 50 and c.mature_days == 182 and c.shrink_films == 4


def test_chunk_size_is_50000():
    # brief: chunk_size=50_000 (not covered by the spec-defaults test above)
    assert load_config(Path("config/data.yaml")).chunk_size == 50_000


def test_dataclass_defaults_match_brief():
    # The code defaults are separate from the yaml; both must carry the brief's values.
    c = DataConfig()
    assert (c.train_max_year, c.calibration_years, c.validation_years, c.test_from_year) == (2013, (2014, 2015), (2016, 2018), 2019)
    assert (c.base_model_cutoff, c.min_votes, c.chunk_size, c.mature_days, c.shrink_films) == (None, 50, 50_000, 182, 4)


def test_yaml_and_dataclass_defaults_agree():
    assert load_config(Path("config/data.yaml")) == DataConfig()


def test_config_is_frozen():
    # brief: frozen dataclass, so a stage cannot shift a split year at run time
    c = load_config(Path("config/data.yaml"))
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.train_max_year = 2020


def test_year_ranges_are_tuples_and_ordered():
    # splits must not overlap: train <= 2013 < calibration < validation < test_from_year
    c = load_config(Path("config/data.yaml"))
    assert isinstance(c.calibration_years, tuple) and isinstance(c.validation_years, tuple)
    assert c.train_max_year < c.calibration_years[0] <= c.calibration_years[1]
    assert c.calibration_years[1] < c.validation_years[0] <= c.validation_years[1] < c.test_from_year


def test_cutoff_parses_as_date(tmp_path):
    p = tmp_path / "d.yaml"
    p.write_text(Path("config/data.yaml").read_text().replace("base_model_cutoff: null", "base_model_cutoff: 2024-06-30"))
    assert load_config(p).base_model_cutoff == date(2024, 6, 30)
