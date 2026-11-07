import dataclasses
from datetime import date
from pathlib import Path

import pytest

from llm_movies.config import DataConfig, load_config


def test_defaults_match_spec():
    c = load_config(Path("config/data.yaml"))
    assert (c.train_max_year, c.calibration_years, c.validation_years, c.test_from_year) == (2013, (2014, 2015), (2016, 2018), 2019)
    assert c.base_model_cutoff is None and c.min_votes == 50 and c.mature_days == 182 and c.shrink_films == 4


def test_chunk_size_is_5000():
    # owner's decision: 5,000 ids per chunk, so a dead Colab session loses about 2 minutes of pulling
    assert load_config(Path("config/data.yaml")).chunk_size == 5_000


def test_dataclass_defaults_match_brief():
    # The code defaults are separate from the yaml; both must carry the brief's values.
    c = DataConfig()
    assert (c.train_max_year, c.calibration_years, c.validation_years, c.test_from_year) == (2013, (2014, 2015), (2016, 2018), 2019)
    assert (c.base_model_cutoff, c.min_votes, c.chunk_size, c.mature_days, c.shrink_films) == (None, 50, 5_000, 182, 4)


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


@pytest.mark.parametrize(
    "changes",
    [
        {"calibration_years": (2015, 2016), "validation_years": (2017, 2018)},  # gap before calibration (2014)
        {"validation_years": (2017, 2018)},  # gap before validation (2016)
        {"test_from_year": 2020},  # gap before test (2019)
        {"validation_years": (2015, 2018)},  # overlap with calibration
        {"test_from_year": 2018},  # overlap of test with validation
        # reversed pairs where every seam still holds, so only the ordering check can catch them
        {"train_max_year": 2014, "calibration_years": (2015, 2014), "validation_years": (2015, 2018)},
        {"validation_years": (2016, 2015), "test_from_year": 2016},
    ],
)
def test_non_contiguous_bands_are_rejected(changes):
    with pytest.raises(ValueError, match="contiguous"):
        dataclasses.replace(DataConfig(), **changes)


def test_other_contiguous_bands_are_accepted():
    c = DataConfig(train_max_year=2000, calibration_years=(2001, 2001), validation_years=(2002, 2004), test_from_year=2005)
    assert c.test_from_year == 2005
    # one-year validation band is also valid (v0 == v1)
    DataConfig(train_max_year=2000, calibration_years=(2001, 2001), validation_years=(2002, 2002), test_from_year=2003)


def test_bad_yaml_is_rejected(tmp_path):
    p = tmp_path / "d.yaml"
    p.write_text(Path("config/data.yaml").read_text().replace("test_from_year: 2019", "test_from_year: 2020"))
    with pytest.raises(ValueError, match="contiguous"):
        load_config(p)
