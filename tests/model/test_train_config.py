import dataclasses
from pathlib import Path

import pytest

from llm_movies.model.config import TrainConfig, load_train_config


def test_extends_overrides(tmp_path):
    (tmp_path / "base.yaml").write_text("run_name: base\nbase_model: some/model\nlora_r: 8\nepochs: 3\n", encoding="utf-8")
    child = tmp_path / "child.yaml"
    child.write_text("extends: base.yaml\nrun_name: child\nlora_r: 4\ndrop_groups: [tagline]\n", encoding="utf-8")
    cfg = load_train_config(child)
    assert cfg.run_name == "child"  # child overrides
    assert cfg.lora_r == 4
    assert cfg.epochs == 3  # inherited
    assert cfg.base_model == "some/model"  # inherited
    assert cfg.lora_alpha == TrainConfig.__dataclass_fields__["lora_alpha"].default  # dataclass default
    assert cfg.drop_groups == ("tagline",)


def test_extends_resolves_next_to_child(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "base.yaml").write_text("run_name: b\nbase_model: m\nseed: 7\n", encoding="utf-8")
    (sub / "c.yaml").write_text("extends: base.yaml\nrun_name: c\n", encoding="utf-8")
    assert load_train_config(sub / "c.yaml").seed == 7


def test_repo_base_config_loads():
    cfg = load_train_config("configs/train/base.yaml")
    assert cfg.run_name == "base" and cfg.base_model and cfg.lora_r == 16 and cfg.max_len == 832  # Task 11 Step 1
    assert cfg.val_sample == 2000  # early-stopping checks on a fixed 2000-film validation sample


def test_unknown_key_raises(tmp_path):
    p = tmp_path / "x.yaml"
    p.write_text("run_name: x\nbase_model: m\nlora_rank: 4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="lora_rank"):
        load_train_config(p)


def test_bad_drop_group_raises(tmp_path):
    p = tmp_path / "x.yaml"
    p.write_text("run_name: x\nbase_model: m\ndrop_groups: [tagline, nonsense]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="nonsense"):
        load_train_config(p)


def test_val_sample_defaults_to_the_full_frame():
    assert TrainConfig(run_name="x", base_model="m").val_sample is None


@pytest.mark.parametrize("bad", [0, -5, 2.5, True, "2000"])
def test_bad_val_sample_raises(bad):
    with pytest.raises(ValueError, match="val_sample"):
        TrainConfig(run_name="x", base_model="m", val_sample=bad)


def test_val_sample_null_in_yaml_is_none(tmp_path):
    p = tmp_path / "x.yaml"
    p.write_text("run_name: x\nbase_model: m\nval_sample: null\n", encoding="utf-8")
    assert load_train_config(p).val_sample is None
    p.write_text("run_name: x\nbase_model: m\nval_sample: 300\n", encoding="utf-8")
    assert load_train_config(p).val_sample == 300


# ---- Task 11: the experiment configs in configs/train ----

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "train"  # independent of the working directory
EXPERIMENTS = {
    "e1_small": {},
    "e1_large": {"base_model": "Qwen/Qwen3-1.7B-Base"},
    "e2_anon": {"anonymize": True},
    "e3_notext": {"drop_groups": ("synopsis", "tagline")},
    "e4_noleak": {"drop_groups": ("release_type", "distributor", "themes")},
    "e5_nosource": {"drop_groups": ("source_strength",)},
    "smoke": {"sample_frac": 0.01, "epochs": 1},  # Task 11 Step 4 smoke run: 1% of the films, 1 epoch
}


def test_the_config_folder_holds_base_and_exactly_the_experiments():
    assert sorted(p.stem for p in CONFIG_DIR.glob("*.yaml")) == sorted(["base", *EXPERIMENTS])


def test_base_is_the_smaller_chosen_model_on_30_percent_of_the_films():
    base = load_train_config(CONFIG_DIR / "base.yaml")
    assert (base.run_name, base.base_model, base.sample_frac) == ("base", "Qwen/Qwen3-0.6B-Base", 0.3)
    assert base.drop_groups == () and base.anonymize is False


@pytest.mark.parametrize("name", list(EXPERIMENTS))
def test_each_experiment_differs_from_base_only_in_its_run_name_and_its_own_change(name):
    base = load_train_config(CONFIG_DIR / "base.yaml")
    cfg = load_train_config(CONFIG_DIR / f"{name}.yaml")
    expected = dataclasses.replace(base, run_name=name, **EXPERIMENTS[name])
    assert cfg == expected


def test_run_names_are_unique_and_match_the_file_names():
    paths = sorted(CONFIG_DIR.glob("*.yaml"))
    assert len(paths) == 1 + len(EXPERIMENTS)  # an empty or wrong folder must not let the checks below pass over nothing
    names = [load_train_config(p).run_name for p in paths]
    assert len(names) == len(set(names))
    assert names == [p.stem for p in paths]
