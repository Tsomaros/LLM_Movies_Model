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
    assert cfg.run_name == "base" and cfg.base_model and cfg.lora_r == 16 and cfg.max_len == 768


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
