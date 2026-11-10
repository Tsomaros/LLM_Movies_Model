import copy
from dataclasses import replace
from datetime import date

import pandas as pd
import pytest
import torch

from llm_movies.model import data as data_module
from llm_movies.model.config import TrainConfig
from llm_movies.model.data import Collator, FilmDataset, TargetScaler, load_frames, revenue_sampler
from llm_movies.prompt.render import NOT_GIVEN, RANDOM_HIDE, render
from llm_movies.prompt.resolve import ResolvedFilm

CFG = TrainConfig(run_name="t", base_model="unused", seed=7)


def _person(name):
    return {"name": name, "films": 5, "revenue_usd": 1e8, "score": 7.0, "wins": 1, "nominations": 2}


def _film(year=2010):
    return ResolvedFilm(
        genres=["Drama"], release_year=year, release_month=5, release_day=14, runtime=110, language="en",
        certification="PG-13", release_type="Wide", countries=["US"], themes=["Heist"], budget=5e7,
        synopsis="A crew plans one last job.", tagline="One last job.", source_material="Original",
        franchise="Heist Saga", installment=2, director=_person("Dina Dir"), writers=[_person("Wanda Wri")],
        cast=[_person("Carl Cast")], producers=[_person("Pat Pro")], composer=_person("Cora Comp"),
        distributor={"name": "Big Dist", "films": 40, "revenue_usd": 1e9, "score": 6.5},
        companies=[{"name": "Studio One", "films": 9, "revenue_usd": 5e8, "score": 6.8}], competition=3,
        collaborations=1, director_age=45, lead_age=38, source_strength=2,
    )


def _frame():
    # (split, revenue_log, score, score_weight)
    spec = [
        ("train", 18.0, 6.0, 5.0), ("train", 19.0, None, None), ("train", None, 7.0, 4.0), ("train", 20.0, 8.0, 6.0),
        ("calibration", 17.0, 5.0, 3.0), ("validation", 18.5, 6.5, 5.5), ("validation", None, None, None),
    ]
    rows = [
        {
            "tmdb_id": i + 1, "release_date": date(2010 + i, 5, 14), "split": split,
            "resolved_json": _film(2010 + i).to_json(), "revenue_log": rev, "score": score, "score_weight": w,
        }
        for i, (split, rev, score, w) in enumerate(spec)
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def data_dir(tmp_path):
    _frame().to_parquet(tmp_path / "films.parquet")
    return tmp_path


@pytest.fixture
def frames(data_dir):
    return load_frames(data_dir, CFG)


def _spy_hidden(monkeypatch):
    seen = []
    real = data_module.render

    def spy(r, hidden, **kw):
        seen.append(frozenset(hidden))
        return real(r, hidden, **kw)

    monkeypatch.setattr(data_module, "render", spy)
    return seen


def test_load_frames_splits_and_accepts_file_or_dir(data_dir):
    train, val = load_frames(data_dir, CFG)
    assert set(train.split) == {"train"} and set(val.split) == {"validation"}  # calibration dropped
    train2, _ = load_frames(data_dir / "films.parquet", CFG)
    assert list(train2.tmdb_id) == list(train.tmdb_id)


def test_load_frames_sample_frac_is_seeded(data_dir):
    cfg = replace(CFG, sample_frac=0.5)
    a, _ = load_frames(data_dir, cfg)
    b, _ = load_frames(data_dir, cfg)
    assert len(a) == 2 and list(a.tmdb_id) == list(b.tmdb_id)


def test_locked_file_never_opened(tmp_path):
    _frame().to_parquet(tmp_path / "films.parquet")
    (tmp_path / "locked.parquet").write_bytes(b"not a parquet file")  # reading it would raise
    train, val = load_frames(tmp_path, CFG)
    assert len(train) == 4 and len(val) == 2


def test_validation_renders_without_random_hiding(frames, monkeypatch):
    train, val = frames
    seen = _spy_hidden(monkeypatch)
    ds = FilmDataset(val, TargetScaler.fit(train), CFG, train=False)
    ds.set_epoch(0)
    first, second = ds[0]["text"], ds[0]["text"]
    ds.set_epoch(3)
    assert ds[0]["text"] == first == second
    assert set(seen) == {frozenset()}
    full = render(ResolvedFilm.from_json(val.iloc[0].resolved_json), frozenset())
    assert first == full and NOT_GIVEN not in first  # the fully populated film shows everything


def test_drop_groups_always_hidden(frames, monkeypatch):
    train, val = frames
    cfg = replace(CFG, drop_groups=("budget", "themes"))
    seen = _spy_hidden(monkeypatch)
    for df, is_train in ((train, True), (val, False)):
        ds = FilmDataset(df, TargetScaler.fit(train), cfg, train=is_train)
        for epoch in range(3):
            ds.set_epoch(epoch)
            for i in range(len(df)):
                ds[i]
    assert seen and all({"budget", "themes"} <= h for h in seen)
    text = FilmDataset(val, TargetScaler.fit(train), cfg, train=False)[0]["text"]
    assert "Budget: not given" in text and "Themes: not given" in text


def test_hiding_rate_close_to_hide_p(frames, monkeypatch):
    train, _ = frames
    seen = _spy_hidden(monkeypatch)
    ds = FilmDataset(train.iloc[:1], TargetScaler.fit(train), CFG, train=True)
    n = 2000
    for epoch in range(n):
        ds.set_epoch(epoch)
        ds[0]
    assert len(seen) == n
    for g in RANDOM_HIDE:
        rate = sum(g in h for h in seen) / n
        assert abs(rate - CFG.hide_p) < 0.04, (g, rate)
    assert all(h <= set(RANDOM_HIDE) for h in seen)  # only random-hide groups, nothing else


def test_same_seed_epoch_index_same_text(frames):
    train, _ = frames
    scaler = TargetScaler.fit(train)
    a, b = FilmDataset(train, scaler, CFG, train=True), FilmDataset(train, scaler, CFG, train=True)
    a.set_epoch(2)
    b.set_epoch(2)
    assert [a[i]["text"] for i in range(len(train))] == [b[i]["text"] for i in range(len(train))]
    b.set_epoch(3)
    assert any(a[i]["text"] != b[i]["text"] for i in range(len(train)))  # a different epoch hides different groups
    other = FilmDataset(train, scaler, replace(CFG, seed=8), train=True)
    other.set_epoch(2)
    assert any(other[i]["text"] != a[i]["text"] for i in range(len(train)))


def test_item_targets_masks_and_weights(frames):
    train, _ = frames
    scaler = TargetScaler.fit(train)
    ds = FilmDataset(train, scaler, CFG, train=False)
    items = [ds[i] for i in range(len(train))]
    assert [it["mask"].tolist() for it in items] == [[True, True], [True, False], [False, True], [True, True]]
    for it in items:
        assert it["target"].shape == (2,) and it["mask"].dtype == torch.bool
        assert torch.isfinite(it["target"]).all() and torch.isfinite(it["weight"]).all()
        assert it["weight"][0] == 1.0
    assert items[1]["target"][1] == 0.0 and items[1]["weight"][1] == 0.0
    assert items[2]["target"][0] == 0.0 and items[2]["weight"][0] == 1.0
    assert items[0]["weight"][1] == pytest.approx(5.0)


def test_scaler_roundtrip(frames):
    train, _ = frames
    scaler = TargetScaler.fit(train)
    rev = train.revenue_log.dropna()
    assert scaler.mean[0] == pytest.approx(rev.mean()) and scaler.std[0] == pytest.approx(rev.std(ddof=0))
    target, mask = scaler.transform(train)
    assert target.shape == (len(train), 2) and mask.shape == (len(train), 2)
    assert target[mask[:, 0], 0].mean().abs() < 1e-5  # standardised over known rows
    back = scaler.inverse(target[:, :, None].repeat(1, 1, 3))
    assert back.shape == (len(train), 2, 3)
    for j, col in enumerate(("revenue_log", "score")):
        known = train[col].notna().to_numpy().copy()
        expected = torch.tensor(train[col].dropna().to_numpy().copy(), dtype=back.dtype)
        assert torch.allclose(back[known, j, 1], expected, atol=1e-4)
    again = TargetScaler.from_json(scaler.to_json())
    assert again.mean == pytest.approx(scaler.mean) and again.std == pytest.approx(scaler.std)


def test_sampler_oversamples_revenue():
    n, r = 20_000, 0.25
    has = torch.arange(n) % 4 == 0
    df = pd.DataFrame({"revenue_log": [1.0 if h else None for h in has.tolist()]})
    idx = list(revenue_sampler(df, 2.0, seed=1, epoch=0))
    assert len(idx) == n
    share = has[idx].float().mean().item()
    assert share == pytest.approx(2 * r / (1 + r), abs=0.03)
    assert idx == list(revenue_sampler(df, 2.0, seed=1, epoch=0))
    assert idx != list(revenue_sampler(df, 2.0, seed=1, epoch=1))


def _left(tok):
    t = copy.copy(tok)
    t.padding_side = "left"
    t.truncation_side = "left"
    return t


def _collate(tok, texts, max_len):
    batch = [{"text": t, "target": torch.zeros(2), "mask": torch.ones(2, dtype=torch.bool), "weight": torch.ones(2)}
             for t in texts]
    return Collator(tok, max_len)(batch)


def test_collator_shapes(tiny_tokenizer, golden_prompts):
    out = _collate(tiny_tokenizer, golden_prompts[:3], 64)
    assert set(out) == {"input_ids", "attention_mask", "target", "mask", "weight"}
    assert out["target"].shape == (3, 2) and out["mask"].dtype == torch.bool and out["weight"].shape == (3, 2)
    assert out["input_ids"].shape == out["attention_mask"].shape and out["input_ids"].shape[1] <= 64


def test_collator_forces_right_padding_and_truncation_on_left_tokenizer(tiny_tokenizer, golden_prompts):
    left = _left(tiny_tokenizer)
    long_ = golden_prompts[0]
    out = _collate(left, ["Release", long_], 16)
    mask = out["attention_mask"]
    assert mask.shape[1] == 16 and mask[1].sum() == 16 and 0 < mask[0].sum() < 16
    n = int(mask[0].sum())
    assert mask[0, :n].all() and not mask[0, n:].any()  # real tokens first, padding after
    assert out["input_ids"][0, n:].eq(left.pad_token_id).all()
    expected_head = tiny_tokenizer(long_, truncation=True, max_length=16)["input_ids"]
    assert out["input_ids"][1].tolist() == expected_head  # right truncation keeps the start of the text
    assert left.padding_side == "left" and left.truncation_side == "left"  # the caller's tokenizer is untouched
    assert tiny_tokenizer.padding_side == "right"


@pytest.mark.parametrize("bad", ["", "   ", " \n\t "])
def test_collator_rejects_empty_text(tiny_tokenizer, golden_prompts, bad):
    with pytest.raises(ValueError):
        _collate(tiny_tokenizer, [golden_prompts[0], bad], 32)


def test_collator_rejects_all_zero_attention_row(tiny_tokenizer, golden_prompts):
    class ZeroMask:
        padding_side = "right"
        truncation_side = "right"

        def __call__(self, texts, **kw):
            enc = tiny_tokenizer(texts, **kw)
            enc["attention_mask"] = torch.zeros_like(enc["attention_mask"])
            return enc

    with pytest.raises(ValueError):
        _collate(ZeroMask(), golden_prompts[:2], 32)
