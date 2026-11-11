"""What the trainer feeds the predictor: frames, target scaling, per-item rendering, batching, sampling.

    load_frames(path, cfg)                 -> (train_df, validation_df); reads films.parquet only, never locked.parquet
    TargetScaler.fit(train_df)             -> mean/std of revenue_log and score over known train rows
    FilmDataset(df, scaler, cfg, train)    -> items {"text", "target", "mask", "weight"}; .set_epoch(epoch)
    Collator(tokenizer, max_len)           -> input_ids, attention_mask, target, mask, weight (always right-padded)
    revenue_sampler(df, factor, seed, epoch)

A missing target is 0.0 with mask False and a missing score weight is 0.0, never NaN. Training hides each RANDOM_HIDE
group independently with probability hide_p, from an RNG seeded by (seed, epoch, index), so a resumed run renders the same
text. Validation hides only cfg.drop_groups.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler, WeightedRandomSampler

from llm_movies.data.export import FILMS_FILE
from llm_movies.model import TARGETS
from llm_movies.prompt.render import RANDOM_HIDE, render
from llm_movies.prompt.resolve import ResolvedFilm

_COLUMNS = ("revenue_log", "score")  # in TARGETS order


def load_frames(path: Path, cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train, validation) from films.parquet. `path` is that file or the folder holding it; calibration rows are dropped."""
    path = Path(path)
    df = pd.read_parquet(path / FILMS_FILE if path.is_dir() else path)
    train = df[df["split"] == "train"]
    if cfg.sample_frac < 1.0:
        train = train.sample(frac=cfg.sample_frac, random_state=cfg.seed)
    return train.reset_index(drop=True), df[df["split"] == "validation"].reset_index(drop=True)


def validation_sample(val_df: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    """A fixed sample of `n` validation films for the early-stopping checks, or the whole frame when `n` is None or not
    smaller than it. Stratified on having revenue (its share is kept, rounded) and drawn after sorting by tmdb_id, so the
    same seed gives the same films whatever the row order. A drawn sample comes back with whole rows, in tmdb_id order."""
    if n is None or n >= len(val_df):
        return val_df
    df = val_df.sort_values("tmdb_id", kind="stable").reset_index(drop=True)
    has = df["revenue_log"].notna()
    n_rev = round(n * has.mean())  # n < len(df) keeps both draws within their stratum
    parts = [df[has].sample(n=n_rev, random_state=seed), df[~has].sample(n=n - n_rev, random_state=seed)]
    return pd.concat(parts).sort_values("tmdb_id", kind="stable").reset_index(drop=True)


class TargetScaler:
    def __init__(self, mean: list[float], std: list[float]):
        self.mean, self.std = list(mean), list(std)

    @classmethod
    def fit(cls, train_df: pd.DataFrame) -> TargetScaler:
        cols = [train_df[c].dropna() for c in _COLUMNS]
        means = [float(c.mean()) for c in cols]
        stds = [float(c.std(ddof=0)) for c in cols]
        means = [m if math.isfinite(m) else 0.0 for m in means]  # an all-null column
        stds = [s if math.isfinite(s) and s >= 1e-8 else 1.0 for s in stds]  # all-null or constant column
        return cls(means, stds)

    def transform(self, df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        """(target[N,2] standardised, 0.0 where unknown; mask[N,2] True where known)."""
        raw = np.stack([pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64) for c in _COLUMNS], axis=1)
        known = ~np.isnan(raw)
        scaled = np.where(known, (raw - np.array(self.mean)) / np.array(self.std), 0.0)
        return torch.tensor(scaled, dtype=torch.float32), torch.tensor(known)

    def inverse(self, pred: torch.Tensor) -> torch.Tensor:
        """pred[..., 2, 3] in standardised units -> original units (revenue_log, score)."""
        mean = torch.tensor(self.mean, dtype=pred.dtype, device=pred.device)[:, None]
        std = torch.tensor(self.std, dtype=pred.dtype, device=pred.device)[:, None]
        return pred * std + mean

    def to_json(self) -> str:
        return json.dumps({"targets": list(TARGETS), "mean": self.mean, "std": self.std})

    @classmethod
    def from_json(cls, text: str) -> TargetScaler:
        raw = json.loads(text)
        return cls(raw["mean"], raw["std"])


class FilmDataset(Dataset):
    def __init__(self, df: pd.DataFrame, scaler: TargetScaler, cfg, train: bool):
        self.cfg, self.train, self.epoch = cfg, train, 0
        self.json = df["resolved_json"].tolist()
        self.target, self.mask = scaler.transform(df)
        score_weight = pd.to_numeric(df["score_weight"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        self.weight = torch.stack([torch.ones(len(df)), torch.tensor(score_weight)], dim=1)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.json)

    def _hidden(self, index: int) -> frozenset[str]:
        hidden = set(self.cfg.drop_groups)
        if self.train:
            draws = np.random.default_rng([self.cfg.seed, self.epoch, index]).random(len(RANDOM_HIDE))
            hidden |= {g for g, u in zip(RANDOM_HIDE, draws, strict=True) if u < self.cfg.hide_p}
        return frozenset(hidden)

    def __getitem__(self, index: int) -> dict:
        film = ResolvedFilm.from_json(self.json[index])
        text = render(film, self._hidden(index), anonymize=self.cfg.anonymize)
        return {"text": text, "target": self.target[index], "mask": self.mask[index], "weight": self.weight[index]}


class Collator:
    """Tokenizes a batch, always right-padded and right-truncated (the predictor pools at the last real token)."""

    def __init__(self, tokenizer, max_len: int):
        # a deep copy, so forcing the sides or the pad token never changes the caller's tokenizer
        self.tokenizer = copy.deepcopy(tokenizer)
        if getattr(self.tokenizer, "pad_token", None) is None:
            if getattr(self.tokenizer, "eos_token", None) is None:
                raise ValueError("tokenizer has neither a pad token nor an eos token to pad with")
            self.tokenizer.pad_token = self.tokenizer.eos_token  # base Llama/Qwen checkpoints ship without a pad token
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "right"
        self.max_len = max_len

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        texts = [item["text"] for item in batch]
        if any(not t.strip() for t in texts):
            raise ValueError("empty text in batch")
        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
        if (enc["attention_mask"].sum(dim=1) == 0).any():
            raise ValueError("a text produced no tokens (all-zero attention row)")
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "target": torch.stack([item["target"] for item in batch]),
            "mask": torch.stack([item["mask"] for item in batch]),
            "weight": torch.stack([item["weight"] for item in batch]),
        }


def revenue_sampler(df: pd.DataFrame, factor: float, seed: int, epoch: int) -> Sampler:
    """len(df) draws per epoch with replacement; a film with a known revenue is `factor` times as likely."""
    has_revenue = df["revenue_log"].notna().to_numpy(dtype=np.float64)
    weights = torch.tensor(1.0 + (factor - 1.0) * has_revenue, dtype=torch.double)
    generator = torch.Generator().manual_seed(int(np.random.SeedSequence([seed, epoch]).generate_state(1)[0]))
    return WeightedRandomSampler(weights, num_samples=len(df), replacement=True, generator=generator)
