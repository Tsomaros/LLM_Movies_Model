"""Prompt token lengths over the open splits, to set the training max_len.

python -m llm_movies.eval.tokens --tokenizer ID [--parquet PATH]

    token_lengths(parquet, tokenizer) -> {"p50", "p90", "p99", "max", "recommended_max_len"}

`tokenizer` is a Hub id (str) or a ready tokenizer. Every open-split film in films.parquet is rendered with nothing
hidden; locked.parquet is never read. recommended_max_len = ceil(p99 / 64) * 64.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from llm_movies.data.export import FILMS_FILE
from llm_movies.prompt.render import render
from llm_movies.prompt.resolve import ResolvedFilm

PAD_MULTIPLE = 64


def token_lengths(parquet: Path, tokenizer) -> dict[str, float]:
    if isinstance(tokenizer, str):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    path = Path(parquet)
    df = pd.read_parquet(path / FILMS_FILE if path.is_dir() else path, columns=["resolved_json"])
    texts = [render(ResolvedFilm.from_json(j)) for j in df["resolved_json"]]
    lengths = np.array([len(ids) for ids in tokenizer(texts, add_special_tokens=True)["input_ids"]])
    p99 = float(np.percentile(lengths, 99))
    return {
        "p50": float(np.percentile(lengths, 50)),
        "p90": float(np.percentile(lengths, 90)),
        "p99": p99,
        "max": float(lengths.max()),
        "recommended_max_len": math.ceil(p99 / PAD_MULTIPLE) * PAD_MULTIPLE,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--parquet", type=Path, default=Path("data/export"))
    args = p.parse_args(argv)
    for key, value in token_lengths(args.parquet, args.tokenizer).items():
        print(f"{key}: {value:g}")


if __name__ == "__main__":
    main()
