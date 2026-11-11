# LLM Movie Predictor

A small open LLM, fine-tuned with LoRA and a quantile regression head, predicts a planned film's worldwide revenue and
audience score as calibrated p10/p50/p90 ranges and explains in English which inputs raise or lower them. A learning and
portfolio project. See `CLAUDE.md` and `docs/superpowers/specs/` for the design.

## Development

```
uv sync --extra train --extra cpu
uv run pytest
uv run ruff check .
```

PyTorch comes from one of two mutually exclusive extras: `cpu` (tests, CI, CPU runtimes) or `cu132` (a Colab/Kaggle T4). Add
`--extra train` for transformers, peft and accelerate. `uv run` keeps the extras installed by the last `uv sync`.
