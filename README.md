# LLM Movie Predictor

A small open LLM, fine-tuned with LoRA and a quantile regression head, predicts a planned film's worldwide revenue and
audience score as calibrated p10/p50/p90 ranges and explains in English which inputs raise or lower them. A learning and
portfolio project. See `CLAUDE.md` and `docs/superpowers/specs/` for the design.

## Development

```
uv sync
uv run pytest
uv run ruff check .
```
