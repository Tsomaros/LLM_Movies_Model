"""Money helpers: unknown-amount cleaning, FX to USD, and CPI to 2025 dollars."""

import csv
import math
from pathlib import Path

MIN_MONEY = 1_000


def _rows(path: Path):
    with open(path, newline="", encoding="utf-8") as f:
        yield from csv.DictReader(line for line in f if not line.startswith("#"))


def load_cpi(path: Path) -> dict[int, float]:
    return {int(r["year"]): float(r["cpi"]) for r in _rows(path)}


def load_fx(path: Path) -> dict[tuple[str, int], float]:
    """Units of currency per 1 USD, keyed by (currency, year)."""
    return {(r["currency"], int(r["year"])): float(r["per_usd"]) for r in _rows(path)}


def clean_amount(amount: float | None) -> float | None:
    """TMDB 0, missing, NaN and placeholders below MIN_MONEY mean unknown."""
    if amount is None or math.isnan(amount) or amount < MIN_MONEY:
        return None
    return amount


def to_usd(amount: float | None, currency: str, year: int, fx: dict[tuple[str, int], float]) -> float | None:
    if amount is None:
        return None
    if currency == "USD":
        return amount
    rate = fx.get((currency, year))
    return None if rate is None else amount / rate


def to_2025_usd(amount_usd: float | None, year: int, cpi: dict[int, float]) -> float | None:
    if amount_usd is None or year not in cpi or 2025 not in cpi:
        return None
    return amount_usd * cpi[2025] / cpi[year]
