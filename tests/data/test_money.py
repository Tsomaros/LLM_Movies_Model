from pathlib import Path

import pytest

from llm_movies.data.money import clean_amount, load_cpi, load_fx, to_2025_usd, to_usd


def test_zero_and_placeholder_are_unknown():
    assert clean_amount(0) is None and clean_amount(7) is None and clean_amount(None) is None and clean_amount(1_000) == 1_000


def test_2025_dollars_scale_by_cpi():
    cpi = {2000: 172.2, 2025: 322.0}
    assert to_2025_usd(100.0, 2000, cpi) == pytest.approx(100 * 322.0 / 172.2)


def test_unknown_currency_is_none():
    assert to_usd(1e6, "XYZ", 2010, {("EUR", 2010): 0.75}) is None


def test_eur_converts():
    assert to_usd(75.0, "EUR", 2010, {("EUR", 2010): 0.75}) == pytest.approx(100.0)


def test_nan_and_just_below_threshold_are_unknown():
    # Ruling: NaN is unknown (pandas feeds Task 7); MIN_MONEY = 1_000 is the boundary.
    assert clean_amount(float("nan")) is None
    assert clean_amount(999) is None
    assert clean_amount(999.99) is None
    assert clean_amount(1_000.0) == 1_000.0
    assert clean_amount(150_000_000) == 150_000_000


def test_usd_passes_through_and_none_stays_none():
    assert to_usd(5e6, "USD", 1975, {}) == 5e6
    assert to_usd(None, "EUR", 2010, {("EUR", 2010): 0.75}) is None


def test_known_currency_in_missing_year_is_none():
    # Years FRED lacks stay absent, so no rate may be invented.
    assert to_usd(1e6, "EUR", 1998, {("EUR", 2010): 0.75}) is None


def test_2025_dollars_edge_cases():
    cpi = {2000: 172.2, 2025: 322.0}
    assert to_2025_usd(None, 2000, cpi) is None
    assert to_2025_usd(100.0, 1850, cpi) is None
    assert to_2025_usd(100.0, 2000, {2000: 172.2}) is None
    assert to_2025_usd(100.0, 2025, cpi) == pytest.approx(100.0)


def test_cpi_file_has_1960_to_2026():
    cpi = load_cpi(Path("data/cpi_us_annual.csv"))
    assert min(cpi) == 1960 and max(cpi) >= 2026


def test_cpi_file_is_complete_and_ordered():
    cpi = load_cpi(Path("data/cpi_us_annual.csv"))
    assert sorted(cpi) == list(range(1960, max(cpi) + 1))  # no gap year
    assert cpi[2000] == pytest.approx(172.2, abs=0.1)  # CPI-U 2000 annual average, as in the brief
    assert cpi[2025] > cpi[2000] > cpi[1960]


def test_fx_file_all_currencies_cover_1999_to_2026_and_gaps_stay_absent():
    fx = load_fx(Path("data/fx_usd_annual.csv"))
    for cur in ("EUR", "GBP", "JPY", "INR", "CNY", "KRW", "AUD", "CAD", "FRF", "DEM", "ITL", "ESP"):
        for year in range(1999, 2027):
            assert (cur, year) in fx and fx[(cur, year)] > 0, (cur, year)
        assert fx[(cur, 2026)] == fx[(cur, 2025)]  # 2026 repeats 2025 (ruling)
    # The file's own comment rows: EUR starts 1999, GBP starts 1971; earlier years are not invented.
    assert ("EUR", 1998) not in fx
    assert ("GBP", 1970) not in fx


def test_fx_file_magnitudes_per_usd():
    fx = load_fx(Path("data/fx_usd_annual.csv"))
    # Real 2010 averages (units per USD): EUR ~0.75, INR ~46, CNY ~6.8, KRW ~1150, AUD ~1.09, CAD ~1.03.
    assert 0.70 < fx[("EUR", 2010)] < 0.80
    assert 40 < fx[("INR", 2010)] < 50
    assert 6.5 < fx[("CNY", 2010)] < 7.0
    assert 1000 < fx[("KRW", 2010)] < 1300
    assert 1.0 < fx[("AUD", 2010)] < 1.2
    assert 0.95 < fx[("CAD", 2010)] < 1.1


def test_fx_file_legacy_rates():
    fx = load_fx(Path("data/fx_usd_annual.csv"))
    for year in (1999, 2010, 2025):
        assert fx[("ITL", year)] == pytest.approx(fx[("EUR", year)] * 1936.27, rel=1e-4)
        assert fx[("ESP", year)] == pytest.approx(fx[("EUR", year)] * 166.386, rel=1e-4)


def test_fx_file_direction_and_legacy_rates():
    fx = load_fx(Path("data/fx_usd_annual.csv"))
    assert fx[("GBP", 2010)] < 1
    assert fx[("JPY", 2010)] > 50
    for year in (1999, 2010, 2025):
        assert fx[("FRF", year)] == pytest.approx(fx[("EUR", year)] * 6.55957, rel=1e-4)
        assert fx[("DEM", year)] == pytest.approx(fx[("EUR", year)] * 1.95583, rel=1e-4)
