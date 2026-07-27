"""
Verify the indicators match Pine's `ta.*` semantics, and that the O(1)
incremental states are numerically identical to a full recomputation.
"""

import numpy as np
import pytest

import indicators as ind


@pytest.fixture
def series():
    rng = np.random.default_rng(42)
    return np.cumsum(rng.normal(0, 2, 300)) + 500.0


# --------------------------------------------------------------------- basics
def test_sma_matches_manual(series):
    out = ind.sma(series, 10)
    assert np.isnan(out[:9]).all()
    assert out[9] == pytest.approx(series[:10].mean())
    assert out[-1] == pytest.approx(series[-10:].mean())


def test_ema_seeded_with_sma(series):
    out = ind.ema(series, 20)
    assert np.isnan(out[:19]).all()
    # Pine seeds the EMA with the SMA of the first `length` values
    assert out[19] == pytest.approx(series[:20].mean())
    alpha = 2 / 21
    assert out[20] == pytest.approx(alpha * series[20] + (1 - alpha) * out[19])


def test_rma_is_wilder(series):
    out = ind.rma(series, 14)
    assert out[13] == pytest.approx(series[:14].mean())
    assert out[14] == pytest.approx((out[13] * 13 + series[14]) / 14)


def test_rsi_range_and_known_case():
    # strictly rising series -> RSI pinned at 100
    rising = np.arange(1, 60, dtype=float)
    out = ind.rsi(rising, 14)
    assert out[-1] == pytest.approx(100.0)

    falling = np.arange(60, 1, -1, dtype=float)
    out = ind.rsi(falling, 14)
    assert out[-1] == pytest.approx(0.0)


def test_rsi_bounds(series):
    out = ind.rsi(series, 14)
    valid = out[~np.isnan(out)]
    assert len(valid) > 250
    assert valid.min() >= 0.0 and valid.max() <= 100.0


def test_macd_hist_is_line_minus_signal(series):
    line, sig, hist = ind.macd(series, 12, 26, 9)
    ok = ~np.isnan(hist)
    assert np.allclose(hist[ok], line[ok] - sig[ok])
    # macd line = ema(12) - ema(26)
    ef, es = ind.ema(series, 12), ind.ema(series, 26)
    v = ~np.isnan(line)
    assert np.allclose(line[v], (ef - es)[v])


def test_highest_excludes_nothing(series):
    out = ind.highest(series, 26)
    assert out[-1] == pytest.approx(series[-26:].max())


# ----------------------------------------------------- incremental == full
def test_ema_state_matches_full_recompute(series):
    hist, nxt = series[:-1], series[-1]
    state = ind.build_ema_state(hist, 20)
    assert state.step(nxt) == pytest.approx(ind.ema(series, 20)[-1])


def test_rsi_state_matches_full_recompute(series):
    hist, nxt = series[:-1], series[-1]
    state = ind.build_rsi_state(hist, 14)
    assert state.step(nxt) == pytest.approx(ind.rsi(series, 14)[-1], abs=1e-9)


def test_macd_state_matches_full_recompute(series):
    hist, nxt = series[:-1], series[-1]
    state = ind.build_macd_state(hist, 12, 26, 9)
    _, _, full = ind.macd(series, 12, 26, 9)
    assert state.step(nxt) == pytest.approx(full[-1], abs=1e-9)


def test_sma_state_matches_full_recompute(series):
    hist, nxt = series[:-1], series[-1]
    state = ind.build_sma_state(hist, 10)
    assert state.step(nxt) == pytest.approx(ind.sma(series, 10)[-1])


def test_state_step_is_pure(series):
    """Stepping must not mutate the frozen state - the scanner calls it per bar."""
    state = ind.build_ema_state(series, 20)
    first = state.step(600.0)
    state.step(1.0)
    state.step(999.0)
    assert state.step(600.0) == pytest.approx(first)


def test_states_none_when_history_too_short():
    short = np.arange(5, dtype=float)
    assert ind.build_ema_state(short, 50) is None
    assert ind.build_rsi_state(short, 14) is None
    assert ind.build_macd_state(short, 12, 26, 9) is None
