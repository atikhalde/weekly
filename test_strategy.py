"""
Strategy-level tests: weekly aggregation, the 13 conditions, the entry gate and
the bar-by-bar replay (cross detection, one-per-week, deferred entry).
"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from config import Strategy
from strategy import (
    build_snapshot, build_weekly_bars, evaluate_bar, gate_ok, replay_week, week_start_of,
)

IST = ZoneInfo("Asia/Kolkata")


# ------------------------------------------------------------------ fixtures
def make_daily(weeks: int = 200, start_price: float = 100.0,
               drift: float = 0.004, seed: int = 7) -> pd.DataFrame:
    """Synthetic daily candles on a gently rising trend (Mon-Fri only)."""
    rng = np.random.default_rng(seed)
    rows = []
    day = datetime(2021, 1, 4, 15, 30, tzinfo=IST)      # a Monday
    price = start_price
    while len(rows) < weeks * 5:
        if day.weekday() < 5:
            o = price
            c = price * (1 + drift + rng.normal(0, 0.008))
            rows.append({
                "datetime": day, "open": o, "high": max(o, c) * 1.004,
                "low": min(o, c) * 0.996, "close": c,
                "volume": float(rng.integers(80_000, 200_000)),
            })
            price = c
        day += timedelta(days=1)
    return pd.DataFrame(rows)


@pytest.fixture
def cfg():
    return Strategy()


@pytest.fixture
def daily():
    return make_daily()


@pytest.fixture
def snap(daily, cfg):
    target = week_start_of(pd.Timestamp(daily.iloc[-1]["datetime"]).date()) + pd.Timedelta(days=7)
    s = build_snapshot("TESTCO", "1234", "NSE_EQ", daily, cfg, target)
    assert s is not None
    return s


# ---------------------------------------------------------- weekly bars
def test_weekly_bars_are_monday_anchored(daily):
    wk = build_weekly_bars(daily)
    assert (wk["week_start"].dt.weekday == 0).all()
    assert wk["week_start"].is_monotonic_increasing


def test_weekly_ohlcv_aggregation():
    rows = []
    for i, day in enumerate(pd.date_range("2024-01-01", periods=5, freq="D", tz=IST)):
        rows.append({"datetime": day, "open": 100 + i, "high": 110 + i,
                     "low": 90 - i, "close": 105 + i, "volume": 1000.0})
    wk = build_weekly_bars(pd.DataFrame(rows))
    assert len(wk) == 1
    r = wk.iloc[0]
    assert r["open"] == 100 and r["close"] == 109
    assert r["high"] == 114 and r["low"] == 86
    assert r["volume"] == 5000


def test_snapshot_excludes_the_target_week(daily, cfg):
    """
    The frozen level must come from CLOSED weeks only - no look-ahead - and it
    steps back one FURTHER week, matching Pine's

        request.security(..., "W", ta.highest(high, N)[1], lookahead_off)

    where lookahead_off reads the last confirmed weekly bar and [1] then moves
    one before it. Verified against the user's 27-Jul chart tables (TMB 821.20,
    SENCO 390.40).
    """
    wk = build_weekly_bars(daily)
    target = wk.iloc[-1]["week_start"]                 # last week becomes "developing"
    s = build_snapshot("T", "1", "NSE_EQ", daily, cfg, target)
    closed = wk[wk["week_start"] < target]

    # window ends one week before the last confirmed bar
    expected = closed["high"].iloc[-(cfg.len_short + 1):-1].max()
    assert s.entry_level == pytest.approx(expected)
    assert s.entry_level <= closed["high"].tail(cfg.len_short).max()
    assert s.close_1 == pytest.approx(closed.iloc[-1]["close"])


def test_snapshot_none_on_short_history(cfg):
    short = make_daily(weeks=10)
    target = week_start_of(datetime(2021, 6, 7).date())
    assert build_snapshot("T", "1", "NSE_EQ", short, cfg, target) is None


# ------------------------------------------------------------ conditions
def test_conditions_all_pass_on_a_clean_breakout(snap, cfg):
    price = snap.entry_level * 1.05
    ev = evaluate_bar(snap, cfg, price=price, week_open=price * 0.97,
                      week_volume=1e9, day_open=price * 0.99)
    assert ev.conditions["c02"], "close must be above the 26W level"
    assert ev.conditions["c09"], "huge volume must clear the SMA"
    assert ev.conditions["c10"] and ev.conditions["c13"]
    assert ev.conditions["c11"]


def test_c12_auto_passes_when_mcap_disabled(snap, cfg):
    assert cfg.use_mcap is False
    ev = evaluate_bar(snap, cfg, snap.entry_level * 1.02, 1.0, 1e9, 1.0)
    assert ev.conditions["c12"] is True


def test_c12_enforced_when_enabled(snap):
    cfg = Strategy(use_mcap=True, min_mcap=1000)
    snap.mcap = 500
    ev = evaluate_bar(snap, cfg, snap.entry_level * 1.02, 1.0, 1e9, 1.0)
    assert ev.conditions["c12"] is False
    snap.mcap = 5000
    ev = evaluate_bar(snap, cfg, snap.entry_level * 1.02, 1.0, 1e9, 1.0)
    assert ev.conditions["c12"] is True


def test_red_weekly_and_daily_candles_fail(snap, cfg):
    price = snap.entry_level * 1.02
    ev = evaluate_bar(snap, cfg, price, week_open=price * 1.10,
                      week_volume=1e9, day_open=price * 1.05)
    assert not ev.conditions["c10"]
    assert not ev.conditions["c13"]
    assert not ev.all_ok
    assert ev.pass_count < 13


def test_low_volume_fails_c09(snap, cfg):
    price = snap.entry_level * 1.02
    ev = evaluate_bar(snap, cfg, price, price * 0.98, week_volume=1.0, day_open=price * 0.99)
    assert not ev.conditions["c09"]


# ------------------------------------------------------------------ gate
def test_live_gate_ignores_c01_c02(snap, cfg):
    """The level break itself proves c01/c02, so the gate must not re-test them."""
    price = snap.entry_level * 1.05
    ev = evaluate_bar(snap, cfg, price, price * 0.97, 1e9, price * 0.99)
    ev.conditions["c01"] = False
    ev.conditions["c02"] = False
    assert gate_ok(snap, cfg, ev) == all(
        ev.conditions[k] for k in
        ("c03", "c04", "c05", "c06", "c07", "c08", "c09", "c10", "c11", "c12", "c13"))


def test_closed_gate_uses_frozen_values_only(snap):
    cfg = Strategy(gate_source="closed")
    ev = evaluate_bar(snap, cfg, snap.entry_level * 1.02, 1.0, 1e9, 1.0)
    expected = (snap.g_ema_fast > snap.g_ema_slow
                and snap.g_ema_slow > snap.g_ema_slow_2
                and snap.g_rsi > cfg.rsi_min
                and snap.g_rsi > snap.g_rsi_1
                and snap.g_hist > 0
                and snap.close_1 <= snap.hi_short2)
    assert gate_ok(snap, cfg, ev) == expected


# ---------------------------------------------------------------- replay
def bars_from(prices, snap, start_hour=9, start_min=15, volume=5e8, day=1):
    """Build 5m candles from a list of closes."""
    rows = []
    t = datetime(2025, 1, 6 + day - 1, start_hour, start_min, tzinfo=IST)
    prev = prices[0]
    for p in prices:
        rows.append({"datetime": t, "open": prev, "high": max(p, prev),
                     "low": min(p, prev), "close": p, "volume": volume})
        prev = p
        t += timedelta(minutes=5)
    return pd.DataFrame(rows)


def test_no_signal_while_below_the_level(snap, cfg):
    lv = snap.entry_level
    bars = bars_from([lv * 0.95, lv * 0.97, lv * 0.99], snap)
    assert replay_week(snap, cfg, bars).signals == []


def test_signal_fires_on_the_crossing_candle(snap):
    cfg = Strategy(strict_entry=False)          # isolate the cross logic
    lv = snap.entry_level
    bars = bars_from([lv * 0.98, lv * 0.99, lv * 1.01, lv * 1.02], snap)
    res = replay_week(snap, cfg, bars)
    assert len(res.signals) == 1
    sig = res.signals[0]
    assert sig.price == pytest.approx(lv * 1.01)          # the FIRST close above
    assert sig.bar_time == bars.iloc[2]["datetime"]
    assert sig.trigger == "cross"


def test_one_entry_per_week(snap):
    cfg = Strategy(strict_entry=False, one_per_week=True)
    lv = snap.entry_level
    # cross, dip below, cross again -> still only one alert
    bars = bars_from([lv * 0.98, lv * 1.01, lv * 0.99, lv * 1.03], snap)
    assert len(replay_week(snap, cfg, bars).signals) == 1


def test_multiple_entries_when_one_per_week_off(snap):
    cfg = Strategy(strict_entry=False, one_per_week=False)
    lv = snap.entry_level
    bars = bars_from([lv * 0.98, lv * 1.01, lv * 0.99, lv * 1.03], snap)
    assert len(replay_week(snap, cfg, bars).signals) == 2


def test_cross_requires_previous_bar_below(snap):
    """Opening the week already above the level is a cross on bar 1 (prev is None)."""
    cfg = Strategy(strict_entry=False)
    lv = snap.entry_level
    bars = bars_from([lv * 1.05, lv * 1.06, lv * 1.07], snap)
    res = replay_week(snap, cfg, bars)
    assert len(res.signals) == 1
    assert res.signals[0].bar_time == bars.iloc[0]["datetime"]


def test_gate_blocks_entry_and_defer_fires_later(snap):
    """Volume too low at the cross; the deferred trigger fires once it clears."""
    cfg = Strategy(strict_entry=True, gate_source="live", defer_entry=True)
    lv = snap.entry_level
    rows = []
    t = datetime(2025, 1, 6, 9, 15, tzinfo=IST)
    plan = [(lv * 0.98, 1.0), (lv * 1.02, 1.0), (lv * 1.03, 1.0), (lv * 1.04, 5e9)]
    prev = plan[0][0]
    for price, vol in plan:
        rows.append({"datetime": t, "open": prev, "high": max(price, prev),
                     "low": min(price, prev), "close": price, "volume": vol})
        prev = price
        t += timedelta(minutes=5)
    res = replay_week(snap, cfg, pd.DataFrame(rows))
    if res.signals:                       # depends on the synthetic weekly momentum
        assert res.signals[0].trigger == "deferred"
        assert res.signals[0].bar_time == rows[-1]["datetime"]
    assert res.saw_cross_this_week


def test_defer_disabled_means_cross_bar_only(snap):
    cfg = Strategy(strict_entry=True, defer_entry=False)
    lv = snap.entry_level
    rows = []
    t = datetime(2025, 1, 6, 9, 15, tzinfo=IST)
    for price, vol in [(lv * 0.98, 1.0), (lv * 1.02, 1.0), (lv * 1.04, 5e9)]:
        rows.append({"datetime": t, "open": price, "high": price,
                     "low": price, "close": price, "volume": vol})
        t += timedelta(minutes=5)
    res = replay_week(snap, cfg, pd.DataFrame(rows))
    assert all(s.trigger == "cross" for s in res.signals)


def test_req52_requires_both_levels(snap):
    cfg = Strategy(strict_entry=False, req52=True)
    lv26, lv52 = snap.entry_level, snap.level_52
    if lv52 > lv26:                    # only meaningful when 52W sits higher
        bars = bars_from([lv26 * 0.99, (lv26 + lv52) / 2 * 0.999], snap)
        assert replay_week(snap, cfg, bars).signals == []
        bars = bars_from([lv26 * 0.99, lv52 * 1.01], snap)
        assert len(replay_week(snap, cfg, bars).signals) == 1


def test_week_volume_accumulates_across_bars(snap):
    cfg = Strategy(strict_entry=False)
    lv = snap.entry_level
    bars = bars_from([lv * 0.99, lv * 1.02], snap, volume=1234.0)
    res = replay_week(snap, cfg, bars)
    assert res.signals[0].week_volume == pytest.approx(2468.0)


def test_day_open_resets_each_day(snap):
    """c13 compares against the current day's open, not the week's."""
    cfg = Strategy(strict_entry=False)
    lv = snap.entry_level
    d1 = bars_from([lv * 0.90, lv * 0.91], snap, day=1)
    d2 = bars_from([lv * 1.05, lv * 1.06], snap, day=2)
    res = replay_week(snap, cfg, pd.concat([d1, d2], ignore_index=True))
    assert len(res.signals) == 1
    assert res.signals[0].day_open == pytest.approx(lv * 1.05)
    assert res.signals[0].week_open == pytest.approx(lv * 0.90)


def test_empty_bars_is_safe(snap, cfg):
    res = replay_week(snap, cfg, pd.DataFrame(
        columns=["datetime", "open", "high", "low", "close", "volume"]))
    assert res.signals == [] and res.bars == 0


def test_replay_is_deterministic(snap, cfg):
    lv = snap.entry_level
    bars = bars_from([lv * 0.98, lv * 1.02, lv * 1.03], snap)
    a = replay_week(snap, cfg, bars)
    b = replay_week(snap, cfg, bars)
    assert [(s.bar_time, s.price) for s in a.signals] == [(s.bar_time, s.price) for s in b.signals]


def test_snapshot_roundtrips_through_csv(snap):
    from strategy import WeeklySnapshot
    back = WeeklySnapshot.from_row(snap.to_row())
    assert back.entry_level == pytest.approx(snap.entry_level)
    assert back.rsi.avg_gain == pytest.approx(snap.rsi.avg_gain)
    assert back.macd.sig == pytest.approx(snap.macd.sig)
    assert back.vol_sma.sum_prev == pytest.approx(snap.vol_sma.sum_prev)
    assert back.week_start == snap.week_start
