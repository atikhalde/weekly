"""
End-to-end: build a snapshot from synthetic daily history, then replay a week of
5-minute candles containing one engineered breakout, and assert the alert lands
on exactly the candle the Pine indicator would mark.
"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from config import Strategy
from state import AlertState
from strategy import build_snapshot, replay_week, week_start_of
from telegram import Telegram, format_signal

IST = ZoneInfo("Asia/Kolkata")


def strong_uptrend_daily(weeks=160, seed=11,
                         peak_week=138, dip_end_week=150,
                         dip_drift=-0.006, recovery_drift=0.005):
    """
    A realistic setup rather than a straight line: a long rally, a
    consolidation that carves out the 26W high ABOVE current price, then a
    re-acceleration. That shape is what makes the scan meaningful --

      * the consolidation leaves close[1] <= 26W-high-of-2-weeks-ago  -> c03
      * the recovery turns the MACD histogram positive and lifts RSI  -> c06/c08
      * price is still just under the 26W high, so the breakout is genuinely
        ahead of us and the entry candle is unambiguous.
    """
    rng = np.random.default_rng(seed)
    rows, day, price = [], datetime(2022, 1, 3, 15, 30, tzinfo=IST), 200.0
    total = weeks * 5
    while len(rows) < total:
        if day.weekday() < 5:
            week = len(rows) // 5
            if week < peak_week:
                drift = 0.0035
            elif week < dip_end_week:
                drift = dip_drift
            else:
                drift = recovery_drift
            o = price
            c = price * (1 + drift + rng.normal(0, 0.0025))
            rows.append({"datetime": day, "open": o, "high": max(o, c) * 1.003,
                         "low": min(o, c) * 0.997, "close": c,
                         "volume": float(rng.integers(500_000, 900_000))})
            price = c
        day += timedelta(days=1)
    return pd.DataFrame(rows)


@pytest.fixture
def setup():
    daily = strong_uptrend_daily()
    cfg = Strategy(strict_entry=True, gate_source="live")
    last_week = week_start_of(pd.Timestamp(daily.iloc[-1]["datetime"]).date())
    target = last_week + pd.Timedelta(days=7)
    snap = build_snapshot("BREAKOUT", "9999", "NSE_EQ", daily, cfg, target)
    assert snap is not None
    return snap, cfg


def week_of_bars(snap, breakout_index, n=60, big_volume=True):
    """
    A week of 5m candles that climbs to just under the 26W level, then closes
    above it on bar `breakout_index`.
    """
    lv = snap.entry_level
    vol = (snap.vol_sma.sum_prev / max(snap.vol_sma.length - 1, 1)) * 2.0 if big_volume else 1.0
    per_bar = vol / max(breakout_index, 1)

    rows, t = [], datetime(2025, 3, 3, 9, 15, tzinfo=IST)
    week_open = lv * 0.965
    prev = week_open
    for i in range(n):
        if i < breakout_index:
            close = week_open + (lv * 0.995 - week_open) * (i / max(breakout_index - 1, 1))
        else:
            close = lv * (1.004 + 0.0008 * (i - breakout_index))
        rows.append({"datetime": t, "open": prev, "high": max(prev, close) * 1.001,
                     "low": min(prev, close) * 0.999, "close": close, "volume": per_bar})
        prev = close
        t += timedelta(minutes=5)
    return pd.DataFrame(rows)


def test_alert_lands_on_the_exact_breakout_candle(setup):
    snap, cfg = setup
    idx = 30
    bars = week_of_bars(snap, breakout_index=idx)
    res = replay_week(snap, cfg, bars)

    assert res.saw_cross_this_week
    assert len(res.signals) == 1, "one_per_week must collapse this to a single alert"

    sig = res.signals[0]
    expected = bars.iloc[idx]
    assert sig.bar_time == expected["datetime"], "signal must sit on the crossing candle"
    assert sig.price == pytest.approx(expected["close"])
    assert sig.price > sig.entry_level
    # the bar before must still be below the level
    assert bars.iloc[idx - 1]["close"] <= snap.entry_level


def test_all_thirteen_conditions_pass_at_the_signal(setup):
    snap, cfg = setup
    res = replay_week(snap, cfg, week_of_bars(snap, breakout_index=25))
    assert res.signals, "engineered week should produce a signal"
    ev = res.signals[0].evaluation
    assert ev.all_ok, f"failing: {ev.failed}"
    assert ev.pass_count == 13


def test_no_alert_when_price_never_clears_the_level(setup):
    snap, cfg = setup
    lv = snap.entry_level
    rows, t, prev = [], datetime(2025, 3, 3, 9, 15, tzinfo=IST), lv * 0.95
    for i in range(40):
        close = lv * (0.95 + 0.001 * i)          # tops out just below
        close = min(close, lv * 0.999)
        rows.append({"datetime": t, "open": prev, "high": close, "low": prev * 0.999,
                     "close": close, "volume": 1e6})
        prev = close
        t += timedelta(minutes=5)
    assert replay_week(snap, cfg, pd.DataFrame(rows)).signals == []


def test_low_volume_week_is_blocked_by_the_gate(setup):
    snap, cfg = setup
    bars = week_of_bars(snap, breakout_index=20, big_volume=False)
    res = replay_week(snap, cfg, bars)
    assert res.saw_cross_this_week, "the cross still happens"
    assert res.signals == [], "but c09 keeps the gate shut all week"


def test_strict_entry_off_fires_regardless_of_volume(setup):
    snap, _ = setup
    loose = Strategy(strict_entry=False)
    bars = week_of_bars(snap, breakout_index=20, big_volume=False)
    assert len(replay_week(snap, loose, bars).signals) == 1


def test_signal_survives_the_full_alert_pipeline(setup, tmp_path):
    """replay -> dedupe state -> Telegram formatting, the way scan.py runs it."""
    snap, cfg = setup
    res = replay_week(snap, cfg, week_of_bars(snap, breakout_index=18))
    assert res.signals
    sig = res.signals[0]

    state = AlertState(tmp_path / "state.json")
    assert not state.already_alerted(snap.week_start, sig.symbol)

    tg = Telegram("", "", dry_run=True)
    assert tg.send_signal(sig)
    state.mark(snap.week_start, sig.symbol, sig.bar_time, sig.price)
    state.save()

    # a second scan in the same week must stay silent
    assert AlertState(tmp_path / "state.json").already_alerted(snap.week_start, sig.symbol)

    msg = format_signal(sig)
    assert "BREAKOUT" in msg
    assert "13/13 PASS" in msg          # all-pass header
    assert "⚠️" not in msg               # no partial-pass warning


def test_rescanning_the_same_week_is_idempotent(setup):
    """Every cron run replays the week from bar 1 - the answer must not drift."""
    snap, cfg = setup
    bars = week_of_bars(snap, breakout_index=22)
    full = replay_week(snap, cfg, bars)
    # simulate an earlier cron run that only saw the first 30 bars
    partial = replay_week(snap, cfg, bars.iloc[:30].reset_index(drop=True))
    assert full.signals[0].bar_time == partial.signals[0].bar_time
    assert full.signals[0].price == pytest.approx(partial.signals[0].price)


def test_partial_week_before_breakout_is_silent(setup):
    snap, cfg = setup
    bars = week_of_bars(snap, breakout_index=22)
    early = replay_week(snap, cfg, bars.iloc[:20].reset_index(drop=True))
    assert early.signals == []
