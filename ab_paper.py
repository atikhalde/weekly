#!/usr/bin/env python3
"""
A/B paper trading: run two competing models over the SAME live data.

    MODEL A "GATED"  - full weekly gate incl. c09, exit on 9-EMA momentum loss
    MODEL B "MOVER"  - raw cross + volume surge, run the move to +3R

Both are intraday-only (square-off 15:15, nothing overnight), both use the
SAME stop (entry candle low - 0.02%), the same universe, capital and costs.
The only differences are the entry filters and the exit rule - which is the
whole point of the experiment.

Nothing here places orders.

Usage
    python ab_paper.py --from-snapshot --days 5
    python ab_paper.py MONARCH TMB SENCO --days 5 --source yahoo
    python ab_paper.py --from-snapshot --days 1 --ledger ab_ledger.csv

The ledger is APPEND-ONLY and de-duplicated on (model, symbol, signal time),
so running it daily builds a week of evidence without double-counting.
"""

from __future__ import annotations

import argparse
import collections
import sys
from dataclasses import dataclass, field
from datetime import datetime, time as dtime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from config import load_config
from mcap import load_table as load_mcap_table
from paper import COST_ROUND_TRIP, SL_BUFFER, SQUARE_OFF, PaperTrade, _stamp, fetch_5m
from strategy import (IST_TZ as IST, Signal, build_snapshot, replay_week, week_start_of)

import indicators as ind

EMA_LEN = 9
ROOT = Path(__file__).resolve().parent
MCAP: dict[str, float] = {}


# --------------------------------------------------------------------------- #
#  Model definitions
# --------------------------------------------------------------------------- #
@dataclass
class Model:
    key: str
    label: str
    strategy: dict
    exit: dict
    horizon: str = "intraday"      # "intraday" | "swing"
    hold_days: int = 0             # swing only: sessions before the time exit

    @property
    def is_swing(self) -> bool:
        return self.horizon == "swing"


def load_models(path: Path | str | None = None) -> tuple[dict, list[Model]]:
    p = Path(path) if path else ROOT / "models.yaml"
    raw = yaml.safe_load(p.read_text())
    defaults = raw.get("defaults", {}) or {}
    models = []
    for k, v in (raw.get("models", {}) or {}).items():
        hz = str(v.get("horizon", "intraday")).lower()
        if hz not in ("intraday", "swing"):
            raise SystemExit(f"models.yaml: {k} has unknown horizon '{hz}'")
        models.append(Model(key=k, label=v.get("label", k),
                            strategy=v.get("strategy", {}) or {},
                            exit=v.get("exit", {}) or {},
                            horizon=hz,
                            hold_days=int(v.get("hold_days", 0) or 0)))
    return defaults, models


def apply_overrides(base_strategy, overrides: dict):
    """Return a copy of the Strategy dataclass with `overrides` applied."""
    import dataclasses
    valid = {f.name for f in dataclasses.fields(base_strategy)}
    unknown = set(overrides) - valid
    if unknown:
        raise SystemExit(f"models.yaml: unknown strategy keys {sorted(unknown)}")
    return dataclasses.replace(base_strategy, **overrides)


# --------------------------------------------------------------------------- #
#  Exit engine
# --------------------------------------------------------------------------- #
def simulate_model(sig, bars_after: pd.DataFrame, capital: float,
                   bars_before: pd.DataFrame | None,
                   exit_rule: dict, square_off: str = SQUARE_OFF,
                   cost_round_trip: float = COST_ROUND_TRIP) -> PaperTrade:
    """
    One trade, one model's exit rule.

    Shared with paper.py: entry at the signal candle close, stop at the entry
    candle low - SL_BUFFER, mandatory intraday square-off, costs on both legs.

    exit_rule:
        rule       "ema"    first 5m close below the 9-EMA   (model A)
                   "target" run to target_r x initial risk   (model B)
        be_at_r    move the stop to breakeven once +N x risk is seen (0 = off)
        target_r   exit at N x initial risk                  (0 = off)
        target_pct exit at a fixed percent gain              (0 = off)
    """
    signal_close = float(sig.price)
    sig_d, sig_t = _stamp(sig.bar_time)

    bar_low = float(getattr(sig, "bar_low", 0.0) or 0.0)
    if not (0.0 < bar_low < signal_close):
        bar_low = signal_close
    stop = bar_low * (1.0 - SL_BUFFER)

    t = PaperTrade(
        symbol=sig.symbol, week=sig.week_start,
        signal_date=sig_d, signal_time=sig_t, signal_close=signal_close,
        entry_date=sig_d, entry_time=sig_t, entry=signal_close,
        qty=0, invested=0.0, bar_low=bar_low, stop=stop,
        level_26w=float(sig.entry_level), level_52w=float(sig.level_52),
        trigger=sig.trigger,
        rsi=float(sig.evaluation.values.get("rsi", float("nan"))),
        macd_hist=float(sig.evaluation.values.get("macd_hist", float("nan"))),
    )
    if bars_after.empty:
        t.exit_reason = "NO_FILL"
        t.exit_note = "no candle after the signal"
        return t

    entry = signal_close
    qty = int(capital // entry) if entry > 0 else 0
    if qty <= 0:
        t.exit_reason = "NO_FILL"
        t.exit_note = f"price {entry:,.2f} above capital"
        return t
    t.qty = qty
    t.invested = qty * entry

    risk_per_share = max(entry - stop, entry * SL_BUFFER)
    t.r_multiple = 0.0

    rule = str(exit_rule.get("rule", "ema")).lower()
    be_at_r = float(exit_rule.get("be_at_r", 0.0) or 0.0)
    target_r = float(exit_rule.get("target_r", 0.0) or 0.0)
    target_pct = float(exit_rule.get("target_pct", 0.0) or 0.0)

    target = None
    if target_r > 0:
        target = entry + target_r * risk_per_share
    elif target_pct > 0:
        target = entry * (1.0 + target_pct / 100.0)

    # 9-EMA warmed on real pre-entry candles (never seeded at the entry price)
    alpha = 2.0 / (EMA_LEN + 1.0)
    if bars_before is not None and len(bars_before) >= EMA_LEN:
        hist = np.asarray(bars_before["close"].astype(float))[-(EMA_LEN * 10):]
        series = ind.ema(hist, EMA_LEN)
        ema_prev = float(series[-1])
        warmup = 0
    else:
        ema_prev = entry
        warmup = EMA_LEN

    hi = lo = entry
    peak = entry
    ret_reason = None

    for i, (_, bar) in enumerate(bars_after.iterrows(), start=1):
        o = float(bar["open"]); high = float(bar["high"])
        low = float(bar["low"]); close = float(bar["close"])
        bar_d, bar_t = _stamp(bar["datetime"])

        # --- INTRADAY: never hold past the session
        if bar_d != sig_d:
            t.exit = o
            t.exit_reason = "EOD"
            t.exit_note = "forced square-off: next session reached"
            t.bars_held = i
            t.exit_date, t.exit_time = bar_d, bar_t
            ret_reason = True
            break
        if bar_t >= square_off:
            t.exit = close
            t.exit_reason = "EOD"
            t.exit_note = f"intraday square-off at {square_off}"
            t.bars_held = i
            t.exit_date, t.exit_time = bar_d, bar_t
            ret_reason = True
            break

        hi, lo = max(hi, high), min(lo, low)

        # --- stop first: a resting SL-M triggers intrabar (conservative)
        if low <= stop:
            t.exit = o if o <= stop else stop
            if o <= stop:
                t.exit_note = "gap through stop"
            t.exit_reason = "SL" if stop < entry else "BE"
            t.bars_held = i
            t.exit_date, t.exit_time = bar_d, bar_t
            ret_reason = True
            break

        # --- target
        if target is not None and high >= target:
            t.exit = target
            t.exit_reason = "TGT"
            t.bars_held = i
            t.exit_date, t.exit_time = bar_d, bar_t
            ret_reason = True
            break

        peak = max(peak, high)

        # --- breakeven ratchet (model B): only ever raises the stop
        if be_at_r > 0 and (peak - entry) >= be_at_r * risk_per_share:
            stop = max(stop, entry * 1.0005)

        # --- momentum exit (model A)
        ema_now = alpha * close + (1.0 - alpha) * ema_prev
        if rule == "ema" and i > warmup and close < ema_now:
            t.exit = close
            t.exit_reason = "EMA9"
            t.bars_held = i
            t.exit_date, t.exit_time = bar_d, bar_t
            ret_reason = True
            break
        ema_prev = ema_now

    if not ret_reason:
        last = bars_after.iloc[-1]
        t.exit = float(last["close"])
        t.exit_reason = "OPEN"
        t.exit_note = "still running at end of data"
        t.bars_held = len(bars_after)
        t.exit_date, t.exit_time = _stamp(last["datetime"])

    t.gross_pnl = (t.exit - entry) * qty
    turnover = (entry + t.exit) * qty
    t.costs = turnover * (cost_round_trip / 100.0) / 2.0
    t.pnl = t.gross_pnl - t.costs
    t.pnl_pct = (t.exit / entry - 1.0) * 100.0 - cost_round_trip
    t.mfe_pct = (hi / entry - 1.0) * 100.0
    t.mae_pct = (lo / entry - 1.0) * 100.0
    t.r_multiple = (t.exit - entry) / risk_per_share
    return t


# --------------------------------------------------------------------------- #
#  Swing execution (Model C) - DAILY bars, multi-day hold
# --------------------------------------------------------------------------- #
def simulate_swing(sig, daily_after: pd.DataFrame, capital: float,
                   exit_rule: dict, hold_days: int = 5,
                   cost_round_trip: float = COST_ROUND_TRIP) -> PaperTrade:
    """
    A multi-day hold on DAILY candles.

    Differs from the intraday path in three ways, all deliberate:
      * the stop is a PERCENT of entry, not the entry-candle low. Measured on
        452 breakouts, the entry-candle low acts like a ~4% stop that fires on
        a third of trades; 7% fires on 7% and earns +5.59 vs +4.05 per trade.
      * there is no square-off. The position is meant to be held overnight.
      * the time exit is `hold_days` completed sessions, not a clock time.

    Conservative intrabar ordering: if a day's low breaches the stop AND its
    high tags the target, the STOP is taken. Real fills are not knowable from
    daily bars, so the pessimistic branch is the honest one.
    """
    entry = float(sig.price)
    sig_d, sig_t = _stamp(sig.bar_time)

    stop_pct = float(exit_rule.get("stop_pct", 7.0) or 7.0)
    be_at_r = float(exit_rule.get("be_at_r", 0.0) or 0.0)
    target_r = float(exit_rule.get("target_r", 0.0) or 0.0)
    hold = int(exit_rule.get("hold_days", hold_days) or hold_days)

    stop = entry * (1.0 - stop_pct / 100.0)
    risk = entry - stop

    eval_obj = getattr(sig, "evaluation", None)
    rsi_val = float(eval_obj.values.get("rsi", float("nan"))) if (eval_obj and hasattr(eval_obj, "values")) else float("nan")
    macd_val = float(eval_obj.values.get("macd_hist", float("nan"))) if (eval_obj and hasattr(eval_obj, "values")) else float("nan")

    t = PaperTrade(
        symbol=sig.symbol, week=sig.week_start,
        signal_date=sig_d, signal_time=sig_t, signal_close=entry,
        entry_date=sig_d, entry_time=sig_t, entry=entry,
        qty=0, invested=0.0, bar_low=float(getattr(sig, "bar_low", 0.0) or 0.0),
        stop=stop, level_26w=float(sig.entry_level),
        level_52w=float(sig.level_52), trigger=sig.trigger,
        rsi=rsi_val,
        macd_hist=macd_val,
    )
    if daily_after.empty or risk <= 0:
        t.exit_reason = "NO_FILL"
        t.exit_note = "no daily candle after the signal"
        return t

    qty = int(capital // entry) if entry > 0 else 0
    if qty <= 0:
        t.exit_reason = "NO_FILL"
        t.exit_note = f"price {entry:,.2f} above capital"
        return t
    t.qty = qty
    t.invested = qty * entry

    target = entry + target_r * risk if target_r > 0 else None
    hi = lo = entry
    peak = entry
    done = False

    for i, (_, bar) in enumerate(daily_after.head(hold).iterrows(), start=1):
        o = float(bar["open"]); high = float(bar["high"])
        low = float(bar["low"]); close = float(bar["close"])
        d, _tm = _stamp(bar["datetime"])
        hi, lo = max(hi, high), min(lo, low)

        if low <= stop:                       # stop wins ties
            t.exit = o if o <= stop else stop
            if o <= stop:
                t.exit_note = "gap through stop"
            t.exit_reason = "SL" if stop < entry else "BE"
            t.bars_held = i
            t.exit_date, t.exit_time = d, "close"
            done = True
            break
        if target is not None and high >= target:
            t.exit = target
            t.exit_reason = "TGT"
            t.bars_held = i
            t.exit_date, t.exit_time = d, "close"
            done = True
            break

        peak = max(peak, high)
        if be_at_r > 0 and (peak - entry) >= be_at_r * risk:
            stop = max(stop, entry * 1.0005)

    if not done:
        seg = daily_after.head(hold)
        last = seg.iloc[-1]
        t.exit = float(last["close"])
        t.exit_reason = "TIME"
        t.exit_note = f"{len(seg)}-day time exit"
        t.bars_held = len(seg)
        t.exit_date, t.exit_time = _stamp(last["datetime"])[0], "close"

    t.gross_pnl = (t.exit - entry) * qty
    turnover = (entry + t.exit) * qty
    t.costs = turnover * (cost_round_trip / 100.0) / 2.0
    t.pnl = t.gross_pnl - t.costs
    t.pnl_pct = (t.exit / entry - 1.0) * 100.0 - cost_round_trip
    t.mfe_pct = (hi / entry - 1.0) * 100.0
    t.mae_pct = (lo / entry - 1.0) * 100.0
    t.r_multiple = (t.exit - entry) / risk
    return t


def _atr_pct(daily: pd.DataFrame, length: int = 14) -> float:
    """ATR over `length` daily bars as a percent of the last close."""
    if daily is None or len(daily) < 2:
        return 0.0
    h = daily["high"].to_numpy(float); lo = daily["low"].to_numpy(float)
    c = daily["close"].to_numpy(float)
    pc = np.roll(c, 1); pc[0] = c[0]
    tr = np.maximum(h - lo, np.maximum(np.abs(h - pc), np.abs(lo - pc)))
    last = c[-1]
    return float(np.mean(tr[-length:]) / last * 100.0) if last > 0 else 0.0


def btst_tier_for(day_bar, prev_bars, atr_pct: float) -> str | None:
    """
    Which BTST tier (if any) the breakout DAY qualifies for.

    Delegates the thresholds to btst.py so the paper model and the nightly
    scanner cannot drift apart - a regression test asserts they agree.
    """
    import btst as _b

    o = float(day_bar["open"]); h = float(day_bar["high"])
    lo = float(day_bar["low"]); c = float(day_bar["close"])
    v = float(day_bar.get("volume", 0.0) or 0.0)
    if c <= 0 or o <= 0:
        return None
    rng = h - lo
    day_ret = (c / o - 1.0) * 100.0
    close_pos = ((c - lo) / rng) if rng > 0 else 0.5
    vma = float(prev_bars["volume"].tail(50).mean()) if len(prev_bars) else 0.0
    rvol = (v / vma) if vma > 0 else 0.0

    if day_ret >= _b.TIER_A_DAY and close_pos >= _b.TIER_A_CLOSE_POS:
        return "A"
    if (close_pos >= _b.TIER_B_CLOSE_POS and rvol >= _b.TIER_B_RVOL
            and atr_pct >= _b.TIER_B_ATR):
        return "B"
    return None


def _f(row, key, default=0.0):
    """
    Read a numeric CSV field safely. Missing / blank / NaN -> `default`.

    BUG 77. Three sites read picks-file columns as `float(row.get(k) or 0)`.
    That works for a MISSING key and for an empty string, but a blank cell
    read back by pandas is **NaN, which is truthy**, so `or 0` never fires.
    `float(nan)` is harmless; `int(float(nan))` RAISES.

    BUG 76 fixed the `tradeable` instance. This one - `age` - has the identical
    shape and crashed the A/B run again on the very next execution, this time
    at the END of the replay instead of 25 symbols in. Fixing one instance of a
    pattern and not grepping for its siblings is what made that a second
    outage rather than one.
    """
    v = row.get(key, default)
    if v is None:
        return default
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return default if f != f else f


def _is_tradeable(row) -> bool:
    """
    Was this pick actually enterable? Missing/blank -> YES.

    BUG 76. The original guard was `int(float(row.get("tradeable", 1) or 0))`.
    It was written for picks files predating BUG 55, whose rows have no
    `tradeable` column at all - and for a MISSING key the default 1 works.

    But rows written before the column existed and re-read from CSV come back
    with a BLANK cell, which pandas parses as NaN - and **NaN is truthy**, so
    `or 0` never fires, and `int(float(nan))` raises. That crashed the whole
    A/B paper run on 07-Aug:

        ValueError: cannot convert float NaN to integer
        ab_paper.py line 1067, in main

    Both scheduled runs that day died, so the paper ledger - the only forward
    evidence this project has - silently stopped recording.

    Unknown means "written before the flag existed", which means it was a
    normal live pick. Treat it as tradeable.
    """
    v = row.get("tradeable", 1)
    if v is None:
        return True
    try:
        f = float(v)
    except (TypeError, ValueError):
        return True                      # unparseable -> do not silently drop
    if f != f:                           # NaN
        return True
    return bool(int(f))


def load_btst_picks(path: Path | str | None = None) -> tuple[dict, dict]:
    """
    The top-5 list btst.py wrote at 15:20.

    Returns ({(date, symbol): row}, {date: True}). The second map matters: it
    lets the caller tell "this day has a picks file and the name is not in it"
    (so it was NOT taken) apart from "no file for this day at all" (so fall
    back to reconstruction). Without that distinction a missing file would
    silently trade everything.
    """
    import btst as _b

    p = Path(path) if path else (ROOT / _b.PICKS_FILE)
    if not p.exists():
        return {}, {}
    try:
        df = pd.read_csv(p)
    except Exception:
        return {}, {}
    lookup, days = {}, {}
    for r in df.to_dict("records"):
        d = str(r.get("date", "")).strip()
        sym = str(r.get("symbol", "")).strip()
        if not d or not sym:
            continue
        lookup[(d, sym)] = r
        days[d] = True
    return lookup, days


def simulate_btst(sig, daily_after: pd.DataFrame, capital: float,
                  exit_rule: dict, hold_days: int = 10,
                  cost_round_trip: float = COST_ROUND_TRIP) -> PaperTrade:
    """
    BTST simulation: buy the breakout/anticipation close at 15:20.

    Exit rules:
      * 'morning_open' (Recommended Playbook): Sell at next morning 09:15 open
        to capture overnight gap profits. Hold if opened locked at Upper Circuit;
        cut if opened gap-down <= -1.5%.
      * 'btst': Standard 1% stop / 2% target on close.
      * 'swing': Multi-day hold with stop and target.
    """
    entry = float(sig.price)
    sig_d, sig_t = _stamp(sig.bar_time)

    rule_name = str(exit_rule.get("rule", "btst")).lower()
    stop_pct = float(exit_rule.get("stop_pct", 2.0) or 2.0)
    take_pct = float(exit_rule.get("take_pct", 2.0) or 2.0)
    hold = int(exit_rule.get("hold_days", hold_days) or hold_days)

    stop = entry * (1.0 - stop_pct / 100.0)
    risk = entry - stop
    target = entry * (1.0 + take_pct / 100.0)

    eval_obj = getattr(sig, "evaluation", None)
    rsi_val = float(eval_obj.values.get("rsi", float("nan"))) if (eval_obj and hasattr(eval_obj, "values")) else float("nan")
    macd_val = float(eval_obj.values.get("macd_hist", float("nan"))) if (eval_obj and hasattr(eval_obj, "values")) else float("nan")

    t = PaperTrade(
        symbol=sig.symbol, week=sig.week_start,
        signal_date=sig_d, signal_time=sig_t, signal_close=entry,
        entry_date=sig_d, entry_time="close", entry=entry,
        qty=0, invested=0.0, bar_low=float(getattr(sig, "bar_low", 0.0) or 0.0),
        stop=stop, level_26w=float(sig.entry_level),
        level_52w=float(sig.level_52), trigger=sig.trigger,
        rsi=rsi_val,
        macd_hist=macd_val,
    )
    if daily_after.empty or risk <= 0:
        t.exit_reason = "NO_FILL"
        t.exit_note = "no daily candle after the signal"
        return t
    qty = int(capital // entry) if entry > 0 else 0
    if qty <= 0:
        t.exit_reason = "NO_FILL"
        t.exit_note = f"price {entry:,.2f} above capital"
        return t
    t.qty = qty
    t.invested = qty * entry

    # Morning Open Exit Rule (Optimized Playbook)
    if rule_name == "morning_open":
        bar1 = daily_after.iloc[0]
        o1 = float(bar1["open"]); h1 = float(bar1["high"]); l1 = float(bar1["low"]); c1 = float(bar1["close"])
        d1, _ = _stamp(bar1["datetime"])

        # 1. Gap-down cut (cut at open if opened <= -1.5%)
        if o1 <= entry * (1.0 - 0.015):
            t.exit = o1
            t.exit_reason = "GAP_DOWN_CUT"
            t.exit_note = f"opened gap-down {o1:,.2f} (cut at 09:15 open)"
            t.bars_held = 1
            t.exit_date, t.exit_time = d1, "open"
        # 2. Upper circuit rider (if opened locked at UC >= +4.5%, ride into D+2)
        elif o1 >= entry * 1.045 and (h1 - c1) / max(c1, 1) < 0.005 and len(daily_after) >= 2:
            bar2 = daily_after.iloc[1]
            c2 = float(bar2["close"])
            d2, _ = _stamp(bar2["datetime"])
            t.exit = c2
            t.exit_reason = "CIRCUIT_RIDER_D2"
            t.exit_note = f"rode upper circuit into D+2 close {c2:,.2f}"
            t.bars_held = 2
            t.exit_date, t.exit_time = d2, "close"
        # 3. 50/50 Asymmetric Hybrid: 50% sold at morning open, 50% multi-day runner
        else:
            part1_exit = o1
            # Leg 2: Runner carried if D+1 closed green above entry
            if c1 > entry and len(daily_after) >= 2:
                trail_stop = max(l1, entry * 1.003)  # breakeven floor
                bar2 = daily_after.iloc[1]
                l2 = float(bar2["low"]); c2 = float(bar2["close"])
                if l2 <= trail_stop:
                    part2_exit = trail_stop
                    bars_held = 2
                elif len(daily_after) >= 3 and c2 > c1:
                    bar3 = daily_after.iloc[2]
                    l3 = float(bar3["low"]); c3 = float(bar3["close"])
                    trail_stop2 = max(l2, trail_stop)
                    if l3 <= trail_stop2:
                        part2_exit = trail_stop2
                    else:
                        part2_exit = c3
                    bars_held = 3
                else:
                    part2_exit = c2
                    bars_held = 2
            else:
                part2_exit = c1
                bars_held = 1

            t.exit = 0.5 * part1_exit + 0.5 * part2_exit
            t.exit_reason = "50_OPEN_50_RUNNER"
            t.exit_note = f"50% @ open {o1:,.2f} + 50% runner @ {part2_exit:,.2f}"
            t.bars_held = bars_held
            t.exit_date, t.exit_time = d1, "open+runner"

        t.gross_pnl = (t.exit - entry) * qty
        turnover = (entry + t.exit) * qty
        t.costs = turnover * (cost_round_trip / 100.0) / 2.0
        t.pnl = t.gross_pnl - t.costs
        t.pnl_pct = (t.exit / entry - 1.0) * 100.0 - cost_round_trip
        return t

    hi = lo = entry
    done = False
    for i, (_, bar) in enumerate(daily_after.head(hold).iterrows(), start=1):
        o = float(bar["open"]); high = float(bar["high"])
        low = float(bar["low"]); close = float(bar["close"])
        d, _tm = _stamp(bar["datetime"])
        hi, lo = max(hi, high), min(lo, low)

        if low <= stop:                       # stop wins ties
            gapped = o <= stop
            t.exit = o if gapped else stop
            if gapped:
                t.exit_note = f"gapped through the stop, filled {o:,.2f}"
            t.exit_reason = "SL"
            t.bars_held = i
            t.exit_date, t.exit_time = d, "close"
            done = True
            break
        if close >= target:
            t.exit = close
            t.exit_reason = "TGT"
            t.exit_note = f"closed >= +{take_pct:g}%"
            t.bars_held = i
            t.exit_date, t.exit_time = d, "close"
            done = True
            break
        # otherwise: carry to the next session

    if not done:
        seg = daily_after.head(hold)
        last = seg.iloc[-1]
        t.exit = float(last["close"])
        t.exit_reason = "TIME"
        t.exit_note = f"{len(seg)}-day time exit"
        t.bars_held = len(seg)
        t.exit_date, t.exit_time = _stamp(last["datetime"])[0], "close"

    t.gross_pnl = (t.exit - entry) * qty
    turnover = (entry + t.exit) * qty
    t.costs = turnover * (cost_round_trip / 100.0) / 2.0
    t.pnl = t.gross_pnl - t.costs
    t.pnl_pct = (t.exit / entry - 1.0) * 100.0 - cost_round_trip
    t.mfe_pct = (hi / entry - 1.0) * 100.0
    t.mae_pct = (lo / entry - 1.0) * 100.0
    t.r_multiple = (t.exit - entry) / risk
    return t


# --------------------------------------------------------------------------- #
#  Ledger
# --------------------------------------------------------------------------- #
LEDGER_COLS = ["model", "model_label", "horizon", "symbol", "week", "signal_date",
               "signal_time", "signal_close", "entry", "qty", "invested",
               "bar_low", "stop", "level_26w", "trigger", "rsi", "macd_hist",
               "exit_date", "exit_time", "exit", "exit_reason", "exit_note",
               "bars_held", "gross_pnl", "costs", "pnl", "pnl_pct",
               "r_multiple", "mfe_pct", "mae_pct",
               # BTST (Model E) only; blank for every other model.
               # btst_source records whether the trade came from the 15:20
               # picks file ("picks") or was rebuilt after the close
               # ("reconstructed") - the two are NOT equally trustworthy.
               "btst_tier", "btst_day_ret", "btst_rank", "btst_source",
               # BUG 53: which arm the pick came from - fresh_A / fresh_B /
               # aged_B - and how old the breakout was. Recorded so the arms
               # can be scored SEPARATELY on forward data instead of blended.
               # Aged tier B is the newest and least forward-verified arm; if
               # it underperforms live this column is how that gets caught.
               "btst_arm", "btst_age"]


def append_ledger(path: Path, rows: list[dict]) -> tuple[int, int]:
    """Append, de-duplicating on (model, symbol, signal_date, signal_time)."""
    new = pd.DataFrame(rows)
    if new.empty:
        return 0, 0
    for c in LEDGER_COLS:
        if c not in new.columns:
            new[c] = ""
    new = new[LEDGER_COLS]
    key = ["model", "symbol", "signal_date", "signal_time"]

    if path.exists():
        old = pd.read_csv(path)
        for c in LEDGER_COLS:
            if c not in old.columns:
                old[c] = ""
        old = old[LEDGER_COLS]
        before = len(new)
        merged = pd.concat([old, new], ignore_index=True)
        merged = merged.drop_duplicates(subset=key, keep="first")
        added = len(merged) - len(old)
        merged.to_csv(path, index=False)
        return added, before - added
    new = new.drop_duplicates(subset=key, keep="first")
    new.to_csv(path, index=False)
    return len(new), 0


# --------------------------------------------------------------------------- #
#  Reporting
# --------------------------------------------------------------------------- #
def summarise(df: pd.DataFrame, initial_capital: float = 100000.0) -> dict:
    if df.empty:
        return dict(n=0)
    closed = df[df.exit_reason != "NO_FILL"].copy()
    if closed.empty:
        return dict(n=0)
    pnl = closed["pnl"].astype(float)
    pct = closed["pnl_pct"].astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    gw, gl = wins.sum(), abs(losses.sum())

    # Dynamic daily compounding (CAGR method) starting from initial capital
    df_sorted = closed.sort_values("signal_date").copy()
    days = sorted(df_sorted["signal_date"].astype(str).unique())
    equity = initial_capital
    curve = [equity]
    for d in days:
        sub = df_sorted[df_sorted["signal_date"].astype(str) == d]
        n_pos = max(len(sub), 1)
        cap_pos = equity / n_pos
        day_pnl = 0.0
        for _, r in sub.iterrows():
            trade_pct = float(r.get("pnl_pct", 0.0) or 0.0)
            day_pnl += cap_pos * (trade_pct / 100.0)
        equity += day_pnl
        curve.append(equity)

    tot_comp_ret = (equity - initial_capital) / initial_capital * 100.0
    if len(days) >= 2:
        d0 = pd.to_datetime(days[0])
        d1 = pd.to_datetime(days[-1])
        years = max((d1 - d0).days / 365.25, 0.05)
        cagr = (((equity / initial_capital) ** (1.0 / years)) - 1.0) * 100.0 if equity > 0 else -100.0
    else:
        years = 1.0 / 252.0
        cagr = tot_comp_ret

    arr = np.array(curve)
    peaks = np.maximum.accumulate(arr)
    dd = (arr - peaks) / peaks * 100.0 if len(peaks) else np.array([0.0])
    max_dd = float(dd.min())

    return dict(
        n=len(closed),
        win=100.0 * (pnl > 0).mean(),
        net=pnl.sum(),
        gross=closed["gross_pnl"].astype(float).sum(),
        costs=closed["costs"].astype(float).sum(),
        comp_equity=equity,
        comp_net=equity - initial_capital,
        comp_ret=tot_comp_ret,
        cagr=cagr,
        comp_max_dd=max_dd,
        avg_pct=pct.mean(),
        med_pct=pct.median(),
        pf=(gw / gl) if gl > 0 else float("inf"),
        avg_r=closed["r_multiple"].astype(float).mean(),
        best=pct.max(), worst=pct.min(),
        mfe=closed["mfe_pct"].astype(float).mean(),
        mae=closed["mae_pct"].astype(float).mean(),
        big=100.0 * (closed["mfe_pct"].astype(float) >= 5).mean(),
        held=100.0 * (closed["mae_pct"].astype(float) > -0.05).mean(),
        bars=closed["bars_held"].astype(float).mean(),
    )


def print_report(ledger: pd.DataFrame, models: list[Model]) -> None:
    print("\n" + "=" * 78)
    title = ("PAPER TRADING — HEAD TO HEAD" if len(models) > 1
             else "PAPER TRADING")
    print(title)
    print("=" * 78)

    if ledger.empty:
        print("\nNo trades recorded yet.")
        return

    days = sorted(ledger["signal_date"].astype(str).unique())
    print(f"Sessions: {len(days)}  ({days[0]} .. {days[-1]})")

    stats = {}
    for m in models:
        sub = ledger[ledger["model"] == m.key]
        stats[m.key] = summarise(sub)

    rows = [
        ("Trades", "n", "{:.0f}"),
        ("Win rate %", "win", "{:.1f}"),
        ("Compounded Equity Rs", "comp_equity", "{:,.0f}"),
        ("Compounded Net P&L Rs", "comp_net", "{:+,.0f}"),
        ("Compounded Return %", "comp_ret", "{:+,.2f}"),
        ("CAGR %", "cagr", "{:+,.2f}"),
        ("Compounded MaxDD %", "comp_max_dd", "{:.2f}"),
        ("Net P&L (Fixed 1L) Rs", "net", "{:,.0f}"),
        ("Gross P&L Rs", "gross", "{:,.0f}"),
        ("Costs Rs", "costs", "{:,.0f}"),
        ("Avg per trade %", "avg_pct", "{:+.2f}"),
        ("Median %", "med_pct", "{:+.2f}"),
        ("Profit factor", "pf", "{:.2f}"),
        ("Avg R", "avg_r", "{:+.2f}"),
        ("Best %", "best", "{:+.2f}"),
        ("Worst %", "worst", "{:+.2f}"),
        ("Avg MFE %", "mfe", "{:+.2f}"),
        ("Avg MAE %", "mae", "{:+.2f}"),
        ("Reached +5% MFE %", "big", "{:.0f}"),
        ("Never broke entry low %", "held", "{:.0f}"),
        ("Avg bars held", "bars", "{:.1f}"),
    ]
    keys = [m.key for m in models]
    print()
    print(f"{'metric':26}" + "".join(f"{k:>24}" for k in keys))
    print("-" * (26 + 24 * len(keys)))
    for label, key, fmt in rows:
        line = f"{label:26}"
        for k in keys:
            v = stats[k].get(key)
            line += f"{(fmt.format(v) if v is not None else '-'):>24}"
        print(line)

    print()
    for m in models:
        s = stats[m.key]
        if s.get("n"):
            print(f"  {m.key}: {m.label}")

    # verdict
    print("\n" + "-" * 78)
    live = [(k, stats[k]) for k in keys if stats[k].get("n")]
    if len(live) == 1:
        # Single model: there is nothing to rank, so report whether the
        # sample is big enough to mean anything yet.
        k, s = live[0]
        if s["n"] < 10:
            print(f"VERDICT: too early. {k} has {s['n']} closed trade(s); "
                  "wait for >= 10.")
        else:
            verdict = ("PROFITABLE so far" if s["avg_pct"] > 0
                       else "LOSING so far")
            print(f"VERDICT: {k} is {verdict} — {s['avg_pct']:+.2f}% per trade "
                  f"over {s['n']} trades (PF {s['pf']:.2f}).")
            if s["n"] < 30:
                print("         Indicative only until ~30 trades.")
    elif not live:
        print("VERDICT: no closed trades yet.")
    else:
        thin = [f"{k} n={s['n']}" for k, s in live if s["n"] < 10]
        ranked = sorted(live, key=lambda kv: -kv[1]["avg_pct"])
        board = " · ".join(f"{k} {s['avg_pct']:+.2f}%" for k, s in ranked)
        if thin:
            print(f"VERDICT: too early ({', '.join(thin)}). "
                  "Need >= 10 closed trades each.")
            print(f"         Standing: {board}")
        else:
            (ka, sa), (kb, sb) = ranked[0], ranked[1]
            print(f"VERDICT: {ka} leads by {sa['avg_pct'] - sb['avg_pct']:.2f}% "
                  f"per trade over {kb}.")
            print(f"         Standing: {board}")
            print("         Indicative until each side has ~30 trades. Note A/B "
                  "are intraday and C is a 5-day hold, so C locks capital "
                  "longer per trade.")
    print("-" * 78)


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def todays_trades_message(ledger: pd.DataFrame, models: list[Model],
                          day: str | None = None) -> str:
    """
    Per-symbol list of the positions each model took TODAY, so the morning
    watchlist can be reconciled against what actually happened.

    The standings message answers "is the model working"; this one answers
    "which stocks am I actually in". They are different questions and mixing
    them into one block made the second impossible to read.

    Positions still OPEN are listed separately from those already closed,
    because for a BTST or swing model an open position is an instruction for
    tomorrow, not a result.
    """
    from telegram import _esc, _fmt

    if ledger is None or ledger.empty:
        return ""
    led = ledger.copy()
    led["signal_date"] = led["signal_date"].astype(str)
    if day is None:
        day = max(led["signal_date"])
    today = led[led["signal_date"] == day]
    # NO_FILL is not a trade - it is a signal that could not be taken.
    taken = today[today["exit_reason"].astype(str) != "NO_FILL"]
    if taken.empty:
        return (f"📒 <b>Trades taken — {day}</b>\n\n"
                f"<i>No positions taken today.</i>")

    order = [m.key for m in models]
    lines = [f"📒 <b>Trades taken — {day}</b>",
             f"<i>{len(taken)} position(s) across "
             f"{taken['model'].nunique()} model(s)</i>", ""]

    for key in order:
        grp = taken[taken["model"] == key]
        if grp.empty:
            continue
        lines.append(f"<b>{key}</b>")
        for r in grp.sort_values("symbol").itertuples():
            sym = _esc(str(r.symbol))
            entry = float(getattr(r, "entry", 0) or 0)
            qty = int(getattr(r, "qty", 0) or 0)
            reason = str(getattr(r, "exit_reason", "") or "")
            # A position whose exit date is not today is still open.
            exit_date = str(getattr(r, "exit_date", "") or "")
            still_open = reason in ("", "nan", "OPEN") or (
                exit_date in ("", "nan", "None"))
            tier = str(getattr(r, "btst_tier", "") or "")
            tag = f" <i>[{tier}]</i>" if tier and tier != "nan" else ""

            if still_open:
                stop = float(getattr(r, "stop", 0) or 0)
                lines.append(
                    f"  🟡 <b>{sym}</b>{tag} · bought <b>{_fmt(entry)}</b> "
                    f"× {qty} · SL <code>{_fmt(stop)}</code> · <b>OPEN</b>")
            else:
                pnl = float(getattr(r, "pnl_pct", 0) or 0)
                exitp = float(getattr(r, "exit", 0) or 0)
                icon = "🟢" if pnl > 0 else ("🔴" if pnl < 0 else "⚪")
                held = int(getattr(r, "bars_held", 0) or 0)
                lines.append(
                    f"  {icon} <b>{sym}</b>{tag} · {_fmt(entry)} → "
                    f"{_fmt(exitp)} · <b>{pnl:+.2f}%</b> · {reason}"
                    + (f" · {held}d" if held > 1 else ""))
        lines.append("")

    # anything that signalled but could NOT be taken - the watchlist will show
    # these names, so say plainly why they are absent from the list above
    nofill = today[today["exit_reason"].astype(str) == "NO_FILL"]
    if not nofill.empty:
        names = ", ".join(sorted({_esc(str(x)) for x in nofill["symbol"]}))
        lines.append(f"<i>signalled but not taken ({nofill['symbol'].nunique()}): "
                     f"{names}</i>")
    lines.append("<i>Paper only. No orders are placed.</i>")
    return "\n".join(lines)


def telegram_summary(ledger: pd.DataFrame, models: list[Model]) -> str:
    """Compact HTML standings for Telegram."""
    if ledger.empty:
        return "📊 <b>Paper Trading</b>\n\nNo trades recorded yet."
    days = sorted(ledger["signal_date"].astype(str).unique())
    stats = {m.key: summarise(ledger[ledger["model"] == m.key]) for m in models}

    lines = ["📊 <b>Paper Trading</b>",
             f"<i>{len(days)} session(s): {days[0]} .. {days[-1]}</i>", ""]
    for m in models:
        s = stats[m.key]
        if not s.get("n"):
            lines.append(f"<b>{m.key}</b> — no trades yet")
            continue
        icon = "🟢" if s["net"] > 0 else ("🔴" if s["net"] < 0 else "⚪")
        lines += [
            f"{icon} <b>{m.key}</b> · {m.label.split('·')[-1].strip()}",
            f"   Equity <b>Rs {s.get('comp_equity', 100000):,.0f}</b> · CAGR <b>{s.get('cagr', 0):+.1f}%</b> · Net <b>Rs {s.get('comp_net', s['net']):+,.0f} ({s.get('comp_ret', 0):+.1f}%)</b>",
            f"   trades <b>{s['n']}</b> · win <b>{s['win']:.0f}%</b> · "
            f"PF <b>{s['pf']:.2f}</b> · MaxDD <b>{s.get('comp_max_dd', 0):.1f}%</b>",
            f"   avg trade <b>{s['avg_pct']:+.2f}%</b> "
            f"· avgR <b>{s['avg_r']:+.2f}</b> · +5% MFE <b>{s['big']:.0f}%</b>",
            "",
        ]
    live = [(m.key, stats[m.key]) for m in models if stats[m.key].get("n")]
    if len(live) == 1:
        k, s = live[0]
        if s["n"] < 10:
            lines.append(f"⏳ Too early — {k} has {s['n']} trade(s), need ≥10.")
        else:
            icon = "✅" if s["avg_pct"] > 0 else "❌"
            lines.append(f"{icon} <b>{k}</b> {s['avg_pct']:+.2f}%/trade "
                         f"over {s['n']} trades")
    elif len(live) >= 2:
        ranked = sorted(live, key=lambda kv: -kv[1]["avg_pct"])
        if min(s["n"] for _k, s in ranked) < 10:
            thin = ", ".join(f"{k} {s['n']}" for k, s in ranked)
            lines.append(f"⏳ Too early — need ≥10 trades each ({thin}).")
        else:
            (ka, sa), (_kb, sb) = ranked[0], ranked[1]
            gap = sa["avg_pct"] - sb["avg_pct"]
            lines.append(f"🏁 <b>{ka}</b> leads by {gap:.2f}%/trade")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("symbols", nargs="*")
    ap.add_argument("--telegram", action="store_true",
                    help="send the standings to Telegram")
    ap.add_argument("--from-snapshot", action="store_true",
                    help="use every symbol in weekly_snapshot.csv")
    ap.add_argument("--days", type=int, default=5,
                    help="how many sessions back to replay (default 5)")
    ap.add_argument("--capital", type=float, default=None)
    ap.add_argument("--source", choices=["dhan", "yahoo"], default="dhan")
    ap.add_argument("--ledger", default="ab_ledger.csv")
    ap.add_argument("--models", default=None, help="path to models.yaml")
    ap.add_argument("--config", default=None)
    ap.add_argument("--report-only", action="store_true",
                    help="print the standings from the existing ledger")
    args = ap.parse_args()

    defaults, models = load_models(args.models)

    # Market caps for c12. Without this every snapshot carries mcap=None; with
    # use_mcap on that is "unknown" (which passes), but loading the real table
    # means the paper models see exactly what the live scanner sees.
    global MCAP
    MCAP = load_mcap_table(load_config(args.config).paths["mcap"])
    ledger_path = Path(args.ledger)
    if not ledger_path.is_absolute():
        ledger_path = ROOT / ledger_path

    if args.report_only:
        if not ledger_path.exists():
            print(f"{ledger_path} not found")
            return 1
        led = pd.read_csv(ledger_path)
        print_report(led, models)
        if args.telegram:
            _send_telegram(todays_trades_message(led, models))
            _send_telegram(telegram_summary(led, models))
        return 0

    cfg = load_config(args.config)
    capital = args.capital or float(defaults.get("capital", 100000))
    square_off = str(defaults.get("square_off", SQUARE_OFF))
    cost = float(defaults.get("cost_round_trip", COST_ROUND_TRIP))
    include_deferred = bool(defaults.get("include_deferred", False))
    interval = cfg.runtime.bar_interval_min

    symbols = [s.upper() for s in args.symbols]
    if args.from_snapshot:
        p = cfg.paths["snapshot"]
        if not p.exists():
            print(f"{p} not found - run build_snapshot.py first")
            return 1
        symbols = sorted(pd.read_csv(p, dtype=str)["symbol"].str.upper().unique())
    if not symbols:
        print("give symbols, or --from-snapshot")
        return 2

    print("=" * 78)
    print("PAPER TRADING")
    print("=" * 78)
    for m in models:
        print(f"  {m.key:10} {m.label}")
    print(f"\n  capital Rs {capital:,.0f}/trade · square-off {square_off} · "
          f"costs {cost:.2f}% round trip")
    for m in models:
        if m.is_swing and str(m.exit.get("rule", "")).lower() == "btst":
            print(f"  {m.key:10} BTST · buy at close · "
                  f"{m.exit.get('stop_pct', 5.0):.1f}% stop · "
                  f"exit on a close >= +{m.exit.get('take_pct', 2.0):.1f}% · "
                  f"else carry (max {m.hold_days}d)")
        elif m.is_swing:
            print(f"  {m.key:10} SWING · {m.hold_days}d hold · "
                  f"{m.exit.get('stop_pct', 7.0):.1f}% stop (no square-off)")
        else:
            print(f"  {m.key:10} INTRADAY · square-off {square_off} · "
                  f"stop = entry candle low - {SL_BUFFER*100:.2f}%")
    print(f"  {len(symbols)} symbol(s), last {args.days} session(s)\n")

    # ---- the 15:20 picks file: Model E's trade list ------------------------
    picks_lookup, picks_have_day = load_btst_picks()
    ant_lookup, ant_have_day = load_btst_picks(ROOT / "anticipate_picks.csv")
    if ant_lookup:
        print(f"  anticipate picks: {len(ant_lookup)} entry(ies) across "
              f"{len(ant_have_day)} session(s) - Model F trades THAT list")
    if picks_lookup:
        print(f"  btst picks: {len(picks_lookup)} entry(ies) across "
              f"{len(picks_have_day)} session(s) - Model E will trade THAT list")
    elif any(str(m.exit.get("rule", "")).lower() == "btst" for m in models):
        print("  btst picks: none found - BTST models will RECONSTRUCT from "
              "completed candles (marked btst_source=reconstructed)")

    # ---- data source
    today = datetime.now(IST).date()
    start_day = today - timedelta(days=max(args.days * 2, args.days + 4))
    start_week = week_start_of(start_day)

    # Scope active symbols for the requested window to finish in seconds
    if args.from_snapshot and not getattr(args, "all", False):
        from state import AlertState
        target_days = {str(today - timedelta(days=i)) for i in range(max(args.days * 2, 7))}
        active_syms = {s.upper() for (d, s) in picks_lookup.keys() if d in target_days}
        active_syms |= {s.upper() for (d, s) in ant_lookup.keys() if d in target_days}
        st = AlertState(cfg.paths["state"])
        for w, m in getattr(st, "_data", {}).items():
            if isinstance(m, dict):
                for k in m.keys():
                    k_str = str(k).upper()
                    if k_str not in ("STALE", "") and len(k_str) >= 2:
                        active_syms.add(k_str)
        if active_syms:
            symbols = sorted([s for s in symbols if s in active_syms])
            print(f"  scoped to {len(symbols)} active candidate(s) across target sessions")

    if args.source == "yahoo":
        fetch = _yahoo_fetch
        sec_map = {s: (s, "NSE_EQ") for s in symbols}
        client = None
    else:
        from dhan import DhanClient, DhanError, last_n_years
        if not cfg.secrets.dhan_access_token:
            print("DHAN_ACCESS_TOKEN not set - running A/B paper with yfinance as primary source")
        client = DhanClient(cfg.secrets.dhan_client_id,
                            cfg.secrets.dhan_access_token,
                            data_rate=cfg.runtime.data_rate_per_sec,
                            quote_rate=cfg.runtime.quote_rate_per_sec)
        print("Loading instrument list ...")
        ins = DhanClient.fetch_instruments(cfg.universe.exchange_segments,
                                           cfg.universe.series,
                                           exclude_etf=cfg.universe.exclude_etf)
        sec_map = {i.symbol.upper(): (i.security_id, i.exchange_segment)
                   for i in ins}
        fetch = None

    rows: list[dict] = []
    per_model_counts = {m.key: 0 for m in models}

    for n, sym in enumerate(symbols, 1):
        if sym not in sec_map:
            continue
        sid, seg = sec_map[sym]

        try:
            if args.source == "yahoo":
                daily, five = _yahoo_fetch(sym)
            else:
                from dhan import DhanError, last_n_years
                from_date, to_date = last_n_years(cfg.runtime.history_years)
                daily = client.daily_candles(sid, seg, from_date, to_date, symbol=sym)
                five = fetch_5m(
                    client, sid, seg,
                    datetime.combine(start_week.date(), dtime(9, 0)).replace(tzinfo=IST),
                    datetime.now(IST), interval, symbol=sym)
        except Exception as exc:                       # noqa: BLE001
            print(f"  {sym}: fetch failed ({str(exc)[:60]})")
            continue
        if daily is None or daily.empty or five is None or five.empty:
            continue

        five = five.copy()
        fts = pd.to_datetime(five["datetime"])
        try:
            fts = fts.dt.tz_convert(IST)
        except (TypeError, AttributeError):
            pass
        five["_day"] = fts.dt.tz_localize(None).dt.normalize()

        d = daily.copy()
        d["_day"] = pd.to_datetime(d["datetime"]).dt.tz_localize(None).dt.normalize()

        for m in models:
            strat = apply_overrides(cfg.strategy, m.strategy)
            busy_until = None
            wk = start_week
            end_week = week_start_of(today)
            while wk <= end_week:
                hist = d[d["_day"] < wk]
                snap = (build_snapshot(sym, str(sid), seg, hist, strat, wk,
                                       mcap=MCAP.get(sym.upper()))
                        if not hist.empty else None)
                if snap is not None:
                    wbars = five[(five["_day"] >= wk) &
                                 (five["_day"] < wk + pd.Timedelta(days=7))]
                    if not wbars.empty:
                        res = replay_week(snap, strat, wbars, history=five)
                        for sig in res.signals:
                            sig_ts = pd.Timestamp(sig.bar_time)
                            if sig_ts.tzinfo is None:
                                sig_ts = sig_ts.tz_localize(IST)
                            if busy_until is not None and sig_ts <= busy_until:
                                continue
                            if not include_deferred and sig.trigger != "cross":
                                continue
                            if m.is_swing:
                                # Multi-day hold: step forward on DAILY bars
                                # from the session AFTER the breakout day.
                                sig_day = pd.Timestamp(sig.bar_time).normalize().tz_localize(None)
                                dafter = d[d["_day"] > sig_day]
                                if str(m.exit.get("rule", "")).lower() == "btst":
                                    day_row = d[d["_day"] == sig_day]
                                    if day_row.empty:
                                        continue
                                    day_key = str(sig_day.date())
                                    if m.strategy.get("anticipate_only"):
                                        # Model F trades the PRE-breakout list.
                                        # It never reconstructs: the whole
                                        # signal is "closed at the high while
                                        # still below the level", which cannot
                                        # be inferred from a breakout replay.
                                        ap_ = ant_lookup.get((day_key, sig.symbol))
                                        if ap_ is None:
                                            continue
                                        # BUG 63: an anticipation pick written
                                        # after the close was never enterable.
                                        # Same guard Model E already has.
                                        # Older files have no column -> 1.
                                        if not _is_tradeable(ap_):
                                            continue
                                        if (m.strategy.get("anticipate_mode") == "only"
                                                or m.key == "F_anticipate_only"):
                                            # Anticipated ONLY: skip if it qualified for confirmed BTST at 15:20
                                            if picks_lookup.get((day_key, sig.symbol)) is not None:
                                                continue
                                        sig.price = float(ap_["entry"])
                                        sig.btst_tier = "F_only" if (m.key == "F_anticipate_only") else "F"
                                        sig.btst_day_ret = _f(ap_, "day_ret", 0.0)
                                        sig.btst_rank = int(_f(ap_, "rank", 0.0))
                                        sig.btst_source = "anticipate"
                                        tr = simulate_btst(sig, dafter, capital,
                                                           m.exit, m.hold_days, cost)
                                        rec = dict(tr.__dict__)
                                        rec["model"] = m.key
                                        rec["model_label"] = m.label
                                        rec["horizon"] = m.horizon
                                        rec["btst_tier"] = sig.btst_tier
                                        rec["btst_day_ret"] = sig.btst_day_ret
                                        rec["btst_rank"] = sig.btst_rank
                                        rec["btst_source"] = "anticipate"
                                        rows.append(rec)
                                        per_model_counts[m.key] += 1
                                        continue
                                    # ---- PREFER THE 15:20 PICKS FILE -------
                                    # btst.py already decided, at 15:20, which
                                    # five names are taken and at what price.
                                    # Model E must trade THAT list at THAT
                                    # price - otherwise the alert and the
                                    # ledger describe different trades.
                                    pick = picks_lookup.get((day_key, sig.symbol))
                                    # BUG 55: a pick the 15:20 job produced
                                    # AFTER the close was never enterable.
                                    # Booking it would make the paper ledger
                                    # claim a fill that could not have
                                    # happened - exactly the drift the picks
                                    # file exists to prevent. Older files have
                                    # no such column; those all predate the
                                    # guard and are treated as tradeable.
                                    if pick is not None and not _is_tradeable(pick):
                                        continue
                                    if pick is not None:
                                        sig.price = float(pick["entry"])
                                        sig.btst_tier = str(pick.get("tier") or "")
                                        sig.btst_day_ret = _f(pick, "day_ret", 0.0)
                                        sig.btst_rank = int(_f(pick, "rank", 0.0))
                                        sig.btst_source = "picks"
                                        # BUG 53: carry the arm through from
                                        # the 15:20 file. Older picks files
                                        # have no such column - default to
                                        # fresh, which is what they all were.
                                        sig.btst_arm = str(pick.get("arm") or "")
                                        sig.btst_age = int(_f(pick, "age", 0.0))
                                    elif picks_have_day.get(day_key):
                                        # The picks file covers this day and
                                        # this name is not in it -> it was not
                                        # taken. Do NOT invent a trade.
                                        continue
                                    else:
                                        # No 15:20 file for this day (backfill,
                                        # or the job did not run). Reconstruct
                                        # from the completed candle and SAY SO.
                                        sig.price = float(day_row.iloc[-1]["close"])
                                        if m.strategy.get("btst_only"):
                                            prior = d[d["_day"] < sig_day]
                                            atrp = _atr_pct(prior)
                                            tier = btst_tier_for(day_row.iloc[-1],
                                                                 prior, atrp)
                                            if tier is None:
                                                continue
                                            sig.btst_tier = tier
                                            sig.btst_day_ret = (
                                                float(day_row.iloc[-1]["close"])
                                                / float(day_row.iloc[-1]["open"]) - 1) * 100
                                        sig.btst_rank = 0
                                        sig.btst_source = "reconstructed"
                                    tr = simulate_btst(sig, dafter, capital,
                                                       m.exit, m.hold_days, cost)
                                else:
                                    tr = simulate_swing(sig, dafter, capital,
                                                        m.exit, m.hold_days, cost)
                            else:
                                ts = pd.to_datetime(five["datetime"])
                                after = five[ts > pd.Timestamp(sig.bar_time)]
                                before = five[ts <= pd.Timestamp(sig.bar_time)]
                                tr = simulate_model(sig, after, capital, before,
                                                    m.exit, square_off, cost)
                            rec = dict(tr.__dict__)
                            rec["model"] = m.key
                            rec["model_label"] = m.label
                            rec["horizon"] = m.horizon
                            # carried through so the top-N cap can rank on it
                            rec["btst_tier"] = getattr(sig, "btst_tier", None)
                            rec["btst_day_ret"] = getattr(sig, "btst_day_ret", None)
                            rec["btst_rank"] = getattr(sig, "btst_rank", None)
                            rec["btst_source"] = getattr(sig, "btst_source", None)
                            rec["btst_arm"] = getattr(sig, "btst_arm", None)
                            rec["btst_age"] = getattr(sig, "btst_age", None)
                            rows.append(rec)
                            per_model_counts[m.key] += 1
                            if tr.exit_date:
                                # A swing trade blocks the symbol for DAYS, so
                                # the busy window is end-of-that-session.
                                # Swing trades block the symbol for DAYS; the
                                # intraday exit_time may be "close", which is
                                # not parseable, so normalise both here.
                                et = tr.exit_time
                                if m.is_swing or not et or ":" not in str(et):
                                    et = "15:30"
                                bu = pd.Timestamp(f"{tr.exit_date} {et}")
                                busy_until = (bu.tz_localize(IST)
                                              if bu.tzinfo is None else bu)
                wk += pd.Timedelta(days=7)

        # ---- DIRECT PICKS SIMULATION (MODELS E, E-WIDE, F, F-ONLY) ---------
        # Ensure 100% of the 15:20 alerted picks from btst_picks.csv and
        # anticipate_picks.csv are faithfully simulated regardless of 5m cross.
        import types
        for (day_k, p_sym), pick in picks_lookup.items():
            if p_sym.upper() != sym.upper() or not _is_tradeable(pick):
                continue
            try:
                p_entry = float(pick.get("entry") or 0.0)
            except (TypeError, ValueError):
                continue
            if p_entry <= 0:
                continue
            sig_day = pd.Timestamp(day_k).normalize().tz_localize(None)
            dafter = d[d["_day"] > sig_day]
            sig = types.SimpleNamespace(
                symbol=sym, bar_time=pd.Timestamp(f"{day_k} 15:20:00"),
                price=p_entry, trigger="cross",
                entry_level=float(pick.get("level", 0.0) or 0.0),
                level_52=float(pick.get("level_52", 0.0) or 0.0),
                week_start=str(week_start_of(pd.Timestamp(day_k).date()).date()),
                bar_low=0.0, evaluation=None,
                btst_tier=str(pick.get("tier") or ""),
                btst_day_ret=_f(pick, "day_ret", 0.0),
                btst_rank=int(_f(pick, "rank", 0.0)),
                btst_source="picks",
                btst_arm=str(pick.get("arm") or ""),
                btst_age=int(_f(pick, "age", 0.0)),
            )
            for m in [mm for mm in models if mm.key in ("E_btst", "E_btst_wide")]:
                tr = simulate_btst(sig, dafter, capital, m.exit, m.hold_days, cost)
                rec = dict(tr.__dict__)
                rec["model"] = m.key
                rec["model_label"] = m.label
                rec["horizon"] = m.horizon
                rec["btst_tier"] = sig.btst_tier
                rec["btst_day_ret"] = sig.btst_day_ret
                rec["btst_rank"] = sig.btst_rank
                rec["btst_source"] = "picks"
                rec["btst_arm"] = sig.btst_arm
                rec["btst_age"] = sig.btst_age
                rows.append(rec)
                per_model_counts[m.key] += 1

        for (day_k, a_sym), ap_ in ant_lookup.items():
            if a_sym.upper() != sym.upper() or not _is_tradeable(ap_):
                continue
            try:
                a_entry = float(ap_.get("entry") or 0.0)
            except (TypeError, ValueError):
                continue
            if a_entry <= 0:
                continue
            sig_day = pd.Timestamp(day_k).normalize().tz_localize(None)
            dafter = d[d["_day"] > sig_day]
            sig = types.SimpleNamespace(
                symbol=sym, bar_time=pd.Timestamp(f"{day_k} 15:20:00"),
                price=a_entry, trigger="anticipate",
                entry_level=float(ap_.get("level", 0.0) or 0.0),
                level_52=float(ap_.get("level_52", 0.0) or 0.0),
                week_start=str(week_start_of(pd.Timestamp(day_k).date()).date()),
                bar_low=0.0, evaluation=None,
                btst_day_ret=_f(ap_, "day_ret", 0.0),
                btst_rank=int(_f(ap_, "rank", 0.0)),
                btst_source="anticipate",
            )
            for m in [mm for mm in models if mm.key in ("F_anticipate", "F_anticipate_only")]:
                if m.key == "F_anticipate_only" and picks_lookup.get((day_k, sym)) is not None:
                    continue
                sig.btst_tier = "F_only" if (m.key == "F_anticipate_only") else "F"
                tr = simulate_btst(sig, dafter, capital, m.exit, m.hold_days, cost)
                rec = dict(tr.__dict__)
                rec["model"] = m.key
                rec["model_label"] = m.label
                rec["horizon"] = m.horizon
                rec["btst_tier"] = sig.btst_tier
                rec["btst_day_ret"] = sig.btst_day_ret
                rec["btst_rank"] = sig.btst_rank
                rec["btst_source"] = "anticipate"
                rows.append(rec)
                per_model_counts[m.key] += 1

        if len(symbols) > 25 and n % 25 == 0:
            print(f"  ...{n}/{len(symbols)}  {len(rows)} trades so far")

    # ---- BTST top-N cap ----------------------------------------------------
    # Model E takes only the best N setups per DAY, which is what a human with
    # finite capital would actually do. Ranking: Tier A before Tier B, then the
    # largest breakout-day move. Applied here rather than at signal time
    # because it needs the whole day's candidates, which only exist once every
    # symbol has been walked.
    capped = []
    by_model_day: dict[tuple, list] = {}
    for rec in rows:
        mk = rec.get("model")
        mdl = next((mm for mm in models if mm.key == mk), None)
        top_n = int((mdl.strategy.get("btst_top_n") or 0) if mdl else 0)
        if top_n <= 0:
            capped.append(rec)
            continue
        by_model_day.setdefault((mk, rec.get("signal_date")), []).append(rec)
    for (mk, day), group in by_model_day.items():
        mdl = next((mm for mm in models if mm.key == mk), None)
        top_n = int(mdl.strategy.get("btst_top_n") or 0)
        # When the 15:20 file decided the order, KEEP that order - re-ranking
        # here on post-close data could pick a different five than the alert
        # you actually received.
        def _rank(r):
            rk = r.get("btst_rank")
            try:
                rk = int(rk)
            except (TypeError, ValueError):
                rk = 0
            if rk > 0:
                return (0, rk, 0.0)
            return (1, 0, -_f(r, "btst_day_ret", 0.0)
                    + (0.0 if r.get("btst_tier") == "A" else 1e6))
        group.sort(key=_rank)
        kept = group[:top_n]
        capped.extend(kept)
        dropped = len(group) - len(kept)
        if dropped:
            print(f"  {mk}: {day} had {len(group)} candidates, kept top {len(kept)}")
    if len(capped) != len(rows):
        print(f"  BTST top-N cap: {len(rows)} -> {len(capped)} trade(s)")
    rows = capped
    per_model_counts = collections.Counter(r["model"] for r in rows)

    for k, v in per_model_counts.items():
        print(f"  {k}: {v} trade(s) this run")

    added, dupes = append_ledger(ledger_path, rows)
    print(f"\nLedger {ledger_path.name}: +{added} new, {dupes} already recorded")

    if ledger_path.exists():
        led = pd.read_csv(ledger_path)
        print_report(led, models)
        if args.telegram:
            # WHAT was taken first, then HOW the models are doing. Two
            # messages, because they answer different questions.
            _send_telegram(todays_trades_message(led, models))
            _send_telegram(telegram_summary(led, models))
    return 0


def _send_telegram(msg: str) -> None:
    try:
        from telegram import build_telegram
        from config import load_config as _lc
        c = _lc(None)
        tg = build_telegram(c)
        tg.send(msg)
        print("\nTelegram: sent")
    except Exception as exc:                            # noqa: BLE001
        print(f"\nTelegram: failed ({str(exc)[:80]})")


def _yahoo_fetch(sym: str):
    """Offline/dry-run data source so the A/B can be tested without Dhan."""
    import json
    import urllib.request

    def grab(rng, iv):
        u = (f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}.NS"
             f"?range={rng}&interval={iv}")
        req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
        d = json.load(urllib.request.urlopen(req, timeout=30))
        res = d["chart"]["result"][0]
        ts = res["timestamp"]; q = res["indicators"]["quote"][0]
        out = []
        for i, t in enumerate(ts):
            if q["close"][i] is None:
                continue
            out.append({"datetime": datetime.fromtimestamp(t, IST).replace(tzinfo=None),
                        "open": q["open"][i], "high": q["high"][i],
                        "low": q["low"][i], "close": q["close"][i],
                        "volume": q["volume"][i] or 0})
        return pd.DataFrame(out)

    return grab("2y", "1d"), grab("60d", "5m")


if __name__ == "__main__":
    sys.exit(main())
