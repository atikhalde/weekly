#!/usr/bin/env python3
"""
BTST & ANTICIPATION BACKTEST - replay LIVE tier and approach logic over history.

WHY THIS EXISTS
The alert says "Tier A measured +3.04%/trade (t 6.4, n=302)". Those numbers came
from throwaway analysis scripts in /tmp that no longer exist and were never in
the repo. There has been no way to ask "show me the actual trades behind that",
which is exactly the question worth asking before trusting a number.

This reproduces those figures from the SHIPPED CODE and prints the trade list.
It also supports replaying the ANTICIPATION model (Model F) - including setups
that appeared as anticipated but did NOT qualify for confirmed BTST at 15:20.

-----------------------------------------------------------------------------
THE ONE RULE THAT MAKES THIS TRUSTWORTHY
-----------------------------------------------------------------------------
It imports btst.classify(), btst.classify_approach(), btst.exhausted(),
btst.conviction() and every threshold from btst.py. It does NOT reimplement them.

A backtest that restates the rule measures a DIFFERENT rule the moment either
copy changes, and it always drifts in the flattering direction because nobody
re-checks a green number. Here, changing TIER_B_CLOSE_POS in btst.py changes
this backtest on the next run, automatically. If they ever disagree, this file
is wrong by construction.

-----------------------------------------------------------------------------
POINT-IN-TIME DISCIPLINE
-----------------------------------------------------------------------------
For each historical day D:
  * the 26W level uses weekly highs STRICTLY BEFORE the week containing D,
    exactly as the Monday snapshot freezes it
  * classify() and classify_approach() see daily bars up to and including D, never beyond
  * the outcome (D+1 close) is read only AFTER the decision is made
  * age comes from breakout_age() on the same truncated frame

There is no survivorship filter beyond "the symbol has data", and losers are
kept - the whole point is to see them.

-----------------------------------------------------------------------------
WHAT IT CANNOT TELL YOU
-----------------------------------------------------------------------------
* It uses the DAILY close as the entry. The live scan buys at ~15:20 on a
  PARTIAL candle. Measured, that entry is ~0.14% CHEAPER than the close, so
  this is mildly pessimistic - but it is not the same fill.
* Yahoo daily data, not Dhan. Volumes can differ per symbol (see the SONACOMS
  rvol discrepancy), so rvol-gated Tier B counts may differ slightly from live.
* No slippage beyond the flat cost. Real fills on a 3%+ ATR stock at the close
  are worse than the print.
* One regime. Everything here is 2021-2026, a broadly rising smallcap tape.

    python btst_backtest.py                                  # BTST confirmed, 2y summary
    python btst_backtest.py --mode anticipated_only --trades # anticipated non-BTST trades
    python btst_backtest.py --mode anticipated               # all anticipated setups
    python btst_backtest.py --mode all                       # confirmed BTST + anticipated
    python btst_backtest.py --years 5 --trades               # every trade, printed
    python btst_backtest.py --symbol SBCL                    # one stock's history
    python btst_backtest.py --from 2026-08-01                # a specific window
    python btst_backtest.py --csv out.csv                    # full trade list to CSV
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import btst                                    # noqa: E402  the LIVE rule
from strategy import build_weekly_bars         # noqa: E402

DATA_DIR = os.environ.get("BTST_BACKTEST_DATA", "/tmp/daily")
COST_ROUND_TRIP = 0.22          # matches models.yaml defaults
DEFAULT_CAPITAL = 100000.0      # Rs 1,00,000 per trade (single trade max 1L without CAGR)
MAX_DAILY_TRADES = 3            # max 3 trades taken per day
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0 Safari/537.36"}


# --------------------------------------------------------------------------- #
#  DATA
# --------------------------------------------------------------------------- #
def fetch_history(symbols: list[str], out_dir: str, years: int = 6,
                  workers: int = 12) -> int:
    """Download daily bars once, cache to CSV. Skips what is already there."""
    import threading

    import requests

    os.makedirs(out_dir, exist_ok=True)
    loc = threading.local()

    def sess():
        if not hasattr(loc, "s"):
            s = requests.Session()
            s.headers.update(UA)
            loc.s = s
        return loc.s

    def one(sym: str) -> str:
        p = os.path.join(out_dir, f"{sym}.csv")
        if os.path.exists(p):
            return "cached"
        for attempt in range(3):
            try:
                r = sess().get(
                    f"https://query2.finance.yahoo.com/v8/finance/chart/{sym}.NS",
                    params={"range": f"{years}y", "interval": "1d"}, timeout=25)
                if r.status_code == 429:
                    time.sleep(2 + attempt * 3)
                    continue
                res = r.json()["chart"]["result"]
                if not res:
                    return "empty"
                res = res[0]
                ts = res.get("timestamp")
                if not ts:
                    return "empty"
                q = res["indicators"]["quote"][0]
                df = pd.DataFrame({
                    "datetime": pd.to_datetime(ts, unit="s", utc=True)
                                  .tz_convert("Asia/Kolkata").tz_localize(None)
                                  .normalize(),
                    "open": q["open"], "high": q["high"], "low": q["low"],
                    "close": q["close"], "volume": q["volume"]}).dropna()
                min_bars = min(max(int(years * 100), 120), 250)
                if len(df) < min_bars:
                    return "short"
                df.to_csv(p, index=False)
                return "ok"
            except Exception:
                time.sleep(1 + attempt)
        return "fail"

    from concurrent.futures import ThreadPoolExecutor
    got = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(one, symbols)):
            if r in ("ok", "cached"):
                got += 1
            if i and i % 400 == 0:
                print(f"    {i}/{len(symbols)} ...", flush=True)
    return got


# --------------------------------------------------------------------------- #
#  CIRCUIT-LOCK & CIRCUIT-OPEN WINDOW TRACKING
# --------------------------------------------------------------------------- #
def get_circuit_tracking(d: pd.DataFrame, idx: int) -> dict:
    """
    Track daily upper circuit (UC) band (5%, 10%, 20%), distance to circuit,
    lock status, circuit-open windows, and pre-circuit entry price.

    In Indian markets (NSE/BSE):
      * 75% of circuit-touching stocks OPEN their circuit during the session
        (profit-taking / volume waves), providing active fillable windows!
      * Hard locks (flat all day, 0 range) are unfillable.
    """
    if idx <= 0 or idx >= len(d):
        return {"band": 10.0, "uc_limit": 0.0, "dist_to_uc": 0.0, "is_locked": False,
                "circuit_opened": True, "lock_reason": "fillable", "pre_entry": 0.0}
    row = d.iloc[idx]
    prev = d.iloc[idx - 1]
    c = float(row["close"])
    h = float(row["high"])
    l = float(row["low"])
    o = float(row["open"])
    pc = float(prev["close"])
    if pc <= 0 or c <= 0:
        return {"band": 10.0, "uc_limit": 0.0, "dist_to_uc": 0.0, "is_locked": False,
                "circuit_opened": True, "lock_reason": "fillable", "pre_entry": 0.0}
    day_gain = (c / pc - 1.0) * 100.0
    rng = h - l
    rng_pct = (rng / c) * 100.0 if c > 0 else 0.0
    cp = ((c - l) / rng) if rng > 0 else 1.0

    if day_gain >= 14.0:
        band = 20.0
    elif day_gain >= 7.5:
        band = 10.0
    elif day_gain >= 3.5:
        band = 5.0
    else:
        band = 2.0

    uc_limit = round(pc * (1.0 + band / 100.0), 2)
    dist_to_uc = round((uc_limit - c) / c * 100.0, 2)

    is_circuit_day = False
    for b in (2.0, 5.0, 10.0, 20.0):
        if abs(day_gain - b) <= 0.25 and (h - c) / c < 0.005:
            is_circuit_day = True
            break

    # Hard flat lock: open == high == low == close (0 range, locked from 09:15 open)
    is_hard_locked = is_circuit_day and (rng_pct < 0.3)
    # Circuit opened during session: stock hit circuit but traded with real range/volume
    circuit_opened = is_circuit_day and (rng_pct >= 0.3)
    is_locked_at_close = is_circuit_day and (cp >= 0.99)

    if is_hard_locked:
        lock_reason = f"hard_flat_lock_{int(band)}%"
        fillable_status = "unfillable"
    elif is_locked_at_close:
        lock_reason = f"circuit_opened_{int(band)}%" if circuit_opened else f"uc_locked_{int(band)}%"
        fillable_status = "fillable_on_open_window" if circuit_opened else "pre_circuit_only"
    else:
        lock_reason = "fillable"
        fillable_status = "fillable_open"

    # Pre-circuit / open-window entry price: ~0.8% below the upper circuit limit
    pre_entry = round(uc_limit * 0.992, 2) if is_circuit_day else c

    return {
        "band": band,
        "uc_limit": uc_limit,
        "dist_to_uc": dist_to_uc,
        "is_circuit_day": is_circuit_day,
        "is_hard_locked": is_hard_locked,
        "circuit_opened": circuit_opened,
        "is_locked": is_locked_at_close,
        "lock_reason": lock_reason,
        "fillable_status": fillable_status,
        "pre_entry": pre_entry,
    }


# --------------------------------------------------------------------------- #
#  REPLAY - one symbol
# --------------------------------------------------------------------------- #
def replay_symbol(path: str, start: str, end: str, mode: str = "btst",
                  exclude_circuit_locks: bool = False,
                  use_pre_circuit_entry: bool = False) -> list[dict]:
    """
    Replay historical days and identify qualifying setups with their outcomes.

    Modes:
      * 'btst': confirmed BTST breakouts (Tier A / Tier B).
      * 'anticipated': all pre-breakout anticipation setups (Model F).
      * 'anticipated_only': anticipation setups that did NOT qualify for BTST at 15:20.
      * 'all': both confirmed BTST and anticipated setups.
    """
    sym = os.path.basename(path)[:-4]
    try:
        d = pd.read_csv(path, parse_dates=["datetime"])
    except Exception:
        return []
    d = d[(d.volume > 0) & d.close.notna()].sort_values("datetime").reset_index(drop=True)
    if len(d) < 320:
        return []

    dt = pd.to_datetime(d.datetime)
    ws = dt.dt.normalize() - pd.to_timedelta(dt.dt.weekday, unit="D")
    wk = build_weekly_bars(d)
    if len(wk) < 30:
        return []

    wk_hi = wk.high.to_numpy(float)
    wk_start = pd.to_datetime(wk.week_start)
    widx = {k: i for i, k in enumerate(wk_start)}
    wi = ws.map(widx).to_numpy()

    # 26W high as of the LAST CLOSED week, per the live snapshot
    n_wk = len(wk_hi)
    lvl_by_week = np.full(n_wk, np.nan)
    lvl_d_by_week = np.full(n_wk, np.nan)
    for i in range(26, n_wk):
        lvl_by_week[i] = wk_hi[i - 26:i].max()
        if i >= 27:
            lvl_d_by_week[i] = wk_hi[i - 27:i - 1].max()

    close = d.close.to_numpy(float)
    lo_i = int(np.searchsorted(dt.values, np.datetime64(start)))
    hi_i = int(np.searchsorted(dt.values, np.datetime64(end), side="right"))
    lo_i = max(lo_i, 300)
    hi_i = min(hi_i, len(d) - 1)          # need D+1 for the outcome

    out = []
    want_btst = mode in ("btst", "all")
    want_ant = mode in ("anticipated", "anticipate", "anticipated_only", "anticipated_non_btst", "all")
    ant_only = mode in ("anticipated_only", "anticipated_non_btst")

    for j in range(lo_i, hi_i):
        w = wi[j]
        if w is None or (isinstance(w, float) and np.isnan(w)):
            continue
        level_c = lvl_by_week[int(w)]
        if not np.isfinite(level_c) or level_c <= 0:
            continue
        level_d = lvl_d_by_week[int(w)] if np.isfinite(lvl_d_by_week[int(w)]) else level_c

        hist = d.iloc[:j + 1]                       # bars up to and incl. D
        age = btst.breakout_age(hist, float(level_c))

        m_btst = btst.classify(hist, float(level_c), partial_frac=1.0, age=age)
        is_btst = bool(m_btst and m_btst.get("tier"))

        m_ant = None
        if want_ant:
            # Fast O(1) screen: only evaluate full metrics for days in the anticipation band
            rng = float(d.iloc[j]["high"]) - float(d.iloc[j]["low"])
            cp_quick = ((close[j] - float(d.iloc[j]["low"])) / rng) if rng > 0 else 0.0
            if cp_quick >= (btst.ANTICIPATE_CLOSE_POS - 0.02):
                cand = [level_c, level_d]
                above = [x for x in cand if close[j] <= x]
                in_band = False
                if above:
                    lvl = min(above)
                    in_band = ((lvl - close[j]) / lvl * 100.0 <= btst.ANTICIPATE_NEAR + 0.1)
                else:
                    lvl = max(cand)
                    in_band = ((close[j] / lvl - 1) * 100.0 <= btst.ANTICIPATE_ABOVE_MAX + 0.1)
                if in_band:
                    m_ant = btst.classify_approach(hist, float(level_c), float(level_d),
                                                  partial_frac=1.0, dbg_symbol=sym)
        is_ant = bool(m_ant and m_ant.get("ok"))

        # Circuit tracking: upper circuit level, band, distance, and pre-circuit entry
        ckt = get_circuit_tracking(d, j)
        is_locked = ckt["is_locked"]
        if exclude_circuit_locks and is_locked:
            continue

        # Effective entry price: pre-circuit entry if enabled and locked, else regular close
        entry_eff = ckt["pre_entry"] if (use_pre_circuit_entry and is_locked) else close[j]
        if entry_eff <= 0:
            continue

        # 50/50 Asymmetric Model Exit Execution (Matching live paper playbook)
        bar1 = d.iloc[j + 1]
        o1 = float(bar1["open"])
        h1 = float(bar1["high"])
        l1 = float(bar1["low"])
        c1 = close[j + 1]
        d1_o_ret = (o1 / entry_eff - 1.0) * 100.0

        # 1. Gap-down cut (<= -1.5% at open): 100% exit at 09:15 morning open
        if d1_o_ret <= -1.5:
            nxt = o1
            net_pct = round(d1_o_ret - COST_ROUND_TRIP, 3)
            exit_reason = "09:15 Gap-Down Cut (≤ -1.5%)"
        # 2. Upper Circuit Rider (if opened locked at UC >= +4.5%): ride into D+2
        elif d1_o_ret >= 4.5 and (h1 - c1) / max(c1, 1) < 0.005 and j + 2 < len(d):
            bar2 = d.iloc[j + 2]
            nxt = float(bar2["close"])
            c2_ret = (nxt / entry_eff - 1.0) * 100.0
            net_pct = round(c2_ret - COST_ROUND_TRIP, 3)
            exit_reason = "Rode Upper Circuit to D+2"
        # 3. 50/50 Asymmetric Model: 50% sold at 09:15 open, 50% runner with BE stop
        else:
            leg1_pct = d1_o_ret - COST_ROUND_TRIP
            be_stop = entry_eff * 1.003
            if l1 <= be_stop:
                leg2_pct = 0.3 - COST_ROUND_TRIP
                nxt = round(0.5 * o1 + 0.5 * be_stop, 2)
                exit_reason = "09:15 Open 50% + BE Stop 50%"
            else:
                bar2_c = float(d.iloc[j + 2]["close"]) if j + 2 < len(d) else c1
                leg2_pct = max((bar2_c / entry_eff - 1.0) * 100.0 - COST_ROUND_TRIP, 0.3 - COST_ROUND_TRIP)
                nxt = round(0.5 * o1 + 0.5 * bar2_c, 2)
                exit_reason = "09:15 Open 50% + D+2 Runner 50%"
            net_pct = round(0.5 * leg1_pct + 0.5 * leg2_pct, 3)

        gross = net_pct + COST_ROUND_TRIP
        # Sizing: strictly capped at Rs 1,00,000 max per single trade without CAGR
        cap_trade = min(DEFAULT_CAPITAL, 100000.0)
        qty = int(cap_trade // entry_eff) if entry_eff > 0 else 0
        invested = round(qty * entry_eff, 2)
        net_pnl = round(qty * entry_eff * (net_pct / 100.0), 2)
        gross_pnl = round(qty * entry_eff * (gross / 100.0), 2)
        costs = round(gross_pnl - net_pnl, 2)

        # Confirmed BTST trade
        if want_btst and is_btst:
            conv, why = btst.conviction(m_btst)
            out.append(dict(
                date=dt.iloc[j].date().isoformat(), symbol=sym,
                mode="btst",
                tier=m_btst["tier"],
                arm=("fresh_A" if m_btst.get("fresh") and m_btst["tier"] == "A"
                     else "fresh_B" if m_btst.get("fresh") else "aged_B"),
                age=int(m_btst.get("age", 0)),
                qty=qty, invested=invested,
                entry=round(entry_eff, 2), exit=round(nxt, 2),
                gross_pnl=gross_pnl, costs=costs, net_pnl=net_pnl,
                level=round(float(level_c), 2),
                day_ret=round(m_btst["day_ret"], 2),
                close_pos=round(m_btst["close_pos"], 3),
                rvol=round(float(m_btst.get("rvol") or 0), 2),
                atr_pct=round(m_btst["atr_pct"], 2),
                ext_pct=round(float(m_btst.get("ext_pct") or 0), 2),
                ret_12m=btst._num(m_btst.get("ret_12m"), 1),
                pre=int(m_btst.get("pre", 0)),
                conviction=conv, why=";".join(why),
                circuit_band=f"{int(ckt['band'])}%",
                uc_limit=ckt["uc_limit"],
                dist_to_uc=ckt["dist_to_uc"],
                circuit_locked="yes" if is_locked else "no",
                lock_reason=ckt["lock_reason"],
                pre_circuit_entry=ckt["pre_entry"],
                fillable="no" if is_locked else "yes",
                btst_qualified="yes",
                gross_pct=round(gross, 3),
                net_pct=round(gross - COST_ROUND_TRIP, 3),
            ))

        # Anticipated trade
        if want_ant and is_ant:
            if ant_only and is_btst:
                # User selected anticipated-only (non-BTST): skip if it qualified for BTST
                continue
            side = str(m_ant.get("side", "below"))
            arm = f"ant_{side}"
            tier_name = f"ANT_{side.upper()}"
            why_ant = [f"PRE={m_ant.get('pre', 0)}/8", f"side={side}", f"gap={m_ant.get('gap_pct', 0):.1f}%"]
            out.append(dict(
                date=dt.iloc[j].date().isoformat(), symbol=sym,
                mode="anticipated_only" if (not is_btst) else "anticipated",
                tier=tier_name,
                arm=arm,
                age=0,
                qty=qty, invested=invested,
                entry=round(entry_eff, 2), exit=round(nxt, 2),
                gross_pnl=gross_pnl, costs=costs, net_pnl=net_pnl,
                level=round(float(m_ant.get("level", level_c)), 2),
                day_ret=round(m_ant.get("day_ret", 0.0), 2),
                close_pos=round(m_ant.get("close_pos", 0.0), 3),
                rvol=round(float(m_ant.get("rvol") or 0), 2),
                atr_pct=round(m_ant.get("atr_pct", 0.0), 2),
                ext_pct=round(float(-m_ant.get("gap_pct", 0.0) if side == "above" else 0.0), 2),
                ret_12m=btst._num(m_ant.get("ret_12m"), 1),
                pre=int(m_ant.get("pre", 0)),
                conviction=int(m_ant.get("pre", 0) >= 7),
                why=";".join(why_ant),
                circuit_band=f"{int(ckt['band'])}%",
                uc_limit=ckt["uc_limit"],
                dist_to_uc=ckt["dist_to_uc"],
                circuit_locked="yes" if is_locked else "no",
                lock_reason=ckt["lock_reason"],
                pre_circuit_entry=ckt["pre_entry"],
                fillable="no" if is_locked else "yes",
                btst_qualified="yes" if is_btst else "no",
                gross_pct=round(gross, 3),
                net_pct=round(gross - COST_ROUND_TRIP, 3),
            ))

    return out

    return out

    return out


def _rs(args):
    try:
        path, start, end, mode, excl_locks, pre_ckt = args
        return replay_symbol(path, start, end, mode=mode, exclude_circuit_locks=excl_locks,
                             use_pre_circuit_entry=pre_ckt)
    except Exception:
        return []


# --------------------------------------------------------------------------- #
#  REPORTING
# --------------------------------------------------------------------------- #
def _pf(v: pd.Series) -> float:
    g = v[v > 0].sum()
    l = -v[v < 0].sum()
    return float(g / l) if l > 0 else float("inf")


def _stats(x: pd.DataFrame, label: str) -> str:
    if x.empty:
        return f"{label:<20}{'-':>7}"
    v = x.net_pct
    pnl = x.net_pnl if "net_pnl" in x.columns else v * 1000.0
    t = v.mean() / (v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 2 else float("nan")
    wks = max((pd.to_datetime(x.date).max() - pd.to_datetime(x.date).min()).days / 7, 1)
    tot_pnl = float(pnl.sum())
    avg_pnl = float(pnl.mean())
    return (f"{label:<20}{len(v):>7,}{len(v)/wks:>6.1f}{(v>0).mean()*100:>7.1f}"
            f"{v.mean():>+8.2f}%{v.median():>+7.1f}%{_pf(v):>6.2f}{t:>6.1f}"
            f"{(v>=5).mean()*100:>7.1f}%{tot_pnl:>+13,.0f}{avg_pnl:>+9,.0f}")


HDR = (f"{'slice':<20}{'trades':>7}{'/wk':>6}{'win%':>7}{'mean%':>9}"
       f"{'med%':>8}{'PF':>6}{'t':>6}{'P(+5%)':>8}{'Net PnL (Rs)':>13}{'Avg Rs':>9}")


def _calc_priority(r: pd.DataFrame) -> pd.Series:
    """Vectorised twin of btst.actionable_priority() - and asserted to agree.

    2026-08-14 fix - "backtest ranks a different top-3 than the alert":
    this function had the anticipate and fresh_B weights TRANSPOSED
    relative to btst.actionable_priority(), the live alert's own rule:

        live alert   ... + is_ant * 200.0 + is_fresh_b * 100.0
        backtest     ... + is_fresh_b * 200.0 + is_ant * 100.0   <-- wrong

    So for two otherwise-identical candidates the alert preferred the
    anticipate name and the backtest preferred the fresh_B name - a
    straight 100-point inversion that reorders the top-3 book the backtest
    reports as "the LIVE rule replayed over history". It is not the live
    rule if the ordering differs.

    The weights are no longer restated here. btst.actionable_priority() is
    THE rule; this only applies it row-wise, so the two cannot drift again.
    A hand-kept vectorised copy is exactly what produced this bug, and the
    same reasoning is already documented on classify()/exhausted()/
    conviction() being imported from btst.py rather than duplicated.
    """
    if r.empty:
        return pd.Series([], dtype=float, index=r.index)
    # Missing/NaN handling stays HERE, matching the previous vectorised
    # defaults (rvol->0, close_pos->1.0, pre->0). actionable_priority() sees
    # only clean floats, because `float(nan) or 0.0` is nan (nan is truthy),
    # which would silently poison the score.
    arm = (r["arm"].astype("string").fillna("")
           if "arm" in r.columns else pd.Series("", index=r.index))
    def _col(name, default):
        if name not in r.columns:
            return pd.Series(default, index=r.index, dtype=float)
        return pd.to_numeric(r[name], errors="coerce").fillna(default).astype(float)
    rvol = _col("rvol", 0.0)
    cp = _col("close_pos", 1.0)
    pre = _col("pre", 0.0)
    return pd.Series(
        [btst.actionable_priority({"arm": a, "rvol": v, "close_pos": c, "pre": p})
         for a, v, c, p in zip(arm, rvol, cp, pre)],
        index=r.index, dtype=float)


def calculate_compounding_portfolio(df: pd.DataFrame, initial_capital_per_slot: float = DEFAULT_CAPITAL,
                                    max_trades_per_day: int = MAX_DAILY_TRADES,
                                    initial_capital: float | None = None) -> dict:
    """
    Simulate daily compounding (CAGR method) starting with 3 slots of Rs 1,00,000 (Rs 3,00,000 total portfolio).
    Takes at most max_trades_per_day (3) trades per day (ranked by Prime #1 & #2 priority).
    Every next day adjusts position sizing according to cumulative PnL.
    """
    if initial_capital is not None:
        initial_capital_per_slot = initial_capital
    total_initial = initial_capital_per_slot * max_trades_per_day
    if df.empty:
        return {"initial": total_initial, "equity": total_initial, "net_pnl": 0.0,
                "total_return_pct": 0.0, "cagr_pct": 0.0, "max_dd_pct": 0.0, "active_days": 0}
    r = df.copy()
    r["_k"] = _calc_priority(r)
    book = r.sort_values(["date", "_k"], ascending=[True, False]).groupby("date").head(max_trades_per_day)
    days = sorted(book.date.unique())
    equity = total_initial
    curve = [equity]
    for d in days:
        sub = book[book.date == d]
        n_pos = min(len(sub), max_trades_per_day)
        cap_pos = equity / max_trades_per_day
        day_pnl = 0.0
        for _, row in sub.iterrows():
            day_pnl += cap_pos * (float(row["net_pct"]) / 100.0)
        equity += day_pnl
        curve.append(equity)

    tot_ret = (equity - total_initial) / total_initial * 100.0
    d0 = pd.to_datetime(days[0])
    d1 = pd.to_datetime(days[-1])
    years = max((d1 - d0).days / 365.25, 0.05)
    cagr = (((equity / total_initial) ** (1.0 / years)) - 1.0) * 100.0 if equity > 0 else -100.0

    arr = np.array(curve)
    peaks = np.maximum.accumulate(arr)
    dd = (arr - peaks) / peaks * 100.0 if len(peaks) else np.array([0.0])
    max_dd = float(dd.min())

    return {
        "initial": total_initial,
        "equity": equity,
        "net_pnl": equity - total_initial,
        "total_return_pct": tot_ret,
        "cagr_pct": cagr,
        "max_dd_pct": max_dd,
        "active_days": len(days),
    }


def report(df: pd.DataFrame, show_trades: bool, top: int, mode: str = "btst",
           capital: float = DEFAULT_CAPITAL) -> None:
    if df.empty:
        print(f"\nNo {mode} setups in this window.")
        return
    df = df.sort_values("date").reset_index(drop=True)

    mode_label = {
        "btst": "CONFIRMED BTST",
        "anticipated_only": "ANTICIPATED ONLY (DID NOT QUALIFY FOR BTST)",
        "anticipated": "ALL ANTICIPATED SETUPS",
        "all": "CONFIRMED BTST + ANTICIPATED",
    }.get(mode, mode.upper())

    tot_net_rs = df.net_pnl.sum() if "net_pnl" in df.columns else df.net_pct.sum() * (capital / 100.0)
    comp = calculate_compounding_portfolio(df, initial_capital=capital)

    print("\n" + "=" * 115)
    print(f"BTST / ANTICIPATION BACKTEST ({mode_label}) - the LIVE rule replayed over history")
    print("=" * 115)
    print(f"window      {df.date.min()} .. {df.date.max()} ({comp['active_days']} active sessions)")
    print(f"capital     Rs {capital:,.0f} initial -> Compounded Equity: Rs {comp['equity']:,.0f} (Net: Rs {comp['net_pnl']:+,.0f}, {comp['total_return_pct']:+.2f}%)")
    print(f"cagr        {comp['cagr_pct']:+.2f}% CAGR · Compounded Max Drawdown: {comp['max_dd_pct']:.2f}%")
    print(f"names hit   {df.symbol.nunique():,} distinct symbols produced a setup")
    print(f"thresholds  TIER_A day>={btst.TIER_A_DAY:g}% cp>={btst.TIER_A_CLOSE_POS} | "
          f"TIER_B cp>={btst.TIER_B_CLOSE_POS} rvol>={btst.TIER_B_RVOL:g} "
          f"atr>={btst.TIER_B_ATR:g}%")
    print(f"            aged band {btst.AGED_EXT_MIN:+g}..{btst.AGED_EXT_MAX:+g}%, "
          f"exhaustion base<{btst.MAX_BASE_FROM_HIGH:g}% & 3m<={btst.MAX_RET_3M_PRIOR:g}%")
    print(f"            anticipate near <={btst.ANTICIPATE_NEAR:g}% below / +{btst.ANTICIPATE_ABOVE_MAX:g}% above, "
          f"cp>={btst.ANTICIPATE_CLOSE_POS}, pre>={btst.MIN_PRE_CONFIRM}, ret_12m>={btst.MIN_RET_12M:g}%")

    print("\n" + HDR)
    print(_stats(df, "ALL"))
    print()

    # Slices
    arms = sorted(df.arm.dropna().unique())
    if len(arms) > 1:
        for arm in arms:
            print(_stats(df[df.arm == arm], f"  {arm}"))
        print()

    tiers = sorted(df.tier.dropna().unique())
    if len(tiers) > 1:
        for tier in tiers:
            print(_stats(df[df.tier == tier], f"  {tier}"))
        print()

    if "circuit_locked" in df.columns and df.circuit_locked.nunique() > 1:
        print(_stats(df[df.circuit_locked == "no"], "  fillable (no lock)"))
        print(_stats(df[df.circuit_locked == "yes"], "  circuit locked (UC)"))
        print()

    if "btst_qualified" in df.columns and df.btst_qualified.nunique() > 1:
        for bq in sorted(df.btst_qualified.unique()):
            label = "BTST qualified" if bq == "yes" else "BTST not qualified"
            print(_stats(df[df.btst_qualified == bq], f"  {label}"))
        print()

    if "pre" in df.columns and df.pre.max() > 0:
        for p in sorted(df.pre.unique()):
            if p >= 5:
                print(_stats(df[df.pre == p], f"  PRE score {p}/8"))
        print()

    print("\nBY YEAR")
    print(HDR)
    for y, g in df.assign(y=pd.to_datetime(df.date).dt.year).groupby("y"):
        print(_stats(g, f"  {y}"))

    # a top-3/day book (max 3 trades per day)
    print(f"\nTOP-{MAX_DAILY_TRADES} PER DAY (the traded portfolio: max {MAX_DAILY_TRADES} trades/day, Rs {capital:,.0f}/trade max)")
    r = df.copy()
    r["_k"] = _calc_priority(r)
    book = r.sort_values(["date", "_k"], ascending=[True, False]).groupby("date").head(MAX_DAILY_TRADES)
    print(HDR)
    print(_stats(book, f"  top-{MAX_DAILY_TRADES}/day"))
    day = book.groupby("date").net_pct.mean()
    day_pnl = book.groupby("date").net_pnl.sum() if "net_pnl" in book.columns else day * 1000.0
    eq = (1 + day / 100).cumprod()
    dd = (eq / eq.cummax() - 1).min() * 100
    print(f"\n  active days {len(day):,}   total net Rs {day_pnl.sum():>+,.0f}   "
          f"mean day {day.mean():+.3f}% (Rs {day_pnl.mean():>+,.0f}/day)   "
          f"max drawdown {dd:.1f}%   worst day {day.min():+.2f}% (Rs {day_pnl.min():>+,.0f})")

    if show_trades:
        print("\n" + "=" * 128)
        print(f"TOP-{MAX_DAILY_TRADES} DAILY TRADED BOOK (most recent {top}) — Rs {capital:,.0f} per trade")
        print("=" * 128)
        print(f"{'date':<12}{'symbol':<12}{'tier':<10}{'qty':>5}{'entry':>9}{'exit':>9}"
              f"{'day%':>6}{'rvol':>5}{'pre':>4}{'circuit':>8}{'fillable':>12}{'net%':>7}{'Net P&L (Rs)':>14}")
        for r_ in book.tail(top).itertuples():
            pnl_str = f"{r_.net_pnl:>+13,.0f}" if hasattr(r_, "net_pnl") else f"{r_.net_pct*1000:>+13,.0f}"
            qty_val = getattr(r_, "qty", int(capital // r_.entry))
            pre_val = getattr(r_, "pre", 0)
            ckt_band = getattr(r_, "circuit_band", "-")
            lock_r = getattr(r_, "lock_reason", "")
            if "hard" in lock_r or "flat" in lock_r:
                fill_stat = "🔒 HARD_LOCK"
            elif "opened" in lock_r or getattr(r_, "circuit_opened", False):
                fill_stat = "⚡ OPEN_WIN"
            elif getattr(r_, "circuit_locked", "no") == "yes":
                fill_stat = "🔒 UC_LOCK"
            else:
                fill_stat = "✅ OPEN"
            print(f"{r_.date:<12}{r_.symbol:<12}{r_.tier:<10}{qty_val:>5d}"
                  f"{r_.entry:>9.2f}{r_.exit:>9.2f}{r_.day_ret:>6.1f}{r_.rvol:>5.1f}"
                  f"{pre_val:>4}{ckt_band:>8}{fill_stat:>12}{r_.net_pct:>+6.2f}%{pnl_str}")

    print("\n" + "-" * 128)
    print("CIRCUIT ENTRY: In 75% of circuit cases, the circuit OPENS during the day (open windows with active sellers).")
    print("Use --pre-circuit to simulate entering on open windows/ramps ~0.8% before the final ceiling.")
    print(f"Capital sized at Rs {capital:,.0f} per trade; costs charged at {COST_ROUND_TRIP}% round trip.")
    print("Yahoo daily data, not Dhan - per-symbol volume can differ, so counts may not match live exactly.")
    print("One regime (2021-2026). Past results are not forward validation.")


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capital", type=float, default=DEFAULT_CAPITAL,
                    help="rupees deployed per trade (default 1,00,000)")
    ap.add_argument("--mode", choices=["btst", "anticipated", "anticipated_only", "all"],
                    default="all",
                    help="all (default): confirmed BTST + anticipated setups | "
                         "btst: confirmed breakouts only | "
                         "anticipated_only: anticipated setups that did NOT qualify for BTST at 15:20 | "
                         "anticipated: all pre-breakout anticipation setups")
    ap.add_argument("--exclude-circuit-locks", action="store_true",
                    help="exclude trades where stock closed locked at upper circuit (unfillable at 15:20)")
    ap.add_argument("--all-candidates", action="store_true",
                    help="write all qualifying candidates to CSV instead of the Top-3 traded book")
    ap.add_argument("--pre-circuit", action="store_true",
                    help="simulate pre-circuit entry (entered ~0.8%% before circuit level was touched)")
    ap.add_argument("--years", type=float, default=2.0,
                    help="how far back to replay (default 2)")
    ap.add_argument("--from", dest="from_date", default="",
                    help="start date YYYY-MM-DD (overrides --years)")
    ap.add_argument("--to", dest="to_date", default="", help="end date YYYY-MM-DD")
    ap.add_argument("--symbol", default="", help="one symbol, or comma-separated")
    ap.add_argument("--trades", action="store_true", help="print the trade list")
    ap.add_argument("--top", type=int, default=60, help="how many trades to print")
    ap.add_argument("--csv", default="", help="write every trade to this CSV")
    ap.add_argument("--data", default=DATA_DIR, help="daily-bar cache directory")
    ap.add_argument("--no-fetch", action="store_true",
                    help="use the cache only, do not download")
    ap.add_argument("--universe", default="universe.csv")
    args = ap.parse_args()

    end = args.to_date or pd.Timestamp.today().strftime("%Y-%m-%d")
    start = (args.from_date
             or (pd.Timestamp(end) - pd.Timedelta(days=int(args.years * 365.25))
                 ).strftime("%Y-%m-%d"))

    if args.symbol:
        syms = [s.strip().upper() for s in args.symbol.split(",") if s.strip()]
    else:
        uni = pd.read_csv(args.universe)
        syms = sorted(uni.symbol.astype(str).str.strip().unique())

    print(f"BTST / Anticipation backtest ({args.mode})  {start} .. {end}   {len(syms):,} symbol(s)")
    print(f"cache: {args.data}")

    if not args.no_fetch:
        missing = [s for s in syms if not os.path.exists(os.path.join(args.data, f"{s}.csv"))]
        if missing:
            print(f"fetching daily history for {len(missing)} symbol(s) ...")
            t0 = time.time()
            years_fetch = max(int(args.years) + 3, 5)
            got = fetch_history(missing, args.data, years=years_fetch)
            print(f"  {got:,} symbols downloaded in {time.time()-t0:.0f}s")

    files = [os.path.join(args.data, f"{s}.csv") for s in syms]
    files = [f for f in files if os.path.exists(f)]
    if not files:
        print("\nNo cached data. Drop --no-fetch, or check --data.")
        return 2

    t0 = time.time()
    rows: list[dict] = []
    jobs = [(f, start, end, args.mode, args.exclude_circuit_locks, args.pre_circuit) for f in files]
    if len(files) > 8:
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
            for i, r in enumerate(ex.map(_rs, jobs, chunksize=8)):
                rows.extend(r)
                if i and i % 500 == 0:
                    print(f"  replayed {i}/{len(files)} ...", flush=True)
    else:
        for j in jobs:
            rows.extend(_rs(j))

    df = pd.DataFrame(rows)
    print(f"replayed {len(files):,} symbols in {time.time()-t0:.0f}s -> "
          f"{len(df):,} setups")

    report(df, args.trades or bool(args.symbol), args.top, mode=args.mode, capital=args.capital)

    if args.csv and not df.empty:
        r = df.copy()
        r["_k"] = _calc_priority(r)
        if args.all_candidates:
            out_df = r.sort_values(["date", "_k"], ascending=[True, False]).drop(columns=["_k"], errors="ignore")
        else:
            out_df = r.sort_values(["date", "_k"], ascending=[True, False]).groupby("date").head(MAX_DAILY_TRADES).drop(columns=["_k"], errors="ignore")
        out_df.sort_values("date").to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv} ({len(out_df):,} traded rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
