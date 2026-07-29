"""
Exact Python translation of "Weekly Breakout Scanner + 5m Entry [Chartink] v3".

Two objects matter here:

WeeklySnapshot
    Everything the entry needs that is FROZEN for the whole week, derived from
    CLOSED weekly bars only. Pine offset mapping (offset 0 = developing week):
        offset 1 -> closed[-1]      offset k -> closed[-k]

BarReplay
    Walks the current week's 5-minute candles in order, reproducing the Pine
    bar-state variables (`tookThisWeek`, `sawCrossThisWeek`, `close[1]`) so the
    signal lands on exactly the candle the indicator marks with BUY.

Deliberate deviation (one, and it is the one you asked for)
    c12 "market cap > 1000" is disabled: Dhan exposes no shares-outstanding
    field, and Pine's `request.financial` has no API equivalent. Set
    `strategy.use_mcap: true` plus a data/mcap.csv to re-enable it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

import indicators as ind
from config import Strategy

# --------------------------------------------------------------------------- #
#  Weekly bar construction
# --------------------------------------------------------------------------- #
def build_weekly_bars(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily candles into Monday-anchored weekly bars, matching the
    TradingView "W" resolution used by request.security(..., "W", ...).
    """
    if daily.empty:
        return pd.DataFrame(columns=["week_start", "open", "high", "low", "close", "volume"])

    df = daily.copy()
    df["day"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None).dt.normalize()
    df["week_start"] = df["day"] - pd.to_timedelta(df["day"].dt.weekday, unit="D")

    weekly = df.groupby("week_start", sort=True).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).reset_index()
    return weekly


def week_start_of(d: date) -> pd.Timestamp:
    ts = pd.Timestamp(d).normalize()
    return ts - pd.Timedelta(days=ts.weekday())


# --------------------------------------------------------------------------- #
#  Frozen weekly state
# --------------------------------------------------------------------------- #
@dataclass
class WeeklySnapshot:
    """Closed-week values; constant from Monday to Friday."""
    symbol: str
    security_id: str
    exchange_segment: str
    week_start: str                     # the developing week this snapshot serves

    # breakout levels (Pine: entryLevel / level52 / wHiLong / wHiShort2)
    entry_level: float                  # highest(high,26)[1]
    level_52: float                     # highest(high,52)[1]
    hi_short2: float                    # highest(high,26)[2]
    close_1: float                      # close[1]

    # developing-week indicator states
    ema_fast: ind.EmaState
    ema_slow: ind.EmaState
    ema_slow_2: float                   # ema(close,50)[emaSlowBack]
    rsi: ind.RsiState
    rsi_1: float                        # rsi[1]
    macd: ind.MacdState
    vol_sma: ind.SmaState

    # closed-week gate (Pine f_gate)
    g_ema_fast: float                   # ema(fast)[1]
    g_ema_slow: float                   # ema(slow)[1]
    g_ema_slow_2: float                 # ema(slow)[1+emaSlowBack]
    g_rsi: float                        # rsi[1]
    g_rsi_1: float                      # rsi[2]
    g_hist: float                       # macd hist[1]

    # previous completed daily bar (Pine prevDClose / prevDOpen)
    prev_daily_close: float
    prev_daily_open: float

    mcap: float | None = None

    def to_row(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "security_id": self.security_id,
            "exchange_segment": self.exchange_segment,
            "week_start": self.week_start,
            "entry_level": self.entry_level,
            "level_52": self.level_52,
            "hi_short2": self.hi_short2,
            "close_1": self.close_1,
            "ema_fast_prev": self.ema_fast.prev,
            "ema_fast_len": self.ema_fast.length,
            "ema_slow_prev": self.ema_slow.prev,
            "ema_slow_len": self.ema_slow.length,
            "ema_slow_2": self.ema_slow_2,
            "rsi_len": self.rsi.length,
            "rsi_avg_gain": self.rsi.avg_gain,
            "rsi_avg_loss": self.rsi.avg_loss,
            "rsi_prev_close": self.rsi.prev_close,
            "rsi_1": self.rsi_1,
            "macd_fast": self.macd.fast,
            "macd_slow": self.macd.slow,
            "macd_signal": self.macd.signal,
            "macd_ema_fast": self.macd.ema_fast,
            "macd_ema_slow": self.macd.ema_slow,
            "macd_sig": self.macd.sig,
            "vol_sma_len": self.vol_sma.length,
            "vol_sma_sum_prev": self.vol_sma.sum_prev,
            "g_ema_fast": self.g_ema_fast,
            "g_ema_slow": self.g_ema_slow,
            "g_ema_slow_2": self.g_ema_slow_2,
            "g_rsi": self.g_rsi,
            "g_rsi_1": self.g_rsi_1,
            "g_hist": self.g_hist,
            "prev_daily_close": self.prev_daily_close,
            "prev_daily_open": self.prev_daily_open,
            "mcap": self.mcap if self.mcap is not None else "",
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "WeeklySnapshot":
        f = lambda k: float(row[k])  # noqa: E731
        i = lambda k: int(float(row[k]))  # noqa: E731
        mcap_raw = row.get("mcap", "")
        return cls(
            symbol=str(row["symbol"]),
            security_id=str(row["security_id"]),
            exchange_segment=str(row["exchange_segment"]),
            week_start=str(row["week_start"]),
            entry_level=f("entry_level"),
            level_52=f("level_52"),
            hi_short2=f("hi_short2"),
            close_1=f("close_1"),
            ema_fast=ind.EmaState(length=i("ema_fast_len"), prev=f("ema_fast_prev")),
            ema_slow=ind.EmaState(length=i("ema_slow_len"), prev=f("ema_slow_prev")),
            ema_slow_2=f("ema_slow_2"),
            rsi=ind.RsiState(length=i("rsi_len"), avg_gain=f("rsi_avg_gain"),
                             avg_loss=f("rsi_avg_loss"), prev_close=f("rsi_prev_close")),
            rsi_1=f("rsi_1"),
            macd=ind.MacdState(fast=i("macd_fast"), slow=i("macd_slow"),
                               signal=i("macd_signal"), ema_fast=f("macd_ema_fast"),
                               ema_slow=f("macd_ema_slow"), sig=f("macd_sig")),
            vol_sma=ind.SmaState(length=i("vol_sma_len"), sum_prev=f("vol_sma_sum_prev")),
            g_ema_fast=f("g_ema_fast"),
            g_ema_slow=f("g_ema_slow"),
            g_ema_slow_2=f("g_ema_slow_2"),
            g_rsi=f("g_rsi"),
            g_rsi_1=f("g_rsi_1"),
            g_hist=f("g_hist"),
            prev_daily_close=f("prev_daily_close"),
            prev_daily_open=f("prev_daily_open"),
            mcap=float(mcap_raw) if str(mcap_raw).strip() not in ("", "nan") else None,
        )


def build_snapshot(symbol: str, security_id: str, exchange_segment: str,
                   daily: pd.DataFrame, cfg: Strategy,
                   target_week: pd.Timestamp,
                   mcap: float | None = None) -> WeeklySnapshot | None:
    """
    Build the frozen weekly state for `target_week` using only weeks that closed
    before it. Returns None when history is too short for a valid comparison.
    """
    weekly = build_weekly_bars(daily)
    if weekly.empty:
        return None

    closed = weekly[weekly["week_start"] < target_week].reset_index(drop=True)

    # len_long + 1 / len_short + 2 cover the extra week the breakout levels now
    # step back (see the note on entry_level below).
    need = max(cfg.len_long + 1, cfg.len_short + 2, cfg.ema_slow_len + cfg.ema_slow_back + 1,
               cfg.macd_slow + cfg.macd_sig, cfg.rsi_len + 3, cfg.vol_sma_len)
    if len(closed) < need:
        return None

    closes = closed["close"].to_numpy(dtype=float)
    highs = closed["high"].to_numpy(dtype=float)
    volumes = closed["volume"].to_numpy(dtype=float)

    # ---- breakout levels
    #
    # Pine:  request.security(..., "W", ta.highest(high, N)[1], lookahead_off)
    #
    # `[1]` steps back one bar from the DEVELOPING week, i.e. the window ends
    # at the last CLOSED week and includes it. `closed` already excludes the
    # developing week, so its final row IS that bar and must be kept.
    #
    # Verified live on TMB, Tue 28-Jul 15:08, mid-session:
    #     "2. Wk close > 26W high (prev)   885.5  needs > 847"   -> 847.00
    #     "3. Prev wk close <= 26W high (2w ago)  <= 821.2"      -> 821.20
    # 847.00 is the high of w/c 20-Jul (the last closed week), so that week is
    # INSIDE the 26W window. 821.20 is the separate 2-weeks-ago value (hi_short2).
    #
    # An earlier build shifted this back one extra week after reading a chart
    # captured at 21:30, once TradingView had already rolled the weekly bar -
    # that snapshot showed next week's framing, not the live level. Reverted.
    entry_level = float(highs[-cfg.len_short:].max())
    level_52 = float(highs[-cfg.len_long:].max())
    hi_short2 = float(highs[-(cfg.len_short + 1):-1].max())

    # ---- developing-week states
    ema_fast_state = ind.build_ema_state(closes, cfg.ema_fast_len)
    ema_slow_state = ind.build_ema_state(closes, cfg.ema_slow_len)
    rsi_state = ind.build_rsi_state(closes, cfg.rsi_len)
    macd_state = ind.build_macd_state(closes, cfg.macd_fast, cfg.macd_slow, cfg.macd_sig)
    vol_state = ind.build_sma_state(volumes, cfg.vol_sma_len)
    if not all((ema_fast_state, ema_slow_state, rsi_state, macd_state, vol_state)):
        return None

    ema_slow_series = ind.ema(closes, cfg.ema_slow_len)
    ema_fast_series = ind.ema(closes, cfg.ema_fast_len)
    rsi_series = ind.rsi(closes, cfg.rsi_len)
    _, _, hist_series = ind.macd(closes, cfg.macd_fast, cfg.macd_slow, cfg.macd_sig)

    def at(series: np.ndarray, offset: int) -> float:
        """Pine `series[offset]` where offset 1 == last closed week."""
        idx = len(series) - offset
        if idx < 0 or np.isnan(series[idx]):
            return float("nan")
        return float(series[idx])

    ema_slow_2 = at(ema_slow_series, cfg.ema_slow_back)
    rsi_1 = at(rsi_series, 1)

    g_ema_fast = at(ema_fast_series, 1)
    g_ema_slow = at(ema_slow_series, 1)
    g_ema_slow_2 = at(ema_slow_series, 1 + cfg.ema_slow_back)
    g_rsi = at(rsi_series, 1)
    g_rsi_1 = at(rsi_series, 2)
    g_hist = at(hist_series, 1)

    for val in (ema_slow_2, rsi_1, g_ema_fast, g_ema_slow, g_ema_slow_2, g_rsi, g_rsi_1, g_hist):
        if np.isnan(val):
            return None

    # ---- previous completed daily bar
    prev_daily_close = prev_daily_open = float("nan")
    if not daily.empty:
        d = daily.copy()
        d["day"] = pd.to_datetime(d["datetime"]).dt.tz_localize(None).dt.normalize()
        before = d[d["day"] < target_week + pd.Timedelta(days=7)]
        completed = before[before["day"] < pd.Timestamp(datetime.now().date())]
        src = completed if not completed.empty else before
        if not src.empty:
            prev_daily_close = float(src.iloc[-1]["close"])
            prev_daily_open = float(src.iloc[-1]["open"])

    return WeeklySnapshot(
        symbol=symbol, security_id=str(security_id), exchange_segment=exchange_segment,
        week_start=str(target_week.date()),
        entry_level=entry_level, level_52=level_52, hi_short2=hi_short2,
        close_1=float(closes[-1]),
        ema_fast=ema_fast_state, ema_slow=ema_slow_state, ema_slow_2=ema_slow_2,
        rsi=rsi_state, rsi_1=rsi_1, macd=macd_state, vol_sma=vol_state,
        g_ema_fast=g_ema_fast, g_ema_slow=g_ema_slow, g_ema_slow_2=g_ema_slow_2,
        g_rsi=g_rsi, g_rsi_1=g_rsi_1, g_hist=g_hist,
        prev_daily_close=prev_daily_close, prev_daily_open=prev_daily_open,
        mcap=mcap,
    )


# --------------------------------------------------------------------------- #
#  Per-bar evaluation
# --------------------------------------------------------------------------- #
from zoneinfo import ZoneInfo
IST_TZ = ZoneInfo("Asia/Kolkata")

# --------------------------------------------------------------------------- #
#  Snapshot logic version - SINGLE SOURCE OF TRUTH
#
#  Bump whenever anything that changes a STORED snapshot value is edited
#  (breakout levels, indicator maths, columns) - including when a change is
#  REVERTED. Both the writer (build_snapshot.py) and every reader (scan.py,
#  watch.py) compare against this, so a row from an older generation is
#  discarded instead of being silently trusted.
#
#  3 (29-Jul-2026): breakout level settled as Option B -
#      entry_level = highest(high, 26) INCLUDING the last closed week.
#    Version 2 rows were written while the reverted one-week shift was live and
#    hold levels ~1% too low. scan.py used to load them regardless of version,
#    which is how RADICO alerted against 4161.80 instead of 4193.00 and how
#    JKPAPER / THYROCARE / VAIBHAVGBL produced outright false alerts.
LOGIC_VERSION = 3

BARS_PER_WEEK = 375        # NSE: 75 five-minute bars x 5 sessions
ATR_LOOKBACK = 14          # 5m bars used by the min_atr_pct intraday filter

COND_LABELS = {
    "c01": "Wk close > 52W high",
    "c02": "Wk close > 26W high",
    "c03": "Prev wk close <= 26W high (2w ago)",
    "c04": "Wk EMA20 > EMA50",
    "c05": "Wk EMA50 rising",
    "c06": "Wk RSI > 60",
    "c07": "Wk RSI rising",
    "c08": "Wk MACD hist > 0",
    "c09": "Wk vol > SMA(vol,10)",
    "c10": "Weekly candle green",
    "c11": "Daily close > 100",
    "c12": "Market cap filter",
    "c13": "Daily candle green",
}


@dataclass
class BarEval:
    """The 13 conditions plus the derived values, for one 5-minute bar."""
    conditions: dict[str, bool]
    values: dict[str, float]

    @property
    def all_ok(self) -> bool:
        return all(self.conditions.values())

    @property
    def pass_count(self) -> int:
        return sum(1 for v in self.conditions.values() if v)

    @property
    def failed(self) -> list[str]:
        return [COND_LABELS[k] for k, v in self.conditions.items() if not v]


@dataclass
class Signal:
    symbol: str
    security_id: str
    exchange_segment: str
    bar_time: datetime
    price: float
    entry_level: float
    level_52: float
    trigger: str                    # "cross" | "deferred"
    evaluation: BarEval
    week_start: str
    week_volume: float
    day_open: float
    week_open: float
    # OHLC of the triggering 5m candle. Optional so existing callers and any
    # pickled/constructed Signals keep working; paper trading needs `bar_low`
    # to place the stop just under the entry candle.
    bar_open: float = 0.0
    bar_high: float = 0.0
    bar_low: float = 0.0
    rvol: float = float('nan')   # relative volume at the signal bar



def compute_rvol(bars: pd.DataFrame, history: pd.DataFrame,
                 at: pd.Timestamp, sessions: int = 20) -> float:
    """
    Relative volume at a point in the session.

    Volume traded TODAY up to `at`, divided by the median volume traded up to
    the same clock time across the previous `sessions` days.

    1.0 = a normal day. 5.0 = five times the usual participation by now.

    Unlike Pine's c09 (accumulating week vs full-week average) this is
    time-of-day aware, so it is equally meaningful at 09:20 and at 15:20.

    Returns NaN when it cannot be computed - notably when the current bar
    reports zero volume, which some feeds do for the 09:15 opening bar.
    Callers must treat NaN as "unknown", never as "weak".
    """
    if bars.empty:
        return float("nan")
    at = pd.Timestamp(at)
    minute = at.hour * 60 + at.minute

    def _mins(df):
        ts = pd.to_datetime(df["datetime"])
        try:
            ts = ts.dt.tz_convert(IST_TZ)
        except (TypeError, AttributeError):
            pass
        return ts.dt.hour * 60 + ts.dt.minute, ts.dt.tz_localize(None).dt.normalize()

    bmin, bday = _mins(bars)
    today = bars[(bday == at.normalize().tz_localize(None)) & (bmin <= minute)]
    vol_today = float(today["volume"].sum())
    if vol_today <= 0:
        return float("nan")          # opening-bar artifact; unknown, not weak

    if history is None or history.empty:
        return float("nan")
    hmin, hday = _mins(history)
    past = history[hday < at.normalize().tz_localize(None)]
    if past.empty:
        return float("nan")
    pmin, pday = _mins(past)
    totals = []
    for _, grp in past.assign(_m=pmin, _d=pday).groupby("_d"):
        v = float(grp[grp["_m"] <= minute]["volume"].sum())
        if v > 0:
            totals.append(v)
    totals = totals[-sessions:]
    if not totals:
        return float("nan")
    base = float(np.median(totals))
    return vol_today / base if base > 0 else float("nan")


def compute_bar_rvol(bars: pd.DataFrame, history: pd.DataFrame,
                     at: pd.Timestamp, sessions: int = 20) -> float:
    """
    Single-bar relative volume.

    Volume of the 5-minute candle at `at`, divided by the median volume of the
    SAME clock-minute candle over the previous `sessions` days.

    Where `compute_rvol` answers "is today busy?", this answers "did something
    just happen in THIS candle?". An explosive breakout is normally preceded by
    a single bar trading many multiples of its usual size: measured on the
    27-Jul movers, MONARCH printed 20.8x, SENCO 19.6x and TMB 382.7x on the
    breakout candle.

    Returns NaN when it cannot be computed. Callers must treat NaN as
    "unknown", never as "weak".
    """
    if bars.empty:
        return float("nan")
    at = pd.Timestamp(at)
    minute = at.hour * 60 + at.minute

    def _mins(df):
        ts = pd.to_datetime(df["datetime"])
        try:
            ts = ts.dt.tz_convert(IST_TZ)
        except (TypeError, AttributeError):
            pass
        return ts.dt.hour * 60 + ts.dt.minute, ts.dt.tz_localize(None).dt.normalize()

    bmin, bday = _mins(bars)
    day = at.normalize().tz_localize(None)
    cur = bars[(bday == day) & (bmin == minute)]
    if cur.empty:
        return float("nan")
    vol_now = float(cur["volume"].sum())
    if vol_now <= 0:
        return float("nan")          # opening-bar artifact; unknown, not weak

    if history is None or history.empty:
        return float("nan")
    hmin, hday = _mins(history)
    past = history[(hday < day) & (hmin == minute)]
    if past.empty:
        return float("nan")
    pmin, pday = _mins(past)
    totals = []
    for _, grp in past.assign(_d=pday).groupby("_d"):
        v = float(grp["volume"].sum())
        if v > 0:
            totals.append(v)
    totals = totals[-sessions:]
    if not totals:
        return float("nan")
    base = float(np.median(totals))
    return vol_now / base if base > 0 else float("nan")


def evaluate_bar(snap: WeeklySnapshot, cfg: Strategy, price: float,
                 week_open: float, week_volume: float, day_open: float,
                 week_fraction: float = 1.0) -> BarEval:
    """
    The 13-row condition table for a single 5-minute close.

    `week_fraction` is how much of the trading week has elapsed (0..1). It only
    affects c09. Weekly volume ACCUMULATES while the 10-week SMA is a full-week
    figure, so comparing them mid-week is apples to oranges: at Monday 09:25
    only 0.8% of the week's volume exists and c09 can never pass. Pro-rating
    the target asks the fair question - "is this stock trading above its normal
    pace SO FAR?" - which is what the condition is really testing.
    """
    ema_f = snap.ema_fast.step(price)
    ema_s = snap.ema_slow.step(price)
    rsi_v = snap.rsi.step(price)
    hist = snap.macd.step(price)
    vol_sma = snap.vol_sma.step(week_volume)
    # Pace-adjusted volume target. week_fraction == 1.0 reproduces Pine exactly
    # (used for the end-of-week table); intraday it scales the bar.
    frac = min(max(float(week_fraction), 1e-6), 1.0)
    mode = getattr(cfg, "volume_mode", None)
    if mode is None:
        mode = "bar" if getattr(cfg, "volume_prorate", False) else "off"
    if mode == "bar":
        # elapsed 5m bars / 375. Very permissive early in the week.
        vol_target = vol_sma * frac
    elif mode == "pace":
        # Compare like with like: weekly volume SO FAR against the share of the
        # weekly average that SHOULD have traded by now, times a conviction
        # multiple. NSE volume accumulates near-linearly through the week
        # (measured: Mon 17.7%, Tue 39.2%, Wed 59.0%, Thu 80.7% of the total),
        # so elapsed-bar fraction is a sound proxy for expected pace.
        #
        # multiple = 1.0 -> merely average participation (too loose)
        # multiple = 2-3 -> genuine conviction, and reachable on Monday
        vol_target = vol_sma * frac * float(getattr(cfg, "volume_pace_mult", 2.5))
    elif mode == "day":
        # Pro-rate by whole SESSIONS: a Monday breakout must already show one
        # full day of the weekly average, Tuesday two, and so on. Strict enough
        # to demand real conviction, fair enough that a genuine Monday move is
        # not judged against a whole week's volume it cannot yet have.
        sessions = min(max(math.ceil(frac * 5.0), 1), 5)
        vol_target = vol_sma * (sessions / 5.0)
    else:
        vol_target = vol_sma

    if cfg.use_mcap:
        c12 = snap.mcap is not None and snap.mcap > cfg.min_mcap
    else:
        c12 = True

    conditions = {
        "c01": price > snap.level_52,
        "c02": price > snap.entry_level,
        "c03": snap.close_1 <= snap.hi_short2,
        "c04": ema_f > ema_s,
        "c05": ema_s > snap.ema_slow_2,
        "c06": rsi_v > cfg.rsi_min,
        "c07": rsi_v > snap.rsi_1,
        "c08": hist > 0.0,
        "c09": week_volume > vol_target,
        "c10": price > week_open,
        "c11": price > cfg.min_price,
        "c12": c12,
        "c13": price > day_open,
    }
    values = {
        "price": price, "ema_fast": ema_f, "ema_slow": ema_s, "ema_slow_2": snap.ema_slow_2,
        "rsi": rsi_v, "rsi_1": snap.rsi_1, "macd_hist": hist,
        "week_volume": week_volume, "vol_sma": vol_sma,
        "vol_target": vol_target, "week_fraction": frac,
        "week_open": week_open, "day_open": day_open,
        "entry_level": snap.entry_level, "level_52": snap.level_52,
    }
    return BarEval(conditions=conditions, values=values)


def gate_ok(snap: WeeklySnapshot, cfg: Strategy, ev: BarEval) -> bool:
    """
    Pine `gateOK`, with an optional tolerance.

    `gate_tolerance` allows up to N of the gate conditions to fail. It exists so
    a setup that is momentum-healthy but marginal on one or two rows (a weekly
    RSI at 59.8, an EMA slope that just flattened) still triggers on the
    breakout candle instead of waiting and printing as "deferred".

    The BREAKOUT ITSELF IS NEVER RELAXED. c01/c02 - the close above the 26W and
    52W levels - are proven by the cross and are not part of this count, so no
    tolerance can ever manufacture an entry without a genuine level break.
    """
    tol = max(0, int(getattr(cfg, "gate_tolerance", 0)))

    # ---- MOVER MODE: drop c09 from the live gate ---------------------------
    # c09 ("weekly volume > 10-week average") cannot be true early in a week -
    # the week has not traded enough volume yet. Measured over 446 paired
    # signals it delayed 72% of entries by a median of 290 minutes, forcing
    # a chase of +1.91% above the true cross. The consequence is direct:
    #
    #     entering at the raw cross : 50.2% of trades break the entry low
    #     entering at the c09 gate  : 67.5% break it
    #     reaching +5% intraday     : 19.7% raw vs 11.0% gated
    #
    # Late entry is what makes the stop get hunted. `drop_c09` removes only
    # that row; every other weekly-momentum condition still applies, and the
    # level break itself (c01/c02) is untouched. Volume conviction is then
    # enforced properly by the time-of-day-aware RVOL filter, which is what
    # c09 was always trying and failing to express intraday.
    drop_c09 = bool(getattr(cfg, "drop_c09", False))

    if cfg.gate_source == "live":
        # gateLive = c03..c13 (c01/c02 are the level break itself)
        keys = ("c03", "c04", "c05", "c06", "c07", "c08",
                "c09", "c10", "c11", "c12", "c13")
        if drop_c09:
            keys = tuple(k for k in keys if k != "c09")
        # Rows that must ALWAYS hold, whatever the tolerance:
        #   c03 fresh breakout - without it we re-enter an old, extended move
        #   c11 min price      - a hard tradability filter
        #   c12 market cap     - a hard universe filter
        mandatory = ("c03", "c11", "c12")
        if not all(ev.conditions[k] for k in mandatory):
            return False
        failed = sum(1 for k in keys if not ev.conditions[k])
        gate = failed <= tol
    else:
        checks = (
            snap.g_ema_fast > snap.g_ema_slow,
            snap.g_ema_slow > snap.g_ema_slow_2,
            snap.g_rsi > cfg.rsi_min,
            snap.g_rsi > snap.g_rsi_1,
            snap.g_hist > 0.0,
        )
        # fresh-breakout and market-cap stay mandatory here too
        if not (snap.close_1 <= snap.hi_short2 and ev.conditions["c12"]):
            return False
        gate = sum(1 for c in checks if not c) <= tol

    if cfg.gate_daily:
        pd_close, pd_open = snap.prev_daily_close, snap.prev_daily_open
        daily_extra = (not np.isnan(pd_close) and not np.isnan(pd_open)
                       and pd_close > cfg.min_price and pd_close > pd_open)
        gate = gate and daily_extra
    return gate


# --------------------------------------------------------------------------- #
#  Week replay
# --------------------------------------------------------------------------- #
@dataclass
class ReplayResult:
    signals: list[Signal] = field(default_factory=list)
    took_this_week: bool = False
    saw_cross_this_week: bool = False
    last_eval: BarEval | None = None
    bars: int = 0


def replay_week(snap: WeeklySnapshot, cfg: Strategy, bars: pd.DataFrame,
                history: pd.DataFrame | None = None) -> ReplayResult:
    """
    Walk this week's 5-minute candles in order and reproduce the Pine bar state.

    `bars` must be the current week's 5m candles, ascending, columns:
    datetime, open, high, low, close, volume.
    """
    result = ReplayResult()
    if bars.empty:
        return result

    bars = bars.sort_values("datetime").reset_index(drop=True)
    result.bars = len(bars)

    week_open = float(bars.iloc[0]["open"])
    week_volume = 0.0
    day_open = float(bars.iloc[0]["open"])
    current_day = pd.Timestamp(bars.iloc[0]["datetime"]).date()

    bars_seen = 0
    prev_close: float | None = None      # Pine close[1]
    took = False
    saw_cross = False

    level = snap.entry_level
    lvl_ok = not np.isnan(level) and (not cfg.req52 or not np.isnan(snap.level_52))

    for _, bar in bars.iterrows():
        ts = pd.Timestamp(bar["datetime"])
        price = float(bar["close"])
        bars_seen += 1

        if ts.date() != current_day:                       # isNewDay
            current_day = ts.date()
            day_open = float(bar["open"])

        week_volume += float(bar["volume"])

        # Fraction of the trading week elapsed, by 5m bars. NSE runs 75 bars a
        # day, 5 days a week = 375. Used to pro-rate the c09 volume target so a
        # Monday-morning breakout is judged on pace, not on a full week's total.
        week_fraction = min(bars_seen / float(BARS_PER_WEEK), 1.0)
        ev = evaluate_bar(snap, cfg, price, week_open, week_volume, day_open,
                          week_fraction)
        result.last_eval = ev

        if not lvl_ok:
            prev_close = price
            continue

        bar_above = price > level and (not cfg.req52 or price > snap.level_52)
        prev_above = (prev_close is not None and prev_close > level
                      and (not cfg.req52 or prev_close > snap.level_52))

        cross_up = bar_above and not prev_above
        if cross_up:
            saw_cross = True

        gate = gate_ok(snap, cfg, ev) if cfg.strict_entry else True
        trig_cross = cross_up and gate
        trig_defer = (cfg.defer_entry and cfg.strict_entry and saw_cross
                      and not cross_up and bar_above and gate)

        entry = (trig_cross or trig_defer) and (not took if cfg.one_per_week else True)

        # Intraday volatility filter. Like RVOL below, only consulted when a
        # trade is otherwise ready, so it costs nothing on non-signal bars.
        #
        # Mean 5m bar range over the last ATR_LOOKBACK bars (including this
        # one) as a percent of price. A breakout in a stock that barely moves
        # cannot cover round-trip costs, and the measurement bears that out:
        # requiring >= 1.0% turned a losing set (PF 0.72) into PF 1.84, and it
        # was the only filter that survived an out-of-sample split.
        min_atr = float(getattr(cfg, "min_atr_pct", 0.0) or 0.0)
        if entry and min_atr > 0.0:
            # Window is the last ATR_LOOKBACK bars INCLUDING this one. Even a
            # single bar is a usable estimate (it is this candle's own range),
            # so never skip the check for want of history - skipping would let
            # the quietest possible breakout through on the week's first bar.
            lo_i = max(0, bars_seen - ATR_LOOKBACK)
            window = bars.iloc[lo_i:bars_seen]
            if len(window) >= 1 and price > 0:
                atr_pct = float(
                    (window["high"].astype(float)
                     - window["low"].astype(float)).mean()) / price * 100.0
                if atr_pct < min_atr:
                    entry = False

        # Reject a breakout that is already extended far above the level -
        # by then most of the move has happened and the stop must sit far
        # away. Measured: ext < 3% lifted the +5% hit rate 8.3% -> 10.7%.
        max_ext = float(getattr(cfg, "max_ext_above_level", 0.0) or 0.0)
        if entry and max_ext > 0.0 and level > 0:
            if (price - level) / level * 100.0 > max_ext:
                entry = False

        # Relative-volume filter. Only consulted when a trade is otherwise
        # ready, so it costs nothing on the thousands of non-signal bars.
        bar_rvol = float("nan")
        this_bar_rvol = float("nan")
        need_rvol = getattr(cfg, "rvol_mode", "off") != "off"
        bar_rvol_min = float(getattr(cfg, "bar_rvol_min", 0.0) or 0.0)
        if entry and (need_rvol or bar_rvol_min > 0.0):
            hist = history if history is not None else bars
            bar_rvol = compute_rvol(bars, hist, ts,
                                    getattr(cfg, "rvol_lookback_sessions", 20))
            if need_rvol and cfg.rvol_mode == "on" and not np.isnan(bar_rvol) \
                    and bar_rvol < cfg.rvol_min:
                entry = False        # NaN never blocks - unknown is not weak

            # Single-bar relative volume: THIS candle against the median
            # volume of the same clock-minute over the lookback. This is the
            # "something just happened" detector - the surge that precedes an
            # explosive move - where rvol is the "today is a big day" one.
            if entry and bar_rvol_min > 0.0:
                this_bar_rvol = compute_bar_rvol(
                    bars, hist, ts,
                    getattr(cfg, "rvol_lookback_sessions", 20))
                if not np.isnan(this_bar_rvol) and this_bar_rvol < bar_rvol_min:
                    entry = False    # NaN never blocks

        if entry:
            took = True
            result.signals.append(Signal(
                symbol=snap.symbol,
                security_id=snap.security_id,
                exchange_segment=snap.exchange_segment,
                bar_time=ts.to_pydatetime(),
                price=price,
                entry_level=level,
                level_52=snap.level_52,
                trigger="cross" if trig_cross else "deferred",
                evaluation=ev,
                week_start=snap.week_start,
                week_volume=week_volume,
                day_open=day_open,
                week_open=week_open,
                bar_open=float(bar["open"]),
                bar_high=float(bar["high"]),
                bar_low=float(bar["low"]),
                rvol=bar_rvol,
            ))

        prev_close = price

    result.took_this_week = took
    result.saw_cross_this_week = saw_cross
    return result
