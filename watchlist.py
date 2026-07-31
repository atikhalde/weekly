#!/usr/bin/env python3
"""
Daily watchlist - the stocks worth watching for the CURRENT week.

The 26W breakout level and the weekly momentum state are FROZEN when the week
opens (that is what weekly_snapshot.csv holds), so a pre-market digest can be
built from one bulk-quote pass.

WHAT IT REPORTS

    APPROACHING   ranked shortlist: names close to their 26W level that pass
                  the tradeability + volatility gates, ranked by PRE score
    FIRED         names that already triggered, with a BRK confidence tag
    SCREENED      counted, not listed: near the level but failed a hard gate

Everything else is counted, not listed. The full table always goes out as a CSV.

-----------------------------------------------------------------------------
THE HARD GATES (added 30-Jul-2026, measured - see RALLY_FILTERS.md)
-----------------------------------------------------------------------------
A 5-year study of 33,755 point-in-time signals (1,864 symbols) asked which
breakouts turn into large rallies, target P(MFE >= +30% in 30 days).

TRADEABILITY. The strongest raw factors were all illiquidity proxies: the
study's headline "+0.96%/trade" came two-thirds from microcaps that cannot be
filled. Inside a tradeable universe the same edge is +0.28%. A Rs 20 stock
moves one tick = 5%; 0.6 Cr/day of turnover cannot absorb a 1-lakh order at
anything like the assumed 0.22% cost. So price and turnover are now HARD gates,
not ranking inputs:

        turnover_20d >= 2 Cr/day        px >= 30

VOLATILITY. `atr_pct` (ATR14 as a percent of price) was the single most
reliable factor measured, and it survived every out-of-sample split:

        ATR% decile 1  (0.94-2.65%)    P(+30%) =  5.5%
        ATR% decile 5  (3.96-4.30%)    P(+30%) = 19.4%
        ATR% decile 10 (6.22-11.6%)    P(+30%) = 22.4%

A four-fold difference. Note this CONTRADICTS the classic "tight coiled base"
idea: in this data a quiet base predicts a dud. The breakout is a continuation
of existing energy, not a spring releasing. Hence `atr_pct >= 3.0`.

WHAT THESE GATES DO NOT DO. They do not make a losing signal profitable. The
top decile of a walk-forward model still fails to produce a +30% rally 73% of
the time, and selecting for volatility RAISES drawdown (64.8% of top-decile
trades still touch -7%). This is a shortlist, not a prediction.

-----------------------------------------------------------------------------
RANKING - PRE score (0-8), replacing "conditions passed"
-----------------------------------------------------------------------------
The old rank used SCORED_ROWS, the 8 frozen gate rows. Those rows measured
almost nothing: a stock sitting near a 26-week high nearly always HAS good
weekly momentum, which is why it is there. The PRE score uses the factors that
actually separated the rallies, all computable pre-market from daily bars:

    +1  atr_pct        >= 3.5     already moving
    +1  base_tight     >= 4.0     lively 130-day base, not a dead stock
    +1  ret_12m        >= 25      12-month momentum
    +1  dist_50dma     >= 12      extended above the 50DMA, and that is GOOD
    +1  base_depth_pct >= -45     shallow base - not a wreck reclaiming losses
    +1  spike_level    >= 3       level backed by real bars, not one lone wick
    +1  dma200_slope   >= 2       200DMA rising
    +1  px             <= 800     avoid the very high-priced

Out-of-sample, P(+30% in 30d) by PRE score: 1 -> 2.5%, 4 -> 7.4%, 6 -> 13.7%,
7 -> 20.4%, 8 -> 22.8%. Roughly a 9x spread end to end.

`spike_level` is the measured form of the CENTENKA observation: a 26W high that
sits far above the 4th-highest bar in the base is one manipulated print, and it
scores WORSE. (The related idea - that the high bar's upper wick matters -
was tested and FAILED out of sample, so it is deliberately not used.)

-----------------------------------------------------------------------------
BRK score (0-5) - confidence tag on names that ALREADY FIRED
-----------------------------------------------------------------------------
    +1  brk_range_pct >= 4     wide-range breakout day
    +1  ext_pct2      >= 2     closed clear of the level, not scraping it
    +1  brk_rvol      >= 1.8   real volume behind it
    +1  brk_close_pos >= 0.85  closed in the top 15% of the day's range
    +1  gap_pct       >= 1     gapped up

Out-of-sample P(+30%): BRK 0 -> 3.8%, BRK 3 -> 13.4%, BRK 5 -> 15.9%.

SCOPE. This is shown here for names already triggered, because that is the only
place the watchlist can see a completed breakout day. Wiring BRK into the LIVE
5-minute alert is a change to scan.py/telegram.py and has NOT been made.

-----------------------------------------------------------------------------
    python watchlist.py                  # send today's digest
    python watchlist.py --near 3 --top 15
    python watchlist.py --dry-run        # print, do not send
    python watchlist.py --no-gates       # show what the gates removed
    python watchlist.py --no-structure   # rank on distance alone, no metrics

Never places orders.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from config import load_config
from dhan import IST, DhanClient, DhanError
from mcap import load_table as load_mcap_table
from scan import QUOTE_BATCH, load_snapshots
from state import AlertState
from strategy import WeeklySnapshot, week_start_of
from telegram import build_telegram, _esc, _fmt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger("watchlist")

# How many APPROACHING names to print. The complete table is attached as a CSV,
# so this is purely about keeping the message readable on a phone.
TOP_N = 15

# ---- HARD GATES ------------------------------------------------------------
# Measured thresholds. See the module docstring and RALLY_FILTERS.md.
MIN_TURNOVER_CR = 2.0      # 20-day average traded value, in crore per day
MIN_PRICE_GATE = 30.0      # rupees. Separate from strategy.min_price (=100),
                           # which is a Pine gate row (c11) and measured
                           # NEGATIVE for returns; this one is about fillability.
MIN_ATR_PCT = 3.0          # ATR14 as a percent of price

# Daily history needed for the metrics. 200DMA needs 200 sessions, ret_12m
# needs 252, so ~500 calendar days covers both with room for holidays.
METRIC_DAYS = 500
BASE_LOOKBACK = 130        # sessions used for the base-structure factors


# ---------------------------------------------------------------------------
#  Frozen-snapshot eligibility (unchanged)
# ---------------------------------------------------------------------------
def is_eligible(s: WeeklySnapshot, cfg, mcap: float | None = None) -> bool:
    """
    The FROZEN screens that are safe to apply pre-market.

    All use last-closed-week values, so they cannot change mid-week and can be
    evaluated straight from the snapshot with no extra API calls.

        c03  fresh breakout   close[1] <= 26W high two weeks ago
        c08  MACD histogram > 0
        c05  weekly EMA50 rising
        c12  market cap > min_mcap (when use_mcap is on)

    Deliberately NOT included: c04 (EMA20>EMA50), c06 (RSI>60), c07 (RSI
    rising). Each of those drops names that went on to fire this week - c06 at
    the strict 60 threshold alone loses 9 of 24.

    `mcap` is passed in from mcap.csv rather than read off the snapshot: a
    snapshot built before the market-cap feature carries mcap=None for every
    row, which would silently disable the filter here. Unknown still PASSES -
    never hide a name just because its cap could not be resolved.
    """
    if not (s.close_1 <= s.hi_short2
            and s.g_hist > 0.0
            and s.g_ema_slow > s.g_ema_slow_2):
        return False

    # Accept either the full Config or a Strategy. The knobs live on
    # cfg.strategy; reading them off the Config silently returned False and
    # skipped the whole market-cap screen (QMSMEDI, 264 Cr, reached the digest).
    strat = getattr(cfg, "strategy", cfg)
    if getattr(strat, "use_mcap", False):
        if mcap is None:
            return True                       # unknown is not small
        margin = 1.0 - float(getattr(strat, "mcap_margin_pct", 0.0) or 0.0) / 100.0
        if mcap <= strat.min_mcap * margin:
            return False
    return True


# The 8 gate rows that can be scored PRE-MARKET, straight from the snapshot.
#
# RETAINED for the CSV and for --no-structure, but NO LONGER the ranking key.
# Measured on 33,755 signals, these rows barely separate outcomes: a stock near
# a 26-week high almost always has good weekly momentum, so the score is close
# to constant across the shortlist. The PRE score below replaced it as the rank.
SCORED_ROWS = ("c03", "c04", "c05", "c06", "c07", "c08", "c11", "c12")


def score_conditions(s: WeeklySnapshot, cfg, mcap: float | None = None,
                     ltp: float | None = None) -> tuple[int, list[str]]:
    """
    How many of the pre-market-knowable gate rows this name already satisfies.

    Returns (passed, failing_row_names). Max is len(SCORED_ROWS) = 8.
    A row that cannot be judged (unknown market cap) counts as PASSING, for the
    same reason c12 does in the live gate: unknown is not disqualifying.
    """
    strat = getattr(cfg, "strategy", cfg)
    checks = {
        "c03": s.close_1 <= s.hi_short2,
        "c04": s.g_ema_fast > s.g_ema_slow,
        "c05": s.g_ema_slow > s.g_ema_slow_2,
        "c06": s.g_rsi > strat.rsi_min,
        "c07": s.g_rsi > s.g_rsi_1,
        "c08": s.g_hist > 0.0,
        "c11": (ltp is None) or (ltp > strat.min_price),
        "c12": True,
    }
    if getattr(strat, "use_mcap", False) and mcap is not None:
        margin = 1.0 - float(getattr(strat, "mcap_margin_pct", 0.0) or 0.0) / 100.0
        checks["c12"] = mcap > strat.min_mcap * margin
    failed = [k for k in SCORED_ROWS if not checks[k]]
    return len(SCORED_ROWS) - len(failed), failed


# ---------------------------------------------------------------------------
#  Daily-bar metrics
# ---------------------------------------------------------------------------
def compute_metrics(daily: pd.DataFrame, level: float,
                    ltp: float | None = None) -> dict | None:
    """
    The factors the rally study found, from daily candles.

    `daily` must be ascending with columns open/high/low/close/volume. Returns
    None when there is not enough history to judge - the caller treats that as
    UNSCREENED (kept, flagged, ranked last) rather than as a pass or a fail.

    Every value uses only CLOSED daily bars, so this is stable pre-market.
    """
    if daily is None or daily.empty or len(daily) < 60:
        return None
    d = daily.dropna(subset=["open", "high", "low", "close"])
    if len(d) < 60:
        return None

    o = d["open"].to_numpy(float)
    h = d["high"].to_numpy(float)
    lo = d["low"].to_numpy(float)
    c = d["close"].to_numpy(float)
    v = d["volume"].to_numpy(float) if "volume" in d else np.zeros(len(d))

    px = float(ltp) if ltp and ltp > 0 else float(c[-1])
    if px <= 0:
        return None

    # ---- ATR14 as a percent of price
    prev = np.roll(c, 1)
    prev[0] = c[0]
    tr = np.maximum(h - lo, np.maximum(np.abs(h - prev), np.abs(lo - prev)))
    n_atr = min(14, len(tr))
    atr_pct = float(np.mean(tr[-n_atr:]) / px * 100.0)

    # ---- 20-day average traded value, in crore
    n_to = min(20, len(c))
    turnover_cr = float(np.mean(c[-n_to:] * v[-n_to:]) / 1e7)

    # ---- trend / momentum
    def ret_over(nb: int) -> float:
        return float((px / c[-nb] - 1) * 100.0) if len(c) > nb and c[-nb] > 0 else float("nan")

    ret_12m = ret_over(252)
    ret_3m = ret_over(63)
    ret_1m = ret_over(21)

    ma50 = float(np.mean(c[-50:])) if len(c) >= 50 else float("nan")
    ma200 = float(np.mean(c[-200:])) if len(c) >= 200 else float("nan")
    dist_50dma = (px / ma50 - 1) * 100.0 if ma50 and ma50 > 0 else float("nan")
    dist_200dma = (px / ma200 - 1) * 100.0 if ma200 and ma200 > 0 else float("nan")

    dma200_slope = float("nan")
    if len(c) >= 221:
        prev200 = float(np.mean(c[-221:-21]))
        if prev200 > 0:
            dma200_slope = (ma200 / prev200 - 1) * 100.0

    # ---- base structure over the last BASE_LOOKBACK sessions
    seg = slice(max(0, len(c) - BASE_LOOKBACK), len(c))
    sh, sl, sc = h[seg], lo[seg], c[seg]
    base_tight = float(np.mean((sh - sl) / np.maximum(sc, 1e-9)) * 100.0)
    base_depth_pct = float((sl.min() / level - 1) * 100.0) if level > 0 else float("nan")

    # spike_level: how far the 26W level sits above the 4th-highest bar of the
    # base. A big number means the level is one lonely print (CENTENKA's 622 on
    # a 90x-volume spike day) and measured WORSE, not better.
    top = np.sort(sh)[::-1]
    ref = top[min(3, len(top) - 1)]
    spike_level = float((top[0] / ref - 1) * 100.0) if ref > 0 else float("nan")

    return dict(px=px, atr_pct=atr_pct, turnover_cr=turnover_cr,
                ret_12m=ret_12m, ret_3m=ret_3m, ret_1m=ret_1m,
                dist_50dma=dist_50dma, dist_200dma=dist_200dma,
                dma200_slope=dma200_slope, base_tight=base_tight,
                base_depth_pct=base_depth_pct, spike_level=spike_level)


def gate_reasons(m: dict | None, min_turnover: float = MIN_TURNOVER_CR,
                 min_px: float = MIN_PRICE_GATE,
                 min_atr: float = MIN_ATR_PCT) -> list[str]:
    """
    Which hard gates this name FAILS. Empty list == tradeable.

    `m is None` returns [] - unknown is NOT a failure. A transient Dhan error
    must not silently empty the watchlist, and it must not silently promote a
    junk name either, so the caller marks these UNSCREENED and ranks them last.
    """
    if m is None:
        return []
    bad = []
    if not (m.get("turnover_cr") is not None and m["turnover_cr"] >= min_turnover):
        bad.append(f"turnover<{min_turnover:g}Cr")
    if not (m.get("px") is not None and m["px"] >= min_px):
        bad.append(f"px<{min_px:g}")
    atr = m.get("atr_pct")
    if atr is None or (isinstance(atr, float) and math.isnan(atr)) or atr < min_atr:
        bad.append(f"atr<{min_atr:g}%")
    return bad


def score_pre(m: dict | None) -> tuple[int, list[str]]:
    """
    PRE score 0-8: the factors that separated large rallies, all knowable
    before the open. Returns (score, names_of_conditions_passed).

    A NaN input fails its condition rather than raising - a name with too
    little history simply scores lower, which is the right treatment.
    """
    if m is None:
        return 0, []

    def ge(key: str, thr: float) -> bool:
        v = m.get(key)
        if v is None:
            return False
        try:
            v = float(v)
        except (TypeError, ValueError):
            return False
        return not math.isnan(v) and v >= thr

    def le(key: str, thr: float) -> bool:
        v = m.get(key)
        if v is None:
            return False
        try:
            v = float(v)
        except (TypeError, ValueError):
            return False
        return not math.isnan(v) and v <= thr

    checks = {
        "atr>=3.5": ge("atr_pct", 3.5),
        "tight>=4": ge("base_tight", 4.0),
        "ret12m>=25": ge("ret_12m", 25.0),
        "dist50>=12": ge("dist_50dma", 12.0),
        "depth>=-45": ge("base_depth_pct", -45.0),
        "spike>=3": ge("spike_level", 3.0),
        "200slope>=2": ge("dma200_slope", 2.0),
        "px<=800": le("px", 800.0),
    }
    passed = [k for k, ok in checks.items() if ok]
    return len(passed), passed


PRE_MAX = 8
BRK_MAX = 5


def btst_ready(m: dict | None) -> bool:
    """
    Is this name capable of producing the YASHO shape if it breaks out?

    NOT a prediction that it will. It is a capability check: the measured BTST
    tiers require a wide-range close at the high on heavy volume, and a stock
    whose average daily range is 1.5% essentially cannot print a +15% day.

    Measured (5 years, 18,259 tradeable breakouts, entry at the breakout-day
    close, exit next close, net): the BASELINE over all breakouts is +0.01%,
    i.e. nothing. The edge lives entirely in high-ATR names that close hard at
    the high, so flagging the capable ones in advance is the useful signal.
    """
    if not m:
        return False
    try:
        atr = float(m.get("atr_pct") or 0)
        r12 = float(m.get("ret_12m") or 0)
        tight = float(m.get("base_tight") or 0)
    except (TypeError, ValueError):
        return False
    if math.isnan(atr) or math.isnan(tight):
        return False
    return atr >= 4.0 and tight >= 4.0 and (math.isnan(r12) or r12 >= 25.0)


def score_brk(daily: pd.DataFrame | None, level: float) -> tuple[int | None, list[str]]:
    """
    BRK score 0-5 for a name that has ALREADY broken out: how convincing was
    the breakout day itself? Uses the last completed daily bar.

    Returns (None, []) when it cannot be computed, so the caller can omit the
    tag rather than print a misleading zero.
    """
    if daily is None or len(daily) < 51 or not level or level <= 0:
        return None, []
    d = daily.dropna(subset=["open", "high", "low", "close"])
    if len(d) < 51:
        return None, []
    o = float(d["open"].iloc[-1]); h = float(d["high"].iloc[-1])
    lo = float(d["low"].iloc[-1]); c = float(d["close"].iloc[-1])
    pc = float(d["close"].iloc[-2])
    v = float(d["volume"].iloc[-1]) if "volume" in d else 0.0
    vma = float(d["volume"].iloc[-51:-1].mean()) if "volume" in d else 0.0
    rng = h - lo
    if c <= 0:
        return None, []

    checks = {
        "range>=4%": (rng / c * 100.0) >= 4.0,
        "ext>=2%": ((c / level - 1) * 100.0) >= 2.0,
        "rvol>=1.8": vma > 0 and (v / vma) >= 1.8,
        "close@high": rng > 0 and ((c - lo) / rng) >= 0.85,
        "gap>=1%": pc > 0 and ((o / pc - 1) * 100.0) >= 1.0,
    }
    passed = [k for k, ok in checks.items() if ok]
    return len(passed), passed


def fetch_daily_metrics(client: DhanClient, snaps: list[WeeklySnapshot],
                        levels: dict[str, float], ltps: dict[str, float],
                        workers: int = 4, days: int = METRIC_DAYS,
                        want_brk: set[str] | None = None) -> dict[str, dict]:
    """
    Daily candles for the CANDIDATES ONLY, in parallel.

    Deliberately not the whole universe: 2,100 symbols at the configured data
    rate would take ~9 minutes and blow the rate limit, for metrics that only
    matter for the ~60 names actually near their level. Scoped this way the
    extra cost is a few seconds.
    """
    today = datetime.now(IST).date()
    start = today - timedelta(days=days)
    want_brk = want_brk or set()
    out: dict[str, dict] = {}

    def one(s: WeeklySnapshot):
        try:
            df = client.daily_candles(str(s.security_id), s.exchange_segment,
                                      start, today)
        except DhanError as exc:
            log.debug("daily fetch failed for %s: %s", s.symbol, str(exc)[:120])
            return s.symbol, None, None
        lvl = levels.get(s.symbol, 0.0)
        m = compute_metrics(df, lvl, ltps.get(s.symbol))
        b = score_brk(df, lvl) if s.symbol in want_brk else (None, [])
        return s.symbol, m, b

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        for sym, m, b in ex.map(one, snaps):
            rec = dict(m) if m else {}
            if b and b[0] is not None:
                rec["brk_score"] = b[0]
                rec["brk_passed"] = ",".join(b[1])
            rec["_ok"] = m is not None
            out[sym] = rec
    return out


def fetch_ltp(client: DhanClient, snaps: list[WeeklySnapshot]) -> dict[str, float]:
    """symbol -> last traded price, via bulk quotes (1000 per request)."""
    by_seg: dict[str, list[int]] = {}
    ident: dict[tuple[str, str], str] = {}
    for s in snaps:
        by_seg.setdefault(s.exchange_segment, []).append(int(s.security_id))
        ident[(s.exchange_segment, str(s.security_id))] = s.symbol

    out: dict[str, float] = {}
    for seg, ids in by_seg.items():
        for i in range(0, len(ids), QUOTE_BATCH):
            chunk = ids[i:i + QUOTE_BATCH]
            try:
                part = client.ltp({seg: chunk})
            except DhanError as exc:
                msg = str(exc)
                if ("auth failed" in msg or "401" in msg or "403" in msg
                        or "DH-901" in msg or "Authentication" in msg):
                    raise DhanError(
                        "Dhan authentication failed - refresh DHAN_ACCESS_TOKEN"
                    ) from exc
                log.warning("quote batch failed: %s", msg[:140])
                continue
            for s, m in part.items():
                for sid, px in m.items():
                    sym = ident.get((s, str(sid)))
                    if sym and px > 0:
                        out[sym] = float(px)
    return out


def build_message(week: str, rows: list[dict], counts: dict,
                  near_pct: float, top_n: int = TOP_N) -> str:
    fired = [r for r in rows if r["bucket"] == "FIRED"]
    # Rank by PRE score (desc), then distance to the level (asc).
    #
    # The old key was the frozen gate-row count, which barely varied across the
    # shortlist - nearly every name at a 26-week high passes those rows. PRE
    # score uses the factors that measurably separated rallies (out-of-sample
    # P(+30%) runs 2.5% at score 1 to 22.8% at score 8), so the ordering now
    # carries information. UNSCREENED names sort last: score None -> -1.
    watch = sorted((r for r in rows if r["bucket"] == "WATCH"),
                   key=lambda x: (-(x.get("pre_score")
                                    if x.get("pre_score") is not None else -1),
                                  x["gap"]))

    scr = counts.get("screened", 0)
    uns = counts.get("unscreened", 0)
    lines = [
        f"📋 <b>Watchlist — week of {week}</b>",
        f"<i>{datetime.now(IST):%d-%b-%Y %H:%M} IST · "
        f"{counts['universe']} symbols · {counts.get('eligible', 0)} eligible"
        + (f" · {counts['capped']} with mcap" if counts.get("capped") else "")
        + "</i>",
        "",
        f"👀 approaching <b>{len(watch)}</b> · 🔥 fired <b>{len(fired)}</b>"
        + (f" · 🚫 screened out <b>{scr}</b>" if scr else ""),
    ]

    if watch:
        shown = watch[:top_n]
        lines += ["", f"👀 <b>CLOSEST TO BREAKOUT (top {len(shown)})</b>",
                  f"<i>within {near_pct:g}% · turnover ≥{MIN_TURNOVER_CR:g}Cr · "
                  f"ATR ≥{MIN_ATR_PCT:g}% · ranked by PRE score</i>",
                  "<i>🌙 = capable of a BTST-grade move (high ATR, lively, "
                  "trending)</i>"]
        for r in shown:
            tag = r.get("which") or ""
            cap = r.get("mcap_cr")
            sc = r.get("pre_score")
            head = f"<b>{sc}/{PRE_MAX}</b> " if sc is not None else "<b>?</b> "
            extra = f"  <i>[{tag}]</i>" if tag else ""
            if cap:
                extra += f" <i>{cap:,.0f}Cr</i>"
            atr = r.get("atr_pct")
            if atr is not None and not (isinstance(atr, float) and math.isnan(atr)):
                extra += f" <i>atr {atr:.1f}%</i>"
            if r.get("btst_ready"):
                extra += " 🌙"
            if not r.get("screened_ok", True):
                extra += " <i>⚠unscreened</i>"
            lines.append(
                f"• {head}<b>{_esc(r['symbol'])}</b>  {_fmt(r['ltp'])} → "
                f"<code>{_fmt(r['level'])}</code>  "
                f"<b>{r['gap']:.2f}%</b> away{extra}")
        if len(watch) > len(shown):
            lines.append(f"<i>… {len(watch) - len(shown)} more in the CSV</i>")
    else:
        lines += ["", "<i>Nothing approaching its level right now.</i>"]

    if fired:
        lines += ["", f"🔥 <b>ALREADY TRIGGERED ({len(fired)})</b>",
                  "<i>n/5 = breakout-day conviction</i>"]
        for r in sorted(fired, key=lambda x: -(x.get("brk_score") or -1))[:20]:
            b = r.get("brk_score")
            tag = f"<b>{b}/{BRK_MAX}</b> " if b is not None else ""
            lines.append(f"• {tag}{_esc(r['symbol'])}")
        if len(fired) > 20:
            lines.append(f"<i>… +{len(fired) - 20} more in the CSV</i>")

    if scr or uns:
        bits = []
        if scr:
            bits.append(f"{scr} screened out (illiquid / too quiet)")
        if uns:
            bits.append(f"{uns} unscreened (no daily data)")
        lines += ["", f"<i>{' · '.join(bits)}</i>"]

    lines += ["", "<i>Levels are frozen all week. Alerts fire on the first 5m "
                  "close above the 26W high.</i>"]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--near", type=float, default=3.0,
                    help="percent below the level to consider (default 3)")
    ap.add_argument("--top", type=int, default=TOP_N,
                    help="how many names to list (default 15)")
    ap.add_argument("--no-structure", action="store_true",
                    help="rank on distance only, skip the c03/c05/c08 screen")
    ap.add_argument("--no-gates", action="store_true",
                    help="compute metrics but do NOT drop anything - shows what "
                         "the tradeability/volatility gates would remove")
    ap.add_argument("--no-metrics", action="store_true",
                    help="skip the daily-candle pass entirely (no gates, no "
                         "PRE score); falls back to distance ranking")
    ap.add_argument("--min-turnover", type=float, default=MIN_TURNOVER_CR)
    ap.add_argument("--min-px", type=float, default=MIN_PRICE_GATE)
    ap.add_argument("--min-atr", type=float, default=MIN_ATR_PCT)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    week = week_start_of(datetime.now(IST).date())
    week_s = str(week.date())

    snaps = load_snapshots(cfg, week_s)
    if not snaps:
        # load_snapshots already logged why. Exit 0: this runs on a schedule and
        # a missing snapshot is an operational problem, not a crash.
        log.error("no usable snapshot for %s - nothing to report", week_s)
        return 0
    if not cfg.secrets.dhan_access_token:
        log.error("DHAN_ACCESS_TOKEN not set")
        return 0

    client = DhanClient(cfg.secrets.dhan_client_id, cfg.secrets.dhan_access_token,
                        data_rate=cfg.runtime.data_rate_per_sec,
                        quote_rate=cfg.runtime.quote_rate_per_sec)

    log.info("pricing %d symbols ...", len(snaps))
    ltp = fetch_ltp(client, snaps)
    log.info("got %d quotes", len(ltp))

    # Market caps come from mcap.csv, NOT the snapshot: a snapshot built before
    # the c12 feature carries mcap=None for every row and would disable the
    # filter here without any visible symptom.
    #
    # BUG 37: load_snapshots() now applies the same override for the live
    # scanner, so s.mcap is already correct here. This table is still loaded
    # because the digest prints the cap next to each name and needs it for
    # symbols the snapshot never carried.
    caps = load_mcap_table(cfg.paths["mcap"]) if cfg.strategy.use_mcap else {}
    if cfg.strategy.use_mcap and not caps:
        log.warning("use_mcap is ON but %s is empty - the watchlist will not "
                    "apply the market-cap screen. Run `python mcap.py`.",
                    cfg.paths["mcap"])

    state = AlertState(cfg.paths["state"])

    # ---- pass 1: levels, distance, frozen eligibility -----------------------
    prelim: list[dict] = []
    by_symbol: dict[str, WeeklySnapshot] = {}
    eligible = 0
    for s in snaps:
        px = ltp.get(s.symbol)
        if px is None:
            continue

        # Track BOTH levels. Model C triggers on entry_level, Model D on
        # hi_short2, and hi_short2 <= entry_level, so the nearer of the two is
        # the one a breakout will reach first. Reporting only entry_level hid
        # every name that was about to trigger D.
        cap = caps.get(s.symbol.upper())
        lvl_c = s.entry_level
        lvl_d = s.hi_short2
        cand = [x for x in (lvl_c, lvl_d) if x and x > 0]
        if not cand:
            continue
        above = [x for x in cand if px <= x]
        level = min(above) if above else max(cand)
        which = "D" if level == lvl_d and lvl_d != lvl_c else "C"
        if lvl_c == lvl_d:
            which = "C+D"

        ok = args.no_structure or is_eligible(s, cfg, cap)
        eligible += bool(ok)
        score, failing = score_conditions(s, cfg, cap, px)
        pct = (px - level) / level * 100.0

        if state.already_alerted(week_s, s.symbol):
            bucket = "FIRED"
        elif ok and -args.near <= pct <= 0:
            bucket = "WATCH"
        else:
            bucket = "OTHER"

        by_symbol[s.symbol] = s
        prelim.append(dict(symbol=s.symbol, ltp=px, level=level, which=which,
                           level_c=lvl_c, level_d=lvl_d, mcap_cr=cap,
                           level_52=s.level_52, pct=pct, gap=-pct,
                           score=score, max_score=len(SCORED_ROWS),
                           failing=",".join(failing),
                           eligible=ok, bucket=bucket))

    # ---- pass 2: daily metrics, CANDIDATES ONLY -----------------------------
    # Only names that are actionable need the extra API calls: those near the
    # level, plus the fired names (for the BRK tag).
    cand_rows = [r for r in prelim if r["bucket"] in ("WATCH", "FIRED")]
    metrics: dict[str, dict] = {}
    if cand_rows and not args.no_metrics:
        cand_snaps = [by_symbol[r["symbol"]] for r in cand_rows]
        levels = {r["symbol"]: r["level"] for r in cand_rows}
        want_brk = {r["symbol"] for r in cand_rows if r["bucket"] == "FIRED"}
        log.info("fetching daily history for %d candidates ...", len(cand_snaps))
        metrics = fetch_daily_metrics(client, cand_snaps, levels, ltp,
                                      workers=cfg.runtime.max_workers,
                                      want_brk=want_brk)
        got = sum(1 for m in metrics.values() if m.get("_ok"))
        log.info("metrics resolved for %d/%d candidates", got, len(cand_snaps))
        # A partial failure is tolerable; a wholesale one means the gates are
        # not really being applied and that must be loud, not silent.
        if cand_snaps and got < 0.7 * len(cand_snaps):
            log.error("only %d of %d candidates returned daily data - the "
                      "tradeability/volatility gates are NOT fully applied. "
                      "Names without data are marked UNSCREENED and ranked last.",
                      got, len(cand_snaps))

    # ---- pass 3: gates, PRE score, final buckets ----------------------------
    rows: list[dict] = []
    screened = unscreened = 0
    for r in prelim:
        m = metrics.get(r["symbol"])
        have = bool(m and m.get("_ok"))
        mm = {k: v for k, v in (m or {}).items() if not k.startswith("_")}
        r.update({k: v for k, v in mm.items() if k not in ("brk_passed",)})

        if r["bucket"] in ("WATCH", "FIRED") and not args.no_metrics:
            r["screened_ok"] = have
            if not have:
                unscreened += 1

        if r["bucket"] == "WATCH" and not args.no_metrics:
            ps, passed = score_pre(mm if have else None)
            r["pre_score"] = ps if have else None
            r["pre_passed"] = ",".join(passed)
            r["pre_max"] = PRE_MAX
            r["btst_ready"] = btst_ready(mm if have else None)
            bad = gate_reasons(mm if have else None, args.min_turnover,
                               args.min_px, args.min_atr)
            r["gate_failed"] = ",".join(bad)
            if bad and not args.no_gates:
                r["bucket"] = "SCREENED"
                screened += 1
        rows.append(r)

    counts = {"universe": len(snaps), "priced": len(ltp), "eligible": eligible,
              "capped": sum(1 for r in rows if r["mcap_cr"] is not None),
              "screened": screened, "unscreened": unscreened}
    msg = build_message(week_s, rows, counts, args.near, args.top)

    table = pd.DataFrame(rows).sort_values("gap")
    csv_path = Path(args.csv) if args.csv else Path(f"watchlist_{week_s}.csv")
    table.to_csv(csv_path, index=False)
    log.info("wrote %s (%d rows)", csv_path, len(table))

    if args.dry_run:
        print(msg)
        return 0

    tg = build_telegram(cfg)
    tg.send(msg)

    # Attach every actionable name, INCLUDING the screened-out ones so a wrong
    # gate is visible in the data rather than silently deleting candidates.
    actionable = table[table["bucket"].isin(("WATCH", "FIRED", "SCREENED"))]
    if not actionable.empty:
        att = csv_path.with_name(
            f"watchlist_{week_s}_{datetime.now(IST):%d%b}.csv")
        actionable.to_csv(att, index=False)
        tg.send_document(att, caption=f"Watchlist — week of {week_s} "
                                      f"({len(actionable)} names)")
    log.info("sent: %d watch, %d fired, %d screened out, %d unscreened",
             sum(1 for r in rows if r["bucket"] == "WATCH"),
             sum(1 for r in rows if r["bucket"] == "FIRED"),
             screened, unscreened)
    return 0


if __name__ == "__main__":
    sys.exit(main())
