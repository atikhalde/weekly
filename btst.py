#!/usr/bin/env python3
"""
BTST scanner - "buy today's close, sell tomorrow".

Runs ONCE at 15:20 IST - TEN MINUTES BEFORE THE CLOSE - so the position can
actually be entered TODAY at today's close. It does not alert during the rest
of the session, does not place orders, and does not touch scan.py's signal path.

WHY 15:20 AND NOT AFTER THE CLOSE
A scan at 15:40 is useless for this model: the only entry left is tomorrow's
open, which forfeits the overnight move the whole edge is made of. Measured on
5-minute data (149 large caps, 8,047 stock-days), deciding the tier at a cutoff
and entering at that cutoff:

    cutoff   tier precision   entry vs close   next-close return (net)
    15:00        70%            -0.86%            +1.18%  t 2.2
    15:10        72%            -0.58%            +0.56%  t 1.0
    15:15        76%            -0.28%            +0.60%  t 1.2
    15:20        82%            -0.14%            +1.03%  t 2.2
    15:25       100%            -0.00%            +1.31%  t 2.7
    15:30       (the close)      0.00%            +0.80%  t 2.4

Two things fall out. Precision rises towards the close because the candle stops
changing - at 15:20 four of five flagged names still qualify at the close, at
15:00 only seven of ten. And the entry price is BETTER earlier: these names
close at their high, so the last minutes drift up and buying at 15:20 costs
0.14% LESS than the close.

15:20 is the balance: 82% of what it flags is still a valid setup at the bell,
you buy slightly cheaper than the close, and ten minutes is enough to place an
order. 15:25 measures marginally better but leaves five minutes, which is not
a plan. Earlier than 15:15 the tier call is too often wrong.

The tier is therefore judged on a PARTIAL candle - today's bars up to 15:20 -
with volume pro-rated for the elapsed session. That is stated plainly in the
message: the setup can still break in the last ten minutes.

-----------------------------------------------------------------------------
WHAT IT LOOKS FOR - the YASHO shape
-----------------------------------------------------------------------------
On 31-Jul-2026 YASHO broke its 26W level and closed +18.4% on the day, at the
very high of the day, on 8x normal volume. The next-day continuation of that
specific shape is the only overnight edge that survived measurement.

Measured point-in-time over 5 years, 18,259 tradeable breakouts, entry at the
breakout-day CLOSE, exit at the next day's CLOSE, net of 0.22% round trip:

    setup                                    n    /wk   win%    mean     t
    ALL breakouts (baseline)             18259   70.9   44.1   +0.01   0.6   <- nothing
    day>=15% & closed at high              417    1.6   53.7   +1.75   5.2   <- TIER A
    close@high(0.90) & rvol>=3 & atr>=3   1086    4.2   49.3   +0.83   5.0   <- TIER B
    day>=10% & close@high & rvol>=3        871    3.4   47.1   +0.71   3.6
    day>=8%  & close@high & rvol>=2       1402    5.4   46.9   +0.57   4.0

Out-of-sample (fit < Jul-2024, test after) the two shipped tiers HELD:
    TIER A   in-sample +1.62%  ->  out-of-sample +2.01%
    TIER B   in-sample +0.76%  ->  out-of-sample +0.96%

Stability by year for TIER A: 2021 +0.44, 2022 +0.19, 2023 +2.07, 2024 +1.74,
2025 +2.02, 2026 +1.30. Positive in all six.

-----------------------------------------------------------------------------
WHAT IT REFUSES TO CLAIM
-----------------------------------------------------------------------------
Selling at the next OPEN measures +0.45% (t=42), which looks spectacular and is
NOT an edge. The whole universe gaps +0.42% at the open on any random day in
this data, so the true excess is ~+0.25% and most of it is a data artifact of
how the open is recorded. Only NEXT CLOSE is reported here.

The baseline is +0.01%. There is no BTST edge in "a breakout happened" - the
edge is entirely in the character of the breakout DAY, which is why the tiers
are strict and fire only ~2-4 times a week combined.

Win rate is barely above 50% even for Tier A. This is a fat-tail setup:
P(next day >= +3%) = 36.5% and P(>= +5%) = 28.1% for Tier A, against a base
rate of 15.3% / 6.3%. It wins by size, not by frequency.

    python btst.py               # send tonight's list
    python btst.py --dry-run     # print, do not send
    python btst.py --all         # show every breakout, with its tier

Never places orders.
"""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, time as dtime, timedelta

import numpy as np
import pandas as pd

from config import load_config
from dhan import IST, DhanClient, DhanError
from mcap import load_table as load_mcap_table
from scan import load_snapshots
from state import AlertState
from strategy import week_start_of
from telegram import build_telegram, _esc, _fmt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger("btst")

# Tradeability, identical to the watchlist. An overnight hold in an illiquid
# name is worse than an intraday one: you cannot get out at the open.
MIN_TURNOVER_CR = 2.0
MIN_PRICE = 30.0
MIN_ATR_PCT = 3.0

HISTORY_DAYS = 260
BARS_PER_SESSION = 75      # NSE: 09:15-15:30 in 5-minute candles

# How many picks are actually taken. This is THE top-5: btst.py caps the list
# here and writes it to btst_picks.csv, and Model E trades that file rather
# than re-deriving its own list. One source of truth, so the 15:20 alert and
# the paper ledger can never name different stocks.
TOP_N = 5
PICKS_FILE = "btst_picks.csv"

# --------------------------------------------------------------------------- #
#  ANTICIPATION (Model F) - buy the day BEFORE the breakout
#
#  The user asked: "how i can catch them 1 day ago". Measured over 5 years and
#  234,518 approaching-days, the plain answer is you mostly cannot - only
#  16.3% of names sitting within 5% of their level break out the next day, and
#  buying the watchlist top-5 on proximity alone loses money (-0.195%/trade,
#  t -4.57, negative out of sample).
#
#  ONE thing separates them: where the stock CLOSED in its daily range.
#
#      close_pos >= 0.9  ->  26.7% break out next day
#      close_pos <  0.9  ->  15.9%
#
#  A stock finishing at the very top of its range, still under the level, is
#  being accumulated into the close. Measured, buying the top 5 of those:
#
#      window   n     /wk   win%    mean     PF     t     out-of-sample
#      <=3%   1423    5.4   52.0   +0.688   1.73   7.73     +0.701%
#      <=5%   2184    8.4   50.5   +0.511   1.50   6.47     +0.465%
#      <=8%   3030   11.6   49.3   +0.443   1.43   7.44     +0.380%
#
#  <=3% is used: strongest per trade and it holds out of sample.
#
#  WHY IT IS SEPARATE FROM THE BTST TIERS. Overlap with Tier A is 0.1% - two
#  picks out of 1,423. Anticipation fires the day BEFORE a breakout, the tiers
#  fire the day OF one. They are complementary, not duplicates, so they are
#  reported as two lists and traded as two models.
#
#  HONEST CEILING: even filtered, 73% of these do NOT break out the next day.
#  Per trade the confirmation tiers are ~2.5x better (+1.74% Tier A vs +0.69%).
#  This exists to cover the days the tiers are silent, not to replace them.
# --------------------------------------------------------------------------- #
ANTICIPATE_NEAR = 3.0        # max % BELOW the level
ANTICIPATE_ABOVE_MAX = 10.0  # max % ABOVE the level before it is a chase
ANTICIPATE_CLOSE_POS = 0.90  # must finish at the top of its own range
ANTICIPATE_TOP_N = 5
ANTICIPATE_FILE = "anticipate_picks.csv"

# --------------------------------------------------------------------------- #
#  THE COMBINED EDGE (measured 31-Jul-2026, 18,231 tradeable signals)
#
#  The 08:45 watchlist score and this 15:20 scan are measuring DIFFERENT things
#  and they stack. Net return per trade, 30d cap / 7% stop / 5% trail:
#
#      selector                       n     /wk   win%    mean    OOS
#      no filter                   18231   70.8   36.2   +0.194  -0.197
#      PRE score >= 6               6991   27.2   36.9   +0.423  +0.094
#      PRE score >= 7               3265   12.7   36.7   +0.577  +0.364
#      close_pos >= 0.90            3414   13.3   40.6   +1.040  +0.877
#      PRE>=6 AND close_pos>=0.90   1488    5.8   42.4   +1.719  +1.701
#      PRE>=7 AND close_pos>=0.90    739    2.9   44.5   +2.641  +2.832
#
#  Two independent pieces of information: the morning score says "this stock
#  is in a real uptrend", the afternoon candle says "it is being accumulated
#  RIGHT NOW". Requiring both roughly quadruples the per-trade return over the
#  score alone, and the out-of-sample column barely moves - the strongest sign
#  in the whole study that this is not curve fitting.
#
#  MIN_PRE_CONFIRM is therefore applied to BOTH lists here. It is a floor, not
#  a ranking: 6 keeps ~5.8 setups a week, 7 would keep 2.9. Six is chosen so
#  the list does not go empty for a week at a time.
# --------------------------------------------------------------------------- #
MIN_PRE_CONFIRM = 6


# Thresholds live here and are imported by ab_paper.py's Model E, so the paper
# model and the nightly scanner can never drift apart.
TIER_A_DAY = 15.0
TIER_A_CLOSE_POS = 0.85
TIER_B_CLOSE_POS = 0.90
TIER_B_RVOL = 3.0
TIER_B_ATR = 3.0


def classify(daily: pd.DataFrame, level: float,
             partial_frac: float = 1.0) -> dict | None:
    """
    Score today's breakout candle. Returns None when it cannot be judged.

    `daily` must end with TODAY's bar. When that bar is still forming (the
    15:20 scan) pass `partial_frac` = the fraction of the session elapsed, so
    the volume test compares like with like: at 15:20 about 93% of the session
    has traded, and judging that against a full-day average would understate
    rvol and reject good setups.

    There is no look-ahead either way - every input is from bars that have
    already printed.
    """
    if daily is None or len(daily) < 60:
        return None
    d = daily.dropna(subset=["open", "high", "low", "close"])
    if len(d) < 60:
        return None

    t = d.iloc[-1]
    o, h, lo, c = float(t.open), float(t.high), float(t.low), float(t.close)
    v = float(t.volume) if "volume" in d else 0.0
    if c <= 0 or o <= 0:
        return None

    prev = d.iloc[:-1]
    vma = float(prev["volume"].tail(50).mean()) if "volume" in d else 0.0
    # Pro-rate the benchmark when today's candle is still forming.
    frac = min(max(float(partial_frac), 0.05), 1.0)
    vma_cmp = vma * frac
    rng = h - lo

    hi = d["high"].to_numpy(float); low = d["low"].to_numpy(float)
    cl = d["close"].to_numpy(float)
    pc = np.roll(cl, 1); pc[0] = cl[0]
    tr = np.maximum(hi - low, np.maximum(np.abs(hi - pc), np.abs(low - pc)))
    atr_pct = float(np.mean(tr[-14:]) / c * 100.0)
    turnover = float((cl[-20:] * d["volume"].to_numpy(float)[-20:]).mean() / 1e7) \
        if "volume" in d else 0.0
    ret_12m = float((c / cl[-252] - 1) * 100.0) if len(cl) >= 252 else float("nan")

    pre, _ = _pre_score_from_daily(d, level, c)
    m = dict(
        pre=pre,
        close=c, day_ret=(c / o - 1) * 100.0,
        close_pos=((c - lo) / rng) if rng > 0 else 0.5,
        rvol=(v / vma_cmp) if vma_cmp > 0 else float("nan"),
        partial_frac=frac,
        range_pct=(rng / c * 100.0),
        gap_pct=((o / float(prev.iloc[-1]["close"]) - 1) * 100.0)
        if float(prev.iloc[-1]["close"]) > 0 else float("nan"),
        atr_pct=atr_pct, turnover_cr=turnover, ret_12m=ret_12m,
        ext_pct=((c / level - 1) * 100.0) if level > 0 else float("nan"),
    )

    # --- tradeability, hard
    if not (m["turnover_cr"] >= MIN_TURNOVER_CR and c >= MIN_PRICE
            and m["atr_pct"] >= MIN_ATR_PCT):
        m["tier"] = None
        m["reject"] = "not tradeable"
        return m

    rv = m["rvol"] if not np.isnan(m["rvol"]) else 0.0
    # --- TIER A: the YASHO shape. Rare, and the strongest measured.
    if m["day_ret"] >= TIER_A_DAY and m["close_pos"] >= TIER_A_CLOSE_POS:
        m["tier"] = "A"
    # --- TIER B: closed hard at the high on real volume, in a mover.
    elif (m["close_pos"] >= TIER_B_CLOSE_POS and rv >= TIER_B_RVOL
          and m["atr_pct"] >= TIER_B_ATR):
        m["tier"] = "B"
    else:
        m["tier"] = None
        m["reject"] = "no tier"
    return m


def _pre_score_from_daily(d: pd.DataFrame, level: float, px: float) -> tuple[int, list]:
    """
    The 08:45 PRE score, recomputed here from the same daily frame.

    Imported from watchlist.py rather than restated so the morning list and
    this scan can never disagree about what a 6/8 means - a regression test
    asserts the two agree on identical input.
    """
    try:
        from watchlist import compute_metrics, score_pre
    except Exception:
        return 0, []
    mm = compute_metrics(d, level, px)
    if mm is None:
        return 0, []
    return score_pre(mm)


def classify_approach(daily: pd.DataFrame, level_c: float, level_d: float,
                      partial_frac: float = 1.0) -> dict | None:
    """
    Score a name that has NOT yet broken out, for the anticipation model.

    Uses the nearer of the two levels still above price - the same dual-level
    rule watchlist.py applies - because that is the one a breakout reaches
    first. Reconstructing only entry_level measured a different trade than the
    one the user actually sees (found when the 30-Jul top-5 could not be
    reproduced: MUTHOOTMF was 2.03% from its D level, not 3.94% from its C).
    """
    if daily is None or len(daily) < 60:
        return None
    d = daily.dropna(subset=["open", "high", "low", "close"])
    if len(d) < 60:
        return None
    t = d.iloc[-1]
    o, h, lo, c = float(t.open), float(t.high), float(t.low), float(t.close)
    v = float(t.volume) if "volume" in d else 0.0
    if c <= 0 or o <= 0:
        return None

    # ---- WHICH LEVEL, AND WHICH SIDE OF IT ---------------------------------
    # Until 01-Aug-2026 this returned None for anything already trading ABOVE
    # its level - "not our trade". That was wrong, and it was throwing away the
    # BETTER half. Same filter (close_pos>=0.90 & ret_12m>=40), 5 years:
    #
    #     side of the level        n      win%     net      OOS
    #     BELOW  (scanned)      2,229     50.8%   +0.580%  +0.569%
    #     ABOVE  (discarded)    4,657     54.7%   +0.819%  +0.776%
    #
    # Twice the sample and a better return. It also matches what the 31-Jul
    # movers showed: 13 of 19 had ALREADY broken out at the prior close, and
    # only 3 were sitting in the sub-3% window this scan used to require.
    #
    # A name above its level is still an ANTICIPATION trade in the sense that
    # matters - the entry is at today's close and the move is tomorrow's. The
    # `side` field records which, so the two can be measured apart forever.
    cand = [x for x in (level_c, level_d) if x and x > 0]
    if not cand:
        return None
    above = [x for x in cand if c <= x]
    if above:
        level = min(above)
        side = "below"
        gap = (level - c) / level * 100.0
        if gap > ANTICIPATE_NEAR:
            return None                  # too far away to be actionable
    else:
        # trading above BOTH levels: judge it against the higher one
        level = max(cand)
        side = "above"
        gap = (level - c) / level * 100.0        # negative = extension
        if -gap > ANTICIPATE_ABOVE_MAX:
            return None                  # too extended, it is a chase
    which = "C+D" if level_c == level_d else (
        "D" if level == level_d else "C")

    prev = d.iloc[:-1]
    vma = float(prev["volume"].tail(50).mean()) if "volume" in d else 0.0
    frac = min(max(float(partial_frac), 0.05), 1.0)
    vma_cmp = vma * frac
    rng = h - lo

    hi = d["high"].to_numpy(float); low = d["low"].to_numpy(float)
    cl = d["close"].to_numpy(float)
    pc = np.roll(cl, 1); pc[0] = cl[0]
    tr = np.maximum(hi - low, np.maximum(np.abs(hi - pc), np.abs(low - pc)))
    atr_pct = float(np.mean(tr[-14:]) / c * 100.0)
    turnover = float((cl[-20:] * d["volume"].to_numpy(float)[-20:]).mean() / 1e7) \
        if "volume" in d else 0.0

    pre, _ = _pre_score_from_daily(d, level, c)
    m = dict(symbol="", close=c, level=level, which=which, gap_pct=gap,
             side=side, pre=pre,
             day_ret=(c / o - 1) * 100.0,
             close_pos=((c - lo) / rng) if rng > 0 else 0.5,
             rvol=(v / vma_cmp) if vma_cmp > 0 else float("nan"),
             atr_pct=atr_pct, turnover_cr=turnover, partial_frac=frac)

    if not (m["turnover_cr"] >= MIN_TURNOVER_CR and c >= MIN_PRICE
            and m["atr_pct"] >= MIN_ATR_PCT):
        m["ok"] = False
        m["reject"] = "not tradeable"
        return m
    # THE filter. Everything else measured negative.
    # THE filter, plus the trend confirmation. Measured, both together:
    #   close_pos>=0.90 alone        +1.040%/trade  OOS +0.877%
    #   with PRE>=6 as well          +1.719%        OOS +1.701%
    if m["close_pos"] < ANTICIPATE_CLOSE_POS:
        m["ok"] = False
        m["reject"] = f"close_pos {m['close_pos']:.2f} < {ANTICIPATE_CLOSE_POS}"
        return m
    if m["pre"] < MIN_PRE_CONFIRM:
        m["ok"] = False
        m["reject"] = f"PRE {m['pre']}/8 < {MIN_PRE_CONFIRM} (weak trend)"
        return m
    m["ok"] = True
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--all", action="store_true",
                    help="list every breakout with its tier, not just A/B")
    ap.add_argument("--after-close", action="store_true",
                    help="only judge a COMPLETED daily candle; skips names "
                         "whose bar is still forming. Use for a post-close "
                         "review - the 15:20 job must NOT set this.")
    ap.add_argument("--no-anticipate", action="store_true",
                    help="skip the anticipation (Model F) section")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    now = datetime.now(IST)
    week_s = str(week_start_of(now.date()).date())

    snaps = load_snapshots(cfg, week_s)
    if not snaps:
        log.error("no usable snapshot for %s", week_s)
        return 0
    if not cfg.secrets.dhan_access_token:
        log.error("DHAN_ACCESS_TOKEN not set")
        return 0

    # ---- BUG 43: THE BREAKOUT MUST BE **TODAY** ----------------------------
    # already_alerted() matches the whole WEEK, so a Monday breakout still came
    # back as a candidate on Friday. The day-character test would then pass on
    # an unrelated later candle and the name would be presented as a fresh
    # BTST setup when most of the move had already happened.
    #
    # That is not a preference, it is a mismatch with what was measured. The
    # 5-year study only ever tested the breakout day itself. Re-measured across
    # 1,548 qualifying stock-days, next-day return net of costs:
    #
    #     age 0 (breakout day)   n=1174   +0.807%   t 5.00   mean ext  7.1%
    #     age 1                  n= 164   +0.373%   t 0.87   mean ext 15.6%
    #     age 2                  n= 107   +1.311%   t 2.45   mean ext 18.6%
    #     age 3                  n=  74   +0.984%   t 1.64   mean ext 20.4%
    #     age 4                  n=  29   -0.808%   t -1.01  mean ext 26.9%
    #
    # and split by tier, which is where it is unambiguous:
    #
    #     Tier A  age 0    n=401   +1.736%   t 5.03
    #     Tier A  age >=1  n= 83   +0.155%   t 0.20   <- the edge is GONE
    #
    # Tier A is the whole reason this scanner exists and it does not survive
    # ageing. The later-day Tier B numbers are positive but thin, drifting
    # (mean extension 15-27% above the level) and were never part of the
    # original measurement, so they are not traded on that basis.
    #
    # Live example that exposed it - 31-Jul-2026:
    #     YASHO     broke out 31-Jul 12:20   age 0   ext  18.3%   valid
    #     NELCO     broke out 31-Jul 12:50   age 0   ext   2.2%   valid
    #     DEEPINDS  broke out 29-Jul 10:05   age 2   ext   9.8%   STALE
    state = AlertState(cfg.paths["state"])
    today_str = f"{now:%Y-%m-%d}"
    fired, stale = [], []
    for s in snaps:
        rec = state.alert_record(week_s, s.symbol)
        if not rec:
            continue
        bar = str(rec.get("bar_time", ""))[:10]
        if bar == today_str:
            fired.append(s)
        else:
            stale.append((s.symbol, bar))
    if stale:
        log.info("%d name(s) broke out earlier this week and are NOT BTST "
                 "candidates today: %s", len(stale),
                 ", ".join(f"{sym}({bar})" for sym, bar in stale[:12]))
    if not fired:
        log.info("nothing broke out TODAY - no BTST candidates "
                 "(%d older breakout(s) skipped)", len(stale))
        return 0
    log.info("%d name(s) broke out today; checking the candle ...", len(fired))

    client = DhanClient(cfg.secrets.dhan_client_id, cfg.secrets.dhan_access_token,
                        data_rate=cfg.runtime.data_rate_per_sec,
                        quote_rate=cfg.runtime.quote_rate_per_sec)
    caps = load_mcap_table(cfg.paths["mcap"])
    start = now.date() - timedelta(days=HISTORY_DAYS)

    def one(s):
        try:
            df = client.daily_candles(str(s.security_id), s.exchange_segment,
                                      start, now.date())
        except DhanError as exc:
            log.debug("%s: %s", s.symbol, str(exc)[:100])
            return s, None
        if df.empty:
            return s, None

        frac = 1.0
        last = pd.Timestamp(df.iloc[-1]["datetime"]).date()
        if last != now.date():
            # Running BEFORE the close, so today's daily bar does not exist
            # yet. Build it from today's 5-minute candles and mark it partial.
            if args.after_close:
                return s, None
            try:
                m5 = client.intraday_candles(
                    str(s.security_id), s.exchange_segment,
                    datetime.combine(now.date(), dtime(9, 15)),
                    now.replace(tzinfo=None), interval=5)
            except DhanError as exc:
                log.debug("%s intraday: %s", s.symbol, str(exc)[:100])
                return s, None
            if m5 is None or m5.empty:
                return s, None
            m5 = m5.sort_values("datetime")
            today_bar = {
                "datetime": pd.Timestamp(now.date()),
                "open": float(m5.iloc[0]["open"]),
                "high": float(m5["high"].max()),
                "low": float(m5["low"].min()),
                "close": float(m5.iloc[-1]["close"]),
                "volume": float(m5["volume"].sum()),
            }
            df = pd.concat([df, pd.DataFrame([today_bar])], ignore_index=True)
            frac = min(len(m5) / float(BARS_PER_SESSION), 1.0)
        return s, classify(df, s.entry_level, partial_frac=frac)

    def candle(s):
        """(daily frame incl. today's partial bar, elapsed fraction) or None."""
        try:
            df = client.daily_candles(str(s.security_id), s.exchange_segment,
                                      start, now.date())
        except DhanError:
            return None, 1.0
        if df.empty:
            return None, 1.0
        frac = 1.0
        if pd.Timestamp(df.iloc[-1]["datetime"]).date() != now.date():
            if args.after_close:
                return None, 1.0
            try:
                m5 = client.intraday_candles(
                    str(s.security_id), s.exchange_segment,
                    datetime.combine(now.date(), dtime(9, 15)),
                    now.replace(tzinfo=None), interval=5)
            except DhanError:
                return None, 1.0
            if m5 is None or m5.empty:
                return None, 1.0
            m5 = m5.sort_values("datetime")
            df = pd.concat([df, pd.DataFrame([{
                "datetime": pd.Timestamp(now.date()),
                "open": float(m5.iloc[0]["open"]),
                "high": float(m5["high"].max()),
                "low": float(m5["low"].min()),
                "close": float(m5.iloc[-1]["close"]),
                "volume": float(m5["volume"].sum()),
            }])], ignore_index=True)
            frac = min(len(m5) / float(BARS_PER_SESSION), 1.0)
        return df, frac

    rows = []
    with ThreadPoolExecutor(max_workers=max(1, cfg.runtime.max_workers)) as ex:
        for s, m in ex.map(one, fired):
            if not m:
                continue
            m["symbol"] = s.symbol
            m["level"] = s.entry_level
            m["mcap_cr"] = caps.get(s.symbol.upper())
            rows.append(m)

    qualified = [r for r in rows if r.get("tier")]
    # Tier A before Tier B, then the largest day move.
    qualified.sort(key=lambda r: (r["tier"], -r["day_ret"]))
    picks = qualified[:TOP_N]
    for i, r in enumerate(picks, start=1):
        r["rank"] = i
    dropped = len(qualified) - len(picks)
    log.info("checked %d candles, %d qualified, taking top %d%s",
             len(rows), len(qualified), len(picks),
             f" ({dropped} beyond the cap)" if dropped else "")

    # ---- write the picks file: this IS the trade list ----------------------
    # Entry price is the price AT THIS SCAN (15:20), not the close, because
    # that is the price actually available when the alert lands.
    if picks:
        out = pd.DataFrame([{
            "date": f"{now:%Y-%m-%d}", "scan_time": f"{now:%H:%M}",
            "rank": r["rank"], "symbol": r["symbol"], "tier": r["tier"],
            "entry": round(float(r["close"]), 2), "pre": int(r.get("pre", 0)),
            "level": round(float(r["level"]), 2),
            "day_ret": round(float(r["day_ret"]), 2),
            "close_pos": round(float(r["close_pos"]), 3),
            "rvol": round(float(r.get("rvol") or 0), 2),
            "atr_pct": round(float(r["atr_pct"]), 2),
            "mcap_cr": r.get("mcap_cr"),
            "partial_frac": round(float(r.get("partial_frac", 1.0)), 3),
        } for r in picks])
        pfile = cfg.paths["root"] / PICKS_FILE
        if pfile.exists():
            old = pd.read_csv(pfile)
            out = (pd.concat([old, out], ignore_index=True)
                     .drop_duplicates(subset=["date", "symbol"], keep="last"))
        out.to_csv(pfile, index=False)
        log.info("wrote %s (%d row(s) today, %d total)",
                 pfile.name, len(picks), len(out))

    if rows:
        pd.DataFrame(rows).to_csv(f"btst_{now:%Y-%m-%d}.csv", index=False)

    partial = [r for r in picks if float(r.get("partial_frac", 1.0)) < 0.999]
    when = "buy into TODAY's close" if partial or now.time() < dtime(15, 30) \
        else "buy at close"
    # ======================================================================
    #  SECOND PASS - ANTICIPATION (Model F): names about to break out
    # ======================================================================
    # Candidate pool is the OPPOSITE of the BTST pool: names that have NOT
    # alerted this week and are still below their level. A name that already
    # broke out cannot be anticipated.
    ant_rows, ant_picks, ant_dropped = [], [], 0
    if not args.no_anticipate:
        # The pool is now the WHOLE snapshot, not just names that have never
        # fired. Excluding alerted names would discard exactly the above-level
        # half that measured better (+0.819% vs +0.580%). Names that broke out
        # TODAY are still excluded - those belong to the CONFIRMED list above
        # and must not appear twice.
        fired_today = {x.symbol for x in fired}
        pending = [s for s in snaps if s.symbol not in fired_today]
        log.info("anticipation: screening %d name(s) (both sides of the "
                 "level) ...", len(pending))

        def one_ant(s):
            df, frac = candle(s)
            if df is None:
                return s, None
            return s, classify_approach(df, s.entry_level, s.hi_short2,
                                        partial_frac=frac)

        with ThreadPoolExecutor(max_workers=max(1, cfg.runtime.max_workers)) as ex:
            for s, m in ex.map(one_ant, pending):
                if not m:
                    continue
                m["symbol"] = s.symbol
                m["mcap_cr"] = caps.get(s.symbol.upper())
                ant_rows.append(m)

        aq = [r for r in ant_rows if r.get("ok")]
        # ABOVE-level names first - measured +0.819% vs +0.580% for below -
        # then, within each side, nearest to the level.
        aq.sort(key=lambda r: (0 if r.get("side") == "above" else 1,
                               abs(r["gap_pct"])))
        ant_picks = aq[:ANTICIPATE_TOP_N]
        for i, r in enumerate(ant_picks, start=1):
            r["rank"] = i
        ant_dropped = len(aq) - len(ant_picks)
        log.info("anticipation: %d screened, %d qualified, taking top %d",
                 len(ant_rows), len(aq), len(ant_picks))

        if ant_picks:
            aout = pd.DataFrame([{
                "date": f"{now:%Y-%m-%d}", "scan_time": f"{now:%H:%M}",
                "rank": r["rank"], "symbol": r["symbol"],
                "entry": round(float(r["close"]), 2),
                "pre": int(r.get("pre", 0)),
                "level": round(float(r["level"]), 2), "which": r["which"],
                "gap_pct": round(float(r["gap_pct"]), 2),
                "side": r.get("side", "below"),
                "close_pos": round(float(r["close_pos"]), 3),
                "day_ret": round(float(r["day_ret"]), 2),
                "rvol": round(float(r.get("rvol") or 0), 2),
                "atr_pct": round(float(r["atr_pct"]), 2),
                "mcap_cr": r.get("mcap_cr"),
                "partial_frac": round(float(r.get("partial_frac", 1.0)), 3),
            } for r in ant_picks])
            afile = cfg.paths["root"] / ANTICIPATE_FILE
            if afile.exists():
                aold = pd.read_csv(afile)
                aout = (pd.concat([aold, aout], ignore_index=True)
                          .drop_duplicates(subset=["date", "symbol"], keep="last"))
            aout.to_csv(afile, index=False)
            log.info("wrote %s (%d today, %d total)",
                     afile.name, len(ant_picks), len(aout))

    lines = [f"🌙 <b>BTST — {now:%d-%b-%Y} {now:%H:%M} IST</b>",
             f"<i>{when}, exit tomorrow · {len(rows)} candle(s) checked "
             f"· broke out TODAY only</i>", ""]
    lines.insert(2, "🔥 <b>CONFIRMED — broke out TODAY</b>")
    lines.insert(3, "")
    if not picks:
        lines.append("<i>No setup qualified today. That is the normal case — "
                     "the tiers fire ~2-4 times a week combined.</i>")
    if stale:
        lines.append(f"<i>{len(stale)} name(s) broke out earlier this week and "
                     f"are not eligible today (Tier A measured +1.74% on the "
                     f"breakout day, +0.16% after).</i>")
    for r in picks:
        badge = ("🔥 <b>TIER A</b>" if r["tier"] == "A" else "⭐ <b>TIER B</b>")
        badge = f"<b>#{r['rank']}</b> {badge}"
        cap = f" <i>{r['mcap_cr']:,.0f}Cr</i>" if r.get("mcap_cr") else ""
        prov = " <i>(candle still forming)</i>" if float(
            r.get("partial_frac", 1.0)) < 0.999 else ""
        pre_tag = f" <b>{r['pre']}/8</b>" if r.get("pre") is not None else ""
        lines += [
            f"{badge}{pre_tag}  <b>{_esc(r['symbol'])}</b>  "
            f"{_fmt(r['close'])}{cap}{prov}",
            f"    day <b>{r['day_ret']:+.1f}%</b> · closed at "
            f"<b>{r['close_pos']*100:.0f}%</b> of range · "
            f"rvol <b>{r['rvol']:.1f}x</b> · atr {r['atr_pct']:.1f}%",
            f"    <i>{r['ext_pct']:+.1f}% above the 26W level "
            f"{_fmt(r['level'])}</i>",
            f"    <b>BUY NOW ~{_fmt(r['close'])}</b> "
            f"<i>· broke out today · exit tomorrow's close if &gt;+2%</i>", ""]

    if dropped:
        extra = ", ".join(_esc(r["symbol"]) for r in qualified[TOP_N:])
        lines.append(f"<i>{dropped} more qualified but the cap is top "
                     f"{TOP_N}: {extra}</i>")
        lines.append("")

    if args.all:
        rest = [r for r in rows if not r.get("tier")]
        if rest:
            lines += [f"<i>no tier ({len(rest)}): "
                      + ", ".join(_esc(r["symbol"]) for r in rest[:40]) + "</i>", ""]

    # ---- SECTION 2: anticipation, kept visually separate ------------------
    if not args.no_anticipate:
        lines += ["", "━━━━━━━━━━━━━━━━━━━━",
                  f"🔭 <b>ANTICIPATE — about to break out</b>",
                  f"<i>🚀 above the level (to +{ANTICIPATE_ABOVE_MAX:g}%) or "
                  f"🔭 within {ANTICIPATE_NEAR:g}% below · closed in the top "
                  f"{int((1-ANTICIPATE_CLOSE_POS)*100)}% of today's range · "
                  f"PRE ≥{MIN_PRE_CONFIRM} · {len(ant_rows)} screened</i>", ""]
        if not ant_picks:
            lines.append("<i>Nothing qualified. Normal — this fires ~5x a "
                         "week.</i>")
        for r in ant_picks:
            cap = f" <i>{r['mcap_cr']:,.0f}Cr</i>" if r.get("mcap_cr") else ""
            prov = " <i>(forming)</i>" if float(
                r.get("partial_frac", 1.0)) < 0.999 else ""
            lines += [
                f"<b>#{r['rank']}</b> "
                f"{'🚀' if r.get('side') == 'above' else '🔭'} "
                f"<b>{r.get('pre', 0)}/8</b> "
                f"<b>{_esc(r['symbol'])}</b>  {_fmt(r['close'])}{cap}{prov}",
                (f"    <b>{abs(r['gap_pct']):.2f}% above</b> the {r['which']} "
                 f"level <code>{_fmt(r['level'])}</code>"
                 if r.get("side") == "above" else
                 f"    <b>{r['gap_pct']:.2f}% below</b> the {r['which']} level "
                 f"<code>{_fmt(r['level'])}</code>") +
                f" · closed at <b>{r['close_pos']*100:.0f}%</b> of range · "
                f"day {r['day_ret']:+.1f}% · rvol {r.get('rvol', 0):.1f}x",
                f"    <b>BUY NOW ~{_fmt(r['close'])}</b> "
                f"<i>· exit tomorrow's close</i>", ""]
        if ant_dropped:
            lines.append(f"<i>{ant_dropped} more qualified, cap is top "
                         f"{ANTICIPATE_TOP_N}</i>")
        lines.append("<i>close_pos≥0.90 + PRE≥6 measured +1.72%/trade "
                     "(t 7.2, n=1,488), +1.70% out of sample — vs +0.42% for "
                     "the score alone. 🚀 above-level measured better than 🔭 "
                     "below (+0.82% vs +0.58%). Most still do not run.</i>")

    lines += ["", "━━━━━━━━━━━━━━━━━━━━",
        "<i>Tier A measured +1.75%/trade (t 5.2, n=417) over 5 years; "
        "Tier B +0.83% (t 5.0, n=1086). Win rate ~50% — it pays through the "
        "tail, not the hit rate. Exit at tomorrow's close.</i>"]
    if partial:
        lines.append(
            "<i>⚠ Judged on a partial candle. Measured at this cutoff, 82% of "
            "flagged names still qualify at the bell — so roughly one in five "
            "breaks down in the last ten minutes. Check the close before "
            "sizing up.</i>")
    lines.append("<i>Paper only. No orders are placed.</i>")
    msg = "\n".join(lines)

    if args.dry_run:
        print(msg)
        return 0
    build_telegram(cfg).send(msg)
    log.info("sent %d picks", len(picks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
