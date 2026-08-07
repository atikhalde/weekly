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

-----------------------------------------------------------------------------
RECALL - WHY MOST BIG MOVERS ARE NOT ON THIS LIST
-----------------------------------------------------------------------------
This scanner is a PRECISION instrument, and the cost is stated here so it is
not mistaken for a defect. Measured over 892,121 tradeable stock-days:

    stocks that jump >= +5% tomorrow      31 PER WEEK, market-wide
    this scanner flags                   4.5 per week
    of those flagged, actually jump      23.4%   (precision)
    of all jumpers, we catch              3.4%   (RECALL)

So roughly 30 of every 31 big movers happen without us. Opening a mover list
the next morning and finding almost none of them on the previous evening's
BTST message is the EXPECTED outcome, not evidence of a broken scan.

That trade-off is forced, not chosen. The top-5/day cap allows 25 trades a
week; at a realistic ~23% precision, catching even 5 movers a week means
taking ~22 - the cap binds and quality collapses. Every second tier tested
measured NEGATIVE standing alone:

    cp>=0.90 & rvol>=5 & day>=8   n=688  -0.132%  OOS -0.346
    cp>=0.90 & rvol>=8            n=543  -0.065%  OOS -0.053
    day>=15 & cp>=0.90            n=285  -0.250%  OOS -0.450
    rvol>=10 & cp>=0.85           n=901  -0.259%  OOS -0.190
    cp>=0.95 & rvol>=5            n=174  +0.551%  OOS t 1.26  (best, still thin)

Adding any of them lowers the mean, the win rate and usually the drawdown.
The 97% we miss buys the 63% win rate. Do not "fix" recall by loosening.

    python btst.py               # send tonight's list
    python btst.py --dry-run     # print, do not send
    python btst.py --all         # show every breakout, with its tier

Never places orders.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from datetime import datetime, time as dtime, timedelta

import numpy as np
import pandas as pd

from config import load_config
from dhan import IST, DhanClient, DhanError
from mcap import load_table as load_mcap_table
from scan import load_snapshots
from watchlist import fetch_ltp
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

# BUG 62: this is CALENDAR days, and it was too short to compute anything
# annual. 260 calendar days is only ~179 trading bars after weekends and
# holidays, but ret_12m needs cl[-252] and the 200DMA needs 200 bars - so
# ret_12m was ALWAYS NaN on the confirmed path and dist_200dma was never
# computed at all. The live 04-Aug picks file shows it: ret_12m blank,
# dist_200dma 0.0 for both MOREPENLAB and RBA.
#
# 420 calendar days ~= 288 trading days, which clears 252 with margin for a
# long holiday run. The cost is a slightly larger daily-history payload per
# symbol, not an extra request.
HISTORY_DAYS = 420

# BUG 66: wall-clock budget for a POST-CLOSE / --after-close review run, which
# has no 15:26 deadline to race. It must still stop, because the workflow
# timeout (14 min) kills the job outright and produces nothing at all. 9
# minutes leaves ~5 for checkout, pip, the Telegram send and the git commit.
REVIEW_BUDGET_MIN = 9
BARS_PER_SESSION = 75      # NSE: 09:15-15:30 in 5-minute candles

# How many picks are actually taken. This is THE top-5: btst.py caps the list
# here and writes it to btst_picks.csv, and Model E trades that file rather
# than re-deriving its own list. One source of truth, so the 15:20 alert and
# the paper ledger can never name different stocks.
TOP_N = 5
PICKS_FILE = "btst_picks.csv"

# --------------------------------------------------------------------------- #
#  BUG 53 (04-Aug-2026) - THE AGE GATE APPLIES TO TIER A ONLY
#
#  BUG 43 required the breakout to have happened TODAY. That was measured on
#  BREAKOUT DAYS ONLY, which has no control group and so could not answer
#  "does the breakout day matter". Re-measured across the WHOLE universe -
#  892,858 tradeable stock-days, 2,075 names, Aug-2021 to Jul-2026, entry at
#  today's close, exit tomorrow's close, net 0.22%:
#
#      arm                     IS        OOS      OOS t
#      fresh tier A         +1.171    +2.157       2.79
#      fresh tier B         +0.179    +1.230       2.93
#      aged  tier A         +0.795    -0.843      -1.08   <- BUG 43 was RIGHT
#      aged  tier B         +0.999    +0.885       3.19   <- BUG 43 was WRONG
#
#  Tier A is a MAGNITUDE test (+15% day) and magnitude expires - three days
#  later it is a chase, median extension 15%, and it mean-reverts. Tier B is
#  a CHARACTER test (closed at the high, 3x volume, volatile name) and
#  character has no expiry date. Aged tier B has the SMALLEST in/out-of-sample
#  gap of the four arms - it is the most stable thing in the whole study.
#
#  The universe baseline is -0.135% (t=-43), so none of this is drift.
#
#  WHAT THIS BUYS, top-5/day portfolio, 5 years:
#      shipped (fresh only)   918 trades   48.7% win   PF 1.48   maxDD -60.0%
#      with aged tier B     1,043 trades   56.0% win   PF 1.79   maxDD -28.4%
#
#  HONEST NOTE ON THE PAIRED TEST. On days BOTH rules trade, the gain is
#  +0.575%/day (t=4.26) - but most of that comes from the close_pos floor
#  below, not from the aged arm. The aged arm's own contribution is COVERAGE:
#  the shipped rule sits in cash on 60% of business days. Measured alone at
#  the old thresholds it was +0.127%/day, t=1.29 - NOT significant. It is
#  shipped for coverage and drawdown, both of which are large and consistent,
#  not because it is a better signal per trade.
#
#  NOT ADOPTED: dropping the level condition entirely. Tier+trend+PRE>=6 with
#  no level test at all measures +0.813% (t=11.2, n=5,186) - the same number.
#  The 26W level is how these names are FOUND, not why they work. Removing it
#  is a much bigger change than this one and is not made on that evidence.
#
#  ---- THE ORIGINAL BUG 43 MEASUREMENT, KEPT SO THE NARROWING IS TRACEABLE --
#  Measured across 1,548 qualifying stock-days (breakout days only - that is
#  the flaw), next-day return net of costs:
#
#      age 0 (breakout day)   n=1174   +0.807%   t 5.00   mean ext  7.1%
#      age 1                  n= 164   +0.373%   t 0.87   mean ext 15.6%
#      age 2                  n= 107   +1.311%   t 2.45   mean ext 18.6%
#      age 3                  n=  74   +0.984%   t 1.64   mean ext 20.4%
#      age 4                  n=  29   -0.808%   t -1.01  mean ext 26.9%
#
#      Tier A  age 0    n=401   +1.736%   t 5.03
#      Tier A  age >=1  n= 83   +0.155%   t 0.20   <- still true, still excluded
#
#  The tier A rows survive re-measurement and are why AGE_GATE_TIERS exists.
#  What did NOT survive is applying the same conclusion to tier B, which was
#  never separately tested in that study.
#
#  Live example that motivated BUG 43 - 31-Jul-2026:
#      YASHO     broke out 31-Jul 12:20   age 0   ext  18.3%   valid
#      NELCO     broke out 31-Jul 12:50   age 0   ext   2.2%   valid
#      DEEPINDS  broke out 29-Jul 10:05   age 2   ext   9.8%   tier B -> now
#                                                              eligible again
AGE_GATE_TIERS = ("A",)      # tiers that MUST have broken out today

# Aged tier B is only reachable if it is near enough to its frozen level to
# survive the bulk-LTP pre-filter (see BUG 52 - the 15:20 job has ~250-400
# per-symbol history calls of budget before it misses the close). Measured
# share of the aged tier B edge retained, and the pool it admits per day:
#      band          keeps   per-trade   est. names/day
#      -3 .. +10      36%     +1.075          115
#      -5 .. +15      55%     +1.110          185
#     -10 .. +20      76%     +0.935          411   <- chosen
#     -15 .. +30      89%     +0.945          700   too many, misses the close
# -10..+20 is the widest band that fits the measured time budget.
AGED_EXT_MIN = -10.0
AGED_EXT_MAX = 20.0
AGED_MAX_AGE_DAYS = 250      # beyond this the cross is not a reference point

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

# --------------------------------------------------------------------------- #
#  TREND FLOOR (BUG 51, 03-Aug-2026)
#
#  Of 11 live alerts on 03-Aug, the four clean losers were the four names with
#  no established uptrend:
#      BHAGCHEM  ret_12m  -9.8%   dist200 13.1%   ->  -2.64%
#      TBOTEK    ret_12m  +8.0%   dist200  5.5%   ->  -3.48%
#      SPORTKING ret_12m +51.9%   gapped +15%     ->  -1.24%
#      SWANDEF   no 12m history                   ->  -0.40%
#  while all six winners had ret_12m >= 53% and dist200 >= 25%.
#
#  n=11 is far too small to act on, so it was re-tested on 18,202 tradeable
#  signals over 5 years. The trend effect is real and monotonic:
#
#      ret_12m bucket     n       win%    mean      OOS
#      < 10 (no trend)    1,693   33.3%   -0.339%   -0.397%
#      10-50              6,145   35.3%   -0.099%   -0.568%
#      >= 50             10,364   37.2%   +0.456%   +0.151%
#      >= 100             5,273   38.3%   +0.711%   +0.489%
#
#  and stacked on the shipped close_pos filter:
#
#      close_pos>=0.9                       n 3,405  +1.043%  OOS +0.881%
#      + ret_12m>=40  (previous)            n 2,376  +1.404%  OOS +1.341%
#      + ret_12m>=50 & dist200>=25  (now)   n 1,822  +1.610%  OOS +1.647%
#
#  OOS improves +1.341% -> +1.647% and still leaves ~7 setups a week.
#
#  NOT ADOPTED: a gap cap. It looked decisive on 03-Aug (SPORTKING gapped
#  +15%, BHAGCHEM +8.5%, both lost) but over 5 years gap_pct<=4 measured
#  +0.160% against a +0.194% baseline - slightly WORSE. Two observations are
#  not evidence.
MIN_RET_12M = 50.0     # percent, trailing twelve months
MIN_DIST_200DMA = 25.0  # percent above the 200-day moving average
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
#  ---- CORRECTION (BUG 56, 04-Aug-2026): IT IS APPLIED TO ONE LIST, NOT TWO
#  This block used to claim "MIN_PRE_CONFIRM is therefore applied to BOTH
#  lists here". It never was. classify_approach() enforces it; the confirmed
#  tier path in classify() does not, and never did.
#
#  Caught from the live btst_picks.csv, which recorded DALMIASUG at PRE 4/8 -
#  impossible if the floor applied. Every table in this file that quotes a
#  "PRE>=6" figure for the CONFIRMED list therefore describes a rule the
#  scanner was not running.
#
#  MEASURED BEFORE CHANGING ANYTHING. The confirmed-tier list, 5 years, with
#  the BUG 54 floors in place:
#
#      variant                       n     /wk    mean     win%   PF    OOS
#      as it actually runs (no PRE) 786    3.0   +2.092%   62.7  2.36  +2.715
#      as it was documented (>=6)   745    2.9   +2.085%   62.4  2.32  +2.744
#      the slice that leaks in       41          +2.229%   68.3  3.66  +2.334
#
#  The leaking slice is BETTER than the list it leaks into: t=3.47, positive
#  in all six years, and it survives removing its top 3 trades (+1.642%).
#  Top-5/day portfolio: no floor 62.7% win / maxDD -28.6%, with floor 62.4% /
#  -34.5%. The accident is mildly HELPING.
#
#  So the CODE IS LEFT ALONE and the comment is corrected instead. Adding the
#  floor now would remove ~41 good trades to satisfy a sentence.
#
#  Why the PRE score adds nothing here any more: it was measured as a proxy
#  for trend quality against close_pos>=0.90. BUG 51 added an explicit trend
#  floor (ret_12m>=50, dist200>=25) and BUG 54 raised close_pos to 0.98 -
#  both of which test the same thing more directly. By PRE bucket on today's
#  rule the ordering is flat-to-inverted (5/8 +2.838%, 7/8 +1.897%,
#  8/8 +2.116%), so there is nothing left for it to add.
#
#  It STILL GATES THE ANTICIPATION LIST, where it was measured and where
#  close_pos is only required to be 0.90. Do not "tidy" it away from there.
# --------------------------------------------------------------------------- #
MIN_PRE_CONFIRM = 6

# --------------------------------------------------------------------------- #
#  BUG 65 - PRIOR EXHAUSTION  (user observation, 05-Aug-2026)
#
#  The user objected that the picks "look like extended rallies" and that a
#  stock which already ran should not be expected to run again. Two separate
#  claims were tested on 858,401 tradeable stock-days, losers kept as control.
#
#  CLAIM A - "it moved +20% TODAY so it won't move big tomorrow" - REFUTED,
#  and refuted hard. P(next day >= +5%) by today's move:
#
#      today's move    n        P(+5%)   lift
#      -5..0%       464,722      3.6%    0.86
#       0..3%       278,241      3.8%    0.92
#       3..6%        64,664      5.3%    1.28
#       6..10%       17,781      8.2%    1.99
#      10..15%        4,527     11.7%    2.83
#      15..20%        1,520     21.3%    5.15
#
#  Monotonic the OPPOSITE way to the claim. Restricted to names that also
#  closed at their high, a +15-20% day reaches 33.4% - an 8x lift. TIER_A_DAY
#  stays at 15. Momentum does not exhaust overnight.
#
#  CLAIM B - "there is no tight-structure / VCP logic" - TRUE, and it stays
#  that way, because tightness measured NEGATIVE at every horizon:
#
#      prior-base Bollinger width  0-15th pct (tightest)  +2.478%  lift 0.93
#                                  85-100th   (widest)    +2.504%  lift 1.07
#      base contraction  <0.70 (coiled)   n= 76  P(+5%)  9.2%  lift 0.40
#                        >1.20 (expanding) n=474 P(+5%) 31.2%  lift 1.35
#      tight base vs loose base, same breakout shape:
#          TIGHT  n=123  +1.868%  P(+5%) 20.3%  OOS +1.696%
#          LOOSE  n=714  +2.673%  P(+5%) 29.6%  OOS +2.658%
#
#  This restates the STRUCTURE FINDING already in the repo (391,207 sampled
#  stock-days): coiled bases break out less reliably, expanding ones run.
#  A squeeze is a good ENTRY pattern for a swing trade; it is not a predictor
#  of a large OVERNIGHT gap, which is what this model sells.
#
#  BUT THE UNDERLYING INSTINCT WAS RIGHT, AT A DIFFERENT TIME SCALE. What is
#  exhausted is not today's candle, it is the QUARTER behind it:
#
#      slice                                  n     mean    P(+5%)  lift
#      base sat within 2% of its 252d high   325   +1.333%   12.6%  0.54
#      prior 3-month return > +100%          106   +1.592%    7.5%  0.32
#      both true                              71   +1.827%    4.2%  0.18
#      everything else                     1,366   +2.624%   25.9%  1.12
#
#  So: a breakout is worth taking when the stock is emerging from somewhere,
#  and worth skipping when it is already pinned at 52-week highs after
#  doubling. Welch t on the difference 4.09; date-clustered t on the kept set
#  13.90; every year positive; the kept set survives de-duplication (n=1,227,
#  +2.629%) and removal of its 10 most frequent names (+2.619%).
#
#  Threshold grid -1..-8% x 60..150% all land +2.55..+2.74% - not a
#  cherry-pick. -2% / +100% is the loosest pair that captures the effect.
#
#  HONEST LIMITS. Paired daily t on the top-5 portfolio is +1.79, so this is
#  suggestive rather than decisive at the portfolio level; it earns its place
#  because the DROPPED slice is independently and consistently poor, not
#  because the headline moved. It also costs ~1.4 setups a week, and drawdown
#  is slightly worse (-19.8% -> -21.1%) because fewer names share the risk.
#
#  Applied to the live 04-Aug list: MOREPENLAB (base -10.7%, 3m +37%) and RBA
#  (base -19.3%, 3m +8%) both SURVIVE - they were not the extended ones. Two
#  of the three ANTICIPATE picks, RAIN (base -1.6%) and TFCILTD (base -1.5%),
#  are dropped. The user's eye was right about the anticipation list.
MAX_BASE_FROM_HIGH = -2.0    # yesterday's close must be >2% below the 252d high
MAX_RET_3M_PRIOR = 100.0     # and the prior quarter must not have doubled


def exhausted(m: dict) -> str | None:
    """Reason this setup is already spent, or None if it is still fresh.

    NaN PASSES here, unlike the trend floor. A young listing has no 252-day
    high to be pinned against, and refusing it would re-introduce the BUG 51
    over-reach on a population where it was never measured.
    """
    bfh = m.get("base_from_high", float("nan"))
    r3 = m.get("ret_3m_prior", float("nan"))
    if bfh == bfh and bfh >= MAX_BASE_FROM_HIGH:
        return f"base was {bfh:+.1f}% off its 252d high (already at highs)"
    if r3 == r3 and r3 > MAX_RET_3M_PRIOR:
        return f"prior 3m {r3:+.0f}% (already doubled)"
    return None


# Thresholds live here and are imported by ab_paper.py's Model E, so the paper
# model and the nightly scanner can never drift apart.
TIER_A_DAY = 15.0

# --------------------------------------------------------------------------- #
#  BUG 54 (04-Aug-2026) - TIER A GETS A CLOSE_POS FLOOR TOO
#
#  RETRACTION. BUG 53 said, in this file: "TIER A IS LEFT AT 0.85 ON PURPOSE
#  ... tightening the one thing that is already performing, on the same data
#  that says it performs, is how a good rule gets overfitted into a rare one."
#
#  That reasoning was sound but the conclusion was wrong, and the check that
#  proves it is the one BUG 53 failed to run: what does the DISCARDED slice
#  look like? Tier A signals with close_pos 0.85-0.98:
#
#      n=193   mean +0.101%   win 42.5%   OOS +0.275%
#      by year: 2021 -0.06 · 2022 -0.86 · 2023 +0.61 · 2024 +0.46
#               2025 -0.07 · 2026 +0.39
#
#  That is not a slice being sacrificed for purity - it is worthless. It is
#  ~0.75 trades a week of noise sitting in the same five slots as real setups.
#
#  Full sweep of tier A signals only (n, mean, OOS, all-years-positive):
#      0.85   392   +1.425   +2.157   NO
#      0.90   318   +1.643   +2.602   NO
#      0.95   242   +2.331   +3.927   NO
#      0.97   215   +2.703   +4.742   yes
#      0.98   199   +2.709   +4.499   yes   <- chosen
#      0.99   174   +3.127   +4.143   yes
#
#  Monotonic, and 0.97 is the first floor where EVERY YEAR is positive.
#
#  WHY THIS IS NOT DOUBLE-COUNTING THE SAME BET. A tier A day is >=+15% by
#  definition. Requiring it to ALSO finish in the top 2% of its range is a
#  different question: it is the difference between a +15% day that HELD and
#  a +15% day that FADED. Same logic as the tier B floor, applied consistently
#  instead of only to the arm nobody was defending.
#
#  Portfolio effect (top-5/day, tier B floor at 0.98 throughout):
#      TIER_A_CLOSE_POS   trades   per-trade   win%    PF     OOS
#          0.85             928     +1.646     58.3   1.98   +2.243
#          0.98             742     +2.087     62.4   2.32   +2.744
#  Paired on the 476 common days: +0.122%/day, t=+1.83. Not significant on
#  its own - but the discarded slice is +0.101% over five years, so this is
#  removing noise rather than making a bet.
#
#  COST, STATED PLAINLY: tier A drops from 1.5 to 0.8 signals a week, and the
#  portfolio loses 77 trading days of coverage (553 -> 476). Drawdown gets
#  WORSE, -29.4% -> -34.5%, because tier B's steadier aged arm now fills a
#  larger share of the slots. Taken because per-trade, win rate, PF and OOS
#  all improve together and the discarded rows are demonstrably empty.
TIER_A_CLOSE_POS = 0.98

# --------------------------------------------------------------------------- #
#  BUG 53b - TIER B CLOSE_POS RAISED 0.90 -> 0.95
#
#  close_pos was already known to be the strongest single factor (THE_EDGE.md).
#  Swept properly on the full universe it is cleanly MONOTONIC, which is what
#  separates a real effect from a cherry-picked cell:
#
#      close_pos >=   trades   per-trade   win%    PF     IS      OOS
#          0.90        1,667     +0.951    52.2   1.56  +0.794  +1.343
#          0.92        1,328     +1.168    54.7   1.68  +1.016  +1.533
#          0.94        1,122     +1.443    57.0   1.85  +1.268  +1.863
#          0.95        1,065     +1.584    58.4   1.95  +1.407  +2.017
#          0.97          948     +1.931    61.7   2.22  +1.693  +2.535
#          0.99          824     +2.137    63.3   2.41  +1.950  +2.623
#
#  It improves ALL THREE arms independently (fresh A +1.425->+2.331, fresh B
#  +0.448->+1.235, aged B +0.964->+1.415) and it also improves the CURRENTLY
#  SHIPPED fresh-only rule (+0.859 -> +1.758), which is independent
#  confirmation that this is not an artifact of the new aged arm.
#
#  CIRCUIT-LOCK CHECK. In India a stock locked at the upper circuit closes
#  exactly at its high and CANNOT BE BOUGHT - there is no offer. If the top
#  bucket were full of locks the measured return would be unreachable. It is
#  not: the 0.99-1.00 band has a median day range of 8.3% and median rvol
#  5.4x, i.e. names that traded freely all day and closed strong. A
#  conservative lock proxy (close_pos>=0.995 AND range < half the day's gain)
#  flags only 0.7% of rows, and removing them changes nothing material.
#
#  ---- BUG 54: 0.95 WAS THE WRONG PLACE. RAISED TO 0.98 --------------------
#  RETRACTION of the paragraph that used to sit here ("0.95 rather than
#  0.97/0.99 deliberately ... thinner is not automatically better").
#
#  The sweep above is monotonic, so 0.95 looked like a reasonable
#  frequency/quality trade. It is not, because the monotonic curve hides a
#  DEAD BAND. Measured on the shipped pool, the slice a 0.95 floor admits and
#  a 0.98 floor rejects:
#
#      close_pos 0.95-0.99 (the band 0.95 lets in)
#          n=235   mean -0.222%   win 41.7%
#
#  Those trades are NEGATIVE. The gain from 0.95 -> 0.98 is not "buying
#  quality with frequency", it is deleting a losing bucket.
#
#  Confirmed WITHIN each arm, so it is not an arm-mix artifact:
#      arm        cp<0.99            cp>=0.99
#      fresh_A    +0.067% (n=218)    +3.127% (n=174)
#      fresh_B    -0.649% (n= 68)    +1.984% (n=171)
#      aged_B     -0.285% (n= 99)    +1.969% (n=333)
#
#  DATA-ARTIFACT CHECK, because "closed exactly at the high" is suspicious.
#  Over 509,562 random stock-days: close==high 2.97%, close==low 2.24%,
#  ratio 1.33. Roughly symmetric, so this is real market behaviour (strong
#  closes cluster) and not a feed rounding close into high.
#
#  WHY 0.98 AND NOT 0.99. 0.99 measures marginally better per trade
#  (+1.702 vs +1.646) but drawdown jumps -29.4% -> -40.1%, and the live scan
#  judges close_pos on a PARTIAL 15:20 candle. A 0.99 gate is knife-edge
#  against a bar with ten minutes left to run; 0.98 leaves room for the last
#  wobble. The 15:20 cutoff study says these names drift UP into the bell
#  (entry vs close -0.14%, tier precision 82.3%), so the partial reading is
#  usually conservative - but not always, and 0.98 costs almost nothing.
TIER_B_CLOSE_POS = 0.98
TIER_B_RVOL = 3.0
TIER_B_ATR = 3.0


# --------------------------------------------------------------------------- #
#  BUG 54b - THE CONVICTION SCORE: "which of tonight's picks is the big one"
#
#  The user asked how to find the BIG winners, not just the good ones. This is
#  a lift analysis - P(big winner | feature) / base rate - with every losing
#  signal kept as the control, because profiling winners alone is what
#  produced the "16 named winners" dead end (feature ratios all ~1.0).
#
#  "Big winner" = next-day close >= +5% net. Base rate on the shipped pool is
#  18.3%. Lifts, measured over 892,858 tradeable stock-days:
#
#      feature                P(+5%)   lift    interpretation
#      gap >= +8%              38.0%   2.07    strongest single lift
#      fresh TIER A            26.8%   1.46    the +15% day itself
#      atr_pct >= 7%           29.2%   1.59    only volatile names go far
#      atr_pct >= 5%           23.5%   1.28
#      day_ret >= +6%          26.0%   1.42
#      ret_12m 50-100%         27.1%   1.48    EARLY trend, not late
#      ret_12m >= 400%         6.4%    0.35    exhausted - avoid
#      dist200 >= 120%         9.4%    0.51    exhausted - avoid
#      aged tier B             10.0%   0.54    steady, NOT explosive
#
#  TWO COUNTERINTUITIVE ONES, both stable:
#    * ret_12m is INVERTED for the fat tail. A stock already up 400% rarely
#      adds another 5% overnight. The trend floor (>=50%) is still right -
#      it removes no-trend junk - but MORE trend is not better past ~200%.
#    * aged tier B has the BEST win rate (62.3%) and the WORST big-winner
#      rate. It is the steady arm. That is a feature, not a fault, and it is
#      why the answer is a RANK and not a FILTER.
#
#  SCORED AS A RANK, NOT A GATE. Measured as a filter it does nothing:
#      conv >= 2   627 trades   +1.672%   CAGR 264%   DD -33.4%
#      conv >= 3   467 trades   +1.664%   CAGR 176%   DD -47.6%
#      all         938 trades   +1.677%   CAGR 494%   DD -29.4%
#  Filtering on it costs coverage and RAISES drawdown for no gain in mean.
#  But it orders the fat tail cleanly, which is what the question was:
#      conv  n     P(+5%)   P(+10%)   mean
#       0    141    4.3%      1.4%   +1.840
#       1    170    8.2%      2.4%   +1.557
#       2    160   20.6%      5.0%   +1.696
#       3    265   27.2%      9.4%   +1.489
#       4    202   31.2%     14.4%   +1.894
#  P(+5%) goes 4.3% -> 31.2% and P(+10%) 1.4% -> 14.4%, monotonically, while
#  the MEAN stays flat. Read that carefully: a high score does not predict a
#  better average trade, it predicts a more EXPLOSIVE, less reliable one.
#  Note conv 0 has the highest win rate (69.5%) and the lowest P(+5%).
#
#  So it is shown in the alert and used as a tie-break inside each arm. It
#  never removes a trade. If the user wants to size up the fat tail, this is
#  the number to size on - accepting a lower hit rate for a fatter tail.
CONVICTION_MAX = 4


def conviction(m: dict) -> tuple[int, list[str]]:
    """0-4: how likely this setup is to be a BIG (>=+5%) winner, not just a
    winner. See the block above - this is a RANK, never a gate."""
    def f(k):
        try:
            v = float(m.get(k))
        except (TypeError, ValueError):
            return float("nan")
        return v

    hits = []
    if m.get("fresh") and m.get("tier") == "A":
        hits.append("fresh TIER A")
    a = f("atr_pct")
    if a == a and a >= 5.0:
        hits.append("atr>=5%")
    dr = f("day_ret")
    if dr == dr and dr >= 6.0:
        hits.append("day>=+6%")
    r12 = f("ret_12m")
    # early-trend, not exhausted. NaN fails, consistent with the trend floor.
    if r12 == r12 and r12 <= 200.0:
        hits.append("trend not exhausted")
    return len(hits), hits


def repair_today_bar(df: pd.DataFrame, quote: dict | None,
                     today) -> pd.DataFrame:
    """
    Trust the bulk OHLC quote for TODAY's open/high/low.

    BUG 64. On 04-Aug the scan reported E2E at day_ret 0.00% when the real bar
    was 552.00 -> 570.15 (+3.29%). The daily candle came back with
    open == close == high, a degenerate bar: day_ret collapses to zero and
    close_pos to 1.000. RAIN and TFCILTD matched Yahoo within feed noise on
    the same run, so this is a per-symbol data defect, not a formula error.

    It matters because day_ret is a GATE, not decoration: TIER_A_DAY needs
    >= +15% and conviction awards a point at >= +6%. A zeroed open silently
    demotes a tier A setup to nothing - the scanner would simply never
    mention the best trade of the day.

    The bulk /marketfeed/ohlc response already carries today's true open,
    high and low for every symbol, at no extra request, so it is used to
    repair the last bar whenever it is today's. The CLOSE is left alone: the
    daily bar (or the 5-minute reconstruction) is the better source for it,
    and only the open was observed to be wrong.
    """
    if df is None or df.empty or not quote:
        return df
    try:
        if pd.Timestamp(df.iloc[-1]["datetime"]).date() != today:
            return df
        q_o = float(quote.get("open") or 0.0)
        q_h = float(quote.get("high") or 0.0)
        q_l = float(quote.get("low") or 0.0)
    except (TypeError, ValueError, KeyError):
        return df
    if q_o <= 0:
        return df
    i = df.index[-1]
    c = float(df.at[i, "close"])
    df.at[i, "open"] = q_o
    # keep the bar self-consistent: the range must span open and close
    hi = max(float(df.at[i, "high"]), q_h, q_o, c)
    lo = min(float(df.at[i, "low"]) or q_l, q_l or float(df.at[i, "low"]), q_o, c)
    df.at[i, "high"] = hi
    if lo > 0:
        df.at[i, "low"] = lo
    return df


def vma_debug(prev: pd.DataFrame, symbol: str, n: int = 50) -> None:
    """
    BUG 72 INSTRUMENTATION. Log exactly what the volume window contains.

    I have now been wrong TWICE about SONACOMS reporting rvol 3.0x when an
    independent 50-day average says 1.56x. First I blamed the partial-candle
    fraction (BUG 70); it was not that. Then I blamed zero-volume rows in the
    window (BUG 71); the fix went live and the number did not move, so it is
    not that either.

    Both times I inferred a mechanism from the symptom instead of looking at
    the input. This prints the actual window so the next diagnosis is made
    from data. Enabled by --debug-vma; costs nothing when off.
    """
    if "volume" not in prev or prev.empty:
        log.info("VMA %s: no volume column", symbol)
        return
    v = pd.to_numeric(prev["volume"], errors="coerce")
    w = v.tail(n)
    nz = w[w > 0]
    log.info("VMA %s: window=%d nonzero=%d nan=%d zero=%d "
             "mean=%.0f median=%.0f min=%.0f max=%.0f first=%s last=%s",
             symbol, len(w), len(nz), int(w.isna().sum()), int((w == 0).sum()),
             nz.mean() if len(nz) else 0, nz.median() if len(nz) else 0,
             nz.min() if len(nz) else 0, nz.max() if len(nz) else 0,
             f"{w.iloc[0]:.0f}" if len(w) else "-",
             f"{w.iloc[-1]:.0f}" if len(w) else "-")


def _vma50(prev: pd.DataFrame, n: int = 50) -> float:
    """
    Average daily volume over the last `n` bars that ACTUALLY TRADED.

    BUG 71. This was `prev["volume"].tail(50).mean()`. Both classify() and
    classify_approach() dropna on open/high/low/close but NOT on volume, so a
    row with a valid close and volume 0 or NaN - an exchange holiday, a
    suspension, or feed padding - stays in the window and drags the mean down.

    vma is the DENOMINATOR of rvol, so a halved average doubles rvol.
    Measured on the 07-Aug list: SONACOMS displayed 3.0x against a true 50-day
    figure of 1.56x - the implied vma was 1,249,780 vs an actual 2,402,774,
    a ratio of 1.92, i.e. roughly half the window was non-trading rows.

    On the ANTICIPATE list rvol is display-only, so that was cosmetic. On the
    CONFIRMED list TIER_B_RVOL >= 3.0 is a hard gate, and this could promote a
    quiet stock into tier B on nothing but holidays in its history.

    Zero-volume days are excluded rather than zero-filled: "did not trade" is
    not "traded nothing", and averaging them in understates normal turnover.
    """
    if "volume" not in prev or prev.empty:
        return 0.0
    v = pd.to_numeric(prev["volume"], errors="coerce")
    v = v[v > 0].tail(n)
    return float(v.mean()) if len(v) else 0.0


def _rvol_detail(r: dict) -> str:
    """
    "13.7x (6.0M / 0.4M over 50d)" - the ratio AND its inputs.

    BUG 73. rvol is a hard gate on the confirmed list (TIER_B_RVOL) and the
    most-questioned number in the message. When SONACOMS displayed 3.0x
    against an independently computed 1.54x, the ratio alone made the cause
    unknowable: three hypotheses were tested and all three were wrong, purely
    because today's volume and the 50-day average were never shown.

    Printing the inputs turns "that number looks wrong" into "that number is
    wrong BECAUSE the average used 12 bars, not 50" - visible in the alert,
    with no workflow log and no debug flag needed.
    """
    def m(x):
        try:
            x = float(x)
        except (TypeError, ValueError):
            return "?"
        if x != x:
            return "?"
        if x >= 1e7:
            return f"{x/1e7:.1f}Cr"
        if x >= 1e5:
            return f"{x/1e5:.1f}L"
        if x >= 1e3:
            return f"{x/1e3:.0f}K"
        return f"{x:.0f}"

    v = r.get("vol_today")
    a = r.get("vma50")
    n = r.get("vma_bars")
    if v is None or a is None or not a:
        return ""
    return f" <i>({m(v)}/{m(a)} over {n}d)</i>"


def _vma_n(prev: pd.DataFrame, n: int = 50) -> int:
    """How many bars actually fed _vma50 - 0 means the average is meaningless."""
    if "volume" not in prev or prev.empty:
        return 0
    v = pd.to_numeric(prev["volume"], errors="coerce")
    return int(len(v[v > 0].tail(n)))


_DBG_VMA: set[str] = set()


def session_fraction(now_ist, n_bars: int | None = None) -> float:
    """
    Fraction of the NSE session elapsed, from the CLOCK.

    BUG 70. This used to be `len(m5) / 75` - how many 5-minute bars the API
    returned, divided by a full session. That silently conflates two very
    different things:

        session still running   ->  fewer bars, frac < 1   CORRECT
        API returned a partial  ->  fewer bars, frac < 1   WRONG

    `frac` divides the volume benchmark, so a wrongly-low frac INFLATES rvol.
    On the 07-Aug 17:34 review run - two hours after the close - SONACOMS came
    back with roughly half the day's bars and its rvol doubled from 1.5x to
    3.1x, stepping over the TIER_B_RVOL >= 3.0 gate. A post-close rerun was
    manufacturing a Tier B setup that did not exist at the bell.

    The clock cannot be fooled that way: after 15:30 the elapsed fraction is
    1.0 by definition, and missing bars are a data gap to be reported, not
    evidence of an unfinished session.

    `n_bars` is accepted only so the caller can flag a suspicious gap; it
    never shortens the fraction.
    """
    t = now_ist.time()
    if t >= dtime(15, 30):
        return 1.0
    if t <= dtime(9, 15):
        return 0.05
    elapsed = ((now_ist.hour * 60 + now_ist.minute)
               - (9 * 60 + 15))
    total = (15 * 60 + 30) - (9 * 60 + 15)      # 375 minutes
    return min(max(elapsed / total, 0.05), 1.0)


def _num(v, nd: int = 2):
    """Round for the picks file, but keep NaN/None BLANK rather than 0.

    BUG 62. `float(x or 0)` silently turns "no 12-month history" into "+0.0%
    return", which is a different and much more confident claim. Any later
    review of the ledger would read it as a flat stock rather than an unknown.
    """
    try:
        f = float(v)
    except (TypeError, ValueError):
        return ""
    return round(f, nd) if f == f else ""


def run_until(fn, items, workers: int, deadline, log_label: str):
    """
    Map `fn` over `items` and STOP AT `deadline`, whatever is still in flight.

    BUG 61. The previous version used ex.map(), which yields IN SUBMISSION
    ORDER. A single slow symbol therefore blocked every deadline check behind
    it: the loop could not break because nothing was being yielded. One
    unlucky request can take 210 seconds (5 attempts x 30s HTTP timeout plus
    exponential backoff) and a symbol needs TWO calls, so ONE name can hold
    the scan for seven minutes - which is the whole entry window.

    This iterates in COMPLETION order and re-checks the wall clock on every
    tick, so a stall costs only the results still outstanding. Pending work is
    cancelled; in-flight requests are abandoned (their threads are daemonic
    from the pool's point of view and the process exits after sending).

    `deadline` may be None to disable the cap (post-close review).
    Returns (results, n_unfinished).
    """
    out, pending = [], []
    ex = ThreadPoolExecutor(max_workers=max(1, workers))
    try:
        futs = {ex.submit(fn, it): it for it in items}
        pending = set(futs)
        while pending:
            if deadline is not None:
                left = (deadline - datetime.now(IST)).total_seconds()
                if left <= 0:
                    break
            else:
                left = None
            done, pending = wait(pending, timeout=left, return_when=FIRST_COMPLETED)
            if not done:
                break                      # timed out with nothing new
            for f in done:
                try:
                    out.append(f.result())
                except Exception as exc:   # one bad symbol must not kill the run
                    log.debug("%s: %s", log_label, str(exc)[:120])
        for f in pending:
            f.cancel()
        return out, len(pending)
    finally:
        # do NOT block on in-flight requests - that is the hang we are fixing
        ex.shutdown(wait=False, cancel_futures=True)


def fetch_ohlc(client, snaps: list) -> dict:
    """
    symbol -> {last_price, open, high, low, prev_close} via bulk quotes.

    BUG 59. This is the cheap half of the tier test. /marketfeed/ohlc returns
    today's open/high/low plus the LTP for up to 1000 instruments in ONE
    request, which is everything needed to compute close_pos and day_ret -
    the two mandatory gates - with no per-symbol history call at all.
    """
    from watchlist import QUOTE_BATCH

    by_seg: dict[str, list[int]] = {}
    ident: dict[tuple[str, str], str] = {}
    for s in snaps:
        by_seg.setdefault(s.exchange_segment, []).append(int(s.security_id))
        ident[(s.exchange_segment, str(s.security_id))] = s.symbol

    out: dict[str, dict] = {}
    for seg, ids in by_seg.items():
        for i in range(0, len(ids), QUOTE_BATCH):
            try:
                part = client.ohlc({seg: ids[i:i + QUOTE_BATCH]})
            except DhanError as exc:
                log.warning("ohlc batch failed: %s", str(exc)[:140])
                continue
            for sg, m in part.items():
                for sid, payload in m.items():
                    sym = ident.get((sg, str(sid)))
                    if sym:
                        out[sym] = payload
    return out


def cheap_close_pos(q: dict) -> tuple[float, float]:
    """
    (close_pos, day_ret) from a bulk OHLC payload alone. NaN when unusable.

    Uses the LTP as the running close, exactly as the partial-candle path
    does. High/low are today's so far, so this is the same quantity
    classify() computes - just without paying for history.
    """
    try:
        px = float(q.get("last_price") or 0.0)
        hi = float(q.get("high") or 0.0)
        lo = float(q.get("low") or 0.0)
        op = float(q.get("open") or 0.0)
    except (TypeError, ValueError):
        return float("nan"), float("nan")
    if px <= 0 or hi <= 0 or lo <= 0 or hi < lo:
        return float("nan"), float("nan")
    rng = hi - lo
    cp = ((px - lo) / rng) if rng > 0 else 0.5
    dr = ((px / op - 1) * 100.0) if op > 0 else float("nan")
    return cp, dr


def breakout_age(daily: pd.DataFrame, level: float,
                 lookback: int = AGED_MAX_AGE_DAYS) -> int:
    """
    Trading days since the last FRESH cross of `level`. 0 == crossed today.

    A cross is a close above the level immediately after a close at or below
    it, which is the same definition replay_week uses, so "age 0" here and a
    same-day alert mean the same event. Returns 999 when there is no cross
    inside `lookback` - that name has been above (or below) the whole time and
    has no breakout to age.

    Uses CLOSED bars plus today's forming bar; no look-ahead.
    """
    if daily is None or level is None or level <= 0 or len(daily) < 2:
        return 999
    c = daily["close"].to_numpy(float)[-(lookback + 1):]
    above = c > level
    if len(above) < 2:
        return 999
    # walk back from today looking for the transition below -> above
    for k in range(len(above) - 1, 0, -1):
        if above[k] and not above[k - 1]:
            return int(len(above) - 1 - k)
    return 999


def classify(daily: pd.DataFrame, level: float,
             partial_frac: float = 1.0, age: int | None = None,
             dbg_symbol: str = "") -> dict | None:
    """
    Score today's breakout candle. Returns None when it cannot be judged.

    `age` is days since the level was crossed (0 == today). When given, the
    BUG 53 age gate applies: tier A requires age 0, tier B does not. When
    None the age is computed from `daily` itself.

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
    if _DBG_VMA and str(dbg_symbol or '').upper() in _DBG_VMA:
        vma_debug(prev, str(dbg_symbol))
    vma = _vma50(prev)
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
    # BUG 62: dist_200dma was never computed here, so the picks file recorded
    # 0.0 for every confirmed pick and the trend context was unauditable.
    # These are REPORTING fields on this path - BUG 57 removed the trend floor
    # from the tier gate - but they are what makes a pick reviewable later,
    # and Model E carries them into the ledger.
    dist_200dma = (float((c / np.mean(cl[-200:]) - 1) * 100.0)
                   if len(cl) >= 200 else float("nan"))

    # ---- BUG 65: PRIOR EXHAUSTION (user observation, 05-Aug-2026) ---------
    # Measured on the PRIOR bars only - today is excluded, because today's
    # breakout is the trigger, not the exhaustion.
    #   base_from_high : where yesterday's close sat vs the 252-day high
    #   ret_3m_prior   : the move already made in the previous quarter
    hi252_prior = float(np.max(hi[-253:-1])) if len(hi) >= 253 else float("nan")
    prior_close = float(cl[-2]) if len(cl) >= 2 else float("nan")
    base_from_high = (float((prior_close / hi252_prior - 1) * 100.0)
                      if hi252_prior == hi252_prior and hi252_prior > 0
                      else float("nan"))
    ret_3m_prior = (float((prior_close / cl[-64] - 1) * 100.0)
                    if len(cl) >= 64 and cl[-64] > 0 else float("nan"))

    pre, _ = _pre_score_from_daily(d, level, c)
    m = dict(
        pre=pre,
        close=c, day_ret=(c / o - 1) * 100.0,
        close_pos=((c - lo) / rng) if rng > 0 else 0.5,
        rvol=(v / vma_cmp) if vma_cmp > 0 else float("nan"),
        partial_frac=frac,
        # BUG 73: carry the rvol INPUTS, not just the ratio. A wrong rvol is
        # undiagnosable from the ratio alone - see the SONACOMS 3.0x-vs-1.54x
        # hunt, where three separate hypotheses were tested and all wrong
        # because nobody could see today's volume or the 50-day average.
        vol_today=v, vma50=vma, vma_bars=int(_vma_n(prev)),
        range_pct=(rng / c * 100.0),
        gap_pct=((o / float(prev.iloc[-1]["close"]) - 1) * 100.0)
        if float(prev.iloc[-1]["close"]) > 0 else float("nan"),
        atr_pct=atr_pct, turnover_cr=turnover, ret_12m=ret_12m,
        dist_200dma=dist_200dma,
        base_from_high=base_from_high, ret_3m_prior=ret_3m_prior,
        ext_pct=((c / level - 1) * 100.0) if level > 0 else float("nan"),
    )

    # --- tradeability, hard
    if not (m["turnover_cr"] >= MIN_TURNOVER_CR and c >= MIN_PRICE
            and m["atr_pct"] >= MIN_ATR_PCT):
        m["tier"] = None
        m["reject"] = "not tradeable"
        return m

    # ---- BUG 53: how old is the breakout, and does this tier tolerate age?
    if age is None:
        age = breakout_age(d, level)
    m["age"] = int(age)
    m["fresh"] = bool(age == 0)

    # BUG 65: reject a breakout that is already spent, BEFORE tiering it.
    spent = exhausted(m)
    if spent:
        m["tier"] = None
        m["reject"] = spent
        return m

    rv = m["rvol"] if not np.isnan(m["rvol"]) else 0.0
    # --- TIER A: the YASHO shape. Rare, and the strongest measured.
    #     FRESH ONLY - aged tier A measured IS +0.795 -> OOS -0.843.
    #
    # ORDERING NOTE (found by cross-validating this code against the study).
    # An AGED name can have a tier-A-shaped day AND independently pass the
    # tier B character test. Testing A first rejects it outright. Letting it
    # through as B was measured as an alternative and it is a WASH:
    #
    #     rule                                trades  per-trade  win%   DD
    #     strict (this code, A-shape aged out) 1,043    +1.370   56.0  -28.4%
    #     loose  (rescued as tier B)           1,088    +1.378   55.9  -27.7%
    #
    # 45 extra trades in 5 years and no difference worth the special case, so
    # the simpler rule stands. The decomposition is worth keeping though,
    # because it explains WHY aged tier A dies:
    #
    #     aged + A-shape, NO tier B character   n=112  -0.875%  OOS -2.506%
    #     aged + A-shape, WITH tier B character n=123  +1.424%  OOS +1.208%
    #
    # The toxic half is the aged big-day WITHOUT accumulation character - a
    # spike that already happened and is not still being bought. That is the
    # real reason for the gate, and it is a stronger statement than "age is
    # bad". Do not reopen this on the n=54 in-band subset (t=1.54, one year
    # at -5.71%): it is too thin to justify branching the logic.
    if m["day_ret"] >= TIER_A_DAY and m["close_pos"] >= TIER_A_CLOSE_POS:
        if m["fresh"]:
            m["tier"] = "A"
        else:
            m["tier"] = None
            m["reject"] = f"tier A but {age}d old (aged A measured OOS -0.84%)"
            return m
    # --- TIER B: closed hard at the high on real volume, in a mover.
    #     Valid at ANY age, provided it is still near its level (the aged
    #     band is both an edge filter and the 15:20 API-time budget).
    elif (m["close_pos"] >= TIER_B_CLOSE_POS and rv >= TIER_B_RVOL
          and m["atr_pct"] >= TIER_B_ATR):
        ex = m["ext_pct"]
        if m["fresh"]:
            m["tier"] = "B"
        elif age >= 999:
            m["tier"] = None
            m["reject"] = "tier B but no cross of the level in 250d"
        elif ex == ex and AGED_EXT_MIN <= ex <= AGED_EXT_MAX:
            m["tier"] = "B"
        else:
            m["tier"] = None
            m["reject"] = (f"tier B, {age}d old, {ex:+.1f}% from level "
                           f"(band {AGED_EXT_MIN:+.0f}..{AGED_EXT_MAX:+.0f}%)")
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
                      partial_frac: float = 1.0,
                      dbg_symbol: str = "") -> dict | None:
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
    if _DBG_VMA and str(dbg_symbol or '').upper() in _DBG_VMA:
        vma_debug(prev, str(dbg_symbol))
    vma = _vma50(prev)
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
    # Reuse watchlist.compute_metrics rather than recomputing the trend here,
    # so the 08:45 shortlist and this scan can never disagree about a number.
    trend = {}
    try:
        from watchlist import compute_metrics as _cm
        trend = _cm(d, level, c) or {}
    except Exception:
        trend = {}
    m = dict(symbol="", close=c, level=level, which=which, gap_pct=gap,
             side=side, pre=pre,
             ret_12m=float(trend.get("ret_12m", float("nan"))),
             dist_200dma=float(trend.get("dist_200dma", float("nan"))),
             day_ret=(c / o - 1) * 100.0,
             close_pos=((c - lo) / rng) if rng > 0 else 0.5,
             rvol=(v / vma_cmp) if vma_cmp > 0 else float("nan"),
             vol_today=v, vma50=vma, vma_bars=int(_vma_n(prev)),
             atr_pct=atr_pct, turnover_cr=turnover, partial_frac=frac,
             # BUG 65: prior-exhaustion inputs, computed from bars BEFORE
             # today so the breakout itself is not counted as exhaustion.
             base_from_high=(
                 float((cl[-2] / np.max(hi[-253:-1]) - 1) * 100.0)
                 if len(cl) >= 253 and np.max(hi[-253:-1]) > 0
                 else float("nan")),
             ret_3m_prior=(
                 float((cl[-2] / cl[-64] - 1) * 100.0)
                 if len(cl) >= 64 and cl[-64] > 0 else float("nan")))

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
    # BUG 51 trend floor. NaN FAILS here rather than passing - unlike the
    # market-cap rule, "no 12-month history" is not an unknown to be given the
    # benefit of the doubt, it is a young listing with no established trend.
    # SWANDEF (no 12m history) lost on 03-Aug.
    r12 = m.get("ret_12m", float("nan"))
    if not (r12 == r12) or r12 < MIN_RET_12M:
        m["ok"] = False
        shown = f"{r12:.0f}%" if r12 == r12 else "unknown"
        m["reject"] = f"ret_12m {shown} < {MIN_RET_12M:.0f}% (no uptrend)"
        return m
    d200 = m.get("dist_200dma", float("nan"))
    if not (d200 == d200) or d200 < MIN_DIST_200DMA:
        m["ok"] = False
        shown = f"{d200:.0f}%" if d200 == d200 else "unknown"
        m["reject"] = f"dist200 {shown} < {MIN_DIST_200DMA:.0f}%"
        return m
    # BUG 65: the anticipation list is where the user's "extended rally"
    # objection actually landed - on 04-Aug two of its three picks (RAIN,
    # TFCILTD) had bases sitting within 2% of their 52-week high.
    spent = exhausted(m)
    if spent:
        m["ok"] = False
        m["reject"] = spent
        return m
    m["ok"] = True
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--debug-vma", default="",
                    help="comma-separated symbols: log the raw 50-bar volume "
                         "window used for rvol (BUG 72 instrumentation)")
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
    dbg_vma = {x.strip().upper() for x in (args.debug_vma or '').split(',') if x.strip()}
    if dbg_vma:
        globals()['_DBG_VMA'] = dbg_vma
        log.info('BUG 72: logging the raw volume window for %s', ', '.join(sorted(dbg_vma)))
    week_s = str(week_start_of(now.date()).date())

    # ---- BUG 55: THE ENTRY WINDOW IS HARD ---------------------------------
    # On 03-Aug the BTST message arrived stamped 15:48 IST and still said
    # "BUY NOW". That is not a tradeable instruction - the market shut at
    # 15:30. The job is scheduled for 15:20 but GitHub's scheduler is
    # best-effort and had queued it ~28 minutes late.
    #
    # The scan itself was not wrong; what was wrong is that it had no concept
    # of being too late. Nothing in the code compared the clock to the close,
    # so a delayed run produced a confident, unfillable alert - the worst
    # possible failure mode, because it looks exactly like a good one.
    #
    # The entry window is now explicit and enforced:
    #   before 15:00  too EARLY. Measured tier precision at 15:00 is only
    #                 70% vs 82% at 15:20 - the candle is still changing.
    #                 Run anyway (it may be a manual/backfill run) but the
    #                 message says the reading is provisional.
    #   15:00-15:30   the tradeable window. Normal behaviour.
    #   after 15:30   TOO LATE for a buy-at-close entry. The picks file is
    #                 still written and the alert is still sent - suppressing
    #                 it would hide the outage, which is BUG 49's lesson -
    #                 but every "BUY NOW" becomes "MISSED", and Model E must
    #                 not treat these as trades it took.
    #
    # --after-close is the deliberate post-close review and is exempt from
    # the late warning, because being after the close is its whole purpose.
    entry_open = dtime(15, 0)
    entry_close = dtime(15, 30)
    too_late = (not args.after_close) and now.time() > entry_close
    too_early = (not args.after_close) and now.time() < entry_open
    if too_late:
        log.error("RAN AT %s IST - AFTER THE %s CLOSE. The buy-at-close entry "
                  "is no longer available; picks are recorded as MISSED.",
                  f"{now:%H:%M}", f"{entry_close:%H:%M}")
    elif too_early:
        log.warning("running at %s IST, before the %s entry window - the "
                    "candle is still changing and the tier call is only ~70%% "
                    "reliable this early", f"{now:%H:%M}", f"{entry_open:%H:%M}")

    snaps = load_snapshots(cfg, week_s)
    if not snaps:
        # BUG 49: fail loudly. A BTST scan that finds nothing because the
        # snapshot is stale looks identical to a genuinely quiet day.
        from scan import report_stale_snapshot
        report_stale_snapshot(cfg, week_s, "BTST Scan")
        return 2
    if not cfg.secrets.dhan_access_token:
        log.error("DHAN_ACCESS_TOKEN not set")
        return 0

    # ---- BUG 43 (REVISED BY BUG 53): WHO IS A CANDIDATE --------------------
    # BUG 43 required the breakout to have happened TODAY, for both tiers.
    # BUG 53 re-measured that on the whole universe instead of on breakout
    # days only and found it half right - see the AGE_GATE_TIERS block at the
    # top of this file. Tier A must still be same-day. Tier B does not.
    #
    # So there are now TWO candidate pools:
    #
    #   FIRED   names whose alert fired TODAY. Cheap - they come straight from
    #           the alert state. Eligible for tier A or tier B.
    #
    #   AGED    names that did NOT alert today but may still be a tier B
    #           setup. These are not in the alert state at all (the best aged
    #           signals are ~55 days old and the state only tracks the current
    #           week), so they have to be found from the frozen snapshot level
    #           plus ONE bulk LTP call - the same BUG 52 trick the anticipation
    #           pass uses. Only names inside the aged band can ever pass
    #           classify(), so filter on the quote first and fetch history
    #           second. Eligible for tier B ONLY.
    #
    # The LTP call is shared with the anticipation pass below, so this adds
    # one bulk request, not two.
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
        log.info("%d name(s) broke out earlier this week; "
                 "tier B candidates only (BUG 53): %s", len(stale),
                 ", ".join(f"{sym}({bar})" for sym, bar in stale[:12]))

    # BUG 61: the defaults (timeout 30s, 5 retries with backoff to 30s) let a
    # SINGLE call burn 210 seconds and a single symbol - two calls - burn
    # seven minutes, which is the whole entry window. Inside a ten-minute
    # window a slow symbol is worth abandoning, not retrying: there are
    # hundreds of others and the tier gate needs a fast, complete candle.
    # The post-close review keeps the patient defaults.
    urgent = not (args.after_close or too_late)
    client = DhanClient(cfg.secrets.dhan_client_id, cfg.secrets.dhan_access_token,
                        data_rate=cfg.runtime.data_rate_per_sec,
                        quote_rate=cfg.runtime.quote_rate_per_sec,
                        timeout=8 if urgent else 30,
                        max_retries=2 if urgent else 5)
    # ---- BUG 69: THE BULK QUOTE MUST NOT INHERIT THE URGENT TIMEOUT -------
    # BUG 61 set timeout=8s / retries=2 while the entry window is live, which
    # is right for a PER-SYMBOL history call: there are hundreds of them and
    # a slow one is worth abandoning.
    #
    # It is wrong for the bulk quote. fetch_ohlc sends ~2,100 security ids as
    # 3 requests of up to 1000 instruments each; those payloads are large and
    # 8 seconds is aggressive. And unlike a per-symbol call this one is NOT
    # redundant - if it fails, `ltp_all` is empty, the aged tier B pool is
    # never built at all, and the scan silently degenerates to "names that
    # crossed today".
    #
    # That is the 07-Aug signature exactly:
    #     15:18 (urgent, 8s)  -> 7 candidates,  SBCL absent
    #     16:08 (not urgent)  -> 17 candidates, SBCL present as TIER B
    # SBCL crossed 814 on 07-Aug (06-Aug close 767.45 -> 920.90) and was
    # +13.1% above its level with close_pos 1.000 by 15:18, so it passed both
    # aged screens. Only an empty quote explains its absence.
    #
    # One request that gates an entire pass deserves patience; three of them
    # cost at most ~30s of a 480s budget. A separate client keeps the urgent
    # per-symbol behaviour untouched.
    quote_client = client if not urgent else DhanClient(
        cfg.secrets.dhan_client_id, cfg.secrets.dhan_access_token,
        data_rate=cfg.runtime.data_rate_per_sec,
        quote_rate=cfg.runtime.quote_rate_per_sec,
        timeout=30, max_retries=4)
    caps = load_mcap_table(cfg.paths["mcap"])
    start = now.date() - timedelta(days=HISTORY_DAYS)

    # BUG 67: count every reason a symbol is dropped. Six code paths used to
    # `return s, None` with at most a DEBUG line, so a name could disappear
    # from the scan with nothing in the log and nothing in the message. That
    # is how SBCL - already +8.19% at close_pos 1.000 by 15:15 - was missing
    # from the 15:18 list on 07-Aug.
    from collections import Counter
    drops = Counter()

    # ---- one bulk quote, reused by the aged pass and the anticipation pass -
    # BUG 59: this is now an OHLC call, not just LTP. Same cost (1 request per
    # 1000 names) but it also returns today's open/high/low, which is what
    # makes the close_pos pre-screen below possible.
    fired_syms = {s.symbol for s in fired}
    ohlc_all = {}
    try:
        ohlc_all = fetch_ohlc(quote_client, snaps)
    except DhanError as exc:
        log.warning("bulk ohlc failed (%s) - falling back to LTP only",
                    str(exc)[:120])
    try:
        ltp_all = ({k: v["last_price"] for k, v in ohlc_all.items()
                    if v.get("last_price")} if ohlc_all
                   else fetch_ltp(quote_client, snaps))
    except DhanError as exc:
        log.warning("bulk quote failed (%s) - aged tier B pass is SKIPPED "
                    "and only today's breakouts are scanned", str(exc)[:120])
        ltp_all = {}

    aged_pool = []
    quote_failed = False
    if ltp_all:
        for s in snaps:
            if s.symbol in fired_syms:
                continue
            px = ltp_all.get(s.symbol)
            lvl = s.entry_level
            if not px or px <= 0 or not lvl or lvl <= 0:
                continue
            ex = (px / lvl - 1) * 100.0
            if AGED_EXT_MIN <= ex <= AGED_EXT_MAX:
                aged_pool.append(s)
        in_band = len(aged_pool)

        # ---- BUG 59: SECOND, FREE SCREEN ON close_pos ----------------------
        # The band alone left 628 names on 04-Aug. At 2 history calls each and
        # the measured ~4 names/minute (BUG 52), 647 candidates is 2.6 HOURS -
        # the job was killed by the 30-minute timeout at 15:48, long after the
        # close. Widening the pool in BUG 53 without re-checking the time
        # budget is my error; the band was sized against the OLD pool.
        #
        # close_pos >= TIER_B_CLOSE_POS is MANDATORY for tier B and, since
        # BUG 54, for tier A too. It needs only today's high/low/LTP, which
        # the bulk OHLC call already returned. So it can be applied before any
        # history request, for free.
        #
        # LOSSLESS, verified on 157,995 stock-days: screening at 0.98 retains
        # 100.00% of names that ultimately qualify, while keeping only 5.5% of
        # the universe. A small margin is allowed because the quote and the
        # 5-minute reconstruction can disagree by a tick.
        if ohlc_all:
            margin = 0.02
            kept = []
            for s in aged_pool:
                q = ohlc_all.get(s.symbol)
                if not q:
                    kept.append(s)          # unknown -> do not silently drop
                    continue
                cp, _ = cheap_close_pos(q)
                if cp != cp or cp >= TIER_B_CLOSE_POS - margin:
                    kept.append(s)
            aged_pool = kept
        log.info("aged tier B pool: %d in the %+.0f..%+.0f%% band -> %d after "
                 "the close_pos>=%.2f pre-screen (of %d not fired today)",
                 in_band, AGED_EXT_MIN, AGED_EXT_MAX, len(aged_pool),
                 TIER_B_CLOSE_POS - 0.02, len(snaps) - len(fired))
    else:
        # BUG 68: the aged pass is built ONLY when the bulk quote returned
        # something. If it came back empty the pool is silently zero and the
        # scan degenerates to "names that crossed today" - which is how SBCL
        # (already +8.2% at close_pos 1.000, but ABOVE its level since the
        # 09:15 open, so never a fresh cross) was absent at 15:18 and present
        # at 16:08. That must be loud, not inferred from a small number.
        log.error("NO BULK QUOTE - the aged tier B pass was skipped entirely. "
                  "Only names that crossed their level TODAY are in scope; "
                  "a gap-up above the level is invisible to this run.")
        quote_failed = True

    # ---- BUG 59b: A HARD DEADLINE, NOT A TIMEOUT -------------------------
    # The workflow's timeout-minutes kills the job and produces NOTHING. That
    # is the worst outcome: no alert, no picks file, and no explanation. A
    # scan that has processed 400 of 600 names by 15:26 should SEND WHAT IT
    # HAS, not die at 15:48 with empty hands.
    #
    # Names are ordered so the ones that fired TODAY are always processed
    # first - they are the tier A candidates and the highest-value half of
    # the list - and the aged pool is truncated to whatever the remaining
    # budget allows.
    # BUG 60: THE DEADLINE ONLY APPLIES WHILE THE ENTRY IS STILL LIVE.
    # On a 16:15 run the 15:26 deadline is already in the past, so budget=0,
    # room=0, and the loop broke instantly - "checked 0 candles". That threw
    # away the whole point of BUG 55, which is that a late run must STILL
    # screen and STILL record its picks as MISSED so the day is auditable and
    # Model E can see what would have been taken. Racing a deadline that has
    # already passed is pointless; there is nothing left to protect.
    SEC_PER_NAME = 15.0 / 4.0      # BUG 52 measured ~4 names/minute
    deadline = now.replace(hour=15, minute=26, second=0, microsecond=0)
    enforce_deadline = (not args.after_close) and not too_late

    # ---- BUG 66: A POST-CLOSE RUN STILL NEEDS A BUDGET -------------------
    # BUG 60 disabled the 15:26 deadline once the entry window had closed -
    # correct, because racing a deadline that has already passed protects
    # nothing. But it left NO stopping rule at all in that mode, while the
    # workflow keeps a hard 14-minute timeout. On 05-Aug the 22:23 IST review
    # run took >18s per name (heavy rate limiting at that hour, vs the 3.19s
    # measured intraday), hit the timeout at 46 names, and was CANCELED:
    #
    #     16:53:28  23 fired + 23 aged; checking the candles ...
    #     17:07:22  ##[error]The operation was canceled.
    #
    # Canceled means no alert, no picks file, no explanation - exactly the
    # failure mode BUG 59b was written to eliminate, reintroduced by my own
    # BUG 60 fix in the one branch it did not cover.
    #
    # So the deadline is never "off"; it just moves. Off-hours it becomes a
    # wall-clock budget sized to finish inside the workflow timeout with room
    # to send the message and commit the CSV.
    if not enforce_deadline:
        deadline = now + timedelta(minutes=REVIEW_BUDGET_MIN)
    budget_s = max(0.0, (deadline - now).total_seconds())
    affordable = int(budget_s / SEC_PER_NAME) if budget_s > 0 else 0
    room = max(0, affordable - len(fired))
    if enforce_deadline and len(aged_pool) > room:
        # keep the aged names CLOSEST to their level - highest prior odds
        aged_pool.sort(key=lambda s: abs(
            (ltp_all.get(s.symbol, 0.0) / s.entry_level - 1) * 100.0
            if s.entry_level else 999))
        log.warning("time budget: %.0f min to %s leaves room for ~%d names; "
                    "trimming the aged pool %d -> %d (closest to level kept)",
                    budget_s / 60, f"{deadline:%H:%M}", affordable,
                    len(aged_pool), room)
        aged_pool = aged_pool[:room]

    if not fired and not aged_pool:
        log.info("no BTST candidates today - nothing broke out and nothing "
                 "is inside the aged band")
        return 0
    log.info("%d fired today + %d aged; checking the candles ... (%s)",
             len(fired), len(aged_pool),
             f"deadline {deadline:%H:%M}" if enforce_deadline
             else "post-close review - no deadline, recording as MISSED")

    def one(s):
        try:
            df = client.daily_candles(str(s.security_id), s.exchange_segment,
                                      start, now.date())
        except DhanError as exc:
            drops["daily error"] += 1
            log.debug("%s: %s", s.symbol, str(exc)[:100])
            return s, None
        if df.empty:
            drops["daily empty"] += 1
            return s, None

        frac = 1.0
        last = pd.Timestamp(df.iloc[-1]["datetime"]).date()
        if last != now.date():
            # Running BEFORE the close, so today's daily bar does not exist
            # yet. Build it from today's 5-minute candles and mark it partial.
            if args.after_close:
                drops["no bar (after-close)"] += 1
                return s, None
            m5 = None
            try:
                m5 = client.intraday_candles(
                    str(s.security_id), s.exchange_segment,
                    datetime.combine(now.date(), dtime(9, 15)),
                    now.replace(tzinfo=None), interval=5)
            except DhanError as exc:
                log.debug("%s intraday: %s", s.symbol, str(exc)[:100])
            if m5 is not None and not m5.empty:
                m5 = m5.sort_values("datetime")
                today_bar = {
                    "datetime": pd.Timestamp(now.date()),
                    "open": float(m5.iloc[0]["open"]),
                    "high": float(m5["high"].max()),
                    "low": float(m5["low"].min()),
                    "close": float(m5.iloc[-1]["close"]),
                    "volume": float(m5["volume"].sum()),
                }
                # BUG 70: from the CLOCK, never from the bar count.
                frac = session_fraction(now, len(m5))
                if frac >= 0.999 and len(m5) < BARS_PER_SESSION * 0.9:
                    drops["partial intraday after close"] += 1
            else:
                # ---- BUG 67: FALL BACK TO THE BULK QUOTE -------------------
                # Losing the intraday call used to DELETE the symbol. On
                # 07-Aug the 15:18 scan checked 7 candles while the 16:08
                # rerun checked 17 - SBCL was already +8.19% at close_pos
                # 1.000 by 15:15 and simply vanished, because intraday is
                # needed for EVERY name while the market is open and is the
                # first thing to rate-limit at 15:18.
                #
                # The bulk /marketfeed/ohlc response already holds today's
                # true open/high/low/LTP for all ~2,100 names, fetched in one
                # request. It is a complete substitute for the tier test -
                # only `volume` is missing, so rvol is unavailable and the
                # name can still make TIER A (day + close_pos) but not the
                # rvol-dependent TIER B. Losing the symbol entirely is
                # strictly worse than losing one gate.
                q = ohlc_all.get(s.symbol) or {}
                px = float(q.get("last_price") or 0.0)
                q_o = float(q.get("open") or 0.0)
                q_h = float(q.get("high") or 0.0)
                q_l = float(q.get("low") or 0.0)
                if px <= 0 or q_o <= 0:
                    drops["no intraday, no quote"] += 1
                    return s, None
                drops["quote fallback"] += 1
                q_v = float(q.get("volume") or 0.0)
                today_bar = {
                    "datetime": pd.Timestamp(now.date()),
                    "open": q_o, "high": max(q_h, px), "low": min(q_l or px, px),
                    "close": px, "volume": q_v,
                }
                # frac stays 1.0: the quote's volume is the full day so far,
                # exactly like the 5-minute sum it replaces.
                frac = 1.0
            df = pd.concat([df, pd.DataFrame([today_bar])], ignore_index=True)
        # BUG 64: trust the bulk quote for today's open/high/low
        df = repair_today_bar(df, ohlc_all.get(s.symbol), now.date())
        # A name whose alert fired today is age 0 by definition - trust the
        # alert rather than re-deriving it, so classify() and the alert can
        # never disagree about what "today" means. For the aged pool the age
        # is measured from the daily closes.
        age = 0 if s.symbol in fired_syms else breakout_age(df, s.entry_level)
        return s, classify(df, s.entry_level, partial_frac=frac, age=age,
                           dbg_symbol=s.symbol)

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
            # BUG 70: clock-based, same reason as the confirmed pass.
            frac = session_fraction(now, len(m5))
        # BUG 64: same repair for the anticipation pass
        df = repair_today_bar(df, ohlc_all.get(s.symbol), now.date())
        return df, frac

    # BUG 61: completion-order iteration with a hard wall clock. ex.map()
    # yielded in submission order, so one slow symbol blocked every deadline
    # check behind it and the loop could never break.
    rows = []
    todo = fired + aged_pool
    res, stopped_early = run_until(
        one, todo, cfg.runtime.max_workers, deadline, "btst")
    if stopped_early:
        log.warning("deadline %s reached - %d name(s) unchecked, sending what "
                    "is ready", f"{deadline:%H:%M}", stopped_early)
    for s, m in res:
        if not m:
            continue
        m["symbol"] = s.symbol
        m["level"] = s.entry_level
        m["mcap_cr"] = caps.get(s.symbol.upper())
        rows.append(m)

    # BUG 67: a symbol that never reached classify() is a BLIND SPOT, not a
    # rejection. Say so loudly - and put it in the message, because "7 candles
    # checked" out of 46 read like a quiet day rather than a data outage.
    missing = len(todo) - stopped_early - len(rows)
    if drops or missing:
        log.warning("coverage: %d/%d candidates produced a candle; drops: %s",
                    len(rows), len(todo) - stopped_early,
                    dict(drops) or "none")

    # ---- BUG 53c: THE TREND FLOOR NOW APPLIES TO THE CONFIRMED LIST TOO ----
    # BUG 51 added MIN_RET_12M / MIN_DIST_200DMA but wired them only into
    # classify_approach() - the anticipation path. The confirmed tiers never
    # checked them, so the 03-Aug losers the floor was written to stop
    # (BHAGCHEM ret_12m -9.8%, TBOTEK +8.0%, SWANDEF no history) could still
    # reach the BTST list through the tier route. That was an oversight, not a
    # decision. Every number quoted in this file's tier tables was measured
    # WITH the trend floor applied, so this makes the code match the study.
    # ---- BUG 57: THE TREND FLOOR IS REMOVED FROM THE CONFIRMED LIST -------
    # RETRACTION of BUG 53c, which added it here 3 days ago.
    #
    # WHY IT WAS ADDED. BUG 51 built the floor (ret_12m>=50, dist200>=25) from
    # the 03-Aug alerts, where the four clean losers had no established
    # uptrend. BUG 53c then noticed the floor was only wired into
    # classify_approach() and "fixed" the inconsistency by applying it to the
    # tier list too. That was reasoning by symmetry, not measurement.
    #
    # WHAT IT ACTUALLY DOES. On the TIER list (close_pos>=0.98 + rvol>=3), the
    # floor deletes the single best slice in the whole study:
    #
    #     removed by the floor   n=324   +3.173%   64.5% win   PF 3.56
    #                            P(next day >= +5%) = 30.2%
    #                            IS +3.405 -> OOS +2.985
    #
    # versus +2.074% for the list it was protecting. Every year positive
    # (2021 +4.93 ... 2026 +4.03), date-clustered t = 7.74, survives
    # de-duplication (n=301, +3.260%) and removing its top 10% (+1.638%).
    #
    # Confirmed list, with and without:
    #     with floor (BUG 53c)   830 rows  3.2/wk  +2.074%  P(+5%) 20.7%
    #     without    (now)     1,154 rows  4.5/wk  +2.382%  P(+5%) 23.4%
    # and the top-5/day portfolio DRAWDOWN IMPROVES, -28.0% -> -22.7%.
    #
    # WHY BUG 51 WAS NOT WRONG, ONLY MISAPPLIED. Its losers were SCAN alerts -
    # any 26W breakout. There the floor screens genuine junk. A BTST TIER pick
    # must already close in the top 2% of its range on 3x+ volume; those names
    # median rvol 12.97 and day_ret +14.7%. What the floor removes from THAT
    # pool is not junk, it is EARLY TREND: median ret_12m 24.5% instead of
    # 117.5%. A stock that has not yet run 50% has the most room left, which
    # is the same inversion BUG 54b found (ret_12m >= 400% has lift 0.35).
    #
    # THE FLOOR REMAINS IN classify_approach() (anticipation), where it was
    # measured and where the candle test is far weaker. Do not "unify" these
    # two again - that symmetry argument is what caused this bug.
    tiered = [r for r in rows if r.get("tier")]
    qualified = list(tiered)

    # ---- ranking: fresh tier A first, then aged tier B, then fresh tier B --
    # Measured on the top-5/day portfolio (per-trade net, 5 years):
    #     fresh tier A   +1.353%   49.5% win   PF 1.64   OOS +2.157%
    #     aged  tier B   +1.478%   62.7% win   PF 2.02   OOS +1.551%
    #     fresh tier B   +1.196%   54.6% win   PF 1.75   OOS +1.967%
    # Tier A leads on size and is the rarest, so it must never be crowded out
    # of the five slots. Aged tier B has the best win rate and PF and goes
    # second. Within a group, higher rvol first.
    #
    # Ranking barely matters at top-5 (rvol-only 51.7% win, this 51.9%) - the
    # cap is rarely binding. It is set explicitly so the order is a decision
    # and not an accident of sort stability.
    for r in qualified:
        r["conviction"], r["conv_why"] = conviction(r)

    def rank_key(r: dict) -> tuple:
        fresh = bool(r.get("fresh", True))
        tier = r.get("tier")
        grp = 0 if (fresh and tier == "A") else (1 if not fresh else 2)
        return (grp, -int(r.get("conviction", 0)),
                -float(r.get("rvol") or 0.0), -float(r.get("day_ret") or 0.0))

    qualified.sort(key=rank_key)
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
            # BUG 55: was this pick actually enterable? A run that finishes
            # after 15:30 cannot buy at today's close, and Model E must not
            # book a fill it could never have got. "missed" rows stay in the
            # file - deleting them would erase the evidence of the outage.
            "tradeable": 0 if too_late else 1,
            "rank": r["rank"], "symbol": r["symbol"], "tier": r["tier"],
            # BUG 53: age/arm are recorded so the paper ledger can score the
            # fresh and aged arms apart forever, instead of blending them.
            "age": int(r.get("age", 0)),
            "arm": ("fresh_A" if r.get("fresh") and r["tier"] == "A"
                    else "fresh_B" if r.get("fresh") else "aged_B"),
            "conviction": int(r.get("conviction", 0)),
            "ext_pct": round(float(r.get("ext_pct") or 0), 2),
            # BUG 62: "or 0" turned an unknown into a confident 0.0. A young
            # listing with no 12-month history is NOT a stock that returned
            # 0% - blank says "unknown", which is the truth.
            "ret_12m": _num(r.get("ret_12m"), 1),
            "dist_200dma": _num(r.get("dist_200dma"), 1),
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
    if too_late:
        when = "MISSED - ran after the close"
    elif partial or now.time() < entry_close:
        when = "buy into TODAY's close"
    else:
        when = "buy at close"
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
        # BUG 53e: exclude anything ALREADY TAKEN as a confirmed pick. Before
        # BUG 53 the two lists could not collide - confirmed was same-day
        # breakouts and anticipate was names that had not broken out. Now an
        # aged tier B name sits in both pools, and without this it would be
        # bought twice on the same day by Model E and Model F, which would
        # also double-count it in the ledger.
        fired_today = {x.symbol for x in fired}
        taken = {r["symbol"] for r in picks}
        pending = [s for s in snaps
                   if s.symbol not in fired_today and s.symbol not in taken]

        # ---- BUG 52: PRE-FILTER ON THE FROZEN LEVELS BEFORE ANY API CALL ----
        # BUG 47 widened this pool from "never fired" to the whole snapshot,
        # which is right for the MEASUREMENT but was ~2,100 names x 2 requests
        # each. On 03-Aug the job ran 42 minutes and only reached 171 of them
        # before it was cut short - so the top-5 was drawn from a fraction of
        # the universe and the ANTICIPATE list came back empty.
        #
        # A 15:20 job that finishes at 16:02 is useless however good its picks
        # are. The distance window is knowable from the FROZEN weekly levels
        # plus one bulk quote, with no per-symbol history, so filter first and
        # fetch second.
        #
        # Only names plausibly in range can ever pass classify_approach:
        #     within ANTICIPATE_NEAR below the nearer level, or
        #     within ANTICIPATE_ABOVE_MAX above the higher one.
        # A generous margin is applied because the quote is a snapshot and the
        # close can still move.
        MARGIN = 1.5
        # BUG 53: the quote was already fetched once for the aged tier B pool
        # above. Reuse it rather than paying for a second bulk request inside
        # the ten minutes before the close.
        ltp = ltp_all
        if not ltp:
            try:
                ltp = fetch_ltp(quote_client, pending)
            except DhanError as exc:
                log.warning("bulk quote failed (%s) - screening without a "
                            "pre-filter", str(exc)[:120])
                ltp = {}
        if ltp:
            def in_range(s):
                px = ltp.get(s.symbol)
                if px is None or px <= 0:
                    return False
                cand = [x for x in (s.entry_level, s.hi_short2) if x and x > 0]
                if not cand:
                    return False
                above = [x for x in cand if px <= x]
                if above:
                    lvl = min(above)
                    return (lvl - px) / lvl * 100.0 <= ANTICIPATE_NEAR + MARGIN
                lvl = max(cand)
                return (px / lvl - 1) * 100.0 <= ANTICIPATE_ABOVE_MAX + MARGIN
            before = len(pending)
            pending = [s for s in pending if in_range(s)]
            log.info("anticipation pre-filter: %d -> %d name(s) in range",
                     before, len(pending))
        # ---- BUG 60b: THE SAME FREE SCREEN, AND THE SAME DEADLINE ----------
        # BUG 59 protected the confirmed pass and left this one unbounded. On
        # 04-Aug it still queued 296 names x ~3.75s = ~18 minutes with no
        # stopping rule - the identical BUG 52/59 failure on the path I did
        # not fix. Three strikes now: any pool built from the snapshot MUST be
        # screened cheaply and MUST have a deadline.
        #
        # classify_approach() requires close_pos >= ANTICIPATE_CLOSE_POS, and
        # that comes free from the bulk OHLC call. Verified on 157,995
        # stock-days: screening at 0.90 retains 100.00% of names that finally
        # qualify while keeping 19.4% of the universe.
        if ohlc_all:
            before = len(pending)
            margin = 0.02
            pending = [s for s in pending
                       if (lambda cp: cp != cp or cp >= ANTICIPATE_CLOSE_POS - margin)(
                           cheap_close_pos(ohlc_all.get(s.symbol) or {})[0])]
            log.info("anticipation close_pos pre-screen: %d -> %d",
                     before, len(pending))
        # BUG 66: trim against whatever budget is left, in BOTH modes. The
        # deadline is now always set - 15:26 while the entry is live, or a
        # wall-clock review budget off-hours - so this must not be gated on
        # enforce_deadline any more.
        ant_budget = max(0.0, (deadline - datetime.now(IST)).total_seconds())
        ant_room = int(ant_budget / SEC_PER_NAME)
        if len(pending) > ant_room:
            log.warning("anticipation: %.1f min left allows ~%d names; "
                        "trimming %d -> %d", ant_budget / 60, ant_room,
                        len(pending), max(0, ant_room))
            pending = pending[:max(0, ant_room)]
        log.info("anticipation: screening %d name(s) (both sides of the "
                 "level) ...", len(pending))

        def one_ant(s):
            df, frac = candle(s)
            if df is None:
                return s, None
            return s, classify_approach(df, s.entry_level, s.hi_short2,
                                        partial_frac=frac,
                                        dbg_symbol=s.symbol)

        # BUG 61: same completion-order guard as the confirmed pass.
        ant_res, ant_stopped = run_until(
            one_ant, pending, cfg.runtime.max_workers,
            deadline, "anticipate")
        if ant_stopped:
            log.warning("anticipation: deadline %s reached, %d unchecked",
                        f"{deadline:%H:%M}", ant_stopped)
        for s, m in ant_res:
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
                # BUG 63: Model F must not book a fill that was never
                # available either. Same flag, same meaning as the confirmed
                # picks file.
                "tradeable": 0 if too_late else 1,
                "rank": r["rank"], "symbol": r["symbol"],
                "entry": round(float(r["close"]), 2),
                "pre": int(r.get("pre", 0)),
                "level": round(float(r["level"]), 2), "which": r["which"],
                "gap_pct": round(float(r["gap_pct"]), 2),
                "side": r.get("side", "below"),
                "ret_12m": _num(r.get("ret_12m"), 1),
                "dist_200dma": _num(r.get("dist_200dma"), 1),
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

    # ---- BUG 53d: THE MESSAGE MUST NOT CLAIM EVERYTHING BROKE OUT TODAY ----
    # The header used to say "broke out TODAY only" unconditionally. With aged
    # tier B in the list that would be a lie, and a name 55 days past its
    # cross presented as a fresh breakout is exactly the kind of thing that
    # gets acted on wrongly. Each pick now states its own age.
    n_fresh = sum(1 for r in picks if r.get("fresh"))
    n_aged = len(picks) - n_fresh
    # BUG 68: "N candle(s) checked" reports only SUCCESSES, so a small pool
    # and a broken feed look identical. On 07-Aug the 15:18 run said "7
    # candle(s) checked" and the 16:08 rerun said 17 - the difference was the
    # POOL (how many names were candidates at all), not lost candles, but the
    # message gave no way to tell. State the composition explicitly.
    lines = [f"🌙 <b>BTST — {now:%d-%b-%Y} {now:%H:%M} IST</b>",
             f"<i>{when}, exit tomorrow · {len(rows)} of {len(todo)} candidate(s) "
             f"screened ({len(fired)} broke out today + {len(aged_pool)} aged"
             f"{f', {len(snaps)} in snapshot' if not aged_pool else ''})</i>", ""]
    # BUG 55: a late run must say so at the TOP, before any price.
    if too_late:
        lines.insert(2, f"⛔ <b>TOO LATE — ran {now:%H:%M}, market closed "
                        f"{entry_close:%H:%M}</b>")
        lines.insert(3, "<i>These are NOT tradeable today. The buy-at-close "
                        "entry is gone; they are logged for the record only. "
                        "Do not chase them at tomorrow's open — that forfeits "
                        "the overnight move this model exists to capture.</i>")
        lines.insert(4, "")
    elif too_early:
        lines.insert(2, f"⏳ <b>PROVISIONAL — ran {now:%H:%M}, before "
                        f"{entry_open:%H:%M}</b>")
        lines.insert(3, "<i>The candle is still changing; only ~70% of names "
                        "flagged this early still qualify at the close. Wait "
                        "for the 15:20 list before acting.</i>")
        lines.insert(4, "")
    lines.insert(2 if not (too_late or too_early) else 5,
                 "🔥 <b>CONFIRMED — today's setups</b>")
    lines.insert(3 if not (too_late or too_early) else 6, "")
    if stopped_early:
        lines.append(f"<i>⏱ Scan stopped at the {deadline:%H:%M} deadline with "
                     f"{stopped_early} name(s) unchecked, so the list may be "
                     f"incomplete. Everything shown was fully screened.</i>")
        lines.append("")
    # BUG 68: a missing bulk quote silently disables the entire aged pass.
    if quote_failed:
        lines.append(
            "⚠️ <b>Quote feed unavailable — aged setups were NOT scanned.</b> "
            "<i>Only names that crossed their level today are in scope, so a "
            "stock that gapped above its level this morning is invisible to "
            "this run. Treat the list as partial.</i>")
        lines.append("")
    # BUG 67: never let a data outage read as a quiet market.
    if drops:
        lost = sum(v for k, v in drops.items() if k != "quote fallback")
        if lost:
            lines.append(
                f"⚠️ <b>{lost} of {len(todo)} candidate(s) returned no candle</b> "
                f"<i>— {', '.join(f'{k}: {v}' for k, v in drops.items() if k != 'quote fallback')}. "
                f"This list is INCOMPLETE; a setup may exist that was never "
                f"screened.</i>")
            lines.append("")
        if drops.get("quote fallback"):
            lines.append(
                f"<i>ℹ️ {drops['quote fallback']} name(s) judged from the bulk "
                f"quote (intraday feed unavailable) — no volume, so TIER B "
                f"could not be tested on them.</i>")
            lines.append("")
    if not picks:
        lines.append("<i>No setup qualified today. That is the normal case — "
                     "the tiers fire ~4 times a week combined.</i>")
    elif n_aged:
        lines.append(f"<i>{n_fresh} broke out today · {n_aged} already above "
                     f"the level (Tier B holds with age; Tier A does not).</i>")
        lines.append("")
    for r in picks:
        badge = ("🔥 <b>TIER A</b>" if r["tier"] == "A" else "⭐ <b>TIER B</b>")
        badge = f"<b>#{r['rank']}</b> {badge}"
        cap = f" <i>{r['mcap_cr']:,.0f}Cr</i>" if r.get("mcap_cr") else ""
        prov = " <i>(candle still forming)</i>" if float(
            r.get("partial_frac", 1.0)) < 0.999 else ""
        pre_tag = f" <b>{r['pre']}/8</b>" if r.get("pre") is not None else ""
        age = int(r.get("age", 0))
        age_tag = "" if r.get("fresh") else f" <i>· {age}d old</i>"
        where = ("above" if float(r.get("ext_pct") or 0) >= 0 else "below")
        when_txt = ("broke out today" if r.get("fresh")
                    else f"broke out {age} session(s) ago")
        # BUG 54b: conviction is about the SIZE of the move, not the odds of
        # a win. Labelled so it cannot be misread as "safest".
        cv = int(r.get("conviction", 0))
        cv_line = (f"    ⚡ <b>{cv}/{CONVICTION_MAX} explosive</b> "
                   f"<i>· {', '.join(r.get('conv_why') or []) or 'none'}"
                   f"{' · high tail, lower hit-rate' if cv >= 3 else ''}</i>")
        lines += [
            f"{badge}{pre_tag}  <b>{_esc(r['symbol'])}</b>  "
            f"{_fmt(r['close'])}{cap}{prov}{age_tag}",
            f"    day <b>{r['day_ret']:+.1f}%</b> · closed at "
            f"<b>{r['close_pos']*100:.0f}%</b> of range · "
            f"rvol <b>{r['rvol']:.1f}x</b>{_rvol_detail(r)} · atr {r['atr_pct']:.1f}%",
            f"    <i>{r['ext_pct']:+.1f}% {where} the 26W level "
            f"{_fmt(r['level'])}</i>",
            cv_line,
            (f"    ⛔ <b>MISSED ~{_fmt(r['close'])}</b> "
             f"<i>· {when_txt} · scan ran after the close, not tradeable</i>"
             if too_late else
             f"    <b>BUY NOW ~{_fmt(r['close'])}</b> "
             f"<i>· {when_txt} · exit tomorrow's close if &gt;+2%</i>"), ""]

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
                  f"PRE ≥{MIN_PRE_CONFIRM} · 12m ≥{MIN_RET_12M:.0f}% · "
                  f"≥{MIN_DIST_200DMA:.0f}% over 200DMA · "
                  f"{len(ant_rows)} screened</i>", ""]
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
                f"day {r['day_ret']:+.1f}% · rvol {r.get('rvol', 0):.1f}x"
                f"{_rvol_detail(r)}",
                f"    <i>12m {r.get('ret_12m', 0):+.0f}% · "
                f"{r.get('dist_200dma', 0):+.0f}% over the 200DMA</i>",
                # BUG 63: this line ignored too_late. At 20:43 the confirmed
                # list correctly read MISSED while the anticipation list still
                # said BUY NOW on the same dead entry - the exact failure
                # BUG 55 was written to stop, left in the one place it was not
                # applied.
                (f"    ⛔ <b>MISSED ~{_fmt(r['close'])}</b> "
                 f"<i>· scan ran after the close, not tradeable</i>"
                 if too_late else
                 f"    <b>BUY NOW ~{_fmt(r['close'])}</b> "
                 f"<i>· exit tomorrow's close</i>"), ""]
        if ant_dropped:
            lines.append(f"<i>{ant_dropped} more qualified, cap is top "
                         f"{ANTICIPATE_TOP_N}</i>")
        # BUG 58: re-measured. The old line claimed above-level BEAT below
        # (+0.82% vs +0.58%) - that was BUG 47's number, taken on a different
        # population (no PRE floor, no trend floor, any distance). Inside the
        # window this list actually screens, the ordering is REVERSED:
        #     BELOW the level  n=810   +1.325%  t  9.4  OOS +1.418%
        #     ABOVE the level  n=2,795 +0.886%  t 10.6  OOS +0.918%
        # The sort still puts above-level first and that is left alone: at
        # top-5 the cap almost never binds, so both orderings measure the
        # SAME (+0.964%/trade, 55.9% win). Only the false claim is removed.
        lines.append("<i>close_pos≥0.90 + 12m≥50% + 200DMA≥25% measured "
                     "+0.99%/trade (t 13.7, n=3,605), +1.03% out of sample. "
                     "🔭 below-level beat 🚀 above (+1.33% vs +0.89%). "
                     "Most still do not run — it pays through the tail.</i>")

    # BUG 58: these footers quoted the ORIGINAL tier study (+1.75%/+0.83%,
    # "win rate ~50%"). Those numbers described the pre-BUG-53/54/57 rule and
    # were left untouched through three threshold changes, so the message was
    # under-reporting the shipped rule by more than half and claiming a win
    # rate 13 points below the real one. Re-measured on the CURRENT rule
    # (close_pos>=0.98 both tiers, aged tier B, no trend floor on the tier
    # path), 892,121 tradeable stock-days, entry at the 15:20 price, exit at
    # tomorrow's close, net 0.22%:
    #     TIER A (fresh)  n=302   1.2/wk  57.0% win  +3.044%  t  6.4  OOS +3.809%
    #     TIER B          n=852   3.3/wk  65.0% win  +2.148%  t 11.5  OOS +2.450%
    #     the list        n=1154  4.5/wk  62.9% win  +2.382%  t 12.8  OOS +2.760%
    # Any future threshold change MUST update these three lines - a regression
    # test now pins them to the live constants.
    lines += ["", "━━━━━━━━━━━━━━━━━━━━",
        "<i>Tier A measured +3.04%/trade (t 6.4, n=302, 57% win) over 5 years; "
        "Tier B +2.15% (t 11.5, n=852, 65% win). Out of sample +3.81% / "
        "+2.45%. Roughly 3 of every 5 win, and the average is carried by the "
        "tail — one in five gains 5%+ overnight. Exit at tomorrow's close.</i>",
        "<i>Reality check: ~31 stocks a week jump 5%+ across the market and "
        "this list flags ~4.5, catching about 3% of them. Missing most big "
        "movers is the price of the hit rate, not a fault.</i>"]
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
    code = main()
    # BUG 61: HARD EXIT.
    # Abandoning a slow HTTP request is not enough. ThreadPoolExecutor
    # registers an atexit hook that JOINS its worker threads, and those
    # threads are non-daemon, so the interpreter blocks on the way out even
    # after shutdown(wait=False, cancel_futures=True) has returned. Measured:
    # a worker sleeping 60s holds the process for the full 60s at exit.
    #
    # That is precisely the observed hang - the scan finishes, the alert is
    # sent, and the job still sits there until the workflow timeout kills it,
    # which then reports FAILURE for a run that actually succeeded.
    #
    # Everything that matters (Telegram, picks CSV) is already flushed to the
    # network/disk by this point, so there is nothing to lose by not waiting
    # for a stuck socket read.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code if isinstance(code, int) else 0)
