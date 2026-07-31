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

    m = dict(
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--all", action="store_true",
                    help="list every breakout with its tier, not just A/B")
    ap.add_argument("--after-close", action="store_true",
                    help="only judge a COMPLETED daily candle; skips names "
                         "whose bar is still forming. Use for a post-close "
                         "review - the 15:20 job must NOT set this.")
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

    # Only names that ALERTED this week are candidates - BTST is a follow-up to
    # a real breakout, not an independent screen.
    state = AlertState(cfg.paths["state"])
    fired = [s for s in snaps if state.already_alerted(week_s, s.symbol)]
    if not fired:
        log.info("nothing fired this week - no BTST candidates")
        return 0
    log.info("%d names fired this week; checking today's candle ...", len(fired))

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

    rows = []
    with ThreadPoolExecutor(max_workers=max(1, cfg.runtime.max_workers)) as ex:
        for s, m in ex.map(one, fired):
            if not m:
                continue
            m["symbol"] = s.symbol
            m["level"] = s.entry_level
            m["mcap_cr"] = caps.get(s.symbol.upper())
            rows.append(m)

    picks = [r for r in rows if r.get("tier")]
    picks.sort(key=lambda r: (r["tier"], -r["day_ret"]))
    log.info("checked %d candles, %d qualified", len(rows), len(picks))

    if rows:
        pd.DataFrame(rows).to_csv(f"btst_{now:%Y-%m-%d}.csv", index=False)

    partial = [r for r in picks if float(r.get("partial_frac", 1.0)) < 0.999]
    when = "buy into TODAY's close" if partial or now.time() < dtime(15, 30) \
        else "buy at close"
    lines = [f"🌙 <b>BTST — {now:%d-%b-%Y} {now:%H:%M} IST</b>",
             f"<i>{when}, exit tomorrow · {len(rows)} breakouts checked</i>", ""]
    if not picks:
        lines.append("<i>No setup qualified today. That is the normal case — "
                     "the tiers fire ~2-4 times a week combined.</i>")
    for r in picks:
        badge = "🔥 <b>TIER A</b>" if r["tier"] == "A" else "⭐ <b>TIER B</b>"
        cap = f" <i>{r['mcap_cr']:,.0f}Cr</i>" if r.get("mcap_cr") else ""
        prov = " <i>(candle still forming)</i>" if float(
            r.get("partial_frac", 1.0)) < 0.999 else ""
        lines += [
            f"{badge}  <b>{_esc(r['symbol'])}</b>  {_fmt(r['close'])}{cap}{prov}",
            f"    day <b>{r['day_ret']:+.1f}%</b> · closed at "
            f"<b>{r['close_pos']*100:.0f}%</b> of range · "
            f"rvol <b>{r['rvol']:.1f}x</b> · atr {r['atr_pct']:.1f}%",
            f"    <i>{r['ext_pct']:+.1f}% above the 26W level "
            f"{_fmt(r['level'])}</i>", ""]

    if args.all:
        rest = [r for r in rows if not r.get("tier")]
        if rest:
            lines += [f"<i>no tier ({len(rest)}): "
                      + ", ".join(_esc(r["symbol"]) for r in rest[:40]) + "</i>", ""]

    lines += [
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
