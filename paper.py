"""
Paper trading — simulates real positions on the exact live entry signal.

RULES
-----
Signal  the same 5-minute candle the scanner alerts on (identical code path)
Size    fixed rupee notional per stock (default Rs 1,00,000), whole shares
Stop    signal candle LOW minus a 0.02% buffer
Target  none - the position runs until a 5-minute candle CLOSES below its
        9-period EMA (EMA of 5m closes, warmed on real pre-entry candles)

EXECUTION MODEL - why the timestamps look "one bar later"
---------------------------------------------------------
NSE 5m candles are stamped by their OPEN time: a bar labelled 10:15 spans
10:15:00-10:19:59 and only completes at 10:20:00.

You therefore cannot trade the signal bar's own close. The condition is only
known once that bar has finished, so the earliest realistic fill is the OPEN of
the NEXT bar. This module models that explicitly:

    signal bar   10:15  (closes 10:20)  <- condition detected here
    entry fill   10:20 open             <- what you could actually get

The same applies to the EMA exit: a close below the EMA is only known at the
bar's end, so the exit fills at the next bar's open. Filling at the signal
bar's own close would be look-ahead and would flatter every result.

Stops are different - a resting SL-M order triggers intrabar, so a stop fills
at the stop price when touched. On a gap down (bar opens below the stop) the
fill is the OPEN, which is worse; that slippage is modelled.

Set --fill close-of-signal-bar to reproduce the older, optimistic behaviour.

Both exits are evaluated on every bar after entry, stop first: if a candle both
breaches the stop and closes under the EMA, the stop wins. That is the
conservative reading - within a single candle you cannot know which came first.

    python paper.py RATNAVEER TMB MONARCH SENCO --weeks 8
    python paper.py --from-snapshot --weeks 4 --csv paper_trades.csv
    python paper.py TMB --capital 200000 --source yahoo

Uses the live strategy modules, so entries match the alerts candle-for-candle.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, time as dtime, timedelta

import numpy as np
import pandas as pd

import indicators as ind
from config import load_config
from dhan import IST, DhanClient, DhanError, last_n_years
from strategy import build_snapshot, replay_week, week_start_of

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
log = logging.getLogger("paper")

SL_BUFFER = 0.0002          # 0.02% below the entry candle low
EMA_LEN = 9                 # exit EMA on 5m closes

# --------------------------------------------------------------------------- #
#  INTRADAY MODE (added 28-Jul-2026 at the user's request: "only intraday").
#
#  MIS/intraday positions on NSE are force-squared-off by the broker in the
#  afternoon. Nothing may be carried overnight, so every trade MUST close on
#  its own session. SQUARE_OFF is the last 5m bar we are willing to hold into;
#  anything still open is closed at that bar's close and marked "EOD".
#
#  15:15 is the conservative choice: most brokers auto-square MIS equity
#  between 15:10 and 15:20, and forced liquidation gets a worse price than a
#  voluntary exit. Holding to 15:25 risks the broker closing it for you.
#
#  COST_ROUND_TRIP is the all-in cost of a round trip as a PERCENT of turnover:
#  brokerage + STT + exchange charges + GST + stamp duty + slippage. Measured
#  sweep (see INTRADAY_FINDINGS.md): the strategy's edge is entirely consumed
#  somewhere between 0.40% and 0.50%, so this number is not cosmetic - it
#  decides whether the system makes money at all. 0.22% is a realistic
#  discount-broker figure INCLUDING ~0.05% slippage per side.
# --------------------------------------------------------------------------- #
SQUARE_OFF = "15:15"        # hard intraday exit; no overnight positions
COST_ROUND_TRIP = 0.22      # % of turnover, buy+sell all-in


@dataclass
class PaperTrade:
    symbol: str
    week: str
    signal_date: str            # candle that produced the signal
    signal_time: str            # its OPEN stamp (closes 5 min later)
    signal_close: float         # close of that candle (the breakout price)
    entry_date: str             # bar we could actually buy on
    entry_time: str
    entry: float                # its OPEN = realistic fill
    qty: int
    invested: float
    bar_low: float
    stop: float
    level_26w: float
    level_52w: float
    trigger: str
    rsi: float
    macd_hist: float
    slippage_pct: float = 0.0   # entry vs signal close

    exit_date: str = ""
    exit_time: str = ""
    exit: float = 0.0
    exit_reason: str = "OPEN"
    exit_note: str = ""         # e.g. "gap" when a stop filled below itself
    bars_held: int = 0
    pnl: float = 0.0               # NET of costs
    pnl_pct: float = 0.0           # NET of costs
    gross_pnl: float = 0.0         # before costs
    costs: float = 0.0             # round-trip brokerage+taxes+slippage
    mfe_pct: float = 0.0
    mae_pct: float = 0.0
    r_multiple: float = 0.0        # profit / initial risk per share


def _stamp(ts) -> tuple[str, str]:
    ts = pd.Timestamp(ts)
    return ts.strftime("%Y-%m-%d"), ts.strftime("%H:%M")


def simulate(sig, bars_after: pd.DataFrame, capital: float,
             bars_before: pd.DataFrame | None = None,
             fill: str = "signal-close") -> PaperTrade:
    """
    Model a realistic fill on the live breakout signal.

    `bars_after`  ascending 5m candles strictly AFTER the signal bar.
    `bars_before` candles up to AND INCLUDING the signal bar, used to warm the
                  9-EMA on real history.
    `fill`        "signal-close" (default) entry at the signal candle's close.
                               This is what the Pine indicator marks with its
                               BUY arrow, so the report lines up with the chart.
                  "next-open"  entry at the following bar's OPEN - what a market
                               order sent on the alert would realistically get.
                               Use it to measure expected slippage.

    Timestamps follow NSE convention: a bar stamped 10:15 spans 10:15-10:19:59.
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

    # ---- ENTRY -------------------------------------------------------------
    if fill == "signal-close":
        entry = signal_close
        exit_scan = bars_after
    else:
        first = bars_after.iloc[0]
        entry = float(first["open"])
        e_d, e_t = _stamp(first["datetime"])
        t.entry_date, t.entry_time = e_d, e_t
        # the entry bar itself can still stop us out, so include it
        exit_scan = bars_after

    t.entry = entry
    t.slippage_pct = (entry / signal_close - 1.0) * 100.0
    qty = int(capital // entry)
    t.qty = qty
    t.invested = qty * entry
    if qty == 0:
        t.exit_reason = "NO_FILL"
        t.exit_note = "price above capital per stock"
        return t

    risk_per_share = max(entry - stop, entry * SL_BUFFER)

    # ---- 9-EMA warmed on real pre-signal candles ---------------------------
    alpha = 2.0 / (EMA_LEN + 1.0)
    ema_prev: float | None = None
    if bars_before is not None and len(bars_before) >= EMA_LEN:
        hist = np.asarray(bars_before["close"].astype(float))[-(EMA_LEN * 10):]
        series = ind.ema(hist, EMA_LEN)
        if not np.isnan(series[-1]):
            ema_prev = float(series[-1])
    warmup = 0
    if ema_prev is None:
        ema_prev = entry
        warmup = EMA_LEN

    # ---- walk forward ------------------------------------------------------
    hi = lo = entry
    pending_ema_exit = False        # EMA breach seen; fill on the NEXT open
    rows = list(exit_scan.iterrows())

    for i, (_, bar) in enumerate(rows, start=1):
        o = float(bar["open"]); high = float(bar["high"])
        low = float(bar["low"]); close = float(bar["close"])

        bar_d, bar_t = _stamp(bar["datetime"])

        # --- INTRADAY: never hold past the session, never hold overnight.
        # Checked FIRST so it cannot be skipped by a later branch. If the bar
        # is on a later date the position should already have closed, so we
        # square off defensively at this bar's open.
        if bar_d != sig_d:
            t.exit = o
            t.exit_reason = "EOD"
            t.exit_note = "forced square-off: next session reached"
            t.bars_held = i
            t.exit_date, t.exit_time = bar_d, bar_t
            break
        if bar_t >= SQUARE_OFF:
            t.exit = close
            t.exit_reason = "EOD"
            t.exit_note = f"intraday square-off at {SQUARE_OFF}"
            t.bars_held = i
            t.exit_date, t.exit_time = bar_d, bar_t
            break

        # An EMA breach detected on the previous bar fills at THIS bar's open.
        if pending_ema_exit:
            t.exit = o
            t.exit_reason = "EMA9"
            t.bars_held = i
            t.exit_date, t.exit_time = _stamp(bar["datetime"])
            break

        hi, lo = max(hi, high), min(lo, low)

        # --- stop: a resting SL-M triggers intrabar.
        if low <= stop:
            if o <= stop:                      # gapped through - fill at open
                t.exit = o
                t.exit_note = "gap through stop"
            else:
                t.exit = stop
            t.exit_reason = "SL"
            t.bars_held = i
            t.exit_date, t.exit_time = _stamp(bar["datetime"])
            break

        ema_now = alpha * close + (1.0 - alpha) * ema_prev
        if i > warmup and close < ema_now:
            if fill == "signal-close":
                t.exit = close                 # optimistic mode
                t.exit_reason = "EMA9"
                t.bars_held = i
                t.exit_date, t.exit_time = _stamp(bar["datetime"])
                break
            pending_ema_exit = True            # realistic: exit next open
        ema_prev = ema_now
    else:
        last = rows[-1][1]
        t.exit = float(last["close"])
        t.exit_reason = "OPEN"
        t.exit_note = "still running at end of data"
        t.bars_held = len(rows)
        t.exit_date, t.exit_time = _stamp(last["datetime"])

    # ---- P&L, NET OF COSTS -------------------------------------------------
    # Intraday edges are small enough that gross P&L is misleading. Costs are
    # charged on both legs, so they scale with turnover, not with profit.
    t.gross_pnl = (t.exit - entry) * qty
    turnover = (entry + t.exit) * qty
    t.costs = turnover * (COST_ROUND_TRIP / 100.0) / 2.0
    t.pnl = t.gross_pnl - t.costs
    t.pnl_pct = (t.exit / entry - 1.0) * 100.0 - COST_ROUND_TRIP
    t.mfe_pct = (hi / entry - 1.0) * 100.0
    t.mae_pct = (lo / entry - 1.0) * 100.0
    t.r_multiple = (t.exit - entry) / risk_per_share
    return t


# --------------------------------------------------------------------------- #
#  Data
# --------------------------------------------------------------------------- #
def fetch_5m(client, sec_id: str, seg: str, start: datetime, end: datetime,
             interval: int) -> pd.DataFrame:
    """5m candles across a long window, in Dhan's 90-day slices."""
    frames, cursor = [], start
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=85), end)
        try:
            part = client.intraday_candles(sec_id, seg, cursor, chunk_end,
                                           interval=interval)
            if not part.empty:
                frames.append(part)
        except DhanError as exc:
            log.warning("intraday %s: %s", cursor.date(), str(exc)[:80])
        cursor = chunk_end + timedelta(days=1)
    if not frames:
        return pd.DataFrame()
    return (pd.concat(frames, ignore_index=True)
              .drop_duplicates(subset=["datetime"])
              .sort_values("datetime")
              .reset_index(drop=True))


# --------------------------------------------------------------------------- #
#  Report
# --------------------------------------------------------------------------- #
def money(x: float) -> str:
    return f"{x:>12,.0f}"


def report(trades: list[PaperTrade], capital: float) -> None:
    if not trades:
        print("\nNo trades in the tested period.")
        return

    trades.sort(key=lambda t: (t.entry_date, t.entry_time))

    print("\n" + "=" * 150)
    print("  PAPER TRADES".ljust(78) + f"Rs {capital:,.0f} per stock".rjust(72))
    print("=" * 150)
    hdr = (f"{'symbol':<11}{'date':<11}{'signal':>7}{'sigClose':>10}"
           f"{'entry':>7}{'fill':>10}{'qty':>6}{'SL':>10}{'26W lvl':>10}  "
           f"{'exit date':<11}{'exit':>7}{'price':>10}  {'why':<5}{'bars':>5}"
           f"{'P&L Rs':>11}{'P&L %':>8}{'R':>7}")
    print(hdr)
    print("-" * 150)
    for t in trades:
        print(f"{t.symbol:<11}{t.signal_date:<11}{t.signal_time:>7}"
              f"{t.signal_close:>10.2f}{t.entry_time:>7}{t.entry:>10.2f}"
              f"{t.qty:>6}{t.stop:>10.2f}{t.level_26w:>10.2f}  "
              f"{t.exit_date:<11}{t.exit_time:>7}{t.exit:>10.2f}  "
              f"{t.exit_reason:<5}{t.bars_held:>5}{t.pnl:>11,.0f}"
              f"{t.pnl_pct:>+8.2f}{t.r_multiple:>+7.2f}")

    closed = [t for t in trades if t.exit_reason in ("SL", "EMA9", "EOD")]
    openx = [t for t in trades if t.exit_reason == "OPEN"]
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    total = sum(t.pnl for t in trades)
    deployed = sum(t.invested for t in trades)

    print("\n" + "=" * 150)
    print("  SUMMARY")
    print("=" * 150)
    print(f"  trades            {len(trades)}   (closed {len(closed)}, still open {len(openx)})")
    print(f"  capital per stock Rs {capital:,.0f}      total deployed Rs {deployed:,.0f}")
    print(f"  net P&L           Rs {total:,.0f}   ({total/deployed*100 if deployed else 0:+.2f}% on deployed)")
    print(f"  win rate          {len(wins)/len(trades)*100:.1f}%   ({len(wins)}W / {len(losses)}L)")
    if wins:
        print(f"  avg win           Rs {sum(t.pnl for t in wins)/len(wins):,.0f}"
              f"   ({sum(t.pnl_pct for t in wins)/len(wins):+.2f}%)")
    if losses:
        print(f"  avg loss          Rs {sum(t.pnl for t in losses)/len(losses):,.0f}"
              f"   ({sum(t.pnl_pct for t in losses)/len(losses):+.2f}%)")
    gross_w = sum(t.pnl for t in wins)
    gross_l = abs(sum(t.pnl for t in losses))
    if gross_l:
        print(f"  profit factor     {gross_w/gross_l:.2f}")
    print(f"  expectancy        Rs {total/len(trades):,.0f} per trade")
    rs = [t.r_multiple for t in trades]
    print(f"  avg R multiple    {sum(rs)/len(rs):+.2f}R")
    print(f"  avg bars held     {sum(t.bars_held for t in trades)/len(trades):.0f}"
          f"  (~{sum(t.bars_held for t in trades)/len(trades)*5/60:.1f} trading hours)")

    by_reason: dict[str, list[PaperTrade]] = {}
    for t in trades:
        by_reason.setdefault(t.exit_reason, []).append(t)
    print("\n  exits:")
    for reason, group in sorted(by_reason.items()):
        pnl = sum(x.pnl for x in group)
        print(f"    {reason:<6} n={len(group):<4} P&L Rs {pnl:>12,.0f}   "
              f"avg {pnl/len(group):>10,.0f}")

    print(f"\n  avg MFE {sum(t.mfe_pct for t in trades)/len(trades):+.2f}%   "
          f"avg MAE {sum(t.mae_pct for t in trades)/len(trades):+.2f}%")


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("symbols", nargs="*")
    ap.add_argument("--from-snapshot", action="store_true",
                    help="use every symbol in weekly_snapshot.csv")
    ap.add_argument("--weeks", type=int, default=8, help="weeks of history to replay")
    ap.add_argument("--capital", type=float, default=100000.0,
                    help="rupees deployed per stock (default 1,00,000)")
    ap.add_argument("--source", choices=["dhan", "yahoo"], default="dhan")
    ap.add_argument("--fill", choices=["signal-close", "next-open"],
                    default="signal-close",
                    help="signal-close (default): fill at the close of the "
                         "signal candle - matches the indicator's BUY arrow. "
                         "next-open: fill at the next bar's open, which is what "
                         "a market order placed on the alert would realistically "
                         "get; use it to size expected slippage.")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--include-deferred", action="store_true",
                    help="also paper-trade deferred entries (default: cross "
                         "only; alerts always send both)")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
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

    if not cfg.secrets.dhan_access_token:
        print("DHAN_ACCESS_TOKEN not set - 5m history requires Dhan")
        return 2
    client = DhanClient(cfg.secrets.dhan_client_id, cfg.secrets.dhan_access_token,
                        data_rate=cfg.runtime.data_rate_per_sec,
                        quote_rate=cfg.runtime.quote_rate_per_sec)

    print("Loading instrument list ...")
    ins = DhanClient.fetch_instruments(cfg.universe.exchange_segments,
                                       cfg.universe.series,
                                       exclude_etf=cfg.universe.exclude_etf)
    sec_map = {i.symbol.upper(): (i.security_id, i.exchange_segment) for i in ins}

    today = datetime.now(IST).date()
    end_week = week_start_of(today)
    start_week = end_week - pd.Timedelta(weeks=args.weeks)
    from_date, to_date = last_n_years(cfg.runtime.history_years)

    print(f"Paper trading {start_week.date()} .. {today}  "
          f"({len(symbols)} symbol(s), Rs {args.capital:,.0f}/stock)")
    print(f"  entry  = live 5m breakout signal")
    print(f"  stop   = entry candle low - {SL_BUFFER*100:.2f}%")
    print(f"  exit   = first 5m close below the {EMA_LEN}-EMA")
    print(f"  scope  = {'ALL signals' if args.include_deferred else 'CROSS entries only (deferred excluded)'}")

    trades: list[PaperTrade] = []
    for n, sym in enumerate(symbols, 1):
        if sym not in sec_map:
            print(f"  {sym}: not in the universe")
            continue
        sid, seg = sec_map[sym]
        try:
            daily = client.daily_candles(sid, seg, from_date, to_date)
        except DhanError as exc:
            print(f"  {sym}: daily fetch failed ({str(exc)[:60]})")
            continue
        if daily.empty:
            continue

        five = fetch_5m(client, sid, seg,
                        datetime.combine(start_week.date(), dtime(9, 0)).replace(tzinfo=IST),
                        datetime.now(IST), interval)
        if five.empty:
            print(f"  {sym}: no 5m data")
            continue
        five["_day"] = (pd.to_datetime(five["datetime"]).dt.tz_convert(IST)
                        .dt.tz_localize(None).dt.normalize())

        d = daily.copy()
        d["_day"] = pd.to_datetime(d["datetime"]).dt.tz_localize(None).dt.normalize()

        found = 0
        busy_until = None   # per-symbol: no overlapping positions
        wk = start_week
        while wk <= end_week:
            hist = d[d["_day"] < wk]
            snap = build_snapshot(sym, sid, seg, hist, cfg.strategy, wk) if not hist.empty else None
            if snap is not None:
                wbars = five[(five["_day"] >= wk) & (five["_day"] < wk + pd.Timedelta(days=7))]
                if not wbars.empty:
                    res = replay_week(snap, cfg.strategy, wbars)
                    for sig in res.signals:
                        # Real capital cannot hold two positions in one symbol,
                        # so ignore a signal that fires while the previous
                        # trade is still open.
                        if busy_until is not None and \
                                pd.Timestamp(sig.bar_time) <= busy_until:
                            continue
                        # Paper trading takes CROSS entries only. A deferred
                        # fill enters hours after the breakout, at a worse and
                        # more extended price, so including it distorts the
                        # performance picture. Alerts still send BOTH - this
                        # filter is about measurement, not notification.
                        if not getattr(args, "include_deferred", False) \
                                and sig.trigger != "cross":
                            continue
                        ts = pd.to_datetime(five["datetime"])
                        after = five[ts > pd.Timestamp(sig.bar_time)]
                        before = five[ts <= pd.Timestamp(sig.bar_time)]
                        tr = simulate(sig, after, args.capital, before, args.fill)
                        trades.append(tr)
                        found += 1
                        if tr.exit_date:
                            busy_until = pd.Timestamp(
                                f"{tr.exit_date} {tr.exit_time}").tz_localize(IST)
            wk += pd.Timedelta(days=7)

        if len(symbols) <= 25:
            print(f"  {sym}: {found} trade(s)")
        elif n % 25 == 0:
            print(f"  ...{n}/{len(symbols)}, {len(trades)} trades")

    report(trades, args.capital)

    if args.csv and trades:
        pd.DataFrame([t.__dict__ for t in trades]).to_csv(args.csv, index=False)
        print(f"\nWrote {len(trades)} trades -> {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
