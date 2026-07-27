"""
Paper trading — simulates real positions on the exact live entry signal.

RULES
-----
Entry   the same 5-minute candle the scanner alerts on (identical code path)
Size    fixed rupee notional per stock (default Rs 1,00,000), whole shares
Stop    entry candle LOW minus a 0.02% buffer, checked intrabar on every
        subsequent 5m candle low
Target  none - the position runs until a 5-minute candle CLOSES below its
        9-period EMA (EMA of 5m closes, seeded and stepped exactly like Pine)

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


@dataclass
class PaperTrade:
    symbol: str
    week: str
    entry_date: str
    entry_time: str
    entry: float
    qty: int
    invested: float
    bar_low: float
    stop: float
    level_26w: float
    level_52w: float
    trigger: str
    rsi: float
    macd_hist: float

    exit_date: str = ""
    exit_time: str = ""
    exit: float = 0.0
    exit_reason: str = "OPEN"
    bars_held: int = 0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    mfe_pct: float = 0.0
    mae_pct: float = 0.0
    r_multiple: float = 0.0        # profit / initial risk per share


def simulate(sig, bars_after: pd.DataFrame, capital: float) -> PaperTrade:
    """
    Walk every 5m candle after entry and apply the two exit rules.

    `bars_after` must be ascending 5m candles strictly AFTER the entry bar.
    """
    entry = float(sig.price)
    qty = int(capital // entry)

    # `bar_low` was added to Signal for paper trading. Use getattr so a stale
    # strategy.py (partial deploy) degrades to an entry-based stop instead of
    # crashing the whole report with AttributeError.
    bar_low = float(getattr(sig, "bar_low", 0.0) or 0.0)

    # Guard against a malformed candle (bad tick, or low >= close). The stop
    # must always sit BELOW the entry, otherwise it would trigger instantly and
    # the R-multiple explodes. Fall back to the buffer applied to the entry.
    if not (0.0 < bar_low < entry):
        bar_low = entry
    stop = bar_low * (1.0 - SL_BUFFER)
    risk_per_share = max(entry - stop, entry * SL_BUFFER)

    t = PaperTrade(
        symbol=sig.symbol,
        week=sig.week_start,
        entry_date=sig.bar_time.strftime("%Y-%m-%d"),
        entry_time=sig.bar_time.strftime("%H:%M"),
        entry=entry,
        qty=qty,
        invested=qty * entry,
        bar_low=bar_low,
        stop=stop,
        level_26w=float(sig.entry_level),
        level_52w=float(sig.level_52),
        trigger=sig.trigger,
        rsi=float(sig.evaluation.values.get("rsi", float("nan"))),
        macd_hist=float(sig.evaluation.values.get("macd_hist", float("nan"))),
    )
    if qty == 0 or bars_after.empty:
        t.exit_reason = "NO_FILL" if qty == 0 else "OPEN"
        return t

    # 9-EMA of 5m closes. Seed from candles up to and including the entry bar
    # so the EMA is already "warm", exactly as it would be on a live chart.
    closes = list(bars_after["close"].astype(float))
    seed = ind.ema(np.array([entry] * EMA_LEN, dtype=float), EMA_LEN)
    ema_prev = float(seed[-1])
    alpha = 2.0 / (EMA_LEN + 1.0)

    hi = lo = entry
    for i, (_, bar) in enumerate(bars_after.iterrows(), start=1):
        low = float(bar["low"])
        high = float(bar["high"])
        close = float(bar["close"])
        hi, lo = max(hi, high), min(lo, low)

        # --- stop first: conservative when both trigger on the same candle
        if low <= stop:
            t.exit = stop
            t.exit_reason = "SL"
            t.bars_held = i
            ts = pd.Timestamp(bar["datetime"])
            t.exit_date, t.exit_time = ts.strftime("%Y-%m-%d"), ts.strftime("%H:%M")
            break

        ema_now = alpha * close + (1.0 - alpha) * ema_prev
        if close < ema_now:
            t.exit = close
            t.exit_reason = "EMA9"
            t.bars_held = i
            ts = pd.Timestamp(bar["datetime"])
            t.exit_date, t.exit_time = ts.strftime("%Y-%m-%d"), ts.strftime("%H:%M")
            break
        ema_prev = ema_now
    else:
        # never exited - mark to the last available close
        last = bars_after.iloc[-1]
        t.exit = float(last["close"])
        t.exit_reason = "OPEN"
        t.bars_held = len(bars_after)
        ts = pd.Timestamp(last["datetime"])
        t.exit_date, t.exit_time = ts.strftime("%Y-%m-%d"), ts.strftime("%H:%M")

    t.pnl = (t.exit - entry) * qty
    t.pnl_pct = (t.exit / entry - 1.0) * 100.0
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

    print("\n" + "=" * 132)
    print("  PAPER TRADES".ljust(60) + f"Rs {capital:,.0f} per stock".rjust(72))
    print("=" * 132)
    hdr = (f"{'symbol':<11}{'entry date':<11}{'in':>6}{'entry':>10}{'qty':>6}"
           f"{'SL':>10}{'26W lvl':>10}  {'exit date':<11}{'out':>6}{'exit':>10}"
           f"  {'why':<5}{'bars':>5}{'P&L Rs':>11}{'P&L %':>8}{'R':>7}")
    print(hdr)
    print("-" * 132)
    for t in trades:
        print(f"{t.symbol:<11}{t.entry_date:<11}{t.entry_time:>6}{t.entry:>10.2f}"
              f"{t.qty:>6}{t.stop:>10.2f}{t.level_26w:>10.2f}  "
              f"{t.exit_date:<11}{t.exit_time:>6}{t.exit:>10.2f}  "
              f"{t.exit_reason:<5}{t.bars_held:>5}{t.pnl:>11,.0f}"
              f"{t.pnl_pct:>+8.2f}{t.r_multiple:>+7.2f}")

    closed = [t for t in trades if t.exit_reason in ("SL", "EMA9")]
    openx = [t for t in trades if t.exit_reason == "OPEN"]
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    total = sum(t.pnl for t in trades)
    deployed = sum(t.invested for t in trades)

    print("\n" + "=" * 132)
    print("  SUMMARY")
    print("=" * 132)
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
    ap.add_argument("--csv", default=None)
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
        wk = start_week
        while wk <= end_week:
            hist = d[d["_day"] < wk]
            snap = build_snapshot(sym, sid, seg, hist, cfg.strategy, wk) if not hist.empty else None
            if snap is not None:
                wbars = five[(five["_day"] >= wk) & (five["_day"] < wk + pd.Timedelta(days=7))]
                if not wbars.empty:
                    res = replay_week(snap, cfg.strategy, wbars)
                    for sig in res.signals:
                        after = five[pd.to_datetime(five["datetime"]) >
                                     pd.Timestamp(sig.bar_time)]
                        trades.append(simulate(sig, after, args.capital))
                        found += 1
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
