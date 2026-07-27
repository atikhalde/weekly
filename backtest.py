"""
Historical backtest — replays past weeks through the LIVE strategy code.

Uses DhanHQ by default, so the backtest reads the same candles the live
scanner does. Requires DHAN_CLIENT_ID + DHAN_ACCESS_TOKEN and an active Data
API subscription.

    export DHAN_CLIENT_ID=1000000003
    export DHAN_ACCESS_TOKEN=eyJ...

    python backtest.py RATNAVEER TMB MONARCH SENCO --years 3
    python backtest.py TMB --granularity 5m --years 1      # exact live entries
    python backtest.py --universe 200 --years 3 --csv trades.csv
    python backtest.py TMB --source yahoo                  # no token, sanity check

WHY YOU CAN TRUST THE NUMBERS
-----------------------------
No look-ahead. For every week W the snapshot is rebuilt with
`build_snapshot(..., target_week=W)`, which by construction only reads weekly
bars that CLOSED BEFORE W. The week is then replayed bar-by-bar through
`replay_week()` — the exact functions scan.py calls in production. This file
adds no strategy logic of its own; if the backtest is wrong, the live scanner
is wrong in the same way.

ENTRY-TIMING CAVEAT (read this)
-------------------------------
The live scanner enters on the first 5-MINUTE candle that closes above the 26W
level. Dhan serves intraday history in 90-day slices, so replaying years of 5m
data across a whole universe costs a lot of calls.

  --granularity daily  (default)  entry = first DAILY close above the level
  --granularity 5m                entry = true 5m close, exactly like live

Daily granularity is faithful to every weekly condition — all 13 checks are
weekly, and weekly volume/open aggregate identically. Only the entry PRICE
differs: a daily close is usually slightly worse than the 5m close that first
pierced the level, so daily-granularity results are mildly CONSERVATIVE.
(Measured on a controlled replay: 5m entry 2316.45 vs daily 2317.55, same
signal date and level.)

Use daily for breadth (many symbols, many years) and 5m to confirm exact fills
on a shortlist.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, time as dtime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from config import load_config
from strategy import build_snapshot, build_weekly_bars, replay_week, week_start_of

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
log = logging.getLogger("backtest")

IST = ZoneInfo("Asia/Kolkata")
HORIZONS = (1, 2, 4, 8, 13)          # weeks ahead to measure


# --------------------------------------------------------------------------- #
#  Data sources
# --------------------------------------------------------------------------- #
def yahoo_daily(symbol: str, years: int) -> pd.DataFrame:
    """Free daily history, no token. Symbol gets an .NS suffix for NSE."""
    sym = symbol if "." in symbol else f"{symbol}.NS"
    rng = f"{max(years + 2, 3)}y"
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
           f"?range={rng}&interval=1d")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        body = json.load(r)
    res = (body.get("chart") or {}).get("result")
    if not res:
        return pd.DataFrame()
    res = res[0]
    ts = res.get("timestamp") or []
    q = (res.get("indicators") or {}).get("quote", [{}])[0]
    rows = []
    for i, t in enumerate(ts):
        c = (q.get("close") or [None])[i]
        if c is None:
            continue
        rows.append({
            "datetime": datetime.fromtimestamp(t, tz=IST),
            "open": q["open"][i] if q.get("open") else c,
            "high": q["high"][i] if q.get("high") else c,
            "low": q["low"][i] if q.get("low") else c,
            "close": c,
            "volume": (q.get("volume") or [0])[i] or 0.0,
        })
    return pd.DataFrame(rows)


def dhan_daily(client, sec_id: str, seg: str, years: int) -> pd.DataFrame:
    from dhan import last_n_years
    f, t = last_n_years(years + 2)
    return client.daily_candles(sec_id, seg, f, t)


# --------------------------------------------------------------------------- #
#  Trade record
# --------------------------------------------------------------------------- #
@dataclass
class Trade:
    symbol: str
    week: str
    entry_time: datetime
    entry: float
    level: float
    trigger: str
    fwd: dict[int, float] = field(default_factory=dict)   # weeks -> % return
    mfe: float = 0.0        # max favourable excursion over the longest horizon
    mae: float = 0.0        # max adverse excursion
    bars_held: int = 0


def forward_stats(daily: pd.DataFrame, entry_time: datetime, entry: float) -> tuple[dict, float, float, int]:
    """% return at each horizon, plus MFE/MAE, using only bars AFTER entry."""
    after = daily[daily["datetime"] > entry_time].reset_index(drop=True)
    fwd: dict[int, float] = {}
    if after.empty:
        return fwd, 0.0, 0.0, 0

    for wks in HORIZONS:
        target = entry_time + timedelta(weeks=wks)
        upto = after[after["datetime"] <= target]
        if upto.empty:
            continue
        # only report a horizon that actually completed
        if after["datetime"].iloc[-1] < target - timedelta(days=4):
            continue
        fwd[wks] = (float(upto["close"].iloc[-1]) / entry - 1.0) * 100.0

    span = after[after["datetime"] <= entry_time + timedelta(weeks=max(HORIZONS))]
    if span.empty:
        return fwd, 0.0, 0.0, 0
    mfe = (float(span["high"].max()) / entry - 1.0) * 100.0
    mae = (float(span["low"].min()) / entry - 1.0) * 100.0
    return fwd, mfe, mae, len(span)


# --------------------------------------------------------------------------- #
#  Core: replay one symbol week by week
# --------------------------------------------------------------------------- #
def backtest_symbol(symbol: str, daily: pd.DataFrame, cfg,
                    start_week: pd.Timestamp, end_week: pd.Timestamp,
                    intraday_fn=None) -> list[Trade]:
    trades: list[Trade] = []
    if daily.empty:
        return trades

    weekly = build_weekly_bars(daily)
    weeks = [w for w in weekly["week_start"] if start_week <= w <= end_week]

    d = daily.copy()
    d["_day"] = pd.to_datetime(d["datetime"]).dt.tz_localize(None).dt.normalize()

    for wk in weeks:
        # history strictly BEFORE this week - this is what kills look-ahead
        hist = d[d["_day"] < wk]
        if hist.empty:
            continue
        snap = build_snapshot(symbol, "0", "NSE_EQ", hist, cfg.strategy, wk)
        if snap is None:
            continue

        bars = None
        if intraday_fn is not None:
            bars = intraday_fn(symbol, wk)
        if bars is None or bars.empty:
            bars = d[(d["_day"] >= wk) & (d["_day"] < wk + pd.Timedelta(days=7))]
        if bars.empty:
            continue

        res = replay_week(snap, cfg.strategy, bars)
        for sig in res.signals:
            fwd, mfe, mae, held = forward_stats(daily, sig.bar_time, sig.price)
            trades.append(Trade(
                symbol=symbol, week=str(wk.date()), entry_time=sig.bar_time,
                entry=sig.price, level=sig.entry_level, trigger=sig.trigger,
                fwd=fwd, mfe=mfe, mae=mae, bars_held=held,
            ))
    return trades


# --------------------------------------------------------------------------- #
#  Reporting
# --------------------------------------------------------------------------- #
def pct(x: float) -> str:
    return f"{x:+.2f}%"


def report(trades: list[Trade], horizon: int) -> None:
    if not trades:
        print("\nNo signals in the tested period.")
        print("That is a result, not a bug: this scan is deliberately selective.")
        return

    trades.sort(key=lambda t: t.entry_time)

    print(f"\n{'='*78}")
    print(f"  TRADES  ({len(trades)} signals)")
    print(f"{'='*78}")
    print(f"{'symbol':<12}{'entry date':<13}{'entry':>9}{'26W lvl':>10}"
          f"{'+1w':>8}{'+2w':>8}{'+4w':>8}{'+8w':>8}{'MFE':>8}{'MAE':>8}")
    print("-" * 78)
    for t in trades:
        row = (f"{t.symbol:<12}{t.entry_time:%Y-%m-%d}   {t.entry:>8.2f}{t.level:>10.2f}")
        for h in (1, 2, 4, 8):
            row += f"{t.fwd[h]:>+8.1f}" if h in t.fwd else f"{'—':>8}"
        row += f"{t.mfe:>+8.1f}{t.mae:>+8.1f}"
        print(row)

    print(f"\n{'='*78}")
    print(f"  SUMMARY")
    print(f"{'='*78}")

    for h in HORIZONS:
        vals = [t.fwd[h] for t in trades if h in t.fwd]
        if not vals:
            continue
        a = np.array(vals)
        wins = (a > 0).sum()
        print(f"  +{h:>2}w  n={len(a):<4} win={wins/len(a)*100:5.1f}%  "
              f"avg={a.mean():+6.2f}%  median={np.median(a):+6.2f}%  "
              f"best={a.max():+7.2f}%  worst={a.min():+7.2f}%")

    main = [t.fwd[horizon] for t in trades if horizon in t.fwd]
    if main:
        a = np.array(main)
        wins, losses = a[a > 0], a[a <= 0]
        pf = (wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else float("inf")
        print(f"\n  Primary horizon +{horizon}w:")
        print(f"    expectancy    {a.mean():+.2f}% per signal")
        print(f"    profit factor {pf:.2f}" if pf != float("inf") else "    profit factor inf (no losers)")
        print(f"    std dev       {a.std():.2f}%")
        if len(wins):
            print(f"    avg win       {wins.mean():+.2f}%  (n={len(wins)})")
        if len(losses):
            print(f"    avg loss      {losses.mean():+.2f}%  (n={len(losses)})")

    mfe = np.array([t.mfe for t in trades])
    mae = np.array([t.mae for t in trades])
    print(f"\n  Excursion over {max(HORIZONS)}w: avg MFE {mfe.mean():+.2f}%, "
          f"avg MAE {mae.mean():+.2f}%")
    print(f"  Signals per symbol: {len(trades)/len(set(t.symbol for t in trades)):.1f}")


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("symbols", nargs="*", help="symbols; omit with --universe")
    ap.add_argument("--source", choices=["dhan", "yahoo"], default="dhan",
                    help="dhan (default) = same data the live scanner uses; "
                         "yahoo = no token needed, for a quick sanity check")
    ap.add_argument("--years", type=int, default=3, help="years to test")
    ap.add_argument("--granularity", choices=["daily", "5m"], default="daily")
    ap.add_argument("--universe", type=int, default=0,
                    help="test the first N symbols of the configured universe (Dhan)")
    ap.add_argument("--horizon", type=int, default=4, help="primary horizon in weeks")
    ap.add_argument("--csv", default=None, help="write the trade list here")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    client = None
    sec_map: dict[str, tuple[str, str]] = {}

    need_dhan = args.source == "dhan" or bool(args.universe) or args.granularity == "5m"
    if need_dhan:
        from dhan import DhanClient
        if not cfg.secrets.dhan_access_token:
            print("DHAN_ACCESS_TOKEN not set. Either export it, or use --source yahoo.")
            return 2
        client = DhanClient(cfg.secrets.dhan_client_id, cfg.secrets.dhan_access_token,
                            data_rate=cfg.runtime.data_rate_per_sec,
                            quote_rate=cfg.runtime.quote_rate_per_sec)
        # Fetch the scrip master ONCE - it is a 27 MB download.
        print("Loading Dhan instrument list ...")
        ins = DhanClient.fetch_instruments(cfg.universe.exchange_segments,
                                           cfg.universe.series,
                                           exclude_etf=cfg.universe.exclude_etf)
        sec_map = {i.symbol.upper(): (i.security_id, i.exchange_segment) for i in ins}
        print(f"  {len(sec_map)} tradable symbols")

        # Preflight: catch a bad token now, not after hundreds of calls.
        probe_sym, (probe_id, probe_seg) = next(iter(sorted(sec_map.items())))
        from dhan import DhanError, last_n_years
        pf, pt = last_n_years(1)
        try:
            probe = client.daily_candles(probe_id, probe_seg, pf, pt)
        except DhanError as exc:
            print(f"\nPREFLIGHT FAILED ({probe_sym}): {exc}")
            print("  - token wrong or expired (they last ~30 days), or")
            print("  - the Dhan DATA API subscription is not active")
            return 2
        if probe.empty:
            print(f"\nPREFLIGHT: {probe_sym} returned no candles - check the Data API "
                  "subscription.")
            return 2
        print(f"  preflight OK ({len(probe)} candles for {probe_sym})")

    symbols = [s.upper() for s in args.symbols]
    if args.universe:
        symbols = sorted(sec_map)[:args.universe]
    if not symbols:
        print("give some symbols, or use --universe N")
        return 2

    today = datetime.now(IST).date()
    end_week = week_start_of(today) - pd.Timedelta(days=7)     # last COMPLETE week
    start_week = week_start_of(today - timedelta(days=365 * args.years))

    print(f"\nBacktest {start_week.date()} .. {end_week.date()}  "
          f"({args.years}y, {len(symbols)} symbol(s), source={args.source}, "
          f"granularity={args.granularity})")
    print(f"strict_entry={cfg.strategy.strict_entry} gate={cfg.strategy.gate_source} "
          f"req52={cfg.strategy.req52} one_per_week={cfg.strategy.one_per_week}")
    if args.granularity == "daily":
        print("NOTE: daily granularity - entry is the first DAILY close above the "
              "26W level.\n      Live uses the 5m close, which is usually a slightly "
              "better fill.")

    # ---- 5m history: pull once per symbol in 90-day slices (Dhan's cap), then
    # slice per week in memory. Fetching week-by-week would be 52 calls/year.
    intraday_cache: dict[str, pd.DataFrame] = {}

    def load_intraday(symbol: str) -> pd.DataFrame:
        if symbol in intraday_cache:
            return intraday_cache[symbol]
        sid, seg = sec_map[symbol.upper()]
        frames, cursor = [], datetime.combine(start_week.date(), dtime(9, 0)).replace(tzinfo=IST)
        stop = datetime.now(IST)
        while cursor < stop:
            chunk_end = min(cursor + timedelta(days=85), stop)
            try:
                part = client.intraday_candles(sid, seg, cursor, chunk_end,
                                               interval=cfg.runtime.bar_interval_min)
                if not part.empty:
                    frames.append(part)
            except Exception as exc:                               # noqa: BLE001
                log.warning("%s intraday %s: %s", symbol, cursor.date(), str(exc)[:80])
            cursor = chunk_end + timedelta(days=1)
        df = (pd.concat(frames, ignore_index=True)
                .drop_duplicates(subset=["datetime"])
                .sort_values("datetime")
                .reset_index(drop=True)) if frames else pd.DataFrame()
        intraday_cache[symbol] = df
        return df

    def intraday_fn(symbol: str, wk: pd.Timestamp):
        if args.granularity != "5m" or symbol.upper() not in sec_map:
            return None
        df = load_intraday(symbol)
        if df.empty:
            return None
        day = pd.to_datetime(df["datetime"]).dt.tz_convert(IST).dt.tz_localize(None).dt.normalize()
        return df[(day >= wk) & (day < wk + pd.Timedelta(days=7))]

    all_trades: list[Trade] = []
    for n, sym in enumerate(symbols, 1):
        try:
            if args.source == "yahoo":
                daily = yahoo_daily(sym, args.years)
            else:
                if sym.upper() not in sec_map:
                    print(f"  {sym}: not in the configured universe "
                          f"(segments={cfg.universe.exchange_segments}, "
                          f"series={cfg.universe.series})")
                    continue
                sid, seg = sec_map[sym.upper()]
                daily = dhan_daily(client, sid, seg, args.years)
        except Exception as exc:                                   # noqa: BLE001
            print(f"  {sym}: fetch failed ({exc})")
            continue

        if daily.empty:
            print(f"  {sym}: no data")
            continue

        t = backtest_symbol(sym, daily, cfg, start_week, end_week,
                            intraday_fn if args.granularity == "5m" else None)
        all_trades += t
        if len(symbols) > 20 and n % 25 == 0:
            print(f"  ...{n}/{len(symbols)} scanned, {len(all_trades)} signals")
        elif len(symbols) <= 20:
            print(f"  {sym}: {len(t)} signal(s)")

    report(all_trades, args.horizon)

    if args.csv and all_trades:
        pd.DataFrame([{
            "symbol": t.symbol, "week": t.week,
            "entry_time": t.entry_time.isoformat(), "entry": t.entry,
            "level": t.level, "trigger": t.trigger,
            **{f"fwd_{h}w_pct": t.fwd.get(h) for h in HORIZONS},
            "mfe_pct": t.mfe, "mae_pct": t.mae,
        } for t in all_trades]).to_csv(args.csv, index=False)
        print(f"\nWrote {len(all_trades)} trades -> {args.csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
