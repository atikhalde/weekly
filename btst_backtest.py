#!/usr/bin/env python3
"""
BTST BACKTEST - replay the LIVE tier logic over history and show every trade.

WHY THIS EXISTS
The alert says "Tier A measured +3.04%/trade (t 6.4, n=302)". Those numbers came
from throwaway analysis scripts in /tmp that no longer exist and were never in
the repo. There has been no way to ask "show me the actual trades behind that",
which is exactly the question worth asking before trusting a number.

This reproduces those figures from the SHIPPED CODE and prints the trade list.

-----------------------------------------------------------------------------
THE ONE RULE THAT MAKES THIS TRUSTWORTHY
-----------------------------------------------------------------------------
It imports btst.classify(), btst.exhausted(), btst.conviction() and every
threshold from btst.py. It does NOT reimplement them.

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
  * classify() sees daily bars up to and including D, never beyond
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

    python btst_backtest.py                      # last 2 years, summary
    python btst_backtest.py --years 5 --trades   # every trade, printed
    python btst_backtest.py --symbol SBCL        # one stock's history
    python btst_backtest.py --from 2026-08-01    # a specific window
    python btst_backtest.py --csv out.csv        # full trade list to CSV
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
                if len(df) < 300:
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
#  REPLAY - one symbol
# --------------------------------------------------------------------------- #
def replay_symbol(path: str, start: str, end: str) -> list[dict]:
    """
    Every day this symbol would have produced a BTST pick, with its outcome.

    The level is rebuilt weekly from bars strictly before the current week,
    which is what the Monday snapshot freezes. classify() then sees only bars
    up to that day.
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
    for i in range(26, n_wk):
        lvl_by_week[i] = wk_hi[i - 26:i].max()

    close = d.close.to_numpy(float)
    lo_i = int(np.searchsorted(dt.values, np.datetime64(start)))
    hi_i = int(np.searchsorted(dt.values, np.datetime64(end), side="right"))
    lo_i = max(lo_i, 300)
    hi_i = min(hi_i, len(d) - 1)          # need D+1 for the outcome

    out = []
    for j in range(lo_i, hi_i):
        w = wi[j]
        if w is None or (isinstance(w, float) and np.isnan(w)):
            continue
        level = lvl_by_week[int(w)]
        if not np.isfinite(level) or level <= 0:
            continue

        hist = d.iloc[:j + 1]                       # bars up to and incl. D
        age = btst.breakout_age(hist, float(level))
        m = btst.classify(hist, float(level), partial_frac=1.0, age=age)
        if not m or not m.get("tier"):
            continue

        conv, why = btst.conviction(m)
        entry = close[j]
        nxt = close[j + 1]
        gross = (nxt / entry - 1) * 100.0
        out.append(dict(
            date=dt.iloc[j].date().isoformat(), symbol=sym, tier=m["tier"],
            arm=("fresh_A" if m.get("fresh") and m["tier"] == "A"
                 else "fresh_B" if m.get("fresh") else "aged_B"),
            age=int(m.get("age", 0)), entry=round(entry, 2),
            exit=round(nxt, 2), level=round(float(level), 2),
            day_ret=round(m["day_ret"], 2),
            close_pos=round(m["close_pos"], 3),
            rvol=round(float(m.get("rvol") or 0), 2),
            atr_pct=round(m["atr_pct"], 2),
            ext_pct=round(float(m.get("ext_pct") or 0), 2),
            ret_12m=btst._num(m.get("ret_12m"), 1),
            conviction=conv, why=";".join(why),
            gross_pct=round(gross, 3),
            net_pct=round(gross - COST_ROUND_TRIP, 3),
        ))
    return out


def _rs(args):
    try:
        return replay_symbol(*args)
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
        return f"{label:<20}{'-':>8}"
    v = x.net_pct
    t = v.mean() / (v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 2 else float("nan")
    wks = max((pd.to_datetime(x.date).max() - pd.to_datetime(x.date).min()).days / 7, 1)
    return (f"{label:<20}{len(v):>8,}{len(v)/wks:>7.1f}{(v>0).mean()*100:>8.1f}"
            f"{v.mean():>+10.3f}{v.median():>+9.2f}{_pf(v):>7.2f}{t:>7.1f}"
            f"{(v>=5).mean()*100:>8.1f}{v.min():>9.1f}{v.max():>9.1f}")


HDR = (f"{'slice':<20}{'trades':>8}{'/wk':>7}{'win%':>8}{'mean':>10}"
       f"{'med':>9}{'PF':>7}{'t':>7}{'P(+5%)':>8}{'worst':>9}{'best':>9}")


def report(df: pd.DataFrame, show_trades: bool, top: int) -> None:
    if df.empty:
        print("\nNo BTST setups in this window.")
        return
    df = df.sort_values("date").reset_index(drop=True)

    print("\n" + "=" * 108)
    print("BTST BACKTEST - the LIVE rule replayed over history")
    print("=" * 108)
    print(f"window      {df.date.min()} .. {df.date.max()}")
    # NOT the universe size - the number of names that ever produced a setup.
    # Labelling it "symbols" implied only 412 stocks were scanned.
    print(f"names hit   {df.symbol.nunique():,} distinct symbols produced a setup")
    print(f"cost        {COST_ROUND_TRIP}% round trip, entry = daily close, "
          f"exit = next close")
    print(f"thresholds  TIER_A day>={btst.TIER_A_DAY:g}% cp>={btst.TIER_A_CLOSE_POS} | "
          f"TIER_B cp>={btst.TIER_B_CLOSE_POS} rvol>={btst.TIER_B_RVOL:g} "
          f"atr>={btst.TIER_B_ATR:g}%")
    print(f"            aged band {btst.AGED_EXT_MIN:+g}..{btst.AGED_EXT_MAX:+g}%, "
          f"exhaustion base<{btst.MAX_BASE_FROM_HIGH:g}% & 3m<={btst.MAX_RET_3M_PRIOR:g}%")

    print("\n" + HDR)
    print(_stats(df, "ALL"))
    print()
    for arm in ("fresh_A", "fresh_B", "aged_B"):
        print(_stats(df[df.arm == arm], f"  {arm}"))
    print()
    for tier in ("A", "B"):
        print(_stats(df[df.tier == tier], f"  TIER {tier}"))
    print()
    for c in sorted(df.conviction.unique()):
        print(_stats(df[df.conviction == c], f"  conviction {c}/4"))

    print("\nBY YEAR")
    print(HDR)
    for y, g in df.assign(y=pd.to_datetime(df.date).dt.year).groupby("y"):
        print(_stats(g, f"  {y}"))

    # a top-5/day book, which is what actually gets traded
    print("\nTOP-5 PER DAY (the traded book: fresh A first, then aged, by rvol)")
    r = df.copy()
    r["_k"] = ((r.arm == "fresh_A") * 200 + (r.arm == "aged_B") * 100
               + r.rvol.clip(0, 50))
    book = r.sort_values(["date", "_k"], ascending=[True, False]).groupby("date").head(5)
    print(HDR)
    print(_stats(book, "  top-5/day"))
    day = book.groupby("date").net_pct.mean()
    eq = (1 + day / 100).cumprod()
    dd = (eq / eq.cummax() - 1).min() * 100
    print(f"\n  active days {len(day):,}   mean day {day.mean():+.3f}%   "
          f"max drawdown {dd:.1f}%   worst day {day.min():+.2f}%")

    if show_trades:
        print("\n" + "=" * 108)
        print(f"TRADES (most recent {top})")
        print("=" * 108)
        print(f"{'date':<12}{'symbol':<13}{'tier':<5}{'arm':<9}{'age':>4}"
              f"{'entry':>10}{'exit':>10}{'day%':>7}{'rvol':>7}{'cv':>4}{'net%':>8}")
        for r_ in df.tail(top).itertuples():
            print(f"{r_.date:<12}{r_.symbol:<13}{r_.tier:<5}{r_.arm:<9}{r_.age:>4}"
                  f"{r_.entry:>10.2f}{r_.exit:>10.2f}{r_.day_ret:>7.1f}"
                  f"{r_.rvol:>7.1f}{r_.conviction:>4}{r_.net_pct:>+8.2f}")

    print("\n" + "-" * 108)
    print("Entry is the DAILY CLOSE; the live scan buys ~15:20 on a partial "
          "candle (~0.14% cheaper), so this is mildly pessimistic.")
    print("Yahoo daily data, not Dhan - per-symbol volume can differ, so "
          "rvol-gated TIER B counts may not match live exactly.")
    print("One regime (2021-2026). Past results are not forward validation.")


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
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

    print(f"BTST backtest  {start} .. {end}   {len(syms):,} symbol(s)")
    print(f"cache: {args.data}")

    if not args.no_fetch:
        have = len(glob.glob(os.path.join(args.data, "*.csv")))
        if have < len(syms):
            print(f"fetching daily history ({have} cached) ...")
            t0 = time.time()
            got = fetch_history(syms, args.data)
            print(f"  {got:,} symbols available in {time.time()-t0:.0f}s")

    files = [os.path.join(args.data, f"{s}.csv") for s in syms]
    files = [f for f in files if os.path.exists(f)]
    if not files:
        print("\nNo cached data. Drop --no-fetch, or check --data.")
        return 2

    t0 = time.time()
    rows: list[dict] = []
    jobs = [(f, start, end) for f in files]
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

    report(df, args.trades or bool(args.symbol), args.top)

    if args.csv and not df.empty:
        df.sort_values("date").to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv} ({len(df):,} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
