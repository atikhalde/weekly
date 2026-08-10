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
def replay_symbol(path: str, start: str, end: str, mode: str = "btst") -> list[dict]:
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
            m_ant = btst.classify_approach(hist, float(level_c), float(level_d),
                                          partial_frac=1.0, dbg_symbol=sym)
        is_ant = bool(m_ant and m_ant.get("ok"))

        entry = close[j]
        nxt = close[j + 1]
        gross = (nxt / entry - 1) * 100.0

        # Confirmed BTST trade
        if want_btst and is_btst:
            conv, why = btst.conviction(m_btst)
            out.append(dict(
                date=dt.iloc[j].date().isoformat(), symbol=sym,
                mode="btst",
                tier=m_btst["tier"],
                arm=("fresh_A" if m_btst.get("fresh") and m_btst["tier"] == "A"
                     else "fresh_B" if m_btst.get("fresh") else "aged_B"),
                age=int(m_btst.get("age", 0)), entry=round(entry, 2),
                exit=round(nxt, 2), level=round(float(level_c), 2),
                day_ret=round(m_btst["day_ret"], 2),
                close_pos=round(m_btst["close_pos"], 3),
                rvol=round(float(m_btst.get("rvol") or 0), 2),
                atr_pct=round(m_btst["atr_pct"], 2),
                ext_pct=round(float(m_btst.get("ext_pct") or 0), 2),
                ret_12m=btst._num(m_btst.get("ret_12m"), 1),
                pre=int(m_btst.get("pre", 0)),
                conviction=conv, why=";".join(why),
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
                age=0, entry=round(entry, 2),
                exit=round(nxt, 2), level=round(float(m_ant.get("level", level_c)), 2),
                day_ret=round(m_ant.get("day_ret", 0.0), 2),
                close_pos=round(m_ant.get("close_pos", 0.0), 3),
                rvol=round(float(m_ant.get("rvol") or 0), 2),
                atr_pct=round(m_ant.get("atr_pct", 0.0), 2),
                ext_pct=round(float(-m_ant.get("gap_pct", 0.0) if side == "above" else 0.0), 2),
                ret_12m=btst._num(m_ant.get("ret_12m"), 1),
                pre=int(m_ant.get("pre", 0)),
                conviction=int(m_ant.get("pre", 0) >= 7),
                why=";".join(why_ant),
                btst_qualified="yes" if is_btst else "no",
                gross_pct=round(gross, 3),
                net_pct=round(gross - COST_ROUND_TRIP, 3),
            ))

    return out


def _rs(args):
    try:
        path, start, end, mode = args
        return replay_symbol(path, start, end, mode=mode)
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


def report(df: pd.DataFrame, show_trades: bool, top: int, mode: str = "btst") -> None:
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

    print("\n" + "=" * 108)
    print(f"BTST / ANTICIPATION BACKTEST ({mode_label}) - the LIVE rule replayed over history")
    print("=" * 108)
    print(f"window      {df.date.min()} .. {df.date.max()}")
    print(f"names hit   {df.symbol.nunique():,} distinct symbols produced a setup")
    print(f"cost        {COST_ROUND_TRIP}% round trip, entry = daily close, "
          f"exit = next close")
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

    # a top-5/day book, which is what actually gets traded
    print("\nTOP-5 PER DAY (the traded book)")
    r = df.copy()
    r["_k"] = ((r.arm == "fresh_A") * 300 + (r.arm == "aged_B") * 200
               + (r.arm.str.startswith("ant_")) * 100
               + r.rvol.fillna(0).clip(0, 50))
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
        print(f"{'date':<12}{'symbol':<13}{'tier':<12}{'arm':<11}{'age':>4}"
              f"{'entry':>9}{'exit':>9}{'day%':>7}{'rvol':>6}{'pre':>4}{'btst':>5}{'net%':>8}")
        for r_ in df.tail(top).itertuples():
            btst_tag = getattr(r_, "btst_qualified", "-")
            pre_val = getattr(r_, "pre", 0)
            print(f"{r_.date:<12}{r_.symbol:<13}{r_.tier:<12}{r_.arm:<11}{r_.age:>4}"
                  f"{r_.entry:>9.2f}{r_.exit:>9.2f}{r_.day_ret:>7.1f}"
                  f"{r_.rvol:>6.1f}{pre_val:>4}{btst_tag:>5}{r_.net_pct:>+8.2f}")

    print("\n" + "-" * 108)
    print("Entry is the DAILY CLOSE; the live scan buys ~15:20 on a partial "
          "candle (~0.14% cheaper), so this is mildly pessimistic.")
    print("Yahoo daily data, not Dhan - per-symbol volume can differ, so "
          "counts may not match live exactly.")
    print("One regime (2021-2026). Past results are not forward validation.")


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["btst", "anticipated", "anticipated_only", "all"],
                    default="btst",
                    help="btst (default): confirmed breakouts | "
                         "anticipated_only: anticipated setups that did NOT qualify for BTST at 15:20 | "
                         "anticipated: all pre-breakout anticipation setups | "
                         "all: confirmed BTST + anticipated setups")
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
    jobs = [(f, start, end, args.mode) for f in files]
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

    report(df, args.trades or bool(args.symbol), args.top, mode=args.mode)

    if args.csv and not df.empty:
        df.sort_values("date").to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv} ({len(df):,} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
