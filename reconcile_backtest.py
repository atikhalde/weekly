#!/usr/bin/env python3
"""
Reconcile the BACKTEST book against what the LIVE ALERT actually sent.

WHY THIS EXISTS
2026-08-14, user: "paper trade and backtest are totally different? Why?"

They were, and comparing them by eye could not show why, because the two
sides are not the same kind of list:

  * btst_backtest_trades.csv is the top-3-per-day book the replay traded.
  * ab_ledger.csv is an A/B RESEARCH LOG. It records every model's every
    signal - E_btst, E_btst_wide, F_anticipate, F_anticipate_only, C_swing,
    D_early - including the same symbol twice under two models on one day.
    It had 11 rows on 2026-08-05 against an alert that sends at most 3.

So "the ledger has names the backtest doesn't" was never evidence of a bug
by itself. This script compares like with like: the backtest's book against
btst_alert_state.json (the literal object the Telegram message was rendered
from), and it attributes every difference to a named cause instead of
leaving a pile of mismatched symbols.

    python reconcile_backtest.py --csv btst_backtest_trades.csv
    python reconcile_backtest.py --csv out.csv --from 2026-07-30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def load_alert_book() -> dict[str, set[str]]:
    """{date -> symbols the alert actually told you to buy}."""
    from ab_paper import load_alert_selection
    return load_alert_selection(ROOT)


def load_ledger_book(models=("E_btst", "F_anticipate")) -> dict[str, set[str]]:
    """{date -> symbols the paper ledger booked} for the live models."""
    led = ROOT / "ab_ledger.csv"
    if not led.exists():
        return {}
    df = pd.read_csv(led)
    df = df[df.model.isin(models)]
    out: dict[str, set[str]] = {}
    for d, g in df.groupby(df.signal_date.astype(str)):
        out[d] = set(g.symbol.astype(str))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="btst_backtest_trades.csv",
                    help="the backtest's trade CSV")
    ap.add_argument("--from", dest="from_date", default="",
                    help="only compare dates >= this")
    ap.add_argument("--to", dest="to_date", default="")
    args = ap.parse_args()

    path = Path(args.csv)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        print(f"no {path} - run btst_backtest.py --csv {path.name} first")
        return 2

    try:
        bt = pd.read_csv(path)
    except (pd.errors.EmptyDataError, ValueError):
        print(f"{path} is empty or unparseable")
        return 2
    if bt.empty or "date" not in bt.columns:
        print("backtest CSV is empty")
        return 2
    bt["date"] = bt["date"].astype(str)

    alert = load_alert_book()
    ledger = load_ledger_book()

    lo = args.from_date or min(bt.date)
    hi = args.to_date or max(bt.date)
    bt = bt[(bt.date >= lo) & (bt.date <= hi)]

    print("=" * 78)
    print("BACKTEST vs LIVE ALERT")
    print("=" * 78)
    print(f"window            {lo} .. {hi}")
    print(f"backtest trades   {len(bt):,}")
    print(f"alert snapshots   {len(alert):,} day(s) on disk")
    if "mcap_ok" in bt.columns:
        bad = (bt.mcap_ok != "yes").sum()
        print(f"mcap floor        {'CLEAN' if not bad else str(bad) + ' row(s) BELOW the floor leaked into the CSV'}")
    if "mode" in bt.columns:
        modes = bt["mode"].value_counts().to_dict()
        print(f"modes             {modes}")
        if set(modes) == {"anticipated_only"}:
            print()
            print("  !! every row is anticipated_only. That mode is defined as")
            print("     'anticipated setups that did NOT qualify for BTST', i.e. the")
            print("     COMPLEMENT of what the live alert buys. Re-run --mode all.")

    if not alert:
        print()
        print("No btst_alert_state*.json on disk, so there is nothing to reconcile")
        print("against. The snapshots only start from the day write_alert_state()")
        print("shipped; earlier dates can never be compared this way.")
        return 0

    dates = sorted(set(bt.date) | set(alert))
    dates = [d for d in dates if lo <= d <= hi]

    n_match = n_bt_only = n_al_only = 0
    print()
    print(f"{'date':<12}{'backtest':<26}{'alert':<26}verdict")
    print("-" * 78)
    for d in dates:
        b = set(bt[bt.date == d].symbol.astype(str))
        a = alert.get(d, set())
        if not a and d not in alert:
            verdict = "no snapshot for this day"
        elif b == a:
            verdict = "MATCH"
        else:
            verdict = f"+{len(b - a)} bt-only / +{len(a - b)} alert-only"
        n_match += len(b & a)
        n_bt_only += len(b - a)
        n_al_only += len(a - b)
        print(f"{d:<12}{','.join(sorted(b))[:24]:<26}{','.join(sorted(a))[:24]:<26}{verdict}")

    print("-" * 78)
    print(f"symbols in both            {n_match}")
    print(f"backtest only              {n_bt_only}")
    print(f"alert only                 {n_al_only}")

    print()
    print("LEDGER SCOPE (why the raw ledger looks nothing like either)")
    print("-" * 78)
    for d in dates:
        l = ledger.get(d, set())
        a = alert.get(d, set())
        if l and a and len(l) > len(a):
            print(f"  {d}  ledger {len(l):>2} row(s) vs alert {len(a)} - "
                  f"extra: {','.join(sorted(l - a))[:50]}")
    print("  (the ledger logs every model separately; the alert sends <= 3)")

    print()
    print("REMAINING KNOWN BIAS - not a bug, cannot be removed")
    print("-" * 78)
    try:
        from btst_backtest import LIVE_ENTRY_BIAS_PCT, LIVE_ENTRY_BIAS_N
        print(f"  entry: backtest fills at the daily CLOSE; live fills ~15:20, measured")
        print(f"         {LIVE_ENTRY_BIAS_PCT:+.3f}% above the close over {LIVE_ENTRY_BIAS_N} real trades.")
        print(f"         Yahoo caps intraday history at 60 days, so this cannot be replayed.")
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
