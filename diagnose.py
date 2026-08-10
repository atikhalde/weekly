"""
Explain, per symbol, exactly why an alert did or did not fire.

    python diagnose.py RATNAVEER TMB MONARCH SENCO
    python diagnose.py --from-snapshot RATNAVEER     # use the committed levels

For each symbol it prints the 26W breakout level, the current price, and every
one of the 13 conditions with PASS/FAIL, so "no alert" always has a reason you
can check against the chart.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, time as dtime, timedelta

import pandas as pd

from config import load_config
from dhan import IST, DhanClient, DhanError, last_n_years
from strategy import (
    COND_LABELS, WeeklySnapshot, build_snapshot, build_weekly_bars,
    evaluate_bar, gate_ok, replay_week, week_start_of,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")


def resolve(symbols: list[str], cfg) -> dict[str, tuple[str, str]]:
    """symbol -> (security_id, exchange_segment)"""
    print("Downloading Dhan scrip master ...")
    instruments = DhanClient.fetch_instruments(
        cfg.universe.exchange_segments, cfg.universe.series,
        exclude_etf=cfg.universe.exclude_etf)
    by_sym = {i.symbol.upper(): i for i in instruments}
    out = {}
    for s in symbols:
        ins = by_sym.get(s.upper())
        if ins is None:
            print(f"  {s}: NOT in the configured universe "
                  f"(segments={cfg.universe.exchange_segments}, series={cfg.universe.series})")
        else:
            out[s.upper()] = (ins.security_id, ins.exchange_segment)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("symbols", nargs="+")
    ap.add_argument("--from-snapshot", action="store_true",
                    help="use levels from weekly_snapshot.csv instead of recomputing")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if not cfg.secrets.dhan_access_token:
        print("DHAN_ACCESS_TOKEN not set - running diagnose with yfinance as primary source")

    client = DhanClient(cfg.secrets.dhan_client_id, cfg.secrets.dhan_access_token,
                        data_rate=cfg.runtime.data_rate_per_sec,
                        quote_rate=cfg.runtime.quote_rate_per_sec)

    now = datetime.now(IST)
    target_week = week_start_of(now.date())
    print(f"\nNow: {now:%Y-%m-%d %H:%M} IST | week starts {target_week.date()}")
    print(f"strict_entry={cfg.strategy.strict_entry} "
          f"gate_tolerance={cfg.strategy.gate_tolerance} "
          f"gate={cfg.strategy.gate_source} "
          f"req52={cfg.strategy.req52} one_per_week={cfg.strategy.one_per_week}\n")

    snaps_from_file: dict[str, WeeklySnapshot] = {}
    if args.from_snapshot:
        p = cfg.paths["snapshot"]
        if not p.exists():
            print(f"!! {p} does not exist - the scanner has nothing to scan.")
            print("   Run the Weekly Snapshot workflow first.")
            return 1
        for row in pd.read_csv(p, dtype=str).to_dict("records"):
            try:
                s = WeeklySnapshot.from_row(row)
                snaps_from_file[s.symbol.upper()] = s
            except (KeyError, ValueError):
                continue
        print(f"Loaded {len(snaps_from_file)} snapshots from {p.name}\n")

    resolved = resolve(args.symbols, cfg)
    from_date, to_date = last_n_years(cfg.runtime.history_years)

    for sym in args.symbols:
        key = sym.upper()
        print("=" * 68)
        print(f"  {key}")
        print("=" * 68)
        if key not in resolved:
            continue
        sec_id, seg = resolved[key]

        try:
            daily = client.daily_candles(sec_id, seg, from_date, to_date, symbol=sym)
        except DhanError as exc:
            print(f"  daily candles failed: {exc}")
            continue
        if daily.empty:
            print("  no daily candles returned")
            continue

        if key in snaps_from_file:
            snap = snaps_from_file[key]
            print(f"  (levels from snapshot, week {snap.week_start})")
        else:
            snap = build_snapshot(key, sec_id, seg, daily, cfg.strategy, target_week)
            if snap is None:
                wk = build_weekly_bars(daily)
                closed = wk[wk["week_start"] < target_week]
                need = max(cfg.strategy.len_long + 1, cfg.strategy.len_short + 2,
                           cfg.strategy.ema_slow_len + cfg.strategy.ema_slow_back + 1)
                print(f"  NO SNAPSHOT: only {len(closed)} closed weekly bars, need ~{need}")
                print(f"  -> too little history; this symbol can never alert.")
                continue

        # this week's 5-minute candles
        from_dt = datetime.combine(target_week.date(), dtime(9, 0)).replace(tzinfo=IST)
        try:
            bars = client.intraday_candles(sec_id, seg, from_dt,
                                           now + timedelta(minutes=5),
                                           interval=cfg.runtime.bar_interval_min,
                                           symbol=sym)
        except DhanError as exc:
            print(f"  intraday failed: {exc}")
            continue

        bars = bars[pd.to_datetime(bars["datetime"]).dt.tz_convert(IST)
                    >= pd.Timestamp(from_dt)].reset_index(drop=True) if not bars.empty else bars

        print(f"  26W level : {snap.entry_level:.2f}")
        print(f"  52W level : {snap.level_52:.2f}")
        print(f"  5m bars this week: {len(bars)}")

        if bars.empty:
            print("  no intraday bars yet this week")
            continue

        res = replay_week(snap, cfg.strategy, bars)
        price = float(bars.iloc[-1]["close"])
        print(f"  last close: {price:.2f}  ({'ABOVE' if price > snap.entry_level else 'below'} the 26W level)")

        ev = res.last_eval
        if ev:
            print(f"\n  Conditions: {ev.pass_count}/13")
            for k in sorted(ev.conditions):
                mark = "PASS" if ev.conditions[k] else "FAIL"
                print(f"    [{mark}] {k}  {COND_LABELS[k]}")
            print(f"\n  gate      : {gate_ok(snap, cfg.strategy, ev)}")
        print(f"  cross seen: {res.saw_cross_this_week}")
        print(f"  SIGNALS   : {len(res.signals)}")
        for s in res.signals:
            print(f"    -> {s.bar_time:%d-%b %H:%M} @ {s.price:.2f} ({s.trigger})")
        if not res.signals:
            if not res.saw_cross_this_week:
                print("    reason: price never crossed UP through the 26W level this week")
                print("            (a cross needs the PREVIOUS 5m bar at/below the level;")
                print("             if it gapped above at Monday's open, bar 1 counts as the cross)")
            else:
                print("    reason: crossed, but the gate blocked it (see FAIL rows above)")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
