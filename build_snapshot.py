"""
Weekly preparation job (run once a week, before Monday's open).

Downloads ~5y of daily candles for every symbol in the universe, folds them into
weekly bars, and freezes the closed-week indicator state into
data/weekly_snapshot.csv. The intraday scanner then needs zero heavy history
calls: it only pulls the current week's 5-minute candles.

    python build_snapshot.py [--limit N] [--universe-only]
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pandas as pd

from config import load_config
from dhan import IST, DhanClient, DhanError, Instrument, last_n_years
from strategy import build_snapshot, week_start_of

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger("build_snapshot")


def load_universe(cfg) -> list[Instrument]:
    uni = cfg.universe
    log.info("downloading Dhan scrip master ...")
    instruments = DhanClient.fetch_instruments(
        uni.exchange_segments, uni.series, exclude_etf=uni.exclude_etf)

    if uni.include_symbols:
        keep = {s.upper() for s in uni.include_symbols}
        instruments = [i for i in instruments if i.symbol.upper() in keep]
    if uni.exclude_symbols:
        drop = {s.upper() for s in uni.exclude_symbols}
        instruments = [i for i in instruments if i.symbol.upper() not in drop]

    instruments.sort(key=lambda i: (i.exchange_segment, i.symbol))
    if uni.max_symbols:
        instruments = instruments[:uni.max_symbols]

    with open(cfg.paths["universe"], "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["security_id", "symbol", "name", "exchange_segment", "series", "type"])
        for i in instruments:
            w.writerow([i.security_id, i.symbol, i.name, i.exchange_segment,
                        i.series, i.instrument_type])
    log.info("universe: %d instruments -> %s", len(instruments), cfg.paths["universe"])
    return instruments


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only the first N symbols")
    ap.add_argument("--universe-only", action="store_true", help="write universe.csv and stop")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    instruments = load_universe(cfg)
    if args.universe_only:
        return 0
    if args.limit:
        instruments = instruments[:args.limit]

    if not cfg.secrets.dhan_access_token:
        log.error("DHAN_ACCESS_TOKEN not set")
        return 2

    client = DhanClient(cfg.secrets.dhan_client_id, cfg.secrets.dhan_access_token,
                        data_rate=cfg.runtime.data_rate_per_sec,
                        quote_rate=cfg.runtime.quote_rate_per_sec)

    from_date, to_date = last_n_years(cfg.runtime.history_years)
    target_week = week_start_of(datetime.now(IST).date())

    # ---- PREFLIGHT ---------------------------------------------------------
    # Fetch ONE symbol before starting the long run. A bad token or an inactive
    # Data API subscription otherwise burns 10+ minutes of API calls and ends
    # with an unhelpful "no snapshots produced".
    probe = instruments[0]
    log.info("preflight: fetching %s to verify API access ...", probe.symbol)
    try:
        test = client.daily_candles(probe.security_id, probe.exchange_segment,
                                    from_date, to_date)
    except DhanError as exc:
        log.error("PREFLIGHT FAILED for %s: %s", probe.symbol, exc)
        log.error("Nothing was downloaded. Most likely causes:")
        log.error("  1. DHAN_ACCESS_TOKEN is wrong or expired (they last ~30 days)")
        log.error("  2. The Dhan DATA API subscription is not active - this is a")
        log.error("     paid add-on and historical candles need it")
        log.error("  3. DHAN_CLIENT_ID does not match the token")
        return 2
    if test.empty:
        log.error("PREFLIGHT: %s returned no candles. The token authenticated but "
                  "returned no data - check the Data API subscription.", probe.symbol)
        return 2
    log.info("preflight OK - %d daily candles for %s", len(test), probe.symbol)
    log.info("building snapshots for week starting %s (%d symbols)",
             target_week.date(), len(instruments))

    rows, errors, skipped = [], 0, 0
    no_data, short_hist = 0, 0
    started = time.time()

    def work(ins: Instrument):
        """Returns (instrument, snapshot, reason). reason is None on success."""
        daily = client.daily_candles(ins.security_id, ins.exchange_segment, from_date, to_date)
        if daily.empty:
            # The API answered but sent no candles - a very different problem
            # from "this stock is too young", so count it separately.
            return ins, None, "nodata"
        snap = build_snapshot(ins.symbol, ins.security_id, ins.exchange_segment,
                              daily, cfg.strategy, target_week)
        if snap is None:
            return ins, None, f"short_history({len(daily)} daily bars)"
        return ins, snap, None

    with ThreadPoolExecutor(max_workers=cfg.runtime.max_workers) as pool:
        futures = {pool.submit(work, i): i for i in instruments}
        for n, fut in enumerate(as_completed(futures), 1):
            ins = futures[fut]
            try:
                _, snap, reason = fut.result()
                if snap is None:
                    skipped += 1
                    if reason and reason.startswith("nodata"):
                        no_data += 1
                    else:
                        short_hist += 1
                    if skipped <= 5:            # show the first few, with the reason
                        log.warning("skip %s: %s", ins.symbol, reason)
                else:
                    rows.append(snap.to_row())
            except DhanError as exc:
                errors += 1
                log.warning("%s: %s", ins.symbol, str(exc)[:140])
            except Exception as exc:                       # noqa: BLE001
                errors += 1
                log.warning("%s: unexpected %s", ins.symbol, exc)
            if n % 200 == 0:
                log.info("  %d/%d  ok=%d skip=%d (nodata=%d short=%d) err=%d  %.0fs",
                         n, len(instruments), len(rows), skipped, no_data, short_hist,
                         errors, time.time() - started)

    if not rows:
        log.error("no snapshots produced from %d symbols "
                  "(skipped=%d [nodata=%d short_history=%d], errors=%d)",
                  len(instruments), skipped, no_data, short_hist, errors)
        if no_data > short_hist:
            log.error("Most symbols returned NO CANDLES. The token authenticated, "
                      "so check the Data API subscription and the date window.")
        if errors > skipped:
            log.error("Most symbols raised API errors - check the token, the Data "
                      "API subscription, and the rate limit.")
        else:
            log.error("Most symbols were SKIPPED for insufficient history. The scan "
                      "needs ~%d closed weekly bars (history_years=%d).",
                      max(cfg.strategy.len_long + 1,
                          cfg.strategy.ema_slow_len + cfg.strategy.ema_slow_back + 1),
                      cfg.runtime.history_years)
        log.error("Keeping any existing snapshot rather than overwriting it.")
        return 1

    df = pd.DataFrame(rows).sort_values("symbol")
    df.to_csv(cfg.paths["snapshot"], index=False)
    log.info("wrote %d snapshots (skipped %d [nodata=%d short=%d], errors %d) "
             "in %.0fs -> %s", len(df), skipped, no_data, short_hist, errors,
             time.time() - started, cfg.paths["snapshot"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
