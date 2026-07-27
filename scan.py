"""
Intraday scanner - run every 5 minutes during market hours.

Two-stage funnel so a full cash-segment universe fits inside one cron slot:

  Stage 1  bulk quotes (1000 instruments per request, ~1 req/sec)
           Drop anything whose LTP is not above its frozen 26W breakout level
           and above the min price. Both are week-constant, so this filter can
           never discard a candle that the indicator would have marked.

  Stage 2  for the handful that survive, pull this week's 5-minute candles and
           replay them bar by bar through the exact Pine logic.

    python scan.py [--force] [--symbols A,B] [--heartbeat]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, time as dtime, timedelta

import pandas as pd

from config import load_config
from dhan import IST, DhanClient, DhanError
from state import AlertState
from strategy import WeeklySnapshot, replay_week, week_start_of
from telegram import Telegram, format_heartbeat

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger("scan")

QUOTE_BATCH = 1000


def parse_hhmm(text: str) -> dtime:
    h, m = text.split(":")
    return dtime(int(h), int(m))


def market_is_open(cfg, now: datetime) -> bool:
    if now.weekday() >= 5:
        return False
    start = parse_hhmm(cfg.runtime.market_open)
    end = parse_hhmm(cfg.runtime.market_close)
    return start <= now.time() <= (datetime.combine(now.date(), end)
                                   + timedelta(minutes=10)).time()


def load_snapshots(cfg, week: str) -> list[WeeklySnapshot]:
    """
    Load the frozen weekly state. Returns [] (not an exception) when the file is
    missing: this job runs every 5 minutes, so a missing snapshot must be a quiet
    skip rather than a hard failure that alerts on every single run.
    """
    path = cfg.paths["snapshot"]
    if not path.exists():
        log.warning("no snapshot at %s - run the Weekly Snapshot workflow first", path)
        return []
    df = pd.read_csv(path, dtype=str)
    snaps, stale = [], 0
    for row in df.to_dict("records"):
        try:
            snap = WeeklySnapshot.from_row(row)
        except (KeyError, ValueError):
            continue
        if snap.week_start != week:
            stale += 1
        snaps.append(snap)
    if stale:
        log.warning("%d/%d snapshots are for a different week (expected %s) - "
                    "re-run build_snapshot", stale, len(snaps), week)
    return snaps


def prefilter(client: DhanClient, snaps: list[WeeklySnapshot], cfg) -> list[WeeklySnapshot]:
    """Keep only names trading above their frozen 26W level (and 52W if required)."""
    by_seg: dict[str, list[int]] = {}
    index: dict[tuple[str, str], WeeklySnapshot] = {}
    for s in snaps:
        by_seg.setdefault(s.exchange_segment, []).append(int(s.security_id))
        index[(s.exchange_segment, s.security_id)] = s

    quotes: dict[str, dict[str, float]] = {}
    for seg, ids in by_seg.items():
        for i in range(0, len(ids), QUOTE_BATCH):
            chunk = ids[i:i + QUOTE_BATCH]
            try:
                part = client.ltp({seg: chunk})
            except DhanError as exc:
                log.warning("quote batch failed (%s) - those names fall through: %s",
                            seg, str(exc)[:140])
                for sid in chunk:                      # fail open, never miss a signal
                    quotes.setdefault(seg, {})[str(sid)] = float("inf")
                continue
            for s, m in part.items():
                quotes.setdefault(s, {}).update(m)

    keep, headroom = [], 1.0 - cfg.runtime.prefilter_headroom_pct / 100.0
    for (seg, sid), snap in index.items():
        ltp = quotes.get(seg, {}).get(sid)
        if ltp is None:                                 # no quote -> don't risk it
            keep.append(snap)
            continue
        if ltp <= cfg.strategy.min_price:
            continue
        if ltp <= snap.entry_level * headroom:
            continue
        if cfg.strategy.req52 and ltp <= snap.level_52 * headroom:
            continue
        keep.append(snap)
    return keep


def scan_symbol(client: DhanClient, snap: WeeklySnapshot, cfg, week_start_dt):
    """Fetch this week's 5m candles and replay them."""
    now = datetime.now(IST)
    from_dt = datetime.combine(week_start_dt.date(), dtime(9, 0)).replace(tzinfo=IST)
    to_dt = now + timedelta(minutes=5)
    bars = client.intraday_candles(snap.security_id, snap.exchange_segment,
                                   from_dt, to_dt, interval=cfg.runtime.bar_interval_min)
    if bars.empty:
        return snap, None

    bars = bars[pd.to_datetime(bars["datetime"]).dt.tz_convert(IST)
                >= pd.Timestamp(from_dt)].reset_index(drop=True)
    if bars.empty:
        return snap, None
    return snap, replay_week(snap, cfg.strategy, bars)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="ignore market-hours check")
    ap.add_argument("--symbols", default=None, help="comma-separated subset")
    ap.add_argument("--heartbeat", action="store_true", help="send a summary even with 0 signals")
    ap.add_argument("--no-prefilter", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    started = time.time()
    now = datetime.now(IST)

    if not args.force and not market_is_open(cfg, now):
        log.info("market closed (%s IST) - nothing to do", now.strftime("%a %H:%M"))
        return 0

    week = str(week_start_of(now.date()).date())
    snaps = load_snapshots(cfg, week)
    if args.symbols:
        want = {s.strip().upper() for s in args.symbols.split(",")}
        snaps = [s for s in snaps if s.symbol.upper() in want]
    if not snaps:
        # Exit 0 on purpose: this runs every 5 minutes, and a hard failure here
        # would raise a workflow alert on every scheduled run until fixed.
        log.warning("no snapshots to scan - nothing to do")
        return 0

    tg = Telegram(cfg.secrets.telegram_bot_token, cfg.secrets.telegram_chat_id,
                  dry_run=cfg.runtime.dry_run)
    state = AlertState(cfg.paths["state"])
    client = DhanClient(cfg.secrets.dhan_client_id, cfg.secrets.dhan_access_token,
                        data_rate=cfg.runtime.data_rate_per_sec,
                        quote_rate=cfg.runtime.quote_rate_per_sec)

    # ---- skip names already alerted this week (Pine onePerWeek)
    if cfg.strategy.one_per_week:
        before = len(snaps)
        snaps = [s for s in snaps if not state.already_alerted(week, s.symbol)]
        if before != len(snaps):
            log.info("skipping %d already alerted this week", before - len(snaps))

    # ---- stage 1
    candidates = snaps
    if cfg.runtime.prefilter and not args.no_prefilter:
        candidates = prefilter(client, snaps, cfg)
        log.info("prefilter: %d -> %d candidates", len(snaps), len(candidates))

    # ---- stage 2
    signals, errors, last_bar = [], 0, None
    with ThreadPoolExecutor(max_workers=cfg.runtime.max_workers) as pool:
        futures = {pool.submit(scan_symbol, client, s, cfg, week_start_of(now.date())): s
                   for s in candidates}
        for fut in as_completed(futures):
            snap = futures[fut]
            try:
                _, res = fut.result()
            except DhanError as exc:
                errors += 1
                log.warning("%s: %s", snap.symbol, str(exc)[:140])
                continue
            except Exception as exc:                        # noqa: BLE001
                errors += 1
                log.warning("%s: unexpected %s", snap.symbol, exc)
                continue
            if not res or not res.signals:
                continue
            for sig in res.signals:
                if cfg.strategy.one_per_week and state.already_alerted(week, sig.symbol):
                    continue
                signals.append(sig)
                state.mark(week, sig.symbol, sig.bar_time, sig.price)
                last_bar = max(last_bar or sig.bar_time, sig.bar_time)

    signals.sort(key=lambda s: (s.bar_time, s.symbol))
    if signals:
        log.info("%d signal(s): %s", len(signals), ", ".join(s.symbol for s in signals))
        if tg.send_batch(signals):
            state.prune()
            state.save()
        else:
            log.error("Telegram delivery failed - state NOT saved so the next run retries")
    else:
        log.info("no signals")
        state.prune()
        state.save()

    if args.heartbeat:
        tg.send(format_heartbeat(len(snaps), len(candidates), len(signals),
                                 last_bar or now, time.time() - started, errors))

    log.info("done in %.0fs (universe %d, candidates %d, signals %d, errors %d)",
             time.time() - started, len(snaps), len(candidates), len(signals), errors)
    return 0


if __name__ == "__main__":
    sys.exit(main())
