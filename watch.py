"""
Low-latency resident scanner — alerts ~10-20 s after a 5-minute candle closes.

WHY THIS EXISTS
---------------
The GitHub Actions path costs ~3-4 minutes per alert, and almost none of that
is real work:

    waiting for the next 5-min cron tick   avg 2.5 min   <- the big one
    runner boot + checkout + pip install       ~50 s
    actually scanning                          ~15 s

This process stays resident, so all the fixed cost disappears. It sleeps until
the exact moment a 5m candle closes, waits a short settle delay for the
exchange feed, then scans immediately.

    python watch.py                 # run during market hours
    python watch.py --settle 20     # wait 20 s after each candle close
    python watch.py --once          # single pass, for testing

Everything strategy-related is imported from the same modules scan.py uses, so
signals are identical to the GitHub path — only the timing differs.

RUN IT SOMEWHERE ALWAYS-ON
--------------------------
A cheap VPS, a Raspberry Pi, or any machine left on during 09:15-15:30 IST.
Keep the GitHub workflow enabled as a safety net: it is idempotent and
state.json de-duplicates, so the two cannot double-alert the same signal
(see --state-file if both run against the same repo).
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from datetime import datetime, time as dtime, timedelta

import pandas as pd

from config import load_config
from dhan import IST, DhanClient, DhanError
from scan import load_snapshots, market_is_open, parse_hhmm, prefilter, scan_symbol
from state import AlertState
from strategy import week_start_of
from telegram import Telegram, format_heartbeat

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("watch")

_stop = False


def _handle_signal(signum, frame):        # noqa: ARG001
    global _stop
    _stop = True
    log.info("shutdown requested - finishing current cycle")


def next_candle_close(now: datetime, interval_min: int) -> datetime:
    """The next wall-clock instant a 5m candle closes (09:20, 09:25, ...)."""
    base = now.replace(second=0, microsecond=0)
    minute = (base.minute // interval_min) * interval_min
    slot = base.replace(minute=minute)
    while slot <= now:
        slot += timedelta(minutes=interval_min)
    return slot


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settle", type=float, default=12.0,
                    help="seconds to wait after a candle closes before scanning "
                         "(gives Dhan time to publish the finished candle)")
    ap.add_argument("--once", action="store_true", help="one cycle then exit")
    ap.add_argument("--force", action="store_true", help="ignore market hours")
    ap.add_argument("--heartbeat-every", type=int, default=0,
                    help="send a Telegram summary every N cycles (0 = never)")
    ap.add_argument("--symbols", default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    cfg = load_config(args.config)
    interval = cfg.runtime.bar_interval_min

    tg = Telegram(cfg.secrets.telegram_bot_token, cfg.secrets.telegram_chat_id,
                  dry_run=cfg.runtime.dry_run)
    state = AlertState(cfg.paths["state"])
    client = DhanClient(cfg.secrets.dhan_client_id, cfg.secrets.dhan_access_token,
                        data_rate=cfg.runtime.data_rate_per_sec,
                        quote_rate=cfg.runtime.quote_rate_per_sec)

    # ---- Load the snapshot ONCE. It is frozen for the whole week, so there is
    # no reason to re-read 800 KB of CSV on every cycle.
    week = str(week_start_of(datetime.now(IST).date()).date())
    snaps = load_snapshots(cfg, week)
    if args.symbols:
        want = {s.strip().upper() for s in args.symbols.split(",")}
        snaps = [s for s in snaps if s.symbol.upper() in want]
    if not snaps:
        log.error("no snapshots for week %s - run build_snapshot.py first", week)
        return 1
    log.info("loaded %d snapshots for week %s", len(snaps), week)
    log.info("settle=%.0fs interval=%dm | alerts fire ~%.0f-%.0fs after each candle close",
             args.settle, interval, args.settle, args.settle + 15)

    cycles = 0
    while not _stop:
        now = datetime.now(IST)

        # Reload the snapshot when the week rolls over.
        cur_week = str(week_start_of(now.date()).date())
        if cur_week != week:
            log.info("new week %s - reloading snapshot", cur_week)
            week = cur_week
            snaps = load_snapshots(cfg, week)
            if not snaps:
                log.warning("no snapshot for %s yet - waiting", week)
                time.sleep(60)
                continue

        if not args.force and not market_is_open(cfg, now):
            close = parse_hhmm(cfg.runtime.market_close)
            if now.weekday() < 5 and now.time() < parse_hhmm(cfg.runtime.market_open):
                target = datetime.combine(now.date(), parse_hhmm(cfg.runtime.market_open),
                                          tzinfo=IST)
                wait = (target - now).total_seconds()
                log.info("pre-market - sleeping %.0f min until open", wait / 60)
                time.sleep(min(wait, 300))
            else:
                log.info("market closed (%s) - checking again in 5 min",
                         now.strftime("%a %H:%M"))
                time.sleep(300)
            if args.once:
                return 0
            continue

        # ---- sleep until just after the next candle close
        target = next_candle_close(now, interval) + timedelta(seconds=args.settle)
        wait = (target - datetime.now(IST)).total_seconds()
        if wait > 0:
            log.info("next candle closes %s - scanning at %s",
                     (target - timedelta(seconds=args.settle)).strftime("%H:%M:%S"),
                     target.strftime("%H:%M:%S"))
            # wake up periodically so Ctrl-C is responsive
            end = time.monotonic() + wait
            while time.monotonic() < end and not _stop:
                time.sleep(min(1.0, end - time.monotonic()))
        if _stop:
            break

        t0 = time.time()
        cycles += 1

        pending = [s for s in snaps
                   if not (cfg.strategy.one_per_week and state.already_alerted(week, s.symbol))]
        if not pending:
            log.info("every symbol already alerted this week - idle")
            if args.once:
                break
            continue

        try:
            candidates = prefilter(client, pending, cfg) if cfg.runtime.prefilter else pending
        except DhanError as exc:
            log.error("prefilter failed: %s", str(exc)[:160])
            if args.once:
                return 1
            continue

        signals, errors = [], 0
        week_start_dt = week_start_of(datetime.now(IST).date())
        for snap in candidates:
            if _stop:
                break
            try:
                _, res = scan_symbol(client, snap, cfg, week_start_dt)
            except DhanError as exc:
                errors += 1
                log.warning("%s: %s", snap.symbol, str(exc)[:100])
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

        elapsed = time.time() - t0
        if signals:
            signals.sort(key=lambda s: (s.bar_time, s.symbol))
            log.info("%d signal(s) in %.1fs: %s", len(signals), elapsed,
                     ", ".join(s.symbol for s in signals))
            if tg.send_batch(signals):
                state.prune()
                state.save()
            else:
                log.error("Telegram failed - state NOT saved, will retry next cycle")
        else:
            log.info("no signals (%d candidates, %.1fs, %d errors)",
                     len(candidates), elapsed, errors)
            state.prune()
            state.save()

        if args.heartbeat_every and cycles % args.heartbeat_every == 0:
            tg.send(format_heartbeat(len(pending), len(candidates), len(signals),
                                     datetime.now(IST), elapsed, errors))

        if args.once:
            break

    log.info("stopped after %d cycle(s)", cycles)
    return 0


if __name__ == "__main__":
    sys.exit(main())
