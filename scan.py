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
from mcap import load_table as load_mcap_table
from state import AlertState
from strategy import LOGIC_VERSION, WeeklySnapshot, replay_week, week_start_of
from telegram import build_telegram, format_heartbeat, _esc

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

    # ---- REFUSE STALE LEVELS -------------------------------------------------
    # A snapshot row is only trustworthy if it was produced by the CURRENT
    # strategy logic AND for the CURRENT week. Previously this function loaded
    # every row regardless, so on 27/28-Jul the scanner alerted against levels
    # from the reverted one-week shift:
    #     RADICO   quoted 4161.80  (true 26W high 4193.00)
    #     JKPAPER  quoted  401.50  (true  421.95)  -> FALSE ALERT
    #     THYROCARE quoted 578.25  (true  597.60)  -> FALSE ALERT
    #     VAIBHAVGBL quoted 268.08 (true  273.00)  -> FALSE ALERT
    # The rows were rebuilt correctly a day later, but by then the alerts had
    # already gone out. Levels are the one input that must never be guessed:
    # a wrong level is a wrong trade, so a stale row is DROPPED, not used.
    dropped_version = dropped_week = 0
    snaps = []
    for row in df.to_dict("records"):
        raw_lv = str(row.get("logic_version", "")).strip()
        try:
            row_lv = int(float(raw_lv)) if raw_lv not in ("", "nan") else None
        except ValueError:
            row_lv = None
        if row_lv != LOGIC_VERSION:
            dropped_version += 1
            continue
        if str(row.get("week_start", "")).strip() != week:
            dropped_week += 1
            continue
        try:
            snaps.append(WeeklySnapshot.from_row(row))
        except (KeyError, ValueError):
            continue

    if dropped_version or dropped_week:
        log.warning(
            "discarded %d row(s) from an older logic version and %d row(s) "
            "from another week - re-run the Weekly Snapshot workflow",
            dropped_version, dropped_week)
    if not snaps:
        log.error(
            "NO USABLE SNAPSHOT ROWS for week %s at logic_version %d. "
            "Refusing to scan rather than alert against stale breakout levels. "
            "Run the Weekly Snapshot workflow.", week, LOGIC_VERSION)
        return snaps

    # ---- BUG 37: MARKET CAP COMES FROM mcap.csv, NOT THE SNAPSHOT ------------
    # The snapshot stores whatever market cap was known WHEN IT WAS BUILT. A
    # snapshot written before mcap.py existed carries mcap="" on every row, and
    # because "unknown is not small" is the correct rule inside evaluate_bar,
    # every one of those rows PASSES c12. The filter then looks enabled in the
    # config and does nothing at all - silently.
    #
    # That is not hypothetical. Measured on the live 27-Jul snapshot:
    #     rows with a populated mcap: 1 of 2100   (only SAKHTISUG)
    # and two names below the 1000 Cr floor alerted that week:
    #     GANESHBE  881 Cr        PYRAMID  665 Cr
    #
    # mcap.csv is refreshed by the snapshot workflow and is the live source of
    # truth, so it OVERRIDES the frozen row. Market cap is a slow-moving
    # universe filter, not a frozen weekly level - there is no correctness
    # reason to freeze it, and freezing it is what broke the filter.
    #
    # A symbol absent from the table keeps whatever the snapshot had (usually
    # None -> passes). Unknown must still never block a name.
    if getattr(cfg.strategy, "use_mcap", False):
        caps = load_mcap_table(cfg.paths["mcap"])
        if not caps:
            # Loud, because the config claims a filter that cannot run.
            log.error(
                "use_mcap is ON but %s is empty or missing - the market-cap "
                "filter CANNOT be applied and every stock will pass c12. "
                "Run `python mcap.py` (the Weekly Snapshot workflow does this).",
                cfg.paths["mcap"])
        else:
            filled = 0
            for s in snaps:
                cap = caps.get(s.symbol.upper())
                if cap is not None:
                    s.mcap = cap
                    filled += 1
            log.info("market cap: %d/%d rows resolved from %s (min %g Cr, "
                     "margin %g%%)", filled, len(snaps), cfg.paths["mcap"].name,
                     cfg.strategy.min_mcap,
                     getattr(cfg.strategy, "mcap_margin_pct", 0.0))
            # If the table covers almost nothing the filter is effectively off.
            if filled < 0.5 * len(snaps):
                log.error(
                    "only %d of %d snapshot rows got a market cap - the c12 "
                    "filter is mostly INERT. Rebuild mcap.csv.",
                    filled, len(snaps))
    return snaps


def snapshot_age_days(cfg) -> int | None:
    """
    How many days old the committed snapshot's week_start is, or None if the
    file is missing/unreadable. Used to turn a silent no-op into a loud alarm.
    """
    path = cfg.paths["snapshot"]
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, dtype=str, usecols=["week_start"])
    except Exception:
        return None
    weeks = [w for w in df["week_start"].dropna().unique() if str(w).strip()]
    if not weeks:
        return None
    try:
        newest = max(pd.Timestamp(str(w)) for w in weeks)
    except Exception:
        return None
    return int((pd.Timestamp(datetime.now(IST).date()) - newest).days)


def report_stale_snapshot(cfg, week: str, job: str) -> None:
    """
    BUG 49 - a missing/stale snapshot used to be a SILENT no-op.

    On Mon 03-Aug-2026 the Sunday rebuild failed and the Monday retry hung.
    The committed snapshot still said week_start=2026-07-27, so every job
    dropped all 2,100 rows (correctly - that is BUG 25's stale-level guard),
    logged an error, and returned 0. Three workflows reported SUCCESS while
    doing nothing for a whole trading day, and no failure notice fired because
    nothing had actually failed.

    A data outage must be as loud as a crash. This sends one Telegram message
    naming the job and the staleness, and callers exit non-zero so the
    workflow goes red.
    """
    age = snapshot_age_days(cfg)
    age_txt = f"{age} day(s) old" if age is not None else "missing"
    log.error("STALE SNAPSHOT (%s): no usable rows for week %s at "
              "logic_version %d - %s is doing nothing. Run the Weekly "
              "Snapshot workflow.", age_txt, week, LOGIC_VERSION, job)
    try:
        tg = build_telegram(cfg)
        tg.send(
            f"\u26a0\ufe0f <b>{_esc(job)} produced NOTHING</b>\n"
            f"<i>The weekly snapshot is {_esc(age_txt)} - it has no rows for "
            f"the current week ({_esc(week)}).</i>\n\n"
            f"No signals can be generated until it is rebuilt. This is a data "
            f"outage, not a quiet day.\n\n"
            f"<b>Fix:</b> Actions \u2192 <b>Weekly Snapshot</b> \u2192 Run "
            f"workflow (tick <b>fresh</b>).")
    except Exception as exc:      # never let the alarm itself crash the job
        log.error("could not send the stale-snapshot alert: %s", str(exc)[:200])


def prefilter(client: DhanClient, snaps: list[WeeklySnapshot], cfg) -> list[WeeklySnapshot]:
    """Keep only names trading above their frozen 26W level (and 52W if required)."""
    by_seg: dict[str, list[int]] = {}
    index: dict[tuple[str, str], WeeklySnapshot] = {}
    for s in snaps:
        by_seg.setdefault(s.exchange_segment, []).append(int(s.security_id))
        index[(s.exchange_segment, s.security_id)] = s

    quotes: dict[str, dict[str, float]] = {}
    auth_failed = False
    for seg, ids in by_seg.items():
        for i in range(0, len(ids), QUOTE_BATCH):
            chunk = ids[i:i + QUOTE_BATCH]
            try:
                part = client.ltp({seg: chunk})
            except DhanError as exc:
                msg = str(exc)
                # An auth failure will hit EVERY subsequent call too. Failing
                # open here would push all ~2000 symbols into the per-symbol
                # stage, each retrying with backoff, until the job is killed by
                # the workflow timeout. Abort loudly instead.
                if ("auth failed" in msg or "401" in msg or "403" in msg
                        or "DH-901" in msg or "Authentication" in msg):
                    auth_failed = True
                    log.error("quote auth failed - aborting scan: %s", msg[:200])
                    break
                log.warning("quote batch failed (%s) - those names fall through: %s",
                            seg, msg[:140])
                for sid in chunk:                      # fail open, never miss a signal
                    quotes.setdefault(seg, {})[str(sid)] = float("inf")
                continue
            for s, m in part.items():
                quotes.setdefault(s, {}).update(m)
        if auth_failed:
            break

    if auth_failed:
        raise DhanError(
            "Dhan authentication failed. The access token is wrong or expired "
            "(they last ~30 days) - refresh DHAN_ACCESS_TOKEN.")

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
        # BUG 49: this used to log a warning and exit 0. The scan then reported
        # SUCCESS on every 5-minute run for a whole day while producing
        # nothing. It now alarms ONCE per day (state-tracked, so the remaining
        # ~75 runs stay quiet) and exits non-zero so the workflow goes red.
        st = AlertState(cfg.paths["state"])
        today = f"{datetime.now(IST):%Y-%m-%d}"
        if not st.stale_alerted(today):
            report_stale_snapshot(cfg, week, "Intraday Scan")
            st.mark_stale(today)
            st.save()
        else:
            log.error("stale snapshot for week %s - already alerted today", week)
        return 2

    tg = build_telegram(cfg, dry_run=cfg.runtime.dry_run)
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
        try:
            candidates = prefilter(client, snaps, cfg)
        except DhanError as exc:
            # Almost always an expired token. Tell the user on Telegram rather
            # than letting the job grind to a timeout with a cryptic exit code.
            log.error("%s", exc)
            tg.send(f"⚠️ <b>Scanner stopped</b>\n{exc}")
            return 1
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
