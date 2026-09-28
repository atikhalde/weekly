#!/usr/bin/env python3
"""
Precision OB Tap scanner — the second stage behind the weekly breakout scanner.

WHY THIS EXISTS
---------------
`scan.py` answers "which stocks broke their 26-week high this week?" — on the
week of 07-Sep-2026 that was 67 names, and most of them went nowhere. This
scanner answers the follow-up question the user actually trades:

    "Of the stocks the weekly scanner already flagged, which one has pulled
     back into a precise institutional order block and is being defended
     RIGHT NOW?"

    scan.py marks state.json  ──►  WAITING LIST
                                       │  daily bars, closed-bar replay
                                       ▼
                       displacement → origin candle → precision OB
                                       │  armed by departure, age >= 3 bars
                                       ▼
                    live intraday low taps the pre-order level
                                       │
                                       ▼
                              🟠 TAP 1  ──► Telegram

The waiting-list feed uses scan.py's first alert for the active 26-week breakout
cycle, not later weekly duplicate rows. The OB zone, tap, and event de-duplication
rules below remain independent and unchanged.

The order-block logic is a Pine-exact port of `precision.txt`
("Institutional OB — Precision Tap & Pre-Order", Pine v6). `ob_precision.py`
documents the two deliberate deviations, and both exist to keep the alert
NON-REPAINTING: a zone is only ever born on a CLOSED daily bar and its geometry
is frozen for life, and the live thresholds are pinned to the last CLOSED ATR
instead of the forming bar's. A tap is derived from the session's actual low,
which can only go lower, so a fired tap can never be retracted.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
* It never writes `state.json`. The weekly scanner's alert state is READ-ONLY
  input here, exactly the way `btst.py` treats it. Its own state lives in
  `ob_precision_state.json`, so the two systems can never corrupt each other.
* It places no orders. Nothing in this repo does.
* It does not re-alert: one message per (symbol, zone, event, tap number),
  de-duplicated across runs through its own state file.

COST MODEL — why a 5-minute cron is affordable
----------------------------------------------
The closed-bar replay needs daily history: one call per symbol. That is paid
ONCE per session per waiting symbol. Every other run of the day judges the live
tap from ONE bulk OHLC request for the whole list — the same trick `scan.py`'s
stage-1 prefilter uses, and the reason a backfilled list of ~370 names does not
turn into ~370 history calls every five minutes. Symbols with no live zone are
not quoted at all: with nothing to tap, a quote would be waste.

    python ob_tap_scan.py [--force] [--symbols A,B] [--heartbeat] [--config X]
                          [--state-file PATH] [--refresh-only]

STATE FILE
----------
`ob_precision_state.json`, committed by the workflow the same way scan.py
commits `state.json`, because a GitHub runner starts with a clean filesystem
every run and "already alerted" has to live somewhere. It holds three things:
the waiting list (with the 26-week level each name broke, captured at first
sight), the derived zone cache that lets the intraday runs skip the history
call, and the alert de-dupe keys. It is only rewritten when something other
than the timestamp actually changed - see save_state().
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from config import load_config
from dhan import IST, DhanClient, DhanError
from ob_precision import OBParams, bars_from_frame, live_pass, replay
from scan import market_is_open, parse_hhmm
from state import AlertState
from telegram import build_telegram, _esc, _fmt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger("ob_tap")

# Bumped whenever the SHAPE of ob_precision_state.json changes. A file from an
# older generation keeps its waiting list and its alert de-dupe keys (losing
# those would re-alert names that already fired) and drops only the derived
# zone cache, which the next refresh rebuilds from candles anyway.
STATE_LOGIC_VERSION = 1


# --------------------------------------------------------------------------- #
#  Clock
# --------------------------------------------------------------------------- #
def _now() -> datetime:
    """
    The one place this module reads the wall clock, so a test can freeze it.
    Everything downstream (market-hours gate, session split, timestamps) takes
    the value from here rather than calling datetime.now() itself.
    """
    return datetime.now(IST)


# --------------------------------------------------------------------------- #
#  Own state file
# --------------------------------------------------------------------------- #
def empty_state() -> dict[str, Any]:
    return {
        "logic_version": STATE_LOGIC_VERSION,
        "updated_at": None,
        "waiting": {},      # symbol -> waiting-list record
        "zones": {},        # symbol -> persisted replay context (derived)
        "alerts": {},       # dedupe key -> {sent_at, kind, session, price}
        # Date of the last run that failed with a TOTAL data outage. It survives
        # between runs so a five-minute cron cannot send the same failure notice
        # 78 times a day; `load_state` must carry it back or the guard is a no-op.
        "data_outage_on": None,
        # Slot -> date the daily waiting-list digest was last DELIVERED on
        # ("post_close" / "pre_open"). Same one-per-day discipline as the outage
        # marker, and again only written once Telegram confirms delivery, so a
        # failed send is retried by the next run instead of being swallowed.
        # Additive: an older state file simply has no key, which reads as "never
        # sent", so STATE_LOGIC_VERSION does not move and the zone cache - which
        # costs ~370 history calls to rebuild - stays warm across the upgrade.
        "digest_on": {},
    }


def load_state(path: Path) -> dict[str, Any]:
    """
    Read the scanner's own state. A missing or corrupt file is a quiet restart,
    not a crash: this job runs every five minutes and the zone cache is derived
    data that the next refresh rebuilds.
    """
    data = empty_state()
    if not path.exists():
        return data
    try:
        raw = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("could not read %s (%s) - starting fresh", path.name, exc)
        return data
    if not isinstance(raw, dict):
        return data
    if int(raw.get("logic_version") or 0) != STATE_LOGIC_VERSION:
        log.warning("state file logic_version %s != %d - keeping the waiting "
                    "list and alert history, rebuilding the zone cache",
                    raw.get("logic_version"), STATE_LOGIC_VERSION)
        data["waiting"] = raw.get("waiting") or {}
        data["alerts"] = raw.get("alerts") or {}
        return data
    for key in ("waiting", "zones", "alerts", "updated_at", "data_outage_on",
                "digest_on"):
        if key in raw:
            data[key] = raw[key]
    if not isinstance(data.get("digest_on"), dict):
        data["digest_on"] = {}
    return data


def save_state(path: Path, data: dict[str, Any],
               previous: dict[str, Any] | None = None) -> bool:
    """
    Atomic write, same discipline as state.py: tmp file then replace.

    Returns False WITHOUT writing when the only difference from `previous` is
    the timestamp. state.py gets this behaviour from its `_dirty` flag, and it
    matters more here: the workflow commits this file, and `updated_at` changes
    on every run, so an unconditional write would mean a commit every five
    minutes (~78 a day) for a file whose contents did not change. A real change
    is a new waiting-list name, a refreshed zone cache, a resolved status or a
    delivered alert - roughly two or three commits a day.

    Comparing against the loaded copy rather than tracking a flag means a
    mutation cannot be forgotten and silently lost.
    """
    data["logic_version"] = STATE_LOGIC_VERSION
    if previous is not None:
        strip = lambda d: {k: v for k, v in d.items() if k != "updated_at"}  # noqa: E731
        if strip(data) == strip(previous):
            return False
    data["updated_at"] = _now().isoformat(timespec="seconds")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, default=str))
    tmp.replace(path)
    return True


# --------------------------------------------------------------------------- #
#  Universe / level lookup
# --------------------------------------------------------------------------- #
def resolve_universe(cfg) -> dict[str, tuple[str, str]]:
    """
    symbol -> (security_id, exchange_segment).

    The weekly snapshot first (it is exactly the scanned universe and carries
    the frozen levels), then universe.csv for names that have since dropped out
    of the snapshot — a waiting-list symbol must stay resolvable for as long as
    it is on the list, which can outlive the snapshot row that produced it.
    """
    out: dict[str, tuple[str, str]] = {}
    for name, cols in (("universe", ("symbol", "security_id", "exchange_segment")),
                       ("snapshot", ("symbol", "security_id", "exchange_segment"))):
        path = cfg.paths[name]
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, dtype=str)
        except Exception as exc:                              # noqa: BLE001
            log.warning("could not read %s: %s", path.name, exc)
            continue
        if not set(cols).issubset(df.columns):
            continue
        for row in df.to_dict("records"):
            sym = str(row["symbol"]).strip().upper()
            sid = str(row["security_id"]).strip()
            seg = str(row["exchange_segment"]).strip()
            if sym and sid and seg:
                out[sym] = (sid, seg)          # snapshot read last -> it wins
    return out


def snapshot_levels(cfg) -> dict[str, tuple[str, float]]:
    """symbol -> (week_start, entry_level) from the CURRENT weekly snapshot."""
    path = cfg.paths["snapshot"]
    out: dict[str, tuple[str, float]] = {}
    if not path.exists():
        return out
    try:
        df = pd.read_csv(path, dtype=str)
    except Exception:                                          # noqa: BLE001
        return out
    for row in df.to_dict("records"):
        try:
            out[str(row["symbol"]).strip().upper()] = (
                str(row.get("week_start", "")).strip(), float(row["entry_level"]))
        except (KeyError, TypeError, ValueError):
            continue
    return out


# --------------------------------------------------------------------------- #
#  Waiting list
# --------------------------------------------------------------------------- #
def _weekly_alerts_from_data(data: dict[str, Any],
                             backfill_weeks: int) -> dict[str, dict[str, Any]]:
    """Build the backfill feed, anchored to each symbol's FIRST cycle alert.

    ``weeks`` is intentionally pruned to six weeks by scan.py. The persistent
    ``breakout_alerts`` map is the authoritative cross-week record and can be
    older than that; using only ``weeks`` silently drops a still-locked cycle
    (OAL was lost this way after its July alert). Include cycle records within
    the configured backfill horizon, then use retained weekly rows only as a
    legacy fallback for state files predating breakout_alerts.
    """
    weeks = data.get("weeks") or {}
    cycle_alerts = data.get("breakout_alerts") or {}
    horizon = max(1, int(backfill_weeks))
    cutoff = _now().date() - timedelta(weeks=horizon)
    out: dict[str, dict[str, Any]] = {}

    # Index retained rows so a canonical record missing price/level can recover
    # those fields only from the exact same alert, never a later duplicate.
    retained: dict[tuple[str, str], dict[str, Any]] = {}
    for wk, rows in weeks.items():
        for raw_sym, rec in (rows or {}).items():
            if isinstance(rec, dict):
                retained[(str(raw_sym).strip().upper(), str(rec.get("bar_time", "")))] = rec

    for raw_sym, cycle in cycle_alerts.items():
        sym = str(raw_sym).strip().upper()
        if not sym or not isinstance(cycle, dict):
            continue
        bar = str(cycle.get("bar_time", ""))
        day = as_date(bar)
        if not day or day < cutoff:
            continue
        try:
            week = (day - timedelta(days=day.weekday())).isoformat()
        except Exception:  # noqa: BLE001
            continue
        exact = retained.get((sym, bar), {})
        price = cycle.get("price", exact.get("price"))
        try:
            price = float(price) if price is not None else None
        except (TypeError, ValueError):
            price = None
        out[sym] = {"week": week, "breakout_bar": bar,
                    "breakout_price": price}

    # Legacy fallback: only symbols without a canonical record use weekly rows.
    # This preserves old state compatibility without letting duplicate rows
    # refresh the anchor for symbols that do have a cycle record.
    for wk in sorted(weeks)[-horizon:]:
        for raw_sym, rec in (weeks.get(wk) or {}).items():
            if not isinstance(rec, dict):
                continue
            sym = str(raw_sym).strip().upper()
            if sym in out or sym in cycle_alerts:
                continue
            bar = str(rec.get("bar_time", ""))
            day = as_date(bar)
            if not day or day < cutoff:
                continue
            try:
                price = float(rec.get("price")) if rec.get("price") is not None else None
            except (TypeError, ValueError):
                price = None
            prev = out.get(sym)
            if prev and str(prev.get("breakout_bar", "")) >= bar:
                continue
            out[sym] = {"week": wk, "breakout_bar": bar,
                        "breakout_price": price}
    return out


def weekly_alerts(cfg, backfill_weeks: int) -> dict[str, dict[str, Any]]:
    """
    One canonical weekly-breakout alert per symbol's current 26-week cycle,
    read from scan.py's state file.

    The weekly `weeks` map still holds legacy and same-week de-duplication
    records. `breakout_alerts` is the cross-week source of truth: while a cycle
    is active, only its first alert may seed or refresh the OB waiting list.
    After the 26-week lock expires, scan.py replaces that record only when a
    fresh post-lockout 26-week cross actually alerts.

    READ-ONLY. `state.json` belongs to the weekly scanner; this module reads it
    and never marks anything there.
    """
    st = AlertState(cfg.paths["state"])
    return _weekly_alerts_from_data(getattr(st, "_data", {}), backfill_weeks)


def harvest_waiting(state: dict[str, Any], alerts: dict[str, dict[str, Any]],
                    ids: dict[str, tuple[str, str]],
                    levels: dict[str, tuple[str, float]],
                    today: str) -> list[str]:
    """
    Fold the weekly scanner's alerts into the waiting list.

    A symbol is copied in ONCE and then lives independently of `state.json`.
    That matters: `state.py.prune()` keeps only six weeks, while a breakout
    cycle can remain eligible for the full 26-week lock. Its canonical first
    alert must not vanish or be replaced by a later weekly repeat.

    The 26-week level is captured at first sight for the same reason: the
    snapshot is overwritten every Monday, so next week the level that this
    breakout actually cleared is gone from the repo.
    """
    added: list[str] = []
    waiting = state.setdefault("waiting", {})
    for sym, rec in sorted(alerts.items()):
        sid_seg = ids.get(sym)
        if sid_seg is None:
            log.warning("%s alerted weekly but is not resolvable to a "
                        "security id - skipping", sym)
            continue
        existing = waiting.get(sym)
        if existing is None:
            week = rec["week"]
            lvl_week, lvl = levels.get(sym, ("", None))
            waiting[sym] = {
                "symbol": sym,
                "security_id": sid_seg[0],
                "exchange_segment": sid_seg[1],
                "week": week,
                "breakout_bar": rec["breakout_bar"],
                "breakout_price": rec["breakout_price"],
                # Only trust the snapshot level when the row is for the SAME
                # week the alert fired in. A level from a different week is a
                # different number, and printing the wrong one next to an alert
                # is worse than printing none.
                "level_26w": lvl if lvl_week == week else None,
                "added_at": today,
                "status": "waiting",
                "resolved_at": None,
                "resolved_reason": None,
            }
            added.append(sym)
        elif existing.get("status") in ("waiting", "expired") and \
                rec["breakout_bar"] != str(existing.get("breakout_bar", "")):
            old_date = as_date(existing.get("breakout_bar"))
            new_date = as_date(rec.get("breakout_bar"))
            if old_date is not None and new_date is not None and new_date > old_date:
                # A later canonical record can only be a fresh post-lockout
                # cycle, not a same-cycle weekly repeat. Start a new OB wait and
                # discard the previous cycle's derived zone cache.
                existing.update(status="waiting", resolved_at=None,
                                resolved_reason=None, added_at=today)
                (state.get("zones") or {}).pop(sym, None)
            # Otherwise this is a correction to the original cycle anchor.
            existing.update(week=rec["week"], breakout_bar=rec["breakout_bar"],
                            breakout_price=rec["breakout_price"])
            lvl_week, lvl = levels.get(sym, ("", None))
            existing["level_26w"] = lvl if lvl_week == rec["week"] else None
    return added


def reconcile_waiting_cycles(state: dict[str, Any],
                             cycle_alerts: dict[str, Any]) -> list[str]:
    """
    Correct an active waiter that predates the persistent cycle feed and was
    anchored to a later duplicate, even when the canonical first row has since
    fallen outside the weekly state's six-week retention window.

    This only moves an existing waiting anchor backward. It does not seed old
    symbols that are no longer in the backfill window or reopen resolved OB
    records. Price/level details are cleared when the canonical state lacks
    them rather than keeping numbers from the later duplicate.
    """
    corrected: list[str] = []
    waiting = state.get("waiting") or {}
    for raw_sym, cycle in (cycle_alerts or {}).items():
        sym = str(raw_sym).strip().upper()
        existing = waiting.get(sym)
        if (not isinstance(existing, dict) or existing.get("status") != "waiting"
                or not isinstance(cycle, dict)):
            continue
        first_bar = str(cycle.get("bar_time", ""))
        old_bar = str(existing.get("breakout_bar", ""))
        first_day, old_day = as_date(first_bar), as_date(old_bar)
        if first_day is None or old_day is None:
            continue
        if first_day > old_day or (first_day == old_day and first_bar >= old_bar):
            continue
        existing["week"] = (first_day - timedelta(days=first_day.weekday())).isoformat()
        existing["breakout_bar"] = first_bar
        existing["breakout_price"] = cycle.get("price")
        existing["level_26w"] = cycle.get("entry_level")
        (state.get("zones") or {}).pop(sym, None)
        corrected.append(sym)
    return corrected


def harvest_stage(cfg, ob, state: dict[str, Any], today: str) -> tuple[list[str], bool]:
    """
    Stage 0: fold the weekly scanner's alerts into the waiting list.

    Returns (names added this run, whether state.json had any alerts at all) -
    the second value is what tells main() "there is no source to build a list
    from", which is a harder failure than an empty list.

    Three committed files and no API calls, which is why the pre-open and
    --digest-only runs do it as well: a manual "show me the list" on a cold
    start, or a Monday morning after a weekend the scheduler skipped, must not
    print an empty list while state.json is full of names.
    """
    scanner_state = AlertState(cfg.paths["state"])
    scanner_data = getattr(scanner_state, "_data", {})
    alerts = _weekly_alerts_from_data(scanner_data, ob.backfill_weeks)
    corrected = reconcile_waiting_cycles(
        state, scanner_data.get("breakout_alerts") or {})
    if corrected:
        log.info("waiting list: corrected first-cycle anchor for %s",
                 ", ".join(corrected[:12]) + (" ..." if len(corrected) > 12 else ""))
    if not alerts:
        # Not a Telegram alarm: scan.py owns the stale-snapshot outage notice
        # (BUG 49) and a second message would only contradict it. It is not
        # fatal here either - names ALREADY on the waiting list keep being
        # scanned, because the list is designed to outlive state.json's
        # six-week prune. Only an empty list with no source to refill it is an
        # error worth a red workflow.
        log.error("no weekly alerts in %s - the waiting list cannot be EXTENDED "
                  "this run. Is the intraday scan running?", cfg.paths["state"])
        return [], False
    ids = resolve_universe(cfg)
    levels = snapshot_levels(cfg)
    added = harvest_waiting(state, alerts, ids, levels, today)
    if added:
        log.info("waiting list: +%d new (%s)", len(added),
                 ", ".join(added[:12]) + (" ..." if len(added) > 12 else ""))
    return added, True


def active_waiting(state: dict[str, Any]) -> list[str]:
    return sorted(s for s, r in (state.get("waiting") or {}).items()
                  if r.get("status") == "waiting")


def expire_waiting_cycles(state: dict[str, Any], today: str,
                          cooldown_weeks: int) -> list[str]:
    """Retire waiters whose canonical first-breakout cycle has fully expired."""
    try:
        weeks = max(1, int(cooldown_weeks))
        today_date = as_date(today)
    except (TypeError, ValueError):
        return []
    if today_date is None:
        return []
    expired: list[str] = []
    for sym in active_waiting(state):
        rec = state["waiting"][sym]
        first = as_date(rec.get("breakout_bar") or rec.get("added_at"))
        if first is None or today_date <= first + timedelta(weeks=weeks):
            continue
        rec.update(status="expired", resolved_at=today,
                   resolved_reason=f"first breakout cycle expired after {weeks} weeks")
        expired.append(sym)
    if expired:
        log.info("waiting-list sweep: expired %d breakout cycle(s) (%s)",
                 len(expired), ", ".join(expired[:12]))
    return expired


def first_zone_after_breakout(rec: dict[str, Any],
                              ctx: dict[str, Any] | None) -> dict[str, Any] | None:
    """
    The first precision OB born AFTER the weekly breakout.

    A weekly breakout is a level event (close > 26W high). The precision OB
    that matters is the one that forms AFTER that breakout, not an older
    zone that happened to be lying around from months before. Without this
    filter an old tap is considered here; breakout-cycle completion is handled
    separately by sweep_waiting_list using the actual tap date.

    SMSPHARMA is the case that motivated the sweep: it broke out on
    2026-09-11, formed its first post-breakout OB on 2026-09-11 as well
    (origin 2026-09-10), and that OB was tapped on 2026-09-17. Once that tap
    is done the name has completed its "until tapped or invalidated"
    lifetime and should leave the waiting list - it is retired, not waiting.

    PTCIL is the counter-case that must stay: it broke out, formed a zone,
    but that zone has never been tapped, so it remains armed and the name
    stays on the list.

    Args:
        rec: waiting-list record with breakout_bar (ISO timestamp).
        ctx: persisted replay context with a "zones" list.

    Returns:
        The earliest zone dict whose born_session is >= breakout date, or None
        when there is no zone yet or the breakout date cannot be parsed.
    """
    if not ctx or not isinstance(ctx.get("zones"), list):
        return None
    breakout_date = as_date(rec.get("breakout_bar") or rec.get("added_at"))
    if breakout_date is None:
        return None
    # zones are not guaranteed to be sorted; sort by born_session to find
    # the earliest one that is on or after the breakout.
    candidates = []
    for z in ctx.get("zones") or []:
        if not isinstance(z, dict):
            continue
        born = as_date(z.get("born_session"))
        if born is None:
            continue
        if born >= breakout_date:
            candidates.append((born, z))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def sweep_waiting_list(state: dict[str, Any], zones: dict[str, Any],
                       alerts: dict[str, Any], today: str) -> list[str]:
    """
    Retire a breakout cycle after a zone has tapped on a prior session.

    A zone may have formed just before the weekly breakout; if it taps after
    that breakout, the stock has already delivered the setup. A later zone's
    own TAP 1 must not be presented as the stock's first tap. This sweep uses
    persisted zone tap counters/session dates, with alert de-duplication keys as
    a compatibility fallback for older state files.

    A zone tapped BEFORE the breakout does not retire the new cycle. A tap on
    today's session is left to normal event delivery so its alert can be sent;
    the waiting record is resolved after successful delivery.

    Returns:
        List of symbols retired by this sweep.
    """
    retired: list[str] = []
    waiting = state.get("waiting") or {}
    for sym in active_waiting(state):
        rec = waiting.get(sym)
        ctx = zones.get(sym) or {}
        breakout_date = as_date(rec.get("breakout_bar") or rec.get("added_at"))
        if breakout_date is None:
            continue

        # The trade is complete when ANY zone is tapped on/after the canonical
        # breakout date, even if that zone was born just before the breakout.
        # This prevents a later OB in the same breakout cycle from generating a
        # fresh symbol-level TAP 1 after the stock already tapped its earlier OB.
        tapped_zone = None
        for z in ctx.get("zones") or []:
            if not isinstance(z, dict) or int(z.get("taps") or 0) <= 0:
                continue
            tap_day = as_date(z.get("tap_session"))
            if tap_day is not None and breakout_date <= tap_day < as_date(today):
                tapped_zone = z
                break

        # Backward-compatible path for old state where zone tap counters were
        # not persisted but the first post-breakout zone's alert key was.
        if tapped_zone is None:
            first = first_zone_after_breakout(rec, ctx)
            if first:
                sig = first.get("signature") or (
                    f"{first.get('born_session')}|{first.get('top')}|{first.get('bottom')}")
                if any(k.startswith(f"{sym}|{sig}|tap") for k in (alerts or {})):
                    tapped_zone = first

        if tapped_zone:
            rec["status"] = "tapped"
            rec["resolved_at"] = today
            rec["resolved_reason"] = (
                f"OB {tapped_zone.get('born_session')} tapped on/after breakout - sweep")
            retired.append(sym)
    if retired:
        log.info("waiting-list sweep: retired %d (%s)", len(retired),
                 ", ".join(retired[:12]) + (" ..." if len(retired) > 12 else ""))
    return retired


# --------------------------------------------------------------------------- #
#  Sessions
# --------------------------------------------------------------------------- #
def as_date(v: Any) -> date | None:
    """Normalise a bar timestamp (Timestamp/datetime/date/str) to a session date."""
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    s = str(v or "")[:10]
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        return None


# The daily bar is only treated as CLOSED five minutes after the bell. At 15:30
# sharp the feed is still settling, and a refresh that ran then would freeze a
# bar missing its last prints - and `post_close_done` would stop the 15:35 run
# from fixing it. Five minutes costs nothing (the cron still gets the 15:35 and
# 15:40 runs, since market_is_open allows ten) and buys a complete candle.
POST_CLOSE_GRACE_MIN = 5

# How many failed refreshes make a data OUTAGE rather than a broken symbol.
OUTAGE_MIN_SAMPLE = 2


def session_closed(now: datetime, cfg) -> bool:
    """True once the NSE cash session is over for the day (or it is a weekend)."""
    if now.weekday() >= 5:
        return True
    # Keep `now`'s tzinfo: datetime.combine() drops it, and comparing an aware
    # "now" against a naive close time raises instead of answering.
    close = (datetime.combine(now.date(), parse_hhmm(cfg.runtime.market_close),
                              tzinfo=now.tzinfo)
             + timedelta(minutes=POST_CLOSE_GRACE_MIN))
    return now >= close


# --------------------------------------------------------------------------- #
#  Closed-bar replay (once per session per symbol)
# --------------------------------------------------------------------------- #
def event_dict(e) -> dict[str, Any]:
    """
    Flatten an OBEvent into the plain-dict shape the Telegram formatters and the
    alert de-dupe keys use, so persisted (JSON) and in-memory events are handled
    by exactly the same code path.

    `signature` is the zone's identity - born session plus its frozen top and
    bottom. It is deliberately NOT the zone's positional index, which shifts
    every day as the fetch window slides; keying on that would re-alert the same
    order block on every single run.
    """
    return {
        "kind": e.kind, "signature": e.zone.signature(), "tap_number": e.tap_number,
        "session": e.bar_session, "price": e.price, "low": e.low, "high": e.high,
        "top": e.zone.top, "bottom": e.zone.bottom, "entry": e.zone.entry,
        "stop": e.zone.stop, "atr": e.atr, "rvol": e.rvol,
        "born_session": e.zone.born_session, "origin_session": e.zone.origin_session,
        "detail": dict(e.detail or {}), "reason": e.reason, "confirmed": e.confirmed,
    }


def refresh_symbol(client: DhanClient, rec: dict[str, Any], params: OBParams,
                   sessions: int, now: datetime, closed_today: bool) -> dict[str, Any] | None:
    """
    Pull daily history, replay the CLOSED bars, and return the context the
    intraday pass continues from.

    Today's bar counts as closed only once the session is over, which is what
    lets a new order block be reported the same evening instead of next
    morning. During the session it is excluded here and judged live instead —
    that split is the whole non-repainting argument.
    """
    today = now.date()
    to_d = today
    from_d = today - timedelta(days=int(sessions * 1.7) + 20)
    # chunk_days=365 only matters on the Dhan fallback (yfinance answers the
    # whole window in one request), where a >1-year daily ask can come back
    # short - and where a symbol listed inside the window would otherwise raise
    # NoDataError for the chunk that predates its listing instead of returning
    # the chunk that does not.
    df = client.daily_candles(rec["security_id"], rec["exchange_segment"],
                              from_d, to_d, chunk_days=365,
                              symbol=rec.get("symbol"))
    if df is None or df.empty:
        return None
    bars = bars_from_frame(df)
    if not bars:
        return None
    # The fetch window is deliberately wider than `sessions` in CALENDAR days so
    # holidays cannot shorten the series; trim back to the configured number of
    # bars so the replay cost stays bounded. Trimming shifts every positional
    # index by one each day - which is exactly why zone identity is the born
    # SESSION and never born_index (see Zone.signature).
    bars = bars[-max(params.atr_len + params.vol_len, sessions):]

    closed, todays = [], None
    for b in bars:
        d = as_date(b.time)
        if d is None:
            continue
        if d < today:
            closed.append(b)
        elif d == today:
            todays = b                        # last one wins
    if closed_today and todays is not None:
        closed.append(todays)
    if not closed:
        return None

    res = replay(closed, params)
    if not res.atr or res.atr[-1] != res.atr[-1]:
        # No closed ATR (fewer than atr_len sessions of history): nothing can be
        # armed, tapped or judged, and a NaN in the state file would poison the
        # change detection in save_state. Treat it as "no history".
        return None
    ctx = res.context(closed, params)
    ctx["zones_seen"] = sum(1 for e in res.events if e.kind == "ob")
    ctx["refreshed_on"] = today.isoformat()
    ctx["post_close_done"] = today.isoformat() if closed_today else None
    # A new OB on a CLOSED bar is alertable in its own right, so the refresh
    # returns its events too; the live pass can only ever add taps.
    ctx["closed_events"] = [event_dict(e) for e in res.events
                            if e.kind in ("ob", "tap", "invalid")]
    return ctx


def prune_closed_events(ctx: dict[str, Any], kinds: list[str], tap_numbers: list[int],
                        not_before: str) -> None:
    """
    Trim the persisted event list to the ones that could still be alerted.

    A full replay emits an event for EVERY order block the window contains - up
    to `max_zones` of them, months old. Keeping all of them would put hundreds
    of kilobytes per symbol into a file the workflow commits, to no purpose: the
    next refresh recomputes them anyway. What must survive a run is only the
    small set that is recent enough and enabled enough to be sent, so that a
    failed delivery can be retried later the same session.
    """
    keep = []
    for ev in ctx.get("closed_events") or []:
        if ev.get("kind") not in kinds:
            continue
        if ev.get("kind") == "tap" and int(ev.get("tap_number") or 0) not in tap_numbers:
            continue
        if str(ev.get("session", "")) < not_before:
            continue
        keep.append(ev)
    ctx["closed_events"] = keep


# --------------------------------------------------------------------------- #
#  Alert text
# --------------------------------------------------------------------------- #
def _breakout_line(rec: dict[str, Any]) -> str:
    bar = str(rec.get("breakout_bar") or "")
    when = bar.replace("T", " ").replace("+05:30", " IST")
    px = rec.get("breakout_price")
    lvl = rec.get("level_26w")
    txt = f"📅 Weekly breakout {when} @ <b>{_fmt(px)}</b>" if px else "📅 Weekly breakout"
    if lvl:
        pct = (px / lvl - 1.0) * 100.0 if px else 0.0
        txt += f" &gt; 26W <b>{_fmt(lvl)}</b> (+{pct:.2f}%)"
    return txt


def format_ob(ev: dict[str, Any], rec: dict[str, Any], sym: str) -> str:
    """🎯 a new precision order block, born on a closed daily bar."""
    d = ev.get("detail") or {}
    lines = [
        f"🎯 <b>PRECISION OB — {_esc(sym)}</b>",
        f"<code>{_esc(rec.get('exchange_segment', 'NSE_EQ'))}</code> · daily order block "
        f"from the <b>{_esc(ev.get('born_session', ''))}</b> displacement",
        f"Zone <b>{_fmt(ev.get('top'))}</b> – <b>{_fmt(ev.get('bottom'))}</b> "
        f"(open→low of the {_esc(d.get('origin_session', ''))} origin candle)",
        f"▶ Pre-order entry <b>{_fmt(ev.get('entry'))}</b>   "
        f"⛔ stop <b>{_fmt(ev.get('stop'))}</b>",
        f"ATR {_fmt(ev.get('atr'))} · RVOL {_fmt(ev.get('rvol'))} · "
        f"zone width {_fmt(d.get('width'))}",
        _breakout_line(rec),
        "<i>Zone is frozen — born on a closed bar, never recomputed.</i>",
    ]
    return "\n".join(lines)


def tapped_entry(ev: dict[str, Any]) -> Any:
    """
    The pre-order level that was actually touched.

    Tap 1 raises the zone's entry to the defended low, and the event carries the
    zone AFTER that mutation, so printing `ev["entry"]` would show a level price
    has already left. The pre-raise value is captured in the event detail.
    """
    d = ev.get("detail") or {}
    return d.get("tapped_entry", ev.get("entry"))


def format_tap(ev: dict[str, Any], rec: dict[str, Any], sym: str) -> str:
    """🟠 price has tapped the pre-order level."""
    n = int(ev.get("tap_number") or 1)
    d = ev.get("detail") or {}
    live = not ev.get("confirmed", False)
    ltp = ev.get("price")
    lines = [
        f"🟠 <b>TAP {n} — {_esc(sym)}</b>",
        ("⚡ <i>live intrabar touch — not close-confirmed</i>" if live
         else "✅ closed-bar touch"),
        f"Entry <b>{_fmt(tapped_entry(ev))}</b> · session low "
        f"<b>{_fmt(ev.get('low'))}</b>" + (f" · LTP <b>{_fmt(ltp)}</b>" if ltp else ""),
        f"Zone {_fmt(ev.get('top'))} – {_fmt(ev.get('bottom'))} · "
        f"stop <b>{_fmt(ev.get('stop'))}</b> · ATR {_fmt(ev.get('atr'))}"
        + (f" · RVOL {_fmt(ev.get('rvol'))}" if ev.get("rvol") else ""),
        f"Born {_esc(ev.get('born_session', ''))} · tapped on "
        f"<b>{_esc(ev.get('session', ''))}</b>",
    ]
    nxt = d.get("next_entry")
    if nxt:
        touched = tapped_entry(ev)
        # The indicator adapts the entry with math.max(entry, low + atr * 0.05),
        # so it can only ever move UP. A tap defended well below the pre-order -
        # the common case on a sharp shakeout - leaves the level exactly where it
        # was, and saying "raised" then would be a lie on a live trade alert.
        moved = touched is None or float(nxt) > float(touched)
        lines.append(f"Next pre-order after this tap: <b>{_fmt(nxt)}</b> "
                     + ("(raised above the defended low)" if moved else
                        "(unchanged - the defended low closed below it)"))
    lines.append(_breakout_line(rec))
    return "\n".join(lines)


def format_message(events: list[tuple[str, dict, dict]]) -> str:
    """
    Render one Telegram message. A single event gets the full block; several get
    a header and compact blocks so a busy session is still one notification.
    """
    if len(events) == 1:
        sym, ev, rec = events[0]
        return (format_tap(ev, rec, sym) if ev["kind"] == "tap"
                else format_ob(ev, rec, sym))
    taps = sum(1 for _s, e, _r in events if e["kind"] == "tap")
    obs = len(events) - taps
    head = f"{'🟠' if taps and not obs else '🎯'} <b>{len(events)} precision-OB events</b>"
    if taps:
        head += f" — {taps} tap(s)"
    if obs:
        head += f" · {obs} new zone(s)"
    blocks = []
    for sym, ev, rec in events:
        if ev["kind"] == "tap":
            blocks.append(
                f"🟠 <b>{_esc(sym)}</b> TAP {int(ev.get('tap_number') or 1)} — "
                f"entry {_fmt(tapped_entry(ev))}, low {_fmt(ev.get('low'))}, "
                f"stop {_fmt(ev.get('stop'))}\n   {_breakout_line(rec)}")
        else:
            blocks.append(
                f"🎯 <b>{_esc(sym)}</b> new OB {_fmt(ev.get('top'))}–"
                f"{_fmt(ev.get('bottom'))}, entry {_fmt(ev.get('entry'))}, "
                f"stop {_fmt(ev.get('stop'))}\n   {_breakout_line(rec)}")
    return head + "\n\n" + "\n\n".join(blocks)


def format_heartbeat(waiting: int, refreshed: int, quoted: int, events: int,
                     errors: int, elapsed: float, now: datetime) -> str:
    return ("\n".join([
        "📊 <b>Precision OB scan complete</b>",
        f"{now.strftime('%d-%b %H:%M')} IST · {elapsed:.0f}s",
        f"Waiting list {waiting} · history refreshed {refreshed} · "
        f"live-quoted {quoted} · alerts <b>{events}</b>",
    ] + ([f"⚠️ {errors} symbol error(s)"] if errors else [])))


# --------------------------------------------------------------------------- #
#  The daily waiting-list digest
# --------------------------------------------------------------------------- #
SLOT_LABELS = {"post_close": "post-close recap", "pre_open": "pre-open plan",
               "manual": "on demand"}


def _pre_open_time(ob):
    """
    `digest_pre_open_at` as a time, or None when it does not parse.

    A typo in a COSMETIC knob must not take the tap alerts down with it. This is
    read on the scan path as well as the digest path, so letting parse_hhmm raise
    here would red-flag every run of the day and stop the live pass - the exact
    opposite of what a tracking aid is for. Say so loudly, drop the pre-open
    slot, and carry on scanning.
    """
    try:
        return parse_hhmm(ob.digest_pre_open_at)
    except (AttributeError, TypeError, ValueError):
        log.error("ob_precision.digest_pre_open_at=%r is not HH:MM - the pre-open "
                  "digest is off for this run (scanning is unaffected)",
                  ob.digest_pre_open_at)
        return None


def digest_slot(now: datetime, cfg, ob) -> str | None:
    """
    Which of the day's two digest slots this run falls in, or None.

    "post_close" from five minutes after the bell, "pre_open" from
    `digest_pre_open_at` up to the bell. Everything in between is the ordinary
    five-minute scan and sends no list: a digest every five minutes would be
    noise rather than something to track by hand.
    """
    if now.weekday() >= 5:
        return None                      # the cron does not run at the weekend
    if session_closed(now, cfg):
        return "post_close"
    at = _pre_open_time(ob)
    if at is not None and not market_is_open(cfg, now) and now.time() >= at:
        return "pre_open"
    return None


def digest_due(state: dict[str, Any], slot: str | None, today: str, ob) -> bool:
    """
    One digest per slot per day, and only while that slot is switched on.

    Anything that is not one of the two scheduled slots - a weekend, an evening
    run, mid-session - is never "due" on its own. Only an explicit --digest or
    --digest-only sends off-schedule, and that is labelled "manual" so it cannot
    spend either of the day's real slots.
    """
    if not ob.daily_digest or slot not in ("post_close", "pre_open"):
        return False
    if slot == "post_close" and not ob.digest_after_close:
        return False
    if slot == "pre_open" and not ob.digest_before_open:
        return False
    return (state.get("digest_on") or {}).get(slot) != today


def post_close_pending(waiting: list[str], zones: dict[str, Any], today: str) -> list[str]:
    """Waiting names whose post-close replay has not run yet today."""
    return [s for s in waiting
            if (zones.get(s) or {}).get("post_close_done") != today]


def last_scan_of_day(now: datetime, cfg) -> bool:
    """
    True on the final run the cron will make today: `market_is_open` gives up ten
    minutes after the bell, so 15:40 IST is the last one.

    The post-close digest would rather wait until every symbol's cache is current
    for today, but it must not wait past this run - a large list spreads its
    refreshes over two runs by design, and one permanently broken symbol would
    otherwise cost the day's list entirely.
    """
    end = datetime.combine(now.date(), parse_hhmm(cfg.runtime.market_close),
                           tzinfo=now.tzinfo)
    return now >= end + timedelta(minutes=10)


def _d(iso: Any) -> str:
    """2026-08-26 -> 26-Aug. Short, because this prints ~370 times a message."""
    day = as_date(iso)
    return day.strftime("%d-%b") if day else "?"


def _digest_row(sym: str, rec: dict[str, Any], ctx: dict[str, Any] | None) -> str:
    """One waiting name on one line: what it broke out on, and what is armed."""
    ctx = ctx or {}
    brk = f"brk {_d(str(rec.get('breakout_bar') or '')[:10])}"
    if rec.get("breakout_price"):
        brk += f" @{_fmt(rec.get('breakout_price'))}"
        if rec.get("level_26w"):
            brk += f" >{_fmt(rec.get('level_26w'))}"
    live = [z for z in (ctx.get("zones") or []) if isinstance(z, dict)]
    if live:
        z = max(live, key=lambda z: str(z.get("born_session") or ""))
        arm = (f"OB {_d(z.get('born_session'))} entry {_fmt(z.get('entry'))} "
               f"stop {_fmt(z.get('stop'))}")
        if int(z.get("taps") or 0):
            arm += f" · tapped {_d(z.get('tap_session'))}"
        if len(live) > 1:
            arm += f" · +{len(live) - 1} older zone(s)"
    elif ctx.get("no_history"):
        arm = "no daily history"
    else:
        arm = "no live zone yet"
    return f"<b>{_esc(sym)}</b> · {brk} · {arm}"


# Telegram's hard limit is 4096 characters. `telegram._split` would chop a long
# list for us, but it chops blindly: everything after the first chunk arrives
# with no header, and 266 waiting names is four chunks. Paging here instead means
# each message says what it is and which part of the list it carries.
DIGEST_PAGE_CHARS = 3800


def format_digest(state: dict[str, Any], zones: dict[str, Any], now: datetime,
                  slot: str, ob, added: list[str] | None = None) -> list[str]:
    """
    The whole active waiting list, newest breakout first, as one string per
    Telegram message.

    This is the manual-tracking view: alerts only ever name the symbols that did
    something, so a name sitting armed and untouched for three weeks is invisible
    in the chat even though it is exactly the one worth watching. Every level
    printed here is already frozen in the cache - nothing is recomputed for the
    message, so it cannot disagree with an alert.
    """
    waiting = state.get("waiting") or {}
    active = [s for s in active_waiting(state)]
    recs = sorted(((s, waiting[s]) for s in active), key=lambda x: x[0])
    recs.sort(key=lambda x: str(x[1].get("breakout_bar") or ""), reverse=True)

    armed = sum(1 for s, _r in recs if (zones.get(s) or {}).get("zones"))
    today_iso = now.date().isoformat()
    tapped = sum(1 for _s, r in waiting.items() if r.get("status") == "tapped"
                 and str(r.get("resolved_at") or "") == today_iso)
    invalid = sum(1 for _s, r in waiting.items() if r.get("status") == "invalid"
                  and str(r.get("resolved_at") or "") == today_iso)

    label = _esc(str(SLOT_LABELS.get(slot, slot)))
    head = [
        f"📋 <b>PRECISION WAITING LIST — {label}</b>",
        f"{now.strftime('%d-%b-%Y %H:%M')} IST · <b>{len(recs)}</b> waiting · "
        f"{armed} armed"
        + (f" · +{len(added)} new today" if added else "")
        + (f" · {tapped} tapped today" if tapped else "")
        + (f" · {invalid} invalidated today" if invalid else ""),
        "<i>newest breakout first · entry/stop are the newest live zone's, "
        "frozen at birth</i>",
    ]
    if slot == "post_close":
        # A recap that quietly mixed yesterday's levels into today's would be
        # worse than no recap. The pre-open plan is EXPECTED to show yesterday,
        # so it carries no warning - this is only for the run after the bell,
        # where a failed or still-pending refresh is the exception.
        stale = post_close_pending([s for s, _r in recs], zones, today_iso)
        if stale:
            head.append(f"⚠️ {len(stale)} name(s) not replayed since the bell — "
                        f"their levels are from an earlier session")
    if not recs:
        return ["\n".join(head + ["<i>The waiting list is empty.</i>"])]

    try:
        cap = int(ob.digest_max_rows or 0)
    except (TypeError, ValueError):
        log.error("ob_precision.digest_max_rows=%r is not a number - sending the "
                  "whole list", ob.digest_max_rows)
        cap = 0
    rows = [_digest_row(s, r, zones.get(s)) for s, r in recs]
    if cap > 0 and len(rows) > cap:
        head.append(f"… showing the {cap} newest of {len(rows)} "
                    f"(digest_max_rows={cap})")
        rows = rows[:cap]

    # Same header on every page: the (k/n) suffix below is what marks a
    # continuation, and repeating the slot label means a page read on its own -
    # forwarded, or scrolled to days later - still says what it is.
    cont = head[0]
    pages: list[list[str]] = []
    cur: list[str] = []
    cur_len = 0
    for row in rows:
        while True:
            if not cur:
                cur = list(head) if not pages else [cont]
                cur_len = sum(len(x) + 1 for x in cur)
            fits = cur_len + len(row) + 1 <= DIGEST_PAGE_CHARS
            if fits or len(cur) == (len(head) if not pages else 1):
                cur.append(row)                 # a row longer than a page still
                cur_len += len(row) + 1         # has to go somewhere
                break
            pages.append(cur)
            cur, cur_len = [], 0
    if cur:
        pages.append(cur)
    if len(pages) > 1:
        for i, page in enumerate(pages):
            page[0] = page[0].replace("</b>", f" ({i + 1}/{len(pages)})</b>", 1)
    return ["\n".join(page) for page in pages]


def send_digest(state: dict[str, Any], zones: dict[str, Any], now: datetime,
                cfg, ob, tg, slot: str, today: str,
                added: list[str] | None = None) -> bool:
    """
    Deliver the list and mark the slot - in that order.

    The marker is written only once Telegram confirms delivery, which is the same
    rule the alerts follow: a failed send must be retried by the next run rather
    than silently swallowing the day's list.
    """
    pages = format_digest(state, zones, now, slot, ob, added)
    for i, page in enumerate(pages):
        if not tg.send(page):
            # Stop rather than deliver half a list: the slot stays unmarked, so
            # the next run resends the whole thing. A page may therefore arrive
            # twice after a partial failure, which beats a list with a hole in it.
            log.error("daily digest (%s) page %d/%d NOT delivered - slot left "
                      "unmarked so the next run retries", slot, i + 1, len(pages))
            return False
    state.setdefault("digest_on", {})[slot] = today
    log.info("daily digest (%s) sent: %d waiting over %d message(s)", slot,
             len(active_waiting(state)), len(pages))
    return True


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="ignore market-hours check")
    ap.add_argument("--symbols", default=None, help="comma-separated subset")
    ap.add_argument("--heartbeat", action="store_true",
                    help="send a summary even with nothing to report")
    ap.add_argument("--config", default=None)
    ap.add_argument("--state-file", default=None,
                    help="override ob_precision.state_file")
    ap.add_argument("--refresh-only", action="store_true",
                    help="rebuild the zone cache and exit (no live tap pass)")
    ap.add_argument("--digest", action="store_true",
                    help="send the waiting-list digest now, whatever the slot")
    ap.add_argument("--digest-only", action="store_true",
                    help="send the digest and exit: no history calls, no quotes")
    ap.add_argument("--announce-existing", action="store_true",
                    help="one-shot announce of all currently armed zones, "
                         "even if they were born long ago (for backfill / manual audit)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ob = cfg.ob_precision
    if not ob.enabled:
        log.info("ob_precision.enabled is false - nothing to do")
        return 0
    try:
        params = ob.params()
    except ValueError as exc:
        log.error("%s", exc)
        return 1

    started = time.time()
    now = _now()
    today = now.date().isoformat()
    # The first refresh replays ~250 sessions, which contain order blocks born
    # months ago. Alerting every one of them would bury the chat on day one, so
    # CLOSED-bar events are only alertable while still recent. The live pass
    # (today's developing bar) is never filtered - it is by definition new.
    not_before = (now.date()
                  - timedelta(days=max(0, ob.event_lookback_days))).isoformat()

    state_path = Path(args.state_file) if args.state_file else cfg.paths["ob_state"]
    state = load_state(state_path)
    # A copy of what was on disk, so save_state() can tell "something changed"
    # from "only the timestamp moved" and skip the write (and the commit) when
    # nothing did.
    original = json.loads(json.dumps(state, default=str))
    zones = state.setdefault("zones", {})

    # ---- outside market hours the only thing a run can do is send the list ---
    # The pre-open plan is built entirely from the cache that yesterday's
    # post-close replay froze, so this branch never constructs a DhanClient: a
    # 09:10 run that pulled ~370 histories would duplicate the 09:15 one and
    # spend the workflow's rate budget to learn nothing new.
    if args.digest_only or (not args.force and not market_is_open(cfg, now)):
        slot = digest_slot(now, cfg, ob)
        subset = bool(args.symbols)
        if args.digest or args.digest_only or (not subset
                                               and digest_due(state, slot, today, ob)):
            added, _source_ok = harvest_stage(cfg, ob, state, today)
            expire_waiting_cycles(state, today, cfg.runtime.breakout_cooldown_weeks)
            tg = build_telegram(cfg, dry_run=cfg.runtime.dry_run)
            # Off-schedule sends are labelled (and marked) "manual", so a forced
            # digest cannot spend the day's pre-open or post-close slot.
            send_digest(state, zones, now, cfg, ob, tg, slot or "manual", today,
                        added)
            save_state(state_path, state, original)
        else:
            log.info("market closed (%s IST) - nothing to do",
                     now.strftime("%a %H:%M"))
        return 0
    closed_today = session_closed(now, cfg)

    # ---- stage 0: the waiting list, harvested from the weekly scanner -------
    added, source_ok = harvest_stage(cfg, ob, state, today)
    expire_waiting_cycles(state, today, cfg.runtime.breakout_cooldown_weeks)

    want = None
    if args.symbols:
        want = {s.strip().upper() for s in args.symbols.split(",")}
    waiting = [s for s in active_waiting(state) if want is None or s in want]
    if not waiting:
        save_state(state_path, state, original)
        if not source_ok:
            return 2          # nothing to scan, and no source to build one from
        log.info("waiting list is empty - nothing to scan")
        return 0
    log.info("waiting list: %d active symbol(s)", len(waiting))

    tg = build_telegram(cfg, dry_run=cfg.runtime.dry_run)
    client = DhanClient(cfg.secrets.dhan_client_id, cfg.secrets.dhan_access_token,
                        data_rate=cfg.runtime.data_rate_per_sec,
                        quote_rate=cfg.runtime.quote_rate_per_sec)
    alerts_seen = state.setdefault("alerts", {})
    errors = 0

    # ---- stage 1: closed-bar replay, once per session per symbol ------------
    # `needs` is deliberately narrow. The expensive part of this scanner is a
    # per-symbol daily-history call, and it buys nothing after the first run of
    # a session - the closed bars cannot change until the next one closes.
    needs: list[str] = []
    for sym in waiting:
        rec = zones.get(sym)
        if rec is None:
            needs.append(sym)
        elif rec.get("refreshed_on") != today:
            needs.append(sym)
        elif closed_today and rec.get("post_close_done") != today:
            needs.append(sym)          # one post-close pass: today becomes a closed bar
    needs = needs[:max(1, ob.max_refresh_per_run)]
    refreshed = 0
    if needs:
        log.info("refreshing daily history for %d symbol(s)", len(needs))
        with ThreadPoolExecutor(max_workers=cfg.runtime.max_workers) as pool:
            futs = {pool.submit(refresh_symbol, client, state["waiting"][s], params,
                                ob.sessions, now, closed_today): s for s in needs}
            for fut in as_completed(futs):
                sym = futs[fut]
                try:
                    ctx = fut.result()
                except DhanError as exc:
                    errors += 1
                    log.warning("%s: %s", sym, str(exc)[:140])
                    continue
                except Exception as exc:                        # noqa: BLE001
                    errors += 1
                    log.warning("%s: unexpected %s", sym, exc)
                    continue
                if not ctx:
                    errors += 1
                    log.info("%s: no daily history", sym)
                    # Remember the miss for the rest of the session. A suspended,
                    # delisted or freshly listed name would otherwise cost a
                    # history call every five minutes, forever. A DhanError above
                    # is deliberately NOT recorded: that is usually transient and
                    # should be retried on the next run.
                    zones[sym] = {
                        "refreshed_on": today, "no_history": True,
                        "post_close_done": today if closed_today else None,
                        "zones": [], "zones_seen": 0, "closed_events": [],
                    }
                    continue
                prune_closed_events(ctx, ob.alert_kinds, ob.alert_taps, not_before)
                zones[sym] = ctx
                refreshed += 1

    # ---- stage 2: the live tap, from ONE bulk quote ------------------------
    # Only symbols that have at least one LIVE zone and whose last closed
    # session is not today are worth quoting. A name still waiting for its
    # first order block has nothing to tap, and skipping those is what keeps a
    # ~370-name backfilled list off the one-request-per-symbol path.
    quotes: dict[tuple[str, str], dict[str, Any]] = {}
    live_syms: list[str] = []
    if not closed_today and not args.refresh_only:
        for s in waiting:
            ctx = zones.get(s) or {}
            if ctx.get("zones") and ctx.get("as_of") != today:
                live_syms.append(s)
    if live_syms:
        by_seg: dict[str, list[Any]] = {}
        for sym in live_syms:
            rec = state["waiting"][sym]
            sid = str(rec["security_id"])
            by_seg.setdefault(rec["exchange_segment"], []).append(
                int(sid) if sid.isdigit() else sid)
        batch = max(1, ob.quote_batch)
        for seg, sids in by_seg.items():
            for i in range(0, len(sids), batch):
                try:
                    part = client.ohlc({seg: sids[i:i + batch]}) or {}
                except DhanError as exc:
                    errors += 1
                    log.warning("bulk quote failed (%s) - those symbols get no "
                                "live pass this run: %s", seg, str(exc)[:140])
                    continue
                for sid, q in (part.get(seg) or {}).items():
                    if q:
                        quotes[(seg, str(sid))] = q
    quoted = len(quotes)

    # ---- collect events ----------------------------------------------------
    events: list[tuple[str, dict[str, Any], dict[str, Any]]] = []

    def consider(sym: str, ev: dict[str, Any], *, closed: bool) -> None:
        if ev["kind"] not in ob.alert_kinds:
            return
        if ev["kind"] == "tap" and int(ev.get("tap_number") or 0) not in ob.alert_taps:
            return
        if closed and str(ev.get("session", "")) < not_before:
            return
        key = f"{sym}|{ev.get('signature')}|{ev['kind']}"
        if ev["kind"] == "tap":
            key += f"|tap{int(ev.get('tap_number') or 0)}"
        if key in alerts_seen:
            return                          # already sent on an earlier run
        events.append((sym, dict(ev, dedupe_key=key), state["waiting"][sym]))

    # closed-bar events: a new order block, or a tap on a session that closed
    for sym in waiting:
        for ev in (zones.get(sym) or {}).get("closed_events") or []:
            consider(sym, ev, closed=True)

    # ---- one-shot announce of existing armed zones -------------------------
    # --announce-existing is for a manual audit / backfill: it announces every
    # currently armed zone that has never been announced before, even if it was
    # born long ago. Without this a name that has been armed for three weeks
    # but never tapped is invisible in the chat - and it is exactly the one
    # worth watching by hand. The flag is one-shot: once announced, the de-dupe
    # key prevents it from firing again, so it can be run once after a cold
    # start and then forgotten.
    if args.announce_existing:
        for sym in waiting:
            ctx = zones.get(sym) or {}
            for z in ctx.get("zones") or []:
                if not isinstance(z, dict):
                    continue
                # Synthesize an OB event for the currently armed zone
                sig = z.get("signature") or (
                    f"{z.get('born_session')}|{z.get('top')}|{z.get('bottom')}"
                )
                ev = {
                    "kind": "ob",
                    "signature": sig,
                    "session": z.get("born_session"),
                    "top": z.get("top"), "bottom": z.get("bottom"),
                    "entry": z.get("entry"), "stop": z.get("stop"),
                    "atr": z.get("atr"), "rvol": z.get("rvol"),
                    "born_session": z.get("born_session"),
                    "origin_session": z.get("origin_session"),
                    "detail": {"width": z.get("width")},
                    "reason": "announce_existing",
                    "confirmed": True,
                }
                consider(sym, ev, closed=False)

    # live taps on today's developing bar
    for sym in live_syms:
        rec = state["waiting"][sym]
        q = quotes.get((rec["exchange_segment"], str(rec["security_id"])))
        if not q or not q.get("last_price"):
            continue
        ctx = zones.get(sym) or {}
        # A bulk quote carries no date. On a weekday market holiday the feed
        # hands back the PREVIOUS session's numbers unchanged, and replaying
        # them as "today" would count a second tap on an already-tapped zone.
        # Identical high/low/close against the last closed bar is that quote.
        if (float(q.get("high") or 0.0) == float(ctx.get("as_of_high", -1.0))
                and float(q.get("low") or 0.0) == float(ctx.get("as_of_low", -1.0))
                and float(q.get("last_price") or 0.0) == float(ctx.get("as_of_close", -1.0))):
            log.info("%s: quote is identical to the last closed session - "
                     "stale, skipping the live pass", sym)
            continue
        for e in live_pass(ctx.get("zones") or [], ctx, params,
                           open_=float(q.get("open") or 0.0),
                           high=float(q.get("high") or 0.0),
                           low=float(q.get("low") or 0.0),
                           last_price=float(q.get("last_price") or 0.0),
                           volume=float(q.get("volume") or 0.0),
                           session=today):
            consider(sym, event_dict(e), closed=False)

    # ---- resolve statuses from the closed replay ---------------------------
    resolved = 0
    for sym in waiting:
        rec = state["waiting"][sym]
        if rec.get("status") != "waiting":
            continue
        ctx = zones.get(sym) or {}
        if not ctx:
            continue
        seen = int(ctx.get("zones_seen") or 0)
        alive = len(ctx.get("zones") or [])
        if seen and not alive:
            # Every order block this name produced has been invalidated or
            # exhausted: that is the "or invalidated" half of the waiting-list
            # lifetime, so the name comes off the list.
            rec["status"] = "invalid"
            rec["resolved_at"] = today
            rec["resolved_reason"] = f"all {seen} order block(s) invalidated/exhausted"
            resolved += 1

    # ---- waiting-list sweep: prior tap closes the breakout cycle -----------
    # If an earlier zone tapped after the canonical breakout date, don't emit
    # another zone's TAP 1 for the same cycle. Today's first tap is not swept:
    # it still needs normal Telegram delivery before resolution.
    swept = sweep_waiting_list(state, zones, alerts_seen, today)
    resolved += len(swept)
    if swept:
        # Events may have been collected above from a newer zone on the same
        # symbol. If an earlier zone already tapped on a prior session, that
        # breakout cycle is complete and the newly collected TAP 1 is stale.
        retired = set(swept)
        # A past TAP 1 closes the symbol-level breakout setup and must block a
        # later zone's own TAP 1. Keep higher tap numbers from the same zone:
        # they remain meaningful follow-up touches when explicitly enabled.
        events = [item for item in events if not (
            item[0] in retired and
            (item[1].get("kind") == "ob" or
             (item[1].get("kind") == "tap"
              and int(item[1].get("tap_number") or 0) == 1))
        )]

    # ---- send ---------------------------------------------------------------
    sent_ok = True
    if events:
        events.sort(key=lambda x: (x[1].get("session", ""), x[0]))
        log.info("%d alert(s): %s", len(events),
                 ", ".join(f"{s}:{e['kind']}" for s, e, _r in events))
        sent_ok = tg.send(format_message(events))
        if sent_ok:
            for sym, ev, rec in events:
                alerts_seen[ev["dedupe_key"]] = {
                    "sent_at": _now().isoformat(timespec="seconds"),
                    "kind": ev["kind"], "session": ev.get("session", ""),
                    "price": ev.get("price"), "tap_number": ev.get("tap_number"),
                }
                if ev["kind"] == "tap" and ob.resolve_on_tap:
                    rec["status"] = "tapped"
                    rec["resolved_at"] = today
                    rec["resolved_reason"] = (
                        f"tap {int(ev.get('tap_number') or 1)} on the precision OB")
        else:
            # Same rule as scan.py: if the message did not arrive, do not record
            # it as sent, so the next run retries instead of swallowing it.
            log.error("Telegram delivery failed - alert keys NOT saved, "
                      "the next run retries")
    else:
        log.info("no precision-OB events")

    # ---- a total data outage deserves exactly one red run per day ----------
    # Every refresh failing means the feed is down, and this scanner would then
    # sit silently for the rest of the session with the waiting list unwatched.
    # The workflow's `Notify on failure` step already knows how to say that, so
    # the run only has to go red - but once, or a five-minute cron turns a single
    # outage into seventy-eight Telegram notices.
    # Two or more, because a single name failing is a broken symbol (delisted,
    # suspended, wrong security id), not a broken feed - and late in a session
    # one straggler is often the only thing left to refresh.
    # A run that only asked about names the feed has NEVER had bars for is not
    # evidence about the feed - it is evidence about those names. That is exactly
    # what the TAIL of a capped backfill looks like: with more waiting names than
    # max_refresh_per_run, the second run of a session asks about the
    # alphabetically-last few, and if those happen to be delisted, suspended or
    # freshly listed then every one of them comes back empty.
    #
    # Found at real scale (266 waiting names against a cap of 250): the 09:20 run
    # asked about the 16 the 09:15 run had not reached, all of them no-data in
    # that simulation, and concluded the feed was down - rc 3, a red workflow and
    # a failure notice in the chat, five minutes after 250 symbols had refreshed
    # perfectly well.
    #
    # `refreshed_on == today` is what keeps this from hiding a real outage: on
    # the FIRST run of a session nothing has been refreshed today yet, so a
    # feed-down morning fails the run exactly as it did before.
    feed_answered_today = any(
        isinstance(c, dict) and c.get("refreshed_on") == today
        and not c.get("no_history") for c in zones.values())
    outage = (len(needs) >= OUTAGE_MIN_SAMPLE and refreshed == 0
              and errors >= len(needs) and not feed_answered_today)
    new_outage = outage and state.get("data_outage_on") != today
    if new_outage:
        state["data_outage_on"] = today
    elif refreshed and state.get("data_outage_on"):
        # Real data flowed again: clear it so a LATER outage still gets reported.
        # Note the test is `refreshed`, not `not outage` - a run that refreshed
        # nothing because everything was already cached says nothing either way.
        state["data_outage_on"] = None

    # ---- the daily list, once the day's statuses are final ------------------
    # Deliberately AFTER the taps are sent and the resolved statuses are set, so
    # the counts in the header describe the day that just finished, and BEFORE
    # save_state so the slot marker is persisted with everything else.
    slot = digest_slot(now, cfg, ob)
    # A --symbols run is a debug run over a SUBSET: its cache is partial by
    # definition, so it must neither send nor spend the day's slot. An explicit
    # --digest still sends, because that is the operator asking for it by name.
    due = bool(args.digest) or (not args.symbols
                                and digest_due(state, slot, today, ob))
    if due and slot == "post_close" and not args.digest:
        pending = post_close_pending(active_waiting(state), zones, today)
        if pending and not last_scan_of_day(now, cfg):
            log.info("post-close digest deferred: %d waiting symbol(s) have not "
                     "been replayed since the bell", len(pending))
            due = False
    if due:
        send_digest(state, zones, now, cfg, ob, tg, slot or "manual", today, added)

    if save_state(state_path, state, original):
        log.info("state saved: %d waiting, %d cached, %d alert key(s), %d KB",
                 len(state.get("waiting") or {}), len(zones), len(alerts_seen),
                 state_path.stat().st_size // 1024)
    else:
        log.info("state unchanged - not rewritten")

    if args.heartbeat:
        tg.send(format_heartbeat(len(waiting), refreshed, quoted, len(events),
                                 errors, time.time() - started, now))

    log.info("done in %.0fs (waiting %d, refreshed %d, quoted %d, alerts %d, "
             "resolved %d, errors %d)", time.time() - started, len(waiting),
             refreshed, quoted, len(events), resolved, errors)
    if new_outage:
        log.error("all %d history refresh(es) failed - the data source looks "
                  "down; failing the run so the workflow sends its failure "
                  "notice (at most once a day)", len(needs))
        return 3
    if outage:
        log.error("all %d history refresh(es) failed again - already reported "
                  "today, staying green so the chat is not spammed every five "
                  "minutes", len(needs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
