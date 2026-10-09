#!/usr/bin/env python3
"""
Precision OB Entry - the "buy the OB candle itself" rule, as a live alert.

THE RULE (precision_ob_backtest.py, 06-Oct-2026 - the accepted Tap 1
configuration with an earlier entry)
--------------------------------------------------------------------------
  Event    the FIRST precision order block born on/after the 26W breakout of
           the cycle - the live scanner's own first_zone_after_breakout rule -
           with the structural row the backtest applies: the OB candle (the
           origin candle) must ALSO sit on/after the breakout session. A zone
           born on the breakout session whose origin candle predates it (the
           breakout bar itself is the displacement - the SMSPHARMA shape) is
           excluded.
  Entry    the OB candle's close, taken only when that close is ABOVE the
           26W breakout level
  Target   the highest high printed between the breakout session and the entry
           (the OB candle's) session - a sell limit
  Stop     the 26W breakout level - a stop order
  Time     stop: 90 trading sessions from the entry session, exit at the close

  Backtest 2021-26, eod2 NSE daily (split/bonus adjusted), live config.yaml
  zone defaults, 2,272 trades:
      88.5% win · avg win +7.27% · avg loss −4.16% · timeout 0.5% ·
      +5.77% net/trade · +1.62R · median win completes in 1 session
      (round-trip cost 0.22% · median reward:risk at entry 0.9x)

THE HINDSIGHT PROBLEM - AND WHAT THIS JOB DOES INSTEAD OF PRETENDING
--------------------------------------------------------------------
The OB candle only BECOMES a precision OB when the displacement bar after it
closes, so buying its close uses one bar of foresight. The displacement close
("entry B") is the first price at which the OB is knowable in real time, and
the backtest reports it beside A everywhere:

      entry                     stop         n      win   net/trade   exp R
      A  OB candle close        26W level   2,272   89%    +5.77%     +1.62
      B  displacement close     26W level   6,624   83%    +1.32%     +0.60

Every alert this job sends therefore carries BOTH prices - A as the rule's own
number, B as the price that could actually be filled - and the exit report
shows the net return from each. A is never presented as a fill you could have
taken without foresight.

WHEN ALERTS FIRE (all three read the stage-2 scanner's committed state)
-----------------------------------------------------------------------
  --mode intraday   ~15:12 IST, BEFORE the close: a name with no post-breakout
                    OB yet whose live bar is the displacement that would BIRTH
                    the rule's OB. The OB candle is resolved from the closed
                    bars with the scanner's own find_origin(), so the rule's
                    two structural rows (origin on/after the breakout, origin
                    close above the level) are checked for real. This is the
                    only moment entry B - the fillable entry - is still open.
                    (An entry-A heads-up on the OB candle's own close is
                    available as `candidate_alerts`, off by default: at that
                    point the order block does not exist yet and a red candle
                    above the level is common.)
  --mode postclose  ~16:05 IST: the confirmation. The complete trade plan (both
                    entries, stop, target, size, 90-session time stop), the
                    trade recorded in the book, and every open trade checked
                    against the target / stop / time stop.
  --mode explain    one symbol, no alerts: print the rule's funnel state -
                    cycle, first OB, OB candle, and which filter decided.
  --mode digest     no market data: print the open book and the exclusions.

DESIGN - deliberately a spectator of the live scanner
------------------------------------------------------
Like strategy_alert.py, this job READS ob_precision_state.json (the stage-2
scanner's committed cache) and never writes it, and keeps its own book in
precision_ob_entry_state.json. It fetches daily history ONLY for symbols whose
rule event is being evaluated; every other symbol costs nothing.

First-run safety: a rule event older than `catchup_sessions` (default 3) is
marked seen without an alert, so deploying this job never replays history into
the chat. An event inside that window is alerted - and if the trade already
resolved, the alert says so instead of quoting a plan nobody can take.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from config import load_config
from dhan import DhanClient, DhanError
from ob_precision import Bar, bars_from_frame, find_origin, replay
from telegram import _esc, _fmt, build_telegram

ROOT = Path(__file__).resolve().parent
STATE_IN = ROOT / "ob_precision_state.json"         # the scanner's - READ ONLY
STATE_OWN = ROOT / "precision_ob_entry_state.json"  # ours

IST = ZoneInfo("Asia/Kolkata")

# The backtest's hindsight check, quoted in every alert so the number on
# screen never drifts from the number in the report.
STATS_A = ("A · the OB candle's close: 88.5% win · avg win +7.27% · "
           "avg loss −4.16% · +5.77% net/trade · +1.62R · median win 1 session "
           "(n=2,272, 2021-26)")
STATS_B = ("B · the displacement close: 83% win · avg win +3.0% · "
           "avg loss −5.3% · +1.32% net/trade · +0.60R (n=6,624)")

log = logging.getLogger("precision_ob_entry")


# --------------------------------------------------------------------------- #
#  Settings (config.yaml `precision_ob_entry`)
# --------------------------------------------------------------------------- #
@dataclass
class RuleSettings:
    """The numbers the description pins, plus this job's alert switches."""
    enabled: bool = True
    state_file: str = "precision_ob_entry_state.json"
    # The rule's own exit numbers - both quoted in the backtest.
    time_stop_sessions: int = 90
    round_trip_cost_pct: float = 0.22
    # How many sessions late a rule event may still be alerted. GitHub's cron
    # skips slots (the repo's BUG 55) and entry A's price is history either
    # way; what matters is that a late alert never quotes a live plan for a
    # finished trade, which is why the resolution is walked first.
    catchup_sessions: int = 3
    # The three alerts, each switchable on its own.
    confirm_alerts: bool = True      # postclose: the rule event + exit tracking
    forming_alerts: bool = True      # intraday: entry B, fillable before close
    candidate_alerts: bool = False   # intraday: the OB candle's own close
    # Depth of the daily fetch used to reach back to the breakout session.
    lookback_days: int = 560


def rule_settings(cfg) -> RuleSettings:
    """The config section, defaulted - a bare Config (as every test builds)
    still yields working settings."""
    raw = getattr(cfg, "precision_ob_entry", None)
    known = set(RuleSettings.__dataclass_fields__)
    if raw is None:
        return RuleSettings()
    return RuleSettings(**{k: getattr(raw, k) for k in known
                           if hasattr(raw, k)})


# --------------------------------------------------------------------------- #
#  Small helpers (the same shapes as strategy_alert.py, deliberately
#  duplicated: a spectator job must not break when the job it watches is
#  edited, and vice versa)
# --------------------------------------------------------------------------- #
def as_date(v) -> date | None:
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    try:
        return date.fromisoformat(str(v or "")[:10])
    except ValueError:
        return None


def load_json(path: Path) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("cannot read %s: %s", path.name, exc)
        return {}


def save_state(st: dict, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(st, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def weekday_sessions(d0: date | None, d1: date | None) -> int:
    """Mon-Fri session count between two dates. NSE holidays are not modelled
    (the state carries no calendar and a normal pass must not cost a data
    call); at a 90-session time stop the drift is ~2 sessions. The bar walks,
    which DO have the bars, count real sessions instead."""
    if d0 is None or d1 is None or d1 <= d0:
        return 0
    n, d = 0, d0
    while d < d1:
        d += timedelta(days=1)
        if d.weekday() < 5:
            n += 1
    return n


def sessions_between(bars: list, d0: date | None, d1: date | None) -> int:
    """Real session count from the bars, excluding both endpoints."""
    if d0 is None or d1 is None or d1 <= d0:
        return 0
    return sum(1 for b in bars if d0 < (as_date(b.session) or d0) <= d1)


# NSE's close, as minutes since IST midnight. A session is only "completed"
# once the clock passes this.
CLOSE_IST_MIN = 15 * 60 + 30


def session_date(now: datetime | None = None) -> date:
    """The most recent COMPLETED session as of `now` (IST).

    BUG (08-Oct-2026 audit): the postclose pass used `datetime.now(IST).date()`.
    The workflow picks its pass from the UTC hour, so postclose runs from
    15:30 IST until 05:29 IST - and from midnight onwards `now().date()` names
    a session that HAS NOT HAPPENED YET. Every `weekday_sessions(..., today)`
    in the catch-up window counted that phantom session, inflating the OB
    candle's age and the event's lateness by one. An event sitting exactly on
    the window boundary at 23:45 IST was silently suppressed - and marked seen
    forever - from 00:00, so roughly 22 overnight runs a night could kill an
    alert the next 16:10 slot would have sent. EIMCOELECO passed at ob_age
    exactly == catchup_sessions; one session of phantom inflation and it would
    never have alerted at all.

    Weekends roll back to Friday. NSE holidays are not modelled, exactly as in
    `weekday_sessions` - a holiday makes this one session optimistic, which is
    the same drift the rest of the window already carries.
    """
    now = now or datetime.now(IST)
    d = now.date()
    if now.hour * 60 + now.minute < CLOSE_IST_MIN:
        d -= timedelta(days=1)          # today's session is not complete yet
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def _sent(st: dict, key: str) -> bool:
    return key in st.setdefault("sent", {})


def _mark(st: dict, key: str) -> None:
    st["sent"][key] = datetime.now(IST).isoformat(timespec="seconds")


def _cycle_key(sym: str, brk: dict) -> str:
    """The rule yields exactly ONE event per breakout cycle - the FIRST
    precision OB born on/after the cross - so the cycle is the stable identity
    of that event."""
    return f"{sym}|{str(brk.get('session'))[:10]}"


def _cycle_handled(st: dict, ckey: str) -> bool:
    """True when this cycle's event was already resolved on an earlier run.

    BUG (08-Oct-2026 audit): the dedupe key is `rule|SYM|BORN`, but `born` is
    re-derived from freshly fetched bars after the early `_sent()` gate has
    already run. When the state's birth and the bars' birth disagree - ARFIN
    said 2026-10-06, its bars said 2026-04-13 - the early gate never matched,
    so the symbol fell through to a full 560-day daily-history fetch on EVERY
    run and was only short-circuited afterwards. At the live 15-minute cadence
    that is ~210 wasted history requests a day on one already-excluded name,
    and it contradicts the documented cost shape ("NO market-data call unless a
    rule event fires"). Recording the cycle beside the key closes the gap
    without touching the key format, so no existing marker is invalidated.
    """
    marked = (st.get("cycles") or {}).get(ckey)
    return bool(marked) and marked in st.get("sent", {})


def _mark_cycle(st: dict, ckey: str, key: str) -> None:
    st.setdefault("cycles", {})[ckey] = key


# `last_run` carries a clock time, so it differs on every run and would force a
# commit even when nothing happened. It is the one field excluded from the
# quiet-run comparison; `last_run` itself is session-scoped so it still records
# which pass covered which session.
_VOLATILE = ("last_run_at",)


def _book_changed(before: dict, after: dict) -> bool:
    def _stable(st: dict) -> dict:
        return {k: v for k, v in (st or {}).items() if k not in _VOLATILE}
    return _stable(before) != _stable(after)


def _send_alerts(tg, messages: list[str]) -> bool:
    """Send every queued alert; report whether all were confirmed. The caller
    persists dedupe/trade state only after successful delivery, so an
    undelivered signal is retried rather than silently lost."""
    all_sent = True
    for i, message in enumerate(messages, start=1):
        try:
            sent = tg.send(message)
        except Exception as exc:                  # noqa: BLE001 - degrade
            log.error("Telegram send %d/%d raised: %s", i, len(messages), exc)
            all_sent = False
            continue
        if not sent:
            log.error("Telegram did not confirm alert %d/%d", i, len(messages))
            all_sent = False
    return all_sent


def _dry_run(cfg, args, tg) -> bool:
    return bool(getattr(args, "dry_run", False)
                or getattr(cfg.runtime, "dry_run", False)
                or getattr(tg, "dry_run", False))


# --------------------------------------------------------------------------- #
#  Detection (pure: state in, events out) - the live scanner's own rule
# --------------------------------------------------------------------------- #
def breakout_of(rec: dict, ctx: dict) -> dict | None:
    """{session, level, close} of this cycle's candle-derived c02 cross.

    The backtest dates its cycles exactly this way ("26W breakout cycles are
    dated from the candle-derived c02 cross (derive_26w_breakout semantics)
    with the 26-week cross-lock"), which is the field the scanner persists as
    `breakout_26w_session`. The scanner's own first_zone_after_breakout() reads
    the RECORDED anchor (`breakout_bar`) instead; that anchor is a 5-minute
    alert timestamp and exists to keep an armed symbol stable, not to date the
    cycle's history. The level is the same number either way, and an empty
    c02 dict means the derivation ran and could not answer - in which case the
    rule has no boundary to work from and nothing is invented."""
    for src in (rec, ctx):
        brk = (src or {}).get("breakout_26w_session")
        if isinstance(brk, dict) and brk.get("session") and brk.get("level"):
            return brk
    return None


def _norm_zone(z: dict, born: str, kind: str) -> dict:
    """One shape for both sources: state zone dicts (live) and closed events."""
    d = dict(z or {})
    d["born_session"] = born
    d["origin_session"] = d.get("origin_session") \
        or (d.get("detail") or {}).get("origin_session") or ""
    d["_source"] = kind
    return d


def ob_births(rec: dict, ctx: dict) -> list[tuple[str, dict]]:
    """Every precision OB born ON/AFTER this cycle's breakout session,
    earliest first - the universe the live scanner's first_zone_after_breakout
    rule chooses from.

    Both sources are read because they cover different lifetimes: the live
    `zones` list carries everything still alive, and `closed_events` carries
    events already pruned from it. A closed event wins a tie - it carries the
    displacement bar's own price."""
    brk = breakout_of(rec, ctx)
    if not brk:
        return []
    b_day = as_date(brk["session"])
    if b_day is None:
        return []
    seen: dict[str, dict] = {}
    for z in ctx.get("zones") or []:
        if not isinstance(z, dict):
            continue
        born = str(z.get("born_session") or "")[:10]
        if as_date(born) and as_date(born) >= b_day:
            seen.setdefault(born, _norm_zone(z, born, "zone"))
    for ev in ctx.get("closed_events") or []:
        if not isinstance(ev, dict) or ev.get("kind") != "ob":
            continue
        born = str(ev.get("born_session") or ev.get("session") or "")[:10]
        if as_date(born) and as_date(born) >= b_day:
            seen[born] = _norm_zone(ev, born, "event")
    return sorted(seen.items(), key=lambda kv: kv[0])


def first_ob_after_breakout(rec: dict, ctx: dict) -> tuple[str, dict] | None:
    """The cycle's FIRST precision OB born on/after the breakout - the event
    the rule replays. A later OB of the same cycle is a different (unmeasured)
    trade, so it is never the event."""
    births = ob_births(rec, ctx)
    return births[0] if births else None


def births_from_bars(bars: list, brk: dict, params) -> list[tuple[str, object]]:
    """Every precision OB born on/after the breakout session, recomputed from
    the bars by the scanner's OWN port (`ob_precision.replay`, the Pine-exact
    code `ob_tap_scan.py` runs).

    The committed state is a trigger, not the truth: its `zones` list holds
    only the zones still alive and its `closed_events` list is pruned, so a
    cycle whose first OB has since died can look like it starts at a LATER
    order block. Replaying the fetched bars puts the event back on the
    indicator's own terms - if the true first OB was born earlier, the later
    one is not this cycle's event, and this job will not present it as one."""
    b_day = as_date((brk or {}).get("session") or "")
    if b_day is None or not bars:
        return []
    out: list[tuple[str, object]] = []
    for e in replay(bars, params).events:
        if e.kind != "ob":
            continue
        d = as_date(e.zone.born_session)
        if d is not None and d >= b_day:
            out.append((e.zone.born_session, e.zone))
    out.sort(key=lambda kv: kv[0])
    return out


def zone_rvol(zone) -> float | None:
    """The displacement rvol of a birth, from a Zone object or a state dict."""
    if zone is None:
        return None
    rvol = getattr(zone, "rvol_at_birth", None)
    if rvol is None and isinstance(zone, dict):
        rvol = zone.get("rvol") or (zone.get("detail") or {}).get("rvol")
    try:
        return float(rvol) if rvol is not None else None
    except (TypeError, ValueError):
        return None


def resolve_origin(bars: list, born_session: str, params) -> Bar | None:
    """The OB candle, recomputed from the bars with the indicator's own
    find_origin: the nearest bearish or neutral candle within `origin_search`
    bars before the displacement. Recomputed rather than read off the state so
    this job applies the indicator's rule, not a copy of a copy of it."""
    idx = next((i for i, b in enumerate(bars) if b.session == born_session),
               None)
    if idx is None or idx <= 0:
        return None
    off = find_origin(bars, idx, params)
    if not off or idx - off < 0:
        return None
    return bars[idx - off]


# --------------------------------------------------------------------------- #
#  The rule
# --------------------------------------------------------------------------- #
@dataclass
class Rule:
    """One evaluation of the described rule on one OB birth."""
    ok: bool
    reason: str = ""
    level: float | None = None
    breakout_session: str = ""
    born_session: str = ""
    origin_session: str = ""
    entry_a: float | None = None          # the OB candle's close
    entry_b: float | None = None          # the displacement close
    stop: float | None = None             # = the 26W breakout level
    target: float | None = None           # highest high, breakout -> OB candle


def evaluate_rule(bars: list, brk: dict, born_session: str, params) -> Rule:
    """Apply the rule exactly as the backtest describes it:

      event    the first precision OB born on/after the 26W breakout
      row 1    the OB candle (the origin) must sit on/after the breakout
               session - "OB candle predates the breakout (zone born on the
               breakout, origin the day before)" is an exclusion
      row 2    the OB candle's close must be ABOVE the 26W breakout level -
               "OB close back below the 26W level" is an exclusion
      entry    the OB candle's close (entry A)
      target   the highest high printed between the breakout session and the
               entry session - it EXCLUDES the displacement bar and always
               includes the OB candle's own high, so the target can never sit
               below the entry
      stop     the 26W breakout level
      time     90 sessions
    """
    level = brk.get("level")
    brk_s = str(brk.get("session") or "")[:10]
    if level is None or not brk_s:
        return Rule(False, "the 26W breakout level is unknown")
    level = float(level)
    born = next((b for b in bars if b.session == born_session), None)
    origin = resolve_origin(bars, born_session, params)
    if origin is None:
        return Rule(False, f"no origin candle resolved for the {born_session} "
                           "displacement", level=level,
                    breakout_session=brk_s, born_session=born_session)
    origin_s = origin.session
    if origin_s < brk_s:
        return Rule(False, "OB candle predates the breakout (zone born on the "
                           "breakout, origin the day before)", level=level,
                    breakout_session=brk_s, born_session=born_session,
                    origin_session=origin_s)
    if origin.close <= level:
        return Rule(False, "OB close back below the 26W level", level=level,
                    breakout_session=brk_s, born_session=born_session,
                    origin_session=origin_s, entry_a=origin.close)
    highs = [b.high for b in bars if brk_s <= b.session <= origin_s]
    if not highs:
        return Rule(False, "no bars between the breakout and the OB candle",
                    level=level, breakout_session=brk_s,
                    born_session=born_session, origin_session=origin_s)
    return Rule(True, "the cycle's first post-breakout precision OB with the "
                      "OB candle above the 26W level",
                level=level, breakout_session=brk_s,
                born_session=born_session, origin_session=origin_s,
                entry_a=origin.close,
                entry_b=(born.close if born else None), stop=level,
                target=max(highs))


# --------------------------------------------------------------------------- #
#  Fills - gap-aware, the backtest's own policy
# --------------------------------------------------------------------------- #
def resolve_bar(entry: float, stop: float, target: float | None,
                o: float | None, h: float, l: float
                ) -> tuple[str, float, str, str] | None:
    """(outcome, price, reason, fill) for one session, or None when neither
    order filled.

    The backtest's policy verbatim: "through the target fills at the open
    (better), through the stop at the open (worse); a session touching both
    resolves by its open, else counts conservatively as a loss". The first two
    branches are the "resolves by its open" half; when neither order is gapped
    through and both are touched inside the bar, the conservative loss is what
    remains."""
    if target is not None and o is not None and o >= target:
        return ("win", float(o), "target gapped through - filled at the open "
                "(better than the limit)", "gap-open")
    if o is not None and o <= stop:
        return ("loss", float(o), "stop gapped through - filled at the open "
                "(worse than the stop)", "gap-open")
    if target is not None and h >= target and l <= stop:
        return ("loss", float(stop), "both orders touched in one session - "
                "resolved conservatively as a loss", "order")
    if target is not None and h >= target:
        return ("win", float(target), "swing-high target filled", "order")
    if l <= stop:
        return ("loss", float(stop), "26W level stop filled", "order")
    return None


def walk_bars(trade: dict, bars: list, through: str | None,
              time_stop_sessions: int) -> dict | None:
    """Every session strictly AFTER the entry session and up to `through`,
    applying the fill policy with the real opens it needs. Returns the first
    resolution, or None when the trade is still open as of `through`."""
    d0 = as_date(trade.get("entry_session"))
    if d0 is None:
        return None
    end = as_date(through)
    for b in bars:
        d = as_date(b.session)
        if d is None or d <= d0:
            continue
        if end is not None and d > end:
            break
        res = resolve_bar(float(trade["entry"]), float(trade["stop"]),
                          float(trade["target"]) if trade.get("target") else None,
                          b.open, b.high, b.low)
        held = sessions_between(bars, d0, d)
        if res:
            outcome, price, reason, fill = res
            r_dict = {"outcome": outcome, "session": b.session, "price": price,
                      "reason": reason, "fill": fill, "sessions": held}
            if trade.get("born_session") and b.session < trade["born_session"]:
                r_dict["pre_confirmation"] = True
                r_dict["never_live"] = True
            return r_dict
        if time_stop_sessions and held >= time_stop_sessions:
            r_dict = {"outcome": "timeout", "session": b.session,
                      "price": b.close, "fill": "close", "sessions": held,
                      "reason": f"{time_stop_sessions}-session time stop - exit "
                                "at the close"}
            if trade.get("born_session") and b.session < trade["born_session"]:
                r_dict["pre_confirmation"] = True
                r_dict["never_live"] = True
            return r_dict
    return None


def b_walk(trade: dict, ex: dict, bars: list | None,
           settings: RuleSettings, through: str | None = None) -> dict | None:
    """Entry B's own exit, from the same bars.

    B buys the displacement bar's close, so its first walkable session is the
    one AFTER the born session - usually one session later than A. Reusing A's
    exit price for B is therefore wrong whenever A resolves on or before the
    born session (the MANINDS shape: the displacement bar itself gapped through
    the target, A is a win, B has not held a single session). B is walked over
    its own sessions and reported as still open when it has not resolved.

    Returns B's exit dict, {"open": True} when it has not resolved by
    `through` (A's own exit session unless the caller says otherwise), or None
    when there is no B trade to walk at all."""
    born, entry_b = trade.get("born_session"), trade.get("entry_b")
    if not born or not entry_b or not bars:
        return None
    b_trade = {"entry": float(entry_b), "stop": float(trade["stop"]),
               "target": float(trade["target"]) if trade.get("target") else None,
               "entry_session": born}
    res = walk_bars(b_trade, bars, through=through or str(ex.get("session") or ""),
                    time_stop_sessions=settings.time_stop_sessions)
    return res or {"open": True}


def check_exit(trade: dict, ctx: dict, time_stop_sessions: int) -> dict | None:
    """One open trade against the scanner state's latest closed bar - zero
    market-data calls, the normal daily pass."""
    as_of = ctx.get("as_of")
    if not as_of or str(as_of) <= str(trade.get("last_checked") or ""):
        return None
    h, l = ctx.get("as_of_high"), ctx.get("as_of_low")
    if h is None or l is None:
        return None
    res = resolve_bar(float(trade["entry"]), float(trade["stop"]),
                      float(trade["target"]) if trade.get("target") else None,
                      ctx.get("as_of_open"), float(h), float(l))
    if res:
        outcome, price, reason, fill = res
        ex = {"outcome": outcome, "session": as_of, "price": price,
              "reason": reason, "fill": fill}
        if trade.get("born_session") and as_of < trade["born_session"]:
            ex["pre_confirmation"] = True
            ex["never_live"] = True
        return ex
    held = weekday_sessions(as_date(trade["entry_session"]), as_date(as_of))
    if time_stop_sessions and held >= time_stop_sessions:
        ex = {"outcome": "timeout", "session": as_of,
              "price": float(ctx.get("as_of_close") or trade["entry"]),
              "fill": "close",
              "reason": f"{time_stop_sessions}-session time stop - exit at "
                        "the close (weekday count; NSE holidays not "
                        "modelled)"}
        if trade.get("born_session") and as_of < trade["born_session"]:
            ex["pre_confirmation"] = True
            ex["never_live"] = True
        return ex
    return None


def exit_with_backfill(trade: dict, ctx: dict, client, rec: dict,
                       settings: RuleSettings) -> dict | None:
    """check_exit, plus a bar walk over any sessions a missed run skipped.

    GitHub's scheduler drops slots (BUG 55), and a stop that filled on a day
    nobody ran must not sit unnoticed until the next extreme. The walk costs
    one daily-history call, so it is only taken when a run was actually missed
    or the state has no closed bar to judge the trade against at all."""
    hist = None
    last = as_date(trade.get("last_checked") or trade.get("entry_session"))
    as_of = as_date(ctx.get("as_of"))
    missed = max(0, weekday_sessions(last, as_of) - 1) if last and as_of else 0
    if client is not None and (rec or {}).get("security_id") \
            and (missed > 0 or not ctx.get("as_of")):
        if missed:
            log.info("%s: %d session(s) skipped since %s - bar-walking",
                     trade.get("symbol"), missed, trade.get("last_checked"))
        hist = fetch_bars(client, rec, settings)
        if hist:
            # Walk every session from the entry through the state's own latest
            # closed bar - a fill on a SKIPPED session must be reported with
            # that session, and check_exit only ever judges the newest one.
            walk = walk_bars(trade, hist,
                             through=ctx.get("as_of")
                             or trade.get("last_checked"),
                             time_stop_sessions=settings.time_stop_sessions)
            if walk:
                return _attach_b(trade, walk, hist, settings)
            if not ctx.get("as_of"):
                b = hist[-1]
                ctx = dict(ctx, as_of=b.session, as_of_open=b.open,
                           as_of_high=b.high, as_of_low=b.low,
                           as_of_close=b.close)
    if not ctx.get("as_of"):
        return None
    ex = check_exit(trade, ctx, settings.time_stop_sessions)
    if ex is None or ex.get("fill") != "order" \
            or ctx.get("as_of_open") is not None or client is None \
            or not (rec or {}).get("security_id"):
        return _attach_b(trade, ex, hist, settings)
    # Something resolved AT an order price on the state's newest bar, and the
    # state carries no open for that bar. Whether the order filled at its own
    # price or was gapped through is exactly what the backtest's fill policy
    # is about, so spend one history call - only when a trade resolves.
    if hist is None:
        hist = fetch_bars(client, rec, settings)
    if not hist:
        return ex
    b = next((b for b in hist if b.session == ctx.get("as_of")), None)
    if b is None:
        return _attach_b(trade, ex, hist, settings)
    res = resolve_bar(float(trade["entry"]), float(trade["stop"]),
                      float(trade["target"]) if trade.get("target") else None,
                      b.open, b.high, b.low)
    if not res:
        return _attach_b(trade, ex, hist, settings)
    outcome, price, reason, fill = res
    return _attach_b(trade, dict(ex, outcome=outcome, session=b.session,
                                 price=price, reason=reason, fill=fill),
                     hist, settings)


# --------------------------------------------------------------------------- #
#  The plan
# --------------------------------------------------------------------------- #
def _rr_txt(target: float | None, entry: float | None, stop: float) -> str:
    if not target or not entry or entry <= stop:
        return "-"
    return f"{(target - entry) / (entry - stop):.1f}x"


@dataclass
class TradePlan:
    symbol: str
    entry_a: float                     # the OB candle's close - the rule
    entry_b: float | None              # the displacement close - fillable
    stop: float
    target: float | None
    breakout_session: str
    born_session: str
    origin_session: str
    capital: float
    risk_pct: float
    late_sessions: int = 0
    time_stop_sessions: int = 90

    @property
    def entry(self) -> float:
        return self.entry_a

    @property
    def risk_amt(self) -> float:
        return self.capital * self.risk_pct / 100.0

    @property
    def risk_gap(self) -> float:
        return self.entry_a - self.stop

    def risk_pct_of(self, px: float | None) -> float:
        if not px or self.risk_gap <= 0:
            return 0.0
        return (px - self.stop) / px * 100.0

    @property
    def risk_pct_price(self) -> float:
        return self.risk_pct_of(self.entry_a)

    @property
    def rew_pct(self) -> float | None:
        if not self.target or not self.entry_a:
            return None
        return (self.target / self.entry_a - 1.0) * 100.0

    @property
    def rr(self) -> float | None:
        if self.target and self.risk_gap > 0:
            return (self.target - self.entry_a) / self.risk_gap
        return None

    @property
    def risk_qty(self) -> int:
        return int(math.floor(self.risk_amt / self.risk_gap)) \
            if self.risk_gap > 0 else 0

    @property
    def cap_qty(self) -> int:
        return int(math.floor(self.capital / self.entry_a)) \
            if self.entry_a > 0 else 0

    @property
    def is_capped(self) -> bool:
        return self.risk_gap > 0 and self.risk_qty > self.cap_qty > 0

    @property
    def qty(self) -> int:
        if self.risk_gap <= 0:
            return 0
        rq = self.risk_qty
        cq = self.cap_qty
        return min(rq, cq) if cq > 0 else rq

    @property
    def value(self) -> float:
        return self.qty * self.entry_a


def plan_html(p: TradePlan, extra: str = "") -> str:
    """The complete plan, one Telegram block - the rule's numbers first."""
    tgt = (f"<b>{_fmt(p.target)}</b> ({p.rew_pct:+.1f}% from A)"
           if p.rew_pct is not None else "set from the chart")
    if p.qty <= 0:
        size = f"stop too far for a ₹{p.risk_amt:,.0f} risk budget"
    elif p.is_capped:
        size = (f"{p.qty:,} shares ≈ ₹{p.value:,.0f} "
                f"(capped by capital; risk-sized was {p.risk_qty:,} shares)")
    else:
        size = f"{p.qty:,} shares ≈ ₹{p.value:,.0f}"
    age_str = f"{p.late_sessions} session(s) old" if p.late_sessions \
        else "today's OB candle"
    lines = [
        f"🎯 <b>PRECISION OB ENTRY — {_esc(p.symbol)}</b>",
        f"OB candle <b>{_esc(p.origin_session)}</b> ({age_str}) · "
        f"born <b>{_esc(p.born_session)}</b> (displacement bar)",
        f"First precision OB of the 26W breakout cycle · breakout "
        f"{_esc(p.breakout_session)} · 26W level <b>{_fmt(p.stop)}</b>",
        "",
        "<b>THE RULE — exactly as backtested</b>",
        f"▶ Entry A: the OB candle's close <b>{_fmt(p.entry_a)}</b>"
        + (f" · {p.late_sessions} session(s) old" if p.late_sessions
           else " · today's OB candle close"),
        f"🎯 Target {tgt} — the highest high printed between the breakout "
        "and the OB candle",
        f"⛔ Stop <b>{_fmt(p.stop)}</b> (−{p.risk_pct_price:.1f}%) — the 26W "
        "breakout level",
        f"⏱ Time stop: {p.time_stop_sessions} sessions from "
        f"{_esc(p.origin_session)}",
        f"📊 Size: {size} · risk ₹{p.risk_amt:,.0f} ({p.risk_pct:.1f}% of "
        f"₹{p.capital:,.0f}) · R:R {_rr_txt(p.target, p.entry_a, p.stop)}",
    ]
    if p.entry_b:
        lines += [
            "",
            "<b>THE FILLABLE VERSION (entry B)</b>",
            f"▶ The displacement close <b>{_fmt(p.entry_b)}</b> — the first "
            "real-time price at which this OB was knowable",
            f"   from B: target {((p.target / p.entry_b - 1) * 100):+.1f}% · "
            f"stop −{p.risk_pct_of(p.entry_b):.1f}% · R:R "
            f"{_rr_txt(p.target, p.entry_b, p.stop)}",
        ]
    if extra:
        lines += ["", extra]
    lines += [
        "",
        f"<i>{STATS_A}</i>",
        f"<i>{STATS_B}</i>",
        "<i>Entry A uses one bar of hindsight — the OB only becomes knowable "
        "when the displacement bar after it closes, so A's close was already "
        "history by the time this event could be detected. Orders: sell limit "
        "at the target, stop order at the level. Verify on chart before "
        "executing.</i>",
    ]
    return "\n".join(lines)


def _net(entry: float | None, exit_px: float, cost_pct: float) -> float | None:
    if not entry:
        return None
    return (exit_px / entry - 1.0) * 100.0 - cost_pct


def b_note(trade: dict, ex: dict, cost_pct: float) -> str:
    """The Entry-B line of an exit message - exact, never approximated.

    B's number is quoted only when B's own walk produced it (`ex["b"]`, set by
    exit_with_backfill). When B has not resolved it is said to be still open;
    a trade whose born session is at/after A's exit session cannot have held a
    session yet, and that is knowable without any bars."""
    b = ex.get("b")
    if b is None and trade.get("born_session") \
            and str(trade["born_session"]) >= str(ex.get("session") or "~"):
        b = {"open": True}
    if b is None:
        return ""
    if b.get("outcome"):
        net = _net(trade.get("entry_b"), float(b["price"]), cost_pct)
        if net is None:
            return ""
        session = f" {_esc(b['session'])}" if str(b.get("session")) != str(ex.get("session")) else ""
        return f" · from B {net:+.2f}%{session}"
    return " · B (displacement close) still open"


def _attach_b(trade: dict, ex: dict | None, bars: list | None,
              settings: RuleSettings) -> dict | None:
    """Only used by exit_with_backfill: it already holds the bars, so B's own
    exit rides along with A's at no extra data cost."""
    if ex is None:
        return None
    b = b_walk(trade, ex, bars, settings)
    return dict(ex, b=b) if b else ex


def exit_html(trade: dict, ex: dict, settings: RuleSettings) -> str:
    sym = _esc(trade["symbol"])
    pre = bool(ex.get("pre_confirmation") or (ex.get("session") and trade.get("born_session")
               and ex["session"] < trade["born_session"]))
    if pre:
        icon, word = "📖", "BOOK RECORD"
    elif ex.get("outcome") == "win":
        icon, word = "🎯", "TARGET HIT"
    elif ex.get("outcome") == "timeout":
        icon, word = "⏱", "TIME STOP"
    else:
        icon, word = "🛑", "STOPPED OUT"
    cost = settings.round_trip_cost_pct
    net_a = _net(trade.get("entry_a"), float(ex["price"]), cost)
    held = ex.get("sessions")
    if held is None:
        held = weekday_sessions(as_date(trade.get("entry_session")),
                                as_date(ex["session"]))
    fill = "gapped through, filled at the open" if ex.get("fill") == "gap-open" \
        else "filled at the order"
    pre_note = " · no position was ever live" if pre else ""
    return (f"{icon} <b>{word} — {sym} (Precision OB Entry)</b>\n"
            f"Exit <b>{_fmt(ex['price'])}</b> on {_esc(ex['session'])} "
            f"({fill}) · net from A <b>{net_a:+.2f}%</b>"
            f"{b_note(trade, ex, cost)}\n"
            f"held ≈{held} session(s) · {_esc(ex['reason'])}{pre_note}")


def late_html(trade: dict, ex: dict, settings: RuleSettings) -> str:
    """A rule event alerted inside the catch-up window whose trade already
    resolved. No plan, no action - the honest version of "you missed it"."""
    cost = settings.round_trip_cost_pct
    net_a = _net(trade.get("entry_a"), float(ex["price"]), cost)
    late_s = trade.get('late_sessions', 0)
    age_str = f"{late_s} session(s) old" if late_s else "today's OB candle"
    pre = bool(ex.get("pre_confirmation") or (ex.get("session") and trade.get("born_session")
               and ex["session"] < trade["born_session"]))
    outcome_str = "BOOK RECORD" if pre else str(ex.get('outcome', '')).upper()
    never_live = " (no position was ever live)" if pre else ""
    trail_note = (".\nNo action — no position was ever live; recorded in the book for the audit trail."
                  if pre else
                  ".\nNo action — recorded in the book for the audit trail.")
    return (f"⏱ <b>MISSED — {_esc(trade['symbol'])} (Precision OB Entry)</b>\n"
            f"OB candle <b>{_esc(trade['origin_session'])}</b> ({age_str}) · "
            f"born <b>{_esc(trade['born_session'])}</b> (displacement bar) · "
            f"first post-breakout OB of the cycle.\n"
            f"The rule's own trade already resolved{never_live}: "
            f"<b>{outcome_str}</b> at {_fmt(ex['price'])} on "
            f"{_esc(ex['session'])} — net from A {net_a:+.2f}%"
            + b_note(trade, ex, cost)
            + trail_note)


# --------------------------------------------------------------------------- #
#  Data fetch (only for the symbols whose rule event is being evaluated)
# --------------------------------------------------------------------------- #
def fetch_bars(client, rec: dict, settings: RuleSettings,
               lookback_days: int | None = None) -> list | None:
    days = int(lookback_days or settings.lookback_days)
    from_d = datetime.now(IST).date() - timedelta(days=days)
    try:
        df = client.daily_candles(rec["security_id"],
                                  rec["exchange_segment"], from_d,
                                  datetime.now(IST).date(), chunk_days=365,
                                  symbol=rec.get("symbol"))
    except (DhanError, Exception) as exc:      # noqa: B014 - degrade, never die
        log.warning("daily fetch failed for %s: %s", rec.get("symbol"), exc)
        return None
    if df is None or getattr(df, "empty", True):
        return None
    try:
        return bars_from_frame(df)
    except Exception as exc:                   # noqa: BLE001
        log.warning("bars parse failed for %s: %s", rec.get("symbol"), exc)
        return None


# --------------------------------------------------------------------------- #
#  The trade book
# --------------------------------------------------------------------------- #
def new_trade(sym: str, rule: Rule, late: int) -> dict:
    """entry_a is the rule's entry and the one the book tracks; entry_b is
    carried alongside so every report can show the fillable version's net."""
    return {"id": f"{sym}-OBE-{rule.born_session}", "symbol": sym,
            "strategy": "precision_ob_entry",
            "entry": float(rule.entry_a), "entry_a": float(rule.entry_a),
            "entry_b": float(rule.entry_b) if rule.entry_b else None,
            "stop": float(rule.stop),
            "target": float(rule.target) if rule.target else None,
            "breakout_session": rule.breakout_session,
            "born_session": rule.born_session,
            "origin_session": rule.origin_session,
            "entry_session": rule.origin_session,
            "last_checked": rule.origin_session,
            "sessions_held": 0, "late_sessions": late}


# --------------------------------------------------------------------------- #
#  Modes
# --------------------------------------------------------------------------- #
def run_postclose(cfg, args, tg) -> int:
    # The session the rule is evaluated AGAINST: the last COMPLETED one, never
    # the phantom date `now().date()` becomes after midnight IST (session_date).
    today = args.today or session_date().isoformat()
    settings = rule_settings(cfg)
    params = cfg.ob_precision.params()
    st_in = load_json(STATE_IN)
    own = load_json(STATE_OWN)
    # Kept so a quiet run can tell whether the book actually moved, and skip
    # rewriting the file when it did not (see _book_changed).
    on_disk = json.loads(json.dumps(own))
    own.setdefault("sent", {})
    own.setdefault("open", [])
    own.setdefault("closed", [])
    own.setdefault("excluded", {})
    own.setdefault("cycles", {})

    client = None
    if not args.no_data:
        try:
            client = DhanClient(cfg.secrets.dhan_client_id,
                                cfg.secrets.dhan_access_token,
                                data_rate=cfg.runtime.data_rate_per_sec,
                                quote_rate=cfg.runtime.quote_rate_per_sec)
        except Exception as exc:                # noqa: BLE001
            log.warning("no market data client (%s) - rule events deferred", exc)
            client = None

    waiting = {s: r for s, r in (st_in.get("waiting") or {}).items()
               if r.get("status") in ("waiting", "tapped")}
    if args.symbols:
        keep = {s.strip().upper() for s in args.symbols.split(",")}
        waiting = {s: r for s, r in waiting.items() if s in keep}
    log.info("postclose %s: %d waiting-list symbol(s)", today, len(waiting))

    msgs: list[str] = []
    seen_history: list[str] = []
    events_total = 0
    events_handled = 0
    events_deferred = 0
    events_stale_ob = 0
    events_excluded = 0
    events_alerted = 0
    for sym, rec in sorted(waiting.items()):
        # One name's bad record must cost itself its check, never the others.
        try:
            ctx = (st_in.get("zones") or {}).get(sym) or {}
            brk = breakout_of(rec, ctx)
            if not brk:
                # The c02 cross could not be dated for this name. The rule is
                # defined per breakout cycle, so there is no event to evaluate.
                continue
            first = first_ob_after_breakout(rec, ctx)
            if not first:
                continue                        # no post-breakout OB yet
            events_total += 1
            born, z = first
            key = f"rule|{sym}|{born}"
            ckey = _cycle_key(sym, brk)
            if _sent(own, key) or _cycle_handled(own, ckey) \
                    or not settings.confirm_alerts:
                # Already handled. The cycle check is what catches a name whose
                # state-derived birth disagrees with the bars' own (ARFIN): the
                # key above would never match, and the symbol would otherwise
                # pay a full daily-history fetch on every run to discover that
                # its cycle was settled long ago.
                if not settings.confirm_alerts:
                    _mark(own, key)
                    _mark_cycle(own, ckey, key)
                events_handled += 1
                continue
            # Cheap gate before any market-data call: an OB candle this old
            # cannot be inside the window, whatever the port says about it.
            origin_hint = z.get("origin_session") or (z.get("detail") or {}).get("origin_session")
            gate_session = origin_hint or born
            if weekday_sessions(as_date(gate_session), as_date(today)) \
                    > settings.catchup_sessions + 1:
                _mark(own, key)
                _mark_cycle(own, ckey, key)
                seen_history.append(f"{sym}({gate_session})")
                events_stale_ob += 1
                continue
            if client is None:
                log.warning("%s: no market data - rule event %s deferred "
                            "(retried next run)", sym, born)
                events_deferred += 1
                continue                        # NOT marked: retry next run
            bars = fetch_bars(client, rec, settings)
            if not bars:
                log.warning("%s: no daily bars - rule event %s deferred "
                            "(retried next run)", sym, born)
                events_deferred += 1
                continue
            # The state said where to look; the port says what the event is.
            bar_births = births_from_bars(bars, brk, params)
            if bar_births:
                born, z = bar_births[0]
            key = f"rule|{sym}|{born}"
            if _sent(own, key):
                # Heal the cycle index: a key marked before this job tracked
                # cycles still settles the cycle, so record it and every later
                # run short-circuits before paying for the fetch above.
                _mark_cycle(own, ckey, key)
                events_handled += 1
                continue
            rule = evaluate_rule(bars, brk, born, params)
            if not rule.ok:
                _mark(own, key)
                _mark_cycle(own, ckey, key)
                own["excluded"][f"{sym}|{born}"] = rule.reason
                events_excluded += 1
                log.info("%s: rule event %s excluded - %s", sym, born,
                         rule.reason)
                continue

            # Window = the OB candle's age, not the confirmation bar's. Both
            # are measured against the last COMPLETED session (`today` from
            # session_date), never against a phantom date rolled over after
            # midnight IST - which used to add a session to every age here and
            # suppress events sitting on the boundary.
            late = weekday_sessions(as_date(born), as_date(today))
            ob_age = weekday_sessions(as_date(rule.origin_session), as_date(today))
            if ob_age < 0 or ob_age > settings.catchup_sessions or late < 0 or late > settings.catchup_sessions:
                _mark(own, key)                 # history: never replayed
                _mark_cycle(own, ckey, key)
                events_stale_ob += 1
                seen_history.append(f"{sym}({rule.origin_session}, {ob_age}s)")
                log.info("%s: OB candle %s is %d session(s) old as of %s "
                         "(> %d) — stale OB candle, suppressed",
                         sym, rule.origin_session, ob_age, today,
                         settings.catchup_sessions)
                continue

            cap, rpct = args.capital, args.risk_pct
            plan = TradePlan(sym, entry_a=float(rule.entry_a),
                             entry_b=float(rule.entry_b) if rule.entry_b else None,
                             stop=float(rule.stop),
                             target=float(rule.target) if rule.target else None,
                             breakout_session=rule.breakout_session,
                             born_session=rule.born_session,
                             origin_session=rule.origin_session,
                             capital=cap, risk_pct=rpct, late_sessions=late,
                             time_stop_sessions=settings.time_stop_sessions)
            trade = new_trade(sym, rule, late)
            # A late alert must never quote a live plan for a finished trade:
            # walk the bars the trade would have seen before saying anything.
            ex = walk_bars(trade, bars, through=today,
                           time_stop_sessions=settings.time_stop_sessions) \
                if late > 0 else None
            if ex:
                ex["reason"] += " (the alert ran late)"
                b = b_walk(trade, ex, bars, settings, through=today)
                if b:
                    ex["b"] = b
                trade["exit"] = ex
                own["closed"].append(trade)
                msgs.append(late_html(trade, ex, settings))
            else:
                detail = (f"Funnel: OB candle <b>{_esc(rule.origin_session)}"
                          f"</b> sits on/after the breakout · its close "
                          f"<b>{_fmt(rule.entry_a)}</b> is "
                          f"{(rule.entry_a / rule.stop - 1) * 100:+.2f}% "
                          f"above the 26W level · this OB is the FIRST born "
                          f"after the breakout")
                rvol = zone_rvol(z)
                if rvol:
                    detail += f" · displacement rvol {_fmt(rvol)}"
                msgs.append(plan_html(plan, detail))
                own["open"].append(trade)
            events_alerted += 1
            _mark(own, key)
            _mark_cycle(own, ckey, key)
        except Exception as exc:                # noqa: BLE001 - degrade
            log.warning("rule check failed for %s: %s", sym, exc)

    if seen_history:
        # One line, not three hundred: a fresh deploy walks the whole waiting
        # list once and every old birth is marked seen without an alert.
        log.info("%d rule event(s) outside the %d-session alert window marked "
                 "seen (first-run safety): %s%s", len(seen_history),
                 settings.catchup_sessions, ", ".join(seen_history[:8]),
                 " ..." if len(seen_history) > 8 else "")

    summary = (f"events: {events_total} total ({events_handled} already handled, "
               f"{events_deferred} deferred, {events_stale_ob} stale OB candle, "
               f"{events_excluded} excluded, {events_alerted} alerted)")
    log.info("postclose done: %d alert(s), %d open trade(s) — %s",
             len(msgs), len(own["open"]), summary)

    # ---- exits -------------------------------------------------------------
    still_open = []
    for trade in own["open"]:
        try:
            sym = trade["symbol"]
            ctx = (st_in.get("zones") or {}).get(sym) or {}
            rec = (st_in.get("waiting") or {}).get(sym) or {}
            ex = exit_with_backfill(trade, ctx, client, rec, settings)
            if ex:
                msgs.append(exit_html(trade, ex, settings))
                trade["exit"] = ex
                own["closed"].append(trade)
                _mark(own, f"exit|{trade['id']}")
            else:
                if ctx.get("as_of"):
                    trade["last_checked"] = ctx["as_of"]
                    trade["sessions_held"] = weekday_sessions(
                        as_date(trade["entry_session"]), as_date(ctx["as_of"]))
                still_open.append(trade)
        except Exception as exc:                # noqa: BLE001
            log.warning("exit check failed for %s: %s",
                        (trade or {}).get("id", "?"), exc)
            still_open.append(trade)
    own["open"] = still_open

    if not _send_alerts(tg, msgs):
        log.error("state not saved so undelivered alerts will retry; "
                  "already-delivered messages may repeat")
        return 1
    if _dry_run(cfg, args, tg):
        log.info("postclose dry-run complete: state not saved")
        return 0
    # Session-scoped on purpose. This used to be written, with a clock time, at
    # the TOP of the pass, so the file differed on every single run and the
    # workflow committed it every 15 minutes - 145 commits in a day and a half
    # for a job that sent two alerts. `last_run` now names the pass and the
    # session; the clock time lives in a field the quiet-run test ignores.
    own["last_run"] = f"postclose {today}"
    own["last_run_at"] = datetime.now(IST).isoformat(timespec="seconds")
    if not _book_changed(on_disk, own):
        log.info("postclose quiet: the book did not move - state not rewritten")
        return 0
    log.info("postclose done: %d alert(s), %d open trade(s)",
             len(msgs), len(own["open"]))
    save_state(own, STATE_OWN)
    return 0


def quote_is_stale(q: dict, ctx: dict) -> bool:
    """The scanner's own market-holiday test: a bulk quote carries no date, and
    on a weekday holiday the feed re-serves the previous session's numbers
    unchanged - which would read as a displacement that is not forming."""
    def _f(v, default=-1.0) -> float:
        try:
            return float(default if v is None else v)
        except (TypeError, ValueError):
            return float(default)

    return (_f(q.get("high"), 0.0) == _f(ctx.get("as_of_high"))
            and _f(q.get("low"), 0.0) == _f(ctx.get("as_of_low"))
            and _f(q.get("last_price"), 0.0) == _f(ctx.get("as_of_close")))


def live_displacement(q: dict, ctx: dict, params) -> dict | None:
    """The displacement test of precision.txt applied to the DEVELOPING bar,
    with the same config the live scanner runs and the same window the
    indicator uses (the last `vol_len` volumes including this bar, the 8-bar
    structure high before it). Returns the metrics when the bar, as it stands
    now, is a displacement - the condition under which the cycle's first
    post-breakout OB would be born at the close."""
    try:
        o, h, l, c = (float(q.get("open") or 0), float(q.get("high") or 0),
                      float(q.get("low") or 0),
                      float(q.get("last_price") or 0))
        v = float(q.get("volume") or 0)
    except (TypeError, ValueError):
        return None
    if not (o and h and l and c):
        return None
    rng = max(h - l, params.mintick)
    clv = (c - l) / rng
    body = abs(c - o) / rng
    atr = float(ctx.get("atr") or 0)
    pv = ctx.get("prev_volumes") or []
    window = pv[-(params.vol_len - 1):] + [v] if v else []
    rvol = (v / (sum(window) / len(window))) if window and v else 0.0
    highs = ctx.get("prev_highs") or []
    if not highs:
        return None
    structure = max(highs[-params.structure_len:])
    if not (c > o and rvol >= params.min_rvol
            and (not atr or rng >= atr * params.min_range_atr)
            and body >= params.min_body_frac and clv >= params.min_clv
            and c > structure):
        return None
    return {"rvol": rvol, "clv": clv, "body": body, "range": rng, "atr": atr,
            "structure": structure}


def forming_origin(bars: list, q: dict, today: str, params):
    """The OB candle a displacement on TODAY's developing bar would have: the
    scanner's own find_origin, run with the live bar appended. Returns
    (origin_bar, live_bar), either of which may be None."""
    live = Bar(open=float(q.get("open") or 0), high=float(q.get("high") or 0),
               low=float(q.get("low") or 0),
               close=float(q.get("last_price") or 0),
               volume=float(q.get("volume") or 0), time=today)
    all_bars = list(bars) + [live]
    off = find_origin(all_bars, len(bars), params)
    if not off or len(bars) - off < 0:
        return None, live
    return all_bars[len(bars) - off], live


def run_intraday(cfg, args, tg) -> int:
    # Here `today` is the LIVE, forming session - the wall-clock IST date is
    # correct, and deliberately NOT session_date(): this pass exists to catch
    # entry B before today's close, so it must name today. It is also
    # self-protecting after midnight, because the `refreshed_on != today` gate
    # below skips every symbol the scanner has not refreshed for a session that
    # has not started yet.
    today = args.today or datetime.now(IST).date().isoformat()
    settings = rule_settings(cfg)
    params = cfg.ob_precision.params()
    if not (settings.forming_alerts or settings.candidate_alerts):
        log.info("intraday: both forming and candidate alerts are off - "
                 "nothing to do")
        return 0
    st_in = load_json(STATE_IN)
    own = load_json(STATE_OWN)
    on_disk = json.loads(json.dumps(own))
    own.setdefault("sent", {})
    own.setdefault("cycles", {})
    try:
        client = DhanClient(cfg.secrets.dhan_client_id,
                            cfg.secrets.dhan_access_token,
                            data_rate=cfg.runtime.data_rate_per_sec,
                            quote_rate=cfg.runtime.quote_rate_per_sec)
    except Exception as exc:                    # noqa: BLE001
        log.warning("no market data client (%s) - intraday pass skipped", exc)
        return 0
    waiting = {s: r for s, r in (st_in.get("waiting") or {}).items()
               if r.get("status") == "waiting"}
    if args.symbols:
        keep = {s.strip().upper() for s in args.symbols.split(",")}
        waiting = {s: r for s, r in waiting.items() if s in keep}

    # One bulk quote for the whole waiting list - the scanner's own cost shape.
    by_seg: dict[str, list] = {}
    cands: list[str] = []
    for sym, rec in waiting.items():
        ctx = (st_in.get("zones") or {}).get(sym) or {}
        if not ctx.get("prev_highs"):
            continue
        cands.append(sym)
        sid = str(rec["security_id"])
        by_seg.setdefault(rec["exchange_segment"], []).append(
            int(sid) if sid.isdigit() else sid)
    quotes: dict[tuple[str, str], dict] = {}
    for seg, sids in by_seg.items():
        for i in range(0, len(sids), 100):
            try:
                part = client.ohlc({seg: sids[i:i + 100]}) or {}
            except (DhanError, Exception) as exc:   # noqa: B014
                log.warning("bulk quote failed (%s): %s", seg, str(exc)[:120])
                continue
            for sid, q in (part.get(seg) or {}).items():
                if q:
                    quotes[(seg, str(sid))] = q

    msgs: list[str] = []
    for sym in cands:
        try:
            rec, ctx = waiting[sym], (st_in.get("zones") or {}).get(sym) or {}
            q = quotes.get((rec["exchange_segment"], str(rec["security_id"])))
            if not q:
                continue
            if quote_is_stale(q, ctx):
                log.info("%s: quote identical to the last closed session - "
                         "stale (holiday/dead feed), skipping", sym)
                continue
            if ctx.get("refreshed_on") != today:
                log.info("%s: scanner context last refreshed %s, not today - "
                         "skipping", sym, ctx.get("refreshed_on") or "never")
                continue
            brk = breakout_of(rec, ctx)
            if not brk:
                continue
            # The cycle's event has already happened if ANY post-breakout OB
            # exists - from the state, and (for the forming path) from the
            # port's own replay of the bars it has to fetch anyway.
            if first_ob_after_breakout(rec, ctx):
                continue
            level = float(brk["level"])
            brk_s = str(brk["session"])[:10]
            price = float(q.get("last_price") or 0)
            disp = live_displacement(q, ctx, params)
            bars = None

            # ---- FORMING: today's bar is the displacement that would birth
            #      the cycle's FIRST post-breakout OB. This is the only moment
            #      entry B is still fillable, and the rule's structural rows
            #      are checked for real against the OB candle find_origin
            #      picks out of the CLOSED bars.
            if disp and settings.forming_alerts and price > level \
                    and not _sent(own, f"forming|{sym}|{today}"):
                bars = fetch_bars(client, rec, settings)
                origin = None
                if bars and births_from_bars(bars, brk, params):
                    log.info("%s: the cycle already has a post-breakout OB per "
                             "the scanner's own replay - no forming heads-up",
                             sym)
                    bars = None
                if bars:
                    origin, _live = forming_origin(bars, q, today, params)
                if origin is not None:
                    o_s = origin.session
                    if o_s < brk_s:
                        log.info("%s: the forming displacement's origin %s "
                                 "predates the breakout - the SMSPHARMA shape "
                                 "the rule excludes", sym, o_s)
                    elif origin.close <= level:
                        log.info("%s: the forming OB candle closed %.2f, below "
                                 "the 26W level %.2f - excluded", sym,
                                 origin.close, level)
                    else:
                        highs = [b.high for b in bars if brk_s <= b.session
                                 <= o_s]
                        target = max(highs) if highs else None
                        plan = TradePlan(
                            sym, entry_a=float(origin.close), entry_b=price,
                            stop=level, target=target,
                            breakout_session=brk_s, born_session=today,
                            origin_session=o_s, capital=args.capital,
                            risk_pct=args.risk_pct,
                            time_stop_sessions=settings.time_stop_sessions)
                        msgs.append(
                            f"⚡ <b>PRECISION OB FORMING — {_esc(sym)}</b>\n"
                            f"Today's bar is displacing: <b>{_fmt(price)}</b> "
                            f"above the 8-bar structure "
                            f"{_fmt(disp['structure'])} · rvol "
                            f"{disp['rvol']:.1f}x · clv {disp['clv']:.2f} · "
                            f"body {disp['body']:.2f}\n"
                            f"Traced with the scanner's own find_origin, the "
                            f"OB candle is <b>{_esc(o_s)}</b> (close "
                            f"{_fmt(origin.close)}, "
                            f"{(origin.close / level - 1) * 100:+.2f}% above "
                            f"the 26W level {_fmt(level)}) — a close like "
                            f"this BIRTHS the cycle's first post-breakout "
                            f"precision OB.\n"
                            f"▶ Fillable now: <b>today's close</b> = entry B, "
                            f"the displacement close\n"
                            f"▶ The rule's own entry A = the OB candle's "
                            f"close {_fmt(origin.close)} — one bar of "
                            f"hindsight, already history\n"
                            f"🎯 Target <b>{_fmt(target)}</b> · ⛔ stop "
                            f"<b>{_fmt(level)}</b> · ⏱ {settings.time_stop_sessions} "
                            f"sessions from {_esc(o_s)}\n"
                            f"<i>Final only if the bar closes like this; the "
                            f"complete plan follows after the close.</i>\n"
                            f"<i>{STATS_A}</i>\n<i>{STATS_B}</i>")
                        _mark(own, f"forming|{sym}|{today}")

            # ---- CANDIDATE (off by default): today's own candle is the OB
            #      candle - the only moment entry A could ever be filled, at
            #      the price of acting before a displacement exists. The
            #      backtest's funnel says post-breakout red/neutral candles
            #      above the level are common and the events are ~1.5/day
            #      universe-wide, so this is opt-in.
            elif (settings.candidate_alerts and price > level
                    and not _sent(own, f"candidate|{sym}|{today}")
                    and today >= brk_s):
                bars = bars or fetch_bars(client, rec, settings)
                o = float(q.get("open") or 0)
                h = float(q.get("high") or 0)
                l = float(q.get("low") or 0)
                rng = max(h - l, params.mintick)
                bearish = price < o
                neutral = (params.allow_neutral
                           and abs(price - o) / rng <= params.neutral_body)
                if bars and (bearish or neutral) and not disp:
                    highs = [b.high for b in bars
                             if brk_s <= b.session <= today]
                    target = max(highs + [h]) if highs else h
                    plan = TradePlan(
                        sym, entry_a=price, entry_b=None, stop=level,
                        target=target, breakout_session=brk_s,
                        born_session="pending a displacement",
                        origin_session=today, capital=args.capital,
                        risk_pct=args.risk_pct,
                        time_stop_sessions=settings.time_stop_sessions)
                    msgs.append(
                        f"🕯 <b>OB CANDIDATE — {_esc(sym)}</b>\n"
                        f"Today's candle is {'red' if bearish else 'neutral'} "
                        f"and <b>{(price / level - 1) * 100:+.2f}%</b> above "
                        f"the 26W level {_fmt(level)}.\n"
                        f"If the NEXT session displaces, today IS the OB "
                        f"candle of the cycle's first post-breakout precision "
                        f"OB, and the rule's entry A is <b>today's close "
                        f"{_fmt(plan.entry_a)}</b> — fillable only by acting "
                        f"now.\n"
                        f"🎯 Target <b>{_fmt(plan.target)}</b> · ⛔ stop "
                        f"<b>{_fmt(plan.stop)}</b> · ⏱ "
                        f"{plan.time_stop_sessions} sessions\n"
                        f"<i>Speculative: the order block does not exist "
                        f"until a displacement follows, and a close like this "
                        f"is common — the confirmed event is not.</i>\n"
                        f"<i>{STATS_A}</i>")
                    _mark(own, f"candidate|{sym}|{today}")
        except Exception as exc:                # noqa: BLE001 - degrade
            log.warning("intraday check failed for %s: %s", sym, exc)

    if not _send_alerts(tg, msgs):
        log.error("state not saved so undelivered alerts will retry")
        return 1
    if _dry_run(cfg, args, tg):
        log.info("intraday dry-run complete: state not saved")
        return 0
    own["last_run"] = f"intraday {today}"
    own["last_run_at"] = datetime.now(IST).isoformat(timespec="seconds")
    if not _book_changed(on_disk, own):
        log.info("intraday quiet: the book did not move - state not rewritten")
        return 0
    log.info("intraday done: %d alert(s)", len(msgs))
    save_state(own, STATE_OWN)
    return 0


def run_explain(cfg, args, tg) -> int:
    """One symbol at a time: print the rule's whole funnel state, so a name
    that did not alert can be explained instead of guessed at."""
    settings = rule_settings(cfg)
    params = cfg.ob_precision.params()
    st_in = load_json(STATE_IN)
    own = load_json(STATE_OWN)
    want = {s.strip().upper() for s in (args.symbols or "").split(",")
            if s.strip()}
    waiting = st_in.get("waiting") or {}
    zones = st_in.get("zones") or {}
    for sym in sorted(want or waiting):
        rec = waiting.get(sym)
        if not rec:
            print(f"{sym}: not on the waiting list")
            continue
        ctx = zones.get(sym) or {}
        brk = breakout_of(rec, ctx)
        print(f"\n{sym}: status {rec.get('status')} · "
              f"scanner refreshed {ctx.get('refreshed_on')}")
        if not brk:
            print("  c02 cross: UNKNOWN - the rule has no cycle boundary")
            continue
        print(f"  26W breakout: {brk['session']} · level "
              f"{float(brk['level']):.2f}")
        for key, sent_at in sorted(own.get("sent", {}).items()):
            if key.split("|")[1:2] == [sym]:
                print(f"  alert key {key}: sent {sent_at}")
        settled = (own.get("cycles") or {}).get(_cycle_key(sym, brk))
        if settled:
            # Without this line the cycle index is an invisible reason for a
            # name to be skipped - and it short-circuits BEFORE the bars are
            # fetched, so `explain` would otherwise print nothing about it.
            print(f"  cycle {str(brk['session'])[:10]} already settled by "
                  f"{settled} - the rule allows one event per cycle, so this "
                  f"name costs no data call")
        births = ob_births(rec, ctx)
        if not births:
            print("  post-breakout OBs: none yet - the rule is still waiting")
            continue
        born, z = births[0]
        print(f"  first post-breakout OB: born {born} "
              f"(origin recorded {z.get('origin_session') or '?'} · "
              f"rvol {z.get('rvol') or (z.get('detail') or {}).get('rvol')})")
        reason = (own.get("excluded") or {}).get(f"{sym}|{born}")
        if reason:
            print(f"  EXCLUDED: {reason}")
        if args.no_data:
            continue
        try:
            client = DhanClient(cfg.secrets.dhan_client_id,
                                cfg.secrets.dhan_access_token,
                                data_rate=cfg.runtime.data_rate_per_sec,
                                quote_rate=cfg.runtime.quote_rate_per_sec)
        except Exception as exc:                # noqa: BLE001
            print(f"  no data client: {exc}")
            continue
        bars = fetch_bars(client, rec, settings)
        if not bars:
            print("  no bars fetched")
            continue
        rule = evaluate_rule(bars, brk, born, params)
        if not rule.ok:
            print(f"  RULE: excluded - {rule.reason}")
            continue
        print(f"  RULE: taken · OB candle {rule.origin_session} close "
              f"{rule.entry_a:.2f} · displacement close {rule.entry_b} · "
              f"stop {rule.stop:.2f} · target {rule.target:.2f} · R:R "
              f"{_rr_txt(rule.target, rule.entry_a, rule.stop)}")
        res = walk_bars({"entry": rule.entry_a, "stop": rule.stop,
                         "target": rule.target,
                         "entry_session": rule.origin_session,
                         "born_session": rule.born_session},
                        bars, through=ctx.get("as_of"),
                        time_stop_sessions=settings.time_stop_sessions)
        print(f"  status: {res if res else 'open - neither order filled yet'}")
        if rule.entry_b:
            b_res = walk_bars({"entry": rule.entry_b, "stop": rule.stop,
                               "target": rule.target,
                               "entry_session": rule.born_session,
                               "born_session": rule.born_session},
                              bars, through=ctx.get("as_of"),
                              time_stop_sessions=settings.time_stop_sessions)
            print(f"  status B (entered at the {rule.born_session} close): "
                  f"{b_res if b_res else 'open'}")
    return 0


def run_digest(cfg, args, tg) -> int:
    own = load_json(STATE_OWN)
    print(f"precision OB entry - open trades: {len(own.get('open', []))}")
    for t in own.get("open", []):
        b = t.get("entry_b")
        b_txt = f"{b:.2f}" if b else "  -  "
        print(f"  {t['symbol']:16s} A {t['entry']:.2f} · B {b_txt} · stop "
              f"{t['stop']:.2f} · target {(t.get('target') or 0):>9.2f} · "
              f"entry session {t['entry_session']} "
              f"(~{t.get('sessions_held', 0)} ses)")
    print(f"closed: {len(own.get('closed', []))}")
    for t in own.get("closed", [])[-10:]:
        ex = t.get("exit") or {}
        print(f"  {t['symbol']:16s} {ex.get('outcome', '?'):7s} at "
              f"{(ex.get('price') or 0):.2f} on {ex.get('session', '?')}")
    excluded = own.get("excluded") or {}
    if excluded:
        print(f"excluded rule events: {len(excluded)}")
        for k, v in list(excluded.items())[-10:]:
            print(f"  {k}: {v}")
    return 0


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--mode",
                    choices=("postclose", "intraday", "explain", "digest"),
                    default="postclose")
    ap.add_argument("--today", default=None, help="override the session date")
    ap.add_argument("--symbols", default=None, help="comma-separated subset")
    ap.add_argument("--capital", type=float,
                    default=float(os.environ.get("STRATEGY_CAPITAL", "100000")),
                    help="capital for sizing (default STRATEGY_CAPITAL)")
    ap.add_argument("--risk-pct", type=float,
                    default=float(os.environ.get("STRATEGY_RISK_PCT", "1.0")),
                    help="risk %% per trade (default STRATEGY_RISK_PCT)")
    ap.add_argument("--no-data", action="store_true",
                    help="skip market-data fetches (rule events deferred)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print messages instead of sending")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    cfg = load_config()
    settings = rule_settings(cfg)
    if not settings.enabled and args.mode in ("postclose", "intraday"):
        log.info("precision_ob_entry.enabled is false - nothing to do")
        return 0
    # The config names the state file; the module constant is the default the
    # tests and a config-less run use.
    global STATE_OWN
    name = str(settings.state_file or "").strip()
    if name and name != STATE_OWN.name:
        STATE_OWN = Path(name) if os.path.isabs(name) else ROOT / name
    if args.dry_run:
        cfg.runtime.dry_run = True
    tg = build_telegram(cfg, dry_run=cfg.runtime.dry_run)
    if args.mode in ("postclose", "intraday") \
            and getattr(tg, "dry_run", False) and not cfg.runtime.dry_run:
        log.error("Telegram is not configured for live delivery; refusing to "
                  "run the alert pass. Set credentials or use --dry-run.")
        return 2

    if args.mode == "intraday":
        return run_intraday(cfg, args, tg)
    if args.mode == "explain":
        return run_explain(cfg, args, tg)
    if args.mode == "digest":
        return run_digest(cfg, args, tg)
    return run_postclose(cfg, args, tg)


if __name__ == "__main__":
    sys.exit(main())
