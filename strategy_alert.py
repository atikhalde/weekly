#!/usr/bin/env python3
"""
The two-leg breakout-pullback strategy alert - complete trade plans, sent
to Telegram the moment a leg triggers, with exit tracking until every trade
resolves.

THE STRATEGY (STRATEGY_PLAN.md, built from the 2021-26 backtests)
-----------------------------------------------------------------
After a 26-week breakout, buy the pullback and sell into the swing high it
retreated from, stopped at the breakout level:

  LEG 1 - the OB-day entry. The first precision order block born after the
          26W breakout; entry at the displacement close. Backtest: 83% win,
          +1.32%/trade net (n=6,624, positive every year 2021-26).
  LEG 2 - the Tap 1 entry. The cycle's first tap of a precision OB, taken
          only when the tap closes ABOVE the 26W level. Backtest: 47% win,
          +2.14%/trade net, +1.09R (n=3,617).

  Exits (both legs): sell limit at the swing high since the breakout,
  stop order at the 26W breakout level, time stop 90 trading sessions.

DESIGN - deliberately a spectator of the live scanner
-----------------------------------------------------
This job READS ob_precision_state.json (the scanner's committed cache) and
never writes it. It keeps its own state in strategy_alert_state.json. No
existing file's behaviour changes; if this job dies, the scanner never
notices.

  --mode intraday   ~15:12 IST: ONE bulk quote for the waiting list, then
                    "forming" heads-ups - a displacement building on a name
                    with no post-breakout OB yet (Leg 1), a tap in progress
                    holding above the level (Leg 2). The entry window is
                    before the close; the plan is in the alert.
  --mode postclose  ~16:05 IST: confirmations from the closed-bar state the
                    scanner refreshed at the close - the complete trade plan
                    (entry, stop, target, size, time stop), entries recorded,
                    and every open trade checked for target/stop/timeout.
  --mode digest     no market data: print the open book.

One daily-history fetch happens ONLY for symbols that fire a signal (to
compute the swing-high target precisely); everything else reads the state.

First-run safety: alerts fire only for events dated TODAY (or later than the
last seen run) - deploying this job never replays history into the chat.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from config import load_config
from dhan import DhanClient, DhanError
from ob_precision import bars_from_frame
from telegram import _esc, _fmt, build_telegram

ROOT = Path(__file__).resolve().parent
STATE_IN = ROOT / "ob_precision_state.json"      # the scanner's - READ ONLY
STATE_OWN = ROOT / "strategy_alert_state.json"   # ours

IST = ZoneInfo("Asia/Kolkata")
TIME_STOP_SESSIONS = 90
STRUCTURE_LEN = 8
VOL_LEN = 20
MIN_RVOL = 1.8
MIN_RANGE_ATR = 1.2
MIN_BODY = 0.55
MIN_CLV = 0.72

LEG1_BACKTEST = "83% win · +1.32%/trade net · positive every year 2021-26 (n=6,624)"
LEG2_BACKTEST = "47% win · +2.14%/trade net · +1.09R (n=3,617) · avg win +13.3%, be patient"

log = logging.getLogger("strategy_alert")


# --------------------------------------------------------------------------- #
#  Small helpers
# --------------------------------------------------------------------------- #
def as_date(v) -> date | None:
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    s = str(v or "")[:10]
    try:
        return date.fromisoformat(s)
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


def weekday_sessions(d0: date, d1: date) -> int:
    """Mon-Fri session count between two dates (NSE holidays not modelled -
    at a 90-session time stop the drift is ~2 sessions; this is an alert,
    the plan's exits are the orders, not the counter)."""
    if d1 <= d0:
        return 0
    n, d = 0, d0
    while d < d1:
        d += timedelta(days=1)
        if d.weekday() < 5:
            n += 1
    return n


# --------------------------------------------------------------------------- #
#  The trade plan
# --------------------------------------------------------------------------- #
@dataclass
class TradePlan:
    symbol: str
    leg: int
    entry: float
    stop: float
    target: float | None
    entry_session: str
    breakout_session: str
    capital: float
    risk_pct: float

    @property
    def risk_amt(self) -> float:
        return self.capital * self.risk_pct / 100.0

    @property
    def risk_gap(self) -> float:
        return self.entry - self.stop

    @property
    def risk_pct_price(self) -> float:
        return (self.risk_gap / self.entry * 100.0) if self.entry > 0 else 0.0

    @property
    def rew_pct(self) -> float | None:
        if not self.target or self.target <= 0:
            return None
        return (self.target / self.entry - 1.0) * 100.0

    @property
    def rr(self) -> float | None:
        if self.target and self.risk_gap > 0:
            return (self.target - self.entry) / self.risk_gap
        return None

    @property
    def qty(self) -> int:
        return int(math.floor(self.risk_amt / self.risk_gap)) \
            if self.risk_gap > 0 else 0

    @property
    def value(self) -> float:
        return self.qty * self.entry


def plan_html(p: TradePlan, extra: str = "") -> str:
    """The complete trade plan, one Telegram block."""
    tgt = (f"<b>{_fmt(p.target)}</b> (+{p.rew_pct:.1f}%)"
           if p.rew_pct is not None else
           "<i>swing high since breakout - set from the chart</i>")
    rr_txt = "-"
    if p.rew_pct is not None and p.risk_pct_price > 0:
        rr_txt = f"{p.rew_pct / p.risk_pct_price:.1f}x"
    size = (f"{p.qty:,} shares ≈ ₹{p.value:,.0f}"
            if p.qty > 0 else
            f"stop too far for a ₹{p.risk_amt:,.0f} risk budget")
    lines = [
        f"🟦 <b>STRATEGY LEG {p.leg} — {_esc(p.symbol)}</b>"
        if p.leg == 1 else f"🟧 <b>STRATEGY LEG {p.leg} — {_esc(p.symbol)}</b>",
        f"Breakout {_esc(p.breakout_session)} · entry session "
        f"<b>{_esc(p.entry_session)}</b>",
        "",
        "<b>TRADE PLAN</b>",
        f"▶ Entry <b>{_fmt(p.entry)}</b>",
        f"🎯 Target {tgt} — the swing high since the breakout",
        f"⛔ Stop <b>{_fmt(p.stop)}</b> (−{p.risk_pct_price:.1f}%) — "
        "the 26W breakout level",
        f"⏱ Time stop: 90 trading sessions from entry",
        f"📊 Size: {size} · risk ₹{p.risk_amt:,.0f} "
        f"({p.risk_pct:.1f}% of ₹{p.capital:,.0f}) · R:R {rr_txt}",
    ]
    if extra:
        lines.append("")
        lines.append(extra)
    lines.append("")
    lines.append(f"<i>Backtest: {LEG1_BACKTEST if p.leg == 1 else LEG2_BACKTEST}"
                 ". Orders: sell limit at the target, stop order at the level. "
                 "Verify on chart before executing.</i>")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
#  Detection (pure: state in, events out)
# --------------------------------------------------------------------------- #
def breakout_of(rec: dict, ctx: dict) -> dict | None:
    """{session, level, close} - the c02-dated 26W breakout of this cycle."""
    brk = rec.get("breakout_26w_session") or ctx.get("breakout_26w_session")
    if isinstance(brk, dict) and brk.get("session") and brk.get("level"):
        return brk
    return None


def zone_births(rec: dict, ctx: dict) -> list[tuple[str, dict]]:
    """All post-breakout zone births we can see: (born_session, zone-ish dict)
    from BOTH the live zone list and the pruned closed events."""
    brk = breakout_of(rec, ctx)
    if not brk:
        return []
    b_day = as_date(brk["session"])
    seen: dict[str, dict] = {}
    # closed events carry the full event (price, rvol) - prefer them
    for ev in ctx.get("closed_events") or []:
        if ev.get("kind") != "ob":
            continue
        born = ev.get("born_session")
        if born and as_date(born) and as_date(born) >= b_day:
            seen.setdefault(born, ev)
    for z in ctx.get("zones") or []:
        born = z.get("born_session")
        if born and as_date(born) and as_date(born) >= b_day:
            seen.setdefault(born, z)
    return sorted(seen.items(), key=lambda kv: kv[0])


def first_tap_after_breakout(rec: dict, ctx: dict) -> tuple[str, dict] | None:
    """(session, tap event) of the CYCLE's first tap - any zone, matching the
    Tap 1 backtest. Closed events carry the full event (price, tap number);
    the live zones' tap_session/taps fields only backfill when the event was
    pruned, and a zone's tap_session is its LAST tap - best-effort there."""
    brk = breakout_of(rec, ctx)
    if not brk:
        return None
    b_day = as_date(brk["session"])
    best: tuple[str, dict] | None = None

    def _earlier(ts: str, other: str | None) -> bool:
        return other is None or as_date(ts) < as_date(other)

    # real events first - they carry the tap close (ev["price"])
    for ev in ctx.get("closed_events") or []:
        if ev.get("kind") != "tap":
            continue
        ts = ev.get("session") or ev.get("bar_session")
        if ts and as_date(ts) and as_date(ts) >= b_day:
            if best is None or ts <= best[0]:
                best = (ts, ev)
    # zone-level taps only fill gaps the pruned events left behind
    for z in ctx.get("zones") or []:
        ts = z.get("tap_session")
        if not (ts and z.get("taps", 0) >= 1 and as_date(ts)
                and as_date(ts) >= b_day):
            continue
        if _earlier(ts, best[0] if best else None):
            best = (ts, {"kind": "tap", "session": ts, "tap_number": 1,
                         "zone": z, **{k: z.get(k) for k in
                                       ("top", "bottom", "entry", "stop",
                                        "born_session")}})
    return best


def leg1_today(st: dict, sym: str, today: str) -> dict | None:
    """LEG 1 fires when the cycle's FIRST post-breakout OB is born today.
    Returns the birth info (zone/event dict) or None."""
    rec = st.get("waiting", {}).get(sym) or {}
    ctx = (st.get("zones") or {}).get(sym) or {}
    births = zone_births(rec, ctx)
    if not births:
        return None
    born, z = births[0]
    if born != today:
        return None                     # first OB already happened (or not today)
    brk = breakout_of(rec, ctx)
    return {"symbol": sym, "born": born, "zone": z, "brk": brk, "ctx": ctx}


def leg2_today(st: dict, sym: str, today: str) -> dict | None:
    """LEG 2 fires when the CYCLE's first tap is today, tap number 1, and the
    close is above the 26W level (the level filter - below it, no trade)."""
    rec = st.get("waiting", {}).get(sym) or {}
    ctx = (st.get("zones") or {}).get(sym) or {}
    first = first_tap_after_breakout(rec, ctx)
    if not first:
        return None
    ts, ev = first
    if ts != today:
        return None
    if int(ev.get("tap_number") or 1) != 1:
        return None
    brk = breakout_of(rec, ctx)
    if not brk:
        return None
    close = None
    if ctx.get("as_of") == today:
        close = ctx.get("as_of_close")
    close = close if close else ev.get("price")
    if close is None or float(close) <= float(brk["level"]):
        return None                     # closed back below the level: no trade
    return {"symbol": sym, "tap": ev, "brk": brk, "ctx": ctx,
            "entry": float(close)}


# --------------------------------------------------------------------------- #
#  Exits
# --------------------------------------------------------------------------- #
def check_exit(trade: dict, ctx: dict) -> dict | None:
    """Check one open trade against the scanner state's latest closed bar.
    Returns an exit dict when the trade resolved, else None. Both orders
    touched in one session resolve as the conservative LOSS (backtest
    policy); the gap cases are reported with the actual fill."""
    as_of = ctx.get("as_of")
    if not as_of or as_of <= trade.get("last_checked", ""):
        return None
    hi, lo = ctx.get("as_of_high"), ctx.get("as_of_low")
    if hi is None or lo is None:
        return None
    entry, stop, target = (float(trade["entry"]), float(trade["stop"]),
                           float(trade["target"]) if trade.get("target") else None)
    hit_t = target is not None and float(hi) >= target
    hit_s = float(lo) <= stop
    if hit_s:
        return {"outcome": "loss", "session": as_of,
                "price": stop, "reason": "26W level stop filled"}
    if hit_t:
        return {"outcome": "win", "session": as_of,
                "price": target, "reason": "swing-high target filled"}
    # no order filled: time stop?
    sessions = weekday_sessions(as_date(trade["entry_session"]),
                                as_date(as_of))
    if sessions >= TIME_STOP_SESSIONS:
        return {"outcome": "timeout", "session": as_of,
                "price": float(ctx.get("as_of_close") or entry),
                "reason": "90-session time stop - exit at the close"}
    return None


def exit_with_backfill(trade: dict, ctx: dict, client, rec: dict) -> dict | None:
    """check_exit, plus a bar-walk over any sessions a missed run skipped.
    A missed day is not hypothetical (the repo's BUG 55: GitHub cron skips
    slots), and a stop that filled on the missed day must not sit unnoticed
    until the next extreme. Also backfills a target that could not be
    computed at entry (no client / failed fetch then)."""
    as_of = ctx.get("as_of")
    if not as_of or as_of <= trade.get("last_checked", ""):
        return None
    stop = float(trade["stop"])
    target = trade.get("target")
    if client is not None:
        bars = fetch_bars(client, rec)
        if bars is not None:
            if not target and trade.get("breakout_session"):
                target = swing_target(bars, trade["breakout_session"],
                                      trade["entry_session"])
                if target:
                    trade["target"] = target
                    log.info("%s: target backfilled to %.2f",
                             trade["symbol"], target)
            if target:
                start = as_date(trade.get("last_checked")
                                or trade["entry_session"])
                for b in bars:
                    d = as_date(b.session)
                    if d is None or d <= start or d > as_date(as_of):
                        continue
                    if b.low <= stop:
                        return {"outcome": "loss", "session": b.session,
                                "price": stop,
                                "reason": "26W level stop filled"}
                    if b.high >= float(target):
                        return {"outcome": "win", "session": b.session,
                                "price": float(target),
                                "reason": "swing-high target filled"}
    return check_exit(trade, ctx)


def exit_html(trade: dict, ex: dict) -> str:
    sym = _esc(trade["symbol"])
    leg = trade.get("leg", "?")
    pnl = ((float(ex["price"]) / float(trade["entry"]) - 1.0) * 100.0
           - 0.22)
    icon = {"win": "🎯", "loss": "🛑", "timeout": "⏱"}[ex["outcome"]]
    word = {"win": "TARGET HIT", "loss": "STOPPED OUT",
            "timeout": "TIME STOP"}[ex["outcome"]]
    held = weekday_sessions(as_date(trade["entry_session"]),
                            as_date(ex["session"]))
    return (f"{icon} <b>{word} — {sym} (Leg {leg})</b>\n"
            f"Exit <b>{_fmt(ex['price'])}</b> on {_esc(ex['session'])} · "
            f"net <b>{pnl:+.2f}%</b> · held ≈{held} sessions\n"
            f"{_esc(ex['reason'])}")


# --------------------------------------------------------------------------- #
#  Data fetch (only for symbols that fire)
# --------------------------------------------------------------------------- #
def fetch_bars(client, rec: dict, lookback_days: int = 400):
    from_d = datetime.now(IST).date() - timedelta(days=lookback_days)
    try:
        df = client.daily_candles(rec["security_id"],
                                  rec["exchange_segment"], from_d,
                                  datetime.now(IST).date(), chunk_days=365,
                                  symbol=rec.get("symbol"))
    except (DhanError, Exception) as exc:      # noqa: B014 - degrade, never die
        log.warning("daily fetch failed for %s: %s", rec.get("symbol"), exc)
        return None
    if df is None or df.empty:
        return None
    try:
        return bars_from_frame(df)
    except Exception as exc:
        log.warning("bars parse failed for %s: %s", rec.get("symbol"), exc)
        return None


def swing_target(bars: list, breakout_session: str, through_session: str):
    """The highest high printed between the breakout session and the entry
    session, inclusive - the strategy's sell limit."""
    lo = as_date(breakout_session)
    hi = as_date(through_session)
    if lo is None or hi is None or hi < lo:
        return None
    highs = [b.high for b in bars
             if lo <= as_date(b.session) <= hi]
    return max(highs) if highs else None


# --------------------------------------------------------------------------- #
#  Modes
# --------------------------------------------------------------------------- #
def _sent(st: dict, key: str) -> bool:
    return key in st.setdefault("sent", {})


def _mark(st: dict, key: str) -> None:
    st["sent"][key] = datetime.now(IST).isoformat(timespec="seconds")


def run_postclose(cfg, args, tg) -> int:
    today = args.today or datetime.now(IST).date().isoformat()
    st_in = load_json(STATE_IN)
    own = load_json(STATE_OWN)
    own.setdefault("sent", {})
    own.setdefault("open", [])
    own.setdefault("closed", [])
    own["last_run"] = f"postclose {datetime.now(IST).isoformat(timespec='seconds')}"
    client = None
    if not args.no_data:
        try:
            client = DhanClient(cfg.secrets.dhan_client_id,
                                cfg.secrets.dhan_access_token,
                                data_rate=cfg.runtime.data_rate_per_sec,
                                quote_rate=cfg.runtime.quote_rate_per_sec)
        except Exception as exc:                    # degrade, never die
            log.warning("no market data client (%s) - targets degrade", exc)
            client = None

    waiting = {s: r for s, r in (st_in.get("waiting") or {}).items()
               if r.get("status") in ("waiting", "tapped")}
    if args.symbols:
        keep = {s.strip().upper() for s in args.symbols.split(",")}
        waiting = {s: r for s, r in waiting.items() if s in keep}
    log.info("postclose %s: %d waiting-list symbol(s)", today, len(waiting))

    msgs: list[str] = []

    # ---- new entries --------------------------------------------------------
    for sym, rec in sorted(waiting.items()):
        cap, rpct = args.capital, args.risk_pct

        ev1 = leg1_today(st_in, sym, today)
        if ev1 and not _sent(own, f"leg1|{sym}|{ev1['born']}"):
            brk = ev1["brk"]
            z = ev1["zone"]
            entry = z.get("price") or z.get("as_of_close")
            if entry is None and ev1["ctx"].get("as_of") == today:
                entry = ev1["ctx"].get("as_of_close")
            if entry is None:
                entry = (st_in["zones"].get(sym) or {}).get("as_of_close")
            level = float(brk["level"])
            if entry is not None and float(entry) > level:
                entry = float(entry)
                target = None
                if client is not None:
                    bars = fetch_bars(client, rec)
                    if bars:
                        target = swing_target(bars, brk["session"], today)
                plan = TradePlan(sym, 1, entry, level, target, today,
                                 brk["session"], cap, rpct)
                origin = (z.get("detail") or {}).get("origin_session") \
                    or z.get("origin_session") or ""
                msgs.append(plan_html(
                    plan,
                    f"First precision OB after the breakout: zone "
                    f"<b>{_fmt(z.get('top'))}</b>–<b>{_fmt(z.get('bottom'))}"
                    f"</b> (origin {_esc(origin)}) · displacement rvol "
                    f"{_fmt(z.get('rvol') if z.get('rvol') else (z.get('detail') or {}).get('rvol'))}"
                    f" · buy at/after today's close"))
                _mark(own, f"leg1|{sym}|{ev1['born']}")
                own["open"].append({
                    "id": f"{sym}-L1-{ev1['born']}", "symbol": sym, "leg": 1,
                    "entry": entry, "stop": level, "target": target,
                    "entry_session": today, "last_checked": today,
                    "sessions_held": 0})
            else:
                _mark(own, f"leg1|{sym}|{ev1['born']}")   # below level: spent

        ev2 = leg2_today(st_in, sym, today)
        if ev2 and not _sent(own, f"leg2|{sym}|{today}"):
            brk, ev = ev2["brk"], ev2["tap"]
            level, entry = float(brk["level"]), ev2["entry"]
            target = None
            if client is not None:
                bars = fetch_bars(client, rec)
                if bars:
                    target = swing_target(bars, brk["session"], today)
            plan = TradePlan(sym, 2, entry, level, target, today,
                             brk["session"], cap, rpct)
            tapped = (ev.get("detail") or {}).get("tapped_entry") \
                or ev.get("entry")
            msgs.append(plan_html(
                plan,
                f"Tap 1 of the cycle on the zone born "
                f"{_esc(ev.get('born_session', ''))} · tapped entry "
                f"<b>{_fmt(tapped)}</b> · close held ABOVE the 26W level "
                f"(+{(entry / level - 1.0) * 100:.2f}%)"))
            _mark(own, f"leg2|{sym}|{today}")
            own["open"].append({
                "id": f"{sym}-L2-{today}", "symbol": sym, "leg": 2,
                "entry": entry, "stop": level, "target": target,
                "breakout_session": brk["session"],
                "entry_session": today, "last_checked": today,
                "sessions_held": 0})

    # ---- exits ---------------------------------------------------------------
    still_open = []
    for trade in own["open"]:
        ctx = (st_in.get("zones") or {}).get(trade["symbol"]) or {}
        ex = check_exit(trade, ctx) if ctx else None
        if ex:
            msgs.append(exit_html(trade, ex))
            trade["exit"] = ex
            own["closed"].append(trade)
            _mark(own, f"exit|{trade['id']}")
        else:
            if ctx.get("as_of"):
                trade["last_checked"] = ctx["as_of"]
                trade["sessions_held"] = weekday_sessions(
                    as_date(trade["entry_session"]), as_date(ctx["as_of"]))
            still_open.append(trade)
    own["open"] = still_open

    for m in msgs:
        tg.send(m)
    log.info("postclose done: %d alert(s), %d open trade(s)",
             len(msgs), len(own["open"]))
    save_state(own, STATE_OWN)
    return 0


def run_intraday(cfg, args, tg) -> int:
    today = args.today or datetime.now(IST).date().isoformat()
    st_in = load_json(STATE_IN)
    own = load_json(STATE_OWN)
    own.setdefault("sent", {})
    try:
        client = DhanClient(cfg.secrets.dhan_client_id,
                            cfg.secrets.dhan_access_token,
                            data_rate=cfg.runtime.data_rate_per_sec,
                            quote_rate=cfg.runtime.quote_rate_per_sec)
    except Exception as exc:
        log.warning("no market data client (%s) - intraday pass skipped", exc)
        return 0
    waiting = {s: r for s, r in (st_in.get("waiting") or {}).items()
               if r.get("status") == "waiting"}
    if args.symbols:
        keep = {s.strip().upper() for s in args.symbols.split(",")}
        waiting = {s: r for s, r in waiting.items() if s in keep}

    # one bulk quote for every waiting name with a live zone or a pending
    # Leg-1 (no zone yet) - the scanner's own cost shape, one request
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
            except (DhanError, Exception) as exc:      # noqa: B014
                log.warning("bulk quote failed (%s): %s", seg, str(exc)[:120])
                continue
            for sid, q in (part.get(seg) or {}).items():
                if q:
                    quotes[(seg, str(sid))] = q

    msgs: list[str] = []
    for sym in cands:
        rec, ctx = waiting[sym], (st_in.get("zones") or {}).get(sym) or {}
        q = quotes.get((rec["exchange_segment"], str(rec["security_id"])))
        if not q:
            continue
        brk = breakout_of(rec, ctx)
        if not brk:
            continue
        level = float(brk["level"])
        o, h, l, c, v = (float(q.get("open") or 0), float(q.get("high") or 0),
                         float(q.get("low") or 0),
                         float(q.get("last_price") or 0),
                         float(q.get("volume") or 0))
        if not (o and h and l and c):
            continue
        births = zone_births(rec, ctx)

        # ---- LEG 1 FORMING: a displacement building on a name whose first
        #      post-breakout OB has not been born yet
        if not births and h > max(ctx["prev_highs"][-STRUCTURE_LEN:]):
            rng = max(h - l, 1e-9)
            clv = (c - l) / rng
            body = abs(c - o) / rng
            atr = float(ctx.get("atr") or 0)
            pv = ctx.get("prev_volumes") or []
            window = pv[-(VOL_LEN - 1):] + [v] if v else []
            rvol = (v / (sum(window) / len(window))) if window and v else 0.0
            if (c > o and clv >= MIN_CLV and body >= MIN_BODY
                    and rvol >= MIN_RVOL and (not atr or rng >= atr * MIN_RANGE_ATR)
                    and c > level):
                if not _sent(own, f"leg1f|{sym}|{today}"):
                    gap_pct = (c / level - 1.0) * 100.0
                    msgs.append(
                        f"⚡ <b>LEG 1 FORMING — {_esc(sym)}</b>\n"
                        f"Displacement building: price <b>{_fmt(c)}</b> above "
                        f"the 8-bar structure <b>{_fmt(max(ctx['prev_highs'][-STRUCTURE_LEN:]))}</b>"
                        f" · running rvol <b>{rvol:.1f}x</b> · holding near "
                        f"the high (clv {clv:.2f})\n"
                        f"Entry window: <b>NOW → close</b> "
                        f"(the backtest entry is the displacement close)\n"
                        f"⛔ Stop: 26W level <b>{_fmt(level)}</b> "
                        f"(−{(c / level - 1.0) * 100:.1f}% below)\n"
                        f"🎯 Target: the swing high since the breakout - "
                        f"full plan follows at the close\n"
                        f"<i>The OB confirms only at today's close. "
                        f"{LEG1_BACKTEST}</i>")
                    _mark(own, f"leg1f|{sym}|{today}")

        # ---- LEG 2 FORMING: a tap in progress, holding above the level
        for z in ctx.get("zones") or []:
            if z.get("taps", 0) or not z.get("departed"):
                continue
            zentry = float(z.get("entry") or 0)
            if not zentry or l > zentry or c <= level:
                continue
            born = z.get("born_session")
            if not born or (as_date(brk["session"])
                            and as_date(born) < as_date(brk["session"])):
                continue
            if _sent(own, f"leg2f|{sym}|{today}"):
                continue
            msgs.append(
                f"⚡ <b>LEG 2 FORMING — {_esc(sym)}</b>\n"
                f"Tap 1 in progress: session low <b>{_fmt(l)}</b> touched the "
                f"pre-order entry <b>{_fmt(zentry)}</b> (zone born "
                f"{_esc(born)})\n"
                f"Price <b>{_fmt(c)}</b> is holding ABOVE the 26W level "
                f"<b>{_fmt(level)}</b>\n"
                f"Entry: <b>buy at/into the close</b> if it holds above the "
                f"level · ⛔ stop <b>{_fmt(level)}</b>\n"
                f"🎯 Target: the swing high since the breakout - full plan "
                f"follows at the close\n"
                f"<i>{LEG2_BACKTEST}</i>")
            _mark(own, f"leg2f|{sym}|{today}")
            break

    for m in msgs:
        tg.send(m)
    log.info("intraday done: %d forming alert(s)", len(msgs))
    own["last_run"] = f"intraday {datetime.now(IST).isoformat(timespec='seconds')}"
    save_state(own, STATE_OWN)
    return 0


def run_digest(cfg, args, tg) -> int:
    own = load_json(STATE_OWN)
    print(f"open trades: {len(own.get('open', []))}")
    for t in own.get("open", []):
        print(f"  {t['symbol']:16s} Leg {t.get('leg')} entry {t['entry']:.2f} "
              f"stop {t['stop']:.2f} target "
              f"{t['target'] if t.get('target') else '-':>9} "
              f"since {t['entry_session']} (~{t.get('sessions_held', 0)} ses)")
    print(f"closed: {len(own.get('closed', []))}")
    for t in own.get("closed", [])[-10:]:
        ex = t.get("exit") or {}
        print(f"  {t['symbol']:16s} Leg {t.get('leg')} {ex.get('outcome', '?'):7s} "
              f"at {ex.get('price', 0):.2f} on {ex.get('session', '?')}")
    return 0


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--mode", choices=("postclose", "intraday", "digest"),
                    default="postclose")
    ap.add_argument("--today", default=None,
                    help="override the session date (testing)")
    ap.add_argument("--symbols", default=None,
                    help="comma-separated subset")
    ap.add_argument("--capital", type=float,
                    default=float(os.environ.get("STRATEGY_CAPITAL", "100000")),
                    help="capital for sizing (default STRATEGY_CAPITAL)")
    ap.add_argument("--risk-pct", type=float,
                    default=float(os.environ.get("STRATEGY_RISK_PCT", "1.0")),
                    help="risk %% per trade (default STRATEGY_RISK_PCT)")
    ap.add_argument("--no-data", action="store_true",
                    help="skip market-data fetches (targets degrade)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print messages instead of sending")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    cfg = load_config()
    if args.dry_run:
        cfg.runtime.dry_run = True
    tg = build_telegram(cfg, dry_run=cfg.runtime.dry_run)

    if args.mode == "intraday":
        return run_intraday(cfg, args, tg)
    if args.mode == "digest":
        return run_digest(cfg, args, tg)
    return run_postclose(cfg, args, tg)


if __name__ == "__main__":
    sys.exit(main())
