#!/usr/bin/env python3
"""
Daily watchlist - every stock QUALIFYING for the current week, sent to Telegram.

"Qualifying" is a precise thing here. The 26W breakout level and the weekly
momentum state are FROZEN when the week opens (that is what weekly_snapshot.csv
holds). So on any given day a symbol sits in exactly one bucket:

    FIRED      the 5m close already crossed the level and the alert went out
    ABOVE      trading above its level right now but no alert yet - the weekly
               gate is still blocking it (almost always c09, weekly volume)
    NEAR       within `--near` percent of its level; these are the names that
               could trigger tomorrow
    FAR        everything else - not reported, just counted

The point of the digest is the middle two. FIRED is history, FAR is noise;
ABOVE and NEAR are what to have on screen in the morning.

Cost: one bulk-quote pass (1000 instruments per request, ~3 requests for the
whole cash segment), plus a per-symbol 5m replay ONLY for the handful sitting
above their level. It is cheap enough to run daily without touching the
100k/day Dhan budget.

    python watchlist.py                 # send today's digest
    python watchlist.py --near 5        # widen the "approaching" band
    python watchlist.py --dry-run       # print, do not send
    python watchlist.py --csv wl.csv    # also write the full table

Never places orders.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import load_config
from dhan import IST, DhanClient, DhanError
from scan import QUOTE_BATCH, load_snapshots
from state import AlertState
from strategy import (COND_LABELS, WeeklySnapshot, replay_week, week_start_of)
from telegram import Telegram, _esc, _fmt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger("watchlist")

# How many names to print per section before truncating. Telegram splits at
# 4096 chars; a 2000-symbol universe can easily blow past that, and a wall of
# text nobody reads is worse than a short list plus a count. The COMPLETE table
# always goes out as a CSV attachment, so nothing is actually lost.
MAX_ABOVE = 25
MAX_NEAR = 15


def fetch_ltp(client: DhanClient, snaps: list[WeeklySnapshot]) -> dict[str, float]:
    """symbol -> last traded price, via bulk quotes."""
    by_seg: dict[str, list[int]] = {}
    ident: dict[tuple[str, str], str] = {}
    for s in snaps:
        by_seg.setdefault(s.exchange_segment, []).append(int(s.security_id))
        ident[(s.exchange_segment, str(s.security_id))] = s.symbol

    out: dict[str, float] = {}
    for seg, ids in by_seg.items():
        for i in range(0, len(ids), QUOTE_BATCH):
            chunk = ids[i:i + QUOTE_BATCH]
            try:
                part = client.ltp({seg: chunk})
            except DhanError as exc:
                msg = str(exc)
                if ("auth failed" in msg or "401" in msg or "403" in msg
                        or "DH-901" in msg or "Authentication" in msg):
                    raise DhanError(
                        "Dhan authentication failed - refresh DHAN_ACCESS_TOKEN"
                    ) from exc
                log.warning("quote batch failed: %s", msg[:140])
                continue
            for s, m in part.items():
                for sid, px in m.items():
                    sym = ident.get((s, str(sid)))
                    if sym and px > 0:
                        out[sym] = float(px)
    return out


def blocking_rows(client, cfg, snap: WeeklySnapshot, week: pd.Timestamp) -> str:
    """
    Replay this week's 5m candles for ONE symbol and report which conditions
    are still false on the latest bar. Only called for names already above
    their level, so the cost stays tiny.
    """
    from scan import scan_symbol                          # local: avoids a cycle
    try:
        _, res = scan_symbol(client, snap, cfg, week)
    except Exception as exc:                              # noqa: BLE001
        return f"(no intraday data: {str(exc)[:40]})"
    if res is None:
        return "(no intraday data)"
    ev = res.last_eval
    if ev is None:
        return "(no evaluation)"
    if res.signals:
        return "triggered"
    failed = [k for k in ("c03", "c04", "c05", "c06", "c07", "c08",
                          "c09", "c10", "c11", "c12", "c13")
              if not ev.conditions.get(k, True)]
    if not failed:
        return "gate open - awaiting a fresh cross"
    return ", ".join(COND_LABELS.get(k, k) for k in failed)


def build_message(week: str, rows: list[dict], counts: dict, near_pct: float) -> str:
    fired = [r for r in rows if r["bucket"] == "FIRED"]
    above = [r for r in rows if r["bucket"] == "ABOVE"]
    near = [r for r in rows if r["bucket"] == "NEAR"]

    lines = [
        f"📋 <b>Weekly Watchlist — week of {week}</b>",
        f"<i>{datetime.now(IST):%d-%b-%Y %H:%M} IST · "
        f"{counts['universe']} symbols in the snapshot</i>",
        "",
        f"🔥 fired <b>{len(fired)}</b> · "
        f"⚡ above level <b>{len(above)}</b> · "
        f"👀 within {near_pct:g}% <b>{len(near)}</b>",
    ]

    if above:
        lines += ["", f"⚡ <b>ABOVE LEVEL, NOT YET TRIGGERED ({len(above)})</b>",
                  "<i>price is through the 26W high; the weekly gate is holding "
                  "the entry back</i>"]
        for r in sorted(above, key=lambda x: -x["pct"])[:MAX_ABOVE]:
            lines.append(
                f"• <b>{_esc(r['symbol'])}</b> {_fmt(r['ltp'])} "
                f"vs <code>{_fmt(r['level'])}</code> "
                f"(+{r['pct']:.2f}%)"
                + (f"\n   <i>blocked: {_esc(r['why'])}</i>" if r.get("why") else ""))
        if len(above) > MAX_ABOVE:
            lines.append(f"<i>… and {len(above) - MAX_ABOVE} more — see the CSV</i>")

    if near:
        lines += ["", f"👀 <b>APPROACHING — within {near_pct:g}% ({len(near)})</b>",
                  "<i>these are tomorrow's candidates</i>"]
        for r in sorted(near, key=lambda x: x["gap"])[:MAX_NEAR]:
            lines.append(
                f"• <b>{_esc(r['symbol'])}</b> {_fmt(r['ltp'])} "
                f"→ <code>{_fmt(r['level'])}</code> "
                f"({r['gap']:.2f}% away)")
        if len(near) > MAX_NEAR:
            lines.append(f"<i>… and {len(near) - MAX_NEAR} more — see the CSV</i>")

    if fired:
        lines += ["", f"🔥 <b>ALREADY TRIGGERED THIS WEEK ({len(fired)})</b>"]
        names = ", ".join(_esc(r["symbol"]) for r in sorted(
            fired, key=lambda x: x["symbol"])[:60])
        lines.append(names + (f" … +{len(fired) - 60}" if len(fired) > 60 else ""))

    if not above and not near:
        lines += ["", "<i>Nothing above or approaching its level right now.</i>"]

    lines += ["", "<i>Levels are frozen for the whole week. Alerts fire on the "
                  "first 5m close above the 26W high.</i>"]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--near", type=float, default=3.0,
                    help="percent below the level to count as approaching")
    ap.add_argument("--explain", type=int, default=25,
                    help="max ABOVE names to diagnose with a 5m replay (0 = none)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    week = week_start_of(datetime.now(IST).date())
    week_s = str(week.date())

    snaps = load_snapshots(cfg, week_s)
    if not snaps:
        # load_snapshots already logged why. Exit 0: this runs on a schedule and
        # a missing snapshot is an operational problem, not a crash.
        log.error("no usable snapshot for %s - nothing to report", week_s)
        return 0

    if not cfg.secrets.dhan_access_token:
        log.error("DHAN_ACCESS_TOKEN not set")
        return 0

    client = DhanClient(cfg.secrets.dhan_client_id, cfg.secrets.dhan_access_token,
                        data_rate=cfg.runtime.data_rate_per_sec,
                        quote_rate=cfg.runtime.quote_rate_per_sec)

    log.info("pricing %d symbols ...", len(snaps))
    ltp = fetch_ltp(client, snaps)
    log.info("got %d quotes", len(ltp))

    state = AlertState(cfg.paths["state"])
    rows: list[dict] = []
    for s in snaps:
        px = ltp.get(s.symbol)
        if px is None or s.entry_level <= 0:
            continue
        pct = (px - s.entry_level) / s.entry_level * 100.0
        already = state.already_alerted(week_s, s.symbol)
        if already:
            bucket = "FIRED"
        elif pct > 0:
            bucket = "ABOVE"
        elif pct >= -args.near:
            bucket = "NEAR"
        else:
            bucket = "FAR"
        rows.append(dict(symbol=s.symbol, ltp=px, level=s.entry_level,
                         level_52=s.level_52, pct=pct, gap=-pct,
                         bucket=bucket, snap=s))

    # Diagnose only the ABOVE names - the ones where "why hasn't it fired?" is
    # the actual question.
    above = sorted([r for r in rows if r["bucket"] == "ABOVE"],
                   key=lambda x: -x["pct"])
    for r in above[:max(0, args.explain)]:
        try:
            r["why"] = blocking_rows(client, cfg, r["snap"], week)
        except Exception as exc:                          # noqa: BLE001
            log.debug("diagnose failed for %s: %s", r["symbol"], exc)

    counts = {"universe": len(snaps), "priced": len(ltp)}
    msg = build_message(week_s, rows, counts, args.near)

    # The full table always goes out as an attachment. The Telegram message is
    # capped so it stays readable, so without this the tail would be lost.
    table = pd.DataFrame(
        [{k: v for k, v in r.items() if k != "snap"} for r in rows]
    ).sort_values("pct", ascending=False)
    csv_path = Path(args.csv) if args.csv else Path(f"watchlist_{week_s}.csv")
    table.to_csv(csv_path, index=False)
    log.info("wrote %s (%d rows)", csv_path, len(table))

    if args.dry_run:
        print(msg)
        return 0

    tg = Telegram(cfg.secrets.telegram_bot_token, cfg.secrets.telegram_chat_id)
    tg.send(msg)
    # Only the actionable rows are worth attaching; FAR is the bulk of the file.
    actionable = table[table["bucket"].isin(("ABOVE", "NEAR", "FIRED"))]
    if not actionable.empty:
        att = csv_path.with_name(f"watchlist_{week_s}_{datetime.now(IST):%d%b}.csv")
        actionable.to_csv(att, index=False)
        tg.send_document(att, caption=f"Full watchlist — week of {week_s} "
                                      f"({len(actionable)} names)")
    log.info("watchlist sent (%d above, %d near)",
             len(above), sum(1 for r in rows if r["bucket"] == "NEAR"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
