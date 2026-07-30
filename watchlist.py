#!/usr/bin/env python3
"""
Daily watchlist - the stocks worth watching for the CURRENT week.

The 26W breakout level and the weekly momentum state are FROZEN when the week
opens (that is what weekly_snapshot.csv holds), so a pre-market digest can be
built from one bulk-quote pass.

WHAT IT REPORTS

    APPROACHING   ranked shortlist: closest to its 26W level, among names that
                  are structurally eligible to fire this week
    FIRED         names that already triggered (one line, for context)

Everything else is counted, not listed. The full table always goes out as a CSV.

WHY A RANKED SHORTLIST AND NOT A FILTER
    Measured on the live 29-Jul book (2,099 symbols, 88 names within 3%):
        no filter          88
        MACD hist > 0      88   <- no effect at all
        RSI > 50           83
        EMA50 rising       81
        fresh breakout     71
        all three          64
    Momentum filters barely cut anything, because a stock sitting near a
    26-week high almost always HAS good weekly momentum - that is why it is
    there. Distance to the level is the only thing that separates "could break
    tomorrow" from "needs a 3% move first", so the list is RANKED by distance
    and capped. A name 2.8% away is not actionable in the morning.

WHY NOT THE FULL 13-ROW SCREEN
    Applying every frozen gate row cuts 2,099 -> 136, which looks great until
    you check it against reality: it drops 13 of the 24 names that actually
    fired this week, including MONARCH and RADICO. Those rows are LAST CLOSED
    WEEK values, but the gate evaluates on the developing week - an RSI of 56
    on Friday can be 72 by Wednesday. Filtering on stale momentum throws away
    exactly the stocks that are accelerating. The three conditions kept below
    (c03 fresh breakout, c08 MACD>0, c05 EMA50 rising) retain 23 of those 24.

    python watchlist.py                  # send today's digest
    python watchlist.py --near 3 --top 15
    python watchlist.py --dry-run        # print, do not send
    python watchlist.py --no-structure   # rank on distance alone

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
from mcap import load_table as load_mcap_table
from scan import QUOTE_BATCH, load_snapshots
from state import AlertState
from strategy import WeeklySnapshot, week_start_of
from telegram import build_telegram, _esc, _fmt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger("watchlist")

# How many APPROACHING names to print. The complete table is attached as a CSV,
# so this is purely about keeping the message readable on a phone.
TOP_N = 15


def is_eligible(s: WeeklySnapshot, cfg, mcap: float | None = None) -> bool:
    """
    The FROZEN screens that are safe to apply pre-market.

    All use last-closed-week values, so they cannot change mid-week and can be
    evaluated straight from the snapshot with no extra API calls.

        c03  fresh breakout   close[1] <= 26W high two weeks ago
        c08  MACD histogram > 0
        c05  weekly EMA50 rising
        c12  market cap > min_mcap (when use_mcap is on)

    Deliberately NOT included: c04 (EMA20>EMA50), c06 (RSI>60), c07 (RSI
    rising). Each of those drops names that went on to fire this week - c06 at
    the strict 60 threshold alone loses 9 of 24.

    `mcap` is passed in from mcap.csv rather than read off the snapshot: a
    snapshot built before the market-cap feature carries mcap=None for every
    row, which would silently disable the filter here. Unknown still PASSES -
    never hide a name just because its cap could not be resolved.
    """
    if not (s.close_1 <= s.hi_short2
            and s.g_hist > 0.0
            and s.g_ema_slow > s.g_ema_slow_2):
        return False
    if getattr(cfg, "use_mcap", False) and mcap is not None:
        margin = 1.0 - float(getattr(cfg, "mcap_margin_pct", 0.0) or 0.0) / 100.0
        if mcap <= cfg.min_mcap * margin:
            return False
    return True


def fetch_ltp(client: DhanClient, snaps: list[WeeklySnapshot]) -> dict[str, float]:
    """symbol -> last traded price, via bulk quotes (1000 per request)."""
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


def build_message(week: str, rows: list[dict], counts: dict,
                  near_pct: float, top_n: int = TOP_N) -> str:
    fired = [r for r in rows if r["bucket"] == "FIRED"]
    watch = sorted((r for r in rows if r["bucket"] == "WATCH"),
                   key=lambda x: x["gap"])

    lines = [
        f"📋 <b>Watchlist — week of {week}</b>",
        f"<i>{datetime.now(IST):%d-%b-%Y %H:%M} IST · "
        f"{counts['universe']} symbols · {counts.get('eligible', 0)} eligible"
        + (f" · {counts['capped']} with mcap" if counts.get("capped") else "")
        + "</i>",
        "",
        f"👀 approaching <b>{len(watch)}</b> · 🔥 fired <b>{len(fired)}</b>",
    ]

    if watch:
        shown = watch[:top_n]
        lines += ["", f"👀 <b>CLOSEST TO BREAKOUT (top {len(shown)})</b>",
                  f"<i>within {near_pct:g}% of the 26W level, weekly structure "
                  f"intact</i>"]
        for r in shown:
            tag = r.get("which") or ""
            cap = r.get("mcap_cr")
            extra = f"  <i>[{tag}]</i>" if tag else ""
            if cap:
                extra += f" <i>{cap:,.0f}Cr</i>"
            lines.append(
                f"• <b>{_esc(r['symbol'])}</b>  {_fmt(r['ltp'])} → "
                f"<code>{_fmt(r['level'])}</code>  "
                f"<b>{r['gap']:.2f}%</b> away{extra}")
        if len(watch) > len(shown):
            lines.append(f"<i>… {len(watch) - len(shown)} more in the CSV</i>")
    else:
        lines += ["", "<i>Nothing approaching its level right now.</i>"]

    if fired:
        lines += ["", f"🔥 <b>ALREADY TRIGGERED ({len(fired)})</b>"]
        names = ", ".join(_esc(r["symbol"]) for r in
                          sorted(fired, key=lambda x: x["symbol"])[:60])
        lines.append(names + (f" … +{len(fired) - 60}" if len(fired) > 60 else ""))

    lines += ["", "<i>Levels are frozen all week. Alerts fire on the first 5m "
                  "close above the 26W high.</i>"]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--near", type=float, default=3.0,
                    help="percent below the level to consider (default 3)")
    ap.add_argument("--top", type=int, default=TOP_N,
                    help="how many names to list (default 15)")
    ap.add_argument("--no-structure", action="store_true",
                    help="rank on distance only, skip the c03/c05/c08 screen")
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

    # Market caps come from mcap.csv, NOT the snapshot: a snapshot built before
    # the c12 feature carries mcap=None for every row and would disable the
    # filter here without any visible symptom.
    caps = load_mcap_table(cfg.paths["mcap"]) if cfg.strategy.use_mcap else {}
    if cfg.strategy.use_mcap and not caps:
        log.warning("use_mcap is ON but %s is empty - the watchlist will not "
                    "apply the market-cap screen. Run `python mcap.py`.",
                    cfg.paths["mcap"])

    state = AlertState(cfg.paths["state"])
    rows: list[dict] = []
    eligible = 0
    for s in snaps:
        px = ltp.get(s.symbol)
        if px is None:
            continue

        # Track BOTH levels. Model C triggers on entry_level, Model D on
        # hi_short2, and hi_short2 <= entry_level, so the nearer of the two is
        # the one a breakout will reach first. Reporting only entry_level hid
        # every name that was about to trigger D.
        cap = caps.get(s.symbol.upper())
        lvl_c = s.entry_level
        lvl_d = s.hi_short2
        cand = [x for x in (lvl_c, lvl_d) if x and x > 0]
        if not cand:
            continue
        # nearest level still ABOVE price; if price is above both, use the higher
        above = [x for x in cand if px <= x]
        level = min(above) if above else max(cand)
        which = "D" if level == lvl_d and lvl_d != lvl_c else "C"
        if lvl_c == lvl_d:
            which = "C+D"

        ok = args.no_structure or is_eligible(s, cfg, cap)
        eligible += bool(ok)
        pct = (px - level) / level * 100.0

        if state.already_alerted(week_s, s.symbol):
            bucket = "FIRED"
        elif ok and -args.near <= pct <= 0:
            # Only names still BELOW the level are actionable: once price is
            # above it either the alert has fired or the weekly gate is holding
            # it back, and neither is something to watch for tomorrow.
            bucket = "WATCH"
        else:
            bucket = "OTHER"

        rows.append(dict(symbol=s.symbol, ltp=px, level=level, which=which,
                         level_c=lvl_c, level_d=lvl_d, mcap_cr=cap,
                         level_52=s.level_52, pct=pct, gap=-pct,
                         eligible=ok, bucket=bucket))

    counts = {"universe": len(snaps), "priced": len(ltp), "eligible": eligible,
              "capped": sum(1 for r in rows if r["mcap_cr"] is not None)}
    msg = build_message(week_s, rows, counts, args.near, args.top)

    table = pd.DataFrame(rows).sort_values("gap")
    csv_path = Path(args.csv) if args.csv else Path(f"watchlist_{week_s}.csv")
    table.to_csv(csv_path, index=False)
    log.info("wrote %s (%d rows)", csv_path, len(table))

    if args.dry_run:
        print(msg)
        return 0

    tg = build_telegram(cfg)
    tg.send(msg)

    # Attach every actionable name, so capping the message never loses data.
    actionable = table[table["bucket"].isin(("WATCH", "FIRED"))]
    if not actionable.empty:
        att = csv_path.with_name(
            f"watchlist_{week_s}_{datetime.now(IST):%d%b}.csv")
        actionable.to_csv(att, index=False)
        tg.send_document(att, caption=f"Watchlist — week of {week_s} "
                                      f"({len(actionable)} names)")
    log.info("sent: %d watch, %d fired",
             sum(1 for r in rows if r["bucket"] == "WATCH"),
             sum(1 for r in rows if r["bucket"] == "FIRED"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
