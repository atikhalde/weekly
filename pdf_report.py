#!/usr/bin/env python3
"""
BTST & Anticipation Master Paper Trade PDF Report Generator.

Generates an institutional-grade PDF summary containing:
  1. Stocks Entered Today (exactly what the Telegram alert picked)
  2. Yesterday's BTST Results (Realized P&L, from the real ab_ledger.csv)
  3. Active Open Positions & Multi-Day Runners (marked to a live quote)
  4. Closest Anticipated Watchlist for Tomorrow (same as the alert)
  5. Live-ledger performance summary (real trades only - see note below)

Can be called standalone or invoked automatically by post-market workflows (ab.yml).

--------------------------------------------------------------------------
2026-08-13 REWRITE - "report doesn't match the alert" bug fix
--------------------------------------------------------------------------
Every section below used to be a hardcoded Python literal (fake KPI banner,
a fixed "yesterday's results" list, a fixed "open positions" list, a fixed
"anticipated watchlist", and a fabricated "5-year, 1,564 trade" backtest
table) with a `ledger_path` parameter that was accepted but never once
referenced in the function body. None of it reflected what the live
scanner actually did, which is why the PDF disagreed with the Telegram
alert on a specific real-world date (2026-08-13: alert said "no setups
qualified, watch KMEW"; PDF said "MUNJALAU/JUBLCPL entered, watch
HAPPYFORGE/VINDHYATEL" - none of which came from any computation).

This rewrite computes every section from real, on-disk data:
  - "Stocks Entered Today" / "Anticipated Watchlist" -> btst_alert_state.json,
    the literal object btst.py's Telegram message was rendered from (see
    btst.py's write_alert_state()). This is the ONLY way to guarantee the
    PDF can never show a different pick than the alert, because multiple
    btst.py runs per day (two cron slots + a repository_dispatch trigger +
    occasional manual after-close reviews) each append to the same
    btst_picks.csv/anticipate_picks.csv, so "read today's rows" is not one
    consistent answer - re-deriving from those files was the root cause.
    Falls back to re-deriving from the latest same-day scan_time batch in
    the CSVs (via btst.build_actionable_lists, the same importable rule
    the alert itself uses) only for older dates that predate this fix.
  - "Yesterday's Results" / "Active Open Positions" -> ab_ledger.csv, the
    real, already-tested paper-trading ledger ab_paper.py builds.
  - KPI summary -> ab_paper.summarise(), the SAME function ab_paper.py's
    own Telegram standings message already uses - not a second, competing
    stats engine that could drift from it.

No 5-year / year-by-year backtest table: the live ledger is only ~2-3
weeks old (first real trade 2026-07-27), so a "5-year, 1,564 trade" claim
is not just wrong in its numbers but impossible in kind. Per an explicit
product decision (2026-08-13), this section instead shows real live-ledger
stats with an honest sample-size caveat, and does NOT pull in
btst_backtest.py's numbers, since that workflow is deliberately read-only
and never commits its output anywhere this script could read it.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, date
from pathlib import Path

import numpy as np
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

ROOT = Path(__file__).resolve().parent

# Color Palette
COLOR_NAVY = colors.HexColor("#0f172a")     # Dark Slate/Navy
COLOR_PRIMARY = colors.HexColor("#1e3a8a")  # Deep Blue
COLOR_ACCENT = colors.HexColor("#0284c7")   # Sky Blue
COLOR_GREEN = colors.HexColor("#15803d")    # Forest Green
COLOR_RED = colors.HexColor("#b91c1c")      # Deep Red
COLOR_BG_LIGHT = colors.HexColor("#f8fafc") # Slate Light
COLOR_BG_ALT = colors.HexColor("#f1f5f9")   # Alt row light gray
COLOR_BORDER = colors.HexColor("#cbd5e1")   # Border gray
COST_ROUND_TRIP = 0.22

# The real, live BTST/Anticipation models in ab_ledger.csv - these are the
# ones that represent an actual traded pick. E_btst_wide / F_anticipate_only
# / C_swing / D_early are internal comparison arms measuring alternative
# stops/exits on the SAME signal and must not be counted as separate trades
# here (see AB_TEST_GUIDE.md).
LIVE_MODELS = ("E_btst", "F_anticipate")


def _fmt(v: float | None) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    return f"{v:,.2f}"


def _is_blank(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() in ("", "nan", "None")


# --------------------------------------------------------------------------- #
#  Real data loaders (replace the old hardcoded literals)
# --------------------------------------------------------------------------- #
def load_alert_state(today_str: str, root: Path = ROOT) -> dict | None:
    """The exact top3_actionable/next_anticipated the Telegram alert used.

    Returns None if btst.py hasn't run since this fix shipped, or hasn't
    run today at all - callers must fall back rather than fabricate data.
    """
    path = root / "btst_alert_state.json"
    if not path.exists():
        return None
    try:
        state = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if state.get("date") != today_str:
        return None
    return state


def derive_today_lists_from_csv(today_str: str, root: Path = ROOT) -> tuple[list, list, list]:
    """Legacy fallback for dates before btst_alert_state.json existed.

    Reuses btst.py's own build_actionable_lists()/actionable_priority() -
    the same rule the alert is computed from - rather than restating a
    second copy of the ranking here. Restricted to the LATEST scan_time
    seen for today in each file, since btst.py can run more than once a
    day and earlier runs' rows for other symbols would otherwise leak in
    as if they were part of the same decision.
    """
    sys.path.insert(0, str(root))
    import btst as _btst

    bp_path = root / "btst_picks.csv"
    ap_path = root / "anticipate_picks.csv"
    bp_df = pd.read_csv(bp_path) if bp_path.exists() else pd.DataFrame()
    ap_df = pd.read_csv(ap_path) if ap_path.exists() else pd.DataFrame()

    def _latest_batch(df):
        if df.empty or "date" not in df.columns:
            return []
        sub = df[df["date"].astype(str) == today_str]
        if sub.empty or "scan_time" not in sub.columns:
            return sub.to_dict("records")
        latest_time = sub["scan_time"].astype(str).max()
        return sub[sub["scan_time"].astype(str) == latest_time].to_dict("records")

    picks = []
    for r in _latest_batch(bp_df):
        picks.append({
            "symbol": r.get("symbol"), "close": r.get("entry"), "day_ret": r.get("day_ret"),
            "rvol": r.get("rvol"), "tier": r.get("tier"), "fresh": str(r.get("arm", "")).startswith("fresh"),
            "close_pos": r.get("close_pos"), "high": r.get("entry"), "pre": r.get("pre", 0),
        })
    ant_picks = []
    for r in _latest_batch(ap_df):
        ant_picks.append({
            "symbol": r.get("symbol"), "close": r.get("entry"), "level": r.get("level"),
            "side": r.get("side"), "pre": r.get("pre", 0), "day_ret": r.get("day_ret"),
            "rvol": r.get("rvol"), "close_pos": r.get("close_pos"), "gap_pct": r.get("gap_pct"),
            "mcap_cr": r.get("mcap_cr"), "ret_12m": r.get("ret_12m"), "dist_200dma": r.get("dist_200dma"),
        })
    return _btst.build_actionable_lists(picks, ant_picks, ant_picks)


def load_ledger(ledger_path: str | Path) -> pd.DataFrame:
    ledger_path = Path(ledger_path)
    if not ledger_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(ledger_path)
    df["signal_date"] = df["signal_date"].astype(str)
    return df


def dedupe_live_trades(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (symbol, signal_date) for the live BTST/Anticipate models.

    ab_ledger.csv's own de-dup key is (model, symbol, signal_date,
    signal_time) - if btst.py ran more than once on the same day and both
    runs still flagged the same symbol (e.g. an early aged-pool pass and
    the official 15:20 close scan), that symbol gets TWO ledger rows with
    different signal_time, which would double count it in any KPI roll-up.
    Prefers the canonical 15:20 close-scan row when more than one exists.
    """
    sub = df[df["model"].isin(LIVE_MODELS)].copy()
    if sub.empty:
        return sub
    sub["_pref"] = (sub["signal_time"].astype(str) == "15:20").astype(int)
    sub = sub.sort_values(["symbol", "signal_date", "_pref"], ascending=[True, True, False])
    sub = sub.drop_duplicates(subset=["symbol", "signal_date"], keep="first")
    return sub.drop(columns=["_pref"])


def fetch_live_quotes(symbols: list[str], root: Path = ROOT) -> dict[str, float]:
    """Best-effort live LTP for open positions' mark-to-market.

    Uses DhanClient.ltp(), whose primary source is yfinance (no broker
    credentials needed) with Dhan as a fallback - see dhan.py. Returns {}
    on any failure; callers must show 'quote unavailable' rather than a
    stale/fake price, never fabricate a number.
    """
    if not symbols:
        return {}
    try:
        from dhan import DhanClient
        universe_path = root / "universe.csv"
        if not universe_path.exists():
            return {}
        uni = pd.read_csv(universe_path, dtype=str)
        uni = uni[uni["symbol"].str.upper().isin([s.upper() for s in symbols])]
        if uni.empty:
            return {}
        by_seg: dict[str, list[int]] = {}
        ident: dict[tuple, str] = {}
        for r in uni.itertuples():
            seg = r.exchange_segment
            sid = int(r.security_id)
            by_seg.setdefault(seg, []).append(sid)
            ident[(seg, str(sid))] = r.symbol
        client = DhanClient(
            os.environ.get("DHAN_CLIENT_ID", ""),
            os.environ.get("DHAN_ACCESS_TOKEN", ""),
        )
        raw = client.ltp(by_seg)
        out = {}
        for seg, m in raw.items():
            for sid, px in m.items():
                sym = ident.get((seg, str(sid)))
                if sym and px:
                    out[sym] = float(px)
        return out
    except Exception as exc:
        print(f"[pdf_report] live quote fetch failed ({exc}) - "
              f"open positions will show 'quote unavailable' instead of a fake price")
        return {}


def compute_yesterday_results(ledger: pd.DataFrame, today_str: str) -> list[dict]:
    """Real, resolved (closed) trades whose signal was BEFORE today.

    'Yesterday' means the most recent prior session with a resolved BTST/
    Anticipate trade in the ledger, not literally calendar-yesterday (the
    previous session could be a Friday, or skip a holiday).
    """
    live = dedupe_live_trades(ledger)
    if live.empty:
        return []
    prior = live[(live["signal_date"] < today_str) & (live["exit_reason"].astype(str) != "NO_FILL")
                 & (~live["exit_date"].apply(_is_blank))]
    if prior.empty:
        return []
    last_session = prior["signal_date"].max()
    rows = prior[prior["signal_date"] == last_session]
    out = []
    for r in rows.sort_values("symbol").itertuples():
        out.append({
            "symbol": r.symbol,
            "entry_date": r.signal_date,
            "entry": float(r.entry or 0),
            "exit": float(r.exit or 0),
            "pnl_pct": float(r.pnl_pct or 0),
            "pnl_rs": float(r.pnl or 0),
            "reason": str(r.exit_reason or "") + (f" · {r.exit_note}" if not _is_blank(getattr(r, "exit_note", None)) else ""),
        })
    return out


def compute_open_positions(ledger: pd.DataFrame, today_str: str, root: Path = ROOT) -> list[dict]:
    """Real still-open multi-day BTST/Anticipate positions (not today's
    fresh entries - those belong in 'Stocks Entered Today')."""
    live = dedupe_live_trades(ledger)
    if live.empty:
        return []
    open_rows = live[(live["signal_date"] < today_str) & (live["exit_date"].apply(_is_blank))
                      & (live["exit_reason"].astype(str) != "NO_FILL")]
    if open_rows.empty:
        return []
    quotes = fetch_live_quotes(list(open_rows["symbol"].unique()), root=root)
    out = []
    for r in open_rows.sort_values("symbol").itertuples():
        entry = float(r.entry or 0)
        stop = float(r.stop or 0)
        cur = quotes.get(r.symbol)
        u_pct = ((cur - entry) / entry * 100.0) if (cur and entry) else None
        out.append({
            "symbol": r.symbol, "entry_date": r.signal_date, "entry": entry,
            "current": cur, "stop": stop, "unrealized_pct": u_pct,
        })
    return out


def compute_kpi_summary(ledger: pd.DataFrame) -> dict:
    """Real KPI roll-up over the live ledger, reusing ab_paper.summarise()
    (the same math already trusted by ab_paper.py's own Telegram standings
    message) rather than a second, independently-drifting stats engine."""
    sys.path.insert(0, str(ROOT))
    from ab_paper import summarise

    live = dedupe_live_trades(ledger)
    if live.empty:
        return {"n": 0}
    stats = summarise(live)
    stats["since"] = live["signal_date"].min() if not live.empty else None
    return stats


# --------------------------------------------------------------------------- #
#  PDF builder
# --------------------------------------------------------------------------- #
def build_pdf_report(
    ledger_path: str | Path = "ab_ledger.csv",
    output_pdf: str | Path = "btst_paper_trade_report.pdf",
    today_str: str | None = None
) -> str:
    """Build the complete, accurate BTST & Anticipation Master PDF Report."""
    now = datetime.now()
    if today_str is None:
        today_str = now.strftime("%Y-%m-%d")

    ledger = load_ledger(ledger_path)
    alert_state = load_alert_state(today_str)
    if alert_state is not None:
        top3_actionable = alert_state.get("top3_actionable", [])
        next_anticipated = alert_state.get("next_anticipated", [])
        excluded_locked = alert_state.get("excluded_locked", [])
        data_source_note = f"alert snapshot · {alert_state.get('scan_time', '?')} IST"
    else:
        top3_actionable, next_anticipated, excluded_locked = derive_today_lists_from_csv(today_str)
        data_source_note = "reconstructed from picks CSV (no alert snapshot for this date)"

    yesterday_rows = compute_yesterday_results(ledger, today_str)
    open_rows = compute_open_positions(ledger, today_str)
    kpis = compute_kpi_summary(ledger)

    # Document setup
    doc = SimpleDocTemplate(
        str(output_pdf),
        pagesize=letter,
        topMargin=28, bottomMargin=28, leftMargin=28, rightMargin=28,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleStyle", parent=styles["Heading1"], fontName="Helvetica-Bold",
        fontSize=15, leading=18, textColor=COLOR_NAVY, spaceAfter=2,
    )
    subtitle_style = ParagraphStyle(
        "SubtitleStyle", parent=styles["Normal"], fontName="Helvetica",
        fontSize=8, leading=10, textColor=colors.HexColor("#475569"),
    )
    section_head = ParagraphStyle(
        "SectionHead", parent=styles["Heading2"], fontName="Helvetica-Bold",
        fontSize=10.5, leading=13, textColor=COLOR_NAVY, spaceBefore=6, spaceAfter=3,
    )
    cell_head = ParagraphStyle(
        "CellHead", parent=styles["Normal"], fontName="Helvetica-Bold",
        fontSize=7.5, leading=9.5, textColor=colors.white, alignment=1,
    )
    cell_txt = ParagraphStyle(
        "CellText", parent=styles["Normal"], fontName="Helvetica",
        fontSize=7, leading=8.5, textColor=COLOR_NAVY,
    )
    cell_txt_center = ParagraphStyle(
        "CellTextCenter", parent=styles["Normal"], fontName="Helvetica",
        fontSize=7, leading=8.5, textColor=COLOR_NAVY, alignment=1,
    )
    cell_txt_bold = ParagraphStyle(
        "CellTextBold", parent=styles["Normal"], fontName="Helvetica-Bold",
        fontSize=7, leading=8.5, textColor=COLOR_NAVY,
    )
    cell_green = ParagraphStyle(
        "CellGreen", parent=styles["Normal"], fontName="Helvetica-Bold",
        fontSize=7, leading=8.5, textColor=COLOR_GREEN, alignment=2,
    )
    cell_red = ParagraphStyle(
        "CellRed", parent=styles["Normal"], fontName="Helvetica-Bold",
        fontSize=7, leading=8.5, textColor=COLOR_RED, alignment=2,
    )

    story = []

    # 1. Header Banner
    story.append(Paragraph("🌙 BTST & ANTICIPATION MASTER TRADING REPORT", title_style))
    story.append(Paragraph(
        f"<b>Session Date:</b> {today_str} · <b>Generated:</b> {now:%d-%b-%Y %H:%M} IST · "
        f"<b>Today's entries source:</b> {data_source_note} · "
        f"<b>Sizing:</b> ₹1,00,000 / trade",
        subtitle_style
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=COLOR_PRIMARY, spaceBefore=0, spaceAfter=6))

    # 2. Portfolio KPIs Banner - real numbers from ab_ledger.csv, or an
    # honest "not enough data yet" note. No fabricated CAGR/5-year claims.
    n_trades = int(kpis.get("n", 0) or 0)
    if n_trades > 0:
        since = kpis.get("since", "?")
        kpi_data = [[
            Paragraph(f"<b>Trades (closed)</b><br/>{n_trades} since {since}", cell_txt_center),
            Paragraph(f"<b>Win Rate</b><br/>{kpis.get('win', 0):.1f}%", cell_txt_center),
            Paragraph(f"<b>Profit Factor</b><br/>{kpis.get('pf', 0):.2f}", cell_txt_center),
            Paragraph(f"<b>Avg / Median %</b><br/>{kpis.get('avg_pct', 0):+.2f}% / {kpis.get('med_pct', 0):+.2f}%", cell_txt_center),
            Paragraph(f"<b>Net P&amp;L (fixed sizing)</b><br/>₹{kpis.get('net', 0):+,.0f}", cell_txt_center),
        ]]
    else:
        kpi_data = [[Paragraph(
            "<b>No closed BTST/Anticipate trades in ab_ledger.csv yet</b> - KPIs will appear here once trades resolve.",
            cell_txt_center)]]
    kpi_table = Table(kpi_data, colWidths=[112, 114, 114, 112, 112] if n_trades > 0 else [564])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), COLOR_BG_ALT),
        ('BOX', (0, 0), (-1, -1), 1, COLOR_BORDER),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, COLOR_BORDER),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(kpi_table)
    if n_trades > 0:
        story.append(Paragraph(
            f"<i>Live-ledger stats only ({n_trades} real closed trade(s) since {kpis.get('since','?')}) - "
            f"not a multi-year backtest. Run the 'BTST Backtest (on demand)' workflow separately for "
            f"long-horizon history; its output is not auto-committed, so it cannot be shown here live.</i>",
            subtitle_style
        ))
    story.append(Spacer(1, 6))

    # 3. Table 1: STOCKS ENTERED TODAY
    story.append(Paragraph(f"🛒 1. STOCKS ENTERED TODAY — {today_str}", section_head))
    today_picks = []
    for act in top3_actionable:
        qty = int(act.get("qty", 0) or 0)
        price = float(act.get("price", 0) or 0)
        badge = act.get("badge", "")
        prime = act.get("prime", "")
        label = f"{badge} ({prime})" if prime else badge
        today_picks.append({
            "symbol": act["symbol"], "badge": label, "qty": qty, "entry": price,
            "invested": qty * price, "stop": price * 0.99,
            "day_ret": float(act.get("day_ret", 0) or 0), "rvol": float(act.get("rvol", 0) or 0),
            "status": "🟡 OPEN (Exit 09:15)",
        })

    if today_picks:
        t1_rows = [[
            Paragraph("<b>#</b>", cell_head), Paragraph("<b>Symbol</b>", cell_head),
            Paragraph("<b>Setup Category</b>", cell_head), Paragraph("<b>Qty</b>", cell_head),
            Paragraph("<b>Entry (₹)</b>", cell_head), Paragraph("<b>Invested (₹)</b>", cell_head),
            Paragraph("<b>Stop Loss (₹)</b>", cell_head), Paragraph("<b>Day % / RVOL</b>", cell_head),
            Paragraph("<b>Execution Status</b>", cell_head),
        ]]
        for idx, item in enumerate(today_picks, 1):
            t1_rows.append([
                Paragraph(str(idx), cell_txt_center),
                Paragraph(f"<b>{item['symbol']}</b>", cell_txt_bold),
                Paragraph(item['badge'], cell_txt),
                Paragraph(f"{item['qty']:,}", cell_txt_center),
                Paragraph(f"₹{_fmt(item['entry'])}", cell_txt_center),
                Paragraph(f"₹{item['invested']:,.0f}", cell_txt_center),
                Paragraph(f"₹{_fmt(item['stop'])}", cell_txt_center),
                Paragraph(f"{item['day_ret']:+.1f}% · {item['rvol']:.1f}x", cell_txt_center),
                Paragraph(f"<b>{item['status']}</b>", cell_txt_center),
            ])
        t1_table = Table(t1_rows, colWidths=[18, 72, 120, 36, 60, 68, 60, 60, 70])
        t1_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), COLOR_PRIMARY),
            ('BOX', (0, 0), (-1, -1), 1, COLOR_BORDER),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, COLOR_BORDER),
            ('TOPPADDING', (0, 0), (-1, -1), 2.5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ]))
        story.append(t1_table)
    else:
        story.append(Paragraph("<i>No setups qualified today matching quality filters.</i>", cell_txt))
    if excluded_locked:
        lock_names = ", ".join(f"<b>{s}</b> ({why})" for s, why in excluded_locked)
        story.append(Spacer(1, 2))
        story.append(Paragraph(f"🚫 <i>Excluded: {lock_names} — 0 sellers.</i>", subtitle_style))
    story.append(Spacer(1, 6))

    # 4. Table 2: YESTERDAY'S BTST RESULTS - real, from ab_ledger.csv
    story.append(Paragraph("📊 2. YESTERDAY'S BTST RESULTS (Realized & Closed)", section_head))
    if yesterday_rows:
        t2_rows = [[
            Paragraph("<b>Symbol</b>", cell_head), Paragraph("<b>Entry Date</b>", cell_head),
            Paragraph("<b>Entry (₹)</b>", cell_head), Paragraph("<b>Exit (₹)</b>", cell_head),
            Paragraph("<b>Net Move %</b>", cell_head), Paragraph("<b>Realized P&amp;L (₹)</b>", cell_head),
            Paragraph("<b>Exit Reason</b>", cell_head),
        ]]
        for item in yesterday_rows:
            p_style = cell_green if item["pnl_pct"] > 0 else cell_red
            t2_rows.append([
                Paragraph(f"<b>{item['symbol']}</b>", cell_txt_bold),
                Paragraph(str(item["entry_date"]), cell_txt_center),
                Paragraph(f"₹{_fmt(item['entry'])}", cell_txt_center),
                Paragraph(f"₹{_fmt(item['exit'])}", cell_txt_center),
                Paragraph(f"{item['pnl_pct']:+.2f}%", p_style),
                Paragraph(f"{item['pnl_rs']:+,.0f}", p_style),
                Paragraph(str(item["reason"]), cell_txt),
            ])
        t2_table = Table(t2_rows, colWidths=[75, 65, 65, 65, 64, 80, 150])
        t2_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), COLOR_NAVY),
            ('BOX', (0, 0), (-1, -1), 1, COLOR_BORDER),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, COLOR_BORDER),
            ('TOPPADDING', (0, 0), (-1, -1), 2.5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ]))
        story.append(t2_table)
    else:
        story.append(Paragraph("<i>No resolved BTST/Anticipate trades from the prior session yet.</i>", cell_txt))
    story.append(Spacer(1, 6))

    # 5. Table 3: ACTIVE OPEN POSITIONS & MULTI-DAY RUNNERS - real, marked
    # to a live quote when available.
    story.append(Paragraph("🟡 3. ACTIVE OPEN POSITIONS & MULTI-DAY RUNNERS", section_head))
    if open_rows:
        t3_rows = [[
            Paragraph("<b>Symbol</b>", cell_head), Paragraph("<b>Entry Date</b>", cell_head),
            Paragraph("<b>Entry Price (₹)</b>", cell_head), Paragraph("<b>Current Price (₹)</b>", cell_head),
            Paragraph("<b>Stop (₹)</b>", cell_head), Paragraph("<b>Unrealized %</b>", cell_head),
            Paragraph("<b>Position Status</b>", cell_head),
        ]]
        for item in open_rows:
            cur_txt = f"₹{_fmt(item['current'])}" if item["current"] is not None else "quote unavailable"
            u_txt = f"{item['unrealized_pct']:+.2f}%" if item["unrealized_pct"] is not None else "-"
            u_style = (cell_green if (item["unrealized_pct"] or 0) > 0 else cell_red) if item["unrealized_pct"] is not None else cell_txt_center
            t3_rows.append([
                Paragraph(f"<b>{item['symbol']}</b>", cell_txt_bold),
                Paragraph(str(item["entry_date"]), cell_txt_center),
                Paragraph(f"₹{_fmt(item['entry'])}", cell_txt_center),
                Paragraph(cur_txt, cell_txt_center),
                Paragraph(f"₹{_fmt(item['stop'])}", cell_txt_center),
                Paragraph(u_txt, u_style),
                Paragraph("🟡 Active (multi-day runner)", cell_txt),
            ])
        t3_table = Table(t3_rows, colWidths=[75, 65, 75, 75, 75, 65, 134])
        t3_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), COLOR_PRIMARY),
            ('BOX', (0, 0), (-1, -1), 1, COLOR_BORDER),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, COLOR_BORDER),
            ('TOPPADDING', (0, 0), (-1, -1), 2.5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ]))
        story.append(t3_table)
    else:
        story.append(Paragraph("<i>No multi-day runners currently open.</i>", cell_txt))
    story.append(Spacer(1, 6))

    # 6. Table 4: CLOSEST ANTICIPATED WATCHLIST - same data as the alert
    story.append(Paragraph("🔭 4. CLOSEST ANTICIPATED WATCHLIST (Top 2 Setups for Tomorrow)", section_head))
    if next_anticipated:
        t4_rows = [[
            Paragraph("<b>Symbol</b>", cell_head), Paragraph("<b>LTP (₹)</b>", cell_head),
            Paragraph("<b>26W Level (₹)</b>", cell_head), Paragraph("<b>Proximity Gap</b>", cell_head),
            Paragraph("<b>PRE Score</b>", cell_head), Paragraph("<b>Day % / RVOL</b>", cell_head),
            Paragraph("<b>Watchlist Target Action</b>", cell_head),
        ]]
        for r in next_anticipated:
            side = "above" if r.get("side") == "above" else "below"
            t4_rows.append([
                Paragraph(f"<b>{r['symbol']}</b>", cell_txt_bold),
                Paragraph(f"₹{_fmt(float(r.get('close', 0) or 0))}", cell_txt_center),
                Paragraph(f"₹{_fmt(float(r.get('level', 0) or 0))}", cell_txt_center),
                Paragraph(f"<b>{abs(float(r.get('gap_pct', 0) or 0)):.2f}% {side}</b>", cell_txt_center),
                Paragraph(f"{int(r.get('pre', 0) or 0)}/8", cell_txt_center),
                Paragraph(f"{float(r.get('day_ret', 0) or 0):+.1f}% · {float(r.get('rvol', 0) or 0):.1f}x", cell_txt_center),
                Paragraph("Watch for tomorrow's breakout cross", cell_txt),
            ])
        t4_table = Table(t4_rows, colWidths=[75, 65, 75, 80, 55, 75, 139])
        t4_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), COLOR_NAVY),
            ('BOX', (0, 0), (-1, -1), 1, COLOR_BORDER),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, COLOR_BORDER),
            ('TOPPADDING', (0, 0), (-1, -1), 2.5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ]))
        story.append(t4_table)
    else:
        story.append(Paragraph("<i>No anticipated setups within range today.</i>", cell_txt))

    # Build PDF document
    doc.build(story)
    return str(output_pdf)


if __name__ == "__main__":
    out_file = sys.argv[1] if len(sys.argv) > 1 else "btst_paper_trade_report.pdf"
    pdf_path = build_pdf_report(output_pdf=out_file)
    print(f"Generated PDF: {pdf_path}")
