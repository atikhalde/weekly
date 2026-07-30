#!/usr/bin/env python3
"""MODEL 9 — CASH-UNIVERSE DATA FARM × TOP-20 GAINERS GATE (29-Jul-2026).

Shipped as a paper-only evidence lab (user pick A: data-farm even though the
same-day offline proof of the RAW variant ran net-negative). Its job is
EVIDENCE, not profit: it feeds the Phase-0 learning log the gated cash-segment
signal universe — and every BLOCKED signal is still journaled with its reason
and gets counterfactual outcome labels from label_learn next morning, so the
ML dataset captures both sides of the gate.

UNIVERSE (m9_universe.csv, rebuilt 29-Jul): NSE cash EQ series only — NOT in
F&O, NOT in Nifty 200, market cap > Rs1,000 cr (Yahoo snapshot). 1,050 symbols
split into 3 alphabetical shards A/B/C (350 each) — 2 shards were measured too
slow for the 25-min workflow budget; the union of the 3 ledgers is the model.

ENTRY (user spec card, confirmed "go, all looks good" 29-Jul):
  BUY master signals only, ALL EX variants (no B2/EX9 filters), no OI gate, no
  sector gate. Shared blocks: 90/290 scanner-table previews + pre-09:26 window.
  - signal close (LTP) must be > Rs200                       (rule 4)
  - entry-candle range (high-low)/low <= 1.25%               (rule 7)
  - entry = signal-bar close, causal once-per-bar-close scan (rule 6)
  - 1-open-trade-per-symbol rule kept (same as every model)
  + GATE (user pick A + "replace M9", 29-Jul): stock must be on NSE's live
    "top gainers — ALL SECURITIES" snapshot (top-20 by %change), direction-
    aligned like M2's BUY side. Feed offline -> strict: no entries that cycle.
EXITS:
  - SL = entry-candle LOW - 0.02%                            (rule 8)
  - no target; from the NEXT closed 5m bar, any close < EMA(5) of closes
    (continuous series, seeded from committed history warmup) -> exit 100%
    at that bar's close; the signal bar itself is exempt     (rule 9)
  - forced square-off at the 15:20 bar close                 (rule 10)
  - sizing: Rs50k notional, planned risk <= Rs900 (qty shrunk) (rule 11)
  - July-2026 costs + slippage model via costs.trade_costs
SELL-side master signals are NOT traded (BUY-only model) but are logged into
the learn log as skip rows — evidence for a future short-side variant.

ALERTS (user 29-Jul: "should send the alert as well"): paper-trade alerts go
OUT LOUD — ENTRY / EXIT_SL / EXIT_EMA5 / EXIT_EOD, each tagged by shard
(🅼9-A/B/C). Skips stay quiet in the logs + EOD xlsx (matches the M1/M2
only-paper-trade-alerts rule). Silent heartbeat each cycle; loud EOD + xlsx.
Ledger state9a/b/c.json · tag M9A/M9B/M9C · workflows "9A/9B/9C. LIVE M9-*" · joins the Learning Log
automatically as models M9A/M9B/M9C (labels next morning via label_learn).
Usage: python -u m9_runner.py --shard A [--loop N] [--sleep SECS]
"""
import argparse
import datetime as dt
import json
import sys
import time

import pandas as pd

import live_runner as L                  # engine, engine contract, state helpers
import learn_log
import feeds, trader, report, gate
import telegram_bot as tg
import costs

HIST9 = L.ROOT / "data" / "history9"
learn_log.HIST = HIST9   # learn-log prev-close reads M9 history (monkeypatch, M9 process only)

MIN_PRICE = 200.0                 # rule 4: LTP > Rs200 at the entry bar
MAX_RANGE = 0.0125                # rule 7: entry-candle range cap 1.25%
SL_BUF = 0.0002                   # rule 8: SL = entry-candle low -0.02%
SQOFF = "15:20"                   # rule 10
NOTIONAL = 50000.0                # rule 11
RISK_CAP = 900.0                  # rule 11
EMA_SPAN = 5                      # rule 9
MIN_HIST_BARS = 300               # warmup sanity guard: skip symbols with no/thin history

M9_RULES = ("M9 CASH DATA-FARM × TOP-20 GAINERS GATE (evidence, not a profit model) · universe: NSE cash EQ, "
            "not F&O, not Nifty-200, mcap>Rs1,000cr · BUY master signals only, ALL EX variants · no OI/sector gates · "
            "GATE: stock must be on NSE live top-20 gainers (all-securities) snapshot; feed offline = strict no-trade · "
            "90/290 + <09:26 blocked · entry @ signal-bar close, LTP>Rs200, candle range<=1.25% · "
            "SL entry-candle low -0.02% · no target: exit on any 5m close < EMA(5) (continuous series) · "
            "sq-off 15:20 · Rs50k notional, risk<=Rs900 · 1-open/stock · " + costs.NOTE)

UNI = pd.read_csv(L.ROOT / "m9_universe.csv").sort_values("symbol").reset_index(drop=True)


# ---------------------------------------------------------------- M9 trade sim
def _ema5(warm, bars):
    """EMA(5) of closes on the continuous (history-warmup + today) close series,
    returned aligned to `bars` (offset convention identical to trader._indicators)."""
    cols = ["open", "high", "low", "close"]
    base = pd.concat([warm, bars[cols].reset_index(drop=True)], ignore_index=True) \
        if warm is not None and len(warm) else bars[cols].reset_index(drop=True)
    e5 = base["close"].ewm(span=EMA_SPAN, adjust=False).mean().values
    return e5, len(base) - len(bars)


def evaluate_m9(sym, etime, entry, signal, bars, warmup=None):
    """M9 paper trade (BUY-only), deterministic over today's bars — safe to
    re-run every live cycle. bars: today's 5m df RangeIndex with 't' column.
    Returns the same dict shape as trader.evaluate (report/costs compatible)."""
    ei_list = bars.index[bars["t"] == etime].tolist()
    if not ei_list:
        return {"symbol": sym, "error": "entry bar missing"}
    ei = ei_list[0]
    entry = float(entry)
    e5, off = _ema5(warmup, bars)

    elow = float(bars["low"].iloc[ei])
    sl = elow * (1 - SL_BUF)                       # rule 8: entry-candle LOW -0.02%
    risk = entry - sl
    if risk <= 0:
        return {"symbol": sym, "error": "SL beyond entry (degenerate candle)"}
    risk_pct = risk / entry * 100
    qty_full = int(NOTIONAL // entry)
    qty = qty_full
    if qty * risk > RISK_CAP:                      # rule 11: shrink qty, SL untouched
        qty = max(1, int(RISK_CAP // risk))
    if qty < 1:
        return {"symbol": sym, "error": "qty=0"}

    legs, events = [], [{"key": "ENTRY", "time": etime, "price": entry}]
    closed, exit_t = False, None
    why = None
    for j in range(ei + 1, len(bars)):             # signal bar exempt (rule 9)
        o = float(bars["open"].iloc[j]); h = float(bars["high"].iloc[j])
        l = float(bars["low"].iloc[j]); c = float(bars["close"].iloc[j]); t = str(bars["t"].iloc[j])
        # 1) SL first inside a bar (gap-through fills at the open)
        if l <= sl:
            px = o if o < sl else sl
            legs.append((f"SL {t}", qty, px, t))
            events.append({"key": "EXIT_SL", "time": t, "price": px})
            closed, exit_t, why = True, t, "SL"
            break
        # 2) EMA(5) trail: close below EMA5 -> exit at that bar's close
        if c < float(e5[off + j]):
            legs.append((f"EMA5 {t}", qty, c, t))
            events.append({"key": "EXIT_EMA5", "time": t, "price": c})
            closed, exit_t, why = True, t, "EMA5"
            break
        # 3) forced square-off
        if t == SQOFF:
            legs.append((f"EOD {t}", qty, c, t))
            events.append({"key": "EXIT_EOD", "time": t, "price": c})
            closed, exit_t, why = True, t, "EOD"
            break

    if not closed:
        last_c, last_t = float(bars["close"].iloc[-1]), str(bars["t"].iloc[-1])
        legs.append((f"OPEN {last_t}", qty, last_c, last_t))

    pnl = sum((px - entry) * q for _lbl, q, px, _t in legs)
    risk_rs = risk * qty
    r_total = pnl / risk_rs if risk_rs else 0.0
    exit_text = (f"100% {why} {exit_t}" if closed else "OPEN")
    last_leg_t = legs[-1][3] if legs else None
    return {
        "symbol": sym, "side": "BUY", "time": etime, "signal": signal,
        "setup": "M9-DataFarm", "sl_mode": "entry-candle-low",
        "entry": round(entry, 2), "sl": round(sl, 2), "sl_anchor": "entry-candle low −0.02%",
        "risk_pts": round(risk, 2), "risk_pct": round(risk_pct, 3), "risk_rs": round(risk_rs, 0),
        "qty": qty, "qty_full": qty_full, "qty_capped": qty < qty_full,
        "capital": round(qty * entry, 0),
        "trail_armed": False, "trail_style": "EMA5 close",
        "legs": legs, "exit_text": exit_text, "leg2_time": last_leg_t,
        "pnl": round(pnl, 0), "r_total": round(r_total, 2), "closed": closed,
        "events": events,
    }


# ---------------------------------------------------------------- state
def state_path(shard):
    return L.ROOT / f"state9{shard.lower()}.json"


def load_state(shard, today):
    fp = state_path(shard)
    if fp.exists():
        st = json.loads(fp.read_text())
        if st.get("date") == today:
            return st
    return {"date": today, "signals": {}, "trades": {}, "alerts": [], "gate": {},
            "eod_done": False, "cycles": 0}


def save_state(shard, st):
    state_path(shard).write_text(json.dumps(st, indent=1))


def shard_universe(shard):
    """Deterministic alphabetical 3-way split (A=0,B=1,C=2) — compute detail only."""
    k = {"A": 0, "B": 1, "C": 2}[shard]
    u = UNI[UNI.index % 3 == k].reset_index(drop=True)
    return u["symbol"].tolist(), dict(zip(u["symbol"], u["dhan_security_id"]))


def _hist(sym):
    """Committed M9 history for one symbol (read at most once per cycle)."""
    fp = HIST9 / f"{sym}.csv"
    if not fp.exists():
        return None
    try:
        h = pd.read_csv(fp, parse_dates=["dt"])
        h["dt"] = pd.to_datetime(h["dt"], utc=True).dt.tz_convert("Asia/Kolkata") \
            if h["dt"].dt.tz is None else h["dt"]
        return h
    except Exception:
        return None


def _frame(hdf, today_bars, today):
    """L.engine_frame with a pre-read history df (one csv read per symbol per
    cycle instead of one per scanned bar — compute only, engine input identical)."""
    df = today_bars.set_index("dt")[["open", "high", "low", "close", "volume"]]
    if hdf is not None:
        h = hdf[hdf["dt"].dt.strftime("%Y-%m-%d") != today].set_index("dt")
        df = pd.concat([h[["open", "high", "low", "close", "volume"]], df])
    df = df.sort_index()
    df.index = pd.to_datetime(df.index, utc=True).tz_convert("Asia/Kolkata")
    df.index.name = None
    return df


def skip_log(st, sym, side, name, etime, entry, why, rng=None):
    st.setdefault("skipped", []).append(
        {"symbol": sym, "side": side, "signal": name, "time": etime,
         "entry": round(entry, 2), "why": why})


def fmt_m9_alert(shard, tr, key):
    """Telegram text for M9 paper-trade alerts (user 29-Jul: 'should send the
    alert as well'). Entry + exit legs go OUT LOUD, tagged with the shard;
    skips stay quiet in the logs (matches the M1/M2 only-paper-trade-alerts rule)."""
    tag = f"🅼9-{shard} "
    base = f"<b>{tr['symbol']}</b> 🟢 BUY · {tr['signal']}"
    if key == "ENTRY":
        cap = f" (capped from {tr['qty_full']} for the ₹900 max-loss rule)" if tr.get("qty_capped") else ""
        return (f"🚨 {tag}ENTRY · {base}\n"
                f"Time {tr['time']} · ₹{tr['entry']} · Qty {tr['qty']}{cap} (₹{tr['capital']:,.0f})\n"
                f"SL ₹{tr['sl']} ({tr['sl_anchor']} · max loss ₹{tr['risk_rs']:,.0f})\n"
                f"📈 on NSE top-20 gainers (all-sec) · target: OPEN — rides until a 5m close < EMA(5) · 15:20 sq-off")
    if key == "EXIT_EMA5":
        return (f"📉 {tag}EMA5 EXIT · {base}\n"
                f"5m close below EMA(5) @ ₹{tr['legs'][-1][2]} {tr['legs'][-1][3]} · "
                f"P&L ₹{tr['pnl']:+,.0f} ({tr['r_total']:+.2f}R)")
    if key == "EXIT_SL":
        return (f"⛔ {tag}EXIT SL · {base}\n"
                f"@ ₹{tr['legs'][-1][2]} {tr['legs'][-1][3]} · P&L ₹{tr['pnl']:+,.0f} ({tr['r_total']:+.2f}R)")
    if key == "EXIT_EOD":
        return (f"🏁 {tag}EXIT 15:20 · {base}\n"
                f"@ ₹{tr['legs'][-1][2]} · P&L ₹{tr['pnl']:+,.0f} ({tr['r_total']:+.2f}R)")
    return f"{tag}{base}"


# ---------------------------------------------------------------- live cycle
def mode_live(shard):
    """One M9 shard cycle. Returns True if a cycle ran, False if idle."""
    now = L.now_ist()
    today = now.strftime("%Y-%m-%d")
    st = load_state(shard, today)
    hhmm = now.strftime("%H:%M")
    tag = f"M9{shard}"
    if st["eod_done"]:
        print(f"{tag}: EOD done — idle.")
        save_state(shard, st); return False
    if hhmm < "09:16":
        print(f"{tag}: pre-market — idle.")
        save_state(shard, st); return False

    syms, sids = shard_universe(shard)

    # --- fetch today's bars (one bad feed must never kill the cycle)
    bars_map = {}
    for sym in syms:
        try:
            b, _src = feeds.fetch_today(sym, sids[sym], now)
            if b is not None and not b.empty:
                b = b.sort_values("dt").drop_duplicates("dt").reset_index(drop=True)
                b["t"] = b["dt"].dt.strftime("%H:%M")
                bars_map[sym] = b
        except Exception as e:
            print(f"  feed {sym}: {type(e).__name__}: {e}")
        time.sleep(0.35)              # gentle: ~350 dhan calls per shard cycle

    # --- live NSE top-gainers (all-securities) snapshot for this cycle's gate
    gainers, gmeta = gate.nse_movers_gainers(top_n=20)
    st["gate"] = {"status": f"{tag} DATA-FARM × top-20 gainers gate (BUY-only cash >₹1,000cr)",
                  "source": (f"{len(syms)}-sym shard · gainers feed {gmeta['status']} "
                             f"({gmeta.get('count', 0)} parsed → top-20) · px>₹200 · candle≤1.25% · "
                             f"SL candle-low−0.02% · EMA5 trail · 15:20 sqoff · {len(bars_map)} fed"),
                  "gainers_pass": len(gainers)}
    print(f"  gainers snapshot: {gmeta['status']} · {len(gainers)}/{gmeta.get('count', 0)} names")

    # --- history cache + warmup guard (engine with no warmup = garbage signals)
    hist_map = {}
    no_hist = []
    for sym in bars_map:
        h = _hist(sym)
        if h is None or len(h) < MIN_HIST_BARS:
            no_hist.append(sym)
        else:
            hist_map[sym] = h
    if no_hist:
        print(f"  {tag}: {len(no_hist)} symbols w/o usable history9 — skipped from scan "
              f"(bootstrap pending or thin history): {', '.join(no_hist[:8])}{' …' if len(no_hist) > 8 else ''}")

    # --- manage open trades FIRST (deterministic re-eval; trades stay in ledger)
    for tkey in list(st["trades"].keys()):
        sym = tkey.split("#")[0]
        tbars = bars_map.get(sym)
        if tbars is None:
            continue
        tr = st["trades"][tkey]
        try:
            new_tr = evaluate_m9(sym, tr["time"], float(tr["entry"]), tr["signal"], tbars,
                                 warmup=trader.load_warmup(HIST9 / f"{sym}.csv", today))
            if "error" in new_tr:
                continue
            st["trades"][tkey] = new_tr
            for ev in new_tr["events"]:
                key = f"{tkey}:{ev['key']}"
                if ev["key"] != "ENTRY" and key not in st["alerts"]:
                    print(f"  {tag} exit (alert) {sym} {ev['key']} @ {ev['price']} {ev['time']}")
                    tg.send_message(fmt_m9_alert(shard, new_tr, ev["key"]))
                    st["alerts"].append(key)
                    save_state(shard, st)   # persist alert registry instantly (no-repeat guarantee)
        except Exception as e:
            print(f"  manage {tkey}: {type(e).__name__}: {e}")

    # --- engine -> master signals -> NO GATES -> M9 paper entry (causal, per bar close)
    params = L.ms.Params(enable_buy_ex10=False, enable_buy_ex11=False)
    entries_now = 0
    skipped_now = 0
    for sym, tbars in bars_map.items():
        if sym not in hist_map:
            continue                            # no warmup -> not scanned (guard above)
        n_today = len(tbars)
        known = int(st["signals"].get(sym, {}).get("nbars", 0))
        if known > n_today:
            known = 0
        for j in range(known, n_today):
            t_bar = tbars["dt"].iloc[j]
            try:
                tk = pd.Timestamp(t_bar)
                if tk.tzinfo is None:
                    tk = tk.tz_localize(now.tz)
                if tk + pd.Timedelta(minutes=5) > pd.Timestamp(now):
                    break
                df = _frame(hist_map[sym], tbars.iloc[: j + 1], today)
                res = L.ms.run_symbol(df, params)
                row = res.iloc[-1]
                # POINTER-FIRST (29-Jul dry-run finding): the scan cursor must advance
                # BEFORE the skip/entry dispatch — a `continue` in any skip branch must
                # never leave the bar un-registered, else every later cycle re-scans it:
                # duplicate skip rows, and a skip can silently become an entry when a
                # time-varying gate (e.g. gainers feed recovering) flips state.
                st["signals"][sym] = st["signals"].get(sym, {})
                st["signals"][sym]["nbars"] = j + 1
            except Exception as e:
                print(f"  engine {sym}: {e}")
                st.setdefault("scan_err", {})[sym] = j
                break                           # keep pointer, retry next cycle
            try:
                code = row.get("scan_code")
                if not pd.isna(code) and int(code) in L.MASTER_CODES:
                    code = int(code)
                    side = "BUY" if code < 200 else "SELL"
                    etime = tk.strftime("%H:%M")
                    entry = float(tbars["close"].iloc[j])
                    name = str(row.get("scan_name", code))
                    bh = float(tbars["high"].iloc[j]); bl = float(tbars["low"].iloc[j])
                    rng = (bh - bl) / bl if bl > 0 else 9.99
                    why = None
                    if code in (90, 290):
                        why = "scanner-table preview (90/290) — no TradingView chart label, blocked"
                    elif etime < L.CHART_MIN_TIME:
                        why = f"signal before {L.CHART_MIN_TIME} chart window (not on TradingView)"
                    elif side == "SELL":
                        why = "SELL-side master signal — M9 is BUY-only (logged for evidence)"
                    elif L.sym_has_open(st, sym):
                        why = "open position already on stock (1-open-trade rule)"
                    elif not (entry > MIN_PRICE):
                        why = f"signal close ₹{entry:,.2f} ≤ ₹{MIN_PRICE:.0f} — price floor (rule 4)"
                    elif rng > MAX_RANGE:
                        why = f"entry-candle range {rng * 100:.2f}% > 1.25% — volatility brake (rule 7)"
                    elif not gainers:
                        why = "NSE top-gainers feed OFFLINE — strict mode: no M9 entries this cycle"
                    elif sym not in gainers:
                        why = "not on NSE top-20 gainers (all-securities) snapshot — entry blocked (user rule A)"
                    if why:
                        print(f"  {tag} {sym} {side} {name} @ {etime} — SKIPPED: {why}")
                        skip_log(st, sym, side, name, etime, entry, why)
                        skipped_now += 1
                        continue
                    tr = evaluate_m9(sym, etime, entry, name, tbars,
                                     warmup=trader.load_warmup(HIST9 / f"{sym}.csv", today))
                    if "error" in tr:
                        print(f"  {tag} {sym} BUY @ {etime} — sim rejected: {tr.get('error')}")
                        continue
                    tkey, k = sym, 2
                    while tkey in st["trades"]:
                        tkey = f"{sym}#{k}"; k += 1
                    st["trades"][tkey] = tr
                    st["alerts"].append(f"{tkey}:ENTRY")
                    save_state(shard, st)     # persist alert registry instantly (no-repeat guarantee)
                    tg.send_message(fmt_m9_alert(shard, tr, "ENTRY"))
                    entries_now += 1
                    print(f"  >>> {tag} ENTRY (alert) {tkey} BUY @ {entry} qty {tr['qty']} · "
                          f"SL {tr['sl']} (EMA5 trail / 15:20 EOD)")
            except Exception as e:
                print(f"  signals {sym}: {type(e).__name__}: {e}")

    # --- EOD at/after 15:25 — but only once the scan backlog is complete,
    #     so a slow day never truncates the ledger (queued cycles keep retrying)
    if hhmm >= "15:25":
        backlog = []
        errset = set(st.get("scan_err", {}))
        for sym, tbars in bars_map.items():
            if sym not in hist_map or sym in errset:
                continue
            if int(st["signals"].get(sym, {}).get("nbars", 0)) < len(tbars):
                backlog.append(sym)
        if backlog:
            print(f"  {tag}: EOD deferred — {len(backlog)} symbols still unscanned "
                  f"(next queued cycle retries)")
        else:
            try:
                done = [t for t in st["trades"].values() if "symbol" in t]
                dlbl = now.strftime("%d-%b-%Y") + f" (M9-{shard}: cash data farm)"
                sk = {}
                for it in st.get("skipped", []):
                    sk.setdefault(it["why"], []).append(
                        [it["symbol"], it["side"], it["signal"], it["time"], it["entry"]])
                out = report.build(done, dlbl, st["gate"],
                                   str(L.ROOT / f"paper_test_M9{shard}_{today}.xlsx"),
                                   skipped=sk or None, rules_note=M9_RULES)
                learn_log.harvest(tag, today, st, None, bars_map, extra={"m9_shard": shard})
                msg = _eod_text(shard, done, dlbl, st)
                tg.send_message(msg)
                tg.send_document(out, caption=f"🅼9-{shard} 📄 M9 cash data-farm report {today}")
                st["eod_done"] = True
                save_state(shard, st)         # so a crashed run never re-sends the EOD
            except Exception as e:
                print(f"  {tag} EOD report: {type(e).__name__}: {e}")

    # --- per-cycle silent heartbeat
    if "09:20" <= hhmm < "15:26":
        tg.send_message(f"💓 🅼9-{shard} {hhmm} IST · {len(st['trades'])} trades · "
                        f"{len(bars_map)} fed · cash data-farm (silent)", silent=True)

    st["cycles"] += 1
    save_state(shard, st)
    print(f"{tag} cycle done: {len(st['trades'])} trades · {len(bars_map)} fed · "
          f"entries+{entries_now} · skips+{skipped_now} · no-hist {len(no_hist)}")
    return True


def _eod_text(shard, done, dlbl, st):
    """Compact EOD (up to ~120 trades/day/shard — full per-trade list would
    blow Telegram's 4096-char limit; complete detail lives in the xlsx)."""
    c = {id(t): costs.trade_costs(t) for t in done}
    gross = sum(t["pnl"] for t in done)
    drag = sum(c[id(t)]["drag"] for t in done)
    net = gross - drag
    wins = sum(1 for t in done if c[id(t)]["net"] > 0)
    sls = sum(1 for t in done if t["exit_text"].startswith("100% SL"))
    ema = sum(1 for t in done if t["exit_text"].startswith("100% EMA5"))
    eod = sum(1 for t in done if t["exit_text"].startswith("100% EOD"))
    lines = [f"🅼9-{shard} EOD · {dlbl}",
             f"Trades {len(done)} · net wins {wins} ({wins * 100 // max(1, len(done))}%)",
             f"Gross ₹{gross:+,.0f} · costs+slip −₹{drag:,.0f} · <b>NET ₹{net:+,.0f}</b>",
             f"Exits: SL {sls} · EMA5 {ema} · EOD {eod} — skipped signals {len(st.get('skipped', []))}",
             "(cash data-farm × top-20 gainers gate — evidence, NOT a profit model)"]
    nets = sorted(((c[id(t)]["net"], t) for t in done), key=lambda x: -x[0])
    for n, t in nets[:3]:
        lines.append(f"🟢 {t['symbol']} {t['signal']} net ₹{n:+,.0f}")
    for n, t in nets[-3:]:
        lines.append(f"🔴 {t['symbol']} {t['signal']} net ₹{n:+,.0f}")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", choices=["A", "B", "C"], required=True)
    ap.add_argument("--loop", type=int, default=1)
    ap.add_argument("--sleep", type=int, default=240)
    a = ap.parse_args()
    for i in range(max(1, a.loop)):
        active = mode_live(a.shard)
        if not active:
            break
        if i < a.loop - 1:
            print(f"--- M9-{a.shard} loop: cycle {i + 2} of {a.loop} in ~{a.sleep}s ---")
            time.sleep(a.sleep)
