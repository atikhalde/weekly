#!/usr/bin/env python3
"""
Market-cap table builder (Pine condition c12: "market cap > 1000 Cr").

WHY THIS FILE EXISTS
    Dhan exposes no shares-outstanding field, and Pine's `request.financial`
    has no API equivalent, so c12 was hard-disabled from day one. Every alert
    you have ever received passed c12 for free - which is how sub-1000 Cr
    names like GANESHBE (872 Cr) reached your phone.

WHERE THE NUMBER COMES FROM
    market cap = shares outstanding x last price

    Shares outstanding come from Yahoo's fundamentals-timeseries endpoint
    (the most recent quarterly, falling back to annual, BasicAverageShares).
    The price is the current close. Validated against the c12 value printed
    on nine of your own TradingView screenshots:

        symbol       computed     chart      error
        DIVISLAB      206505     205937      +0.3%
        LALPATHLAB     31437      31455      -0.1%
        GPTHEALTH       1368       1376      -0.6%
        MONARCH         3078       3107      -0.9%
        GANESHBE         881        872      +1.0%
        OAL             1341       1322      +1.4%
        TMB            13699      14022      -2.3%
        RADICO         58556      57255      +2.3%
        APCOTEXIND      3534       3677      -3.9%

    Within a few percent throughout. That is close enough for a 1000 Cr
    threshold, but NOT for a knife-edge call: a stock computing at 1010 Cr
    might really be 980. See `mcap_margin_pct` in config.yaml.

    Share counts are a REPORTED figure and move only on issuance, so the
    table is rebuilt weekly alongside the snapshot rather than every scan.

    python mcap.py              # rebuild mcap.csv for the whole universe
    python mcap.py --limit 50   # quick smoke test

Never places orders.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue

from config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger("mcap")

UA = {"User-Agent": "Mozilla/5.0"}
TS_URL = ("https://query1.finance.yahoo.com/ws/fundamentals-timeseries/v1/"
          "finance/timeseries/{sym}.NS?symbol={sym}.NS"
          "&type=quarterlyBasicAverageShares,annualBasicAverageShares"
          "&period1=1600000000&period2=1900000000")
PX_URL = ("https://query1.finance.yahoo.com/v8/finance/chart/"
          "{sym}.NS?range=1d&interval=1d")
CRORE = 1e7


def _get(url: str, timeout: int = 25):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        return json.load(fh)


def latest_shares(symbol: str) -> float | None:
    """Most recent reported share count, quarterly preferred over annual."""
    try:
        data = _get(TS_URL.format(sym=symbol))
    except (urllib.error.URLError, json.JSONDecodeError, OSError):
        return None
    best_date, best_val = None, None
    for block in (data.get("timeseries", {}) or {}).get("result", []) or []:
        kind = (block.get("meta", {}) or {}).get("type", [None])[0]
        if not kind:
            continue
        for entry in block.get(kind) or []:
            if not entry:
                continue
            raw = (entry.get("reportedValue") or {}).get("raw")
            as_of = entry.get("asOfDate")
            if raw and as_of and (best_date is None or as_of > best_date):
                best_date, best_val = as_of, float(raw)
    return best_val


def last_price(symbol: str) -> float | None:
    try:
        data = _get(PX_URL.format(sym=symbol))
        meta = data["chart"]["result"][0]["meta"]
        px = meta.get("regularMarketPrice") or meta.get("chartPreviousClose")
        return float(px) if px else None
    except (urllib.error.URLError, json.JSONDecodeError, KeyError,
            IndexError, TypeError, OSError):
        return None


def market_cap_cr(symbol: str) -> float | None:
    """Market cap in CRORE, or None when it cannot be determined."""
    shares = latest_shares(symbol)
    if not shares:
        return None
    price = last_price(symbol)
    if not price:
        return None
    return shares * price / CRORE


def build(symbols: list[str], workers: int = 10) -> dict[str, float]:
    out: dict[str, float] = {}
    q: Queue = Queue()
    for s in symbols:
        q.put(s)
    lock = threading.Lock()
    done = [0, 0]

    def worker():
        while True:
            try:
                sym = q.get_nowait()
            except Empty:
                return
            mc = market_cap_cr(sym)
            with lock:
                if mc and mc > 0:
                    out[sym] = round(mc, 2)
                    done[0] += 1
                else:
                    done[1] += 1
                n = done[0] + done[1]
                if n % 200 == 0:
                    log.info("  %d/%d (%d resolved)", n, len(symbols), done[0])
            time.sleep(0.05)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log.info("resolved %d, unresolved %d", done[0], done[1])
    return out


def load_table(path: str | Path) -> dict[str, float]:
    """symbol -> market cap in crore. Missing file returns {}."""
    p = Path(path)
    if not p.exists():
        return {}
    out: dict[str, float] = {}
    try:
        with p.open(newline="") as fh:
            for row in csv.DictReader(fh):
                sym = (row.get("symbol") or "").strip().upper()
                raw = (row.get("mcap_cr") or row.get("mcap") or "").strip()
                if not sym or not raw:
                    continue
                try:
                    val = float(raw)
                except ValueError:
                    continue
                if val > 0:
                    out[sym] = val
    except OSError as exc:
        log.warning("could not read %s: %s", p, exc)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="output CSV (default: mcap.csv)")
    ap.add_argument("--limit", type=int, default=0, help="cap symbols (testing)")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_path = Path(args.out) if args.out else cfg.paths["mcap"]

    uni = cfg.paths["universe"]
    if uni.exists():
        import pandas as pd
        symbols = sorted(pd.read_csv(uni, dtype=str)["symbol"].str.upper().unique())
        log.info("universe.csv: %d symbols", len(symbols))
    else:
        from dhan import DhanClient
        log.info("no universe.csv - downloading the Dhan scrip master")
        ins = DhanClient.fetch_instruments(cfg.universe.exchange_segments,
                                           cfg.universe.series,
                                           exclude_etf=cfg.universe.exclude_etf)
        symbols = sorted({i.symbol.upper() for i in ins})

    if args.limit:
        symbols = symbols[:args.limit]

    started = time.time()
    table = build(symbols, workers=args.workers)
    if not table:
        log.error("no market caps resolved - NOT overwriting %s", out_path)
        return 1

    # Never shrink the table drastically on a bad run: a partial fetch that
    # overwrites a good file would silently disqualify hundreds of stocks.
    existing = load_table(out_path)
    if existing and len(table) < 0.5 * len(existing):
        log.error("only %d rows vs %d already on disk - refusing to overwrite",
                  len(table), len(existing))
        return 1

    with Path(out_path).open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["symbol", "mcap_cr", "updated"])
        stamp = datetime.now().strftime("%Y-%m-%d")
        for sym in sorted(table):
            w.writerow([sym, f"{table[sym]:.2f}", stamp])

    above = sum(1 for v in table.values() if v > cfg.strategy.min_mcap)
    log.info("wrote %s: %d symbols in %.0fs (%d above %.0f Cr)",
             out_path, len(table), time.time() - started,
             above, cfg.strategy.min_mcap)
    return 0


if __name__ == "__main__":
    sys.exit(main())
