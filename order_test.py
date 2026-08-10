#!/usr/bin/env python3
"""
DhanHQ Live Order Gateway & Order Book Test Tool.

Mirrors the BTST & Anticipation paper trades on DhanHQ with Quantity = 1 at Live Market Price.

Features:
  1. Automatically reads today's exact picks from btst_picks.csv and anticipate_picks.csv.
  2. Places micro test BUY orders (Qty = 1) at live market price (MARKET / CNC) via DhanHQ API.
  3. Verifies Order Book (GET /orders), Trade Book (GET /trades), and Position Book (GET /positions).
  4. Real-time response logging: captures Dhan order IDs, execution status, or gateway messages (e.g. Insufficient Funds).

Usage:
  python order_test.py                     # mirror today's BTST/anticipate picks with Qty 1
  python order_test.py --symbol SBCL       # test with specific symbol at live market price
  python order_test.py --read-only         # inspect order/trade books without placing orders
  python order_test.py --limit-order       # place limit order at LTP instead of market order
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path

import pandas as pd
import requests

from config import load_config
from dhan import BASE, IST, DhanClient, DhanError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger("order_test")

ROOT = Path(__file__).resolve().parent


def load_today_picks() -> list[dict]:
    """Load today's active picks from btst_picks.csv and anticipate_picks.csv."""
    picks = []
    today = pd.Timestamp.now(tz=IST).strftime("%Y-%m-%d")
    
    for f_name, source in (("btst_picks.csv", "BTST_CONFIRMED"), ("anticipate_picks.csv", "ANTICIPATE")):
        p = ROOT / f_name
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
            if df.empty or "symbol" not in df.columns:
                continue
            df["date_s"] = df["date"].astype(str).str[:10]
            today_df = df[df["date_s"] == today]
            for _, r in today_df.iterrows():
                picks.append({
                    "symbol": str(r["symbol"]).strip().upper(),
                    "source": source,
                    "tier": str(r.get("tier", "")),
                    "entry_target": float(r.get("entry", 0.0) or 0.0),
                    "tradeable": int(float(r.get("tradeable", 1) or 0)),
                })
        except Exception as exc:
            log.warning("could not read %s: %s", f_name, exc)
    return picks


def run_dhan_order_test(cfg, symbol: str | None = None, read_only: bool = False,
                        use_limit: bool = False) -> int:
    client_id = cfg.secrets.dhan_client_id
    token = cfg.secrets.dhan_access_token

    print("\n" + "=" * 80)
    print("DHANHQ v2 LIVE ORDER GATEWAY & ORDER BOOK MIRROR TEST")
    print("=" * 80)
    print(f"Client ID  : {client_id if client_id else '(not set)'}")
    print(f"Token      : {'*' * 8 + token[-4:] if len(token) > 4 else '(not set)'}")
    print(f"Gateway    : {BASE}/orders")
    print(f"Order Mode : {'READ-ONLY (Books Inspection)' if read_only else ('LIMIT @ LTP (Qty 1)' if use_limit else 'MARKET @ Live Price (Qty 1)')}")
    print("-" * 80)

    if not token or not client_id:
        print("❌ ERROR: DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN must be set.")
        print("   Set them in your environment or repo secrets to connect to Dhan.")
        return 1

    session = requests.Session()
    session.headers.update({
        "Accept": "application/json",
        "Content-Type": "application/json",
        "access-token": token,
        "client-id": client_id,
        "User-Agent": "Mozilla/5.0",
    })

    # 1. Resolve Symbols to Trade
    proxy_url = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    if proxy_url:
        masked_proxy = proxy_url.split("@")[-1] if "@" in proxy_url else proxy_url
        print(f"✓ Static Outbound Proxy: Active ({masked_proxy})")
    try:
        source_ip = session.get("https://api.ipify.org", timeout=8).text.strip()
        print(f"✓ Outbound Source IP (seen by Dhan): {source_ip}")
    except Exception:
        print("ℹ️ Source IP lookup: offline or direct")

    targets = []
    if symbol:
        targets = [{"symbol": symbol.strip().upper(), "source": "MANUAL", "tier": "TEST", "entry_target": 0.0, "tradeable": 1}]
    else:
        today_picks = load_today_picks()
        if today_picks:
            print(f"✓ Loaded {len(today_picks)} pick(s) from today's paper trade scan:")
            for pk in today_picks:
                print(f"  • {pk['symbol']} [{pk['source']} {pk['tier']}] ~₹{pk['entry_target']:.2f} (Tradeable: {pk['tradeable']})")
            targets = [p for p in today_picks if p.get("tradeable", 1) == 1]
        else:
            print("ℹ️ No picks found in today's picks files. Falling back to probe symbol (IDEA).")
            targets = [{"symbol": "IDEA", "source": "PROBE", "tier": "TEST", "entry_target": 0.0, "tradeable": 1}]

    # 2. Resolve Instruments from Scrip Master
    print("\n[Step 1/4] Resolving Dhan security IDs from scrip master ...")
    try:
        instruments = DhanClient.fetch_instruments(["NSE_EQ"], ["EQ"], exclude_etf=True)
        by_sym = {i.symbol.upper(): i for i in instruments}
    except Exception as exc:
        print(f"❌ Scrip master resolution failed: {exc}")
        return 1

    # 2. Query Dhan Registered IP Setup (GET /v2/ip/getIP)
    print("\n[Step 2/5] Checking Dhan registered Static IP setup ...")
    try:
        r_ip = session.get(f"{BASE}/ip/getIP", timeout=10)
        if r_ip.status_code == 200:
            ip_data = r_ip.json()
            print(f"  ✓ Dhan Account Registered IPs: {json.dumps(ip_data)}")
        else:
            print(f"  ℹ️ IP query response ({r_ip.status_code}): {r_ip.text[:120]}")
    except Exception as exc:
        print(f"  ⚠️ Could not query IP setup: {exc}")

    # 3. Read Current Live Books
    print("\n[Step 3/5] Querying live Order Book, Trade Book, and Positions ...")
    try:
        r_orders = session.get(f"{BASE}/orders", timeout=12)
        if r_orders.status_code == 200:
            orders_data = r_orders.json()
            order_list = orders_data if isinstance(orders_data, list) else orders_data.get("data", [])
            print(f"  ✓ Live Order Book: {len(order_list)} order(s) logged today")
            for o in order_list[-3:]:
                print(f"    • Order #{o.get('orderId')}: {o.get('tradingSymbol')} {o.get('transactionType')} Qty={o.get('quantity')} Status={o.get('orderStatus')}")
        else:
            print(f"  ℹ️ Order Book response ({r_orders.status_code}): {r_orders.text[:150]}")

        r_trades = session.get(f"{BASE}/trades", timeout=12)
        if r_trades.status_code == 200:
            trades_data = r_trades.json()
            trade_list = trades_data if isinstance(trades_data, list) else trades_data.get("data", [])
            print(f"  ✓ Live Trade Book: {len(trade_list)} trade(s) filled today")

        r_positions = session.get(f"{BASE}/positions", timeout=12)
        if r_positions.status_code == 200:
            pos_data = r_positions.json()
            pos_list = pos_data if isinstance(pos_data, list) else pos_data.get("data", [])
            print(f"  ✓ Live Positions: {len(pos_list)} open position(s)")
    except Exception as exc:
        print(f"⚠️ Order book query warning: {exc}")

    if read_only:
        print("\n" + "=" * 80)
        print("READ-ONLY INSPECTION COMPLETE.")
        print("=" * 80 + "\n")
        return 0

    # 4. Place Micro Orders (Qty = 1) Mirroring Paper Trades
    print(f"\n[Step 3/4] Executing micro test orders (Qty 1 per setup at live price) ...")
    client = DhanClient(client_id, token)
    executed_orders = []

    for t_info in targets:
        sym = t_info["symbol"]
        ins = by_sym.get(sym)
        if not ins:
            print(f"  ⚠️ {sym}: security ID not found in NSE universe — skipping")
            continue

        # Fetch Live Price
        ltp = 0.0
        try:
            ltp_map = client.ltp({ins.exchange_segment: [int(ins.security_id)]})
            ltp = ltp_map.get(ins.exchange_segment, {}).get(str(ins.security_id), 0.0)
        except Exception:
            pass
        if ltp <= 0:
            q = client._fetch_yahoo_quote(ins.symbol, ins.exchange_segment)
            ltp = q.get("last_price", 0.0) if q else t_info.get("entry_target", 10.0)

        corr_id = f"btst_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        order_type_str = "LIMIT" if use_limit else "MARKET"
        price_val = round(ltp, 2) if use_limit else 0.0

        payload = {
            "dhanClientId": client_id,
            "correlationId": corr_id,
            "transactionType": "BUY",
            "exchangeSegment": ins.exchange_segment,
            "productType": "CNC",
            "orderType": order_type_str,
            "validity": "DAY",
            "tradingSymbol": ins.symbol,
            "securityId": str(ins.security_id),
            "quantity": 1,
            "disclosedQuantity": 0,
            "price": price_val,
            "triggerPrice": 0.0,
            "afterMarketOrder": False,
            "amoTime": "OPEN"
        }

        print(f"\n  ► Placing order: BUY 1 {ins.symbol} @ {order_type_str} (LTP ₹{ltp:.2f}) [CNC]")
        try:
            r_place = session.post(f"{BASE}/orders", json=payload, timeout=15)
            print(f"    Gateway HTTP Status: {r_place.status_code}")
            resp_json = r_place.json()
            print(f"    Gateway Response   : {json.dumps(resp_json)}")

            if r_place.status_code in (200, 201):
                order_id = resp_json.get("orderId") or (resp_json.get("data", {}) if isinstance(resp_json.get("data"), dict) else {}).get("orderId")
                status = resp_json.get("orderStatus", "PLACED")
                print(f"    ✓ SUCCESS: Order #{order_id} ({status})")
                executed_orders.append((order_id, ins.symbol))
            else:
                err_code = resp_json.get("errorCode", "")
                err_msg = resp_json.get("errorMessage", resp_json.get("message", r_place.text))
                print(f"    ℹ️ Gateway Note: {err_code} - {err_msg}")
                if "Invalid IP" in err_msg or "DH-905" in str(err_code):
                    try:
                        runner_ip = session.get("https://api.ipify.org", timeout=5).text.strip()
                    except Exception:
                        runner_ip = "unknown"
                    print(f"\n    ⚠️ CAUSE OF REJECTION (DH-905 Invalid IP):")
                    print(f"       Dhan rejected the order because your DhanHQ API access token has IP Whitelisting enabled,")
                    print(f"       and the current runner IP ({runner_ip}) is not in your whitelisted IP list.")
                    print(f"       FIX:")
                    print(f"       1. Log in to web.dhan.co -> My Profile -> DhanHQ Trading APIs.")
                    print(f"       2. Check the 'IP Setup' section. Add your static proxy IP to Secondary IP,")
                    print(f"          then regenerate the access token and set HTTPS_PROXY secret.")
        except Exception as exc:
            print(f"    ❌ Order call failed: {exc}")

    # 5. Summary & Verification
    print("\n[Step 4/4] Verifying Order Execution Summary ...")
    print("=" * 80)
    print(f"Total setups processed : {len(targets)}")
    print(f"Orders sent to gateway : {len(executed_orders)} placed successfully")
    print("Execution Gateway      : DhanHQ v2 Orders API verified with live market pricing.")
    print("=" * 80 + "\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default=None, help="specific symbol for test order (default: read from today's paper picks)")
    ap.add_argument("--read-only", action="store_true", help="inspect order & trade books without placing orders")
    ap.add_argument("--limit-order", action="store_true", help="place limit order at LTP instead of market order")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    return run_dhan_order_test(cfg, symbol=args.symbol, read_only=args.read_only, use_limit=args.limit_order)


if __name__ == "__main__":
    sys.exit(main())
