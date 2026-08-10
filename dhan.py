"""
Minimal DhanHQ v2 client + primary yfinance data provider:
instrument master, daily/intraday candles, bulk quotes.

Data provider strategy:
    Primary:  yfinance / Yahoo Finance (free, high-throughput, no token needed)
    Fallback: DhanHQ API (switches immediately if yfinance fails / missing data)

Rate limits enforced for Dhan (per Dhan docs):
    Data APIs  (charts/*)      5 req/sec, 100k/day
    Quote APIs (marketfeed/*)  1 req/sec, up to 1000 instruments per request
    Non-trading APIs          20 req/sec

Timestamps: Dhan returns a custom epoch counted from 1980-01-01 IST. We convert
to tz-aware Asia/Kolkata datetimes on the way in, so everything downstream is
plain IST.
"""

from __future__ import annotations

import io
import json
import logging
import os
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import pandas as pd
import requests

log = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")
BASE = "https://api.dhan.co/v2"
SCRIP_MASTER = "https://images.dhan.co/api-data/api-scrip-master.csv"
YAHOO_CHART_BASE = "https://query1.finance.yahoo.com/v8/finance/chart"

# Dhan's epoch base: seconds since 1980-01-01 00:00:00 IST.
DHAN_EPOCH_OFFSET = int(datetime(1980, 1, 1, tzinfo=IST).timestamp())


class DhanError(RuntimeError):
    """Raised when the Dhan API returns a non-recoverable error."""


class NoDataError(DhanError):
    """
    The API worked but has no candles for the requested window.

    Almost always a stock that listed after fromDate. A subclass of DhanError
    so existing handlers still catch it, but callers can treat it as a skip
    rather than a failure.
    """


class RateLimiter:
    """Thread-safe minimum-interval limiter."""

    def __init__(self, per_second: float):
        self._min_interval = 1.0 / per_second if per_second > 0 else 0.0
        self._lock = threading.Lock()
        self._next_at = 0.0

    def acquire(self) -> None:
        if self._min_interval <= 0:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                wait = self._next_at - now
                if wait <= 0:
                    self._next_at = max(now, self._next_at) + self._min_interval
                    return
            # Sleep OUTSIDE the lock so a global pause applies to every thread
            # instead of serialising them behind one holder.
            time.sleep(wait)

    def pause(self, seconds: float) -> None:
        """Hold back ALL threads after a 429, not just the one that hit it."""
        with self._lock:
            self._next_at = max(self._next_at, time.monotonic() + seconds)


def to_ist(epoch: float) -> datetime:
    """Convert a Dhan chart timestamp (1980-based epoch) to tz-aware IST."""
    return datetime.fromtimestamp(float(epoch) + DHAN_EPOCH_OFFSET, tz=IST)


def to_ist_auto(epoch: float) -> datetime:
    """
    Convert a Dhan chart timestamp, auto-detecting the epoch convention.

    Dhan's docs describe a custom epoch counted from 1980-01-01 IST, and their
    daily-data sample matches that. But the intraday endpoint (and some API
    revisions) return a PLAIN Unix epoch. Applying the +10-year offset to a
    plain Unix timestamp dates every candle ~10 years in the future, which
    silently emptied the "closed weeks" filter and made every symbol look like
    it had insufficient history.

    Rule: decode both ways and keep the one that is not in the future. A real
    candle can never be dated after today.
    """
    e = float(epoch)
    now = datetime.now(tz=IST)
    shifted = datetime.fromtimestamp(e + DHAN_EPOCH_OFFSET, tz=IST)
    if shifted <= now + timedelta(days=1):
        return shifted
    # The offset pushed it into the future -> the value was already plain Unix.
    return datetime.fromtimestamp(e, tz=IST)


@dataclass
class Instrument:
    security_id: str
    symbol: str
    name: str
    exchange_segment: str
    series: str
    instrument_type: str

    @property
    def key(self) -> str:
        return f"{self.exchange_segment}:{self.symbol}"


# Global symbol mapping cache: (segment, security_id) -> symbol and symbol -> (security_id, segment)
_SYMBOL_MAP_LOCK = threading.Lock()
_SEC_TO_SYM: dict[tuple[str, str], str] = {}
_SYM_TO_SEC: dict[str, tuple[str, str]] = {}


def _record_instruments(instruments: Iterable[Instrument]) -> None:
    with _SYMBOL_MAP_LOCK:
        for ins in instruments:
            _SEC_TO_SYM[(ins.exchange_segment, str(ins.security_id))] = ins.symbol
            _SEC_TO_SYM[("", str(ins.security_id))] = ins.symbol
            _SYM_TO_SEC[ins.symbol.upper()] = (ins.security_id, ins.exchange_segment)


def _ensure_symbol_map() -> None:
    if _SEC_TO_SYM:
        return
    # Attempt to load from universe.csv or weekly_snapshot.csv in workspace
    for p in (Path("universe.csv"), Path("weekly_snapshot.csv"), Path(__file__).parent / "universe.csv"):
        if p.exists():
            try:
                df = pd.read_csv(p, dtype=str)
                if "security_id" in df.columns and "symbol" in df.columns:
                    with _SYMBOL_MAP_LOCK:
                        for _, row in df.iterrows():
                            sid = str(row["security_id"]).strip()
                            sym = str(row["symbol"]).strip()
                            seg = str(row.get("exchange_segment", "NSE_EQ")).strip()
                            _SEC_TO_SYM[(seg, sid)] = sym
                            _SEC_TO_SYM[("", sid)] = sym
                            _SYM_TO_SEC[sym.upper()] = (sid, seg)
                    if _SEC_TO_SYM:
                        return
            except Exception:
                pass


def yahoo_ticker(symbol: str, exchange_segment: str = "NSE_EQ") -> str:
    """Format symbol as a Yahoo Finance ticker (e.g. RELIANCE.NS, TCS.NS, SBIN.BO)."""
    sym = symbol.strip()
    if sym.endswith(".NS") or sym.endswith(".BO"):
        return sym
    if "BSE" in exchange_segment.upper():
        return f"{sym}.BO"
    return f"{sym}.NS"


class DhanClient:
    def __init__(self, client_id: str = "", access_token: str = "",
                 data_rate: float = 5.0, quote_rate: float = 1.0,
                 timeout: int = 30, max_retries: int = 5,
                 primary: str = "yfinance"):
        self.client_id = client_id or ""
        self.access_token = access_token or ""
        self.timeout = timeout
        self.max_retries = max_retries
        self.primary = primary
        self._data_limiter = RateLimiter(data_rate)
        self._quote_limiter = RateLimiter(quote_rate)
        self._session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=100, pool_maxsize=100, max_retries=3)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)
        if self.access_token:
            self._session.headers.update({
                "Accept": "application/json",
                "Content-Type": "application/json",
                "access-token": self.access_token,
                "client-id": self.client_id,
            })
        self._yahoo_session = requests.Session()
        self._yahoo_session.mount("https://", adapter)
        self._yahoo_session.mount("http://", adapter)
        self._yahoo_session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json",
        })
        _ensure_symbol_map()

    def _resolve_symbol(self, security_id: str, exchange_segment: str) -> str | None:
        sid_str = str(security_id).strip()
        # If security_id is not all digits, it might be the symbol itself (e.g. in test or Yahoo mode)
        if not sid_str.isdigit():
            return sid_str
        _ensure_symbol_map()
        sym = _SEC_TO_SYM.get((exchange_segment, sid_str)) or _SEC_TO_SYM.get(("", sid_str))
        return sym

    # ------------------------------------------------------------------ core
    def _post(self, path: str, payload: dict, limiter: RateLimiter | None = None) -> Any:
        token = getattr(self, "access_token", None)
        if token == "":
            raise DhanError("DHAN_ACCESS_TOKEN is empty")
        url = f"{BASE}{path}"
        last_exc: Exception | None = None
        max_retries = getattr(self, "max_retries", 5)
        timeout = getattr(self, "timeout", 30)
        session = getattr(self, "_session", requests.Session())
        for attempt in range(max_retries):
            if limiter:
                limiter.acquire()
            try:
                r = session.post(url, json=payload, timeout=timeout)
            except requests.RequestException as exc:
                last_exc = exc
                time.sleep(1.5 * (attempt + 1))
                continue

            if r.status_code == 429:                     # rate limited
                # Exponential backoff with a global pause. A plain per-thread
                # sleep is not enough: the other worker threads keep hammering
                # the same bucket, so the retry lands on another 429.
                wait = min(2.0 * (2 ** attempt), 30.0)
                if limiter:
                    limiter.pause(wait)
                time.sleep(wait)
                last_exc = DhanError("429 rate limited")
                continue
            if r.status_code in (401, 403):
                raise DhanError(
                    f"auth failed ({r.status_code}) - check DHAN_ACCESS_TOKEN "
                    f"and that the Data API subscription is active: {r.text[:200]}"
                )
            if r.status_code >= 500:
                last_exc = DhanError(f"server error {r.status_code}")
                time.sleep(1.5 * (attempt + 1))
                continue
            if r.status_code != 200:
                # DH-905 on a chart request means "no data for this window" -
                # normal for a stock listed after fromDate. Flag it so callers
                # can skip quietly instead of logging it as a failure.
                if r.status_code == 400 and "DH-905" in r.text:
                    raise NoDataError(f"{path}: no data in range (DH-905)")
                raise DhanError(f"{path} -> HTTP {r.status_code}: {r.text[:300]}")

            try:
                body = r.json()
            except ValueError as exc:
                raise DhanError(f"{path} -> non-JSON response") from exc

            if isinstance(body, dict) and body.get("status") == "failure":
                raise DhanError(f"{path} -> {body}")
            return body

        raise DhanError(f"{path} failed after {self.max_retries} attempts: {last_exc}")

    # ------------------------------------------------------- instrument list
    @staticmethod
    def fetch_instruments(exchange_segments: Iterable[str],
                          series: Iterable[str],
                          exclude_etf: bool = True) -> list[Instrument]:
        """Download and filter the compact scrip master (public, no auth)."""
        seg_map = {"NSE_EQ": ("NSE", "E"), "BSE_EQ": ("BSE", "E")}
        wanted = {s.upper() for s in series}
        segs = [s.upper() for s in exchange_segments]

        df = None
        try:
            r = requests.get(SCRIP_MASTER, timeout=180)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text), low_memory=False)
        except Exception as exc:
            log.warning("could not download live scrip master (%s) - checking local universe.csv", exc)
            for p in (Path("universe.csv"), Path(__file__).parent / "universe.csv"):
                if p.exists():
                    df_loc = pd.read_csv(p, dtype=str)
                    out_loc: list[Instrument] = []
                    for _, row in df_loc.iterrows():
                        out_loc.append(Instrument(
                            security_id=str(row["security_id"]),
                            symbol=str(row["symbol"]).strip(),
                            name=str(row.get("name", "")).strip(),
                            exchange_segment=str(row.get("exchange_segment", "NSE_EQ")).strip(),
                            series=str(row.get("series", "EQ")).strip(),
                            instrument_type=str(row.get("type", "ES")).strip(),
                        ))
                    _record_instruments(out_loc)
                    return out_loc
            raise DhanError(f"failed to fetch instruments and no local universe found: {exc}")

        out: list[Instrument] = []
        for seg in segs:
            if seg not in seg_map:
                log.warning("unsupported exchange segment %s - skipped", seg)
                continue
            exch, segment_code = seg_map[seg]
            sub = df[(df["SEM_EXM_EXCH_ID"] == exch)
                     & (df["SEM_SEGMENT"] == segment_code)
                     & (df["SEM_INSTRUMENT_NAME"] == "EQUITY")]
            if wanted:
                sub = sub[sub["SEM_SERIES"].astype(str).str.upper().isin(wanted)]
            if exclude_etf:
                sub = sub[sub["SEM_EXCH_INSTRUMENT_TYPE"].astype(str).str.upper() != "ETF"]
            # Exchange test instruments (011NSETEST, G1NSETEST, ...) are live in
            # the scrip master but have no real price history. They also sort
            # first alphabetically, so a `--limit N` trial run would spend all
            # its calls on dummies.
            sym_u = sub["SEM_TRADING_SYMBOL"].astype(str).str.upper()
            sub = sub[~sym_u.str.contains("NSETEST|BSETEST", regex=True, na=False)]
            for _, row in sub.iterrows():
                out.append(Instrument(
                    security_id=str(int(row["SEM_SMST_SECURITY_ID"])),
                    symbol=str(row["SEM_TRADING_SYMBOL"]).strip(),
                    name=str(row.get("SM_SYMBOL_NAME", "")).strip(),
                    exchange_segment=seg,
                    series=str(row.get("SEM_SERIES", "")).strip(),
                    instrument_type=str(row.get("SEM_EXCH_INSTRUMENT_TYPE", "")).strip(),
                ))
        # de-duplicate on (segment, security id)
        seen, uniq = set(), []
        for ins in out:
            k = (ins.exchange_segment, ins.security_id)
            if k not in seen:
                seen.add(k)
                uniq.append(ins)
        _record_instruments(uniq)
        return uniq

    # -------------------------------------------------------------- candles
    def _fetch_yahoo_daily(self, sym: str, exchange_segment: str,
                           from_date: date, to_date: date) -> pd.DataFrame:
        """Fetch daily history from Yahoo Finance."""
        ticker = yahoo_ticker(sym, exchange_segment)
        days = (to_date - from_date).days
        years = max(1, int(days / 365.25) + 1)
        url = f"{YAHOO_CHART_BASE}/{urllib.parse.quote(ticker)}?range={years}y&interval=1d"
        try:
            r = self._yahoo_session.get(url, timeout=min(self.timeout, 12))
            if r.status_code != 200:
                return pd.DataFrame()
            body = r.json()
            res = (body.get("chart") or {}).get("result")
            if not res:
                return pd.DataFrame()
            res0 = res[0]
            ts = res0.get("timestamp") or []
            q = (res0.get("indicators") or {}).get("quote", [{}])[0]
            if not ts or not q:
                return pd.DataFrame()
            rows = []
            for i, t in enumerate(ts):
                c = (q.get("close") or [None])[i]
                if c is None:
                    continue
                dt = datetime.fromtimestamp(t, tz=IST)
                if from_date <= dt.date() <= to_date:
                    rows.append({
                        "datetime": dt,
                        "open": float(q["open"][i] if q.get("open") and q["open"][i] is not None else c),
                        "high": float(q["high"][i] if q.get("high") and q["high"][i] is not None else c),
                        "low": float(q["low"][i] if q.get("low") and q["low"][i] is not None else c),
                        "close": float(c),
                        "volume": float((q.get("volume") or [0])[i] or 0.0),
                        "timestamp": int(t),
                    })
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows)
            return df.sort_values("datetime").reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def _fetch_yahoo_intraday(self, sym: str, exchange_segment: str,
                             from_dt: datetime, to_dt: datetime,
                             interval: int = 5) -> pd.DataFrame:
        """Fetch intraday history from Yahoo Finance."""
        ticker = yahoo_ticker(sym, exchange_segment)
        days = max(1, (to_dt.date() - from_dt.date()).days + 2)
        range_str = f"{min(max(days, 5), 59)}d"
        url = f"{YAHOO_CHART_BASE}/{urllib.parse.quote(ticker)}?range={range_str}&interval={interval}m"
        try:
            r = self._yahoo_session.get(url, timeout=min(self.timeout, 12))
            if r.status_code != 200:
                return pd.DataFrame()
            body = r.json()
            res = (body.get("chart") or {}).get("result")
            if not res:
                return pd.DataFrame()
            res0 = res[0]
            ts = res0.get("timestamp") or []
            q = (res0.get("indicators") or {}).get("quote", [{}])[0]
            if not ts or not q:
                return pd.DataFrame()
            rows = []
            for i, t in enumerate(ts):
                c = (q.get("close") or [None])[i]
                if c is None:
                    continue
                dt = datetime.fromtimestamp(t, tz=IST)
                if from_dt <= dt <= to_dt:
                    rows.append({
                        "datetime": dt,
                        "open": float(q["open"][i] if q.get("open") and q["open"][i] is not None else c),
                        "high": float(q["high"][i] if q.get("high") and q["high"][i] is not None else c),
                        "low": float(q["low"][i] if q.get("low") and q["low"][i] is not None else c),
                        "close": float(c),
                        "volume": float((q.get("volume") or [0])[i] or 0.0),
                        "timestamp": int(t),
                    })
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows)
            return df.sort_values("datetime").reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def daily_candles(self, security_id: str, exchange_segment: str,
                      from_date: date, to_date: date,
                      chunk_days: int = 0,
                      symbol: str | None = None) -> pd.DataFrame:
        """
        Daily candles for the whole window.

        Primary: yfinance / Yahoo Finance.
        Fallback: DhanHQ historical candles API (directly switches without waiting if yfinance fails).
        """
        # Attempt primary (yfinance) if enabled
        primary_mode = getattr(self, "primary", "yfinance")
        if primary_mode == "yfinance":
            sym = symbol or self._resolve_symbol(security_id, exchange_segment)
            if sym:
                try:
                    df = self._fetch_yahoo_daily(sym, exchange_segment, from_date, to_date)
                    if not df.empty:
                        return df
                except Exception:
                    pass

        # Fallback directly to Dhan
        windows: list[tuple[date, date]] = []
        if chunk_days and chunk_days > 0:
            cursor = from_date
            while cursor < to_date:
                end = min(cursor + timedelta(days=chunk_days), to_date)
                windows.append((cursor, end))
                cursor = end + timedelta(days=1)
        else:
            windows.append((from_date, to_date))

        frames, failures = [], 0
        data_limiter = getattr(self, "_data_limiter", None)
        for start, end in windows:
            payload = {
                "securityId": str(security_id),
                "exchangeSegment": exchange_segment,
                "instrument": "EQUITY",
                "expiryCode": 0,
                "oi": False,
                "fromDate": start.strftime("%Y-%m-%d"),
                "toDate": end.strftime("%Y-%m-%d"),
            }
            try:
                part = self._to_frame(
                    self._post("/charts/historical", payload, data_limiter))
            except (NoDataError, DhanError):
                # Common and harmless when the window predates the listing.
                failures += 1
                if len(windows) == 1:
                    raise
                continue
            if not part.empty:
                frames.append(part)

        if not frames:
            if failures and len(windows) > 1:
                raise DhanError(f"all {failures} chunks failed for {security_id}")
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume", "timestamp", "datetime"])
        out = pd.concat(frames, ignore_index=True)
        return (out.drop_duplicates(subset=["datetime"])
                   .sort_values("datetime")
                   .reset_index(drop=True))

    def intraday_candles(self, security_id: str, exchange_segment: str,
                         from_dt: datetime, to_dt: datetime,
                         interval: int = 5,
                         symbol: str | None = None) -> pd.DataFrame:
        """
        Intraday candles for the window.

        Primary: yfinance / Yahoo Finance.
        Fallback: DhanHQ intraday chart API.
        """
        primary_mode = getattr(self, "primary", "yfinance")
        if primary_mode == "yfinance":
            sym = symbol or self._resolve_symbol(security_id, exchange_segment)
            if sym:
                try:
                    df = self._fetch_yahoo_intraday(sym, exchange_segment, from_dt, to_dt, interval=interval)
                    if not df.empty:
                        return df
                except Exception:
                    pass

        # Fallback to Dhan
        payload = {
            "securityId": str(security_id),
            "exchangeSegment": exchange_segment,
            "instrument": "EQUITY",
            "interval": str(interval),
            "oi": False,
            "fromDate": from_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "toDate": to_dt.strftime("%Y-%m-%d %H:%M:%S"),
        }
        data_limiter = getattr(self, "_data_limiter", None)
        return self._to_frame(self._post("/charts/intraday", payload, data_limiter))

    @staticmethod
    def _to_frame(body: dict) -> pd.DataFrame:
        cols = ("open", "high", "low", "close", "volume", "timestamp")
        if not isinstance(body, dict):
            return pd.DataFrame(columns=list(cols))
        # Dhan returns the OHLC arrays at the top level on some endpoints and
        # wrapped in {"data": {...}} on others (and the wrapper appeared in
        # later API revisions). Unwrap it, otherwise every symbol looks empty
        # and gets silently counted as "skipped".
        if "timestamp" not in body and isinstance(body.get("data"), dict):
            body = body["data"]
        if "timestamp" not in body:
            return pd.DataFrame(columns=list(cols))
        n = len(body.get("timestamp") or [])
        if n == 0:
            return pd.DataFrame(columns=list(cols))
        data = {c: list(body.get(c) or [])[:n] for c in cols}
        df = pd.DataFrame(data)
        df["datetime"] = [to_ist_auto(t) for t in df["timestamp"]]
        for c in ("open", "high", "low", "close", "volume"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.dropna(subset=["close"]).sort_values("datetime").reset_index(drop=True)

    # --------------------------------------------------------------- quotes
    def _fetch_yahoo_quote(self, sym: str, exchange_segment: str) -> dict | None:
        """Fetch quote (OHLC + LTP + volume) from Yahoo for one symbol."""
        ticker = yahoo_ticker(sym, exchange_segment)
        url = f"{YAHOO_CHART_BASE}/{urllib.parse.quote(ticker)}?range=1d&interval=5m"
        try:
            r = self._yahoo_session.get(url, timeout=6)
            if r.status_code != 200:
                return None
            data = r.json().get("chart", {}).get("result")
            if not data:
                return None
            res0 = data[0]
            meta = res0.get("meta", {})
            q = (res0.get("indicators") or {}).get("quote", [{}])[0]
            closes = [c for c in (q.get("close") or []) if c is not None]
            highs = [h for h in (q.get("high") or []) if h is not None]
            lows = [l for l in (q.get("low") or []) if l is not None]
            opens = [o for o in (q.get("open") or []) if o is not None]
            vols = [v for v in (q.get("volume") or []) if v is not None]

            px = meta.get("regularMarketPrice") or (closes[-1] if closes else 0.0)
            op = meta.get("regularMarketDayOpen") or (opens[0] if opens else px)
            hi = meta.get("regularMarketDayHigh") or (max(highs) if highs else px)
            lo = meta.get("regularMarketDayLow") or (min(lows) if lows else px)
            pc = meta.get("chartPreviousClose") or meta.get("previousClose") or px
            vol = meta.get("regularMarketVolume") or (sum(vols) if vols else 0.0)

            return {
                "last_price": float(px or 0.0),
                "open": float(op or 0.0),
                "high": float(hi or 0.0),
                "low": float(lo or 0.0),
                "prev_close": float(pc or 0.0),
                "volume": float(vol or 0.0),
            }
        except Exception:
            return None

    def ltp(self, securities: dict[str, list[int]]) -> dict[str, dict[str, float]]:
        """
        Bulk LTP. `securities` maps segment -> list of security ids (<=1000 total).
        Primary: yfinance / Yahoo Finance quotes.
        Fallback: DhanHQ /marketfeed/ltp for any missing symbols.
        """
        primary_mode = getattr(self, "primary", "yfinance")
        out: dict[str, dict[str, float]] = {}
        missing: dict[str, list[int]] = {}

        if primary_mode == "yfinance":
            tasks = []
            for seg, ids in securities.items():
                for sid in ids:
                    sym = self._resolve_symbol(str(sid), seg)
                    if sym:
                        tasks.append((seg, str(sid), sym))
                    else:
                        missing.setdefault(seg, []).append(sid)

            if tasks:
                def worker(t):
                    s_seg, s_sid, s_sym = t
                    q = self._fetch_yahoo_quote(s_sym, s_seg)
                    return s_seg, s_sid, (q.get("last_price") if q else None)

                with ThreadPoolExecutor(max_workers=min(len(tasks), 20)) as pool:
                    for s_seg, s_sid, price in pool.map(worker, tasks):
                        if price and price > 0:
                            out.setdefault(s_seg, {})[s_sid] = float(price)
                        else:
                            missing.setdefault(s_seg, []).append(int(s_sid))

            if not missing:
                return out

        # Fallback to Dhan for missing items
        target = missing if primary_mode == "yfinance" else securities
        if target and getattr(self, "access_token", None):
            quote_limiter = getattr(self, "_quote_limiter", None)
            body = self._post("/marketfeed/ltp", target, quote_limiter)
            data = (body or {}).get("data", {}) if isinstance(body, dict) else {}
            for seg, items in data.items():
                for sid, payload in (items or {}).items():
                    out.setdefault(seg, {})[str(sid)] = float(payload.get("last_price") or 0.0)
        return out

    def ohlc(self, securities: dict[str, list[int]]) -> dict[str, dict[str, dict]]:
        """
        Bulk OHLC + LTP for the current day.
        Primary: yfinance / Yahoo Finance quotes.
        Fallback: DhanHQ /marketfeed/ohlc.
        """
        primary_mode = getattr(self, "primary", "yfinance")
        out: dict[str, dict[str, dict]] = {}
        missing: dict[str, list[int]] = {}

        if primary_mode == "yfinance":
            tasks = []
            for seg, ids in securities.items():
                for sid in ids:
                    sym = self._resolve_symbol(str(sid), seg)
                    if sym:
                        tasks.append((seg, str(sid), sym))
                    else:
                        missing.setdefault(seg, []).append(sid)

            if tasks:
                def worker(t):
                    s_seg, s_sid, s_sym = t
                    q = self._fetch_yahoo_quote(s_sym, s_seg)
                    return s_seg, s_sid, q

                with ThreadPoolExecutor(max_workers=min(len(tasks), 20)) as pool:
                    for s_seg, s_sid, q in pool.map(worker, tasks):
                        if q and q.get("last_price"):
                            out.setdefault(s_seg, {})[s_sid] = {
                                "last_price": float(q.get("last_price") or 0.0),
                                "open": float(q.get("open") or 0.0),
                                "high": float(q.get("high") or 0.0),
                                "low": float(q.get("low") or 0.0),
                                "prev_close": float(q.get("prev_close") or 0.0),
                                "volume": float(q.get("volume") or 0.0),
                            }
                        else:
                            missing.setdefault(s_seg, []).append(int(s_sid))

            if not missing:
                return out

        # Fallback to Dhan for missing items
        target = missing if primary_mode == "yfinance" else securities
        if target and getattr(self, "access_token", None):
            quote_limiter = getattr(self, "_quote_limiter", None)
            body = self._post("/marketfeed/ohlc", target, quote_limiter)
            data = (body or {}).get("data", {}) if isinstance(body, dict) else {}
            for seg, items in data.items():
                for sid, payload in (items or {}).items():
                    o = payload.get("ohlc") or {}
                    out.setdefault(seg, {})[str(sid)] = {
                        "last_price": float(payload.get("last_price") or 0.0),
                        "open": float(o.get("open") or 0.0),
                        "high": float(o.get("high") or 0.0),
                        "low": float(o.get("low") or 0.0),
                        "prev_close": float(o.get("close") or 0.0),
                        "volume": float(payload.get("volume")
                                        or payload.get("last_quantity") or 0.0),
                    }
        return out


def last_n_years(years: int) -> tuple[date, date]:
    """
    Date window for the daily-history call.

    toDate is TODAY, never tomorrow: some Dhan endpoints reject or return an
    empty payload for a future toDate, which silently produced "no data" for
    every symbol.
    """
    today = datetime.now(IST).date()
    return today - timedelta(days=int(365.25 * years) + 10), today
