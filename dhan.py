"""
Minimal DhanHQ v2 client: instrument master, daily/intraday candles, bulk quotes.

Rate limits enforced (per Dhan docs):
    Data APIs  (charts/*)      5 req/sec, 100k/day
    Quote APIs (marketfeed/*)  1 req/sec, up to 1000 instruments per request
    Non-trading APIs          20 req/sec

Timestamps: Dhan returns a custom epoch counted from 1980-01-01 IST. We convert
to tz-aware Asia/Kolkata datetimes on the way in, so everything downstream is
plain IST.
"""

from __future__ import annotations

import io
import logging
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import pandas as pd
import requests

log = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")
BASE = "https://api.dhan.co/v2"
SCRIP_MASTER = "https://images.dhan.co/api-data/api-scrip-master.csv"

# Dhan's epoch base: seconds since 1980-01-01 00:00:00 IST.
DHAN_EPOCH_OFFSET = int(datetime(1980, 1, 1, tzinfo=IST).timestamp())


class DhanError(RuntimeError):
    """Raised when the Dhan API returns a non-recoverable error."""


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
    """Convert a Dhan chart timestamp to a tz-aware IST datetime."""
    return datetime.fromtimestamp(float(epoch) + DHAN_EPOCH_OFFSET, tz=IST)


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


class DhanClient:
    def __init__(self, client_id: str, access_token: str,
                 data_rate: float = 5.0, quote_rate: float = 1.0,
                 timeout: int = 30, max_retries: int = 3):
        if not access_token:
            raise DhanError("DHAN_ACCESS_TOKEN is empty")
        self.client_id = client_id
        self.timeout = timeout
        self.max_retries = max_retries
        self._data_limiter = RateLimiter(data_rate)
        self._quote_limiter = RateLimiter(quote_rate)
        self._session = requests.Session()
        self._session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json",
            "access-token": access_token,
            "client-id": client_id,
        })

    # ------------------------------------------------------------------ core
    def _post(self, path: str, payload: dict, limiter: RateLimiter) -> Any:
        url = f"{BASE}{path}"
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            limiter.acquire()
            try:
                r = self._session.post(url, json=payload, timeout=self.timeout)
            except requests.RequestException as exc:
                last_exc = exc
                time.sleep(1.5 * (attempt + 1))
                continue

            if r.status_code == 429:                     # rate limited
                # Exponential backoff with a global pause. A plain per-thread
                # sleep is not enough: the other worker threads keep hammering
                # the same bucket, so the retry lands on another 429.
                wait = min(2.0 * (2 ** attempt), 30.0)
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

        r = requests.get(SCRIP_MASTER, timeout=180)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), low_memory=False)

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
        return uniq

    # -------------------------------------------------------------- candles
    def daily_candles(self, security_id: str, exchange_segment: str,
                      from_date: date, to_date: date) -> pd.DataFrame:
        payload = {
            "securityId": str(security_id),
            "exchangeSegment": exchange_segment,
            "instrument": "EQUITY",
            "expiryCode": 0,
            "oi": False,
            "fromDate": from_date.strftime("%Y-%m-%d"),
            "toDate": to_date.strftime("%Y-%m-%d"),
        }
        return self._to_frame(self._post("/charts/historical", payload, self._data_limiter))

    def intraday_candles(self, security_id: str, exchange_segment: str,
                         from_dt: datetime, to_dt: datetime,
                         interval: int = 5) -> pd.DataFrame:
        payload = {
            "securityId": str(security_id),
            "exchangeSegment": exchange_segment,
            "instrument": "EQUITY",
            "interval": str(interval),
            "oi": False,
            "fromDate": from_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "toDate": to_dt.strftime("%Y-%m-%d %H:%M:%S"),
        }
        return self._to_frame(self._post("/charts/intraday", payload, self._data_limiter))

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
        df["datetime"] = [to_ist(t) for t in df["timestamp"]]
        for c in ("open", "high", "low", "close", "volume"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.dropna(subset=["close"]).sort_values("datetime").reset_index(drop=True)

    # --------------------------------------------------------------- quotes
    def ltp(self, securities: dict[str, list[int]]) -> dict[str, dict[str, float]]:
        """Bulk LTP. `securities` maps segment -> list of security ids (<=1000 total)."""
        body = self._post("/marketfeed/ltp", securities, self._quote_limiter)
        data = (body or {}).get("data", {}) if isinstance(body, dict) else {}
        out: dict[str, dict[str, float]] = {}
        for seg, items in data.items():
            for sid, payload in (items or {}).items():
                out.setdefault(seg, {})[str(sid)] = float(payload.get("last_price") or 0.0)
        return out

    def ohlc(self, securities: dict[str, list[int]]) -> dict[str, dict[str, dict]]:
        """Bulk OHLC + LTP for the current day."""
        body = self._post("/marketfeed/ohlc", securities, self._quote_limiter)
        data = (body or {}).get("data", {}) if isinstance(body, dict) else {}
        out: dict[str, dict[str, dict]] = {}
        for seg, items in data.items():
            for sid, payload in (items or {}).items():
                o = payload.get("ohlc") or {}
                out.setdefault(seg, {})[str(sid)] = {
                    "last_price": float(payload.get("last_price") or 0.0),
                    "open": float(o.get("open") or 0.0),
                    "high": float(o.get("high") or 0.0),
                    "low": float(o.get("low") or 0.0),
                    "prev_close": float(o.get("close") or 0.0),
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
