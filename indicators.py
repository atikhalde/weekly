"""
Pine-Script-exact technical indicators.

Every function here mirrors the behaviour of the corresponding TradingView
`ta.*` built-in, including the seeding rules, because the entry signal must
match the indicator candle-for-candle.

Seeding rules replicated
------------------------
ta.ema(src, n)  -> first value = SMA(src, n) at bar n-1, then
                   ema = alpha*src + (1-alpha)*ema[1],  alpha = 2/(n+1)
ta.rma(src, n)  -> first value = SMA(src, n) at bar n-1, then
                   rma = (rma[1]*(n-1) + src)/n            (Wilder smoothing)
ta.rsi(src, n)  -> 100 - 100/(1 + rma(gain,n)/rma(loss,n))
ta.macd(...)    -> [macd, signal, hist] with macd = ema(fast)-ema(slow),
                   signal = ema(macd, sig), hist = macd - signal
ta.sma / ta.highest -> plain rolling window

Incremental "developing bar" support
------------------------------------
The scanner has to evaluate the *developing* weekly bar hundreds of times
(once per 5-minute candle) while every closed weekly bar stays frozen.
Recomputing the whole series each time would be O(n) per 5m bar; instead each
indicator exposes a frozen state at the last CLOSED bar plus a `step()` method
that returns the value for a hypothetical next bar in O(1). The result is
numerically identical to a full recomputation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


# --------------------------------------------------------------------------- #
#  Full-series implementations (used for history / tests)
# --------------------------------------------------------------------------- #
def sma(values: Sequence[float], length: int) -> np.ndarray:
    """ta.sma - simple moving average. NaN until `length` bars exist."""
    v = np.asarray(values, dtype=float)
    out = np.full(v.shape, np.nan)
    if len(v) < length:
        return out
    csum = np.cumsum(np.insert(v, 0, 0.0))
    out[length - 1:] = (csum[length:] - csum[:-length]) / length
    return out


def ema(values: Sequence[float], length: int) -> np.ndarray:
    """ta.ema - EMA seeded with the SMA of the first `length` values."""
    v = np.asarray(values, dtype=float)
    out = np.full(v.shape, np.nan)
    if len(v) < length:
        return out
    alpha = 2.0 / (length + 1.0)
    out[length - 1] = v[:length].mean()
    for i in range(length, len(v)):
        out[i] = alpha * v[i] + (1.0 - alpha) * out[i - 1]
    return out


def rma(values: Sequence[float], length: int) -> np.ndarray:
    """ta.rma - Wilder's smoothing, seeded with the SMA of the first `length`."""
    v = np.asarray(values, dtype=float)
    out = np.full(v.shape, np.nan)
    if len(v) < length:
        return out
    out[length - 1] = v[:length].mean()
    for i in range(length, len(v)):
        out[i] = (out[i - 1] * (length - 1) + v[i]) / length
    return out


def rsi(values: Sequence[float], length: int) -> np.ndarray:
    """ta.rsi - Wilder RSI built on rma(gain)/rma(loss)."""
    v = np.asarray(values, dtype=float)
    out = np.full(v.shape, np.nan)
    if len(v) < length + 1:
        return out
    delta = np.diff(v, prepend=v[0])
    delta[0] = 0.0
    gain = np.where(delta > 0.0, delta, 0.0)
    loss = np.where(delta < 0.0, -delta, 0.0)
    # Pine computes rma over the change series starting at bar 1.
    ag = rma(gain[1:], length)
    al = rma(loss[1:], length)
    for i in range(len(ag)):
        if np.isnan(ag[i]):
            continue
        if al[i] == 0.0:
            out[i + 1] = 100.0
        elif ag[i] == 0.0:
            out[i + 1] = 0.0
        else:
            rs = ag[i] / al[i]
            out[i + 1] = 100.0 - 100.0 / (1.0 + rs)
    return out


def macd(values: Sequence[float], fast: int, slow: int, signal: int):
    """ta.macd -> (macd_line, signal_line, histogram)."""
    v = np.asarray(values, dtype=float)
    ema_fast = ema(v, fast)
    ema_slow = ema(v, slow)
    macd_line = ema_fast - ema_slow

    # Pine feeds the (partially NaN) macd line into ta.ema; the signal EMA is
    # seeded from the first `signal` non-NaN macd values.
    sig = np.full(v.shape, np.nan)
    valid = np.where(~np.isnan(macd_line))[0]
    if len(valid) >= signal:
        start = valid[0]
        seg = macd_line[start:]
        seg_sig = ema(seg, signal)
        sig[start:] = seg_sig
    return macd_line, sig, macd_line - sig


def highest(values: Sequence[float], length: int) -> np.ndarray:
    """ta.highest - rolling maximum, NaN until `length` bars exist."""
    v = np.asarray(values, dtype=float)
    out = np.full(v.shape, np.nan)
    for i in range(length - 1, len(v)):
        out[i] = v[i - length + 1:i + 1].max()
    return out


# --------------------------------------------------------------------------- #
#  O(1) incremental states for the developing bar
# --------------------------------------------------------------------------- #
@dataclass
class EmaState:
    """EMA frozen at the last closed bar."""
    length: int
    prev: float          # ema value at the last CLOSED bar

    @property
    def alpha(self) -> float:
        return 2.0 / (self.length + 1.0)

    def step(self, value: float) -> float:
        """EMA if `value` closes the next bar."""
        return self.alpha * value + (1.0 - self.alpha) * self.prev


@dataclass
class RsiState:
    """Wilder RSI frozen at the last closed bar."""
    length: int
    avg_gain: float      # rma(gain) at the last CLOSED bar
    avg_loss: float      # rma(loss) at the last CLOSED bar
    prev_close: float    # close of the last CLOSED bar

    def step(self, value: float) -> float:
        delta = value - self.prev_close
        gain = delta if delta > 0.0 else 0.0
        loss = -delta if delta < 0.0 else 0.0
        n = self.length
        ag = (self.avg_gain * (n - 1) + gain) / n
        al = (self.avg_loss * (n - 1) + loss) / n
        if al == 0.0:
            return 100.0
        if ag == 0.0:
            return 0.0
        return 100.0 - 100.0 / (1.0 + ag / al)


@dataclass
class MacdState:
    """MACD histogram frozen at the last closed bar."""
    fast: int
    slow: int
    signal: int
    ema_fast: float
    ema_slow: float
    sig: float

    def step(self, value: float) -> float:
        """Histogram if `value` closes the next bar."""
        af = 2.0 / (self.fast + 1.0)
        asw = 2.0 / (self.slow + 1.0)
        asig = 2.0 / (self.signal + 1.0)
        ef = af * value + (1.0 - af) * self.ema_fast
        es = asw * value + (1.0 - asw) * self.ema_slow
        macd_line = ef - es
        sig = asig * macd_line + (1.0 - asig) * self.sig
        return macd_line - sig


@dataclass
class SmaState:
    """SMA over a window whose last slot is the developing bar."""
    length: int
    sum_prev: float      # sum of the last (length-1) CLOSED values

    def step(self, value: float) -> float:
        return (self.sum_prev + value) / self.length


# --------------------------------------------------------------------------- #
#  State builders
# --------------------------------------------------------------------------- #
def build_ema_state(closes: Sequence[float], length: int) -> EmaState | None:
    series = ema(closes, length)
    if len(series) == 0 or np.isnan(series[-1]):
        return None
    return EmaState(length=length, prev=float(series[-1]))


def build_rsi_state(closes: Sequence[float], length: int) -> RsiState | None:
    v = np.asarray(closes, dtype=float)
    if len(v) < length + 1:
        return None
    delta = np.diff(v)
    gain = np.where(delta > 0.0, delta, 0.0)
    loss = np.where(delta < 0.0, -delta, 0.0)
    ag = rma(gain, length)
    al = rma(loss, length)
    if np.isnan(ag[-1]) or np.isnan(al[-1]):
        return None
    return RsiState(
        length=length,
        avg_gain=float(ag[-1]),
        avg_loss=float(al[-1]),
        prev_close=float(v[-1]),
    )


def build_macd_state(closes: Sequence[float], fast: int, slow: int, sig: int) -> MacdState | None:
    v = np.asarray(closes, dtype=float)
    ef = ema(v, fast)
    es = ema(v, slow)
    _, sigline, _ = macd(v, fast, slow, sig)
    if np.isnan(ef[-1]) or np.isnan(es[-1]) or np.isnan(sigline[-1]):
        return None
    return MacdState(
        fast=fast, slow=slow, signal=sig,
        ema_fast=float(ef[-1]), ema_slow=float(es[-1]), sig=float(sigline[-1]),
    )


def build_sma_state(values: Sequence[float], length: int) -> SmaState | None:
    v = np.asarray(values, dtype=float)
    if len(v) < length - 1:
        return None
    window = v[-(length - 1):] if length > 1 else np.array([])
    return SmaState(length=length, sum_prev=float(window.sum()))
