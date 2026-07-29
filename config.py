"""Configuration loading: YAML file for strategy inputs, env vars for secrets."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Flat layout: everything lives beside this file.
ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config.yaml"


@dataclass
class Strategy:
    """Mirrors the Pine `input.*` block one-for-one."""
    len_long: int = 52
    len_short: int = 26
    ema_fast_len: int = 20
    ema_slow_len: int = 50
    ema_slow_back: int = 2
    rsi_len: int = 14
    rsi_min: float = 60.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_sig: int = 9
    vol_sma_len: int = 10
    min_price: float = 100.0

    # market cap (Pine c12) - disabled: Dhan exposes no shares-outstanding field
    use_mcap: bool = False
    min_mcap: float = 1000.0

    # entry block
    strict_entry: bool = True
    # Allow up to N gate conditions to fail. The level break (c01/c02) and the
    # mandatory rows (fresh breakout, min price, market cap) are never relaxed.
    gate_tolerance: int = 0
    # Pro-rate the c09 weekly-volume target by how much of the week has
    # elapsed. Weekly volume accumulates but the 10w SMA is a full-week value,
    # so an un-prorated compare is impossible to pass on Monday morning.
    volume_prorate: bool = False        # legacy alias for volume_mode="bar"
    # c09 volume target:
    #   "off" - raw Pine compare (weekly total vs 10w SMA)
    #   "day" - scale the target by whole sessions elapsed (Mon=1/5 ... Fri=5/5)
    #   "bar" - scale by elapsed 5m bars (very permissive early in the week)
    volume_mode: str = "off"
    # Conviction multiple for volume_mode="pace": require this many times the
    # normal pace-adjusted volume. 1.0 = merely average (too loose).
    volume_pace_mult: float = 2.5

    # ---- Relative-volume filter (ADDITIONAL to Pine's c09, not a replacement)
    # Pine's c09 compares an ACCUMULATING weekly total against a full-week
    # average, so it is near-impossible on Monday and near-free on Friday - it
    # measures elapsed time as much as conviction.
    #
    # RVOL asks the fair question instead: is volume TODAY UP TO THIS MINUTE
    # above what this stock normally does by this minute? Measured on the
    # 27-Jul breakouts it separated the valid names from the false ones by
    # ~20x (valid median 84.5 vs false 4.1), where c09 separated them by ~1x.
    #
    # off  - disabled (default; pure Pine)
    # warn - compute and report it, do not block  <- start here
    # on   - require rvol >= rvol_min
    rvol_mode: str = "off"
    rvol_min: float = 5.0
    rvol_lookback_sessions: int = 20

    # --- MOVER MODE (hunt the explosive breakout) ---------------------------
    # Turn the scan from "every weekly breakout" into "only the ones that are
    # moving hard RIGHT NOW". Three knobs, all measured on 2,378 raw crosses
    # over 12 weeks of the whole NSE cash universe.
    #
    # drop_c09: remove the weekly-volume row from the live gate. It is
    #   structurally impossible early in a week and delays 72% of entries by
    #   a median of 290 minutes (chasing +1.91% higher). Entering at the raw
    #   cross instead of the c09 gate cuts stop-outs from 67.5% to 50.2% and
    #   nearly doubles the rate of +5% moves (11.0% -> 19.7%).
    #
    # rvol_min / bar_rvol_min: real volume conviction, time-of-day aware, so
    #   it works at 09:20 as well as 15:20 - which is exactly what c09 cannot
    #   do. Measured hit rate for a +5% intraday move:
    #       no filter                        2.2%
    #       rvol > 10                        6.5%
    #       rvol > 10 & bar_rvol > 20        ~9%
    #       + atr > 1.5 & ext < 3           10.7%   (4.8x lift)
    #   MONARCH scored rvol 15.8x / bar_rvol 20.8x, TMB 12.3x / 382.7x,
    #   SENCO 10.3x / 19.6x - the names the user wants, all caught.
    #
    # bar_rvol_min compares THIS 5m candle's volume against the median volume
    # of the same clock-minute over the lookback. It is the "something just
    # happened" detector; rvol_min is the "today is a big day" detector.
    drop_c09: bool = False
    bar_rvol_min: float = 0.0

    # --- Gate rows that MEASURED NEGATIVE at a multi-day horizon -------------
    # Both default to True (Pine behaviour, unchanged for the live scanner).
    # Model C turns them off because, over 3,166 breakouts with a 5-day hold,
    # the trades these rows REJECT outperformed the ones they admit:
    #     c03 fresh breakout   passes +2.80%   fails +4.58%   (all 12 months)
    #     c11 price > 100      passes +3.13%   fails +4.52%
    # An already-extended breakout keeps running - momentum begets momentum -
    # and the sub-100 names are where the sharpest moves are. Neither finding
    # holds intraday, which is why this is opt-in rather than a default change.
    require_c03: bool = True
    require_c11: bool = True

    # Weekly volume must be at least N x its 10-week average at the breakout.
    # 0 disables. Measured (5-day hold): 1.5x -> +4.49%, 2.0x -> +5.14%,
    # against +3.86% with no volume requirement.
    min_weekly_vol_x: float = 0.0

    # Reject entries already extended far above the breakout level - chasing a
    # move that has mostly happened. Measured: restricting to ext < 3% lifted
    # the +5% hit rate from 8.3% to 10.7% and cut stop-outs.
    max_ext_above_level: float = 0.0   # 0 = disabled

    # --- Intraday volatility filter (min_atr_pct) ---------------------------
    # Average 5m bar range over the 14 bars up to the signal, as a percent of
    # the entry price. Requires the stock to actually MOVE enough intraday to
    # clear costs before the trade is worth taking.
    #
    # Measured over 449 real signals, 12 weeks, whole NSE cash universe,
    # trailing-stop exit, 0.22% round-trip cost:
    #     no filter    n=449  win 31.4%  avg -0.08%  PF 0.72   <- loses money
    #     atr >= 1.0%  n=175  win 45.7%  avg +0.20%  PF 1.84
    #     atr >= 1.5%  n= 95  win 51.6%  avg +0.24%  PF 2.14
    # It is the ONLY filter tested that held up out of sample
    # (first 6 weeks +0.19%, last 6 weeks +0.21%). Time-of-day did not
    # (before-10:30 went +0.10% -> -0.02%).
    #
    # 0 disables it. See INTRADAY_FINDINGS.md before changing.
    min_atr_pct: float = 0.0
    gate_source: str = "live"        # "live" | "closed"
    defer_entry: bool = True
    gate_daily: bool = False
    req52: bool = False
    one_per_week: bool = True


@dataclass
class Universe:
    exchange_segments: list[str] = field(default_factory=lambda: ["NSE_EQ"])
    series: list[str] = field(default_factory=lambda: ["EQ", "BE"])
    exclude_etf: bool = True
    include_symbols: list[str] = field(default_factory=list)
    exclude_symbols: list[str] = field(default_factory=list)
    max_symbols: int | None = None


@dataclass
class Runtime:
    history_years: int = 5
    min_weekly_bars: int = 60
    prefilter: bool = True
    prefilter_headroom_pct: float = 0.0
    max_workers: int = 5
    data_rate_per_sec: float = 5.0
    quote_rate_per_sec: float = 1.0
    market_open: str = "09:15"
    market_close: str = "15:30"
    bar_interval_min: int = 5
    alert_cooldown_bars: int = 0
    dry_run: bool = False


@dataclass
class Secrets:
    dhan_client_id: str = ""
    dhan_access_token: str = ""
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    @property
    def telegram_ready(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_id)


@dataclass
class Config:
    strategy: Strategy
    universe: Universe
    runtime: Runtime
    secrets: Secrets
    paths: dict[str, Path]


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    val = raw.get(key) or {}
    if not isinstance(val, dict):
        raise ValueError(f"config section '{key}' must be a mapping")
    return val


def _build(cls, data: dict[str, Any]):
    known = {f for f in cls.__dataclass_fields__}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown keys in {cls.__name__}: {sorted(unknown)}")
    return cls(**data)


def load_config(path: str | Path | None = None) -> Config:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    raw: dict[str, Any] = {}
    if cfg_path.exists():
        raw = yaml.safe_load(cfg_path.read_text()) or {}

    strategy = _build(Strategy, _section(raw, "strategy"))
    universe = _build(Universe, _section(raw, "universe"))
    runtime = _build(Runtime, _section(raw, "runtime"))

    if strategy.gate_source not in ("live", "closed"):
        raise ValueError("strategy.gate_source must be 'live' or 'closed'")

    secrets = Secrets(
        dhan_client_id=os.environ.get("DHAN_CLIENT_ID", "").strip(),
        dhan_access_token=os.environ.get("DHAN_ACCESS_TOKEN", "").strip(),
        telegram_bot_token=os.environ.get("TELEGRAM_BOT_TOKEN", "").strip(),
        telegram_chat_id=os.environ.get("TELEGRAM_CHAT_ID", "").strip(),
    )

    # Flat layout: generated files sit beside the config file, not in a data/
    # subfolder. Anchoring to the CONFIG's directory (rather than this module's)
    # means an alternate --config points at its own data set, which keeps tests
    # and side-by-side setups isolated.
    base = cfg_path.resolve().parent if cfg_path.exists() else ROOT
    paths = {
        "root": base,
        "data": base,
        "snapshot": base / "weekly_snapshot.csv",
        "universe": base / "universe.csv",
        "state": base / "state.json",
    }
    return Config(strategy=strategy, universe=universe, runtime=runtime,
                  secrets=secrets, paths=paths)
