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

    # Flat layout: generated files sit in the repo root, not a data/ subfolder.
    paths = {
        "root": ROOT,
        "data": ROOT,
        "snapshot": ROOT / "weekly_snapshot.csv",
        "universe": ROOT / "universe.csv",
        "state": ROOT / "state.json",
    }
    return Config(strategy=strategy, universe=universe, runtime=runtime,
                  secrets=secrets, paths=paths)
