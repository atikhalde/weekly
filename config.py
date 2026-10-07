"""Configuration loading: YAML file for strategy inputs, env vars for secrets."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
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
    # Our market cap is shares-outstanding x price, and the share count is a
    # reported figure that can be a few percent stale. Validated against nine
    # TradingView c12 readings the error ran -3.9%..+2.3%, so a stock computing
    # at 1010 Cr might really be 980. This margin keeps borderline names IN
    # rather than silently dropping a genuine setup: with 5.0 the filter
    # rejects only below min_mcap * 0.95.
    mcap_margin_pct: float = 5.0

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

    # --- BTST day-character gate (Model E) -----------------------------------
    # btst_only restricts a model to breakouts whose CANDLE matches a measured
    # BTST tier - the YASHO shape. Buying every breakout close and selling the
    # next close earns +0.01% (nothing); the edge is entirely in the character
    # of the breakout day:
    #     TIER A  day >= +15% and closed in the top 15% of range   +1.75%, t 5.2
    #     TIER B  closed top 10% of range, rvol >= 3, atr >= 3%    +0.83%, t 5.0
    # btst_top_n caps how many are taken per day, best first (Tier A, then the
    # largest day move). Both are inert unless a model sets them, so the live
    # scanner is unaffected.
    btst_only: bool = False
    btst_top_n: int = 0

    # --- Anticipation gate (Model F) -----------------------------------------
    # anticipate_only makes a model trade the PRE-breakout list written by
    # btst.py (anticipate_picks.csv) instead of the breakout list. Measured:
    # buying on proximity alone is -0.195%/trade (t -4.57), but requiring the
    # stock to close in the top 10% of its daily range turns that into
    # +0.688% (t 7.73, +0.701% out of sample). Inert unless a model sets it.
    anticipate_only: bool = False
    anticipate_mode: str = "all"

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

    # Which level the 5-minute cross fires on.
    #   "entry"     highest(high,26) as of the last closed week  (Pine default)
    #   "hi_short2" highest(high,26) as of two weeks ago - table row 3, a
    #               lower level that triggers earlier and more often.
    trigger_level: str = "entry"

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
    breakout_cooldown_weeks: int = 26      # require a fresh 26W cross after lockout
    dry_run: bool = False
    data_source: str = "yfinance"


@dataclass
class Secrets:
    dhan_client_id: str = ""
    dhan_access_token: str = ""
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    # Optional SECOND destination - a different bot and/or a different chat.
    # Everything the primary receives is mirrored here. Leave blank to disable.

    @property
    def telegram_ready(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_id)

    @property
    def telegram_destinations(self) -> list[tuple[str, str, str]]:
        """
        (token, chat_id, label) for every configured destination, primary first.

        A destination needs BOTH a token and a chat id; a half-configured pair
        is skipped rather than silently posting to the wrong place.

        2026-08-14: the second bot was removed. TELEGRAM_CHAT_ID_2 had been
        answering "400 chat not found" on every single send for weeks, three
        times per document upload, and nobody was reading that chat. The
        fan-out machinery below is kept because it costs nothing and makes
        adding a real second destination a config change again.
        """
        out: list[tuple[str, str, str]] = []
        if self.telegram_bot_token and self.telegram_chat_id:
            out.append((self.telegram_bot_token, self.telegram_chat_id, "primary"))
        return out


@dataclass
class OBPrecision:
    """
    Inputs for the precision order-block tap scanner (`ob_tap_scan.py`).

    Two halves live in one block on purpose:

    * the RUNTIME knobs below, which belong to the scanner (how often it is
      allowed to pull history, what it alerts on, where its state file is); and
    * the INDICATOR inputs, which are a field-for-field mirror of `precision.txt`
      ("Institutional OB — Precision Tap & Pre-Order"). `params()` hands those to
      `ob_precision.OBParams`, which validates them and refuses unknown keys, so
      a typo here fails loudly at start-up instead of silently changing the
      strategy.

    The scanner is additive: nothing else in the repo reads this section, and
    `enabled: false` turns the whole thing off without touching any other job.
    """

    # --- scanner runtime -----------------------------------------------------
    enabled: bool = True
    # Its OWN state file. state.json belongs to scan.py and is read-only here.
    state_file: str = "ob_precision_state.json"
    # Daily bars per symbol. 250 sessions is ~one year: enough for ATR(14),
    # sma(volume,20) and a meaningful set of order blocks, and short enough
    # that a 370-name backfill fits inside the workflow's rate limits.
    sessions: int = 250
    # Cap on per-symbol history calls in ONE run. The backfill spreads over
    # several runs rather than blowing the 12-minute job timeout; every run
    # after the first only refreshes names whose session has not been replayed.
    max_refresh_per_run: int = 250
    # How many retained weeks of state.json to seed the waiting list from.
    # state.py prunes to six weeks, so this is the full backfill.
    backfill_weeks: int = 6
    quote_batch: int = 1000          # Dhan's OHLC limit per request
    # Which events reach Telegram. "ob" = a new precision order block,
    # "tap" = price tapped the pre-order level. "approach" / "confirm" /
    # "invalid" are computed and can be switched on, but are noisy by default.
    alert_kinds: list[str] = field(default_factory=lambda: ["ob", "tap"])
    alert_taps: list[int] = field(default_factory=lambda: [1])
    # On the FIRST refresh a symbol replays a year of history, which contains
    # order blocks born months ago. Only closed-bar events from this many days
    # back are alertable, so the backfill cannot bury the chat. Live events
    # (today's developing bar) are never filtered.
    event_lookback_days: int = 5
    # "Until tapped or invalidated": take the name off the active list on the
    # first alerted tap. The record is kept for audit either way.
    resolve_on_tap: bool = True

    # --- the daily waiting-list digest ---------------------------------------
    # Alerts only ever name the symbols that DID something. The list itself is
    # the thing worth tracking by hand - what is armed, at what price, and how
    # long it has been waiting - so once per slot the whole active list goes to
    # the same chat. Both slots ride the existing 5-minute cron: no extra
    # schedule, and the pre-open one costs no API calls at all because it is
    # built from the cache the post-close run left behind.
    daily_digest: bool = True
    # Post-close recap: the first run after the bell whose zone cache is
    # complete for today (the 15:35 run, or the 15:40 one when a large list is
    # still spreading its refreshes over runs).
    digest_after_close: bool = True
    # Pre-open plan: the first run at or after this time (IST) on a weekday,
    # before the market opens. Yesterday's closed levels, read before the bell.
    digest_before_open: bool = True
    digest_pre_open_at: str = "09:10"
    # 0 = every waiting name, however many Telegram messages that takes. A cap
    # drops the OLDEST breakouts and says how many were dropped.
    digest_max_rows: int = 0

    # --- indicator inputs (precision.txt, defaults unchanged) ----------------
    # 1: volume-confirmed displacement
    vol_len: int = 20
    atr_len: int = 14
    min_rvol: float = 1.8
    min_range_atr: float = 1.20
    min_body_frac: float = 0.55
    min_clv: float = 0.72
    structure_len: int = 8
    origin_search: int = 8
    allow_neutral: bool = True
    neutral_body: float = 0.20
    # 2: exact order-block zone
    zone_method: str = "Open to low"
    entry_mode: str = "Proximal"
    front_run_mode: str = "Auto"
    front_run_atr: float = 0.18
    front_run_ticks: int = 2
    max_zone_buffer: float = 0.40
    entry_offset_ticks: int = 0
    stop_atr: float = 0.15
    approach_atr: float = 0.25
    min_age: int = 3
    max_zones: int = 30
    max_touches: int = 4
    raise_after_first_tap: bool = True
    repeat_tap_atr: float = 0.05
    require_departure: float = 1.0
    # 3: optional defence confirmation
    confirm_bars: int = 3
    confirm_rvol: float = 1.3
    confirm_clv: float = 0.65
    confirm_bos_len: int = 3
    require_sweep: bool = False
    sweep_len: int = 5
    # execution details Pine takes from the chart
    mintick: float = 0.05
    # Deviation #1: freeze the developing bar's thresholds at the last CLOSED
    # ATR, so an intraday alert can never be retracted by a later ATR move.
    live_atr_from_closed: bool = True

    def params(self):
        """The indicator half of this block, as a validated `OBParams`."""
        from ob_precision import OBParams     # local: keeps config.py import-light
        known = set(OBParams.__dataclass_fields__)
        return OBParams.from_mapping(
            {k: v for k, v in asdict(self).items() if k in known})


@dataclass
class PrecisionOBEntry:
    """
    Settings for `precision_ob_entry.py` - the "buy the OB candle itself"
    rule (the accepted Tap 1 configuration with an earlier entry).

    The rule's own numbers live here; the zone definition is NOT repeated.
    `cfg.ob_precision.params()` supplies the displacement/origin thresholds, so
    the OB this job trades can never drift from the OB the live scanner draws.

    Like every spectator job in this repo, it reads `ob_precision_state.json`
    read-only, keeps its own state file, and `enabled: false` turns it off
    without touching anything else.
    """

    enabled: bool = True
    # Its OWN state file - the stage-2 scanner's belongs to ob_tap_scan.py.
    state_file: str = "precision_ob_entry_state.json"
    # The rule as backtested: a 90-session time stop and a flat 0.22% round
    # trip (the cost every number in the report is net of).
    time_stop_sessions: int = 90
    round_trip_cost_pct: float = 0.22
    # A rule event is alertable for this many sessions (measured by the OB
    # candle's age, not the confirmation bar's). GitHub's cron skips slots
    # (BUG 55) and entry A's price is history either way; a late alert walks
    # the bars first and reports "already resolved" instead of quoting a plan
    # nobody can take. Events older than this are marked seen silently, so
    # deploying the job never replays history into the chat.
    catchup_sessions: int = 3
    # postclose ~16:05 IST: the rule event + the complete plan + exit tracking.
    confirm_alerts: bool = True
    # intraday ~15:12 IST: the OB forming on today's bar - entry B, the price
    # that is actually fillable, with the OB candle already resolved from the
    # closed bars. Off means the job only reports confirmed events.
    forming_alerts: bool = True
    # intraday: a heads-up on the OB candle's OWN close - the only moment
    # entry A could ever be filled, but the order block does not exist until a
    # displacement follows, and post-breakout red/neutral candles above the
    # level are common. Off by default; the backtest's own caveat ("treat
    # entry B as the tradable version") is why.
    candidate_alerts: bool = False
    # Daily bars fetched per evaluated symbol - must reach back past the
    # breakout session (the cycle is up to 26 weeks old) to the OB candle.
    lookback_days: int = 560

    def settings(self):
        """The rule's knobs, as the alert module's own settings object."""
        from precision_ob_entry import RuleSettings   # local: keep import-light
        known = set(RuleSettings.__dataclass_fields__)
        return RuleSettings(**{k: getattr(self, k) for k in known
                               if hasattr(self, k)})


@dataclass
class Config:
    strategy: Strategy
    universe: Universe
    runtime: Runtime
    secrets: Secrets
    paths: dict[str, Path]
    # Last and defaulted, so every existing Config(...) construction keeps working.
    ob_precision: OBPrecision = field(default_factory=OBPrecision)
    precision_ob_entry: PrecisionOBEntry = field(
        default_factory=PrecisionOBEntry)


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
    # Optional: an absent section means "every default", so an older config.yaml
    # keeps working unchanged. Unknown keys inside it still raise, via _build.
    ob_precision = _build(OBPrecision, _section(raw, "ob_precision"))
    # Also optional: an older config.yaml without the section gets the
    # defaults, and a typo inside it still raises.
    precision_ob_entry = _build(PrecisionOBEntry,
                                _section(raw, "precision_ob_entry"))

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
        "mcap": base / "mcap.csv",
        # The precision-OB scanner's OWN state (waiting list, zone cache, alert
        # de-dupe keys). Deliberately a separate file: state.json is scan.py's,
        # and two jobs writing one file is how alert history gets lost.
        "ob_state": base / ob_precision.state_file,
    }
    return Config(strategy=strategy, universe=universe, runtime=runtime,
                  secrets=secrets, paths=paths, ob_precision=ob_precision,
                  precision_ob_entry=precision_ob_entry)
