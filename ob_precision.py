"""
Pine-exact Python port of `precision.txt` —
**"Institutional OB — Precision Tap & Pre-Order"** (Pine v6, overlay).

WHAT THE INDICATOR DOES
-----------------------
1. DETECT a volume-confirmed bullish *displacement* bar on the chart timeframe:
   green, RVOL >= 1.8, range >= 1.20 x ATR(14), body >= 55% of range,
   close in the top 28% of the range (CLV >= 0.72) and a close above the
   8-bar structure high.
2. ANCHOR a tight order block to the nearest bearish/neutral *origin* candle
   in the 8 bars preceding that displacement. With the default
   "Open to low" method the zone is `open[origin] .. low[origin]`.
3. PLACE a pre-order slightly ABOVE the proximal edge (liquid orders front-run
   the visible zone) and a structural stop below the distal edge.
4. ARM the zone only once price has *departed* (high >= top + 1.00 x ATR), then
   alert on approach, on the exact TAP of the pre-order level, on a closed-bar
   DEFENCE confirmation, and on invalidation.

HOW THIS PORT DIFFERS FROM THE PINE (two deviations, both deliberate)
--------------------------------------------------------------------
* **Live tests use the ATR of the last CLOSED bar.** In Pine, `ta.atr(14)` on
  the forming bar moves with it, so the departure/approach/tap thresholds drift
  during the session and an alert can be justified by a number that no longer
  exists at the close. `OBParams.live_atr_from_closed = True` freezes those
  thresholds for the developing bar. Zone geometry is unaffected: top, bottom,
  entry and stop are computed on a CONFIRMED bar and never recomputed, exactly
  as in the Pine.
* **Everything else is parity.** Same seeding (`ta.atr` = `ta.rma(ta.tr, n)`,
  `ta.sma(volume, n)`), same na-comparison semantics (a comparison against a
  value that does not exist yet is False, never True), same per-bar ordering of
  departure -> approach -> tap -> adaptive entry -> defence -> revert ->
  invalid, same one-tap-per-bar rule, same duplicate-zone rejection, same
  oldest-first eviction at `max_zones`.

NON-REPAINTING GUARANTEE
------------------------
Zones are only ever created from a CONFIRMED bar (`barstate.isconfirmed` in the
Pine, `confirmed=True` here), so a zone's geometry can never change once it
exists. A tap fires from the session's actual low, which can only go lower, so
a fired tap can never be retracted. What CAN change intraday is whether the day
later closes below the stop and kills the zone — that is resolved on the next
closed bar, and every alert this port produces is stamped `confirmed=False`
when it came from a developing bar so the reader knows it.

The port is dependency-light on purpose: `replay()` takes a list of `Bar` and
returns zones plus events, so it is testable without any market feed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from typing import Any, Sequence

import indicators as ind

# Zone states - identical to the Pine's `states` array.
FRESH = 0
TAPPED = 1
CONFIRMED = 2
DEAD = -1

NAN = float("nan")


# --------------------------------------------------------------------------- #
#  Inputs (Pine `input.*` block, one-for-one)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OBParams:
    """Every input of precision.txt, at the file's default value."""

    # --- 1: volume-confirmed displacement
    vol_len: int = 20                 # ta.sma(volume, volLen)
    atr_len: int = 14                 # ta.atr(atrLen)
    min_rvol: float = 1.8
    min_range_atr: float = 1.20
    min_body_frac: float = 0.55
    min_clv: float = 0.72
    structure_len: int = 8            # close > ta.highest(high[1], 8)
    origin_search: int = 8
    allow_neutral: bool = True
    neutral_body: float = 0.20

    # --- 2: exact order-block zone
    zone_method: str = "Open to low"  # | "Body to low" | "Body only" | "Lower half of candle"
    entry_mode: str = "Proximal"      # | "50%" | "62%" | "70.5%" | "79%" | "Distal + 1 tick"
    front_run_mode: str = "Auto"      # | "ATR" | "Ticks" | "Off"
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

    # --- 3: optional defence confirmation
    confirm_bars: int = 3
    confirm_rvol: float = 1.3
    confirm_clv: float = 0.65
    confirm_bos_len: int = 3
    require_sweep: bool = False
    sweep_len: int = 5

    # --- execution details Pine gets from the chart
    mintick: float = 0.05             # NSE equity tick size
    # Deviation #1 above: freeze ATR for the developing bar at the last closed
    # bar's value so live thresholds cannot drift (or repaint) intraday.
    live_atr_from_closed: bool = True

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "OBParams":
        """Build from a config mapping, ignoring nothing: unknown keys raise."""
        data = dict(data or {})
        known = set(cls.__dataclass_fields__)
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown keys in ob_precision: {sorted(unknown)}")
        return cls(**data)


# --------------------------------------------------------------------------- #
#  Bars
# --------------------------------------------------------------------------- #
@dataclass
class Bar:
    """One chart bar. `time` is the session date for daily bars."""
    open: float
    high: float
    low: float
    close: float
    volume: float
    time: Any = None                  # datetime | date | str | None

    @property
    def session(self) -> str:
        """YYYY-MM-DD, used for stable zone identity across runs."""
        t = self.time
        if isinstance(t, (datetime, date)):
            return t.strftime("%Y-%m-%d")
        return str(t or "")


def bars_from_frame(df, time_col: str = "datetime") -> list[Bar]:
    """Convert a dhan.py candle DataFrame into `Bar`s, oldest first."""
    out: list[Bar] = []
    for row in df.to_dict("records"):
        out.append(Bar(
            open=float(row["open"]), high=float(row["high"]),
            low=float(row["low"]), close=float(row["close"]),
            volume=float(row.get("volume") or 0.0),
            time=row.get(time_col),
        ))
    out.sort(key=lambda b: (b.time is None, str(b.time)))
    return out


# --------------------------------------------------------------------------- #
#  Series helpers - Pine seeding rules come from indicators.py, which is the
#  single tested source of truth for ta.rma / ta.sma in this repo.
# --------------------------------------------------------------------------- #
def true_ranges(bars: Sequence[Bar]) -> list[float]:
    """
    ta.tr(handle_na=false): the first bar has no previous close, so Pine falls
    back to high-low rather than na.
    """
    tr: list[float] = []
    prev_close: float | None = None
    for b in bars:
        if prev_close is None:
            tr.append(b.high - b.low)
        else:
            tr.append(max(b.high - b.low,
                          abs(b.high - prev_close),
                          abs(b.low - prev_close)))
        prev_close = b.close
    return tr


def atr_series(bars: Sequence[Bar], length: int) -> list[float]:
    """ta.atr(length) == ta.rma(ta.tr, length). NaN until `length` bars exist."""
    if not bars:
        return []
    return [float(x) for x in ind.rma(true_ranges(bars), length)]


def vol_ma_series(bars: Sequence[Bar], length: int) -> list[float]:
    """ta.sma(volume, length). NaN until `length` bars exist."""
    if not bars:
        return []
    return [float(x) for x in ind.sma([b.volume for b in bars], length)]


def _highest_prior(values: Sequence[float], i: int, length: int) -> float:
    """ta.highest(series[1], length) at bar i -> max(values[i-length .. i-1])."""
    lo = max(0, i - length)
    if i <= 0 or lo >= i:
        return NAN
    return max(values[lo:i])


def _lowest_prior(values: Sequence[float], i: int, length: int) -> float:
    """ta.lowest(series[1], length) at bar i."""
    lo = max(0, i - length)
    if i <= 0 or lo >= i:
        return NAN
    return min(values[lo:i])


# --------------------------------------------------------------------------- #
#  Zone geometry
# --------------------------------------------------------------------------- #
_DEPTH = {"Proximal": 0.0, "50%": 0.50, "62%": 0.62,
          "70.5%": 0.705, "79%": 0.79, "Distal + 1 tick": 1.0}


def zone_depth(params: OBParams) -> float:
    try:
        return _DEPTH[params.entry_mode]
    except KeyError:
        raise ValueError(f"ob_precision.entry_mode must be one of {sorted(_DEPTH)}")


def front_run_buffer(width: float, atr: float, params: OBParams) -> float:
    """
    f_front_buffer(_width, _atr) - the pre-order is placed ABOVE the proximal
    edge because liquid orders front-run the visible zone.

    Auto takes the LARGER of the tick and ATR buffers, then caps it at
    `max_zone_buffer` x zone width so a hairline zone cannot get an entry
    pitched absurdly far above itself.
    """
    tick_buffer = params.front_run_ticks * params.mintick
    atr_buffer = params.front_run_atr * atr
    mode = params.front_run_mode
    if mode == "ATR":
        return atr_buffer
    if mode == "Ticks":
        return tick_buffer
    if mode == "Off":
        return 0.0
    if mode != "Auto":
        raise ValueError("ob_precision.front_run_mode must be Auto|ATR|Ticks|Off")
    if not (atr_buffer == atr_buffer):          # NaN ATR -> no buffer
        atr_buffer = 0.0
    return min(max(tick_buffer, atr_buffer), width * params.max_zone_buffer)


def zone_edges(bar: Bar, params: OBParams) -> tuple[float, float]:
    """rawTop / rawBot for the origin candle, per `zone_method`."""
    o, c, h, l = bar.open, bar.close, bar.high, bar.low
    method = params.zone_method
    if method == "Open to low":
        return o, l
    if method == "Body to low":
        return max(o, c), l
    if method == "Body only":
        return max(o, c), min(o, c)
    if method == "Lower half of candle":
        return l + (h - l) * 0.50, l
    raise ValueError("ob_precision.zone_method must be one of "
                     "'Open to low', 'Body to low', 'Body only', "
                     "'Lower half of candle'")


def pre_order_entry(top: float, bottom: float, atr: float, params: OBParams) -> float:
    """The executable buy level for a zone (Pine `entry`)."""
    width = top - bottom
    base = top - width * zone_depth(params)
    if params.entry_mode == "Distal + 1 tick":
        # A distal entry is already inside the OB, so it is never front-run.
        return bottom + params.mintick
    return base + front_run_buffer(width, atr, params) \
        + params.entry_offset_ticks * params.mintick


# --------------------------------------------------------------------------- #
#  Zone + events
# --------------------------------------------------------------------------- #
@dataclass
class Zone:
    """One precision order block. Mirrors the Pine's parallel arrays."""
    top: float
    bottom: float
    entry: float
    stop: float
    born_index: int                   # bar_index of the displacement bar
    born_session: str                 # YYYY-MM-DD - stable identity across runs
    origin_index: int
    origin_session: str
    atr_at_birth: float
    width: float
    state: int = FRESH
    taps: int = 0
    tap_bar_index: int = -1
    tap_session: str = ""
    departed: bool = False
    # metrics captured at birth, for the alert text
    rvol_at_birth: float = NAN
    range_atr_at_birth: float = NAN
    body_frac_at_birth: float = NAN
    clv_at_birth: float = NAN

    @property
    def live(self) -> bool:
        return self.state >= 0

    def signature(self) -> str:
        """
        Stable identity for de-duplication.

        Deliberately built from the SESSION DATE and the frozen geometry, never
        from `born_index`: the index is positional inside whatever history
        window a run happened to fetch, so it shifts by one every session. A
        signature that moved would re-alert the same tap every single day.
        """
        return f"{self.born_session}|{self.top:.2f}|{self.bottom:.2f}"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, float) and math.isnan(v):
                d[k] = None
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Zone":
        known = cls.__dataclass_fields__
        kwargs = {k: v for k, v in d.items() if k in known}
        for k in ("rvol_at_birth", "range_atr_at_birth",
                  "body_frac_at_birth", "clv_at_birth", "atr_at_birth"):
            if kwargs.get(k) is None:
                kwargs[k] = NAN
        return cls(**kwargs)


@dataclass
class OBEvent:
    """One alertcondition firing."""
    kind: str                         # "ob" | "approach" | "tap" | "confirm" | "invalid"
    zone: Zone
    bar_index: int
    bar_session: str
    bar_time: Any
    confirmed: bool                   # False == came from a developing bar
    price: float                      # the bar's close (or LTP when live)
    low: float
    high: float
    atr: float
    rvol: float
    tap_number: int = 0
    reason: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    def dedupe_key(self, symbol: str) -> str:
        """One alert per (symbol, zone, kind, tap number)."""
        extra = f"|tap{self.tap_number}" if self.kind == "tap" else ""
        return f"{symbol}|{self.zone.signature()}|{self.kind}{extra}"


@dataclass
class ReplayResult:
    zones: list[Zone] = field(default_factory=list)
    events: list[OBEvent] = field(default_factory=list)
    atr: list[float] = field(default_factory=list)
    rvol: list[float] = field(default_factory=list)
    bars: int = 0

    def live_zones(self) -> list[Zone]:
        return [z for z in self.zones if z.live]

    def context(self, bars: Sequence[Bar], params: OBParams) -> dict[str, Any]:
        """
        Everything the intraday pass needs to continue this replay onto a
        developing bar WITHOUT re-fetching history: the last closed ATR, the
        index of the last closed bar, and the prior-bar windows the sweep and
        micro-BOS tests look at.
        """
        if not bars:
            return {}
        highs = [b.high for b in bars]
        lows = [b.low for b in bars]
        volumes = [b.volume for b in bars]
        n = len(bars)
        return {
            "session_index": n - 1,
            "as_of": bars[-1].session,
            "atr": self.atr[-1] if self.atr else NAN,
            "prev_highs": highs[-max(params.structure_len, params.confirm_bos_len):],
            "prev_lows": lows[-params.sweep_len:],
            "prev_volumes": volumes[-max(params.vol_len - 1, 1):],
            "prev_close": bars[-1].close,
            # The last closed bar's own extremes. A bulk quote carries no date,
            # so the caller compares today's quote against these to detect a
            # STALE quote (a market holiday returns the previous session's
            # numbers unchanged) and skip the live pass instead of replaying a
            # session that is already in the closed history.
            "as_of_high": bars[-1].high,
            "as_of_low": bars[-1].low,
            "as_of_close": bars[-1].close,
            "zones": [z.to_dict() for z in self.live_zones()],
        }


# --------------------------------------------------------------------------- #
#  The state machine
# --------------------------------------------------------------------------- #
def _emit(events: list[OBEvent], kind: str, zone: Zone, bar: Bar, i: int,
          confirmed: bool, atr: float, rvol: float, tap_number: int = 0,
          reason: str = "", **detail: Any) -> None:
    events.append(OBEvent(
        kind=kind, zone=zone, bar_index=i, bar_session=bar.session,
        bar_time=bar.time, confirmed=confirmed, price=bar.close,
        low=bar.low, high=bar.high, atr=atr, rvol=rvol,
        tap_number=tap_number, reason=reason, detail=detail,
    ))


def find_origin(bars: Sequence[Bar], i: int, params: OBParams) -> int | None:
    """
    Nearest bearish or neutral candle in the `origin_search` bars before the
    displacement. The loop starts at 1, so the first match is the CLOSEST one.
    Falls back to the immediately preceding candle when there is none.
    """
    for j in range(1, params.origin_search + 1):
        k = i - j
        if k < 0:
            continue
        b = bars[k]
        candle_range = max(b.high - b.low, params.mintick)
        bearish = b.close < b.open
        neutral = (params.allow_neutral
                   and abs(b.close - b.open) / candle_range <= params.neutral_body)
        if bearish or neutral:
            return j
    return 1 if i >= 1 else None


def process_bar(zones: list[Zone], bar: Bar, i: int, *, confirmed: bool,
                atr: float, rvol: float, params: OBParams,
                events: list[OBEvent], bars: Sequence[Bar] | None = None,
                clv: float = NAN, prior_structure: float = NAN,
                prior_sweep_low: float = NAN, micro_bos_high: float = NAN
                ) -> None:
    """
    One bar of the Pine, in the Pine's order:

        displacement -> origin -> duplicate check -> zone creation -> eviction
        then, per zone: departure -> approach -> tap -> adaptive entry
                        -> defence -> revert -> invalid

    `zones` is mutated in place. `confirmed=False` marks a developing bar, which
    suppresses zone creation and defence confirmation exactly like
    `barstate.isconfirmed` does, while still allowing the live approach/tap
    tests - the Pine's own comment: "A live touch alert can fire intrabar."
    """
    rng = max(bar.high - bar.low, params.mintick)

    # ---- displacement (closed bars only) ----------------------------------
    displacement = bool(
        confirmed
        and bar.close > bar.open
        and rvol >= params.min_rvol
        and rng >= atr * params.min_range_atr
        and (abs(bar.close - bar.open) / rng) >= params.min_body_frac
        and clv >= params.min_clv
        and bar.close > prior_structure
    )

    if displacement:
        origin_offset = find_origin(bars or [], i, params)
        if origin_offset is not None:
            ob = (bars or [])[i - origin_offset]
            top, bottom = zone_edges(ob, params)
            if top > bottom:
                entry = pre_order_entry(top, bottom, atr, params)
                duplicate = any(z.live and bottom <= entry <= z.top for z in zones)
                if not duplicate:
                    zones.append(Zone(
                        top=top, bottom=bottom, entry=entry,
                        stop=bottom - atr * params.stop_atr,
                        born_index=i, born_session=bar.session,
                        origin_index=i - origin_offset,
                        origin_session=ob.session,
                        atr_at_birth=atr, width=top - bottom,
                        rvol_at_birth=rvol,
                        range_atr_at_birth=(rng / atr) if atr else NAN,
                        body_frac_at_birth=abs(bar.close - bar.open) / rng,
                        clv_at_birth=clv,
                    ))
                    _emit(events, "ob", zones[-1], bar, i, confirmed, atr, rvol,
                          origin_session=ob.session, origin_offset=origin_offset,
                          entry=entry, stop=zones[-1].stop,
                          width=top - bottom)

    # ---- oldest-first eviction (Pine shifts index 0) -----------------------
    while len(zones) > params.max_zones:
        zones.pop(0)

    # ---- per-zone life cycle ----------------------------------------------
    # `atr` is whatever the caller decided is authoritative for this bar: the
    # bar's own ATR when it is closed (Pine parity), the last CLOSED bar's ATR
    # when it is developing (deviation #1, applied by replay()/live_pass()).
    live_atr = atr

    for z in zones:
        if not z.live:
            continue
        age = i - z.born_index

        # 1. departure arms the zone
        if not z.departed and bar.high >= z.top + live_atr * params.require_departure:
            z.departed = True

        armed = z.departed and age >= params.min_age

        # 2. approach (pre-alert)
        if armed and bar.close > z.entry and bar.low <= z.entry + live_atr * params.approach_atr:
            _emit(events, "approach", z, bar, i, confirmed, live_atr, rvol,
                  distance=bar.low - z.entry)

        # 3. the tap itself - one per zone per bar
        touched = armed and bar.low <= z.entry and bar.high >= z.bottom
        swept = bar.low < prior_sweep_low
        if touched and (not params.require_sweep or swept):
            if z.tap_bar_index < 0 or i > z.tap_bar_index:
                z.taps += 1
                z.tap_bar_index = i
                z.tap_session = bar.session
                z.state = TAPPED
                # Captured BEFORE the adaptive raise below: an alert has to say
                # which level was actually touched, and by the time the event is
                # read `z.entry` may already be the new, higher one.
                tapped_entry = z.entry
                next_entry = z.entry
                if params.raise_after_first_tap and z.taps == 1:
                    # Later tests commonly turn just above Tap 1 instead of
                    # revisiting the theoretical OB, so adopt the defended price.
                    next_entry = max(z.entry, bar.low + live_atr * params.repeat_tap_atr)
                    z.entry = next_entry
                _emit(events, "tap", z, bar, i, confirmed, live_atr, rvol,
                      tap_number=z.taps, tapped_entry=tapped_entry,
                      next_entry=next_entry, swept=bool(swept))

        # 4. closed-bar defence confirmation
        pending = (z.state == TAPPED and z.tap_bar_index >= 0
                   and i - z.tap_bar_index <= params.confirm_bars)
        defence = bool(
            confirmed and pending
            and bar.close > bar.open
            and clv >= params.confirm_clv
            and rvol >= params.confirm_rvol
            and bar.close > z.top
            and bar.close > micro_bos_high
        )
        if defence:
            z.state = CONFIRMED
            _emit(events, "confirm", z, bar, i, confirmed, live_atr, rvol)

        # 5. a tap that was never confirmed goes back to fresh
        if z.state == TAPPED and z.tap_bar_index >= 0 \
                and i - z.tap_bar_index > params.confirm_bars:
            z.state = FRESH

        # 6. invalidation
        if z.live and (bar.close < z.stop or z.taps > params.max_touches):
            z.state = DEAD
            reason = ("closed below the structural stop"
                      if bar.close < z.stop
                      else f"exhausted after {z.taps} touches")
            _emit(events, "invalid", z, bar, i, confirmed, live_atr, rvol,
                  reason=reason)


def replay(bars: Sequence[Bar], params: OBParams | None = None,
           live_bars: int = 0) -> ReplayResult:
    """
    Replay a bar series through the indicator.

    `live_bars` is the number of TRAILING bars that are still developing
    (0 = every bar is closed, 1 = the last bar is today's forming candle).
    Developing bars can arm/tap/kill a zone but can never CREATE one or confirm
    a defence - the non-repainting half of the port.
    """
    params = params or OBParams()
    res = ReplayResult(bars=len(bars))
    if not bars:
        return res

    res.atr = atr_series(bars, params.atr_len)
    volumes = [b.volume for b in bars]
    vol_ma = vol_ma_series(bars, params.vol_len)
    # Pine: rvol = volMA > 0 ? volume / volMA : 0.0  - a missing average reads
    # as ZERO participation, not as "unknown", so it can never pass min_rvol.
    res.rvol = [(v / m) if (m == m and m > 0) else 0.0
                for v, m in zip(volumes, vol_ma)]

    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    first_live = len(bars) - max(0, live_bars)

    zones: list[Zone] = []
    for i, bar in enumerate(bars):
        confirmed = i < first_live
        rng = max(bar.high - bar.low, params.mintick)
        clv = (bar.close - bar.low) / rng
        atr_i = res.atr[i] if i < len(res.atr) else NAN
        # Deviation #1: a developing bar is judged against the last CLOSED ATR.
        atr_eff = atr_i
        if not confirmed and params.live_atr_from_closed and i > 0:
            atr_eff = res.atr[i - 1]
        process_bar(
            zones, bar, i,
            confirmed=confirmed, atr=atr_eff, rvol=res.rvol[i], params=params,
            events=res.events, bars=bars, clv=clv,
            prior_structure=_highest_prior(highs, i, params.structure_len),
            prior_sweep_low=_lowest_prior(lows, i, params.sweep_len),
            micro_bos_high=_highest_prior(highs, i, params.confirm_bos_len),
        )

    res.zones = zones
    return res


def live_pass(zones: Sequence[dict[str, Any]] | Sequence[Zone],
              ctx: dict[str, Any], params: OBParams | None = None, *,
              open_: float, high: float, low: float, last_price: float,
              volume: float = 0.0, bar_time: Any = None,
              session: str = "") -> list[OBEvent]:
    """
    Continue a persisted replay onto TODAY's developing bar using only a bulk
    quote - no per-symbol history call.

    This is what makes a 5-minute cron affordable: the closed-bar replay runs
    once per session per symbol, and every run in between judges the live tap
    from one bulk OHLC request for the whole waiting list.

    The quote's high/low are the session's extremes SO FAR. A tap derived from
    them cannot be retracted, because a session low can only go lower.
    """
    params = params or OBParams()
    if not ctx or not zones:
        return []
    atr = ctx.get("atr", NAN)
    if atr is None or (isinstance(atr, float) and math.isnan(atr)):
        return []                        # no closed ATR -> no frozen thresholds
    live = [z if isinstance(z, Zone) else Zone.from_dict(z) for z in zones]
    bar = Bar(open=float(open_ or 0.0), high=float(high or 0.0),
              low=float(low or 0.0), close=float(last_price or 0.0),
              volume=float(volume or 0.0), time=bar_time or session or None)
    if not session:
        session = bar.session
    else:
        bar.time = bar.time or session

    i = int(ctx.get("session_index", -1)) + 1
    if i <= 0:
        return []

    # Pine computes ta.sma(volume, volLen) over a window that INCLUDES the
    # forming bar, so the live RVOL is today's volume against the last
    # (volLen-1) closed sessions plus today. Reported for context only: no
    # confirmed-bar test (displacement, defence) can run on a developing bar.
    prev_vols = list(ctx.get("prev_volumes") or [])
    need = params.vol_len - 1
    if volume > 0 and len(prev_vols) >= need > 0:
        window = prev_vols[-need:] + [float(volume)]
        avg = sum(window) / len(window)
        rvol = (float(volume) / avg) if avg > 0 else 0.0
    else:
        rvol = 0.0

    # The developing bar's own session, for prior-bar windows we do not need:
    # sweep and micro-BOS only matter for confirmed-bar logic, and the live bar
    # is never confirmed, so both are fed NaN -> those comparisons stay False.
    events: list[OBEvent] = []
    process_bar(live, bar, i, confirmed=False, atr=float(atr), rvol=rvol,
                params=params, events=events, bars=None,
                clv=NAN, prior_structure=NAN, prior_sweep_low=NAN,
                micro_bos_high=NAN)
    # `bars=None` blocks zone creation; `confirmed=False` already did. Stamp the
    # session so the caller can key de-duplication on today rather than on the
    # (empty) developing bar's own time.
    for e in events:
        if session:
            e.bar_session = session
    return events


def restore_context(ctx: dict[str, Any]) -> list[Zone]:
    """Zones persisted by `ReplayResult.context()`, oldest first."""
    return [Zone.from_dict(z) for z in (ctx.get("zones") or [])]
