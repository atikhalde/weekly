"""
Tests for `ob_precision.py` - the Pine-exact port of precision.txt
("Institutional OB — Precision Tap & Pre-Order").

Three things are pinned down here, in order of how badly they would hurt if
they were wrong:

1. PARITY. Every leg of the displacement test, the origin-candle search, the
   zone geometry and the arm/tap/defence/invalidate life cycle must reproduce
   the indicator's numbers, including Pine's `ta.atr` / `ta.sma` seeding. The
   expected values are derived BY HAND (or by an independent re-implementation
   of the Pine formula) rather than by calling the port, so a wrong formula
   cannot mark its own homework.

2. NON-REPAINTING. A zone may only be born on a CLOSED bar, its geometry is
   frozen for life, and a developing bar is judged against the last CLOSED ATR
   (deviation #1). A fired tap must never be retractable.

3. STABLE IDENTITY. Zone signatures and de-dupe keys must survive a shift of
   the fetch window and a JSON round-trip, or the scanner would re-alert the
   same order block on every single run.

Run with:  python -m pytest test_ob_precision.py -q
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import pandas as pd
import pytest

from ob_precision import (
    CONFIRMED, DEAD, FRESH, OBEvent, OBParams, TAPPED, Bar, ReplayResult, Zone,
    atr_series, bars_from_frame, find_origin, front_run_buffer, live_pass,
    pre_order_entry, process_bar, replay, restore_context, true_ranges,
    vol_ma_series, zone_depth, zone_edges,
)

P = OBParams()
D0 = date(2026, 8, 3)


def d(i: int) -> date:
    return D0 + timedelta(days=i)


# --------------------------------------------------------------------------- #
#  Fixtures
# --------------------------------------------------------------------------- #
def quiet(i: int, vol: float = 100.0) -> Bar:
    """A bearish do-nothing bar: close < open, range 3.0, volume 100."""
    return Bar(100.0, 101.0, 98.0, 99.0, vol, d(i))


def scenario(n_quiet: int = 20) -> list[Bar]:
    """
    The canonical set-up used by most tests below.

        idx 0..n-1  quiet bearish bars        o100   h101   l98    c99    vol 100
        idx n       DISPLACEMENT              o100   h107   l99.5  c106.5 vol 300
        idx n+1     departure (clears top+ATR) o106.5 h112   l106   c111.5 vol 150
        idx n+2     drift (age 2)             o111.5 h112.5 l109   c110   vol 120
        idx n+3     THE TAP (age 3)           o110   h110.5 l100.5 c105   vol 200

    With the default n_quiet = 20 the displacement is bar 20 and the hand
    arithmetic is:

      tr[20]  = max(107-99.5, |107-99|, |99.5-99|) = 8.0
      atr[13] = mean(tr[0..13]) = 3.0        (Pine seeds the RMA with an SMA)
      atr[20] = (3.0*13 + 8.0)/14 = 47/14 = 3.357142857...

      origin candle = idx 19 (nearest bearish) -> top = open = 100.0,
                                                 bottom = low = 98.0, width 2.0
      front-run buffer (Auto) = min(max(2*0.05, 0.18*atr), 0.40*width)
                              = min(max(0.10, 0.604285714), 0.80) = 0.604285714
      entry = 100.0 + 0.604285714 = 100.604285714
      stop  = 98.0 - 0.15*atr     = 97.496428571

      atr[21] = (atr[20]*13 + 6.0)/14    (tr[21] = 112-106 = 6.0)
      atr[22] = (atr[21]*13 + 3.5)/14    (tr[22] = 112.5-109 = 3.5)
      atr[23] = (atr[22]*13 + 10.0)/14   (tr[23] = 110.5-100.5 = 10.0)
      Tap 1 raises the entry to max(entry, low + 0.05*atr[23])
                              = max(100.604286, 100.5 + 0.200194) = 100.700194
    """
    n = n_quiet
    bars = [quiet(i) for i in range(n)]
    bars += [
        Bar(100.0, 107.0, 99.5, 106.5, 300.0, d(n)),        # displacement
        # Departure bar: rvol is only 1.33, so it is not a second displacement.
        Bar(106.5, 112.0, 106.0, 111.5, 150.0, d(n + 1)),
        Bar(111.5, 112.5, 109.0, 110.0, 120.0, d(n + 2)),   # drift, age 2
        Bar(110.0, 110.5, 100.5, 105.0, 200.0, d(n + 3)),   # age 3 -> Tap 1
    ]
    return bars


ATR20 = 47.0 / 14.0
ATR21 = (ATR20 * 13 + 6.0) / 14.0
ATR22 = (ATR21 * 13 + 3.5) / 14.0
ATR23 = (ATR22 * 13 + 10.0) / 14.0
BUFFER20 = min(max(2 * 0.05, 0.18 * ATR20), 2.0 * 0.40)
ENTRY20 = 100.0 + BUFFER20
STOP20 = 98.0 - 0.15 * ATR20
RAISED_ENTRY = 100.5 + 0.05 * ATR23


def pine_atr(bars: list[Bar], n: int = 14) -> list[float]:
    """
    An INDEPENDENT re-derivation of ta.atr: true range, seeded with the SMA of
    the first `n` values, then Wilder-smoothed. Written out here on purpose so
    the port is not graded by its own helper.
    """
    tr = true_ranges(bars)
    out = [float("nan")] * len(bars)
    if len(bars) < n:
        return out
    out[n - 1] = sum(tr[:n]) / n
    for i in range(n, len(bars)):
        out[i] = (out[i - 1] * (n - 1) + tr[i]) / n
    return out


def kinds(res: ReplayResult, confirmed: bool | None = None) -> list[str]:
    return [e.kind for e in res.events
            if confirmed is None or e.confirmed is confirmed]


def one_zone(res: ReplayResult) -> Zone:
    assert len(res.zones) == 1, f"expected exactly one zone, got {res.zones}"
    return res.zones[0]


def mk_event(zone: Zone, kind: str, tap: int = 0) -> OBEvent:
    return OBEvent(kind=kind, zone=zone, bar_index=0, bar_session="",
                   bar_time=None, confirmed=True, price=0.0, low=0.0, high=0.0,
                   atr=0.0, rvol=0.0, tap_number=tap)


# --------------------------------------------------------------------------- #
#  1. Pine parity - series helpers
# --------------------------------------------------------------------------- #
def test_true_range_first_bar_falls_back_to_high_minus_low():
    bars = [Bar(100.0, 105.0, 99.0, 104.0, 1.0, d(0)),
            Bar(104.0, 110.0, 90.0, 95.0, 1.0, d(1))]
    # ta.tr(handle_na=false): no previous close on bar 0 -> high-low, not na.
    assert true_ranges(bars) == [6.0, 20.0]


def test_atr_matches_an_independent_wilder_derivation():
    bars = scenario() + [Bar(105.0, 108.0, 104.0, 107.0, 180.0, d(24))]
    got, want = atr_series(bars, 14), pine_atr(bars, 14)
    # NaN != NaN, so compare the warm-up mask and the settled values separately.
    assert [x != x for x in got] == [x != x for x in want] == [True] * 13 + [False] * 12
    assert got[13:] == pytest.approx(want[13:])
    assert got[20] == pytest.approx(ATR20)
    assert got[23] == pytest.approx(ATR23)


def test_atr_is_nan_until_length_bars_exist():
    assert all(x != x for x in atr_series(scenario()[:10], 14))
    assert atr_series([], 14) == []


def test_volume_average_is_a_plain_sma_and_rvol_reads_zero_without_it():
    bars = scenario()
    vma = vol_ma_series(bars, 20)
    assert all(x != x for x in vma[:19])
    assert vma[19] == pytest.approx(100.0)                  # 20 quiet bars
    assert vma[20] == pytest.approx((19 * 100.0 + 300.0) / 20.0)
    # rvol = volMA > 0 ? volume / volMA : 0.0 - a missing average reads as ZERO
    # participation, never as "unknown", so it can never pass min_rvol.
    res = replay(bars, P)
    assert res.rvol[20] == pytest.approx(300.0 / vma[20])
    assert res.rvol[20] >= P.min_rvol
    assert all(r == 0.0 for r in res.rvol[:19])


def test_no_displacement_is_possible_before_the_averages_warm_up():
    """A 15-bar history cannot produce a zone: rvol is 0.0."""
    bars = [quiet(i) for i in range(14)]
    bars.append(Bar(100.0, 130.0, 99.0, 129.0, 100000.0, d(14)))
    assert replay(bars, P).zones == []


# --------------------------------------------------------------------------- #
#  2. Pine parity - displacement
# --------------------------------------------------------------------------- #
def test_displacement_creates_one_zone_with_hand_derived_geometry():
    res = replay(scenario()[:21], P)
    z = one_zone(res)
    assert z.born_session == d(20).isoformat()
    assert z.origin_session == d(19).isoformat()
    assert (z.top, z.bottom, z.width) == (100.0, 98.0, 2.0)
    assert z.entry == pytest.approx(ENTRY20)
    assert z.stop == pytest.approx(STOP20)
    assert z.atr_at_birth == pytest.approx(ATR20)
    # FRESH and untouched, but already DEPARTED: Pine's per-zone loop runs after
    # the array.push, so the displacement bar's own high (107) is tested against
    # top + ATR (103.36) and arms the zone it just created.
    assert z.state == FRESH and z.taps == 0 and z.departed
    assert z.range_atr_at_birth == pytest.approx(7.5 / ATR20)
    assert z.body_frac_at_birth == pytest.approx(6.5 / 7.5)
    assert z.clv_at_birth == pytest.approx(7.0 / 7.5)
    assert kinds(res) == ["ob"]


@pytest.mark.parametrize("o,h,l,c,v,why", [
    # Each row breaks exactly one leg of Pine's `displacement` expression and
    # leaves the other five passing, so a silent `and` -> `or` cannot hide.
    (100.0, 107.0, 99.5, 106.5, 100.0, "rvol 100/110 = 0.91 < min_rvol 1.8"),
    (100.0, 103.0, 99.5, 102.5, 300.0, "range 3.5 < 1.2 * ATR 3.69"),
    (103.0, 107.0, 99.5, 106.5, 300.0, "body/range 3.5/7.5 = 0.47 < 0.55"),
    (101.5, 108.0, 101.2, 106.0, 300.0, "clv (106-101.2)/6.8 = 0.71 < 0.72"),
    # A bearish bar can never satisfy body_frac and clv at the same time, so
    # this one breaks `close > open` and, necessarily, its neighbours with it.
    (108.0, 108.5, 100.0, 101.0, 300.0, "close < open: not a bullish displacement"),
])
def test_every_leg_of_the_displacement_test_is_required(o, h, l, c, v, why):
    bars = scenario()[:21]
    bars[20] = Bar(o, h, l, c, v, d(20))
    assert replay(bars, P).zones == [], why


def test_displacement_must_close_above_the_prior_structural_high():
    bars = scenario()[:21]
    # Bar 19 spikes to 120, so ta.highest(high[1], 8) at idx 20 is 120 and the
    # displacement's close of 106.5 no longer clears it.
    bars[19] = Bar(100.0, 120.0, 98.0, 99.0, 100.0, d(19))
    assert replay(bars, P).zones == []


def test_origin_is_the_nearest_bearish_candle_not_the_first_one_found():
    bars = scenario()[:21]
    bars[19] = Bar(100.0, 101.0, 98.0, 100.8, 100.0, d(19))    # bullish -> skipped
    assert one_zone(replay(bars, P)).origin_session == d(18).isoformat()
    assert find_origin(bars, 20, P) == 2


def test_origin_search_is_bounded_and_falls_back_to_the_previous_bar():
    bars = scenario()[:21]
    for i in range(12, 20):                                     # 8 bullish bars
        bars[i] = Bar(100.0, 101.0, 98.0, 100.9, 100.0, d(i))
    # Nothing bearish/neutral inside origin_search -> Pine's own fallback: 1.
    assert find_origin(bars, 20, P) == 1
    assert one_zone(replay(bars, P)).origin_session == d(19).isoformat()


def test_neutral_origin_is_accepted_only_when_allowed():
    bars = scenario()[:21]
    bars[19] = Bar(100.0, 101.0, 98.0, 100.0, 100.0, d(19))     # body 0 -> neutral
    assert one_zone(replay(bars, P)).origin_session == d(19).isoformat()
    # With neutral disallowed the search walks back to the next bearish candle.
    strict = one_zone(replay(bars, OBParams(allow_neutral=False)))
    assert strict.origin_session == d(18).isoformat()


def test_neutral_body_threshold_is_a_fraction_of_the_candle_range():
    bars = scenario()[:21]
    # body 1.0 over range 3.0 = 0.33 > neutral_body 0.20 -> not neutral, and
    # close > open means not bearish either, so the search moves on.
    bars[19] = Bar(100.0, 101.5, 98.5, 101.0, 100.0, d(19))
    assert find_origin(bars, 20, P) == 2
    assert find_origin(bars, 20, OBParams(neutral_body=0.40)) == 1


# --------------------------------------------------------------------------- #
#  3. Pine parity - zone geometry
# --------------------------------------------------------------------------- #
ORIGIN = Bar(100.0, 101.0, 98.0, 99.0, 100.0, d(19))


@pytest.mark.parametrize("method,top,bottom", [
    ("Open to low", 100.0, 98.0),
    ("Body to low", 100.0, 98.0),        # max(open, close) = open on a bearish bar
    ("Body only", 100.0, 99.0),
    ("Lower half of candle", 99.5, 98.0),
])
def test_zone_methods(method, top, bottom):
    assert zone_edges(ORIGIN, OBParams(zone_method=method)) == (top, bottom)


def test_zone_method_rejects_an_unknown_value():
    with pytest.raises(ValueError, match="zone_method"):
        zone_edges(ORIGIN, OBParams(zone_method="Fibonacci"))


@pytest.mark.parametrize("mode,depth", [
    ("Proximal", 0.0), ("50%", 0.50), ("62%", 0.62),
    ("70.5%", 0.705), ("79%", 0.79), ("Distal + 1 tick", 1.0),
])
def test_zone_depth(mode, depth):
    assert zone_depth(OBParams(entry_mode=mode)) == depth


def test_entry_modes_place_the_pre_order_at_the_right_depth():
    top, bottom, atr = 100.0, 98.0, ATR20
    buf = min(max(0.10, 0.18 * atr), 0.80)
    assert pre_order_entry(top, bottom, atr, OBParams(entry_mode="Proximal")) \
        == pytest.approx(top + buf)
    assert pre_order_entry(top, bottom, atr, OBParams(entry_mode="50%")) \
        == pytest.approx(99.0 + buf)
    # A distal entry is already INSIDE the OB, so it is never front-run.
    assert pre_order_entry(top, bottom, atr, OBParams(entry_mode="Distal + 1 tick")) \
        == pytest.approx(98.05)
    with pytest.raises(ValueError, match="entry_mode"):
        pre_order_entry(top, bottom, atr, OBParams(entry_mode="80%"))


@pytest.mark.parametrize("mode,expected", [
    ("Auto", 0.6042857142857143),   # max(2 ticks, 0.18*ATR) = 0.6043, cap 0.80
    ("ATR", 0.18 * ATR20),          # 0.6043 - uncapped
    ("Ticks", 0.10),                # 2 * mintick
    ("Off", 0.0),
])
def test_front_run_modes(mode, expected):
    assert front_run_buffer(2.0, ATR20, OBParams(front_run_mode=mode)) \
        == pytest.approx(expected)


def test_auto_front_run_buffer_is_capped_at_a_fraction_of_the_zone():
    """A hairline zone cannot get an entry pitched absurdly far above itself."""
    assert front_run_buffer(0.5, ATR20, P) == pytest.approx(0.5 * 0.40)
    # ...but a wide zone takes the full ATR buffer.
    assert front_run_buffer(10.0, ATR20, P) == pytest.approx(0.18 * ATR20)
    with pytest.raises(ValueError, match="front_run_mode"):
        front_run_buffer(2.0, ATR20, OBParams(front_run_mode="Smart"))


def test_entry_offset_ticks_shifts_the_pre_order():
    base = pre_order_entry(100.0, 98.0, ATR20, P)
    shifted = pre_order_entry(100.0, 98.0, ATR20, OBParams(entry_offset_ticks=4))
    assert shifted == pytest.approx(base + 4 * 0.05)


def test_stop_sits_below_the_zone_by_stop_atr():
    z = one_zone(replay(scenario()[:21], OBParams(stop_atr=0.5)))
    assert z.stop == pytest.approx(98.0 - 0.5 * ATR20)


def test_an_origin_candle_with_no_height_creates_nothing():
    """top > bottom is required; a flat zone is skipped, not crashed on."""
    bars = scenario()[:21]
    bars[19] = Bar(98.0, 99.0, 98.0, 98.0, 100.0, d(19))        # open == low
    assert replay(bars, P).zones == []


# --------------------------------------------------------------------------- #
#  4. Duplicate zones and eviction
# --------------------------------------------------------------------------- #
def two_displacement_scenario(second_origin: Bar) -> list[Bar]:
    """A second displacement at idx 23 whose origin candle is `second_origin`."""
    bars = [quiet(i) for i in range(20)]
    bars += [
        Bar(100.0, 107.0, 99.5, 106.5, 300.0, d(20)),           # zone A, origin idx 19
        Bar(106.5, 112.0, 106.0, 111.5, 150.0, d(21)),
        second_origin,                                          # idx 22
        Bar(112.0, 120.0, 111.5, 119.5, 400.0, d(23)),          # displacement B
    ]
    return bars


def test_a_second_zone_inside_an_existing_one_is_rejected():
    # Origin B = open 99.5 / low 98.5 -> width 1.0, so the Auto buffer is capped
    # at 0.40 and the entry lands on 99.90 - INSIDE zone A's [98, 100].
    bars = two_displacement_scenario(Bar(99.5, 100.0, 98.5, 99.0, 120.0, d(22)))
    res = replay(bars, P)
    assert len(res.zones) == 1
    assert res.zones[0].born_session == d(20).isoformat()
    assert kinds(res) == ["ob"]                                 # no second "ob"


def test_a_second_zone_outside_the_existing_one_is_kept():
    bars = two_displacement_scenario(Bar(105.0, 106.0, 103.5, 104.0, 120.0, d(22)))
    res = replay(bars, P)
    assert [z.born_session for z in res.zones] == [d(20).isoformat(),
                                                   d(23).isoformat()]
    assert kinds(res) == ["ob", "ob"]
    # width 1.5 -> the 0.40 cap binds again, so the entry is 105.60
    assert res.zones[1].entry == pytest.approx(105.0 + 1.5 * 0.40)


def test_max_zones_evicts_the_oldest_first():
    """Pine shifts index 0 out; the port must drop the OLDEST zone, not the newest."""
    old = Zone(top=100.0, bottom=98.0, entry=100.6, stop=97.5, born_index=10,
               born_session="2026-08-13", origin_index=9,
               origin_session="2026-08-12", atr_at_birth=3.0, width=2.0)
    zones = [old]
    bar = Bar(112.0, 120.0, 111.5, 119.5, 400.0, d(23))
    process_bar(zones, bar, 23, confirmed=True, atr=4.0, rvol=3.0,
                params=OBParams(max_zones=1), events=[],
                bars=[quiet(i) for i in range(23)] + [bar],
                clv=0.94, prior_structure=112.0 - 1e-9)
    assert len(zones) == 1
    assert zones[0].born_session == d(23).isoformat()


# --------------------------------------------------------------------------- #
#  5. Arming: departure and minimum age
# --------------------------------------------------------------------------- #
def test_departure_arms_the_zone():
    assert one_zone(replay(scenario(), P)).departed


def test_no_departure_means_no_tap_even_when_price_reaches_the_entry():
    bars = scenario()
    # 4 * ATR is more than this series ever clears above the zone top, so the
    # zone stays un-armed even though bar 23 dips right onto the entry.
    res = replay(bars, OBParams(require_departure=4.0))
    z = one_zone(res)
    assert not z.departed and z.taps == 0
    assert "tap" not in kinds(res)
    # ...and the very same bars do tap under the normal 1 * ATR requirement.
    assert one_zone(replay(bars, P)).taps == 1


def test_the_birth_bar_itself_can_depart_the_zone():
    """
    Parity detail worth locking down: Pine pushes the new zone and THEN runs the
    per-zone loop over every zone including the new one, so a displacement that
    closes well above its own origin candle arms the zone on bar zero. Anything
    else would silently delay every tap by one session.
    """
    assert one_zone(replay(scenario()[:21], P)).departed


def test_departure_threshold_scales_with_require_departure():
    bars = scenario()
    # The departure bar's high of 112 is 12 above the zone top: enough for
    # 1 * ATR (3.55) but not for 4 * ATR (14.18).
    assert one_zone(replay(bars, OBParams(require_departure=4.0))).departed is False
    assert one_zone(replay(bars, OBParams(require_departure=1.0))).departed is True


def test_no_tap_before_min_age():
    bars = scenario()
    bars[22] = Bar(111.5, 112.0, 100.5, 101.0, 120.0, d(22))    # dips at age 2
    assert one_zone(replay(bars[:23], P)).taps == 0
    assert "tap" not in kinds(replay(bars[:23], P))
    # ...and the same dip one bar later, at age 3, does tap.
    bars2 = bars[:23] + [Bar(101.0, 110.5, 100.5, 105.0, 200.0, d(23))]
    assert one_zone(replay(bars2, P)).taps == 1


# --------------------------------------------------------------------------- #
#  6. The tap
# --------------------------------------------------------------------------- #
def test_tap_fires_once_the_entry_is_touched_from_above():
    res = replay(scenario(), P)
    z = one_zone(res)
    taps = [e for e in res.events if e.kind == "tap"]
    assert z.taps == 1 and z.state == TAPPED
    assert z.tap_session == d(23).isoformat()
    assert len(taps) == 1 and taps[0].tap_number == 1
    assert taps[0].confirmed is True
    assert taps[0].low == 100.5 and taps[0].atr == pytest.approx(ATR23)
    assert taps[0].detail["next_entry"] == pytest.approx(RAISED_ENTRY)
    # The event must also carry the level that was ACTUALLY touched: Tap 1
    # raises zone.entry, so an alert built from the zone alone would print a
    # price the market has already left.
    assert taps[0].detail["tapped_entry"] == pytest.approx(ENTRY20)


def test_tap1_raises_the_entry_to_the_defended_low():
    z = one_zone(replay(scenario(), P))
    assert z.entry == pytest.approx(RAISED_ENTRY)
    assert z.entry > ENTRY20


def test_the_entry_is_not_raised_again_without_the_switch():
    z = one_zone(replay(scenario(), OBParams(raise_after_first_tap=False)))
    assert z.entry == pytest.approx(ENTRY20)


def test_repeated_dips_count_one_tap_per_bar_and_ratchet_the_entry_once():
    dip = lambda i, low: Bar(110.0, 110.5, low, 105.0, 200.0, d(i))   # noqa: E731
    bars = scenario() + [dip(24, 100.5), dip(25, 100.6)]
    res = replay(bars, P)
    z = one_zone(res)
    assert z.taps == 3
    assert [e.tap_number for e in res.events if e.kind == "tap"] == [1, 2, 3]
    entries = [e.detail["next_entry"] for e in res.events if e.kind == "tap"]
    # The raise happens after Tap 1 only; later taps keep the defended level.
    assert entries[0] == pytest.approx(RAISED_ENTRY)
    assert RAISED_ENTRY > ENTRY20
    assert entries[1] == entries[2] == pytest.approx(entries[0])
    assert z.live                                   # 3 taps <= max_touches 4


def test_a_bar_entirely_below_the_zone_is_not_a_tap():
    """Pine requires high >= bottom: a gap through the zone is a break, not a tap."""
    bars = scenario()
    bars[23] = Bar(97.9, 97.95, 96.0, 97.6, 100.0, d(23))       # high < bottom 98.0
    res = replay(bars, P)
    z = one_zone(res)
    assert z.taps == 0 and z.live                   # close 97.6 is above the stop
    assert kinds(res) == ["ob"]


def test_tap_precedes_invalidation_on_the_same_bar():
    """Pine order: the tap is reported, THEN the close below the stop kills it."""
    bars = scenario()
    bars[23] = Bar(105.0, 106.0, 96.0, 97.0, 200.0, d(23))      # dips to 96, closes 97
    res = replay(bars, P)
    z = one_zone(res)
    assert z.taps == 1 and z.state == DEAD
    assert kinds(res) == ["ob", "tap", "invalid"]
    assert res.events[-1].reason == "closed below the structural stop"


def test_max_touches_exhausts_the_zone():
    dip = lambda i: Bar(110.0, 110.5, 100.5, 105.0, 200.0, d(i))       # noqa: E731
    bars = scenario() + [dip(24), dip(25), dip(26), dip(27)]
    res = replay(bars, P)
    z = one_zone(res)
    assert z.taps == 5 and z.state == DEAD
    assert res.events[-1].kind == "invalid"
    assert res.events[-1].reason == "exhausted after 5 touches"
    assert not res.live_zones()


def test_max_touches_is_configurable():
    dip = lambda i: Bar(110.0, 110.5, 100.5, 105.0, 200.0, d(i))       # noqa: E731
    res = replay(scenario() + [dip(24)], OBParams(max_touches=1))
    assert one_zone(res).state == DEAD


# --------------------------------------------------------------------------- #
#  7. Defence confirmation
# --------------------------------------------------------------------------- #
# At idx 24 the micro-BOS reference is ta.highest(high[1], 3) = 112.5 (idx 22),
# so a confirming close has to beat that as well as the zone top of 100.
DEFENCE = Bar(110.0, 114.0, 109.5, 113.5, 400.0, d(24))


def test_defence_confirms_a_tap_within_the_window():
    res = replay(scenario() + [DEFENCE], P)
    z = one_zone(res)
    assert z.state == CONFIRMED
    ev = [e for e in res.events if e.kind == "confirm"]
    assert len(ev) == 1 and ev[0].confirmed is True
    assert ev[0].bar_session == d(24).isoformat()


@pytest.mark.parametrize("bar,why", [
    (Bar(113.9, 114.0, 109.5, 113.5, 400.0, d(24)), "close 113.5 < open 113.9"),
    (Bar(110.0, 114.0, 113.0, 113.5, 400.0, d(24)), "clv 0.5/1.0 = 0.50 < 0.65"),
    (Bar(110.0, 114.0, 109.5, 113.5, 110.0, d(24)), "rvol 0.92 < 1.3"),
    (Bar(110.0, 112.4, 109.5, 112.2, 400.0, d(24)), "close 112.2 < micro-BOS 112.5"),
    # `close > zone top` is implied by the micro-BOS leg in this fixture (the
    # top is 100 and the recent highs are 112.5), so this row breaks both.
    (Bar(98.5, 100.0, 98.5, 99.9, 400.0, d(24)), "close 99.9 below the zone top 100"),
])
def test_defence_needs_every_leg(bar, why):
    res = replay(scenario() + [bar], P)
    assert one_zone(res).state != CONFIRMED, why


def test_defence_window_is_confirm_bars_wide():
    filler = [Bar(105.0, 108.0, 104.0, 107.0, 150.0, d(i)) for i in (24, 25, 26)]
    late = Bar(110.0, 114.0, 109.5, 113.5, 400.0, d(27))        # 4 bars after the tap
    res = replay(scenario() + filler + [late], P)
    assert one_zone(res).state == FRESH           # tapped, window expired, reverted
    assert "confirm" not in kinds(res)


def test_an_unconfirmed_tap_reverts_to_fresh():
    filler = [Bar(105.0, 108.0, 104.0, 107.0, 150.0, d(i)) for i in (24, 25, 26, 27)]
    res = replay(scenario() + filler, P)
    z = one_zone(res)
    assert z.taps == 1 and z.state == FRESH
    assert "confirm" not in kinds(res)


def test_confirm_bars_zero_disables_the_confirmation_stage():
    res = replay(scenario() + [DEFENCE], OBParams(confirm_bars=0))
    assert one_zone(res).state != CONFIRMED


# --------------------------------------------------------------------------- #
#  8. Optional liquidity sweep
# --------------------------------------------------------------------------- #
def test_require_sweep_blocks_a_tap_without_a_prior_low_sweep():
    # ta.lowest(low[1], 5) at idx 23 is 98.0 (the quiet bars); the tap bar's low
    # of 100.5 never takes it out.
    res = replay(scenario(), OBParams(require_sweep=True))
    assert one_zone(res).taps == 0 and "tap" not in kinds(res)


def test_require_sweep_allows_a_tap_that_takes_out_the_prior_low():
    bars = scenario()
    bars[23] = Bar(110.0, 110.5, 97.9, 105.0, 200.0, d(23))     # low 97.9 < 98.0
    res = replay(bars, OBParams(require_sweep=True))
    z = one_zone(res)
    assert z.taps == 1 and z.state == TAPPED      # close 105 is above the stop
    ev = [e for e in res.events if e.kind == "tap"][0]
    assert ev.detail["swept"] is True


# --------------------------------------------------------------------------- #
#  9. Non-repainting
# --------------------------------------------------------------------------- #
WILD = Bar(101.0, 140.0, 98.5, 139.0, 5000.0, d(24))


def test_a_developing_bar_can_never_create_a_zone():
    base = replay(scenario(), P)
    live = replay(scenario() + [WILD], P, live_bars=1)
    geom = lambda zs: [(z.born_session, z.top, z.bottom, z.entry, z.stop) for z in zs]  # noqa: E731
    assert geom(base.zones) == geom(live.zones)
    assert not [e for e in live.events if e.kind == "ob" and not e.confirmed]


def test_a_developing_bar_can_never_confirm_a_defence():
    live = replay(scenario() + [Bar(110.0, 114.0, 109.5, 113.5, 400.0, d(24))],
                  P, live_bars=1)
    assert one_zone(live).state == TAPPED          # not CONFIRMED
    assert "confirm" not in kinds(live)
    # ...and the very same bar, once closed, does confirm.
    assert one_zone(replay(scenario() + [DEFENCE], P)).state == CONFIRMED


def test_a_developing_bar_is_judged_against_the_last_closed_atr():
    """Deviation #1: frozen thresholds, so an intraday alert cannot be retracted."""
    frozen = replay(scenario() + [WILD], OBParams(live_atr_from_closed=True), live_bars=1)
    drift = replay(scenario() + [WILD], OBParams(live_atr_from_closed=False), live_bars=1)
    live_f = [e for e in frozen.events if not e.confirmed]
    live_d = [e for e in drift.events if not e.confirmed]
    assert live_f and live_d
    assert all(e.atr == pytest.approx(frozen.atr[-2]) for e in live_f)
    assert all(e.atr == pytest.approx(drift.atr[-1]) for e in live_d)
    assert frozen.atr[-2] != pytest.approx(drift.atr[-1])


def test_a_developing_bar_may_still_tap_and_may_still_be_killed():
    """Live touches are the whole point; only creation/confirmation are barred."""
    res = replay(scenario() + [WILD], P, live_bars=1)
    taps = [e for e in res.events if e.kind == "tap"]
    assert taps and taps[-1].confirmed is False
    # WILD lows at 98.5 and closes at 139, both above the stop -> zone survives.
    assert one_zone(res).live


def test_a_developing_bar_that_breaks_the_stop_kills_the_zone_immediately():
    broken = Bar(101.0, 102.0, 90.0, 91.0, 900.0, d(24))
    res = replay(scenario() + [broken], P, live_bars=1)
    assert one_zone(res).state == DEAD
    assert [e.kind for e in res.events if not e.confirmed][-1] == "invalid"


def test_closed_zone_geometry_is_unchanged_by_later_bars():
    """Appending history never rewrites what an older zone decided."""
    early = replay(scenario(), P)
    snap = [(z.born_session, z.top, z.bottom, z.stop) for z in early.zones]
    later = replay(scenario() + [Bar(105.0, 108.0, 104.0, 107.0, 180.0, d(i))
                                 for i in (24, 25, 26)], P)
    assert len(later.zones) == 1
    assert [(z.born_session, z.top, z.bottom, z.stop) for z in later.zones] == snap


# --------------------------------------------------------------------------- #
#  10. The intraday continuation (live_pass)
# --------------------------------------------------------------------------- #
LIVE_QUOTE = dict(open_=101.0, high=103.0, low=100.60, last_price=102.5, volume=260.0)
LIVE_BAR = Bar(101.0, 103.0, 100.60, 102.5, 260.0, d(24))


def test_live_pass_reproduces_the_same_bar_replayed_as_developing():
    closed = scenario()
    ctx = replay(closed, P).context(closed, P)
    via_live = live_pass(ctx["zones"], ctx, P, session=d(24).isoformat(), **LIVE_QUOTE)
    via_replay = [e for e in replay(closed + [LIVE_BAR], P, live_bars=1).events
                  if not e.confirmed]
    shape = lambda evs: [(e.kind, e.zone.born_session, e.tap_number) for e in evs]  # noqa: E731
    assert shape(via_live) == shape(via_replay)
    assert via_live and all(e.confirmed is False for e in via_live)
    assert all(e.bar_session == d(24).isoformat() for e in via_live)
    # The tap is Tap 2: the closed replay already took Tap 1 on idx 23.
    assert [e.tap_number for e in via_live if e.kind == "tap"] == [2]


def test_live_pass_is_idempotent_over_repeated_runs():
    """
    The scanner reloads the persisted zones on every run, so quoting the same
    session twice must not produce a second tap.
    """
    closed = scenario()
    ctx = replay(closed, P).context(closed, P)
    first = live_pass(ctx["zones"], ctx, P, session=d(24).isoformat(), **LIVE_QUOTE)
    second = live_pass(ctx["zones"], ctx, P, session=d(24).isoformat(), **LIVE_QUOTE)
    key = lambda evs: [(e.kind, e.zone.signature(), e.tap_number) for e in evs]  # noqa: E731
    assert key(first) == key(second)
    # ...and the persisted context was not mutated by the pass.
    assert ctx["zones"][0]["entry"] == pytest.approx(RAISED_ENTRY)


def test_live_pass_computes_rvol_the_way_pine_does():
    """ta.sma(volume, 20) over a window that INCLUDES the forming bar."""
    closed = scenario()
    ctx = replay(closed, P).context(closed, P)
    evs = live_pass(ctx["zones"], ctx, P, session=d(24).isoformat(), **LIVE_QUOTE)
    window = [b.volume for b in closed][-19:] + [260.0]
    assert evs[0].rvol == pytest.approx(260.0 / (sum(window) / len(window)))


def test_live_pass_without_a_volume_reports_zero_rvol():
    closed = scenario()
    ctx = replay(closed, P).context(closed, P)
    quote = dict(LIVE_QUOTE, volume=0.0)
    evs = live_pass(ctx["zones"], ctx, P, session=d(24).isoformat(), **quote)
    assert evs and all(e.rvol == 0.0 for e in evs)


def test_live_pass_needs_a_closed_atr_and_a_zone():
    short = scenario()[:10]                        # too short for ATR(14)
    ctx = replay(short, P).context(short, P)
    assert live_pass(ctx.get("zones") or [], ctx, P, session=d(10).isoformat(),
                     **LIVE_QUOTE) == []
    assert live_pass([], {}, P, session="2026-08-27", **LIVE_QUOTE) == []


def test_context_carries_the_stale_quote_guard():
    """
    A bulk quote has no date, so the context stores the last closed bar's
    extremes: on a weekday market holiday the feed returns the previous session
    unchanged, and the scanner compares against these before trusting it.
    """
    closed = scenario()
    ctx = replay(closed, P).context(closed, P)
    last = closed[-1]
    assert (ctx["as_of_high"], ctx["as_of_low"], ctx["as_of_close"]) \
        == (last.high, last.low, last.close)


# --------------------------------------------------------------------------- #
#  11. Persistence and stable identity
# --------------------------------------------------------------------------- #
def test_zone_signature_ignores_the_positional_index():
    a = Zone(100.0, 98.0, 100.6, 97.5, 20, "2026-08-23", 19, "2026-08-22", 3.3, 2.0)
    b = Zone(100.0, 98.0, 100.6, 97.5, 517, "2026-08-23", 516, "2026-08-22", 3.3, 2.0)
    assert a.signature() == b.signature() == "2026-08-23|100.00|98.00"


def test_signature_is_stable_when_the_fetch_window_slides():
    """
    Every run fetches a sliding window, so born_index shifts daily. If the
    de-dupe key moved with it, the scanner would re-alert the same tap forever.
    """
    full = scenario(n_quiet=30)                    # 30 quiet bars: room to slice
    a = replay(full, P)
    b = replay(full[4:], P)
    assert one_zone(a).signature() == one_zone(b).signature()
    assert one_zone(a).born_index != one_zone(b).born_index
    assert [e.dedupe_key("X") for e in a.events] == [e.dedupe_key("X") for e in b.events]
    assert "tap" in kinds(b)


def test_dedupe_keys_separate_zones_kinds_and_tap_numbers():
    z1 = Zone(100.0, 98.0, 100.6, 97.5, 20, "2026-08-23", 19, "2026-08-22", 3.3, 2.0)
    z2 = Zone(100.0, 98.0, 100.6, 97.5, 21, "2026-08-24", 19, "2026-08-22", 3.3, 2.0)
    assert mk_event(z1, "tap", 1).dedupe_key("S") != mk_event(z2, "tap", 1).dedupe_key("S")
    assert mk_event(z1, "tap", 1).dedupe_key("S") != mk_event(z1, "tap", 2).dedupe_key("S")
    assert mk_event(z1, "tap", 1).dedupe_key("S") != mk_event(z1, "ob", 0).dedupe_key("S")
    assert mk_event(z1, "ob").dedupe_key("S") == "S|2026-08-23|100.00|98.00|ob"
    assert mk_event(z1, "tap", 3).dedupe_key("S") == "S|2026-08-23|100.00|98.00|tap|tap3"


def test_zone_survives_a_json_round_trip():
    z = one_zone(replay(scenario(), P))
    back = Zone.from_dict(json.loads(json.dumps(z.to_dict(), default=str)))
    assert back.top == z.top and back.bottom == z.bottom
    assert back.entry == pytest.approx(z.entry) and back.stop == pytest.approx(z.stop)
    assert back.born_session == z.born_session and back.state == z.state
    assert back.taps == z.taps and back.departed == z.departed
    assert back.signature() == z.signature()


def test_nan_birth_metrics_round_trip_as_null_and_back():
    z = Zone(100.0, 98.0, 100.6, 97.5, 20, "2026-08-23", 19, "2026-08-22",
             float("nan"), 2.0)
    dumped = json.loads(json.dumps(z.to_dict()))
    assert dumped["atr_at_birth"] is None           # NaN is not valid JSON
    assert Zone.from_dict(dumped).atr_at_birth != Zone.from_dict(dumped).atr_at_birth


def test_context_round_trips_and_still_taps():
    closed = scenario()
    ctx = json.loads(json.dumps(replay(closed, P).context(closed, P), default=str))
    zones = restore_context(ctx)
    assert len(zones) == 1
    assert zones[0].signature() == one_zone(replay(closed, P)).signature()
    evs = live_pass(zones, ctx, P, session=d(24).isoformat(), **LIVE_QUOTE)
    assert [e.kind for e in evs if e.kind == "tap"] == ["tap"]


def test_context_records_what_the_live_pass_needs():
    closed = scenario()
    ctx = replay(closed, P).context(closed, P)
    assert ctx["session_index"] == len(closed) - 1
    assert ctx["as_of"] == d(23).isoformat()
    assert ctx["atr"] == pytest.approx(ATR23)
    assert ctx["prev_close"] == closed[-1].close
    assert len(ctx["prev_lows"]) == P.sweep_len
    assert len(ctx["prev_volumes"]) == P.vol_len - 1
    assert len(ctx["prev_highs"]) == max(P.structure_len, P.confirm_bos_len)
    assert len(ctx["zones"]) == 1
    assert replay([], P).context([], P) == {}


def test_only_live_zones_are_persisted():
    """Dead zones are dropped from the context, so they can never tap again."""
    dip = lambda i: Bar(110.0, 110.5, 100.5, 105.0, 200.0, d(i))       # noqa: E731
    bars = scenario() + [dip(24), dip(25), dip(26), dip(27)]
    res = replay(bars, P)
    assert res.zones and not res.live_zones()
    assert res.context(bars, P)["zones"] == []


# --------------------------------------------------------------------------- #
#  12. Input plumbing
# --------------------------------------------------------------------------- #
def test_bars_from_frame_sorts_and_reads_the_session_in_ist():
    tz = "Asia/Kolkata"
    df = pd.DataFrame({
        "datetime": [pd.Timestamp("2026-08-05 09:15", tz=tz),
                     pd.Timestamp("2026-08-03 09:15", tz=tz),
                     pd.Timestamp("2026-08-04 09:15", tz=tz)],
        "open": [3.0, 1.0, 2.0], "high": [3.5, 1.5, 2.5],
        "low": [2.5, 0.5, 1.5], "close": [3.0, 1.0, 2.0],
        "volume": [30.0, 10.0, 20.0],
    })
    bars = bars_from_frame(df)
    assert [b.session for b in bars] == ["2026-08-03", "2026-08-04", "2026-08-05"]
    assert [b.close for b in bars] == [1.0, 2.0, 3.0]


def test_bars_from_frame_tolerates_missing_volume():
    df = pd.DataFrame({"datetime": [d(0), d(1)], "open": [1.0, 2.0],
                       "high": [1.5, 2.5], "low": [0.5, 1.5],
                       "close": [1.0, 2.0]})
    assert [b.volume for b in bars_from_frame(df)] == [0.0, 0.0]


def test_bar_session_handles_dates_strings_and_none():
    assert Bar(1, 1, 1, 1, 1, d(3)).session == "2026-08-06"
    assert Bar(1, 1, 1, 1, 1, "2026-08-06 09:15:00+05:30").session \
        == "2026-08-06 09:15:00+05:30"
    assert Bar(1, 1, 1, 1, 1, None).session == ""


def test_from_mapping_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown keys"):
        OBParams.from_mapping({"min_rvol": 2.0, "min_rvol_typo": 2.0})
    assert OBParams.from_mapping({"min_rvol": 2.0}).min_rvol == 2.0
    assert OBParams.from_mapping(None) == OBParams()


def test_replay_of_nothing_is_empty_not_an_error():
    res = replay([], P)
    assert res.zones == [] and res.events == [] and res.bars == 0
    assert res.live_zones() == []
