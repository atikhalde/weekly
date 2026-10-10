"""Tests for precision_ob_entry.py - the "buy the OB candle itself" alert.

Pure-logic tests on a synthetic ob_precision_state-shaped dict: the funnel
(first post-breakout OB, OB candle on/after the breakout, OB close above the
level), the gap-aware fill policy the backtest uses, the 90-session time stop,
the plan's two entries, the forming/dispatch rules and the first-run safety.
No network, no Dhan, no Telegram.
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest

import precision_ob_entry as pye  # noqa: E402
from ob_precision import Bar, OBParams  # noqa: E402
from precision_ob_entry import (  # noqa: E402
    RuleSettings, TradePlan, check_exit, evaluate_rule, first_ob_after_breakout,
    walk_bars,
)

PARAMS = OBParams()          # the live config.yaml defaults


# --------------------------------------------------------------------------- #
#  Fixtures: bars, state
# --------------------------------------------------------------------------- #
def bar(session, o, h, l, c, v=1000.0):
    return Bar(open=o, high=h, low=l, close=c, volume=v, time=session)


# A synthetic cycle: breakout 2026-08-20 at level 100; a swing high of 120 on
# 2026-09-29; a bearish OB candle on 2026-09-30 (close 110, above the level);
# the displacement that births the order block on 2026-10-01 (close 123).
BREAKOUT = "2026-08-20"
OB_CANDLE = "2026-09-30"
BORN = "2026-10-01"
LEVEL = 100.0


def cycle_bars():
    return [
        bar("2026-08-19", 97, 99, 96, 98),
        bar(BREAKOUT, 99, 103, 98, 102),
        bar("2026-08-21", 102, 106, 101, 105),
        bar("2026-09-28", 112, 118, 111, 117),
        bar("2026-09-29", 117, 120, 116, 119),          # the swing high: 120
        bar(OB_CANDLE, 115, 116, 108, 110),             # the OB candle
        bar(BORN, 112, 124, 111, 123),                  # the displacement
    ]


def quiet_bars():
    """cycle_bars() with a tame displacement bar, so the rule's trade does NOT
    resolve on the born session - the case a still-open book needs."""
    bars = cycle_bars()
    bars[-1] = bar(BORN, 112, 119, 111, 118)
    return bars


def state(sym="TEST", level=LEVEL, brk=BREAKOUT, events=None, zones=None,
          status="waiting", as_of=None, as_of_high=None, as_of_low=None,
          as_of_close=None, refreshed_on=None, as_of_open=None):
    return {
        "waiting": {sym: {
            "status": status, "security_id": "1", "exchange_segment": "NSE_EQ",
            "breakout_bar": f"{brk}T09:30+05:30",
            "breakout_26w_session": {"session": brk, "level": level,
                                     "close": level * 1.01},
        }},
        "zones": {sym: {
            "as_of": as_of, "as_of_open": as_of_open,
            "as_of_high": as_of_high, "as_of_low": as_of_low,
            "as_of_close": as_of_close, "refreshed_on": refreshed_on,
            "zones": zones or [], "closed_events": events or [],
            "prev_highs": [90.0] * 8, "prev_volumes": [1000.0] * 19,
            "atr": 5.0, "breakout_26w_session": {"session": brk,
                                                 "level": level},
        }},
    }


def ob_event(born=BORN, origin=OB_CANDLE, price=123.0, top=116.0,
             bottom=108.0, rvol=2.6):
    return {"kind": "ob", "born_session": born, "origin_session": origin,
            "session": born, "price": price, "top": top, "bottom": bottom,
            "entry": 113.0, "stop": 105.0, "rvol": rvol, "confirmed": True,
            "detail": {"origin_session": origin, "origin_offset": 1}}


def rule_for(bars=None, brk=None, born=BORN, params=PARAMS):
    bars = bars if bars is not None else cycle_bars()
    brk = brk or {"session": BREAKOUT, "level": LEVEL}
    return evaluate_rule(bars, brk, born, params)


# --------------------------------------------------------------------------- #
#  Sessions
# --------------------------------------------------------------------------- #
def test_weekday_sessions_counts_weekdays_only():
    assert pye.weekday_sessions(date(2026, 10, 5), date(2026, 10, 12)) == 5


def test_sessions_between_counts_real_bars():
    bars = [bar("2026-09-30", 1, 1, 1, 1), bar("2026-10-01", 1, 1, 1, 1),
            bar("2026-10-05", 1, 1, 1, 1)]
    assert pye.sessions_between(bars, date(2026, 9, 30), date(2026, 10, 5)) == 2


# --------------------------------------------------------------------------- #
#  Detection - the live scanner's first_zone_after_breakout rule
# --------------------------------------------------------------------------- #
def test_first_ob_is_the_earliest_born_on_or_after_the_breakout():
    st = state(events=[ob_event(born="2026-09-10"), ob_event(born=BORN)])
    born, _z = first_ob_after_breakout(st["waiting"]["TEST"],
                                       st["zones"]["TEST"])
    assert born == "2026-09-10"


def test_zones_born_before_the_breakout_are_not_candidates():
    st = state(events=[ob_event(born="2026-07-01"), ob_event(born=BORN)])
    born, _z = first_ob_after_breakout(st["waiting"]["TEST"],
                                       st["zones"]["TEST"])
    assert born == BORN


def test_live_zones_and_closed_events_are_merged():
    """The zone list covers what is alive; closed events cover what the
    scanner has already pruned from it. A birth missing from one is still
    found through the other."""
    st = state(events=[ob_event(born="2026-09-10")],
               zones=[{"born_session": BORN, "origin_session": OB_CANDLE,
                       "rvol_at_birth": 2.2}])
    births = pye.ob_births(st["waiting"]["TEST"], st["zones"]["TEST"])
    assert [b for b, _ in births] == ["2026-09-10", BORN]


def test_no_breakout_means_no_event():
    st = state()
    st["waiting"]["TEST"]["breakout_26w_session"] = {}
    st["zones"]["TEST"]["breakout_26w_session"] = {}
    assert first_ob_after_breakout(st["waiting"]["TEST"],
                                   st["zones"]["TEST"]) is None


def test_breakout_of_prefers_the_candle_derived_cross():
    """The backtest dates cycles from the c02 cross (derive_26w_breakout
    semantics). The 5-minute `breakout_bar` anchor exists to keep an armed
    symbol stable and is deliberately NOT the cycle's date."""
    rec = {"breakout_bar": "2026-09-02T09:35+05:30",
           "breakout_26w_session": {"session": BREAKOUT, "level": LEVEL,
                                    "close": 102.0}}
    assert pye.breakout_of(rec, {})["session"] == BREAKOUT


def long_cycle_bars():
    """cycle_bars() with enough warm-up for the scanner's own replay() to
    compute ATR (atr_len=14) and the volume average (vol_len=20), and a volume
    spike on the displacement bar so the OB is actually born. This is the
    fixture for the port-side recomputation: `ob_precision.replay` cannot
    create a zone without a real ATR and rvol."""
    bars = []
    d = date(2026, 6, 1)
    while len(bars) < 30:
        if d.weekday() < 5:
            bars.append(bar(d.isoformat(), 90, 91, 89, 90))
        d += timedelta(days=1)
    bars += [
        bar("2026-08-19", 97, 99, 96, 98),
        bar(BREAKOUT, 99, 103, 98, 102),
        bar("2026-08-21", 102, 106, 101, 105),
        bar("2026-09-28", 112, 118, 111, 117),
        bar("2026-09-29", 117, 120, 116, 119),
        bar(OB_CANDLE, 115, 116, 108, 110),
        bar(BORN, 112, 124, 111, 123, v=5000.0),          # the displacement
    ]
    return bars


def test_births_from_bars_recomputes_the_event_with_the_scanner_s_own_port():
    """The state is a trigger, not the truth: the event is put back on the
    indicator's own terms by replaying the fetched bars with the same port
    ob_tap_scan.py runs."""
    births = pye.births_from_bars(long_cycle_bars(),
                                  {"session": BREAKOUT, "level": LEVEL},
                                  PARAMS)
    assert births and births[0][0] == BORN
    zone = births[0][1]
    assert zone.origin_session == OB_CANDLE          # the OB candle, again
    assert zone.born_session == BORN
    assert pye.zone_rvol(zone) >= PARAMS.min_rvol    # a real displacement


def test_births_from_bars_restores_the_first_ob_a_pruned_state_has_lost():
    """A cycle whose first OB has since died can look like it starts at a
    LATER order block in the committed state. The replay must not let the
    later one be presented as the cycle's event."""
    bars = long_cycle_bars() + [
        bar("2026-10-02", 123, 126, 121, 125),
        bar("2026-10-05", 125, 127, 120, 121),
        bar("2026-10-06", 121, 122, 115, 116),
        bar("2026-10-07", 116, 130, 115, 129, v=6000.0),  # a second OB
    ]
    births = pye.births_from_bars(bars, {"session": BREAKOUT, "level": LEVEL},
                                  PARAMS)
    assert [b for b, _z in births] == [BORN, "2026-10-07"]
    assert births[0][1].origin_session == OB_CANDLE
    assert pye.zone_rvol(births[0][1]) >= PARAMS.min_rvol


def test_births_from_bars_is_empty_before_the_cycle_s_breakout():
    assert pye.births_from_bars(long_cycle_bars(),
                                {"session": "2026-10-05", "level": LEVEL},
                                PARAMS) == []


def test_zone_rvol_reads_a_zone_object_and_a_state_event_alike():
    assert pye.zone_rvol({"rvol": 2.5}) == 2.5
    assert pye.zone_rvol({"detail": {"rvol": 3.0}}) == 3.0
    assert pye.zone_rvol(None) is None
    assert pye.zone_rvol({"rvol": None, "detail": {}}) is None


# --------------------------------------------------------------------------- #
#  The rule - exactly the description
# --------------------------------------------------------------------------- #
def test_rule_taken_entry_is_the_ob_candle_close():
    r = rule_for()
    assert r.ok
    assert r.entry_a == 110.0                  # the OB candle's close
    assert r.entry_b == 123.0                  # the displacement close
    assert r.stop == LEVEL                     # the 26W breakout level
    assert r.target_b == 124.0                 # B's own target, see below
    assert r.origin_session == OB_CANDLE
    assert r.born_session == BORN


def test_rule_target_is_the_highest_high_breakout_to_ob_candle():
    """120 was printed on 2026-09-29, between the breakout and the OB candle.
    The displacement bar's own 124 is NOT in the window - the target is set
    from the entry session backwards."""
    assert rule_for().target == 120.0


def test_rule_target_never_sits_below_the_entry():
    """The target window includes the OB candle's own high, so the sell limit
    can never sit behind the entry - the reason the backtest's median win
    completes in one session."""
    r = rule_for()
    assert r.target >= r.entry_a


def test_rule_target_b_is_the_highest_high_breakout_to_displacement():
    """Entry B's own target (the backtest's tgt_b): the highest high between
    the breakout and the DISPLACEMENT session. The displacement bar's own 124
    IS in the window, so B's sell limit always sits above B's entry - the
    reason the report's B row averages a +3.0% win."""
    r = rule_for()
    assert r.target_b == 124.0                 # the displacement bar's high
    assert r.target_b >= r.target              # the window only grows past A's
    assert r.target_b > r.entry_b              # never behind B's own entry
    bars = quiet_bars()                        # a tame displacement (high 119)
    r = evaluate_rule(bars, {"session": BREAKOUT, "level": LEVEL}, BORN, PARAMS)
    assert r.ok and r.target_b == 120.0        # the swing high, not the 119


def test_rule_excluded_rows_still_carry_both_targets():
    """The backtest's zone-stop variants measure the excluded events too, so a
    Rule rejected by a funnel row still carries its targets."""
    bars = cycle_bars()
    bars[5] = bar(OB_CANDLE, 99, 100, 95, 97)     # close below the level
    r = evaluate_rule(bars, {"session": BREAKOUT, "level": LEVEL}, BORN, PARAMS)
    assert not r.ok and r.reason == "OB close back below the 26W level"
    assert r.target == 120.0 and r.target_b == 124.0


def test_rule_walks_back_to_the_nearest_bearish_candle():
    """A green candle is not an origin: find_origin walks past it, exactly as
    the indicator does."""
    bars = cycle_bars()
    bars[5] = bar(OB_CANDLE, 108, 116, 107, 115)      # green - not an origin
    bars[4] = bar("2026-09-29", 120, 121, 116, 119)   # bearish, one bar back
    r = evaluate_rule(bars, {"session": BREAKOUT, "level": LEVEL}, BORN, PARAMS)
    assert r.ok and r.origin_session == "2026-09-29"
    assert r.entry_a == 119.0 and r.target == 121.0


def test_rule_falls_back_to_the_immediately_previous_candle():
    """The indicator's documented fallback when nothing in the 8-bar window is
    bearish or neutral - and then the level filter is what decides."""
    bars = [bar("2026-08-19", 97, 99, 96, 98), bar(BREAKOUT, 99, 103, 98, 102),
            bar("2026-09-29", 110, 112, 109, 111.5),
            bar(OB_CANDLE, 111, 113, 110, 112),
            bar(BORN, 112, 124, 111, 123)]
    r = evaluate_rule(bars, {"session": BREAKOUT, "level": LEVEL}, BORN, PARAMS)
    assert r.ok and r.origin_session == OB_CANDLE and r.entry_a == 112.0


def test_rule_excludes_an_ob_candle_that_predates_the_breakout():
    """The SMSPHARMA shape: the breakout bar IS the displacement, so the OB
    candle is the day before it. 3,450 of 7,357 replayed events."""
    r = evaluate_rule(cycle_bars(), {"session": BORN, "level": LEVEL},
                      BORN, PARAMS)
    assert not r.ok
    assert "predates the breakout" in r.reason
    assert r.origin_session == OB_CANDLE


def test_rule_excludes_an_ob_close_back_below_the_level():
    """1,629 of the replayed events: the OB candle closed back under the 26W
    level, and the rule does not take those."""
    bars = cycle_bars()
    bars[5] = bar(OB_CANDLE, 99, 100, 95, 97)         # close below the level
    r = evaluate_rule(bars, {"session": BREAKOUT, "level": LEVEL}, BORN, PARAMS)
    assert not r.ok
    assert r.reason == "OB close back below the 26W level"
    assert r.entry_a == 97.0


def test_rule_takes_a_close_exactly_at_the_level_only_above_it():
    bars = cycle_bars()
    bars[5] = bar(OB_CANDLE, 100, 101, 98, 100)       # close == level
    r = evaluate_rule(bars, {"session": BREAKOUT, "level": LEVEL}, BORN, PARAMS)
    assert not r.ok and r.reason == "OB close back below the 26W level"


def test_rule_reports_when_the_birth_is_outside_the_fetched_window():
    bars = cycle_bars()[:-1]                      # no displacement bar
    r = rule_for(bars=bars)
    assert not r.ok and "no origin candle resolved" in r.reason


# --------------------------------------------------------------------------- #
#  Fills - the backtest's gap-aware policy
# --------------------------------------------------------------------------- #
def test_fill_target_at_the_limit():
    out, px, _why, fill = pye.resolve_bar(100, 90, 120, 105, 121, 100.0)
    assert (out, px, fill) == ("win", 120.0, "order")


def test_fill_stop_at_the_stop():
    out, px, _why, fill = pye.resolve_bar(100, 90, 120, 105, 106, 89.0)
    assert (out, px, fill) == ("loss", 90.0, "order")


def test_gap_through_the_target_fills_at_the_open_better():
    out, px, _why, fill = pye.resolve_bar(100, 90, 120, 125, 130, 120.0)
    assert (out, px, fill) == ("win", 125.0, "gap-open")


def test_gap_through_the_stop_fills_at_the_open_worse():
    out, px, _why, fill = pye.resolve_bar(100, 90, 120, 85, 95, 84.0)
    assert (out, px, fill) == ("loss", 85.0, "gap-open")


def test_both_orders_touched_resolves_conservatively_as_a_loss():
    out, px, why, _fill = pye.resolve_bar(100, 90, 120, 105, 125, 85.0)
    assert out == "loss" and px == 90.0 and "conservatively" in why


def test_nothing_touched_leaves_the_trade_open():
    assert pye.resolve_bar(100, 90, 120, 100, 110, 95.0) is None


# --------------------------------------------------------------------------- #
#  Exits
# --------------------------------------------------------------------------- #
def trade(**kw):
    base = {"id": "TEST-OBE-2026-09-30", "symbol": "TEST", "entry": 110.0,
            "entry_a": 110.0, "entry_b": 123.0, "stop": LEVEL, "target": 120.0,
            "target_b": 124.0,          # B's own target: the displacement high
            "entry_session": OB_CANDLE, "last_checked": OB_CANDLE}
    base.update(kw)
    return base


def test_exit_target_hit_on_the_displacement_bar():
    """The typical win: the OB is born and its sell limit fills in the same
    session - the backtest's median win of 1 session."""
    ex = check_exit(trade(), {"as_of": BORN, "as_of_high": 124.0,
                              "as_of_low": 111.0, "as_of_close": 123.0}, 90)
    assert ex["outcome"] == "win" and ex["price"] == 120.0


def test_exit_gap_fill_uses_the_open_when_the_state_carries_one():
    ex = check_exit(trade(), {"as_of": BORN, "as_of_open": 122.0,
                              "as_of_high": 130.0, "as_of_low": 118.0,
                              "as_of_close": 128.0}, 90)
    assert ex["fill"] == "gap-open" and ex["price"] == 122.0


def test_exit_stop_hit():
    ex = check_exit(trade(), {"as_of": BORN, "as_of_high": 112.0,
                              "as_of_low": 99.0, "as_of_close": 101.0}, 90)
    assert ex["outcome"] == "loss" and ex["price"] == LEVEL


def test_exit_time_stop_at_90_sessions():
    ex = check_exit(trade(entry_session="2026-06-01",
                          last_checked="2026-06-01"),
                    {"as_of": "2026-10-06", "as_of_high": 115.0,
                     "as_of_low": 105.0, "as_of_close": 112.0}, 90)
    assert ex["outcome"] == "timeout" and "90-session" in ex["reason"]


def test_exit_none_while_nothing_filled():
    ex = check_exit(trade(), {"as_of": BORN, "as_of_high": 119.0,
                              "as_of_low": 105.0, "as_of_close": 115.0}, 90)
    assert ex is None


def test_exit_skips_a_session_already_checked():
    ex = check_exit(trade(last_checked=BORN),
                    {"as_of": BORN, "as_of_high": 200.0,
                     "as_of_low": 50.0, "as_of_close": 150.0}, 90)
    assert ex is None


def test_walk_ignores_the_entry_session_itself():
    """Entry is the OB candle's CLOSE, so nothing on or before that session
    can fill an order - even a bar that swept both levels."""
    bars = [bar(OB_CANDLE, 110, 130, 90, 110), bar(BORN, 112, 113, 111, 112)]
    assert walk_bars(trade(), bars, through=BORN,
                     time_stop_sessions=90) is None


def test_walk_finds_a_stop_that_filled_on_a_skipped_session():
    """BUG 55: the cron skipped a slot. The stop is reported with the session
    it filled on and the stop price - not carried to the next extreme."""
    bars = quiet_bars() + [bar("2026-10-02", 118, 119, 115, 116),
                           bar("2026-10-05", 115, 116, 95, 96)]
    ex = walk_bars(trade(), bars, through="2026-10-05", time_stop_sessions=90)
    assert ex["outcome"] == "loss" and ex["session"] == "2026-10-05"
    assert ex["price"] == LEVEL and ex["sessions"] == 3


def test_walk_counts_real_sessions_for_the_time_stop():
    days = [date(2026, 1, 5) + timedelta(days=i) for i in range(140)]
    bars = [bar(d.isoformat(), 102, 105, 101, 103) for d in days
            if d.weekday() < 5]
    ex = walk_bars(trade(entry_session=bars[0].session), bars,
                   through=bars[-1].session, time_stop_sessions=90)
    assert ex["outcome"] == "timeout"
    assert ex["sessions"] == 90
    assert ex["session"] == bars[90].session


def test_backfill_makes_no_data_call_on_a_normal_daily_pass(monkeypatch):
    calls = []
    monkeypatch.setattr(pye, "fetch_bars",
                        lambda *a, **k: calls.append(a) or cycle_bars())
    ctx = {"as_of": BORN, "as_of_high": 119.0, "as_of_low": 105.0,
           "as_of_close": 115.0}
    assert pye.exit_with_backfill(trade(last_checked=OB_CANDLE), ctx,
                                  object(), {"security_id": "1"},
                                  RuleSettings()) is None
    assert calls == []


def test_backfill_walks_when_a_run_was_missed(monkeypatch):
    bars = quiet_bars() + [bar("2026-10-02", 115, 116, 95, 96),
                           bar("2026-10-05", 96, 97, 95, 96)]
    monkeypatch.setattr(pye, "fetch_bars", lambda *a, **k: bars)
    ctx = {"as_of": "2026-10-05", "as_of_high": 97.0, "as_of_low": 95.0,
           "as_of_close": 96.0}
    ex = pye.exit_with_backfill(trade(last_checked=BORN), ctx, object(),
                                {"security_id": "1"}, RuleSettings())
    assert ex["outcome"] == "loss" and ex["session"] == "2026-10-02"


def test_daily_exit_spends_one_call_to_make_a_gap_fill_exact(monkeypatch):
    """The state's newest bar carries no open, so a stop touched there MIGHT
    have been gapped through - the backtest's fill policy cares. When a trade
    actually resolves, the exact open is worth one history call."""
    calls = []

    def fake_fetch(client, rec, settings, lookback_days=None):
        calls.append(rec.get("symbol"))
        return cycle_bars() + [bar("2026-10-02", 96, 97, 94, 95)]

    monkeypatch.setattr(pye, "fetch_bars", fake_fetch)
    ctx = {"as_of": "2026-10-02", "as_of_high": 99.0, "as_of_low": 94.0,
           "as_of_close": 95.0}
    ex = pye.exit_with_backfill(trade(last_checked=BORN), ctx, object(),
                                {"symbol": "TEST", "security_id": "1"},
                                RuleSettings())
    assert ex["outcome"] == "loss" and ex["fill"] == "gap-open"
    assert ex["price"] == 96.0                # the open, worse than the stop
    assert calls == ["TEST"]


def test_daily_exit_makes_no_call_when_nothing_resolved(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("no data call belongs on an unresolved pass")

    monkeypatch.setattr(pye, "fetch_bars", boom)
    ctx = {"as_of": "2026-10-02", "as_of_high": 115.0, "as_of_low": 105.0,
           "as_of_close": 112.0}
    assert pye.exit_with_backfill(trade(last_checked=BORN), ctx, object(),
                                  {"symbol": "TEST", "security_id": "1"},
                                  RuleSettings()) is None


def test_daily_exit_uses_the_states_own_open_when_it_has_one(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the state already carried the open")

    monkeypatch.setattr(pye, "fetch_bars", boom)
    ctx = {"as_of": "2026-10-02", "as_of_open": 122.0, "as_of_high": 126.0,
           "as_of_low": 118.0, "as_of_close": 124.0}
    ex = pye.exit_with_backfill(trade(last_checked=BORN), ctx, object(),
                                {"symbol": "TEST", "security_id": "1"},
                                RuleSettings())
    assert ex["outcome"] == "win" and ex["fill"] == "gap-open"
    assert ex["price"] == 122.0               # the open, better than the limit


# --------------------------------------------------------------------------- #
#  The plan
# --------------------------------------------------------------------------- #
def plan(**kw):
    base = dict(symbol="TEST", entry_a=110.0, entry_b=123.0, stop=100.0,
                target=120.0, target_b=124.0, breakout_session=BREAKOUT,
                born_session=BORN, origin_session=OB_CANDLE, capital=100000.0,
                risk_pct=1.0)
    base.update(kw)
    return TradePlan(**base)


def test_plan_math_is_the_rule_s_risk():
    p = plan()
    assert p.risk_gap == 10.0 and p.risk_amt == 1000.0
    assert p.qty == 100 and p.value == 11000.0
    assert p.rew_pct == pytest.approx(9.0909, abs=1e-3)
    assert p.rr == pytest.approx(1.0)
    assert p.risk_pct_of(123.0) == pytest.approx(18.699, abs=1e-3)


def test_plan_html_carries_both_entries_and_the_rule_s_numbers():
    html = pye.plan_html(plan())
    assert "Entry A" in html and "110.00" in html
    assert "entry b" in html.lower() and "123.00" in html
    assert "100.00" in html                       # the stop / 26W level
    assert "120.00" in html                       # A's target
    assert "124.00" in html                       # B's own target
    assert "from B: target" in html
    assert "2,272" in html and "6,624" in html    # both backtest rows
    assert "hindsight" in html


def test_plan_html_marks_a_late_entry():
    assert "2 session(s) old" in pye.plan_html(plan(late_sessions=2))


def test_plan_html_says_stop_too_far_rather_than_sizing_nonsense():
    p = plan(entry_a=110.0, stop=40.0, capital=1000.0, risk_pct=1.0)
    html = pye.plan_html(p)
    assert "stop too far" in html and p.qty == 0


def test_exit_html_reports_net_from_a_and_never_quotes_a_stale_b():
    """A resolves ON the born session - B buys that same close, so B has not
    held a session and must not be given A's exit price as its own result."""
    ex = {"outcome": "win", "session": BORN, "price": 120.0, "fill": "order",
          "reason": "swing-high target filled", "sessions": 1}
    html = pye.exit_html(trade(born_session=BORN), ex, RuleSettings())
    assert "TARGET HIT" in html
    assert "+8.87%" in html         # from A: 120/110-1 = +9.09%, less 0.22
    assert "from B" not in html
    assert "B (displacement close) still open" in html


def test_exit_html_quotes_b_from_b_s_own_walk():
    ex = {"outcome": "win", "session": "2026-10-05", "price": 120.0,
          "fill": "order", "reason": "swing-high target filled", "sessions": 4,
          "b": {"outcome": "win", "session": "2026-10-05", "price": 120.0,
                "sessions": 3}}
    html = pye.exit_html(trade(born_session=BORN), ex, RuleSettings())
    assert "+8.87%" in html and "-2.66%" in html   # B: 120/123 less 0.22


def test_exit_html_dates_b_when_it_resolved_on_another_session():
    ex = {"outcome": "loss", "session": "2026-10-05", "price": 100.0,
          "fill": "order", "reason": "26W level stop filled", "sessions": 4,
          "b": {"outcome": "loss", "session": "2026-10-06", "price": 100.0,
                "sessions": 2}}
    html = pye.exit_html(trade(born_session=BORN), ex, RuleSettings())
    assert "from B -18.92% 2026-10-06" in html


def test_late_html_is_a_notice_not_a_plan():
    t = trade(late_sessions=2)
    t["born_session"], t["origin_session"] = BORN, OB_CANDLE
    ex = {"outcome": "win", "session": BORN, "price": 120.0, "fill": "order",
          "reason": "swing-high target filled", "sessions": 1}
    html = pye.late_html(t, ex, RuleSettings())
    assert "MISSED" in html and "No action" in html
    assert "Entry A" not in html
    assert "B (displacement close) still open" in html


def test_b_walk_is_one_session_behind_a_and_reports_its_own_exit():
    """A exits on the born session (the displacement gaps through the target);
    B entered at that very close, so the same bars leave B still open until
    its own session resolves. B is also walked against its OWN target (the
    displacement bar's high, 124) - not A's 120 - so its win is its own fill
    at its own limit, not A's gapped-through price."""
    bars = cycle_bars() + [bar("2026-10-05", 121, 125, 120, 124)]
    t = trade(born_session=BORN, origin_session=OB_CANDLE)
    a = walk_bars(t, bars, through="2026-10-05", time_stop_sessions=90)
    assert a is not None and a["session"] == BORN and a["outcome"] == "win"
    assert pye.b_walk(t, a, bars, RuleSettings()) == {"open": True}   # to A
    b = pye.b_walk(t, a, bars, RuleSettings(), through="2026-10-05")
    assert b["outcome"] == "win" and b["session"] == "2026-10-05"
    assert b["price"] == 124.0                    # B's own target, at the limit
    assert b["fill"] == "order"
    b_no_bars = pye.b_walk(t, a, None, RuleSettings())
    assert b_no_bars is None                      # nothing to walk, nothing said
    assert pye.b_walk(dict(t, entry_b=None), a, bars, RuleSettings()) is None


def test_b_walk_rebuilds_b_s_target_for_a_trade_recorded_without_one():
    """A trade recorded before B carried its own target is healed from the
    bars: the highest high between the breakout and the displacement bar -
    the report's own construction for entry B."""
    bars = cycle_bars() + [bar("2026-10-05", 121, 125, 120, 124)]
    t = trade(born_session=BORN, origin_session=OB_CANDLE)
    del t["target_b"]                       # the pre-fix state shape
    a = walk_bars(t, bars, through="2026-10-05", time_stop_sessions=90)
    b = pye.b_walk(t, a, bars, RuleSettings(), through="2026-10-05")
    assert b["outcome"] == "win" and b["price"] == 124.0


def test_b_walk_leaves_b_open_when_it_has_not_resolved():
    bars = cycle_bars() + [bar("2026-10-05", 119, 119, 115, 118)]
    t = trade(born_session=BORN, origin_session=OB_CANDLE)
    a = {"outcome": "win", "session": BORN, "price": 120.0, "fill": "order"}
    assert pye.b_walk(t, a, bars, RuleSettings()) == {"open": True}


# --------------------------------------------------------------------------- #
#  Forming (intraday) detection
# --------------------------------------------------------------------------- #
QUOTE = {"open": 112.0, "high": 124.0, "low": 111.0, "last_price": 123.0,
         "volume": 5000.0}


def test_live_displacement_matches_the_indicator_s_thresholds():
    st = state()
    d = pye.live_displacement(QUOTE, st["zones"]["TEST"], PARAMS)
    assert d and d["rvol"] > 1.8 and d["clv"] > 0.72


def test_live_displacement_rejects_a_bar_below_the_structure():
    st = state()
    q = dict(QUOTE, last_price=89.0, low=88.0)
    assert pye.live_displacement(q, st["zones"]["TEST"], PARAMS) is None


def test_forming_origin_uses_the_same_find_origin_as_the_indicator():
    bars = cycle_bars()[:-1]                       # everything but the birth
    origin, live = pye.forming_origin(bars, QUOTE, BORN, PARAMS)
    assert origin.session == OB_CANDLE and origin.close == 110.0
    assert live.session == BORN and live.close == 123.0


def test_quote_is_stale_matches_the_scanner_s_holiday_test():
    ctx = {"as_of_high": 124.0, "as_of_low": 111.0, "as_of_close": 123.0}
    assert pye.quote_is_stale(dict(QUOTE), ctx)
    assert not pye.quote_is_stale(dict(QUOTE, last_price=124.0), ctx)


# --------------------------------------------------------------------------- #
#  Integration: postclose (synthetic state files, no network)
# --------------------------------------------------------------------------- #
class FakeTG:
    def __init__(self):
        self.sent = []

    def send(self, msg):
        self.sent.append(msg)
        return True


class FailedTG(FakeTG):
    def send(self, msg):
        super().send(msg)
        return False


class Args:
    def __init__(self, **kw):
        self.today = None
        self.symbols = None
        self.capital = 100000.0
        self.risk_pct = 1.0
        self.no_data = False
        self.dry_run = False
        for k, v in kw.items():
            setattr(self, k, v)


def fake_cfg(**entry):
    return SimpleNamespace(
        secrets=SimpleNamespace(dhan_client_id="x", dhan_access_token="y"),
        runtime=SimpleNamespace(data_rate_per_sec=1, quote_rate_per_sec=1,
                                dry_run=False),
        ob_precision=SimpleNamespace(params=lambda: PARAMS),
        precision_ob_entry=SimpleNamespace(**entry),
    )


def run_pc(tmp_path, monkeypatch, st_in, own=None, cfg=None, tg=None,
           bars=None, **kw):
    p_in = tmp_path / "ob_precision_state.json"
    p_own = tmp_path / "precision_ob_entry_state.json"
    p_in.write_text(json.dumps(st_in))
    if own is not None:
        p_own.write_text(json.dumps(own))
    monkeypatch.setattr(pye, "STATE_IN", p_in)
    monkeypatch.setattr(pye, "STATE_OWN", p_own)
    monkeypatch.setattr(pye, "DhanClient", lambda *a, **k: object())
    monkeypatch.setattr(pye, "fetch_bars",
                        lambda *a, **k: cycle_bars() if bars is None else bars)
    tg = tg or FakeTG()
    rc = pye.run_postclose(cfg or fake_cfg(), Args(**kw), tg)
    saved = json.loads(p_own.read_text()) if p_own.exists() else {}
    return rc, tg, saved


def test_postclose_fires_on_today_s_ob_birth_and_records_the_trade(
        tmp_path, monkeypatch):
    st = state(events=[ob_event()], as_of=BORN, as_of_high=119.0,
               as_of_low=111.0, as_of_close=118.0)
    rc, tg, saved = run_pc(tmp_path, monkeypatch, st, today=BORN)
    assert rc == 0
    assert len(tg.sent) == 1
    assert "PRECISION OB ENTRY" in tg.sent[0]
    assert len(saved["open"]) == 1
    t = saved["open"][0]
    assert t["entry_a"] == 110.0 and t["entry_b"] == 123.0
    assert t["stop"] == LEVEL and t["target"] == 120.0
    assert t["target_b"] == 124.0    # B's own target: the displacement high
    assert t["entry_session"] == OB_CANDLE      # the OB candle's session
    assert t["breakout_session"] == BREAKOUT
    assert f"rule|TEST|{BORN}" in saved["sent"]


def test_postclose_also_records_the_win_the_displacement_bar_delivered(
        tmp_path, monkeypatch):
    """The backtest's median win completes in one session: the sell limit
    usually fills on the very bar that births the OB. The plan goes out, and
    so does the exit - the book must not pretend the trade is still open."""
    st = state(events=[ob_event()], as_of=BORN, as_of_high=124.0,
               as_of_low=111.0, as_of_close=123.0)
    rc, tg, saved = run_pc(tmp_path, monkeypatch, st, today=BORN)
    assert rc == 0
    assert saved["open"] == [] and len(saved["closed"]) == 1
    assert saved["closed"][0]["exit"]["outcome"] == "win"
    assert saved["closed"][0]["exit"]["session"] == BORN
    assert any("TARGET HIT" in m for m in tg.sent)


def test_postclose_never_replays_history_on_a_first_run(tmp_path, monkeypatch):
    """Deploying the job must not dump weeks of old events into the chat: a
    birth older than the catch-up window is marked seen, silently."""
    st = state(events=[ob_event(born="2026-09-10")])
    rc, tg, saved = run_pc(tmp_path, monkeypatch, st, today=BORN)
    assert rc == 0 and tg.sent == []
    assert "rule|TEST|2026-09-10" in saved["sent"]


def test_postclose_alerts_a_birth_inside_the_catchup_window(
        tmp_path, monkeypatch):
    bars = quiet_bars()          # a tame displacement: the trade is still open
    st = state(events=[ob_event(born="2026-09-30", origin="2026-09-29")])
    rc, tg, saved = run_pc(tmp_path, monkeypatch, st, bars=bars, today=BORN)
    assert rc == 0 and len(tg.sent) == 1
    assert "1 session(s) old" in tg.sent[0]
    assert len(saved["open"]) == 1


def test_postclose_reports_a_late_event_that_already_resolved(
        tmp_path, monkeypatch):
    """Two sessions late and the trade already hit its target: the alert must
    say so instead of quoting a plan, and the book keeps the audit record."""
    bars = cycle_bars() + [bar("2026-10-02", 112, 113, 111, 112)]
    st = state(events=[ob_event(born="2026-09-30", origin="2026-09-29")],
               as_of=BORN, as_of_high=124.0, as_of_low=111.0,
               as_of_close=123.0)
    rc, tg, saved = run_pc(tmp_path, monkeypatch, st, bars=bars,
                           today="2026-10-02")
    assert rc == 0
    assert any("MISSED" in m for m in tg.sent)
    assert not any("Entry A" in m for m in tg.sent)
    assert len(saved["closed"]) == 1
    assert saved["closed"][0]["exit"]["outcome"] == "win"
    assert saved["open"] == []


def test_postclose_records_an_excluded_rule_event_without_alerting(
        tmp_path, monkeypatch):
    bars = cycle_bars()
    bars[5] = bar(OB_CANDLE, 99, 100, 95, 97)     # close back below the level
    st = state(events=[ob_event()], as_of=BORN, as_of_high=119.0,
               as_of_low=111.0, as_of_close=118.0)
    rc, tg, saved = run_pc(tmp_path, monkeypatch, st, bars=bars, today=BORN)
    assert rc == 0 and tg.sent == []
    assert saved["excluded"][f"TEST|{BORN}"] == \
        "OB close back below the 26W level"
    assert saved["open"] == []


def test_postclose_defers_not_forgets_when_bars_are_unavailable(
        tmp_path, monkeypatch):
    """A data outage must not burn the event: the key is left unmarked so the
    next run evaluates the same birth."""
    st = state(events=[ob_event()])
    rc, tg, saved = run_pc(tmp_path, monkeypatch, st, bars=None, today=BORN,
                           no_data=True)
    assert rc == 0 and tg.sent == []
    assert f"rule|TEST|{BORN}" not in saved.get("sent", {})


def test_postclose_does_not_persist_state_when_telegram_fails(
        tmp_path, monkeypatch):
    st = state(events=[ob_event()], as_of=BORN, as_of_high=119.0,
               as_of_low=111.0, as_of_close=118.0)
    original = {"sent": {}, "open": [], "closed": [], "excluded": {},
                "last_run": "before"}
    rc, tg, saved = run_pc(tmp_path, monkeypatch, st, own=original,
                           tg=FailedTG(), today=BORN)
    assert rc == 1 and len(tg.sent) == 1
    assert saved == original        # the next pass retries the same event


def test_postclose_dry_run_persists_nothing(tmp_path, monkeypatch):
    st = state(events=[ob_event()], as_of=BORN, as_of_high=119.0,
               as_of_low=111.0, as_of_close=118.0)
    original = {"sent": {}, "open": [], "closed": [], "excluded": {},
                "last_run": "before"}
    rc, tg, saved = run_pc(tmp_path, monkeypatch, st, own=original,
                           dry_run=True, today=BORN)
    assert rc == 0 and len(tg.sent) == 1
    assert saved == original


def test_postclose_resolves_an_older_open_trade(tmp_path, monkeypatch):
    st = state(as_of="2026-10-05", as_of_high=119.0, as_of_low=95.0,
               as_of_close=96.0)
    own = {"sent": {}, "closed": [], "excluded": {}, "open": [
        {"id": "TEST-OBE-2026-09-30", "symbol": "TEST", "entry": 110.0,
         "entry_a": 110.0, "entry_b": 123.0, "stop": LEVEL, "target": 120.0,
         "entry_session": OB_CANDLE, "last_checked": "2026-10-02",
         "breakout_session": BREAKOUT},
    ]}
    rc, tg, saved = run_pc(tmp_path, monkeypatch, st, own=own,
                           today="2026-10-05")
    assert rc == 0 and saved["open"] == [] and len(saved["closed"]) == 1
    assert saved["closed"][0]["exit"]["outcome"] == "loss"
    assert any("STOPPED OUT" in m for m in tg.sent)


def test_postclose_bar_walks_a_slot_the_cron_skipped(tmp_path, monkeypatch):
    """last checked on the birth, the state now two sessions later: the walk
    must report the stop on the session it actually filled on."""
    bars = quiet_bars() + [bar("2026-10-02", 115, 116, 95, 96),
                           bar("2026-10-05", 96, 97, 95, 96)]
    st = state(as_of="2026-10-05", as_of_high=97.0, as_of_low=95.0,
               as_of_close=96.0)
    own = {"sent": {}, "closed": [], "excluded": {}, "open": [
        {"id": "TEST-OBE-2026-09-30", "symbol": "TEST", "entry": 110.0,
         "entry_a": 110.0, "entry_b": 123.0, "stop": LEVEL, "target": 120.0,
         "entry_session": OB_CANDLE, "last_checked": BORN,
         "breakout_session": BREAKOUT},
    ]}
    rc, tg, saved = run_pc(tmp_path, monkeypatch, st, own=own, bars=bars,
                           today="2026-10-05")
    assert rc == 0 and len(saved["closed"]) == 1
    ex = saved["closed"][0]["exit"]
    assert ex["outcome"] == "loss" and ex["session"] == "2026-10-02"


def test_postclose_exit_carries_b_s_own_result(tmp_path, monkeypatch):
    """B is the fillable entry, so the book must state B's own outcome: A
    resolved on the born session, B only one session later."""
    bars = quiet_bars() + [bar("2026-10-05", 121, 125, 120, 124)]
    st = state(as_of="2026-10-05", as_of_high=125.0, as_of_low=120.0,
               as_of_close=124.0)
    own = {"sent": {}, "closed": [], "excluded": {}, "open": [
        {"id": "TEST-OBE-2026-09-30", "symbol": "TEST", "entry": 110.0,
         "entry_a": 110.0, "entry_b": 123.0, "stop": LEVEL, "target": 120.0,
         "entry_session": OB_CANDLE, "born_session": BORN,
         "last_checked": "2026-10-02", "breakout_session": BREAKOUT},
    ]}
    rc, tg, saved = run_pc(tmp_path, monkeypatch, st, own=own, bars=bars,
                           today="2026-10-05")
    assert rc == 0 and len(saved["closed"]) == 1
    ex = saved["closed"][0]["exit"]
    assert ex["outcome"] == "win" and ex["session"] == "2026-10-05"
    assert ex["b"]["outcome"] == "win" and ex["b"]["price"] == 121.0
    msg = [m for m in tg.sent if "TARGET HIT" in m][0]
    # This trade predates target_b, so b_walk rebuilds it from the bars: the
    # highest high between the breakout and the born session is the swing high
    # 120 (the tame displacement bar printed only 119). B bought 123 and that
    # target is below B's own entry, so B's "win" is still a net loss -
    # quoted from B's own walk, not reused from A's price.
    assert "from B -1.85%" in msg and "still open" not in msg


def test_one_malformed_trade_cannot_kill_the_run(tmp_path, monkeypatch):
    st = state(as_of="2026-10-05", as_of_high=119.0, as_of_low=95.0,
               as_of_close=96.0)
    own = {"sent": {}, "closed": [], "excluded": {}, "open": [
        {"id": "MALFORMED"},
        {"id": "TEST-OBE-2026-09-30", "symbol": "TEST", "entry": 110.0,
         "entry_a": 110.0, "entry_b": 123.0, "stop": LEVEL, "target": 120.0,
         "entry_session": OB_CANDLE, "last_checked": "2026-10-02",
         "breakout_session": BREAKOUT},
    ]}
    rc, _tg, saved = run_pc(tmp_path, monkeypatch, st, own=own,
                            today="2026-10-05")
    assert rc == 0 and len(saved["closed"]) == 1
    assert [t["id"] for t in saved["open"]] == ["MALFORMED"]


# --------------------------------------------------------------------------- #
#  Integration: intraday (one bulk quote, synthetic)
# --------------------------------------------------------------------------- #
def run_id(tmp_path, monkeypatch, st_in, own=None, cfg=None, tg=None,
           quote=None, bars=None, **kw):
    p_in = tmp_path / "ob_precision_state.json"
    p_own = tmp_path / "precision_ob_entry_state.json"
    p_in.write_text(json.dumps(st_in))
    if own is not None:
        p_own.write_text(json.dumps(own))
    monkeypatch.setattr(pye, "STATE_IN", p_in)
    monkeypatch.setattr(pye, "STATE_OWN", p_own)
    q = dict(QUOTE)
    q.update(quote or {})

    class C:
        def __init__(self, *a, **k):
            pass

        def ohlc(self, request):
            return {"NSE_EQ": {"1": q}}

    monkeypatch.setattr(pye, "DhanClient", C)
    monkeypatch.setattr(pye, "fetch_bars",
                        lambda *a, **k: cycle_bars()[:-1]
                        if bars is None else bars)
    tg = tg or FakeTG()
    rc = pye.run_intraday(cfg or fake_cfg(), Args(**kw), tg)
    saved = json.loads(p_own.read_text()) if p_own.exists() else {}
    return rc, tg, saved


def test_intraday_forming_alert_carries_the_fillable_entry_b(tmp_path,
                                                             monkeypatch):
    """The only moment entry B is still open: the OB is being born on today's
    bar, and the OB candle is already resolved - so the rule's structural rows
    are checked for real, not guessed."""
    st = state(as_of=OB_CANDLE, as_of_high=116.0, as_of_low=108.0,
               as_of_close=110.0, refreshed_on=BORN)
    rc, tg, saved = run_id(tmp_path, monkeypatch, st, today=BORN)
    assert rc == 0 and len(tg.sent) == 1
    msg = tg.sent[0]
    assert "PRECISION OB FORMING" in msg
    assert "123.00" in msg                       # entry B = today's close
    assert "110.00" in msg                       # entry A = the OB candle
    assert f"forming|TEST|{BORN}" in saved["sent"]


def test_intraday_forming_target_excludes_the_displacement_bar(tmp_path,
                                                               monkeypatch):
    """The rule's target is the highest high between the breakout and the OB
    candle. Today's 124 belongs to the displacement bar and is NOT in it -
    but it IS in B's own target, whose entry session is today."""
    st = state(as_of=OB_CANDLE, as_of_high=116.0, as_of_low=108.0,
               as_of_close=110.0, refreshed_on=BORN)
    _rc, tg, _saved = run_id(tmp_path, monkeypatch, st, today=BORN)
    assert "🎯 Target <b>120.00</b>" in tg.sent[0]
    assert "from B: target <b>124.00</b>" in tg.sent[0]


def test_intraday_silent_when_the_ob_candle_is_below_the_level(tmp_path,
                                                               monkeypatch):
    bars = cycle_bars()[:-1]
    bars[-1] = bar(OB_CANDLE, 99, 100, 95, 97)
    st = state(as_of=OB_CANDLE, as_of_high=100.0, as_of_low=95.0,
               as_of_close=97.0, refreshed_on=BORN)
    rc, tg, saved = run_id(tmp_path, monkeypatch, st, bars=bars, today=BORN)
    assert rc == 0 and tg.sent == []
    assert "forming|TEST" not in json.dumps(saved.get("sent", {}))


def test_intraday_silent_when_the_ob_candle_predates_the_breakout(
        tmp_path, monkeypatch):
    """SMSPHARMA: the breakout bar itself is the displacement, so the origin
    is the day before. The rule excludes it; so does the forming alert."""
    st = state(brk=BORN, as_of=OB_CANDLE, as_of_high=116.0, as_of_low=108.0,
               as_of_close=110.0, refreshed_on=BORN)
    rc, tg, _saved = run_id(tmp_path, monkeypatch, st, today=BORN)
    assert rc == 0 and tg.sent == []


def test_intraday_silent_when_the_cycle_s_ob_already_exists(tmp_path,
                                                            monkeypatch):
    st = state(events=[ob_event(born="2026-09-10")],
               as_of=OB_CANDLE, as_of_high=116.0, as_of_low=108.0,
               as_of_close=110.0, refreshed_on=BORN)
    rc, tg, _saved = run_id(tmp_path, monkeypatch, st, today=BORN)
    assert rc == 0 and tg.sent == []


def test_intraday_silent_on_a_stale_holiday_quote(tmp_path, monkeypatch):
    """A weekday holiday re-serves the last closed session's numbers - which
    would read as a displacement that is not forming."""
    st = state(as_of=OB_CANDLE, as_of_high=124.0, as_of_low=111.0,
               as_of_close=123.0, refreshed_on=BORN)
    rc, tg, _saved = run_id(tmp_path, monkeypatch, st, today=BORN)
    assert rc == 0 and tg.sent == []


def test_intraday_silent_when_the_scanner_did_not_refresh_today(
        tmp_path, monkeypatch):
    st = state(as_of=OB_CANDLE, as_of_high=116.0, as_of_low=108.0,
               as_of_close=110.0, refreshed_on=OB_CANDLE)
    rc, tg, _saved = run_id(tmp_path, monkeypatch, st, today=BORN)
    assert rc == 0 and tg.sent == []


def test_intraday_does_not_persist_on_telegram_failure(tmp_path, monkeypatch):
    st = state(as_of=OB_CANDLE, as_of_high=116.0, as_of_low=108.0,
               as_of_close=110.0, refreshed_on=BORN)
    p_in = tmp_path / "ob_precision_state.json"
    p_own = tmp_path / "precision_ob_entry_state.json"
    p_in.write_text(json.dumps(st))
    monkeypatch.setattr(pye, "STATE_IN", p_in)
    monkeypatch.setattr(pye, "STATE_OWN", p_own)
    monkeypatch.setattr(pye, "DhanClient",
                        lambda *a, **k: SimpleNamespace(
                            ohlc=lambda req: {"NSE_EQ": {"1": dict(QUOTE)}}))
    monkeypatch.setattr(pye, "fetch_bars", lambda *a, **k: cycle_bars()[:-1])
    rc = pye.run_intraday(fake_cfg(), Args(today=BORN), FailedTG())
    assert rc == 1 and not p_own.exists()


# --------------------------------------------------------------------------- #
#  Candidate mode (opt-in) and the off switches
# --------------------------------------------------------------------------- #
RED_QUOTE = {"open": 120.0, "high": 121.0, "low": 112.0, "last_price": 113.0,
             "volume": 900.0}


def test_candidate_mode_is_off_by_default(tmp_path, monkeypatch):
    """A red candle above the level is not the rule's event - the order block
    does not exist until a displacement follows."""
    st = state(as_of=OB_CANDLE, as_of_high=116.0, as_of_low=108.0,
               as_of_close=110.0, refreshed_on=BORN)
    rc, tg, _saved = run_id(tmp_path, monkeypatch, st, quote=RED_QUOTE,
                            today=BORN)
    assert rc == 0 and tg.sent == []


def test_candidate_mode_warns_before_a_displacement_exists(tmp_path,
                                                           monkeypatch):
    st = state(as_of=OB_CANDLE, as_of_high=116.0, as_of_low=108.0,
               as_of_close=110.0, refreshed_on=BORN)
    rc, tg, saved = run_id(tmp_path, monkeypatch, st,
                           cfg=fake_cfg(candidate_alerts=True),
                           quote=RED_QUOTE, today=BORN)
    assert rc == 0 and len(tg.sent) == 1
    assert "OB CANDIDATE" in tg.sent[0] and "Speculative" in tg.sent[0]
    assert "113.00" in tg.sent[0]               # entry A = today's close
    assert f"candidate|TEST|{BORN}" in saved["sent"]


def test_candidate_mode_never_fires_on_a_green_candle(tmp_path, monkeypatch):
    st = state(as_of=OB_CANDLE, as_of_high=116.0, as_of_low=108.0,
               as_of_close=110.0, refreshed_on=BORN)
    rc, tg, _saved = run_id(tmp_path, monkeypatch, st,
                            cfg=fake_cfg(candidate_alerts=True),
                            quote={"open": 105.0, "high": 108.0, "low": 104.0,
                                   "last_price": 107.5, "volume": 900.0},
                            today=BORN)
    assert rc == 0 and tg.sent == []


def test_intraday_is_a_noop_when_both_alert_switches_are_off(tmp_path,
                                                             monkeypatch):
    p_own = tmp_path / "precision_ob_entry_state.json"
    monkeypatch.setattr(pye, "STATE_OWN", p_own)
    cfg = fake_cfg(forming_alerts=False, candidate_alerts=False)
    assert pye.run_intraday(cfg, Args(today=BORN), FakeTG()) == 0
    assert not p_own.exists()          # off means off, not "ran and saved"


def test_confirm_alerts_off_marks_events_seen_without_sending(tmp_path,
                                                              monkeypatch):
    st = state(events=[ob_event()], as_of=BORN, as_of_high=119.0,
               as_of_low=111.0, as_of_close=118.0)
    rc, tg, saved = run_pc(tmp_path, monkeypatch, st,
                           cfg=fake_cfg(confirm_alerts=False), today=BORN)
    assert rc == 0 and tg.sent == []
    assert f"rule|TEST|{BORN}" in saved["sent"]


# --------------------------------------------------------------------------- #
#  The rule as the description states it - pinned in one place
# --------------------------------------------------------------------------- #
def test_the_rule_is_the_description():
    r = rule_for()
    settings = RuleSettings()
    assert (r.ok, r.entry_a, r.stop, r.target, r.target_b,
            settings.time_stop_sessions) == (True, 110.0, LEVEL, 120.0, 124.0,
                                             90)
    assert settings.round_trip_cost_pct == 0.22
    # the fill policy, in the description's own words
    assert pye.resolve_bar(100, 90, 120, 125, 130, 120)[:2] == ("win", 125.0)
    assert pye.resolve_bar(100, 90, 120, 85, 95, 84)[:2] == ("loss", 85.0)
    assert pye.resolve_bar(100, 90, 120, 105, 125, 85)[0] == "loss"


def test_backtest_rows_are_quoted_verbatim():
    from precision_ob_entry import STATS_A, STATS_B
    for token in ("2,272", "88.5%", "+5.77%", "+1.62R"):
        assert token in STATS_A
    for token in ("6,624", "83%", "+1.32%", "+0.60R"):
        assert token in STATS_B


# --------------------------------------------------------------------------- #
#  Window, pre-confirmation resolution, size capping, quiet run audit tests
# --------------------------------------------------------------------------- #
def test_window_is_ob_candle_age_suppresses_stale_candle(tmp_path, monkeypatch):
    """Window = the OB candle's age, not the confirmation bar's: a zone born
    yesterday whose OB candle was 4 sessions ago (> catchup_sessions 3) is
    suppressed as stale, not alerted."""
    bars = cycle_bars()
    st = state(events=[ob_event(born="2026-10-06", origin="2026-10-01")],
               as_of="2026-10-06", as_of_high=119.0, as_of_low=111.0, as_of_close=118.0)
    rc, tg, saved = run_pc(tmp_path, monkeypatch, st, bars=bars, today="2026-10-07")
    assert rc == 0 and tg.sent == []
    assert "rule|TEST|2026-10-06" in saved["sent"]


def test_stale_ob_candle_counted_in_postclose_summary(tmp_path, monkeypatch, caplog):
    """A stale OB candle is counted under 'stale OB candle' in the run summary."""
    import logging
    # cycle_bars has origin OB_CANDLE = "2026-09-30" and born BORN = "2026-10-01".
    # As of today = "2026-10-08", OB_CANDLE is 6 sessions old (> 3).
    st = state(events=[ob_event(born=BORN, origin=OB_CANDLE)],
               as_of=BORN, as_of_high=119.0, as_of_low=111.0, as_of_close=118.0)
    with caplog.at_level(logging.INFO):
        rc, tg, _saved = run_pc(tmp_path, monkeypatch, st, bars=cycle_bars(), today="2026-10-08")
    assert rc == 0 and tg.sent == []
    assert "1 stale OB candle" in caplog.text


def test_cheap_gate_skips_stale_ob_candle_before_data_fetch(tmp_path, monkeypatch):
    """The cheap gate checks origin_session if present, skipping data fetch for very old candles."""
    st = state(events=[ob_event(born="2026-09-20", origin="2026-09-15")],
               as_of="2026-09-20", as_of_high=119.0, as_of_low=111.0, as_of_close=118.0)
    rc, tg, saved = run_pc(tmp_path, monkeypatch, st, today="2026-10-07")
    assert rc == 0 and tg.sent == []
    assert "rule|TEST|2026-09-20" in saved["sent"]


def test_walk_bars_flags_pre_confirmation_resolution():
    """A stop or target hit before the confirmation bar is flagged pre_confirmation."""
    t = trade(born_session="2026-10-06", origin_session="2026-10-01", entry_session="2026-10-01")
    t["stop"] = 263.80
    t["target"] = 280.00
    t["entry"] = 264.00
    bars = [
        bar("2026-10-01", 263.0, 265.0, 262.5, 264.0),
        bar("2026-10-02", 264.0, 266.0, 263.9, 265.0),
        bar("2026-10-05", 265.0, 265.5, 262.0, 263.5),
        bar("2026-10-06", 263.5, 270.0, 263.0, 269.0),
    ]
    res = pye.walk_bars(t, bars, through="2026-10-06", time_stop_sessions=90)
    assert res is not None
    assert res["session"] == "2026-10-05"
    assert res.get("pre_confirmation") is True
    assert res.get("never_live") is True


def test_pre_confirmation_resolution_in_exit_html_is_book_record():
    """A resolution before the confirmation bar is reported as BOOK RECORD with 'no position was ever live'."""
    t = trade(born_session="2026-10-06", origin_session="2026-10-01")
    ex = {"outcome": "loss", "session": "2026-10-05", "price": 263.80, "fill": "order",
          "reason": "26W level stop filled", "pre_confirmation": True}
    html = pye.exit_html(t, ex, RuleSettings())
    assert "BOOK RECORD" in html
    assert "STOPPED OUT" not in html
    assert "no position was ever live" in html


def test_pre_confirmation_resolution_in_missed_notice():
    """A missed notice for a pre-confirmation resolution states 'no position was ever live'."""
    t = trade(born_session="2026-10-06", origin_session="2026-10-01")
    ex = {"outcome": "loss", "session": "2026-10-05", "price": 263.80, "fill": "order",
          "reason": "26W level stop filled", "pre_confirmation": True}
    html = pye.late_html(t, ex, RuleSettings())
    assert "MISSED" in html
    assert "BOOK RECORD" in html
    assert "no position was ever live" in html


def test_position_size_capped_by_capital():
    """Tight stop gap (e.g. 0.076%) risk-sizes 4,999 shares but is capped by capital to 378 shares."""
    p = pye.TradePlan(
        symbol="TEST",
        entry_a=264.0,
        entry_b=None,
        stop=263.80,
        target=280.0,
        breakout_session="2026-09-10",
        born_session="2026-10-06",
        origin_session="2026-10-01",
        capital=100000.0,
        risk_pct=1.0,
    )
    assert p.risk_qty == 5000
    assert p.cap_qty == 378
    assert p.is_capped is True
    assert p.qty == 378
    assert p.value <= 100000.0


def test_plan_html_prints_size_capped_reason():
    """plan_html prints '(capped by capital...' when size is capped."""
    p = pye.TradePlan(
        symbol="TEST",
        entry_a=264.0,
        entry_b=None,
        stop=263.80,
        target=280.0,
        breakout_session="2026-09-10",
        born_session="2026-10-06",
        origin_session="2026-10-01",
        capital=100000.0,
        risk_pct=1.0,
    )
    html = pye.plan_html(p)
    assert "capped by capital" in html
    assert "378 shares" in html


def test_messages_lead_with_ob_candle_and_age():
    """Messages lead with the OB candle and its age."""
    p = plan(late_sessions=2)
    html = pye.plan_html(p)
    lines = html.splitlines()
    assert "OB candle" in lines[1]
    assert "2 session(s) old" in lines[1]
    assert "displacement bar" in lines[1]

    t = trade(born_session=BORN, origin_session=OB_CANDLE, late_sessions=1)
    ex = {"outcome": "win", "session": BORN, "price": 120.0, "fill": "order",
          "reason": "swing-high target filled"}
    m_html = pye.late_html(t, ex, RuleSettings())
    assert f"OB candle <b>{OB_CANDLE}</b> (1 session(s) old)" in m_html


def test_quiet_run_summary_breakdown(tmp_path, monkeypatch, caplog):
    """A quiet run logs why: events breakdown with total, handled, deferred, etc."""
    import logging
    st = state(events=[ob_event(born="2026-09-10")], as_of=BORN)
    with caplog.at_level(logging.INFO):
        rc, tg, _saved = run_pc(tmp_path, monkeypatch, st, today=BORN)
    assert rc == 0
    assert "events: 1 total" in caplog.text
    assert "already handled" in caplog.text


def test_paragmilk_and_stltech_shapes_end_to_end(tmp_path, monkeypatch):
    """End-to-end audit: PARAGMILK and STLTECH shapes with older OB candles are suppressed;
    pre-confirmation stops are treated as book records stating 'no position was ever live'."""
    st_stl = state(sym="STLTECH", events=[ob_event(born="2026-10-05", origin="2026-10-01")],
                   as_of="2026-10-05", as_of_high=1003.0, as_of_low=932.0, as_of_close=1001.0)
    rc, tg, saved = run_pc(tmp_path, monkeypatch, st_stl, bars=cycle_bars(), today="2026-10-07")
    assert rc == 0 and tg.sent == []
    assert "rule|STLTECH|2026-10-05" in saved["sent"]

    t = {"id": "PARAGMILK-OBE-2026-10-06", "symbol": "PARAGMILK", "strategy": "precision_ob_entry",
         "entry": 264.0, "entry_a": 264.0, "stop": 263.80, "target": 286.0,
         "born_session": "2026-10-06", "origin_session": "2026-10-01",
         "entry_session": "2026-10-01", "sessions_held": 0, "late_sessions": 1}
    p_bars = [
        bar("2026-10-01", 263.0, 265.0, 262.5, 264.0),
        bar("2026-10-02", 264.0, 266.0, 263.9, 265.0),
        bar("2026-10-05", 265.0, 265.5, 262.0, 263.5),
        bar("2026-10-06", 263.5, 270.0, 263.0, 269.0),
    ]
    ex = pye.walk_bars(t, p_bars, through="2026-10-06", time_stop_sessions=90)
    assert ex is not None
    assert ex["session"] == "2026-10-05"
    assert ex.get("pre_confirmation") is True
    exit_msg = pye.exit_html(t, ex, RuleSettings())
    assert "BOOK RECORD" in exit_msg
    assert "no position was ever live" in exit_msg


# --------------------------------------------------------------------------- #
#  08-Oct-2026 audit regressions
#
#  The job ran 150/150 green and delivered two messages, but both were "MISSED"
#  notices and the book never held a live trade. Three of the reasons were bugs
#  rather than the rule being strict, and each one is pinned here.
# --------------------------------------------------------------------------- #
IST = ZoneInfo("Asia/Kolkata")


def test_session_date_never_names_an_incomplete_session():
    """`session_date` is the last COMPLETED session, not `now().date()`.

    The workflow picks its pass from the clock, so postclose runs from 15:30 IST
    until 05:29 IST. After midnight the old `datetime.now(IST).date()` named a
    session that has not happened yet.
    """
    # The real slot, and the evening around it: unchanged by the fix.
    assert pye.session_date(datetime(2026, 10, 8, 16, 10, tzinfo=IST)) == date(2026, 10, 8)
    assert pye.session_date(datetime(2026, 10, 8, 23, 45, tzinfo=IST)) == date(2026, 10, 8)
    # The overnight runs that used to name tomorrow.
    assert pye.session_date(datetime(2026, 10, 9, 0, 0, tzinfo=IST)) == date(2026, 10, 8)
    assert pye.session_date(datetime(2026, 10, 9, 3, 0, tzinfo=IST)) == date(2026, 10, 8)
    assert pye.session_date(datetime(2026, 10, 9, 5, 29, tzinfo=IST)) == date(2026, 10, 8)
    # Before the close, today's session is not complete either.
    assert pye.session_date(datetime(2026, 10, 9, 11, 0, tzinfo=IST)) == date(2026, 10, 8)
    assert pye.session_date(datetime(2026, 10, 9, 15, 29, tzinfo=IST)) == date(2026, 10, 8)
    assert pye.session_date(datetime(2026, 10, 9, 15, 30, tzinfo=IST)) == date(2026, 10, 9)
    # A weekend rolls back to Friday.
    assert pye.session_date(datetime(2026, 10, 10, 18, 0, tzinfo=IST)) == date(2026, 10, 9)
    assert pye.session_date(datetime(2026, 10, 11, 2, 0, tzinfo=IST)) == date(2026, 10, 9)


def test_overnight_run_does_not_inflate_the_catchup_window():
    """EIMCOELECO's real shape, and the alert that nearly did not happen.

    OB candle 2026-10-05, born 2026-10-06, catchup_sessions 3. It alerted at
    15:30 IST on 08-Oct with `ob_age` exactly == 3 - on the boundary. Every run
    after midnight named 09-Oct, which made ob_age 4 and would have suppressed
    the same event permanently, with no alert at all.
    """
    s = RuleSettings()
    origin, born = date(2026, 10, 5), date(2026, 10, 6)
    for hh, mm in [(15, 30), (20, 0), (23, 45)]:
        today = pye.session_date(datetime(2026, 10, 8, hh, mm, tzinfo=IST))
        assert pye.weekday_sessions(origin, today) <= s.catchup_sessions
        assert pye.weekday_sessions(born, today) <= s.catchup_sessions
    for hh, mm in [(0, 0), (3, 0), (5, 29)]:        # overnight on 09-Oct
        today = pye.session_date(datetime(2026, 10, 9, hh, mm, tzinfo=IST))
        assert today == date(2026, 10, 8)
        assert pye.weekday_sessions(origin, today) <= s.catchup_sessions
    # What the old wall-clock `today` computed at 00:00 IST on 09-Oct.
    assert pye.weekday_sessions(origin, date(2026, 10, 9)) > s.catchup_sessions


def test_postclose_evaluates_against_the_completed_session(tmp_path, monkeypatch,
                                                           caplog):
    """With no --today, the pass takes its session from session_date(), so an
    overnight run evaluates against the same session as the 16:10 slot."""
    import logging
    monkeypatch.setattr(pye, "session_date", lambda now=None: date(2026, 10, 8))
    st = state(events=[ob_event(born="2026-09-10")], as_of=BORN)
    with caplog.at_level(logging.INFO):
        rc, _tg, _saved = run_pc(tmp_path, monkeypatch, st)   # no today=
    assert rc == 0
    assert "postclose 2026-10-08:" in caplog.text


def test_cycle_gate_avoids_refetching_a_settled_cycle(tmp_path, monkeypatch):
    """ARFIN's shape: the state's first post-breakout birth is not the birth the
    bars report, so `rule|SYM|BORN` never matched and the symbol paid a full
    560-day daily-history fetch on EVERY run - ~210 a day at the live 15-minute
    cadence - only to rediscover that its cycle had been settled long ago. With
    the cycle on the index the run must short-circuit before any data call.
    """
    p_in = tmp_path / "ob_precision_state.json"
    p_own = tmp_path / "precision_ob_entry_state.json"
    st = state(events=[ob_event(born="2026-10-07", origin="2026-10-06")],
               as_of="2026-10-07")
    own = {"sent": {f"rule|TEST|{BREAKOUT}": "2026-10-07T15:54:49+05:30"},
           "cycles": {f"TEST|{BREAKOUT}": f"rule|TEST|{BREAKOUT}"},
           "open": [], "closed": [], "excluded": {}}
    p_in.write_text(json.dumps(st))
    p_own.write_text(json.dumps(own))
    monkeypatch.setattr(pye, "STATE_IN", p_in)
    monkeypatch.setattr(pye, "STATE_OWN", p_own)
    monkeypatch.setattr(pye, "DhanClient", lambda *a, **k: object())
    calls: list = []

    def counting_fetch(*a, **k):
        calls.append(1)
        return cycle_bars()

    monkeypatch.setattr(pye, "fetch_bars", counting_fetch)
    tg = FakeTG()
    rc = pye.run_postclose(fake_cfg(), Args(today="2026-10-08"), tg)
    assert rc == 0 and tg.sent == []
    assert calls == []                     # the point of the fix: no data call


def test_cycle_index_is_recorded_when_an_event_is_marked(tmp_path, monkeypatch):
    """The index is what makes that fix self-healing: a key marked by an older
    build still settles its cycle, so it is recorded on the way past and the
    next run is cheap."""
    st = state(events=[ob_event(born="2026-09-10")])
    rc, tg, saved = run_pc(tmp_path, monkeypatch, st, today="2026-10-08")
    assert rc == 0 and tg.sent == []
    assert saved["cycles"] == {f"TEST|{BREAKOUT}": "rule|TEST|2026-09-10"}


def test_quiet_run_does_not_rewrite_the_state_file(tmp_path, monkeypatch):
    """The workflow commits whatever changed. `last_run` used to carry a clock
    time and was written at the TOP of the pass, so every 15-minute run produced
    a commit - 145 commits in a day and a half, for a job that sent two alerts.
    A run that moves the book nowhere must leave the file byte-identical.
    """
    st = state(events=[ob_event(born="2026-09-10")], as_of=BORN)
    rc, _tg, _saved = run_pc(tmp_path, monkeypatch, st, today=BORN)
    assert rc == 0
    p_own = tmp_path / "precision_ob_entry_state.json"
    assert p_own.exists()                  # the first run did move the book
    book = json.loads(p_own.read_text())
    # Session-scoped on purpose: a clock time here is what made every run differ.
    assert book["last_run"] == f"postclose {BORN}"
    assert "last_run_at" in book           # the clock time lives in its own field
    writes: list = []
    monkeypatch.setattr(pye, "save_state", lambda s_, p: writes.append(p))
    rc2, tg2, _saved2 = run_pc(tmp_path, monkeypatch, st, today=BORN)
    assert rc2 == 0 and tg2.sent == []
    assert writes == []                    # a quiet run must not touch the file


def test_quiet_intraday_run_does_not_rewrite_the_state_file(tmp_path,
                                                            monkeypatch):
    """Same rule for the intraday pass: a stale holiday quote moves nothing."""
    st = state(as_of=OB_CANDLE, as_of_high=124.0, as_of_low=111.0,
               as_of_close=123.0, refreshed_on=BORN)
    rc, _tg, _saved = run_id(tmp_path, monkeypatch, st, today=BORN)
    assert rc == 0
    p_own = tmp_path / "precision_ob_entry_state.json"
    assert p_own.exists()
    assert json.loads(p_own.read_text())["last_run"] == f"intraday {BORN}"
    writes: list = []
    monkeypatch.setattr(pye, "save_state", lambda s_, p: writes.append(p))
    rc2, tg2, _s2 = run_id(tmp_path, monkeypatch, st, today=BORN)
    assert rc2 == 0 and tg2.sent == []
    assert writes == []
