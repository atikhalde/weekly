"""Tests for strategy_alert.py - the two-leg strategy alert.

Pure-logic tests on a synthetic ob_precision_state-shaped dict: detection
rules, the level filter, cycle-spent logic, exit checks (including the
conservative both-touch policy), the trade-plan math and the stale-quote
guard. No network, no Dhan, no Telegram.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest

from strategy_alert import (  # noqa: E402
    TradePlan, check_exit, first_tap_after_breakout, leg1_today,
    leg2_today, plan_html, swing_target, weekday_sessions,
)


def synth_state(sym="TEST", level=100.0, brk="2026-06-01", zones=None,
                events=None, status="waiting", as_of=None, as_of_high=None,
                as_of_low=None, as_of_close=None, refreshed_on=None):
    return {
        "waiting": {sym: {
            "status": status, "security_id": "1",
            "exchange_segment": "NSE_EQ",
            "breakout_26w_session": {"session": brk, "level": level,
                                     "close": level * 1.01},
        }},
        "zones": {sym: {
            "as_of": as_of, "as_of_high": as_of_high, "as_of_low": as_of_low,
            "as_of_close": as_of_close, "refreshed_on": refreshed_on,
            "zones": zones or [], "closed_events": events or [],
            "prev_highs": [90.0] * 8, "prev_volumes": [1000.0] * 19,
            "atr": 5.0,
        }},
    }


def ob_event(born, origin=None, price=110.0, top=105.0, bottom=100.0,
             rvol=2.5):
    return {"kind": "ob", "born_session": born, "session": born,
            "price": price, "top": top, "bottom": bottom, "rvol": rvol,
            "entry": 103.0, "stop": 99.0,
            "detail": {"origin_session": origin or born,
                       "origin_offset": 1}}


def tap_event(session, price=101.0, number=1, born="2026-06-10",
              tapped=103.0):
    return {"kind": "tap", "session": session, "price": price,
            "tap_number": number, "born_session": born, "entry": tapped,
            "confirmed": True, "top": 105.0, "bottom": 100.0,
            "detail": {"tapped_entry": tapped}}


# --------------------------------------------------------------------------- #
#  session counting + plan math
# --------------------------------------------------------------------------- #
def test_weekday_sessions_excludes_weekend():
    assert weekday_sessions(date(2026, 10, 5), date(2026, 10, 12)) == 5


def test_plan_math_qty_risk_reward():
    p = TradePlan("TEST", 2, 100.0, 90.0, 120.0, "2026-10-07", "2026-06-01",
                  100000.0, 1.0)
    assert p.risk_amt == 1000.0
    assert p.risk_gap == 10.0
    assert p.qty == 100
    assert p.value == 10000.0
    assert p.rew_pct == pytest.approx(20.0)
    assert p.risk_pct_price == pytest.approx(10.0)
    assert p.rr == pytest.approx(2.0)


def test_plan_html_warns_when_stop_too_far():
    p = TradePlan("TEST", 2, 100.0, 40.0, 120.0, "2026-10-07", "2026-06-01",
                  1000.0, 1.0)
    html = plan_html(p)
    assert "stop too far" in html
    assert "₹10" in html


# --------------------------------------------------------------------------- #
#  Leg 1: the cycle's FIRST post-breakout OB, born today
# --------------------------------------------------------------------------- #
def test_leg1_fires_on_first_zone_born_today():
    st = synth_state(events=[ob_event("2026-10-07", "2026-10-06",
                                      price=110.0)])
    ev = leg1_today(st, "TEST", "2026-10-07")
    assert ev is not None and ev["born"] == "2026-10-07"


def test_leg1_silent_when_first_zone_was_earlier():
    st = synth_state(events=[ob_event("2026-09-01"),
                             ob_event("2026-10-07")])
    assert leg1_today(st, "TEST", "2026-10-07") is None


def test_leg1_silent_on_other_days():
    st = synth_state(events=[ob_event("2026-10-06")])
    assert leg1_today(st, "TEST", "2026-10-07") is None


def test_leg1_uses_event_price_over_zone_dict():
    z = [{"born_session": "2026-10-07", "origin_session": "2026-10-06",
          "top": 105.0, "bottom": 100.0, "entry": 103.0, "stop": 99.0,
          "taps": 0, "tap_session": "", "departed": False, "state": 0}]
    e = [ob_event("2026-10-07", price=111.0)]
    st = synth_state(zones=z, events=e)
    ev = leg1_today(st, "TEST", "2026-10-07")
    assert ev["zone"]["price"] == 111.0


# --------------------------------------------------------------------------- #
#  Leg 2: the cycle's FIRST tap, close above the level
# --------------------------------------------------------------------------- #
def test_leg2_fires_on_first_tap_above_level():
    st = synth_state(events=[tap_event("2026-10-07", price=101.0)],
                     as_of="2026-10-07", as_of_close=101.0)
    ev = leg2_today(st, "TEST", "2026-10-07")
    assert ev is not None and ev["entry"] == 101.0


def test_leg2_silent_when_close_below_level():
    st = synth_state(events=[tap_event("2026-10-07", price=99.0)],
                     as_of="2026-10-07", as_of_close=99.0)
    assert leg2_today(st, "TEST", "2026-10-07") is None


def test_leg2_silent_when_first_tap_was_earlier():
    st = synth_state(events=[tap_event("2026-09-01"),
                             tap_event("2026-10-07")])
    assert leg2_today(st, "TEST", "2026-10-07") is None


def test_leg2_silent_on_repeat_tap_number():
    st = synth_state(events=[tap_event("2026-10-07", number=2)])
    assert leg2_today(st, "TEST", "2026-10-07") is None


def test_first_tap_prefers_price_carrying_event_over_zone_synthetic():
    z = [{"born_session": "2026-06-10", "taps": 1,
          "tap_session": "2026-10-07", "entry": 103.0}]
    e = [tap_event("2026-10-07", price=101.5)]
    st = synth_state(zones=z, events=e)
    ts, ev = first_tap_after_breakout(st["waiting"]["TEST"], st["zones"]["TEST"])
    assert ts == "2026-10-07" and ev.get("price") == 101.5


# --------------------------------------------------------------------------- #
#  Exits
# --------------------------------------------------------------------------- #
def trade(**kw):
    base = {"entry": 100.0, "stop": 90.0, "target": 120.0,
            "entry_session": "2026-10-07", "last_checked": "2026-10-07"}
    base.update(kw)
    return base


def test_exit_target_hit():
    ex = check_exit(trade(), {"as_of": "2026-10-08", "as_of_high": 121.0,
                              "as_of_low": 95.0, "as_of_close": 118.0})
    assert ex["outcome"] == "win" and ex["price"] == 120.0


def test_exit_stop_hit():
    ex = check_exit(trade(), {"as_of": "2026-10-08", "as_of_high": 105.0,
                              "as_of_low": 89.0, "as_of_close": 92.0})
    assert ex["outcome"] == "loss" and ex["price"] == 90.0


def test_exit_both_touched_is_conservative_loss():
    ex = check_exit(trade(), {"as_of": "2026-10-08", "as_of_high": 125.0,
                              "as_of_low": 85.0, "as_of_close": 100.0})
    assert ex["outcome"] == "loss"


def test_exit_time_stop_at_90_sessions():
    ex = check_exit(trade(entry_session="2026-06-01",
                          last_checked="2026-06-01"),
                    {"as_of": "2026-10-08", "as_of_high": 105.0,
                     "as_of_low": 95.0, "as_of_close": 100.0})
    assert ex["outcome"] == "timeout"


def test_exit_none_when_nothing_filled():
    ex = check_exit(trade(), {"as_of": "2026-10-08", "as_of_high": 105.0,
                              "as_of_low": 95.0, "as_of_close": 100.0})
    assert ex is None


def test_exit_skips_already_checked_session():
    ex = check_exit(trade(), {"as_of": "2026-10-07", "as_of_high": 130.0,
                              "as_of_low": 80.0, "as_of_close": 100.0})
    assert ex is None


# --------------------------------------------------------------------------- #
#  Target computation
# --------------------------------------------------------------------------- #
class B:
    def __init__(self, session, high):
        self.session, self.high = session, high


def test_swing_target_inclusive_window():
    bars = [B("2026-06-01", 100.0), B("2026-06-02", 110.0),
            B("2026-06-03", 105.0), B("2026-06-04", 108.0)]
    assert swing_target(bars, "2026-06-01", "2026-06-04") == 110.0
    assert swing_target(bars, "2026-06-02", "2026-06-04") == 110.0
    assert swing_target(bars, "2026-06-03", "2026-06-04") == 108.0
