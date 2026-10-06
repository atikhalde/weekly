"""Tests for strategy_alert.py - the two-leg strategy alert.

Pure-logic tests on a synthetic ob_precision_state-shaped dict: detection
rules, the level filter, cycle-spent logic, exit checks (including the
conservative both-touch policy), the trade-plan math and the stale-quote
guard. No network, no Dhan, no Telegram.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest

import strategy_alert as sa  # noqa: E402
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


# --------------------------------------------------------------------------- #
#  Market-holiday guard (stale bulk quote)
# --------------------------------------------------------------------------- #
def test_quote_is_stale_when_the_feed_reserves_the_closed_session():
    """A weekday holiday hands back the previous session's numbers
    unchanged; replaying them would fire a 'forming' heads-up for a candle
    that is not forming at all."""
    ctx = {"as_of_high": 110.0, "as_of_low": 100.0, "as_of_close": 105.0}
    assert sa.quote_is_stale(
        {"high": 110.0, "low": 100.0, "last_price": 105.0}, ctx)


def test_quote_is_not_stale_once_the_session_moves():
    ctx = {"as_of_high": 110.0, "as_of_low": 100.0, "as_of_close": 105.0}
    assert not sa.quote_is_stale(
        {"high": 111.0, "low": 100.0, "last_price": 105.0}, ctx)
    assert not sa.quote_is_stale(
        {"high": 110.0, "low": 100.0, "last_price": 106.0}, ctx)


def test_quote_staleness_survives_a_context_with_no_closed_bar():
    """An empty/None context must not read as 'identical' and must not raise."""
    assert not sa.quote_is_stale({"high": 0.0, "low": 0.0, "last_price": 0.0},
                                 {})


# --------------------------------------------------------------------------- #
#  Spent cycle (Leg 2 is the cycle's FIRST tap, and only that)
# --------------------------------------------------------------------------- #
def test_cycle_is_spent_once_tap_one_has_happened():
    st = synth_state(events=[ob_event("2026-06-10"),
                             tap_event("2026-06-20")])
    assert sa.cycle_is_spent(st["waiting"]["TEST"], st["zones"]["TEST"])


def test_cycle_is_not_spent_before_the_first_tap():
    st = synth_state(events=[ob_event("2026-06-10")])
    assert not sa.cycle_is_spent(st["waiting"]["TEST"], st["zones"]["TEST"])


# --------------------------------------------------------------------------- #
#  Missed-run gap protection (BUG 55: GitHub cron skips slots)
# --------------------------------------------------------------------------- #
class WalkBar:
    def __init__(self, session, high, low):
        self.session, self.high, self.low = session, high, low


def test_missed_sessions_is_zero_on_a_normal_daily_pass():
    assert sa.missed_sessions({"last_checked": "2026-10-07"},
                              {"as_of": "2026-10-08"}) == 0


def test_missed_sessions_counts_only_the_skipped_ones():
    # checked Wed 07, state now Mon 12: Thu 08 and Fri 09 were skipped
    assert sa.missed_sessions({"last_checked": "2026-10-07"},
                              {"as_of": "2026-10-12"}) == 2


def test_bar_walk_catches_a_stop_filled_on_a_skipped_day(monkeypatch):
    """The stop filled on a day no run happened. The latest closed bar shows
    nothing, so plain check_exit returns None - the walk must report the
    loss with the SKIPPED day's session and the stop price."""
    bars = [WalkBar("2026-10-08", 105.0, 96.0),
            WalkBar("2026-10-09", 104.0, 88.0),      # stop 90 swept here
            WalkBar("2026-10-12", 103.0, 99.0)]
    ctx = {"as_of": "2026-10-12", "as_of_high": 103.0,
           "as_of_low": 99.0, "as_of_close": 100.0}
    t = trade(symbol="TEST", last_checked="2026-10-07",
              breakout_session="2026-06-01")

    assert check_exit(dict(t), ctx) is None          # the bug, unprotected
    monkeypatch.setattr(sa, "fetch_bars", lambda c, r, **k: bars)
    ex = sa.exit_with_backfill(t, ctx, object(), {"security_id": "1"})
    assert ex["outcome"] == "loss"
    assert ex["session"] == "2026-10-09"
    assert ex["price"] == 90.0


def test_bar_walk_makes_no_data_call_on_a_normal_daily_pass(monkeypatch):
    """Cost shape: the walk buys nothing when no run was missed and the
    target is known, so it must not spend a daily-history call."""
    calls = []
    monkeypatch.setattr(sa, "fetch_bars",
                        lambda c, r, **k: calls.append(r) or [])
    t = trade(symbol="TEST", last_checked="2026-10-07",
              breakout_session="2026-06-01")
    ctx = {"as_of": "2026-10-08", "as_of_high": 105.0,
           "as_of_low": 95.0, "as_of_close": 100.0}
    assert sa.exit_with_backfill(t, ctx, object(), {"security_id": "1"}) is None
    assert calls == []


def test_bar_walk_backfills_a_target_that_was_unknown_at_entry(monkeypatch):
    monkeypatch.setattr(sa, "fetch_bars", lambda c, r, **k: [
        WalkBar("2026-06-01", 130.0, 100.0), WalkBar("2026-10-08", 105.0, 95.0)])
    monkeypatch.setattr(sa, "swing_target", lambda b, lo, hi: 130.0)
    t = trade(symbol="TEST", target=None, last_checked="2026-10-07",
              breakout_session="2026-06-01")
    sa.exit_with_backfill(t, ctx_daily(), object(), {"security_id": "1"})
    assert t["target"] == 130.0


def ctx_daily():
    return {"as_of": "2026-10-08", "as_of_high": 105.0,
            "as_of_low": 95.0, "as_of_close": 100.0}


# --------------------------------------------------------------------------- #
#  postclose integration (synthetic state files, no network)
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
        self.no_data = True
        self.dry_run = False
        for k, v in kw.items():
            setattr(self, k, v)


class FakeCfg:
    class secrets:
        dhan_client_id = "x"
        dhan_access_token = "y"

    class runtime:
        data_rate_per_sec = 1
        quote_rate_per_sec = 1
        dry_run = False


def run_pc(tmp_path, monkeypatch, st_in, own=None, cfg=None, tg=None, **kw):
    p_in = tmp_path / "ob_precision_state.json"
    p_own = tmp_path / "strategy_alert_state.json"
    p_in.write_text(json.dumps(st_in))
    if own is not None:
        p_own.write_text(json.dumps(own))
    monkeypatch.setattr(sa, "STATE_IN", p_in)
    monkeypatch.setattr(sa, "STATE_OWN", p_own)
    tg = tg or FakeTG()
    rc = sa.run_postclose(cfg or FakeCfg(), Args(**kw), tg)
    saved = json.loads(p_own.read_text()) if p_own.exists() else {}
    return rc, tg, saved


def test_leg1_trade_records_the_breakout_session(tmp_path, monkeypatch):
    """Without it the target can never be backfilled later - the bar-walk
    has no window to compute the swing high from."""
    st = synth_state(events=[ob_event("2026-10-07", price=110.0)],
                     as_of="2026-10-07", as_of_close=110.0)
    rc, tg, saved = run_pc(tmp_path, monkeypatch, st, today="2026-10-07")
    assert rc == 0
    assert len(saved["open"]) == 1
    assert saved["open"][0]["leg"] == 1
    assert saved["open"][0]["breakout_session"] == "2026-06-01"


def test_postclose_does_not_persist_state_when_telegram_fails(tmp_path,
                                                              monkeypatch):
    st = synth_state(events=[ob_event("2026-10-07", price=110.0)],
                     as_of="2026-10-07", as_of_close=110.0)
    original = {"sent": {}, "open": [], "closed": [], "last_run": "before"}
    rc, tg, saved = run_pc(tmp_path, monkeypatch, st, own=original,
                           tg=FailedTG(), today="2026-10-07")
    assert rc == 1
    assert len(tg.sent) == 1
    # The next scheduled pass must see the event as unsent and retry it.
    assert saved == original


def test_postclose_dry_run_does_not_persist_trade_or_dedupe(tmp_path,
                                                            monkeypatch):
    st = synth_state(events=[ob_event("2026-10-07", price=110.0)],
                     as_of="2026-10-07", as_of_close=110.0)
    original = {"sent": {}, "open": [], "closed": [], "last_run": "before"}
    rc, tg, saved = run_pc(tmp_path, monkeypatch, st, own=original,
                           today="2026-10-07", dry_run=True)
    assert rc == 0
    assert len(tg.sent) == 1
    assert saved == original


def test_one_malformed_trade_cannot_kill_the_postclose_run(tmp_path,
                                                           monkeypatch):
    """A half-written record must cost itself its check, not cost every
    other open trade theirs."""
    st = synth_state(as_of="2026-10-07", as_of_high=121.0,
                     as_of_low=95.0, as_of_close=118.0)
    own = {"sent": {}, "closed": [], "open": [
        {"id": "MALFORMED"},                       # no symbol, no entry
        {"id": "TEST-L1-2026-10-01", "symbol": "TEST", "leg": 1,
         "entry": 100.0, "stop": 90.0, "target": 120.0,
         "breakout_session": "2026-06-01",
         "entry_session": "2026-10-01", "last_checked": "2026-10-01"},
    ]}
    rc, tg, saved = run_pc(tmp_path, monkeypatch, st, own=own,
                           today="2026-10-07")
    assert rc == 0
    # the good trade still resolved, and its alert still went out
    assert len(saved["closed"]) == 1
    assert saved["closed"][0]["exit"]["outcome"] == "win"
    assert any("TARGET HIT" in m for m in tg.sent)
    # the malformed one survived in the book rather than vanishing
    assert [t["id"] for t in saved["open"]] == ["MALFORMED"]


def test_postclose_reports_a_stop_filled_on_a_skipped_day(tmp_path,
                                                          monkeypatch):
    """End-to-end: the gap protection is actually WIRED into the exits loop,
    not merely defined."""
    monkeypatch.setattr(sa, "DhanClient", lambda *a, **k: object())
    monkeypatch.setattr(sa, "fetch_bars", lambda c, r, **k: [
        WalkBar("2026-10-08", 105.0, 96.0),
        WalkBar("2026-10-09", 104.0, 88.0),        # stop swept, no run that day
        WalkBar("2026-10-12", 103.0, 99.0)])
    st = synth_state(as_of="2026-10-12", as_of_high=103.0,
                     as_of_low=99.0, as_of_close=100.0)
    own = {"sent": {}, "closed": [], "open": [
        {"id": "TEST-L2-2026-10-07", "symbol": "TEST", "leg": 2,
         "entry": 100.0, "stop": 90.0, "target": 120.0,
         "breakout_session": "2026-06-01",
         "entry_session": "2026-10-07", "last_checked": "2026-10-07"},
    ]}
    rc, tg, saved = run_pc(tmp_path, monkeypatch, st, own=own,
                           today="2026-10-12", no_data=False)
    assert rc == 0
    assert len(saved["closed"]) == 1
    ex = saved["closed"][0]["exit"]
    assert ex["outcome"] == "loss"
    assert ex["session"] == "2026-10-09"
    assert ex["price"] == 90.0
    assert any("STOPPED OUT" in m for m in tg.sent)


@pytest.mark.parametrize("tg_class,dry_run,expected_rc", [
    (FailedTG, False, 1),
    (FakeTG, True, 0),
])
def test_intraday_does_not_persist_dedupe_on_failure_or_dry_run(
        tmp_path, monkeypatch, tg_class, dry_run, expected_rc):
    today = "2026-10-07"
    st = synth_state(refreshed_on=today)
    p_in = tmp_path / "ob_precision_state.json"
    p_own = tmp_path / "strategy_alert_state.json"
    p_in.write_text(json.dumps(st))
    monkeypatch.setattr(sa, "STATE_IN", p_in)
    monkeypatch.setattr(sa, "STATE_OWN", p_own)

    class QuoteClient:
        def __init__(self, *args, **kwargs):
            pass

        def ohlc(self, request):
            return {"NSE_EQ": {"1": {
                "open": 100.0, "high": 120.0, "low": 99.0,
                "last_price": 119.0, "volume": 5000.0,
            }}}

    monkeypatch.setattr(sa, "DhanClient", QuoteClient)
    tg = tg_class()
    rc = sa.run_intraday(FakeCfg(), Args(today=today, dry_run=dry_run), tg)

    assert rc == expected_rc
    assert len(tg.sent) == 1
    assert not p_own.exists()
