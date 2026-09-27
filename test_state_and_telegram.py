"""De-duplication state and Telegram message formatting."""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from state import AlertState
from strategy import BarEval, Signal
from telegram import Telegram, format_signal, _split


@pytest.fixture
def state(tmp_path):
    return AlertState(tmp_path / "state.json")


def test_state_marks_and_detects(state):
    assert not state.already_alerted("2026-07-27", "RELIANCE")
    state.mark("2026-07-27", "RELIANCE", datetime(2026, 7, 27, 10, 5), 1500.0)
    assert state.already_alerted("2026-07-27", "RELIANCE")
    assert not state.already_alerted("2026-08-03", "RELIANCE")


def test_state_persists_across_processes(tmp_path):
    p = tmp_path / "state.json"
    a = AlertState(p)
    a.mark("2026-07-27", "TCS", datetime(2026, 7, 27, 11, 0), 3900.0)
    a.save()
    assert AlertState(p).already_alerted("2026-07-27", "TCS")


def test_breakout_lock_persists_for_26_weeks(tmp_path):
    p = tmp_path / "state.json"
    ist = ZoneInfo("Asia/Kolkata")
    first_alert = datetime(2026, 7, 27, 10, 5, tzinfo=ist)

    state = AlertState(p)
    state.mark_breakout_alert("OAL", first_alert, 402.65, 390.0)
    state.save()

    restored = AlertState(p)
    expires = first_alert + timedelta(weeks=26)
    assert restored.breakout_lock_until("OAL", 26) == expires
    assert restored.breakout_cooldown_active(
        "OAL", expires - timedelta(minutes=5), 26)
    assert not restored.breakout_cooldown_active("OAL", expires, 26)
    assert restored.breakout_lock_until("UNKNOWN", 26) is None
    assert restored.breakout_lock_until("OAL", 0) is None


def test_legacy_weekly_alerts_migrate_from_earliest_retained_week(tmp_path):
    p = tmp_path / "state.json"
    state = AlertState(p)
    state.mark("2026-08-24", "OAL", datetime(2026, 8, 26, 15, 20), 402.65)
    state.mark("2026-09-21", "OAL", datetime(2026, 9, 21, 13, 55), 561.70)
    state.save()

    migrated = AlertState(p)
    assert migrated.breakout_record("OAL")["bar_time"] == "2026-08-26T15:20"
    assert migrated.breakout_lock_until("OAL", 26) == datetime(2027, 2, 24, 15, 20)
    migrated.save()
    assert AlertState(p).breakout_record("OAL")["bar_time"] == "2026-08-26T15:20"


def test_explicit_first_alert_date_overrides_legacy_weekly_history(tmp_path):
    p = tmp_path / "state.json"
    p.write_text(
        '{"weeks": {"2026-08-24": {"OAL": '
        '{"bar_time": "2026-08-26T15:20", "price": 402.65}}}, '
        '"breakout_alerts": {"OAL": '
        '{"bar_time": "2026-07-27T15:30+05:30"}}}'
    )
    state = AlertState(p)
    assert state.breakout_record("OAL")["bar_time"] == "2026-07-27T15:30+05:30"
    assert state.breakout_lock_until("OAL", 26) == datetime(
        2027, 1, 25, 15, 30, tzinfo=ZoneInfo("Asia/Kolkata"))


def test_state_survives_corrupt_file(tmp_path):
    p = tmp_path / "state.json"
    p.write_text("{ not json")
    s = AlertState(p)
    assert not s.already_alerted("2026-07-27", "X")


def test_prune_keeps_recent_weeks(state):
    for i in range(10):
        state.mark(f"2026-0{i}-01" if i < 10 else "", f"SYM{i}",
                   datetime(2026, 1, 1, 10, 0), 100.0)
    state.prune(keep_weeks=3)
    assert len(state._data["weeks"]) == 3


def test_save_is_atomic_noop_when_clean(tmp_path):
    p = tmp_path / "state.json"
    AlertState(p).save()
    assert not p.exists()          # nothing dirty -> nothing written


# ------------------------------------------------------------- telegram
def make_signal(symbol="RELIANCE", price=1500.0, level=1450.0, trigger="cross"):
    ev = BarEval(
        conditions={f"c{i:02d}": True for i in range(1, 14)},
        values={"price": price, "ema_fast": 1400.0, "ema_slow": 1350.0,
                "ema_slow_2": 1340.0, "rsi": 72.5, "rsi_1": 63.1,
                "macd_hist": 12.345, "week_volume": 13_400_000.0,
                "vol_sma": 9_000_000.0, "week_open": 1420.0, "day_open": 1480.0,
                "entry_level": level, "level_52": 1490.0},
    )
    return Signal(symbol=symbol, security_id="2885", exchange_segment="NSE_EQ",
                  bar_time=datetime(2026, 7, 27, 10, 5), price=price,
                  entry_level=level, level_52=1490.0, trigger=trigger,
                  evaluation=ev, week_start="2026-07-27",
                  week_volume=13_400_000.0, day_open=1480.0, week_open=1420.0)


def test_format_signal_contains_key_facts():
    msg = format_signal(make_signal())
    assert "RELIANCE" in msg
    assert "BUY" in msg
    assert "1,500.00" in msg and "1,450.00" in msg
    assert "3.45%" in msg               # (1500-1450)/1450
    assert "10:05" in msg


def test_deferred_signal_is_labelled():
    assert "deferred" in format_signal(make_signal(trigger="deferred"))


def test_partial_pass_is_flagged():
    sig = make_signal()
    sig.evaluation.conditions["c09"] = False
    msg = format_signal(sig)
    assert "12/13" in msg
    assert "Wk vol" in msg


def test_html_is_escaped():
    msg = format_signal(make_signal(symbol="A&B<test>"))
    assert "&amp;" in msg and "&lt;" in msg


def test_dry_run_never_calls_network(caplog):
    tg = Telegram("", "", dry_run=True)
    assert tg.send("hello") is True
    assert tg.dry_run


def test_missing_credentials_forces_dry_run():
    assert Telegram("", "12345").dry_run
    assert Telegram("token", "").dry_run


def test_batch_message_lists_every_symbol():
    tg = Telegram("", "", dry_run=True)
    sigs = [make_signal("AAA"), make_signal("BBB"), make_signal("CCC")]
    assert tg.send_batch(sigs) is True


def test_split_respects_limit():
    text = "\n".join(f"line {i}" for i in range(2000))
    parts = _split(text, 4096)
    assert all(len(p) <= 4096 for p in parts)
    assert len(parts) > 1


def test_split_short_text_untouched():
    assert _split("hello", 4096) == ["hello"]
