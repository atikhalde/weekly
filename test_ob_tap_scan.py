"""
Tests for `ob_tap_scan.py` - the scanner that turns the weekly breakout list
into precision order-block tap alerts.

Nothing here touches the network. The Dhan client, the Telegram fan-out and the
wall clock are all replaced, and every run happens in a throw-away config
directory, so these tests drive the real `main()` end to end: waiting-list
harvest -> closed-bar replay -> bulk quote -> live tap -> alert -> state save.

The behaviours that matter most, and why:

* `state.json` is READ-ONLY here. It belongs to scan.py, and a second writer
  would corrupt the weekly de-dupe history. One test asserts the bytes never
  change.
* The waiting list must OUTLIVE state.json, which prunes at six weeks. The user
  asked for "until tapped or invalidated", not "until the file forgets".
* Nothing may be alerted twice, and the first (backfill) run may not dump a
  year of historical order blocks into the chat.

Run with:  python -m pytest test_ob_tap_scan.py -q
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import ob_precision
import ob_tap_scan
from dhan import DhanError, IST
from ob_precision import Bar, bars_from_frame
from ob_tap_scan import (
    STATE_LOGIC_VERSION, active_waiting, as_date, empty_state, format_heartbeat,
    format_tap, harvest_waiting, load_state, prune_closed_events, save_state,
    session_closed, weekly_alerts,
)

# Thursday, mid-session: market_is_open() is True and session_closed() is False.
FIXED = datetime(2026, 8, 27, 11, 0, tzinfo=IST)
D0 = date(2026, 8, 3)

SYM = "TESTSYM"
SID = "12345"
OLD = "OLDSYM"
OLD_SID = "777"

STATE_JSON = {
    "updated_at": "2026-08-26T10:00:00",
    "weeks": {
        "2026-08-17": {OLD: {"bar_time": "2026-08-18T09:45+05:30", "price": 55.0}},
        "2026-08-24": {
            SYM: {"bar_time": "2026-08-26T10:00+05:30", "price": 105.5},
            # Alerted weekly, but present in neither universe.csv nor the
            # snapshot: unresolvable, so it must be skipped rather than crash.
            "GHOST": {"bar_time": "2026-08-26T11:00+05:30", "price": 12.0},
        },
    },
}

SNAPSHOT_CSV = (
    "symbol,security_id,exchange_segment,week_start,entry_level\n"
    f"{SYM},{SID},NSE_EQ,2026-08-24,104.0\n"
    f"{OLD},{OLD_SID},NSE_EQ,2026-08-24,54.0\n"
)
UNIVERSE_CSV = (
    "security_id,symbol,name,exchange_segment,series,type\n"
    f"{SID},{SYM},Test Sym Ltd,NSE_EQ,EQ,ES\n"
    f"{OLD_SID},{OLD},Old Sym Ltd,NSE_EQ,EQ,ES\n"
)


# --------------------------------------------------------------------------- #
#  Fakes
# --------------------------------------------------------------------------- #
def scenario(n_quiet: int = 20, shift_days: int = 0) -> list[Bar]:
    """
    The same canonical series as test_ob_precision.py - quiet bars, then a
    displacement, a departure and a drift - so the closed replay ends with ONE
    armed zone whose entry is 100.60 and whose stop is 97.50.

    `shift_days` slides the WHOLE history into the past, which is how the tests
    below build "an order block born a month ago" without inventing candles.
    """
    n = n_quiet
    bars = [Bar(100.0, 101.0, 98.0, 99.0, 100.0, D0 + timedelta(days=i))
            for i in range(n)]
    bars += [
        Bar(100.0, 107.0, 99.5, 106.5, 300.0, D0 + timedelta(days=n)),
        Bar(106.5, 112.0, 106.0, 111.5, 150.0, D0 + timedelta(days=n + 1)),
        Bar(111.5, 112.5, 109.0, 110.0, 120.0, D0 + timedelta(days=n + 2)),
    ]
    if shift_days:
        back = timedelta(days=shift_days)
        bars = [Bar(b.open, b.high, b.low, b.close, b.volume, b.time - back)
                for b in bars]
    return bars


TAP_BAR = Bar(110.0, 110.5, 100.5, 105.0, 200.0, D0 + timedelta(days=23))
# A session where nothing traded: the vendor still returns a row, filled with the
# previous close. TradingView draws no bar for it, so it must not age a zone.
PHANTOM = Bar(111.5, 111.5, 111.5, 111.5, 0.0, D0 + timedelta(days=22))


def frame(bars) -> pd.DataFrame:
    return pd.DataFrame({
        "datetime": [pd.Timestamp(b.time, tz="Asia/Kolkata") for b in bars],
        "open": [b.open for b in bars], "high": [b.high for b in bars],
        "low": [b.low for b in bars], "close": [b.close for b in bars],
        "volume": [b.volume for b in bars],
    })


class FakeClient:
    """Stands in for DhanClient: canned daily frames and canned bulk quotes."""

    def __init__(self):
        self.frames: dict[str, pd.DataFrame] = {}
        self.quotes: dict[str, dict] = {}
        self.history_calls: list[str] = []
        self.ohlc_calls: list[dict] = []
        self.fail_history: set[str] = set()
        self.fail_ohlc = False

    def daily_candles(self, security_id, exchange_segment, from_date, to_date,
                      chunk_days=0, symbol=None):
        self.history_calls.append(symbol or str(security_id))
        if symbol in self.fail_history:
            raise DhanError("daily candles failed")
        return self.frames.get(symbol, pd.DataFrame())

    def ohlc(self, securities):
        self.ohlc_calls.append(securities)
        if self.fail_ohlc:
            raise DhanError("quote failed")
        out = {}
        for seg, sids in securities.items():
            out[seg] = {str(s): dict(self.quotes[str(s)])
                        for s in sids if str(s) in self.quotes}
        return out


class FakeTelegram:
    def __init__(self, ok: bool = True):
        self.sent: list[str] = []
        self.ok = ok

    def send(self, text, disable_preview=True):
        self.sent.append(text)
        return self.ok


# Today's developing bar, dipping just onto the 100.60 pre-order entry.
TAP_QUOTE = {"open": 101.0, "high": 103.0, "low": 100.60, "last_price": 102.5,
             "prev_close": 110.0, "volume": 260.0}
FLAT_QUOTE = dict(TAP_QUOTE, low=110.0)          # never reaches the entry


@pytest.fixture
def ws(tmp_path, monkeypatch):
    """A throw-away config directory wired to the fakes, plus a `run()` helper."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("runtime:\n  dry_run: true\n")
    (tmp_path / "state.json").write_text(json.dumps(STATE_JSON, indent=2))
    (tmp_path / "weekly_snapshot.csv").write_text(SNAPSHOT_CSV)
    (tmp_path / "universe.csv").write_text(UNIVERSE_CSV)

    tg = FakeTelegram()
    client = FakeClient()
    monkeypatch.setattr(ob_tap_scan, "build_telegram", lambda cfg, dry_run=False: tg)
    monkeypatch.setattr(ob_tap_scan, "DhanClient", lambda *a, **k: client)
    monkeypatch.setattr(ob_tap_scan, "_now", lambda: FIXED)

    def run(*args):
        monkeypatch.setattr(sys, "argv",
                            ["ob_tap_scan.py", "--config", str(cfg_path), *args])
        return ob_tap_scan.main()

    def state():
        return json.loads((tmp_path / "ob_precision_state.json").read_text())

    def configure(**ob_kwargs):
        """Rewrite config.yaml with an ob_precision block."""
        body = "\n".join(f"  {k}: {json.dumps(v)}" for k, v in ob_kwargs.items())
        cfg_path.write_text("runtime:\n  dry_run: true\nob_precision:\n" + body + "\n")

    def arm(symbol=SYM, sid=SID, bars=None, quote=None):
        client.frames[symbol] = frame(bars if bars is not None else scenario())
        if quote is not None:
            client.quotes[sid] = dict(quote)

    return SimpleNamespace(dir=tmp_path, tg=tg, client=client, run=run, state=state,
                           configure=configure, arm=arm,
                           cfg=ob_tap_scan.load_config(cfg_path))


# --------------------------------------------------------------------------- #
#  Waiting list
# --------------------------------------------------------------------------- #
def test_weekly_alerts_reads_state_json_without_writing_it(ws):
    before = (ws.dir / "state.json").read_bytes()
    alerts = weekly_alerts(ws.cfg, 6)
    assert set(alerts) == {SYM, OLD, "GHOST"}
    assert alerts[SYM] == {"week": "2026-08-24",
                           "breakout_bar": "2026-08-26T10:00+05:30",
                           "breakout_price": 105.5}
    assert (ws.dir / "state.json").read_bytes() == before


def test_weekly_alerts_keeps_only_the_latest_breakout_per_symbol(ws):
    raw = json.loads((ws.dir / "state.json").read_text())
    raw["weeks"]["2026-08-31"] = {SYM: {"bar_time": "2026-09-02T10:00+05:30",
                                        "price": 120.0}}
    (ws.dir / "state.json").write_text(json.dumps(raw))
    alerts = weekly_alerts(ws.cfg, 6)
    assert alerts[SYM]["week"] == "2026-08-31"
    assert alerts[SYM]["breakout_price"] == 120.0


def test_backfill_weeks_limits_the_harvest(ws):
    assert set(weekly_alerts(ws.cfg, 1)) == {SYM, "GHOST"}    # newest week only


def test_harvest_skips_symbols_it_cannot_resolve(ws):
    state = empty_state()
    added = harvest_waiting(state, weekly_alerts(ws.cfg, 6),
                            ob_tap_scan.resolve_universe(ws.cfg),
                            ob_tap_scan.snapshot_levels(ws.cfg), "2026-08-27")
    assert SYM in added and "GHOST" not in added
    assert SYM in state["waiting"] and "GHOST" not in state["waiting"]


def test_harvest_captures_the_level_only_for_the_matching_week(ws):
    state = empty_state()
    harvest_waiting(state, weekly_alerts(ws.cfg, 6),
                    ob_tap_scan.resolve_universe(ws.cfg),
                    ob_tap_scan.snapshot_levels(ws.cfg), "2026-08-27")
    assert state["waiting"][SYM]["level_26w"] == 104.0        # same week
    # OLDSYM alerted in week 2026-08-17 but its snapshot row is for 2026-08-24:
    # printing that level would be printing the WRONG number next to an alert.
    assert state["waiting"][OLD]["level_26w"] is None


def test_waiting_list_survives_the_six_week_prune(ws):
    """
    state.json forgets a symbol once its week is pruned. The waiting list must
    not: the user asked for "until tapped or invalidated", and the 26-week level
    is unrecoverable once the snapshot has been rebuilt for a new week.
    """
    ws.arm(quote=TAP_QUOTE)
    assert ws.run() == 0
    kept = ws.state()["waiting"][SYM]

    (ws.dir / "state.json").write_text(json.dumps({"weeks": {}, "updated_at": None}))
    assert ws.run() == 0
    after = ws.state()["waiting"][SYM]
    assert after["level_26w"] == kept["level_26w"] == 104.0
    assert after["breakout_price"] == kept["breakout_price"] == 105.5
    assert after["week"] == kept["week"] == "2026-08-24"


def test_a_tapped_symbol_is_not_resurrected_by_a_later_harvest(ws):
    state = empty_state()
    ids = ob_tap_scan.resolve_universe(ws.cfg)
    harvest_waiting(state, weekly_alerts(ws.cfg, 6), ids, {}, "2026-08-27")
    state["waiting"][SYM]["status"] = "tapped"
    newer = {SYM: {"week": "2026-08-31", "breakout_bar": "2026-09-02T10:00+05:30",
                   "breakout_price": 999.0}}
    harvest_waiting(state, newer, ids, {}, "2026-09-02")
    assert state["waiting"][SYM]["status"] == "tapped"
    assert state["waiting"][SYM]["breakout_price"] == 105.5    # reference untouched


def test_a_still_waiting_symbol_adopts_a_fresh_breakout(ws):
    state = empty_state()
    ids = ob_tap_scan.resolve_universe(ws.cfg)
    harvest_waiting(state, weekly_alerts(ws.cfg, 6), ids, {}, "2026-08-27")
    newer = {SYM: {"week": "2026-08-31", "breakout_bar": "2026-09-02T10:00+05:30",
                   "breakout_price": 120.0}}
    harvest_waiting(state, newer, ids, {}, "2026-09-02")
    assert state["waiting"][SYM]["week"] == "2026-08-31"
    assert state["waiting"][SYM]["breakout_price"] == 120.0


def test_active_waiting_excludes_resolved_symbols(ws):
    state = empty_state()
    harvest_waiting(state, weekly_alerts(ws.cfg, 6),
                    ob_tap_scan.resolve_universe(ws.cfg), {}, "2026-08-27")
    assert active_waiting(state) == [OLD, SYM]
    state["waiting"][OLD]["status"] = "invalid"
    state["waiting"][SYM]["status"] = "tapped"
    assert active_waiting(state) == []


# --------------------------------------------------------------------------- #
#  Sessions and timestamps
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("when,closed", [
    (datetime(2026, 8, 27, 9, 14, tzinfo=IST), False),   # Thursday pre-open
    (datetime(2026, 8, 27, 11, 0, tzinfo=IST), False),   # Thursday mid-session
    (datetime(2026, 8, 27, 15, 30, tzinfo=IST), False),  # the bell, feed settling
    (datetime(2026, 8, 27, 15, 35, tzinfo=IST), True),   # grace expired
    (datetime(2026, 8, 29, 11, 0, tzinfo=IST), True),    # Saturday
])
def test_session_closed(ws, when, closed):
    assert session_closed(when, ws.cfg) is closed


def test_as_date_normalises_everything_the_feed_can_return():
    assert as_date(pd.Timestamp("2026-08-27 09:15", tz="Asia/Kolkata")) == date(2026, 8, 27)
    assert as_date(datetime(2026, 8, 27, 9, 15)) == date(2026, 8, 27)
    assert as_date(date(2026, 8, 27)) == date(2026, 8, 27)
    assert as_date("2026-08-27 09:15:00+05:30") == date(2026, 8, 27)
    assert as_date(None) is None and as_date("nonsense") is None


# --------------------------------------------------------------------------- #
#  State file
# --------------------------------------------------------------------------- #
def test_state_round_trips_and_leaves_no_temp_file(tmp_path):
    path = tmp_path / "ob_precision_state.json"
    data = empty_state()
    data["waiting"][SYM] = {"symbol": SYM, "status": "waiting"}
    save_state(path, data)
    assert not path.with_suffix(".tmp").exists()          # tmp file was replaced
    back = load_state(path)
    assert back["waiting"][SYM]["status"] == "waiting"
    assert back["logic_version"] == STATE_LOGIC_VERSION
    assert back["updated_at"]


def test_a_corrupt_state_file_restarts_quietly(tmp_path):
    path = tmp_path / "ob_precision_state.json"
    path.write_text("{not json")
    assert load_state(path) == empty_state()
    assert load_state(tmp_path / "missing.json") == empty_state()
    path.write_text("[1, 2, 3]")
    assert load_state(path) == empty_state()


def test_a_state_version_change_keeps_the_waiting_list_and_alert_history(tmp_path):
    """Losing the de-dupe keys would re-alert names that already fired."""
    path = tmp_path / "ob_precision_state.json"
    path.write_text(json.dumps({
        "logic_version": STATE_LOGIC_VERSION - 1,
        "waiting": {SYM: {"symbol": SYM, "status": "waiting"}},
        "zones": {SYM: {"as_of": "2020-01-01", "zones": []}},
        "alerts": {"k": {"sent_at": "x"}},
    }))
    back = load_state(path)
    assert back["waiting"] == {SYM: {"symbol": SYM, "status": "waiting"}}
    assert back["alerts"] == {"k": {"sent_at": "x"}}
    assert back["zones"] == {}                            # derived: rebuilt


# --------------------------------------------------------------------------- #
#  End to end
# --------------------------------------------------------------------------- #
def test_a_live_tap_alerts_and_resolves_the_symbol(ws):
    ws.arm(quote=TAP_QUOTE)                               # closed through 2026-08-25
    ws.arm(OLD, OLD_SID, quote=TAP_QUOTE)
    before = (ws.dir / "state.json").read_bytes()

    assert ws.run() == 0

    text = "\n".join(ws.tg.sent)
    assert "TAP 1" in text
    assert SYM in text and OLD in text
    assert "100.60" in text                               # the session low
    assert "not close-confirmed" in text or "TAP 1" in text
    assert "Weekly breakout" in text                      # the stage-1 reference
    assert "26W" in text and "104.00" in text             # the level it cleared

    st = ws.state()
    assert st["waiting"][SYM]["status"] == "tapped"
    assert st["waiting"][SYM]["resolved_reason"].startswith("tap 1")
    assert [k for k in st["alerts"] if k.endswith("|tap|tap1")]
    # The weekly scanner's file is untouched - byte for byte.
    assert (ws.dir / "state.json").read_bytes() == before


def test_nothing_is_alerted_twice(ws):
    ws.arm(quote=TAP_QUOTE)
    assert ws.run() == 0
    first = list(ws.tg.sent)
    # The symbol resolved to "tapped", so it leaves the active list entirely.
    assert ws.run() == 0
    assert ws.tg.sent == first


def test_a_repeated_quote_of_the_same_session_does_not_double_count(ws):
    """
    Two runs, one session: the second must not turn Tap 1 into Tap 2. The zones
    are reloaded from state (never mutated across runs) and the alert key is
    identical, so the second run is silent.
    """
    ws.configure(**{"alert_kinds": ["tap"], "resolve_on_tap": False})
    ws.arm(quote=TAP_QUOTE)
    assert ws.run() == 0
    assert ws.run() == 0
    taps = [t for t in ws.tg.sent if "TAP" in t]
    assert len(taps) == 1 and "TAP 1" in taps[0]
    assert "TAP 2" not in "\n".join(ws.tg.sent)


def test_one_history_call_per_session_then_only_bulk_quotes(ws):
    """
    The cost model: a per-symbol daily fetch once per session, then ONE bulk
    quote for the whole list on every later run.
    """
    ws.arm(quote=FLAT_QUOTE)
    ws.arm(OLD, OLD_SID, quote=FLAT_QUOTE)
    assert ws.run() == 0
    assert sorted(ws.client.history_calls) == [OLD, SYM]
    assert len(ws.client.ohlc_calls) == 1                 # ONE bulk request
    assert ws.run() == 0
    assert len(ws.client.history_calls) == 2              # no re-fetch
    assert len(ws.client.ohlc_calls) == 2


def test_symbols_without_a_zone_are_not_quoted(ws):
    """Nothing to tap, nothing to pay for: OLDSYM has no history at all."""
    ws.arm(quote=FLAT_QUOTE)
    assert ws.run() == 0
    quoted = [str(s) for call in ws.client.ohlc_calls for sids in call.values()
              for s in sids]
    assert quoted == [SID]


def test_max_refresh_per_run_spreads_the_backfill_over_runs(ws):
    ws.configure(**{"max_refresh_per_run": 1})
    ws.arm(quote=FLAT_QUOTE)
    ws.arm(OLD, OLD_SID, quote=FLAT_QUOTE)
    assert ws.run() == 0
    assert len(ws.client.history_calls) == 1
    assert ws.run() == 0
    assert len(ws.client.history_calls) == 2              # the rollover catches up


def test_a_stale_quote_does_not_replay_yesterday_as_today(ws):
    """
    A bulk quote carries no date. On a weekday market holiday the feed returns
    the previous session unchanged; replaying it as "today" would count a second
    tap on a zone that was already tapped.
    """
    bars = scenario()
    ws.client.frames[SYM] = frame(bars)
    last = bars[-1]
    ws.client.quotes[SID] = {"open": last.open, "high": last.high, "low": last.low,
                             "last_price": last.close, "prev_close": last.close,
                             "volume": last.volume}
    assert ws.run() == 0
    assert "TAP" not in "\n".join(ws.tg.sent)
    assert not [k for k in ws.state()["alerts"] if "|tap" in k]


def test_ancient_history_is_not_alerted_on_the_backfill_run(ws):
    """
    The first refresh replays ~250 sessions, which contain order blocks born
    months ago. Only recent closed-bar events are alertable; a live tap today
    never is filtered.
    """
    ws.arm(bars=scenario(shift_days=30), quote=TAP_QUOTE)
    assert ws.run() == 0
    text = "\n".join(ws.tg.sent)
    assert "TAP 1" in text                       # live, today: always alertable
    assert "new OB" not in text and "PRECISION OB" not in text


def test_a_recent_closed_order_block_is_alerted(ws):
    ws.configure(**{"alert_kinds": ["ob"]})
    ws.arm(bars=scenario(), quote=FLAT_QUOTE)             # born 2026-08-23
    assert ws.run() == 0
    assert "PRECISION OB" in "\n".join(ws.tg.sent)


def test_alert_kinds_and_tap_numbers_are_config_driven(ws):
    ws.configure(**{"alert_kinds": ["tap"], "alert_taps": [1]})
    ws.arm(quote=TAP_QUOTE)
    assert ws.run() == 0
    assert len(ws.tg.sent) == 1
    assert "PRECISION OB" not in ws.tg.sent[0] and "TAP 1" in ws.tg.sent[0]


def test_alert_taps_can_select_a_later_touch(ws):
    ws.configure(**{"alert_kinds": ["tap"], "alert_taps": [2]})
    # The closed history already contains Tap 1 (2026-08-26), so today's live
    # touch is Tap 2 - and Tap 2 is what the config asks for.
    ws.arm(bars=scenario() + [TAP_BAR], quote=TAP_QUOTE)
    assert ws.run() == 0
    assert "TAP 2" in "\n".join(ws.tg.sent)


def test_post_close_run_replays_todays_bar_as_closed(ws, monkeypatch):
    """15:37 IST: today's daily bar is finished, so it joins the closed replay."""
    monkeypatch.setattr(ob_tap_scan, "_now",
                        lambda: datetime(2026, 8, 27, 15, 37, tzinfo=IST))
    ws.configure(**{"alert_kinds": ["tap"]})
    ws.arm(bars=scenario() + [Bar(101.0, 103.0, 100.60, 102.5, 260.0, date(2026, 8, 27))],
           quote=TAP_QUOTE)
    assert ws.run() == 0
    text = "\n".join(ws.tg.sent)
    assert "TAP 1" in text and "closed-bar touch" in text
    assert ws.client.ohlc_calls == []                     # no live pass needed
    assert ws.state()["zones"][SYM]["as_of"] == "2026-08-27"


def test_all_zones_dead_takes_the_symbol_off_the_list(ws):
    ws.configure(**{"alert_kinds": ["tap"]})
    # A bar that closes below the structural stop kills the only zone this name
    # ever produced - the "or invalidated" half of the waiting-list rule.
    killer = Bar(105.0, 106.0, 90.0, 91.0, 400.0, D0 + timedelta(days=23 - 30))
    ws.arm(bars=scenario(shift_days=30) + [killer], quote=FLAT_QUOTE)
    assert ws.run() == 0
    rec = ws.state()["waiting"][SYM]
    assert rec["status"] == "invalid"
    assert "invalidated" in rec["resolved_reason"]
    assert SYM not in active_waiting(ws.state())


def test_a_dead_list_with_no_weekly_alerts_is_a_red_run(ws):
    (ws.dir / "state.json").write_text(json.dumps({"weeks": {}}))
    assert ws.run() == 2
    assert ws.tg.sent == []                # scan.py owns the outage alarm, not us


def test_an_empty_day_with_a_sourced_list_is_not_an_error(ws):
    ws.arm(quote=FLAT_QUOTE)
    assert ws.run("--symbols", "NOSUCH") == 0
    assert ws.client.history_calls == []


def test_enabled_false_is_a_complete_no_op(ws):
    ws.configure(**{"enabled": False})
    ws.arm(quote=TAP_QUOTE)
    assert ws.run() == 0
    assert ws.client.history_calls == [] and ws.tg.sent == []
    assert not (ws.dir / "ob_precision_state.json").exists()


def test_refresh_only_skips_the_live_pass(ws):
    ws.arm(quote=TAP_QUOTE)
    assert ws.run("--refresh-only") == 0
    assert ws.client.ohlc_calls == []
    assert ws.state()["zones"][SYM]["zones"]


def test_a_history_failure_does_not_kill_the_run(ws):
    ws.arm(quote=TAP_QUOTE)
    ws.arm(OLD, OLD_SID, quote=TAP_QUOTE)
    ws.client.fail_history = {OLD}
    assert ws.run() == 0
    assert OLD not in ws.state()["zones"]
    assert ws.state()["zones"][SYM]["zones"]              # the healthy symbol ran
    assert ws.run() == 0                                  # ...and the bad one retries
    assert ws.client.history_calls.count(OLD) == 2


def test_a_quote_failure_does_not_kill_the_run(ws):
    ws.arm(quote=TAP_QUOTE)
    ws.client.fail_ohlc = True
    assert ws.run() == 0
    assert ws.tg.sent == [] or "TAP" not in "\n".join(ws.tg.sent)
    assert ws.state()["zones"][SYM]["zones"]              # history still saved


def test_a_symbol_with_no_history_is_skipped_not_crashed_on(ws):
    ws.arm(OLD, OLD_SID, quote=FLAT_QUOTE)                # one healthy symbol
    assert ws.run() == 0
    st = ws.state()
    assert st["zones"][SYM]["no_history"] is True
    assert st["zones"][SYM]["zones"] == []
    assert st["waiting"][SYM]["status"] == "waiting"      # no zones != invalidated


def test_a_symbol_with_no_history_is_not_refetched_every_run(ws):
    """
    A suspended or delisted name has no daily history and never will intraday.
    Without this it costs one history call per name per five-minute run.
    """
    ws.arm(OLD, OLD_SID, quote=FLAT_QUOTE)
    assert ws.run() == 0
    assert ws.run() == 0
    assert ws.client.history_calls.count(SYM) == 1
    quoted = [str(s) for call in ws.client.ohlc_calls for sids in call.values()
              for s in sids]
    assert SID not in quoted                              # nothing to quote either


def test_a_transient_history_error_is_retried_next_run(ws):
    ws.arm(OLD, OLD_SID, quote=FLAT_QUOTE)                # keeps the run healthy
    ws.client.fail_history = {SYM}
    assert ws.run() == 0
    assert SYM not in ws.state()["zones"]                 # not remembered as a miss
    ws.client.fail_history = set()
    ws.arm(quote=FLAT_QUOTE)
    assert ws.run() == 0
    assert ws.state()["zones"][SYM]["zones"] is not None


def test_a_total_data_outage_fails_the_run_once_a_day(ws, monkeypatch):
    """
    Every refresh failing means the feed is down and the waiting list is going
    unwatched. The run goes red so the workflow's failure notice fires - but only
    once a day, because a five-minute cron would otherwise send 78 notices.
    """
    assert ws.run() == 3                                  # nothing fetchable at all
    assert ws.state()["data_outage_on"] == "2026-08-27"
    assert ws.run() == 0                                  # same day: already reported
    assert ws.state()["data_outage_on"] == "2026-08-27"

    # Next session with the feed back: real refreshes clear the flag, so a LATER
    # outage is still reported instead of being swallowed for good.
    monkeypatch.setattr(ob_tap_scan, "_now",
                        lambda: datetime(2026, 8, 28, 11, 0, tzinfo=IST))
    ws.arm(quote=FLAT_QUOTE)
    ws.arm(OLD, OLD_SID, quote=FLAT_QUOTE)
    assert ws.run() == 0
    assert ws.state()["data_outage_on"] is None

    # ...and a fresh outage on a later session goes red again.
    monkeypatch.setattr(ob_tap_scan, "_now",
                        lambda: datetime(2026, 8, 31, 11, 0, tzinfo=IST))
    ws.client.fail_history = {SYM, OLD}
    assert ws.run() == 3


def test_one_broken_symbol_is_not_reported_as_a_data_outage(ws):
    """
    A delisted or badly-mapped name fails every single run. That is a symbol
    problem, not a broken feed, so it must not turn the workflow red and fire a
    Telegram failure notice - not even when it is the only thing left to refresh.
    """
    ws.arm(OLD, OLD_SID, quote=FLAT_QUOTE)                # one healthy name
    ws.client.fail_history = {SYM}
    assert ws.run() == 0
    assert ws.state()["data_outage_on"] is None
    assert ws.run("--refresh-only", "--symbols", SYM) == 0
    assert ws.state()["data_outage_on"] is None


def test_failed_delivery_keeps_the_alert_pending(ws):
    """scan.py's rule: if it did not arrive, do not record it as sent."""
    ws.tg.ok = False
    ws.arm(quote=TAP_QUOTE)
    assert ws.run() == 0
    assert ws.state()["alerts"] == {}
    assert ws.state()["waiting"][SYM]["status"] == "waiting"
    ws.tg.ok = True
    assert ws.run() == 0
    assert ws.state()["alerts"]
    assert "TAP 1" in "\n".join(ws.tg.sent[1:])


def test_heartbeat_reports_the_run(ws):
    ws.arm(quote=TAP_QUOTE)
    assert ws.run("--heartbeat") == 0
    assert "Precision OB scan complete" in ws.tg.sent[-1]


def test_market_closed_is_a_quiet_exit(ws, monkeypatch):
    monkeypatch.setattr(ob_tap_scan, "_now",
                        lambda: datetime(2026, 8, 29, 11, 0, tzinfo=IST))  # Saturday
    ws.arm(quote=TAP_QUOTE)
    assert ws.run() == 0
    assert ws.client.history_calls == [] and ws.tg.sent == []
    assert ws.run("--force") == 0                         # --force overrides
    assert ws.client.history_calls


# --------------------------------------------------------------------------- #
#  Alert text
# --------------------------------------------------------------------------- #
def test_a_single_event_gets_the_full_block(ws):
    ws.configure(**{"alert_kinds": ["tap"]})
    ws.arm(quote=TAP_QUOTE)
    ws.run()
    text = ws.tg.sent[0]
    assert text.startswith("🟠 <b>TAP 1 — TESTSYM</b>")
    assert "Entry <b>100.60</b>" in text
    assert "stop" in text.lower()
    assert "+1.44%" in text                    # 105.5 over the 104.0 26-week level


def test_several_events_are_batched_into_one_message(ws):
    ws.arm(quote=TAP_QUOTE)
    ws.arm(OLD, OLD_SID, quote=TAP_QUOTE)
    ws.run()
    assert len(ws.tg.sent) == 1
    assert "precision-OB events" in ws.tg.sent[0]
    assert ws.tg.sent[0].count("🟠") == 2


def test_the_header_is_unmistakable_next_to_the_weekly_scanner(ws):
    """The user asked for a distinct header so the two systems never get confused."""
    ws.arm(quote=TAP_QUOTE)
    ws.run()
    text = "\n".join(ws.tg.sent)
    assert "🟢 <b>BUY" not in text             # scan.py's header, never reused
    assert text.startswith("🎯") or text.startswith("🟠")


def test_symbols_are_html_escaped():
    ev = {"kind": "tap", "tap_number": 1, "entry": 1.0, "low": 0.9, "top": 1.1,
          "bottom": 0.8, "stop": 0.7, "atr": 0.1, "session": "2026-08-27",
          "born_session": "2026-08-20", "confirmed": False, "price": 1.0,
          "detail": {}}
    rec = {"exchange_segment": "NSE_EQ", "breakout_bar": "", "breakout_price": None,
           "level_26w": None}
    text = format_tap(ev, rec, "A<B&C")
    assert "A&lt;B&amp;C" in text and "A<B&C" not in text


def test_a_closed_tap_says_so():
    ev = {"kind": "tap", "tap_number": 1, "entry": 1.0, "low": 0.9, "top": 1.1,
          "bottom": 0.8, "stop": 0.7, "atr": 0.1, "session": "2026-08-26",
          "born_session": "2026-08-20", "confirmed": True, "price": 1.0,
          "detail": {"next_entry": 1.02}}
    rec = {"exchange_segment": "NSE_EQ", "breakout_bar": "2026-08-20T10:00+05:30",
           "breakout_price": 105.5, "level_26w": 104.0}
    text = format_tap(ev, rec, "SYM")
    assert "closed-bar touch" in text and "not close-confirmed" not in text
    assert "Next pre-order after this tap" in text


def test_a_no_trade_session_does_not_age_a_zone(ws):
    """
    End-to-end shape of the SMSPHARMA bug. History is quiet bars, the displacement
    and the departure, then a session with volume 0. The live bar is therefore
    only age 2, and `minAge` is 3, so there is no Tap 1 - even though the quote
    sits exactly on the pre-order entry. Counting the phantom made this fire a
    session early: 2026-09-16 instead of the chart's 2026-09-17.
    """
    ws.configure(**{"alert_kinds": ["tap"]})
    ws.arm(bars=scenario()[:22] + [PHANTOM], quote=TAP_QUOTE)
    assert ws.run() == 0
    assert ws.tg.sent == []


def test_a_no_trade_session_still_taps_one_session_later(ws):
    """The same series plus ONE real session: now the live bar is age 3."""
    ws.configure(**{"alert_kinds": ["tap"]})
    ws.arm(bars=scenario()[:22] + [PHANTOM,
                                   Bar(111.5, 112.5, 109.0, 110.0, 120.0,
                                       D0 + timedelta(days=23))],
           quote=TAP_QUOTE)
    assert ws.run() == 0
    assert "TAP 1" in "\n".join(ws.tg.sent)


# --------------------------------------------------------------------------- #
#  Real data: SMSPHARMA, September 2026
# --------------------------------------------------------------------------- #
SMS_DAILY = Path(__file__).parent / "test_smspharma_daily.csv"


def _walk(bars, params, first="2026-09-11"):
    """
    Walk the bars the way the scanner walks them: for each session, replay the
    history that is CLOSED (everything strictly before it) and judge that session
    live from its own high/low. Returns the live events as
    (session, kind, tap_number, entry, stop).
    """
    out = []
    for i, b in enumerate(bars):
        if str(b.session) < first:
            continue
        closed = bars[:i]
        res = ob_precision.replay(closed, params)
        ctx = res.context(closed, params)
        for e in ob_precision.live_pass(ctx["zones"], ctx, params, open_=b.open,
                                        high=b.high, low=b.low,
                                        last_price=b.close, volume=b.volume,
                                        session=b.session):
            out.append((str(b.session), e.kind, int(e.tap_number or 0),
                        round(e.zone.entry, 2), round(e.zone.stop, 2)))
    return out


def test_real_smspharma_tap1_is_the_17th_not_the_16th():
    """
    Real bars, real chart, the bug this fixture exists for.

    SMSPHARMA broke out on Fri 2026-09-11 (417.85 -> 463.45 on 10.4m shares,
    about 11x its 20-day average). Monday 2026-09-14 came back from the feed as
    O=H=L=C=463.45 with volume 0: a session in which nothing traded. The chart
    draws no bar for it, so Pine's `age = bar_index - born` never counts it, and
    `age >= minAge` (3) puts the first legal tap on Thu 2026-09-17 - which is
    what the indicator shows. A port that counted the no-trade session aged the
    zone one bar too fast and fired Tap 1 on Wed 2026-09-16.

    There are four such rows in this one year of SMSPHARMA (2026-01-15,
    2026-05-28, 2026-06-26, 2026-09-14), plus one all-null market holiday, so
    this is a systematic property of the feed rather than a one-off glitch.
    """
    df = pd.read_csv(SMS_DAILY)
    bars = bars_from_frame(df.rename(columns={"date": "datetime"}))
    assert len(df) == 252
    assert len(bars) == 247                  # 4 no-trade sessions + 1 holiday
    sessions = {b.session for b in bars}
    assert "2026-09-14" not in sessions and "2026-05-01" not in sessions

    ev = _walk(bars, ob_precision.OBParams())
    taps = [r for r in ev if r[1] == "tap" and r[2] == 1]
    assert [r[0] for r in taps] == ["2026-09-17"]      # NOT the 15th or 16th
    assert not [r for r in ev if r[1] == "tap" and r[0] < "2026-09-17"]
    assert taps[0][3] == 406.18              # frozen pre-order entry
    # Stop is bottom - stopATR*ATR, and ATR is a Wilder RMA over the bar series -
    # so dropping the four no-trade sessions moves it by a paisa (399.23 -> 399.22)
    # and brings it onto the same series the chart computes it from.
    assert taps[0][4] == 399.22


def test_counting_a_no_trade_session_fires_the_tap_a_day_early():
    """
    The same walk with the phantom rows kept - i.e. the pre-fix behaviour - to
    prove the test above actually discriminates, and to pin the symptom so that
    loosening the filter in bars_from_frame fails loudly.
    """
    df = pd.read_csv(SMS_DAILY).dropna()     # keeps the four zero-volume rows
    bars = [ob_precision.Bar(float(r.open), float(r.high), float(r.low),
                             float(r.close), float(r.volume), r.date)
            for r in df.itertuples()]
    taps = [r for r in _walk(bars, ob_precision.OBParams())
            if r[1] == "tap" and r[2] == 1]
    assert [r[0] for r in taps] == ["2026-09-16"]     # the wrong answer


PGIL_DAILY = Path(__file__).parent / "test_pgil_daily.csv"


def test_real_pgil_tap1_is_the_8th_of_july():
    """
    A second symbol, a second shape of the same story.

    PEARL GLOBAL (PGIL) displaced on Wed 2026-06-24: 932.55 -> 1040.50 (+10.2%)
    on 5,228,316 shares against a ~200k average, and that same session took out
    the frozen 26-week level (985.05) the weekly scanner had been waiting on.
    The origin candle is Tue 2026-06-23, so the zone is its open->low,
    922.75-946.80, and the pre-order entry freezes at 955.05 with the stop at
    915.87.

    Fri 2026-06-26 then came back O=H=L=C=1035.50 with volume 0 - the third of
    five no-trade sessions in PGIL's year, and the same phantom that broke
    SMSPHARMA. Price first reaches the entry on Wed 2026-07-08 (low 950.65),
    when the zone is 9 bars old, comfortably past minAge 3.
    """
    df = pd.read_csv(PGIL_DAILY)
    bars = bars_from_frame(df.rename(columns={"date": "datetime"}))
    assert len(df) == 252
    assert len(bars) == 247                       # five no-trade sessions
    assert "2026-06-26" not in {b.session for b in bars}

    ev = _walk(bars, ob_precision.OBParams(), first="2026-06-24")
    taps = [r for r in ev if r[1] == "tap" and r[2] == 1]
    assert [r[0] for r in taps] == ["2026-07-08"]
    assert taps[0][3] == 955.05                   # frozen pre-order entry
    assert taps[0][4] == 915.87                   # bottom - stopATR * ATR
    # the pullback of 29-30 June (lows 995.00 / 985.55) never reached the entry,
    # so nothing fires in between - the zone is not tapped on the way down
    assert not [r for r in ev if r[1] == "tap" and r[0] < "2026-07-08"]


def test_a_phantom_bar_far_from_the_age_gate_still_moves_the_frozen_levels():
    """
    The complement to the SMSPHARMA test above, and the reason the filter is not
    optional even when it looks harmless.

    PGIL's tap lands 9 bars after birth, so counting the 2026-06-26 no-trade
    session does NOT change the DATE - it only ages the zone to 10 instead of 9,
    and both clear minAge 3. What it does change is every number the alert
    prints: ATR is a Wilder RMA over the bar series, so one phantom bar pulls it
    from 42.84 to 40.59 and with it the frozen entry (955.05 -> 954.93) and stop
    (915.87 -> 915.97). Those are the levels a live trade is sized against.

    So the filter is not "a fix for one unlucky symbol in September": either the
    series matches the chart's bars or the levels are wrong, and only sometimes
    is the date wrong too.
    """
    df = pd.read_csv(PGIL_DAILY)
    bars = [ob_precision.Bar(float(r.open), float(r.high), float(r.low),
                             float(r.close), float(r.volume), r.date)
            for r in df.itertuples()]            # bypasses the filter
    taps = [r for r in _walk(bars, ob_precision.OBParams(), first="2026-06-24")
            if r[1] == "tap" and r[2] == 1]
    assert [r[0] for r in taps] == ["2026-07-08"]  # date survives
    assert (taps[0][3], taps[0][4]) == (954.93, 915.97)   # levels do not


def test_the_next_entry_line_says_whether_the_level_actually_moved():
    """
    Real SMSPHARMA, 2026-09-16: the tap was defended at 393.25, far BELOW the
    406.18 pre-order, so math.max leaves the entry where it was. The message must
    not claim it was raised.
    """
    rec = {"exchange_segment": "NSE_EQ", "breakout_bar": "2026-09-11T10:00+05:30",
           "breakout_price": 455.05, "level_26w": 447.8}
    base = {"kind": "tap", "tap_number": 1, "low": 393.25, "top": 405.0,
            "bottom": 402.05, "stop": 399.23, "atr": 22.27, "session": "2026-09-16",
            "born_session": "2026-09-11", "confirmed": False, "price": 402.95}
    unchanged = dict(base, entry=406.18,
                     detail={"tapped_entry": 406.18, "next_entry": 406.18})
    text = format_tap(unchanged, rec, "SMSPHARMA")
    assert "unchanged" in text and "raised" not in text

    raised = dict(base, entry=394.36,
                  detail={"tapped_entry": 393.25, "next_entry": 394.36})
    text = format_tap(raised, rec, "SMSPHARMA")
    assert "raised above the defended low" in text and "unchanged" not in text


def test_the_breakout_line_degrades_without_a_level():
    line = ob_tap_scan._breakout_line({"breakout_bar": "2026-08-26T10:00+05:30",
                                       "breakout_price": 105.5, "level_26w": None})
    assert "105.50" in line and "26W" not in line
    assert "10:00 IST" in line


def test_heartbeat_counts():
    text = format_heartbeat(367, 250, 41, 2, 3, 12.5, FIXED)
    assert "Waiting list 367" in text and "refreshed 250" in text
    assert "live-quoted 41" in text and "alerts <b>2</b>" in text
    assert "3 symbol error(s)" in text
    assert "symbol error" not in format_heartbeat(1, 1, 1, 0, 0, 1.0, FIXED)


def test_a_new_order_block_message_names_the_origin_candle(ws):
    ws.configure(**{"alert_kinds": ["ob"]})
    ws.arm(bars=scenario(), quote=FLAT_QUOTE)
    ws.run()
    text = "\n".join(ws.tg.sent)
    assert "PRECISION OB — TESTSYM" in text
    assert "100.00" in text and "98.00" in text        # zone top / bottom
    assert "2026-08-23" in text                        # the displacement session
    assert "2026-08-22" in text                        # the origin candle's session
    assert "never recomputed" in text                  # the non-repaint promise


# --------------------------------------------------------------------------- #
#  State size and commit churn
# --------------------------------------------------------------------------- #
def test_an_unchanged_run_does_not_rewrite_the_state_file(ws):
    """
    The workflow commits this file, and `updated_at` moves on every run, so an
    unconditional write would mean ~78 commits a day for unchanged contents.
    """
    ws.arm(quote=FLAT_QUOTE)
    assert ws.run() == 0
    first = (ws.dir / "ob_precision_state.json").read_bytes()
    assert ws.run() == 0
    assert (ws.dir / "ob_precision_state.json").read_bytes() == first


def test_save_state_skips_a_timestamp_only_change(tmp_path):
    path = tmp_path / "s.json"
    data = empty_state()
    data["waiting"][SYM] = {"symbol": SYM, "status": "waiting"}
    assert save_state(path, data, json.loads(json.dumps(data))) is False
    assert not path.exists()
    assert save_state(path, data) is True                  # no baseline -> write
    assert path.exists()
    changed = json.loads(path.read_text())
    changed["waiting"][SYM]["status"] = "tapped"
    assert save_state(path, changed, json.loads(path.read_text())) is True


def test_prune_closed_events_keeps_only_what_could_still_be_sent():
    ctx = {"closed_events": [
        {"kind": "ob", "session": "2026-08-25"},                 # recent, enabled
        {"kind": "ob", "session": "2026-01-05"},                 # ancient
        {"kind": "tap", "session": "2026-08-26", "tap_number": 1},
        {"kind": "tap", "session": "2026-08-26", "tap_number": 3},   # not alerted
        {"kind": "invalid", "session": "2026-08-26"},            # kind disabled
    ]}
    prune_closed_events(ctx, ["ob", "tap"], [1], "2026-08-22")
    assert [(e["kind"], e.get("tap_number")) for e in ctx["closed_events"]] == [
        ("ob", None), ("tap", 1)]


def test_the_state_file_stays_small_enough_to_commit(ws):
    """~370 waiting names must not turn into megabytes of committed JSON."""
    ws.arm(quote=TAP_QUOTE)
    ws.arm(OLD, OLD_SID, quote=TAP_QUOTE)
    ws.run()
    size = (ws.dir / "ob_precision_state.json").stat().st_size
    per_symbol = size / 2
    assert per_symbol < 8192, f"{per_symbol:.0f} bytes per symbol is too fat to commit"
