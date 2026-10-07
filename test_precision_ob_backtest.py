"""Tests for precision_ob_backtest.py - the research tool behind the alert.

Two things must hold or the numbers the alert quotes are fiction:

  1. it MEASURES THE SHIPPED RULE - the cycle walk agrees with the live
     `ob_tap_scan.derive_26w_breakout`, and the exits are the alert's own
     `walk_bars` (not a second implementation that can drift);
  2. it is READ-ONLY - no Telegram, no state file, no writes, exactly like
     btst_backtest.py (test_bug78's contract for research tools).

No network: every fixture below is synthetic daily candles.
"""

from __future__ import annotations

import re
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import precision_ob_backtest as pbt  # noqa: E402  (import path guard)
from ob_precision import Bar, OBParams  # noqa: E402
from ob_tap_scan import derive_26w_breakout  # noqa: E402

PARAMS = OBParams()
ROOT = Path(__file__).resolve().parent


def bar(session, o, h, l, c, v=1000.0):
    return Bar(open=o, high=h, low=l, close=c, volume=v, time=session)


def quiet_days(upto, start=date(2025, 1, 6), px=90.0):
    """Flat weekday sessions in [start, upto] - warm-up for the weekly level
    and the ATR. Ends strictly before `upto` so explicit cycle bars never
    collide with the base."""
    out, d = [], start
    while d < upto:
        if d.weekday() < 5:
            out.append(bar(d.isoformat(), px, px + 1, px - 1, px))
        d += timedelta(days=1)
    return out


# --------------------------------------------------------------------------- #
#  Cycles - the live rule, forward
# --------------------------------------------------------------------------- #
def test_cycles_locks_a_second_cross_inside_the_26_week_window():
    """Two crosses four weeks apart are ONE cycle: the lock starts at the
    first alert of a cycle (runtime.breakout_cooldown_weeks)."""
    bars = quiet_days(date(2025, 8, 8))          # ~31 weeks of base
    bars += [
        bar("2025-08-11", 95, 100, 94, 99),      # week 1: cross
        bar("2025-08-15", 99, 105, 98, 104),
        bar("2025-09-08", 104, 110, 103, 109),   # week 5: cross again
    ]
    cyc = pbt.cycles(bars, date(2025, 1, 1))
    assert len(cyc) == 1
    assert cyc[0]["session"] == "2025-08-11"


def test_cycles_starts_a_fresh_cycle_after_the_lock_expires():
    """27 weeks of trading inside the lock is still ONE cycle; the fresh cross
    after the lock is the next one."""
    bars = quiet_days(date(2025, 8, 8))                 # 90/91 base
    bars += [bar("2025-08-11", 95, 100, 94, 99),        # the cross
             bar("2025-08-15", 99, 105, 98, 104)]
    bars += quiet_days(date(2026, 2, 13), start=date(2025, 8, 18), px=97.0)
    bars += quiet_days(date(2026, 3, 16), start=date(2026, 2, 16), px=110.0)
    bars += [bar("2026-03-16", 115, 122, 114, 121)]     # the fresh cross
    cyc = pbt.cycles(bars, date(2025, 1, 1))
    # the 110 base gapping up off a 97 base IS the fresh cross (level = the
    # 26-week high it cleared, 98), and it is 27 weeks after the first one
    assert [c["session"] for c in cyc] == ["2025-08-11", "2026-02-16"]
    assert cyc[0]["level"] == 91.0 and cyc[1]["level"] == 98.0


def test_cycles_never_dates_a_cycle_before_the_start_date():
    bars = quiet_days(date(2025, 8, 8)) + [bar("2025-08-11", 95, 100, 94, 99)]
    assert pbt.cycles(bars, date(2026, 1, 1)) == []


def test_cycles_agrees_with_the_live_derive_26w_breakout():
    """The property the report depends on: the enumeration's LAST cycle is the
    cycle the live scanner would date today."""
    bars = quiet_days(date(2025, 8, 8)) + [
        bar("2025-08-11", 95, 100, 94, 99),
        bar("2025-08-15", 99, 105, 98, 104),
        bar("2025-08-18", 104, 108, 103, 107),
    ]
    live = derive_26w_breakout(bars)
    mine = pbt.cycles(bars, date(2000, 1, 1))
    assert live is not None and mine
    assert mine[-1]["session"] == live["session"]
    assert abs(mine[-1]["level"] - live["level"]) < 1e-9


# --------------------------------------------------------------------------- #
#  The funnel
# --------------------------------------------------------------------------- #
CYCLE = [
    bar("2025-08-11", 95, 100, 94, 99),      # cross: close 99 > the 90/91 base
    bar("2025-08-15", 99, 105, 98, 104),
    bar("2025-09-15", 112, 118, 111, 117),
    bar("2025-09-16", 117, 120, 116, 119),   # the swing high: 120
    bar("2025-09-17", 115, 116, 108, 110),   # the OB candle (close 110)
    bar("2025-09-18", 112, 124, 111, 123, v=9000.0),   # the displacement
]


def cycle_bars(extra=()):
    """One clean cycle: the base, the cross, a pullback whose OB candle closes
    above the level, and the displacement that births the order block."""
    return quiet_days(date(2025, 8, 8)) + list(CYCLE) + list(extra)


def test_the_funnel_counts_one_event_and_one_taken_trade():
    bars = cycle_bars()
    first = next(iter(pbt.cycles(bars, date(2025, 1, 1))))
    r = pbt.run_symbol(bars, PARAMS, date(2025, 1, 1), 90, 0.22)
    assert r["cycles"] == 1 and r["with_ob"] == 1 and r["events"] == 1
    assert r["pullback"] == 1 and r["predates"] == 0 and r["below"] == 0
    assert len(r["trades"]) + r["censored"] == 1
    assert r["trades"] or r["censored"] == 1
    assert first["session"] == "2025-08-11"


def test_an_ob_candle_below_the_level_is_excluded_not_traded():
    bent = list(CYCLE)
    bent[4] = bar("2025-09-17", 95, 96, 88, 90)      # close back below the level
    r = pbt.run_symbol(cycle_bars(), PARAMS, date(2025, 1, 1), 90, 0.22)
    assert r["trades"]                                # the clean fixture trades
    r = pbt.run_symbol(quiet_days(date(2025, 8, 8)) + bent, PARAMS,
                       date(2025, 1, 1), 90, 0.22)
    assert r["events"] == 1 and r["below"] == 1 and not r["trades"]


def test_a_birth_after_the_lock_belongs_to_the_next_cycle():
    """The event is the first OB born inside the cycle's own 26-week window. A
    birth after it is the NEXT cycle's event - both cycles see the same OB,
    only one may claim it."""
    bars = quiet_days(date(2025, 8, 8)) + list(CYCLE[:4])
    bars += quiet_days(date(2026, 5, 4), start=date(2025, 10, 1), px=105.0)
    bars += [bar("2026-05-04", 105, 118, 104, 117, v=9000.0)]  # late birth
    r = pbt.run_symbol(bars, PARAMS, date(2025, 1, 1), 90, 0.22)
    assert r["cycles"] == 2
    assert r["with_ob"] == 2          # both cycles spot the same birth
    assert r["events"] == 1           # only the cycle whose window holds it


# --------------------------------------------------------------------------- #
#  Exits are the alert's own walker
# --------------------------------------------------------------------------- #
def test_simulate_uses_the_alert_s_fill_policy():
    """A gap through the target fills at the open - better than the limit -
    which is precision_ob_entry.resolve_bar's rule, reached through
    walk_bars."""
    bars = [bar("2025-09-17", 115, 116, 108, 110),
            bar("2025-09-18", 120, 126, 119, 125)]
    sim = pbt.simulate(bars, 110.0, 100.0, 120.0, "2025-09-17", 90, 0.22)
    assert sim["outcome"] == "win" and sim["price"] == 120.0
    assert sim["gross"] > 9.0 and sim["net"] == sim["gross"] - 0.22


def test_simulate_time_stops_at_the_ninety_session_close():
    bars = [bar("2025-01-06", 110, 111, 109, 110)]
    d = date(2025, 1, 7)
    while len(bars) < 92:
        if d.weekday() < 5:
            bars.append(bar(d.isoformat(), 110, 111, 109, 110))
        d += timedelta(days=1)
    sim = pbt.simulate(bars, 105.0, 100.0, 200.0, "2025-01-06", 90, 0.22)
    assert sim["outcome"] == "timeout" and sim["sessions"] == 90


def test_simulate_is_none_while_the_data_runs_out():
    bars = [bar("2026-09-24", 110, 111, 109, 110),
            bar("2026-09-25", 110, 111, 109, 110)]
    assert pbt.simulate(bars, 110.0, 100.0, 200.0, "2026-09-24", 90, 0.22) is None


# --------------------------------------------------------------------------- #
#  Universe + read-only
# --------------------------------------------------------------------------- #
def test_universe_files_keeps_eq_and_be_and_drops_sme(tmp_path):
    for stem, series in (("aaa", "EQ"), ("bbb", "BE"), ("ccc", "SM")):
        (tmp_path / f"{stem}.csv").write_text(
            "Date,Open,High,Low,Close,Volume,Series\n"
            f"2026-09-25,1,2,0.5,1.5,100,{series}\n")
    got = {p.stem for p in pbt.universe_files(tmp_path)}
    assert got == {"aaa", "bbb"}


def test_universe_files_uses_the_names_last_series(tmp_path):
    (tmp_path / "ddd.csv").write_text(
        "Date,Open,High,Low,Close,Volume,Series\n"
        "2020-01-01,1,2,0.5,1.5,100,BE\n"
        "2026-09-25,1,2,0.5,1.5,100,EQ\n")
    assert [p.stem for p in pbt.universe_files(tmp_path)] == ["ddd"]


def test_the_backtest_is_read_only():
    """test_bug78's contract: a research tool must not be able to fire an
    alert, touch live state, or write a file."""
    src = (ROOT / "precision_ob_backtest.py").read_text()
    # The live entry points that can alert or mutate state - checked as CODE
    # (imports and calls), because merely naming a file in a comment is not a
    # liability: test_bug78 learned that lesson for btst_backtest.py.
    for bad in ("build_telegram", "save_state", "STATE_OWN", "STATE_IN",
                "import telegram", "AlertState", "DhanClient"):
        assert bad not in src, f"the backtest must not touch {bad}"
    for pattern in (r"open\([^)]*[\"']w", r"to_csv\(", r"write_text\("):
        assert not re.search(pattern, src), \
            f"the backtest must not write files ({pattern})"


def test_the_backtest_measures_the_shipped_rule():
    src = (ROOT / "precision_ob_backtest.py").read_text()
    for fn in ("births_from_bars", "evaluate_rule", "walk_bars"):
        assert fn in src, f"the backtest must call the live {fn}"
    assert "derive_26w_breakout" in src, \
        "the cycle walk must be checked against the live derivation"
