"""
Tests for the 2026-08-13 fix: "paper trade report / backtest don't match
the alert".

Covers:
  - actionable_priority()/build_actionable_lists() extracted from btst.py's
    main() - must reproduce the exact ranking the Telegram alert used to
    compute inline, now importable so pdf_report.py can reuse it instead of
    restating (and drifting from) the rule.
  - write_alert_state() - the JSON snapshot pdf_report.py reads instead of
    re-deriving "today's picks" from the raw, multi-run-per-day CSVs.
  - pdf_report.py's real (non-hardcoded) section builders.
"""
import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest


# --------------------------------------------------------------------------- #
#  actionable_priority / build_actionable_lists (extracted from btst.py)
# --------------------------------------------------------------------------- #
def test_actionable_priority_matches_documented_tiers():
    import btst

    # Prime #1 (ant_below, PRE>=6) and Prime #2 (fresh_A/B, rvol>=5, close_pos>=0.98)
    # both score the 500-point band and rank above a bare Tier A (400).
    prime1 = {"arm": "ant_below", "pre": 6, "rvol": 1.0, "close_pos": 0.9}
    prime2 = {"arm": "fresh_A", "pre": 0, "rvol": 6.0, "close_pos": 0.99}
    tier_a = {"arm": "fresh_A", "pre": 0, "rvol": 1.0, "close_pos": 0.9}
    aged_b = {"arm": "aged_B", "pre": 0, "rvol": 1.0, "close_pos": 0.9}
    ant_above = {"arm": "ant_above", "pre": 0, "rvol": 1.0, "close_pos": 0.9}
    fresh_b = {"arm": "fresh_B", "pre": 0, "rvol": 1.0, "close_pos": 0.9}

    scores = {k: btst.actionable_priority(v) for k, v in
              [("prime1", prime1), ("prime2", prime2), ("tier_a", tier_a),
               ("aged_b", aged_b), ("ant_above", ant_above), ("fresh_b", fresh_b)]}

    assert scores["prime1"] > scores["tier_a"] > scores["aged_b"] > scores["ant_above"] > scores["fresh_b"]
    assert scores["prime2"] > scores["tier_a"]


def _confirmed_row(symbol, close, day_ret=5.0, rvol=1.0, tier="B", fresh=True, close_pos=0.9, high=None):
    prev_close = close / (1 + day_ret / 100.0)
    return {
        "symbol": symbol, "close": close, "day_ret": day_ret, "rvol": rvol,
        "tier": tier, "fresh": fresh, "close_pos": close_pos,
        "high": high if high is not None else close, "pre": 0,
    }


def _ant_row(symbol, close, level, side="below", pre=6, day_ret=3.0, rvol=1.0, close_pos=0.9, gap_pct=1.0):
    return {
        "symbol": symbol, "close": close, "level": level, "side": side, "pre": pre,
        "day_ret": day_ret, "rvol": rvol, "close_pos": close_pos, "gap_pct": gap_pct,
    }


def test_build_actionable_lists_excludes_circuit_locked():
    import btst

    # MOCKSTOCK gained 20% today and sits essentially at its own 20% upper
    # circuit band -> get_circuit_info() must flag it locked, and it must
    # be excluded from the actionable list entirely (0 sellers - unfillable).
    locked_close = 120.0  # prev_close 100 -> +20% day_ret -> 20% band -> uc_limit 120.0 (locked at the limit)
    picks = [
        _confirmed_row("MOCKSTOCK", close=locked_close, day_ret=20.0, tier="A"),
        _confirmed_row("REALPICK", close=101.0, day_ret=1.0, tier="A"),
    ]
    ant_picks = [_ant_row("ANTPICK", close=95.0, level=97.0)]

    top3, next_ant, excluded = btst.build_actionable_lists(picks, ant_picks, ant_picks)

    excluded_syms = {s for s, _ in excluded}
    assert "MOCKSTOCK" in excluded_syms
    picked_syms = {a["symbol"] for a in top3}
    assert "MOCKSTOCK" not in picked_syms
    assert "REALPICK" in picked_syms


def test_build_actionable_lists_next_anticipated_excludes_already_taken():
    import btst

    picks = []
    ant_picks = [
        _ant_row("FIRST", close=95.0, level=97.0, pre=7),
        _ant_row("SECOND", close=50.0, level=52.0, pre=6),
        _ant_row("THIRD", close=20.0, level=21.0, pre=6),
    ]
    top3, next_ant, excluded = btst.build_actionable_lists(picks, ant_picks, ant_picks)

    # All 3 anticipate rows are eligible for top3_actionable (no confirmed
    # picks competing), so whichever land in top3 must NOT reappear in
    # next_anticipated, and next_anticipated must be capped at 3
    # (2026-08-14: the watchlist carries the nearest THREE, was two).
    taken = {a["symbol"] for a in top3}
    assert not (taken & {r["symbol"] for r in next_ant})
    assert len(next_ant) <= 3


def test_build_actionable_lists_empty_when_nothing_qualifies():
    import btst

    top3, next_ant, excluded = btst.build_actionable_lists([], [], [])
    assert top3 == []
    assert next_ant == []
    assert excluded == []


# --------------------------------------------------------------------------- #
#  write_alert_state
# --------------------------------------------------------------------------- #
def test_write_alert_state_round_trips(tmp_path, monkeypatch):
    import btst
    from datetime import datetime

    class FakeCfg:
        paths = {"root": tmp_path}

    top3 = [{"symbol": "ABC", "badge": "🟢 Confirmed", "price": 100.0}]
    next_ant = [{"symbol": "XYZ", "close": 50.0, "level": 52.0}]
    excluded = [("LOCKED1", "Locked at Upper Circuit +10%")]

    now = datetime(2026, 8, 13, 15, 16, 38)
    btst.write_alert_state(FakeCfg(), now, "2026-08-13", "15:16", False, False,
                            top3, next_ant, excluded)

    out_path = tmp_path / btst.ALERT_STATE_FILE
    assert out_path.exists()
    state = json.loads(out_path.read_text())
    assert state["date"] == "2026-08-13"
    assert state["scan_time"] == "15:16"
    assert state["top3_actionable"] == top3
    assert state["next_anticipated"] == next_ant
    assert state["excluded_locked"] == [["LOCKED1", "Locked at Upper Circuit +10%"]]
    assert state["too_late"] is False


# --------------------------------------------------------------------------- #
#  pdf_report.py - no more hardcoded/fabricated data
# --------------------------------------------------------------------------- #
def test_pdf_report_has_no_hardcoded_fake_symbols():
    """Regression guard for the original bug: these literal fake trading
    symbols/numbers must never again appear as unconditional source code in
    pdf_report.py - they were the hardcoded 'demo' data that shipped instead
    of a real computation from ab_ledger.csv / anticipate_picks.csv."""
    # Skip the module docstring (the first """..."""), which legitimately
    # documents the OLD bug's fake numbers/symbols for posterity - only the
    # executable code below it must be free of them.
    src = Path("pdf_report.py").read_text()
    first = src.find('"""')
    second = src.find('"""', first + 3)
    code_only = src[second + 3:] if second != -1 else src

    fake_literals = [
        '"LUMAXTECH"', '"AARTIPHARM"', '"SHAILY"',  # hardcoded "yesterday's results"
        '"COMSYN"', '"SBCL"',                        # hardcoded "open positions" (as a literal list)
        '"HAPPYFORGE"', '"VINDHYATEL"',               # hardcoded "anticipated watchlist"
        "2026-08-12",                                  # the leftover one-date special case
        "171.4%", "8,14,290", "1,564",                # hardcoded KPI banner numbers
    ]
    for lit in fake_literals:
        assert lit not in code_only, f"found leftover hardcoded literal {lit!r} in pdf_report.py's executable code"


def test_pdf_report_ledger_path_is_actually_used():
    src = Path("pdf_report.py").read_text()
    # The original bug: build_pdf_report(ledger_path=...) accepted the
    # parameter but never referenced it again in the function body.
    assert src.count("ledger_path") >= 3, (
        "ledger_path parameter is not referenced in the function body - "
        "this was exactly the original bug (dead parameter, all sections "
        "hardcoded instead of computed from the real ledger)"
    )
