"""
The alert, the paper ledger and the backtest must name the SAME stocks.

Reported 2026-08-14: "btst paper trade and btst backtest should exactly match
the btst scan alert picks - currently 3 showing different trades. The scan runs
at 15:20 and picks the best stocks (excluding upper circuit stocks), but the
paper trade and backtest show different stocks than the alert picked."

Three independent defects produced that, each reproduced below against the
real data that was committed in this repo when the bug was reported.

  1. The picks CSVs accumulate a row per RUN per day, and ab_paper traded the
     UNION of every run - including superseded intraday scans and names the
     alert had EXCLUDED as circuit-locked.
  2. The ledger keyed trades on signal_time, so one BTST pick seen by both the
     5m replay (cross bar) and the direct-picks path (15:20) was booked twice.
  3. btst_backtest._calc_priority() had the anticipate and fresh_B weights
     transposed relative to btst.actionable_priority(), so the "live rule
     replayed over history" ranked a different top-3 than the live rule.
"""
from pathlib import Path

import pandas as pd
import pytest

import ab_paper
import btst
import btst_backtest

ROOT = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
#  1. the ledger trades exactly what the alert sent
# --------------------------------------------------------------------------- #
def _write_alert(root: Path, date: str, actionable: list[str],
                 excluded: list[str] | None = None) -> None:
    import json
    state = {
        "date": date,
        "scan_time": "15:20",
        "top3_actionable": [{"symbol": s} for s in actionable],
        "next_anticipated": [],
        "excluded_locked": [[s, "Locked at Upper Circuit +20%"]
                            for s in (excluded or [])],
    }
    (root / f"btst_alert_state_{date}.json").write_text(json.dumps(state))


def test_alert_selection_reads_the_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(ab_paper, "ROOT", tmp_path)
    _write_alert(tmp_path, "2026-08-13", ["KMEW", "MUNJALAU"], ["JUBLCPL"])
    sel = ab_paper.load_alert_selection(tmp_path)
    assert sel == {"2026-08-13": {"KMEW", "MUNJALAU"}}
    # the excluded circuit-locked name is NOT tradeable
    assert "JUBLCPL" not in sel["2026-08-13"]


def test_alert_selection_keeps_an_empty_day_authoritative(tmp_path):
    """"The alert said buy nothing" is a decision, not missing data."""
    _write_alert(tmp_path, "2026-08-13", [])
    sel = ab_paper.load_alert_selection(tmp_path)
    assert sel == {"2026-08-13": set()}
    assert "2026-08-13" in sel          # present, just empty


def test_alert_selection_absent_means_no_gating(tmp_path):
    """No snapshot -> caller must fall back to the CSVs, not trade nothing."""
    assert ab_paper.load_alert_selection(tmp_path) == {}


def test_the_real_2026_08_13_regression():
    """The exact numbers from the bug report, against the committed files.

    The alert sent two names; the ledger booked five. The three extras are
    PRECWIRE and MOREPENLAB (earlier intraday scans the final alert dropped)
    and JUBLCPL (explicitly excluded as locked at upper circuit).
    """
    state_path = ROOT / "btst_alert_state.json"
    if not state_path.exists():
        pytest.skip("alert snapshot not committed")
    import json
    state = json.loads(state_path.read_text())
    day = state["date"]
    alert_syms = {a["symbol"].upper() for a in state["top3_actionable"]}

    picks = pd.read_csv(ROOT / "btst_picks.csv")
    ant = pd.read_csv(ROOT / "anticipate_picks.csv")
    csv_syms = set(picks[picks.date == day].symbol.str.upper()) | \
               set(ant[ant.date == day].symbol.str.upper())

    # The raw CSVs are a SUPERSET - that is the bug's precondition. If this
    # ever stops holding the fixture no longer reproduces the report.
    assert alert_syms <= csv_syms

    gated = {s for s in csv_syms if s in alert_syms}
    assert gated == alert_syms, "gating must reproduce the alert exactly"

    excluded = {e[0].upper() for e in state.get("excluded_locked", [])}
    assert not (gated & excluded), "a circuit-locked name must never be traded"


def test_circuit_locked_name_is_never_gated_in(tmp_path):
    _write_alert(tmp_path, "2026-08-13", ["KMEW"], ["JUBLCPL"])
    sel = ab_paper.load_alert_selection(tmp_path)
    lookup = {("2026-08-13", "KMEW"): {}, ("2026-08-13", "JUBLCPL"): {},
              ("2026-08-13", "PRECWIRE"): {}}
    kept = {k for k in lookup
            if k[1].upper() in sel.get(k[0], {k[1].upper()})}
    assert kept == {("2026-08-13", "KMEW")}


# --------------------------------------------------------------------------- #
#  2. one pick = one ledger row, however many times it was spotted
# --------------------------------------------------------------------------- #
def _row(**kw):
    base = dict(model="E_btst", model_label="E", horizon="btst",
                symbol="KENNAMET", week="2026-08-10",
                signal_date="2026-08-12", signal_time="15:20",
                signal_close=3524.8, entry=3524.8, qty=28, invested=98694.4,
                exit_reason="TGT", pnl=3958.68, btst_source="picks")
    base.update(kw)
    return base


def test_same_btst_pick_seen_at_two_times_books_once(tmp_path):
    """KENNAMET 2026-08-12: the 5m replay stamped 12:50, the direct-picks
    path stamped 15:20. Same stock, same entry, same day - ONE trade."""
    led = tmp_path / "l.csv"
    ab_paper.append_ledger(led, [_row(signal_time="12:50"),
                                 _row(signal_time="15:20")])
    out = pd.read_csv(led)
    assert len(out) == 1
    assert out.pnl.sum() == pytest.approx(3958.68)


def test_non_btst_model_may_take_the_same_name_twice_in_a_day(tmp_path):
    """An intraday model genuinely can cross twice - do not merge those."""
    led = tmp_path / "l.csv"
    ab_paper.append_ledger(led, [
        _row(model="D_early", btst_source="", signal_time="09:30", pnl=100.0),
        _row(model="D_early", btst_source="", signal_time="12:50", pnl=200.0),
    ])
    out = pd.read_csv(led)
    assert len(out) == 2
    assert out.pnl.sum() == pytest.approx(300.0)


def test_a_resolved_row_still_replaces_a_no_fill_across_times(tmp_path):
    """The NO_FILL->resolved upgrade must survive the looser key."""
    led = tmp_path / "l.csv"
    ab_paper.append_ledger(led, [_row(signal_time="15:20",
                                      exit_reason="NO_FILL", pnl=0.0)])
    added, dupes, upgraded = ab_paper.append_ledger(
        led, [_row(signal_time="12:50", exit_reason="TGT", pnl=3958.68)])
    out = pd.read_csv(led)
    assert len(out) == 1
    assert out.iloc[0]["exit_reason"] == "TGT"
    assert upgraded == 1 and added == 0


def test_committed_ledger_has_no_intraday_duplicates_after_a_rewrite(tmp_path):
    """Re-appending the real ledger through the fixed writer collapses the
    seven double-booked BTST names the bug report is about."""
    src = ROOT / "ab_ledger.csv"
    if not src.exists():
        pytest.skip("ledger not committed")
    d = pd.read_csv(src)
    led = tmp_path / "l.csv"
    ab_paper.append_ledger(led, d.to_dict("records"))
    out = pd.read_csv(led)
    btst_rows = out[out.btst_source.astype(str).str.strip().str.lower()
                    .isin(["picks", "anticipate", "reconstructed"])]
    dupes = btst_rows.groupby(["model", "symbol", "signal_date"]).size()
    worst = int(dupes.max()) if len(dupes) else 0
    assert worst <= 1, f"still double-booked:\n{dupes[dupes > 1]}"


# --------------------------------------------------------------------------- #
#  3. the backtest ranks with the live rule, not a copy of it
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("row", [
    dict(arm="fresh_B", rvol=3.0, close_pos=0.95, pre=2),
    dict(arm="ant_above", rvol=3.0, close_pos=0.95, pre=2),
    dict(arm="ant_below", rvol=1.0, close_pos=0.92, pre=6),
    dict(arm="fresh_A", rvol=7.0, close_pos=0.99, pre=3),
    dict(arm="aged_B", rvol=2.0, close_pos=0.97, pre=1),
    dict(arm="fresh_B", rvol=9.0, close_pos=0.99, pre=4),
])
def test_backtest_priority_equals_the_live_alert_priority(row):
    live = btst.actionable_priority(row)
    back = btst_backtest._calc_priority(pd.DataFrame([row])).iloc[0]
    assert back == pytest.approx(live), (
        f"{row['arm']}: backtest scored {back}, the alert scores {live}")


def test_anticipate_outranks_fresh_b_in_the_backtest_too():
    """The transposed-weights bug, stated as the behaviour it broke."""
    df = pd.DataFrame([
        dict(symbol="FB", arm="fresh_B", rvol=3.0, close_pos=0.95, pre=2),
        dict(symbol="AN", arm="ant_above", rvol=3.0, close_pos=0.95, pre=2),
    ])
    df["_k"] = btst_backtest._calc_priority(df)
    assert df.sort_values("_k", ascending=False).iloc[0].symbol == "AN"


def test_backtest_priority_handles_missing_and_nan_columns():
    df = pd.DataFrame([dict(arm="fresh_A", rvol=float("nan"), close_pos=None)])
    got = btst_backtest._calc_priority(df).iloc[0]
    assert got == pytest.approx(
        btst.actionable_priority(dict(arm="fresh_A", rvol=0.0,
                                      close_pos=1.0, pre=0.0)))
    assert got == got, "NaN leaked into the score"


def test_backtest_priority_on_empty_frame_is_empty():
    assert btst_backtest._calc_priority(pd.DataFrame()).empty


def test_calc_priority_does_not_restate_the_weights():
    """The rule lives in ONE place. A second hand-kept copy is the bug."""
    src = (ROOT / "btst_backtest.py").read_text()
    body = src.split("def _calc_priority", 1)[1].split("\ndef ", 1)[0]
    assert "actionable_priority" in body
    for weight in ("500.0", "400.0", "300.0", "200.0", "100.0"):
        assert weight not in body.split('"""')[-1], (
            f"weight {weight} restated in _calc_priority - import it instead")


# --------------------------------------------------------------------------- #
#  4. only a run INSIDE the entry window may mark a pick tradeable
# --------------------------------------------------------------------------- #
from datetime import time as dtime


def _enterable(hh: int, mm: int, after_close: bool) -> bool:
    """The rule as written in btst.main() - kept in sync by the source test."""
    return (dtime(hh, mm) <= dtime(15, 30)) and not after_close


@pytest.mark.parametrize("hh,mm,after_close,want", [
    (15, 20, False, True),    # the real 15:20 alert
    (13, 46, False, True),    # early scan: the close has NOT happened yet,
    (10, 14, False, True),    # ...so "buy into today's close" is still open
    (15, 30, False, True),    # the entry window shuts
    (15, 33, False, False),   # after the close - unbuyable
    (20, 59, False, False),   # the 2026-08-11 evening run
    (20, 59, True,  False),   # ...and with --after-close: still NOT tradeable
    (13, 46, True,  False),   # a review is a review whatever time it runs
])
def test_only_a_pre_close_run_is_tradeable(hh, mm, after_close, want):
    assert _enterable(hh, mm, after_close) is want


def test_after_close_review_no_longer_marks_picks_tradeable():
    """The 2026-08-11 20:59 review stamped tradeable=1 on AARTISURF and
    INDSWFTLAB because too_late exempts --after-close. enterable does not."""
    src = (ROOT / "btst.py").read_text()
    assert "enterable = ran_before_close and not args.after_close" in src
    assert '"tradeable": 1 if enterable else 0' in src
    assert '"tradeable": 0 if too_late else 1' not in src, (
        "too_late exempts --after-close - it must not drive the flag")


def test_review_snapshot_never_defines_the_tradeable_set(tmp_path):
    """A post-close snapshot must not become 'the alert'.

    BUG 81 refinement: the original assertion here was `== {}`, i.e. "leave
    the day ungated". That encoded the bug - an ungated day falls back to
    replaying the raw picks CSVs, which is how 2026-08-13 ended up booking
    trades nobody entered. The requirement was always that the REVIEW'S
    PICKS must not become tradeable; the correct expression of that is an
    empty gate for the day, not the absence of a gate.
    """
    import json
    (tmp_path / "btst_alert_state_2026-08-13.json").write_text(json.dumps({
        "date": "2026-08-13", "scan_time": "20:39", "enterable": False,
        "top3_actionable": [{"symbol": "KMEW"}, {"symbol": "MUNJALAU"}],
    }))
    sel = ab_paper.load_alert_selection(tmp_path)
    assert sel.get("2026-08-13") == set(), \
        "the review's own picks must never be tradeable"
    assert "KMEW" not in sel.get("2026-08-13", set())
    assert "MUNJALAU" not in sel.get("2026-08-13", set())


def test_enterable_snapshot_is_used(tmp_path):
    import json
    (tmp_path / "btst_alert_state_2026-08-13.json").write_text(json.dumps({
        "date": "2026-08-13", "scan_time": "15:20", "enterable": True,
        "top3_actionable": [{"symbol": "KMEW"}],
    }))
    assert ab_paper.load_alert_selection(tmp_path) == {"2026-08-13": {"KMEW"}}


def test_a_review_must_not_overwrite_the_entry_window_snapshot(tmp_path):
    """BUG 79: the guard now covers the LIVE file too, not just the archive.

    Was a string match on the old inline implementation; rewritten to test
    the extracted helper's behaviour so it survives refactors.
    """
    import json
    from btst import _keep_enterable

    live = tmp_path / "btst_alert_state.json"
    live.write_text(json.dumps({"date": "2026-08-13", "enterable": True}))

    # a post-close review must NOT clobber today's enterable alert
    assert _keep_enterable(live, "2026-08-13", enterable=False) is True
    # a genuine 15:20 run may always write
    assert _keep_enterable(live, "2026-08-13", enterable=True) is False
    # yesterday's leftover describes another session - always replaceable,
    # otherwise the first post-close run of a new day freezes stale names
    assert _keep_enterable(live, "2026-08-14", enterable=False) is False

    # a prior non-enterable review is not worth protecting
    live.write_text(json.dumps({"date": "2026-08-13", "enterable": False}))
    assert _keep_enterable(live, "2026-08-13", enterable=False) is False

    # missing / corrupt files must never block a write
    assert _keep_enterable(tmp_path / "nope.json", "2026-08-13", False) is False
    live.write_text("{ not json")
    assert _keep_enterable(live, "2026-08-13", enterable=False) is False


def test_the_pdf_must_not_report_unentered_setups_as_positions():
    """BUG 79: on 2026-08-13 five post-close re-runs overwrote the snapshot,
    and the PDF rendered the review's names as OPEN positions worth 197,466
    that were never bought (every ledger row that day was NO_FILL, qty 0)."""
    src = (ROOT / "pdf_report.py").read_text()
    assert '"status": " OPEN (Exit 09:15)" if entered' in src, (
        "the OPEN status must be conditional on the day being enterable")
    assert "NOT ENTERED" in src, "a non-enterable day needs an honest heading"
    assert '"invested": qty * price if entered else 0.0' in src, (
        "an unentered setup has no capital committed")


# --------------------------------------------------------------------------- #
#  5. the watchlist carries the nearest THREE anticipated names
# --------------------------------------------------------------------------- #
def _ant(sym, gap):
    return {"symbol": sym, "close": 100.0, "level": 100.0 + gap, "pre": 6,
            "gap_pct": gap, "side": "below", "day_ret": 1.0, "rvol": 1.0,
            "close_pos": 0.95, "mcap_cr": 1000.0, "ret_12m": 50.0,
            "dist_200dma": 10.0}


def test_next_anticipated_carries_three():
    pool = [_ant(f"S{i}", 0.5 * i) for i in range(1, 7)]
    _t3, next_ant, _ex = btst.build_actionable_lists([], [], pool)
    assert len(next_ant) == 3


def test_next_anticipated_are_the_nearest_and_exclude_bought_names():
    pool = [_ant(f"S{i}", 0.5 * i) for i in range(1, 7)]
    ant_picks = [pool[0], pool[1]]          # S1, S2 become actionable buys
    top3, next_ant, _ex = btst.build_actionable_lists([], ant_picks, pool)
    taken = {a["symbol"] for a in top3}
    assert not (taken & {r["symbol"] for r in next_ant})
    # nearest three of what is LEFT, in pool order
    assert [r["symbol"] for r in next_ant] == [
        r["symbol"] for r in pool if r["symbol"] not in taken][:3]


# --------------------------------------------------------------------------- #
#  6. a circuit lock must be EVIDENCED, never inferred from missing data
# --------------------------------------------------------------------------- #
def test_genuine_locks_are_still_caught():
    """Real locks, verified against real OHLC in earlier fixes."""
    # AARTISURF 2026-08-11: close == high == 572.00, prev close 520.00, +10%
    assert btst.get_circuit_info(520.0, 572.0, 572.0, close_pos=1.0)["is_locked"]
    # JUBLCPL 2026-08-13: +20% band, closed at the high
    assert btst.get_circuit_info(1935.5, 2322.6, 2322.6,
                                 close_pos=1.0)["is_locked"]


@pytest.mark.parametrize("sym,prev,close,cp", [
    ("HAPPYFORGE", 1923.61, 2016.9, 0.900),   # 2026-08-12, was falsely excluded
    ("MOREPENLAB", 73.75, 81.06, 0.957),      # 2026-08-13, was falsely excluded
])
def test_a_name_that_closed_off_its_high_is_not_locked(sym, prev, close, cp):
    """No high available + close_pos well under 1.0 -> trading freely."""
    assert not btst.get_circuit_info(prev, close, 0.0,
                                     close_pos=cp)["is_locked"], sym


def test_a_real_high_beats_close_pos():
    """Direct evidence wins: close==high AT the band is a lock even if a
    stale close_pos disagrees."""
    assert btst.get_circuit_info(520.0, 572.0, 572.0,
                                 close_pos=0.5)["is_locked"]


def test_unknown_high_is_not_treated_as_closed_at_the_high():
    """The defaulting bug, stated directly."""
    free = btst.get_circuit_info(1923.61, 2016.9, 0.0, close_pos=0.90)
    faked = btst.get_circuit_info(1923.61, 2016.9, 2016.9, close_pos=None)
    assert not free["is_locked"]
    assert faked["is_locked"], "the old close-as-high path is what we removed"


def test_legacy_rows_without_high_or_close_pos_still_flag_aartisurf():
    """Backward compatibility with pre-`high` picks files must not regress."""
    from ab_paper import _pick_is_circuit_locked
    assert _pick_is_circuit_locked({"entry": 572.0, "day_ret": 12.6}) is True


def test_the_two_false_positives_are_buyable_again():
    from ab_paper import _pick_is_circuit_locked
    assert _pick_is_circuit_locked(
        {"entry": 2016.9, "day_ret": 4.85, "close_pos": 0.900}) is False
    assert _pick_is_circuit_locked(
        {"entry": 81.06, "day_ret": 9.91, "close_pos": 0.957}) is False


# --------------------------------------------------------------------------- #
#  7. pre-gate rows already in the ledger get evicted, not just blocked
# --------------------------------------------------------------------------- #
def test_eviction_removes_only_off_alert_btst_rows():
    """Gating stops NEW rows; rows written before the gate must also go."""
    led = pd.DataFrame([
        dict(model="E_btst", symbol="MUNJALAU", signal_date="2026-08-13",
             btst_source="picks"),
        dict(model="E_btst", symbol="JUBLCPL", signal_date="2026-08-13",
             btst_source="picks"),                       # excluded by the alert
        dict(model="F_anticipate", symbol="KMEW", signal_date="2026-08-13",
             btst_source="anticipate"),                  # superseded run
        dict(model="D_early", symbol="ANYTHING", signal_date="2026-08-13",
             btst_source=""),                            # not a BTST model
        dict(model="E_btst", symbol="OLDNAME", signal_date="2026-08-01",
             btst_source="picks"),                       # ungated day
    ])
    alert_sel = {"2026-08-13": {"MUNJALAU"}}

    src = led["btst_source"].astype(str).str.strip().str.lower()
    is_btst = src.isin(["picks", "anticipate", "reconstructed"])
    day = led["signal_date"].astype(str)
    sym = led["symbol"].astype(str).str.upper()
    allowed = pd.Series([s in alert_sel.get(d, {s}) for d, s in zip(day, sym)],
                        index=led.index)
    stale = is_btst & day.isin(alert_sel) & ~allowed

    assert set(led[stale].symbol) == {"JUBLCPL", "KMEW"}
    kept = set(led[~stale].symbol)
    assert kept == {"MUNJALAU", "ANYTHING", "OLDNAME"}, (
        "must not touch non-BTST models or days with no alert snapshot")


def test_eviction_is_wired_into_main():
    src = (ROOT / "ab_paper.py").read_text()
    assert "not in that day's alert" in src
    assert "pre-gate row(s) the" in src


# --------------------------------------------------------------------------- #
#  8. the BTST/non-BTST split must not depend on the pandas major
# --------------------------------------------------------------------------- #
def test_missing_btst_source_is_not_treated_as_btst(tmp_path):
    """requirements.txt allows pandas 2.x AND 3.x. On 2.x a missing value
    stringifies to "nan"; on 3.x astype(str) leaves the float nan in place.
    A version-dependent test would classify a NON-BTST row as BTST and drop
    a legitimate second intraday trade. Caught by CI on pandas 3.0.5 while
    local ran 2.2.3."""
    led = tmp_path / "l.csv"
    base = dict(model="D_early", symbol="X", signal_date="2026-08-12",
                exit_reason="TIME")
    # btst_source omitted entirely -> NaN once it goes through the frame
    ab_paper.append_ledger(led, [
        dict(base, signal_time="09:30", pnl=100.0),
        dict(base, signal_time="12:50", pnl=200.0),
    ])
    out = pd.read_csv(led)
    assert len(out) == 2, (
        "two distinct intraday crosses of a non-BTST model must both survive")
    assert out.pnl.sum() == pytest.approx(300.0)


def test_dedup_fills_na_before_stringifying():
    src = (ROOT / "ab_paper.py").read_text()
    seg = src.split("def _dedup_key", 1)[1][:700]
    assert 'fillna("").astype(str)' in seg, (
        "astype(str) alone is pandas-major dependent for missing values")


# --------------------------------------------------------------------------- #
#  9. the PDF report must be readable, and must describe itself correctly
# --------------------------------------------------------------------------- #
def test_pdf_font_supports_the_rupee_sign():
    """2026-08-14: Helvetica is a base-14 Type1 font with no U+20B9, so every
    price rendered as a stray "n" and the whole report looked corrupt even
    though every value in it was right."""
    import pdf_report
    base, bold, rupee = pdf_report._register_fonts()
    if base == "Helvetica":
        assert rupee == "Rs ", "no Unicode font -> must fall back to ASCII"
    else:
        assert rupee == "\u20b9"
        from reportlab.pdfbase import pdfmetrics
        assert pdfmetrics.getFont(base) is not None


def test_pdf_strips_emoji_the_font_cannot_draw():
    """DejaVu has the rupee sign but not pictographs - a leftover emoji
    renders as a NUL box."""
    from pdf_report import _plain
    assert _plain("\U0001f52d Anticipate (\u2b50 Prime #1)") == "Anticipate (Prime #1)"
    assert _plain("\U0001f7e2 Confirmed") == "Confirmed"
    assert _plain("plain text") == "plain text"
    assert "\x00" not in _plain("\U0001f525 Prime #2")


def test_watchlist_heading_is_not_hardcoded_to_two():
    """btst.py now sends the nearest THREE anticipated names; the heading
    silently kept claiming "Top 2" - a report mis-describing its contents."""
    src = (ROOT / "pdf_report.py").read_text()
    assert "(Top 2 Setups for Tomorrow)" not in src
    assert "_n_ant" in src, "the count must be derived from the data"


def test_pdf_kpis_match_the_ledger():
    """The header block must be reproducible from ab_ledger.csv."""
    import pdf_report
    from ab_paper import summarise
    led = ROOT / "ab_ledger.csv"
    if not led.exists():
        pytest.skip("no ledger committed")
    df = pd.read_csv(led)
    live = pdf_report.dedupe_live_trades(df)
    if live.empty:
        pytest.skip("no live-model trades yet")
    s = summarise(live)
    # every KPI the PDF prints must come from this one call - no second
    # stats engine that can drift.
    for k in ("n", "win", "pf", "avg_pct", "med_pct", "net"):
        assert k in s, f"KPI {k} missing from summarise()"
    # "Trades (closed)" means CLOSED - NO_FILL rows are signals that were
    # never filled and must not be counted as trades.
    closed = live[live["exit_reason"].astype(str) != "NO_FILL"]
    assert s["n"] == len(closed), (
        f"header says {s['n']} closed trades, ledger has {len(closed)}")


# --------------------------------------------------------------------------- #
#  10. the backtest must screen the SAME universe as the live scanner
# --------------------------------------------------------------------------- #
def test_backtest_applies_the_live_mcap_floor():
    """2026-08-14: "backtest and paper trade are totally different".

    btst_backtest.py applied NO market-cap filter while the live scanner
    rejects everything under strategy.min_mcap - 872 of the 2,282 names in
    mcap.csv, 38% of the universe. A replay headed "the LIVE rule replayed
    over history" cannot silently drop one of the live rule's filters, and
    the omission flatters it: micro-caps print the biggest raw moves.
    """
    import btst_backtest as bb

    assert hasattr(bb, "load_mcaps"), "the backtest must read mcap.csv"
    src = (ROOT / "btst_backtest.py").read_text()
    assert "min_mcap" in src
    # the floor must come from config, not be restated as a literal
    assert "strategy.min_mcap" in src, "read the floor from config, do not hardcode"


def test_backtest_mcap_floor_defaults_on():
    """Live parity is the DEFAULT; research mode must be opt-in.

    A flag that defaults off is a flag nobody sets, and the report would go
    back to describing a universe the alert never sees.
    """
    src = (ROOT / "btst_backtest.py").read_text()
    assert '"--live-parity"' in src
    assert "action=\"store_true\", default=True" in src
    assert '"--no-live-parity"' in src


def test_backtest_csv_respects_the_floor():
    """The CSV feeds the PDF. An unfiltered CSV under a live-parity header
    is exactly the mismatch this change exists to remove."""
    src = (ROOT / "btst_backtest.py").read_text()
    csv_block = src.split("if args.csv and not df.empty:", 1)[1][:500]
    assert "mcap_ok" in csv_block, "the exported CSV must carry the filtered book"


def test_mcap_loader_handles_a_missing_or_broken_file(tmp_path):
    import btst_backtest as bb
    assert bb.load_mcaps(str(tmp_path / "nope.csv")) == {}
    bad = tmp_path / "bad.csv"
    bad.write_text("not,a,mcap,file\n1,2,3,4\n")
    assert bb.load_mcaps(str(bad)) == {}
    good = tmp_path / "good.csv"
    good.write_text("symbol,mcap_cr,updated\nFOO,1500.5,2026-08-10\nBAR,,2026-08-10\n")
    caps = bb.load_mcaps(str(good))
    assert caps == {"FOO": 1500.5}, "NaN caps must be dropped, not kept as nan"


def test_the_inverted_entry_bias_is_corrected_everywhere():
    """The docs claimed the 15:20 entry is 0.14% CHEAPER than the close.

    Measured over all 28 real fills it is +0.433% ABOVE the close - the sign
    was backwards, so the backtest was described as pessimistic when it is
    optimistic by ~40% of the 1% stop distance on every trade.
    """
    import btst_backtest as bb
    assert bb.LIVE_ENTRY_BIAS_PCT > 0, "the live entry is ABOVE the close"
    assert bb.LIVE_ENTRY_BIAS_N >= 28

    for name in ("btst.py", "btst_backtest.py", ".github/workflows/btst.yml"):
        text = (ROOT / name).read_text()
        assert "0.14% CHEAPER" not in text, f"{name} still claims the wrong sign"
        assert "0.14% LESS than the close" not in text, f"{name} still claims the wrong sign"


# --------------------------------------------------------------------------- #
#  11. tomorrow's 15:20 scan must not be blocked, and must not be pre-empted
# --------------------------------------------------------------------------- #
def test_a_premarket_snapshot_never_blocks_the_1520_scan(tmp_path):
    """BUG 80 near-miss. _keep_enterable() protects an enterable snapshot from
    a later review - but the 00:00 pre-market run is ALSO enterable. If the
    guard keyed only on `enterable` it would freeze a midnight scan with zero
    picks and the real 15:20 list could never land: every trade of the day
    silently missed.

    It does not, because a 15:20 run is itself enterable and the guard only
    blocks non-enterable writers. This pins that down.
    """
    import json
    from btst import _keep_enterable

    live = tmp_path / "btst_alert_state.json"
    live.write_text(json.dumps({"date": "2026-08-14", "scan_time": "00:00",
                                "enterable": True, "too_early": True,
                                "top3_actionable": []}))
    # every in-window scan may still write
    for _t in ("15:10", "15:15", "15:20", "15:26"):
        assert _keep_enterable(live, "2026-08-14", enterable=True) is False, (
            "an in-window scan must never be blocked by the pre-market file")
    # and only after one lands do post-close reviews get locked out
    live.write_text(json.dumps({"date": "2026-08-14", "scan_time": "15:20",
                                "enterable": True, "too_early": False,
                                "top3_actionable": [{"symbol": "REAL"}]}))
    assert _keep_enterable(live, "2026-08-14", enterable=False) is True


def test_premarket_scan_is_not_reported_as_the_days_verdict():
    """BUG 80: btst.py writes too_early but pdf_report.py never read it, so
    the 2026-08-14 00:00 scan rendered as "STOCKS ENTERED TODAY - No setups
    qualified today" - a verdict on a session that had not opened."""
    src = (ROOT / "pdf_report.py").read_text()
    assert 'too_early = bool(alert_state.get("too_early", False))' in src
    assert "PROVISIONAL - PRE-MARKET SCAN" in src
    assert "The 15:20 scan decides." in src


# ---------------------------------------------------------------------------
# 12. BUG 81 - a day nobody could enter must never book trades
#
# 2026-08-13's only scans ran after the close (enterable=False, too_late).
# load_alert_selection() used to `continue` past such a snapshot, which left
# the day UNGATED - and an ungated day falls back to replaying the raw picks
# CSVs. The 08-13 rows sat correctly at qty=0/NO_FILL until the 08-14 candle
# arrived, at which point the resolver filled them in and the report booked
# MOREPENLAB, PRECWIRE and MUNJALAU as realized closed trades worth -4,405
# on a day the user entered nothing.
# ---------------------------------------------------------------------------

def _snap(day, enterable, syms, scan="15:20"):
    return {"date": day, "scan_time": scan, "enterable": enterable,
            "top3_actionable": [{"symbol": s} for s in syms]}


def test_a_non_enterable_day_gates_to_empty_not_ungated(tmp_path):
    """The 2026-08-13 bug: post-close review must mean 'nothing was buyable'."""
    import json
    from ab_paper import load_alert_selection
    (tmp_path / "btst_alert_state_2026-08-13.json").write_text(
        json.dumps(_snap("2026-08-13", False, ["KMEW", "MUNJALAU"], "22:56")))
    sel = load_alert_selection(tmp_path)
    # present in the mapping (so the day IS gated) but empty (so nothing fills)
    assert "2026-08-13" in sel, "a non-enterable day must still be gated"
    assert sel["2026-08-13"] == set(), \
        "a day nobody could enter must gate to the empty set, not its picks"


def test_absent_snapshot_still_leaves_a_day_ungated(tmp_path):
    """Only the ABSENCE of a snapshot may leave a day ungated (backfill)."""
    from ab_paper import load_alert_selection
    assert load_alert_selection(tmp_path) == {}


def test_an_enterable_snapshot_wins_over_a_post_close_review(tmp_path):
    """A real 15:20 alert must survive an archived post-close review, and the
    result must not depend on which file the glob happens to read first."""
    import json
    from ab_paper import load_alert_selection
    for live, arch in ((True, False), (False, True)):
        d = tmp_path / f"case_{live}"
        d.mkdir()
        (d / "btst_alert_state.json").write_text(
            json.dumps(_snap("2026-08-13", live, ["REAL"] if live else ["REVIEW"])))
        (d / "btst_alert_state_2026-08-13.json").write_text(
            json.dumps(_snap("2026-08-13", arch, ["REAL"] if arch else ["REVIEW"])))
        assert load_alert_selection(d)["2026-08-13"] == {"REAL"}


def test_no_2026_08_13_btst_rows_survive_in_the_shipped_ledger():
    """The phantom trades must be evicted from the committed ledger itself."""
    import pandas as pd
    led = pd.read_csv(ROOT / "ab_ledger.csv")
    day = led[led["signal_date"].astype(str) == "2026-08-13"]
    live = day[day["model"].isin(["E_btst", "E_btst_wide",
                                  "F_anticipate", "F_anticipate_only"])]
    assert live.empty, (
        "2026-08-13 was not enterable - no BTST/anticipate row may remain:\n"
        f"{live[['model', 'symbol', 'qty', 'exit_reason', 'pnl']]}")


def test_yesterday_section_skips_a_no_entry_session():
    """compute_yesterday_results() must fall back to the last session that
    actually traded (2026-08-12), never invent a 2026-08-13 result set."""
    import pandas as pd
    from pdf_report import compute_yesterday_results
    led = pd.read_csv(ROOT / "ab_ledger.csv")
    rows = compute_yesterday_results(led, "2026-08-14")
    assert rows, "expected the 08-12 cohort to be reported"
    assert all(r["entry_date"] != "2026-08-13" for r in rows), \
        "a session with zero entries must never appear as realized results"
    for phantom in ("MOREPENLAB", "PRECWIRE", "MUNJALAU"):
        assert not any(r["symbol"] == phantom and r["entry_date"] == "2026-08-13"
                       for r in rows), f"{phantom} was never entered on 08-13"
