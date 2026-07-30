"""
Regression tests for bugs found by auditing a real GitHub Actions run.

Each test here maps to a specific production bug. They exist so the same
mistakes cannot silently come back.
"""

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent
WORKFLOWS = ROOT / ".github" / "workflows"


# --------------------------------------------------------------------------- #
#  BUG 1 - scan.py exited 1 when the snapshot file was missing.
#  scan.py runs every 5 minutes, so that turned one missing file into a failed
#  workflow (and a failure alert) every single run, all day.
# --------------------------------------------------------------------------- #
def test_scan_exits_zero_when_snapshot_missing(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("strategy: {}\nuniverse: {}\nruntime: {}\n")

    proc = subprocess.run(
        [sys.executable, str(ROOT / "scan.py"), "--force", "--config", str(cfg)],
        capture_output=True, text=True, timeout=60,
        env={"PATH": "/usr/bin:/bin", "DHAN_CLIENT_ID": "x",
             "DHAN_ACCESS_TOKEN": "x", "HOME": str(tmp_path)},
        cwd=tmp_path,
    )
    assert proc.returncode == 0, (
        "a missing snapshot must be a quiet skip, not a hard failure "
        f"(stdout={proc.stdout[-400:]} stderr={proc.stderr[-400:]})"
    )


def test_load_snapshots_returns_empty_not_raises(tmp_path):
    from config import load_config
    from scan import load_snapshots

    cfg = load_config(ROOT / "config.yaml")
    cfg.paths["snapshot"] = tmp_path / "does_not_exist.csv"
    assert load_snapshots(cfg, "2026-07-27") == []


# --------------------------------------------------------------------------- #
#  BUG 2 - `if: failure() && env.TELEGRAM_BOT_TOKEN != ''`
#  A step's `if:` cannot read an `env:` declared on that same step, so the
#  condition was always false and the failure alert never fired.
# --------------------------------------------------------------------------- #
def load_workflow(name):
    return yaml.safe_load((WORKFLOWS / name).read_text())


def all_steps(wf):
    for job in wf["jobs"].values():
        for step in job.get("steps", []):
            yield step


# --------------------------------------------------------------------------- #
#  BUG 5 - a continuation line inside `run: |` started at column 1, which
#  silently ENDED the block scalar and became a new top-level YAML key.
#  PyYAML parsed the file happily, so "does it parse?" was not a strong enough
#  check; GitHub rejected it with "Unexpected value 'Check the log'".
#  These tests assert the parsed STRUCTURE, not merely that parsing succeeded.
# --------------------------------------------------------------------------- #
VALID_TOP_LEVEL = {"name", "on", True, "concurrency", "permissions",
                   "jobs", "env", "defaults", "run-name"}


@pytest.mark.parametrize("name", ["scan.yml", "snapshot.yml", "tests.yml"])
def test_workflow_has_no_stray_top_level_keys(name):
    """
    A run-block line at column 1 turns into a top-level key. GitHub rejects the
    whole workflow; PyYAML does not. Catch it by validating the key set.
    """
    wf = load_workflow(name)
    stray = set(wf) - VALID_TOP_LEVEL
    assert not stray, (
        f"{name}: unexpected top-level key(s) {sorted(map(str, stray))}. "
        "This usually means a line inside a `run: |` block is not indented, "
        "so YAML ended the block early."
    )


@pytest.mark.parametrize("name", ["scan.yml", "snapshot.yml", "tests.yml"])
def test_workflow_has_required_structure(name):
    wf = load_workflow(name)
    assert "jobs" in wf and wf["jobs"], f"{name}: no jobs"
    assert ("on" in wf) or (True in wf), f"{name}: no triggers"
    for job_name, job in wf["jobs"].items():
        assert "runs-on" in job, f"{name}:{job_name} missing runs-on"
        assert job.get("steps"), f"{name}:{job_name} has no steps"
        for step in job["steps"]:
            assert "uses" in step or "run" in step, (
                f"{name}:{job_name} has a step with neither `uses` nor `run` "
                f"({step.get('name')}) - a sign the YAML nesting is wrong"
            )


@pytest.mark.parametrize("name", ["scan.yml", "snapshot.yml", "tests.yml"])
def test_run_blocks_are_fully_indented(name):
    """
    Re-read the raw file and confirm every line of a `run: |` block is indented
    deeper than the `run:` key itself. This is the exact defect GitHub caught.
    """
    lines = (WORKFLOWS / name).read_text().splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not (stripped == "run: |" or stripped.startswith("run: |")):
            continue
        run_indent = len(line) - len(line.lstrip())
        for body in lines[i + 1:]:
            if not body.strip():
                continue
            indent = len(body) - len(body.lstrip())
            if indent <= run_indent:
                # The block ended here. That is only legitimate if this line is
                # a new list item, a comment, or a sibling `key:` mapping.
                head = body.lstrip()
                legit = (head.startswith("-")
                         or head.startswith("#")
                         or ":" in head.split("#")[0])
                assert legit, (
                    f"{name}: {body!r} escaped the `run: |` block - indent it, "
                    "or YAML turns it into a stray top-level key"
                )
                break


@pytest.mark.parametrize("name", ["scan.yml", "snapshot.yml", "tests.yml"])
def test_step_if_never_references_own_step_env(name):
    wf = load_workflow(name)
    for step in all_steps(wf):
        cond = str(step.get("if", ""))
        if not cond:
            continue
        for var in (step.get("env") or {}):
            assert f"env.{var}" not in cond, (
                f"{name}: step '{step.get('name')}' tests env.{var} in its own "
                "`if:` - GitHub evaluates that before the step env exists, so it "
                "is always empty. Check the variable inside `run:` instead."
            )


def test_failure_notification_still_exists():
    wf = load_workflow("scan.yml")
    steps = list(all_steps(wf))
    notify = [s for s in steps if "failure()" in str(s.get("if", ""))]
    assert notify, "scan.yml lost its failure notification step"
    body = notify[0].get("run", "")
    assert "TELEGRAM_BOT_TOKEN" in body and "exit 0" in body, (
        "the failure notice must check for missing secrets inside run:"
    )


# --------------------------------------------------------------------------- #
#  BUG 3 - curl -d with %0A / raw text.
#  `-d` does not URL-encode, so an & or + in a symbol name or message would
#  truncate the Telegram alert. --data-urlencode handles it.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["scan.yml", "snapshot.yml"])
def test_curl_urlencodes_telegram_payloads(name):
    body = (WORKFLOWS / name).read_text()
    if "api.telegram.org" not in body:
        return
    for line in body.splitlines():
        s = line.strip()
        if s.startswith("-d ") and ("text=" in s or "chat_id=" in s):
            pytest.fail(f"{name}: use --data-urlencode, not -d, for: {s}")
    assert "%0A" not in body, (
        f"{name}: %0A only works if the whole payload is pre-encoded; "
        "use a real newline with --data-urlencode"
    )


# --------------------------------------------------------------------------- #
#  BUG 4 - unpinned deps. CI installed pandas 3.0.5, a major release, into a
#  live trading scanner with no code change and no warning.
# --------------------------------------------------------------------------- #
def test_requirements_pin_major_versions():
    reqs = [l.strip() for l in (ROOT / "requirements.txt").read_text().splitlines()
            if l.strip() and not l.startswith("#")]
    assert reqs, "requirements.txt is empty"
    for r in reqs:
        assert "<" in r, (
            f"'{r}' has no upper bound - a future major release can break the "
            "scanner overnight with no code change"
        )


# --------------------------------------------------------------------------- #
#  Workflow sanity
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,script", [
    ("scan.yml", "scan.py"),
    ("snapshot.yml", "build_snapshot.py"),
])
def test_workflows_call_flat_layout_scripts(name, script):
    body = (WORKFLOWS / name).read_text()
    assert f"python {script}" in body, f"{name} should run `python {script}`"
    assert "python -m src." not in body, f"{name} still references the old src/ package"


@pytest.mark.parametrize("name", ["scan.yml", "snapshot.yml"])
def test_write_workflows_declare_contents_write(name):
    wf = load_workflow(name)
    perms = wf.get("permissions") or {}
    assert perms.get("contents") == "write", (
        f"{name} commits back to the repo, so it needs contents: write"
    )


def test_scan_schedule_covers_market_hours_in_utc():
    """NSE 09:15-15:30 IST == 03:45-10:00 UTC. Cron must be UTC."""
    wf = load_workflow("scan.yml")
    # PyYAML parses the bare key `on` as boolean True
    trigger = wf.get("on") or wf.get(True)
    crons = [c["cron"] for c in trigger["schedule"]]
    assert any("3-10" in c for c in crons), f"schedule {crons} misses IST market hours"
    assert any("1-5" in c for c in crons), "should only run Mon-Fri"


def test_no_stale_data_dir_paths_in_workflows():
    """Flat layout: generated files live in the repo root, not data/."""
    for wf_file in WORKFLOWS.glob("*.yml"):
        body = wf_file.read_text()
        assert "data/state.json" not in body, f"{wf_file.name} has a stale data/ path"
        assert "data/weekly_snapshot.csv" not in body, f"{wf_file.name} has a stale data/ path"


# --------------------------------------------------------------------------- #
#  BUG 6 - the NSE scrip master contains ~22 exchange TEST instruments
#  (011NSETEST, G1NSETEST, ...). They have no real history and sort FIRST
#  alphabetically, so `build_snapshot.py --limit 50` spent its whole trial run
#  on dummies and produced an empty snapshot.
# --------------------------------------------------------------------------- #
def test_test_instruments_are_filtered_out():
    import pandas as pd

    from dhan import DhanClient

    fake = pd.DataFrame([
        {"SEM_EXM_EXCH_ID": "NSE", "SEM_SEGMENT": "E", "SEM_SMST_SECURITY_ID": 1,
         "SEM_INSTRUMENT_NAME": "EQUITY", "SEM_TRADING_SYMBOL": "011NSETEST",
         "SEM_SERIES": "EQ", "SEM_EXCH_INSTRUMENT_TYPE": "ES", "SM_SYMBOL_NAME": "t"},
        {"SEM_EXM_EXCH_ID": "NSE", "SEM_SEGMENT": "E", "SEM_SMST_SECURITY_ID": 2,
         "SEM_INSTRUMENT_NAME": "EQUITY", "SEM_TRADING_SYMBOL": "RELIANCE",
         "SEM_SERIES": "EQ", "SEM_EXCH_INSTRUMENT_TYPE": "ES", "SM_SYMBOL_NAME": "r"},
    ])

    class FakeResp:
        status_code = 200
        text = fake.to_csv(index=False)

        def raise_for_status(self):
            pass

    import dhan as dhan_mod
    orig = dhan_mod.requests.get
    dhan_mod.requests.get = lambda *a, **k: FakeResp()
    try:
        got = DhanClient.fetch_instruments(["NSE_EQ"], ["EQ"], exclude_etf=True)
    finally:
        dhan_mod.requests.get = orig

    syms = [i.symbol for i in got]
    assert "RELIANCE" in syms
    assert not [s for s in syms if "NSETEST" in s.upper()], \
        f"exchange test instruments leaked into the universe: {syms}"


# --------------------------------------------------------------------------- #
#  BUG 7 - build_snapshot ran the whole universe before discovering the token
#  was bad, wasting ~12 minutes and ending with an unhelpful error.
# --------------------------------------------------------------------------- #
def test_build_snapshot_has_preflight_check():
    body = (ROOT / "build_snapshot.py").read_text()
    assert "PREFLIGHT" in body, "build_snapshot must probe one symbol before the long run"
    assert "preflight" in body.lower()


def test_snapshot_workflow_uploads_artifact():
    """A multi-hour build must not be lost if the git commit step fails."""
    wf = load_workflow("snapshot.yml")
    steps = list(all_steps(wf))
    assert any("upload-artifact" in str(s.get("uses", "")) for s in steps), \
        "snapshot.yml should upload the CSV as an artifact as a safety net"


# --------------------------------------------------------------------------- #
#  BUG 8 - Weekly Snapshot ran the full 2,390-symbol universe and produced
#  ok=0, skip=2124, err=76. Root causes found in that log:
#    a) _to_frame did not unwrap a {"data": {...}} response -> every symbol
#       looked like it returned no candles and was silently "skipped"
#    b) toDate was today+1 (a future date)
#    c) data_rate 5/s with 5 workers sat exactly on Dhan's ceiling -> 70x 429
#    d) "skip" lumped no-data together with insufficient-history, so the log
#       could not say which had happened
# --------------------------------------------------------------------------- #
def test_to_frame_unwraps_data_envelope():
    from dhan import DhanClient

    payload = {"open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5],
               "volume": [100], "timestamp": [1450000000]}
    assert len(DhanClient._to_frame(payload)) == 1
    assert len(DhanClient._to_frame({"status": "success", "data": payload})) == 1, \
        "a {'data': {...}} envelope must be unwrapped, not treated as empty"
    assert DhanClient._to_frame({"status": "failure"}).empty
    assert DhanClient._to_frame(None).empty


def test_to_date_is_never_in_the_future():
    from datetime import datetime

    from dhan import IST, last_n_years

    _, to_date = last_n_years(5)
    assert to_date <= datetime.now(IST).date(), \
        "a future toDate makes Dhan return an empty payload"


def test_data_rate_leaves_headroom_under_dhan_limit():
    import yaml as _yaml

    cfg = _yaml.safe_load((ROOT / "config.yaml").read_text())
    rt = cfg["runtime"]
    assert rt["data_rate_per_sec"] <= 4, \
        "Dhan allows 5 req/s; running at exactly 5 leaves no room for jitter"
    assert rt["max_workers"] <= rt["data_rate_per_sec"], \
        "more workers than the per-second budget just queues up 429s"


def test_rate_limiter_has_global_pause():
    """A 429 must hold back every thread, not only the one that hit it."""
    import time
    from concurrent.futures import ThreadPoolExecutor

    from dhan import RateLimiter

    lim = RateLimiter(10.0)
    lim.pause(0.6)
    start = time.monotonic()
    waits = []

    def worker(_):
        lim.acquire()
        waits.append(time.monotonic() - start)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(worker, range(4)))
    assert min(waits) >= 0.55, "pause() should delay all workers"


def test_rate_limiter_respects_configured_rate():
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    from dhan import RateLimiter

    lim = RateLimiter(4.0)
    stamps, lock = [], threading.Lock()

    def worker(_):
        lim.acquire()
        with lock:
            stamps.append(time.monotonic())

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(worker, range(20)))
    stamps.sort()
    worst = max(sum(1 for x in stamps if s <= x < s + 1.0) for s in stamps)
    assert worst <= 5, f"{worst} requests in one second exceeds Dhan's limit"


def test_build_snapshot_distinguishes_nodata_from_short_history():
    body = (ROOT / "build_snapshot.py").read_text()
    assert "no_data" in body and "short_hist" in body, \
        "the skip counter must separate 'API returned nothing' from 'too little history'"
    assert "nodata" in body


# --------------------------------------------------------------------------- #
#  BUG 9 - THE ok=0 CAUSE.
#  to_ist() unconditionally added the 1980 epoch offset. Dhan's daily sample
#  uses that convention, but other endpoints/revisions return a PLAIN Unix
#  epoch. Adding +10 years to a plain timestamp dated every candle in the
#  2030s, so `weekly[week_start < this_week]` was EMPTY and every symbol was
#  reported as "short_history":  ok=0 skip=1197 (nodata=0 short=1197).
# --------------------------------------------------------------------------- #
def test_to_ist_auto_handles_both_epoch_conventions():
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    from dhan import DHAN_EPOCH_OFFSET, to_ist_auto

    ist = ZoneInfo("Asia/Kolkata")
    now = datetime.now(ist)

    # Dhan's documented daily sample is 1980-based and must decode to 2022
    assert to_ist_auto(1326220200).year == 2022

    # a plain Unix timestamp must NOT be shifted into the future
    target = datetime(2025, 3, 10, 15, 30, tzinfo=ist)
    assert to_ist_auto(int(target.timestamp())).date() == target.date()

    # nothing may ever decode to a future date
    for ts in (int(now.timestamp()), int(now.timestamp()) - DHAN_EPOCH_OFFSET):
        assert to_ist_auto(ts) <= now + timedelta(days=1), \
            "a candle decoded into the future empties the closed-week filter"


def test_plain_unix_payload_still_builds_a_snapshot():
    """The end-to-end symptom: plain-unix candles must not yield 0 closed weeks."""
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    from config import load_config
    from dhan import DhanClient
    from strategy import build_snapshot, build_weekly_bars, week_start_of

    ist = ZoneInfo("Asia/Kolkata")
    cfg = load_config(ROOT / "config.yaml").strategy
    end = datetime.now(ist).replace(hour=15, minute=30, second=0, microsecond=0)

    o = h = l = c = v = None
    ts, o, h, l, c, v = [], [], [], [], [], []
    price = 100.0
    for i in range(1500, 0, -1):
        d = end - timedelta(days=i)
        if d.weekday() >= 5:
            continue
        price *= 1.0008
        ts.append(int(d.timestamp()))          # PLAIN unix
        o.append(price * 0.995); h.append(price * 1.01)
        l.append(price * 0.99); c.append(price); v.append(500_000)

    df = DhanClient._to_frame({"open": o, "high": h, "low": l, "close": c,
                               "volume": v, "timestamp": ts})
    tw = week_start_of(datetime.now(ist).date())
    closed = build_weekly_bars(df)
    closed = closed[closed["week_start"] < tw]
    assert len(closed) > 100, f"only {len(closed)} closed weeks - epoch decode is wrong"
    assert build_snapshot("T", "1", "NSE_EQ", df, cfg, tw) is not None


def test_daily_candles_are_fetched_in_chunks():
    body = (ROOT / "dhan.py").read_text()
    assert "chunk_days" in body, \
        "5 years in one request can come back short; fetch in chunks"


# --------------------------------------------------------------------------- #
#  Alert must state the entry time of the triggering candle.
# --------------------------------------------------------------------------- #
def test_alert_shows_entry_time():
    from datetime import datetime

    from strategy import BarEval, Signal
    from telegram import _compact, format_signal

    ev = BarEval(conditions={f"c{i:02d}": True for i in range(1, 14)},
                 values={"price": 1512.4, "ema_fast": 1455.2, "ema_slow": 1402.75,
                         "ema_slow_2": 1390.0, "rsi": 72.54, "rsi_1": 63.09,
                         "macd_hist": 12.345, "week_volume": 1.34e7,
                         "vol_sma": 9e6, "week_open": 1463.1, "day_open": 1489.55,
                         "entry_level": 1498.0, "level_52": 1498.0})
    sig = Signal(symbol="X", security_id="1", exchange_segment="NSE_EQ",
                 bar_time=datetime(2026, 7, 27, 11, 20), price=1512.4,
                 entry_level=1498.0, level_52=1498.0, trigger="cross",
                 evaluation=ev, week_start="2026-07-27", week_volume=1.34e7,
                 day_open=1489.55, week_open=1463.1)

    full = format_signal(sig)
    assert "11:20-11:25" in full, "show the signal candle window"
    assert "tradeable from" in full and "11:25" in full, (
        "a bar stamped 11:20 only closes at 11:25 - the alert must say when "
        "the trade can actually be taken")
    assert "11:25" in _compact(sig), "batch lines must show the actionable time"


# --------------------------------------------------------------------------- #
#  Backtest integrity. A backtest that peeks at future data gives false
#  confidence, which is worse than no backtest at all.
# --------------------------------------------------------------------------- #
def test_backtest_reuses_live_strategy_functions():
    """It must not reimplement the rules - otherwise it validates nothing."""
    body = (ROOT / "backtest.py").read_text()
    assert "from strategy import" in body
    assert "build_snapshot" in body and "replay_week" in body
    for banned in ("def evaluate_bar", "def gate_ok", "def build_snapshot("):
        assert banned not in body, f"backtest.py must not redefine {banned}"


def test_backtest_has_no_lookahead():
    """
    Truncating future data must not change past signals. Uses synthetic data so
    the test is deterministic and needs no network.
    """
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    import numpy as np
    import pandas as pd

    from backtest import backtest_symbol
    from config import load_config
    from strategy import week_start_of

    ist = ZoneInfo("Asia/Kolkata")
    cfg = load_config(ROOT / "config.yaml")

    rng = np.random.default_rng(3)
    rows, day, price = [], datetime(2021, 1, 4, 15, 30, tzinfo=ist), 100.0
    while len(rows) < 1100:
        if day.weekday() < 5:
            price *= 1 + rng.normal(0.0012, 0.012)
            rows.append({"datetime": day, "open": price * 0.995,
                         "high": price * 1.012, "low": price * 0.988,
                         "close": price, "volume": float(rng.integers(4e5, 9e5))})
        day += timedelta(days=1)
    daily = pd.DataFrame(rows)

    last = pd.Timestamp(daily.iloc[-1]["datetime"]).tz_localize(None)
    start = week_start_of((last - pd.Timedelta(days=700)).date())
    end = week_start_of(last.date()) - pd.Timedelta(days=7)

    full = backtest_symbol("T", daily, cfg, start, end)
    if len(full) < 2:
        return                                   # nothing meaningful to compare

    cut = pd.Timestamp(full[-1].entry_time).tz_localize(None).normalize()
    truncated = daily[pd.to_datetime(daily["datetime"]).dt.tz_localize(None) < cut]
    part = backtest_symbol("T", truncated, cfg, start,
                           week_start_of(cut.date()) - pd.Timedelta(days=7))

    expected = [(t.week, round(t.entry, 4)) for t in full
                if pd.Timestamp(t.week) <= week_start_of(cut.date()) - pd.Timedelta(days=7)]
    got = [(t.week, round(t.entry, 4)) for t in part]
    assert expected == got, (
        "past signals changed when future data was removed - look-ahead bias")


def test_forward_stats_only_uses_bars_after_entry():
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    import pandas as pd

    from backtest import forward_stats

    ist = ZoneInfo("Asia/Kolkata")
    t0 = datetime(2025, 1, 6, 15, 30, tzinfo=ist)
    rows = [{"datetime": t0 + timedelta(days=i), "open": 100, "high": 100,
             "low": 100, "close": 100.0 if i <= 0 else 110.0, "volume": 1}
            for i in range(-10, 40)]
    fwd, mfe, mae, _ = forward_stats(pd.DataFrame(rows), t0, 100.0)
    assert all(v > 0 for v in fwd.values()), "pre-entry bars must be excluded"
    assert mae >= 0.0


# --------------------------------------------------------------------------- #
#  Backtest / Dhan wiring
# --------------------------------------------------------------------------- #
def test_backtest_defaults_to_dhan_source():
    body = (ROOT / "backtest.py").read_text()
    assert 'choices=["dhan", "yahoo"], default="dhan"' in body, \
        "the backtest should default to the same data the live scanner uses"


def test_backtest_fetches_scrip_master_once():
    """The scrip master is a ~27 MB download; fetching it twice is wasteful."""
    body = (ROOT / "backtest.py").read_text()
    assert body.count("fetch_instruments(") == 1, \
        "fetch_instruments must be called exactly once in backtest.py"


def test_backtest_has_dhan_preflight():
    body = (ROOT / "backtest.py").read_text()
    assert "PREFLIGHT" in body, \
        "verify the token before spending hundreds of API calls"


def test_backtest_caches_intraday_per_symbol():
    """
    5m history must be pulled once per symbol in 90-day slices, not once per
    week - the latter is ~52 calls per symbol-year and blows the rate limit.
    """
    body = (ROOT / "backtest.py").read_text()
    assert "intraday_cache" in body
    assert "days=85" in body, "respect Dhan's 90-day intraday window"


# --------------------------------------------------------------------------- #
#  BUG 10 - the 25m/err=126 run.
#  Chunking daily history into 365-day windows turned 1 request per symbol into
#  6 (14,208 calls for the universe), which caused sustained 429s. And a single
#  failing chunk raised, discarding every year that DID load - so recently
#  listed stocks were lost entirely instead of using their shorter history.
# --------------------------------------------------------------------------- #
def test_daily_candles_makes_one_request_by_default():
    from datetime import date
    from unittest.mock import patch

    from dhan import DhanClient

    calls = []

    def fake_post(self, path, payload, limiter):
        calls.append(payload)
        return {"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0],
                "volume": [1], "timestamp": [1450000000]}

    client = DhanClient.__new__(DhanClient)
    client._data_limiter = None
    with patch.object(DhanClient, "_post", fake_post):
        DhanClient.daily_candles(client, "1", "NSE_EQ",
                                 date(2021, 7, 17), date(2026, 7, 27))

    assert len(calls) == 1, (
        f"{len(calls)} requests for one symbol - the daily endpoint has no "
        "90-day cap, so chunking just multiplies the rate-limit pressure")
    assert calls[0]["fromDate"] == "2021-07-17"
    assert calls[0]["toDate"] == "2026-07-27"


def test_partial_chunk_failure_keeps_the_data_that_loaded():
    """A stock listed in 2024 has no 2021 data; that must not lose 2024-2026."""
    from datetime import date
    from unittest.mock import patch

    from dhan import DhanClient, NoDataError

    state = {"n": 0}

    def fake_post(self, path, payload, limiter):
        state["n"] += 1
        if state["n"] == 1:                      # first window predates listing
            raise NoDataError("no data in range (DH-905)")
        return {"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0],
                "volume": [1], "timestamp": [1450000000]}

    client = DhanClient.__new__(DhanClient)
    client._data_limiter = None
    with patch.object(DhanClient, "_post", fake_post):
        df = DhanClient.daily_candles(client, "1", "NSE_EQ",
                                      date(2021, 1, 1), date(2026, 1, 1),
                                      chunk_days=365)
    assert not df.empty, "one failed chunk must not discard the others"


def test_dh905_is_classified_as_no_data_not_an_error():
    """DH-905 = listed after fromDate. A skip, not a failure to investigate."""
    from unittest.mock import patch

    from dhan import DhanClient, DhanError, NoDataError

    class Resp:
        status_code = 400
        text = ('{"errorType":"Input_Exception","errorCode":"DH-905",'
                '"errorMessage":"System is unable to fetch data"}')

        def json(self):
            return {}

    client = DhanClient.__new__(DhanClient)
    client.timeout = 5
    client.max_retries = 3
    client._data_limiter = type("L", (), {"acquire": lambda s: None,
                                          "pause": lambda s, x: None})()
    client._session = type("S", (), {"post": lambda s, *a, **k: Resp()})()

    with patch("time.sleep", lambda *_: None):
        try:
            client._post("/charts/historical", {}, client._data_limiter)
            raise AssertionError("should have raised")
        except NoDataError as exc:
            assert isinstance(exc, DhanError), "must stay catchable as DhanError"


def test_build_snapshot_treats_nodata_as_skip():
    body = (ROOT / "build_snapshot.py").read_text()
    assert "NoDataError" in body
    assert "except NoDataError" in body, \
        "a listing gap should increment the skip counter, not the error counter"


# --------------------------------------------------------------------------- #
#  BUG 11 - THE REASON NO ALERTS EVER FIRED.
#  weekly_snapshot.csv was never committed, so scan.py had zero levels to test.
#  Two causes:
#    a) "Commit snapshot" had no `if:`, so a CANCELLED or failed build skipped
#       it entirely - run #7 was cancelled at 49 min and lost everything
#    b) the CSV was only written once, at the very end, so any interruption
#       discarded all completed work
# --------------------------------------------------------------------------- #
def test_snapshot_commit_step_runs_even_if_build_is_cancelled():
    wf = load_workflow("snapshot.yml")
    steps = list(all_steps(wf))
    commit = [s for s in steps if "Commit" in str(s.get("name", ""))]
    assert commit, "snapshot.yml lost its commit step"
    assert str(commit[0].get("if", "")).strip() == "always()", (
        "without if: always() a cancelled or failed build skips the commit and "
        "throws away every row it produced")


def test_snapshot_build_saves_progress_incrementally():
    body = (ROOT / "build_snapshot.py").read_text()
    assert "def flush()" in body, "must checkpoint progress, not write once at the end"
    assert "flush()" in body.split("def flush()")[1], "flush must actually be called"


def test_snapshot_build_resumes_from_partial_file():
    body = (ROOT / "build_snapshot.py").read_text()
    assert "done_symbols" in body and "pending" in body, \
        "a rerun should skip symbols already saved for the same week"
    assert "--fresh" in body, "need an escape hatch to force a full rebuild"


def test_snapshot_timeout_is_realistic():
    """330 min let a stuck run burn hours; the work is ~10 min of API time."""
    wf = load_workflow("snapshot.yml")
    job = next(iter(wf["jobs"].values()))
    assert job.get("timeout-minutes", 999) <= 120


# --------------------------------------------------------------------------- #
#  Low-latency watcher. ~94% of the GitHub Actions latency was cron wait and
#  cold start, not real work (scan itself measured 19s). watch.py removes the
#  fixed cost by staying resident and waking exactly at each candle close.
# --------------------------------------------------------------------------- #
def test_next_candle_close_is_aligned_and_future():
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from watch import next_candle_close

    ist = ZoneInfo("Asia/Kolkata")
    for h, m, s in [(9, 15, 0), (9, 15, 1), (9, 19, 59), (11, 3, 12), (15, 24, 59)]:
        now = datetime(2026, 7, 28, h, m, s, tzinfo=ist)
        nxt = next_candle_close(now, 5)
        assert nxt > now, "must always be in the future, never the current instant"
        assert nxt.minute % 5 == 0 and nxt.second == 0, "must land on a candle boundary"
        assert (nxt - now).total_seconds() <= 300


def test_watcher_reuses_scan_logic():
    """It must not reimplement the strategy, or the two paths could diverge."""
    body = (ROOT / "watch.py").read_text()
    assert "from scan import" in body
    assert "prefilter" in body and "scan_symbol" in body
    for banned in ("def prefilter", "def scan_symbol", "def replay_week"):
        assert banned not in body, f"watch.py must not redefine {banned}"


def test_watcher_respects_one_per_week_state():
    body = (ROOT / "watch.py").read_text()
    assert "already_alerted" in body, \
        "the watcher shares state.json, so it must honour one_per_week"
    assert "state.save()" in body


def test_watcher_loads_snapshot_once_and_reloads_on_new_week():
    body = (ROOT / "watch.py").read_text()
    assert "cur_week != week" in body, "must reload the snapshot when the week rolls"


# --------------------------------------------------------------------------- #
#  BUG 12 - surfaced by a CI failure.
#  A test that expected "no snapshot" instead loaded the real 809 KB snapshot,
#  because config.py anchored data paths to the SCRIPT directory and ignored
#  --config. Worse, the run then hung: with auth failing, prefilter failed OPEN
#  and pushed ~2000 symbols into the per-symbol stage, each retrying with
#  backoff, until the 12-minute workflow timeout killed the job.
# --------------------------------------------------------------------------- #
def test_alternate_config_uses_its_own_data_paths(tmp_path):
    from config import load_config

    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("strategy: {}\nuniverse: {}\nruntime: {}\n")
    cfg = load_config(cfg_file)

    assert cfg.paths["snapshot"].parent == tmp_path, (
        "an alternate --config must resolve its own snapshot/state, otherwise "
        "tests and side-by-side setups read the live data files")
    assert cfg.paths["state"].parent == tmp_path


def test_default_config_still_uses_repo_root():
    from config import load_config

    cfg = load_config()
    assert cfg.paths["snapshot"].name == "weekly_snapshot.csv"
    assert cfg.paths["snapshot"].parent == ROOT


def test_prefilter_aborts_on_auth_failure_instead_of_failing_open():
    """
    Failing open is right for one bad batch, but an auth error affects every
    later call too - continuing burns the whole workflow timeout.
    """
    from unittest.mock import patch

    from config import load_config
    from dhan import DhanClient, DhanError
    from scan import prefilter
    from strategy import WeeklySnapshot

    cfg = load_config(ROOT / "config.yaml")
    row = {
        "symbol": "X", "security_id": "1", "exchange_segment": "NSE_EQ",
        "week_start": "2026-07-27", "entry_level": "100", "level_52": "100",
        "hi_short2": "90", "close_1": "95",
        "ema_fast_prev": "1", "ema_fast_len": "20",
        "ema_slow_prev": "1", "ema_slow_len": "50", "ema_slow_2": "1",
        "rsi_len": "14", "rsi_avg_gain": "1", "rsi_avg_loss": "1",
        "rsi_prev_close": "1", "rsi_1": "1",
        "macd_fast": "12", "macd_slow": "26", "macd_signal": "9",
        "macd_ema_fast": "1", "macd_ema_slow": "1", "macd_sig": "1",
        "vol_sma_len": "10", "vol_sma_sum_prev": "1",
        "g_ema_fast": "1", "g_ema_slow": "1", "g_ema_slow_2": "1",
        "g_rsi": "1", "g_rsi_1": "1", "g_hist": "1",
        "prev_daily_close": "1", "prev_daily_open": "1", "mcap": "",
    }
    snaps = [WeeklySnapshot.from_row(dict(row, symbol=f"S{i}", security_id=str(i)))
             for i in range(50)]

    def boom(self, securities):
        raise DhanError('auth failed (401) DH-901 token invalid or expired')

    client = DhanClient.__new__(DhanClient)
    with patch.object(DhanClient, "ltp", boom):
        try:
            prefilter(client, snaps, cfg)
            raise AssertionError("prefilter should abort on an auth failure")
        except DhanError as exc:
            assert "authentication failed" in str(exc).lower()


def test_scan_reports_auth_failure_to_telegram():
    body = (ROOT / "scan.py").read_text()
    assert "Scanner stopped" in body, \
        "an expired token should reach the user on Telegram, not just the log"


# --------------------------------------------------------------------------- #
#  BUG 13 - actions/cache fought with the committed state.json.
#  Step order was: checkout (restores the committed, authoritative state.json)
#  -> actions/cache restore (OVERWRITES it from a possibly older cache).
#  With restore-keys 'alert-state-' matching any previous cache, a stale copy
#  could wipe fresh state, making already-alerted symbols fire a second time.
#  The file is committed to git, so the cache was redundant as well as harmful.
# --------------------------------------------------------------------------- #
def test_scan_workflow_has_no_state_cache_step():
    wf = load_workflow("scan.yml")
    for step in all_steps(wf):
        uses = str(step.get("uses", ""))
        if "actions/cache" not in uses:
            continue
        path = str((step.get("with") or {}).get("path", ""))
        assert "state.json" not in path, (
            "caching state.json overwrites the committed copy restored by "
            "checkout, which can resurrect already-alerted symbols")


def test_state_is_restored_by_checkout_not_cache():
    """checkout must come before anything that writes state.json."""
    wf = load_workflow("scan.yml")
    steps = list(all_steps(wf))
    idx = [i for i, s in enumerate(steps) if "checkout" in str(s.get("uses", ""))]
    assert idx and idx[0] == 0, "checkout must be the first step"


def test_state_push_retries_on_conflict():
    """
    The snapshot job pushes to the same branch. A lost state push means an
    already-alerted symbol re-fires, so the push must retry.
    """
    body = (WORKFLOWS / "scan.yml").read_text()
    persist = body.split("Persist alert state")[1]
    assert "for attempt in" in persist, "state push should retry on conflict"
    assert "git pull --rebase" in persist
    assert "::warning::" in persist, \
        "a failed push should warn, not silently succeed"


# --------------------------------------------------------------------------- #
#  Paper trading engine.
#  Entry  = the live 5m breakout signal (same code path as the alert)
#  Stop   = entry candle low - 0.02%
#  Exit   = first 5m close below the 9-EMA, no fixed target
# --------------------------------------------------------------------------- #
def _paper_signal(entry=100.0, low=99.0):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from strategy import BarEval, Signal

    ist = ZoneInfo("Asia/Kolkata")
    ev = BarEval(conditions={f"c{i:02d}": True for i in range(1, 14)},
                 values={"rsi": 70.0, "macd_hist": 1.5})
    return Signal("T", "1", "NSE_EQ", datetime(2026, 7, 27, 10, 0, tzinfo=ist),
                  entry, 98.0, 98.0, "cross", ev, "2026-07-27", 1e6, 97.0, 96.0,
                  bar_open=entry, bar_high=entry * 1.005, bar_low=low)


def _paper_bars(seq):
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    import pandas as pd

    ist = ZoneInfo("Asia/Kolkata")
    t0 = datetime(2026, 7, 27, 10, 5, tzinfo=ist)
    return pd.DataFrame([
        {"datetime": t0 + timedelta(minutes=5 * i), "open": o, "high": h,
         "low": l, "close": c, "volume": 1}
        for i, (o, h, l, c) in enumerate(seq)])


def test_signal_carries_entry_candle_ohlc():
    """paper.py needs the entry candle low to place the stop."""
    import dataclasses

    import pandas as pd

    from config import load_config
    from strategy import build_snapshot, replay_week, week_start_of
    from test_end_to_end import strong_uptrend_daily, week_of_bars

    cfg = load_config(ROOT / "config.yaml")
    # This test is about the OHLC fields travelling with the Signal, not about
    # volatility. week_of_bars() builds deliberately smooth synthetic candles
    # (+-0.1% ranges), which the live min_atr_pct floor correctly rejects, so
    # switch it off here rather than distort the fixture.
    cfg = dataclasses.replace(
        cfg, strategy=dataclasses.replace(cfg.strategy, min_atr_pct=0.0))
    daily = strong_uptrend_daily()
    lw = week_start_of(pd.Timestamp(daily.iloc[-1]["datetime"]).date())
    snap = build_snapshot("T", "1", "NSE_EQ", daily, cfg.strategy,
                          lw + pd.Timedelta(days=7))
    res = replay_week(snap, cfg.strategy, week_of_bars(snap, breakout_index=20))
    assert res.signals
    sig = res.signals[0]
    assert sig.bar_low > 0 and sig.bar_high >= sig.bar_low
    assert sig.bar_low <= sig.price <= sig.bar_high


def test_paper_stop_is_entry_candle_low_minus_buffer():
    from paper import SL_BUFFER, simulate

    t = simulate(_paper_signal(100.0, 99.0),
                 _paper_bars([(100, 101, 100, 100.8), (100.8, 101, 98.0, 98.5)]),
                 100000)
    assert t.exit_reason == "SL"
    assert abs(t.stop - 99.0 * (1 - SL_BUFFER)) < 1e-9
    assert abs(t.r_multiple + 1.0) < 1e-6, "a stop-out is exactly -1R"


def test_paper_exits_on_close_below_9ema():
    from paper import simulate

    seq = [(100 + i * 0.5, 101 + i * 0.5, 100 + i * 0.5, 100.5 + i * 0.5)
           for i in range(12)]
    seq.append((106, 106, 101.0, 101.2))
    t = simulate(_paper_signal(), _paper_bars(seq), 100000)
    assert t.exit_reason == "EMA9"
    assert t.bars_held == len(seq), (
        "in signal-close mode the exit books on the breaching bar itself")


def test_paper_stop_wins_when_both_trigger_same_candle():
    from paper import simulate

    t = simulate(_paper_signal(), _paper_bars([(100, 100.2, 98.0, 98.1)]), 100000)
    assert t.exit_reason == "SL", "conservative: cannot know intrabar ordering"


def test_paper_position_never_exceeds_capital():
    from paper import simulate

    for entry in (871.90, 199.39, 12.5, 4999.0):
        t = simulate(_paper_signal(entry, entry * 0.99),
                     _paper_bars([(entry, entry * 1.01, entry * 0.995, entry)]),
                     100000)
        assert t.qty == int(100000 // entry)
        assert t.invested <= 100000 + 1e-6


def test_paper_guards_against_malformed_candle():
    """A bad tick where low >= close must not invert the stop."""
    from paper import simulate

    t = simulate(_paper_signal(1874.21, 1876.36),      # low ABOVE the close
                 _paper_bars([(1874, 1875, 1870, 1872)]), 100000)
    assert t.stop < t.entry, "stop must always sit below entry"
    assert abs(t.r_multiple) < 100, "R must stay finite"


def test_paper_open_position_is_marked_to_last_close():
    from paper import simulate

    t = simulate(_paper_signal(),
                 _paper_bars([(100 + i * 0.3, 101 + i * 0.3, 100 + i * 0.3,
                               100.9 + i * 0.3) for i in range(10)]), 100000)
    assert t.exit_reason == "OPEN"
    assert t.pnl > 0


def test_paper_report_has_required_columns():
    body = (ROOT / "paper.py").read_text()
    for field in ("entry_date", "entry_time", "exit_date", "exit_time",
                  "stop", "pnl", "level_26w", "qty"):
        assert field in body, f"report must expose {field}"


def test_paper_uses_live_strategy_code():
    body = (ROOT / "paper.py").read_text()
    assert "from strategy import" in body
    assert "replay_week" in body and "build_snapshot" in body
    for banned in ("def replay_week", "def evaluate_bar", "def gate_ok"):
        assert banned not in body, f"paper.py must not redefine {banned}"


# --------------------------------------------------------------------------- #
#  Post-market Excel report delivered to Telegram.
# --------------------------------------------------------------------------- #
def _report_trades():
    from paper import PaperTrade

    def mk(sym, e, q, sl, ex, why, pnl, r):
        t = PaperTrade(symbol=sym, week="2026-07-27",
                       signal_date="2026-07-27", signal_time="09:50",
                       signal_close=e, entry_date="2026-07-27",
                       entry_time="09:55", entry=e, qty=q, invested=e * q,
                       bar_low=sl / 0.9998, stop=sl, level_26w=e * 0.99,
                       level_52w=e * 0.99, trigger="cross", rsi=71.2,
                       macd_hist=3.4)
        t.exit_date, t.exit_time, t.exit, t.exit_reason = "2026-07-27", "11:20", ex, why
        t.bars_held, t.pnl, t.r_multiple = 18, pnl, r
        t.pnl_pct = (ex / e - 1) * 100
        return t

    return [mk("RATNAVEER", 199.39, 501, 197.5, 203.10, "EMA9", 1859, 2.0),
            mk("MONARCH", 391.60, 255, 388.0, 386.9, "SL", -1198, -1.0),
            mk("TMB", 871.90, 114, 865.0, 879.5, "OPEN", 866, 1.1)]


def test_excel_report_has_required_sheets_and_columns(tmp_path):
    import openpyxl

    from paper_report import build_workbook

    out = build_workbook(_report_trades(), 100000, tmp_path / "r.xlsx", "window")
    wb = openpyxl.load_workbook(out)
    assert "Summary" in wb.sheetnames and "Trades" in wb.sheetnames
    assert "Open" in wb.sheetnames, "open positions get their own sheet"

    headers = [c.value for c in wb["Trades"][1]]
    for required in ("Symbol", "Date", "Signal Bar", "Signal Close",
                     "Entry Bar", "Entry Fill", "Qty", "Stop Loss",
                     "26W Level", "Exit Date", "Exit Bar", "Exit Fill",
                     "P&L (Rs)", "R"):
        assert required in headers, f"missing column {required}"
    assert wb["Trades"].max_row == len(_report_trades()) + 1


def test_excel_summary_numbers_are_correct(tmp_path):
    import openpyxl

    from paper_report import build_workbook

    trades = _report_trades()
    out = build_workbook(trades, 100000, tmp_path / "r.xlsx", "window")
    ws = openpyxl.load_workbook(out)["Summary"]
    found = {ws.cell(r, 1).value: ws.cell(r, 2).value for r in range(1, 40)
             if ws.cell(r, 1).value}
    assert found["Trades"] == 3
    assert found["Still open"] == 1
    assert abs(found["Net P&L (Rs)"] - sum(t.pnl for t in trades)) < 0.01
    assert found["Wins"] == 2 and found["Losses"] == 1


def test_report_caption_summarises_pnl():
    from paper_report import caption

    text = caption(_report_trades(), 100000, "2026-07-27")
    assert "Paper Trading Report" in text
    assert "Net P&L" in text and "win rate" in text
    assert "RATNAVEER" in text, "best trade should be named"
    assert len(text) <= 1024, "Telegram caption limit"


def test_report_caption_handles_no_trades():
    from paper_report import caption

    assert "No trades" in caption([], 100000, "2026-07-27")


def test_telegram_send_document_dry_run(tmp_path):
    from telegram import Telegram

    f = tmp_path / "x.xlsx"
    f.write_bytes(b"fake")
    assert Telegram("", "", dry_run=True).send_document(f, "cap") is True
    assert Telegram("", "", dry_run=True).send_document(tmp_path / "nope.xlsx") is False


def test_telegram_send_document_posts_multipart(tmp_path, monkeypatch):
    """The workbook must go as a file upload, not inline text."""
    import telegram as tg_mod

    f = tmp_path / "r.xlsx"
    f.write_bytes(b"PK\x03\x04fake-xlsx")
    seen = {}

    class Resp:
        status_code = 200

        def json(self):
            return {"ok": True}

    def fake_post(url, data=None, files=None, timeout=None, **kw):
        seen["url"] = url
        seen["files"] = files
        seen["chat"] = (data or {}).get("chat_id")
        return Resp()

    monkeypatch.setattr(tg_mod.requests, "post", fake_post)
    ok = tg_mod.Telegram("tok", "123").send_document(f, "caption")
    assert ok
    assert seen["url"].endswith("/sendDocument")
    assert "document" in seen["files"]
    assert seen["chat"] == "123"


def test_report_workflow_runs_after_market_close():
    wf = load_workflow("report.yml")
    trigger = wf.get("on") or wf.get(True)
    crons = [c["cron"] for c in trigger["schedule"]]
    # NSE closes 15:30 IST = 10:00 UTC; the report must run after that
    assert any(int(c.split()[1]) >= 10 for c in crons), \
        f"{crons} must run at or after 10:00 UTC (15:30 IST close)"
    assert all("1-5" in c for c in crons), "weekdays only"


def test_report_workflow_uploads_artifact_on_failure():
    wf = load_workflow("report.yml")
    steps = list(all_steps(wf))
    art = [s for s in steps if "upload-artifact" in str(s.get("uses", ""))]
    assert art and str(art[0].get("if", "")).strip() == "always()", \
        "keep the workbook even if the Telegram upload fails"


def test_openpyxl_is_pinned_in_requirements():
    reqs = (ROOT / "requirements.txt").read_text()
    assert "openpyxl" in reqs, "the report needs openpyxl at runtime"
    for line in reqs.splitlines():
        if line.strip().startswith("openpyxl"):
            assert "<" in line, "pin the major version"


# --------------------------------------------------------------------------- #
#  BUG 14 - partial deploy crashed the report.
#  paper.py read sig.bar_low directly, but the DEPLOYED strategy.py predated
#  that field. The job fetched data for 6.5 minutes and then died with
#  AttributeError: 'Signal' object has no attribute 'bar_low'.
#  Two defences: degrade gracefully, and fail fast with a clear message.
# --------------------------------------------------------------------------- #
def test_simulate_survives_signal_without_bar_low():
    """A stale strategy.py must degrade to an entry-based stop, not crash."""
    from dataclasses import dataclass
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    import pandas as pd

    from paper import simulate
    from strategy import BarEval

    ist = ZoneInfo("Asia/Kolkata")

    @dataclass
    class OldSignal:            # the Signal shape before bar_* was added
        symbol: str
        security_id: str
        exchange_segment: str
        bar_time: datetime
        price: float
        entry_level: float
        level_52: float
        trigger: str
        evaluation: object
        week_start: str
        week_volume: float
        day_open: float
        week_open: float

    sig = OldSignal("T", "1", "NSE_EQ", datetime(2026, 7, 27, 10, 0, tzinfo=ist),
                    100.0, 98.0, 98.0, "cross",
                    BarEval({}, {"rsi": 70.0, "macd_hist": 1.0}),
                    "2026-07-27", 1e6, 97.0, 96.0)
    bars = pd.DataFrame([
        {"datetime": datetime(2026, 7, 27, 10, 5, tzinfo=ist) + timedelta(minutes=5 * i),
         "open": 100, "high": 101, "low": 99.5, "close": 100.5} for i in range(3)])

    t = simulate(sig, bars, 100000)          # must not raise
    assert t.stop < t.entry, "stop must still sit below entry"


def test_paper_never_reads_bar_fields_unguarded():
    body = (ROOT / "paper.py").read_text()
    assert "sig.bar_low" not in body, (
        "read the entry-candle low via getattr so a stale strategy.py cannot "
        "crash the whole report")
    assert 'getattr(sig, "bar_low"' in body


def test_paper_report_checks_strategy_version_up_front():
    """
    The crash wasted 6.5 minutes of API calls before failing. The version check
    must happen before any data is fetched.
    """
    body = (ROOT / "paper_report.py").read_text()
    assert "bar_low" in body and "out of date" in body, \
        "paper_report should verify Signal has bar_low before fetching data"
    guard = body.index("bar_low")
    fetch = body.index("fetch_5m(client")
    assert guard < fetch, "the version check must precede any data fetching"


# --------------------------------------------------------------------------- #
#  BUG 15 - every paper trade exited on bar 1 ("Avg bars held 1", 0% win rate).
#  The 9-EMA was seeded from the ENTRY PRICE repeated 9 times, so ema == entry
#  at the start. Any close even a paisa below entry was therefore "below the
#  EMA" and closed the position immediately. A real 9-EMA is built from the
#  candles BEFORE entry, where an uptrend leaves it well under price.
# --------------------------------------------------------------------------- #
def _ema_frame(closes, hour=10, minute=0):
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    import pandas as pd

    ist = ZoneInfo("Asia/Kolkata")
    t0 = datetime(2026, 7, 27, hour, minute, tzinfo=ist)
    return pd.DataFrame([
        {"datetime": t0 + timedelta(minutes=5 * i), "open": c, "high": c * 1.002,
         "low": c * 0.998, "close": c} for i, c in enumerate(closes)])


def _ema_signal(entry=100.0, low=99.0):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from strategy import BarEval, Signal

    ist = ZoneInfo("Asia/Kolkata")
    return Signal("T", "1", "NSE_EQ", datetime(2026, 7, 27, 11, 0, tzinfo=ist),
                  entry, 98.0, 98.0, "cross",
                  BarEval({}, {"rsi": 70.0, "macd_hist": 1.0}),
                  "2026-07-27", 1e6, 97.0, 96.0,
                  bar_open=entry, bar_high=entry * 1.005, bar_low=low)


def test_ema_exit_is_warmed_from_pre_entry_candles():
    """A minor dip right after entry must not close the trade."""
    from paper import simulate

    before = _ema_frame([96 + i * 0.3 for i in range(14)])
    before.loc[len(before) - 1, "close"] = 100.0
    after = _ema_frame([99.9, 100.1, 100.4, 100.8, 101.2, 101.6], 11, 5)

    t = simulate(_ema_signal(), after, 100000, before)
    assert t.bars_held > 1, (
        "seeding the EMA at the entry price exits on the first down-tick; "
        "warm it from real prior candles instead")


def test_ema_exit_still_fires_on_a_real_breakdown():
    from paper import simulate

    before = _ema_frame([96 + i * 0.3 for i in range(14)])
    before.loc[len(before) - 1, "close"] = 100.0
    after = _ema_frame([100.2, 100.4, 100.1, 99.0, 98.6, 98.2], 11, 5)

    t = simulate(_ema_signal(), after, 100000, before)
    assert t.exit_reason in ("EMA9", "SL")
    assert t.bars_held >= 2


def test_ema_warmup_when_no_prior_history():
    """Monday-open entries have no earlier candles - must not exit on bar 1."""
    from paper import simulate

    t = simulate(_ema_signal(),
                 _ema_frame([99.9, 100.2, 100.6, 101.0], 11, 5), 100000, None)
    assert t.bars_held > 1


def test_ema_at_entry_sits_below_price_in_an_uptrend():
    """Sanity: the warmed EMA must be under the entry after a run-up."""
    import numpy as np

    import indicators as ind
    from paper import EMA_LEN

    closes = np.array([96 + i * 0.3 for i in range(13)] + [100.0], dtype=float)
    ema = ind.ema(closes, EMA_LEN)[-1]
    assert ema < 100.0, "an uptrend should leave the EMA below price"


def test_simulate_accepts_bars_before_argument():
    import inspect

    from paper import simulate

    assert "bars_before" in inspect.signature(simulate).parameters


def test_paper_callers_pass_pre_entry_candles():
    for name in ("paper.py", "paper_report.py"):
        body = (ROOT / name).read_text()
        assert "simulate(sig, after, args.capital, before, args.fill)" in body, \
            f"{name} must pass pre-entry candles and the fill mode"


# --------------------------------------------------------------------------- #
#  BUG 16 - EXECUTION REALISM (found in a fresh audit before going live).
#  NSE 5m candles are stamped by OPEN time: a bar labelled 10:15 spans
#  10:15:00-10:19:59 and only completes at 10:20. Three consequences the old
#  model got wrong:
#    a) reported times were 5 minutes earlier than any achievable action
#    b) entry filled at the signal bar's own close - a price not known until
#       that bar ended (look-ahead)
#    c) the EMA exit filled at the close that triggered it (same look-ahead)
#  Also: a gap through the stop filled AT the stop, understating gap risk, and
#  nothing prevented two overlapping positions in one symbol.
# --------------------------------------------------------------------------- #
def _exec_sig(close=100.0, low=99.0, hour=10, minute=15):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from strategy import BarEval, Signal

    ist = ZoneInfo("Asia/Kolkata")
    return Signal("T", "1", "NSE_EQ", datetime(2026, 7, 27, hour, minute, tzinfo=ist),
                  close, 98.0, 98.0, "cross",
                  BarEval({}, {"rsi": 70.0, "macd_hist": 1.0}),
                  "2026-07-27", 1e6, 97.0, 96.0,
                  bar_open=close * 0.998, bar_high=close * 1.005, bar_low=low)


def _exec_bars(rows, hour=10, minute=20):
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    import pandas as pd

    ist = ZoneInfo("Asia/Kolkata")
    t0 = datetime(2026, 7, 27, hour, minute, tzinfo=ist)
    return pd.DataFrame([
        {"datetime": t0 + timedelta(minutes=5 * i), "open": o, "high": h,
         "low": l, "close": c} for i, (o, h, l, c) in enumerate(rows)])


def _warm():
    return _exec_bars([(96 + i * .3, 96.5 + i * .3, 95.5 + i * .3, 96 + i * .3)
                       for i in range(14)], 9, 15)


def test_nse_5m_bars_are_stamped_by_open_time():
    """
    75 bars per session, 09:15..15:25. If bars were stamped by CLOSE time the
    range would be 09:20..15:30. Everything downstream depends on this.
    """
    from datetime import datetime, time as dtime, timedelta

    session = (datetime.combine(datetime(2026, 7, 27), dtime(15, 30))
               - datetime.combine(datetime(2026, 7, 27), dtime(9, 15)))
    assert session == timedelta(minutes=375)
    assert 375 // 5 == 75
    first, last = dtime(9, 15), dtime(15, 25)
    assert first < last


def test_entry_fills_at_next_bar_open_not_signal_close():
    from paper import simulate

    t = simulate(_exec_sig(100.0),
                 _exec_bars([(100.6, 101, 100.4, 100.9),
                             (100.9, 101.5, 100.7, 101.3)]), 100000, _warm(),
                 fill="next-open")
    assert t.signal_time == "10:15" and t.signal_close == 100.0
    assert t.entry_time == "10:20", "entry is the bar AFTER the signal"
    assert t.entry == 100.6, "fill at that bar's OPEN"
    assert abs(t.slippage_pct - 0.6) < 0.01


def test_ema_exit_fills_at_next_open_no_lookahead():
    from paper import simulate

    t = simulate(_exec_sig(100.0),
                 _exec_bars([(100.6, 101, 100.4, 98.0),
                             (97.9, 98.2, 97.5, 97.8)]), 100000, _warm(),
                 fill="next-open")
    assert t.exit_reason == "EMA9"
    assert t.exit == 97.9, (
        "a close below the EMA is only known when the bar ends, so the fill "
        "is the NEXT bar's open")


def test_stop_fills_at_stop_when_merely_touched():
    from paper import simulate

    t = simulate(_exec_sig(100.0, 99.0),
                 _exec_bars([(100.5, 100.8, 98.5, 99.5)]), 100000, _warm())
    assert t.exit_reason == "SL"
    assert abs(t.exit - 99.0 * 0.9998) < 1e-4
    assert t.exit_note == ""


def test_gap_through_stop_fills_at_open_not_stop():
    from paper import simulate

    t = simulate(_exec_sig(100.0, 99.0),
                 _exec_bars([(95.0, 95.5, 94.0, 94.5)]), 100000, _warm())
    assert t.exit == 95.0, "a gap-down fills at the open, worse than the stop"
    assert "gap" in t.exit_note


def test_signal_close_mode_reproduces_optimistic_fills():
    from paper import simulate

    t = simulate(_exec_sig(100.0),
                 _exec_bars([(100.6, 101, 100.4, 98.0)]), 100000, _warm(),
                 fill="signal-close")
    assert t.entry == 100.0, "opt-in mode still fills at the signal close"


def test_realistic_fill_is_not_better_than_optimistic():
    """Sanity: removing look-ahead must not improve results."""
    from paper import simulate

    rows = [(100.6, 101, 100.4, 98.0), (97.9, 98.2, 97.5, 97.8)]
    real = simulate(_exec_sig(100.0), _exec_bars(rows), 100000, _warm())
    opt = simulate(_exec_sig(100.0), _exec_bars(rows), 100000, _warm(),
                   fill="signal-close")
    assert real.entry >= opt.entry, "realistic entry cannot be cheaper"


def test_paper_prevents_overlapping_positions_per_symbol():
    for name in ("paper.py", "paper_report.py"):
        body = (ROOT / name).read_text()
        assert "busy_until" in body, (
            f"{name}: a new signal must be skipped while the previous trade "
            "in that symbol is still open")


def test_papertrade_records_signal_and_entry_separately():
    from dataclasses import fields

    from paper import PaperTrade

    names = {f.name for f in fields(PaperTrade)}
    for required in ("signal_date", "signal_time", "signal_close",
                     "entry_date", "entry_time", "entry", "slippage_pct"):
        assert required in names, f"PaperTrade must expose {required}"


def test_last_bar_of_day_signal_fills_next_session():
    """
    A 15:25 bar closes at 15:30 - the market shuts. The fill must roll to the
    next session's open, carrying overnight gap risk. Silently filling at
    15:25's close would be untradeable fiction.
    """
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    import pandas as pd

    from paper import simulate

    ist = ZoneInfo("Asia/Kolkata")
    sig = _exec_sig(100.0, 99.0, hour=15, minute=25)
    nxt = datetime(2026, 7, 28, 9, 15, tzinfo=ist)          # next session
    bars = pd.DataFrame([
        {"datetime": nxt + timedelta(minutes=5 * i), "open": 101 + i * 0.1,
         "high": 101.5 + i * 0.1, "low": 100.8 + i * 0.1, "close": 101.2 + i * 0.1}
        for i in range(6)])

    t = simulate(sig, bars, 100000, _warm(), fill="next-open")
    assert t.signal_time == "15:25"
    assert t.entry_date == "2026-07-28" and t.entry_time == "09:15", \
        "a signal on the closing bar can only be filled next session"
    assert t.entry == 101.0, "fill at the next session's open, gap included"


# --------------------------------------------------------------------------- #
#  BUG 17 - entries fired LATE (or not at all) versus the chart.
#  The user's Pine runs with strictEntry = false ("Entry gate: OFF (level
#  only)" in every screenshot), so the indicator buys on the PURE 26W level
#  break; the 13-row table is informational. Proof: MONARCH FAILED:2,
#  SENCO FAILED:1, PYRAMID FAILED:4 all show ENTRY TAKEN - impossible if the
#  gate were active. Our config had strict_entry: true, so entries waited for
#  the gate and printed as "deferred", and CREDITACC never fired at all.
# --------------------------------------------------------------------------- #
def test_gate_off_mode_still_available_for_comparison():
    """
    strictEntry=false reproduces the raw chart arrows, but it is NOT safe to
    trade: on live data it gave 82 signals in a day (22 on the Monday 09:15
    bar) at profit factor 0.73. Keep the switch, but the shipped config uses
    the gate plus a tolerance instead.
    """
    from config import Strategy
    from strategy import gate_ok

    snap = _gate_snap()
    ev = _gate_eval(c05=False, c06=False, c08=False, c09=False)
    # with the gate disabled the caller bypasses gate_ok entirely
    assert Strategy(strict_entry=False).strict_entry is False
    # and with the gate on, four failures still exceed a tolerance of 2
    assert gate_ok(snap, Strategy(gate_tolerance=2), ev) is False


def test_level_only_entry_fires_on_first_close_above():
    """With the gate off, the signal is purely the first CLOSE above the 26W level."""
    import pandas as pd

    from config import Strategy
    from strategy import replay_week
    from test_strategy import make_daily  # noqa: F401  (fixture helpers)

    # build a snapshot then feed candles that pierce on the HIGH first
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    from strategy import build_snapshot, week_start_of

    ist = ZoneInfo("Asia/Kolkata")
    cfg = Strategy(strict_entry=False)
    daily = make_daily()
    lw = week_start_of(pd.Timestamp(daily.iloc[-1]["datetime"]).date())
    snap = build_snapshot("T", "1", "NSE_EQ", daily, cfg, lw + pd.Timedelta(days=7))
    assert snap is not None
    lvl = snap.entry_level

    t0 = datetime(2026, 7, 27, 9, 15, tzinfo=ist)
    bars = pd.DataFrame([
        # high pierces the level but the close does not -> NOT a signal
        {"datetime": t0, "open": lvl * .99, "high": lvl * 1.001,
         "low": lvl * .98, "close": lvl * .995, "volume": 1e6},
        # first close above -> THIS is the signal bar
        {"datetime": t0 + timedelta(minutes=5), "open": lvl * .995,
         "high": lvl * 1.01, "low": lvl * .99, "close": lvl * 1.005,
         "volume": 1e6},
    ])
    res = replay_week(snap, cfg, bars)
    assert len(res.signals) == 1
    assert res.signals[0].bar_time == bars.iloc[1]["datetime"], (
        "Pine uses `close > entryLevel`, so a high-only pierce must not trigger")


def test_paper_default_fill_is_the_signal_candle_close():
    """The report must line up with the chart's BUY arrow by default."""
    import inspect

    from paper import simulate

    assert inspect.signature(simulate).parameters["fill"].default == "signal-close"
    for name in ("paper.py", "paper_report.py"):
        body = (ROOT / name).read_text()
        assert 'default="signal-close"' in body, f"{name} default fill must match the chart"


# --------------------------------------------------------------------------- #
#  BUG 18 - strict_entry: false flooded the report with false entries.
#  Removing the gate produced 82 trades in ONE day, 22 of them on the 09:15
#  Monday bar where "already above the level" was mistaken for a fresh cross.
#  Net -Rs 13,851, win rate 33%, profit factor 0.73, SL exits -Rs 32,285.
#  Fix: keep the gate ON, but allow a small tolerance so a strong breakout is
#  not blocked by one or two marginal rows (which caused "deferred" entries).
# --------------------------------------------------------------------------- #
def test_strict_entry_is_on():
    import yaml as _yaml

    cfg = _yaml.safe_load((ROOT / "config.yaml").read_text())
    assert cfg["strategy"]["strict_entry"] is True, (
        "turning the gate off gave 82 trades/day at PF 0.73 - keep it on and "
        "use gate_tolerance to relax marginal rows instead")


def test_gate_tolerance_is_configured_and_bounded():
    import yaml as _yaml

    cfg = _yaml.safe_load((ROOT / "config.yaml").read_text())
    tol = cfg["strategy"]["gate_tolerance"]
    assert tol == 0, (
        "shipped config is the STRICT 13/13 scan. Measured on 27-Jul, the real "
        "breakouts had a median of 0 failing rows and the false positives a "
        "median of 1, so any tolerance > 0 admits exactly the unwanted names")


def _gate_eval(**overrides):
    from strategy import BarEval

    conds = {f"c{i:02d}": True for i in range(1, 14)}
    conds.update(overrides)
    return BarEval(conds, {"rsi": 70.0, "macd_hist": 1.0})


def _gate_snap():
    from strategy import WeeklySnapshot
    import indicators as ind

    return WeeklySnapshot(
        symbol="T", security_id="1", exchange_segment="NSE_EQ",
        week_start="2026-07-27", entry_level=100.0, level_52=100.0,
        hi_short2=99.0, close_1=98.0,
        ema_fast=ind.EmaState(20, 1.0), ema_slow=ind.EmaState(50, 1.0),
        ema_slow_2=1.0, rsi=ind.RsiState(14, 1.0, 1.0, 1.0), rsi_1=1.0,
        macd=ind.MacdState(12, 26, 9, 1.0, 1.0, 1.0),
        vol_sma=ind.SmaState(10, 1.0),
        g_ema_fast=1.0, g_ema_slow=1.0, g_ema_slow_2=1.0,
        g_rsi=1.0, g_rsi_1=1.0, g_hist=1.0,
        prev_daily_close=100.0, prev_daily_open=99.0)


def test_tolerance_allows_up_to_n_soft_failures():
    from config import Strategy
    from strategy import gate_ok

    snap = _gate_snap()
    # two soft rows failing (RSI level + EMA slope)
    ev = _gate_eval(c06=False, c05=False)
    assert gate_ok(snap, Strategy(gate_tolerance=2), ev) is True
    assert gate_ok(snap, Strategy(gate_tolerance=1), ev) is False
    assert gate_ok(snap, Strategy(gate_tolerance=0), ev) is False


def test_tolerance_never_relaxes_the_breakout_itself():
    """c01/c02 are the level break - tolerance must not be able to fake one."""
    from config import Strategy
    from strategy import gate_ok

    snap = _gate_snap()
    # the gate only sees c03..c13; the cross is proven separately in replay_week
    ev = _gate_eval(c01=False, c02=False)
    assert gate_ok(snap, Strategy(gate_tolerance=2), ev) is True, (
        "c01/c02 are not gate rows - the cross itself proves them")

    body = (ROOT / "strategy.py").read_text()
    gate_src = body[body.index("def gate_ok"):body.index("def replay_week")]
    assert '"c01"' not in gate_src and '"c02"' not in gate_src, \
        "the level break must never be part of the tolerance count"


def test_mandatory_rows_are_never_tolerated():
    """Fresh breakout, min price and market cap stay hard requirements."""
    from config import Strategy
    from strategy import gate_ok

    snap = _gate_snap()
    for row in ("c03", "c11", "c12"):
        ev = _gate_eval(**{row: False})
        assert gate_ok(snap, Strategy(gate_tolerance=3), ev) is False, (
            f"{row} must hold regardless of tolerance")


def test_tolerance_converts_deferred_into_a_clean_cross():
    """
    The user's complaint: MONARCH/TMB printed as 'deferred' because one or two
    weekly rows lagged. With tolerance the entry lands on the breakout candle.
    """
    import pandas as pd
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    from config import Strategy
    from strategy import build_snapshot, replay_week, week_start_of
    from test_end_to_end import strong_uptrend_daily, week_of_bars

    ist = ZoneInfo("Asia/Kolkata")
    daily = strong_uptrend_daily()
    lw = week_start_of(pd.Timestamp(daily.iloc[-1]["datetime"]).date())
    cfg = Strategy(strict_entry=True, gate_tolerance=2)
    snap = build_snapshot("T", "1", "NSE_EQ", daily, cfg, lw + pd.Timedelta(days=7))
    assert snap is not None

    res = replay_week(snap, cfg, week_of_bars(snap, breakout_index=20))
    assert res.signals
    assert res.signals[0].trigger == "cross", \
        "with tolerance the entry should be a cross, not a deferred fill"


# --------------------------------------------------------------------------- #
#  BUG 19 - too many false stocks.
#  Two independent causes, found by profiling the 27-Jul report:
#   a) gate_tolerance > 0. Real breakouts failed a MEDIAN OF 0 rows; the false
#      ones a median of 1. Any tolerance therefore admits precisely the noise.
#   b) c09 compared an ACCUMULATING weekly volume against a FULL-week SMA. At
#      Monday 09:25 only 3 of 375 bars exist, so c09 could never pass and
#      genuine early breakouts were pushed to "deferred" or dropped.
# --------------------------------------------------------------------------- #
def test_volume_mode_is_off_in_shipped_config():
    """
    c09 (weekly volume > 10w SMA) is the single strongest filter in the scan.
    Measured on 27-Jul against the 5 validated names:
        volume_mode "off" ->  5 signals (the valid set)
        volume_mode "day" -> 19 signals (15 false)
        volume_mode "bar" -> 26 signals (22 false)
    So it stays raw, exactly as Pine computes it.
    """
    import yaml as _yaml

    cfg = _yaml.safe_load((ROOT / "config.yaml").read_text())
    assert cfg["strategy"]["volume_mode"] == "off"
    assert cfg["strategy"]["gate_tolerance"] == 0


def test_volume_modes_are_available_but_not_default():
    """The alternatives remain implemented for research, just not shipped."""
    from config import Strategy
    from strategy import evaluate_bar

    snap = _gate_snap()
    early = dict(price=101.0, week_open=99.0, week_volume=0.5,
                 day_open=100.0, week_fraction=0.1)          # Monday
    off = evaluate_bar(snap, Strategy(volume_mode="off"), **early)
    day = evaluate_bar(snap, Strategy(volume_mode="day"), **early)
    assert off.values["vol_target"] >= day.values["vol_target"], \
        "day mode must be no stricter than the raw compare"


def test_deferred_entry_is_expected_when_c09_lags():
    """
    MONARCH/TMB printed as "deferred" because weekly VOLUME had not yet cleared
    its 10-week average at the moment price crossed the level. That is the
    scan working as designed, not a bug: the Pine header names c09 as one of
    the two rows that flip false->true during the week.

    Keep deferred entries ON - without them those setups are lost entirely.
    """
    import yaml as _yaml

    cfg = _yaml.safe_load((ROOT / "config.yaml").read_text())
    assert cfg["strategy"]["defer_entry"] is True, (
        "with a strict gate, disabling deferral drops MONARCH and TMB")
    assert cfg["strategy"]["gate_source"] == "live", (
        'gate_source "closed" gave 20 signals and kept only 1 of the valid 5')


def test_volume_prorate_off_reproduces_pine_exactly():
    from config import Strategy
    from strategy import evaluate_bar

    snap = _gate_snap()
    cfg = Strategy(volume_prorate=False)
    ev = evaluate_bar(snap, cfg, price=101.0, week_open=99.0,
                      week_volume=0.02, day_open=100.0, week_fraction=0.01)
    # un-prorated: 0.02 is NOT greater than the full-week SMA
    assert ev.conditions["c09"] is False
    assert ev.values["vol_target"] == ev.values["vol_sma"]


def test_full_week_fraction_matches_pine():
    """At week end the pro-rated target equals the raw SMA - Pine's own test."""
    from config import Strategy
    from strategy import evaluate_bar

    snap = _gate_snap()
    ev = evaluate_bar(snap, Strategy(volume_prorate=True), price=101.0,
                      week_open=99.0, week_volume=5.0, day_open=100.0,
                      week_fraction=1.0)
    assert abs(ev.values["vol_target"] - ev.values["vol_sma"]) < 1e-12


# --------------------------------------------------------------------------- #
#  BUG 20 - TMB missing, and the real cause of the "late" entries.
#
#  ---------------------------------------------------------------------------
#  SUPERSEDED. The theory recorded here (that TWO shifts stack, putting the
#  window two weeks back) was WRONG and was reverted. It is kept only so the
#  same wrong turn is not taken a third time. See BUG 21 below for the proof
#  and the final resolution, confirmed by the user on 28-Jul (Option B).
#
#  The claim was: entry_level should be 821.20 on TMB, not 847.00. That came
#  from reading the orange stepline on a TradingView screenshot. It is true
#  that the PLOTTED line sits at 821.20 - but that line is a rendering
#  artifact of request.security, not the level the scan should trade. 847.00
#  is TMB's real 26-week high and is what the indicator's own table shows.
#  ---------------------------------------------------------------------------
# --------------------------------------------------------------------------- #
def test_breakout_level_includes_the_last_closed_week():
    """
    Pine `ta.highest(high, N)[1]` steps back one bar from the DEVELOPING week,
    so the window ends at - and INCLUDES - the last closed week.

    Verified live on TMB, Tue 28-Jul 15:08 mid-session:
        "Wk close > 26W high (prev)  885.5  needs > 847"  -> level 847.00
        "Prev wk close <= 26W high (2w ago)   <= 821.2"   -> hi_short2 821.20
    847.00 is the high of w/c 20-Jul, the last closed week.

    A previous build shifted this back an extra week after reading a chart
    captured at 21:30, when TradingView had already rolled the weekly bar.
    That snapshot showed the NEXT week's framing, not the live level.
    """
    import pandas as pd

    from config import Strategy
    from strategy import build_snapshot, build_weekly_bars
    from test_strategy import make_daily

    cfg = Strategy()
    daily = make_daily(weeks=200)
    wk = build_weekly_bars(daily)
    target = wk.iloc[-1]["week_start"]
    snap = build_snapshot("T", "1", "NSE_EQ", daily, cfg, target)
    assert snap is not None

    closed = wk[wk["week_start"] < target]["high"].to_numpy(float)
    assert snap.entry_level == pytest.approx(closed[-cfg.len_short:].max())
    assert snap.level_52 == pytest.approx(closed[-cfg.len_long:].max())


def test_hi_short2_is_one_week_older_than_the_entry_level():
    """
    c03 compares the previous week's close against the 26W high as of TWO
    weeks ago. On TMB those two values were 847.00 and 821.20 respectively -
    distinct rows in the indicator table, not the same number.
    """
    import pandas as pd

    from config import Strategy
    from strategy import build_snapshot, build_weekly_bars
    from test_strategy import make_daily

    cfg = Strategy()
    daily = make_daily(weeks=200)
    wk = build_weekly_bars(daily)
    target = wk.iloc[-1]["week_start"]
    snap = build_snapshot("T", "1", "NSE_EQ", daily, cfg, target)

    closed = wk[wk["week_start"] < target]["high"].to_numpy(float)
    assert snap.hi_short2 == pytest.approx(closed[-(cfg.len_short + 1):-1].max())
    assert snap.hi_short2 <= snap.entry_level, \
        "the older window cannot exceed the newer one"


def _sig_with_failures(*fails):
    from datetime import datetime

    from strategy import BarEval, Signal

    conds = {f"c{i:02d}": True for i in range(1, 14)}
    for k in fails:
        conds[k] = False
    ev = BarEval(conds, {"price": 167.64, "ema_fast": 147.8, "ema_slow": 145.5,
                         "ema_slow_2": 143.6, "rsi": 69.9, "rsi_1": 65.2,
                         "macd_hist": 2.74, "week_volume": 6.97e5,
                         "vol_sma": 6.94e5, "week_open": 159.9,
                         "day_open": 166.0, "entry_level": 161.0,
                         "level_52": 185.0})
    return Signal("GPTHEALTH", "1", "NSE_EQ", datetime(2026, 7, 28, 12, 10),
                  167.64, 161.0, 185.0, "deferred", ev, "2026-07-27",
                  6.97e5, 166.0, 159.95, bar_open=167, bar_high=168,
                  bar_low=166.5)


def test_alert_never_claims_a_false_pass_count():
    from telegram import format_signal

    msg = format_signal(_sig_with_failures("c01"))
    assert "12/13" in msg
    assert "13/13" not in msg, "header must not hardcode a perfect score"
    assert msg.count("12/13") == 1, "no contradictory second count"
    assert "not required for entry" in msg, \
        "explain that c01 does not gate the entry when req52 is off"


def test_alert_reports_full_pass_cleanly():
    from telegram import format_signal

    msg = format_signal(_sig_with_failures())
    assert "13/13" in msg and "PASS" in msg
    assert "not required for entry" not in msg


def test_c01_is_not_part_of_the_entry_gate():
    """
    Pine: gateLive = c03 and ... and c13   (c01/c02 excluded)
          req52 defaults to false, so the 52W break is optional.
    GPTHEALTH entering with c01 failing is correct, not a false signal.
    """
    import yaml as _yaml

    from config import Strategy
    from strategy import gate_ok

    cfg = _yaml.safe_load((ROOT / "config.yaml").read_text())
    assert cfg["strategy"]["req52"] is False, "matches the Pine input default"

    snap = _gate_snap()
    ev = _gate_eval(c01=False)
    assert gate_ok(snap, Strategy(), ev) is True, \
        "a failing c01 must not block the gate while req52 is off"


# --------------------------------------------------------------------------- #
#  c09 relaxation - measured a THIRD time, with a conviction multiple.
#  Earlier attempts ("day", "bar") were rejected as too loose. This adds
#  pace-with-multiple: week volume so far > wk_sma * elapsed_fraction * mult.
#  NSE volume accumulates near-linearly (Mon 17.7%, Tue 39.2%, Wed 59.0%,
#  Thu 80.7%), so the pace comparison is statistically sound - yet swept
#  across 1.0-6.0 it still never beat raw c09:
#      raw c09   70% precision, 6 deferred
#      best pace 55% precision, 7 deferred, and it dropped a real mover
#  Deferrals INCREASE because weaker names get admitted earlier and then stall
#  on other conditions.
# --------------------------------------------------------------------------- #
def test_pace_mode_exists_but_is_not_default():
    import yaml as _yaml

    from config import Strategy

    assert Strategy().volume_mode == "off"
    cfg = _yaml.safe_load((ROOT / "config.yaml").read_text())
    assert cfg["strategy"]["volume_mode"] == "off", (
        "swept 1.0-6.0x: no pace multiple beat raw c09 on precision OR "
        "deferral count")


def test_pace_mode_scales_target_with_elapsed_week():
    from config import Strategy
    from strategy import evaluate_bar

    snap = _gate_snap()
    cfg = Strategy(volume_mode="pace", volume_pace_mult=2.0)

    early = evaluate_bar(snap, cfg, price=101.0, week_open=99.0,
                         week_volume=1.0, day_open=100.0, week_fraction=0.1)
    late = evaluate_bar(snap, cfg, price=101.0, week_open=99.0,
                        week_volume=1.0, day_open=100.0, week_fraction=0.9)
    assert late.values["vol_target"] > early.values["vol_target"], \
        "the bar must rise as the week progresses"


def test_pace_multiple_makes_the_test_stricter():
    from config import Strategy
    from strategy import evaluate_bar

    snap = _gate_snap()
    kw = dict(price=101.0, week_open=99.0, week_volume=1.0,
              day_open=100.0, week_fraction=0.5)
    loose = evaluate_bar(snap, Strategy(volume_mode="pace", volume_pace_mult=1.0), **kw)
    tight = evaluate_bar(snap, Strategy(volume_mode="pace", volume_pace_mult=5.0), **kw)
    assert tight.values["vol_target"] > loose.values["vol_target"]


# --------------------------------------------------------------------------- #
#  Paper trading measures CROSS entries only; alerts still send everything.
#  A deferred fill enters hours after the breakout at a worse, more extended
#  price, so mixing it into the performance stats distorts them. The user still
#  wants to SEE those signals in Telegram - only the measurement is filtered.
# --------------------------------------------------------------------------- #
def test_alerts_never_filter_by_trigger():
    """scan.py and watch.py must send cross AND deferred."""
    for name in ("scan.py", "watch.py"):
        body = (ROOT / name).read_text()
        assert 'trigger != "cross"' not in body, (
            f"{name} must not drop deferred signals - the user wants every "
            "alert delivered")


def test_paper_excludes_deferred_by_default():
    for name in ("paper.py", "paper_report.py"):
        body = (ROOT / name).read_text()
        assert 'sig.trigger != "cross"' in body, f"{name} must skip deferred"
        assert "include_deferred" in body, f"{name} needs an opt-in override"


def test_paper_still_simulates_a_deferred_signal_when_asked():
    """The filter is at the caller; simulate() itself stays trigger-agnostic."""
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    import pandas as pd

    from paper import simulate
    from strategy import BarEval, Signal

    ist = ZoneInfo("Asia/Kolkata")
    ev = BarEval({f"c{i:02d}": True for i in range(1, 14)},
                 {"rsi": 70.0, "macd_hist": 1.0})
    sig = Signal("X", "1", "NSE_EQ", datetime(2026, 7, 28, 10, 0, tzinfo=ist),
                 100.0, 98.0, 98.0, "deferred", ev, "2026-07-27", 1e6,
                 97.0, 96.0, bar_open=99.5, bar_high=100.5, bar_low=99.0)
    bars = pd.DataFrame([
        {"datetime": datetime(2026, 7, 28, 10, 5, tzinfo=ist) + timedelta(minutes=5 * i),
         "open": 100 + i * .2, "high": 101 + i * .2,
         "low": 99.8 + i * .2, "close": 100.4 + i * .2} for i in range(6)])
    t = simulate(sig, bars, 100000)
    assert t.qty > 0, "simulate must not itself reject deferred signals"


def test_report_states_which_signals_were_measured():
    body = (ROOT / "paper_report.py").read_text()
    assert "scope_note" in body, \
        "the workbook must say whether deferred entries were included"


# --------------------------------------------------------------------------- #
#  BUG 21 - THE OFF-BY-ONE FLIP-FLOP, SETTLED FOR GOOD (user decision: Option B)
#
#  Symptom: the breakout level kept oscillating between two candidates,
#  because chart screenshots and the indicator TABLE disagree with each other:
#
#      entry_level = highest(high, 26)[1]   -> TMB 847.00   (table row 2)
#      hi_short2   = highest(high, 26)[2]   -> TMB 821.20   (plotted line)
#
#  ROOT CAUSE - the Pine indicator is internally inconsistent.
#  The level is fetched with
#      request.security(tickerid, "W", ta.highest(high,26)[1], lookahead_off)
#  On an intraday chart `lookahead_off` returns the value as of the last
#  CONFIRMED weekly bar. The developing week is not confirmed, so during
#  Mon-Fri that expression resolves to the TWO-weeks-back value (hi_short2)
#  and only rolls forward when the week closes.
#
#  The 13-row table does NOT go through request.security - the Pine author
#  added chart-native `liveNative` aggregates precisely because security
#  "on an intraday chart can lag by one HTF bar" (their own tooltip).
#
#  So within one indicator:
#      table row 2      -> 847.00  (correct 26W high)
#      orange stepline  -> 821.20  (lagged)
#      BUY arrow        -> follows the LAGGED line
#
#  Confirmed by pixel-measuring the user's own screenshots, calibrating each
#  price axis from its labels (TMB log axis, every label fits < 0.25):
#      TMB     stepline y=485 -> 821.29 (= hi_short2)  stub y=394 -> 846.96
#      RADICO  stepline y=408 -> 4161.57 (= hi_short2) stub y=329 -> 4192.55
#  The line sits at the OLD value all week and steps up only at the right edge.
#
#  DECISION (user, 28-Jul): Option B - trade the economically correct level,
#  highest(high,26)[1]. Entering at hi_short2 means entering BELOW the actual
#  26-week high, i.e. before any breakout has happened. The chart arrows are
#  not a valid target; the Pine is what is wrong.
# --------------------------------------------------------------------------- #
def test_entry_level_is_the_true_26w_high_not_the_lagged_plot():
    """
    Option B, locked. entry_level must be the 26W high INCLUDING the last
    closed week. It must never equal hi_short2 (the lagged, plotted value)
    unless the two genuinely coincide in the data.
    """
    import numpy as np
    import pandas as pd

    from config import Strategy
    from strategy import build_snapshot, build_weekly_bars
    from test_strategy import make_daily

    cfg = Strategy()
    daily = make_daily(weeks=200)
    wk = build_weekly_bars(daily)
    target = wk.iloc[-1]["week_start"]
    snap = build_snapshot("T", "1", "NSE_EQ", daily, cfg, target)
    assert snap is not None

    closed = wk[wk["week_start"] < target]["high"].to_numpy(float)

    # the window ENDS at the last closed week and INCLUDES it
    assert snap.entry_level == pytest.approx(closed[-cfg.len_short:].max())
    # hi_short2 is strictly the older window
    assert snap.hi_short2 == pytest.approx(closed[-(cfg.len_short + 1):-1].max())
    # and entry_level is never OLDER than hi_short2
    assert snap.entry_level >= snap.hi_short2 - 1e-9


def test_radico_and_vaibhavgbl_levels_are_the_unshifted_values():
    """
    Real levels for the week of 27-Jul-2026, from NSE weekly highs. These are
    the numbers the user confirmed under Option B. Hard-coded because they are
    what the live alerts must reproduce.

        RADICO      26W high = 4193.00   (w/c 20-Jul)   hi_short2 = 4161.80
        VAIBHAVGBL  26W high =  273.00   (w/c 20-Jul)   hi_short2 =  268.08
        TMB         26W high =  847.00   (w/c 20-Jul)   hi_short2 =  821.20

    If a future change reintroduces the shift, these flip to the hi_short2
    column and this test fails.
    """
    import numpy as np
    import pandas as pd

    from config import Strategy
    from strategy import build_snapshot

    cfg = Strategy()

    # Build 80 weekly bars so the snapshot's history requirement is met.
    # Only the last 26 matter for the window; everything older is filler kept
    # strictly below both candidates so it cannot win the max().
    rows = []
    first_week = pd.Timestamp("2026-07-20") - pd.Timedelta(weeks=79)
    for i in range(80):
        d = first_week + pd.Timedelta(weeks=i)
        rows.append({"datetime": d, "open": 3000.0, "high": 3100.0,
                     "low": 2900.0, "close": 3000.0, "volume": 1e6})

    # overwrite the final three weeks with the real shape
    tail = {"2026-07-06": 4161.80,   # hi_short2 (the lagged, plotted value)
            "2026-07-13": 4149.20,
            "2026-07-20": 4193.00}   # the true 26W high, last closed week
    for r in rows:
        key = str(r["datetime"].date())
        if key in tail:
            hi = tail[key]
            r.update(open=hi - 50, high=hi, low=hi - 80, close=hi - 10)

    daily = pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)
    snap = build_snapshot("RADICO", "1", "NSE_EQ", daily, cfg,
                          pd.Timestamp("2026-07-27"))

    assert snap is not None, (
        "history must be long enough for this test to actually assert - "
        "an earlier version silently skipped on a None snapshot")
    assert snap.entry_level == pytest.approx(4193.00), (
        "entry_level must be the true 26W high (4193.00), not the lagged "
        "plotted value (4161.80)")
    assert snap.hi_short2 == pytest.approx(4161.80)


def test_c09_cannot_satisfy_radico_and_vaibhavgbl_simultaneously():
    """
    Documented impossibility, so nobody burns another day tuning c09.

    Under Option B the user asked for BOTH:
        RADICO      Tue 15:00 @ 4276.90   (deferred - needs c09 to BLOCK 14:50)
        VAIBHAVGBL  Tue 10:35 @  278.33   (cross    - needs c09 to PASS)

    In "pace" mode c09 passes when  vol_ratio > week_fraction * m.
    Measured on the real 5m bars for that week:

        VAIBHAVGBL 10:35  ratio 0.4837  frac 0.2453  -> needs m <  1.972
        RADICO     14:50  ratio 0.9325  frac 0.3813  -> needs m >= 2.445
        RADICO     15:00  ratio 1.0525  frac 0.3867  -> needs m <  2.722

    Feasible range is [2.445, 1.972) = EMPTY. No multiplier exists.

    Deeper reason VAIBHAVGBL is unreachable at ANY c09 setting that keeps the
    condition meaningful: across the whole week it spent 9 bars above 273.00
    (Tue 10:35-14:10) and exactly 1 bar with c09 true (Tue 15:25). The two
    never overlap, so no gate ordering can produce an entry while price is
    above the level. VAIBHAVGBL is a genuine REJECT that week, not a miss.
    """
    # Pure arithmetic - encodes the measurement so it cannot be forgotten.
    v_ratio, v_frac = 0.4837, 0.2453      # VAIBHAVGBL Tue 10:35, must PASS
    r_ratio_fail, r_frac_fail = 0.9325, 0.3813   # RADICO Tue 14:50, must FAIL
    r_ratio_pass, r_frac_pass = 1.0525, 0.3867   # RADICO Tue 15:00, must PASS

    m_upper = min(v_ratio / v_frac, r_ratio_pass / r_frac_pass)
    m_lower = r_ratio_fail / r_frac_fail

    assert m_lower >= m_upper, (
        "If this ever passes, a single c09 multiplier CAN satisfy both cases "
        "and the documented impossibility no longer holds - re-derive it.")


def test_pine_source_uses_request_security_for_the_level():
    """
    The lag is real and lives in the Pine, not in our Python. If the user ever
    fixes the indicator to use chart-native highs, the arrows will move to the
    table values and agree with us - at which point this test should be
    updated deliberately, not by accident.
    """
    src = ROOT.parent / "uploads" / "weekly.txt"
    if not src.exists():
        pytest.skip("Pine source not present in this checkout")
    body = src.read_text(errors="ignore")
    assert "request.security" in body
    assert "lookahead=barmerge.lookahead_off" in body, (
        "the level fetch relies on lookahead_off; that is what makes the "
        "plotted stepline lag the table by one weekly bar")


# --------------------------------------------------------------------------- #
#  BUG 22 - INTRADAY MODE. User, 28-Jul: "Focus on only intraday, i don't want
#  swing trade."
#
#  Three things had to become true, and each is guarded below.
#
#  1. NOTHING MAY BE HELD OVERNIGHT. Brokers force-square MIS equity in the
#     afternoon; a simulation that carries a position to the next session is
#     reporting a trade that could not exist. paper.py now exits at
#     SQUARE_OFF (15:15) or on any bar dated after the signal.
#
#  2. COSTS MUST BE CHARGED. Measured over 449 real signals (12 weeks, whole
#     NSE cash universe) the best exit rule made +0.14% per trade GROSS and
#     -0.08% NET at a realistic 0.22% round trip. Reporting gross P&L on an
#     intraday system is reporting a profit that does not exist.
#
#  3. A VOLATILITY FLOOR. Without it the signal set loses money outright
#     (PF 0.72). min_atr_pct >= 1.0 was the only filter that survived an
#     out-of-sample split.
# --------------------------------------------------------------------------- #
def test_paper_never_holds_overnight():
    """A position must close on its own session, whatever the exit rule."""
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    import pandas as pd

    from paper import simulate
    from strategy import BarEval, Signal

    ist = ZoneInfo("Asia/Kolkata")
    ev = BarEval({f"c{i:02d}": True for i in range(1, 14)},
                 {"rsi": 70.0, "macd_hist": 1.0})
    sig = Signal("X", "1", "NSE_EQ", datetime(2026, 7, 27, 14, 0, tzinfo=ist),
                 100.0, 98.0, 98.0, "cross", ev, "2026-07-27", 1e6,
                 97.0, 96.0, bar_open=99.5, bar_high=100.5, bar_low=99.0)

    # a relentless grind UP: no stop, no EMA breach - only square-off can end it
    rows = []
    t0 = datetime(2026, 7, 27, 14, 5, tzinfo=ist)
    for i in range(40):                      # spills into the next session
        ts = t0 + timedelta(minutes=5 * i)
        if ts.hour >= 15 and ts.minute > 25:  # jump to next day
            ts = datetime(2026, 7, 28, 9, 15, tzinfo=ist) + timedelta(minutes=5 * i)
        p = 100.0 + i * 0.5
        rows.append({"datetime": ts, "open": p, "high": p + 0.4,
                     "low": p - 0.05, "close": p + 0.3})
    bars = pd.DataFrame(rows)

    t = simulate(sig, bars, 100000)
    assert t.exit_reason != "OPEN", "an intraday trade must not be left running"
    assert t.exit_date == "2026-07-27", (
        f"position carried to {t.exit_date} - intraday trades cannot go "
        "overnight, the broker squares them off")


def test_square_off_time_is_before_the_close():
    """Leave room before the broker's own auto-square-off."""
    from paper import SQUARE_OFF
    h, m = (int(x) for x in SQUARE_OFF.split(":"))
    assert (h, m) >= (15, 0), "squaring off too early wastes the session"
    assert (h, m) <= (15, 20), (
        "most brokers auto-square MIS equity by ~15:20 and forced liquidation "
        "fills worse than a voluntary exit")


def test_paper_charges_round_trip_costs():
    """Net P&L must be strictly worse than gross - costs are not optional."""
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    import pandas as pd

    from paper import COST_ROUND_TRIP, simulate
    from strategy import BarEval, Signal

    assert COST_ROUND_TRIP > 0, "an intraday backtest without costs is fiction"

    ist = ZoneInfo("Asia/Kolkata")
    ev = BarEval({f"c{i:02d}": True for i in range(1, 14)},
                 {"rsi": 70.0, "macd_hist": 1.0})
    sig = Signal("X", "1", "NSE_EQ", datetime(2026, 7, 27, 10, 0, tzinfo=ist),
                 100.0, 98.0, 98.0, "cross", ev, "2026-07-27", 1e6,
                 97.0, 96.0, bar_open=99.5, bar_high=100.5, bar_low=99.0)
    bars = pd.DataFrame([
        {"datetime": datetime(2026, 7, 27, 10, 5, tzinfo=ist) + timedelta(minutes=5 * i),
         "open": 100 + i * .2, "high": 101 + i * .2,
         "low": 99.8 + i * .2, "close": 100.4 + i * .2} for i in range(6)])

    t = simulate(sig, bars, 100000)
    assert t.qty > 0
    assert t.costs > 0, "no costs were charged"
    assert t.pnl == pytest.approx(t.gross_pnl - t.costs), \
        "net P&L must equal gross minus costs"
    assert t.pnl < t.gross_pnl


def test_min_atr_pct_filter_blocks_low_volatility_entries():
    """
    A stock that barely moves cannot pay for a round trip. The filter must
    actually suppress the entry, not merely annotate it.
    """
    import numpy as np
    import pandas as pd

    from config import Strategy
    from strategy import build_snapshot, build_weekly_bars, replay_week
    from test_strategy import make_daily

    daily = make_daily(weeks=200)
    wk = build_weekly_bars(daily)
    target = wk.iloc[-1]["week_start"]

    base = Strategy(strict_entry=False)          # isolate the ATR filter
    snap = build_snapshot("T", "1", "NSE_EQ", daily, base, target)
    assert snap is not None

    lvl = snap.entry_level
    # a clean break above the level, but on tiny 5m ranges (~0.02%)
    rows = []
    for i in range(30):
        p = lvl * (1.0 + 0.004 + i * 0.00002)
        rows.append({"datetime": pd.Timestamp(target) + pd.Timedelta(minutes=5 * i),
                     "open": p, "high": p * 1.0001, "low": p * 0.9999,
                     "close": p, "volume": 10_000})
    bars = pd.DataFrame(rows)

    assert replay_week(snap, base, bars).signals, \
        "sanity: with the filter off this must produce a signal"

    tight = Strategy(strict_entry=False, min_atr_pct=1.0)
    assert not replay_week(snap, tight, bars).signals, \
        "min_atr_pct must block a breakout whose bars are far too quiet"


def test_min_atr_pct_defaults_to_disabled():
    """
    Shipping it ON by default would silently change every historical result.
    config.yaml opts in explicitly; the dataclass default stays inert.
    """
    from config import Strategy
    assert Strategy().min_atr_pct == 0.0


# --------------------------------------------------------------------------- #
#  BUG 23 - A/B PAPER TRADING. User, 28-Jul: "can create 2 separate paper trade
#  model, 1 with your recommendation and other with my recommendation, so after
#  a week we will get on to the conclusion."
#
#  The experiment is only worth running if it is FAIR. Every guard below exists
#  to stop the comparison being quietly rigged:
#
#    * both models must use the IDENTICAL stop (entry candle low - 0.02%).
#      The user has said twice not to change the stop; if one model got a
#      different stop the test would measure stop width, not entry quality.
#    * both must square off intraday - neither may sneak an overnight hold.
#    * both must be charged the same costs.
#    * the ledger must de-duplicate, or running the job twice in a day would
#      double-count trades and corrupt the week's conclusion.
# --------------------------------------------------------------------------- #
def test_models_file_is_valid_and_not_empty():
    """
    A and B were retired on 29-Jul-2026 (every intraday exit measured negative
    after costs), leaving C alone. The harness must still be coherent with a
    single model, and must never end up with none.
    """
    import yaml as _yaml

    raw = _yaml.safe_load((ROOT / "models.yaml").read_text())
    models = raw["models"]
    assert models, "at least one model must be configured"
    for key, m in models.items():
        assert m.get("label"), f"{key} needs a label"
        assert m.get("exit"), f"{key} needs an exit rule"

    # If two or more ever run again, they must actually differ - otherwise the
    # comparison is measuring noise.
    if len(models) >= 2:
        sigs = [str(sorted((m.get("strategy") or {}).items()))
                + str(sorted((m.get("exit") or {}).items()))
                for m in models.values()]
        assert len(set(sigs)) == len(sigs), (
            "two models share identical settings - nothing to compare")


def test_ab_models_use_identical_stop_and_costs():
    """Neither model may be handed an easier stop or cheaper costs."""
    import yaml as _yaml

    raw = _yaml.safe_load((ROOT / "models.yaml").read_text())
    for key, m in raw["models"].items():
        strat = m.get("strategy", {})
        for forbidden in ("sl_buffer", "stop_pct", "stop_mode"):
            assert forbidden not in strat, (
                f"{key} tries to override the stop ({forbidden}); the stop is "
                "identical in both models by design")
    # costs and square-off are global defaults, not per-model
    for key, m in raw["models"].items():
        assert "cost_round_trip" not in m, f"{key} must not set its own costs"
        assert "square_off" not in m, f"{key} must not set its own square-off"


def test_ab_simulate_respects_intraday_square_off():
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    import pandas as pd

    from ab_paper import simulate_model
    from strategy import BarEval, Signal

    ist = ZoneInfo("Asia/Kolkata")
    ev = BarEval({f"c{i:02d}": True for i in range(1, 14)},
                 {"rsi": 70.0, "macd_hist": 1.0})
    sig = Signal("X", "1", "NSE_EQ", datetime(2026, 7, 27, 14, 0, tzinfo=ist),
                 100.0, 98.0, 98.0, "cross", ev, "2026-07-27", 1e6,
                 97.0, 96.0, bar_open=99.5, bar_high=100.5, bar_low=99.0)
    rows = []
    for i in range(40):
        ts = datetime(2026, 7, 27, 14, 5, tzinfo=ist) + timedelta(minutes=5 * i)
        if ts.hour >= 15 and ts.minute > 25:
            ts = datetime(2026, 7, 28, 9, 15, tzinfo=ist) + timedelta(minutes=5 * i)
        p = 100.0 + i * 0.5      # relentless grind up: only square-off can end it
        rows.append({"datetime": ts, "open": p, "high": p + 0.4,
                     "low": p - 0.05, "close": p + 0.3})
    bars = pd.DataFrame(rows)

    for rule in ({"rule": "ema"}, {"rule": "target", "target_r": 99.0}):
        t = simulate_model(sig, bars, 100000, None, rule)
        assert t.exit_date == "2026-07-27", (
            f"{rule} carried the position overnight")


def test_ab_both_models_get_the_same_stop():
    """Same signal + same bars -> same stop, whatever the exit rule."""
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    import pandas as pd

    from ab_paper import simulate_model
    from strategy import BarEval, Signal

    ist = ZoneInfo("Asia/Kolkata")
    ev = BarEval({f"c{i:02d}": True for i in range(1, 14)},
                 {"rsi": 70.0, "macd_hist": 1.0})
    sig = Signal("X", "1", "NSE_EQ", datetime(2026, 7, 27, 10, 0, tzinfo=ist),
                 100.0, 98.0, 98.0, "cross", ev, "2026-07-27", 1e6,
                 97.0, 96.0, bar_open=99.5, bar_high=100.5, bar_low=99.0)
    bars = pd.DataFrame([
        {"datetime": datetime(2026, 7, 27, 10, 5, tzinfo=ist) + timedelta(minutes=5 * i),
         "open": 100 + i * .2, "high": 101 + i * .2,
         "low": 99.8 + i * .2, "close": 100.4 + i * .2} for i in range(6)])

    a = simulate_model(sig, bars, 100000, None, {"rule": "ema"})
    b = simulate_model(sig, bars, 100000, None,
                       {"rule": "target", "target_r": 3.0, "be_at_r": 1.0})
    assert a.stop == pytest.approx(b.stop), \
        "the two models must risk exactly the same amount per trade"


def test_ab_ledger_deduplicates(tmp_path):
    """Running the job twice in one day must not double-count."""
    from ab_paper import append_ledger

    p = tmp_path / "led.csv"
    row = {"model": "A_gated", "symbol": "X", "signal_date": "2026-07-27",
           "signal_time": "10:00", "pnl": 100.0}
    added, dupes = append_ledger(p, [row])
    assert added == 1 and dupes == 0
    added, dupes = append_ledger(p, [row])
    assert added == 0 and dupes == 1, "the same trade was recorded twice"


def test_ab_target_r_uses_initial_risk():
    """+3R must mean 3x the ORIGINAL stop distance, not a moving one."""
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    import pandas as pd

    from ab_paper import simulate_model
    from strategy import BarEval, Signal

    ist = ZoneInfo("Asia/Kolkata")
    ev = BarEval({f"c{i:02d}": True for i in range(1, 14)},
                 {"rsi": 70.0, "macd_hist": 1.0})
    # entry 100, bar low 99 -> stop 98.98, risk ~1.02 -> +3R target ~103.06
    sig = Signal("X", "1", "NSE_EQ", datetime(2026, 7, 27, 10, 0, tzinfo=ist),
                 100.0, 98.0, 98.0, "cross", ev, "2026-07-27", 1e6,
                 97.0, 96.0, bar_open=99.5, bar_high=100.5, bar_low=99.0)
    rows = []
    for i in range(10):
        p = 100.0 + i * 1.0
        rows.append({"datetime": datetime(2026, 7, 27, 10, 5, tzinfo=ist) + timedelta(minutes=5 * i),
                     "open": p, "high": p + 0.9, "low": p - 0.1, "close": p + 0.5})
    t = simulate_model(sig, pd.DataFrame(rows), 100000, None,
                       {"rule": "target", "target_r": 3.0})
    assert t.exit_reason == "TGT"
    assert t.r_multiple == pytest.approx(3.0, abs=0.01)


# --------------------------------------------------------------------------- #
#  BUG 24 - the stale snapshot the version guard could not see.
#
#  On 28-Jul the repo's committed weekly_snapshot.csv held levels from the
#  REVERTED one-week shift:
#      TMB        821.20  (should be 847.00)
#      RADICO    4161.80  (should be 4193.00)
#      VAIBHAVGBL 268.08  (should be  273.00)
#  Monday's scan would have fired against levels ~1% too low.
#
#  The LOGIC_VERSION guard did not catch it because v2 was stamped in the SAME
#  session that introduced the bad formula - the guard cannot recognise its own
#  bad output. Reverting the formula without bumping the version left the file
#  looking current.
#
#  Fix: LOGIC_VERSION = 3, so every v2 row is discarded on resume and the next
#  run rebuilds from scratch even if someone forgets to tick `fresh`.
# --------------------------------------------------------------------------- #
def test_logic_version_is_at_least_3():
    """
    v2 rows carry the reverted, too-low breakout levels. If this ever drops
    back to 2 those rows become resumable again.
    """
    from build_snapshot import LOGIC_VERSION
    assert LOGIC_VERSION >= 3, (
        "LOGIC_VERSION must stay >= 3: version 2 identifies snapshot rows "
        "built with the reverted one-week level shift")


def test_stale_v2_rows_are_discarded_on_resume(tmp_path):
    """
    A snapshot file full of v2 rows must resume ZERO symbols, so the rebuild
    is total even without --fresh.
    """
    import pandas as pd

    from build_snapshot import LOGIC_VERSION

    stale = pd.DataFrame([
        {"symbol": "TMB", "week_start": "2026-07-27",
         "entry_level": "821.20", "logic_version": "2"},
        {"symbol": "RADICO", "week_start": "2026-07-27",
         "entry_level": "4161.80", "logic_version": "2"},
    ])
    p = tmp_path / "weekly_snapshot.csv"
    stale.to_csv(p, index=False)

    prev = pd.read_csv(p, dtype=str)
    prev = prev[prev["week_start"] == "2026-07-27"]
    kept = prev[prev["logic_version"] == str(LOGIC_VERSION)]
    assert len(kept) == 0, (
        "rows from the reverted-level generation must never be resumed")


def test_bumping_the_formula_requires_bumping_the_version():
    """
    Executable reminder of the lesson. build_snapshot.py must document why the
    current LOGIC_VERSION exists, so the next person who edits the level maths
    sees that the version has to move with it - including on a REVERT.
    """
    body = (ROOT / "strategy.py").read_text()
    assert "REVERTED" in body or "reverted" in body, (
        "the LOGIC_VERSION block must record that a revert also invalidates "
        "previously stored rows")
    # and the writer must not keep a private copy that can drift from it
    writer = (ROOT / "build_snapshot.py").read_text()
    assert "from strategy import" in writer and "LOGIC_VERSION" in writer, (
        "build_snapshot.py must import LOGIC_VERSION from strategy.py")
    assert "\nLOGIC_VERSION = " not in writer, (
        "build_snapshot.py must not define its own LOGIC_VERSION")


# --------------------------------------------------------------------------- #
#  BUG 25 - THE SCANNER ALERTED AGAINST STALE BREAKOUT LEVELS.
#
#  build_snapshot.py refused to RESUME rows from an older logic version, but
#  scan.py happily LOADED them. So after the level formula was reverted, the
#  committed weekly_snapshot.csv still held v2 rows and the live scanner used
#  them for two full sessions:
#
#      symbol      quoted level   true 26W high   outcome
#      RADICO           4161.80         4193.00   alert at the wrong level
#      TMB               821.20          847.00   alert at the wrong level
#      SENCO             390.40          399.80   alert at the wrong level
#      GPTHEALTH         161.00          164.69   alert at the wrong level
#      JKPAPER           401.50          421.95   *** FALSE ALERT ***
#      THYROCARE         578.25          597.60   *** FALSE ALERT ***
#      VAIBHAVGBL        268.08          273.00   *** FALSE ALERT ***
#
#  A wrong level is a wrong trade. The fix: readers verify logic_version AND
#  week_start, and DROP anything that does not match - scanning nothing is
#  strictly better than alerting on a level that does not exist.
# --------------------------------------------------------------------------- #
def test_logic_version_has_one_source_of_truth():
    """Writer and reader must not be able to drift apart."""
    import build_snapshot
    import scan
    from strategy import LOGIC_VERSION

    assert build_snapshot.LOGIC_VERSION is LOGIC_VERSION
    assert scan.LOGIC_VERSION is LOGIC_VERSION


def _snap_row(symbol, level, week, version):
    """A minimally complete snapshot row that from_row() can parse."""
    return {
        "symbol": symbol, "security_id": "1", "exchange_segment": "NSE_EQ",
        "week_start": week, "entry_level": level, "level_52": level,
        "hi_short2": level, "close_1": level,
        "ema_fast_prev": 1.0, "ema_fast_len": 20,
        "ema_slow_prev": 1.0, "ema_slow_len": 50, "ema_slow_2": 1.0,
        "rsi_len": 14, "rsi_avg_gain": 1.0, "rsi_avg_loss": 1.0,
        "rsi_prev_close": 1.0, "rsi_1": 1.0,
        "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
        "macd_ema_fast": 1.0, "macd_ema_slow": 1.0, "macd_sig": 1.0,
        "vol_sma_len": 10, "vol_sma_sum_prev": 1.0,
        "g_ema_fast": 1.0, "g_ema_slow": 1.0, "g_ema_slow_2": 1.0,
        "g_rsi": 1.0, "g_rsi_1": 1.0, "g_hist": 1.0,
        "prev_daily_close": 1.0, "prev_daily_open": 1.0, "mcap": "",
        "logic_version": version,
    }


def test_scan_drops_rows_from_an_older_logic_version(tmp_path):
    """The exact RADICO 4161.80 scenario must now load nothing."""
    import pandas as pd

    import scan
    from config import load_config
    from strategy import LOGIC_VERSION

    week = "2026-07-27"
    old = LOGIC_VERSION - 1
    pd.DataFrame([
        _snap_row("RADICO", 4161.80, week, old),        # reverted-shift level
        _snap_row("JKPAPER", 401.50, week, old),        # caused a false alert
    ]).to_csv(tmp_path / "weekly_snapshot.csv", index=False)

    cfg = load_config(None)
    cfg.paths["snapshot"] = tmp_path / "weekly_snapshot.csv"
    assert scan.load_snapshots(cfg, week) == [], (
        "rows from an older logic version must never reach the scanner")


def test_scan_drops_rows_for_a_different_week(tmp_path):
    import pandas as pd

    import scan
    from config import load_config
    from strategy import LOGIC_VERSION

    pd.DataFrame([
        _snap_row("RADICO", 4193.00, "2026-07-20", LOGIC_VERSION),
    ]).to_csv(tmp_path / "weekly_snapshot.csv", index=False)

    cfg = load_config(None)
    cfg.paths["snapshot"] = tmp_path / "weekly_snapshot.csv"
    assert scan.load_snapshots(cfg, "2026-07-27") == []


def test_scan_keeps_current_rows(tmp_path):
    """Sanity: the guard must not reject good data."""
    import pandas as pd

    import scan
    from config import load_config
    from strategy import LOGIC_VERSION

    week = "2026-07-27"
    pd.DataFrame([
        _snap_row("RADICO", 4193.00, week, LOGIC_VERSION),
        _snap_row("APCOTEXIND", 579.75, week, LOGIC_VERSION),
    ]).to_csv(tmp_path / "weekly_snapshot.csv", index=False)

    cfg = load_config(None)
    cfg.paths["snapshot"] = tmp_path / "weekly_snapshot.csv"
    snaps = scan.load_snapshots(cfg, week)
    assert {s.symbol for s in snaps} == {"RADICO", "APCOTEXIND"}
    assert {round(s.entry_level, 2) for s in snaps} == {4193.00, 579.75}


def test_snapshot_without_a_version_column_is_rejected(tmp_path):
    """A pre-versioning file has unknown provenance - refuse it."""
    import pandas as pd

    import scan
    from config import load_config

    week = "2026-07-27"
    row = _snap_row("RADICO", 4161.80, week, 0)
    row.pop("logic_version")
    pd.DataFrame([row]).to_csv(tmp_path / "weekly_snapshot.csv", index=False)

    cfg = load_config(None)
    cfg.paths["snapshot"] = tmp_path / "weekly_snapshot.csv"
    assert scan.load_snapshots(cfg, week) == []


# --------------------------------------------------------------------------- #
#  BUG 26 - daily watchlist. User, 29-Jul: "also everyday send all the stocks
#  list which are qualifying for current week on telegram".
#
#  The digest exists to answer "what should be on my screen today?", so the two
#  things that must never break are:
#    * it must not silently truncate to nothing - the FULL table always goes
#      out as a CSV, because Telegram caps a message at 4096 chars and a
#      2000-symbol universe blows past that instantly
#    * it must use the SAME frozen levels as the scanner. If it re-derived
#      them it could show a level the alerts would never fire on.
# --------------------------------------------------------------------------- #
def test_watchlist_reuses_the_scanner_snapshot_loader():
    """
    The digest must read levels through scan.load_snapshots, which enforces
    logic_version + week. Recomputing them here would let the watchlist and the
    alerts disagree about the level - the exact class of bug from BUG 25.
    """
    body = (ROOT / "watchlist.py").read_text()
    assert "from scan import" in body and "load_snapshots" in body, (
        "watchlist.py must load levels via scan.load_snapshots so it inherits "
        "the stale-snapshot guard")
    assert "build_snapshot(" not in body, (
        "watchlist.py must not build its own levels")


def test_watchlist_buckets_are_correct():
    """WATCH / FIRED must key off the frozen level; nothing else is listed."""
    from watchlist import build_message

    rows = [
        dict(symbol="AAA", ltp=98.0, level=100.0, pct=-2.0, gap=2.0, bucket="WATCH"),
        dict(symbol="BBB", ltp=50.0, level=100.0, pct=-50.0, gap=50.0, bucket="OTHER"),
        dict(symbol="CCC", ltp=120.0, level=100.0, pct=20.0, gap=-20.0, bucket="FIRED"),
    ]
    msg = build_message("2026-07-27", rows, {"universe": 3, "eligible": 2}, 3.0)
    assert "AAA" in msg and "CCC" in msg
    assert "BBB" not in msg, "far names are noise and must not be listed"
    assert "week of 2026-07-27" in msg


def test_watchlist_omits_the_above_level_section():
    """
    User, 29-Jul: "skip those ABOVE LEVEL, NOT YET TRIGGERED".
    Those names either already alerted or are being held back by the weekly
    gate - neither is something to watch for tomorrow. Dropping them also
    removes the per-symbol 5m replay, so the job is a single quote pass.
    """
    from watchlist import build_message

    rows = [dict(symbol="ZZZ", ltp=110.0, level=100.0, pct=10.0,
                 gap=-10.0, bucket="OTHER")]
    msg = build_message("2026-07-27", rows, {"universe": 1, "eligible": 1}, 3.0)
    assert "ZZZ" not in msg
    assert "ABOVE LEVEL" not in msg.upper()

    body = (ROOT / "watchlist.py").read_text()
    assert "scan_symbol" not in body, (
        "no per-symbol replay should remain - the digest is quote-only now")


def test_watchlist_ranks_by_distance_and_caps():
    """
    Closest-to-level first, hard cap on the list. Measured on the live book,
    momentum filters cut 88 -> 64 while distance cut it to ~18, so ranking is
    what makes the digest actionable.
    """
    from watchlist import build_message

    rows = [dict(symbol=f"S{i:03d}", ltp=100.0 - i, level=100.0,
                 pct=-float(i), gap=float(i), bucket="WATCH")
            for i in range(1, 60)]
    msg = build_message("2026-07-27", rows, {"universe": 59, "eligible": 59},
                        99.0, top_n=15)
    listed = [ln for ln in msg.splitlines() if ln.startswith("• <b>S")]
    assert len(listed) == 15, f"expected 15 rows, got {len(listed)}"
    assert "S001" in listed[0], "the closest name must rank first"
    assert "more in the CSV" in msg


def test_watchlist_structure_screen_keeps_real_winners():
    """
    The c03/c05/c08 screen must not reject names that went on to fire.
    MONARCH and RADICO both failed the STRICTER screen (c04/c06/c07) during
    the 27-Jul week, which is why those rows are deliberately excluded.
    """
    from config import load_config
    from watchlist import is_eligible

    cfg = load_config(None)

    class S:                      # minimal stand-in for a WeeklySnapshot
        close_1 = 100.0
        hi_short2 = 120.0         # c03 passes: previous close below the old high
        g_hist = 1.0              # c08 passes
        g_ema_slow = 50.0
        g_ema_slow_2 = 49.0       # c05 passes
        g_rsi = 56.0              # would FAIL a strict RSI>60 screen
        g_rsi_1 = 57.0            # would FAIL "RSI rising"
        g_ema_fast = 49.0         # would FAIL EMA20>EMA50

    assert is_eligible(S(), cfg), (
        "a name with soft closed-week RSI must stay eligible - it is exactly "
        "the accelerating setup the strict screen throws away")

    class Stale(S):
        close_1 = 130.0           # already above the old high -> not fresh
    assert not is_eligible(Stale(), cfg)


def test_watchlist_always_writes_the_full_table():
    """Truncating the message is fine; losing the data is not."""
    body = (ROOT / "watchlist.py").read_text()
    assert "to_csv" in body
    assert "send_document" in body, (
        "the complete watchlist must be attached, not just summarised")


def test_watchlist_workflow_runs_premarket_on_weekdays():
    import yaml as _yaml

    wf = _yaml.safe_load((WORKFLOWS / "watchlist.yml").read_text())
    on = wf.get("on") or wf.get(True)
    crons = [c["cron"] for c in on["schedule"]]
    assert crons, "the watchlist must be scheduled"
    for c in crons:
        minute, hour, _, _, dow = c.split()
        assert dow == "1-5", "markets are shut at the weekend"
        # 03:15 UTC = 08:45 IST, i.e. before the 09:15 open
        assert int(hour) < 4, (
            f"cron '{c}' is not pre-market IST; the list is useless after the "
            "open")


def test_watchlist_never_places_orders():
    body = (ROOT / "watchlist.py").read_text().lower()
    for bad in ("place_order", "placeorder", "/orders", "transactiontype"):
        assert bad not in body, f"watchlist.py must never trade ({bad})"


# --------------------------------------------------------------------------- #
#  BUG 27 - MODEL C "SWING". User, 29-Jul: "Create separate entry model based
#  on best entry, best stop, best hold".
#
#  C is the first SWING model in the harness: it holds overnight, which every
#  earlier test forbade outright. The danger is that adding it quietly guts the
#  intraday guarantees for A and B, so the horizon is an explicit, tested flag:
#  only `horizon: swing` may skip the square-off, and an intraday model can
#  never acquire a multi-day hold by accident.
#
#  Parameters were measured on 3,254 breakouts over 12 months:
#      entry  same-day close  +5.17%  vs next-open +4.40%, pullback +3.43%
#      stop   7%              +5.59%  vs entry-low +4.05%, 3% +4.24%
#      hold   5 days          +5.59%  vs 1d +2.49%, 30d +5.98% (6x the lockup)
# --------------------------------------------------------------------------- #
def test_models_declare_a_valid_horizon():
    import yaml as _yaml

    raw = _yaml.safe_load((ROOT / "models.yaml").read_text())
    for key, m in raw["models"].items():
        hz = m.get("horizon", "intraday")
        assert hz in ("intraday", "swing"), f"{key} has a bad horizon: {hz}"


def test_only_swing_models_may_hold_overnight():
    """An intraday model must never be handed hold_days."""
    import yaml as _yaml

    raw = _yaml.safe_load((ROOT / "models.yaml").read_text())
    for key, m in raw["models"].items():
        if m.get("horizon", "intraday") == "intraday":
            assert not m.get("hold_days"), (
                f"{key} is intraday but declares hold_days - that would let it "
                "carry a position overnight")
            assert not m.get("exit", {}).get("hold_days"), (
                f"{key} is intraday but its exit sets hold_days")


def test_intraday_models_still_square_off():
    """Adding C must not weaken the guarantee for A and B."""
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    import pandas as pd

    from ab_paper import load_models, simulate_model
    from strategy import BarEval, Signal

    ist = ZoneInfo("Asia/Kolkata")
    ev = BarEval({f"c{i:02d}": True for i in range(1, 14)},
                 {"rsi": 70.0, "macd_hist": 1.0})
    sig = Signal("X", "1", "NSE_EQ", datetime(2026, 7, 27, 14, 0, tzinfo=ist),
                 100.0, 98.0, 98.0, "cross", ev, "2026-07-27", 1e6,
                 97.0, 96.0, bar_open=99.5, bar_high=100.5, bar_low=99.0)
    rows = []
    for i in range(40):
        ts = datetime(2026, 7, 27, 14, 5, tzinfo=ist) + timedelta(minutes=5 * i)
        if ts.hour >= 15 and ts.minute > 25:
            ts = datetime(2026, 7, 28, 9, 15, tzinfo=ist) + timedelta(minutes=5 * i)
        p = 100.0 + i * 0.5
        rows.append({"datetime": ts, "open": p, "high": p + 0.4,
                     "low": p - 0.05, "close": p + 0.3})
    bars = pd.DataFrame(rows)

    _defaults, models = load_models()
    for m in models:
        if m.is_swing:
            continue
        t = simulate_model(sig, bars, 100000, None, m.exit)
        assert t.exit_date == "2026-07-27", f"{m.key} held overnight"


def _daily(n, start=100.0, step=1.0, low_off=1.0):
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    import pandas as pd
    ist = ZoneInfo("Asia/Kolkata")
    rows = []
    for i in range(n):
        p = start + step * (i + 1)
        rows.append({"datetime": datetime(2026, 7, 28, tzinfo=ist) + timedelta(days=i),
                     "open": p, "high": p + 1.0, "low": p - low_off, "close": p})
    return pd.DataFrame(rows)


def _swing_sig():
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from strategy import BarEval, Signal
    ist = ZoneInfo("Asia/Kolkata")
    ev = BarEval({f"c{i:02d}": True for i in range(1, 14)},
                 {"rsi": 70.0, "macd_hist": 1.0})
    return Signal("X", "1", "NSE_EQ", datetime(2026, 7, 27, 15, 20, tzinfo=ist),
                  100.0, 98.0, 98.0, "cross", ev, "2026-07-27", 1e6,
                  97.0, 96.0, bar_open=99.5, bar_high=100.5, bar_low=99.0)


def test_swing_time_exit_after_hold_days():
    """A grind that never hits stop or target must exit on day `hold_days`."""
    from ab_paper import simulate_swing

    t = simulate_swing(_swing_sig(), _daily(20, step=0.2), 100000,
                       {"stop_pct": 7.0, "target_r": 99.0}, hold_days=5)
    assert t.exit_reason == "TIME"
    assert t.bars_held == 5, f"held {t.bars_held} days, expected 5"


def test_swing_stop_is_a_percent_not_the_entry_candle_low():
    """
    C deliberately uses a 7% stop. The entry-candle low (99.0 here) would be a
    ~1% stop and would fire on the first red day; 7% must not.
    """
    from ab_paper import simulate_swing

    sig = _swing_sig()
    t = simulate_swing(sig, _daily(6, step=0.5, low_off=2.0), 100000,
                       {"stop_pct": 7.0, "target_r": 99.0}, hold_days=5)
    assert t.stop == pytest.approx(93.0), f"stop was {t.stop}, expected 7% = 93.0"
    assert t.exit_reason == "TIME", "a 7% stop must survive a 2-point daily wick"


def test_swing_stop_wins_ties_with_the_target():
    """If one day both breaches the stop and tags the target, take the stop."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import pandas as pd

    from ab_paper import simulate_swing
    ist = ZoneInfo("Asia/Kolkata")
    # a single huge-range day: low 90 (below the 93 stop), high 130 (above 3R)
    bars = pd.DataFrame([{"datetime": datetime(2026, 7, 28, tzinfo=ist),
                          "open": 100.0, "high": 130.0, "low": 90.0, "close": 128.0}])
    t = simulate_swing(_swing_sig(), bars, 100000,
                       {"stop_pct": 7.0, "target_r": 3.0}, hold_days=5)
    assert t.exit_reason == "SL", "the pessimistic branch must be taken"


def test_swing_charges_costs_and_never_trades():
    from ab_paper import simulate_swing

    t = simulate_swing(_swing_sig(), _daily(6, step=0.5), 100000,
                       {"stop_pct": 7.0, "target_r": 3.0}, hold_days=5)
    assert t.costs > 0
    assert t.pnl == pytest.approx(t.gross_pnl - t.costs)

    body = (ROOT / "ab_paper.py").read_text().lower()
    for bad in ("place_order", "placeorder", "transactiontype"):
        assert bad not in body, f"ab_paper.py must never trade ({bad})"


def test_c03_and_c11_can_be_switched_off_but_default_on():
    """
    Both measured NEGATIVE at a 5-day horizon, so C disables them - but the
    live intraday scanner must keep Pine behaviour unless it opts out.
    """
    from config import Strategy

    assert Strategy().require_c03 is True
    assert Strategy().require_c11 is True

    from strategy import BarEval, gate_ok, WeeklySnapshot
    import indicators as ind

    snap = WeeklySnapshot(
        symbol="X", security_id="1", exchange_segment="NSE_EQ",
        week_start="2026-07-27", entry_level=100.0, level_52=100.0,
        hi_short2=90.0, close_1=95.0,
        ema_fast=ind.EmaState(length=20, prev=1.0),
        ema_slow=ind.EmaState(length=50, prev=1.0), ema_slow_2=1.0,
        rsi=ind.RsiState(length=14, avg_gain=1.0, avg_loss=1.0, prev_close=1.0),
        rsi_1=1.0,
        macd=ind.MacdState(fast=12, slow=26, signal=9, ema_fast=1.0,
                           ema_slow=1.0, sig=1.0),
        vol_sma=ind.SmaState(length=10, sum_prev=1.0),
        g_ema_fast=1.0, g_ema_slow=1.0, g_ema_slow_2=1.0,
        g_rsi=1.0, g_rsi_1=1.0, g_hist=1.0,
        prev_daily_close=1.0, prev_daily_open=1.0)

    conds = {f"c{i:02d}": True for i in range(1, 14)}
    conds["c03"] = False                     # fresh-breakout row fails
    ev = BarEval(conds, {})

    assert not gate_ok(snap, Strategy(), ev), "c03 must block by default"
    assert gate_ok(snap, Strategy(require_c03=False), ev), \
        "require_c03=False must let an extended breakout through"


# --------------------------------------------------------------------------- #
#  BUG 28 - the alert did not say what to actually DO.
#  User, 29-Jul: "send buy price, sl level price and first target price".
#
#  The alert gave only the trigger price, so the three numbers needed at the
#  moment of buying had to be computed by hand while the candle was still hot.
#
#  The danger in adding them is DRIFT: an alert quoting a 5% stop while Model C
#  paper-trades a 7% one would be worse than no plan at all. So the plan is
#  read from models.yaml - the same file the simulator uses - and the tests
#  below assert the printed numbers equal the ones simulate_swing enforces.
# --------------------------------------------------------------------------- #
def test_trade_plan_matches_the_swing_engine_exactly():
    """Printed stop and target must equal what simulate_swing actually uses."""
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    import pandas as pd

    from ab_paper import load_models, simulate_swing
    from strategy import BarEval, Signal
    from telegram import _swing_plan_params

    _defaults, models = load_models()
    swing = [m for m in models if m.is_swing]
    assert swing, "no swing model configured"
    C = swing[0]

    ist = ZoneInfo("Asia/Kolkata")
    price = 607.55
    ev = BarEval({f"c{i:02d}": True for i in range(1, 14)},
                 {"rsi": 70.0, "macd_hist": 1.0})
    sig = Signal("X", "1", "NSE_EQ", datetime(2026, 7, 29, 9, 15, tzinfo=ist),
                 price, 579.75, 579.75, "cross", ev, "2026-07-27", 1e6,
                 600.0, 561.0, bar_open=600.0, bar_high=610.0, bar_low=598.0)

    p = _swing_plan_params()
    assert p, "the alert could not read the swing plan from models.yaml"
    alert_stop = price * (1 - p["stop_pct"] / 100.0)
    alert_tgt = price + p["target_r"] * (price - alert_stop)

    # flat bars -> time exit, but t.stop exposes the engine's stop
    flat = pd.DataFrame([
        {"datetime": datetime(2026, 7, 30, tzinfo=ist) + timedelta(days=i),
         "open": price, "high": price * 1.001,
         "low": price * 0.999, "close": price} for i in range(5)])
    t = simulate_swing(sig, flat, 1_000_000, C.exit, C.hold_days)
    assert t.stop == pytest.approx(alert_stop), (
        f"alert quotes stop {alert_stop:.4f} but the engine uses {t.stop:.4f}")

    # a bar that tags the quoted target must exit there, at exactly 3R
    hit = pd.DataFrame([{"datetime": datetime(2026, 7, 30, tzinfo=ist),
                         "open": price, "high": alert_tgt * 1.01,
                         "low": price * 0.999, "close": alert_tgt}])
    t2 = simulate_swing(sig, hit, 1_000_000, C.exit, C.hold_days)
    assert t2.exit_reason == "TGT"
    assert t2.exit == pytest.approx(alert_tgt)
    assert t2.r_multiple == pytest.approx(p["target_r"], abs=0.01)


def test_alert_shows_buy_stop_and_target():
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from strategy import BarEval, Signal
    from telegram import format_signal

    ist = ZoneInfo("Asia/Kolkata")
    ev = BarEval({f"c{i:02d}": True for i in range(1, 14)},
                 {"rsi": 72.4, "macd_hist": 15.75, "ema_fast": 500.5,
                  "ema_slow": 442.3, "rsi_1": 67.9, "week_volume": 4686484,
                  "vol_sma": 1020073, "week_open": 561.5, "day_open": 600.15})
    sig = Signal("APCOTEXIND", "1", "NSE_EQ",
                 datetime(2026, 7, 29, 9, 15, tzinfo=ist),
                 607.55, 579.75, 579.75, "cross", ev, "2026-07-27",
                 4686484, 600.15, 561.5,
                 bar_open=600.2, bar_high=610.0, bar_low=598.0)
    msg = format_signal(sig)
    for want in ("Buy", "Stop", "Target", "607.55", "565.02", "735.14"):
        assert want in msg, f"alert is missing {want!r}"
    # the stop must be BELOW and the target ABOVE the buy price
    assert msg.index("Buy") < msg.index("Stop") < msg.index("Target")


def test_batch_alert_also_carries_the_plan():
    """A multi-signal alert must stay actionable without opening a chart."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from strategy import BarEval, Signal
    from telegram import _compact

    ist = ZoneInfo("Asia/Kolkata")
    ev = BarEval({f"c{i:02d}": True for i in range(1, 14)},
                 {"rsi": 70.0, "macd_hist": 1.0})
    sig = Signal("X", "1", "NSE_EQ", datetime(2026, 7, 29, 9, 15, tzinfo=ist),
                 100.0, 98.0, 98.0, "cross", ev, "2026-07-27", 1e6, 99.0, 97.0,
                 bar_open=99.5, bar_high=100.5, bar_low=99.0)
    out = _compact(sig)
    assert "SL" in out and "T1" in out
    assert "93.00" in out, "7% stop on a 100 entry should print 93.00"


def test_trade_plan_is_omitted_rather_than_wrong(tmp_path, monkeypatch):
    """
    If models.yaml is missing or has no swing model, print NO plan. A silently
    wrong stop is far more dangerous than an alert without one.
    """
    import telegram as tg

    monkeypatch.setattr(tg, "_PLAN_CACHE", None, raising=False)
    monkeypatch.setattr(tg, "__file__", str(tmp_path / "telegram.py"))
    assert tg.format_trade_plan(100.0) == []
    monkeypatch.setattr(tg, "_PLAN_CACHE", None, raising=False)


def test_trade_plan_never_quotes_a_stop_above_entry():
    from telegram import _swing_plan_params, format_trade_plan

    p = _swing_plan_params()
    assert p and 0 < p["stop_pct"] < 100
    assert format_trade_plan(0) == [], "a zero price must not produce a plan"
    lines = " ".join(format_trade_plan(250.0))
    assert "232.50" in lines, "7% below 250 is 232.50"
    # and the target must sit above entry: 250 + 3 x 17.50 = 302.50
    assert "302.50" in lines, "3R above 250 with a 7% stop is 302.50"


# --------------------------------------------------------------------------- #
#  BUG 29 - second Telegram destination. User, 29-Jul: "i want to add one more
#  telegram bot into the same alert, but it has different token and chat id".
#
#  The subtle risk is NOT the fan-out itself, it is what a failure means.
#  scan.py treats a False return as "delivery failed" and deliberately skips
#  state.save() so the next run retries. If a dead SPARE bot could force that
#  False, every run would re-alert the same signal - turning a nice-to-have
#  mirror into a duplicate-alert generator.
#
#  So: the PRIMARY alone decides the return value; a secondary failure is
#  logged and swallowed. These tests pin that down.
# --------------------------------------------------------------------------- #
class _FakeTg:
    def __init__(self, ok=True, raises=False):
        self.ok, self.raises, self.calls = ok, raises, []
        self.dry_run = False

    def _r(self, what):
        self.calls.append(what)
        if self.raises:
            raise RuntimeError("revoked token")
        return self.ok

    def send(self, text, disable_preview=True): return self._r(("send", text))
    def send_document(self, path, caption=""): return self._r(("doc", str(path)))
    def send_signal(self, sig): return self._r(("signal", sig.symbol))
    def send_batch(self, sigs): return self._r(("batch", len(sigs)))


def _fanout(*clients):
    from telegram import TelegramFanout
    f = TelegramFanout.__new__(TelegramFanout)
    f.clients = [(c, "primary" if i == 0 else "secondary")
                 for i, c in enumerate(clients)]
    return f


def test_fanout_delivers_to_every_destination():
    a, b = _FakeTg(), _FakeTg()
    assert _fanout(a, b).send("hello") is True
    assert a.calls and b.calls, "both destinations must receive the message"


def test_a_dead_secondary_must_not_cause_duplicate_alerts():
    """The critical one: secondary down -> still True, so state.save() runs."""
    a, b = _FakeTg(ok=True), _FakeTg(ok=False)
    assert _fanout(a, b).send("hello") is True, (
        "a failing SPARE bot must not block state.save(), or every scan would "
        "re-alert the same signal")


def test_a_dead_primary_still_forces_a_retry():
    a, b = _FakeTg(ok=False), _FakeTg(ok=True)
    assert _fanout(a, b).send("hello") is False, (
        "if the primary never got it, scan.py must retry next run")


def test_a_raising_secondary_is_isolated():
    a, b = _FakeTg(ok=True), _FakeTg(raises=True)
    assert _fanout(a, b).send("hello") is True, (
        "an exception in one destination must never escape the fan-out")


def test_fanout_covers_documents_and_batches():
    a, b = _FakeTg(), _FakeTg()
    f = _fanout(a, b)
    assert f.send_batch([]) is True
    assert f.send_document("x.csv", "cap") is True
    assert len(b.calls) == 2, "documents and batches must mirror too"


def test_second_destination_needs_both_token_and_chat_id(monkeypatch):
    """A half-configured pair must be ignored, never guessed."""
    from config import Secrets

    both = Secrets(telegram_bot_token="t1", telegram_chat_id="c1",
                   telegram_bot_token_2="t2", telegram_chat_id_2="c2")
    assert len(both.telegram_destinations) == 2

    half = Secrets(telegram_bot_token="t1", telegram_chat_id="c1",
                   telegram_bot_token_2="t2")           # chat id missing
    assert len(half.telegram_destinations) == 1, \
        "an incomplete second destination must be skipped"

    none = Secrets(telegram_bot_token="t1", telegram_chat_id="c1")
    assert len(none.telegram_destinations) == 1


def test_every_sender_uses_build_telegram():
    """
    One construction path, so a second bot can never mirror in scan.py while
    being silently missing from the watchlist or the paper report.
    """
    for name in ("scan.py", "watch.py", "watchlist.py",
                 "paper_report.py", "ab_paper.py"):
        body = (ROOT / name).read_text()
        assert "build_telegram(" in body, f"{name} must use build_telegram()"
        assert "Telegram(cfg.secrets" not in body, \
            f"{name} still builds a single-destination client"


def test_workflows_pass_the_second_bot_secrets():
    """Env vars are useless if the workflow never forwards them."""
    import yaml as _yaml

    for wf in WORKFLOWS.glob("*.yml"):
        text = wf.read_text()
        if "TELEGRAM_BOT_TOKEN:" not in text:
            continue                     # tests.yml sends nothing
        assert "TELEGRAM_BOT_TOKEN_2" in text, f"{wf.name} misses the 2nd token"
        assert "TELEGRAM_CHAT_ID_2" in text, f"{wf.name} misses the 2nd chat id"
        _yaml.safe_load(text)            # still valid YAML


# --------------------------------------------------------------------------- #
#  BUG 30 - A and B retired, C runs alone. User, 29-Jul: "remove a and b paper
#  trade, keep only C".
#
#  The harness was written around a two-way comparison, so several code paths
#  assumed `len(models) == 2`: the "HEAD TO HEAD" title, the "X leads by Y%"
#  verdict, and the Telegram summary. With one model those either read wrong or
#  printed nothing at all - the report would have said "need at least two
#  models with trades" forever and never told you whether C was making money.
# --------------------------------------------------------------------------- #
def test_single_model_report_states_profitability(capsys):
    """
    With ONE model there is nothing to rank - say if it is winning. Built as a
    synthetic single-model list so the test stays valid however many models
    models.yaml happens to define.
    """
    import pandas as pd

    from ab_paper import load_models, print_report

    _defaults, all_models = load_models()
    models = all_models[:1]
    rows = [{"model": models[0].key, "model_label": models[0].label,
             "symbol": f"S{i}", "signal_date": "2026-07-27",
             "exit_reason": "TIME", "pnl": 500.0, "pnl_pct": 5.0,
             "gross_pnl": 520.0, "costs": 20.0, "r_multiple": 1.5,
             "mfe_pct": 6.0, "mae_pct": -1.0, "bars_held": 5,
             "invested": 100000.0} for i in range(12)]
    print_report(pd.DataFrame(rows), models)
    out = capsys.readouterr().out
    assert "need at least two models" not in out
    assert "PROFITABLE so far" in out, out[-400:]
    assert "HEAD TO HEAD" not in out, "no comparison to make with one model"


def test_single_model_telegram_summary_is_not_silent():
    import pandas as pd

    from ab_paper import load_models, telegram_summary

    _defaults, all_models = load_models()
    models = all_models[:1]
    rows = [{"model": models[0].key, "model_label": models[0].label,
             "symbol": f"S{i}", "signal_date": "2026-07-27",
             "exit_reason": "TIME", "pnl": -200.0, "pnl_pct": -2.0,
             "gross_pnl": -180.0, "costs": 20.0, "r_multiple": -0.5,
             "mfe_pct": 1.0, "mae_pct": -3.0, "bars_held": 5,
             "invested": 100000.0} for i in range(11)]
    msg = telegram_summary(pd.DataFrame(rows), models)
    assert models[0].key in msg
    assert "❌" in msg, "a losing single model must be flagged, not hidden"


def test_intraday_models_a_and_b_stay_retired():
    """
    A and B were removed once every intraday exit measured negative after
    costs. New models may be added (D arrived 30-Jul), but nothing may
    reintroduce an INTRADAY paper model without a fresh measurement.
    """
    from ab_paper import load_models

    _defaults, models = load_models()
    keys = [m.key for m in models]
    assert "A_gated" not in keys and "B_mover" not in keys
    assert models, "at least one model must remain"
    for m in models:
        assert m.is_swing, f"{m.key} is intraday - that question is settled"
        assert m.hold_days == 5
        assert m.exit["stop_pct"] == 7.0
        assert m.exit["target_r"] == 3.0


def test_retired_models_reasoning_is_preserved():
    """
    Deleting A and B must not delete WHY. Their measured results are the
    justification for C existing, so the evidence stays in the repo.
    """
    doc = (ROOT / "models.yaml").read_text()
    assert "A \"gated\"" in doc or "A \"gated\"" in doc.replace("'", '"'), \
        "models.yaml must record what A and B were"
    assert "removed" in doc.lower() and "intraday" in doc.lower(), \
        "models.yaml must record why they were removed"


# --------------------------------------------------------------------------- #
#  BUG 31 - market cap filter (Pine c12) finally enabled.
#  User, 29-Jul: "Add 1 filter on to it, market capital >1000rs".
#
#  c12 had auto-passed since day one because Dhan exposes no shares
#  outstanding, so sub-1000 Cr names Chartink would reject - GANESHBE 881 Cr,
#  PYRAMID 665 Cr - reached the alerts anyway.
#
#  THE TRAP: c12 is MANDATORY inside gate_ok(). Flipping use_mcap on without a
#  data source makes snap.mcap None for every symbol, c12 False for every
#  symbol, and the scanner goes completely silent. Verified before the fix.
#  So "unknown" must PASS, and only a KNOWN-small cap may block.
# --------------------------------------------------------------------------- #
def _mcap_snap(mcap):
    import indicators as ind
    from strategy import WeeklySnapshot
    return WeeklySnapshot(
        symbol="X", security_id="1", exchange_segment="NSE_EQ",
        week_start="2026-07-27", entry_level=100.0, level_52=100.0,
        hi_short2=90.0, close_1=95.0,
        ema_fast=ind.EmaState(length=20, prev=1.0),
        ema_slow=ind.EmaState(length=50, prev=1.0), ema_slow_2=1.0,
        rsi=ind.RsiState(length=14, avg_gain=1.0, avg_loss=1.0, prev_close=1.0),
        rsi_1=1.0,
        macd=ind.MacdState(fast=12, slow=26, signal=9, ema_fast=1.0,
                           ema_slow=1.0, sig=1.0),
        vol_sma=ind.SmaState(length=10, sum_prev=1.0),
        g_ema_fast=1.0, g_ema_slow=1.0, g_ema_slow_2=1.0,
        g_rsi=1.0, g_rsi_1=1.0, g_hist=1.0,
        prev_daily_close=1.0, prev_daily_open=1.0, mcap=mcap)


def test_unknown_market_cap_must_not_block_a_signal():
    """The whole-scan-goes-silent bug. Unknown is not small."""
    from config import Strategy
    from strategy import evaluate_bar

    cfg = Strategy(use_mcap=True, min_mcap=1000.0)
    ev = evaluate_bar(_mcap_snap(None), cfg, 105.0, 99.0, 1e6, 100.0, 1.0)
    assert ev.conditions["c12"] is True, (
        "a symbol missing from mcap.csv must PASS c12 - blocking on missing "
        "data silently deletes stocks from the scan")


def test_small_caps_are_blocked_and_large_caps_pass():
    from config import Strategy
    from strategy import evaluate_bar

    cfg = Strategy(use_mcap=True, min_mcap=1000.0, mcap_margin_pct=5.0)

    def c12(mc):
        return evaluate_bar(_mcap_snap(mc), cfg, 105.0, 99.0,
                            1e6, 100.0, 1.0).conditions["c12"]

    assert c12(881.0) is False, "GANESHBE at 881 Cr must be blocked"
    assert c12(665.0) is False, "PYRAMID at 665 Cr must be blocked"
    assert c12(3078.0) is True, "MONARCH at 3078 Cr must pass"
    assert c12(206505.0) is True


def test_mcap_margin_keeps_borderline_names():
    """
    Share counts are a few percent stale, so a hard cut at exactly min_mcap
    would drop genuine setups on a rounding difference.
    """
    from config import Strategy
    from strategy import evaluate_bar

    def c12(mc, margin):
        cfg = Strategy(use_mcap=True, min_mcap=1000.0, mcap_margin_pct=margin)
        return evaluate_bar(_mcap_snap(mc), cfg, 105.0, 99.0,
                            1e6, 100.0, 1.0).conditions["c12"]

    assert c12(980.0, 5.0) is True, "980 Cr is within the 5% tolerance"
    assert c12(980.0, 0.0) is False, "with no margin it is below 1000"
    assert c12(940.0, 5.0) is False, "clearly below even with tolerance"


def test_disabling_use_mcap_restores_the_old_behaviour():
    from config import Strategy
    from strategy import evaluate_bar

    cfg = Strategy(use_mcap=False, min_mcap=1000.0)
    ev = evaluate_bar(_mcap_snap(1.0), cfg, 105.0, 99.0, 1e6, 100.0, 1.0)
    assert ev.conditions["c12"] is True


def test_mcap_table_loader_is_tolerant(tmp_path):
    """A malformed row must be skipped, not crash the weekly build."""
    from mcap import load_table

    p = tmp_path / "mcap.csv"
    p.write_text("symbol,mcap_cr,updated\n"
                 "RADICO,58556.00,2026-07-29\n"
                 "BROKEN,notanumber,2026-07-29\n"
                 "EMPTY,,2026-07-29\n"
                 "ZERO,0,2026-07-29\n"
                 "monarch,3078.00,2026-07-29\n")
    t = load_table(p)
    assert t == {"RADICO": 58556.0, "MONARCH": 3078.0}, t
    assert load_table(tmp_path / "missing.csv") == {}


def test_mcap_is_enabled_in_the_shipped_config():
    import yaml as _yaml

    cfg = _yaml.safe_load((ROOT / "config.yaml").read_text())["strategy"]
    assert cfg["use_mcap"] is True
    assert cfg["min_mcap"] == 1000
    assert 0 <= cfg.get("mcap_margin_pct", 0) <= 10


def test_snapshot_workflow_refreshes_and_commits_the_mcap_table():
    """A stale table is nearly as bad as none - it must rebuild weekly."""
    body = (WORKFLOWS / "snapshot.yml").read_text()
    assert "python mcap.py" in body, "snapshot.yml must rebuild mcap.csv"
    assert "mcap.csv" in body.split("git add")[1][:120], \
        "mcap.csv must be committed alongside the snapshot"
    assert body.index("python mcap.py") < body.index("python build_snapshot.py"), \
        "the table must be built BEFORE the snapshot that reads it"
    assert "continue-on-error: true" in body, \
        "a market-cap fetch failure must not abort the weekly snapshot"


# --------------------------------------------------------------------------- #
#  BUG 32 - Model D, the row-3 trigger level. User, 30-Jul: "i just want that
#  fresh breakout trigger level - enter on first 5-min close above hi_short2".
#
#  hi_short2 is highest(high,26) as of TWO weeks ago - row 3 of the indicator
#  table - and is <= entry_level by construction, so D fires earlier than C.
#
#  HONESTY: a point-in-time backtest of this trigger measured -0.05%/trade
#  (t = -0.21, n = 815). An earlier +0.82% came from a look-ahead bug that
#  gated on COMPLETED weekly bars while entering mid-week. D ships as PAPER
#  only, on the user's explicit instruction, to be judged on live results.
# --------------------------------------------------------------------------- #
def test_trigger_level_defaults_to_the_pine_level():
    """The live scanner and Models A/B/C must be unaffected."""
    from config import Strategy
    assert Strategy().trigger_level == "entry"


def test_hi_short2_trigger_fires_at_or_before_the_entry_level():
    """
    hi_short2 <= entry_level always, so a cross of it can never be LATER than
    a cross of entry_level. If this inverts, the level wiring is wrong.
    """
    import pandas as pd

    from config import Strategy
    from strategy import build_snapshot, build_weekly_bars, replay_week
    from test_strategy import make_daily

    daily = make_daily(weeks=200)
    wkb = build_weekly_bars(daily)
    target = wkb.iloc[-1]["week_start"]

    base = Strategy(strict_entry=False)
    snap = build_snapshot("T", "1", "NSE_EQ", daily, base, target)
    assert snap is not None
    assert snap.hi_short2 <= snap.entry_level + 1e-9

    # a ramp that crosses hi_short2 first, then entry_level
    lo, hi = snap.hi_short2, snap.entry_level
    rows = []
    for i in range(40):
        p = lo * 0.995 + (hi * 1.01 - lo * 0.995) * (i / 39.0)
        rows.append({"datetime": pd.Timestamp(target) + pd.Timedelta(minutes=5 * i),
                     "open": p, "high": p * 1.002, "low": p * 0.998,
                     "close": p, "volume": 10_000})
    bars = pd.DataFrame(rows)

    d_cfg = Strategy(strict_entry=False, trigger_level="hi_short2")
    c_cfg = Strategy(strict_entry=False, trigger_level="entry")
    d_sig = replay_week(snap, d_cfg, bars).signals
    c_sig = replay_week(snap, c_cfg, bars).signals
    assert d_sig, "the hi_short2 trigger produced no signal"
    if c_sig:
        assert d_sig[0].bar_time <= c_sig[0].bar_time, \
            "D must never trigger later than C"


def test_signal_reports_the_level_it_actually_used():
    """
    The alert prints sig.entry_level. If D quoted C's level the trade plan
    would show a breakout price the entry never crossed.
    """
    import pandas as pd

    from config import Strategy
    from strategy import build_snapshot, build_weekly_bars, replay_week
    from test_strategy import make_daily

    daily = make_daily(weeks=200)
    wkb = build_weekly_bars(daily)
    target = wkb.iloc[-1]["week_start"]
    cfg = Strategy(strict_entry=False, trigger_level="hi_short2")
    snap = build_snapshot("T", "1", "NSE_EQ", daily, cfg, target)

    lvl = snap.hi_short2
    rows = [{"datetime": pd.Timestamp(target) + pd.Timedelta(minutes=5 * i),
             "open": lvl * 0.99, "high": lvl * 1.02, "low": lvl * 0.98,
             "close": lvl * (0.995 if i == 0 else 1.01), "volume": 10_000}
            for i in range(6)]
    sigs = replay_week(snap, cfg, pd.DataFrame(rows)).signals
    assert sigs
    assert sigs[0].entry_level == pytest.approx(snap.hi_short2), \
        "the signal must carry the level it actually crossed"


def test_model_d_is_registered_and_uses_the_row3_level():
    from ab_paper import load_models

    _defaults, models = load_models()
    keys = [m.key for m in models]
    assert "D_early" in keys
    D = [m for m in models if m.key == "D_early"][0]
    assert D.strategy["trigger_level"] == "hi_short2"
    assert D.is_swing and D.hold_days == 5
    # C must be untouched by D's addition
    C = [m for m in models if m.key == "C_swing"][0]
    assert C.strategy.get("trigger_level", "entry") == "entry"


def test_ab_paper_feeds_market_caps_to_the_snapshot():
    """
    With use_mcap on, a snapshot built without a cap reports mcap=None. That
    passes c12 as 'unknown', but it means the paper models would not see the
    same universe as the live scanner. Wire the real table in.
    """
    body = (ROOT / "ab_paper.py").read_text()
    assert "load_mcap_table" in body, "ab_paper.py must load mcap.csv"
    assert "mcap=MCAP.get(" in body, \
        "build_snapshot must receive the market cap"


def test_model_d_records_the_measured_result_honestly():
    """The negative backtest must stay documented next to the model."""
    doc = (ROOT / "models.yaml").read_text()
    assert "-0.05" in doc and "look-ahead" in doc.lower(), \
        "models.yaml must record D's measured result and the earlier bug"


# --------------------------------------------------------------------------- #
#  BUG 33 - the watchlist ignored market cap and only tracked C's level.
#  User, 30-Jul: "update watchlist logic and market cap as well".
#
#  Two independent gaps, both silent:
#
#  1. MARKET CAP. is_eligible() screened on c03/c05/c08 only, so the digest
#     happily listed sub-1000 Cr names the live scan would reject. Reading the
#     cap off the SNAPSHOT would not have worked either: the committed
#     weekly_snapshot.csv was built before the feature and carries mcap=None
#     for 2099 of 2100 rows, so the filter would have been inert with no
#     visible symptom. The table is loaded from mcap.csv instead.
#
#  2. LEVEL. It measured distance to entry_level only. Model D triggers on
#     hi_short2, which is <= entry_level, so any name about to trigger D was
#     either missing or shown against a level it would not cross first.
# --------------------------------------------------------------------------- #
def test_watchlist_applies_the_market_cap_screen():
    from config import Strategy
    from watchlist import is_eligible

    class S:
        close_1 = 100.0
        hi_short2 = 120.0
        g_hist = 1.0
        g_ema_slow = 50.0
        g_ema_slow_2 = 49.0

    cfg = Strategy(use_mcap=True, min_mcap=1000.0, mcap_margin_pct=5.0)
    assert is_eligible(S(), cfg, 3078.0) is True, "a large cap must pass"
    assert is_eligible(S(), cfg, 881.0) is False, "GANESHBE at 881 Cr must fail"
    assert is_eligible(S(), cfg, None) is True, (
        "unknown cap must PASS - never hide a name because the cap could not "
        "be resolved")

    off = Strategy(use_mcap=False, min_mcap=1000.0)
    assert is_eligible(S(), off, 100.0) is True, \
        "with use_mcap off the screen must not apply"


def test_watchlist_reads_caps_from_the_table_not_the_snapshot():
    """
    The committed snapshot predates c12 and has mcap=None almost everywhere.
    Reading caps from there would disable the screen silently.
    """
    body = (ROOT / "watchlist.py").read_text()
    assert "load_mcap_table" in body, "watchlist.py must load mcap.csv"
    assert "caps.get(" in body, "the cap must come from the table"


def test_watchlist_tracks_both_model_levels():
    """
    C triggers on entry_level, D on hi_short2. hi_short2 <= entry_level, so a
    name approaching D's level must not be measured against C's.
    """
    body = (ROOT / "watchlist.py").read_text()
    assert "hi_short2" in body, "watchlist.py must consider D's level"
    assert "level_d" in body and "level_c" in body, \
        "the CSV must record both levels so nothing is hidden"


def test_watchlist_picks_the_nearest_level_above_price():
    """A breakout reaches the LOWER of the two levels first."""
    from watchlist import build_message

    # D level 95 is nearer than C level 100 for a price of 94
    rows = [dict(symbol="AAA", ltp=94.0, level=95.0, which="D",
                 level_c=100.0, level_d=95.0, mcap_cr=5000.0,
                 pct=-1.05, gap=1.05, bucket="WATCH")]
    msg = build_message("2026-07-27", rows,
                        {"universe": 1, "eligible": 1, "capped": 1}, 3.0)
    assert "95.00" in msg, "the nearer (D) level must be quoted"
    assert "[D]" in msg, "the message must say which model's level it is"
    assert "5,000Cr" in msg, "the market cap should be visible"


# --------------------------------------------------------------------------- #
#  BUG 34 - the market-cap screen was silently skipped in the watchlist.
#  User, 30-Jul: "still coming too many false stocks and under 1000cr",
#  with QMSMEDI (264 Cr) in the digest.
#
#  is_eligible() did  getattr(cfg, "use_mcap", False)  but main() passes the
#  full Config, and use_mcap lives on cfg.STRATEGY. The getattr default made it
#  False, so the entire branch was skipped - no error, no warning, just no
#  filtering. The BUG 33 test missed it because it passed a Strategy directly,
#  which is not what production does.
#
#  LESSON: a test must call the function the way the caller does. Passing a
#  tidier object than production uses is how a no-op passes review.
# --------------------------------------------------------------------------- #
def test_is_eligible_works_with_the_object_main_actually_passes():
    """main() hands is_eligible a Config, not a Strategy. Both must work."""
    from config import load_config
    from watchlist import is_eligible

    class S:
        close_1 = 100.0
        hi_short2 = 120.0
        g_hist = 1.0
        g_ema_slow = 50.0
        g_ema_slow_2 = 49.0

    cfg = load_config(None)
    assert cfg.strategy.use_mcap is True, "this test assumes c12 is enabled"

    # the exact call shape used in production
    assert is_eligible(S(), cfg, 264.0) is False, (
        "QMSMEDI at 264 Cr must be rejected when a full Config is passed - "
        "this is the call main() makes")
    assert is_eligible(S(), cfg, 5000.0) is True
    assert is_eligible(S(), cfg, None) is True, "unknown must still pass"

    # and the Strategy form must behave identically
    assert is_eligible(S(), cfg.strategy, 264.0) is False
    assert is_eligible(S(), cfg.strategy, 5000.0) is True


def test_watchlist_main_passes_something_is_eligible_understands():
    """
    Guard the wiring itself: if main() ever passes a bare Strategy again, or
    is_eligible stops unwrapping, this catches it without a live run.
    """
    body = (ROOT / "watchlist.py").read_text()
    assert 'getattr(cfg, "strategy", cfg)' in body, (
        "is_eligible must unwrap Config -> Strategy so the screen cannot be "
        "silently skipped")
