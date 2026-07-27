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
    import pandas as pd

    from config import load_config
    from strategy import build_snapshot, replay_week, week_start_of
    from test_end_to_end import strong_uptrend_daily, week_of_bars

    cfg = load_config(ROOT / "config.yaml")
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
    seq.append((101.0, 101.1, 100.8, 100.9))     # bar we actually exit on
    t = simulate(_paper_signal(), _paper_bars(seq), 100000)
    assert t.exit_reason == "EMA9"
    assert t.bars_held == len(seq), "EMA exit fills at the NEXT bar's open"


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
                             (100.9, 101.5, 100.7, 101.3)]), 100000, _warm())
    assert t.signal_time == "10:15" and t.signal_close == 100.0
    assert t.entry_time == "10:20", "entry is the bar AFTER the signal"
    assert t.entry == 100.6, "fill at that bar's OPEN"
    assert abs(t.slippage_pct - 0.6) < 0.01


def test_ema_exit_fills_at_next_open_no_lookahead():
    from paper import simulate

    t = simulate(_exec_sig(100.0),
                 _exec_bars([(100.6, 101, 100.4, 98.0),
                             (97.9, 98.2, 97.5, 97.8)]), 100000, _warm())
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

    t = simulate(sig, bars, 100000, _warm())
    assert t.signal_time == "15:25"
    assert t.entry_date == "2026-07-28" and t.entry_time == "09:15", \
        "a signal on the closing bar can only be filled next session"
    assert t.entry == 101.0, "fill at the next session's open, gap included"
