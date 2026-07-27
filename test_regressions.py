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
        capture_output=True, text=True, timeout=120,
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
    assert "11:20" in full and "Entry" in full
    assert "11:20-11:25" in full, "show the 5m candle window"
    assert "11:20" in _compact(sig), "batch lines must show entry time too"


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
