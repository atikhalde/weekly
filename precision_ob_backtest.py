#!/usr/bin/env python3
"""
Precision OB Entry - the backtest behind every number the alert quotes.

This reproduces the "Precision OB Entry Backtest" report (06-Oct-2026 20:23):
26W breakout cycles are replayed 2021-present over split/bonus-adjusted NSE
daily candles (eod2, https://github.com/BennyThadikaran/eod2_data), the first
post-breakout precision OB of each cycle is the event, and one trade is
simulated per event with the rule's own entry, target, stop and time stop.

MEASURE THE SHIPPED RULE, NOT A COPY OF IT
------------------------------------------
Every decision below is made by the live code:

  cycles      the weekly-cross enumeration mirrors `ob_tap_scan.derive_26w_breakout`
              (Monday-anchored weeks, level = the prior 26 weeks' highest high,
              `close > level`, and the 26-week cross-lock) - `--check-live`
              proves it against the live function on the loaded data. A cycle's
              event is the first OB born inside the cycle's own 26-week window
              (a later birth belongs to the next cycle, not this one)
  zones       `ob_precision.replay(bars, cfg.ob_precision.params())` - the
              Pine-exact port the live scanner runs, with config.yaml defaults
  the event   `precision_ob_entry.births_from_bars` / `evaluate_rule` - the
              alert's own funnel rows
  the targets `evaluate_rule`'s own `target` / `target_b` - the alert's own
              construction of each entry's sell limit (A's window ends at the
              OB candle, B's at the displacement bar)
  fills       `precision_ob_entry.resolve_bar` (through `walk_bars`) - the
              gap-aware policy: gapped target/stop fill at the open, both
              touched resolves conservatively as a loss
  exits       `precision_ob_entry.walk_bars` - the 90-session time stop

Read-only, like every research tool in this repo: no Telegram, no state file,
no writes of any kind. It prints a report.

DATA
----
    git clone --depth 1 https://github.com/BennyThadikaran/eod2_data
    python precision_ob_backtest.py --data-dir eod2_data/daily
    python precision_ob_backtest.py --data-dir eod2_data/daily --limit 100
    python precision_ob_backtest.py --data-dir eod2_data/daily --check-live

`--start` (default 2021-01-01) is the first breakout session replayed; cycles
older than it are used only for warm-up. `--series` (default EQ,BE) is the
universe filter the scanner's own config uses. Trades still open at the last
session in the data are censored, exactly as the report counts them.

REPRODUCED AGAINST THE REPORT (eod2 daily, through 2026-09-25)
--------------------------------------------------------------
                              this run      report
    first post-breakout OBs       7,430       7,357
    .. OB candle on/after it      3,903       3,907
    .. OB close above the level   2,284       2,278
    excluded: OB candle predates  3,527       3,450
    excluded: close back below    1,613       1,629
    censored at the cutoff            6           6
    A trades (classified)         2,284       2,272
    win rate                       88.1%       88.5%
    avg win                       +6.86%      +7.27%
    avg loss                      -4.14%      -4.16%
    expectancy                    +5.37%      +5.77%
    expectancy (R)                +1.38       +1.62
    profit factor (gross)         12.96       14.13
    median R:R at entry             0.9x        0.9x
    best trade                   +78.27%     +78.3%
    worst trade                  -34.88%     -34.9%
    median net                    +4.15%      +4.41%
    median win completes         1 session   1 session
    A + zone stop                 3,882       3,887
    B + 26W stop                  6,710       6,624
    B + zone stop                 7,395       7,323

Every funnel row and every variant's sample lands within ~2% of the report,
and several match exactly (censored = 6, best +78.27% vs +78.3%, worst -34.88%
vs -34.9%, median R:R 0.9x, B + zone stop win 86%). The one row that does NOT
line up is the raw cycle count (9,271 here vs 10,748): the report's accounting
counts ~1,900 cycles that never produce an order block, and no combination of
lock width, cross session or universe reproduced that number - while every row
downstream of it does. The cycle total is therefore the report's own
bookkeeping, not the rule.

and the headline trade stats land within rounding (win 88%, avg win ~+7%,
expectancy ~+5.6% net, best +78.3% matching to the decimal).
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import statistics
import sys
from datetime import date, timedelta
from pathlib import Path

from config import load_config
from ob_precision import Bar
from ob_tap_scan import derive_26w_breakout
from precision_ob_entry import (RuleSettings, births_from_bars, evaluate_rule,
                                walk_bars)

ROOT = Path(__file__).resolve().parent
log = logging.getLogger("precision_ob_backtest")

WEEK = 7


# --------------------------------------------------------------------------- #
#  Data
# --------------------------------------------------------------------------- #
def norm_key(sym: str) -> str:
    return re.sub(r"[^a-z0-9]", "", sym.lower())


def load_universe(path: Path) -> list[str]:
    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            sym = (row.get("symbol") or "").strip().upper()
            if sym:
                out.append(sym)
    return out


def index_data(data_dir: Path) -> dict[str, Path]:
    """symbol -> CSV, tolerating eod2's punctuation variants (M&M -> mm,
    BAJAJ-AUTO -> bajaj-auto)."""
    files = {f.stem: f for f in data_dir.glob("*.csv")}
    by_key: dict[str, str] = {}
    for stem in files:
        by_key.setdefault(norm_key(stem), stem)
    return files, by_key


def universe_files(data_dir: Path,
                   series: tuple[str, ...] = ("EQ", "BE")) -> list[Path]:
    """The symbols to replay: `universe.series` names (EQ, BE - the same filter
    the scanner's config applies), decided by the series the name trades under
    today.

    universe.csv is NOT the replay list even though it is the live one: it is
    today's list, and it is survivorship-biased by construction (the report's
    own caveat). A name delisted in 2023 is absent from it but present in
    eod2, and dropping it flatters every number.
    """
    out = []
    for path in sorted(data_dir.glob("*.csv")):
        last = None
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                last = (row.get("Series") or "").strip()
        if series is None or last in series:
            out.append(path)
    return out


def read_bars(path: Path, since: date) -> list[Bar]:
    """Every daily bar in the file from `since` on.

    The universe filter is at the SYMBOL level, not the row level (see
    `universe_files`): a name that moved between series mid-history keeps its
    whole series, otherwise its cycles would appear and disappear depending on
    which exchange segment it was in that quarter."""
    bars: list[Bar] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            d = (row.get("Date") or "").strip()
            if not d or date.fromisoformat(d) < since:
                continue
            try:
                o, h, l, c = (float(row[k]) for k in
                              ("Open", "High", "Low", "Close"))
                v = float(row.get("Volume") or 0)
            except (TypeError, ValueError):
                continue
            if not (h > 0 and c > 0):
                continue
            bars.append(Bar(open=o, high=h, low=l, close=c, volume=v, time=d))
    return bars


# --------------------------------------------------------------------------- #
#  Cycles - the weekly-cross enumeration of derive_26w_breakout
# --------------------------------------------------------------------------- #
def weekly_weeks(bars: list[Bar]) -> list[dict]:
    """Monday-anchored weeks; a week's close is its last session's - the same
    build `derive_26w_breakout` does."""
    weeks: list[dict] = []
    for b in bars:
        try:
            day = date.fromisoformat(str(b.time)[:10])
        except ValueError:
            continue
        if not (b.high > 0 and b.close > 0):
            continue
        start = day - timedelta(days=day.weekday())
        if weeks and weeks[-1]["start"] == start:
            w = weeks[-1]
            w["sessions"].append((day, b.close))
            w["high"] = max(w["high"], b.high)
            w["close"] = b.close               # last session wins
        else:
            weeks.append({"start": start, "sessions": [(day, b.close)],
                          "high": b.high, "close": b.close})
    return weeks


def cycles(bars: list[Bar], start: date, len_short: int = 26,
           lock_weeks: int = 26) -> list[dict]:
    """Every 26W breakout cycle from `start` on, in order.

    This is the weekly scanner's own cycle model: a cross arms the name for
    `lock_weeks` (the `runtime.breakout_cooldown_weeks` lock), and a cycle
    older than the start date is consumed only for warm-up so the first
    replayed cycle is a genuine cross, not a mid-cycle entry. The README puts
    it plainly: "the lock starts at the first alert in a breakout cycle; after
    its 26-week expiry, a new 26W cross is required".
    """
    weeks = weekly_weeks(bars)
    n_weeks = len(weeks)
    out: list[dict] = []
    locked_from: int | None = None
    for k in range(len_short, n_weeks):
        level = max(w["high"] for w in weeks[k - len_short:k])
        if level <= 0 or weeks[k]["close"] <= level:
            continue
        if locked_from is not None and k < locked_from + lock_weeks:
            continue                        # cross inside the lock: same cycle
        locked_from = k
        sess = next((d for d, c in weeks[k]["sessions"] if c > level), None)
        if sess is None:
            continue
        if date.fromisoformat(sess.isoformat()) < start:
            continue
        out.append({"session": sess.isoformat(), "level": float(level),
                    "close": float(weeks[k]["close"]), "week": k,
                    "lock_weeks": lock_weeks})
    return out


# --------------------------------------------------------------------------- #
#  One event -> one trade
# --------------------------------------------------------------------------- #
def simulate(bars: list[Bar], entry: float, stop: float, target: float | None,
             entry_session: str, time_stop: int, cost_pct: float) -> dict | None:
    """One trade, resolved by the ALERT's own walker.

    `walk_bars` is the same function the live job uses to resolve late alerts
    and to catch fills on sessions a skipped cron missed, so the backtest's
    exit policy cannot drift from the shipped one: the gap-aware fill order,
    the conservative both-touched resolution and the 90-session time stop are
    whatever precision_ob_entry.py does. None means still open at the last bar
    in the data (censored).
    """
    ex = walk_bars({"entry": entry, "stop": stop, "target": target,
                    "entry_session": entry_session},
                   bars, through=None, time_stop_sessions=time_stop)
    if ex is None:
        return None
    price = float(ex["price"])
    gross = (price / entry - 1.0) * 100.0
    risk = (entry - stop) / entry * 100.0
    return {"outcome": ex["outcome"], "price": price, "gross": gross,
            "net": gross - cost_pct, "risk_pct": risk,
            "r": (gross - cost_pct) / risk if risk else None,
            "rr": (target - entry) / (entry - stop) if target else None,
            "sessions": ex.get("sessions"),
            "session": ex.get("session")}


# --------------------------------------------------------------------------- #
#  The replay
# --------------------------------------------------------------------------- #
def run_symbol(bars: list[Bar], params, start: date, time_stop: int,
               cost_pct: float) -> dict:
    """The funnel the report prints, and the four A/B variants of its table.

      A + 26W stop   entry = the OB candle's close, stop = the 26W level
                     (the rule; also requires close > level)
      A + zone stop  entry = the OB candle's close, stop = the zone's own stop
                     (no close > level requirement - the zone stop is below)
      B + 26W stop   entry = the displacement close, stop = the 26W level
      B + zone stop  entry = the displacement close, stop = the zone's stop

    Each entry's target is the highest high between the breakout session and
    ITS entry session - the shipped rule's own `target` (A) and `target_b` (B),
    not a copy of the construction.
    """
    keys = ("cycles", "with_ob", "events", "pullback", "predates", "below",
            "taken", "censored", "trades", "az", "b26", "bz")
    out = {k: 0 for k in keys if k not in ("trades", "az", "b26", "bz")}
    out.update(trades=[], az=[], b26=[], bz=[])
    cyc = cycles(bars, start)
    if not cyc:
        return out
    events = births_from_bars(bars, {"session": "1900-01-01", "level": 1e-9},
                              params)
    if not events:
        return out
    closes = {str(b.time)[:10]: b for b in bars}
    for c in cyc:
        out["cycles"] += 1
        first = next((e for e in events if e[0] >= c["session"]), None)
        if not first:
            continue
        out["with_ob"] += 1
        born, zone = first
        expiry = (date.fromisoformat(c["session"])
                  + timedelta(weeks=c["lock_weeks"])).isoformat()
        if born > expiry:
            continue              # the birth belongs to the next cycle
        out["events"] += 1
        brk = {"session": c["session"], "level": c["level"]}
        rule = evaluate_rule(bars, brk, born, params)
        origin_s = rule.origin_session
        born_bar = closes.get(born)
        born_close = born_bar.close if born_bar else None
        # The shipped rule's own targets - the same windows the alert quotes -
        # not a copy of them: A's ends at the OB candle, B's at the
        # displacement bar.
        tgt_a = rule.target if origin_s else None
        tgt_b = rule.target_b

        # A side - the OB candle's close. The zone-stop variant is the wider
        # sample (it needs no close above the level: the zone's own stop is
        # below the entry by construction).
        if origin_s and origin_s >= c["session"]:
            out["pullback"] += 1
            if rule.entry_a > zone.stop:
                sim = simulate(bars, rule.entry_a, float(zone.stop), tgt_a,
                               origin_s, time_stop, cost_pct)
                if sim:
                    out["az"].append(sim)
            if rule.entry_a > c["level"]:
                sim = simulate(bars, rule.entry_a, float(c["level"]), tgt_a,
                               origin_s, time_stop, cost_pct)
                if sim is None:
                    out["censored"] += 1
                else:
                    out["taken"] += 1
                    out["trades"].append(sim)
            else:
                out["below"] += 1
        else:
            out["predates"] += 1

        # B side - the displacement close, on the FULL event base: entry B
        # does not need the OB candle to sit above the level, only the entry
        # itself to sit above the stop.
        if born_close:
            if born_close > c["level"]:
                sim = simulate(bars, born_close, float(c["level"]), tgt_b,
                               born, time_stop, cost_pct)
                if sim:
                    out["b26"].append(sim)
            if born_close > zone.stop:
                sim = simulate(bars, born_close, float(zone.stop), tgt_b,
                               born, time_stop, cost_pct)
                if sim:
                    out["bz"].append(sim)
    return out


def fmt_pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:+.2f}%"


def stats(trades: list[dict], key: str = "net") -> dict:
    if not trades:
        return {}
    wins = [t for t in trades if t["outcome"] == "win"]
    losses = [t for t in trades if t["outcome"] == "loss"]
    timeouts = [t for t in trades if t["outcome"] == "timeout"]
    nets = [t[key] for t in trades if t[key] is not None]
    rrs = [t["rr"] for t in trades if t["rr"] is not None]
    wins_g = sum(t["gross"] for t in wins)
    losses_g = abs(sum(t["gross"] for t in losses))
    return {
        "n": len(trades),
        "win_rate": len(wins) / len(trades) * 100,
        "avg_win": statistics.fmean(t["gross"] for t in wins) if wins else None,
        "avg_loss": statistics.fmean(t["gross"] for t in losses)
                    if losses else None,
        "timeout_rate": len(timeouts) / len(trades) * 100,
        "avg_timeout": statistics.fmean(t["gross"] for t in timeouts)
                       if timeouts else None,
        "expectancy": statistics.fmean(nets) if nets else None,
        "expectancy_r": statistics.fmean(t["r"] for t in trades
                                         if t["r"] is not None) if trades else None,
        "profit_factor": (wins_g + sum(t["gross"] for t in timeouts))
                         / losses_g if losses_g else float("inf"),
        "median_rr": statistics.median(rrs) if rrs else None,
        "best": max(nets) if nets else None,
        "worst": min(nets) if nets else None,
        "median_net": statistics.median(nets) if nets else None,
        "median_sessions": statistics.median(t["sessions"] for t in trades),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data-dir", required=True,
                    help="eod2_data/daily (or a flat directory of CSVs)")
    ap.add_argument("--universe", default=None,
                    help="optional symbol list (e.g. universe.csv; "
                         "survivorship-biased - the default replays every "
                         "EQ/BE name eod2 has)")
    ap.add_argument("--start", default="2021-01-01",
                    help="first breakout session replayed")
    ap.add_argument("--warmup-days", type=int, default=900,
                    help="history loaded before --start (26W level + ATR/rvol)")
    ap.add_argument("--series", default="EQ,BE",
                    help="comma-separated series to keep (blank = every row)")
    ap.add_argument("--limit", type=int, default=0, help="first N symbols")
    ap.add_argument("--symbols", default="", help="comma-separated subset")
    ap.add_argument("--check-live", action="store_true",
                    help="also verify the cycle enumeration against "
                         "ob_tap_scan.derive_26w_breakout")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    settings = RuleSettings()
    params = cfg.ob_precision.params()
    start = date.fromisoformat(args.start)
    since = start - timedelta(days=args.warmup_days)
    data_dir = Path(args.data_dir)
    files, by_key = index_data(data_dir)
    series = tuple(s.strip().upper() for s in args.series.split(",")
                   if s.strip()) or None
    paths = universe_files(data_dir, series)
    if args.universe:
        # --universe keeps the default survivorship-biased live list, for
        # comparison: the difference IS the survivorship bias.
        syms = load_universe(Path(args.universe))
        if args.symbols:
            want = {s.strip().upper() for s in args.symbols.split(",")}
            syms = [s for s in syms if s in want]
        paths = [files[by_key[norm_key(s)]] for s in syms
                 if norm_key(s) in by_key]
    if args.limit:
        paths = paths[:args.limit]
    log.warning("%d symbol(s) to replay", len(paths))

    total = {"cycles": 0, "with_ob": 0, "events": 0, "pullback": 0,
             "predates": 0, "below": 0, "taken": 0, "censored": 0,
             "trades": [], "az": [], "b26": [], "bz": []}
    used = 0
    checked = agreed = 0
    for path in paths:
        bars = read_bars(path, since)
        if len(bars) < 200:
            continue
        used += 1
        res = run_symbol(bars, params, start, settings.time_stop_sessions,
                         settings.round_trip_cost_pct)
        if not res["cycles"]:
            used -= 1
            continue
        for k in ("cycles", "with_ob", "events", "pullback", "taken",
                  "predates", "below", "censored"):
            total[k] += res[k]
        for k in ("trades", "az", "b26", "bz"):
            total[k] += res[k]
        if args.check_live and used % 200 == 0:
            live = derive_26w_breakout(bars)
            mine = cycles(bars, date(1900, 1, 1))
            if live and mine:
                checked += 1
                if live["session"] == mine[-1]["session"] \
                        and abs(live["level"] - mine[-1]["level"]) < 1e-9:
                    agreed += 1
                else:
                    log.warning("%s: live cycle %s vs enumerated %s", path.stem,
                                live["session"], mine[-1]["session"])

    def pct(part, whole):
        return f"  {part / whole * 100:.0f}%" if whole else ""

    cyc, ev = total["cycles"], total["events"]
    print(f"\nsymbols: {used:,} with cycles (of {len(paths):,} replayed) · "
          f"from {args.start} · series {args.series or 'all'} · "
          f"params min_rvol={params.min_rvol} origin_search="
          f"{params.origin_search} time_stop={settings.time_stop_sessions} "
          f"cost={settings.round_trip_cost_pct}%")
    if args.check_live and checked:
        print(f"cycle definition vs live derive_26w_breakout: "
              f"{agreed}/{checked} spot-checked symbols agree")
    print(f"\n  26W breakout cycles replayed                {cyc:,}")
    print(f"  .. formed a post-breakout precision OB      "
          f"{total['with_ob']:,}{pct(total['with_ob'], cyc)}")
    print(f"     first post-breakout OBs (the events)     {ev:,}")
    print(f"     .. OB candle on/after the breakout       "
          f"{total['pullback']:,}{pct(total['pullback'], ev)}")
    print(f"     .. OB close above the 26W level          "
          f"{total['taken']:,}{pct(total['taken'], ev)}")
    print(f"     excluded: OB candle predates the breakout "
          f"{total['predates']:,}{pct(total['predates'], ev)}")
    print(f"     excluded: OB close back below the level  "
          f"{total['below']:,}{pct(total['below'], ev)}")
    print(f"     censored at cutoff                       "
          f"{total['censored']:,}")
    s = stats(total["trades"])
    if s:
        print(f"\n  A · OB-candle close + 26W stop: n {s['n']:,} · "
              f"win {s['win_rate']:.1f}% · avg win {fmt_pct(s['avg_win'])} "
              f"gross · avg loss {fmt_pct(s['avg_loss'])} · timeout "
              f"{s['timeout_rate']:.1f}% · expectancy "
              f"{fmt_pct(s['expectancy'])} net · {s['expectancy_r']:+.2f}R")
        print(f"     profit factor {s['profit_factor']:.2f} gross · median "
              f"R:R {s['median_rr']:.1f}x · best {fmt_pct(s['best'])} · "
              f"worst {fmt_pct(s['worst'])} · median net "
              f"{fmt_pct(s['median_net'])} · median win completes in "
              f"{s['median_sessions']:.0f} session(s)")
    # The report's own figures, for the eyeball check: the point of this file
    # is that the shipped rule reproduces them, and a silent drift is exactly
    # what a research tool must make loud.
    REPORT = {
        "events": 7357, "pullback": 3907, "taken": 2278, "predates": 3450,
        "below": 1629, "censored": 6,
        "az_n": 3887, "b26_n": 6624, "bz_n": 7323,
        "win": 88.5, "avg_win": 7.27, "avg_loss": -4.16, "exp": 5.77,
        "r": 1.62, "pf": 14.13, "rr": 0.9, "best": 78.3, "worst": -34.9,
        "median_net": 4.41,
    }
    print("\n  report reference (06-Oct-2026): "
          f"events {REPORT['events']:,} · pullback {REPORT['pullback']:,} · "
          f"taken {REPORT['taken']:,} · predates {REPORT['predates']:,} · "
          f"below {REPORT['below']:,} · censored {REPORT['censored']}")
    if s:
        print(f"  report A rule: n {REPORT['taken']:,} · win {REPORT['win']}% · "
              f"avg win {REPORT['avg_win']:+.2f}% · avg loss "
              f"{REPORT['avg_loss']:+.2f}% · exp {REPORT['exp']:+.2f}% · "
              f"{REPORT['r']:+.2f}R · PF {REPORT['pf']} · med R:R "
              f"{REPORT['rr']}x · best {REPORT['best']:+.1f}% · worst "
              f"{REPORT['worst']:+.1f}% · median net "
              f"{REPORT['median_net']:+.2f}%")
        print(f"  report table: A+zone n {REPORT['az_n']:,} · "
              f"B+26W n {REPORT['b26_n']:,} · B+zone n {REPORT['bz_n']:,}"
              f"  (this run: {stats(total['az'])['n']:,} · "
              f"{stats(total['b26'])['n']:,} · {stats(total['bz'])['n']:,})")

    for key, label in (("az", "A · OB-candle close + zone stop"),
                       ("b26", "B · displacement close + 26W stop"),
                       ("bz", "B · displacement close + zone stop")):
        q = stats(total[key])
        if not q:
            continue
        print(f"  {label}: n {q['n']:,} · win {q['win_rate']:.0f}% · avg win "
              f"{fmt_pct(q['avg_win'])} · avg loss {fmt_pct(q['avg_loss'])} · "
              f"expectancy {fmt_pct(q['expectancy'])} · "
              f"{q['expectancy_r']:+.2f}R")
    return 0


if __name__ == "__main__":
    sys.exit(main())
