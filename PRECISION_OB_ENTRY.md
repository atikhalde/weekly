# Precision OB Entry — "buy the OB candle itself"

_The rule from `precision_ob_backtest.py` (generated 06-Oct-2026 20:23), run
live as a spectator of the stage-2 scanner: `precision_ob_entry.py`, its own
two workflow slots and its own state file. This document maps every clause of
the description onto the code that implements it, so the match can be checked
line by line rather than taken on faith._

---

## 1. The rule, as implemented

| Description | Implementation |
|---|---|
| **Event** — the first precision order block born after the 26W breakout (the live scanner's `first_zone_after_breakout` rule) | `first_ob_after_breakout()` — the earliest OB born on/after the cycle's c02-dated breakout session, merged from the state's live `zones` list and its pruned `closed_events` |
| **…the OB candle on/after the breakout (the pullback candle) 3,907 53%** | `evaluate_rule()` row 1 — the origin candle is re-resolved from the bars with the scanner's own `ob_precision.find_origin()`; `origin_session < breakout_session` is excluded |
| **excluded: OB candle predates the breakout (zone born on the breakout, origin the day before) 3,450 47%** | same row, with the description's own wording in the exclusion reason (the SMSPHARMA shape) |
| **Entry** — the OB candle's close, taken only when it closes ABOVE the 26-week breakout level | `evaluate_rule()` row 2 + `Rule.entry_a = origin.close`; `origin.close <= level` → "OB close back below the 26W level" (1,629 excluded) |
| **Target** — the first swing high after the breakout; "the highest high printed between the breakout session and the entry session (sell limit)" | `Rule.target = max(high for bars with breakout_session <= session <= origin_session)` — it excludes the displacement bar and always includes the OB candle's own high, so the limit can never sit behind the entry (the backtest's 0.9x median R:R) |
| **Stop** — the 26W breakout level | `Rule.stop = brk["level"]`, the same candle-derived c02 level the backtest dates its cycles from |
| **Time stop** — 90 trading sessions | `RuleSettings.time_stop_sessions = 90`; counted from the entry (the OB candle's) session. The bar walks count real sessions; the zero-call daily check falls back to a weekday count |
| **Fills** — "through the target fills at the open (better), through the stop at the open (worse); a session touching both resolves by its open, else counts conservatively as a loss" | `resolve_bar()` — the exact policy, in order: gap through the target → open; gap through the stop → open; both touched → conservative loss at the stop; else limit/stop. The bar walks always have the opens; the state-only pass uses `as_of_open` when the scanner carries it |
| **Round-trip cost 0.22%** | `RuleSettings.round_trip_cost_pct`, subtracted in every exit report (net from A, net from B) |

The zone definition is **not** re-implemented: the job calls
`cfg.ob_precision.params()`, so the displacement and origin thresholds
(`min_rvol 1.8`, `min_range_atr 1.20`, `min_body_frac 0.55`, `min_clv 0.72`,
`structure_len 8`, `origin_search 8`, `allow_neutral`/`neutral_body`) are the
ones the live scanner runs. The OB this job trades cannot drift from the OB
`ob_tap_scan.py` draws.

## 2. The backtest these alerts quote

Funnel (replayed 2021-26, eod2 NSE daily, split/bonus adjusted, live config
defaults):

```
26W breakout cycles replayed                10,748
.. formed a post-breakout precision OB       8,768   82%
   first post-breakout OBs (the events)      7,357  100%
   .. OB candle on/after the breakout        3,907   53%
   .. OB close above the 26W level           2,278   58%
   .. classified                             2,272   31%
   excluded: OB candle predates the breakout 3,450   47%
   excluded: OB close back below the level   1,629   22%
```

The trade (n = 2,272): **88.5% win · avg win +7.27% · avg loss −4.16% ·
expectancy +5.77% net · +1.62R · median win completes in 1 session · profit
factor 14.13 · best +78.3% · worst −34.9%.**

## 3. The one honest problem: entry A is one bar old

The OB candle only **becomes** a precision OB when the displacement bar after
it closes. Its close is therefore already history at the first moment the rule
can be evaluated — the backtest's own hindsight check:

| entry | stop | n | win | avg win | avg loss | expectancy | exp R |
|---|---|---|---|---|---|---|---|
| **A** OB candle close | 26W level | 2,272 | 89% | +7.3% | −4.2% | +5.77% | +1.62 |
| **B** displacement close | 26W level | 6,624 | 83% | +3.0% | −5.3% | +1.32% | +0.60 |

This job does not paper over that. **Every alert carries both prices**, the
plan and its R:R are computed from A (the rule), the fillable version is shown
from B, and the exit report gives the net from each. Three alert moments cover
the three ways to act:

| slot | mode | what it says | what is fillable |
|---|---|---|---|
| **09:44 UTC / 15:14 IST** | `intraday` | "today's bar is displacing; traced back with the scanner's own `find_origin`, the OB candle is X" — the rule's structural rows are already checked against that OB candle | **entry B** — today's close, before the bell |
| **10:40 UTC / 16:10 IST** | `postclose` | the confirmed event: the complete plan, the trade recorded, exits checked | nothing today (both closes are printed) — the ledger entry and the live orders from here on |
| optional, same intraday slot | `candidate_alerts: true` | "today's candle is the OB candle — if a displacement follows, entry A is today's close" | **entry A** — the rule's own price, at the risk of acting before the order block exists |

The candidate mode is **off by default**: at that moment the OB does not exist
yet, and post-breakout red/neutral candles above the level are common (the
funnel finds ~3,900 OB candles in five years of the whole universe, roughly
1.5/day). The confirmation is the rule; the candidate is a bet that the
displacement comes.

## 4. Alert behaviour you can rely on

- **First-run safety.** A rule event older than `catchup_sessions` (default 3)
  is marked seen silently. Deploying the job cannot dump weeks of history into
  the chat.
- **The event is recomputed, not inherited.** The committed state is a
  trigger, not the truth: its zone lists are pruned, and a cycle whose first
  OB has since died would otherwise look like it starts at a *later* order
  block. The fetched bars are replayed with the scanner's own port
  (`ob_precision.replay`), so "the first OB after the breakout" means the
  indicator's first, not the first one still in the cache.
- **A late event never quotes a plan.** GitHub cron skips slots (BUG 55). A
  rule event inside the catch-up window is bar-walked first: if the trade
  already resolved (the median win does so in one session), the alert is a
  `MISSED` notice with the realised outcome — not a plan nobody can take.
- **Exits are gap-aware and never silent.** Every open trade is checked on
  every postclose pass against the state's latest closed bar. That is normally
  free; when a trade actually resolves there, one history call makes the fill
  exact (the state carries no open, and "gapped through" fills at the open),
  and any session a skipped run missed is bar-walked, so a stop that filled on
  a day nobody ran is reported with *that* session and price.
- **The book survives a bad record.** One malformed trade costs itself its
  exit check, never the others'.
- **Delivery is at-least-once.** State (dedupe keys, the book) is persisted
  only after Telegram confirms; a failed send is retried next run.

## 4b. Timing

The intraday slot is worth exactly as much as it lands before 15:30 IST — that
is when entry B (today's close) is still available. GitHub's `schedule:` is
best-effort and can run late (the repo's BUG 55); the run still works after the
bell, but the fillable moment is gone and the postclose pass becomes the
authoritative one. The workflow also accepts a `repository_dispatch` of type
`precision_ob_entry`, so a cron-job.org ping at **15:12 IST** makes the slot as
reliable as `scan.yml` and `btst.yml` (`CRON_JOBS.md`).

## 5. Cost shape

| pass | market-data calls |
|---|---|
| `postclose` | **none**, unless a rule event fires — then one daily-history call for that symbol (origin candle + target) |
| `intraday` | one bulk quote for the whole waiting list, plus one daily-history call per name whose live bar is actually displacing |

The 5-minute `ob_tap_scan.py` remains the only heavy consumer. `--no-data`
defers rule events instead of evaluating them (they are retried next run, never
marked seen).

## 5b. The backtest, reproduced

`precision_ob_backtest.py` measures the shipped rule (`births_from_bars`,
`evaluate_rule`, `walk_bars` — the alert's own functions) over split/bonus
adjusted NSE daily candles from eod2 (through 2026-09-25, the report's data):

```bash
git clone --depth 1 https://github.com/BennyThadikaran/eod2_data
python precision_ob_backtest.py --data-dir eod2_data/daily
```

| | this run | report |
|---|---|---|
| first post-breakout OBs | 7,430 | 7,357 |
| .. OB candle on/after it | 3,903 | 3,907 |
| .. OB close above the level | 2,284 | 2,278 |
| excluded: OB candle predates | 3,527 | 3,450 |
| excluded: close back below | 1,613 | 1,629 |
| censored at the cutoff | **6** | **6** |
| A trades | 2,284 | 2,272 |
| win rate | 88.1% | 88.5% |
| avg win / avg loss | +6.86% / −4.14% | +7.27% / −4.16% |
| expectancy | +5.37% | +5.77% |
| median R:R at entry | **0.9x** | **0.9x** |
| best / worst | **+78.27% / −34.88%** | **+78.3% / −34.9%** |
| median win completes | 1 session | 1 session |
| B + 26W stop | 6,710 | 6,624 |
| B + zone stop | 7,395 | 7,323 |

Every row lands within ~2% of the report; several match exactly. The one row
that does not is the raw cycle count (~9,300 vs 10,748): the report's
accounting counts ~1,900 cycles that never produce an order block, and no
combination of lock width, cross session or universe reproduced that number —
while every row downstream of it does.

## 5c. The live scanner's state, compared

The same rule run against the scanner's own committed state and eod2 bars, for
all 291 names on the waiting list:

* **177** of 206 comparable names: the first post-breakout OB is the *same
  session* as the state's earliest zone.
* **28** of the other 29: the replay finds an *earlier* OB, and the state's
  later one is present in the replay too — the state simply lost the earlier
  zone. `ob_tap_scan.prune_closed_events` trims the persisted event list by
  design ("a full replay emits an event for EVERY order block the window
  contains... the next refresh recomputes them anyway"), so dead zones leave
  no trace. This is exactly why this job recomputes the event from the bars
  instead of trusting the state's zone list.
* **1**: the state's birth is not in the eod2 bars at all (the CSV set ends
  2026-09-25).

## 6. Running it

```bash
python precision_ob_entry.py --mode postclose --dry-run   # what would be sent
python precision_ob_entry.py --mode digest                # the open book
python precision_ob_entry.py --mode explain --symbols SMSPHARMA
python precision_ob_entry.py --mode explain --symbols AARTIIND --no-data
python -m pytest test_precision_ob_entry.py test_precision_ob_backtest.py -q
```

`explain` prints the funnel state for a name — cycle date, first OB, which
filter decided, and (with data) the rule's numbers — so a symbol that did not
alert can be explained rather than guessed at.

## 7. What this job is not

- It places **no orders**. It sends plans; the orders are yours.
- It reads `ob_precision_state.json` **read-only** and writes only
  `precision_ob_entry_state.json`.
- The sample is in-sample and descriptive: the configuration was chosen from
  the same history it is evaluated on, and no portfolio simulation (slots,
  correlation, capital) exists. Overlapping signals count as independent
  trades in the backtest.
- The 26W level used as a stop is **close-based** in the live scanner's zone
  invalidation and order-based here (any low at/below the level stops the
  trade), exactly as the backtest states.

## 8. Verification

`test_precision_ob_entry.py` pins each clause of the description, so a future
edit that drifts from the backtest fails the suite:

- the funnel: the first post-breakout OB, the origin-on/after-breakout row,
  the "close above the level" row — including a test named for the SMSPHARMA
  shape;
- the rule's four numbers on a synthetic cycle: entry = OB candle close, stop =
  the 26W level, target = the highest high breakout→OB candle, 90-session time
  stop;
- the fill policy, branch by branch (gap through target/stop, both touched,
  nothing touched);
- the target window never including the displacement bar, and never sitting
  below the entry;
- first-run safety, catch-up, the MISSED notice for an already-resolved late
  event, dry-run and failed-delivery semantics;
- the intraday forming path: the OB candle resolved with `find_origin`, and
  the four reasons it stays silent (level, pre-breakout origin, cycle already
  has its OB, stale/holiday quote).
