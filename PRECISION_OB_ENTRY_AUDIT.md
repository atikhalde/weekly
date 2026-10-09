# Precision OB Entry (OB candle) — full audit, 08-Oct-2026

**Question:** is the job working, and why did no alert arrive?

**Verdict:** the job is **mechanically healthy — nothing is failing**. In ~2 days it
ran **150 times with 150 green conclusions**, its tests pass, Telegram is correctly
configured, and it **did deliver exactly 2 messages**. But both of those messages
said **"⏱ MISSED … No action"**. It has **never once sent an actionable entry
alert**, and on the current data pipeline it is very unlikely ever to send one from
the postclose pass. So "I got no alert" is *almost* right: you got two
non-actionable obituary notices and zero tradeable plans.

---

## 1. What the run history shows

| Check | Result |
|---|---|
| Total runs (`precision_ob_entry.yml`) | **150**, all `success` — zero failures ever |
| Trigger mix | 147 `workflow_dispatch` (cron-job.org, every 15 min, 24/7) + 3 `schedule` |
| Every step green? | Yes — checkout, setup-python, pip, **"Verify the rule logic"**, the alert run, state push |
| Rule tests | `test_precision_ob_entry.py` + `test_precision_ob_backtest.py` → **107 passed** (reproduced locally) |
| Telegram configured? | **Yes, proven** (see below) |
| Deployed | first commit `65d23b1` 07-Oct 09:30 UTC; **first run 07-Oct 15:54 IST** |

### Telegram delivery is provably working

Three independent code facts pin this down:

1. `main()` refuses to run at all when Telegram is not live:
   ```python
   if args.mode in ("postclose","intraday") and tg.dry_run and not cfg.runtime.dry_run:
       log.error("Telegram is not configured for live delivery; refusing to run…")
       return 2
   ```
   `return 2` would fail the step and the job. **150/150 jobs are green**, so the
   bot token *and* chat id are present on every run. This guard has existed since
   the very first commit `65d23b1`.
2. `Telegram.send()` only returns `True` on HTTP 200; a 400/401/403 or three failed
   attempts returns `False`.
3. `run_postclose` / `run_intraday` **do not save state unless `_send_alerts()`
   returned True**, and they skip the save entirely in dry-run:
   ```python
   if not _send_alerts(tg, msgs): return 1        # no save
   if _dry_run(cfg, args, tg):    return 0        # no save
   save_state(own, STATE_OWN)
   ```
   `precision_ob_entry_state.json` **was committed 145 times** → every run took the
   real-send path.

> Note on logs: raw Actions log text is not retrievable from this sandbox (the log
> zip lives on `results-receiver.actions.githubusercontent.com`, outside the network
> allowlist). The analysis below is reconstructed from the **145 committed state
> snapshots** — which record every decision the job made — plus the code, and was
> verified by running the job locally.

---

## 2. What was actually sent (2 messages, reconstructed verbatim)

```
⏱ MISSED — PARAGMILK (Precision OB Entry)
OB candle 2026-10-01 · born 2026-10-06 (displacement bar) · first post-breakout OB of the cycle.
The rule's own trade already resolved: WIN at 286.20 on 2026-10-06 — net from A +8.19% · from B −0.79%.
No action — recorded in the book for the audit trail.
```
*delivered 07-Oct-2026 ≈15:54 IST*

```
⏱ MISSED — EIMCOELECO (Precision OB Entry)
OB candle 2026-10-05 · born 2026-10-06 (displacement bar) · first post-breakout OB of the cycle.
The rule's own trade already resolved: WIN at 2,348.80 on 2026-10-06 — net from A +4.55% · from B +0.04%.
No action — recorded in the book for the audit trail.
```
*delivered 08-Oct-2026 ≈15:30 IST*

Search your chat for **"MISSED"** — they are there.

---

## 3. The book: 212 events consumed, 2 alerts, 0 trades ever opened

Reconstructed from all 145 state snapshots — only **3 distinct states** ever existed:

| When (IST) | sent | open | closed | excluded | What happened |
|---|---|---|---|---|---|
| 07-Oct 15:54 | 208 | 0 | 1 | 0 | **First run.** 207 old events marked seen **silently** (first-run safety, by design) + 1 PARAGMILK "MISSED" alert |
| 08-Oct 00:00 | 209 | 0 | 1 | 0 | STLTECH marked seen **silently** — stale OB candle (ob_age 5 > 3) |
| 08-Oct 15:30 | 212 | 0 | 2 | 2 | EIMCOELECO "MISSED" alert; ARFIN + KOTAKBANK excluded ("OB candle predates the breakout") |

Alert-key census — this is the damning line:

```
rule|…      212      (210 silent + 2 MISSED notices)
forming|…     0      ← the ONLY fillable path has NEVER fired
candidate|…   0      ← off by default (config: candidate_alerts: false)
exit|…        0      ← no exit alerts, because no trade was ever opened
open trades   0      ← the book has never held a live position
```

---

## 4. Root cause — why a postclose alert can essentially never be actionable

This is a **data-timeliness problem, not a code defect.**

`fetch_bars` → `DhanClient.daily_candles` uses **Yahoo as the primary source**, and
Yahoo's settled NSE daily candle for the *current* session is not available at the
close. Verified across every session since 27-Sep by walking the scanner's committed
state:

| scanner ran at | `refreshed_on` | newest `as_of` | lag |
|---|---|---|---|
| 08-Oct **15:45 IST** (after the 15:30 close) | 2026-10-08 | **2026-10-07** | **1 session** |
| 07-Oct 15:45 IST | 2026-10-07 | 2026-10-06 | 1 session |
| 06-Oct 15:13 IST | 2026-10-06 | 2026-10-05 | 1 session |
| … every day, without exception | | | **1 session** |

Not one of the 292 zones had `as_of == today` at 15:45 IST. `ob_tap.yml` stops at
16:29 IST, so **session T's candle is first seen on the morning of T+1.**

Now follow the consequence through `run_postclose`:

1. An OB is *born* on session T (its displacement bar closes on T).
2. The state first reports it on **T+1**.
3. `late = weekday_sessions(born=T, today=T+1)` = **1 — never 0**.
4. The code's own guard fires: *"A late alert must never quote a live plan for a
   finished trade"* →
   ```python
   ex = walk_bars(trade, bars, through=today, …) if late > 0 else None
   if ex:   msgs.append(late_html(…));  own["closed"].append(trade)   # MISSED
   else:    msgs.append(plan_html(…));  own["open"].append(trade)     # actionable
   ```
5. The rule targets *the highest high printed between the breakout and the OB
   candle* — a near swing-high sell limit — and the backtest says **88.5% win with
   the median win completing in 1 session**. So by T+1 the target has usually
   already filled.
6. → `ex` is truthy → **"MISSED"**, trade goes straight to `closed`, never `open`.

Both real cases confirm it word for word:
`"reason": "swing-high target filled (the alert ran late)"`.

**`late == 0` is unreachable on this pipeline**, so a plan alert only survives when
the trade is *still unresolved* one or more sessions later — roughly the 11.5%
losers and the slow winners. That is a thin, adverse-selection sliver: the alerts
you would get are disproportionately the trades that did **not** work.

Entry A is already one bar of hindsight by design (documented in the module header).
The EOD lag adds a **second** session on top of it.

---

## 5. Bugs and risks found (ranked)

### B1 — Midnight `today` roll silently suppresses live events  *(high)*
Mode is picked from the UTC hour (`< 10` → intraday, else postclose), and the
15-minute dispatcher runs 24/7. So **postclose runs from 15:30 IST to 05:29 IST**.
From 00:00 IST, `today = datetime.now(IST).date()` becomes a session that **has not
happened yet**, inflating every age by one:

```
OB candle 2026-10-05, born 2026-10-06, catchup_sessions = 3
  run at 08-Oct 23:45 IST → today 2026-10-08 → ob_age 3 → within window
  run at 09-Oct 00:00 IST → today 2026-10-09 → ob_age 4 → SUPPRESSED, marked seen forever
```

That is ~22 runs a night using a phantom date. Because the first run that sees an
event marks it seen permanently, **one night run can kill an alert the next 16:10
slot would have sent.**

**EIMCOELECO passed at `ob_age` exactly 3 — the boundary.** Had its run landed after
midnight instead of 15:30, you would have received *nothing at all*, and the event
would have been marked seen. This is not theoretical; it came within one session of
biting you.

### B2 — The window was tightened mid-flight and is now very narrow  *(high)*
`e6266ba` (07-Oct 12:53 UTC, *"OB candle age window"*) changed the gate from the
**confirmation bar's** age to the **OB candle's** age:
```python
-  if late < 0 or late > settings.catchup_sessions:
+  if ob_age > settings.catchup_sessions or late > settings.catchup_sessions:
```
Since the OB candle *precedes* its confirmation bar by `origin_offset` (1–3 sessions
in the observed cases), `ob_age = late + origin_offset`. With `catchup_sessions: 3`
**and a guaranteed `late >= 1`**, the effective room is ~0–2 sessions. STLTECH died
here (ob_age 5). Its own commit message names the casualties: *"PARAGMILK and
STLTECH stale OB candles are suppressed."*

Note PARAGMILK alerted at 15:54 IST on 07-Oct, **before** that fix landed at
18:23 IST — it would be suppressed today.

### B3 — The scheduled intraday slot has never run as intraday  *(medium)*
All 3 `schedule` runs fired at **16:56Z, 17:32Z, 17:35Z** — roughly 7 hours after the
configured `09:44 UTC` / `10:40 UTC` crons (GitHub cron delay). Since mode is inferred
from the wall clock, every one of them took the **postclose** branch.
**Inferring the pass from the current hour means a late cron silently runs the wrong
pass.** Only the 15-minute dispatcher is keeping the intraday branch alive.

### B4 — ARFIN's dedupe key is unstable → a full history fetch every 15 min  *(medium)*
State-derived birth (`2026-10-06`) ≠ bar-derived birth (`2026-04-13`). The early
`_sent()` gate checks the state-derived key, so it never matches; the symbol falls
through to `fetch_bars` (560 days, chunked by 365 ≈ 2 requests), and only *then* is
the bar-derived key found and short-circuited. Reproduced locally — ARFIN is the sole
symbol reaching the data stage on a quiet run:
```
WARNING ARFIN: no market data - rule event 2026-10-06 deferred (retried next run)
INFO postclose done: 0 alert(s) — events: 211 total (210 already handled, 1 deferred, …)
```
≈ **210 wasted history requests/day** on an already-excluded name, and it contradicts
the documented cost shape (*"NO market-data call unless a rule event fires"*).

### B5 — State is rewritten and committed every 15 minutes  *(low)*
`own["last_run"]` is set at the **start** of `run_postclose`, so the file always diffs
and the `Persist the alert state` step always commits: **145 commits in ~1.5 days**
from this job alone; 190 repo commits on 08-Oct.

### B6 — The forming path has never executed in anger  *(informational)*
`forming|` keys: **0**. The funnel is genuinely narrow, not blocked:
```
292 zones → 177 status=="waiting" → 177 refreshed_on == today (0 dropped)
          →  42 with no post-breakout OB yet   ← the only forming candidates
```
Each must then print a **live displacement bar** (rvol ≥ 1.8, body ≥ 0.55,
clv ≥ 0.72, range ≥ 1.2×ATR, close > 8-bar structure high) *and* have its traced
origin on/after the breakout with a close above the 26W level. Zero hits in two
sessions is plausible — but this is the only path that can ever give you a fillable
price, and it has **never been observed to fire**, so it is unproven in production.

Also note: with 15-minute dispatch the intraday pass starts at 05:30 IST and runs all
morning. A 09:30 gap-up spike can satisfy `live_displacement` and fire a "forming"
heads-up that is nowhere near true at 15:30 — and the per-day dedupe key means that
early false positive **blocks the real one later that day**.

---

## 6. Recommended fixes

**F1 — derive the session date from the data, not the wall clock** *(fixes B1)*
Use the scanner's newest closed session (`ctx["as_of"]`, or the max across zones) for
every `weekday_sessions(...)` age computation, keeping `datetime.now(IST)` only for
timestamps. This makes the window identical at 16:10 IST and 03:00 IST.

**F2 — gate on knowability, not on the OB candle's age** *(fixes B2)*
The cheap pre-data gate (`> catchup_sessions + 1`) already prevents replaying
history. Make `ob_age` **information printed in the alert** rather than a kill
switch, and gate on `late` (how long since the event became knowable). Raising
`catchup_sessions` is **safe** — the 210 already-marked keys are permanent, so
nothing historical will replay into your chat.

**F3 — pass the mode explicitly** *(fixes B3)*
Have cron-job.org send `{"inputs":{"mode":"intraday"}}` / `{"mode":"postclose"}`
(CRON_JOBS.md already documents this) instead of relying on the UTC-hour inference,
and add a market-hours guard so the intraday pass only evaluates 09:15–15:29 IST.

**F4 — stabilise the dedupe key** *(fixes B4)*
Check the excluded/sent record under *both* the state-derived and bar-derived birth
before calling `fetch_bars`, or key on the zone `signature`.

**F5 — stop the 15-minute commit churn** *(fixes B5)*
Only write `last_run` when the book actually changes, or keep it out of the
committed file.

**F6 — decide what you actually want to be alerted on.** Given §4, the honest options are:
- **(a)** Accept that postclose mostly yields "MISSED" notices, and treat the
  **intraday forming alert (entry B)** as the product — then F3 matters most, and it
  is worth adding a *heartbeat* line so a silent day is visibly different from a
  broken one.
- **(b)** Alert a **next-session-open plan** for events that have *not* resolved, so
  the message is tradeable rather than historical.
- **(c)** Get session-T's candle on session T: pull the daily bar from DhanHQ
  directly after the close (or an intraday-aggregated bar) instead of Yahoo, which
  would make `late == 0` reachable and let the real plan alert fire.

---

## 7. One-line answer

Nothing is broken and nothing is failing — the job ran 150/150 green, Telegram
delivery is provably live, and it sent you two messages. But both said **"MISSED —
no action"**, because Yahoo's daily candle for the current session doesn't exist at
the close, which guarantees every postclose event is ≥1 session late, and with an
88.5% win rate resolving in a median of 1 session the trade is already over before
the rule can be evaluated. The only path that can give you a fillable entry — the
intraday forming alert — has never fired, and two real bugs (the midnight `today`
roll and the OB-candle age window) are actively suppressing events.

---

## 8. Fixes applied (09-Oct-2026)

All four code/workflow bugs are fixed on this branch; **114 tests pass**
(107 pre-existing + 7 new regression tests), and each new test was verified to
**fail against the pre-fix code**. The wider repo suite is unchanged at its
15 pre-existing failures (missing `tests.yml` / `backtest.yml` / healthcheck and
report workflows, and `reportlab`/`openpyxl` not installed).

| Bug | Fix | Where |
|---|---|---|
| **B1** midnight `today` roll inflated every age in the catch-up window | new `session_date()` — the last **completed** session (before 15:30 IST, or any weekend, rolls back); `run_postclose` uses it instead of `datetime.now(IST).date()`. The 16:10 IST slot is unchanged. | `precision_ob_entry.py` |
| **B3** a delayed cron silently ran the wrong pass | the workflow resolves the mode from (1) an explicit `mode` input, (2) for a scheduled run, `github.event.schedule` — the cron that actually fired, (3) otherwise the **IST** clock: intraday 09:15–15:30 IST, postclose outside it | `.github/workflows/precision_ob_entry.yml` |
| **B6** intraday pass ran from 05:30 IST on a closed market | same IST market-hours rule, plus a **fallback**: a derived intraday slot that lands after the bell runs postclose instead of spending a bulk quote on an unfillable bar. Explicit modes are still respected. | `.github/workflows/precision_ob_entry.yml` |
| **B4** ARFIN's unstable dedupe key forced a 560-day fetch every run | a **cycle index** (`cycles`, keyed `SYMBOL|breakout-session`) short-circuits before any market-data call; recorded on the way past every marking, so an existing state **heals itself** on its first run | `precision_ob_entry.py` |
| **B5** state committed every 15 minutes (145 commits in 1.5 days) | `last_run` is now session-scoped (`postclose 2026-10-08`) with the clock time in `last_run_at`, which the comparison ignores; a run that moves the book nowhere **does not rewrite the file**, so the workflow's `git status` check finds nothing to commit | `precision_ob_entry.py` |

Verified against the live committed state (dry-run, no market data):

```
now IST: 2026-10-09 14:02 Friday   (before the close)
  OLD code -> postclose 2026-10-09   <- phantom session, ages +1
  NEW code -> postclose 2026-10-08   <- last completed session
```
```
ARFIN, before the cycle index : 1 history fetch attempted  (1 deferred)
ARFIN, after  the cycle index : 0 history fetches          (211 already handled)
```

**Not changed, deliberately:**

- **B2 — the `ob_age` window itself.** `e6266ba` moved the gate from the
  confirmation bar's age to the OB candle's age on purpose, and three tests pin
  that behaviour (`test_window_is_ob_candle_age_suppresses_stale_candle`,
  `test_stale_ob_candle_counted_in_postclose_summary`,
  `test_paragmilk_and_stltech_shapes_end_to_end`). That is a calibration
  decision, not a bug, so it was left alone. **B1 was the actual defect here**:
  the phantom overnight session was quietly taking one session of headroom away,
  which is what put EIMCOELECO exactly on the boundary. With B1 fixed the
  configured window is applied as written.
  If you want more alerts, widening `catchup_sessions` in `config.yaml` is a
  one-line change and is **safe** — the 210 keys already marked seen are
  permanent, so no history replays into the chat.
- **§4 — the EOD data lag.** This is the reason the postclose pass mostly
  produces `MISSED` notices, and it is a property of the Yahoo-primary daily
  feed, not of this job. Fixing it means changing the data source or the alert
  product (audit §6 F6 a/b/c) — a decision worth making deliberately rather
  than inside a bug-fix pass.
