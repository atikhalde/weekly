# Weekly Breakout Scanner → Telegram Alerts

Python port of the Pine indicator **"Weekly Breakout Scanner + 5m Entry [Chartink] v3"**.
Runs on GitHub Actions, reads market data from DhanHQ, and fires a Telegram alert
on the **exact 5-minute candle** the indicator marks with `BUY`.

Alerts only — no orders are placed anywhere in this codebase.

---

## How the entry stays identical to the chart

The whole design exists to protect one property: the alert must land on the same
candle as the arrow on your chart, never one bar late.

**1. The week is frozen before it starts.**
Every value the entry depends on comes from *closed* weekly bars — the 26W/52W
breakout levels, the EMA/RSI/MACD states, the volume average. This is the fix
described in your script's own header comment: a week is either eligible or it
isn't, and nothing developing intraday can shift the trigger bar.

**2. Indicators are re-implemented to Pine's exact seeding rules.**
`ta.ema` seeds from an SMA, `ta.rsi` uses Wilder's RMA, `ta.macd` feeds a
partially-NaN line into the signal EMA. Generic library versions get these
wrong at the third decimal, which is enough to move a trigger. Each has an O(1)
`step()` for the developing bar, and tests assert `step()` equals a full
recomputation to 1e-9.

**3. The week is replayed bar by bar, every run.**
The scanner never asks "is it breaking out right now?" — it replays the week
from Monday's first 5-minute candle, reproducing Pine's `close[1]`,
`tookThisWeek` and `sawCrossThisWeek`. So a cross is a genuine *cross* (the
previous bar must be below), and `onePerWeek` behaves exactly as on the chart.

The practical payoff: **a missed cron slot cannot lose a signal.** If GitHub's
scheduler is late or a run fails, the next run still replays the same week and
reports the same breakout candle. Verified by
`test_rescanning_the_same_week_is_idempotent`.

### One deliberate deviation

Condition 12, `market cap > 1000`, is **off**. Pine gets shares outstanding from
`request.financial()`; Dhan has no equivalent endpoint. You asked to skip it, so
c12 auto-passes and the alert reports 13/13.

To re-enable: set `use_mcap: true` in `config.yaml` and add `mcap.csv`
(`symbol,mcap_in_crore`).

Note also that `strict_entry` defaults to **true** here, while the Pine input
defaults to false. True is what reproduces the full 13-condition scan; set it to
false in `config.yaml` for a pure level break.

---

## Architecture

Roughly 2,400 NSE `EQ`/`BE` symbols cannot each get a 5-year history call inside
a 5-minute window. The work is split in two:

```
Weekly  ──  build_snapshot.py  ──  5y daily candles → weekly bars
(Mon 08:15 IST)                    → frozen indicator state
                                   → weekly_snapshot.csv

Every 5m ──  scan.py  ── stage 1: bulk quotes, 1000 symbols/request
(mkt hours)              drop anything not above its frozen 26W level
                       ── stage 2: for survivors only, pull this week's
                          5m candles and replay them through the Pine logic
                       ── Telegram
```

The stage-1 filter compares LTP against a **week-constant** level, so it can
never discard a candle the indicator would have flagged. If a quote request
fails it fails *open* — those symbols fall through to the full check rather than
being silently skipped.

Typical run: ~2,400 symbols → 3 quote requests → a handful of candidates →
well under a minute.

| File | Role |
| --- | --- |
| `indicators.py` | Pine-exact EMA/RMA/RSI/MACD/SMA + O(1) incremental states |
| `strategy.py` | The 13 conditions, entry gate, week replay |
| `dhan.py` | DhanHQ v2 client, rate limiting, IST timestamps |
| `scan.py` | Intraday scanner (5-min cron) |
| `build_snapshot.py` | Weekly preparation job |
| `state.py` | Cross-run de-duplication |
| `telegram.py` | Alert formatting and delivery |
| `config.py` | Settings loader |
| `precision.txt` | The Pine v6 order-block indicator this repo ports (source of truth) |
| `ob_precision.py` | Pine-exact port of `precision.txt`: displacement → order block → tap |
| `ob_tap_scan.py` | Stage-2 scanner (5-min cron): taps of the weekly-breakout waiting list |
| `strategy_alert.py` | The two-leg breakout-pullback strategy alert (15:12 + 16:05 IST): complete trade plans + exit tracking; exits bar-walk any session a skipped run missed, a market holiday's stale quote fires nothing, and a spent cycle gets no second heads-up; reads `ob_precision_state.json` read-only, own state in `strategy_alert_state.json` |

**Flat layout:** every Python file is in the repo root; `.github/workflows/` is
the only folder. Generated files (`weekly_snapshot.csv`, `state.json`,
`universe.csv`, `ob_precision_state.json`) are written to the root too.

---

## Stage 2 — precision order-block taps (`ob_tap_scan.py`)

The weekly scanner answers *"which stocks broke their 26-week high?"* — on the
week of 07-Sep-2026 that was 67 names, and most went nowhere. Stage 2 answers
the follow-up:

> **Of the stocks the weekly scanner already flagged, which one has pulled back
> into a precise institutional order block and is being defended right now?**

```
scan.py marks state.json  ──►  WAITING LIST  (this scanner's own state file)
                                     │
              daily bars ──► volume-confirmed displacement
                                     │
                          nearest bearish origin candle
                                     │
                                     ▼
                       🎯 PRECISION OB  (zone frozen for life)
                                     │  armed once price departs by 1×ATR
                                     │  and the zone is ≥ 3 sessions old
                                     ▼
                  live intraday low taps the pre-order entry
                                     │
                                     ▼
                        🟠 TAP 1  ──►  Telegram
```

The waiting-list feed uses the weekly scanner's canonical first alert for each
26-week breakout cycle, directly from its persistent cycle record. This remains
available after `state.json` prunes weekly rows, so an active cycle cannot vanish
or be re-anchored to a later weekly repeat. Later zones also cannot generate a
fresh symbol-level Tap 1 after a zone already tapped on a prior session in that
breakout cycle. Identical order blocks from the same origin are deduplicated even
when front-running places their entries above the zone. This does **not** change
the weekly scanner's original breakout trigger or filters.

#### The 26W date on the alert is derived, the anchor is not

Those two are different numbers, and only one of them is a price event.

`state.py`'s migration seeds `breakout_alerts` from the **earliest weekly row
still in `state.json`**, and `prune()` throws weekly rows away after six weeks.
So on a state file that was ever rebuilt, that anchor is "the oldest alert row
that had not been pruned yet" — and in the committed file, 265 of 268 records
landed in the same five days. It is a retention artefact. NITINSPIN's tap alert
read `📅 Weekly breakout 2026-08-18` when the stock had in fact cleared its
26-week high on **Mon 04-May-2026**, six weeks into the move — so the alert
dated the setup to a breakout that had not happened yet when its own order block
was born.

`ob_tap_scan.py` therefore derives the date from the candles the zone replay
already holds, at zero extra API cost, using the weekly scanner's own
definition: for week *k* the frozen level is `max(high[k-26:k])` and condition
`c02` is `close_k > level`. The answer is the **first** such cross inside the
26-week cooldown window — the one that started the move, not the newest of the
many new highs a grinding trend prints along the way. Daily close stands in for
the 5-minute close the alert fired on, which can only ever be a day late.

The anchor (`breakout_bar`) is deliberately left alone. It is a cycle boundary:
it decides which order block a name is armed on and when its cycle expires, and
moving it under a symbol that is already armed would change a live setup. So the
alert header and the digest sort/label on the derived date, and everything that
decides *what to trade* still reads the anchor. Records already resolved — a name
that tapped, was invalidated or expired — are corrected too, once, on the next
refresh; the pass is self-limiting and a name the candles cannot date is marked
answered rather than re-asked every five minutes.

The zone logic is a **Pine-exact port of `precision.txt`** (*"Institutional OB —
Precision Tap & Pre-Order"*, Pine v6). Every input of that indicator is a key in
the `ob_precision:` block of `config.yaml`, at the file's own defaults.

### It does not repaint

An intraday alert that later disappears is worse than no alert, so three rules
are enforced (and tested):

| Rule | Why |
| --- | --- |
| A zone is born **only on a closed daily bar** | a developing bar's shape is not known yet; `barstate.isconfirmed` in Pine |
| Zone geometry is **frozen at birth** — top, bottom, entry, stop never move | re-deriving them from a shifting window is what makes indicators repaint |
| Live thresholds use the **last closed ATR**, not the forming bar's | deviation #1: otherwise the tap level drifts as the day develops |

A tap is judged from the session's actual low, and a session low can only go
lower — so once a tap has fired it can never be retracted. Live events are
stamped *"live intrabar touch — not close-confirmed"* so you can tell them from
a closed-bar touch.

### The waiting list outlives `state.json`

`state.py` prunes weekly rows after six weeks. A name stays on the waiting list
until tapped, invalidated, or its **26-week breakout cycle expires**. The list
lives in this scanner's own `ob_precision_state.json` and keeps the level it
broke — the weekly snapshot is overwritten every Monday, so that number is
otherwise unrecoverable.
The scanner seeds from first-alert cycle records across the full 26-week lock,
not just the six weeks of weekly rows currently retained by `state.py`; this is
what keeps names such as OAL from disappearing before the OB stage can track
them. `state.json` is **read-only** here; only `scan.py` writes it.

Once every order block a name produced has been invalidated or exhausted
(5 touches, or a close below the structural stop), it is marked `invalid` and
drops off the active list. The record is kept for audit.

### What it costs

Daily history is one call **per symbol**, so it is paid once per session; every
later run in the day judges the live tap from **one bulk quote** for the whole
list, and symbols with no live zone are not quoted at all.

| Run | Cost |
| --- | --- |
| first of the session | 1 daily-history call per waiting symbol (capped by `max_refresh_per_run`, rolls over) |
| every later run | 1 bulk OHLC request per 1,000 quoted symbols |
| after 15:35 IST | one more history call per symbol, so today's closed bar can create/confirm zones the same evening |

The first run backfills every week `state.json` still retains (~370 names),
spread over several runs by the cap. Historical order blocks are **not** dumped
into the chat: only closed-bar events from the last `event_lookback_days` are
alertable, and live taps are always new.

### Alerts

Two kinds, both restricted to waiting-list names, both de-duplicated across runs
by `(symbol, zone, kind, tap number)`:

- `🎯 PRECISION OB — SYMBOL` — a new order block, with zone top/bottom, the
  pre-order entry, the structural stop, ATR/RVOL, and the weekly breakout it
  came from
- `🟠 TAP 1 — SYMBOL` — price tapped that entry, with the session low, LTP, the
  stop, and the raised next-entry level

The headers are deliberately unlike `scan.py`'s `🟢 BUY —`, so the two systems
can never be confused in the same chat. `alert_kinds` also accepts `approach`,
`confirm` and `invalid`; they are computed but off by default because they are
chatty.

### The daily waiting-list digest

Alerts only ever name the symbols that *did* something. A name sitting armed and
untouched for three weeks is invisible in the chat — and it is exactly the one
worth tracking by hand — so the whole active list goes out twice a session:

- **pre-open plan, 09:10 IST** — yesterday's frozen levels, read before the bell.
  It costs no API calls at all: that branch never builds a `DhanClient`, because
  everything it prints was already computed by the post-close replay.
- **post-close recap, 15:35 IST** — today's closed bar, including any zone born
  on it. When a large list is still spreading its refreshes over runs
  (`max_refresh_per_run`), the recap defers to the 15:40 run rather than print
  half-stale levels — but never past the last run of the day, or one broken
  symbol would cost the list entirely.

Newest **actual** breakout first (the derived 26W date, not the recorded alert
bar), one line per name:

```text
📋 PRECISION WAITING LIST — post-close recap
25-Sep-2026 15:35 IST · 367 waiting · 42 armed · +3 new today · 1 tapped today
newest breakout first · entry/stop are the newest live zone's, frozen at birth
PGIL · brk 24-Jun @986.35 >985.05 · OB 24-Jun entry 955.05 stop 915.87
SMSPHARMA · brk 11-Sep @455.05 >447.80 · OB 11-Sep entry 406.18 stop 399.22 · tapped 17-Sep
AAKASH · brk 18-Aug @55.00 · no live zone yet
```

A name with no zone yet says so instead of being omitted, and one whose history
cannot be fetched says `no daily history`, so a hole in the list is always a real
hole.

The real list is ~266 names and ~15k characters, which is more than one Telegram
message, so the digest pages itself at 3800 characters and repeats the header on
every page with a `(k/n)` suffix — `telegram._split` would chop it for us, but
blindly, and everything after the first chunk would arrive with no header at all.
`digest_max_rows: 0` (the default) sends every name; a cap drops the *oldest*
breakouts and says how many were dropped. `daily_digest: false` turns both slots
off, and `digest_after_close` / `digest_before_open` / `digest_pre_open_at`
control them one at a time.

If a name could not be replayed since the bell, the recap says so in the header
(`⚠️ N name(s) not replayed since the bell`) rather than quietly mixing an older
session's levels into today's list. The pre-open plan carries no such warning —
yesterday's levels are exactly what it is for.

Both slots ride the existing `*/5 3-10 * * 1-5` UTC cron, so **no new schedule is
needed** — 03:40 UTC *is* 09:10 IST, and 10:05/10:10 UTC are the recap. Each slot sends once per day, and the marker
in `ob_precision_state.json` is written only after Telegram confirms delivery, so
a failed send is retried by the next run instead of being swallowed. An explicit
`--digest` / `--digest-only` sends off-schedule and is marked `manual`, which
cannot spend either of the day's real slots.

Both digest paths harvest the waiting list from `state.json` first — three
committed files, still no API calls — so pressing the dispatch button on a cold
start, or a Monday morning after a weekend the scheduler skipped, prints the real
list instead of an empty one. A `--symbols` debug run never sends or spends the
day's slot: its cache is a subset by definition, and printing that as the recap
would suppress the real one. And a malformed `digest_pre_open_at` disables the
pre-open slot with a logged error rather than raising, because that knob is read
on the scan path too — a typo in a tracking aid must not take the tap alerts down.

### Bar parity with the chart

`bars_from_frame()` drops two kinds of row before the replay, because TradingView
would have drawn no bar for either and every rule in `precision.txt` is written
against the chart's own `bar_index`, not against the calendar:

- **a session with no trades** (`volume == 0`). The feed still returns a row,
  filled with the previous close as `open=high=low=close`, so it looks perfectly
  valid. Counting one inflates `age = bar_index - born` for every zone by a bar —
  and `age >= minAge` is what gates the *first* tap, so Tap 1 fires a whole
  session early. Volume only disqualifies a row when the feed actually reports
  it: a frame with no volume column means *unknown*, not *no trades*.
- **a row the feed could not price** (missing or non-finite OHLC) — the all-null
  placeholder Yahoo emits for a market holiday.

This was found on real data: SMSPHARMA broke out on 2026‑09‑11 and the chart put
Tap 1 on **2026‑09‑17**, while a port that counted the no‑trade session of
2026‑09‑14 put it on 2026‑09‑16. That symbol had four such sessions in one year.
`test_smspharma_daily.csv` is the real year of bars, kept as a fixture: one test
walks it the way the scanner does and asserts the 17th, another keeps the
phantom rows and asserts the wrong answer, so loosening the filter fails loudly.
Dropping the rows also moves ATR (a Wilder RMA over the bar series) onto the same
series the chart computes it from.

A second fixture, `test_pgil_daily.csv`, covers the case where the date survives
but the levels do not. PGIL displaced on 2026‑06‑24 and its no‑trade session fell
on 2026‑06‑26; the first tap came on **2026‑07‑08**, nine bars after birth, so
counting the phantom only ages the zone 10 instead of 9 and both clear
`minAge = 3`. The frozen numbers still change — ATR 42.84 → 40.59, entry
955.05 → 954.93, stop 915.87 → 915.97 — and those are the levels a live trade is
sized against. Bar parity is therefore not a one-symbol quirk: either the series
matches the chart's bars or the alert prints the wrong prices, and only sometimes
is the date wrong too.

### Running it

```bash
python ob_tap_scan.py                       # normal 5-minute run
python ob_tap_scan.py --force --heartbeat   # out of hours, with a summary
python ob_tap_scan.py --symbols SMSPHARMA   # one name
python ob_tap_scan.py --refresh-only        # rebuild the zone cache, no live pass
python ob_tap_scan.py --digest              # send the waiting list now
python ob_tap_scan.py --digest-only         # ...and do nothing else: no API calls
```

`--digest-only` is also a `workflow_dispatch` input on **Precision OB Tap (5m)**,
so the list can be pulled from the Actions tab at any hour.

`ob_precision.enabled: false` in `config.yaml` switches the whole thing off
without touching any other job. The GitHub workflow is
**Precision OB Tap (5m)** (`ob_tap.yml`), on the same `*/5 3-10 * * 1-5` UTC
schedule as the intraday scan; it commits `ob_precision_state.json` and never
`state.json`.

### Failure behaviour

A run exits `0` even when individual symbols fail — one delisted or badly-mapped
name must not stop the rest of the list, it just logs a warning. The exception is
a **total data outage**: when two or more history refreshes are due and every one
of them fails, the run exits `3` so the workflow's `Notify on failure` step puts
a message in Telegram. That happens at most once a day (`data_outage_on` in the
state file), because a five-minute cron would otherwise turn a single outage into
~78 notices; the flag clears on the first run that fetches data again, which
re-arms it for later in the day. A run that refreshed nothing simply because
everything was already cached today says nothing either way and stays green.

---

## Setup

Full step-by-step instructions are in **SETUP.md** — Telegram bot, Dhan
credentials, GitHub secrets, and the first run.

Quick version:

```bash
# 1. push to a private GitHub repo
bash push_to_github.sh https://github.com/YOUR_USERNAME/YOUR_REPO.git

# 2. add 4 secrets in Settings > Secrets and variables > Actions:
#    DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

# 3. Settings > Actions > General > Read and write permissions

# 4. Actions > Weekly Snapshot > Run workflow  (limit: 50 to test)
# 5. Actions > Intraday Scan  > Run workflow  (force + heartbeat to test)
# 6. Actions > Weekly Snapshot > Run workflow  (limit blank = full universe)
```

⚠️ A **Dhan Data API subscription** is required, and access tokens expire about
every 30 days.

---

## Local use

```bash
pip install -r requirements.txt
cp .env.example .env          # fill in, then: set -a; source .env; set +a

python build_snapshot.py --limit 50     # small snapshot
python scan.py --force --heartbeat      # scan regardless of clock
python scan.py --force --symbols RELIANCE,TCS
python -m pytest -q                     # 59 tests, no network needed
```

Set `dry_run: true` in `config.yaml` to log alerts instead of sending them.

---

## Tuning

`config.yaml` mirrors the Pine inputs one-for-one.

| Key | Meaning |
| --- | --- |
| `strict_entry` | `false` = pure level break, `true` = full 13-condition scan |
| `gate_source` | `live` (matches the table, Pine default) or `closed` (non-repainting) |
| `defer_entry` | fire later in the week if the gate turns true after the cross |
| `req52` | also require a close above the 52W level |
| `one_per_week` | prevents duplicate alerts for a symbol within the same week |
| `runtime.breakout_cooldown_weeks` | cross-week 26W breakout lockout (default 26); after expiry, a fresh 26W cross is required |
| `universe.exchange_segments` | `[NSE_EQ]`, add `BSE_EQ` for BSE cash |
| `universe.series` | `[EQ, BE]` |
| `runtime.prefilter` | stage-1 quote funnel; disable to force the full path |

---

## Things worth knowing

- **Scheduling is best-effort.** GitHub cron can run late under load. Because
  every run replays the whole week, lateness delays an alert but never loses it.
- **Weekly bars are Monday-anchored** from daily candles, matching TradingView's
  `"W"` resolution. A market holiday shortens the week, exactly as on the chart.
- **State is committed back to the repo** (`state.json`) so the same-week
  duplicate guard and the 26-week per-symbol breakout lock survive across runs.
  The lock starts at the first alert in a breakout cycle; after its 26-week
  expiry, a new 26W cross is required. If Telegram delivery fails, state is
  deliberately *not* saved, so the next run retries the alert.
- **Rate limits** honoured: Data 5/s, Quote 1/s, 100k requests/day.
- Alerts fire on a **5-minute candle close**, not on every tick — same as the
  indicator.

## Disclaimer

For research and education. Signals are not investment advice; verify against
your chart before acting. Markets carry risk of loss.
