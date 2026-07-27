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

**Flat layout:** every Python file is in the repo root; `.github/workflows/` is
the only folder. Generated files (`weekly_snapshot.csv`, `state.json`,
`universe.csv`) are written to the root too.

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
| `one_per_week` | first entry per symbol per week only |
| `universe.exchange_segments` | `[NSE_EQ]`, add `BSE_EQ` for BSE cash |
| `universe.series` | `[EQ, BE]` |
| `runtime.prefilter` | stage-1 quote funnel; disable to force the full path |

---

## Things worth knowing

- **Scheduling is best-effort.** GitHub cron can run late under load. Because
  every run replays the whole week, lateness delays an alert but never loses it.
- **Weekly bars are Monday-anchored** from daily candles, matching TradingView's
  `"W"` resolution. A market holiday shortens the week, exactly as on the chart.
- **State is committed back to the repo** (`state.json`) so `one_per_week`
  survives across runs. If Telegram delivery fails, state is deliberately *not*
  saved, so the next run retries the alert.
- **Rate limits** honoured: Data 5/s, Quote 1/s, 100k requests/day.
- Alerts fire on a **5-minute candle close**, not on every tick — same as the
  indicator.

## Disclaimer

For research and education. Signals are not investment advice; verify against
your chart before acting. Markets carry risk of loss.
