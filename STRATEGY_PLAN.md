# The Breakout Pullback Strategy — The Full Plan

_Built from five backtests, 2021–2026, ~2,560 NSE symbols, eod2 adjusted
daily data: the Tap 1 study, the recovery study, the target-trade study, the
precision-OB entry study, and the no-filter control. 180,000+ replayed
trades. Every rule below exists because a number said so._

---

## 1. The idea in one paragraph

After a stock breaks out of a 26-week range, it pulls back. The pullback
bottoms with a **precision OB** (a bearish candle followed by a
displacement: strong close above the 8-bar structure, rvol ≥ 1.8, range ≥
1.2 ATR, body ≥ 55%, close near the high). The trade buys that pullback and
sells into the **swing high the pullback started from**, with the **26W
breakout level as the stop**. There are two entry moments — the day the OB
is born, and the first retest of its zone (Tap 1) — run as one book with
one skeleton of exit orders.

## 2. The shared skeleton (both legs, non-negotiable)

| Rule | What | Why (the number) |
|---|---|---|
| **Target** | Sell limit at the swing high since the breakout (highest high breakout → entry) | The tested target; mean win +13.3% (tap), +7.3% (OB day) |
| **Stop** | Stop order at the **26W breakout level** | 26W stop +2.14%/trade vs tight zone stop +0.70% (swept in ~70% of races) |
| **Time stop** | Exit at the close after **90 sessions** if neither order filled | 90d beats 30/60d at every configuration (+0.45/+0.57/+0.70 zone-stop tap) |
| **The level filter** | Only enter if the entry close is **above the 26W level** | Above: 33% win / +0.68% vs below: 19% / +0.43% (tap); below-level OB closes excluded |
| **Costs** | Assume 0.22% round trip | The backtest's flat cost |

## 3. The two legs

### Leg 1 — the OB-day entry (fast, stable, your intraday entry)

**Trigger:** the scanner reports a new precision OB born after the breakout
(the 🎯 alert — it already exists in `ob_tap_scan.py`).

**Entry:** at ~15:15 IST on the displacement day, when the intraday
checklist holds:
- price is above the 8-bar structure high, and holding near the session high
- volume is running ≥ ~1.8× the 20-day average
- (miss the day → take the close, or skip — never chase later)

**Proven numbers (n = 6,624, 2021–26):**

| | win | avg win | avg loss | expectancy | R |
|---|---|---|---|---|---|
| At the displacement close (floor) | 83% | +3.0% | −5.3% | **+1.32%** | +0.60 |
| At the structure-break level (ideal intraday fill, ceiling) | 65% | +6.1% | −4.7% | **+2.13%** | +0.08 |

- Wins complete fast: median **1 session**. This leg turns capital over quickly.
- **Positive in every year 2021–26** (+0.52% to +2.21%) — the steadiest
  signal found, including 2022 and 2025 where the tap entry lost money.
- Realistic expectation for a 15:15 entry: **between +1.3% and +2.1%** per
  trade. Where you land depends on how early in the bar you qualify it.

### Leg 2 — the Tap 1 entry (bigger, slower)

**Trigger:** the scanner's Tap 1 alert (first tap of the first post-breakout
OB — already live), **only if the tap closes above the 26W level**.

**Entry:** at/near the tap session's close.

**Proven numbers (n = 3,617, 2021–26):**

| win | avg win | avg loss | expectancy | R | median win |
|---|---|---|---|---|---|
| 47% | +13.3% | −8.0% | **+2.14%** | **+1.09** | 11 sessions |

- Fewer, bigger, slower trades. Loses in weak regimes (2022 −1.69%, 2025
  −2.02%) — which is exactly why Leg 1 runs beside it.

### Both legs on one symbol = staggered

Half on the OB day, half at Tap 1: average entry between the two, same
target/stop, risk budget split. The pullback pays you for the second half.

## 4. Position sizing and the book

- **1R = distance from entry to the 26W level** (median ~6.6% on OB-day
  entries). Risk **0.5–1% of capital per trade**.
- Worked example: ₹10L capital, 1% risk = ₹10,000. Stop 6.6% away →
  position ≈ ₹1.5L (15% of capital). A full stop loses ~₹8-10k; the average
  win returns +3% (Leg 1) to +13% (Leg 2) on the position.
- **Book: 10–15 concurrent positions.** The replay produces ~3–5 new
  qualified signals per day at the peak (the live feed is a subset — the
  momentum gate), so ranking decides what you take.
- **Ranking (from the by-type tables):** 1) displacement rvol ≥ 2.5
  (+6.22% vs +4.77%), 2) bearish origin candle (+6.21% vs +4.40% neutral),
  3) breakout age > 45 days (+7.18%), 4) origin deeper than adjacent
  (+6.69% vs +5.19%). For taps: no-sweep taps win most (57%).
- Rebalance nothing. Orders do the work: limit at the target, stop at the
  level, calendar exit at 90 sessions.

## 5. Proven wrong — do not do these

1. **Never buy the pullback candle intraday without the displacement
   confirmed.** The control: the same buy-stop at the structure break on
   every post-breakout day = **155,881 trades, 9% win, −4.92%/trade**. The
   displacement filter is the entire edge. Your "perfect candle at 15:15"
   cannot be known at 15:15 — the marker appears 1–8 sessions later by
   definition (61% next day).
2. **No tight zone stop** — swept ~70% of the time, cuts expectancy to a
   third.
3. **No entries below the 26W level.**
4. **No 30/60-day time stops** — they amputate the big winners.
5. Don't skip the OB-day leg because its per-trade number is smaller — it is
   the only leg positive in every single year.

## 6. Honest limits

- In-sample: the rules were read off the same history they're evaluated on.
- Today's universe (survivorship: 2021-era delistings absent).
- Per-trade numbers are proven; the **portfolio level (slots, ranking,
  concurrency, correlation clusters) is not yet simulated** — that is Step 1
  below, before real money.
- Costs flat 0.22%; slippage on illiquid names not modelled.

## 6b. Edge cases the live alert handles

Normal days were never the problem; these are the days that are not normal.

- **A skipped run (BUG 55).** GitHub's scheduler drops slots. Exits are
  therefore bar-walked across every session between the last check and the
  state's latest closed bar, so a stop that filled on a day nobody ran is
  reported with *that day's* session and the stop price — not silently
  carried until the next extreme. The walk costs one daily-history call, so
  it is only taken when a run was actually missed or the target is still
  unknown: a normal daily pass makes **zero** market-data calls.
- **A market holiday.** A bulk quote carries no date, and on a weekday
  holiday the feed re-serves the previous session's numbers unchanged —
  which reads as a fat displacement candle forming. The intraday pass
  applies the scanner's own three-way test (high, low and last all equal
  the last closed bar) and skips the name, and additionally requires the
  scanner to have refreshed that symbol's context today.
- **A spent cycle.** Leg 2 is the cycle's **first** tap and nothing else —
  that is the trade the 47% / +2.14% backtest measured. Once Tap 1 has
  happened the cycle is spent, and the intraday pass no longer sends a
  "forming" heads-up for a later tap on any zone in that cycle.
- **A malformed record.** One bad trade in the book costs itself its exit
  check, never the other open trades theirs.

## 7. Build sequence

1. **Portfolio simulation of this exact plan** (both legs, 10–15 slots,
   ranking, ₹-sized equity curve and drawdown) → PDF like the others.
   Backtest-only, touches nothing live. **This is the gate for real money.**
2. **Scanner alert upgrade** (needs your OK — scanner code, diff shown
   first, nothing pushed without approval): every OB/Tap alert carries the
   trade plan — entry, stop (26W level), target (swing high), risk%, size —
   plus a **15:15 IST intraday displacement alert** ("OB forming now:
   structure broken, volume on track") so Leg 1 is a one-tap decision.
3. **Paper trade 30–60 signals** (the repo's ab_paper infrastructure)
   — measure real fills vs the backtest's fill assumptions, especially the
   15:15 entry quality.
4. **Size up gradually**: quarter risk for the first month, half for the
   second, full only if live expectancy lands within ~30% of backtest.

---

_All numbers from `tap1_target_trade_report.pdf` (Tap 1 leg),
`precision_ob_trade_report.pdf` (OB-day leg, entry-timing check, no-filter
control) and the studies behind them. Generated 07-Oct-2026._
