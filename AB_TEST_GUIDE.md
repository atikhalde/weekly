# A/B Paper Trading — how to run the week

Built 28-Jul-2026. Tests: **200 passing** (was 194).

Your idea was the right one. Two models, same live data, same rules, and the
market decides in a week — on data neither of us can overfit to.

---

## The two models

| | **A · GATED** (my recommendation) | **B · MOVER** (yours) |
|---|---|---|
| c09 weekly volume | **ON** — full 13-row gate | **OFF** — enter at the raw cross |
| Volume conviction | none extra | rvol > 10× **and** bar_rvol > 10× |
| Volatility floor | ATR ≥ 1.0% | ATR ≥ 1.0% |
| Chase guard | none | reject if > 3% above the level |
| Exit | first 5m close below 9-EMA | run to **+3R**, stop to breakeven at +1R |

**Identical in both, by design:** universe, capital (₹1,00,000/trade), the stop
(entry candle low − 0.02%), intraday square-off at 15:15, costs 0.22% round trip,
cross entries only. The *only* differences are the entry filters and the exit —
which is the whole point.

I added 6 regression tests that fail the build if anyone gives one model a
different stop, different costs, or an overnight hold.

---

## Historical baseline — your model already wins

Same 12 weeks, whole NSE cash universe, both models run identically:

| | A · GATED | B · MOVER |
|---|---|---|
| Trades | 52 (4.3/wk) | **110 (9.2/wk)** |
| Win rate | 21.2% | **24.5%** |
| Profit factor | 0.41 | **0.91** |
| Avg per trade | −0.73% | **−0.10%** |
| Avg R | −0.25 | **+0.06** |
| Reached +5% MFE | 8% | **15%** |
| Net P&L | −₹37,713 | **−₹10,499** |

**B beats A by +0.63% per trade.** Your thesis — don't let c09 delay the entry,
demand real volume, then give the move room — is measurably better than mine.
A is significantly negative (t = −2.21); **B is statistically indistinguishable
from breakeven** (t = −0.35), which is a genuinely different situation.

B also caught MONARCH at 09:25 for **+3R (+4.84%)**, the exact trade you
described. A entered it at 10:05 and got nothing.

So the live week is really asking: *can B cross from breakeven into profit?*

---

## Running it

Daily, after close (automatic via `.github/workflows/ab.yml`, 16:15 IST):

```bash
python ab_paper.py --from-snapshot --days 1 --telegram
```

Standings at any time, no data fetch:

```bash
python ab_paper.py --report-only
```

Dry run without Dhan:

```bash
python ab_paper.py MONARCH TMB SENCO --source yahoo --days 5
```

The ledger `ab_ledger.csv` is **append-only and de-duplicated** on
(model, symbol, signal date, signal time) — verified by test. Re-running the
same day adds nothing, so the workflow is safe to trigger manually.

---

## Reading the result honestly

The report prints `VERDICT: too early` until **each** side has ≥10 closed
trades, because below that the numbers are noise. Expect roughly 45 trades for
B and 20 for A in a normal week, so one week should clear the bar for B and be
marginal for A.

Watch these three, not just P&L:

1. **Avg R** — the fairest cross-model comparison, since both risk the same amount.
2. **Reached +5% MFE** — is B actually finding your explosive movers?
3. **Never broke entry low** — your core thesis. Historically 8% (A) vs 5% (B),
   both far below the 74% seen among *known* big movers. If B's number is high
   this week, your "genuine moves don't hunt the SL" claim is live-confirmed.

One caution: at ~10 trades/week a single +8% outlier swings B's average by
~0.8%. **One week will be indicative, not conclusive.** Two to three weeks gets
you to something you could actually size up on. I'd rather say that now than
have you act on a lucky Tuesday.

---

## Files

- `models.yaml` — both model definitions, with the measured evidence in comments
- `ab_paper.py` — the A/B engine (shares `paper.py`'s validated simulation core)
- `.github/workflows/ab.yml` — daily run, commits the ledger, Telegram standings
- `ab_ledger.csv` — created on first run

Upload `UPDATE_THESE.zip`. Nothing in the live scanner changed: `config.yaml`
still ships the conservative settings, and the A/B is a separate measurement
track that never places orders.
