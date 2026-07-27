# Setup Guide — flat layout

Every Python file sits in the repo root. The **only** folder is
`.github/workflows/`, which GitHub requires.

Follow these in order. About 20 minutes, plus a long unattended build at Step 4.3.

No server, no VPS, no PC left running — it all runs on GitHub for free.

---

# Part 1 — Telegram bot (5 minutes)

## Step 1.1 — Create the bot

1. In Telegram, search for **@BotFather** (blue checkmark).
2. Send `/newbot`.
3. Give it a **name** — anything, e.g. `My Breakout Alerts`.
4. Give it a **username** — must be unique and end in `bot`,
   e.g. `hamad_breakout_alerts_bot`.
5. BotFather replies with a token like:

```
8123456789:AAHk9__ExampleTokenStringGoesHere_xyz
```

**Copy it now.** That's your `TELEGRAM_BOT_TOKEN`. Anyone with it can post as
your bot, so never commit it or paste it in a chat.

## Step 1.2 — Message your bot first

Open your new bot and press **Start** (or send it `hi`).

Not optional. Telegram blocks bots from messaging a user who has never contacted
them — skip this and alerts fail with
`403: bot can't initiate conversation with a user`.

## Step 1.3 — Get your chat ID

Search **@userinfobot**, press Start, and it replies with your numeric ID
(e.g. `587654321`). That's your `TELEGRAM_CHAT_ID`.

Or via the API — paste in a browser with your token filled in:

```
https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
```

Look for `"chat":{"id":587654321,...}`.

**For a group instead:** add the bot to the group, send a message there, then
call `getUpdates`. Group IDs are negative and start with `-100`,
e.g. `-1001234567890`. Keep the minus sign.

## Step 1.4 — Test it

Fill in both placeholders and open in a browser:

```
https://api.telegram.org/bot<YOUR_TOKEN>/sendMessage?chat_id=<YOUR_CHAT_ID>&text=working
```

You should see "working" in Telegram and `{"ok":true,...}` in the browser.

| Response | Cause |
| --- | --- |
| `401 Unauthorized` | Token wrong or has a typo |
| `400 chat not found` | Chat ID wrong, or you skipped Step 1.2 |
| `403 bot can't initiate conversation` | You skipped Step 1.2 |

Don't continue until this works — everything downstream depends on it.

---

# Part 2 — Dhan credentials (5 minutes)

## Step 2.1 — Generate an access token

1. Log in to [web.dhan.co](https://web.dhan.co) on desktop.
2. Click your **profile icon** (top right).
3. Open **DhanHQ Trading APIs** / **API Access**.
4. **Generate Access Token**, name it e.g. `breakout-scanner`.
5. Copy the long `eyJ...` JWT — that's `DHAN_ACCESS_TOKEN`.

## Step 2.2 — Note your Client ID

Same page. A number like `1000000003` — that's `DHAN_CLIENT_ID`.

## Step 2.3 — Enable the Data API subscription

Check the **Data API** subscription is active on that page.

It's a paid add-on and this project **cannot work without it** — historical
candles and quotes both need it. Without it every request returns `401`/`403`.

## Step 2.4 — Know the expiry

> ⚠️ **Dhan tokens expire, typically after ~30 days.**

When yours does, runs fail and you get a Telegram warning. Repeat Step 2.1 and
update the secret. Worth a monthly calendar reminder.

---

# Part 3 — GitHub (10 minutes)

## Step 3.1 — Unzip

```
weekly-breakout-flat/
├── .github/workflows/    scan.yml, snapshot.yml, tests.yml   ← the only folder
├── scan.py               indicators.py   strategy.py
├── build_snapshot.py     dhan.py         telegram.py
├── config.py             state.py
├── test_*.py             (4 test files)
├── config.yaml           requirements.txt
├── README.md             SETUP.md
├── .env.example          .gitignore
└── push_to_github.sh
```

## Step 3.2 — Create the repository

**Make it private** — it will hold your snapshot data.

### Option A — Website upload

1. [github.com/new](https://github.com/new) → name it → **Private** → **Create**.
2. Click **uploading an existing file**.
3. Open the unzipped folder, select all files **inside** it (Ctrl+A / Cmd+A),
   drag them in.
4. **Commit changes**.

Because everything is flat, this works cleanly — with one exception:

> ⚠️ **`.github` is hidden** (leading dot), so browsers usually skip it.
> Without it you get **no automation at all**.
>
> Reveal hidden files to check it's there:
> macOS Finder `Cmd + Shift + .` · Windows Explorer View → Show → Hidden items
>
> If it didn't upload, create the files manually:
> 1. **Add file → Create new file**
> 2. Filename, typed exactly: `.github/workflows/scan.yml`
>    (typing `/` makes GitHub create the folders)
> 3. Paste that file's contents from your unzipped copy → **Commit**
> 4. Repeat for `snapshot.yml` and `tests.yml`

### Option B — The included script (avoids the above entirely)

Create an **empty** private repo (don't tick "Add a README"), copy its URL, then:

```bash
cd path/to/weekly-breakout-flat
bash push_to_github.sh https://github.com/YOUR_USERNAME/YOUR_REPO.git
```

It verifies the workflow files are staged and refuses to push without them.

If asked for a password, GitHub no longer accepts account passwords — create a
**Personal Access Token** at
[github.com/settings/tokens](https://github.com/settings/tokens)
(classic → tick **repo**) and paste that instead.

### How to know it worked

The repo root shows your `.py` files, `config.yaml`, and a **`.github`** folder.
Inside `.github/workflows/` there are three `.yml` files. The **Actions** tab
lists *Intraday Scan*, *Weekly Snapshot*, *Tests*.

**Empty Actions tab = workflows didn't upload.** Nothing else will work until
they do.

## Step 3.3 — Confirm the tests pass

**Actions** tab → a **Tests** run should finish green in under a minute. That
proves the upload is complete and the code is intact.

## Step 3.4 — Add your four secrets

**Settings** → **Secrets and variables** → **Actions** → **New repository secret**.

Names are case-sensitive and must match exactly:

| Name | Value |
| --- | --- |
| `DHAN_CLIENT_ID` | from Step 2.2, e.g. `1000000003` |
| `DHAN_ACCESS_TOKEN` | the `eyJ...` token from Step 2.1 |
| `TELEGRAM_BOT_TOKEN` | from Step 1.1 |
| `TELEGRAM_CHAT_ID` | from Step 1.3 |

No quotes, no trailing spaces. A stray space is the most common cause of `401`.

## Step 3.5 — Allow workflows to commit

The scanner saves `state.json` back to the repo so it remembers which alerts
already went out.

**Settings** → **Actions** → **General** → **Workflow permissions** →
**Read and write permissions** → **Save**.

Skip this and you get `403` push errors and possibly repeated alerts.

---

# Part 4 — First run

## Step 4.1 — Small snapshot, to prove credentials work

Don't start with the full universe.

1. **Actions** → **Weekly Snapshot** → **Run workflow**.
2. In **limit**, type `50`.
3. Run it.

**Expected:** green tick, and Telegram says "✅ Weekly snapshot rebuilt".

| Error in the log | Fix |
| --- | --- |
| `auth failed (401)` | Token wrong/expired, or Data API inactive (Step 2.3) |
| `DHAN_ACCESS_TOKEN is empty` | Secret missing or misnamed (Step 3.4) |
| `403` on the commit step | Workflow permissions (Step 3.5) |
| No Telegram message | Re-run the Step 1.4 browser test |

## Step 4.2 — Test the scanner end to end

1. **Actions** → **Intraday Scan** → **Run workflow**.
2. **force** = `true` (runs even when the market is shut).
3. **heartbeat** = `true` (summary even with zero signals).

You should get:

```
📊 Scan complete
Bar: 27-Jul 14:35 IST · 12s
Universe 50 → gate-eligible 3 → signals 0
```

**Zero signals is correct and expected.** These are rare weekly breakouts. The
heartbeat proves the whole chain works: Dhan → strategy → Telegram.

## Step 4.3 — Build the full universe

1. **Actions** → **Weekly Snapshot** → **Run workflow**.
2. Leave **limit** completely **blank**.

> ⏳ **This takes 2–5 hours.** Normal, not a bug — 5 years of daily history for
> ~2,400 symbols at Dhan's 5 requests/second cap. It's unattended and runs once
> a week. Close the tab; Telegram pings you when it's done.

## Step 4.4 — You're live

- **Weekly Snapshot** rebuilds Monday 08:15 IST, before the open.
- **Intraday Scan** runs every 5 minutes, 09:15–15:30 IST, Mon–Fri.
- Alerts arrive within ~5 minutes of the breakout candle closing.

```
🟢 BUY — RELIANCE
NSE_EQ · 5m close 1,512.40 > 26W level 1,498.00 (+0.96%)
🕐 27-Jul-2026 11:20 IST

Weekly conditions — 13/13 PASS
• 52W high  1,498.00
• EMA20 1,455.20 > EMA50 1,402.75
• RSI 72.54 (prev 63.09)
• MACD hist 12.345
• Vol 1.34Cr > avg 90.00L
• Wk open 1,463.10 · Day open 1,489.55
```

---

# Part 5 — Living with it

## Monthly: refresh the Dhan token

Redo Step 2.1, then **Settings → Secrets and variables → Actions** →
`DHAN_ACCESS_TOKEN` → **Update secret**.

## Changing strategy settings

Edit `config.yaml` in GitHub (pencil icon) and commit. Every value maps 1:1 to a
Pine input, so the Python entry stays in step with your chart.

| Setting | Effect |
| --- | --- |
| `strict_entry: false` | Alert on any 26W break, ignoring the other conditions |
| `req52: true` | Also demand a close above the 52W high (fewer alerts) |
| `one_per_week: false` | Allow repeat alerts in the same week |
| `exchange_segments: [NSE_EQ, BSE_EQ]` | Add the BSE cash segment |
| `max_symbols: 200` | Cap the universe (faster runs while experimenting) |

After changing anything under `universe:`, re-run **Weekly Snapshot**.

## Pausing alerts

**Actions** → **Intraday Scan** → **⋯** → **Disable workflow**.

## No alerts for days?

Usually correct — the 13-condition scan is demanding and often matches nothing
for a week or more.

To confirm the plumbing is alive, run **Intraday Scan** manually with
`force: true` and `heartbeat: true`.

## A missed run is safe

GitHub's scheduler is best-effort and can run late. Every run replays the whole
week from Monday's first 5-minute candle, so a late run still finds the exact
same breakout candle. Lateness can delay an alert; it cannot lose one.

---

# Local use (optional)

```bash
pip install -r requirements.txt
cp .env.example .env          # fill in, then: set -a; source .env; set +a

python build_snapshot.py --limit 50
python scan.py --force --heartbeat
python scan.py --force --symbols RELIANCE,TCS
python -m pytest -q           # 59 tests, no network needed
```

Set `dry_run: true` in `config.yaml` to log alerts instead of sending them.

---

# Quick reference

| What | Where |
| --- | --- |
| Run manually | **Actions** → pick workflow → **Run workflow** |
| Why a run failed | **Actions** → red run → click the failed step |
| Change credentials | **Settings** → **Secrets and variables** → **Actions** |
| Change strategy | Edit `config.yaml`, commit |
| Alert history | `state.json` in the repo |
| Weekly levels in use | `weekly_snapshot.csv` in the repo |

**Reminder:** alerts only — this never places an order. Check the chart before
you act on a signal.
