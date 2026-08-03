# Cron jobs — what they are and how to set them up

## What a cron job is

A **cron job** is just "run this thing at this time, automatically, forever."
The name is old Unix jargon; the idea is a kitchen timer that never forgets.

Your scanner is a set of Python scripts. Something has to *press the button* at
15:20 every weekday. That something is a scheduler. You have two available:

| | who runs it | reliability |
|---|---|---|
| **GitHub `schedule:`** | GitHub, inside the repo | **best-effort** — runs late or skips entirely under load |
| **cron-job.org** | a third-party website | fires on time, every time |

GitHub's own docs call `schedule:` best-effort. In practice it queues your job
behind everyone else's when the platform is busy. That is not a bug you can
fix — it is what the free tier is.

**cron-job.org** is a free website that does one thing: at a time you choose,
it sends an HTTP request to a URL you choose. GitHub exposes a URL that means
"start this workflow now." Wire the two together and your job starts on time.

---

## Why this matters (03-Aug-2026, measured)

| workflow | what GitHub's scheduler did |
|---|---|
| `btst.yml` | ran **15:48** — 28 min late, *after* the 15:30 close |
| `report.yml` | **never ran** |
| `ab.yml` | **never ran** |

The BTST scan has a **ten-minute entry window** (15:20 → 15:30). Late is not
"degraded", it is worthless — you cannot buy at a close that already happened.

`scan.yml` was already on cron-job.org and fired fine all day. That is the
proof this works.

---

## CORRECTION to the previous version of this file

The last revision told you to create a **new classic PAT with `repo` scope**
and use the `repository_dispatch` endpoint. **Ignore that.** It works, but it
is the harder path and it needs a second, broader token.

All three workflows also accept **`workflow_dispatch`** — the exact same
mechanism your working `scan.yml` job already uses. So:

> **Use the token you already have. Copy your existing scan job three times
> and change two fields.**

No new token, no new permissions, no new payload format.

*(Why the difference: the `workflow_dispatch` API needs a fine-grained token
with **Actions: read and write**, which is what you already made. The
`repository_dispatch` API needs **Contents: write** instead — a different
permission and, for a classic token, a much broader one. Same result, more
work. The workflows accept both, so use the easy one.)*

---

## Setup — three jobs, ~5 minutes total

Go to <https://console.cron-job.org/jobs>. For each row below: open your
existing **`NSE 5m breakout scan`** job → **Clone**, then change only the
**Title**, **URL** and **Schedule**.

| # | Title | URL (append to `https://api.github.com/repos/atikhalde/weekly/actions/workflows/`) | Time IST |
|---|---|---|---|
| 1 | BTST 15:20 scan | `btst.yml/dispatches` | **15:18** |
| 2 | Paper trading report | `report.yml/dispatches` | **16:00** |
| 3 | A/B paper ledger | `ab.yml/dispatches` | **16:15** |

Everything else stays **identical to the scan job**:

- **Method** `POST`
- **Headers**
  ```
  Accept: application/vnd.github+json
  Authorization: Bearer github_pat_YOUR_EXISTING_TOKEN
  X-GitHub-Api-Version: 2022-11-28
  Content-Type: application/json
  ```
- **Body** `{"ref":"main"}`
- **Schedule** → set timezone **Asia/Kolkata** first, then Days of week =
  **Mon–Fri**, Hours/Minutes per the table, Days of month = Every, Months = Every
- **Notifications** → enable *on failure* (a dead token then emails you instead
  of failing silently)

Save → **TEST RUN** → expect **HTTP 204** (success, empty body). Then check the
repo's **Actions** tab to confirm the run appeared.

### Why 15:18 and not 15:20

The request is instant, but the GitHub runner needs ~45–60 s to boot and
install dependencies before `btst.py` starts. Firing at 15:18 puts the actual
scan at roughly 15:19–15:20, inside the window.

---

## Troubleshooting

| Response | Meaning |
|---|---|
| **204** | success — this is what you want |
| 401 | token wrong or **expired** (they expire — diarise renewal) |
| 403 | token missing **Actions: read and write** |
| 404 | wrong repo or filename, or the token can't see the repo |
| 422 | branch is not `main` |

A 404 on `btst.yml/dispatches` most likely means **the file isn't in the repo
yet** — which is still true as of now. Upload `UPDATE.zip` first, or this job
cannot work.

---

## Do the GitHub crons need removing?

**No — leave them.** They stay as a fallback for when cron-job.org itself has
an outage. They cannot cause double-runs:

- `concurrency:` groups in each workflow prevent overlapping runs
- `state.json` de-duplicates alerts
- the BTST picks file de-duplicates on `(date, symbol)`

---

## What still runs on GitHub's scheduler only

| workflow | IST | why it's fine |
|---|---|---|
| `snapshot.yml` | Mon 08:15, Sun 18:00 | only has to finish before Monday's open |
| `healthcheck.yml` | 08:30 | it *is* the thing that reports failures |
| `watchlist.yml` | 08:45 | pre-market list; minutes don't matter |
| `tests.yml` | on push | not time-based |

---

## The safety net (BUG 55)

Even if a schedule slips again, `btst.py` will no longer lie about it:

- **after 15:30** → `⛔ TOO LATE`, picks marked `MISSED`, `tradeable=0` in the
  picks file, and Model E skips them so the paper ledger can't book a fill that
  was never possible
- **before 15:00** → `⏳ PROVISIONAL` (tier precision is ~70% that early vs 82%
  at 15:20)

The alert still sends and picks are still written — suppressing them would hide
the outage, which is the mistake BUG 49 was about. You will *see* the problem
at the top of the message instead of acting on an unfillable price.
