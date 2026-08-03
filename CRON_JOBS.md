# Which files need an external cron job

**Answer as of 04-Aug-2026: FOUR jobs — `scan`, `btst`, `report`, `ab`.**

---

## RETRACTION (BUG 55)

This file used to say:

> "cron-job.org is only needed for `scan.yml`. Everything else can stay on
> GitHub's own schedule." … "On the post-market jobs, being 15 minutes late
> changes nothing."

**That was wrong, and 03-Aug-2026 proved it three separate ways:**

| workflow | what actually happened on 03-Aug |
|---|---|
| `btst.yml` | ran **15:48 IST** — 28 min late, *after* the 15:30 close. The alert still said "BUY NOW" on an entry that no longer existed. |
| `report.yml` | **zero scheduled runs.** No paper report after the market. |
| `ab.yml` | **zero scheduled runs.** |

Two errors in the old reasoning:

1. **"Being 15 minutes late changes nothing"** is false for `btst.yml`. It has
   a **ten-minute entry window** (15:20 → 15:30). Late is not degraded, it is
   *worthless*.
2. **"Late" was the wrong risk.** The real failure was **not running at all**.
   GitHub's `schedule:` silently drops ticks under load — you get no run, no
   failure, and no notification.

---

## The four jobs

All four workflows now accept `repository_dispatch`, so cron-job.org can fire
them directly instead of hoping GitHub's scheduler wakes up. The `schedule:`
crons are kept as a **fallback**.

| workflow | IST time | event_type | why it needs this |
|---|---|---|---|
| `scan.yml` | every 5 min, 08:45–15:55 | *(existing setup)* | 4-min delay on a 5-min candle = alert after the move |
| **`btst.yml`** | **15:18** | `btst` | ten-minute entry window; late = untradeable |
| **`report.yml`** | **16:00** | `report` | didn't fire at all on 03-Aug |
| **`ab.yml`** | **16:15** | `ab` | didn't fire at all on 03-Aug |

`snapshot.yml` and `healthcheck.yml` genuinely don't need it — the snapshot
only has to finish before Monday's open, and the health check is itself the
thing that reports failures.

---

## Setup, per job

Create a **PAT** with `repo` scope (Settings → Developer settings → Personal
access tokens). Then on cron-job.org create one job per row above:

- **URL** `https://api.github.com/repos/atikhalde/weekly/dispatches`
- **Method** `POST`
- **Headers**
  ```
  Authorization: Bearer <YOUR_PAT>
  Accept: application/vnd.github+json
  Content-Type: application/json
  ```
- **Body** — one of:
  ```json
  {"event_type":"btst"}
  {"event_type":"report"}
  {"event_type":"ab"}
  ```
- **Schedule** the IST time from the table, **Mon–Fri**. Set the job's timezone
  to `Asia/Kolkata` so you don't have to do UTC arithmetic.

A successful dispatch returns **HTTP 204** with an empty body. If you get 401
the PAT is wrong or expired; 404 usually also means the PAT lacks `repo` scope.

### Why 15:18 and not 15:20

The dispatch itself takes a few seconds and the runner needs ~40–60 s to boot
and install dependencies. Firing at 15:18 puts the scan itself at ~15:20.

---

## The safety net

Even with all of this, `btst.py` now refuses to lie about a late run
(**BUG 55**):

- **after 15:30** → header reads `⛔ TOO LATE`, every pick shows `MISSED`
  instead of `BUY NOW`, and the picks file records `tradeable=0` so Model E
  does not book a fill that could never have happened
- **before 15:00** → header reads `⏳ PROVISIONAL` (tier precision is only
  ~70% that early, vs 82% at 15:20)

The alert is still **sent** and the picks are still **written** in both cases.
Suppressing them would hide the outage, which is the mistake BUG 49 was about.

So if a schedule slips again you will *see* it, in the message, at the top —
rather than acting on an unfillable price.
