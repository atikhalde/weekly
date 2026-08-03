# Which files to create cron jobs for

Short answer: **cron-job.org is only needed for `scan.yml`.** Everything else
can stay on GitHub's own schedule.

---

## Why only scan.yml

GitHub's `schedule:` is **best-effort**. Under load it runs late — commonly
3–4 minutes, sometimes skipping ticks entirely. That matters differently for
each workflow:

| workflow | GitHub schedule (IST) | does lateness hurt? | needs external cron? |
|---|---|---|---|
| **scan.yml** | 08:45–15:55, every 5 min | **YES** — a 4-min delay on a 5-min candle means you get the alert after the move | **YES** |
| snapshot.yml | Mon 08:15, Sun 18:00 | No — it just has to finish before Monday's open | No |
| report.yml | 16:00 Mon–Fri | No — post-market summary | No |
| ab.yml | 16:15 Mon–Fri | No — post-market summary | No |
| tests.yml | on push only | n/a | No |

You are hunting moves that happen in minutes. On `scan.yml` an external
trigger fires in ~80–90 s instead of 3–4 min. On the post-market jobs, being
15 minutes late changes nothing.

**So: one cron job, pointed at `scan.yml`.** That is what `CRONJOB_SETUP.md`
already documents, and it is still the right call.

---

## The single job to create

<https://console.cron-job.org/jobs> → CREATE CRONJOB

| Field | Value |
|---|---|
| Title | `NSE 5m breakout scan` |
| URL | `https://api.github.com/repos/atikhalde/weekly/actions/workflows/scan.yml/dispatches` |
| Method | **POST** |
| Request body | `{"ref":"main"}` |

Headers:

```
Accept: application/vnd.github+json
Authorization: Bearer <YOUR_PAT>
X-GitHub-Api-Version: 2022-11-28
Content-Type: application/json
```

Schedule — **set timezone to Asia/Kolkata FIRST**, then:

| Field | Value |
|---|---|
| Days of month | Every day |
| Months | Every month |
| Days of week | Mon–Fri |
| Hours | 9, 10, 11, 12, 13, 14, 15 |
| Minutes | 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55 |

= 84 fires/day. Success is **HTTP 204** with no body.

Full step-by-step, including the PAT setup and the error-code table, is in
`CRONJOB_SETUP.md` in the repo.

---

## Do NOT disable the GitHub schedules

Keep both running. They cannot double-alert:

- `state.json` de-duplicates every signal
- `concurrency: scan` in the workflow stops overlapping runs

The GitHub schedule is your fallback if cron-job.org is down.

---

## Optional second job

If you want the A/B standings to land promptly each day you *could* add a
second job pointed at `ab.yml` (same headers/body, 16:15 IST, Mon–Fri). It is
genuinely optional — the workflow already runs itself, and being a few minutes
late on a post-market report costs nothing.

I would not bother. One job, one thing to renew, one thing to monitor.

---

## One thing to diarise

The GitHub PAT expires (90 days if you followed the guide). **An expired token
means no external trigger and no error** — cron-job.org will just log 401s
while GitHub's slower schedule quietly carries the load. Set a calendar
reminder to rotate it.

To check it is working: cron-job.org → your job → history should show a wall
of **204**s. Anything else, see the error table in `CRONJOB_SETUP.md`.
