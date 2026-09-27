"""
Alert de-duplication across cron runs.

GitHub Actions gives every run a clean filesystem, so "already alerted" has to
survive somewhere. This module keeps a small JSON file that the workflow commits
back to the repo (or restores from cache). Weekly records enforce the Pine
`onePerWeek` rule; a separate per-symbol breakout record prevents repeat 26-week
breakout alerts across weeks.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class AlertState:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._data: dict[str, Any] = {"weeks": {}, "updated_at": None}
        self._dirty = False
        self.load()
        self._migrate_weekly_alerts_to_breakout_records()

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            loaded = json.loads(self.path.read_text())
            if isinstance(loaded, dict) and "weeks" in loaded:
                self._data = loaded
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("could not read state file (%s) - starting fresh", exc)

    def _migrate_weekly_alerts_to_breakout_records(self) -> None:
        """
        Backfill the earliest retained weekly alert for each symbol.

        Older state files only tracked alerts per week. The workflow prunes those
        rows to six weeks, so this is the earliest available history when the
        cross-week breakout lock is first enabled. An explicit breakout record
        takes precedence because it may preserve an older first-alert date.
        """
        records = self._data.get("breakout_alerts")
        if not isinstance(records, dict):
            records = {}
            self._data["breakout_alerts"] = records

        # Walk in week order so repeated legacy alerts seed the cycle from the
        # first one still present, rather than restarting the lock from the last.
        weeks = self._data.get("weeks", {})
        if not isinstance(weeks, dict):
            return
        for week in sorted(weeks):
            by_symbol = weeks.get(week, {})
            if not isinstance(by_symbol, dict):
                continue
            for symbol, rec in by_symbol.items():
                key = str(symbol).strip().upper()
                if not key or key in records or not isinstance(rec, dict):
                    continue
                raw_time = str(rec.get("bar_time", "")).strip()
                try:
                    datetime.fromisoformat(raw_time)
                except ValueError:
                    continue
                records[key] = {"bar_time": raw_time}
                if "price" in rec:
                    records[key]["price"] = rec["price"]
                self._dirty = True

    def breakout_record(self, symbol: str) -> dict | None:
        """The first-alert record for the symbol's current 26-week cycle."""
        rec = self._data.get("breakout_alerts", {}).get(str(symbol).strip().upper())
        return rec if isinstance(rec, dict) else None

    def breakout_lock_until(self, symbol: str, weeks: int = 26) -> datetime | None:
        """When a fresh breakout cross may next alert; None means no lock."""
        try:
            weeks = int(weeks)
        except (TypeError, ValueError):
            return None
        if weeks <= 0:
            return None
        rec = self.breakout_record(symbol)
        if not rec:
            return None
        try:
            first_alert = datetime.fromisoformat(str(rec["bar_time"]))
        except (KeyError, TypeError, ValueError):
            return None
        return first_alert + timedelta(weeks=weeks)

    def breakout_cooldown_active(self, symbol: str, at: datetime,
                                 weeks: int = 26) -> bool:
        """Whether a candidate signal time falls inside the symbol's lockout."""
        until = self.breakout_lock_until(symbol, weeks)
        if until is None:
            return False
        if at.tzinfo is None and until.tzinfo is not None:
            at = at.replace(tzinfo=until.tzinfo)
        elif at.tzinfo is not None and until.tzinfo is None:
            until = until.replace(tzinfo=at.tzinfo)
        return at < until

    def mark_breakout_alert(self, symbol: str, bar_time: datetime, price: float,
                            entry_level: float) -> None:
        """Start a new 26-week lock after an eligible, delivered alert."""
        key = str(symbol).strip().upper()
        if not key:
            return
        records = self._data.setdefault("breakout_alerts", {})
        records[key] = {
            "bar_time": bar_time.isoformat(timespec="minutes"),
            "price": float(price),
            "entry_level": float(entry_level),
        }
        self._dirty = True

    def save(self) -> None:
        if not self._dirty:
            return
        self._data["updated_at"] = datetime.now().isoformat(timespec="seconds")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2, sort_keys=True))
        tmp.replace(self.path)
        self._dirty = False

    # ------------------------------------------------------------------ api
    def already_alerted(self, week: str, symbol: str) -> bool:
        return symbol in self._data.get("weeks", {}).get(week, {})

    def alert_record(self, week: str, symbol: str) -> dict | None:
        """
        The stored {bar_time, price} for an alert, or None.

        already_alerted() answers "did this fire at any point this week", which
        is the right question for de-duplication and the WRONG one for anything
        that cares WHEN. btst.py needs the date: a Monday breakout is not a
        Friday BTST setup (measured: Tier A +1.74% on the breakout day,
        +0.16% on any later day).
        """
        rec = self._data.get("weeks", {}).get(week, {}).get(symbol)
        return rec if isinstance(rec, dict) else None

    def alert_date(self, week: str, symbol: str) -> str | None:
        """YYYY-MM-DD of the alert, or None. Convenience over alert_record."""
        rec = self.alert_record(week, symbol)
        if not rec:
            return None
        bar = str(rec.get("bar_time", ""))
        return bar[:10] or None

    def mark(self, week: str, symbol: str, bar_time: datetime, price: float) -> None:
        wk = self._data.setdefault("weeks", {}).setdefault(week, {})
        wk[symbol] = {"bar_time": bar_time.isoformat(timespec="minutes"), "price": price}
        self._dirty = True

    # ---- stale-snapshot alarm de-duplication (BUG 49) --------------------
    # The 5-minute scanner would otherwise send the same outage alert ~75
    # times a day. One per calendar day is enough to be impossible to miss
    # without being impossible to read.
    def stale_alerted(self, day: str) -> bool:
        return self._data.get("stale_alert") == day

    def mark_stale(self, day: str) -> None:
        if self._data.get("stale_alert") != day:
            self._data["stale_alert"] = day
            self._dirty = True

    def prune(self, keep_weeks: int = 6) -> None:
        weeks = self._data.get("weeks", {})
        if len(weeks) <= keep_weeks:
            return
        for stale in sorted(weeks)[:-keep_weeks]:
            weeks.pop(stale, None)
        self._dirty = True

    def count(self, week: str) -> int:
        return len(self._data.get("weeks", {}).get(week, {}))
