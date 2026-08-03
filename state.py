"""
Alert de-duplication across cron runs.

GitHub Actions gives every run a clean filesystem, so "already alerted" has to
survive somewhere. This module keeps a small JSON file that the workflow commits
back to the repo (or restores from cache). Keys are per (symbol, week, bar) so
the Pine `onePerWeek` rule holds across separate processes.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class AlertState:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._data: dict[str, Any] = {"weeks": {}, "updated_at": None}
        self._dirty = False
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            loaded = json.loads(self.path.read_text())
            if isinstance(loaded, dict) and "weeks" in loaded:
                self._data = loaded
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("could not read state file (%s) - starting fresh", exc)

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
