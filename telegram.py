"""Telegram Bot API notifier with HTML formatting and safe chunking."""

from __future__ import annotations

import html
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

from strategy import Signal

log = logging.getLogger(__name__)

API = "https://api.telegram.org/bot{token}/sendMessage"
DOC_API = "https://api.telegram.org/bot{token}/sendDocument"
MAX_LEN = 4096


class Telegram:
    def __init__(self, token: str, chat_id: str, dry_run: bool = False):
        self.token = token
        self.chat_id = chat_id
        self.dry_run = dry_run or not (token and chat_id)
        if self.dry_run:
            log.warning("Telegram in DRY-RUN (missing token/chat id) - messages go to the log")

    def send(self, text: str, disable_preview: bool = True) -> bool:
        if self.dry_run:
            log.info("[telegram dry-run]\n%s", text)
            return True

        ok = True
        for chunk in _split(text, MAX_LEN):
            for attempt in range(3):
                try:
                    r = requests.post(
                        API.format(token=self.token),
                        json={
                            "chat_id": self.chat_id,
                            "text": chunk,
                            "parse_mode": "HTML",
                            "disable_web_page_preview": disable_preview,
                        },
                        timeout=20,
                    )
                    if r.status_code == 200:
                        break
                    if r.status_code == 429:
                        retry = (r.json().get("parameters", {}) or {}).get("retry_after", 3)
                        time.sleep(float(retry) + 1)
                        continue
                    log.error("telegram %s: %s", r.status_code, r.text[:300])
                    if attempt == 2:
                        ok = False
                except requests.RequestException as exc:
                    log.error("telegram network error: %s", exc)
                    if attempt == 2:
                        ok = False
                    time.sleep(2 * (attempt + 1))
            time.sleep(0.4)          # stay under ~30 msg/sec
        return ok

    def send_document(self, path: str | Path, caption: str = "") -> bool:
        """Upload a file (Excel report, CSV, ...) to the chat."""
        p = Path(path)
        if not p.exists():
            log.error("send_document: %s does not exist", p)
            return False
        if self.dry_run:
            log.info("[telegram dry-run] would upload %s (%d bytes)\n%s",
                     p.name, p.stat().st_size, caption)
            return True

        url = DOC_API.format(token=self.token)
        for attempt in range(3):
            try:
                with p.open("rb") as fh:
                    r = requests.post(
                        url,
                        data={"chat_id": self.chat_id,
                              "caption": caption[:1024],
                              "parse_mode": "HTML"},
                        files={"document": (p.name, fh)},
                        timeout=120,
                    )
                if r.status_code == 200:
                    return True
                if r.status_code == 429:
                    retry = (r.json().get("parameters", {}) or {}).get("retry_after", 3)
                    time.sleep(float(retry) + 1)
                    continue
                log.error("telegram document %s: %s", r.status_code, r.text[:300])
            except requests.RequestException as exc:
                log.error("telegram document network error: %s", exc)
            time.sleep(2 * (attempt + 1))
        return False

    # ------------------------------------------------------------------ views
    def send_signal(self, sig: Signal) -> bool:
        return self.send(format_signal(sig))

    def send_batch(self, signals: list[Signal]) -> bool:
        if not signals:
            return True
        if len(signals) == 1:
            return self.send_signal(signals[0])
        header = f"🚨 <b>{len(signals)} BUY signals</b> — weekly breakout + 5m entry\n"
        return self.send(header + "\n" + "\n\n".join(_compact(s) for s in signals))


def _esc(v: str) -> str:
    return html.escape(str(v), quote=False)


def _fmt(v: float, nd: int = 2) -> str:
    if v is None:
        return "n/a"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "n/a"
    if f != f:
        return "n/a"
    if abs(f) >= 1_00_00_000:
        return f"{f/1_00_00_000:.2f}Cr"
    if abs(f) >= 1_00_000:
        return f"{f/1_00_000:.2f}L"
    return f"{f:,.{nd}f}"


def format_signal(sig: Signal) -> str:
    ev = sig.evaluation
    v = ev.values
    pct = ((sig.price - sig.entry_level) / sig.entry_level * 100.0) if sig.entry_level else 0.0
    tag = "" if sig.trigger == "cross" else "  <i>(deferred)</i>"

    lines = [
        f"🟢 <b>BUY — {_esc(sig.symbol)}</b>{tag}",
        f"<code>{_esc(sig.exchange_segment)}</code> · 5m close "
        f"<b>{_fmt(sig.price)}</b> &gt; 26W level <b>{_fmt(sig.entry_level)}</b> "
        f"(+{pct:.2f}%)",
        f"🕐 <b>Signal candle {sig.bar_time.strftime('%H:%M')}"
        f"-{(sig.bar_time + timedelta(minutes=5)).strftime('%H:%M')} IST</b> "
        f"({sig.bar_time.strftime('%d-%b-%Y')})",
        f"    closed <b>{_fmt(sig.price)}</b> · "
        f"tradeable from <b>{(sig.bar_time + timedelta(minutes=5)).strftime('%H:%M')}</b>",
        "",
        "<b>Weekly conditions — 13/13 PASS</b>",
        f"• 52W high  <code>{_fmt(sig.level_52)}</code>",
        f"• EMA20 <code>{_fmt(v['ema_fast'])}</code> &gt; EMA50 <code>{_fmt(v['ema_slow'])}</code>",
        f"• RSI <code>{_fmt(v['rsi'])}</code> (prev {_fmt(v['rsi_1'])})",
        f"• MACD hist <code>{_fmt(v['macd_hist'], 3)}</code>",
        f"• Vol <code>{_fmt(v['week_volume'])}</code> &gt; avg <code>{_fmt(v['vol_sma'])}</code>",
        f"• Wk open <code>{_fmt(v['week_open'])}</code> · Day open <code>{_fmt(v['day_open'])}</code>",
    ]
    if not ev.all_ok:
        lines += ["", f"⚠️ <i>{ev.pass_count}/13 — {_esc(', '.join(ev.failed))}</i>"]
    return "\n".join(lines)


def _compact(sig: Signal) -> str:
    v = sig.evaluation.values
    pct = ((sig.price - sig.entry_level) / sig.entry_level * 100.0) if sig.entry_level else 0.0
    tag = "" if sig.trigger == "cross" else " (deferred)"
    return (f"🟢 <b>{_esc(sig.symbol)}</b>{tag} — <b>{_fmt(sig.price)}</b> "
            f"&gt; {_fmt(sig.entry_level)} (+{pct:.2f}%)\n"
            f"   candle {sig.bar_time.strftime('%H:%M')}, act from "
            f"<b>{(sig.bar_time + timedelta(minutes=5)).strftime('%H:%M')}</b> · "
            f"RSI {_fmt(v['rsi'])} · hist {_fmt(v['macd_hist'], 3)}")


def format_heartbeat(scanned: int, eligible: int, signals: int,
                     bar_time: datetime | None, elapsed: float,
                     errors: int = 0) -> str:
    when = bar_time.strftime("%d-%b %H:%M") if bar_time else "n/a"
    lines = [
        "📊 <b>Scan complete</b>",
        f"Bar: {when} IST · {elapsed:.0f}s",
        f"Universe {scanned} → gate-eligible {eligible} → signals <b>{signals}</b>",
    ]
    if errors:
        lines.append(f"⚠️ {errors} symbol error(s)")
    return "\n".join(lines)


def _split(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    parts, cur = [], ""
    for line in text.split("\n"):
        if len(cur) + len(line) + 1 > limit:
            if cur:
                parts.append(cur)
            cur = line[:limit]
        else:
            cur = f"{cur}\n{line}" if cur else line
    if cur:
        parts.append(cur)
    return parts
