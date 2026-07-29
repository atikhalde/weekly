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


class TelegramFanout:
    """
    Send the same message to SEVERAL Telegram destinations.

    Exposes the same surface as `Telegram`, so every existing call site keeps
    working unchanged.

    Two rules, both deliberate:

    * DELIVERY IS JUDGED ON THE PRIMARY ONLY. In scan.py a False return blocks
      `state.save()` so the next run retries - which is right when the alert
      never arrived, but wrong if only a spare bot was down: you would get the
      same alert again on every run. So the primary decides the return value
      and a secondary failure is logged loudly instead.
    * ONE BAD DESTINATION CANNOT SILENCE THE OTHERS. Every send is wrapped, so
      a revoked token or a bot that was never started in the target chat does
      not raise past this class.
    """

    def __init__(self, destinations, dry_run: bool = False):
        # destinations: list of (token, chat_id, label), primary first
        self.clients: list[tuple[Telegram, str]] = []
        for token, chat_id, label in destinations or []:
            self.clients.append((Telegram(token, chat_id, dry_run=dry_run), label))
        if not self.clients:                     # nothing configured -> dry-run
            self.clients.append((Telegram("", "", dry_run=True), "primary"))
        if len(self.clients) > 1:
            log.info("Telegram fan-out: %d destinations (%s)",
                     len(self.clients),
                     ", ".join(lbl for _c, lbl in self.clients))

    @property
    def dry_run(self) -> bool:
        return self.clients[0][0].dry_run

    def _fan(self, method: str, *args, **kwargs) -> bool:
        primary_ok = True
        for i, (client, label) in enumerate(self.clients):
            try:
                ok = getattr(client, method)(*args, **kwargs)
            except Exception as exc:                          # noqa: BLE001
                log.error("telegram %s destination failed: %s", label, exc)
                ok = False
            if i == 0:
                primary_ok = ok
            elif not ok:
                log.error("telegram %s destination did not receive the message "
                          "(primary was fine, so no retry will be attempted)",
                          label)
        return primary_ok

    def send(self, text: str, disable_preview: bool = True) -> bool:
        return self._fan("send", text, disable_preview)

    def send_document(self, path, caption: str = "") -> bool:
        return self._fan("send_document", path, caption)

    def send_signal(self, sig: Signal) -> bool:
        return self._fan("send_signal", sig)

    def send_batch(self, signals: list[Signal]) -> bool:
        return self._fan("send_batch", signals)


def build_telegram(cfg, dry_run: bool = False):
    """
    The one way callers should construct a notifier.

    Returns a fan-out over every configured destination, so adding a second
    bot is a matter of setting two environment variables - no code change and
    no risk of one script mirroring while another does not.
    """
    return TelegramFanout(cfg.secrets.telegram_destinations, dry_run=dry_run)


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


# --------------------------------------------------------------------------- #
#  Model C (swing) trade plan
#
#  The alert used to give only the trigger price, which left the three numbers
#  you actually need at the moment of buying - entry, stop, first target -
#  to be worked out by hand under time pressure.
#
#  These are read from models.yaml so they can NEVER drift from what Model C
#  is actually paper-trading. If C is missing or malformed the block is simply
#  omitted: an alert with no plan is fine, an alert with a WRONG plan is not.
# --------------------------------------------------------------------------- #
_PLAN_CACHE: dict | None = None


def _swing_plan_params() -> dict | None:
    """Read stop_pct / target_r / hold_days for the swing model, once."""
    global _PLAN_CACHE
    if _PLAN_CACHE is not None:
        return _PLAN_CACHE or None
    try:
        import yaml
        path = Path(__file__).resolve().parent / "models.yaml"
        raw = yaml.safe_load(path.read_text()) or {}
        for _key, m in (raw.get("models") or {}).items():
            if str(m.get("horizon", "")).lower() != "swing":
                continue
            ex = m.get("exit", {}) or {}
            stop_pct = float(ex.get("stop_pct", 0) or 0)
            target_r = float(ex.get("target_r", 0) or 0)
            if stop_pct <= 0 or target_r <= 0:
                continue
            _PLAN_CACHE = {
                "stop_pct": stop_pct,
                "target_r": target_r,
                "be_at_r": float(ex.get("be_at_r", 0) or 0),
                "hold_days": int(m.get("hold_days", ex.get("hold_days", 0)) or 0),
            }
            return _PLAN_CACHE
    except Exception as exc:                                  # noqa: BLE001
        log.debug("no swing plan available: %s", exc)
    _PLAN_CACHE = {}
    return None


def format_trade_plan(price: float) -> list[str]:
    """
    Entry / stop / first target for one signal, as Telegram HTML lines.

    Returns [] when no swing model is configured, so callers can splice the
    result in unconditionally.
    """
    p = _swing_plan_params()
    if not p or price <= 0:
        return []
    stop = price * (1.0 - p["stop_pct"] / 100.0)
    risk = price - stop
    target = price + p["target_r"] * risk
    tgt_pct = (target / price - 1.0) * 100.0

    lines = [
        "",
        "📐 <b>Trade plan (Model C · swing)</b>",
        f"• Buy      <code>{_fmt(price)}</code>",
        f"• Stop     <code>{_fmt(stop)}</code>  "
        f"(−{p['stop_pct']:.1f}% · risk {_fmt(risk)}/sh)",
        f"• Target   <code>{_fmt(target)}</code>  "
        f"(+{tgt_pct:.1f}% · {p['target_r']:.0f}R)",
    ]
    if p["be_at_r"] > 0:
        be_trigger = price + p["be_at_r"] * risk
        lines.append(f"• Move SL to cost once <code>{_fmt(be_trigger)}</code> "
                     f"trades (+{p['be_at_r']:.0f}R)")
    if p["hold_days"] > 0:
        lines.append(f"• Time exit after <b>{p['hold_days']} sessions</b> "
                     f"if neither hits")
    lines.append("<i>Paper-trading plan, not advice. Size so one stop is a "
                 "loss you can take.</i>")
    return lines


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
        *format_trade_plan(sig.price),
        "",
        # Never hardcode the count. c01 (52W high) is deliberately NOT part of
        # the entry gate when req52 is off, so a perfectly valid signal can
        # show 12/13. Printing "13/13" then listing a failure underneath was
        # self-contradictory.
        (f"<b>Weekly conditions — {ev.pass_count}/13</b>"
         + (" PASS" if ev.all_ok else
            f" · not required for entry: {_esc(', '.join(ev.failed))}")),
        f"• 52W high  <code>{_fmt(sig.level_52)}</code>",
        f"• EMA20 <code>{_fmt(v['ema_fast'])}</code> &gt; EMA50 <code>{_fmt(v['ema_slow'])}</code>",
        f"• RSI <code>{_fmt(v['rsi'])}</code> (prev {_fmt(v['rsi_1'])})",
        f"• MACD hist <code>{_fmt(v['macd_hist'], 3)}</code>",
        f"• Vol <code>{_fmt(v['week_volume'])}</code> &gt; avg <code>{_fmt(v['vol_sma'])}</code>",
        f"• Wk open <code>{_fmt(v['week_open'])}</code> · Day open <code>{_fmt(v['day_open'])}</code>",
    ]
    # The header above already reports the count and any non-blocking failure,
    # so no contradictory footer here.
    return "\n".join(lines)


def _compact(sig: Signal) -> str:
    v = sig.evaluation.values
    pct = ((sig.price - sig.entry_level) / sig.entry_level * 100.0) if sig.entry_level else 0.0
    tag = "" if sig.trigger == "cross" else " (deferred)"
    out = (f"🟢 <b>{_esc(sig.symbol)}</b>{tag} — <b>{_fmt(sig.price)}</b> "
           f"&gt; {_fmt(sig.entry_level)} (+{pct:.2f}%)\n"
           f"   candle {sig.bar_time.strftime('%H:%M')}, act from "
           f"<b>{(sig.bar_time + timedelta(minutes=5)).strftime('%H:%M')}</b> · "
           f"RSI {_fmt(v['rsi'])} · hist {_fmt(v['macd_hist'], 3)}")
    # One compact plan line so a batch alert is still actionable without
    # opening the chart. Same numbers as the detailed format.
    p = _swing_plan_params()
    if p and sig.price > 0:
        stop = sig.price * (1.0 - p["stop_pct"] / 100.0)
        target = sig.price + p["target_r"] * (sig.price - stop)
        out += (f"\n   📐 buy <b>{_fmt(sig.price)}</b> · SL <b>{_fmt(stop)}</b> "
                f"· T1 <b>{_fmt(target)}</b>")
    return out


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
