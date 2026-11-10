"""Trading-session rules: US equity regular hours, holidays, early closes.

* US stocks: no entries outside [open + skip_after_open, close - no_entry_before_close],
  and every stock position is flattened ``flatten_before_close_min`` before the bell
  (no overnight gap / swap risk).
* FX and metals trade ~24/5; weekend gaps are handled naturally (no ticks => no signals).

Times are evaluated in America/New_York so DST shifts are handled automatically.
(Windows needs the ``tzdata`` package for zoneinfo.)
"""

from __future__ import annotations

from datetime import date, datetime, time as dtime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from src.config import SessionSettings
from src.universe import US_STOCK, Instrument


def _parse_hhmm(value: str) -> dtime:
    hh, mm = value.split(":")
    return dtime(int(hh), int(mm))


class SessionCalendar:
    def __init__(self, cfg: SessionSettings, instruments: dict[str, Instrument]) -> None:
        self.cfg = cfg
        self.instruments = instruments
        self.tz = ZoneInfo(cfg.timezone)
        self.open_t = _parse_hhmm(cfg.open_time)
        self.close_t = _parse_hhmm(cfg.close_time)
        self.early_close_t = _parse_hhmm(cfg.early_close_time)
        self.holidays = {date.fromisoformat(d) for d in cfg.holidays}
        self.early_closes = {date.fromisoformat(d) for d in cfg.early_closes}

    def _is_stock(self, symbol: str) -> bool:
        inst = self.instruments.get(symbol)
        return inst is not None and inst.asset_class == US_STOCK

    def us_session(self, ts: float) -> Optional[tuple[datetime, datetime]]:
        """(open, close) of the regular session on the local date of ``ts``; None if closed all day."""
        local = datetime.fromtimestamp(ts, self.tz)
        d = local.date()
        if local.weekday() >= 5 or d in self.holidays:
            return None
        close_t = self.early_close_t if d in self.early_closes else self.close_t
        return datetime.combine(d, self.open_t, self.tz), datetime.combine(d, close_t, self.tz)

    def entry_block_reason(self, symbol: str, ts: float) -> Optional[str]:
        if not self.cfg.enforce or not self._is_stock(symbol):
            return None
        session = self.us_session(ts)
        if session is None:
            return "us_market_closed"
        open_dt, close_dt = session
        now = datetime.fromtimestamp(ts, self.tz)
        if now < open_dt + timedelta(minutes=self.cfg.skip_after_open_min):
            return "us_pre_open"
        if now >= close_dt - timedelta(minutes=self.cfg.no_entry_before_close_min):
            return "us_near_close"
        return None

    def must_flatten(self, symbol: str, ts: float) -> bool:
        if not self.cfg.enforce or not self._is_stock(symbol):
            return False
        session = self.us_session(ts)
        if session is None:
            return False  # market shut: orders would be rejected anyway
        _, close_dt = session
        now = datetime.fromtimestamp(ts, self.tz)
        return close_dt - timedelta(minutes=self.cfg.flatten_before_close_min) <= now < close_dt
