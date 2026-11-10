"""Shared domain models used across modules."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"

    @property
    def sign(self) -> int:
        return 1 if self is Side.BUY else -1

    @property
    def opposite(self) -> "Side":
        return Side.SELL if self is Side.BUY else Side.BUY


@dataclass(frozen=True, slots=True)
class SymbolSpec:
    """Contract specification needed for sizing and price rounding."""

    name: str
    digits: int
    point: float
    tick_size: float
    tick_value: float
    contract_size: float
    volume_min: float
    volume_max: float
    volume_step: float
    stops_level: int = 0
    filling_mode: int = 0
    # overnight financing (MT5 SYMBOL_SWAP_*). swap_rollover3days uses MT5 weekday numbering (Sunday=0).
    swap_long: float = 0.0
    swap_short: float = 0.0
    swap_mode: int = 0
    swap_rollover3days: int = 3
    tick_spread_median: float = 0.0  # median ask-bid measured from real ticks (price units), 0 = unknown

    @classmethod
    def from_info(cls, info: Any) -> "SymbolSpec":
        point = float(info.point)
        tick_size = float(getattr(info, "trade_tick_size", 0.0) or point)
        return cls(
            name=str(info.name),
            digits=int(info.digits),
            point=point,
            tick_size=tick_size,
            tick_value=float(info.trade_tick_value),
            contract_size=float(getattr(info, "trade_contract_size", 1.0)),
            volume_min=float(info.volume_min),
            volume_max=float(info.volume_max),
            volume_step=float(info.volume_step),
            stops_level=int(getattr(info, "trade_stops_level", 0)),
            filling_mode=int(getattr(info, "filling_mode", 0)),
            swap_long=float(getattr(info, "swap_long", 0.0) or 0.0),
            swap_short=float(getattr(info, "swap_short", 0.0) or 0.0),
            swap_mode=int(getattr(info, "swap_mode", 0) or 0),
            swap_rollover3days=int(getattr(info, "swap_rollover3days", 3)),
        )

    def round_price(self, price: float) -> float:
        return round(round(price / self.tick_size) * self.tick_size, self.digits)

    def round_volume_down(self, volume: float) -> float:
        """Round a lot size DOWN to the volume step; returns 0 if below the minimum."""
        if volume <= 0 or not math.isfinite(volume):
            return 0.0
        steps = math.floor(volume / self.volume_step + 1e-9)
        rounded = round(steps * self.volume_step, 8)
        if rounded < self.volume_min:
            return 0.0
        return min(rounded, self.volume_max)

    def money_per_price_unit(self, volume: float) -> float:
        """Account-currency P&L for a 1.0 price move at ``volume`` lots."""
        return self.tick_value / self.tick_size * volume

    def swap_per_night(self, long: bool, volume: float, price: float) -> float:
        """Account-currency swap for holding ``volume`` lots over one rollover (negative = cost).

        Modes follow MT5 ENUM_SYMBOL_SWAP_MODE: 1 points; 2/3/4 money per lot (treated as account
        currency - exact for USD accounts on USD-margined symbols); 5/6 annual interest % on price
        (360-day year); 7/8 re-open modes approximated as points. 0 = disabled.
        """
        rate = self.swap_long if long else self.swap_short
        if rate == 0.0 or volume <= 0:
            return 0.0
        mode = self.swap_mode
        if mode in (1, 7, 8):
            return rate * self.point * self.money_per_price_unit(volume)
        if mode in (2, 3, 4):
            return rate * volume
        if mode in (5, 6):
            return price * self.money_per_price_unit(volume) * rate / 100.0 / 360.0
        return 0.0

    def rollover_nights(self, day_from: int, day_to: int) -> int:
        """Swap-nights charged when a position is held from day index ``day_from`` to ``day_to``.

        Days are integer days since 1970-01-01 (server time). One rollover per calendar day held;
        Saturday/Sunday roll for free and the ``swap_rollover3days`` weekday counts three times.
        """
        nights = 0
        for d in range(day_from, day_to):
            mt5_wd = (d + 4) % 7  # 1970-01-01 was a Thursday (MT5 weekday 4)
            if mt5_wd in (0, 6):
                continue
            nights += 3 if mt5_wd == self.swap_rollover3days else 1
        return nights


@dataclass(slots=True)
class PositionInfo:
    ticket: int
    symbol: str
    side: Side
    volume: float
    price_open: float
    sl: float
    tp: float
    profit: float
    magic: int
    price_current: float = 0.0
