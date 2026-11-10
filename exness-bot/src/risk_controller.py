"""Capital protection: Kelly sizing, VaR, drawdown, VPIN filter and circuit breaker.

Kelly criterion for a binary bet with win probability p and payoff ratio b
(average win / average loss):

    f* = p - (1 - p) / b

A fractional Kelly (default 1/4) is applied and hard-capped at ``max_risk_per_trade``.
The resulting fraction is the share of equity put at risk between entry and stop:

    lots = equity * f / (stop_distance / tick_size * tick_value)

Circuit breaker trips (and triggers an emergency flatten) on:
  * latency > ``max_latency_ms`` (local calls) / ``max_order_latency_ms`` (order_send, which
    includes the network round trip) for ``latency_breach_count`` consecutive critical calls,
  * ``max_consecutive_errors`` consecutive I/O errors,
  * drawdown from peak >= ``max_drawdown_pct``  (hard trip: manual reset required),
  * intraday loss >= ``max_daily_loss_pct``      (hard trip: manual reset required).
"""

from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from statistics import NormalDist
from typing import Any, Awaitable, Callable, Deque, Optional, Sequence

import numpy as np
from loguru import logger

from src.config import RiskLimits
from src.models import SymbolSpec

TripCallback = Callable[[str], Awaitable[None]]

HARD_TRIP_PREFIXES = ("max_drawdown", "daily_loss")


class BreakerState(str, Enum):
    ARMED = "armed"
    TRIPPED = "tripped"


# ------------------------------------------------------------ pure math
def kelly_fraction(win_rate: float, payoff_ratio: float) -> float:
    """Full-Kelly fraction f* = p - (1-p)/b, floored at 0 (no negative bets)."""
    if not 0.0 <= win_rate <= 1.0:
        raise ValueError("win_rate must be within [0, 1]")
    if payoff_ratio <= 0:
        return 0.0
    return max(0.0, win_rate - (1.0 - win_rate) / payoff_ratio)


def historical_var(returns: Sequence[float] | np.ndarray, confidence: float) -> float:
    """Historical VaR as a positive loss fraction."""
    arr = np.asarray(returns, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return max(0.0, -float(np.quantile(arr, 1.0 - confidence)))


def parametric_var(returns: Sequence[float] | np.ndarray, confidence: float) -> float:
    """Gaussian VaR: z_alpha * sigma - mu (positive loss fraction)."""
    arr = np.asarray(returns, dtype=np.float64)
    if arr.size < 2:
        return 0.0
    z = NormalDist().inv_cdf(confidence)
    return max(0.0, z * float(arr.std(ddof=1)) - float(arr.mean()))


def max_drawdown(equity: Sequence[float] | np.ndarray) -> float:
    """Maximum peak-to-trough drawdown as a fraction."""
    arr = np.asarray(equity, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(arr)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peaks > 0, (peaks - arr) / peaks, 0.0)
    return float(dd.max())


@dataclass
class TradeStats:
    wins: int = 0
    losses: int = 0
    gross_win: float = 0.0
    gross_loss: float = 0.0

    def record(self, pnl: float) -> None:
        if pnl > 0:
            self.wins += 1
            self.gross_win += pnl
        elif pnl < 0:
            self.losses += 1
            self.gross_loss += -pnl

    @property
    def n(self) -> int:
        return self.wins + self.losses

    @property
    def win_rate(self) -> float:
        return self.wins / self.n if self.n else 0.0

    @property
    def payoff_ratio(self) -> float:
        if self.wins == 0 or self.losses == 0:
            return 0.0
        return (self.gross_win / self.wins) / (self.gross_loss / self.losses)


# -------------------------------------------------------------- controller
class RiskController:
    LATENCY_OPS = frozenset({"symbol_info_tick", "copy_ticks_from", "market_book_get", "order_send", "pipeline"})

    def __init__(
        self,
        limits: RiskLimits,
        vpin_threshold: float,
        on_trip: Optional[TripCallback] = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.limits = limits
        self.vpin_threshold = vpin_threshold
        self.on_trip = on_trip
        self._clock = clock

        self.state = BreakerState.ARMED
        self.trip_reason: Optional[str] = None
        self.trip_ts: Optional[float] = None
        self.trip_count = 0

        self._latency_breaches = 0
        self._consecutive_errors = 0
        self.last_latency_ms: dict[str, float] = {}
        self.max_latency_seen_ms: dict[str, float] = {}

        self._equity: Deque[float] = deque(maxlen=limits.var_window + 1)
        self.peak_equity = 0.0
        self.current_equity = 0.0
        self.drawdown = 0.0
        self._day: Optional[str] = None
        self.day_start_equity = 0.0

        self.stats = TradeStats()
        self._tasks: set[asyncio.Task[Any]] = set()

    # ---------------------------------------------------- breaker inputs
    @property
    def tripped(self) -> bool:
        return self.state is BreakerState.TRIPPED

    def record_latency(self, op: str, latency_ms: float) -> None:
        if op not in self.LATENCY_OPS:
            return
        self.last_latency_ms[op] = latency_ms
        self.max_latency_seen_ms[op] = max(latency_ms, self.max_latency_seen_ms.get(op, 0.0))
        limit = self.limits.max_order_latency_ms if op == "order_send" else self.limits.max_latency_ms
        if latency_ms > limit:
            self._latency_breaches += 1
            logger.warning(
                "Latency breach {} = {:.1f} ms (> {:.0f} ms) [{}/{}]",
                op, latency_ms, limit, self._latency_breaches, self.limits.latency_breach_count,
            )
            if self._latency_breaches >= self.limits.latency_breach_count:
                self.trip(f"latency_breach:{op}={latency_ms:.1f}ms")
        else:
            self._latency_breaches = 0

    def record_error(self, source: str, exc: BaseException) -> None:
        self._consecutive_errors += 1
        if self._consecutive_errors >= self.limits.max_consecutive_errors:
            self.trip(f"consecutive_errors:{source}:{type(exc).__name__}")

    def record_success(self) -> None:
        self._consecutive_errors = 0

    def record_trade_result(self, pnl: float) -> None:
        self.stats.record(pnl)

    # ------------------------------------------------------------ equity
    def update_equity(self, equity: float, ts: Optional[float] = None) -> None:
        if not math.isfinite(equity) or equity <= 0:
            self.trip(f"invalid_equity:{equity}")
            return
        t = self._clock() if ts is None else ts
        day = datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d")
        if day != self._day:
            self._day = day
            self.day_start_equity = equity

        self.current_equity = equity
        self._equity.append(equity)
        self.peak_equity = max(self.peak_equity, equity)
        self.drawdown = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0.0

        if self.drawdown >= self.limits.max_drawdown_pct:
            self.trip(f"max_drawdown:{self.drawdown:.2%}")
        daily_loss = (self.day_start_equity - equity) / self.day_start_equity if self.day_start_equity else 0.0
        if daily_loss >= self.limits.max_daily_loss_pct:
            self.trip(f"daily_loss:{daily_loss:.2%}")

    def returns(self) -> np.ndarray:
        eq = np.fromiter(self._equity, dtype=np.float64)
        if eq.size < 2:
            return np.empty(0)
        return np.diff(eq) / eq[:-1]

    def current_var(self) -> float:
        r = self.returns()
        if r.size < self.limits.min_var_observations:
            return 0.0
        conf = self.limits.var_confidence
        return max(historical_var(r, conf), parametric_var(r, conf))

    # ------------------------------------------------------------ sizing
    def realized_kelly(self) -> Optional[float]:
        """Full Kelly from the bot's own closed trades (None until enough trades)."""
        if self.stats.n >= self.limits.min_trades_for_kelly and self.stats.payoff_ratio > 0:
            return kelly_fraction(self.stats.win_rate, self.stats.payoff_ratio)
        return None

    def kelly_risk_fraction(self, model_kelly: Optional[float] = None) -> float:
        """Share of equity to risk on the next trade.

        ``model_kelly`` = per-trade full Kelly from the signal (OU win probability and the
        trade's real TP/SL ratio). Once enough trades exist, the realised Kelly caps it
        (the model is never trusted more than the track record). Fractional Kelly and the
        hard per-trade cap are applied last.
        """
        realized = self.realized_kelly()
        if model_kelly is not None:
            f = max(model_kelly, 0.0)
            if realized is not None:
                f = min(f, realized)
        elif realized is not None:
            f = realized
        else:
            f = kelly_fraction(self.limits.prior_win_rate, self.limits.prior_payoff_ratio)
        return min(f * self.limits.kelly_fraction, self.limits.max_risk_per_trade)

    def cap_by_margin(self, spec: SymbolSpec, lots: float, margin_required: Optional[float], free_margin: float) -> float:
        """Shrink ``lots`` so the new trade uses at most ``max_margin_usage`` of free margin."""
        if lots <= 0 or margin_required is None or margin_required <= 0:
            return lots
        budget = max(free_margin, 0.0) * self.limits.max_margin_usage
        if margin_required <= budget:
            return lots
        return spec.round_volume_down(lots * budget / margin_required)

    def position_size(self, spec: SymbolSpec, equity: float, stop_distance: float, model_kelly: Optional[float] = None) -> float:
        """Lots such that hitting the stop loses ``equity * kelly_risk_fraction(model_kelly)``."""
        if stop_distance <= 0 or equity <= 0:
            return 0.0
        risk_money = equity * self.kelly_risk_fraction(model_kelly)
        loss_per_lot = stop_distance / spec.tick_size * spec.tick_value
        if loss_per_lot <= 0 or not math.isfinite(loss_per_lot):
            return 0.0
        return spec.round_volume_down(risk_money / loss_per_lot)

    # ------------------------------------------------------- pre-trade gate
    def pre_trade_check(
        self,
        symbol: str,
        vpin: Optional[float],
        open_symbol: int,
        open_total: int,
        asset_class: Optional[str] = None,
        open_class: int = 0,
        class_net_direction: int = 0,
        side_sign: int = 0,
    ) -> tuple[bool, str]:
        """``class_net_direction`` = (#long - #short) open in the class; ``side_sign`` = +1 buy / -1 sell."""
        self.maybe_rearm()
        if self.tripped:
            return False, f"breaker_tripped:{self.trip_reason}"
        if vpin is None:
            return False, "vpin_unavailable"
        if vpin >= self.vpin_threshold:
            return False, f"toxic_flow:vpin={vpin:.3f}"
        if open_symbol >= self.limits.max_positions_per_symbol:
            return False, "max_positions_per_symbol"
        if open_total >= self.limits.max_total_positions:
            return False, "max_total_positions"
        if asset_class is not None:
            cap = self.limits.max_positions_per_class.get(asset_class)
            if cap is not None and open_class >= cap:
                return False, f"max_positions_class:{asset_class}"
            net_cap = self.limits.max_net_direction_per_class.get(asset_class)
            if net_cap is not None and side_sign and abs(class_net_direction + side_sign) > net_cap:
                return False, f"max_net_direction:{asset_class}"
        var = self.current_var()
        if var > self.limits.max_var_pct:
            return False, f"var_limit:{var:.4f}"
        return True, "ok"

    # ------------------------------------------------------------ breaker
    def trip(self, reason: str) -> None:
        if self.tripped:
            return
        self.state = BreakerState.TRIPPED
        self.trip_reason = reason
        self.trip_ts = self._clock()
        self.trip_count += 1
        logger.critical("CIRCUIT BREAKER TRIPPED: {} -> emergency flatten", reason)
        if self.on_trip is None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.error("No running event loop; emergency flatten must be invoked manually")
            return
        task = loop.create_task(self._run_trip(reason))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run_trip(self, reason: str) -> None:
        assert self.on_trip is not None
        try:
            await self.on_trip(reason)
        except Exception:
            logger.exception("emergency flatten callback failed")

    @property
    def is_hard_trip(self) -> bool:
        return bool(self.trip_reason and self.trip_reason.startswith(HARD_TRIP_PREFIXES))

    def maybe_rearm(self) -> None:
        if not self.tripped or self.is_hard_trip or self.trip_ts is None:
            return
        if self._clock() - self.trip_ts >= self.limits.breaker_cooldown_s:
            logger.warning("Circuit breaker re-armed after cooldown (was: {})", self.trip_reason)
            self.reset()

    def reset(self) -> None:
        """Manual reset (also required after hard trips)."""
        self.state = BreakerState.ARMED
        self.trip_reason = None
        self.trip_ts = None
        self._latency_breaches = 0
        self._consecutive_errors = 0
        if self.current_equity > 0:
            self.peak_equity = self.current_equity
            self.drawdown = 0.0

    def snapshot(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "trip_reason": self.trip_reason,
            "trip_count": self.trip_count,
            "equity": self.current_equity,
            "peak_equity": self.peak_equity,
            "drawdown": self.drawdown,
            "var": self.current_var(),
            "kelly_risk_fraction": self.kelly_risk_fraction(),
            "trades": self.stats.n,
            "win_rate": self.stats.win_rate,
            "payoff_ratio": self.stats.payoff_ratio,
            "last_latency_ms": dict(self.last_latency_ms),
            "max_latency_ms": dict(self.max_latency_seen_ms),
        }
