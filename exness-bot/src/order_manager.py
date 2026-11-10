"""Low-latency order routing & position management on top of the MT5 API.

* Market orders with automatic filling-mode detection (FOK / IOC / RETURN).
* Bounded re-quote / price-changed retries with a fresh quote each attempt.
* SL/TP sanitisation against the broker's stops level.
* Emergency flatten: cancel pending orders + close every bot position, with retries.
* ``dry_run`` paper mode: identical code path up to ``order_send``; fills are simulated
  locally against live quotes, SL/TP are enforced on mark-to-market.
"""

from __future__ import annotations

import asyncio
import itertools
import time
from dataclasses import dataclass, field, replace
from typing import Any, Optional

from loguru import logger

from src.models import PositionInfo, Side, SymbolSpec
from src.mt5_client import AsyncMT5Client, MT5Error

_RETRYABLE = ("TRADE_RETCODE_REQUOTE", "TRADE_RETCODE_PRICE_CHANGED", "TRADE_RETCODE_PRICE_OFF")
_SUCCESS = ("TRADE_RETCODE_DONE", "TRADE_RETCODE_PLACED")
# Request-format rejections: try another filling mode / drop type_time
_FORMAT_ERRORS = ("TRADE_RETCODE_INVALID", "TRADE_RETCODE_INVALID_FILL")
FILLING_NAMES = {0: "FOK", 1: "IOC", 2: "RETURN"}

OrderFormat = tuple[int, bool]  # (type_filling, include type_time)


@dataclass(frozen=True, slots=True)
class OrderResult:
    success: bool
    retcode: int
    ticket: int = 0
    price: float = 0.0
    volume: float = 0.0
    latency_ms: float = 0.0
    comment: str = ""
    pnl: float = 0.0
    attempts: int = 1
    request: dict[str, Any] = field(default_factory=dict)


class OrderManager:
    def __init__(
        self,
        client: AsyncMT5Client,
        magic: int,
        deviation_points: int = 10,
        dry_run: bool = True,
        max_retries: int = 2,
    ) -> None:
        self._client = client
        self.magic = magic
        self.deviation = deviation_points
        self.dry_run = dry_run
        self.max_retries = max_retries
        self._specs: dict[str, SymbolSpec] = {}
        self._formats: dict[str, OrderFormat] = {}  # request format the broker accepted, per symbol
        self._paper: dict[int, PositionInfo] = {}
        self._paper_tickets = itertools.count(1)
        self.paper_realized_pnl = 0.0
        self._flatten_lock = asyncio.Lock()
        self.orders_sent = 0
        self.orders_failed = 0

    # ------------------------------------------------------------ helpers
    def _c(self, name: str) -> int:
        return self._client.const(name)

    def _codes(self, names: tuple[str, ...]) -> set[int]:
        out: set[int] = set()
        for n in names:
            try:
                out.add(self._c(n))
            except AttributeError:
                continue
        return out

    async def symbol_spec(self, symbol: str) -> SymbolSpec:
        spec = self._specs.get(symbol)
        if spec is None:
            info = await self._client.call("symbol_info", symbol)
            if info is None:
                code, msg = await self._client.last_error()
                raise MT5Error(f"symbol_info({symbol}) failed: {msg}", code)
            spec = self._specs[symbol] = SymbolSpec.from_info(info)
        return spec

    async def margin_required(self, symbol: str, side: Side, volume: float, price: float) -> Optional[float]:
        """Margin (account currency) for a new position, via ``order_calc_margin``."""
        try:
            action = self._c("ORDER_TYPE_BUY" if side is Side.BUY else "ORDER_TYPE_SELL")
            value = await self._client.call("order_calc_margin", action, symbol, float(volume), float(price))
            return None if value is None else float(value)
        except Exception as exc:
            logger.warning("order_calc_margin({}) failed: {}", symbol, exc)
            return None

    def _filling(self, spec: SymbolSpec) -> int:
        fm = spec.filling_mode
        if fm & self._c("SYMBOL_FILLING_FOK"):
            return self._c("ORDER_FILLING_FOK")
        if fm & self._c("SYMBOL_FILLING_IOC"):
            return self._c("ORDER_FILLING_IOC")
        return self._c("ORDER_FILLING_RETURN")

    def _format_candidates(self, spec: SymbolSpec) -> list[OrderFormat]:
        """All (filling, type_time) combinations, the cached/preferred one first."""
        preferred = self._filling(spec)
        fillings = [preferred] + [self._c(n) for n in ("ORDER_FILLING_FOK", "ORDER_FILLING_IOC", "ORDER_FILLING_RETURN")
                                  if self._c(n) != preferred]
        cands = [(f, t) for f in fillings for t in (True, False)]
        cached = self._formats.get(spec.name)
        if cached in cands:
            cands.remove(cached)
            cands.insert(0, cached)
        return cands

    def _request(self, spec: SymbolSpec, side: Side, volume: float, price: float, fmt: OrderFormat, comment: str) -> dict[str, Any]:
        filling, with_time = fmt
        req: dict[str, Any] = {
            "action": self._c("TRADE_ACTION_DEAL"),
            "symbol": spec.name,
            "volume": float(volume),
            "type": self._c("ORDER_TYPE_BUY" if side is Side.BUY else "ORDER_TYPE_SELL"),
            "price": float(price),
            "deviation": int(self.deviation),
            "magic": int(self.magic),
            "comment": comment[:31],  # MT5 comment limit
            "type_filling": int(filling),
        }
        if with_time:
            req["type_time"] = self._c("ORDER_TIME_GTC")
        return req

    async def probe_order_format(self, symbol: str) -> tuple[Optional[OrderFormat], Any, list[str]]:
        """Find a request format the broker accepts, using order_check (no order is placed).

        Returns (format or None, last OrderCheckResult, human-readable attempt log).
        """
        spec = await self.symbol_spec(symbol)
        tick = await self._tick(symbol)
        format_codes = self._codes(_FORMAT_ERRORS)
        log: list[str] = []
        last: Any = None
        for fmt in self._format_candidates(spec):
            req = self._request(spec, Side.BUY, spec.volume_min, float(tick.ask), fmt, "check")
            last = await self._client.call("order_check", req)
            code = None if last is None else int(last.retcode)
            label = f"{FILLING_NAMES.get(fmt[0], fmt[0])}{'' if fmt[1] else ' (no type_time)'}"
            log.append(f"{label}: {code} {getattr(last, 'comment', '')}".strip())
            if code is not None and code not in format_codes:
                self._formats[symbol] = fmt
                logger.info("{}: broker accepts filling={} type_time={}", symbol, FILLING_NAMES.get(fmt[0]), fmt[1])
                return fmt, last, log
        return None, last, log

    async def _tick(self, symbol: str) -> Any:
        tick = await self._client.call("symbol_info_tick", symbol)
        if tick is None:
            code, msg = await self._client.last_error()
            raise MT5Error(f"symbol_info_tick({symbol}) failed: {msg}", code)
        return tick

    @staticmethod
    def sanitize_stops(
        spec: SymbolSpec, side: Side, entry: float, sl: Optional[float], tp: Optional[float]
    ) -> tuple[float, float]:
        """Return (sl, tp) rounded and at least ``stops_level`` away; 0.0 means 'none'."""
        min_dist = max(spec.stops_level * spec.point, spec.tick_size)
        out_sl, out_tp = 0.0, 0.0
        if sl:
            if side is Side.BUY:
                sl = min(sl, entry - min_dist)
            else:
                sl = max(sl, entry + min_dist)
            out_sl = spec.round_price(sl)
        if tp:
            valid = (tp >= entry + min_dist) if side is Side.BUY else (tp <= entry - min_dist)
            out_tp = spec.round_price(tp) if valid else 0.0
        return out_sl, out_tp

    # ------------------------------------------------------------ orders
    async def market_order(
        self,
        symbol: str,
        side: Side,
        volume: float,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        comment: str = "",
        position_ticket: Optional[int] = None,
    ) -> OrderResult:
        if volume <= 0:
            return OrderResult(False, -1, comment="volume<=0")
        spec = await self.symbol_spec(symbol)
        done_codes = self._codes(_SUCCESS)
        retry_codes = self._codes(_RETRYABLE)
        format_codes = self._codes(_FORMAT_ERRORS)
        request: dict[str, Any] = {}
        if not self.dry_run and symbol not in self._formats:
            try:
                await self.probe_order_format(symbol)
            except Exception as exc:
                logger.warning("order format probe failed for {}: {}", symbol, exc)
        candidates = self._format_candidates(spec)
        fmt_idx = 0
        requotes = 0
        attempt = 0

        while True:
            attempt += 1
            tick = await self._tick(symbol)
            price = float(tick.ask if side is Side.BUY else tick.bid)
            request = self._request(spec, side, volume, price, candidates[fmt_idx], comment)
            if position_ticket is not None:
                request["position"] = int(position_ticket)
            else:
                s, t = self.sanitize_stops(spec, side, price, sl, tp)
                if s:
                    request["sl"] = s
                if t:
                    request["tp"] = t

            if self.dry_run:
                return self._paper_fill(spec, side, request, attempt)

            t0 = time.perf_counter_ns()
            result = await self._client.call("order_send", request)
            latency_ms = (time.perf_counter_ns() - t0) / 1e6
            self.orders_sent += 1

            if result is None:
                code, msg = await self._client.last_error()
                self.orders_failed += 1
                logger.error("order_send returned None for {} {}: [{}] {}", side.value, symbol, code, msg)
                return OrderResult(False, code, comment=msg, latency_ms=latency_ms, attempts=attempt, request=request)

            retcode = int(result.retcode)
            if retcode in done_codes:
                self._formats[symbol] = candidates[fmt_idx]
                logger.success(
                    "FILLED {} {} {:.2f} @ {} (sl={} tp={}) {:.2f} ms ticket={}",
                    side.value.upper(), symbol, result.volume or volume, result.price or price,
                    request.get("sl", "-"), request.get("tp", "-"), latency_ms, result.order,
                )
                return OrderResult(
                    True, retcode, ticket=int(result.order), price=float(result.price or price),
                    volume=float(result.volume or volume), latency_ms=latency_ms,
                    comment=str(result.comment), attempts=attempt, request=request,
                )
            if retcode in format_codes and fmt_idx + 1 < len(candidates):
                logger.warning("{} rejected request format ({} {}); trying another filling mode", symbol, retcode, result.comment)
                fmt_idx += 1
                continue
            if retcode in retry_codes and requotes < self.max_retries:
                requotes += 1
                logger.warning("Retryable retcode {} ({}) on {} attempt {}", retcode, result.comment, symbol, attempt)
                continue
            self.orders_failed += 1
            logger.error("Order rejected {} {}: retcode={} {}", side.value, symbol, retcode, result.comment)
            return OrderResult(False, retcode, comment=str(result.comment), latency_ms=latency_ms, attempts=attempt, request=request)

    def _paper_fill(self, spec: SymbolSpec, side: Side, request: dict[str, Any], attempt: int) -> OrderResult:
        price = float(request["price"])
        pos_ticket = request.get("position")
        if pos_ticket is not None:
            pos = self._paper.pop(int(pos_ticket), None)
            if pos is None:
                return OrderResult(False, -1, comment="paper position not found", request=request)
            pnl = pos.side.sign * (price - pos.price_open) * spec.money_per_price_unit(pos.volume)
            self.paper_realized_pnl += pnl
            logger.info("[PAPER] CLOSE {} {} {:.2f} @ {} pnl={:+.2f}", pos.side.value, pos.symbol, pos.volume, price, pnl)
            return OrderResult(True, 0, ticket=pos.ticket, price=price, volume=pos.volume, pnl=round(pnl, 2), attempts=attempt, request=request)

        ticket = next(self._paper_tickets)
        self._paper[ticket] = PositionInfo(
            ticket=ticket, symbol=request["symbol"], side=side, volume=float(request["volume"]),
            price_open=price, sl=float(request.get("sl", 0.0)), tp=float(request.get("tp", 0.0)),
            profit=0.0, magic=self.magic, price_current=price,
        )
        logger.info("[PAPER] OPEN {} {} {:.2f} @ {} sl={} tp={}", side.value, request["symbol"], request["volume"], price, request.get("sl"), request.get("tp"))
        return OrderResult(True, 0, ticket=ticket, price=price, volume=float(request["volume"]), attempts=attempt, request=request)

    async def close_position(self, pos: PositionInfo, comment: str = "close") -> OrderResult:
        result = await self.market_order(
            pos.symbol, pos.side.opposite, pos.volume, comment=comment, position_ticket=pos.ticket
        )
        if result.success and not self.dry_run:
            # best estimate of realised P&L: last marked floating profit
            result = replace(result, pnl=pos.profit)
        return result

    # --------------------------------------------------------- positions
    async def get_positions(self, symbol: Optional[str] = None) -> list[PositionInfo]:
        if self.dry_run:
            return await self._paper_positions(symbol)
        raw = await (self._client.call("positions_get", symbol=symbol) if symbol else self._client.call("positions_get"))
        if raw is None:
            code, msg = await self._client.last_error()
            raise MT5Error(f"positions_get failed: {msg}", code)
        buy_type = self._c("POSITION_TYPE_BUY")
        return [
            PositionInfo(
                ticket=int(p.ticket), symbol=str(p.symbol), side=Side.BUY if p.type == buy_type else Side.SELL,
                volume=float(p.volume), price_open=float(p.price_open), sl=float(p.sl), tp=float(p.tp),
                profit=float(p.profit), magic=int(p.magic), price_current=float(p.price_current),
            )
            for p in raw
            if int(p.magic) == self.magic
        ]

    async def _paper_positions(self, symbol: Optional[str]) -> list[PositionInfo]:
        """Mark paper positions to market and enforce SL/TP."""
        out: list[PositionInfo] = []
        for ticket, pos in list(self._paper.items()):
            if symbol and pos.symbol != symbol:
                continue
            spec = await self.symbol_spec(pos.symbol)
            tick = await self._tick(pos.symbol)
            price = float(tick.bid if pos.side is Side.BUY else tick.ask)
            pos.price_current = price
            pos.profit = round(pos.side.sign * (price - pos.price_open) * spec.money_per_price_unit(pos.volume), 2)
            hit_sl = pos.sl > 0 and (price <= pos.sl if pos.side is Side.BUY else price >= pos.sl)
            hit_tp = pos.tp > 0 and (price >= pos.tp if pos.side is Side.BUY else price <= pos.tp)
            if hit_sl or hit_tp:
                self._paper.pop(ticket, None)
                self.paper_realized_pnl += pos.profit
                logger.info("[PAPER] {} hit on {} #{} pnl={:+.2f}", "SL" if hit_sl else "TP", pos.symbol, ticket, pos.profit)
                continue
            out.append(pos)
        return out

    def paper_floating_pnl(self) -> float:
        return sum(p.profit for p in self._paper.values())

    # --------------------------------------------------------- emergency
    async def cancel_all_pending(self) -> int:
        if self.dry_run:
            return 0
        orders = await self._client.call("orders_get")
        cancelled = 0
        for order in orders or ():
            if int(order.magic) != self.magic:
                continue
            res = await self._client.call("order_send", {"action": self._c("TRADE_ACTION_REMOVE"), "order": int(order.ticket)})
            if res is not None and int(res.retcode) in self._codes(_SUCCESS):
                cancelled += 1
            else:
                logger.error("failed to cancel pending order #{}", order.ticket)
        return cancelled

    async def close_all(self, symbol: Optional[str] = None, comment: str = "close_all") -> list[tuple[PositionInfo, OrderResult]]:
        positions = await self.get_positions(symbol)
        results = await asyncio.gather(*(self.close_position(p, comment) for p in positions), return_exceptions=True)
        out: list[tuple[PositionInfo, OrderResult]] = []
        for pos, res in zip(positions, results):
            if isinstance(res, BaseException):
                logger.error("close #{} failed: {}", pos.ticket, res)
                out.append((pos, OrderResult(False, -1, comment=str(res))))
            else:
                out.append((pos, res))
        return out

    async def emergency_flatten(self, reason: str, attempts: int = 3) -> list[tuple[PositionInfo, OrderResult]]:
        """Cancel pending orders and close every bot position; retries until flat."""
        async with self._flatten_lock:
            logger.critical("EMERGENCY FLATTEN ({})", reason)
            closed: list[tuple[PositionInfo, OrderResult]] = []
            try:
                n = await self.cancel_all_pending()
                if n:
                    logger.warning("cancelled {} pending orders", n)
            except Exception:
                logger.exception("cancel_all_pending failed during flatten")
            for i in range(1, attempts + 1):
                try:
                    closed.extend(r for r in await self.close_all(comment="EMERGENCY") if r[1].success)
                    remaining = await self.get_positions()
                except Exception:
                    logger.exception("flatten attempt {} failed", i)
                    remaining = None
                if remaining == []:
                    logger.critical("Flatten complete: {} positions closed", len(closed))
                    return closed
                await asyncio.sleep(0.2 * i)
            logger.critical("FLATTEN INCOMPLETE after {} attempts - MANUAL INTERVENTION REQUIRED", attempts)
            return closed
