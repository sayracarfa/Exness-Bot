"""In-process simulator implementing the subset of the MetaTrader5 API the bot uses.

It lets the full pipeline run on macOS/Linux/CI without a terminal. Prices follow an
exact-discretised Ornstein-Uhlenbeck process; ticks are kept in a history buffer so
``copy_ticks_from`` / ``copy_ticks_range`` behave like the real terminal. A universe of
FX pairs, metals and US stock CFDs is provided. Constants mirror real MT5 values.
"""

from __future__ import annotations

import itertools
import math
import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Deque, NamedTuple, Optional

import numpy as np

TICK_DTYPE = np.dtype([
    ("time", "<i8"), ("bid", "<f8"), ("ask", "<f8"), ("last", "<f8"), ("volume", "<u8"),
    ("time_msc", "<i8"), ("flags", "<u4"), ("volume_real", "<f8"),
])


class Tick(NamedTuple):
    time: int
    bid: float
    ask: float
    last: float
    volume: int
    time_msc: int
    flags: int
    volume_real: float


class BookInfo(NamedTuple):
    type: int
    price: float
    volume: int
    volume_dbl: float


class SymbolInfo(NamedTuple):
    name: str
    description: str
    path: str
    exchange: str
    currency_base: str
    currency_profit: str
    digits: int
    point: float
    trade_tick_size: float
    trade_tick_value: float
    trade_contract_size: float
    volume_min: float
    volume_max: float
    volume_step: float
    filling_mode: int
    trade_stops_level: int
    trade_mode: int
    spread: int
    visible: bool


class AccountInfo(NamedTuple):
    login: int
    server: str
    currency: str
    leverage: int
    balance: float
    equity: float
    margin: float
    margin_free: float
    trade_allowed: bool
    trade_mode: int = 0  # ACCOUNT_TRADE_MODE_DEMO


class TradePosition(NamedTuple):
    ticket: int
    time: int
    type: int
    magic: int
    volume: float
    price_open: float
    sl: float
    tp: float
    price_current: float
    profit: float
    symbol: str
    comment: str


class OrderSendResult(NamedTuple):
    retcode: int
    deal: int
    order: int
    volume: float
    price: float
    bid: float
    ask: float
    comment: str
    request_id: int


@dataclass(frozen=True)
class _Template:
    description: str
    path: str
    mid: float
    theta: float  # per second
    sd: float  # stationary sd
    spread: float
    digits: int
    tick_value: float
    contract: float
    leverage: int
    exchange: str = ""
    depth: bool = True
    volume_min: float = 0.01


_FX = "Forex\\Majors"
_MET = "Metals"
_STK = "Stocks\\US"

TEMPLATES: dict[str, _Template] = {
    "EURUSD": _Template("Euro vs US Dollar", _FX, 1.08500, 0.6, 0.00025, 0.00008, 5, 1.0, 100_000, 500),
    "GBPUSD": _Template("Great Britain Pound vs US Dollar", _FX, 1.27000, 0.6, 0.00030, 0.00010, 5, 1.0, 100_000, 500),
    "XAUUSD": _Template("Gold vs US Dollar", _MET, 2650.000, 0.5, 0.60, 0.15, 3, 0.1, 100, 200),
    "XAGUSD": _Template("Silver vs US Dollar", _MET, 31.000, 0.5, 0.02, 0.02, 3, 5.0, 5_000, 100),
    "XPTUSD": _Template("Platinum vs US Dollar", _MET, 960.00, 0.5, 0.80, 0.80, 2, 1.0, 100, 50),
    "XPDUSD": _Template("Palladium vs US Dollar", _MET, 1010.00, 0.5, 1.20, 1.50, 2, 1.0, 100, 50),
    "AAPL": _Template("Apple Inc.", _STK, 228.00, 0.4, 0.12, 0.04, 2, 0.01, 1, 20, "NASDAQ", False, 1.0),
    "MSFT": _Template("Microsoft Corporation", _STK, 420.00, 0.4, 0.20, 0.06, 2, 0.01, 1, 20, "NASDAQ", False, 1.0),
    "NVDA": _Template("NVIDIA Corporation", _STK, 138.00, 0.4, 0.10, 0.03, 2, 0.01, 1, 20, "NASDAQ", False, 1.0),
    "AMZN": _Template("Amazon.com Inc.", _STK, 190.00, 0.4, 0.10, 0.04, 2, 0.01, 1, 20, "NASDAQ", False, 1.0),
    "GOOGL": _Template("Alphabet Inc. Class A", _STK, 170.00, 0.4, 0.09, 0.04, 2, 0.01, 1, 20, "NASDAQ", False, 1.0),
    "META": _Template("Meta Platforms Inc.", _STK, 570.00, 0.4, 0.30, 0.10, 2, 0.01, 1, 20, "NASDAQ", False, 1.0),
    "TSLA": _Template("Tesla Inc.", _STK, 250.00, 0.4, 0.25, 0.06, 2, 0.01, 1, 20, "NASDAQ", False, 1.0),
    "JPM": _Template("JPMorgan Chase & Co.", _STK, 220.00, 0.4, 0.10, 0.05, 2, 0.01, 1, 20, "NYSE", False, 1.0),
    "V": _Template("Visa Inc. Class A", _STK, 290.00, 0.4, 0.12, 0.06, 2, 0.01, 1, 20, "NYSE", False, 1.0),
    "KO": _Template("The Coca-Cola Company", _STK, 70.00, 0.4, 0.03, 0.02, 2, 0.01, 1, 20, "NYSE", False, 1.0),
    # non-tradable-by-default classes, to exercise the classifier
    "US500": _Template("US SPX 500 Index", "Indices", 5800.0, 0.5, 1.5, 0.5, 1, 0.1, 1, 200, "", False),
    "BTCUSD": _Template("Bitcoin vs US Dollar", "Crypto", 68_000.0, 0.5, 30.0, 15.0, 2, 0.01, 1, 400, "", False),
}


@dataclass
class _SymbolSim:
    spec: SymbolInfo
    tpl: _Template
    mid: float
    sigma: float
    last_ts: float = field(default_factory=time.time)
    tick: Optional[Tick] = None
    history: Deque[Tick] = field(default_factory=lambda: deque(maxlen=20_000))
    book_enabled: bool = False


class MT5Simulator:
    """Thread-safe fake MT5 terminal."""

    # --- constants (same numeric values as MetaTrader5) ---
    ACCOUNT_TRADE_MODE_DEMO = 0
    ACCOUNT_TRADE_MODE_REAL = 2
    SYMBOL_TRADE_MODE_DISABLED = 0
    SYMBOL_TRADE_MODE_CLOSEONLY = 3
    SYMBOL_TRADE_MODE_FULL = 4
    COPY_TICKS_ALL = -1
    COPY_TICKS_INFO = 1
    COPY_TICKS_TRADE = 2
    TRADE_ACTION_DEAL = 1
    TRADE_ACTION_PENDING = 5
    TRADE_ACTION_SLTP = 6
    TRADE_ACTION_REMOVE = 8
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    ORDER_TIME_GTC = 0
    ORDER_FILLING_FOK = 0
    ORDER_FILLING_IOC = 1
    ORDER_FILLING_RETURN = 2
    SYMBOL_FILLING_FOK = 1
    SYMBOL_FILLING_IOC = 2
    POSITION_TYPE_BUY = 0
    POSITION_TYPE_SELL = 1
    BOOK_TYPE_SELL = 1
    BOOK_TYPE_BUY = 2
    TRADE_RETCODE_REQUOTE = 10004
    TRADE_RETCODE_REJECT = 10006
    TRADE_RETCODE_INVALID = 10013
    TRADE_RETCODE_INVALID_FILL = 10030
    TRADE_RETCODE_PLACED = 10008
    TRADE_RETCODE_DONE = 10009
    TRADE_RETCODE_INVALID_VOLUME = 10014
    TRADE_RETCODE_INVALID_STOPS = 10016
    TRADE_RETCODE_NO_MONEY = 10019
    TRADE_RETCODE_PRICE_CHANGED = 10020
    TRADE_RETCODE_PRICE_OFF = 10021
    TRADE_RETCODE_POSITION_CLOSED = 10036

    def __init__(
        self,
        symbols: Optional[list[str]] = None,
        balance: float = 10_000.0,
        tick_interval_s: float = 0.01,
        seed: Optional[int] = None,
        requote_prob: float = 0.0,
        latency_s: float = 0.0,
        suffix: str = "m",
        supported_filling: Optional[set[int]] = None,
        reject_type_time: bool = False,
    ) -> None:
        self.supported_filling = supported_filling
        self.reject_type_time = reject_type_time
        self._lock = threading.RLock()
        self._rng = random.Random(seed)
        self._tick_interval = tick_interval_s
        self._requote_prob = requote_prob
        self._latency_s = latency_s
        self._balance = balance
        self._initialized = False
        self._last_error: tuple[int, str] = (1, "Success")
        self._tickets = itertools.count(1_000_001)
        self._positions: dict[int, TradePosition] = {}
        self._symbols: dict[str, _SymbolSim] = {}
        names = symbols if symbols is not None else [k + suffix for k in TEMPLATES]
        for name in names:
            self._add_symbol(name)

    # ------------------------------------------------------------------ setup
    def _add_symbol(self, name: str) -> None:
        key = next((k for k in sorted(TEMPLATES, key=len, reverse=True) if name.upper().startswith(k)), "EURUSD")
        tpl = TEMPLATES[key]
        point = 10.0 ** -tpl.digits
        is_fx_like = tpl.path in (_FX, _MET) or key == "BTCUSD"
        spec = SymbolInfo(
            name=name,
            description=tpl.description,
            path=f"{tpl.path}\\{name}",
            exchange=tpl.exchange,
            currency_base=key[:3] if is_fx_like else "USD",
            currency_profit="USD",
            digits=tpl.digits,
            point=point,
            trade_tick_size=point,
            trade_tick_value=tpl.tick_value,
            trade_contract_size=tpl.contract,
            volume_min=tpl.volume_min,
            volume_max=200.0 if tpl.volume_min < 1 else 5_000.0,
            volume_step=tpl.volume_min,
            filling_mode=self.SYMBOL_FILLING_FOK | self.SYMBOL_FILLING_IOC,
            trade_stops_level=0,
            trade_mode=self.SYMBOL_TRADE_MODE_FULL,
            spread=int(round(tpl.spread / point)),
            visible=True,
        )
        self._symbols[name] = _SymbolSim(spec=spec, tpl=tpl, mid=tpl.mid, sigma=tpl.sd * math.sqrt(2.0 * tpl.theta))

    def _sleep(self) -> None:
        if self._latency_s > 0:
            time.sleep(self._latency_s)

    def _err(self, code: int, msg: str) -> None:
        self._last_error = (code, msg)

    # ------------------------------------------------------------- lifecycle
    def initialize(self, path: Optional[str] = None, **kwargs: Any) -> bool:
        self._initialized = True
        self._err(1, "Success")
        return True

    def shutdown(self) -> None:
        self._initialized = False

    def last_error(self) -> tuple[int, str]:
        return self._last_error

    def terminal_info(self) -> Any:
        return SimpleNamespace(trade_allowed=True, connected=True, ping_last=1_500, name="Simulator")

    # ------------------------------------------------------------ market data
    def _step(self, sim: _SymbolSim, dt: float, ts: float) -> None:
        theta = sim.tpl.theta
        decay = math.exp(-theta * dt)
        sd = sim.sigma * math.sqrt((1.0 - decay * decay) / (2.0 * theta))
        sim.mid = sim.tpl.mid + (sim.mid - sim.tpl.mid) * decay + sd * self._rng.gauss(0.0, 1.0)
        digits = sim.spec.digits
        half = sim.tpl.spread / 2.0
        bid = round(sim.mid - half, digits)
        ask = round(sim.mid + half, digits)
        if ask <= bid:
            ask = round(bid + sim.spec.point, digits)
        ms = int(ts * 1000)
        if sim.tick is not None and ms <= sim.tick.time_msc:
            ms = sim.tick.time_msc + 1
        sim.tick = Tick(ms // 1000, bid, ask, 0.0, 0, ms, 6, 0.0)
        sim.history.append(sim.tick)

    def _advance(self, sim: _SymbolSim) -> None:
        now = time.time()
        if sim.tick is None:
            self._step(sim, max(self._tick_interval, 1e-3), now)
            sim.last_ts = now
            return
        dt = now - sim.last_ts
        if self._tick_interval <= 0:
            # deterministic mode (tests): one small fixed step per call, independent of how fast
            # the machine is - wall-clock steps made price moves (and requotes) timing-dependent
            self._step(sim, 1e-3, now)
            sim.last_ts = now
            return
        n = int(dt / self._tick_interval)
        if n <= 0:
            return
        n = min(n, 500)
        start = now - n * self._tick_interval
        for i in range(1, n + 1):
            self._step(sim, self._tick_interval, start + i * self._tick_interval)
        sim.last_ts = now

    def symbols_total(self) -> int:
        return len(self._symbols)

    def symbols_get(self, group: Optional[str] = None) -> tuple[SymbolInfo, ...]:
        with self._lock:
            return tuple(s.spec for s in self._symbols.values())

    def symbol_select(self, symbol: str, enable: bool = True) -> bool:
        with self._lock:
            ok = symbol in self._symbols
            if not ok:
                self._err(-1, f"unknown symbol {symbol}")
            return ok

    def symbol_info(self, symbol: str) -> Optional[SymbolInfo]:
        with self._lock:
            sim = self._symbols.get(symbol)
            return sim.spec if sim else None

    def symbol_info_tick(self, symbol: str) -> Optional[Tick]:
        self._sleep()
        with self._lock:
            sim = self._symbols.get(symbol)
            if sim is None:
                self._err(-1, f"unknown symbol {symbol}")
                return None
            self._advance(sim)
            return sim.tick

    @staticmethod
    def _to_seconds(value: Any) -> float:
        return value.timestamp() if isinstance(value, datetime) else float(value)

    def _ticks_array(self, ticks: list[Tick]) -> np.ndarray:
        return np.array([tuple(t) for t in ticks], dtype=TICK_DTYPE)

    def copy_ticks_from(self, symbol: str, date_from: Any, count: int, flags: int) -> Optional[np.ndarray]:
        self._sleep()
        with self._lock:
            sim = self._symbols.get(symbol)
            if sim is None:
                self._err(-1, f"unknown symbol {symbol}")
                return None
            self._advance(sim)
            start_ms = int(self._to_seconds(date_from) * 1000)
            out = [t for t in sim.history if t.time_msc >= start_ms][: int(count)]
            return self._ticks_array(out)

    def copy_ticks_range(self, symbol: str, date_from: Any, date_to: Any, flags: int) -> Optional[np.ndarray]:
        with self._lock:
            sim = self._symbols.get(symbol)
            if sim is None:
                return None
            self._advance(sim)
            a = int(self._to_seconds(date_from) * 1000)
            b = int(self._to_seconds(date_to) * 1000)
            return self._ticks_array([t for t in sim.history if a <= t.time_msc <= b])

    def market_book_add(self, symbol: str) -> bool:
        with self._lock:
            sim = self._symbols.get(symbol)
            if sim is None or not sim.tpl.depth:
                self._err(-1, "market depth not available")
                return False
            sim.book_enabled = True
            return True

    def market_book_release(self, symbol: str) -> bool:
        with self._lock:
            sim = self._symbols.get(symbol)
            if sim is not None:
                sim.book_enabled = False
            return True

    def market_book_get(self, symbol: str) -> Optional[tuple[BookInfo, ...]]:
        self._sleep()
        with self._lock:
            sim = self._symbols.get(symbol)
            if sim is None or not sim.book_enabled:
                self._err(-1, "market book not subscribed")
                return None
            self._advance(sim)
            assert sim.tick is not None
            levels: list[BookInfo] = []
            step = sim.spec.point * 2
            for i in range(5):
                vol_a = round(self._rng.lognormvariate(0.5, 0.6), 2)
                vol_b = round(self._rng.lognormvariate(0.5, 0.6), 2)
                levels.append(BookInfo(self.BOOK_TYPE_SELL, round(sim.tick.ask + i * step, sim.spec.digits), int(vol_a), vol_a))
                levels.append(BookInfo(self.BOOK_TYPE_BUY, round(sim.tick.bid - i * step, sim.spec.digits), int(vol_b), vol_b))
            levels.sort(key=lambda b: -b.price)
            return tuple(levels)

    # --------------------------------------------------------------- account
    def _profit(self, pos: TradePosition, sim: _SymbolSim) -> tuple[float, float]:
        assert sim.tick is not None
        price = sim.tick.bid if pos.type == self.POSITION_TYPE_BUY else sim.tick.ask
        sign = 1.0 if pos.type == self.POSITION_TYPE_BUY else -1.0
        profit = sign * (price - pos.price_open) / sim.spec.trade_tick_size * sim.spec.trade_tick_value * pos.volume
        return price, round(profit, 2)

    def _margin(self, sim: _SymbolSim, volume: float, price: float) -> float:
        return volume * sim.spec.trade_contract_size * price / sim.tpl.leverage

    def _refresh_positions(self) -> None:
        for ticket, pos in list(self._positions.items()):
            sim = self._symbols[pos.symbol]
            self._advance(sim)
            price, profit = self._profit(pos, sim)
            is_buy = pos.type == self.POSITION_TYPE_BUY
            hit_sl = pos.sl > 0 and ((is_buy and price <= pos.sl) or (not is_buy and price >= pos.sl))
            hit_tp = pos.tp > 0 and ((is_buy and price >= pos.tp) or (not is_buy and price <= pos.tp))
            if hit_sl or hit_tp:
                self._balance += profit
                del self._positions[ticket]
                continue
            self._positions[ticket] = pos._replace(price_current=price, profit=profit)

    def _used_margin(self) -> float:
        return sum(self._margin(self._symbols[p.symbol], p.volume, p.price_open) for p in self._positions.values())

    def account_info(self) -> AccountInfo:
        with self._lock:
            self._refresh_positions()
            floating = sum(p.profit for p in self._positions.values())
            equity = self._balance + floating
            margin = self._used_margin()
            return AccountInfo(
                login=99_999_999, server="Simulator", currency="USD", leverage=500,
                balance=round(self._balance, 2), equity=round(equity, 2), margin=round(margin, 2),
                margin_free=round(equity - margin, 2), trade_allowed=True,
            )

    def positions_get(self, symbol: Optional[str] = None, ticket: Optional[int] = None, **_: Any) -> tuple[TradePosition, ...]:
        with self._lock:
            self._refresh_positions()
            return tuple(
                p for p in self._positions.values()
                if (symbol is None or p.symbol == symbol) and (ticket is None or p.ticket == ticket)
            )

    def orders_get(self, symbol: Optional[str] = None, **_: Any) -> tuple[Any, ...]:
        return ()

    def order_calc_margin(self, action: int, symbol: str, volume: float, price: float) -> Optional[float]:
        with self._lock:
            sim = self._symbols.get(symbol)
            return None if sim is None else round(self._margin(sim, volume, price), 2)

    # ---------------------------------------------------------------- trading
    def _result(self, retcode: int, comment: str, *, volume: float = 0.0, price: float = 0.0, order: int = 0, tick: Optional[Tick] = None) -> OrderSendResult:
        return OrderSendResult(
            retcode=retcode, deal=order, order=order, volume=volume, price=price,
            bid=tick.bid if tick else 0.0, ask=tick.ask if tick else 0.0, comment=comment,
            request_id=self._rng.randint(1, 1_000_000),
        )

    def _format_error(self, request: dict[str, Any]) -> Optional[int]:
        if self.reject_type_time and "type_time" in request:
            return self.TRADE_RETCODE_INVALID
        if self.supported_filling is not None and request.get("type_filling") not in self.supported_filling:
            return self.TRADE_RETCODE_INVALID_FILL
        return None

    def order_check(self, request: dict[str, Any]) -> Any:
        bad = self._format_error(request)
        if bad is not None:
            return SimpleNamespace(retcode=bad, comment="Invalid request", margin=0.0, margin_free=0.0, balance=0.0, equity=0.0)
        with self._lock:
            sim = self._symbols.get(request.get("symbol", ""))
            if sim is None:
                return None
            acct = self.account_info()
            self._advance(sim)
            assert sim.tick is not None
            margin = self._margin(sim, float(request.get("volume", 0)), sim.tick.ask)
            ok = margin <= acct.margin_free
            return SimpleNamespace(retcode=0 if ok else self.TRADE_RETCODE_NO_MONEY, comment="Done" if ok else "No money",
                                   margin=round(margin, 2), margin_free=acct.margin_free, balance=acct.balance, equity=acct.equity)

    def order_send(self, request: dict[str, Any]) -> Optional[OrderSendResult]:
        self._sleep()
        with self._lock:
            action = request.get("action")
            if action == self.TRADE_ACTION_REMOVE:
                return self._result(self.TRADE_RETCODE_DONE, "Request executed")
            if action != self.TRADE_ACTION_DEAL:
                return self._result(self.TRADE_RETCODE_REJECT, "unsupported action")

            symbol = request.get("symbol", "")
            sim = self._symbols.get(symbol)
            if sim is None:
                self._err(-2, f"unknown symbol {symbol}")
                return None
            self._advance(sim)
            tick = sim.tick
            assert tick is not None
            bad = self._format_error(request)
            if bad is not None:
                return self._result(bad, "Invalid request", tick=tick)

            volume = float(request.get("volume", 0.0))
            if volume < sim.spec.volume_min or volume > sim.spec.volume_max:
                return self._result(self.TRADE_RETCODE_INVALID_VOLUME, "Invalid volume", tick=tick)
            if self._rng.random() < self._requote_prob:
                return self._result(self.TRADE_RETCODE_REQUOTE, "Requote", tick=tick)

            order_type = request.get("type")
            fill_price = tick.ask if order_type == self.ORDER_TYPE_BUY else tick.bid
            deviation = int(request.get("deviation", 0)) * sim.spec.point
            req_price = float(request.get("price", fill_price))
            if abs(req_price - fill_price) > deviation + 1e-12:
                return self._result(self.TRADE_RETCODE_PRICE_CHANGED, "Price changed", tick=tick)

            ticket = next(self._tickets)
            pos_ticket = request.get("position")
            if pos_ticket:
                pos = self._positions.get(int(pos_ticket))
                if pos is None:
                    return self._result(self.TRADE_RETCODE_POSITION_CLOSED, "Position closed", tick=tick)
                _, profit = self._profit(pos, sim)
                self._balance += profit
                del self._positions[pos.ticket]
                return self._result(self.TRADE_RETCODE_DONE, "Request executed", volume=volume, price=fill_price, order=ticket, tick=tick)

            equity = self._balance + sum(p.profit for p in self._positions.values())
            if self._used_margin() + self._margin(sim, volume, fill_price) > equity:
                return self._result(self.TRADE_RETCODE_NO_MONEY, "No money", tick=tick)

            sl = float(request.get("sl", 0.0) or 0.0)
            tp = float(request.get("tp", 0.0) or 0.0)
            is_buy = order_type == self.ORDER_TYPE_BUY
            if (sl and ((is_buy and sl >= fill_price) or (not is_buy and sl <= fill_price))) or (
                tp and ((is_buy and tp <= fill_price) or (not is_buy and tp >= fill_price))
            ):
                return self._result(self.TRADE_RETCODE_INVALID_STOPS, "Invalid stops", tick=tick)

            self._positions[ticket] = TradePosition(
                ticket=ticket, time=int(time.time()), type=self.POSITION_TYPE_BUY if is_buy else self.POSITION_TYPE_SELL,
                magic=int(request.get("magic", 0)), volume=volume, price_open=fill_price, sl=sl, tp=tp,
                price_current=fill_price, profit=0.0, symbol=symbol, comment=str(request.get("comment", "")),
            )
            return self._result(self.TRADE_RETCODE_DONE, "Request executed", volume=volume, price=fill_price, order=ticket, tick=tick)
