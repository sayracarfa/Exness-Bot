"""Async wrapper around the (blocking, non thread-safe) MetaTrader5 Python API.

The official ``MetaTrader5`` package is synchronous and not thread-safe, so every call
is funnelled through ONE dedicated worker thread. This keeps the asyncio event loop
free while guaranteeing serialized access to the terminal IPC channel. Every call is
timed and reported to an optional latency hook (used by the circuit breaker).
"""

from __future__ import annotations

import asyncio
import functools
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional

from loguru import logger

from src.config import MT5Credentials

LatencyHook = Callable[[str, float], None]

# Numeric values of the MT5 constants the bot uses (identical to MQL5). The official
# Python module does not export every one of them (e.g. SYMBOL_FILLING_FOK/IOC are
# missing), so these are used as a fallback.
MT5_CONSTANTS: dict[str, int] = {
    "ACCOUNT_TRADE_MODE_DEMO": 0, "ACCOUNT_TRADE_MODE_CONTEST": 1, "ACCOUNT_TRADE_MODE_REAL": 2,
    "SYMBOL_TRADE_MODE_DISABLED": 0, "SYMBOL_TRADE_MODE_LONGONLY": 1, "SYMBOL_TRADE_MODE_SHORTONLY": 2,
    "SYMBOL_TRADE_MODE_CLOSEONLY": 3, "SYMBOL_TRADE_MODE_FULL": 4,
    "SYMBOL_FILLING_FOK": 1, "SYMBOL_FILLING_IOC": 2, "SYMBOL_FILLING_BOC": 4,
    "TIMEFRAME_M1": 1, "TIMEFRAME_M5": 5, "TIMEFRAME_H1": 16385, "TIMEFRAME_D1": 16408,
    "COPY_TICKS_ALL": -1, "COPY_TICKS_INFO": 1, "COPY_TICKS_TRADE": 2,
    "TRADE_ACTION_DEAL": 1, "TRADE_ACTION_PENDING": 5, "TRADE_ACTION_SLTP": 6,
    "TRADE_ACTION_MODIFY": 7, "TRADE_ACTION_REMOVE": 8, "TRADE_ACTION_CLOSE_BY": 10,
    "ORDER_TYPE_BUY": 0, "ORDER_TYPE_SELL": 1,
    "ORDER_TIME_GTC": 0, "ORDER_FILLING_FOK": 0, "ORDER_FILLING_IOC": 1, "ORDER_FILLING_RETURN": 2,
    "POSITION_TYPE_BUY": 0, "POSITION_TYPE_SELL": 1,
    "BOOK_TYPE_SELL": 1, "BOOK_TYPE_BUY": 2, "BOOK_TYPE_SELL_MARKET": 3, "BOOK_TYPE_BUY_MARKET": 4,
    "TRADE_RETCODE_REQUOTE": 10004, "TRADE_RETCODE_REJECT": 10006, "TRADE_RETCODE_PLACED": 10008,
    "TRADE_RETCODE_DONE": 10009, "TRADE_RETCODE_INVALID": 10013, "TRADE_RETCODE_INVALID_FILL": 10030,
    "TRADE_RETCODE_INVALID_VOLUME": 10014, "TRADE_RETCODE_INVALID_STOPS": 10016,
    "TRADE_RETCODE_NO_MONEY": 10019, "TRADE_RETCODE_PRICE_CHANGED": 10020, "TRADE_RETCODE_PRICE_OFF": 10021,
    "TRADE_RETCODE_CLIENT_DISABLES_AT": 10027, "TRADE_RETCODE_POSITION_CLOSED": 10036,
}


# Parameter names of MT5 functions (MetaTrader5 Python docs). Some builds reject positional
# arguments for some functions ("Unnamed arguments not allowed"); the client then retries
# with these keyword names and remembers the function.
MT5_SIGNATURES: dict[str, tuple[str, ...]] = {
    "order_calc_margin": ("action", "symbol", "volume", "price"),
    "order_calc_profit": ("action", "symbol", "volume", "price_open", "price_close"),
    "copy_ticks_from": ("symbol", "date_from", "count", "flags"),
    "copy_ticks_range": ("symbol", "date_from", "date_to", "flags"),
    "copy_rates_range": ("symbol", "timeframe", "date_from", "date_to"),
    "copy_rates_from_pos": ("symbol", "timeframe", "start_pos", "count"),
    "symbol_select": ("symbol", "enable"),
    "symbol_info": ("symbol",),
    "symbol_info_tick": ("symbol",),
    "market_book_add": ("symbol",),
    "market_book_get": ("symbol",),
    "market_book_release": ("symbol",),
}


DICT_REQUEST_FUNCS = frozenset({"order_check", "order_send"})
REQUEST_MODES = ("positional", "fields", "keyword")


class MT5Error(RuntimeError):
    """Raised when the MT5 terminal returns an error or is unavailable."""

    def __init__(self, message: str, code: Optional[int] = None) -> None:
        super().__init__(message if code is None else f"{message} (code={code})")
        self.code = code


def load_backend(simulate: bool, **sim_kwargs: Any) -> Any:
    """Return the real ``MetaTrader5`` module or the in-process simulator."""
    if simulate:
        from src.mt5_simulator import MT5Simulator

        return MT5Simulator(**sim_kwargs)
    try:
        import MetaTrader5 as mt5  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - platform dependent
        raise MT5Error(
            "MetaTrader5 package is not available. It only runs on Windows with an MT5 "
            "terminal installed. Use --simulate on other platforms."
        ) from exc
    return mt5


class AsyncMT5Client:
    """Serialize blocking MT5 calls onto a single worker thread and time them."""

    def __init__(self, backend: Any, latency_hook: Optional[LatencyHook] = None) -> None:
        self._backend = backend
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mt5-io")
        self._latency_hook = latency_hook
        self._connected = False
        self._closed = False
        self._keyword_only: set[str] = set()
        self._dict_mode: dict[str, str] = {}  # order_check / order_send calling convention

    @property
    def backend(self) -> Any:
        return self._backend

    @property
    def connected(self) -> bool:
        return self._connected

    def set_latency_hook(self, hook: Optional[LatencyHook]) -> None:
        self._latency_hook = hook

    def const(self, name: str) -> int:
        """Read an MT5 constant (e.g. ``TRADE_ACTION_DEAL``); falls back to the MQL5 value."""
        value = getattr(self._backend, name, None)
        if value is None:
            if name not in MT5_CONSTANTS:
                raise AttributeError(f"unknown MT5 constant {name}")
            return MT5_CONSTANTS[name]
        return int(value)

    async def call(self, fn_name: str, *args: Any, **kwargs: Any) -> Any:
        """Run ``backend.<fn_name>(*args, **kwargs)`` on the MT5 worker thread."""
        if self._closed:
            raise MT5Error(f"client closed; cannot call {fn_name}")
        fn = getattr(self._backend, fn_name)
        loop = asyncio.get_running_loop()
        t0 = time.perf_counter_ns()
        try:
            return await loop.run_in_executor(self._executor, functools.partial(self._invoke, fn_name, fn, args, kwargs))
        finally:
            latency_ms = (time.perf_counter_ns() - t0) / 1e6
            if self._latency_hook is not None:
                try:
                    self._latency_hook(fn_name, latency_ms)
                except Exception:  # never let a monitoring hook break trading I/O
                    logger.exception("latency hook failed")

    def _invoke(self, fn_name: str, fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        """Runs on the MT5 worker thread. Retries with keyword arguments when the terminal
        build rejects positional ones (the request was not processed, so a retry is safe)."""
        if fn_name in DICT_REQUEST_FUNCS and len(args) == 1 and isinstance(args[0], dict) and not kwargs:
            return self._invoke_request(fn_name, fn, args[0])
        names = MT5_SIGNATURES.get(fn_name)
        if names and args and fn_name in self._keyword_only:
            return fn(**{**dict(zip(names, args)), **kwargs})
        result = fn(*args, **kwargs)
        if result is None and names and args:
            try:
                err = str(self._backend.last_error())
            except Exception:
                err = ""
            if "unnamed arguments" in err.lower():
                self._keyword_only.add(fn_name)
                logger.info("MT5 {}: switching to keyword arguments", fn_name)
                return fn(**{**dict(zip(names, args)), **kwargs})
        return result

    @staticmethod
    def _request_parsed(result: Any, request: dict[str, Any]) -> bool:
        """True unless the terminal's echo of the request shows it was read as empty."""
        echo = getattr(result, "request", None)
        if echo is None or "symbol" not in request:
            return True
        return str(getattr(echo, "symbol", "")) == str(request["symbol"])

    def _invoke_request(self, fn_name: str, fn: Callable[..., Any], request: dict[str, Any]) -> Any:
        """order_check / order_send: MetaTrader5 builds differ in how the request dict must be
        passed (positional dict, the fields as keyword arguments, or ``request=``). Try each,
        accept the first one whose echo proves the terminal actually read the request, and
        remember it. A request the terminal did not read is not executed, so retrying is safe."""
        cached = self._dict_mode.get(fn_name)
        modes = ([cached] if cached else []) + [m for m in REQUEST_MODES if m != cached]
        result: Any = None
        for mode in modes:
            try:
                if mode == "positional":
                    result = fn(request)
                elif mode == "fields":
                    result = fn(**request)
                else:
                    result = fn(request=request)
            except TypeError:
                continue  # this build does not accept that calling convention
            if result is None:
                try:
                    err = str(self._backend.last_error()).lower()
                except Exception:
                    err = ""
                if "argument" in err:
                    continue
                return None  # a genuine failure (e.g. no connection): let the caller handle it
            if self._request_parsed(result, request):
                if cached != mode:
                    self._dict_mode[fn_name] = mode
                    logger.info("MT5 {}: using '{}' request convention", fn_name, mode)
                return result
        return result

    async def last_error(self) -> tuple[int, str]:
        try:
            code, msg = await self.call("last_error")
            return int(code), str(msg)
        except Exception:
            return -1, "unknown error"

    def is_demo(self, account: Any) -> bool:
        demo_mode = int(getattr(self._backend, "ACCOUNT_TRADE_MODE_DEMO", 0))
        return int(getattr(account, "trade_mode", -1)) == demo_mode

    async def connect(self, creds: MT5Credentials, require_demo: bool = False) -> None:
        """Initialize the terminal connection and verify the trading account."""
        args: tuple[Any, ...] = (creds.terminal_path,) if creds.terminal_path else ()
        kwargs: dict[str, Any] = {"timeout": creds.timeout_ms}
        if creds.login:
            kwargs.update(
                login=creds.login,
                password=creds.password.get_secret_value(),
                server=creds.server,
            )
        ok = await self.call("initialize", *args, **kwargs)
        if not ok:
            code, msg = await self.last_error()
            raise MT5Error(f"MT5 initialize failed: {msg}", code)

        account = await self.call("account_info")
        if account is None:
            code, msg = await self.last_error()
            raise MT5Error(f"account_info unavailable after initialize: {msg}", code)
        self._connected = True
        if require_demo and not self.is_demo(account):
            await self.disconnect()
            raise MT5Error(
                f"Account {getattr(account, 'login', '?')} is NOT a demo account "
                f"(trade_mode={getattr(account, 'trade_mode', '?')}); refusing to start. "
                "Set require_demo=false (BOT_REQUIRE_DEMO=false) only when you intend to trade real money."
            )
        logger.info(
            "Connected to MT5 | server={} login={} balance={:.2f} {} leverage=1:{}",
            getattr(account, "server", "?"),
            getattr(account, "login", "?"),
            float(account.balance),
            getattr(account, "currency", ""),
            getattr(account, "leverage", "?"),
        )
        if not getattr(account, "trade_allowed", True):
            logger.warning("Trading is NOT allowed on this account (trade_allowed=False)")

    async def disconnect(self) -> None:
        if self._closed:
            return
        try:
            if self._connected:
                await self.call("shutdown")
        except Exception:
            logger.exception("MT5 shutdown failed")
        finally:
            self._connected = False
            self._closed = True
            self._executor.shutdown(wait=False, cancel_futures=True)
            logger.info("MT5 connection closed")
