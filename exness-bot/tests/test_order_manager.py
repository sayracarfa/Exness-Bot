from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from src.models import Side
from src.mt5_client import AsyncMT5Client
from src.mt5_simulator import MT5Simulator
from src.order_manager import OrderManager


async def _manager(sim: MT5Simulator, dry_run: bool = False, retries: int = 2) -> tuple[AsyncMT5Client, OrderManager]:
    client = AsyncMT5Client(sim)
    await client.call("initialize")
    return client, OrderManager(client, magic=777, deviation_points=10, dry_run=dry_run, max_retries=retries)


def test_open_and_close_live_path(sim) -> None:
    async def run() -> None:
        client, om = await _manager(sim)
        tick = sim.symbol_info_tick("EURUSDm")
        res = await om.market_order("EURUSDm", Side.BUY, 0.1, sl=tick.bid - 0.002, tp=tick.ask + 0.002, comment="test")
        assert res.success and res.ticket > 0
        assert res.request["type_filling"] == sim.ORDER_FILLING_FOK
        positions = await om.get_positions("EURUSDm")
        assert len(positions) == 1 and positions[0].side is Side.BUY and positions[0].magic == 777
        close = await om.close_position(positions[0])
        assert close.success
        assert await om.get_positions() == []
        await client.disconnect()

    asyncio.run(run())


def test_requote_retry_then_success() -> None:
    class Flaky(MT5Simulator):
        def __init__(self) -> None:
            super().__init__(symbols=["EURUSDm"], seed=1, tick_interval_s=0.0)
            self.sends = 0

        def order_send(self, request):
            self.sends += 1
            if self.sends == 1:
                return SimpleNamespace(retcode=self.TRADE_RETCODE_REQUOTE, comment="Requote", order=0, price=0, volume=0)
            return super().order_send(request)

    async def run() -> None:
        sim = Flaky()
        _, om = await _manager(sim)
        res = await om.market_order("EURUSDm", Side.SELL, 0.05)
        assert res.success and res.attempts == 2 and sim.sends == 2

    asyncio.run(run())


def test_retries_exhausted_and_none_result() -> None:
    class AlwaysRequote(MT5Simulator):
        def order_send(self, request):
            return SimpleNamespace(retcode=self.TRADE_RETCODE_REQUOTE, comment="Requote", order=0, price=0, volume=0)

    class NoneResult(MT5Simulator):
        def order_send(self, request):
            self._err(-10004, "No IPC connection")
            return None

    async def run() -> None:
        _, om = await _manager(AlwaysRequote(symbols=["EURUSDm"], tick_interval_s=0), retries=1)
        res = await om.market_order("EURUSDm", Side.BUY, 0.01)
        assert not res.success and res.attempts == 2 and om.orders_failed == 1

        _, om2 = await _manager(NoneResult(symbols=["EURUSDm"], tick_interval_s=0))
        res2 = await om2.market_order("EURUSDm", Side.BUY, 0.01)
        assert not res2.success and res2.retcode == -10004

    asyncio.run(run())


def test_sanitize_stops(eurusd_spec) -> None:
    sl, tp = OrderManager.sanitize_stops(eurusd_spec, Side.BUY, 1.08500, 1.084001234, 1.08600)
    assert sl == pytest.approx(1.08400) and tp == pytest.approx(1.08600)
    # TP on the wrong side is dropped, SL too close is pushed out
    sl, tp = OrderManager.sanitize_stops(eurusd_spec, Side.SELL, 1.08500, 1.08500, 1.08600)
    assert sl > 1.08500 and tp == 0.0


def test_paper_mode_does_not_send(sim) -> None:
    sent: list[dict] = []
    original = sim.order_send
    sim.order_send = lambda req: sent.append(req) or original(req)

    async def run() -> None:
        _, om = await _manager(sim, dry_run=True)
        res = await om.market_order("EURUSDm", Side.BUY, 0.1)
        assert res.success
        positions = await om.get_positions()
        assert len(positions) == 1
        close = await om.close_position(positions[0])
        assert close.success and isinstance(close.pnl, float)
        assert om.paper_realized_pnl == pytest.approx(close.pnl, abs=0.01)

    asyncio.run(run())
    assert sent == []


def test_emergency_flatten_closes_everything(sim) -> None:
    async def run() -> None:
        _, om = await _manager(sim)
        await om.market_order("EURUSDm", Side.BUY, 0.1)
        await om.market_order("XAUUSDm", Side.SELL, 0.02)
        assert len(await om.get_positions()) == 2
        closed = await om.emergency_flatten("test")
        assert len(closed) == 2
        assert await om.get_positions() == []

    asyncio.run(run())


def test_ignores_foreign_magic(sim) -> None:
    async def run() -> None:
        _, om = await _manager(sim)
        tick = sim.symbol_info_tick("EURUSDm")
        sim.order_send({"action": 1, "symbol": "EURUSDm", "volume": 0.1, "type": 0, "price": tick.ask, "magic": 1, "deviation": 10})
        assert await om.get_positions() == []

    asyncio.run(run())


def test_constants_fallback_when_module_lacks_them() -> None:
    """The real MetaTrader5 module has no SYMBOL_FILLING_FOK/IOC: the client must fall back."""

    class RealLike:
        def __init__(self, inner: MT5Simulator) -> None:
            self._inner = inner

        def __getattr__(self, name: str):
            if name in ("SYMBOL_FILLING_FOK", "SYMBOL_FILLING_IOC"):
                raise AttributeError(name)
            return getattr(self._inner, name)

    async def run() -> None:
        client = AsyncMT5Client(RealLike(MT5Simulator(symbols=["EURUSDm"], tick_interval_s=0)))
        await client.call("initialize")
        assert client.const("SYMBOL_FILLING_FOK") == 1 and client.const("SYMBOL_FILLING_IOC") == 2
        om = OrderManager(client, magic=1, dry_run=False)
        res = await om.market_order("EURUSDm", Side.BUY, 0.01)
        assert res.success and res.request["type_filling"] == 0  # ORDER_FILLING_FOK
        with pytest.raises(AttributeError):
            client.const("NOT_A_REAL_CONSTANT")

    asyncio.run(run())


class KeywordOnlySim(MT5Simulator):
    """Mimics MT5 builds that reject positional arguments for some functions."""

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.positional_rejections = 0

    def _kw_only(self, name, args, kwargs, impl):
        if args:
            self.positional_rejections += 1
            self._err(-2, "Unnamed arguments not allowed")
            return None
        return impl(**kwargs)

    def order_check(self, *args, **kwargs):
        return self._kw_only("order_check", args, kwargs, super().order_check)

    def order_send(self, *args, **kwargs):
        return self._kw_only("order_send", args, kwargs, super().order_send)


def test_keyword_argument_retry() -> None:
    sim = KeywordOnlySim(symbols=["EURUSDm"], seed=2, tick_interval_s=0)

    async def run() -> None:
        client = AsyncMT5Client(sim)
        await client.call("initialize")
        om = OrderManager(client, magic=5, dry_run=False)
        r1 = await om.market_order("EURUSDm", Side.BUY, 0.01)
        r2 = await om.market_order("EURUSDm", Side.SELL, 0.01)
        assert r1.success and r2.success
        # one rejection per function (order_check probe + order_send), then remembered
        assert sim.positional_rejections == 2
        tick = sim.symbol_info_tick("EURUSDm")
        chk = await client.call("order_check", {"action": 1, "symbol": "EURUSDm", "volume": 0.01, "type": 0, "price": tick.ask})
        assert chk is not None and chk.retcode == 0

    asyncio.run(run())


def test_filling_mode_auto_detection() -> None:
    """Broker only accepts IOC and rejects type_time: the manager must find that format."""
    sim = MT5Simulator(symbols=["EURUSDm"], seed=4, tick_interval_s=0,
                       supported_filling={MT5Simulator.ORDER_FILLING_IOC}, reject_type_time=True)

    async def run() -> None:
        client = AsyncMT5Client(sim)
        await client.call("initialize")
        om = OrderManager(client, magic=9, dry_run=False)
        fmt, chk, log = await om.probe_order_format("EURUSDm")
        assert fmt == (MT5Simulator.ORDER_FILLING_IOC, False), log
        assert chk.retcode == 0 and len(log) >= 2
        res = await om.market_order("EURUSDm", Side.BUY, 0.01)
        assert res.success and res.request["type_filling"] == MT5Simulator.ORDER_FILLING_IOC
        assert "type_time" not in res.request

    asyncio.run(run())


def test_filling_fallback_during_send() -> None:
    """Even without a successful probe, order_send rotates through formats."""
    sim = MT5Simulator(symbols=["EURUSDm"], seed=5, tick_interval_s=0,
                       supported_filling={MT5Simulator.ORDER_FILLING_RETURN})

    async def run() -> None:
        client = AsyncMT5Client(sim)
        await client.call("initialize")
        om = OrderManager(client, magic=9, dry_run=False)
        om._formats["EURUSDm"] = (MT5Simulator.ORDER_FILLING_FOK, True)  # stale/wrong cache, probe skipped
        res = await om.market_order("EURUSDm", Side.SELL, 0.01)
        assert res.success and res.request["type_filling"] == MT5Simulator.ORDER_FILLING_RETURN
        assert om._formats["EURUSDm"][0] == MT5Simulator.ORDER_FILLING_RETURN

    asyncio.run(run())


def test_unfixable_format_fails_cleanly() -> None:
    sim = MT5Simulator(symbols=["EURUSDm"], seed=6, tick_interval_s=0, supported_filling=set())

    async def run() -> None:
        client = AsyncMT5Client(sim)
        await client.call("initialize")
        om = OrderManager(client, magic=9, dry_run=False)
        fmt, _, log = await om.probe_order_format("EURUSDm")
        assert fmt is None and len(log) == 6
        res = await om.market_order("EURUSDm", Side.BUY, 0.01)
        assert not res.success and res.retcode == MT5Simulator.TRADE_RETCODE_INVALID_FILL

    asyncio.run(run())


class FieldsOnlySim(MT5Simulator):
    """Mimics the MetaTrader5 build seen on the VPS: a positional dict is rejected
    ("Unnamed arguments not allowed"), ``request=`` is silently ignored (empty request,
    retcode 10013), and only the request fields passed as keyword arguments work."""

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.calls: list[str] = []

    def _fields_only(self, impl, args, kwargs):
        from types import SimpleNamespace

        if args:
            self.calls.append("positional")
            self._err(-2, "Unnamed arguments not allowed")
            return None
        if set(kwargs) == {"request"}:
            self.calls.append("keyword")
            empty = SimpleNamespace(symbol="", volume=0.0, action=0)
            return SimpleNamespace(retcode=10013, comment="Invalid request", request=empty, order=0,
                                   volume=0.0, price=0.0, margin=0.0, margin_free=0.0)
        self.calls.append("fields")
        res = impl(dict(kwargs))
        echo = SimpleNamespace(symbol=kwargs.get("symbol", ""))
        data = res._asdict() if hasattr(res, "_asdict") else dict(vars(res))
        return SimpleNamespace(**data, request=echo)

    def order_check(self, *args, **kwargs):
        return self._fields_only(super().order_check, args, kwargs)

    def order_send(self, *args, **kwargs):
        return self._fields_only(super().order_send, args, kwargs)


def test_fields_only_request_convention() -> None:
    sim = FieldsOnlySim(symbols=["EURUSDm"], seed=8, tick_interval_s=0)

    async def run() -> None:
        client = AsyncMT5Client(sim)
        await client.call("initialize")
        om = OrderManager(client, magic=3, dry_run=False)
        fmt, chk, log = await om.probe_order_format("EURUSDm")
        assert fmt is not None and chk.retcode == 0, log
        r1 = await om.market_order("EURUSDm", Side.BUY, 0.01)
        assert r1.success, r1
        positions = await om.get_positions()
        close = await om.close_position(positions[0])
        assert close.success
        # convention learned once per function, then used directly
        assert sim.calls.count("positional") == 2 and sim.calls.count("fields") == 3  # check, open, close
        assert client._dict_mode == {"order_check": "fields", "order_send": "fields"}

    asyncio.run(run())
