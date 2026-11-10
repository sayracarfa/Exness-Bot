from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from src.data_ingestion import DataIngestor, compute_microprice, parse_book
from src.mt5_client import AsyncMT5Client, MT5Error


class TestMicroprice:
    def test_formula(self) -> None:
        # P = (Va*Pb + Vb*Pa)/(Va+Vb) = (1*100 + 3*101)/4 = 100.75
        assert compute_microprice(100.0, 101.0, bid_volume=3.0, ask_volume=1.0) == pytest.approx(100.75)

    def test_balanced_book_is_mid(self) -> None:
        assert compute_microprice(1.1, 1.2, 5, 5) == pytest.approx(1.15)

    def test_bid_heavy_pulls_toward_ask(self) -> None:
        p = compute_microprice(1.0, 2.0, bid_volume=9, ask_volume=1)
        assert 1.5 < p <= 2.0

    def test_zero_volume_falls_back_to_mid(self) -> None:
        assert compute_microprice(1.0, 1.2, 0, 0) == pytest.approx(1.1)

    @pytest.mark.parametrize("bid,ask,vb,va", [(0, 1, 1, 1), (1.2, 1.1, 1, 1), (1, 1.1, -1, 1)])
    def test_invalid_inputs(self, bid: float, ask: float, vb: float, va: float) -> None:
        with pytest.raises(ValueError):
            compute_microprice(bid, ask, vb, va)

    def test_bounded_by_touch(self) -> None:
        for vb, va in [(0.1, 10), (10, 0.1), (3, 7)]:
            p = compute_microprice(1.0850, 1.0851, vb, va)
            assert 1.0850 <= p <= 1.0851


def test_parse_book_orders_levels() -> None:
    e = lambda t, p, v: SimpleNamespace(type=t, price=p, volume=int(v), volume_dbl=v)  # noqa: E731
    bids, asks = parse_book([e(1, 1.3, 2), e(1, 1.2, 1), e(2, 1.1, 4), e(2, 1.0, 5)], buy_type=2, sell_type=1)
    assert [b.price for b in bids] == [1.1, 1.0]
    assert [a.price for a in asks] == [1.2, 1.3]
    assert asks[0].volume == 1


def test_fetch_symbol_from_simulator(sim) -> None:
    async def run() -> None:
        client = AsyncMT5Client(sim)
        await client.call("initialize")
        ing = DataIngestor(client, ["EURUSDm"], depth_symbols=["EURUSDm"])
        await ing.start()
        snaps = await ing.fetch_symbol("EURUSDm")
        assert len(snaps) == 1
        snap = snaps[-1]
        assert snap.is_latest and not snap.warmup and snap.has_depth
        assert snap.bid <= snap.microprice <= snap.ask
        assert snap.trade_volume == 1.0
        await client.disconnect()

    asyncio.run(run())


def test_batch_returns_all_new_ticks_once() -> None:
    from src.mt5_simulator import MT5Simulator

    sim = MT5Simulator(symbols=["AAPLm"], seed=3, tick_interval_s=0.005)

    async def run() -> None:
        client = AsyncMT5Client(sim)
        await client.call("initialize")
        ing = DataIngestor(client, ["AAPLm"], depth_symbols=["AAPLm"])
        await ing.start()
        assert not ing._depth_enabled["AAPLm"]  # stock CFDs: no DOM
        await asyncio.sleep(0.1)
        batch = await ing.fetch_symbol("AAPLm")
        assert len(batch) >= 5  # ~20 ticks generated while we slept
        assert [s.is_latest for s in batch].count(True) == 1 and batch[-1].is_latest
        times = [s.time_msc for s in batch]
        assert times == sorted(times)
        again = await ing.fetch_symbol("AAPLm")  # no duplicates across calls
        assert all(s.time_msc > times[-1] for s in again)
        await client.disconnect()

    asyncio.run(run())


def test_warmup_replays_history_flagged() -> None:
    from src.mt5_simulator import MT5Simulator

    sim = MT5Simulator(symbols=["EURUSDm"], seed=4, tick_interval_s=0.005)

    async def run() -> None:
        client = AsyncMT5Client(sim)
        await client.call("initialize")
        sim.symbol_info_tick("EURUSDm")
        await asyncio.sleep(0.2)  # build some history
        ing = DataIngestor(client, ["EURUSDm"], warmup_seconds=60)
        await ing.start()
        n = ing.queue.qsize()
        assert n >= 10
        items = [ing.queue.get_nowait() for _ in range(n)]
        assert all(s.warmup and not s.is_latest for s in items)
        assert "EURUSDm" not in ing.latest  # warm-up is not "live"
        await client.disconnect()

    asyncio.run(run())


def test_run_sweeps_universe(sim) -> None:
    async def run() -> None:
        client = AsyncMT5Client(sim)
        await client.call("initialize")
        ing = DataIngestor(client, ["EURUSDm", "XAUUSDm"], poll_interval_s=0.001)
        await ing.start()
        stop = asyncio.Event()
        task = asyncio.create_task(ing.run(stop))
        await asyncio.sleep(0.2)
        stop.set()
        await task
        assert ing.published > 10 and ing.sweeps > 10
        assert set(ing.latest) == {"EURUSDm", "XAUUSDm"}
        await client.disconnect()

    asyncio.run(run())


def test_unknown_symbol_dropped_and_errors_backoff() -> None:
    class Broken:
        BOOK_TYPE_BUY, BOOK_TYPE_SELL, COPY_TICKS_INFO = 2, 1, 1

        def symbol_select(self, symbol, enable=True):
            return symbol != "BADm"

        def symbol_info_tick(self, symbol):
            return None

        def copy_ticks_from(self, *a):
            return None

        def last_error(self):
            return (-10004, "No IPC connection")

    errors: list[str] = []

    async def run() -> DataIngestor:
        client = AsyncMT5Client(Broken())
        ing = DataIngestor(client, ["EURUSDm", "BADm"], error_hook=lambda src, exc: errors.append(src))
        await ing.start()
        assert ing.symbols == ["EURUSDm"]
        with pytest.raises(MT5Error):
            await ing.fetch_symbol("EURUSDm")
        await ing.poll_once()
        await ing.poll_once()  # inside back-off window -> skipped
        return ing

    asyncio.run(run())
    assert errors == ["ingestion:EURUSDm"]


def test_queue_drops_oldest_when_full(snapshot_factory) -> None:
    async def run() -> None:
        ing = DataIngestor(AsyncMT5Client(object()), ["EURUSDm"], queue_maxsize=2)
        for i in range(5):
            ing._publish(snapshot_factory(price=1.0 + i * 1e-4, ts_ns=i))
        assert ing.queue.qsize() == 2
        assert ing.dropped == 3
        assert (await ing.queue.get()).recv_ns == 3

    asyncio.run(run())
