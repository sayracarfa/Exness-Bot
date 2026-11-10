"""Asynchronous MT5 tick + Depth-of-Market (Level-2) collector for large universes.

The MT5 Python API has no push callbacks. Polling ``symbol_info_tick`` for hundreds of
symbols would both saturate the single MT5 I/O thread and silently skip ticks, so the
collector sweeps the universe with ``copy_ticks_from`` and a per-symbol cursor: every
sweep returns ALL ticks since the previous one (nothing is lost, VPIN volume stays
correct). Only the newest tick of each batch is marked ``is_latest`` and is eligible
to trigger trades; older ticks just update the models. On start-up recent history is
replayed as ``warmup`` ticks so models are ready immediately.

Micro-price (Stoikov):

    P_micro = (V_a * P_b + V_b * P_a) / (V_a + V_b)

Book volumes are only known when DOM is available (FX/metals on some accounts); the
micro-price then falls back to the mid price.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

from loguru import logger

from src.mt5_client import AsyncMT5Client, MT5Error

ErrorHook = Callable[[str, BaseException], None]
SuccessHook = Callable[[], None]


@dataclass(frozen=True, slots=True)
class BookLevel:
    price: float
    volume: float


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    symbol: str
    time_msc: int  # broker tick time (ms)
    recv_ns: int  # local wall-clock receive time (ns, epoch)
    recv_perf_ns: int  # local monotonic receive time (ns) for latency measurement
    bid: float
    ask: float
    bid_volume: float
    ask_volume: float
    microprice: float
    trade_volume: float  # volume attributed to this update (tick-count fallback = 1)
    has_depth: bool
    bids: tuple[BookLevel, ...] = field(default=())
    asks: tuple[BookLevel, ...] = field(default=())
    is_latest: bool = True  # newest tick of its batch -> may trigger trades
    warmup: bool = False  # historical replay -> never trades

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def imbalance(self) -> float:
        total = self.bid_volume + self.ask_volume
        return 0.0 if total <= 0 else (self.bid_volume - self.ask_volume) / total


def compute_microprice(best_bid: float, best_ask: float, bid_volume: float, ask_volume: float) -> float:
    """P_micro = (Va*Pb + Vb*Pa) / (Va + Vb); falls back to mid when volumes are unknown."""
    if best_bid <= 0 or best_ask <= 0:
        raise ValueError(f"non-positive quotes bid={best_bid} ask={best_ask}")
    if best_ask < best_bid:
        raise ValueError(f"crossed book bid={best_bid} > ask={best_ask}")
    if bid_volume < 0 or ask_volume < 0:
        raise ValueError("volumes must be non-negative")
    total = bid_volume + ask_volume
    if total <= 0:
        return 0.5 * (best_bid + best_ask)
    return (ask_volume * best_bid + bid_volume * best_ask) / total


def parse_book(entries: Iterable[Any], buy_type: int, sell_type: int) -> tuple[tuple[BookLevel, ...], tuple[BookLevel, ...]]:
    """Split MT5 ``BookInfo`` entries into (bids desc, asks asc)."""
    bids: list[BookLevel] = []
    asks: list[BookLevel] = []
    for entry in entries:
        volume = float(getattr(entry, "volume_dbl", 0.0) or getattr(entry, "volume", 0.0))
        level = BookLevel(float(entry.price), volume)
        if entry.type == buy_type:
            bids.append(level)
        elif entry.type == sell_type:
            asks.append(level)
    bids.sort(key=lambda lv: -lv.price)
    asks.sort(key=lambda lv: lv.price)
    return tuple(bids), tuple(asks)


@dataclass
class _Cursor:
    msc: int = 0
    keys_at_msc: set[tuple[float, float]] = field(default_factory=set)


class DataIngestor:
    """Sweeps the symbol universe and publishes :class:`MarketSnapshot` objects."""

    def __init__(
        self,
        client: AsyncMT5Client,
        symbols: list[str],
        poll_interval_s: float = 0.01,
        depth_symbols: Optional[Iterable[str]] = None,
        queue_maxsize: int = 500_000,
        batch_size: int = 5_000,
        warmup_seconds: float = 0.0,
        error_hook: Optional[ErrorHook] = None,
        success_hook: Optional[SuccessHook] = None,
    ) -> None:
        if not symbols:
            raise ValueError("at least one symbol is required")
        self._client = client
        self.symbols = list(symbols)
        self._poll_interval_s = poll_interval_s
        self._want_depth = set(depth_symbols or ())
        self._depth_enabled: dict[str, bool] = {s: False for s in symbols}
        self._cursors: dict[str, _Cursor] = {s: _Cursor() for s in symbols}
        self._batch_size = batch_size
        self._warmup_seconds = warmup_seconds
        self._error_hook = error_hook
        self._success_hook = success_hook
        self._sym_errors: dict[str, int] = {s: 0 for s in symbols}
        self._skip_until: dict[str, float] = {}
        self.queue: asyncio.Queue[MarketSnapshot] = asyncio.Queue(maxsize=queue_maxsize)
        self.latest: dict[str, MarketSnapshot] = {}
        self.dropped = 0
        self.published = 0
        self.sweeps = 0
        self.last_sweep_ms = 0.0

    # ---------------------------------------------------------- lifecycle
    async def start(self) -> None:
        """Select symbols, subscribe DOM where wanted, replay warm-up history."""
        valid: list[str] = []
        for symbol in self.symbols:
            if not await self._client.call("symbol_select", symbol, True):
                code, msg = await self._client.last_error()
                logger.warning("symbol_select({}) failed: [{}] {} - symbol dropped", symbol, code, msg)
                continue
            valid.append(symbol)
            if symbol in self._want_depth:
                self._depth_enabled[symbol] = bool(await self._client.call("market_book_add", symbol))
        if not valid:
            raise MT5Error("no tradable symbols could be selected")
        self.symbols = valid
        n_depth = sum(self._depth_enabled.get(s, False) for s in valid)
        logger.info("Subscribed {} symbols ({} with depth of market)", len(valid), n_depth)

        for symbol in valid:
            try:
                await self._warmup(symbol)
            except Exception as exc:
                logger.warning("warm-up failed for {}: {}", symbol, exc)

    async def stop(self) -> None:
        for symbol, enabled in self._depth_enabled.items():
            if enabled:
                try:
                    await self._client.call("market_book_release", symbol)
                except Exception:
                    logger.exception("market_book_release({}) failed", symbol)

    # ---------------------------------------------------------- fetching
    def _flags(self) -> int:
        try:
            return self._client.const("COPY_TICKS_INFO")
        except AttributeError:
            return 1

    async def _warmup(self, symbol: str) -> None:
        tick = await self._client.call("symbol_info_tick", symbol)
        if tick is None:
            return
        last_msc = int(tick.time_msc)
        cur = self._cursors[symbol]
        if self._warmup_seconds > 0:
            end_s = last_msc / 1000.0 + 1
            ticks = await self._client.call("copy_ticks_range", symbol, int(end_s - self._warmup_seconds), int(end_s) + 1, self._flags())
            if ticks is not None and len(ticks):
                for snap in self._to_snapshots(symbol, ticks[-self._batch_size:], cur, warmup=True):
                    self._publish(snap)
        if cur.msc < last_msc:
            cur.msc = last_msc
            cur.keys_at_msc = {(float(tick.bid), float(tick.ask))}

    def _to_snapshots(self, symbol: str, ticks: Any, cur: _Cursor, warmup: bool = False, book: Optional[tuple[tuple[BookLevel, ...], tuple[BookLevel, ...]]] = None) -> list[MarketSnapshot]:
        """Convert a numpy tick array to snapshots, skipping ticks at/before the cursor."""
        recv_perf_ns = time.perf_counter_ns()
        recv_ns = time.time_ns()
        fresh: list[tuple[int, float, float, float]] = []
        for t in ticks:
            msc = int(t["time_msc"])
            bid, ask = float(t["bid"]), float(t["ask"])
            if msc < cur.msc or (msc == cur.msc and (bid, ask) in cur.keys_at_msc):
                continue
            if bid <= 0 or ask <= 0 or ask < bid:
                continue
            if msc > cur.msc:
                cur.msc = msc
                cur.keys_at_msc = set()
            cur.keys_at_msc.add((bid, ask))
            vol = float(t["volume_real"]) if float(t["volume_real"]) > 0 else 1.0
            fresh.append((msc, bid, ask, vol))

        out: list[MarketSnapshot] = []
        last_i = len(fresh) - 1
        for i, (msc, bid, ask, vol) in enumerate(fresh):
            bids: tuple[BookLevel, ...] = ()
            asks: tuple[BookLevel, ...] = ()
            bv = av = 0.0
            has_depth = False
            if i == last_i and book and book[0] and book[1] and book[0][0].price <= book[1][0].price:
                bids, asks = book
                has_depth = True
                bid, ask = bids[0].price, asks[0].price
                bv, av = bids[0].volume, asks[0].volume
            out.append(MarketSnapshot(
                symbol=symbol, time_msc=msc, recv_ns=recv_ns, recv_perf_ns=recv_perf_ns,
                bid=bid, ask=ask, bid_volume=bv, ask_volume=av,
                microprice=compute_microprice(bid, ask, bv, av), trade_volume=vol,
                has_depth=has_depth, bids=bids, asks=asks,
                is_latest=(i == last_i and not warmup), warmup=warmup,
            ))
        return out

    async def fetch_symbol(self, symbol: str) -> list[MarketSnapshot]:
        """All new ticks of ``symbol`` since its cursor (newest flagged ``is_latest``)."""
        cur = self._cursors[symbol]
        from_s = max(cur.msc // 1000, 0) if cur.msc else int(time.time()) - 1
        ticks = await self._client.call("copy_ticks_from", symbol, from_s, self._batch_size, self._flags())
        if ticks is None:
            code, msg = await self._client.last_error()
            raise MT5Error(f"copy_ticks_from({symbol}) failed: {msg}", code)
        if len(ticks) == 0:
            return []
        book = None
        if self._depth_enabled.get(symbol):
            raw = await self._client.call("market_book_get", symbol)
            if raw:
                book = parse_book(raw, self._client.const("BOOK_TYPE_BUY"), self._client.const("BOOK_TYPE_SELL"))
        return self._to_snapshots(symbol, ticks, cur, book=book)

    def _publish(self, snap: MarketSnapshot) -> None:
        if not snap.warmup:
            self.latest[snap.symbol] = snap
        try:
            self.queue.put_nowait(snap)
        except asyncio.QueueFull:
            try:
                self.queue.get_nowait()
                self.dropped += 1
            except asyncio.QueueEmpty:
                pass
            self.queue.put_nowait(snap)
        self.published += 1

    async def poll_once(self) -> int:
        """One sweep over the universe. Returns number of snapshots published."""
        n = 0
        now = time.monotonic()
        for symbol in self.symbols:
            if self._skip_until.get(symbol, 0.0) > now:
                continue
            try:
                for snap in await self.fetch_symbol(symbol):
                    self._publish(snap)
                    n += 1
                self._sym_errors[symbol] = 0
                if self._success_hook:
                    self._success_hook()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                errs = self._sym_errors[symbol] = self._sym_errors.get(symbol, 0) + 1
                backoff = min(0.05 * 2 ** errs, 30.0)
                self._skip_until[symbol] = now + backoff
                logger.error("ingestion error on {} (#{}; retry in {:.2f}s): {}", symbol, errs, backoff, exc)
                if self._error_hook:
                    self._error_hook(f"ingestion:{symbol}", exc)
            await asyncio.sleep(0)  # let the strategy consume between symbols
        self.sweeps += 1
        return n

    async def run(self, stop: asyncio.Event) -> None:
        logger.info("Data ingestion started: {} symbols, sweep pause {:.1f} ms", len(self.symbols), self._poll_interval_s * 1000)
        while not stop.is_set():
            t0 = time.perf_counter()
            await self.poll_once()
            self.last_sweep_ms = (time.perf_counter() - t0) * 1000
            await asyncio.sleep(self._poll_interval_s)
        logger.info("Data ingestion stopped (published={}, dropped={}, sweeps={})", self.published, self.dropped, self.sweeps)
