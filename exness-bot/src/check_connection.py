"""Pre-flight check for a real MT5 terminal / Exness demo account.

Run BEFORE starting the bot:

    python -m src.check_connection                 # read-only checks (+ order_check, no order placed)
    python -m src.check_connection --test-order    # also open & close one minimum-lot order (DEMO only)

Checks: terminal connection, account type (demo/real), Algo Trading permission,
symbol availability & contract spec, Depth of Market availability, tick latency
percentiles vs the 50 ms breaker threshold, margin check, optional round-trip order.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from collections import Counter
from typing import Any, Optional

import numpy as np
from loguru import logger

from src.config import Settings, load_settings
from src.models import Side
from src.mt5_client import AsyncMT5Client, MT5Error, load_backend
from src.order_manager import FILLING_NAMES, OrderManager
from src.universe import US_STOCK, discover_universe

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, status: str, check: str, detail: str = "") -> None:
        self.rows.append((status, check, detail))
        log = {PASS: logger.success, WARN: logger.warning, FAIL: logger.error}[status]
        log("[{}] {} {}", status, check, f"- {detail}" if detail else "")

    @property
    def failed(self) -> bool:
        return any(s == FAIL for s, _, _ in self.rows)

    def print_summary(self) -> None:
        print("\n" + "=" * 78)
        for status, check, detail in self.rows:
            print(f"{status:<5} {check:<34} {detail}")
        print("=" * 78)
        print("RESULT:", "FAILED - fix the FAIL items before running the bot" if self.failed else "OK - ready for dry-run / demo testing")


async def _latency(client: AsyncMT5Client, symbol: str, n: int) -> np.ndarray:
    samples = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        await client.call("symbol_info_tick", symbol)
        samples.append((time.perf_counter_ns() - t0) / 1e6)
        await asyncio.sleep(0.002)
    return np.asarray(samples)


async def run_checks(settings: Settings, test_order: bool, allow_real: bool, samples: int, backend: Any = None, detail_per_class: int = 2) -> Report:
    rep = Report()
    client = AsyncMT5Client(backend if backend is not None else load_backend(settings.simulate, symbols=settings.symbols if settings.universe.mode == "manual" else None))

    # 1. connection -----------------------------------------------------
    try:
        await client.connect(settings.mt5, require_demo=False)
    except MT5Error as exc:
        rep.add(FAIL, "MT5 connection", str(exc))
        rep.add(FAIL, "hint", "Is the Exness MT5 terminal installed & logged in? Are MT5_LOGIN/MT5_PASSWORD/MT5_SERVER correct?")
        return rep
    rep.add(PASS, "MT5 connection")

    try:
        # 2. account ----------------------------------------------------
        acct = await client.call("account_info")
        demo = client.is_demo(acct)
        detail = f"login={acct.login} server={getattr(acct, 'server', '?')} balance={acct.balance} {getattr(acct, 'currency', '')} 1:{getattr(acct, 'leverage', '?')}"
        if demo:
            rep.add(PASS, "account is DEMO", detail)
        elif allow_real:
            rep.add(WARN, "account is REAL", detail)
        else:
            rep.add(FAIL, "account is NOT demo", detail + " (use a demo account, or --allow-real)")
        rep.add(PASS if getattr(acct, "trade_allowed", True) else FAIL, "account trade_allowed")

        term = await client.call("terminal_info") if hasattr(client.backend, "terminal_info") else None
        if term is not None:
            algo = bool(getattr(term, "trade_allowed", False))
            rep.add(PASS if algo else FAIL, "terminal Algo Trading enabled",
                    "" if algo else "press the 'Algo Trading' button in the MT5 toolbar")
            ping_ms = float(getattr(term, "ping_last", 0)) / 1000.0  # microseconds -> ms
            if ping_ms:
                lim = settings.risk.max_order_latency_ms
                rep.add(PASS if ping_ms <= lim * 0.6 else WARN, "terminal -> server ping",
                        f"{ping_ms:.1f} ms (order latency limit {lim:.0f} ms; lower ping = better fills)")

        # 3. universe ---------------------------------------------------
        om = OrderManager(client, settings.magic, settings.deviation_points, dry_run=False, max_retries=settings.max_order_retries)
        universe = await discover_universe(client, settings) if settings.universe.mode == "auto" else {}
        if settings.universe.mode == "auto":
            counts = Counter(i.asset_class for i in universe.values())
            rep.add(PASS if universe else FAIL, "universe discovery", f"{len(universe)} symbols {dict(counts)}")
            if US_STOCK in settings.universe.asset_classes and not counts.get(US_STOCK):
                rep.add(WARN, "no US stocks found", "run `python -m src.universe` and adjust universe.stock_path_regex")
            # one full sweep over the universe, as the ingestor does it
            if universe:
                t0 = time.perf_counter()
                for sym in universe:
                    await client.call("symbol_select", sym, True)
                    await client.call("copy_ticks_from", sym, int(time.time()) - 2, 1000, client.const("COPY_TICKS_INFO"))
                sweep_ms = (time.perf_counter() - t0) * 1000
                rep.add(PASS if sweep_ms < 1000 else WARN, "full universe tick sweep",
                        f"{sweep_ms:.0f} ms for {len(universe)} symbols ({sweep_ms / len(universe):.2f} ms/symbol)")
        sample: list[str] = []
        per_class: Counter[str] = Counter()
        for sym, inst in universe.items():
            if per_class[inst.asset_class] < detail_per_class:
                sample.append(sym)
                per_class[inst.asset_class] += 1
        for sym in settings.symbols:
            if sym not in sample:
                sample.append(sym)

        for symbol in sample:
            if not await client.call("symbol_select", symbol, True):
                rep.add(FAIL, f"{symbol} available", "not found - check the suffix in Market Watch (EURUSDm vs EURUSD)")
                continue
            spec = await om.symbol_spec(symbol)
            rep.add(PASS, f"{symbol} spec",
                    f"digits={spec.digits} tick={spec.tick_size} tick_value={spec.tick_value} "
                    f"lots[{spec.volume_min}..{spec.volume_max} step {spec.volume_step}] stops_level={spec.stops_level}")

            tick = await client.call("symbol_info_tick", symbol)
            if tick is None or tick.bid <= 0:
                rep.add(FAIL, f"{symbol} tick", "no quotes (market closed?)")
                continue
            age_s = time.time() - tick.time_msc / 1000.0
            rep.add(PASS if age_s < 60 else WARN, f"{symbol} tick",
                    f"bid={tick.bid} ask={tick.ask} spread={(tick.ask - tick.bid) / spec.point:.0f} pts age~{age_s:.0f}s"
                    + ("" if age_s < 60 else " (stale: market closed or broker/local clock offset)"))

            book: Optional[tuple[Any, ...]] = None
            if await client.call("market_book_add", symbol):
                await asyncio.sleep(0.3)
                book = await client.call("market_book_get", symbol)
                await client.call("market_book_release", symbol)
            if book:
                rep.add(PASS, f"{symbol} depth of market", f"{len(book)} levels")
            else:
                rep.add(WARN, f"{symbol} depth of market", "unavailable -> microprice falls back to mid (normal on Exness)")

            lat = await _latency(client, symbol, samples)
            p50, p99, mx = np.percentile(lat, 50), np.percentile(lat, 99), lat.max()
            rep.add(PASS if p99 < settings.risk.max_latency_ms else WARN, f"{symbol} tick latency",
                    f"p50={p50:.2f} p99={p99:.2f} max={mx:.2f} ms over {samples} calls")

            fmt, chk, attempts = await om.probe_order_format(symbol)
            code = None if chk is None else int(chk.retcode)
            if fmt is not None and code == 0:
                rep.add(PASS, f"{symbol} order_check (min lot)",
                        f"filling={FILLING_NAMES.get(fmt[0])} type_time={'yes' if fmt[1] else 'no'} "
                        f"margin={getattr(chk, 'margin', '?')} free={getattr(chk, 'margin_free', '?')}")
            elif fmt is not None:
                rep.add(WARN, f"{symbol} order_check (min lot)",
                        f"format OK (filling={FILLING_NAMES.get(fmt[0])}) but retcode={code} {getattr(chk, 'comment', '')}")
            else:
                _, msg = await client.last_error()
                echo = getattr(chk, "request", None)
                rep.add(FAIL, f"{symbol} order_check (min lot)",
                        f"symbol filling_mode flags={spec.filling_mode}; tried: {' | '.join(attempts)}; last_error={msg}"
                        + (f"; terminal read request as: {echo}" if echo is not None else ""))

        # 4. optional round trip -----------------------------------------
        if test_order:
            if not demo:
                rep.add(FAIL, "test order", "refused: --test-order only runs on DEMO accounts")
            else:
                symbol = settings.symbols[0]
                spec = await om.symbol_spec(symbol)
                res = await om.market_order(symbol, Side.BUY, spec.volume_min, comment="preflight")
                if not res.success:
                    rep.add(FAIL, "test order open", f"retcode={res.retcode} {res.comment}")
                else:
                    rep.add(PASS if res.latency_ms < settings.risk.max_order_latency_ms else WARN, "test order open",
                            f"#{res.ticket} @ {res.price} in {res.latency_ms:.1f} ms")
                    await asyncio.sleep(1.0)
                    closes = await om.close_all(symbol, comment="preflight close")
                    ok = closes and all(r.success for _, r in closes)
                    lat = max((r.latency_ms for _, r in closes), default=0.0)
                    rep.add(PASS if ok else FAIL, "test order close", f"{len(closes)} closed in {lat:.1f} ms" if ok else "CLOSE MANUALLY in MT5!")
    except Exception as exc:
        logger.exception("pre-flight check crashed")
        rep.add(FAIL, "unexpected error", repr(exc))
    finally:
        await client.disconnect()
    return rep


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="MT5 / Exness pre-flight check")
    p.add_argument("--config")
    p.add_argument("--simulate", action="store_true")
    p.add_argument("--test-order", action="store_true", help="open+close one min-lot order (demo only)")
    p.add_argument("--allow-real", action="store_true", help="do not fail on a real account (read-only checks)")
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--detail-per-class", type=int, default=2, help="symbols per asset class to check in detail")
    args = p.parse_args(argv)

    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<level>{message}</level>")
    settings = load_settings(args.config)
    settings.simulate = settings.simulate or args.simulate
    rep = asyncio.run(run_checks(settings, args.test_order, args.allow_real, args.samples, detail_per_class=args.detail_per_class))
    rep.print_summary()
    return 1 if rep.failed else 0


if __name__ == "__main__":
    sys.exit(main())
