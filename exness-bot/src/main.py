"""Entry point: wires universe discovery, ingestion, NLP, strategy, risk and execution.

Usage
-----
    python -m src.main --simulate --duration 60      # full pipeline against the MT5 simulator
    python -m src.main                               # real MT5 terminal, dry-run (paper) mode
    python -m src.main --live --i-understand-the-risk   # real orders (demo enforced by default)
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from loguru import logger

from src.api import create_app, serve_api
from src.config import Settings, load_settings
from src.data_ingestion import DataIngestor, MarketSnapshot
from src.models import PositionInfo, Side
from src.mt5_client import AsyncMT5Client, load_backend
from src.nlp_engine import NLPEngine, build_sentiment_model, make_synthetic_source
from src.order_manager import OrderManager, OrderResult
from src.portfolio_risk import PortfolioRisk
from src.risk_controller import RiskController
from src.sessions import SessionCalendar
from src.strategy import Signal, SignalAction, SignalEngine
from src.universe import OTHER, Instrument, discover_universe


def setup_logging(settings: Settings) -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
        enqueue=True,
    )
    Path(settings.log_dir).mkdir(parents=True, exist_ok=True)
    logger.add(
        Path(settings.log_dir) / "bot_{time:YYYYMMDD}.log",
        level="DEBUG",
        rotation="50 MB",
        retention="14 days",
        compression="zip",
        enqueue=True,
        backtrace=True,
        diagnose=False,  # never dump local variables (credentials) into logs
    )


class TradingBot:
    def __init__(self, settings: Settings, backend: Any = None) -> None:
        self.settings = settings
        if backend is None:
            sim_symbols = settings.symbols if settings.universe.mode == "manual" else None
            backend = load_backend(settings.simulate, symbols=sim_symbols, tick_interval_s=0.02)
        self.risk = RiskController(settings.risk, settings.strategy.vpin_threshold)
        self.client = AsyncMT5Client(backend, latency_hook=self.risk.record_latency)
        self.orders = OrderManager(
            self.client, settings.magic, settings.deviation_points, settings.dry_run, settings.max_order_retries
        )
        self.risk.on_trip = self._on_breaker_trip
        self.portfolio: Optional[PortfolioRisk] = PortfolioRisk(settings.portfolio) if settings.portfolio.enabled else None
        self.portfolio_var: Optional[float] = None

        # built after universe discovery (needs the terminal)
        self.instruments: dict[str, Instrument] = {}
        self.ingestor: Optional[DataIngestor] = None
        self.nlp: Optional[NLPEngine] = None
        self.sessions: Optional[SessionCalendar] = None
        self.strategy: Optional[SignalEngine] = None

        self._stop = asyncio.Event()
        self._positions: dict[str, list[PositionInfo]] = {}
        self._last_profit: dict[int, float] = {}
        self._bot_closed: set[int] = set()
        self._in_flight: set[str] = set()
        self._pos_lock = asyncio.Lock()
        self._tasks: set[asyncio.Task[Any]] = set()
        self._start_balance = 0.0
        self.equity = 0.0
        self.free_margin = 0.0
        self.started_at = 0.0
        self.snapshots = 0
        self.decisions = 0
        self.signal_counts: dict[str, int] = {a.value: 0 for a in SignalAction}
        self.journal: list[dict[str, Any]] = []

    # ------------------------------------------------------------ building
    async def build_components(self) -> None:
        s = self.settings
        if s.universe.mode == "manual":
            infos = {}
            for sym in s.symbols:
                info = await self.client.call("symbol_info", sym)
                if info is not None:
                    infos[sym] = info
            from src.universe import build_universe

            self.instruments = build_universe(infos.values(), s)
        else:
            self.instruments = await discover_universe(self.client, s)
        if not self.instruments:
            raise RuntimeError("Universe is empty - check symbols / universe settings (python -m src.universe)")

        symbols = list(self.instruments)
        depth = [sym for sym, i in self.instruments.items() if s.use_depth and i.asset_class in s.universe.depth_asset_classes]
        self.ingestor = DataIngestor(
            self.client, symbols,
            poll_interval_s=s.poll_interval_ms / 1000.0,
            depth_symbols=depth,
            batch_size=s.tick_batch_size,
            warmup_seconds=s.warmup_seconds,
            error_hook=self.risk.record_error,
            success_hook=self.risk.record_success,
        )
        if s.nlp.enabled:
            model = build_sentiment_model(s.nlp.use_finbert, s.nlp.finbert_model)
            sources = [make_synthetic_source(symbols)] if s.nlp.synthetic_news else None
            self.nlp = NLPEngine(s.nlp, self.instruments, model, sources=sources)
        self.sessions = SessionCalendar(s.sessions, self.instruments)
        symbol_params = {sym: s.params_for(i.asset_class) for sym, i in self.instruments.items()}
        self.strategy = SignalEngine(
            s.strategy,
            sentiment_provider=self.nlp.sentiment if self.nlp else None,
            blackout_provider=self.nlp.blackout_reason if self.nlp else None,
            session_provider=self.sessions.entry_block_reason,
            symbol_params=symbol_params,
        )

    def asset_class(self, symbol: str) -> str:
        inst = self.instruments.get(symbol)
        return inst.asset_class if inst else OTHER

    # ------------------------------------------------------------ lifecycle
    def request_stop(self) -> None:
        if not self._stop.is_set():
            logger.warning("Stop requested")
            self._stop.set()

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self.request_stop)
            except (NotImplementedError, RuntimeError):
                pass  # Windows: rely on KeyboardInterrupt

    async def run(self, duration_s: Optional[float] = None) -> None:
        self._install_signal_handlers()
        self.started_at = time.time()
        mode = "SIMULATION" if self.settings.simulate else ("DRY-RUN (paper)" if self.settings.dry_run else "LIVE ORDERS")
        logger.info("Starting bot | mode={} universe={}", mode, self.settings.universe.mode)

        await self.client.connect(self.settings.mt5, require_demo=self.settings.require_demo)
        try:
            account = await self.client.call("account_info")
            self._start_balance = float(account.balance)
            self.equity = float(account.equity)
            self.free_margin = float(getattr(account, "margin_free", account.equity))
            self.risk.update_equity(self.equity)
            await self.build_components()
            assert self.ingestor is not None
            await self.ingestor.start()
            await self._refresh_positions()

            coros = [self.ingestor.run(self._stop), self._strategy_loop(), self._monitor_loop()]
            if self.nlp:
                coros.append(self.nlp.run(self._stop))
            if self.settings.api_enabled:
                coros.append(serve_api(create_app(self.state), self.settings.api_host, self.settings.api_port, self._stop))
            if duration_s:
                coros.append(self._stop_after(duration_s))

            tasks = [asyncio.create_task(c) for c in coros]
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
            for t in done:
                if not t.cancelled() and t.exception() is not None:
                    logger.opt(exception=t.exception()).critical("Fatal task error")
            self.request_stop()
            await asyncio.wait(tasks, timeout=5)
            for t in tasks:
                t.cancel()
        finally:
            await self.shutdown()

    async def _stop_after(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            logger.info("Run duration of {:.0f}s reached", seconds)
            self.request_stop()

    async def shutdown(self) -> None:
        self._stop.set()
        if self._tasks:
            await asyncio.wait(self._tasks, timeout=3)
        try:
            if self.ingestor:
                await self.ingestor.stop()
            if self.settings.flatten_on_exit and self.client.connected:
                for pos, res in await self.orders.close_all(comment="shutdown"):
                    self._record_close(pos, res, "shutdown")
        except Exception:
            logger.exception("error while flattening on shutdown")
        self._export_journal()
        await self.client.disconnect()
        logger.info("Bot stopped | {}", self._summary())

    # ------------------------------------------------------------ main loops
    async def _strategy_loop(self) -> None:
        assert self.ingestor is not None and self.strategy is not None
        q = self.ingestor.queue
        max_lat = self.settings.risk.max_latency_ms
        while not self._stop.is_set():
            try:
                snap = await asyncio.wait_for(q.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            try:
                self.snapshots += 1
                if self.portfolio is not None:
                    self.portfolio.on_price(snap.symbol, snap.time_msc / 1000.0, snap.mid)
                held = self._positions.get(snap.symbol)
                sig = self.strategy.on_snapshot(snap, held[0].side if held else None)
                if sig is None:
                    continue  # model update only (older tick of a batch / warm-up)
                self.decisions += 1
                pipeline_ms = (time.perf_counter_ns() - snap.recv_perf_ns) / 1e6
                self.risk.record_latency("pipeline", pipeline_ms)
                self.signal_counts[sig.action.value] += 1

                if sig.action is SignalAction.HOLD or snap.symbol in self._in_flight:
                    continue
                if sig.action is not SignalAction.CLOSE and pipeline_ms > max_lat:
                    logger.warning("Skipping stale entry signal on {} ({:.1f} ms old)", snap.symbol, pipeline_ms)
                    continue
                self._spawn(self._execute(sig, snap))
            except Exception as exc:
                logger.exception("strategy loop error")
                self.risk.record_error("strategy", exc)

    async def _monitor_loop(self) -> None:
        last_status = 0.0
        while not self._stop.is_set():
            try:
                account = await self.client.call("account_info")
                if account is None:
                    raise RuntimeError("account_info returned None")
                await self._refresh_positions()
                if self.settings.dry_run and not self.settings.simulate:
                    self.equity = self._start_balance + self.orders.paper_realized_pnl + self.orders.paper_floating_pnl()
                    self.free_margin = self.equity
                else:
                    self.equity = float(account.equity)
                    self.free_margin = float(getattr(account, "margin_free", account.equity))
                self.risk.update_equity(self.equity)
                self.risk.maybe_rearm()
                self.risk.record_success()
                self._session_flatten()
                if self.portfolio is not None:
                    self.portfolio_var = self.portfolio.portfolio_var(await self._exposures())

                if time.time() - last_status >= self.settings.status_log_s:
                    last_status = time.time()
                    self._log_status()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("monitor error: {}", exc)
                self.risk.record_error("monitor", exc)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.settings.equity_poll_s)
            except asyncio.TimeoutError:
                pass

    def _session_flatten(self) -> None:
        """Close US stock positions shortly before the closing bell."""
        if self.sessions is None:
            return
        now = time.time()
        for symbol, positions in self._positions.items():
            if positions and symbol not in self._in_flight and self.sessions.must_flatten(symbol, now):
                logger.warning("Session close approaching: flattening {}", symbol)
                self._spawn(self._close_symbol(symbol, "session_close"))

    # ------------------------------------------------------------ execution
    def _spawn(self, coro: Any) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _close_symbol(self, symbol: str, reason: str) -> None:
        self._in_flight.add(symbol)
        try:
            for pos in list(self._positions.get(symbol, [])):
                res = await self.orders.close_position(pos, comment=f"exit {reason}"[:31])
                self._record_close(pos, res, reason)
            await self._refresh_positions()
        except Exception as exc:
            logger.exception("close error on {}", symbol)
            self.risk.record_error("execution", exc)
        finally:
            self._in_flight.discard(symbol)

    async def _execute(self, sig: Signal, snap: MarketSnapshot) -> None:
        if sig.action is SignalAction.CLOSE:
            await self._close_symbol(sig.symbol, sig.reason)
            return
        self._in_flight.add(sig.symbol)
        try:
            await self._open(sig, snap)
            await self._refresh_positions()
        except Exception as exc:
            logger.exception("execution error on {}", sig.symbol)
            self.risk.record_error("execution", exc)
        finally:
            self._in_flight.discard(sig.symbol)

    def _class_exposure(self, asset_class: str) -> tuple[int, int]:
        count = net = 0
        for sym, positions in self._positions.items():
            if self.asset_class(sym) != asset_class:
                continue
            for p in positions:
                count += 1
                net += p.side.sign
        return count, net

    async def _exposures(self) -> dict[str, float]:
        """Signed account-currency notional per symbol of the bot's open positions."""
        out: dict[str, float] = {}
        for sym, positions in self._positions.items():
            if not positions:
                continue
            spec = await self.orders.symbol_spec(sym)
            last = self.ingestor.latest.get(sym) if self.ingestor else None
            for p in positions:
                price = last.mid if last else (p.price_current or p.price_open)
                out[sym] = out.get(sym, 0.0) + p.side.sign * spec.money_per_price_unit(p.volume) * price
        return out

    async def _portfolio_size(self, sig: Signal, snap: MarketSnapshot, spec: Any, side: Side, lots: float) -> float:
        """Volatility targeting + correlation-aware VaR cap on top of the Kelly size."""
        if self.portfolio is None or lots <= 0:
            return lots
        if self.portfolio.vol_spike(sig.symbol):
            logger.info("Entry {} {} blocked: volatility spike (short/long vol {:.2f})", side.value, sig.symbol,
                        self.portfolio.vol_ratio(sig.symbol) or 0)
            return 0.0
        mult = self.portfolio.vol_multiplier(sig.symbol)
        sized = spec.round_volume_down(lots * mult)
        if sized <= 0:
            return 0.0
        limit = self.equity * self.settings.portfolio.max_portfolio_var_pct
        value_per_lot = side.sign * spec.money_per_price_unit(1.0) * snap.mid
        cap = self.portfolio.max_lots_for_var(sig.symbol, value_per_lot, await self._exposures(), limit)
        if cap is not None and cap < sized:
            capped = spec.round_volume_down(cap)
            logger.info("{}: size {:.2f} -> {:.2f} lots (portfolio VaR limit {:.2f})", sig.symbol, sized, capped, limit)
            sized = capped
        if sized < lots:
            logger.debug("{}: Kelly {:.2f} -> {:.2f} lots after vol targeting (x{:.2f}) / VaR", sig.symbol, lots, sized, mult)
        return sized

    async def _open(self, sig: Signal, snap: MarketSnapshot) -> None:
        side = sig.side
        assert side is not None
        ac = self.asset_class(sig.symbol)
        open_class, net = self._class_exposure(ac)
        ok, reason = self.risk.pre_trade_check(
            sig.symbol, sig.vpin,
            open_symbol=len(self._positions.get(sig.symbol, [])),
            open_total=sum(len(v) for v in self._positions.values()),
            asset_class=ac, open_class=open_class, class_net_direction=net, side_sign=side.sign,
        )
        if not ok:
            logger.debug("Entry {} {} blocked by risk: {}", side.value, sig.symbol, reason)
            return
        spec = await self.orders.symbol_spec(sig.symbol)
        lots = self.risk.position_size(spec, self.equity, sig.stop_distance, model_kelly=sig.kelly)
        entry = snap.ask if side is Side.BUY else snap.bid
        lots = await self._portfolio_size(sig, snap, spec, side, lots)
        if lots > 0:
            margin = await self.orders.margin_required(sig.symbol, side, lots, entry)
            capped = self.risk.cap_by_margin(spec, lots, margin, self.free_margin)
            if capped < lots:
                logger.info("{}: size {:.2f} -> {:.2f} lots (margin {:.2f} vs free {:.2f})", sig.symbol, lots, capped, margin or 0, self.free_margin)
            lots = capped
        if lots <= 0:
            logger.debug("Entry {} {} skipped: size rounds to 0", side.value, sig.symbol)
            return
        sl = entry - side.sign * sig.stop_distance
        logger.info(
            "SIGNAL {} {} [{}] | z={:+.2f} alpha={:+.2f} vpin={:.3f} S={:+.2f} mu={:.5f} p={:.2f} b={:.2f} kelly={:.3f} lots={:.2f}",
            sig.action.value, sig.symbol, ac, sig.z or 0, sig.alpha or 0, sig.vpin or 0, sig.sentiment, sig.mu or 0,
            sig.win_prob or 0, sig.payoff_ratio or 0, sig.kelly or 0, lots,
        )
        res = await self.orders.market_order(sig.symbol, side, lots, sl=sl, tp=sig.take_profit, comment=f"OU z={sig.z:+.2f}")
        self.journal.append({
            "ts": time.time(), "event": "open", "symbol": sig.symbol, "class": ac, "side": side.value, "lots": lots,
            "price": res.price, "success": res.success, "retcode": res.retcode, "latency_ms": res.latency_ms,
            "z": sig.z, "vpin": sig.vpin, "sentiment": sig.sentiment, "reason": sig.reason, "pnl": None,
            "win_prob": sig.win_prob, "payoff_ratio": sig.payoff_ratio, "kelly": sig.kelly,
        })

    def _record_close(self, pos: PositionInfo, res: OrderResult, reason: str) -> None:
        if not res.success:
            logger.error("Failed to close #{} on {}: {}", pos.ticket, pos.symbol, res.comment)
            return
        self._bot_closed.add(pos.ticket)
        self.risk.record_trade_result(res.pnl)
        logger.info("CLOSED #{} {} {} pnl={:+.2f} ({})", pos.ticket, pos.side.value, pos.symbol, res.pnl, reason)
        self.journal.append({
            "ts": time.time(), "event": "close", "symbol": pos.symbol, "class": self.asset_class(pos.symbol),
            "side": pos.side.value, "lots": pos.volume, "price": res.price, "success": True, "retcode": res.retcode,
            "latency_ms": res.latency_ms, "z": None, "vpin": None, "sentiment": None, "reason": reason, "pnl": res.pnl,
        })

    async def _refresh_positions(self) -> None:
        async with self._pos_lock:
            positions = await self.orders.get_positions()
            current = {p.ticket for p in positions}
            for ticket, profit in list(self._last_profit.items()):
                if ticket not in current:
                    if ticket not in self._bot_closed:
                        self.risk.record_trade_result(profit)
                        logger.info("Position #{} closed by broker (SL/TP) pnl~{:+.2f}", ticket, profit)
                        self.journal.append({"ts": time.time(), "event": "close", "symbol": None, "class": None,
                                             "side": None, "lots": None, "price": None, "success": True,
                                             "retcode": None, "latency_ms": None, "z": None, "vpin": None,
                                             "sentiment": None, "reason": "sl_tp", "pnl": profit})
                    del self._last_profit[ticket]
                    self._bot_closed.discard(ticket)
            grouped: dict[str, list[PositionInfo]] = {}
            for p in positions:
                grouped.setdefault(p.symbol, []).append(p)
                self._last_profit[p.ticket] = p.profit
            self._positions = grouped

    async def _on_breaker_trip(self, reason: str) -> None:
        closed = await self.orders.emergency_flatten(reason)
        for pos, res in closed:
            self._record_close(pos, res, f"breaker:{reason}")
        await self._refresh_positions()

    # ------------------------------------------------------------ reporting
    def state(self) -> dict[str, Any]:
        symbols: dict[str, Any] = {}
        if self.strategy and self.ingestor:
            for s, inst in self.instruments.items():
                last = self.strategy.last_signal(s)
                snap = self.ingestor.latest.get(s)
                symbols[s] = {
                    "class": inst.asset_class,
                    "bid": snap.bid if snap else None,
                    "ask": snap.ask if snap else None,
                    "microprice": snap.microprice if snap else None,
                    "vpin": self.strategy.vpin(s),
                    "sentiment": self.nlp.sentiment(s) if self.nlp else 0.0,
                    "z": last.z if last else None,
                    "last_reason": last.reason if last else None,
                    "positions": len(self._positions.get(s, [])),
                }
        return {
            "uptime_s": round(time.time() - self.started_at, 1) if self.started_at else 0,
            "mode": "simulate" if self.settings.simulate else ("dry_run" if self.settings.dry_run else "live"),
            "universe": dict(Counter(i.asset_class for i in self.instruments.values())),
            "snapshots": self.snapshots,
            "decisions": self.decisions,
            "dropped": self.ingestor.dropped if self.ingestor else 0,
            "sweep_ms": round(self.ingestor.last_sweep_ms, 2) if self.ingestor else None,
            "signals": self.signal_counts,
            "risk": self.risk.snapshot(),
            "portfolio_var": self.portfolio_var,
            "portfolio_var_limit": self.equity * self.settings.portfolio.max_portfolio_var_pct,
            "vol_targeting": self.portfolio.snapshot(self.instruments) if self.portfolio else None,
            "positions": {s: [p.side.value for p in v] for s, v in self._positions.items() if v},
            "symbols": symbols,
        }

    def _log_status(self) -> None:
        if not self.strategy or not self.ingestor:
            return
        r = self.risk
        reasons: Counter[str] = Counter()
        movers: list[tuple[float, str]] = []
        for s in self.instruments:
            last = self.strategy.last_signal(s)
            if last is None:
                reasons["no_data"] += 1
                continue
            reasons[last.reason.split(":")[0]] += 1
            if last.z is not None:
                movers.append((abs(last.z), f"{s}{last.z:+.1f}"))
        top = " ".join(m for _, m in sorted(movers, reverse=True)[:5])
        open_pos = sum(len(v) for v in self._positions.values())
        logger.info(
            "STATUS eq={:.2f} dd={:.2%} VaR={:.3%} pVaR={} breaker={} open={} trades={} sweep={:.1f}ms | top|z|: {} | {}",
            self.equity, r.drawdown, r.current_var(),
            "-" if self.portfolio_var is None else f"{self.portfolio_var:.2f}", r.state.value, open_pos, r.stats.n,
            self.ingestor.last_sweep_ms, top or "-", dict(reasons.most_common(6)),
        )

    def _summary(self) -> str:
        r = self.risk
        return (
            f"symbols={len(self.instruments)} snapshots={self.snapshots} decisions={self.decisions} "
            f"signals={self.signal_counts} trades={r.stats.n} win_rate={r.stats.win_rate:.1%} "
            f"equity={self.equity:.2f} start={self._start_balance:.2f} breaker_trips={r.trip_count}"
        )

    def _export_journal(self) -> None:
        if not self.journal:
            return
        try:
            Path(self.settings.log_dir).mkdir(parents=True, exist_ok=True)
            path = Path(self.settings.log_dir) / f"trades_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            pd.DataFrame(self.journal).to_csv(path, index=False)
            logger.info("Trade journal written to {}", path)
        except Exception:
            logger.exception("failed to export trade journal")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exness MT5 HFT bot")
    p.add_argument("--config", help="JSON config file")
    p.add_argument("--simulate", action="store_true", help="use the in-process MT5 simulator")
    p.add_argument("--live", action="store_true", help="send orders to the account (disables dry-run)")
    p.add_argument("--i-understand-the-risk", action="store_true", dest="confirm_live", help="required together with --live")
    p.add_argument("--duration", type=float, default=None, help="stop after N seconds")
    p.add_argument("--manual", action="store_true", help="trade only Settings.symbols (no auto universe)")
    p.add_argument("--no-api", action="store_true")
    p.add_argument("--no-nlp", action="store_true")
    p.add_argument("--log-level", default=None)
    return p.parse_args(argv)


def build_settings(args: argparse.Namespace) -> Settings:
    settings = load_settings(args.config)
    if args.simulate:
        settings.simulate = True
        settings.dry_run = False  # the simulator itself is the paper venue
        settings.nlp.synthetic_news = True
        settings.nlp.calendar_url = None
        settings.sessions.enforce = False  # simulated stocks trade around the clock
        settings.warmup_seconds = 0
    if args.live:
        if not args.confirm_live:
            raise SystemExit("--live requires --i-understand-the-risk")
        settings.dry_run = False
    if args.manual:
        settings.universe.mode = "manual"
    if args.no_api:
        settings.api_enabled = False
    if args.no_nlp:
        settings.nlp.enabled = False
    if args.log_level:
        settings.log_level = args.log_level.upper()
    return settings


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    settings = build_settings(args)
    setup_logging(settings)
    bot = TradingBot(settings)
    try:
        asyncio.run(bot.run(args.duration))
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
    except Exception:
        logger.exception("Bot crashed")
        return 1
    finally:
        logger.complete()
    return 0


if __name__ == "__main__":
    sys.exit(main())
