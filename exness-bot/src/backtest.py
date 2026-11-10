"""Tick-level backtester with walk-forward validation.

Replays historical ticks through the SAME :class:`SignalEngine` and risk sizing the live
bot uses, with conservative execution:

* orders fill ``fill_delay_ticks`` after the signal (default 1) at that tick's bid/ask,
  plus ``slippage_ticks`` adverse slippage;
* SL/TP are checked on every tick against the side the broker watches (bid for longs,
  ask for shorts); a gap through the level fills at the gapped price;
* commission per lot per side is charged on entry and exit;
* the strategy's own exits (mean reverted, time stop) close at market.

Walk-forward: the data is cut into ``folds + 1`` consecutive blocks; parameters are
chosen on block k-1 and evaluated, untouched, on block k. Only the out-of-sample
results count.

CLI (run ``python -m src.backtest -h``):
    python -m src.backtest fetch --symbol EURUSDm --days 10           # on the Windows VPS
    python -m src.backtest run --data data/EURUSDm.csv                # single backtest
    python -m src.backtest walkforward --data data/EURUSDm.csv --folds 4
    python -m src.backtest run --synthetic ou                         # demo without data
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.config import PortfolioRiskSettings, RiskLimits, StrategyParams, load_settings
from src.data_ingestion import MarketSnapshot
from src.models import Side, SymbolSpec
from src.portfolio_risk import PortfolioRisk
from src.risk_controller import RiskController
from src.strategy import SignalAction, SignalEngine

TICK_COLUMNS = ["time_msc", "bid", "ask", "volume_real"]


# ------------------------------------------------------------------ data
@dataclass
class BacktestConfig:
    initial_equity: float = 10_000.0
    commission_per_lot_side: float = 0.0  # e.g. 3.5 USD on Exness Raw Spread
    slippage_ticks: int = 0
    fill_delay_ticks: int = 1
    asset_class: Optional[str] = None  # "us_stock" enables NYSE session rules
    apply_sessions: bool = True


@dataclass
class Trade:
    side: str
    open_ms: int
    close_ms: int
    entry: float
    exit: float
    lots: float
    pnl: float
    reason: str
    ticks_held: int
    win_prob: Optional[float] = None
    payoff_ratio: Optional[float] = None
    kelly: Optional[float] = None


@dataclass
class BacktestResult:
    trades: list[Trade]
    metrics: dict[str, Any]
    params: dict[str, Any] = field(default_factory=dict)


def load_ticks_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(TICK_COLUMNS[:3]) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    if "volume_real" not in df.columns:
        df["volume_real"] = 0.0
    df = df[TICK_COLUMNS].dropna()
    df = df[(df.bid > 0) & (df.ask >= df.bid)].sort_values("time_msc", kind="stable").reset_index(drop=True)
    return df


def load_spec(path: str | Path) -> SymbolSpec:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return SymbolSpec(**{k: data[k] for k in SymbolSpec.__dataclass_fields__ if k in data})


def synthetic_ticks(kind: str = "ou", n: int = 200_000, mid: float = 1.08500, spread: float = 0.00008,
                    sd: float = 0.00030, b: float = 0.97, dt_ms: int = 250, seed: int = 7) -> pd.DataFrame:
    """Synthetic tick series: 'ou' (mean-reverting) or 'rw' (random walk, no edge)."""
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n)
    x = np.empty(n)
    if kind == "rw":
        x = mid + np.cumsum(eps) * sd * math.sqrt(1 - b * b)
    elif kind == "ou":
        scale = sd * math.sqrt(1 - b * b)
        x[0] = mid
        for i in range(1, n):
            x[i] = mid + b * (x[i - 1] - mid) + scale * eps[i]
    else:
        raise ValueError("kind must be 'ou' or 'rw'")
    t0 = int(pd.Timestamp("2026-10-05 08:00", tz="UTC").timestamp() * 1000)
    return pd.DataFrame({
        "time_msc": t0 + np.arange(n, dtype=np.int64) * dt_ms,
        "bid": np.round(x - spread / 2, 5),
        "ask": np.round(x + spread / 2, 5),
        "volume_real": 0.0,
    })


def default_spec(symbol: str = "EURUSDm") -> SymbolSpec:
    return SymbolSpec(symbol, 5, 1e-5, 1e-5, 1.0, 100_000, 0.01, 200.0, 0.01, 0, 1)


# ------------------------------------------------------------- engine
def run_backtest(
    ticks: pd.DataFrame,
    spec: SymbolSpec,
    params: StrategyParams,
    limits: RiskLimits,
    cfg: BacktestConfig = BacktestConfig(),
    symbol: Optional[str] = None,
    portfolio: Optional[PortfolioRiskSettings] = PortfolioRiskSettings(),
) -> BacktestResult:
    symbol = symbol or spec.name
    session_provider = flatten_check = None
    if cfg.apply_sessions and cfg.asset_class == "us_stock":
        from src.config import SessionSettings
        from src.sessions import SessionCalendar
        from src.universe import Instrument

        cal = SessionCalendar(SessionSettings(), {symbol: Instrument(symbol, "us_stock")})
        session_provider, flatten_check = cal.entry_block_reason, cal.must_flatten

    engine = SignalEngine(params, session_provider=session_provider)
    risk = RiskController(limits, params.vpin_threshold)
    port = PortfolioRisk(portfolio) if portfolio is not None and portfolio.enabled else None
    blocked_vol_spike = 0
    hold_reasons: dict[str, int] = {}
    money_per_price = spec.tick_value / spec.tick_size
    slip = cfg.slippage_ticks * spec.tick_size

    t_ms = ticks["time_msc"].to_numpy(np.int64)
    bids = ticks["bid"].to_numpy(np.float64)
    asks = ticks["ask"].to_numpy(np.float64)
    vols = ticks["volume_real"].to_numpy(np.float64)

    equity = cfg.initial_equity
    trades: list[Trade] = []
    pos: Optional[dict[str, Any]] = None
    pending: Optional[dict[str, Any]] = None

    def close(i: int, price: float, reason: str) -> None:
        nonlocal pos, equity
        assert pos is not None
        side: Side = pos["side"]
        pnl = side.sign * (price - pos["entry"]) * money_per_price * pos["lots"]
        pnl -= 2 * cfg.commission_per_lot_side * pos["lots"]
        equity += pnl
        risk.record_trade_result(pnl)
        trades.append(Trade(side.value, pos["open_ms"], int(t_ms[i]), pos["entry"], price, pos["lots"], pnl,
                            reason, i - pos["open_i"], pos["win_prob"], pos["payoff"], pos["kelly"]))
        pos = None

    for i in range(len(t_ms)):
        bid, ask = bids[i], asks[i]

        # 1) execute the pending order on this tick
        if pending is not None:
            pending["delay"] -= 1
            if pending["delay"] <= 0:
                sig = pending["signal"]
                if pending["kind"] == "close" and pos is not None:
                    close(i, (bid - slip) if pos["side"] is Side.BUY else (ask + slip), sig.reason)
                elif pending["kind"] == "open" and pos is None:
                    side: Side = sig.side
                    entry = (ask + slip) if side is Side.BUY else (bid - slip)
                    tp = sig.take_profit
                    if tp is not None and side.sign * (tp - entry) > 0:  # still worth taking
                        pos = {"side": side, "entry": entry, "lots": pending["lots"], "open_ms": int(t_ms[i]),
                               "open_i": i, "sl": entry - side.sign * sig.stop_distance, "tp": tp,
                               "win_prob": sig.win_prob, "payoff": sig.payoff_ratio, "kelly": sig.kelly}
                pending = None

        # 2) broker-side SL / TP
        if pos is not None:
            if pos["side"] is Side.BUY:
                if bid <= pos["sl"]:
                    close(i, bid, "stop_loss")
                elif bid >= pos["tp"]:
                    close(i, bid, "take_profit")
            else:
                if ask >= pos["sl"]:
                    close(i, ask, "stop_loss")
                elif ask <= pos["tp"]:
                    close(i, ask, "take_profit")
        if pos is not None and flatten_check is not None and flatten_check(symbol, t_ms[i] / 1000.0):
            close(i, bid if pos["side"] is Side.BUY else ask, "session_close")

        # 3) strategy
        mid = 0.5 * (bid + ask)
        if port is not None:
            port.on_price(symbol, t_ms[i] / 1000.0, mid)
        snap = MarketSnapshot(
            symbol=symbol, time_msc=int(t_ms[i]), recv_ns=int(t_ms[i]) * 1_000_000, recv_perf_ns=0,
            bid=bid, ask=ask, bid_volume=0.0, ask_volume=0.0, microprice=mid,
            trade_volume=vols[i] if vols[i] > 0 else 1.0, has_depth=False,
        )
        sig = engine.on_snapshot(snap, pos["side"] if pos else None)
        if sig is not None and sig.action is SignalAction.HOLD and pos is None:
            key = sig.reason.split(":")[0]
            hold_reasons[key] = hold_reasons.get(key, 0) + 1
        if sig is None or pending is not None:
            continue
        if sig.action is SignalAction.CLOSE and pos is not None:
            pending = {"kind": "close", "signal": sig, "delay": cfg.fill_delay_ticks}
        elif sig.action in (SignalAction.OPEN_LONG, SignalAction.OPEN_SHORT) and pos is None:
            lots = risk.position_size(spec, equity, sig.stop_distance, model_kelly=sig.kelly)
            if port is not None and lots > 0:
                if port.vol_spike(symbol):
                    blocked_vol_spike += 1
                    lots = 0.0
                else:
                    lots = spec.round_volume_down(lots * port.vol_multiplier(symbol))
                    assert portfolio is not None
                    cap = port.max_lots_for_var(symbol, sig.side.sign * money_per_price * mid, {},
                                                equity * portfolio.max_portfolio_var_pct)
                    if cap is not None and cap < lots:
                        lots = spec.round_volume_down(cap)
            if lots > 0:
                pending = {"kind": "open", "signal": sig, "lots": lots, "delay": cfg.fill_delay_ticks}
        if pending is not None and pending["delay"] <= 0:
            pending["delay"] = 1

    if pos is not None:
        last = len(t_ms) - 1
        close(last, bids[last] if pos["side"] is Side.BUY else asks[last], "end_of_data")

    metrics = compute_metrics(trades, cfg.initial_equity, t_ms)
    metrics["entries_blocked_vol_spike"] = blocked_vol_spike
    total_holds = sum(hold_reasons.values()) or 1
    metrics["why_no_entry_pct"] = {k: round(100 * v / total_holds, 1)
                                   for k, v in sorted(hold_reasons.items(), key=lambda kv: -kv[1])}
    return BacktestResult(trades, metrics, params=params.model_dump())


def compute_metrics(trades: list[Trade], initial_equity: float, t_ms: Optional[np.ndarray] = None) -> dict[str, Any]:
    n = len(trades)
    out: dict[str, Any] = {"trades": n}
    if t_ms is not None and len(t_ms):
        out["ticks"] = int(len(t_ms))
        out["days"] = round((int(t_ms[-1]) - int(t_ms[0])) / 86_400_000, 2)
    if n == 0:
        out.update(net_pnl=0.0, return_pct=0.0)
        return out
    pnl = np.array([t.pnl for t in trades])
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    curve = initial_equity + np.cumsum(pnl)
    peaks = np.maximum.accumulate(np.concatenate(([initial_equity], curve)))[1:]
    dd = float(((peaks - curve) / peaks).max())
    std = float(pnl.std(ddof=1)) if n > 1 else 0.0
    probs = [t.win_prob for t in trades if t.win_prob is not None]
    reasons: dict[str, int] = {}
    for t in trades:
        reasons[t.reason] = reasons.get(t.reason, 0) + 1
    out.update(
        net_pnl=round(float(pnl.sum()), 2),
        return_pct=round(100 * float(pnl.sum()) / initial_equity, 3),
        win_rate=round(len(wins) / n, 4),
        avg_win=round(float(wins.mean()), 2) if len(wins) else 0.0,
        avg_loss=round(float(losses.mean()), 2) if len(losses) else 0.0,
        payoff_ratio=round(float(wins.mean() / -losses.mean()), 3) if len(wins) and len(losses) else None,
        profit_factor=round(float(wins.sum() / -losses.sum()), 3) if len(losses) else None,
        expectancy=round(float(pnl.mean()), 3),
        t_stat=round(float(pnl.mean() / std * math.sqrt(n)), 2) if std > 0 else None,
        max_drawdown_pct=round(100 * dd, 3),
        avg_ticks_held=round(float(np.mean([t.ticks_held for t in trades])), 1),
        predicted_win_prob=round(float(np.mean(probs)), 4) if probs else None,
        exit_reasons=reasons,
    )
    return out


# --------------------------------------------------------- walk-forward
DEFAULT_GRID: dict[str, list[Any]] = {"z_entry": [2.0, 2.5, 3.0], "ou_window": [200, 300, 500]}


def _with(params: StrategyParams, combo: dict[str, Any]) -> StrategyParams:
    data = {**params.model_dump(), **combo}
    if "ou_window" in combo:
        data["ou_min_obs"] = min(data["ou_min_obs"], int(combo["ou_window"]))
    return StrategyParams.model_validate(data)


def walk_forward(
    ticks: pd.DataFrame,
    spec: SymbolSpec,
    params: StrategyParams,
    limits: RiskLimits,
    cfg: BacktestConfig = BacktestConfig(),
    folds: int = 4,
    grid: Optional[dict[str, list[Any]]] = None,
    min_train_trades: int = 10,
    portfolio: Optional[PortfolioRiskSettings] = PortfolioRiskSettings(),
) -> dict[str, Any]:
    grid = grid or DEFAULT_GRID
    blocks = np.array_split(np.arange(len(ticks)), folds + 1)
    combos = [dict(zip(grid, values)) for values in itertools.product(*grid.values())]
    fold_reports: list[dict[str, Any]] = []
    oos_trades: list[Trade] = []

    for k in range(1, folds + 1):
        train = ticks.iloc[blocks[k - 1]].reset_index(drop=True)
        test = ticks.iloc[blocks[k]].reset_index(drop=True)
        best: Optional[tuple[float, dict[str, Any], dict[str, Any]]] = None
        most_active: Optional[tuple[dict[str, Any], dict[str, Any]]] = None
        for combo in combos:
            res = run_backtest(train, spec, _with(params, combo), limits, cfg, portfolio=portfolio)
            m = res.metrics
            if most_active is None or m["trades"] > most_active[1]["trades"]:
                most_active = (combo, m)
            if m["trades"] < min_train_trades:
                continue
            score = m["net_pnl"]
            if best is None or score > best[0]:
                best = (score, combo, m)
        if best is None or best[0] <= 0:
            diag: dict[str, Any] = {"fold": k, "chosen": None, "note": "no profitable parameter set in training -> no trading"}
            if best is not None:
                diag["best_in_training"] = {"params": best[1], **_brief(best[2])}
            if most_active is not None:
                diag["most_active_in_training"] = {
                    "params": most_active[0], **_brief(most_active[1]),
                    "why_no_entry_pct": most_active[1].get("why_no_entry_pct"),
                }
            fold_reports.append(diag)
            logger.info("fold {}: no profitable parameters in training block; sitting out", k)
            continue
        _, combo, train_m = best
        test_res = run_backtest(test, spec, _with(params, combo), limits, cfg, portfolio=portfolio)
        oos_trades.extend(test_res.trades)
        fold_reports.append({"fold": k, "chosen": combo, "train": _brief(train_m), "test": _brief(test_res.metrics)})
        logger.info("fold {}: chose {} | train pnl={} | TEST pnl={} trades={}", k, combo, train_m["net_pnl"],
                    test_res.metrics["net_pnl"], test_res.metrics["trades"])

    return {"folds": fold_reports, "out_of_sample": compute_metrics(oos_trades, cfg.initial_equity)}


def _brief(m: dict[str, Any]) -> dict[str, Any]:
    keys = ("trades", "net_pnl", "win_rate", "profit_factor", "t_stat", "max_drawdown_pct")
    return {k: m.get(k) for k in keys}


# ----------------------------------------------------------- MT5 fetch
async def fetch_mt5_ticks(symbol: str, days: float, out_dir: Path, chunk_hours: float = 6.0, config: Optional[str] = None) -> tuple[Path, Path]:
    """Download historical ticks + contract spec from the MT5 terminal (Windows)."""
    from src.mt5_client import AsyncMT5Client, MT5Error, load_backend

    settings = load_settings(config)
    client = AsyncMT5Client(load_backend(False))
    await client.connect(settings.mt5)
    try:
        if not await client.call("symbol_select", symbol, True):
            raise MT5Error(f"symbol {symbol} not found")
        info = await client.call("symbol_info", symbol)
        tick = await client.call("symbol_info_tick", symbol)
        end = int(tick.time_msc // 1000) + 60
        start = int(end - days * 86_400)
        frames = []
        step = int(chunk_hours * 3600)
        flags = client.const("COPY_TICKS_INFO")
        for t in range(start, end, step):
            arr = await client.call("copy_ticks_range", symbol, t, min(t + step, end), flags)
            if arr is not None and len(arr):
                frames.append(pd.DataFrame(arr)[TICK_COLUMNS])
            logger.info("{}: {} -> {} ticks so far", symbol, pd.Timestamp(t, unit="s"), sum(len(f) for f in frames))
        if not frames:
            code, msg = await client.last_error()
            raise MT5Error(f"no ticks returned ({msg}); open the symbol chart once so the terminal downloads history", code)
        df = pd.concat(frames).drop_duplicates().sort_values("time_msc", kind="stable")
        out_dir.mkdir(parents=True, exist_ok=True)
        data_path = out_dir / f"{symbol}.csv"
        spec_path = out_dir / f"{symbol}.spec.json"
        df.to_csv(data_path, index=False)
        spec = SymbolSpec.from_info(info)
        meta = {**asdict(spec), "path": getattr(info, "path", ""), "description": getattr(info, "description", "")}
        spec_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.success("Saved {} ticks to {} (+ {})", len(df), data_path, spec_path)
        return data_path, spec_path
    finally:
        await client.disconnect()


TIMEFRAME_SECONDS = {"M1": 60, "H1": 3600, "D1": 86_400}


async def _tick_spread_median(client: Any, symbol: str, end: int, days: float = 7.0,
                              attempts: int = 4, wait_s: float = 3.0) -> float:
    """Median ask-bid over the last ``days`` of real ticks (0.0 if the terminal returns none).

    MT5 downloads tick history from the server lazily: the first request often returns nothing
    while the download runs, so we retry a few times with a pause.
    """
    flags = client.const("COPY_TICKS_ALL")
    for attempt in range(attempts):
        try:
            arr = await client.call("copy_ticks_range", symbol, int(end - days * 86_400), int(end), flags)
            if arr is not None and len(arr):
                df = pd.DataFrame(arr)
                df = df[(df["bid"] > 0) & (df["ask"] > 0)]
                if len(df):
                    return float((df["ask"] - df["bid"]).median())
        except Exception as exc:
            logger.debug("{}: tick request failed: {}", symbol, exc)
        if attempt < attempts - 1:
            await asyncio.sleep(wait_s)
    logger.warning("{}: no ticks returned - spread will come from the bar spread field only", symbol)
    return 0.0


async def _download_rates(client: Any, symbol: str, tf: int, start: int, end: int, step: int,
                          max_bars: int) -> pd.DataFrame:
    """One pass: ranged chunks, then a count-based request; keeps whichever returned more."""
    frames = []
    for t in range(start, end, step):
        try:
            arr = await client.call("copy_rates_range", symbol, tf, t, min(t + step, end))
        except Exception as exc:
            logger.debug("{}: copy_rates_range failed: {}", symbol, exc)
            arr = None
        if arr is not None and len(arr):
            frames.append(pd.DataFrame(arr)[["time", "open", "close", "spread"]])
    ranged = pd.concat(frames) if frames else pd.DataFrame(columns=["time", "open", "close", "spread"])
    try:
        arr = await client.call("copy_rates_from_pos", symbol, tf, 0, int(max_bars))
        counted = pd.DataFrame(arr)[["time", "open", "close", "spread"]] if arr is not None and len(arr) else None
    except Exception as exc:
        logger.debug("{}: copy_rates_from_pos failed: {}", symbol, exc)
        counted = None
    if counted is not None:
        counted = counted[counted["time"] >= start]
        if len(counted) > len(ranged):
            ranged = counted
    return ranged.drop_duplicates("time").sort_values("time").reset_index(drop=True)


async def fetch_mt5_bars(symbol: str, days: float, out_dir: Path, config: Optional[str] = None,
                         chunk_days: Optional[float] = None, timeframe: str = "M1",
                         client: Any = None, attempts: int = 6, wait_s: float = 5.0,
                         min_bars: int = 2) -> tuple[Path, Path]:
    """Download bars (bid OHLC + spread in points) and the symbol spec incl. swap rates.

    Writes ``{symbol}_{timeframe}.csv`` and ``{symbol}.spec.json``; the spec also stores the
    median spread measured from the last days of real ticks (``tick_spread_median``).
    MT5 fetches history from the server in the background, so the request is repeated (with a
    pause) until the bar count stops growing. Raises if fewer than ``min_bars`` bars arrive.
    Pass an already-connected ``client`` to download many symbols over one connection.
    """
    from src.mt5_client import AsyncMT5Client, MT5Error, load_backend

    timeframe = timeframe.upper()
    if timeframe not in TIMEFRAME_SECONDS:
        raise ValueError(f"timeframe must be one of {sorted(TIMEFRAME_SECONDS)}")
    own = client is None
    if own:
        settings = load_settings(config)
        client = AsyncMT5Client(load_backend(False))
        await client.connect(settings.mt5)
    try:
        if not await client.call("symbol_select", symbol, True):
            raise MT5Error(f"symbol {symbol} not found")
        info = await client.call("symbol_info", symbol)
        tick = await client.call("symbol_info_tick", symbol)
        width = TIMEFRAME_SECONDS[timeframe]
        end = int(tick.time_msc // 1000) + width
        start = int(end - days * 86_400)
        chunk_days = chunk_days or {"M1": 15.0, "H1": 120.0, "D1": 365.0}[timeframe]
        step = int(chunk_days * 86_400)
        tf = client.const(f"TIMEFRAME_{timeframe}")
        max_bars = int(days * 86_400 / width) + 10

        df = pd.DataFrame(columns=["time", "open", "close", "spread"])
        for attempt in range(attempts):
            got = await _download_rates(client, symbol, tf, start, end, step, max_bars)
            grew = len(got) > len(df)
            if grew:
                df = got
            if attempt > 0 and not grew and len(df) >= min_bars:
                break  # count is stable: the terminal has everything the server offers
            if attempt < attempts - 1:
                logger.info("{}: {} bars so far, waiting for MT5 to load history...", symbol, len(df))
                await asyncio.sleep(wait_s)
        if len(df) < min_bars:
            code, msg = await client.last_error()
            raise MT5Error(f"only {len(df)} bars returned after {attempts} tries ({msg}) - open the symbol's "
                           f"{timeframe} chart in MT5, scroll back, then fetch again", code)
        got_days = (df.time.iloc[-1] - df.time.iloc[0]) / 86_400
        if timeframe == "M1" and got_days < 0.8 * days:
            logger.warning("Only {:.0f} of {:.0f} days returned. In MT5 set Tools > Options > Charts > "
                           "'Max bars in chart' to Unlimited, restart MT5 and fetch again.", got_days, days)
        out_dir.mkdir(parents=True, exist_ok=True)
        data_path = out_dir / f"{symbol}_{timeframe}.csv"
        spec_path = out_dir / f"{symbol}.spec.json"
        df.to_csv(data_path, index=False)
        spec = replace(SymbolSpec.from_info(info), tick_spread_median=await _tick_spread_median(client, symbol, end))
        spec_path.write_text(json.dumps({**asdict(spec), "path": getattr(info, "path", ""),
                                         "description": getattr(info, "description", "")}, indent=2), encoding="utf-8")
        logger.success("{}: {} {} bars ({:.0f} days), tick spread median {:.6g}, swap long/short {}/{} (mode {})",
                       symbol, len(df), timeframe, got_days, spec.tick_spread_median, spec.swap_long,
                       spec.swap_short, spec.swap_mode)
        return data_path, spec_path
    finally:
        if own:
            await client.disconnect()


# ------------------------------------------------------------------ CLI
def _print(title: str, payload: dict[str, Any]) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(payload, indent=2, default=str, ensure_ascii=False))


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Tick backtester / walk-forward")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download ticks from MT5 (Windows)")
    f.add_argument("--symbol", required=True)
    f.add_argument("--days", type=float, default=10)
    f.add_argument("--out", default="data")
    f.add_argument("--config", help="bot JSON config (default: config.json if present)")

    fb = sub.add_parser("fetch-bars", help="download M1/H1/D1 bars + spec (incl. swaps) from MT5")
    fb.add_argument("--symbol", required=True)
    fb.add_argument("--timeframe", default="M1", choices=["M1", "H1", "D1"])
    fb.add_argument("--days", type=float, default=180)
    fb.add_argument("--out", default="data")
    fb.add_argument("--config", help="bot JSON config (default: config.json if present)")

    for name in ("run", "walkforward"):
        s = sub.add_parser(name)
        s.add_argument("--data", help="tick CSV (time_msc,bid,ask[,volume_real])")
        s.add_argument("--spec", help="contract spec JSON (default: <data>.spec.json or EURUSD-like)")
        s.add_argument("--synthetic", choices=["ou", "rw"], help="use synthetic ticks instead of --data")
        s.add_argument("--ticks", type=int, default=200_000, help="synthetic tick count")
        s.add_argument("--config", help="bot JSON config (strategy + risk sections)")
        s.add_argument("--asset-class", choices=["forex", "metal", "us_stock"], default=None)
        s.add_argument("--commission", type=float, default=0.0, help="per lot per side, account currency")
        s.add_argument("--slippage-ticks", type=int, default=0)
        s.add_argument("--delay-ticks", type=int, default=1)
        s.add_argument("--equity", type=float, default=10_000.0)
        s.add_argument("--trades-out", help="write the trade list to CSV")
        if name == "walkforward":
            s.add_argument("--folds", type=int, default=4)

    args = p.parse_args(argv)
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<level>{message}</level>")

    if args.cmd == "fetch":
        asyncio.run(fetch_mt5_ticks(args.symbol, args.days, Path(args.out), config=args.config))
        return 0
    if args.cmd == "fetch-bars":
        asyncio.run(fetch_mt5_bars(args.symbol, args.days, Path(args.out), config=args.config,
                                   timeframe=args.timeframe))
        return 0

    if args.synthetic:
        ticks = synthetic_ticks(args.synthetic, n=args.ticks)
        spec = default_spec()
    elif args.data:
        ticks = load_ticks_csv(args.data)
        spec_path = Path(args.spec) if args.spec else Path(args.data).with_suffix(".spec.json")
        spec = load_spec(spec_path) if spec_path.is_file() else default_spec(Path(args.data).stem)
        if not spec_path.is_file():
            logger.warning("No spec file found; assuming EURUSD-like contract (tick 0.00001 = 1 USD/lot)")
    else:
        p.error("--data or --synthetic is required")

    settings = load_settings(args.config)
    asset_class = args.asset_class
    params = settings.params_for(asset_class) if asset_class else settings.strategy
    cfg = BacktestConfig(args.equity, args.commission, args.slippage_ticks, args.delay_ticks, asset_class)
    logger.info("{} ticks over {:.1f} days, spread median {:.6f}", len(ticks),
                (ticks.time_msc.iloc[-1] - ticks.time_msc.iloc[0]) / 86_400_000, float((ticks.ask - ticks.bid).median()))

    t0 = time.perf_counter()
    if args.cmd == "run":
        res = run_backtest(ticks, spec, params, settings.risk, cfg, portfolio=settings.portfolio)
        _print("BACKTEST (in-sample, no optimisation)", res.metrics)
        if args.trades_out:
            pd.DataFrame([asdict(t) for t in res.trades]).to_csv(args.trades_out, index=False)
    else:
        rep = walk_forward(ticks, spec, params, settings.risk, cfg, folds=args.folds, portfolio=settings.portfolio)
        _print("WALK-FORWARD folds", {"folds": rep["folds"]})
        _print("OUT-OF-SAMPLE (the number that matters)", rep["out_of_sample"])
    logger.info("done in {:.1f}s", time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
