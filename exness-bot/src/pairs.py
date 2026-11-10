"""Statistical-arbitrage pairs trading (Engle-Granger cointegration + OU spread).

Idea
----
Two instruments tied by an economic relationship (gold/silver, EURUSD/GBPUSD, KO/PEP)
may drift individually, but a combination of them can be stationary:

    s_t = log Y_t - beta * log X_t - alpha          (hedge ratio beta by OLS)

If the residual s_t is stationary (Engle-Granger: ADF on the residuals rejects a unit
root, 5% critical value about -3.34 for two series), it is modelled as an OU process
and traded with z-scores:

    z = (s - mu) / sigma_eq
    z >= +z_entry  -> short the spread: sell Y, buy beta * notional of X
    z <= -z_entry  -> long the spread:  buy Y, sell beta * notional of X
    exit at |z| <= z_exit, stop at |z| >= z_stop, time stop after k half-lives

Lessons from the single-instrument test are built in:
  * prices are sampled into bars (default 60 s), not raw ticks;
  * the hedge ratio and the cointegration test are refreshed only every
    ``refit_bars`` bars (far fewer tests -> far fewer false positives);
  * both legs pay their spread + slippage; signals execute at the NEXT bar's
    opening quotes; expected value after costs must be positive.

CLI:
    python -m src.pairs run --y data/XAUUSDm.csv --x data/XAGUSDm.csv
    python -m src.pairs walkforward --y data/XAUUSDm.csv --x data/XAGUSDm.csv --folds 4
    python -m src.pairs run --synthetic coint        # sanity check without data
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.backtest import Trade, compute_metrics, default_spec, load_spec, load_ticks_csv
from src.config import RiskLimits, load_settings
from src.models import SymbolSpec
from src.risk_controller import RiskController
from src.strategy import cached_hit_probability

# MacKinnon (2010) critical values, Engle-Granger test, 2 variables, constant
EG_CRITICAL = {0.01: -3.90, 0.05: -3.34, 0.10: -3.04}


# ------------------------------------------------------------------ params
@dataclass(frozen=True)
class PairParams:
    bar_seconds: int = 60
    window: int = 480  # bars used to estimate beta / OU (8 h of 1-min bars)
    refit_bars: int = 30  # re-estimate + re-test every 30 bars
    eg_critical: float = EG_CRITICAL[0.05]
    z_entry: float = 2.0
    z_exit: float = 0.5
    z_stop: float = 4.0
    min_half_life: float = 2.0  # bars
    max_half_life: float = 120.0  # bars
    time_stop_half_lives: float = 3.0
    cost_mult: float = 1.5  # |s - mu| must exceed cost_mult * round-trip cost
    slippage_ticks: int = 1
    commission_per_lot_side: float = 0.0
    win_prob_haircut: float = 0.02
    max_win_prob: float = 0.8
    min_sigma_to_cost: float = 1.0  # the spread's normal swing must exceed the round-trip cost
    max_leg_notional_mult: float = 3.0  # each leg's notional <= this x equity (caps blow-ups)
    swap_mult: float = 1.0  # 1 = broker swap rates as published, 0 = ignore financing
    initial_equity: float = 10_000.0


@dataclass(frozen=True)
class PairFit:
    alpha: float
    beta: float
    mu: float
    sigma_eq: float
    b: float  # AR(1) coefficient of the spread
    half_life: float  # bars
    eg_t: float
    cointegrated: bool


# --------------------------------------------------------------- bar data
def build_pair_bars(ticks_y: pd.DataFrame, ticks_x: pd.DataFrame, bar_seconds: int = 60) -> pd.DataFrame:
    """Align two tick series into common bars (only bars where BOTH legs traded).

    Columns: bar, t_ms, and for each leg L in (y, x): L_bid_o, L_ask_o (first quote of the
    bar, used for execution) and L_bid, L_ask (last quote, used for signals).
    """
    width = int(bar_seconds * 1000)

    def agg(df: pd.DataFrame, leg: str) -> pd.DataFrame:
        d = df[["time_msc", "bid", "ask"]].copy()
        d["bar"] = d["time_msc"] // width
        g = d.groupby("bar", sort=True)
        out = pd.DataFrame({
            f"{leg}_bid_o": g["bid"].first(), f"{leg}_ask_o": g["ask"].first(),
            f"{leg}_bid": g["bid"].last(), f"{leg}_ask": g["ask"].last(),
        })
        return out

    bars = agg(ticks_y, "y").join(agg(ticks_x, "x"), how="inner").reset_index()
    bars["t_ms"] = (bars["bar"] + 1) * width
    return bars


def infer_bar_seconds(rates: pd.DataFrame) -> int:
    """Bar width from the most common time step (60 for M1, 3600 for H1, 86400 for D1)."""
    d = np.diff(rates["time"].to_numpy(np.int64))
    d = d[d > 0]
    if not len(d):
        return 60
    vals, counts = np.unique(d, return_counts=True)
    return int(vals[np.argmax(counts)])


def rates_to_leg_bars(rates: pd.DataFrame, spec: SymbolSpec, leg: str, spread_floor: float = 0.0,
                      bar_seconds: int = 60) -> pd.DataFrame:
    """MT5 rates (bid open/close + spread in points) -> one leg's bar columns.

    ask = bid + spread. The rate's spread field can understate real costs, so the spread
    used is max(rate spread, ``spread_floor``) (floor = median spread measured from ticks).
    """
    spread = np.maximum(rates["spread"].to_numpy(float) * spec.point, spread_floor)
    bar = (rates["time"].to_numpy(np.int64) // bar_seconds).astype(np.int64)
    return pd.DataFrame({
        "bar": bar,
        f"{leg}_bid_o": rates["open"].to_numpy(float), f"{leg}_ask_o": rates["open"].to_numpy(float) + spread,
        f"{leg}_bid": rates["close"].to_numpy(float), f"{leg}_ask": rates["close"].to_numpy(float) + spread,
    }).set_index("bar")


def build_pair_bars_from_rates(rates_y: pd.DataFrame, spec_y: SymbolSpec, floor_y: float,
                               rates_x: pd.DataFrame, spec_x: SymbolSpec, floor_x: float,
                               bar_seconds: Optional[int] = None) -> pd.DataFrame:
    """Inner-join two legs on bar index; ``t_ms`` is the bar's close time (decisions use closes)."""
    width = bar_seconds or min(infer_bar_seconds(rates_y), infer_bar_seconds(rates_x))
    bars = rates_to_leg_bars(rates_y, spec_y, "y", floor_y, width).join(
        rates_to_leg_bars(rates_x, spec_x, "x", floor_x, width), how="inner").reset_index()
    bars["t_ms"] = (bars["bar"] + 1) * width * 1000
    return bars


# ------------------------------------------------------------------- model
def fit_pair(log_y: np.ndarray, log_x: np.ndarray, eg_critical: float) -> Optional[PairFit]:
    """OLS hedge ratio, Engle-Granger ADF on residuals, AR(1)/OU on the spread."""
    n = len(log_y)
    if n < 30:
        return None
    xm, ym = log_x.mean(), log_y.mean()
    dx = log_x - xm
    sxx = float(dx @ dx)
    if sxx <= 0:
        return None
    beta = float(dx @ (log_y - ym)) / sxx
    alpha = ym - beta * xm
    e = log_y - alpha - beta * log_x

    # Engle-Granger: Delta e_t = gamma * e_{t-1} + u_t   (no constant; residuals are mean-zero)
    e0, de = e[:-1], np.diff(e)
    s00 = float(e0 @ e0)
    if s00 <= 0:
        return None
    gamma = float(e0 @ de) / s00
    u = de - gamma * e0
    sig_u = math.sqrt(float(u @ u) / max(len(u) - 1, 1))
    eg_t = gamma / (sig_u / math.sqrt(s00)) if sig_u > 0 else -math.inf

    # OU on the spread: e_{t+1} = c + b e_t + eps
    x, y = e[:-1], e[1:]
    mx, my = x.mean(), y.mean()
    vx = float((x - mx) @ (x - mx))
    b = float((x - mx) @ (y - my)) / vx if vx > 0 else 1.0
    if not 0.0 < b < 1.0:
        return PairFit(alpha, beta, 0.0, 0.0, b, math.inf, eg_t, False)
    c = my - b * mx
    resid = y - (c + b * x)
    sig_eps = math.sqrt(float(resid @ resid) / max(len(resid) - 2, 1))
    sigma_eq = sig_eps / math.sqrt(1.0 - b * b)
    half_life = math.log(2.0) / -math.log(b)
    return PairFit(alpha, beta, c / (1.0 - b), sigma_eq, b, half_life, eg_t, eg_t < eg_critical)


# ---------------------------------------------------------------- backtest
@dataclass
class PairsResult:
    trades: list[Trade]
    metrics: dict[str, Any]
    params: dict[str, Any]


def _value_per_lot(spec: SymbolSpec, price: float) -> float:
    """Account-currency notional of 1 lot at ``price``."""
    return spec.tick_value / spec.tick_size * price


def run_pairs_backtest(
    bars: pd.DataFrame,
    spec_y: SymbolSpec,
    spec_x: SymbolSpec,
    p: PairParams = PairParams(),
    limits: RiskLimits = RiskLimits(),
) -> PairsResult:
    n = len(bars)
    yb, ya = bars["y_bid"].to_numpy(float), bars["y_ask"].to_numpy(float)
    xb, xa = bars["x_bid"].to_numpy(float), bars["x_ask"].to_numpy(float)
    ybo, yao = bars["y_bid_o"].to_numpy(float), bars["y_ask_o"].to_numpy(float)
    xbo, xao = bars["x_bid_o"].to_numpy(float), bars["x_ask_o"].to_numpy(float)
    t_ms = bars["t_ms"].to_numpy(np.int64)
    log_y = np.log(0.5 * (yb + ya))
    log_x = np.log(0.5 * (xb + xa))
    slip_y = p.slippage_ticks * spec_y.tick_size
    slip_x = p.slippage_ticks * spec_x.tick_size

    risk = RiskController(limits, vpin_threshold=1.0)
    equity = p.initial_equity
    trades: list[Trade] = []
    reasons: dict[str, int] = {}
    fit: Optional[PairFit] = None
    refits = coint_passes = 0
    pos: Optional[dict[str, Any]] = None
    pending: Optional[dict[str, Any]] = None

    def note(reason: str) -> None:
        reasons[reason] = reasons.get(reason, 0) + 1

    def leg_pnl(side: int, entry: float, exit_: float, lots: float, spec: SymbolSpec) -> float:
        return side * (exit_ - entry) * spec.tick_value / spec.tick_size * lots

    # server-time day index of each bar's start (t_ms is the bar close); swaps roll at day change
    day = (t_ms - 1) // 86_400_000
    swap_total = 0.0

    def swap_for(pos_: dict[str, Any], d_from: int, d_to: int, i_: int) -> float:
        if p.swap_mult == 0.0 or d_to <= d_from:
            return 0.0
        d_ = pos_["dir"]
        ny, nx = spec_y.rollover_nights(d_from, d_to), spec_x.rollover_nights(d_from, d_to)
        mid_y_, mid_x_ = 0.5 * (ybo[i_] + yao[i_]), 0.5 * (xbo[i_] + xao[i_])
        return p.swap_mult * (ny * spec_y.swap_per_night(d_ > 0, pos_["lots_y"], mid_y_)
                              + nx * spec_x.swap_per_night(d_ < 0, pos_["lots_x"], mid_x_))

    for i in range(n):
        # 0) overnight financing for positions carried into this bar's day
        if pos is not None and i > pos["open_i"] and day[i] > day[i - 1]:
            sw = swap_for(pos, int(day[i - 1]), int(day[i]), i)
            pos["swap"] += sw
            swap_total += sw
        # 1) execute pending order at this bar's opening quotes
        if pending is not None:
            if pending["kind"] == "open" and pos is None:
                d = pending["dir"]  # +1 long spread (buy Y, sell X), -1 short spread
                ey = (yao[i] + slip_y) if d > 0 else (ybo[i] - slip_y)
                ex = (xbo[i] - slip_x) if d > 0 else (xao[i] + slip_x)
                pos = {**pending, "ey": ey, "ex": ex, "open_i": i, "open_ms": int(t_ms[i]), "swap": 0.0}
            elif pending["kind"] == "close" and pos is not None:
                d = pos["dir"]
                xy = (ybo[i] - slip_y) if d > 0 else (yao[i] + slip_y)
                xx = (xao[i] + slip_x) if d > 0 else (xbo[i] - slip_x)
                pnl = leg_pnl(d, pos["ey"], xy, pos["lots_y"], spec_y) + leg_pnl(-d, pos["ex"], xx, pos["lots_x"], spec_x)
                pnl -= 2 * p.commission_per_lot_side * (pos["lots_y"] + pos["lots_x"])
                pnl += pos["swap"]
                equity += pnl
                risk.record_trade_result(pnl)
                trades.append(Trade("long_spread" if d > 0 else "short_spread", pos["open_ms"], int(t_ms[i]),
                                    pos["ey"], xy, pos["lots_y"], pnl, pending["reason"], i - pos["open_i"],
                                    pos["win_prob"], pos["payoff"], pos["kelly"]))
                pos = None
            pending = None

        # 2) periodic re-estimation (hedge ratio frozen between refits)
        if i + 1 >= p.window and (fit is None or (i + 1 - p.window) % p.refit_bars == 0):
            fit = fit_pair(log_y[i + 1 - p.window:i + 1], log_x[i + 1 - p.window:i + 1], p.eg_critical)
            refits += 1
            coint_passes += int(bool(fit and fit.cointegrated))
        if fit is None or i == n - 1:
            continue

        # 3) manage an open position with the model it was opened with (consistent hedge)
        if pos is not None:
            d, ef = pos["dir"], pos["fit"]
            held = i - pos["open_i"]
            z_pos = (log_y[i] - ef.alpha - ef.beta * log_x[i] - ef.mu) / ef.sigma_eq
            # mark-to-market at the prices we could exit at now
            mtm = (leg_pnl(d, pos["ey"], yb[i] if d > 0 else ya[i], pos["lots_y"], spec_y)
                   + leg_pnl(-d, pos["ex"], xa[i] if d > 0 else xb[i], pos["lots_x"], spec_x) + pos["swap"])
            if mtm <= -pos["max_loss"]:
                pending = {"kind": "close", "reason": "money_stop"}
            elif -d * z_pos <= p.z_exit:  # spread back near its mean
                pending = {"kind": "close", "reason": "mean_reverted"}
            elif -d * z_pos >= p.z_stop:
                pending = {"kind": "close", "reason": "z_stop"}
            elif held >= p.time_stop_half_lives * ef.half_life:
                pending = {"kind": "close", "reason": "time_stop"}
            continue

        s = log_y[i] - fit.alpha - fit.beta * log_x[i]
        z = (s - fit.mu) / fit.sigma_eq if fit.sigma_eq > 0 else 0.0

        # 4) entry filters
        if not fit.cointegrated:
            note("not_cointegrated")
            continue
        if not p.min_half_life <= fit.half_life <= p.max_half_life:
            note("half_life_out_of_range")
            continue
        if abs(z) < p.z_entry:
            note("no_edge")
            continue
        d = -1 if z > 0 else 1
        mid_y, mid_x = 0.5 * (yb[i] + ya[i]), 0.5 * (xb[i] + xa[i])
        cost = ((ya[i] - yb[i]) + 2 * slip_y) / mid_y + abs(fit.beta) * ((xa[i] - xb[i]) + 2 * slip_x) / mid_x
        if fit.sigma_eq < p.min_sigma_to_cost * cost:
            note("spread_vol_below_cost")
            continue
        edge = abs(s - fit.mu)
        if edge < p.cost_mult * cost:
            note("edge_below_cost")
            continue
        win = (abs(z) - p.z_exit) * fit.sigma_eq - cost
        loss = (p.z_stop - abs(z)) * fit.sigma_eq + cost
        if win <= 0 or loss <= 0:
            note("edge_below_cost")
            continue
        u_target, u_stop = (p.z_exit, p.z_stop) if z > 0 else (-p.z_exit, -p.z_stop)
        horizon = int(math.ceil(p.time_stop_half_lives * fit.half_life))
        p_raw = cached_hit_probability(z, u_target, u_stop, fit.b, horizon)
        prob = min(max(p_raw - p.win_prob_haircut, 0.0), p.max_win_prob)
        payoff = win / loss
        kelly = prob - (1 - prob) / payoff
        if prob * win - (1 - prob) * loss <= 0 or kelly <= 0:
            note("negative_ev")
            continue

        # 5) sizing: notional of Y such that a stop-out loses equity * risk_fraction
        risk_money = equity * risk.kelly_risk_fraction(model_kelly=kelly)
        cap = p.max_leg_notional_mult * max(equity, 0.0)
        notional_y = min(risk_money / loss, cap, cap / max(abs(fit.beta), 1e-9))
        lots_y = spec_y.round_volume_down(notional_y / _value_per_lot(spec_y, mid_y))
        lots_x = spec_x.round_volume_down(abs(fit.beta) * notional_y / _value_per_lot(spec_x, mid_x))
        if lots_y <= 0 or lots_x <= 0:
            note("size_zero")
            continue
        if fit.beta <= 0:
            note("negative_hedge_ratio")
            continue
        pending = {"kind": "open", "dir": d, "lots_y": lots_y, "lots_x": lots_x, "fit": fit,
                   "max_loss": risk_money, "win_prob": prob, "payoff": payoff, "kelly": kelly}

    if pos is not None:  # close at the last bar's quotes
        i, d = n - 1, pos["dir"]
        xy = yb[i] if d > 0 else ya[i]
        xx = xa[i] if d > 0 else xb[i]
        pnl = leg_pnl(d, pos["ey"], xy, pos["lots_y"], spec_y) + leg_pnl(-d, pos["ex"], xx, pos["lots_x"], spec_x)
        pnl -= 2 * p.commission_per_lot_side * (pos["lots_y"] + pos["lots_x"])
        pnl += pos["swap"]
        trades.append(Trade("long_spread" if d > 0 else "short_spread", pos["open_ms"], int(t_ms[i]), pos["ey"], xy,
                            pos["lots_y"], pnl, "end_of_data", i - pos["open_i"], pos["win_prob"], pos["payoff"], pos["kelly"]))

    metrics = compute_metrics(trades, p.initial_equity, t_ms)
    metrics.pop("ticks", None)
    metrics["bars"] = n
    metrics["swap_total"] = round(swap_total, 2)
    metrics["cointegrated_pct_of_refits"] = round(100 * coint_passes / refits, 1) if refits else None
    total = sum(reasons.values()) or 1
    metrics["why_no_entry_pct"] = {k: round(100 * v / total, 1) for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])}
    if fit is not None:
        metrics["last_fit"] = {"beta": round(fit.beta, 4), "half_life_bars": round(fit.half_life, 1),
                               "eg_t": round(fit.eg_t, 2), "cointegrated": fit.cointegrated}
    return PairsResult(trades, metrics, asdict(p))


# ------------------------------------------------------------ walk-forward
PAIR_GRID: dict[str, list[Any]] = {"z_entry": [1.5, 2.0, 2.5], "window": [240, 480, 960]}


def _brief(m: dict[str, Any]) -> dict[str, Any]:
    keys = ("trades", "net_pnl", "win_rate", "profit_factor", "t_stat", "max_drawdown_pct", "cointegrated_pct_of_refits")
    return {k: m.get(k) for k in keys}


def walk_forward_pairs(
    bars: pd.DataFrame,
    spec_y: SymbolSpec,
    spec_x: SymbolSpec,
    p: PairParams = PairParams(),
    limits: RiskLimits = RiskLimits(),
    folds: int = 4,
    grid: Optional[dict[str, list[Any]]] = None,
    min_train_trades: int = 5,
    train_prefix: bool = False,
) -> dict[str, Any]:
    """Parameters chosen on block k-1, evaluated untouched on block k.

    Each test block is prefixed with the last ``window`` bars of history so the model is
    estimated on past data only (no look-ahead) and can trade from the block's first bar.
    """
    grid = grid or PAIR_GRID
    blocks = np.array_split(np.arange(len(bars)), folds + 1)
    combos = [dict(zip(grid, v)) for v in itertools.product(*grid.values())]
    reports: list[dict[str, Any]] = []
    oos: list[Trade] = []
    for k in range(1, folds + 1):
        train_first = int(blocks[k - 1][0])
        train_first_ms = int(bars["t_ms"].iloc[train_first])
        best: Optional[tuple[float, dict[str, Any], dict[str, Any]]] = None
        for combo in combos:
            cp_ = replace(p, **combo)
            # optional warm-up prefix: the model is estimated on history before the block, so
            # short daily blocks are not mostly spent waiting for the first estimation window
            t0 = max(train_first - cp_.window, 0) if train_prefix else train_first
            train = bars.iloc[t0:int(blocks[k - 1][-1]) + 1].reset_index(drop=True)
            tr = run_pairs_backtest(train, spec_y, spec_x, cp_, limits).trades
            m = compute_metrics([t for t in tr if t.open_ms >= train_first_ms], p.initial_equity)
            if m["trades"] >= min_train_trades and (best is None or m["net_pnl"] > best[0]):
                best = (m["net_pnl"], combo, m)
        if best is None or best[0] <= 0:
            reports.append({"fold": k, "chosen": None, "note": "no profitable parameter set in training -> no trading",
                            "best_in_training": None if best is None else {"params": best[1], **_brief(best[2])}})
            logger.info("pair fold {}: nothing profitable in training; sitting out", k)
            continue
        _, combo, train_m = best
        cp = replace(p, **combo)
        start = max(int(blocks[k][0]) - cp.window, 0)
        test = bars.iloc[start:int(blocks[k][-1]) + 1].reset_index(drop=True)
        res = run_pairs_backtest(test, spec_y, spec_x, cp, limits)
        first_test_ms = int(bars["t_ms"].iloc[int(blocks[k][0])])
        fold_trades = [t for t in res.trades if t.open_ms >= first_test_ms]
        oos.extend(fold_trades)
        reports.append({"fold": k, "chosen": combo, "train": _brief(train_m),
                        "test": _brief(compute_metrics(fold_trades, p.initial_equity))})
        logger.info("pair fold {}: chose {} | train pnl={} | TEST pnl={:.2f} trades={}", k, combo, train_m["net_pnl"],
                    sum(t.pnl for t in fold_trades), len(fold_trades))
    return {"folds": reports, "out_of_sample": compute_metrics(oos, p.initial_equity)}


# --------------------------------------------------------------- synthetic
def synthetic_pair(kind: str = "coint", n_ticks: int = 690_000, dt_ms: int = 1000, seed: int = 11,
                   half_life_bars: float = 15.0, spread_sd: float = 0.0015) -> tuple[pd.DataFrame, pd.DataFrame]:
    """'coint': log Y = a + b log X + OU noise.  'indep': two unrelated random walks."""
    rng = np.random.default_rng(seed)
    ticks_per_bar = 60_000 // dt_ms
    lx = np.log(30.0) + np.cumsum(rng.standard_normal(n_ticks)) * 0.00012
    if kind == "coint":
        b = math.exp(-math.log(2) / (half_life_bars * ticks_per_bar))
        scale = spread_sd * math.sqrt(1 - b * b)
        e = np.empty(n_ticks)
        e[0] = 0.0
        eps = rng.standard_normal(n_ticks) * scale
        for i in range(1, n_ticks):
            e[i] = b * e[i - 1] + eps[i]
        ly = math.log(2600.0) - 1.0 * math.log(30.0) + 1.0 * lx + e
    elif kind == "indep":
        ly = np.log(2600.0) + np.cumsum(rng.standard_normal(n_ticks)) * 0.00012
    else:
        raise ValueError("kind must be 'coint' or 'indep'")
    t0 = int(pd.Timestamp("2026-10-05 00:00", tz="UTC").timestamp() * 1000)
    t = t0 + np.arange(n_ticks, dtype=np.int64) * dt_ms

    def frame(logp: np.ndarray, half_spread: float, digits: int) -> pd.DataFrame:
        mid = np.exp(logp)
        return pd.DataFrame({"time_msc": t, "bid": np.round(mid - half_spread, digits),
                             "ask": np.round(mid + half_spread, digits), "volume_real": 0.0})

    return frame(ly, 0.12, 3), frame(lx, 0.01, 3)


def synthetic_specs() -> tuple[SymbolSpec, SymbolSpec]:
    gold = SymbolSpec("YYYm", 3, 0.001, 0.001, 0.1, 100, 0.01, 200.0, 0.01)
    silver = SymbolSpec("XXXm", 3, 0.001, 0.001, 5.0, 5000, 0.01, 200.0, 0.01)
    return gold, silver


# --------------------------------------------------------------------- CLI
def _load_leg(path: str) -> tuple[pd.DataFrame, SymbolSpec]:
    ticks = load_ticks_csv(path)
    spec_path = Path(path).with_suffix(".spec.json")
    if not spec_path.is_file():
        logger.warning("{} not found: download with `python -m src.backtest fetch` to get the contract spec", spec_path)
        return ticks, default_spec(Path(path).stem)
    return ticks, load_spec(spec_path)


def _load_rates_leg(path: str, spread_mult: float = 1.0) -> tuple[pd.DataFrame, SymbolSpec, float, str]:
    """Load *_M1/_H1/_D1.csv bars + spec.

    Spread floor = ``spread_mult`` x median spread from real ticks: the matching tick CSV if
    present, otherwise the ``tick_spread_median`` stored in the spec by fetch-bars.
    """
    rates = pd.read_csv(path)
    missing = {"time", "open", "close", "spread"} - set(rates.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    stem = Path(path).stem
    symbol = stem.rsplit("_", 1)[0] if stem.rsplit("_", 1)[-1] in ("M1", "H1", "D1") else stem
    spec_path = Path(path).with_name(f"{symbol}.spec.json")
    spec = load_spec(spec_path) if spec_path.is_file() else default_spec(symbol)
    tick_path = Path(path).with_name(f"{symbol}.csv")
    floor, source = 0.0, "bar spread field only"
    if tick_path.is_file():
        t = load_ticks_csv(tick_path)
        floor = spread_mult * float((t.ask - t.bid).median())
        source = f"max(bar spread, {spread_mult:g} x tick median = {floor:.6g})"
    elif spec.tick_spread_median > 0:
        floor = spread_mult * spec.tick_spread_median
        source = f"max(bar spread, {spread_mult:g} x tick median = {floor:.6g})"
    return rates, spec, floor, source


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Pairs trading backtest / walk-forward")
    ap.add_argument("cmd", choices=["run", "walkforward"])
    ap.add_argument("--y", help="tick CSV of the first leg (e.g. data/XAUUSDm.csv)")
    ap.add_argument("--x", help="tick CSV of the second leg (e.g. data/XAGUSDm.csv)")
    ap.add_argument("--y-bars", help="1-minute bar CSV of the first leg (python -m src.backtest fetch-bars)")
    ap.add_argument("--x-bars", help="1-minute bar CSV of the second leg")
    ap.add_argument("--synthetic", choices=["coint", "indep"])
    ap.add_argument("--bar-seconds", type=int, default=60)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--commission", type=float, default=0.0, help="per lot per side, both legs")
    ap.add_argument("--slippage-ticks", type=int, default=1)
    ap.add_argument("--equity", type=float, default=10_000.0)
    ap.add_argument("--config")
    ap.add_argument("--trades-out")
    a = ap.parse_args(argv)
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<level>{message}</level>")

    bars: Optional[pd.DataFrame] = None
    if a.synthetic:
        ty, tx = synthetic_pair(a.synthetic)
        sy, sx = synthetic_specs()
    elif a.y_bars and a.x_bars:
        ry, sy, fy, src_y = _load_rates_leg(a.y_bars)
        rx, sx, fx, src_x = _load_rates_leg(a.x_bars)
        logger.info("spread used: {} = {}; {} = {}", sy.name, src_y, sx.name, src_x)
        bars = build_pair_bars_from_rates(ry, sy, fy, rx, sx, fx)
        a.bar_seconds = 60
    elif a.y and a.x:
        ty, sy = _load_leg(a.y)
        tx, sx = _load_leg(a.x)
    else:
        ap.error("--y-bars/--x-bars, --y/--x or --synthetic is required")

    p = PairParams(bar_seconds=a.bar_seconds, slippage_ticks=a.slippage_ticks,
                   commission_per_lot_side=a.commission, initial_equity=a.equity)
    limits = load_settings(a.config).risk
    if bars is None:
        bars = build_pair_bars(ty, tx, p.bar_seconds)
    logger.info("{} / {}: {} common {}-second bars over {:.1f} days", sy.name, sx.name, len(bars), p.bar_seconds,
                (bars.t_ms.iloc[-1] - bars.t_ms.iloc[0]) / 86_400_000 if len(bars) else 0)
    t0 = time.perf_counter()
    if a.cmd == "run":
        res = run_pairs_backtest(bars, sy, sx, p, limits)
        print("\n=== PAIRS BACKTEST (in-sample, default parameters) ===")
        print(json.dumps(res.metrics, indent=2, default=str))
        if a.trades_out:
            pd.DataFrame([asdict(t) for t in res.trades]).to_csv(a.trades_out, index=False)
    else:
        rep = walk_forward_pairs(bars, sy, sx, p, limits, folds=a.folds)
        print("\n=== PAIRS WALK-FORWARD folds ===")
        print(json.dumps({"folds": rep["folds"]}, indent=2, default=str))
        print("\n=== OUT-OF-SAMPLE (the number that matters) ===")
        print(json.dumps(rep["out_of_sample"], indent=2, default=str))
        m = rep["out_of_sample"]
        ok = m["trades"] >= 100 and (m.get("profit_factor") or 0) > 1.2 and (m.get("t_stat") or 0) > 2
        print(f"\nPRE-REGISTERED CRITERIA (>=100 trades, PF>1.2, t>2): {'PASS' if ok else 'FAIL'}")
    logger.info("done in {:.1f}s", time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
