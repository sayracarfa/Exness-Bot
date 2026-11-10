"""Pre-registered DAILY pairs scan (with spreads, slippage and swaps).

The protocol below was fixed BEFORE any daily result was seen. Do not edit the pair list,
costs, splits or thresholds after looking at results - that would turn the holdout into
another training set and the verdict would mean nothing.

    Pairs     16 economically linked pairs (PAIRS)
    Data      daily bars, longest history available (up to 10 years)
    Costs     spread = max(bar spread, 2 x median spread measured from real ticks),
              1 tick slippage per fill, broker swap rates (triple-swap weekday from the broker)
    Selection first 70% of each pair's bars. 3-fold walk-forward; the pair passes screening
              if its out-of-sample folds give >= 8 trades, profit factor > 1.2 and t > 1
    Holdout   last 30%, evaluated ONCE, only for pairs that passed screening, with the
              parameters that earned the most on the whole selection period
    Verdict   combined holdout trades of the selected pairs: >= 30 trades, PF > 1.2, t > 2 -> PASS

Usage (Windows VPS, MT5 open and logged in):
    python -m src.pair_scan fetch-all --out data_d1
    python -m src.pair_scan run --data data_d1
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.backtest import Trade, compute_metrics, fetch_mt5_bars
from src.config import RiskLimits, load_settings
from src.models import SymbolSpec
from src.pairs import (PairParams, _load_rates_leg, build_pair_bars_from_rates, run_pairs_backtest,
                       walk_forward_pairs)

# ----------------------------------------------------------- pre-registration
PAIRS: list[tuple[str, str]] = [
    ("XAUUSD", "XAGUSD"), ("XAUUSD", "XPTUSD"), ("XPTUSD", "XPDUSD"), ("XAUUSD", "EURUSD"),
    ("EURUSD", "GBPUSD"), ("AUDUSD", "NZDUSD"), ("EURJPY", "GBPJPY"),
    ("KO", "PEP"), ("V", "MA"), ("XOM", "CVX"), ("JPM", "BAC"), ("GS", "MS"),
    ("HD", "LOW"), ("UPS", "FDX"), ("PFE", "MRK"), ("MCD", "YUM"),
]
HISTORY_DAYS = 3650
SPREAD_MULT = 2.0
SELECTION_FRACTION = 0.70
SCREEN_FOLDS = 3
SCREEN_RULE = {"min_trades": 8, "min_pf": 1.2, "min_t": 1.0}
FINAL_RULE = {"min_trades": 30, "min_pf": 1.2, "min_t": 2.0}
MIN_TRAIN_TRADES = 5
MIN_BARS = 500
# Notional equity is large on purpose: daily stops are several % wide, so with a small account
# most signals round below the broker's minimum lot and the test would measure account size,
# not edge. PF and t-stat do not depend on scale; account feasibility is a separate question.
DAILY_PARAMS = PairParams(bar_seconds=86_400, window=120, refit_bars=5, min_half_life=1.0, max_half_life=30.0,
                          slippage_ticks=1, swap_mult=1.0, initial_equity=1_000_000.0)
DAILY_GRID: dict[str, list[Any]] = {"z_entry": [1.5, 2.0, 2.5], "window": [60, 120, 250]}


def passes(m: dict[str, Any], rule: dict[str, float]) -> bool:
    """trades / profit factor / t-stat rule. PF is None when there are no losing trades."""
    n = m.get("trades") or 0
    if n < rule["min_trades"]:
        return False
    pf = m.get("profit_factor")
    pf_ok = (m.get("net_pnl", 0) > 0) if pf is None else pf > rule["min_pf"]
    t = m.get("t_stat")
    return bool(pf_ok and t is not None and t > rule["min_t"])


def _brief(m: dict[str, Any]) -> dict[str, Any]:
    return {k: m.get(k) for k in ("trades", "net_pnl", "win_rate", "profit_factor", "t_stat", "max_drawdown_pct")}


# ------------------------------------------------------------------ one pair
def scan_pair(bars: pd.DataFrame, spec_y: SymbolSpec, spec_x: SymbolSpec, p: PairParams = DAILY_PARAMS,
              limits: RiskLimits = RiskLimits(), grid: Optional[dict[str, list[Any]]] = None) -> dict[str, Any]:
    """Screen on the first 70%; returns the selection-period results and the frozen parameters.

    The holdout is NOT touched here - see :func:`evaluate_holdout`.
    """
    grid = grid or DAILY_GRID
    n = len(bars)
    split = int(n * SELECTION_FRACTION)
    out: dict[str, Any] = {"bars": n, "selection_bars": split, "holdout_bars": n - split}
    if n < MIN_BARS:
        out.update(status="insufficient_data", screen_pass=False)
        return out
    sel = bars.iloc[:split].reset_index(drop=True)
    wf = walk_forward_pairs(sel, spec_y, spec_x, p, limits, folds=SCREEN_FOLDS, grid=grid,
                            min_train_trades=MIN_TRAIN_TRADES, train_prefix=True)
    screen = wf["out_of_sample"]
    out["screen"] = _brief(screen)
    out["screen_folds"] = [{"fold": f["fold"], "chosen": f["chosen"], "test": f.get("test")} for f in wf["folds"]]
    out["screen_pass"] = passes(screen, SCREEN_RULE)

    # parameters for the holdout: best net P&L over the whole selection period
    best: Optional[tuple[float, dict[str, Any], dict[str, Any]]] = None
    for z in grid.get("z_entry", [p.z_entry]):
        for w in grid.get("window", [p.window]):
            combo = {"z_entry": z, "window": w}
            m = run_pairs_backtest(sel, spec_y, spec_x, replace(p, **combo), limits).metrics
            if m["trades"] >= MIN_TRAIN_TRADES and (best is None or m["net_pnl"] > best[0]):
                best = (m["net_pnl"], combo, m)
    out["frozen_params"] = None if best is None else best[1]
    out["selection_full"] = None if best is None else {**_brief(best[2]), "swap_total": best[2].get("swap_total")}
    out["status"] = "selected" if out["screen_pass"] and best is not None else "rejected"
    return out


def evaluate_holdout(bars: pd.DataFrame, spec_y: SymbolSpec, spec_x: SymbolSpec, combo: dict[str, Any],
                     p: PairParams = DAILY_PARAMS, limits: RiskLimits = RiskLimits()) -> tuple[list[Trade], dict[str, Any]]:
    """Run once on the last 30% (with a warm-up prefix of history so the model uses past data only)."""
    split = int(len(bars) * SELECTION_FRACTION)
    cp = replace(p, **combo)
    start = max(split - cp.window, 0)
    res = run_pairs_backtest(bars.iloc[start:].reset_index(drop=True), spec_y, spec_x, cp, limits)
    first_ms = int(bars["t_ms"].iloc[split])
    trades = [t for t in res.trades if t.open_ms >= first_ms]
    m = compute_metrics(trades, p.initial_equity)
    return trades, {**_brief(m), "swap_total": round(res.metrics.get("swap_total", 0.0), 2)}


# ------------------------------------------------------------------ whole scan
def run_scan(pairs_data: dict[str, tuple[pd.DataFrame, SymbolSpec, SymbolSpec, dict[str, Any]]],
             p: PairParams = DAILY_PARAMS, limits: RiskLimits = RiskLimits(),
             grid: Optional[dict[str, list[Any]]] = None) -> dict[str, Any]:
    """``pairs_data``: name -> (pair bars, spec_y, spec_x, info dict). Returns the full report."""
    rows: dict[str, Any] = {}
    for name, (bars, sy, sx, info) in pairs_data.items():
        logger.info("screening {} ({} daily bars)", name, len(bars))
        rows[name] = {**info, **scan_pair(bars, sy, sx, p, limits, grid)}

    selected = [k for k, r in rows.items() if r["status"] == "selected"]
    logger.info("{} of {} pairs passed screening: {}", len(selected), len(rows), selected or "none")
    holdout_trades: list[Trade] = []
    for name in selected:  # the locked holdout is opened exactly once, for selected pairs only
        bars, sy, sx, _ = pairs_data[name]
        tr, m = evaluate_holdout(bars, sy, sx, rows[name]["frozen_params"], p, limits)
        rows[name]["holdout"] = m
        holdout_trades.extend(tr)
    portfolio = compute_metrics(sorted(holdout_trades, key=lambda t: t.open_ms), p.initial_equity)
    verdict = "PASS" if selected and passes(portfolio, FINAL_RULE) else "FAIL"
    return {
        "protocol": {"pairs": len(PAIRS), "spread_mult": SPREAD_MULT, "selection_fraction": SELECTION_FRACTION,
                     "screen_folds": SCREEN_FOLDS, "screen_rule": SCREEN_RULE, "final_rule": FINAL_RULE,
                     "grid": grid or DAILY_GRID, "params": asdict(p)},
        "pairs": rows,
        "selected": selected,
        "portfolio_holdout": _brief(portfolio),
        "verdict": verdict,
        "verdict_reason": ("no pair passed screening" if not selected else
                           f"combined holdout {'meets' if verdict == 'PASS' else 'misses'} "
                           f">= {FINAL_RULE['min_trades']} trades, PF > {FINAL_RULE['min_pf']}, t > {FINAL_RULE['min_t']}"),
    }


# ------------------------------------------------------------------ data
def symbol_names(suffix: str) -> list[str]:
    return sorted({s + suffix for pair in PAIRS for s in pair})


def load_pairs(data_dir: Path, suffix: str, spread_mult: float = SPREAD_MULT
               ) -> tuple[dict[str, tuple[pd.DataFrame, SymbolSpec, SymbolSpec, dict[str, Any]]], list[str]]:
    data: dict[str, tuple[pd.DataFrame, SymbolSpec, SymbolSpec, dict[str, Any]]] = {}
    missing: list[str] = []
    for y, x in PAIRS:
        fy, fx = data_dir / f"{y}{suffix}_D1.csv", data_dir / f"{x}{suffix}_D1.csv"
        name = f"{y}/{x}"
        if not fy.is_file() or not fx.is_file():
            missing.append(f"{name} (missing {', '.join(f.name for f in (fy, fx) if not f.is_file())})")
            continue
        ry, sy, fly, _ = _load_rates_leg(str(fy), spread_mult)
        rx, sx, flx, _ = _load_rates_leg(str(fx), spread_mult)
        if sy.tick_spread_median <= 0 or sx.tick_spread_median <= 0:
            logger.warning("{}: no tick spread measurement in spec - using the bar spread field only", name)
        bars = build_pair_bars_from_rates(ry, sy, fly, rx, sx, flx, bar_seconds=86_400)
        years = (bars["t_ms"].iloc[-1] - bars["t_ms"].iloc[0]) / 86_400_000 / 365.25 if len(bars) else 0.0
        info = {"years": round(float(years), 1), "spread_floor": [round(fly, 6), round(flx, 6)],
                "swap_y": [sy.swap_long, sy.swap_short, sy.swap_mode],
                "swap_x": [sx.swap_long, sx.swap_short, sx.swap_mode]}
        data[name] = (bars, sy, sx, info)
    return data, missing


MIN_FETCH_BARS = 250  # less than ~1 year of daily bars means MT5 did not deliver the history


async def fetch_all(out_dir: Path, suffix: str, days: float = HISTORY_DAYS, config: Optional[str] = None,
                    only: Optional[list[str]] = None, wait_s: float = 5.0) -> list[str]:
    """Download D1 bars + spec (swaps, tick spread median) for every symbol in PAIRS over one connection.

    ``only``: re-download just these symbols (e.g. the ones that failed last time).
    A failed symbol's old files are removed so a stale partial download is never scanned.
    """
    from src.mt5_client import AsyncMT5Client, load_backend

    settings = load_settings(config)
    client = AsyncMT5Client(load_backend(False))
    await client.connect(settings.mt5)
    failed: list[str] = []
    try:
        for sym in (only or symbol_names(suffix)):
            try:
                await fetch_mt5_bars(sym, days, out_dir, timeframe="D1", client=client,
                                     min_bars=MIN_FETCH_BARS, wait_s=wait_s)
            except Exception as exc:
                logger.error("{}: {}", sym, exc)
                failed.append(sym)
                for f in (out_dir / f"{sym}_D1.csv", out_dir / f"{sym}.spec.json"):
                    f.unlink(missing_ok=True)
    finally:
        await client.disconnect()
    if failed:
        logger.warning("Not downloaded: {} - the pairs using them will be reported as missing. To retry only "
                       "these: python -m src.pair_scan fetch-all --only {}", failed, " ".join(failed))
    return failed


# ------------------------------------------------------------------ synthetic check
def synthetic_daily_rates(kind: str, n_days: int = 2600, seed: int = 3, half_life: float = 6.0,
                          spread_pts: int = 30) -> tuple[pd.DataFrame, pd.DataFrame, SymbolSpec, SymbolSpec]:
    """Weekday daily bars for a cointegrated ('coint') or independent ('indep') pair."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2016-01-04", periods=n_days)
    times = ((days - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)).to_numpy(np.int64)
    log_x = np.log(100.0) + np.cumsum(0.012 * rng.standard_normal(n_days))
    if kind == "coint":
        b = 0.5 ** (1 / half_life)
        e = np.zeros(n_days)
        for i in range(1, n_days):
            e[i] = b * e[i - 1] + 0.012 * rng.standard_normal()
        log_y = np.log(50.0) + 1.0 * (log_x - log_x[0]) + e
    else:
        log_y = np.log(50.0) + np.cumsum(0.012 * rng.standard_normal(n_days))

    def rates(lg: np.ndarray) -> pd.DataFrame:
        close = np.exp(lg)
        open_ = np.concatenate([[close[0]], close[:-1]]) * (1 + 0.002 * rng.standard_normal(n_days))
        return pd.DataFrame({"time": times, "open": np.round(open_, 2), "close": np.round(close, 2),
                             "spread": np.full(n_days, spread_pts)})

    spec_y = SymbolSpec("YYm", 2, 0.01, 0.01, 1.0, 100, 0.1, 500.0, 0.1, swap_long=-3.0, swap_short=-1.0, swap_mode=1)
    spec_x = SymbolSpec("XXm", 2, 0.01, 0.01, 1.0, 100, 0.1, 500.0, 0.1, swap_long=-3.0, swap_short=-1.0, swap_mode=1)
    return rates(log_y), rates(log_x), spec_y, spec_x


# ------------------------------------------------------------------ report / CLI
def _fmt(v: Any) -> str:
    return "-" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))


def print_report(rep: dict[str, Any], missing: list[str]) -> None:
    print("\n=== PRE-REGISTERED DAILY PAIRS SCAN ===")
    print("selection = first 70% | screen: OOS walk-forward >= 8 trades, PF > 1.2, t > 1")
    print("holdout  = last 30%, opened once | PASS: >= 30 trades, PF > 1.2, t > 2 (selected pairs combined)")
    print(f"costs: spread = max(bar spread, {SPREAD_MULT:g} x tick median), 1 tick slippage, broker swaps\n")
    hdr = f"{'pair':<16}{'years':>6}{'scr.trades':>11}{'scr.PF':>8}{'scr.t':>7}{'scr.pnl':>10}  status"
    print(hdr)
    print("-" * len(hdr))
    for name, r in rep["pairs"].items():
        s = r.get("screen") or {}
        print(f"{name:<16}{_fmt(r.get('years')):>6}{_fmt(s.get('trades')):>11}{_fmt(s.get('profit_factor')):>8}"
              f"{_fmt(s.get('t_stat')):>7}{_fmt(s.get('net_pnl')):>10}  {r['status']}")
    for m in missing:
        print(f"{m:<16}  -> skipped")
    if rep["selected"]:
        print("\nLOCKED HOLDOUT (selected pairs only):")
        for name in rep["selected"]:
            h = rep["pairs"][name]["holdout"]
            print(f"  {name:<14} params={rep['pairs'][name]['frozen_params']} trades={h['trades']} "
                  f"pnl={_fmt(h['net_pnl'])} PF={_fmt(h['profit_factor'])} t={_fmt(h['t_stat'])} swap={h['swap_total']}")
        print(f"  PORTFOLIO: {rep['portfolio_holdout']}")
    incomplete = [k for k, r in rep["pairs"].items() if r["status"] == "insufficient_data"]
    if incomplete:
        print(f"\nWARNING: {len(incomplete)} pair(s) had too little data: {incomplete}. If MT5 simply did not "
              "deliver the history, re-fetch them before reading the verdict.")
    print(f"\nVERDICT: {rep['verdict']} - {rep['verdict_reason']}")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Pre-registered daily pairs scan")
    sub = ap.add_subparsers(dest="cmd", required=True)
    fa = sub.add_parser("fetch-all", help="download daily bars + specs for all 16 pairs from MT5")
    fa.add_argument("--out", default="data_d1")
    fa.add_argument("--suffix", default="m", help="broker symbol suffix (Exness Standard: m)")
    fa.add_argument("--days", type=float, default=HISTORY_DAYS)
    fa.add_argument("--config")
    fa.add_argument("--only", nargs="+", help="re-download only these symbols, e.g. --only XAUUSDm GBPJPYm")
    rn = sub.add_parser("run", help="run the scan on downloaded data")
    rn.add_argument("--data", default="data_d1")
    rn.add_argument("--suffix", default="m")
    rn.add_argument("--report", default="pair_scan_report.json")
    rn.add_argument("--synthetic", choices=["coint", "indep"], help="self-check on synthetic daily pairs")
    args = ap.parse_args(argv)
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<level>{level: <7}</level> {message}")

    if args.cmd == "fetch-all":
        asyncio.run(fetch_all(Path(args.out), args.suffix, args.days, args.config, args.only))
        return 0

    if args.synthetic:
        ry, rx, sy, sx = synthetic_daily_rates(args.synthetic)
        bars = build_pair_bars_from_rates(ry, sy, 0.0, rx, sx, 0.0, bar_seconds=86_400)
        data, missing = {f"synthetic-{args.synthetic}": (bars, sy, sx, {"years": 10.0})}, []
    else:
        data, missing = load_pairs(Path(args.data), args.suffix)
        if not data:
            logger.error("No pair data in {} - run `python -m src.pair_scan fetch-all` first", args.data)
            return 2
    rep = run_scan(data)
    print_report(rep, missing)
    Path(args.report).write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    print(f"(full report: {args.report})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
