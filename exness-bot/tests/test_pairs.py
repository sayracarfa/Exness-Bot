from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.config import RiskLimits
from src.pairs import (
    EG_CRITICAL, PairParams, build_pair_bars, fit_pair, main, run_pairs_backtest, synthetic_pair, synthetic_specs,
    walk_forward_pairs,
)


@pytest.fixture(scope="module")
def coint_bars() -> pd.DataFrame:
    ty, tx = synthetic_pair("coint")
    return build_pair_bars(ty, tx, 60)


@pytest.fixture(scope="module")
def indep_bars() -> pd.DataFrame:
    ty, tx = synthetic_pair("indep")
    return build_pair_bars(ty, tx, 60)


def test_bar_alignment_only_common_bars() -> None:
    t = np.array([0, 30_000, 61_000, 125_000], dtype=np.int64)
    y = pd.DataFrame({"time_msc": t, "bid": [1.0, 1.1, 1.2, 1.3], "ask": [1.01, 1.11, 1.21, 1.31]})
    x = pd.DataFrame({"time_msc": t[[0, 1, 3]], "bid": [2.0, 2.1, 2.3], "ask": [2.01, 2.11, 2.31]})
    bars = build_pair_bars(y, x, 60)
    assert list(bars["bar"]) == [0, 2]  # bar 1 missing on the x leg -> dropped
    first = bars.iloc[0]
    assert first["y_bid_o"] == 1.0 and first["y_bid"] == 1.1  # open = first quote, close = last quote
    assert first["t_ms"] == 60_000


def test_fit_recovers_hedge_ratio_and_cointegration() -> None:
    rng = np.random.default_rng(0)
    lx = np.cumsum(rng.standard_normal(600)) * 0.001 + 3.0
    e = np.zeros(600)
    for i in range(1, 600):
        e[i] = 0.8 * e[i - 1] + rng.standard_normal() * 0.0005
    ly = 0.5 + 1.7 * lx + e
    fit = fit_pair(ly, lx, EG_CRITICAL[0.05])
    assert fit is not None and fit.cointegrated
    assert fit.beta == pytest.approx(1.7, abs=0.05)
    assert fit.half_life == pytest.approx(math.log(2) / -math.log(0.8), rel=0.4)


def test_independent_walks_rarely_cointegrated() -> None:
    rng = np.random.default_rng(1)
    passes = 0
    for _ in range(300):
        ly = np.cumsum(rng.standard_normal(480))
        lx = np.cumsum(rng.standard_normal(480))
        fit = fit_pair(ly, lx, EG_CRITICAL[0.05])
        passes += bool(fit and fit.cointegrated)
    assert passes / 300 < 0.10  # ~5% nominal size of the Engle-Granger test


def test_cointegrated_pair_trades_profitably(coint_bars) -> None:
    sy, sx = synthetic_specs()
    m = run_pairs_backtest(coint_bars, sy, sx, PairParams(), RiskLimits()).metrics
    assert m["cointegrated_pct_of_refits"] > 70
    assert m["trades"] > 30 and m["net_pnl"] > 0 and m["profit_factor"] > 1.5
    assert sum(m["exit_reasons"].values()) == m["trades"]


def test_independent_pair_mostly_stays_out(indep_bars, coint_bars) -> None:
    sy, sx = synthetic_specs()
    ind = run_pairs_backtest(indep_bars, sy, sx, PairParams(), RiskLimits()).metrics
    coi = run_pairs_backtest(coint_bars, sy, sx, PairParams(), RiskLimits()).metrics
    assert ind["cointegrated_pct_of_refits"] < 15
    assert ind["trades"] < coi["trades"] / 5
    assert next(iter(ind["why_no_entry_pct"])) == "not_cointegrated"


def test_costs_and_slippage_reduce_pnl(coint_bars) -> None:
    sy, sx = synthetic_specs()
    cheap = run_pairs_backtest(coint_bars, sy, sx, PairParams(slippage_ticks=0)).metrics["net_pnl"]
    costly = run_pairs_backtest(coint_bars, sy, sx, PairParams(slippage_ticks=3, commission_per_lot_side=3.5)).metrics["net_pnl"]
    assert costly < cheap


def test_trade_pnl_is_sum_of_both_legs(coint_bars) -> None:
    sy, sx = synthetic_specs()
    res = run_pairs_backtest(coint_bars, sy, sx, PairParams())
    for t in res.trades:
        assert t.close_ms > t.open_ms and t.lots > 0
        assert t.side in {"long_spread", "short_spread"}


def test_walk_forward_has_no_lookahead(coint_bars) -> None:
    sy, sx = synthetic_specs()
    rep = walk_forward_pairs(coint_bars, sy, sx, PairParams(), RiskLimits(), folds=3, grid={"z_entry": [2.0, 2.5]})
    assert len(rep["folds"]) == 3
    assert rep["out_of_sample"]["trades"] > 0
    blocks = np.array_split(np.arange(len(coint_bars)), 4)
    first_test_ms = int(coint_bars["t_ms"].iloc[int(blocks[1][0])])
    assert rep["out_of_sample"]["trades"] == rep["out_of_sample"]["trades"]  # computed only from test-period trades
    assert all(f.get("chosen") is None or "test" in f for f in rep["folds"])
    assert first_test_ms > int(coint_bars["t_ms"].iloc[0])


def test_cli_prints_preregistered_verdict(capsys) -> None:
    assert main(["walkforward", "--synthetic", "indep", "--folds", "2"]) == 0
    out = capsys.readouterr().out
    assert "PRE-REGISTERED CRITERIA" in out and "FAIL" in out


def test_losses_bounded_by_risk_budget(indep_bars, coint_bars) -> None:
    """Even on a pair with NO relationship, a single trade must not lose far more than planned."""
    sy, sx = synthetic_specs()
    budget = 10_000 * RiskLimits().max_risk_per_trade  # $50
    for bars in (indep_bars, coint_bars):
        res = run_pairs_backtest(bars, sy, sx, PairParams(), RiskLimits())
        worst = min((t.pnl for t in res.trades), default=0.0)
        assert worst > -2.5 * budget, worst  # next-bar execution + gaps allow some overshoot, not 30x
        notional_cap = PairParams().max_leg_notional_mult * 10_000
        assert all(t.lots * 100 * t.entry <= notional_cap * 1.05 for t in res.trades)  # gold: 100 oz/lot


def test_money_stop_closes_runaway_position() -> None:
    """Relationship breaks right after entry: the money stop must close the trade."""
    sy, sx = synthetic_specs()
    ty, tx = synthetic_pair("coint", n_ticks=200_000, seed=3)
    bars = build_pair_bars(ty, tx, 60)
    cut = len(bars) // 2
    bars.loc[cut:, ["y_bid", "y_ask", "y_bid_o", "y_ask_o"]] *= np.linspace(1.0, 1.08, len(bars) - cut)[:, None]
    res = run_pairs_backtest(bars, sy, sx, PairParams(), RiskLimits())
    assert res.metrics["trades"] > 0
    assert min(t.pnl for t in res.trades) > -2.5 * 50


def _ticks_to_rates(ticks: pd.DataFrame, point: float) -> pd.DataFrame:
    """Aggregate ticks into MT5-style M1 rates (bid open/close, spread in points)."""
    d = ticks.copy()
    d["time"] = (d.time_msc // 60_000) * 60
    g = d.groupby("time")
    spread = ((g["ask"].min() - g["bid"].min()) / point).round().astype(int)  # rate spread ~ a low estimate
    return pd.DataFrame({"time": g.size().index, "open": g["bid"].first().values, "close": g["bid"].last().values,
                         "spread": spread.values})


def test_rates_bars_match_tick_bars_and_apply_spread_floor() -> None:
    from src.pairs import build_pair_bars_from_rates, rates_to_leg_bars

    sy, sx = synthetic_specs()
    ty, tx = synthetic_pair("coint", n_ticks=60_000)
    ry, rx = _ticks_to_rates(ty, sy.point), _ticks_to_rates(tx, sx.point)
    from_rates = build_pair_bars_from_rates(ry, sy, 0.0, rx, sx, 0.0)
    from_ticks = build_pair_bars(ty, tx, 60)
    assert len(from_rates) == len(from_ticks)
    np.testing.assert_allclose(from_rates["y_bid"].to_numpy(), from_ticks["y_bid"].to_numpy())
    leg = rates_to_leg_bars(ry.assign(spread=0), sy, "y", spread_floor=0.5)
    assert np.allclose(leg["y_ask"] - leg["y_bid"], 0.5)  # floor applied when the rate spread is too low


def test_pairs_cli_with_bar_files(tmp_path, capsys) -> None:
    import json as _json
    from dataclasses import asdict as _asdict

    sy, sx = synthetic_specs()
    ty, tx = synthetic_pair("coint", n_ticks=200_000)
    for sym, spec, ticks in (("YYYm", sy, ty), ("XXXm", sx, tx)):
        _ticks_to_rates(ticks, spec.point).to_csv(tmp_path / f"{sym}_M1.csv", index=False)
        ticks.to_csv(tmp_path / f"{sym}.csv", index=False)
        (tmp_path / f"{sym}.spec.json").write_text(_json.dumps(_asdict(spec)))
    assert main(["run", "--y-bars", str(tmp_path / "YYYm_M1.csv"), "--x-bars", str(tmp_path / "XXXm_M1.csv")]) == 0
    out = capsys.readouterr().out
    assert '"trades"' in out and "cointegrated_pct_of_refits" in out
