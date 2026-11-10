"""Tests for the priority-1 additions: ADF test, time stop, per-trade Kelly, backtester."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from src.backtest import BacktestConfig, compute_metrics, load_ticks_csv, run_backtest, synthetic_ticks, walk_forward, default_spec
from src.config import RiskLimits, StrategyParams
from src.models import Side
from src.risk_controller import RiskController
from src.strategy import (
    OUEstimator, SignalAction, SignalEngine, ou_hit_probability, ou_hit_probability_mc,
)


def ar1(n: int, b: float, seed: int, level: float = 100.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = b * x[i - 1] + rng.standard_normal()
    return x + level


class TestADF:
    def test_random_walks_mostly_rejected(self) -> None:
        rng = np.random.default_rng(0)
        passed = 0
        for _ in range(400):
            est = OUEstimator(300, 120, 150.0, adf_critical=-2.87)
            for v in np.cumsum(rng.standard_normal(300)):
                est.update(float(v))
            passed += est.fit() is not None
        assert passed / 400 < 0.10  # ~5% nominal size (was ~84% without the test)

    def test_without_adf_random_walks_pass(self) -> None:
        rng = np.random.default_rng(0)
        passed = 0
        for _ in range(200):
            est = OUEstimator(300, 120, 150.0, adf_critical=None)
            for v in np.cumsum(rng.standard_normal(300)):
                est.update(float(v))
            passed += est.fit() is not None
        assert passed / 200 > 0.6  # documents the bias the ADF test fixes

    def test_strong_reversion_accepted(self) -> None:
        est = OUEstimator(300, 120, 150.0)
        for v in ar1(300, 0.9, seed=1):
            est.update(float(v))
        p = est.fit()
        assert p is not None and p.adf_t < -2.87

    def test_rolling_sums_match_batch_ols(self) -> None:
        series = ar1(5_000, 0.95, seed=2, level=1.085) * 1e-3 + 1.0
        est = OUEstimator(300, 120, 1e6, adf_critical=None)
        for v in series:
            est.update(float(v))
        p = est.fit()
        w = series[-300:]
        x, y = w[:-1], w[1:]
        b = np.cov(x, y, bias=True)[0, 1] / np.var(x)
        a = y.mean() - b * x.mean()
        assert p is not None
        assert p.b == pytest.approx(b, rel=1e-9)
        assert p.mu == pytest.approx(a / (1 - b), rel=1e-9)


class TestHitProbability:
    def test_mc_matches_analytic_without_time_limit(self) -> None:
        exact = ou_hit_probability(-1.0, 0.0, -2.0)
        mc = ou_hit_probability_mc(-1.0, 0.0, -2.0, b=0.99, max_ticks=20_000, paths=4000, rng=np.random.default_rng(3))
        assert mc == pytest.approx(exact, abs=0.04)

    def test_time_limit_lowers_probability(self) -> None:
        long = ou_hit_probability_mc(-2.5, 0.0, -5.0, b=0.98, max_ticks=1000, rng=np.random.default_rng(4))
        short = ou_hit_probability_mc(-2.5, 0.0, -5.0, b=0.98, max_ticks=20, rng=np.random.default_rng(4))
        assert short < long

    def test_short_side_symmetry(self) -> None:
        a = ou_hit_probability_mc(-2.0, 0.0, -4.0, 0.97, 200, rng=np.random.default_rng(5))
        b = ou_hit_probability_mc(2.0, 0.0, 4.0, 0.97, 200, rng=np.random.default_rng(5))
        assert a == pytest.approx(b, abs=0.03)


def _engine(**over) -> SignalEngine:
    p = StrategyParams(ou_window=400, ou_min_obs=200, ou_max_half_life_ticks=500, vpin_bucket_volume=5,
                       vpin_num_buckets=10, vpin_threshold=0.95, signal_cooldown_s=0, min_edge_to_spread=0.0, **over)
    return SignalEngine(p)


def _warm(engine: SignalEngine, factory, n: int = 400) -> None:
    for i, v in enumerate(ar1(n, 0.9, seed=5, level=0) * 0.0002 / math.sqrt(1 / (1 - 0.81)) + 1.085):
        engine.on_snapshot(factory(price=float(v), ts_ns=i * 1_000_000))


class TestSignalEconomics:
    def test_entry_carries_probability_payoff_and_kelly(self, snapshot_factory) -> None:
        eng = _engine()
        _warm(eng, snapshot_factory)
        sig = eng.on_snapshot(snapshot_factory(price=1.0850 - 0.0012, ts_ns=10**12))
        assert sig.action is SignalAction.OPEN_LONG, sig.reason
        assert 0 < sig.win_prob <= 0.8
        assert sig.payoff_ratio > 0 and sig.kelly > 0 and sig.expected_value > 0
        assert sig.half_life is not None

    def test_negative_ev_blocks_entry(self, snapshot_factory) -> None:
        eng = _engine()
        eng.max_model_win_prob = 0.05  # model says: almost never wins
        _warm(eng, snapshot_factory)
        sig = eng.on_snapshot(snapshot_factory(price=1.0850 - 0.0012, ts_ns=10**12))
        assert sig.action is SignalAction.HOLD and sig.reason == "negative_ev"

    def test_time_stop_closes_stale_position(self, snapshot_factory) -> None:
        eng = _engine()  # default: exit after 3 half-lives
        _warm(eng, snapshot_factory)
        sig = eng.on_snapshot(snapshot_factory(price=1.0850 - 0.0012, ts_ns=10**12))
        assert sig.action is SignalAction.OPEN_LONG
        hl = sig.half_life
        out = None
        for k in range(int(3 * hl) + 5):  # price stays far below the mean -> never reverts
            out = eng.on_snapshot(snapshot_factory(price=1.0850 - 0.0012, ts_ns=10**12 + k), position_side=Side.BUY)
            if out.action is SignalAction.CLOSE:
                break
        assert out is not None and out.action is SignalAction.CLOSE and out.reason == "time_stop"

    def test_time_stop_counter_resets_when_flat(self, snapshot_factory) -> None:
        eng = _engine()
        snap = snapshot_factory()
        eng.on_snapshot(snap, position_side=Side.BUY)
        eng.on_snapshot(replace(snap, is_latest=False), position_side=Side.BUY)
        assert eng._states["EURUSDm"].ticks_in_position == 2
        eng.on_snapshot(snap)
        assert eng._states["EURUSDm"].ticks_in_position == 0


class TestKellySizing:
    def test_model_kelly_used_then_capped_by_track_record(self, eurusd_spec) -> None:
        rc = RiskController(RiskLimits(kelly_fraction=0.25, max_risk_per_trade=0.05, min_trades_for_kelly=10), 0.7)
        assert rc.kelly_risk_fraction(model_kelly=0.2) == pytest.approx(0.05)
        assert rc.kelly_risk_fraction(model_kelly=-0.1) == 0.0
        for _ in range(5):
            rc.record_trade_result(10.0)
        for _ in range(5):
            rc.record_trade_result(-10.0)  # realised: p=0.5, b=1 -> Kelly 0
        assert rc.kelly_risk_fraction(model_kelly=0.2) == 0.0
        assert rc.position_size(eurusd_spec, 10_000, 0.001, model_kelly=0.2) == 0.0


class TestBacktest:
    def test_mean_reverting_data_is_profitable(self) -> None:
        ticks = synthetic_ticks("ou", n=60_000, seed=1)
        res = run_backtest(ticks, default_spec(), StrategyParams(), RiskLimits())
        m = res.metrics
        assert m["trades"] > 20 and m["net_pnl"] > 0 and m["profit_factor"] > 1.5

    def test_random_walk_has_no_edge(self) -> None:
        ticks = synthetic_ticks("rw", n=60_000, seed=1)
        ou = run_backtest(synthetic_ticks("ou", n=60_000, seed=1), default_spec(), StrategyParams(), RiskLimits())
        rw = run_backtest(ticks, default_spec(), StrategyParams(), RiskLimits())
        assert rw.metrics["trades"] < ou.metrics["trades"] / 3  # ADF keeps the bot mostly out

    def test_costs_reduce_pnl(self) -> None:
        ticks = synthetic_ticks("ou", n=40_000, seed=2)
        base = run_backtest(ticks, default_spec(), StrategyParams(), RiskLimits())
        costly = run_backtest(ticks, default_spec(), StrategyParams(), RiskLimits(),
                              BacktestConfig(commission_per_lot_side=3.5, slippage_ticks=1))
        assert costly.metrics["net_pnl"] < base.metrics["net_pnl"]

    def test_trades_are_consistent(self) -> None:
        res = run_backtest(synthetic_ticks("ou", n=30_000, seed=3), default_spec(), StrategyParams(), RiskLimits())
        for t in res.trades:
            assert t.close_ms >= t.open_ms and t.lots > 0
            sign = 1 if t.side == "buy" else -1
            assert t.pnl == pytest.approx(sign * (t.exit - t.entry) / 1e-5 * 1.0 * t.lots, abs=1e-6)
        assert sum(res.metrics["exit_reasons"].values()) == len(res.trades)

    def test_walk_forward_reports_out_of_sample(self) -> None:
        ticks = synthetic_ticks("ou", n=60_000, seed=4)
        rep = walk_forward(ticks, default_spec(), StrategyParams(), RiskLimits(), folds=2,
                           grid={"z_entry": [2.0, 2.5]})
        assert len(rep["folds"]) == 2
        assert rep["out_of_sample"]["trades"] > 0
        assert all(f["chosen"] for f in rep["folds"])

    def test_csv_roundtrip(self, tmp_path) -> None:
        df = synthetic_ticks("ou", n=1000)
        path = tmp_path / "t.csv"
        df.sample(frac=1, random_state=0).to_csv(path, index=False)  # shuffled on disk
        back = load_ticks_csv(path)
        assert back.time_msc.is_monotonic_increasing and len(back) == 1000

    def test_metrics_empty(self) -> None:
        assert compute_metrics([], 10_000)["trades"] == 0
