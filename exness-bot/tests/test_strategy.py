from __future__ import annotations

import math

import numpy as np
import pytest

from src.config import StrategyParams
from src.models import Side
from src.strategy import OUEstimator, SignalAction, SignalEngine, VPINCalculator, norm_cdf


def simulate_ou(n: int, mu: float, b: float, sd_eq: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sig_eps = sd_eq * math.sqrt(1 - b * b)
    x = np.empty(n)
    x[0] = mu
    for i in range(1, n):
        x[i] = mu + b * (x[i - 1] - mu) + sig_eps * rng.standard_normal()
    return x


class TestOU:
    def test_recovers_parameters(self) -> None:
        mu, b, sd = 1.0850, 0.97, 0.0002
        est = OUEstimator(window=5000, min_obs=100, max_half_life=1e6)
        for v in simulate_ou(5000, mu, b, sd, seed=1):
            est.update(float(v))
        p = est.fit()
        assert p is not None
        assert p.mu == pytest.approx(mu, abs=3e-5)
        assert p.b == pytest.approx(b, abs=0.01)
        assert p.sigma_eq == pytest.approx(sd, rel=0.15)
        assert p.half_life == pytest.approx(math.log(2) / -math.log(b), rel=0.35)

    def test_zscore(self) -> None:
        est = OUEstimator(window=3000, min_obs=100, max_half_life=1e6)
        for v in simulate_ou(3000, 100.0, 0.9, 1.0, seed=2):
            est.update(float(v))
        p = est.fit()
        assert p is not None
        assert OUEstimator.zscore(p.mu + 2 * p.sigma_eq, p) == pytest.approx(2.0)

    def test_warming_up_and_random_walk_rejected(self) -> None:
        est = OUEstimator(window=500, min_obs=100, max_half_life=50)
        est.update(1.0)
        assert est.fit() is None and est.reason == "warming_up"
        rng = np.random.default_rng(3)
        for v in np.cumsum(rng.standard_normal(500)) + 100:  # random walk: b ~ 1
            est.update(float(v))
        assert est.fit() is None
        assert est.reason in {"not_mean_reverting", "half_life_too_long", "no_stationarity"}

    def test_flat_price(self) -> None:
        est = OUEstimator(window=200, min_obs=50, max_half_life=100)
        for _ in range(100):
            est.update(1.0)
        assert est.fit() is None and est.reason == "flat_price"


class TestVPIN:
    def test_norm_cdf(self) -> None:
        assert norm_cdf(0) == pytest.approx(0.5)
        assert norm_cdf(1.96) == pytest.approx(0.975, abs=1e-3)

    def test_not_ready_until_buckets_full(self) -> None:
        v = VPINCalculator(bucket_volume=10, num_buckets=5)
        for i in range(30):
            v.update(100 + (i % 2) * 0.01, 1.0)
        assert v.value is None
        assert v.buckets_filled == 2

    def test_balanced_flow_low_toxic_flow_high(self) -> None:
        rng = np.random.default_rng(4)
        balanced = VPINCalculator(bucket_volume=20, num_buckets=20)
        price = 100.0
        for _ in range(2000):
            price += rng.normal(0, 0.01)
            balanced.update(price, 1.0)
        toxic = VPINCalculator(bucket_volume=20, num_buckets=20)
        price = 100.0
        for i in range(2000):
            # one-sided informed buying: persistent up-moves
            price += abs(rng.normal(0.01, 0.002)) if i > 200 else rng.normal(0, 0.01)
            toxic.update(price, 1.0)
        assert balanced.value is not None and toxic.value is not None
        assert 0.0 <= balanced.value < 0.5
        assert toxic.value > 0.75
        assert toxic.value > balanced.value

    def test_large_trade_spans_buckets(self) -> None:
        v = VPINCalculator(bucket_volume=10, num_buckets=3)
        v.update(1.0, 1.0)
        v.update(1.1, 35.0)  # fills three buckets in one update
        assert v.buckets_filled == 3
        assert v.value == pytest.approx(1.0)  # tick-rule: all buy


def _engine(**kw) -> tuple[SignalEngine, StrategyParams]:
    base = dict(
        ou_window=400, ou_min_obs=200, ou_max_half_life_ticks=500, z_entry=2.0, z_exit=0.3,
        vpin_bucket_volume=5, vpin_num_buckets=10, vpin_threshold=0.95, signal_cooldown_s=0,
        min_edge_to_spread=0.0,
    )
    params = StrategyParams(**{**base, **kw.pop("params", {})})
    return SignalEngine(params, **kw), params


def _warm(engine: SignalEngine, factory, n: int = 400, seed: int = 5) -> None:
    for i, v in enumerate(simulate_ou(n, 1.0850, 0.9, 0.0002, seed=seed)):
        engine.on_snapshot(factory(price=float(v), ts_ns=i * 1_000_000))


class TestSignalEngine:
    def test_warming_up(self, snapshot_factory) -> None:
        engine, _ = _engine()
        sig = engine.on_snapshot(snapshot_factory())
        assert sig.action is SignalAction.HOLD and sig.reason == "warming_up"

    def test_long_entry_on_negative_z(self, snapshot_factory) -> None:
        engine, _ = _engine()
        _warm(engine, snapshot_factory)
        sig = engine.on_snapshot(snapshot_factory(price=1.0850 - 0.0012, ts_ns=10**12))
        assert sig.action is SignalAction.OPEN_LONG, sig.reason
        assert sig.z is not None and sig.z < -2
        assert sig.take_profit == pytest.approx(sig.mu)
        assert sig.stop_distance > 0
        assert sig.side is Side.BUY

    def test_short_entry_on_positive_z(self, snapshot_factory) -> None:
        engine, _ = _engine()
        _warm(engine, snapshot_factory)
        sig = engine.on_snapshot(snapshot_factory(price=1.0850 + 0.0012, ts_ns=10**12))
        assert sig.action is SignalAction.OPEN_SHORT, sig.reason

    def test_toxic_flow_blocks_entry(self, snapshot_factory) -> None:
        engine, _ = _engine(params={"vpin_threshold": 0.01})
        _warm(engine, snapshot_factory)
        sig = engine.on_snapshot(snapshot_factory(price=1.0838, ts_ns=10**12))
        assert sig.action is SignalAction.HOLD and sig.reason == "toxic_flow"

    def test_sentiment_veto(self, snapshot_factory) -> None:
        engine, _ = _engine(sentiment_provider=lambda s, t: -0.9, params={"sentiment_weight": 0.0})
        _warm(engine, snapshot_factory)
        sig = engine.on_snapshot(snapshot_factory(price=1.0838, ts_ns=10**12))
        assert sig.action is SignalAction.HOLD and sig.reason == "sentiment_veto"
        assert sig.sentiment == pytest.approx(-0.9)

    def test_sentiment_shifts_alpha(self, snapshot_factory) -> None:
        engine, _ = _engine(sentiment_provider=lambda s, t: 0.4, params={"sentiment_weight": 1.0})
        _warm(engine, snapshot_factory)
        sig = engine.on_snapshot(snapshot_factory(price=1.0849, ts_ns=10**12))
        assert sig.alpha == pytest.approx(-sig.z + 0.4)

    def test_blackout_blocks_entry(self, snapshot_factory) -> None:
        engine, _ = _engine(blackout_provider=lambda s, t: "USD:NFP")
        _warm(engine, snapshot_factory)
        sig = engine.on_snapshot(snapshot_factory(price=1.0838, ts_ns=10**12))
        assert sig.reason == "news_blackout:USD:NFP"

    def test_exit_when_mean_reverted(self, snapshot_factory) -> None:
        engine, _ = _engine()
        _warm(engine, snapshot_factory)
        last = engine.last_signal("EURUSDm")
        sig = engine.on_snapshot(snapshot_factory(price=last.mu, ts_ns=10**12), position_side=Side.BUY)
        assert sig.action is SignalAction.CLOSE and sig.reason == "mean_reverted"

    def test_edge_below_cost(self, snapshot_factory) -> None:
        engine, _ = _engine(params={"min_edge_to_spread": 1.5})
        _warm(engine, snapshot_factory)
        sig = engine.on_snapshot(snapshot_factory(price=1.0838, spread=0.01, ts_ns=10**12))
        assert sig.reason == "edge_below_cost"


def test_params_validation() -> None:
    with pytest.raises(ValueError):
        StrategyParams(z_entry=1.0, z_exit=1.5)


def test_only_latest_tick_decides(snapshot_factory) -> None:
    from dataclasses import replace

    engine, _ = _engine()
    snap = snapshot_factory()
    assert engine.on_snapshot(replace(snap, is_latest=False)) is None
    assert engine.on_snapshot(replace(snap, warmup=True, is_latest=False)) is None
    assert engine.on_snapshot(snap) is not None
    assert len(engine._states["EURUSDm"].ou) == 3  # all ticks updated the model


def test_session_gate_and_symbol_params(snapshot_factory) -> None:
    engine, base = _engine(session_provider=lambda s, t: "us_market_closed")
    _warm(engine, snapshot_factory)
    sig = engine.on_snapshot(snapshot_factory(price=1.0838, ts_ns=10**12))
    assert sig.reason == "us_market_closed"
    strict = base.model_copy(update={"z_entry": 50.0, "z_exit": 0.1})
    engine2 = SignalEngine(base, symbol_params={"EURUSDm": strict})
    assert engine2.params_for("EURUSDm").z_entry == 50.0
    assert engine2.params_for("XAUUSDm").z_entry == base.z_entry
