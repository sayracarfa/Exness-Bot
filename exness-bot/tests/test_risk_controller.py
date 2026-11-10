from __future__ import annotations

import asyncio

import numpy as np
import pytest

from src.config import RiskLimits
from src.risk_controller import (
    BreakerState,
    RiskController,
    historical_var,
    kelly_fraction,
    max_drawdown,
    parametric_var,
)


def test_kelly_fraction() -> None:
    assert kelly_fraction(0.6, 1.0) == pytest.approx(0.2)
    assert kelly_fraction(0.5, 2.0) == pytest.approx(0.25)
    assert kelly_fraction(0.4, 1.0) == 0.0  # negative edge => no bet
    assert kelly_fraction(0.6, 0.0) == 0.0
    with pytest.raises(ValueError):
        kelly_fraction(1.5, 1.0)


def test_var_and_drawdown() -> None:
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.01, 20_000)
    assert historical_var(r, 0.99) == pytest.approx(0.0233, abs=0.002)
    assert parametric_var(r, 0.99) == pytest.approx(0.0233, abs=0.001)
    assert historical_var([], 0.99) == 0.0
    assert max_drawdown([100, 120, 90, 130, 104]) == pytest.approx(0.25)


def test_position_size_uses_fractional_kelly_and_cap(eurusd_spec) -> None:
    limits = RiskLimits(prior_win_rate=0.55, prior_payoff_ratio=1.0, kelly_fraction=0.25, max_risk_per_trade=0.05)
    rc = RiskController(limits, vpin_threshold=0.7)
    # f* = 0.1, quarter Kelly = 0.025; equity 10k => risk $250; stop 50 pips => $500/lot => 0.5 lots
    assert rc.kelly_risk_fraction() == pytest.approx(0.025)
    assert rc.position_size(eurusd_spec, 10_000, 0.0050) == pytest.approx(0.5)

    capped = RiskController(RiskLimits(prior_win_rate=0.55, max_risk_per_trade=0.005), 0.7)
    assert capped.position_size(eurusd_spec, 10_000, 0.0050) == pytest.approx(0.1)
    assert capped.position_size(eurusd_spec, 10_000, 0.0) == 0.0
    assert capped.position_size(eurusd_spec, 10, 0.0050) == 0.0  # below volume_min => no trade


def test_kelly_uses_observed_stats_after_min_trades() -> None:
    rc = RiskController(RiskLimits(min_trades_for_kelly=10, max_risk_per_trade=0.05, kelly_fraction=1.0), 0.7)
    for _ in range(7):
        rc.record_trade_result(20.0)
    for _ in range(3):
        rc.record_trade_result(-10.0)
    # p = 0.7, b = 2 => f* = 0.7 - 0.3/2 = 0.55, capped at 0.05
    assert rc.stats.win_rate == pytest.approx(0.7)
    assert rc.stats.payoff_ratio == pytest.approx(2.0)
    assert rc.kelly_risk_fraction() == pytest.approx(0.05)


def test_pre_trade_checks() -> None:
    rc = RiskController(RiskLimits(max_positions_per_symbol=1, max_total_positions=2), vpin_threshold=0.7)
    assert rc.pre_trade_check("X", 0.3, 0, 0) == (True, "ok")
    assert rc.pre_trade_check("X", 0.8, 0, 0)[1].startswith("toxic_flow")
    assert rc.pre_trade_check("X", None, 0, 0)[1] == "vpin_unavailable"
    assert rc.pre_trade_check("X", 0.3, 1, 1)[1] == "max_positions_per_symbol"
    assert rc.pre_trade_check("X", 0.3, 0, 2)[1] == "max_total_positions"


def test_var_limit_blocks() -> None:
    rc = RiskController(RiskLimits(max_var_pct=0.001, min_var_observations=10, max_drawdown_pct=0.9, max_daily_loss_pct=0.9), 0.7)
    eq = 10_000.0
    rng = np.random.default_rng(1)
    for i in range(50):
        eq *= 1 + rng.normal(0, 0.01)
        rc.update_equity(eq, ts=1_700_000_000 + i)
    ok, reason = rc.pre_trade_check("X", 0.1, 0, 0)
    assert not ok and reason.startswith("var_limit")


def test_latency_breaker_trips_and_flattens() -> None:
    calls: list[str] = []

    async def on_trip(reason: str) -> None:
        calls.append(reason)

    async def run() -> RiskController:
        rc = RiskController(RiskLimits(max_latency_ms=50, latency_breach_count=3), 0.7, on_trip=on_trip)
        rc.record_latency("copy_ticks_from", 80)
        rc.record_latency("copy_ticks_from", 10)  # resets the streak
        rc.record_latency("copy_ticks_from", 80)
        rc.record_latency("symbol_info_tick", 90)
        assert not rc.tripped
        rc.record_latency("account_info", 999)  # non-critical op ignored
        rc.record_latency("pipeline", 70)
        await asyncio.sleep(0)  # let the flatten task run
        return rc

    rc = asyncio.run(run())
    assert rc.state is BreakerState.TRIPPED
    assert calls and calls[0].startswith("latency_breach:pipeline")
    ok, reason = rc.pre_trade_check("X", 0.1, 0, 0)
    assert not ok and reason.startswith("breaker_tripped")


def test_instant_latency_mode() -> None:
    rc = RiskController(RiskLimits(latency_breach_count=1), 0.7)
    rc.record_latency("symbol_info_tick", 51)
    assert rc.tripped


def test_error_breaker_and_cooldown_rearm() -> None:
    now = [1000.0]
    rc = RiskController(RiskLimits(max_consecutive_errors=3, breaker_cooldown_s=60), 0.7, clock=lambda: now[0])
    for _ in range(2):
        rc.record_error("io", RuntimeError())
    rc.record_success()
    for _ in range(3):
        rc.record_error("io", RuntimeError())
    assert rc.tripped and not rc.is_hard_trip
    now[0] += 61
    rc.maybe_rearm()
    assert not rc.tripped


def test_drawdown_is_hard_trip() -> None:
    now = [1_700_000_000.0]
    rc = RiskController(RiskLimits(max_drawdown_pct=0.05, max_daily_loss_pct=0.5, breaker_cooldown_s=0), 0.7, clock=lambda: now[0])
    rc.update_equity(10_000)
    rc.update_equity(10_500)
    rc.update_equity(9_900)  # 5.7% below peak
    assert rc.tripped and rc.is_hard_trip
    now[0] += 10_000
    rc.maybe_rearm()
    assert rc.tripped  # hard trips need a manual reset
    rc.reset()
    assert not rc.tripped


def test_daily_loss_trip() -> None:
    rc = RiskController(RiskLimits(max_daily_loss_pct=0.02, max_drawdown_pct=0.5), 0.7)
    rc.update_equity(10_000, ts=1_700_000_000)
    rc.update_equity(9_790, ts=1_700_000_100)
    assert rc.tripped and rc.trip_reason.startswith("daily_loss")


def test_class_limits_and_net_direction() -> None:
    rc = RiskController(RiskLimits(max_total_positions=20, max_positions_per_class={"us_stock": 3},
                                   max_net_direction_per_class={"us_stock": 2}), 0.7)
    ok = lambda **kw: rc.pre_trade_check("AAPLm", 0.2, 0, 0, asset_class="us_stock", **kw)  # noqa: E731
    assert ok(open_class=2, class_net_direction=1, side_sign=-1) == (True, "ok")
    assert ok(open_class=3, class_net_direction=0, side_sign=1)[1] == "max_positions_class:us_stock"
    assert ok(open_class=2, class_net_direction=2, side_sign=1)[1] == "max_net_direction:us_stock"
    assert ok(open_class=2, class_net_direction=2, side_sign=-1)[0]  # hedging trade allowed


def test_cap_by_margin(eurusd_spec) -> None:
    rc = RiskController(RiskLimits(max_margin_usage=0.25), 0.7)
    assert rc.cap_by_margin(eurusd_spec, 1.0, margin_required=200.0, free_margin=10_000) == 1.0
    assert rc.cap_by_margin(eurusd_spec, 1.0, margin_required=5_000.0, free_margin=10_000) == pytest.approx(0.5)
    assert rc.cap_by_margin(eurusd_spec, 1.0, margin_required=None, free_margin=0) == 1.0
    assert rc.cap_by_margin(eurusd_spec, 1.0, margin_required=1e9, free_margin=100) == 0.0


def test_order_latency_has_its_own_limit() -> None:
    rc = RiskController(RiskLimits(max_latency_ms=50, max_order_latency_ms=400, latency_breach_count=1), 0.7)
    rc.record_latency("order_send", 150)  # normal network round trip to the broker
    assert not rc.tripped
    rc.record_latency("copy_ticks_from", 60)  # local call: 50 ms limit
    assert rc.tripped
