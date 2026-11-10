from __future__ import annotations

import asyncio
import math

import numpy as np
import pandas as pd
import pytest

from src.backtest import default_spec, run_backtest, synthetic_ticks
from src.config import PortfolioRiskSettings, RiskLimits, StrategyParams
from src.portfolio_risk import PortfolioRisk


def feed_pair(pr: PortfolioRisk, n: int = 20_000, rho: float = 0.8, seed: int = 0, dt: float = 2.0) -> None:
    rng = np.random.default_rng(seed)
    z1 = rng.standard_normal(n)
    z2 = rho * z1 + math.sqrt(1 - rho * rho) * rng.standard_normal(n)
    pa = 100 * np.exp(np.cumsum(z1 * 1e-4))
    pb = 50 * np.exp(np.cumsum(z2 * 1e-4))
    for i in range(n):
        pr.on_price("A", i * dt, float(pa[i]))
        pr.on_price("B", i * dt, float(pb[i]))


class TestVolTargeting:
    def test_unknown_until_min_bars(self) -> None:
        pr = PortfolioRisk(PortfolioRiskSettings(min_bars=60))
        for i in range(100):
            pr.on_price("A", i * 2.0, 100 + 0.01 * i)
        assert pr.vol("A") is None
        assert pr.vol_multiplier("A") == 0.5  # unknown_vol_multiplier
        assert not pr.vol_spike("A")

    def test_calm_regime_full_size(self) -> None:
        pr = PortfolioRisk(PortfolioRiskSettings())
        feed_pair(pr)
        assert pr.vol_multiplier("A") == pytest.approx(1.0, abs=0.15)

    def test_vol_spike_shrinks_and_blocks(self) -> None:
        pr = PortfolioRisk(PortfolioRiskSettings())
        feed_pair(pr, n=20_000)
        rng = np.random.default_rng(9)
        price = 100.0
        t0 = 20_000 * 2.0
        for i in range(150):  # 5 minutes at 6x volatility (news-like jump)
            price *= math.exp(rng.standard_normal() * 6e-4)
            pr.on_price("A", t0 + i * 2.0, price)
        ratio = pr.vol_ratio("A")
        assert ratio > 2.0
        assert pr.vol_spike("A")
        assert pr.vol_multiplier("A") == pytest.approx(max(0.25, 1 / ratio))

    def test_overnight_gap_does_not_create_fake_vol(self) -> None:
        pr = PortfolioRisk(PortfolioRiskSettings())
        for i in range(2_000):
            pr.on_price("A", i * 2.0, 100.0 * (1 + 1e-5 * math.sin(i)))
        before = pr.vol("A")
        pr.on_price("A", 2_000 * 2.0 + 16 * 3600, 110.0)  # reopens 16 h later, +10%
        pr.on_price("A", 2_000 * 2.0 + 16 * 3600 + 20, 110.0)
        after = pr.vol("A")
        assert before is not None and after is not None
        assert after[0] < 10 * before[0] + 1e-9  # the 10% gap was not counted as a return


class TestCorrelationVaR:
    def test_correlation_estimate(self) -> None:
        pr = PortfolioRisk(PortfolioRiskSettings())
        feed_pair(pr, rho=0.8)
        assert pr.correlation("A", "B") == pytest.approx(0.8, abs=0.1)

    def test_portfolio_var_formula(self) -> None:
        cfg = PortfolioRiskSettings()
        pr = PortfolioRisk(cfg)
        feed_pair(pr)
        cov = pr.covariance(["A", "B"])
        e = np.array([10_000.0, -5_000.0])
        z = 2.3263478740408408
        expected = z * math.sqrt(cfg.var_horizon_s / cfg.bar_seconds * float(e @ cov @ e))
        assert pr.portfolio_var({"A": 10_000.0, "B": -5_000.0}) == pytest.approx(expected, rel=1e-9)
        assert pr.portfolio_var({}) == 0.0

    def test_diversification(self) -> None:
        pr = PortfolioRisk(PortfolioRiskSettings())
        feed_pair(pr, rho=0.8)
        long_long = pr.portfolio_var({"A": 10_000.0, "B": 10_000.0})
        long_short = pr.portfolio_var({"A": 10_000.0, "B": -10_000.0})
        assert long_short < long_long / 2  # offsetting correlated legs

    def test_max_lots_respects_correlation(self) -> None:
        pr = PortfolioRisk(PortfolioRiskSettings())
        feed_pair(pr, rho=0.8)
        alone = pr.max_lots_for_var("B", 50.0, {}, 100.0)
        same = pr.max_lots_for_var("B", 50.0, {"A": 10_000.0}, 100.0)
        hedge = pr.max_lots_for_var("B", -50.0, {"A": 10_000.0}, 100.0)
        assert same < alone < hedge
        var_at_cap = pr.portfolio_var({"A": 10_000.0, "B": same * 50.0})
        assert var_at_cap == pytest.approx(100.0, rel=1e-6)  # exactly on the limit

    def test_over_limit_book_allows_only_risk_reducing(self) -> None:
        pr = PortfolioRisk(PortfolioRiskSettings())
        feed_pair(pr, rho=0.8)
        big = {"A": 60_000.0}
        assert pr.portfolio_var(big) > 100.0
        assert pr.max_lots_for_var("B", 50.0, big, 100.0) == 0.0
        x = pr.max_lots_for_var("B", -50.0, big, 100.0)
        assert x > 0
        assert pr.portfolio_var({"A": 60_000.0, "B": -50.0 * x}) < pr.portfolio_var(big)

    def test_unknown_covariance_returns_none(self) -> None:
        pr = PortfolioRisk(PortfolioRiskSettings())
        pr.on_price("A", 0.0, 100.0)
        assert pr.max_lots_for_var("A", 100.0, {}, 50.0) is None
        assert pr.portfolio_var({"A": 1000.0}) is None


def test_backtest_blocks_entries_in_vol_spike() -> None:
    calm = synthetic_ticks("ou", n=40_000, seed=5)
    wild = synthetic_ticks("ou", n=8_000, seed=6, sd=0.0030)  # 10x volatility regime
    wild["time_msc"] = wild["time_msc"] - wild["time_msc"].iloc[0] + calm["time_msc"].iloc[-1] + 250
    ticks = pd.concat([calm, wild], ignore_index=True)
    res = run_backtest(ticks, default_spec(), StrategyParams(), RiskLimits())
    assert res.metrics["entries_blocked_vol_spike"] > 0
    off = run_backtest(ticks, default_spec(), StrategyParams(), RiskLimits(), portfolio=None)
    assert off.metrics["entries_blocked_vol_spike"] == 0


def test_bot_reports_portfolio_state(settings) -> None:
    from src.main import TradingBot
    from src.mt5_simulator import MT5Simulator

    settings.portfolio.bar_seconds = 0.1
    settings.portfolio.min_bars = 10
    bot = TradingBot(settings, backend=MT5Simulator(symbols=settings.symbols, seed=3, tick_interval_s=0.005))
    asyncio.run(bot.run(duration_s=3))
    st = bot.state()
    assert "EURUSDm" in st["vol_targeting"]
    assert st["vol_targeting"]["EURUSDm"]["vol_ratio"] is not None
    assert st["portfolio_var"] == 0.0  # flat book
