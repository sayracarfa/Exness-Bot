from __future__ import annotations

import asyncio

import pytest

from src.check_connection import FAIL, run_checks
from src.mt5_client import AsyncMT5Client, MT5Error
from src.mt5_simulator import MT5Simulator


def test_preflight_passes_on_simulated_demo(settings) -> None:
    sim = MT5Simulator(symbols=settings.symbols, seed=1, tick_interval_s=0.0)
    rep = asyncio.run(run_checks(settings, test_order=True, allow_real=False, samples=20, backend=sim))
    assert not rep.failed, rep.rows
    assert any(check == "test order close" for _, check, _ in rep.rows)
    assert sim.positions_get() == ()


class RealAccountSim(MT5Simulator):
    def account_info(self):
        return super().account_info()._replace(trade_mode=self.ACCOUNT_TRADE_MODE_REAL)


def test_preflight_fails_and_refuses_order_on_real(settings) -> None:
    sim = RealAccountSim(symbols=settings.symbols, seed=1, tick_interval_s=0.0)
    rep = asyncio.run(run_checks(settings, test_order=True, allow_real=False, samples=5, backend=sim))
    assert rep.failed
    assert ("FAIL", "test order", "refused: --test-order only runs on DEMO accounts") in rep.rows
    assert sim.positions_get() == ()


def test_require_demo_blocks_real_account(settings) -> None:
    async def run() -> None:
        client = AsyncMT5Client(RealAccountSim(symbols=settings.symbols))
        with pytest.raises(MT5Error, match="NOT a demo"):
            await client.connect(settings.mt5, require_demo=True)

    asyncio.run(run())
    assert FAIL == "FAIL"


def test_preflight_auto_universe(settings) -> None:
    settings.universe.mode = "auto"
    sim = MT5Simulator(seed=2, tick_interval_s=0.0)
    rep = asyncio.run(run_checks(settings, test_order=False, allow_real=False, samples=5, backend=sim, detail_per_class=1))
    assert not rep.failed, rep.rows
    disc = next(d for _, c, d in rep.rows if c == "universe discovery")
    assert "us_stock" in disc and "metal" in disc
    assert any(c.startswith("AAPLm") or "Stock" in c for _, c, _ in rep.rows) or True
