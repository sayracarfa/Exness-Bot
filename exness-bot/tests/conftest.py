from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Settings  # noqa: E402
from src.mt5_simulator import MT5Simulator  # noqa: E402


@pytest.fixture
def sim() -> MT5Simulator:
    return MT5Simulator(symbols=["EURUSDm", "XAUUSDm"], seed=42, tick_interval_s=0.0)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    s = Settings()
    s.simulate = True
    s.dry_run = False
    s.api_enabled = False
    s.log_dir = str(tmp_path / "logs")
    s.nlp.synthetic_news = True
    s.nlp.calendar_url = None
    s.symbols = ["EURUSDm"]
    s.universe.mode = "manual"
    s.sessions.enforce = False
    s.warmup_seconds = 0
    return s


def make_snapshot(symbol: str = "EURUSDm", price: float = 1.085, spread: float = 0.00002, ts_ns: int = 0, volume: float = 1.0):
    """Lightweight MarketSnapshot factory."""
    from src.data_ingestion import MarketSnapshot

    bid, ask = price - spread / 2, price + spread / 2
    return MarketSnapshot(
        symbol=symbol, time_msc=ts_ns // 1_000_000, recv_ns=ts_ns, recv_perf_ns=0, bid=bid, ask=ask,
        bid_volume=1.0, ask_volume=1.0, microprice=price, trade_volume=volume, has_depth=True,
    )


@pytest.fixture
def snapshot_factory():
    return make_snapshot


@pytest.fixture
def eurusd_spec():
    from src.models import SymbolSpec

    info = SimpleNamespace(
        name="EURUSDm", digits=5, point=1e-5, trade_tick_size=1e-5, trade_tick_value=1.0,
        trade_contract_size=100_000, volume_min=0.01, volume_max=200.0, volume_step=0.01,
        trade_stops_level=0, filling_mode=1,
    )
    return SymbolSpec.from_info(info)
