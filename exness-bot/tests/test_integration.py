"""End-to-end: run the full bot against the MT5 simulator for a few seconds."""

from __future__ import annotations

import asyncio

from src.main import TradingBot, build_settings, parse_args
from src.mt5_simulator import MT5Simulator


def test_bot_runs_end_to_end(settings) -> None:
    settings.strategy.ou_min_obs = 60
    settings.strategy.ou_window = 120
    settings.strategy.vpin_num_buckets = 5
    settings.strategy.vpin_bucket_volume = 5
    settings.poll_interval_ms = 2
    sim = MT5Simulator(symbols=settings.symbols, seed=7, tick_interval_s=0.005)
    bot = TradingBot(settings, backend=sim)

    asyncio.run(bot.run(duration_s=3))

    assert bot.snapshots > 50
    assert bot.signal_counts["hold"] > 0
    assert sum(bot.signal_counts.values()) == bot.decisions <= bot.snapshots
    assert bot.risk.current_equity > 0
    state = bot.state()
    assert state["symbols"]["EURUSDm"]["microprice"] is not None
    # flatten_on_exit leaves no open positions
    assert sim.positions_get() == ()


def test_breaker_trip_flattens_positions(settings) -> None:
    sim = MT5Simulator(symbols=settings.symbols, seed=3, tick_interval_s=0.0)
    bot = TradingBot(settings, backend=sim)

    async def run() -> None:
        await bot.client.connect(settings.mt5)
        await bot.orders.market_order("EURUSDm", __import__("src.models", fromlist=["Side"]).Side.BUY, 0.1)
        await bot._refresh_positions()
        assert len(sim.positions_get()) == 1
        bot.risk.trip("latency_breach:order_send=120.0ms")
        await asyncio.sleep(0.2)
        assert sim.positions_get() == ()
        assert bot.risk.stats.n == 1  # flatten P&L recorded
        await bot.client.disconnect()

    asyncio.run(run())


def test_cli_settings() -> None:
    s = build_settings(parse_args(["--simulate", "--no-api", "--no-nlp"]))
    assert s.simulate and not s.dry_run and not s.api_enabled and not s.nlp.enabled
    try:
        build_settings(parse_args(["--live"]))
    except SystemExit as exc:
        assert "--i-understand-the-risk" in str(exc)
    else:
        raise AssertionError("--live without confirmation must exit")


def test_bot_auto_universe_stocks_and_metals(settings) -> None:
    """Auto-discovered universe: all simulated US stocks + metals (+ configured EURUSD)."""
    settings.universe.mode = "auto"
    settings.symbols = ["EURUSDm"]
    for cls in ("us_stock", "metal"):
        settings.strategy_overrides.setdefault(cls, {}).update({"ou_min_obs": 60, "ou_window": 120, "vpin_num_buckets": 5, "vpin_bucket_volume": 5})
    settings.strategy.ou_min_obs, settings.strategy.ou_window = 60, 120
    settings.strategy.vpin_num_buckets = settings.strategy.vpin_bucket_volume = 5
    sim = MT5Simulator(seed=11, tick_interval_s=0.01)
    bot = TradingBot(settings, backend=sim)

    asyncio.run(bot.run(duration_s=4))

    classes = {i.asset_class for i in bot.instruments.values()}
    assert classes == {"us_stock", "metal", "forex"}
    assert "US500m" not in bot.instruments and "BTCUSDm" not in bot.instruments
    assert sum(1 for i in bot.instruments.values() if i.asset_class == "us_stock") == 10
    assert bot.strategy.params_for("AAPLm").z_entry == 2.5  # class override applied
    assert bot.decisions > 100
    st = bot.state()
    assert st["universe"]["metal"] == 4
    assert sim.positions_get() == ()


def test_config_json_autoload_with_bom(tmp_path, monkeypatch) -> None:
    import json as _json

    from src.config import load_settings

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("BOT_CONFIG", raising=False)
    for k in ("MT5_LOGIN", "MT5_PASSWORD", "MT5_SERVER"):
        monkeypatch.delenv(k, raising=False)
    cfg = {"mt5": {"login": 12345678, "password": "s3cret", "server": "Exness-MT5Trial9"}, "symbols": ["EURUSDm"]}
    (tmp_path / "config.json").write_text(_json.dumps(cfg), encoding="utf-8-sig")  # Notepad-style BOM
    s = load_settings()
    assert s.mt5.login == 12345678 and s.mt5.server == "Exness-MT5Trial9"
    assert s.mt5.password.get_secret_value() == "s3cret"
    (tmp_path / "config.json").write_text('{"mt5": {"login": 1,}}', encoding="utf-8")
    try:
        load_settings()
    except ValueError as exc:
        assert "invalid JSON" in str(exc)
    else:
        raise AssertionError("bad JSON must raise a readable error")
