"""Daily pairs scan: swaps, rollover calendar, bar width, fetch and pre-registered screening."""
from __future__ import annotations

import asyncio
import datetime as dt
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.backtest import fetch_mt5_bars, load_spec
from src.models import SymbolSpec
from src.pair_scan import (DAILY_PARAMS, FINAL_RULE, PAIRS, SCREEN_RULE, load_pairs, main, passes, scan_pair,
                           symbol_names, synthetic_daily_rates)
from src.pairs import _load_rates_leg, build_pair_bars_from_rates, infer_bar_seconds, run_pairs_backtest


def _day(y: int, m: int, d: int) -> int:
    return (dt.date(y, m, d) - dt.date(1970, 1, 1)).days


def _spec(**kw) -> SymbolSpec:
    base = SymbolSpec("T", 2, 0.01, 0.01, 1.0, 100, 0.01, 100, 0.01)
    return replace(base, **kw)


# ---------------------------------------------------------------- swap maths
def test_rollover_nights_wednesday_triple_and_free_weekend():
    s = _spec(swap_rollover3days=3)
    mon = _day(2026, 11, 2)
    assert [s.rollover_nights(mon + i, mon + i + 1) for i in range(7)] == [1, 1, 3, 1, 1, 0, 0]
    assert s.rollover_nights(mon + 4, mon + 7) == 1  # Friday -> Monday: one ordinary night
    assert s.rollover_nights(mon, mon) == 0


def test_rollover_triple_day_configurable():
    s = _spec(swap_rollover3days=5)  # some stock CFDs charge the weekend on Friday
    mon = _day(2026, 11, 2)
    assert s.rollover_nights(mon, mon + 7) == 7
    assert s.rollover_nights(mon + 4, mon + 5) == 3


@pytest.mark.parametrize("mode,expected", [(0, 0.0), (1, -5.0 * 0.01 * 100 * 2), (4, -5.0 * 2),
                                           (5, 200.0 * 100 * 2 * -5.0 / 100 / 360)])
def test_swap_per_night_modes(mode, expected):
    s = _spec(swap_long=-5.0, swap_short=1.0, swap_mode=mode)
    assert s.swap_per_night(True, 2.0, 200.0) == pytest.approx(expected)


def test_swap_short_side_uses_short_rate():
    s = _spec(swap_long=-5.0, swap_short=1.5, swap_mode=1)
    assert s.swap_per_night(False, 1.0, 10.0) == pytest.approx(1.5 * 0.01 * 100)


def test_from_info_reads_swaps():
    info = SimpleNamespace(name="XAUUSDm", digits=3, point=0.001, trade_tick_size=0.001, trade_tick_value=0.1,
                           trade_contract_size=100, volume_min=0.01, volume_max=200, volume_step=0.01,
                           swap_long=-40.5, swap_short=10.2, swap_mode=1, swap_rollover3days=3)
    s = SymbolSpec.from_info(info)
    assert (s.swap_long, s.swap_short, s.swap_mode, s.swap_rollover3days) == (-40.5, 10.2, 1, 3)


# ---------------------------------------------------------------- bars
def test_daily_bars_and_bar_width():
    ry, rx, sy, sx = synthetic_daily_rates("coint", n_days=300)
    assert infer_bar_seconds(ry) == 86_400
    bars = build_pair_bars_from_rates(ry, sy, 0.0, rx, sx, 0.0)
    assert len(bars) == 300
    assert (np.diff(bars["t_ms"]) >= 86_400_000).all()
    # t_ms is the bar close: one day after the bar's own timestamp
    assert bars["t_ms"].iloc[0] == (ry["time"].iloc[0] + 86_400) * 1000


def test_swap_accrues_while_position_open():
    ry, rx, sy, sx = synthetic_daily_rates("coint", n_days=1200, seed=1)
    bars = build_pair_bars_from_rates(ry, sy, 0.0, rx, sx, 0.0)
    p = replace(DAILY_PARAMS, window=250, z_entry=1.5)
    with_swap = run_pairs_backtest(bars, sy, sx, p)
    no_swap = run_pairs_backtest(bars, sy, sx, replace(p, swap_mult=0.0))
    assert with_swap.metrics["trades"] > 0
    assert with_swap.metrics["swap_total"] < 0  # both synthetic rates are negative except short-x
    assert no_swap.metrics["swap_total"] == 0
    assert sum(t.pnl for t in with_swap.trades) < sum(t.pnl for t in no_swap.trades)


def test_intraday_bars_charge_swap_only_at_day_change():
    s = _spec(swap_long=-10.0, swap_short=-10.0, swap_mode=1)
    t0 = _day(2026, 11, 2) * 86_400  # Monday 00:00
    times = t0 + np.arange(0, 3 * 86_400, 3600)  # three days of H1 bars
    r = pd.DataFrame({"time": times, "open": 100.0, "close": 100.0, "spread": 1})
    bars = build_pair_bars_from_rates(r, s, 0.0, r, s, 0.0)
    assert infer_bar_seconds(r) == 3600 and len(bars) == 72


# ---------------------------------------------------------------- rules
def test_pass_rules():
    assert passes({"trades": 30, "profit_factor": 1.3, "t_stat": 2.1, "net_pnl": 5}, FINAL_RULE)
    assert not passes({"trades": 29, "profit_factor": 3, "t_stat": 5, "net_pnl": 5}, FINAL_RULE)
    assert not passes({"trades": 40, "profit_factor": 1.1, "t_stat": 5, "net_pnl": 5}, FINAL_RULE)
    assert not passes({"trades": 40, "profit_factor": 2, "t_stat": 1.9, "net_pnl": 5}, FINAL_RULE)
    assert passes({"trades": 8, "profit_factor": None, "t_stat": 1.5, "net_pnl": 5}, SCREEN_RULE)


def test_pre_registered_pair_list_is_fixed():
    assert len(PAIRS) == 16
    assert ("XAUUSD", "XAGUSD") in PAIRS and ("MCD", "YUM") in PAIRS
    assert "XAUUSDm" in symbol_names("m")


# ---------------------------------------------------------------- scanner power / false positives
def test_scanner_selects_cointegrated_and_rejects_independent():
    sel = rej = 0
    for seed in range(4):
        for kind in ("coint", "indep"):
            ry, rx, sy, sx = synthetic_daily_rates(kind, seed=seed)
            r = scan_pair(build_pair_bars_from_rates(ry, sy, 0.0, rx, sx, 0.0), sy, sx)
            if kind == "coint":
                sel += r["screen_pass"]
            else:
                rej += not r["screen_pass"]
    assert rej == 4
    assert sel >= 2


def test_scan_insufficient_data():
    ry, rx, sy, sx = synthetic_daily_rates("coint", n_days=200)
    r = scan_pair(build_pair_bars_from_rates(ry, sy, 0.0, rx, sx, 0.0), sy, sx)
    assert r["status"] == "insufficient_data" and not r["screen_pass"]


# ---------------------------------------------------------------- files / CLI
def _write_leg(dir_, sym, rates, spec, tick_median):
    rates.to_csv(dir_ / f"{sym}_D1.csv", index=False)
    from dataclasses import asdict
    (dir_ / f"{sym}.spec.json").write_text(json.dumps({**asdict(replace(spec, name=sym,
                                                                        tick_spread_median=tick_median))}))


def test_load_pairs_uses_tick_median_floor(tmp_path):
    ry, rx, sy, sx = synthetic_daily_rates("coint", n_days=600)
    _write_leg(tmp_path, "XAUUSDm", ry, sy, 0.5)
    _write_leg(tmp_path, "XAGUSDm", rx, sx, 0.0)
    data, missing = load_pairs(tmp_path, "m")
    assert list(data) == ["XAUUSD/XAGUSD"] and len(missing) == 15
    bars, *_ , info = data["XAUUSD/XAGUSD"]
    assert info["spread_floor"][0] == pytest.approx(1.0)  # 2 x 0.5
    assert (bars["y_ask"] - bars["y_bid"]).min() == pytest.approx(1.0)
    _, _, floor, _ = _load_rates_leg(str(tmp_path / "XAGUSDm_D1.csv"), 2.0)
    assert floor == 0.0  # no measurement -> bar spread only


def test_cli_run(tmp_path, capsys):
    ry, rx, sy, sx = synthetic_daily_rates("coint", n_days=900)
    _write_leg(tmp_path, "KOm", ry, sy, 0.02)
    _write_leg(tmp_path, "PEPm", rx, sx, 0.02)
    rep = tmp_path / "rep.json"
    assert main(["run", "--data", str(tmp_path), "--report", str(rep)]) == 0
    out = capsys.readouterr().out
    assert "KO/PEP" in out and "VERDICT" in out
    assert json.loads(rep.read_text())["verdict"] in ("PASS", "FAIL")


def test_cli_no_data(tmp_path):
    assert main(["run", "--data", str(tmp_path), "--report", str(tmp_path / "r.json")]) == 2


class _FakeClient:
    """Minimal async client: D1 rates + ticks for spread measurement."""

    def __init__(self):
        self.calls: list[tuple] = []

    def const(self, name):
        return {"TIMEFRAME_D1": 16408, "COPY_TICKS_ALL": -1}[name]

    async def call(self, fn, *args):
        self.calls.append((fn, *args))
        if fn == "symbol_select":
            return True
        if fn == "symbol_info":
            return SimpleNamespace(name="KOm", digits=2, point=0.01, trade_tick_size=0.01, trade_tick_value=0.01,
                                   trade_contract_size=1, volume_min=1, volume_max=1000, volume_step=1,
                                   swap_long=-0.5, swap_short=-0.2, swap_mode=1, swap_rollover3days=5,
                                   path="Stocks\\KO", description="Coca-Cola")
        if fn == "symbol_info_tick":
            return SimpleNamespace(time_msc=1_762_000_000_000)
        if fn == "copy_rates_range":
            _, tf, start, end = args
            t = np.arange(start - start % 86_400, end, 86_400)
            return pd.DataFrame({"time": t, "open": 60.0, "high": 61.0, "low": 59.0, "close": 60.5,
                                 "tick_volume": 1, "spread": 3, "real_volume": 0}).to_records(index=False)
        if fn == "copy_rates_from_pos":
            return None
        if fn == "copy_ticks_range":
            return pd.DataFrame({"bid": [60.0, 60.0, 60.0], "ask": [60.02, 60.03, 60.04]}).to_records(index=False)
        raise AssertionError(fn)

    async def last_error(self):
        return 0, "ok"


def test_fetch_daily_bars_writes_swaps_and_tick_median(tmp_path):
    client = _FakeClient()
    data_path, spec_path = asyncio.run(fetch_mt5_bars("KOm", 400, tmp_path, timeframe="D1", client=client,
                                                         wait_s=0))
    assert data_path.name == "KOm_D1.csv"
    df = pd.read_csv(data_path)
    assert len(df) >= 399 and list(df.columns) == ["time", "open", "close", "spread"]
    spec = load_spec(spec_path)
    assert spec.swap_long == -0.5 and spec.swap_rollover3days == 5
    assert spec.tick_spread_median == pytest.approx(0.03)
    assert any(c[0] == "copy_rates_range" and c[2] == 16408 for c in client.calls)


class _LazyClient(_FakeClient):
    """Terminal that returns 1 bar until history has 'loaded' (third request), like real MT5."""

    def __init__(self, never: bool = False):
        super().__init__()
        self.rate_calls, self.never = 0, never

    async def call(self, fn, *args):
        if fn == "copy_rates_from_pos":
            self.rate_calls += 1
            n = 1 if (self.never or self.rate_calls < 3) else 300
            t = 1_762_000_000 - np.arange(n)[::-1] * 86_400
            return pd.DataFrame({"time": t, "open": 1.0, "close": 1.0, "spread": 2}).to_records(index=False)
        if fn == "copy_rates_range":
            return None
        if fn == "copy_ticks_range" and self.rate_calls < 3:
            return None
        return await super().call(fn, *args)


def test_fetch_waits_for_lazy_history(tmp_path):
    client = _LazyClient()
    path, spec_path = asyncio.run(fetch_mt5_bars("KOm", 400, tmp_path, timeframe="D1", client=client,
                                                 wait_s=0, min_bars=250))
    assert len(pd.read_csv(path)) == 300 and client.rate_calls >= 3
    assert load_spec(spec_path).tick_spread_median > 0


def test_fetch_all_reports_and_cleans_failures(tmp_path, monkeypatch):
    from src import pair_scan
    (tmp_path / "KOm_D1.csv").write_text("time,open,close,spread\n1,1,1,1\n")  # stale 1-bar file

    class _Conn(_LazyClient):
        async def connect(self, *a, **k):
            return None

        async def disconnect(self):
            return None

    monkeypatch.setattr("src.mt5_client.AsyncMT5Client", lambda backend: _Conn(never=True))
    monkeypatch.setattr("src.mt5_client.load_backend", lambda simulate: None)
    failed = asyncio.run(pair_scan.fetch_all(tmp_path, "m", days=400, only=["KOm"], wait_s=0))
    assert failed == ["KOm"]
    assert not (tmp_path / "KOm_D1.csv").exists()
