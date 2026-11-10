from __future__ import annotations

import asyncio
import time
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from src.config import NLPSettings, SessionSettings, Settings
from src.mt5_simulator import MT5Simulator
from src.nlp_engine import NLPEngine, instrument_exposure
from src.sessions import SessionCalendar
from src.universe import (
    FOREX, METAL, OTHER, US_STOCK, build_stock_patterns, build_universe, classify, clean_company_name, make_instrument, ticker_of,
)


def info(name, path="", desc="", ccy="USD", exchange="", trade_mode=4):
    return SimpleNamespace(name=name, path=path, description=desc, currency_profit=ccy, exchange=exchange, trade_mode=trade_mode)


class TestClassification:
    def test_ticker_suffixes(self) -> None:
        assert ticker_of("AAPLm") == "AAPL"
        assert ticker_of("AAPL") == "AAPL"
        assert ticker_of("BRK.Bm") == "BRK.B"
        assert ticker_of("XAUUSDm") == "XAUUSD"

    @pytest.mark.parametrize("sym,path,exch,ccy,expected", [
        ("XAUUSDm", "Metals\\XAUUSDm", "", "USD", METAL),
        ("XAGEURm", "Metals", "", "EUR", METAL),
        ("EURUSDm", "Forex\\Majors", "", "USD", FOREX),
        ("AAPLm", "Stocks\\US\\AAPLm", "NASDAQ", "USD", US_STOCK),
        ("KOm", "Shares\\NYSE", "", "USD", US_STOCK),
        ("SAPm", "Stocks\\EU", "XETRA", "EUR", OTHER),       # non-USD
        ("BABAm", "Stocks", "HKEX", "USD", OTHER),           # non-US exchange
        ("US500m", "Indices\\US500m", "", "USD", OTHER),
        ("BTCUSDm", "Crypto\\BTCUSDm", "", "USD", OTHER),
    ])
    def test_classify(self, sym, path, exch, ccy, expected) -> None:
        assert classify(info(sym, path, ccy=ccy, exchange=exch), Settings().universe) == expected

    def test_company_name(self) -> None:
        assert clean_company_name("Apple Inc.") == "Apple"
        assert clean_company_name("Alphabet Inc. Class A") == "Alphabet"
        assert clean_company_name("The Coca-Cola Company") == "Coca-Cola"
        assert clean_company_name("JPMorgan Chase & Co.") == "JPMorgan Chase"

    def test_build_universe_filters(self) -> None:
        s = Settings()
        s.symbols = ["EURUSDm", "MISSINGm"]
        s.universe.exclude = ["TSLAm"]
        infos = [
            info("AAPLm", "Stocks\\US", "Apple Inc."), info("TSLAm", "Stocks\\US", "Tesla Inc."),
            info("OLDm", "Stocks\\US", "Delisted Corp", trade_mode=3),  # close-only
            info("XAGUSDm", "Metals"), info("EURUSDm", "Forex"), info("US30m", "Indices"),
        ]
        u = build_universe(infos, s)
        assert set(u) == {"AAPLm", "XAGUSDm", "EURUSDm"}
        assert u["AAPLm"].calendar_currencies == ("USD",)
        assert u["XAGUSDm"].calendar_currencies == ("XAG", "USD")

    def test_simulator_universe(self) -> None:
        sim = MT5Simulator(seed=1)
        u = build_universe(sim.symbols_get(), Settings())
        assert sum(i.asset_class == US_STOCK for i in u.values()) == 10
        assert sum(i.asset_class == METAL for i in u.values()) == 4


class TestStockNews:
    def _match(self, ticker, desc, text):
        return any(p.search(text) for p in build_stock_patterns(ticker, desc))

    def test_ticker_name_alias(self) -> None:
        assert self._match("AAPL", "Apple Inc.", "AAPL rises premarket")
        assert self._match("AAPL", "Apple Inc.", "iPhone demand surges in China")
        assert self._match("NVDA", "NVIDIA Corporation", "Nvidia beats estimates")
        assert self._match("GOOGL", "Alphabet Inc. Class A", "Google faces antitrust ruling")
        assert not self._match("AAPL", "Apple Inc.", "Microsoft launches new Surface")

    def test_ambiguous_tickers_need_cashtag(self) -> None:
        assert not self._match("V", "Visa Inc.", "V-shaped recovery expected")
        assert self._match("V", "Visa Inc.", "Visa reports record volumes")
        assert self._match("V", "Visa Inc.", "Shares of $V climb")
        assert not self._match("KO", "The Coca-Cola Company", "Boxer wins by KO")
        assert self._match("KO", "The Coca-Cola Company", "Coca-Cola raises dividend")

    def test_exposure_and_market_beta(self) -> None:
        aapl = make_instrument(info("AAPLm", "Stocks\\US", "Apple Inc."), US_STOCK)
        msft = make_instrument(info("MSFTm", "Stocks\\US", "Microsoft Corporation"), US_STOCK)
        assert instrument_exposure(aapl, "Apple jumps", set(), 0.4) == 1.0
        assert instrument_exposure(msft, "Apple jumps", set(), 0.4) == 0.0
        assert instrument_exposure(msft, "Wall Street rallies", set(), 0.4) == pytest.approx(0.4)
        assert instrument_exposure(aapl, "Apple leads Wall Street higher", set(), 0.4) == pytest.approx(1.4)

    def test_engine_routes_stock_and_silver_news(self) -> None:
        class Model:
            name = "fake"

            async def score(self, texts):
                return [0.8 for _ in texts]

        s = Settings()
        u = build_universe([info("AAPLm", "Stocks\\US", "Apple Inc."), info("MSFTm", "Stocks\\US", "Microsoft Corporation"),
                            info("XAGUSDm", "Metals")], s)
        eng = NLPEngine(NLPSettings(), u, Model(), sources=[])
        from src.nlp_engine import NewsItem
        now = time.time()
        asyncio.run(eng.process_headlines([NewsItem("1", "Apple shares surge", now), NewsItem("2", "Silver rallies", now)], now=now))
        assert eng.sentiment("AAPLm", now) == pytest.approx(0.8)
        assert eng.sentiment("MSFTm", now) == 0.0
        assert eng.sentiment("XAGUSDm", now) == pytest.approx(0.8)

    def test_stock_feed_batches(self) -> None:
        s = Settings()
        infos = [info(f"T{i:02d}AAm", "Stocks\\US", f"Company {i}") for i in range(45)]
        u = build_universe(infos, s)
        eng = NLPEngine(NLPSettings(rss_feeds=[], stock_feed_batch_size=20), u, object(), sources=None)
        assert len(eng.sources) == 3  # 45 tickers / 20 per feed


class TestSessions:
    NY = ZoneInfo("America/New_York")

    def _cal(self, **kw) -> SessionCalendar:
        u = build_universe([info("AAPLm", "Stocks\\US", "Apple Inc."), info("EURUSDm", "Forex")], Settings())
        return SessionCalendar(SessionSettings(**kw), u)

    def ts(self, *args) -> float:
        return datetime(*args, tzinfo=self.NY).timestamp()

    def test_regular_session(self) -> None:
        cal = self._cal()
        assert cal.entry_block_reason("AAPLm", self.ts(2026, 11, 3, 9, 32)) == "us_pre_open"
        assert cal.entry_block_reason("AAPLm", self.ts(2026, 11, 3, 11, 0)) is None
        assert cal.entry_block_reason("AAPLm", self.ts(2026, 11, 3, 15, 45)) == "us_near_close"
        assert cal.entry_block_reason("AAPLm", self.ts(2026, 11, 3, 20, 0)) == "us_near_close"
        assert cal.entry_block_reason("AAPLm", self.ts(2026, 11, 7, 11, 0)) == "us_market_closed"  # Saturday
        assert cal.entry_block_reason("EURUSDm", self.ts(2026, 11, 7, 11, 0)) is None  # FX unaffected

    def test_flatten_window_and_early_close(self) -> None:
        cal = self._cal()
        assert cal.must_flatten("AAPLm", self.ts(2026, 11, 3, 15, 50))
        assert not cal.must_flatten("AAPLm", self.ts(2026, 11, 3, 15, 40))
        assert not cal.must_flatten("EURUSDm", self.ts(2026, 11, 3, 15, 50))
        assert cal.must_flatten("AAPLm", self.ts(2026, 11, 27, 12, 50))  # day after Thanksgiving: 13:00 close
        assert cal.entry_block_reason("AAPLm", self.ts(2026, 11, 26, 11, 0)) == "us_market_closed"  # Thanksgiving

    def test_dst_handled(self) -> None:
        cal = self._cal()
        # 14:30 UTC is 09:30 NY in winter (EST) but 10:30 NY in summer (EDT)
        winter = datetime(2026, 12, 1, 14, 36, tzinfo=ZoneInfo("UTC")).timestamp()
        summer = datetime(2026, 7, 1, 13, 36, tzinfo=ZoneInfo("UTC")).timestamp()
        assert cal.entry_block_reason("AAPLm", winter) is None
        assert cal.entry_block_reason("AAPLm", summer) is None

    def test_disabled(self) -> None:
        assert self._cal(enforce=False).entry_block_reason("AAPLm", self.ts(2026, 11, 7, 3, 0)) is None
