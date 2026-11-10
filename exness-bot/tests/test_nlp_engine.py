from __future__ import annotations

import asyncio
import json
import math
import time

import pytest

from src.config import NLPSettings
from src.nlp_engine import (
    LexiconSentimentModel,
    NewsItem,
    NLPEngine,
    SentimentAggregator,
    decay_lambda,
    decayed_sentiment,
    detect_currencies,
    parse_calendar,
    parse_rss,
    symbol_exposure,
)


class FakeModel:
    name = "fake"

    def __init__(self, score: float) -> None:
        self._score = score
        self.calls = 0

    async def score(self, texts):
        self.calls += 1
        return [self._score for _ in texts]


def test_decay_formula() -> None:
    lam = decay_lambda(600)
    assert lam == pytest.approx(math.log(2) / 600)
    assert decayed_sentiment(0.8, lam, t=1000.0, t0=1000.0) == pytest.approx(0.8)
    assert decayed_sentiment(0.8, lam, t=1600.0, t0=1000.0) == pytest.approx(0.4)  # one half-life
    assert decayed_sentiment(0.8, lam, t=999.0, t0=1000.0) == 0.0  # future news has no effect
    with pytest.raises(ValueError):
        decay_lambda(0)


def test_aggregator_sums_clips_and_prunes() -> None:
    agg = SentimentAggregator(half_life_s=100)
    agg.add("EURUSDm", 0.7, 0.0)
    agg.add("EURUSDm", 0.7, 0.0)
    assert agg.value("EURUSDm", now=0.0) == 1.0  # clipped
    assert agg.value("EURUSDm", now=100.0) == pytest.approx(0.7)
    assert agg.value("EURUSDm", now=10_000.0) == pytest.approx(0.0, abs=1e-3)
    assert agg.value("GBPUSDm", now=0.0) == 0.0


def test_currency_detection_and_exposure() -> None:
    assert detect_currencies("Fed's Powell signals hike; ECB holds") == {"USD", "EUR"}
    assert "USD" not in detect_currencies("Tell us about the weather")
    assert "USD" in detect_currencies("US GDP beats estimates")
    assert detect_currencies("Gold jumps to record") == {"XAU"}
    assert symbol_exposure("EURUSDm", "EUR") == 1
    assert symbol_exposure("EURUSDm", "USD") == -1
    assert symbol_exposure("XAUUSDm", "XAU") == 1
    assert symbol_exposure("EURUSDm", "JPY") == 0


def test_lexicon_model() -> None:
    scores = asyncio.run(LexiconSentimentModel().score(["Euro gains on strong data", "Stocks plunge on recession fears", "Meeting today"]))
    assert scores[0] > 0 > scores[1]
    assert scores[2] == 0.0


def test_process_headlines_maps_sign_by_exposure() -> None:
    engine = NLPEngine(NLPSettings(), ["EURUSDm", "XAUUSDm"], FakeModel(0.8), sources=[])
    now = time.time()
    items = [NewsItem("1", "Fed turns hawkish, dollar rallies", now, "t")]
    events = asyncio.run(engine.process_headlines(items, now=now))
    # positive USD news => bearish EURUSD and XAUUSD (USD is the quote)
    assert {e.symbol for e in events} == {"EURUSDm", "XAUUSDm"}
    assert all(e.score0 == pytest.approx(-0.8) for e in events)
    assert engine.sentiment("EURUSDm", now) == pytest.approx(-0.8)
    # duplicates are ignored
    assert asyncio.run(engine.process_headlines(items, now=now)) == []


def test_old_news_ignored() -> None:
    model = FakeModel(0.5)
    engine = NLPEngine(NLPSettings(max_news_age_s=60), ["EURUSDm"], model, sources=[])
    now = time.time()
    events = asyncio.run(engine.process_headlines([NewsItem("x", "ECB hikes", now - 3600, "t")], now=now))
    assert events == [] and model.calls == 0


def test_poll_once_survives_failing_source() -> None:
    async def bad():
        raise ConnectionError("down")

    async def good():
        return [NewsItem("g", "Euro surges", time.time(), "t")]

    engine = NLPEngine(NLPSettings(), ["EURUSDm"], FakeModel(0.6), sources=[bad, good])
    events = asyncio.run(engine.poll_once())
    assert len(events) == 1 and events[0].score0 == pytest.approx(0.6)


def test_parse_rss_and_atom() -> None:
    rss = b"""<rss><channel><item><title>Euro gains</title><guid>a1</guid>
    <pubDate>Mon, 02 Nov 2026 10:00:00 GMT</pubDate></item></channel></rss>"""
    items = parse_rss(rss, "src")
    assert len(items) == 1 and items[0].headline == "Euro gains"
    assert items[0].published_ts == pytest.approx(1793613600.0)
    atom = b"""<feed xmlns="http://www.w3.org/2005/Atom"><entry><title>Gold slumps</title>
    <id>z</id><updated>2026-11-02T10:00:00Z</updated></entry></feed>"""
    assert parse_rss(atom, "src")[0].headline == "Gold slumps"


def test_calendar_blackout() -> None:
    ts = time.time() + 120
    payload = json.dumps([
        {"title": "Non-Farm Employment Change", "country": "USD", "date": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(ts)), "impact": "High"},
        {"title": "Bank Holiday", "country": "JPY", "date": "2026-11-03T00:00:00+00:00", "impact": "Holiday"},
    ]).encode()
    engine = NLPEngine(NLPSettings(blackout_before_s=300), ["EURUSDm"], FakeModel(0), sources=[])
    engine.calendar = parse_calendar(payload)
    assert len(engine.calendar) == 2
    assert engine.blackout_reason("EURUSDm") == "USD:Non-Farm Employment Change"
    assert engine.blackout_reason("EURUSDm", now=ts + 3600) is None
    assert engine.blackout_reason("GBPJPYm") is None  # JPY event is not high impact
