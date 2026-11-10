"""Asynchronous news / economic-calendar ingestion and FinBERT sentiment scoring.

Each headline gets a score S0 = P(positive) - P(negative) in [-1, 1] at time t0.
It is mapped onto every traded symbol through currency exposure (base +1, quote -1)
and its influence decays exponentially:

    S_decayed(t) = S0 * exp(-lambda * (t - t0)),   lambda = ln(2) / half_life

The symbol's sentiment is the clipped sum of all live decayed contributions.
High-impact calendar events additionally create "blackout" windows in which new
entries are blocked.

Note: headline polarity is a proxy for currency direction, not a perfect mapping.
"""

from __future__ import annotations

import asyncio
import email.utils
import hashlib
import json
import math
import random
import re
import time
import urllib.request
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Awaitable, Callable, Deque, Optional, Protocol, Sequence

import pandas as pd
from loguru import logger

from src.config import NLPSettings
from src.universe import FOREX, MARKET_PATTERNS, METAL, METALS, OTHER, US_STOCK, Instrument

NewsSource = Callable[[], Awaitable[list["NewsItem"]]]


# --------------------------------------------------------------------- data
@dataclass(frozen=True, slots=True)
class NewsItem:
    id: str
    headline: str
    published_ts: float
    source: str = ""


@dataclass(frozen=True, slots=True)
class CalendarEvent:
    title: str
    currency: str
    impact: str
    ts: float


@dataclass(frozen=True, slots=True)
class SentimentEvent:
    symbol: str
    score0: float
    t0: float
    headline: str


# --------------------------------------------------------------- math core
def decay_lambda(half_life_s: float) -> float:
    if half_life_s <= 0:
        raise ValueError("half_life_s must be positive")
    return math.log(2.0) / half_life_s


def decayed_sentiment(s0: float, lam: float, t: float, t0: float) -> float:
    """S_decayed(t) = S0 * exp(-lambda * (t - t0)); zero before the news arrives."""
    if t < t0:
        return 0.0
    return s0 * math.exp(-lam * (t - t0))


class SentimentAggregator:
    """Per-symbol sum of exponentially decaying sentiment impulses."""

    def __init__(self, half_life_s: float, max_events: int = 1_000, prune_below: float = 1e-3) -> None:
        self.lam = decay_lambda(half_life_s)
        self._events: dict[str, Deque[tuple[float, float]]] = {}
        self._max_events = max_events
        self._prune_below = prune_below

    def add(self, symbol: str, score0: float, t0: float) -> None:
        self._events.setdefault(symbol, deque(maxlen=self._max_events)).append((score0, t0))

    def value(self, symbol: str, now: Optional[float] = None) -> float:
        events = self._events.get(symbol)
        if not events:
            return 0.0
        t = time.time() if now is None else now
        total = 0.0
        keep: Deque[tuple[float, float]] = deque(maxlen=self._max_events)
        for s0, t0 in events:
            contrib = decayed_sentiment(s0, self.lam, t, t0)
            if t < t0 or abs(contrib) >= self._prune_below:
                keep.append((s0, t0))
            total += contrib
        self._events[symbol] = keep
        return max(-1.0, min(1.0, total))


# ------------------------------------------------------- currency mapping
_CURRENCY_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "USD": [
        re.compile(r"\b(fed|fomc|powell|dollar|greenback|treasur(y|ies)|nonfarm|payrolls|nfp|wall street)\b", re.I),
        re.compile(r"\bU\.?S\.?A?\b"),  # case-sensitive: avoid the pronoun "us"
    ],
    "EUR": [re.compile(r"\b(ecb|lagarde|euro|eurozone|euro area|bund)\b", re.I)],
    "GBP": [re.compile(r"\b(boe|bank of england|bailey|sterling|pound|gilt)\b", re.I), re.compile(r"\bUK\b")],
    "JPY": [re.compile(r"\b(boj|bank of japan|ueda|yen|japan(ese)?)\b", re.I)],
    "CHF": [re.compile(r"\b(snb|swiss|franc)\b", re.I)],
    "AUD": [re.compile(r"\b(rba|aussie|australia(n)?)\b", re.I)],
    "CAD": [re.compile(r"\b(boc|bank of canada|loonie|canad(a|ian))\b", re.I)],
    "XAU": [re.compile(r"\b(gold|bullion)\b", re.I)],
    "XAG": [re.compile(r"\bsilver\b", re.I)],
    "XPT": [re.compile(r"\bplatinum\b", re.I)],
    "XPD": [re.compile(r"\bpalladium\b", re.I)],
}

_SYMBOL_RE = re.compile(r"^([A-Z]{3})([A-Z]{3})")


def detect_currencies(text: str) -> set[str]:
    return {ccy for ccy, pats in _CURRENCY_PATTERNS.items() if any(p.search(text) for p in pats)}


def symbol_exposure(symbol: str, currency: str) -> int:
    """+1 if the currency is the base, -1 if the quote, 0 otherwise (handles broker suffixes)."""
    match = _SYMBOL_RE.match(symbol.upper())
    if not match:
        return 0
    base, quote = match.groups()
    if currency == base:
        return 1
    if currency == quote:
        return -1
    return 0


def instrument_exposure(inst: Instrument, text: str, currencies: set[str], market_beta: float) -> float:
    """Signed sensitivity of an instrument to a headline.

    FX / metals: +1 per base-currency mention, -1 per quote-currency mention.
    US stocks:   +1 if the company (ticker / name / alias) is mentioned,
                 +market_beta for broad US equity-market headlines.
    """
    if inst.asset_class == US_STOCK:
        exposure = 1.0 if inst.matches_own(text) else 0.0
        if market_beta > 0 and any(p.search(text) for p in MARKET_PATTERNS):
            exposure += market_beta
        return exposure
    return float(sum(symbol_exposure(inst.symbol, c) for c in currencies))


def symbol_currencies(symbol: str) -> tuple[str, ...]:
    match = _SYMBOL_RE.match(symbol.upper())
    return match.groups() if match else ()


# --------------------------------------------------------- sentiment models
class SentimentModel(Protocol):
    name: str

    async def score(self, texts: Sequence[str]) -> list[float]: ...


class LexiconSentimentModel:
    """Dependency-free fallback scorer (Loughran-McDonald style keyword lists)."""

    name = "lexicon"
    _POS = {
        "beat", "beats", "surge", "surges", "rally", "rallies", "gain", "gains", "rise", "rises", "strong",
        "stronger", "growth", "upbeat", "hawkish", "optimism", "record", "boost", "boosts", "jump", "jumps",
        "improve", "improves", "improved", "expansion", "robust", "upgrade", "soar", "soars", "higher",
    }
    _NEG = {
        "miss", "misses", "plunge", "plunges", "fall", "falls", "drop", "drops", "weak", "weaker", "recession",
        "dovish", "fear", "fears", "crisis", "slump", "slumps", "cut", "cuts", "decline", "declines", "contraction",
        "downgrade", "loss", "losses", "lower", "tumble", "tumbles", "default", "sell-off", "selloff",
    }
    _TOKEN = re.compile(r"[a-z\-]+")

    async def score(self, texts: Sequence[str]) -> list[float]:
        out: list[float] = []
        for text in texts:
            tokens = self._TOKEN.findall(text.lower())
            pos = sum(t in self._POS for t in tokens)
            neg = sum(t in self._NEG for t in tokens)
            out.append(0.0 if pos + neg == 0 else (pos - neg) / (pos + neg))
        return out


class FinBertSentimentModel:
    """ProsusAI/FinBERT via HuggingFace transformers, run off the event loop."""

    name = "finbert"

    def __init__(self, model_name: str = "ProsusAI/finbert", batch_size: int = 16) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._pipe = None
        self._load_lock = asyncio.Lock()

    def _load_sync(self) -> None:
        from transformers import pipeline  # heavy import; deferred

        self._pipe = pipeline(
            "text-classification",
            model=self.model_name,
            tokenizer=self.model_name,
            top_k=None,
            truncation=True,
        )
        logger.info("FinBERT model loaded: {}", self.model_name)

    def _score_sync(self, texts: list[str]) -> list[float]:
        assert self._pipe is not None
        outputs = self._pipe(texts, batch_size=self.batch_size)
        scores: list[float] = []
        for out in outputs:
            rows = out if isinstance(out, list) else [out]
            probs = {str(r["label"]).lower(): float(r["score"]) for r in rows}
            scores.append(probs.get("positive", 0.0) - probs.get("negative", 0.0))
        return scores

    async def score(self, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        loop = asyncio.get_running_loop()
        if self._pipe is None:
            async with self._load_lock:
                if self._pipe is None:
                    await loop.run_in_executor(None, self._load_sync)
        return await loop.run_in_executor(None, self._score_sync, list(texts))


def build_sentiment_model(use_finbert: bool, model_name: str) -> SentimentModel:
    if use_finbert:
        try:
            import transformers  # noqa: F401
            import torch  # noqa: F401

            return FinBertSentimentModel(model_name)
        except ImportError:
            logger.warning("transformers/torch not installed -> using lexicon sentiment fallback")
    return LexiconSentimentModel()


# ------------------------------------------------------------- fetchers
def _http_get(url: str, timeout: float) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (exness-hft-bot)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - URLs come from config
        return resp.read(5_000_000)


def _parse_ts(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    value = value.strip()
    try:
        return email.utils.parsedate_to_datetime(value).timestamp()
    except (TypeError, ValueError):
        pass
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        return None


def parse_rss(payload: bytes, source: str) -> list[NewsItem]:
    """Parse RSS 2.0 or Atom feeds into :class:`NewsItem` objects."""
    root = ET.fromstring(payload)
    items: list[NewsItem] = []
    now = time.time()
    atom = "{http://www.w3.org/2005/Atom}"
    for node in list(root.iter("item")) + list(root.iter(f"{atom}entry")):
        title = (node.findtext("title") or node.findtext(f"{atom}title") or "").strip()
        if not title:
            continue
        ts = _parse_ts(
            node.findtext("pubDate") or node.findtext(f"{atom}published") or node.findtext(f"{atom}updated")
        )
        guid = node.findtext("guid") or node.findtext("link") or node.findtext(f"{atom}id") or title
        item_id = hashlib.sha1(f"{source}|{guid}".encode()).hexdigest()
        items.append(NewsItem(item_id, title, ts if ts is not None else now, source))
    return items


def parse_calendar(payload: bytes) -> list[CalendarEvent]:
    """Parse a ForexFactory-style JSON calendar (title, country, date, impact)."""
    data = json.loads(payload)
    if not isinstance(data, list) or not data:
        return []
    df = pd.DataFrame(data)
    required = {"title", "country", "date", "impact"}
    if not required.issubset(df.columns):
        raise ValueError(f"calendar payload missing columns: {required - set(df.columns)}")
    df["ts"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"])
    return [
        CalendarEvent(str(r.title), str(r.country).upper(), str(r.impact), r.ts.timestamp())
        for r in df.itertuples(index=False)
    ]


def make_rss_source(url: str, timeout: float) -> NewsSource:
    async def _fetch() -> list[NewsItem]:
        loop = asyncio.get_running_loop()
        payload = await loop.run_in_executor(None, _http_get, url, timeout)
        return parse_rss(payload, url)

    return _fetch


def make_synthetic_source(symbols: list[str], probability: float = 0.5, seed: Optional[int] = None) -> NewsSource:
    """Random macro headlines for simulation runs."""
    rng = random.Random(seed)
    templates = [
        "Apple shares jump after iPhone sales beat estimates",
        "Nvidia slumps as chip export fears weigh on outlook",
        "Wall Street rallies as tech earnings beat expectations",
        "Tesla stock falls after weak delivery numbers",
        "Silver surges to record on strong industrial demand",
        "Fed signals hawkish stance as US inflation beats expectations",
        "ECB's Lagarde warns of weak eurozone growth, hints at rate cuts",
        "Gold rallies to record as safe-haven demand surges",
        "US nonfarm payrolls miss forecasts, dollar falls",
        "Euro gains after strong German PMI data",
        "Gold slumps as dollar strengthens on upbeat US data",
    ]

    async def _fetch() -> list[NewsItem]:
        if rng.random() > probability:
            return []
        headline = rng.choice(templates)
        now = time.time()
        return [NewsItem(hashlib.sha1(f"{headline}{now}".encode()).hexdigest(), headline, now, "synthetic")]

    return _fetch


def instruments_from_symbols(symbols: list[str]) -> dict[str, Instrument]:
    """Minimal instruments for plain FX/metal symbol lists (tests, manual mode)."""
    out: dict[str, Instrument] = {}
    for s in symbols:
        m = _SYMBOL_RE.match(s.upper())
        if m:
            base, quote = m.groups()
            out[s] = Instrument(s, METAL if base in METALS else FOREX, base=base, quote=quote, ticker=s)
        else:
            out[s] = Instrument(s, OTHER, ticker=s)
    return out


def default_sources(settings: NLPSettings, instruments: dict[str, Instrument]) -> list[NewsSource]:
    sources = [make_rss_source(u, settings.http_timeout_s) for u in settings.rss_feeds]
    tickers = sorted({i.ticker for i in instruments.values() if i.asset_class == US_STOCK and i.ticker})
    if settings.stock_feed_template and tickers:
        n = settings.stock_feed_batch_size
        for k in range(0, len(tickers), n):
            url = settings.stock_feed_template.format(tickers=",".join(tickers[k:k + n]))
            sources.append(make_rss_source(url, settings.http_timeout_s))
    return sources


# ---------------------------------------------------------------- engine
class NLPEngine:
    """Polls news sources and the calendar; exposes per-symbol decayed sentiment."""

    def __init__(
        self,
        settings: NLPSettings,
        instruments: "dict[str, Instrument] | list[str]",
        model: SentimentModel,
        sources: Optional[list[NewsSource]] = None,
    ) -> None:
        self.settings = settings
        self.instruments = instruments if isinstance(instruments, dict) else instruments_from_symbols(instruments)
        self.symbols = list(self.instruments)
        self.model = model
        self.aggregator = SentimentAggregator(settings.half_life_s)
        if sources is None:
            sources = default_sources(settings, self.instruments)
        self.sources = sources
        self.calendar: list[CalendarEvent] = []
        self._seen: set[str] = set()
        self._seen_order: Deque[str] = deque()
        self._seen_cap = 10_000
        self.recent_events: Deque[SentimentEvent] = deque(maxlen=50)
        self._last_calendar = 0.0

    # --- public API used by the strategy -------------------------------
    def sentiment(self, symbol: str, now: Optional[float] = None) -> float:
        return self.aggregator.value(symbol, now)

    def blackout_reason(self, symbol: str, now: Optional[float] = None) -> Optional[str]:
        t = time.time() if now is None else now
        inst = self.instruments.get(symbol)
        ccys = set(inst.calendar_currencies) if inst else set(symbol_currencies(symbol))
        for ev in self.calendar:
            if ev.currency in ccys and ev.impact in self.settings.blackout_impacts:
                if ev.ts - self.settings.blackout_before_s <= t <= ev.ts + self.settings.blackout_after_s:
                    return f"{ev.currency}:{ev.title}"
        return None

    # --- processing -----------------------------------------------------
    def _mark_seen(self, item_id: str) -> None:
        self._seen.add(item_id)
        self._seen_order.append(item_id)
        while len(self._seen_order) > self._seen_cap:
            self._seen.discard(self._seen_order.popleft())

    async def process_headlines(self, items: list[NewsItem], now: Optional[float] = None) -> list[SentimentEvent]:
        t = time.time() if now is None else now
        fresh: list[NewsItem] = []
        for item in items:
            if item.id in self._seen:
                continue
            self._mark_seen(item.id)
            if t - item.published_ts <= self.settings.max_news_age_s:
                fresh.append(item)
        if not fresh:
            return []

        scores = await self.model.score([i.headline for i in fresh])
        events: list[SentimentEvent] = []
        for item, score in zip(fresh, scores):
            if score == 0:
                continue
            currencies = detect_currencies(item.headline)
            hits: list[str] = []
            for symbol, inst in self.instruments.items():
                exposure = instrument_exposure(inst, item.headline, currencies, self.settings.market_beta_weight)
                if exposure == 0:
                    continue
                s0 = max(-1.0, min(1.0, score * exposure))
                t0 = min(item.published_ts, t)
                self.aggregator.add(symbol, s0, t0)
                ev = SentimentEvent(symbol, s0, t0, item.headline)
                events.append(ev)
                self.recent_events.append(ev)
                hits.append(f"{symbol}:{s0:+.2f}")
            if hits:
                shown = " ".join(hits[:8]) + (f" (+{len(hits) - 8} more)" if len(hits) > 8 else "")
                logger.info("NEWS score={:+.2f} -> {} | {}", score, shown, item.headline[:120])
        return events

    async def refresh_calendar(self) -> None:
        if not self.settings.calendar_url:
            return
        loop = asyncio.get_running_loop()
        payload = await loop.run_in_executor(None, _http_get, self.settings.calendar_url, self.settings.http_timeout_s)
        self.calendar = parse_calendar(payload)
        self._last_calendar = time.time()
        logger.info("Economic calendar refreshed: {} events", len(self.calendar))

    async def poll_once(self) -> list[SentimentEvent]:
        results = await asyncio.gather(*(src() for src in self.sources), return_exceptions=True)
        items: list[NewsItem] = []
        for res in results:
            if isinstance(res, BaseException):
                logger.warning("news source failed: {}", res)
            else:
                items.extend(res)
        return await self.process_headlines(items)

    async def run(self, stop: asyncio.Event) -> None:
        logger.info("NLP engine started (model={}, sources={})", self.model.name, len(self.sources))
        while not stop.is_set():
            try:
                if self.settings.calendar_url and time.time() - self._last_calendar >= self.settings.calendar_refresh_s:
                    try:
                        await self.refresh_calendar()
                    except Exception as exc:
                        self._last_calendar = time.time()  # avoid hammering on failure
                        logger.warning("calendar refresh failed: {}", exc)
                await self.poll_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("NLP loop error")
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.settings.poll_interval_s)
            except asyncio.TimeoutError:
                pass
        logger.info("NLP engine stopped")
