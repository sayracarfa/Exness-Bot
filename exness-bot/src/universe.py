"""Instrument universe: discover tradable symbols in the terminal and classify them.

Exness symbol naming differs per account type (``AAPLm`` on Standard, ``AAPL`` on
Raw/Zero ...), so instead of hard-coding hundreds of tickers the bot reads every
symbol from the terminal (``symbols_get``) and classifies it with the symbol's
``path``, ``exchange`` and currency fields:

    metal     XAU / XAG / XPT / XPD crosses
    forex     ISO currency pairs
    us_stock  path looks like stocks/shares + profit currency USD (+ US exchange if reported)
    other     indices, crypto, energies ... (not traded unless listed explicitly)

Each instrument also gets news keyword patterns (ticker + company name + aliases)
so headline sentiment can be routed to individual stocks.

CLI:  python -m src.universe            -> prints the classification, writes universe.csv
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Pattern

from loguru import logger

from src.config import Settings, UniverseSettings, load_settings

FOREX, METAL, US_STOCK, OTHER = "forex", "metal", "us_stock", "other"

ISO_CURRENCIES = {
    "USD", "EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD", "SEK", "NOK", "DKK", "PLN", "HUF", "CZK",
    "TRY", "ZAR", "MXN", "SGD", "HKD", "CNH", "THB", "ILS", "INR", "BRL", "KRW", "RUB",
}
METALS = {"XAU": "gold", "XAG": "silver", "XPT": "platinum", "XPD": "palladium"}

_PAIR_RE = re.compile(r"^([A-Z]{3})([A-Z]{3})")
_TICKER_RE = re.compile(r"^([A-Z0-9]+(?:[.\-][A-Z])?)")

# Headlines often use brand names that differ from the legal description.
STOCK_ALIASES: dict[str, tuple[str, ...]] = {
    "AAPL": ("Apple", "iPhone", "Tim Cook"),
    "MSFT": ("Microsoft", "Azure", "Satya Nadella"),
    "NVDA": ("Nvidia", "Jensen Huang"),
    "AMZN": ("Amazon", "AWS", "Andy Jassy"),
    "GOOGL": ("Google", "Alphabet", "YouTube", "Sundar Pichai"),
    "GOOG": ("Google", "Alphabet", "YouTube"),
    "META": ("Meta Platforms", "Facebook", "Instagram", "WhatsApp", "Zuckerberg"),
    "TSLA": ("Tesla", "Elon Musk"),
    "BRK.B": ("Berkshire", "Buffett"),
    "JPM": ("JPMorgan", "JP Morgan", "Jamie Dimon"),
    "V": ("Visa",),
    "MA": ("Mastercard",),
    "KO": ("Coca-Cola", "Coke"),
    "PEP": ("PepsiCo", "Pepsi"),
    "DIS": ("Disney",),
    "NFLX": ("Netflix",),
    "AMD": ("AMD", "Advanced Micro Devices", "Lisa Su"),
    "INTC": ("Intel",),
    "BA": ("Boeing",),
    "WMT": ("Walmart",),
    "XOM": ("Exxon", "ExxonMobil"),
    "CVX": ("Chevron",),
    "GS": ("Goldman Sachs", "Goldman"),
    "MS": ("Morgan Stanley",),
    "BAC": ("Bank of America",),
    "C": ("Citigroup", "Citi"),
    "WFC": ("Wells Fargo",),
    "NKE": ("Nike",),
    "MCD": ("McDonald's", "McDonalds"),
    "PFE": ("Pfizer",),
    "JNJ": ("Johnson & Johnson",),
    "UNH": ("UnitedHealth",),
    "LLY": ("Eli Lilly", "Lilly"),
    "ORCL": ("Oracle",),
    "CRM": ("Salesforce",),
    "ADBE": ("Adobe",),
    "AVGO": ("Broadcom",),
    "PYPL": ("PayPal",),
    "UBER": ("Uber",),
    "ABNB": ("Airbnb",),
    "COIN": ("Coinbase",),
    "PLTR": ("Palantir",),
}

# Tickers that are ordinary English words / too short: only match as $TICKER or (TICKER)
AMBIGUOUS_TICKERS = {
    "A", "B", "C", "D", "F", "G", "K", "L", "O", "T", "U", "V", "X", "AI", "ALL", "AN", "ARE", "BE", "BIG",
    "CAN", "CAR", "DAY", "EAT", "FOR", "FUN", "GE", "GM", "GO", "HAS", "IT", "KEY", "LOW", "MA", "MS", "NOW",
    "ON", "ONE", "OPEN", "PLAY", "REAL", "RUN", "SAFE", "SEE", "SO", "TEAM", "TRUE", "WELL", "YOU", "META",
    "NET", "CAT", "DOW", "HD", "PM", "SPY", "AMP", "LIFE", "MAIN", "PEAK", "WAY", "LOVE", "HOOD", "BILL",
}

_NAME_NOISE = re.compile(
    r"\b(inc|incorporated|corp|corporation|co|company|ltd|limited|plc|holdings?|group|class\s+[a-c]|"
    r"n\.?v|s\.?a|ag|se|adr|ads|common\s+stock|ordinary\s+shares?|shares?|the|cfd|com)\b\.?",
    re.IGNORECASE,
)

MARKET_PATTERNS: tuple[Pattern[str], ...] = (
    re.compile(r"\b(Wall Street|S&P 500|S&P500|Nasdaq|Dow Jones|Russell 2000|US stocks|U\.S\. stocks|stock market|equities)\b", re.I),
)


@dataclass(frozen=True)
class Instrument:
    symbol: str
    asset_class: str
    description: str = ""
    ticker: str = ""
    base: str = ""
    quote: str = ""
    path: str = ""
    patterns: tuple[Pattern[str], ...] = field(default=(), compare=False)

    @property
    def calendar_currencies(self) -> tuple[str, ...]:
        """Currencies whose high-impact macro events should block new entries."""
        if self.asset_class == US_STOCK:
            return ("USD",)
        return tuple(c for c in (self.base, self.quote) if c)

    def matches_own(self, text: str) -> bool:
        return any(p.search(text) for p in self.patterns)


# ---------------------------------------------------------------- helpers
def clean_company_name(description: str) -> str:
    name = re.sub(r"\(.*?\)", " ", description)
    name = _NAME_NOISE.sub(" ", name)
    name = re.sub(r"[,]", " ", name)
    return re.sub(r"\s+", " ", name).strip(" .-&")


def ticker_of(symbol: str) -> str:
    """Strip broker suffixes: 'AAPLm' -> 'AAPL', 'BRK.Bm' -> 'BRK.B', 'XAUUSDm' -> 'XAUUSD'."""
    m = _TICKER_RE.match(symbol)
    return m.group(1) if m else symbol.upper()


def _name_pattern(name: str) -> Pattern[str]:
    flags = re.IGNORECASE if len(name) >= 5 else 0
    return re.compile(rf"(?<![\w$]){re.escape(name)}(?!\w)", flags)


def build_stock_patterns(ticker: str, description: str) -> tuple[Pattern[str], ...]:
    pats: list[Pattern[str]] = []
    t = re.escape(ticker)
    if ticker in AMBIGUOUS_TICKERS or len(ticker) <= 2:
        pats.append(re.compile(rf"\${t}\b|\({t}\)|(?:NYSE|NASDAQ|Nasdaq):\s*{t}\b"))
    else:
        pats.append(re.compile(rf"(?<![A-Za-z])\$?{t}\b"))
    names = {clean_company_name(description)} | set(STOCK_ALIASES.get(ticker, ()))
    for name in sorted(n for n in names if n and len(n) >= 3):
        pats.append(_name_pattern(name))
    return tuple(pats)


def classify(info: Any, cfg: UniverseSettings) -> str:
    name = str(info.name)
    ticker = ticker_of(name)
    path = str(getattr(info, "path", "") or "")
    exchange = str(getattr(info, "exchange", "") or "")
    m = _PAIR_RE.match(ticker)
    if m and m.group(1) in METALS:
        return METAL
    if m and m.group(1) in ISO_CURRENCIES and m.group(2) in ISO_CURRENCIES and len(ticker) == 6:
        return FOREX
    if re.search(cfg.exclude_path_regex, path):
        return OTHER
    profit_ccy = str(getattr(info, "currency_profit", "") or "")
    if re.search(cfg.stock_path_regex, path) and profit_ccy == cfg.stock_profit_currency:
        if exchange and not re.search(cfg.us_exchange_regex, exchange):
            return OTHER  # non-US listing quoted in USD
        return US_STOCK
    return OTHER


def make_instrument(info: Any, asset_class: str) -> Instrument:
    name = str(info.name)
    ticker = ticker_of(name)
    description = str(getattr(info, "description", "") or "")
    path = str(getattr(info, "path", "") or "")
    if asset_class in (FOREX, METAL):
        m = _PAIR_RE.match(ticker)
        base, quote = (m.group(1), m.group(2)) if m else ("", "")
        return Instrument(name, asset_class, description, ticker, base, quote, path)
    if asset_class == US_STOCK:
        return Instrument(name, asset_class, description, ticker, "", "USD", path, build_stock_patterns(ticker, description))
    return Instrument(name, asset_class, description, ticker, "", "", path)


def is_tradable(info: Any, full_mode: int) -> bool:
    mode = getattr(info, "trade_mode", full_mode)
    return int(mode) == int(full_mode)


def build_universe(infos: Iterable[Any], settings: Settings, full_mode: int = 4) -> dict[str, Instrument]:
    """Pure function: classify symbol infos and apply the configured filters."""
    cfg = settings.universe
    by_name = {str(i.name): i for i in infos}
    selected: dict[str, Instrument] = {}

    if cfg.mode == "auto":
        wanted = set(cfg.asset_classes)
        for name, info in sorted(by_name.items()):
            if name in cfg.exclude or not is_tradable(info, full_mode):
                continue
            ac = classify(info, cfg)
            if ac in wanted:
                selected[name] = make_instrument(info, ac)

    for name in settings.symbols:  # explicit symbols are always included (if they exist)
        info = by_name.get(name)
        if info is None:
            logger.warning("Configured symbol {} not found in terminal - skipped", name)
            continue
        if name not in selected and name not in cfg.exclude:
            selected[name] = make_instrument(info, classify(info, cfg))

    if len(selected) > cfg.max_symbols:
        logger.warning("Universe has {} symbols; truncating to max_symbols={}", len(selected), cfg.max_symbols)
        keep = list(settings.symbols) + [s for s in selected if s not in settings.symbols]
        selected = {s: selected[s] for s in keep[: cfg.max_symbols] if s in selected}
    return selected


async def discover_universe(client: Any, settings: Settings) -> dict[str, Instrument]:
    infos = await client.call("symbols_get")
    if infos is None:
        code, msg = await client.last_error()
        raise RuntimeError(f"symbols_get failed: [{code}] {msg}")
    try:
        full_mode = client.const("SYMBOL_TRADE_MODE_FULL")
    except AttributeError:
        full_mode = 4
    universe = build_universe(infos, settings, full_mode)
    counts = Counter(i.asset_class for i in universe.values())
    logger.info("Universe: {} symbols {} (of {} in terminal)", len(universe), dict(counts), len(infos))
    return universe


# -------------------------------------------------------------------- CLI
def main(argv: Optional[list[str]] = None) -> int:
    import pandas as pd

    from src.mt5_client import AsyncMT5Client, load_backend

    p = argparse.ArgumentParser(description="List and classify all terminal symbols")
    p.add_argument("--config")
    p.add_argument("--simulate", action="store_true")
    p.add_argument("--out", default="universe.csv")
    args = p.parse_args(argv)
    settings = load_settings(args.config)
    settings.simulate = settings.simulate or args.simulate

    async def run() -> int:
        client = AsyncMT5Client(load_backend(settings.simulate))
        await client.connect(settings.mt5)
        try:
            infos = await client.call("symbols_get") or ()
            selected = build_universe(infos, settings, client.const("SYMBOL_TRADE_MODE_FULL"))
            rows = [{
                "symbol": i.name, "class": classify(i, settings.universe), "selected": i.name in selected,
                "description": getattr(i, "description", ""), "path": getattr(i, "path", ""),
                "exchange": getattr(i, "exchange", ""), "profit_ccy": getattr(i, "currency_profit", ""),
                "trade_mode": getattr(i, "trade_mode", ""),
            } for i in infos]
            df = pd.DataFrame(rows).sort_values(["selected", "class", "symbol"], ascending=[False, True, True])
            df.to_csv(args.out, index=False)
            print(df.groupby(["class", "selected"]).size().to_string())
            print(f"\nSelected {len(selected)} symbols. Full list written to {args.out}")
            print("Example stock symbols:", ", ".join(s for s, i in selected.items() if i.asset_class == US_STOCK)[:400])
        finally:
            await client.disconnect()
        return 0

    return asyncio.run(run())


if __name__ == "__main__":
    sys.exit(main())
