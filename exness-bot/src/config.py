"""Central configuration: account credentials, symbols, risk limits and model thresholds.

Values are loaded (in increasing priority) from:
  1. the defaults below,
  2. an optional JSON file (``--config`` or ``BOT_CONFIG`` env var),
  3. environment variables (credentials should ONLY come from here).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, SecretStr, model_validator


class MT5Credentials(BaseModel):
    """MT5 terminal / Exness account settings."""

    login: int = 0  # 0 => use the account already logged into the terminal
    password: SecretStr = SecretStr("")
    server: str = "Exness-MT5Trial"  # e.g. Exness-MT5Real8, Exness-MT5Trial6 ...
    terminal_path: Optional[str] = None  # path to terminal64.exe (optional)
    timeout_ms: int = Field(10_000, ge=1_000)


class RiskLimits(BaseModel):
    """Capital-protection limits used by :mod:`src.risk_controller`."""

    # Kelly sizing
    max_risk_per_trade: float = Field(0.005, gt=0, le=0.05)  # hard cap: 0.5% equity at risk
    kelly_fraction: float = Field(0.25, gt=0, le=1.0)  # fractional Kelly multiplier
    prior_win_rate: float = Field(0.52, gt=0, lt=1)  # used until enough trades are observed
    prior_payoff_ratio: float = Field(1.0, gt=0)
    min_trades_for_kelly: int = Field(30, ge=5)

    # VaR / drawdown
    var_confidence: float = Field(0.99, gt=0.5, lt=1)
    var_window: int = Field(500, ge=50)
    min_var_observations: int = Field(30, ge=10)
    max_var_pct: float = Field(0.02, gt=0)
    max_drawdown_pct: float = Field(0.05, gt=0, lt=1)
    max_daily_loss_pct: float = Field(0.02, gt=0, lt=1)

    # Circuit breaker
    max_latency_ms: float = Field(50.0, gt=0)  # local terminal calls (ticks, DOM) + internal pipeline
    # order_send includes the network round trip to the broker server: judged separately
    max_order_latency_ms: float = Field(400.0, gt=0)
    latency_breach_count: int = Field(3, ge=1)  # consecutive breaches before tripping (1 = instant)
    max_consecutive_errors: int = Field(5, ge=1)
    breaker_cooldown_s: float = Field(300.0, ge=0)

    # Exposure
    max_positions_per_symbol: int = Field(1, ge=1)
    max_total_positions: int = Field(8, ge=1)
    # Per asset class caps (US stocks are highly correlated: limit count AND net direction)
    max_positions_per_class: Dict[str, int] = Field(
        default_factory=lambda: {"us_stock": 5, "metal": 2, "forex": 3}
    )
    max_net_direction_per_class: Dict[str, int] = Field(default_factory=lambda: {"us_stock": 3})
    max_margin_usage: float = Field(0.25, gt=0, le=1)  # max share of free margin one new trade may use


class StrategyParams(BaseModel):
    """Alpha / signal engine thresholds used by :mod:`src.strategy`."""

    # Ornstein-Uhlenbeck mean reversion
    ou_window: int = Field(300, ge=50)
    ou_min_obs: int = Field(120, ge=30)
    ou_max_half_life_ticks: float = Field(150.0, gt=0)
    z_entry: float = Field(2.0, gt=0)
    z_exit: float = Field(0.3, ge=0)

    # News sentiment overlay
    sentiment_weight: float = Field(0.75, ge=0)  # in z-score units
    sentiment_veto: float = Field(0.5, gt=0, le=1)

    # VPIN toxicity filter
    vpin_bucket_volume: float = Field(50.0, gt=0)
    vpin_num_buckets: int = Field(50, ge=5)
    vpin_threshold: float = Field(0.7, gt=0, le=1)

    # Stationarity test: Dickey-Fuller t-stat of gamma in dX = a + gamma*X(t-1) + e
    # must be below this critical value (5%: -2.87, 1%: -3.44). None disables the test.
    adf_critical: Optional[float] = -2.87

    # Execution economics
    stop_sigma_mult: float = Field(3.0, gt=0)
    min_edge_to_spread: float = Field(1.5, ge=0)
    signal_cooldown_s: float = Field(1.0, ge=0)
    # Exit if the trade has not reverted within this many half-lives (tick time). None disables.
    time_stop_half_lives: Optional[float] = Field(3.0, gt=0)
    # Require positive expected value per trade using the OU first-passage win probability.
    require_positive_ev: bool = True
    win_prob_haircut: float = Field(0.02, ge=0, lt=0.5)  # model-risk margin subtracted from p

    @model_validator(mode="after")
    def _check_consistency(self) -> "StrategyParams":
        if self.z_exit >= self.z_entry:
            raise ValueError("z_exit must be smaller than z_entry")
        if self.ou_min_obs > self.ou_window:
            raise ValueError("ou_min_obs cannot exceed ou_window")
        return self


class NLPSettings(BaseModel):
    """News feed and sentiment model settings used by :mod:`src.nlp_engine`."""

    enabled: bool = True
    rss_feeds: List[str] = Field(
        default_factory=lambda: [
            "https://www.forexlive.com/feed/news",
            "https://www.fxstreet.com/rss/news",
        ]
    )
    calendar_url: Optional[str] = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    poll_interval_s: float = Field(60.0, ge=5)
    calendar_refresh_s: float = Field(1800.0, ge=60)
    http_timeout_s: float = Field(10.0, gt=0)
    half_life_s: float = Field(900.0, gt=0)  # lambda = ln(2) / half_life
    max_news_age_s: float = Field(3600.0, gt=0)
    use_finbert: bool = True
    finbert_model: str = "ProsusAI/finbert"
    blackout_before_s: float = Field(300.0, ge=0)
    blackout_after_s: float = Field(300.0, ge=0)
    blackout_impacts: List[str] = Field(default_factory=lambda: ["High"])
    synthetic_news: bool = False  # simulation only

    # Per-ticker stock headlines (Yahoo Finance RSS accepts comma-separated tickers).
    stock_feed_template: Optional[str] = (
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s={tickers}&region=US&lang=en-US"
    )
    stock_feed_batch_size: int = Field(20, ge=1, le=50)
    market_beta_weight: float = Field(0.4, ge=0, le=1)  # broad "Wall Street" news -> every US stock


class UniverseSettings(BaseModel):
    """Which instruments to trade.

    mode="auto": discover tradable symbols in the terminal (symbols_get) and keep the
    requested asset classes; ``Settings.symbols`` are always added on top.
    mode="manual": trade exactly ``Settings.symbols``.
    """

    mode: Literal["auto", "manual"] = "auto"
    asset_classes: List[str] = Field(default_factory=lambda: ["us_stock", "metal"])
    exclude: List[str] = Field(default_factory=list)
    max_symbols: int = Field(250, ge=1)
    # classification hints (MT5 symbol "path" / "exchange" fields; broker specific)
    stock_path_regex: str = r"(?i)(stock|share|equit|nasdaq|nyse)"
    exclude_path_regex: str = r"(?i)(crypto|index|indices|energ|commodit|etf)"
    us_exchange_regex: str = r"(?i)(nasdaq|nyse|amex|arca|bats|us)"
    stock_profit_currency: str = "USD"
    depth_asset_classes: List[str] = Field(default_factory=lambda: ["forex", "metal"])


class SessionSettings(BaseModel):
    """US equity trading session (NYSE/Nasdaq regular hours, America/New_York)."""

    enforce: bool = True
    timezone: str = "America/New_York"
    open_time: str = "09:30"
    close_time: str = "16:00"
    early_close_time: str = "13:00"
    skip_after_open_min: float = Field(5.0, ge=0)  # opening auction volatility
    no_entry_before_close_min: float = Field(30.0, ge=0)
    flatten_before_close_min: float = Field(15.0, ge=0)  # no overnight stock positions
    # Update yearly: NYSE full holidays / early (13:00) closes.
    holidays: List[str] = Field(default_factory=lambda: ["2026-11-26", "2026-12-25"])
    early_closes: List[str] = Field(default_factory=lambda: ["2026-11-27", "2026-12-24"])


class PortfolioRiskSettings(BaseModel):
    """Volatility targeting + correlation-aware portfolio VaR (src/portfolio_risk.py).

    Prices are sampled into fixed-length bars (calendar time) so volatilities and
    correlations are comparable across symbols. Very short bars understate correlations
    (Epps effect); 10-30 s is a reasonable compromise for intraday FX/metals/stocks.
    """

    enabled: bool = True
    bar_seconds: float = Field(10.0, gt=0)
    max_gap_bars: int = Field(30, ge=1)  # longer gaps (closed market) break return continuity
    vol_short_halflife_bars: float = Field(30.0, gt=0)  # ~5 min at 10 s bars
    vol_long_halflife_bars: float = Field(360.0, gt=0)  # ~1 h at 10 s bars
    min_bars: int = Field(60, ge=10)
    vol_target_floor: float = Field(0.25, gt=0, le=1)  # smallest size multiplier
    # short/long vol above this -> no new entries. The two EWMAs move together, so even a
    # sudden 6-10x jump peaks around 2.5-3 within minutes: keep this near 2.
    vol_spike_ratio: float = Field(2.0, gt=1)
    unknown_vol_multiplier: float = Field(0.5, gt=0, le=1)  # size multiplier while vol is unknown
    corr_window_bars: int = Field(360, ge=30)
    var_confidence: float = Field(0.99, gt=0.5, lt=1)
    var_horizon_s: float = Field(300.0, gt=0)
    max_portfolio_var_pct: float = Field(0.01, gt=0, lt=1)  # of equity, over var_horizon_s


class Settings(BaseModel):
    """Top-level bot settings."""

    mt5: MT5Credentials = Field(default_factory=MT5Credentials)
    # Exness Standard hesaplarda semboller "m" son ekiyle gelir (EURUSDm). Raw/Zero hesaplarda ek yoktur.
    symbols: List[str] = Field(default_factory=lambda: ["EURUSDm", "XAUUSDm"])
    magic: int = 20261103
    dry_run: bool = True  # paper trading on a real feed: orders are NOT sent
    require_demo: bool = True  # refuse to start unless the MT5 account is a DEMO account
    simulate: bool = False  # use the in-process MT5 simulator instead of a terminal
    poll_interval_ms: float = Field(10.0, gt=0)  # pause between full universe sweeps
    use_depth: bool = True
    deviation_points: int = Field(10, ge=0)
    max_order_retries: int = Field(2, ge=0)
    flatten_on_exit: bool = True
    equity_poll_s: float = Field(1.0, gt=0)
    status_log_s: float = Field(10.0, gt=0)

    api_enabled: bool = True
    api_host: str = "127.0.0.1"
    api_port: int = Field(8080, ge=1, le=65535)

    log_level: str = "INFO"
    log_dir: str = "logs"

    risk: RiskLimits = Field(default_factory=RiskLimits)
    strategy: StrategyParams = Field(default_factory=StrategyParams)
    # Per asset class overrides of StrategyParams fields, e.g. {"us_stock": {"z_entry": 2.5}}
    strategy_overrides: Dict[str, Dict[str, Any]] = Field(
        default_factory=lambda: {
            "us_stock": {"z_entry": 2.5, "min_edge_to_spread": 2.0, "vpin_bucket_volume": 20.0},
            "metal": {"z_entry": 2.2},
        }
    )
    nlp: NLPSettings = Field(default_factory=NLPSettings)
    universe: UniverseSettings = Field(default_factory=UniverseSettings)
    sessions: SessionSettings = Field(default_factory=SessionSettings)
    portfolio: PortfolioRiskSettings = Field(default_factory=PortfolioRiskSettings)
    warmup_seconds: float = Field(900.0, ge=0)  # tick history used to warm models at start
    tick_batch_size: int = Field(5000, ge=100)

    def params_for(self, asset_class: str) -> StrategyParams:
        override = self.strategy_overrides.get(asset_class)
        if not override:
            return self.strategy
        return StrategyParams.model_validate({**self.strategy.model_dump(), **override})


def _env_overrides() -> dict[str, Any]:
    """Collect overrides from environment variables (credentials live here)."""
    mt5: dict[str, Any] = {}
    if login := os.getenv("MT5_LOGIN"):
        mt5["login"] = int(login)
    if password := os.getenv("MT5_PASSWORD"):
        mt5["password"] = password
    if server := os.getenv("MT5_SERVER"):
        mt5["server"] = server
    if path := os.getenv("MT5_PATH"):
        mt5["terminal_path"] = path

    out: dict[str, Any] = {}
    if mt5:
        out["mt5"] = mt5
    if symbols := os.getenv("BOT_SYMBOLS"):
        out["symbols"] = [s.strip() for s in symbols.split(",") if s.strip()]
    if dry := os.getenv("BOT_DRY_RUN"):
        out["dry_run"] = dry.strip().lower() in {"1", "true", "yes", "on"}
    if demo := os.getenv("BOT_REQUIRE_DEMO"):
        out["require_demo"] = demo.strip().lower() in {"1", "true", "yes", "on"}
    if level := os.getenv("BOT_LOG_LEVEL"):
        out["log_level"] = level
    return out


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


DEFAULT_CONFIG_FILE = "config.json"


def load_settings(path: Optional[str | Path] = None) -> Settings:
    """Build validated :class:`Settings` from file + environment.

    File lookup order: explicit ``path`` -> ``BOT_CONFIG`` env var -> ``config.json`` in the
    current directory (if present). Environment variables override file values.
    """
    data: dict[str, Any] = {}
    config_path = path or os.getenv("BOT_CONFIG")
    if not config_path and Path(DEFAULT_CONFIG_FILE).is_file():
        config_path = DEFAULT_CONFIG_FILE
    if config_path:
        file_path = Path(config_path)
        if not file_path.is_file():
            raise FileNotFoundError(f"Config file not found: {file_path}")
        # utf-8-sig: tolerate the BOM some Windows editors (Notepad) write
        with file_path.open("r", encoding="utf-8-sig") as fh:
            try:
                data = json.load(fh)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{file_path}: invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}. "
                    "Check commas between entries and double quotes around text values."
                ) from exc
    data = _deep_merge(data, _env_overrides())
    return Settings.model_validate(data)
