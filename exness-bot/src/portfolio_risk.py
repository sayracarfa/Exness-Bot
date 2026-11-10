"""Volatility targeting and correlation-aware portfolio VaR.

Prices are sampled into fixed calendar-time bars (``bar_seconds``). For every symbol:

* **Volatility targeting** — two EWMA estimates of bar-return variance,
      var_t = lambda * var_{t-1} + (1 - lambda) * r_t^2,   lambda = 0.5^(1 / half_life)
  a short one (current regime) and a long one (normal level). The size multiplier is
      m = clamp(sigma_long / sigma_short, floor, 1)
  so positions shrink when the market is more volatile than usual and are never
  enlarged beyond the base Kelly/risk size. If sigma_short / sigma_long exceeds
  ``vol_spike_ratio`` new entries are blocked.

* **Portfolio VaR** — with money exposures e_i (signed notional in account currency)
  and the covariance matrix S of bar log-returns:
      VaR = z_alpha * sqrt(H) * sqrt(e' S e)          (H = horizon in bars)
  For a new position of x lots (value v per lot, signed by side) the variance is a
  quadratic  a x^2 + 2 b x + c  with  a = v^2 S_nn,  b = v (S e)_n,  c = e' S e,
  so the largest x keeping VaR <= limit is solved exactly. Correlated same-direction
  trades get smaller, hedging trades are allowed more.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any, Deque, Iterable, Optional

import numpy as np

from src.config import PortfolioRiskSettings


@dataclass
class _SymbolBars:
    cur_bar: Optional[int] = None
    cur_close: float = 0.0
    prev_close: Optional[float] = None
    prev_bar: Optional[int] = None
    var_short: Optional[float] = None
    var_long: Optional[float] = None
    n_returns: int = 0
    closes: Deque[tuple[int, float]] = field(default_factory=deque)


class PortfolioRisk:
    def __init__(self, cfg: PortfolioRiskSettings) -> None:
        self.cfg = cfg
        self._lam_s = 0.5 ** (1.0 / cfg.vol_short_halflife_bars)
        self._lam_l = 0.5 ** (1.0 / cfg.vol_long_halflife_bars)
        self._z = NormalDist().inv_cdf(cfg.var_confidence)
        self._horizon_bars = max(cfg.var_horizon_s / cfg.bar_seconds, 1.0)
        self._sym: dict[str, _SymbolBars] = {}

    # ------------------------------------------------------------ prices
    def on_price(self, symbol: str, ts: float, price: float) -> None:
        """Feed every observed price (``ts`` = broker/epoch seconds)."""
        if not (price > 0 and math.isfinite(price)):
            return
        st = self._sym.get(symbol)
        if st is None:
            st = self._sym[symbol] = _SymbolBars(closes=deque(maxlen=self.cfg.corr_window_bars + 1))
        bar = int(ts // self.cfg.bar_seconds)
        if st.cur_bar is None:
            st.cur_bar, st.cur_close = bar, price
            return
        if bar < st.cur_bar:
            return  # out-of-order tick
        if bar > st.cur_bar:
            self._close_bar(st, st.cur_bar, st.cur_close)
            st.cur_bar = bar
        st.cur_close = price

    def _close_bar(self, st: _SymbolBars, bar: int, close: float) -> None:
        if st.prev_close is not None and st.prev_bar is not None:
            gap = bar - st.prev_bar
            if gap > self.cfg.max_gap_bars:
                st.closes.clear()  # market was closed / feed stalled: no return across the gap
            else:
                r = math.log(close / st.prev_close)
                for _ in range(gap - 1):  # empty bars in between = no price change
                    self._update_vol(st, 0.0)
                self._update_vol(st, r)
        st.prev_close, st.prev_bar = close, bar
        st.closes.append((bar, close))

    def _update_vol(self, st: _SymbolBars, r: float) -> None:
        r2 = r * r
        if st.var_short is None or st.var_long is None:
            st.var_short = st.var_long = r2
        else:
            st.var_short = self._lam_s * st.var_short + (1 - self._lam_s) * r2
            st.var_long = self._lam_l * st.var_long + (1 - self._lam_l) * r2
        st.n_returns += 1

    # ---------------------------------------------------- volatility gate
    def has_vol(self, symbol: str) -> bool:
        st = self._sym.get(symbol)
        return st is not None and st.n_returns >= self.cfg.min_bars and bool(st.var_long)

    def vol(self, symbol: str) -> Optional[tuple[float, float]]:
        """(short, long) per-bar return volatility, or None while warming up."""
        if not self.has_vol(symbol):
            return None
        st = self._sym[symbol]
        return math.sqrt(st.var_short or 0.0), math.sqrt(st.var_long or 0.0)

    def vol_ratio(self, symbol: str) -> Optional[float]:
        v = self.vol(symbol)
        if v is None or v[1] <= 0:
            return None
        return v[0] / v[1]

    def vol_multiplier(self, symbol: str) -> float:
        ratio = self.vol_ratio(symbol)
        if ratio is None:
            return self.cfg.unknown_vol_multiplier
        if ratio <= 0:
            return 1.0
        return min(1.0, max(self.cfg.vol_target_floor, 1.0 / ratio))

    def vol_spike(self, symbol: str) -> bool:
        ratio = self.vol_ratio(symbol)
        return ratio is not None and ratio > self.cfg.vol_spike_ratio

    # -------------------------------------------------------- covariance
    def covariance(self, symbols: list[str]) -> Optional[np.ndarray]:
        """Covariance of aligned bar log-returns (forward-filled); None if too little overlap."""
        series = []
        for s in symbols:
            st = self._sym.get(s)
            if st is None or len(st.closes) < 3:
                return None
            series.append(dict(st.closes))
        end = max(max(d) for d in series)
        start = end - self.cfg.corr_window_bars
        first = max(min(d) for d in series)  # common start: every symbol has a price
        grid = range(max(start, first), end + 1)
        if len(grid) < self.cfg.min_bars + 1:
            return None
        mat = np.empty((len(grid), len(series)))
        for j, d in enumerate(series):
            last = None
            for i, b in enumerate(grid):
                last = d.get(b, last)
                mat[i, j] = np.nan if last is None else last
        mat = mat[~np.isnan(mat).any(axis=1)]
        if len(mat) < self.cfg.min_bars + 1:
            return None
        rets = np.diff(np.log(mat), axis=0)
        if len(symbols) == 1:
            return np.array([[float(rets[:, 0].var(ddof=1))]])
        return np.cov(rets, rowvar=False)

    def correlation(self, a: str, b: str) -> Optional[float]:
        cov = self.covariance([a, b])
        if cov is None or cov[0, 0] <= 0 or cov[1, 1] <= 0:
            return None
        return float(cov[0, 1] / math.sqrt(cov[0, 0] * cov[1, 1]))

    # ---------------------------------------------------------------- VaR
    def portfolio_var(self, exposures: dict[str, float]) -> Optional[float]:
        """Parametric VaR (account currency) of signed money exposures; None if unknown."""
        syms = [s for s, e in exposures.items() if e]
        if not syms:
            return 0.0
        cov = self.covariance(syms)
        if cov is None:
            return None
        e = np.array([exposures[s] for s in syms])
        variance = float(e @ cov @ e)
        return self._z * math.sqrt(self._horizon_bars * max(variance, 0.0))

    def max_lots_for_var(
        self,
        symbol: str,
        value_per_lot: float,
        exposures: dict[str, float],
        limit: float,
    ) -> Optional[float]:
        """Largest lot size for a NEW position on ``symbol`` keeping portfolio VaR <= ``limit``.

        ``value_per_lot`` is signed (+ long / - short) account-currency notional per lot.
        Returns None when the covariance is not yet known (caller decides the fallback).
        """
        others = [s for s, e in exposures.items() if e and s != symbol]
        syms = [symbol] + others
        cov = self.covariance(syms)
        if cov is None:
            return None
        e = np.array([0.0] + [exposures[s] for s in others])
        v = value_per_lot
        a = v * v * cov[0, 0]
        b = v * float((cov @ e)[0])
        c = float(e @ cov @ e)
        k = (limit / (self._z * math.sqrt(self._horizon_bars))) ** 2
        if c >= k:
            # Book already at/over the limit: only a trade that LOWERS total variance is
            # allowed, sized at most to the variance-minimising point x = -b / a.
            return max(-b / a, 0.0) if a > 0 and b < 0 else 0.0
        if a <= 0:
            return math.inf
        disc = b * b - a * (c - k)
        return max((-b + math.sqrt(max(disc, 0.0))) / a, 0.0)

    # ------------------------------------------------------------ report
    def snapshot(self, symbols: Iterable[str]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for s in symbols:
            ratio = self.vol_ratio(s)
            out[s] = {
                "vol_ratio": None if ratio is None else round(ratio, 3),
                "size_mult": round(self.vol_multiplier(s), 3),
                "spike": self.vol_spike(s),
            }
        return out
