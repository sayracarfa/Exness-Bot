"""Mathematical alpha & signal engine.

Components
----------
1. Ornstein-Uhlenbeck mean reversion on the micro-price
       dX = theta (mu - X) dt + sigma dW
   Exact discretisation (step dt = 1 tick):
       X_{k+1} = a + b X_k + eps,  b = exp(-theta),  a = mu (1 - b)
   so  theta = -ln b,  mu = a / (1 - b),  sigma_eq = sd(eps) / sqrt(1 - b^2)
   z = (X - mu) / sigma_eq

2. VPIN (Easley, Lopez de Prado & O'Hara, 2012) with bulk-volume classification
       V_buy = V * Phi(dP / sigma_dP),   V_sell = V - V_buy
       VPIN  = sum_{tau=1..n} |V_buy,tau - V_sell,tau| / (n * V)
   High VPIN => informed/toxic flow => market makers get adversely selected => no entries.

3. News sentiment overlay S_decayed(t) (see nlp_engine) blended into the alpha:
       alpha = -z + w_s * S_decayed(t)
"""

from __future__ import annotations

import functools
import math
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Deque, Optional

import numpy as np
from loguru import logger

from src.config import StrategyParams
from src.data_ingestion import MarketSnapshot
from src.models import Side

SentimentProvider = Callable[[str, float], float]
BlackoutProvider = Callable[[str, float], Optional[str]]

_SQRT2 = math.sqrt(2.0)


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


# ------------------------------------------------------------------- OU
@dataclass(frozen=True, slots=True)
class OUParams:
    mu: float
    theta: float  # per tick
    sigma: float  # diffusion per sqrt(tick)
    sigma_eq: float  # stationary standard deviation
    b: float  # AR(1) coefficient
    half_life: float  # ticks
    n: int
    adf_t: float = 0.0  # Dickey-Fuller t-statistic of (b - 1)


class OUEstimator:
    """Rolling-window OLS estimate of OU parameters with a Dickey-Fuller stationarity test.

    Regression on pairs (X_t, X_{t+1}) inside the window:  X_{t+1} = a + b X_t + e.
    Dickey-Fuller:  t = (b - 1) / se(b).  OLS on a pure random walk is biased towards
    b < 1, so "0 < b < 1" alone accepts most random walks as mean-reverting; requiring
    t < critical (-2.87 at 5%) rejects them.

    Sums are maintained incrementally (O(1) per tick) around a reference price and fully
    recomputed every ``window`` updates to stop floating-point drift.
    """

    def __init__(self, window: int, min_obs: int, max_half_life: float, adf_critical: Optional[float] = -2.87) -> None:
        if min_obs < 3 or min_obs > window:
            raise ValueError("require 3 <= min_obs <= window")
        self.window = window
        self._buf: Deque[float] = deque(maxlen=window)  # deviations from self._ref
        self._min_obs = min_obs
        self._max_half_life = max_half_life
        self._adf_critical = adf_critical
        self._ref: Optional[float] = None
        self._sx = self._sy = self._sxx = self._sxy = self._syy = 0.0
        self._updates = 0
        self._params: Optional[OUParams] = None
        self._dirty = True
        self.reason = "warming_up"

    def __len__(self) -> int:
        return len(self._buf)

    def _add_pair(self, x: float, y: float, sign: float) -> None:
        self._sx += sign * x
        self._sy += sign * y
        self._sxx += sign * x * x
        self._sxy += sign * x * y
        self._syy += sign * y * y

    def _recompute(self) -> None:
        self._sx = self._sy = self._sxx = self._sxy = self._syy = 0.0
        buf = self._buf
        for i in range(len(buf) - 1):
            self._add_pair(buf[i], buf[i + 1], 1.0)

    def update(self, price: float) -> None:
        if not math.isfinite(price):
            return
        if self._ref is None:
            self._ref = price
        x = price - self._ref
        buf = self._buf
        if len(buf) == buf.maxlen:  # oldest pair (buf[0], buf[1]) leaves the window
            self._add_pair(buf[0], buf[1], -1.0)
        if buf:
            self._add_pair(buf[-1], x, 1.0)
        buf.append(x)
        self._updates += 1
        if self._updates % self.window == 0:
            self._recompute()
        self._dirty = True

    def fit(self) -> Optional[OUParams]:
        if not self._dirty:
            return self._params
        self._dirty = False
        self._params = None
        n = len(self._buf)
        if n < self._min_obs:
            self.reason = "warming_up"
            return None
        m = n - 1  # number of regression pairs
        mx, my = self._sx / m, self._sy / m
        sxx = self._sxx - m * mx * mx
        sxy = self._sxy - m * mx * my
        syy = self._syy - m * my * my
        if sxx <= 1e-30:
            self.reason = "flat_price"
            return None
        b = sxy / sxx
        if not 0.0 < b < 1.0:
            self.reason = "not_mean_reverting"
            return None
        ssr = max(syy - b * sxy, 0.0)
        dof = max(m - 2, 1)
        sig_eps = math.sqrt(ssr / dof)
        se_b = sig_eps / math.sqrt(sxx)
        adf_t = (b - 1.0) / se_b if se_b > 0 else -math.inf
        if self._adf_critical is not None and adf_t >= self._adf_critical:
            self.reason = "no_stationarity"  # cannot reject a random walk
            return None
        theta = -math.log(b)
        half_life = math.log(2.0) / theta
        if half_life > self._max_half_life:
            self.reason = "half_life_too_long"
            return None
        one_minus_b2 = 1.0 - b * b
        sigma_eq = sig_eps / math.sqrt(one_minus_b2)
        if sigma_eq <= 0.0 or not math.isfinite(sigma_eq):
            self.reason = "degenerate_variance"
            return None
        a = my - b * mx
        assert self._ref is not None
        self._params = OUParams(
            mu=a / (1.0 - b) + self._ref,
            theta=theta,
            sigma=sig_eps * math.sqrt(2.0 * theta / one_minus_b2),
            sigma_eq=sigma_eq,
            b=b,
            half_life=half_life,
            n=n,
            adf_t=adf_t,
        )
        self.reason = "ok"
        return self._params

    @staticmethod
    def zscore(x: float, params: OUParams) -> float:
        return (x - params.mu) / params.sigma_eq


def ou_hit_probability(u0: float, u_target: float, u_stop: float, points: int = 400) -> float:
    """P(an OU process in stationary-sd units reaches ``u_target`` before ``u_stop``) from ``u0``.

    For dU = -theta U dt + sqrt(2 theta) dW the scale density is s(u) = exp(u^2 / 2), so
        P = integral_{u_stop}^{u0} s / integral_{u_stop}^{u_target} s.
    ``u_target`` and ``u_stop`` must bracket ``u0``.
    """
    lo, hi = (u_stop, u_target) if u_stop < u_target else (u_target, u_stop)
    if not lo < u0 < hi:
        return 1.0 if (u0 >= u_target) == (u_target > u_stop) else 0.0
    grid = np.linspace(lo, hi, points)
    dens = np.exp(0.5 * (grid * grid - max(lo * lo, hi * hi)))  # rescaled to avoid overflow
    cum = np.concatenate(([0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(grid))))
    total = cum[-1]
    at_u0 = float(np.interp(u0, grid, cum))
    from_stop = at_u0 if u_stop < u_target else total - at_u0
    return float(min(max(from_stop / total, 0.0), 1.0))


def ou_hit_probability_mc(
    u0: float,
    u_target: float,
    u_stop: float,
    b: float,
    max_ticks: int,
    paths: int = 2000,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Monte-Carlo P(reach ``u_target`` before ``u_stop`` AND within ``max_ticks``).

    Exact AR(1) discretisation in stationary-sd units:  u' = b u + sqrt(1 - b^2) * N(0, 1).
    A trade that times out counts as "not a win" (the time stop closes it).
    """
    rng = rng or np.random.default_rng()
    up = u_target > u_stop
    u = np.full(paths, u0, dtype=np.float64)
    alive = np.ones(paths, dtype=bool)
    wins = 0
    scale = math.sqrt(max(1.0 - b * b, 1e-12))
    for _ in range(max(int(max_ticks), 1)):
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break
        u[idx] = b * u[idx] + scale * rng.standard_normal(idx.size)
        hit_t = u[idx] >= u_target if up else u[idx] <= u_target
        hit_s = u[idx] <= u_stop if up else u[idx] >= u_stop
        wins += int(np.count_nonzero(hit_t))
        alive[idx[hit_t | hit_s]] = False
    return wins / paths


@functools.lru_cache(maxsize=20_000)
def _hit_prob_key(u0: float, u_target: float, u_stop: float, b: float, max_ticks: int) -> float:
    seed = hash((u0, u_target, u_stop, b, max_ticks)) & 0xFFFFFFFF
    return ou_hit_probability_mc(u0, u_target, u_stop, b, max_ticks, paths=1500, rng=np.random.default_rng(seed))


def cached_hit_probability(u0: float, u_target: float, u_stop: float, b: float, max_ticks: int) -> float:
    """Quantised + memoised Monte-Carlo first-passage probability (deterministic per key)."""
    q = lambda v: round(v * 20.0) / 20.0  # noqa: E731  (0.05 sd buckets)
    return _hit_prob_key(q(u0), q(u_target), q(u_stop), round(b, 3), min(int(max_ticks), 2_000))


# ------------------------------------------------------------------ VPIN
class VPINCalculator:
    """Volume-synchronised probability of informed trading with bulk volume classification."""

    def __init__(self, bucket_volume: float, num_buckets: int, sigma_window: int = 1_000, min_sigma_obs: int = 20) -> None:
        if bucket_volume <= 0 or num_buckets < 1:
            raise ValueError("bucket_volume > 0 and num_buckets >= 1 required")
        self.bucket_volume = bucket_volume
        self.num_buckets = num_buckets
        self._imbalances: Deque[float] = deque(maxlen=num_buckets)
        self._dp: Deque[float] = deque(maxlen=sigma_window)
        self._dp_sum = 0.0
        self._dp_sumsq = 0.0
        self._min_sigma_obs = min_sigma_obs
        self._cur_buy = 0.0
        self._cur_sell = 0.0
        self._last_price: Optional[float] = None

    def _push_dp(self, dp: float) -> None:
        if len(self._dp) == self._dp.maxlen:
            old = self._dp[0]
            self._dp_sum -= old
            self._dp_sumsq -= old * old
        self._dp.append(dp)
        self._dp_sum += dp
        self._dp_sumsq += dp * dp

    def _sigma(self) -> Optional[float]:
        n = len(self._dp)
        if n < self._min_sigma_obs:
            return None
        var = (self._dp_sumsq - self._dp_sum * self._dp_sum / n) / (n - 1)
        return math.sqrt(var) if var > 1e-24 else None

    @property
    def value(self) -> Optional[float]:
        if len(self._imbalances) < self.num_buckets:
            return None
        return sum(self._imbalances) / (len(self._imbalances) * self.bucket_volume)

    @property
    def buckets_filled(self) -> int:
        return len(self._imbalances)

    def update(self, price: float, volume: float) -> Optional[float]:
        if volume <= 0 or not math.isfinite(price):
            return self.value
        if self._last_price is None:
            self._last_price = price
            return self.value
        dp = price - self._last_price
        self._last_price = price
        self._push_dp(dp)
        sigma = self._sigma()
        if sigma is not None:
            buy_frac = norm_cdf(dp / sigma)
        else:  # tick-rule fallback while sigma is warming up
            buy_frac = 0.5 if dp == 0 else (1.0 if dp > 0 else 0.0)

        remaining = volume
        while remaining > 1e-12:
            space = self.bucket_volume - (self._cur_buy + self._cur_sell)
            take = min(space, remaining)
            self._cur_buy += take * buy_frac
            self._cur_sell += take * (1.0 - buy_frac)
            remaining -= take
            if self._cur_buy + self._cur_sell >= self.bucket_volume - 1e-12:
                self._imbalances.append(abs(self._cur_buy - self._cur_sell))
                self._cur_buy = self._cur_sell = 0.0
        return self.value


# ---------------------------------------------------------------- signals
class SignalAction(str, Enum):
    OPEN_LONG = "open_long"
    OPEN_SHORT = "open_short"
    CLOSE = "close"
    HOLD = "hold"


@dataclass(frozen=True, slots=True)
class Signal:
    symbol: str
    action: SignalAction
    ts: float
    price: float
    reason: str
    z: Optional[float] = None
    vpin: Optional[float] = None
    sentiment: float = 0.0
    alpha: Optional[float] = None
    mu: Optional[float] = None
    sigma_eq: Optional[float] = None
    stop_distance: float = 0.0
    take_profit: Optional[float] = None
    half_life: Optional[float] = None
    win_prob: Optional[float] = None  # OU first-passage probability (after haircut)
    payoff_ratio: Optional[float] = None  # (TP distance after spread) / (SL distance)
    kelly: Optional[float] = None  # full-Kelly fraction f* = p - (1 - p) / b
    expected_value: Optional[float] = None  # per unit volume, price units

    @property
    def side(self) -> Optional[Side]:
        if self.action is SignalAction.OPEN_LONG:
            return Side.BUY
        if self.action is SignalAction.OPEN_SHORT:
            return Side.SELL
        return None


class _SymbolState:
    __slots__ = ("p", "ou", "vpin", "last_entry_ts", "last", "ticks_in_position", "entry_half_life")

    def __init__(self, p: StrategyParams) -> None:
        self.p = p
        self.ou = OUEstimator(p.ou_window, p.ou_min_obs, p.ou_max_half_life_ticks, p.adf_critical)
        self.vpin = VPINCalculator(p.vpin_bucket_volume, p.vpin_num_buckets)
        self.last_entry_ts = 0.0
        self.last: Optional[Signal] = None
        self.ticks_in_position = 0
        self.entry_half_life: Optional[float] = None


class SignalEngine:
    """Combines OU z-score, VPIN toxicity and decayed news sentiment into trade signals."""

    def __init__(
        self,
        params: StrategyParams,
        sentiment_provider: Optional[SentimentProvider] = None,
        blackout_provider: Optional[BlackoutProvider] = None,
        session_provider: Optional[BlackoutProvider] = None,
        symbol_params: Optional[dict[str, StrategyParams]] = None,
    ) -> None:
        self.p = params
        self._sentiment = sentiment_provider or (lambda _s, _t: 0.0)
        self._blackout = blackout_provider or (lambda _s, _t: None)
        self._session = session_provider or (lambda _s, _t: None)
        self._symbol_params = symbol_params or {}
        self._states: dict[str, _SymbolState] = {}
        self._rng = np.random.default_rng(12345)
        self.max_model_win_prob = 0.8  # never trust the model beyond this

    def params_for(self, symbol: str) -> StrategyParams:
        return self._symbol_params.get(symbol, self.p)

    def _state(self, symbol: str) -> _SymbolState:
        st = self._states.get(symbol)
        if st is None:
            st = self._states[symbol] = _SymbolState(self.params_for(symbol))
        return st

    def symbols(self) -> list[str]:
        return list(self._states)

    def vpin(self, symbol: str) -> Optional[float]:
        st = self._states.get(symbol)
        return st.vpin.value if st else None

    def last_signal(self, symbol: str) -> Optional[Signal]:
        st = self._states.get(symbol)
        return st.last if st else None

    def on_snapshot(self, snap: MarketSnapshot, position_side: Optional[Side] = None) -> Optional[Signal]:
        """Update models; return a decision only for the newest live tick of a batch."""
        st = self._state(snap.symbol)
        price = snap.microprice
        ts = snap.recv_ns / 1e9
        st.ou.update(price)
        vpin = st.vpin.update(price, snap.trade_volume)
        if position_side is None:
            st.ticks_in_position = 0
        elif not snap.warmup:
            st.ticks_in_position += 1  # tick time, same clock as the OU half-life
        if snap.warmup or not snap.is_latest:
            return None

        try:
            sentiment = float(self._sentiment(snap.symbol, ts))
        except Exception:
            logger.exception("sentiment provider failed")
            sentiment = 0.0

        signal = self._decide(snap, st, price, ts, vpin, sentiment, position_side)
        st.last = signal
        return signal

    def _decide(
        self,
        snap: MarketSnapshot,
        st: _SymbolState,
        price: float,
        ts: float,
        vpin: Optional[float],
        sentiment: float,
        position_side: Optional[Side],
    ) -> Signal:
        p = st.p
        base = dict(symbol=snap.symbol, ts=ts, price=price, vpin=vpin, sentiment=sentiment)

        # ---- time stop: works even when the model is currently unusable ------
        if position_side is not None and p.time_stop_half_lives is not None:
            hl = st.entry_half_life or (p.ou_window / 2.0)
            if st.ticks_in_position > p.time_stop_half_lives * hl:
                return Signal(action=SignalAction.CLOSE, reason="time_stop", **base)

        ou = st.ou.fit()
        if ou is None:
            return Signal(action=SignalAction.HOLD, reason=st.ou.reason, **base)

        z = OUEstimator.zscore(price, ou)
        alpha = -z + p.sentiment_weight * sentiment
        model = dict(z=z, alpha=alpha, mu=ou.mu, sigma_eq=ou.sigma_eq)

        # ---- exits (always allowed, even under toxic flow) -------------
        if position_side is Side.BUY:
            if z >= -p.z_exit:
                return Signal(action=SignalAction.CLOSE, reason="mean_reverted", **base, **model)
            if sentiment <= -p.sentiment_veto:
                return Signal(action=SignalAction.CLOSE, reason="sentiment_reversal", **base, **model)
            return Signal(action=SignalAction.HOLD, reason="in_position", **base, **model)
        if position_side is Side.SELL:
            if z <= p.z_exit:
                return Signal(action=SignalAction.CLOSE, reason="mean_reverted", **base, **model)
            if sentiment >= p.sentiment_veto:
                return Signal(action=SignalAction.CLOSE, reason="sentiment_reversal", **base, **model)
            return Signal(action=SignalAction.HOLD, reason="in_position", **base, **model)

        # ---- entry filters ----------------------------------------------
        def hold(reason: str) -> Signal:
            return Signal(action=SignalAction.HOLD, reason=reason, **base, **model)

        if vpin is None:
            return hold("vpin_warming_up")
        if vpin >= p.vpin_threshold:
            return hold("toxic_flow")
        session_block = self._session(snap.symbol, ts)
        if session_block:
            return hold(session_block)
        blackout = self._blackout(snap.symbol, ts)
        if blackout:
            return hold(f"news_blackout:{blackout}")
        if ts - st.last_entry_ts < p.signal_cooldown_s:
            return hold("cooldown")

        if alpha >= p.z_entry and z < 0:
            side = Side.BUY
        elif alpha <= -p.z_entry and z > 0:
            side = Side.SELL
        else:
            return hold("no_edge")

        if side is Side.BUY and sentiment <= -p.sentiment_veto:
            return hold("sentiment_veto")
        if side is Side.SELL and sentiment >= p.sentiment_veto:
            return hold("sentiment_veto")

        edge = abs(price - ou.mu)
        if edge < p.min_edge_to_spread * snap.spread:
            return hold("edge_below_cost")

        # ---- trade economics: TP at mu, SL at stop_sigma_mult * sigma_eq ----------
        sign = side.sign
        entry = snap.ask if side is Side.BUY else snap.bid
        stop_distance = max(p.stop_sigma_mult * ou.sigma_eq, 2.0 * snap.spread)
        win = sign * (ou.mu - entry)  # exit on the opposite side of the book at mu
        loss = stop_distance
        if win <= 0:
            return hold("edge_below_cost")
        payoff = win / loss

        # OU state in stationary-sd units, evaluated on the side the exit order watches
        half = 0.5 * snap.spread
        sl_price = entry - sign * stop_distance
        u0 = (snap.mid - ou.mu) / ou.sigma_eq
        u_target = sign * half / ou.sigma_eq
        u_stop = (sl_price + sign * half - ou.mu) / ou.sigma_eq
        horizon = (p.time_stop_half_lives or 10.0) * ou.half_life
        p_raw = cached_hit_probability(u0, u_target, u_stop, ou.b, int(math.ceil(horizon)))
        win_prob = min(max(p_raw - p.win_prob_haircut, 0.0), self.max_model_win_prob)
        ev = win_prob * win - (1.0 - win_prob) * loss
        kelly = win_prob - (1.0 - win_prob) / payoff
        econ = dict(half_life=ou.half_life, win_prob=win_prob, payoff_ratio=payoff, kelly=kelly, expected_value=ev)
        if p.require_positive_ev and (ev <= 0 or kelly <= 0):
            return Signal(action=SignalAction.HOLD, reason="negative_ev", **base, **model, **econ)

        st.last_entry_ts = ts
        st.entry_half_life = ou.half_life
        action = SignalAction.OPEN_LONG if side is Side.BUY else SignalAction.OPEN_SHORT
        return Signal(
            action=action,
            reason=f"ou_reversion hl={ou.half_life:.0f}t p={win_prob:.2f} b={payoff:.2f}",
            stop_distance=stop_distance,
            take_profit=ou.mu,
            **base,
            **model,
            **econ,
        )
