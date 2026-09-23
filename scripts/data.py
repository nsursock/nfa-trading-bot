"""Synthetic crypto-perp tape for the HRL env. Pure MLX.

One perpetual, `n_envs` independent episodes, 24/7 bars (no session gaps).
Price is GBM on a substep grid collapsed to OHLC; volume is lognormal;
funding is an OU (crypto funding rate) plus a regime bias. Optional AR(1)
alpha is latent and only shows up by shifting each bar's drift, log-volume,
and funding — never as its own column.

Crypto market regimes (calm / chop / bull / bear / squeeze / cascade) drive
per-bar drift, vol scale, volume, funding bias, and jump risk. They are
latent like alpha: stored on `Market.regime` for diagnostics/viz, but absent
from `features()` / `observations()`. Use `regime=None` for a flat GBM with
the base `mu`/`sigma`; `regime="bull"` (etc.) to lock one regime; or
`regime="mixed"` for a persistent Markov switch across all six.

Bars line up on t = 0 .. n_bars-1. `open[:, 0] = s0` and
`open[:, t] = close[:, t-1]`. mu and sigma are per unit time; a bar lasts
`dt` (dt=1 means "per bar", e.g. one crypto candle).

The policy must not be fed the bar it is scored on. `decisions()` lags:

    obs[t]      completed bar t
    ret[t]      close[t+1] / close[t] - 1
    funding[t]  funding[t+1]
    next_obs[t] completed bar t+1

Reward for a signed position (long > 0):

    position * ret[t] - position * funding[t]

Positive funding charges longs (typical crypto-perp convention).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, NamedTuple

import mlx.core as mx
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CHANNELS = ("open", "high", "low", "close", "volume", "funding")
OBS_CHANNELS = ("log_ret", "log_high", "log_low", "log_volume", "funding")

# Crypto-perp regimes. Indices are stable and match Market.regime values.
REGIMES = ("calm", "chop", "bull", "bear", "squeeze", "cascade")
RegimeName = Literal["calm", "chop", "bull", "bear", "squeeze", "cascade"]
RegimeMode = RegimeName | Literal["mixed"]

# Stationary prior for mixed-Markov draws (and for the first bar).
REGIME_PRIOR = (0.12, 0.30, 0.20, 0.20, 0.09, 0.09)


@dataclass(frozen=True)
class RegimeKit:
    """Overlays applied on top of MarketSpec when a regime is active."""

    mu: float              # absolute drift contribution (per unit time)
    sigma_scale: float     # multiplies spec.sigma
    volume_shift: float    # added to log-volume mean
    funding_bias: float    # added to the OU funding level
    jump_p: float          # per-bar jump probability
    jump_mu: float         # mean jump in log-price (last substep)


REGIME_KITS: dict[str, RegimeKit] = {
    "calm": RegimeKit(0.0, 0.45, -0.9, 0.0, 0.0, 0.0),
    "chop": RegimeKit(0.0, 0.90, 0.0, 0.0, 0.0, 0.0),
    "bull": RegimeKit(0.0020, 1.15, 0.45, 1.0e-4, 0.012, 0.010),
    "bear": RegimeKit(-0.0020, 1.30, 0.55, -1.0e-4, 0.018, -0.012),
    "squeeze": RegimeKit(0.0045, 1.90, 1.30, 3.0e-4, 0.040, 0.025),
    "cascade": RegimeKit(-0.0055, 2.50, 1.70, -3.0e-4, 0.060, -0.045),
}


class MarketSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_envs: int = Field(default=1, ge=1)
    n_bars: int = Field(default=256, ge=2)
    seed: int = Field(default=0, ge=0)

    # Crypto-perp defaults: one bar ≈ one candle, vol a few percent.
    s0: float = Field(default=100.0, gt=0)
    mu: float = 0.0
    sigma: float = Field(default=0.018, ge=0)
    dt: float = Field(default=1.0, gt=0)
    n_substeps: int = Field(default=16, ge=1)

    volume_mu: float = 0.0
    volume_sigma: float = Field(default=0.55, ge=0)

    funding_theta: float = Field(default=0.20, gt=0)
    funding_mu: float = 0.0
    funding_sigma: float = Field(default=5e-5, ge=0)
    funding_x0: float | None = None

    alpha_phi: float | None = None
    alpha_sigma: float = Field(default=0.004, ge=0)
    alpha_x0: float | None = None

    # None = flat base GBM; a name = locked; "mixed" = Markov across REGIMES.
    regime: RegimeMode | None = "mixed"
    regime_persist: float = Field(default=0.97, gt=0.0, lt=1.0)

    @field_validator("alpha_phi")
    @classmethod
    def _stationary_phi(cls, v: float | None) -> float | None:
        if v is not None and abs(v) >= 1.0:
            raise ValueError("alpha_phi must lie in (-1, 1)")
        return v

    @model_validator(mode="after")
    def _check_regime(self) -> MarketSpec:
        if self.regime is not None and self.regime != "mixed" and self.regime not in REGIME_KITS:
            raise ValueError(f"regime must be None, 'mixed', or one of {REGIMES}")
        return self


class Market(NamedTuple):
    open: mx.array       # (n_envs, n_bars) float32
    high: mx.array
    low: mx.array
    close: mx.array
    volume: mx.array
    funding: mx.array
    regime: mx.array | None  # (n_envs, n_bars) int32 into REGIMES; None if flat

    def features(self) -> mx.array:
        """Raw channels, shape (n_envs, n_bars, 6), order `CHANNELS`."""
        return mx.stack(
            [self.open, self.high, self.low, self.close, self.volume, self.funding],
            axis=-1,
        )

    def channels(self) -> tuple[str, ...]:
        return CHANNELS

    def observations(self) -> mx.array:
        """Stationary view of every completed bar, (n_envs, n_bars, 5)."""
        return mx.stack(
            [
                mx.log(self.close / self.open),
                mx.log(self.high / self.close),
                mx.log(self.low / self.close),
                mx.log(self.volume),
                self.funding,
            ],
            axis=-1,
        ).astype(mx.float32)

    def decisions(self) -> "Decisions":
        """Act on bar t, mark bar t+1. See the module docstring."""
        full = self.observations()
        ret = self.close[:, 1:] / self.close[:, :-1] - 1.0
        return Decisions(
            obs=full[:, :-1, :],
            next_obs=full[:, 1:, :],
            ret=ret.astype(mx.float32),
            funding=self.funding[:, 1:],
        )

    def regime_name(self, env: int, bar: int) -> str | None:
        if self.regime is None:
            return None
        return REGIMES[int(self.regime[env, bar].item())]


class Decisions(NamedTuple):
    obs: mx.array       # (n_envs, n_bars - 1, 5)
    next_obs: mx.array  # (n_envs, n_bars - 1, 5)
    ret: mx.array       # (n_envs, n_bars - 1)
    funding: mx.array   # (n_envs, n_bars - 1)


def generate(spec: MarketSpec) -> Market:
    key = mx.random.key(spec.seed)
    k_px, k_vol, k_fund, k_alpha, k_reg, k_jump = mx.random.split(key, 6)
    regime = _sample_regimes(spec, k_reg)
    alpha = _latent_alpha(spec, k_alpha)
    overlays = _regime_overlays(spec, regime)
    open_, high, low, close = _gbm_ohlc(spec, k_px, k_jump, alpha, overlays)
    volume = _lognormal_volume(spec, k_vol, alpha, overlays)
    funding = _ou_funding(spec, k_fund, alpha, overlays)
    market = Market(open_, high, low, close, volume, funding, regime)
    leaves = [market.open, market.high, market.low, market.close,
              market.volume, market.funding]
    if market.regime is not None:
        leaves.append(market.regime)
    mx.eval(*leaves)
    return market


def _sample_regimes(spec: MarketSpec, key) -> mx.array | None:
    if spec.regime is None:
        return None
    n, bars = spec.n_envs, spec.n_bars
    if spec.regime != "mixed":
        idx = REGIMES.index(spec.regime)
        return mx.full((n, bars), idx, dtype=mx.int32)

    logits = mx.log(mx.array(REGIME_PRIOR, dtype=mx.float32))
    keys = mx.random.split(key, bars + 1)
    state = mx.random.categorical(logits, shape=(n,), key=keys[0]).astype(mx.int32)
    rows = [state]
    persist = spec.regime_persist
    for t in range(1, bars):
        k_stay, k_draw = mx.random.split(keys[t])
        stay = mx.random.bernoulli(persist, shape=(n,), key=k_stay)
        nxt = mx.random.categorical(logits, shape=(n,), key=k_draw).astype(mx.int32)
        state = mx.where(stay, state, nxt)
        rows.append(state)
        if t % 64 == 0:
            mx.eval(state)
    return mx.stack(rows, axis=1)


class _Overlays(NamedTuple):
    mu: mx.array            # (n_envs, n_bars)
    sigma: mx.array
    volume_shift: mx.array
    funding_bias: mx.array
    jump_p: mx.array
    jump_mu: mx.array


def _regime_overlays(spec: MarketSpec, regime: mx.array | None) -> _Overlays:
    n, bars = spec.n_envs, spec.n_bars
    if regime is None:
        z = mx.zeros((n, bars), dtype=mx.float32)
        return _Overlays(
            mu=mx.full((n, bars), spec.mu, dtype=mx.float32),
            sigma=mx.full((n, bars), spec.sigma, dtype=mx.float32),
            volume_shift=z,
            funding_bias=z,
            jump_p=z,
            jump_mu=z,
        )
    mu = mx.array([REGIME_KITS[r].mu for r in REGIMES], dtype=mx.float32)
    scale = mx.array([REGIME_KITS[r].sigma_scale for r in REGIMES], dtype=mx.float32)
    vshift = mx.array([REGIME_KITS[r].volume_shift for r in REGIMES], dtype=mx.float32)
    fbias = mx.array([REGIME_KITS[r].funding_bias for r in REGIMES], dtype=mx.float32)
    jp = mx.array([REGIME_KITS[r].jump_p for r in REGIMES], dtype=mx.float32)
    jm = mx.array([REGIME_KITS[r].jump_mu for r in REGIMES], dtype=mx.float32)
    return _Overlays(
        mu=spec.mu + mu[regime],
        sigma=spec.sigma * scale[regime],
        volume_shift=vshift[regime],
        funding_bias=fbias[regime],
        jump_p=jp[regime],
        jump_mu=jm[regime],
    )


def _latent_alpha(spec: MarketSpec, key) -> mx.array:
    n, bars = spec.n_envs, spec.n_bars
    if spec.alpha_phi is None:
        return mx.zeros((n, bars), dtype=mx.float32)
    phi = spec.alpha_phi
    std = (0.0 if abs(phi) >= 1.0 - 1e-12
           else spec.alpha_sigma / math.sqrt(1.0 - phi * phi))
    x0 = _initial(n, spec.alpha_x0, 0.0, std, key)
    z = mx.random.normal(shape=(n, bars), dtype=mx.float32, key=_innov_key(key))
    return _ar1(phi, spec.alpha_sigma, z, x0)


def _gbm_ohlc(
    spec: MarketSpec, key, jump_key, alpha: mx.array, overlays: _Overlays,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    n, bars, sub = spec.n_envs, spec.n_bars, spec.n_substeps
    dt_sub = spec.dt / sub
    k_j, k_jz = mx.random.split(jump_key)
    z = mx.random.normal(shape=(n, bars * sub), dtype=mx.float32, key=key)
    drift = overlays.mu + alpha
    sigma = overlays.sigma
    drift_sub = mx.repeat(drift, sub, axis=1)
    sigma_sub = mx.repeat(sigma, sub, axis=1)
    log_inc = (
        (drift_sub - 0.5 * sigma_sub ** 2) * dt_sub
        + sigma_sub * math.sqrt(dt_sub) * z
    ).reshape(n, bars, sub)

    hit = mx.random.bernoulli(overlays.jump_p, key=k_j)
    jump = hit.astype(mx.float32) * (
        overlays.jump_mu
        + 0.35 * mx.abs(overlays.jump_mu)
        * mx.random.normal(shape=(n, bars), dtype=mx.float32, key=k_jz)
    )
    # Dump the jump into the last substep so OHLC wicks catch it.
    last = log_inc[:, :, -1] + jump
    log_inc = mx.concatenate([log_inc[:, :, :-1], last[:, :, None]], axis=2)

    log_s = math.log(spec.s0) + mx.cumsum(log_inc.reshape(n, bars * sub), axis=1)
    path = mx.exp(log_s).reshape(n, bars, sub)
    close = path[:, :, -1]
    open_ = mx.concatenate(
        [mx.full((n, 1), spec.s0, dtype=mx.float32), close[:, :-1]],
        axis=1,
    )
    high = mx.maximum(open_, mx.max(path, axis=2))
    low = mx.minimum(open_, mx.min(path, axis=2))
    return (open_.astype(mx.float32), high.astype(mx.float32),
            low.astype(mx.float32), close.astype(mx.float32))


def _lognormal_volume(
    spec: MarketSpec, key, alpha: mx.array, overlays: _Overlays,
) -> mx.array:
    z = mx.random.normal(shape=(spec.n_envs, spec.n_bars), dtype=mx.float32, key=key)
    return mx.exp(
        spec.volume_mu + overlays.volume_shift + alpha + spec.volume_sigma * z
    ).astype(mx.float32)


def _ou_funding(
    spec: MarketSpec, key, alpha: mx.array, overlays: _Overlays,
) -> mx.array:
    """Exact OU to funding_mu, then regime bias + latent alpha."""
    decay = math.exp(-spec.funding_theta * spec.dt)
    var = (spec.funding_sigma**2
           * (1.0 - math.exp(-2.0 * spec.funding_theta * spec.dt))
           / (2.0 * spec.funding_theta))
    scale = math.sqrt(var)
    x0 = _initial(spec.n_envs, spec.funding_x0, spec.funding_mu,
                  spec.funding_sigma / math.sqrt(2.0 * spec.funding_theta), key)
    z = mx.random.normal(shape=(spec.n_envs, spec.n_bars), dtype=mx.float32, key=_innov_key(key))
    ou = _ar1(decay, scale, z, x0 - spec.funding_mu) + spec.funding_mu
    return (ou + overlays.funding_bias + alpha).astype(mx.float32)


def _initial(n: int, x0: float | None, mean: float, std: float, key) -> mx.array:
    if x0 is not None:
        return mx.full((n,), x0, dtype=mx.float32)
    k_x0, _ = mx.random.split(key)
    if std == 0.0:
        return mx.full((n,), mean, dtype=mx.float32)
    draw = mx.random.normal(shape=(n,), dtype=mx.float32, key=k_x0)
    return (mean + std * draw).astype(mx.float32)


def _innov_key(key):
    _, k_z = mx.random.split(key)
    return k_z


def _chunk_len(phi: float) -> int:
    if abs(phi) < 1e-12:
        return 1
    n = int(40.0 / -math.log(abs(phi)))
    return min(4096, max(1, n))


def _ar1(phi: float, scale: float, z: mx.array, x0: mx.array) -> mx.array:
    """x[:, 0] = x0, x[:, t] = phi * x[:, t-1] + scale * z[:, t]. z[:, 0] unused."""
    n, length = z.shape
    out = mx.zeros((n, length), dtype=mx.float32)
    x0 = x0.astype(mx.float32)
    out[:, 0] = x0
    state = x0
    step = _chunk_len(phi)
    t = 1
    while t < length:
        size = min(step, length - t)
        chunk, state = _ar_chunk(phi, scale, z[:, t:t + size], state)
        out[:, t:t + size] = chunk
        t += size
        mx.eval(out, state)
    return out


def _ar_chunk(phi: float, scale: float, z: mx.array, x0: mx.array):
    if abs(phi) < 1e-12:
        y = (scale * z).astype(mx.float32)
        return y, y[:, -1]
    k = mx.arange(1, z.shape[1] + 1, dtype=mx.float32)
    ak = (phi ** k).astype(mx.float32)
    summed = mx.cumsum(z / ak, axis=1)
    y = (ak * x0[:, None] + scale * ak * summed).astype(mx.float32)
    return y, y[:, -1]
