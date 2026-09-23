"""Vectorized multi-crypto leveraged trading env (pure MLX).

Observations come from ``scripts.data`` synthetic tapes on two Binance-style
timeframes. The worker acts every low-TF bar; the manager reads high-TF
features and sets a goal that conditions the worker.

Worker action per asset (continuous, in [-1, 1]):
  side, leverage, collateral_frac, take_profit, stop_loss

Liquidations use isolated margin vs a maintenance-margin rate. Reward
encourages steady equity growth with quick drawdown recoveries.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

import mlx.core as mx
from pydantic import BaseModel, ConfigDict, Field, model_validator

from scripts.data import OBS_CHANNELS, MarketSpec, generate
from scripts.envs.base import StepResult
from scripts.ledger import TradeLedger

# Binance kline intervals (minutes). High TF must be an integer multiple of low.
TF_MINUTES: dict[str, int] = {
    "1m": 1,
    "3m": 3,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "6h": 360,
    "8h": 480,
    "12h": 720,
    "1d": 1440,
    "3d": 4320,
    "1w": 10080,
}

BinanceTF = Literal[
    "1m", "3m", "5m", "15m", "30m",
    "1h", "2h", "4h", "6h", "8h", "12h",
    "1d", "3d", "1w",
]

N_OBS_CH = len(OBS_CHANNELS)  # 5


class TradingEnvConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_envs: int = Field(default=1, ge=1)
    n_assets: int = Field(default=2, ge=1)
    seed: int = Field(default=0, ge=0)
    # Per-asset symbols (Binance-style). Padded/truncated to n_assets.
    symbols: list[str] = Field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    # Synthetic wall-clock for ledger datetimes (ISO UTC).
    clock_start: str = "2024-01-01T00:00:00Z"

    low_tf: BinanceTF = "5m"
    high_tf: BinanceTF = "1h"
    n_low_bars: int = Field(default=512, ge=8)
    lookback_low: int = Field(default=8, ge=1)
    lookback_high: int = Field(default=4, ge=1)

    initial_balance: float = Field(default=10_000.0, gt=0)
    min_leverage: float = Field(default=1.0, ge=1.0)
    max_leverage: float = Field(default=20.0, gt=1.0)
    min_collateral: float = Field(default=0.02, gt=0.0, le=1.0)
    max_collateral: float = Field(default=0.25, gt=0.0, le=1.0)
    min_tp: float = Field(default=0.005, gt=0.0)
    max_tp: float = Field(default=0.08, gt=0.0)
    min_sl: float = Field(default=0.005, gt=0.0)
    max_sl: float = Field(default=0.05, gt=0.0)
    risk_per_trade: float = Field(default=0.02, gt=0.0, le=1.0)
    maintenance_margin: float = Field(default=0.005, ge=0.0, lt=1.0)
    fee_rate: float = Field(default=0.0004, ge=0.0)
    side_deadzone: float = Field(default=0.25, gt=0.0, lt=1.0)

    liquidation_penalty: float = 1.0
    reward_pnl_scale: float = 1.0
    reward_dd_penalty: float = 0.5
    reward_recovery_bonus: float = 0.35
    reward_vol_penalty: float = 0.05
    reward_hold_cost: float = 0.0

    manager_horizon: int = Field(default=12, ge=1)
    goal_dim: int = Field(default=4, ge=1)

    # Forwarded into MarketSpec (flat GBM fields; regime still available).
    market_mu: float = 0.0
    market_sigma: float = Field(default=0.012, ge=0)
    market_regime: str | None = "mixed"
    market_dt: float = Field(default=1.0, gt=0)
    market_n_substeps: int = Field(default=8, ge=1)

    @model_validator(mode="after")
    def _check_tfs_and_ranges(self) -> TradingEnvConfig:
        if self.low_tf not in TF_MINUTES or self.high_tf not in TF_MINUTES:
            raise ValueError("low_tf/high_tf must be Binance intervals")
        lo, hi = TF_MINUTES[self.low_tf], TF_MINUTES[self.high_tf]
        if hi % lo != 0 or hi // lo < 2:
            raise ValueError(
                f"high_tf={self.high_tf!r} must be an integer multiple of "
                f"low_tf={self.low_tf!r} with ratio >= 2"
            )
        if self.min_leverage >= self.max_leverage:
            raise ValueError("min_leverage must be < max_leverage")
        if self.min_collateral >= self.max_collateral:
            raise ValueError("min_collateral must be < max_collateral")
        if self.min_tp >= self.max_tp or self.min_sl >= self.max_sl:
            raise ValueError("min_tp/sl must be < max_tp/sl")
        need = self.lookback_low + self.tf_ratio * self.lookback_high + 2
        if self.n_low_bars < need:
            raise ValueError(
                f"n_low_bars={self.n_low_bars} too small for lookbacks "
                f"(need >= {need})"
            )
        # Normalize symbols to n_assets length.
        syms = list(self.symbols)
        if not syms:
            syms = [f"ASSET{i}USDT" for i in range(self.n_assets)]
        while len(syms) < self.n_assets:
            syms.append(f"ASSET{len(syms)}USDT")
        object.__setattr__(self, "symbols", syms[: self.n_assets])
        return self

    @property
    def tf_ratio(self) -> int:
        return TF_MINUTES[self.high_tf] // TF_MINUTES[self.low_tf]


def _unit_to_range(u: mx.array, lo: float, hi: float) -> mx.array:
    """Map [-1, 1] -> [lo, hi]."""
    return lo + (hi - lo) * 0.5 * (u + 1.0)


def _aggregate_high(
    open_: mx.array,
    high: mx.array,
    low: mx.array,
    close: mx.array,
    volume: mx.array,
    funding: mx.array,
    ratio: int,
) -> tuple[mx.array, ...]:
    """Collapse low-TF OHLC to high-TF. Shapes (E, A, T) -> (E, A, T//ratio)."""
    e, a, t = open_.shape
    n = t // ratio
    usable = n * ratio
    o = open_[:, :, :usable].reshape(e, a, n, ratio)
    h = high[:, :, :usable].reshape(e, a, n, ratio)
    l = low[:, :, :usable].reshape(e, a, n, ratio)
    c = close[:, :, :usable].reshape(e, a, n, ratio)
    v = volume[:, :, :usable].reshape(e, a, n, ratio)
    f = funding[:, :, :usable].reshape(e, a, n, ratio)
    return (
        o[:, :, :, 0],
        mx.max(h, axis=3),
        mx.min(l, axis=3),
        c[:, :, :, -1],
        mx.sum(v, axis=3),
        f[:, :, :, -1],
    )


def _obs_from_ohlcv(
    open_: mx.array, high: mx.array, low: mx.array, close: mx.array,
    volume: mx.array, funding: mx.array,
) -> mx.array:
    """Stationary bar features, last axis = OBS_CHANNELS."""
    return mx.stack(
        [
            mx.log(close / open_),
            mx.log(high / close),
            mx.log(low / close),
            mx.log(mx.maximum(volume, 1e-8)),
            funding,
        ],
        axis=-1,
    ).astype(mx.float32)


class TradingEnv:
    """VecEnv-style leveraged multi-asset crypto env."""

    is_discrete = False

    def __init__(self, config: TradingEnvConfig | None = None, **kwargs):
        self.cfg = config if config is not None else TradingEnvConfig(**kwargs)
        cfg = self.cfg
        self.num_envs = cfg.n_envs
        self.n_assets = cfg.n_assets
        self.action_dim = 5 * cfg.n_assets
        self.action_low = -1.0
        self.action_high = 1.0
        self.goal_dim = cfg.goal_dim
        self.manager_horizon = cfg.manager_horizon

        # Portfolio features: equity_n, cash_n, peak_n, dd, + per asset
        # (side, lev_n, coll_n, upnl_n)
        self._port_dim = 4 + 4 * cfg.n_assets
        self.worker_market_dim = (
            cfg.n_assets * cfg.lookback_low * N_OBS_CH
            + cfg.n_assets * cfg.lookback_high * N_OBS_CH
        )
        self.manager_obs_dim = (
            cfg.n_assets * cfg.lookback_high * N_OBS_CH + self._port_dim
        )
        self.worker_obs_dim = (
            self.worker_market_dim + self._port_dim + cfg.goal_dim
        )
        # Default env.obs_dim is the worker view (what the step loop sees).
        self.obs_dim = self.worker_obs_dim

        self._key = mx.random.key(cfg.seed)
        self._t = mx.zeros((cfg.n_envs,), dtype=mx.int32)
        self._steps = mx.zeros((cfg.n_envs,), dtype=mx.int32)
        self._ep_ret = mx.zeros((cfg.n_envs,), dtype=mx.float32)
        self._ep_len = mx.zeros((cfg.n_envs,), dtype=mx.int32)
        self._goal = mx.zeros((cfg.n_envs, cfg.goal_dim), dtype=mx.float32)

        self._cash = mx.full((cfg.n_envs,), cfg.initial_balance, dtype=mx.float32)
        self._peak = mx.full((cfg.n_envs,), cfg.initial_balance, dtype=mx.float32)
        # Per-asset position state
        z = mx.zeros((cfg.n_envs, cfg.n_assets), dtype=mx.float32)
        self._side = z  # -1 / 0 / 1
        self._entry = z
        self._coll = z
        self._lev = z
        self._tp = z
        self._sl = z

        # Filled on reset
        self._low_obs = None
        self._high_obs = None
        self._low_open = None
        self._low_high = None
        self._low_low = None
        self._low_close = None
        self._low_funding = None
        self._n_decision = 0
        self._max_episode_steps = 0
        self._start_t = 0
        self.ledger: TradeLedger | None = None

        self._step_fn = mx.compile(self._physics)

    def _split_key(self):
        self._key, sub = mx.random.split(self._key)
        return sub

    def attach_ledger(self, path: str) -> TradeLedger:
        """Enable trade-history CSV (intended for test only)."""
        cfg = self.cfg
        raw = cfg.clock_start.replace("Z", "+00:00")
        try:
            start = datetime.fromisoformat(raw)
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
        except ValueError:
            start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        self.ledger = TradeLedger(
            path,
            symbols=cfg.symbols,
            bar_minutes=TF_MINUTES[cfg.low_tf],
            clock_start=start,
        )
        return self.ledger

    def detach_ledger(self) -> None:
        if self.ledger is not None:
            self.ledger.close()
            self.ledger = None

    def set_goal(self, goal: mx.array) -> None:
        """Manager call: set (n_envs, goal_dim) goal in [-1, 1]."""
        g = mx.asarray(goal, dtype=mx.float32)
        if g.ndim == 1:
            g = mx.broadcast_to(g[None, :], (self.num_envs, self.goal_dim))
        self._goal = mx.clip(g, -1.0, 1.0)

    def _generate_books(self, key) -> None:
        cfg = self.cfg
        keys = mx.random.split(key, cfg.n_assets + 1)
        opens, highs, lows, closes, vols, funds = [], [], [], [], [], []
        for i in range(cfg.n_assets):
            seed_i = int(mx.random.randint(0, 2**31 - 1, (), key=keys[i]).item())
            spec = MarketSpec(
                n_envs=cfg.n_envs,
                n_bars=cfg.n_low_bars,
                seed=seed_i,
                mu=cfg.market_mu,
                sigma=cfg.market_sigma,
                dt=cfg.market_dt,
                n_substeps=cfg.market_n_substeps,
                regime=cfg.market_regime,  # type: ignore[arg-type]
            )
            m = generate(spec)
            opens.append(m.open)
            highs.append(m.high)
            lows.append(m.low)
            closes.append(m.close)
            vols.append(m.volume)
            funds.append(m.funding)
        # (E, A, T)
        open_ = mx.stack(opens, axis=1)
        high = mx.stack(highs, axis=1)
        low = mx.stack(lows, axis=1)
        close = mx.stack(closes, axis=1)
        volume = mx.stack(vols, axis=1)
        funding = mx.stack(funds, axis=1)
        mx.eval(open_, high, low, close, volume, funding)

        ratio = cfg.tf_ratio
        ho, hh, hl, hc, hv, hf = _aggregate_high(
            open_, high, low, close, volume, funding, ratio
        )
        self._low_obs = _obs_from_ohlcv(open_, high, low, close, volume, funding)
        self._high_obs = _obs_from_ohlcv(ho, hh, hl, hc, hv, hf)
        self._low_open = open_
        self._low_high = high
        self._low_low = low
        self._low_close = close
        self._low_funding = funding
        # Act on completed bar t, mark on t+1 (same lag as data.decisions).
        self._n_decision = cfg.n_low_bars - 1
        self._start_t = max(
            cfg.lookback_low - 1,
            ratio * cfg.lookback_high - 1,
        )
        self._max_episode_steps = self._n_decision - self._start_t

    def _upnl(self, mark: mx.array) -> mx.array:
        """Unrealized PnL (E, A) from mark prices."""
        entry = mx.maximum(self._entry, 1e-8)
        ret = (mark / entry) - 1.0
        # long: +ret * notional; short: -ret * notional; flat: 0
        notional = self._coll * self._lev
        return self._side * notional * ret

    def _equity(self, upnl: mx.array) -> mx.array:
        return self._cash + mx.sum(self._coll + upnl, axis=1)

    def _portfolio_features(self, upnl: mx.array, equity: mx.array) -> mx.array:
        cfg = self.cfg
        inv = 1.0 / cfg.initial_balance
        peak = mx.maximum(self._peak, equity)
        dd = (peak - equity) / mx.maximum(peak, 1e-8)
        lev_n = self._lev / cfg.max_leverage
        coll_n = self._coll * inv
        upnl_n = upnl * inv
        per = mx.concatenate(
            [self._side, lev_n, coll_n, upnl_n], axis=1
        )
        base = mx.stack(
            [
                equity * inv,
                self._cash * inv,
                peak * inv,
                dd,
            ],
            axis=1,
        )
        return mx.concatenate([base, per], axis=1).astype(mx.float32)

    def _gather_lookback(self, series: mx.array, t: mx.array, lb: int) -> mx.array:
        """series (E,A,T,C), t (E,) -> (E, A*lb*C) ending at t inclusive."""
        e, a, _, c = series.shape
        # Build indices (E, lb)
        offs = mx.arange(lb, dtype=mx.int32)
        idx = t[:, None] - (lb - 1) + offs[None, :]
        idx = mx.clip(idx, 0, series.shape[2] - 1)
        # Gather per env: for simplicity loop assets with take_along_axis on T
        # Expand idx to (E, A, lb, C)
        idx_exp = mx.broadcast_to(idx[:, None, :, None], (e, a, lb, c))
        # series gathered: use a flat gather via one-hot-ish — MLX take_along
        # wants same ndim. Reshape series to (E, A, T, C) and gather on axis=2.
        gathered = mx.take_along_axis(series, idx_exp, axis=2)
        return gathered.reshape(e, a * lb * c)

    def _market_obs(self, t: mx.array) -> tuple[mx.array, mx.array]:
        cfg = self.cfg
        low = self._gather_lookback(self._low_obs, t, cfg.lookback_low)
        # High-TF index of the last fully closed high bar at low time t
        high_t = (t + 1) // cfg.tf_ratio - 1
        high_t = mx.maximum(high_t, 0)
        high = self._gather_lookback(self._high_obs, high_t, cfg.lookback_high)
        return low, high

    def manager_obs(self) -> mx.array:
        mark = self._marks(self._t)
        upnl = self._upnl(mark)
        equity = self._equity(upnl)
        _, high = self._market_obs(self._t)
        port = self._portfolio_features(upnl, equity)
        return mx.concatenate([high, port], axis=1)

    def _marks(self, t: mx.array) -> mx.array:
        """Close prices at per-env time t -> (E, A)."""
        e, a, T = self._low_close.shape
        t_clip = mx.clip(t, 0, T - 1)
        idx = mx.broadcast_to(t_clip[:, None, None], (e, a, 1))
        return mx.take_along_axis(self._low_close, idx, axis=2).squeeze(2)

    def _bar_path(
        self, t: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
        """OHLC + funding for the *next* bar (t+1) that we mark against."""
        e, a, _T = self._low_close.shape
        nxt = mx.clip(t + 1, 0, self._low_close.shape[2] - 1)
        idx = mx.broadcast_to(nxt[:, None, None], (e, a, 1))
        o = mx.take_along_axis(self._low_open, idx, axis=2).squeeze(2)
        h = mx.take_along_axis(self._low_high, idx, axis=2).squeeze(2)
        l = mx.take_along_axis(self._low_low, idx, axis=2).squeeze(2)
        c = mx.take_along_axis(self._low_close, idx, axis=2).squeeze(2)
        f = mx.take_along_axis(self._low_funding, idx, axis=2).squeeze(2)
        return o, h, l, c, f

    def _worker_obs_from_state(self, t: mx.array) -> mx.array:
        mark = self._marks(t)
        upnl = self._upnl(mark)
        equity = self._equity(upnl)
        low, high = self._market_obs(t)
        port = self._portfolio_features(upnl, equity)
        return mx.concatenate([low, high, port, self._goal], axis=1)

    def achieved_goal(self) -> mx.array:
        """Worker-relative achievement vector in [-1ish, 1ish], dim=goal_dim.

        Uses equity return, drawdown (negated), net exposure, and mean leverage.
        """
        cfg = self.cfg
        mark = self._marks(self._t)
        upnl = self._upnl(mark)
        equity = self._equity(upnl)
        ret = (equity / cfg.initial_balance) - 1.0
        peak = mx.maximum(self._peak, equity)
        dd = (peak - equity) / mx.maximum(peak, 1e-8)
        exposure = mx.mean(self._side, axis=1)
        lev = mx.mean(self._lev, axis=1) / cfg.max_leverage
        feats = [ret * 10.0, 1.0 - 2.0 * dd, exposure, 2.0 * lev - 1.0]
        # Pad / truncate to goal_dim
        while len(feats) < cfg.goal_dim:
            feats.append(mx.zeros_like(ret))
        stacked = mx.stack(feats[: cfg.goal_dim], axis=1)
        return mx.clip(stacked, -1.0, 1.0).astype(mx.float32)

    def reset(self) -> mx.array:
        cfg = self.cfg
        self._generate_books(self._split_key())
        n = cfg.n_envs
        a = cfg.n_assets
        self._t = mx.full((n,), self._start_t, dtype=mx.int32)
        self._steps = mx.zeros((n,), dtype=mx.int32)
        self._ep_ret = mx.zeros((n,), dtype=mx.float32)
        self._ep_len = mx.zeros((n,), dtype=mx.int32)
        self._cash = mx.full((n,), cfg.initial_balance, dtype=mx.float32)
        self._peak = mx.full((n,), cfg.initial_balance, dtype=mx.float32)
        z = mx.zeros((n, a), dtype=mx.float32)
        self._side = z
        self._entry = z
        self._coll = z
        self._lev = z
        self._tp = z
        self._sl = z
        self._goal = mx.zeros((n, cfg.goal_dim), dtype=mx.float32)
        obs = self._worker_obs_from_state(self._t)
        mx.eval(obs, self._cash, self._t)
        return obs

    def _decode_actions(self, actions: mx.array, equity: mx.array):
        cfg = self.cfg
        a = cfg.n_assets
        act = mx.clip(actions.reshape(self.num_envs, a, 5), -1.0, 1.0)
        side_raw = act[:, :, 0]
        side = mx.where(
            side_raw > cfg.side_deadzone,
            mx.ones_like(side_raw),
            mx.where(
                side_raw < -cfg.side_deadzone,
                -mx.ones_like(side_raw),
                mx.zeros_like(side_raw),
            ),
        )
        lev = _unit_to_range(act[:, :, 1], cfg.min_leverage, cfg.max_leverage)
        coll_frac = _unit_to_range(
            act[:, :, 2], cfg.min_collateral, cfg.max_collateral
        )
        tp_pct = _unit_to_range(act[:, :, 3], cfg.min_tp, cfg.max_tp)
        sl_pct = _unit_to_range(act[:, :, 4], cfg.min_sl, cfg.max_sl)
        # Risk cap: coll * lev * sl_pct <= equity * risk_per_trade
        max_coll = (equity[:, None] * cfg.risk_per_trade) / mx.maximum(
            lev * sl_pct, 1e-8
        )
        want = equity[:, None] * coll_frac
        coll = mx.minimum(want, max_coll)
        return side, lev, coll, tp_pct, sl_pct

    def _physics(
        self,
        t, steps, ep_ret, ep_len, cash, peak,
        side, entry, coll, lev, tp, sl,
        actions, goal,
    ):
        cfg = self.cfg
        inv = 1.0 / cfg.initial_balance

        mark0 = self._marks(t)
        upnl0 = self._upnl_static(side, entry, coll, lev, mark0)
        equity0 = cash + mx.sum(coll + upnl0, axis=1)
        peak0 = mx.maximum(peak, equity0)
        dd0 = (peak0 - equity0) / mx.maximum(peak0, 1e-8)

        # Decode desired position for next bar
        d_side, d_lev, d_coll, d_tp, d_sl = self._decode_actions(actions, equity0)

        # Close everything that flips or goes flat (realize PnL into cash)
        o, h, l, c, funding = self._bar_path(t)
        fill = o
        close_mask = (mx.abs(side) > 0.5) & (
            (mx.abs(d_side) < 0.5) | (side * d_side < 0)
        )
        close_side = side
        close_entry = entry
        close_coll = coll
        close_lev = lev
        close_pnl = self._upnl_static(side, entry, coll, lev, fill)
        fee_close = cfg.fee_rate * coll * lev * close_mask.astype(mx.float32)
        cash = cash + mx.sum(
            (coll + close_pnl) * close_mask.astype(mx.float32) - fee_close,
            axis=1,
        )
        side = mx.where(close_mask, mx.zeros_like(side), side)
        coll = mx.where(close_mask, mx.zeros_like(coll), coll)
        lev = mx.where(close_mask, mx.zeros_like(lev), lev)
        entry = mx.where(close_mask, mx.zeros_like(entry), entry)
        tp = mx.where(close_mask, mx.zeros_like(tp), tp)
        sl = mx.where(close_mask, mx.zeros_like(sl), sl)

        # Open new positions where flat and desired non-flat
        open_mask = (mx.abs(side) < 0.5) & (mx.abs(d_side) > 0.5)
        new_coll = mx.where(open_mask, d_coll, mx.zeros_like(d_coll))
        total_new = mx.sum(new_coll, axis=1)
        scale = mx.minimum(
            mx.ones_like(total_new), cash / mx.maximum(total_new, 1e-8)
        )
        new_coll = new_coll * scale[:, None]
        fee_open = cfg.fee_rate * new_coll * d_lev
        cash = cash - mx.sum(new_coll + fee_open, axis=1)
        open_side = d_side
        open_lev = d_lev
        open_coll = new_coll
        open_px = fill
        side = mx.where(open_mask, d_side, side)
        coll = mx.where(open_mask, new_coll, coll)
        lev = mx.where(open_mask, d_lev, lev)
        entry = mx.where(open_mask, fill, entry)
        tp_open = mx.where(
            d_side > 0, fill * (1.0 + d_tp), fill * (1.0 - d_tp),
        )
        sl_open = mx.where(
            d_side > 0, fill * (1.0 - d_sl), fill * (1.0 + d_sl),
        )
        tp = mx.where(open_mask, tp_open, tp)
        sl = mx.where(open_mask, sl_open, sl)

        same = (mx.abs(side) > 0.5) & (side * d_side > 0) & (mx.abs(d_side) > 0.5)
        ref = mx.maximum(entry, 1e-8)
        tp_same = mx.where(
            d_side > 0, ref * (1.0 + d_tp), ref * (1.0 - d_tp),
        )
        sl_same = mx.where(
            d_side > 0, ref * (1.0 - d_sl), ref * (1.0 + d_sl),
        )
        tp = mx.where(same, tp_same, tp)
        sl = mx.where(same, sl_same, sl)

        # Intra-bar TP / SL / liquidation using high/low
        long_m = side > 0.5
        short_m = side < -0.5
        hit_tp = (long_m & (h >= tp)) | (short_m & (l <= tp))
        hit_sl = (long_m & (l <= sl)) | (short_m & (h >= sl))

        adverse = mx.where(long_m, l, mx.where(short_m, h, c))
        upnl_adv = self._upnl_static(side, entry, coll, lev, adverse)
        notional = coll * lev
        mm = notional * cfg.maintenance_margin
        liq = (mx.abs(side) > 0.5) & ((coll + upnl_adv) <= mm)

        # Priority: liquidation > take_profit > stop_loss
        liq_mask = liq
        tp_mask = hit_tp & ~liq
        sl_mask = hit_sl & ~liq & ~hit_tp
        exit_mask = (tp_mask | sl_mask | liq_mask) & (mx.abs(side) > 0.5)

        exit_side = side
        exit_entry = entry
        exit_coll = coll
        exit_lev = lev
        exit_px = mx.where(
            liq_mask,
            adverse,
            mx.where(tp_mask, tp, mx.where(sl_mask, sl, c)),
        )
        exit_pnl = self._upnl_static(side, entry, coll, lev, exit_px)
        exit_return = mx.where(
            liq_mask,
            mx.zeros_like(coll),
            coll + exit_pnl,
        )
        fee_exit = cfg.fee_rate * notional * exit_mask.astype(mx.float32) * (
            1.0 - liq_mask.astype(mx.float32)
        )
        # Realized for ledger: liq loses collateral; else pnl - fees
        exit_realized = mx.where(
            liq_mask,
            -exit_coll,
            exit_pnl - fee_exit,
        )
        cash = cash + mx.sum(
            exit_return * exit_mask.astype(mx.float32) - fee_exit, axis=1
        )
        liq_any = mx.any(liq_mask & exit_mask, axis=1)

        side = mx.where(exit_mask, mx.zeros_like(side), side)
        coll = mx.where(exit_mask, mx.zeros_like(coll), coll)
        lev = mx.where(exit_mask, mx.zeros_like(lev), lev)
        entry = mx.where(exit_mask, mx.zeros_like(entry), entry)
        tp = mx.where(exit_mask, mx.zeros_like(tp), tp)
        sl = mx.where(exit_mask, mx.zeros_like(sl), sl)

        # Snapshot bar index for ledger (pre-increment)
        ev_t = t

        # Funding on remaining open notionals at bar close
        upnl_c = self._upnl_static(side, entry, coll, lev, c)
        fund_pnl = -side * coll * lev * funding
        cash = cash + mx.sum(fund_pnl, axis=1)

        equity1 = cash + mx.sum(coll + upnl_c, axis=1)
        peak1 = mx.maximum(peak0, equity1)
        dd1 = (peak1 - equity1) / mx.maximum(peak1, 1e-8)

        pnl = (equity1 - equity0) * inv
        dd_inc = mx.maximum(dd1 - dd0, 0.0)
        recovery = mx.maximum(dd0 - dd1, 0.0) * (dd0 > 1e-6).astype(mx.float32)

        reward = (
            cfg.reward_pnl_scale * pnl
            - cfg.reward_dd_penalty * dd_inc
            + cfg.reward_recovery_bonus * recovery
            - cfg.reward_vol_penalty * mx.abs(pnl)
            - cfg.liquidation_penalty * liq_any.astype(mx.float32)
            - cfg.reward_hold_cost * mx.mean(mx.abs(side), axis=1)
        ).astype(mx.float32)

        t = t + 1
        steps = steps + 1
        terminated = equity1 <= cfg.initial_balance * 0.05
        truncated = (t >= self._n_decision - 1) | (
            steps >= self._max_episode_steps
        )
        truncated = truncated & ~terminated
        done = terminated | truncated

        ep_ret = ep_ret + reward
        ep_len = ep_len + 1
        res_ret = mx.where(done, ep_ret, mx.zeros_like(ep_ret))
        res_len = mx.where(done, ep_len, mx.zeros_like(ep_len))

        # Terminal obs before reset
        term_obs = self._obs_compiled(
            t, cash, peak1, side, entry, coll, lev, tp, sl, goal
        )

        # Auto-reset finished envs (new tape would be slow; soft-reset portfolio
        # and rewind time on the same book for speed — manager/worker still see
        # a fresh episode window).
        reset_t = mx.full_like(t, self._start_t)
        reset_cash = mx.full_like(cash, cfg.initial_balance)
        z = mx.zeros_like(side)
        t = mx.where(done, reset_t, t)
        steps = mx.where(done, mx.zeros_like(steps), steps)
        ep_ret = mx.where(done, mx.zeros_like(ep_ret), ep_ret)
        ep_len = mx.where(done, mx.zeros_like(ep_len), ep_len)
        cash = mx.where(done, reset_cash, cash)
        peak = mx.where(done, reset_cash, peak1)
        side = mx.where(done[:, None], z, side)
        entry = mx.where(done[:, None], z, entry)
        coll = mx.where(done[:, None], z, coll)
        lev = mx.where(done[:, None], z, lev)
        tp = mx.where(done[:, None], z, tp)
        sl = mx.where(done[:, None], z, sl)

        obs = self._obs_compiled(
            t, cash, peak, side, entry, coll, lev, tp, sl, goal
        )
        # Event pack for optional ledger (test only). Masks are bool (E, A).
        return (
            t, steps, ep_ret, ep_len, cash, peak,
            side, entry, coll, lev, tp, sl,
            obs, reward, done, terminated, truncated, term_obs, res_ret, res_len,
            ev_t,
            close_mask, close_side, close_entry, close_coll, close_lev,
            fill, fee_close, close_pnl - fee_close,
            open_mask, open_side, open_px, open_coll, open_lev, fee_open,
            tp_mask, sl_mask, liq_mask,
            exit_side, exit_entry, exit_coll, exit_lev, exit_px,
            fee_exit, exit_realized,
        )

    @staticmethod
    def _upnl_static(side, entry, coll, lev, mark):
        ent = mx.maximum(entry, 1e-8)
        ret = (mark / ent) - 1.0
        return side * coll * lev * ret

    def _obs_compiled(self, t, cash, peak, side, entry, coll, lev, tp, sl, goal):
        cfg = self.cfg
        inv = 1.0 / cfg.initial_balance
        mark = self._marks(t)
        upnl = self._upnl_static(side, entry, coll, lev, mark)
        equity = cash + mx.sum(coll + upnl, axis=1)
        peak = mx.maximum(peak, equity)
        dd = (peak - equity) / mx.maximum(peak, 1e-8)
        lev_n = lev / cfg.max_leverage
        port = mx.concatenate(
            [
                mx.stack(
                    [equity * inv, cash * inv, peak * inv, dd], axis=1
                ),
                side,
                lev_n,
                coll * inv,
                upnl * inv,
            ],
            axis=1,
        )
        low, high = self._market_obs(t)
        return mx.concatenate([low, high, port, goal], axis=1).astype(mx.float32)

    def step(self, actions: mx.array) -> StepResult:
        actions = mx.asarray(actions, dtype=mx.float32)
        if actions.ndim == 1:
            actions = actions[None, :]
        out = self._step_fn(
            self._t, self._steps, self._ep_ret, self._ep_len,
            self._cash, self._peak,
            self._side, self._entry, self._coll, self._lev, self._tp, self._sl,
            actions, self._goal,
        )
        (
            self._t, self._steps, self._ep_ret, self._ep_len,
            self._cash, self._peak,
            self._side, self._entry, self._coll, self._lev, self._tp, self._sl,
            obs, reward, done, terminated, truncated, term_obs, ep_ret, ep_len,
            ev_t,
            close_mask, close_side, close_entry, close_coll, close_lev,
            close_px, fee_close, close_realized,
            open_mask, open_side, open_px, open_coll, open_lev, fee_open,
            tp_mask, sl_mask, liq_mask,
            exit_side, exit_entry, exit_coll, exit_lev, exit_px,
            fee_exit, exit_realized,
        ) = out
        mx.eval(
            self._t, self._cash, self._side, obs, reward, done,
            terminated, truncated, term_obs, ep_ret, ep_len,
        )
        if self.ledger is not None:
            self._record_events(
                ev_t,
                close_mask, close_side, close_entry, close_coll, close_lev,
                close_px, fee_close, close_realized,
                open_mask, open_side, open_px, open_coll, open_lev, fee_open,
                tp_mask, sl_mask, liq_mask,
                exit_side, exit_entry, exit_coll, exit_lev, exit_px,
                fee_exit, exit_realized,
            )
        return StepResult(
            obs=obs,
            reward=reward,
            done=done,
            terminated=terminated,
            truncated=truncated,
            terminal_obs=term_obs,
            ep_ret=ep_ret,
            ep_len=ep_len,
        )

    def _record_events(
        self,
        ev_t,
        close_mask, close_side, close_entry, close_coll, close_lev,
        close_px, fee_close, close_realized,
        open_mask, open_side, open_px, open_coll, open_lev, fee_open,
        tp_mask, sl_mask, liq_mask,
        exit_side, exit_entry, exit_coll, exit_lev, exit_px,
        fee_exit, exit_realized,
    ) -> None:
        ledger = self.ledger
        assert ledger is not None
        mx.eval(
            ev_t, close_mask, open_mask, tp_mask, sl_mask, liq_mask,
            close_side, close_entry, close_coll, close_lev, close_px,
            fee_close, close_realized,
            open_side, open_px, open_coll, open_lev, fee_open,
            exit_side, exit_entry, exit_coll, exit_lev, exit_px,
            fee_exit, exit_realized,
        )
        e, a = close_mask.shape
        t_list = ev_t.tolist()
        batches = [
            ("market_close", close_mask, close_side, close_entry, close_coll,
             close_lev, close_entry, close_px, fee_close, close_realized, True),
            ("market_open", open_mask, open_side, open_px, open_coll,
             open_lev, open_px, None, fee_open, None, False),
            ("take_profit", tp_mask, exit_side, exit_entry, exit_coll,
             exit_lev, exit_entry, exit_px, fee_exit, exit_realized, True),
            ("stop_loss", sl_mask, exit_side, exit_entry, exit_coll,
             exit_lev, exit_entry, exit_px, fee_exit, exit_realized, True),
            ("liquidation", liq_mask, exit_side, exit_entry, exit_coll,
             exit_lev, exit_entry, exit_px, fee_exit, exit_realized, True),
        ]
        for (
            etype, mask, side, entry, coll, lev, open_p, exit_p, fee, pnl, is_exit
        ) in batches:
            m = mask.tolist()
            side_l = side.tolist()
            entry_l = entry.tolist()
            coll_l = coll.tolist()
            lev_l = lev.tolist()
            open_l = open_p.tolist()
            exit_l = None if exit_p is None else exit_p.tolist()
            fee_l = fee.tolist()
            pnl_l = None if pnl is None else pnl.tolist()
            for i in range(e):
                bar_t = int(t_list[i])
                for j in range(a):
                    if not m[i][j]:
                        continue
                    coll_v = float(coll_l[i][j])
                    if coll_v < 1e-12 and etype != "liquidation":
                        continue
                    ledger.record(
                        bar_t=bar_t,
                        env_id=i,
                        asset_i=j,
                        side=float(side_l[i][j]),
                        leverage=float(lev_l[i][j]),
                        event_type=etype,
                        open_price=float(open_l[i][j]),
                        exit_price=(
                            None if not is_exit else float(exit_l[i][j])
                        ),
                        collateral=coll_v,
                        fees_usdc=float(fee_l[i][j]),
                        pnl_usdc=(
                            None if pnl_l is None else float(pnl_l[i][j])
                        ),
                    )
