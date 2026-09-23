import math

import mlx.core as mx
import pytest
from pydantic import ValidationError

from scripts.data import (
    CHANNELS,
    OBS_CHANNELS,
    REGIMES,
    REGIME_KITS,
    MarketSpec,
    _ar1,
    generate,
)


def _ref_ar(phi, scale, z, x0):
    x = x0
    cols = [x0]
    for t in range(1, z.shape[1]):
        x = phi * x + scale * z[:, t]
        cols.append(x)
    return mx.stack(cols, axis=1)


def test_shapes_and_channels_are_always_ohlcv_funding():
    for phi, regime in ((None, None), (0.4, "mixed")):
        spec = MarketSpec(n_envs=4, n_bars=16, seed=1, alpha_phi=phi, regime=regime)
        m = generate(spec)
        assert m.channels() == CHANNELS
        assert not hasattr(m, "alpha")
        for name in CHANNELS:
            col = getattr(m, name)
            assert col.shape == (4, 16)
            assert col.dtype == mx.float32
        assert m.features().shape == (4, 16, 6)
        assert m.observations().shape == (4, 16, len(OBS_CHANNELS))
        d = m.decisions()
        assert d.obs.shape == (4, 15, len(OBS_CHANNELS))
        assert d.next_obs.shape == d.obs.shape
        assert d.ret.shape == (4, 15)
        assert d.funding.shape == (4, 15)


def test_alpha_moves_ohlcv_and_funding_not_a_column():
    base = MarketSpec(
        n_envs=3, n_bars=12, seed=2, sigma=0.0, volume_sigma=0.0, funding_sigma=0.0,
        funding_x0=0.0, alpha_sigma=0.0, alpha_x0=0.05, regime=None,
    )
    off = generate(base)
    on = generate(base.model_copy(update={"alpha_phi": 0.5}))
    assert on.features().shape[-1] == 6
    assert on.observations().shape[-1] == len(OBS_CHANNELS)
    assert not bool(mx.all(off.close == on.close))
    assert not bool(mx.all(off.volume == on.volume))
    assert not bool(mx.all(off.funding == on.funding))
    for t in range(12):
        a = 0.05 * (0.5 ** t)
        log_ret = float(mx.log(on.close[0, t] / on.open[0, t]))
        assert abs(log_ret - a) < 1e-4
        assert abs(float(mx.log(on.volume[0, t])) - a) < 1e-5
        assert abs(float(on.funding[0, t]) - a) < 1e-5


def test_ohlc_constraints_and_bar_continuity():
    m = generate(MarketSpec(n_envs=8, n_bars=32, seed=3, mu=0.01, sigma=0.03, regime=None))
    assert bool(mx.all(m.high + 1e-6 >= mx.maximum(m.open, m.close)))
    assert bool(mx.all(m.low <= mx.minimum(m.open, m.close) + 1e-6))
    assert bool(mx.all(m.high + 1e-6 >= m.low))
    assert bool(mx.all(m.open[:, 1:] == m.close[:, :-1]))
    assert bool(mx.all(m.open[:, 0] == 100.0))
    assert bool(mx.all(m.close > 0))
    assert bool(mx.all(m.volume > 0))
    assert bool(mx.all(mx.isfinite(m.funding)))
    obs = m.observations()
    assert bool(mx.all(obs[:, :, 1] >= -1e-5))
    assert bool(mx.all(obs[:, :, 2] <= 1e-5))


def test_deterministic_gbm_when_sigma_is_zero():
    spec = MarketSpec(
        n_envs=2, n_bars=6, seed=0, s0=50.0, mu=0.2, sigma=0.0, dt=1.0,
        n_substeps=4, regime=None,
    )
    m = generate(spec)
    t = mx.arange(6, dtype=mx.float32)
    want_open = 50.0 * mx.exp(0.2 * t)
    want_close = 50.0 * mx.exp(0.2 * (t + 1))
    assert bool(mx.all(mx.abs(m.open - want_open) < 1e-3))
    assert bool(mx.all(mx.abs(m.close - want_close) < 1e-3))
    assert bool(mx.all(mx.abs(m.low - m.open) < 1e-3))
    assert bool(mx.all(mx.abs(m.high - m.close) < 1e-3))
    other = generate(spec.model_copy(update={"n_substeps": 32}))
    assert bool(mx.all(mx.abs(m.close - other.close) < 1e-3))


def test_gbm_log_return_moments():
    mu, sigma, dt = 0.05, 0.2, 1.0
    m = generate(MarketSpec(
        n_envs=2048, n_bars=32, seed=4, mu=mu, sigma=sigma, dt=dt, n_substeps=8,
        regime=None,
    ))
    log_ret = mx.log(m.close / m.open)
    mean = float(mx.mean(log_ret))
    var = float(mx.var(log_ret))
    assert abs(mean - (mu - 0.5 * sigma**2) * dt) < 0.01
    assert abs(math.sqrt(var) - sigma * math.sqrt(dt)) < 0.01


def test_log_volume_moments():
    m = generate(MarketSpec(
        n_envs=2048, n_bars=16, seed=5, volume_mu=1.5, volume_sigma=0.8, regime=None,
    ))
    log_v = mx.log(m.volume)
    assert abs(float(mx.mean(log_v)) - 1.5) < 0.02
    assert abs(float(mx.var(log_v)) - 0.64) < 0.05


def test_ou_decay_without_noise_and_stationary_mean():
    theta, dt, x0, mu = 0.3, 2.0, 0.01, 0.001
    m = generate(MarketSpec(
        n_envs=2, n_bars=10, seed=0, funding_theta=theta, funding_mu=mu,
        funding_sigma=0.0, funding_x0=x0, dt=dt, regime=None,
    ))
    decay = math.exp(-theta * dt)
    for t in range(10):
        want = mu + (x0 - mu) * decay**t
        assert abs(float(m.funding[0, t]) - want) < 1e-6

    noisy = generate(MarketSpec(
        n_envs=4096, n_bars=32, seed=6, funding_theta=1.5, funding_mu=0.002,
        funding_sigma=0.001, regime=None,
    ))
    assert abs(float(mx.mean(noisy.funding)) - 0.002) < 2e-4


def test_ar1_matches_recurrence_across_chunks():
    phi, scale = -0.5, 0.7
    z = mx.random.normal(shape=(4, 80), dtype=mx.float32, key=mx.random.key(7))
    x0 = mx.array([0.4, -0.2, 1.0, 0.0])
    got = _ar1(phi, scale, z, x0)
    ref = _ref_ar(phi, scale, z, x0)
    assert float(mx.max(mx.abs(got - ref))) < 1e-5


def test_decisions_lag_the_scored_bar():
    m = generate(MarketSpec(n_envs=2, n_bars=5, seed=8, mu=0.0, sigma=0.02, regime=None))
    d = m.decisions()
    obs = m.observations()
    assert bool(mx.all(d.obs == obs[:, :-1, :]))
    assert bool(mx.all(d.next_obs == obs[:, 1:, :]))
    assert bool(mx.all(d.next_obs[:, :-1, :] == d.obs[:, 1:, :]))
    ret = m.close[:, 1:] / m.close[:, :-1] - 1.0
    assert bool(mx.all(mx.abs(d.ret - ret) < 1e-6))
    assert bool(mx.all(d.funding == m.funding[:, 1:]))
    log_ret = mx.log(m.close / m.open)
    assert bool(mx.all(mx.abs(d.obs[:, :, 0] - log_ret[:, :-1]) < 1e-6))


def test_same_seed_is_repeatable():
    spec = MarketSpec(n_envs=2, n_bars=8, seed=9, alpha_phi=0.2, regime="mixed")
    a, b = generate(spec), generate(spec)
    assert bool(mx.all(a.close == b.close))
    assert bool(mx.all(a.funding == b.funding))
    assert bool(mx.all(a.volume == b.volume))
    assert bool(mx.all(a.regime == b.regime))


def test_locked_bull_regime_drives_drift_volume_funding():
    kit = REGIME_KITS["bull"]
    m = generate(MarketSpec(
        n_envs=2, n_bars=20, seed=10, sigma=0.0, volume_sigma=0.0, funding_sigma=0.0,
        funding_x0=0.0, regime="bull",
    ))
    assert m.regime is not None
    assert bool(mx.all(m.regime == REGIMES.index("bull")))
    assert m.regime_name(0, 0) == "bull"
    log_ret = mx.log(m.close / m.open)
    assert abs(float(mx.mean(log_ret)) - kit.mu) < 1e-4
    assert abs(float(mx.mean(mx.log(m.volume))) - kit.volume_shift) < 1e-5
    assert abs(float(mx.mean(m.funding)) - kit.funding_bias) < 1e-5


def test_mixed_regimes_switch_and_stay_out_of_features():
    m = generate(MarketSpec(
        n_envs=4, n_bars=128, seed=11, regime="mixed", regime_persist=0.9,
    ))
    assert m.regime.shape == (4, 128)
    assert m.features().shape[-1] == 6
    assert set(int(x) for x in m.regime.reshape(-1).tolist()).issubset(set(range(len(REGIMES))))
    # With persist 0.9 over 128 bars, expect at least one switch somewhere.
    switches = int(mx.sum(m.regime[:, 1:] != m.regime[:, :-1]).item())
    assert switches >= 1
    # Sticky: most steps should stay.
    assert switches < 4 * 128 * 0.5


def test_cascade_is_weaker_than_bull_on_average():
    bull = generate(MarketSpec(n_envs=64, n_bars=64, seed=12, regime="bull", sigma=0.01))
    cascade = generate(MarketSpec(n_envs=64, n_bars=64, seed=12, regime="cascade", sigma=0.01))
    assert float(mx.mean(mx.log(cascade.close / cascade.open))) < float(
        mx.mean(mx.log(bull.close / bull.open))
    )
    assert float(mx.mean(cascade.funding)) < float(mx.mean(bull.funding))
    assert float(mx.mean(mx.log(cascade.volume))) > float(mx.mean(mx.log(bull.volume)))


def test_flat_regime_has_no_regime_array():
    m = generate(MarketSpec(n_envs=1, n_bars=8, seed=0, regime=None))
    assert m.regime is None
    assert m.regime_name(0, 0) is None


def test_spec_rejects_bad_inputs():
    with pytest.raises(ValidationError):
        MarketSpec(s0=-1.0)
    with pytest.raises(ValidationError):
        MarketSpec(n_bars=1)
    with pytest.raises(ValidationError):
        MarketSpec(alpha_phi=1.0)
    with pytest.raises(ValidationError):
        MarketSpec(alpha_phi=-1.0)
    with pytest.raises(ValidationError):
        MarketSpec(not_a_field=1)
    with pytest.raises(ValidationError):
        MarketSpec(regime="moon")
    with pytest.raises(ValidationError):
        MarketSpec(regime_persist=1.0)
