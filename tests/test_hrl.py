"""Smoke tests for the HRL trading env / agent / main."""

import csv
import math
from pathlib import Path

import mlx.core as mx
import pytest
from pydantic import ValidationError

from scripts.agent import HRLConfig
from scripts.env import TF_MINUTES, TradingEnv, TradingEnvConfig
from scripts.ledger import LEDGER_HEADER
from scripts.main import run


def test_env_config_binance_tfs():
    cfg = TradingEnvConfig(low_tf="5m", high_tf="1h", n_low_bars=128)
    assert cfg.tf_ratio == 12
    assert "1m" in TF_MINUTES
    with pytest.raises(ValidationError):
        TradingEnvConfig(low_tf="1h", high_tf="5m")
    with pytest.raises(ValidationError):
        TradingEnvConfig(min_leverage=10, max_leverage=2)


def test_env_reset_step_shapes():
    cfg = TradingEnvConfig(
        n_envs=2, n_assets=2, n_low_bars=96, lookback_low=4, lookback_high=2,
        low_tf="5m", high_tf="1h", seed=1, market_n_substeps=4,
        manager_horizon=4, goal_dim=4,
    )
    env = TradingEnv(cfg)
    obs = env.reset()
    assert obs.shape == (2, env.obs_dim)
    assert env.manager_obs().shape == (2, env.manager_obs_dim)
    env.set_goal(mx.zeros((2, 4)))
    act = mx.zeros((2, env.action_dim))
    res = env.step(act)
    assert res.obs.shape == obs.shape
    assert res.reward.shape == (2,)
    assert res.done.shape == (2,)
    assert bool(mx.all(mx.isfinite(res.reward)))
    assert env.achieved_goal().shape == (2, 4)


def test_liquidation_path_does_not_nan():
    cfg = TradingEnvConfig(
        n_envs=1, n_assets=1, n_low_bars=64, lookback_low=2, lookback_high=2,
        low_tf="1m", high_tf="5m", max_leverage=50.0, min_leverage=20.0,
        maintenance_margin=0.05, min_collateral=0.5, max_collateral=0.9,
        risk_per_trade=0.5, market_sigma=0.08, market_regime="cascade",
        market_n_substeps=4, seed=7,
    )
    env = TradingEnv(cfg)
    env.reset()
    env.set_goal(mx.zeros((1, cfg.goal_dim)))
    for _ in range(30):
        act = mx.array([[1.0, 1.0, 1.0, -1.0, -1.0]])
        res = env.step(act)
        assert bool(mx.all(mx.isfinite(res.reward)))
        assert bool(mx.all(mx.isfinite(res.obs)))


def test_ledger_only_when_attached(tmp_path):
    cfg = TradingEnvConfig(
        n_envs=1, n_assets=1, n_low_bars=64, lookback_low=2, lookback_high=2,
        low_tf="1m", high_tf="5m", seed=3, market_n_substeps=2,
        symbols=["BTCUSDT"],
    )
    env = TradingEnv(cfg)
    env.reset()
    env.set_goal(mx.zeros((1, cfg.goal_dim)))
    for _ in range(5):
        env.step(mx.array([[1.0, 0.0, 0.0, 0.0, 0.0]]))
    assert env.ledger is None
    path = tmp_path / "ledger.csv"
    env.attach_ledger(str(path))
    env.detach_ledger()
    assert path.exists()
    with open(path) as f:
        rows = list(csv.reader(f))
    assert rows[0] == LEDGER_HEADER
    assert len(rows) == 1  # header only — no steps while attached


def test_ledger_trade_events_controlled_under_10s(tmp_path):
    """Ledger correctness via scripted actions (not random policy samples)."""
    import time

    t0 = time.perf_counter()
    flat = mx.array([[0.0, 0.0, 0.0, 0.0, 0.0]])
    all_rows: list[dict] = []

    def _scripted(name: str, cfg: TradingEnvConfig, acts: list[mx.array]) -> None:
        env = TradingEnv(cfg)
        env.reset()
        env.set_goal(mx.zeros((cfg.n_envs, cfg.goal_dim)))
        path = tmp_path / name
        env.attach_ledger(str(path))
        for act in acts:
            env.step(act)
        env.detach_ledger()
        with open(path) as f:
            all_rows.extend(csv.DictReader(f))

    base = dict(
        n_envs=1, n_assets=1, n_low_bars=96, lookback_low=2, lookback_high=2,
        low_tf="1m", high_tf="5m", market_n_substeps=2, symbols=["BTCUSDT"],
        side_deadzone=0.1, fee_rate=0.0004,
    )

    # market_open + market_close
    _scripted(
        "close.csv",
        TradingEnvConfig(
            **base, seed=11, market_sigma=0.001, market_regime=None,
            min_leverage=1.0, max_leverage=5.0, min_collateral=0.1,
            max_collateral=0.3, min_tp=0.05, max_tp=0.15, min_sl=0.05,
            max_sl=0.15, risk_per_trade=0.2, maintenance_margin=0.001,
        ),
        [mx.array([[1.0, 0.0, 0.0, 1.0, 1.0]])] * 2 + [flat] * 2,
    )
    # take_profit
    _scripted(
        "tp.csv",
        TradingEnvConfig(
            **base, seed=5, market_sigma=0.05, market_regime="bull",
            min_leverage=1.0, max_leverage=5.0, min_collateral=0.1,
            max_collateral=0.3, min_tp=0.005, max_tp=0.08, min_sl=0.05,
            max_sl=0.15, risk_per_trade=0.2, maintenance_margin=0.001,
        ),
        [mx.array([[1.0, 0.0, 0.0, -1.0, 1.0]])] * 40,
    )
    # stop_loss
    _scripted(
        "sl.csv",
        TradingEnvConfig(
            **base, seed=3, market_sigma=0.03, market_regime="bear",
            min_leverage=1.0, max_leverage=5.0, min_collateral=0.1,
            max_collateral=0.3, min_tp=0.05, max_tp=0.15, min_sl=0.005,
            max_sl=0.08, risk_per_trade=0.2, maintenance_margin=0.0001,
        ),
        [mx.array([[1.0, -1.0, 0.0, 1.0, -1.0]])] * 40,
    )
    # liquidation
    _scripted(
        "liq.csv",
        TradingEnvConfig(
            **base, seed=7, market_sigma=0.08, market_regime="cascade",
            min_leverage=10.0, max_leverage=25.0, min_collateral=0.4,
            max_collateral=0.9, min_tp=0.05, max_tp=0.15, min_sl=0.05,
            max_sl=0.15, risk_per_trade=0.6, maintenance_margin=0.05,
        ),
        [mx.array([[1.0, 1.0, 1.0, 1.0, 1.0]])] * 40,
    )

    elapsed = time.perf_counter() - t0
    assert elapsed < 10.0, f"ledger test took {elapsed:.2f}s (>10s)"

    assert all_rows, "expected trade events from scripted actions"
    assert list(all_rows[0].keys()) == LEDGER_HEADER
    types = {r["type"] for r in all_rows}
    assert types >= {
        "market_open", "market_close", "take_profit", "stop_loss", "liquidation",
    }

    opens = [r for r in all_rows if r["type"] == "market_open"]
    closes = [r for r in all_rows if r["type"] == "market_close"]
    assert opens and closes
    for r in opens:
        assert r["pair"] == "BTCUSDT"
        assert r["symbol"] == "BTC"
        assert r["side"] in ("long", "short")
        assert r["exit_price"] == ""
        assert r["pnl_usdc"] == ""
        assert float(r["collateral"]) > 0
        assert float(r["size_usdc"]) > 0
        assert float(r["fees_usdc"]) >= 0
    for r in closes:
        assert r["exit_price"] != ""
        assert r["pnl_usdc"] != ""
        assert r["pnl_pct"] != ""
        assert float(r["open_price"]) > 0
        assert float(r["exit_price"]) > 0


@pytest.mark.parametrize("pair", ["ppo_sac", "sac_sac", "ppo_td3"])
def test_hrl_smoke_pairs(tmp_path, pair):
    """Fast HRL train+test wiring check — ledger content is covered elsewhere."""
    cfg = HRLConfig.from_yaml("configs/smoke.yaml")
    cfg = cfg.model_copy(update={
        "pair": pair,
        "mode": "full",
        "verbose": 0,
        "log_dir": str(tmp_path),
        "ckpt_dir": str(tmp_path / "ckpts"),
        "total_timesteps": 128,
        "test_episodes": 1,
        "test_random_actions": False,  # do not rely on random for assertions
        "worker_learning_starts": 32,
        "manager_learning_starts": 8,
    })
    env_cfg = cfg.env.model_copy(update={
        "n_envs": 2, "n_assets": 1, "n_low_bars": 64,
        "lookback_low": 2, "lookback_high": 2, "market_n_substeps": 2,
        "symbols": ["BTCUSDT"],
    })
    cfg = cfg.model_copy(update={"env": env_cfg})
    summary = run(cfg)
    assert summary is not None
    assert math.isfinite(summary["mean_return"])
    runs = [p for p in Path(tmp_path).iterdir() if p.is_dir()]
    assert len(runs) == 1
    run_dir = runs[0]
    m_algo = "ppo" if pair.startswith("ppo") else "sac"
    w_algo = "td3" if pair.endswith("td3") else "sac"
    assert (run_dir / f"stats_manager_{m_algo}.csv").exists()
    assert (run_dir / f"stats_worker_{w_algo}.csv").exists()
    assert (run_dir / "ledger.csv").exists()  # attached in test mode
    assert (run_dir / "breakdown.txt").exists()
    assert (run_dir / "performance.png").exists()
    assert (run_dir / "distributions.png").exists()
    assert not list(run_dir.glob("performance_ep*.png"))
    assert (run_dir / "config.yaml").exists()


def test_train_mode_has_no_ledger(tmp_path):
    cfg = HRLConfig.from_yaml("configs/smoke.yaml")
    env_cfg = cfg.env.model_copy(update={
        "n_envs": 2, "n_assets": 1, "n_low_bars": 64,
        "lookback_low": 2, "lookback_high": 2, "market_n_substeps": 2,
        "symbols": ["BTCUSDT"],
    })
    cfg = cfg.model_copy(update={
        "env": env_cfg,
        "mode": "train",
        "verbose": 0,
        "log_dir": str(tmp_path),
        "ckpt_dir": str(tmp_path),
        "total_timesteps": 64,
        "worker_learning_starts": 16,
        "manager_learning_starts": 8,
    })
    run(cfg)
    runs = [p for p in Path(tmp_path).iterdir() if p.is_dir()]
    assert len(runs) == 1
    assert (runs[0] / "stats_manager_ppo.csv").exists()
    assert (runs[0] / "stats_worker_sac.csv").exists()
    assert not (runs[0] / "ledger.csv").exists()


def test_main_config_loads():
    cfg = HRLConfig.from_yaml("configs/smoke.yaml")
    assert cfg.pair == "ppo_sac"
    assert cfg.train_schedule == "joint"
    assert cfg.env.low_tf == "5m"
    assert cfg.env.high_tf == "1h"
    assert cfg.total_timesteps == 512


def test_alternating_schedule_phases():
    cfg = HRLConfig.from_yaml("configs/smoke.yaml")
    env_cfg = cfg.env.model_copy(update={
        "n_envs": 2, "n_assets": 1, "n_low_bars": 64,
        "lookback_low": 2, "lookback_high": 2, "market_n_substeps": 2,
    })
    cfg = cfg.model_copy(update={
        "env": env_cfg,
        "train_schedule": "alternating",
        "alt_worker_timesteps": 40,
        "alt_manager_timesteps": 40,
        "total_timesteps": 160,
        "verbose": 0,
        "mode": "train",
        "worker_learning_starts": 16,
        "manager_learning_starts": 8,
        "log_dir": "outputs",
    })
    from scripts.agent import HRLAgent
    from scripts.env import TradingEnv

    agent = HRLAgent(cfg, TradingEnv(cfg.env))
    assert agent._phase_allows(0) == (True, False, "worker")
    assert agent._phase_allows(39) == (True, False, "worker")
    assert agent._phase_allows(40) == (False, True, "manager")
    assert agent._phase_allows(79) == (False, True, "manager")
    assert agent._phase_allows(80) == (True, False, "worker")
    agent.config = cfg.model_copy(update={"train_schedule": "joint"})
    assert agent._phase_allows(0) == (True, True, "joint")
