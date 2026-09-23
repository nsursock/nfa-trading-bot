import csv
import math

import pytest
from pydantic import ValidationError

from scripts.algos.ppo import PPO, PPOConfig
from scripts.envs.cartpole import CartPoleEnv

CSV_HEADER = [
    "time/iterations", "time/total_timesteps", "time/fps", "time/time_elapsed",
    "rollout/ep_rew_mean", "rollout/ep_len_mean", "train/approx_kl",
    "train/clip_fraction", "train/clip_range", "train/entropy_loss",
    "train/explained_variance", "train/learning_rate", "train/loss",
    "train/n_updates", "train/policy_gradient_loss", "train/value_loss",
]


def test_ppo_config_validation():
    cfg = PPOConfig()
    assert cfg.learning_rate == 3e-4
    assert cfg.n_steps == 2048
    assert cfg.batch_size == 64
    assert cfg.n_epochs == 10
    assert cfg.gamma == 0.99
    assert cfg.gae_lambda == 0.95
    assert cfg.clip_range == 0.2
    assert cfg.ent_coef == 0.0
    assert cfg.vf_coef == 0.5
    assert cfg.max_grad_norm == 0.5
    with pytest.raises(ValidationError):
        PPOConfig(n_steps=100, batch_size=64, n_envs=1)
    with pytest.raises(ValidationError):
        PPOConfig(nonexistent_field=1)


def test_ppo_from_yaml():
    cfg = PPOConfig.from_yaml("configs/ppo_cartpole.yaml")
    assert cfg.env_id == "CartPole-v1"
    assert cfg.n_envs == 8
    assert cfg.n_steps == 256
    assert cfg.batch_size == 256
    assert cfg.n_epochs == 10
    assert cfg.total_timesteps == 100_000
    assert cfg.seed == 0


def test_ppo_smoke(tmp_path):
    cfg = PPOConfig.from_yaml("configs/ppo_cartpole.yaml")
    cfg = cfg.model_copy(update={
        "log_dir": str(tmp_path), "verbose": 0, "n_envs": 2, "n_steps": 32,
        "batch_size": 32, "n_epochs": 1, "total_timesteps": 128,
    })
    env = CartPoleEnv(n_envs=cfg.n_envs, seed=cfg.seed or 0)
    agent = PPO(cfg, env).learn()

    csv_path = tmp_path / "stats_ppo_CartPole-v1.csv"
    assert csv_path.exists()
    with open(csv_path) as f:
        rows = list(csv.reader(f))
    assert rows[0] == CSV_HEADER
    assert len(rows) >= 2
    for row in rows[1:]:
        for cell in row:
            if cell != "":
                assert math.isfinite(float(cell))
    last = dict(zip(rows[0], rows[-1]))
    assert int(last["time/total_timesteps"]) == 128
    assert int(last["train/n_updates"]) > 0

    obs = env.reset()
    actions = agent.predict(obs)
    assert actions.shape == (cfg.n_envs,)
    assert set(actions.tolist()) <= {0, 1}
