import csv
import math

import mlx.core as mx
import pytest
from pydantic import ValidationError

from scripts.algos.common import make_env
from scripts.algos.sac import SAC, SACConfig

HEADER = [
    "time/episodes", "time/fps", "time/time_elapsed", "time/total_timesteps",
    "rollout/ep_rew_mean", "rollout/ep_len_mean", "train/actor_loss",
    "train/critic_loss", "train/ent_coef", "train/ent_coef_loss",
    "train/learning_rate", "train/n_updates",
]


def test_sac_config_defaults():
    cfg = SACConfig(env_id="Pendulum-v1", total_timesteps=100)
    assert cfg.learning_rate == 3e-4
    assert cfg.batch_size == 256
    assert cfg.tau == 0.005
    assert cfg.gamma == 0.99
    assert cfg.ent_coef == "auto"
    assert cfg.target_entropy == "auto"
    assert cfg.net_arch == [256, 256]
    with pytest.raises(ValidationError):
        SACConfig(env_id="Pendulum-v1", total_timesteps=1, bogus=1)


def test_sac_from_yaml():
    cfg = SACConfig.from_yaml("configs/sac_pendulum.yaml")
    assert cfg.env_id == "Pendulum-v1"
    assert cfg.total_timesteps == 30000
    assert cfg.seed == 0


def test_sac_smoke(tmp_path):
    cfg = SACConfig.from_yaml("configs/sac_pendulum.yaml")
    cfg = cfg.model_copy(update={
        "log_dir": str(tmp_path), "verbose": 0, "total_timesteps": 200,
        "learning_starts": 50, "batch_size": 32, "buffer_size": 1000,
        "log_interval": 1, "net_arch": [32, 32],
    })
    env = make_env(cfg.env_id, n_envs=cfg.n_envs, seed=cfg.seed)
    agent = SAC(cfg, env).learn()

    csv_path = tmp_path / f"stats_sac_{cfg.env_id}.csv"
    assert csv_path.exists()
    with open(csv_path) as f:
        rows = list(csv.reader(f))
    assert rows[0] == HEADER
    assert len(rows) >= 2
    for row in rows[1:]:
        for cell in row:
            if cell != "":
                assert math.isfinite(float(cell))
    last = dict(zip(rows[0], rows[-1]))
    assert int(last["time/total_timesteps"]) == cfg.total_timesteps
    assert int(last["train/n_updates"]) > 0
    assert float(last["train/ent_coef"]) > 0

    obs = env.reset()
    actions = agent.predict(obs)
    assert actions.shape == (cfg.n_envs, env.action_dim)
    assert bool(mx.all(actions >= env.action_low))
    assert bool(mx.all(actions <= env.action_high))
