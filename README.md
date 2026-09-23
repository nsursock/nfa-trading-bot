# Trading Bot — MLX RL algorithms

Reinforcement-learning algorithms (PPO, SAC, TD3) implemented from scratch in
[Apple MLX](https://github.com/ml-explore/mlx), with vectorized MLX
environments (CartPole-v1, Pendulum-v1) for testing. No PyTorch, NumPy,
Gymnasium or Stable-Baselines3 — everything runs on MLX.

The API mirrors Stable-Baselines3: hyperparameters use SB3 names and defaults
(validated with pydantic), and training statistics are written as CSV with
SB3's TensorBoard keys (`rollout/ep_rew_mean`, `train/approx_kl`, ...).

## Layout

```
configs/            YAML hyperparameters, one file per algo/env pair
  ppo_cartpole.yaml
  sac_pendulum.yaml
  td3_pendulum.yaml
scripts/
  algos/
    common.py       BaseAlgoConfig (+ from_yaml), StatsLogger, ReplayBuffer, MLP, polyak_update, make_env
    ppo.py          PPO  (discrete actions, GAE, clipped surrogate)
    sac.py          SAC  (squashed Gaussian, twin critics, auto entropy coef)
    td3.py          TD3  (delayed policy updates, target policy smoothing)
  envs/
    cartpole.py     Vectorized CartPole-v1 (exact gym dynamics)
    pendulum.py     Vectorized Pendulum-v1 (exact gym dynamics)
outputs/            stats_{algo}_{env}.csv written by training runs
tests/              fast unit / smoke tests (~2 s)
```

## Setup

Requires macOS with Apple silicon and Python 3.11.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training

Every parameter lives in a YAML file under `configs/`; the file is loaded into
a pydantic config (`PPOConfig`, `SACConfig`, `TD3Config`) with
`extra="forbid"`, so typos fail fast.

```bash
python -m scripts.algos.ppo                                  # configs/ppo_cartpole.yaml
python -m scripts.algos.sac                                  # configs/sac_pendulum.yaml
python -m scripts.algos.td3                                  # configs/td3_pendulum.yaml
python -m scripts.algos.ppo --config path/to/other.yaml
```

Each run overwrites `outputs/stats_{algo}_{env_id}.csv`, e.g.
`outputs/stats_ppo_CartPole-v1.csv`, with one row per logging event
(per rollout iteration for PPO, every `log_interval` episodes for SAC/TD3).

| Algo | Logged columns |
|------|----------------|
| PPO  | `time/iterations time/total_timesteps time/fps time/time_elapsed rollout/ep_rew_mean rollout/ep_len_mean train/approx_kl train/clip_fraction train/clip_range train/entropy_loss train/explained_variance train/learning_rate train/loss train/n_updates train/policy_gradient_loss train/value_loss` |
| SAC  | `time/episodes time/fps time/time_elapsed time/total_timesteps rollout/ep_rew_mean rollout/ep_len_mean train/actor_loss train/critic_loss train/ent_coef train/ent_coef_loss train/learning_rate train/n_updates` |
| TD3  | `time/episodes time/fps time/time_elapsed time/total_timesteps rollout/ep_rew_mean rollout/ep_len_mean train/actor_loss train/critic_loss train/learning_rate train/n_updates` |

### Programmatic use

```python
from scripts.algos.common import make_env
from scripts.algos.ppo import PPO, PPOConfig

cfg = PPOConfig.from_yaml("configs/ppo_cartpole.yaml")
env = make_env(cfg.env_id, n_envs=cfg.n_envs, seed=cfg.seed or 0)
agent = PPO(cfg, env).learn()
actions = agent.predict(env.reset(), deterministic=True)
```

## Environments

Both environments reproduce the Gymnasium dynamics exactly and follow SB3
`VecEnv` semantics: `reset()` returns a `(n_envs, obs_dim)` array,
`step(actions)` returns `(obs, reward, done, infos)`, finished envs auto-reset,
and `infos[i]` carries `episode={"r", "l"}`, `terminal_observation` and
`TimeLimit.truncated` when an episode ends.

## Tests

```bash
pytest -q
```

The suite covers environment dynamics, config validation / YAML loading, and a
short smoke run per algorithm (CSV schema, finite values, `predict` shapes).
Long training runs (scaling with `n_envs`, time to solve) are intentionally
not part of the test suite.
