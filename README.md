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
utils/bench/
  scale.py          n_envs sweep: env FPS and train FPS
  throughput.py     fixed replay ratio: wall time for a fixed transition count
  solve.py          n_envs sweep: time to solve (gymnasium reward thresholds)
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
`step(actions)` returns a `StepResult` (`scripts/envs/base.py`), and finished
envs auto-reset with the pre-reset observation in `terminal_obs`.

| field          | shape          | meaning |
|----------------|----------------|---------|
| `obs`          | `(n, obs_dim)` | observation after auto-reset |
| `reward`       | `(n,)`         | step reward |
| `done`         | `(n,)`         | `terminated \| truncated` |
| `terminated`   | `(n,)`         | real termination |
| `truncated`    | `(n,)`         | time limit hit and not terminated (`TimeLimit.truncated`) |
| `terminal_obs` | `(n, obs_dim)` | obs before auto-reset (== `obs` where not done) |
| `ep_ret`       | `(n,)`         | return of the episode that ended this step, 0 elsewhere |
| `ep_len`       | `(n,)`         | length of that episode, 0 elsewhere |

## Tests

```bash
pytest -q
```

The suite covers environment dynamics, config validation / YAML loading, and a
short smoke run per algorithm (CSV schema, finite values, `predict` shapes).
Long training runs (scaling with `n_envs`, time to solve) are intentionally
not part of the test suite; see `utils/bench/`.

## Benchmarks

```bash
python -m utils.bench.scale                       # n_envs 32, 64, 128, 3 repeats each
python -m utils.bench.scale --start 16 --end 256  # geometric sweep (x2 per step)
python -m utils.bench.scale --n-envs 32 48 64 --algos ppo --seconds 10
```

`scale.py` runs each configuration in a fresh process, discards
`--warmup-seconds` (1s default), then measures at least `--seconds` (3s) of
steady-state training, stopping at complete cycle boundaries. Per
`(algo, env, n_envs)` it reports `env_fps` (bare `env.step`, one eval per
step), `train_fps` (median over `--repeats`, with `train_min`/`train_max`),
`updates/s` and `samples/s` over the same window, `samples/step` (replay
ratio), and the raw per-repeat details in `bench_scale.trials.jsonl`.
Hyperparameters scale with `n_envs` (PPO: batch_size; SAC/TD3: replayed
samples by `--replay-scaling` sqrt, batch capped by `--max-batch-size`);
`--fixed-hparams` keeps the YAML values.

```bash
python -m utils.bench.throughput                 # SAC and TD3 at 1024/2048/4096
python -m utils.bench.throughput --algos td3 --repeats 3
python -m utils.bench.throughput --algos ppo --timesteps 1048576
```

`throughput.py` asks how many transitions per second the learner can consume
when the learning problem does not get easier as `n_envs` grows. Batch size,
replay buffer, and `learning_starts` stay at the YAML values. SAC/TD3
`gradient_steps` scales with `n_envs`, so each transition is replayed 256
times at every width (the YAML ratio). PPO keeps `batch_size`, `n_steps`, and
`n_epochs`. Every width is timed on the same transition count, after two
warmup cycles. The default count is eight vector steps of the widest env
(32768 at 4096); PPO rounds that up to a whole rollout (`n_steps * n_envs`).

```bash
python -m utils.bench.solve                    # time-to-solve, 32/64/128 envs
python -m utils.bench.solve --budget-mult 4 --seeds 0 1 2
```

`solve.py` trains until the 100-episode mean return reaches gymnasium's
`reward_threshold` (CartPole-v1: 475; Pendulum-v1 has none, -200 is used) and
reports wall time and timesteps to solve, or `solved=no` with the best return
within the budget. Results go to `outputs/bench_{scale,throughput,solve}.csv`.
