"""Shared pieces for the MLX RL algorithms (PPO, SAC, TD3)."""

import csv
import math
import os
import time
from collections import deque

import mlx.core as mx
import mlx.nn as nn
import yaml
from mlx.utils import tree_map
from pydantic import BaseModel, ConfigDict


class BaseAlgoConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int | None = None
    verbose: int = 1
    total_timesteps: int
    env_id: str
    log_dir: str = "outputs"
    n_envs: int = 1
    stats_window_size: int = 100

    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**(data or {}))


def make_env(env_id, n_envs=1, seed=0):
    from scripts.envs.cartpole import CartPoleEnv
    from scripts.envs.pendulum import PendulumEnv

    registry = {"CartPole-v1": CartPoleEnv, "Pendulum-v1": PendulumEnv}
    if env_id not in registry:
        raise ValueError(f"unknown env_id {env_id!r}; have {list(registry)}")
    return registry[env_id](n_envs=n_envs, seed=seed)


class StatsLogger:
    def __init__(self, path, header, verbose=1):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.header = list(header)
        self.verbose = verbose
        self._file = open(path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.header)
        self._file.flush()

    def log(self, row: dict):
        self._writer.writerow([row.get(k, "") for k in self.header])
        self._file.flush()
        if self.verbose >= 1:
            parts = []
            for k in self.header:
                v = row.get(k, "")
                if v == "":
                    continue
                name = k.split("/", 1)[-1]
                if isinstance(v, float):
                    parts.append(f"{name} {v:.4g}")
                else:
                    parts.append(f"{name} {v}")
            print(" | ".join(parts))

    def close(self):
        self._file.close()


class ReplayBuffer:
    """Ring buffer of mx.arrays. `dones` stored as float32 where done means
    terminated AND NOT truncated (SB3 handle_timeout_termination=True), so
    the TD target bootstraps through truncations."""

    def __init__(self, buffer_size, obs_dim, action_dim):
        self.buffer_size = buffer_size
        self.obs = mx.zeros((buffer_size, obs_dim), dtype=mx.float32)
        self.next_obs = mx.zeros((buffer_size, obs_dim), dtype=mx.float32)
        self.actions = mx.zeros((buffer_size, action_dim), dtype=mx.float32)
        self.rewards = mx.zeros((buffer_size, 1), dtype=mx.float32)
        self.dones = mx.zeros((buffer_size, 1), dtype=mx.float32)
        self.pos = 0
        self.full = False

    def add(self, obs, next_obs, action, reward, done):
        """Vectorized over n_envs rows. `done` is a float array (1 = real
        termination, 0 = truncation or not done)."""
        n = obs.shape[0]
        idx = mx.arange(self.pos, self.pos + n) % self.buffer_size
        self.obs[idx] = obs
        self.next_obs[idx] = next_obs
        self.actions[idx] = action
        self.rewards[idx] = reward.reshape(n, 1)
        self.dones[idx] = done.reshape(n, 1).astype(mx.float32)
        self.full = self.full or self.pos + n >= self.buffer_size
        self.pos = (self.pos + n) % self.buffer_size

    def __len__(self):
        return self.buffer_size if self.full else self.pos

    def sample(self, batch_size):
        idx = mx.random.randint(0, len(self), (batch_size,))
        return (
            self.obs[idx],
            self.next_obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.dones[idx],
        )


class MLP(nn.Module):
    """Plain MLP with default nn.Linear init (PyTorch-like, as SB3 SAC/TD3)."""

    def __init__(self, in_dim, hidden, out_dim, activation="relu",
                 output_activation=None):
        super().__init__()
        acts = {"relu": nn.relu, "tanh": nn.tanh}
        act = acts[activation]
        dims = [in_dim] + list(hidden) + [out_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            is_last = i == len(dims) - 2
            if not is_last:
                layers.append(act)
            elif output_activation == "tanh":
                layers.append(nn.tanh)
        self.net = nn.Sequential(*layers)

    def __call__(self, x):
        return self.net(x)


class BatchedLinear(nn.Module):
    """n independent Linear layers evaluated with one broadcast matmul."""

    def __init__(self, n, in_dim, out_dim):
        super().__init__()
        scale = math.sqrt(1.0 / in_dim)  # same init as nn.Linear
        self.weight = mx.random.uniform(-scale, scale, (n, in_dim, out_dim))
        self.bias = mx.random.uniform(-scale, scale, (n, 1, out_dim))

    def __call__(self, x):  # x: (n, B, in) or (B, in) broadcast
        return mx.matmul(x, self.weight) + self.bias


class TwinCritics(nn.Module):
    """Two Q-networks with stacked weights: one kernel per layer instead of two."""

    def __init__(self, obs_dim, action_dim, net_arch, activation="relu"):
        super().__init__()
        dims = [obs_dim + action_dim] + list(net_arch) + [1]
        self.layers = [
            BatchedLinear(2, dims[i], dims[i + 1])
            for i in range(len(dims) - 1)
        ]
        acts = {"relu": nn.relu, "tanh": nn.tanh}
        self.act = acts[activation]

    def __call__(self, obs, action):
        x = mx.concatenate([obs, action], axis=-1)[None]
        for i, lin in enumerate(self.layers):
            x = lin(x)
            if i < len(self.layers) - 1:
                x = self.act(x)
        q = x.squeeze(-1)  # (2, B)
        return q[0], q[1]


def orthogonal_init(module, gain):
    """SB3-style orthogonal init for an nn.Linear (zero bias elsewhere)."""
    rows, cols = module.weight.shape
    flat = mx.random.normal((rows, cols))
    transposed = rows < cols
    if transposed:
        flat = flat.T
    try:
        q, r = mx.linalg.qr(flat)
    except Exception:
        q, r = mx.linalg.qr(flat, stream=mx.cpu)
    q = q * mx.sign(mx.diag(r))
    if transposed:
        q = q.T
    module.weight = gain * q
    module.bias = mx.zeros_like(module.bias)
    return module


def polyak_update(params, target_params, tau):
    return tree_map(
        lambda p, t: (1.0 - tau) * t + tau * p, params, target_params
    )


def explained_variance(y_pred, y_true):
    var_y = mx.var(y_true).item()
    if var_y < 1e-12:
        return float("nan")
    return 1.0 - mx.var(y_true - y_pred).item() / var_y


class DimEnv:
    """Minimal env stub so algo constructors only need shapes / bounds."""

    def __init__(
        self,
        n_envs: int,
        obs_dim: int,
        action_dim: int = 1,
        *,
        is_discrete: bool = False,
        n_actions: int | None = None,
        action_low: float = -1.0,
        action_high: float = 1.0,
    ):
        self.num_envs = n_envs
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.is_discrete = is_discrete
        self.n_actions = n_actions if n_actions is not None else action_dim
        self.action_low = action_low
        self.action_high = action_high
