"""Vectorized Pendulum-v1 environment in pure MLX (exact gym dynamics)."""

import mlx.core as mx

from scripts.envs.base import StepResult


def angle_normalize(x):
    return ((x + mx.pi) % (2 * mx.pi)) - mx.pi


class PendulumEnv:
    max_speed = 8.0
    max_torque = 2.0
    dt = 0.05
    g = 10.0
    m = 1.0
    l = 1.0
    max_episode_steps = 200

    obs_dim = 3
    action_dim = 1
    is_discrete = False
    action_low = -2.0
    action_high = 2.0

    def __init__(self, n_envs: int = 1, seed: int = 0):
        self.num_envs = n_envs
        self._key = mx.random.key(seed)
        # state: [theta, theta_dot]
        self.state = mx.zeros((n_envs, 2))
        self.steps = mx.zeros((n_envs,), dtype=mx.int32)
        self._ep_ret = mx.zeros((n_envs,))
        self._ep_len = mx.zeros((n_envs,), dtype=mx.int32)
        self._step_fn = mx.compile(self._physics)

    def _split_key(self):
        self._key, sub = mx.random.split(self._key)
        return sub

    def _sample_state(self, n, key=None):
        key = key if key is not None else self._split_key()
        k1, k2 = mx.random.split(key)
        th = mx.random.uniform(low=-mx.pi, high=mx.pi, shape=(n, 1), key=k1)
        thdot = mx.random.uniform(low=-1.0, high=1.0, shape=(n, 1), key=k2)
        return mx.concatenate([th, thdot], axis=1).astype(mx.float32)

    def _obs(self, state):
        th, thdot = state[:, 0], state[:, 1]
        return mx.stack([mx.cos(th), mx.sin(th), thdot], axis=1).astype(mx.float32)

    def _physics(self, state, steps, ep_ret, ep_len, actions, key):
        th, thdot = state[:, 0], state[:, 1]
        u = mx.clip(actions.reshape(-1), -self.max_torque, self.max_torque)

        cost = (
            angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * u**2
        )
        newthdot = thdot + (
            3 * self.g / (2 * self.l) * mx.sin(th)
            + 3.0 / (self.m * self.l**2) * u
        ) * self.dt
        newthdot = mx.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * self.dt

        new_state = mx.stack([newth, newthdot], axis=1).astype(mx.float32)
        steps = steps + 1

        reward = -cost.astype(mx.float32)
        terminated = mx.zeros((self.num_envs,), dtype=mx.bool_)
        truncated = steps >= self.max_episode_steps
        done = truncated  # never terminates early

        ep_ret = ep_ret + reward
        ep_len = ep_len + 1
        res_ret = mx.where(done, ep_ret, mx.zeros_like(ep_ret))
        res_len = mx.where(done, ep_len, mx.zeros_like(ep_len))

        resets = self._sample_state(self.num_envs, key)
        state = mx.where(done[:, None], resets, new_state)
        steps = mx.where(done, mx.zeros_like(steps), steps)
        ep_ret = mx.where(done, mx.zeros_like(ep_ret), ep_ret)
        ep_len = mx.where(done, mx.zeros_like(ep_len), ep_len)

        return (
            state, steps, ep_ret, ep_len,
            self._obs(state), reward, done, terminated, truncated,
            self._obs(new_state),
            res_ret, res_len,
        )

    def reset(self):
        self.state = self._sample_state(self.num_envs)
        self.steps = mx.zeros((self.num_envs,), dtype=mx.int32)
        self._ep_ret = mx.zeros((self.num_envs,))
        self._ep_len = mx.zeros((self.num_envs,), dtype=mx.int32)
        return self._obs(self.state)

    def step(self, actions):
        self._key, sub = mx.random.split(self._key)
        (
            self.state, self.steps, self._ep_ret, self._ep_len,
            obs, reward, done, terminated, truncated, terminal_obs,
            ep_ret, ep_len,
        ) = self._step_fn(
            self.state, self.steps, self._ep_ret, self._ep_len, actions, sub
        )
        return StepResult(
            obs=obs,
            reward=reward,
            done=done,
            terminated=terminated,
            truncated=truncated,
            terminal_obs=terminal_obs,
            ep_ret=ep_ret,
            ep_len=ep_len,
        )
