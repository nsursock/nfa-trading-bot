"""Vectorized CartPole-v1 environment in pure MLX.

Reproduces gymnasium's CartPole-v1 dynamics (Euler integration) with
SB3 VecEnv semantics: auto-reset on done, episode stats in StepResult.
"""

import mlx.core as mx

from scripts.envs.base import StepResult


class CartPoleEnv:
    gravity = 9.8
    masscart = 1.0
    masspole = 0.1
    total_mass = masspole + masscart
    length = 0.5  # half the pole's length
    polemass_length = masspole * length
    force_mag = 10.0
    tau = 0.02

    theta_threshold_radians = 12 * 2 * mx.pi / 360
    x_threshold = 2.4
    max_episode_steps = 500

    obs_dim = 4
    n_actions = 2
    is_discrete = True

    def __init__(self, n_envs: int = 1, seed: int = 0):
        self.num_envs = n_envs
        self._key = mx.random.key(seed)
        self.state = mx.zeros((n_envs, 4))
        self.steps = mx.zeros((n_envs,), dtype=mx.int32)
        self._ep_ret = mx.zeros((n_envs,))
        self._ep_len = mx.zeros((n_envs,), dtype=mx.int32)
        self._step_fn = mx.compile(self._physics)

    def _split_key(self):
        self._key, sub = mx.random.split(self._key)
        return sub

    def _sample_state(self, n, key=None):
        return mx.random.uniform(
            low=-0.05, high=0.05, shape=(n, 4),
            key=key if key is not None else self._split_key(),
        ).astype(mx.float32)

    def _physics(self, state, steps, ep_ret, ep_len, actions, key):
        x, x_dot, theta, theta_dot = (
            state[:, 0],
            state[:, 1],
            state[:, 2],
            state[:, 3],
        )

        force = mx.where(actions == 1, self.force_mag, -self.force_mag)
        costheta = mx.cos(theta)
        sintheta = mx.sin(theta)

        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        new_state = mx.stack([x, x_dot, theta, theta_dot], axis=1).astype(mx.float32)
        steps = steps + 1

        terminated = (mx.abs(x) > self.x_threshold) | (
            mx.abs(theta) > self.theta_threshold_radians
        )
        truncated = (steps >= self.max_episode_steps) & ~terminated
        done = terminated | truncated

        reward = mx.ones((self.num_envs,), dtype=mx.float32)
        ep_ret = ep_ret + reward
        ep_len = ep_len + 1
        res_ret = mx.where(done, ep_ret, mx.zeros_like(ep_ret))
        res_len = mx.where(done, ep_len, mx.zeros_like(ep_len))

        # Auto-reset finished envs (SB3 VecEnv semantics)
        resets = self._sample_state(self.num_envs, key)
        state = mx.where(done[:, None], resets, new_state)
        steps = mx.where(done, mx.zeros_like(steps), steps)
        ep_ret = mx.where(done, mx.zeros_like(ep_ret), ep_ret)
        ep_len = mx.where(done, mx.zeros_like(ep_len), ep_len)

        return (
            state, steps, ep_ret, ep_len,
            reward, done, terminated, truncated, new_state, res_ret, res_len,
        )

    def reset(self):
        self.state = self._sample_state(self.num_envs)
        self.steps = mx.zeros((self.num_envs,), dtype=mx.int32)
        self._ep_ret = mx.zeros((self.num_envs,))
        self._ep_len = mx.zeros((self.num_envs,), dtype=mx.int32)
        return self.state

    def step(self, actions):
        self._key, sub = mx.random.split(self._key)
        (
            self.state, self.steps, self._ep_ret, self._ep_len,
            reward, done, terminated, truncated, terminal_obs,
            ep_ret, ep_len,
        ) = self._step_fn(
            self.state, self.steps, self._ep_ret, self._ep_len, actions, sub
        )
        return StepResult(
            obs=self.state,
            reward=reward,
            done=done,
            terminated=terminated,
            truncated=truncated,
            terminal_obs=terminal_obs,
            ep_ret=ep_ret,
            ep_len=ep_len,
        )
