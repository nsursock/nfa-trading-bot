"""Vectorized CartPole-v1 environment in pure MLX.

Reproduces gymnasium's CartPole-v1 dynamics (Euler integration) with
SB3 VecEnv semantics: auto-reset on done, episode stats in infos.
"""

import mlx.core as mx


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

    def _split_key(self):
        self._key, sub = mx.random.split(self._key)
        return sub

    def _sample_state(self, n):
        return mx.random.uniform(
            low=-0.05, high=0.05, shape=(n, 4), key=self._split_key()
        ).astype(mx.float32)

    def reset(self):
        self.state = self._sample_state(self.num_envs)
        self.steps = mx.zeros((self.num_envs,), dtype=mx.int32)
        self._ep_ret = mx.zeros((self.num_envs,))
        self._ep_len = mx.zeros((self.num_envs,), dtype=mx.int32)
        return self.state

    def step(self, actions):
        x, x_dot, theta, theta_dot = (
            self.state[:, 0],
            self.state[:, 1],
            self.state[:, 2],
            self.state[:, 3],
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
        self.steps = self.steps + 1

        terminated = (mx.abs(x) > self.x_threshold) | (
            mx.abs(theta) > self.theta_threshold_radians
        )
        truncated = self.steps >= self.max_episode_steps
        done = terminated | truncated

        reward = mx.ones((self.num_envs,), dtype=mx.float32)
        self._ep_ret = self._ep_ret + reward
        self._ep_len = self._ep_len + 1

        done_list = done.tolist()
        ep_ret = self._ep_ret.tolist()
        ep_len = self._ep_len.tolist()
        term_list = terminated.tolist()
        trunc_list = truncated.tolist()

        infos = []
        for i in range(self.num_envs):
            info = {}
            if done_list[i]:
                info["episode"] = {"r": float(ep_ret[i]), "l": int(ep_len[i])}
                info["terminal_observation"] = new_state[i]
                info["TimeLimit.truncated"] = bool(trunc_list[i]) and not bool(
                    term_list[i]
                )
            infos.append(info)

        # Auto-reset finished envs (SB3 VecEnv semantics)
        if any(done_list):
            resets = self._sample_state(self.num_envs)
            mask = done[:, None]
            self.state = mx.where(mask, resets, new_state)
            self.steps = mx.where(done, mx.zeros_like(self.steps), self.steps)
            self._ep_ret = mx.where(done, mx.zeros_like(self._ep_ret), self._ep_ret)
            self._ep_len = mx.where(done, mx.zeros_like(self._ep_len), self._ep_len)
        else:
            self.state = new_state

        return self.state, reward, done, infos
