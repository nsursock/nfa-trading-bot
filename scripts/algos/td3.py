"""TD3 in Apple MLX, SB3-style. Train: python -m scripts.algos.td3 [--config ...]"""

import argparse
import time
from collections import deque
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optimizers

from scripts.algos.common import (
    BaseAlgoConfig, MLP, ReplayBuffer, StatsLogger, TwinCritics, make_env,
    polyak_update,
)

HEADER = [
    "time/episodes", "time/fps", "time/time_elapsed", "time/total_timesteps",
    "rollout/ep_rew_mean", "rollout/ep_len_mean", "train/actor_loss",
    "train/critic_loss", "train/learning_rate", "train/n_updates",
]


class TD3Config(BaseAlgoConfig):
    learning_rate: float = 1e-3
    buffer_size: int = 1_000_000
    learning_starts: int = 100
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 1
    policy_delay: int = 2
    target_policy_noise: float = 0.2
    target_noise_clip: float = 0.5
    action_noise_std: float = 0.1
    net_arch: list[int] = [400, 300]
    log_interval: int = 4
    env_id: str = "Pendulum-v1"
    total_timesteps: int = 30_000


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, net_arch):
        super().__init__()
        self.net = MLP(obs_dim, net_arch, action_dim, activation="relu",
                       output_activation="tanh")

    def __call__(self, obs):
        return self.net(obs)


class TD3:
    def __init__(self, config: TD3Config, env):
        self.config = config
        self.env = env
        if config.seed is not None:
            mx.random.seed(config.seed)
        obs_dim, act_dim = env.obs_dim, env.action_dim
        self.actor = Actor(obs_dim, act_dim, config.net_arch)
        self.actor_target = Actor(obs_dim, act_dim, config.net_arch)
        self.actor_target.update(self.actor.parameters())
        self.critics = TwinCritics(obs_dim, act_dim, config.net_arch)
        self.critics_target = TwinCritics(obs_dim, act_dim, config.net_arch)
        self.critics_target.update(self.critics.parameters())

        self.opt_actor = optimizers.Adam(learning_rate=config.learning_rate)
        self.opt_critic = optimizers.Adam(learning_rate=config.learning_rate)

        self.buffer = ReplayBuffer(config.buffer_size, obs_dim, act_dim)
        self.ep_info_buffer = deque(maxlen=config.stats_window_size)
        self.act_low = env.action_low
        self.act_high = env.action_high
        self._n_updates = 0

        state = [
            self.actor.state,
            self.actor_target.state,
            self.critics.state,
            self.critics_target.state,
            self.opt_critic.state,
            self.opt_actor.state,
            mx.random.state,
        ]
        critic_vg = nn.value_and_grad(self.critics, self._critic_loss)
        actor_vg = nn.value_and_grad(self.actor, self._actor_loss)

        def _make_step(do_actor):
            @partial(mx.compile, inputs=state, outputs=state)
            def step(obs, next_obs, act, rew, done):
                critic_loss, cgrads = critic_vg(obs, next_obs, act, rew, done)
                self.opt_critic.update(self.critics, cgrads)
                if not do_actor:
                    return critic_loss
                actor_loss, agrads = actor_vg(obs)
                self.opt_actor.update(self.actor, agrads)
                self.critics_target.update(
                    polyak_update(
                        self.critics.parameters(),
                        self.critics_target.parameters(),
                        self.config.tau,
                    )
                )
                self.actor_target.update(
                    polyak_update(
                        self.actor.parameters(),
                        self.actor_target.parameters(),
                        self.config.tau,
                    )
                )
                return actor_loss, critic_loss

            return step

        self._step = {True: _make_step(True), False: _make_step(False)}

    def _critic_loss(self, obs, next_obs, act, rew, done):
        cfg = self.config
        noise = mx.clip(
            mx.random.normal(act.shape) * cfg.target_policy_noise,
            -cfg.target_noise_clip,
            cfg.target_noise_clip,
        )
        next_a = mx.clip(self.actor_target(next_obs) + noise, -1.0, 1.0)
        tq1, tq2 = self.critics_target(next_obs, next_a)
        y = mx.stop_gradient(
            rew.squeeze(-1)
            + (1 - done.squeeze(-1)) * cfg.gamma * mx.minimum(tq1, tq2)
        )
        q1, q2 = self.critics(obs, act)
        return mx.mean((q1 - y) ** 2) + mx.mean((q2 - y) ** 2)

    def _actor_loss(self, obs):
        q1, _ = self.critics(obs, self.actor(obs))
        return -mx.mean(q1)

    def _scale(self, a):
        return a * (self.act_high - self.act_low) / 2 + (
            self.act_high + self.act_low
        ) / 2

    def predict(self, obs, deterministic=True):
        obs = mx.asarray(obs, dtype=mx.float32)
        if obs.ndim == 1:
            obs = obs[None, :]
        a = self.actor(obs)
        if not deterministic:
            a = a + mx.random.normal(a.shape) * self.config.action_noise_std
            a = mx.clip(a, -1.0, 1.0)
        return self._scale(a)

    def _update(self):
        cfg = self.config
        obs, next_obs, act, rew, done = self.buffer.sample(cfg.batch_size)

        self._n_updates += 1
        if self._n_updates % cfg.policy_delay == 0:
            actor_loss, critic_loss = self._step[True](
                obs, next_obs, act, rew, done
            )
        else:
            critic_loss = self._step[False](obs, next_obs, act, rew, done)
            actor_loss = None
        return actor_loss, critic_loss

    def learn(self, callback=None):
        cfg = self.config
        logger = StatsLogger(
            f"{cfg.log_dir}/stats_td3_{cfg.env_id}.csv", HEADER, cfg.verbose
        )
        obs = self.env.reset()
        n_envs = self.env.num_envs
        num_timesteps = 0
        n_episodes = 0
        t0 = time.time()
        self._n_updates = 0
        acc = {"actor": [], "critic": []}

        while num_timesteps < cfg.total_timesteps:
            if num_timesteps < cfg.learning_starts:
                action = mx.random.uniform(
                    low=-1.0, high=1.0, shape=(n_envs, self.env.action_dim)
                )
            else:
                action = self.actor(obs) + mx.random.normal(
                    (n_envs, self.env.action_dim)
                ) * cfg.action_noise_std
                action = mx.clip(action, -1.0, 1.0)

            res = self.env.step(self._scale(action))
            self.buffer.add(
                obs, res.terminal_obs, action, res.reward,
                res.terminated.astype(mx.float32),
            )
            obs = res.obs
            num_timesteps += n_envs

            if num_timesteps >= cfg.learning_starts and (
                num_timesteps % cfg.train_freq == 0
            ):
                n_grad = (
                    cfg.train_freq if cfg.gradient_steps == -1
                    else cfg.gradient_steps
                )
                for _ in range(n_grad):
                    al, cl = self._update()
                    if al is not None:
                        acc["actor"].append(al)
                    acc["critic"].append(cl)
                mx.eval(
                    self.actor.parameters(),
                    self.critics.parameters(),
                    self.critics_target.parameters(),
                    self.actor_target.parameters(),
                    *acc["actor"], *acc["critic"],
                )
                if len(acc["critic"]) >= 64:
                    acc = {
                        k: ([mx.mean(mx.stack(v))] if v else [])
                        for k, v in acc.items()
                    }

            fin = mx.stack([
                res.done.astype(mx.float32), res.ep_ret,
                res.ep_len.astype(mx.float32),
            ]).tolist()
            for d_, r_, l_ in zip(*fin):
                if d_:
                    self.ep_info_buffer.append((r_, int(l_)))
                    n_episodes += 1

            if n_episodes > 0 and n_episodes % cfg.log_interval == 0 and (
                n_episodes != getattr(self, "_last_logged_ep", 0)
            ):
                self._last_logged_ep = n_episodes
                elapsed = time.time() - t0
                means = {}
                for k, v in acc.items():
                    means[k] = mx.mean(mx.stack(v)).item() if v else ""
                acc = {"actor": [], "critic": []}
                ep_rew = ep_len = ""
                if self.ep_info_buffer:
                    ep_rew = sum(e[0] for e in self.ep_info_buffer) / len(
                        self.ep_info_buffer
                    )
                    ep_len = sum(e[1] for e in self.ep_info_buffer) / len(
                        self.ep_info_buffer
                    )
                logger.log({
                    "time/episodes": n_episodes,
                    "time/fps": int(num_timesteps / elapsed) if elapsed else 0,
                    "time/time_elapsed": round(elapsed, 2),
                    "time/total_timesteps": num_timesteps,
                    "rollout/ep_rew_mean": round(ep_rew, 2) if ep_rew != "" else "",
                    "rollout/ep_len_mean": round(ep_len, 2) if ep_len != "" else "",
                    "train/actor_loss": round(means["actor"], 4)
                    if means["actor"] != "" else "",
                    "train/critic_loss": round(means["critic"], 4)
                    if means["critic"] != "" else "",
                    "train/learning_rate": cfg.learning_rate,
                    "train/n_updates": self._n_updates,
                })
            if callback is not None and callback(num_timesteps, self._n_updates):
                break
        logger.close()
        return self


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/td3_pendulum.yaml")
    args = p.parse_args()
    cfg = TD3Config.from_yaml(args.config)
    env = make_env(cfg.env_id, n_envs=cfg.n_envs, seed=cfg.seed or 0)
    TD3(cfg, env).learn()


if __name__ == "__main__":
    main()
