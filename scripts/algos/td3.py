"""TD3 in Apple MLX, SB3-style. Train: python -m scripts.algos.td3 [--config ...]"""

import argparse
import time
from collections import deque

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optimizers

from scripts.algos.common import (
    BaseAlgoConfig, MLP, ReplayBuffer, StatsLogger, make_env, polyak_update,
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


class Critics(nn.Module):
    def __init__(self, obs_dim, action_dim, net_arch):
        super().__init__()
        self.q1 = MLP(obs_dim + action_dim, net_arch, 1, activation="relu")
        self.q2 = MLP(obs_dim + action_dim, net_arch, 1, activation="relu")

    def __call__(self, obs, action):
        x = mx.concatenate([obs, action], axis=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


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
        self.critics = Critics(obs_dim, act_dim, config.net_arch)
        self.critics_target = Critics(obs_dim, act_dim, config.net_arch)
        self.critics_target.update(self.critics.parameters())

        self.opt_actor = optimizers.Adam(learning_rate=config.learning_rate)
        self.opt_critic = optimizers.Adam(learning_rate=config.learning_rate)

        self.buffer = ReplayBuffer(config.buffer_size, obs_dim, act_dim)
        self.ep_info_buffer = deque(maxlen=config.stats_window_size)
        self.act_low = env.action_low
        self.act_high = env.action_high

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

        def critic_loss_fn(critics):
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
            q1, q2 = critics(obs, act)
            return mx.mean((q1 - y) ** 2) + mx.mean((q2 - y) ** 2)

        critic_loss, cgrads = nn.value_and_grad(self.critics, critic_loss_fn)(
            self.critics
        )
        self.opt_critic.update(self.critics, cgrads)

        self._n_updates += 1
        actor_loss = None
        if self._n_updates % cfg.policy_delay == 0:
            def actor_loss_fn(actor):
                q1, _ = self.critics(obs, actor(obs))
                return -mx.mean(q1)

            actor_loss, agrads = nn.value_and_grad(self.actor, actor_loss_fn)(
                self.actor
            )
            self.opt_actor.update(self.actor, agrads)
            self.critics_target.update(
                polyak_update(
                    self.critics.parameters(),
                    self.critics_target.parameters(),
                    cfg.tau,
                )
            )
            self.actor_target.update(
                polyak_update(
                    self.actor.parameters(),
                    self.actor_target.parameters(),
                    cfg.tau,
                )
            )
        mx.eval(
            self.actor.parameters(),
            self.critics.parameters(),
            self.critics_target.parameters(),
            self.actor_target.parameters(),
        )
        return actor_loss, critic_loss

    def learn(self):
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

            next_obs, reward, done, infos = self.env.step(self._scale(action))

            real_done = mx.zeros((n_envs,), dtype=mx.float32)
            buf_next_obs = next_obs
            done_list = done.tolist()
            if any(done_list):
                for i, info in enumerate(infos):
                    if "episode" in info:
                        self.ep_info_buffer.append(
                            (info["episode"]["r"], info["episode"]["l"])
                        )
                        n_episodes += 1
                        buf_next_obs = mx.where(
                            (mx.arange(n_envs) == i)[:, None],
                            info["terminal_observation"][None, :],
                            buf_next_obs,
                        )
                        if not info.get("TimeLimit.truncated", False):
                            real_done = mx.where(
                                mx.arange(n_envs) == i,
                                mx.ones_like(real_done),
                                real_done,
                            )
            self.buffer.add(obs, buf_next_obs, action, reward, real_done)
            obs = next_obs
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
                if len(acc["critic"]) >= 64:
                    acc = {
                        k: ([mx.mean(mx.stack(v))] if v else [])
                        for k, v in acc.items()
                    }

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
