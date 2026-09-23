"""SAC in Apple MLX, SB3-style. Train: python -m scripts.algos.sac [--config ...]"""

import argparse
import math
import time
from collections import deque
from typing import Literal

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optimizers

from scripts.algos.common import (
    BaseAlgoConfig, MLP, ReplayBuffer, StatsLogger, make_env, polyak_update,
)

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0
HEADER = [
    "time/episodes", "time/fps", "time/time_elapsed", "time/total_timesteps",
    "rollout/ep_rew_mean", "rollout/ep_len_mean", "train/actor_loss",
    "train/critic_loss", "train/ent_coef", "train/ent_coef_loss",
    "train/learning_rate", "train/n_updates",
]


class SACConfig(BaseAlgoConfig):
    learning_rate: float = 3e-4
    buffer_size: int = 1_000_000
    learning_starts: int = 100
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 1
    ent_coef: Literal["auto"] | float = "auto"
    target_update_interval: int = 1
    target_entropy: Literal["auto"] | float = "auto"
    net_arch: list[int] = [256, 256]
    log_interval: int = 4
    env_id: str = "Pendulum-v1"
    total_timesteps: int = 30_000


class EntCoef(nn.Module):
    def __init__(self, init_value=1.0):
        super().__init__()
        self.log_ent_coef = mx.array(math.log(init_value))


class Actor(nn.Module):
    """Squashed-Gaussian actor: trunk -> mean and log_std heads."""

    def __init__(self, obs_dim, action_dim, net_arch):
        super().__init__()
        self.trunk = MLP(obs_dim, net_arch[:-1] if len(net_arch) > 1 else [],
                         net_arch[-1], activation="relu")
        self.mu = nn.Linear(net_arch[-1], action_dim)
        self.log_std = nn.Linear(net_arch[-1], action_dim)

    def _dist(self, obs):
        f = self.trunk(obs)
        mu = self.mu(f)
        log_std = mx.clip(self.log_std(f), LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, obs):
        mu, log_std = self._dist(obs)
        std = mx.exp(log_std)
        eps = mx.random.normal(mu.shape)
        u = mu + std * eps
        a = mx.tanh(u)
        # log prob of the squashed Gaussian (SB3 formulation)
        logp_gauss = (
            -0.5 * (eps**2) - log_std - 0.5 * math.log(2 * math.pi)
        ).sum(axis=-1, keepdims=True)
        logp = logp_gauss - mx.log(1 - a**2 + 1e-6).sum(axis=-1, keepdims=True)
        return a, logp.squeeze(-1)

    def deterministic(self, obs):
        mu, _ = self._dist(obs)
        return mx.tanh(mu)


class Critics(nn.Module):
    def __init__(self, obs_dim, action_dim, net_arch):
        super().__init__()
        self.q1 = MLP(obs_dim + action_dim, net_arch, 1, activation="relu")
        self.q2 = MLP(obs_dim + action_dim, net_arch, 1, activation="relu")

    def __call__(self, obs, action):
        x = mx.concatenate([obs, action], axis=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class SAC:
    def __init__(self, config: SACConfig, env):
        self.config = config
        self.env = env
        if config.seed is not None:
            mx.random.seed(config.seed)
        obs_dim, act_dim = env.obs_dim, env.action_dim
        self.actor = Actor(obs_dim, act_dim, config.net_arch)
        self.critics = Critics(obs_dim, act_dim, config.net_arch)
        self.critics_target = Critics(obs_dim, act_dim, config.net_arch)
        self.critics_target.update(self.critics.parameters())

        self.opt_actor = optimizers.Adam(learning_rate=config.learning_rate)
        self.opt_critic = optimizers.Adam(learning_rate=config.learning_rate)

        if config.ent_coef == "auto":
            self.ent_module = EntCoef(1.0)
            self.log_ent_coef = self.ent_module.log_ent_coef
            self.opt_ent = optimizers.Adam(learning_rate=config.learning_rate)
        else:
            self.ent_module = None
            self.log_ent_coef = mx.array(math.log(config.ent_coef))
            self.opt_ent = None
        self.target_entropy = (
            -float(act_dim)
            if config.target_entropy == "auto"
            else float(config.target_entropy)
        )

        self.buffer = ReplayBuffer(config.buffer_size, obs_dim, act_dim)
        self.ep_info_buffer = deque(maxlen=config.stats_window_size)
        self.act_low = env.action_low
        self.act_high = env.action_high

    def _ent_coef(self):
        return mx.stop_gradient(mx.exp(self.log_ent_coef))

    def _scale(self, a):
        return a * (self.act_high - self.act_low) / 2 + (
            self.act_high + self.act_low
        ) / 2

    def predict(self, obs, deterministic=True):
        obs = mx.asarray(obs, dtype=mx.float32)
        if obs.ndim == 1:
            obs = obs[None, :]
        a = (
            self.actor.deterministic(obs)
            if deterministic
            else self.actor.sample(obs)[0]
        )
        return self._scale(a)

    def _update(self):
        cfg = self.config
        ent_coef = self._ent_coef()

        obs, next_obs, act, rew, done = self.buffer.sample(cfg.batch_size)

        def critic_loss_fn(critics):
            next_a, next_logp = self.actor.sample(next_obs)
            tq1, tq2 = self.critics_target(next_obs, next_a)
            target_q = mx.minimum(tq1, tq2) - ent_coef * next_logp
            y = mx.stop_gradient(
                rew.squeeze(-1) + (1 - done.squeeze(-1)) * cfg.gamma * target_q
            )
            q1, q2 = critics(obs, act)
            return 0.5 * (mx.mean((q1 - y) ** 2) + mx.mean((q2 - y) ** 2))

        critic_loss, cgrads = nn.value_and_grad(self.critics, critic_loss_fn)(
            self.critics
        )
        self.opt_critic.update(self.critics, cgrads)

        def actor_loss_fn(actor):
            a_pi, logp = actor.sample(obs)
            q1, q2 = self.critics(obs, a_pi)
            return mx.mean(ent_coef * logp - mx.minimum(q1, q2))

        actor_loss, agrads = nn.value_and_grad(self.actor, actor_loss_fn)(
            self.actor
        )
        self.opt_actor.update(self.actor, agrads)

        ent_loss = mx.array(0.0)
        if self.opt_ent is not None:
            _, logp = self.actor.sample(obs)

            def ent_loss_fn(ec):
                return -mx.mean(
                    ec.log_ent_coef
                    * mx.stop_gradient(logp + self.target_entropy)
                )

            ent_loss, egrads = nn.value_and_grad(self.ent_module, ent_loss_fn)(
                self.ent_module
            )
            self.opt_ent.update(self.ent_module, egrads)
            self.log_ent_coef = self.ent_module.log_ent_coef

        self._n_updates += 1
        if self._n_updates % cfg.target_update_interval == 0:
            self.critics_target.update(
                polyak_update(
                    self.critics.parameters(),
                    self.critics_target.parameters(),
                    cfg.tau,
                )
            )
        mx.eval(
            self.actor.parameters(),
            self.critics.parameters(),
            self.critics_target.parameters(),
            self.log_ent_coef,
        )
        return actor_loss, critic_loss, ent_loss

    # ---- learn ----
    def learn(self):
        cfg = self.config
        logger = StatsLogger(
            f"{cfg.log_dir}/stats_sac_{cfg.env_id}.csv", HEADER, cfg.verbose
        )
        obs = self.env.reset()
        n_envs = self.env.num_envs
        num_timesteps = 0
        n_episodes = 0
        t0 = time.time()
        self._n_updates = 0
        # accumulated losses (mx arrays) since last log row
        acc = {"actor": [], "critic": [], "ent": []}

        while num_timesteps < cfg.total_timesteps:
            if num_timesteps < cfg.learning_starts:
                action = mx.random.uniform(
                    low=-1.0, high=1.0, shape=(n_envs, self.env.action_dim)
                )
            else:
                action, _ = self.actor.sample(obs)

            next_obs, reward, done, infos = self.env.step(self._scale(action))

            real_done = mx.zeros((n_envs,), dtype=mx.float32)
            buf_next_obs = next_obs
            done_list = done.tolist()
            if any(done_list):
                rows = []
                for i, info in enumerate(infos):
                    if "episode" in info:
                        self.ep_info_buffer.append(
                            (info["episode"]["r"], info["episode"]["l"])
                        )
                        n_episodes += 1
                        term_obs = info["terminal_observation"]
                        buf_next_obs = mx.where(
                            (mx.arange(n_envs) == i)[:, None],
                            term_obs[None, :],
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
                    al, cl, el = self._update()
                    acc["actor"].append(al)
                    acc["critic"].append(cl)
                    acc["ent"].append(el)
                if len(acc["actor"]) >= 64:
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
                    means[k] = (
                        mx.mean(mx.stack(v)).item() if v else ""
                    )
                acc = {"actor": [], "critic": [], "ent": []}
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
                    "train/ent_coef": round(
                        mx.exp(self.log_ent_coef).item(), 4
                    ),
                    "train/ent_coef_loss": round(means["ent"], 4)
                    if means["ent"] != "" else "",
                    "train/learning_rate": cfg.learning_rate,
                    "train/n_updates": self._n_updates,
                })
        logger.close()
        return self


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/sac_pendulum.yaml")
    args = p.parse_args()
    cfg = SACConfig.from_yaml(args.config)
    env = make_env(cfg.env_id, n_envs=cfg.n_envs, seed=cfg.seed or 0)
    SAC(cfg, env).learn()


if __name__ == "__main__":
    main()
