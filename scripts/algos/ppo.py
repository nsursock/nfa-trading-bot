"""PPO in Apple MLX, SB3-style. Train: python -m scripts.algos.ppo [--config ...]"""

import argparse
import math
import time
from collections import deque
from functools import partial
from typing import Literal

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optimizers
from pydantic import model_validator

from scripts.algos.common import (
    BaseAlgoConfig, StatsLogger, explained_variance, make_env, orthogonal_init,
)


class PPOConfig(BaseAlgoConfig):
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: float | None = None
    normalize_advantage: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    net_arch: list[int] = [64, 64]
    activation: Literal["tanh", "relu"] = "tanh"
    total_timesteps: int = 100_000
    env_id: str = "CartPole-v1"

    @model_validator(mode="after")
    def check_rollout_divisible(self):
        if (self.n_steps * self.n_envs) % self.batch_size != 0:
            raise ValueError(
                f"n_steps*n_envs ({self.n_steps * self.n_envs}) must be divisible "
                f"by batch_size ({self.batch_size})"
            )
        return self


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, activation, out_gain):
        super().__init__()
        dims = [in_dim] + list(hidden)
        self.layers = []
        for i in range(len(dims) - 1):
            self.layers.append(
                orthogonal_init(nn.Linear(dims[i], dims[i + 1]), math.sqrt(2))
            )
        self.head = orthogonal_init(nn.Linear(dims[-1], out_dim), out_gain)
        self.act = nn.tanh if activation == "tanh" else nn.relu

    def __call__(self, x):
        for lin in self.layers:
            x = self.act(lin(x))
        return self.head(x)


class ActorCritic(nn.Module):
    """Separate actor/critic MLPs. Categorical policy head for discrete
    actions; a Gaussian head can be added alongside later."""

    def __init__(self, obs_dim, n_actions, net_arch, activation):
        super().__init__()
        self.actor = MLP(obs_dim, net_arch, n_actions, activation, 0.01)
        self.critic = MLP(obs_dim, net_arch, 1, activation, 1.0)

    def logits(self, obs):
        return self.actor(obs)

    def value(self, obs):
        return self.critic(obs).squeeze(-1)


def _log_softmax(logits):
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


class PPO:
    def __init__(self, config: PPOConfig, env):
        self.config = config
        self.env = env
        if config.seed is not None:
            mx.random.seed(config.seed)
        self.policy = ActorCritic(
            env.obs_dim, env.n_actions, config.net_arch, config.activation
        )
        self.optimizer = optimizers.Adam(
            learning_rate=config.learning_rate, eps=1e-5
        )
        self.ep_info_buffer = deque(maxlen=config.stats_window_size)

        loss_and_grad = nn.value_and_grad(self.policy, self._loss_and_aux)
        state = [self.policy.state, self.optimizer.state]

        @partial(mx.compile, inputs=state, outputs=state)
        def _step(obs, actions, old_logp, adv, returns, old_values):
            (loss, aux), grads = loss_and_grad(
                obs, actions, old_logp, adv, returns, old_values
            )
            grads, _ = optimizers.clip_grad_norm(grads, self.config.max_grad_norm)
            self.optimizer.update(self.policy, grads)
            return mx.stack([loss, *aux])

        self._step = _step

        @partial(
            mx.compile,
            inputs=[self.policy.state, mx.random.state],
            outputs=[self.policy.state, mx.random.state],
        )
        def _act(obs):
            logits = self.policy.logits(obs)
            actions = mx.random.categorical(logits)
            logp_all = _log_softmax(logits)
            logp = mx.take_along_axis(
                logp_all, actions[:, None], axis=1
            ).squeeze(-1)
            values = self.policy.value(obs)
            return actions, logp, values

        self._act_c = _act

    # ---- acting ----
    def _act(self, obs):
        return self._act_c(obs)

    def predict(self, obs, deterministic=True):
        obs = mx.asarray(obs, dtype=mx.float32)
        if obs.ndim == 1:
            obs = obs[None, :]
        logits = self.policy.logits(obs)
        if deterministic:
            return mx.argmax(logits, axis=-1)
        return mx.random.categorical(logits)

    # ---- rollout ----
    def _collect_rollout(self, obs, last_done):
        cfg = self.config
        n_steps = cfg.n_steps
        obs_l, act_l, logp_l, val_l, rew_l, start_l = [], [], [], [], [], []
        done_l, ep_ret_l, ep_len_l = [], [], []

        for t in range(n_steps):
            actions, logp, values = self._act(obs)
            res = self.env.step(actions)

            # bootstrap value of truncated episodes (SB3 behavior)
            tv = self.policy.value(res.terminal_obs)
            rewards = res.reward + cfg.gamma * tv * res.truncated.astype(mx.float32)

            obs_l.append(obs)
            act_l.append(actions)
            logp_l.append(logp)
            val_l.append(values)
            rew_l.append(rewards)
            start_l.append(last_done.astype(mx.float32))
            done_l.append(res.done)
            ep_ret_l.append(res.ep_ret)
            ep_len_l.append(res.ep_len)
            obs, last_done = res.obs, res.done

        obs_b, act_b, logp_b, val_b, rew_b, start_b = (
            mx.stack(lst)
            for lst in (obs_l, act_l, logp_l, val_l, rew_l, start_l)
        )
        done_b = mx.stack(done_l)
        ep_ret_b = mx.stack(ep_ret_l)
        ep_len_b = mx.stack(ep_len_l)
        mx.eval(
            obs_b, act_b, logp_b, val_b, rew_b, start_b,
            done_b, ep_ret_b, ep_len_b,
        )
        d = done_b.reshape(-1).tolist()
        r = ep_ret_b.reshape(-1).tolist()
        l = ep_len_b.reshape(-1).tolist()
        self.ep_info_buffer.extend(
            (ri, li) for ri, li, di in zip(r, l, d) if di
        )
        last_values = self.policy.value(obs)
        return obs, last_done, (obs_b, act_b, logp_b, val_b, rew_b, start_b), last_values

    def _gae(self, val_b, rew_b, start_b, last_values, last_done):
        cfg = self.config
        n_steps = cfg.n_steps
        adv = mx.zeros_like(rew_b)
        last_gae = mx.zeros((self.env.num_envs,))
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_non_terminal = 1.0 - last_done.astype(mx.float32)
                next_value = last_values
            else:
                next_non_terminal = 1.0 - start_b[t + 1]
                next_value = val_b[t + 1]
            delta = rew_b[t] + cfg.gamma * next_value * next_non_terminal - val_b[t]
            last_gae = delta + cfg.gamma * cfg.gae_lambda * next_non_terminal * last_gae
            adv[t] = last_gae
        return adv, adv + val_b

    # ---- update ----
    def _loss_and_aux(
        self, obs, actions, old_logp, adv, returns, old_values
    ):
        cfg = self.config
        policy = self.policy
        logits = policy.logits(obs)
        logp_all = _log_softmax(logits)
        new_logp = mx.take_along_axis(logp_all, actions[:, None], axis=1).squeeze(-1)
        values = policy.value(obs)

        a = adv
        if cfg.normalize_advantage:
            a = (a - mx.mean(a)) / (mx.std(a) + 1e-8)

        log_ratio = new_logp - old_logp
        ratio = mx.exp(log_ratio)
        pg1 = -a * ratio
        pg2 = -a * mx.clip(ratio, 1 - cfg.clip_range, 1 + cfg.clip_range)
        policy_loss = mx.mean(mx.maximum(pg1, pg2))

        if cfg.clip_range_vf is None:
            value_loss = mx.mean((returns - values) ** 2)
        else:
            v_clipped = old_values + mx.clip(
                values - old_values, -cfg.clip_range_vf, cfg.clip_range_vf
            )
            value_loss = mx.mean(
                mx.maximum((returns - values) ** 2, (returns - v_clipped) ** 2)
            )

        entropy = -mx.mean(mx.sum(mx.exp(logp_all) * logp_all, axis=-1))
        entropy_loss = -entropy
        loss = policy_loss + cfg.ent_coef * entropy_loss + cfg.vf_coef * value_loss

        approx_kl = mx.mean((ratio - 1) - log_ratio)
        clip_fraction = mx.mean(
            (mx.abs(ratio - 1) > cfg.clip_range).astype(mx.float32)
        )
        return loss, (policy_loss, value_loss, entropy_loss, approx_kl, clip_fraction)

    def _update(self, buffer, adv, returns):
        cfg = self.config
        obs_b, act_b, logp_b, val_b, _, _ = buffer
        n = cfg.n_steps * self.env.num_envs
        flat = lambda x: x.reshape(n, *x.shape[2:])
        obs_f, act_f = flat(obs_b), flat(act_b).astype(mx.int32)
        logp_f, val_f = flat(logp_b), flat(val_b)
        adv_f, ret_f = flat(adv), flat(returns)

        outs = []
        n_updates = 0
        stop = False
        for _ in range(cfg.n_epochs):
            perm = mx.random.permutation(n)
            for start in range(0, n, cfg.batch_size):
                idx = perm[start : start + cfg.batch_size]
                batch = (
                    obs_f[idx], act_f[idx], logp_f[idx], adv_f[idx], ret_f[idx],
                    val_f[idx],
                )
                if cfg.target_kl is not None:
                    kl = self._loss_and_aux(*batch)[1][3].item()
                    if kl > 1.5 * cfg.target_kl:
                        stop = True
                        break
                out = self._step(*batch)
                mx.eval(self.policy.parameters(), self.optimizer.state, out)
                outs.append(out)
                n_updates += 1
            if stop:
                break
        keys = ("loss", "pg", "vf", "ent", "kl", "clip")
        if outs:
            means = mx.mean(mx.stack(outs), axis=0).tolist()
        else:
            means = [float("nan")] * len(keys)
        return dict(zip(keys, means)), n_updates

    # ---- learn ----
    def learn(self, callback=None):
        cfg = self.config
        header = [
            "time/iterations", "time/total_timesteps", "time/fps",
            "time/time_elapsed", "rollout/ep_rew_mean", "rollout/ep_len_mean",
            "train/approx_kl", "train/clip_fraction", "train/clip_range",
            "train/entropy_loss", "train/explained_variance",
            "train/learning_rate", "train/loss", "train/n_updates",
            "train/policy_gradient_loss", "train/value_loss",
        ]
        logger = StatsLogger(
            f"{cfg.log_dir}/stats_ppo_{cfg.env_id}.csv", header, cfg.verbose
        )

        obs = self.env.reset()
        last_done = mx.zeros((self.env.num_envs,), dtype=mx.bool_)
        rollout_size = cfg.n_steps * self.env.num_envs
        n_iterations = max(1, cfg.total_timesteps // rollout_size)
        total_ts = 0
        total_updates = 0
        t0 = time.time()

        for it in range(1, n_iterations + 1):
            obs, last_done, buffer, last_values = self._collect_rollout(
                obs, last_done
            )
            adv, returns = self._gae(
                buffer[3], buffer[4], buffer[5], last_values, last_done
            )
            train_stats, n_updates = self._update(buffer, adv, returns)
            total_ts += rollout_size
            total_updates += n_updates
            elapsed = time.time() - t0
            fps = int(total_ts / elapsed) if elapsed > 0 else 0

            if self.ep_info_buffer:
                ep_rew = sum(e[0] for e in self.ep_info_buffer) / len(
                    self.ep_info_buffer
                )
                ep_len = sum(e[1] for e in self.ep_info_buffer) / len(
                    self.ep_info_buffer
                )
            else:
                ep_rew = ep_len = ""

            ev = explained_variance(buffer[3].reshape(-1), returns.reshape(-1))

            logger.log({
                "time/iterations": it,
                "time/total_timesteps": total_ts,
                "time/fps": fps,
                "time/time_elapsed": round(elapsed, 2),
                "rollout/ep_rew_mean": round(ep_rew, 2) if ep_rew != "" else "",
                "rollout/ep_len_mean": round(ep_len, 2) if ep_len != "" else "",
                "train/approx_kl": round(train_stats["kl"], 6),
                "train/clip_fraction": round(train_stats["clip"], 4),
                "train/clip_range": cfg.clip_range,
                "train/entropy_loss": round(train_stats["ent"], 6),
                "train/explained_variance": round(ev, 4),
                "train/learning_rate": cfg.learning_rate,
                "train/loss": round(train_stats["loss"], 6),
                "train/n_updates": n_updates,
                "train/policy_gradient_loss": round(train_stats["pg"], 6),
                "train/value_loss": round(train_stats["vf"], 6),
            })

            if callback is not None and callback(total_ts, total_updates):
                break

        logger.close()
        return self


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/ppo_cartpole.yaml")
    args = p.parse_args()
    cfg = PPOConfig.from_yaml(args.config)
    env = make_env(cfg.env_id, n_envs=cfg.n_envs, seed=cfg.seed or 0)
    PPO(cfg, env).learn()


if __name__ == "__main__":
    main()
