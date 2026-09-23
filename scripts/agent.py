"""Hierarchical RL agent: manager sets goals, worker executes trades.

Composes ``scripts.algos`` learners (PPO / SAC / TD3) — no duplicated update math.

Supported manager/worker pairs:
  - PPO / SAC
  - SAC / SAC
  - PPO / TD3

Manager acts every ``manager_horizon`` low-TF steps with continuous goals in
[-1, 1]^goal_dim. Worker receives (market + portfolio + goal). Intrinsic worker
reward is negative L2 distance to the manager goal in achieved-goal space.

Training schedule (``train_schedule``):
  - joint (default): manager and worker update every eligible step
  - alternating: freeze one level while the other updates, cycling
    ``alt_worker_timesteps`` / ``alt_manager_timesteps`` env steps
"""

from __future__ import annotations

import os
import time
from collections import deque
from typing import Literal

import mlx.core as mx
from pydantic import BaseModel, ConfigDict, Field, model_validator
from tqdm import tqdm

from scripts.algos.common import DimEnv, StatsLogger
from scripts.algos.ppo import PPO, PPOConfig, PPO_HEADER
from scripts.algos.sac import HEADER as SAC_HEADER
from scripts.algos.sac import SAC, SACConfig
from scripts.algos.td3 import HEADER as TD3_HEADER
from scripts.algos.td3 import TD3, TD3Config
from scripts.env import TradingEnv, TradingEnvConfig

PairName = Literal["ppo_sac", "sac_sac", "ppo_td3"]


class HRLConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int | None = 0
    verbose: int = 1
    mode: Literal["train", "test", "full"] = "train"
    pair: PairName = "ppo_sac"
    log_dir: str = "outputs"
    ckpt_dir: str = "outputs/ckpts"
    stats_window_size: int = 100

    total_timesteps: int = 50_000
    test_episodes: int = 4
    test_deterministic: bool = True
    test_random_actions: bool = False
    manager_horizon: int | None = None

    train_schedule: Literal["joint", "alternating"] = "joint"
    alt_worker_timesteps: int = Field(default=2_048, ge=1)
    alt_manager_timesteps: int = Field(default=1_024, ge=1)

    manager_net_arch: list[int] = Field(default_factory=lambda: [64, 64])
    worker_net_arch: list[int] = Field(default_factory=lambda: [64, 64])
    learning_rate: float = 3e-4

    worker_buffer_size: int = 50_000
    worker_learning_starts: int = 256
    worker_batch_size: int = 64
    worker_tau: float = 0.005
    worker_gamma: float = 0.99
    worker_train_freq: int = 1
    worker_gradient_steps: int = 1
    worker_ent_coef: Literal["auto"] | float = "auto"
    worker_policy_delay: int = 2
    worker_target_policy_noise: float = 0.2
    worker_target_noise_clip: float = 0.5
    worker_action_noise_std: float = 0.1
    intrinsic_coef: float = 0.5
    extrinsic_coef: float = 1.0

    manager_buffer_size: int = 20_000
    manager_learning_starts: int = 64
    manager_batch_size: int = 32
    manager_tau: float = 0.005
    manager_gamma: float = 0.99
    manager_ent_coef: Literal["auto"] | float = "auto"

    manager_n_steps: int = 64
    manager_batch_size_ppo: int = 32
    manager_n_epochs: int = 4
    manager_gae_lambda: float = 0.95
    manager_clip_range: float = 0.2
    manager_ent_coef_ppo: float = 0.01
    manager_vf_coef: float = 0.5
    manager_max_grad_norm: float = 0.5
    manager_log_std_init: float = -0.5

    env: TradingEnvConfig = Field(default_factory=TradingEnvConfig)

    @model_validator(mode="after")
    def _sync_horizon(self) -> HRLConfig:
        if self.manager_horizon is None:
            object.__setattr__(
                self, "manager_horizon", self.env.manager_horizon
            )
        else:
            self.env.manager_horizon = self.manager_horizon
        if self.pair in ("ppo_sac", "ppo_td3"):
            n = self.manager_n_steps * self.env.n_envs
            if n % self.manager_batch_size_ppo != 0:
                raise ValueError(
                    "manager_n_steps * n_envs must be divisible by "
                    "manager_batch_size_ppo"
                )
        return self

    @classmethod
    def from_yaml(cls, path: str) -> HRLConfig:
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f) or {}
        env_data = data.pop("env", {}) or {}
        data["env"] = TradingEnvConfig(**env_data)
        return cls(**data)


class HRLAgent:
    """Orchestrates manager + worker algos over a shared TradingEnv loop."""

    def __init__(self, config: HRLConfig, env: TradingEnv | None = None):
        self.config = config
        if config.seed is not None:
            mx.random.seed(config.seed)
        self.env = env if env is not None else TradingEnv(config.env)
        self.horizon = int(config.manager_horizon or self.env.manager_horizon)
        self.pair = config.pair
        self.manager_algo = "ppo" if config.pair.startswith("ppo") else "sac"
        self.worker_algo = "td3" if config.pair.endswith("td3") else "sac"

        self.manager = self._build_manager()
        self.worker = self._build_worker()
        self.ep_info = deque(maxlen=config.stats_window_size)

    def _build_manager(self):
        cfg = self.config
        m_env = DimEnv(
            cfg.env.n_envs,
            self.env.manager_obs_dim,
            self.env.goal_dim,
            is_discrete=False,
        )
        if self.manager_algo == "ppo":
            pcfg = PPOConfig(
                seed=cfg.seed,
                verbose=0,
                total_timesteps=cfg.total_timesteps,
                env_id="HRL-Manager",
                log_dir=cfg.log_dir,
                n_envs=cfg.env.n_envs,
                stats_window_size=cfg.stats_window_size,
                learning_rate=cfg.learning_rate,
                n_steps=cfg.manager_n_steps,
                batch_size=cfg.manager_batch_size_ppo,
                n_epochs=cfg.manager_n_epochs,
                gamma=cfg.manager_gamma,
                gae_lambda=cfg.manager_gae_lambda,
                clip_range=cfg.manager_clip_range,
                ent_coef=cfg.manager_ent_coef_ppo,
                vf_coef=cfg.manager_vf_coef,
                max_grad_norm=cfg.manager_max_grad_norm,
                net_arch=cfg.manager_net_arch,
                activation="tanh",
                log_std_init=cfg.manager_log_std_init,
            )
            return PPO(pcfg, m_env)
        scfg = SACConfig(
            seed=cfg.seed,
            verbose=0,
            total_timesteps=cfg.total_timesteps,
            env_id="HRL-Manager",
            log_dir=cfg.log_dir,
            n_envs=cfg.env.n_envs,
            stats_window_size=cfg.stats_window_size,
            learning_rate=cfg.learning_rate,
            buffer_size=cfg.manager_buffer_size,
            learning_starts=cfg.manager_learning_starts,
            batch_size=cfg.manager_batch_size,
            tau=cfg.manager_tau,
            gamma=cfg.manager_gamma,
            ent_coef=cfg.manager_ent_coef,
            net_arch=cfg.manager_net_arch,
        )
        return SAC(scfg, m_env)

    def _build_worker(self):
        cfg = self.config
        w_env = DimEnv(
            cfg.env.n_envs,
            self.env.worker_obs_dim,
            self.env.action_dim,
            is_discrete=False,
        )
        if self.worker_algo == "sac":
            scfg = SACConfig(
                seed=cfg.seed,
                verbose=0,
                total_timesteps=cfg.total_timesteps,
                env_id="HRL-Worker",
                log_dir=cfg.log_dir,
                n_envs=cfg.env.n_envs,
                stats_window_size=cfg.stats_window_size,
                learning_rate=cfg.learning_rate,
                buffer_size=cfg.worker_buffer_size,
                learning_starts=cfg.worker_learning_starts,
                batch_size=cfg.worker_batch_size,
                tau=cfg.worker_tau,
                gamma=cfg.worker_gamma,
                train_freq=cfg.worker_train_freq,
                gradient_steps=cfg.worker_gradient_steps,
                ent_coef=cfg.worker_ent_coef,
                net_arch=cfg.worker_net_arch,
            )
            return SAC(scfg, w_env)
        tcfg = TD3Config(
            seed=cfg.seed,
            verbose=0,
            total_timesteps=cfg.total_timesteps,
            env_id="HRL-Worker",
            log_dir=cfg.log_dir,
            n_envs=cfg.env.n_envs,
            stats_window_size=cfg.stats_window_size,
            learning_rate=cfg.learning_rate,
            buffer_size=cfg.worker_buffer_size,
            learning_starts=cfg.worker_learning_starts,
            batch_size=cfg.worker_batch_size,
            tau=cfg.worker_tau,
            gamma=cfg.worker_gamma,
            train_freq=cfg.worker_train_freq,
            gradient_steps=cfg.worker_gradient_steps,
            policy_delay=cfg.worker_policy_delay,
            target_policy_noise=cfg.worker_target_policy_noise,
            target_noise_clip=cfg.worker_target_noise_clip,
            action_noise_std=cfg.worker_action_noise_std,
            net_arch=cfg.worker_net_arch,
        )
        return TD3(tcfg, w_env)

    def _phase_allows(self, num_ts: int) -> tuple[bool, bool, str]:
        cfg = self.config
        if cfg.train_schedule == "joint":
            return True, True, "joint"
        cycle = cfg.alt_worker_timesteps + cfg.alt_manager_timesteps
        pos = num_ts % cycle
        if pos < cfg.alt_worker_timesteps:
            return True, False, "worker"
        return False, True, "manager"

    def _manager_act(self, m_obs, deterministic=False):
        if self.manager_algo == "ppo":
            goal, logp, value = self.manager.act(m_obs, deterministic=deterministic)
            return goal, logp, value
        goal = self.manager.sample_unscaled(m_obs, deterministic=deterministic)
        return goal, None, None

    def _worker_act(self, w_obs, deterministic=False):
        return self.worker.sample_unscaled(w_obs, deterministic=deterministic)

    @property
    def _n_w_updates(self) -> int:
        return int(self.worker._n_updates)

    @property
    def _n_m_updates(self) -> int:
        return int(self.manager._n_updates)

    def learn(self):
        cfg = self.config
        os.makedirs(cfg.log_dir, exist_ok=True)

        m_header = PPO_HEADER if self.manager_algo == "ppo" else SAC_HEADER
        w_header = SAC_HEADER if self.worker_algo == "sac" else TD3_HEADER
        m_logger = StatsLogger(
            f"{cfg.log_dir}/stats_manager_{self.manager_algo}.csv",
            m_header, verbose=0,
        )
        w_logger = StatsLogger(
            f"{cfg.log_dir}/stats_worker_{self.worker_algo}.csv",
            w_header, verbose=0,
        )

        env = self.env
        w_obs = env.reset()
        m_obs = env.manager_obs()
        goal, m_logp, m_val = self._manager_act(m_obs, deterministic=False)
        if m_logp is None:
            m_logp = mx.zeros((env.num_envs,), dtype=mx.float32)
        if m_val is None:
            m_val = mx.zeros((env.num_envs,), dtype=mx.float32)
        env.set_goal(goal)
        w_obs = env._worker_obs_from_state(env._t)

        m_obs_l, m_act_l, m_logp_l, m_val_l, m_rew_l, m_start_l = (
            [], [], [], [], [], []
        )
        m_last_done = mx.zeros((env.num_envs,), dtype=mx.bool_)
        m_seg_rew = mx.zeros((env.num_envs,), dtype=mx.float32)
        m_start_obs, m_start_goal = m_obs, goal
        m_start_logp, m_start_val = m_logp, m_val
        horizon_i = 0

        num_ts = 0
        n_episodes = 0
        m_iters = 0
        t0 = time.time()
        phase = "joint"
        w_acc = {"actor": [], "critic": [], "ent": []}
        m_acc = {"actor": [], "critic": [], "ent": []}
        pbar = tqdm(
            total=cfg.total_timesteps,
            desc=f"HRL[{cfg.pair}|{cfg.train_schedule}]",
            disable=cfg.verbose < 1,
            unit="step",
            dynamic_ncols=True,
        )

        def _ep_means():
            if not self.ep_info:
                return None, None
            return (
                sum(e[0] for e in self.ep_info) / len(self.ep_info),
                sum(e[1] for e in self.ep_info) / len(self.ep_info),
            )

        def _mean_acc(acc, key):
            v = acc[key]
            if not v:
                return ""
            return mx.mean(mx.stack(v)).item()

        def _log_offpolicy(logger, algo, acc, n_upd, lr, ent_coef=None):
            elapsed = time.time() - t0
            ep_rew, ep_len = _ep_means()
            row = {
                "time/episodes": n_episodes,
                "time/fps": int(num_ts / elapsed) if elapsed else 0,
                "time/time_elapsed": round(elapsed, 2),
                "time/total_timesteps": num_ts,
                "rollout/ep_rew_mean": (
                    round(ep_rew, 4) if ep_rew is not None else ""
                ),
                "rollout/ep_len_mean": (
                    round(ep_len, 2) if ep_len is not None else ""
                ),
                "train/actor_loss": (
                    round(_mean_acc(acc, "actor"), 4)
                    if acc["actor"] else ""
                ),
                "train/critic_loss": (
                    round(_mean_acc(acc, "critic"), 4)
                    if acc["critic"] else ""
                ),
                "train/learning_rate": lr,
                "train/n_updates": n_upd,
            }
            if algo == "sac":
                row["train/ent_coef"] = (
                    round(ent_coef, 4) if ent_coef is not None else ""
                )
                row["train/ent_coef_loss"] = (
                    round(_mean_acc(acc, "ent"), 4) if acc["ent"] else ""
                )
            logger.log(row)
            for k in acc:
                acc[k] = []

        def _log_ppo(logger, stats, n_upd):
            nonlocal m_iters
            m_iters += 1
            elapsed = time.time() - t0
            ep_rew, ep_len = _ep_means()
            logger.log({
                "time/iterations": m_iters,
                "time/total_timesteps": num_ts,
                "time/fps": int(num_ts / elapsed) if elapsed else 0,
                "time/time_elapsed": round(elapsed, 2),
                "rollout/ep_rew_mean": (
                    round(ep_rew, 4) if ep_rew is not None else ""
                ),
                "rollout/ep_len_mean": (
                    round(ep_len, 2) if ep_len is not None else ""
                ),
                "train/approx_kl": round(stats["kl"], 6),
                "train/clip_fraction": round(stats["clip"], 4),
                "train/clip_range": cfg.manager_clip_range,
                "train/entropy_loss": round(stats["ent"], 6),
                "train/explained_variance": "",
                "train/learning_rate": cfg.learning_rate,
                "train/loss": round(stats["loss"], 6),
                "train/n_updates": n_upd,
                "train/policy_gradient_loss": round(stats["pg"], 6),
                "train/value_loss": round(stats["vf"], 6),
            })

        while num_ts < cfg.total_timesteps:
            do_worker, do_manager, phase = self._phase_allows(num_ts)

            if num_ts < cfg.worker_learning_starts:
                action = mx.random.uniform(
                    -1.0, 1.0, (env.num_envs, env.action_dim)
                )
            else:
                action = self._worker_act(w_obs, deterministic=False)

            res = env.step(action)
            ag_after = env.achieved_goal()
            intrinsic = -mx.mean((ag_after - goal) ** 2, axis=-1)
            w_rew = (
                cfg.extrinsic_coef * res.reward
                + cfg.intrinsic_coef * intrinsic
            )
            self.worker.store(
                w_obs, res.terminal_obs, action, w_rew, res.terminated
            )
            m_seg_rew = m_seg_rew + res.reward
            horizon_i += 1
            num_ts += env.num_envs
            pbar.update(env.num_envs)

            fin = mx.stack([
                res.done.astype(mx.float32), res.ep_ret,
                res.ep_len.astype(mx.float32),
            ]).tolist()
            for d_, r_, l_ in zip(*fin):
                if d_:
                    self.ep_info.append((r_, int(l_)))
                    n_episodes += 1

            manager_tick = horizon_i >= self.horizon or bool(
                mx.any(res.done).item()
            )
            if manager_tick:
                next_m = env.manager_obs()
                if self.manager_algo == "sac":
                    if num_ts >= cfg.manager_learning_starts:
                        self.manager.store(
                            m_start_obs, next_m, m_start_goal, m_seg_rew,
                            res.terminated,
                        )
                    if do_manager:
                        out = self.manager.update()
                        if out is not None:
                            al, cl, el = out
                            m_acc["actor"].append(al)
                            m_acc["critic"].append(cl)
                            m_acc["ent"].append(el)
                else:
                    m_obs_l.append(m_start_obs)
                    m_act_l.append(m_start_goal)
                    m_logp_l.append(m_start_logp)
                    m_val_l.append(m_start_val)
                    m_rew_l.append(m_seg_rew)
                    m_start_l.append(m_last_done.astype(mx.float32))
                    if len(m_obs_l) >= cfg.manager_n_steps:
                        if do_manager:
                            obs_b = mx.stack(m_obs_l)
                            act_b = mx.stack(m_act_l)
                            logp_b = mx.stack(m_logp_l)
                            val_b = mx.stack(m_val_l)
                            rew_b = mx.stack(m_rew_l)
                            start_b = mx.stack(m_start_l)
                            last_v = self.manager.value(next_m)
                            adv, ret = self.manager.gae(
                                val_b, rew_b, start_b, last_v, res.done
                            )
                            stats, n_upd = self.manager.update_rollout(
                                (obs_b, act_b, logp_b, val_b, rew_b, start_b),
                                adv, ret,
                            )
                            _log_ppo(m_logger, stats, n_upd)
                        m_obs_l.clear()
                        m_act_l.clear()
                        m_logp_l.clear()
                        m_val_l.clear()
                        m_rew_l.clear()
                        m_start_l.clear()

                m_last_done = res.done
                m_obs = next_m
                goal, m_logp, m_val = self._manager_act(
                    m_obs, deterministic=False
                )
                if m_logp is None:
                    m_logp = mx.zeros((env.num_envs,), dtype=mx.float32)
                if m_val is None:
                    m_val = mx.zeros((env.num_envs,), dtype=mx.float32)
                env.set_goal(goal)
                m_start_obs, m_start_goal = m_obs, goal
                m_start_logp, m_start_val = m_logp, m_val
                m_seg_rew = mx.zeros((env.num_envs,), dtype=mx.float32)
                horizon_i = 0

            w_obs = env._worker_obs_from_state(env._t)

            if (
                do_worker
                and num_ts >= cfg.worker_learning_starts
                and num_ts % cfg.worker_train_freq == 0
            ):
                for _ in range(cfg.worker_gradient_steps):
                    out = self.worker.update()
                    if out is None:
                        continue
                    if self.worker_algo == "sac":
                        al, cl, el = out
                        w_acc["actor"].append(al)
                        w_acc["critic"].append(cl)
                        w_acc["ent"].append(el)
                    else:
                        al, cl = out
                        if al is not None:
                            w_acc["actor"].append(al)
                        w_acc["critic"].append(cl)

            elapsed = time.time() - t0
            fps = int(num_ts / elapsed) if elapsed else 0
            ep_rew, ep_len = _ep_means()
            pbar.set_postfix(
                phase=phase,
                fps=fps,
                ep_rew=f"{ep_rew:.3f}" if ep_rew is not None else "-",
                w=self._n_w_updates,
                m=self._n_m_updates,
                refresh=False,
            )

            # Off-policy loggers: same cadence as SAC/TD3 learn()
            if n_episodes > 0 and n_episodes % 4 == 0 and (
                n_episodes != getattr(self, "_last_logged_ep", 0)
            ):
                self._last_logged_ep = n_episodes
                if self.worker_algo == "sac":
                    ent = float(mx.exp(self.worker.log_ent_coef).item())
                else:
                    ent = None
                if w_acc["critic"] or w_acc["actor"]:
                    _log_offpolicy(
                        w_logger, self.worker_algo, w_acc,
                        self._n_w_updates, cfg.learning_rate, ent,
                    )
                if self.manager_algo == "sac" and (
                    m_acc["critic"] or m_acc["actor"]
                ):
                    ment = float(mx.exp(self.manager.log_ent_coef).item())
                    _log_offpolicy(
                        m_logger, "sac", m_acc,
                        self._n_m_updates, cfg.learning_rate, ment,
                    )

        pbar.close()
        m_logger.close()
        w_logger.close()
        self.save(os.path.join(cfg.ckpt_dir, f"hrl_{cfg.pair}"))
        return self

    def test(self, n_episodes: int | None = None) -> dict:
        cfg = self.config
        n_ep = n_episodes or cfg.test_episodes
        env = self.env
        returns, lengths = [], []
        env.reset()
        episodes = 0
        det = cfg.test_deterministic
        goal, _, _ = self._manager_act(env.manager_obs(), deterministic=det)
        env.set_goal(goal)
        w_obs = env._worker_obs_from_state(env._t)
        steps_in_goal = 0
        while episodes < n_ep:
            if cfg.test_random_actions:
                action = mx.random.uniform(
                    -1.0, 1.0, (env.num_envs, env.action_dim)
                )
            else:
                action = self._worker_act(w_obs, deterministic=det)
            res = env.step(action)
            steps_in_goal += 1
            if steps_in_goal >= self.horizon:
                goal, _, _ = self._manager_act(
                    env.manager_obs(), deterministic=det
                )
                env.set_goal(goal)
                steps_in_goal = 0
            w_obs = env._worker_obs_from_state(env._t)
            fin = mx.stack([
                res.done.astype(mx.float32), res.ep_ret,
                res.ep_len.astype(mx.float32),
            ]).tolist()
            for d_, r_, l_ in zip(*fin):
                if d_:
                    returns.append(r_)
                    lengths.append(int(l_))
                    episodes += 1
                    if episodes >= n_ep:
                        break
        summary = {
            "n_episodes": len(returns),
            "mean_return": float(sum(returns) / max(len(returns), 1)),
            "mean_length": float(sum(lengths) / max(len(lengths), 1)),
            "ledger_rows": (
                env.ledger.n_rows if env.ledger is not None else 0
            ),
        }
        if cfg.verbose:
            print(
                f"test[{cfg.pair}] episodes={summary['n_episodes']} "
                f"mean_return={summary['mean_return']:.4f} "
                f"mean_len={summary['mean_length']:.1f} "
                f"ledger_rows={summary['ledger_rows']}"
            )
        return summary

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        payload = {}
        if self.manager_algo == "ppo":
            payload["m_policy"] = self.manager.policy.parameters()
        else:
            payload["m_actor"] = self.manager.actor.parameters()
            payload["m_critics"] = self.manager.critics.parameters()
        payload["w_actor"] = self.worker.actor.parameters()
        payload["w_critics"] = self.worker.critics.parameters()
        mx.savez(path + ".npz", **_flatten_params(payload))

    def load(self, path: str):
        data = mx.load(path + ".npz")
        if self.manager_algo == "ppo":
            self.manager.policy.update(_unflatten(data, "m_policy"))
        else:
            self.manager.actor.update(_unflatten(data, "m_actor"))
            self.manager.critics.update(_unflatten(data, "m_critics"))
            self.manager.critics_target.update(self.manager.critics.parameters())
        self.worker.actor.update(_unflatten(data, "w_actor"))
        self.worker.critics.update(_unflatten(data, "w_critics"))
        self.worker.critics_target.update(self.worker.critics.parameters())
        if self.worker_algo == "td3":
            self.worker.actor_target.update(self.worker.actor.parameters())
        return self


def _flatten_params(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten_params(v, p))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(_flatten_params(v, f"{prefix}.{i}"))
    elif isinstance(obj, mx.array):
        out[prefix] = obj
    return out


def _unflatten(data, prefix):
    from mlx.utils import tree_unflatten

    items = []
    pre = prefix + "."
    for k, v in data.items():
        if k.startswith(pre):
            items.append((k[len(pre):], v))
    if not items:
        return {}
    return tree_unflatten(items)
