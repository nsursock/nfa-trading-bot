"""Steady-state env/train throughput across vectorized environment counts.

    python -m utils.bench.scale --start 32 --end 4096
    python -m utils.bench.scale --warmup-seconds 1 --seconds 3 --repeats 3
    python -m utils.bench.scale --fixed-hparams --n-envs 32 64 128

Every repeat runs in a fresh process with the same seed. Warmup is excluded;
measurement ends at complete training-cycle boundaries after at least the
requested duration and optional minimum --timesteps. Rates are medians of
repeats, with min/max train FPS; counts and measured seconds are summed.
Raw per-repeat results and resolved configs are saved beside the summary.

Env FPS includes random actions, physics, reset and episode arrays, with one
eval per vector step. It is a separate microbenchmark, not an upper bound on
training (PPO evaluates whole rollouts). Train FPS includes rollout, updates,
and logging. Updates/s and samples/s use this SAME end-to-end window, not
isolated GPU kernel time. An update is one PPO minibatch or one SAC/TD3 critic
update; samples count minibatch rows once per update, not each network pass.

Default workload scaling is unchanged: PPO scales batch size with n_envs;
SAC/TD3 scale replayed samples by sqrt(n_envs), rounded to supported batches
and integer update counts. --fixed-hparams preserves YAML hyperparameters.
Neither mode promises linear train FPS or measures GPU utilization.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

import mlx.core as mx

from scripts.algos.common import make_env
from scripts.algos.ppo import PPO, PPOConfig
from scripts.algos.sac import SAC, SACConfig
from scripts.algos.td3 import TD3, TD3Config

ALGOS = {
    "ppo": (PPOConfig, PPO, "configs/ppo_cartpole.yaml"),
    "sac": (SACConfig, SAC, "configs/sac_pendulum.yaml"),
    "td3": (TD3Config, TD3, "configs/td3_pendulum.yaml"),
}
COLUMNS = [
    "algo", "env", "n_envs", "batch_size", "grad_steps", "env_fps", "train_fps",
    "env/train", "train/prev", "updates/s", "samples/s", "samples/step",
    "train_min", "train_max", "repeats", "train_wall_s", "timesteps", "updates",
]
ROOT = Path(__file__).resolve().parents[2]


def geometric_range(start, end, factor):
    if start <= 0 or end < start or factor <= 1:
        raise ValueError("require 0 < start <= end and factor > 1")
    out, n = [], start
    while n <= end:
        out.append(n)
        n *= factor
    return out


class MeasurementWindow:
    def __init__(self, warmup, seconds, min_steps=0, min_warmup_updates=2,
                 clock=time.perf_counter, synchronize=mx.synchronize):
        self.warmup, self.seconds, self.min_steps = warmup, seconds, min_steps
        self.min_warmup_updates = min_warmup_updates
        self.clock, self.synchronize = clock, synchronize
        self.start = clock()
        self.baseline = None
        self.result = None
        self.warmup_cycles = 0
        self.measured_cycles = 0

    def __call__(self, steps, updates):
        now = self.clock()
        if self.baseline is None:
            self.warmup_cycles += 1
            if (now - self.start < self.warmup or updates < self.min_warmup_updates
                    or self.warmup_cycles < 2):
                return False
            self.synchronize()
            self.baseline = (self.clock(), steps, updates)
            return False
        start, initial_steps, initial_updates = self.baseline
        self.measured_cycles += 1
        if now - start < self.seconds or steps - initial_steps < self.min_steps:
            return False
        self.synchronize()
        self.result = {
            "wall_s": self.clock() - start,
            "timesteps": steps - initial_steps,
            "updates": updates - initial_updates,
            "cycles": self.measured_cycles,
            "warmup_s": start - self.start,
            "warmup_steps": initial_steps,
            "warmup_updates": initial_updates,
        }
        return True


def measure_env(env_id, n_envs, warmup, seconds, seed):
    mx.random.seed(seed)
    env = make_env(env_id, n_envs=n_envs, seed=seed)
    if env.is_discrete:
        act = lambda: mx.random.randint(0, env.n_actions, (n_envs,))
    else:
        act = lambda: mx.random.uniform(
            env.action_low, env.action_high, (n_envs, env.action_dim)
        )
    mx.eval(env.reset())
    window = MeasurementWindow(warmup, seconds, min_warmup_updates=1)
    steps = 0
    while True:
        result = env.step(act())
        mx.eval(*result, env.state, env.steps, env._ep_ret, env._ep_len, env._key)
        steps += n_envs
        if window(steps, steps // n_envs):
            return window.result


REPLAY_SCALINGS = {
    "linear": lambda f: f,          # replay ratio constant (compute-bound)
    "sqrt": lambda f: math.sqrt(f), # replay ratio ~ 1/sqrt(n_envs)
    "none": lambda f: 1.0,          # 1 YAML batch per vectorized step (SB3 default)
}


def build_config(algo, n_envs, timesteps, seed, log_dir, fixed_hparams=False,
                 max_batch_size=2048, replay_scaling="sqrt", **extra):
    cfg_cls, _, default_yaml = ALGOS[algo]
    cfg = cfg_cls.from_yaml(default_yaml)
    update = {"n_envs": n_envs, "verbose": 0, "log_dir": log_dir, "seed": seed, **extra}
    if timesteps is not None:
        update["total_timesteps"] = timesteps
    if not fixed_hparams:
        if algo == "ppo":
            # keep the YAML's minibatches-per-epoch constant so the number of
            # gradient steps per timestep shrinks with n_envs (SB3/CleanRL practice)
            n_minibatches = cfg.n_steps * cfg.n_envs // cfg.batch_size
            update["batch_size"] = cfg.n_steps * n_envs // n_minibatches
        else:
            # one vectorized step = n_envs timesteps. Replayed samples per
            # vectorized step = YAML samples x scaling(n_envs / yaml n_envs);
            # grow batch_size first (cheaper per sample), remainder in gradient_steps.
            factor = REPLAY_SCALINGS[replay_scaling](max(1, n_envs / cfg.n_envs))
            samples = cfg.batch_size * cfg.gradient_steps * factor
            batch = int(min(max(cfg.batch_size, max_batch_size), samples) // 64 * 64)
            update["batch_size"] = batch
            update["gradient_steps"] = max(1, round(samples / batch))
    cfg = cfg.model_copy(update=update)
    return cfg_cls(**cfg.model_dump())  # re-validate (PPO divisibility check)


def make_agent(algo, cfg):
    env = make_env(cfg.env_id, n_envs=cfg.n_envs, seed=cfg.seed or 0)
    return ALGOS[algo][1](cfg, env)


def measure_train(algo, cfg, warmup, seconds, min_steps):
    agent = make_agent(algo, cfg)
    min_updates = max(2, getattr(cfg, "policy_delay", 1),
                      getattr(cfg, "target_update_interval", 1))
    mx.synchronize()
    window = MeasurementWindow(warmup, seconds, min_steps, min_updates)
    agent.learn(callback=window)
    if window.result is None:
        raise RuntimeError("training ended before the measurement window completed")
    result = window.result
    result["samples"] = result["updates"] * cfg.batch_size
    return result


def run_trial(spec):
    with tempfile.TemporaryDirectory() as tmp:
        cfg = build_config(
            spec["algo"], spec["n_envs"], 2**60, spec["seed"], tmp,
            spec["fixed_hparams"], spec["max_batch_size"], spec["replay_scaling"],
        )
        if spec["n_envs"] > getattr(cfg, "buffer_size", math.inf):
            raise ValueError("n_envs must not exceed replay buffer capacity")
        env_result = measure_env(cfg.env_id, cfg.n_envs, spec["warmup"],
                                 spec["seconds"], spec["seed"])
        train_result = measure_train(spec["algo"], cfg, spec["warmup"],
                                     spec["seconds"], spec["min_steps"])
        return {
            "algo": spec["algo"], "env": cfg.env_id, "n_envs": cfg.n_envs,
            "repeat": spec["repeat"], "pid": os.getpid(), "seed": spec["seed"],
            "config": cfg.model_dump(), "settings": spec,
            "environment": env_result, "training": train_result,
        }


def run_isolated(spec):
    result = subprocess.run(
        [sys.executable, "-m", "utils.bench.scale", "--worker-spec", json.dumps(spec)],
        cwd=ROOT, capture_output=True, text=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr or result.stdout)
    return json.loads(result.stdout)


def summarize(trials):
    first = trials[0]
    cfg = first["config"]
    training = [t["training"] for t in trials]
    env_rates = [t["environment"]["timesteps"] / t["environment"]["wall_s"]
                 for t in trials]
    rates = [t["timesteps"] / t["wall_s"] for t in training]
    steps = sum(t["timesteps"] for t in training)
    updates = sum(t["updates"] for t in training)
    return {
        "algo": first["algo"], "env": first["env"], "n_envs": first["n_envs"],
        "batch_size": cfg["batch_size"], "grad_steps": cfg.get("gradient_steps", ""),
        "env_fps": round(statistics.median(env_rates)),
        "train_fps": round(statistics.median(rates)),
        "updates/s": round(statistics.median(t["updates"] / t["wall_s"]
                                             for t in training), 2),
        "samples/s": round(statistics.median(t["samples"] / t["wall_s"]
                                             for t in training)),
        "samples/step": round(sum(t["samples"] for t in training) / steps, 2),
        "train_min": round(min(rates)), "train_max": round(max(rates)),
        "repeats": len(trials), "train_wall_s": round(sum(t["wall_s"] for t in training), 3),
        "timesteps": steps, "updates": updates,
    }


def add_ratios(row, previous):
    key = (row["algo"], row["env"])
    fps = row["train_fps"]
    prev_fps = previous.get(key, 0)
    row["env/train"] = round(row["env_fps"] / fps, 2) if fps > 0 else ""
    row["train/prev"] = round(fps / prev_fps, 2) if prev_fps > 0 else ""
    previous[key] = fps


def format_table(rows, columns=COLUMNS):
    def cell(row, column):
        value = row[column]
        if column in ("env/train", "train/prev"):
            return f"{value:.2f}x" if value != "" else "—"
        return str(value)

    cells = [[cell(r, c) for c in columns] for r in rows]
    widths = [max([len(c)] + [row[i] for row in [[len(x) for x in r] for r in cells]])
              for i, c in enumerate(columns)]
    line = lambda vals: "| " + " | ".join(v.ljust(w) for v, w in zip(vals, widths)) + " |"
    out = [line(columns), "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    out += [line(row) for row in cells]
    return "\n".join(out)


def write_csv(path, rows, columns=COLUMNS):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {path}")


def add_sweep_args(p):
    p.add_argument("--start", type=int, default=32)
    p.add_argument("--end", type=int, default=128)
    p.add_argument("--factor", type=int, default=2, help="multiplier between sweep points")
    p.add_argument("--n-envs", type=int, nargs="+", help="explicit list, overrides start/end")
    p.add_argument("--algos", nargs="+", choices=list(ALGOS), default=list(ALGOS))
    p.add_argument("--timesteps", type=int, help="override total_timesteps for every run")
    p.add_argument("--fixed-hparams", action="store_true",
                   help="keep YAML batch_size (PPO) / gradient_steps (SAC, TD3) "
                        "instead of scaling them with n_envs")
    p.add_argument("--max-batch-size", type=int, default=2048,
                   help="SAC/TD3: cap for the scaled batch_size; the rest of the "
                        "n_envs factor goes into gradient_steps")
    p.add_argument("--replay-scaling", choices=list(REPLAY_SCALINGS), default="sqrt",
                   help="SAC/TD3: how replayed samples per vectorized step grow "
                        "with n_envs (linear = constant replay ratio)")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_sweep_args(p)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="outputs/bench_scale.csv")
    p.add_argument("--warmup-seconds", type=float, default=1.0)
    p.add_argument("--seconds", type=float, default=3.0,
                   help="minimum measured seconds per env/train repeat")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--worker-spec", help=argparse.SUPPRESS)
    p.set_defaults(timesteps=0)
    for action in p._actions:
        if action.dest == "timesteps":
            action.help = "minimum measured training transitions per repeat (not a budget)"
    args = p.parse_args()
    if args.worker_spec:
        print(json.dumps(run_trial(json.loads(args.worker_spec))))
        return
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        p.error("--seconds must be finite and positive")
    if not math.isfinite(args.warmup_seconds) or args.warmup_seconds < 0:
        p.error("--warmup-seconds must be finite and nonnegative")
    if args.repeats <= 0 or args.timesteps < 0 or args.max_batch_size <= 0:
        p.error("repeats/max-batch-size must be positive; timesteps must be nonnegative")
    try:
        sweep = args.n_envs or geometric_range(args.start, args.end, args.factor)
    except ValueError as exc:
        p.error(str(exc))
    if any(n <= 0 for n in sweep) or len(set(sweep)) != len(sweep):
        p.error("--n-envs must contain distinct positive values")
    if len(set(args.algos)) != len(args.algos):
        p.error("--algos must not contain duplicates")

    combinations = [(algo, n) for algo in args.algos for n in sweep]
    collected = {key: [] for key in combinations}
    raw_path = Path(args.out).with_suffix(".trials.jsonl")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Warmup {args.warmup_seconds}s; measurement >= {args.seconds}s; "
          f"{args.repeats} isolated repeats. Learning hyperparameters unchanged.", flush=True)
    with raw_path.open("w") as raw:
        for repeat in range(args.repeats):
            order = combinations if repeat % 2 == 0 else list(reversed(combinations))
            for algo, n in order:
                print(f"[{algo}] n_envs={n} repeat={repeat + 1}/{args.repeats} ...", flush=True)
                spec = {
                    "algo": algo, "n_envs": n, "repeat": repeat + 1, "seed": args.seed,
                    "warmup": args.warmup_seconds, "seconds": args.seconds,
                    "min_steps": args.timesteps, "fixed_hparams": args.fixed_hparams,
                    "max_batch_size": args.max_batch_size, "replay_scaling": args.replay_scaling,
                }
                trial = run_isolated(spec)
                raw.write(json.dumps(trial) + "\n")
                raw.flush()
                collected[(algo, n)].append(trial)
                train = trial["training"]
                print(f"    train {train['timesteps'] / train['wall_s']:.0f} fps "
                      f"({train['wall_s']:.3f}s measured; {train['updates']} updates)", flush=True)

    rows, previous = [], {}
    for key in combinations:
        row = summarize(collected[key])
        add_ratios(row, previous)
        rows.append(row)
    print()
    print(format_table(rows))
    write_csv(args.out, rows)
    print(f"raw repeats: {raw_path}")


if __name__ == "__main__":
    main()
