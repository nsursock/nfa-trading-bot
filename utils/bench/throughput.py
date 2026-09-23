"""Wall-clock time to a fixed transition count at fixed learning intensity.

    python -m utils.bench.throughput
    python -m utils.bench.throughput --algos td3 --repeats 3
    python -m utils.bench.throughput --algos ppo --n-envs 1024 2048 4096

`scale.py` grows replayed samples slower than `n_envs` (sqrt by default), so
`samples/step` falls as the sweep gets wider. This benchmark does not. Every
width uses the YAML batch size, replay buffer, and learning_starts. For
SAC/TD3, `gradient_steps` grows with `n_envs` so the YAML update-to-data ratio
stays put (256 replayed samples per environment transition). PPO keeps YAML
`batch_size`, `n_steps`, and `n_epochs`, which already replays each transition
`n_epochs` times.

Every width is timed over the same number of transitions, in a fresh process.
Two training cycles are discarded (compile, and the step that crosses
`learning_starts`). The clock then runs until exactly `--timesteps` more
transitions have been collected and learned from. `--timesteps` defaults to
eight vector steps at the widest env, rounded up to a whole PPO rollout when
PPO is included.
"""

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

import mlx.core as mx

from utils.bench.scale import (
    ALGOS, MeasurementWindow, build_config, format_table, make_agent, write_csv,
)

COLUMNS = [
    "algo", "env", "n_envs", "batch_size", "grad_steps", "buffer_size",
    "timesteps", "wall_s", "transitions/s", "trans/prev", "trans_min", "trans_max",
    "updates", "samples/step", "repeats",
]
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_N_ENVS = [1024, 2048, 4096]


def fixed_intensity_config(algo, n_envs, seed, log_dir):
    """YAML batch and buffer; SAC/TD3 replay ratio held at the YAML ratio."""
    cfg_cls, _, yaml_path = ALGOS[algo]
    base = cfg_cls.from_yaml(yaml_path)
    if algo == "ppo":
        return build_config(algo, n_envs, None, seed, log_dir, fixed_hparams=True)
    return build_config(
        algo, n_envs, None, seed, log_dir,
        fixed_hparams=False, max_batch_size=base.batch_size, replay_scaling="linear",
    )


def replay_per_step(algo, cfg):
    if algo == "ppo":
        return float(cfg.n_epochs)
    return cfg.gradient_steps * cfg.batch_size / cfg.n_envs


def resolve_budget(algos, sweep, timesteps):
    """One transition count shared by every width in the sweep."""
    if any(n <= 0 for n in sweep):
        raise ValueError("n_envs must be positive")
    if timesteps is None:
        timesteps = 8 * math.lcm(*sweep)
        if "ppo" in algos:
            n_steps = ALGOS["ppo"][0].from_yaml(ALGOS["ppo"][2]).n_steps
            need = math.lcm(*[n_steps * n for n in sweep])
            timesteps = math.ceil(timesteps / need) * need
    if timesteps <= 0:
        raise ValueError("timesteps must be positive")
    for n in sweep:
        if timesteps % n != 0:
            raise ValueError(f"{timesteps} transitions is not a multiple of n_envs={n}")
    if "ppo" in algos:
        n_steps = ALGOS["ppo"][0].from_yaml(ALGOS["ppo"][2]).n_steps
        rollouts = [n_steps * n for n in sweep]
        for rollout, n in zip(rollouts, sweep):
            if timesteps % rollout != 0:
                raise ValueError(
                    f"{timesteps} transitions is not a multiple of the PPO rollout "
                    f"n_steps*n_envs={rollout} (n_envs={n}). "
                    f"Smallest shared budget: {math.lcm(*rollouts)}"
                )
    return timesteps


def measure(algo, n_envs, timesteps, seed):
    with tempfile.TemporaryDirectory() as tmp:
        cfg = fixed_intensity_config(algo, n_envs, seed, tmp)
        if n_envs > getattr(cfg, "buffer_size", math.inf):
            raise ValueError("n_envs must not exceed replay buffer capacity")
        cfg = type(cfg)(**{**cfg.model_dump(), "total_timesteps": 2**60})
        agent = make_agent(algo, cfg)
        min_updates = max(2, getattr(cfg, "policy_delay", 1),
                          getattr(cfg, "target_update_interval", 1))
        mx.synchronize()
        window = MeasurementWindow(
            warmup=0.0, seconds=0.0, min_steps=timesteps,
            min_warmup_updates=min_updates,
        )
        agent.learn(callback=window)
        if window.result is None:
            raise RuntimeError("training ended before the transition budget was met")
        result = window.result
        if result["timesteps"] != timesteps:
            raise RuntimeError(
                f"measured {result['timesteps']} transitions, budget is {timesteps}"
            )
        result["samples"] = result["updates"] * cfg.batch_size
        return cfg, result


def run_trial(spec):
    cfg, result = measure(spec["algo"], spec["n_envs"], spec["timesteps"], spec["seed"])
    return {
        "algo": spec["algo"], "env": cfg.env_id, "n_envs": cfg.n_envs,
        "repeat": spec["repeat"], "pid": os.getpid(), "seed": spec["seed"],
        "config": cfg.model_dump(),
        "replay_per_step": replay_per_step(spec["algo"], cfg),
        "training": result,
    }


def run_isolated(spec):
    result = subprocess.run(
        [sys.executable, "-m", "utils.bench.throughput", "--worker-spec", json.dumps(spec)],
        cwd=ROOT, capture_output=True, text=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr or result.stdout)
    return json.loads(result.stdout)


def summarize(trials):
    first = trials[0]
    cfg = first["config"]
    training = [t["training"] for t in trials]
    walls = [t["wall_s"] for t in training]
    steps = training[0]["timesteps"]
    rates = [steps / t["wall_s"] for t in training]
    wall = statistics.median(walls)
    samples = [t["samples"] / t["timesteps"] for t in training]
    return {
        "algo": first["algo"], "env": first["env"], "n_envs": first["n_envs"],
        "batch_size": cfg["batch_size"],
        "grad_steps": cfg.get("gradient_steps", ""),
        "buffer_size": cfg.get("buffer_size", ""),
        "timesteps": steps,
        "wall_s": round(wall, 3),
        "transitions/s": round(steps / wall),
        "trans_min": round(min(rates)),
        "trans_max": round(max(rates)),
        "updates": round(statistics.median(t["updates"] for t in training)),
        "samples/step": round(statistics.median(samples), 2),
        "repeats": len(trials),
    }


def add_ratios(row, previous):
    key = (row["algo"], row["env"])
    rate = row["transitions/s"]
    prev = previous.get(key, 0)
    row["trans/prev"] = round(rate / prev, 2) if prev else ""
    previous[key] = rate


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--n-envs", type=int, nargs="+", default=DEFAULT_N_ENVS)
    p.add_argument("--algos", nargs="+", choices=list(ALGOS), default=["sac", "td3"])
    p.add_argument("--timesteps", type=int,
                   help="environment transitions measured at every width "
                        "(default: 8 vector steps of the widest env)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--out", default="outputs/bench_throughput.csv")
    p.add_argument("--worker-spec", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.worker_spec:
        print(json.dumps(run_trial(json.loads(args.worker_spec))))
        return
    sweep = args.n_envs
    if len(set(sweep)) != len(sweep):
        p.error("--n-envs must contain distinct positive values")
    if len(set(args.algos)) != len(args.algos):
        p.error("--algos must not contain duplicates")
    if args.repeats <= 0:
        p.error("--repeats must be positive")
    try:
        budget = resolve_budget(args.algos, sweep, args.timesteps)
    except ValueError as exc:
        p.error(str(exc))

    combinations = [(algo, n) for algo in args.algos for n in sweep]
    collected = {key: [] for key in combinations}
    raw_path = Path(args.out).with_suffix(".trials.jsonl")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Fixed replay ratio, YAML batch and buffer. "
          f"{budget} transitions after 2 warmup cycles; "
          f"{args.repeats} isolated repeat(s).", flush=True)
    with raw_path.open("w") as raw:
        for repeat in range(args.repeats):
            order = combinations if repeat % 2 == 0 else list(reversed(combinations))
            for algo, n in order:
                print(f"[{algo}] n_envs={n} repeat={repeat + 1}/{args.repeats} ...",
                      flush=True)
                spec = {
                    "algo": algo, "n_envs": n, "repeat": repeat + 1,
                    "seed": args.seed, "timesteps": budget,
                }
                trial = run_isolated(spec)
                raw.write(json.dumps(trial) + "\n")
                raw.flush()
                collected[(algo, n)].append(trial)
                train = trial["training"]
                print(f"    {train['timesteps'] / train['wall_s']:.0f} transitions/s "
                      f"({train['wall_s']:.3f}s, {train['updates']} updates, "
                      f"{train['samples'] / train['timesteps']:.2f} samples/step)",
                      flush=True)

    rows, previous = [], {}
    for key in combinations:
        row = summarize(collected[key])
        add_ratios(row, previous)
        rows.append(row)
    print()
    print(format_table(rows, COLUMNS))
    write_csv(args.out, rows, COLUMNS)
    print(f"raw repeats: {raw_path}")


if __name__ == "__main__":
    main()
