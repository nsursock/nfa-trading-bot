"""Time-to-solve (TTS) sweep over n_envs for every algo/env pair.

    python -m utils.bench.solve                          # 32, 64, 128 envs, seed 0
    python -m utils.bench.solve --seeds 0 1 2 --budget-mult 4  # 4x YAML total_timesteps
    python -m utils.bench.solve --timesteps 20000000 --n-envs 128 256 512 1024 2048 --seeds 0 1 2 3 4
    python -m utils.bench.solve --algos ppo --n-envs 8 --threshold 475

An env counts as solved the first time the mean return over the last
`--window` (100) completed episodes reaches the threshold, using gymnasium's
`reward_threshold` where one exists:

    CartPole-v1   475     (gymnasium registration)
    Pendulum-v1  -200     (gymnasium has none; common convention)

Training stops as soon as the env is solved, so `tts_s` is the wall time of
`learn()` up to that point and `tts_steps` the timesteps consumed. `steps/s`
is `tts_steps / tts_s` on solved runs: transitions consumed per second up to
the solution, not a steady-state throughput window. Runs that do not solve
within the budget (`total_timesteps`, `--budget-mult`, or `--timesteps`) are
reported with `solved=no` and the best mean return seen. The check runs at the
algorithms' own logging cadence (every PPO iteration; every `log_interval`
episodes for SAC/TD3).

Per-seed rows go to the `--out` CSV. `*.summary.csv` aggregates each
`(algo, env, n_envs)`: how many seeds solved, and among the solved seeds the
median, mean, standard deviation, and P25/P75 of wall time and transitions,
plus the median `steps/s`. Unsolved seeds stay in the success count and out of
those averages. Odd seed passes walk the sweep in reverse so the last width
is not always the hot one.
"""

import argparse
import statistics
import tempfile
import time
from contextlib import contextmanager

import mlx.core as mx

import scripts.algos.ppo
import scripts.algos.sac
import scripts.algos.td3
from scripts.algos.common import StatsLogger
from utils.bench.scale import (
    ALGOS, add_sweep_args, build_config, format_table, geometric_range,
    make_agent, write_csv,
)

THRESHOLDS = {"CartPole-v1": 475.0, "Pendulum-v1": -200.0}
COLUMNS = [
    "algo", "env", "n_envs", "seed", "threshold", "solved", "tts_s", "tts_steps",
    "steps/s", "best_rew", "budget_steps",
]
SUMMARY_COLUMNS = [
    "algo", "env", "n_envs", "solved",
    "tts_s_median", "tts_s_p25", "tts_s_p75", "tts_s_mean", "tts_s_std",
    "tts_steps_median", "tts_steps_p25", "tts_steps_p75", "tts_steps_mean", "tts_steps_std",
    "steps/s",
]
_MODULES = {"ppo": scripts.algos.ppo, "sac": scripts.algos.sac, "td3": scripts.algos.td3}


class Solved(Exception):
    pass


class SolveLogger(StatsLogger):
    """Drop-in StatsLogger that stops training once the env is solved."""

    def __init__(self, path, header, verbose, agent, threshold, window):
        super().__init__(path, header, verbose)
        self.agent, self.threshold, self.window = agent, threshold, window
        self.best = float("-inf")
        self.solved_at = None

    def log(self, row):
        super().log(row)
        rew = row.get("rollout/ep_rew_mean", "")
        if rew == "":
            return
        self.best = max(self.best, rew)
        if len(self.agent.ep_info_buffer) >= self.window and rew >= self.threshold:
            self.solved_at = int(row["time/total_timesteps"])
            self.close()
            raise Solved


@contextmanager
def patched_logger(algo, factory):
    """`learn()` looks up StatsLogger in its own module namespace; swap it there."""
    mod = _MODULES[algo]
    original = mod.StatsLogger
    mod.StatsLogger = factory
    try:
        yield
    finally:
        mod.StatsLogger = original


def run_one(algo, n_envs, timesteps, budget_mult, seed, log_dir, fixed_hparams,
            max_batch_size, replay_scaling, threshold, window):
    cfg = build_config(algo, n_envs, timesteps, seed, log_dir, fixed_hparams,
                       max_batch_size, replay_scaling, stats_window_size=window)
    if timesteps is None and budget_mult != 1:
        cfg = cfg.model_copy(
            update={"total_timesteps": int(cfg.total_timesteps * budget_mult)}
        )
    thr = THRESHOLDS[cfg.env_id] if threshold is None else threshold
    mx.random.seed(seed)
    agent = make_agent(algo, cfg)
    holder = {}

    def factory(path, header, verbose=1):
        holder["logger"] = SolveLogger(path, header, verbose, agent, thr, window)
        return holder["logger"]

    t0 = time.perf_counter()
    with patched_logger(algo, factory):
        try:
            agent.learn()
            solved = False
        except Solved:
            solved = True
    wall = time.perf_counter() - t0
    logger = holder["logger"]
    return {
        "algo": algo,
        "env": cfg.env_id,
        "n_envs": n_envs,
        "seed": seed,
        "threshold": thr,
        "solved": "yes" if solved else "no",
        "tts_s": round(wall, 2) if solved else "",
        "tts_steps": logger.solved_at if solved else "",
        "steps/s": round(logger.solved_at / wall) if solved and wall > 0 else "",
        "best_rew": round(logger.best, 2) if logger.best > float("-inf") else "",
        "budget_steps": cfg.total_timesteps,
    }


def _spread(values, digits):
    blank = {key: "" for key in ("median", "mean", "std", "p25", "p75")}
    if not values:
        return blank

    def rnd(value):
        return round(value, digits) if digits else round(value)

    median = rnd(statistics.median(values))
    out = {
        "median": median,
        "mean": rnd(statistics.mean(values)),
        "std": rnd(statistics.stdev(values)) if len(values) >= 2 else "",
        "p25": median,
        "p75": median,
    }
    if len(values) >= 2:
        quartiles = statistics.quantiles(values, n=4, method="inclusive")
        out["p25"] = rnd(quartiles[0])
        out["p75"] = rnd(quartiles[2])
    return out


def summarize_runs(rows):
    """One row per (algo, env, n_envs). Time stats use solved seeds only."""
    groups, order = {}, []
    for row in rows:
        key = (row["algo"], row["env"], row["n_envs"])
        if key not in groups:
            order.append(key)
            groups[key] = []
        groups[key].append(row)
    summary = []
    for key in order:
        group = groups[key]
        solved = [
            row for row in group
            if row["solved"] == "yes" and row["tts_s"] != "" and row["tts_steps"] != ""
        ]
        times = _spread([row["tts_s"] for row in solved], 2)
        steps = _spread([row["tts_steps"] for row in solved], 0)
        rates = [row["tts_steps"] / row["tts_s"] for row in solved if row["tts_s"]]
        algo, env, n_envs = key
        summary.append({
            "algo": algo,
            "env": env,
            "n_envs": n_envs,
            "solved": f"{len(solved)}/{len(group)}",
            "tts_s_median": times["median"],
            "tts_s_p25": times["p25"],
            "tts_s_p75": times["p75"],
            "tts_s_mean": times["mean"],
            "tts_s_std": times["std"],
            "tts_steps_median": steps["median"],
            "tts_steps_p25": steps["p25"],
            "tts_steps_p75": steps["p75"],
            "tts_steps_mean": steps["mean"],
            "tts_steps_std": steps["std"],
            "steps/s": round(statistics.median(rates)) if rates else "",
        })
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_sweep_args(p)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--budget-mult", type=float, default=1.0,
                   help="multiply each YAML total_timesteps (ignored with --timesteps)")
    p.add_argument("--threshold", type=float, help="override the per-env solve threshold")
    p.add_argument("--window", type=int, default=100, help="episodes averaged for the check")
    p.add_argument("--out", default="outputs/bench_solve.csv")
    args = p.parse_args()

    sweep = args.n_envs or geometric_range(args.start, args.end, args.factor)
    combos = [(algo, n) for algo in args.algos for n in sweep]
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for index, seed in enumerate(args.seeds):
            order = combos if index % 2 == 0 else list(reversed(combos))
            for algo, n in order:
                print(f"[{algo}] n_envs={n} seed={seed} ...", flush=True)
                rows.append(run_one(algo, n, args.timesteps, args.budget_mult,
                                    seed, tmp, args.fixed_hparams,
                                    args.max_batch_size, args.replay_scaling,
                                    args.threshold, args.window))
                r = rows[-1]
                rate = f", {r['steps/s']} steps/s" if r["steps/s"] != "" else ""
                status = (f"solved in {r['tts_s']}s / {r['tts_steps']} steps{rate}"
                          if r["solved"] == "yes" else
                          f"not solved (best {r['best_rew']})")
                print(f"    {status}", flush=True)

    print()
    print(format_table(rows, COLUMNS))
    summary = summarize_runs(rows)
    print()
    print(format_table(summary, SUMMARY_COLUMNS))
    write_csv(args.out, rows, COLUMNS)
    summary_path = args.out.replace(".csv", ".summary.csv") if args.out.endswith(".csv") \
        else args.out + ".summary.csv"
    write_csv(summary_path, summary, SUMMARY_COLUMNS)


if __name__ == "__main__":
    main()
