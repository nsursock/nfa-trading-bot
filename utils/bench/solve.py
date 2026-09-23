"""Time-to-solve (TTS) sweep over n_envs for every algo/env pair.

    python -m utils.bench.solve                          # 32, 64, 128 envs, seed 0
    python -m utils.bench.solve --seeds 0 1 2 --budget-mult 4  # 4x YAML total_timesteps
    python -m utils.bench.solve --algos ppo --n-envs 8 --threshold 475

An env counts as solved the first time the mean return over the last
`--window` (100) completed episodes reaches the threshold, using gymnasium's
`reward_threshold` where one exists:

    CartPole-v1   475     (gymnasium registration)
    Pendulum-v1  -200     (gymnasium has none; common convention)

Training stops as soon as the env is solved, so `tts_s` is the wall time of
`learn()` up to that point and `tts_steps` the timesteps consumed. Runs that do
not solve within the budget (`total_timesteps`, `--budget-mult`, or
`--timesteps`) are reported with `solved=no` and the best
mean return seen. The check runs at the algorithms' own logging cadence (every
PPO iteration; every `log_interval` episodes for SAC/TD3).
"""

import argparse
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
    "best_rew", "budget_steps",
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
        "best_rew": round(logger.best, 2) if logger.best > float("-inf") else "",
        "budget_steps": cfg.total_timesteps,
    }


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
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for algo in args.algos:
            for n in sweep:
                for seed in args.seeds:
                    print(f"[{algo}] n_envs={n} seed={seed} ...", flush=True)
                    rows.append(run_one(algo, n, args.timesteps, args.budget_mult,
                                        seed, tmp, args.fixed_hparams,
                                        args.max_batch_size, args.replay_scaling,
                                        args.threshold, args.window))
                    r = rows[-1]
                    status = (f"solved in {r['tts_s']}s / {r['tts_steps']} steps"
                              if r["solved"] == "yes" else
                              f"not solved (best {r['best_rew']})")
                    print(f"    {status}", flush=True)

    print()
    print(format_table(rows, COLUMNS))
    write_csv(args.out, rows, COLUMNS)


if __name__ == "__main__":
    main()
