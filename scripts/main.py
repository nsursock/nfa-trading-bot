"""HRL trading bot entrypoint.

Modes:
  train  — learn manager + worker (stats CSV in a timestamped run folder)
  test   — evaluate; writes trade ledger.csv (test only)
  full   — train then test in the same run folder

Examples:
  python -m scripts.main --config configs/smoke.yaml
  python -m scripts.main --config configs/smoke.yaml --mode full --pair ppo_sac
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

import yaml

from scripts.agent import HRLAgent, HRLConfig
from scripts.env import TradingEnv


def make_run_dir(base: str, pair: str, schedule: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = os.path.join(base, f"{ts}_{pair}_{schedule}")
    os.makedirs(path, exist_ok=True)
    return path


def build_agent(cfg: HRLConfig) -> HRLAgent:
    env = TradingEnv(cfg.env)
    return HRLAgent(cfg, env)


def run(cfg: HRLConfig) -> dict | None:
    run_dir = make_run_dir(cfg.log_dir, cfg.pair, cfg.train_schedule)
    cfg = cfg.model_copy(update={"log_dir": run_dir, "ckpt_dir": run_dir})
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg.model_dump(), f, default_flow_style=False, sort_keys=False)

    agent = build_agent(cfg)
    mode = cfg.mode
    result = None
    if mode in ("train", "full"):
        # Training never attaches a ledger (keeps the step path fast).
        agent.env.detach_ledger()
        agent.learn()
    if mode in ("test", "full"):
        ledger_path = os.path.join(run_dir, "ledger.csv")
        agent.env.attach_ledger(ledger_path)
        try:
            result = agent.test()
        finally:
            agent.env.detach_ledger()
        if cfg.verbose:
            print(f"run_dir={run_dir}")
            print(f"ledger={ledger_path}")
    elif cfg.verbose:
        print(f"run_dir={run_dir}")
    return result


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="HRL crypto trading bot")
    p.add_argument("--config", default="configs/smoke.yaml")
    p.add_argument(
        "--mode", choices=["train", "test", "full"], default=None,
        help="Override YAML mode",
    )
    p.add_argument(
        "--pair",
        choices=["ppo_sac", "sac_sac", "ppo_td3"],
        default=None,
        help="Override manager/worker pair",
    )
    p.add_argument(
        "--schedule",
        choices=["joint", "alternating"],
        default=None,
        help="Override train_schedule (joint=default, alternating)",
    )
    args = p.parse_args(argv)

    cfg = HRLConfig.from_yaml(args.config)
    updates: dict = {}
    if args.mode is not None:
        updates["mode"] = args.mode
    if args.pair is not None:
        updates["pair"] = args.pair
    if args.schedule is not None:
        updates["train_schedule"] = args.schedule
    if updates:
        cfg = cfg.model_copy(update=updates)

    if cfg.verbose:
        print(
            f"HRL pair={cfg.pair} mode={cfg.mode} "
            f"schedule={cfg.train_schedule} "
            f"timesteps={cfg.total_timesteps} "
            f"n_envs={cfg.env.n_envs} assets={cfg.env.n_assets} "
            f"tf={cfg.env.low_tf}/{cfg.env.high_tf}"
        )
    run(cfg)


if __name__ == "__main__":
    main()
