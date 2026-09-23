import pytest

from utils.bench.throughput import (
    fixed_intensity_config, replay_per_step, resolve_budget,
)


def test_offpolicy_replay_ratio_is_independent_of_n_envs():
    widths = (1024, 2048, 4096)
    for algo in ("sac", "td3"):
        cfgs = [fixed_intensity_config(algo, n, seed=0, log_dir="unused") for n in widths]
        ratios = [replay_per_step(algo, cfg) for cfg in cfgs]
        assert ratios == [256.0, 256.0, 256.0]
        assert {cfg.batch_size for cfg in cfgs} == {256}
        assert {cfg.buffer_size for cfg in cfgs} == {1_000_000}
        assert {cfg.learning_starts for cfg in cfgs} == {100}
        assert [cfg.gradient_steps for cfg in cfgs] == list(widths)


def test_ppo_keeps_yaml_batch_and_epochs():
    cfgs = [fixed_intensity_config("ppo", n, seed=0, log_dir="unused") for n in (1024, 4096)]
    assert all(cfg.batch_size == 256 and cfg.n_epochs == 10 and cfg.n_steps == 256
               for cfg in cfgs)
    assert [replay_per_step("ppo", cfg) for cfg in cfgs] == [10.0, 10.0]


def test_default_budget_is_shared():
    sweep = [1024, 2048, 4096]
    budget = resolve_budget(["sac", "td3"], sweep, None)
    assert budget == 8 * 4096
    assert all(budget % n == 0 for n in sweep)


def test_ppo_budget_must_be_whole_rollouts():
    with pytest.raises(ValueError, match="PPO rollout"):
        resolve_budget(["ppo"], [1024, 2048, 4096], 32_768)
    budget = resolve_budget(["ppo"], [1024, 2048, 4096], None)
    assert budget % (256 * 1024) == 0
    assert budget % (256 * 4096) == 0
