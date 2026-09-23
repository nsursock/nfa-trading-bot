import math

import mlx.core as mx

from scripts.envs.pendulum import PendulumEnv


def test_pendulum_reset():
    env = PendulumEnv(n_envs=4, seed=0)
    obs = env.reset()
    assert obs.shape == (4, 3)
    cos_sq = obs[:, 0] ** 2 + obs[:, 1] ** 2
    assert bool(mx.all(mx.abs(cos_sq - 1.0) < 1e-5))
    assert bool(mx.all(mx.abs(obs[:, 2]) <= 1.0))


def test_pendulum_reward_range():
    env = PendulumEnv(n_envs=8, seed=0)
    env.reset()
    for _ in range(50):
        res = env.step(
            mx.random.uniform(-2.0, 2.0, shape=(8, 1))
        )
        assert bool(mx.all(res.reward <= 0.0))
        assert bool(mx.all(res.reward >= -16.2736044))


def test_pendulum_truncation():
    env = PendulumEnv(n_envs=2, seed=0)
    env.reset()
    for step_i in range(200):
        res = env.step(mx.zeros((2, 1)))
        if bool(mx.any(res.done)):
            assert step_i == 199
            for i in range(2):
                assert bool(res.truncated[i]) is True
                assert bool(res.terminated[i]) is False
                assert res.ep_len[i].item() == 200
                r = res.ep_ret[i].item()
                assert math.isfinite(r) and r < 0
            # auto-reset: fresh obs consistent + step counter restarted
            assert env.steps.tolist() == [0, 0]
            break
    else:
        raise AssertionError("Pendulum should truncate at 200 steps")


def test_pendulum_single_step():
    env = PendulumEnv(n_envs=1, seed=0)
    env.reset()
    env.state = mx.array([[0.1, 0.0]])
    res = env.step(mx.array([[0.0]]))

    th, thdot, u = 0.1, 0.0, 0.0
    cost = th**2 + 0.1 * thdot**2 + 0.001 * u**2
    newthdot = thdot + (3 * 10.0 / 2 * math.sin(th) + 3.0 * u) * 0.05
    newthdot = min(max(newthdot, -8.0), 8.0)
    newth = th + newthdot * 0.05

    o = res.obs[0].tolist()
    assert abs(o[0] - math.cos(newth)) < 1e-5
    assert abs(o[1] - math.sin(newth)) < 1e-5
    assert abs(o[2] - newthdot) < 1e-5
    assert abs(res.reward[0].item() - (-cost)) < 1e-5
    assert not bool(res.done[0])
