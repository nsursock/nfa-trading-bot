import mlx.core as mx

from scripts.envs.cartpole import CartPoleEnv


def test_cartpole_dynamics():
    env = CartPoleEnv(n_envs=3, seed=0)
    obs = env.reset()
    assert obs.shape == (3, 4)
    assert bool(mx.all(mx.abs(obs) <= 0.05))

    done_at = None
    for step_i in range(500):
        res = env.step(mx.ones((3,), dtype=mx.int32))
        assert res.reward.shape == (3,)
        assert bool(mx.all(res.reward == 1.0))
        if done_at is None and bool(mx.any(res.done)):
            done_at = step_i
            i = res.done.tolist().index(True)
            assert res.ep_len[i].item() == step_i + 1
            assert res.ep_ret[i].item() == step_i + 1
            assert res.terminal_obs.shape == (3, 4)
            # auto-reset: obs is fresh, within +-0.05
            assert bool(mx.all(mx.abs(res.obs[i]) <= 0.05))
            break
    assert done_at is not None, "pushing right constantly should terminate"


def test_cartpole_truncation():
    env = CartPoleEnv(n_envs=2, seed=0)
    env.reset()
    env.state = mx.zeros((2, 4))
    env.steps = mx.zeros((2,), dtype=mx.int32)
    truncated_at = None
    for step_i in range(500):
        # PD-like bang-bang controller keeps the pole up from the zero state
        theta, theta_dot = env.state[:, 2], env.state[:, 3]
        actions = (theta + 0.5 * theta_dot > 0).astype(mx.int32)
        res = env.step(actions)
        if bool(mx.any(res.done)):
            truncated_at = step_i
            for i in range(2):
                if res.done.tolist()[i]:
                    assert bool(res.truncated[i]) is True
                    assert bool(res.terminated[i]) is False
                    assert res.ep_len[i].item() == 500
            break
    assert truncated_at == 499
