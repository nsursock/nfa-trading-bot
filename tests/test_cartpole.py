import mlx.core as mx

from scripts.envs.cartpole import CartPoleEnv


def test_cartpole_dynamics():
    env = CartPoleEnv(n_envs=3, seed=0)
    obs = env.reset()
    assert obs.shape == (3, 4)
    assert bool(mx.all(mx.abs(obs) <= 0.05))

    done_at = None
    for step_i in range(500):
        obs, reward, done, infos = env.step(mx.ones((3,), dtype=mx.int32))
        assert reward.shape == (3,)
        assert bool(mx.all(reward == 1.0))
        if done_at is None and bool(mx.any(done)):
            done_at = step_i
            i = done.tolist().index(True)
            assert "episode" in infos[i]
            assert infos[i]["episode"]["l"] == step_i + 1
            assert infos[i]["terminal_observation"].shape == (4,)
            # auto-reset: obs is fresh, within +-0.05
            assert bool(mx.all(mx.abs(obs[i]) <= 0.05))
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
        obs, reward, done, infos = env.step(actions)
        if bool(mx.any(done)):
            truncated_at = step_i
            for i in range(2):
                if done.tolist()[i]:
                    assert infos[i]["TimeLimit.truncated"] is True
                    assert infos[i]["episode"]["l"] == 500
            break
    assert truncated_at == 499
