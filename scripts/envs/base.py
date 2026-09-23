from typing import NamedTuple

import mlx.core as mx


class StepResult(NamedTuple):
    obs: mx.array           # (n, obs_dim) observation after auto-reset
    reward: mx.array        # (n,) float32
    done: mx.array          # (n,) bool = terminated | truncated
    terminated: mx.array    # (n,) bool  real termination
    truncated: mx.array     # (n,) bool  time limit hit AND not terminated (SB3 "TimeLimit.truncated")
    terminal_obs: mx.array  # (n, obs_dim) obs before auto-reset (== obs where not done)
    ep_ret: mx.array        # (n,) float32 return of the episode that ended this step, 0 elsewhere
    ep_len: mx.array        # (n,) int32   length of that episode, 0 elsewhere
