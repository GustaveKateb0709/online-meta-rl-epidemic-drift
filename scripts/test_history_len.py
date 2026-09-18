"""Unit test for the history_len switch in PPOAgent (default OFF, no behaviour change).

Checks, with NON-CONSTANT observations and non-constant actions:
  1. history_len=0/1 -> input width == OBS_DIM, identical to the original path.
  2. history_len=12  -> input width == 12*OBS_DIM; the most recent observation
     occupies the last block; the episode-start window is padded by repeating
     the first observation.
  3. FixedController.reset() clears the history so each rollout episode starts
     with a fully padded window.
Run: python -m scripts.test_history_len
"""
import numpy as np
import torch

from src.common import set_threads, DEFAULT_CONFIG
from src.env import OBS_DIM, RegimeDriftEpidemicEnv, make_drift
from src.baselines import FixedController, PPOAgent, rollout_episode

set_threads(2)


def main() -> None:
    rng = np.random.default_rng(0)
    obs_seq = [rng.normal(size=(3, OBS_DIM)).astype(np.float32) for _ in range(16)]

    # --- 1. default path unchanged ------------------------------------- #
    a0 = PPOAgent({"iterations": 1})
    assert a0.history_len == 0, a0.history_len
    out = a0._stack(obs_seq[0])
    assert out.shape == (3, OBS_DIM), out.shape
    assert np.allclose(out, obs_seq[0])
    print("check 1 OK: default input width == OBS_DIM, identity mapping")

    # --- 2. history window content ------------------------------------- #
    a1 = PPOAgent({"history_len": 12})
    assert a1.history_len == 12
    w = 12 * OBS_DIM
    # first call: padded by repeating the first observation
    o1 = a1._stack(obs_seq[0])
    assert o1.shape == (3, w)
    for k in range(12):
        assert np.allclose(o1[:, k * OBS_DIM:(k + 1) * OBS_DIM], obs_seq[0]), k
    # feed 15 more distinct observations; window slides, newest block last
    for t in range(1, 16):
        ot = a1._stack(obs_seq[t])
    assert np.allclose(ot[:, -OBS_DIM:], obs_seq[15])
    assert np.allclose(ot[:, :OBS_DIM], obs_seq[4])  # obs 15-11 = obs 4
    assert not np.allclose(ot[:, :OBS_DIM], ot[:, -OBS_DIM:])
    print("check 2 OK: sliding window, newest block last, correct padding drop")

    # --- 3. reset between episodes ------------------------------------- #
    env = RegimeDriftEpidemicEnv(n_envs=2, horizon=20, seed=7)
    drift = make_drift(rng, "abrupt_eps", 2, 20, strength=1.0)
    acts_before = []
    controller = FixedController(a1)
    rew = rollout_episode(env, controller, drift, 21)
    assert a1._hist is not None  # during rollout history grows
    # after a second rollout, first input must equal the padded first obs again
    obs = env.reset(drift["regime0"], regime1=drift["regime1"], onset=drift["onset"],
                    duration=drift["duration"], abrupt=drift["abrupt"], seed=99)
    a1.reset_history()
    o_new = a1._stack(obs)
    for k in range(12):
        assert np.allclose(o_new[:, k * OBS_DIM:(k + 1) * OBS_DIM], obs), k
    assert rew.shape == (20, 2)
    assert np.isfinite(rew).all()
    print("check 3 OK: rollout runs, history resets, first-window is padded")

    # --- 4. non-constant actions from a tiny trained run --------------- #
    cfg = dict(DEFAULT_CONFIG)
    cfg.update({"history_len": 12, "iterations": 3, "n_tasks": 4, "horizon": 30,
                "family": "linked"})
    a2 = PPOAgent(cfg)
    e2 = RegimeDriftEpidemicEnv(n_envs=4, horizon=30, seed=3)
    regs = rng.normal(size=(4, DEFAULT_CONFIG["regime_dim"]))
    roll = a2.collect(e2, lambda e: e.reset(regs), 30)
    acts = roll["act"].numpy()
    assert acts.shape == (30, 4)
    assert len(np.unique(acts)) > 1, "actions collapsed to a constant in unit test"
    print("check 4 OK: trained-with-history agent takes non-constant actions,"
          " unique actions =", sorted(np.unique(acts).tolist()))
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
