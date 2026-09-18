"""What is actually achievable in this environment?

Before spending compute on learning, we bound the payoff structure by brute
force. For a batch of tasks we compute, under identical dynamics and seeds:

  * the return of every constant action;
  * the best return achievable by a simple state-dependent rule that reads the
    same observation the agent sees (a threshold on the observed case level, and
    a threshold on the observed growth).

If a one-parameter threshold rule beats the best constant action by a wide
margin, then the decision problem contains a large exploitable gap and any
learner that fails to capture it is under-trained rather than up against an
impossible task. If no simple rule helps, the problem is genuinely hard for
memoryless policies and the latent parameters must be inferred from history.

Run:  python scripts/validate_achievable.py [family]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.env import (FAMILIES, FAMILY_ACTION_REVEALED, FAMILY_PASSIVE, N_ACTIONS,
                     OBS_DIM, REGIME_DIM, RegimeDriftEpidemicEnv, make_drift)

HORIZON = 156
N_TASKS = 96
DRIFT = "stationary"


def reset_env(env, n, regimes, regime1=None, onset=None, duration=None,
              abrupt=None, seed=0):
    env.n_envs = n
    env.horizon = HORIZON
    env.regime0 = np.zeros((n, REGIME_DIM))
    env.regime1 = np.zeros((n, REGIME_DIM))
    env.onset = np.zeros(n, dtype=int)
    env.duration = np.ones(n, dtype=int)
    env.abrupt = np.zeros(n, dtype=bool)
    env.state = np.zeros((n, 4))
    env.obs = np.zeros((n, OBS_DIM))
    env.prev_cases = np.zeros(n)
    env.prev_action = np.zeros(n)
    env.done = np.zeros(n, dtype=bool)
    env.regime_hist = np.zeros((n, REGIME_DIM))
    return env.reset(regimes, regime1=regime1, onset=onset, duration=duration,
                     abrupt=abrupt, seed=seed)


def rollout(env, n, regimes, policy, drift=None, seed=0):
    """Return per-task returns and post-onset returns for a callable policy."""
    if drift is None:
        obs = reset_env(env, n, regimes, seed=seed)
    else:
        obs = reset_env(env, n, regimes, regime1=drift["regime1"],
                        onset=drift["onset"], duration=drift["duration"],
                        abrupt=drift["abrupt"], seed=seed)
    total = np.zeros(n)
    post = np.zeros(n)
    onset = drift["onset"] if drift is not None else np.full(n, 10 ** 6)
    for t in range(HORIZON):
        a = policy(obs, t)
        obs, r, _, _ = env.step(a)
        total += r
        post += r * (t >= onset)
    return total, post


def main():
    family = sys.argv[1] if len(sys.argv) > 1 else FAMILY_ACTION_REVEALED
    if family not in FAMILIES:
        raise SystemExit(f"family must be one of {FAMILIES}")

    rng = np.random.default_rng(123)
    drift = make_drift(rng, DRIFT, N_TASKS, HORIZON, strength=1.0, family=family)
    regimes = drift["regime0"]
    env = RegimeDriftEpidemicEnv(n_envs=N_TASKS, horizon=HORIZON, seed=0)

    const = np.zeros((N_TASKS, N_ACTIONS))
    for a in range(N_ACTIONS):
        const[:, a] = rollout(env, N_TASKS, regimes,
                              lambda o, t, a=a: np.full(N_TASKS, a, dtype=int),
                              seed=7)[0]
    best_const = const.max(axis=1)
    print(f"family = {family}, {N_TASKS} tasks, horizon {HORIZON}, drift = {DRIFT}\n")
    for a in range(N_ACTIONS):
        print(f"  constant action a={a}: mean return {const[:, a].mean():8.2f}")
    print(f"  best constant per task (oracle over constants): {best_const.mean():8.2f}")

    # Rule family 1: intervene strongly whenever the observed case level is high.
    print("\n  threshold on observed case level  (a=2 if obs[0] > theta else 0)")
    best_rule1 = np.full(N_TASKS, -1e18)
    for theta in np.linspace(0.0, 2.5, 26):
        ret = rollout(env, N_TASKS, regimes,
                      lambda o, t, th=theta: np.where(o[:, 0] > th, 2, 0).astype(int),
                      seed=7)[0]
        best_rule1 = np.maximum(best_rule1, ret)
        if abs(theta - round(theta * 2) / 2) < 1e-9:
            print(f"    theta={theta:4.2f}  mean return {ret.mean():8.2f}")
    print(f"  best per task over this rule family:        {best_rule1.mean():8.2f}")

    # Rule family 2: react to the observed growth signal.
    print("\n  threshold on observed growth      (a=2 if obs[1] > theta else 0)")
    best_rule2 = np.full(N_TASKS, -1e18)
    for theta in np.linspace(-0.05, 0.30, 36):
        ret = rollout(env, N_TASKS, regimes,
                      lambda o, t, th=theta: np.where(o[:, 1] > th, 2, 0).astype(int),
                      seed=7)[0]
        best_rule2 = np.maximum(best_rule2, ret)
    print(f"  best per task over this rule family:        {best_rule2.mean():8.2f}")

    simple = np.maximum(best_rule1, best_rule2)
    print("\n  --- summary ---")
    print(f"  best constant action            : {best_const.mean():8.2f}")
    print(f"  best simple state-dependent rule: {simple.mean():8.2f}")
    print(f"  gap available to a capable learner: "
          f"{simple.mean() - best_const.mean():8.2f}")
    print(f"  share of tasks where a rule beats every constant: "
          f"{np.mean(simple > best_const + 1e-9):.2f}")

    # Also report the oracle regime-dependent payoff: the best constant action
    # chosen per task *with knowledge of the true regime*.
    eps = regimes[:, 6]
    beh = regimes[:, 5]
    order = np.argsort(beh - eps)
    print(f"\n  mean (beh - eps), the sign that decides whether to intervene: "
          f"{np.mean(beh - eps):+.3f}")
    print(f"  share of tasks with beh > eps (intervening raises transmission): "
          f"{np.mean(beh > eps):.2f}")
    low, high = order[:N_TASKS // 3], order[-N_TASKS // 3:]
    print(f"  tasks where the rebound dominates: best constant mean "
          f"{const[high].max(axis=1).mean():8.2f} "
          f"(best action a={np.bincount(const[high].argmax(axis=1), minlength=3)})")
    print(f"  tasks where direct efficacy dominates: best constant mean "
          f"{const[low].max(axis=1).mean():8.2f} "
          f"(best action a={np.bincount(const[low].argmax(axis=1), minlength=3)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
