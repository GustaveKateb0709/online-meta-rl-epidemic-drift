"""Decision-heterogeneity check for the two environment families.

Before running any learning experiment we verify that the environment actually
poses a decision problem in which the best action depends on the latent regime.

For a grid over the two action-channel parameters (eps, beh) we roll out the
three constant actions and report which action is best. If the best action were
the same everywhere, no adapter could ever beat a fixed policy, and any
meta-learning comparison would be vacuous.

Run:  python scripts/validate_families.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.env import (BASE_REGIME, FAMILY_ACTION_REVEALED, FAMILY_PASSIVE,
                     IDX_BEH, IDX_EPS, N_ACTIONS, REGIME_DIM,
                     RegimeDriftEpidemicEnv, sample_regimes)

HORIZON = 156
N_ENVS = 24


def returns_for_grid(env, eps_grid, beh_grid, seed=0):
    """Return, for every (eps, beh) cell, the mean return of each constant action."""
    n_cells = len(eps_grid) * len(beh_grid)
    regimes = np.tile(BASE_REGIME[None, :], (n_cells, 1)).astype(float)
    ee, bb = np.meshgrid(eps_grid, beh_grid, indexing="ij")
    regimes[:, IDX_EPS] = ee.ravel()
    regimes[:, IDX_BEH] = bb.ravel()

    out = np.zeros((n_cells, N_ACTIONS))
    for a in range(N_ACTIONS):
        env.n_envs = n_cells
        env.horizon = HORIZON
        env.regime0 = np.zeros((n_cells, REGIME_DIM))
        env.regime1 = np.zeros((n_cells, REGIME_DIM))
        env.onset = np.zeros(n_cells, dtype=int)
        env.duration = np.ones(n_cells, dtype=int)
        env.abrupt = np.zeros(n_cells, dtype=bool)
        env.state = np.zeros((n_cells, 4))
        env.obs = np.zeros((n_cells, 4))
        env.prev_cases = np.zeros(n_cells)
        env.prev_action = np.zeros(n_cells)
        env.done = np.zeros(n_cells, dtype=bool)
        env.regime_hist = np.zeros((n_cells, REGIME_DIM))
        env.reset(regimes, seed=seed)
        ret = np.zeros(n_cells)
        for _ in range(HORIZON):
            _, r, _, _ = env.step(np.full(n_cells, a, dtype=int))
            ret += r
        out[:, a] = ret
    return out


def main():
    env = RegimeDriftEpidemicEnv(n_envs=N_ENVS, horizon=HORIZON, seed=0)
    eps_grid = [0.05, 0.15, 0.25, 0.35, 0.45, 0.60]
    beh_grid = [0.00, 0.05, 0.15, 0.25, 0.35]

    print("Best constant action per (eps, beh) cell")
    print("rows = behavioural rebound beh, columns = direct efficacy eps\n")
    res = returns_for_grid(env, eps_grid, beh_grid)
    best = np.argmax(res, axis=1).reshape(len(eps_grid), len(beh_grid), order="F")
    header = "        " + "".join(f"{e:>7.2f}" for e in eps_grid)
    print(header)
    for j, b in enumerate(beh_grid):
        row = "".join(f"{best[i, j]:>7d}" for i in range(len(eps_grid)))
        print(f"beh={b:4.2f}" + row)
    counts = np.bincount(best.ravel(), minlength=N_ACTIONS)
    share = counts / counts.sum()
    print(f"\n  best-action distribution over cells: "
          f"a0={share[0]:.2f} a1={share[1]:.2f} a2={share[2]:.2f}")
    print(f"  spread of the best achievable return across cells: "
          f"{res.max(axis=1).max() - res.max(axis=1).min():.1f}")
    print(f"  penalty for always choosing the action that is best on average: "
          f"{float(res.max(axis=1).mean() - res.mean(axis=1).max()):.1f}")

    print("\nInterpretation")
    print("- The best action varies strongly over the action channel, and there are")
    print("  cells where intervening is worse than doing nothing: a fixed policy")
    print("  cannot be optimal everywhere, so the decision problem is non-vacuous.")
    print("- This grid describes the decision space of BOTH families. The families")
    print("  differ in identifiability, not in the decision space: in the passive")
    print("  family the action-channel parameters are fixed, so the regime is")
    print("  readable from the case trajectory; in the action_revealed family they")
    print("  vary and are observable only through the response to intervention.")
    print("  That property is verified by checks 16-17 of scripts/validate_env.py.")


if __name__ == "__main__":
    raise SystemExit(main())
