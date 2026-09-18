"""Small-scale diagnostics for the meta-RL agent on the fixed environment.

Answers three questions before committing to the long run:

1. Does the online meta-policy still collapse to a (near-)constant action?
2. Does it beat the best constant-action reference and the Robust baseline?
3. How sensitive is it to the KL weight and the context length K?

Everything is run at a deliberately small scale; numbers here are diagnostics,
not final results. All values printed come from real runs.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines import (ConstantController, FixedController, PPOAgent,
                           eval_metrics, rollout_episode)
from src.common import seed_everything, set_threads
from src.env import RegimeDriftEpidemicEnv, make_drift, sample_regimes
from src.gating import online_rollout
from src.meta_rl import MetaRLAgent, train_meta

H = 120
N_TASKS = 16
ITERS = 100
N_EVAL = 8
KINDS = ["gradual_beta", "endogenous_beh"]
STRENGTHS = [1.0, 2.0]


def action_hist(actions):
    a = actions.reshape(-1)
    return [float((a == k).mean()) for k in (0, 1, 2)]


def eval_agent(agent, env, drift, seed, z_mode="online"):
    roll = online_rollout(agent, env, drift, seed, deterministic=True, z_mode=z_mode)
    m = eval_metrics(roll["rew"], drift["onset"], env.horizon)
    m["action_frac"] = action_hist(roll["actions"])
    return m


def main():
    set_threads(4)
    env = RegimeDriftEpidemicEnv(n_envs=N_EVAL, horizon=H, seed=777)
    rng = np.random.default_rng(31337)
    drifts = {(k, s): make_drift(rng, k, N_EVAL, H, strength=s)
              for k in KINDS for s in STRENGTHS}

    # constant references
    print("=" * 78)
    print("constant-action reference returns (mean over tasks)")
    const_ref = {}
    for k in KINDS:
        for s in STRENGTHS:
            vals = {}
            for a in (0, 1, 2):
                rew = rollout_episode(env, ConstantController(a), drifts[(k, s)], 1)
                vals[a] = rew.sum(0).mean()
            const_ref[(k, s)] = vals
            print(f"  {k:<15} s={s}: const0 {vals[0]:8.2f}  const1 {vals[1]:8.2f}  const2 {vals[2]:8.2f}")

    # robust reference
    seed_everything(0)
    rb = PPOAgent(dict(n_tasks=N_TASKS, horizon=H, iterations=ITERS, hidden=64))
    rngt = np.random.default_rng(0)
    etr = RegimeDriftEpidemicEnv(n_envs=N_TASKS, horizon=H, seed=0)
    t0 = time.perf_counter()
    for _ in range(ITERS):
        regs = sample_regimes(rngt, N_TASKS)
        roll = rb.collect(etr, lambda e: e.reset(regs), H)
        rb.update(roll)
    print(f"\n[robust trained in {time.perf_counter()-t0:.1f}s]")
    for k in KINDS:
        for s in STRENGTHS:
            rew = rollout_episode(env, FixedController(rb), drifts[(k, s)], 21)
            m = eval_metrics(rew, drifts[(k, s)]["onset"], H)
            print(f"  robust {k:<15} s={s}: {m['return_mean']:8.2f}")

    # ---- sweeps: kl_coef x ctx_len ---------------------------------- #
    kl_grid = [0.01, 0.1, 1.0]
    k_grid = [8, 12, 24]
    print("\n" + "=" * 78)
    print("meta_online sweep (100 iters, 16 tasks, horizon 120, hidden 64)")
    print(f"{'kl':>6}{'K':>4} | " + " | ".join(f"{k[:9]}_s{s}" for k in KINDS for s in STRENGTHS)
          + " | a0/a1/a2 frac")
    results = []
    for kl in kl_grid:
        for K in k_grid:
            cfg = dict(n_tasks=N_TASKS, horizon=H, iterations=ITERS, hidden=64,
                       z_dim=5, ctx_len=K, kl_coef=kl, z_mode="online")
            ag = MetaRLAgent(cfg)
            tt = time.perf_counter()
            train_meta(ag, 0, verbose=False)
            row = []
            frac_last = None
            for k in KINDS:
                for s in STRENGTHS:
                    m = eval_agent(ag, env, drifts[(k, s)], 11)
                    row.append(m["return_mean"])
                    frac_last = m["action_frac"]
            results.append((kl, K, row, frac_last, time.perf_counter() - tt))
            print(f"{kl:>6}{K:>4} | " + " | ".join(f"{v:11.2f}" for v in row)
                  + f" | {frac_last[0]:.2f}/{frac_last[1]:.2f}/{frac_last[2]:.2f}")

    print("\nBest sweep config by mean return across the 4 cells:")
    def score(r):
        return float(np.mean(r[2]))
    for kl, K, row, frac, dt in sorted(results, key=score, reverse=True)[:3]:
        print(f"  kl={kl} K={K}: mean {score((kl,K,row,frac,dt)):.2f}  "
              f"per-cell {[round(v,1) for v in row]}  action-frac a0/a1/a2 "
              f"{frac[0]:.2f}/{frac[1]:.2f}/{frac[2]:.2f}  ({dt:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
