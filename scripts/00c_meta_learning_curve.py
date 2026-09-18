"""Learning-curve probe: does online latent inference buy SAMPLE EFFICIENCY?

Trains meta_online, meta_frozen and robust (plain PPO) in lockstep on the same
environment budget and evaluates all three every EVAL_EVERY iterations on four
drift cells. The paper's claim is about sample efficiency / adaptation speed, so
this is the relevant comparison -- final-return ties do not settle it.

All numbers are real runs; small scale, so treat as diagnostics.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines import FixedController, PPOAgent, eval_metrics, rollout_episode
from src.common import seed_everything, set_threads
from src.env import RegimeDriftEpidemicEnv, make_drift, sample_regimes
from src.gating import online_rollout
from src.meta_rl import MetaRLAgent

H = 120
N_TASKS = 16
TOTAL_ITERS = 200
EVAL_EVERY = 25
N_EVAL = 8
KINDS = ["gradual_beta", "endogenous_beh"]
STRENGTHS = [1.0, 2.0]


def mean_return_across_cells(eval_env, drifts, eval_fn):
    vals = [eval_fn(drifts[(k, s)])["return_mean"] for k in KINDS for s in STRENGTHS]
    return float(np.mean(vals)), vals


def main():
    set_threads(4)
    seed_everything(0)
    eval_env = RegimeDriftEpidemicEnv(n_envs=N_EVAL, horizon=H, seed=777)
    rng = np.random.default_rng(31337)
    drifts = {(k, s): make_drift(rng, k, N_EVAL, H, strength=s)
              for k in KINDS for s in STRENGTHS}

    base = dict(n_tasks=N_TASKS, horizon=H, hidden=64, ppo_epochs=4, n_minibatch=4, ctx_len=12)
    meta = MetaRLAgent(dict(base, kl_coef=0.1, z_mode="online"))
    frozen = MetaRLAgent(dict(base, kl_coef=0.1, z_mode="frozen"))
    robust = PPOAgent(base)

    env_m = RegimeDriftEpidemicEnv(n_envs=N_TASKS, horizon=H, seed=101)
    env_f = RegimeDriftEpidemicEnv(n_envs=N_TASKS, horizon=H, seed=102)
    env_r = RegimeDriftEpidemicEnv(n_envs=N_TASKS, horizon=H, seed=103)
    rng_m = np.random.default_rng(1)
    rng_f = np.random.default_rng(2)
    rng_r = np.random.default_rng(3)

    def eval_meta(agent, drift):
        return eval_metrics(online_rollout(agent, eval_env, drift, 11, z_mode=agent.cfg["z_mode"])["rew"],
                            drift["onset"], H)

    def eval_robust(drift):
        return eval_metrics(rollout_episode(eval_env, FixedController(robust), drift, 21),
                            drift["onset"], H)

    print(f"{'iters':>6}{'env_steps':>12}{'meta_online':>13}{'meta_frozen':>13}{'robust':>11}")
    rows = []
    for it in range(1, TOTAL_ITERS + 1):
        rm = meta.collect_rollout(env_m, sample_regimes(rng_m, N_TASKS))
        meta.update(rm)
        rf = frozen.collect_rollout(env_f, sample_regimes(rng_f, N_TASKS))
        frozen.update(rf)
        rp = robust.collect(env_r, lambda e: e.reset(sample_regimes(rng_r, N_TASKS)), H)
        robust.update(rp)
        if it % EVAL_EVERY == 0 or it == 1:
            mo, _ = mean_return_across_cells(eval_env, drifts, lambda d: eval_meta(meta, d))
            mf, _ = mean_return_across_cells(eval_env, drifts, lambda d: eval_meta(frozen, d))
            ro, _ = mean_return_across_cells(eval_env, drifts, eval_robust)
            print(f"{it:>6}{it*N_TASKS*H:>12}{mo:>13.2f}{mf:>13.2f}{ro:>11.2f}")
            rows.append((it, mo, mf, ro))

    print("\nreturn at last three checkpoints (mean over 4 cells):")
    for it, mo, mf, ro in rows[-3:]:
        print(f"  iters={it}: meta_online {mo:.2f} | meta_frozen {mf:.2f} | robust {ro:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
