"""Where (if anywhere) does online latent inference help?

Trains meta_online, meta_frozen and robust at the same budget, then evaluates
per drift KIND (all four kinds, both strengths) plus the stationary control. The
hypothesis under test: latent inference should help most when observations are
AMBIGUOUS about the true state -- notably `gradual_rho`, where the reporting
rate drops so observed case counts understate true burden.

All numbers are real runs at small scale (diagnostics).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines import (ConstantController, FixedController, PPOAgent,
                           eval_metrics, rollout_episode)
from src.common import seed_everything, set_threads
from src.env import DRIFT_KINDS, RegimeDriftEpidemicEnv, make_drift, sample_regimes
from src.gating import online_rollout
from src.meta_rl import MetaRLAgent

H = 120
N_TASKS = 16
ITERS = 150
N_EVAL = 8
STRENGTHS = [1.0, 2.0]


def main():
    set_threads(4)
    seed_everything(0)
    eval_env = RegimeDriftEpidemicEnv(n_envs=N_EVAL, horizon=H, seed=777)
    rng = np.random.default_rng(31337)
    cells = [("stationary", 1.0)] + [(k, s) for k in DRIFT_KINDS for s in STRENGTHS]
    drifts = {(k, s): make_drift(rng, k, N_EVAL, H, strength=s) for (k, s) in cells}

    base = dict(n_tasks=N_TASKS, horizon=H, hidden=64, ppo_epochs=4, n_minibatch=4, ctx_len=8)
    meta = MetaRLAgent(dict(base, kl_coef=0.1, z_mode="online"))
    frozen = MetaRLAgent(dict(base, kl_coef=0.1, z_mode="frozen"))
    robust = PPOAgent(base)
    envs = [RegimeDriftEpidemicEnv(N_TASKS, H, 100 + i) for i in range(3)]
    rngs = [np.random.default_rng(10 + i) for i in range(3)]

    for it in range(1, ITERS + 1):
        meta.update(meta.collect_rollout(envs[0], sample_regimes(rngs[0], N_TASKS)))
        frozen.update(frozen.collect_rollout(envs[1], sample_regimes(rngs[1], N_TASKS)))
        robust.update(robust.collect(envs[2], lambda e: e.reset(sample_regimes(rngs[2], N_TASKS)), H))

    def em(agent, d):
        return eval_metrics(online_rollout(agent, eval_env, d, 11, z_mode=agent.cfg["z_mode"])["rew"],
                            d["onset"], H)["return_mean"]

    def er(d):
        return eval_metrics(rollout_episode(eval_env, FixedController(robust), d, 21),
                            d["onset"], H)["return_mean"]

    def ec(a, d):
        return eval_metrics(rollout_episode(eval_env, ConstantController(a), d, 31),
                            d["onset"], H)["return_mean"]

    print(f"{'kind':<16}{'s':>5}{'meta_online':>13}{'meta_frozen':>13}{'robust':>10}"
          f"{'const2':>10}{'meta-robust':>13}")
    for (k, s) in cells:
        d = drifts[(k, s)]
        mo, mf, ro, c2 = em(meta, d), em(frozen, d), er(d), ec(2, d)
        print(f"{k:<16}{s:>5}{mo:>13.2f}{mf:>13.2f}{ro:>10.2f}{c2:>10.2f}{mo - ro:>13.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
