"""Commitment-formalization self-test (tiny scale; commitment stays OFF elsewhere).

Under the commitment formalization the controller picks one intervention level
per decision cycle and cannot change it (default: one cycle per episode). This
turns the problem into a 3-arm regime-identification problem.

This script only checks that the code path runs and prints preliminary numbers:
it trains ``meta_online`` and the Robust baseline, with commitment ON and OFF, on
the SAME 8 stationary ``action_revealed`` tasks, and reports the committed-level
distribution and the share of the available gap that is captured.

No conclusion is drawn here -- this is a contingency, and the default stays OFF.

Run:  python -u scripts/00i_commitment_selftest.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines import (ConstantController, FixedController, RobustBaseline,  # noqa: E402
                           eval_metrics, rollout_episode)
from src.commitment import CommitmentWrapper, wrap_cfg  # noqa: E402
from src.common import DEFAULT_CONFIG, seed_everything, set_threads  # noqa: E402
from src.env import (FAMILY_ACTION_REVEALED, N_ACTIONS, OBS_DIM,  # noqa: E402
                     RegimeDriftEpidemicEnv, sample_regimes)
from src.meta_rl import ContextBuffer, MetaRLAgent, train_meta  # noqa: E402

ACTIONS = (0, 1, 2)
BIG = 10 ** 6


def stationary_drift(regimes: np.ndarray) -> dict:
    B = regimes.shape[0]
    return {"regime0": regimes.copy(), "regime1": regimes.copy(),
            "onset": np.full(B, BIG), "duration": np.ones(B, dtype=int),
            "abrupt": np.zeros(B, dtype=bool)}


@torch.no_grad()
def meta_rollout(agent, env, regimes, horizon, seed, z_mode="online"):
    """Deterministic meta rollout recording the APPLIED action (post-commitment)."""
    dev = agent.dev
    B = env.n_envs
    buf = ContextBuffer(B, agent.cfg["ctx_len"], OBS_DIM, N_ACTIONS)
    obs = env.reset(regimes, seed=seed)
    fz = torch.zeros(B, agent.policy_z_dim, device=dev)
    fr = torch.zeros(B, dtype=torch.bool, device=dev)
    applied = np.zeros((horizon, B), dtype=int)
    rew = np.zeros((horizon, B))
    for t in range(horizon):
        z, fz, fr = agent._step_z(buf, env, z_mode, fz, fr)
        o = torch.as_tensor(obs, dtype=torch.float32, device=dev)
        a = agent.policy(o, z).argmax(-1).cpu().numpy()
        nobs, raw, done, _ = env.step(a)
        applied[t] = (env.applied if isinstance(env, CommitmentWrapper)
                      else np.asarray(a, dtype=int))
        buf.push(obs, a, raw * agent.cfg["reward_scale"], nobs)
        rew[t] = raw
        obs = nobs
    return applied, rew


def run_mode(commit, cfg_base, regimes, env_factory, seed, B, H):
    cfg = dict(cfg_base)
    cfg.update(commitment=bool(commit), commitment_cycle_len=None, commitment_warmup=0)
    seed_everything(seed)

    agent = MetaRLAgent(dict(cfg, z_mode="online"))
    t0 = time.perf_counter()
    train_meta(agent, seed, commitment=commit)
    meta_wall = time.perf_counter() - t0

    robust = RobustBaseline(cfg)
    t1 = time.perf_counter()
    robust.fit(seed)
    rob_wall = time.perf_counter() - t1

    drift = stationary_drift(regimes)
    env = wrap_cfg(env_factory(), cfg)
    applied, rew = meta_rollout(agent, env, regimes, H, seed + 11, "online")
    meta_ret = float(rew.sum(0).mean())

    rob_rew = rollout_episode(wrap_cfg(env_factory(), cfg), FixedController(robust.agent),
                              drift, seed + 21)
    rob_ret = float(rob_rew.sum(0).mean())

    consts = {}
    for a in ACTIONS:
        r = rollout_episode(wrap_cfg(env_factory(), cfg), ConstantController(a), drift, seed + 60 + a)
        consts[a] = float(r.sum(0).mean())

    committed = np.array([np.bincount(applied[:, i], minlength=3).argmax() for i in range(B)])
    return {"commitment": bool(commit), "meta_return": meta_ret, "robust_return": rob_ret,
            "const_returns": {str(a): consts[a] for a in ACTIONS},
            "committed_levels": [int(x) for x in committed],
            "committed_level_counts": [int(c) for c in np.bincount(committed, minlength=3)],
            "applied_step_counts": [float((applied == a).mean()) for a in ACTIONS],
            "meta_wall_seconds": meta_wall, "robust_wall_seconds": rob_wall}


def main():
    set_threads(2)  # two threads keeps this diagnostic light on an 8 GB machine
    art = ROOT / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    logf = open(art / "commitment_selftest.txt", "w")

    class _Tee:
        def __init__(self, *s):
            self.s = s

        def write(self, x):
            for f in self.s:
                f.write(x)

        def flush(self):
            for f in self.s:
                f.flush()

    sys.stdout = _Tee(sys.__stdout__, logf)

    cfg = dict(DEFAULT_CONFIG)
    cfg.update(n_tasks=8, horizon=120, iterations=80, hidden=64, ctx_len=12, z_dim=5,
               ppo_epochs=3, n_minibatch=3, family=FAMILY_ACTION_REVEALED)
    B, H, SEED = 8, cfg["horizon"], 10

    rng = np.random.default_rng(SEED)
    regimes = sample_regimes(rng, B, family=FAMILY_ACTION_REVEALED)
    env_factory = lambda: RegimeDriftEpidemicEnv(n_envs=B, horizon=H, seed=SEED + 777)

    consts = {}
    for a in ACTIONS:
        draft = stationary_drift(regimes)
        r = rollout_episode(env_factory(), ConstantController(a), draft, SEED + 60 + a)
        consts[a] = r.sum(0)
    cr = np.stack([consts[a] for a in ACTIONS], axis=1)   # (B,3)
    astar = cr.argmax(1)
    best_single = float(cr.mean(0).max())
    best_per_task = float(cr.max(1).mean())
    gap = best_per_task - best_single

    print("=" * 78, flush=True)
    print("COMMITMENT SELF-TEST (tiny; default OFF elsewhere)")
    print(f"family={FAMILY_ACTION_REVEALED} stationary, B={B}, H={H}, iters={cfg['iterations']}")
    print(f"best single const = {best_single:.2f} | per-task optimal = {best_per_task:.2f} "
          f"| available gap = {gap:.2f}")
    print(f"a* per task = {[int(a) for a in astar]}")
    print("=" * 78, flush=True)

    out = {"cfg": cfg, "best_single_const": best_single, "best_per_task_const": best_per_task,
           "available_gap": gap, "a_star": [int(a) for a in astar], "modes": {}}
    outdir = ROOT / "results" / "validation"
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "commitment_selftest.json"

    for commit in (True, False):
        print(f"\n-> running commitment={commit} ...", flush=True)
        rec = run_mode(commit, cfg, regimes, env_factory, SEED, B, H)
        rec["meta_correct"] = float(np.mean(np.array(rec["committed_levels"]) == astar))
        rec["meta_captured"] = (float((rec["meta_return"] - best_single) / gap)
                                if gap > 1e-9 else float("nan"))
        rec["robust_captured"] = (float((rec["robust_return"] - best_single) / gap)
                                  if gap > 1e-9 else float("nan"))
        out["modes"][f"commitment_{commit}"] = rec
        a = rec["applied_step_counts"]
        print(f"   meta ret={rec['meta_return']:.1f} captured={rec['meta_captured']:+.2f} "
              f"correct={rec['meta_correct']:.2f} | "
              f"robust ret={rec['robust_return']:.1f} captured={rec['robust_captured']:+.2f}", flush=True)
        print(f"   committed-level counts (a0/a1/a2) = {rec['committed_level_counts']} | "
              f"applied step shares = {a[0]:.2f}/{a[1]:.2f}/{a[2]:.2f}", flush=True)
        outpath.write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 78)
    print("PRELIMINARY ONLY -- no conclusion; commitment remains OFF by default.")
    print(f"wrote {outpath}")
    logf.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
