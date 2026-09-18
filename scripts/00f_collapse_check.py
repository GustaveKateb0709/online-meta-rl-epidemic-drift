"""Diagnostic: do the trained policies actually use z, or collapse to a constant action?

Motivation
----------
In the Step-1 validation, ``meta_frozen``, ``meta_nograd``, ``meta_oracle`` and
``robust`` returned almost exactly the same value on ``action_revealed``. That is
only possible if all of them emit the *same constant action* regardless of the
latent regime, i.e. the z-conditioned policy has collapsed onto an
observation-independent, regime-independent controller. This script measures the
action distribution pre- vs post-drift for each agent, which is the direct
evidence for or against that collapse.

Run:  python scripts/00f_collapse_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.zgrad_common import small_cfg, train_agent  # noqa: E402
from src.baselines import RobustBaseline  # noqa: E402
from src.common import seed_everything, set_threads  # noqa: E402
from src.env import RegimeDriftEpidemicEnv, make_drift  # noqa: E402
from src.gating import online_rollout  # noqa: E402


def action_fracs(actions: np.ndarray, onset: np.ndarray):
    """Return (frac pre, frac post) action frequencies, each a length-3 vector."""
    H, B = actions.shape
    t = np.arange(H)[:, None]
    pre = actions[t < onset[None, :]]
    post = actions[t >= onset[None, :]]
    def fr(a):
        if a.size == 0:
            return np.zeros(3)
        return np.array([(a == k).mean() for k in (0, 1, 2)])
    return fr(pre), fr(post)


def main():
    set_threads(4)
    family = "action_revealed"
    cfg = small_cfg(family)
    H = cfg["horizon"]
    seed = 10
    seed_everything(seed)

    agents = {}
    agents["meta_online_grad"] = train_agent(cfg, seed, "online", True)
    agents["meta_nograd"] = train_agent(cfg, seed, "online", False)
    agents["meta_oracle"] = train_agent(cfg, seed + 2, "oracle", False)
    robust = RobustBaseline(cfg)
    robust.fit(seed)
    agents["robust"] = robust.agent

    env = RegimeDriftEpidemicEnv(n_envs=8, horizon=H, seed=seed + 777)
    rng = np.random.default_rng(seed + 31337)
    for kind in ("stationary", "endogenous_beh", "abrupt_eps"):
        drift = make_drift(rng, kind, env.n_envs, H, strength=1.0, family=family)
        print(f"\n=== action_revealed : {kind} ===")
        print(f"  {'agent':<18}{'pre a0/a1/a2':>22}{'post a0/a1/a2':>22}")
        for name, ag in agents.items():
            if name == "robust":
                from src.baselines import FixedController, rollout_episode
                roll = online_rollout(agents["meta_online_grad"], env, drift, seed + 11,
                                      deterministic=True, z_mode="online")
                # robust actions via its own fixed policy
                obs = env.reset(drift["regime0"], regime1=drift["regime1"], onset=drift["onset"],
                                duration=drift["duration"], abrupt=drift["abrupt"], seed=seed + 11)
                acts = np.zeros((H, env.n_envs), dtype=int)
                for t in range(H):
                    a = ag.act(obs, deterministic=True)
                    acts[t] = a
                    obs, _, _, _ = env.step(a)
            else:
                zm = ag.cfg["z_mode"]
                roll = online_rollout(ag, env, drift, seed + 11, deterministic=True, z_mode=zm)
                acts = roll["actions"]
            p, q = action_fracs(acts, drift["onset"])
            print(f"  {name:<18}{p[0]:>7.2f}/{p[1]:.2f}/{p[2]:.2f}{q[0]:>10.2f}/{q[1]:.2f}/{q[2]:.2f}")
        # oracle-best constant action for this cell
        from src.baselines import ConstantController, rollout_episode, eval_metrics
        cret = {}
        for a in (0, 1, 2):
            rew = rollout_episode(env, ConstantController(a), drift, seed + 60 + a)
            cret[a] = eval_metrics(rew, drift["onset"], H)["return_mean"]
        best = max(cret, key=cret.get)
        print(f"  constant-action returns a0={cret[0]:.1f} a1={cret[1]:.1f} a2={cret[2]:.1f}  "
              f"-> best a*={best}")


if __name__ == "__main__":
    main()
