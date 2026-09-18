"""Step-1 validation of fix (B): policy conditioned on the posterior mean with a
gradient path from the PPO objective back into the context encoder.

Hypothesis under test
---------------------
In the ``action_revealed`` family the decision-relevant parameters enter the
dynamics only through the action term. A context model that is trained *only*
by the unsupervised decoder objective cannot tell those parameters apart, so a
naive online meta-agent should not beat the non-adapting Robust baseline. Fix
(B) adds the control objective as a second training signal for the encoder
(the policy is differentiated through the posterior mean ``mu``), which should,
if the premise holds, let ``action_revealed`` show separation while the
``passive`` family acts as the control (it is already identifiable without
acting, so ``z_grad`` should matter much less there).

What this script does (small scale, ~10 min)
--------------------------------------------
For each family it trains, from the *same* seed so that only the switch differs:
  * meta_online  with z_grad=True   (fix B on)
  * meta_online  with z_grad=False  (fix B off)
  * meta_frozen  (single inference, z_grad=True)
  * meta_oracle  (true regime; upper bound)
plus the Robust baseline, and evaluates all of them (and the three constant
actions) on a few drift kinds. Prints real numbers and a per-cell verdict.

Run:  python scripts/00e_zgrad_validation.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines import (ConstantController, RobustBaseline, eval_metrics,  # noqa: E402
                           rollout_episode)
from src.common import seed_everything, set_threads  # noqa: E402
from src.env import (FAMILIES, RegimeDriftEpidemicEnv,  # noqa: E402
                     make_drift)
from src.gating import online_rollout  # noqa: E402
from src.meta_rl import MetaRLAgent, train_meta  # noqa: E402
from scripts.zgrad_common import small_cfg  # noqa: E402


def _eval_meta(agent, env, drift, seed, z_mode):
    roll = online_rollout(agent, env, drift, seed, deterministic=True, z_mode=z_mode)
    return eval_metrics(roll["rew"], drift["onset"], env.horizon)


def run_family(family: str, seed: int) -> dict:
    cfg = small_cfg(family)
    H = cfg["horizon"]
    env = RegimeDriftEpidemicEnv(n_envs=8, horizon=H, seed=seed + 777)

    agents = {}
    t0 = time.perf_counter()
    agents["meta_online"] = MetaRLAgent(dict(cfg, z_mode="online", z_grad=True))
    train_meta(agents["meta_online"], seed, verbose=False)
    agents["meta_nograd"] = MetaRLAgent(dict(cfg, z_mode="online", z_grad=False))
    train_meta(agents["meta_nograd"], seed, verbose=False)
    agents["meta_frozen"] = MetaRLAgent(dict(cfg, z_mode="frozen", z_grad=True))
    train_meta(agents["meta_frozen"], seed + 1, verbose=False)
    agents["meta_oracle"] = MetaRLAgent(dict(cfg, z_mode="oracle", z_grad=False))
    train_meta(agents["meta_oracle"], seed + 2, verbose=False)
    robust = RobustBaseline(cfg)
    robust.fit(seed)
    train_wall = time.perf_counter() - t0

    kinds = ["stationary", "gradual_beta", "endogenous_beh", "abrupt_eps"]
    rng = np.random.default_rng(seed + 31337)
    out = {"family": family, "train_wall_seconds": train_wall, "kinds": {}}
    for kind in kinds:
        drift = make_drift(rng, kind, env.n_envs, H, strength=1.0, family=family)
        row = {}
        for name, ag in agents.items():
            zm = ag.cfg["z_mode"]
            row[name] = _eval_meta(ag, env, drift, seed + 11, zm)
        row["robust"] = robust.evaluate(drift, env, seed + 21)
        for a in (0, 1, 2):
            rew = rollout_episode(env, ConstantController(a), drift, seed + 60 + a)
            row[f"const{a}"] = eval_metrics(rew, drift["onset"], H)
        out["kinds"][kind] = row
    return out


def _fmt(v):
    return f"{v:>9.2f}"


def report(rec: dict) -> None:
    family = rec["family"]
    print("=" * 78)
    print(f"FAMILY = {family}   (train wall {rec['train_wall_seconds']:.1f}s)")
    print("=" * 78)
    names = ["meta_online", "meta_nograd", "meta_frozen", "meta_oracle",
             "robust", "const0", "const1", "const2"]
    for kind, row in rec["kinds"].items():
        best_ref = max(max(row["robust"]["return_mean"],
                           row[f"const{a}"]["return_mean"]) for a in (0, 1, 2))
        print(f"\n[{family}:{kind}]  best(robust,const) = {best_ref:.2f}")
        print(f"  {'agent':<13}{'return':>9}{'post':>9}")
        for n in names:
            r = row[n]
            tag = ""
            if n in ("meta_online", "meta_nograd"):
                tag = "  <- beats best ref" if r["return_mean"] > best_ref else ""
            print(f"  {n:<13}{_fmt(r['return_mean'])}{_fmt(r['post_return_mean'])}{tag}")
        on = row["meta_online"]["return_mean"]
        ng = row["meta_nograd"]["return_mean"]
        print(f"  VERDICT: meta_online(grad) - meta_nograd = {on - ng:+.2f}"
              f" | meta_online - best_ref = {on - best_ref:+.2f}")


def main():
    set_threads(4)
    seeds = [10]
    results = {}
    for fam in FAMILIES:
        recs = []
        for s in seeds:
            seed_everything(s)
            recs.append(run_family(fam, s))
        results[fam] = recs
        for rec in recs:
            report(rec)
    outdir = ROOT / "results" / "validation"
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "zgrad_validation.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
