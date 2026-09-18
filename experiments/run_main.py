"""Unified entry point for the regime-drift meta-RL experiments.

Runs, for each seed, the full agent suite over BOTH environment families
(``passive`` and ``action_revealed``), all drift kinds, and both drift
strengths, plus the gating diagnostics and the Passive Identifiability Index.

Example
-------
    python -m experiments.run_main --seeds 10,11,12 --outdir results/main

Writes one JSON per seed under ``<outdir>/seed_<s>.json``. All randomness is
explicitly seeded; the thread count and host provenance are recorded.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.baselines import (ADAPTATION_STEPS_DEFINITION, AGENT_NAMES,
                           ConstantController, RetrainBaseline, RobustBaseline,
                           TransferBaseline, attach_captured,
                           constant_baseline_block, eval_metrics, rollout_episode)
from src.commitment import wrap_cfg
from src.common import (DEFAULT_CONFIG, provenance, seed_everything, set_threads)
from src.env import (BASE_REGIME, DRIFT_KINDS, FAMILIES, FAMILY_PASSIVE,
                     RegimeDriftEpidemicEnv, make_drift)
from src.gating import (compute_adaptation_labels, online_rollout, evaluate_gate,
                        calibrate_residual_scale)
from src.identifiability import PII_DEFINITION, passive_identifiability_index
from src.meta_rl import MetaRLAgent, train_meta

RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"

DEFINITIONS = {
    "family_passive": ("eps and beh are fixed constants; the whole regime is "
                       "reflected in the passive (a=0) case trajectory, so it is "
                       "identifiable without intervening."),
    "family_action_revealed": ("eps and beh vary widely and enter the dynamics only "
                               "through the action term; two such tasks have identical "
                               "observation distributions until the agent intervenes."),
    "return": ("return = sum of RAW per-step rewards (reward = -(burden + 0.25*a), "
               "burden = I / 0.02) over the horizon; no reward_scale applied."),
    "post_return": ("post_return = same as return but summed only over steps t >= "
                    "drift onset (the window where adaptation is visible)."),
    "adaptation_steps": ADAPTATION_STEPS_DEFINITION,
    "regret": ("regret = (best agent return_mean within this family/kind/strength/"
               "seed) - (agent return_mean), best over all reported agents incl. "
               "constant-action references. regret_post uses post_return_mean."),
    "regret_vs_oracle": ("regret_vs_oracle = (meta_oracle return_mean) - (agent "
                         "return_mean); the _post suffix uses post_return_mean. "
                         "null when the oracle agent was not trained."),
    "adaptation_harmful": ("Ground-truth gate label. For each test episode and step t, "
                           "the environment state is cloned (snapshot/restore) and two "
                           "branches are rolled to the horizon with independent seeds: "
                           "(a) online meta-policy, (b) non-adapting Robust baseline. "
                           "labels[t] = 1 iff remaining return of (a) < (b)."),
    "pii": PII_DEFINITION,
}


def build_cfg(args, family: str) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(n_tasks=args.n_tasks, horizon=args.horizon, iterations=args.iterations,
               z_dim=args.z_dim, ctx_len=args.ctx_len, hidden=args.hidden,
               ppo_epochs=args.ppo_epochs, n_minibatch=args.n_minibatch,
               z_grad=args.z_grad, family=family,
               commitment=args.commitment,
               commitment_cycle_len=(args.commitment_cycle_len or None),
               commitment_warmup=args.commitment_warmup)
    return cfg


def kind_key(kind: str, strength: float) -> str:
    return kind if abs(strength - 1.0) < 1e-9 else f"{kind}_ood"


def _meta_metrics(agent, eval_env, drift, seed, z_mode):
    roll = online_rollout(agent, eval_env, drift, seed, deterministic=True, z_mode=z_mode)
    return eval_metrics(roll["rew"], drift["onset"], eval_env.horizon)


def _run_gate(agent, robust_agent, eval_env, kinds, strength, family, seed, cfg, args):
    K = cfg["ctx_len"]
    scale_env = RegimeDriftEpidemicEnv(n_envs=eval_env.n_envs, horizon=cfg["horizon"], seed=seed + 1)
    scale_env = wrap_cfg(scale_env, cfg)
    calib = calibrate_residual_scale(agent, scale_env, np.tile(BASE_REGIME, (eval_env.n_envs, 1)),
                                     n_iters=2, seed=seed)
    sc = {"consistency": [], "entropy": [], "variance": []}
    y_all, grp_all = [], []
    rng = np.random.default_rng(seed + 424242)
    H = eval_env.horizon
    for ki, kind in enumerate(kinds):
        drift = make_drift(rng, kind, eval_env.n_envs, H, strength=strength, family=family)
        roll = online_rollout(agent, eval_env, drift, seed + 900 + ki, deterministic=True,
                              record=True, resid_scale=calib)
        labels, _rm, _rr = compute_adaptation_labels(
            roll, agent, robust_agent, eval_env.n_envs, H, seed + 7000 + ki, stride=args.label_stride)
        tt, ee = np.meshgrid(np.arange(H), np.arange(eval_env.n_envs), indexing="ij")
        valid = (tt >= K) & np.isfinite(roll["residual"])
        for name, arr in (("consistency", roll["residual"]), ("entropy", roll["entropy"]),
                          ("variance", roll["pvariance"])):
            sc[name].append(arr[valid])
        y_all.append(labels[valid])
        grp_all.append(ee[valid] + ki * 1000)
    labels_all = np.concatenate(y_all).astype(int)
    groups_all = np.concatenate(grp_all)
    out = {}
    for name in ("consistency", "entropy", "variance"):
        s = np.concatenate(sc[name])
        out[name] = evaluate_gate(s, labels_all, groups=groups_all, n_boot=args.gate_boot, seed=seed)
    return {"family": family,
            "auroc_consistency": out["consistency"]["auroc"],
            "auroc_entropy": out["entropy"]["auroc"],
            "auroc_variance": out["variance"]["auroc"],
            "pr_auc_consistency": out["consistency"]["pr_auc"],
            "pos_rate": out["consistency"]["pos_rate"],
            "n_steps": out["consistency"]["n"],
            "calib_scale": float(calib),
            "ci": {"consistency": out["consistency"]["ci"],
                   "entropy": out["entropy"]["ci"],
                   "variance": out["variance"]["ci"]}}


def run_family(family: str, seed: int, args) -> dict:
    cfg = build_cfg(args, family)
    H = cfg["horizon"]
    n_eval = args.tasks_per_kind

    agent = MetaRLAgent(dict(cfg, z_mode="online"))
    train_info = train_meta(agent, seed, log_every=args.log_every if args.verbose else 0,
                            verbose=args.verbose)
    agent_frozen = None
    if args.with_frozen:
        agent_frozen = MetaRLAgent(dict(cfg, z_mode="frozen"))
        train_meta(agent_frozen, seed + 1, verbose=False)
    agent_oracle = None
    if args.with_oracle:
        agent_oracle = MetaRLAgent(dict(cfg, z_mode="oracle"))
        train_meta(agent_oracle, seed + 2, verbose=False)

    robust = RobustBaseline(cfg)
    robust.fit(seed)
    transfer = TransferBaseline(cfg, finetune_steps=args.finetune_steps)
    transfer.fit(seed)
    retrain = RetrainBaseline(cfg, budget_steps=args.retrain_budget, n_replicas=8)

    eval_env = RegimeDriftEpidemicEnv(n_envs=n_eval, horizon=H, seed=seed + 777)
    eval_env = wrap_cfg(eval_env, cfg)
    strengths = [float(s) for s in args.strengths.split(",")]
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    gate_kinds = [k.strip() for k in args.gate_kinds.split(",") if k.strip()]

    eval_out = {}
    rng = np.random.default_rng(seed + 31337)
    for strength in strengths:
        for kind in kinds:
            if kind == "stationary" and abs(strength - 1.0) > 1e-9:
                continue
            key = kind_key(kind, strength)
            drift = make_drift(rng, kind, n_eval, H, strength=strength, family=family)
            row = {}
            row["meta_online"] = _meta_metrics(agent, eval_env, drift, seed + 11, "online")
            if agent_frozen is not None:
                row["meta_frozen"] = _meta_metrics(agent_frozen, eval_env, drift, seed + 12, "frozen")
            if agent_oracle is not None:
                row["meta_oracle"] = _meta_metrics(agent_oracle, eval_env, drift, seed + 13, "oracle")
            row["robust"] = robust.evaluate(drift, eval_env, seed + 21)
            transfer.finetune(drift, eval_env, seed + 31)
            row["transfer"] = transfer.evaluate(drift, eval_env, seed + 41)
            row["transfer"]["finetune_steps"] = transfer.finetune_steps
            row["transfer"]["total_interactions"] = transfer._interactions()
            if args.with_retrain:
                row["retrain"] = retrain.fit_and_evaluate(drift, seed + 51)
                row["retrain"]["budget_per_task"] = retrain.budget_steps
                row["retrain"]["interactions_total"] = retrain._interactions()
            const_per_task = {}
            for a in (0, 1, 2):
                rew = rollout_episode(eval_env, ConstantController(a), drift, seed + 60 + a)
                row[f"const{a}"] = eval_metrics(rew, drift["onset"], H)
                const_per_task[a] = rew.sum(axis=0)

            best = max(row[n]["return_mean"] for n in AGENT_NAMES if n in row)
            best_post = max(row[n]["post_return_mean"] for n in AGENT_NAMES if n in row)
            orc = row["meta_oracle"]["return_mean"] if agent_oracle is not None else None
            orc_post = row["meta_oracle"]["post_return_mean"] if agent_oracle is not None else None
            for n in AGENT_NAMES:
                if n not in row:
                    continue
                r = row[n]
                r["regret"] = float(best - r["return_mean"])
                r["regret_post"] = float(best_post - r["post_return_mean"])
                r["regret_vs_oracle"] = float(orc - r["return_mean"]) if orc is not None else None
                r["regret_vs_oracle_post"] = float(orc_post - r["post_return_mean"]) if orc is not None else None
            # per-task constant baselines + the 'captured' primary criterion
            block = constant_baseline_block(const_per_task, n_eval)
            attach_captured(row, block)
            row["family"] = family
            row["drift_kind"] = kind
            row["strength"] = float(strength)
            row["baselines_const"] = block
            eval_out[key] = row

    gate = _run_gate(agent, robust.agent, eval_env, gate_kinds, 1.0, family, seed, cfg, args)
    pii = passive_identifiability_index(family=family, n_tasks=args.pii_tasks,
                                        horizon=min(H, 120), seed=seed)
    return {"train": {"env_steps": train_info["env_steps"],
                      "wall_seconds": train_info["wall_seconds"],
                      "final_return": train_info["final_return"],
                      "n_params": agent.n_params(),
                      "final_losses": {k: v for k, v in train_info["history"][-1].items()}},
            "eval": eval_out, "gate": gate, "pii": pii, "cfg": cfg}


def run_seed(seed: int, args) -> dict:
    set_threads(args.threads)
    seed_everything(seed)
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    per_family = {fam: run_family(fam, seed, args) for fam in families}
    return {
        "seed": seed,
        "torch_threads": int(torch.get_num_threads()),
        "families": families,
        "config": per_family[families[0]]["cfg"],
        "train": {fam: per_family[fam]["train"] for fam in families},
        "eval": {f"{fam}:{k}": per_family[fam]["eval"][k]
                 for fam in families for k in per_family[fam]["eval"]},
        "gate": {fam: per_family[fam]["gate"] for fam in families},
        "pii": {fam: per_family[fam]["pii"] for fam in families},
        "definitions": DEFINITIONS,
        "provenance": provenance(),
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Regime-drift meta-RL experiments")
    p.add_argument("--seeds", default="10,11,12,13,14,15,16,17,18,19")
    p.add_argument("--tasks-per-kind", type=int, default=12)
    p.add_argument("--families", default=",".join(FAMILIES))
    p.add_argument("--kinds", default="stationary," + ",".join(DRIFT_KINDS))
    p.add_argument("--gate-kinds", default="gradual_beta,gradual_rho,abrupt_eps")
    p.add_argument("--strengths", default="1.0,2.0")
    p.add_argument("--outdir", default=str(RESULTS_ROOT / "main"))
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--n-tasks", type=int, default=DEFAULT_CONFIG["n_tasks"])
    p.add_argument("--horizon", type=int, default=DEFAULT_CONFIG["horizon"])
    p.add_argument("--iterations", type=int, default=DEFAULT_CONFIG["iterations"])
    p.add_argument("--z-dim", type=int, default=DEFAULT_CONFIG["z_dim"])
    p.add_argument("--ctx-len", type=int, default=DEFAULT_CONFIG["ctx_len"])
    p.add_argument("--hidden", type=int, default=DEFAULT_CONFIG["hidden"])
    p.add_argument("--ppo-epochs", type=int, default=DEFAULT_CONFIG["ppo_epochs"])
    p.add_argument("--n-minibatch", type=int, default=DEFAULT_CONFIG["n_minibatch"])
    p.add_argument("--no-z-grad", dest="z_grad", action="store_false", default=True)
    p.add_argument("--commitment", action="store_true", default=False)
    p.add_argument("--commitment-cycle-len", type=int, default=0)
    p.add_argument("--commitment-warmup", type=int, default=0)
    p.add_argument("--finetune-steps", type=int, default=4000)
    p.add_argument("--retrain-budget", type=int, default=4000)
    p.add_argument("--label-stride", type=int, default=4)
    p.add_argument("--gate-boot", type=int, default=1000)
    p.add_argument("--pii-tasks", type=int, default=150)
    p.add_argument("--with-frozen", action="store_true", default=True)
    p.add_argument("--with-oracle", action="store_true", default=True)
    p.add_argument("--with-retrain", action="store_true", default=True)
    p.add_argument("--verbose", action="store_true", default=False)
    p.add_argument("--log-every", type=int, default=25)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    for s in seeds:
        t0 = time.perf_counter()
        rec = run_seed(s, args)
        out = outdir / f"seed_{s}.json"
        out.write_text(json.dumps(rec, indent=2))
        pa = rec["pii"]["passive"]
        ar = rec["pii"]["action_revealed"]
        print(f"[seed {s}] wrote {out} in {time.perf_counter() - t0:.1f}s | "
              f"PII gap passive={pa['continuous_gap']:.1f} revealed={ar['continuous_gap']:.1f} | "
              f"gate(consistency) passive={rec['gate']['passive']['auroc_consistency']:.3f} "
              f"revealed={rec['gate']['action_revealed']['auroc_consistency']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
