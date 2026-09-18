"""Diagnostic ladder D1-D4: can ANY learner capture the regime-identification gap?

Context
-------
An independent measurement (scripts/validate_achievable.py, 96 tasks,
action_revealed, stationary) found:
    best single constant action      -164.37
    best per-task constant action    -134.65   (+29.7 from regime identification)
    best state-dependent threshold   -130.56   (+4.1 on top of the above)
So ~30 points are available and almost all of it comes from deciding, per task,
whether intervening helps at all -- i.e. the problem is regime identification.
This ladder asks whether our learner can obtain any of those 30 points.

Stages (run in order; D2/D3 only if D1 succeeds)
-----------------------------------------------
D1  1 task, TRUE regime appended to the observation, long training.
    Q: does the action distribution converge to that task's optimal constant?
D2  8 tasks, TRUE regime visible.
    Q: does one policy differentiate the tasks (different tasks -> different actions)?
D3  8 tasks, truth HIDDEN (the paper's setting) using the fixed z channel.
    Q: does online inference recover D2's discrimination?
D4  Only if D1 fails: one probe at a time (longer training / gamma / entropy /
    reward scale / minibatch), reporting which one unlocks learning.

This script never edits the frozen environment; it augments observations in a thin
wrapper used only by the diagnostic.

Run:  python scripts/00g_diag_ladder.py --stage d1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines import ConstantController, rollout_episode  # noqa: E402
from src.common import as_t, seed_everything, set_threads  # noqa: E402
from src.env import (BASE_REGIME, FAMILY_ACTION_REVEALED, IDX_BEH, IDX_EPS,  # noqa: E402
                     OBS_DIM, REGIME_DIM, RegimeDriftEpidemicEnv, sample_regimes)
from src.meta_rl import mlp  # noqa: E402

ACTIONS = (0, 1, 2)
BIG_ONSET = 10 ** 6
DEV = torch.device("cpu")


# ---------------------------------------------------------------------- #
# observation augmentation (diagnostic only; frozen env untouched)
# ---------------------------------------------------------------------- #
def augment(obs: np.ndarray, regime: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return obs
    if mode == "sign":
        s = np.sign(regime[:, IDX_BEH] - regime[:, IDX_EPS])[:, None].astype(np.float32)
        return np.concatenate([obs.astype(np.float32), s], axis=1)
    if mode == "regime":
        return np.concatenate([obs.astype(np.float32), regime.astype(np.float32)], axis=1)
    raise ValueError(f"unknown augment mode: {mode}")


def aug_dim(mode: str) -> int:
    return {"none": OBS_DIM, "sign": OBS_DIM + 1, "regime": OBS_DIM + REGIME_DIM}[mode]


def stationary_drift(regimes: np.ndarray) -> dict:
    B = regimes.shape[0]
    return {"regime0": regimes.copy(), "regime1": regimes.copy(),
            "onset": np.full(B, BIG_ONSET), "duration": np.ones(B, dtype=int),
            "abrupt": np.zeros(B, dtype=bool)}


# ---------------------------------------------------------------------- #
# a minimal, self-contained PPO (configurable obs_dim and reward scale)
# ---------------------------------------------------------------------- #
DEFAULT_DIAG = {
    "horizon": 156, "iterations": 800, "hidden": 64, "lr": 3e-4, "gamma": 0.99,
    "lam": 0.95, "clip": 0.2, "ppo_epochs": 4, "n_minibatch": 4, "value_coef": 0.5,
    "entropy_coef": 0.01, "reward_scale": 1.0, "max_grad_norm": 0.5,
}


class TinyPPO:
    def __init__(self, obs_dim: int, cfg: dict):
        self.cfg = cfg
        h = cfg["hidden"]
        self.policy = mlp([obs_dim, h, h, 3])
        self.value = mlp([obs_dim, h, h, 1])
        self.opt = torch.optim.Adam(list(self.policy.parameters())
                                    + list(self.value.parameters()), lr=cfg["lr"])

    @torch.no_grad()
    def collect(self, env, regimes, mode):
        c = self.cfg
        H, B = c["horizon"], env.n_envs
        obs = env.reset(regimes)
        O = np.zeros((H, B, aug_dim(mode)), np.float32)
        A = np.zeros((H, B), np.int64)
        LP = np.zeros((H, B), np.float32)
        RS = np.zeros((H, B), np.float32)
        D = np.zeros((H, B), np.float32)
        V = np.zeros((H, B), np.float32)
        for t in range(H):
            o = augment(obs, env.current_regime(), mode)
            o_t = as_t(o)
            dist = Categorical(logits=self.policy(o_t))
            a = dist.sample()
            lp = dist.log_prob(a)
            v = self.value(o_t).squeeze(-1)
            nobs, raw, done, _ = env.step(a.cpu().numpy())
            O[t], A[t] = o, a.cpu().numpy()
            LP[t], V[t] = lp.cpu().numpy(), v.cpu().numpy()
            RS[t], D[t] = raw * c["reward_scale"], done.astype(np.float32)
            obs = nobs
        return {"obs": torch.as_tensor(O), "act": torch.as_tensor(A),
                "logp": torch.as_tensor(LP), "rew": torch.as_tensor(RS),
                "done": torch.as_tensor(D), "val": torch.as_tensor(V)}

    def update(self, roll):
        c = self.cfg
        obs, act, logp_old = roll["obs"], roll["act"], roll["logp"]
        rew, done, val = roll["rew"], roll["done"], roll["val"]
        H, B = rew.shape
        adv = torch.zeros_like(rew)
        last = torch.zeros(B)
        for t in reversed(range(H)):
            nv = val[t + 1] if t + 1 < H else torch.zeros(B)
            nt = 1.0 - done[t]
            delta = rew[t] + c["gamma"] * nv * nt - val[t]
            last = delta + c["gamma"] * c["lam"] * nt * last
            adv[t] = last
        ret = adv + val
        f_obs = obs.reshape(H * B, -1)
        f_act = act.reshape(H * B)
        f_logp = logp_old.reshape(H * B)
        f_ret = ret.reshape(H * B)
        f_adv = adv.reshape(H * B)
        adv_mean, adv_std = float(f_adv.mean()), float(f_adv.std())
        f_adv = (f_adv - f_adv.mean()) / (f_adv.std() + 1e-8)
        N = H * B
        mb = max(1, N // c["n_minibatch"])
        pg_l = vf_l = ent_l = 0.0
        nb = 0
        for _ in range(c["ppo_epochs"]):
            perm = torch.randperm(N)
            for s in range(0, N, mb):
                idx = perm[s:s + mb]
                dist = Categorical(logits=self.policy(f_obs[idx]))
                ratio = torch.exp(dist.log_prob(f_act[idx]) - f_logp[idx])
                pg = -torch.min(ratio * f_adv[idx],
                                torch.clamp(ratio, 1 - c["clip"], 1 + c["clip"]) * f_adv[idx]).mean()
                vf = ((self.value(f_obs[idx]).squeeze(-1) - f_ret[idx]) ** 2).mean()
                ent = dist.entropy().mean()
                loss = pg + c["value_coef"] * vf - c["entropy_coef"] * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.policy.parameters())
                                         + list(self.value.parameters()), c["max_grad_norm"])
                self.opt.step()
                pg_l += float(pg.detach()); vf_l += float(vf.detach()); ent_l += float(ent.detach())
                nb += 1
        return {"pg": pg_l / max(1, nb), "vf": vf_l / max(1, nb), "ent": ent_l / max(1, nb),
                "adv_mean": adv_mean, "adv_std": adv_std,
                "rew_mean": float(rew.mean()), "rew_std": float(rew.std())}

    @torch.no_grad()
    def evaluate(self, env, regimes, mode, seed):
        """Deterministic rollout; return (counts, dominant-per-env, mean_return)."""
        H, B = self.cfg["horizon"], env.n_envs
        obs = env.reset(regimes, seed=seed)
        acts = np.zeros((H, B), np.int64)
        ret = 0.0
        for t in range(H):
            o = augment(obs, env.current_regime(), mode)
            a = self.policy(as_t(o)).argmax(-1).cpu().numpy()
            acts[t] = a
            obs, r, done, _ = env.step(a)
            ret += float(r.mean())
        counts = np.array([(acts == k).mean() for k in ACTIONS])
        dom = np.array([np.bincount(acts[:, i], minlength=3).argmax() for i in range(B)])
        return counts, dom, ret


# ---------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------- #
def const_returns_per_task(regimes, horizon, seed):
    """(B,3) matrix of per-task constant-action returns."""
    B = regimes.shape[0]
    env = RegimeDriftEpidemicEnv(n_envs=B, horizon=horizon, seed=seed)
    drift = stationary_drift(regimes)
    out = np.zeros((B, 3))
    for k, a in enumerate(ACTIONS):
        rew = rollout_episode(env, ConstantController(a), drift, seed + 11 + k)
        out[:, k] = rew.sum(0)
    return out


def train(cfg, obs_dim, env, regimes, mode, seed, reward_scale=None):
    seed_everything(seed)
    cfg = dict(cfg)
    if reward_scale is not None:
        cfg["reward_scale"] = reward_scale
    ag = TinyPPO(obs_dim, cfg)
    t0 = time.perf_counter()
    last = None
    for _ in range(cfg["iterations"]):
        roll = ag.collect(env, regimes, mode)
        last = ag.update(roll)
    wall = time.perf_counter() - t0
    return ag, wall, last


# ---------------------------------------------------------------------- #
# D1
# ---------------------------------------------------------------------- #
def d1(cfg, seed):
    horizon = cfg["horizon"]
    tasks = {}
    r_h = BASE_REGIME.copy(); r_h[IDX_BEH] = 0.30; r_h[IDX_EPS] = 0.10   # beh>eps -> a*=0
    r_b = BASE_REGIME.copy(); r_b[IDX_BEH] = 0.05; r_b[IDX_EPS] = 0.50   # eps>beh -> a*=2
    tasks["harmful_beh_gt_eps"] = r_h
    tasks["beneficial_eps_gt_beh"] = r_b
    n_rep = cfg["n_envs"]
    env = RegimeDriftEpidemicEnv(n_envs=n_rep, horizon=horizon, seed=seed)
    # (label, augment mode, reward_scale): raw rewards vs the pipeline's 0.05,
    # so we can test whether reward scaling alone disables learning.
    variants = [("none", "none", 1.0), ("sign", "sign", 1.0), ("regime", "regime", 1.0),
                ("sign_scale0.05", "sign", 0.05), ("regime_scale0.05", "regime", 0.05)]
    out = {"stage": "D1", "cfg": cfg, "tasks": {}}
    for tname, r in tasks.items():
        regimes = np.tile(r, (n_rep, 1))
        cr = const_returns_per_task(np.tile(r, (1, 1)), horizon, seed).ravel()
        astar = int(np.argmax(cr))
        rec = {"const_returns": {str(a): float(cr[a]) for a in ACTIONS},
               "a_star": astar, "modes": {}}
        for label, mode, rscale in variants:
            ag, wall, last = train(cfg, aug_dim(mode), env, regimes, mode, seed,
                                   reward_scale=rscale)
            counts, dom, ret = ag.evaluate(env, regimes, mode, seed + 7)
            rec["modes"][label] = {
                "augment": mode, "reward_scale": rscale,
                "action_counts": [float(c) for c in counts],
                "dominant": int(np.bincount(dom, minlength=3).argmax()),
                "return": float(ret),
                "gap_to_const": float(cr[astar] - ret),
                "wall_seconds": wall, "last_loss": last}
        out["tasks"][tname] = rec
    return out


def report_d1(rec):
    print("=" * 80)
    print("D1: single task, TRUE regime appended; raw rewards vs pipeline scale=0.05")
    print("=" * 80)
    ok_any = False
    for tname, r in rec["tasks"].items():
        cr = r["const_returns"]
        print(f"\n[{tname}]  a*={r['a_star']}  const a0={cr['0']:.1f} a1={cr['1']:.1f} a2={cr['2']:.1f}")
        print(f"  {'variant':<18}{'a0/a1/a2':>20}{'dom':>5}{'return':>10}{'gap':>8}{'adv_sd':>9}")
        for label, m in r["modes"].items():
            c = m["action_counts"]
            print(f"  {label:<18}{c[0]:>6.2f}/{c[1]:.2f}/{c[2]:.2f}{m['dominant']:>5}"
                  f"{m['return']:>10.1f}{m['gap_to_const']:>8.1f}{m['last_loss']['adv_std']:>9.3f}")
        for label in ("sign", "regime"):
            m = r["modes"][label]
            ok_any = ok_any or (m["dominant"] == r["a_star"] and m["gap_to_const"] < 10)
    print(f"\nD1 VERDICT: {'PASS' if ok_any else 'FAIL'}")
    return ok_any


# ---------------------------------------------------------------------- #
# D2
# ---------------------------------------------------------------------- #
def d2(cfg, seed):
    horizon = cfg["horizon"]
    rng = np.random.default_rng(seed)
    B = cfg["n_tasks"]
    regimes = sample_regimes(rng, B, family=FAMILY_ACTION_REVEALED)
    env = RegimeDriftEpidemicEnv(n_envs=B, horizon=horizon, seed=seed)
    cr = const_returns_per_task(regimes, horizon, seed)
    astar = cr.argmax(1)
    out = {"stage": "D2", "cfg": cfg, "a_star": [int(a) for a in astar],
           "best_single_const": float(cr.mean(0).max()),
           "best_per_task_const": float(cr.max(1).mean()), "modes": {}}
    variants = [("none", "none", 1.0), ("sign", "sign", 1.0), ("regime", "regime", 1.0),
                ("sign_scale0.05", "sign", 0.05)]
    for label, mode, rscale in variants:
        ag, wall, last = train(cfg, aug_dim(mode), env, regimes, mode, seed,
                               reward_scale=rscale)
        counts, dom, ret = ag.evaluate(env, regimes, mode, seed + 7)
        agree = float((dom == astar).mean())
        out["modes"][label] = {
            "augment": mode, "reward_scale": rscale,
            "action_counts": [float(c) for c in counts],
            "per_task_dominant": [int(d) for d in dom],
            "per_task_correct": float(np.mean(dom == astar)),
            "return_mean": float(ret), "agreement": agree,
            "distinct_dominant": int(len(set(dom.tolist()))),
            "wall_seconds": wall, "last_loss": last}
    return out


def report_d2(rec):
    print("\n" + "=" * 74)
    print("D2: 8 tasks, TRUE regime visible")
    print("=" * 74)
    print(f"best single const = {rec['best_single_const']:.1f} | "
          f"per-task optimal = {rec['best_per_task_const']:.1f}")
    print(f"a* per task = {rec['a_star']}")
    for label, m in rec["modes"].items():
        print(f"[{label}] dominant={m['per_task_dominant']} distinct={m['distinct_dominant']} "
              f"correct={m['per_task_correct']:.2f} return={m['return_mean']:.1f}")
    return rec


# ---------------------------------------------------------------------- #
# D3
# ---------------------------------------------------------------------- #
def d3(cfg, seed):
    """8 tasks, truth HIDDEN, using the paper's z channel (online inference)."""
    from src.gating import online_rollout
    from src.meta_rl import MetaRLAgent, train_meta
    horizon, B = cfg["horizon"], cfg["n_tasks"]
    rng = np.random.default_rng(seed)
    regimes = sample_regimes(rng, B, family=FAMILY_ACTION_REVEALED)
    cr = const_returns_per_task(regimes, horizon, seed)
    astar = cr.argmax(1)
    out = {"stage": "D3", "cfg": cfg, "a_star": [int(a) for a in astar],
           "best_single_const": float(cr.mean(0).max()),
           "best_per_task_const": float(cr.max(1).mean()), "variants": {}}
    mcfg = dict(cfg)
    mcfg.update(n_tasks=B, family=FAMILY_ACTION_REVEALED, z_mode="online",
                ctx_len=12, z_dim=5)
    env = RegimeDriftEpidemicEnv(n_envs=B, horizon=horizon, seed=seed)
    drift = stationary_drift(regimes)
    for label, z_mode, z_grad in (("online_grad", "online", True),
                                  ("online_nograd", "online", False),
                                  ("frozen_grad", "frozen", True)):
        agent = MetaRLAgent(dict(mcfg, z_mode=z_mode, z_grad=z_grad))
        t0 = time.perf_counter()
        train_meta(agent, seed, verbose=False)
        wall = time.perf_counter() - t0
        roll = online_rollout(agent, env, drift, seed + 7, deterministic=True, z_mode=z_mode)
        acts = roll["actions"]
        counts = [float((acts == k).mean()) for k in ACTIONS]
        dom = [int(np.bincount(acts[:, i], minlength=3).argmax()) for i in range(B)]
        ret = float(roll["rew"].sum(0).mean())
        out["variants"][label] = {
            "action_counts": counts, "per_task_dominant": dom,
            "per_task_correct": float(np.mean(np.array(dom) == astar)),
            "distinct_dominant": int(len(set(dom))), "return_mean": ret,
            "wall_seconds": wall}
    return out


def report_d3(rec):
    print("\n" + "=" * 80)
    print("D3: 8 tasks, truth HIDDEN, z channel (paper's setting)")
    print("=" * 80)
    print(f"best single const = {rec['best_single_const']:.1f} | "
          f"per-task optimal = {rec['best_per_task_const']:.1f} | a* = {rec['a_star']}")
    for label, m in rec["variants"].items():
        print(f"[{label}] dominant={m['per_task_dominant']} distinct={m['distinct_dominant']} "
              f"correct={m['per_task_correct']:.2f} return={m['return_mean']:.1f}")
    return rec


# ---------------------------------------------------------------------- #
# D4
# ---------------------------------------------------------------------- #
def d4(cfg, seed):
    """One-probe-at-a-time optimization checks on a task whose optimum is a=0."""
    horizon, B = cfg["horizon"], cfg["n_envs"]
    r = BASE_REGIME.copy(); r[IDX_BEH] = 0.30; r[IDX_EPS] = 0.10
    regimes = np.tile(r, (B, 1))
    cr = const_returns_per_task(np.tile(r, (1, 1)), horizon, seed).ravel()
    astar = int(cr.argmax())
    env = RegimeDriftEpidemicEnv(n_envs=B, horizon=horizon, seed=seed)
    probes = {
        "baseline": {},
        "longer_iters": {"iterations": 1200},
        "gamma_0.995": {"gamma": 0.995},
        "entropy_0.05": {"entropy_coef": 0.05},
        "minibatch_2": {"n_minibatch": 2},
        "epochs_8": {"ppo_epochs": 8},
        "scale_0.05": {"reward_scale": 0.05},
    }
    out = {"stage": "D4", "task": "harmful_beh_gt_eps", "a_star": astar,
           "const_returns": {str(a): float(cr[a]) for a in ACTIONS}, "probes": {}}
    for name, ov in probes.items():
        c = dict(cfg); c.update(ov)
        ag, wall, last = train(c, aug_dim("sign"), env, regimes, "sign", seed,
                               reward_scale=c.get("reward_scale", 1.0))
        counts, dom, ret = ag.evaluate(env, regimes, "sign", seed + 7)
        out["probes"][name] = {
            "action_counts": [float(x) for x in counts], "dominant": int(dom[0]),
            "return": float(ret), "gap_to_const": float(cr[astar] - ret),
            "adv_std": last["adv_std"], "wall_seconds": wall}
    return out


def report_d4(rec):
    print("\n" + "=" * 80)
    print("D4: one-probe-at-a-time optimization checks (task: optimum a=0)")
    print("=" * 80)
    cr = rec["const_returns"]
    print(f"a*={rec['a_star']}  const a0={cr['0']:.1f} a1={cr['1']:.1f} a2={cr['2']:.1f}")
    print(f"  {'probe':<16}{'a0/a1/a2':>20}{'dom':>5}{'return':>10}{'gap':>8}{'adv_sd':>9}")
    for name, m in rec["probes"].items():
        c = m["action_counts"]
        print(f"  {name:<16}{c[0]:>6.2f}/{c[1]:.2f}/{c[2]:.2f}{m['dominant']:>5}"
              f"{m['return']:>10.1f}{m['gap_to_const']:>8.1f}{m['adv_std']:>9.3f}")
    return rec


# ---------------------------------------------------------------------- #
def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="d1", choices=["d1", "d2", "d3", "d4", "all"])
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args(argv)
    set_threads(args.threads)

    # Mirror stdout to artifacts/diag_ladder_<stage>.txt so a run always leaves a
    # log on disk, even if the caller forgets to tee. Unbuffered (python -u).
    art = ROOT / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    logf = open(art / f"diag_ladder_{args.stage}.txt", "w")

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

    cfg = dict(DEFAULT_DIAG)
    cfg.update(horizon=156, iterations=400, hidden=64, n_envs=4, n_tasks=8)

    results = {}
    if args.stage in ("d1", "all"):
        d1rec = d1(cfg, args.seed)
        report_d1(d1rec)
        results["d1"] = d1rec
    if args.stage in ("d2", "all"):
        d2rec = d2(cfg, args.seed)
        report_d2(d2rec)
        results["d2"] = d2rec
    if args.stage in ("d3", "all"):
        d3rec = d3(cfg, args.seed)
        report_d3(d3rec)
        results["d3"] = d3rec
    if args.stage == "d4":
        d4rec = d4(cfg, args.seed)
        report_d4(d4rec)
        results["d4"] = d4rec

    outdir = ROOT / "results" / "validation"
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"diag_ladder_{args.stage}.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {path}")
    logf.flush()


if __name__ == "__main__":
    main()
