"""Diagnose why the robust baseline is weakest in the ``linked`` family.

Three hypotheses, tested one at a time (Assignment G):

* H1 under-training        : double the pretraining budget (2000 vs 1000 iters).
* H2 memoryless-policy limit: give robust a K=12 observation-history window
  (same 1000-iter budget) and see whether it approaches meta_frozen.
* H3 coverage trade-off    : report the per-task captured distribution, split
  by the per-task optimal constant action a* (esp. tasks where a*=0).

Everything replicates the landed main-matrix conditions exactly for the
``linked`` family: the same per-cell RNG, eval env, rollout seeds and config
(n_tasks=8, horizon=156, iterations=1000, 16 eval tasks). The meta agents are
NOT retrained; meta_frozen / meta_oracle references are read from the landed
cell JSONs. All numbers are computed from runs -- no hardcoded estimates.

Output: one JSON per (variant, seed) under results/diag_robust/, plus a
flushed progress log at artifacts/00k_progress.txt.

Run (background, 2 threads):
    python -u scripts/00k_robust_diagnosis.py --threads 2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_matrix import cell_rng  # noqa: E402
from src.baselines import (ConstantController, RobustBaseline, eval_metrics,  # noqa: E402
                           rollout_episode)
from src.commitment import wrap_cfg  # noqa: E402
from src.common import DEFAULT_CONFIG, provenance, seed_everything, set_threads  # noqa: E402
from src.env import DRIFT_KINDS, RegimeDriftEpidemicEnv, make_drift  # noqa: E402

OUTDIR = ROOT / "results" / "diag_robust"
PROGRESS = ROOT / "artifacts" / "00k_progress.txt"
KINDS = ("stationary",) + tuple(DRIFT_KINDS)
STRENGTHS = (1.0, 2.0)

VARIANTS = {
    "control": {"iterations": 1000, "history_len": 0},
    "double": {"iterations": 2000, "history_len": 0},
    "hist12": {"iterations": 1000, "history_len": 12},
}


class ActionRecorder:
    """Wrap a controller and record every action it takes."""

    def __init__(self, controller):
        self.controller = controller
        self.actions = []

    def reset(self, env=None):
        self.controller.reset(env=env)

    def act(self, obs, t, info):
        a = self.controller.act(obs, t, info)
        self.actions.append(np.asarray(a, dtype=int).copy())
        return a


def linked_cells() -> list:
    cells = []
    for kind in KINDS:
        for st in STRENGTHS:
            if kind == "stationary" and abs(st - 1.0) > 1e-9:
                continue
            cells.append((kind, float(st)))
    return cells


def const_block_for(seed: int, kind: str, strength: float, H: int, n_eval: int) -> dict:
    """Per-task constant-action returns (shared across variants; deterministic)."""
    env = wrap_cfg(RegimeDriftEpidemicEnv(n_envs=n_eval, horizon=H, seed=seed + 777),
                   dict(DEFAULT_CONFIG, family="linked"))
    drift = make_drift(cell_rng(seed, "linked", kind, strength), kind, n_eval, H,
                       strength=strength, family="linked")
    per_task = {}
    for a in (0, 1, 2):
        rew = rollout_episode(env, ConstantController(a), drift, seed + 60 + a)
        per_task[a] = rew.sum(axis=0)
    mat = np.stack([per_task[a] for a in (0, 1, 2)], axis=1)
    best_single = float(mat.mean(0).max())
    best_per_task = float(mat.max(1).mean())
    a_star = mat.argmax(1)
    return {"drift": drift, "per_task": per_task, "a_star": a_star,
            "best_single": best_single, "best_per_task": best_per_task,
            "gap": best_per_task - best_single}


def evaluate_variant(agent, seed: int, kind: str, strength: float, block: dict,
                     H: int, n_eval: int) -> dict:
    """One (variant, seed, kind, strength) cell: per-task returns + action counts."""
    env = wrap_cfg(RegimeDriftEpidemicEnv(n_envs=n_eval, horizon=H, seed=seed + 777),
                   dict(DEFAULT_CONFIG, family="linked"))
    drift = block["drift"]
    from src.baselines import FixedController  # local import keeps the top clean
    rec = ActionRecorder(FixedController(agent))
    rew = rollout_episode(env, rec, drift, seed + 21)
    total = rew.sum(axis=0)
    t_idx = np.arange(H)[:, None]
    post_mask = t_idx >= np.asarray(drift["onset"])[None, :]
    post = (rew * post_mask).sum(axis=0)
    A = np.stack(rec.actions, axis=0)  # (H, n_eval)
    counts = np.stack([(A == a).mean(axis=0) for a in (0, 1, 2)], axis=1)  # per-task share
    gap, base = block["gap"], block["best_single"]
    return {
        "kind": kind, "strength": float(strength),
        "return_per_task": [float(x) for x in total],
        "post_return_per_task": [float(x) for x in post],
        "captured_per_task": ([float((t - base) / gap) for t in total]
                              if abs(gap) > 1e-12 else None),
        "captured_post_per_task": ([float((p - base) / gap) for p in post]
                                   if abs(gap) > 1e-12 else None),
        "captured_post": float((post.mean() - base) / gap) if abs(gap) > 1e-12 else None,
        "captured": float((total.mean() - base) / gap) if abs(gap) > 1e-12 else None,
        "action_share_per_task": {str(a): [float(x) for x in counts[:, a]]
                                  for a in (0, 1, 2)},
        "a_star_per_task": [int(x) for x in block["a_star"]],
        "best_single_const": base, "best_per_task_const": block["best_per_task"],
        "available_gap": float(gap),
    }


def load_landed_reference(seed: int) -> dict:
    """meta_frozen / meta_oracle / landed-robust captured_post per cell."""
    refs = {}
    main = ROOT / "results" / "main"
    for kind, st in linked_cells():
        p = main / f"linked_{kind}_s{st:g}_seed{seed}.json"
        if not p.exists():
            continue
        rec = json.loads(p.read_text())
        ev = rec["eval"]
        refs[f"{kind}_s{st:g}"] = {
            "meta_frozen_captured_post": ev["meta_frozen"].get("captured_post"),
            "meta_oracle_captured_post": ev["meta_oracle"].get("captured_post"),
            "robust_captured_post_landed": ev["robust"].get("captured_post"),
        }
    return refs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="10,11,12,13,14")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--n-tasks", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=156)
    ap.add_argument("--tasks-per-kind", type=int, default=16)
    args = ap.parse_args()
    set_threads(args.threads)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)

    def logp(msg: str) -> None:
        with open(PROGRESS, "a") as fh:
            fh.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
            fh.flush()
        print(f"[00k] {msg}", flush=True)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    cells = linked_cells()
    H, n_eval, n_tasks = args.horizon, args.tasks_per_kind, args.n_tasks
    logp(f"start: {len(VARIANTS)} variants x {len(seeds)} seeds x {len(cells)} linked cells | "
         f"threads={args.threads} n_tasks={n_tasks} H={H} n_eval={n_eval}")

    for seed in seeds:
        refs = load_landed_reference(seed)
        # const blocks are variant-independent -> compute once per seed
        blocks = {}
        for kind, st in cells:
            blocks[f"{kind}_s{st:g}"] = const_block_for(seed, kind, st, H, n_eval)
        logp(f"seed {seed}: const blocks ready")

        for vname, over in VARIANTS.items():
            out_path = OUTDIR / f"{vname}_seed{seed}.json"
            if out_path.exists():
                try:
                    json.loads(out_path.read_text())
                    logp(f"seed {seed} {vname}: already done, skip")
                    continue
                except Exception:
                    pass
            t0 = time.perf_counter()
            cfg = dict(DEFAULT_CONFIG)
            cfg.update(family="linked", n_tasks=n_tasks, horizon=H,
                       iterations=over["iterations"], history_len=over["history_len"])
            try:
                seed_everything(seed)
                rb = RobustBaseline(cfg)
                rb.fit(seed)
                cells_out = {}
                for kind, st in cells:
                    key = f"{kind}_s{st:g}"
                    cells_out[key] = evaluate_variant(rb.agent, seed, kind, st,
                                                      blocks[key], H, n_eval)
                cp = [c["captured_post"] for c in cells_out.values()
                      if c["captured_post"] is not None]
                record = {
                    "variant": vname, "seed": int(seed),
                    "overrides": over,
                    "config": {k: cfg[k] for k in sorted(cfg)},
                    "threads": int(args.threads),
                    "wall_seconds": time.perf_counter() - t0,
                    "captured_post_mean_cells": float(np.mean(cp)),
                    "captured_post_sd_cells": float(np.std(cp, ddof=1)) if len(cp) > 1 else None,
                    "cells": cells_out,
                    "landed_reference": refs,
                    "provenance": provenance(),
                }
                out_path.write_text(json.dumps(record, indent=2))
                logp(f"seed {seed} {vname}: captured_post={np.mean(cp):.2f} "
                     f"wall={record['wall_seconds']:.0f}s -> {out_path.name}")
            except Exception as err:  # noqa: BLE001 - keep the batch alive
                out_path.with_suffix(".failed").write_text(repr(err))
                logp(f"seed {seed} {vname}: FAILED {err!r}")
    logp("all variants done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
