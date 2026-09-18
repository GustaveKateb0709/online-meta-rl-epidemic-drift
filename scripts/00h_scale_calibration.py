"""Scale calibration for the regime-identification gap.

The D1/D2 diagnostic finding: the learner CAN learn when the regime signal is
present, and reward_scale=0.05 looked best in the 8-task D2 setting; but it only
captured 18% of the available gap. This script sweeps the two knobs that most
plausibly explain the shortfall -- training length and the number of parallel
tasks -- holding reward_scale=0.05 fixed, to see whether ANY cell reaches

    correct >= 0.6  AND  captured >= 0.5  AND  action distribution non-degenerate

where, per cell (family=action_revealed, stationary, seed=10):
    a*        = per-task argmax over a in {0,1,2} of the constant-action return
    correct   = fraction of tasks whose learned dominant action equals a*
    captured  = (return - best_single_const) / (best_per_task_const - best_single_const)
    counts    = overall a0/a1/a2 share of the learned (deterministic) actions

The regime signal is provided through the SAME truth-visible augmentation used in
D1/D2 ('sign'), i.e. this is an UPPER BOUND on what the hidden-z channel could do.

Results are written incrementally to results/validation/scale_calibration.json and
stdout is mirrored to artifacts/scale_calibration.txt.

Run:  python -u scripts/00h_scale_calibration.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import set_threads  # noqa: E402
from src.env import (FAMILY_ACTION_REVEALED, RegimeDriftEpidemicEnv,  # noqa: E402
                     sample_regimes)

# reuse the exact harness used by the diagnostic run, so numbers are comparable
_spec = importlib.util.spec_from_file_location(
    "diag_ladder", ROOT / "scripts" / "00g_diag_ladder.py")
dl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dl)


GRID_ITERS = (400, 1000, 2000)
GRID_TASKS = (8, 32)
REWARD_SCALE = 0.05
HORIZON = 156
SEED = 10
MODE = "sign"


def run_cell(iterations: int, n_tasks: int) -> dict:
    cfg = dict(dl.DEFAULT_DIAG)
    cfg.update(horizon=HORIZON, hidden=64, n_envs=n_tasks, n_tasks=n_tasks,
               iterations=iterations)
    rng = np.random.default_rng(SEED)
    regimes = sample_regimes(rng, n_tasks, family=FAMILY_ACTION_REVEALED)
    env = RegimeDriftEpidemicEnv(n_envs=n_tasks, horizon=HORIZON, seed=SEED)
    cr = dl.const_returns_per_task(regimes, HORIZON, SEED)
    astar = cr.argmax(1)
    best_single = float(cr.mean(0).max())
    best_per_task = float(cr.max(1).mean())

    ag, wall, last = dl.train(cfg, dl.aug_dim(MODE), env, regimes, MODE, SEED,
                              reward_scale=REWARD_SCALE)
    counts, dom, ret = ag.evaluate(env, regimes, MODE, SEED + 7)

    correct = float(np.mean(dom == astar))
    denom = best_per_task - best_single
    captured = float((ret - best_single) / denom) if denom > 1e-9 else float("nan")
    majority = float(np.bincount(astar, minlength=3).max() / astar.size)
    degenerate = bool(counts.max() >= 0.98)
    success = bool(correct >= 0.6 and captured >= 0.5 and not degenerate)
    return {
        "iterations": iterations, "n_tasks": n_tasks, "reward_scale": REWARD_SCALE,
        "mode": MODE, "horizon": HORIZON, "seed": SEED,
        "a_star_counts": [int(c) for c in np.bincount(astar, minlength=3)],
        "majority_baseline": majority,
        "best_single_const": best_single, "best_per_task_const": best_per_task,
        "available_gap": denom,
        "return_mean": float(ret),
        "correct": correct, "captured": captured,
        "action_counts": [float(c) for c in counts],
        "per_task_dominant": [int(d) for d in dom],
        "distinct_dominant": int(len(set(dom.tolist()))),
        "degenerate": degenerate, "success": success,
        "adv_std": float(last["adv_std"]), "wall_seconds": float(wall)}


def fmt_cell(c: dict) -> str:
    a = c["action_counts"]
    return (f"iter={c['iterations']:>4} tasks={c['n_tasks']:>2} | "
            f"correct={c['correct']:.2f} (chance {c['majority_baseline']:.2f}) | "
            f"captured={c['captured']:+.2f} | "
            f"a0/a1/a2={a[0]:.2f}/{a[1]:.2f}/{a[2]:.2f} | "
            f"ret={c['return_mean']:.1f} | {c['wall_seconds']:.0f}s"
            f"{'  <== SUCCESS' if c['success'] else ''}")


def main():
    set_threads(4)
    art = ROOT / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    logf = open(art / "scale_calibration.txt", "w")

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

    outdir = ROOT / "results" / "validation"
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "scale_calibration.json"
    (outdir / "scale_calibration.started").write_text(
        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    print("=" * 78, flush=True)
    print("SCALE CALIBRATION: reward_scale=0.05, action_revealed, stationary, seed=10")
    print(f"grid: iterations {GRID_ITERS} x n_tasks {GRID_TASKS}, mode='{MODE}'")
    print("=" * 78, flush=True)

    results = []
    t_all = time.perf_counter()
    for n_tasks in GRID_TASKS:
        for iterations in GRID_ITERS:
            print(f"\n-> running cell iter={iterations} tasks={n_tasks} ...", flush=True)
            c = run_cell(iterations, n_tasks)
            results.append(c)
            print("   " + fmt_cell(c), flush=True)
            outpath.write_text(json.dumps({"cells": results}, indent=2))
    print(f"\nTOTAL wall = {time.perf_counter() - t_all:.0f}s", flush=True)

    print("\n" + "=" * 78)
    print("SUMMARY (sorted by captured)")
    print("=" * 78)
    for c in sorted(results, key=lambda x: -x["captured"]):
        print(fmt_cell(c))
    wins = [c for c in results if c["success"]]
    print(f"\nSUCCESS CRITERION (correct>=0.6 & captured>=0.5 & non-degenerate): "
          f"{'MET by ' + str(len(wins)) + ' cell(s)' if wins else 'NOT MET'}")
    if wins:
        best = max(wins, key=lambda x: x["captured"])
        print(f"best: iter={best['iterations']} tasks={best['n_tasks']} "
              f"correct={best['correct']:.2f} captured={best['captured']:.2f}")
    print(f"\nwrote {outpath}")
    logf.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
