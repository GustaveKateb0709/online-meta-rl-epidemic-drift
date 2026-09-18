"""Fast end-to-end smoke test for the paper-17 pipeline.

Runs the full chain at a tiny scale (8 tasks, 20 meta-iterations, short horizon)
over BOTH environment families so that every code path -- meta-RL with and
without context, baselines, gating with counterfactual labels, and the Passive
Identifiability Index -- is exercised in well under two minutes. Prints the real
numbers it obtains; nothing here is hard-coded.

Run:  python scripts/00_smoke_meta_rl.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.run_main import parse_args, run_seed  # noqa: E402
from src.common import set_threads  # noqa: E402


def main():
    argv = [
        "--seeds", "10",
        "--families", "passive,action_revealed",
        "--n-tasks", "8",
        "--horizon", "60",
        "--iterations", "20",
        "--hidden", "64",
        "--z-dim", "5",
        "--ctx-len", "12",
        "--ppo-epochs", "2",
        "--n-minibatch", "2",
        "--tasks-per-kind", "6",
        "--kinds", "stationary,gradual_beta,endogenous_beh,abrupt_eps",
        "--gate-kinds", "gradual_beta,abrupt_eps",
        "--strengths", "1.0",
        "--label-stride", "3",
        "--gate-boot", "200",
        "--finetune-steps", "1920",
        "--retrain-budget", "1920",
        "--pii-tasks", "60",
        "--threads", "4",
        "--with-oracle",
        "--verbose",
        "--log-every", "5",
    ]
    args = parse_args(argv)
    print("=" * 78)
    print("SMOKE TEST: two-family regime-drift meta-RL pipeline")
    print(f"torch threads = {set_threads(args.threads)}")
    print("=" * 78)

    t0 = time.perf_counter()
    rec = run_seed(10, args)
    wall = time.perf_counter() - t0

    outdir = ROOT / "results" / "smoke"
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "seed_10.json"
    import json
    outpath.write_text(json.dumps(rec, indent=2))

    for fam in rec["families"]:
        tr = rec["train"][fam]
        print(f"\n=== family = {fam} ===")
        print(f"meta-training: env_steps={tr['env_steps']:,} "
              f"wall={tr['wall_seconds']:.1f}s final_return={tr['final_return']:.2f} "
              f"n_params={tr['n_params']:,}")

    print("\n--- drift-task returns by family:kind ---")
    agents = ("meta_online", "meta_frozen", "meta_oracle", "robust", "transfer",
              "retrain", "const0", "const1", "const2")
    for key, row in rec["eval"].items():
        print(f"[{key}]")
        print(f"  {'agent':<12}{'return':>10}{'post':>10}{'adapt_med':>11}{'regret':>9}")
        for agent in agents:
            if agent in row:
                r = row[agent]
                print(f"  {agent:<12}{r['return_mean']:>10.2f}{r['post_return_mean']:>10.2f}"
                      f"{r.get('adaptation_steps_median', r['adaptation_steps']):>11.1f}"
                      f"{r['regret']:>9.2f}")
        best = max(row[a]["return_mean"] for a in agents if a in row)
        print(f"  best return in cell = {best:.2f}")

    print("\n--- gating diagnostics (per family) ---")
    for fam, g in rec["gate"].items():
        print(f"[{fam}] pos_rate={g['pos_rate']:.3f} n_steps={g['n_steps']} "
              f"calib_scale={g['calib_scale']:.4f}")
        print(f"    AUROC consistency {g['auroc_consistency']:.3f} "
              f"CI {fmt_ci(g['ci']['consistency']['auroc'])}   (proposed)")
        print(f"    AUROC entropy     {g['auroc_entropy']:.3f}   "
              f"AUROC variance {g['auroc_variance']:.3f}   PR-AUC {g['pr_auc_consistency']:.3f}")

    print("\n--- Passive Identifiability Index (per family) ---")
    for fam, p in rec["pii"].items():
        print(f"[{fam}] accuracy={p['accuracy']:.3f} chance={p['chance']:.3f} "
              f"macro_f1={p['macro_f1']:.3f} continuous_gap={p['continuous_gap']:.2f} "
              f"best_return_r2={p['best_return_r2']:.3f} counts={p['best_action_counts']}")

    print(f"\nTOTAL smoke wall time: {wall:.1f}s")
    print(f"wrote {outpath}")
    return 0


def fmt_ci(ci):
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci and np.isfinite(ci[0]) else "[-]"


if __name__ == "__main__":
    raise SystemExit(main())
