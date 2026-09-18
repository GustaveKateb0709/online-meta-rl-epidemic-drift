"""Build the two supplementary items that the manuscript references.

S1 Fig  training-budget diagnostic ladder (single-seed diagnostics)
S4 Table gating AUROC and its within-seed bootstrap intervals

Both are produced from files that already exist, so nothing here invents a
measurement. The reward-scale by drift-strength sweep that an earlier draft
referred to was never run and that reference has been removed from the
manuscript.

Run:  python scripts/make_supplementary.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import (FixedFormatter, FixedLocator,  # noqa: E402
                               NullFormatter, NullLocator)

from src.plotstyle import (PALETTE, WIDTH_ONE_HALF, apply_style,  # noqa: E402
                           panel_label, save_figure)

VALID = ROOT / "results" / "validation"
LEDGER = ROOT / "results" / "paper_numbers.json"


def load_json(p, default=None):
    try:
        return json.loads(p.read_text())
    except Exception:
        return default


def s1_figure():
    """Training-budget calibration plus the single-task / multi-task rungs."""
    cal = load_json(VALID / "scale_calibration.json", {}) or {}
    cells = [c for c in cal.get("cells", []) if c.get("n_tasks") == 8]
    if not cells:
        print("[skip] S1 Fig: no n_tasks=8 calibration cells")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_ONE_HALF, 2.8))
    src = []

    ax = axes[0]
    it = [c["iterations"] for c in cells]
    cap = [c["captured"] for c in cells]
    ax.plot(it, cap, "o-", color=PALETTE["ours"], label="captured fraction")
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Training iterations")
    ax.set_ylabel("Captured fraction of\nthe available gap")
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([400, 600, 1000, 2000]))
    ax.xaxis.set_major_formatter(FixedFormatter(["400", "600", "1000", "2000"]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    panel_label(ax, "a")
    _offs = {it[0]: (5, -8, "left"), it[-1]: (7, 5, "left")}
    for x, y in zip(it, cap):
        dx, dy, ha = _offs.get(x, (0, 8, "center"))
        ax.annotate(f"{y:+.2f}", (x, y), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, fontsize=7.5)
        src.append({"panel": "a", "iterations": x, "captured": y, "n_tasks": 8})
    ax.set_ylim(-0.24, 0.78)

    # right panel: the correctness of the per-task dominant action
    ax = axes[1]
    cor = [c["correct"] for c in cells]
    ax.plot(it, cor, "s-", color=PALETTE["ablation"], label="per-task agreement")
    ax.axhline(1 / 3, color="black", linewidth=0.8, linestyle=":")
    ax.annotate("chance", (it[0], 1 / 3), textcoords="offset points",
                xytext=(2, 4), fontsize=7.5)
    ax.set_xlabel("Training iterations")
    ax.set_ylabel("Per-task agreement with\nthe best constant action")
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([400, 600, 1000, 2000]))
    ax.xaxis.set_major_formatter(FixedFormatter(["400", "600", "1000", "2000"]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(0, 1)
    panel_label(ax, "b")
    for x, y in zip(it, cor):
        src.append({"panel": "b", "iterations": x, "per_task_agreement": y,
                    "n_tasks": 8})

    fig.tight_layout()
    save_figure(fig, "figS1_diagnostic_ladder", figdir=str(ROOT / "figures"))
    out = ROOT / "source_data" / "figS1_diagnostic_ladder.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = ["panel", "iterations", "captured", "per_task_agreement", "n_tasks"]
    with out.open("w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in src:
            fh.write(",".join("" if r.get(k) is None else str(r[k]) for k in keys) + "\n")
    print("[ok] figures/figS1_diagnostic_ladder.(png|pdf) + source_data csv")
    return True


def s4_table():
    L = load_json(LEDGER, {})
    if not L:
        print("[skip] S4 Table: ledger missing")
        return None
    fams = ["passive", "linked", "action_revealed"]
    sigs = [("consistency", "Consistency residual (proposed)"),
            ("entropy", "Posterior entropy"),
            ("variance", "Predictive variance")]
    lines = ["# S4 Table. Gating AUROC by family and signal\n",
             "The label is `adaptation_harmful`: at each step, two counterfactual "
             "trajectories are started from the same environment state, one following "
             "the online meta-policy and one following the non-adapting baseline, and "
             "the step is labelled harmful when the adaptive branch ends with the lower "
             "remaining return.\n",
             "AUROC is computed per cell and summarised over the cells of the "
             "family that use the main setting (five non-stationary drift kinds at "
             "strength 1.0, five seeds, so n = 25 cells). Standard deviations are "
             "across cells.\n",
             "| Family | Signal | AUROC | SD | n (cells) | Positive rate of the label |",
             "|---|---|---|---|---|---|"]
    for fam in fams:
        pos = L.get(f"gate.pos_rate.{fam}", {})
        posv = f"{pos.get('mean', float('nan')):.3f}" if pos else "--"
        for key, label in sigs:
            v = L.get(f"auroc.{key}.{fam}")
            if not v:
                continue
            lines.append(f"| `{fam}` | {label} | {v['mean']:.3f} | {v['sd']:.3f} | "
                         f"{v['n']} | {posv} |")
    ci = L.get("auroc.consistency.action_revealed.ci")
    if isinstance(ci, dict):
        lines.append(f"\nFor the proposed signal in the action-revealed family, the "
                     f"bootstrap 95% intervals computed within each seed and averaged "
                     f"across seeds span {ci['lo_mean']:.2f} to {ci['hi_mean']:.2f} "
                     f"(n = {ci['n']} cells).\n")
    lines.append("\nThe proposed consistency residual exceeds the posterior-entropy "
                 "baseline in both families but does not exceed the predictive-variance "
                 "baseline in either, and is reported as a negative result rather than "
                 "as a contribution.\n")
    out = ROOT / "manuscript" / "supplementary" / "s4_table_gating.md"
    out.write_text("\n".join(lines))
    print(f"[ok] {out.relative_to(ROOT)}")
    return True


def main():
    apply_style()
    ok1 = s1_figure()
    ok2 = s4_table()
    plt.close("all")
    return 0 if (ok1 and ok2) else 1


if __name__ == "__main__":
    raise SystemExit(main())
