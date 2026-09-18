"""Build the main figures of the manuscript from the result matrix.

Reads every per-cell JSON under ``results/main`` and writes Fig 2 to Fig 5 in the
shared journal style. Fig 1 is produced separately by
``scripts/02_drift_empirics.py`` because it uses the real surveillance data.

All figures are rendered at a width that satisfies the journal limits, enforced
by ``src.plotstyle.assert_width``, and every figure is accompanied by a
machine-readable source-data CSV for the journal's Source Data requirement.

Run:  python scripts/make_figures.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from src.plotstyle import (PALETTE, WIDTH_DOUBLE_COL, WIDTH_ONE_HALF,  # noqa: E402
                           apply_style, panel_label, save_figure)

MAIN = ROOT / "results" / "main"
FIGDIR = ROOT / "figures"
SRCDIR = ROOT / "source_data"

AGENTS = ["meta_online", "meta_frozen", "meta_oracle", "robust", "transfer",
          "retrain", "const0", "const1", "const2"]
# The comparison-relevant subset shown in the main boundary-condition figure.
FIG2_AGENTS = ["meta_online", "meta_frozen", "meta_oracle", "robust"]
AGENT_LABEL = {
    "meta_online": "Meta-RL (online)",
    "meta_frozen": "Meta-RL (frozen)",
    "meta_oracle": "Meta-RL (oracle)",
    "robust": "Robust (no adaptation)",
    "transfer": "Transfer + fine-tune",
    "retrain": "Retrain from scratch",
    "const0": "Constant none",
    "const1": "Constant moderate",
    "const2": "Constant strong",
}
def _soft(hex_color, f=0.5):
    """Blend a hex colour toward white (visual desaturation for bar fills)."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = (round(c + (255 - c) * f) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


AGENT_COLOR = {
    "meta_online": PALETTE["ours"],
    "meta_frozen": PALETTE["ablation"],
    "meta_oracle": PALETTE["oracle"],
    "robust": PALETTE["robust"],
    "transfer": PALETTE["transfer"],
    "retrain": PALETTE["retrain"],
    "const0": PALETTE["const"],
    "const1": PALETTE["const"],
    "const2": PALETTE["const"],
}
DRIFT_ORDER = ["stationary", "gradual_beta", "abrupt_beta", "gradual_rho",
               "endogenous_beh", "abrupt_eps"]
DRIFT_LABEL = {
    "stationary": "Stationary",
    "gradual_beta": "Gradual $\\beta$",
    "abrupt_beta": "Abrupt $\\beta$",
    "gradual_rho": "Gradual $\\rho$",
    "endogenous_beh": "Behavioural rebound",
    "abrupt_eps": "Abrupt efficacy loss",
}
FAMILIES = ["passive", "linked", "action_revealed"]
FAMILY_LABEL = {
    "passive": "Passive family",
    "linked": "Linked family",
    "action_revealed": "Action-revealed family",
}


def families_in(cells):
    """Families actually present in the data, in the canonical order."""
    seen = {c["family"] for c in cells}
    return [f for f in FAMILIES if f in seen] or list(seen)


# --------------------------------------------------------------------------- #
def load_cells():
    """Return per-cell records plus the run-level gate and PII blocks.

    One file per cell. Inside a file, ``eval`` holds the agents keyed by name and
    also the cell metadata and the constant-action reference block.
    """
    cells, gates, piis, seeds = [], defaultdict(list), defaultdict(list), set()
    for fp in sorted(MAIN.glob("*.json")):
        try:
            d = json.loads(fp.read_text())
        except Exception:
            continue
        seeds.add(d.get("seed"))
        ev = d.get("eval")
        if not isinstance(ev, dict):
            continue
        fam = d.get("family") or ev.get("family")
        kind = d.get("drift_kind") or ev.get("drift_kind")
        strength = d.get("strength", ev.get("strength"))
        try:
            strength = float(strength)
        except (TypeError, ValueError):
            strength = None

        rec = {k: v for k, v in ev.items() if k in AGENTS and isinstance(v, dict)}
        rec["family"] = fam
        rec["drift_kind"] = kind
        rec["strength"] = strength
        rec["seed"] = d.get("seed")
        cells.append(rec)

        g = d.get("gate")
        if isinstance(g, dict):
            g2 = dict(g)
            g2.setdefault("family", fam)
            gates[g2["family"]].append(g2)
        p = d.get("pii")
        if isinstance(p, dict):
            p2 = dict(p)
            p2.setdefault("family", fam)
            piis[p2["family"]].append(p2)
    return cells, dict(gates), dict(piis), sorted(s for s in seeds if s is not None)


def _cap(cell, agent, window="full"):
    a = cell.get(agent)
    if not isinstance(a, dict):
        return None
    key = "captured" if window == "full" else "captured_post"
    v = a.get(key)
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)


def _cap_mean(cells, agent, window="full"):
    """Mean captured over a list of cells, or None when nothing is available."""
    vals = [_cap(c, agent, window) for c in cells]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def _gather(cells, family, agents, kinds=None, strength=None, window="full"):
    """Mean and sd across seeds for each agent, respecting the cell filters."""
    out = {}
    for ag in agents:
        per_seed = defaultdict(list)
        for c in cells:
            if c["family"] != family:
                continue
            if kinds is not None and c["drift_kind"] not in kinds:
                continue
            if strength is not None and abs(float(c.get("strength", 0)) - strength) > 1e-9:
                continue
            v = _cap(c, ag, window)
            if v is not None:
                per_seed[c["seed"]].append(v)
        vals = [float(np.mean(v)) for v in per_seed.values() if v]
        if vals:
            out[ag] = (float(np.mean(vals)), float(np.std(vals)), len(vals))
    return out


# --------------------------------------------------------------------------- #
def _gather_perseed(cells, family, agents, kinds=None, strength=None,
                    window="full"):
    """Per-seed group means for each agent, for the fig2 strip plot."""
    out = {}
    for ag in agents:
        per_seed = defaultdict(list)
        for c in cells:
            if c["family"] != family:
                continue
            if kinds is not None and c["drift_kind"] not in kinds:
                continue
            if strength is not None and abs(float(c.get("strength", 0)) - strength) > 1e-9:
                continue
            v = _cap(c, ag, window)
            if v is not None:
                per_seed[c["seed"]].append(v)
        seeds = sorted(per_seed)
        vals = [float(np.mean(per_seed[s])) for s in seeds]
        if vals:
            out[ag] = {"seeds": seeds, "values": vals}
    return out


def fig2_boundary(cells):
    """Captured fraction by family and agent: the boundary condition.

    Strip plot: one point per training seed and a bar at the group mean. The
    panels use independent vertical scales, because a single outlier seed in
    the action-revealed family would otherwise compress the linked panel to a
    strip. The passive family is deliberately excluded. Its available gap is
    close to zero (about 2.5 return units), so the ratio of realised to
    available gap is numerically unstable there and would not be comparable
    with the other families on a shared axis. The passive family appears in
    S1 Table and in the ledger as the no-gap reference, reported as raw return.
    """
    nonstat = [k for k in DRIFT_ORDER if k != "stationary"]
    fams = [f for f in families_in(cells) if f != "passive"] or families_in(cells)
    fig, axes = plt.subplots(1, len(fams), figsize=(WIDTH_DOUBLE_COL, 3.0))
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(20260917)
    src = []
    for ax, fam, letter in zip(axes, fams, "abcdef"):
        perseed = _gather_perseed(cells, fam, FIG2_AGENTS, kinds=nonstat,
                                  strength=1.0)
        order = [a for a in FIG2_AGENTS if a in perseed]
        x = np.arange(len(order))
        for xi, a in zip(x, order):
            vals = np.array(perseed[a]["values"])
            jit = rng.uniform(-0.17, 0.17, size=vals.size)
            ax.scatter(xi + jit, vals, s=9, color=AGENT_COLOR[a],
                       alpha=0.6, linewidths=0, zorder=3)
            m, s_ = float(vals.mean()), float(vals.std())
            ax.plot([xi - 0.30, xi + 0.30], [m, m], color="#222222",
                    lw=2.0, zorder=4, solid_capstyle="butt")
            for sd_, sv_ in zip(perseed[a]["seeds"], perseed[a]["values"]):
                src.append({"panel": fam, "agent": a, "seed": sd_,
                            "captured_seed": sv_, "group_mean": m,
                            "group_sd": s_, "group_n": int(vals.size)})
        ax.axhline(0.0, color="#999999", linewidth=0.7, linestyle="--",
                   zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels([AGENT_LABEL[a] for a in order], rotation=45,
                           ha="right", fontsize=7.5)
        ax.set_xlim(-0.6, len(order) - 0.4)
        ax.set_title(FAMILY_LABEL[fam], fontsize=9)
        panel_label(ax, letter)
        ax.margins(y=0.10)
    axes[0].set_ylabel("Captured fraction of the\navailable gap")
    fig.tight_layout()
    save_figure(fig, "fig2_boundary_condition", figdir=str(FIGDIR))
    _write_source("fig2_boundary_condition.csv", src)
    return fig


def fig3_gate(gates):
    """Gating signal comparison on the action-revealed family."""
    fig, ax = plt.subplots(figsize=(WIDTH_ONE_HALF, 3.2))
    signals = [("consistency", PALETTE["ours"], "Consistency residual"),
               ("entropy", PALETTE["ablation"], "Posterior entropy"),
               ("variance", PALETTE["oracle"], "Predictive variance")]
    fams = [f for f in FAMILIES if f in gates]
    src = []
    for i, (key, col, lab) in enumerate(signals):
        x = np.arange(len(fams)) + (i - (len(signals) - 1) / 2.0) * 0.26
        mean, err = [], []
        for fam in fams:
            rows = gates.get(fam, [])
            v = [r.get(f"auroc_{key}") for r in rows]
            v = [float(t) for t in v if t is not None and not np.isnan(t)]
            mean.append(float(np.mean(v)) if v else np.nan)
            err.append(float(np.std(v)) if v else 0.0)
            src.append({"family": fam, "signal": key,
                        "auroc_mean": mean[-1], "auroc_sd": err[-1], "n": len(v)})
        ax.bar(x, mean, yerr=err, capsize=2.5, width=0.24,
               color=_soft(col), edgecolor=col, linewidth=0.8,
               error_kw=dict(ecolor="#666666", elinewidth=0.9, capsize=2.5,
                             zorder=4),
               label=lab)
    ax.axhline(0.5, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(np.arange(len(fams)))
    ax.set_xticklabels([FAMILY_LABEL.get(f, f) for f in fams], fontsize=8)
    ax.set_ylabel("AUROC for $\\it{adaptation\\ harmful}$")
    ax.legend(fontsize=7.5, loc="lower left", bbox_to_anchor=(0.0, 1.02),
              ncol=3, frameon=False, columnspacing=1.2, handlelength=1.6)
    fig.tight_layout()
    save_figure(fig, "fig3_gating", figdir=str(FIGDIR))
    _write_source("fig3_gating.csv", src)
    return fig


def fig4_drift_heterogeneity(cells):
    """Captured fraction by drift kind: means over seeds, both strengths.

    Markers are the across-seed means; filled markers are strength 1.0 and
    open markers strength 2.0. Whiskers are omitted because their scale is
    dominated by the same small-denominator cells that make the captured
    fraction unstable (Section 3.1); the per-kind standard deviations are
    provided in the Source Data. Panels use independent vertical scales.
    """
    agents = ["meta_online", "meta_frozen"]
    marker = {"meta_online": "o", "meta_frozen": "s"}
    fams = [f for f in families_in(cells) if f != "passive"] or families_in(cells)
    fig, axes = plt.subplots(1, len(fams), figsize=(WIDTH_DOUBLE_COL, 3.1))
    axes = np.atleast_1d(axes)
    src = []
    for ax, fam, letter in zip(axes, fams, "abcdef"):
        kinds = [k for k in DRIFT_ORDER if k != "stationary"]
        x = np.arange(len(kinds))
        for i, ag in enumerate(agents):
            for st, filled, lsty, alpha in ((1.0, True, "-", 0.9),
                                            (2.0, False, "--", 0.6)):
                mean = []
                for k in kinds:
                    s = _gather(cells, fam, [ag], kinds=[k], strength=st)
                    if ag in s:
                        mean.append(s[ag][0])
                        src.append({"family": fam, "drift_kind": k, "agent": ag,
                                    "strength": st, "captured_mean": s[ag][0],
                                    "captured_sd": s[ag][1], "n_seeds": s[ag][2]})
                    else:
                        mean.append(np.nan)
                off = (i - 0.5) * 0.16
                ax.plot(x + off, mean, lsty, color=AGENT_COLOR[ag], lw=1.0,
                        alpha=alpha, zorder=2)
                ax.plot(x + off, mean, marker[ag], ms=4.6, linestyle="none",
                        mfc=AGENT_COLOR[ag] if filled else "white",
                        mec=AGENT_COLOR[ag], mew=1.0, alpha=alpha, zorder=3)
        ax.axhline(0.0, color="#999999", linewidth=0.7, linestyle="--",
                   zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels([DRIFT_LABEL[k] for k in kinds], rotation=35,
                           ha="right", fontsize=7.5)
        ax.set_title(FAMILY_LABEL.get(fam, fam), fontsize=9)
        panel_label(ax, letter)
        ax.margins(y=0.12)
    axes[0].set_ylabel("Captured fraction of the\navailable gap")
    handles = [Line2D([], [], color=AGENT_COLOR["meta_online"], marker="o",
                      ls="none", ms=4.6, label="Meta-RL (online)"),
               Line2D([], [], color=AGENT_COLOR["meta_frozen"], marker="s",
                      ls="none", ms=4.6, label="Meta-RL (frozen)")]
    axes[0].legend(handles=handles, fontsize=7.5, loc="lower left",
                   frameon=False)
    fig.tight_layout()
    save_figure(fig, "fig4_drift_heterogeneity", figdir=str(FIGDIR))
    _write_source("fig4_drift_heterogeneity.csv", src)
    return fig


def fig5_pii(cells, piis):
    """Passive identifiability against the contrast that carries the claim.

    One point per (family, drift kind). The horizontal position is the family's
    passive identifiability index, which is a property of the family, so points
    of a family share an x value; the vertical position is the paired
    online-minus-frozen contrast in that drift kind. The passive family is not
    shown, because its available gap is close to zero and the ratio is unstable.
    """
    fig, ax = plt.subplots(figsize=(WIDTH_ONE_HALF, 3.2))
    src, xs, ys = [], [], []
    fams = [f for f in families_in(cells) if f != "passive"] or families_in(cells)
    for i, fam in enumerate(fams):
        p = piis.get(fam, [])
        x = float(np.mean([t.get("best_return_r2", np.nan) for t in p])) if p else np.nan
        vals, kinds = [], []
        for kind in [k for k in DRIFT_ORDER if k != "stationary"]:
            grp = [c for c in cells if c["family"] == fam
                   and c["drift_kind"] == kind and c["strength"] == 1.0]
            # paired within seed, then spread across seeds
            delta = []
            for seed in sorted({c["seed"] for c in grp}):
                gg = [c for c in grp if c["seed"] == seed]
                on = _cap_mean(gg, "meta_online", "post")
                fz = _cap_mean(gg, "meta_frozen", "post")
                if on is not None and fz is not None:
                    delta.append(on - fz)
            if delta:
                vals.append(float(np.mean(delta)))
                kinds.append(kind)
                xs.append(x)
                ys.append(float(np.mean(delta)))
                src.append({"family": fam, "drift_kind": kind,
                            "pii_best_return_r2": x,
                            "contrast_online_minus_frozen_post": float(np.mean(delta)),
                            "n_seeds": len(delta)})
        if vals:
            ax.scatter([x] * len(vals), vals, s=42, marker=["o", "s", "^"][i % 3],
                       color=[PALETTE["ours"], PALETTE["ablation"],
                              PALETTE["oracle"]][i % 3],
                       edgecolor="black", linewidth=0.5, zorder=3,
                       label=FAMILY_LABEL.get(fam, fam))
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Passive identifiability index\n(out-of-sample $R^2$; higher = more recoverable)")
    ax.set_ylabel("Online $-$ frozen contrast\n(post-drift captured)")
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    fig.tight_layout()
    save_figure(fig, "fig5_pii_dose_response", figdir=str(FIGDIR))
    _write_source("fig5_pii_dose_response.csv", src)
    return fig


# --------------------------------------------------------------------------- #
def _write_source(name, rows):
    SRCDIR.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with (SRCDIR / name).open("w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join("" if r.get(k) is None else str(r[k]) for k in keys) + "\n")


def main():
    apply_style()
    cells, gates, piis, seeds = load_cells()
    print(f"loaded {len(cells)} cells over {len(seeds)} seeds")
    if not cells:
        print("no cells yet; run the experiment matrix first")
        return 1
    made = []
    for name, fn, args in (("fig2", fig2_boundary, (cells,)),
                           ("fig3", fig3_gate, (gates,)),
                           ("fig4", fig4_drift_heterogeneity, (cells,)),
                           ("fig5", fig5_pii, (cells, piis))):
        try:
            fn(*args)
            made.append(name)
            print(f"[ok] {name}")
        except Exception as exc:  # keep going, report
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
        plt.close("all")
    print(f"built {len(made)} of 4 figures")
    return 0 if len(made) == 4 else 1


if __name__ == "__main__":
    raise SystemExit(main())
