"""Robust across-seed statistics for the two paired contrasts that carry the
boundary-condition claim.

The ledger (``analyze_matrix.py``) reports each contrast as a mean and an
across-seed standard deviation over 25 seeds. The across-seed distribution is
heavy-tailed because a small number of cells have a near-degenerate available
gap (the same denominator pathology that makes the captured fraction
unreportable in the passive family), so this script additionally computes,
deterministically and from the per-cell files:

* per-seed contrast values (mean over the five non-stationary drift kinds at
  strength 1.0), for the paired contrasts online-minus-frozen and
  oracle-minus-online in the linked and action_revealed families;
* median, interquartile range, count of positive seeds;
* the across-family ordering count (in how many seeds the action_revealed
  contrast exceeds the linked contrast);
* the same contrasts in raw post-drift return units, which are immune to the
  denominator pathology;
* the captured-unit contrasts after dropping cells whose available gap is
  below 5.0 (a sensitivity check, not the reported estimate);
* the list of cells whose available gap is below that threshold.

Output: ``results/robust_contrasts.json``.

Run:  python scripts/robust_contrasts.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "results" / "main"
OUT = ROOT / "results" / "robust_contrasts.json"

KINDS = ["gradual_beta", "abrupt_beta", "gradual_rho", "endogenous_beh", "abrupt_eps"]
FAMILIES = ["linked", "action_revealed"]
CONTRASTS = [("meta_oracle", "meta_online", "oracle_online"),
             ("meta_online", "meta_frozen", "online_frozen")]
MIN_GAP_SENSITIVITY = 5.0


def load_cells():
    cells = {}
    for fp in sorted(MAIN.glob("*.json")):
        d = json.loads(fp.read_text())
        fam, kind = d.get("family"), d.get("drift_kind")
        try:
            strength = float(d.get("strength"))
        except (TypeError, ValueError):
            continue
        if fam in FAMILIES and kind in KINDS and strength == 1.0:
            cells.setdefault((fam, int(d["seed"])), {})[kind] = d
    return cells


def per_seed(cells, fam, seed, a, b, field, min_gap=None):
    vals = []
    for kind in KINDS:
        cell = cells.get((fam, seed), {}).get(kind)
        if cell is None:
            continue
        ev = cell["eval"]
        x = ev.get(a, {}).get(field)
        y = ev.get(b, {}).get(field)
        if x is None or y is None:
            continue
        if min_gap is not None:
            gap = (ev.get("baselines_const") or {}).get("available_gap")
            if gap is None or gap < min_gap:
                continue
        vals.append(x - y)
    return float(np.mean(vals)) if vals else None


def summarise(values):
    x = np.array([v for v in values if v is not None], dtype=float)
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "sd": float(x.std(ddof=1)),
        "median": float(np.median(x)),
        "iqr_lo": float(np.percentile(x, 25)),
        "iqr_hi": float(np.percentile(x, 75)),
        "n_positive": int((x > 0).sum()),
        "per_seed": [None if v is None else round(v, 6) for v in values],
    }


def main():
    cells = load_cells()
    seeds = sorted({s for (_, s) in cells})
    out = {"n_seeds": len(seeds), "kinds": KINDS, "strength": 1.0}

    for a, b, name in CONTRASTS:
        for fam in FAMILIES:
            cap = [per_seed(cells, fam, s, a, b, "captured_post") for s in seeds]
            ret = [per_seed(cells, fam, s, a, b, "post_return_mean") for s in seeds]
            capf = [per_seed(cells, fam, s, a, b, "captured_post",
                             min_gap=MIN_GAP_SENSITIVITY) for s in seeds]
            out[f"{name}.{fam}"] = {
                "captured_post": summarise(cap),
                "post_return": summarise(ret),
                "captured_post_gap_ge_5": summarise(capf),
            }
        cap_a = [per_seed(cells, "action_revealed", s, a, b, "captured_post") for s in seeds]
        cap_l = [per_seed(cells, "linked", s, a, b, "captured_post") for s in seeds]
        ret_a = [per_seed(cells, "action_revealed", s, a, b, "post_return_mean") for s in seeds]
        ret_l = [per_seed(cells, "linked", s, a, b, "post_return_mean") for s in seeds]
        pairs_cap = [(x, y) for x, y in zip(cap_a, cap_l) if x is not None and y is not None]
        pairs_ret = [(x, y) for x, y in zip(ret_a, ret_l) if x is not None and y is not None]
        out[f"ordering.{name}"] = {
            "action_revealed_gt_linked_captured": {
                "n": len(pairs_cap),
                "count": int(sum(x > y for x, y in pairs_cap))},
            "action_revealed_gt_linked_post_return": {
                "n": len(pairs_ret),
                "count": int(sum(x > y for x, y in pairs_ret))},
        }

    # Cells whose available gap is small enough to make the captured ratio
    # unstable (same pathology as the passive family). Recorded so that the
    # heavy tail of the contrasts is traceable to specific cells.
    degenerate = []
    for (fam, seed), kinds_map in sorted(cells.items()):
        for kind, cell in sorted(kinds_map.items()):
            gap = (cell.get("eval").get("baselines_const") or {}).get("available_gap")
            if gap is not None and float(gap) < MIN_GAP_SENSITIVITY:
                degenerate.append({"family": fam, "kind": kind, "seed": seed,
                                   "available_gap": round(float(gap), 3)})
    out["degenerate_cells"] = degenerate
    out["degenerate_gap_threshold"] = MIN_GAP_SENSITIVITY

    # Exact one-sided sign tests against a zero-median null, for the counts
    # quoted in the manuscript. Deterministic given the counts.
    from scipy.stats import binomtest

    def sign_p(k, n):
        return float(binomtest(k, n, 0.5, alternative="greater").pvalue)

    oc = out["ordering.oracle_online"]
    out["sign_tests"] = {
        "oracle_online.action_revealed.positive_captured": {
            "k": out["oracle_online.action_revealed"]["captured_post"]["n_positive"],
            "n": out["oracle_online.action_revealed"]["captured_post"]["n"],
            "p_one_sided": sign_p(out["oracle_online.action_revealed"]["captured_post"]["n_positive"],
                                  out["oracle_online.action_revealed"]["captured_post"]["n"])},
        "ordering.oracle_online.action_revealed_gt_linked.captured": {
            "k": oc["action_revealed_gt_linked_captured"]["count"],
            "n": oc["action_revealed_gt_linked_captured"]["n"],
            "p_one_sided": sign_p(oc["action_revealed_gt_linked_captured"]["count"],
                                  oc["action_revealed_gt_linked_captured"]["n"])},
        "ordering.oracle_online.action_revealed_gt_linked.post_return": {
            "k": oc["action_revealed_gt_linked_post_return"]["count"],
            "n": oc["action_revealed_gt_linked_post_return"]["n"],
            "p_one_sided": sign_p(oc["action_revealed_gt_linked_post_return"]["count"],
                                  oc["action_revealed_gt_linked_post_return"]["n"])},
    }

    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[ok] wrote {OUT.relative_to(ROOT)}")
    for key in ("oracle_online.linked", "oracle_online.action_revealed",
                "online_frozen.linked", "online_frozen.action_revealed"):
        c = out[key]["captured_post"]
        print(f"  {key:36s} captured: median {c['median']:+.3f} "
              f"IQR ({c['iqr_lo']:+.3f},{c['iqr_hi']:+.3f}) "
              f"pos {c['n_positive']}/{c['n']}")
    for name in ("oracle_online", "online_frozen"):
        o = out[f"ordering.{name}"]
        print(f"  ordering {name}: action_revealed>linked "
              f"captured {o['action_revealed_gt_linked_captured']['count']}/{o['action_revealed_gt_linked_captured']['n']}"
              f", return {o['action_revealed_gt_linked_post_return']['count']}/{o['action_revealed_gt_linked_post_return']['n']}")
    print(f"  degenerate cells (gap < {MIN_GAP_SENSITIVITY}): {len(degenerate)}")
    for c in degenerate:
        print(f"    {c['family']}/{c['kind']}/seed{c['seed']}: gap {c['available_gap']}")


if __name__ == "__main__":
    main()
