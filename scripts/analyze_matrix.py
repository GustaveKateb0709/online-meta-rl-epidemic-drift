"""Aggregate the result matrix into the numeric ledger used by the manuscript.

Reads every per-cell JSON under ``results/main`` and writes
``results/paper_numbers.json``. Every key in that file corresponds to exactly one
``[[R:...]]`` placeholder in ``manuscript/manuscript.md``, so that each number in
the text can be traced to a single named entry with a reported sample size.

Also copies the external-calibration percentiles out of
``results/calibration/drift_empirics.json`` so that the whole ledger comes from
two files.

Run:  python scripts/analyze_matrix.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "results" / "main"
CALIB = ROOT / "results" / "calibration" / "drift_empirics.json"
OUT = ROOT / "results" / "paper_numbers.json"

AGENTS = ["meta_online", "meta_frozen", "meta_oracle", "robust", "transfer",
          "retrain", "const0", "const1", "const2"]
FAMILIES = ["passive", "linked", "action_revealed"]
NON_STATIONARY = ["gradual_beta", "abrupt_beta", "gradual_rho",
                  "endogenous_beh", "abrupt_eps"]
MAIN_STRENGTH = 1.0
MAIN_FAMILY = "action_revealed"


def load():
    """Read one per-cell file per matrix cell.

    Each file holds a single cell: ``eval`` is keyed by agent name, and the
    cell metadata (family, drift kind, strength) plus the constant-action
    reference block sit alongside the agents under the same ``eval`` key.
    """
    cells = []
    gates, piis = defaultdict(list), defaultdict(list)
    for fp in sorted(MAIN.glob("*.json")):
        try:
            d = json.loads(fp.read_text())
        except Exception:
            continue
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
        rec["baselines_const"] = ev.get("baselines_const")
        rec["seed"] = d.get("seed")
        rec["file"] = fp.name
        cells.append(rec)

        g = d.get("gate")
        if isinstance(g, dict):
            g2 = dict(g)
            g2.setdefault("family", fam)
            g2["drift_kind"] = kind
            g2["strength"] = strength
            gates[g2["family"]].append(g2)
        p = d.get("pii")
        if isinstance(p, dict):
            p2 = dict(p)
            p2.setdefault("family", fam)
            piis[p2["family"]].append(p2)
    return cells, dict(gates), dict(piis)


def stat(values):
    v = [float(x) for x in values if x is not None and not np.isnan(float(x))]
    if not v:
        return None
    return {"mean": float(np.mean(v)), "sd": float(np.std(v)), "n": len(v)}


def select(cells, family=None, kind=None, strength=None):
    out = []
    for c in cells:
        if family is not None and c["family"] != family:
            continue
        if kind is not None and c["drift_kind"] != kind:
            continue
        if strength is not None and abs(float(c.get("strength", -1)) - strength) > 1e-9:
            continue
        out.append(c)
    return out


def per_seed_mean(cells, agent, field):
    """Average over the cells of a group within each seed, then keep the seeds."""
    return list(per_seed_mean_map(cells, agent, field).values())


def per_seed_mean_map(cells, agent, field):
    """Same as :func:`per_seed_mean` but keyed by seed, for paired contrasts."""
    by_seed = defaultdict(list)
    for c in cells:
        a = c.get(agent)
        if not isinstance(a, dict):
            continue
        v = a.get(field)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        if c["seed"] is None:
            continue
        by_seed[c["seed"]].append(float(v))
    return {k: float(np.mean(v)) for k, v in by_seed.items() if v}


def paired_contrast(cells, agent_a, agent_b, field="captured_post"):
    """Paired across-seed difference of two agents on one group of cells."""
    ma = per_seed_mean_map(cells, agent_a, field)
    mb = per_seed_mean_map(cells, agent_b, field)
    diffs = [ma[s] - mb[s] for s in sorted(set(ma) & set(mb))]
    return stat(diffs)


def main():
    if not MAIN.exists() or not list(MAIN.glob("*.json")):
        print("no cells under results/main yet")
        return 1
    cells, gates, piis = load()
    if not cells:
        print("no parsed cells")
        return 1

    ledger = {"_provenance": {
        "n_cells": len(cells),
        "n_seeds": len({c["seed"] for c in cells}),
        "families": sorted({c["family"] for c in cells}),
        "main_family": MAIN_FAMILY,
        "main_strength": MAIN_STRENGTH,
        "main_kinds": NON_STATIONARY,
        "note": ("main = mean over the five non-stationary drift kinds at "
                 "strength 1.0; stationary is excluded because it poses no "
                 "adaptation demand"),
    }}

    # ---- external calibration ------------------------------------------- #
    if CALIB.exists():
        try:
            cal = json.loads(CALIB.read_text())
            mapping = cal.get("mapping_strength_to_real_percentile", {})
            for s, tag in (("strength_1.0", "s1"), ("strength_2.0", "s2"),
                           ("strength_3.0", "s3")):
                lc = (mapping.get(s) or {}).get("level_channel", {})
                if "percentile_in_real_level_breaks_ILI" in lc:
                    ledger[f"calib.pct_{tag}"] = lc["percentile_in_real_level_breaks_ILI"]
                if "sim_median_fold_change" in lc:
                    ledger[f"calib.fold_{tag}"] = lc["sim_median_fold_change"]
            real_lc = cal.get("real_drift_magnitude", {}).get("level_channel", {})
            if "ILI_fold" in real_lc:
                for q in ("p10", "p50", "p90"):
                    if q in real_lc["ILI_fold"]:
                        ledger[f"calib.real_fold_{q}"] = real_lc["ILI_fold"][q]
            for key, name in (("n_ILI", "n_level_changepoints"),
                              ("n_all", "n_level_changepoints_all_series")):
                if key in real_lc:
                    ledger[f"calib.{name}"] = real_lc[key]
            seasonal = cal.get("seasonal_decomposition", {})
            if "ILI_seasonal_amplitude_log1p_mean" in seasonal:
                ledger["calib.seasonal_amplitude_log1p"] = \
                    seasonal["ILI_seasonal_amplitude_log1p_mean"]
            for s in cal.get("seasonal_decomposition", {}).get("per_series", []):
                if s.get("series") == "ILI_national":
                    ledger["calib.national_weeks"] = s.get("n_weeks")
                    break
        except Exception as exc:
            ledger["calib._error"] = str(exc)

    # ---- captured, main aggregate and per drift kind --------------------- #
    for fam in FAMILIES:
        grp = select(cells, family=fam, strength=MAIN_STRENGTH)
        grp = [c for c in grp if c["drift_kind"] in NON_STATIONARY]
        for ag in AGENTS:
            s = stat(per_seed_mean(grp, ag, "captured"))
            if s:
                ledger[f"captured.{fam}.{ag}"] = s
            s = stat(per_seed_mean(grp, ag, "captured_post"))
            if s:
                ledger[f"captured_post.{fam}.{ag}"] = s
            r = stat(per_seed_mean(grp, ag, "return_mean"))
            if r:
                ledger[f"return.{fam}.{ag}"] = r
        s = stat(per_seed_mean(grp, "meta_online", "captured"))
        if s:
            ledger[f"captured.{fam}.main"] = s

    # Per drift kind, ALWAYS qualified by family. An unqualified
    # ``captured.<kind>.<agent>`` key would be overwritten once per family and
    # would silently end up holding whichever family was processed last.
    for fam in FAMILIES:
        for kind in NON_STATIONARY + ["stationary"]:
            grp = select(cells, family=fam, kind=kind, strength=MAIN_STRENGTH)
            for ag in ("meta_online", "robust", "meta_frozen", "meta_oracle"):
                s = stat(per_seed_mean(grp, ag, "captured"))
                if s:
                    ledger[f"captured.{fam}.{kind}.{ag}"] = s

    # ---- available gap and a_star coherence ------------------------------ #
    for fam in FAMILIES:
        gaps, b1, b2 = [], [], []
        for c in select(cells, family=fam, strength=MAIN_STRENGTH):
            bc = c.get("baselines_const")
            if not isinstance(bc, dict):
                continue
            g = bc.get("available_gap")
            if g is None:
                continue
            gaps.append(float(g))
            b1.append(float(bc.get("best_single_const")))
            b2.append(float(bc.get("best_per_task_const")))
        if gaps:
            ledger[f"gap.{fam}.available"] = stat(gaps)
            ledger[f"gap.{fam}.best_single_const"] = stat(b1)
            ledger[f"gap.{fam}.best_per_task_const"] = stat(b2)

    # ---- paired contrasts that carry the boundary-condition claim --------- #
    # Paired within seed, so the reported sd is the across-seed spread of the
    # difference itself rather than the spread of two independent means.
    for fam in FAMILIES:
        grp = [c for c in select(cells, family=fam, strength=MAIN_STRENGTH)
               if c["drift_kind"] in NON_STATIONARY]
        for field, tag in (("captured_post", "post"), ("captured", "full")):
            s = paired_contrast(grp, "meta_online", "meta_frozen", field)
            if s:
                ledger[f"contrast.online_frozen.{fam}.{tag}"] = s
            s = paired_contrast(grp, "meta_oracle", "meta_online", field)
            if s:
                ledger[f"contrast.oracle_online.{fam}.{tag}"] = s
            s = paired_contrast(grp, "meta_online", "robust", field)
            if s:
                ledger[f"contrast.online_robust.{fam}.{tag}"] = s
        # and per drift kind, for the heterogeneity figure
        for kind in NON_STATIONARY:
            gk = select(cells, family=fam, kind=kind, strength=MAIN_STRENGTH)
            s = paired_contrast(gk, "meta_online", "meta_frozen")
            if s:
                ledger[f"contrast.online_frozen.{fam}.{kind}"] = s

    # ---- raw returns, used wherever a ratio would be ill-conditioned ------- #
    for fam in FAMILIES:
        grp = [c for c in select(cells, family=fam, strength=MAIN_STRENGTH)
               if c["drift_kind"] in NON_STATIONARY]
        for ag in AGENTS:
            s = stat(per_seed_mean(grp, ag, "post_return_mean"))
            if s:
                ledger[f"post_return.{fam}.{ag}"] = s
            s = stat(per_seed_mean(grp, ag, "regret"))
            if s:
                ledger[f"regret.{fam}.{ag}"] = s
            s = stat(per_seed_mean(grp, ag, "adaptation_steps"))
            if s:
                ledger[f"adaptation_steps.{fam}.{ag}"] = s

    # ---- gating ---------------------------------------------------------- #
    for fam, rows_all in gates.items():
        rows = [r for r in rows_all
                if r.get("drift_kind") in NON_STATIONARY
                and r.get("strength") == MAIN_STRENGTH]
        if not rows:
            rows = rows_all
        for sig in ("consistency", "entropy", "variance"):
            vals = [r.get(f"auroc_{sig}") for r in rows]
            s = stat(vals)
            if s:
                ledger[f"auroc.{sig}.{fam}"] = s
            los, his = [], []
            for r in rows:
                ci = (r.get("ci") or {}).get(sig, {}).get("auroc")
                if ci and len(ci) == 2 and not any(x is None for x in ci):
                    try:
                        if not (np.isnan(ci[0]) or np.isnan(ci[1])):
                            los.append(float(ci[0]))
                            his.append(float(ci[1]))
                    except TypeError:
                        pass
            if los:
                ledger[f"auroc.{sig}.{fam}.ci"] = {
                    "lo_mean": float(np.mean(los)), "hi_mean": float(np.mean(his)),
                    "n": len(los)}
        pos = [r.get("pos_rate") for r in rows]
        s = stat(pos)
        if s:
            ledger[f"gate.pos_rate.{fam}"] = s
        ns = [r.get("n_steps") for r in rows]
        s = stat(ns)
        if s:
            ledger[f"gate.n_steps.{fam}"] = s
        # the unrestricted version, for completeness
        for sig in ("consistency", "entropy", "variance"):
            s = stat([r.get(f"auroc_{sig}") for r in rows_all])
            if s:
                ledger[f"auroc.{sig}.{fam}.all_cells"] = s
        s = stat([r.get("pos_rate") for r in rows_all])
        if s:
            ledger[f"gate.pos_rate.{fam}.all_cells"] = s

    # ---- passive identifiability index ----------------------------------- #
    for fam, rows in piis.items():
        for field, name in (("accuracy", "accuracy"), ("chance", "chance"),
                            ("best_return_r2", "r2"),
                            ("continuous_gap", "gap"),
                            ("macro_f1", "macro_f1")):
            s = stat([r.get(field) for r in rows])
            if s:
                ledger[f"pii.{name}.{fam}"] = s

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(ledger, indent=2, sort_keys=True))

    # ---- human-readable summary ------------------------------------------ #
    print(f"cells {len(cells)} | seeds {ledger['_provenance']['n_seeds']} | "
          f"families {ledger['_provenance']['families']}")
    print(f"\nwrote {OUT.relative_to(ROOT)}  ({len(ledger)} entries)")

    print("\ncaptured, main aggregate (5 non-stationary kinds, strength 1.0)")
    hdr = f"  {'family':16s}" + "".join(f"{a:>13s}" for a in
                                        ("meta_online", "meta_frozen", "meta_oracle", "robust"))
    print(hdr)
    for fam in FAMILIES:
        row = f"  {fam:16s}"
        for ag in ("meta_online", "meta_frozen", "meta_oracle", "robust"):
            s = ledger.get(f"captured.{fam}.{ag}")
            row += f"{s['mean']:+8.3f}       " if s else f"{'--':>13s}"
        print(row)

    print("\navailable gap by family")
    for fam in FAMILIES:
        s = ledger.get(f"gap.{fam}.available")
        print(f"  {fam:16s} {s['mean']:8.2f}  (n={s['n']})" if s else f"  {fam:16s} --")

    print("\ngating AUROC")
    for fam in FAMILIES:
        row = f"  {fam:16s}"
        for sig in ("consistency", "entropy", "variance"):
            s = ledger.get(f"auroc.{sig}.{fam}")
            row += f"{sig[:4]}={s['mean']:.3f}  " if s else f"{sig[:4]}=  --   "
        print(row)

    print("\npaired contrasts (post-drift window, paired within seed)")
    for fam in FAMILIES:
        row = f"  {fam:16s}"
        for key, lab in (("contrast.online_frozen", "online-frozen"),
                         ("contrast.oracle_online", "oracle-online"),
                         ("contrast.online_robust", "online-robust")):
            s = ledger.get(f"{key}.{fam}.post")
            row += f"{lab}={s['mean']:+.3f}±{s['sd']:.2f}  " if s else f"{lab}=  --  "
        print(row)

    print("\nPII")
    for fam in FAMILIES:
        a = ledger.get(f"pii.accuracy.{fam}")
        c = ledger.get(f"pii.chance.{fam}")
        r2 = ledger.get(f"pii.r2.{fam}")
        g = ledger.get(f"pii.gap.{fam}")
        if a:
            print(f"  {fam:16s} acc={a['mean']:.3f} chance={c['mean']:.3f} "
                  f"r2={r2['mean']:+.3f} gap={g['mean']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
