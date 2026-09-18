"""Rebuild ``manuscript/supplementary/s3_table_performance.md``.

Main table: read from the aggregated ledger produced by ``analyze_matrix.py``
(25 seeds; per-seed value = mean over the five non-stationary drift kinds at
strength 1.0; parenthesised value = across-seed standard deviation). All
ledger keys used here are computed over that same cell window.

Paired contrasts: differences are computed within each seed and summarised
across seeds. Following the reporting rule stated in Materials and Methods,
the passive family is reported in raw post-drift return units and never as a
captured fraction; the linked and action_revealed rows are in captured units.
The per-seed contrast values are recomputed here directly from the per-cell
files so that medians and per-seed sign counts can be reported alongside the
mean and standard deviation.

Run:  python scripts/make_s3_table.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "results" / "paper_numbers.json"
MAIN = ROOT / "results" / "main"
OUT = ROOT / "manuscript" / "supplementary" / "s3_table_performance.md"

AGENTS = ["meta_online", "meta_frozen", "meta_oracle", "robust", "transfer",
          "retrain", "const0", "const1", "const2"]
FAMILIES = ["passive", "linked", "action_revealed"]
KINDS = ["gradual_beta", "abrupt_beta", "gradual_rho", "endogenous_beh", "abrupt_eps"]
COLUMNS = [("Post-drift return", "post_return"),
           ("Captured (full)", "captured"),
           ("Captured (post-drift)", "captured_post"),
           ("Regret", "regret"),
           ("Adaptation steps", "adaptation_steps")]


def ledger_stat(ledger, fam, agent, field):
    entry = ledger.get(f"{field}.{fam}.{agent}")
    if not isinstance(entry, dict) or "mean" not in entry:
        return None, None
    return entry["mean"], entry.get("sd")


def load_cells():
    cells = {}
    for fp in sorted(MAIN.glob("*.json")):
        d = json.loads(fp.read_text())
        if d.get("family") in FAMILIES and d.get("drift_kind") in KINDS:
            try:
                if float(d.get("strength")) == 1.0:
                    cells.setdefault((d["family"], int(d["seed"])), {})[d["drift_kind"]] = d
            except (TypeError, ValueError):
                continue
    return cells


def contrast_row(cells, seeds, fam, a, b):
    """Per-seed paired contrast for one family; passive uses raw post-drift
    return units, the others captured units. Returns
    (mean, sd, median, n_positive, n) or Nones."""
    field = "post_return_mean" if fam == "passive" else "captured_post"
    vals = []
    for seed in seeds:
        diffs = []
        for kind in KINDS:
            cell = cells.get((fam, seed), {}).get(kind)
            if cell is None:
                continue
            ev = cell["eval"]
            x = ev.get(a, {}).get(field)
            y = ev.get(b, {}).get(field)
            if x is None or y is None:
                continue
            diffs.append(x - y)
        if diffs:
            vals.append(float(np.mean(diffs)))
    if not vals:
        return None, None, None, None, None
    x = np.array(vals, dtype=float)
    # sd uses the population convention (ddof=0), matching the ledger's stat().
    return (float(x.mean()), float(x.std(ddof=0)), float(np.median(x)),
            int((x > 0).sum()), int(x.size))


def fmt(mean, sd):
    if mean is None:
        return "--"
    if sd is None:
        return f"{mean:.2f}"
    return f"{mean:.2f} ({sd:.2f})"


def signed(value, decimals=3):
    if value is None:
        return "--"
    return f"{'+' if value >= 0 else ''}{value:.{decimals}f}"


def main():
    ledger = json.loads(LEDGER.read_text())
    cells = load_cells()
    seeds = sorted({s for (_, s) in cells})

    lines = []
    lines.append("# S3 Table. Return, regret and adaptation speed by family and agent")
    lines.append("")
    lines.append(
        "Values are means over the five non-stationary drift kinds at strength 1.0, "
        "with the across-seed standard deviation in parentheses (25 seeds). "
        "`captured` is the fraction of the available gap obtained, defined as "
        "(return - best single constant action) / (best per-task constant action - "
        "best single constant action); it is reported only for the linked and "
        "action_revealed families, because the available gap of the passive family "
        "is close to zero (2.53 under the experimental protocol) and the ratio is "
        "numerically unstable there. For the passive family the raw post-drift "
        "return is given instead. `regret` and `adaptation_steps` follow the "
        "definitions in Materials and Methods Section 4.8.")
    lines.append("")
    lines.append("| Family | Agent | Post-drift return | Captured (full) | "
                 "Captured (post-drift) | Regret | Adaptation steps |")
    lines.append("|---|---|---|---|---|---|---|")
    for fam in FAMILIES:
        for agent in AGENTS:
            row = [f"`{fam}`", f"`{agent}`"]
            for _label, field in COLUMNS:
                mean, sd = ledger_stat(ledger, fam, agent, field)
                if field in ("captured", "captured_post") and fam == "passive":
                    row.append("--")
                else:
                    row.append(fmt(mean, sd))
            lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## Paired contrasts")
    lines.append("")
    lines.append(
        "Differences are computed within each seed and then summarised across "
        "seeds, so the standard deviation is the across-seed spread of the "
        "difference itself. The post-drift window is the part of the episode "
        "after drift onset. Consistent with the reporting rule for the passive "
        "family, the passive rows are in raw post-drift return units; the linked "
        "and action_revealed rows are in captured units. The across-seed "
        "distribution of the captured-unit contrasts is heavy-tailed because a "
        "small number of cells draw a near-degenerate available gap, so the "
        "median and the number of positive seeds are reported alongside the mean "
        "and standard deviation.")
    lines.append("")
    lines.append("| Contrast | Family | Units | Mean | SD | Median | Positive seeds | n (seeds) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for label, a, b in [("Online - frozen", "meta_online", "meta_frozen"),
                        ("Oracle - online", "meta_oracle", "meta_online"),
                        ("Online - non-adapting", "meta_online", "robust")]:
        for fam in FAMILIES:
            mean, sd, median, npos, n = contrast_row(cells, seeds, fam, a, b)
            units = "post-drift return" if fam == "passive" else "captured"
            lines.append(f"| {label} | `{fam}` | {units} | {signed(mean)} | "
                         f"{f'{sd:.3f}' if sd is not None else '--'} | {signed(median)} | "
                         f"{npos if npos is not None else '--'}/{n if n is not None else '--'} | "
                         f"{n if n is not None else '--'} |")
    lines.append("")

    OUT.write_text("\n".join(lines))
    print(f"[ok] wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
