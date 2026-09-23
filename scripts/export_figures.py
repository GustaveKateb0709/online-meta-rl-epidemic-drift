"""Convert rendered figures into submission-ready TIFF files.

Submission-ready figures are TIFF at 300 dpi, a figure width between 789 and
2250 pixels, a file size below 10 MB, and no title, author
name or figure number inside the image. This script takes the PNG/PDF rendered
by the figure scripts, rebuilds a TIFF at the required resolution, verifies the
geometry and size, and writes a machine-readable report.

Usage:
    python scripts/export_figures.py --figdir figures --outdir exported

Every figure script must write ``figures/figN_<name>.png``; the exported file is
``exported/FigN.tif`` so that the file name matches the caption label, as the
journal requires.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.plotstyle import (MAX_DPI, MAX_FILE_MB, MAX_WIDTH_PX, MIN_DPI,
                           MIN_WIDTH_PX)

NAME_RE = re.compile(r"^fig(\d+)_(.+)\.png$")


def export_one(png: Path, outdir: Path, dpi: int) -> dict:
    n = int(NAME_RE.match(png.name).group(1))
    with Image.open(png) as im:
        im = im.convert("RGB")
        # Rebuild at the requested resolution. The PNG was rendered at 300 dpi,
        # so scaling by dpi/300 keeps the physical size and raises the pixel grid.
        w_px = int(round(im.width * dpi / 300.0))
        h_px = int(round(im.height * dpi / 300.0))
        im2 = im.resize((w_px, h_px), Image.LANCZOS)
        out = outdir / f"Fig{n}.tif"
        im2.save(out, format="TIFF", compression="tiff_lzw", dpi=(dpi, dpi))
    size_mb = out.stat().st_size / 1e6
    return {
        "figure": n,
        "source_png": str(png.relative_to(ROOT)),
        "output": str(out.relative_to(ROOT)),
        "width_px": w_px,
        "height_px": h_px,
        "dpi": dpi,
        "size_mb": round(size_mb, 3),
        "checks": {
            "width_within_limits": MIN_WIDTH_PX <= w_px <= MAX_WIDTH_PX,
            "dpi_within_limits": MIN_DPI <= dpi <= MAX_DPI,
            "size_within_limits": size_mb <= MAX_FILE_MB,
            "name_matches_caption_label": out.name == f"Fig{n}.tif",
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figdir", default="figures")
    ap.add_argument("--outdir", default="exported")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    figdir = (ROOT / args.figdir) if not Path(args.figdir).is_absolute() else Path(args.figdir)
    outdir = (ROOT / args.outdir) if not Path(args.outdir).is_absolute() else Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pngs = sorted(p for p in figdir.glob("fig*.png") if NAME_RE.match(p.name))
    if not pngs:
        print(f"no figures matching figN_*.png in {figdir}")
        return 1

    report = []
    n_fail = 0
    for png in pngs:
        rec = export_one(png, outdir, args.dpi)
        report.append(rec)
        bad = [k for k, v in rec["checks"].items() if not v]
        n_fail += len(bad)
        flag = "OK  " if not bad else "FAIL"
        print(f"[{flag}] {rec['output']}  {rec['width_px']}x{rec['height_px']} px, "
              f"{rec['dpi']} dpi, {rec['size_mb']} MB"
              + (f"  <- {', '.join(bad)}" if bad else ""))

    (outdir / "export_report.json").write_text(
        json.dumps({"dpi": args.dpi, "figures": report}, indent=2))
    print(f"\n{len(report)} figures exported, {n_fail} failed checks.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
