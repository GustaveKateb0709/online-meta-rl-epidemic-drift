"""Fetch real public respiratory-pathogen surveillance data (no credentials).

Sources (all public, key-free, verified reachable from this host):

  1. CMU Delphi Epidata - FluView ILINet weekly, national + selected states.
       base: https://api.delphi.cmu.edu/epidata/
  2. WHO FluNet official CSV (global, country-week laboratory surveillance).
       https://xmart-api-public.who.int/FLUMART/VIW_FNT?$format=csv
  3. OWID COVID-19 daily series (used as a second pathogen for robustness).
       https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv

Design rules
------------
* No absolute paths: everything is derived from Path(__file__).
* Idempotent: if the output already exists and its sha256 matches the sidecar
  record, the step is skipped.
* Every network call retries with exponential backoff (raw.githubusercontent.com
  is intermittently slow on this host).
* Single output CSV is kept under 20 MB (WHO FluNet is trimmed on ingest).
* Prints the REAL number of rows matched at every step.

Outputs
-------
  data/raw/fluview_national_weekly.csv      (+ .meta.json)
  data/raw/fluview_states_weekly.csv        (+ .meta.json)
  data/raw/who_flunet_weekly_trimmed.csv    (+ .meta.json)
  data/raw/owid_covid_trimmed.csv           (+ .meta.json)
  results/calibration/fetch_log.json
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CAL = ROOT / "results" / "calibration"
RAW.mkdir(parents=True, exist_ok=True)
CAL.mkdir(parents=True, exist_ok=True)

DELPHI_BASE = "https://api.delphi.cmu.edu/epidata/fluview/"
FLUNET_URL = "https://xmart-api-public.who.int/FLUMART/VIW_FNT?$format=csv"
OWID_URL = (
    "https://raw.githubusercontent.com/owid/covid-19-data/"
    "master/public/data/owid-covid-data.csv"
)

# Time windows are chosen to give long, contiguous weekly series while keeping
# every stored file well below 20 MB.
FLUVIEW_EPIWEEKS = "201001-202452"
FLUVIEW_STATES = ["ca", "tx", "ny", "fl", "il", "wa"]

FLUNET_KEEP_COLS = [
    "WHOREGION", "COUNTRY_CODE", "ISO_YEAR", "ISO_WEEK", "ISO_WEEKSTARTDATE",
    "INF_ALL", "INF_A", "INF_B", "SPEC_PROCESSED_NB", "RSV",
]
FLUNET_MIN_YEAR = 2015

OWID_KEEP_COLS = [
    "location", "date", "new_cases", "new_cases_smoothed", "total_cases",
    "population",
]
OWID_KEEP_LOCATIONS = [
    "World", "United States", "China", "Japan", "Germany", "Brazil",
    "India", "United Kingdom",
]
OWID_MIN_DATE = "2020-01-01"

USER_AGENT = "paper17-public-data/1.0 (academic reproducibility)"
FETCH_LOG: list[dict] = []


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def http_bytes(url: str, retries: int = 5, timeout: int = 240) -> bytes:
    """Download a URL to bytes with retries and exponential backoff."""
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001 - report and retry
            last_err = exc
            wait = min(2 ** attempt, 30)
            print(f"    attempt {attempt}/{retries} failed ({exc}); retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"download failed after {retries} attempts: {url} ({last_err})")


def http_json(url: str, retries: int = 5, timeout: int = 240) -> dict:
    return json.loads(http_bytes(url, retries=retries, timeout=timeout).decode("utf-8"))


def sidecar_path(out: Path) -> Path:
    return out.with_suffix(out.suffix + ".meta.json")


def is_fresh(out: Path) -> bool:
    """True if output exists and matches the sha256 recorded in its sidecar."""
    meta_p = sidecar_path(out)
    if not (out.exists() and meta_p.exists()):
        return False
    try:
        meta = json.loads(meta_p.read_text())
    except Exception:  # noqa: BLE001
        return False
    return meta.get("sha256") == _sha256(out)


def write_output(df: pd.DataFrame, out: Path, source_url: str, note: str,
                 extra: dict | None = None) -> dict:
    df.to_csv(out, index=False)
    size_mb = out.stat().st_size / 1e6
    meta = {
        "source_url": source_url,
        "downloaded_at_utc": _now(),
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns),
        "size_mb": round(size_mb, 3),
        "sha256": _sha256(out),
        "note": note,
    }
    if extra:
        meta.update(extra)
    sidecar_path(out).write_text(json.dumps(meta, indent=2))
    if size_mb > 20.0:
        print(f"    WARNING: {out.name} is {size_mb:.1f} MB (>20 MB budget)")
    return meta


def record(step: str, meta: dict, skipped: bool) -> None:
    FETCH_LOG.append({"step": step, "skipped": skipped, **meta})


# --------------------------------------------------------------------------- #
# source 1: Delphi FluView
# --------------------------------------------------------------------------- #
def fetch_fluview(regions: list[str], out: Path, label: str) -> None:
    url = f"{DELPHI_BASE}?regions={','.join(regions)}&epiweeks={FLUVIEW_EPIWEEKS}"
    if is_fresh(out):
        print(f"  [{label}] up to date, skipped (sha256 match)")
        record(label, json.loads(sidecar_path(out).read_text()), True)
        return

    print(f"  [{label}] GET {url}")
    payload = http_json(url)
    if payload.get("result") != 1:
        raise RuntimeError(f"Delphi error for {label}: {payload.get('message')}")
    rows = payload.get("epidata", [])
    print(f"  [{label}] API returned {len(rows)} weekly records")

    df = pd.DataFrame(rows)
    # Keep analysis columns only; num_age_* are not used downstream.
    keep = [c for c in ["region", "epiweek", "release_date", "issue", "lag",
                        "ili", "wili", "num_ili", "num_patients",
                        "num_providers"] if c in df.columns]
    df = df[keep]
    df["epiweek"] = df["epiweek"].astype(int)
    df["year"] = df["epiweek"] // 100
    df["week"] = df["epiweek"] % 100
    df = df.sort_values(["region", "epiweek"]).reset_index(drop=True)

    n_regions = df["region"].nunique()
    span = f"{df['epiweek'].min()}-{df['epiweek'].max()}"
    meta = write_output(
        df, out, "https://api.delphi.cmu.edu/epidata/fluview/", note=(
            "Delphi Epidata FluView ILINet weekly; ili and wili are percentages "
            "of outpatient visits; num_ili/num_patients are raw counts."
        ),
        extra={"regions": sorted(df["region"].unique().tolist()),
               "n_regions": int(n_regions), "epiweek_span": span,
               "epiweeks_requested": FLUVIEW_EPIWEEKS},
    )
    print(f"  [{label}] kept {len(df)} rows across {n_regions} regions, "
          f"span {span}, {meta['size_mb']} MB")
    record(label, meta, False)


# --------------------------------------------------------------------------- #
# source 2: WHO FluNet (streamed and trimmed to stay under 20 MB)
# --------------------------------------------------------------------------- #
def fetch_flunet(out: Path) -> None:
    label = "who_flunet"
    if is_fresh(out):
        print(f"  [{label}] up to date, skipped (sha256 match)")
        record(label, json.loads(sidecar_path(out).read_text()), True)
        return

    print(f"  [{label}] GET {FLUNET_URL} (streaming to temp file)")
    tmp = RAW / "_flunet_full_tmp.csv"
    if tmp.exists():
        tmp.unlink()
    last_err = None
    for attempt in range(1, 6):
        try:
            req = urllib.request.Request(FLUNET_URL, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=300) as resp, open(tmp, "wb") as f:
                shutil.copyfileobj(resp, f, length=1 << 20)
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            wait = min(2 ** attempt, 30)
            print(f"    attempt {attempt}/5 failed ({exc}); retry in {wait}s")
            time.sleep(wait)
    else:
        raise RuntimeError(f"FluNet download failed: {last_err}")

    raw_mb = tmp.stat().st_size / 1e6
    print(f"  [{label}] raw download {raw_mb:.1f} MB; trimming")
    df = pd.read_csv(tmp, usecols=lambda c: c in FLUNET_KEEP_COLS,
                     low_memory=False)
    tmp.unlink()  # never persist the oversized raw file

    n_raw = len(df)
    df = df[df["ISO_YEAR"] >= FLUNET_MIN_YEAR].copy()
    df = df.rename(columns={"ISO_WEEKSTARTDATE": "ISO_WEEKSTARTDATE"})
    df = df.sort_values(["WHOREGION", "COUNTRY_CODE", "ISO_YEAR", "ISO_WEEK"])
    df = df.reset_index(drop=True)

    n_regions = df["WHOREGION"].nunique()
    n_countries = df["COUNTRY_CODE"].nunique()
    span = f"{df['ISO_YEAR'].min()}-{df['ISO_YEAR'].max()}"
    meta = write_output(
        df, out, FLUNET_URL, note=(
            "WHO FluNet country-week laboratory surveillance, trimmed to "
            "ISO_YEAR>=%d and 10 analysis columns; INF_ALL = all influenza "
            "positive specimens, SPEC_PROCESSED_NB = specimens processed." %
            FLUNET_MIN_YEAR
        ),
        extra={"raw_n_rows_before_trim": int(n_raw),
               "raw_size_mb": round(raw_mb, 3),
               "n_who_regions": int(n_regions),
               "n_countries": int(n_countries), "year_span": span},
    )
    print(f"  [{label}] raw {n_raw} rows -> kept {len(df)} rows "
          f"({n_countries} countries, {n_regions} WHO regions, {span}), "
          f"{meta['size_mb']} MB")
    record(label, meta, False)


# --------------------------------------------------------------------------- #
# source 3: OWID COVID-19 (second pathogen, robustness)
# --------------------------------------------------------------------------- #
def fetch_owid(out: Path) -> None:
    label = "owid_covid"
    if is_fresh(out):
        print(f"  [{label}] up to date, skipped (sha256 match)")
        record(label, json.loads(sidecar_path(out).read_text()), True)
        return

    # The full OWID file is ~98 MB and raw.githubusercontent.com is slow on this
    # host, so we stream it line by line and keep only the rows we need. The
    # oversized raw file is never written to disk.
    print(f"  [{label}] GET {OWID_URL} (streaming + on-the-fly filter)")
    loc_set = set(OWID_KEEP_LOCATIONS)
    last_err = None
    rows = None
    for attempt in range(1, 4):
        try:
            req = urllib.request.Request(OWID_URL, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=1800) as resp:
                text = io.TextIOWrapper(resp, encoding="utf-8", newline="")
                reader = csv.DictReader(text)
                fnames = reader.fieldnames or []
                keep = [c for c in OWID_KEEP_COLS if c in fnames]
                rows = []
                n_total = 0
                for row in reader:
                    n_total += 1
                    if row.get("location") in loc_set and row.get("date", "") >= OWID_MIN_DATE:
                        rows.append({k: row.get(k) for k in keep})
            print(f"  [{label}] streamed {n_total} raw rows, matched {len(rows)}")
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            wait = min(2 ** attempt * 5, 30)
            print(f"    attempt {attempt}/3 failed ({exc}); retry in {wait}s")
            time.sleep(wait)
    if rows is None:
        raise RuntimeError(f"OWID download failed: {last_err}")

    df = pd.DataFrame(rows)
    n_raw = n_total
    df = df.sort_values(["location", "date"]).reset_index(drop=True)

    meta = write_output(
        df, out, OWID_URL, note=(
            "OWID COVID-19 daily, trimmed to %d locations and date>=%s; "
            "new_cases are daily confirmed cases, new_cases_smoothed is the "
            "7-day average." % (len(OWID_KEEP_LOCATIONS), OWID_MIN_DATE)
        ),
        extra={"raw_n_rows_before_trim": int(n_raw),
               "locations": OWID_KEEP_LOCATIONS,
               "date_min": OWID_MIN_DATE},
    )
    print(f"  [{label}] raw {n_raw} rows -> kept {len(df)} rows "
          f"({len(OWID_KEEP_LOCATIONS)} locations), {meta['size_mb']} MB")
    record(label, meta, False)


# --------------------------------------------------------------------------- #
def main() -> int:
    t0 = time.time()
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1]

    def wanted(tag: str) -> bool:
        return only is None or only == tag

    if wanted("fluview_nat"):
        print("== Step 1: Delphi FluView national ==")
        fetch_fluview(["nat"], RAW / "fluview_national_weekly.csv", "fluview_nat")

    if wanted("fluview_states"):
        print("== Step 2: Delphi FluView states ==")
        fetch_fluview(FLUVIEW_STATES, RAW / "fluview_states_weekly.csv",
                      "fluview_states")

    if wanted("flunet"):
        print("== Step 3: WHO FluNet trimmed ==")
        fetch_flunet(RAW / "who_flunet_weekly_trimmed.csv")

    if wanted("owid"):
        print("== Step 4: OWID COVID trimmed ==")
        fetch_owid(RAW / "owid_covid_trimmed.csv")

    log = {
        "generated_at_utc": _now(),
        "elapsed_sec": round(time.time() - t0, 1),
        "steps": FETCH_LOG,
    }
    (CAL / "fetch_log.json").write_text(json.dumps(log, indent=2))
    print(f"\nWrote {CAL / 'fetch_log.json'}")
    print(f"Done in {log['elapsed_sec']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
