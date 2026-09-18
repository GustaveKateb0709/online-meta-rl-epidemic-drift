# Online regime inference for meta-reinforcement learning in epidemic intervention

Code and data accompanying the manuscript *"Adapting to drift you cannot see:
online regime inference for meta-reinforcement learning in epidemic
intervention"*.

The study asks a single question: **when does explicit online inference of a
latent regime actually pay off in sequential public-health decision making?**

It answers this by holding the decision-relevant variation fixed and varying only
whether that variation can be recovered from passive observation. Three
environment families are compared:

| family | decision-relevant variation | recoverable without intervening |
|---|---|---|
| `passive` | absent (action channel fixed) | yes |
| `linked` | present | yes (by construction, exactly) |
| `action_revealed` | present | no (identical observations until the agent acts) |

Everything runs on CPU. The reference machine is a laptop with 8 GB of unified
memory; no GPU is used.

## Layout

```
src/            environment, meta-RL agent, baselines, gating, identifiability
scripts/        the reproduction pipeline, numbered in run order
experiments/    single-family experiment entry point
results/        machine-readable outputs (paper_numbers.json, robust_contrasts.json)
data/raw/       public surveillance data with per-file provenance metadata
figures/        rendered figures (PNG)
source_data/    plotting source data, one CSV per figure
manifest.json   file inventory with MD5 checksums
```

## Environment

`src/env.py` implements a batched, pure-NumPy seasonal SEIRS-type transmission
model with a reporting layer and three levels of intervention intensity. The
regime vector has seven components (baseline transmission, progression, recovery,
reporting rate, seasonal amplitude, behavioural rebound, direct efficacy). Only
four quantities are observable: the reported case level, its week-on-week change,
a calendar index, and the previous action. The regime itself never enters the
observation.

The intervention acts through two opposing channels. It directly lowers
transmissibility and simultaneously triggers a behavioural rebound that raises
contacts. When the rebound dominates, intervening raises transmission. Whether it
does depends on parameters that the `action_revealed` family reveals only through
the response to the agent's own actions.

## Reproduction

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. verify the environment (22 mechanistic assertions)
python scripts/validate_env.py
python scripts/validate_families.py

# 2. fetch and calibrate the public surveillance data
python scripts/01_fetch_public_data.py
python scripts/02_drift_empirics.py
python scripts/03_build_contracts.py

# 3. run the experiment matrix (long; resumable, per-cell output)
python scripts/run_matrix.py \
  --families passive,linked,action_revealed \
  --kinds stationary,gradual_beta,abrupt_beta,gradual_rho,endogenous_beh,abrupt_eps \
  --strengths 1.0,2.0 --seeds 10,11,12,13,14 \
  --iterations 1000 --n-tasks 8 --horizon 156 --tasks-per-kind 16 \
  --threads 4 --with-frozen --with-oracle --outdir results/main

# 4. aggregate into the numeric ledger used by the manuscript
python scripts/analyze_matrix.py

# 5. figures
python scripts/make_figures.py
python scripts/export_figures.py --figdir figures --outdir exported
```

`run_matrix.py` skips any cell whose JSON already exists and parses, so the
matrix can be interrupted and resumed without losing completed work. Each cell
draws its task batch from a deterministic per-cell generator, so results do not
depend on execution order.

## Data

The external calibration uses only sources that can be downloaded directly,
without registration or authorisation:

- **Delphi Epidata API** (Carnegie Mellon University) for US ILINet influenza-like
  illness, national and by state.
- **WHO FluNet** for global influenza virological surveillance.
- **Our World in Data** COVID-19 series.

Raw downloads are written to `data/raw/` with a sidecar JSON recording the source
URL, download timestamp, row count and checksum. `artifacts/data_contract.json`
documents every field; `artifacts/sample_construction.json` records the
inclusion flow.

Sources that require credentialing (for example MIMIC-IV, eICU, UK Biobank) are
deliberately not used.

## Notes

- Figures are rendered at the width required by the journal (789-2250 px at
  300 dpi) and `scripts/export_figures.py` fails loudly if a figure is outside
  the allowed geometry or size.
- Every quantitative claim in the manuscript is traceable to an entry in
  `results/paper_numbers.json`, produced by `scripts/analyze_matrix.py`.

## License

MIT. See `LICENSE`.
