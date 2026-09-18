"""Resumable batch runner for the regime-drift meta-RL matrix.

The formal matrix is a multi-hour background job, so this runner is built to be
interruptible and restartable:

* one JSON per (family, drift_kind, strength, seed) cell, written under
  ``results/main`` with a parseable name ``{family}_{kind}_s{strength}_seed{seed}.json``;
* on startup every already-written, parseable cell is skipped (``--force`` reruns);
* every cell is wrapped in try/except; a failure writes a ``.failed`` marker and
  the run continues, with a failure summary printed at the end;
* one progress line per completed cell is appended to
  ``artifacts/matrix_progress.txt`` (flushed immediately).

Hyper-parameters live in the config and are echoed into every cell JSON, so each
cell is self-describing. Nothing here trains unless the run is started explicitly
(``--dry-run`` prints the plan and exits).

Example
-------
    python -u scripts/run_matrix.py --dry-run
    python -u scripts/run_matrix.py --seeds 10,11 --iterations 1000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.run_main import (DEFINITIONS, _meta_metrics,  # noqa: E402
                                  _run_gate)
from src.baselines import (ADAPTATION_STEPS_DEFINITION, AGENT_NAMES,  # noqa: E402
                           ConstantController, RetrainBaseline, RobustBaseline,
                           TransferBaseline, attach_captured,
                           constant_baseline_block, eval_metrics, rollout_episode)
from src.commitment import wrap_cfg  # noqa: E402
from src.common import DEFAULT_CONFIG, provenance, seed_everything, set_threads  # noqa: E402
from src.env import (DRIFT_KINDS, FAMILIES, RegimeDriftEpidemicEnv,  # noqa: E402
                     make_drift)
from src.identifiability import PII_DEFINITION, passive_identifiability_index  # noqa: E402
from src.meta_rl import MetaRLAgent, train_meta  # noqa: E402

ARTIFACTS = ROOT / "artifacts"
RESULTS_ROOT = ROOT / "results"
_KINDS_ALL = ("stationary",) + tuple(DRIFT_KINDS)


# ---------------------------------------------------------------------- #
# config / cell bookkeeping
# ---------------------------------------------------------------------- #
def build_cfg(args, family: str) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(n_tasks=args.n_tasks, horizon=args.horizon, iterations=args.iterations,
               z_dim=args.z_dim, ctx_len=args.ctx_len, hidden=args.hidden,
               ppo_epochs=args.ppo_epochs, n_minibatch=args.n_minibatch,
               z_grad=args.z_grad, family=family,
               commitment=args.commitment,
               commitment_cycle_len=(args.commitment_cycle_len or None),
               commitment_warmup=args.commitment_warmup)
    return cfg


def cell_name(family: str, kind: str, strength: float, seed: int) -> str:
    return f"{family}_{kind}_s{strength:g}_seed{seed}"


def cell_path(outdir: Path, family: str, kind: str, strength: float, seed: int) -> Path:
    return outdir / f"{cell_name(family, kind, strength, seed)}.json"


def cell_rng(seed: int, family: str, kind: str, strength: float) -> np.random.Generator:
    """Deterministic per-cell RNG, independent of the order cells are run in."""
    fam = list(FAMILIES).index(family) if family in FAMILIES else 99
    ki = _KINDS_ALL.index(kind) if kind in _KINDS_ALL else 99
    return np.random.default_rng([int(seed), int(fam), int(ki), int(round(strength * 1000))])


def enumerate_cells(families, kinds, strengths, seeds):
    """All (family, kind, strength, seed) cells; stationary keeps strength 1 only."""
    out = []
    for family in families:
        for kind in kinds:
            for strength in strengths:
                if kind == "stationary" and abs(strength - 1.0) > 1e-9:
                    continue
                for seed in seeds:
                    out.append((family, kind, float(strength), int(seed)))
    return out


def cell_done(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        json.loads(path.read_text())
        return True
    except Exception:
        return False


def write_failed(path: Path, err: Exception) -> None:
    path.with_suffix(".failed").write_text(
        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n"
        + "".join(traceback.format_exception(type(err), err, err.__traceback__)))


def log_progress(progress_path: Path, line: str) -> None:
    with open(progress_path, "a") as fh:
        fh.write(line + "\n")
        fh.flush()


# ---------------------------------------------------------------------- #
# training and evaluation of one family/seed agent bundle
# ---------------------------------------------------------------------- #
def train_agents(family: str, seed: int, cfg: dict, args) -> dict:
    seed_everything(seed)
    meta = {}
    online = MetaRLAgent(dict(cfg, z_mode="online"))
    info = train_meta(online, seed, verbose=False)
    meta["meta_online"] = (online, "online")
    if args.with_frozen:
        frozen = MetaRLAgent(dict(cfg, z_mode="frozen"))
        train_meta(frozen, seed + 1, verbose=False)
        meta["meta_frozen"] = (frozen, "frozen")
    if args.with_oracle:
        oracle = MetaRLAgent(dict(cfg, z_mode="oracle"))
        train_meta(oracle, seed + 2, verbose=False)
        meta["meta_oracle"] = (oracle, "oracle")
    robust = RobustBaseline(cfg)
    robust.fit(seed)
    transfer = TransferBaseline(cfg, finetune_steps=args.finetune_steps)
    transfer.fit(seed)
    retrain = RetrainBaseline(cfg, budget_steps=args.retrain_budget,
                              n_replicas=args.retrain_replicas)
    return {"meta": meta, "robust": robust, "transfer": transfer, "retrain": retrain,
            "train_info": info, "n_params": online.n_params()}


def eval_cell(bundle: dict, cfg: dict, family: str, kind: str, strength: float,
              seed: int, args) -> dict:
    """Evaluate every agent on this cell; return the full cell record."""
    H = cfg["horizon"]
    n_eval = args.tasks_per_kind
    eval_env = wrap_cfg(RegimeDriftEpidemicEnv(n_envs=n_eval, horizon=H, seed=seed + 777), cfg)
    drift = make_drift(cell_rng(seed, family, kind, strength), kind, n_eval, H,
                       strength=strength, family=family)

    row = {}
    for name, (agent, z_mode) in bundle["meta"].items():
        row[name] = _meta_metrics(agent, eval_env, drift, seed + 11, z_mode)
    row["robust"] = bundle["robust"].evaluate(drift, eval_env, seed + 21)
    bundle["transfer"].finetune(drift, eval_env, seed + 31)
    row["transfer"] = bundle["transfer"].evaluate(drift, eval_env, seed + 41)
    row["transfer"]["finetune_steps"] = bundle["transfer"].finetune_steps
    row["transfer"]["total_interactions"] = bundle["transfer"]._interactions()
    row["retrain"] = bundle["retrain"].fit_and_evaluate(drift, seed + 51)
    row["retrain"]["budget_per_task"] = bundle["retrain"].budget_steps
    row["retrain"]["interactions_total"] = bundle["retrain"]._interactions()
    const_per_task = {}
    for a in (0, 1, 2):
        rew = rollout_episode(eval_env, ConstantController(a), drift, seed + 60 + a)
        row[f"const{a}"] = eval_metrics(rew, drift["onset"], H)
        const_per_task[a] = rew.sum(axis=0)

    best = max(row[n]["return_mean"] for n in AGENT_NAMES if n in row)
    best_post = max(row[n]["post_return_mean"] for n in AGENT_NAMES if n in row)
    orc = row.get("meta_oracle", {}).get("return_mean")
    orc_post = row.get("meta_oracle", {}).get("post_return_mean")
    for n in AGENT_NAMES:
        if n not in row:
            continue
        r = row[n]
        r["regret"] = float(best - r["return_mean"])
        r["regret_post"] = float(best_post - r["post_return_mean"])
        r["regret_vs_oracle"] = float(orc - r["return_mean"]) if orc is not None else None
        r["regret_vs_oracle_post"] = (float(orc_post - r["post_return_mean"])
                                      if orc_post is not None else None)

    # per-task constant baselines + the 'captured' primary criterion
    block = constant_baseline_block(const_per_task, n_eval)
    attach_captured(row, block)
    row["family"] = family
    row["drift_kind"] = kind
    row["strength"] = float(strength)
    row["baselines_const"] = block

    gate = _run_gate(bundle["meta"]["meta_online"][0], bundle["robust"].agent,
                     eval_env, [kind], strength, family, seed, cfg, args)
    pii = passive_identifiability_index(family=family, n_tasks=args.pii_tasks,
                                        horizon=min(H, 120), seed=seed)

    info = bundle["train_info"]
    return {
        "cell": cell_name(family, kind, strength, seed),
        "seed": int(seed), "family": family, "drift_kind": kind, "strength": float(strength),
        "config": {k: cfg[k] for k in sorted(cfg)},
        "threads": int(args.threads),
        "train": {"env_steps": int(info["env_steps"]),
                  "wall_seconds": float(info["wall_seconds"]),
                  "final_return": float(info["final_return"]),
                  "n_params": int(bundle["n_params"])},
        "eval": row,
        "gate": gate,
        "pii": pii,
        "definitions": DEFINITIONS,
        "pii_definition": PII_DEFINITION,
        "adaptation_steps_definition": ADAPTATION_STEPS_DEFINITION,
        "provenance": provenance(),
    }


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Resumable regime-drift meta-RL matrix runner")
    p.add_argument("--families", default=",".join(FAMILIES))
    p.add_argument("--kinds", default="stationary," + ",".join(DRIFT_KINDS))
    p.add_argument("--strengths", default="1.0,2.0")
    p.add_argument("--seeds", default="10,11,12,13,14,15,16,17,18,19")
    p.add_argument("--outdir", default=str(RESULTS_ROOT / "main"))
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--tasks-per-kind", type=int, default=12)
    p.add_argument("--n-tasks", type=int, default=DEFAULT_CONFIG["n_tasks"])
    p.add_argument("--horizon", type=int, default=DEFAULT_CONFIG["horizon"])
    p.add_argument("--z-dim", type=int, default=DEFAULT_CONFIG["z_dim"])
    p.add_argument("--ctx-len", type=int, default=DEFAULT_CONFIG["ctx_len"])
    p.add_argument("--hidden", type=int, default=DEFAULT_CONFIG["hidden"])
    p.add_argument("--ppo-epochs", type=int, default=DEFAULT_CONFIG["ppo_epochs"])
    p.add_argument("--n-minibatch", type=int, default=DEFAULT_CONFIG["n_minibatch"])
    p.add_argument("--no-z-grad", dest="z_grad", action="store_false", default=True)
    p.add_argument("--commitment", action="store_true", default=False)
    p.add_argument("--commitment-cycle-len", type=int, default=0)
    p.add_argument("--commitment-warmup", type=int, default=0)
    p.add_argument("--finetune-steps", type=int, default=4000)
    p.add_argument("--retrain-budget", type=int, default=4000)
    p.add_argument("--retrain-replicas", type=int, default=8)
    p.add_argument("--pii-tasks", type=int, default=150)
    p.add_argument("--gate-boot", type=int, default=1000)
    p.add_argument("--label-stride", type=int, default=4)
    p.add_argument("--with-frozen", action="store_true", default=True)
    p.add_argument("--with-oracle", action="store_true", default=True)
    p.add_argument("--force", action="store_true", default=False)
    p.add_argument("--dry-run", action="store_true", default=False)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    set_threads(args.threads)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    strengths = [float(s) for s in args.strengths.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    cells = enumerate_cells(families, kinds, strengths, seeds)

    # group by (family, seed) so the agents are trained once per group
    groups = {}
    for fam, kind, st, sd in cells:
        groups.setdefault((fam, sd), []).append((kind, st))

    n_done = sum(1 for fam, kind, st, sd in cells
                 if cell_done(cell_path(outdir, fam, kind, st, sd)))
    print(f"[run_matrix] {len(cells)} cells | {len(groups)} (family,seed) groups | "
          f"{n_done} already done | iterations={args.iterations} threads={args.threads}",
          flush=True)

    if args.dry_run:
        print("[run_matrix] DRY RUN -- planned cells (no training):", flush=True)
        for fam, kind, st, sd in cells:
            path = cell_path(outdir, fam, kind, st, sd)
            status = "DONE(skip)" if (not args.force and cell_done(path)) else "RUN"
            print(f"  {status:<11} {path.name}", flush=True)
        print(f"[run_matrix] would run {sum(1 for fam, kind, st, sd in cells if args.force or not cell_done(cell_path(outdir, fam, kind, st, sd)))} "
              f"of {len(cells)} cells", flush=True)
        return 0

    progress_path = ARTIFACTS / "matrix_progress.txt"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    failures = []
    t_all = time.perf_counter()
    n_completed = 0

    for fam, sd in sorted(groups):
        group_cells = groups[(fam, sd)]
        missing = [(k, st) for (k, st) in group_cells
                   if args.force or not cell_done(cell_path(outdir, fam, k, st, sd))]
        if not missing:
            print(f"[run_matrix] {fam} seed={sd}: all {len(group_cells)} cells done, skip",
                  flush=True)
            continue
        print(f"[run_matrix] {fam} seed={sd}: training agents, {len(missing)} cell(s) to do",
              flush=True)
        cfg = build_cfg(args, fam)
        try:
            bundle = train_agents(fam, sd, cfg, args)
        except Exception as err:  # noqa: BLE001 - keep the batch alive
            for k, st in missing:
                path = cell_path(outdir, fam, k, st, sd)
                write_failed(path, err)
                failures.append((path.name, repr(err)))
                log_progress(progress_path,
                             f"{time.strftime('%H:%M:%S')} FAIL {path.name} {err!r}")
            continue

        for k, st in missing:
            path = cell_path(outdir, fam, k, st, sd)
            t_cell = time.perf_counter()
            try:
                rec = eval_cell(bundle, cfg, fam, k, st, sd, args)
                path.write_text(json.dumps(rec, indent=2))
                wall = time.perf_counter() - t_cell
                n_completed += 1
                cum = time.perf_counter() - t_all
                mret = rec["eval"]["meta_online"]["return_mean"]
                mreg = rec["eval"]["meta_online"]["regret"]
                gau = rec["gate"].get("auroc_consistency") if isinstance(rec["gate"], dict) else None
                log_progress(
                    progress_path,
                    f"{time.strftime('%H:%M:%S')} {path.name} | meta_online ret={mret:.2f} "
                    f"regret={mreg:.2f} | gate_auroc={gau} | wall={wall:.1f}s cum={cum:.0f}s")
                print(f"[run_matrix]   wrote {path.name} ({wall:.1f}s, cum {cum:.0f}s)",
                      flush=True)
            except Exception as err:  # noqa: BLE001 - keep the batch alive
                write_failed(path, err)
                failures.append((path.name, repr(err)))
                log_progress(progress_path,
                             f"{time.strftime('%H:%M:%S')} FAIL {path.name} {err!r}")
                print(f"[run_matrix]   FAILED {path.name}: {err!r}", flush=True)

    total = time.perf_counter() - t_all
    print(f"\n[run_matrix] finished: {n_completed} cells completed in {total:.0f}s", flush=True)
    if failures:
        print(f"[run_matrix] {len(failures)} failure(s):", flush=True)
        for name, err in failures:
            print(f"  - {name}: {err}", flush=True)
    else:
        print("[run_matrix] no failures", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
