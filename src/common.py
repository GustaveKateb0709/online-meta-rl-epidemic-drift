"""Shared configuration, seeding and provenance helpers.

This module is deliberately dependency-light. It centralises the defaults that
both the meta-RL agent and the baselines must share so that comparisons are
apples-to-apples.
"""
from __future__ import annotations

import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from src.env import (N_ACTIONS, OBS_DIM, REGIME_DIM, RegimeDriftEpidemicEnv,
                     make_drift, sample_regimes)

# ---------------------------------------------------------------------- #
# configuration
# ---------------------------------------------------------------------- #
# Defaults chosen so a full meta-training run fits the CPU / time budget.
DEFAULT_CONFIG = {
    "obs_dim": OBS_DIM,
    "n_actions": N_ACTIONS,
    "regime_dim": REGIME_DIM,
    "z_dim": 5,            # latent context dimension
    "ctx_len": 12,         # K: sliding-window length for z inference
    "hidden": 128,         # shared hidden width
    "lr": 3e-4,
    "gamma": 0.99,
    "lam": 0.95,
    "clip": 0.2,
    "ppo_epochs": 4,
    "n_minibatch": 4,
    "value_coef": 0.5,
    "entropy_coef": 0.01,
    "kl_coef": 0.1,        # weight on KL(q(z|c) || N(0,I))
    "decoder_coef": 1.0,   # weight on decoder negative log-likelihood
    "reward_scale": 0.05,  # training-time reward rescaling
    "max_grad_norm": 0.5,
    "n_tasks": 32,         # parallel tasks (envs) per meta-iteration
    "horizon": 200,
    "iterations": 300,     # meta-training iterations
    "z_mode": "online",    # online | frozen | oracle
    "z_grad": True,        # differentiate policy/value through mu into the encoder
    "family": "passive",   # passive | action_revealed
    "commitment": False,   # lock one intervention level per decision cycle (OFF)
    "commitment_cycle_len": None,  # None -> one cycle per episode
    "commitment_warmup": 0,        # free (unlocked) steps before the first cycle
    "history_len": 0,      # concat last K observations as policy input (0/1 = OFF)
    "aux_batch": 256,      # context windows per auxiliary update
    "seed": 0,
}

N_THREADS = 4


def set_threads(n: int = N_THREADS) -> int:
    """Pin the torch thread count (CPU-only, 8 GB machine)."""
    torch.set_num_threads(int(n))
    return int(torch.get_num_threads())


def device() -> torch.device:
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(False)


def as_t(x, dtype=torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dtype)
    return torch.as_tensor(np.asarray(x), dtype=dtype)


# ---------------------------------------------------------------------- #
# environment factories
# ---------------------------------------------------------------------- #
def make_env(n_envs: int, horizon: int, seed: int) -> RegimeDriftEpidemicEnv:
    return RegimeDriftEpidemicEnv(n_envs=int(n_envs), horizon=int(horizon), seed=int(seed))


def stationary_regimes(rng: np.random.Generator, n: int) -> np.ndarray:
    return sample_regimes(rng, n)


def drift_task(rng: np.random.Generator, kind: str, n: int, horizon: int,
               strength: float) -> dict:
    return make_drift(rng, kind, n, horizon, strength=strength)


# ---------------------------------------------------------------------- #
# provenance
# ---------------------------------------------------------------------- #
def _git_commit() -> str | None:
    try:
        root = Path(__file__).resolve().parents[1]
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return out.stdout.strip() or None
    except Exception:
        pass
    return None


def _host_ram_gb() -> float:
    try:
        if platform.system() == "Darwin":
            out = subprocess.run(["sysctl", "-n", "hw.memsize"],
                                 capture_output=True, text=True, timeout=5)
            if out.returncode == 0:
                return round(int(out.stdout.strip()) / 1024 ** 3, 1)
        pages = os.sysconf("SC_PHYS_PAGES")
        psize = os.sysconf("SC_PAGE_SIZE")
        return round(pages * psize / 1024 ** 3, 1)
    except Exception:
        return 8.0


def provenance() -> dict:
    cpu = platform.processor() or platform.machine() or "unknown"
    return {
        "git_commit": _git_commit(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host_ram_gb": _host_ram_gb(),
        "cpu": cpu,
        "torch_threads": int(torch.get_num_threads()),
    }
