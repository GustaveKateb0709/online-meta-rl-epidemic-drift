"""Shared helpers for the small-scale z-gradient validation / collapse diagnostics."""
from __future__ import annotations

from src.common import DEFAULT_CONFIG
from src.meta_rl import MetaRLAgent, train_meta


def small_cfg(family: str) -> dict:
    """Small, fast configuration used by the validation and collapse diagnostics."""
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(n_tasks=16, horizon=120, iterations=80, hidden=96, ctx_len=12,
               z_dim=5, ppo_epochs=3, n_minibatch=3, lr=3e-4, family=family)
    return cfg


def train_agent(cfg: dict, seed: int, z_mode: str, z_grad: bool) -> MetaRLAgent:
    """Construct and meta-train one agent under a fixed (z_mode, z_grad) switch."""
    agent = MetaRLAgent(dict(cfg, z_mode=z_mode, z_grad=z_grad))
    train_meta(agent, seed, verbose=False)
    return agent
