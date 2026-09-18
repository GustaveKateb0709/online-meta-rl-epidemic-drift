"""Gating diagnostics: can an observable-only signal decide when to stop adapting?

Three candidate signals are computed online from the meta-agent's own context
model (no regime labels):

* ``consistency_residual`` : normalised one-step decoder error over the recent
  window under the inferred ``z`` (the proposed signal);
* ``posterior_entropy``    : entropy of the context posterior (baseline);
* ``predictive_variance``  : Monte-Carlo decoder predictive variance (baseline).

The ground-truth label ``adaptation_harmful[t]`` is obtained by *state cloning*:
from the state at step ``t`` we roll out two counterfactual branches to the end
of the episode -- (a) the online meta-policy, (b) the non-adapting Robust
baseline -- and mark the step as harmful when the adapting branch earns a lower
remaining return. Branches reseed the reporting-noise generator so their noise
is independent yet reproducible.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import torch

from src.common import as_t, device
from src.env import N_ACTIONS, OBS_DIM, RegimeDriftEpidemicEnv
from src.meta_rl import ContextBuffer

LOG2PI = math.log(2.0 * math.pi)


# ---------------------------------------------------------------------- #
# signal definitions
# ---------------------------------------------------------------------- #
@torch.no_grad()
def consistency_residual(decoder, obs, act_oh, rew, next_obs, z, scale: float,
                         mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Normalised RMS one-step decoder error over the window (B,)."""
    B, K = obs.shape[0], obs.shape[1]
    zz = z.unsqueeze(1).expand(-1, K, -1)
    r_hat, no_hat = decoder(obs, act_oh, zz)
    err = (r_hat - rew) ** 2 + ((no_hat - next_obs) ** 2).mean(-1)   # (B,K)
    if mask is None:
        mask = torch.ones_like(err)
    w = mask / mask.sum(1, keepdim=True).clamp(min=1.0)
    per = (err * w).sum(1)
    return torch.sqrt(per / max(scale, 1e-8))


@torch.no_grad()
def posterior_entropy(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Differential entropy of the diagonal Gaussian posterior (B,)."""
    return 0.5 * (logvar + 1.0 + LOG2PI).sum(-1)


@torch.no_grad()
def predictive_variance(decoder, obs, act_oh, mu, logvar, mask=None,
                        n_samples: int = 8) -> torch.Tensor:
    """Mean MC predictive variance of the next observation across posterior draws (B,)."""
    B, K = obs.shape[0], obs.shape[1]
    std = torch.exp(0.5 * logvar)
    draw = []
    for _ in range(n_samples):
        z = mu + std * torch.randn_like(std)
        _, no_hat = decoder(obs, act_oh, z.unsqueeze(1).expand(-1, K, -1))
        draw.append(no_hat)
    var = torch.stack(draw, 0).var(0).mean(-1)                       # (B,K)
    if mask is None:
        mask = torch.ones_like(var)
    w = mask / mask.sum(1, keepdim=True).clamp(min=1.0)
    return (var * w).sum(1)


# ---------------------------------------------------------------------- #
# online meta rollout with signal + snapshot recording
# ---------------------------------------------------------------------- #
@torch.no_grad()
def online_rollout(agent, env: RegimeDriftEpidemicEnv, drift: dict, seed: int,
                   deterministic: bool = True, record: bool = False,
                   resid_scale: float = 1.0, pv_samples: int = 8,
                   z_mode: str = "online") -> dict:
    """Run the meta-policy on drifting tasks, recording gate signals.

    ``z_mode`` selects online re-inference (the proposal), a single frozen
    inference (ablation), or the true regime (oracle upper bound). Also records,
    at every step, the environment snapshot and the context buffer *before*
    acting, which the label builder needs.
    """
    c = agent.cfg
    K = c["ctx_len"]
    dev = agent.dev
    B, H = env.n_envs, env.horizon
    buf = ContextBuffer(B, K, OBS_DIM, N_ACTIONS)
    obs = env.reset(drift["regime0"], regime1=drift["regime1"], onset=drift["onset"],
                    duration=drift["duration"], abrupt=drift["abrupt"], seed=seed)

    frozen_z = torch.zeros(B, agent.policy_z_dim, device=dev)
    frozen_ready = np.zeros(B, dtype=bool)

    rew = np.zeros((H, B))
    resid = np.full((H, B), np.nan)
    ent = np.full((H, B), np.nan)
    pvar = np.full((H, B), np.nan)
    actions = np.zeros((H, B), dtype=int)
    snaps: List[dict] = []
    bufs: List[ContextBuffer] = []

    for t in range(H):
        if z_mode == "oracle":
            z = as_t(env.current_regime()).to(dev)
            mu, logvar = z, torch.zeros_like(z)
        else:
            o, a_oh, r, no, m = buf.tensors()
            o, a_oh, r, no, m = o.to(dev), a_oh.to(dev), r.to(dev), no.to(dev), m.to(dev)
            mu, logvar = agent.encoder(o, a_oh, r, no, m)
            if z_mode == "frozen":
                ready_now = (buf.count >= K) & (~frozen_ready)
                if ready_now.any():
                    frozen_z[torch.as_tensor(ready_now, device=dev)] = mu[torch.as_tensor(ready_now, device=dev)]
                    frozen_ready = frozen_ready | ready_now
                z = torch.where(torch.as_tensor(frozen_ready, device=dev)[:, None], frozen_z, mu)
            else:
                z = mu  # policy is conditioned on the posterior mean
        if record:
            snaps.append(env.snapshot())
            bufs.append(buf.copy())
        if z_mode != "oracle" and int(m.sum().item()) > 0:
            resid[t] = consistency_residual(agent.decoder, o, a_oh, r, no, z, resid_scale, m).cpu().numpy()
            ent[t] = posterior_entropy(mu, logvar).cpu().numpy()
            pvar[t] = predictive_variance(agent.decoder, o, a_oh, mu, logvar, m, pv_samples).cpu().numpy()
        obs_t = as_t(obs).to(dev)
        logits = agent.policy(obs_t, z)
        a = logits.argmax(-1) if deterministic else torch.distributions.Categorical(logits=logits).sample()
        a_np = a.cpu().numpy()
        nobs, raw, done, _ = env.step(a_np)
        buf.push(obs, a_np, raw * c["reward_scale"], nobs)
        rew[t] = raw
        actions[t] = a_np
        obs = nobs

    return {"rew": rew, "residual": resid, "entropy": ent, "pvariance": pvar,
            "actions": actions, "snaps": snaps, "bufs": bufs, "onset": drift["onset"]}


@torch.no_grad()
def calibrate_residual_scale(agent, env: RegimeDriftEpidemicEnv, regimes: np.ndarray,
                             n_iters: int = 4, seed: int = 0) -> float:
    """In-distribution calibration scale for the consistency residual.

    Mean raw residual across stationary (in-distribution) rollouts. Gate scores
    are divided by this scale so that they are comparable across contexts.
    """
    c = agent.cfg
    K = c["ctx_len"]
    dev = agent.dev
    B = env.n_envs
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_iters):
        buf = ContextBuffer(B, K, OBS_DIM, N_ACTIONS)
        obs = env.reset(regimes)
        for t in range(env.horizon):
            o, a_oh, r, no, m = buf.tensors()
            o, a_oh, r, no, m = o.to(dev), a_oh.to(dev), r.to(dev), no.to(dev), m.to(dev)
            mu, logvar = agent.encoder(o, a_oh, r, no, m)
            if int(m.sum().item()) > 0:
                zz = mu.unsqueeze(1).expand(-1, K, -1)
                r_hat, no_hat = agent.decoder(o, a_oh, zz)
                err = (r_hat - r) ** 2 + ((no_hat - no) ** 2).mean(-1)
                w = m / m.sum(1, keepdim=True).clamp(min=1.0)
                vals.append(float((err * w).sum(1).mean()))
            logits = agent.policy(as_t(obs).to(dev), mu)
            a = logits.argmax(-1).cpu().numpy()
            nobs, raw, done, _ = env.step(a)
            buf.push(obs, a, raw * c["reward_scale"], nobs)
            obs = nobs
    return float(np.mean(vals)) if vals else 1.0


# ---------------------------------------------------------------------- #
# counterfactual labels via state cloning
# ---------------------------------------------------------------------- #
@torch.no_grad()
def _branch_meta(agent, env, snap: dict, buf0: ContextBuffer, horizon: int,
                 seed: int) -> np.ndarray:
    c = agent.cfg
    dev = agent.dev
    buf = buf0.copy()
    obs = env.restore(snap, seed=seed)
    rem = np.zeros(env.n_envs)
    for _t in range(int(snap["t"]), horizon):
        o, a_oh, r, no, m = buf.tensors()
        o, a_oh, r, no, m = o.to(dev), a_oh.to(dev), r.to(dev), no.to(dev), m.to(dev)
        mu, logvar = agent.encoder(o, a_oh, r, no, m)
        z = mu
        logits = agent.policy(as_t(obs).to(dev), z)
        a_np = logits.argmax(-1).cpu().numpy()
        nobs, raw, done, _ = env.step(a_np)
        buf.push(obs, a_np, raw * c["reward_scale"], nobs)
        rem += raw
        obs = nobs
    return rem


@torch.no_grad()
def _branch_robust(robust_agent, env, snap: dict, horizon: int, seed: int) -> np.ndarray:
    obs = env.restore(snap, seed=seed)
    rem = np.zeros(env.n_envs)
    for _t in range(int(snap["t"]), horizon):
        a_np = robust_agent.act(obs, deterministic=True)
        obs, raw, done, _ = env.step(a_np)
        rem += raw
    return rem


def compute_adaptation_labels(roll: dict, agent, robust_agent, n_envs: int, horizon: int,
                              seed: int, stride: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (labels (H,B) bool, remaining_meta (H,B), remaining_robust (H,B)).

    ``labels[t, i] = remaining_meta[t, i] < remaining_robust[t, i]`` -- adapting
    from the state at (t, i) yields a worse remaining return than not adapting.
    Only steps present in ``snaps`` (every ``stride``) are filled; others are
    left False with NaN returns.
    """
    snaps = roll["snaps"]
    bufs = roll["bufs"]
    n_rec = len(snaps)
    H = horizon
    labels = np.zeros((H, n_envs), dtype=bool)
    rem_m = np.full((H, n_envs), np.nan)
    rem_r = np.full((H, n_envs), np.nan)

    envA = RegimeDriftEpidemicEnv(n_envs=n_envs, horizon=horizon, seed=seed)
    envB = RegimeDriftEpidemicEnv(n_envs=n_envs, horizon=horizon, seed=seed)
    for j in range(n_rec):
        t = int(snaps[j]["t"])
        if stride > 1 and (t % stride) != 0:
            continue
        s_a = seed * 100003 + t * 17 + 1
        s_b = seed * 100003 + t * 17 + 2
        ra = _branch_meta(agent, envA, snaps[j], bufs[j], horizon, s_a)
        rb = _branch_robust(robust_agent, envB, snaps[j], horizon, s_b)
        rem_m[t] = ra
        rem_r[t] = rb
        labels[t] = ra < rb
    return labels, rem_m, rem_r


# ---------------------------------------------------------------------- #
# gate evaluation
# ---------------------------------------------------------------------- #
def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if labels.min() == labels.max():
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _pr_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    if labels.sum() == 0:
        return float("nan")
    return float(average_precision_score(labels, scores))


def evaluate_gate(scores: np.ndarray, labels: np.ndarray, groups: Optional[np.ndarray] = None,
                  n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> dict:
    """AUROC / PR-AUC with an episode-level bootstrap confidence interval.

    ``groups`` gives the episode index of each sample; the bootstrap resamples
    episodes (with replacement) to respect within-episode correlation.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    ok = np.isfinite(scores)
    scores, labels = scores[ok], labels[ok]
    if groups is not None:
        groups = np.asarray(groups)[ok]
    out = {"auroc": _auroc(scores, labels), "pr_auc": _pr_auc(scores, labels),
           "n": int(scores.size), "n_pos": int(labels.sum()),
           "pos_rate": float(labels.mean()) if labels.size else float("nan")}
    rng = np.random.default_rng(seed)
    if groups is None or scores.size == 0:
        out["ci"] = {"auroc": [float("nan")] * 2, "pr_auc": [float("nan")] * 2}
        return out
    uniq = np.unique(groups)
    aurocs, prs = [], []
    idx_by_group = {g: np.where(groups == g)[0] for g in uniq}
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        sel = np.concatenate([idx_by_group[g] for g in pick])
        s, y = scores[sel], labels[sel]
        a = _auroc(s, y)
        p = _pr_auc(s, y)
        if np.isfinite(a):
            aurocs.append(a)
        if np.isfinite(p):
            prs.append(p)
    def _ci(v):
        if not v:
            return [float("nan"), float("nan")]
        return [float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2))]
    out["ci"] = {"auroc": _ci(aurocs), "pr_auc": _ci(prs)}
    return out
