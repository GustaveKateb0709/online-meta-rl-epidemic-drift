"""Passive Identifiability Index (PII).

The paper's boundary-condition claim is that explicit online inference only
helps when decision-relevant parameters are *not* recoverable from passive
(policy-free) observations. PII turns that narrative into a measurable number.

Procedure
---------
For a batch of tasks drawn from one environment family:

1. Simulate each task under the three constant actions ``a in {0, 1, 2}`` and
   record the per-action return. The task's oracle constant action is the
   arg-max.
2. Build features from the **passive** trajectory (``a == 0``) only -- the
   history an agent would see without intervening.
3. Fit a small probe (multinomial logistic regression) to predict the oracle
   constant action from those passive features, evaluated with K-fold
   cross-validation.
4. ``accuracy`` is the out-of-sample 3-class accuracy; the continuous version
   is the expected-return gap between the truly best constant action and the
   action chosen by the probe's posterior (``continuous_gap``, lower is better).

Expected: high accuracy in ``passive`` (params show up in the passive
trajectory) and ~chance in ``action_revealed`` (params enter only through the
action term, so a passive trajectory cannot reveal them).
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import KFold, StratifiedKFold

from src.env import FAMILY_PASSIVE, OBS_DIM, RegimeDriftEpidemicEnv, sample_regimes

PII_DEFINITION = (
    "PII = K-fold cross-validated out-of-sample accuracy of a multinomial "
    "logistic-regression probe that predicts a task's oracle constant action "
    "(argmax over a in {0,1,2} of the constant-action return) from features of "
    "its PASSIVE trajectory (a=0): per observation dimension, [mean, std, max, "
    "min, last]. 'chance' is the majority-class rate. 'continuous_gap' is the "
    "mean of (true best constant-action return - expected return under the "
    "probe's class posterior); lower is better. 'best_return_r2' is the "
    "5-fold out-of-sample R^2 of a Ridge probe predicting the best "
    "constant-action return from the same passive features."
)


def _rollout_const(env: RegimeDriftEpidemicEnv, action: int, regimes, seed: int):
    """Roll a constant action; return (returns (H,B), obs series (H+1,B,obs))."""
    B, H = env.n_envs, env.horizon
    obs = env.reset(regimes, seed=seed)
    O = np.zeros((H + 1, B, OBS_DIM), np.float32)
    R = np.zeros((H, B), np.float64)
    O[0] = obs
    a = np.full(B, int(action), dtype=int)
    for t in range(H):
        obs, r, done, _ = env.step(a)
        R[t] = r
        O[t + 1] = obs
    return R, O


def _features(obs_series: np.ndarray) -> np.ndarray:
    """(H+1, B, obs_dim) -> (B, obs_dim * 5) passive-history features."""
    x = obs_series.transpose(1, 2, 0)  # (B, obs_dim, T)
    feats = [x.mean(-1), x.std(-1), x.max(-1), x.min(-1), obs_series[-1]]
    return np.concatenate(feats, axis=1)


def passive_identifiability_index(family: str = FAMILY_PASSIVE, n_tasks: int = 120,
                                  horizon: int = 120, seed: int = 0,
                                  n_splits: int = 5) -> Dict[str, float]:
    """Compute the PII for one environment family."""
    rng = np.random.default_rng(seed)
    regimes = sample_regimes(rng, n_tasks, family=family)
    env = RegimeDriftEpidemicEnv(n_envs=n_tasks, horizon=horizon, seed=seed + 1)

    returns = np.zeros((n_tasks, 3))
    passive = None
    for k, a in enumerate((0, 1, 2)):
        R, O = _rollout_const(env, a, regimes, seed=seed + 10 + k)
        returns[:, k] = R.sum(0)
        if a == 0:
            passive = O
    best = returns.argmax(1)

    X = _features(passive)
    counts = np.bincount(best, minlength=3)
    chance = float(counts.max() / counts.sum())

    # Manual CV that always returns a (n, 3) class-posterior matrix, so absent
    # classes (e.g. the passive family, where the constant-action optimum is
    # almost always "strong") do not break column alignment.
    present = counts[counts > 0]
    min_class = int(present.min()) if present.size else 0
    if present.size == 3 and min_class >= 2:
        n_splits = max(2, min(n_splits, min_class))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    else:
        n_splits = 5
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    proba = np.zeros((n_tasks, 3))
    try:
        for tr, te in cv.split(X, best):
            clf = LogisticRegression(max_iter=2000).fit(X[tr], best[tr])
            p = clf.predict_proba(X[te])
            for j, cls in enumerate(clf.classes_):
                proba[te, int(cls)] = p[:, j]
        pred = proba.argmax(1)
        acc = float((pred == best).mean())
        macro_f1 = float(f1_score(best, pred, average="macro", labels=[0, 1, 2],
                                  zero_division=0.0))
        expected = (proba * returns).sum(1)
        gap = float((returns.max(1) - expected).mean())
    except Exception:
        acc, macro_f1, gap = float("nan"), float("nan"), float("nan")

    # Continuous probe: predict the best constant-action return directly.
    best_ret = returns.max(1)
    kf2 = KFold(n_splits=5, shuffle=True, random_state=seed + 7)
    oof = np.zeros(n_tasks)
    try:
        for tr, te in kf2.split(X):
            reg = Ridge(alpha=1.0).fit(X[tr], best_ret[tr])
            oof[te] = reg.predict(X[te])
        best_return_r2 = float(r2_score(best_ret, oof))
    except Exception:
        best_return_r2 = float("nan")
    return {"family": family, "accuracy": acc, "chance": chance, "macro_f1": macro_f1,
            "continuous_gap": gap, "best_return_r2": best_return_r2,
            "n_tasks": int(n_tasks), "n_splits": int(n_splits),
            "best_action_counts": [int(c) for c in counts],
            "return_spread": float(returns.max(1).mean() - returns.min(1).mean())}
