"""Commitment policy: a contingency formalization, disabled by default.

Idea
----
Instead of choosing an intervention level at every step, the controller commits
to one level for a whole *decision cycle* and cannot change it mid-cycle. The
action space then collapses from "a per-step action" to "one commitment per
cycle", which turns the task into a small regime-identification problem, matching
where the measurable value actually is.

Implementation
--------------
``CommitmentWrapper`` is a read-only wrapper around the frozen batched
environment. It intercepts ``step`` and, once a cycle is active, replaces every
action with the first action issued in that cycle. ``reset`` starts a new
episode (and clears the lock). A cycle begins after ``warmup`` free steps and
repeats every ``cycle_len`` steps; ``cycle_len=None`` means a single cycle per
episode.

The wrapper delegates all other attributes (``n_envs``, ``horizon``,
``current_regime``, ``snapshot`` / ``restore``, ...) to the wrapped environment,
so it is a drop-in replacement and the frozen environment is never modified.

Everything is OFF unless explicitly switched on through the config keys
``commitment`` / ``commitment_cycle_len`` / ``commitment_warmup`` or through an
explicit argument to :func:`wrap_cfg`.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class CommitmentWrapper:
    """Lock the intervention level to the first action of each decision cycle."""

    def __init__(self, env, cycle_len: Optional[int] = None, warmup: int = 0):
        self._env = env
        self.cycle_len = None if (cycle_len is None or int(cycle_len) <= 0) else int(cycle_len)
        self.warmup = int(warmup or 0)
        self._locked = None
        self._applied = None
        self._t = 0

    @property
    def n_envs(self) -> int:
        return int(self._env.n_envs)

    @property
    def horizon(self) -> int:
        return int(self._env.horizon)

    @property
    def applied(self):
        """The action actually passed to the environment at the last step."""
        return None if self._applied is None else self._applied.copy()

    def __getattr__(self, name):
        # Reached only when the attribute is not found on the wrapper itself.
        return getattr(self._env, name)

    def reset(self, *args, **kwargs):
        self._locked = None
        self._applied = None
        self._t = 0
        return self._env.reset(*args, **kwargs)

    def snapshot(self) -> dict:
        """Delegate, but also capture the commitment state so a cloned branch
        resumes with the same lock (needed by the counterfactual gate labels)."""
        snap = self._env.snapshot()
        snap["_commit_state"] = {
            "locked": None if self._locked is None else self._locked.copy(),
            "applied": None if self._applied is None else self._applied.copy(),
            "t": int(self._t),
        }
        return snap

    def restore(self, snap: dict, seed=None):
        cs = snap.get("_commit_state")
        if cs is not None:
            self._locked = None if cs["locked"] is None else cs["locked"].copy()
            self._applied = None if cs["applied"] is None else cs["applied"].copy()
            self._t = int(cs["t"])
        return self._env.restore(snap, seed=seed)

    def step(self, actions):
        a = np.asarray(actions, dtype=int)
        if self._t >= self.warmup:
            if self.cycle_len is None:
                # A single cycle per episode: commit once, at the first step
                # after warmup, and never release the lock again.
                at_cycle_start = (self._t == self.warmup)
            else:
                at_cycle_start = ((self._t - self.warmup) % self.cycle_len == 0)
            if at_cycle_start or self._locked is None:
                self._locked = a.copy()
            a = self._locked
        self._applied = np.asarray(a, dtype=int).copy()
        self._t += 1
        return self._env.step(a)

    def unlocked_actions(self, actions):
        """The action each env would play if it were not committed (diagnostics)."""
        return np.asarray(actions, dtype=int)


def enabled(cfg: Optional[dict], commitment: Optional[bool] = None) -> bool:
    """Resolve the commitment switch: explicit argument wins over the config."""
    if commitment is not None:
        return bool(commitment)
    return bool((cfg or {}).get("commitment", False))


def wrap(env, commitment: bool, cycle_len: Optional[int] = None, warmup: int = 0):
    """Wrap ``env`` when commitment is on; otherwise return it unchanged."""
    if not commitment or isinstance(env, CommitmentWrapper):
        return env
    return CommitmentWrapper(env, cycle_len=cycle_len, warmup=warmup)


def wrap_cfg(env, cfg: Optional[dict], commitment: Optional[bool] = None):
    """Wrap ``env`` according to a config dict (and an optional explicit flag)."""
    c = cfg or {}
    if not enabled(c, commitment):
        return env
    return wrap(env, True, c.get("commitment_cycle_len"), c.get("commitment_warmup", 0))
