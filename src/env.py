"""Regime-drifting respiratory-pathogen intervention environment.

A batched, pure-NumPy implementation of a seasonal SEIRS-type transmission
model with a reporting layer and three levels of intervention intensity.
The regime vector that governs transmission, reporting and behaviour is
*never* exposed to the agent: only reported case counts and calendar
information are observable. The regime may drift during an episode, so the
mapping from observations to the optimal intervention level is non-stationary.

Everything here is deterministic given a seed, runs on CPU, and is vectorised
across a batch of independent environments.

Regime vector (length 7), in this fixed order:

    0  beta0   baseline transmission rate
    1  sigma   progression rate (E -> I)
    2  gamma   recovery rate (I -> R)
    3  rho     reporting rate (fraction of incident infections reported)
    4  amp     amplitude of the seasonal forcing
    5  beh     behavioural rebound parameter (contacts rise with intervention)
    6  eps     direct efficacy parameter (transmissibility falls with intervention)

Action space (3 discrete levels of intervention intensity):
    0  none, 1  moderate, 2  strong

Two environment families are supported, and the difference between them is the
scientific object of the study:

``passive``
    ``eps`` and ``beh`` are held at fixed values. Every regime component is then
    reflected in the *passive* case trajectory, so the regime is identifiable
    without intervening.

``linked``
    ``eps`` and ``beh`` vary widely across tasks, but each is an affine function
    of the baseline transmission rate, which is itself reflected in the passive
    case trajectory. The decision-relevant variation is therefore heterogeneous
    and yet recoverable without intervening. This is the control condition that
    separates "the regime varies" from "the regime must be probed for".

``action_revealed``
    ``eps`` and ``beh`` vary widely across tasks and enter the dynamics ONLY
    through the action term. Two such tasks produce identical observation
    distributions until the agent intervenes, so the regime can only be learned
    by acting. This is the ambiguity that explicit online inference is meant to
    resolve.

The intervention acts through two opposing channels: a direct reduction of
transmissibility and a behavioural rebound that raises contacts. When the
rebound dominates, intervening is actively harmful.
"""

from __future__ import annotations

import numpy as np

OBS_DIM = 4
N_ACTIONS = 3
REGIME_DIM = 7

# Scale used to turn the infected fraction into an O(1) burden term.
BURDEN_REF = 0.02
# Per-step cost of intervention intensity level a is ACTION_COST * a.
ACTION_COST = 0.25
# Default direct efficacy, used as the fixed value in the ``passive`` family.
NPI_EFFICACY = 0.35
# Fixed behavioural rebound in the ``passive`` family.
PASSIVE_BEH = 0.05
# Ranges for the two action-channel parameters in the ``action_revealed`` family.
REVEALED_EPS_RANGE = (0.05, 0.60)
REVEALED_BEH_RANGE = (0.00, 0.35)
# Link used by the ``linked`` family: both parameters are affine functions of the
# normalised baseline transmission rate, so the decision-relevant variation is
# heterogeneous but passively recoverable. At s = +1 (severe) the rebound
# dominates and intervening is harmful; at s = -1 (mild) it is beneficial.
LINKED_EPS_MID = 0.325
LINKED_EPS_SPAN = 0.275
LINKED_BEH_MID = 0.175
LINKED_BEH_SPAN = 0.175
# Waning immunity (weekly).
OMEGA = 1.0 / 52.0

FAMILY_PASSIVE = "passive"
FAMILY_LINKED = "linked"
FAMILY_ACTION_REVEALED = "action_revealed"
FAMILIES = (FAMILY_PASSIVE, FAMILY_LINKED, FAMILY_ACTION_REVEALED)
# Index of the direct-efficacy and behavioural-rebound entries in the regime.
IDX_BEH = 5
IDX_EPS = 6

# Weekly reporting population. The dynamical state is expressed as population
# fractions, so this constant only sets the scale of the reported case counts.
# It must be large enough that weekly counts are not dominated by reporting
# zeros: if almost every observed count is zero, the observation carries no
# information about the latent regime and no adapter can work.
REPORTING_POPULATION = 1.0e5
# Unobserved warm-up steps taken at reset so that the first observation
# already carries a usable case history.
WARMUP_STEPS = 3

_POP = REPORTING_POPULATION


class RegimeDriftEpidemicEnv:
    """Batched SEIRS environment with unobservable, drifting regime."""

    OBS_DIM = OBS_DIM
    N_ACTIONS = N_ACTIONS
    REGIME_DIM = REGIME_DIM

    def __init__(self, n_envs: int = 64, horizon: int = 200, seed: int = 0):
        self.n_envs = int(n_envs)
        self.horizon = int(horizon)
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.regime0 = np.zeros((self.n_envs, REGIME_DIM))
        self.regime1 = np.zeros((self.n_envs, REGIME_DIM))
        self.onset = np.zeros(self.n_envs, dtype=int)
        self.duration = np.ones(self.n_envs, dtype=int)
        self.abrupt = np.zeros(self.n_envs, dtype=bool)
        self.state = np.zeros((self.n_envs, 4))  # S, E, I, R
        self.obs = np.zeros((self.n_envs, OBS_DIM))
        self.prev_cases = np.zeros(self.n_envs)
        self.prev_action = np.zeros(self.n_envs)
        self.done = np.zeros(self.n_envs, dtype=bool)
        self.regime_hist = np.zeros((self.n_envs, REGIME_DIM))

    # ------------------------------------------------------------------ #
    # regime trajectory
    # ------------------------------------------------------------------ #
    def _current_regime(self) -> np.ndarray:
        """Regime at the current step; used by analysis only, never by the agent."""
        frac = (self.t - self.onset) / np.maximum(self.duration, 1)
        frac = np.clip(frac, 0.0, 1.0)
        if self.abrupt.any():
            frac = np.where(self.abrupt, (self.t >= self.onset).astype(float), frac)
        return self.regime0 + (self.regime1 - self.regime0) * frac[:, None]

    # ------------------------------------------------------------------ #
    # episode control
    # ------------------------------------------------------------------ #
    def reset(self, regime0: np.ndarray, regime1=None, onset=None, duration=None,
              abrupt=None, seed=None) -> np.ndarray:
        """Start new episodes.

        Parameters are arrays of length ``n_envs`` (or scalars). ``regime1``
        defaults to ``regime0`` (stationary episode). ``onset`` is the step at
        which drift begins, ``duration`` its length in steps; ``abrupt=True``
        turns the drift into a step change at ``onset``.
        """
        n = self.n_envs
        r0 = np.broadcast_to(np.asarray(regime0, dtype=float), (n, REGIME_DIM)).copy()
        self.regime0 = r0
        if regime1 is None:
            self.regime1 = r0.copy()
        else:
            self.regime1 = np.broadcast_to(np.asarray(regime1, dtype=float), (n, REGIME_DIM)).copy()
        self.onset = (np.broadcast_to(np.asarray(onset, dtype=int), (n,)).copy()
                      if onset is not None else np.full(n, 10 ** 6, dtype=int))
        self.duration = (np.broadcast_to(np.asarray(duration, dtype=int), (n,)).copy()
                         if duration is not None else np.ones(n, dtype=int))
        self.abrupt = (np.broadcast_to(np.asarray(abrupt, dtype=bool), (n,)).copy()
                       if abrupt is not None else np.zeros(n, dtype=bool))
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Initial condition: a small, regime-dependent seed of infection.
        i0 = 5e-4 * (1.0 + 0.5 * self.rng.random(n))
        e0 = i0 * self.regime0[:, 2] / np.maximum(self.regime0[:, 1], 1e-6)
        self.state = np.stack([1.0 - i0 - e0, e0, i0, np.zeros(n)], axis=1)
        self.state = np.clip(self.state, 0.0, 1.0)

        self.t = 0
        self.prev_action = np.zeros(n)
        self.done[:] = False
        self.regime_hist = self._current_regime()
        # Unobserved warm-up, so that the first observation already carries a
        # case history and the growth channel is meaningful.
        cases = self._report(self.state[:, 1])
        for _ in range(WARMUP_STEPS):
            prev_cases = cases
            self._advance_state(np.zeros(n, dtype=int))
            cases = self._report(self.state[:, 1])
        self.prev_cases = cases
        self.obs = self._observe(cases, prev_cases)
        return self.obs.copy()

    def _report(self, exposed: np.ndarray) -> np.ndarray:
        """Incident infections are reported with rate rho and negative-binomial noise."""
        regime = self._current_regime()
        rho = regime[:, 3]
        sigma = regime[:, 1]
        mean = np.maximum(rho * _POP * sigma * exposed, 1e-9)
        disp = 12.0  # negative-binomial dispersion (higher = closer to Poisson)
        p = disp / (disp + mean)
        return self.rng.negative_binomial(disp, np.clip(p, 1e-9, 1 - 1e-9)).astype(float)

    def _observe(self, cases: np.ndarray, prev_cases: np.ndarray) -> np.ndarray:
        """Observable summary; the regime itself is deliberately absent.

        ``prev_cases`` must be the count reported *before* ``cases``; passing
        the same array twice would silently make the growth channel constant.
        """
        lc = np.log1p(cases)
        growth = (lc - np.log1p(prev_cases)) / 3.0
        seasonal = np.full(lc.shape, np.cos(2.0 * np.pi * (self.t % 52) / 52.0))
        obs = np.stack([lc / 3.0, growth, seasonal, self.prev_action / 2.0], axis=1)
        return np.clip(obs, -5.0, 5.0)

    def _advance_state(self, a: np.ndarray) -> None:
        """Apply one week of SEIRS dynamics. Does not touch time or reporting."""
        regime = self._current_regime()
        beta0, sigma, gamma, rho, amp = (regime[:, k] for k in range(5))
        beh = regime[:, IDX_BEH]
        eps = regime[:, IDX_EPS]

        seasonal = 1.0 + amp * np.cos(2.0 * np.pi * (self.t % 52) / 52.0)
        # Two opposing channels: the intervention directly lowers transmissibility
        # (eps) and simultaneously triggers a behavioural rebound (beh). For
        # regimes where the rebound dominates, intervening raises transmission.
        factor = 1.0 - eps * a + beh * a
        beta_eff = beta0 * seasonal * np.clip(factor, 0.05, 2.5)

        S, E, I, R = (self.state[:, k] for k in range(4))
        new_inf = beta_eff * S * I
        S2 = S - new_inf + OMEGA * R
        E2 = E + new_inf - sigma * E
        I2 = I + sigma * E - gamma * I
        R2 = R + gamma * I - OMEGA * R
        s = np.clip(S2 + E2 + I2 + R2, 1e-9, None)
        self.state = np.clip(np.stack([S2, E2, I2, R2], axis=1) / s[:, None], 0.0, 1.0)

    # ------------------------------------------------------------------ #
    # transition
    # ------------------------------------------------------------------ #
    def step(self, actions: np.ndarray):
        """Advance one week. ``actions`` is an integer array of length n_envs."""
        a = np.asarray(actions, dtype=int)
        regime = self._current_regime()
        prev_cases = self.prev_cases

        self._advance_state(a)
        cases = self._report(self.state[:, 1])
        burden = self.state[:, 2] / BURDEN_REF
        reward = -(burden + ACTION_COST * a)

        self.prev_cases = cases
        self.prev_action = a.astype(float)
        self.t += 1
        self.regime_hist = self._current_regime()
        self.done = np.full(self.n_envs, self.t >= self.horizon, dtype=bool)
        self.obs = self._observe(cases, prev_cases)

        info = {
            "cases": cases,
            "burden": burden.copy(),
            "regime": regime.copy(),
            "action": a.copy(),
        }
        return self.obs.copy(), reward, self.done.copy(), info

    # ------------------------------------------------------------------ #
    def current_regime(self) -> np.ndarray:
        """True latent regime at the current step (analysis / oracle only)."""
        return self._current_regime()

    # ------------------------------------------------------------------ #
    # state cloning (added for counterfactual / gate-label analysis)
    # ------------------------------------------------------------------ #
    def snapshot(self) -> dict:
        """Deep copy of every mutable field needed to resume exactly.

        Added by the analysis pipeline; it does not alter any existing
        behaviour. All arrays are copies, so the returned dict is safe to
        keep while the live environment keeps stepping.
        """
        return {
            "n_envs": int(self.n_envs),
            "horizon": int(self.horizon),
            "t": int(self.t),
            "regime0": self.regime0.copy(),
            "regime1": self.regime1.copy(),
            "onset": self.onset.copy(),
            "duration": self.duration.copy(),
            "abrupt": self.abrupt.copy(),
            "state": self.state.copy(),
            "obs": self.obs.copy(),
            "prev_cases": self.prev_cases.copy(),
            "prev_action": self.prev_action.copy(),
            "done": self.done.copy(),
            "regime_hist": self.regime_hist.copy(),
        }

    def restore(self, snap: dict, seed: int | None = None) -> np.ndarray:
        """Resume from a snapshot produced by :meth:`snapshot`.

        ``seed`` reseeds the reporting-noise generator so that two branches
        cloned from the same snapshot can be made independent yet
        reproducible. Returns the restored observation array.
        """
        self.n_envs = int(snap["n_envs"])
        self.horizon = int(snap["horizon"])
        self.t = int(snap["t"])
        self.regime0 = snap["regime0"].copy()
        self.regime1 = snap["regime1"].copy()
        self.onset = snap["onset"].copy()
        self.duration = snap["duration"].copy()
        self.abrupt = snap["abrupt"].copy()
        self.state = snap["state"].copy()
        self.obs = snap["obs"].copy()
        self.prev_cases = snap["prev_cases"].copy()
        self.prev_action = snap["prev_action"].copy()
        self.done = snap["done"].copy()
        self.regime_hist = snap["regime_hist"].copy()
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        return self.obs.copy()


# ---------------------------------------------------------------------- #
# task distribution
# ---------------------------------------------------------------------- #
BASE_REGIME = np.array([1.05, 0.60, 0.52, 0.35, 0.25, PASSIVE_BEH, NPI_EFFICACY])

# Per-dimension half-width of the stationary meta-training support. The width of
# the progression and recovery entries is kept modest so that the basic
# reproduction number stays above one across the support.
TRAIN_SPREAD = np.array([0.22, 0.08, 0.08, 0.12, 0.10, 0.10, 0.10])
# Uncertainty (1 sigma) of the meta-training support, used to build drift targets
# that sit *outside* the training support (the out-of-distribution condition).
TRAIN_SIGMA = TRAIN_SPREAD / 2.0


def sample_regimes(rng: np.random.Generator, n: int,
                   family: str = FAMILY_PASSIVE) -> np.ndarray:
    """Sample regimes from the meta-training support of one environment family.

    In ``passive`` the two action-channel parameters are held fixed, so the whole
    regime is reflected in the passive case trajectory. In ``action_revealed``
    they are drawn over wide ranges, which makes otherwise identical tasks
    distinguishable only after intervening.
    """
    u = rng.uniform(-1.0, 1.0, size=(n, REGIME_DIM))
    out = BASE_REGIME[None, :] + u * TRAIN_SPREAD[None, :]
    if family == FAMILY_ACTION_REVEALED:
        out[:, IDX_EPS] = rng.uniform(*REVEALED_EPS_RANGE, size=n)
        out[:, IDX_BEH] = rng.uniform(*REVEALED_BEH_RANGE, size=n)
    elif family == FAMILY_PASSIVE:
        out[:, IDX_EPS] = NPI_EFFICACY
        out[:, IDX_BEH] = PASSIVE_BEH
    elif family == FAMILY_LINKED:
        # The action channel varies across tasks, but it is a deterministic
        # function of the baseline transmission rate, which is itself reflected
        # in the passive case trajectory. The decision-relevant variation is
        # therefore heterogeneous AND recoverable without intervening.
        # Severe regimes get a weak direct effect and a strong behavioural
        # rebound, so the best action flips sign along the family.
        s = np.clip((out[:, 0] - BASE_REGIME[0]) / TRAIN_SPREAD[0], -1.0, 1.0)
        out[:, IDX_EPS] = np.clip(LINKED_EPS_MID - LINKED_EPS_SPAN * s,
                                  *REVEALED_EPS_RANGE)
        out[:, IDX_BEH] = np.clip(LINKED_BEH_MID + LINKED_BEH_SPAN * s,
                                  *REVEALED_BEH_RANGE)
    else:
        raise ValueError(f"unknown family: {family}")
    return out


DRIFT_KINDS = ("gradual_beta", "abrupt_beta", "gradual_rho",
               "endogenous_beh", "abrupt_eps")


def make_drift(rng: np.random.Generator, kind: str, n: int, horizon: int,
               strength: float = 1.0, family: str = FAMILY_PASSIVE):
    """Build drift targets and timing for a batch of episodes.

    ``strength`` scales how far the drifted regime sits from the training
    support; values above 1 are used to build out-of-distribution test tasks.
    """
    base = sample_regimes(rng, n, family=family)
    target = base.copy()
    onset = rng.integers(int(0.25 * horizon), int(0.55 * horizon) + 1, size=n)
    duration = rng.integers(10, 46, size=n)
    abrupt = np.zeros(n, dtype=bool)

    if kind == "gradual_beta":
        shift = TRAIN_SIGMA[0] * strength * (1.6 + 0.8 * rng.random(n))
        target[:, 0] = base[:, 0] + shift
    elif kind == "abrupt_beta":
        shift = TRAIN_SIGMA[0] * strength * (1.6 + 0.8 * rng.random(n))
        target[:, 0] = base[:, 0] + shift
        duration = np.ones(n, dtype=int)
        abrupt = np.ones(n, dtype=bool)
    elif kind == "gradual_rho":
        shift = TRAIN_SIGMA[3] * strength * (1.8 + 1.0 * rng.random(n))
        target[:, 3] = np.clip(base[:, 3] - shift, 0.05, 0.95)
    elif kind == "endogenous_beh":
        target[:, IDX_BEH] = base[:, IDX_BEH] + TRAIN_SIGMA[IDX_BEH] * strength * (4.0 + 2.0 * rng.random(n))
        target[:, IDX_BEH] = np.clip(target[:, IDX_BEH], 0.0, 0.6)
    elif kind == "abrupt_eps":
        # The intervention abruptly becomes less effective: its direct efficacy
        # collapses while the behavioural rebound is untouched.
        drop = 0.35 * strength * (0.6 + 0.4 * rng.random(n))
        target[:, IDX_EPS] = np.clip(base[:, IDX_EPS] - drop, 0.02, 0.60)
        duration = np.ones(n, dtype=int)
        abrupt = np.ones(n, dtype=bool)
    elif kind == "stationary":
        pass
    else:
        raise ValueError(f"unknown drift kind: {kind}")

    return {"regime0": base, "regime1": target, "onset": onset,
            "duration": duration, "abrupt": abrupt, "family": family}
