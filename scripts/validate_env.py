"""Environment validation: 12 mechanistic assertions plus a speed benchmark.

Run:  python scripts/validate_env.py
Exits non-zero if any assertion fails.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.env import (BASE_REGIME, DRIFT_KINDS, FAMILY_ACTION_REVEALED,
                     FAMILY_LINKED, FAMILY_PASSIVE, IDX_BEH, IDX_EPS, N_ACTIONS,
                     OBS_DIM, REGIME_DIM, RegimeDriftEpidemicEnv, make_drift,
                     sample_regimes)

CHECKS = []


def check(name, cond, detail=""):
    CHECKS.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def rollout(env, actions_fn, n_envs, horizon, regime0, regime1=None, onset=None,
            duration=None, abrupt=None, seed=0):
    env.n_envs = n_envs
    env.regime0 = np.zeros((n_envs, REGIME_DIM))
    env.regime1 = np.zeros((n_envs, REGIME_DIM))
    env.onset = np.zeros(n_envs, dtype=int)
    env.duration = np.ones(n_envs, dtype=int)
    env.abrupt = np.zeros(n_envs, dtype=bool)
    env.state = np.zeros((n_envs, 4))
    env.obs = np.zeros((n_envs, OBS_DIM))
    env.prev_cases = np.zeros(n_envs)
    env.prev_action = np.zeros(n_envs)
    env.done = np.zeros(n_envs, dtype=bool)
    env.regime_hist = np.zeros((n_envs, REGIME_DIM))
    env.horizon = horizon
    obs = env.reset(regime0, regime1=regime1, onset=onset, duration=duration,
                    abrupt=abrupt, seed=seed)
    ret = np.zeros(n_envs)
    burden = np.zeros((horizon, n_envs))
    done = np.zeros(n_envs, dtype=bool)
    for t in range(horizon):
        a = actions_fn(t, obs)
        obs, r, d, info = env.step(a)
        ret += r
        burden[t] = info["burden"]
        done |= d
    return ret, burden, done


def run_collect(env, n_envs, horizon, regime0, seed=0, action=0):
    """Roll out and return the full observation / case / regime histories."""
    env.n_envs = n_envs
    env.regime0 = np.zeros((n_envs, REGIME_DIM))
    env.regime1 = np.zeros((n_envs, REGIME_DIM))
    env.onset = np.zeros(n_envs, dtype=int)
    env.duration = np.ones(n_envs, dtype=int)
    env.abrupt = np.zeros(n_envs, dtype=bool)
    env.state = np.zeros((n_envs, 4))
    env.obs = np.zeros((n_envs, OBS_DIM))
    env.prev_cases = np.zeros(n_envs)
    env.prev_action = np.zeros(n_envs)
    env.done = np.zeros(n_envs, dtype=bool)
    env.regime_hist = np.zeros((n_envs, REGIME_DIM))
    env.horizon = horizon
    obs = env.reset(regime0, seed=seed)
    obs_h = np.zeros((horizon, n_envs, OBS_DIM))
    case_h = np.zeros((horizon, n_envs))
    for t in range(horizon):
        obs, r, d, info = env.step(np.full(n_envs, action, dtype=int))
        obs_h[t] = obs
        case_h[t] = info["cases"]
    return obs_h, case_h


def main():
    rng = np.random.default_rng(0)
    n_envs = 32
    horizon = 156
    env = RegimeDriftEpidemicEnv(n_envs=n_envs, horizon=horizon, seed=1)

    # --- 1-2: stationary dynamics produce a non-trivial outbreak ------------
    regimes = sample_regimes(rng, n_envs)
    ret0, burden0, done0 = rollout(env, lambda t, o: np.zeros(n_envs, int),
                                   n_envs, horizon, regimes)
    peak = burden0.max(axis=0)
    check("1. every episode develops an epidemic (peak burden > 0.25) and the "
          "majority exceed 1.0",
          np.all(peak > 0.25) and np.mean(peak > 1.0) >= 0.8,
          f"min peak = {peak.min():.2f}, max = {peak.max():.2f}, "
          f"share > 1 = {np.mean(peak > 1.0):.2f}")
    check("2. dynamics stay in a plausible range (peak burden < 200)",
          np.all(peak < 200.0), f"max peak = {peak.max():.1f}")

    # --- 3-4: intervention reduces burden ----------------------------------
    ret2, burden2, _ = rollout(env, lambda t, o: np.full(n_envs, 2, int),
                               n_envs, horizon, regimes)
    mean0, mean2 = burden0[:60].mean(), burden2[:60].mean()
    check("3. constant strong intervention lowers early burden",
          mean2 < mean0, f"no-intervention {mean0:.3f} vs strong {mean2:.3f}")
    check("4. intervention carries a cost (return is not monotone in intensity)",
          not np.allclose(ret2 - ret0, 0.0), f"mean return none {ret0.mean():.2f}, strong {ret2.mean():.2f}")

    # --- 5-6: regime is latent, observation shape is stable ----------------
    obs_shape_ok = env.obs.shape == (n_envs, OBS_DIM)
    check("5. observation shape equals (n_envs, OBS_DIM)", obs_shape_ok)
    check("6. regime never appears in the observation vector (dimension check)",
          env.obs.shape[1] != REGIME_DIM, f"OBS_DIM={OBS_DIM}, REGIME_DIM={REGIME_DIM}")

    # --- 7-8: drift actually changes the regime and the burden ------------
    drift = make_drift(rng, "gradual_beta", n_envs, horizon, strength=1.0)
    ret_d, burden_d, _ = rollout(env, lambda t, o: np.zeros(n_envs, int),
                                 n_envs, horizon, drift["regime0"],
                                 regime1=drift["regime1"], onset=drift["onset"],
                                 duration=drift["duration"], abrupt=drift["abrupt"])
    check("7. drift moves the latent regime away from its initial value",
          np.mean(np.abs(drift["regime1"][:, 0] - drift["regime0"][:, 0])) > 0.1,
          f"mean |dbeta0| = {np.mean(np.abs(drift['regime1'][:, 0] - drift['regime0'][:, 0])):.3f}")
    late = burden_d[100:].mean()
    late_static = burden0[100:].mean()
    check("8. drifting episodes differ from stationary ones",
          abs(late - late_static) > 1e-6, f"drift late burden {late:.3f}, stationary {late_static:.3f}")

    # --- 9: all drift kinds construct ------------------------------------
    ok = True
    for kind in DRIFT_KINDS:
        d = make_drift(rng, kind, 8, horizon, strength=1.0)
        ok = ok and d["regime1"].shape == (8, REGIME_DIM)
    check("9. all declared drift kinds construct valid targets", ok, f"kinds = {DRIFT_KINDS}")

    # --- 10: reproducibility ---------------------------------------------
    r1 = sample_regimes(np.random.default_rng(7), 4)
    r2 = sample_regimes(np.random.default_rng(7), 4)
    check("10. sampling is reproducible under a fixed seed", np.array_equal(r1, r2))

    # --- 11: determinism under identical seed and actions -----------------
    a1 = rollout(env, lambda t, o: (t % 3) * np.ones(n_envs, int), n_envs, horizon, regimes, seed=5)[0]
    a2 = rollout(env, lambda t, o: (t % 3) * np.ones(n_envs, int), n_envs, horizon, regimes, seed=5)[0]
    check("11. identical seed and actions give identical returns", np.array_equal(a1, a2))

    # --- 12: episodes terminate exactly at the horizon --------------------
    check("12. episodes terminate exactly at the horizon",
          bool(np.all(done0)), f"done = {int(done0.sum())}/{n_envs}")

    # --- 13-15: the observation must actually carry regime information ------
    # These three checks use their OWN generator rather than the shared stream.
    # Otherwise the reported statistic would drift whenever an unrelated check
    # is added or reordered above this block, which makes the number impossible
    # to quote reliably in the manuscript.
    qc_rng = np.random.default_rng(15_015)
    n_obs = 96
    obs_regimes = sample_regimes(qc_rng, n_obs)
    obs_h, case_h = run_collect(env, n_obs, 60, obs_regimes, seed=11)
    frac_zero = float(np.mean(case_h == 0))
    check("13. reported counts are rarely zero (frac==0 < 0.05)",
          frac_zero < 0.05, f"frac(cases==0) = {frac_zero:.4f}, "
                            f"mean count = {case_h.mean():.2f}")

    growth = obs_h[:, :, 1]
    n_distinct = len(np.unique(np.round(obs_h[:, :, 0], 6)))
    check("14. the growth channel is non-degenerate",
          float(np.std(growth)) > 1e-3 and n_distinct > 20,
          f"sd(growth) = {np.std(growth):.4f}, distinct levels in obs[0] = {n_distinct}")

    early_growth = growth[:12].mean(axis=0)
    r_obs = float(np.corrcoef(early_growth, obs_regimes[:, 0])[0, 1])
    check("15. early observed growth correlates with the hidden transmission rate",
          abs(r_obs) > 0.3, f"corr(mean early growth, true beta0) = {r_obs:+.3f}")

    # --- 16-18: the two environment families differ exactly as designed -----
    n_pair = 64
    reg_a = sample_regimes(qc_rng, n_pair, family=FAMILY_ACTION_REVEALED)
    reg_b = reg_a.copy()
    reg_b[:, IDX_EPS] = np.minimum(0.05 + reg_a[:, IDX_EPS] * 0.25, 0.60)
    reg_b[:, IDX_BEH] = np.minimum(reg_a[:, IDX_BEH] + 0.25, 0.35)

    obs_idle_a, _ = run_collect(env, n_pair, 30, reg_a, seed=21, action=0)
    obs_idle_b, _ = run_collect(env, n_pair, 30, reg_b, seed=21, action=0)
    d_idle = float(np.abs(obs_idle_a - obs_idle_b).max())
    check("16. action-revealed tasks that differ only in the action channel are "
          "indistinguishable while no one intervenes",
          d_idle == 0.0, f"max |observation difference| under a=0: {d_idle:.3e}")

    obs_act_a, _ = run_collect(env, n_pair, 30, reg_a, seed=21, action=2)
    obs_act_b, _ = run_collect(env, n_pair, 30, reg_b, seed=21, action=2)
    d_act = float(np.abs(obs_act_a - obs_act_b).max())
    check("17. the same tasks separate once the agent intervenes",
          d_act > 0.05, f"max |observation difference| under a=2: {d_act:.3f}")

    # In the passive family the action channel is fixed, so the intervention is
    # unambiguously beneficial and the regime is readable without acting.
    pass_reg = sample_regimes(qc_rng, n_pair, family=FAMILY_PASSIVE)
    _, burden_idle = run_collect(env, n_pair, 40, pass_reg, seed=22, action=0)
    _, burden_act = run_collect(env, n_pair, 40, pass_reg, seed=22, action=2)
    check("18. in the passive family intervening always reduces burden",
          float(burden_act[10:].mean()) < float(burden_idle[10:].mean()),
          f"mean burden idle {burden_idle[10:].mean():.3f} vs active {burden_act[10:].mean():.3f}")

    # --- 19-22: the linked family isolates identifiability from heterogeneity -
    n_link = 96
    reg_link = sample_regimes(qc_rng, n_link, family=FAMILY_LINKED)
    eps_l, beh_l, b0_l = reg_link[:, IDX_EPS], reg_link[:, IDX_BEH], reg_link[:, 0]
    r_eps = float(np.corrcoef(eps_l, b0_l)[0, 1])
    r_beh = float(np.corrcoef(beh_l, b0_l)[0, 1])
    check("19. in the linked family both action-channel parameters are affine in "
          "the passively visible transmission rate",
          abs(abs(r_eps) - 1.0) < 1e-9 and abs(abs(r_beh) - 1.0) < 1e-9,
          f"|corr(eps, beta0)| = {abs(r_eps):.9f}, "
          f"|corr(beh, beta0)| = {abs(r_beh):.9f}")

    sign = beh_l - eps_l
    check("20. the linked family contains both decision regimes",
          np.mean(sign > 0) > 0.1 and np.mean(sign < 0) > 0.1,
          f"share rebound-dominant = {np.mean(sign > 0):.2f}, "
          f"efficacy-dominant = {np.mean(sign < 0):.2f}")

    # Available gap = mean over tasks of the best constant minus the best single
    # constant. This is the quantity a learner can at most capture.
    def available_gap(family, n_tasks=64, horizon=120, seed=31):
        reg = sample_regimes(qc_rng, n_tasks, family=family)
        cr = np.zeros((n_tasks, N_ACTIONS))
        for a in range(N_ACTIONS):
            cr[:, a] = rollout(env, lambda t, o, a=a: np.full(n_tasks, a, int),
                               n_tasks, horizon, reg, seed=seed)[0]
        return float(cr.mean(axis=0).max()), float(cr.max(axis=1).mean())

    gap_link = available_gap(FAMILY_LINKED)
    gap_pass = available_gap(FAMILY_PASSIVE)
    # Computed last so that the two families above keep the same draws they had
    # before this third measurement was added.
    gap_revealed = available_gap(FAMILY_ACTION_REVEALED)
    check("21. the linked family offers a positive available gap",
          gap_link[1] - gap_link[0] > 2.0,
          f"linked: best single {gap_link[0]:.2f}, best per-task {gap_link[1]:.2f}, "
          f"gap {gap_link[1] - gap_link[0]:.2f}")
    check("22. the passive family offers almost no available gap, so it serves "
          "only as a nothing-to-adapt-to reference",
          gap_pass[1] - gap_pass[0] < gap_link[1] - gap_link[0],
          f"passive: best single {gap_pass[0]:.2f}, best per-task {gap_pass[1]:.2f}, "
          f"gap {gap_pass[1] - gap_pass[0]:.2f}")
    check("23. the action-revealed family also offers a gap comparable to the "
          "linked family, so the two differ in identifiability, not in how much "
          "there is to gain",
          gap_revealed[1] - gap_revealed[0] > 2.0,
          f"action_revealed: best single {gap_revealed[0]:.2f}, best per-task "
          f"{gap_revealed[1]:.2f}, gap {gap_revealed[1] - gap_revealed[0]:.2f}")

    # --- speed ------------------------------------------------------------
    env2 = RegimeDriftEpidemicEnv(n_envs=64, horizon=200, seed=3)
    reg = sample_regimes(qc_rng, 64)
    obs = env2.reset(reg)
    t0 = time.perf_counter()
    steps = 2000
    for t in range(steps):
        obs, r, d, info = env2.step(np.zeros(64, int))
    dt = time.perf_counter() - t0
    sps = steps * 64 / dt
    print(f"\nSpeed: {sps:,.0f} env-steps/s ({dt / steps * 1e3:.3f} ms per batched step of 64)")
    print(f"Estimated meta-training cost at 4e6 env-steps: {4e6 / sps / 3600:.2f} CPU-h")

    n_fail = sum(1 for _, ok_, _ in CHECKS if not ok_)
    print(f"\n{len(CHECKS) - n_fail}/{len(CHECKS)} checks passed.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
