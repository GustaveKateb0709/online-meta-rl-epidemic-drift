"""Baselines for the regime-drift meta-RL study.

All three baselines share one plain-PPO core (no latent context) with
hyper-parameters aligned to the meta-RL agent, so the comparison isolates the
effect of *online regime inference*.

* ``RetrainBaseline``  : train a fresh policy on each test task from scratch,
  under a bounded interaction budget (the actual budget used is reported).
* ``TransferBaseline`` : pre-train on the stationary task distribution, then
  fine-tune for a fixed number of steps on the test task.
* ``RobustBaseline``   : domain-randomised training on the task distribution,
  deployed with **no** adaptation (the "do-not-adapt" counterfactual).
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from src.common import DEFAULT_CONFIG, as_t, device
from src.commitment import wrap_cfg
from src.env import N_ACTIONS, OBS_DIM, RegimeDriftEpidemicEnv
from src.meta_rl import mlp, sample_train_regimes

_ADAPT_LOOKBACK = 15
_ADAPT_WINDOW = 5


# ---------------------------------------------------------------------- #
# plain PPO core
# ---------------------------------------------------------------------- #
class PPOAgent:
    """Standard PPO without any latent context (used by every baseline)."""

    def __init__(self, cfg: Optional[dict] = None, dev: Optional[torch.device] = None):
        c = dict(DEFAULT_CONFIG)
        if cfg:
            c.update(cfg)
        self.cfg = c
        self.dev = dev or device()
        h = c["hidden"]
        # Optional observation history (diagnostic switch, default OFF). With
        # history_len <= 1 the model input is exactly the current observation, so
        # the network and the behaviour are identical to the original baseline.
        self.history_len = int(c.get("history_len", 0) or 0)
        obs_w = OBS_DIM * max(1, self.history_len)
        self.policy = mlp([obs_w, h, h, N_ACTIONS]).to(self.dev)
        self.value = mlp([obs_w, h, h, 1]).to(self.dev)
        self.opt = torch.optim.Adam(list(self.policy.parameters()) + list(self.value.parameters()),
                                    lr=c["lr"])
        self._hist = None

    def reset_history(self) -> None:
        """Clear the observation history (a no-op unless history_len > 1)."""
        self._hist = None

    def _stack(self, obs: np.ndarray) -> np.ndarray:
        """Current observation, or the last ``history_len`` observations concatenated.

        The window is padded by repeating the first observation of the episode, so
        the input width is constant from step 0 on. The most recent observation
        occupies the last block.
        """
        k = self.history_len
        if k <= 1:
            return obs
        o = np.asarray(obs, dtype=np.float32)
        if self._hist is None:
            self._hist = np.repeat(o[None, :, :], k, axis=0)
        else:
            self._hist = np.concatenate([self._hist[1:], o[None, :, :]], axis=0)
        return self._hist.transpose(1, 0, 2).reshape(o.shape[0], -1)

    @torch.no_grad()
    def act(self, obs_np, deterministic: bool = True):
        obs = as_t(self._stack(np.asarray(obs_np, dtype=np.float32))).to(self.dev)
        logits = self.policy(obs)
        dist = Categorical(logits=logits)
        a = logits.argmax(-1) if deterministic else dist.sample()
        return a.cpu().numpy()

    @torch.no_grad()
    def collect(self, env, reset_fn: Callable, horizon: int) -> Dict[str, torch.Tensor]:
        c = self.cfg
        obs = reset_fn(env)
        self.reset_history()
        B = env.n_envs
        w = OBS_DIM * max(1, self.history_len)
        O = np.zeros((horizon, B, w), np.float32)
        A = np.zeros((horizon, B), np.int64)
        LP = np.zeros((horizon, B), np.float32)
        RS = np.zeros((horizon, B), np.float32)
        RAW = np.zeros((horizon, B), np.float32)
        D = np.zeros((horizon, B), np.float32)
        V = np.zeros((horizon, B), np.float32)
        for t in range(horizon):
            obs_in = self._stack(obs)
            obs_t = as_t(obs_in).to(self.dev)
            logits = self.policy(obs_t)
            dist = Categorical(logits=logits)
            a = dist.sample()
            logp = dist.log_prob(a)
            v = self.value(obs_t).squeeze(-1)
            nobs, raw, done, _ = env.step(a.cpu().numpy())
            O[t], A[t] = obs_in, a.cpu().numpy()
            LP[t], V[t] = logp.cpu().numpy(), v.cpu().numpy()
            RS[t], RAW[t] = raw * c["reward_scale"], raw
            D[t] = done.astype(np.float32)
            obs = nobs
        return {"obs": torch.as_tensor(O), "act": torch.as_tensor(A),
                "logp": torch.as_tensor(LP), "rew": torch.as_tensor(RS),
                "done": torch.as_tensor(D), "val": torch.as_tensor(V),
                "raw_rew": torch.as_tensor(RAW)}

    def update(self, roll: Dict[str, torch.Tensor]) -> Dict[str, float]:
        c = self.cfg
        dev = self.dev
        obs, act, logp_old = roll["obs"].to(dev), roll["act"].to(dev), roll["logp"].to(dev)
        rew, done, val = roll["rew"].to(dev), roll["done"].to(dev), roll["val"].to(dev)
        H, B = rew.shape
        adv = torch.zeros_like(rew)
        last = torch.zeros(B, device=dev)
        for t in reversed(range(H)):
            nextval = val[t + 1] if t + 1 < H else torch.zeros(B, device=dev)
            nonterm = 1.0 - done[t]
            delta = rew[t] + c["gamma"] * nextval * nonterm - val[t]
            last = delta + c["gamma"] * c["lam"] * nonterm * last
            adv[t] = last
        ret = adv + val
        f_obs, f_act = obs.reshape(H * B, -1), act.reshape(H * B)
        f_logp, f_ret = logp_old.reshape(H * B), ret.reshape(H * B)
        f_adv = adv.reshape(H * B)
        f_adv = (f_adv - f_adv.mean()) / (f_adv.std() + 1e-8)
        N = H * B
        mb = max(1, N // c["n_minibatch"])
        pg_l = vf_l = 0.0
        for _ in range(c["ppo_epochs"]):
            perm = torch.randperm(N, device=dev)
            for s in range(0, N, mb):
                idx = perm[s:s + mb]
                dist = Categorical(logits=self.policy(f_obs[idx]))
                ratio = torch.exp(dist.log_prob(f_act[idx]) - f_logp[idx])
                pg = -torch.min(ratio * f_adv[idx],
                                torch.clamp(ratio, 1 - c["clip"], 1 + c["clip"]) * f_adv[idx]).mean()
                vf = ((self.value(f_obs[idx]).squeeze(-1) - f_ret[idx]) ** 2).mean()
                loss = pg + c["value_coef"] * vf
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.policy.parameters()) + list(self.value.parameters()),
                                         c["max_grad_norm"])
                self.opt.step()
                pg_l += float(pg.detach()); vf_l += float(vf.detach())
        return {"policy_loss": pg_l / max(1, c["ppo_epochs"]), "value_loss": vf_l / max(1, c["ppo_epochs"])}

    def n_params(self) -> int:
        return sum(p.numel() for p in self.policy.parameters()) + \
               sum(p.numel() for p in self.value.parameters())

    def state_dict(self) -> dict:
        return {"policy": self.policy.state_dict(), "value": self.value.state_dict()}

    def load_state_dict(self, blob: dict) -> None:
        self.policy.load_state_dict(blob["policy"])
        self.value.load_state_dict(blob["value"])


# ---------------------------------------------------------------------- #
# evaluation helpers
# ---------------------------------------------------------------------- #
class FixedController:
    """Wrap a fitted PPO policy as a stateless controller."""

    def __init__(self, agent: PPOAgent):
        self.agent = agent

    def reset(self, env=None):
        # Clear any observation history so each evaluation episode starts fresh
        # (a no-op unless the policy was trained with history_len > 1).
        self.agent.reset_history()

    def act(self, obs, t, info):
        return self.agent.act(obs, deterministic=True)


class ConstantController:
    """Heuristic reference that always plays the same intervention level."""

    def __init__(self, action: int):
        self.action = int(action)

    def reset(self, env=None):
        pass

    def act(self, obs, t, info):
        return np.full(np.asarray(obs).shape[0], self.action, dtype=int)


def rollout_episode(env, controller, drift: dict, seed: int) -> np.ndarray:
    """Run one batched episode on drifting tasks; return raw rewards (H, B)."""
    controller.reset()
    obs = env.reset(drift["regime0"], regime1=drift["regime1"], onset=drift["onset"],
                    duration=drift["duration"], abrupt=drift["abrupt"], seed=seed)
    H, B = env.horizon, env.n_envs
    rew = np.zeros((H, B), np.float64)
    info = None
    for t in range(H):
        a = controller.act(obs, t, info)
        obs, r, done, info = env.step(np.asarray(a, dtype=int))
        rew[t] = r
    return rew


ADAPTATION_STEPS_DEFINITION = (
    "Per episode: pre = mean reward over the ADAPT_LOOKBACK=15 steps strictly "
    "before drift onset; adaptation_steps = the first post-onset step t (0-indexed "
    "offset from onset) whose forward ADAPT_WINDOW=5-step mean reward is >= pre; "
    "censored at horizon-onset if recovery is never reached. Reported as the "
    "distribution across episodes (mean, median, q1, q3)."
)


def adaptation_steps_array(rew: np.ndarray, onset: np.ndarray, horizon: int) -> np.ndarray:
    """Per-episode steps after drift onset to recover the pre-drift reward level."""
    H, B = rew.shape
    out = []
    for i in range(B):
        o = int(onset[i])
        lo = max(0, o - _ADAPT_LOOKBACK)
        pre = rew[lo:o, i].mean() if o > lo else rew[:max(1, o), i].mean()
        steps = horizon - o
        for t in range(o, H):
            win = rew[t:min(H, t + _ADAPT_WINDOW), i]
            if win.mean() >= pre:
                steps = t - o
                break
        out.append(steps)
    return np.asarray(out, dtype=float)


def adaptation_steps(rew: np.ndarray, onset: np.ndarray, horizon: int) -> float:
    """Mean steps after drift onset to recover the pre-drift reward level."""
    return float(adaptation_steps_array(rew, onset, horizon).mean())


def eval_metrics(rew: np.ndarray, onset: np.ndarray, horizon: int) -> Dict[str, float]:
    total = rew.sum(axis=0)
    t = np.arange(rew.shape[0])[:, None]
    post_mask = t >= np.asarray(onset)[None, :]
    post = (rew * post_mask).sum(axis=0)
    a = adaptation_steps_array(rew, onset, horizon)
    return {
        "return_mean": float(total.mean()),
        "return_sd": float(total.std(ddof=1)) if total.size > 1 else 0.0,
        "post_return_mean": float(post.mean()),
        "post_return_sd": float(post.std(ddof=1)) if post.size > 1 else 0.0,
        "adaptation_steps": float(a.mean()),
        "adaptation_steps_median": float(np.median(a)),
        "adaptation_steps_q1": float(np.quantile(a, 0.25)),
        "adaptation_steps_q3": float(np.quantile(a, 0.75)),
    }


# ---------------------------------------------------------------------- #
# per-task constant baselines + the 'captured' primary criterion
# ---------------------------------------------------------------------- #
AGENT_NAMES = ("meta_online", "meta_frozen", "meta_oracle", "robust", "transfer",
               "retrain", "const0", "const1", "const2")


def constant_baseline_block(const_per_task: Dict[int, np.ndarray], n_tasks: int) -> Dict[str, object]:
    """Per-task constant-action returns and the reference quantities for ``captured``.

    ``captured`` (defined per agent) is the share of the available gap that the
    agent obtains:
        captured = (return - best_single_const) / (best_per_task_const - best_single_const)
    where ``best_single_const = max_a mean_task const_a`` and
    ``best_per_task_const = mean_task max_a const_a``. The denominator needs the
    per-task returns, which is why they are stored explicitly here.
    """
    mat = np.stack([np.asarray(const_per_task[a], dtype=float) for a in (0, 1, 2)], axis=1)
    best_single = float(mat.mean(0).max())
    best_per_task = float(mat.max(1).mean())
    return {
        "const_return_per_task": {str(a): [float(x) for x in const_per_task[a]]
                                  for a in (0, 1, 2)},
        "a_star_per_task": [int(x) for x in mat.argmax(1)],
        "best_single_const": best_single,
        "best_per_task_const": best_per_task,
        "available_gap": float(best_per_task - best_single),
        "n_tasks": int(n_tasks),
    }


def attach_captured(row: Dict[str, dict], block: Dict[str, object],
                    names=AGENT_NAMES) -> None:
    """Add ``captured`` / ``captured_post`` to each agent dict in ``row`` (in place).

    Values are NOT clipped: a negative value means the agent is worse than the
    best single constant action, which is real information.
    """
    gap = float(block["available_gap"])
    base = float(block["best_single_const"])
    for name in names:
        r = row.get(name)
        if not isinstance(r, dict) or "return_mean" not in r:
            continue
        r["captured"] = (float((r["return_mean"] - base) / gap)
                         if abs(gap) > 1e-12 else None)
        r["captured_post"] = (float((r["post_return_mean"] - base) / gap)
                              if abs(gap) > 1e-12 else None)


# ---------------------------------------------------------------------- #
# baselines
# ---------------------------------------------------------------------- #
def _tile_drift(drift: dict, i: int, n_rep: int) -> dict:
    out = {}
    for k in ("regime0", "regime1"):
        out[k] = np.tile(drift[k][i:i + 1], (n_rep, 1))
    out["onset"] = np.full(n_rep, int(drift["onset"][i]))
    out["duration"] = np.full(n_rep, int(drift["duration"][i]))
    out["abrupt"] = np.full(n_rep, bool(drift["abrupt"][i]))
    return out


class _BaseBaseline:
    name = "base"

    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = dict(DEFAULT_CONFIG)
        if cfg:
            self.cfg.update(cfg)

    def _interactions(self) -> int:
        return int(getattr(self, "_steps", 0))


class RobustBaseline(_BaseBaseline):
    """Domain-randomised training, deployed without adaptation."""

    name = "robust"

    def fit(self, seed: int) -> PPOAgent:
        c = self.cfg
        rng = np.random.default_rng(seed)
        env = RegimeDriftEpidemicEnv(n_envs=c["n_tasks"], horizon=c["horizon"], seed=seed)
        env = wrap_cfg(env, c)
        agent = PPOAgent(c)
        steps = 0
        for _ in range(c["iterations"]):
            regs = sample_train_regimes(rng, c["n_tasks"], family=c.get("family", "passive"))
            roll = agent.collect(env, lambda e: e.reset(regs), c["horizon"])
            agent.update(roll)
            steps += c["n_tasks"] * c["horizon"]
        self._steps = steps
        self.agent = agent
        return agent

    def evaluate(self, drift: dict, eval_env: RegimeDriftEpidemicEnv, seed: int) -> Dict[str, float]:
        rew = rollout_episode(wrap_cfg(eval_env, self.cfg), FixedController(self.agent), drift, seed)
        return eval_metrics(rew, drift["onset"], eval_env.horizon)


class TransferBaseline(_BaseBaseline):
    """Pre-train on the task distribution, then fine-tune a fixed number of steps."""

    name = "transfer"

    def __init__(self, cfg: Optional[dict] = None, finetune_steps: int = 4000):
        super().__init__(cfg)
        self.finetune_steps = int(finetune_steps)

    def fit(self, seed: int) -> PPOAgent:
        c = self.cfg
        rng = np.random.default_rng(seed)
        env = RegimeDriftEpidemicEnv(n_envs=c["n_tasks"], horizon=c["horizon"], seed=seed)
        env = wrap_cfg(env, c)
        agent = PPOAgent(c)
        steps = 0
        for _ in range(c["iterations"]):
            regs = sample_train_regimes(rng, c["n_tasks"], family=c.get("family", "passive"))
            roll = agent.collect(env, lambda e: e.reset(regs), c["horizon"])
            agent.update(roll)
            steps += c["n_tasks"] * c["horizon"]
        self._pretrain_steps = steps
        self._steps = steps
        self.agent = agent
        self._dev = agent.dev
        return agent

    def finetune(self, drift: dict, eval_env: RegimeDriftEpidemicEnv, seed: int) -> int:
        """Fine-tune on the (drifting) test tasks; returns interactions used."""
        c = self.cfg
        rng = np.random.default_rng(seed + 7919)
        used = 0
        per_iter = eval_env.n_envs * eval_env.horizon
        n_iter = max(1, int(round(self.finetune_steps / per_iter)))
        fit_env = wrap_cfg(eval_env, c)

        def reset(e):
            d = drift
            return e.reset(d["regime0"], regime1=d["regime1"], onset=d["onset"],
                           duration=d["duration"], abrupt=d["abrupt"],
                           seed=int(rng.integers(1 << 30)))

        for _ in range(n_iter):
            roll = self.agent.collect(fit_env, reset, eval_env.horizon)
            self.agent.update(roll)
            used += per_iter
        self._steps = self._pretrain_steps + used
        return used

    def evaluate(self, drift: dict, eval_env: RegimeDriftEpidemicEnv, seed: int) -> Dict[str, float]:
        rew = rollout_episode(wrap_cfg(eval_env, self.cfg), FixedController(self.agent), drift, seed)
        return eval_metrics(rew, drift["onset"], eval_env.horizon)


class RetrainBaseline(_BaseBaseline):
    """Train a fresh policy per test task, under a bounded interaction budget."""

    name = "retrain"

    def __init__(self, cfg: Optional[dict] = None, budget_steps: int = 4000,
                 n_replicas: int = 8):
        super().__init__(cfg)
        self.budget_steps = int(budget_steps)
        self.n_replicas = int(n_replicas)

    def fit_and_evaluate(self, drift: dict, seed: int) -> Dict[str, float]:
        c = self.cfg
        n_test = len(drift["onset"])
        H = c["horizon"]
        per_iter = self.n_replicas * H
        n_iter = max(1, int(round(self.budget_steps / per_iter)))
        used_per_task = n_iter * per_iter
        per = []
        for i in range(n_test):
            rng = np.random.default_rng(seed * 1000003 + i)
            env = RegimeDriftEpidemicEnv(n_envs=self.n_replicas, horizon=H, seed=seed + i + 1)
            env = wrap_cfg(env, c)
            agent = PPOAgent(c)
            d_i = _tile_drift(drift, i, self.n_replicas)

            def reset(e, d=d_i):
                return e.reset(d["regime0"], regime1=d["regime1"], onset=d["onset"],
                               duration=d["duration"], abrupt=d["abrupt"],
                               seed=int(rng.integers(1 << 30)))

            for _ in range(n_iter):
                roll = agent.collect(env, reset, H)
                agent.update(roll)
            # evaluate on one deterministic episode of this single task
            ev = RegimeDriftEpidemicEnv(n_envs=1, horizon=H, seed=seed + 555)
            ev = wrap_cfg(ev, c)
            d_eval = _tile_drift(drift, i, 1)
            rew = rollout_episode(ev, FixedController(agent), d_eval, seed + 999)
            per.append(eval_metrics(rew, d_eval["onset"], H))
        self._steps = used_per_task * n_test
        self._used_per_task = used_per_task
        # Average every metric across the freshly-retrained per-task agents, so the
        # result dict matches the schema produced by eval_metrics for all agents.
        return {k: float(np.mean([p[k] for p in per])) for k in per[0]}
