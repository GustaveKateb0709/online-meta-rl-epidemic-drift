"""PEARL-style meta-reinforcement learning with probabilistic context inference.

Design
------
The agent treats the (unobservable) regime as a latent variable ``z``. A
permutation-invariant *context encoder* maps a set of transitions to a diagonal
Gaussian posterior ``q(z | c)``. A policy and a value function are conditioned
on ``(observation, z)``. A *decoder* reconstructs reward and the next
observation from ``(observation, action, z)``; this is an unsupervised
auxiliary objective that never uses the true regime vector, which is the
technical pre-condition for the paper's "label-free" claim.

Parameter inference
-------------------
``z_mode`` controls how ``z`` is produced at decision time:

* ``online``  : re-infer ``z`` from the most recent ``ctx_len`` transitions at
  every step (the paper's proposal);
* ``frozen``  : infer ``z`` once from the first ``ctx_len`` transitions and
  freeze it for the rest of the episode (ablation);
* ``oracle``  : use the true regime vector as ``z`` (analysis-only upper bound,
  read from ``env.current_regime()``).

The policy is conditioned on the posterior **mean** ``mu`` (not a sample); this
removes per-step latent noise and, crucially, makes the PPO importance ratio
exactly one at the start of an update when ``z`` is recomputed with gradients.

Training objective
------------------
``PPO`` clipped surrogate + value MSE + ``KL(q(z|c) || N(0, I))`` + decoder
negative log-likelihood. When ``z_grad`` is true (default) the policy/value
losses are differentiated *through* ``mu`` back into the encoder, so the encoder
is trained by both the auxiliary objective and the control objective.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from src.common import DEFAULT_CONFIG, device
from src.commitment import wrap_cfg
from src.env import (FAMILY_PASSIVE, N_ACTIONS, OBS_DIM, REGIME_DIM,
                     RegimeDriftEpidemicEnv)


def mlp(sizes, activation=nn.ReLU):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class ContextEncoder(nn.Module):
    """Permutation-invariant map from a variable-length transition set to ``q(z|c)``.

    Each transition (obs, action_onehot, reward, next_obs) is embedded by a
    shared MLP, then averaged (mean pooling) so the output is invariant to the
    order of the transitions. This is what allows re-inference from an
    arbitrary sliding window of the most recent K steps.
    """

    def __init__(self, obs_dim: int = OBS_DIM, n_actions: int = N_ACTIONS,
                 z_dim: int = 5, hidden: int = 128):
        super().__init__()
        in_dim = 2 * obs_dim + n_actions + 1
        self.mlp = mlp([in_dim, hidden, hidden])
        self.fc_mu = nn.Linear(hidden, z_dim)
        self.fc_logvar = nn.Linear(hidden, z_dim)
        self.z_dim = z_dim

    def forward(self, obs, act_oh, rew, next_obs, mask):
        x = torch.cat([obs, act_oh, rew.unsqueeze(-1), next_obs], dim=-1)
        h = self.mlp(x)
        h = h * mask.unsqueeze(-1)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        h = h.sum(dim=1) / denom
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        has = (mask.sum(dim=1, keepdim=True) > 0).to(mu.dtype)
        return mu * has, logvar * has  # empty context -> standard-normal prior


class Policy(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, z_dim: int = 5,
                 n_actions: int = N_ACTIONS, hidden: int = 128):
        super().__init__()
        self.mlp = mlp([obs_dim + z_dim, hidden, hidden, n_actions])

    def forward(self, obs, z):
        return self.mlp(torch.cat([obs, z], dim=-1))


class ValueNet(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, z_dim: int = 5, hidden: int = 128):
        super().__init__()
        self.mlp = mlp([obs_dim + z_dim, hidden, hidden, 1])

    def forward(self, obs, z):
        return self.mlp(torch.cat([obs, z], dim=-1)).squeeze(-1)


class Decoder(nn.Module):
    """Unsupervised auxiliary head: predict (reward, next_obs) from (obs, action, z).

    No regime label is ever used -- only observable quantities. This is what
    makes the context posterior learnable from reward/observation streams alone.
    """

    def __init__(self, obs_dim: int = OBS_DIM, n_actions: int = N_ACTIONS,
                 z_dim: int = 5, hidden: int = 128):
        super().__init__()
        self.mlp = mlp([obs_dim + n_actions + z_dim, hidden, hidden])
        self.r_head = nn.Linear(hidden, 1)
        self.o_head = nn.Linear(hidden, obs_dim)

    def forward(self, obs, act_oh, z):
        h = self.mlp(torch.cat([obs, act_oh, z], dim=-1))
        return self.r_head(h).squeeze(-1), self.o_head(h)


class ContextBuffer:
    """Fixed-length sliding window of recent transitions, per parallel env."""

    def __init__(self, n_envs: int, k: int, obs_dim: int, n_actions: int):
        self.n = n_envs
        self.k = k
        self.obs = np.zeros((n_envs, k, obs_dim), dtype=np.float32)
        self.act = np.zeros((n_envs, k, n_actions), dtype=np.float32)
        self.rew = np.zeros((n_envs, k), dtype=np.float32)
        self.nobs = np.zeros((n_envs, k, obs_dim), dtype=np.float32)
        self.mask = np.zeros((n_envs, k), dtype=np.float32)
        self.count = np.zeros(n_envs, dtype=int)

    def push(self, obs, act_idx, rew, next_obs):
        k = self.k
        for i in range(self.n):
            c = self.count[i]
            if c < k:
                j = c
            else:
                self.obs[i, :-1] = self.obs[i, 1:]
                self.act[i, :-1] = self.act[i, 1:]
                self.rew[i, :-1] = self.rew[i, 1:]
                self.nobs[i, :-1] = self.nobs[i, 1:]
                self.mask[i, :-1] = self.mask[i, 1:]
                j = k - 1
            self.obs[i, j] = obs[i]
            self.act[i, j] = _onehot(int(act_idx[i]), self.act.shape[-1])
            self.rew[i, j] = rew[i]
            self.nobs[i, j] = next_obs[i]
            self.mask[i, j] = 1.0
            self.count[i] = min(c + 1, k)

    def tensors(self):
        return (torch.as_tensor(self.obs), torch.as_tensor(self.act),
                torch.as_tensor(self.rew), torch.as_tensor(self.nobs),
                torch.as_tensor(self.mask))

    def copy(self) -> "ContextBuffer":
        b = ContextBuffer(self.n, self.k, self.obs.shape[-1], self.act.shape[-1])
        b.obs = self.obs.copy()
        b.act = self.act.copy()
        b.rew = self.rew.copy()
        b.nobs = self.nobs.copy()
        b.mask = self.mask.copy()
        b.count = self.count.copy()
        return b


def _onehot(idx: int, n: int) -> np.ndarray:
    v = np.zeros(n, dtype=np.float32)
    v[idx] = 1.0
    return v


class MetaRLAgent:
    """Probabilistic context meta-RL agent (context encoder + z-conditioned PPO)."""

    def __init__(self, cfg: Optional[dict] = None, dev: Optional[torch.device] = None):
        c = dict(DEFAULT_CONFIG)
        if cfg:
            c.update(cfg)
        self.cfg = c
        self.dev = dev or device()
        z_mode = c["z_mode"]
        self.z_dim = int(c["z_dim"])
        self.policy_z_dim = REGIME_DIM if z_mode == "oracle" else self.z_dim

        self.encoder = ContextEncoder(OBS_DIM, N_ACTIONS, self.z_dim, c["hidden"]).to(self.dev)
        self.policy = Policy(OBS_DIM, self.policy_z_dim, N_ACTIONS, c["hidden"]).to(self.dev)
        self.value = ValueNet(OBS_DIM, self.policy_z_dim, c["hidden"]).to(self.dev)
        self.decoder = Decoder(OBS_DIM, N_ACTIONS, self.z_dim, c["hidden"]).to(self.dev)

        params = (list(self.encoder.parameters()) + list(self.policy.parameters())
                  + list(self.value.parameters()) + list(self.decoder.parameters()))
        self.opt = torch.optim.Adam(params, lr=c["lr"])

    # ------------------------------------------------------------------ #
    # inference / acting
    # ------------------------------------------------------------------ #
    def infer_z(self, context: Dict[str, torch.Tensor], deterministic: bool = False):
        """Reparameterised posterior sample from a context set.

        ``context`` holds tensors obs/act_oh/rew/next_obs/mask of shape
        (B, K, ...). Returns ``(mu, logvar, z)``.
        """
        mu, logvar = self.encoder(context["obs"], context["act_oh"],
                                  context["rew"], context["next_obs"], context["mask"])
        if deterministic:
            z = mu
        else:
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return mu, logvar, z

    def act(self, obs, z, deterministic: bool = False):
        logits = self.policy(obs, z)
        dist = Categorical(logits=logits)
        a = logits.argmax(dim=-1) if deterministic else dist.sample()
        return a, dist.log_prob(a)

    def _step_z(self, buf: ContextBuffer, env, z_mode: str,
                frozen_z: torch.Tensor, frozen_ready: torch.Tensor):
        """Policy conditioning at the current step (posterior mean, or regime)."""
        if z_mode == "oracle":
            z = torch.as_tensor(env.current_regime(), dtype=torch.float32, device=self.dev)
            return z, frozen_z, frozen_ready
        o, a, r, no, m = buf.tensors()
        o, a, r, no, m = o.to(self.dev), a.to(self.dev), r.to(self.dev), no.to(self.dev), m.to(self.dev)
        mu, _ = self.encoder(o, a, r, no, m)
        if z_mode == "frozen":
            ready_now = (buf.count >= buf.k)
            new = torch.as_tensor(ready_now, device=self.dev) & (~frozen_ready)
            if new.any():
                frozen_z = torch.where(new[:, None], mu.detach(), frozen_z)
                frozen_ready = frozen_ready | new
            return torch.where(frozen_ready[:, None], frozen_z, mu), frozen_z, frozen_ready
        return mu, frozen_z, frozen_ready

    # ------------------------------------------------------------------ #
    # rollout collection
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def collect_rollout(self, env, regimes: np.ndarray, z_mode: Optional[str] = None,
                        deterministic: bool = False):
        """Run one batched episode over tasks and return a PPO batch."""
        c = self.cfg
        z_mode = z_mode or c["z_mode"]
        B, H, K = env.n_envs, c["horizon"], c["ctx_len"]
        obs = env.reset(regimes)
        buf = ContextBuffer(B, K, OBS_DIM, N_ACTIONS)
        frozen_z = torch.zeros(B, self.policy_z_dim, device=self.dev)
        frozen_ready = torch.zeros(B, dtype=torch.bool, device=self.dev)

        O = np.zeros((H, B, OBS_DIM), np.float32)
        NO = np.zeros((H, B, OBS_DIM), np.float32)
        A = np.zeros((H, B), np.int64)
        LP = np.zeros((H, B), np.float32)
        RS = np.zeros((H, B), np.float32)
        RAW = np.zeros((H, B), np.float32)
        D = np.zeros((H, B), np.float32)
        V = np.zeros((H, B), np.float32)
        Z = np.zeros((H, B, self.policy_z_dim), np.float32)

        for t in range(H):
            z, frozen_z, frozen_ready = self._step_z(buf, env, z_mode, frozen_z, frozen_ready)
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.dev)
            logits = self.policy(obs_t, z)
            dist = Categorical(logits=logits)
            a = logits.argmax(-1) if deterministic else dist.sample()
            logp = dist.log_prob(a)
            v = self.value(obs_t, z)
            a_np = a.cpu().numpy()
            nobs, raw_r, done, _ = env.step(a_np)
            rew_s = raw_r * c["reward_scale"]
            buf.push(obs, a_np, rew_s, nobs)

            O[t] = obs
            NO[t] = nobs
            A[t] = a_np
            LP[t] = logp.cpu().numpy()
            RS[t] = rew_s
            RAW[t] = raw_r
            D[t] = done.astype(np.float32)
            V[t] = v.cpu().numpy()
            Z[t] = z.cpu().numpy()
            obs = nobs

        return {"obs": torch.as_tensor(O), "next_obs": torch.as_tensor(NO),
                "act": torch.as_tensor(A), "logp": torch.as_tensor(LP),
                "rew": torch.as_tensor(RS), "done": torch.as_tensor(D),
                "val": torch.as_tensor(V), "z": torch.as_tensor(Z),
                "raw_rew": torch.as_tensor(RAW)}

    # ------------------------------------------------------------------ #
    # context windows reconstructed from a rollout (for gradient to encoder)
    # ------------------------------------------------------------------ #
    def _gather_context(self, roll, t_idx, env_idx, z_mode):
        """Context of transitions strictly BEFORE the step, as tensors.

        Reproduces the sliding window from the stored rollout so that ``mu`` can
        be recomputed with gradients inside the PPO update.
        """
        K = self.cfg["ctx_len"]
        dev = self.dev
        O, NO, A, RS = roll["obs"].to(dev), roll["next_obs"].to(dev), roll["act"].to(dev), roll["rew"].to(dev)
        offs = torch.arange(K, device=dev)
        if z_mode == "frozen":
            cnt = torch.clamp(t_idx, max=K)
            j = offs[None, :].expand(t_idx.shape[0], K)
            mask = (j < cnt[:, None]).float()
            jj = j.clamp(max=K - 1)
        else:  # online
            j = t_idx[:, None] - 1 - offs[None, :]
            mask = (j >= 0).float()
            jj = j.clamp(min=0)
        o = O[jj, env_idx[:, None]]
        no = NO[jj, env_idx[:, None]]
        a = A[jj, env_idx[:, None]]
        r = RS[jj, env_idx[:, None]]
        a_oh = F.one_hot(a, N_ACTIONS).float()
        return o, a_oh, r, no, mask

    # ------------------------------------------------------------------ #
    # learning
    # ------------------------------------------------------------------ #
    def update(self, roll: Dict[str, torch.Tensor]) -> Dict[str, float]:
        c = self.cfg
        dev = self.dev
        obs, act, logp_old = roll["obs"].to(dev), roll["act"].to(dev), roll["logp"].to(dev)
        rew, done, val, z = roll["rew"].to(dev), roll["done"].to(dev), roll["val"].to(dev), roll["z"].to(dev)
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

        f_obs = obs.reshape(H * B, -1)
        f_act = act.reshape(H * B)
        f_logp = logp_old.reshape(H * B)
        f_adv = adv.reshape(H * B)
        f_ret = ret.reshape(H * B)
        f_z = z.reshape(H * B, -1)
        f_adv = (f_adv - f_adv.mean()) / (f_adv.std() + 1e-8)

        z_mode = c["z_mode"]
        z_grad = bool(c.get("z_grad", True)) and z_mode != "oracle"
        flat_t = torch.arange(H * B, device=dev) // B
        flat_i = torch.arange(H * B, device=dev) % B

        N = H * B
        mb_size = max(1, N // c["n_minibatch"])
        pg_loss = vf_loss = ent_loss = 0.0
        for _ in range(c["ppo_epochs"]):
            perm = torch.randperm(N, device=dev)
            for s in range(0, N, mb_size):
                idx = perm[s:s + mb_size]
                if z_grad:
                    ctx = self._gather_context(roll, flat_t[idx], flat_i[idx], z_mode)
                    mu_mb, _ = self.encoder(*ctx)
                    z_mb = mu_mb
                else:
                    z_mb = f_z[idx]
                dist = Categorical(logits=self.policy(f_obs[idx], z_mb))
                logp = dist.log_prob(f_act[idx])
                ratio = torch.exp(logp - f_logp[idx])
                s1 = ratio * f_adv[idx]
                s2 = torch.clamp(ratio, 1 - c["clip"], 1 + c["clip"]) * f_adv[idx]
                pg = -torch.min(s1, s2).mean()
                v = self.value(f_obs[idx], z_mb)
                vf = ((v - f_ret[idx]) ** 2).mean()
                ent = dist.entropy().mean()
                loss = pg + c["value_coef"] * vf - c["entropy_coef"] * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._params(), c["max_grad_norm"])
                self.opt.step()
                pg_loss += float(pg.detach()); vf_loss += float(vf.detach()); ent_loss += float(ent.detach())
        n_updates = c["ppo_epochs"] * max(1, (N + mb_size - 1) // mb_size)

        aux = self._aux_step(roll, c["aux_batch"])
        return {"policy_loss": pg_loss / n_updates, "value_loss": vf_loss / n_updates,
                "entropy": ent_loss / n_updates, "kl": aux["kl"], "decoder": aux["decoder"],
                "adv_mean": float(f_adv.mean())}

    def _params(self):
        return (list(self.encoder.parameters()) + list(self.policy.parameters())
                + list(self.value.parameters()) + list(self.decoder.parameters()))

    def _aux_step(self, roll: Dict[str, torch.Tensor], m: int) -> Dict[str, float]:
        c = self.cfg
        if c["z_mode"] == "oracle":
            return {"kl": 0.0, "decoder": 0.0}
        dev = self.dev
        H, B = roll["rew"].shape
        K = c["ctx_len"]
        t_obs = roll["obs"].to(dev)
        t_no = roll["next_obs"].to(dev)
        t_act = roll["act"].to(dev)
        t_rew = roll["rew"].to(dev)
        t_oh = F.one_hot(t_act, N_ACTIONS).float()

        n = min(m, H * B)
        env_idx = torch.randint(0, B, (n,), device=dev)
        t_idx = torch.randint(0, H, (n,), device=dev)
        offs = torch.arange(K, device=dev)
        idx = t_idx[:, None] - offs[None, :]
        mask = (idx >= 0).float()
        idxc = idx.clamp(min=0)
        o = t_obs[idxc, env_idx[:, None]]
        no = t_no[idxc, env_idx[:, None]]
        a = t_oh[idxc, env_idx[:, None]]
        r = t_rew[idxc, env_idx[:, None]]
        mu, logvar = self.encoder(o, a, r, no, mask)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(-1).mean()
        r_hat, no_hat = self.decoder(o, a, z.unsqueeze(1).expand(-1, K, -1))
        rec_r = ((r_hat - r) ** 2)
        rec_o = ((no_hat - no) ** 2).mean(-1)
        w = mask / mask.sum(-1, keepdim=True).clamp(min=1.0)
        recon = (w * (rec_r + rec_o)).sum(-1).mean()
        loss = c["kl_coef"] * kl + c["decoder_coef"] * recon
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._params(), c["max_grad_norm"])
        self.opt.step()
        return {"kl": float(kl.detach()), "decoder": float(recon.detach())}

    # ------------------------------------------------------------------ #
    # persistence
    # ------------------------------------------------------------------ #
    def state_dict(self) -> dict:
        return {"encoder": self.encoder.state_dict(), "policy": self.policy.state_dict(),
                "value": self.value.state_dict(), "decoder": self.decoder.state_dict(),
                "cfg": self.cfg}

    def save(self, path) -> None:
        torch.save(self.state_dict(), str(path))

    def load(self, path) -> None:
        blob = torch.load(str(path), map_location=self.dev, weights_only=False)
        self.encoder.load_state_dict(blob["encoder"])
        self.policy.load_state_dict(blob["policy"])
        self.value.load_state_dict(blob["value"])
        self.decoder.load_state_dict(blob["decoder"])

    def n_params(self) -> int:
        return sum(p.numel() for p in self._params())


def sample_train_regimes(rng: np.random.Generator, n: int, family: str = FAMILY_PASSIVE) -> np.ndarray:
    """Draw stationary regimes from the meta-training support of one family."""
    from src.env import sample_regimes
    return sample_regimes(rng, n, family=family)


def train_meta(agent: MetaRLAgent, seed: int, log_every: int = 0,
               verbose: bool = True, commitment: Optional[bool] = None) -> dict:
    """Meta-train on the stationary task distribution of the agent's family.

    ``commitment`` overrides the config switch for this call; when on (config or
    argument), the training environment locks one intervention level per decision
    cycle through the shared commitment wrapper (single code path).
    """
    import time as _time
    c = agent.cfg
    rng = np.random.default_rng(seed)
    env = RegimeDriftEpidemicEnv(n_envs=c["n_tasks"], horizon=c["horizon"], seed=seed)
    env = wrap_cfg(env, c, commitment)
    total_steps = 0
    t0 = _time.perf_counter()
    hist = []
    last_ret = float("nan")
    for it in range(c["iterations"]):
        regimes = sample_train_regimes(rng, c["n_tasks"], family=c.get("family", FAMILY_PASSIVE))
        roll = agent.collect_rollout(env, regimes)
        losses = agent.update(roll)
        total_steps += c["n_tasks"] * c["horizon"]
        last_ret = float(roll["raw_rew"].sum(dim=0).mean())
        hist.append(losses)
        if verbose and log_every and (it + 1) % log_every == 0:
            print(f"[meta] iter {it+1}/{c['iterations']} ret={last_ret:.2f} "
                  f"pg={losses['policy_loss']:.3f} vf={losses['value_loss']:.3f} "
                  f"kl={losses['kl']:.3f} dec={losses['decoder']:.3f}")
    wall = _time.perf_counter() - t0
    return {"env_steps": total_steps, "wall_seconds": wall,
            "final_return": last_ret, "history": hist}
