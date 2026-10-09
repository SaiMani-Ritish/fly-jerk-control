"""PPO for MLP residuals and truncated-BPTT recurrent residuals.

Recurrent actors are trained on sequential windows. Transitions are not
shuffled as independent i.i.d. observations for the recurrent actor.

The critic is a separate feedforward head on the observation (not shared
with the residual backbone beyond optional later experiments).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from fly_jerk_control.controllers.mlp import MLPResidual
from fly_jerk_control.controllers.recurrent import RecurrentResidual
from fly_jerk_control.controllers.residual_base import ResidualController
from fly_jerk_control.envs.mass_stop_env import MassStopEnv
from fly_jerk_control.training.buffers import RolloutBuffer, compute_gae
from fly_jerk_control.training.checkpoints import save_checkpoint


def _is_recurrent(ctrl: ResidualController) -> bool:
    return isinstance(ctrl, RecurrentResidual)


def collect_rollout(
    env: MassStopEnv,
    ctrl: ResidualController,
    n_steps: int,
    *,
    split: str,
) -> tuple[RolloutBuffer, np.ndarray, float]:
    buf = RolloutBuffer()
    obs, _info = env.reset(options={"split": split, "episode_index": 0})
    ctrl.reset()
    episode_returns: list[float] = []
    ret = 0.0
    reset_flag = 1.0
    last_obs = obs
    for _ in range(n_steps):
        hidden = None
        if _is_recurrent(ctrl):
            hidden = ctrl.get_recurrent_state().detach().cpu().numpy()
        raw, extras = ctrl.raw_action(obs, deterministic=False)
        next_obs, reward, terminated, truncated, info = env.step(np.array([raw], dtype=np.float32))
        done = float(terminated or truncated)
        buf.add(
            obs=obs,
            action=np.array([raw], dtype=np.float32),
            logp=extras.get("logp", 0.0),
            reward=reward,
            done=done,
            value=extras.get("value", 0.0),
            hidden=hidden if hidden is not None else np.zeros(1, dtype=np.float32),
            reset=reset_flag,
        )
        ret += float(reward)
        reset_flag = 0.0
        if terminated or truncated:
            episode_returns.append(ret)
            ret = 0.0
            obs, _info = env.reset(options={"split": split, "episode_index": env.episode_index + 1})
            ctrl.reset()
            reset_flag = 1.0
        else:
            obs = next_obs
        last_obs = obs
    last_value = 0.0
    if hasattr(ctrl, "value"):
        last_value = float(ctrl.value(last_obs).detach().cpu())
    return buf, np.asarray(episode_returns, dtype=float), last_value


def ppo_update_mlp(
    model: MLPResidual,
    buf: RolloutBuffer,
    last_value: float,
    cfg: dict[str, Any],
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    tcfg = cfg["training"]
    data = buf.as_tensors()
    adv, ret = compute_gae(
        data["rewards"],
        data["values"],
        data["dones"],
        last_value,
        gamma=float(tcfg["gamma"]),
        gae_lambda=float(tcfg["gae_lambda"]),
    )
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    n = data["obs"].shape[0]
    idx = np.arange(n)
    clip = float(tcfg["clip_range"])
    stats = []
    for _ in range(int(tcfg["n_epochs"])):
        np.random.shuffle(idx)
        mb = int(tcfg["minibatch_size"])
        for start in range(0, n, mb):
            sl = idx[start : start + mb]
            obs = data["obs"][sl]
            act = data["actions"][sl]
            old_logp = data["logp"][sl]
            logp, ent, value = model.evaluate_actions(obs, act)
            ratio = torch.exp(logp - old_logp)
            a = adv[sl]
            surr1 = ratio * a
            surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * a
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * (value - ret[sl]).pow(2).mean()
            loss = policy_loss + float(tcfg["vf_coef"]) * value_loss - float(tcfg["ent_coef"]) * ent.mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(tcfg["max_grad_norm"]))
            optimizer.step()
            stats.append(float(loss.detach()))
    return {"loss": float(np.mean(stats)) if stats else float("nan")}


def ppo_update_recurrent(
    model: RecurrentResidual,
    buf: RolloutBuffer,
    last_value: float,
    cfg: dict[str, Any],
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    """Sequence PPO with truncated BPTT windows. No i.i.d. shuffle of steps."""
    tcfg = cfg["training"]
    if not tcfg.get("recurrent_supported", True):
        raise RuntimeError("Recurrent PPO marked unsupported in config.")
    data = buf.as_tensors()
    adv, ret = compute_gae(
        data["rewards"],
        data["values"],
        data["dones"],
        last_value,
        gamma=float(tcfg["gamma"]),
        gae_lambda=float(tcfg["gae_lambda"]),
    )
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    t_len = int(tcfg["bptt_len"])
    n = data["obs"].shape[0]
    clip = float(tcfg["clip_range"])
    stats = []
    # Windows along the collected trajectory; batch dim = 1.
    starts = list(range(0, max(n - t_len, 0) + 1, t_len))
    if not starts:
        starts = [0]
        t_len = n
    for _ in range(int(tcfg["n_epochs"])):
        for s in starts:
            e = min(s + t_len, n)
            if e - s < 2:
                continue
            obs = data["obs"][s:e].unsqueeze(1)
            act = data["actions"][s:e].unsqueeze(1)
            resets = data["resets"][s:e].unsqueeze(1)
            h0 = data["hiddens"][s].unsqueeze(0)
            old_logp = data["logp"][s:e]
            logp, ent, value = model.sequence_forward(obs, act, h0, resets)
            logp = logp.squeeze(1)
            ent = ent.squeeze(1)
            value = value.squeeze(1)
            ratio = torch.exp(logp - old_logp)
            a = adv[s:e]
            surr1 = ratio * a
            surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * a
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * (value - ret[s:e]).pow(2).mean()
            loss = policy_loss + float(tcfg["vf_coef"]) * value_loss - float(tcfg["ent_coef"]) * ent.mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(tcfg["max_grad_norm"]))
            optimizer.step()
            with torch.no_grad():
                model.w_graph.mul_(model.mask)
            stats.append(float(loss.detach()))
    return {"loss": float(np.mean(stats)) if stats else float("nan")}


def train_ppo(
    env: MassStopEnv,
    ctrl: ResidualController,
    cfg: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    if not isinstance(ctrl, nn.Module):
        raise TypeError("PPO requires a torch residual module")
    tcfg = cfg["training"]
    opt = torch.optim.Adam(ctrl.parameters(), lr=float(tcfg["lr"]))
    total = int(tcfg["total_env_steps"])
    n_steps = int(tcfg["n_steps"])
    curves: list[dict[str, float]] = []
    steps = 0
    update = 0
    while steps < total:
        buf, ep_rets, last_value = collect_rollout(env, ctrl, n_steps, split=str(tcfg.get("split", "train")))
        steps += n_steps
        update += 1
        if isinstance(ctrl, RecurrentResidual):
            stats = ppo_update_recurrent(ctrl, buf, last_value, cfg, opt)
        elif isinstance(ctrl, MLPResidual):
            stats = ppo_update_mlp(ctrl, buf, last_value, cfg, opt)
        else:
            raise TypeError(f"Unsupported residual type {type(ctrl)}")
        rec = {
            "update": update,
            "steps": steps,
            "loss": stats["loss"],
            "ep_return_mean": float(np.mean(ep_rets)) if ep_rets.size else float("nan"),
        }
        curves.append(rec)
        ckpt = run_dir / "checkpoints" / f"update_{update:04d}.pt"
        save_checkpoint(ckpt, model=ctrl, cfg=cfg, extra={"update": update, "steps": steps})
    save_checkpoint(run_dir / "checkpoints" / "final.pt", model=ctrl, cfg=cfg, extra={"update": update, "steps": steps})
    pd.DataFrame(curves).to_csv(run_dir / "training_curves.csv", index=False)
    return {"updates": update, "steps": steps, "curves": curves}
