#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train Unitree G1 walking with a from-scratch PPO implementation.

This file is intentionally educational:
  - no stable-baselines3
  - no Gymnasium wrapper requirement
  - PPO rollout, GAE, clipped objective, value loss, entropy, mini-batch SGD
    are all implemented in this file

External dependencies are only the basic building blocks:
    pip install mujoco torch numpy

Example training:
    python humanoid/scripts/new_ppo_scratch_train.py --total-timesteps 5000000 --num-envs 4

Example evaluation:
    python humanoid/scripts/new_ppo_scratch_train.py --eval-only --render --resume outputs/ppo_scratch_train/models/g1_scratch_ppo_final.pt
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    import mujoco
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Normal
except ImportError as exc:
    missing = exc.name or str(exc)
    raise SystemExit(
        f"\nMissing dependency: {missing}\n"
        "Install the minimal training stack:\n"
        "  pip install mujoco torch numpy\n"
    ) from exc


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_XML = ROOT_DIR / "resources" / "robots" / "unitree_g1" / "scene.xml"
DEFAULT_OUTPUT = ROOT_DIR / "outputs" / "ppo_scratch_train"


@dataclass
class EnvConfig:
    xml_path: Path = DEFAULT_XML
    frame_skip: int = 10
    episode_length: int = 1000
    action_scale: float = 0.18
    target_height: float = 0.79
    command_x: float = 0.25
    command_yaw: float = 0.0
    command_x_range: Tuple[float, float] = (0.10, 0.45)
    command_yaw_range: Tuple[float, float] = (-0.25, 0.25)
    randomize_commands: bool = True
    healthy_height_range: Tuple[float, float] = (0.58, 1.05)
    terminate_upright_threshold: float = 0.55
    reset_noise_scale: float = 0.01
    gait_period: float = 0.80


@dataclass
class PPOConfig:
    total_timesteps: int = 5_000_000
    num_envs: int = 4
    rollout_steps: int = 2048
    minibatch_size: int = 512
    update_epochs: int = 5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3.0e-4
    clip_coef: float = 0.20
    value_clip_coef: float = 0.20
    entropy_coef: float = 0.005
    value_coef: float = 0.50
    max_grad_norm: float = 0.50
    target_kl: float = 0.03
    init_log_std: float = -0.7
    hidden_size: int = 256
    save_interval: int = 10
    seed: int = 42


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def exp_reward(error: np.ndarray | float, sigma: float) -> float:
    return float(np.exp(-np.square(error).sum() / (sigma * sigma)))


class RunningMeanStd:
    """Online observation normalizer using the parallel variance update."""

    def __init__(self, shape: Tuple[int, ...], epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        self.mean = new_mean
        self.var = m_2 / total_count
        self.count = total_count

    def normalize(self, x: np.ndarray, clip: float = 10.0) -> np.ndarray:
        y = (x - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(y, -clip, clip).astype(np.float32)

    def state_dict(self) -> Dict[str, np.ndarray | float]:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state: Dict[str, np.ndarray | float]) -> None:
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.var = np.asarray(state["var"], dtype=np.float64)
        self.count = float(state["count"])


class UnitreeG1WalkingEnv:
    """Minimal MuJoCo environment. No Gymnasium dependency is needed."""

    def __init__(self, cfg: EnvConfig, seed: int = 0, render: bool = False):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.render_enabled = render
        self.viewer = None

        self.model = mujoco.MjModel.from_xml_path(str(cfg.xml_path))
        self.data = mujoco.MjData(self.model)
        self.dt = float(self.model.opt.timestep * cfg.frame_skip)

        self.nu = int(self.model.nu)
        self.nq = int(self.model.nq)
        self.nv = int(self.model.nv)
        self.num_joints = self.nq - 7

        stand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        if stand_id < 0:
            raise ValueError("The model must contain a keyframe named 'stand'.")
        self.default_qpos = self.model.key_qpos[stand_id].copy()
        self.default_ctrl = self.model.key_ctrl[stand_id].copy()

        self.ctrl_low = self.model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_high = self.model.actuator_ctrlrange[:, 1].copy()
        finite_ctrl = np.isfinite(self.ctrl_low) & np.isfinite(self.ctrl_high)
        self.ctrl_low = np.where(finite_ctrl, self.ctrl_low, self.default_ctrl - 0.5)
        self.ctrl_high = np.where(finite_ctrl, self.ctrl_high, self.default_ctrl + 0.5)

        self.left_foot_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
        self.right_foot_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")
        self.torso_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "imu_in_torso")
        if min(self.left_foot_site, self.right_foot_site, self.torso_site) < 0:
            raise ValueError("Expected model sites: left_foot, right_foot, imu_in_torso.")

        self.left_foot_body = self.model.site_bodyid[self.left_foot_site]
        self.right_foot_body = self.model.site_bodyid[self.right_foot_site]

        self.action_dim = self.nu
        self.obs_dim = 3 + 3 + 3 + 2 + self.num_joints + self.num_joints + self.nu + 2 + 2 + 2
        self.step_count = 0
        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self.command = np.array([cfg.command_x, 0.0, cfg.command_yaw], dtype=np.float32)

    def reset(self) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.default_qpos
        self.data.qvel[:] = 0.0

        if self.cfg.reset_noise_scale > 0:
            self.data.qpos[7:] += self.rng.normal(0.0, self.cfg.reset_noise_scale, self.num_joints)
            self.data.qvel[6:] = self.rng.normal(0.0, 0.02, self.num_joints)

        self.data.ctrl[:] = self.default_ctrl
        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self.prev_action.fill(0.0)
        if self.cfg.randomize_commands:
            self.command[0] = float(self.rng.uniform(*self.cfg.command_x_range))
            self.command[2] = float(self.rng.uniform(*self.cfg.command_yaw_range))
        else:
            self.command[:] = (self.cfg.command_x, 0.0, self.cfg.command_yaw)
        return self._get_obs()

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        target = self.default_ctrl + self.cfg.action_scale * action
        target[15:] = self.default_ctrl[15:] + 0.35 * self.cfg.action_scale * action[15:]
        self.data.ctrl[:] = np.clip(target, self.ctrl_low, self.ctrl_high)

        for _ in range(self.cfg.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        reward, terms = self._compute_reward(action)
        terminated = self._is_unhealthy()
        truncated = self.step_count >= self.cfg.episode_length
        done = terminated or truncated
        if terminated:
            reward -= 15.0

        obs = self._get_obs()
        info = {
            "terminated": terminated,
            "truncated": truncated,
            "x_velocity": terms["base_vx"],
            "command_x": float(self.command[0]),
            "reward_terms": terms,
        }
        self.prev_action = action.copy()
        return obs, float(reward), bool(done), info

    def _get_obs(self) -> np.ndarray:
        qpos = self.data.qpos
        qvel = self.data.qvel
        rot = quat_to_rotmat(qpos[3:7])
        local_lin_vel = rot.T @ qvel[:3]
        local_ang_vel = rot.T @ qvel[3:6]
        projected_gravity = rot.T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)
        joint_pos = qpos[7:] - self.default_qpos[7:]
        joint_vel = qvel[6:]
        contacts = self._foot_contacts()
        phase = self._phase()
        return np.concatenate(
            [
                local_lin_vel,
                local_ang_vel,
                projected_gravity,
                self.command[[0, 2]],
                joint_pos,
                0.05 * joint_vel,
                self.prev_action,
                contacts,
                np.array([math.sin(phase), math.cos(phase)], dtype=np.float32),
                np.array([qpos[2] - self.cfg.target_height, self._upright()], dtype=np.float32),
            ]
        ).astype(np.float32)

    def _compute_reward(self, action: np.ndarray):
        qpos = self.data.qpos
        qvel = self.data.qvel
        rot = quat_to_rotmat(qpos[3:7])
        local_lin_vel = rot.T @ qvel[:3]
        local_ang_vel = rot.T @ qvel[3:6]
        base_vx = float(local_lin_vel[0])

        height = float(qpos[2])
        upright = self._upright()
        vel_track = exp_reward(local_lin_vel[:2] - np.array([self.command[0], 0.0]), 0.35)
        yaw_track = exp_reward(float(local_ang_vel[2] - self.command[2]), 0.45)
        height_reward = exp_reward(height - self.cfg.target_height, 0.12)

        phase = self._phase()
        left_desired = 1.0 if math.sin(phase) > 0 else 0.0
        right_desired = 1.0 - left_desired
        contacts = self._foot_contacts()
        gait_contact = 1.0 - 0.5 * (abs(contacts[0] - left_desired) + abs(contacts[1] - right_desired))
        gait_contact = max(0.0, gait_contact)

        left_vel = np.linalg.norm(self.data.cvel[self.left_foot_body][3:5])
        right_vel = np.linalg.norm(self.data.cvel[self.right_foot_body][3:5])
        stance_slide = contacts[0] * left_vel + contacts[1] * right_vel

        left_z = float(self.data.site_xpos[self.left_foot_site][2])
        right_z = float(self.data.site_xpos[self.right_foot_site][2])
        swing_clearance = 0.0
        if contacts[0] < 0.5:
            swing_clearance += exp_reward(left_z - 0.08, 0.08)
        if contacts[1] < 0.5:
            swing_clearance += exp_reward(right_z - 0.08, 0.08)

        joint_error = qpos[7:] - self.default_qpos[7:]
        leg_error = np.mean(np.square(joint_error[:12]))
        waist_arm_error = np.mean(np.square(joint_error[12:]))
        action_rate = np.mean(np.square(action - self.prev_action))
        action_mag = np.mean(np.square(action))
        joint_vel = np.mean(np.square(qvel[6:]))
        ctrl_change = np.mean(np.square(self.data.ctrl - self.default_ctrl))

        reward = (
            2.50 * vel_track
            + 0.70 * yaw_track
            + 1.25 * max(0.0, upright)
            + 0.80 * height_reward
            + 0.25 * gait_contact
            + 0.15 * swing_clearance
            + 0.05
            - 0.25 * action_rate
            - 0.05 * action_mag
            - 0.40 * leg_error
            - 0.70 * waist_arm_error
            - 0.01 * joint_vel
            - 0.02 * ctrl_change
            - 0.20 * float(abs(local_lin_vel[1]))
            - 0.10 * float(abs(local_ang_vel[0]) + abs(local_ang_vel[1]))
            - 0.08 * float(stance_slide)
        )
        terms = {
            "reward": float(reward),
            "vel_track": float(vel_track),
            "upright": float(upright),
            "height": height,
            "base_vx": base_vx,
            "action_rate": float(action_rate),
            "gait_contact": float(gait_contact),
        }
        return float(reward), terms

    def _is_unhealthy(self) -> bool:
        height = float(self.data.qpos[2])
        if height < self.cfg.healthy_height_range[0] or height > self.cfg.healthy_height_range[1]:
            return True
        return self._upright() < self.cfg.terminate_upright_threshold

    def _upright(self) -> float:
        xmat = self.data.site_xmat[self.torso_site].reshape(3, 3)
        return float(np.dot(xmat[:, 2], np.array([0.0, 0.0, 1.0])))

    def _foot_contacts(self) -> np.ndarray:
        contacts = np.zeros(2, dtype=np.float32)
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            body1 = self.model.geom_bodyid[con.geom1]
            body2 = self.model.geom_bodyid[con.geom2]
            if body1 == self.left_foot_body or body2 == self.left_foot_body:
                contacts[0] = 1.0
            if body1 == self.right_foot_body or body2 == self.right_foot_body:
                contacts[1] = 1.0
        return contacts

    def _phase(self) -> float:
        return 2.0 * math.pi * (self.step_count * self.dt / self.cfg.gait_period)

    def render(self) -> None:
        if not self.render_enabled:
            return
        if self.viewer is None:
            import mujoco.viewer

            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


class ActorCritic(nn.Module):
    """Gaussian actor and scalar critic sharing only the input."""

    def __init__(self, obs_dim: int, action_dim: int, cfg: PPOConfig):
        super().__init__()
        h = cfg.hidden_size
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, h),
            nn.ELU(),
            nn.Linear(h, h),
            nn.ELU(),
            nn.Linear(h, 128),
            nn.ELU(),
            nn.Linear(128, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, h),
            nn.ELU(),
            nn.Linear(h, h),
            nn.ELU(),
            nn.Linear(h, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), cfg.init_log_std))
        self.apply(self._orthogonal_init)

    @staticmethod
    def _orthogonal_init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
            nn.init.constant_(module.bias, 0.0)

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean = torch.tanh(self.actor(obs))
        std = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std)

    def get_action_and_value(self, obs: torch.Tensor, action: Optional[torch.Tensor] = None):
        dist = self.distribution(obs)
        if action is None:
            raw_action = dist.sample()
            action = torch.clamp(raw_action, -1.0, 1.0)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(obs).squeeze(-1)
        return action, log_prob, entropy, value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def act_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.actor(obs))


class RolloutBuffer:
    """Stores one on-policy PPO batch: T steps across N environments."""

    def __init__(self, steps: int, num_envs: int, obs_dim: int, action_dim: int, device: torch.device):
        self.steps = steps
        self.num_envs = num_envs
        self.device = device
        self.obs = torch.zeros((steps, num_envs, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((steps, num_envs, action_dim), dtype=torch.float32, device=device)
        self.log_probs = torch.zeros((steps, num_envs), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((steps, num_envs), dtype=torch.float32, device=device)
        self.dones = torch.zeros((steps, num_envs), dtype=torch.float32, device=device)
        self.values = torch.zeros((steps, num_envs), dtype=torch.float32, device=device)
        self.advantages = torch.zeros((steps, num_envs), dtype=torch.float32, device=device)
        self.returns = torch.zeros((steps, num_envs), dtype=torch.float32, device=device)

    def compute_gae(self, last_values: torch.Tensor, last_dones: torch.Tensor, cfg: PPOConfig) -> None:
        # Generalized Advantage Estimation:
        # delta_t = r_t + gamma * V(s_{t+1}) * nonterminal - V(s_t)
        # A_t = delta_t + gamma * lambda * nonterminal * A_{t+1}
        last_gae = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        for t in reversed(range(self.steps)):
            if t == self.steps - 1:
                next_values = last_values
                next_nonterminal = 1.0 - last_dones
            else:
                next_values = self.values[t + 1]
                next_nonterminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + cfg.gamma * next_values * next_nonterminal - self.values[t]
            last_gae = delta + cfg.gamma * cfg.gae_lambda * next_nonterminal * last_gae
            self.advantages[t] = last_gae
        self.returns = self.advantages + self.values

    def flattened(self):
        return (
            self.obs.reshape(-1, self.obs.shape[-1]),
            self.actions.reshape(-1, self.actions.shape[-1]),
            self.log_probs.reshape(-1),
            self.advantages.reshape(-1),
            self.returns.reshape(-1),
            self.values.reshape(-1),
        )


def make_envs(env_cfg: EnvConfig, ppo_cfg: PPOConfig, render: bool = False):
    envs = []
    for i in range(ppo_cfg.num_envs):
        envs.append(UnitreeG1WalkingEnv(env_cfg, seed=ppo_cfg.seed + i, render=render and i == 0))
    return envs


def save_checkpoint(path: Path, model: ActorCritic, optimizer: optim.Optimizer, obs_rms: RunningMeanStd, env_cfg: EnvConfig, ppo_cfg: PPOConfig, update: int, global_step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "obs_rms": obs_rms.state_dict(),
            "env_cfg": asdict(env_cfg),
            "ppo_cfg": asdict(ppo_cfg),
            "update": update,
            "global_step": global_step,
        },
        path,
    )


def load_checkpoint(path: Path, model: ActorCritic, optimizer: Optional[optim.Optimizer] = None, obs_rms: Optional[RunningMeanStd] = None, device: str | torch.device = "cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if obs_rms is not None and "obs_rms" in ckpt:
        obs_rms.load_state_dict(ckpt["obs_rms"])
    return ckpt


def ppo_update(model: ActorCritic, optimizer: optim.Optimizer, buffer: RolloutBuffer, cfg: PPOConfig):
    b_obs, b_actions, b_log_probs, b_advantages, b_returns, b_values = buffer.flattened()
    b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
    batch_size = b_obs.shape[0]
    indices = torch.randperm(batch_size, device=buffer.device)

    metrics = {}
    for epoch in range(cfg.update_epochs):
        for start in range(0, batch_size, cfg.minibatch_size):
            mb_idx = indices[start : start + cfg.minibatch_size]
            _, new_log_prob, entropy, new_value = model.get_action_and_value(b_obs[mb_idx], b_actions[mb_idx])

            # Importance sampling ratio between new and old policy.
            log_ratio = new_log_prob - b_log_probs[mb_idx]
            ratio = log_ratio.exp()

            # PPO clipped policy gradient objective.
            mb_adv = b_advantages[mb_idx]
            pg_loss_unclipped = -mb_adv * ratio
            pg_loss_clipped = -mb_adv * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
            policy_loss = torch.max(pg_loss_unclipped, pg_loss_clipped).mean()

            # Clipped value loss, same idea as policy clipping.
            value_pred_clipped = b_values[mb_idx] + torch.clamp(
                new_value - b_values[mb_idx],
                -cfg.value_clip_coef,
                cfg.value_clip_coef,
            )
            value_loss_unclipped = torch.square(new_value - b_returns[mb_idx])
            value_loss_clipped = torch.square(value_pred_clipped - b_returns[mb_idx])
            value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

            entropy_loss = entropy.mean()
            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                clip_fraction = ((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()
            metrics = {
                "loss": float(loss.item()),
                "policy_loss": float(policy_loss.item()),
                "value_loss": float(value_loss.item()),
                "entropy": float(entropy_loss.item()),
                "approx_kl": approx_kl,
                "clip_fraction": clip_fraction,
            }

        if metrics.get("approx_kl", 0.0) > cfg.target_kl:
            break
    return metrics


def train(args: argparse.Namespace) -> None:
    env_cfg = EnvConfig(xml_path=args.xml, command_x=args.command_x, randomize_commands=not args.no_random_commands)
    ppo_cfg = PPOConfig(
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        seed=args.seed,
    )
    torch.manual_seed(ppo_cfg.seed)
    np.random.seed(ppo_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    output_dir = args.output_dir
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    envs = make_envs(env_cfg, ppo_cfg)
    obs = np.stack([env.reset() for env in envs], axis=0)
    obs_dim = envs[0].obs_dim
    action_dim = envs[0].action_dim
    obs_rms = RunningMeanStd((obs_dim,))
    obs_rms.update(obs)

    model = ActorCritic(obs_dim, action_dim, ppo_cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=ppo_cfg.learning_rate, eps=1e-5)

    start_update = 1
    global_step = 0
    if args.resume:
        ckpt = load_checkpoint(args.resume, model, optimizer, obs_rms, device)
        start_update = int(ckpt.get("update", 0)) + 1
        global_step = int(ckpt.get("global_step", 0))
        print(f"[resume] loaded {args.resume}, update={start_update}, global_step={global_step}")

    updates = math.ceil((ppo_cfg.total_timesteps - global_step) / (ppo_cfg.rollout_steps * ppo_cfg.num_envs))
    episode_returns = np.zeros(ppo_cfg.num_envs, dtype=np.float32)
    episode_lengths = np.zeros(ppo_cfg.num_envs, dtype=np.int32)
    completed_returns = []
    completed_lengths = []
    start_time = time.time()

    print(f"[ppo_scratch] device={device}, obs_dim={obs_dim}, action_dim={action_dim}")
    print(f"[ppo_scratch] xml={env_cfg.xml_path}")
    print(f"[ppo_scratch] total_timesteps={ppo_cfg.total_timesteps}, num_envs={ppo_cfg.num_envs}, rollout_steps={ppo_cfg.rollout_steps}")

    for update in range(start_update, start_update + updates):
        buffer = RolloutBuffer(ppo_cfg.rollout_steps, ppo_cfg.num_envs, obs_dim, action_dim, device)
        norm_obs = obs_rms.normalize(obs)

        for step in range(ppo_cfg.rollout_steps):
            global_step += ppo_cfg.num_envs
            obs_tensor = torch.as_tensor(norm_obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                action_tensor, log_prob, _, value = model.get_action_and_value(obs_tensor)

            actions = action_tensor.cpu().numpy()
            next_obs_list, rewards, dones = [], [], []
            for i, env in enumerate(envs):
                next_obs, reward, done, _ = env.step(actions[i])
                episode_returns[i] += reward
                episode_lengths[i] += 1
                if done:
                    completed_returns.append(float(episode_returns[i]))
                    completed_lengths.append(int(episode_lengths[i]))
                    episode_returns[i] = 0.0
                    episode_lengths[i] = 0
                    next_obs = env.reset()
                next_obs_list.append(next_obs)
                rewards.append(reward)
                dones.append(done)

            next_obs_batch = np.stack(next_obs_list, axis=0)
            obs_rms.update(next_obs_batch)

            buffer.obs[step].copy_(obs_tensor)
            buffer.actions[step].copy_(action_tensor)
            buffer.log_probs[step].copy_(log_prob)
            buffer.rewards[step].copy_(torch.as_tensor(rewards, dtype=torch.float32, device=device))
            buffer.dones[step].copy_(torch.as_tensor(dones, dtype=torch.float32, device=device))
            buffer.values[step].copy_(value)

            obs = next_obs_batch
            norm_obs = obs_rms.normalize(obs)

        with torch.no_grad():
            last_obs_tensor = torch.as_tensor(norm_obs, dtype=torch.float32, device=device)
            last_values = model.get_value(last_obs_tensor)
        last_dones = torch.as_tensor(dones, dtype=torch.float32, device=device)
        buffer.compute_gae(last_values, last_dones, ppo_cfg)

        metrics = ppo_update(model, optimizer, buffer, ppo_cfg)

        if completed_returns:
            mean_return = float(np.mean(completed_returns[-20:]))
            mean_length = float(np.mean(completed_lengths[-20:]))
        else:
            mean_return = float(np.mean(episode_returns))
            mean_length = float(np.mean(episode_lengths))
        fps = int(global_step / max(time.time() - start_time, 1e-6))
        print(
            f"update={update:04d} step={global_step:09d} "
            f"return={mean_return:8.2f} len={mean_length:6.1f} fps={fps:5d} "
            f"pi={metrics['policy_loss']:+.4f} vf={metrics['value_loss']:.4f} "
            f"ent={metrics['entropy']:.3f} kl={metrics['approx_kl']:.5f} clip={metrics['clip_fraction']:.3f}"
        )

        if update % ppo_cfg.save_interval == 0:
            save_checkpoint(model_dir / f"g1_scratch_ppo_{update}.pt", model, optimizer, obs_rms, env_cfg, ppo_cfg, update, global_step)

    final_path = model_dir / "g1_scratch_ppo_final.pt"
    save_checkpoint(final_path, model, optimizer, obs_rms, env_cfg, ppo_cfg, update, global_step)
    print(f"[ppo_scratch] saved final checkpoint: {final_path}")
    for env in envs:
        env.close()


def evaluate(args: argparse.Namespace) -> None:
    if not args.resume:
        raise SystemExit("--eval-only requires --resume path/to/checkpoint.pt")

    env_cfg = EnvConfig(xml_path=args.xml, command_x=args.command_x, randomize_commands=False)
    env = UnitreeG1WalkingEnv(env_cfg, seed=args.seed, render=args.render)
    obs = env.reset()
    obs_rms = RunningMeanStd((env.obs_dim,))
    model = ActorCritic(env.obs_dim, env.action_dim, PPOConfig()).to("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    device = next(model.parameters()).device
    load_checkpoint(args.resume, model, obs_rms=obs_rms, device=device)
    model.eval()

    episode_return = 0.0
    episode_length = 0
    for _ in range(env_cfg.episode_length * args.eval_episodes):
        norm_obs = obs_rms.normalize(obs)
        obs_tensor = torch.as_tensor(norm_obs[None, :], dtype=torch.float32, device=device)
        with torch.no_grad():
            action = model.act_deterministic(obs_tensor).squeeze(0).cpu().numpy()
        obs, reward, done, info = env.step(action)
        episode_return += reward
        episode_length += 1
        env.render()
        if done:
            print(f"episode_return={episode_return:.2f}, length={episode_length}, info={info}")
            episode_return = 0.0
            episode_length = 0
            obs = env.reset()
    env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--total-timesteps", type=int, default=5_000_000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--command-x", type=float, default=0.25)
    parser.add_argument("--no-random-commands", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.eval_only:
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
