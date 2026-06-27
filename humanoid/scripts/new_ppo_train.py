#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train Unitree G1 walking with a mature PPO implementation.

This script intentionally does not reuse the hand-written PPO agent.  It builds a
Gymnasium-compatible MuJoCo environment around resources/robots/unitree_g1/scene.xml
and trains with stable-baselines3 PPO on CUDA when available.

Recommended install in your training environment:
    pip install "stable-baselines3[extra]>=2.3.0" gymnasium mujoco torch tensorboard

Example:
    python humanoid/scripts/new_ppo_train.py --total-timesteps 5000000 --num-envs 8
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    import gymnasium as gym
    from gymnasium import spaces
    import mujoco
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
except ImportError as exc:
    missing = exc.name or str(exc)
    raise SystemExit(
        f"\nMissing dependency: {missing}\n"
        "Install the mature PPO stack in the Python environment you use for training:\n"
        "  pip install \"stable-baselines3[extra]>=2.3.0\" gymnasium mujoco torch tensorboard\n"
    ) from exc


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_XML = ROOT_DIR / "resources" / "robots" / "unitree_g1" / "scene.xml"
DEFAULT_OUTPUT = ROOT_DIR / "outputs" / "new_ppo_train"


@dataclass
class G1WalkingConfig:
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


class UnitreeG1WalkingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, cfg: Optional[G1WalkingConfig] = None, render_mode: Optional[str] = None):
        super().__init__()
        self.cfg = cfg or G1WalkingConfig()
        self.render_mode = render_mode
        self.model = mujoco.MjModel.from_xml_path(str(self.cfg.xml_path))
        self.data = mujoco.MjData(self.model)
        self.viewer = None

        self.dt = float(self.model.opt.timestep * self.cfg.frame_skip)
        self.nu = int(self.model.nu)
        self.nq = int(self.model.nq)
        self.nv = int(self.model.nv)
        self.num_joints = self.nq - 7

        stand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        if stand_id < 0:
            raise ValueError("The MuJoCo model must define keyframe 'stand'.")
        self.default_qpos = self.model.key_qpos[stand_id].copy()
        self.default_ctrl = self.model.key_ctrl[stand_id].copy()

        self.ctrl_low = self.model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_high = self.model.actuator_ctrlrange[:, 1].copy()
        finite_ctrl = np.isfinite(self.ctrl_low) & np.isfinite(self.ctrl_high)
        if not finite_ctrl.all():
            # Position actuators with inheritrange normally fill this range.  The
            # fallback keeps the policy near the keyframe if a model variant omits it.
            self.ctrl_low = np.where(finite_ctrl, self.ctrl_low, self.default_ctrl - 0.5)
            self.ctrl_high = np.where(finite_ctrl, self.ctrl_high, self.default_ctrl + 0.5)

        self.left_foot_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
        self.right_foot_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")
        self.torso_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "imu_in_torso")
        if min(self.left_foot_site, self.right_foot_site, self.torso_site) < 0:
            raise ValueError("Expected sites: left_foot, right_foot, imu_in_torso.")

        self.left_foot_body = self.model.site_bodyid[self.left_foot_site]
        self.right_foot_body = self.model.site_bodyid[self.right_foot_site]

        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.nu,), dtype=np.float32)
        obs_dim = 3 + 3 + 3 + 2 + self.num_joints + self.num_joints + self.nu + 2 + 2 + 2
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        self.step_count = 0
        self.prev_action = np.zeros(self.nu, dtype=np.float32)
        self.prev_base_xy = np.zeros(2, dtype=np.float32)
        self.command = np.array([self.cfg.command_x, 0.0, self.cfg.command_yaw], dtype=np.float32)
        self._last_reward_terms: Dict[str, float] = {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.default_qpos
        self.data.qvel[:] = 0.0

        if self.cfg.reset_noise_scale > 0:
            joint_noise = self.np_random.normal(0.0, self.cfg.reset_noise_scale, size=self.num_joints)
            self.data.qpos[7:] += joint_noise
            self.data.qvel[6:] = self.np_random.normal(0.0, 0.02, size=self.num_joints)

        self.data.ctrl[:] = self.default_ctrl
        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self.prev_action.fill(0.0)
        self.prev_base_xy = self.data.qpos[:2].astype(np.float32).copy()

        if self.cfg.randomize_commands:
            self.command[0] = float(self.np_random.uniform(*self.cfg.command_x_range))
            self.command[2] = float(self.np_random.uniform(*self.cfg.command_yaw_range))
        else:
            self.command[:] = (self.cfg.command_x, 0.0, self.cfg.command_yaw)

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        target = self.default_ctrl + self.cfg.action_scale * action
        target[15:] = self.default_ctrl[15:] + 0.35 * self.cfg.action_scale * action[15:]
        self.data.ctrl[:] = np.clip(target, self.ctrl_low, self.ctrl_high)

        for _ in range(self.cfg.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        obs = self._get_obs()
        reward, terms = self._compute_reward(action)
        terminated = self._is_unhealthy()
        truncated = self.step_count >= self.cfg.episode_length
        if terminated:
            reward -= 15.0

        info = {
            "x_velocity": terms["base_vx"],
            "command_x": float(self.command[0]),
            "reward_terms": terms,
            "is_success": bool(terms["base_vx"] > 0.15 and not terminated),
        }
        self._last_reward_terms = terms
        self.prev_action = action.copy()
        return obs, float(reward), terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        qpos = self.data.qpos
        qvel = self.data.qvel
        rot = quat_to_rotmat(qpos[3:7])
        local_lin_vel = rot.T @ qvel[:3]
        local_ang_vel = rot.T @ qvel[3:6]
        projected_gravity = rot.T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)
        joint_pos = qpos[7:] - self.default_qpos[7:]
        joint_vel = qvel[6:]
        foot_contacts = self._foot_contacts()
        phase = self._phase()
        obs = np.concatenate(
            [
                local_lin_vel,
                local_ang_vel,
                projected_gravity,
                self.command[[0, 2]],
                joint_pos,
                0.05 * joint_vel,
                self.prev_action,
                foot_contacts,
                np.array([math.sin(phase), math.cos(phase)], dtype=np.float32),
                np.array([qpos[2] - self.cfg.target_height, self._upright()], dtype=np.float32),
            ]
        )
        return obs.astype(np.float32)

    def _compute_reward(self, action: np.ndarray) -> Tuple[float, Dict[str, float]]:
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
        upright_reward = max(0.0, upright)

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
            + 1.25 * upright_reward
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
            "yaw_track": float(yaw_track),
            "upright": float(upright),
            "height": height,
            "height_reward": float(height_reward),
            "gait_contact": float(gait_contact),
            "swing_clearance": float(swing_clearance),
            "base_vx": base_vx,
            "base_vy": float(local_lin_vel[1]),
            "action_rate": float(action_rate),
            "joint_vel": float(joint_vel),
        }
        return float(reward), terms

    def _is_unhealthy(self) -> bool:
        height = float(self.data.qpos[2])
        if height < self.cfg.healthy_height_range[0] or height > self.cfg.healthy_height_range[1]:
            return True
        return self._upright() < self.cfg.terminate_upright_threshold

    def _upright(self) -> float:
        torso_xmat = self.data.site_xmat[self.torso_site].reshape(3, 3)
        return float(np.dot(torso_xmat[:, 2], np.array([0.0, 0.0, 1.0])))

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

    def render(self):
        if self.render_mode == "rgb_array":
            renderer = mujoco.Renderer(self.model, height=480, width=640)
            renderer.update_scene(self.data)
            image = renderer.render()
            renderer.close()
            return image
        if self.render_mode == "human":
            if self.viewer is None:
                import mujoco.viewer

                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
        return None

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


class RewardTermCallback(BaseCallback):
    def __init__(self, log_freq: int = 2048):
        super().__init__()
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq != 0:
            return True
        infos = self.locals.get("infos", [])
        accum: Dict[str, list[float]] = {}
        for info in infos:
            for key, value in info.get("reward_terms", {}).items():
                accum.setdefault(key, []).append(float(value))
        for key, values in accum.items():
            self.logger.record(f"reward_terms/{key}", float(np.mean(values)))
        return True


def make_env(cfg: G1WalkingConfig, rank: int, monitor_dir: Path):
    def _init():
        env = UnitreeG1WalkingEnv(cfg)
        return Monitor(env, filename=str(monitor_dir / f"env_{rank}.csv"))

    return _init


def build_vec_env(cfg: G1WalkingConfig, num_envs: int, monitor_dir: Path):
    env_fns = [make_env(cfg, i, monitor_dir) for i in range(num_envs)]
    if num_envs == 1:
        return DummyVecEnv(env_fns)
    return SubprocVecEnv(env_fns, start_method="spawn")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--total-timesteps", type=int, default=5_000_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--command-x", type=float, default=0.25)
    parser.add_argument("--no-random-commands", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    cfg = G1WalkingConfig(
        xml_path=args.xml,
        command_x=args.command_x,
        randomize_commands=not args.no_random_commands,
    )
    run_dir = args.output_dir
    model_dir = run_dir / "models"
    log_dir = run_dir / "tb"
    monitor_dir = run_dir / "monitor"
    for directory in (model_dir, log_dir, monitor_dir):
        directory.mkdir(parents=True, exist_ok=True)

    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[new_ppo_train] device={device}, cuda_available={torch.cuda.is_available()}")
    print(f"[new_ppo_train] xml={cfg.xml_path}")

    vec_env = build_vec_env(cfg, args.num_envs, monitor_dir)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)

    eval_env = DummyVecEnv([make_env(cfg, 9999, monitor_dir)])
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False, clip_obs=10.0)

    if args.resume:
        print(f"[new_ppo_train] resume={args.resume}")
        model = PPO.load(str(args.resume), env=vec_env, device=device)
        vec_norm_path = args.resume.with_suffix(".vecnormalize.pkl")
        if vec_norm_path.exists():
            vec_env = VecNormalize.load(str(vec_norm_path), vec_env.venv)
            vec_env.training = True
            model.set_env(vec_env)
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            device=device,
            seed=args.seed,
            verbose=1,
            tensorboard_log=str(log_dir),
            n_steps=2048,
            batch_size=512,
            n_epochs=5,
            gamma=0.99,
            gae_lambda=0.95,
            learning_rate=3.0e-4,
            clip_range=0.2,
            ent_coef=0.005,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                activation_fn=torch.nn.ELU,
                net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
                ortho_init=True,
            ),
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(10_000 // max(1, args.num_envs), 1),
        save_path=str(model_dir),
        name_prefix="g1_walk_ppo",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir / "best"),
        log_path=str(run_dir / "eval"),
        eval_freq=max(25_000 // max(1, args.num_envs), 1),
        deterministic=True,
        render=False,
        n_eval_episodes=5,
    )
    reward_cb = RewardTermCallback(log_freq=max(2048 // max(1, args.num_envs), 1))

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_cb, eval_cb, reward_cb],
        progress_bar=True,
        tb_log_name="ppo_g1_walk",
    )

    final_model = model_dir / "g1_walk_ppo_final.zip"
    final_norm = model_dir / "g1_walk_ppo_final.vecnormalize.pkl"
    model.save(str(final_model))
    vec_env.save(str(final_norm))
    print(f"[new_ppo_train] saved model: {final_model}")
    print(f"[new_ppo_train] saved vecnormalize: {final_norm}")
    vec_env.close()
    eval_env.close()


def evaluate(args: argparse.Namespace) -> None:
    if not args.resume:
        raise SystemExit("--eval-only needs --resume path/to/model.zip")
    cfg = G1WalkingConfig(xml_path=args.xml, command_x=args.command_x, randomize_commands=False)
    env = UnitreeG1WalkingEnv(cfg, render_mode="human" if args.render else None)
    vec_env = DummyVecEnv([lambda: Monitor(env)])               # 环境包装成 Stable-Baselines3 能用的格式
    norm_path = args.resume.with_suffix(".vecnormalize.pkl")
    if norm_path.exists():
        vec_env = VecNormalize.load(str(norm_path), vec_env)  # 模型旁边一般有个 vecnormalize.pkl 归一化文件
        vec_env.training = False
        vec_env.norm_reward = False
    model = PPO.load(str(args.resume), env=vec_env, device="cuda" if torch.cuda.is_available() else "cpu")

    obs = vec_env.reset()
    episode_reward = 0.0
    for _ in range(cfg.episode_length * 3):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = vec_env.step(action)
        episode_reward += float(rewards[0])
        if args.render:
            env.render()
        if dones[0]:
            print(f"episode_reward={episode_reward:.3f}, info={infos[0]}")
            episode_reward = 0.0
    vec_env.close()


def main() -> None:
    args = parse_args()
    if args.eval_only:
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
