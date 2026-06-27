
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate a trained Unitree G1 walking policy and record video.

Dependencies:
    pip install mujoco torch numpy imageio imageio-ffmpeg

Usage:
    # Record an MP4 video
    python humanoid/scripts/eval_record.py \
        --checkpoint outputs/ppo_scratch_train/models/g1_scratch_ppo_final.pt \
        --episodes 3 \
        --video g1_walking.mp4

    # Record a GIF (smaller file, good for portfolio)
    python humanoid/scripts/eval_record.py \
        --checkpoint outputs/ppo_scratch_train/models/g1_scratch_ppo_final.pt \
        --episodes 1 \
        --format gif \
        --video g1_walking.gif
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import mujoco

# Import classes from the training script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from humanoid.scripts.new_ppo_scratch_train import (
    EnvConfig,
    PPOConfig,
    UnitreeG1WalkingEnv,
    ActorCritic,
    RunningMeanStd,
    load_checkpoint,
    DEFAULT_XML,
    DEFAULT_OUTPUT,
)


def record_video(
    checkpoint_path: Path,
    output_path: Path,
    num_episodes: int = 3,
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
    no_render_overlay: bool = False,
) -> None:
    """Run evaluation and record video using MuJoCo offscreen rendering."""

    # Build the environment (no GUI render, but we will manually render offscreen)
    env_cfg = EnvConfig(
        xml_path=checkpoint_path.parent.parent.parent / "resources" / "robots" / "unitree_g1" / "scene.xml"
        if not DEFAULT_XML.exists()
        else DEFAULT_XML,
        command_x=0.25,
        randomize_commands=False,
    )
    env = UnitreeG1WalkingEnv(env_cfg, seed=42, render=False)

    # Build the model and load weights
    obs_dim = env.obs_dim
    action_dim = env.action_dim
    ppo_cfg = PPOConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ActorCritic(obs_dim, action_dim, ppo_cfg).to(device)
    obs_rms = RunningMeanStd((obs_dim,))
    load_checkpoint(checkpoint_path, model, obs_rms=obs_rms, device=device)
    model.eval()
    print(f"[eval_record] Loaded checkpoint: {checkpoint_path}")
    print(f"[eval_record] device={device}, obs_dim={obs_dim}, action_dim={action_dim}")

    # Set up MuJoCo offscreen renderer
    renderer = mujoco.Renderer(env.model, width=width, height=height)
    renderer.enable_depth_rendering()

    # Set camera for a nice viewing angle
    renderer.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
    renderer.camera.fixedcamid = 0  # first camera in the XML, or track the robot
    renderer.camera.distance = 2.5
    renderer.camera.azimuth = 45
    renderer.camera.elevation = -20
    renderer.camera.lookat = np.array([0.0, 0.0, 0.5])

    try:
        import imageio
    except ImportError:
        raise SystemExit(
            "Missing imageio. Install it:\n  pip install imageio imageio-ffmpeg"
        )

    ext = output_path.suffix.lower()
    if ext not in (".mp4", ".gif"):
        output_path = output_path.with_suffix(".mp4")

    writer = imageio.get_writer(
        str(output_path),
        fps=fps,
        codec="libx264" if ext == ".mp4" else None,
        quality=8,
    )

    total_frames = 0
    start_time = time.time()

    for episode in range(1, num_episodes + 1):
        obs = env.reset()
        episode_reward = 0.0
        step = 0
        info = {}

        while True:
            # Policy inference
            norm_obs = obs_rms.normalize(obs)
            obs_tensor = torch.as_tensor(norm_obs[None, :], dtype=torch.float32, device=device)
            with torch.no_grad():
                action = model.act_deterministic(obs_tensor).squeeze(0).cpu().numpy()

            # Step the physics
            next_obs, reward, done, info = env.step(action)
            episode_reward += reward
            step += 1

            # Render a frame (offscreen)
            renderer.update_scene(env.data, camera="track")
            frame = renderer.render()
            writer.append_data(frame)
            total_frames += 1

            if done:
                print(
                    f"  episode {episode}: reward={episode_reward:7.2f}  "
                    f"steps={step:4d}  x_vel={info.get('x_velocity', 0.0):.2f} m/s"
                )
                break

            obs = next_obs

    writer.close()
    elapsed = time.time() - start_time
    print(f"\n[video saved] {output_path}")
    print(f"  {total_frames} frames, {total_frames // fps}s @ {fps} fps, {elapsed:.1f}s render time")


def main() -> None:
    parser = argparse.ArgumentParser(description="Record G1 walking video from a trained checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--video", type=Path, default=Path("g1_walking.mp4"), help="Output video path")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to record")
    parser.add_argument("--fps", type=int, default=30, help="Video frame rate")
    parser.add_argument("--width", type=int, default=1280, help="Frame width")
    parser.add_argument("--height", type=int, default=720, help="Frame height")
    parser.add_argument("--format", choices=["mp4", "gif"], default="mp4", help="Output format")
    args = parser.parse_args()

    if args.format == "gif":
        args.video = args.video.with_suffix(".gif")


    record_video(
        checkpoint_path=args.checkpoint.resolve(),
        output_path=args.video.resolve(),
        num_episodes=args.episodes,
        fps=args.fps,
        width=args.width,
        height=args.height,
    )


if __name__ == "__main__":
    main()

