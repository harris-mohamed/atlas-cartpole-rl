"""
CartPole Episode Playback
--------------------------
Load a trained model and record episodes as an animated GIF.

Usage:
    python playback.py --run runs/20240101_120000
    python playback.py --run runs/20240101_120000 --episodes 5
    python playback.py --run runs/20240101_120000 --model model_interrupted

Output:
    runs/<timestamp>/playback.gif
"""

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from stable_baselines3 import DQN


def parse_args():
    parser = argparse.ArgumentParser(description="Record CartPole playback GIF")
    parser.add_argument("--run", required=True,
                        help="Path to run directory (e.g. runs/20240101_120000)")
    parser.add_argument("--model", default="model",
                        help="Model filename without .zip (default: model)")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Number of episodes to record (overrides config)")
    parser.add_argument("--fps", type=int, default=None,
                        help="GIF frame rate (overrides config)")
    parser.add_argument("--deterministic", action="store_true", default=True,
                        help="Use deterministic actions (default: True)")
    return parser.parse_args()


def render_episode(model, env) -> list:
    """Run one episode and collect rendered frames."""
    frames = []
    obs, _ = env.reset()
    done = False

    while not done:
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated

    # Capture final frame
    frame = env.render()
    if frame is not None:
        frames.append(frame)

    return frames


def add_episode_overlay(frames: list, episode_num: int, total_reward: float) -> list:
    """Burn episode number and reward into each frame."""
    annotated = []
    for i, frame in enumerate(frames):
        fig, ax = plt.subplots(figsize=(frame.shape[1] / 100, frame.shape[0] / 100),
                               dpi=100)
        ax.imshow(frame)
        ax.axis("off")
        ax.text(
            0.01, 0.97,
            f"Episode {episode_num}  |  Step {i + 1}  |  Reward so far: {min(i + 1, total_reward):.0f}",
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            color="white",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5),
        )
        fig.tight_layout(pad=0)
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        annotated.append(buf.reshape(h, w, 3))
        plt.close(fig)

    return annotated


def main():
    args = parse_args()
    run_dir = Path(args.run)

    if not run_dir.exists():
        print(f"[playback] Error: run directory not found: {run_dir}")
        sys.exit(1)

    model_path = run_dir / f"{args.model}.zip"
    if not model_path.exists():
        print(f"[playback] Error: model not found: {model_path}")
        sys.exit(1)

    # Load config from run (fallback to defaults)
    config_path = run_dir / "config.yaml"
    cfg = {}
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}

    n_episodes = args.episodes or cfg.get("playback", {}).get("n_episodes", 3)
    fps = args.fps or cfg.get("playback", {}).get("fps", 30)

    print(f"\n[playback] Loading model from {model_path}")
    model = DQN.load(model_path)

    env = gym.make("CartPole-v1", render_mode="rgb_array")

    all_frames = []
    episode_rewards = []

    for ep in range(1, n_episodes + 1):
        print(f"[playback] Recording episode {ep}/{n_episodes}...", end=" ", flush=True)
        frames = render_episode(model, env)
        reward = len(frames)  # CartPole reward = steps survived
        episode_rewards.append(reward)
        annotated = add_episode_overlay(frames, ep, float(reward))
        all_frames.extend(annotated)

        # Add a short pause (blank-ish frame) between episodes
        if ep < n_episodes:
            pause = np.zeros_like(annotated[-1])
            all_frames.extend([pause] * int(fps * 0.5))  # 0.5s gap

        print(f"reward={reward}")

    env.close()

    # Save GIF
    gif_path = run_dir / "playback.gif"
    print(f"\n[playback] Writing {len(all_frames)} frames to {gif_path}...")
    imageio.mimsave(str(gif_path), all_frames, fps=fps, loop=0)

    print(f"[playback] Done!")
    print(f"           Episodes: {n_episodes}")
    print(f"           Rewards:  {episode_rewards}")
    print(f"           Avg:      {np.mean(episode_rewards):.1f}")
    print(f"           GIF:      {gif_path}")


if __name__ == "__main__":
    main()
