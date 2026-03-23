"""
CartPole DQN Trainer
--------------------
Usage:
    python train.py                         # use config.yaml defaults
    python train.py --steps 50000           # override total timesteps
    python train.py --lr 0.001 --seed 0     # override multiple params
    python train.py --config my_config.yaml # use a different config file

Outputs (saved to runs/<timestamp>/):
    model.zip       - trained model checkpoint
    rewards.png     - reward + rolling average plot
    epsilon.png     - epsilon decay plot
    config.yaml     - copy of the config used for this run
"""

import argparse
import os
import shutil
import signal
import sys
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import numpy as np
import yaml
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor


# ---------------------------------------------------------------------------
# Callback: collect per-episode rewards and epsilon values during training
# ---------------------------------------------------------------------------

class TrainingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_timesteps = []
        self.epsilon_values = []
        self.epsilon_timesteps = []
        self._current_episode_reward = 0.0

    def _on_step(self) -> bool:
        # Accumulate reward
        reward = self.locals["rewards"][0]
        self._current_episode_reward += reward

        # Episode ended
        done = self.locals["dones"][0]
        if done:
            self.episode_rewards.append(self._current_episode_reward)
            self.episode_timesteps.append(self.num_timesteps)
            self._current_episode_reward = 0.0

        # Track epsilon every 500 steps
        if self.num_timesteps % 500 == 0:
            eps = self.model.exploration_rate
            self.epsilon_values.append(eps)
            self.epsilon_timesteps.append(self.num_timesteps)

        return True  # returning False would abort training


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_rewards(timesteps, rewards, run_dir: Path, window: int = 20):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(timesteps, rewards, alpha=0.4, color="steelblue", label="Episode reward")

    if len(rewards) >= window:
        rolling = np.convolve(rewards, np.ones(window) / window, mode="valid")
        rolling_ts = timesteps[window - 1:]
        ax.plot(rolling_ts, rolling, color="steelblue", linewidth=2,
                label=f"{window}-ep rolling avg")

    ax.axhline(y=500, color="green", linestyle="--", alpha=0.6, label="Solved (500)")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Training Reward Curve — CartPole-v1")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = run_dir / "rewards.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_epsilon(timesteps, epsilons, run_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(timesteps, epsilons, color="darkorange", linewidth=1.5)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Epsilon")
    ax.set_title("Exploration Rate (Epsilon) Decay")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = run_dir / "epsilon.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Config loading + CLI override
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Apply any CLI overrides on top of the loaded config."""
    if args.steps is not None:
        cfg["training"]["total_timesteps"] = args.steps
    if args.lr is not None:
        cfg["model"]["learning_rate"] = args.lr
    if args.batch_size is not None:
        cfg["model"]["batch_size"] = args.batch_size
    if args.buffer_size is not None:
        cfg["model"]["buffer_size"] = args.buffer_size
    if args.gamma is not None:
        cfg["model"]["gamma"] = args.gamma
    if args.seed is not None:
        cfg["training"]["seed"] = args.seed
    if args.exploration_fraction is not None:
        cfg["exploration"]["exploration_fraction"] = args.exploration_fraction
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Train DQN on CartPole-v1")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to YAML config file (default: config.yaml)")
    # Shorthand overrides
    parser.add_argument("--steps", type=int, default=None,
                        help="Total training timesteps")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate")
    parser.add_argument("--batch-size", type=int, dest="batch_size", default=None,
                        help="Minibatch size")
    parser.add_argument("--buffer-size", type=int, dest="buffer_size", default=None,
                        help="Replay buffer size")
    parser.add_argument("--gamma", type=float, default=None,
                        help="Discount factor")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    parser.add_argument("--exploration-fraction", type=float,
                        dest="exploration_fraction", default=None,
                        help="Fraction of training for epsilon decay")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args)

    # Create timestamped run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save the exact config used so this run is reproducible
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    print(f"\n[train] Run directory: {run_dir}")
    print(f"[train] Total timesteps: {cfg['training']['total_timesteps']:,}")
    print(f"[train] Learning rate:   {cfg['model']['learning_rate']}")
    print(f"[train] Seed:            {cfg['training']['seed']}\n")

    # Environment
    env = Monitor(gym.make("CartPole-v1"))

    # Build model from config
    m = cfg["model"]
    e = cfg["exploration"]
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=m["learning_rate"],
        batch_size=m["batch_size"],
        buffer_size=m["buffer_size"],
        learning_starts=m["learning_starts"],
        train_freq=m["train_freq"],
        target_update_interval=m["target_update_interval"],
        gamma=m["gamma"],
        tau=m["tau"],
        exploration_fraction=e["exploration_fraction"],
        exploration_initial_eps=e["exploration_initial_eps"],
        exploration_final_eps=e["exploration_final_eps"],
        verbose=1,
        seed=cfg["training"]["seed"],
    )

    callback = TrainingCallback()

    # Graceful Ctrl+C: save checkpoint before exit
    def handle_interrupt(sig, frame):
        print("\n\n[train] Interrupted! Saving checkpoint...")
        model.save(run_dir / "model_interrupted")
        _save_plots(callback, run_dir)
        print(f"[train] Saved to {run_dir}/model_interrupted.zip")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_interrupt)

    # Train
    model.learn(
        total_timesteps=cfg["training"]["total_timesteps"],
        callback=callback,
        progress_bar=True,
    )

    # Save final model
    model_path = run_dir / "model"
    model.save(model_path)
    print(f"\n[train] Model saved to {model_path}.zip")

    # Save plots
    _save_plots(callback, run_dir)

    env.close()
    print(f"\n[train] Done. Results in: {run_dir}/")
    print(f"        Playback: python playback.py --run {run_dir}")


def _save_plots(callback: TrainingCallback, run_dir: Path):
    if callback.episode_rewards:
        ts = np.array(callback.episode_timesteps)
        rw = np.array(callback.episode_rewards)
        p = plot_rewards(ts, rw, run_dir)
        print(f"[train] Reward plot saved to {p}")

    if callback.epsilon_values:
        p = plot_epsilon(callback.epsilon_timesteps, callback.epsilon_values, run_dir)
        print(f"[train] Epsilon plot saved to {p}")


if __name__ == "__main__":
    main()
