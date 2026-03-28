"""
CartPole DQN Trainer
--------------------
Usage:
    python train.py                                 # fresh run with config.yaml
    python train.py --steps 50000                   # override total timesteps
    python train.py --lr 0.001 --seed 0             # override multiple params
    python train.py --resume runs/20240101_120000   # resume a paused run

Ctrl+C at any time to pause — saves model, replay buffer, and progress.
Resume later with --resume <run_dir>.

Outputs (saved to runs/<timestamp>/):
    model.zip              - latest model checkpoint
    replay_buffer.pkl      - replay buffer (needed for clean resume)
    progress.yaml          - timestep count + episode history
    rewards.png            - reward + rolling average plot
    epsilon.png            - epsilon decay plot
    config.yaml            - copy of the config used for this run
"""

import argparse
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
    def __init__(self, prior_episode_rewards=None, prior_episode_timesteps=None,
                 prior_epsilon_values=None, prior_epsilon_timesteps=None, verbose=0):
        super().__init__(verbose)
        # Seed with history from a previous run segment so plots are continuous
        self.episode_rewards = list(prior_episode_rewards or [])
        self.episode_timesteps = list(prior_episode_timesteps or [])
        self.epsilon_values = list(prior_epsilon_values or [])
        self.epsilon_timesteps = list(prior_epsilon_timesteps or [])
        self._current_episode_reward = 0.0

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]
        self._current_episode_reward += reward

        done = self.locals["dones"][0]
        if done:
            self.episode_rewards.append(self._current_episode_reward)
            self.episode_timesteps.append(self.num_timesteps)
            self._current_episode_reward = 0.0

        if self.num_timesteps % 500 == 0:
            self.epsilon_values.append(self.model.exploration_rate)
            self.epsilon_timesteps.append(self.num_timesteps)

        return True


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_rewards(timesteps, rewards, run_dir: Path, window: int = 20):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(timesteps, rewards, alpha=0.4, color="steelblue", label="Episode reward")

    if len(rewards) >= window:
        rolling = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(timesteps[window - 1:], rolling, color="steelblue", linewidth=2,
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


def save_plots(callback: TrainingCallback, run_dir: Path):
    if callback.episode_rewards:
        p = plot_rewards(np.array(callback.episode_timesteps),
                         np.array(callback.episode_rewards), run_dir)
        print(f"[train] Reward plot → {p}")
    if callback.epsilon_values:
        p = plot_epsilon(callback.epsilon_timesteps, callback.epsilon_values, run_dir)
        print(f"[train] Epsilon plot → {p}")


# ---------------------------------------------------------------------------
# Progress state (pause/resume)
# ---------------------------------------------------------------------------

def save_progress(run_dir: Path, model, callback: TrainingCallback, total_timesteps: int):
    """Persist everything needed to resume this run later."""
    model.save(run_dir / "model")
    model.save_replay_buffer(run_dir / "replay_buffer")

    progress = {
        "timesteps_elapsed": int(model.num_timesteps),
        "total_timesteps": int(total_timesteps),
        "episode_rewards": [float(r) for r in callback.episode_rewards],
        "episode_timesteps": [int(t) for t in callback.episode_timesteps],
        "epsilon_values": [float(e) for e in callback.epsilon_values],
        "epsilon_timesteps": [int(t) for t in callback.epsilon_timesteps],
    }
    with open(run_dir / "progress.yaml", "w") as f:
        yaml.dump(progress, f, default_flow_style=False)

    print(f"[train] Checkpoint saved ({model.num_timesteps:,} / {total_timesteps:,} steps)")


def load_progress(run_dir: Path) -> dict:
    path = run_dir / "progress.yaml"
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
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
    parser = argparse.ArgumentParser(
        description="Train DQN on CartPole-v1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py                              # fresh run
  python train.py --steps 50000 --lr 0.001    # custom hyperparams
  python train.py --resume runs/20240101_120000  # resume after Ctrl+C
        """,
    )
    parser.add_argument("--config", default="config.yaml",
                        help="YAML config file (default: config.yaml)")
    parser.add_argument("--resume", metavar="RUN_DIR", default=None,
                        help="Resume a paused run from this directory")
    # Hyperparameter overrides
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, dest="batch_size", default=None)
    parser.add_argument("--buffer-size", type=int, dest="buffer_size", default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--exploration-fraction", type=float,
                        dest="exploration_fraction", default=None)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    env = Monitor(gym.make("CartPole-v1"))

    if args.resume:
        # ── RESUME MODE ────────────────────────────────────────────────────
        run_dir = Path(args.resume)
        if not run_dir.exists():
            print(f"[train] Error: run directory not found: {run_dir}")
            sys.exit(1)

        # Load config from the original run (ignore current config.yaml)
        cfg = load_config(run_dir / "config.yaml")
        # Allow overriding total_timesteps so you can extend a run
        cfg = apply_overrides(cfg, args)

        progress = load_progress(run_dir)
        if not progress:
            print(f"[train] Error: no progress.yaml found in {run_dir}. "
                  "Was this run paused with Ctrl+C?")
            sys.exit(1)

        timesteps_elapsed = progress["timesteps_elapsed"]
        total_timesteps = progress["total_timesteps"]
        remaining = total_timesteps - timesteps_elapsed

        if remaining <= 0:
            print(f"[train] Run already complete ({timesteps_elapsed:,} steps done).")
            print(f"        For playback: python playback.py --run {run_dir}")
            sys.exit(0)

        print(f"\n[train] Resuming run: {run_dir}")
        print(f"[train] Progress: {timesteps_elapsed:,} / {total_timesteps:,} steps "
              f"({100 * timesteps_elapsed / total_timesteps:.1f}% done)")
        print(f"[train] Remaining: {remaining:,} steps\n")

        model = DQN.load(run_dir / "model", env=env)

        # Reload replay buffer so the agent doesn't start cold
        replay_buffer_path = run_dir / "replay_buffer.pkl"
        if replay_buffer_path.exists():
            model.load_replay_buffer(replay_buffer_path)
            print(f"[train] Replay buffer loaded ({model.replay_buffer.size()} samples)")
        else:
            print("[train] Warning: no replay buffer found, starting with empty buffer")

        callback = TrainingCallback(
            prior_episode_rewards=progress.get("episode_rewards"),
            prior_episode_timesteps=progress.get("episode_timesteps"),
            prior_epsilon_values=progress.get("epsilon_values"),
            prior_epsilon_timesteps=progress.get("epsilon_timesteps"),
        )

    else:
        # ── FRESH RUN ──────────────────────────────────────────────────────
        cfg = load_config(args.config)
        cfg = apply_overrides(cfg, args)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path("runs") / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)

        with open(run_dir / "config.yaml", "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)

        print(f"\n[train] Run directory: {run_dir}")
        print(f"[train] Total timesteps: {cfg['training']['total_timesteps']:,}")
        print(f"[train] Learning rate:   {cfg['model']['learning_rate']}")
        print(f"[train] Seed:            {cfg['training']['seed']}\n")

        m, e = cfg["model"], cfg["exploration"]
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

        total_timesteps = cfg["training"]["total_timesteps"]
        remaining = total_timesteps
        callback = TrainingCallback()

    # ── SHARED: TRAINING LOOP ─────────────────────────────────────────────

    def handle_interrupt(sig, frame):
        print("\n\n[train] Pausing... saving checkpoint.")
        save_progress(run_dir, model, callback, total_timesteps)
        save_plots(callback, run_dir)
        print(f"\n[train] Paused. Resume with:")
        print(f"        python train.py --resume {run_dir}\n")
        env.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_interrupt)
    signal.signal(signal.SIGTERM, handle_interrupt)  # docker stop sends SIGTERM

    model.learn(
        total_timesteps=remaining,
        callback=callback,
        progress_bar=False,
        reset_num_timesteps=False,  # preserves step count on resume
    )

    # Save final state (fully trained)
    save_progress(run_dir, model, callback, total_timesteps)
    save_plots(callback, run_dir)

    env.close()
    print(f"\n[train] Training complete! Results in: {run_dir}/")
    print(f"        Playback: python playback.py --run {run_dir}")


if __name__ == "__main__":
    main()
