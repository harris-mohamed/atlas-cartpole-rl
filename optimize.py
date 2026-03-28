"""
CartPole DQN Hyperparameter Optimizer
--------------------------------------
Uses Optuna to search for the best DQN hyperparameters overnight.
Each trial trains for a fixed number of timesteps, then evaluates
mean reward. Bad trials are pruned early so time isn't wasted.

Study is saved to SQLite — safe to interrupt and resume anytime.

Usage:
    python optimize.py                     # 50 trials, default settings
    python optimize.py --trials 100        # more trials = better search
    python optimize.py --eval-steps 30000  # faster trials (less accurate)
    python optimize.py --resume            # continue an interrupted study
    python optimize.py --best              # show best params found so far

Outputs (saved to optimization/):
    study.db                   - SQLite database (for --resume)
    best_params.yaml           - best hyperparameters found
    results.csv                - all trial results
    optimization_history.png   - reward per trial + best-so-far line
    param_importances.png      - which hyperparams mattered most
"""

import argparse
import warnings
from pathlib import Path

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import yaml
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


STUDY_NAME = "cartpole-dqn"
OUTPUT_DIR = Path("optimization")


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------

def sample_hyperparams(trial: optuna.Trial) -> dict:
    """Define which hyperparameters to search and their ranges."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "buffer_size": trial.suggest_categorical("buffer_size", [10_000, 50_000, 100_000]),
        "gamma": trial.suggest_float("gamma", 0.90, 0.9999),
        "target_update_interval": trial.suggest_categorical("target_update_interval", [100, 500, 1000]),
        "learning_starts": trial.suggest_categorical("learning_starts", [500, 1000, 2000]),
        "exploration_fraction": trial.suggest_float("exploration_fraction", 0.05, 0.5),
        "exploration_final_eps": trial.suggest_float("exploration_final_eps", 0.01, 0.1),
    }


# ---------------------------------------------------------------------------
# Trial evaluation
# ---------------------------------------------------------------------------

def evaluate_trial(trial: optuna.Trial, eval_timesteps: int, n_eval_episodes: int) -> float:
    """Train a DQN with sampled params. Returns mean reward over eval episodes."""
    params = sample_hyperparams(trial)

    env = Monitor(gym.make("CartPole-v1"))
    eval_env = Monitor(gym.make("CartPole-v1"))

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=params["learning_rate"],
        batch_size=params["batch_size"],
        buffer_size=params["buffer_size"],
        gamma=params["gamma"],
        target_update_interval=params["target_update_interval"],
        learning_starts=params["learning_starts"],
        exploration_fraction=params["exploration_fraction"],
        exploration_final_eps=params["exploration_final_eps"],
        verbose=0,
    )

    # Train in chunks so Optuna can prune bad trials early
    n_chunks = 5
    chunk_size = max(eval_timesteps // n_chunks, 1000)

    try:
        for i in range(n_chunks):
            model.learn(total_timesteps=chunk_size, reset_num_timesteps=(i == 0))

            # Report intermediate reward so pruner can kill bad trials
            mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=10, warn=False)
            trial.report(mean_reward, step=i)

            if trial.should_prune():
                raise optuna.TrialPruned()

        # Final evaluation with full episode count
        mean_reward, _ = evaluate_policy(
            model, eval_env, n_eval_episodes=n_eval_episodes, warn=False
        )
    finally:
        env.close()
        eval_env.close()

    return mean_reward


# ---------------------------------------------------------------------------
# Plots and output
# ---------------------------------------------------------------------------

def save_plots(study: optuna.Study):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    completed = [t for t in study.trials if t.value is not None]

    if not completed:
        return

    values = [t.value for t in completed]
    best_so_far = np.maximum.accumulate(values)

    # 1. Optimization history
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(range(len(values)), values, alpha=0.4, color="steelblue", s=20,
               label="Trial reward")
    ax.plot(range(len(best_so_far)), best_so_far, color="steelblue", linewidth=2,
            label="Best so far")
    ax.axhline(y=500, color="green", linestyle="--", alpha=0.6, label="Solved (500)")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Optimization History — CartPole DQN")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "optimization_history.png", dpi=150)
    plt.close(fig)

    # 2. Hyperparameter importance (needs >= 4 completed trials)
    if len(completed) >= 4:
        try:
            importances = optuna.importance.get_param_importances(study)
            params_sorted = sorted(importances.items(), key=lambda x: x[1])
            labels = [k for k, _ in params_sorted]
            vals = [v for _, v in params_sorted]

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(labels, vals, color="steelblue")
            ax.set_xlabel("Importance Score")
            ax.set_title("Hyperparameter Importance")
            ax.grid(True, alpha=0.3, axis="x")
            plt.tight_layout()
            fig.savefig(OUTPUT_DIR / "param_importances.png", dpi=150)
            plt.close(fig)
        except Exception:
            pass  # not enough diversity yet

    # 3. Results CSV
    if completed:
        param_keys = list(completed[0].params.keys())
        with open(OUTPUT_DIR / "results.csv", "w") as f:
            f.write("trial,reward," + ",".join(param_keys) + "\n")
            for t in completed:
                row = (f"{t.number},{t.value:.2f},"
                       + ",".join(str(t.params[k]) for k in param_keys))
                f.write(row + "\n")


def save_best_params(study: optuna.Study) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best = study.best_trial
    output = {
        "best_trial": best.number,
        "best_mean_reward": round(best.value, 2),
        "params": best.params,
    }
    with open(OUTPUT_DIR / "best_params.yaml", "w") as f:
        yaml.dump(output, f, default_flow_style=False)
    return output


def print_best(study: optuna.Study):
    try:
        best = study.best_trial
        pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        completed = len([t for t in study.trials if t.value is not None])

        print(f"\n{'='*52}")
        print(f"  Best trial:        #{best.number}")
        print(f"  Best mean reward:  {best.value:.2f} / 500")
        print(f"  Trials completed:  {completed} ({pruned} pruned early)")
        print(f"\n  Best hyperparameters:")
        for k, v in best.params.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.6g}")
            else:
                print(f"    {k}: {v}")
        print(f"{'='*52}")
        print(f"\n  Run a full training with these params:")
        p = best.params
        print(f"  python train.py \\")
        print(f"    --lr {p['learning_rate']:.6g} \\")
        print(f"    --batch-size {p['batch_size']} \\")
        print(f"    --buffer-size {p['buffer_size']} \\")
        print(f"    --gamma {p['gamma']:.4f}")
        print()
    except ValueError:
        print("No completed trials yet.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Overnight hyperparameter optimization for CartPole DQN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python optimize.py                     # 50 trials (good overnight run)
  python optimize.py --trials 100        # more thorough search
  python optimize.py --eval-steps 30000  # faster but less accurate per trial
  python optimize.py --resume            # continue an interrupted study
  python optimize.py --best              # show best params found so far
        """,
    )
    parser.add_argument("--trials", type=int, default=50,
                        help="Number of trials to run (default: 50)")
    parser.add_argument("--eval-steps", type=int, default=50_000,
                        help="Training timesteps per trial (default: 50000)")
    parser.add_argument("--eval-episodes", type=int, default=20,
                        help="Evaluation episodes per trial for final score (default: 20)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume an interrupted optimization study")
    parser.add_argument("--best", action="store_true",
                        help="Print best params found so far and exit")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{OUTPUT_DIR}/study.db"

    # ── Just show best ─────────────────────────────────────────────────────
    if args.best:
        try:
            study = optuna.load_study(study_name=STUDY_NAME, storage=storage)
            print_best(study)
            save_best_params(study)
            save_plots(study)
            print(f"[optimize] Plots updated in {OUTPUT_DIR}/")
        except Exception as e:
            print(f"[optimize] No study found. Run optimize.py first. ({e})")
        return

    # ── Suppress noise ─────────────────────────────────────────────────────
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore")

    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=2)

    if args.resume:
        print(f"[optimize] Resuming study from {OUTPUT_DIR}/study.db")
        study = optuna.load_study(
            study_name=STUDY_NAME,
            storage=storage,
            sampler=sampler,
            pruner=pruner,
        )
        already_done = len([t for t in study.trials if t.value is not None])
        print(f"[optimize] {already_done} trials already completed\n")
    else:
        study = optuna.create_study(
            study_name=STUDY_NAME,
            storage=storage,
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )

    total_after = len(study.trials) + args.trials
    print(f"[optimize] Running {args.trials} trials ({args.eval_steps:,} steps each)")
    print(f"[optimize] Bad trials pruned early to save time")
    print(f"[optimize] Results → {OUTPUT_DIR}/")
    print(f"[optimize] Safe to Ctrl+C — resume with: python optimize.py --resume\n")

    def objective(trial):
        n = trial.number + 1
        print(f"[optimize] Trial {n}/{total_after} ...", end="", flush=True)
        reward = evaluate_trial(trial, args.eval_steps, args.eval_episodes)
        print(f" reward={reward:.1f}")
        return reward

    try:
        study.optimize(objective, n_trials=args.trials, catch=(Exception,))
    except KeyboardInterrupt:
        print("\n[optimize] Interrupted — saving results so far.")

    print("\n[optimize] Done.")

    try:
        save_best_params(study)
        save_plots(study)
        print_best(study)
        print(f"[optimize] optimization_history.png → {OUTPUT_DIR}/")
        print(f"[optimize] param_importances.png    → {OUTPUT_DIR}/")
        print(f"[optimize] best_params.yaml         → {OUTPUT_DIR}/")
    except ValueError:
        print("[optimize] No completed trials to report.")


if __name__ == "__main__":
    main()
