"""Main training script for DRL Drone Collision Avoidance.

Usage:
    # Train with default config
    python -m training.train

    # Train with custom config
    python -m training.train --config training/configs/my_experiment.yaml

    # Override specific parameters from CLI
    python -m training.train --agent.gamma 0.95 --training.max_episodes 5000

    # Monitor with TensorBoard
    tensorboard --logdir tb_logs
"""

import os
import sys
import numpy as np

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["OMP_NUM_THREADS"] = "1"

import torch

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

from training.configs.config import load_config, parse_cli_to_overrides
from training.agents.dueling_ddqn import DuelingDDQNAgent
from training.envs.env_factory import get_make_env_fn


def main():
    # Load config with CLI overrides
    config_path, overrides = parse_cli_to_overrides()
    cfg = load_config(config_path, overrides)

    print("=" * 60)
    print("DRL Drone Collision Avoidance - Training")
    print("=" * 60)
    print(f"Config:\n{cfg}")

    # Create output directories
    os.makedirs(cfg.paths.results_dir, exist_ok=True)
    os.makedirs(cfg.paths.model_dir, exist_ok=True)
    os.makedirs(cfg.paths.log_dir, exist_ok=True)

    # Environment factory
    make_env_fn, make_env_kargs = get_make_env_fn(cfg)

    # Train across seeds
    all_results = []
    best_agent = None
    best_eval_score = float("-inf")

    for seed in cfg.training.seeds:
        print(f"\n{'='*60}")
        print(f"Training with seed={seed}")
        print(f"{'='*60}")

        agent = DuelingDDQNAgent(cfg)
        result, final_eval_score, training_time, wallclock_time = agent.train(
            make_env_fn, make_env_kargs, seed
        )

        all_results.append(result)

        if final_eval_score > best_eval_score:
            best_eval_score = final_eval_score
            best_agent = agent

    # Save the best model
    if best_agent is not None:
        model_path = best_agent.save_model()
        print(f"\nBest model (eval={best_eval_score:.2f}) saved to: {model_path}")

    # Save training curves
    all_results = np.array(all_results)
    results_path = os.path.join(cfg.paths.results_dir, "training_results.npy")
    np.save(results_path, all_results)
    print(f"Training results saved to: {results_path}")

    # Plot if matplotlib is available
    try:
        _plot_results(all_results, cfg)
    except Exception as e:
        print(f"[WARN] Could not plot results: {e}")

    print("\nDone!")


def _plot_results(results, cfg):
    """Plot training curves and save to file."""
    import matplotlib.pyplot as plt

    # results shape: (n_seeds, max_episodes, 5)
    # columns: total_steps, mean_100_reward, mean_100_eval, training_time, wallclock
    mean_r = np.nanmean(results, axis=0)
    max_r = np.nanmax(results, axis=0)
    min_r = np.nanmin(results, axis=0)

    fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    episodes = np.arange(mean_r.shape[0])

    # Mean 100 training reward
    axs[0].plot(episodes, mean_r[:, 1], "b-", linewidth=2, label="Mean")
    if results.shape[0] > 1:
        axs[0].fill_between(episodes, min_r[:, 1], max_r[:, 1], alpha=0.2)
    axs[0].set_title("Training Reward (Moving Avg 100)")
    axs[0].set_ylabel("Reward")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    # Mean 100 eval score
    axs[1].plot(episodes, mean_r[:, 2], "r-", linewidth=2, label="Mean")
    if results.shape[0] > 1:
        axs[1].fill_between(episodes, min_r[:, 2], max_r[:, 2], alpha=0.2)
    axs[1].set_title("Evaluation Score (Moving Avg 100)")
    axs[1].set_ylabel("Score")
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    # Total steps
    axs[2].plot(episodes, mean_r[:, 0], "g-", linewidth=2)
    axs[2].set_title("Total Steps")
    axs[2].set_xlabel("Episode")
    axs[2].set_ylabel("Steps")
    axs[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(cfg.paths.results_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Training curves saved to: {plot_path}")
    plt.close()


if __name__ == "__main__":
    main()
