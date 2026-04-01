"""Batch training script for experiment models.

Trains models in sets of 3 (grouped by observation type) by merging
the experiment-specific YAML on top of the default config.

Usage:
    python -m training.train_all --set D   # Train D-5, D-10, D-15 (ray)
    python -m training.train_all --set S   # Train S-5, S-10, S-15 (sensor)
    python -m training.train_all --set X   # Train X-5, X-10, X-15 (sector)
    python -m training.train_all --models S-5 S-10   # train specific models only
"""

import os
import sys
import time
import argparse

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["OMP_NUM_THREADS"] = "1"

import torch

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

from training.configs.config import load_config, _deep_merge
from training.agents.dueling_ddqn import DuelingDDQNAgent
from training.envs.env_factory import get_make_env_fn

# All 9 experiment model names in structured order
ALL_MODELS = [
    "D-5",  "D-10", "D-15",
    "S-5",  "S-10", "S-15",
    "X-5",  "X-10", "X-15",
]

CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "configs", "experiments")


def train_model(model_name: str):
    """Train a single model by merging its experiment config on top of default."""
    print(f"\n{'#'*70}")
    print(f"# TRAINING MODEL: {model_name}")
    print(f"{'#'*70}\n")

    # Load default config, then merge experiment overrides
    default_path = os.path.join(os.path.dirname(__file__), "configs", "default.yaml")
    experiment_path = os.path.join(CONFIGS_DIR, f"{model_name}.yaml")

    if not os.path.exists(experiment_path):
        print(f"[ERROR] Config not found: {experiment_path}")
        return None

    cfg = load_config(default_path)

    # Load experiment overrides and merge
    import yaml
    with open(experiment_path, "r") as f:
        overrides = yaml.safe_load(f)
    merged = _deep_merge(cfg.to_dict(), overrides)

    from training.configs.config import _dict_to_namespace, Config
    ns = _dict_to_namespace(merged)
    cfg = Config(**vars(ns))

    print(f"  obs_choice:   {cfg.env.obs_choice}")
    print(f"  num_sensors:  {cfg.sensor.num_sensors}")
    print(f"  num_drones:   {cfg.env.num_drones}")
    print(f"  max_episodes: {cfg.training.max_episodes}")
    print(f"  results_dir:  {cfg.paths.results_dir}")
    print(f"  model_dir:    {cfg.paths.model_dir}")
    print(f"  log_dir:      {cfg.paths.log_dir}")

    # Create output directories
    os.makedirs(cfg.paths.results_dir, exist_ok=True)
    os.makedirs(cfg.paths.model_dir, exist_ok=True)
    os.makedirs(cfg.paths.log_dir, exist_ok=True)

    # Environment factory
    make_env_fn, make_env_kargs = get_make_env_fn(cfg)

    # Train
    start_time = time.time()
    for seed in cfg.training.seeds:
        print(f"\n  Training with seed={seed}")
        agent = DuelingDDQNAgent(cfg)
        result, final_eval_score, training_time, wallclock_time = agent.train(
            make_env_fn, make_env_kargs, seed
        )

    elapsed = time.time() - start_time
    print(f"\n  [{model_name}] Done in {elapsed:.0f}s. Final eval: {final_eval_score:.2f}")

    # Save the best model with structured name
    if agent.online_model is not None:
        model_path = os.path.join(cfg.paths.model_dir, "best_model.pth")
        torch.save(agent.online_model.state_dict(), model_path)
        print(f"  Model saved -> {model_path}")

    # Save training results
    results_path = os.path.join(cfg.paths.results_dir, "training_results.npy")
    np.save(results_path, result)

    return final_eval_score


# Model sets grouped by observation type
MODEL_SETS = {
    "D": ["D-5", "D-10", "D-15"],   # obs_choice: ray
    "S": ["S-5", "S-10", "S-15"],   # obs_choice: sensor
    "X": ["X-5", "X-10", "X-15"],   # obs_choice: sector
}


def main():
    parser = argparse.ArgumentParser(description="Train experiment models by set")
    parser.add_argument(
        "--set", type=str, default=None, choices=["D", "S", "X"],
        help="Train one set: D (ray), S (sensor), or X (sector)"
    )
    parser.add_argument(
        "--models", nargs="*", default=None,
        help=f"Specific models to train. Choices: {ALL_MODELS}"
    )
    args = parser.parse_args()

    if args.models:
        models_to_train = args.models
    elif args.set:
        models_to_train = MODEL_SETS[args.set]
    else:
        print("Please specify --set (D, S, or X) or --models.")
        print("  --set D   -> D-5, D-10, D-15  (ray)")
        print("  --set S   -> S-5, S-10, S-15  (sensor)")
        print("  --set X   -> X-5, X-10, X-15  (sector)")
        sys.exit(1)

    # Validate model names
    for m in models_to_train:
        if m not in ALL_MODELS:
            print(f"[ERROR] Unknown model: {m}. Valid: {ALL_MODELS}")
            sys.exit(1)

    print("=" * 70)
    print("BATCH TRAINING - DRL Drone Collision Avoidance")
    print("=" * 70)
    print(f"Models to train: {models_to_train}")
    print(f"Total: {len(models_to_train)} models")

    results_summary = {}
    total_start = time.time()

    for model_name in models_to_train:
        score = train_model(model_name)
        results_summary[model_name] = score

    total_elapsed = time.time() - total_start

    # Print summary
    print(f"\n{'='*70}")
    print("BATCH TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {total_elapsed:.0f}s ({total_elapsed/3600:.1f}h)")
    print(f"\n{'Model':<10} {'Final Eval Score':>18}")
    print("-" * 30)
    for name, score in results_summary.items():
        score_str = f"{score:.2f}" if score is not None else "FAILED"
        print(f"{name:<10} {score_str:>18}")


if __name__ == "__main__":
    main()
