"""Evaluation / testing script for trained DRL models.

Usage:
    # Evaluate with GUI visualization
    python -m training.evaluate --model trained_models/best_model.pth --gui

    # Evaluate without GUI (headless, for metrics only)
    python -m training.evaluate --model trained_models/best_model.pth --episodes 50

    # Use a specific config
    python -m training.evaluate --model best_model.pth --config training/configs/default.yaml
"""

import os
import sys
import time
import argparse
import itertools

import numpy as np
import torch
import pybullet as p

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from training.configs.config import load_config
from training.networks.fc_dueling_q import FCDuelingQ
from training.envs.env_factory import make_env


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained DRL model")
    parser.add_argument("--model", type=str, required=True, help="Path to .pth model file")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--episodes", type=int, default=20, help="Number of evaluation episodes")
    parser.add_argument("--gui", action="store_true", help="Enable PyBullet GUI")
    parser.add_argument("--realtime", action="store_true", help="Sync to real-time (slower, for visualization)")
    return parser.parse_args()


def select_action(model, state, device="cpu"):
    """Greedy action selection."""
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = model(state_t)
    return q_values.argmax(dim=-1).item()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    print("=" * 60)
    print("DRL Drone Collision Avoidance - Evaluation")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Episodes: {args.episodes}")
    print(f"GUI: {args.gui}")

    # Create environment
    env = make_env(cfg, gui=args.gui)
    nS = env.observation_space.shape[0]
    nA = env.action_space.n

    # Load model
    device = "cpu"
    model = FCDuelingQ(
        input_dim=nS,
        output_dim=nA,
        hidden_dims=tuple(cfg.network.hidden_dims),
        device=device,
    )
    model.load_state_dict(
        torch.load(args.model, map_location=torch.device(device), weights_only=True)
    )
    model.eval()
    print(f"Model loaded: {nS} obs -> {nA} actions")

    # Run evaluation
    episode_rewards = []
    episode_steps = []
    episode_outcomes = []  # "reached", "collision", "timeout", "truncated"

    ACTION_LABELS = {0: "ACCELERATE", 1: "DECELERATE", 2: "CONSTANT"}
    ACTION_COLORS = {0: [0, 1, 0], 1: [1, 0.5, 0], 2: [0, 0.5, 1]}
    action_text_id = -1

    for ep in range(args.episodes):
        state, _ = env.reset()
        ep_reward = 0.0
        step = 0
        start_time = time.time()

        while True:
            action = select_action(model, state, device)
            next_state, reward, terminated, truncated, info = env.step(action)

            ep_reward += reward
            step += 1
            state = next_state

            if args.gui:
                env.render()
                if args.realtime:
                    from gym_pybullet_drones.utils.utils import sync
                    sync(step, start_time, env.CTRL_TIMESTEP)

                # Following camera on the agent drone
                p.resetDebugVisualizerCamera(
                    cameraDistance=35, cameraYaw=0, cameraPitch=-60,
                    cameraTargetPosition=env.routing[0].CUR_POS)

                # Action label above the agent drone
                drone_pos = env.routing[0].CUR_POS
                label_pos = [drone_pos[0], drone_pos[1], drone_pos[2] + 1.5]
                label = ACTION_LABELS.get(action, str(action))
                color = ACTION_COLORS.get(action, [1, 1, 1])
                if action_text_id >= 0:
                    p.removeUserDebugItem(action_text_id)
                action_text_id = p.addUserDebugText(
                    label, label_pos, textColorRGB=color, textSize=2.0, lifeTime=0.1)

            if terminated or truncated:
                # Determine outcome
                drone_state = env._getDroneStateVector(0)
                d2dest = np.linalg.norm(env.routing[0].DESTINATION - drone_state[0:3])

                if d2dest < cfg.reward.reach_threshold_m:
                    outcome = "reached"
                elif int(env.CONTACT_FLAGS[0]) == 1:
                    outcome = "collision"
                elif truncated:
                    outcome = "timeout"
                else:
                    outcome = "unknown"

                episode_outcomes.append(outcome)
                break

        episode_rewards.append(ep_reward)
        episode_steps.append(step)

        print(
            f"  Episode {ep+1:3d}/{args.episodes}: "
            f"reward={ep_reward:+8.2f}, steps={step:4d}, outcome={outcome}"
        )

    env.close()

    # Summary statistics
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Episodes:     {args.episodes}")
    print(f"Mean reward:  {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    print(f"Mean steps:   {np.mean(episode_steps):.1f} +/- {np.std(episode_steps):.1f}")

    # Outcome breakdown
    from collections import Counter
    outcome_counts = Counter(episode_outcomes)
    total = len(episode_outcomes)
    print(f"\nOutcomes:")
    for outcome in ["reached", "collision", "timeout", "unknown"]:
        count = outcome_counts.get(outcome, 0)
        pct = 100 * count / total if total > 0 else 0
        print(f"  {outcome:12s}: {count:3d} ({pct:5.1f}%)")

    # Success rate
    success_rate = 100 * outcome_counts.get("reached", 0) / total if total > 0 else 0
    collision_rate = 100 * outcome_counts.get("collision", 0) / total if total > 0 else 0
    print(f"\nSuccess rate:   {success_rate:.1f}%")
    print(f"Collision rate: {collision_rate:.1f}%")


if __name__ == "__main__":
    main()
