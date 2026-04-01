"""Dueling Double DQN agent with clean structure and TensorBoard logging.

Key fixes over the original implementation:
1. Experience tuple order is correct (states, actions, rewards, next_states, terminals)
2. Proper handling of terminated vs truncated (timeout != failure)
3. Evaluation uses different seeds for generalization
4. TensorBoard logging for all metrics
5. Configurable via Config object
"""

import os
import csv
import time
import glob
import random
import tempfile
from datetime import datetime

import numpy as np
import torch

from training.buffers.replay_buffer import ReplayBuffer
from training.strategies.exploration import GreedyStrategy


class DuelingDDQNAgent:
    """Dueling Double DQN with soft target updates and proper logging."""

    def __init__(self, cfg):
        """Initialize from a Config namespace.

        Parameters
        ----------
        cfg : Config
            Must contain: cfg.agent, cfg.network, cfg.optimizer,
            cfg.buffer, cfg.exploration, cfg.training, cfg.paths
        """
        self.cfg = cfg

        # Will be set during train()
        self.online_model = None
        self.target_model = None
        self.value_optimizer = None
        self.replay_buffer = None
        self.training_strategy = None
        self.evaluation_strategy = None

        # Metrics
        self.episode_rewards = []
        self.episode_timesteps = []
        self.episode_exploration = []
        self.evaluation_scores = []
        self.episode_seconds = []
        self.episode_outcomes = []  # "reached", "collision", "timeout"
        self.episode_intrusions = []  # count of intrusion steps per episode
        self._ep_intrusion_count = 0
        self._last_info = {}

        # TensorBoard writer (lazy init)
        self._writer = None

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    def _build_model(self, nS, nA):
        from training.networks.fc_dueling_q import FCDuelingQ

        return FCDuelingQ(
            input_dim=nS,
            output_dim=nA,
            hidden_dims=tuple(self.cfg.network.hidden_dims),
        )

    def _build_optimizer(self, model):
        lr = self.cfg.optimizer.lr
        opt_type = self.cfg.optimizer.type.lower()
        if opt_type == "adam":
            return torch.optim.Adam(model.parameters(), lr=lr)
        elif opt_type == "rmsprop":
            return torch.optim.RMSprop(model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unknown optimizer type: {opt_type}")

    def _build_exploration_strategy(self):
        from training.strategies.exploration import (
            EGreedyExpStrategy,
            EGreedyLinearStrategy,
            SoftMaxStrategy,
        )

        s = self.cfg.exploration
        name = s.strategy.lower()
        if name == "e_greedy_exp":
            return EGreedyExpStrategy(s.init_epsilon, s.min_epsilon, s.decay_steps)
        elif name == "e_greedy_linear":
            return EGreedyLinearStrategy(s.init_epsilon, s.min_epsilon, s.decay_steps)
        elif name == "softmax":
            return SoftMaxStrategy()
        else:
            raise ValueError(f"Unknown exploration strategy: {name}")

    def _get_writer(self):
        if self._writer is None:
            try:
                from torch.utils.tensorboard import SummaryWriter

                log_dir = os.path.join(
                    self.cfg.paths.log_dir,
                    datetime.now().strftime("%Y%m%d_%H%M%S"),
                )
                self._writer = SummaryWriter(log_dir=log_dir)
                print(f"[INFO] TensorBoard logs -> {log_dir}")
            except ImportError:
                print("[WARN] tensorboard not installed, logging disabled")
                self._writer = None
        return self._writer

    # ------------------------------------------------------------------
    # Core algorithm
    # ------------------------------------------------------------------

    def optimize_model(self, experiences):
        """Compute Dueling DDQN loss and update online network."""
        states, actions, rewards, next_states, is_terminals = experiences
        batch_size = len(is_terminals)

        # Double DQN: select action with online, evaluate with target
        with torch.no_grad():
            argmax_a_q_sp = self.online_model(next_states).argmax(dim=1)
            q_sp = self.target_model(next_states)
            max_a_q_sp = q_sp.gather(1, argmax_a_q_sp.unsqueeze(1))
            target_q_sa = rewards + (self.cfg.agent.gamma * max_a_q_sp * (1 - is_terminals))

        q_sa = self.online_model(states).gather(1, actions)

        td_error = q_sa - target_q_sa
        loss = td_error.pow(2).mul(0.5).mean()

        self.value_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.online_model.parameters(), self.cfg.agent.max_gradient_norm
        )
        self.value_optimizer.step()

        return loss.item()

    def interaction_step(self, state, env):
        """Take one step in the environment, store transition.

        FIX: Properly distinguishes terminated vs truncated.
        - terminated + not truncated = real failure (store is_terminal=1)
        - truncated (timeout) = NOT a failure (store is_terminal=0)
          so the value estimate is not zeroed out for timeouts.
        """
        action = self.training_strategy.select_action(self.online_model, state)
        new_state, reward, terminated, truncated, info = env.step(action)

        # Only treat as terminal for value bootstrapping if it's a real failure
        # (collision or reaching goal), NOT a timeout
        is_failure = float(terminated and not truncated)

        experience = (state, action, reward, new_state, is_failure)
        self.replay_buffer.store(experience)

        self.episode_rewards[-1] += reward
        self.episode_timesteps[-1] += 1
        self.episode_exploration[-1] += int(
            self.training_strategy.exploratory_action_taken
        )

        # Track per-step info metrics
        self._last_info = info
        if info.get("intrusion", False):
            self._ep_intrusion_count += 1

        episode_over = terminated or truncated
        return new_state, episode_over

    @torch.no_grad()
    def soft_update_target(self):
        """Polyak averaging: target = (1-tau)*target + tau*online."""
        tau = self.cfg.agent.tau
        for target_p, online_p in zip(
            self.target_model.parameters(), self.online_model.parameters()
        ):
            target_p.data.mul_(1.0 - tau).add_(online_p.data, alpha=tau)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, make_env_fn, make_env_kargs, seed):
        """Main training loop.

        Parameters
        ----------
        make_env_fn : callable
            Factory function that creates the gym environment.
        make_env_kargs : dict
            Keyword arguments for make_env_fn.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        result : np.ndarray
            (max_episodes, 5) array of training metrics per episode.
        final_eval_score : float
        training_time : float
        wallclock_time : float
        """
        tcfg = self.cfg.training
        acfg = self.cfg.agent

        training_start = time.time()
        last_log_time = float("-inf")

        self.checkpoint_dir = tempfile.mkdtemp()
        self.make_env_fn = make_env_fn
        self.make_env_kargs = make_env_kargs
        self.seed = seed

        # Seed everything
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Create environment
        env = make_env_fn(**make_env_kargs, seed=seed)
        nS = env.observation_space.shape[0]
        nA = env.action_space.n
        print(f"[INFO] Observation dim: {nS}, Action dim: {nA}")

        # Build components
        self.online_model = self._build_model(nS, nA)
        self.target_model = self._build_model(nS, nA)
        # Initialize target = online
        self.target_model.load_state_dict(self.online_model.state_dict())

        self.value_optimizer = self._build_optimizer(self.online_model)
        self.replay_buffer = ReplayBuffer(
            max_size=self.cfg.buffer.max_size, batch_size=self.cfg.buffer.batch_size
        )
        self.training_strategy = self._build_exploration_strategy()
        self.evaluation_strategy = GreedyStrategy()

        # Reset metrics
        self.episode_rewards = []
        self.episode_timesteps = []
        self.episode_exploration = []
        self.evaluation_scores = []
        self.episode_seconds = []

        writer = self._get_writer()

        result = np.full((tcfg.max_episodes, 5), np.nan)
        training_time = 0
        total_steps = 0
        best_eval_score = float("-inf")

        for episode in range(1, tcfg.max_episodes + 1):
            episode_start = time.time()

            state, _ = env.reset()
            self.episode_rewards.append(0.0)
            self.episode_timesteps.append(0.0)
            self.episode_exploration.append(0.0)
            self._ep_intrusion_count = 0

            episode_losses = []
            step = 0
            while True:
                state, episode_over = self.interaction_step(state, env)
                step += 1
                total_steps += 1

                # Learn from replay buffer
                min_samples = self.replay_buffer.batch_size * acfg.n_warmup_batches
                if len(self.replay_buffer) > min_samples:
                    experiences = self.replay_buffer.sample()
                    experiences = self.online_model.load_experiences(experiences)
                    loss = self.optimize_model(experiences)
                    episode_losses.append(loss)

                # Soft target update
                if total_steps % acfg.update_target_every_steps == 0:
                    self.soft_update_target()

                if episode_over:
                    break

            # Determine episode outcome from last info
            info = self._last_info
            if info.get("reached", False):
                outcome = "reached"
            elif info.get("collision", False):
                outcome = "collision"
            else:
                outcome = "timeout"
            self.episode_outcomes.append(outcome)
            self.episode_intrusions.append(self._ep_intrusion_count)

            # Episode stats
            episode_elapsed = time.time() - episode_start
            self.episode_seconds.append(episode_elapsed)
            training_time += episode_elapsed

            # Periodic evaluation
            eval_score = None
            if episode % tcfg.eval_every_episodes == 0:
                eval_score, eval_std = self.evaluate(
                    self.online_model, env, n_episodes=tcfg.eval_episodes
                )
                self.evaluation_scores.append(eval_score)

                if eval_score > best_eval_score:
                    best_eval_score = eval_score
                    self._save_best_model()
            else:
                # Use last known eval score for running stats
                if self.evaluation_scores:
                    self.evaluation_scores.append(self.evaluation_scores[-1])
                else:
                    self.evaluation_scores.append(0.0)

            # Save checkpoint
            if episode % tcfg.save_every_episodes == 0:
                self.save_checkpoint(episode, self.online_model)

            # Compute running stats
            mean_10_reward = np.mean(self.episode_rewards[-10:])
            mean_100_reward = np.mean(self.episode_rewards[-100:])
            mean_100_eval = np.mean(self.evaluation_scores[-100:])
            std_100_eval = np.std(self.evaluation_scores[-100:])

            ep_exploration_ratio = (
                self.episode_exploration[-1] / max(self.episode_timesteps[-1], 1)
            )
            mean_loss = np.mean(episode_losses) if episode_losses else 0.0

            wallclock_elapsed = time.time() - training_start
            result[episode - 1] = [
                total_steps,
                mean_100_reward,
                mean_100_eval,
                training_time,
                wallclock_elapsed,
            ]

            # TensorBoard logging
            if writer is not None:
                writer.add_scalar("reward/episode", self.episode_rewards[-1], episode)
                writer.add_scalar("reward/mean_10", mean_10_reward, episode)
                writer.add_scalar("reward/mean_100", mean_100_reward, episode)
                writer.add_scalar("loss/td_error", mean_loss, episode)
                writer.add_scalar("exploration/epsilon", self.training_strategy.epsilon, episode)
                writer.add_scalar("exploration/ratio", ep_exploration_ratio, episode)
                writer.add_scalar("episode/steps", self.episode_timesteps[-1], episode)
                writer.add_scalar("episode/seconds", episode_elapsed, episode)
                writer.add_scalar("episode/intrusions", self.episode_intrusions[-1], episode)
                # Running outcome rates (last 100 episodes)
                recent = self.episode_outcomes[-100:]
                writer.add_scalar("outcome/success_rate_100", recent.count("reached") / len(recent), episode)
                writer.add_scalar("outcome/collision_rate_100", recent.count("collision") / len(recent), episode)
                writer.add_scalar("outcome/timeout_rate_100", recent.count("timeout") / len(recent), episode)
                if eval_score is not None:
                    writer.add_scalar("eval/score", eval_score, episode)
                    writer.add_scalar("eval/mean_100", mean_100_eval, episode)

            # Console logging
            reached_log_time = time.time() - last_log_time >= tcfg.log_every_seconds
            reached_max_minutes = wallclock_elapsed >= tcfg.max_minutes * 60
            reached_max_episodes = episode >= tcfg.max_episodes
            reached_goal = mean_100_eval >= tcfg.goal_mean_100_reward
            training_is_over = reached_max_minutes or reached_max_episodes or reached_goal

            elapsed_str = time.strftime(
                "%H:%M:%S", time.gmtime(wallclock_elapsed)
            )
            msg = (
                f"[{elapsed_str}] ep {episode:04d} | "
                f"steps {total_steps:06d} | "
                f"r={self.episode_rewards[-1]:+.1f} | "
                f"r10={mean_10_reward:+.1f} | "
                f"r100={mean_100_reward:+.1f} | "
                f"eps={self.training_strategy.epsilon:.3f} | "
                f"loss={mean_loss:.4f} | "
                f"{outcome}"
            )
            if eval_score is not None:
                msg += f" | eval={eval_score:.1f}\u00B1{eval_std:.1f}"

            print(msg, end="\r", flush=True)
            if reached_log_time or training_is_over:
                print(msg, flush=True)
                last_log_time = time.time()

            if training_is_over:
                if reached_max_minutes:
                    print("--> Reached max training time")
                if reached_max_episodes:
                    print("--> Reached max episodes")
                if reached_goal:
                    print("--> Reached goal reward!")
                break

        # Final evaluation
        final_eval_score, score_std = self.evaluate(
            self.online_model, env, n_episodes=50
        )
        wallclock_time = time.time() - training_start
        print(
            f"\nTraining complete. Final eval: {final_eval_score:.2f}\u00B1{score_std:.2f} "
            f"in {training_time:.0f}s training, {wallclock_time:.0f}s wall-clock."
        )

        if writer is not None:
            writer.close()

        # Export training history to CSV
        self.export_training_csv()

        env.close()
        del env
        self._cleanup_checkpoints()

        return result, final_eval_score, training_time, wallclock_time

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, model, env, n_episodes=5):
        """Evaluate the model greedily for n_episodes.

        FIX: Does NOT use the training seed, so evaluation tests generalization.
        """
        rewards = []
        for ep in range(n_episodes):
            state, _ = env.reset()  # No fixed seed -> random scenarios
            episode_reward = 0.0
            while True:
                action = self.evaluation_strategy.select_action(model, state)
                state, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                if terminated or truncated:
                    break
            rewards.append(episode_reward)
        return float(np.mean(rewards)), float(np.std(rewards))

    def export_training_csv(self, path: str = None):
        """Export per-episode training metrics to CSV."""
        if path is None:
            os.makedirs(self.cfg.paths.results_dir, exist_ok=True)
            path = os.path.join(self.cfg.paths.results_dir, "training_history.csv")

        n = len(self.episode_rewards)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode", "reward", "steps", "seconds", "outcome",
                "intrusion_steps", "epsilon", "eval_score",
            ])
            for i in range(n):
                writer.writerow([
                    i + 1,
                    f"{self.episode_rewards[i]:.4f}",
                    int(self.episode_timesteps[i]),
                    f"{self.episode_seconds[i]:.2f}" if i < len(self.episode_seconds) else "",
                    self.episode_outcomes[i] if i < len(self.episode_outcomes) else "",
                    self.episode_intrusions[i] if i < len(self.episode_intrusions) else 0,
                    f"{self.training_strategy.epsilon:.4f}" if hasattr(self, 'training_strategy') else "",
                    f"{self.evaluation_scores[i]:.4f}" if i < len(self.evaluation_scores) else "",
                ])
        print(f"[INFO] Training CSV -> {path}")
        return path

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save_model(self, path: str = None):
        """Save the online model weights."""
        if path is None:
            os.makedirs(self.cfg.paths.model_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(
                self.cfg.paths.model_dir, f"DuelingDDQN_{timestamp}.pth"
            )
        torch.save(self.online_model.state_dict(), path)
        print(f"[INFO] Model saved -> {path}")
        return path

    def _save_best_model(self):
        """Save the current best model."""
        os.makedirs(self.cfg.paths.model_dir, exist_ok=True)
        path = os.path.join(self.cfg.paths.model_dir, "best_model.pth")
        torch.save(self.online_model.state_dict(), path)

    def load_model(self, path: str, nS: int, nA: int):
        """Load model weights from a file."""
        self.online_model = self._build_model(nS, nA)
        self.online_model.load_state_dict(
            torch.load(path, map_location=self.online_model.device, weights_only=True)
        )
        self.online_model.eval()
        self.evaluation_strategy = GreedyStrategy()
        print(f"[INFO] Model loaded <- {path}")

    def save_checkpoint(self, episode_idx, model):
        path = os.path.join(self.checkpoint_dir, f"ckpt_{episode_idx:05d}.pth")
        torch.save(model.state_dict(), path)

    def _cleanup_checkpoints(self, keep=5):
        """Keep only the last N checkpoints."""
        paths = sorted(glob.glob(os.path.join(self.checkpoint_dir, "ckpt_*.pth")))
        for p in paths[:-keep]:
            os.unlink(p)
