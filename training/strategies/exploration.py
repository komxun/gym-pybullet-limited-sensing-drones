"""Exploration strategies for value-based DRL agents."""

import torch
import numpy as np


class GreedyStrategy:
    """Pure greedy action selection (for evaluation)."""

    def __init__(self):
        self.exploratory_action_taken = False

    def select_action(self, model, state):
        with torch.no_grad():
            q_values = model(state)
            return q_values.argmax(dim=-1).item()


class EGreedyExpStrategy:
    """Epsilon-greedy with exponential decay schedule."""

    def __init__(self, init_epsilon=1.0, min_epsilon=0.05, decay_steps=50000):
        self.init_epsilon = init_epsilon
        self.min_epsilon = min_epsilon
        self.decay_steps = decay_steps

        # Pre-compute exponential decay schedule
        self.epsilons = (
            0.01 / np.logspace(-2, 0, decay_steps, endpoint=False) - 0.01
        )
        self.epsilons = self.epsilons * (init_epsilon - min_epsilon) + min_epsilon

        self.epsilon = init_epsilon
        self.t = 0
        self.exploratory_action_taken = False

    def _update_epsilon(self):
        self.epsilon = (
            self.min_epsilon if self.t >= self.decay_steps else self.epsilons[self.t]
        )
        self.t += 1

    def select_action(self, model, state, n_actions=3):
        self.exploratory_action_taken = False
        if np.random.rand() <= self.epsilon:
            # Random action — skip the forward pass entirely
            action = np.random.randint(n_actions)
            self.exploratory_action_taken = True
        else:
            with torch.no_grad():
                action = model(state).argmax(dim=-1).item()

        self._update_epsilon()
        return action


class EGreedyLinearStrategy:
    """Epsilon-greedy with linear decay schedule."""

    def __init__(self, init_epsilon=1.0, min_epsilon=0.1, decay_steps=20000):
        self.t = 0
        self.epsilon = init_epsilon
        self.init_epsilon = init_epsilon
        self.min_epsilon = min_epsilon
        self.decay_steps = decay_steps
        self.exploratory_action_taken = False

    def _update_epsilon(self):
        frac = 1 - self.t / self.decay_steps
        self.epsilon = (self.init_epsilon - self.min_epsilon) * frac + self.min_epsilon
        self.epsilon = np.clip(self.epsilon, self.min_epsilon, self.init_epsilon)
        self.t += 1

    def select_action(self, model, state, n_actions=3):
        self.exploratory_action_taken = False
        if np.random.rand() <= self.epsilon:
            action = np.random.randint(n_actions)
            self.exploratory_action_taken = True
        else:
            with torch.no_grad():
                action = model(state).argmax(dim=-1).item()

        self._update_epsilon()
        return action


class SoftMaxStrategy:
    """Boltzmann (softmax) exploration with temperature annealing."""

    def __init__(
        self, init_temp=1.0, min_temp=0.3, exploration_ratio=0.8, max_steps=25000
    ):
        self.t = 0
        self.init_temp = init_temp
        self.min_temp = min_temp
        self.exploration_ratio = exploration_ratio
        self.max_steps = max_steps
        self.exploratory_action_taken = False

    def _update_temp(self):
        frac = 1 - self.t / (self.max_steps * self.exploration_ratio)
        temp = (self.init_temp - self.min_temp) * frac + self.min_temp
        temp = np.clip(temp, self.min_temp, self.init_temp)
        self.t += 1
        return temp

    def select_action(self, model, state, n_actions=3):
        self.exploratory_action_taken = False
        temp = self._update_temp()

        with torch.no_grad():
            q_values = model(state).squeeze(0)
            scaled = (q_values / temp)
            probs = torch.softmax(scaled, dim=-1).cpu().numpy()

        action = int(np.random.choice(len(probs), p=probs))
        greedy = int(q_values.argmax().item())
        self.exploratory_action_taken = action != greedy
        return action
