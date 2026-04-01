"""Dueling DQN network with correct experience loading."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FCDuelingQ(nn.Module):
    """Fully-connected Dueling Q-network.

    Architecture: shared trunk -> value stream V(s) + advantage stream A(s,a)
    Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple = (512, 128),
        activation_fc=F.relu,
        device: str = None,
    ):
        super().__init__()
        self.activation_fc = activation_fc

        # Shared trunk
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))

        # Dueling heads
        self.value_head = nn.Linear(hidden_dims[-1], 1)
        self.advantage_head = nn.Linear(hidden_dims[-1], output_dim)

        # Device setup
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.to(self.device)

    def _format(self, state):
        """Ensure input is a properly shaped tensor on the correct device."""
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            if x.dim() == 1:
                x = x.unsqueeze(0)
        return x

    def forward(self, state):
        x = self._format(state)
        x = self.activation_fc(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))

        value = self.value_head(x)
        advantage = self.advantage_head(x)
        # Q = V + (A - mean(A))
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q

    def load_experiences(self, experiences):
        """Convert numpy experience tuple to device tensors.

        CRITICAL FIX: The replay buffer returns
            (states, actions, rewards, next_states, is_terminals)
        This method must unpack in the SAME order.

        The old code had rewards and next_states swapped, which corrupted
        the TD-target computation and caused poor training.
        """
        states, actions, rewards, next_states, is_terminals = experiences
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        is_terminals = torch.from_numpy(is_terminals).float().to(self.device)
        return states, actions, rewards, next_states, is_terminals
