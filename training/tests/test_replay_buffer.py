"""Unit tests for the replay buffer."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from training.buffers.replay_buffer import ReplayBuffer


class TestReplayBuffer:
    def test_store_and_len(self):
        buf = ReplayBuffer(max_size=100, batch_size=4)
        assert len(buf) == 0

        state = np.array([1.0, 2.0, 3.0])
        buf.store((state, 0, 1.0, state, 0.0))
        assert len(buf) == 1

    def test_circular_overwrite(self):
        buf = ReplayBuffer(max_size=5, batch_size=2)
        for i in range(10):
            s = np.array([float(i)])
            buf.store((s, 0, 0.0, s, 0.0))
        assert len(buf) == 5

    def test_sample_shape(self):
        buf = ReplayBuffer(max_size=100, batch_size=8)
        obs_dim = 5
        for i in range(20):
            s = np.random.randn(obs_dim)
            ns = np.random.randn(obs_dim)
            buf.store((s, i % 3, float(i), ns, 0.0))

        states, actions, rewards, next_states, terminals = buf.sample()
        assert states.shape == (8, obs_dim)
        assert actions.shape == (8, 1)
        assert rewards.shape == (8, 1)
        assert next_states.shape == (8, obs_dim)
        assert terminals.shape == (8, 1)

    def test_sample_order_matches_store_order(self):
        """Critical test: verify the tuple order is (s, a, r, ns, d)."""
        buf = ReplayBuffer(max_size=10, batch_size=1)
        s = np.array([1.0, 2.0])
        a = 7
        r = 42.0
        ns = np.array([3.0, 4.0])
        d = 1.0
        buf.store((s, a, r, ns, d))

        states, actions, rewards, next_states, terminals = buf.sample(batch_size=1)
        np.testing.assert_array_almost_equal(states[0], s)
        assert actions[0, 0] == a
        assert rewards[0, 0] == r
        np.testing.assert_array_almost_equal(next_states[0], ns)
        assert terminals[0, 0] == d


class TestExperienceTupleOrder:
    """Regression test for the critical bug where rewards and next_states were swapped."""

    def test_network_load_experiences_order(self):
        """Verify FCDuelingQ.load_experiences preserves the correct order."""
        from training.networks.fc_dueling_q import FCDuelingQ

        model = FCDuelingQ(input_dim=4, output_dim=2, hidden_dims=(8,), device="cpu")

        # Create a known experience tuple
        states = np.array([[1.0, 2.0, 3.0, 4.0]])
        actions = np.array([[1]])
        rewards = np.array([[99.0]])  # Distinctive value
        next_states = np.array([[5.0, 6.0, 7.0, 8.0]])  # Distinctive value
        terminals = np.array([[0.0]])

        loaded = model.load_experiences((states, actions, rewards, next_states, terminals))
        l_states, l_actions, l_rewards, l_next_states, l_terminals = loaded

        # Rewards should still be 99.0, NOT [5, 6, 7, 8]
        assert l_rewards.item() == 99.0, (
            f"Rewards got value {l_rewards.item()}, expected 99.0. "
            "This indicates the experience tuple order is WRONG."
        )
        # Next states should be [5, 6, 7, 8], NOT 99.0
        np.testing.assert_array_almost_equal(
            l_next_states.cpu().numpy()[0], [5.0, 6.0, 7.0, 8.0]
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
