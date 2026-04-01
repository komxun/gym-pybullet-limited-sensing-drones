"""Replay buffer with pre-allocated contiguous numpy arrays.

Performance note: the old implementation used dtype=np.ndarray (object arrays)
which made np.vstack on every sample() extremely slow.  This version
pre-allocates flat float32/int64 arrays so sample() is just fancy indexing.
"""

import numpy as np


class ReplayBuffer:
    """Fixed-size circular replay buffer for off-policy DRL.

    Stores transitions as (state, action, reward, next_state, is_terminal).
    Arrays are allocated lazily on the first store() call (once state_dim is known).
    """

    def __init__(self, max_size: int = 200000, batch_size: int = 64):
        self.max_size = max_size
        self.batch_size = batch_size
        self._idx = 0
        self.size = 0
        self._initialised = False

        # Will be allocated on first store()
        self._ss = None   # states       (max_size, state_dim) float32
        self._ns = None   # next_states  (max_size, state_dim) float32
        self._as = None   # actions      (max_size, 1)         int64
        self._rs = None   # rewards      (max_size, 1)         float32
        self._ds = None   # is_terminal  (max_size, 1)         float32

    def _lazy_init(self, state):
        """Allocate contiguous arrays once state_dim is known."""
        state_dim = np.asarray(state).shape[0]
        self._ss = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self._ns = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self._as = np.zeros((self.max_size, 1), dtype=np.int64)
        self._rs = np.zeros((self.max_size, 1), dtype=np.float32)
        self._ds = np.zeros((self.max_size, 1), dtype=np.float32)
        self._initialised = True

    def store(self, transition: tuple):
        """Store a single (s, a, r, s', done) transition."""
        s, a, r, ns, d = transition
        if not self._initialised:
            self._lazy_init(s)

        self._ss[self._idx] = s
        self._as[self._idx, 0] = a
        self._rs[self._idx, 0] = r
        self._ns[self._idx] = ns
        self._ds[self._idx, 0] = d

        self._idx = (self._idx + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int = None) -> tuple:
        """Sample a random batch.

        Returns
        -------
        tuple of np.ndarray
            (states, actions, rewards, next_states, is_terminals)
            Each is a 2D array of shape (batch_size, feature_dim).

        NOTE: The order here MUST match what the network's load_experiences()
        expects. This was a critical bug in the old code.
        """
        if batch_size is None:
            batch_size = self.batch_size

        idxs = np.random.randint(0, self.size, size=batch_size)

        return (
            self._ss[idxs],
            self._as[idxs],
            self._rs[idxs],
            self._ns[idxs],
            self._ds[idxs],
        )

    def __len__(self):
        return self.size
