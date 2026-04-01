# DRL Drone Collision Avoidance — Clean Training Framework

## Project Structure

```
training/
├── configs/
│   ├── config.py          # YAML config loader with CLI overrides
│   └── default.yaml       # Default hyperparameters (edit this!)
├── agents/
│   └── dueling_ddqn.py    # Dueling DDQN agent with TensorBoard logging
├── networks/
│   └── fc_dueling_q.py    # Dueling Q-network (FIXED experience tuple order)
├── buffers/
│   └── replay_buffer.py   # Circular replay buffer
├── strategies/
│   └── exploration.py     # Epsilon-greedy (exp/linear), softmax, greedy
├── envs/
│   ├── collision_avoidance_env.py  # Fixed environment wrapper
│   └── env_factory.py     # Environment factory from config
├── tests/
│   ├── test_replay_buffer.py  # Buffer + experience order regression tests
│   └── test_config.py     # Config system tests
├── train.py               # Main training entry point
├── evaluate.py            # Evaluation / visualization script
└── README.md              # This file
```

## Quick Start

### Train
```bash
# From the project root directory:
python -m training.train

# With custom config:
python -m training.train --config training/configs/default.yaml

# Override parameters from CLI:
python -m training.train --agent.gamma 0.95 --training.max_episodes 5000 --exploration.min_epsilon 0.05
```

### Evaluate
```bash
# Headless evaluation (metrics only):
python -m training.evaluate --model trained_models/best_model.pth --episodes 50

# With PyBullet GUI:
python -m training.evaluate --model trained_models/best_model.pth --gui --realtime
```

### Monitor Training
```bash
tensorboard --logdir tb_logs
```

### Run Tests
```bash
python -m pytest training/tests/ -v
```

## Critical Bugs Fixed

### 1. Experience Tuple Order (CRITICAL)
**Old code** (`FCDuelingQ.load()`): unpacked as `(states, actions, new_states, rewards, terminals)`
**Replay buffer** returned: `(states, actions, rewards, new_states, terminals)`

**Result**: Rewards and next_states were SWAPPED during training. The TD-target
computation used reward values as next-state features and vice versa. This alone
could explain why training never converged.

**Fix**: `load_experiences()` now unpacks in the correct order matching the buffer.

### 2. Terminated vs Truncated Conflation (HIGH)
**Old code**: Timeout was handled in `_computeTerminated()`, and `_computeTruncated()`
always returned `False`. The agent stored `is_failure = terminated and not truncated`,
which was always `terminated` (since truncated was always False).

**Result**: Timeout episodes were treated as failures in the replay buffer, zeroing
out the value estimate. The agent learned that running out of time = crashing.

**Fix**: Timeout moved to `_computeTruncated()`. The agent now stores
`is_failure = terminated and not truncated` correctly.

### 3. Missing Observation Normalization (HIGH)
**Old code**: `_computeObs()` used raw `state[0:3], state[7:10], ...` without
calling `_clipAndNormalizeState()`. Observation space declared bounds of `[-1, 1]`
but actual values could be much larger.

**Fix**: `_computeObs()` now calls `_clipAndNormalizeState()` before building the
observation vector.

### 4. Reward Function Logic (MEDIUM)
**Old code** (reward_choice=3): When safety penalty fired, it REPLACED the progress
reward entirely (`ret = safety_penalty`). The agent got no progress signal near obstacles.

**Fix**: Safety penalty is now ADDED to progress reward (configurable via
`reward.additive_safety` in the YAML config).

### 5. Hyperparameter Improvements
| Parameter | Old | New | Rationale |
|-----------|-----|-----|-----------|
| gamma | 0.9 | 0.99 | Longer horizon for navigation tasks |
| min_epsilon | 0.3 | 0.05 | Old value kept 30% random actions forever |
| tau | 0.1 | 0.005 | Softer target updates for stability |
| optimizer | RMSprop | Adam | Generally better for DRL |
| lr | 0.0005 | 0.0003 | Slightly lower for stability |
| buffer size | 100k | 200k | More diverse experience |
| warmup batches | 5 | 50 | Don't learn from near-empty buffer |
| max_episodes | 500 | 2000 | Complex task needs more training |
| time_penalty | 0 | -0.01 | Encourage efficiency |

## Configuration

All parameters are in `training/configs/default.yaml`. Key sections:

- **env**: drone model, num_drones, frequencies, obs_choice
- **reward**: reach threshold, penalties, progress weight, additive vs replacement safety
- **agent**: gamma, tau, target update frequency, warmup
- **network**: hidden layer dimensions
- **optimizer**: type (adam/rmsprop), learning rate
- **buffer**: max size, batch size
- **exploration**: strategy type, epsilon schedule
- **training**: max episodes, seeds, eval frequency, logging frequency
- **paths**: where to save models, logs, results
