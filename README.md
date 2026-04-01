# Intelligent Safe Separation of Limited-sensing Uncrewed Aircraft Systems

This repository contains the simulation environment and deep reinforcement learning (DRL) framework accompanying the paper *"Intelligent Safe Separation of Limited-sensing Uncrewed Aircraft Systems"*.

The work addresses the challenge of maintaining safe separation between multiple uncrewed aircraft systems (UAS) that have limited onboard sensing capabilities. A Dueling Double Deep Q-Network (Dueling DDQN) agent learns speed-control policies that adjust each aircraft's velocity along its pre-planned route, enabling collision avoidance without requiring full airspace awareness. The simulation is built on a 3D physics engine (PyBullet) with Interfered Fluid Dynamical System (IFDS) routing and realistic multi-drone scenarios.

## Overview

- **Simulation environment** — A multi-drone PyBullet environment with IFDS-based route guidance, configurable ray-cast sensing, and contact detection.
- **DRL agent** — A Dueling DDQN agent that selects discrete speed commands (accelerate, decelerate, maintain) based on limited local sensor observations.
- **Comparative study** — Nine model configurations varying observation type (ray / sensor / sector) and sensor count (5 / 10 / 15) are trained and evaluated.

## Repository Structure

```
gym-pybullet-drones-routing/
├── gym_pybullet_drones/          # Simulation package
│   ├── assets/                   #   URDF drone and obstacle models
│   ├── control/                  #   PID flight controllers
│   ├── envs/                     #   Gymnasium environments
│   │   ├── BaseAviary.py         #     Core physics engine
│   │   ├── RoutingAviary.py      #     Multi-drone routing + scene
│   │   ├── SceneCreator.py       #     Obstacle scene generation
│   │   ├── ExtendedSARLAviary.py #     RL interface + IFDS routing
│   │   └── AutoroutingSARLAviary.py  # Observation / reward / termination
│   ├── routing/                  #   IFDS route guidance
│   └── utils/                    #   Enums, logging, helpers
├── training/                     # DRL training framework
│   ├── agents/                   #   Dueling DDQN agent
│   ├── buffers/                  #   Experience replay buffer
│   ├── configs/                  #   YAML configuration (default + 9 experiments)
│   ├── envs/                     #   Environment wrapper + factory
│   ├── networks/                 #   Dueling Q-network architecture
│   ├── strategies/               #   Exploration strategies
│   ├── tests/                    #   Unit tests
│   ├── train.py                  #   Single-model training script
│   ├── train_all.py              #   Batch training (all 9 models)
│   ├── evaluate.py               #   Single-model evaluation with GUI
│   ├── evaluate_models.py        #   Batch evaluation + metrics CSV
│   └── compare_models.py         #   Publication plots from results
├── experiments/                  # Experiment results
│   ├── {D,S,X}-{5,10,15}/       #   Per-model: trained weights, logs, results
│   └── plots/                    #   Comparative figures
├── tests/                        # Package-level tests
├── pyproject.toml                # Project metadata and dependencies
├── LICENSE
└── README.md
```

## Installation

```bash
git clone https://github.com/komxun/gym-pybullet-drones-routing.git
cd gym-pybullet-drones-routing/

conda create -n drones python=3.10
conda activate drones

pip install --upgrade pip
pip install -e .
```

**Requirements:** Python 3.10+, PyBullet, PyTorch, Gymnasium, NumPy, SciPy, Matplotlib, PyYAML, TensorBoard.

## Usage

### Train a model

```bash
# Train with default configuration
python -m training.train

# Train with a specific experiment config
python -m training.train --config training/configs/experiments/S-10.yaml

# Override parameters from the command line
python -m training.train --agent.gamma 0.95 --training.max_episodes 5000
```

### Train all 9 experiment models

```bash
python -m training.train_all --set D   # Ray observation: D-5, D-10, D-15
python -m training.train_all --set S   # Sensor observation: S-5, S-10, S-15
python -m training.train_all --set X   # Sector observation: X-5, X-10, X-15
```

### Evaluate a trained model

```bash
# Headless evaluation (metrics only)
python -m training.evaluate --model experiments/S-10/models/best_model.pth --episodes 50

# With PyBullet 3D visualisation
python -m training.evaluate --model experiments/S-10/models/best_model.pth --gui --realtime
```

### Generate comparative plots

```bash
python -m training.compare_models
```

### Monitor training

```bash
tensorboard --logdir tb_logs
```

## Configuration

All hyperparameters are defined in YAML files under `training/configs/`. The default configuration is in `training/configs/default.yaml`. Experiment-specific overrides (observation type, sensor count, output paths) are in `training/configs/experiments/`.

Key configurable sections: environment, sensor, actions, reward, agent, network, optimizer, replay buffer, exploration, training schedule, and mission geometry.

## Citation

If you use this code, please cite:

```bibtex
@article{tamanakijprasart2026intelligent,
    title   = {Intelligent Safe Separation of Limited-sensing Uncrewed Aircraft Systems},
    author  = {Tamanakijprasart, Komsun},
    year    = {2026}
}
```

## Acknowledgements

The simulation environment is built upon [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones) by Panerati et al. (IROS 2021).

## License

This project is released under the [MIT License](LICENSE).
