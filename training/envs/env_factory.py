"""Environment factory: creates the CollisionAvoidanceAviary from config.

Replaces the old get_make_env_fn() which had hardcoded defaults scattered
across multiple files.
"""

from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType
from training.envs.collision_avoidance_env import CollisionAvoidanceAviary
from training.configs.config import _namespace_to_dict


def make_env(cfg, gui=False, seed=None):
    """Create a CollisionAvoidanceAviary from a Config object.

    Parameters
    ----------
    cfg : Config
        Full training config (needs cfg.env and cfg.reward sections).
    gui : bool
        Whether to open the PyBullet GUI.
    seed : int, optional
        Random seed for the environment reset.

    Returns
    -------
    CollisionAvoidanceAviary
    """
    # Extract sensor and action config dicts (if present in YAML)
    sensor_cfg = _namespace_to_dict(cfg.sensor) if hasattr(cfg, 'sensor') else None
    action_cfg = _namespace_to_dict(cfg.actions) if hasattr(cfg, 'actions') else None
    mission_cfg = _namespace_to_dict(cfg.mission) if hasattr(cfg, 'mission') else None

    env = CollisionAvoidanceAviary(
        cfg=cfg,
        drone_model=DroneModel(cfg.env.drone_model),
        num_drones=cfg.env.num_drones,
        physics=Physics(cfg.env.physics),
        pyb_freq=cfg.env.sim_freq_hz,
        ctrl_freq=cfg.env.ctrl_freq_hz,
        gui=gui,
        record=False,
        obs=ObservationType("kin"),
        act=ActionType("autorouting"),
        sensor_cfg=sensor_cfg,
        action_cfg=action_cfg,
        skip_drone_raycasting=getattr(cfg.env, 'skip_drone_raycasting', False),
        obs_choice=getattr(cfg.env, 'obs_choice', 'sensor'),
        episode_len_sec=getattr(cfg.env, 'episode_len_sec', 30),
        mission_cfg=mission_cfg,
    )

    if seed is not None:
        env.reset(seed=seed)

    return env


def get_make_env_fn(cfg):
    """Return a factory function compatible with the agent's train() interface.

    Returns
    -------
    make_env_fn : callable
        Signature: make_env_fn(seed=None, gui=False, **kw) -> env
    make_env_kargs : dict
        Empty dict (all config is captured in the closure).
    """

    def _make_env(seed=None, gui=False, **kw):
        return make_env(cfg, gui=gui, seed=seed)

    return _make_env, {}
