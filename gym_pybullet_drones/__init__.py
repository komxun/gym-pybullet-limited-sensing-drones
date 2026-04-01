from gymnasium.envs.registration import register

register(
    id='autorouting-sa-aviary-v0',
    entry_point='gym_pybullet_drones.envs:AutoroutingSARLAviary',
)
