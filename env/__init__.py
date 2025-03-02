from gymnasium.envs.registration import register

register(
    id='BinPacking3D-v0',
    entry_point='env.bin_packing_env:BinPackingEnv',
    max_episode_steps=1000,
)