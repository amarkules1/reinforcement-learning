from gymnasium.envs.registration import register

# cribbed from https://github.com/monokim/framework_tutorial
register(
    id='Pygame-v0',
    entry_point='gym_games.envs:CustomEnv',
    max_episode_steps=2000,
)
